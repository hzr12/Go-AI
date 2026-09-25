"""FP16 溢出导致的训练步丢弃：可观测性与定位。

现场（4 卡 910A，torch_npu 2.1.0.post3 / CANN 8.0.RC1）：
  swanlab 的 scaler_scale 曲线在 step 500~1770 平稳于 16384，随后断崖跌至
  2048（16384→8192→4096→2048），stdout 伴随大量
  `Gradient overflow. Skipping step`。

一个曾被否掉的假设（此处留测试以防再次误改）：
  曾怀疑「inf 检查早于 clip_grad_norm_，导致范数 > 65504/scale ≈ 4.0 的梯度
  被整步丢弃」。**该假设不成立**——模型参数始终是 FP32（train_sft.py 只设了
  memory_format，从未 .half()），梯度同为 FP32，上限 3.4e38，缩放系数不可能
  把它推到 65504。所以那些 inf/nan 是前向/反向里真实的数值故障（最可疑是
  NPU 强制 math 注意力物化大 logits 时的 FP16 溢出），而非 loss scaling 伪影。
  故**不能**把 clip_grad_norm_ 挪到 unscale_ 之前来「修」它。
"""
import os
import re
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()
import scripts.train_sft as t  # noqa: E402


def _optimizer_step_block():
    """返回优化器步进块的**代码**（剥离整行注释）。

    必须剥离注释：本块的中文注释里会提到 clip_grad_norm_ / unscale_ 这些
    标识符（解释为何不调换顺序），不剥离会让位置比较读到注释里的名字。
    """
    m = re.search(
        r'if \(i \+ 1\) % _accum_steps == 0 or \(i \+ 1\) == n_batches:\n'
        r'.*?optimizer\.zero_grad\(set_to_none=True\)',
        SRC, re.S)
    assert m, '未找到优化器步进块'
    raw = m.group(0)
    return '\n'.join(ln for ln in raw.splitlines()
                     if not ln.lstrip().startswith('#'))


def test_clip_stays_after_unscale():
    """裁剪保持在 unscale_ 之后——挪到前面是错的（见模块 docstring）。

    参数是 FP32，缩放梯度不会在 65504 溢出，因此不存在「本可救回的假溢出」。
    保留原顺序，避免用一个不成立的机制去掩盖真实故障。
    """
    blk = _optimizer_step_block()
    i_unscale = blk.index('scaler.unscale_(')
    i_clip = blk.index('clip_grad_norm_')
    assert i_unscale < i_clip, \
        'unscale_ 必须在 clip_grad_norm_ 之前：参数为 FP32，提前裁剪解决不了真实溢出'
    assert 'max_norm=1.0 * _scale_now' not in blk, \
        '不应在缩放态裁剪（FP32 参数下该假设不成立）'
    assert 'max_norm=1.0' in blk, '应保持 max_norm=1.0'


def test_step_and_update_still_called():
    blk = _optimizer_step_block()
    assert 'scaler.step(optimizer)' in blk
    assert 'scaler.update()' in blk
    assert blk.index('clip_grad_norm_') < blk.index('scaler.step'), \
        '裁剪必须早于 scaler.step'


def test_skipped_steps_are_counted_and_reported():
    """被跳过的步数必须可观测，否则这是诊断盲区。"""
    blk = _optimizer_step_block()
    assert '_n_skipped += 1' in blk, '未统计跳过的步数'
    assert 'scaler.get_scale() < _scale_now' in blk, \
        '未用缩放值下降来判定本步被跳过'
    # 日志与 swanlab 都要能看到
    assert 'skip=%d' in SRC, '日志行缺少 skip 计数'
    assert '"skipped_steps": _n_skipped' in SRC, 'swanlab 未上报 skipped_steps'
    assert '"skip_rate_pct"' in SRC, 'swanlab 未上报 skip_rate_pct'
    assert '"scaler_scale": _scale' in SRC, 'swanlab 未上报 scaler_scale'


def test_overflow_source_is_localized():
    """溢出时必须指出是 value head 还是 backbone，否则无从下手。"""
    assert 'def _locate_overflow(' in SRC
    assert 'torch.isinf(p.grad)' in SRC
    assert 'torch.isnan(p.grad)' in SRC
    assert '--value-loss-weight' in SRC, '缺少对 value head 溢出的可操作提示'
    assert '参数组' in SRC


def test_overflow_warning_threshold_exists():
    assert '_OVERFLOW_WARN_SCALE' in SRC
    assert '首次跌破' in SRC, '缩放值跌破阈值时应只告警一次，避免刷屏'


def test_model_params_stay_fp32():
    """前提守卫：模型从未被转成 FP16。

    若将来有人为了省显存加 .half()，本文件里「缩放梯度不会在 65504 溢出」
    的推理就失效了，溢出诊断需重新评估。
    """
    import re as _re
    body = _re.sub(r'(?m)^\s*#.*$', '', SRC)
    assert not _re.search(r'model\s*\.\s*half\(\)', body), \
        '模型被转成 FP16：溢出机制需重新评估'
    assert not _re.search(r'model\s*\.\s*to\([^)]*float16', body), \
        '模型被转成 FP16：溢出机制需重新评估'


# --------------------------------------------------------------------------- #
# 行为验证
# --------------------------------------------------------------------------- #
def test_locate_overflow_identifies_value_head_by_lr():
    """按 LR 比值区分 value head 与 backbone/policy，不依赖参数组下标。"""
    import logging
    backbone = torch.nn.Parameter(torch.zeros(2))
    value = torch.nn.Parameter(torch.zeros(2))
    value.grad = torch.tensor([float('nan'), 1.0])
    opt = torch.optim.SGD([
        {'params': [backbone], 'lr': 0.00754},
        {'params': [value], 'lr': 0.00754 * 5.0},
    ], lr=0.00754)
    msgs = []

    class _L:
        def warning(self, fmt, *a):
            msgs.append(fmt % a if a else fmt)

    t._locate_overflow(opt, _L())
    text = ' '.join(msgs)
    assert 'value head' in text, f'未识别出 value head 溢出: {text}'
    assert 'value-loss-weight' in text, '缺少下调 value 相关参数的提示'


def test_locate_overflow_classification_is_index_independent():
    """打乱参数组顺序后仍应正确分类（这是按下标判断会踩的坑）。"""
    import logging
    b1 = torch.nn.Parameter(torch.zeros(2))
    v1 = torch.nn.Parameter(torch.zeros(2))
    v1.grad = torch.tensor([float('inf'), 1.0])
    # value 组故意放在下标 0
    opt = torch.optim.SGD([
        {'params': [v1], 'lr': 0.0377},
        {'params': [b1], 'lr': 0.00754},
    ], lr=0.0377)
    msgs = []

    class _L:
        def warning(self, fmt, *a):
            msgs.append(fmt % a if a else fmt)

    t._locate_overflow(opt, _L())
    text = ' '.join(msgs)
    assert 'value head' in text, f'下标为 0 的 value 组被误判: {text}'


def test_locate_overflow_handles_no_grads():
    """梯度全为 None 时不应抛异常。"""
    import logging
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([{'params': [p], 'lr': 0.01}], lr=0.01)
    msgs = []

    class _L:
        def warning(self, fmt, *a):
            msgs.append(fmt % a if a else fmt)

    t._locate_overflow(opt, _L())
    assert any('未在参数组中找到' in m for m in msgs), '应提示未找到 inf/nan'
