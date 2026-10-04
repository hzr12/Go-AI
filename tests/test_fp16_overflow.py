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


# --------------------------------------------------------------------------- #
# 跳过步的记账：两条被真日志打出来的错误（2026-10-04 云端 910A）
# --------------------------------------------------------------------------- #
def test_skip_rate_denominator_cannot_exceed_100():
    """🔴 「跳过占比」的分母不能是 `step`。

    真日志里打出了 `累计跳过 4 步（占比 133.33%）` —— 分母比分子还小。根因：
    `_n_skipped += 1` 在 `scaler.step()` 那一行，而 `step += 1` 在 47 行**之后**
    ⇒ 连续溢出时那一行的 `step` 还没自增。

    修法是引入 `_n_attempted`，与 `_n_skipped` 在**同一处**自增，所以分母恒 ≥ 分子。
    """
    assert '_n_attempted' in SRC, '缺少与 _n_skipped 同处自增的计数器'
    # 两处占比（stdout 告警 + SwanLab）都必须用 _n_attempted
    assert SRC.count('/ max(1, _n_attempted)') == 2, \
        '跳过占比的两处上报都要用 _n_attempted'
    assert '_n_skipped / max(1, step)' not in SRC, \
        '仍有地方拿 step 当分母（会算出 > 100%）'
    # 且 _n_attempted 必须在 _n_skipped **之前**自增（同一个 optimizer-step 块内）
    i_a = SRC.index('_n_attempted += 1')
    i_s = SRC.index('_n_skipped += 1')
    assert i_a < i_s, '_n_attempted 必须先自增（否则某次跳过分母可能反而更小）'
    # 本轮把「诊断用 clip 返回的总范数」插进来后，两处自增之间多了几行
    # ⇒ 断言改为「同一次 unscale 的紧邻前后」，这才是「同口径」真正要保证的。
    i_un = SRC.index('scaler.unscale_(optimizer)')
    assert i_a < i_un <= i_s, \
        '两者必须在同一次 scaler.unscale_ 的紧邻前后自增（同一个 step 块内）'


def test_overflow_diagnostics_are_rank0_only():
    """🔴 溢出诊断必须 `is_main` 门控。

    真日志（2 卡）里每条溢出消息都**打印两遍** —— 无条件 `logger.warning` +
    `_locate_overflow` 的后果。4 卡就是 4 份一模一样的文本，反而掩盖了
    「rank0 先炸」这个真正有用的信息。
    """
    i = SRC.index('if use_scaler and scaler.get_scale() < _scale_now:')
    blk = SRC[i:i + 1200]
    assert 'if is_main:' in blk, '溢出诊断未做 rank0 门控'
    # _locate_overflow 必须在门控之内
    assert blk.index('if is_main:') < blk.index('_locate_overflow('), \
        '_locate_overflow 跑在 is_main 门控之外'


# --------------------------------------------------------------------------- #
# 缩放值策略可配：不必每次都从 65536 猜下来，也不必在平衡点附近震荡
# --------------------------------------------------------------------------- #
def test_scaler_init_scale_and_growth_are_configurable():
    assert "'--scaler-init-scale'" in SRC
    assert "'--scaler-growth-interval'" in SRC
    assert '_scaler_kwargs' in SRC
    assert "init_scale" in SRC and "growth_interval" in SRC
    # 默认不得改变既有行为（0 = 交给 PyTorch 默认 65536 / 2000）
    m = re.search(r"'--scaler-init-scale',\s*type=float,\s*default=([\d.]+)", SRC)
    assert m and float(m.group(1)) == 0.0, 'init-scale 默认应为 0（沿用 PyTorch 默认）'
    m = re.search(r"'--scaler-growth-interval',\s*type=int,\s*default=(-?\d+)", SRC)
    assert m and int(m.group(1)) == 0, 'growth-interval 默认应为 0（沿用 PyTorch 默认）'


def test_npu_grad_scaler_forwards_kwargs():
    """torch.npu.amp.GradScaler 必须能收到 init_scale / growth_interval。"""
    assert 'def npu_grad_scaler(enabled: bool, **kwargs):' in SRC, \
        'npu_grad_scaler 需接受并转发关键字参数，否则 NPU 上的配置无效'
    assert 'torch.npu.amp.GradScaler(enabled=enabled, **kwargs)' in SRC


def test_scaler_kwargs_actually_applied():
    """配置非零时才传参，避免给不支持关键字的旧版后端传空 kwargs。"""
    blk = SRC[max(0, SRC.index('_scaler_kwargs = {}') - 200):
              SRC.index('npu_grad_scaler(enabled=use_scaler')]
    assert 'if args.scaler_init_scale and args.scaler_init_scale > 0:' in blk, \
        'init_scale 应有非零判断'
    assert 'if args.scaler_growth_interval and args.scaler_growth_interval > 0:' in blk, \
        'growth_interval 应有非零判断'


def test_resume_may_override_scale():
    """resume 会 load_state_dict 覆盖缩放值——这是预期行为，但需留有痕迹。

    若用户从 checkpoint 续训并同时给了 --scaler-init-scale，后者会被
    checkpoint 里的实际值盖掉。至少不应报错。
    """
    assert "scaler.load_state_dict(tstate['scaler'])" in SRC
    assert 'except (RuntimeError, KeyError):' in SRC, \
        'scaler 状态不兼容时应优雅降级（BF16->FP16 切换会不匹配）'


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


def test_clip_grad_norm_turns_inf_into_nan_and_hides_it():
    """🔴 **本轮修的正是这个 bug 的成因**，先把机制钉死。

    `clip_grad_norm_(max_norm=1.0)` 在 `total_norm = inf` 时算出
    `clip_coef = 0` 并 `grad.mul_(0)` ⇒ `inf × 0 = NaN`。而
    `total_norm = inf ⇒ clip_coef = 0 < 1` ⇒ 这个分支**一定会进**。

    ⇒ **clip 之后统计「inf 个数」结构上恒为 0**，看到的 nan 全是 clipper 造的。
    真机日志里那行 `有 0 个 inf / 211 个 nan 参数` 就是这么来的，
    它被读成「反向出了 NaN」，于是排查方向被带偏到 loss 与前向 ——
    而**前向与 loss 都有限**，真凶是**反向出了 inf**。
    """
    p_inf = torch.nn.Parameter(torch.zeros(2))
    p_ok = torch.nn.Parameter(torch.zeros(2))
    p_inf.grad = torch.tensor([float('inf'), 0.0])
    p_ok.grad = torch.tensor([1.0, 1.0])
    gn = torch.nn.utils.clip_grad_norm_([p_inf, p_ok], max_norm=1.0)
    assert not bool(torch.isfinite(torch.as_tensor(float(gn)))), \
        '总范数应当是 inf'
    # inf 已经变成 NaN —— 这就是「诊断排在 clip 之后就永远看不到 inf」的原因
    assert int(torch.isinf(p_inf.grad).sum()) == 0, \
        'clip 之后不该还有 inf（若这里有 inf，说明 clip_coef 没被算成 0）'
    assert int(torch.isnan(p_inf.grad).sum()) >= 1, \
        'inf 应被 clip 转成 NaN'
    assert int(torch.isnan(p_ok.grad).sum()) == 0, '有限梯度不该变 NaN'


def test_locate_overflow_names_the_module_that_actually_had_inf():
    """🔴 修复后的诊断必须指出**哪个模块**，而不是只给一个 LR 猜测的组名。

    真机日志里三组全被打成 `backbone/policy`（组名是按 LR 比值猜的，
    `--value-lr-mult` 一改就失效）。而 `inf × 0 = NaN` 是**原地**写在同一个
    张量上，所以「clip 后哪些张量是 NaN」= 「哪些张量原来是 inf」⇒ 归属仍然准确。
    """
    import torch.nn as _nn
    from scripts.train_sft import _locate_overflow as _lo

    class Tiny(_nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _nn.Linear(4, 4)
            self.value_head = _nn.Linear(4, 2)

    net = Tiny()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    net.backbone(torch.randn(2, 4)).sum().backward()
    net.value_head(torch.randn(2, 4)).sum().backward()
    net.value_head.weight.grad[0, 0] = float('inf')
    # 先 clip（复现真机顺序：unscale → clip → 诊断）
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

    msgs = []

    class _L:
        def warning(self, fmt, *a):
            msgs.append(fmt % a if a else fmt)

    _lo(opt, _L(), phase='clip 前有 inf', named_params=dict(net.named_parameters()))
    joined = '\n'.join(msgs)
    assert 'value_head' in joined, f'未点名到真正出 inf 的模块：{msgs}'
    assert '按模块点名' in joined, msgs


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
    assert any('没有 inf/nan' in m or '未在参数组中找到' in m
               for m in msgs), f'应提示未找到 inf/nan：{msgs}'
