"""SwanLab 上报面（2026-10-01 扩充）。

这次改了两类东西，都容易「加了变量却没上报」或「上报了但没意义」：

1. **config 面板此前记的是 9 个 D1 已归档的结构 flag** —— 它们完全不参与建网
   （v21 恒为 `V21_CFG` 那个形状），而 config 面板是对比两次 run 时第一个看的
   东西。记虚构值比不记更糟。
2. **一批「算了但没上报」或「上报了但没有参照」的量**：
   - `grad_norm`：`clip_grad_norm_` 的返回值此前被**丢弃** —— fp16 溢出/梯度爆炸
     唯一的直接信号；
   - `l2_report` / `opt_loss`：此前 `loss` 不可分解；
   - `policy_ce_random` / `value_rmse_zero`：**基线**。没有基线的 loss 曲线分不清
     「学到了」和「从 5.89 降到 5.5」。

本文件钉住「该报的都在报」与「基线成对出现」，防止以后又出现新的「算了没报」。
"""
import ast
import os
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / 'scripts' / 'train_sft.py').read_text(encoding='utf-8')
TREE = ast.parse(SRC)
CODE = '\n'.join(ln for ln in SRC.splitlines() if not ln.lstrip().startswith('#'))


def _fn_src(name):
    fn = next(f for f in ast.walk(TREE)
              if isinstance(f, ast.FunctionDef) and f.name == name)
    return ast.unparse(fn)


MAIN = _fn_src('main')

def _has(hay, key):
    """ast.unparse 会把双引号归一成单引号，两种都认。"""
    return ('"%s"' % key) in hay or ("'%s'" % key) in hay

INIT = _fn_src('_init_swanlab')


# --------------------------------------------------------------------------- #
# 1. 算出来的东西必须真的被上报
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('key', [
    'grad_norm',            # clip_grad_norm_ 的返回值（曾被丢弃）
    'l2_report',            # loss 的第三项
    'opt_loss',             # 真正被 backward 的量
    't_data_max_ms',        # 取数长尾
    'speed_per_card',       # 每卡吞吐
    'eta_min',              # 剩余时间
    'elapsed_min',
])
def test_step_metric_is_reported(key):
    assert _has(MAIN, key), f'{key} 算/定义了但没上报到 swanlab'


@pytest.mark.parametrize('key', [
    'train_top1', 'train_top5', 'value_rmse',
    'policy_ce_random', 'value_rmse_zero', 'policy_entropy',
])
def test_health_metric_is_reported(key):
    assert _has(MAIN, key), f'健康度 {key} 未上报'


@pytest.mark.parametrize('key', [
    'eval_used_ema', 'eval_lr', 'eval_gap_to_best',
])
def test_eval_metric_is_reported(key):
    assert _has(MAIN, key), f'eval 侧 {key} 未上报'


def test_every_loss_curve_has_a_baseline():
    """policy_loss 与 value_loss 各要有一条「什么都没学到」的基线。

    这是这次扩充的核心动机：没有基线时「从 5.89 降到 5.5」和「真的学到了」
    在图上完全一样。所以断言基线与被基线化的量**成对存在**。
    """
    for curve, baseline in (('policy_loss', 'policy_ce_random'),
                            ('value_rmse', 'value_rmse_zero')):
        assert curve in MAIN, f'{curve} 未上报'
        assert baseline in MAIN, \
            f'{curve} 没有基线 {baseline} —— 曲线不可读（分不清学没学到）'


def test_grad_norm_is_taken_after_unscale():
    """`grad_norm` 必须在 `scaler.unscale_()` **之后**取，否则是假值。

    unscale 之前梯度还乘着 loss scale（默认 1024），量出来的范数大三个数量级 ——
    曲线会稳定地「看起来很大」，正好掩盖它本该发现的溢出。
    """
    i_un = CODE.index('scaler.unscale_(optimizer)')
    i_gn = CODE.index('clip_grad_norm_')
    assert i_un < i_gn, 'grad_norm 必须在 unscale_ 之后取'


def test_grad_norm_carries_forward_across_micro_batches():
    """accum>1 时每个 optimizer step 只有一个 grad_norm，打点是每 micro-batch。

    所以必须 carry forward（沿用上一次的），否则曲线在非 optimizer step 的打点上
    是 nan / 旧值 / 跳变，accum 越大越难看。
    """
    assert '_grad_norm_last' in MAIN, '缺 grad_norm 的沿用变量'
    assert MAIN.count('_grad_norm_last') >= 3, \
        '应在「初始化 / 赋值 / 上报」三处出现（沿用语义）'


def test_health_metrics_only_computed_on_log_steps():
    """健康度算在**上报分支**里，不是每个 micro-batch。

    每步算会白付 ~5 次 D2H 同步；`--swanlab-every 10` 下打点频率只有 1/10，
    放错位置等于 10 倍浪费。
    """
    # 用原始源码（ast.unparse 不保序）：健康度块必须落在 `if _do_swanlab:` 之后
    i_branch = SRC.index('if _do_swanlab:')
    i_health = SRC.index('_health_last = {')
    assert i_branch < i_health, '健康度必须在 if _do_swanlab 分支内计算'
    # 且不得在训练主循环里每步都算：主循环里不应再出现 _health_last 的构造
    assert SRC.count('_health_last = {') == 1, \
        '健康度只在上报分支构造一次（每 micro-batch 算会白付 10 倍 D2H 同步）'


# --------------------------------------------------------------------------- #
# 2. config 面板：真结构，不是归档 flag
# --------------------------------------------------------------------------- #
def test_config_panel_has_no_archived_flags():
    """9 个 D1 已归档的结构 flag 不得再出现在 config 面板里。

    它们照旧被 argparse 接受（旧命令行不改），但**不参与建网** —— 记进 config
    会让「对比两次 run」这件最常用的事得到错误答案。
    """
    archived = ('backbone_channels', 'backbone_res_blocks', 'res_blocks',
                'convnext_blocks', 'attn_blocks', 'value_channels',
                'value_res_blocks', 'policy_channels', 'policy_layers')
    cfg = INIT[INIT.index('config={'):]
    leaked = [a for a in archived if _has(cfg, a)]
    assert not leaked, f'config 面板仍记归档 flag（对 v21 无效）：{leaked}'


@pytest.mark.parametrize('key', [
    'v21/in_channels', 'v21/channels', 'v21/blocks', 'v21/params_total',
    'grad_checkpoint', 'batch_size_per_card', 'grad_accum', 'world_size',
    'effective_batch', 'lr', 'weight_decay', 'clip_grad_max_norm',
    'label_smoothing', 'attention_dropout', 'ema_enabled', 'ema_decay',
    'amp_dtype', 'scaler_init_scale', 'attn_sdpa_force_math',
    'attn_query_chunk', 'effective_batch',
])
def test_config_panel_records_the_key(key):
    assert _has(INIT, key), f'config 面板缺 {key}'


def test_config_v21_facts_come_from_v21_cfg():
    """结构字段必须来自 `V21_CFG` 而不是 args（args 里那些是归档 flag）。"""
    cfg = INIT[INIT.index('config={'):]
    assert "V21_CFG['in_channels']" in cfg
    assert "V21_CFG['channels']" in cfg
    assert "V21_CFG['n_res']" in cfg and "V21_CFG['n_mamba']" in cfg


def test_effective_batch_includes_accumulation():
    """config 里的 `effective_batch` 必须含 grad_accum —— 漏乘就把 lr 口径搞错。"""
    expr = re.search(r'"effective_batch":\s*([^,\n]+)', SRC).group(1)
    assert 'batch_size' in expr and '_accum' in expr, \
        f'effective_batch 表达式缺累积因子：{expr}'


# --------------------------------------------------------------------------- #
# 3. run 级事实（config 给不出的那些）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('key', [
    'run/optimizer_steps_per_epoch', 'run/micro_batches_per_epoch',
    'run/total_optimizer_steps', 'run/warmup_steps', 'run/n_train',
    'run/n_eval', 'run/effective_batch',
])
def test_run_level_facts_are_logged_once(key):
    assert _has(MAIN, key), f'run 级 {key} 未上报'


def test_reported_values_are_floats_not_tensors():
    """上报值必须是 python float —— swanlab 收到 tensor 的行为不可依赖。"""
    # `float(...)` 出现在关键标量转换处
    for pat in ('float(l2_report)', 'float(opt_loss)', 'float(_gn)'):
        assert pat in SRC, f'缺少 {pat}'
    # `**_health_last` 里的值在构造时已 float()（逐项断言在下面）
    m = re.search(r"_health_last = \{(.*?)\n\s*\}", SRC, re.S)
    assert m, '找不到 _health_last 的构造'
    body = m.group(1)
    assert body.count('float(') >= 6, \
        f'_health_last 的 6 个指标都必须 float()，实得 {body.count("float(")} 处'