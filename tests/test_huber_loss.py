"""D4（SFT 侧）：--policy-loss / --value-loss / --huber-beta 三参数 + 删 BCE 分支。

对应简报：.superpowers/sdd/2026-09-25-v21-roadmap/task-p4-5-brief.md §3 的
7 个规定用例，另加 2 个接线/语义锁（test_flags_reach_loss_calls、
test_policy_huber_semantics_pinned）。每个用例「能红」的靶子见
task-p4-5-report.md 的对照表。

P4.5-fix 追加（对应 task-p4-5-fix-brief.md）：
* `test_policy_reduction_is_per_sample`  —— 钉住 §1 修掉的 1/A 归约 bug
  （D4 首版对 B×A 求均值，等于白吃一个 1/361 的稀释）。
* `test_gradient_scale_not_collapsed`   —— 把 report `## Fix 增补` 的梯度实测
  变成回归守卫（B=8, A=361, beta=0.5, targets ±1）。**P4.5b 起降级为历史口径的
  存档守卫**（policy 默认已改回 ce）。
* `test_value_loss_weight_default_is_one_no_compensation` —— 钉住「删补偿」。
* `test_huber_beta_rejects_non_positive` —— argparse 校验（0 / 负值 / 非数）。
* `test_losses_under_bf16_autocast`     —— BF16 路径（既定 NPU 部署口径）。
* `test_huber_loss_api_truth_table`      —— huber_loss vs smooth_l1 的真实关系。
* 加固：`test_huber_matches_torch_reference` 里近乎自指的默认 beta 断言改成
  手写公式对拍；`test_value_target_range_documented` 的两条纯 docstring 子串
  断言换成行为断言（梯度方向 + 未被重映射）。

P4.5b 追加（对应 task-p4-5b-brief.md，用户裁决 2026-09-27）：
* `test_policy_loss_default_is_ce`      —— 默认由 huber 改回 ce（**替代**
  `test_default_args_are_huber`），并要求 help 文案写明「为什么默认不是它」。
* `test_l2_report_uses_decay_group_only` —— c‖θ‖² 只覆盖 weight_decay != 0 的组。
* `test_l2_report_scales_with_weight_decay` —— c 就是 --weight-decay，没被硬编码。
* `test_log_loss_identity`               —— §3.1 的核心守卫：opt_loss / log_loss
  是两个量，L2 项不进梯度（用「两个口径反向给出逐位相同的参数梯度」证明）。
* `test_no_new_cli_params`               —— D1：选项名集合零新增（防 --l2-coef）。
* `test_optimizer_param_groups_unchanged` —— 四组 weight_decay 仍是 {wd,0,wd,0}。
* `test_policy_value_gradient_ratio`     —— 生产 FCPolicyHead/FCValueHead 上
  policy:value 梯度比（**简报给的 5.4 复现不出来，见函数 docstring 与 report**）。
* `test_ce_policy_gradient_is_action_space_independent` —— CE 的 policy 梯度与
  动作空间 A 无关，而 Huber(p) 严格按 1/A（sum）/ 1/A²（mean）缩放。

原则：
* Huber 的正确性 oracle = torch 官方 `F.smooth_l1_loss(..., beta=...)`
  逐位对拍 + 一份手写公式的独立复核；**不 import 任何「旧实现」**。
* `ce` / `mse` 的期望值全部手算（手写 log_softmax 与平滑 CE 公式），
  旧口径的「逐位一致」只对照 D4 之前那条**调用表达式本身**（同参同调用）。
* argparse 用 AST 解析真实源码，并重放 add_argument 得到真 parser ——
  不执行 main()（零训练副作用）。
* 任何「loss 值」都不当作梯度证据 —— 归一方式会同时改 loss 值和梯度，
  只有梯度范数才能回答「policy 相对 value 强多少」。
"""
import argparse
import ast
import contextlib
import inspect
import io
import os
import sys
import textwrap

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.train_sft as t  # noqa: E402
from scripts.train_sft import (  # noqa: E402
    compute_policy_loss,
    compute_value_loss,
    huber_loss,
)

SRC_PATH = os.path.join(ROOT, 'scripts', 'train_sft.py')
SRC = open(SRC_PATH, encoding='utf-8').read()

# --------------------------------------------------------------------------- #
# 源码/argparse 提取工具
# --------------------------------------------------------------------------- #


def _module_tree():
    return ast.parse(SRC)


def _fn(name, tree=None):
    tree = tree or _module_tree()
    return next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _add_argument_kwargs(fn):
    """{旗名: {kwarg: AST node}} —— 来自 main() 里真实的 add_argument 调用。"""
    out = {}
    for call in ast.walk(fn):
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'add_argument' and call.args):
            continue
        a0 = call.args[0]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
            out[a0.value] = {kw.arg: kw.value for kw in call.keywords}
    return out


def _replay_parser():
    """重放 main() 里全部 add_argument 得到语义等价的真 parser（不执行 main）。"""
    ap = argparse.ArgumentParser()
    scope = dict(vars(t))
    scope['ap'] = ap
    tree = ast.parse(textwrap.dedent(inspect.getsource(t.main)))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            expr = ast.Expression(body=node)
            ast.copy_location(expr, node)
            exec(compile(expr, '<train_sft argparse>', 'eval'), scope)
    return ap


def _help_entry(ap, metavar):
    """从 `--help` 的 options 段里取出**某一个**选项的整条 help（换行已压平）。

    不能用「find 到下一个 `[--` 为止」那种切法：`format_help()` 的 usage 段里
    也出现同样的 `[--xxx ...]` 字样，混进去会把**别的选项**的文案算进来
    （本文件就踩过：`--policy-loss` 的断言读到了 `--value-loss` 的「默认 huber」）。
    这里改成按缩进解析 options 段：从匹配 `  -` 开头的选项行起，到下一个选项行止。
    """
    txt = ap.format_help()
    lines = txt[txt.rindex('\noptions:') + 1:].splitlines()
    out, on = [], False
    for ln in lines[1:]:
        if ln.startswith('  -') and not ln.startswith('   '):
            if on:
                break
            on = metavar in ln
            if on:
                out.append(ln)
            continue
        if on:
            out.append(ln)
    assert out, f'--help 的 options 段里找不到 {metavar}'
    return ' '.join(' '.join(out).split())

# --------------------------------------------------------------------------- #
# 1. Huber 实现正确性 oracle：与 F.smooth_l1_loss 逐位相同 + 手写公式复核
# --------------------------------------------------------------------------- #


def test_huber_matches_torch_reference():
    g = torch.Generator().manual_seed(20260925)
    cases = [
        (torch.randn(7, generator=g) * 4, torch.randn(7, generator=g), 0.5),
        (torch.randn(64, 17, generator=g) * 3, torch.randn(64, 17, generator=g), 0.5),
        (torch.randn(5, 9, 4, generator=g), torch.randn(5, 9, 4, generator=g), 0.25),
        (torch.randn(33, generator=g), torch.randn(33, generator=g), 1.0),
        (torch.randn(200, generator=g) * 10, torch.randn(200, generator=g) * 10, 2.0),
        # 确定性覆盖：二次段内 / 线性段外 / 分界点 |d|=beta
        (torch.tensor([-1.5, -0.5, -0.49, 0.0, 0.49, 0.5, 1.5]),
         torch.zeros(7), 0.5),
    ]
    max_diff = 0.0
    for pred, target, beta in cases:
        mine = huber_loss(pred, target, beta=beta)
        ref = F.smooth_l1_loss(pred, target, beta=beta)  # reduction 默认 mean
        assert torch.equal(mine, ref), \
            f'与官方参考不逐位相同（beta={beta}）—— 实现换了 API/换了 beta/换了归约？'
        # 独立手写公式复核（防「实现与 smooth_l1 同时被改坏」）；float64 算，
        # 免得把 float32 的舍入差当成公式不符
        manual = _manual_smooth_l1(pred.double(), target.double(), beta).mean()
        max_diff = max(max_diff, abs(float(mine) - float(manual)))
        assert float(mine) == pytest.approx(float(manual), rel=1e-5, abs=1e-6), \
            f'与手写 Huber 公式不符（beta={beta}）'
    assert max_diff < 1e-5, f'手写公式最大偏差 {max_diff} 超阈'

    # reduction 必须是 mean（多元素上 sum 会大 N 倍）
    p4 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t4 = torch.zeros(4)
    got = huber_loss(p4, t4, beta=0.5)
    linear = p4 - 0.25  # |d|>=beta → |d|-0.5*beta
    assert float(got) == pytest.approx(float(linear.mean()), rel=1e-6)
    assert not torch.isclose(got, linear.sum(), rtol=1e-3), '归约滑成了 sum'

    # 默认 beta 必须 = 0.5（与 --huber-beta 默认一致）。
    # ⚠ 不能写成 huber_loss(p,t) == huber_loss(p,t,beta=0.5)（P4.5-fix 更正）：
    # 那是**自指**——只钉住「默认参数等于字面量 0.5」，不验任何数学，
    # 把 beta 默认改成 0.3 也照样绿。改成与**手写公式**（beta=0.5）比。
    assert float(huber_loss(p4, t4)) == pytest.approx(
        float(_manual_smooth_l1(p4, t4, 0.5).mean()), rel=1e-6), \
        '默认 beta 不是 0.5（默认档与手写 Huber(delta=0.5) 公式不符）'

    # reduction='none' 是 policy 侧逐样本聚合用的逃生口：逐元素、不归约
    none = huber_loss(p4, t4, beta=0.5, reduction='none')
    assert none.shape == p4.shape, f'reduction="none" 应保持形状，实得 {none.shape}'
    assert torch.equal(none, _manual_smooth_l1(p4, t4, 0.5)), \
        'reduction="none" 与手写逐元素公式不符'
    assert float(none.mean()) == pytest.approx(float(got), rel=1e-6), \
        'reduction="none" 的 .mean() 应与 reduction="mean" 一致'


def _manual_smooth_l1(pred, target, beta):
    """手写 smooth L1 = 教科书 Huber(delta=beta)（逐元素，不归约）。

    独立于 torch：这是 P4.5-fix 换掉「自指默认 beta 断言」后的唯一 oracle。
    """
    d = pred - target
    return torch.where(d.abs() < beta, 0.5 * d * d / beta,
                       d.abs() - 0.5 * beta)

# --------------------------------------------------------------------------- #
# 2. 分段极限：二次段梯度 = d/beta，线性段梯度 = sign(d)/N（数值梯度验证）
# --------------------------------------------------------------------------- #


def test_huber_limits():
    beta = 0.5

    def loss1(x, y):
        return huber_loss(torch.tensor([x], dtype=torch.float64),
                          torch.tensor([y], dtype=torch.float64), beta=beta)

    def num_grad(x, y, h=1e-4):
        return (float(loss1(x + h, y)) - float(loss1(x - h, y))) / (2 * h)

    # 二次段 |d| < beta：梯度 = d/beta（小误差处是二次的、随误差线性趋 0）
    quad = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.3), (-0.6, -0.4), (0.45, -0.04)]
    for x, y in quad:
        d = x - y
        assert abs(d) < beta, f'用例前提失效 d={d}'
        assert num_grad(x, y) == pytest.approx(d / beta, abs=1e-6), \
            f'二次段梯度应为 d/beta={d / beta}，实得 {num_grad(x, y)}（x={x}, y={y}）'

    # 线性段 |d| > beta：梯度 = sign(d)/N。**下面是 N=1 的单元素张量**，
    # 所以分母恰好是 1、看起来像「±1」—— 这正是 P4.5 记录不实表述的根源：
    # 真实调用是 (B,A) 或 (B,) 的批量张量，mean 归约把每个元素的梯度也除以 N。
    # 批量下的真实值见 test_huber_gradient_is_divided_by_N。
    lin = [(2.0, 0.0, 1.0), (-3.0, 0.0, -1.0), (1.0, 0.4, 1.0),
           (-0.9, 0.05, -1.0), (5.0, -5.0, 1.0)]
    for x, y, sign in lin:
        d = x - y
        assert abs(d) > beta, f'用例前提失效 d={d}'
        assert num_grad(x, y) == pytest.approx(sign, abs=1e-6), \
            f'线性段梯度应为 {sign:+.0f}（N=1），实得 {num_grad(x, y)}（x={x}, y={y}）'

    # 损失值本身也钉在公式上
    assert float(loss1(0.1, 0.0)) == pytest.approx(0.5 * 0.01 / beta, rel=1e-12)
    assert float(loss1(2.0, 0.0)) == pytest.approx(2.0 - 0.5 * beta, rel=1e-12)


def test_huber_gradient_is_divided_by_N():
    """线性段梯度是 `sign(d)/N`，**不是**恒 ±1（P4.5-fix：docstring/help 的更正依据）。

    上一版 docstring 写「线性段梯度恒 ±1」，并拿它当 smooth_l1 vs
    `F.huber_loss` 的选型理由。实测三个 N（beta=0.5, d=1，线性段）：

        N=1 → ±1.000000    N=8 → ±0.125000    N=2888 → ±0.000346

    两者（smooth_l1 与 `F.huber_loss`）在批量下**都**除以 N，真实差异只是一个
    统一的 beta 因子 —— 见 test_huber_loss_api_truth_table。
    """
    beta = 0.5
    for n, want in ((1, 1.0), (8, 0.125), (2888, 0.000346)):
        d = torch.full((n,), 1.0, requires_grad=True)   # |d|=1 >= beta → 线性段
        huber_loss(d, torch.zeros(n), beta=beta).backward()
        assert float(d.grad[0]) == pytest.approx(want, rel=2e-3), \
            f'N={n} 线性段梯度应为 ±1/N={want:.6f}，实得 {float(d.grad[0]):.6f}'
    # smooth_l1 与 torch 的 F.huber_loss 只差一个统一的 beta 因子（不是 ±1 vs ±0.5）
    for beta in (0.25, 0.5, 2.0):
        d = torch.full((8,), 3.0, requires_grad=True)
        F.huber_loss(d, torch.zeros(8), delta=beta).backward()
        hl = float(d.grad[0])
        d2 = torch.full((8,), 3.0, requires_grad=True)
        F.smooth_l1_loss(d2, torch.zeros(8), beta=beta).backward()
        sl = float(d2.grad[0])
        assert hl == pytest.approx(beta * sl, rel=1e-6), \
            f'beta={beta}: huber_loss(delta)={hl:.6f} 应 = beta·smooth_l1={beta * sl:.6f}'

# --------------------------------------------------------------------------- #
# 3. 三个旗的真实 argparse 定义：默认 ce / huber / 0.5
# --------------------------------------------------------------------------- #


def test_policy_loss_default_is_ce():
    """P4.5b 用户裁决：policy 默认由 `huber` 改回 **`ce`**，value 仍是 `huber`。

    为什么默认不是 huber（这条断言的一半价值在**理由**上，所以下面同时查 help
    文案有没有把理由写下来 —— 只钉 `default='ce'` 的话，下一个人照样能改回去）：

    **定义在概率上的损失，梯度尺度必然依赖动作空间 A。** CE 打在 log-prob 上，
    `d(CE)/d(logit_j) = p_j − y_j`，每坐标有界、与 A 无关；Huber 打在概率上，
    链式法则要再乘一层 softmax 雅可比，专家坐标上带一个 `p ≈ 1/A`。于是
    `mean over A` 给 1/A²、`sum over A` 给 A，**没有任何归约能消掉它**。本仓支持
    9/13/19 路（A = 82/170/362）⇒ 同一学习率在 9 路与 19 路之间差 4.4 倍。
    实测见 `test_ce_policy_gradient_is_action_space_independent`（A=82 vs 362：
    CE 的 policy 梯度比 1.005，Huber 0.223 = 1/A）。

    `huber` 选项**保留**（实验可复现），实现一字未改。
    """
    kw = _add_argument_kwargs(_fn('main'))
    for flag in ('--policy-loss', '--value-loss', '--huber-beta'):
        assert flag in kw, f'main() 缺少 {flag}（D4 三参数之一）'

    assert ast.literal_eval(kw['--policy-loss']['default']) == 'ce', \
        '--policy-loss 默认必须是 ce（P4.5b 用户裁决）'
    assert ast.literal_eval(kw['--policy-loss']['choices']) == ['huber', 'ce']
    assert ast.literal_eval(kw['--value-loss']['default']) == 'huber', \
        'value 侧默认仍是 huber（用户未裁决改它）'
    assert ast.literal_eval(kw['--value-loss']['choices']) == ['huber', 'mse']
    assert ast.literal_eval(kw['--huber-beta']['default']) == 0.5
    tnode = kw['--huber-beta']['type']
    # P4.5-fix：`type` 从 `float` 换成 `_positive_beta`（模块级 argparse type），
    # 否则 beta<=0 会一路静默进训练（0 → 纯 L1；负值 → 反向时原生 RuntimeError）
    assert isinstance(tnode, ast.Name) and tnode.id == '_positive_beta', \
        f'--huber-beta 的 type 应是 _positive_beta（>0 校验），实得 {ast.dump(tnode)}'
    assert callable(t._positive_beta), 'train_sft._positive_beta 不存在/不可调用'

    # 真 parser（重放 add_argument，不执行 main）：默认解析 + 显式覆盖 + 非法值被拒
    ap = _replay_parser()
    ns = ap.parse_args(['--data', 'dummy.npz'])
    assert (ns.policy_loss, ns.value_loss, ns.huber_beta) == ('ce', 'huber', 0.5), \
        f'默认解析结果错误: {(ns.policy_loss, ns.value_loss, ns.huber_beta)}'
    ns2 = ap.parse_args(['--data', 'd', '--policy-loss', 'huber',
                         '--value-loss', 'mse', '--huber-beta', '0.1'])
    assert (ns2.policy_loss, ns2.value_loss, ns2.huber_beta) == ('huber', 'mse', 0.1)
    for bad in (['--policy-loss', 'bce'], ['--value-loss', 'bce'],
                ['--policy-loss', 'huber_v2']):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
            ap.parse_args(['--data', 'd'] + bad)
        assert 'invalid choice' in err.getvalue(), \
            f'非法值 {bad} 未被 argparse 拒绝: {err.getvalue()!r}'

    # help 文案必须记录「为什么默认不是 huber」，否则下一个人会再把它设成默认
    ap = _replay_parser()
    entry = _help_entry(ap, '--policy-loss {huber,ce}')
    assert '默认**不是** huber' in entry, \
        f'--policy-loss 的 help 没写明「默认不是 huber」: {entry}'
    for token in ('1/A', '1/A²', 'A=82/170/362', '4.4'):
        assert token in entry, \
            f'--policy-loss 的 help 缺 A 依赖论证的 {token!r}: {entry}'
    assert '默认 huber' not in entry, \
        f'--policy-loss 的 help 仍把 huber 说成默认: {entry}'
    # value 侧留 huber 的理由也要在文案里（单标量输出、无 softmax ⇒ 与 A 无关）
    ventry = _help_entry(ap, '--value-loss {huber,mse}')
    assert '无 softmax' in ventry, \
        f'--value-loss 的 help 没写明「无 softmax 所以尺度与 A 无关」: {ventry}'


def test_huber_beta_rejects_non_positive():
    """`--huber-beta` 必须在**解析期**拒掉 0 / 负值 / 非数（§3）。

    没有校验时的实测后果：`beta=0` 静默退化成纯 L1（拐点消失，梯度不再随误差
    收缩）；`beta<0` 不在解析期报错，而是训练跑到第一个 batch 的**反向**时抛
    `F.smooth_l1_loss` 的原生 RuntimeError —— 栈里全是训练循环，用户看不出是
    哪个旗写错了。
    """
    ap = _replay_parser()
    for bad in ('0', '-1', '-0.5', 'abc', '', 'nan', 'inf'):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
            ap.parse_args(['--data', 'd', '--huber-beta', bad])
        msg = err.getvalue()
        assert 'huber-beta' in msg and '> 0' in msg, \
            f'--huber-beta {bad!r} 被拒了但没给出「必须 > 0」的原因: {msg!r}'
    for good in ('0.5', '0.1', '2.0', '1e-3'):
        got = ap.parse_args(['--data', 'd', '--huber-beta', good]).huber_beta
        assert got == float(good), f'--huber-beta {good} 解析成 {got}'
        assert isinstance(got, float), '--huber-beta 解析结果必须是 float'


def test_value_loss_weight_default_is_one_no_compensation():
    """用户裁决（P4.5-fix §2）：删掉 BCE 时代为平衡 policy/value 梯度的 5.0 倍补偿。

    处置选择：**保留参数、默认改 1.0**（而不是彻底删参数）——全仓扫描发现确有
    地方显式传它（README.md 的训练命令示例 + tests/test_fp16_overflow.py 用字符串
    钉住 `--value-loss-weight` 这个可操作提示），删参数会让那些调用点直接报
    「unrecognized argument」。所有传入点列在 report `## Fix 增补` §2。
    """
    kw = _add_argument_kwargs(_fn('main'))
    assert '--value-loss-weight' in kw, '--value-loss-weight 被删了（需控制器裁决）'
    assert ast.literal_eval(kw['--value-loss-weight']['default']) == 1.0, \
        f'--value-loss-weight 默认应为 1.0（无补偿），实得 ' \
        f'{ast.literal_eval(kw["--value-loss-weight"]["default"])}'
    assert isinstance(kw['--value-loss-weight']['type'], ast.Name) \
        and kw['--value-loss-weight']['type'].id == 'float', \
        '--value-loss-weight 的 type 应仍是 float'
    assert _replay_parser().parse_args(['--data', 'd']).value_loss_weight == 1.0

    # 权重仍作用在 value 项上（只是默认值不再补偿）。
    # ⚠ P4.5b：组合式搬进了 compose_losses（为了让「L2 项不进梯度」能被**按行为**
    # 测，见 test_log_loss_identity），所以断言跟着搬：main() 必须把 args 传进去。
    assert 'opt_loss = policy_loss + value_loss_weight * value_loss' in \
        inspect.getsource(t.compose_losses), \
        'opt_loss 的组合式被改动（应仍是 policy_loss + w * value_loss）'
    main_src = inspect.getsource(t.main)
    assert 'args.value_loss_weight' in main_src, '--value-loss-weight 成了死参数'

    # help 文案不得再宣称默认 5.0（只看本选项自己的条目，不看下一项）
    flat = ' '.join(_replay_parser().format_help().split())
    j = flat.rfind('--value-loss-weight VALUE_LOSS_WEIGHT')
    assert j != -1, '该参数没出现在 --help 里'
    nxt = flat.find('[--', j)
    entry = flat[j:nxt if nxt != -1 else j + 400]
    assert '默认 5.0' not in entry and '默认值 5.0' not in entry, \
        f'--help 仍在把 5.0 说成默认值: {entry}'
    assert '1.0' in entry, f'--help 未写明默认值 1.0: {entry}'

    # ⚠ run.txt / shell/** 不是本任务的文件：只扫描、不修改。若哪天有人往里
    # 加了显式的 --value-loss-weight，这个断言会红 —— 那正是需要复核的信号。
    for rel in (['run.txt'] + [os.path.join('shell', f) for f in
                               sorted(os.listdir(os.path.join(ROOT, 'shell')))]):
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as fh:
            body = fh.read()
        assert 'value-loss-weight' not in body, \
            f'{rel} 显式传了 --value-loss-weight（report 已列为待复核传入点）'

# --------------------------------------------------------------------------- #
# 4. SFT 里不再有 BCE 分支（AST：挪了位置 ≠ 删干净）
# --------------------------------------------------------------------------- #


def test_sft_has_no_bce_branch():
    tree = _module_tree()
    forbidden = {'binary_cross_entropy', 'binary_cross_entropy_with_logits',
                 'bce_loss', 'BCEWithLogits', 'BCELoss'}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            hits.append(f'Name {node.id} @ line {node.lineno}')
        elif isinstance(node, ast.Attribute) and node.attr in forbidden:
            hits.append(f'Attribute {node.attr} @ line {node.lineno}')
    assert not hits, f'scripts/train_sft.py 仍引用 BCE 符号（只是挪了位置？）: {hits}'

    # 三个损失函数体内不得再出现 winrates 分支（删干净 vs 挪进 helper）
    for fname in ('huber_loss', 'compute_policy_loss', 'compute_value_loss'):
        names = {n.attr for n in ast.walk(_fn(fname, tree))
                 if isinstance(n, ast.Attribute)}
        assert 'winrates' not in names, f'{fname}() 里仍有 winrates 分支'
    # main() 的损失调用点同样不再读 winrates（AST 判属性访问，注释里提到
    # 这个词不算 —— 旧 if dataset.winrates 分支必须整段删除）
    main_winrates = [n.attr for n in ast.walk(_fn('main', tree))
                     if isinstance(n, ast.Attribute) and n.attr == 'winrates']
    assert not main_winrates, \
        f'main() 仍按 winrates 分叉损失口径（{len(main_winrates)} 处属性访问）'
    # 但损失本身不能被「删没了」：两个分派函数都必须真的被调用
    main_src = inspect.getsource(t.main)
    assert 'compute_policy_loss(' in main_src, 'main() 未调用 compute_policy_loss'
    assert 'compute_value_loss(' in main_src, 'main() 未调用 compute_value_loss'

# --------------------------------------------------------------------------- #
# 5. 日志键不变：loss / policy_loss / value_loss（有且仅有）
# --------------------------------------------------------------------------- #


def test_log_keys_unchanged():
    log_dicts = []
    for call in ast.walk(_fn('main')):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'log' and call.args
                and isinstance(call.args[0], ast.Dict)):
            keys = [k.value for k in call.args[0].keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if 'policy_loss' in keys:
                log_dicts.append(keys)
    assert len(log_dicts) == 1, \
        f'应恰好找到 1 个训练损失日志字典，实得 {len(log_dicts)} 个'
    keys = log_dicts[0]
    loss_keys = [k for k in keys if 'loss' in k]
    assert sorted(loss_keys) == ['loss', 'policy_loss', 'value_loss'], \
        f'损失键集合被改动（改名/新增/删除都会打红）: {sorted(loss_keys)}'
    for required in ('loss', 'policy_loss', 'value_loss'):
        assert required in keys, f'日志字典缺少 {required}'

    # stdout 打点行与单同步点 helper 一起钉住（下游解析的是这些字面量）
    assert 'loss=%.4f (p=%.4f v=%.4f)' in SRC, \
        'stdout 打点行 loss/p/v 格式变了'
    assert 'return loss.item(), policy_loss.item(), value_loss.item()' in SRC, \
        '_read_log_scalars 的三元组变了'

# --------------------------------------------------------------------------- #
# 6. ce / mse 两条路径数值与改前一致（期望值全部手算）
# --------------------------------------------------------------------------- #


def _manual_log_softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    z = x - m
    return z - torch.log(torch.exp(z).sum(dim=-1, keepdim=True))


def _manual_smoothed_ce(logits, moves, eps):
    """手算 label-smoothed CE：mean[ -(1-eps)·log p[expert] - eps·mean(log p) ]。"""
    logp = _manual_log_softmax(logits)
    nll = -logp.gather(1, moves.view(-1, 1)).squeeze(1)
    uniform = -logp.mean(dim=-1)
    return ((1.0 - eps) * nll + eps * uniform).mean()


def test_ce_and_mse_paths_unchanged():
    g = torch.Generator().manual_seed(11)

    # --- ce：手算期望值 + 与 D4 之前的原始调用逐位一致 ---
    logits = torch.randn(6, 13, generator=g) * 2.5
    moves = torch.tensor([0, 4, 12, 3, 3, 7])
    for eps in (0.0, 0.1, 0.2):
        got = compute_policy_loss(logits, moves, 'ce',
                                  label_smoothing=eps, huber_beta=0.5)
        want = _manual_smoothed_ce(logits, moves, eps)
        assert float(got) == pytest.approx(float(want), rel=1e-5, abs=1e-6), \
            f'ce 与手算期望不符 eps={eps}: got={float(got)} want={float(want)}'
        assert torch.equal(got, F.cross_entropy(logits, moves,
                                                 label_smoothing=eps)), \
            f'ce 分支与 D4 之前的调用不再逐位相同（eps={eps}）'
    # 分派防串：kind 写反（ce 走成 huber）必红
    ce = compute_policy_loss(logits, moves, 'ce',
                             label_smoothing=0.1, huber_beta=0.5)
    hb = compute_policy_loss(logits, moves, 'huber',
                             label_smoothing=0.1, huber_beta=0.5)
    assert not torch.isclose(ce, hb, rtol=1e-3, atol=1e-4), \
        'ce 与 huber 分派疑似串了'

    # --- mse：手算期望值 + 与 D4 之前 winrates 分支的原始调用逐位一致 ---
    pred = torch.tanh(torch.randn(6, 1, generator=g))
    target = torch.tanh(torch.randn(6, 1, generator=g))   # ∈ (-1,1)，winrates 口径
    got_m = compute_value_loss(pred, target, 'mse')
    want_m = ((pred.squeeze() - target.squeeze()) ** 2).mean()
    assert torch.equal(got_m, F.mse_loss(pred.squeeze(), target.squeeze())), \
        'mse 分支与 D4 之前的调用不再逐位相同'
    assert float(got_m) == pytest.approx(float(want_m), rel=1e-6, abs=1e-7), \
        f'mse 与手算期望不符: got={float(got_m)} want={float(want_m)}'

    # 分派防串（刻意造 |d|=2 > beta 的大误差，mse 与 huber 必不等）
    big_pred = torch.tensor([[1.0], [-1.0]])
    big_tgt = torch.tensor([[-1.0], [1.0]])
    m = float(compute_value_loss(big_pred, big_tgt, 'mse'))
    h = float(compute_value_loss(big_pred, big_tgt, 'huber', huber_beta=0.5))
    assert m == pytest.approx(4.0, rel=1e-6), f'mse 大误差值不对: {m}'
    assert h == pytest.approx(2.0 - 0.25, rel=1e-6), f'huber 线性段值不对: {h}'
    assert not torch.isclose(torch.tensor(m), torch.tensor(h), rtol=1e-3), \
        'mse 与 huber 分派疑似串了'

# --------------------------------------------------------------------------- #
# 7. 目标值域：value_t ∈ [-1,1] 原样回归（没有概率化 / 截断 / 取 log）
# --------------------------------------------------------------------------- #


def test_value_target_range_documented():
    """P4.5-fix 加固：前两条断言原来是纯 docstring 子串检查（`'[-1,1]' in doc`），
    任何**行为**改动下都恒真 —— 已被 review 点名为最弱用例之一。改成行为断言：
    「原样回归」真正要保证的是 (a) 越界目标不被压回、clamp 不到 [-1,1]、
    (b) 梯度方向指向**原始**目标（重映射会改方向或改量级）。
    """
    pred = torch.tensor([[0.3], [-0.2], [0.0]])
    target = torch.tensor([[-1.5], [1.25], [-1.0]])   # 越界目标：旧 BCE 分支喂不出这种值

    got_h = compute_value_loss(pred, target, 'huber', huber_beta=0.5)
    ref_h = F.smooth_l1_loss(pred.squeeze(), target.squeeze(),
                             beta=0.5, reduction='mean')
    assert torch.equal(got_h, ref_h), '目标被改写（不再是原样回归）'

    got_m = compute_value_loss(pred, target, 'mse')
    ref_m = F.mse_loss(pred.squeeze(), target.squeeze())
    assert torch.equal(got_m, ref_m), 'mse 路径的目标被改写'

    # 旧 BCE 的「概率化」目标 (v+1)/2*0.8+0.1 塞回来 → 必不等
    remap = (target + 1) / 2 * 0.8 + 0.1
    bce_style = F.smooth_l1_loss(pred.squeeze(), remap.squeeze(),
                                 beta=0.5, reduction='mean')
    assert not torch.isclose(got_h, bce_style, rtol=1e-3, atol=1e-5), \
        'value 目标疑似被重新概率化（旧 BCE 的目标变换回潮）'
    # 截断到 [-1,1] 同样是改目标 → 必不等
    clamped = target.clamp(-1.0, 1.0)
    cl = F.smooth_l1_loss(pred.squeeze(), clamped.squeeze(),
                          beta=0.5, reduction='mean')
    assert not torch.isclose(got_h, cl, rtol=1e-3, atol=1e-5), \
        'value 目标疑似被 clamp（原样回归被破坏）'

    # --- 行为断言 1：梯度方向指向**原始**目标（|d|>beta 的线性段） ---
    p = pred.clone().requires_grad_(True)
    compute_value_loss(p, target, 'huber', huber_beta=0.5).backward()
    want_dir = torch.sign(pred.squeeze() - target.squeeze())
    assert torch.equal(torch.sign(p.grad).flatten(), want_dir), \
        f'梯度方向不是指向原始 value_t：sign(g)={torch.sign(p.grad).flatten()} vs {want_dir}'
    # 线性段逐元素梯度恰为 sign(d)/N（N=B=3）—— 重映射/clamp 都会改掉它
    assert torch.allclose(p.grad.abs().flatten(),
                          torch.full((3,), 1.0 / 3.0)), \
        f'线性段梯度应恰为 ±1/B=±0.3333，实得 {p.grad.flatten().tolist()}'

    # --- 行为断言 2：同一批目标整体平移会改变损失（真·原样回归） ---
    shifted = compute_value_loss(pred, target + 3.0, 'huber', huber_beta=0.5)
    assert not torch.isclose(got_h, shifted, rtol=1e-3, atol=1e-5), \
        '目标平移 3.0 后损失不变 —— 目标被投影/量化了，不是原样回归'
    # 越界目标喂进去必须产生**更大**的损失（|d| 更大），而不是被压到 [-1,1]
    inside = compute_value_loss(pred, target.clamp(-1.0, 1.0),
                                'huber', huber_beta=0.5)
    assert float(got_h) > float(inside), \
        '越界目标的损失不大于 clamp 后的损失（clamp 回潮）'

# --------------------------------------------------------------------------- #
# 8. 接线锁：三个旗真的接到损失调用点上（不是死参数）
# --------------------------------------------------------------------------- #


def test_flags_reach_loss_calls():
    calls = [c for c in ast.walk(_fn('main')) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Name)]
    pc = [c for c in calls if c.func.id == 'compute_policy_loss']
    vc = [c for c in calls if c.func.id == 'compute_value_loss']
    assert len(pc) == 1 and len(vc) == 1, \
        f'main() 应各调用一次损失分派函数，实得 policy={len(pc)} value={len(vc)}'

    def kw_args(call):
        out = {}
        for kw in call.keywords:
            v = kw.value
            if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                    and v.value.id == 'args'):
                out[kw.arg] = v.attr
        return out

    pk, vk = kw_args(pc[0]), kw_args(vc[0])
    # kind 分派参数是第 3 个位置参数：args.policy_loss / args.value_loss
    for call, attr, where in ((pc[0], 'policy_loss', 'policy'),
                              (vc[0], 'value_loss', 'value')):
        assert len(call.args) >= 3 and isinstance(call.args[2], ast.Attribute) \
            and call.args[2].attr == attr, \
            f'{where} 损失的 kind 没接到 args.{attr}（旗是死的）'
    assert pk.get('label_smoothing') == 'label_smoothing', \
        'ce 路径没接到 --label-smoothing'
    assert pk.get('huber_beta') == 'huber_beta', 'policy 侧没接到 --huber-beta'
    assert vk.get('huber_beta') == 'huber_beta', 'value 侧没接到 --huber-beta'
    # value 目标直接是 value_t（原样回归，中间没有任何变换）
    assert isinstance(vc[0].args[1], ast.Name) and vc[0].args[1].id == 'value_t', \
        'value 损失的目标不是 value_t（中间插了变换？）'

# --------------------------------------------------------------------------- #
# 9. policy-huber 语义锁：Huber(softmax(logits), label-smoothed one-hot)
#    归约 = 类内 sum over A + batch mean（P4.5-fix 修，见 §10）
# --------------------------------------------------------------------------- #


def test_policy_huber_semantics_pinned():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(4, 11, generator=g) * 3
    moves = torch.tensor([0, 5, 10, 2])
    eps, beta = 0.1, 0.5
    A = logits.shape[-1]
    p = torch.softmax(logits, dim=-1)

    def manual_huber(pred, y):
        """手算：逐元素 smooth L1 → **类内 sum over A** → batch mean。"""
        return _manual_smooth_l1(pred, y, beta).sum(dim=-1).mean()

    got = compute_policy_loss(logits, moves, 'huber',
                              label_smoothing=eps, huber_beta=beta)
    y = torch.full((4, A), eps / A)
    y[torch.arange(4), moves] = 1.0 - eps + eps / A
    assert float(got) == pytest.approx(float(manual_huber(p, y)),
                                       rel=1e-5, abs=1e-7), \
        'policy-huber 与「Huber(softmax, 平滑 one-hot)，类内 sum + batch mean」的手算期望不符'

    # eps=0 → 纯 one-hot 目标，公式不变
    got0 = compute_policy_loss(logits, moves, 'huber',
                               label_smoothing=0.0, huber_beta=beta)
    y0 = torch.zeros(4, A)
    y0[torch.arange(4), moves] = 1.0
    assert float(got0) == pytest.approx(float(manual_huber(p, y0)),
                                        rel=1e-5, abs=1e-7)

    # --huber-beta 真的起作用（0.1 ≠ 0.5），policy 与 value 两侧都要生效
    got_beta = compute_policy_loss(logits, moves, 'huber',
                                   label_smoothing=eps, huber_beta=0.1)
    assert not torch.isclose(got, got_beta, rtol=1e-3), \
        'policy 侧 --huber-beta 没生效（beta 被写死？）'
    v_pred = torch.tensor([[0.9], [-0.9]])
    v_tgt = torch.tensor([[-1.0], [1.0]])
    vb05 = compute_value_loss(v_pred, v_tgt, 'huber', huber_beta=0.5)
    vb01 = compute_value_loss(v_pred, v_tgt, 'huber', huber_beta=0.1)
    assert not torch.isclose(vb05, vb01, rtol=1e-3), \
        'value 侧 --huber-beta 没生效（beta 被写死？）'

    # 分派防串：huber ≠ ce
    ce = compute_policy_loss(logits, moves, 'ce',
                             label_smoothing=eps, huber_beta=beta)
    assert not torch.isclose(got, ce, rtol=1e-3), 'policy 分派疑似串了'

# --------------------------------------------------------------------------- #
# 10.（P4.5-fix）policy 归约 = 逐样本（类内 sum over A + batch mean）
# --------------------------------------------------------------------------- #
# §1 的 Critical 根因：D4 首版对 B×A 求均值，而被替换的 F.cross_entropy 是
# 逐样本（类内已归一）再对 batch 求均值 —— 差一个 1/A 的稀释（A=361 → 361×），
# 是实现 bug 不是设计选择。


_GRAD_B, _GRAD_A, _GRAD_BETA, _GRAD_EPS = 8, 361, 0.5, 0.1
_GRAD_MOVES = torch.tensor([0, 45, 90, 135, 180, 225, 270, 315])
_GRAD_VT = torch.tensor([[1.], [-1.], [1.], [-1.], [1.], [-1.], [1.], [-1.]])


def _smooth_labeled_target(B, A, eps, moves):
    y = torch.full((B, A), eps / A)
    y.scatter_(1, moves.long().view(-1, 1), 1.0 - eps + eps / A)
    return y


def test_policy_reduction_is_per_sample():
    """policy-huber 必须是「类内 sum over A，再对 batch 求 mean」。

    三件事一起钉：
    1. 与手写的逐样本聚合逐位一致（不是 mean over B*A）；
    2. 正好比 mean over B*A 大 A 倍（1/A 稀释被修掉的那 361×）；
    3. 逐样本口径真的生效：把 batch 里第 0 个样本单独拎出来算，得到的
       逐样本值之和与整体一致（mean 归约会把样本间的差异抹掉）。
    """
    B, A, eps, beta = _GRAD_B, _GRAD_A, _GRAD_EPS, _GRAD_BETA
    # 近均匀初值（logits 全 0 → p = 1/A），与 report 的实测同条件
    logits = torch.zeros(B, A)
    got = compute_policy_loss(logits, _GRAD_MOVES, 'huber',
                              label_smoothing=eps, huber_beta=beta)
    p = F.softmax(logits, dim=-1)
    y = _smooth_labeled_target(B, A, eps, _GRAD_MOVES)
    h = _manual_smooth_l1(p, y, beta)

    want_sum_a = h.sum(dim=-1).mean()
    assert torch.equal(got, want_sum_a), \
        'policy-huber 不是「类内 sum over A + batch mean」口径'
    want_mean_all = h.mean()
    assert not torch.isclose(got, want_mean_all, rtol=1e-3), \
        'policy-huber 退回了 mean over B*A —— 1/A 稀释回来了'
    assert float(got) / float(want_mean_all) == pytest.approx(A, rel=1e-5), \
        f'两者之比应恰为 A={A}，实得 {float(got) / float(want_mean_all):.4f}'

    # 逐样本语义：单样本调用 == 批量调用里那一行（batch 维是纯 mean，不串扰）
    one = compute_policy_loss(logits[:1], _GRAD_MOVES[:1], 'huber',
                              label_smoothing=eps, huber_beta=beta)
    assert torch.equal(one, h[:1].sum()), '单样本结果 != 批量结果的第 0 行'
    per_sample = h.sum(dim=-1)
    assert torch.equal(got, per_sample.mean()), 'batch 维不是纯 mean'

    # 目标/pred 侧的口径不许被顺带改掉：值与手算一致
    assert float(got) == pytest.approx(0.649744, rel=1e-4), \
        f'近均匀初值下的 policy_loss 应 ≈ 0.6497，实得 {float(got):.6f}（见 report）'


def test_gradient_scale_not_collapsed():
    """把 report `## Fix 增补` §8 的梯度实测变成回归守卫（B=8, A=361, β=0.5, targets ±1）。

    量的不是 loss 值（归一方式会同时改 loss 值和梯度），而是**损失输入空间**的
    梯度范数：`||d(policy_loss)/d(policy_logits)||` 与
    `||d(value_loss)/d(value_pred)||`。

    三条断言：
    1. policy 梯度相对 D4 首版（mean over B*A + 5× 补偿）**不得**塌回去 ——
       首版实测 2.72e-06，修后 9.83e-04（+361×）。
    2. value:policy 梯度比必须落在一个显式的坏值区间里：修后 ≈ 360:1
       （精确均匀初值），比老口径 ce+bce 的 2.2:1 差两个数量级。
       **P4.5b 起 policy 默认改回 ce**，本段因此从「残余失衡的守卫」降级为
       「历史口径的存档守卫」：那三个数字是 report §8.1 表的根据，不许漂；
       当前默认的比值由 `test_policy_value_gradient_ratio` 钉。
    3. 老口径的 2.2:1 参照仍成立（证明 harness 没坏、参照系没漂）。
    """
    B, A, beta, eps = _GRAD_B, _GRAD_A, _GRAD_BETA, _GRAD_EPS
    logits0 = torch.zeros(B, A)
    v0 = torch.zeros(B, 1)
    bce_tgt = (_GRAD_VT + 1) / 2 * 0.8 + 0.1
    y = _smooth_labeled_target(B, A, eps, _GRAD_MOVES)

    def grads(pol_kind, w, val_kind='huber'):
        lp = logits0.clone().requires_grad_(True)
        vp = v0.clone().requires_grad_(True)
        if pol_kind == 'ce':
            p = F.cross_entropy(lp, _GRAD_MOVES, label_smoothing=eps)
        else:
            p = compute_policy_loss(lp, _GRAD_MOVES, 'huber',
                                    label_smoothing=eps, huber_beta=beta)
        if val_kind == 'bce':
            # D4 之前的口径：BCEWithLogits on (v+1)/2*0.8+0.1
            v = F.binary_cross_entropy_with_logits(vp.squeeze(), bce_tgt.squeeze())
        else:
            v = F.smooth_l1_loss(vp.squeeze(), _GRAD_VT.squeeze(), beta=beta)
        (p + w * v).backward()
        return float(p.detach()), float(lp.grad.norm()), float(vp.grad.norm())

    def broken_policy_grad():
        """D4 首版：mean over B*A（复刻被删掉的实现，用来算衰减倍数）。"""
        lp = logits0.clone().requires_grad_(True)
        h = _manual_smooth_l1(F.softmax(lp, dim=-1), y, beta)
        h.mean().backward()
        return float(lp.grad.norm())

    # 3. 老口径参照：ce + 5*bce 的 value:policy ≈ 2.2:1
    _, gp_old, gv_old5 = grads('ce', 5.0, val_kind='bce')
    assert gv_old5 / gp_old == pytest.approx(2.225, rel=2e-2), \
        f'老口径参照漂了：value:policy = {gv_old5 / gp_old:.4f}（应 ≈ 2.225）'
    assert gp_old == pytest.approx(0.317757, rel=5e-3), \
        f'ce 路径梯度不再逐位等于 D4 之前的量级：{gp_old:.6g}'

    # 1+2. P4.5 的交付口径（**当时**的默认：huber + huber，w=1.0）
    #    ⚠ P4.5b 起 `--policy-loss` 默认已改回 ce，本段从「残余失衡的守卫」变成
    #    「历史口径的存档」：数字不许漂（它们是 report §8.1 表的根据），但不再
    #    描述当前默认。当前默认的比值由 test_policy_value_gradient_ratio 钉。
    pl, gp, gv = grads('huber', 1.0)
    assert gp > 300 * broken_policy_grad(), \
        f'policy 梯度又塌了：{gp:.6g} vs 首版 {broken_policy_grad():.6g}'
    assert gp == pytest.approx(9.829e-4, rel=5e-3), \
        f'policy 梯度范数与 report 实测不符：{gp:.6g}（应 ≈ 9.829e-04）'
    assert gv == pytest.approx(0.353553, rel=5e-3), \
        f'value 梯度范数与 report 实测不符：{gv:.6g}（应 ≈ 0.353553 = 1/sqrt(8)）'
    ratio = gv / gp
    assert 200.0 < ratio < 700.0, \
        (f'value:policy 梯度比 = {ratio:.1f}:1 掉出 [200, 700] 区间。'
         '变好：policy 梯度被补回来了（可收紧上界）。'
         '变坏：policy 又被稀释/被补偿压下去了 —— 结构性缺陷回来了。'
         '当前状态：P4.5b 已把默认改回 ce 消除它；本断言现在只是存档守卫。')

    # ce 口径：P4.5 记录它是 ~1.1:1（控制器当时未裁决），P4.5b 已把它设为默认
    _, gp_ce, gv_ce = grads('ce', 1.0)
    assert gv_ce / gp_ce == pytest.approx(1.113, rel=2e-2), \
        f'ce 口径的参照漂了：{gv_ce / gp_ce:.4f}（应 ≈ 1.113）'


def test_huber_loss_api_truth_table():
    """`F.huber_loss(delta=b) == b · F.smooth_l1_loss(beta=b)`，仅 b=1 时相等。

    P4.5-fix 复核结论（torch 2.12.0+cpu，float64，逐位）：真关系是

        F.smooth_l1_loss(beta=b) ≡ 教科书 Huber(delta=b)      （拐点确实在 |d|=b）
        F.huber_loss(delta=b)   == b · 上面那个                （仅 b=1 相等）

    两条被证伪的流行说法（都曾被写进本仓的 docstring / help）：
      · 「huber_loss(delta=b) 与 smooth_l1(beta=b) 在 b<=1 等价」—— 差 b 倍；
      · 「smooth_l1(beta=b) 等价于 Huber(delta=1.0)/b，有效 delta 是 1.0」
        —— 线性段不等（b=0.5 时最大差 2.25），拐点就在 |d|=b。
    """
    d = torch.linspace(-3, 3, 4001, dtype=torch.float64)
    z = torch.zeros_like(d)
    for b in (0.1, 0.25, 0.5, 1.0, 2.0):
        sl1 = F.smooth_l1_loss(d, z, beta=b, reduction='none')
        textbook = _manual_smooth_l1(d, z, b)          # 教科书 Huber(delta=b)
        hl = F.huber_loss(d, z, delta=b, reduction='none')
        assert torch.equal(sl1, textbook), \
            f'smooth_l1(beta={b}) 不等于教科书 Huber(delta={b})'
        assert torch.allclose(hl, b * sl1, rtol=1e-12, atol=1e-15), \
            f'huber_loss(delta={b}) != {b} * smooth_l1(beta={b})'
        # 事实表必须**看住实现**：huber_loss 用的确实是 smooth_l1（不是 huber_loss）
        assert torch.equal(huber_loss(d, z, beta=b),
                           F.smooth_l1_loss(d, z, beta=b)), \
            f'实现用的不是 F.smooth_l1_loss(beta={b})'
        if b != 1.0:
            assert not torch.allclose(hl, sl1, rtol=1e-6), \
                f'b={b} 时两者本该差 b 倍，却相等 —— 上面的等式失效'
        # 证伪「有效 delta 是 1.0」：Huber(delta=1)/b 只在 b=1 时与 smooth_l1 相等
        h1_over_b = torch.where(d.abs() < 1.0, 0.5 * d * d, d.abs() - 0.5) / b
        same = torch.equal(sl1, h1_over_b)
        assert same == (b == 1.0), \
            f'b={b}: 「smooth_l1 ≡ Huber(delta=1)/b」的真假 = {same}，与 b==1 不符'

    # 本测试是「事实表」；它得真的看住这些事实被写进了 docstring/help ——
    # 否则事实对了、文案照样骗人（P4.5 就是在文案上翻的车）。
    doc = t.huber_loss.__doc__ or ''
    for banned in ('梯度恒 ±1', '梯度 = ±1（', '有效 delta 是 1.0'):
        assert banned not in doc, f'huber_loss docstring 仍写着不实表述 {banned!r}'
    assert '1/N' in doc or '除以 N' in doc or '±1/N' in doc, \
        'huber_loss docstring 没说明 mean 归约会除以 N（P4.5-fix 的更正要点）'
    flat_help = ' '.join(_replay_parser().format_help().split())
    assert '梯度恒 ±1' not in flat_help, '--huber-beta 的 help 仍写着「梯度恒 ±1」'
    assert 'delta=beta' in flat_help, \
        '--huber-beta 的 help 没写明 smooth_l1 的 beta 就是拐点 delta'


def test_losses_under_bf16_autocast():
    """BF16 路径（P4.1 的 NPU 部署口径）必须能跑、不 NaN、误差在容差内。

    ⚠ `maybe_autocast()` 在 CPU 上返回 `nullcontext`（device 必须是
    cuda/npu 才开），所以这里**显式**写 `torch.autocast(device_type='cpu',
    dtype=torch.bfloat16)` —— 这是对 NPU bf16 路径的**近似**（同一套 autocast
    机制与同一个 dtype，只是 device_type 不同）。断言的是「损失函数在低精度
    输入下的行为」，这部分与后端无关。

    覆盖：policy（默认 huber，逐样本归约）+ value（默认 huber），两侧都要跑通。
    """
    B, A, beta, eps = _GRAD_B, _GRAD_A, _GRAD_BETA, _GRAD_EPS
    logits32 = torch.zeros(B, A) + 0.05          # 近均匀但非精确均匀
    v32 = torch.zeros(B, 1) + 0.02

    ref_p = float(compute_policy_loss(logits32, _GRAD_MOVES, 'huber',
                                      label_smoothing=eps, huber_beta=beta))
    ref_v = float(compute_value_loss(v32, _GRAD_VT, 'huber', huber_beta=beta))

    logits_bf = logits32.to(torch.bfloat16)
    v_bf = v32.to(torch.bfloat16)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        got_p = compute_policy_loss(logits_bf, _GRAD_MOVES, 'huber',
                                    label_smoothing=eps, huber_beta=beta)
        got_v = compute_value_loss(v_bf, _GRAD_VT, 'huber', huber_beta=beta)
        total = got_p + got_v

    for name, val in (('policy_loss', got_p), ('value_loss', got_v),
                      ('loss', total)):
        assert torch.isfinite(val), f'BF16 下 {name} 非有限值: {val}'
    assert float(got_p) > 0.0 and float(got_v) > 0.0, 'BF16 下损失应为正'

    # 相对误差容差：BF16 只有 8 位尾数（~2^-8 = 3.9e-3），且这里的量级是
    # 「0.65 附近、跨 361 项求和」，实测相对误差 1.1e-3 量级；留 1 个数量级余量。
    for name, got, ref in (('policy_loss', float(got_p), ref_p),
                           ('value_loss', float(got_v), ref_v),
                           ('loss', float(total), ref_p + ref_v)):
        rel = abs(got - ref) / max(abs(ref), 1e-12)
        assert rel < 4e-2, \
            f'BF16 与 fp32 的 {name} 相对误差 {rel:.3e} 超容差 4e-2 ' \
            f'(got={got:.6g}, fp32={ref:.6g})'

    # 反向也要能跑通且梯度有限（BF16 训练的实际路径）
    lp = logits_bf.clone().requires_grad_(True)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        loss = compute_policy_loss(lp, _GRAD_MOVES, 'huber',
                                   label_smoothing=eps, huber_beta=beta)
    loss.backward()
    assert lp.grad is not None and torch.isfinite(lp.grad).all(), \
        'BF16 下 policy 反向产生了非有限梯度'
    # 梯度仍然非零（不是被低精度吃掉的全零）
    assert float(lp.grad.abs().max()) > 0.0, 'BF16 下 policy 梯度恒为 0'

# --------------------------------------------------------------------------- #
# 11.（P4.5b）总损失口径：L = L_policy + L_value + c‖θ‖²（c = --weight-decay）
#
#     用户裁决 2026-09-27：**解耦优化 + 日志口径补 c‖θ‖²**。
#     本节锁三件事：
#       (a) c 就是 --weight-decay、没有新增任何 CLI 参数（D1）；
#       (b) l2_report 只覆盖优化器**真的在衰减**的那组参数；
#       (c) 被 backward 的 opt_loss 与被记日志的 log_loss 是**两个量**，
#           L2 项不进梯度（§3.1 —— 把解耦衰减折进梯度是最危险的一步）。
# --------------------------------------------------------------------------- #


_EPS = 0.1          # --label-smoothing 默认
_BETA = 0.5         # --huber-beta 默认
_WD = 1e-4          # --weight-decay 默认（= 裁决里的 c）
_HEAD_SEED = 20260927


class _Tiny(nn.Module):
    """有 bias 的小 MLP：造出 decay（ndim=2）与 no_decay（ndim=1）两类参数。

    必须带一个 `value` 子模块 —— `_build_param_groups` 按 `'value.'` 前缀把
    value 头单列一组，没有它会直接 AttributeError。
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3, bias=True)
        self.value = nn.Linear(4, 1, bias=True)


def _opt(net, weight_decay=_WD, lr=0.1, value_lr_mult=5.0):
    class _A:
        pass
    a = _A()
    a.lr, a.weight_decay, a.value_lr_mult = lr, weight_decay, value_lr_mult
    return torch.optim.AdamW(t._build_param_groups(net, a), lr=lr)


def test_l2_report_uses_decay_group_only():
    """l2_report 只覆盖 `weight_decay != 0` 的参数（§3.2）。

    判据来自 **optimizer.param_groups 本身**（与优化器实际衰减的集合同源），
    不是重新实现一遍 `ndim == 1`。两条行为断言互相把靶子钉死：

    · 往 **no_decay**（norm 权重 / bias）里塞大数 ⇒ l2_report **一字不变**；
    · 往 **decay**（ndim=2 权重）里塞大数 ⇒ l2_report **按平方精确增加**。
    """
    torch.manual_seed(7)
    net = _Tiny()
    opt = _opt(net)
    wds = [g['weight_decay'] for g in opt.param_groups]
    assert wds == [_WD, 0.0, _WD, 0.0], f'分组 weight_decay 布局变了: {wds}'

    base = t.compute_l2_report(opt.param_groups)
    decay_p = opt.param_groups[0]['params'][0]        # fc.weight，ndim=2
    nodecay_p = opt.param_groups[1]['params'][0]     # fc.bias，ndim=1
    assert decay_p.ndim == 2 and nodecay_p.ndim == 1
    ref = float(base)

    # (1) no_decay 参数塞 1e3：l2_report 必须纹丝不动
    with torch.no_grad():
        nodecay_p.add_(torch.full_like(nodecay_p, 1e3))
    assert t.compute_l2_report(opt.param_groups) == ref, \
        'bias（no_decay 组）被算进 l2_report 了 —— 日志会报告一个仓库没施加的强度'

    # (2) decay 参数塞 1e3：l2_report 恰好增加 wd·(‖p'‖² − ‖p‖²)
    before_sq = float(decay_p.detach().pow(2).sum())
    with torch.no_grad():
        decay_p.add_(torch.full_like(decay_p, 1e3))
    after_sq = float(decay_p.detach().pow(2).sum())
    want = ref + _WD * (after_sq - before_sq)
    got = t.compute_l2_report(opt.param_groups)
    assert got == pytest.approx(want, rel=1e-9), \
        f'L2 项没覆盖 decay 参数：got={got} want={want}'

    # (3) 全量参数求和（**错误**口径）会明显更大 —— 证明上面的断言不是恒真
    allsum = _WD * float(sum(float(p.detach().pow(2).sum())
                             for p in net.parameters()))
    assert allsum > got, '全量求和反而更小：本用例失去区分力'
    assert _WD * float(nodecay_p.detach().pow(2).sum()) == pytest.approx(
        allsum - got, rel=1e-6), 'no_decay 组的差值对不上'

    # (4) 返回值必须是**纯 python float**：一旦变成与参数相连的张量，L2 项就
    #     悄悄进了计算图，log_loss 反向会给参数加上一个 wd·2θ 的梯度 ——
    #     那是把解耦衰减变成耦合 L2，且没有任何报错。
    assert isinstance(got, float) and not isinstance(got, torch.Tensor), \
        f'compute_l2_report 必须返回 python float（纯报告量），实得 {type(got)}'
    assert isinstance(t.compute_l2_report(_opt(net, weight_decay=0.0).param_groups),
                      float), '全 no_decay 时也必须返回 float 0.0'


def test_l2_report_scales_with_weight_decay():
    """c 就是 `--weight-decay`，没有被硬编码成别的常数（§4.3 / D1）。

    `--weight-decay 0` ⇒ 恒 0（优化器不衰减 ⇒ 报告里也不该有）；翻倍 ⇒ 翻倍。
    """
    torch.manual_seed(7)
    net = _Tiny()
    seen = {}
    for wd in (0.0, _WD, 2 * _WD, 0.05):
        seen[wd] = t.compute_l2_report(_opt(net, weight_decay=wd).param_groups)
    assert seen[0.0] == 0.0, '--weight-decay 0 时 l2_report 必须恒 0'
    assert seen[_WD] == pytest.approx(seen[2 * _WD] / 2, rel=1e-9), \
        'l2_report 与 --weight-decay 不成正比（c 被硬编码了？）'
    assert seen[0.05] == pytest.approx(500 * seen[_WD], rel=1e-9), \
        'l2_report 与 --weight-decay 不成正比（c 被硬编码了？）'

    # c 只能来自 --weight-decay：模块里不许出现第二个 L2 系数
    lit = [n.value for n in ast.walk(_fn('compute_l2_report'))
           if isinstance(n, ast.Constant) and isinstance(n.value, float)
           and 0 < n.value < 1]
    assert not lit, f'compute_l2_report 里出现了硬编码浮点常数: {lit}'


def test_log_loss_identity():
    """`log_loss == policy + w·value + l2_report` 且 `opt_loss` **不含** L2（§3.1）。

    这是防「把解耦衰减折进梯度」的核心守卫，用**行为**而不是源码子串：

    1. 两个量的数值关系（浮点容差内）；
    2. `log_loss.backward()` 与 `opt_loss.backward()` 给出**逐位相同**的参数
       梯度 —— 即便对含 L2 项的 log_loss 反向，L2 项也贡献不出任何梯度
       （l2_report 是 python float，不在计算图里）。这条一旦变红，说明有人把
       `c‖θ‖²` 变成了可微张量接进 backward ⇒ 优化行为已经改变。

    另外：w ≠ 1 时报告口径必须**同样带权**，否则 `loss` 与被优化的目标会差两项。
    """
    l2 = 0.588066
    for w in (1.0, 2.5):
        pol = torch.tensor(2.0, requires_grad=True)
        val = torch.tensor(0.5, requires_grad=True)
        opt_loss, log_loss = t.compose_losses(pol, val, w, l2)
        opt_v, log_v = float(opt_loss.detach()), float(log_loss.detach())
        assert opt_v == pytest.approx(2.0 + w * 0.5, rel=1e-6), \
            'opt_loss 必须恰好是 policy + w·value（不含 L2）'
        assert log_v == pytest.approx(2.0 + w * 0.5 + l2, rel=1e-6), \
            'log_loss 必须恰好是 policy + w·value + l2_report'
        assert log_v - opt_v == pytest.approx(l2, rel=1e-6), \
            '两个口径之差必须**只有** L2 这一项'

    grads = {}
    for name, idx in (('opt', 0), ('log', 1)):
        pol = torch.tensor(2.0, requires_grad=True)
        val = torch.tensor(0.5, requires_grad=True)
        pair = t.compose_losses(pol, val, 1.0, l2)
        pair[idx].backward()
        grads[name] = (pol.grad.clone(), val.grad.clone())
    assert torch.equal(grads['opt'][0], grads['log'][0]), \
        'log_loss 的梯度与 opt_loss 不同 ⇒ L2 项进了计算图（§3.1 被漏改）'
    assert torch.equal(grads['opt'][1], grads['log'][1])
    assert float(grads['opt'][0]) == 1.0 and float(grads['opt'][1]) == 1.0

    # --- 端到端：喂**真的** l2_report 与真的参数进去（上面用的是手写常数，
    #     抓不到「compute_l2_report 返回了可微张量」这类变异）---
    def end_to_end(idx):
        torch.manual_seed(5)
        net = _Tiny()
        x = torch.randn(4, 4, generator=torch.Generator().manual_seed(9))
        pol_l = compute_policy_loss(net.fc(x), torch.tensor([0, 1, 2, 0]), 'ce',
                                    label_smoothing=_EPS, huber_beta=_BETA)
        val_l = compute_value_loss(net.value(x), torch.ones(4, 1), 'huber',
                                   huber_beta=_BETA)
        l2_real = t.compute_l2_report(_opt(net).param_groups)
        opt_v, log_v = t.compose_losses(pol_l, val_l, 1.0, l2_real)
        # 恒等式的**精度上限是 float32 的舍入**，不是 l2 的大小：log_loss 与
        # opt_loss 都是 O(1)~O(10) 的 fp32，两数相减只剩 ~1e-7 的绝对精度，
        # 而 l2_report 可以小到 1e-4 量级（刚初始化的权重）。所以容差按
        # 「操作数的 float32 eps」给，而不是按 l2 的相对误差给。
        gap = abs(float(log_v.detach() - opt_v.detach()) - l2_real)
        tol = 8 * float(torch.finfo(torch.float32).eps) * abs(float(log_v.detach()))
        assert gap <= tol, f'真值代入后恒等式不成立：差 {gap:.3e} > 容差 {tol:.3e}'
        assert l2_real > 0.0, 'l2_report 应为正'
        (opt_v, log_v)[idx].backward()
        return [p.grad.clone() for p in net.parameters()], l2_real

    g_opt, l2_real = end_to_end(0)
    g_log, _ = end_to_end(1)
    for a, b in zip(g_opt, g_log):
        assert torch.equal(a, b), \
            '真参数上两个口径的梯度不同 ⇒ L2 项进了计算图（§3.1 被漏改）'
    assert any(float(g.abs().max()) > 0 for g in g_opt), '梯度全零，本用例无效'

    # 接线：main() 必须 backward 第一个返回值、记日志第二个返回值，且
    # l2_report 取自 optimizer.param_groups（不是 model.parameters()）
    main_src = inspect.getsource(t.main)
    assert 'compose_losses(' in main_src, 'main() 没用 compose_losses 拆两个口径'
    assert 'scaler.scale(opt_loss / _accum_steps).backward()' in main_src, \
        'backward 的必须是 opt_loss（含 L2 会把解耦衰减变成耦合 L2）'
    assert '_read_log_scalars(log_loss, policy_loss, value_loss)' in main_src, \
        '日志读的第一个标量必须是 log_loss'
    assert 'compute_l2_report(optimizer.param_groups)' in main_src, \
        'l2_report 必须从 optimizer.param_groups 取（与实际衰减同源）'
    # 单个 `loss` 变量不许再同时承担两个角色
    bare = [n.lineno for n in ast.walk(_fn('main'))
            if isinstance(n, ast.Name) and n.id == 'loss'
            and isinstance(n.ctx, ast.Store)]
    assert not bare, f'main() 里仍有一个裸 `loss` 变量（既 backward 又记日志）: {bare}'


def test_no_new_cli_params():
    """D1：除既有 61 个选项外**零新增**（防「顺手加个 --l2-coef」）。

    c 就是 `--weight-decay`（已默认 1e-4），裁决明确「不得新增任何 CLI 参数」。
    这里冻结的是**选项名集合**（含顺序）：删一个、改名一个、插入一个都会红。
    """
    kw = _add_argument_kwargs(_fn('main'))
    expected = [
        '--data', '--max-games-per-tgz', '--device', '--use-amp', '--batch-size',
        '--epochs', '--lr', '--weight-decay', '--board-size', '--save-every',
        '--out', '--ver', '--backbone-channels', '--backbone-res-blocks',
        '--attention-mode', '--num-attention-layers', '--num-heads',
        '--attention-dropout', '--attn-mode', '--attn-window', '--eval-every',
        '--eval-max-batches', '--log-every', '--log-file', '--res-blocks',
        '--convnext-blocks', '--attn-blocks', '--value-res-blocks',
        '--value-channels', '--policy-channels', '--policy-layers',
        '--prefetch-workers', '--prefetch-depth', '--resume', '--model',
        '--value-loss-weight', '--policy-loss', '--value-loss', '--huber-beta',
        '--value-lr-mult', '--label-smoothing', '--use-checkpoint', '--use-ema',
        '--gradient-accumulation-steps', '--scaler-init-scale',
        '--scaler-growth-interval', '--npu-graph-compile', '--compile',
        '--compile-mode', '--flash-attn', '--arch', '--export-onnx',
        '--onnx-quantize', '--swanlab', '--swanlab-api-key', '--swanlab-every',
        '--early-stop', '--early-stop-patience', '--early-stop-metric',
        '--max-gpu-memory', '--c2net',
    ]
    assert list(kw) == expected, (
        f'CLI 选项集合被改动（D1：零新增/零删除/零改名）。'
        f'新增={sorted(set(kw) - set(expected))} 缺失={sorted(set(expected) - set(kw))}')
    # 特别地：L2 系数不许有独立参数，label smoothing 也不许有第二个旋钮
    for banned in ('--l2-coef', '--l2-weight', '--weight-decay-l2',
                   '--l2-report', '--label-smoothing-ce'):
        assert banned not in kw, f'新增了 L2 相关 CLI 参数 {banned}（D1 禁止）'
    assert '--label-smoothing' in kw
    assert sum(1 for k in kw if 'label-smoothing' in k) == 1, \
        '--label-smoothing 出现了第二个入口（用户未要求）'


def test_optimizer_param_groups_unchanged():
    """四组的 `weight_decay` 仍是 `{wd, 0.0, wd, 0.0}` —— 防双重正则（§3.1）。

    P4.5b 只动**报告**，优化器一个字节都没改：解耦衰减仍在 AdamW 里按
    `{非 value, value} × {decay, no_decay}` 四组施加。若有人把 L2 塞进
    `param_groups`（例如给 decay 组再加一个 `l2_coef`），本测试会红。
    """
    torch.manual_seed(3)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.back = nn.Linear(4, 4)          # ndim=2 → decay
            self.backbn = nn.BatchNorm1d(4)     # ndim=1 → no_decay
            self.value = nn.Linear(4, 1)         # value 头，独立 LR

        def forward(self, x):
            return self.back(self.backbn(x)), self.value(x)

    net = _Net()
    for wd in (_WD, 0.0, 0.02):
        opt = _opt(net, weight_decay=wd, lr=0.01)
        assert [g['weight_decay'] for g in opt.param_groups] == \
            [wd, 0.0, wd, 0.0], \
            f'wd={wd} 时四组 weight_decay 布局变了（双重正则？）'
        lrs = [g['lr'] for g in opt.param_groups]
        assert lrs[0] == lrs[1] == 0.01 and lrs[2] == lrs[3] == 0.01 * 5.0, \
            f'per-group lr 布局变了: {lrs}'
        assert all(p.ndim == 2 for p in opt.param_groups[0]['params'])
        assert all(p.ndim == 1 for p in opt.param_groups[1]['params'])
        assert all(p.ndim == 2 for p in opt.param_groups[2]['params'])
        assert all(p.ndim == 1 for p in opt.param_groups[3]['params'])
        assert len(opt.param_groups[3]['params']) > 0, \
            'value 的 no_decay 组恒空（P4.5 修过一次的坑）'

    # AdamW 仍是 AdamW（解耦衰减的载体），没有被换成裸 SGD / LBFGS，
    # 且每一处构造都吃同一个 _build_param_groups 结果（本任务不得改优化器）
    assert isinstance(_opt(net), torch.optim.AdamW)
    main_src = inspect.getsource(t.main)
    assert main_src.count('torch.optim.AdamW(') == \
        main_src.count('AdamW(_opt_groups'), \
        '有 AdamW 构造点没在用 _build_param_groups 的分组'
    assert main_src.count('torch.optim.AdamW(') >= 1, 'AdamW 构造点不见了'
    assert '_build_param_groups(model, args)' in main_src, \
        'param_groups 的来源被换掉了'
    assert 'compute_l2_report' not in ast.unparse(_fn('_build_param_groups')), \
        'compute_l2_report 渗进了 _build_param_groups（优化器不得动）'

# --------------------------------------------------------------------------- #
# 12.（P4.5b 简报 §5 第 7/8 条）生产头上的梯度实测 —— 三次三个数的解药
#
#     harness（**必须逐字复现**，否则测的不是同一个量；这就是前几轮给出三个
#     不同数字的原因：没人写下「在哪个空间量、用哪个头、什么初值」）：
#       · 头 = **生产** FCPolicyHead(in_channels=184, board_size=19) /
#               FCValueHead(in_channels=184)（P4.1 的 v21 两头，不是替身）
#       · 权重 = torch.manual_seed(20260927) 之后实例化（默认初始化）
#       · 特征 x ~ N(0,1)，取自 torch.Generator().manual_seed(20260927)，B=8
#       · 目标 = one-hot moves（arange(8)·(A//8)，A 内不重复）；value_t 交替 ±1
#       · beta=0.5；eps=0.1（--label-smoothing 默认）；w=1.0
#
#     三个「空间」都测，因为它们各自只看不同的东西（**这也是分歧的来源**）：
#       · logit 空间 = ‖∂L/∂policy_logits‖ vs ‖∂L/∂value_pred‖：只看**损失**
#         的选择，与头长什么样无关（换头它不变 ⇒ 测不出「换掉任一头」）。
#       · 头参数空间 = 两个头各自全部参数上的梯度范数：AdamW 真正更新的量，
#         对**头结构**与**损失归约**都敏感。
#       · 共享输入空间 = ‖∂L/∂x‖：主干收到的信号（P4.5 报告 §8.3/§8.4 的要害）。
# --------------------------------------------------------------------------- #


def _prod_pair(board, seed=_HEAD_SEED, B=8):
    from src.networks.policy_network import FCPolicyHead
    from src.networks.value_network import FCValueHead
    torch.manual_seed(seed)
    pol = FCPolicyHead(in_channels=184, board_size=board)
    val = FCValueHead(in_channels=184)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 184, board, board, generator=g)
    A = board * board + 1
    moves = torch.arange(B) * (A // B)
    vt = torch.ones(B, 1)
    vt[1::2] = -1.0
    return pol, val, x, moves, vt, A


def _grad_spaces(board=19, pol_kind='ce', val_kind='huber', w=1.0,
                 eps=_EPS, beta=_BETA, seed=_HEAD_SEED, B=8):
    """返回 {'logit': (gp, gv), 'x': (gp, gv), 'param': (gp, gv)}。"""
    pol, val, x, moves, vt, _ = _prod_pair(board, seed, B)
    x = x.requires_grad_(True)
    logits, vpred = pol(x), val(x)
    L = compute_policy_loss(logits, moves, pol_kind,
                            label_smoothing=eps, huber_beta=beta)
    if val_kind == 'bce':            # D4 之前的口径
        tgt = (vt + 1) / 2 * 0.8 + 0.1
        V = F.binary_cross_entropy_with_logits(vpred.squeeze(), tgt.squeeze())
    else:
        V = compute_value_loss(vpred, vt, val_kind, huber_beta=beta)
    gx_p = torch.autograd.grad(L, x, retain_graph=True)[0]
    gx_v = torch.autograd.grad(V, x, retain_graph=True)[0]
    g_logit = torch.autograd.grad(L, logits, retain_graph=True)[0]
    g_vpred = torch.autograd.grad(V, vpred, retain_graph=True)[0]
    (L + w * V).backward()
    gp = torch.sqrt(sum((p.grad ** 2).sum() for p in pol.parameters()))
    gv = torch.sqrt(sum((p.grad ** 2).sum() for p in val.parameters()))
    return {
        'logit': (float(g_logit.norm()), float(g_vpred.norm())),
        'x': (float(gx_p.norm()), float(gx_v.norm())),
        'param': (float(gp), float(gv)),
    }


def test_policy_value_gradient_ratio():
    """当前默认（policy=ce + value=huber, w=1.0）在生产头上的 policy:value 梯度比。

    **简报 §5 第 7 条给的参照 5.4 在本 harness 的三个空间里都复现不出来**
    （实测 0.92 / 24.8 / 91.7，见 report `## Fix2（b）增补` §5），所以本测试钉的
    是**本 harness 的实测值**，并把 harness 逐字写在上面的模块注释里 ——
    「同一个量」只有在「量法」被钉死之后才存在。断言用区间而不是等值，
    留 torch 版本 / 平台的初始化与浮点差异的余量。

    两条设计约束（简报点名的「换掉任一头、或改任一损失的归约必须变红」）：

    · 头参数空间对**头结构**敏感 ⇒ 换头会红（logit 空间不会，它与头无关，
      见断言 (4)）；
    · 归约敏感性由 `test_ce_policy_gradient_is_action_space_independent` 覆盖
      （把 Huber 的 sum over A 换成 mean over A，policy 梯度再掉 4.4 倍）。
    """
    cur = _grad_spaces(19, pol_kind='ce', val_kind='huber', w=1.0)

    # (1) 头参数空间：policy 强于 value。实测 91.70（eps=0.1）/ 101.92（one-hot）。
    ratio = cur['param'][0] / cur['param'][1]
    assert 40.0 < ratio < 200.0, (
        f'头参数空间 policy:value = {ratio:.2f}:1 掉出 [40, 200]。'
        f'变大：policy 侧被放大（损失或归约被换过？）'
        f'变小：policy 梯度被稀释 —— A 依赖回来了（见下一个测试）。'
        f'当前实测 91.70（eps=0.1）/ 101.92（one-hot）。')
    assert cur['param'][0] == pytest.approx(4.8710, rel=5e-3), \
        f'policy 头参数梯度漂了: {cur["param"][0]:.5f}（应 ≈ 4.8710）'
    assert cur['param'][1] == pytest.approx(0.0531, rel=1e-2), \
        f'value 头参数梯度漂了: {cur["param"][1]:.5f}（应 ≈ 0.0531）'

    # (2) 损失输入空间：value:policy = 1.113 —— 与 P4.5 报告记录的 ce 备选口径
    #     逐位一致（独立 harness 得到同一个数），即「不需要补偿旋钮」。
    vr = cur['logit'][1] / cur['logit'][0]
    assert vr == pytest.approx(1.113, rel=2e-2), \
        f'损失输入空间 value:policy = {vr:.4f}（应 ≈ 1.113，老口径 ce+5·bce 是 2.225）'
    assert 0.9 < vr < 1.4, 'policy/value 在损失输入空间应同量级（1:1 附近）'

    # (3) 方向：P4.5 交付口径（huber+huber）里 value 是强的，现在反过来了 ——
    #     这是「默认改 ce 真的生效了」的可执行证据。
    old = _grad_spaces(19, pol_kind='huber', val_kind='huber', w=1.0)
    old_ratio = old['param'][0] / old['param'][1]
    assert old_ratio < 1.0, \
        'P4.5 交付口径（huber）下 value 本该更强，比值却 > 1 —— 参照系坏了'
    assert old_ratio == pytest.approx(0.29, rel=2e-2), \
        f'P4.5 交付口径的头参数比漂了: {old_ratio:.3f}（应 ≈ 0.29）'
    assert ratio > 10 * old_ratio, \
        '换成 ce 之后 policy 梯度应放大约两个数量级（实测 316×）'

    # (4) 换头的守卫：同样两个损失换 v18 的 PolicyNetwork/ValueNetwork，
    #     头参数空间的比值必须明显不同 —— 否则本测试与「头」无关，等于没测。
    from src.networks.policy_network import PolicyNetwork
    from src.networks.value_network import ValueNetwork

    def alt_pair(pm, vm, seed=_HEAD_SEED, B=8):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(B, 184, 19, 19, generator=g).requires_grad_(True)
        moves = torch.arange(B) * 45
        vt = torch.ones(B, 1)
        vt[1::2] = -1.0
        L = compute_policy_loss(pm(x), moves, 'ce', label_smoothing=_EPS,
                                huber_beta=_BETA)
        V = compute_value_loss(vm(x), vt, 'huber', huber_beta=_BETA)
        (L + V).backward()
        gp = torch.sqrt(sum((p.grad ** 2).sum() for p in pm.parameters()))
        gv = torch.sqrt(sum((p.grad ** 2).sum() for p in vm.parameters()))
        return float(gp) / float(gv)

    alt = alt_pair(PolicyNetwork(in_channels=184, hidden_channels=32,
                                 action_size=362, num_layers=2),
                   ValueNetwork(in_channels=184, hidden_channels=64,
                                num_res_blocks=3, arch='resnet'))
    assert abs(alt / ratio - 1.0) > 0.5, (
        f'换成 v18 两头后比值几乎不变（{alt:.2f} vs {ratio:.2f}）'
        f'—— 本测试对「头」不敏感，测不出换头')


def test_ce_policy_gradient_is_action_space_independent():
    """**为什么默认不是 huber**，钉成可执行的不变式（简报 §5 第 8 条）。

    同一批特征（同一 seed 的同一 generator）、同一组权重（两个头在
    `torch.manual_seed` 之后实例化；conv stem 的形状在 9/19 路下相同 ⇒ 逐位
    相同，分类器 FC 因 flatten 维不同而必然不同 —— 那正是被测的 A 依赖），
    分别在 A=82（9 路）与 A=362（19 路）下测 `‖∂L/∂policy_logits‖`：

        CE              0.3512 → 0.3530    比值 1.005   ← 与 A 无关
        Huber(p) sum    0.0046 → 0.0010    比值 0.223   = 82/362 = 1/A
        Huber(p) mean                →     比值 0.051   = (1/A)²

    机制：CE 打在 log-prob 上，`d(CE)/d(logit_j) = p_j − y_j`，每坐标有界；
    Huber 打在概率上，要再乘一层 softmax 雅可比 `∂p_k/∂logit_j = p_k(δ_kj−p_j)`，
    专家坐标上带一个 `p ≈ 1/A` ⇒ 梯度按 **1/A** 缩放。于是 `sum over A` 给 A、
    `mean over A` 给 1/A²，**没有任何归约能消掉这个 A 依赖**；本仓支持
    9/13/19 路 ⇒ 同一学习率在 9 路与 19 路之间差 **4.4 倍**。

    量在 logit 空间而不是头参数空间：后者会同时混入「分类器参数量随 flatten
    维变化」的结构差异（实测 CE 的头参数比是 2.128，与损失选择无关），把
    要测的信号淹掉。
    """
    def logit_norm(board, kind, reduce='sum', eps=0.0):
        pol, _, x, moves, _, A = _prod_pair(board)
        x = x.requires_grad_(True)
        logits = pol(x)
        if kind == 'ce':
            L = compute_policy_loss(logits, moves, 'ce',
                                    label_smoothing=eps, huber_beta=_BETA)
        else:
            L = _huber_policy(logits, moves, eps, _BETA, reduce)
        return float(torch.autograd.grad(L, logits)[0].norm()), A

    # --- CE：与 A 无关 ---
    c82, A82 = logit_norm(9, 'ce')
    c362, A362 = logit_norm(19, 'ce')
    assert (A82, A362) == (82, 362)
    ratio_ce = c362 / c82
    assert 0.8 < ratio_ce < 1.25, (
        f'CE 的 policy 梯度不该依赖动作空间：实测 362/82 = {ratio_ce:.3f}'
        f'（{c82:.4f} → {c362:.4f}）。> 1.25 说明 CE 也带上了 A 依赖，'
        f'默认值的理由就不成立了。')

    # --- Huber(p) sum over A：严格按 1/A 缩放 ---
    h82, _ = logit_norm(9, 'huber', 'sum')
    h362, _ = logit_norm(19, 'huber', 'sum')
    ratio_h = h362 / h82
    assert ratio_h < 0.5, (
        f'Huber(p) sum over A 的梯度应当随 A 显著变小，实测 362/82 = {ratio_h:.3f}'
        f' —— 若它也变成 A 无关了，本测试就不再证明「为什么默认不是 huber」。')
    assert ratio_h == pytest.approx(A82 / A362, rel=0.05), (
        f'Huber(p) sum over A 的梯度应按 1/A 缩放：实测 {ratio_h:.4f}，'
        f'1/A = {A82 / A362:.4f}（差 {abs(ratio_h / (A82 / A362) - 1) * 100:.1f}%）')

    # --- Huber(p) mean over A：按 1/A² 缩放（更糟，这正是 P4.5 修掉的归约）---
    m82, _ = logit_norm(9, 'huber', 'mean')
    m362, _ = logit_norm(19, 'huber', 'mean')
    ratio_m = m362 / m82
    assert ratio_m < 0.15, (
        f'Huber(p) mean over A 应按 1/A² 缩放（≈ {(A82 / A362) ** 2:.4f}），'
        f'实测 {ratio_m:.4f}')
    assert ratio_m == pytest.approx((A82 / A362) ** 2, rel=0.1), (
        f'Huber(p) mean over A 的 1/A² 律不成立：{ratio_m:.4f} vs '
        f'{(A82 / A362) ** 2:.4f}')

    # --- 相对强度：CE 远强于两种 Huber 归约（近均匀初值下 ∝ p≈1/A）---
    assert c362 > 100 * h362, \
        '同一初值下 CE 的 policy 梯度应比 Huber(p) sum 强两个数量级以上'
    assert c362 > 10 * m362, \
        '同一初值下 CE 的 policy 梯度应比 Huber(p) mean 强一个数量级以上'

    # --- 实现不许偷换归约：默认的 `compute_policy_loss(kind='huber')` 必须就是
    #     上面那个 `sum over A` 口径。否则本测试的 sum 行会与被保留的实验选项
    #     脱节，「A 依赖是 1/A」这句话就不再描述真实可跑的代码。
    pol, _, x, moves, _, _ = _prod_pair(19)
    logits = pol(x)
    impl = compute_policy_loss(logits, moves, 'huber', label_smoothing=0.0,
                               huber_beta=_BETA)
    assert torch.equal(impl, _huber_policy(logits, moves, 0.0, _BETA, 'sum')), \
        'compute_policy_loss 的 huber 归约不再是「类内 sum over A」' \
        '（1/A 依赖的实测会与实现脱节）'


def _huber_policy(logits, moves, eps, beta, reduce):
    """`compute_policy_loss(kind='huber')` 的显式复刻（sum / mean 两种归约）。

    复刻而不是调用：默认口径只有 sum，mean 是**被 P4.5 修掉的那个实现**，
    这里要拿它当「1/A² 律」的反例靶子。
    """
    A = logits.shape[-1]
    with torch.no_grad():
        y = torch.full_like(logits, eps / A)
        y.scatter_(1, moves.long().view(-1, 1), 1.0 - eps + eps / A)
    h = _manual_smooth_l1(F.softmax(logits, dim=-1), y, beta)
    return h.sum(dim=-1).mean() if reduce == 'sum' else h.mean()
