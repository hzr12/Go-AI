"""D4（SFT 侧）：--policy-loss / --value-loss / --huber-beta 三参数 + 删 BCE 分支。

对应简报：.superpowers/sdd/2026-09-25-v21-roadmap/task-p4-5-brief.md §3 的
7 个规定用例，另加 2 个接线/语义锁（test_flags_reach_loss_calls、
test_policy_huber_semantics_pinned）。每个用例「能红」的靶子见
task-p4-5-report.md 的对照表。

P4.5-fix 追加（对应 task-p4-5-fix-brief.md）：
* `test_policy_reduction_is_per_sample`  —— 钉住 §1 修掉的 1/A 归约 bug
  （D4 首版对 B×A 求均值，等于白吃一个 1/361 的稀释）。
* `test_gradient_scale_not_collapsed`   —— 把 report `## Fix 增补` 的梯度实测
  变成回归守卫（B=8, A=361, beta=0.5, targets ±1）。
* `test_value_loss_weight_default_is_one_no_compensation` —— 钉住「删补偿」。
* `test_huber_beta_rejects_non_positive` —— argparse 校验（0 / 负值 / 非数）。
* `test_losses_under_bf16_autocast`     —— BF16 路径（既定 NPU 部署口径）。
* `test_huber_loss_api_truth_table`      —— huber_loss vs smooth_l1 的真实关系。
* 加固：`test_huber_matches_torch_reference` 里近乎自指的默认 beta 断言改成
  手写公式对拍；`test_value_target_range_documented` 的两条纯 docstring 子串
  断言换成行为断言（梯度方向 + 未被重映射）。

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
# 3. 三个旗的真实 argparse 定义：默认 huber / huber / 0.5
# --------------------------------------------------------------------------- #
def test_default_args_are_huber():
    kw = _add_argument_kwargs(_fn('main'))
    for flag in ('--policy-loss', '--value-loss', '--huber-beta'):
        assert flag in kw, f'main() 缺少 {flag}（D4 三参数之一）'

    assert ast.literal_eval(kw['--policy-loss']['default']) == 'huber'
    assert ast.literal_eval(kw['--policy-loss']['choices']) == ['huber', 'ce']
    assert ast.literal_eval(kw['--value-loss']['default']) == 'huber'
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
    assert (ns.policy_loss, ns.value_loss, ns.huber_beta) == ('huber', 'huber', 0.5), \
        f'默认解析结果错误: {(ns.policy_loss, ns.value_loss, ns.huber_beta)}'
    ns2 = ap.parse_args(['--data', 'd', '--policy-loss', 'ce',
                         '--value-loss', 'mse', '--huber-beta', '0.1'])
    assert (ns2.policy_loss, ns2.value_loss, ns2.huber_beta) == ('ce', 'mse', 0.1)
    for bad in (['--policy-loss', 'bce'], ['--value-loss', 'bce'],
                ['--policy-loss', 'huber_v2']):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
            ap.parse_args(['--data', 'd'] + bad)
        assert 'invalid choice' in err.getvalue(), \
            f'非法值 {bad} 未被 argparse 拒绝: {err.getvalue()!r}'


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

    # 权重仍作用在 value 项上（只是默认值不再补偿）
    main_src = inspect.getsource(t.main)
    assert 'policy_loss + args.value_loss_weight * value_loss' in main_src, \
        'loss 的组合式被改动（应仍是 policy_loss + w * value_loss）'
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
       这条**不是**「已修好」的断言 —— 它是**残余失衡的守卫**：哪天 policy 梯度
       再塌一个数量级，或有人用 `--value-loss-weight` 悄悄把它调平（= 把这个
       结构性缺陷藏进一个调参值里），本测试会红。
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

    # 1+2. 修后口径（默认 huber + huber，--value-loss-weight 默认 1.0）
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
         '当前状态：仍是残余失衡，见 report `## Fix 增补` §残余比值。')

    # ce 备选：把 policy 默认换回 ce 就能回到 ~1.1:1（控制器尚未裁决）
    _, gp_ce, gv_ce = grads('ce', 1.0)
    assert gv_ce / gp_ce == pytest.approx(1.113, rel=2e-2), \
        f'ce 备选口径的参照漂了：{gv_ce / gp_ce:.4f}（应 ≈ 1.113）'


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
