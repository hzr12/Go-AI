"""梯度检查点（GC）：**标准实现**、默认关、逐位中性、可选开启。

为什么重新引入
--------------
GC 用显存换一次多余的主干前向。40GB 卡上现有 batch 一般用不上，所以**默认 0**；
它的价值是「OOM 时比砍 batch 更划算」（砍 batch 会加大梯度噪声）。

本文件钉六件事
--------------
1. 旗 ``--grad-checkpoint`` 存在且**默认 0**（从真 argparse 重放）。
2. **默认关**：两条建网路径都不改任何数值、参数、形状。
3. **逐位中性**：开/关前向输出与反向梯度**逐位相同**。这不是「差不多」——
   它是硬相等，否则 loss 曲线在开关之间不可比。
4. **BN 统计不被重算双累加**：没有守卫时 `running_mean` 真的会变、
   `num_batches_tracked` 真的会 1→2（守卫存在与否都能测出来，见下）。
5. **确实重算**：开着时块执行次数 = 2×，不是「建了图但没重算」的假货。
6. **与 `torch.compile` 共存**：V7 路径在 `fullgraph=True` 下不 graph break
   —— 这正是选标准实现（而非自定义 `autograd.Function`）的全部理由。

跑：pytest tests/test_grad_checkpoint.py -v
"""
import argparse
import ast
import copy
import inspect
import io
import os
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.train_sft as t  # noqa: E402
from src.networks.backbone import SharedBackbone  # noqa: E402

_BN = nn.modules.batchnorm._BatchNorm


# --------------------------------------------------------------------------- #
# argparse 工具（与 tests/test_soft_cli.py 同一套口径：重放，不执行 main()）
# --------------------------------------------------------------------------- #
def _main_ast():
    return ast.parse(textwrap.dedent(inspect.getsource(t.main)))


def _replay_parser():
    ap = argparse.ArgumentParser()
    scope = dict(vars(t))
    scope['ap'] = ap
    for node in ast.walk(_main_ast()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            expr = ast.Expression(body=node)
            ast.copy_location(expr, node)
            exec(compile(expr, '<train_sft argparse>', 'eval'), scope)
    return ap


def test_gc_flag_exists_and_defaults_off():
    """`--grad-checkpoint 0|1`，默认 **0**。

    默认关是硬要求：现网 batch 在 40GB 上跑得动，GC 只会白付一次前向。
    """
    ap = _replay_parser()
    ns = ap.parse_args(['--data', 'x.npz'])     # --data 是必填，给个占位
    assert ns.grad_checkpoint == 0
    assert ap.parse_args(['--data', 'x.npz',
                          '--grad-checkpoint', '1']).grad_checkpoint == 1
    with pytest.raises(SystemExit):
        ap.parse_args(['--data', 'x.npz', '--grad-checkpoint', '2'])


def _v7(**kw):
    from src.networks.katago_v7 import build_katago_v7_net
    torch.manual_seed(0)
    return build_katago_v7_net(board_size=19, **kw)


def test_gc_is_off_by_default_on_both_paths():
    """两个构建器都不传时 `grad_checkpoint` 为 False —— 传不进主干。"""
    bb = SharedBackbone(in_channels=12, channels=32, num_res_blocks=3,
                        attention_mode='none', arch='se_bottleneck')
    assert bb.grad_checkpoint is False
    net = _v7()
    assert net.grad_checkpoint is False


# --------------------------------------------------------------------------- #
# 逐位中性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('arch', ['se_bottleneck', 'resnet', 'convnext'])
def test_backbone_gc_is_bit_exact(arch):
    """同一份权重，开/关跑出来的输入梯度**逐位相同**。

    用同一份权重跑两遍（而不是建两个网）—— 两次构造出来的随机权重不同，
    那样比出来的「差异」全是权重的，测不到 GC 自己。
    """
    torch.manual_seed(0)
    bb = SharedBackbone(in_channels=12, channels=32, num_res_blocks=3,
                        attention_mode='none', arch=arch)
    w = copy.deepcopy(bb.state_dict())
    x0 = torch.randn(2, 12, 19, 19)

    def run(gc):
        bb.load_state_dict(w)
        bb.grad_checkpoint = gc
        bb.train()
        x = x0.clone().requires_grad_(True)
        y = bb(x)
        y.sum().backward()
        return y.detach().clone(), x.grad.detach().clone()

    y0, g0 = run(False)
    y1, g1 = run(True)
    assert torch.equal(y0, y1), '前向不逐位一致'
    assert torch.equal(g0, g1), '反向梯度不逐位一致'


def test_v7_gc_is_bit_exact():
    """V7 路径同样逐位中性，且所有输出头都要覆盖到。"""
    torch.manual_seed(0)
    net = _v7()
    w = copy.deepcopy(net.state_dict())
    sp0 = torch.randn(2, 22, 19, 19)
    gl0 = torch.randn(2, 19)

    def run(gc):
        net.load_state_dict(w)
        net.grad_checkpoint = gc
        net.train()
        s = sp0.clone().requires_grad_(True)
        o = net(s, gl0)
        (o['policy_logits'].float().pow(2).mean()
         + o['outcome_logits'].pow(2).mean()).backward()
        return {k: v.detach().clone() for k, v in o.items()}, s.grad.detach().clone()

    o0, g0 = run(False)
    o1, g1 = run(True)
    assert sorted(o0) == sorted(o1)
    for k in o0:
        assert torch.equal(o0[k], o1[k]), f'{k} 不逐位一致'
    assert torch.equal(g0, g1), '输入梯度不逐位一致'


def test_gc_actually_recomputes_the_blocks():
    """开着时块**执行两次**（前向 + 反向重算）。

    防的是「建了 checkpoint 对象但实际没重算」的假货 —— 那种实现省不到显存，
    却白白多了一次前向的记账开销。
    """
    net = _v7()
    cnt = {'n': 0}
    h = net.blocks[0].register_forward_hook(
        lambda m, i, o: cnt.__setitem__('n', cnt['n'] + 1))
    try:
        counts = {}
        for gc in (False, True):
            net.grad_checkpoint = gc
            net.train()
            cnt['n'] = 0
            sp = torch.randn(2, 22, 19, 19, requires_grad=True)
            out = net(sp, torch.randn(2, 19))
            fwd = cnt['n']
            out['policy_logits'].float().pow(2).mean().backward()
            counts[gc] = (fwd, cnt['n'])
    finally:
        h.remove()
    assert counts[False] == (1, 1), counts
    assert counts[True] == (1, 2), counts


# --------------------------------------------------------------------------- #
# BN 统计守卫
# --------------------------------------------------------------------------- #
def test_bn_stats_are_not_double_counted():
    """重算会把块内 BN 的前向再跑一遍 ⇒ 守卫不存在时统计**真的**被双累加。

    这个用例的价值在于它同时钉住了两件事：
      * 守卫**确实在做事**（否则 running_mean 会不同、nbt 会 1→2）；
      * 守卫**没有过度干预**（开着时统计与「完全没开 GC」的基线一致）。

    只测后者的话，一个「把 BN 统计整个冻住」的错实现也能假绿。
    """
    torch.manual_seed(0)
    bb = SharedBackbone(in_channels=12, channels=32, num_res_blocks=3,
                        attention_mode='none', arch='se_bottleneck')
    inner = [m for m in bb.blocks.modules() if isinstance(m, _BN)]
    assert inner, '这个主干里没有 BatchNorm，守卫无从谈起 —— 架构变了要改本用例'
    bn = inner[0]
    w = copy.deepcopy(bb.state_dict())
    x0 = torch.randn(2, 12, 19, 19)

    def run(gc):
        bb.load_state_dict(w)
        bb.grad_checkpoint = gc
        bb.train()
        x = x0.clone().requires_grad_(True)
        bb(x).sum().backward()
        return (bn.running_mean.detach().clone(),
                bn.running_var.detach().clone(),
                bn.num_batches_tracked.item())

    rm0, rv0, n0 = run(False)
    rm1, rv1, n1 = run(True)
    assert torch.equal(rm0, rm1), 'running_mean 被重算双累加了（守卫没生效）'
    assert torch.equal(rv0, rv1), 'running_var 被重算双累加了'
    assert n0 == n1 == 1, f'num_batches_tracked: 关={n0} 开={n1}（应都是 1）'


def test_bn_guard_is_skipped_when_there_is_no_batchnorm():
    """convnext 主干一个 BN 都没有 ⇒ 不挂守卫。

    这不是可有可无的优化：守卫靠 `context_fn` 挂，而 **`context_fn` 在
    `torch.compile` 下不被支持**（PyTorch 检测到就 assert）。所以「没有 BN 就
    不挂守卫」直接等于「convnext / V7 白拿 compile 兼容性」。

    钉住决策本身，免得日后有人「顺手」改成无条件挂守卫，把 compile 弄废。
    """
    torch.manual_seed(0)
    cnn = SharedBackbone(in_channels=12, channels=32, num_res_blocks=3,
                         attention_mode='none', arch='convnext')
    assert not [m for m in cnn.blocks.modules() if isinstance(m, _BN)]
    assert cnn._gc_bn_modules is None

    se = SharedBackbone(in_channels=12, channels=32, num_res_blocks=3,
                        attention_mode='none', arch='se_bottleneck')
    assert [m for m in se.blocks.modules() if isinstance(m, _BN)]
    assert se._gc_bn_modules is se.blocks


def test_v7_backbone_has_no_batchnorm_so_guard_costs_nothing():
    """V7 是现网真正在训的路径（段 A/B/C 都 `--v7 1`），它必须**没有** BN。

    若哪天给 V7 主干塞了 BatchNorm，这个用例会红 —— 那时 `modules=None` 就变成
    了漏守卫，必须改成 `modules=self.blocks`，并接受该路径放弃 compile 兼容。
    """
    net = _v7()
    assert not [m for m in net.modules() if isinstance(m, _BN)]


# --------------------------------------------------------------------------- #
# 与 torch.compile 共存 —— 选标准实现的**全部**理由
# --------------------------------------------------------------------------- #
def test_v7_gc_does_not_break_fullgraph_compile():
    """GC 开着时，V7 在 `fullgraph=True` 下必须能编译。

    `fullgraph=True` 遇到任何 graph break 就抛异常，所以这是「没有断点」的
    硬证明。旧 GC（自定义 `autograd.Function` + tuple 入参）过不了这一关 ——
    Dynamo 直接放弃，整段回退 eager、compile 白开。标准 `use_reentrant=False`
    可以，因为它被 Inductor 正常 trace。
    """
    net = _v7(grad_checkpoint=True)
    net.train()
    sp = torch.randn(2, 22, 19, 19, requires_grad=True)
    gl = torch.randn(2, 19)
    compiled = torch.compile(net, dynamic=False, fullgraph=True,
                             backend='aot_eager')
    out = compiled(sp, gl)
    out['policy_logits'].float().pow(2).mean().backward()
    assert sp.grad is not None


# --------------------------------------------------------------------------- #
# 接线：旗 → 构建器 → 模型
# --------------------------------------------------------------------------- #
def test_flag_reaches_both_networks():
    """`--grad-checkpoint 1` 必须真的透到两条建网路径的主干上。

    「旗存在但没接线」是这类开关最常见的失效形态：help 里写得清清楚楚，
    跑起来却一点没变。
    """
    m, cfg = t.build_katago_se_net(action_size=362, grad_checkpoint=True)
    assert m.backbone.grad_checkpoint is True
    assert cfg['grad_checkpoint'] is True

    m, cfg = t.build_katago_v7_net(board_size=19, grad_checkpoint=True)
    assert m.grad_checkpoint is True
    assert cfg['grad_checkpoint'] is True

    # 默认侧：两个构建器都必须是关的
    assert t.build_katago_se_net(action_size=362)[0].backbone.grad_checkpoint is False
    assert t.build_katago_v7_net(board_size=19)[0].grad_checkpoint is False


def test_both_build_call_sites_forward_the_flag():
    """`main()` 里两个建网调用点都必须把 `args.grad_checkpoint` 传下去。

    上一条测试**测不到这一段** —— 它直接调构建器，绕过了 `args`。而「旗解析对
    了却没接到模型上」恰恰是这类开关最常见的失效形态：help 写得很清楚，跑起来
    却一点没变。所以这里按 AST 定位调用点，钉住那条实参。
    """
    seen = {}
    for node in ast.walk(_main_ast()):
        if not isinstance(node, ast.Call):
            continue
        # 同模块内定义 ⇒ 调用是裸名 `build_xxx(...)`（ast.Name），不是
        # `module.build_xxx`（ast.Attribute）。两种都收，免得日后改成
        # `from scripts.train_sft import ...` 就静默漏掉。
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (
            fn.attr if isinstance(fn, ast.Attribute) else None)
        if name not in ('build_katago_se_net', 'build_katago_v7_net'):
            continue
        kws = {kw.arg: kw.value for kw in node.keywords}
        seen[name] = kws.get('grad_checkpoint')

    for builder in ('build_katago_se_net', 'build_katago_v7_net'):
        assert builder in seen, f'main() 里找不到 {builder} 调用点'
        expr = seen[builder]
        assert expr is not None, f'{builder} 没接 --grad-checkpoint（旗悬空）'
        # 实参必须是 `bool(args.grad_checkpoint)` 之类的、**引用了 args** 的表达式
        src = ast.unparse(expr)
        assert 'args.grad_checkpoint' in src, \
            f'{builder} 的 grad_checkpoint 实参是 {src!r}，没引用 args.grad_checkpoint'
    # 两个调用点都要有，不能只接一条路
    assert len(seen) == 2, sorted(seen)


def test_eval_path_under_gc_does_not_warn_or_recompute():
    """eval（`no_grad`）下 GC 必须是空操作，且**不能**刷警告。

    eval 每 N 步跑一次；每次都吐一遍 checkpoint 的 UserWarning，几天下来日志
    全是噪音，真出问题时反而看不见。这里也顺手确认 eval 不会误触发重算。
    """
    import warnings
    net = _v7(grad_checkpoint=True)
    net.eval()
    cnt = {'n': 0}
    h = net.blocks[0].register_forward_hook(
        lambda m, i, o: cnt.__setitem__('n', cnt['n'] + 1))
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with torch.no_grad():
                net(torch.randn(2, 22, 19, 19), torch.randn(2, 19))
    finally:
        h.remove()
    assert cnt['n'] == 1, f'eval 竟然重算了（块跑了 {cnt["n"]} 次）'
    noisy = [str(x.message) for x in w if 'checkpoint' in str(x.message).lower()]
    assert not noisy, f'eval 路径刷了 checkpoint 警告：{noisy}'


# --------------------------------------------------------------------------- #
# 结构不变
# --------------------------------------------------------------------------- #
def test_gc_changes_no_shapes_or_parameter_count():
    """GC 是**纯行为开关**：开/关的参数量与输出形状必须相同。

    少一个「形状没变」的断言，这个开关就很容易被后来的人当成结构参数塞进
    `NBT_TF_CFG` / `KATAGO_SE_CFG` 的结构键里（那边明写「结构只由这一张表
    决定」，行为参数不许进）。
    """
    torch.manual_seed(1)
    off = _v7(grad_checkpoint=False)
    torch.manual_seed(1)
    on = _v7(grad_checkpoint=True)
    assert sum(p.numel() for p in off.parameters()) == \
           sum(p.numel() for p in on.parameters())

    sp = torch.randn(1, 22, 19, 19)
    gl = torch.randn(1, 19)
    off.eval(), on.eval()
    o_off, o_on = off(sp, gl), on(sp, gl)
    assert sorted(o_off) == sorted(o_on)
    for k in o_off:
        assert o_off[k].shape == o_on[k].shape, k