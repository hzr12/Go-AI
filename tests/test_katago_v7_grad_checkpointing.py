r"""V7 的梯度检查点覆盖：blocks 之外（stem / 三个 head）也要能被重算。

为什么补这块
------------
`scripts/train_sft.py::v7_batch_memory_advice` 的解析模型给 V7 定的预算是
**checkpoint 开 ⇒ 8.1 MB/样本**（6.9 驻留 + 1.2 瞬时），而 910A 实测 batch 3000
用了 **64 GB ≈ 21.3 MB/样本** —— 高出 2.6 倍。`V7_RESIDENT_MB_PER_SAMPLE[True]`
的注释写明那 6.9 MB 是「**只存 block 边界**」，所以差额只可能落在 block 之外的
两处：`stem` 与三个 head（policy / value / scorebelief）。

原来它们走的是裸前向，一次都不重算，于是各自的中间激活在反向全程驻留。

为什么不是"包一层 HeadBank"
---------------------------
那会把 `state_dict` 的键从 `policy_head.*` 变成 `head_bank.policy_head.*`，
A/B/C 三段的权重立刻互相 load 不上（`tests/test_v7_single_model.py::
test_weights_carry_from_board_level_to_packed_stage`钉着这条）。所以检查点只能
在 `forward` 里**逐个内联**，顶层属性名一动不动。

粒度沿用 V18 旧机制
--------------------
`git b65c804`（"v21 检查点粒度改回 V18 旧机制（逐块）—— 段级是 4 卡 OOM 的直接
成因"）的实测：段级 `fwd+bwd 1306.6 MB`，逐块 `149.0 MB`，**耗时不增**。所以
这里一律逐块，且**不**照搬 legacy 的 `uncapped_last=True`（那个开关存在是为了保住
v18 的 31.12 GB 锚点，V7 没有那个锚点）。

数值必须**逐位**相同
--------------------
`katago_v7.py:22` 写明「全网无 BatchNorm」，归一化由块内 `fson` 仿射与末端
`RMSNormMask` 承担。所以重算没有任何有状态的算子，同输入重跑必须
`torch.equal` 为真 —— 这条比"误差很小"强得多，也是本文件大半断言的由来。
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.networks.katago_v7 import (  # noqa: E402
    NBT_TF_CFG, NbtTfNet, ScorebeliefHead, ValueHead)

#: 小形状只为让测试快；语义（无 BN / 逐块 / 三头并联）与真网一致。
SMALL = dict(NBT_TF_CFG)
SMALL.update(trunk_channels=16, nbt_mid=8, num_heads=2, ffn_hidden=24,
             num_blocks=2, policy_channels=8, gpool_channels=8,
             value_channels=8, value_hidden=16, seki_classes=4,
             futurepos_channels=2, extra_score_distr_radius=4,
             score_distr_components=4)

#: 三个 head 的顶层属性名。**不能改** —— 见模块 docstring 的 state_dict 那段。
HEAD_NAMES = ('policy_head', 'value_head', 'scorebelief_head')


def _net(ckpt):
    torch.manual_seed(0)
    net = NbtTfNet(cfg=dict(SMALL), use_checkpoint=ckpt)
    net.train()
    return net.initialize()


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    c = SMALL['board_size']
    return (torch.randn(2, SMALL['in_channels'], c, c, generator=g),
            torch.randn(2, SMALL['global_channels'], generator=g))


def _count(module):
    """数一个模块的 `forward` 被调了几次（含检查点重算的那次）。

    按**实例**计数：同类多实例共享类属性，只按类聚合会把所有头的调用记到第一个
    名字上（`b65c804` 修掉的正是这个仪器缺陷）。
    """
    n = [0]
    orig = module.forward

    def wrapper(*a, **kw):
        n[0] += 1
        return orig(*a, **kw)

    module.forward = wrapper
    return n


def _fwd_bwd(net):
    spatial, gl = _inputs()
    out = net(spatial, gl)
    sum(v.float().square().sum() for v in out.values()).backward()
    return out


def _snapshot(net):
    """跑一遍 fwd+bwd 并返回**脱离计算图**的输出。

    不能用 `copy.deepcopy`：它对带 `grad_fn` 的张量会直接抛。
    """
    out = _fwd_bwd(net)
    return {k: v.detach().clone() for k, v in out.items()}


# --------------------------------------------------------------------------- #
# 1. 逐位一致（这是本改动的**前提**：无 BN ⇒ 重算必须完全可复现）
# --------------------------------------------------------------------------- #
def test_checkpoint_on_and_off_are_bit_identical():
    """开/关检查点的输出与梯度必须逐位相同。"""
    on = _net(True)
    off = _net(False)
    off.load_state_dict(on.state_dict())

    out_on = _snapshot(on)
    out_off = _snapshot(off)

    assert set(out_on) == set(out_off)
    for k in sorted(out_on):
        assert torch.equal(out_on[k], out_off[k]), f'{k} 前向不逐位相同'

    for (n1, p1), (n2, p2) in zip(on.named_parameters(), off.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None, f'{n1} 开检查点时梯度为空'
        assert torch.equal(p1.grad, p2.grad), f'{n1} 梯度不逐位相同'


def test_dict_returning_head_survives_checkpoint():
    """`ValueHead` 返回 **dict**、`ScorebeliefHead` 吃两个上游张量 —— 逐个内联
    checkpoint 后两者都必须拿到**逐位相同**的结果。"""
    net = _net(True)
    spatial, gl = _inputs()
    t = net.trunk(spatial, gl)

    plain_v = net.value_head(t)
    ck = torch.utils.checkpoint.checkpoint(net.value_head, t, use_reentrant=False)
    assert set(plain_v) == set(ck)
    for k in sorted(plain_v):
        assert torch.equal(plain_v[k], ck[k]), f'value_head.{k} 不逐位相同'

    plain_s = net.scorebelief_head(plain_v['value_pooled'], gl)
    ck_s = torch.utils.checkpoint.checkpoint(net.scorebelief_head,
                                             ck['value_pooled'], gl,
                                             use_reentrant=False)
    assert torch.equal(plain_s, ck_s), 'scorebelief_head 不逐位相同'


# --------------------------------------------------------------------------- #
# 2. 覆盖：stem 与三个 head 真的被重算了
# --------------------------------------------------------------------------- #
def test_each_head_is_recomputed_in_backward():
    """开检查点 ⇒ 每个 head 前向 2 次（1 次前向 + 1 次重算）。"""
    net = _net(True)
    for name in HEAD_NAMES:
        head = getattr(net, name)
        n = _count(head)
        _fwd_bwd(net)
        assert n[0] == 2, '%s 只被调 %d 次（期望 2 = 前向 + 重算）' % (name, n[0])
        del head.forward


def test_stem_is_recomputed_in_backward():
    net = _net(True)
    n = _count(net.stem)
    _fwd_bwd(net)
    assert n[0] == 2, 'stem 只被调 %d 次（期望 2）' % n[0]


def test_blocks_stay_per_block_not_per_segment():
    """粒度必须仍是**逐块**（`b65c804` 的裁决），不能并段。"""
    net = _net(True)
    for i, blk in enumerate(net.blocks):
        n = _count(blk)
        _fwd_bwd(net)
        assert n[0] == 2, 'blocks[%d] 被调 %d 次（逐块应为 2）' % (i, n[0])
        del blk.forward


def test_nothing_is_recomputed_when_switch_is_off():
    net = _net(False)
    watched = [net.stem] + [getattr(net, n) for n in HEAD_NAMES] + list(net.blocks)
    counts = [_count(m) for m in watched]
    _fwd_bwd(net)
    for m, n in zip(watched, counts):
        assert n[0] == 1, '%s 被调 %d 次（关检查点时应为 1）' % (
            type(m).__name__, n[0])


# --------------------------------------------------------------------------- #
# 3. eval / no_grad 下零行为变化
# --------------------------------------------------------------------------- #
def test_eval_does_not_recompute_anything():
    """`grad_checkpointing_for` 在 eval下一律 False ⇒ 推理路径逐位不变。"""
    net = _net(True)
    net.eval()
    watched = [net.stem] + [getattr(net, n) for n in HEAD_NAMES] + list(net.blocks)
    counts = [_count(m) for m in watched]
    spatial, gl = _inputs()
    with torch.no_grad():
        net(spatial, gl)
    for m, n in zip(watched, counts):
        assert n[0] == 1, '%s 在 eval 下被调 %d 次' % (type(m).__name__, n[0])


def test_no_grad_does_not_recompute_even_in_train_mode():
    net = _net(True)
    n = _count(net.policy_head)
    spatial, gl = _inputs()
    with torch.no_grad():
        net(spatial, gl)
    assert n[0] == 1, 'no_grad 下 policy_head 被调 %d 次' % n[0]


# --------------------------------------------------------------------------- #
# 4. state_dict 键名是硬约束
# --------------------------------------------------------------------------- #
def test_state_dict_keys_are_unchanged_by_the_checkpoint_wiring():
    """加检查点**不许**动 `state_dict` 的键 —— A/B/C 三段权重要互相 load。"""
    net = _net(True)
    keys = set(net.state_dict())
    for name in HEAD_NAMES:
        assert any(k.startswith(name + '.') for k in keys), \
            '顶层属性 %s 不见了（state_dict 键会变）' % name
    assert any(k.startswith('stem.') for k in keys)
    assert any(k.startswith('blocks.0.') for k in keys)


def test_switching_checkpoint_does_not_change_state_dict():
    """开关是普通实例属性、不进 `state_dict`，切换前后存档逐位相同。"""
    on = _net(True)
    before = {k: v.clone() for k, v in on.state_dict().items()}
    on.use_checkpoint = False
    on.use_checkpoint = True
    after = on.state_dict()
    assert set(before) == set(after)
    for k, v in before.items():
        assert torch.equal(v, after[k]), k


def test_heads_recompute_under_the_same_autocast_as_the_forward():
    """重算时的精度必须与前向一致 —— 否则 fp16 训练里 head 会退回 fp32。

    ``backbone._checkpointed`` 给每段都挂了 ``context_fn``，其中
    ``_recompute_ctx`` 会按**段入口张量的 dtype** 恢复 autocast。这不是优化，是
    正确性：``backbone.py:463-467`` 记着 4 卡 910A 的 OOM 真因就是「重算不在
    autocast 里 ⇒ 整段退回 fp32、体积翻倍」（`Tried to allocate 1.40 GiB` 恰是
    ``(1000,4,46,32,64)`` 的 fp32 体积）。

    本条在 CPU 上也抓得住：``autocast(cpu, bfloat16)`` 是真生效的，而
    ``with`` 块退出后 ``is_autocast_enabled()`` 立刻变回 False —— 所以只要
    重算没有恢复它，第二次调用就必然被看见。
    """
    net = _net(True)
    seen = []
    orig = net.value_head.forward

    def spy(*a, **kw):
        seen.append(bool(torch.is_autocast_enabled('cpu')))
        return orig(*a, **kw)

    net.value_head.forward = spy

    spatial, gl = _inputs()
    # **backward 必须在 autocast 块之外** —— 这才是真实训练的形状
    #（`with autocast: loss = model(x)` 然后块外 `loss.backward()`）。
    # 把它写在块里会让重算"恰好"看到 autocast 开着，测试就变成永远绿。
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        out = net(spatial, gl)
    sum(v.float().square().sum() for v in out.values()).backward()

    assert len(seen) == 2, 'value_head 应被调 2 次，实得 %d' % len(seen)
    assert seen[0] is True, '前向那次不在 autocast 里，测试本身失效'
    assert seen[1] is True, (
        '重算那次**不在** autocast 里 ⇒ fp16 训练时整个 value 头会退回 fp32，'
        '体积翻倍（这正是 4 卡 910A OOM 的机制，backbone.py:463-467）')


def test_heads_recompute_is_safe_when_a_probe_needs_no_sync():
    """`GC_HEADS` 段的重算不得引入主机侧同步（BN 守卫为空时开销为 0）。"""
    net = _net(True)
    src = _fn_code_of_backbone('_checkpointed')
    assert 'context_fn' in src, '段级重算必须挂 context_fn（autocast + BN 守卫）'
    assert 'preserve_rng_state=True' in src


def _fn_code_of_backbone(name):
    import re as _re
    bp = os.path.join(ROOT, 'src', 'networks', 'backbone.py')
    with open(bp, encoding='utf-8') as fh:
        s = fh.read()
    i = s.index('def %s(' % name)
    j = s.index('\ndef ', i + 1)
    out = []
    for ln in s[i:j].splitlines():
        if ln.lstrip().startswith('#'):
            continue
        out.append(ln)
    return _re.sub(r'(?s)("""|\'\'\').*?\1', '', '\n'.join(out))


# --------------------------------------------------------------------------- #
# 5. 逐 kind 开关（若 V7 接了 mixin）
# --------------------------------------------------------------------------- #
def test_kinds_include_stem_and_heads():
    """V7 要有 `stem` / `heads` 两个 kind，blocks 归 `res`。"""
    net = _net(True)
    kinds = net.grad_checkpointing_kinds()
    assert 'stem' in kinds, '缺 stem 这个 kind（现有 kinds=%s）' % sorted(kinds)
    assert 'heads' in kinds, '缺 heads 这个 kind（现有 kinds=%s）' % sorted(kinds)
    assert 'res' in kinds, 'blocks 段应复用 res 这个 kind'


def test_heads_switch_is_independent_of_blocks():
    """关掉 heads 不该顺带关掉 blocks（逐 kind 独立）。"""
    net = _net(True)
    net.set_grad_checkpointing(True, heads=False)
    nb = _count(net.blocks[0])
    nh = _count(net.value_head)
    _fwd_bwd(net)
    assert nb[0] == 2, 'blocks 段被连带关掉了'
    assert nh[0] == 1, 'value_head 在 heads=False 下仍被调 %d 次' % nh[0]


def test_no_grad_makes_grad_checkpointing_for_false():
    """`grad_checkpointing_for` 是判定口径本身，必须在 no_grad 下为假。"""
    net = _net(True)
    assert net.grad_checkpointing_for('heads') is True
    with torch.no_grad():
        assert net.grad_checkpointing_for('heads') is False
    net.eval()
    assert net.grad_checkpointing_for('heads') is False


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_compile_guard_rejects_compiled_submodule():
    """compile 与检查点互斥：子模块被 `torch.compile` 包过就要就地报错。"""
    from src.networks.backbone import assert_grad_checkpoint_compile_compatible

    net = _net(True)
    assert_grad_checkpoint_compile_compatible(net)      # 没编译 ⇒ 不抛

    # `torch.compile` 内部会碰 `torch.jit.script_method`，那是 torch 自己的
    # 弃用警告、与本测试无关 —— 上面那条 filterwarnings 就是为它加的。
    net.blocks[0] = torch.compile(net.blocks[0])
    with pytest.raises(RuntimeError, match='互斥'):
        assert_grad_checkpoint_compile_compatible(net)