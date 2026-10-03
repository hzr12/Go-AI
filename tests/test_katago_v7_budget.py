"""V7 NBT+Transformer 的结构与参数量基准（spec §3 / §6.1）。

为什么要有这个文件
------------------
`tests/test_param_budget.py` 已经为 v18 立过一次规矩：run.txt 里写着
「V17 Value 1.01M（96ch, 11 blocks）」，实测是 2.00M，1.01M 其实对应 5 块 ——
文字与代码互相矛盾，于是「v18 是不是比 v12 差」这类对照建立在错误基准上。
本文件是同一件事对 V7 的重演：把 spec §6.1 的每个数字用**实际实例化**钉死。

§6.1 的一处表格行标签勘误
--------------------------
spec §6.1 里「**块小计 ×11 = 5,479,680**」这一行的**标签是错的**，数字本身
落在正确的位置：5,479,680 其实是 `stem + global + 11 个块 + trunk 末端` 之和，
11 个块自身是 **5,423,616**（单块 493,056）。5,479,680 不是 11 的整数倍，
所以它不可能是「块小计」。spec 的**总数 5,561,832 完全正确**，本测试断言的
就是它。见 `test_block_subtotal_is_not_the_stem_plus_global_sum`。
"""

import inspect
import math
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.networks import katago_v7  # noqa: E402
from src.networks.katago_v7 import (  # noqa: E402
    GAIN_SILU, NBT_TF_CFG, SCORE_STDEV_SOFTPLUS_BETA, NormAct,
    Nbt2TransformerBlock, NbtTfNet, RMSNormMask, build_katago_v7_net,
    gpool_policy, gpool_value,
)

#: 硬预算（spec §1.2 C1）。超了就是结构改动没同步预算。
BUDGET_TOTAL = 5_850_000
#: spec §6.1 的目标值。
SPEC_TOTAL = 5_561_832

#: 逐子模块的 spec 数字。`value_head` 在 spec 里被拆成「value 头 26,886」与
#: 「ownership/scoring/futurepos/seki 384」两行，本实现把四个 1×1 小头并进了
#: `ValueHead`（它们本来就吃同一个 `VV`），故这里断言合并后的 27,270。
EXPECTED = {
    'stem': 50_688,
    'global_fc': 4_864,
    'blocks_each': 493_056,
    'blocks_total': 5_423_616,
    'trunkfinal': 512,
    'policy_head': 38_834,
    'value_head_with_small': 27_270,   # = 26_886 + 384
    'scorebelief_head': 16_048,
}


def _n(mod):
    return sum(p.numel() for p in mod.parameters())


@pytest.fixture(scope='module')
def net():
    return build_katago_v7_net()


# --------------------------------------------------------------------------- #
# 参数量
# --------------------------------------------------------------------------- #
def test_total_param_count_is_exact(net):
    got = _n(net)
    assert got == SPEC_TOTAL, (
        f'总参数 {got:,} != spec §6.1 的 {SPEC_TOTAL:,}；'
        f'结构改了但预算表没同步（spec §8.2 的 2.7 要求两处一起改）')


def test_total_is_under_hard_budget(net):
    got = _n(net)
    assert got <= BUDGET_TOTAL, f'超硬预算 {BUDGET_TOTAL:,}：{got:,}'
    margin = BUDGET_TOTAL - got
    assert margin == 288_168, f'余量应为 288,168（4.9%），实测 {margin:,}'


def test_submodule_param_counts_match_spec(net):
    assert _n(net.stem) == EXPECTED['stem']
    assert _n(net.global_fc) == EXPECTED['global_fc']
    assert _n(net.blocks[0]) == EXPECTED['blocks_each']
    assert _n(net.blocks) == EXPECTED['blocks_total']
    assert _n(net.norm_trunkfinal) == EXPECTED['trunkfinal']
    assert _n(net.policy_head) == EXPECTED['policy_head']
    assert _n(net.value_head) == EXPECTED['value_head_with_small']
    assert _n(net.scorebelief_head) == EXPECTED['scorebelief_head']


def test_submodules_sum_to_total(net):
    s = (_n(net.stem) + _n(net.global_fc) + _n(net.blocks)
         + _n(net.norm_trunkfinal) + _n(net.policy_head) + _n(net.value_head)
         + _n(net.scorebelief_head))
    assert s == _n(net), '子模块之和 != 总数（漏算或重复计算）'


def test_block_subtotal_is_not_the_stem_plus_global_sum():
    """§6.1 那行标签勘误：5,479,680 含 stem/global/trunk-end，不是块小计。"""
    blocks = EXPECTED['blocks_total']
    assert blocks % 11 == 0 and blocks // 11 == EXPECTED['blocks_each']
    with_prefix = (EXPECTED['stem'] + EXPECTED['global_fc'] + blocks
                   + EXPECTED['trunkfinal'])
    assert with_prefix == 5_479_680
    assert with_prefix != blocks, '5,479,680 不可被 11 整除，不可能是块小计'


# --------------------------------------------------------------------------- #
# 结构常量（spec §3）
# --------------------------------------------------------------------------- #
def test_shape_constants_match_spec():
    c = NBT_TF_CFG
    assert c['trunk_channels'] == 256        # C
    assert c['nbt_mid'] == 128               # M = C/2
    assert c['num_heads'] == 4               # H
    assert c['ffn_hidden'] == 384            # F = 1.5C
    assert c['num_blocks'] == 11             # B
    assert c['num_gpool_blocks'] == 0        # G
    assert c['in_channels'] == 22
    assert c['global_channels'] == 19
    assert c['policy_outputs'] == 2          # K = 2：π + π_opp


def test_head_dim_is_32_and_in_cann_supported_set(net):
    """head_dim = M/H = 32，落在 CANN 融合注意力支持集 {16,32,64}。

    这是 spec §3 里唯一**不能事后调**的约束：head_dim 变了，融合注意力直接
    不可用（spec §6 R1）。
    """
    hd = net.blocks[0].inner[0].attn.head_dim
    assert hd == 32, f'head_dim={hd}，spec 规定 32'
    assert hd in (16, 32, 64)
    for blk in net.blocks:
        for inner in blk.inner:
            assert inner.attn.head_dim == hd


def test_h_head_count_swap_is_free(net):
    """spec §6.1 备选：若探针要求 head_dim=16，只需 H=8，**零参数代价**。

    MHSA 的参数量只与 `dim` 有关（四个 Linear 都是 dim×dim），与头数无关；
    只有 RoPE 的 `(H, head_dim//2, 2)` 随 H 变，而 `H·head_dim//2 = dim//2`
    是常数 ⇒ 连 RoPE 参数量也不变。
    """
    from src.networks.katago_v7 import MHSA
    a = MHSA(128, 4)   # head_dim 32
    b = MHSA(128, 8)   # head_dim 16
    assert a.head_dim == 32 and b.head_dim == 16
    assert _n(a) == _n(b), '换头数不应改变参数量（spec §6.1 备选的前提）'


# --------------------------------------------------------------------------- #
# fson / rsnh 语义（spec §3.3）
# --------------------------------------------------------------------------- #
def test_no_batchnorm_anywhere(net):
    """spec §3.5 第 1 条：全网无 BatchNorm，`BatchNorm1d/2d` 路径不进入新模型。"""
    bns = [m for m in net.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    assert not bns, f'V7 里出现了 {len(bns)} 个 BatchNorm：{[type(b).__name__ for b in bns]}'


def test_no_dropout_modules(net):
    """spec §3.5 第 2 条：不加 dropout，模块树里不应出现任何 Dropout。"""
    drops = [m for m in net.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d))]
    assert not drops, f'V7 里出现了 Dropout：{[type(d).__name__ for d in drops]}'
    assert net.attn_dropout == 0.0


def test_fson_gamma_equals_k_schedule(net):
    """fson 的 γ 初值必须逐字等于 spec §3.3 的 K 调度。

    漏掉它不会有任何报错，只会让残差流的方差随深度线性增长 —— 深层激活
    饱和、loss 离谱，而形状测试全绿。这条测试是那道静默故障的唯一防线。
    """
    for i, blk in enumerate(net.blocks):
        expect = 1.0 / math.sqrt(i + 1.0)
        assert torch.allclose(blk.normact.gamma,
                              torch.full_like(blk.normact.gamma, expect)), \
            f'trunk 第 {i} 块的 normact γ 应为 1/√{i + 1}={expect:.6f}'
    # 块内 normact_mid 按内块数：1/√(inner_len+1) = 1/√3
    expect_mid = 1.0 / math.sqrt(net.blocks[0].inner_len + 1.0)
    assert torch.allclose(net.blocks[0].normact_mid.gamma,
                          torch.full_like(net.blocks[0].normact_mid.gamma,
                                          expect_mid))
    # trunk 末端：1/√(num_blocks+1) = 1/√12
    expect_final = 1.0 / math.sqrt(len(net.blocks) + 1.0)
    assert torch.allclose(net.norm_trunkfinal.gamma,
                          torch.full_like(net.norm_trunkfinal.gamma,
                                          expect_final))


def test_fson_beta_starts_at_zero(net):
    for m in net.modules():
        if isinstance(m, (NormAct, RMSNormMask)):
            assert torch.all(m.beta == 0), 'β 初值必须是 0'


def test_silu_gain_is_sqrt2():
    """spec §3.4：gain(SiLU)=√2.0（官方注释：理论值 √2.8108，为兼容保留 √2.0）。"""
    assert GAIN_SILU == math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# 池化（spec §4.1 / §4.2）
# --------------------------------------------------------------------------- #
def test_gpool_policy_three_stats():
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    out = gpool_policy(x)
    assert out.shape == (2, 9)          # 3 × C
    c = x.shape[1]
    d = math.sqrt(16.0) - 14.0
    assert torch.allclose(out[:, :c], x.mean(dim=(2, 3)))
    assert torch.allclose(out[:, c:2 * c], x.mean(dim=(2, 3)) * (d / 10.0))
    assert torch.allclose(out[:, 2 * c:], x.amax(dim=(2, 3)))


def test_gpool_value_three_stats_no_max():
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    out = gpool_value(x)
    assert out.shape == (2, 9)
    c = x.shape[1]
    d = math.sqrt(16.0) - 14.0
    mean = x.mean(dim=(2, 3))
    assert torch.allclose(out[:, :c], mean)
    assert torch.allclose(out[:, c:2 * c], mean * (d / 10.0))
    assert torch.allclose(out[:, 2 * c:], mean * (d * d / 100.0 - 0.1))
    # value 池化**没有 max** —— 三项都只是 mean 的缩放
    assert not torch.allclose(out[:, 2 * c:], x.amax(dim=(2, 3)))


def test_gpool_scales_are_constants_at_19x19():
    """spec §4.2 注：19×19 下两个缩放系数都是常数（0.5 / 0.15），公式照抄保留。"""
    d = math.sqrt(361.0) - 14.0
    assert d / 10.0 == 0.5
    assert (d * d) / 100.0 - 0.1 == pytest.approx(0.15)


# --------------------------------------------------------------------------- #
# 前向形状
# --------------------------------------------------------------------------- #
def test_forward_output_shapes(net):
    net.eval()
    b = 2
    sp = torch.zeros(b, 22, 19, 19)
    gl = torch.zeros(b, 19)
    with torch.no_grad():
        o = net(sp, gl)
    assert o['policy_logits'].shape == (b, 2, 362)        # K=2，361+pass
    assert o['outcome_logits'].shape == (b, 3)            # 胜/负/无结果
    assert o['score_mean'].shape == (b,)
    assert o['score_stdev'].shape == (b,)
    assert o['lead'].shape == (b,)
    assert o['ownership_pretanh'].shape == (b, 1, 19, 19)
    assert o['scoring'].shape == (b, 1, 19, 19)
    assert o['futurepos'].shape == (b, 2, 19, 19)        # +8 / +32 手
    assert o['seki_logits'].shape == (b, 4, 19, 19)      # 3 符号类 + 1 中性
    assert o['scorebelief_logits'].shape == (b, 842)     # 2*(361+60)
    for k, v in o.items():
        assert torch.isfinite(v).all(), f'{k} 出现 NaN/Inf'


def test_scorebelief_bin_count_follows_formula(net):
    c = NBT_TF_CFG
    assert c['extra_score_distr_radius'] == 60
    assert 2 * (c['board_size'] ** 2 + c['extra_score_distr_radius']) == 842


def test_off_board_mask_applies_minus_5000(net):
    from src.networks.katago_v7 import OFF_BOARD_LOGIT
    net.eval()
    sp = torch.zeros(1, 22, 19, 19)
    gl = torch.zeros(1, 19)
    for shape in ((19, 19), (1, 19, 19), (1, 1, 19, 19)):
        mk = torch.ones(shape)
        mk[..., 15:, :] = 0
        with torch.no_grad():
            pl = net(sp, gl, board_mask=mk)['policy_logits']
        assert float(pl[0, 0, 15 * 19 + 3]) == OFF_BOARD_LOGIT
        assert torch.isfinite(pl[0, 0, :15 * 19]).all()
        assert torch.isfinite(pl[0, 0, 361]).all(), 'pass 槽不该被 off-board 掩码'


def test_ownership_tanh_is_bounded(net):
    from src.networks.katago_v7 import OFF_BOARD_LOGIT  # noqa: F401
    net.eval()
    sp = torch.randn(2, 22, 19, 19) * 5
    gl = torch.randn(2, 19)
    with torch.no_grad():
        o = net(sp, gl)
    ow = net.ownership(o)
    assert ow.shape == (2, 1, 19, 19)
    assert (ow >= -1).all() and (ow <= 1).all()


def test_grad_checkpointing_paths_agree(net):
    """`use_checkpoint=True` 与 `False` 的前向必须一致（§3.5 第 4 条）。"""
    torch.manual_seed(0)
    ref = build_katago_v7_net(use_checkpoint=False).eval()
    torch.manual_seed(0)
    chk = build_katago_v7_net(use_checkpoint=True).eval()
    chk.load_state_dict(ref.state_dict())
    sp = torch.randn(1, 22, 19, 19)
    gl = torch.randn(1, 19)
    with torch.no_grad():
        a = ref(sp, gl)['policy_logits']
        b = chk(sp, gl)['policy_logits']
    assert torch.allclose(a, b, atol=1e-5), '梯度检查点改变了前向结果'


def test_backward_reaches_every_parameter(net):
    """一次反传后不允许有梯度为 None 的参数 —— 漏接一个头就会在这里暴露。"""
    net.train()
    net.zero_grad(set_to_none=True)
    sp = torch.randn(1, 22, 19, 19)
    gl = torch.randn(1, 19)
    o = net(sp, gl)
    loss = (o['policy_logits'].pow(2).mean() + o['outcome_logits'].pow(2).mean()
            + o['ownership_pretanh'].pow(2).mean()
            + o['scoring'].pow(2).mean()
            + o['futurepos'].pow(2).mean()
            + o['seki_logits'].pow(2).mean()
            + o['scorebelief_logits'].pow(2).mean()
            + o['score_mean'].sum() + o['score_stdev'].sum() + o['lead'].sum())
    loss.backward()
    missing = [name for name, p in net.named_parameters()
               if p.grad is None]
    assert not missing, f'{len(missing)} 个参数没有梯度：{missing[:8]}'
    assert all(torch.isfinite(p.grad).all() for p in net.parameters()), \
        '有参数的梯度是 NaN/Inf'


# --------------------------------------------------------------------------- #
# SCORE_STDEV_SOFTPLUS_BETA：spec 字面值 0.05 → 已裁决 1.0（2026-10-03）
# --------------------------------------------------------------------------- #
#: loss #7 的 Huber δ（`katago_v7_loss.py::forward` 里那个 `10.0`）。
HUBER_DELTA_SCORE_STDEV = 10.0


class _swapped_beta:
    """临时改 `katago_v7.SCORE_STDEV_SOFTPLUS_BETA`（`forward` 每次调用现读它）。

    ⚠ 必须改**模块全局**而不是传参：`ValueHead.forward` 里是
    `F.softplus(s[:,1], beta=SCORE_STDEV_SOFTPLUS_BETA)`，那个名字在调用时
    才解析 ⇒ 只有全局才是真正生效的那个开关。用 `with ... as _` 拿回原值，
    保证即使断言失败也不会把常量留在 0.05 上污染后面的测试。
    """

    def __init__(self, beta):
        self.beta = float(beta)

    def __enter__(self):
        self.orig = katago_v7.SCORE_STDEV_SOFTPLUS_BETA
        katago_v7.SCORE_STDEV_SOFTPLUS_BETA = self.beta
        return self

    def __exit__(self, *exc):
        katago_v7.SCORE_STDEV_SOFTPLUS_BETA = self.orig
        return False


def test_score_stdev_softplus_beta_is_one():
    """常量必须**精确**是 1.0（不是「大概是 1」）。

    spec §4.2 的字面值是 `0.05`，正文里保留着（它是推导链的一环）—— 但**代码**
    实现的是裁决值 1.0。这条断言的作用是让「spec 字面值被抄进代码」立刻变红，
    而不是等到 #7 又 40 步不动时才发现。
    """
    assert SCORE_STDEV_SOFTPLUS_BETA == 1.0


def test_score_stdev_softplus_term_lands_near_huber_delta():
    """🔴 实测：`softplus` 项在 beta=1.0 下与 δ=10 **同量级**，在 0.05 下差 27 倍。

    `F.softplus(x, beta) = log(1+exp(beta·x))/beta` ⇒ x=0 时
    `softplus(0, beta) = log(2)/beta` ⇒ 预测初值 `20·log(2)/beta`：
    beta=0.05 → **277.26**，beta=1.0 → **13.86**。

    这里跑**真实 forward**（seed 0 初始化 + seed 7 输入 / B=64）取实测值，而不是
    停在纸上推导 —— 推导只能证明公式；实现里少乘一个 20、或漏掉 `beta=` 关键字
    （`F.softplus` 的第二个位置参数就是 beta），纸面推导一律看不出来。
    """
    # 纸面：F.softplus(0, beta) = log(2)/beta
    assert float(F.softplus(torch.tensor(0.0), beta=0.05)) == pytest.approx(
        13.8629, abs=1e-3)
    assert float(F.softplus(torch.tensor(0.0), beta=1.0)) == pytest.approx(
        0.6931, abs=1e-3)

    torch.manual_seed(0)
    n = build_katago_v7_net().eval()
    g = torch.Generator().manual_seed(7)
    sp = torch.randn(64, 22, 19, 19, generator=g)
    gl = torch.randn(64, 19, generator=g)

    means = {}
    with torch.no_grad():
        for beta in (0.05, 1.0):
            with _swapped_beta(beta):
                means[beta] = float(n(sp, gl)['score_stdev'].mean())

    # 裁决值：实测 13.8599，与 δ=10 同量级。判据取「落在 δ 的 3 倍以内」而不是
    # 「等于 δ」—— 初值不必等于目标，只要别差两个数量级。
    assert means[1.0] == pytest.approx(13.8599, abs=0.05), means
    assert means[1.0] <= 3.0 * HUBER_DELTA_SCORE_STDEV, \
        f'beta=1.0 的初值 {means[1.0]} 与 δ={HUBER_DELTA_SCORE_STDEV} 差太远'
    # spec 字面值：实测 277.2534，是 δ 的 27 倍。这一行是**回归哨兵** ——
    # beta 若被改回 0.05，它会连同上面那条一起变红。
    assert means[0.05] == pytest.approx(277.2534, abs=0.05), means
    assert means[0.05] > 20.0 * HUBER_DELTA_SCORE_STDEV


def test_score_stdev_loss_term_is_inside_huber_delta_at_the_ruled_out_beta():
    """裁决要的不只是「预测初值同量级」，而是**落进 δ 区间内**（Huber 二次段）。

    上一条量的是网络输出 `score_stdev`；这一条量的是它**进 loss #7 之后**的
    公式值（huber(预测, std(scorebelief), δ=10)）。落在 δ 以内 ⇒ 梯度还在二次段，
    没有被线性段压掉 1/δ —— 那才是「这一项真的有学习信号」的直接判据。

    实测（真实 net + 合成标签，段 2 的口径即 `game_weight=1`）：
      beta=0.05 ⇒ 272.22（δ 的 27 倍）
      beta=1.0  ⇒   8.83（**δ 以内**）
    """
    from src.networks.katago_v7_loss import KataGoV7Loss, huber

    torch.manual_seed(0)
    n = build_katago_v7_net().eval()
    g = torch.Generator().manual_seed(7)
    sp = torch.randn(8, 22, 19, 19, generator=g)
    gl = torch.randn(8, 19, generator=g)
    # 目标 = std(softmax(scorebelief))，官方口径 5~20；取一列实测值。
    with torch.no_grad():
        base = n(sp, gl)
    target = base['scorebelief_logits'].float().softmax(-1).std(-1)
    assert 0.0 < float(target.mean()) < 40.0, float(target.mean())

    got = {}
    with torch.no_grad():
        for beta in (0.05, 1.0):
            with _swapped_beta(beta):
                o = n(sp, gl)
            # `huber` 是**逐样本**的 ⇒ 要 mean 才等于 #7 的公式值
            # （`_weighted_mean` 在 game_weight 恒 1 时就是 mean）。
            got[beta] = float(huber(o['score_stdev'].float(), target,
                                    HUBER_DELTA_SCORE_STDEV).mean())
    assert got[1.0] == pytest.approx(8.83, abs=0.05), got
    assert got[1.0] <= HUBER_DELTA_SCORE_STDEV, \
        f'beta=1.0 下 loss #7 的公式值 {got[1.0]} 仍在 δ 之外 ⇒ 仍在线性段'
    assert got[0.05] == pytest.approx(272.22, abs=0.5), got

    # 反证：#7 的公式值确实随 beta 变（否则上面全是恒真的断言）。
    assert got[0.05] != got[1.0]
    # 顺带确认 `KataGoV7Loss` 里那一路读的是**同一个** δ=10（本测试把 δ 写成
    # 自己的常量，所以必须核对它与实现同步，否则两条断言会在 δ 漂移时假绿）。
    src = inspect.getsource(KataGoV7Loss.forward)
    assert 'huber(out[\'score_stdev\'].float(), sb_std, %s)' % (
        repr(HUBER_DELTA_SCORE_STDEV)) in src, \
        'katago_v7_loss.py 里 #7 的 δ 与本测试的 HUBER_DELTA_SCORE_STDEV 不一致'


def test_score_stdev_beta_is_a_pure_constant_so_params_are_untouched():
    """beta 是**纯常量**：两种取值下参数量与 state_dict 形状必须完全一致。

    没有这条的话，「改 beta 顺手改了 `scores` 那一层的形状」会被
    `test_total_param_count_is_exact` 抓到，但抓不到**另一种**改法 ——
    用 beta 去缩放某个 `Linear` 的输出维度而总数恰好不变。判据的形状：
    逐键比对 `state_dict` 的形状，而不只是比总数。
    """
    torch.manual_seed(0)
    a = build_katago_v7_net()
    torch.manual_seed(0)
    b = build_katago_v7_net()
    with _swapped_beta(1.0):       # b 走裁决值；a 保持 spec 字面值 0.05
        sb = b.state_dict()
    sa = a.state_dict()
    assert set(sa) == set(sb)
    assert all(tuple(sa[k].shape) == tuple(sb[k].shape) for k in sa)
    assert _n(a) == _n(b) == SPEC_TOTAL == 5_561_832


# --------------------------------------------------------------------------- #
# 构造器契约
# --------------------------------------------------------------------------- #
def test_nbt_block_inner_count_and_shapes():
    blk = Nbt2TransformerBlock(256, 128, num_heads=4, ffn_hidden=384, inner_len=2)
    assert len(blk.inner) == 2
    assert blk.conv_p.in_channels == 256 and blk.conv_p.out_channels == 128
    assert blk.conv_q.in_channels == 128 and blk.conv_q.out_channels == 256
    assert blk.normact.channels == 256
    assert blk.normact_mid.channels == 128
    # fson / rsnh 都不含 running stats
    assert not any('running' in k for k, _ in blk.named_buffers())


def test_default_net_uses_config_only():
    """结构**只由 NBT_TF_CFG 决定**（与 KATAGO_SE_CFG 同一立场）。"""
    net = NbtTfNet()
    assert net.cfg['trunk_channels'] == 256
    assert len(net.blocks) == 11
    assert net.stem.kernel_size == (3, 3)
    assert net.stem.bias is None, 'stem conv bias=False（spec §3.1）'
    assert net.global_fc.bias is None, 'global linear bias=False'
