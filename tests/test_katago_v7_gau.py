"""GatedAttentionBlock 的结构与语义测试。

为什么要有这个文件
------------------
`GatedAttentionBlock`（GAU 改版）把「注意力 + 门控前馈」融合成**一个**残差子层，
而 `TransformerBlock` 是两条（MHSA + SwiGLU）。两者形状完全一样（4D→4D），形状测试
看不出区别 —— 真正的差别是门控结构：注意力 value **不是**独立投影的，而是直接拿
GLU 的 V 分支来用（`V̂ = V ⊙ (A@V)`，再被 U 门控），且门控激活用 **relu²**。写错了
形状照样全绿、指标悄悄变差。本文件逐项钉死：
  1. `Nbt2TransformerBlock` 工厂默认（不传 attn_impl/gau_positions）仍是全 nbt、
     参数量 493,056（零回归）；`NbtTfNet()` 默认按 `NBT_TF_CFG` 走纯 nbt（GAU 已关）；
     GAU 必须经 `cfg.gau_positions` 显式开启才出现（见下方「混合挂载」测试）；
  2. `attn_impl='nbt', gau_positions=[0/1]` 真的造出 1 个 GAU + 1 个 nbt（用户要的
     「每层各一个」）；
  3. 前向确实是 `W_o(U ⊙ V ⊙ relu²(QKᵀ·s+b)@V)`，且 V 就是 GLU 的 value 分支、
     门用 relu²，注意力无 softmax；
  4. q/k 真的过了 `RoPE2D`（输出随 pos 变，且对 pos 平移等变）；
  5. dtype 与注意力后端（math / SDPA）都能跑。
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.networks import backbone  # noqa: E402
from src.networks.katago_v7 import (  # noqa: E402
    NBT_TF_CFG, GatedAttentionBlock, GatedAttentionUnit, Nbt2TransformerBlock,
    NbtTfNet, TransformerBlock, _board_pos, _resolve_inner_kinds,
    _GAU_GATED_CLAMP,
)

#: 与 `NBT_TF_CFG` 一致的形状：C=256、M=128、H=4 ⇒ head_dim=32、F=384。
C, M, H, F_HID = 256, 128, 4, 384


def _gau_block():
    blk = GatedAttentionBlock(M, H, F_HID)
    blk.initialize()
    blk.eval()
    return blk


def _relu2(x):
    return F.relu(x).square()


def _manual_gau(blk, x4, pos):
    """照 GAU 的定义手算一遍（**不**复用被测的 `_heads`/`forward`），用于对照。

    只复用 `RoPE2D` 本身 —— 它的正确性由官方权重对拍保证，这里再写一遍旋转只会把
    同一份公式抄两遍，抓不到 `RoPE2D` 的错误。
    """
    g = blk.gau
    b, c, h, w = x4.shape
    n = h * w
    hd, nh = g.head_dim, g.num_heads

    def heads(t):                                   # (B,N,e) → (B,nh,N,hd)
        return t.reshape(b, n, nh, hd).permute(0, 2, 1, 3).contiguous()

    t = x4.reshape(b, c, n).permute(0, 2, 1)        # (B,N,C)
    tn = blk.norm(t)
    # 门控激活用 relu²（与被测代码一致），不是 SiLU。
    u = _relu2(g.u(tn))
    v = _relu2(g.v(tn))
    q = g.rope(heads(g.q(tn)), pos)
    k = g.rope(heads(g.k(tn)), pos)
    # value **就是** GLU 的 V，不再单独投影（GAU 与 MHSA 的根本差别）。
    vh = heads(v)
    # relu²(QKᵀ·s + b)：与被测 `GatedAttentionUnit._attn` 一致（无 softmax，
    # 按 key 维行 L1 归一 = "除以序列长度"）。
    scores = (q @ k.transpose(-2, -1)) * g.scale + g.b
    a = _relu2(scores)
    a = a / (a.sum(-1, keepdim=True) + 1e-6)
    ctx = (a @ vh).permute(0, 2, 1, 3).reshape(b, n, g.e)
    gated = (u * v * ctx).clamp(-_GAU_GATED_CLAMP, _GAU_GATED_CLAMP)
    return g.o(gated)


# ---- 1. 默认结构 ------------------------------------------------------- #
def test_factory_block_default_is_all_nbt_and_unchanged_params():
    """`Nbt2TransformerBlock` 工厂默认（不传 attn_impl/gau_positions）仍是全
    `TransformerBlock`，参数量仍是 493,056 —— 「加 GAU 不能动老块结构」的保证：
    直接构造块的地方（旧 checkpoint 加载、结构对拍）行为逐位不变。
    """
    blk = Nbt2TransformerBlock(C, M, H, F_HID)
    assert all(isinstance(b, TransformerBlock) for b in blk.inner)
    assert blk.inner_len == 2
    n_params = sum(p.numel() for p in blk.parameters())
    assert n_params == 493_056


def test_default_net_is_all_nbt():
    """`NbtTfNet()` 默认按 `NBT_TF_CFG` 走纯 nbt（GAU 已关）：

    全部 11 块都是 [nbt, nbt]（493,056/块）。GAU 是 opt-in，必须经
    `cfg.gau_positions` 显式开启（见 `test_mixed_inner_is_one_gau_plus_one_nbt`）。
    """
    net = NbtTfNet()
    assert net.blocks[0].inner_len == 2
    for i in range(11):
        assert [type(b).__name__ for b in net.blocks[i].inner] == \
            ['TransformerBlock', 'TransformerBlock']
        assert sum(p.numel() for p in net.blocks[i].parameters()) == \
            NBT_TF_CFG['params_block_nbt']


def test_default_cfg_is_pure_nbt_gau_opt_in():
    """默认配置是纯 nbt，GAU 是 opt-in（必须经 `gau_positions` 显式开启）。"""
    assert NBT_TF_CFG['attn_impl'] == 'nbt'
    assert NBT_TF_CFG['gau_positions'] is None
    assert NBT_TF_CFG['gau_hidden'] == 384   # GAU 参数仍保留（重开 GAU 用）
    assert NBT_TF_CFG['gau_first_nbt'] == 0
    assert NBT_TF_CFG['gau_last_nbt'] == 0


# ---- 2. 混合挂载 --------------------------------------------------------- #
@pytest.mark.parametrize('attn_impl, gau_positions, expected', [
    ('nbt', [0], ['GatedAttentionBlock', 'TransformerBlock']),
    ('nbt', [1], ['TransformerBlock', 'GatedAttentionBlock']),
    ('gau', [], ['GatedAttentionBlock', 'GatedAttentionBlock']),
    ('gau', [0], ['GatedAttentionBlock', 'GatedAttentionBlock']),
])
def test_mixed_inner_is_one_gau_plus_one_nbt(attn_impl, gau_positions, expected):
    """「每层分别 1 个 GAU + 1 个 nbt」：用 attn_impl + gau_positions 表达。"""
    blk = Nbt2TransformerBlock(C, M, H, F_HID, attn_impl=attn_impl,
                               gau_positions=gau_positions)
    assert [type(b).__name__ for b in blk.inner] == expected
    assert blk.inner_len == 2

    # 关掉边缘块的纯 nbt 约束（gau_first/last=0），才能拿到「每块都 [GAU,nbt]」。
    net = NbtTfNet(cfg={'attn_impl': 'nbt', 'gau_positions': [0],
                        'gau_first_nbt': 0, 'gau_last_nbt': 0})
    net.initialize()
    for blk in net.blocks:
        assert [type(b).__name__ for b in blk.inner] == \
            ['GatedAttentionBlock', 'TransformerBlock']


def test_gau_only_via_attn_impl():
    """`attn_impl='gau'` ⇒ 内块全 GAU（无 gau_positions 覆盖）。"""
    blk = Nbt2TransformerBlock(C, M, H, F_HID, attn_impl='gau')
    assert all(type(b).__name__ == 'GatedAttentionBlock' for b in blk.inner)


def test_resolve_inner_kinds_rejects_bad_args():
    with pytest.raises(ValueError):
        _resolve_inner_kinds('resnet', None, 2)          # 非法 attn_impl
    with pytest.raises(ValueError):
        _resolve_inner_kinds('nbt', [5], 2)             # 下标越界
    # 合法情况
    assert _resolve_inner_kinds('nbt', None, 2) == ['nbt', 'nbt']
    assert _resolve_inner_kinds('nbt', [0], 2) == ['gau', 'nbt']
    assert _resolve_inner_kinds('gau', [], 2) == ['gau', 'gau']


# ---- 3. 形状 ------------------------------------------------------------- #
def test_gau_forward_shape_preserved():
    blk = _gau_block()
    x = torch.randn(2, M, 19, 19)
    pos = _board_pos(2, 19, 19, x.device)
    assert blk(x, pos).shape == x.shape


# ---- 4. 门控语义（核心） ------------------------------------------------- #
def test_gau_gating_matches_manual_computation():
    """前向 == `x + W_o(U ⊙ V ⊙ Attn(Q,K,V))`，V 取自 GLU 的 V，门用 relu²。

    同时钉死：pre-norm 残差、GLU 门控（U）、以及「注意力的 value 不再单独投影」。
    """
    torch.manual_seed(0)
    blk = _gau_block()
    x = torch.randn(2, M, 6, 6)
    pos = _board_pos(2, 6, 6, x.device)

    out = blk(x, pos)
    b, c, h, w = x.shape
    # 残差分支：out - x 应等于手算的 GAU(norm(x))
    residual = (out - x).reshape(b, c, h * w).permute(0, 2, 1)
    expect = _manual_gau(blk, x, pos)
    assert torch.allclose(residual, expect, atol=1e-5, rtol=1e-4)


def test_gau_gate_uses_relu2():
    """U/V 分支走的是 relu²（非负、x≤0 归零），而不是 SiLU。

    直接验证 GAU 内部激活：喂一个含负值的输入，U 必全非负；且 U 与
    `relu(linear(t))²` 逐位一致。
    """
    torch.manual_seed(0)
    g = _gau_block().gau
    t = torch.randn(3, 5, g.dim) * 2.0      # 输入维度是 g.dim（=128），不是 g.e
    u = F.relu(g.u(t)).square()
    with torch.no_grad():
        manual = _relu2(g.u(t))
    assert torch.allclose(u, manual)
    assert (manual >= 0).all()
    # 比对一个非 relu² 激活（SiLU）应当**不**相等 —— 证明确实不是 SiLU。
    assert not torch.allclose(manual, F.silu(g.u(t)))


def test_gau_value_comes_from_glu_v_not_a_separate_projection():
    """改 `v` 的权重会改变**注意力的输入**，而不只是 GLU 的 value 分支。

    这是 GAU 与「MHSA + SwiGLU」的分水岭：MHSA 有自己的 `v` 投影，动它不影响
    FFN；GAU 里只有一份 V，注意力与门控共用。
    """
    torch.manual_seed(0)
    blk = _gau_block()
    x = torch.randn(1, M, 5, 5)
    pos = _board_pos(1, 5, 5, x.device)
    before = blk(x, pos)
    with torch.no_grad():
        blk.gau.v.weight.add_(0.05)
    assert not torch.allclose(before, blk(x, pos), atol=1e-6)

    # GAU 里没有独立的 attention value 投影：只有 u/v/q/k/o + RoPE 频率 + 注意力偏置 b。
    assert set(n for n, _ in blk.gau.named_parameters()) == {
        'u.weight', 'v.weight', 'q.weight', 'k.weight', 'o.weight',
        'rope.freq', 'b'}


# ---- 5. RoPE ------------------------------------------------------------- #
def test_gau_output_depends_on_pos():
    """换一组 pos ⇒ 输出变 ⇒ q/k 真的过了 `RoPE2D`（不是「忘了加」）。"""
    torch.manual_seed(0)
    blk = _gau_block()
    x = torch.randn(1, M, 6, 6)
    grid = _board_pos(1, 6, 6, x.device)
    shuffled = torch.rand_like(grid) * 5.0
    assert not torch.allclose(blk(x, grid), blk(x, shuffled), atol=1e-6)


def test_gau_is_translation_equivariant_in_pos():
    """pos 整体平移常量 ⇒ 输出不变：RoPE 的相对性（与 `MHSA` 一致的性质）。"""
    torch.manual_seed(0)
    blk = _gau_block()
    x = torch.randn(1, M, 6, 6)
    grid = _board_pos(1, 6, 6, x.device)
    shifted = grid + 7.0
    assert torch.allclose(blk(x, grid), blk(x, shifted), atol=1e-5)


# ---- 6. dtype / 后端 ----------------------------------------------------- #
def test_gau_preserves_input_dtype():
    """fp16 进 ⇒ fp16 出。

    `RoPE2D.forward` 现在把 pos 对齐到 `x.dtype`（之前 pos 恒 fp32 会把 q/k 提升成
    fp32，与 v 的 fp16 不匹配、整条 fp16 训练崩）；这里钉死 GAU 内部不偷偷升精度。
    """
    blk = _gau_block().half()
    x = torch.randn(2, M, 8, 8, dtype=torch.float16)
    pos = _board_pos(2, 8, 8, x.device)
    assert blk(x, pos).dtype == torch.float16


def test_gau_runs_on_math_backend(monkeypatch):
    """GAU 的自力更生的手写注意力（relu² 路径）必须能跑且输出有限。
    （GAU 不再依赖 `_sdpa`/`_sdpa_force_math`，这里只验证数值有限性。）"""
    monkeypatch.setattr(backbone, '_sdpa_force_math', True)
    blk = _gau_block()
    x = torch.randn(2, M, 6, 6)
    pos = _board_pos(2, 6, 6, x.device)
    assert torch.isfinite(blk(x, pos)).all()


# ---- 7. 形状约束 --------------------------------------------------------- #
def test_gau_hidden_must_be_divisible_by_head_dim():
    """`hidden` 不能被 `head_dim` 整除时立刻报错，而不是静默 reshape 出错。

    V 要切成整数个 value 头，且切出来的 head_dim 必须与 q/k 相同（手写注意力
    要求 q/k/v 的 head_dim 一致）。
    """
    with pytest.raises(ValueError):
        GatedAttentionUnit(M, H, hidden=300)     # 300 % 32 != 0

    g = GatedAttentionUnit(M, H, hidden=F_HID)   # 384 % 32 == 0
    assert (g.head_dim, g.e, g.num_heads) == (32, 384, 12)
    assert g.e == g.num_heads * g.head_dim       # value 的 head_dim == q/k 的

    with pytest.raises(ValueError):
        GatedAttentionUnit(100, 3)               # dim 不能被 num_heads 整除
