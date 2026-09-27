"""v21 三块 + 两头的结构锁（P4.1）。

范围
----
只锁**新加**的 `MambaLTI` / `TransformerBlock` / `CrossAttnRes` / `FCPolicyHead` /
`FCValueHead`（含共用的 `MHSA` 核心），以及「既有类没被动过」（D5）。既有
resnet / convnext 路径的预算锁在 `tests/test_param_budget.py` / `test_v19_budget.py`
里，本文件**不重复**它们，也**不**预跑 v21 全网预算窗口（那是 P4.2 的
`test_v21_budget`；本文件第 11 条只把主干的 6,499,800 加总钉住，作为它的地基）。

为什么逐类用**精确相等**而不是窗口
--------------------------------
参数表是用户逐项给定的，分项**没有自由度**（唯一一处自由度是 §3(a) 的 stem 核，
由第 11 条从 3×3 的 28,520 侧钉住）。窗口会放过「某一层从 184 挪到 192 而总数
不变」这类改动，而这类改动会让 P4.2 的全网预算在错误的结构上绿灯。

⚠ policy 头的数字：权威表 §2 自己写的合计 **2,366,730** 是对的，而它分项里
「FC 128→256：32,896」是笔误（128×256 + 256 = 33,024）。brief §3(b) 因此把
合计当成 2,366,602，但 2,366,602 是任何 `nn.Linear(128, 256, bias=True)` 都达不
不到的数（去掉 bias 则是 2,366,474）。本文件按**可建出来的**结构断言 2,366,730，
推理见 `task-p4-1-report.md` §4(b)。

「每条都要能红」
--------------
第 13 节的「空转守卫」段落里，每个结构性断言都配一条对照：要么断言反向
（该变的必须变），要么断言另一种同样自洽的写法**不被**接受（防止「两个错里蒙对
一个」）。报告里的「哪个测试抓哪个错误改动」对照表是**注入错误改动实测**出来的，
不是推断的。

全部 CPU、秒级：块类用真实 184 通道但只喂 5×5（只有两个头用 19×19，因为
policy 头的 Flatten 维被 board_size 钉死），不加载任何真实权重、不读真实数据。
"""
import ast
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.networks.backbone import (  # noqa: E402
    MHSA, ConvNeXtBlock, CrossAttnRes, MambaLTI, MultiHeadSelfAttention,
    ResBlock, TransformerBlock, set_sdpa_force_math,
)
from src.networks.policy_network import FCPolicyHead, PolicyNetwork  # noqa: E402
from src.networks.value_network import FCValueHead, ValueNetwork  # noqa: E402

BACKBONE_PY = os.path.join(ROOT, 'src', 'networks', 'backbone.py')
ALPHANET_PY = os.path.join(ROOT, 'src', 'networks', 'alphanet.py')

CH = 184          # 主干通道
BOARD = 19        # 19×19 + pass = 362
DROPOUT = 0.1     # train_sft.py 的 --attention-dropout 默认值

# 权威表 §2 的逐块参数（精确值，不是窗口）
EXPECT_MAMBA = 113_528
EXPECT_TRANSFORMER = 224_480
EXPECT_CROSS = 326_416
EXPECT_MHSA = 135_424
EXPECT_POLICY = 2_366_730
EXPECT_VALUE = 142_017
EXPECT_BACKBONE_TOTAL = 6_499_800


def _n(mod):
    return sum(p.numel() for p in mod.parameters())


def _child_counts(mod):
    """直接子模块 + 直接 Parameter 的参数量（定位漂移用：报错能指到是哪一层变了）。"""
    out = {name: _n(child) for name, child in mod.named_children()}
    out.update({name: p.numel() for name, p in mod.named_parameters(recurse=False)})
    return out


# --------------------------------------------------------------------------- #
# 1. 参数量：逐类精确相等
# --------------------------------------------------------------------------- #
def test_mamba_lti_param_count_is_exact():
    m = MambaLTI()
    counts = _child_counts(m)
    assert counts == {
        'norm': 368,            # pre-LN 184
        'in_proj': 67_712,      # 184→368 无bias
        'dw_conv': 920,         # Conv1d(184,184,k=4,groups=184,bias=True)
        'x_proj': 6_624,        # 184→36 无bias
        'dt_proj': 920,         # 4→184 bias
        'A_log': 2_944,         # 184×16
        'D': 184,               # 逐通道直通
        'out_proj': 33_856,     # 184→184 无bias
    }, f'MambaLTI 分项漂移：{counts}'
    assert _n(m) == EXPECT_MAMBA, \
        f'MambaLTI 实测 {_n(m)} != 权威 {EXPECT_MAMBA}'


def test_transformer_block_param_count_is_exact():
    m = TransformerBlock()
    counts = _child_counts(m)
    assert counts == {
        'norm1': 368,           # pre-LN ×2 之 1
        'attn': EXPECT_MHSA,    # Wq/Wk/Wv/Wo 四个独立无bias Linear
        'norm2': 368,           # pre-LN ×2 之 2
        'fc1': 44_160,          # 184→240 无bias
        'fc2': 44_160,          # 240→184 无bias
    }, f'TransformerBlock 分项漂移：{counts}'
    assert _n(m) == EXPECT_TRANSFORMER, \
        f'TransformerBlock 实测 {_n(m)} != 权威 {EXPECT_TRANSFORMER}'


def test_cross_attn_res_param_count_is_exact():
    m = CrossAttnRes()
    counts = _child_counts(m)
    assert counts == {
        'norm_tap': 368,        # pre-LN ×3 之 1（184 维，逐路复用）
        'proj': 101_568,        # Conv2d(552→184, 1, bias=False)
        'norm_attn': 368,       # pre-LN ×3 之 2
        'attn': EXPECT_MHSA,
        'norm_ffn': 368,        # pre-LN ×3 之 3
        'fc1': 44_160,
        'fc2': 44_160,
    }, f'CrossAttnRes 分项漂移：{counts}'
    assert _n(m) == EXPECT_CROSS, \
        f'CrossAttnRes 实测 {_n(m)} != 权威 {EXPECT_CROSS}'


def test_mhsa_core_param_count_is_exact():
    m = MHSA(CH, num_heads=4)
    assert _n(m) == EXPECT_MHSA, f'MHSA 实测 {_n(m)} != 权威 {EXPECT_MHSA}'
    assert sorted(_child_counts(m)) == ['wk', 'wo', 'wq', 'wv'], \
        'MHSA 必须是 Wq/Wk/Wv/Wo 四个**独立** Linear（不是融合 qkv）'


def test_head_param_counts_are_exact():
    p, v = FCPolicyHead(), FCValueHead()
    assert _child_counts(p) == {
        'conv1': 17_664, 'bn1': 192,       # 权威表的 17,856 = 17,664 + 192
        'conv2': 4_608, 'bn2': 96,         # 权威表的  4,704 =  4,608 +  96
        'fc1': 2_218_112, 'fc2': 33_024, 'fc3': 93_034,
    }, f'FCPolicyHead 分项漂移：{_child_counts(p)}'
    assert _n(p) == EXPECT_POLICY, \
        f'FCPolicyHead 实测 {_n(p)} != 权威 {EXPECT_POLICY}'
    assert _child_counts(v) == {
        'conv': 5_888, 'bn': 64, 'gap': 0, 'fc1': 4_224, 'fc2': 66_048,
        'fc3': 65_664, 'fc4': 129, 'out_tanh': 0,
    }, f'FCValueHead 分项漂移：{_child_counts(v)}'
    assert _n(v) == EXPECT_VALUE, \
        f'FCValueHead 实测 {_n(v)} != 权威 {EXPECT_VALUE}'


# --------------------------------------------------------------------------- #
# 2. MambaLTI 的形状自洽（含**前向里**的中间维，不只是构造参数）
# --------------------------------------------------------------------------- #
def test_ssm_init_shapes():
    m = MambaLTI()
    assert m.in_proj.in_features == CH and m.in_proj.out_features == 2 * CH, \
        'in_proj 必须是 184→368（expand=2 是**扇出**，拆两半各 184）'
    assert m.in_proj.bias is None, 'in_proj 必须无 bias'
    assert isinstance(m.dw_conv, nn.Conv1d)
    assert m.dw_conv.in_channels == CH and m.dw_conv.out_channels == CH, \
        'dw conv1d 的通道数应等于 in_channels（深度卷积）'
    assert m.dw_conv.groups == m.dw_conv.in_channels == CH, \
        f'dw conv1d 必须是深度卷积（groups == in_channels == 184），实得 {m.dw_conv.groups}'
    assert m.dw_conv.kernel_size == (4,), \
        f'd_conv=4 ⇒ kernel_size=(4,)，实得 {m.dw_conv.kernel_size}'
    assert m.dw_conv.bias is not None, 'dw conv1d 必须带 bias（920 = 736+184）'
    assert m.x_proj.in_features == CH and m.x_proj.out_features == 4 + 2 * 16 == 36, \
        'x_proj 必须是 184→36（ddt rank 4 + 2N=32）'
    assert m.x_proj.bias is None, 'x_proj 必须无 bias'
    assert m.dt_proj.in_features == 4 and m.dt_proj.out_features == CH, \
        'dt_proj 必须是 4→184'
    assert m.dt_proj.bias is not None, 'dt_proj 必须带 bias（920 = 736+184）'
    assert tuple(m.A_log.shape) == (CH, 16), \
        f'A 状态矩阵必须是 184×16，实得 {tuple(m.A_log.shape)}'
    A = -torch.exp(m.A_log)
    assert bool((A < 0).all()), 'A 必须逐元素为负（否则 exp(dt·A) > 1，递推发散）'
    assert torch.allclose(A[0], -torch.arange(1., 17.), atol=1e-6), \
        'A 初值应为 -(1..N)（Mamba 参考实现；A 以 log 形式存储）'
    assert tuple(m.D.shape) == (CH,) and bool((m.D == 1).all()), 'D 是 184 维逐通道直通'
    assert m.out_proj.in_features == CH and m.out_proj.bias is None, \
        'out_proj 必须是 184→184 无 bias'


def test_ssm_dt_is_softplus_gated_in_source():
    """`dt` 必须是 softplus(dt_proj(...))（静态锁）。

    dt > 0 与 A < 0 合起来才是 exp(dt·A) ∈ (0,1] —— 这是递推不发散、cumsum 闭式的
    指数恒 ≤ 0（不溢出）的**唯一**依据。去掉 softplus 后 dt 可为负、衰减变成
    放大，递推在 T=361 上会炸；而「数值层面」很难用一个短测试稳定地抓住它
    （小 T 上只是偏差变大），故锁源码形状。
    """
    tree = ast.parse(open(BACKBONE_PY, encoding='utf-8').read())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == 'MambaLTI')
    fn = next(f for f in node.body
              if isinstance(f, ast.FunctionDef) and f.name == 'forward')
    wrapped = []
    for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
        name = getattr(call.func, 'attr', None) or getattr(call.func, 'id', None)
        if name in ('softplus', 'silu'):
            for inner in ast.walk(call):
                if isinstance(inner, ast.Call) and (
                        getattr(inner.func, 'attr', None) == 'dt_proj'):
                    wrapped.append(ast.unparse(call))
    assert wrapped, ('MambaLTI.forward 里找不到 softplus(dt_proj(...)) —— dt 必须为正，'
                     '否则 exp(dt·A) > 1，LTI 递推会放大发散')


def test_ssm_intermediate_shapes_in_forward():
    """前向里各中间张量的**末维**：dw_conv→184、x_proj→36、dt_proj→184。"""
    m = MambaLTI().eval().double()
    seen = {}
    for name in ('dw_conv', 'x_proj', 'dt_proj', 'in_proj', 'out_proj'):
        mod = getattr(m, name)
        seen[name] = []
        mod.register_forward_hook(
            lambda _m, _i, out, name=name: seen[name].append(tuple(out.shape)))

    with torch.no_grad():
        y = m(torch.randn(2, CH, 3, 3, dtype=torch.float64))
    assert tuple(y.shape) == (2, CH, 3, 3)
    assert seen['in_proj'] == [(2, 9, 2 * CH)], seen['in_proj']
    assert seen['dw_conv'] == [(2, CH, 9)], seen['dw_conv']
    assert seen['x_proj'] == [(2, 9, 36)], seen['x_proj']
    assert seen['dt_proj'] == [(2, 9, CH)], seen['dt_proj']
    assert seen['out_proj'] == [(2, 9, CH)], seen['out_proj']


# --------------------------------------------------------------------------- #
# 3~5. LTI 递推：朴素参考 / 闭式 cumsum 恒等式 / 因果性
# --------------------------------------------------------------------------- #
def _small_mamba():
    """小配置：让朴素参考的双重/三重 for 循环可读且秒级（形状与 184 版同构）。"""
    torch.manual_seed(7)
    return MambaLTI(channels=8, expand=2, d_conv=4, d_state=4, ddt_rank=2).double()


def _reference_inputs(m, x):
    """参考实现的前半段：pre-LN → in_proj → 因果 dw conv → x_proj/dt_proj → 状态量。

    全部用 torch 原语重写（含 LayerNorm），**不调用**被测的 `MambaLTI.forward`。
    """
    B, C, H, W = x.shape
    T = H * W
    # LN 在通道维（ConvNeXt 风格，LayerNorm2d 同口径）→ 输出 (B,H,W,C) → (B,T,C)
    hn = F.layer_norm(x.permute(0, 2, 3, 1), (C,),
                      m.norm.norm.weight, m.norm.norm.bias, m.norm.norm.eps)
    h = hn.reshape(B, T, C)
    hz = F.linear(h, m.in_proj.weight)
    xs, z = hz[..., :C], hz[..., C:]

    padded = F.pad(xs.transpose(1, 2), (m.d_conv - 1, 0))          # 左填充：因果
    xc = F.silu(F.conv1d(padded, m.dw_conv.weight, m.dw_conv.bias,
                         groups=m.dw_conv.groups)).transpose(1, 2)   # (B, T, C)

    db = F.linear(xc, m.x_proj.weight)
    dt_raw, bc = db[..., :m.ddt_rank], db[..., m.ddt_rank:]
    dt = F.softplus(F.linear(dt_raw, m.dt_proj.weight, m.dt_proj.bias))
    b_vec, c_vec = bc[..., :m.d_state], bc[..., m.d_state:]
    A = -torch.exp(m.A_log)
    drive = dt.unsqueeze(-1) * xc.unsqueeze(-1) * b_vec.unsqueeze(2)
    return dict(x=x, xc=xc, dt=dt, A=A, b_vec=b_vec, c_vec=c_vec,
                drive=drive, z=z, B=B, T=T, C=C, N=m.d_state)


def test_ssm_scan_matches_naive_triple_loop():
    """顺序扫描 == 测试内**三重 for 循环**的朴素解（无向量化、无生产函数复用）。

    朴素解把闭式逐项展开：h_t = Σ_{s≤t} Π_{r=s+1..t} exp(dt_r·A) · drive_s，
    这正是 report §3 里 LTI 递推的原始形式。
    """
    m = _small_mamba()
    x = torch.randn(2, 8, 3, 3, dtype=torch.float64)
    ref = _reference_inputs(m, x)
    dt, A, drive, c_vec = ref['dt'], ref['A'], ref['drive'], ref['c_vec']
    T, C, N = ref['T'], ref['C'], ref['N']

    ys = []
    for t in range(T):
        acc = torch.zeros(ref['B'], C, N, dtype=x.dtype)
        for s in range(t + 1):
            decay = torch.ones(ref['B'], C, N, dtype=x.dtype)
            for r in range(s + 1, t + 1):          # Π_{r=s+1..t} exp(dt_r·A)
                decay = decay * torch.exp(dt[:, r].unsqueeze(-1) * A)
            acc = acc + decay * drive[:, s]
        ys.append((acc * c_vec[:, t].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=1)
    y = y + m.D * ref['xc']                                    # D 直通
    out = F.linear(y, m.out_proj.weight) * F.silu(ref['z'])     # z 走 SiLU 门
    expected = x + out.transpose(1, 2).reshape(x.shape)

    with torch.no_grad():
        got = m(x)
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-12), \
        '顺序扫描与朴素三重循环不一致：max|Δ|={:.3e}'.format(
            (got - expected).abs().max().item())


def test_ssm_scan_equals_dt_strided_cumulative_form():
    """顺序扫描 == 「dt 步长的因果累积」闭式（cumsum 版），即 report §3 的公式。

    cumA_t = Σ_{r≤t} dt_r ⊙ A（单调不增，故 exp(cumA_t − cumA_s) ≤ 1，不溢出）；
    h_t = Σ_{s≤t} exp(cumA_t − cumA_s) · drive_s。两种写法在数学上恒等，
    任何一处把 mask、步长或求和轴搞错都会红。
    """
    m = _small_mamba()
    x = torch.randn(2, 8, 3, 3, dtype=torch.float64)
    ref = _reference_inputs(m, x)
    dt, A, drive, c_vec, T = ref['dt'], ref['A'], ref['drive'], ref['c_vec'], ref['T']

    cumA = torch.cumsum(dt.unsqueeze(-1) * A, dim=1)             # (B,T,C,N)
    mask = torch.ones(T, T, dtype=torch.bool).tril()             # s <= t
    M = torch.exp(cumA.unsqueeze(2) - cumA.unsqueeze(1))        # (B,T_t,T_s,C,N)
    M = M * mask.view(1, T, T, 1, 1)
    h = torch.einsum('btscN,bscN->btcN', M, drive)
    y = (h * c_vec.unsqueeze(2)).sum(-1)
    y = y + m.D * ref['xc']
    out = F.linear(y, m.out_proj.weight) * F.silu(ref['z'])
    expected = x + out.transpose(1, 2).reshape(x.shape)

    with torch.no_grad():
        got = m(x)
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-12), \
        'cumsum 闭式与顺序扫描不一致：max|Δ|={:.3e}'.format(
            (got - expected).abs().max().item())

    # 对照：把下三角 mask 换成**上**三角（只看未来 s > t）必须得到不同的结果 ——
    # 证明 mask 这条断言不是空转
    M_nc = torch.exp(cumA.unsqueeze(2) - cumA.unsqueeze(1)) * \
        torch.ones(T, T, dtype=torch.bool).triu(1).view(1, T, T, 1, 1).to(x.dtype)
    h_nc = torch.einsum('btscN,bscN->btcN', M_nc, drive)
    y_nc = (h_nc * c_vec.unsqueeze(2)).sum(-1)
    assert not torch.allclose(y, y_nc, rtol=1e-6, atol=1e-9), \
        '把因果 mask 换成只看未来后结果竟不变 —— cumsum 闭式这条断言分辨不出因果性'


def test_ssm_is_causal_in_time():
    """改 t 之后的输入**不得**影响 t 的输出（前向对比，float64 紧容差）。

    逐个 t0 扫一遍：既检查 t0 之前逐位不变，也检查 t0 及之后**确实变了**
    （后者是前者的空转守卫：若扰动根本没进计算路径，「不变」证明不了因果性）。
    """
    m = _small_mamba()
    x = torch.randn(2, 8, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        base = m(x)
        for t0 in range(16):
            x2 = x.clone()
            x2[:, :, t0 // 4, t0 % 4] += 5.0
            delta = (m(x2) - base).flatten(2)          # (B, C, T)
            if t0:
                before = delta[:, :, :t0].abs().max().item()
                assert before < 1e-12, \
                    f'因果性被破坏：改 t={t0} 的输入却改变了 t<{t0} 的输出' \
                    f'（max|Δ|={before:.3e}）'
            after = delta[:, :, t0].abs().max().item()
            assert after > 1e-6, \
                f't={t0} 的扰动没进计算路径（max|Δ|={after:.3e}）—— ' \
                f'「之前不变」这条断言可能是空转的'


def test_ssm_gate_is_on_the_z_branch():
    """钉住门的位置：`out = out_proj(y) * SiLU(z)`，SiLU 在 **z** 上（不是 y 上）。

    v19/v21 参考实现都是 `out_proj(y) * silu(z)`；brief §4 的括注写成
    `SiLU(y) * z`，与同句「z 走 SiLU 门」矛盾，本测试按前者（且用对照排除后者）。
    """
    m = _small_mamba()
    x = torch.randn(2, 8, 3, 3, dtype=torch.float64)
    ref = _reference_inputs(m, x)

    def _y_from_sequential():
        T, C, N = ref['T'], ref['C'], ref['N']
        state = torch.zeros(ref['B'], C, N, dtype=x.dtype)
        ys = []
        for t in range(T):
            state = state * torch.exp(ref['dt'][:, t].unsqueeze(-1) * ref['A']) \
                + ref['drive'][:, t]
            ys.append((state * ref['c_vec'][:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1) + m.D * ref['xc']

    y = _y_from_sequential()
    proj = F.linear(y, m.out_proj.weight)
    with torch.no_grad():
        got = m(x)
    expected = x + (proj * F.silu(ref['z'])).transpose(1, 2).reshape(x.shape)
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-12), \
        'SiLU 门不在 z 上（或 out_proj / D 直通 / 残差位置不对）'

    # 对照：SiLU 作用在 y 上必须**不**等于本实现（否则这条断言分辨不出两种读法）
    other = x + (F.silu(proj) * ref['z']).transpose(1, 2).reshape(x.shape)
    assert not torch.allclose(got, other, rtol=1e-6, atol=1e-9), \
        'SiLU 放在 y 上与放在 z 上结果相同 —— 本测试分辨不出门的位置'


# --------------------------------------------------------------------------- #
# 6~7. TransformerBlock / CrossAttnRes
# --------------------------------------------------------------------------- #
def test_transformer_block_structure():
    m = TransformerBlock()
    assert m.attn.num_heads == 4, f'heads 必须是 4，实得 {m.attn.num_heads}'
    assert m.attn.head_dim == CH // 4 == 46
    assert m.fc1.in_features == CH and m.fc1.out_features == 240, \
        f'FFN 必须是 184→240（ratio≈1.304，旧的 276/1.5 已作废），实得 {m.fc1.out_features}'
    assert m.fc2.in_features == 240 and m.fc2.out_features == CH
    assert m.fc1.bias is None and m.fc2.bias is None, 'FFN 两层都必须无 bias'
    linears = [getattr(m.attn, n) for n in ('wq', 'wk', 'wv', 'wo')]
    assert all(isinstance(li, nn.Linear) for li in linears)
    assert all(li.bias is None for li in linears), 'Wq/Wk/Wv/Wo 都必须无 bias'
    assert len({id(li) for li in linears}) == 4, 'Wq/Wk/Wv/Wo 必须是四个独立对象'
    assert isinstance(m.norm1.norm, nn.LayerNorm) and m.norm1.norm.normalized_shape == (CH,)
    assert isinstance(m.norm2.norm, nn.LayerNorm) and m.norm2.norm.normalized_shape == (CH,)

    x = torch.randn(2, CH, 5, 5)
    m.eval()
    with torch.no_grad():
        assert tuple(m(x).shape) == (2, CH, 5, 5)
        # 两条恒等捷径：把两条支路的**末端**权重清零 ⇒ 整块必须是恒等映射
        keep = {k: v.clone() for k, v in m.state_dict().items()}
        with torch.no_grad():
            m.attn.wo.weight.zero_()
            m.fc2.weight.zero_()
        assert torch.equal(m(x), x), 'MHSA/FFN 支路清零后输出必须逐位等于输入（两条恒等捷径）'
        # 对照：恢复权重后输出必须**不**等于输入（否则上面那条是空转的）
        m.load_state_dict(keep)
        assert not torch.equal(m(x), x), '支路权重恢复了输出却逐位相同 —— 恒等捷径断言空转'


def test_cross_attn_res_uses_three_inputs():
    m = CrossAttnRes()
    assert tuple(m.proj.weight.shape) == (CH, 552, 1, 1), \
        f'跨层投影必须是 Conv2d(552→184, k=1)；实得 {tuple(m.proj.weight.shape)}'
    assert m.proj.bias is None, 'proj 必须无 bias（表里没有 BN/ReLU，也没有 bias）'
    assert m.proj.kernel_size == (1, 1)
    assert m.tap_channels == (184, 184, 184)

    for mod in m.modules():                       # 明确「无 BatchNorm、无 ReLU」
        assert not isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)), \
            f'CrossAttnRes 内不得有 BatchNorm（权威的 326,416 不含 BN 的 368）：{mod}'
    assert not any(isinstance(mod, nn.ReLU) for mod in m.modules()), \
        'CrossAttnRes 内不得有 ReLU 子模块'
    assert [n for n, _ in m.named_children() if 'bn' in n.lower()] == [], \
        'CrossAttnRes 不得有任何 *bn* 子模块'

    taps = tuple(torch.randn(2, CH, 5, 5) for _ in range(3))
    m.eval()
    with torch.no_grad():
        x = torch.randn(2, CH, 5, 5)
        out = m(x, taps)
        assert tuple(out.shape) == (2, CH, 5, 5)
    # 换掉抽头内容时输出必须变化。注意不能用 `t + 1.0`：逐路 LN 会把常数平移整个
    # 消掉（LN(t+1) ≡ LN(t)），那样的对照是假绿。
    other = m(x, tuple(torch.randn_like(t) for t in taps))
    assert not torch.allclose(out, other, atol=1e-6), \
        '换掉抽头内容输出却几乎不变 —— 抽头没有真正进计算路径'

    # 把两条恒等捷径的末端权重清零 ⇒ 跨层投影支路可以被**单独**取出来比：
    # 这同时钉住「concat 的次序是 (s1, s5, s9)」与「LN 在 proj 之前、且中间没有
    # 任何 BN/ReLU」（否则手动算式与实测对不上）。
    keep = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        m.attn.wo.weight.zero_()
        m.fc2.weight.zero_()
        got = m(x, taps) - x
        want = m.proj(torch.cat([m.norm_tap(t) for t in taps], dim=1))
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6), \
        f'投影支路实测 {got[0, :2, 0, 0]} vs 手动 {want[0, :2, 0, 0]}（次序/LN 位置/额外算子）'
    for perm in ((1, 2, 0), (2, 0, 1)):
        with torch.no_grad():
            other_order = m.proj(torch.cat([m.norm_tap(taps[i]) for i in perm], dim=1))
        assert not torch.allclose(want, other_order, atol=1e-6), \
            f'把 concat 次序换成 {perm} 结果不变 —— 这条断言分辨不出抽头次序'
    with torch.no_grad():
        m.load_state_dict(keep)

    # 构造时给的每路通道数都被真正用上：故意传错宽度必须抛
    with pytest.raises(ValueError):
        m(x, (torch.randn(2, 192, 5, 5), taps[1], taps[2]))       # 第 1 路错
    with pytest.raises(ValueError):
        m(x, (taps[0], taps[1], torch.randn(2, 192, 5, 5)))       # 第 3 路错
    with pytest.raises(ValueError):
        m(x, taps[:2])                                            # 少一路
    # 构造参数也真的生效：tap_channels 改了，proj.in_channels 必须跟着变
    wide = CrossAttnRes(tap_channels=(96, CH, CH))
    assert wide.proj.in_channels == 96 + 2 * CH
    with pytest.raises(ValueError):
        wide(x, taps)                                             # 仍按 552 传 → 抛


def test_mhsa_core_is_shared_not_duplicated():
    """两个块类共用**同一个** MHSA 实现（brief §1：不要复制两份注意力）。"""
    assert type(TransformerBlock().attn) is MHSA
    assert type(CrossAttnRes().attn) is MHSA

    src = open(BACKBONE_PY, encoding='utf-8').read()
    tree = ast.parse(src)
    owners = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names = {t.attr for st in ast.walk(node) if isinstance(st, ast.Assign)
                     for t in st.targets if isinstance(t, ast.Attribute)}
            if 'wq' in names and 'wo' in names:
                owners.append(node.name)
    assert owners == ['MHSA'], \
        f'Wq/Wk/Wv/Wo 的实现只允许出现在 MHSA 一处，实得 {owners}（重复实现会让两份漂移）'


# --------------------------------------------------------------------------- #
# 8. 两个头的输出
# --------------------------------------------------------------------------- #
def test_head_output_shapes_and_range():
    p = FCPolicyHead().eval()
    v = FCValueHead().eval()
    x = torch.randn(3, CH, BOARD, BOARD)
    with torch.no_grad():
        logits, value = p(x), v(x)
    assert tuple(logits.shape) == (3, BOARD * BOARD + 1), \
        f'policy 头必须输出 362（19×19+pass），实得 {tuple(logits.shape)}'
    assert tuple(value.shape) == (3, 1), f'value 头必须输出标量，实得 {tuple(value.shape)}'
    assert bool((value.abs() <= 1.0).all()), \
        f'value 头末尾有 Tanh，值域必须 ⊂ [-1,1]，实得 [{value.min():.4f}, {value.max():.4f}]'
    # 空转守卫：值不能恒为 0（恒 0 也会满足 |v|≤1）。末层前的 logit 在默认初始化下
    # 只有 O(0.04)，所以阈值取 1e-3 而不是 0.5。
    assert value.abs().max().item() > 1e-3, \
        'value 恒接近 0 —— |v|≤1 这条断言分辨不出「Tanh 在」与「输出恒 0」'
    assert value.std(dim=0).item() > 0.0, 'value 对 batch 内不同样本完全无响应'
    assert isinstance(v.out_tanh, nn.Tanh)

    # 端到端把「Tanh 在不在」钉死（与种子无关的硬证据）：
    # ① 结构上：输出必须**逐位等于** tanh(fc4 的输出)；
    # ② 行为上：把输入放大 1e4 倍，末层 logit 会到 O(100)，有 Tanh 则 |v| 饱和到 1、
    #    无 Tanh 则 |v| 直接飙到 O(100) ⇒ `> 0.5` 仍成立但 `<= 1` 破掉。
    pre = []
    v.fc4.register_forward_hook(lambda _m, _i, o: pre.append(o.detach()))
    with torch.no_grad():
        hot = v(x * 1e4)
    assert torch.equal(hot, torch.tanh(pre[-1])), \
        'value 头的输出不等于 tanh(fc4 输出) —— 末层不是 Tanh（或中间多了别的算子）'
    assert hot.abs().max().item() > 0.5, \
        f'大输入下 |v| 仅 {hot.abs().max().item():.4f} —— 末层没有 Tanh（不饱和）'
    assert bool((hot.abs() <= 1.0).all()), \
        f'大输入下 |v| 越界：{hot.abs().max().item():.4f} > 1 —— 末层没有 Tanh'

    assert FCPolicyHead(action_size=100).eval()(x).shape == (3, 100)
    with pytest.raises(ValueError):        # board_size 被钉死，棋盘不符必须立刻报
        FCPolicyHead().eval()(torch.randn(2, CH, 9, 9))
    # value 头有 GAP ⇒ 与棋盘尺寸无关（19×19 与 13×13 都行）
    assert tuple(FCValueHead().eval()(torch.randn(2, CH, 13, 13)).shape) == (2, 1)


# --------------------------------------------------------------------------- #
# 9. D5：既有类签名一字未改
# --------------------------------------------------------------------------- #
def _init_sig(path, cls):
    """`class X.__init__` 的参数列表（去掉 self），按源码顺序拼成可读字符串。"""
    tree = ast.parse(open(path, encoding='utf-8').read())
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == cls), None)
    assert node is not None, f'{os.path.basename(path)} 里找不到类 {cls}'
    fn = next((f for f in node.body if isinstance(f, ast.FunctionDef)
               and f.name == '__init__'), None)
    assert fn is not None, f'{cls} 没有 __init__'
    a = fn.args
    assert a.posonlyargs == [] and not a.kwonlyargs and not a.vararg and not a.kwarg, \
        f'{cls} 的 __init__ 出现了新的参数种类（positional-only / kwonly / *args / **kwargs）'
    names = [ast.unparse(x) for x in a.args[1:]]          # 去掉 self
    n_def = len(a.defaults)
    for i, d in enumerate(a.defaults):
        names[len(names) - n_def + i] += '=' + ast.unparse(d)
    return ', '.join(names)


def test_legacy_class_signatures_untouched():
    """D5：既有类的 `__init__` 签名逐字未变（含 `AlphaGoNet`）。"""
    expect_backbone = {
        # norm 层也被锁：ConvNeXt 路径依赖 LayerNorm2d、MultiHeadSelfAttention 依赖
        # RMSNorm，改它们的签名同样是 D5 违规（mutation 实测：只锁块类时漏掉了）
        'RMSNorm': 'channels, eps=1e-06',
        'LayerNorm2d': 'channels',
        'ResBlock': 'channels',
        'ConvNeXtBlock': 'channels',
        'MultiHeadSelfAttention': (
            "channels, num_heads=4, dropout=0.0, mode='global', window_size=7"),
        'AttentionResBlock': (
            "channels, num_heads=4, dropout=0.0, attention_mode='global', window_size=7"),
        'SharedBackbone': (
            "in_channels=12, channels=128, num_res_blocks=12, attention_mode='mix', "
            "num_attention_layers=4, num_heads=4, attention_dropout=0.0, "
            "attn_mode='global', attn_window=7, use_checkpoint=False, arch='convnext', "
            "res_blocks=0, convnext_blocks=0, attn_blocks=0"),
    }
    for cls, want in expect_backbone.items():
        got = _init_sig(BACKBONE_PY, cls)
        assert got == want, f'backbone.{cls}.__init__ 签名被改了：\n  实得 {got}\n  期望 {want}'

    assert _init_sig(os.path.join(ROOT, 'src', 'networks', 'policy_network.py'),
                     'PolicyNetwork') == (
        'in_channels=64, hidden_channels=32, action_size=81, num_layers=2')
    assert _init_sig(os.path.join(ROOT, 'src', 'networks', 'value_network.py'),
                     'ValueNetwork') == (
        "in_channels=64, hidden_channels=32, num_res_blocks=3, arch='resnet'")

    # AlphaGoNet.__init__ 零改动（D1）
    got = _init_sig(ALPHANET_PY, 'AlphaGoNet')
    assert got == (
        "in_channels: int=12, backbone_channels: int=128, backbone_res_blocks: int=12, "
        "attention_mode: str='mix', num_attention_layers: int=4, num_heads: int=4, "
        "attention_dropout: float=0.0, attn_mode: str='global', attn_window: int=7, "
        "policy_channels: int=32, value_channels: int=64, action_size: int=362, "
        "use_checkpoint: bool=False, arch: str='resnet', res_blocks: int=0, "
        "convnext_blocks: int=0, attn_blocks: int=0, value_res_blocks: int=3, "
        "policy_layers: int=2"), f'AlphaGoNet.__init__ 签名被改了：{got}'


def test_legacy_forward_untouched():
    """既有类的 forward 源码片段未变（防「顺手重构」掉 v19 的行为）。"""
    src = open(BACKBONE_PY, encoding='utf-8').read()
    for frag in (
        'out = F.relu(self.bn_out(self.conv_out(out)))',
        'self.qkv = nn.Linear(channels, channels * 3, bias=False)',
        'return self.attn_drop if self.training else 0.0',
    ):
        assert frag in src, f'backbone.py 缺片段 {frag!r} —— 既有行为被改动了'
    vsrc = open(os.path.join(ROOT, 'src', 'networks', 'value_network.py'),
                encoding='utf-8').read()
    assert 'x = self.gap(x).flatten(1)\n        x = self.fc(x)\n        return x' in vsrc, \
        'ValueNetwork.forward 被改了（D5：v19 头必须零回归，v21 的 Tanh 只在 FCValueHead）'


# --------------------------------------------------------------------------- #
# 10. 注意力 dropout 必须走 P2.2b 的既有闸门
# --------------------------------------------------------------------------- #
def test_attention_dropout_gated_in_new_blocks():
    for block in (TransformerBlock(attn_dropout=DROPOUT), CrossAttnRes(attn_dropout=DROPOUT)):
        name = type(block).__name__
        assert block.attn.attn_drop == DROPOUT
        block.train()
        assert block.attn.attn_drop_p == DROPOUT, f'{name}: train 下注意力 dropout 未生效'
        block.eval()
        assert block.attn.attn_drop_p == 0.0, \
            f'{name}: eval 下 attn_drop_p 应为 0.0（functional dropout 不认 self.training）'

        x = torch.randn(2, CH, 5, 5)
        taps = (x, x, x)
        args = (x,) if name == 'TransformerBlock' else (x, taps)
        block.eval()
        torch.manual_seed(2024)
        rng0 = torch.get_rng_state().clone()
        with torch.no_grad():
            o1 = block(*args)
            o2 = block(*args)
        assert torch.equal(o1, o2), f'{name}: eval 下同输入两次前向不逐位相同'
        assert torch.equal(rng0, torch.get_rng_state()), \
            f'{name}: eval 期间消耗了 torch 随机 —— 注意力 dropout 没被闸门关掉'
        block.train()
        with torch.no_grad():
            t1, t2 = block(*args), block(*args)
        assert not torch.equal(t1, t2), \
            f'{name}: train 下两次前向逐位相同 —— 注意力 dropout 压根没进计算路径'


def test_new_blocks_add_no_ungated_sdpa_site():
    """静态锁：新块类里**不得**出现任何 dropout 站点（既不许自建 `_sdpa` 调用，
    也不许直接调函数式 dropout API），且本文件的 `_sdpa` 闸门站点仍是 4 处。"""
    src = open(BACKBONE_PY, encoding='utf-8').read()
    tree = ast.parse(src)

    banned = {'_sdpa', 'dropout', 'scaled_dot_product_attention', 'flash_attn_func'}
    for cls in ('MHSA', 'MambaLTI', 'TransformerBlock', 'CrossAttnRes'):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.ClassDef) and n.name == cls)
        offenders = []
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            f = call.func
            name = getattr(f, 'id', None) or getattr(f, 'attr', None)
            if name in banned:
                offenders.append(ast.unparse(call)[:70])
        assert not offenders, \
            f'{cls} 里出现了未过闸门的 dropout 站点 {offenders} —— 必须复用 ' \
            f'MultiHeadSelfAttention.attn_drop_p / _global_attn 这条既有通路'

    sdpa_sites = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        if getattr(call.func, 'id', None) == '_sdpa':
            for kw in call.keywords:
                if kw.arg == 'dropout_p':
                    sdpa_sites.append(ast.unparse(kw.value))
    assert sdpa_sites == ['self.attn_drop_p'] * 4, \
        f'本文件的 _sdpa dropout_p 站点清单变了：{sdpa_sites}（应恰好 4 处全部走 attn_drop_p）'


# --------------------------------------------------------------------------- #
# 11. 主干加总：P4.2 的 test_v21_budget 的地基（8/4/2/2 + stem + out）
# --------------------------------------------------------------------------- #
def test_v21_backbone_budget_sums_to_anchor():
    stem = nn.Sequential(nn.Conv2d(17, CH, 3, padding=1, bias=False), nn.BatchNorm2d(CH))
    out = nn.Sequential(nn.Conv2d(CH, CH, 1, bias=False), nn.BatchNorm2d(CH))
    parts = {
        'stem': _n(stem),                       # 28,520（§3(a)：按 3×3 实施）
        'res_blocks': 8 * _n(ResBlock(CH)),    # 8 × 610,144
        'mamba_lti': 4 * _n(MambaLTI()),       # 4 × 113,528
        'transformer': 2 * _n(TransformerBlock()),
        'cross_attn_res': 2 * _n(CrossAttnRes()),
        'out': _n(out),                         # 34,224
    }
    assert parts == {
        'stem': 28_520, 'res_blocks': 4_881_152, 'mamba_lti': 454_112,
        'transformer': 448_960, 'cross_attn_res': 652_832, 'out': 34_224,
    }, f'主干分项漂移：{parts}'
    assert sum(parts.values()) == EXPECT_BACKBONE_TOTAL, \
        f'主干合计 {sum(parts.values())} != 权威 {EXPECT_BACKBONE_TOTAL}'
    # 对照：7×7 stem（表头文字那个读法）会把合计顶到 6,624,920 ⇒ §3(a) 的 3×3 裁定
    # 是这条断言的判据来源，不是随手选的。7×7 的真值是 49×17×184 = 153,272（+BN=153,640），
    # brief §3(a) 写的 153,296 差 24 —— 同一处笔误，见 report §4(a)。
    stem7 = nn.Sequential(nn.Conv2d(17, CH, 7, padding=3, bias=False), nn.BatchNorm2d(CH))
    assert _n(stem7) == 153_640
    assert sum(parts.values()) - parts['stem'] + _n(stem7) == 6_624_920


def test_sdpa_force_math_restorable():
    """本文件不改变模块级后端开关（`_sdpa_force_math`），收尾复原以便其它测试复用。"""
    assert set_sdpa_force_math(True) is None
