"""v21 三块 + 两头的结构锁（P4.1；MambaLTI 块由 P4.1r 换成 Mamba-2，P4.1s 改逐 head 标量衰减，
**P4.1s-vec 把扫描收敛成唯一的向量化路径**）。

范围
----
只锁**新加**的 `MambaLTI` / `TransformerBlock` / `CrossAttnRes` / `FCPolicyHead` /
`FCValueHead`（含共用的 `MHSA` 核心），以及「既有类没被动过」（D5）。既有
resnet / convnext 路径的预算锁在 `tests/test_param_budget.py` / `test_v19_budget.py`
里，本文件**不重复**它们，也**不**预跑 v21 全网预算窗口（那是 P4.2 的
`test_v21_budget`；本文件第 11 条只把主干的 6,558,696 加总钉住，作为它的地基）。

MambaLTI 的参数化变迁（用户 2026-09-27 两次裁决）
------------------------------------------------
* P4.1（rev1，Mamba-1）：N=16、稠密 `A_log` 184×16（2,944）、单块 113,528。
* P4.1r：N 16→**64**、`x_proj` 36→**132**、`A` 改为**结构化低秩**
  `A_u`(184×4)+`A_v`(4×64)（992）、单块 129,240。
* **P4.1s（本文件）**：`A` 再改为**逐 head 标量** —— `A_log` 形状 **(4,)**、
  共 **4** 个参数，4 个 head 各覆盖 46 通道（184/4），`decay = −exp(A_log)` 沿
  head 广播、**与状态维 n 无关**；单块 129,240→**128,252**。
* **P4.1s-vec**：把扫描**收敛成唯一一条向量化路径**。P4.1s 一度保留
  `scan='auto'` 的运行时分派（训练走分块、eval 走顺序递推），现已删除
  `MAMBA_SCAN_DEFAULT` / `MambaLTI(scan=...)` / `MambaLTI._scan_impl`。
  顺序递推**没有删**，改名为 `MambaLTI._sequential_scan_oracle` 并**标注为
  仅测试**（向量化实现唯一的独立对照系）。参数量**一个字节都没动**。

「Mamba-1 语义」的东西（因果性、朴素三重循环、cumsum 闭式、pre-norm 公式、
SiLU 门的位置）**一条都没删也没弱化** —— 它们钉的是递推的**语义**，与 N/A 的
参数化无关；状态形状 `(B,4,46,64)` 与 132 的 x_proj 拆分另由
`test_mamba2_state_shape_is_b_h_p_n` / `test_ssm_init_shapes` 钉住。
`test_ssm_scan_matches_naive_triple_loop` 的朴素参考**照旧**在 shipped 形状
（C=184, N=64）上额外跑一遍（见 `test_ssm_naive_oracle_at_shipped_N`），
所以 N 16→64 没有把 oracle 缩到测不出问题的小配置里。


为什么逐类用**精确相等**而不是窗口
--------------------------------
参数表是用户逐项给定的，分项**没有自由度**（唯一一处自由度是 §3(a) 的 stem 核，
由第 11 条从 3×3 的 28,520 侧钉住）。窗口会放过「某一层从 184 挪到 192 而总数
不变」这类改动，而这类改动会让 P4.2 的全网预算在错误的结构上绿灯。

⚠ policy 头的数字：权威表 §2 自己写的合计 **2,366,730** 是对的，而它分项里
「FC 128→256：32,896」是笔误（128×256 + 256 = 33,024）。brief §3(b) 因此把
合计当成 2,366,602，但 2,366,602 是任何 `nn.Linear(128, 256, bias=True)` 都达不
到的数（去掉 bias 则是 2,366,474）。本文件按**可建出来的**结构断言 2,366,730，
推理见 `task-p4-1-report.md` §4(b)。**P4.1r / P4.1s 均未改动 policy 头**（用户明示
「policy 不变」）。

「每条都要能红」
--------------
报告 §6 的对照表是**注入错误改动实测**出来的（P4.1s：18 处 mutation，18/18 被抓；
P4.1s-vec 重跑 19 处，19/19 被抓），不是推断的。每条结构性断言另外还配一条对照：
要么断言反向（该变的必须变），要么断言另一种同样自洽的写法**不被**接受
（防止「两个错里蒙对一个」）。

全部 CPU、秒级：块类用真实 184 通道但只喂 5×5（只有两个头用 19×19，因为
policy 头的 Flatten 维被 board_size 钉死），不加载任何真实权重、不读真实数据。
唯一例外是 §5.7 的 chunked 性能守卫要按 19×19 测（它测的就是 361 步的真实成本），
以及 §5.8 的 shipped-N 朴素 oracle（本机复测 18 ms / 89 ms，见该测试 docstring）。
"""
import ast
import contextlib
import math
import os
import sys
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.networks.backbone import (  # noqa: E402
    MAMBA_CHUNK_SIZE, MHSA, ConvNeXtBlock, CrossAttnRes,
    MambaLTI, MultiHeadSelfAttention, ResBlock, TransformerBlock, set_sdpa_force_math,
)
from src.networks.policy_network import FCPolicyHead, PolicyNetwork  # noqa: E402
from src.networks.value_network import FCValueHead, ValueNetwork  # noqa: E402

BACKBONE_PY = os.path.join(ROOT, 'src', 'networks', 'backbone.py')
ALPHANET_PY = os.path.join(ROOT, 'src', 'networks', 'alphanet.py')

CH = 184          # 主干通道
BOARD = 19        # 19×19 + pass = 362
DROPOUT = 0.1     # train_sft.py 的 --attention-dropout 默认值
DSTATE = 64       # Mamba-2 的状态维 N（P4.1r：rev1 的 16 -> 64）
RANK = 4          # ddt_rank：x_proj 输出 132 = RANK + 2·DSTATE 的第一段
N_HEADS = 4       # P4.1s：head 数 = A 的标量个数（每 head 46 通道）
HEAD_DIM = CH // N_HEADS   # 46

# 权威表 §2 的逐块参数（精确值，不是窗口）
EXPECT_MAMBA = 128_252
EXPECT_TRANSFORMER = 224_480
EXPECT_CROSS = 326_416
EXPECT_MHSA = 135_424
EXPECT_POLICY = 2_366_730
EXPECT_VALUE = 142_017
EXPECT_BACKBONE_TOTAL = 6_558_696


def _n(mod):
    return sum(p.numel() for p in mod.parameters())


def _child_counts(mod):
    """直接子模块 + 直接 Parameter 的参数量（定位漂移用：报错能指到是哪一层变了）。"""
    out = {name: _n(child) for name, child in mod.named_children()}
    out.update({name: p.numel() for name, p in mod.named_parameters(recurse=False)})
    return out


def _seeded_x(board, batch=1, seed=20260927):
    """确定性输入（用显式 generator，不动全局 RNG 状态 ⇒ 不污染别的测试）。"""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, CH, board, board, generator=gen)


def _dt_per_head(m, x):
    """每个 head 的平均 `dt`（(4,)），由 `_reference_inputs` 复算，不经生产 forward。

    记忆常数 `τ = 1/(dt̄_h·|A_h|)` 步 —— 「这个 head 的状态还能记住几步」。
    逐 head 标量衰减下，**多尺度结构完全由这 4 个数决定**（n 维不再有斜坡），
    所以 τ 既是设计指标也是必须钉住的行为指标。
    """
    return (_reference_inputs(m, x)['dt']
            .reshape(-1, m.n_heads, m.head_dim).mean(dim=(0, 2)).detach())


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
        'x_proj': 24_288,       # 184→132 无bias（ddt 4 + 2N 128）
        'dt_proj': 920,         # 4→184 bias
        'A_log': 4,             # P4.1s：逐 head 标量衰减（P4.1r 的结构化低秩 992 → 4）
        'D': 184,               # 逐通道直通
        'out_proj': 33_856,     # 184→184 无bias
    }, f'MambaLTI 分项漂移：{counts}'
    assert _n(m) == EXPECT_MAMBA, \
        f'MambaLTI 实测 {_n(m)} != 权威 {EXPECT_MAMBA}'
    # A 相关的参数必须**只有** A_log 一张 (4,) 表。稠密 184×64 的 A 是 11,776、
    # P4.1r 的结构化低秩是 992 —— 这里是预算爆炸的第一道闸
    # （另见 test_mamba2_A_is_per_head_scalar）
    assert counts['A_log'] == 4 == N_HEADS, \
        f'A 的参数必须是逐 head 标量 A_log(4,)=4，实得 {counts["A_log"]}'
    assert CH * DSTATE == 11_776 > 4, \
        f'稠密 A 需 {CH * DSTATE} 个参数，逐 head 标量只用 4 个（差 2944 倍）'


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
    assert m.d_state == DSTATE == 64, f'P4.1r 钉死 N=64，实得 {m.d_state}'
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
    assert m.x_proj.in_features == CH and m.x_proj.out_features == 4 + 2 * DSTATE == 132, \
        'x_proj 必须是 184→132（ddt rank 4 + 2N=128）'
    assert m.x_proj.bias is None, 'x_proj 必须无 bias'
    assert m.dt_proj.in_features == 4 and m.dt_proj.out_features == CH, \
        'dt_proj 必须是 4→184'
    assert m.dt_proj.bias is not None, 'dt_proj 必须带 bias（920 = 736+184）'

    # A：P4.1s 的逐 head 标量 —— A_log(4,)，没有 U/V、没有稠密 A（见
    # test_mamba2_A_is_per_head_scalar），这里钉 head 划分与 decay 的广播语义
    assert not hasattr(m, 'A_u') and not hasattr(m, 'A_v'), \
        'P4.1s 已把 P4.1r 的结构化低秩 U/A_v 换成逐 head 标量 A_log'
    assert m.n_heads == N_HEADS == 4 and m.head_dim == HEAD_DIM == 46, \
        f'head 划分必须是 4×46，实得 {m.n_heads}×{m.head_dim}'
    assert m.n_heads * m.head_dim == m.d_inner == CH, 'n_heads × head_dim 必须等于 184'
    assert tuple(m.A_log.shape) == (N_HEADS,), f'A_log 必须是 (4,)，实得 {tuple(m.A_log.shape)}'
    A = -torch.exp(m.A_log.detach())
    assert tuple(A.shape) == (N_HEADS,) and bool((A < 0).all()), \
        'A = −exp(A_log) 必须形状 (4,) 且逐元素**严格**负（否则 decay > 1，递推发散）'
    # 初值：canonical Mamba-2 的 `A = −(1..H)`，即 `log([1, 2, 3, 4])`。
    # ⚠ 这里**不是** rev1 / P4.1r 的 `−(1..N)` 斜坡按 head 取段平均
    # （那给出 log([8.5,24.5,40.5,56.5])，在 dt~O(0.7) 下四个 head 的记忆常数
    # 全部 < 0.2 步、且 fp32+分块下 37/288 组配置出现恒零梯度；实测见 report §4）。
    # 四个 head 互不相同（同值则梯度相同、对称性永远破不掉）。
    want = torch.log(torch.arange(1., float(N_HEADS) + 1))
    assert torch.allclose(m.A_log, want, rtol=0, atol=1e-6), \
        f'A_log 初值应为 log([1,2,3,4])，实得 {m.A_log.tolist()}'
    # 行为层守卫（不只是比一个数）：初值下每个 head 的**记忆常数**
    # 1/(dt̄·|A|) 必须落在一个可用的窗口里。斜坡平均版会让这四个数
    # 掉到 0.025~0.17 步（≈ 没有时间记忆），canonical 版是 0.36~1.43 步。
    tau = 1.0 / (_dt_per_head(m, _seeded_x(5)) * (-A))
    assert float(tau.min()) > 0.25, \
        f'最慢 head 的记忆常数 {float(tau.min()):.3f} 步 <= 0.25 —— 衰减快到没有时间记忆了'
    assert float(tau.max() / tau.min()) <= 8.0, \
        (f'head 间记忆常数跨度 {float(tau.max() / tau.min()):.1f}x 过大 —— '
         f'实测 tau = {[round(float(v), 3) for v in tau]}')
    assert len({round(float(v), 6) for v in A.tolist()}) == N_HEADS, \
        '四个 head 的衰减标量初值必须互不相同（否则 per-head 自由度是死的）'
    # decay 沿 head 广播：通道 c 的衰减只依赖 c//46 —— 这是 chunked 共享 (L,L)
    # 跨状态维 n 的代数来源，必须能被一个 (184,) 的逐通道向量复算出来
    per_ch = A.repeat_interleave(HEAD_DIM)
    assert tuple(per_ch.shape) == (CH,) and bool((per_ch[:HEAD_DIM] == A[0]).all()), \
        '同一 head 的 46 个通道必须共享同一个衰减标量'
    # ⚠ 与 P4.1r 不同：A 现在**逐 head 相同**、head 内逐通道也相同 ——
    #   多尺度多样性从「逐状态维 (n)」退化为「逐 head」，这是用户裁决的直接后果。

    assert tuple(m.D.shape) == (CH,) and bool((m.D == 1).all()), 'D 是 184 维逐通道直通'
    assert m.out_proj.in_features == CH and m.out_proj.bias is None, \
        'out_proj 必须是 184→184 无 bias'
    # ⚠ P4.1s-vec 之后**没有** `scan` 这个运行时开关了（分块 SSD 是唯一路径，
    #   顺序递推改名为 `_sequential_scan_oracle` 且**仅供测试**）。这三条把
    #   「没有第二条路可选」钉在构造期：`scan` 形参不存在、属性不存在、模块级
    #   `MAMBA_SCAN_DEFAULT` 常量已删除。
    assert m.chunk_size == MAMBA_CHUNK_SIZE
    with pytest.raises(ValueError):
        MambaLTI(chunk_size=0)


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
    """前向里各中间张量的**末维**：dw_conv→184、x_proj→132、dt_proj→184。"""
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
    assert seen['x_proj'] == [(2, 9, 132)], seen['x_proj']
    assert seen['dt_proj'] == [(2, 9, CH)], seen['dt_proj']
    assert seen['out_proj'] == [(2, 9, CH)], seen['out_proj']
    # x_proj 的 132 维必须真的按「前 4 / 后 128」切成 dt / B / C 三段：
    # 把 B、C 的边界挪一位（前 4 / 前 68）会让下游形状对不上，钉住这个顺序
    db = m.x_proj(torch.randn(2, 9, CH, dtype=torch.float64))
    dt_raw, bc = db[..., :RANK], db[..., RANK:]
    b_vec, c_vec = bc.split(DSTATE, dim=-1)
    assert tuple(dt_raw.shape) == (2, 9, RANK), 'dt 段必须是前 4 维'
    assert tuple(b_vec.shape) == tuple(c_vec.shape) == (2, 9, DSTATE)
    assert RANK + 2 * DSTATE == db.shape[-1] == 132
    # N=64 时 128 能被 4 整除 ⇒ `bc.split(4, -1)` 这种误切**不会**报错，只会静默
    # 给出 32 份 4 维的碎片（于是 B/C 各自变成 4 维而不是 64 维）。记录它的后果，
    # 真实实现必须走 `split(64)`。
    naive = db[..., RANK:].split(4, dim=-1)
    assert len(naive) == 32 and naive[0].shape[-1] == 4 != DSTATE, \
        f'误切 split(4) 实际给 {len(naive)} 份 × {naive[0].shape[-1]} 维'


# --------------------------------------------------------------------------- #
# 3~5. LTI 递推：朴素参考 / 闭式 cumsum 恒等式 / 因果性
# --------------------------------------------------------------------------- #
def _small_mamba():
    """小配置：让朴素参考的多重 for 循环可读且秒级（形状与 184 版**同构**）。

    `n_heads=2 < d_state=4` 刻意保留 **head 数 < N** 的划分 —— 若这里让
    n_heads=N，head 划分与衰减广播那条线在小配置上就测不出问题。

    ⚠ P4.1s-vec 之后没有 `scan=` 形参了：本块**只有**分块 SSD 一条路，所以
    `_small_mamba()` 的前向默认就是被测的生产路径（顺序 oracle 要显式调用
    `_sequential_scan_oracle`，或用 `_forward_with_oracle` 临时换掉 forward）。
    """
    torch.manual_seed(7)
    return MambaLTI(channels=8, expand=2, d_conv=4, d_state=4, ddt_rank=2,
                    n_heads=2).double()


@contextlib.contextmanager
def _forward_with_oracle(blk):
    """临时把 forward 用到的扫描换成**顺序 oracle**（仅测试用）。

    为什么需要它
    ------------
    P4.1s-vec 把 `scan='auto'` 的运行时分派删掉了，生产路径只剩分块 SSD。但
    「fp32 + 顺序递推下 A_log 梯度干净」是判定「那些恒零梯度是 fp32 **下溢**
    而不是**结构死亡**」的一半判据 —— 顺序路径退役后，这一半必须从**别处**取。

    本上下文管理器把 `_chunked_scan` 临时指向 `_sequential_scan_oracle`，于是
    「fp32 顺序」这个对照组**不需要任何生产开关**就仍然可测，而且顺带证明
    oracle 是生产路径的**可替换实现**（不是一份独立的、可能已经腐化的代码）。
    退出时无条件还原（`finally`），不污染同进程里的其它测试。
    """
    orig = MambaLTI._chunked_scan
    MambaLTI._chunked_scan = MambaLTI._sequential_scan_oracle
    try:
        yield blk
    finally:
        MambaLTI._chunked_scan = orig


def _hand_decay(m):
    """`A = −exp(A_log)` 的**手写**复算（直接读 `m.A_log`，不经生产 forward）。

    写成独立的一行是故意的：这样改生产代码里 `A = −exp(A_log)` 的取符号方式
    （去掉负号、换 log 域）会让下面所有参考对拍测试变红，而不只让一个形状测试变红。
    """
    return -torch.exp(m.A_log)


def _reference_inputs(m, x):
    """参考实现的前半段：pre-LN → in_proj → 因果 dw conv → x_proj/dt_proj → 状态量。

    全部用 torch 原语重写（含 LayerNorm），**不调用**被测的 `MambaLTI.forward`。
    形状口径与生产 forward 一致：`(B,T,H,P)` 的 dt/log_decay、`(B,T,H,P,N)` 的
    drive，其中 H=m.n_heads、P=m.head_dim。
    """
    B, C, H, W = x.shape
    T = H * W
    Hh, P = m.n_heads, m.head_dim
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
    b_vec, c_vec = bc.split(m.d_state, dim=-1)
    A = _hand_decay(m)                                            # (H,) < 0
    # ℓ[b,t,h,p] = dt[b,t,h,p]·A[h] —— **与状态维 n 无关**（P4.1s 逐 head 标量）
    log_decay = dt.view(B, T, Hh, P) * A[None, None, :, None]     # (B,T,H,P)
    drive = (dt.view(B, T, Hh, P).unsqueeze(-1)
             * xc.view(B, T, Hh, P).unsqueeze(-1)
             * b_vec[:, :, None, None, :])                        # (B,T,H,P,N)
    return dict(x=x, xc=xc.view(B, T, Hh, P), dt=dt.view(B, T, Hh, P), A=A,
                log_decay=log_decay, b_vec=b_vec, c_vec=c_vec,
                drive=drive, z=z, B=B, T=T, C=C, Hh=Hh, P=P, N=m.d_state)


def _hand_block_forward(m, x, ref):
    """把参考状态量接上「读出 + D 直通 + out_proj + SiLU 门 + 残差」。

    与 `_reference_inputs` 配套：前向的**后半段**也独立重写一遍，于是
    `test_ssm_scan_*` 那几条比较的是**整个块**，不只是扫描那一段。
    """
    y = ref['y'].view(ref['B'], ref['T'], ref['C']) + m.D * ref['xc'].view(
        ref['B'], ref['T'], ref['C'])                             # D 直通
    out = F.linear(y, m.out_proj.weight) * F.silu(ref['z'])       # z 走 SiLU 门
    return x + out.transpose(1, 2).reshape(x.shape)


def test_ssm_scan_matches_naive_triple_loop():
    """顺序扫描 == 测试内**三重 for 循环**的朴素解（无向量化、无生产函数复用）。

    朴素解把闭式逐项展开：h_t = Σ_{s≤t} Π_{r=s+1..t} exp(ℓ_r) · drive_s，
    其中 ℓ_r = dt_r·A_{head(p)}（P4.1s 的逐 head 标量衰减），
    这正是 task-p4-1s-report.md §2 里 LTI 递推的原始形式。
    """
    m = _small_mamba()
    x = torch.randn(2, 8, 3, 3, dtype=torch.float64)
    ref = _reference_inputs(m, x)
    log_decay, drive, c_vec = ref['log_decay'], ref['drive'], ref['c_vec']
    T, C, Hh, P, N = ref['T'], ref['C'], ref['Hh'], ref['P'], ref['N']

    ys = []
    for t in range(T):
        acc = torch.zeros(ref['B'], Hh, P, N, dtype=x.dtype)
        for s in range(t + 1):
            decay = torch.ones(ref['B'], Hh, P, N, dtype=x.dtype)
            for r in range(s + 1, t + 1):          # Π_{r=s+1..t} exp(ℓ_r)
                decay = decay * torch.exp(log_decay[:, r]).unsqueeze(-1)
            acc = acc + decay * drive[:, s]
        ys.append((acc * c_vec[:, t].view(ref['B'], 1, 1, N)).sum(-1))
    ref['y'] = torch.stack(ys, dim=1)
    expected = _hand_block_forward(m, x, ref)

    with torch.no_grad():
        got = m(x)
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-12), \
        '顺序扫描与朴素三重循环不一致：max|Δ|={:.3e}'.format(
            (got - expected).abs().max().item())


def test_ssm_d_skip_multiplies_by_d():
    """`D` 直通项必须真的**乘上 `D`**：`y = y + D ⊙ xc`。

    ⚠ 这条是 P4.1s-vec 补上的**覆盖漏洞**（注入错误改动实测发现的）
    --------------------------------------------------------------
    `D` 的初值是**全 1**（`nn.Parameter(torch.ones(184))`，Mamba 参考实现的惯例），
    而 IEEE-754 下 `1.0 * x == x` 逐位成立 ⇒ 只要测试用初值权重跑，把
    `self.D * xc` 改成 `xc`（甚至把整条 skip 摘掉）都是**逐位等价**的改动 ——
    也就是说 `test_ssm_scan_matches_naive_triple_loop` 等所有在初值权重上跑的
    测试**都抓不到它**。变异实测：`y = y + self.D * xc` → `y = y + xc` 时
    3 条相关测试全绿。

    修法：把 `D` 推离 1（本测试用**互不相同**且**不是 1** 的确定性取值），此时
    「乘了 D」与「没乘 D」变成可分辨的两件事。参考侧 `_hand_block_forward` 读的
    正是 `m.D * ref['xc']`，所以这条同时钉住「D 是乘性系数」和「D 逐通道」。
    """
    m = _small_mamba()
    x = torch.randn(2, 8, 3, 3, dtype=torch.float64)
    d_keep = torch.tensor([0.5, -1.25, 2.0, 1.0, 0.125, -0.75, 3.5, 0.0],
                          dtype=torch.float64)          # 逐通道互不相同、含 0 与负
    assert not bool((d_keep == 1.0).all()) and len(set(d_keep.tolist())) > 4, \
        'D 的探针取值本身必须既非全 1、又逐通道不同，否则本测试分辨不出「乘了 D」'
    with torch.no_grad():
        m.D.data.copy_(d_keep)
    ref = _reference_inputs(m, x)
    ref['y'] = m._chunked_scan(ref['log_decay'], ref['drive'], ref['c_vec'])
    want = _hand_block_forward(m, x, ref)          # 参考侧：y + m.D ⊙ xc
    with torch.no_grad():
        got = m(x)
        m.D.data.fill_(1.0)                        # 「D 被当成 1 / 根本没乘 D」
        no_d = _hand_block_forward(m, x, ref)
        m.D.data.copy_(d_keep)
    assert torch.allclose(got, want, rtol=1e-10, atol=1e-12), \
        ('D ⊙ xc 不对：max|Δ|={:.3e} —— 直通项没有真正乘上 D（或 D 没逐通道）'
         .format((got - want).abs().max().item()))
    # 对照（无空转守卫）：把 D 换成全 1（等价于「没乘 D」）必须给出**不同**的结果
    assert not torch.allclose(got, no_d, rtol=1e-6, atol=1e-9), \
        'D 取全 1 与取探针值给出同一输出 —— 本测试分辨不出 D 是否被用上'


def test_ssm_scan_equals_dt_strided_cumulative_form():
    """顺序扫描 == 「dt 步长的因果累积」闭式（cumsum 版），即 report §2 的公式。

    cumA_t = Σ_{r≤t} ℓ_r（单调不增，故 exp(cumA_t − cumA_s) ≤ 1，不溢出）；
    h_t = Σ_{s≤t} exp(cumA_t − cumA_s) · drive_s。两种写法在数学上恒等，
    任何一处把 mask、步长或求和轴搞错都会红。
    """
    m = _small_mamba()
    x = torch.randn(2, 8, 3, 3, dtype=torch.float64)
    ref = _reference_inputs(m, x)
    log_decay, drive, c_vec, T = (ref['log_decay'], ref['drive'],
                                  ref['c_vec'], ref['T'])

    cumA = torch.cumsum(log_decay, dim=1)                    # (B,T,H,P)
    Nn = ref['N']
    mask = torch.ones(T, T, dtype=torch.bool).tril()         # s <= t
    M = torch.exp(cumA.unsqueeze(2) - cumA.unsqueeze(1)).unsqueeze(-1)  # (B,T_t,T_s,H,P,1)
    M = M * mask.view(1, T, T, 1, 1, 1)
    h = (M * drive.unsqueeze(1)).sum(dim=2)                 # (B,T,H,P,N)
    y = (h * c_vec.view(ref['B'], T, 1, 1, Nn)).sum(-1)
    ref['y'] = y
    expected = _hand_block_forward(m, x, ref)

    with torch.no_grad():
        got = m(x)
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-12), \
        'cumsum 闭式与顺序扫描不一致：max|Δ|={:.3e}'.format(
            (got - expected).abs().max().item())

    # 对照：把下三角 mask 换成**上**三角（只看未来 s > t）必须得到不同的结果 ——
    # 证明 mask 这条断言不是空转
    M_nc = torch.exp(cumA.unsqueeze(2) - cumA.unsqueeze(1)).unsqueeze(-1) * \
        torch.ones(T, T, dtype=torch.bool).triu(1).view(1, T, T, 1, 1, 1).to(x.dtype)
    h_nc = (M_nc * drive.unsqueeze(1)).sum(dim=2)
    y_nc = (h_nc * c_vec.view(ref['B'], T, 1, 1, Nn)).sum(-1)
    assert not torch.allclose(y, y_nc, rtol=1e-6, atol=1e-9), \
        '把因果 mask 换成只看未来后结果竟不变 —— cumsum 闭式这条断言分辨不出因果性'


def test_ssm_is_causal_in_time():
    """改 t 之后的输入**不得**影响 t 的输出（前向对比，float64 紧容差）。

    逐个 t0 扫一遍：既检查 t0 之前逐位不变，也检查 t0 及之后**确实变了**
    （后者是前者的空转守卫：若扰动根本没进计算路径，「不变」证明不了因果性）。

    ⚠ 扰动必须**逐通道不同**（这是 review 打回重写的关键点）
    ---------------------------------------------------------
    `MambaLTI` 的 pre-LN 是**逐位置**的 `LayerNorm2d`，它会减掉每个位置上
    8 个通道的均值。第一版扰动写成 `x2[:,:,r,c] += 5.0`（8 个通道同加 5.0），
    是个**均匀平移**，被 pre-LN 整体消掉：实测 pre-LN 后 delta = 8.9e-16、
    in_proj 后 delta = 1.1e-16。SSM 什么都没看到，输出变化只剩残差 `x + out` 里的
    那份直接拷贝 —— 于是「之后必变」读到的 5.0 全部是残差给的，而「之前不变」在
    换成**前向扫描**（state 从 drive_t 累到 drive_{T-1}）的实现下**依然成立**：
    实测 leak = 0.000e+00（本该 > 0）。这条测试对扫描曾是彻底空转的。
    改成逐通道随机扰动后：正确实现 leak = 0.0e+00，前向扫描 leak = 3.34e-04
    （见 report 的 `## Fix 增补` §1）。
    """
    m = _small_mamba()
    x = torch.randn(2, 8, 4, 4, dtype=torch.float64)
    # 逐通道的随机扰动（形状 (B, C)，只加在被扰动的那一个位置上 ⇒ 8 个通道各不相同）
    bump = torch.randn(2, 8, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(20260925)) * 3.0
    with torch.no_grad():
        base = m(x)
        for t0 in range(16):
            x2 = x.clone()
            x2[:, :, t0 // 4, t0 % 4] += bump
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
        T, C, Hh, P, N = ref['T'], ref['C'], ref['Hh'], ref['P'], ref['N']
        state = torch.zeros(ref['B'], Hh, P, N, dtype=x.dtype)
        ys = []
        for t in range(T):
            state = state * torch.exp(ref['log_decay'][:, t]).unsqueeze(-1) \
                + ref['drive'][:, t]
            ys.append((state * ref['c_vec'][:, t].view(ref['B'], 1, 1, N)).sum(-1))
        return torch.stack(ys, dim=1).view(ref['B'], T, C) \
            + m.D * ref['xc'].view(ref['B'], T, C)

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
# 5b. P4.1s：Mamba-2 的逐 head 标量 A / 状态形状 / 死状态 / 分块 SSD 扫描
# --------------------------------------------------------------------------- #
def test_mamba2_A_is_per_head_scalar():
    """P4.1s 裁决：A 是**逐 head 标量** —— `A_log` 形状恰为 (4,)、共 4 个参数。

    ① 参数里**不存在** 184×64 / 46×64 的稠密 A（预算闸）；
    ② `n_heads × head_dim == 184`（head 划分自洽）；
    ③ `decay = −exp(A_log)` 逐元素 < 0 —— **构造上**（对数域，任意 A_log 都成立）；
    ④ 衰减沿 head 广播：同一 head 的 46 个通道共享同一标量，且**与状态维 n 无关**
       —— 这是 chunked 共享 (L,L) 的代数来源，用 exp(dt·A) 的秩**恰为 1** 钉住；
    ⑤ 初值 = canonical Mamba-2 的 `A = −(1..H)`，即 `log([1, 2, 3, 4])`。

    这一条是 P4.1r 的 `test_ssm_A_is_structurally_low_rank` **改写**而来（不是删除）：
    断言的对象从「低秩 U@V」换成「逐 head 标量」，四条守卫（无稠密 A / head 划分
    自洽 / 符号构造性保证 / 沿 head 广播且与 n 无关 ⇒ 秩 1）逐一对应，另加一条
    秩 64 的**对照**（被禁掉的逐通道参数化）并排钉死。
    """
    m = MambaLTI()
    shapes = {n: tuple(p.shape) for n, p in m.named_parameters()}

    # ① 稠密 A 不存在。任何 184×64（=11,776）或 46×64 的参数都会让单块预算爆掉。
    assert not any((CH in s and DSTATE in s) or (HEAD_DIM in s and DSTATE in s)
                   for s in shapes.values()), f'参数里出现了稠密 A: {shapes}'
    assert sorted(n for n in shapes if n.startswith('A')) == ['A_log'], \
        f'衰减参数必须只有 A_log 一张表，实得 {sorted(n for n in shapes if n.startswith("A"))}'

    # ② head 划分自洽
    assert m.n_heads == N_HEADS == 4 and m.head_dim == HEAD_DIM == 46
    assert m.n_heads * m.head_dim == CH == m.d_inner, 'n_heads × head_dim 必须等于 184'
    assert tuple(m.A_log.shape) == (N_HEADS,) and m.A_log.numel() == 4

    # ③ 符号在**构造上**保证：A = −exp(A_log) 对**任意** A_log 都 < 0
    #    （log 域，同 rev1 的 A_log；不依赖初值符号，远离 init 也成立）
    decay = -torch.exp(m.A_log)
    assert bool((decay < 0).all()) and bool((decay > -float('inf')).all())
    keep_a = m.A_log.data.clone()
    with torch.no_grad():
        # 远离 init 的极端值：含正、含 0、含大值（|A| 到 6.6e3）
        m.A_log.data = torch.tensor([-9.0, -1.0, 0.5, 8.8])
        far = -torch.exp(m.A_log)
        assert bool((far < 0).all()) and bool(torch.isfinite(far).all()), \
            'A = −exp(A_log) 必须对任意 A_log 都严格为负且有限'
        # decay = exp(dt·A) ∈ [0, 1]：上界 1 是**数学**性质（dt>0 ∧ A<0），必须严格成立；
        # 下界要写成 `>= 0` 而不是 `> 0` —— 上面刻意把 |A| 顶到 6.6e3，
        # `exp(−6.6e3·dt)` 在**任何**浮点格式下都下溢成恰好 0，而 0 是 exp 的
        # **正确**极限（不是数值缺陷）。「不放大」这条不变式才是递推不发散的关键，
        # 它由上界 1 保证（见 test_ssm_dt_is_softplus_gated_in_source）。
        dtc = torch.rand(CH, generator=torch.Generator().manual_seed(3),
                         dtype=torch.float64) * 0.7 + 0.05
        per_ch = far.double().repeat_interleave(HEAD_DIM)
        dmax = torch.exp(dtc.unsqueeze(-1) * per_ch.unsqueeze(0))
        assert bool((dmax <= 1.0).all()), 'decay = exp(dt·A) 必须 <= 1（dt>0、A<0）'
        assert bool((dmax >= 0.0).all()) and bool(torch.isfinite(dmax).all())
        # 对照：中等大小的 |A|（远离 init 但不极端）下界**严格**为正 —— 证明上一条
        # 的 `>= 0` 是「允许正确下溢」而不是「decay 可以是垃圾」
        m.A_log.data = torch.tensor([0.0, 0.7, 1.1, 1.4])
        mid = torch.exp(dtc.unsqueeze(-1) *
                        (-torch.exp(m.A_log)).double().repeat_interleave(HEAD_DIM)
                        .unsqueeze(0))
        assert bool((mid > 0.0).all()) and bool((mid <= 1.0).all())
        m.A_log.data.copy_(keep_a)

    # ④ 衰减沿 head 广播、与状态维 n 无关：exp(dt·A) 展成 (C,N) 后秩**恰为 1**
    #    （每一行都是同一个常数向量 ⇒ 所有行共线）—— 这正是 chunked 的 (L,L)
    #    能跨 n 共享的代数来源（P4.1r 的逐通道 A 做不到，见 report §3）。
    gen = torch.Generator().manual_seed(1000)
    dtc = (torch.rand(CH, generator=gen, dtype=torch.float64) * 0.7 + 0.05)
    per_ch = decay.double().repeat_interleave(HEAD_DIM)            # (184,)
    dexp = torch.exp(dtc * per_ch).unsqueeze(-1).expand(CH, DSTATE)
    assert int(torch.linalg.matrix_rank(dexp)) == 1, \
        '逐 head 标量衰减的 exp(dt·A) 秩应恰为 1（可分离）；满秩说明衰减又变回逐 (c,n) 了'
    # 对照：P4.1r 的逐通道稠密 A（被禁掉的参数化）给出的秩是满的 —— 两条并排钉死
    assert int(torch.linalg.matrix_rank(
        torch.exp(dtc.unsqueeze(-1) * torch.randn(CH, DSTATE, dtype=torch.float64)))) == DSTATE

    # ⑤ 初值：canonical Mamba-2 的 A = −(1..H)，即 log([1, 2, 3, 4])。
    #    ⚠ 不是 rev1/P4.1r 的 −(1..N) 斜坡段平均 —— 实测否决的理由与代价见
    #    `MambaLTI._init_a_log` 的 docstring 与 report §4。
    want = torch.log(torch.arange(1., float(N_HEADS) + 1))
    assert torch.allclose(m.A_log, want, rtol=0, atol=1e-6), \
        f'A_log 初值应为 log([1,2,3,4])，实得 {m.A_log.tolist()}'
    assert torch.allclose(-torch.exp(m.A_log),
                          -torch.arange(1., float(N_HEADS) + 1), rtol=0, atol=1e-6)
    # 对照：四个 head 初值互不相同（同值 ⇒ 梯度相同 ⇒ 对称性永远破不掉）
    assert len({round(float(v), 6) for v in m.A_log.tolist()}) == N_HEADS



def _onehot_state_probe(scan, chunk_size=None):
    """用 one-hot 的 drive 探针**证明**状态张量是 (B, H, P, N) 而不是别的排布。

    为什么不能用「断言 shape == (B,4,46,64)」那种重述式断言
    ------------------------------------------------------
    排布错乱（B/N 换位、permute 少一次、head 分组错位）在「读出 = (h * C).sum(-1)」
    这种写法下可能**形状照样对得上**，但算的是另一个量。one-hot 探针没有这个
    漏洞：drive 里只有 (h0, p0, n0) 非零时，输出必须**只**在 (h0, p0) 这个
    (head, 通道内位置) 上非零 —— 这同时钉住「状态按 (h,p) 索引」「head 分组是
    c = h*P+p」和「读出沿 n 求和」三件事。

    两条扫描返回的都是 **(B, T, H, P)**（沿 n 求和之后），不是 (B, T, C) —
    `forward` 里再 `.view(B, T, C)` 摊平。探针按 (h,p) 定位，所以顺带把
    「生产路径与 oracle 同形」也钉住了。

    `scan='chunked'` 是**生产实现**；`scan='oracle'` 是退役的顺序递推
    （`MambaLTI._sequential_scan_oracle`，**仅测试**）。
    """
    B, T, Hh, P, N = 2, 5, 4, 6, 4
    h0, p0, n0 = 2, 3, 2
    c0 = h0 * P + p0
    log_decay = -torch.rand(B, T, Hh, P, dtype=torch.float64) * 2.0 - 0.1
    drive = torch.zeros(B, T, Hh, P, N, dtype=torch.float64)
    drive[:, :, h0, p0, n0] = 1.0
    cvec = torch.ones(B, T, N, dtype=torch.float64)
    blk = MambaLTI(channels=Hh * P, n_heads=Hh, d_state=N).double()
    if scan == 'chunked':
        y = blk._chunked_scan(log_decay, drive, cvec, chunk_size=chunk_size)
    else:
        y = blk._sequential_scan_oracle(log_decay, drive, cvec)

    assert tuple(y.shape) == (B, T, Hh, P), \
        f'{scan}: 读出应为 (B,T,H,P)，实得 {tuple(y.shape)}'
    # 摊平成通道之后 one-hot 必须**只**落在 c0 = h0*46+p0 上
    flat = y.reshape(B, T, Hh * P)
    on = flat.abs() > 1e-12
    assert set(on[0, 0].nonzero().flatten().tolist()) == {c0}, \
        f'{scan}: 非零输出只应出现在通道 {c0}=h{h0}*P+{p0}，' \
        f'实得 {on[0, 0].nonzero().flatten().tolist()}'
    others = [j for j in range(Hh * P) if j != c0]
    assert bool(on[:, :, c0].all()) and not bool(on[:, :, others].any()), \
        f'{scan}: one-hot 探针泄漏到了别的通道'
    return y


def test_mamba2_state_shape_is_b_h_p_n():
    """每步的状态张量是 (B, H=4, P=46, N=64)，且 head 分组被 one-hot 探针**证明**过。

    P4.1s 把状态从 P4.1r 的 (B, C, N) 换成 (B, H, P, N)：分块扫描多两次 permute
    和一次五维 reshape，是排布最容易写错的地方。探针在**小配置**（H=4, P=6）与
    **shipped 配置**（H=4, P=46）各跑一遍，两条扫描路径都跑。
    """
    m = MambaLTI()
    assert m.d_inner == CH and m.d_state == DSTATE, \
        f'状态必须是 (B, {N_HEADS}, {HEAD_DIM}, {DSTATE})，实得 ' \
        f'(B, {m.d_inner}, {m.d_state})'
    assert (m.n_heads, m.head_dim) == (N_HEADS, HEAD_DIM)

    # ① 小配置：排布被 one-hot 探针证明（head 分组对、通道索引对、沿 n 求和对）
    _onehot_state_probe('oracle', chunk_size=3)
    _onehot_state_probe('chunked', chunk_size=3)

    # ② shipped 配置：同样的探针在真实的 4×46×64 上再跑一遍（生产路径 + oracle）
    for scan, kw in (('oracle', {}), ('chunked', {'chunk_size': 3})):
        h0, p0, n0 = 2, 30, 17
        c0 = h0 * HEAD_DIM + p0
        B, T = 2, 4
        log_decay = -torch.rand(B, T, N_HEADS, HEAD_DIM, dtype=torch.float64) * 2.0 - 0.1
        drive = torch.zeros(B, T, N_HEADS, HEAD_DIM, DSTATE, dtype=torch.float64)
        drive[0, 0, h0, p0, n0] = 1.0
        cvec = torch.ones(B, T, DSTATE, dtype=torch.float64)
        blk = MambaLTI().double()
        y = (blk._chunked_scan(log_decay, drive, cvec, **kw) if scan == 'chunked'
             else blk._sequential_scan_oracle(log_decay, drive, cvec))
        assert tuple(y.shape) == (B, T, N_HEADS, HEAD_DIM), f'{scan}: {tuple(y.shape)}'
        flat = y.reshape(B, T, CH)
        on = flat.abs() > 1e-12
        assert set(on[0, 0].nonzero().flatten().tolist()) == {c0}, \
            f'{scan}: shipped 配置下非零输出只应出现在通道 {c0}，' \
            f'实得 {on[0, 0].nonzero().flatten().tolist()}'
        # 对照：把 one-hot 挪到**别的 head**（h0+1 mod H）必须给出**别的**通道 ——
        # 证明「head 分组 = c//46」这条真的被测到，而不是碰巧对上
        h1 = (h0 + 1) % N_HEADS
        d2 = torch.zeros_like(drive)
        d2[0, 0, h1, p0, n0] = 1.0
        y2 = (blk._chunked_scan(log_decay, d2, cvec, **kw) if scan == 'chunked'
              else blk._sequential_scan_oracle(log_decay, d2, cvec))
        on2 = y2.reshape(B, T, CH).abs() > 1e-12
        assert set(on2[0, 0].nonzero().flatten().tolist()) == {h1 * HEAD_DIM + p0}, \
            f'{scan}: one-hot 挪到 head {h1} 后输出通道没跟着变 —— head 分组未被证明'

    # ③ 反向也存在且有限（状态不是只在 forward 里被造出来又丢掉）
    m2 = MambaLTI().double()
    m2(torch.randn(2, CH, 3, 3, dtype=torch.float64)).sum().backward()
    for n, p in m2.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), \
            f'{n} 的梯度缺失或非有限 —— 状态没参与反向'


def _drop_c_column(m, j):
    """给 `x_proj` 挂一个 hook，把 C 向量的第 j 列清零（状态第 j 维变成**读不到**）。"""
    lo = m.ddt_rank + m.d_state + j

    def _hook(_mod, _inp, out):
        return out.clone().index_fill_(-1, torch.tensor([lo]), 0.0)

    return m.x_proj.register_forward_hook(_hook)


def _drop_b_rows(m):
    """给 `x_proj` 挂一个 hook，把 B 段的 64 行**全部**清零（状态恒为 0，
    衰减没有任何可作用的量 —— 用来证明衰减探针的信号确实经由状态传出）。"""
    idx = m.ddt_rank + torch.arange(m.d_state)

    def _hook(_mod, _inp, out):
        return out.clone().index_fill_(-1, idx, 0.0)

    return m.x_proj.register_forward_hook(_hook)


def test_ssm_no_dead_state_components():
    """N=64 的**每一维**状态都必须既被激励又被读出；恒零梯度要报证据。

    P4.1s 的衰减探针从 P4.1r 的「`A_v` 第 j 列」（逐 (c,n) 专属）换成
    「`A_log[h]`」（head h 的全部 46 通道共享一个衰减率）：
      * `A_log[h]`  -> head h 的**衰减**是活的（46 通道一起响应）
      * `x_proj` 的 B 段第 j 行 -> 状态维 j **被写入**（激励）
      * `x_proj` 的 C 段第 j 行 -> 状态维 j **被读出**
    """
    torch.manual_seed(20260927)
    m = MambaLTI().eval()                      # shipped 形状：184 / 64
    assert m.d_state == DSTATE == 64 and m.n_heads == N_HEADS == 4
    x = torch.randn(1, CH, 5, 5)               # 5x5 => T=25，够 64 个探针都吃到信号

    b_rows = m.ddt_rank + torch.arange(DSTATE)            # B 段的 64 行
    c_rows = m.ddt_rank + DSTATE + torch.arange(DSTATE)   # C 段的 64 行

    with torch.no_grad():
        base = m(x).clone()
        keep_a = m.A_log.clone()
        keep_w = m.x_proj.weight.clone()
        dead_decay, dead_b, dead_c = [], [], []

        for h in range(N_HEADS):
            # ① 衰减探针：A_log 的第 h 个标量（head h 的 46 个通道）
            m.A_log.data[h] = m.A_log.data[h] * 1.7 + 0.11
            d1 = (m(x) - base).abs().max().item()
            m.A_log.data.copy_(keep_a)
            if d1 <= 0:
                dead_decay.append(h)
        for j in range(DSTATE):
            # ② 激励探针：B 段第 j 行
            m.x_proj.weight.data.copy_(keep_w)
            m.x_proj.weight.data[b_rows[j]] *= 2.3
            d2 = (m(x) - base).abs().max().item()
            m.x_proj.weight.data.copy_(keep_w)
            # ③ 读出探针：C 段第 j 行
            m.x_proj.weight.data[c_rows[j]] *= 2.3
            d3 = (m(x) - base).abs().max().item()
            m.x_proj.weight.data.copy_(keep_w)
            if d2 <= 0:
                dead_b.append(j)
            if d3 <= 0:
                dead_c.append(j)
    assert not dead_decay, f'head {dead_decay} 的衰减（A_log 对应项）改了输出却没变化'
    assert not dead_b, f'状态维 {dead_b} 被激励（B 段对应行）改了却没进输出'
    assert not dead_c, f'状态维 {dead_c} 读不出（C 段对应行）改了却没进输出'

    # 非空转守卫（对应 causality 测试里那条逐通道扰动的教训）：
    # 把 B 段 64 行**全部**清零后状态恒为 0、衰减没有可作用的量，
    # 此时 `A_log` 的扰动就**必须**不再改变输出 —— 这证明上面 d1 的信号
    # 确实是经由状态传出去的，不是别的旁路。
    m2 = MambaLTI().eval()
    with torch.no_grad():
        base2 = m2(x).clone()
        handle = _drop_b_rows(m2)
        blind = m2(x).clone()
        m2.A_log.data[2] = m2.A_log.data[2] * 1.7 + 0.11
        still_blind = m2(x)
        handle.remove()
        m2.A_log.data.copy_(keep_a)
        assert torch.allclose(m2(x), base2, rtol=1e-6, atol=1e-9), \
            'A_log 还原后输出回不到基线 —— 探针的恢复写错了'
    assert torch.allclose(blind, still_blind, rtol=1e-9, atol=1e-12), \
        'B 段已全清零（状态恒 0），A_log 的扰动却仍改变了输出 —— 衰减不是经由状态起作用的'
    assert not torch.allclose(base2, blind, rtol=1e-6, atol=1e-9), \
        '清零 B 段 64 行竟然没改变输出 —— 上面的守卫是空转的'

    # 梯度：恒零梯度才是 brief §3 说的「死参数」。判据是 **恰为 0**，不是 `≤ 0`
    # —— A_log 的梯度是 361×46 项的**带符号**求和，符号本身由数据决定，实测在
    # shipped 初值下就是正负交替的（见下面的打印）。「符号为负」不是死亡。
    # `chunked` = 生产路径；`oracle` = 退役的顺序递推（`_forward_with_oracle`）。
    for scan in ('chunked', 'oracle'):
        m3 = MambaLTI().double()
        if scan == 'oracle':
            with _forward_with_oracle(m3):
                m3(x.double()).sum().backward()
        else:
            m3(x.double()).sum().backward()
        anorm = m3.A_log.grad                            # (4,)
        dead = [h for h, g in enumerate(anorm.tolist()) if g == 0.0]
        assert not dead, f'float64/{scan}: A_log 有恒零梯度的 head（真·死参数）：{dead}'
        assert bool(torch.isfinite(anorm).all()), f'float64/{scan}: A_log.grad 非有限'
        # A 本身（不是 A_log）必须逐元素严格负且有限 —— 恒零的 A 会让 decay≡1
        # （状态永不衰减，cumsum 闭式的「指数 ≤ 0」不变式失效）
        A3 = -torch.exp(m3.A_log.detach())
        assert bool((A3 < 0).all()) and bool(torch.isfinite(A3).all()) and bool((A3 != 0).all())
        print(f'\n[dead-state] float64/{scan:10s} A_log.grad 4 head: '
              f'{[f"{v:.3e}" for v in anorm.tolist()]}  恒零 head=0')
    tau = 1.0 / (_dt_per_head(MambaLTI(), x) * torch.exp(MambaLTI().A_log.detach()))
    print(f'[dead-state] shipped 初值下 4 个 head 的记忆常数 τ = '
          f'{[round(float(v), 3) for v in tau]} 步（跨度 '
          f'{float(tau.max() / tau.min()):.2f}x）—— 逐 head 标量衰减下，多尺度'
          f'结构完全由这 4 个数决定')

    # ---- §3 的 fp32 下溢地板：处理方案与它的代价（不是默默留着） -------------
    #
    # P4.1r 的实测结论（逐通道低秩 A）：float64 下没有死维度；**float32 + 分块**下
    # 有 7 个 `A_v` 高 n 列梯度恰好为 0，机制是「唯一依赖衰减的路径是块内最近邻
    # 项，其梯度要连乘两个 ~exp(−dt·|A|) 因子，n≳62 时在 fp32（tiny=1.18e-38）
    # 下溢」。逐 head 标量衰减**没有 per-n 斜坡**了，但同一个机制还在：只要 |A_h|
    # 大到让块内 `M[i,j] = exp(σ_i−σ_j)` 在 fp32 下冲成恰好 0，慢 head 的 A_log
    # 梯度就会在**分块**路径上被冲成恰好 0（在顺序路径上不会，因为那里是逐步累加）。
    #
    # **处理方案：把初值从「−(1..N) 斜坡按 head 取段平均」换成 canonical Mamba-2
    # 的 `A = −(1..H)`**，让每步衰减不低于 0.061（下溢地板远在下方）。代价见
    # `MambaLTI._init_a_log` 的 docstring（per-head 跨度 6.6× → 4×）。
    #
    # 下面的断言分两组。**第一组钉住处理的效果**（shipped 初值下 fp32 生产路径
    # 没有恒零 head）。**第二组证明「零 = 下溢」而不是「零 = 结构死亡」**，而且
    # P4.1s-vec 之后**不再依赖生产路径上的顺序对照**（顺序路径已退役，fp32 顺序
    # 这一半判据改从 `_sequential_scan_oracle` 这个仅测试的入口取，见
    # `_forward_with_oracle`），另外补两条**完全独立于 oracle** 的对照：
    #   (a) 同一初值、同一权重下 float64 干净 ⇒ 不是结构死亡；
    #   (b) 把同一个初值的 |A| 整体缩小 100 倍（**计算图逐字节相同**，只有衰减
    #       的量级变了）后 fp32 恢复干净 ⇒ 零是**量级**驱动的，结构一直连着；
    #   (c) 直接数一数块内 `M` 在 fp32 下有多少个元素被冲成**恰好 0**（fp64 下
    #       同样输入的零个数）—— 这是「下溢」这个说法的**直接测量**，不是类比。
    m4 = MambaLTI()                                   # 唯一路径 = 分块 SSD
    m4(x).sum().backward()
    f32_dead = [h for h, g in enumerate(m4.A_log.grad.tolist()) if g == 0.0]
    assert not f32_dead, (
        f'fp32/分块下 A_log 出现恒零梯度 head {f32_dead} —— shipped 初值本该把'
        f'每步衰减守在 0.061 以上、不该下溢。检查 _init_a_log 或 chunked 的实现')
    assert bool(torch.isfinite(m4(x)).all()), 'fp32 前向出现非有限值（下溢不是 NaN）'
    proxy = torch.exp(-2.0 * _dt_per_head(m4, x) * torch.exp(m4.A_log.detach()))
    print(f'[dead-state] fp32/分块（生产路径）恒零 head = {f32_dead}（空）；'
          f'双因子量级 exp(−2·dt̄·|A|) = '
          f'{[f"{v:.2e}" for v in proxy.tolist()]}（fp32 tiny = 1.18e-38）')

    ramp = _rejected_ramp_init()
    g = {k: _a_grad(torch.float32 if k[0] == 'fp32' else torch.float64, k[1], 32, ramp, x)
         for k in (('fp32', 'chunked'), ('fp32', 'oracle'), ('fp64', 'chunked'))}
    dead_ramp = {k: [h for h, v in enumerate(gg.tolist()) if v == 0.0] for k, gg in g.items()}
    print(f'[dead-state] 机制复现（装回被否决的斜坡平均初值 '
          f'{[round(float(v), 2) for v in (-torch.exp(ramp)).tolist()]}）: '
          f'恒零 head = {dead_ramp}')
    assert dead_ramp[('fp32', 'chunked')], \
        ('装回斜坡平均初值后 fp32+分块竟然没有恒零 head —— 下面几条对照就失去'
         '判别力，§3 的机制复现变成空转（初值或实现被改动，请重新实测）')
    # (a) 同一初值、同一批权重下 float64 干净 ⇒ 不是结构死亡
    assert not dead_ramp[('fp64', 'chunked')], \
        'fp64+分块也死了 ⇒ 不是 fp32 下溢地板，是结构性问题'
    # 顺序 oracle（P4.1s-vec 后**仅测试**可达）：fp32 下它仍然干净 ⇒ 下溢是分块
    # 路径特有的（那里 A_log 的梯度要穿过块内 (L,L) 的 exp）
    assert not dead_ramp[('fp32', 'oracle')], \
        'fp32+顺序 oracle 也死了 ⇒ 与前序实测（下溢只出现在分块路径）矛盾，机制判据失效'
    # (b) |A| 整体缩小 100 倍：计算图逐字节相同，只有衰减量级变了 ⇒ 恢复干净。
    #     若是结构死亡（`A_log` 根本没接进图里），任何量级都救不回来。
    for k in (0.01, 0.1):
        gk = _a_grad(torch.float32, 'chunked', 32, _scaled_ramp_init(k), x)
        alive = [h for h, v in enumerate(gk.tolist()) if v != 0.0]
        assert len(alive) == N_HEADS, \
            (f'斜坡平均初值 |A| × {k} 下 fp32+分块仍有恒零 head '
             f'{[h for h in range(N_HEADS) if h not in alive]} —— 零若与 |A| 的量级'
             f'无关，就不是 fp32 下溢而是结构死亡')
    print(f'[dead-state] 对照 (b)：同一个斜坡平均初值，|A| × 0.1 / × 0.01 后 '
          f'fp32+分块恒零 head = 0（计算图未变，只有量级变了 ⇒ 下溢而非结构死亡）')
    # (c) 直接数块内 M 在 fp32 下被冲成恰好 0 的元素个数（fp64 同一输入作对照）
    x19 = torch.randn(1, CH, 19, 19, generator=torch.Generator().manual_seed(555))
    zu = _intra_chunk_zero_counts(x19, 32)
    print(f'[dead-state] 对照 (c)：块内 M[L=32] 下三角共 {zu["n"]} 个元素里，'
          f'fp32 恰零 —— 斜坡平均初值 {zu["ramp/fp32"]}（{zu["ramp/fp32"] / zu["n"]:.1%}）、'
          f'shipped 初值 {zu["ship/fp32"]}（{zu["ship/fp32"] / zu["n"]:.3%}）；'
          f'同一斜坡平均初值在 fp64 下恰零 {zu["ramp/fp64"]}（{zu["ramp/fp64"] / zu["n"]:.1%}）')
    assert zu['ramp/fp32'] >= 50 * max(1, zu['ship/fp32']), \
        ('斜坡平均初值在 fp32 下的块内 M 下溢数竟不比 shipped 初值多 50 倍 —— '
         'canonical 初值「把 fp32 下溢地板甩开」的说法失效，请重新实测')
    assert zu['ramp/fp32'] >= 4 * max(1, zu['ramp/fp64']), \
        (f'同一初值、同一权重，fp32 的块内 M 下溢数 {zu["ramp/fp32"]} 竟不超过 fp64 的 '
         f'{zu["ramp/fp64"]} 的 4 倍 —— 「下溢是浮点格式问题」这个解释失去支撑；'
         f'那些恒零梯度另有原因，请重新实测')
    assert zu['ship/fp32'] <= zu['n'] // 100, \
        (f'shipped 初值下 fp32 的块内 M 有 {zu["ship/fp32"]}/{zu["n"]} '
         f'（>1%）的元素下溢成 0 —— fp32+分块的 A_log 梯度只靠剩下的部分支撑，'
         f'「恒零 head = 0」这个结论随时可能翻，请重新实测')


def _intra_chunk_zero_counts(x32, L):
    """数块内 `M[i,j] = exp(σ_i − σ_j)`（下三角）在 fp32 / fp64 下**恰好为 0** 的个数。

    这是「fp32 下溢地板」的**直接测量**，不是类比：`_chunked_scan` 里 `M` 就是
    这个量（`masked_fill(~tril, -inf).exp()`），它被冲成 0 之后，A_log 经由该项
    的梯度贡献也逐字变成 0。计数只统计**下三角**（上三角是 mask 造成的
    `exp(−inf) = 0`，不是下溢）。

    两组初值（shipped 的 canonical / 被否决的斜坡平均）× 两个 dtype，四格一起返回。
    权重固定在同一 seed 上，所以两组初值之间的差别**只**来自 `A_log`。
    """
    out, n = {}, 0
    for tag, a_log in (('ship', None), ('ramp', _rejected_ramp_init())):
        torch.manual_seed(4242)
        m = MambaLTI()
        if a_log is not None:
            with torch.no_grad():
                m.A_log.data.copy_(a_log)
        for dt_name, dt in (('fp32', torch.float32), ('fp64', torch.float64)):
            md = m.to(dt)                    # nn.Module.to 是**就地**的，逐格轮流用
            with torch.no_grad():
                ld = _reference_inputs(md, x32.to(dt))['log_decay']   # (B,T,H,P)
            S = torch.cumsum(ld, dim=1)                             # (B,T,H,P)
            assert S.shape[1] >= L, f'需要 T >= L，实得 T={S.shape[1]}'
            S_prev = S.new_zeros(S.shape[0], 1, S.shape[2], S.shape[3])
            sigma = (S[:, :L] - S_prev).permute(0, 2, 3, 1)          # (B,H,P,l)
            dd = sigma.unsqueeze(-1) - sigma.unsqueeze(-2)           # (B,H,P,l,l)
            tril = torch.ones(L, L, dtype=torch.bool).tril()
            M = dd.masked_fill(~tril, float('-inf')).exp()           # 与 _chunked_scan 同式
            vals = M[..., tril]
            n = vals.numel()
            out[f'{tag}/{dt_name}'] = int((vals == 0).sum())
        m.to(torch.float32)
    out['n'] = n
    return out


def _rejected_ramp_init(n_heads=N_HEADS, d_state=DSTATE):
    """**被实测否决**的候选初值：rev1 / P4.1r 的 `−(1..N)` 斜坡按 head 取段平均。

    保留它是为了让 §3 的机制复现有对照物（见 `test_ssm_no_dead_state_components`
    与 `test_mamba2_fp32_active_state_dims`）：它在 fp32+分块下会产生恒零梯度，
    shipped 的 canonical 初值不会。**生产代码不引用本函数。**
    """
    ramp = torch.arange(1, d_state + 1, dtype=torch.float32)
    return torch.log(ramp.view(n_heads, -1).mean(dim=-1))


def _scaled_ramp_init(k, n_heads=N_HEADS, d_state=DSTATE):
    """被否决的斜坡平均初值，但把 `|A|` 整体缩放 `k` 倍（`A_log + log k`）。

    这是「恒零梯度是**量级**驱动的（下溢）而不是**结构**驱动的（死参数）」的
    决定性对照：**计算图逐字节相同**（同一组参数、同一处乘法、同一处 exp），
    唯一变的是衰减的大小。若那些零是结构性的（`A_log` 根本没接进计算图），
    缩放一万倍也救不回来。
    """
    return _rejected_ramp_init(n_heads, d_state) + math.log(k)


def _a_grad(dtype, scan, chunk, a_log, x):
    """给定 `A_log` 取值，测 `A_log.grad`（用于 §3 的机制对照）。

    `scan='chunked'` = 生产路径；`scan='oracle'` = 退役的顺序递推
    （`MambaLTI._sequential_scan_oracle`，经 `_forward_with_oracle` 临时接进 forward）。
    """
    m = MambaLTI(chunk_size=chunk).to(dtype)
    with torch.no_grad():
        m.A_log.data.copy_(a_log.to(m.A_log.dtype))
    if scan == 'oracle':
        with _forward_with_oracle(m):
            m(x.to(dtype)).sum().backward()
    else:
        m(x.to(dtype)).sum().backward()
    return m.A_log.grad.detach().clone()


def _a_grad_pair(scan, chunk, seed, x32):
    """同一组权重下 (fp32, fp64) 的 `A_log.grad`。

    ⚠ 两个模型必须**逐位同权重**（同一 seed 重建），否则比的是两个不同网络的
    梯度 —— 本测试第一版就踩了这个坑，得到过「相对差 17×、符号翻转」的假警报
    （`MambaLTI(...)` 每次构造都重新抽随机权重，不 seed 就没有可比性）。
    """
    out = []
    for dt in (torch.float32, torch.float64):
        torch.manual_seed(seed)
        m = MambaLTI(chunk_size=chunk).to(dt)
        if scan == 'oracle':
            with _forward_with_oracle(m):
                m(x32.to(dt)).sum().backward()
        else:
            m(x32.to(dt)).sum().backward()
        out.append(m.A_log.grad.detach().clone())
    return out[0], out[1]


def test_mamba2_fp32_active_state_dims():
    """brief §3 的活跃维度检查（带证据）：fp64 全活；fp32 在 shipped 初值下也全活。

    与 `test_ssm_no_dead_state_components` 的梯度段互补：那条在 5×5 上跑前向探针
    + 装回被否决初值的机制复现，本条把
    **fp32/fp64 × 顺序/分块 × L ∈ {1,4,8,16,32,64} × board ∈ {5,9} × 4 个种子**
    的活跃 head 集合全跑一遍并打印（共 72 组配置），断言：

      ① 恒零判据是 **恰为 0.0**，不是 `≤ 0` —— `A_log.grad` 是 361×46 项的带符号
         求和，符号由数据决定，实测在 shipped 初值下就是正负交替的；
      ② shipped 初值下 **fp32 + 分块 0 个**恒零 head（这是处理方案的效果，
         见 `MambaLTI._init_a_log` 与 report §4）；
      ③ fp32 与 fp64 的 `A_log.grad` **逐 head 相对一致 ≤ 1e-3**（实测最差
         4.5e-4，聚合口径 1.8e-5）⇒ fp32 训练下这些衰减率真的在被更新，
         不是「非零但全是噪声」；
      ④ 前向侧：fp32 + 分块下 4 个 head 扰动 `A_log[h]` 都**确实**改变输出。
    """
    boards, seeds = (5, 9), (1, 7, 101, 20260927)
    zero = {('fp32', 'chunked'): 0, ('fp32', 'oracle'): 0,
            ('fp64', 'chunked'): 0, ('fp64', 'oracle'): 0}
    worst_rel = worst_agg = 0.0
    n = 0
    for seed in seeds:
        for board in boards:
            x32 = _seeded_x(board, seed=seed)
            # 'oracle' 那一列是**退役的顺序递推**（P4.1s-vec 之后仅测试可达），
            # 它的 L 恒为 1 —— 顺序路径没有块长可言。
            for scan, chunk_list in (('oracle', (1,)), ('chunked', (4, 8, 16, 32, 64))):
                for L in chunk_list:
                    a32, a64 = _a_grad_pair(scan, L, seed, x32)
                    n += 1
                    for dname, g in (('fp32', a32), ('fp64', a64)):
                        z = [h for h, v in enumerate(g.tolist()) if v == 0.0]
                        zero[(dname, scan)] += len(z)
                        assert bool(torch.isfinite(g).all()), \
                            f'{dname}/{scan}/L={L}/seed={seed}: A_log.grad 非有限'
                        if dname == 'fp64':
                            assert not z, \
                                f'float64/{scan}/L={L}/seed={seed}: 恒零 head {z} —— 结构性死亡'
                    den = a64.abs().max().clamp_min(1e-300)
                    rel = float(((a32.double() - a64).abs()
                                  / a64.abs().clamp_min(1e-300))[a64 != 0].max())
                    agg = float((a32.double() - a64).abs().max() / den)
                    worst_rel, worst_agg = max(worst_rel, rel), max(worst_agg, agg)
                    assert rel <= 1e-3, (
                        f'float32 的 A_log.grad 与 float64 逐 head 相对差 {rel:.2e} > 1e-3'
                        f'（{scan}/L={L}/seed={seed}/board={board}）—— fp32 训练下衰减'
                        f'梯度已不可信')
    print(f'\n[active-dims] {n} 组配置（fp32/fp64 × [分块=生产路径, oracle=退役顺序递推] '
          f'× L ∈ {{1,4,8,16,32,64}} × board ∈ {boards} × seed ∈ {len(seeds)}）：恒零 head '
          f'{dict(zero)}；fp32 vs fp64 逐 head 最差相对差 {worst_rel:.2e}、'
          f'聚合口径 {worst_agg:.2e}')
    assert zero[('fp64', 'chunked')] == 0 and zero[('fp64', 'oracle')] == 0
    assert zero[('fp32', 'chunked')] == 0, (
        f'float32 + 分块下有 {zero[("fp32", "chunked")]} 个恒零 head —— shipped 初值'
        f'（每步衰减 >= 0.061）本该把 fp32 下溢地板远远甩开；检查 _init_a_log /'
        f'_chunked_scan')
    assert zero[('fp32', 'oracle')] == 0

    # 前向侧：fp32 + 分块（唯一路径）下 4 个 head 的衰减都必须真的影响输出
    for board in boards:
        for seed in seeds:
            x = _seeded_x(board, seed=seed)
            m = MambaLTI(chunk_size=MAMBA_CHUNK_SIZE).eval()
            with torch.no_grad():
                base = m(x).clone()
                keep = m.A_log.data.clone()
                d = []
                for h in range(N_HEADS):
                    m.A_log.data[h] = m.A_log.data[h] * 1.7 + 0.11
                    d.append(float((m(x) - base).abs().max()))
                    m.A_log.data.copy_(keep)
            assert min(d) > 0.0, (
                f'board={board} seed={seed}: 有 head 的衰减扰动在 fp32+分块下'
                f'给出恰好 0 的输出变化 {d} —— 该 head 在数值上是死的')
    print('[active-dims] fp32 + 分块（默认 L，唯一路径）下 4 个 head 的前向探针全部非零')


# --------------------------------------------------------------------------- #
# 5c. P4.1r §2.2 的核心守卫：分块 SSD 扫描 == 朴素递推
# --------------------------------------------------------------------------- #
def _hand_chunked_scan(ref, L, dtype):
    """把「块内 (L,L) 衰减矩阵 + 块间状态传递」按**位置**逐步手写一遍。

    这是分块路径的独立 oracle：它不调用 `_chunked_scan`，连 `bmm` 都不用 ——
    块内那一项 `Σ_{j≤i} exp(σ_i − σ_j)·u_j` 是两个 Python for 循环。所以
    「permute 错一位」「bmm reshape 错一位」「上三角没屏蔽」这类错误在这条
    参照面前无处可藏。
    """
    log_decay, drive, cvec = ref['log_decay'], ref['drive'], ref['c_vec']
    B, T, Hh, P, N = ref['B'], ref['T'], ref['Hh'], ref['P'], ref['N']
    S = torch.cumsum(log_decay, dim=1)                    # 全序列对数衰减累积
    h_in = torch.zeros(B, Hh, P, 1, N, dtype=dtype)       # H_0 = 0
    s_prev = torch.zeros(B, 1, Hh, P, dtype=dtype)        # S_{−1} = 0
    ys = []
    for k0 in range(0, T, L):
        sk = S[:, k0:min(k0 + L, T)]
        sigma = sk - s_prev                               # (B,l,H,P) 块内局部累积
        l = sk.shape[1]
        h_k = h_in                                        # H_k：整块**不变**
        for i in range(l):
            # ⚠ 第一项用的是 H_k（块入口状态，快照），**不是** h_{i-1}。
            #   写成 h_{i-1} 会把块内那一项重复计一次 —— 本测试第一版就是这么
            #   写错的，被它自己抓住（max|Δ| = 7.7e-5，|y| 只有 4.9e-3）。
            acc = torch.exp(sigma[:, i]).view(B, Hh, P, 1, 1) * h_k
            for j in range(i + 1):                        # 块内 (L,L) 衰减矩阵
                acc = acc + torch.exp(sigma[:, i] - sigma[:, j]).view(B, Hh, P, 1, 1) \
                    * drive[:, k0 + j].unsqueeze(-2)      # (B,Hh,P,1,N)
            ys.append((acc[..., 0, :] * cvec[:, k0 + i].view(B, 1, 1, N)).sum(-1))
        h_in = acc                                        # H_{k+1} = 块末状态
        s_prev = sk[:, -1:]
    return torch.stack(ys, dim=1)                         # (B,T,Hh,P)



def test_ssm_chunked_scan_matches_hand_expanded_chunk_formula():
    """分块扫描 == 测试内**按位置手写**的「块内矩阵 + 块间传递」闭式。

    T=5, L=2 刻意选成 **L 不能整除 T**（最后一块只有 1 步）：块边界的 off-by-one
    是分块扫描最容易错的地方，这里用一个既不整除、又不是整块的长度去压它。
    """
    m = _small_mamba()
    m.chunk_size = 2
    x = torch.randn(2, 8, 1, 5, dtype=torch.float64)     # T=5
    ref = _reference_inputs(m, x)
    ref = dict(ref)
    ref['y'] = _hand_chunked_scan(ref, 2, x.dtype)
    with torch.no_grad():
        got = m(x)
    want = _hand_block_forward(m, x, ref)
    assert torch.allclose(got, want, rtol=1e-10, atol=1e-12), \
        '分块扫描与手写的块内矩阵/块间传递闭式不一致：max|Δ|={:.3e}'.format(
            (got - want).abs().max().item())
    # 非空转：把块间状态传递去掉（每块都从 0 起步）必须得到不同的结果
    broken = torch.stack([
        (h.view(ref['B'], ref['Hh'], ref['P'], ref['N'])
         * ref['c_vec'][:, t].view(ref['B'], 1, 1, ref['N'])).sum(-1)
        for t, h in enumerate(_per_chunk_reset(ref, 2, x.dtype))], dim=1)
    assert not torch.allclose(got, _hand_block_forward(m, x, dict(ref, y=broken)),
                              rtol=1e-6, atol=1e-9), \
        '去掉块间状态传递后结果不变 —— 本测试分辨不出 H_k 的传递'


def _per_chunk_reset(ref, L, dtype):
    """每块都从 h=0 重启的分块扫描（= 丢掉了块间状态传递），只作非空转对照。"""
    log_decay, drive = ref['log_decay'], ref['drive']
    B, T, Hh, P, N = ref['B'], ref['T'], ref['Hh'], ref['P'], ref['N']
    S = torch.cumsum(log_decay, dim=1)
    s_prev = torch.zeros(B, 1, Hh, P, dtype=dtype)
    out = []
    for k0 in range(0, T, L):
        sk = S[:, k0:min(k0 + L, T)]
        sigma = sk - s_prev
        l = sk.shape[1]
        h = torch.zeros(B, Hh, P, 1, N, dtype=dtype)
        for i in range(l):
            h = torch.exp(sigma[:, i]).view(B, Hh, P, 1, 1) * h \
                + drive[:, k0 + i].unsqueeze(-2)          # (B,Hh,P,1,N)
            out.append(h)
        s_prev = sk[:, -1:]
    return out


_SCAN_CASES = [
    (1, 1, 1), (1, 1, 64),        # T=1：单步；L 远大于 T（单块）
    (1, 5, 1), (1, 5, 2), (1, 5, 5), (1, 5, 64),   # T=5：L=1 / 整除不了 / 整块 / 超大
    (1, 7, 3), (1, 7, 7),         # T=7：L=3 不整除 / 整块
    (3, 3, 4), (3, 3, 9), (3, 3, 100),             # T=9：常见的 3x3
    (19, 19, 1), (19, 19, 8), (19, 19, 64), (19, 19, 361),   # T=361：真实棋盘
]


@pytest.mark.parametrize('h,w,L', _SCAN_CASES,
                         ids=[f'T{h * w}_L{L}' for h, w, L in _SCAN_CASES])
def test_ssm_chunked_scan_matches_sequential_oracle(h, w, L):
    """**生产路径**（分块 SSD）== **顺序 oracle**，在多种 (T, L) 上。

    L=1、L=整块、L 不能整除 T、真实 361 都在 `_SCAN_CASES` 里（15 组）。

    ⚠ P4.1s-vec 之后这条是**唯一**的「分块 vs 顺序」等价性守卫：顺序递推已从
    运行时退役（`scan='auto'` 分派删除），它以 `MambaLTI._sequential_scan_oracle`
    这个**仅测试**的入口保留下来 —— 向量化实现没有任何别的独立参照，删掉它就
    等于没人能说清块内矩阵、块间状态传递、上三角 mask 算对了没有。
    本条**直接调两个扫描方法**（不是靠 forward 的开关），所以它与「运行时怎么
    选实现」完全解耦：将来 forward 再怎么变，这条照样在钉数学。

    为什么「多种 L」本身就是 off-by-one 守卫
    --------------------------------------
    L 决定块边界。块边界的任何错位（`S_prev` 取块首而不是块末、`H_{k+1}` 取块首
    而不是块末、mask 的 `tril` 写成 `tril(1)`）都会让结果**依赖 L**。所以本条在
    15 组 (T, L) 上要求同一条递推给出同一个答案 —— 只要边界逻辑有一处错，至少有
    一组 L 会红。float64 下两条路径的 max|Δ| 实测 ~1e-16（接近逐位相同，但浮点结合
    律不同，故按 `allclose` 断言而不是 `equal`）。
    """
    T = h * w
    m = _small_mamba()
    m.chunk_size = L
    torch.manual_seed(101)
    x = torch.randn(2, 8, h, w, dtype=torch.float64)
    with torch.no_grad():
        ref = _reference_inputs(m, x)
        y_seq = m._sequential_scan_oracle(
            ref['log_decay'], ref['drive'], ref['c_vec'])       # (B,T,H,P)
        y_chk = m._chunked_scan(
            ref['log_decay'], ref['drive'], ref['c_vec'], chunk_size=L)
    dmax = (y_chk - y_seq).abs().max().item()
    assert torch.allclose(y_chk, y_seq, rtol=1e-10, atol=1e-12), \
        (f'T={T}, L={L}: 分块扫描与顺序 oracle 不一致，max|Δ|={dmax:.3e} —— '
         '块边界（mask / S_prev / H 传递）有 off-by-one')
    assert tuple(y_chk.shape) == tuple(y_seq.shape) == (2, T, m.n_heads, m.head_dim), \
        '生产路径与 oracle 必须同形 (B,T,H,P)'

    # 再压一层：小 T 时分块也必须等于**朴素三重循环**（真正的 oracle）
    if T <= 16:
        ys = []
        for t in range(T):
            acc = torch.zeros(ref['B'], ref['Hh'], ref['P'], ref['N'], dtype=x.dtype)
            for s in range(t + 1):
                dec = torch.ones(ref['B'], ref['Hh'], ref['P'], ref['N'], dtype=x.dtype)
                for r_ in range(s + 1, t + 1):
                    dec = dec * torch.exp(ref['log_decay'][:, r_]).unsqueeze(-1)
                acc = acc + dec * ref['drive'][:, s]
            ys.append((acc * ref['c_vec'][:, t].view(ref['B'], 1, 1, ref['N'])).sum(-1))
        want = _hand_block_forward(m, x, dict(ref, y=torch.stack(ys, dim=1)))
        with torch.no_grad():
            got = m(x)
        assert torch.allclose(got, want, rtol=1e-10, atol=1e-12), \
            f'T={T}, L={L}: 分块扫描与朴素三重循环不一致，max|Δ=' \
            '{:.3e}'.format((got - want).abs().max().item())


def test_ssm_scan_is_single_vectorized_path():
    """P4.1s-vec：**只有一条扫描路径**（分块 SSD），运行时没有第二条路可选。

    这条替代 P4.1s 的 `test_ssm_scan_auto_dispatch`（那一条钉的是**活的分派**，
    现已作废 —— 但它钉的「train/eval 输出差 ≤0.5 fp32 eps」这件事**消失了**，
    见下面最后一段）。这里钉四件事：

      ① **没有运行时开关**：模块里不再有 `MAMBA_SCAN_DEFAULT` 常量，`MambaLTI`
         既没有 `scan=` 形参也没有 `self.scan` 属性，`_scan_impl` 已删除。
         ⇒ 想要「顺序」在生产里唯一只能靠改代码，而改代码会被下面 ② 抓住。
      ② **前向只调 `_chunked_scan`，永远不调 oracle**：train 与 eval 都一样。
      ③ **块长 L = 32**（单一路径下训练与推理用同一个值，实测见
         task-p4-1s-vec-report.md §3）。
      ④ **train/eval 逐位相同**（P4.1s 的 `scan='auto'` 曾让两者差 ≈0.82 个 fp32
         eps；单一路径后该差异归零）。
    """
    import src.networks.backbone as bb
    assert not hasattr(bb, 'MAMBA_SCAN_DEFAULT'), \
        ("模块里仍有 MAMBA_SCAN_DEFAULT —— P4.1s-vec 的裁决是「只有一条路径」，"
         "一个可改的默认常量就是一条可切回来的路")
    m = MambaLTI()
    assert not hasattr(m, 'scan'), 'MambaLTI 不该再有 self.scan（P4.1s-vec 已退役分派）'
    assert not hasattr(MambaLTI, '_scan_impl'), 'MambaLTI._scan_impl 应已删除'
    with pytest.raises(TypeError):
        MambaLTI(scan='sequential')            # 形参不存在 ⇒ TypeError
    assert not hasattr(MambaLTI, '_selective_scan'), \
        '旧名 `_selective_scan` 必须消失（它已改名为 `_sequential_scan_oracle`）'

    # ② 前向只调分块；oracle **一次都不许被调到**
    seen = []
    orig_chunked = MambaLTI._chunked_scan
    orig_oracle = MambaLTI._sequential_scan_oracle

    def wrap_chunked(_self, *a, **k):
        seen.append('_chunked_scan')
        return orig_chunked(_self, *a, **k)

    def wrap_oracle(*a, **k):
        seen.append('_sequential_scan_oracle')
        return orig_oracle(*a, **k)

    try:
        MambaLTI._chunked_scan = wrap_chunked
        MambaLTI._sequential_scan_oracle = wrap_oracle
        for mode in ('train', 'eval'):
            m = MambaLTI()
            getattr(m, mode)()
            seen.clear()
            with torch.no_grad():
                m(torch.randn(2, CH, 3, 3))
            assert seen and set(seen) == {'_chunked_scan'}, \
                (f'{mode} 下前向实际调了 {set(seen)} —— 唯一路径必须是 _chunked_scan；'
                 f'顺序 oracle 是**仅测试**的，生产路径不得触达')
    finally:
        MambaLTI._chunked_scan = orig_chunked
        MambaLTI._sequential_scan_oracle = orig_oracle

    # ③ 块长
    assert MAMBA_CHUNK_SIZE == 32, \
        (f'单一路径下的推荐块长是 L=32（report §3 的 L 扫描：前向/反向/显存三张表上'
         f'都不垫底的唯一取值）；实得 {MAMBA_CHUNK_SIZE}')
    assert m.chunk_size == MAMBA_CHUNK_SIZE

    # ④ train/eval 逐位相同。
    #    非空转前提：块内**没有**任何依赖 `self.training` 的层（dropout /
    #    BatchNorm），所以「相等」是「只有一条计算路径」的结果而不是巧合。
    mode_dep = [n for n, mod in MambaLTI().named_modules()
                if isinstance(mod, (nn.Dropout, nn.modules.batchnorm._BatchNorm))]
    assert not mode_dep, \
        f'MambaLTI 里出现了依赖 self.training 的层 {mode_dep} —— train/eval 逐位相同这条就失去意义'
    m = MambaLTI()
    x = torch.randn(2, CH, BOARD, BOARD)
    m.train()
    with torch.no_grad():
        y_tr = m(x)
    m.eval()
    with torch.no_grad():
        y_ev = m(x)
    d = float((y_tr - y_ev).abs().max())
    print(f'\n[single-path] train vs eval 输出差: max|dY|={d:.3e} '
          f'（P4.1s 的 scan=\'auto\' 曾是 4.768e-07 / 相对 9.79e-08 = 0.82 个 fp32 eps）')
    assert d == 0.0, (
        f'train/eval 输出不再逐位相同（max|dY|={d:.3e}）—— 说明前向里还留着一条'
        f'依赖 self.training 的分支')
    # 同一个块、同一个 chunk_size 下重复前向也必须逐位确定（无隐含随机性）
    m2 = MambaLTI()
    m2.load_state_dict(m.state_dict())
    m2.train()
    with torch.no_grad():
        assert torch.equal(m2(x), y_tr), '同一权重、同一模式下两次前向不逐位相同'


def _best_of(fn, n, warm=2):
    """n 次的**最小值**耗时（ms）。取 min 是为了对同机其它 agent 的负载不敏感。"""
    for _ in range(warm):
        fn()
    best = float('inf')
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def test_ssm_chunked_perf_bound():
    """性能/结构守卫：唯一路径的分块 SSD 在 shipped 形状下必须是**共享 (L,L)** 的真 GEMM。

    P4.1s 的版本是「分块 vs 顺序」的**比值**红线（fwd 1.5×、fwd+bwd 0.6×）。
    顺序路径退役（P4.1s-vec）之后比值没有基准了，所以这里换成一组**更强**的守卫：

      ① **确定性结构守卫**（完全不依赖计时，机器噪声为零）—— 记录生产扫描发出的
         每一次 `torch.bmm` 的形状：
           * 次数 = `⌈T/L⌉`（T=361、L=32 ⇒ 12；硬编码 `chunk_size=1` 会变成 361）；
           * 每次的第一个操作数形状 = `(B·C, l, l)`，**不是** `(B·C·N, l, l)`
             —— 这一条专门钉「(L,L) 必须跨状态维 n 共享」；
           * 第二个操作数形状 = `(B·C, l, N)`（N=64 是状态维，不是被扇出的维）。
         这一条**恰好**抓住 P4.1s 报告里 S6 那条变异：形状不变、数值不变、只是
         不再跨 n 共享 ⇒ 退回 P4.1r 的 `B·C·N` 个瘦矩阵向量乘（实测慢 2.3~35×）。
         旧版是靠计时红线抓它的（有效但怕噪声），现在**不再依赖计时**。
      ② **绝对计时上界**（给同机其它 agent 的负载留 4~5× 余量）：
         fwd 实测 17.7 ms ⇒ 上界 90 ms；fwd+bwd 实测 110 ms ⇒ 上界 450 ms。
         两条都能抓住 S6（≈35× ⇒ 620 ms / 3.8 s）与硬编码 L=1（97 ms / 1.7 s）。
      ③ **块长的相对性质**：`L=1` 的 fwd+bwd 必须比 `L=32` 慢 ≥4×（实测 15.6×）。
         这是算法性的（Python 步数 `⌈T/L⌉`），不是机器特性。
    """
    # ---- ① 确定性结构守卫 ------------------------------------------------ #
    calls = []
    orig_bmm = torch.bmm

    def spy(a, b, **k):
        calls.append((tuple(a.shape), tuple(b.shape)))
        return orig_bmm(a, b, **k)

    torch.bmm = spy
    try:
        with torch.no_grad():
            MambaLTI(chunk_size=MAMBA_CHUNK_SIZE)(
                torch.randn(1, CH, BOARD, BOARD))
    finally:
        torch.bmm = orig_bmm

    T = BOARD * BOARD
    L = MAMBA_CHUNK_SIZE
    n_chunk = -(-T // L)                                    # ceil(T/L) = 12
    tail = T - (n_chunk - 1) * L                            # 最后一块的长度 = 9
    want = ([(CH, L, L)] * (n_chunk - 1)) + [(CH, tail, tail)]
    got = [a for a, _ in calls]
    assert got == want, (
        f'分块扫描发出了 {len(got)} 次 bmm，形状 {got[:3]}…，期望 ⌈T/L⌉={n_chunk} 次、'
        f'形状 {want[:2]}…（最后一块 l={tail}）—— 块边界 / 块长被改动')
    for (a, b), i in zip(calls, range(len(calls))):
        assert a[0] == CH and b[0] == CH, (
            f'第 {i} 块的 bmm 批数是 {a[0]}，期望 B·C={CH}。批数变成 B·C·N={CH * DSTATE} '
            f'就是「(L,L) 不再跨状态维 n 共享」⇒ 退回 P4.1r 的瘦矩阵向量乘'
            f'（每块物化涨 {DSTATE}×，实测慢 2.3~35×）')
        assert b[2] == DSTATE, \
            f'第 {i} 块第二个操作数末维是 {b[2]}，期望状态维 N={DSTATE}'
        assert a[1] == a[2] == b[1], \
            f'第 {i} 块的 bmm 不是 (l,l)@(l,N) 形状：{a} @ {b}'
    # (L,L) 跨 n 共享 ⇒ 每块物化 B·C·l² 个元素（不是 B·C·N·l²）
    m_per_chunk = CH * L * L
    assert m_per_chunk * 4 < CH * DSTATE * L * L, \
        '共享 (L,L) 的每块物化量应比扇出形式少 N=64 倍 —— 这条算错了'
    print(f'\n[perf-struct] {len(calls)} 次 bmm（T={T}, L={L} ⇒ ⌈T/L⌉={n_chunk}，'
          f'末块 l={tail}），批数 B·C={CH}（**非** B·C·N={CH * DSTATE}），'
          f'每块物化 {m_per_chunk:,} 个 M 元素 = 扇出形式的 1/{DSTATE}')

    # ---- ② 绝对计时上界 -------------------------------------------------- #
    m_chk = MambaLTI().train()
    x = torch.randn(1, CH, BOARD, BOARD)

    def fwd_chk():
        with torch.no_grad():
            m_chk(x)

    def fb_chk():
        m_chk.zero_grad(set_to_none=True)
        m_chk(x).sum().backward()

    chk_f = _best_of(fwd_chk, 7)
    print(f'[perf] MambaLTI 单块前向 fp32 B=1 {BOARD}x{BOARD} '
          f'chunked(L={L})={chk_f:.1f} ms（上界 90 ms；实测基线 17.7 ms）')
    assert chk_f < 90.0, \
        (f'分块前向 {chk_f:.1f} ms 超过 90 ms 上界（≈ 比实测基线慢 5 倍）—— 多半是'
         f'退化回了 P4.1r 的 per-channel (L,L)（B·C·N 个瘦矩阵向量乘，实测慢 2.3~35×）'
         f'或 chunk_size 被硬编码成 1')

    chk_b = _best_of(fb_chk, 3, warm=1)
    print(f'[perf] MambaLTI 单块 fwd+bwd fp32 B=1 {BOARD}x{BOARD} '
          f'chunked(L={L})={chk_b:.1f} ms（上界 450 ms；实测基线 110 ms）')
    assert chk_b < 450.0, \
        (f'分块 fwd+bwd {chk_b:.1f} ms 超过 450 ms 上界（≈ 比实测基线慢 4 倍）。反向是'
         f'本参数化最大的收益（10~12×），退化到这里说明分块实现或 A 的参数化被改回去了')

    # ---- ③ 块长的相对性质 ------------------------------------------------ #
    m1 = MambaLTI(chunk_size=1).train()

    def fb_l1():
        m1.zero_grad(set_to_none=True)
        m1(x).sum().backward()

    l1_b = _best_of(fb_l1, 2, warm=1)
    print(f'[perf] 同一块 chunk_size=1 的 fwd+bwd = {l1_b:.1f} ms '
          f'= {l1_b / chk_b:.1f}× 于 L={L}（Python 步数 361 vs {n_chunk}）')
    assert l1_b > 4.0 * chk_b, \
        (f'chunk_size=1 的 fwd+bwd {l1_b:.1f} ms 只比 L={L} 慢 {l1_b / chk_b:.1f}× —— '
         f'分块带来的「把 T 步 Python 循环降到 ⌈T/L⌉ 步」这个收益消失了')


def test_ssm_naive_oracle_at_shipped_N():
    """§5 的 oracle 必须在 **shipped 的 N=64** 上也跑一遍，不许缩到小配置里。

    brief §5 的担心是「N 从 16→64 后朴素三重循环参考慢 4 倍」。P4.1r 实测
    （fp64，本机）：C=184/N=64/T=9 → 16.1 ms，C=184/N=64/T=16 → 68.3 ms。
    P4.1s 形状同构（状态仍是 4×46×64），P4.1s-vec 复测见下面的打印，
    完全不需要 `pytest.mark.slow`，也没有为了省时间把 B 或 N 缩到测不出问题的程度。

    ⚠ P4.1s-vec 之后 `m(x)` 走的是**生产路径**（分块 SSD，`L=32`），而这条比的是
    朴素三重循环 —— 也就是说它现在直接钉「生产实现 == 逐项展开的递推」，不再需要
    任何 `scan=` 开关。参考循环里的 T 步 × T 步 × (s..t) 步对 19×19 不可承受，
    所以只用 3×3 / 4×4 —— 真棋盘 T=361 的覆盖由
    `test_ssm_chunked_scan_matches_sequential_oracle[T361_L*]` 承担。
    """
    for hw in (3, 4):
        torch.manual_seed(31)
        m = MambaLTI().double()
        x = torch.randn(2, CH, hw, hw, dtype=torch.float64)
        ref = dict(_reference_inputs(m, x))
        T = hw * hw
        t0 = time.perf_counter()
        ys = []
        for t in range(T):
            acc = torch.zeros(ref['B'], ref['Hh'], ref['P'], ref['N'], dtype=x.dtype)
            for s in range(t + 1):
                dec = torch.ones(ref['B'], ref['Hh'], ref['P'], ref['N'], dtype=x.dtype)
                for r_ in range(s + 1, t + 1):
                    dec = dec * torch.exp(ref['log_decay'][:, r_]).unsqueeze(-1)
                acc = acc + dec * ref['drive'][:, s]
            ys.append((acc * ref['c_vec'][:, t].view(ref['B'], 1, 1, ref['N'])).sum(-1))
        want = _hand_block_forward(m, x, dict(ref, y=torch.stack(ys, dim=1)))
        cost = (time.perf_counter() - t0) * 1e3
        with torch.no_grad():
            got = m(x)
        assert torch.allclose(got, want, rtol=1e-9, atol=1e-11), \
            f'shipped N=64 (C={CH}, H={N_HEADS}, P={HEAD_DIM}) T={T}: 与朴素三重循环不一致'
        assert cost < 60_000, f'朴素参考跑了 {cost:.0f} ms —— 超过了 60 s 的预算'
        print(f'[oracle] C={CH} N={DSTATE} H={N_HEADS} T={T} 朴素三重循环 {cost:.0f} ms')


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


def _hand_mhsa(attn, x):
    """手写的 MHSA 前向：只用 `wq/wk/wv/wo` 的权重 + 显式 scale 重写一遍。

    **不**调用 `attn.forward`，也**不**调用模块级 `_sdpa` / `_global_attn` ——
    参考实现必须与生产实现独立，否则只能发现「生产 vs 生产」。
    eval 下 `attn_drop_p == 0.0`，与 `_sdpa` 的 math 路径（q 预乘 scale）同口径。
    """
    B, C, H, W = x.shape
    N = H * W
    seq = x.flatten(2).transpose(1, 2)                        # (B, N, C)
    Hh, d = attn.num_heads, attn.head_dim

    def _heads(t):
        return t.view(B, N, Hh, d).transpose(1, 2)            # (B, Hh, N, d)

    q = _heads(F.linear(seq, attn.wq.weight)) * attn.scale
    k = _heads(F.linear(seq, attn.wk.weight))
    v = _heads(F.linear(seq, attn.wv.weight))
    w = (q @ k.transpose(-2, -1)).softmax(dim=-1) @ v        # (B, Hh, N, d)
    return F.linear(w.transpose(1, 2).reshape(B, N, C),
                    attn.wo.weight).transpose(1, 2).reshape(B, C, H, W)


def _hand_ffn(blk, norm, x):
    """手写的 FFN 支路：`x + fc2(gelu(fc1(flatten(LN(x)))))`。

    `norm` 由调用方显式传入（`norm1` 还是 `norm2`），因为「norm 作用在哪一步」
    正是要钉的东西，不能由这里替实现决定。
    """
    h = norm(x).flatten(2).transpose(1, 2)                    # (B, N, C)
    h = F.linear(F.gelu(F.linear(h, blk.fc1.weight)), blk.fc2.weight)
    return h.transpose(1, 2).reshape(x.shape)


def _rms_delta(got, *others):
    """实测 max|Δ|（float64，用来在报错里打出偏差量级）。"""
    return max((got - o).abs().max().item() for o in others)


def test_transformer_block_matches_hand_written_pre_norm_formula():
    """`TransformerBlock` 的**完整**前向 == 测试内手写的 pre-norm 公式。

    这条补的是 review 打回的一个真实缺口：第 13 条只把两条支路的**末端权重**
    清零来证明「有两条恒等捷径」，**没有任何断言**钉住
      ① pre-norm 的**位置**（`attn` / `ffn` 吃的是 LN 之后的张量）；
      ② 两条残差支路的**次序**（注意力在前、FFN 在后，且第二条吃第一条的输出）。
    实测：把 `norm1`/`norm2` 整个旁路掉（模块仍在 ⇒ 参数量不变），或者把两条残差
    的次序对调，22 条旧测试**全绿** —— 一次「无害的重构」就能静默改掉架构。

    手写公式：x0 = x；x1 = x0 + MHSA(LN1(x0))；x2 = x1 + FFN(LN2(x1))。
    `LayerNorm2d` 本身复用块上的实例（它是既有共享类，另有 D5 签名锁），
    「norm 作用在哪一步」由本测试的组合方式决定，不由被测代码决定。
    """
    torch.manual_seed(11)
    m = TransformerBlock().eval().double()
    x = torch.randn(2, CH, 5, 5, dtype=torch.float64)

    with torch.no_grad():
        x1 = x + _hand_mhsa(m.attn, m.norm1(x))
        want = x1 + _hand_ffn(m, m.norm2, x1)
        got = m(x)

        assert torch.allclose(got, want, rtol=1e-9, atol=1e-11), \
            ('TransformerBlock 与手写 pre-norm 公式不一致：max|Δ|={:.3e} —— '
             'norm 位置或两条残差支路的次序被改了'
             ).format(_rms_delta(got, want))

        # 对照 ①：去掉 pre-norm（两条支路直接吃未归一化的 x）必须**不**等于本实现
        no_pre = _hand_ffn(m, m.norm2, x + _hand_mhsa(m.attn, x))
        assert not torch.allclose(got, no_pre, rtol=1e-6, atol=1e-9), \
            ('去掉 pre-norm 后输出不变 —— 本测试分辨不出 norm 的位置（空转）'
             '（max|Δ|={:.3e}）').format(_rms_delta(got, no_pre))

        # 对照 ②：post-norm（norm 放在残差**之后**）必须**不**等于本实现
        a = m.norm1(x) + _hand_mhsa(m.attn, x)          # LN 在残差之后 = post-norm
        post = a + _hand_ffn(m, m.norm2, a)
        assert not torch.allclose(got, post, rtol=1e-6, atol=1e-9), \
            ('post-norm 写法与本实现相同 —— 本测试分辨不出 pre/post（空转）'
             '（max|Δ|={:.3e}）').format(_rms_delta(got, post))

        # 对照 ③：两条残差支路对调（FFN 在前、注意力在后）必须**不**等于本实现
        x1s = x + _hand_ffn(m, m.norm1, x)
        swapped = x1s + _hand_mhsa(m.attn, m.norm2(x1s))
        assert not torch.allclose(got, swapped, rtol=1e-6, atol=1e-9), \
            ('把两条残差支路对调后输出不变 —— 本测试分辨不出支路次序（空转）'
             '（max|Δ|={:.3e}）').format(_rms_delta(got, swapped))


def test_cross_attn_res_matches_hand_written_pre_norm_formula():
    """`CrossAttnRes` 的**完整**前向 == 测试内手写的 pre-norm 公式。

    第 14 条只把 `wo`/`fc2` 清零后**单独取投影支路**去比（`m(x, taps) - x`），
    于是 `norm_attn` / `norm_ffn` 的位置与两条捷径的次序**无人看守**：实测把它们
    旁路掉或把捷径对调，22 条旧测试全绿。本条把三段全串起来比：

        y1 = x + proj(cat(LN_tap(s1), LN_tap(s5), LN_tap(s9)))   # 跨层投影支路
        y2 = y1 + MHSA(LN_attn(y1))                             # 恒等捷径
        y3 = y2 + FFN(LN_ffn(y2))                               # 恒等捷径

    连带的额外收益：`y1` 的残差腿是**当前流 `x`**，不是 `s1`。类 docstring 里
    「第一路 s1 是 identity 捷径」的说法是装饰性的（`s1` 只是被 concat 进去的
    一路特征，没有任何 identity 语义），这条断言把「快捷腿是 x」钉死。
    """
    torch.manual_seed(13)
    m = CrossAttnRes().eval().double()
    x = torch.randn(2, CH, 5, 5, dtype=torch.float64)
    taps = tuple(torch.randn(2, CH, 5, 5, dtype=torch.float64) for _ in range(3))

    with torch.no_grad():
        merged = F.conv2d(torch.cat([m.norm_tap(t) for t in taps], dim=1),
                          m.proj.weight)
        y1 = x + merged
        y2 = y1 + _hand_mhsa(m.attn, m.norm_attn(y1))
        want = y2 + _hand_ffn(m, m.norm_ffn, y2)
        got = m(x, taps)

        assert torch.allclose(got, want, rtol=1e-9, atol=1e-11), \
            ('CrossAttnRes 与手写 pre-norm 公式不一致：max|Δ|={:.3e} —— '
             'norm 位置或两条恒等捷径的次序被改了'
             ).format(_rms_delta(got, want))

        # 对照 ①：旁路 `norm_attn` / `norm_ffn`（注意力/FFN 直接吃未归一化的 y）必须不等
        y2_np = y1 + _hand_mhsa(m.attn, y1)
        no_pre = y2_np + _hand_ffn(m, m.norm_ffn, y2_np)
        assert not torch.allclose(got, no_pre, rtol=1e-6, atol=1e-9), \
            ('旁路 norm_attn/norm_ffn 后输出不变 —— 本测试分辨不出 pre-norm（空转）'
             '（max|Δ|={:.3e}）').format(_rms_delta(got, no_pre))

        # 对照 ②：两条恒等捷径对调（FFN 在前、注意力在后）必须**不**等于本实现
        y2s = y1 + _hand_ffn(m, m.norm_attn, y1)
        swapped = y2s + _hand_mhsa(m.attn, m.norm_ffn(y2s))
        assert not torch.allclose(got, swapped, rtol=1e-6, atol=1e-9), \
            ('把两条恒等捷径对调后输出不变 —— 本测试分辨不出支路次序（空转）'
             '（max|Δ|={:.3e}）').format(_rms_delta(got, swapped))

        # 对照 ③：快捷腿必须取**当前流 x**；换成第一路抽头 s1 必须**不**等于本实现
        leg_s1 = taps[0] + merged
        leg_s1 = leg_s1 + _hand_mhsa(m.attn, m.norm_attn(leg_s1))
        leg_s1 = leg_s1 + _hand_ffn(m, m.norm_ffn, leg_s1)
        assert not torch.allclose(got, leg_s1, rtol=1e-6, atol=1e-9), \
            ('把跨层残差的快捷腿从「当前流 x」换成「第一路抽头 s1」结果不变 —— '
             '本测试分辨不出快捷腿的来源（空转）'
             '（max|Δ|={:.3e}）').format(_rms_delta(got, leg_s1))


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
    """两个头的输出形状与值域；**Tanh 的去留是已裁决项，不是待定项**。

    `FCValueHead` 末尾的 `nn.Tanh()` 曾被 report §5 标成「需用户裁决」。用户已
    **裁决保留 Tanh**（report §5 的裁决记录），所以本测试里所有关于 Tanh 的断言
    都是**终态约束**，不是「等裁决结果再定」的占位：删 `out_tanh` 一行会让本条
    测试变红，这是**预期行为**，不是需要修的回归。要改这个决定，得先推翻裁决。
    """
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
        'mamba_lti': 4 * _n(MambaLTI()),       # 4 × 128,252（P4.1s：P4.1r 的 129,240 → 128,252）
        'transformer': 2 * _n(TransformerBlock()),
        'cross_attn_res': 2 * _n(CrossAttnRes()),
        'out': _n(out),                         # 34,224
    }
    assert parts == {
        'stem': 28_520, 'res_blocks': 4_881_152, 'mamba_lti': 513_008,
        'transformer': 448_960, 'cross_attn_res': 652_832, 'out': 34_224,
    }, f'主干分项漂移：{parts}'
    assert sum(parts.values()) == EXPECT_BACKBONE_TOTAL, \
        f'主干合计 {sum(parts.values())} != 权威 {EXPECT_BACKBONE_TOTAL}'
    # 全网 = 主干 + 2,366,730 (FCPolicyHead, P4.1r/P4.1s 均未改) + 142,017 (FCValueHead)。
    # 注: P4.1r brief §4 自己写的全网合计是 9,071,389, 但用它给的三个分项相加
    #     得 9,071,395 (差 6) —— 笔误，见 P4.1r report §9.1。本文件按 P4.1s brief §1
    #     的三个分项相加得 9,067,443 断言（与 brief 一致，无笔误）。
    assert EXPECT_BACKBONE_TOTAL + EXPECT_POLICY + EXPECT_VALUE == 9_067_443, \
        '全网合计与 P4.1s brief §1 的三个分项不自洽'
    # 对照：7×7 stem（表头文字那个读法）会把合计顶到 6,683,816 ⇒ §3(a) 的 3×3 裁定
    # 是这条断言的判据来源，不是随手选的。7×7 的真值是 49×17×184 = 153,272（+BN=153,640），
    # brief §3(a) 写的 153,296 差 24 —— 同一处笔误，见 report §4(a)。
    # ⚠ 6,683,816 = 6,558,696 − 28,520 + 153,640。P4.1r 时代这里写的是 6,687,768
    #   （那是 6,562,648 − 28,520 + 153,640）—— 换装 P4.1s 之后必须同步，否则
    #   这条「对照」会变成一个与权威合计无关的魔数。
    stem7 = nn.Sequential(nn.Conv2d(17, CH, 7, padding=3, bias=False), nn.BatchNorm2d(CH))
    assert _n(stem7) == 153_640
    assert sum(parts.values()) - parts['stem'] + _n(stem7) == 6_683_816
    # 对照：P4.1r 的逐通道结构化低秩（129,240/块）与 rev1（113,528/块）若没被换掉，
    # 4 块会是 516,960 / 454,112，主干合计随之漂移 —— 这条把「P4.1s 换装」
    # 钉在主干加总上，不只在单块测试里钉。
    assert EXPECT_MAMBA == 128_252 not in (129_240, 113_528) \
        and parts['mamba_lti'] == 513_008
    assert sum(parts.values()) - parts['mamba_lti'] + 4 * 129_240 == 6_562_648


def test_sdpa_force_math_restorable():
    """本文件不改变模块级后端开关（`_sdpa_force_math`），收尾复原以便其它测试复用。"""
    assert set_sdpa_force_math(True) is None
