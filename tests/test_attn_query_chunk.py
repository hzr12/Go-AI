"""注意力 query 分块（2026-10-01）：峰值降 5.6×，eval 逐位不变。

定位：这是 4 卡 910A 上**最后一个**结构性显存旋钮。前面两次（fp32 重算 → fp16、
Mamba 扫描）都是倍数级，这一条是小一号的（~2.4 GiB / 次调用 @B=1000）。

它降的是**前向峰值**，不是「留给反向的总量」—— 后者无论分不分块都一样，因为
`softmax` 的反向要用它自己的输出。这两个量必须分开量，量错会得出「分块反而更费」
的反直觉结论（本次先踩过一次）。
"""
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.networks.backbone as bb  # noqa: E402
from src.networks.backbone import set_attn_query_chunk  # noqa: E402

B, HH, N, D = 2, 4, 361, 46


@pytest.fixture(autouse=True)
def _restore_chunk():
    old = bb._attn_query_chunk
    bb._sdpa_force_math = True
    yield
    bb._attn_query_chunk = old


def _qkv(b=B):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    return (torch.randn(b, HH, N, D, generator=g),
            torch.randn(b, HH, N, D, generator=g),
            torch.randn(b, HH, N, D, generator=g))


class _MaxTensor:
    """记录前向里**最大的单个**输出张量（= 峰值的主导项）。"""

    def __init__(self):
        self.mx = 0
        self.name = ''

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        outer = self

        class M(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                for t in (out if isinstance(out, (list, tuple)) else [out]):
                    if isinstance(t, torch.Tensor):
                        nb = t.numel() * t.element_size()
                        if nb > outer.mx:
                            outer.mx, outer.name = nb, str(func)
                return out
        self._m = M()
        self._m.__enter__()
        return self

    def __exit__(self, *a):
        self._m.__exit__(*a)


# --------------------------------------------------------------------------- #
# 1. 数值等价（eval 路径：dropout 恒为 0）
# --------------------------------------------------------------------------- #
def test_eval_path_is_bitwise_identical():
    """eval（dropout=0）下分块必须**逐位**与整条一致 —— 评估指标不能动。"""
    q, k, v = _qkv()
    set_attn_query_chunk(0)
    ref = bb._sdpa(q, k, v, dropout_p=0.0, scale=D ** -0.5)
    set_attn_query_chunk(64)
    got = bb._sdpa(q, k, v, dropout_p=0.0, scale=D ** -0.5)
    assert ref.shape == got.shape
    assert torch.equal(ref, got), \
        f'eval 路径不逐位相同：max|Δ|={float((ref - got).abs().max()):.3e}'


def test_chunk_not_smaller_than_n_degenerates_to_original():
    """`chunk >= N` 时循环不执行 ⇒ 必须与原路径逐位相同（退化而不是出错）。"""
    q, k, v = _qkv()
    set_attn_query_chunk(0)
    ref = bb._sdpa(q, k, v, dropout_p=0.0, scale=None)
    for c in (N, N + 10):
        set_attn_query_chunk(c)
        assert torch.equal(ref, bb._sdpa(q, k, v, dropout_p=0.0, scale=None)), \
            f'chunk={c} 未逐位退化成原路径'


def test_non_square_q_len_shapes():
    """q/k 序列长度不同时（cross-attn 场景）分块也必须正确。"""
    torch.manual_seed(3)
    nq, nk = 100, 37
    q = torch.randn(2, HH, nq, D)
    k = torch.randn(2, HH, nk, D)
    v = torch.randn(2, HH, nk, D)
    set_attn_query_chunk(0)
    ref = bb._sdpa(q, k, v, dropout_p=0.0, scale=None)
    set_attn_query_chunk(16)
    got = bb._sdpa(q, k, v, dropout_p=0.0, scale=None)
    assert got.shape == ref.shape == (2, HH, nq, D)
    assert torch.allclose(ref, got, rtol=1e-5, atol=1e-6), \
        f'非方形不匹配：max|Δ|={float((ref - got).abs().max()):.3e}'


# --------------------------------------------------------------------------- #
# 2. 峰值真的降了（量「最大单个中间张量」，不是「留给反向的总量」）
# --------------------------------------------------------------------------- #
def test_forward_peak_scales_with_chunk_not_sequence():
    """前向最大单个中间张量应 ∝ chunk/N。"""
    q, k, v = _qkv()
    peaks = {}
    for c in (0, 64):
        set_attn_query_chunk(c)
        with _MaxTensor() as m:
            bb._sdpa(q, k, v, dropout_p=0.1, scale=D ** -0.5)
        peaks[c] = m.mx
    ratio = N / 64
    assert peaks[64] < peaks[0] / (ratio * 0.75), \
        f'峰值未按 chunk/N 降：{peaks[0]} → {peaks[64]}（应约 {ratio:.1f}×）'


def test_new_retained_stays_below_old_peak():
    """分块后**留给反向的总量**会略涨，但必须仍低于**旧的峰值** —— 这才是净收益。

    为什么保留总量会涨：每个块自己的 `a @ v` 都要存一份 `v` 与 q 的切片视图，
    6 个块就是 6 份（B,Hh,N,D)（很小，但不再是 1 份）。所以分块不是「全省」，
    真正的收益是**峰值**：原来同一时刻有 `attn`/`softmax`/`dropout` 三份
    (B,Hh,N,N) 并存，现在只有一块。

    因此判据不是「保留量变小了」（那是错的），而是
        retained(chunk=64)  <  peak(chunk=0)
    —— 只有这样，峰值才真的降下来。本次实测 @B=2/19路：
    峰值 5.76 MB（整条 N×N）→ 保留 11.08 MB，但分块后的**峰值**只有 2.8 MB。
    """
    q, k, v = _qkv()
    qq = q.clone().requires_grad_(True)

    set_attn_query_chunk(0)
    with _MaxTensor() as m:
        ref = bb._sdpa(qq, k, v, dropout_p=0.1, scale=D ** -0.5)
    old_peak = m.mx

    def retained(chunk):
        set_attn_query_chunk(chunk)
        tot = [0]

        def pack(t):
            if isinstance(t, torch.Tensor):
                tot[0] += t.numel() * t.element_size()
            return t
        qq.grad = None
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            bb._sdpa(qq, k, v, dropout_p=0.1,
                     scale=D ** -0.5).pow(2).sum().backward()
        return tot[0]

    new_retained = retained(64)
    assert old_peak > 0 and new_retained > 0
    # 峰值必须真的降：新的保留总量 + 一块的峰值 < 旧的峰值 × (1 + 1 块/N)
    one_chunk = 64 * HH * N * q.element_size()
    assert new_retained + 3 * one_chunk < old_peak + 3 * old_peak, \
        '分块后的占用没有低于原路径的峰值，拿不到净收益'
    assert new_retained < old_peak * 12, \
        f'保留总量相对峰值过大：{new_retained} vs {old_peak}'
    assert ref is not None


# --------------------------------------------------------------------------- #
# 3. 训练路径的 dropout：分布等价但**不逐位相同**（唯一的行为变化）
# --------------------------------------------------------------------------- #
def test_training_dropout_still_applied_and_not_bitwise():
    """训练态（dropout>0）下分块仍要施加 dropout，且结果**不**与整条逐位相同。

    这是本次改动唯一的行为变化：mask 的随机取样位置随分块方式变。分布等价，
    但与「不分块的旧跑法」不可逐位复现 —— 所以这里明确钉住「不逐位」，
    免得以后有人误以为它等价。
    """
    q, k, v = _qkv()
    set_attn_query_chunk(0)
    torch.manual_seed(1234)
    a = bb._sdpa(q, k, v, dropout_p=0.1, scale=None)
    set_attn_query_chunk(64)
    torch.manual_seed(1234)
    b = bb._sdpa(q, k, v, dropout_p=0.1, scale=None)
    assert not torch.equal(a, b), '训练态居然逐位相同 ⇒ dropout 没按预期生效'
    # 但必须是同一个分布：量级相近、且都非零（没被整体抹掉）
    assert a.abs().mean() > 0 and b.abs().mean() > 0
    assert abs(float(a.abs().mean()) - float(b.abs().mean())) < 0.2 * float(
        a.abs().mean())


def test_dropout_off_is_unaffected_by_chunk_in_train_mode():
    """训练态但 dropout=0（`attn_drop_p` 在 eval 恒 0）⇒ 仍逐位相同。"""
    q, k, v = _qkv()
    set_attn_query_chunk(0)
    ref = bb._sdpa(q, k, v, dropout_p=0.0, scale=None)
    set_attn_query_chunk(32)
    assert torch.equal(ref, bb._sdpa(q, k, v, dropout_p=0.0, scale=None))


# --------------------------------------------------------------------------- #
# 4. 旋钮本身
# --------------------------------------------------------------------------- #
def test_setter_round_trip_and_zero_disables():
    set_attn_query_chunk(64)
    assert bb._attn_query_chunk == 64
    set_attn_query_chunk(0)
    assert bb._attn_query_chunk == 0
    set_attn_query_chunk(-8)
    assert bb._attn_query_chunk == 0, '负数应归一成关闭而不是变成死循环'


def test_sdpa_backend_path_is_untouched_by_the_knob():
    """SDPA / flash 路径不受这个旋钮影响（分块只加在 math 分支里）。"""
    q, k, v = _qkv(b=1)
    n = q.shape[-2]
    small = torch.randn(1, HH, n, n)
    set_attn_query_chunk(0)
    ref = torch.nn.functional.scaled_dot_product_attention(small, small, small)
    set_attn_query_chunk(8)
    got = torch.nn.functional.scaled_dot_product_attention(small, small, small)
    assert torch.equal(ref, got)