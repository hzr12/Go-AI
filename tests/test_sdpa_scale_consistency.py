""" 回归测试：`_sdpa` 的 scale 口径（唯一后端 = torch 内置 SDPA）。

背景（2026-10-03 parity 排查）
--------------------------------
用官方 `b10c384` 权重做逐位对拍时，V7 的 value 输出与 KataGo 引擎差了约 300 倍，
定位到根因是**注意力的 scale 被施加了两次**：

    q = self._heads(self.q(t)) * self.scale   # 预乘 1/sqrt(head_dim)
    ctx = _sdpa(q, k, v, scale=self.scale)    # SDPA 内部再乘一次 1/sqrt(d)
                                              # ⇒ logits 小 32 倍，softmax 被压平

修复后：scale **只**从 `_sdpa` 的 `scale` 形参进入并透传给 SDPA，调用方不预乘。

本文件钉住
    1. `_sdpa` 真的尊重 `scale`（用非默认值，抓「scale 被忽略」的回归）
    2. `scale=None` 保持 SDPA 原生默认 1/sqrt(d)
    3. `_sdpa` 是 `F.scaled_dot_product_attention` 的纯透传（不再有第二套后端）
    4. 端到端：V7 的 `MHSA.forward` 只施加**一次** 1/sqrt(head_dim)
"""
import math
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.networks.backbone as bb  # noqa: E402

B, HH, N, D = 2, 4, 9, 32
SCALE = 1.0 / math.sqrt(D)


def _qkv():
    g = torch.Generator().manual_seed(1234)
    return tuple(torch.randn(B, HH, N, D, generator=g) for _ in range(3))


def _reference(q, k, v, scale):
    """手写参考实现：明确地只乘一次 scale。"""
    w = ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1)
    return w @ v


def test_sdpa_honors_non_default_scale():
    """ 抓住「SDPA 路径把 `scale` 入参忽略掉」的回归。

    `F.scaled_dot_product_attention(...)` 若不接 `scale=`，无论调用方传什么都用
    1/sqrt(d)。用一个明显不同的 scale 就能分辨。
    """
    q, k, v = _qkv()
    weird = 0.1234
    got = bb._sdpa(q, k, v, scale=weird)
    torch.testing.assert_close(got, _reference(q, k, v, weird),
                               rtol=1e-5, atol=1e-6)
    default = bb._sdpa(q, k, v, scale=SCALE)
    assert not torch.allclose(got, default, rtol=1e-3, atol=1e-4), \
        'SDPA 路径忽略了 scale 参数'


def test_scale_none_keeps_native_default():
    """`scale=None` ⇒ 用 SDPA 原生默认（等价 1/sqrt(d)）。"""
    q, k, v = _qkv()
    got = bb._sdpa(q, k, v, scale=None)
    torch.testing.assert_close(got, _reference(q, k, v, SCALE),
                               rtol=1e-5, atol=1e-6)


def test_sdpa_is_a_pure_passthrough():
    """`_sdpa` 不许再叠一层缩放/温度 —— 必须与 torch 内置逐位相同。"""
    q, k, v = _qkv()
    for s in (None, SCALE, 0.1234):
        torch.testing.assert_close(
            bb._sdpa(q, k, v, scale=s),
            F.scaled_dot_product_attention(q, k, v, scale=s),
            rtol=0, atol=0)


def test_v7_mhsa_scales_exactly_once():
    """端到端不变量：V7 的 MHSA 只施加一次 1/sqrt(head_dim)。

    修复前这条会失败（相对误差约 82%）—— V7 预乘了 q 却又让 SDPA 内部再乘一次。
    这里同时构造「乘两次」的错误参考，钉死它不是碰巧相等。
    """
    from src.networks.katago_v7 import MHSA

    torch.manual_seed(7)
    mhsa = MHSA(dim=HH * D, num_heads=HH).eval()
    t = torch.randn(B, N, HH * D)
    # N=9 ⇒ 3x3 的行列网格。位置编码只关心 (row, col)，与 N 的具体值无关。
    grid = int(round(N ** 0.5))
    assert grid * grid == N
    pos = torch.stack(
        torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij'),
        dim=-1).reshape(1, N, 2).expand(B, N, 2).contiguous()

    c = HH * D

    def _run(extra_scale):
        qkv = mhsa.qkv(t)
        q = mhsa._heads(qkv[..., :c])
        k = mhsa._heads(qkv[..., c:2 * c])
        v = mhsa._heads(qkv[..., 2 * c:])
        q, k = mhsa.rope(q, pos), mhsa.rope(k, pos)
        w = ((q @ k.transpose(-2, -1)) * mhsa.scale * extra_scale).softmax(-1)
        return mhsa.out((w @ v).permute(0, 2, 1, 3).reshape(B, N, c))

    with torch.no_grad():
        got = mhsa(t, pos)
        once = _run(1.0)
        twice = _run(mhsa.scale)

    torch.testing.assert_close(got, once, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(got, twice, rtol=1e-2, atol=1e-2), \
        '1/sqrt(head_dim) 被施加了两次'
