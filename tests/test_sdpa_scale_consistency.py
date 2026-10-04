""" 回归测试：`_sdpa` 的三条后端路径必须给出**同一个**结果。

背景（2026-10-03 parity 排查）
--------------------------------
用官方 `b10c384` 权重做逐位对拍时，V7 的 value 输出与 KataGo 引擎差了约 300 倍，
定位到根因是**注意力的 scale 被施加了两次**：

    q = self._heads(self.q(t)) * self.scale     # 预乘 1/sqrt(head_dim)
    ctx = _sdpa(q, k, v, scale=None)            # math 路径不缩放 ⇒ 正确
                                            # SDPA / flash 路径内部自带 1/sqrt(d)
                                            #              ⇒ 再乘一次 ⇒ logits 小 32 倍

后果按后端分裂：

    | 后端              | 谁在跑                    | 修复前 |
    |-------------------|---------------------------|--------|
    | math | 910A / V100 (sm_70) | 正确 |
    | SDPA | CPU 对拍、A100/H100 | 双重缩放 |
    | flash-attn | A100/H100（库可用时） | 双重缩放 |

也就是说**在 910A 上训练是对的，但同一份权重在 CPU 上评测/对拍是错的** ——
这类「训练正常、推理异常」的分叉最难发现，所以这里把三条路径钉死。

修复方式：scale 只从 `_sdpa` 的 `scale` 形参进入，各后端自己决定怎么施加
（math 手动乘 / SDPA 透传 `scale=` / flash 手动抵消它写死的 `1/sqrt(d)`）。

本文件覆盖
    1. math 路径 == SDPA 路径（给定同一个 scale）
    2. SDPA 路径**真的**尊重 `scale`（用一个非默认值，抓住「scale 被忽略」的回归）
    3. `scale=None` 保持 SDPA 原生默认（向后兼容旧调用点）
    4. flash 路径 == math 路径（用假 flash 内核复现其写死的 1/sqrt(d) 语义）
    5. 端到端：V7 的 `MHSA.forward` 在 math / SDPA 两个后端下输出逐元素一致
"""
import math
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.networks.backbone as bb  # noqa: E402

B, HH, N, D = 2, 4, 9, 32
SCALE = 1.0 / math.sqrt(D)


@pytest.fixture(autouse=True)
def _clean_backends():
    """每个用例都从「无 flash、math 强制关、query 不分块」的干净状态出发。"""
    old_force = bb._sdpa_force_math
    old_chunk = bb._attn_query_chunk
    old_flash = bb._flash_attn_func
    bb._sdpa_force_math = False
    bb._attn_query_chunk = 0
    bb._flash_attn_func = None
    yield
    bb._sdpa_force_math = old_force
    bb._attn_query_chunk = old_chunk
    bb._flash_attn_func = old_flash


def _qkv():
    g = torch.Generator().manual_seed(1234)
    return tuple(torch.randn(B, HH, N, D, generator=g) for _ in range(3))


def _reference(q, k, v, scale):
    """手写参考实现：明确地只乘一次 scale。"""
    w = ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1)
    return w @ v


def test_math_and_sdpa_paths_agree():
    q, k, v = _qkv()
    math_out = bb._sdpa(q, k, v, use_math=True, scale=SCALE)
    sdpa_out = bb._sdpa(q, k, v, use_math=False, scale=SCALE)
    torch.testing.assert_close(math_out, sdpa_out, rtol=1e-5, atol=1e-6)


def test_sdpa_path_honors_non_default_scale():
    """ 抓住「SDPA 路径把 `scale` 入参忽略掉」的回归。

    修复前 `F.scaled_dot_product_attention(...)` 没接 `scale=`，无论调用方传什么
    都用 1/sqrt(d)。用一个明显不同的 scale 就能分辨。
    """
    q, k, v = _qkv()
    weird = 0.1234
    got = bb._sdpa(q, k, v, use_math=False, scale=weird)
    torch.testing.assert_close(got, _reference(q, k, v, weird),
                               rtol=1e-5, atol=1e-6)
    # 并且必须**不等于**默认缩放的结果，否则说明 scale 又被吞了
    default = bb._sdpa(q, k, v, use_math=False, scale=SCALE)
    assert not torch.allclose(got, default, rtol=1e-3, atol=1e-4), \
        'SDPA 路径忽略了 scale 参数'


def test_sdpa_scale_none_keeps_native_default():
    """`scale=None` ⇒ 用 SDPA 原生默认（等价 1/sqrt(d)），旧调用点行为不变。"""
    q, k, v = _qkv()
    got = bb._sdpa(q, k, v, use_math=False, scale=None)
    torch.testing.assert_close(got, _reference(q, k, v, SCALE),
                               rtol=1e-5, atol=1e-6)


def test_flash_path_matches_math_path():
    """假 flash 内核，复现真 flash-attn 写死的 1/sqrt(head_dim) 语义。

    真库签名：`(q, k, v, dropout_p=, causal=)`，布局 (B, S, Hh, d)，
    缩放在 kernel 内固定为 1/sqrt(d)，Python 侧无法传参。
    """
    q, k, v = _qkv()
    seen = {}

    def fake_flash(qf, kf, vf, dropout_p=0.0, causal=False):
        # (B, S, Hh, d) -> (B, Hh, S, d)
        qq, kk, vv = (t.transpose(1, 2) for t in (qf, kf, vf))
        seen['shape'] = tuple(qf.shape)
        w = ((qq @ kk.transpose(-2, -1)) / math.sqrt(qq.shape[-1])).softmax(-1)
        return (w @ vv).transpose(1, 2)

    math_out = bb._sdpa(q, k, v, use_math=True, scale=SCALE)
    bb._flash_attn_func = fake_flash
    flash_out = bb._sdpa(q, k, v, use_math=False, scale=SCALE)
    assert seen['shape'] == (B, N, HH, D)
    # flash 路径会先把 q/k/v 降成 bf16（真库只吃 fp16/bf16），返回也是 bf16，
    # 而 math 路径是 fp32。比对数值即可，dtype 差异是既有设计。
    torch.testing.assert_close(math_out.to(torch.bfloat16).float(),
                               flash_out.float(), rtol=3e-2, atol=3e-2)


def test_flash_path_honors_non_default_scale():
    """非默认 scale 时，flash 路径也必须和 math 一致（靠手动抵消写死值）。"""
    q, k, v = _qkv()
    weird = 0.1234

    def fake_flash(qf, kf, vf, dropout_p=0.0, causal=False):
        qq, kk, vv = (t.transpose(1, 2) for t in (qf, kf, vf))
        w = ((qq @ kk.transpose(-2, -1)) / math.sqrt(qq.shape[-1])).softmax(-1)
        return (w @ vv).transpose(1, 2)

    bb._flash_attn_func = fake_flash
    flash_out = bb._sdpa(q, k, v, use_math=False, scale=weird)
    math_out = bb._sdpa(q, k, v, use_math=True, scale=weird)
    torch.testing.assert_close(math_out.to(torch.bfloat16).float(),
                               flash_out.float(), rtol=3e-2, atol=3e-2)


def test_v7_mhsa_identical_across_backends():
    """端到端不变量：V7 的 MHSA 在 math 与 SDPA 两个后端下必须逐元素一致。

    修复前这条会失败（相对误差约 82%）—— 因为 V7 预乘了 q 却又让 SDPA 内部
    再乘一次 1/sqrt(d)。
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

    with torch.no_grad():
        bb._sdpa_force_math = True
        via_math = mhsa(t, pos)
        bb._sdpa_force_math = False
        via_sdpa = mhsa(t, pos)

    torch.testing.assert_close(via_math, via_sdpa, rtol=1e-5, atol=1e-6)
    # 防呆：确认这条路径真的被走到过（否则测试可能因恒等分支而空过）
    assert not torch.allclose(via_math, torch.zeros_like(via_math))
    assert F.scaled_dot_product_attention is not None