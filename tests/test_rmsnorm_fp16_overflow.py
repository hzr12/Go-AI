# -*- coding: utf-8 -*-
"""`backbone.RMSNorm` 的 fp16 平方和溢出防护。

背景：V7 的 `Nbt2TransformerBlock` 有 11 块 × 2 内块 × 2 个 norm = **44 处**走
`backbone.RMSNorm`，而 910A 无 bf16、AMP 走 fp16。原实现直接 `x.pow(2)`，
`|x|` 超过约 256 时撞上 fp16 的 65504 上限。

空间维的孪生 `katago_v7.RMSNormMask` 早已因同样理由硬化成 fp32（其 docstring
写明「fp16 的 x² 在通道宽 256 时容易溢出到 inf」），token 维这个之前漏了。

**这个缺陷不是NaN 的源头**（实测，见 `test_this_is_not_a_nan_amplifier`）：
它的表现是**静默塌缩**—— 输出全 0、梯度全 0，loss 曲线照样降，像是在训练。
"""
import sys

import pytest
import torch

sys.path.insert(0, '.')

from src.networks.backbone import RMSNorm  # noqa: E402

FP16_MAX = 65504.0


def _naive(x, eps=1e-6):
    """修复前的写法，留作对照oracle。"""
    rms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return x * rms


def _big_input(seed=0):
    """量级约 300~420 的**非恒定** fp16 输入。

    恒定输入不行：RMSNorm 对均匀缩放不变，`dy/dx = rms − rms³·x²` 在常量处
    恰好抵消成 0，那是**数学上正确**的 0。拿它断言「梯度非零」会得到假失败。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 8, 256, generator=g) * 30.0 + 300.0
    return x.to(torch.float16)


def _rms(t):
    return float(t.float().pow(2).mean().sqrt().detach())


def test_premise_naive_fp16_square_sum_overflows():
    """前提自检：写死fp16 上限会变，变了这条就该先红。"""
    x = _big_input()
    assert bool(torch.isinf(x.pow(2)).any()), float(x.pow(2).abs().max())
    assert float(x.abs().max()) ** 2 > FP16_MAX


def test_naive_form_silently_collapses_to_zero():
    """对照：溢出 ⇒ `mean=inf` ⇒ `rsqrt(inf)=0` ⇒ 输出**全 0**。

    注意它**不是 NaN**，前向 `isfinite` 全绿 —— 所以这个缺陷在日志上完全
    不可见，只会表现为「学不动」。
    """
    y = _naive(_big_input())
    assert torch.isfinite(y).all(), y
    assert _rms(y) == pytest.approx(0.0, abs=1e-6), _rms(y)


def test_fixed_form_normalises_correctly():
    """修复后输出 RMS = 1（RMSNorm 的定义；单个元素可大于 1，别断言 max）。"""
    y = RMSNorm(256)(_big_input())
    assert torch.isfinite(y).all(), y
    assert _rms(y) == pytest.approx(1.0, abs=2e-2), _rms(y)


def test_fixed_form_keeps_the_gradient_alive():
    """塌缩会把梯度一起清零 —— 「有限」是因为「没梯度」，等于换个地方坏。

    loss 放大 1e4 是为了让梯度落回 fp16 **正规区**（原始量级约 1.4e-6 落在
    次正规区、精度极低，拿它做判据测的是 fp16 次正规行为而不是这个修复）。
    """
    x = _big_input().requires_grad_(True)
    RMSNorm(256)(x).float().pow(2).mean().mul(1e4).backward()
    gmax = float(x.grad.abs().max().detach())
    assert torch.isfinite(x.grad).all(), x.grad
    assert gmax > 1e-5, gmax
    # 对照：朴素写法在这里是彻底 0
    xn = _big_input().requires_grad_(True)
    _naive(xn).float().pow(2).mean().mul(1e4).backward()
    assert float(xn.grad.abs().max().detach()) == 0.0


def test_this_is_not_a_nan_amplifier():
    """钉住一个**否证**：这个缺陷不会把局部 inf 放大成全张量 NaN。

    我曾怀疑「`inf · 0 = NaN` ⇒ NaN 铺满整个张量 ⇒ 扩散到全部参数」。
    实测（fp16、量级 300+、注入 1 个 inf）：朴素与修复**都只产生 1 个 NaN**。
    因为 `mean` 见到inf 就是 inf、`rsqrt(inf)=0`，于是只有那个 inf 元素
    变成 `inf·0`，其余有限元素都变成 `finite·0 = 0`。**没有放大。**

    留这条是为了防止下一次又把它当成 NaN 的解释 —— 真机 NaN 的源头仍未定位。
    """
    x = _big_input()
    x[0, 0, 0] = float('inf')
    for tag, fn in (('naive', _naive), ('fixed', RMSNorm(256))):
        y = fn(x)
        assert int(torch.isnan(y).sum()) == 1, (tag, int(torch.isnan(y).sum()))
        assert not torch.isinf(y).any(), tag


def test_non_fp16_dtypes_are_bitwise_unchanged():
    """bf16 / fp32 / fp64 必须与朴素写法**逐位相同**（零回归）。

    bf16 的指数位与 fp32 相同（最大约 3.4e38），不存在 fp16 那个溢出，
    所以刻意不改它 —— 否则「只修 fp16」就悄悄变成了「顺手改了 bf16」。
    """
    for dtype in (torch.bfloat16, torch.float32, torch.float64):
        n = RMSNorm(64)
        with torch.no_grad():
            n.weight.copy_(torch.linspace(0.5, 1.5, 64))
        x = (torch.randn(3, 5, 64) * 3.0).to(dtype)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + n.eps).rsqrt()
        want = x * rms * n.weight
        assert torch.equal(n(x), want), '%s 的数值被改了，本测试要求零回归' % dtype


def test_v7_actually_routes_44_norms_through_this_module():
    """确认防护在现役路径上：数一遍 V7 里 `backbone.RMSNorm` 的实例数。

    11 个 trunk 块 × 2 个 inner 块 × 2 个 norm（attn + ffn）= 44。
    将来若 `Nbt2TransformerBlock` 换了归一化实现，这条会失败并提醒重新评估
    fp16 溢出风险 —— 避免防护悄悄失去作用对象。
    """
    from src.networks.katago_v7 import NBT_TF_CFG, Nbt2TransformerBlock

    blocks = NBT_TF_CFG['num_blocks']
    inner = NBT_TF_CFG['num_inner_blocks']
    blk = Nbt2TransformerBlock(NBT_TF_CFG['trunk_channels'],
                               NBT_TF_CFG['nbt_mid'],
                               NBT_TF_CFG['num_heads'],
                               NBT_TF_CFG['ffn_hidden'], inner_len=inner)
    n_rms = sum(isinstance(m, RMSNorm) for m in blk.modules())
    assert n_rms == inner * 2, n_rms
    assert blocks * n_rms == 44, blocks * n_rms