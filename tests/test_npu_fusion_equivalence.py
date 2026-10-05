"""NPU 融合算子（RMSNorm / SwiGLU）与标准路径的数值等价性测试。

仅在 torch_npu 可用且有 NPU 设备时运行；否则整模块 skip，不影响
无 NPU（CUDA/CPU）环境下的 CI。

目的：把 backbone.py / katago_v7.py 里 NPU 融合分支「数学等价」的注释断言
变成「测试保证等价」——覆盖前向 + 反向，fp16 下容忍 ~1e-2（融合 kernel 与
标准 fp32 中间量计算的舍入差异）。
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("NPU device not available", allow_module_level=True)

from src.networks.backbone import RMSNorm, _npu_rms_norm  # noqa: E402
from src.networks.katago_v7 import SwiGLU, _npu_swiglu    # noqa: E402


def test_npu_rms_norm_matches_reference():
    dev = "npu"
    C = 64
    w = torch.ones(C, device=dev, dtype=torch.float32)
    x = torch.randn(2, 10, C, device=dev, dtype=torch.float16)
    out = _npu_rms_norm(x.float(), w, 1e-6)
    xf = x.float()
    ref = (xf * (xf.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt() * w).to(x.dtype)
    assert torch.allclose(out.float(), ref.float(), atol=5e-3, rtol=5e-3)


def test_rmsnorm_forward_backward_equivalence():
    dev = "npu"
    m = RMSNorm(64, eps=1e-6).to(dev)
    x = torch.randn(2, 10, 64, device=dev, dtype=torch.float16, requires_grad=True)
    # 融合路径（走 npu 分支）
    out_fused = m(x)
    out_fused.sum().backward()
    g_fused = x.grad.clone()
    x.grad = None
    # 标准参考（cpu 上等价 fp32 中间量实现）
    ref = RMSNorm(64, eps=1e-6).double()
    with torch.no_grad():
        ref.weight.copy_(m.weight.double())
    xc = x.double().cpu().requires_grad_(True)
    ref(xc).sum().backward()
    assert torch.allclose(out_fused.double().cpu(), ref(xc).detach(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(g_fused.double().cpu(), xc.grad, atol=1e-2, rtol=1e-2)


def test_swiglu_fusion_equivalence():
    dev = "npu"
    sw = SwiGLU(64, 96).to(dev)
    sw.initialize()
    x = torch.randn(2, 10, 64, device=dev, dtype=torch.float16)
    out_fused = sw(x)
    # 标准路径：相同权重放到 cpu 走 F.silu(up) * gate -> down
    sw_cpu = SwiGLU(64, 96)
    sw_cpu.load_state_dict(sw.state_dict())
    xc = x.float().cpu().requires_grad_(True)
    out_ref = sw_cpu(xc).detach().to(dev)
    assert torch.allclose(out_fused.float(), out_ref.float(), atol=5e-3, rtol=5e-3)
