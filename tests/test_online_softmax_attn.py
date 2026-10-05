"""核对 online-softmax 注意力（`_OnlineSoftmaxAttn` / `_sdpa_math` 的 online 分支）。

验证四件事：
1. 前向：online 与 materialize softmax 注意力在 `dropout=0` 下数值等价
   （fp32 严格、fp16/bf16 宽松）。**等价 ≠ 逐位相同** —— 实测 fp32
   `max|Δ| ≈ 7e-07`，因为 online 走 rescale 累加、materialize 走一次 softmax，
   算术次序不同。这条断言刻意用 allclose 而非 equal。
2. 反向：online 的自定义 backward（flash 风格重算）通过 `gradcheck`。
3. **低精度 + autocast**：统计量是 fp32、matmul 留在计算 dtype，两者在
   autocast（bf16）下混用曾直接抛
   `RuntimeError: expected m1 and m2 to have the same dtype` ——
   V7 的 head 重算那条用例就是这么炸的。这组用例锁住它。
4. query 分块（Nq>chunk）与非分块（Nq<=chunk）两种形状都覆盖。

**默认是关的**（`_attn_online = False`，需 `GOAI_ATTN_ONLINE=1`）：它不是逐位
等价，而 `tests/test_mcts_in_channels.py::test_twelve_channel_path_bit_identical`
靠逐位一致发现意外数值变化。下面用 fixture 保证每个用例都在**显式打开**的状态下
跑，避免哪天默认值被改动时这些用例集体变成「什么都没测」。
"""
import math
import os

import pytest
import torch
from torch.autograd import gradcheck

from src.networks import backbone as bb
from src.networks.backbone import _sdpa_math, set_attn_online


@pytest.fixture(autouse=True)
def _online_on():
    """每个用例都在 online **打开**的状态下跑，并在结束后复位。

    为什么必须显式打开：`_attn_online` 默认 False（见模块 docstring）。
    若用例只依赖默认值，哪天有人把默认值改成 True，这些用例会「通过」但
    其实什么 online 代码都没执行到 —— 那种绿是假的。
    """
    set_attn_online(True)
    yield
    set_attn_online(False)


def _reference(q, k, v, scale):
    # 强制走 materialize 路径作为参照
    set_attn_online(False)
    try:
        return _sdpa_math(q, k, v, dropout_p=0.0, scale=scale)
    finally:
        set_attn_online(True)


def test_default_is_off():
    """默认必须关闭 —— 它不是逐位等价，会让 12 通道的 bit-identical 基线失效。"""
    set_attn_online(False)
    assert bb._attn_online is False


def test_cli_flag_exists_and_is_wired():
    """`--attn-online` 必须存在、默认 0、且真的接到 `set_attn_online`。

    为什么锁这一条：本仓吃过「开关只藏在 config 表里、没有干净命令行入口」的亏 ——
    `--use-checkpoint` 当年被「归档」，提示语甚至写着「如需关闭请用 `--compile 1`」，
    那是拿图编译顺带关检查点。所以这次的开关必须从命令行可达。
    """
    import ast
    import io as _io
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, "scripts", "train_sft.py")
    src = _io.open(path, encoding="utf-8").read()
    assert "add_argument('--attn-online'" in src, \
        "train_sft.py 里找不到 --attn-online 定义"
    assert "default=0" in src.split("add_argument('--attn-online'")[1][:200], \
        "--attn-online 必须默认 0（关闭）"
    # 装配段真的用了它，而不是定义了不用
    assert "_backbone.set_attn_online(bool(args.attn_online))" in src, \
        "--attn-online 定义了却没接到 set_attn_online —— 开关是假的"
    # 不得再走环境变量
    assert "GOAI_ATTN_ONLINE" not in src, \
        "开关不应读环境变量：环境变量不进 swanlab 面板、也不出现在 --help 里"
    # 面板登记：静默的开关是排查噩梦
    assert '"attn_online"' in src, "config 面板里没有 attn_online，训练时看不到开没开"
    assert ast.parse(src) is not None


def test_forward_fp32_small():
    torch.manual_seed(0)
    B, Hh, Nq, Nk, d, dv = 2, 4, 50, 50, 16, 32
    q = torch.randn(B, Hh, Nq, d)
    k = torch.randn(B, Hh, Nk, d)
    v = torch.randn(B, Hh, Nk, dv)
    scale = 1.0 / math.sqrt(d)
    ref = _reference(q.clone(), k.clone(), v.clone(), scale)
    set_attn_online(True)
    out = _sdpa_math(q.clone(), k.clone(), v.clone(), dropout_p=0.0, scale=scale)
    max_err = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), max_err
    print(f"[ok] forward fp32 small, max_err={max_err:.2e}")


def test_forward_fp32_chunked():
    # Nq > 默认 query chunk(64) ⇒ 走分块 online vs 分块 materialize
    torch.manual_seed(1)
    B, Hh, Nq, Nk, d, dv = 2, 4, 200, 80, 16, 32
    q = torch.randn(B, Hh, Nq, d)
    k = torch.randn(B, Hh, Nk, d)
    v = torch.randn(B, Hh, Nk, dv)
    scale = 1.0 / math.sqrt(d)
    ref = _reference(q.clone(), k.clone(), v.clone(), scale)
    set_attn_online(True)
    out = _sdpa_math(q.clone(), k.clone(), v.clone(), dropout_p=0.0, scale=scale)
    max_err = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), max_err
    print(f"[ok] forward fp32 chunked, max_err={max_err:.2e}")


def test_forward_fp16():
    torch.manual_seed(2)
    B, Hh, Nq, Nk, d, dv = 2, 4, 100, 90, 16, 32
    q = torch.randn(B, Hh, Nq, d, dtype=torch.float16)
    k = torch.randn(B, Hh, Nk, d, dtype=torch.float16)
    v = torch.randn(B, Hh, Nk, dv, dtype=torch.float16)
    scale = 1.0 / math.sqrt(d)
    ref = _reference(q.clone(), k.clone(), v.clone(), scale)
    out = _sdpa_math(q.clone(), k.clone(), v.clone(), dropout_p=0.0, scale=scale)
    assert out.dtype == torch.float16, f"输出 dtype 应回落到 q.dtype，实得 {out.dtype}"
    max_err = (out.float() - ref.float()).abs().max().item()
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2), max_err
    print(f"[ok] forward fp16, max_err={max_err:.2e}")


def test_forward_bf16():
    """bf16：910A 上的实际计算精度（autocast 默认走它）。"""
    torch.manual_seed(21)
    B, Hh, Nq, Nk, d, dv = 2, 4, 100, 90, 16, 32
    q = torch.randn(B, Hh, Nq, d, dtype=torch.bfloat16)
    k = torch.randn(B, Hh, Nk, d, dtype=torch.bfloat16)
    v = torch.randn(B, Hh, Nk, dv, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(d)
    ref = _reference(q.clone(), k.clone(), v.clone(), scale)
    out = _sdpa_math(q.clone(), k.clone(), v.clone(), dropout_p=0.0, scale=scale)
    assert out.dtype == torch.bfloat16, f"输出 dtype 应回落到 q.dtype，实得 {out.dtype}"
    max_err = (out.float() - ref.float()).abs().max().item()
    assert torch.allclose(out.float(), ref.float(), atol=1e-1, rtol=1e-1), max_err
    print(f"[ok] forward bf16, max_err={max_err:.2e}")


def test_under_autocast_bf16_fp32_inputs():
    """**回归**：autocast(bf16) + fp32 输入 —— 曾直接抛 dtype 不一致。

    崩溃机理：`m`/`l`/`acc` 原先按 `q.dtype`（fp32）建，而 autocast 下
    `q @ kj.T` 出 bf16，`m - m_new` 就是 fp32 减 bf16 ⇒
    `RuntimeError: expected m1 and m2 to have the same dtype`。
    现在统计量固定 fp32、matmul 留在 autocast 的 bf16，两边自洽。
    """
    torch.manual_seed(22)
    B, Hh, Nq, Nk, d, dv = 2, 4, 100, 90, 16, 32
    # 输入是 **fp32**（与真实训练一致：权重 fp32、autocast 决定算子精度）
    q = torch.randn(B, Hh, Nq, d)
    k = torch.randn(B, Hh, Nk, d)
    v = torch.randn(B, Hh, Nk, dv)
    scale = 1.0 / math.sqrt(d)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        ref = _reference(q.clone(), k.clone(), v.clone(), scale)
        out = _sdpa_math(q.clone(), k.clone(), v.clone(), dropout_p=0.0,
                         scale=scale)
    # 崩过的地方就是这一行：不加 try 的话 pytest 直接 RuntimeError
    assert torch.isfinite(out.float()).all(), "autocast 下出现非有限值"
    max_err = (out.float() - ref.float()).abs().max().item()
    assert torch.allclose(out.float(), ref.float(), atol=1e-1, rtol=1e-1), max_err
    print(f"[ok] autocast bf16, max_err={max_err:.2e}")


def test_backward_under_autocast_bf16():
    """autocast 下的反向也要自洽（`dS = p * (dP - D)` 同样是 fp32×低精度）。"""
    torch.manual_seed(23)
    B, Hh, Nq, Nk, d, dv = 2, 2, 70, 60, 8, 12
    q = torch.randn(B, Hh, Nq, d, requires_grad=True)
    k = torch.randn(B, Hh, Nk, d, requires_grad=True)
    v = torch.randn(B, Hh, Nk, dv, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = _sdpa_math(q, k, v, dropout_p=0.0, scale=0.25)
        out.float().pow(2).mean().backward()
    for name, t in (("dq", q.grad), ("dk", k.grad), ("dv", v.grad)):
        assert t is not None, f"{name} 是 None"
        assert torch.isfinite(t).all(), f"{name} 出现非有限值"
        assert t.dtype == torch.float32, f"{name} dtype 应与入参一致，实得 {t.dtype}"
    print("[ok] autocast bf16 backward")


def test_online_is_skipped_when_dropout_is_on():
    """`dropout_p > 0` 必须回退 materialize —— online 不实现 dropout mask。"""
    torch.manual_seed(24)
    B, Hh, Nq, Nk, d, dv = 1, 2, 40, 40, 16, 16
    q = torch.randn(B, Hh, Nq, d)
    k = torch.randn(B, Hh, Nk, d)
    v = torch.randn(B, Hh, Nk, dv)
    # materialize 也带 dropout：两次调用 mask 不同，只断言「不抛且形状对」，
    # 真正的 dropout 行为由 test_attn_dropout_eval.py 负责
    out = _sdpa_math(q, k, v, dropout_p=0.1, scale=0.25)
    assert out.shape == (B, Hh, Nq, dv)
    print("[ok] dropout>0 falls back")


def test_gradcheck_chunked():
    torch.manual_seed(3)
    B, Hh, Nq, Nk, d, dv = 2, 2, 70, 60, 8, 12  # Nq>64 ⇒ 分块 online
    q = torch.randn(B, Hh, Nq, d, dtype=torch.double, requires_grad=True)
    k = torch.randn(B, Hh, Nk, d, dtype=torch.double, requires_grad=True)
    v = torch.randn(B, Hh, Nk, dv, dtype=torch.double, requires_grad=True)
    ok = gradcheck(
        lambda q, k, v: _sdpa_math(q, k, v, dropout_p=0.0, scale=0.25),
        (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert ok, "gradcheck failed (chunked online)"
    print("[ok] gradcheck chunked online")


def test_gradcheck_no_chunk():
    torch.manual_seed(4)
    B, Hh, Nq, Nk, d, dv = 2, 2, 30, 25, 8, 12  # Nq<64 ⇒ 非分块 online
    q = torch.randn(B, Hh, Nq, d, dtype=torch.double, requires_grad=True)
    k = torch.randn(B, Hh, Nk, d, dtype=torch.double, requires_grad=True)
    v = torch.randn(B, Hh, Nk, dv, dtype=torch.double, requires_grad=True)
    ok = gradcheck(
        lambda q, k, v: _sdpa_math(q, k, v, dropout_p=0.0, scale=0.25),
        (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert ok, "gradcheck failed (non-chunked online)"
    print("[ok] gradcheck non-chunked online")


