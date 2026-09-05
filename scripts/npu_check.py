#!/usr/bin/env python3
"""NPU（910A）分级自检：定位卡死/报错发生在哪一级。

用法:
    python scripts/npu_check.py                 # 不带模型，只测算子
    python scripts/npu_check.py --model models/sft_19x19_v3.pth
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def stage(name, fn):
    print(f"\n[stage] {name} ...", flush=True)
    t0 = time.time()
    try:
        r = fn()
        print(f"  OK  {time.time() - t0:.2f}s" + (f"  {r}" if r is not None else ""),
              flush=True)
        return r
    except Exception as e:
        print(f"  FAIL {time.time() - t0:.2f}s  {type(e).__name__}: {e}", flush=True)
        print("  ↑ 卡死/报错发生在这一级，把上面的 stage 名和报错发我", flush=True)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--board-size", type=int, default=19)
    args = ap.parse_args()

    stage("1. import torch", lambda: __import__("torch").__version__)

    def imp_npu():
        import torch_npu  # noqa: F401
        return "torch_npu " + getattr(torch_npu, "__version__", "?")
    stage("2. import torch_npu", imp_npu)

    stage("3. torch.npu.is_available()", lambda: str(torch.npu.is_available()))

    def small_op():
        a = torch.randn(64, 64).to("npu")
        b = a @ a
        torch.npu.synchronize()
        return f"matmul ok {b.shape}"
    stage("4. 小算子 matmul + synchronize", small_op)

    def conv():
        conv = torch.nn.Conv2d(12, 128, 3, padding=1).to("npu")
        y = conv(torch.randn(8, 12, 19, 19).to("npu"))
        torch.npu.synchronize()
        return f"conv ok {y.shape}"
    stage("5. conv2d 大通道 batch=8（CANN 初始化重灾区）", conv)

    def bmm_softmax():
        q = torch.randn(256, 4, 1, 49).to("npu")
        k = torch.randn(256, 4, 49, 64).to("npu")
        att = (q @ k).softmax(dim=-1)
        torch.npu.synchronize()
        return f"bmm+softmax ok {att.shape}"
    stage("6. bmm+softmax（窗口注意力核心路径）", bmm_softmax)

    from src.networks.alphanet import AlphaGoNet

    def model_fp32():
        m = AlphaGoNet(in_channels=12, action_size=args.board_size ** 2 + 1).to("npu")
        y = m(torch.randn(1, 12, args.board_size, args.board_size).to("npu"))
        torch.npu.synchronize()
        return f"fp32 forward ok policy={y[0].shape}"
    stage("7. 完整模型 fp32 前向", model_fp32)

    def model_fp16():
        m = AlphaGoNet(in_channels=12, action_size=args.board_size ** 2 + 1).to("npu")
        with torch.autocast(device_type="npu", dtype=torch.float16):
            y = m(torch.randn(1, 12, args.board_size, args.board_size).to("npu"))
        torch.npu.synchronize()
        return "fp16 autocast forward ok"
    stage("8. 完整模型 fp16 autocast 前向", model_fp16)

    if args.model:
        def goai():
            from src.inference import GoAI
            from src.game.go_rules import GoBoard
            ai = GoAI(model_path=args.model, board_size=args.board_size,
                      device="npu", use_amp=True, attn_mode="window", attn_window=7)
            pol, val = ai.predict(GoBoard(args.board_size), [-1, -1, -1], [-1, -1, -1], 1)
            return f"GoAI.predict ok wr={val:+.3f}"
        stage("9. GoAI 完整推理（含 warmup）", goai)

    print("\n全部通过——NPU 链路正常，问题在更上层（把卡住的脚本名和最后输出发我）")


if __name__ == "__main__":
    main()
