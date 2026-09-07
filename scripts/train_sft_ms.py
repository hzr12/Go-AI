#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MindSpore 版 SFT 训练（Ascend 910B，GRAPH_MODE + BF16 + 动态损失缩放）。

与 scripts/train_sft.py 的对应关系
----------------------------------
- 数据：复用 src/data/dataset.py 的 SupervisedDataset（走 sample_batch_numpy，
  不依赖 torch），因此 npz 格式、12 通道特征、8 对称增强与 torch 版完全一致。
- 损失：与 torch 版一致 —— cross_entropy(policy) + mse(value)，权重 1:1。
- 调度：warmup(5%) + cosine，逐 step，与 torch 版 SequentialLR(LinearLR,
  CosineAnnealingLR) 等价。
- 优化器：AdamWeightDecay（含 weight decay），对标 torch AdamW。
  注：torch 版用 Adam + weight_decay=1e-4，此处 AdamWeightDecay 语义等价。
- 混合精度：ms.amp.auto_mixed_precision(amp_level="O2") + DynamicLossScaleManager，
  对标 torch 的 autocast(bfloat16) + GradScaler。

用法（910B 机器上）
------------------
python scripts/train_sft_ms.py \
    --data data/sgf_19x19_all.npz \
    --board-size 19 \
    --backbone-channels 192 --backbone-res-blocks 17 \
    --batch-size 512 --epochs 8 --lr 0.003 --weight-decay 0.0001 \
    --attn-mode window --attn-window 7 \
    --out models/sft_19x19_big.ckpt

训练后转回 torch 权重给现有推理/MCTS 用：
    python scripts/convert_ckpt.py to-npz   --in models/sft_19x19_big.ckpt --out big.npz --framework ms
    python scripts/convert_ckpt.py from-npz --in big.npz --out models/sft_19x19_big.pth --framework torch
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import amp as ms_amp

from src.data.dataset import SupervisedDataset
from src.networks_ms.alphanet_ms import AlphaGoNet


# --------------------------------------------------------------------------- #
# 损失 + 训练一步（带损失缩放）
# --------------------------------------------------------------------------- #
class GoAILossCell(nn.Cell):
    """policy CE + value MSE（与 torch 版 1:1 权重）。"""

    def __init__(self, net):
        super(GoAILossCell, self).__init__()
        self.net = net
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

    def construct(self, x, move_t, value_t):
        policy, value = self.net(x)
        # value: (B,1)；value_t: (B,1)。MSE 按元素平均
        return self.ce(policy, move_t) + self.mse(value, value_t)


# --------------------------------------------------------------------------- #
# 学习率：warmup + cosine（对标 torch SequentialLR(LinearLR, CosineAnnealingLR)）
# --------------------------------------------------------------------------- #
def build_lr_array(total_steps, warmup_steps, lr_max, lr_start_ratio=0.1):
    """返回 shape=(total_steps,) 的 float32 数组，交给优化器做逐 step 动态 LR。"""
    lrs = np.zeros(total_steps, dtype=np.float32)
    lr_start = lr_max * lr_start_ratio
    for i in range(total_steps):
        if i < warmup_steps:
            # LinearLR: start_factor=0.1 -> 1.0
            lrs[i] = lr_start + (lr_max - lr_start) * (i / max(1, warmup_steps))
        else:
            # CosineAnnealingLR: T_max = total - warmup
            prog = (i - warmup_steps) / max(1, total_steps - warmup_steps)
            lrs[i] = lr_max * 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))
    return lrs


# --------------------------------------------------------------------------- #
def evaluate_top1(net, ds, idxs, bs, board_size, max_batches=None):
    """验证集 top-1 着法准确率（GRAPH_MODE 下逐批前向）。"""
    net.set_train(False)
    correct = 0
    total = 0
    n_batches = min((len(idxs) + bs - 1) // bs, max_batches or 10 ** 9)
    for b in range(n_batches):
        sel = idxs[b * bs:(b + 1) * bs]
        if len(sel) == 0:
            break
        states, moves, _ = ds.sample_batch_numpy(sel)
        policy, _ = net(Tensor(states, ms.float32))
        pred = ops.ArgMaxWithValue(axis=-1)(policy)[0].asnumpy()
        correct += int((pred == moves).sum())
        total += len(sel)
    net.set_train(True)
    return correct / max(total, 1), total


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz 训练集")
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--backbone-channels", type=int, default=128)
    ap.add_argument("--backbone-res-blocks", type=int, default=12)
    ap.add_argument("--attention-mode", default="mix", choices=["none", "mix", "all"])
    ap.add_argument("--num-attention-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--attention-dropout", type=float, default=0.0)
    ap.add_argument("--attn-mode", default="global",
                    choices=["global", "window", "axial", "sparse"])
    ap.add_argument("--attn-window", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", default="models/sft_ms.ckpt")
    ap.add_argument("--init", default="",
                    help="可选：MindSpore .ckpt 权重热启动（用 convert_ckpt.py 从 torch 转来）")
    ap.add_argument("--no-amp", action="store_true", help="关闭 O2 混合精度（调试用）")
    args = ap.parse_args()

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=args.device_id)
    ms.set_seed(0)

    n_actions = args.board_size * args.board_size + 1

    # ---- 数据 ----
    d = np.load(args.data)
    ds = SupervisedDataset({k: d[k] for k in d.files})
    N = len(ds)
    n_train = int(N * 0.98)
    idx_all = np.arange(N)
    rng = np.random.default_rng(0)
    rng.shuffle(idx_all)
    train_idx, eval_idx = idx_all[:n_train], idx_all[n_train:]
    print(f"[data] 总样本数={N} | 训练={n_train} | 验证={len(eval_idx)}", flush=True)

    # ---- 模型 ----
    net = AlphaGoNet(
        in_channels=12,
        backbone_channels=args.backbone_channels,
        backbone_res_blocks=args.backbone_res_blocks,
        attention_mode=args.attention_mode,
        num_attention_layers=args.num_attention_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        attn_mode=args.attn_mode,
        attn_window=args.attn_window,
        action_size=n_actions,
    )
    n_params = sum(p.size for p in net.get_parameters())
    print(f"[model] 参数量={n_params / 1e6:.2f}M | 设备=Ascend:{args.device_id}", flush=True)

    if args.init:
        param_dict = ms.load_checkpoint(args.init)
        ms.load_param_into_net(net, param_dict)
        print(f"[init] 已从 {args.init} 热启动", flush=True)

    # ---- 混合精度 + 损失缩放 ----
    if not args.no_amp:
        net = ms_amp.auto_mixed_precision(net, amp_level="O2")

    loss_cell = GoAILossCell(net)

    bs = args.batch_size
    steps_per_epoch = max(1, (len(train_idx) + bs - 1) // bs)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.05))
    lr_array = build_lr_array(total_steps, warmup_steps, args.lr)
    print(f"[train] 开始训练 | steps/epoch={steps_per_epoch} | 总 steps={total_steps} "
          f"| warmup={warmup_steps}", flush=True)

    opt = nn.AdamWeightDecay(net.trainable_params(),
                             learning_rate=Tensor(lr_array, ms.float32),
                             weight_decay=args.weight_decay)

    if not args.no_amp:
        manager = ms_amp.DynamicLossScaleManager(init_loss_scale=65536,
                                                 scale_factor=2, scale_window=2000)
        train_net = nn.TrainOneStepWithLossScaleCell(
            loss_cell, opt, scale_sense=manager.get_loss_scale())
    else:
        train_net = nn.TrainOneStepCell(loss_cell, opt)

    train_net.set_train()

    # ---- 训练循环 ----
    t0 = time.time()
    step = 0
    running = 0.0
    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        for b in range(steps_per_epoch):
            sel = train_idx[b * bs:(b + 1) * bs]
            if len(sel) == 0:
                continue
            states, moves, values = ds.sample_batch_numpy(sel)
            loss = train_net(Tensor(states, ms.float32),
                             Tensor(moves, ms.int32),
                             Tensor(values, ms.float32))
            running += float(loss.asnumpy())
            step += 1

            if step % args.log_every == 0:
                print(f"[step {step}/{total_steps}] loss={running / args.log_every:.4f} "
                      f"lr={lr_array[min(step, total_steps - 1)]:.3e} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
                running = 0.0

            if args.eval_every and step % args.eval_every == 0:
                acc, n = evaluate_top1(net, ds, eval_idx, bs, args.board_size)
                print(f"[eval] step={step} eval_top1={acc:.4f} (n={n})", flush=True)

            if args.save_every and step % args.save_every == 0:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                ms.save_checkpoint(net, args.out)
                print(f"  -> 已保存 {args.out}", flush=True)

        print(f"epoch {epoch + 1}/{args.epochs} done", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ms.save_checkpoint(net, args.out)
    print(f"训练完成，模型已保存至 {args.out}；总耗时 {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
