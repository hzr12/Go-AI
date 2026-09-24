#!/usr/bin/env bash
# SFT 双卡 910A（DDP / HCCL）
#
# 用法:
#   python run.py --sh shell/train_sft_npu_2card.sh
#   python run.py --sh shell/train_sft_npu_2card.sh --epochs 3
#
# 末尾的 "$@" 让追加参数覆盖默认值（argparse 取最后一个）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- 可调参数 ----
# 每卡 batch 减半，使「有效 batch」与单卡保持一致（3200），这样 LR 曲线可比。
# 有效 batch = BATCH × WORLD_SIZE
WORLD_SIZE=2
BATCH=1600            # 1600 × 2 = 3200，与单卡 910A 一致
LR=0.00403            # 与单卡同有效 batch，故 lr 相同
PREFETCH_W=16         # 每 rank 各自起 PREFETCH_W 个子进程，2 卡共 32 个
PREFETCH_D=16
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_v18_npu2.pth

# ---- 2 卡 DDP 注意事项 ----
# · 每个子进程会**独立加载全量数据集**（34M 样本），主机内存约为单卡的 2 倍
# · DDP 的梯度 allreduce 是每步硬同步：HCCL 若不稳定，可能比单卡更慢甚至 hang
#   （此前记录过 4×910A 网络不稳）。若卡间通信有问题，把 WORLD_SIZE 改回 1。
# · MASTER_ADDR/MASTER_PORT 由 torchrun 自动注入，无需手动设置
# · 不传 --compile / --flash-attn：NPU 上自动禁用
# ⚠ 降显存不要动 --attn-window：要往大调而不是往小调。

torchrun --nproc_per_node="$WORLD_SIZE" scripts/train_sft.py \
  --data "$DATA" \
  --device npu --board-size 19 \
  --backbone-channels 192 --backbone-res-blocks 17 \
  --res-blocks 8 --convnext-blocks 4 --attn-blocks 5 \
  --value-channels 96 --value-res-blocks 8 \
  --policy-channels 128 --policy-layers 3 \
  --batch-size "$BATCH" --epochs 1 \
  --lr "$LR" --weight-decay 0.0001 \
  --attention-mode mix --num-attention-layers 4 --num-heads 4 \
  --attn-mode window_global --attn-window 7 \
  --attention-dropout 0.1 --label-smoothing 0.1 \
  --gradient-accumulation-steps 1 \
  --use-amp 1 --use-ema 1 \
  --prefetch-workers "$PREFETCH_W" --prefetch-depth "$PREFETCH_D" \
  --log-every 50 --swanlab-every 10 --eval-every 2000 --save-every 500 \
  --early-stop 1 --early-stop-patience 3 \
  --out "$OUT" \
  --export-onnx 1 --use-checkpoint 1 \
  --swanlab 1 --ver v18_npu2 \
  "$@"
