#!/usr/bin/env bash
# SFT 单卡 910A（FP16 + GradScaler，注意力走手写 math）
#
# 用法:
#   python run.py --sh shell/train_sft_npu_1card.sh
#   python run.py --sh shell/train_sft_npu_1card.sh --epochs 3
#
# 末尾的 "$@" 让追加参数覆盖默认值（argparse 取最后一个）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- 可调参数（OOM 时优先降 BATCH，并同步按平方根律降 LR）----
WORLD_SIZE=1
BATCH=3200            # 910A 实测可用；有效 batch = BATCH × WORLD_SIZE
LR=0.00403            # 0.00356 × √(3200/2500)，平方根缩放律
PREFETCH_W=32
PREFETCH_D=32
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_v18_npu1.pth

# ---- NPU 注意事项 ----
# · 910A 无 BF16 → 代码自动走 FP16 + GradScaler，无需手改
# · 不传 --compile：NPU 无 inductor，代码会自动禁用并打警告
# · 不传 --flash-attn：NPU 上自动禁用，注意力走手写 math
# · NPU 的 pin_memory 关闭 → 训练侧双缓冲 H2D 不生效（那只对 CUDA 有效）
# ⚠ 降显存不要动 --attn-window：window_global 显存正比于 nW×(ws²+ng)，
#   而 ng=ceil(19/ws)² 在 ws 变小时暴涨，要往大调而不是往小调。

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
  --swanlab 1 --ver v18_npu1 \
  "$@"
