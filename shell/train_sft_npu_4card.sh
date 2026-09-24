#!/usr/bin/env bash
# SFT 四卡 910A（DDP / HCCL）
#
# 用法:
#   python run.py --sh shell/train_sft_npu_4card.sh
#   python run.py --sh shell/train_sft_npu_4card.sh --epochs 3
#
# 末尾的 "$@" 让追加参数覆盖默认值（argparse 取最后一个）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- 可调参数 ----
# 每卡 batch 取 1/4，使「有效 batch」≈ 单卡的 3200，LR 曲线可比。
WORLD_SIZE=4
BATCH=800             # 800 × 4 = 3200
LR=0.00403            # 与单卡同有效 batch，故 lr 相同
PREFETCH_W=16         # 每 rank 16 个子进程，4 卡共 64 个，注意别超订 CPU
PREFETCH_D=16
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_v18_npu4.pth

# ---- ⚠ 4 卡 DDP 风险提示（重要）----
# 此前记录过 4×910A 卡间网络不稳。DDP 的梯度 allreduce 是**每步硬同步**，
# 任一卡通信抖动都会拖住全部 rank，表现为吞吐骤降甚至 hang。
# 建议先用 shell/train_sft_npu_1card.sh 或 _2card.sh 验证通信，
# 确认稳定后再上 4 卡。若卡间通信有问题，把 WORLD_SIZE 调小。
# 每卡 batch 很小（800）时单卡利用率不高，4 卡的吞吐优势可能被通信开销抵消；
# 若要追求吞吐，可提高 BATCH，但必须同步按平方根律提高 LR 并重新验证。
#
# 其它：每 rank 独立加载全量数据集，主机内存约为单卡的 4 倍；
# 不传 --compile / --flash-attn（NPU 自动禁用）；降显存不要动 --attn-window。

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
  --log-every 50 --swanlab-every 10 --eval-every 1000 --save-every 500 \
  --early-stop 1 --early-stop-patience 3 \
  --out "$OUT" \
  --export-onnx 1 --use-checkpoint 1 \
  --swanlab 1 --ver v18_npu4 \
  "$@"
