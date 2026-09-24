#!/usr/bin/env bash
# SFT 单卡 A100 40G（BF16 + flash-attn）—— 平台实测可跑配置
#
# 用法:
#   python run.py --sh shell/train_sft_a100_1card.sh
#   python run.py --sh shell/train_sft_a100_1card.sh --epochs 3   # 覆盖默认值
#   bash shell/train_sft_a100_1card.sh --out models/exp.pth      # 也可单独跑
#
# 末尾的 "$@" 让追加参数覆盖默认值（argparse 取最后一个）。
set -euo pipefail

# 无论从哪个目录调用，都定位到仓库根，保证下面的相对路径一致
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- SwanLab（可选依赖）----
# 云端环境不保证已装 swanlab，这里统一装一次。
# 装不上也不能中断训练：脚本有 set -euo pipefail，故用 || 兜底降级；
# 失败时指标仍可从 stdout 看（每 --log-every 步一行）。
python -m pip install swanlab -q || \
  echo "[warn] swanlab 安装失败（多为无外网），将仅用 stdout 记录指标"

# ---- 可调参数（OOM 时优先降 BATCH，并同步按平方根律降 LR）----
WORLD_SIZE=1
BATCH=3500            # 实测 3500 可跑（40G）
LR=0.00421            # 0.00356 × √(3500/2500)，平方根缩放律
PREFETCH_W=32
PREFETCH_D=32
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_v18.pth
C2NET=1               # 1=启用 OpenI 启智平台对接；平台已自带数据/输出时改 0

# ---- 为什么以前 OOM、现在能到 3500 ----
# 根因不是 batch 本身，而是 --compile-mode reduce-overhead：它走 CUDA Graphs，
# 会维持一个**不归还**的私有内存池，在临界 batch 上直接吃光显存。
# 去掉该参数（只留 --compile 1）后 3500 实测通过。
# 另外两处：--value-res-blocks 11→8（value head 不受 --use-checkpoint 保护，
# 是最大一块常驻激活）、--attn-window 5→7。
# ⚠ 降显存不要动 --attn-window：window_global 显存正比于 nW×(ws²+ng)，
#   而 ng=ceil(19/ws)² 在 ws 变小时暴涨（ws=3 比 ws=7 差 5.4×），要往大调。

torchrun --nproc_per_node="$WORLD_SIZE" scripts/train_sft.py \
  --data "$DATA" \
  --device cuda --board-size 19 \
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
  --compile 1 \
  --flash-attn 1 \
  --prefetch-workers "$PREFETCH_W" --prefetch-depth "$PREFETCH_D" \
  --log-every 50 --swanlab-every 10 --eval-every 2000 --save-every 500 \
  --early-stop 1 --early-stop-patience 3 \
  --out "$OUT" \
  --export-onnx 1 --use-checkpoint 1 \
  --swanlab 1 --ver v18 \
  --c2net "$C2NET" \
  "$@"
