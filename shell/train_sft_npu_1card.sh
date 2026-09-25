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

# ---- SwanLab（可选依赖）----
# 云端环境不保证已装 swanlab。装不上也不中断训练（set -e 下用 || 兜底），
# 此时指标仅从 stdout 查看（每 --log-every 步一行）。
python -m pip install swanlab -q || \
  echo "[warn] swanlab 安装失败（多为无外网），将仅用 stdout 记录指标"

# ---- 可调参数（OOM 时优先降 BATCH，并同步按平方根律降 LR）----
WORLD_SIZE=1
# 910A 是 32GB 卡：BATCH=3200 实测 OOM（backward 的 BatchMatMul 申请 1.46GB
# 时 rtMalloc 失败，报 driver error:out of memory），BATCH=2800 实测可用。
# 注意每卡显存与卡数无关，所以 2/4 卡同样是 3200 装不下。
BATCH=2800            # 32GB 卡实测可用；有效 batch = BATCH × WORLD_SIZE
LR=0.00377            # 0.00356 × √(2800/2500)，平方根缩放律
PREFETCH_W=24         # 预取进程数（单卡即全部），24 核机器铺满
PREFETCH_D=16         # 在途 batch 数；每个约 49MB(B=2800)，16→约0.78GB
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_v18_npu1.pth
C2NET=1               # 1=启用 OpenI 启智平台对接；平台已自带数据/输出时改 0

# NPU TorchAir 图编译（受控实验，默认关）。1=开启，失败自动回退 eager。
# 开启前请注意：常驻显存已 26.7~31.1GB/32GB，图模式的 workspace 可能再炸；
# 且本环境为 torch 2.1.0 / torch_npu 2.1.0.post3 / CANN 8.0.RC1（2023 年代），
# 图编译功能成熟度存疑。**建议先短跑几十步验证再决定是否长训。**
NPU_GRAPH_COMPILE=0
# GradScaler 策略。910A 走 FP16 + GradScaler，默认从 65536 起步要靠减半
# 向下搜平衡点（每次溢出白扔一个 batch），实测本模型平衡于 512~2048。
# 直接给 1024 起步并把回涨间隔设成 100000（约等于关闭），避免缩放值在平衡点
# 附近反复翻倍/减半震荡、持续偷步。设 0 则沿用 PyTorch 默认 65536 / 2000。
SCALER_INIT=1024
SCALER_GROWTH=100000

# ---- NPU 注意事项 ----
# · 910A 无 BF16 → 代码自动走 FP16 + GradScaler，无需手改
# · 不传 --compile：NPU 无 inductor，代码会自动禁用并打警告
# · 不传 --flash-attn：NPU 上自动禁用，注意力走手写 math
# · NPU 的 pin_memory 关闭 → 训练侧双缓冲 H2D 不生效（那只对 CUDA 有效）
# --attn-window 固定 5，不要改：实测（B=2800/heads=4/d=48）
#   ws=3  QK 28.6M + AV 47M = 76M MACs，但注意力激活约 3.1GB（爆）
#   ws=5  QK 184M + AV 215M = 399M MACs，注意力激活约 1.07GB  ← 最优
#   ws=7  QK 287M + AV 237M = 524M MACs，注意力激活约 1.13GB
# ws=5 比 ws=7 少 31% 计算而显存持平。此前「ws 要往大调」的说法只按
# nW×(ws²+ng) 估 K/V 序列、漏掉了查询维 ws²，结论是反的。

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
  --attn-mode window_global --attn-window 5 \
  --attention-dropout 0.1 --label-smoothing 0.1 \
  --gradient-accumulation-steps 1 \
  --use-amp 1 --use-ema 1 \
  --npu-graph-compile "$NPU_GRAPH_COMPILE" \
  --scaler-init-scale "$SCALER_INIT" --scaler-growth-interval "$SCALER_GROWTH" \
  --prefetch-workers "$PREFETCH_W" --prefetch-depth "$PREFETCH_D" \
  --log-every 50 --swanlab-every 10 --eval-every 2000 --save-every 500 \
  --early-stop 1 --early-stop-patience 3 \
  --out "$OUT" \
  --export-onnx 1 --use-checkpoint 1 \
  --swanlab 1 --ver v18_npu1 \
  --c2net "$C2NET" \
  "$@"
