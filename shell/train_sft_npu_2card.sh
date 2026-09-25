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

# ---- SwanLab（可选依赖）----
# 云端环境不保证已装 swanlab。装不上也不中断训练（set -e 下用 || 兜底），
# 此时指标仅从 stdout 查看（每 --log-every 步一行）。
python -m pip install swanlab -q || \
  echo "[warn] swanlab 安装失败（多为无外网），将仅用 stdout 记录指标"

# ---- 可调参数 ----
# 910A 是 32GB 卡：BATCH=3200 实测 OOM（backward 的 BatchMatMul 申请 1.46GB
# 时 rtMalloc 失败），BATCH=2800 实测可用。每卡显存与卡数无关，故 1/2/4 卡
# 的每卡安全上限都是 2800。
# 统一标定式：LR = 0.00356 × √(有效batch / 2500)
#   → 0.00356 × √(5600/2500) = 0.00533
# 有效 batch = BATCH × WORLD_SIZE
WORLD_SIZE=2
BATCH=2800            # 每卡 2800 × 2 卡 = 有效 5600
LR=0.00533            # 按有效 batch 5600 标定：0.00356×√(5600/2500)
PREFETCH_W=12         # 每 rank 12 个，2 卡合计 24（对齐 24 核，避免进程超订）
PREFETCH_D=16         # 在途 batch 数；每个约 49MB(B=2800)，16→约0.78GB/rank
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_v18_npu2.pth
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

# ---- 2 卡 DDP 注意事项 ----
# · 每个子进程会**独立加载全量数据集**（34M 样本），主机内存约为单卡的 2 倍
# · DDP 的梯度 allreduce 是每步硬同步：HCCL 若不稳定，可能比单卡更慢甚至 hang
#   （此前记录过 4×910A 网络不稳）。若卡间通信有问题，把 WORLD_SIZE 改回 1。
# · MASTER_ADDR/MASTER_PORT 由 torchrun 自动注入，无需手动设置
# · 不传 --compile / --flash-attn：NPU 上自动禁用
# · C2NET 注意：prepare() 与 --data 覆盖**故意**在所有 rank 上执行（每个 rank 都要
#   独立加载数据集，而 --data 是 required=True，非 rank 0 拿不到路径会直接退出）。
#   日志已按 is_main 过滤，不会刷重复行。若 c2net 的 prepare() 有写盘副作用，
#   正确修法是挪到 init_process_group 之后由 rank 0 调用再广播路径。
# --attn-window 固定 5：实测 ws=5 比 ws=7 少 31% MACs 而显存持平
#   （ws=3 虽更省算但注意力激活约 3.1GB，会爆）。

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
  --scaler-init-scale "$SCALER_INIT" --scaler-growth-interval "$SCALER_GROWTH" \\
  --prefetch-workers "$PREFETCH_W" --prefetch-depth "$PREFETCH_D" \
  --log-every 50 --swanlab-every 10 --eval-every 2000 --save-every 500 \
  --early-stop 1 --early-stop-patience 3 \
  --out "$OUT" \
  --export-onnx 1 --use-checkpoint 1 \
  --swanlab 1 --ver v18_npu2 \
  --c2net "$C2NET" \
  "$@"
