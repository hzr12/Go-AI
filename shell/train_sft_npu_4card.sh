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

# ---- SwanLab（可选依赖）----
# 云端环境不保证已装 swanlab。装不上也不中断训练（set -e 下用 || 兜底），
# 此时指标仅从 stdout 查看（每 --log-every 步一行）。
python -m pip install swanlab -q || \
  echo "[warn] swanlab 安装失败（多为无外网），将仅用 stdout 记录指标"

# ---- 可调参数 ----
# 910A 是 32GB 卡：BATCH=3200 实测 OOM（backward 的 BatchMatMul 申请 1.46GB
# 时 rtMalloc 失败，报 driver error:out of memory），BATCH=2800 实测可用。
# 每卡显存与卡数无关，故 1/2/4 卡的每卡安全上限都是 2800。
# 有效 batch = BATCH × WORLD_SIZE；LR 按平方根缩放律同步。
# 统一标定式：LR = 0.00356 × √(有效batch / 2500)
#   → 0.00356 × √(11200/2500) = 0.00754
WORLD_SIZE=4
BATCH=2800            # 每卡 2800 × 4 卡 = 有效 11200
LR=0.00754            # 按有效 batch 11200 标定：0.00356×√(11200/2500)
PREFETCH_W=6          # 每 rank 6 个，4 卡合计 24（对齐 24 核）。
                      # 注意：worker 是**每 rank** 各起这么多，早期写 32 时
                      # 4 卡会起 128 个数据构造进程挤 24 核，反而严重拖慢。
PREFETCH_D=16         # 在途 batch 数；每个约 49MB(B=2800)，16→约0.78GB/rank
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_v18_npu4.pth
C2NET=1               # 1=启用 OpenI 启智平台对接；平台已自带数据/输出时改 0

# ---- ⚠ 4 卡 DDP 风险提示（重要）----
# 此前记录过 4×910A 卡间网络不稳。DDP 的梯度 allreduce 是**每步硬同步**，
# 任一卡通信抖动都会拖住全部 rank，表现为吞吐骤降甚至 hang。
# 建议先用 shell/train_sft_npu_1card.sh 或 _2card.sh 验证通信，
# 确认稳定后再上 4 卡。若卡间通信有问题，把 WORLD_SIZE 调小。
# 每卡 batch 很小（800）时单卡利用率不高，4 卡的吞吐优势可能被通信开销抵消；
# 若要追求吞吐，可提高 BATCH，但必须同步按平方根律提高 LR 并重新验证。
#
# --attn-window 固定 5：实测 ws=5 比 ws=7 少 31% MACs 而显存持平
#   （ws=3 虽更省算但注意力激活约 3.1GB，会爆）。
# C2NET 注意：prepare() 与 --data 覆盖**故意**在所有 rank 上执行（每个 rank 都要
#   独立加载数据集，而 --data 是 required=True，非 rank 0 拿不到路径会直接退出）。
#   日志已按 is_main 过滤，不会刷重复行。若 c2net 的 prepare() 有写盘副作用，
#   正确修法是挪到 init_process_group 之后由 rank 0 调用再广播路径。

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
  --prefetch-workers "$PREFETCH_W" --prefetch-depth "$PREFETCH_D" \
  --log-every 50 --swanlab-every 10 --eval-every 1000 --save-every 500 \
  --early-stop 1 --early-stop-patience 3 \
  --out "$OUT" \
  --export-onnx 1 --use-checkpoint 1 \
  --swanlab 1 --ver v18_npu4 \
  --c2net "$C2NET" \
  "$@"
