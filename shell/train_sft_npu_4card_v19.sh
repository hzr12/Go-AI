#!/usr/bin/env bash
# SFT 四卡 910A —— v19：约 8.77M 参数的精简档（DDP / HCCL）
#
# 用法:
#   python run.py --sh shell/train_sft_npu_4card_v19.sh
#   python run.py --sh shell/train_sft_npu_4card_v19.sh --epochs 2
#
# 末尾的 "$@" 让追加参数覆盖默认值（argparse 取最后一个）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- SwanLab（可选依赖）----
# 云端环境不保证已装 swanlab。装不上也不中断训练（set -e 下用 || 兜底）。
python -m pip install swanlab -q || \
  echo "[warn] swanlab 安装失败（多为无外网），将仅用 stdout 记录指标"

# ---- 可调参数 ----
WORLD_SIZE=4
# batch=3000（每卡）× 4 = 有效 12000
# B=3200 曾 OOM，但那是旧 value head(96x8) + attn-window 7 的配置；
# 本档 value 缩到 3 块、attn-window 用 5，实测外推显存 26.7G/32G（余 5.3G）。
BATCH=3000
LR=0.00780            # 0.00356 × √(12000/2500) 平方根缩放律
PREFETCH_W=6          # 每 rank 6 个，4 卡合计 24（对齐 24 核）
PREFETCH_D=16
# GradScaler：910A 走 FP16。从 65536 起步要靠减半搜平衡点（每次溢出白扔一个
# batch），实测本模型平衡于 512~2048。直接给 1024 并关掉回涨，避免震荡偷步。
SCALER_INIT=1024
SCALER_GROWTH=100000
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_v19_npu4.pth
C2NET=1

# NPU TorchAir 图编译（受控实验，默认关）。1=开启，失败自动回退 eager。
# 开启前请注意：常驻显存已 26.7~31.1GB/32GB，图模式的 workspace 可能再炸；
# 且本环境为 torch 2.1.0 / torch_npu 2.1.0.post3 / CANN 8.0.RC1（2023 年代），
# 图编译功能成熟度存疑。**建议先短跑几十步验证再决定是否长训。**
NPU_GRAPH_COMPILE=0

# ---- 为什么是这一档 ----
# v18 是 12.86M / 8.49G FLOPs / 31.2G 显存（32GB 卡的 97.3%，已在悬崖边）。
# 本档 8.77M / 5.97G / 26.7G，吞吐约 4012 samples/s（v18 的 1.52×）。
#
# 保留了什么（都是有意为之）：
#   · backbone_channels 192 —— V12 唯一验证过 top1≈50% 的宽度
#   · attn_blocks 5         —— 与 v18 相同，Go 更吃全局关系
#   · attn_window 5         —— 实测显存最优（ws=3 会到 38.5G）
# 砍了什么：
#   · res_blocks     8 -> 4
#   · convnext_blocks 4 -> 2
#   · value_res_blocks 8 -> 3（value head 不受 --use-checkpoint 保护，
#     每块约 0.46GB 线性吃显存；且它是二分类输出，缩小近乎不损准确率）
#
# 容量比 v18 少 32%。**这是有代价的取舍**：准确率无法本地测量，
# 首次跑请与 v18 的 step-1800 快照（top1=0.2820）对比 loss 与 top1 轨迹。
#
# 工具：python scripts/search_arch.py --curve / --sweep <维度> / --grid
#      python scripts/inspect_ckpt.py <ckpt> --emit-flags
#
# ---- 其它注意 ----
# · 910A 无 BF16 → 代码自动走 FP16 + GradScaler
# · 不传 --compile：NPU 无 inductor，代码自动禁用
# · 不传 --flash-attn：NPU 上自动禁用，注意力走手写 math
# · NPU 的 pin_memory 关闭 → 训练侧双缓冲 H2D 不生效（那只对 CUDA 有效）
# · 4 卡 DDP 的 allreduce 是每步硬同步，任一卡通信抖动都会拖住全部 rank。
#   此前记录过 4×910A 卡间网络不稳，若吞吐骤降或 hang，先用 1/2 卡验证通信。
# · C2NET：prepare() 与 --data 覆盖**故意**在所有 rank 上执行（每个 rank 都要
#   独立加载数据集，而 --data 是 required=True，非 rank 0 拿不到路径会直接退出）。
#   日志已按 is_main 过滤。若 c2net 的 prepare() 有写盘副作用，正确修法是挪到
#   init_process_group 之后由 rank 0 调用再广播路径。
# · backbone-res-blocks 在分段模式下是**死参数**（backbone.py:613-617 算出的
#   total_blocks 传进 _build_segmented_blocks 后从未被使用），写 17 只是留个
#   可读的名义值，真正起作用的是 res/convnext/attn_blocks 三个。

torchrun --nproc_per_node="$WORLD_SIZE" scripts/train_sft.py \
  --data "$DATA" \
  --device npu --board-size 19 \
  --backbone-channels 192 --backbone-res-blocks 17 \
  --res-blocks 4 --convnext-blocks 2 --attn-blocks 5 \
  --value-channels 96 --value-res-blocks 3 \
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
  --log-every 50 --swanlab-every 10 --eval-every 500 --save-every 250 \
  --early-stop 1 --early-stop-patience 3 \
  --out "$OUT" \
  --export-onnx 1 --use-checkpoint 1 \
  --swanlab 1 --ver v19_npu4 \
  --c2net "$C2NET" \
  "$@"
