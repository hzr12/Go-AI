#!/usr/bin/env bash
# RL 自对弈训练（1 卡 910A 训练 + 8 进程 CPU 做 MCTS 搜索）
#
# 用法:
#   python run.py --sh shell/train_rl_npu_1card.sh
#   python run.py --sh shell/train_rl_npu_1card.sh --sims 64
#   bash shell/train_rl_npu_1card.sh --td 0          # 跑旧 tanh 软化基线
#
# 末尾的 "$@" 让追加参数覆盖默认值（argparse 取最后一个）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- 可调参数 ----
PARALLEL_GAMES=8      # 自对弈进程数
MCTS_THREADS=3        # 每进程 MCTS 线程；8×3≈24，正好对上 24 核
SIMS=48               # 每步 MCTS 模拟数（默认 400 对 19 路太慢，48 是 CPU 可用档）
MODEL=models/sft_19x19_v18.pth   # SFT 初始权重
OUT=models/az_best.pth

# ---- ⚠ 关键注意事项 ----
# 1. **不要传 --onnx-model**：当前实现把 .onnx 路径当 torch 权重 torch.load，
#    而 ONNX 是 protobuf 格式，启动即抛 UnpicklingError（已实测）。
#    自对弈推理与训练目前共用同一个 torch 模型（MCTS 推理也在 NPU 上跑），
#    功能可用，但没有 CPU/ONNX 加速路径。
# 2. **单进程**：RL 不做 DDP，不需要 torchrun。NPU 只用于训练，
#    MCTS 搜索由 8 个 CPU 进程承担。
# 3. --td 1 是 C+E 混合价值标签（v_td + r_soft）；--td 0 为旧 tanh 软化基线，
#    做 A/B 对比时用 --td 0 覆盖。
# 4. --spec-prefetch 只有 --mcts-threads>=4 时才生效，当前 3 线程下不启用。
# 5. 同步模式（不加 --async-pipeline）：异步模式虽能提速 1.5-2x，但会改变
#    数据生成与训练的时序，调试期建议先用同步模式。

python scripts/selfplay_train.py \
  --device npu \
  --board-size 19 --iters 20 --sims "$SIMS" \
  --parallel-games "$PARALLEL_GAMES" \
  --mcts-threads "$MCTS_THREADS" \
  --batch-cap 64 \
  --expand-topk 16 \
  --td 1 \
  --buffer-size 500 \
  --use-rollout 1 --leaf-ab-depth 2 \
  --c-puct 2.0 --virtual-loss 8.0 \
  --grad-accum-steps 1 \
  --mcts-vector-backup 1 \
  --swanlab 1 \
  --model "$MODEL" \
  --out "$OUT" \
  "$@"
