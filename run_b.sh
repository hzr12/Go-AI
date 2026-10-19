#!/usr/bin/env bash
# Segment B (soft labels, continues from A). Flags verbatim from run.txt.
#
# Uses whichever `python` is on PATH, so activate the right env first:
#   torch 2.10.0+cu128 (python3.11) is the current cloud env. The attention
#   kernel is torch's built-in SDPA (no flash-attn library needed).
#
# --epochs 5 is deliberate, not a typo: --soft-only-sampling narrows the
#   training set to ~225k rows, so 256x1 gives 881 optimizer steps/epoch and
#   1 epoch would anneal the cosine schedule to zero around step ~100.
#   (5 epochs x 881 = 4405 optimizer steps -- matches the measured run.)
#
# --gradient-accumulation-steps 1 (2026-10-18): accum=8 is retired. Every
#   optimizer step is now a real step, so torch.compile can use the
#   reduce-overhead (CUDA Graphs) mode, and the grad-norm D2H sync happens
#   once per step instead of once per 8 micro-batches. Peak LR stays
#   0.00322 -- that is the value the measured accum=1 run used (its swanlab
#   lr curve matches peak=0.00322 exactly at step 975).
#
# Requires models/a_v7_a100.pth.latest from run.sh to exist.
# Usage: bash run_b.sh    (run from the repo root)
set -euo pipefail
# 每个预取 worker 只占 1 核（否则 24 worker × 满核 OMP 线程互相抢核，
# top 退化成单核 100%，见 run.txt 第 21 行）。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

exec python scripts/train_sft.py \
  --data data/sgf_19x19_full.npz --v7 1 --device cuda --board-size 19 \
  --games-npz data/labels/games.npz \
  --model models/a_v7_a100.pth.latest \
  --soft-index data/labels/soft_index.npz --soft-only-sampling 1 \
  --soft-weight 1.0 --soft-every 1 --policy-loss soft_ce \
  --batch-size 256 --epochs 5 --lr 0.00322 --weight-decay 0.0001 \
  --gradient-accumulation-steps 1 --use-amp 1 --attention-dropout 0 \
  --prefetch-workers 24 --prefetch-depth 8 \
  --log-every 25 --eval-every 100 --eval-max-batches 5 --save-every 20 \
  --early-stop 1 --early-stop-patience 3 \
  --out models/b_v7_a100.pth --swanlab 1 --ver b_v7_a100 --swanlab-every 1