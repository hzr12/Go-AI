#!/usr/bin/env bash
# Segment A (SFT). Flags are verbatim from run.txt's "A segment" block.
#
# Why this file instead of copy-pasting out of run.txt:
#   run.txt is UTF-8 Chinese with 709 distinct non-ASCII glyphs. When the
#   terminal locale is not UTF-8 the whole screen turns to mojibake, and
#   hand-copying silently drops characters (seen: "--eval-every 500" turned
#   into "--eval-every>  500"). This file is pure ASCII on purpose.
#
# Uses whichever `python` is on PATH, so activate the right env first:
#   torch 2.10.0+cu128 (python3.11) is the current cloud env. The attention
#   kernel is torch's built-in SDPA (no flash-attn library needed).
#
# --gradient-accumulation-steps 1 (2026-10-18): accum=8 is retired. Every
#   optimizer step is now a real step, so torch.compile can use the
#   reduce-overhead (CUDA Graphs) mode, and the grad-norm D2H sync happens
#   once per step instead of once per 8 micro-batches. Peak LR stays
#   0.00322 -- that is the value the measured accum=1 run used (its swanlab
#   lr curve matches peak=0.00322 exactly at step 3925).
#
# Usage: bash run.sh    (run from the repo root - all paths are relative)
set -euo pipefail
# 每个预取 worker 只占 1 核（否则 24 worker × 满核 OMP 线程互相抢核，
# top 退化成单核 100%，见 run.txt 第 21 行）。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

exec python scripts/train_sft.py \
  --data data/sgf_19x19_full.npz --v7 1 --device cuda --board-size 19 \
  --games-npz data/labels/games.npz \
  --batch-size 256 --epochs 1 --lr 0.00322 --weight-decay 0.0001 \
  --gradient-accumulation-steps 1 --use-amp 1 --attention-dropout 0 \
  --prefetch-workers 24 --prefetch-depth 8 \
  --log-every 25 --eval-every 500 --save-every 1674 \
  --early-stop 1 --early-stop-patience 3 \
  --out models/a_v7_a100.pth --swanlab 1 --ver a_v7_a100 --swanlab-every 10