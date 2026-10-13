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
#   torch 2.10.0+cu128 + flash-attn 2.8.3 (python3.11) is the one where
#   flash-attn and torch.compile coexist. /opt/conda (python3.10 + torch 2.1.2
#   + flash-attn 2.3.3) makes Dynamo graph-break on FlashAttnFunc.apply.
#
# Usage: bash run.sh    (run from the repo root - all paths are relative)
set -euo pipefail

exec python scripts/train_sft.py \
  --data data/sgf_19x19_full.npz --v7 1 --device cuda --board-size 19 \
  --games-npz data/labels/games.npz \
  --batch-size 256 --epochs 1 --lr 0.00322 --weight-decay 0.0001 \
  --gradient-accumulation-steps 8 --use-amp 1 --use-ema 0 \
  --flash-attn 1 --attention-dropout 0 \
  --prefetch-workers 24 --prefetch-depth 8 \
  --log-every 25 --eval-every 500 --save-every 1674 \
  --early-stop 1 --early-stop-patience 3 \
  --out models/a_v7_a100.pth --swanlab 1 --ver a_v7_a100 --swanlab-every 10