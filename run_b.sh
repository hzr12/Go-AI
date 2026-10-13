#!/usr/bin/env bash
# Segment B (soft labels, continues from A). Flags verbatim from run.txt.
#
# Uses whichever `python` is on PATH, so activate the right env first:
#   torch 2.10.0+cu128 + flash-attn 2.8.3 (python3.11) is the one where
#   flash-attn and torch.compile coexist. /opt/conda (python3.10 + torch 2.1.2
#   + flash-attn 2.3.3) makes Dynamo graph-break on FlashAttnFunc.apply.
#
# --epochs 30 is deliberate, not a typo: --soft-only-sampling narrows the
#   training set to ~225k rows, so 256x8 gives ~111 optimizer steps/epoch and
#   1 epoch would anneal the cosine schedule to zero around step ~100.
#
# --use-ema 0: EMA was 1 in run.txt's segment B, turned off on purpose.
#   The shadow weights were never validated against a loss/eval delta, so
#   they only add a second set of weights to keep explaining. A/B/C now all
#   train the raw weights.
#
# Requires models/a_v7_a100.pth.latest from run.sh to exist.
# Usage: bash run_b.sh    (run from the repo root)
set -euo pipefail

exec python scripts/train_sft.py \
  --data data/sgf_19x19_full.npz --v7 1 --device cuda --board-size 19 \
  --games-npz data/labels/games.npz \
  --model models/a_v7_a100.pth.latest \
  --soft-index data/labels/soft_index.npz --soft-only-sampling 1 \
  --soft-weight 1.0 --soft-every 1 --policy-loss soft_ce \
  --batch-size 256 --epochs 30 --lr 0.00322 --weight-decay 0.0001 \
  --gradient-accumulation-steps 8 --use-amp 1 --use-ema 0 \
  --flash-attn 1 --attention-dropout 0 \
  --prefetch-workers 24 --prefetch-depth 8 \
  --log-every 25 --eval-every 100 --eval-max-batches 5 --save-every 20 \
  --early-stop 1 --early-stop-patience 3 \
  --out models/b_v7_a100.pth --swanlab 1 --ver b_v7_a100 --swanlab-every 1