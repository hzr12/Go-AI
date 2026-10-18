#!/usr/bin/env bash
# Segment C (stdata shards, search distribution built in). Flags verbatim
# from run.txt's segment C.
#
# Uses whichever `python` is on PATH, so activate the right env first:
#   torch 2.10.0+cu128 + flash-attn 2.8.3 (python3.11) is the one where
#   flash-attn and torch.compile coexist. /opt/conda (python3.10 + torch 2.1.2
#   + flash-attn 2.3.3) makes Dynamo graph-break on FlashAttnFunc.apply.
#
# No --games-npz: the label is already in the row.
# No --soft-index: each stdata row carries policy_player_prob /
#   policy_opp_prob, which is the soft label once normalized.
# Batch and LR match A and B on purpose: 256x8 (effective 2048, LR 0.00322).
#   The 384x4 config measured mem=37.96GB of 40GB on 2026-10-11, and it would
#   break the "one effective batch across A->B->C" chain.
#
# Requires models/b_v7_a100.pth from run_b.sh to exist.
# Usage: bash run_c.sh    (run from the repo root)
set -euo pipefail

exec python scripts/train_sft.py \
  --data data/stdata --v7 1 --device cuda --board-size 19 \
  --model models/b_v7_a100.pth \
  --policy-loss soft_ce --soft-weight 1.0 \
  --batch-size 256 --epochs 1 --lr 0.00322 --weight-decay 0.0001 \
  --gradient-accumulation-steps 8 --use-amp 1 --attention-dropout 0 \
  --prefetch-workers 24 --prefetch-depth 8 \
  --log-every 50 --eval-every 500 --save-every 20 \
  --early-stop 1 --early-stop-patience 3 \
  --out models/c_v7_a100.pth --swanlab 1 --ver c_v7_a100 --swanlab-every 1