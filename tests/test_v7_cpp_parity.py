# -*- coding: utf-8 -*-
"""C++ vs Python parity test for spatial_channels_v7.

Runs on the subset from tmp/mat_subset_test/ (200,376 rows, stratified).
Skips if C++ extension is not available (_HAVE_CPP=False).
"""
import os
import numpy as np
import pytest

from src.data._feature_v7_ext import _HAVE_CPP
from src.data._feature_v7_ext import spatial_channels_v7 as cpp_spatial
from src.data.feature_v7 import spatial_channels_v7 as python_spatial

# Paths
MAT_DIR = os.path.join(os.path.dirname(__file__), '..', 'tmp', 'mat_subset_test')
BOARDS = os.path.join(MAT_DIR, 'boards.npy')
TO_PLAY = os.path.join(MAT_DIR, 'to_play.npy')
KO = os.path.join(MAT_DIR, 'ko.npy')
MY_HIST = os.path.join(MAT_DIR, 'my_hist.npy')
OP_HIST = os.path.join(MAT_DIR, 'op_hist.npy')


@pytest.mark.skipif(not _HAVE_CPP, reason="C++ extension not available")
@pytest.mark.parametrize('n_rows', [100, 500, 1000])
def test_parity_all_channels(n_rows):
    """C++ and Python produce identical output for all 22 channels."""
    boards = np.load(BOARDS)[:n_rows].astype(np.int32)
    tp = np.load(TO_PLAY)[:n_rows].astype(np.int32)
    ko = np.load(KO)[:n_rows].astype(np.int32)
    my = np.load(MY_HIST)[:n_rows].astype(np.int32)
    op = np.load(OP_HIST)[:n_rows].astype(np.int32)
    B = boards.shape[0]
    # Pass zeros for prev boards (no history) — matches Python default
    prev_board = np.zeros_like(boards)
    prev_prev_board = np.zeros_like(boards)
    prev_valid = np.ones(B, dtype=np.int32)
    prev_prev_valid = np.ones(B, dtype=np.int32)

    out_cpp = cpp_spatial(boards, tp, ko, my, op,
                          prev_board, prev_prev_board,
                          prev_valid, prev_prev_valid, 0)
    # Python call with same zero prev boards for fair comparison
    out_py = python_spatial(boards, tp, ko, my, op,
                            prev_board=prev_board, prev_prev_board=prev_prev_board)

    assert out_cpp.shape == out_py.shape == (B, 22, 19, 19)
    # C++ returns float32, Python returns float16 — compare as float32
    assert out_cpp.dtype == np.float32

    diff = np.abs(out_cpp.astype(np.float32) - out_py.astype(np.float32))
    max_diff = diff.max()
    if max_diff > 1e-5:
        for ch in range(22):
            ch_diff = diff[:, ch].max()
            if ch_diff > 1e-5:
                pytest.fail(
                    f'Max diff {max_diff:.6f} at ch{ch} (ch max={ch_diff:.6f})'
                )
    assert max_diff < 1e-5, f'Output differs: max={max_diff}'


@pytest.mark.skipif(not _HAVE_CPP, reason="C++ extension not available")
def test_ch0_to_play_invariant():
    """ch0 (on-board) does not depend on to_play."""
    boards = np.load(BOARDS)[:100].astype(np.int32)
    tp = np.load(TO_PLAY)[:100].astype(np.int32)
    ko = np.load(KO)[:100].astype(np.int32)
    my = np.load(MY_HIST)[:100].astype(np.int32)
    op = np.load(OP_HIST)[:100].astype(np.int32)
    B = boards.shape[0]
    prev_board = np.zeros_like(boards)
    prev_prev_board = np.zeros_like(boards)
    prev_valid = np.zeros(B, dtype=np.int32)
    prev_prev_valid = np.zeros(B, dtype=np.int32)

    out1 = cpp_spatial(boards, tp, ko, my, op,
                       prev_board, prev_prev_board,
                       prev_valid, prev_prev_valid, 0)
    out2 = cpp_spatial(boards, -tp, ko, my, op,
                       prev_board, prev_prev_board,
                       prev_valid, prev_prev_valid, 0)
    assert np.allclose(out1[:, 0], out2[:, 0])


@pytest.mark.skipif(not _HAVE_CPP, reason="C++ extension not available")
def test_ch1_ch2_color_invariant():
    """ch1=black, ch2=white — swap to_play, colors stay absolute."""
    boards = np.load(BOARDS)[:100].astype(np.int32)
    tp = np.load(TO_PLAY)[:100].astype(np.int32)
    ko = np.load(KO)[:100].astype(np.int32)
    my = np.load(MY_HIST)[:100].astype(np.int32)
    op = np.load(OP_HIST)[:100].astype(np.int32)
    B = boards.shape[0]
    prev_board = np.zeros_like(boards)
    prev_prev_board = np.zeros_like(boards)
    prev_valid = np.zeros(B, dtype=np.int32)
    prev_prev_valid = np.zeros(B, dtype=np.int32)

    out1 = cpp_spatial(boards, tp, ko, my, op,
                       prev_board, prev_prev_board,
                       prev_valid, prev_prev_valid, 0)
    out2 = cpp_spatial(boards, -tp, ko, my, op,
                       prev_board, prev_prev_board,
                       prev_valid, prev_prev_valid, 0)
    assert np.allclose(out1[:, 1], out2[:, 1])
    assert np.allclose(out1[:, 2], out2[:, 2])
