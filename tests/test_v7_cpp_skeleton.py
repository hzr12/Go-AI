# -*- coding: utf-8 -*-
"""Task 1 test: stub fallback works, C++ extension compiles."""
import numpy as np
import pytest


def test_stub_import():
    """_feature_v7_ext can be imported without C++ built."""
    from src.data import _feature_v7_ext
    assert hasattr(_feature_v7_ext, 'spatial_channels_v7')
    # At this stage (no C++ built), _HAVE_CPP should be False
    # and spatial_channels_v7 should be the Python fallback.


def test_stub_returns_correct_shape():
    """Stub produces (B,22,19,19) float16 output."""
    from src.data.feature_v7 import spatial_channels_v7
    B = 4
    boards = np.zeros((B, 19, 19), dtype=np.int8)
    to_play = np.ones(B, dtype=np.int8)
    ko = np.full(B, -1, dtype=np.int16)
    my_hist = np.full((B, 3), -1, dtype=np.int32)
    op_hist = np.full((B, 3), -1, dtype=np.int32)
    out = spatial_channels_v7(boards, to_play, ko, my_hist, op_hist)
    assert out.shape == (B, 22, 19, 19)
    assert out.dtype == np.float16


def test_ch0_on_board_is_one():
    """ch0 = on-board mask = all ones for 19×19."""
    from src.data.feature_v7 import spatial_channels_v7
    B = 2
    boards = np.zeros((B, 19, 19), dtype=np.int8)
    to_play = np.ones(B, dtype=np.int8)
    ko = np.full(B, -1, dtype=np.int16)
    my_hist = np.full((B, 3), -1, dtype=np.int32)
    op_hist = np.full((B, 3), -1, dtype=np.int32)
    out = spatial_channels_v7(boards, to_play, ko, my_hist, op_hist)
    assert np.allclose(out[:, 0], 1.0)
