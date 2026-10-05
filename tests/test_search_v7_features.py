# -*- coding: utf-8 -*-
"""搜索侧 V7 特征装配（`src/search/v7_features.py`）的契约。

这条路径是原生 MCTS 支持 V7（22 通道）的前提。测试要锁的不是「形状对」，
而是**几个只有走对口径才会对的东西**：

  1. ch15/ch16 必须在**前手盘**上重算梯子 —— 少传 `prev_board` 这两格会退化成
     「整批缺失」，而 MCTS 侧如果不携带前手盘面就正好会犯这个错，且**不报错**
     （形状照样是 22），所以必须用「带上盘面后 ch15/ch16 确实变了」来钉；
  2. 贴目进 ch5/ch18、规则进 ch18/19 —— 贴目/规则传错不会引起任何形状错误，
     只会让模型拿到一份它没见过的输入；
  3. dtype 与 `feature_planes_batched` 一致 —— 两路精度不同会让单图推理与批量
     训练走出不同结果。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.feature_v7 import (  # noqa: E402
    DEFAULT_RULES_FLAGS, GameRow, global_features_v7, spatial_channels_v7)
from src.game.go_rules import GoBoard  # noqa: E402
from src.search.v7_features import (  # noqa: E402
    V7_GLOBAL_CHANNELS, V7_SPATIAL_CHANNELS, is_v7_spatial, v7_leaf_features)

N = 19


def _state(seq=(("b", 61), ("w", 300), ("b", 62), ("w", 299))):
    """走几步并留下每步盘面快照，返回 (board, prev, prev_prev, my, op)。"""
    b = GoBoard(N)
    snaps = []
    for color, mv in seq:
        b.play(mv)
        snaps.append(b.board.copy())
    to_play = b.current_player
    my = [61, -1, -1] if to_play == 1 else [300, -1, -1]
    op = [300, 61, -1] if to_play == 1 else [299, 62, -1]
    # 快照末位是「最后一手之后」的盘面，即当前盘；往前数两张即前一手/前二手
    return b, snaps[-2], snaps[-3], my, op


def _feats(**kw):
    b, prev, prev_prev, my, op = _state()
    args = dict(prev_board=prev, prev_prev_board=prev_prev, komi=7.5,
                rules_flags=DEFAULT_RULES_FLAGS)
    args.update(kw)
    return v7_leaf_features(b, my, op, b.current_player, **args)


#: 一个**真有梯子**的局面（随机搜索得到，见 tmp/coding/scratch/find_ladder_pos.py）。
#: 用它而不是随手摆的几子：4 子盘面上 ladder_channels 全 0，于是「带不带
#: prev_board 是否不同」恒等于「相同」，那条断言会变成一个测不出任何东西的摆设。
_LADDER_MOVES = [88, 62, 259, 342, 237, 26, 252, 53, 253, 12, 100, 0, 63, 167,
                 333, 280, 274, 261, 184, 163, 295, 194, 61, 166]


def _ladder_feats(**kw):
    b = GoBoard(N)
    snaps = []
    for mv in _LADDER_MOVES:
        assert b.play(mv), "梯子局面的重放不该失败于第 %d 手" % mv
        snaps.append(b.board.copy())
    to_play = b.current_player
    # 历史按「相对 to_play、index 0 是最近一手」构造。该局面 24 手、轮到黑(+1)，
    # 所以 tail 里偶数下标是黑的手、奇数下标是白的手。
    tail = _LADDER_MOVES[-6:]
    my = [tail[i] for i in range(5, -1, -2)]      # 5,3,1 -> 黑的三手，由近及远
    op = [tail[i] for i in range(4, -1, -2)]      # 4,2,0 -> 白的三手，由近及远
    args = dict(prev_board=snaps[-2], prev_prev_board=snaps[-3], komi=7.5,
                rules_flags=DEFAULT_RULES_FLAGS)
    args.update(kw)
    return v7_leaf_features(b, my, op, to_play, **args)


# --------------------------------------------------------------------------- #
# 形状与 dtype
# --------------------------------------------------------------------------- #
def test_shapes_are_22_spatial_and_19_global():
    sp, gl = _feats()
    assert sp.shape == (V7_SPATIAL_CHANNELS, N, N)
    assert gl.shape == (V7_GLOBAL_CHANNELS,)
    assert sp.dtype == np.float16 and gl.dtype == np.float16


def test_dtype_matches_the_legacy_feature_path():
    """两路 dtype 必须一致，否则单图推理与批量训练精度不同。"""
    b, prev, prev_prev, my, op = _state()
    sp, _ = _feats()
    legacy = GoBoard(N).feature_planes_batched(
        b.board[None], [list(my)], [list(op)], [b.current_player], [b.ko_point],
        n_channels=17)[0]
    assert legacy.dtype == sp.dtype, (
        f"feature_planes 是 {legacy.dtype}，V7 spatial 是 {sp.dtype}；"
        f"两路不一致会让单图推理与批量训练走出不同结果")


def test_opening_position_needs_no_prev_boards():
    b = GoBoard(N)
    sp, gl = v7_leaf_features(b, [-1, -1, -1], [-1, -1, -1], 1)
    assert sp.shape == (V7_SPATIAL_CHANNELS, N, N)
    assert gl.shape == (V7_GLOBAL_CHANNELS,)


def test_non_19_board_is_rejected_loudly():
    b = GoBoard(9)
    with pytest.raises(ValueError, match="19x19"):
        v7_leaf_features(b, [-1, -1, -1], [-1, -1, -1], 1)


# --------------------------------------------------------------------------- #
# 核心：前手盘面必须真的被用上
# --------------------------------------------------------------------------- #
def test_ladder_channels_on_previous_boards_actually_change_the_output():
    """带上 `prev_board` 与不帶，梯子通道必须不同。

    这是整个 plumbing 的存在理由：MCTS 侧不携带前手盘面就等于走「整批缺失」
    路径，而 `ladder_channels` 对那条路的处理是**回退到当前盘**去算 —— 形状照样
    22 通道、**不报任何错**，只是 ch15/ch16 变成了当前盘的梯子。所以必须用
    「带上盘面后 ch15/ch16 确实变了」来钉，且要在真有梯子的局面上测。
    """
    with_prev, _ = _ladder_feats()
    without, _ = _ladder_feats(prev_board=None, prev_prev_board=None)
    assert not np.array_equal(with_prev[16], without[16]), \
        ("ch16（前二手盘上的梯子）在带与不带 prev_prev_board 时完全相同 —— "
         "说明前手盘面没被用上，MCTS 侧携带两张盘面的 plumbing 就是白做的")
    # ch14 是**当前盘**上的梯子，两路必须一致：这条同时钉住「回退只影响 15/16」
    assert np.array_equal(with_prev[14], without[14]), \
        "ch14 是当前盘梯子，不该受 prev_board 影响"


def test_prev_board_alone_changes_ch15_or_ch16():
    """只给 prev_board（不給 prev_prev_board）也必须与两盘齐全时不同。"""
    both, _ = _ladder_feats()
    only_prev, _ = _ladder_feats(prev_prev_board=None)
    assert (not np.array_equal(both[15], only_prev[15])
            or not np.array_equal(both[16], only_prev[16])), \
        "缺 prev_prev_board 时 ch15/ch16 不该与两盘齐全时相同"


def test_is_v7_spatial_recognises_only_22():
    assert is_v7_spatial(22)
    assert not is_v7_spatial(17)
    assert not is_v7_spatial(12)


# --------------------------------------------------------------------------- #
# 贴目与规则
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ch", [5, 18])
def test_komi_reaches_the_global_features(ch):
    """贴目错传不会引起任何形状错误，只会喂给模型一份它没见过的输入。

    注意索引的是**全局**特征那一份（`_feats` 返回 `(spatial, global)`）——
    贴目只走 global 通道，不进 22 路空间平面。
    """
    _, ga = _feats(komi=7.5)
    _, gb = _feats(komi=0.5)
    assert ga[ch] != gb[ch], f"全局 ch{ch} 应随贴目变化（7.5 vs 0.5）"


def test_rules_flags_reach_the_area_channels():
    """规则位改变时 ch18/19（当前区域）必须跟着变。"""
    from src.data.feature_v7 import SCORING_TERRITORY, TAX_NONE
    a, _ = _feats(rules_flags=DEFAULT_RULES_FLAGS)
    b_, _ = _feats(rules_flags=DEFAULT_RULES_FLAGS | SCORING_TERRITORY | TAX_NONE)
    assert (not np.array_equal(a[18], b_[18])
            or not np.array_equal(a[19], b_[19])), \
        "计分规则（area vs territory）不同，区域通道必须不同"


# --------------------------------------------------------------------------- #
# 与 feature_v7 直调必须逐位一致（本层只是转发，不许夹带私货）
# --------------------------------------------------------------------------- #
def test_adapter_is_a_faithful_forwarder():
    b, prev, prev_prev, my, op = _state()
    sp, gl = _feats()
    ref_sp = spatial_channels_v7(
        b.board[None], [b.current_player], [b.ko_point], [my], [op],
        prev_board=prev[None], prev_prev_board=prev_prev[None],
        rules_flags=DEFAULT_RULES_FLAGS)[0]
    ref_gl = global_features_v7(
        GameRow(komi=7.5, rules_flags=DEFAULT_RULES_FLAGS), [my], [op],
        prev_board=prev[None], prev_prev_board=prev_prev[None],
        rules_flags=DEFAULT_RULES_FLAGS,
        to_play=[b.current_player], board_area=N * N)[0]
    assert np.array_equal(sp, ref_sp), "适配器与 feature_v7 直调结果不一致"
    assert np.array_equal(gl, ref_gl), "适配器与 feature_v7 直调结果不一致"


def test_adapter_does_not_mutate_the_board():
    """搜索侧每步都会造一次特征，顺手改到 board 就是隐蔽的状态污染。"""
    b, prev, prev_prev, my, op = _state()
    before = b.board.copy()
    ko_before, tp_before = b.ko_point, b.current_player
    _feats()
    assert np.array_equal(b.board, before)
    assert b.ko_point == ko_before and b.current_player == tp_before
