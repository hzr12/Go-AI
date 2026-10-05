# -*- coding: utf-8 -*-
"""搜索侧的 V7（22 通道）特征装配 —— 原生 MCTS 接 V7 的唯一入口。

## 为什么需要这一层

V7 是 22 通道 ``NbtTfNet``：``(B,22,H,W)`` 空间 + ``(B,19)`` 全局**双输入**。
而原生 MCTS 这一侧的通道数历来由 ``GoBoard.feature_planes_batched`` 提供，
``go_rules.py::_check_n_channels`` 只允许 12..17 —— ch14~17 是梯子、ch18/19 要贴目，
``feature_planes`` **造不出来**。

所以 V7 上不能沿用 ``feature_planes``，必须直接调 ``feature_v7``。但直接调会
在 MCTS 里散落一堆参数（前两手盘面、贴目、规则），那是把 spec 抄进搜索侧、
迟早抄错的做法。这里把「搜索局面 → V7 输入」收成一个函数，让
``MCTS._planes1`` 只管调用。

## 为什么不把规则/贴目也塞进 board

它们是**局级标量**，不是盘面状态。盘面只回答「谁在哪」，不回答「这局按什么规则
下、贴多少目」。塞进 ``GoBoard`` 会让同一个类同时承担盘面与规则两件事，而
``feature_planes`` 那条路并不需要它们 —— 加了就是纯负担。

## 与官方口径的两处**有意的**差异（照抄 feature_v7 的口径，别在外层再改）

- ch1/ch2 是**绝对色**（黑/白），不是官方的 pla/opp 相对视角；
- ``to_play=-1`` 时因此与官方是对调关系。
  这是 ``feature_v7`` 钉死的口径（``tests/test_v7_assemble.py`` 逐项比对），
  本层原样转发。

## dtype

float16，与 ``feature_planes_batched`` 逐位一致。两路 dtype 必须相同，否则
单图推理与批量训练会拿到不同精度的 planes（``go_rules.py:2434``）。
"""
from __future__ import annotations

import numpy as np

from src.data.feature_v7 import (
    DEFAULT_RULES_FLAGS,
    GameRow,
    global_features_v7,
    spatial_channels_v7,
)

#: V7 的空间通道数（官方 ``fillRowV7``）。
V7_SPATIAL_CHANNELS = 22
#: V7 的全局输入维度（官方 ``nninputs.h``: ``NUM_FEATURES_GLOBAL_V7``）。
V7_GLOBAL_CHANNELS = 19

__all__ = [
    "V7_SPATIAL_CHANNELS",
    "V7_GLOBAL_CHANNELS",
    "BoardView",
    "is_v7_spatial",
    "v7_leaf_features",
]


class BoardView:
    """只含特征所需三项的**只读**盘面视图（盘面 / 边长 / 劫点）。

    为什么需要它：MCTS 展开子节点时手里只有 `board.board` 的 numpy 副本与
    `ko_point`（为了效率不构造完整 `GoBoard`，也免得钉住它的 undo 栈）。
    让搜索侧为此去造真 `GoBoard` 会把「特征计算」和「可下棋的棋盘」耦在一起，
    搜索侧每步造一批棋盘只为读三个字段，浪费且容易顺手改到状态。

    `v7_leaf_features` 只读这三个属性，所以传 `GoBoard` 或传本视图都行 ——
    接口按「需要什么」定，不按「是不是那个类」定。
    """

    __slots__ = ("board", "board_size", "ko_point")

    def __init__(self, board, board_size, ko_point):
        self.board = board
        self.board_size = int(board_size)
        self.ko_point = int(ko_point)


def is_v7_spatial(n_channels) -> bool:
    """这组空间通道是不是 V7 的 22 通道布局？

    判定只认**通道数**这一个量，因为 V7 的空间布局就是由 22 定的；全局 19 维
    是另一路输入，不参与这个判断。留着这个函数而不是到处写 ``== 22``，是为了
    将来若出现第二种 22 通道布局，只改这一处。
    """
    return int(n_channels) == V7_SPATIAL_CHANNELS


def v7_leaf_features(board, my_hist, op_hist, to_play, *, prev_board=None,
                     prev_prev_board=None, komi=7.5,
                     rules_flags=DEFAULT_RULES_FLAGS):
    """把一个**搜索局面**装配成 V7 的 ``(spatial, global_features)``。

    Args:
        board: :class:`GoBoard`，当前盘面（只读，不改）。
        my_hist / op_hist: **相对** ``to_play`` 的最近 3 手，index 0 是最近一手，
            ``-1`` 表示不足（pass 或开局）。这与 MCTS 既有的历史约定一致，
            也是 ``feature_v7.history_five`` 要的形状（ch9..ch13 由 3+3 手拼出）。
        to_play: ±1。
        prev_board / prev_prev_board: 前一手 / 前二手的盘面快照
            ``(n,n) int8``，``None`` 表示历史不足（开局）。
            ch15/ch16 要在**前手盘**上重算梯子，没有它们这两格只能退化成
            「整批缺失」路径。
        komi: 黑 − 白。与训练口径必须一致，否则 ch5/ch18 全错。
        rules_flags: 见 ``feature_v7`` 的位布局段。

    Returns:
        ``(spatial, global_features)`` = ``(22,19,19) float16`` / ``(19,) float16``。

    注意 ``prev_board`` 传的是**数组**而不是 ``GoBoard``：搜索侧每步都会产生
    新盘面，持有整个 board 对象会把 undo 栈一起钉住，内存无界增长。
    """
    if int(board.board_size) != 19:
        raise ValueError(
            f"V7 固定 19x19（官方 fillRowV7），收到 {board.board_size}x"
            f"{board.board_size}。ladder 通道的移植本身就是固定盘面，放行其它"
            f"尺寸只会让静默错位更难发现。")
    tp = int(to_play)
    ko = int(board.ko_point)
    b_now = np.asarray(board.board, dtype=np.int8)

    prev = None if prev_board is None else np.asarray(prev_board, dtype=np.int8)
    prev_prev = (None if prev_prev_board is None
                 else np.asarray(prev_prev_board, dtype=np.int8))

    spatial = spatial_channels_v7(
        b_now[None], [tp], [ko],
        [list(my_hist)], [list(op_hist)],
        prev_board=None if prev is None else prev[None],
        prev_prev_board=None if prev_prev is None else prev_prev[None],
        rules_flags=int(rules_flags),
    )[0]

    gl = global_features_v7(
        GameRow(komi=float(komi), rules_flags=int(rules_flags)),
        [list(my_hist)], [list(op_hist)],
        prev_board=None if prev is None else prev[None],
        prev_prev_board=None if prev_prev is None else prev_prev[None],
        rules_flags=int(rules_flags),
        to_play=[tp],
        board_area=int(board.board_size) * int(board.board_size),
    )[0]
    return spatial, gl
