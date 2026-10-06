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

#: 推理侧 ``my_hist`` / ``op_hist`` 默认是**最老在前**（index 0 = 最老一手）。
#:
#: ⚠ 这不是笔误，是**对齐 12 通路的既有约定**。推理侧所有对局驱动（webui /
#: cli_play / evaluate / selfplay_train …）都用 ``hist.pop(0); hist.append(mv)``
#: 维护历史，也就是「最老在前、最新在末尾」；12 通道的
#: ``GoBoard.feature_planes_batched`` 正是按这个方向解释的（``planes[1+k]=my_hist[k]``），
#: 且与 12 通道训练数据（``build_dataset.split_hist`` 的 ``[::-1]``）一致。
#:
#: 但 **V7 训练口径相反**：``src/data/v7_dataset.py`` 明确换掉了槽位，使
#: ``op_hist[i,0] = moves[i-1]``（**最近**一手），因为 npz 原始槽位是反的
#: （``build_dataset.py:93`` 那句 ``[::-1]`` 把已经「新→旧」的 op 翻成了「旧→新」）。
#: ``feature_v7.history_five`` / ``_HISTORY_SLOTS`` 要的就是「最近在前」。
#:
#: ⇒ 同一个 ``my_hist`` 在 12 通道与 V7 上必须**反向解释**。本模块负责在 V7
#:   这一侧把它翻回来，否则搜索侧看到的过去 5 手是时间倒序的 —— **静默**：
#:   形状全对、不报错、着法看着正常。
#:
#: ⇒ **只在 V7 这一侧翻**。全局翻会把 12 通道弄坏（它是自洽的）。
HIST_IS_OLDEST_FIRST_DEFAULT = True

__all__ = [
    "V7_SPATIAL_CHANNELS",
    "V7_GLOBAL_CHANNELS",
    "HIST_IS_OLDEST_FIRST_DEFAULT",
    "BoardView",
    "is_v7_spatial",
    "v7_leaf_features",
    "v7_batch_features",
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
                     rules_flags=DEFAULT_RULES_FLAGS,
                     hist_is_oldest_first=None):
    """把一个**搜索局面**装配成 V7 的 ``(spatial, global_features)``。

    Args:
        board: :class:`GoBoard`，当前盘面（只读，不改）。
        my_hist / op_hist: **相对** ``to_play`` 的最近 3 手，``-1`` 表示不足
            （pass 或开局）。⚠ 默认按 ``HIST_RECENT_FIRST_DEFAULT``（**最老在前**）
            解释，见该常量的长注释：推理侧的对局驱动都是这么维护的，而 12 通道
            也正是这个方向；V7 训练口径（``v7_dataset`` 换过槽位）则是「最近在前」，
            本函数在 V7 上负责翻回来。
        to_play: ±1。
        prev_board / prev_prev_board: 前一手 / 前二手的盘面快照
            ``(n,n) int8``，``None`` 表示历史不足（开局）。
            ch15/ch16 要在**前手盘**上重算梯子，没有它们这两格只能退化成
            「整批缺失」路径。
        komi: 黑 − 白。与训练口径必须一致，否则 ch5/ch18 全错。
        rules_flags: 见 ``feature_v7`` 的位布局段。
        hist_is_oldest_first: 调用方给的 ``my_hist`` / ``op_hist`` 是否
            **最老在前**（``None`` = 用模块默认 ``True``，即仓库里所有对局驱动
            的既有约定）。只有当调用方本来就是「最近在前」（例如直接拿
            ``v7_dataset`` 重建出的训练侧数组）才传 ``False``。
            给错不会报错、只会让 ch9..13 与全局 ch0..4 静默错位 —— 所以默认走
            模块常量，而不是让每个调用点自己猜。

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

    # 统一到 feature_v7 要的「index 0 = 最近一手」
    oldest_first = (HIST_IS_OLDEST_FIRST_DEFAULT if hist_is_oldest_first is None
                    else bool(hist_is_oldest_first))
    if oldest_first:
        my_hist = list(my_hist)[::-1]
        op_hist = list(op_hist)[::-1]

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


def v7_batch_features(nodes, komi=7.5, rules_flags=DEFAULT_RULES_FLAGS,
                      hist_is_oldest_first=None):
    """把一批搜索节点装配成 V7 的批量输入 ``(spatial, global_features)``。

    与 :func:`v7_leaf_features` **同一口径**（逐项转发 ``feature_v7``），区别只在
    「一次装一批」：lookahead 的逐层批量前向需要把整层节点拼成一次前向，才吃得到
    CPU/ONNX 的大 batch 吞吐；单图版一格格装会把这条优势全部丢掉。

    Args:
        nodes: 节点序列，每个是 ``(board, my_hist, op_hist, to_play)``（4 元组）或
            携带前两手盘面的 ``(board, my_hist, op_hist, to_play, prev_board,
            prev_prev_board)``（6 元组）。后者由 ``lookahead.child_states(with_prev=True)``
            产出，用来给 ch15/ch16 的梯子重算提供前手盘；**不携带时这两格走
            「整批缺失」路径**（与训练口径不一致 ⇒ ch15/ch16 会是降级值）。
        komi / rules_flags: 与训练一致，否则 ch5/ch18 全错（见 :func:`v7_leaf_features`）。
        hist_is_oldest_first: 同 :func:`v7_leaf_features`（``None`` = 模块默认）。

    Returns:
        ``(spatial, global_features)`` = ``(B,22,19,19) float16`` / ``(B,19) float16``。
    """
    if not nodes:
        return (np.zeros((0, V7_SPATIAL_CHANNELS, 19, 19), dtype=np.float16),
                np.zeros((0, V7_GLOBAL_CHANNELS), dtype=np.float16))

    oldest_first = (HIST_IS_OLDEST_FIRST_DEFAULT if hist_is_oldest_first is None
                    else bool(hist_is_oldest_first))

    boards, my_hs, op_hs, tps, kos = [], [], [], [], []
    prev_bs, prev_pbs = [], []
    for nd in nodes:
        b = nd[0]
        boards.append(np.asarray(b.board, dtype=np.int8))
        my_hs.append(list(nd[1])[::-1] if oldest_first else list(nd[1]))
        op_hs.append(list(nd[2])[::-1] if oldest_first else list(nd[2]))
        tps.append(int(nd[3]))
        kos.append(int(b.ko_point))
        prev_bs.append(nd[4] if len(nd) > 4 else None)
        prev_pbs.append(nd[5] if len(nd) > 5 else None)

    n = len(nodes)
    size = int(nodes[0][0].board_size)
    if size != 19:
        raise ValueError(
            f"V7 固定 19x19（官方 fillRowV7），收到 {size}x{size}。")
    stacked = np.stack(boards)

    def _prev_stack(items):
        """把逐节点的 prev 盘面拼成 ``(B,19,19)``；整项皆 None 时返回 None。

        每项先规整成二维 ``(19,19)``：调用方（`child_states(with_prev=True)`、
        `MCTS._prev_board_arrays`）给的是 ``(n,n)``，但夹具/测试可能直接塞一个
        ``(1,n,n)`` —— 不规整就会 `np.stack` 出 ``(B,1,n,n)`` 然后被
        `resolve_ladder_boards` 以「形状不符」拒掉，报错信息还指不到真正的原因。
        """
        if all(p is None for p in items):
            return None
        rows = []
        for p in items:
            if p is None:
                rows.append(np.zeros((size, size), dtype=np.int8))
                continue
            a = np.asarray(p, dtype=np.int8)
            if a.ndim == 3 and a.shape[0] == 1:   # 已是批量的一批
                a = a[0]
            rows.append(a)
        return np.stack(rows)

    prev_arr = _prev_stack(prev_bs)
    prev_prev_arr = _prev_stack(prev_pbs)

    spatial = spatial_channels_v7(
        stacked, tps, kos, my_hs, op_hs,
        prev_board=prev_arr, prev_prev_board=prev_prev_arr,
        rules_flags=int(rules_flags),
        prev_valid=None if prev_arr is None else [p is not None for p in prev_bs],
        prev_prev_valid=(None if prev_prev_arr is None
                         else [p is not None for p in prev_pbs]),
    )
    gl = global_features_v7(
        GameRow(komi=float(komi), rules_flags=int(rules_flags)),
        my_hs, op_hs,
        prev_board=prev_arr, prev_prev_board=prev_prev_arr,
        rules_flags=int(rules_flags),
        to_play=tps,
        board_area=size * size,
    )
    return spatial, gl
