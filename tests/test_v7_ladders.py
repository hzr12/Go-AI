"""``src/data/feature_v7_ladders.py`` 的测试 —— V7 空间通道 ch14/15/16/17。

分三组：

1. **合成盘面**（不依赖 stdata）：把官方的四行语义逐条钉死 —— ch14 是「当前盘
   上的梯子」、ch15/16 走**历史不足时回退复制**的门控、ch17 是「仅对手方、且
   **>1 气**」的 working-move。官方源码不可读，所以这些断言的注释里都直接写出
   对应的 ``fillRowV7`` / ``addLadderFeature`` 那一行。
2. **不变量测试**：增量落子的记账（``_Board``）与「跨块状态不外泄」两条。这两条
   都是真的踩过的坑：前者的症状是「提子恰好发生在边线时才错」，后者是
   ``_iter_ladders`` 把 ``bd.ch_libs`` 缓存成局部变量之后读到被搜索改坏的表
   （候选筛选在循环前做，所以**筛出来的块是对的、只是每块的结论错了**）。
3. **与官方 stdata 逐位对拍**（``tests/test_v7_ladders.py::test_matches_official_*``）。
   这是本移植唯一能证明「算法对」的手段，阈值的来源见那条测试的 docstring。
"""

import io
import os
import tarfile

import numpy as np
import pytest

from src.data.feature_v7_ladders import (
    BOARD_SIZE, LADDER_DTYPE, BLACK, WHITE, C_EMPTY, _Board, _pack, _put,
    assert_consistent, ladder_channels, search_is_ladder_captured,
    search_is_ladder_captured_attacker_first_2_libs,
)

# ---------------------------------------------------------------------------
# 合成盘面
# ---------------------------------------------------------------------------
# 官方只遍历「气数 ∈ {1,2}」的块（``iterLadders`` 的 ``if(libs == 1 || libs == 2)``），
# 所以「无梯子」的用例不需要构造复杂形状：任意气数 ∉ {1,2} 的块都必然 ch14 = 0。
N = BOARD_SIZE


def _board(stones, n=N):
    """``[(row, col, +1/-1)]`` → ``(n,n) int8``。"""
    b = np.zeros((n, n), np.int8)
    for r, c, v in stones:
        b[r, c] = v
    return b


def _pts(mask):
    return sorted(map(tuple, np.argwhere(np.asarray(mask))))


def _one(board, to_play=1, ko=None, **kw):
    """跑单行 :func:`ladder_channels`，返回 4 个 ``(19,19) bool``。

    ``prev_board`` / ``prev_prev_board`` 收 ``(19,19)``（本测试里都是单行）。
    """
    b = np.asarray(board, np.int8).reshape(1, N, N)
    for key in ('prev_board', 'prev_prev_board'):
        if kw.get(key) is not None:
            kw[key] = np.asarray(kw[key], np.int8).reshape(1, N, N)
    out = ladder_channels(b, np.array([to_play], np.int8),
                          ko=None if ko is None else np.array([ko], np.int16),
                          **kw)
    return [out[0, i].astype(bool) for i in range(4)]


def _pad(r, c, n=N):
    """``(r,c)`` → 带墙扁平下标（内部坐标，与 C++ 的 ``Loc`` 对齐）。"""
    return (r + 1) * (n + 2) + (c + 1)


def _unpad(stones, n=N):
    """带墙扁平下标序列 → ``(r,c)`` 序列。"""
    return [(q // (n + 2) - 1, q % (n + 2) - 1) for q in stones]


def _ref_partition(b):
    """**独立**的 4 连通块划分（纯 Python 洪水填充），返回 ``set[frozenset]``。

    ⚠ 必须是**独立实现**：``_label_chains`` 本身就用「横向游程 + 竖向 union-find」，
    拿它自己的输出去验它自己的等价物等于没验。这里只依赖「上下左右相邻且同色」
    这条定义。

    ⚠ 比较用 ``set`` 而**不是 ``sorted(...)``** —— ``frozenset.__lt__`` 的语义是
    「真子集」，**不是全序**，拿它排序会得到一个既不稳定也不传递的顺序，逐项 zip
    对比只会报出对不齐的假差异（而且下一次运行顺序还可能变）。
    """
    left = {(int(r), int(c)) for r, c in np.argwhere(np.asarray(b) != 0)}
    out = set()
    while left:
        seed = left.pop()
        comp, stack = {seed}, [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = (r + dr, c + dc)
                if p in left and b[p] == b[seed]:
                    left.discard(p)
                    comp.add(p)
                    stack.append(p)
        out.add(frozenset(_pad(r, c) for r, c in comp))
    return out


def _ref_liberties(b, chain):
    """某块的**独立**参考气集：块内每颗子四个方向的空邻点，去重。"""
    libs = set()
    for r, c in _unpad(chain):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N and b[rr, cc] == 0:
                libs.add(_pad(rr, cc))
    return frozenset(libs)


#: **1 气的梯子**。白 (9,9) 唯一的气是 (10,9)；黑 (8,9)(9,8)(9,10)(10,8)(11,8)
#: 把它的逃跑路线封死。官方算法：防守方只能下 (10,9)，下完 `moveListLen == 1`
#: 且 `upperBoundLibs <= 1` ⇒ 进攻方下一手就提掉 ⇒ 判为梯子。
#:
#: ⚠ 这个形状用来钉 ch17 的 ``getNumLiberties(loc) > 1``：**只有 1 气 ⇒ 永远不进
#: ch17**，哪怕这块正好是对手方的、哪怕上一条 2 气的链刚好留下了 working moves。
LADDER_1LIB = _board([
    (9, 9, -1),
    (8, 9, 1), (9, 8, 1), (9, 10, 1), (10, 8, 1), (11, 8, 1),
])

#: **2 气的梯子**。白 (9,9) 有 (9,10) 与 (10,9) 两口气。
#: 进攻方黑填 (9,10) ⇒ 白只剩 (10,9)，而 `boundNumLibertiesAfterPlay((10,9))`
#: 给 `lowerBound = 2 < 3`（(11,9) 与 (10,10) 两个空邻点）⇒ 不算脱逃 ⇒ 成梯，
#: 这一手就是 **working move**。
#: 反过来填 (10,9) 则是白方下 (9,10) 拿到 3 气直接跑掉 ⇒ 不是 working move。
#: ⇒ ch17 在这个形状上**只该有 (9,10) 一个点**。
LADDER_2LIB = _board([
    (9, 9, -1),
    (8, 9, 1), (9, 8, 1), (10, 8, 1),
])


# ---- 1. 没有梯子时 ch14 全 0 ------------------------------------------------
def test_empty_board_has_no_ladder():
    ch = _one(np.zeros((N, N), np.int8))
    assert not ch[0].any()
    assert not ch[1].any()
    assert not ch[2].any()
    assert not ch[3].any()


def test_plain_shape_without_ladder():
    """几块气数 ∉ {1,2} 的普通棋（官方 ``if(libs==1 || libs==2)`` 直接跳过）。

    ⚠ 别顺手加一个「边角上的孤子」：19 路角上的孤子恰好是 **2 气**，会真的进
    候选、也会真的被判成梯子（角上的 2 气块追两步就没了），那样这条测试就变成
    在测别的东西了。
    """
    b = _board([
        (3, 3, 1), (3, 4, 1), (4, 3, 1), (4, 4, 1),
        (10, 10, -1), (10, 11, -1), (11, 10, -1), (11, 11, -1),
        (8, 8, 1), (8, 9, -1),
    ])
    bd = _Board(_pack(b), N)
    assert all(len(bd.ch_libs[cid]) >= 3
               for cid, s in enumerate(bd.ch_stones) if s), \
        '前提：盘上每块的气数都 ∉ {1,2}'
    assert not _one(b)[0].any()


def test_atari_but_escapable_group_is_not_a_ladder():
    """被提子有「活路」⇒ 不是梯子。

    白 (9,9) 唯一的气 (10,9)；黑只有 (8,9)(9,8)。
    防守方下 (10,9) 之后整块有 (10,8)(10,10)(11,9) 三个新气 ⇒
    `boundNumLibertiesAfterPlay` 的 `lowerBound = 3` ⇒ 防守方直接脱逃。
    """
    b = _board([(9, 9, -1), (8, 9, 1), (9, 8, 1)])
    assert not _one(b)[0].any()


# ---- 2. 真梯子：ch14 置在正确的点上 ----------------------------------------
def test_one_liberty_ladder_marks_ch14():
    ch = _one(LADDER_1LIB)
    assert _pts(ch[0]) == [(9, 9)]


def test_two_liberty_ladder_marks_ch14():
    ch = _one(LADDER_2LIB)
    assert _pts(ch[0]) == [(9, 9)]


# ---- 3. working-move 的位置 ------------------------------------------------
def test_working_move_position_is_the_attacking_move():
    """ch17 落在「进攻方要补的那一手」上，且**只**落在那一手。

    对应官方 ``addLadderFeature``：
    ``for(j...) setRowBin(rowBin, workingPos, 17, 1.0f, ...)``，
    元素来自 ``searchIsLadderCapturedAttackerFirst2Libs`` 填的 ``workingMoves``。
    """
    ch = _one(LADDER_2LIB, to_play=1)      # to_play=黑 ⇒ opp=白 ⇒ 白是被追的一方
    assert _pts(ch[0]) == [(9, 9)]
    assert _pts(ch[3]) == [(9, 10)]


# ---- 4. `>1 气` 条件（官方 addLadderFeature 的第二个条件）------------------
def test_working_move_requires_more_than_one_liberty():
    """**1 气的梯子块只进 ch14，绝不进 ch17。**

    官方条件是 ``board.colors[loc] == opp && board.getNumLiberties(loc) > 1``，
    两个都要满足。``LADDER_1LIB`` 的白子正好是 ``to_play=黑`` 的**对手方**、
    也**确实被判为梯子**（ch14 置位）—— 唯一挡住 ch17 的就是 ``> 1``。

    ⚠ 这一条在移植里最容易漏：官方在 1 气的分支上**故意不清** ``workingMoves``
    （沿用上一条 2 气链算出来的值），真正的挡板就是这个 ``> 1``。省掉它会把残留
    值泄进 ch17 —— 实测在 stdata 上凭空多出约 4% 的错行。
    """
    ch = _one(LADDER_1LIB, to_play=1)
    assert _pts(ch[0]) == [(9, 9)], '前提：这一块确实被判成了梯子'
    assert _pts(ch[3]) == [], '1 气的块不得进 ch17（官方 libs > 1）'


def test_working_move_requires_opponent_colour():
    """同一块石头，**轮到谁**决定它是不是 ch17 的来源。

    官方条件的第一半是 ``colors[loc] == opp``、``opp = getOpp(nextPlayer)``。
    ``to_play=-1``（白走）时 opp 是黑，而梯子块是白的 ⇒ ch17 必须为空。
    """
    assert _pts(_one(LADDER_2LIB, to_play=1)[3]) == [(9, 10)]
    assert _pts(_one(LADDER_2LIB, to_play=-1)[3]) == []


def test_ch14_does_not_depend_on_to_play():
    """ch14/15/16 与 ``to_play`` **无关**（只有 ch17 依赖）。

    官方的 ``iterLadders`` 签名里根本没有 player 参数，``opp`` 只出现在
    ``addLadderFeature`` 里、且只用来判 ch17。
    """
    a = _one(LADDER_2LIB, to_play=1)
    b = _one(LADDER_2LIB, to_play=-1)
    for i in (0, 1, 2):
        assert np.array_equal(a[i], b[i])


# ---- 5. 历史门控：不足时**回退复制**，不是置 0 ------------------------------
def test_no_history_copies_current_board_into_ch15_and_ch16():
    """history = 0 ⇒ ``prevBoard = prevPrevBoard = board`` ⇒ ch15 = ch14 = ch16。

    官方：
    ``prevBoard = (hideHistory || numTurnsOfHistoryIncluded < 1) ? board : ...``
    ``prevPrevBoard = (hideHistory || numTurnsOfHistoryIncluded < 2) ? prevBoard : ...``
    """
    ch = _one(LADDER_2LIB)
    assert _pts(ch[0]) == [(9, 9)]
    assert np.array_equal(ch[1], ch[0]), 'history=0 ⇒ ch15 必须复制 ch14'
    assert np.array_equal(ch[2], ch[0]), 'history=0 ⇒ ch16 必须复制 ch15'


def test_one_turn_of_history_copies_ch15_into_ch16():
    """history = 1（只有 prev_board）⇒ ``ch16 = ch15``，而 ``ch15`` 是**真**的上一手盘。"""
    prev = LADDER_1LIB                    # 上一手盘上有另一个位置的梯子
    cur = _board([(3, 3, 1), (3, 4, 1), (4, 3, 1), (4, 4, 1)])   # 当前盘没有梯子
    assert _pts(_one(prev)[0]) == [(9, 9)]

    ch = _one(cur, prev_board=prev)
    assert _pts(ch[0]) == [], '当前盘无梯子 ⇒ ch14 全 0'
    assert _pts(ch[1]) == [(9, 9)], 'ch15 必须真的用上一手盘算'
    assert np.array_equal(ch[2], ch[1]), 'history=1 ⇒ ch16 必须复制 ch15'


def test_two_turns_of_history_keeps_all_three_independent():
    """history = 2 ⇒ ch14/15/16 各自用自己的盘面算，互不串味。"""
    cur = _board([(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
    prev = LADDER_1LIB
    prev2 = LADDER_2LIB
    ch = _one(cur, prev_board=prev, prev_prev_board=prev2)
    assert _pts(ch[0]) == []
    assert _pts(ch[1]) == [(9, 9)]
    assert _pts(ch[2]) == [(9, 9)]


# ---- 6. 三块盘面确实各自独立 ------------------------------------------------
def test_three_different_boards_give_three_different_channels():
    """三个明显不同的盘面 ⇒ 三个通道各等于自己单独算的结果，且两两不同。"""
    cur = LADDER_2LIB
    prev = LADDER_1LIB
    prev2 = np.zeros((N, N), np.int8)          # 空盘 ⇒ 必然全 0
    ch = _one(cur, prev_board=prev, prev_prev_board=prev2)

    assert np.array_equal(ch[0], _one(cur)[0])
    assert np.array_equal(ch[1], _one(prev)[0])
    assert np.array_equal(ch[2], _one(prev2)[0])
    assert _pts(ch[0]) == [(9, 9)]
    assert _pts(ch[1]) == [(9, 9)]
    assert not ch[2].any(), '空盘上不可能有梯子'
    assert not np.array_equal(ch[0], ch[2])
    assert not np.array_equal(ch[1], ch[2])


def test_same_board_for_three_channels_gives_identical_results():
    prev2 = LADDER_2LIB
    ch = _one(LADDER_2LIB, prev_board=LADDER_2LIB, prev_prev_board=prev2)
    assert np.array_equal(ch[0], ch[1]) and np.array_equal(ch[1], ch[2])


# ---- kwarg 契约 -------------------------------------------------------------
def test_output_dtype_and_layout():
    out = ladder_channels(LADDER_2LIB.reshape(1, N, N), np.array([1], np.int8))
    assert out.shape == (1, 4, N, N)
    assert out.dtype == LADDER_DTYPE
    assert LADDER_DTYPE == np.float16, '必须与 GoBoard.feature_planes 同 dtype'
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}


def test_ko_point_can_flip_the_verdict():
    """把 working move 那个点设成劫点 ⇒ 进攻方那一手非法 ⇒ 整条梯子不成立。

    官方 ``iterLadders`` 用的就是盘面自带的 ``ko_loc``；它只在
    ``searchIsLadderCapturedAttackerFirst2Libs`` 里对进攻方那两手的
    ``isLegal`` 起作用（``searchIsLadderCaptured`` 走 defenderFirst 根节点时
    会把 ko 清掉）。
    """
    assert _pts(_one(LADDER_2LIB)[0]) == [(9, 9)]
    ko_flat = 9 * N + 10
    assert not _one(LADDER_2LIB, ko=ko_flat)[0].any()


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ladder_channels(np.zeros((2, 9, 9), np.int8), np.ones(2, np.int8))
    with pytest.raises(ValueError):
        ladder_channels(np.zeros((2, N, N), np.int8), np.ones(3, np.int8))
    with pytest.raises(ValueError):
        ladder_channels(np.zeros((2, N, N), np.int8), np.ones(2, np.int8),
                        prev_board=np.zeros((2, N, N + 1), np.int8))


# ---------------------------------------------------------------------------
# 不变量：块提取与增量落子记账
# ---------------------------------------------------------------------------
def test_chain_extraction_handles_horizontal_runs_and_edges():
    """横向连排必须是**一个**块；L 形必须连成一块；边线上的子也必须正确成块。

    ⚠ 这条钉的是 ``_label_chains`` 的两条易错切片：
    1. ``np.maximum.accumulate`` 的默认 axis 是 **0**（必须先 ``ravel()`` 才能得到
       行主序的「游程起点」）；
    2. 「左邻比较」的 ``1:`` 必须落在**列**轴上（写成行轴就变成跟「正上方」比）。
    任一处写错，一条横排的 3 子会被拆成 3 个块，而症状只表现为「某个远处的气数
    算错 ⇒ 梯子判定翻转」，非常难定位。
    """
    b = _board([(17, 2, 1), (17, 3, 1), (17, 4, 1),
                (2, 3, -1), (3, 3, -1), (3, 4, -1),
                (0, 0, -1), (18, 18, 1), (0, 18, 1), (18, 0, -1)])
    bd = _Board(_pack(b), N)
    groups = {}
    for cid, stones in enumerate(bd.ch_stones):
        if stones:
            groups.setdefault(frozenset(stones), bd.ch_libs[cid])

    # ⚠ 整块划分与**独立洪水填充**逐块比对。任一处切片写错（accumulate 的默认
    # axis=0、或「左邻比较」的 `1:` 落到行轴上），一条横排的 3 子就会被拆成 3 块，
    # 这里立刻就炸 —— 而在下游它只表现为「某个远处的气数算错 ⇒ 梯子判定翻转」。
    got_partition = {frozenset(s) for s in bd.ch_stones if s}
    want_partition = _ref_partition(b)
    assert got_partition == want_partition, (
        '块划分与独立 4 连通参考不一致：'
        f'多出 {sorted(map(_unpad, got_partition - want_partition))}、'
        f'少了 {sorted(map(_unpad, want_partition - got_partition))}')
    assert len(got_partition) == 6, '10 颗子应当恰好切成 6 块（3 + 3 + 4 个孤子）'

    # ⚠ 下面每个 key 都必须是 ``frozenset``：``set`` 不可哈希，直接拿它去
    # ``dict.__contains__`` 会抛 ``TypeError: unhashable type: 'set'`` —— 那是**这条
    # 测试自己**的错，与被测代码无关。
    horizontal = frozenset(_pad(17, c) for c in (2, 3, 4))
    assert horizontal in groups, '横向三连必须是同一块'
    assert groups[horizontal] == _ref_liberties(b, horizontal)
    assert len(groups[horizontal]) == 8, '三连的 8 个气：左右各 1 + 上下各 3'

    ell = frozenset((_pad(2, 3), _pad(3, 3), _pad(3, 4)))
    assert ell in groups, 'L 形必须连成一块'
    assert groups[ell] == _ref_liberties(b, ell)
    assert len(groups[ell]) == 7, 'L 形的 7 个气'

    # 边线上的孤子：每块恰好 2 气，且只含自己。
    # ⚠ 角落 (0,0) 的气是 (1,0)/(0,1) —— **不包含墙**；墙点不是空点。
    for r, c in ((0, 0), (18, 18), (0, 18), (18, 0)):
        single = frozenset((_pad(r, c),))
        assert single in groups, f'({r},{c}) 必须自成一块'
        assert groups[single] == _ref_liberties(b, single)
        assert len(groups[single]) == 2, f'({r},{c}) 恰好 2 气（墙上不能算气）'

    assert_consistent(bd)


def test_chain_liberties_are_distinct_empty_neighbours():
    """块的气 = 与之相邻的**去重**空点（官方 ``findLiberties`` 的去重口径）。"""
    b = _board([(3, 3, 1), (3, 4, 1), (4, 3, 1),      # 3 子的 L 形，1 气
                (3, 5, 1), (5, 3, 1)])
    bd = _Board(_pack(b), N)
    arr = np.asarray(bd.colors, np.int8).reshape(N + 2, N + 2)
    for cid, stones in enumerate(bd.ch_stones):
        if not stones:
            continue
        expect = set()
        for q in stones:
            r, c = _unpad([q])[0]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N and b[rr, cc] == 0:
                    expect.add(_pad(rr, cc))
        assert bd.ch_libs[cid] == frozenset(expect)
        assert len(bd.ch_libs[cid]) == len(expect), '气必须去重'


def test_hypothetical_play_keeps_chains_and_liberties_consistent():
    """随机下 200 手（每手之后自检，再 undo 并自检）。

    增量落子的记账是「局部」的：合并友方块、4 邻对方块减一口气、提子之后给提子点
    四周的块各加一口气。任何一处漏掉，症状都是**远处**某个块的气数不对 ⇒ 梯子判定
    翻转，而且往往只在特定形状（比如提子恰好落在边线上）才发作。
    """
    rng = np.random.default_rng(20261002)
    steps = 0
    for trial in range(8):
        b = np.zeros((N, N), np.int8)
        for _ in range(120):
            bd = _Board(_pack(b), N)
            assert_consistent(bd)
            arr = np.asarray(bd.colors, dtype=np.int8)
            empt = np.flatnonzero(arr == C_EMPTY)
            empt = empt[(empt > bd.w + 1) & (empt < bd.w * (N - 1))]
            if empt.size == 0:
                break
            loc = int(rng.choice(empt))
            pla = BLACK if rng.random() < 0.5 else WHITE
            if not bd.is_legal(loc, pla):
                pla = BLACK + WHITE - pla
                if not bd.is_legal(loc, pla):
                    continue
            rec = bd.play(loc, pla)
            assert_consistent(bd)
            bd.undo(rec)
            assert_consistent(bd)
            steps += 1
            g = np.asarray(bd.colors, dtype=np.int8).reshape(bd.w, bd.w)[1:-1, 1:-1]
            b = np.where(g == BLACK, 1, np.where(g == WHITE, -1, 0)).astype(np.int8)
    assert steps > 500, f'样本太小，随机性没被覆盖到（steps={steps}）'


def test_iter_ladders_matches_solving_each_chain_on_a_fresh_board():
    """ch14 必须等于「每块各自在一块**全新**盘面上解一次」的并集。

    官方所有块共用一块 ``Board copy(board)``，靠每轮搜索自身的 play/undo 复位。
    这一点很容易在移植里破：一旦某轮搜索留下了状态，**后面**的块就会读到被改坏的
    气表。而候选筛选发生在循环之前，所以症状不是「选错了块」，而是「选对的块给了
    错的结论」—— 只在部分盘面上显形（实测 250 行里错 30 行）。

    盘面上刻意放了多块 1 气 / 2 气的候选块（两个上角 + 两个下角），
    中间是那个 2 气的真梯子。

    ⚠⚠ **「2 气的块」不是一眼定生死 —— 官方那条搜索会把逃跑路线整个走一遍。**
    实测：在只有中间那 5 颗子的空盘上，``searchIsLadderCaptured`` 一路
    「黑填 / 白长」追到 **(16,17)** 才把白提掉（34 个搜索节点）。所以诱饵里**任何一颗
    落在逃跑延长线上的子都会真的改变结论** —— 原始诱饵表里那颗 (16,17) 白子正好在
    这条线上，白方因此能连上逃脱，(9,9) 就不再是梯子，``assert got[(9,9)]`` 那句
    前提直接为假。已把 (16,17) 从诱饵里挪走（不是调阈值迁就实现：诱饵的唯一职责是
    制造**多块候选链**以暴露跨块状态泄漏，它并不参与验证结论，而 (16,17) 恰好落在
    被验对象自己的逃跑路径上，属于诱饵表写错了）。挪走后候选块仍是 12 块。
    """
    decoys = [
        (1, 1, -1), (0, 1, 1), (1, 0, 1), (1, 2, 1),
        (1, 5, 1), (0, 5, -1), (1, 4, -1), (1, 6, -1),
        (2, 4, -1), (2, 5, 1), (2, 3, 1), (2, 6, 1),
        (0, 17, -1), (0, 16, 1), (0, 18, 1), (1, 17, 1),
        # ⚠ 不要把 (16, 17) 白子加回诱饵表 —— 见 docstring：它落在 (9,9) 那条梯子
        # 的逃跑延长线上，会让 (9,9) 真的不再是梯子。
        (17, 17, 1), (17, 16, -1), (17, 18, -1),
        (17, 0, -1), (18, 0, 1), (16, 0, 1), (17, 1, 1),
    ]
    stones = [(9, 9, -1), (8, 9, 1), (9, 8, 1), (10, 8, 1)] + decoys
    b = _board(stones)

    bd = _Board(_pack(b), N)
    n_candidates = sum(1 for cid, s in enumerate(bd.ch_stones)
                       if s and len(bd.ch_libs[cid]) in (1, 2))
    assert n_candidates >= 6, '候选块太少，这测试兜不住跨块状态泄漏'

    ref = np.zeros((N, N), bool)
    for cid, stones_ in enumerate(bd.ch_stones):
        if not stones_ or len(bd.ch_libs[cid]) not in (1, 2):
            continue
        fresh = _Board(_pack(b), N)          # ← 每块一块全新盘面
        loc = stones_[0]
        libs = len(fresh.ch_libs[fresh.chain_of[loc]])
        ok = (search_is_ladder_captured(fresh, loc, True, [])
              if libs == 1
              else search_is_ladder_captured_attacker_first_2_libs(
                  fresh, loc, [], []))
        if ok:
            # ⚠ `stones_` 是**带墙**扁平下标（`(r+1)*(n+2)+(c+1)`，19 路最大到 400），
            # 而 `ref` 是 `(19,19)`＝361 格。`ref[np.asarray(stones_)] = True` 会把
            # 这些下标按**行轴**解释，越界成
            # ``IndexError: index 38 out of bounds for axis 0 with size 19`` ——
            # 那是**本测试自己**的坐标系错（内部 Loc 与紧凑 (r,c) 混用），与被测代码
            # 无关。必须先 ``_unpad`` 回 (r,c) 再按二元下标写。
            for r, c in _unpad(stones_):
                ref[r, c] = True
    got = _one(b)[0]
    assert _pts(got) == _pts(ref), (
        'ch14 与「每块独立求解」不一致 ⇒ 块与块之间的状态泄漏了')
    assert got[(9, 9)], '前提：中间那个 2 气的真梯子仍在'


def test_board_is_never_mutated_by_iter_ladders():
    """``_Board`` 接管并原地修改 ``colors``，所以 ``_iter_ladders`` 必须给它私有副本。

    若把根盘数组直接交给 ``_Board``，第一手假想落子就会把**根盘**改掉；此后整个
    扫描按根盘颜色判断 ``colors[loc] == opp``，用的是被污染的颜色 ⇒ 静默出错
    （实测 1124 行里错 91 行）。
    """
    board = LADDER_2LIB
    before = board.copy()
    _one(board)
    assert np.array_equal(board, before), '调用方传入的盘面不得被就地修改'


# ---------------------------------------------------------------------------
# 与官方 stdata 逐位对拍
# ---------------------------------------------------------------------------
#: 归档路径。⚠ **流式读**（``r|gz``），**不要 ``getmembers()``** —— 1.5 GB 的
#: 归档会被扫很久。
STDATA = os.path.join('katago', 'stdata', '2026-08-25npzs.tgz')
#: 取归档里**前** ``_N_FILES`` 个 npz 的**前** ``_ROWS_PER_FILE`` 个 19×19 行。
#: 固定取法 ⇒ 样本是确定性的，任何偏差都是真回归而不是采样噪声。
_N_FILES = 12
_ROWS_PER_FILE = 40


def _load_stdata_rows():
    """流式读 stdata，返回 ``(boards, official_ch14, official_ch17, ko)``。"""
    from src.data.katago_npz import (BOARD_STRIDE, board_size_from_packed,
                                     unpack_binary_input)
    boards, l14, l17, kos = [], [], [], []
    tf = tarfile.open(STDATA, 'r|gz')
    files = 0
    try:
        while files < _N_FILES:
            member = tf.next()
            if member is None:
                break
            if not member.name.endswith('.npz'):
                continue
            files += 1
            data = np.load(io.BytesIO(tf.extractfile(member).read()))
            if 'binaryInputNCHWPacked' not in data.files:
                continue
            packed = data['binaryInputNCHWPacked']
            keep = np.flatnonzero(
                board_size_from_packed(packed) == BOARD_STRIDE)[:_ROWS_PER_FILE]
            if keep.size == 0:
                continue
            sp = unpack_binary_input(packed)[keep]
            # ch1 = pla 子、ch2 = opp 子 ⇒ ch1 - ch2 还原 board（−1/0/+1）
            boards.append(sp[:, 1].astype(np.int8) - sp[:, 2].astype(np.int8))
            l14.append(sp[:, 14])
            l17.append(sp[:, 17])
            # ch6 = 劫禁点。官方 iterLadders 用的就是盘面自带的 ko_loc，
            # 不给它会让「2 气的块的气恰好是劫点」的行分叉。
            ko = np.full(keep.size, -1, np.int64)
            for i, mask in enumerate(sp[:, 6]):
                pts = np.argwhere(mask)
                if pts.shape[0]:
                    ko[i] = pts[0][0] * BOARD_STRIDE + pts[0][1]
            kos.append(ko)
            data.close()
    finally:
        tf.close()
    return (np.concatenate(boards), np.concatenate(l14),
            np.concatenate(l17), np.concatenate(kos))


@pytest.mark.skipif(not os.path.exists(STDATA),
                    reason=f'官方 stdata 不在 {STDATA}（跳过对拍，'
                           f'不是「通过」）')
@pytest.fixture(scope='module')
def stdata_rows():
    return _load_stdata_rows()


def test_matches_official_ch14_bit_exact(stdata_rows):
    """**ch14 与官方逐位 100% 相等。**

    阈值 **1.0** 的来源：这是对官方 `fillRowV7` 的 ch14 做的**逐位**对拍，
    样本 = ``2026-08-25npzs.tgz`` 里前 12 个 npz 的前 40 个 19×19 行
    （**实测 301** 行，逐位相等率 **1.000000**）。扩样复核（脚本
    ``tmp/coding/ladder_probe_big.py``）：

    ==================  ==================
    样本                逐位相等率
    ==================  ==================
    12 npz / 301 行     **1.000000**
    40 npz / 776 行     **1.000000**
    200 npz / 4368 行   **1.000000**
    ==================  ==================

    非零格数也逐格对上（4368 行上 mine 22528 == official 22528）。
    ch14 **不依赖 to_play** —— 两种假设下都实测 1.000000，所以这里没有假设要声明，
    它只依赖「当前盘」这一块。
    留 1.0 而不留余量，是因为这个移植的验收标准就是逐位对齐：任何一位的差异都是
    真回归（而且样本取法是固定的 ⇒ 不会引入采样噪声）。

    官方 ch1=pla / ch2=opp ⇒ ``ch1 - ch2`` 就是盘面（−1 白 / +1 黑）。
    """
    boards, official, _, kos = stdata_rows
    assert boards.shape[0] > 0
    bad = 0
    for i in range(boards.shape[0]):
        got = _one(boards[i], to_play=1, ko=int(kos[i]))[0]
        if not np.array_equal(got, official[i]):
            bad += 1
    assert bad == 0, (
        f'ch14 有 {bad}/{boards.shape[0]} 行不逐位相等 '
        f'(率 {1 - bad / boards.shape[0]:.6f})')


def test_matches_official_ch17_and_documents_to_play_assumption(stdata_rows):
    """**ch17 与官方逐位 100% 相等，且只有 ``to_play = +1`` 这个假设能做到。**

    ⚠ **这里声明用的是什么假设**：stdata 是**逐行独立样本**，官方没有直接给出
    ``nextPlayer``。ch17 的官方条件是 ``colors[loc] == opp``、
    ``opp = getOpp(nextPlayer)``，所以必须知道 ``to_play``。
    ``scripts/probe0_join.py`` 假定恒 ``+1``；本测试**沿用并实测复核**这个假定：

    - ``to_play = +1``（黑走 ⇒ opp = 白）→ 实测逐位相等率 **1.000000**
      （301 / 776 / 4368 行上全都是 1.000000，一个不差的都没有）；
    - ``to_play = -1``（白走 ⇒ opp = 黑）→ 实测逐位相等率 **0.186813**
      （200 个 npz / **4368** 行，816 行相等）。

    ⚠ **0.186813 必须连样本量一起引用** —— 它随样本量明显漂移，因为归档是按 npz
    顺序取前缀、**前几个文件不是随机样本**：

    ==================  ==========================
    样本                ``to_play=-1`` 相等率
    ==================  ==========================
    12 npz / 301 行     0.089701
    40 npz / 776 行     0.157216
    60 npz / 1290 行    0.200000
    200 npz / 4368 行   **0.186813**（收敛值）
    ==================  ==========================

    ⇒ 结论：**在这批 stdata 上 ``to_play`` 恒为 +1**，ch17 依赖 ``to_play`` 这件事
    是真的（两个假设差 5 倍以上），但 +1 这个具体取值被官方逐位证实。
    剩下的 ~81% 不是「差一点点」而是「整片错位」——``opp`` 取反后官方条件
    ``colors[loc] == opp`` 整条失效。
    阈值 1.0 的来源与 :func:`test_matches_official_ch14_bit_exact` 相同：逐位对拍
    + 固定取样，不留余量。区分度那条断言只要求 ``-1`` 明显低于 ``+1``（实测
    0.09 ~ 0.20 vs 1.00，样本量再变也差着 5 倍以上，故取 0.5 作分界）。

    附带钉住「假设是有区分度的」：若两个假设都接近 1.0，那这条测试就等于没测。
    """
    boards, _, official, kos = stdata_rows
    n = boards.shape[0]
    ok_plus = ok_minus = 0
    for i in range(n):
        ko = int(kos[i])
        if np.array_equal(_one(boards[i], to_play=1, ko=ko)[3], official[i]):
            ok_plus += 1
        if np.array_equal(_one(boards[i], to_play=-1, ko=ko)[3], official[i]):
            ok_minus += 1
    rate_plus = ok_plus / n
    assert rate_plus == 1.0, (
        f'to_play=+1 假设下 ch17 只有 {ok_plus}/{n} 逐位相等（{rate_plus:.6f}）')
    rate_minus = ok_minus / n
    assert rate_minus < 0.5, (
        f'to_play=-1 假设也有 {rate_minus:.4f} 的相等率 ⇒ 两个假设没区分度，'
        f'这条测试等于没测')


def test_official_ch14_is_sparse_and_we_reproduce_its_density(stdata_rows):
    """密度对得上：官方 ch14 非零格数 == 我们的非零格数。

    之前移植有两个「看起来能跑、实际全错」的版本，密度就是最早的信号：
    少了 ``libs > 1`` 那个挡板会多出约 4% 的 ch17，把 ``boundsAfterPlay`` 里的
    去重去掉会多出 2.6 倍的 ch14。**逐位相等已经蕴含密度相等**，这条只是把
    「量级」单独钉一遍，失败时更好读。
    """
    boards, official, official17, kos = stdata_rows
    mine14 = 0
    for i in range(boards.shape[0]):
        mine14 += int(_one(boards[i], to_play=1, ko=int(kos[i]))[0].sum())
    assert mine14 == int(official.sum()), \
        f'ch14 非零格数 {mine14} != 官方 {int(official.sum())}'