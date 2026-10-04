# -*- coding: utf-8 -*-
"""B1 · V7 的 22 通道装配 + 与官方 stdata 的逐通道对拍。

被测对象 `src/data/feature_v7.py`：
  · `spatial_channels_v7`     —— 把各通道拼成 `(B,22,19,19)` float16
  · `resolve_ladder_boards`   —— 历史门控的**唯一**解析点

运行：
    python -m pytest tests/test_v7_assemble.py -q -rs -s

----
**关于「逐通道命中数字」的口径**：报的是**逐行精确相等率**
（fraction of rows where the whole 19×19 plane matches），不是「逐格一致率」。
后者是个陷阱：ch9 的逐格一致率是 **0.997**（每行只差 1 个点，共 361 个点），
而逐行精确率是 **0.000** —— 用前者汇报会让一个彻底错的通道看起来几乎完美。
两种都印出来，就是为了不让这个陷阱有藏身之处。
"""

import io
import os
import tarfile

import numpy as np
import pytest

from src.data.feature_v7 import (
    DEFAULT_RULES_FLAGS,
    KOMI_SCALE,
    SCORING_AREA,
    SCORING_TERRITORY,
    SPATIAL_CHANNELS,
    SPATIAL_DTYPE,
    TAX_ALL,
    TAX_NONE,
    TAX_SEKI,
    KO_POSITIONAL,
    KO_SIMPLE,
    KO_SITUATIONAL,
    GameRow,
    calculate_area,
    global_features_v7,
    history_five,
    komi_parity_wave,
    liberties_123,
    resolve_ladder_boards,
    rules_flags_from,
    self_komi,
    spatial_channels_v7,
)
from src.data.feature_v7_ladders import ladder_channels
from src.data.katago_npz import board_size_from_packed, unpack_binary_input

N = 19
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STDATA = os.path.join('katago', 'stdata', '2026-08-25npzs.tgz')
needs_stdata = pytest.mark.skipif(
    not os.path.isfile(STDATA),
    reason=f'需要官方 stdata（{STDATA}，不在仓库里）。'
           f' 跳过不等于「通过」—— 本仓唯一可执行 oracle 就是它。')

#: 取归档里**前** ``_N_FILES`` 个 npz 的**前** ``_ROWS_PER_FILE`` 个 19×19 行。
#: 固定取法 ⇒ 样本确定，任何偏差都是真回归而不是采样噪声（与 test_v7_ladders 同构）。
_N_FILES = 12
_ROWS_PER_FILE = 40

#: 官方 stdata 上 ch1 = pla、ch2 = opp，而本仓 ch1 = 黑 / ch2 = 白（绝对色）。
#: `to_play = +1` 时 pla == 黑 ⇒ 两者逐位相同（实测见下方对拍表）。
_OFFICIAL_TO_PLAY = 1


# --------------------------------------------------------------------------- #
# 合成夹具：每个通道都有一个**手算得出**的标记点
# --------------------------------------------------------------------------- #
def make_board(stones):
    """``{(r, c): ±1}`` → ``(1,19,19)`` int8。"""
    b = np.zeros((1, N, N), dtype=np.int8)
    for (r, c), v in stones.items():
        b[0, r, c] = v
    return b


#: 布局说明。三个气桶标记点**刻意放在互不相邻的三处**（行 9 / 行 5 / 行 15），
#: 因为气数是**整块**口径：两颗黑子一旦相邻就会并成一块、两边的气数一起变，
#: 于是「(9,9) 是 3 气」这种断言会在相邻的那次改动里无声失效。
#:   (9,9)  三个白子（左/上/下/右里占 3 个）⇒ **1 气** ⇒ ch3 标记
#:   (5,2)  上下两个白子                ⇒ **2 气** ⇒ ch4 标记
#:   (15,15) 右边一个白子               ⇒ **3 气** ⇒ ch5 标记
LIB_MARKS = {(9, 9): 1, (5, 2): 2, (15, 15): 3}

#: 黑子（ch1）与白子（ch2）的标记点 —— 就是上面那三个气桶的黑子加它们的围子。
BLACK_STONES = [(9, 9), (5, 2), (15, 15),
                (0, 17), (2, 17), (1, 16), (1, 18)]
WHITE_STONES = [(8, 9), (10, 9), (9, 10),
                (4, 2), (6, 2),
                (15, 16),
                (16, 17), (18, 17), (17, 16), (17, 18)]

#: 区域归属的三个标记点（ch18 / ch19）：
#:   (1,17) 四邻全是黑子 ⇒ 归黑； (17,17) 四邻全是白子 ⇒ 归白；
#:   (0,0) 在「哪边都不邻」的大空区里 ⇒ 中立。
#: **旧注释的理由（"只邻黑子 ⇒ 归黑"，Tromp-Taylor 的区域/边界色判定）已作废**。
#:   官方 area 是 Benson + 围空：白先被黑用 `:2221` 写成 +1，再被白用 `:2221`
#:   无条件覆盖回 −1（`board.cpp:2221`/`:1862`，黑先白后）；(1,17) 反过来 —— 白那
#:   一遍没能覆盖（该区域 `bordersNonPassAlive`），于是留下 +1。**取值没变，路径
#:   整个换了**，所以别再拿「只邻一种颜色」当依据。
AREA_BLACK = (1, 17)
AREA_WHITE = (17, 17)
AREA_NEUTRAL = (0, 0)

#: ko 点标记（必须落在空点上）。
KO_PT = (2, 2)
KO_FLAT = KO_PT[0] * N + KO_PT[1]

#: 过去 5 手的落点标记（ch9..13），物理手序、互不相同、都在大空区里。
HIST_POINTS = [(16, 1), (16, 3), (16, 5), (16, 7), (16, 9)]


def _synthetic():
    """造一份让每个通道都有可辨认标记的输入。返回 ``(boards, my, op, ko)``。"""
    stones = {}
    for r, c in BLACK_STONES:
        stones[(r, c)] = 1
    for r, c in WHITE_STONES:
        stones[(r, c)] = -1
    boards = make_board(stones)

    # 物理手序 5 手 → 交错成 (my, op)；index 0 是最近一手。
    phys = [r * N + c for r, c in HIST_POINTS]
    my = np.array([[-1, -1, -1]], dtype=np.int16)
    op = np.array([[-1, -1, -1]], dtype=np.int16)
    for k, mv in enumerate(phys):
        (op if k % 2 == 0 else my)[0, k // 2] = mv
    return boards, my, op, np.array([KO_FLAT], dtype=np.int16)


BOARDS, MY_HIST, OP_HIST, KO = _synthetic()
TO_PLAY = np.array([1], dtype=np.int8)


def assemble(**kw):
    kw.setdefault('prev_board', None)
    kw.setdefault('prev_prev_board', None)
    kw.setdefault('rules_flags', DEFAULT_RULES_FLAGS)
    return spatial_channels_v7(BOARDS, TO_PLAY, KO, MY_HIST, OP_HIST, **kw)


def pt(ch, mask):
    """某个通道在给定坐标上是否置位（返回 bool 列表，形状 ``(1,)``）。"""
    return [bool(mask[0, ch][r, c]) for r, c in _markers()]


def _markers():
    return ([(r, c) for r, c in LIB_MARKS] + [AREA_BLACK, AREA_WHITE,
                                              AREA_NEUTRAL, KO_PT]
            + HIST_POINTS)


# --------------------------------------------------------------------------- #
# 通道 0 / 1 / 2 / 6 / 7 / 8 / 20 / 21 —— 语义手算，不依赖任何参考实现
# --------------------------------------------------------------------------- #
def test_ch0_is_on_board_not_the_black_stones():
    """ **ch0 = on-board**（全 1），**不是**任务书原文的「黑」。

    官方 ``binaryInputNCHWPacked`` 的 ch0 是 on-board 掩码，本仓
    ``katago_npz.board_size_from_packed`` 靠 ch0 的 1 的个数反推棋盘边长
    （spec §9.1 第 2 条）。ch0 一旦写成「黑子」，那条反推就静默失效 ——
    同一个 npz 里混着 9/11/13/18 路与非方阵，喂 19 路重建盘面会错位。
    """
    out = assemble()
    assert out[0, 0].all(), '19×19 无填充 ⇒ ch0 必须整格置位'
    assert int(out[0, 0].sum()) == N * N


def test_ch1_is_black_and_ch2_is_white_in_absolute_terms():
    """ch1 = **黑**（+1）、ch2 = **白**（−1），**绝对色**、与 to_play 无关。"""
    for tp in (1, -1):
        out = spatial_channels_v7(BOARDS, np.array([tp], np.int8), KO,
                                  MY_HIST, OP_HIST)
        for r, c in BLACK_STONES:
            assert out[0, 1, r, c] == 1.0 and out[0, 2, r, c] == 0.0
        for r, c in WHITE_STONES:
            assert out[0, 2, r, c] == 1.0 and out[0, 1, r, c] == 0.0


def test_ch1_ch2_do_not_depend_on_to_play_unlike_official_pla_opp():
    """本仓 ch1/ch2 是绝对色 ⇒ **换 to_play 通道不动**。

    官方 ch1/ch2 是相对 ``nextPlayer`` 的（to_play=−1 时会**对调**）。改成绝对色
    是有意的口径（见 `spatial_channels_v7` 的 docstring）：好处是 ch0..5 与
    ch9..13 整块与 to_play 无关，8 路 dihedral 增广下不需要任何视角修正。
    代价是与官方在 ``to_play = −1`` 时不是同一个东西 —— 这条测试把「代价」与
    「好处」同时钉住，不留给下游去猜。
    """
    a = spatial_channels_v7(BOARDS, np.array([1], np.int8), KO, MY_HIST, OP_HIST)
    b = spatial_channels_v7(BOARDS, np.array([-1], np.int8), KO, MY_HIST, OP_HIST)
    for ch in (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13):
        assert np.array_equal(a[:, ch], b[:, ch]), f'ch{ch} 不该随 to_play 变'
    # 但 ch18/19 **是** pla/opp 相对视角（官方口径）⇒ 必须变
    assert not np.array_equal(a[:, 18], b[:, 18]) or \
        not np.array_equal(a[:, 19], b[:, 19])


def test_ch6_is_the_ko_ban_point():
    """ch6 = simple-ko 点（encore=0 ⇒ 只有 ``ko_loc``，不跟踪 superko 集合）。

    官方 ch6 是 ``ko_loc ∪ (superKoBanned \\ {ko_loc})``；KO_SIMPLE 下 superko
    集合为空 ⇒ 两者等价（spec §2.6 D1）。
    """
    out = assemble()
    assert int(out[0, 6].sum()) == 1
    assert out[0, 6, KO_PT[0], KO_PT[1]] == 1.0
    # 无劫 ⇒ 整格 0
    assert not spatial_channels_v7(BOARDS, TO_PLAY, np.array([-1], np.int16),
                                  MY_HIST, OP_HIST)[0, 6].any()


def test_ch7_ch8_ch20_ch21_are_always_zero_and_kept_in_the_tensor():
    """**8 路恒 0** 里的 4 路空间通道保留不裁剪（spec §2.4）。

    保留理由：严格复刻官方张量布局，接官方 checkpoint 时零改形状；裁剪只省
    0.3% 参数。**不裁剪**这条本身就是契约 —— 有人为了省参数删掉它们，模型就会
    在接官方权重时静默错位（conv 的输入通道顺序对不上）。
    """
    out = assemble()
    assert SPATIAL_CHANNELS == 22
    for ch in (7, 8, 20, 21):
        assert not out[0, ch].any(), f'ch{ch} 必须是恒 0 通道'


# --------------------------------------------------------------------------- #
# 通道 3/4/5 · 9-13 · 14-17 · 18/19 —— 与**各自独立**的参考实现对拍
# --------------------------------------------------------------------------- #
def test_ch3_ch4_ch5_are_the_colour_blind_1_2_3_liberty_buckets():
    """三个标记点各落在**恰好**对应的一格上 —— 桶要是写成 ``>=k`` 就会串。

    同时钉住「不分色」：白子的气桶与黑子的在同一组三格里（17 通道的
    ch10/11/14/15 才是分色的）。
    """
    out = assemble()
    ref = liberties_123(BOARDS)[0]
    for k in range(3):
        assert np.array_equal(out[0, 3 + k].astype(bool), ref[k])
    marks = {1: (9, 9), 2: (5, 2), 3: (15, 15)}
    for libs, (r, c) in marks.items():
        for ch in (3, 4, 5):
            assert out[0, ch, r, c] == (1.0 if ch == 2 + libs else 0.0), (
                f'({r},{c}) 是 {libs} 气 ⇒ 只应在 ch{2 + libs} 置位')


def test_ch9_ch13_are_the_five_recent_moves_in_order():
    """ch9..13 = ``opp, pla, opp, pla, opp`` 的物理手序（最近一手在 ch9）。

    **逐格断言标记点**，而不是只比 `history_five` 的输出：后者在「两个函数同时
    错」时仍会通过，而通道顺序错位是静默的。
    """
    out = assemble()
    ref = history_five(BOARDS, MY_HIST, OP_HIST)[0]
    for ch in range(9, 14):
        assert np.array_equal(out[0, ch].astype(bool), ref[ch - 9])
    for k, (r, c) in enumerate(HIST_POINTS):
        for ch in range(9, 14):
            assert out[0, ch, r, c] == (1.0 if ch == 9 + k else 0.0), (
                f'落点 ({r},{c}) 是第 {k + 1} 手 ⇒ 只应在 ch{9 + k} 置位')


def test_ch14_ch17_equal_ladder_channels_and_ch15_ch16_follow_the_gating():
    """ch14..17 = `ladder_channels` 的四格（算法本身由 `test_v7_ladders.py` 担保）。

    这里钉的是**装配位置**：ch14=梯子 / ch15=前一手 / ch16=前二手 / ch17=working。
    """
    out = assemble()
    ref = ladder_channels(BOARDS, TO_PLAY, ko=KO)[0]
    for k, ch in enumerate((14, 15, 16, 17)):
        assert np.array_equal(out[0, ch].astype(bool), ref[k]), f'ch{ch} 装错了'


def test_ch18_ch19_are_the_two_boolean_area_planes():
    """ch18 = 归 pla 的点、ch19 = 归 opp 的点（**两条布尔平面**，官方口径）。

     `calculate_area` 返回**带符号**的两视角（+1/−1/0）；装配层必须二值化 ——
      否则 −1 会让「有归属的点」与「on-board 的点」在张量上混同，而且无法与
      `unpack_binary_input` 的解包结果对拍。
     **下面三个取值与旧版一模一样，但旧版的理由是错的**（见 `AREA_BLACK` 上方
      的注释：旧版按 Tromp-Taylor 的「只邻一种颜色」解释）。按官方算法重算，
      `(1,17)→ch18`、`(17,17)→ch19`、`(0,0)→中立` 仍然成立，走的却是另一条路
      （Benson + `:2221`/`:2222` 围空 + 黑先白后的覆盖）。所以这条现在钉的是
      **装配层的二值化与通道顺序**，归属语义由 `tests/test_v7_area.py` 与
      `tests/test_v7_area_official.py` 担保。
    """
    out = assemble()
    assert set(np.unique(out[:, 18:20])).issubset({0.0, 1.0}), \
        'ch18/19 必须二值化（否则装进去的是 −1）'
    assert out[0, 18, *AREA_BLACK] == 1.0 and out[0, 19, *AREA_BLACK] == 0.0
    assert out[0, 19, *AREA_WHITE] == 1.0 and out[0, 18, *AREA_WHITE] == 0.0
    assert out[0, 18, *AREA_NEUTRAL] == 0.0 and out[0, 19, *AREA_NEUTRAL] == 0.0
    ref = calculate_area(BOARDS, TO_PLAY, rules_flags=DEFAULT_RULES_FLAGS)[0]
    assert np.array_equal(out[0, 18].astype(bool), ref[0] > 0)
    assert np.array_equal(out[0, 19].astype(bool), ref[1] > 0)


def test_territory_scoring_zeroes_ch18_ch19_and_is_flagged_in_global_ch9():
    """规则条件化落到张量上（spec §2.5）：TERRITORY ⇒ ch18/19 恒 0。

     **这是对齐官方，不是缺陷**：官方的 territory 分支被 ``encorePhase >= 2``
      二次条件包着，而本仓无 encore 机制（spec §2.6 D2）。
    """
    terr = rules_flags_from(scoring=SCORING_TERRITORY)
    out = spatial_channels_v7(BOARDS, TO_PLAY, KO, MY_HIST, OP_HIST,
                              rules_flags=terr)
    assert not out[0, 18].any() and not out[0, 19].any()
    # 其余 20 格一个都不该被规则位影响
    area = assemble()
    for ch in range(20):
        if ch in (18, 19):
            continue
        assert np.array_equal(area[:, ch], out[:, ch]), f'规则位不该动 ch{ch}'


# --------------------------------------------------------------------------- #
# 历史门控：只解析一次，只调一处
# --------------------------------------------------------------------------- #
def test_no_history_copies_ch14_into_ch15_and_ch15_into_ch16():
    """官方原文：``prevBoard = board; prevPrevBoard = prevBoard``。

    ⇒ history=0 时 **ch15 == ch14 且 ch16 == ch15**。
    """
    out = assemble()
    assert np.array_equal(out[0, 14], out[0, 15])
    assert np.array_equal(out[0, 15], out[0, 16])


def test_one_turn_of_history_copies_ch15_into_ch16():
    """history=1 ⇒ **ch16 == ch15**（不是 ch14！）。

     这是这条门控唯一的坑，也是 ``feature_v7_ladders.py`` 里「别把门控从历史
      参数再推一遍」那条注释的由来 —— 那次是从同一个事实**推导两次**、第二次
      抄成了 ch14，于是 history=1 时 ch16 差了整整一手。本测试用一个
      「ch14 ≠ ch15」的前一手盘把这个坑钉死。
    """
    prev = BOARDS.copy()
    prev[0, 0, 0] = 1 if BOARDS[0, 0, 0] != 1 else -1     # 造一个不同的盘面
    out = assemble(prev_board=prev)
    assert np.array_equal(out[0, 15], out[0, 16]), 'history=1 ⇒ ch16 必须抄 ch15'
    assert not np.array_equal(out[0, 14], out[0, 15]), (
        '夹具前提：这个前一手盘上的梯子必须与当前盘不同，否则这条测不出东西')


def test_resolve_ladder_boards_is_the_single_gating_point():
    """``resolve_ladder_boards`` 逐行落官方门控；**整批缺失时给「该传 None」的标志**。

    「传 None」不是纯优化：`ladder_channels` 靠**数组身份**（``pb is b`` /
    ``p2b is pb``）判定回退，传一份 ``boards.copy()`` 会让它白跑一整轮 DFS。
    """
    b = BOARDS
    # 整批都没有历史
    g = resolve_ladder_boards(b)
    assert g.prev_all_missing and g.prev_prev_all_missing
    # 整批都有
    g = resolve_ladder_boards(b, b.copy(), b.copy())
    assert not g.prev_all_missing and not g.prev_prev_all_missing
    # 逐行混合：第 0 行缺前一手
    prev = b.copy()
    g = resolve_ladder_boards(b, prev, prev,
                              prev_valid=np.array([False]),
                              prev_prev_valid=np.array([True]))
    assert not g.prev_all_missing
    assert np.array_equal(g.prev[0], b[0]), '缺前一手的那行必须回退成当前盘'


def test_second_level_fallback_uses_the_resolved_prev_not_the_current_board():
    """ 第二级回退的兜底是**已解析的 prev**，不是 ``board``。

    history=1 时 ``prev`` 是真正的上一手盘（≠ 当前盘）；回退到 ``board`` 会让
    ch16 差一手。这条与 :func:`test_one_turn_of_history_copies_ch15_into_ch16`
    是同一件事的两端（那边看装配结果、这边看解析结果）。
    """
    b = BOARDS
    prev = b.copy()
    prev[0, 0, 0] = 1 if b[0, 0, 0] != 1 else -1
    g = resolve_ladder_boards(b, prev, prev,
                              prev_valid=np.array([True]),
                              prev_prev_valid=np.array([False]))
    assert np.array_equal(g.prev_prev[0], g.prev[0])
    assert not np.array_equal(g.prev_prev[0], b[0]), (
        'prev ≠ board 时这条才抓得到；相等的话测试是空转的')


def test_mixed_history_in_one_batch_matches_per_row_single_row_assembly():
    """**逐行混合的历史门控**：把一批混着「有历史/没历史」的行装配出来的结果，
    必须与「每行单独装配」的结果逐位相同。

    这条是 gather（逐行 valid 掩码）与装配层的**接缝**：预取器取到的批天然是
    混合的（随机抽的行，开局/终局附近的行缺历史），而 ``ladder_channels`` 的
    回退判定是**整批**的 ⇒ 逐行混合只能由装配层解决。
    """
    rng = np.random.default_rng(11)
    rows = []
    for i in range(6):
        rows.append(rng.choice([-1, 0, 1], size=(N, N)).astype(np.int8))
    boards = np.stack(rows)
    prev = np.stack([np.roll(r, 1, axis=1) for r in rows])
    prev2 = np.stack([np.roll(r, 2, axis=1) for r in rows])
    tp = np.array([1, -1, 1, -1, 1, -1], np.int8)
    ko = np.array([-1, 5, -1, 7, -1, 9], np.int16)
    my = np.full((6, 3), -1, np.int16)
    op = np.full((6, 3), -1, np.int16)
    v1 = np.array([True, True, False, False, True, False])
    v2 = np.array([True, False, False, True, True, False])

    batch = spatial_channels_v7(boards, tp, ko, my, op, prev, prev2,
                                prev_valid=v1, prev_prev_valid=v2)
    for i in range(6):
        one = spatial_channels_v7(
            boards[i:i + 1], tp[i:i + 1], ko[i:i + 1], my[i:i + 1], op[i:i + 1],
            prev[i:i + 1] if v1[i] else None,
            prev2[i:i + 1] if v2[i] else None,
            prev_valid=np.array([True]) if v1[i] else None,
            prev_prev_valid=np.array([True]) if v2[i] else None)
        assert np.array_equal(batch[i:i + 1], one), f'第 {i} 行的混合门控跑偏了'


# --------------------------------------------------------------------------- #
# 形状 / dtype / 守卫
# --------------------------------------------------------------------------- #
def test_output_shape_and_dtype_match_the_existing_feature_planes():
    """fp16，与 ``GoBoard.feature_planes`` / ``feature_planes_batched`` **同 dtype**。

    `go_rules.py:2434` 的原话是「两路必须一致，否则 RL/推理走单图与训练走批量
    会拿到不同精度的 planes」。取值域只有 {0,1} ⇒ fp16 逐位无损。
    """
    out = assemble()
    assert out.shape == (1, SPATIAL_CHANNELS, N, N)
    assert out.dtype == SPATIAL_DTYPE == np.float16
    assert set(np.unique(out)) <= {0.0, 1.0}


def test_rejects_non_19_boards_and_bad_to_play():
    with pytest.raises(ValueError, match='19'):
        spatial_channels_v7(np.zeros((1, 13, 13), np.int8), TO_PLAY, KO,
                            MY_HIST, OP_HIST)
    with pytest.raises(ValueError, match=r'±1'):
        spatial_channels_v7(BOARDS, np.array([0], np.int8), KO, MY_HIST, OP_HIST)
    with pytest.raises(ValueError, match='to_play'):
        spatial_channels_v7(BOARDS, np.array([1, 1], np.int8), KO, MY_HIST, OP_HIST)


def test_ko_out_of_range_is_skipped_rather_than_raising():
    """脏 ``ko`` 值静默跳过（只丢一个点），不炸掉整个预取 worker。"""
    out = spatial_channels_v7(BOARDS, TO_PLAY, np.array([400], np.int16),
                              MY_HIST, OP_HIST)
    assert not out[0, 6].any()
    assert not spatial_channels_v7(BOARDS, TO_PLAY, np.array([-2], np.int16),
                                  MY_HIST, OP_HIST)[0, 6].any()


# --------------------------------------------------------------------------- #
# 与官方 stdata 的逐通道对拍
# --------------------------------------------------------------------------- #
def _load_stdata():
    """流式取前 ``_N_FILES`` 个 npz 的前 ``_ROWS_PER_FILE`` 个 **19×19** 行。

     **流式读**（``r|gz``），**不要 ``getmembers()``** —— 1.5 GB 的归档会被扫很久
      （与 ``tests/test_v7_ladders.py`` 同一条纪律）。
     19×19 由 ch0（on-board 掩码）的 1 的个数反推（spec §9.1 第 2 条）。
    """
    S, G, KO = [], [], []
    tf = tarfile.open(STDATA, 'r|gz')
    files = 0
    try:
        while files < _N_FILES:
            m = tf.next()
            if m is None:
                break
            if not m.name.endswith('.npz'):
                continue
            files += 1
            d = np.load(io.BytesIO(tf.extractfile(m).read()))
            if 'binaryInputNCHWPacked' not in d.files:
                continue
            packed = d['binaryInputNCHWPacked']
            keep = np.flatnonzero(board_size_from_packed(packed) == N) \
                [:_ROWS_PER_FILE]
            if keep.size == 0:
                continue
            sp = unpack_binary_input(packed)[keep]
            S.append(sp)
            if 'globalInputNC' in d.files:
                G.append(d['globalInputNC'][keep])
            # 官方 iterLadders 用的就是盘面自带的 ko_loc（与 test_v7_ladders 同一取法）
            ko = np.full(keep.size, -1, np.int64)
            for i, mask in enumerate(sp[:, 6]):
                pts = np.argwhere(mask)
                if pts.shape[0]:
                    ko[i] = pts[0][0] * N + pts[0][1]
            KO.append(ko)
            d.close()
    finally:
        tf.close()
    return (np.concatenate(S), np.concatenate(G) if G else None,
            np.concatenate(KO))


@needs_stdata
@pytest.fixture(scope='module')
def stdata():
    S, G, KO = _load_stdata()
    assert S.shape[0] > 0
    return S, G, KO


def _assemble_stdata(S, KO):
    """按官方口径装配（``to_play = +1`` ⇒ pla == 黑，ch1/ch2 与官方同解）。

    ``my_hist`` / ``op_hist`` 给全 −1：**stdata 是逐行独立样本、没有着法序列**
    ⇒ ch9..13 本来就不可对拍（spec §9.4）。同理没有「前一手/前二手盘」⇒
    ch15/ch16 走官方的回退复制分支。
    """
    boards = S[:, 1].astype(np.int8) - S[:, 2].astype(np.int8)
    B = S.shape[0]
    return spatial_channels_v7(boards, np.full(B, _OFFICIAL_TO_PLAY, np.int8), KO,
                               np.full((B, 3), -1, np.int16),
                               np.full((B, 3), -1, np.int16))


#: 逐通道的「不可对拍」原因（报告里要逐条说清，不静默省略）。
_NOT_COMPARABLE = {
    9: 'stdata 无着法序列（ch9..13 靠历史列，spec §9.4）',
    10: '同上',
    11: '同上',
    12: '同上',
    13: '同上',
    15: 'stdata 无「前一手盘」⇒ 走官方回退复制（spec §9.4）',
    16: 'stdata 无「前二手盘」⇒ 同上',
    18: '本 harness 用**单一**默认 rules_flags（AREA+TAX_NONE）跑全样本，'
        '而官方 ch18/19 是**逐行按规则分岔**的 ⇒ 全样本 row_exact 不是对齐率；'
        '按官方 globalInputNC 逐行分派后可比子集上 1.0，见 '
        'tests/test_v7_area_official.py',
    19: '同 ch18',
}

#: 必须**逐行 100% 精确相等**的通道。 这些掉下来就是**组装顺序**出了问题 ——
#: 查 `spatial_channels_v7`，**不要改底层实现**。
_PINNED_EXACT = (0, 1, 2, 3, 4, 5, 6, 8, 14, 17)

#: 必须**整格全 0 且官方也全 0** 的通道（encore-only / 源码从未写入）。
_PINNED_ZERO_BOTH = (8,)


@needs_stdata
def test_stdata_per_channel_match_table(stdata, capsys):
    """ **22 通道逐位对拍命中数字表**（`2026-08-25npzs.tgz`，前 12 个 npz 的
    前 40 个 19×19 行 = **301 行**）。

    口径是**逐行精确相等率**（整张 19×19 平面完全相同才算这一行相等），并**同时
    打印逐格一致率** —— 后者是个陷阱，见模块 docstring。
    """
    S, _, KO = stdata
    mine = _assemble_stdata(S, KO).astype(bool)
    rows = []
    for ch in range(SPATIAL_CHANNELS):
        row_exact = float((mine[:, ch] == S[:, ch]).all(axis=(1, 2)).mean())
        cell = float((mine[:, ch] == S[:, ch]).mean())
        rows.append((ch, row_exact, cell, int(mine[:, ch].sum()),
                     int(S[:, ch].sum())))
    with capsys.disabled():
        print(f'\n[stdata 逐通道对拍] rows={S.shape[0]} '
              f'({STDATA} 前 {_N_FILES} npz × {_ROWS_PER_FILE} 行)')
        print(' ch  row_exact%  cell_agree%  ours_nz  off_nz  备注')
        for ch, re, cell, onz, fnz in rows:
            note = _NOT_COMPARABLE.get(ch, '')
            if ch in _PINNED_EXACT:
                note = (note + ' / ' if note else '') + ' 已钉 1.0'
            print('%3d  %10.6f  %10.6f  %7d  %7d  %s'
                  % (ch, re, cell, onz, fnz, note))
    # 钉死的通道必须仍是 1.0
    for ch in _PINNED_EXACT:
        re = rows[ch][1]
        assert re == 1.0, (
            f'ch{ch} 与官方逐行精确相等率掉到 {re:.6f}（应为 1.0）。'
            f'\n **查 `spatial_channels_v7` 的组装顺序，不要改底层实现** —— '
            f'ch14/17 的算法由 tests/test_v7_ladders.py 担保（实测 1.000000 / 4368 行）。')
    # 恒 0 的通道：两边都必须全 0
    for ch in _PINNED_ZERO_BOTH:
        assert rows[ch][3] == 0 and rows[ch][4] == 0
    # ch18/19：**正向**断言 —— 在本 harness 的默认 flags 恰好与官方一致的
    # 那个子集上必须逐位全对。
    #
    # **旧断言（已作废）**：这里原来写的是 `assert rows[ch][1] < 1.0`，
    #   失败信息写「那说明本仓也提了死子，去查死子」。**那个根因已被推翻** ——
    #   官方 `Board::calculateArea*`（`board.cpp:1853-1937`）既不提子也不做死子
    #   判定，它的输入只有 `colors`。照着那条信息去查死子会把人带进死胡同。
    #   现在改成正向断言：对齐了就是 1.0，对不上就报真实数字。
    #
    # 为什么不能直接断言全样本 == 1.0：`_assemble_stdata` 用**单一**默认
    #   `rules_flags`（AREA+TAX_NONE+`multiStoneSuicideLegal=false`）跑全部行，
    #   而官方 ch18/19 是**逐行按规则分岔**的（`nninputs.cpp:2391-2439`）。所以
    #   只有「官方自己是 AREA ∧ TAX_NONE ∧ ch8==0」的行才该全对 ——
#   TERRITORY 行官方只在 `encorePhase>=2` 才发 area，TAX_SEKI/ALL 行要走
    #   双活过滤，本 harness 都喂不了。逐规则的完整分解（含
    #   `multiStoneSuicideLegal` 两种取值、4368 行）在
    #   `tests/test_v7_area_official.py`，那里是 1.000000 / 2282 行。
    # 本前缀样本的 `AREA ∧ TAX_NONE` 行 `globalInputNC[:,8]` 全是 1，也就是
    #   说默认 flags 的 `multiStoneSuicideLegal=False` 在这批行上**没被考到** ——
    #   所以这里只断言到 `TAX_NONE` 这一层，别在这里假装覆盖了 suicide 口径。
    G = stdata[1]
    subset = (G[:, 9] < 0.5) & (G[:, 10] < 0.5) & (G[:, 11] < 0.5)
    n_ch8 = int((subset & (G[:, 8] > 0.5)).sum())
    assert subset.sum() > 0, (
        f'这个样本里一条「官方=AREA+TAX_NONE」的行都没有（{S.shape[0]} 行）'
        f'⇒ ch18/19 的正向断言会空跑。换 stdata 批次或调大取样。')
    for ch in (18, 19):
        sub = (mine[subset, ch] == S[subset, ch]).all(axis=(1, 2))
        assert sub.all(), (
            f'ch{ch} 在 {int(subset.sum())} 条「官方=AREA+TAX_NONE」的'
            f'行上只有 {int(sub.sum())} 行逐位全对'
            f'（其中 {n_ch8} 行的 multiStoneSuicideLegal=1）。'
            f'\n 这批行是本 harness 的默认 flags **恰好**覆盖官方分支的子集，'
            f'掉下来就是 ch18/19 的实现回归 —— 查 '
            f'`area_ownership_map` / `_area_for_pla`，不要动装配层，也不要'
            f'去查死子（官方不做死子判定）。')


@needs_stdata
def test_stdata_ch14_and_ch17_are_bit_exact_through_the_assembler(stdata):
    """ch14 / ch17 **经过装配层**仍是逐行 100% 相等（官方口径下 ``to_play=+1``）。

    算法本身由 ``tests/test_v7_ladders.py`` 担保（实测 1.000000 / 4368 行）；
    这条抓的是「装配层把梯子的四格写错了位置」——而通道错位**不报任何错**。
    """
    S, _, KO = stdata
    mine = _assemble_stdata(S, KO).astype(bool)
    for ch, name in ((14, 'ladder（当前盘）'), (17, 'ladder working-move')):
        bad = int((~(mine[:, ch] == S[:, ch]).all(axis=(1, 2))).sum())
        assert bad == 0, f'ch{ch}（{name}）有 {bad}/{S.shape[0]} 行不逐位相等'


@needs_stdata
def test_stdata_ch3_ch4_ch5_stay_bit_exact_through_the_assembler(stdata):
    """ch3 / ch4 / ch5 经过装配层仍是逐行 100% 相等（气桶）。

    官方对这三个通道**不做任何额外处理** ⇒ 可以当 oracle（与 ch18/19 相反）。
    """
    S, _, KO = stdata
    mine = _assemble_stdata(S, KO).astype(bool)
    for ch in (3, 4, 5):
        bad = int((~(mine[:, ch] == S[:, ch]).all(axis=(1, 2))).sum())
        assert bad == 0, f'ch{ch}（{ch - 2} 气）有 {bad}/{S.shape[0]} 行不逐位相等'


@needs_stdata
def test_stdata_ch1_ch2_match_under_the_documented_to_play_assumption(stdata):
    """ch1 / ch2 在 ``to_play = +1`` 下与官方逐位相同（官方 ch1 = pla = 黑）。

     这条把「绝对色口径的对拍前提」写死：官方 stdata 的 ``to_play`` 实测恒 +1
      （`tests/test_v7_ladders.py` 用 ``to_play=+1`` 才拿到 1.000000）。若哪天
      换成 ``to_play=-1`` 的批次，ch1/ch2 会整体对调 —— 那正是这条测试会红的
      时刻，而不是等训练静默学歪。
    """
    S, _, KO = stdata
    mine = _assemble_stdata(S, KO).astype(bool)
    for ch in (1, 2):
        bad = int((~(mine[:, ch] == S[:, ch]).all(axis=(1, 2))).sum())
        assert bad == 0, (
            f'ch{ch} 有 {bad} 行不逐位相等 —— 若换过 stdata 批次，先确认 to_play 假设')


# --------------------------------------------------------------------------- #
# 与官方 globalInputNC 的对拍
# --------------------------------------------------------------------------- #
@needs_stdata
def test_global_ch18_matches_official_batch(stdata, capsys):
    """ **ch18 与官方 ``globalInputNC[:,18]`` 逐行 100% 相等**（301/301）。

    用官方 ch5 反推 ``selfKomi``（``ch5 × 20``）——官方 ch5 已经含 draw-jitter，
    而 ch18 用的正是同一个 selfKomi，所以这是**精确**代入而不是近似。板面积
    由 ch0 的 1 的个数反推；``scoringRule`` 由官方 ch9 读；``encorePhase`` 由
    官方 ch12/13 读（``ch18`` 的官方门控是
    ``scoringRule == AREA || encorePhase >= 2``）。

    这是**唯一**能把 spec §2.3 那张表里最含糊的一维（贴目 × 棋盘奇偶三角波）
    钉死的东西 —— 官方源码里它就是一段纯算术，没有别的可对拍面。
    """
    S, G, _ = stdata
    sk = (G[:, 5].astype(np.float64) * KOMI_SCALE).astype(np.float32)
    pred = np.empty(G.shape[0], dtype=np.float32)
    for i in range(G.shape[0]):
        terr = G[i, 9] > 0.5
        enc = 2 if G[i, 13] > 0.5 else (1 if G[i, 12] > 0.5 else 0)
        pred[i] = komi_parity_wave(sk[i:i + 1], N * N,
                                   SCORING_TERRITORY if terr else SCORING_AREA,
                                   enc)[0]
    hit = np.isclose(pred, G[:, 18], atol=2e-6)
    with capsys.disabled():
        print('\n[global ch18 对拍] %d/%d = %.6f'
              % (int(hit.sum()), len(hit), float(hit.mean())))
    bad = np.flatnonzero(~hit)
    assert bad.size == 0, (
        f'ch18 有 {bad.size}/{len(hit)} 行不等（率 {hit.mean():.6f}）。'
        f'首个失配行 idx={bad[:5].tolist()}：'
        f'selfKomi={sk[bad[:5]].tolist()} 我们={pred[bad[:5]].tolist()} '
        f'官方={G[bad[:5], 18].tolist()}')


@needs_stdata
def test_global_rule_channels_match_official_batch(stdata, capsys):
    """ **规则条件化的 7 个通道（全局 ch6..ch11 + ch17）与官方逐位相等**。

    喂进去的 ``rules_flags`` 是**从官方自己的 ch6..ch11/ch17 反解**的
    （``ko`` 的 3 态 / ``tax`` 的 3 态 / suicide / territory / button），
    ⇒ 这条测的是「规则位 → 通道」这段映射，而不是「规则位怎么来的」。

    官方数据里 9 种规则组合全都出现了（301 行覆盖 3 种 ko × 3 种 tax ×
    suicide × territory × button），所以三条规则、两个布尔的**每个取值**都被
    官方数据验证过 —— 不只是「默认值恰好对」。
    """
    S, G, _ = stdata
    B = G.shape[0]
    ko_rule = np.where(G[:, 6] > 0.5,
                       np.where(G[:, 7] > 0, KO_POSITIONAL, KO_SITUATIONAL),
                       KO_SIMPLE)
    tax = np.where(G[:, 10] < 0.5, TAX_NONE,
                   np.where(G[:, 11] > 0.5, TAX_ALL, TAX_SEKI))
    suicide = G[:, 8] > 0.5
    territory = G[:, 9] > 0.5
    button = G[:, 17] > 0.5
    key = (((ko_rule.astype(np.int64) * 100 + tax) * 4 + suicide) * 2
           + territory) * 2 + button
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(int(k), []).append(i)

    my = np.full((B, 3), -1, np.int16)
    op = np.full((B, 3), -1, np.int16)
    checked, lines = 0, []
    for k, idxs in sorted(groups.items()):
        idxs = np.asarray(idxs)
        fl = rules_flags_from(
            scoring=SCORING_TERRITORY if territory[idxs[0]] else SCORING_AREA,
            tax=tax[idxs[0]], ko=ko_rule[idxs[0]],
            multi_stone_suicide=suicide[idxs[0]], has_button=button[idxs[0]])
        out = global_features_v7(GameRow(komi=0.0), my[idxs], op[idxs],
                                 history_length=0, rules_flags=fl,
                                 to_play=np.ones(len(idxs), np.int8))
        for ch in (6, 7, 8, 9, 10, 11, 17):
            assert np.array_equal(out[:, ch].astype(np.float32), G[idxs, ch]), (
                f'规则组 ko={ko_rule[idxs[0]]} tax={tax[idxs[0]]} '
                f'sui={suicide[idxs[0]]} terr={territory[idxs[0]]} '
                f'btn={button[idxs[0]]} 的全局 ch{ch} 对不上官方')
        checked += len(idxs)
        lines.append('ko=%d tax=%d sui=%d terr=%d btn=%d n=%d'
                     % (ko_rule[idxs[0]], tax[idxs[0]], suicide[idxs[0]],
                        territory[idxs[0]], button[idxs[0]], len(idxs)))
    with capsys.disabled():
        print('\n[global ch6..ch11+ch17 对拍] %d/%d 行、%d 种规则组合'
              % (checked, B, len(groups)))
        for line in lines:
            print('   ', line)
    assert len(groups) >= 8, f'官方数据只覆盖了 {len(groups)} 种规则组合，样本不足'
    assert checked == B


@needs_stdata
def test_global_encore_and_pda_channels_match_official(stdata):
    """全局 ch12 / ch13（encorePhase）与 ch15 / ch16（恒 0）与官方一致。

    官方这批数据里**确实有 encore 行**（所以不能一口断言「官方全 0」）⇒ 逐行
    按官方 ch12/ch13 分组喂 ``encore_phase``，再逐位比。
    ch15/ch16 是 PDA，本仓无 PDA ⇒ 恒 0；官方实测这两格非零率 0%。
    """
    _, G, _ = stdata
    B = G.shape[0]
    my = np.full((B, 3), -1, np.int16)
    op = np.full((B, 3), -1, np.int16)
    enc_of = np.where(G[:, 13] > 0.5, 2, np.where(G[:, 12] > 0.5, 1, 0))
    for enc in (0, 1, 2):
        idxs = np.flatnonzero(enc_of == enc)
        if idxs.size == 0:
            continue
        out = global_features_v7(GameRow(komi=7.5), my[idxs], op[idxs],
                                 history_length=0,
                                 to_play=np.ones(idxs.size, np.int8),
                                 encore_phase=int(enc))
        want12, want13 = float(enc > 0), float(enc > 1)
        assert (out[0, 12], out[0, 13]) == (want12, want13)
        for ch in (12, 13):
            assert np.array_equal(out[:, ch].astype(np.float32), G[idxs, ch]), (
                f'encore_phase={enc} 那组里的全局 ch{ch} 对不上官方')
    assert not G[:, 15].any() and not G[:, 16].any(), \
        '官方这批数据里出现了 PDA 行 ⇒ 本仓「无 PDA ⇒ 恒 0」的假设要重新对账'


@needs_stdata
def test_global_channels_without_a_move_sequence_are_reported_not_faked(stdata):
    """ch0..4 / ch14 依赖着法序列，stdata 没有 ⇒ **如实标注不可对拍**。

    这里断言的是「我们给 0、官方给的不是 0」这个事实本身，而不是把阈值调松
    让它绿。 ch18/19 的教训：猜判据会得到「既不等于本仓也不等于官方」的第三种
      口径，而这类偏差是**静默**的 —— 只会表现为训练分布偏移。
    """
    _, G, _ = stdata
    B = G.shape[0]
    my = np.full((B, 3), -1, np.int16)
    op = np.full((B, 3), -1, np.int16)
    out = global_features_v7(GameRow(komi=7.5), my, op, history_length=0,
                             to_play=np.ones(B, np.int8))
    assert not out[:, 0:5].any()          # 我们的 ch0..4 全 0
    assert G[:, 0:5].sum() > 0            # 官方不是 0 ⇒ 确实不可对拍
    # ch5 我们喂的 komi 是夹具值 ⇒ 也不与官方的 jittered 值可比
    assert out[:, 5].any() and G[:, 5].any()
    # ch14（上一手是不是 pass）同理。 实测**这批官方数据里 ch14 恒 0**
    # （301 行），所以它在这批数据上无法被验证 —— 只能靠单测
    # （`tests/test_v7_globals.py::test_ch14_is_true_exactly_when_the_previous_move_was_a_pass`）。
    # 这里把「官方也没能验证它」这件事记下来，而不是让绿测暗示它被覆盖了。
    assert not G[:, 14].any(), (
        '官方这批数据里出现了 ch14 非 0 的行 ⇒ 端到端对拍可以做了，别再跳过')


# --------------------------------------------------------------------------- #
# 全局与空间两条路径的接缝
# --------------------------------------------------------------------------- #
def test_global_and_spatial_share_one_history_slot_order():
    """全局 ch0..4 与空间 ch9..13 **用同一份物理手序**（`feature_v7._HISTORY_SLOTS`）。

     同一份顺序写两遍就是下一次「通道顺序静默错位」的入口，而通道错位**不报
      任何错**，只是标签指向错误的点。本测试用「最近一手落在 ch0 且 ch9 的同
      一个点上」把两者钉在一起。
    """
    moves = [r * N + c for r, c in HIST_POINTS]      # 最近一手在前
    my = np.array([[-1, -1, -1]], np.int16)
    op = np.array([[-1, -1, -1]], np.int16)
    for k, mv in enumerate(moves):
        (op if k % 2 == 0 else my)[0, k // 2] = mv
    sp = spatial_channels_v7(BOARDS, TO_PLAY, KO, my, op)
    gl = global_features_v7(GameRow(komi=7.5), my, op, history_length=5,
                            to_play=TO_PLAY)
    for k, mv in enumerate(moves):
        r, c = divmod(mv, N)
        assert sp[0, 9 + k, r, c] == 1.0, f'第 {k + 1} 手没落在 ch{9 + k}'
        assert gl[0, k] == 0.0, f'第 {k + 1} 手不是 pass ⇒ 全局 ch{k} 应为 0'


def test_self_komi_is_the_input_of_ch18_so_the_two_agree():
    """``global_features_v7`` 的 ch18 必须由它自己的 ``self_komi`` 算出来。

    这条抓「ch18 用了裸 komi 而不是 selfKomi」—— 那在 ``to_play = +1`` 时完全
    看不出来（符号相同），只有白走的行会错。
    """
    for tp in (1, -1):
        gl = global_features_v7(GameRow(komi=7.5), np.full((1, 3), -1, np.int16),
                                np.full((1, 3), -1, np.int16),
                                history_length=0, to_play=np.array([tp], np.int8))
        sk = self_komi(np.array([7.5]), np.array([tp], np.int8), N * N)
        want = komi_parity_wave(sk, N * N)[0]
        assert gl[0, 18] == pytest.approx(float(want), abs=2e-3)