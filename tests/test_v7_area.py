# -*- coding: utf-8 -*-
"""V7 空间通道 B3：ch18/ch19（`calculateArea`）+ 区域计分归属图。

被测对象 `src/data/feature_v7.py`：
  · `area_ownership_map` —— 逐点归属图（`GoBoard.score()` 的批量版，逐格同构）
  · `calculate_area`      —— 规则条件化 + 视角翻转（spec §2.5）

**核心断言是「与 `GoBoard.score()` 的口径一致性」**（brief 点名的验收项）：
随机盘面上 `ownership_map(board).sum() == GoBoard.score() + komi`。
加上一个**独立写的** Python 逐点参考实现做三方对拍（向量化 ↔ 参考实现 ↔ `score()`），
因为「sum 相等」单独看太弱：正负抵消可以让两张完全不同的归属图给出同一个和。

⚠ **ch18/19 不能与官方 stdata 对拍**。实测 1254 行真实数据，我们比官方多
21,896 点（其中 19,913 个是**子**），原因见
`src/data/feature_v7.py::area_ownership_map` 的 docstring（官方算 area 前先提死子，
`GoBoard.score()` 不做）。本文件因此**不**假装 stdata 是这里的 oracle。

运行：
    python -m pytest tests/test_v7_area.py -q
"""

import os

import numpy as np
import pytest

from src.data.feature_v7 import (
    SCORING_AREA,
    SCORING_TERRITORY,
    TAX_ALL,
    TAX_NONE,
    TAX_SEKI,
    area_ownership_map,
    calculate_area,
)
from src.game.go_rules import GoBoard

N = 19
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def board(rows):
    """ASCII 夹具 → `(1,19,19)` int8。`X`=+1(黑/pla)、`O`=−1(白/opp)、`.`=空。

    短行**右侧补 `.` 到 19**（左对齐）—— 与 `tests/test_v7_planes.py` 同一约定。
    """
    assert len(rows) <= N
    a = np.zeros((1, N, N), dtype=np.int8)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == 'X':
                a[0, r, c] = 1
            elif ch == 'O':
                a[0, r, c] = -1
            elif ch != '.':
                raise ValueError(f'非法字符 {ch!r}')
    return a


# --------------------------------------------------------------------------- #
# 参考实现：独立写的逐点 flood fill（**不复用**被测代码的任何一行）
# --------------------------------------------------------------------------- #
def _ref_ownership(b):
    """`GoBoard.score()` 的逐点版：Python 双层循环 + 栈式 flood fill。

    ⚠ **为什么必须独立写一遍**：向量化版（`area_ownership_map`）与
      `GoBoard.score()` 共用同一套「空区域 + 边界颜色集合」的判定思路，
      而「sum == score() + komi」这一条允许两张不同的图互相抵消。所以
      「三方对拍」（参考实现 ↔ 向量化 ↔ `score()` 的标量）才是可信的证据。
    """
    empty = b == 0
    seen = np.zeros((N, N), dtype=bool)
    own = np.where(b > 0, 1, np.where(b < 0, -1, 0)).astype(np.int8)
    for r in range(N):
        for c in range(N):
            if not empty[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            region = []
            border = set()
            while stack:
                cr, cc = stack.pop()
                region.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < N and 0 <= nc < N):
                        continue
                    if empty[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
                    elif b[nr, nc] != 0:
                        border.add(int(b[nr, nc]))
            if len(border) == 1:
                for cr, cc in region:
                    own[cr, cc] = 1 if border == {1} else -1
    return own


# --------------------------------------------------------------------------- #
# 空盘
# --------------------------------------------------------------------------- #
def test_area_empty_board_is_all_zero():
    """空盘 ⇒ 归属图全 0。

    判别式：整块盘面是一片「四邻**谁都不相邻**」的空区域，`len(border_colors) == 1`
    不成立 ⇒ 中立（`GoBoard.score()` 里 black_territory/white_territory 都是 0）。
    ⚠ 顺带钉住 dtype：空盘返回**全 0 的 int8 数组**而不是空数组/None ——
      一条空盘样本在真实 batch 里是常态（每个新开局的头几手）。
    """
    out = calculate_area(np.zeros((2, N, N), np.int8),
                         np.array([1, -1], np.int8), 0)
    assert out.shape == (2, 2, N, N)
    assert out.dtype == np.int8
    assert not out.any(), '空盘不该有任何属地'


# --------------------------------------------------------------------------- #
# 手工构造：黑围 / 白围 / 两色共邻
# --------------------------------------------------------------------------- #
_ENCLOSURES = board([
    # (5,5) 的空点被 8 颗白子四面围住 ⇒ 归 −1（opp）
    '...................',
    '...................',
    '...................',
    '...................',
    '....OOO............',
    '....O.O............',
    '....OOO............',
    '.......X.O........',      # (7,7)=黑 (7,9)=白 ⇒ (7,8) 两色共邻 ⇒ 中立
    '........XXX........',      # (9,9) 的空点被 8 颗黑子四面围住 ⇒ 归 +1（pla）
    '........X.X........',
    '........XXX........',
] + ['...................'] * (N - 11))


def test_area_single_point_enclosures():
    """黑围的空点 +1、白围的空点 −1、两色共邻的空点 0（**逐格**断言）。

    逐格而不是只看聚合值：三态里错一态时聚合值常常仍然对（一个 +1 和一个 −1
    抵消），而这正是归属图最容易错的地方。
    """
    out = calculate_area(_ENCLOSURES, np.array([1], np.int8), 0)
    pla, opp = out[0, 0], out[0, 1]
    assert pla[5, 5] == -1, '被白子围住的空点不归 pla'
    assert opp[5, 5] == 1, '被白子围住的空点归 opp（符号翻转）'
    assert pla[9, 9] == 1, '被黑子围住的空点归 pla'
    assert opp[9, 9] == -1, '被黑子围住的空点不归 opp'
    assert pla[7, 8] == 0 and opp[7, 8] == 0, '两色共邻的空区应中立'


def test_area_whole_region_is_filled_not_just_one_point():
    """一片被围的空**区域**整片归属，不是只点亮其中一点。

    这是向量化实现最容易犯的错（只对区域里的**代表点**赋值，而不是对整片赋值）：
    单点区域测不出来 —— 单点区域里「代表点」恰好就是全部。
    """
    region = board([
        '..XXX',      # (1,2..4)
        '.X..X',      # (2,2)=(2,3)=空，被黑子封住 ⇒ 整片归黑
        '..XXX',
    ])
    own = area_ownership_map(region)[0]
    assert own[2, 2] == 1 and own[2, 3] == 1, \
        f'两格连成的围空区域应整片归黑，实得 ({own[2, 2]}, {own[2, 3]})'
    assert own[2, 1] == 1 and own[2, 4] == 1, '围空区域两侧的棋子也归黑'


def test_area_stones_belong_to_their_own_colour():
    """有子的点归该子自己的颜色（区域计分里子本身算属地）。

    这条是「区域计分 vs 领土计分」的分界：territory 口径下棋子**不算**属地。
    写错成「只算围空」会让 ch18/ch19 丢掉所有己方棋子 —— 一个巨大的、
    不抛异常的静默错。
    """
    own = area_ownership_map(_ENCLOSURES)[0]
    assert own[7, 7] == 1, '黑子应归黑'
    assert own[4, 5] == -1, '白子应归白'
    assert own[2, 2] == 0, '空的未围点应中立'


# --------------------------------------------------------------------------- #
# 与 GoBoard.score() 的口径一致性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('komi', [0.0, 7.5, -6.5])
def test_ownership_sum_agrees_with_go_board_score(komi):
    """**核心不变式**：`ownership_map(board).sum() == GoBoard.score() + komi`。

    为什么要自己加回 komi：`GoBoard.score()` 返回的是「黑分 − 白分 − komi」的
    **净胜值**，而归属图的和是「黑属地 + 黑子 − 白属地 − 白子」，两者恰好差一个
    komi 常数。这个常数是计分规则的定义、不是实现的偏差，所以测试里显式写清
    「加回来」，而不是把 komi 设成 0 去绕开它（设 0 会让这条断言失去
    「黑方先手必须贴目」这个真实场景的覆盖）。

    ⚠ **为什么这条还不够**：正负抵消可以让两张不同的归属图给出同一个和。
      逐格等价由下面的 `test_ownership_matches_independent_reference` 负责，
      两条合起来才是 brief 要求的「口径一致性」。
    """
    rng = np.random.default_rng(20261002)
    for trial in range(40):
        x = rng.random((N, N))
        b = np.where(x < 0.30, 1, np.where(x < 0.60, -1, 0)).astype(np.int8)
        g = GoBoard(N, komi=komi)
        g.board = b.copy()
        got = int(area_ownership_map(b[None])[0].sum())
        want = g.score() + komi
        assert got == want, (
            f'第 {trial} 个随机盘面（komi={komi}）：归属图求和 {got} '
            f'!= score() {g.score()} + komi {komi} = {want}')


def test_ownership_matches_independent_reference():
    """**逐格**等价于一个独立写的 Python 参考实现（随机盘面 40 个）。

    与上一条互补：上一条管「总量」，这条管「每一格」。两者都要 ——
    只有总量会被抵消骗过，只有逐格抓不到 komi 约定。
    """
    rng = np.random.default_rng(11)
    for trial in range(40):
        x = rng.random((N, N))
        b = np.where(x < 0.28, 1, np.where(x < 0.56, -1, 0)).astype(np.int8)
        got = area_ownership_map(b[None])[0]
        want = _ref_ownership(b)
        if not np.array_equal(got, want):
            rr, cc = np.argwhere(got != want)[0]
            raise AssertionError(
                f'第 {trial} 个随机盘面：({rr},{cc}) 我们={got[rr, cc]} '
                f'参考={want[rr, cc]}（共 {int((got != want).sum())} 格不同）')


def test_ownership_full_board_and_one_stone_per_colour():
    """两个退化形状：满盘（无空点 ⇒ `label()` 返回 `num == 0` 的短路分支），
    以及「黑白各一颗子」（整块盘面是一大片空区、边界只有两种颜色 ⇒ 全中立）。

    两者都不是随机盘面容易覆盖到的分支。
    ⚠ 注意**单独一颗黑子**不是退化形状：其余 360 个空点构成一个只邻黑子的区域，
      按 Tromp-Taylor **全部归黑**（`sum == 361`）。这条容易被误当成 bug，
      所以这里改用「黑白各一颗」—— 边界有两种颜色，那片空区才真的中立。
    """
    full = np.zeros((1, N, N), np.int8)
    full[0, ::2, ::2] = 1
    full[0, 1::2, 1::2] = -1
    assert int(area_ownership_map(full)[0].sum()) == int(full[0].sum())

    two = np.zeros((1, N, N), np.int8)
    two[0, 0, 0] = 1
    two[0, N - 1, N - 1] = -1
    own = area_ownership_map(two)[0]
    assert own[0, 0] == 1 and own[N - 1, N - 1] == -1
    assert int(own.sum()) == 0, \
        f'黑白各一颗 ⇒ 其余空点两色共邻，应全中立；实得 sum={int(own.sum())}'


# --------------------------------------------------------------------------- #
# 通道布局：两个视角 + to_play 翻符号
# --------------------------------------------------------------------------- #
def test_channels_are_two_views_of_one_map_and_to_play_flips_signs():
    """ch19 == −ch18，且 `to_play` 换边时两格整体翻符号。

    钉的是**布局契约**：ch18 恒是「pla 视角」（+1 = pla 的属地）、
    ch19 恒是「opp 视角」，所以下游二值化（`>0` / `<0`）两格对称，
    且换边只需翻 `to_play`、不用改代码。
    """
    boards = np.stack([_ENCLOSURES[0], _ENCLOSURES[0]])
    for to_play in (1, -1):
        out = calculate_area(boards, np.array([to_play, to_play], np.int8), 0)
        assert np.array_equal(out[:, 1], -out[:, 0]), \
            f'to_play={to_play}：ch19 应是 ch18 的精确取反'
        # pla 的属地必须随 to_play 落在不同点上：黑围的空点归 pla 只在 pla=黑时成立
        black_enclosed = np.zeros((N, N), bool)
        black_enclosed[9, 9] = True
        white_enclosed = np.zeros((N, N), bool)
        white_enclosed[5, 5] = True
        pla_plane = out[0, 0] > 0
        if to_play == 1:
            assert pla_plane[9, 9] and not pla_plane[5, 5]
        else:
            assert pla_plane[5, 5] and not pla_plane[9, 9]


def test_output_values_are_strictly_minus_one_zero_plus_one():
    """两格的取值严格 ∈ {−1, 0, +1}（下游要按 `>0`/`<0` 二值化的前提）。"""
    rng = np.random.default_rng(3)
    x = rng.random((5, N, N))
    boards = np.where(x < 0.25, 1, np.where(x < 0.5, -1, 0)).astype(np.int8)
    out = calculate_area(boards, np.array([1, -1, 1, -1, 1], np.int8), 0)
    assert set(np.unique(out).tolist()) <= {-1, 0, 1}


# --------------------------------------------------------------------------- #
# 规则条件化（spec §2.5）
# --------------------------------------------------------------------------- #
def test_scoring_territory_returns_all_zero():
    """**TERRITORY ⇒ 全 0**（spec §2.5）。

    这是**对齐官方**、不是缺陷：官方的 territory 分支被 `encorePhase >= 2`
    二次条件包着，而本仓无 encore 机制（spec §2.6 D2）⇒ 正常阶段恒 0。
    SGF 缺 `RU` 时默认 AREA，所以绝大多数对局不走这条分支。
    ⚠ 这条与「TAX 还没实现时静默返回全 0」在张量上完全一样 —— 所以下面两条
      必须钉「tax 分支抛异常」，否则 territory 的 0 与「算不出来」的 0
      永远分不开。
    """
    flags = SCORING_TERRITORY | (TAX_NONE << 1)
    out = calculate_area(_ENCLOSURES, np.array([1], np.int8), flags)
    assert out.shape == (1, 2, N, N) and out.dtype == np.int8
    assert not out.any(), f'territory 口径下 ch18/19 应全 0（bit0=1, flags={flags}）'


def test_scoring_territory_wins_over_tax_seki_per_official_branch_order():
    """**TERRITORY + TAX_SEKI ⇒ 全 0**（官方 if/else 的顺序：先判 AREA）。

    官方（spec §2.5 照录）::

        if (SCORING_AREA && TAX_NONE)        calculateArea(...)
        elif (SCORING_AREA && (SEKI|ALL))   calculateIndependentLifeArea(keepStones=true)
        elif (SCORING_TERRITORY)            // 仅 encorePhase >= 2 才置位

    所以 TERRITORY 无论配什么 tax 都落在**第三**分支 ⇒ 全 0，而**不是**抛错。
    把 if/else 顺序写反（先判 tax）会让这个组合抛 `NotImplementedError`，
    这条断言专门抓那种改写。
    """
    for tax in (TAX_NONE, TAX_SEKI, TAX_ALL):
        flags = SCORING_TERRITORY | (tax << 1)
        out = calculate_area(_ENCLOSURES, np.array([1], np.int8), flags)
        assert not out.any(), f'territory + tax={tax} 应全 0，实得 {int(out.sum())}'


def test_tax_seki_and_tax_all_raise_not_implemented_error():
    """**AREA + TAX_SEKI / TAX_ALL ⇒ 抛 `NotImplementedError`，绝静默返回全 0。**

    为什么必须抛而不是返回 0：全 0 与「TERRITORY 恒 0」在张量上**逐位相同**，
    静默返回 0 会让这些对局的 ch18/19 变成「无归属」，而模型会照样训练 ——
    唯一症状是这些局的 ownership loss 恒为常数。这类错**没有任何信号**。
    所以这里钉「抛」，让规则条件化那一步能显式拦下（或先补齐实现）。

    官方 `calculateIndependentLifeArea(keepStones=true)` 的语义写在
    `calculate_area` 抛出的消息里，本轮**不猜**它的判据。
    """
    for tax, name in ((TAX_SEKI, 'TAX_SEKI'), (TAX_ALL, 'TAX_ALL')):
        flags = SCORING_AREA | (tax << 1)
        with pytest.raises(NotImplementedError) as ei:
            calculate_area(_ENCLOSURES, np.array([1], np.int8), flags)
        msg = str(ei.value)
        assert name in msg, f'异常消息应点名 {name}，实得 {msg!r}'
        assert 'calculateIndependentLifeArea' in msg, \
            '异常消息应说明官方走的是哪个函数，否则接手的人查不到对照源码'


def test_area_none_rules_is_the_default_and_does_not_raise():
    """`rules_flags` 默认 0 = AREA + TAX_NONE，正是 SGF 缺 `RU` 时的默认值。

    这条钉的是「默认路径可跑」：预取器在还没接上 `games.npz`（spec §5.3）
    之前调本函数时不能炸。
    """
    out = calculate_area(_ENCLOSURES, np.array([1], np.int8))
    assert out[0, 0][9, 9] == 1, '默认 flags 下应该真的算出围空'


def test_calculate_area_rejects_wrong_rank():
    with pytest.raises(ValueError, match='boards 形状不符'):
        calculate_area(np.zeros((N, N), np.int8), np.array([1], np.int8), 0)
