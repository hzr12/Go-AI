# -*- coding: utf-8 -*-
"""V7 空间通道 B3：ch18/ch19 —— 官方 `Board::calculateArea*` 口径的归属图。

被测对象 `src/data/feature_v7.py`：
  · `area_ownership_map` —— 逐点归属图（**官方绝对色口径**，逐字照抄
    `board.cpp:1853-2327`）
  · `calculate_area`      —— 规则条件化 + 视角翻转（spec §2.5）

    ---- 🔴 本文件曾经断言一个**错误的不变式**，已退役 ----
旧版这里的核心断言是 `ownership_map(board).sum() == GoBoard.score() + komi`，
配套一个 Tromp-Taylor 逐点参考实现。**那条不变式是错的**：官方 area 根本不是
Tromp-Taylor 区域计分 —— 它多了两层算法（Benson 无条件存活 + 双活整块过滤），
在真实对局里两者都会大面积生效。旧 docstring 把差异归因为「官方算 area 前先提
死子」，**那个根因也是错的**（官方既不提子也不做死子判定，见
`area_ownership_map` 的 docstring）。两处都已按官方源码改正。

现在这个文件守的是三件事：
  1. **逐格**等价于一份**独立写的**官方算法参考实现（`_ref_area`，纯 Python
     抄写 `board.cpp`，与被测的向量化实现零共享代码）；
  2. 两条从官方源码推出来的、**能区分新旧口径**的结构性质（见
     `test_seki_tax_never_adds_area`、`test_tax_seki_and_tax_all_share_the_area_planes`）；
  3. 规则条件化（TERRITORY 恒 0、TAX_SEKI/ALL 抛错）与输出取值域。

与官方 stdata 的**逐位**对拍在 `tests/test_v7_area_official.py`（那里才有真实
归档；本文件不依赖归档，`pytest tests/test_v7_area.py` 随时可跑）。

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
# 参考实现：**独立写的**官方算法逐字抄写（`board.cpp:1853-2327`）
#
# ⚠ 这里**不能**再放 Tromp-Taylor 参考实现了：旧版那个 `_ref_ownership`
#   编码的是「空区域 + 边界颜色集合」判定，与官方 area 不同族。它作为 oracle
#   会把实现往回拽到旧口径 —— 而旧口径与官方 stdata 逐行 0 行相同。
# --------------------------------------------------------------------------- #
def _nb4(n, cell):
    r, c = cell
    if r > 0:
        yield (r - 1, c)
    if r + 1 < n:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < n:
        yield (r, c + 1)


def _chains(b, n, pla):
    """同色 4-邻接链：`{cell: chain_id}`（`board.cpp` 的 `chain_head`）。"""
    out = {}
    for r in range(n):
        for c in range(n):
            if b[r, c] != pla or (r, c) in out:
                continue
            cid = len(set(out.values())) + 1
            stack = [(r, c)]
            out[(r, c)] = cid
            while stack:
                cur = stack.pop()
                for nb in _nb4(n, cur):
                    if b[nb] == pla and nb not in out:
                        out[nb] = cid
                        stack.append(nb)
    return out


def _regions(b, n, pla, chain):
    """「空点 ∪ opp 子」的 4-邻接极大连通块 + 官方记的属性（`:2006-2130`）。

    `vital` 是**逐点过滤后**的结果（`:2033-2049`）；`num_internal` 封顶 2
    （`:2052-2054`）；`contains_opp`（`:2056-2057`）。
    """
    opp = -pla
    seen = set()
    out = []
    for r in range(n):
        for c in range(n):
            # 🔴 区域**只能从空点起头**（`board.cpp:2083-2086`：`colors[loc] !=
            #   C_EMPTY` 那一支只更新 `atLeastOnePla` 然后 continue）。被本方子
            #   四面围住的一颗对方子因此**不属于任何区域** —— 若让它当区域头，
            #   它会拿到一个「从未经过滤的 vital 初始表」（头部相邻的链全算
            #   vital，`:2100-2120`），那是官方根本不会产生的结果。
            if b[r, c] != 0 or (r, c) in seen:
                continue
            cells, stack = [], [(r, c)]
            seen.add((r, c))
            while stack:
                cur = stack.pop()
                cells.append(cur)
                for nb in _nb4(n, cur):
                    if b[nb] in (0, opp) and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            vital = {chain[nb] for nb in _nb4(n, (r, c)) if b[nb] == pla}
            num_internal = 0
            contains_opp = False
            for cell in cells:
                if num_internal < 2 and not any(b[nb] == pla
                                                for nb in _nb4(n, cell)):
                    num_internal += 1
                if b[cell] == opp:
                    contains_opp = True
                vital = {cid for cid in vital
                         if any(b[nb] == pla and chain[nb] == cid
                                for nb in _nb4(n, cell))}
            out.append({'cells': cells, 'vital': vital,
                        'num_internal': num_internal,
                        'contains_opp': contains_opp, 'borders': False})
    return out


def _ref_area_for_pla(b, n, pla, suicide_legal, result):
    """`Board::calculateAreaForPla(pla, safe=true, unsafe=true, suicide)`。

    ⚠ `result` 由调用方填好 C_EMPTY **一次**（`board.cpp:1860-1862`：先
      `std::fill`，再黑先白后共用同一块缓冲）—— 本函数里绝不能再填，否则白的
      那一遍会把黑的结果整个抹掉，`:2237` 的 C_EMPTY 守卫随之失效。
      这正是被测实现里那条「黑先白后顺序不能换」的由来。
    """
    chain = _chains(b, n, pla)
    cells_of = {}
    for cell, cid in chain.items():
        cells_of.setdefault(cid, []).append(cell)
    regions = _regions(b, n, pla, chain)
    at_least_one_pla = any(b[r, c] == pla for r in range(n) for c in range(n))

    vital_count = {cid: 0 for cid in cells_of}
    for reg in regions:
        for cid in reg['vital']:
            vital_count[cid] += 1

    killed = set()
    while True:                                   # Benson 定点迭代 :2159-2195
        progressed = False
        for cid in list(cells_of):
            if cid in killed:
                continue
            if vital_count[cid] < 2:               # :2168
                killed.add(cid)
                progressed = True
                for cell in cells_of[cid]:
                    for nb in _nb4(n, cell):
                        for reg in regions:
                            if nb in reg['cells'] and not reg['borders']:
                                reg['borders'] = True
                                for h in reg['vital']:
                                    vital_count[h] -= 1
        if not progressed:
            break

    for cid, cells in cells_of.items():
        if cid not in killed:                       # :2202-2211
            for cell in cells:
                result[cell] = pla
    for reg in regions:
        should = (reg['num_internal'] <= 1 and not reg['borders']
                  and at_least_one_pla)             # :2221
        should = should or (not reg['contains_opp'] and not reg['borders']
                            and at_least_one_pla)    # :2222
        if should:
            for cell in reg['cells']:
                result[cell] = pla
        elif not reg['contains_opp'] and at_least_one_pla:     # :2233
            for cell in reg['cells']:
                if result[cell] == 0:
                    result[cell] = pla              # :2237 C_EMPTY 守卫
    return result


def _ref_area(b, suicide_legal=False, tax_rule=TAX_NONE):
    """整条 `area_ownership_map` 的参考实现。"""
    n = len(b)
    basic = {(r, c): 0 for r in range(n) for c in range(n)}
    _ref_area_for_pla(b, n, 1, suicide_legal, basic)       # 黑先
    _ref_area_for_pla(b, n, -1, suicide_legal, basic)      # 白后（同一块缓冲）
    for r in range(n):                                    # :1865-1873 / :1892-1898
        for c in range(n):
            if basic[(r, c)] == 0:
                basic[(r, c)] = b[r, c]
    if tax_rule == TAX_NONE:
        return basic                                       # 官方 calculateArea
    return _ref_independent_life(b, n, basic)


def _ref_independent_life(b, n, basic):
    """双活过滤（`:2264-2296`）+ keepStones（`:1927-1935`）。"""
    atari = set()
    for pla in (1, -1):
        chain = _chains(b, n, pla)
        cells_of = {}
        for cell, cid in chain.items():
            cells_of.setdefault(cid, []).append(cell)
        for cells in cells_of.values():
            libs = {nb for cell in cells for nb in _nb4(n, cell) if b[nb] == 0}
            if len(libs) == 1:                            # :2270 触发①
                atari.update(cells)
    seki = set()
    for cell in sorted(basic):
        if basic[cell] == 0 or cell in seki:
            continue
        trigger = (b[cell] == basic[cell] and cell in atari) or any(
            b[nb] == 0 and basic.get(nb, 0) == 0 for nb in _nb4(n, cell))  # :2272
        if not trigger:
            continue
        pla = basic[cell]
        seki.add(cell)                                   # :2280-2292 整块 flood
        stack = [cell]
        while stack:
            cur = stack.pop()
            for nb in _nb4(n, cur):
                if basic.get(nb) == pla and nb not in seki:
                    seki.add(nb)
                    stack.append(nb)
    own = {cell: (0 if cell in seki else basic[cell]) for cell in basic}
    for cell in basic:                                   # keepStones
        if b[cell] != 0 and basic[cell] == b[cell]:       # :1931
            own[cell] = basic[cell]
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

    ⚠ **旧 docstring 的理由是错的，结论恰好是对的**。旧版写「按 Tromp-Taylor
      归黑/归白」—— 现在这两个断言走的是完全不同的路径，且官方输出**确实**
      是 −1/+1，但理由是：
        · `(5,5)` 被 8 颗白子围住。对**黑**那一遍，它是 1 点区域、
          `numInternal=1` ⇒ `:2221` 命中 ⇒ 写 +1；随后**白**那一遍
          （`board.cpp:1862`，黑先白后）`numInternal=0` ⇒ `:2221` 再次命中，
          而 `:2221` 是**无条件覆盖** ⇒ 最终 −1。
        · `(9,9)` 对称地最终 +1。
      所以这条现在钉的是「**白能覆盖黑**」这个覆盖语义，而不是区域计分。
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
    """有子的点归该子自己的颜色。

    ⚠ **旧 docstring 说这是「区域计分 vs 领土计分」的分界** —— 那是 Tromp-Taylor
      的分界，官方 area 不分这个：子归自己颜色来自 `nonPassAliveStones` 兜底
      （`board.cpp:1865-1873`：`result` 仍为 C_EMPTY 的**子**取自己的颜色），
      或者来自 Benson 的「无条件存活子」（`:2202-2211`）。
      ⚠ **这条断言仍不是无条件的**：一颗被对方围空分支（`:2221`/`:2222`，两者
        都**无条件覆盖**）改写过的己方子，在官方输出里是**对方颜色**；若它落在
        双活块里且 `basicArea != colors`，`keepStones`（`:1931`，判据是
        `basicArea == colors`）也不会补回 ⇒ 该点为中立。这两种情形由
        `tests/test_v7_area_official.py` 钉。
    """
    own = area_ownership_map(_ENCLOSURES)[0]
    assert own[7, 7] == 1, '黑子应归黑'
    assert own[4, 5] == -1, '白子应归白'
    assert own[2, 2] == 0, '空的未围点应中立'


# --------------------------------------------------------------------------- #
# 结构性质：两条从官方源码推出来的、能**区分新旧口径**的不变量
# --------------------------------------------------------------------------- #
def test_seki_tax_never_adds_area():
    """**双活过滤只会抹掉 area，不会新增**（`TAX_SEKI`/`TAX_ALL` ⊆ `TAX_NONE`）。

    为什么这条能区分新旧口径：Tromp-Taylor 没有 tax 这个维度，所以旧实现根本
    写不出这条断言。而它挡住的是一整类很隐蔽的错 —— 双活过滤一旦被写成
    「只清 atari 的子、不清接触 dame 的块」（漏掉 `board.cpp:2272-2275` 那个
    触发），会**凭空多出**大量中立点，row_exact 掉下来但 cell_agree 还很高。
    """
    rng = np.random.default_rng(20261003)
    for _ in range(40):
        x = rng.random((9, 9))
        b = np.where(x < 0.22, 1, np.where(x < 0.44, -1, 0)).astype(np.int8)
        for sl in (False, True):
            none = area_ownership_map(b[None], is_multi_stone_suicide_legal=sl,
                                      tax_rule=TAX_NONE)[0]
            for tax in (TAX_SEKI, TAX_ALL):
                got = area_ownership_map(b[None],
                                         is_multi_stone_suicide_legal=sl,
                                         tax_rule=tax)[0]
                assert not ((got != 0) & (none == 0)).any(), (
                    f'tax={tax} 归出了 TAX_NONE 下不存在的点 ⇒ '
                    f'双活过滤的方向反了')


def test_tax_seki_and_tax_all_share_the_area_planes():
    """**`TAX_SEKI` 与 `TAX_ALL` 的 ch18/ch19 必须逐位相同**。

    依据官方源码：`calculateIndependentLifeArea`（`board.cpp:1876-1937`）**根本
    不读 `taxRule`** —— 它只有 `keepTerritories` / `keepStones` /
    `excludeTerritoryAdjacentToAtari` / `isMultiStoneSuicideLegal` 四个参数。
    `taxRule` 在 `nninputs.cpp` 里只多算一个**全局**标量
    `groupTaxAdjustmentForPla`（`:2436-2437`），**不进任何空间通道**。
    ⇒ 谁要是把 tax 接进空间路径（一个很自然、看起来很合理的"改进"），这条立刻红。
    """
    rng = np.random.default_rng(20261004)
    for _ in range(40):
        x = rng.random((9, 9))
        b = np.where(x < 0.22, 1, np.where(x < 0.44, -1, 0)).astype(np.int8)
        for sl in (False, True):
            seki = area_ownership_map(b[None], is_multi_stone_suicide_legal=sl,
                                      tax_rule=TAX_SEKI)[0]
            alls = area_ownership_map(b[None], is_multi_stone_suicide_legal=sl,
                                      tax_rule=TAX_ALL)[0]
            assert np.array_equal(seki, alls), (
                f'TAX_SEKI 与 TAX_ALL 的 ch18/19 必须相同（taxRule 不进空间通道，'
                f'nninputs.cpp:2436-2437 只算一个全局标量）；'
                f'{int((seki != alls).sum())} 格不同')


def _ref_arr(b, suicide_legal=False, tax_rule=TAX_NONE):
    """`_ref_area` 的 `(n,n)` int8 包装（便于与被测实现直接 `array_equal`）。"""
    n = len(b)
    out = np.zeros((n, n), dtype=np.int8)
    for (r, c), v in _ref_area(b, suicide_legal, tax_rule).items():
        out[r, c] = v
    return out


def test_ownership_matches_independent_reference():
    """**逐格**等价于一份独立写的官方算法参考实现（随机盘面 30 个 × 3 种 tax）。

    ⚠ **旧版这条用的是 Tromp-Taylor 参考实现**（`_ref_ownership`）⇒ 它断言的是
      旧口径，与官方**相反**。旧实现在真实 stdata 上与官方逐行 0 行相同。
      现在的 `_ref_area` 是 `board.cpp:1853-2327` 的纯 Python 抄写，与被测的
      向量化实现零共享代码 —— 这才是「两份独立实现」的对拍。
    """
    rng = np.random.default_rng(11)
    for trial in range(30):
        x = rng.random((9, 9))
        b = np.where(x < 0.24, 1, np.where(x < 0.48, -1, 0)).astype(np.int8)
        sl = bool(rng.integers(0, 2))
        for tax in (TAX_NONE, TAX_SEKI, TAX_ALL):
            want = _ref_arr(b.astype(int), sl, tax)
            got = area_ownership_map(b[None], is_multi_stone_suicide_legal=sl,
                                     tax_rule=tax)[0]
            if not np.array_equal(got, want):
                rr, cc = np.argwhere(got != want)[0]
                raise AssertionError(
                    f'第 {trial} 个随机盘面（tax={tax}, suicide={sl}）：'
                    f'({rr},{cc}) 我们={got[rr, cc]} 参考={want[rr, cc]}'
                    f'（共 {int((got != want).sum())} 格不同）')


def test_ownership_full_board_and_one_stone_per_colour():
    """两个退化形状：满盘（无空点 ⇒ 区域/链标注全空 ⇒ 走短路分支），
    以及「黑白各一颗子」（整块盘面是一大片空区、边界只有两种颜色 ⇒ 全中立）。

    两者都不是随机盘面容易覆盖到的分支。
    ⚠ **旧 docstring 里「按 Tromp-Taylor 全部归黑（sum == 361）」那句已作废** ——
      那是旧口径的理由。官方语义下单颗黑子那一大片空区仍然归黑，但走的是
      `:2222`（`safeBigTerritories && !containsOpp && !borders && atLeastOnePla`），
      前提是那条链**没被 Benson 判死**；一颗孤子只有 1 个 vital 区域 ⇒ 判死
      ⇒ `bordersNonPassAlive` ⇒ `:2222` 不命中，只剩 `:2233` 的 C_EMPTY 兜底。
      结论与旧 docstring 一致纯属巧合，别拿它当依据。
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
