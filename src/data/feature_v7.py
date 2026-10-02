# -*- coding: utf-8 -*-
"""KataGo 官方 V7 空间通道的**纯函数**部分（B2 三个无规则条件化通道 + B3 的 area）。

语义对照 `cpp/neuralnet/nninputs.cpp::fillRowV7`，逐通道表见
`docs/superpowers/specs/2026-10-01-katago-nbt-tf-design.md` §2.2（空间 22 通道）
与 §2.5（ch18/19 的规则条件化）。

**为什么单独一个模块、而不加进 `go_rules.py`**（spec §5.2 已定）：`go_rules.py` 的
`GoBoard.feature_planes` / `feature_planes_batched` 是 17 通道旧布局，spec §7.1
钉死「一字不改」。本模块是**纯新增**，只**只读地 import** 那里已验证的三块内核。

----
**口径来源：只有一份实现，不重写。**

  · 连通块标注 —— `scipy.ndimage.label` + `go_rules._STRUCT3`。该结构元的第 0 轴
    是单位阵、正交十字在第 1/2 轴，所以 `(B,n,n)` 输入得到的是**逐样本独立**的
    4 邻域连通块（不会把 batch 里相邻两局的盘面连成一块）。
  · 块气去重 —— `go_rules._distinct_liberty_counts`，O(B·n²)：把
    「(空点 q, 邻子 p, 块号 g)」按 4 个方向拆成 4 张表后按**块号**去重，
    **每个 (q, g) 关联只数一次**。

⚠ **不要在本模块里另写一份块气算法。** 17 通道的 ch10/11/14/15（「己/敌 × 气 1/2」）
与 V7 的 ch3/4/5（「不分色的 气 1/2/3」）是**同一个量的不同切片**，共享实现才能
保证两条路径逐位相同。仓库里已经吃过一次「同一口径写两遍」的亏：
`go_rules.py:170-177` 记着入射计数口径的一次真 bug（U 形块被多算 ⇒ 通道 10/11 与
14/15 **两个 bucket 同时漏**）。`tests/test_v7_planes.py` 里有一条测试就是把 V7 的
ch3/4 与 17 通道的 ch10/11|ch14/15 逐位对拍，把这个契约钉住。

----
**本模块不含**（留给后续任务，避免本轮就把邻行 gather 的语义猜掉）：

  · 邻行 gather（`boards[i−1]` / `boards[i−2]`，spec §5.4）；
  · `iterLadders`（ch14–17）—— 另一个 agent 的 `src/data/feature_v7_ladders.py`。
    本模块只提供它需要的**历史门控回退**语义 `history_gated`；
  · `passWouldEndPhase`（全局 ch14）、komi 奇偶三角波（全局 ch18）；
  · 把这些拼成 22 通道张量的装配器。
"""

from typing import NamedTuple

import numpy as np
from scipy.ndimage import binary_dilation as _ndi_binary_dilation
from scipy.ndimage import label as _ndi_label

from src.game.go_rules import _STRUCT3, _distinct_liberty_counts

__all__ = [
    'LADDER_CH_BASE',
    'SCORING_AREA',
    'SCORING_TERRITORY',
    'TAX_NONE',
    'TAX_SEKI',
    'TAX_ALL',
    'HISTORY_MOVES',
    'history_five',
    'liberties_123',
    'history_gated',
    'area_ownership_map',
    'calculate_area',
]

#: ladder 三通道块的基通道号（ch14=当前盘 / ch15=前一手 / ch16=前二手，见 spec §2.2）。
LADDER_CH_BASE = 14

#: `calculate_area` 一次覆盖的历史手数（ch9..13）。
HISTORY_MOVES = 5

# ---- `rules_flags` 位定义（spec §2.5 的三个分支）------------------------------
# ⚠ **这不是 KataGo 的 `Rules` 对象**，是把「计分方式 × 税收方式」压进一个 int，
#   供本仓的预取器从局表 `g_rules`（SGF `RU`）直接喂。
#   bit0 = 计分方式：0=AREA / 1=TERRITORY
#   bit1..2 = 税收方式：0=NONE / 1=SEKI / 2=ALL
SCORING_AREA = 0
SCORING_TERRITORY = 1
TAX_NONE = 0
TAX_SEKI = 1
TAX_ALL = 2


def _as_batch_boards(boards):
    """校验并返回 `(B,n,n)` int8 盘面数组。"""
    b = np.asarray(boards)
    if b.ndim != 3:
        raise ValueError(f'boards 形状不符：{b.shape}，期望 (B, n, n)')
    return b


def _as_batch_moves(moves, B, name):
    """校验并返回 `(B, k)` 的扁平坐标列（`-1` = pass 或历史不足）。"""
    m = np.asarray(moves)
    if m.ndim == 1:
        m = m.reshape(B, -1)
    if m.ndim != 2 or m.shape[0] != B:
        raise ValueError(f'{name} 形状不符：{m.shape}，期望 ({B}, k)')
    return m


# --------------------------------------------------------------------------- #
# B2-1 · ch3 / ch4 / ch5：棋子的 1 / 2 / 3 气
# --------------------------------------------------------------------------- #
def liberties_123(boards, to_play=None):
    """返回 `(B, 3, n, n)` bool —— ch3/ch4/ch5 = **不分色**的 1/2/3 气掩码（spec §2.2）。

    每一格的值是「该点所在**整块**的去重气数」，即同块的两颗子夹住同一个空点时，
    那一点对该块只算 1 气 —— 与 `GoBoard._group_liberty_count`（`set((r,c))`）
    及 `_distinct_liberty_counts` **同一口径**。

    ---- 为什么是「每块只数一次」以及怎么保证的 ----
    按点重算气数是 O(k²)：一块 k 颗子，每颗子各扫一遍整块 ⇒ k·k 次邻接判定。
    本实现走 `scipy.ndimage.label` + `_distinct_liberty_counts`：每块的气由
    **空点侧**收集（每个 (空点, 邻块) 关联贡献 1），于是代价是
    O(该块的邻接数)，与块的大小 k **线性**，k 颗子的块和 k 个单子块花一样的钱。
    `tests/test_v7_planes.py` 用「调用次数 + 一个很宽松的耗时上界」把这条不变式
    钉住（理由与数字见该测试的 docstring）。

    ---- 向量化方案：为什么选 `scipy.ndimage.label` 而不是自己写 union-find ----
      · **仓库已有这个依赖**：`go_rules.py:66` 就是 `from scipy.ndimage import label`，
        并已有现成的 `_STRUCT3` 与线程池 `_label_pool`。选它 ⇒ 连通性（4 邻域、
        逐样本独立）与 17 通道路径**逐位同源**，不需要第二套「什么算相邻」的约定。
      · `label` 释放 GIL，而 union-find 在纯 numpy/Python 里要么退化成 Python 循环
        （拿不到批量加速），要么自己写一套带路径压缩的迭代版本（代码量与出错面
        都远大于收益）。
      · ⚠ **未登记依赖**：`requirements.txt` 里**没有** `scipy`，但 `go_rules.py`
        已经在 import 它。本模块沿用现状（不新增依赖类别），但这是一个应该补的
        记账缺口 —— 见任务报告。

    Args:
        boards: `(B, n, n)` int8，取值 −1/0/1（黑 −1 的约定相反：本仓是 −1=白、
            +1=黑，`go_rules.GoBoard.board` 同）。
        to_play: `(B,)` ±1。**不参与取值** —— ch3/4/5 不分色，见下方警告。

    ⚠ **`to_play` 为什么是摆设**：V7 的 ch3/4/5 是**不分色**的气桶（一组三格，
    不是六格），本仓的 npz 侧只有 `boards` 就够算。保留这个形参是为了和
    `feature_planes_batched` 的调用点同形；**它既不读也不校验**。
    真正「分色」的气桶在 17 通道里是 ch10/11（气 1）与 ch14/15（气 2），
    由 `to_play` 定位 —— 那四条通道不在本函数的口径里。
    本函数与它们的等价关系（`ch3 == ch10|ch11`、`ch4 == ch14|ch15`）由
    `tests/test_v7_planes.py::test_liberties_123_match_17ch_union` 钉住。
    """
    b = _as_batch_boards(boards)
    B, n, _ = b.shape
    empty = b == 0
    out = np.zeros((B, 3, n, n), dtype=bool)

    for color in (1, -1):
        mask = b == color
        if not mask.any():
            continue
        labelled, num = _ndi_label(mask, structure=_STRUCT3)
        if num == 0:
            continue
        lib_counts = _distinct_liberty_counts(labelled, num, empty)
        # 物化一次、比较三次：ch4/ch5 相对 ch3 只多两次 `==`，**不多一遍标注**。
        per = lib_counts[labelled]
        for k in range(3):
            out[:, k] |= (per == k + 1) & mask
    return out


# --------------------------------------------------------------------------- #
# B2-2 · ch9..ch13：过去 5 手的落点
# --------------------------------------------------------------------------- #
def history_five(boards, my_hist, op_hist, to_play=None):
    """返回 `(B, 5, n, n)` bool —— ch9..ch13 = **过去 5 手落点**（spec §2.2）。

    顺序是 **`opp, pla, opp, pla, opp`**（自 `nextPlayer` 视角，即 `to_play` 是
    下一手执子方）：因为上一手永远是对手落的，最新的一格属于 opp。

    ---- 交错规则 ----
    本仓的 `my_hist` / `op_hist` 是**相对 to_play** 的（`my` = to_play 这一方、
    `op` = 对手），且 **index 0 是最近一手**（见 `go_rules.py:2382` 与
    `feature_planes_batched` 的通道 1-3 / 5-7）。两边交替 ⇒：

    ==========  ==============  ==============
    通道        物理手序        来源
    ==========  ==============  ==============
    ch9         最近一手(opp)   `op_hist[:, 0]`
    ch10        第二手(pla)     `my_hist[:, 0]`
    ch11        第三手(opp)     `op_hist[:, 1]`
    ch12        第四手(pla)     `my_hist[:, 1]`
    ch13        第五手(opp)     `op_hist[:, 2]`
    ==========  ==============  ==============

    ---- 三条容易写错的语义（都照 spec §2.2 / 官方语义）----
      1. **pass 不占通道** —— 但它**占住自己那个物理手序位**。即
         `my_hist = [-1, a, b]`、`op_hist = [c, -1, d]` 时：ch9=c、ch10 空、
         ch11 空、ch12=a、ch13=d。**更早的手不会往前挤**。来源是 `-1` 同时表示
         「pass」与「历史不足」，而这两种情况在落点通道上的**输出完全一样**（空），
         所以不必、也不能在这里区分。
      2. **历史不足时更早的不占位** —— 不是补 0、也不是复制最近的。缺的手位留空。
      3. 一手只能落一个点，所以同一格最多被点亮一次；两个玩家历史里出现重复坐标
         的输入（脏数据）按「后写覆盖」处理，值仍是 True，不报错。

    Args:
        boards: `(B, n, n)` int8。**只用来取边长**，不参与取值。
        my_hist / op_hist: `(B, k)` 整数（通常 int16），扁平坐标 `r*n+c`，
            `-1` 表示 pass / 无此手。
        to_play: `(B,)` ±1。**不参与取值** —— `my_hist`/`op_hist` 本来就是
            to_play 相对的（与 `feature_planes_batched` 的通道 1-3/5-7 同口径），
            再乘一次 `to_play` 会把两者翻过来。保留形参只为调用点同形。
    """
    b = _as_batch_boards(boards)
    B, n, _ = b.shape
    my = _as_batch_moves(my_hist, B, 'my_hist')
    op = _as_batch_moves(op_hist, B, 'op_hist')
    if my.shape[1] < 3 or op.shape[1] < 3:
        raise ValueError(f'需要每人最近 3 手（5 格交错用 3+2），'
                         f'实得 my={my.shape[1]} / op={op.shape[1]}')

    out = np.zeros((B, HISTORY_MOVES, n, n), dtype=bool)
    upper = n * n
    for ch, (src, slot) in enumerate(((op, 0), (my, 0), (op, 1), (my, 1), (op, 2))):
        col = src[:, slot].astype(np.int64, copy=False)
        valid = (col >= 0) & (col < upper)
        if not valid.any():
            continue
        idx = np.flatnonzero(valid)
        r, c = np.divmod(col[idx], n)
        out[idx, ch, r, c] = True
    return out


# --------------------------------------------------------------------------- #
# B2-3 · 历史门控（ladder 通道 ch15/ch16 的取盘面规则）
# --------------------------------------------------------------------------- #
class GatedBoards(NamedTuple):
    """`history_gated` 的返回值：ladder 块要跑的三块盘面里，后两块。"""

    prev: np.ndarray        #: ch15 的盘面
    prev_prev: np.ndarray   #: ch16 的盘面


def history_gated(prev_board, prev_prev_board, boards, offset=LADDER_CH_BASE):
    """历史门控：**历史不足时回退复制，不是置 0**（spec §2.2「易错点」那段）。

    官方语义（原文照录）::

        prevBoard     = (numTurnsOfHistoryIncluded < 1) ? board     : hist.getRecentBoard(1);
        prevPrevBoard = (numTurnsOfHistoryIncluded < 2) ? prevBoard : hist.getRecentBoard(2);

    ⇒ history=0 时 **ch15 == ch14**、**ch16 == ch15**；
      history=1 时 **ch16 == ch15**。
    ⚠ 关键在**第二行回退到 `prevBoard` 而不是回退到 `board`**：history=1 时
      prevPrev 复制的是 prev，不是当前盘。把第二行写成「< 2 ? board : ...」会在
      history=1 上给出错误（当前盘而非前一手盘）的 ch16 —— 这是这条规则唯一的坑。

    本函数**只做门控**，不跑 ladder：调用方拿 `boards` 跑 ch14、拿返回的两个跑
    ch15/ch16，算法是同一份。

    Args:
        boards: `(B,n,n)` int8，**当前盘**（ch14 用；也是门控回退的兜底）。
        prev_board: `(B,n,n)` 或 `None`（= 邻行 gather 失败/这一行没有上一手）。
        prev_prev_board: `(B,n,n)` 或 `None`。
        offset: ladder 块的基通道号，必须是 `LADDER_CH_BASE`(14)。
            ⚠ **它不参与取值**（回退只看两个入参是不是 `None`），只是把「这三块
            盘面对应 ch14/15/16」这条约定钉在签名上，让调用点自解释；
            传别的值直接报错，免得出现「算对了却写进了错的通道」这种静默错。

    Returns:
        `GatedBoards(prev, prev_prev)`，两个 `(B,n,n)` int8。可以当元组解包。
    """
    if offset != LADDER_CH_BASE:
        raise ValueError(
            f'offset={offset} 不是 ladder 块的基通道号（应为 {LADDER_CH_BASE}）：'
            f'ch{offset}/ch{offset + 1}/ch{offset + 2} 这块在 V7 里是 ladder 三通道，'
            f'没有第二个 ladder 块。门控逻辑本身与 offset 无关，写错通道才是真 bug。')

    cur = _as_batch_boards(boards)
    shape = cur.shape

    def _resolve(board, fallback, name):
        if board is None:
            return fallback
        arr = np.asarray(board)
        if arr.shape != shape:
            raise ValueError(f'{name} 形状不符：{arr.shape}，期望 {shape}')
        return arr

    prev = _resolve(prev_board, cur, 'prev_board')
    # ⚠ 第二级回退的兜底是 `prev`（**不是** `cur`）—— 见 docstring 的坑。
    prev_prev = _resolve(prev_prev_board, prev, 'prev_prev_board')
    return GatedBoards(prev, prev_prev)


# --------------------------------------------------------------------------- #
# B3 · ch18 / ch19：calculateArea
# --------------------------------------------------------------------------- #
def area_ownership_map(boards):
    """返回 `(B, n, n)` int8 的**逐点归属图**（Tromp-Taylor 区域计分），取值 −1/0/+1。

    规则（与 `GoBoard.score()` **逐格同口径**）：
      · 有子的点归该子自己的颜色；
      · 空点按 4 邻域连通成**区域**，区域若**只**与一种颜色相邻则整片归该色，
        与两色相邻（或与谁都不相邻）则为中立 0。

    ---- 与 `GoBoard.score()`（`go_rules.py:2261`）的口径等价性 ----
    `score()` 逐点 flood fill 空区域、按 `len(border_colors) == 1` 归色，然后
    `黑子 + 黑围空 − 白子 − 白围空 − komi`。本函数做的事**逐点同构**，
    区别只在两点实现方式：
      · `score()` 是 Python 双层循环 + 栈式 flood fill，本函数用
        `scipy.ndimage.label`（4 邻域、同 `_STRUCT3`）批量标号 —— 连通性一致；
      · `score()` 用 `set()` 收集 border colors，本函数用**两次 4 邻域膨胀**
        （`binary_dilation(black)` / `binary_dilation(white)`）问「这个区域有没有
        黑色邻子 / 白色邻子」—— 这是 `len(border_colors) ∈ {0,1,2}` 的等价问法，
        且比收集集合更便宜。
    ⇒ 不变式：`ownership_map(board).sum() == GoBoard.score() + komi`。
      由 `tests/test_v7_area.py::test_ownership_sum_agrees_with_go_board_score`
      在随机盘面上逐位对拍。

    ⚠ **没有死子判定 —— 这是有意保留的**（spec §5.1 的白捡二 + 对齐口径优先）。
    `score()` 的 docstring 自陈「没有死子判定」，所以一块**没被提掉**的对方死子
    会让它周围的空点变成「两色共邻」的中立区域，同族的死子点自己仍算它的颜色。
    对局双方应在 pass 认输前提掉对方的死子。

    ---- 与官方 stdata 的 ch18/19 的实测偏差（**不是 bug，是已知的口径差**）----
    用 `katago/stdata` 的 `zzb28c512` 批（第一个含 19×19 行的 npz，**1254 行**）
    对拍：把官方 ch1/ch2 还原成盘面喂给本函数，再与官方 ch18/19 逐点比 ——
    **没有一行完全相同**。偏差的方向是**我们比官方多**：

    ==============================  ========  =====================================
    项                                数值      说明
    ==============================  ========  =====================================
    ch18 官方总点数                    73,298   其中 **4,148** 是空点、其余是子
    ch18 我们总点数                    95,113   其中 **6,089** 是空点
    「我们多出来」的点                 21,896   19,913 个**子** + 1,983 个空点
    「官方多出来」的点                    81
    受影响的行                        295/1254   且缺失的子**不集中在 1 气的块上**
    ==============================  ========  =====================================

    最小的例子：一个被黑子四面围住的单空点（形如 `.O.O. / O.O.O / .O.O.` 的中心）
    按 Tromp-Taylor 归黑，**官方给中立**。合理猜测：官方在算 area 之前**先提掉了
    死子**（于是围住死子的那片空与外面的空连成一片、重新按边界颜色分类），
    而 `GoBoard.score()` 不做这件事。

    ⚠ **本轮刻意不去补死子判定**（brief 明确要求）：那要猜「官方用什么判据判死」
    （块气？局部搜索？`Board` 内部字段？），猜错会得到一个**既不等于 `score()`、
    也不等于官方**的第三种口径，而这类偏差是静默的（不会抛异常，只表现为训练分布
    偏移）。先把「与 `score()` 逐点同构 + `sum == score() + komi`」这个可证的
    不变式钉住（`tests/test_v7_area.py`），死子判定作为**单独一件事**带着判据来做。
    ⇒ 因此 **stdata 不能直接当 ch18/19 的对拍 oracle**，只有上面那两条
      （ch3/4/5 与 ch9-13）可以，见 `tests/test_v7_planes.py`。
    """
    b = _as_batch_boards(boards)
    B, n, _ = b.shape

    empty = b == 0
    labelled, num = _ndi_label(empty, structure=_STRUCT3)

    if num:
        region = labelled > 0
        reg_id = labelled[region]
        # 「区域里有没有黑色/白色邻子」= 该区域任一点的 4 邻域里有黑/白子。
        # `binary_dilation` 的 `origin` 默认居中，`_STRUCT3` 的第 0 轴是单位阵，
        # 所以每个 batch 元素各自膨胀、不会跨样本。
        near_black = _ndi_binary_dilation(b == 1, structure=_STRUCT3)[region]
        near_white = _ndi_binary_dilation(b == -1, structure=_STRUCT3)[region]

        touches_black = np.zeros(num + 1, dtype=bool)
        touches_white = np.zeros(num + 1, dtype=bool)
        touches_black[reg_id[near_black]] = True
        touches_white[reg_id[near_white]] = True
        touches_black[0] = False
        touches_white[0] = False

        # 只有「恰好一种颜色」才归属 —— 这就是 `score()` 的 `len(border_colors)==1`。
        owner_of_region = np.zeros(num + 1, dtype=np.int8)
        owner_of_region[touches_black & ~touches_white] = 1
        owner_of_region[touches_white & ~touches_black] = -1
        region_owner = owner_of_region[labelled]
    else:
        # 满盘无空点：没有区域，只有子。
        region_owner = np.zeros((B, n, n), dtype=np.int8)

    own = np.where(b > 0, np.int8(1), np.where(b < 0, np.int8(-1), region_owner))
    return own.astype(np.int8, copy=False)


def calculate_area(boards, to_play, rules_flags=0):
    """返回 `(B, 2, n, n)` int8 —— ch18/ch19 = 当前区域归属（spec §2.2 / §2.5）。

    通道顺序 `[pla 归属, opp 归属]`，两格取值都 ∈ {−1, 0, +1}：
    **+1 = 该点归 pla（`to_play` 这一方）、−1 = 归 opp、0 = 中立**。
    两格是**同一张归属图的两个视角**：ch19 == −ch18（opp 视角把符号翻过来）。
    ⚠ 这样安排的理由是让**「0 = 中立」在两格里都成立** —— 下游二值化
      （`> 0` / `< 0`）两格对称，且任意一格单独用都读得出完整的三态。
      官方是两条 bool 平面（pla 有 / opp 有），二值化后逐位等价。

    ---- 规则条件化（spec §2.5，照抄官方的 if/else 顺序）----
    ::

        if (SCORING_AREA && TAX_NONE)            calculateArea(...)                     # 常规
        elif (SCORING_AREA && (SEKI|ALL))       calculateIndependentLifeArea(keepStones=true)
        elif (SCORING_TERRITORY)                # 仅 encorePhase >= 2 才置位 ⇒ 恒 0

    ⚠ **顺序要紧**：官方先判 AREA。所以 `SCORING_TERRITORY` + `TAX_SEKI` 落在
      第三分支（**返回全 0**）而不是第二分支；本实现照抄这个顺序。

    ⚠ **TERRITORY ⇒ 全 0 —— 这是对齐官方，不是缺陷。** 官方的 territory 分支被
      `encorePhase >= 2` 二次条件包着，而本仓无 encore 机制（spec §2.6 D2），
      ⇒ 正常阶段 territory 计分的对局 ch18/19 恒 0。SGF 缺 `RU` 时默认 AREA，
      所以绝大多数对局走的是第一个分支。

    Args:
        boards: `(B,n,n)` int8（±1/0）。
        to_play: `(B,)` ±1。
        rules_flags: int。bit0 = 计分方式（0=AREA / 1=TERRITORY），
            bit1..2 = 税收方式（0=NONE / 1=SEKI / 2=ALL）。默认 0 = AREA+NONE。

    Raises:
        NotImplementedError: `SCORING_AREA` 且 `TAX_SEKI` / `TAX_ALL`。
    """
    b = _as_batch_boards(boards)
    B, n, _ = b.shape
    scoring = rules_flags & 0x1
    tax = (rules_flags >> 1) & 0x3

    out = np.zeros((B, 2, n, n), dtype=np.int8)
    if scoring == SCORING_TERRITORY:
        return out

    if tax != TAX_NONE:
        tax_name = {TAX_SEKI: 'TAX_SEKI', TAX_ALL: 'TAX_ALL'}.get(tax, f'tax={tax}')
        raise NotImplementedError(
            f'calculate_area 暂未实现 {tax_name} 分支（官方走 '
            f'calculateIndependentLifeArea(keepStones=true)）。\n'
            f'官方语义：只统计「**无条件活**」的区域 —— 一块子连同它的全部气构成的\n'
            f'区域若被对方子完全封住（对外不再有任何共享气），整片算该方属地；\n'
            f'被对方分断/共享气的块不算。keepStones=true 表示死子也留在图里。\n'
            f'⚠ 这里**故意抛错而不是返回全 0**：全 0 与「TERRITORY 恒 0」在张量上\n'
            f'  逐位相同，会被当成「无归属」静默污染训练。tax=SEKI/ALL 的对局应当\n'
            f'  在规则条件化那一步被显式拦下（或先补齐本分支）。')

    own = area_ownership_map(b)                                # 绝对盘面色
    pla = own * np.asarray(to_play).reshape(B, 1, 1).astype(np.int8)
    out[:, 0] = pla
    out[:, 1] = -pla
    return out
