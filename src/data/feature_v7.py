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

  · 邻行 gather（`boards[i−1]` / `boards[i−2]`，spec §5.4）——
    **已由 `src/data/feature_v7_gather.py` 实现**（`gather_neighbors`，带
    mmap / 越界 / 跨局守卫）。它与本模块分开是因为它带 I/O 与数据集依赖，
    而本模块是纯 numpy 函数层。

**本模块含**（B1 / B5 装配层，追加在文件末尾）：

  · `iterLadders`（ch14–17）的调用方 —— `spatial_channels_v7`；
  · 19 维全局特征 —— `global_features_v7`（spec §2.3）；
  · 历史门控的**唯一**解析点 —— `resolve_ladder_boards`。

⚠ **`history_gated` 与 `ladder_channels` 内部的那份门控不是两份口径**：
  两者是同一条官方原文的两种触发方式（`history_gated` 供只想拿盘面的调用方
  用；`ladder_channels` 自己那份是为了保住 `pb is b` / `p2b is pb` 的快路径）。
  装配层**只经 `resolve_ladder_boards` 解析一次**，绝不第三次推导 ——
  `feature_v7_ladders.py` 曾从同一个事实推导两次、第二次推出了错的 ch16。
"""

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
from scipy.ndimage import binary_dilation as _ndi_binary_dilation
from scipy.ndimage import label as _ndi_label

from src.data.feature_v7_ladders import BOARD_SIZE, ladder_channels
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
    # ---- B1 / B5 装配层 ----
    'SPATIAL_CHANNELS',
    'GLOBAL_CHANNELS',
    'SPATIAL_DTYPE',
    'GLOBAL_DTYPE',
    'KOMI_SCALE',
    'KOMI_CLIP_MARGIN',
    'DEFAULT_RULES_FLAGS',
    'FLAG_SCORING_TERRITORY',
    'FLAG_TAX_MASK',
    'FLAG_KO_MASK',
    'FLAG_MULTISTONE_SUICIDE',
    'FLAG_HAS_BUTTON',
    'KO_SIMPLE',
    'KO_POSITIONAL',
    'KO_SITUATIONAL',
    'rules_flags_from',
    'rules_flags_from_sidecar',
    'LadderBoards',
    'resolve_ladder_boards',
    'spatial_channels_v7',
    'GameRow',
    'self_komi',
    'komi_parity_wave',
    'pass_would_end_phase',
    'global_features_v7',
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
    for ch in range(HISTORY_MOVES):
        # ⚠ 交错顺序取自 `_HISTORY_SLOTS`（模块级唯一定义），**不是在这里重写一遍**
        #   —— 全局 ch0..4（pass 标志）要用同一份顺序。见该常量的注释。
        col = _history_move(my, op, ch)
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
# --------------------------------------------------------------------------- #
# B1 / B5 · 装配层：22 通道分发 + 19 维全局特征
# --------------------------------------------------------------------------- #

#: V7 的空间通道数 / 全局通道数（spec §1.3）。
SPATIAL_CHANNELS = 22
GLOBAL_CHANNELS = 19

#: 装配层输出的 dtype。与 ``GoBoard.feature_planes`` / ``feature_planes_batched``
#: 逐位对齐（`src/game/go_rules.py` 里那句「与 feature_planes_batched 同一个
#: dtype（fp16）」：两路必须一致，否则 RL/推理走单图与训练走批量会拿到不同
#: 精度的 planes）。空间通道取值域只有 {0,1}，fp16 逐位无损。
SPATIAL_DTYPE = np.float16
GLOBAL_DTYPE = np.float16

#: `currentSelfKomi` 的归一化除数。⚠ **V7 是 20，V3/V4 是 15** —— 实测官方
#: `globalInputNC[:,5]` 上 komi=7.5 的行恰好是 0.375 = 7.5/20（若是 /15 会是
#: 0.5）。这属于「同一份语义、随模型版本换常数」的一类，只写一个数会静默错。
KOMI_SCALE = 20.0

#: `currentSelfKomi` 的 clip 半径：官方原文 `if(selfKomi > bArea+1.0f) …`
#: （clip 的是 **selfKomi 本身**，不是归一化后的值）。
KOMI_CLIP_MARGIN = 1.0

# --------------------------------------------------------------------------- #
# `rules_flags` 位布局 —— **在 `calculate_area` 既有布局上扩展，不另起一套**
# --------------------------------------------------------------------------- #
# ⚠ 为什么不能给全局特征另定一套位：``calculate_area``（本文件，已提交、已被
#   ``tests/test_v7_area.py`` 钉住）读的是 **bit0 = 计分 / bit1..2 = 税收**。
#   两套布局并存就是「同一口径写两遍」，而本仓已经吃过这个亏 ——
#   ``go_rules.py:170-177`` 记着入射计数口径被写了两遍、导致 U 形块的
#   ch10/11 与 ch14/15 **两个 bucket 同时漏**；``feature_v7_ladders.py`` 也记着
#   历史门控被解析两次、ch16 抄错了通道。
#   ⇒ 本布局是既有布局的**严格超集**：`calculate_area` 一行都不用改。
# ⚠ **掩码就是「被覆盖的那几位本身」**（不是「左移后的结果」）：取出时统一写
#   ``(fl & MASK) >> 位移``。把掩码写错一位就会与 ``rules_flags_from`` 的打包
#   错开 —— 首版就栽在这儿两次（``0b110 << 1`` / ``0b110 << 3`` 都是 3 位宽，
#   实际要的是 2 位宽的 ``0b11 << 位移``），症状是「改 ko 规则会连 ch10 一起
#   改掉」/「改 ko 规则完全没反应」。
#   `tests/test_v7_globals.py::test_rules_flags_from_is_the_inverse_of_the_shared_bit_layout`
#   与 `::test_each_ko_rule_flag_moves_only_its_own_channels` 就是拿它抓这两条的。
FLAG_SCORING_TERRITORY = 1 << 0      # 0 = SCORING_AREA / 1 = SCORING_TERRITORY
FLAG_TAX_MASK = 0b11 << 1             # bit1..2：0=NONE / 1=SEKI / 2=ALL
FLAG_KO_MASK = 0b11 << 3              # bit3..4：0=SIMPLE / 1=POSITIONAL / 2=SITUATIONAL
FLAG_MULTISTONE_SUICIDE = 1 << 5     # multiStoneSuicideLegal
FLAG_HAS_BUTTON = 1 << 6             # hasButton

#: `FLAG_KO_MASK` 的三个取值（官方 `Rules::koRule` 的三态）。
KO_SIMPLE = 0
KO_POSITIONAL = 1
KO_SITUATIONAL = 2

#: **简单局默认**（= `rules_flags == 0`）：AREA + TAX_NONE + KO_SIMPLE +
#: 单子禁着（multiStoneSuicideLegal=false）+ 无 button。
#: 这正是 SGF 里 **55% 的局没有 `RU`** 时唯一站得住的取值（spec §5.3.2）：
#: `tax` / `ko` 规则 / `suicide` / `button` 四者在 `RU` 字符串里，但缺 `RU` 时
#: 无从得知 ⇒ 一律取默认值。这四条与「简单局」的假设自洽。
DEFAULT_RULES_FLAGS = 0


def rules_flags_from(scoring=SCORING_AREA, tax=TAX_NONE, ko=KO_SIMPLE,
                     multi_stone_suicide=False, has_button=False):
    """按字段名打包成 ``rules_flags`` int（:data:`DEFAULT_RULES_FLAGS` 的逆）。

    优先用这个而不是在调用点手写位运算 —— 位布局只有这一处定义。
    """
    return (FLAG_SCORING_TERRITORY if scoring == SCORING_TERRITORY else 0) \
        | ((int(tax) & 0b11) << 1) \
        | ((int(ko) & 0b11) << 3) \
        | (FLAG_MULTISTONE_SUICIDE if multi_stone_suicide else 0) \
        | (FLAG_HAS_BUTTON if has_button else 0)


def rules_flags_from_sidecar(g_rules):
    """sidecar ``games.npz`` 的 ``g_rules``（spec §5.3 的打包）→ 本布局。

    ⚠ **spec §5.3 的打包与本布局不同，而且是「有意压窄」的**：
    那边是 ``bit0=计分 / bit1=tax / bit2=ko / bit3=suicide / bit4=button`` ——
    ``tax`` 只给了 **1 位**，表达不了 ``TAX_SEKI`` 与 ``TAX_ALL`` 的区别。

    本函数做**显式有损映射**并写明丢了什么：
      · ``tax`` 的 1 位 ``0`` ⇒ ``TAX_NONE``；``1`` ⇒ **按 ``TAX_SEKI`` 解读**
        （``TAX_ALL`` 在 1 位编码下与 ``TAX_SEKI`` 不可区分）。
        ⚠ 这会让 ``calculate_area`` 对这类局抛 ``NotImplementedError``（它对
        ``TAX_SEKI``/``TAX_ALL`` 尚未实现）—— **这是故意的**：静默返回全 0 会
        与「TERRITORY 恒 0」在张量上逐位相同，被当成「无归属」污染训练
        （见 `calculate_area` 的 docstring）。
      · ``ko`` 的 1 位 ``0`` ⇒ ``KO_SIMPLE``；``1`` ⇒ **按 ``KO_POSITIONAL``
        解读**（positional 与 situational 在 1 位编码下不可区分，而两者会让
        全局 ch7 分别取 ``+0.5`` / ``−0.5``）。

    ⇒ sidecar 的 ``g_rules`` 要支持全部 8 种组合，得先把它的打包改成
    :func:`rules_flags_from` 的布局；本函数是那之前的**兼容垫片**，不是终点。
    """
    v = int(g_rules)
    scoring = SCORING_TERRITORY if (v & 0b1) else SCORING_AREA
    tax = TAX_SEKI if (v & 0b10) else TAX_NONE
    ko = KO_POSITIONAL if (v & 0b100) else KO_SIMPLE
    suicide = bool(v & 0b1000)
    button = bool(v & 0b10000)
    return rules_flags_from(scoring, tax, ko, suicide, button)


# --------------------------------------------------------------------------- #
# 过去 5 手的**物理手序**（ch9..13 与全局 ch0..4 共用这一份）
# --------------------------------------------------------------------------- #
# ⚠ 这份顺序在两处被用到（ch9..13 的落点通道、全局 ch0..4 的 pass 标志），
#   所以它是**模块级唯一定义**、两处都从它取。写两遍就是下一次「通道顺序
#   静默错位」的入口 —— 而通道错位**不报任何错**，只是标签指向错误的点。
#: 第 ``k`` 项给出「物理手序第 ``k+1`` 手（``k=0`` 是最近一手）」的来源：
#: ``(哪一方, 第几槽)``。顺序是 ``opp, pla, opp, pla, opp`` —— 上一手永远是
#: 对手落的（自 ``nextPlayer`` 视角，即 ``to_play`` 是下一手执子方）。
_HISTORY_SLOTS = ('op', 0, 'my', 0, 'op', 1, 'my', 1, 'op', 2)


def _history_move(my, op, k):
    """物理手序第 ``k+1`` 手的 ``(B,)`` int64 坐标列（``-1`` = pass / 无此手）。"""
    who = _HISTORY_SLOTS[2 * k]
    slot = _HISTORY_SLOTS[2 * k + 1]
    src = op if who == 'op' else my
    return src[:, slot].astype(np.int64, copy=False)


def _as_to_play(to_play, B):
    """校验 ``(B,)`` 行棋方并转 int8。**每个装配入口都要过它** —— ch1/ch2 的
    绝对色、ch5/ch18 的符号翻转、ch17 的对手方、ch18/19 的 pla/opp 视角全都读它，
    漏一处就是静默的视角错位。"""
    tp = np.asarray(to_play, dtype=np.int8).reshape(-1)
    if tp.size != B:
        raise ValueError(f'to_play 长度 {tp.size} != B {B}')
    if not np.isin(tp, (-1, 1)).all():
        raise ValueError(f'to_play 只能取 ±1，实得 {np.unique(tp)}')
    return tp


def _as_ko(ko, B):
    """校验 ``(B,)`` simple-ko 点并转 int64（``-1`` = 无劫）。

    口径是 ``GoBoard.ko_point``：**紧凑**扁平下标 ``r*n+c``，与主数据集的
    ``ko`` 列同口径（``scripts/build_dataset.py:294`` 取的就是它）。
    """
    if ko is None:
        return np.full(B, -1, dtype=np.int64)
    k = np.asarray(ko).reshape(-1).astype(np.int64, copy=False)
    if k.size == 1:
        return np.full(B, int(k[0]), dtype=np.int64)
    if k.size != B:
        raise ValueError(f'ko 长度 {k.size} != B {B}')
    return k


def _ko_plane(kos, B, n):
    """``ko`` 点掩码 → ``(B,n,n)`` bool（ch6）。

    ⚠ 越界的 ``ko`` 值（脏数据）**静默跳过**而不是抛错：它只会让一个点不被点亮，
    而抛错会让整个预取 worker 死掉 —— 两者相比，显式丢一个点更可控。真正的
    合法性判定在 ``go_rules`` 侧，不靠这个通道（`go_rules.py:2416` 同一判断）。
    """
    out = np.zeros((B, n, n), dtype=bool)
    upper = n * n
    ok = (kos >= 0) & (kos < upper)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    r, c = np.divmod(kos[idx], n)
    out[idx, r, c] = True
    return out


# --------------------------------------------------------------------------- #
# B1-0 · 历史门控的**唯一**解析点
# --------------------------------------------------------------------------- #
class LadderBoards(NamedTuple):
    """:func:`resolve_ladder_boards` 的返回值。

    ``prev`` / ``prev_prev`` 是**逐行已解析好**的盘面（不可用行已被替换成官方
    回退值），非 ``None`` 时可直接喂给 :func:`ladder_channels`。

    ``prev_all_missing`` / ``prev_prev_all_missing`` 标出「整批都缺这一级历史」，
    用来让调用方**传 ``None`` 而不是等价数组** —— 见
    :func:`resolve_ladder_boards` 的 docstring 里那条性能/正确性理由。
    """

    prev: np.ndarray
    prev_prev: np.ndarray
    prev_all_missing: bool
    prev_prev_all_missing: bool


def resolve_ladder_boards(boards, prev_board=None, prev_prev_board=None,
                          prev_valid=None, prev_prev_valid=None):
    """按官方原文**逐行**解析 ladder 三通道要用的三块盘面。

    官方原文（spec §2.2 照录）::

        prevBoard     = (numTurnsOfHistoryIncluded < 1) ? board     : hist.getRecentBoard(1);
        prevPrevBoard = (numTurnsOfHistoryIncluded < 2) ? prevBoard : hist.getRecentBoard(2);

    ⇒ 历史不足时**回退复制**，不是置 0。history=0 ⇒ ch15 == ch14、
    **ch16 == ch15**；history=1 ⇒ **ch16 == ch15**。

    ⚠⚠ **第二级回退的兜底是 ``prev``（解析**之后**的），不是 ``board``。**
    history=1 时 ``prev`` 是真正的上一手盘、并不等于当前盘；回退到 ``board``
    会让 ch16 差一手。这是这条规则唯一的坑，也正是
    ``feature_v7_ladders.py`` 里「别把门控从历史参数再推一遍」那条注释的由来 ——
    那次是从同一个事实**推导两次**、第二次推出了错的答案。

    ---- 为什么本函数是**唯一**的解析点 ----
    同一段门控一旦写两遍就会发散（仓库里已有两处前科，见上面 ``rules_flags``
    段的注释）。所以本函数**只解析一次**，产出两块已解析盘面 +
    两个「整批缺失」标志；``ladder_channels`` 拿到的要么是 ``None``
    （= 它自己内部按官方原文回退，那**一次**），要么是已解析好的等价数组。

    ⚠ **「整批缺失 ⇒ 传 None」不是纯优化，是正确性要求。**
    ``ladder_channels`` 判定回退用的是**数组身份**（``pb is b`` / ``p2b is pb``，
    见它的 docstring —— 那是刻意设计，因为它把「同一份门控事实」用了两次而
    不想再解析一遍）。本函数若在「整批缺失」时传一份 ``boards.copy()``，
    ``ladder_channels`` 会看到「不是同一个对象」⇒ **真去跑一遍梯子搜索**。
    结果是对的（回退后的盘面确实等于当前盘），但白跑一整轮 DFS；更糟的是
    这条路径很难被测到（结果一致，只是慢）。

    Args:
        boards: ``(B,n,n)`` int8，**当前盘**。
        prev_board / prev_prev_board: ``(B,n,n)`` 或 ``None``。``None`` = 该级
            历史**整批**不可用。
        prev_valid / prev_prev_valid: ``(B,)`` bool 或 ``None``。**逐行**的可用性
            （邻行 gather 的越界 / 跨局守卫，见
            ``src/data/feature_v7_gather.py``）。``None`` = 整批可用。

    Returns:
        :class:`LadderBoards`。
    """
    cur = _as_batch_boards(boards)
    shape = cur.shape
    B = shape[0]

    def _gate(given, valid, fallback, name):
        arr = None if given is None else np.asarray(given)
        if arr is not None and arr.shape != shape:
            raise ValueError(f'{name} 形状不符：{arr.shape}，期望 {shape}')
        if valid is None:
            return (fallback, True) if arr is None else (arr, False)
        v = np.asarray(valid).reshape(-1).astype(bool)
        if v.size == 1:
            v = np.repeat(v, B)
        if v.size != B:
            raise ValueError(f'{name}_valid 长度 {v.size} != B {B}')
        if arr is None:
            if v.any():
                raise ValueError(f'{name} 是 None 但 {name}_valid 有 True 行 —— '
                                 f'没有盘面可以填那些行。')
            return fallback, True
        if not v.all():
            return np.where(v.reshape(B, 1, 1), arr, fallback), False
        return arr, False

    prev, prev_missing = _gate(prev_board, prev_valid, cur, 'prev_board')
    # ⚠ 兜底是上面**已解析**的 prev —— 见 docstring 的坑。
    prev_prev, prev_prev_missing = _gate(prev_prev_board, prev_prev_valid, prev,
                                         'prev_prev_board')
    return LadderBoards(prev, prev_prev, prev_missing, prev_prev_missing)


def spatial_channels_v7(boards, to_play, ko, my_hist, op_hist,
                        prev_board=None, prev_prev_board=None, rules_flags=0,
                        prev_valid=None, prev_prev_valid=None):
    """把各通道实现拼成 ``(B,22,19,19)`` float16 的 V7 空间输入（任务 B1）。

    通道表逐字取自 spec §2.2：

    ===  =====================================================  ==================
    ch   语义                                                 来源
    ===  =====================================================  ==================
    0    on-board（19×19 无填充 ⇒ 恒 1）                       ``np.ones``
    1    **黑**（+1）棋子                                      ``boards > 0``
    2    **白**（−1）棋子                                      ``boards < 0``
    3/4/5 棋子的 1 / 2 / 3 气（**不分色**）                     `liberties_123`
    6    ko-ban：encore=0 时只有 ``ko_loc``（不跟踪 superko）   `ko` 列
    7    ``koRecapBlocked`` —— **仅 encore>0**                  恒 0
    8    源码注释写「6,7,8」但**从未写入**                      恒 0
    9-13 过去 5 手落点（顺序 ``opp, pla, opp, pla, opp``）      `history_five`
    14   **当前盘**上的梯子                                    `ladder_channels`
    15   **前一手盘**上的梯子                                  同上（邻行 gather）
    16   **前二手盘**上的梯子                                  同上（邻行 gather）
    17   当前盘梯子的 working-move 位置（**仅对手方、且 >1 气**）同上
    18/19 当前区域（pla / opp 相对视角）                        `calculate_area`
    20/21 second-encore 起始子 —— **仅 encorePhase≥2**          恒 0
    ===  =====================================================  ==================

    ---- ch0 为什么是 on-board、不是「黑」（与任务书原文的一处出入）----
    任务书写「通道 0 = 黑 / 通道 1 = 白」。ch1/ch2 = 黑/白 采纳；但 **ch0 必须是
    on-board**：官方 ``binaryInputNCHWPacked`` 的 ch0 是 on-board 掩码，本仓
    ``katago_npz.board_size_from_packed`` **正是靠 ch0 的 1 的个数反推棋盘边长**
    （spec §9.1 第 2 条：同一个 npz 里混着 9/11/13/18 路与非方阵）。把 ch0 写成
    「黑子」会让那条反推静默失效。

    ---- ch1/ch2 用**绝对色**而不是官方的 pla/opp ----
    官方是「相对 ``nextPlayer``」（ch1 = pla）。本实现固定 ch1 = 黑、ch2 = 白。
    理由：① ``to_play=+1`` 时与官方**逐位相同**（stdata 实测的口径就是
    ``to_play`` 恒 +1，见 ``tests/test_v7_ladders.py``），对拍不失真；
    ② 于是 ch0/1/2/3/4/5/9-13 这一整块**与 to_play 无关**，在 8 路 dihedral
    增广下不需要任何视角修正（spec §5.6）。
    ⚠ 因此 ch1/ch2 与官方在 ``to_play=-1`` 时**是对调关系**。这是有意的口径，
      已在 ``tests/test_v7_assemble.py`` 里逐项钉住，不留给下游去猜。
    ⚠ 对照：ch18/19 **仍按官方的 pla/opp 相对视角**（`calculate_area` 的既有
      口径），所以这一块是 to_play 相关的 —— 别把 ch1/ch2 的绝对色口径外推过去。

    ---- dtype ----
    float16，与 ``GoBoard.feature_planes`` / ``feature_planes_batched`` 逐位一致
    （`go_rules.py:2434`：两路必须同 dtype，否则单图推理与批量训练拿到不同
    精度的 planes）。取值域只有 {0,1} ⇒ 逐位无损。

    ---- 已实测与官方 stdata **逐位对齐**的通道 ----
    ch3 / ch4 / ch5（气桶）、ch9..ch13 的顺序（颜色分布）、ch14 / ch17（梯子，
    实测 1.000000 / 4368 行）。🔴 **ch18/19 对不齐**（官方算 area 前先提死子，
    本仓 ``score()`` 不做 ⇒ 0 行完全相同；实测数字见 ``area_ownership_map``
    的 docstring）。若某个已对齐的通道掉下来，**查这里的组装顺序，不要改底层
    实现**。

    Args:
        boards: ``(B,19,19)`` int8，−1=白 / +1=黑。
        to_play: ``(B,)`` ±1，轮到谁落子。
        ko: ``(B,)`` int16，simple-ko 点（``r*n+c``，``-1``=无）。
        my_hist / op_hist: ``(B,3)`` 整数，**相对 to_play**、index 0 是最近一手。
        prev_board / prev_prev_board: ``(B,19,19)`` 或 ``None``（= 历史整批不足）。
        rules_flags: 见本模块的位布局段；默认 :data:`DEFAULT_RULES_FLAGS`。
        prev_valid / prev_prev_valid: ``(B,)`` bool，逐行可用性（邻行 gather 的
            越界 / 跨局守卫）。与 ``prev_board=None`` 同形但更细，见
            :func:`resolve_ladder_boards`。
    """
    b = _as_batch_boards(boards)
    B, n, _ = b.shape
    if n != BOARD_SIZE:
        raise ValueError(f'V7 固定 {BOARD_SIZE}x{BOARD_SIZE}（spec §1.3），'
                         f'收到 {n}x{n}。ladder 通道的移植本身就是固定盘面，'
                         f'放行其它尺寸只会让静默错位更难发现。')
    tp = _as_to_play(to_play, B)
    kos = _as_ko(ko, B)

    out = np.zeros((B, SPATIAL_CHANNELS, n, n), dtype=SPATIAL_DTYPE)

    # ---- ch0 · on-board（见 docstring：必须是 on-board，不是「黑」）--------
    out[:, 0] = 1.0

    # ---- ch1 / ch2 · 绝对色（黑 / 白）------------------------------------
    out[:, 1] = (b > 0)
    out[:, 2] = (b < 0)

    # ---- ch3 / ch4 / ch5 · 1/2/3 气（不分色，**不读 to_play**）------------
    out[:, 3:6] = liberties_123(b)

    # ---- ch6 · ko-ban（encore=0 ⇒ 只有 ko_loc，见 spec §2.6 D1）----------
    out[:, 6] = _ko_plane(kos, B, n)

    # ---- ch7 / ch8 · 恒 0（encore-only / 源码从未写入）--------------------
    # 保留不裁剪的理由见 spec §2.4：严格复刻官方张量布局，接官方 checkpoint 时
    # 零改形状；裁剪只省 0.3% 参数。

    # ---- ch9..ch13 · 过去 5 手落点 ----------------------------------------
    out[:, 9:14] = history_five(b, my_hist, op_hist)

    # ---- ch14..ch17 · 梯子三通道 + working-move ---------------------------
    gate = resolve_ladder_boards(b, prev_board, prev_prev_board,
                                 prev_valid, prev_prev_valid)
    out[:, 14:18] = ladder_channels(
        b, tp,
        # ⚠ 整批缺失时传 `None`，让 `ladder_channels` 自己走它那份（唯一的）
        #   门控解析、并保住 `is` 快路径。见 `resolve_ladder_boards` 的 docstring。
        None if gate.prev_all_missing else gate.prev,
        None if gate.prev_prev_all_missing else gate.prev_prev,
        ko=kos)

    # ---- ch18 / ch19 · 当前区域（官方是两条 bool 平面）--------------------
    # ⚠ `calculate_area` 返回**带符号**的两视角（+1=归 pla / −1=归 opp），而官方
    #   ch18/19 是两条**布尔**平面（pla 有 / opp 有）。这里做二值化，否则 −1
    #   会让「有归属」的点与「on-board」的点在张量上混同，而且无法与
    #   `unpack_binary_input` 的解包结果对拍。
    area = calculate_area(b, tp, rules_flags=rules_flags)
    out[:, 18] = area[:, 0] > 0
    out[:, 19] = area[:, 1] > 0

    # ---- ch20 / ch21 · 恒 0（second-encore 起始子，仅 encorePhase≥2）------
    return out


# --------------------------------------------------------------------------- #
# B5 · 19 维全局特征
# --------------------------------------------------------------------------- #
class GameRow(NamedTuple):
    """局级标量（sidecar ``games.npz`` 的一行，spec §5.3）里本模块要用的字段。

    ⚠ **局级而非逐行**：逐行存是 3.6 B/行 ⇒ 34.2M 行 123 GB；局级是 1.2 MB。
      读取口径是 ``sidecar[game_ids[idxs]]``（spec §5.3）。
    """

    komi: float                    # SGF `KM`，黑 − 白；缺 `KM` 时为 ``nan``
    rules_flags: int = DEFAULT_RULES_FLAGS


def _komi_of(game_row, B):
    """取 ``(B,)`` float32 的 komi（**未**翻符号、未 clip）。

    接受 :class:`GameRow`、任何有 ``komi`` 属性的对象、或含 ``'komi'`` 键的
    mapping —— 三者都覆盖得到（sidecar 读取、NamedTuple 夹具、裸 dict）。

    ⚠ ``komi`` 为 ``nan`` 时取 **0.0**：SGF 里 **2.2% 的局没有 `KM``**，而 0.0
      是该情形的惯例默认。这里显式兜住，而不是让 ``nan`` 顺着 clip 污染
      全局 ch5 **和** ch18（后者是三角波输入，nan 会毁掉整条曲线）。
    """
    if game_row is None:
        return np.zeros(B, dtype=np.float32)
    if hasattr(game_row, 'komi'):
        raw = game_row.komi
    elif isinstance(game_row, Mapping) and 'komi' in game_row:
        raw = game_row['komi']
    else:
        raise TypeError(
            f'game_row 既没有 .komi 也没有 ["komi"] 键：{type(game_row).__name__}。'
            f'局级标量的形状见 spec §5.3（sidecar games.npz 的 g_komi / g_rules）。')
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.full(B, arr[0], dtype=np.float32)
    elif arr.size != B:
        raise ValueError(f'game_row.komi 长度 {arr.size} != B {B}')
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def self_komi(komi, to_play, board_area):
    """``(B,)`` float32 的 ``currentSelfKomi(nextPlayer)``。

    官方（spec §2.3）：黑走 ⇒ ``+komi``，白走 ⇒ ``−komi``，再 clip 到
    ``±(bArea + 1.0)``。⇒ **符号随 ``to_play`` 翻转**。

    ⚠ **spec §2.6 D3 的有意偏差**：官方这个量还含 **draw-jitter** 与
    （encore 下的）``whiteBonusScore``，本仓直接用局表 ``g_komi`` —— 因为
    jitter 在 SGF 里根本不存在（它只在 KataGo 搜索时按 seed 加）。
    实测官方 ``globalInputNC[:,5]`` 上会看到 5.424 / 7.9 这类「非整数贴目」，
    那正是 jitter 的痕迹。
    """
    sk = np.asarray(komi, dtype=np.float32) * np.asarray(to_play, dtype=np.float32)
    lim = float(board_area) + KOMI_CLIP_MARGIN
    return np.clip(sk, -lim, lim)


def komi_parity_wave(self_komi_, board_area, scoring_rule=SCORING_AREA,
                     encore_phase=0):
    """``(B,)`` float32 的官方全局 **ch18**（贴目 × 棋盘奇偶三角波）。

    官方原文（``nninputs.cpp``；V3 里写在 ``rowGlobal[13]``、V7 里是 18）::

        if(rules.scoringRule == SCORING_AREA || encorePhase >= 2) {
          bool boardAreaIsEven     = (xSize*ySize) % 2 == 0;
          bool drawableKomisAreEven = boardAreaIsEven;
          float komiFloor = drawableKomisAreEven ? floor(selfKomi/2)*2
                                                 : floor((selfKomi-1)/2)*2 + 1;
          float delta = selfKomi - komiFloor;   // clip 到 [0,2]
          float wave  = (delta < 0.5) ? delta
                    : (delta < 1.5) ? 1.0-delta
                                    : delta-2.0;
          rowGlobal[18] = wave;
        }

    为什么是三角波（官方的注释原文）：从白方视角，komi = 0.0 能和棋、
    0.5 赢下原来和的棋、1.0/1.5 与 0.5 差别不大、2.0 又能和棋 ⇒ 贴目的
    「有效 goodness」是 ``0 1 1 1 2 3 3 3 4 …``，这个函数既难学又与棋盘
    面积的奇偶强相关（很像 xor）。加上 ``0.5 × (0 -1 0 1 0 -1 …)`` 之后
    就近似线性了 —— 官方干脆把这个 xor 直接作为输入喂进来。
    ⚠ 源码里紧跟着的一句注释就是「若改动这个特征的下标，必须同步改
    ``model.py`` 里乘进 scorebelief parity 向量的那个下标」—— spec §2.3 记的
    「V3/V4=13、V6=15、V7=18」正是这条注释的产物。

    ⚠ **门控与官方一致**：``SCORING_TERRITORY`` 且 ``encorePhase < 2`` ⇒ 恒 0。
      这与 spec §2.5 的「territory 对局 ch18/19 恒 0」是同一族规则
      （实测官方 ``globalInputNC[:,9]==1`` 的行里 ``[:,18]`` 绝大多数是 0）。

    ⚠ ``board_area`` **奇偶才起作用**：19×19（361）与 11×11（121）都走
      ``komiFloor = floor((selfKomi-1)/2)*2 + 1`` 这一支；偶数盘走另一支。

    ✅ **实测验证**：官方 stdata ``2026-08-25npzs.tgz`` 的 ``globalInputNC``
      上取 7 组不同贴目/盘面，手工按上式复算 **全部命中**（例如 19 路
      komi 7.5 ⇒ ``floor(6.5/2)*2+1 = 7``、delta 0.5 ⇒ wave **0.5**，官方
      那一格就是 0.5；komi 6.68 ⇒ floor 5、delta 1.68 ⇒ wave **−0.32**，
      官方是 −0.3202）。`tests/test_v7_assemble.py` 用**官方 ch5 反推**
      ``selfKomi`` 做成批对拍。
    """
    sk = np.asarray(self_komi_, dtype=np.float32)
    if scoring_rule != SCORING_AREA and encore_phase < 2:
        return np.zeros(sk.shape, dtype=np.float32)

    drawable_even = (int(board_area) % 2) == 0
    if drawable_even:
        floor = np.floor(sk / 2.0) * 2.0
    else:
        floor = np.floor((sk - 1.0) / 2.0) * 2.0 + 1.0
    delta = np.clip(sk - floor, 0.0, 2.0).astype(np.float32)
    wave = np.where(delta < 0.5, delta,
                    np.where(delta < 1.5, 1.0 - delta, delta - 2.0))
    return wave.astype(np.float32, copy=False)


def pass_would_end_phase(my_hist, op_hist, history_length):
    """``(B,)`` bool 的官方全局 **ch14**（pass 会不会结束当前阶段）。

    区域计分只有一个阶段，结束条件是**连续两手 pass** ⇒
    「这一手 pass 会结束阶段」⟺「**上一手已经是 pass**」。上一手永远是对手落的
    ⇒ 看 ``op_hist[:, 0]``。

    ⚠ **历史不足时必须给 False。** ``my_hist`` / ``op_hist`` 用 ``-1`` 同时表示
    「pass」与「无此手」，两者在 ch9..13 上输出相同（空），在这里**不同** ——
      开局第一手（ply 0）没有任何上一手，若当成 pass 会把整段开局的 ch14 打成 1。
      所以只在 ``history_length >= 1`` 时才认这个 pass 标志。
    """
    op = np.asarray(op_hist)
    last = op[:, 0]
    has_prev = np.asarray(history_length).reshape(-1) >= 1
    return (last < 0) & has_prev


def _resolve_history_length(my_hist, op_hist, history_length, B):
    """把 ``history_length`` 归一成 ``(B,)`` int64 的「真实可用手数」。

    ``None`` 时按「六个历史槽**全**是 ``-1`` ⇒ ply 0」推导，否则当成 5 手齐全。

    ⚠ **为什么默认 5 而不是「有几个不是 −1 就数几个」**：后者会把「历史不足」
      也报成 pass —— 而官方的 ``PASS_LOC`` 与 ``NULL_LOC`` 是**不同**的 loc
      （``-1`` 只对应后者），官方对缺失历史给的是 **0**。默认 5 只在 ply ≥ 5 的
      行上生效，而 spec §5.3.4 实测 **100% 的局 ≥ 20 手** ⇒ 开局前几行是唯一
      可能不准的地方，而它们本来就靠 ply 0 判定兜住了。
    ⚠ 已知的一处边缘偏差：ply 1 且第 0 手恰好是 pass 时会被当成 ply 0 ⇒
      ch14 给 False 而不是 True。影响面 = 「开局第一手就 pass」的局 × 开局那一
      行；调用方在意就显式传 ``history_length``。
    """
    if history_length is not None:
        hl = np.asarray(history_length)
        if hl.ndim == 0:
            hl = np.full(B, int(hl))
        hl = hl.reshape(-1).astype(np.int64)
        if hl.size == 1:
            hl = np.full(B, int(hl[0]))
        if hl.size != B:
            raise ValueError(f'history_length 长度 {hl.size} != B {B}')
        return np.clip(hl, 0, HISTORY_MOVES)
    my = np.asarray(my_hist)
    op = np.asarray(op_hist)
    empty = (my < 0).all(axis=1) & (op < 0).all(axis=1)
    return np.where(empty, 0, HISTORY_MOVES).astype(np.int64)


def global_features_v7(game_row, my_hist, op_hist, prev_board=None,
                       prev_prev_board=None, history_length=None,
                       rules_flags=None, *, to_play=None, board_area=None,
                       encore_phase=0):
    """V7 的 ``(B,19)`` float16 全局输入（任务 B5，spec §2.3）。

    ==============  =========================================  ====================
    ch              语义（官方）                                来源
    ==============  =========================================  ====================
    0-4             「第 ``k+1`` 手是不是 pass」                 ``my_hist``/``op_hist``
    5               ``currentSelfKomi(nextPlayer)/20``          局表 `g_komi`
                    clip 到 ``±(bArea+1)``；**符号随 to_play 翻**
    6 / 7           ko 规则：simple=(0,0) / positional=(1,+0.5)
                    / situational=(1,−0.5)                     `rules_flags`
    8               ``multiStoneSuicideLegal``                  `rules_flags`
    9               territory 计分 ⇒ 1（area ⇒ 0）              `rules_flags`
    10 / 11         tax：none=(0,0) / seki=(1,0) / all=(1,1)     `rules_flags`
    12 / 13         ``encorePhase > 0`` / ``> 1``               简单局恒 0
    14              ``passWouldEndPhase``                        `pass_would_end_phase`
    15 / 16         ``playoutDoublingAdvantage`` 非 0 ⇒ 1 /
                    0.5·pda                                      无 PDA ⇒ 恒 0
    17              ``hasButton``                                `rules_flags`
    18              贴目 × 棋盘奇偶三角波                        `komi_parity_wave`
    ==============  =========================================  ====================

    ⚠⚠⚠ **这份定义以 spec §2.3 为准**。**任务书里给的那串「禁贴/禁入/气/提子数/
    自手贴/ko 计数/上次吃子数/距上次吃子回合数/simple-ko 阻塞合法性/局内经过回合/
    对局结果」不是 V7 的全局输入** —— V7 的 19 维里没有「提子数」「距上次吃子
    回合数」「对局结果」（结果在 ``globalTargetsNC`` 侧，是**目标**不是输入）。
    判据链三条，任何一条都足以排除那份转述：

      1. ``nninputs.h`` 明写 ``NUM_FEATURES_GLOBAL_V7 = 19``，且 ``fillRowV7``
         只写 0-18 这 19 格；
      2. stdata 官方 ``globalInputNC (N,19)`` 实测逐项吻合本表：ch5 上
         komi=7.5 的行恰为 **0.375 = 7.5/20**（不是 7.5/15）；ch6/7 只出现
         ``(0,0)`` / ``(1,+0.5)`` / ``(1,−0.5)`` 三态，且 ``ch7 ≠ 0 ⟺ ch6 ≠ 0``；
         ch8/ch9/ch17 是 0/1 布尔；ch10/11 只出现 ``(0,0)/(1,0)/(1,1)``；
         ch15/16 **恒 0**（无 PDA）；
      3. ch18 的三角波从官方源码取到原文，并在 7 组真实官方数据上验算全部命中
         （`tests/test_v7_assemble.py` 有成批对拍）。

    ⚠ **缺 `RU` 时的默认值**：SGF 语料里 **55% 的局没有 `RU`**（spec §5.3.2），
      且从 `RU` 字符串解析不出 tax/ko/suicide/button 的可靠组合 ⇒ 这四位一律取
      **默认值**（:data:`DEFAULT_RULES_FLAGS`：AREA + TAX_NONE + KO_SIMPLE +
      单子禁着 + 无 button），于是全局 ch6/7/8/10/11/17 = ``(0,0) / 0 / (0,0) /
      0``。**扩展位留着**：拿到可靠的规则来源后，只要改
      :func:`rules_flags_from_sidecar` 或直接传 ``rules_flags``，本函数一行都不用改。

    ⚠ **段 1 不以 score 系为主目标**：语料里 **约 81% 的 SGF 是认输**（无分差，
      spec §5.3.3），而本函数**不产出任何 score 相关维度** —— 官方也没有
      （score 在 ``globalTargetsNC`` 侧）。段 1 只训 policy / value / futurepos。

    ---- ``prev_board`` / ``prev_prev_board`` 为什么不参与 ----
    spec §2.3 的这 19 维**没有一维**需要历史盘面：ch14 只需「上一手是不是
    pass」（历史坐标列就够）、ch18 只用 komi 与盘面尺寸。两个形参按任务书签名
    保留（并做形状校验），但**语义上不读** —— 将来若要补「提子数」一类特征，
    位置已经在这里，而不必再改签名。

    Args:
        game_row: 局级标量 —— :class:`GameRow`，或任何有 ``.komi`` / ``['komi']``
            的对象。
        my_hist / op_hist: ``(B,≥3)`` 整数，**相对 to_play**、index 0 是最近一手。
        prev_board / prev_prev_board: 可选，**仅校验形状、不参与取值**（见上）。
        history_length: 真实可用的历史手数（``int`` / ``(B,)`` / ``None``）。
            只影响 ch0..4 与 ch14：``-1`` 既可能是 pass 也可能是「无此手」，
            只有落在 ``history_length`` 内的槽位才判 pass。
        rules_flags: 见本模块位布局段。``None`` ⇒ 取 ``game_row.rules_flags``。
        to_play: ``(B,)`` ±1。**必填**（不给就报错）—— ch5/ch18 的符号翻转靠它。
        board_area: 默认 ``19*19``。只影响 ch5 的 clip 半径与 ch18 的奇偶分支。
        encore_phase: 默认 0（简单局）。只影响 ch12/13 与 ch18 的门控。
    """
    if to_play is None:
        raise ValueError(
            'to_play 是必填的：官方 ch5 = currentSelfKomi(nextPlayer) 与 ch18 的\n'
            '三角波都**随 nextPlayer 翻转符号**，不给它就只能瞎猜一半的全局特征。\n'
            '（spec §2.3 ch5 那行「符号随 to_play 翻转」。）')
    my = np.asarray(my_hist)
    op = np.asarray(op_hist)
    if my.ndim == 1:
        my = my.reshape(-1, 1)
    if op.ndim == 1:
        op = op.reshape(-1, 1)
    if my.shape[0] != op.shape[0]:
        raise ValueError(f'my_hist {my.shape} 与 op_hist {op.shape} 批大小不同')
    B = my.shape[0]
    if my.shape[1] < 3 or op.shape[1] < 3:
        raise ValueError(f'需要每人最近 3 手（5 格交错用 3+2），'
                         f'实得 my={my.shape} / op={op.shape}')
    tp = _as_to_play(to_play, B)
    area = BOARD_SIZE * BOARD_SIZE if board_area is None else int(board_area)

    # ⚠ `prev_board` / `prev_prev_board` **只校验形状、不参与取值**（spec §2.3 的
    #   19 维没有一维需要历史盘面，见本函数 docstring）。仍然校验，是为了让
    #   「传错了形状」在装配层就炸掉，而不是等到将来给它们接上特征时才发现
    #   一直传的是错的形状。
    for name, arr in (('prev_board', prev_board),
                      ('prev_prev_board', prev_prev_board)):
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.shape != (B, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f'{name} 形状不符：{a.shape}，'
                             f'期望 ({B}, {BOARD_SIZE}, {BOARD_SIZE})'
                             f'（该参数当前只校验形状、不参与取值）')

    if rules_flags is None:
        rules_flags = getattr(game_row, 'rules_flags', DEFAULT_RULES_FLAGS)
    fl = np.asarray(rules_flags)

    out = np.zeros((B, GLOBAL_CHANNELS), dtype=GLOBAL_DTYPE)
    hl = _resolve_history_length(my, op, history_length, B)

    # ---- ch0..4 · 过去 5 手各自是不是 pass（物理手序，与 ch9..13 同一份）---
    for k in range(HISTORY_MOVES):
        out[:, k] = (_history_move(my, op, k) < 0) & (k < hl)

    sk = self_komi(_komi_of(game_row, B), tp, area)

    # ---- ch5 · 自手贴目 / 20 ---------------------------------------------
    out[:, 5] = sk / KOMI_SCALE

    # ---- ch6 / ch7 · ko 规则 ---------------------------------------------
    ko_rule = (fl & FLAG_KO_MASK) >> 3
    out[:, 6] = (ko_rule != KO_SIMPLE).astype(np.float32)
    out[:, 7] = np.where(ko_rule == KO_POSITIONAL, 0.5,
                         np.where(ko_rule == KO_SITUATIONAL, -0.5, 0.0))

    # ---- ch8 · multiStoneSuicideLegal ------------------------------------
    out[:, 8] = ((fl & FLAG_MULTISTONE_SUICIDE) != 0).astype(np.float32)

    # ---- ch9 · 是否 territory 计分 ----------------------------------------
    out[:, 9] = ((fl & FLAG_SCORING_TERRITORY) != 0).astype(np.float32)

    # ---- ch10 / ch11 · tax ------------------------------------------------
    tax = (fl & FLAG_TAX_MASK) >> 1
    out[:, 10] = (tax >= TAX_SEKI).astype(np.float32)
    out[:, 11] = (tax == TAX_ALL).astype(np.float32)

    # ---- ch12 / ch13 · encorePhase ----------------------------------------
    out[:, 12] = float(encore_phase > 0)
    out[:, 13] = float(encore_phase > 1)

    # ---- ch14 · passWouldEndPhase ------------------------------------------
    out[:, 14] = pass_would_end_phase(my, op, hl).astype(np.float32)

    # ---- ch15 / ch16 · playoutDoublingAdvantage ---------------------------
    # 本仓无 PDA（spec §2.2：15/16 恒 0，保留不裁剪）。

    # ---- ch17 · hasButton --------------------------------------------------
    out[:, 17] = ((fl & FLAG_HAS_BUTTON) != 0).astype(np.float32)

    # ---- ch18 · 贴目 × 棋盘奇偶三角波 ---------------------------------------
    scoring = SCORING_TERRITORY if (fl & FLAG_SCORING_TERRITORY) != 0 else SCORING_AREA
    out[:, 18] = komi_parity_wave(sk, area, scoring, encore_phase)

    return out