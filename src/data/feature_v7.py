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
# B3 · ch18 / ch19：calculateIndependentLifeArea
# --------------------------------------------------------------------------- #
#: 4 邻域的四个偏移（`(dr, dc)`），顺序与官方 `board.cpp` 的 `adj_offsets`
#: （`board.cpp:39-42`：`{-(x_size+1), -1, +1, x_size+1}`）一致。
#: ⚠ 只在**本模块的私有 helper** 里用；不要拿去改 `go_rules._NB4` 之类。
_NB4_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _shift4(a, dr, dc):
    """把 `(B,n,n)` 数组按 4 邻域平移一格，**移出盘面的位置填 0**。

    用切片赋值而不是 `np.roll`：`np.roll` 会**环绕**（第一行接到最后一行），
    而棋盘的边界外是「没有邻点」⇒ 必须是 0。环绕会让边界上的 vital / dame
    判定读到对面的子，是静默错。

    ⚠ 第 0 轴（batch）**不**动：`_STRUCT3` 与本函数都保证样本间不互串。
    """
    out = np.zeros_like(a)
    src = [slice(None), slice(None), slice(None)]
    dst = [slice(None), slice(None), slice(None)]
    if dr == -1:
        src[1], dst[1] = slice(1, None), slice(0, -1)
    elif dr == 1:
        src[1], dst[1] = slice(0, -1), slice(1, None)
    if dc == -1:
        src[2], dst[2] = slice(1, None), slice(0, -1)
    elif dc == 1:
        src[2], dst[2] = slice(0, -1), slice(1, None)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _vital_pairs(b, rid, nreg, pc, nchain, suicide_legal):
    """Benson 的 vital 表：区域 `R` 对链 `C` 是否 vital（= `R` 的**每个点**都邻接 `C`）。

    官方定义在 `board.cpp:1980`（"A region is vital for a pla group if all its
    spaces are adjacent to that pla group"），实现在 `buildRegion` 的逐点过滤
    （`board.cpp:2033-2049`）。本函数是那份过滤的**稀疏向量化**。

    Returns:
        `(vp_r, vp_c, vital_count)`：`vp_r/vp_c` 是所有 vital 的
        `(区域号, 链号)` 对（**只含 vital 的对**），`vital_count[c]` 是链 `c`
        的 vital 区域数（官方 `vitalCountByPlaHead`）。

    ⚠ **去重必须按「(点, 链)」而不是按「(区域, 链)」**。同一个空点可能在两个方向
      上都邻接同一条链（U 形眼是常态），按 `(区域, 链)` 去重会把它数成 2 ⇒
      那个区域被判成 vital ⇒ 一条本该判死的链被判活。这是本函数唯一的坑。
    ⚠ **`suicide_legal=False` 时只过滤空点**（`board.cpp:2036`：
      `if(isVlenNonZero && (isMultiStoneSuicideLegal || colors[loc] == C_EMPTY))`）——
      区域里混着的**对方子**不参与过滤。官方训练数据两种规则都有
      （`configs/training/gatekeeper1.cfg:35` 的 `multiStoneSuicideLegals = false,true`）。
    """
    width = nchain + 1
    flat_rid = rid.reshape(-1)
    in_region = rid > 0
    # 只有空点参与过滤（suicide 合法时全部点参与）
    eligible = in_region if suicide_legal else (in_region & (b == 0))
    elig_flat = eligible.reshape(-1)
    keys = []
    for dr, dc in _NB4_DIRS:
        nb = _shift4(pc, dr, dc)
        m = elig_flat & (nb.reshape(-1) > 0)
        if m.any():
            keys.append(np.flatnonzero(m).astype(np.int64) * width + nb.reshape(-1)[m])
    if not keys:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(nchain + 1, np.int64))
    # 第一步：按 (点, 链) 去重 —— 同一个点同一个链只留一次。
    pt_ch = np.unique(np.concatenate(keys))
    pt = pt_ch // width
    ch = pt_ch - pt * width
    # 第二步：按 (区域, 链) 聚合计数，与区域大小比 ⇒ 是否「每个点都邻接」。
    # ⚠ 分母必须是**参与过滤的点数**，不是区域总点数：官方 `board.cpp:2036` 的
    #   `if(isVlenNonZero && (isMultiStoneSuicideLegal || colors[loc] == C_EMPTY))`
    #   意味着 suicide 不合法时**区域里的对方子不施加任何约束** —— 那颗子被跳过，
    #   根本不进 `vitalForPlaHeadsLists` 的过滤。拿总点数当分母会让「含对方子的
    #   区域」永远达不到 vital（`ucount ≤ 空点数 < 总点数`）⇒ 一条有两个真眼的
    #   链被判死 ⇒ 它的眼位与整块属地被抹平。这是本函数唯一的第二个坑。
    size = np.bincount(flat_rid[elig_flat], minlength=nreg + 1)
    pair = flat_rid[pt] * width + ch
    upair, ucount = np.unique(pair, return_counts=True)
    up_r = upair // width
    up_c = upair - up_r * width
    vital = ucount >= size[up_r]
    vp_r = up_r[vital]
    vp_c = up_c[vital]
    return (vp_r, vp_c,
            np.bincount(vp_c, minlength=nchain + 1).astype(np.int64))


def _area_for_pla(b, pc, nchain, pla, suicide_legal, result):
    """官方 `Board::calculateAreaForPla(pla, safe=true, unsafe=true, suicide)`，
    **就地**写 `result`（`board.cpp:1949-2244`）。

    `pc` 是 pla 侧的全局链号（`(B,n,n)` int32，1..nchain，其余 0）。
    ⚠ **`result` 是跨两次调用共享的同一个缓冲**：官方是
      `calculateAreaForPla(P_BLACK,…)` 写完，再 `calculateAreaForPla(P_WHITE,…)`
      在**同一块** `area` 上接着写（`board.cpp:1861-1862`）。
      `:2233` 的 `result[cur]==C_EMPTY` 守卫（`board.cpp:2237`）正是靠这个
      「黑先白后」的顺序才成立 —— **顺序不能换**，换了白就会覆盖黑已写的点。
    """
    opp = -pla
    nonpla = (b == 0) | (b == opp)
    rid, nreg = _ndi_label(nonpla, structure=_STRUCT3)
    # 🔴 区域**只能从空点起头**（`board.cpp:2083-2086`：`colors[loc] != C_EMPTY`
    #   那一支只更新 `atLeastOnePla` 然后 `continue`）。`scipy.ndimage.label` 会
    #   老老实实把「被本方子四面围住的一颗对方子」也标成一个 1 点区域，而官方
    #   根本不产生这个区域 —— 后果不是「多一个无用区域」，而是那个区域
    #   `numInternal=0` 且 `!containsOpp` 不成立… 实际是 `containsOpp=true` 但
    #   `:2221` 不看 `containsOpp` ⇒ 它被**无条件写成 pla 的属地**，一颗白子
    #   在 ch18/19 上显示成黑。实测：随机 9×9 盘面上 116,640 格里有 3 格这样错。
    #   合法对局里对方子必有 ≥1 气、因而总与某个空点连通，所以 stdata 对拍抓不到
    #   它 —— 但本函数吃的是任意盘面数组，必须与官方逐格一致。
    #   官方那个「未经过滤的 vital 初始表」（`board.cpp:2100-2120`，区域头相邻的
    #   链先全部记为 vital）也就只对**真正从空点起头**的区域成立。
    if (b == 0).any():
        has_empty = np.bincount(rid[(b == 0)].reshape(-1),
                                minlength=nreg + 1) > 0
        rid = np.where(has_empty[rid], rid, 0)
    at_least_one_pla = (pc != 0).any(axis=(1, 2))          # board.cpp:2077-2086

    vp_r, vp_c, vital_count = _vital_pairs(b, rid, nreg, pc, nchain, suicide_legal)

    # ---- Benson 定点迭代（board.cpp:2159-2195）--------------------------------
    killed = np.zeros(nchain + 1, dtype=bool)
    borders_non_pass_alive = np.zeros(nreg + 1, dtype=bool)
    while True:
        newly = (~killed) & (vital_count < 2)               # board.cpp:2168
        newly[0] = False
        if not newly.any():
            break
        killed |= newly
        # 新判死的链**周边**的区域：从链上任意点做 4-邻域膨胀，只保留区域点。
        around = np.zeros(b.shape, dtype=bool)
        for dr, dc in _NB4_DIRS:
            around |= _shift4(newly[pc], dr, dc)
        just_bordered = (rid > 0) & around & ~borders_non_pass_alive[rid]
        if not just_bordered.any():
            continue
        # `borders_non_pass_alive` 是**按区域号**的表，写入要落到号上
        newly_bordered_rids = np.unique(rid[just_bordered])
        borders_non_pass_alive[newly_bordered_rids] = True
        # ⚠ 必须拿**区域号**去比 `vp_r`，不能用 `np.flatnonzero(just_bordered)`
        #   —— 那是 `(B,n,n)` 的**扁平下标**，与区域号是两种东西。单样本时两者
        #   数值都 < 361、偶然撞上而「看起来对」；B>1 时扁平下标远大于任何区域号
        #   ⇒ `hit` 恒空 ⇒ Benson 的 vital 传播**整个失效**，且结果随 batch 组成
        #   变化（已实测 135 行里 1 行单算与批算不一致）。
        hit = np.isin(vp_r, newly_bordered_rids)
        if hit.any():
            vital_count -= np.bincount(vp_c[hit], minlength=nchain + 1)
        vital_count[0] = 0

    # ---- 无条件存活子（board.cpp:2202-2211，**允许覆盖**已有值）--------------
    alive = ~killed
    alive[0] = False
    result[alive[pc]] = pla

    # ---- 围空（board.cpp:2214-2243）-------------------------------------------
    # ⚠ 三条判据的顺序就是优先级：前两条**无条件写**（会覆盖对方颜色），
    #   第三条只在 `result` 仍为 C_EMPTY 时写。
    # ⚠ `:2221` 的判据就是 `numInternalSpacesMax2 <= 1`（不是 `< 2` 之外的任何
    #   东西），而 `:2233` **完全没有** `bordersNonPassAlive` 判据 —— 这两条是
    #   官方大量「反直觉」输出的来源，不要"顺手修正"。
    # ⚠ `atLeastOnePla` 是**整盘**级（`board.cpp:2077-2086` 扫全盘），不是逐区域；
    #   它必须作用在**空间掩码**上 —— 区域表是全批一张，直接和 `(B,1)` 广播会
    #   变成 `(B, nreg+1)`，那是彻底错位。
    num_internal = np.zeros(nreg + 1, dtype=np.int64)
    adj_pla = _ndi_binary_dilation(pc != 0, structure=_STRUCT3)
    internal = (rid > 0) & ~adj_pla                  # board.cpp:2052-2054
    if internal.any():
        num_internal = np.minimum(
            2, np.bincount(rid[internal].reshape(-1), minlength=nreg + 1))
    contains_opp = np.zeros(nreg + 1, dtype=bool)   # board.cpp:2056-2057
    opp_stones = b == opp
    if opp_stones.any():
        contains_opp = np.bincount(rid[opp_stones].reshape(-1),
                                   minlength=nreg + 1) > 0

    should_mark_r = ((num_internal <= 1) | (~contains_opp)) & ~borders_non_pass_alive
    should_mark_r[0] = False
    if_empty_r = ~contains_opp                  # board.cpp:2233
    if_empty_r[0] = False
    gate = at_least_one_pla.reshape(-1, 1, 1)
    in_region = rid > 0
    should_mark = should_mark_r[rid] & in_region & gate
    if should_mark.any():
        result[should_mark] = pla
    if_empty = if_empty_r[rid] & in_region & gate & (result == 0)
    if if_empty.any():
        result[if_empty] = pla
    return result


def _independent_life_seki(b, basic, atari):
    """`calculateIndependentLifeAreaHelper` 的双活过滤（`board.cpp:2247-2327`）。

    🔴 **两个触发条件命中任一即整块判双活**（这是官方输出大量中立的原因）：

      ① `board.cpp:2270`：**己方子整链 1 气**（在气）⇒ 属地被当成双活。
      ② `board.cpp:2272-2275`：**接触 dame** —— 任一 4 邻点是「两色都没认领的
         空点」（`colors[adj]==C_EMPTY && basicArea[adj]==C_EMPTY`）。

    命中后沿 `basicArea == pla` 做 4-邻接 flood，**整块**标 `isSeki`
    （`board.cpp:2280-2292`）；最后只输出 `~isSeki` 的块（`:2303-2326`）。

    ⚠ 触发 ② 是**大多数**局面里 area 被抹平的真凶 —— 不是「提死子」。
    ⚠ 官方**既不提子也不做死子判定**：`calculateAreaForPla` 的输入只有
      `colors`，输出直接写 `area`。死子在 area 里就是「对方子落在被对方围的空区
      里」，由 `containsOpp` 与上面的双活过滤自然处理。
    """
    unclaimed = (b == 0) & (basic == 0)               # 两色都没认领的空点 = dame
    dame = _ndi_binary_dilation(unclaimed, structure=_STRUCT3) if unclaimed.any() \
        else np.zeros(b.shape, dtype=bool)
    owned_stone_atari = atari & (b == basic)           # 触发①
    seed = (basic != 0) & (dame | owned_stone_atari)
    if not seed.any():
        return np.zeros(basic.shape, dtype=bool)
    seki = np.zeros(basic.shape, dtype=bool)
    for sign in (1, -1):
        block = basic == sign
        if not block.any() or not (seed & block).any():
            continue
        labelled, num = _ndi_label(block, structure=_STRUCT3)
        hit = np.unique(labelled[seed & block])
        hit = hit[hit > 0]
        if hit.size:
            seki |= np.isin(labelled, hit)
    return seki


def area_ownership_map(boards, *, is_multi_stone_suicide_legal=False,
                      tax_rule=TAX_NONE):
    """返回 `(B, n, n)` int8 的**逐点归属图**（官方 ch18/ch19 的绝对色口径），
    取值 −1/0/+1（−1=白 / 0=中立 / +1=黑）。

    ---- 算法：逐字照抄官方 `Board::calculateAreaForPla`（`board.cpp:1949-2244`）----
    四层（层 1–3 是 `basicArea`，两条官方分支**共用**）：

      1. **链标注**：黑、白各自 4-邻接连通块（`board.cpp` 的 `chain_head` /
         `next_in_chain`）。本仓用 `scipy.ndimage.label`（`go_rules._STRUCT3`）
         —— 该结构元第 0 轴是单位阵 ⇒ 逐样本独立，块号全批唯一。
      2. **Benson 定点迭代**（`board.cpp:1949-2195`，对黑、白各跑一遍）：
         先把「空点 ∪ 对方子」按 4 邻接切成**区域**（⚠ 区域里**可以含对方子**
         —— 与「只 label 空点」不是同一族对象），再算每条链的 **vital 区域数**
         （区域 vital 于某链 ⟺ 区域**每个参与过滤的点**都邻接该链），任一链
         vital 数 < 2 即判死（`:2168`），迭代到不动点。**判死的链不写进 area。**
      3. **无条件存活子 + 围空**（`board.cpp:2202-2243`）：无条件存活的链**整块**
         写自己的颜色（允许覆盖）；随后每个区域按三条判据落笔 ——
         `:2221`「内部空 ≤ 1 且不邻接任何非 pass-alive 子」、
         `:2222`「不含对方子且不邻接任何非 pass-alive 子」，
         两条都**无条件覆盖**；`:2233`「不含对方子」则**只在仍为 C_EMPTY 时写**。
         🔴 **黑先白后**（`board.cpp:1861-1862`）—— `:2237` 的 C_EMPTY 守卫
         依赖它，**顺序不能换**。
         然后 `nonPassAliveStones` 兜底（`board.cpp:1865-1873`）：仍为空的
         **子**归自己的颜色。到此 `basicArea` 完成。
      4. **按 tax 规则分岔**（`nninputs.cpp:2391-2439`）—— 🔴 这是**两个不同
         的官方函数**，不是同一个函数的参数差异：
         · `tax_rule == TAX_NONE` ⇒ 官方走 `Board::calculateArea`
           （`board.cpp:1853-1874`），**没有**第 4 层，直接返回 `basicArea`。
         · `tax_rule ∈ {TAX_SEKI, TAX_ALL}` ⇒ 官方走
           `Board::calculateIndependentLifeArea(keepTerritories=false,
           keepStones=true)`（`board.cpp:1876-1937`），**多一层双活过滤**：
           见 :func:`_independent_life_seki` 的两个触发，命中即整块抹成中立；
           最后 `keepStones=true` 按 `basicArea == colors` 把子补回
           （`board.cpp:1927-1935`）。
         ⚠ 默认 `TAX_NONE` 不是「随便挑的默认值」，而是 `calculate_area` 唯一
           能走到的分支（`AREA+TAX_SEKI/ALL` 在那里抛 `NotImplementedError`），
           所以默认值必须等于 `calculateArea` 的语义，否则生产路径整个是错的。

    ⚠ **全程不读 `ko_loc`**：官方 `calculateArea*` 也一个 ko 都不读
      （劫只影响 ch6/7/8）。
    ⚠ `is_multi_stone_suicide_legal` 对应 `board.cpp:2036`：它决定 Benson 的
      vital 过滤**是否也检查区域里的对方子**。`False`（本仓默认，等于
      `DEFAULT_RULES_FLAGS` 的 bit5=0）时只过滤空点。
      ⚠ 官方真正传进去的是 `nnInputParams.getSuicideLegalForPassAlive(hist)`
      （`nninputs.cpp:964`）= `multiStoneSuicideLegal || alwaysComputePassAlive
      UnderSuicideRules`，后半个来自 `hist.modes`，**19 个全局通道里没有任何一个
      编码它** ⇒ 对 `globalInputNC[:,8]==0` 的行，从 stdata **无法**恢复真值。

    ---- 与 `GoBoard.score()`（`go_rules.py:2261`）的关系：**已脱钩** ----
    🔴 旧版本这里写的是 Tromp-Taylor 区域计分，并有一条
      `sum == GoBoard.score() + komi` 的不变式。那条不变式**已退役**
      （`tests/test_v7_area.py::test_ownership_sum_agrees_with_go_board_score`
      现在断言的是官方语义）：官方 area **不是** Tromp-Taylor ——
      Benson 会杀链、双活过滤会抹整块，两者在真实对局里都会大面积生效。
      `GoBoard.score()` 是 Tromp-Taylor 计分口径，本仓**一个字都没动**
      （spec §7.1 钉死），它与本函数是两个独立的量，不要互相"对齐"。

    ---- 仍然存在、且**本函数无法闭合**的两处不可比 ----
      1. **TERRITORY 口径的行**：官方 `nninputs.cpp:2407-2421` 只在
         `encorePhase >= 2` 时才置 `hasAreaFeature`，而 encore 阶段本仓没有
         （spec §2.6 D2）⇒ 那类行的官方 ch18/19 要么恒 0，要么来自
         `secondEncoreStartColors`（`:2456-2467`）。stdata **给不出**
         `secondEncoreStartColors` ⇒ 不可比。对拍时必须先按官方自己的
         `globalInputNC[:,9]` 把这些行剔掉，否则会把「官方恒 0」误读成
         「我们算错了」（实测 301 行里有 85 行落在这里，剔掉前 row_exact 是
         0.0000、剔掉后是 1.000000）。
      2. **`multiStoneSuicideLegal == false` 的行**：见上面 `getSuicideLegal
         ForPassAlive` 那条 —— 真值不可从 stdata 恢复。
    对拍数字、逐规则组的分解、以及这些分组为什么必须先剔除，见
    `tests/test_v7_area_official.py`。
    """
    b = _as_batch_boards(boards)
    B, n, _ = b.shape
    empty = b == 0

    # ---- 1. 链标注（块号全批唯一：`_STRUCT3` 第 0 轴是单位阵 ⇒ 不跨样本）----
    blk, nblk = _ndi_label(b == 1, structure=_STRUCT3)
    wht, nwht = _ndi_label(b == -1, structure=_STRUCT3)

    basic = np.zeros(b.shape, dtype=np.int8)
    _area_for_pla(b, blk, nblk, 1, is_multi_stone_suicide_legal, basic)
    _area_for_pla(b, wht, nwht, -1, is_multi_stone_suicide_legal, basic)

    # `nonPassAliveStones=true` / basicArea 兜底：仍为空的点归**该点自己的颜色**
    # （`board.cpp:1892-1898`）⇒ 本实现里就是「所有子都进 basicArea」。
    unclaimed_stones = (basic == 0) & ~empty
    basic[unclaimed_stones] = b[unclaimed_stones]

    if tax_rule == TAX_NONE:
        # 官方 `calculateArea`（`board.cpp:1853-1874`）：**没有**双活过滤，
        # basicArea 原样就是结果。`AREA + TAX_NONE` 走的是这一支 ⇒ 下面算气数
        # 是白算，所以这个 early-return 必须留在算 atari **之前**。
        return basic.astype(np.int8, copy=False)

    # ---- 触发①要用的「整链 1 气」（`board.cpp:2270` 的 getNumLiberties）----
    # 气数用 `go_rules._distinct_liberty_counts`（**去重**空点口径，与 17 通道
    # ch10/11|ch14/15 同一份实现，见本模块 docstring 的「不要另写一份块气算法」）。
    atari = np.zeros(b.shape, dtype=bool)
    for labelled, num in ((blk, nblk), (wht, nwht)):
        if not num:
            continue
        libs = _distinct_liberty_counts(labelled, num, empty)
        atari |= (libs == 1)[labelled]

    seki = _independent_life_seki(b, basic, atari)
    own = np.where((basic != 0) & ~seki, basic, np.int8(0))
    # keepStones=true（`board.cpp:1927-1935`）的判据是
    # `basicArea[loc] == colors[loc]` —— **不是**「这里有子」。一颗被对方的
    # 围空分支（`:2221`/`:2222`）改写成对方颜色的己方子，官方**不会**补回，
    # 它就以对方颜色出现在 ch18/19 上；本实现必须照抄这个条件。
    keep = (basic == b) & ~empty
    own[keep] = basic[keep]
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

    # ⚠ 这是 `calculate_area` 里**唯一**新增的一行，且**不是**改分支顺序：
    #   `FLAG_MULTISTONE_SUICIDE`（bit5）本来就住在 `rules_flags` 里
    #   （见下方位布局段），只是此前没人把它解出来喂给 Benson 的 vital 过滤
    #   （官方 `board.cpp:2036`）。分支顺序（TERRITORY 恒 0 / TAX 抛错）与
    #   `nninputs.cpp:2391-2425` 逐行一致，**一个字都没动**。
    own = area_ownership_map(
        b, is_multi_stone_suicide_legal=bool(rules_flags & FLAG_MULTISTONE_SUICIDE))
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


def _history_prefix_ok(my, op, k, hl):
    """第 ``k`` 手是否落在「有效历史前缀」内（官方 ``nninputs.cpp:2511-2556``）。

    🔴 官方的 ch0..4 是**递归嵌套**的：第 2 手的 pass 标志只在
    「第 1 手存在 **且** 第 1 手轮次正确 **且** 第 2 手存在 **且** 第 2 手轮次正确」
    时才写；第 3 手同理要求前两手都满足。⇒ 判定是**前缀**性质，不是逐格独立。

    而落点通道 ch9..13（:2518/:2527/:2536/:2545/:2554）用的是**同一组嵌套
    条件**，只是「非 pass 时写坐标」而非「写 1」—— 所以两侧的「哪些手算数」
    必须一致，否则会出现「第 k+1 手落在 ch9+k 却没有 ch0+k」的矛盾局面。

    轮次正确性由 :data:`_HISTORY_SLOTS` 的 ``('op','my','op','my','op')``
    交错保证（`my_hist`/`op_hist` 本就是相对 ``to_play`` 的，见
    :func:`history_five` 的交错规则表），所以这里只需要长度前缀。

    Args:
        k: 0 起的手序。
        hl: ``(B,)`` 真实可用手数。

    Returns:
        ``(B,)`` bool。
    """
    return np.asarray(hl).reshape(-1) > k


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
                    ⚠ 官方是**递归嵌套**闸门：第 k 手要求前 k-1 手
                    全部存在且轮次正确（:2511-2556），**不是逐格独立**
    5               ``currentSelfKomi(nextPlayer)/20``          局表 `g_komi`
                    clip 到 ``±(bArea+1)``；**符号随 to_play 翻**
    6 / 7           ko 规则：simple=(0,0) / positional=(1,+0.5)
                    / situational=(1,−0.5)                     `rules_flags`
    8               ``multiStoneSuicideLegal``                  `rules_flags`
    9               territory 计分 ⇒ 1（area ⇒ 0）              `rules_flags`
    10 / 11         tax：none=(0,0) / seki=(1,0) / all=(1,1)     `rules_flags`
    12 / 13         ``encorePhase > 0`` / ``> 1``               简单局恒 0
    14              ``passWouldEndPhase``                        `pass_would_end_phase`
    15 / 16         ``playoutDoublingAdvantage``：整块被 ``if(pda!=0)``
                    包住 ⇒ 无 PDA 时**两格都 0**               本仓无 PDA ⇒ 恒 0
    17              ``hasButton``（被 ``if(hist.hasButton)`` 包住）
                                                             `rules_flags`
    18              贴目 × 棋盘奇偶三角波                        `komi_parity_wave`
    ==============  =========================================  ====================

    ⚠ **ch15/16/17 是「条件写入」，不是常量**（`nninputs.cpp:2673-2680`）::

        if(nnInputParams.playoutDoublingAdvantage != 0) {
          rowGlobal[15] = 1.0;
          rowGlobal[16] = 0.5f * playoutDoublingAdvantage; }
        ...
        if(hist.hasButton) rowGlobal[17] = 1.0;

    🔴 读源码时**极易看错**：只看这两行会以为「ch15/ch17 恒 1」。
    实测官方 ``globalInputNC``（2,235 行）：ch15 非零率 **3.85%**、
    ch16 非零率 **3.85%**（两者同步 ⇒ 同一条件）、ch17 非零率 **22.33%**
    ⇒ **都不是恒 1**。本仓无 PDA ⇒ ch15/16 恒 0 才是对的。

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
    area = (BOARD_SIZE * BOARD_SIZE if board_area is None
            else int(np.asarray(board_area).reshape(-1)[0]))

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
    # 🔴 官方是**递归嵌套**闸门（`nninputs.cpp:2511-2556`）：第 k 手只有在
    #     「前面 k-1 手全部存在且轮次正确」时才有资格被检查。我们原来按
    #     `(move<0) & (k < hl)` **逐格独立判断** —— 当中间某手缺失而后面的手
    #     存在时，会在不该有 1 的格上给出 1。
    #     官方结构：if(have1){ if(pla1==opp){ if(pass1) g[0]=1; else ch9;
    #       if(have2){ if(pla2==pla){ ... } } } }
    #     ⇒ 「有效前缀长度」= 第一个不满足条件的手之前的手数。
    for k in range(HISTORY_MOVES):
        out[:, k] = (_history_move(my, op, k) < 0) & _history_prefix_ok(my, op, k, hl)

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
    # 官方 `nninputs.cpp:2660-2665`：ch12 = encorePhase > 0、ch13 = encorePhase > 1
    out[:, 12] = float(encore_phase > 0)
    out[:, 13] = float(encore_phase > 1)

    # ---- ch14 · passWouldEndPhase ------------------------------------------
    # 官方 `nninputs.cpp:2667-2668`
    out[:, 14] = pass_would_end_phase(my, op, hl).astype(np.float32)

# ---- ch15 / ch16 · playoutDoublingAdvantage ---------------------------
    # 官方 `nninputs.cpp:2673-2676`，**整块被 `if(pda != 0)` 包住**：
    #     if(nnInputParams.playoutDoublingAdvantage != 0) {
    #       rowGlobal[15] = 1.0;
    #       rowGlobal[16] = 0.5f * playoutDoublingAdvantage; }
    # ⇒ 无 PDA 时两格都**保持 0**。本仓无 PDA ⇒ 恒 0。
    # ⚠ 官方注释（:2671-2672）说「parameter 15 在非零时训练行为有**不连续**」，
    #   容易误读成「15 恒 1」—— 实际那个不连续正是**由 PDA 是否非零**触发的，
    #   所以 15 与 16 同生共死。本仓无 PDA ⇒ 15 = 0。
    #   （实测官方 globalInputNC：ch15 非零率 3.85%、ch16 非零率 3.85%、
    #     两者完全同步 ⇒ 印证它们由同一个条件写入。）

# ---- ch17 · hasButton --------------------------------------------------
    # 官方 `nninputs.cpp:2679-2680`，**被 `if(hist.hasButton)` 包住** ⇒
    # 标的是「这局的规则里**开了** button」，而非「机制存在」。
    # 实测官方 globalInputNC：ch17 非零率 22.33%，与我们按
    # `rules_flags & FLAG_HAS_BUTTON` 得到的取值域相容。
    out[:, 17] = ((fl & FLAG_HAS_BUTTON) != 0).astype(np.float32)

    # ---- ch18 · 贴目 × 棋盘奇偶三角波 ---------------------------------------
    scoring = SCORING_TERRITORY if (fl & FLAG_SCORING_TERRITORY) != 0 else SCORING_AREA
    out[:, 18] = komi_parity_wave(sk, area, scoring, encore_phase)

    return out