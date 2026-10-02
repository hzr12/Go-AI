"""KataGo ``iterLadders`` 的 NumPy 移植 —— V7 空间通道 **ch14 / 15 / 16 / 17**。

对应 spec（``docs/superpowers/specs/2026-10-01-katago-nbt-tf-design.md``）§2.2
的四行与 §5.2 的移植项清单。因为 ``katago/`` 下**只有 Windows 二进制、没有
``cpp/``**，本移植的权威来源是 KataGo 的**公开源码**（逐字对照，见下"来源"一节），
stdata 是唯一可执行 oracle。

来源（逐字核对的公开源码，v1.18.1 与 master 一致）
----------------------------------------------------
- ``cpp/neuralnet/nninputs.cpp::iterLadders`` —— 遍历盘面，对**气数 ∈ {1,2}** 的
  块各求一次"是否逃不掉的提子"，按块头记忆化（``chainHeadsSolved``）。
- ``cpp/game/board.cpp::Board::searchIsLadderCaptured(loc, defenderFirst, buf)``
  —— 攻守交替的**定深迭代 DFS**（C++ 是手写栈，本移植逐字照搬，包括
  ``MAX_LADDER_SEARCH_NODE_BUDGET`` 与"栈满即判赢"两处上限）。
- ``cpp/game/board.cpp::Board::searchIsLadderCapturedAttackerFirst2Libs``
  —— 对 2 气的块，先试"进攻方各填一个气"，把成功的那一手记为 **working move**。
- ``cpp/game/board.cpp`` 的 ``findLiberties`` / ``findLibertyGainingCaptures`` /
  ``getBoundNumLibertiesAfterPlay`` / ``getNumLibertiesAfterPlay`` /
  ``wouldBeKoCapture`` / ``hasLibertyGainingCaptures`` /
  ``countHeuristicConnectionLibertiesX2`` / ``isLegal`` / ``isIllegalSuicide`` /
  ``playMoveAssumeLegal`` / ``undo``。

通道语义
--------
=====  ==========================================================  ==============
ch     语义                                                       依赖
=====  ==========================================================  ==============
14     **当前盘**上的梯子（``iterLadders`` 的 ``f`` 的第一个副作用）  无（与 to_play 无关）
15     **前一手盘**上的梯子                                      只依赖喂进来的盘面
16     **前二手盘**上的梯子                                      只依赖喂进来的盘面
17     当前盘梯子的 **working-move 位置**（**仅对手方、且 >1 气**）  **依赖 to_play**
=====  ==========================================================  ==============

⚠ **ch14 与 to_play 无关**（``iterLadders`` 的签名里根本没有 player 参数），
**ch17 依赖 to_play**（官方条件是 ``board.colors[loc] == opp && libs > 1``，
``opp = getOpp(nextPlayer)``）。这是与 stdata 对拍时的关键事实。

历史门控（官方语义，**最容易写错的地方**）
----------------------------------------
::

    prevBoard     = (hideHistory || numTurnsOfHistoryIncluded < 1) ? board     : hist.getRecentBoard(1);
    prevPrevBoard = (hideHistory || numTurnsOfHistoryIncluded < 2) ? prevBoard : hist.getRecentBoard(2);

⇒ **历史不足时回退复制，不是置 0**：history=0 ⇒ ch15 = ch14、ch16 = ch15；
history=1 ⇒ ch16 = ch15。本模块以 ``None`` 表示"该级历史不足"，
``ladder_channels`` 负责按上式回退。

⚠ 官方还有一条 ``hideHistory``（对局已结束 / 已过正常阶段 / 保守 pass 等）。
本模块**不建模** ``hideHistory``：它在简单局里恒为 false（它的触发条件全是
encore / 游戏结束相关，spec §2.6 D2 已说明我们没有 encore）。若真被触发，
官方会令 prevPrev 复制的是 ``board`` 而不是 ``prevBoard``——只在同时满足
"hideHistory 且确实有历史"时才与本模块不同。

算法选择的理由（精度优先，但不做逐点 Python 循环）
--------------------------------------------------
需求是"纯 numpy、批量"，又明令"不要逐点 Python 循环"。本移植取的是**按块迭代**
（spec 任务书允许的三选一之一）：

1. **块 / 气 的提取全部 numpy 化**：`_label_chains` 用「横向游程 + 竖向
   union-find」一次算出全盘 `chain_of` / `ch_stones` / `ch_libs`，之后
   `numLiberties(loc)` 是 O(1) 查表，不再有任何 19×19 的 Python 循环。
2. **假设落子 `play` 只重算受影响的那一小块**（新增块 + 4 邻块的气的增减 +
   提子后提子点四周的气的增加），不做全盘重算；`undo` 是 O(1) 快照回滚。
   这一步是速度的关键：一个 19 路 DFS 若每次落子都全盘重算会慢 100 倍。
3. **DFS 本身是「每块一次」**，且绝大多数块在**第一个攻/守节点**就终止
   （见 `search_is_ladder_captured` 的 base cases），不做无谓展开。

⚠ **DFS 结果与「链的遍历顺序」无关**（这是能安全不还原 C++ 的
``next_in_chain`` 环形链表的前提，已逐条核对）：
- 所有 move list 只当**集合**用（去重后填进 buf）；
- 唯一用到顺序的地方是 attacker 节点的启发式重排与"两个气不相邻时砍掉一个"
  —— 二者只改**搜索顺序**，完整搜索下布尔结果与顺序无关；
- ``findLiberties`` 在 2 气的防守节点上只返回**唯一那口气**，与顺序无关；
- ``getNumLibertiesAfterPlay(..., max=3)`` 返回 ``min(真实气数, 3)``，与顺序无关。
唯二残留的顺序依赖是 C++ 自己的两个上限（25000 节点预算 / 栈深上限），
在真实盘面上不可达；**本移植按 C++ 原样保留**这两个上限（含
``ko_loc`` 在预算耗尽时不还原的那处上游疏漏），不为它们做优化。

kwarg 说明
----------
``ko``：官方 ``iterLadders`` 用的是**盘面自带的 ``ko_loc``**（它会影响
``searchIsLadderCapturedAttackerFirst2Libs`` 里 ``isLegal(move0/1, opp)`` 的
劫禁判定）。本模块的必需签名里没有它，故**默认按"无劫"**处理，并额外提供
可选 kwarg ``ko``；不给时的唯一后果是"2 气的块的气恰好是劫点"那种行会分叉
（``searchIsLadderCaptured`` 走 defenderFirst 根节点时会把 ko 清掉，所以
ko 只在这一个分支上有影响）。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
#: V7 固定 19×19（spec §1.3）。
BOARD_SIZE = 19
#: 官方 ``binaryInputNCHWPacked`` 解出来的空间通道里 ladder 的下标。
CH_LADDER, CH_LADDER_PREV, CH_LADDER_PREV2, CH_LADDER_WORKING = 14, 15, 16, 17
#: 输出 dtype。与 ``GoBoard.feature_planes`` / ``feature_planes_batched`` 对齐
#: （见 ``src/game/go_rules.py`` 里那句「与 feature_planes_batched 同一个
#: dtype（fp16）」的注释：两路必须一致，否则 RL/推理走单图、训练走批量会拿到
#: 不同精度的 planes）。取值域只有 {0.0, 1.0}，fp16 逐位无损。
LADDER_DTYPE = np.float16

#: 内部颜色编码，与 KataGo 的 ``Color`` 对齐：``P_BLACK=1 / P_WHITE=2``。
#: 输入盘的 −1 映射到 WHITE（2）、+1 映射到 BLACK（1）。
C_EMPTY, BLACK, WHITE, WALL = 0, 1, 2, 3

#: C++ 的 ``MAX_LADDER_SEARCH_NODE_BUDGET``（``board.cpp`` 原值）。
MAX_LADDER_SEARCH_NODE_BUDGET = 25000


def _grid(n):
    """返回 (n+2,) 的四邻偏移，顺序与 C++ 的 ``adj_offsets[0..3]`` 一致
    （上 / 左 / 右 / 下）。"""
    w = n + 2
    return (-w, -1, 1, w)


# ---------------------------------------------------------------------------
# 块（chain）提取：全 numpy
# ---------------------------------------------------------------------------
def _label_chains(colors, n):
    """一次算出全盘的块 id / 块内点 / 块的气集合。

    做法（全程 numpy，无逐点 Python 循环）：
    1. **横向游程**：同一行里连续同色的点归成一个 run，run 号取其左端点的扁平
       下标。``np.maximum.accumulate`` 让每个点拿到"≤ 它的最大 run 起点"，
       也就是所属 run 的左端点——一次 accumulate 搞定全部 run。
    2. **竖向 union-find**：上下相邻、同色、不同 run 的点对做合并。这是唯一
       需要 Python 的一步，循环次数 = 竖向同色相邻对数（≈ 石子数的一半）。
    3. 块内点 / 气按块 id 排序后 ``searchsorted`` 切段。

    Args:
        colors: ``((n+2)*(n+2),) int8`` 带墙的扁平盘面（内层 n×n，外圈 WALL）。

    Returns:
        ``(chain_of, ch_stones, ch_libs)``：``chain_of`` 是 ``(len(colors),)``
        int32、非石子处 −1；``ch_stones`` / ``ch_libs`` 是按块 id 索引的
        ``list[tuple[int, ...]]`` / ``list[frozenset[int]]``。
    """
    w = n + 2
    nb = _grid(n)
    g = colors.reshape(w, w)
    # ⚠ **必须把墙排除掉**：内部颜色是 0/1/2、墙是 WALL=3，裸写 `g > 0` 会把
    # 整圈墙当成一个连通块（它本来就是连通的），后面每一行的块/气全错。
    stone = (g == BLACK) | (g == WHITE)

    # --- 1. 横向游程 ---
    # ⚠ 切片别写成 `same_left[1:, 1:-1] = g[1:,1:-1] == g[:-1,1:-1]`：那个 `1:` 落在
    # **行**轴上，比的是「正上方」而不是「正左方」，于是横向游程整个判错（一条
    # 横排的 3 子会被拆成 3 个块）。左邻比较必须把 `1:` 放在**列**轴上。
    same_left = np.zeros((w, w), dtype=bool)
    same_left[1:-1, 1:] = g[1:-1, 1:] == g[1:-1, :-1]
    run_start = stone & ~same_left
    lab = np.where(run_start, np.arange(w * w).reshape(w, w), 0)
    # ⚠ `accumulate` 的默认 axis 是 **0**、不是 None！直接对二维数组 accumulate
    # 会变成「每列各自纵向取最大」，横向游程的语义整个丢掉 ⇒ 一条横着的 3 子连成一
    # 排会被拆成 3 个块。必须先 ravel（行主序）再 accumulate。
    lab = np.maximum.accumulate(lab.ravel()).reshape(w, w)
    n_stone = int(stone.sum())
    if n_stone == 0:
        return (np.full(w * w, -1, dtype=np.int32), [], [])
    uniq = np.unique(lab[stone])                 # run 号（升序、无重复）
    n_runs = uniq.size
    remap = np.full(w * w, -1, np.int32)
    remap[uniq] = np.arange(n_runs, dtype=np.int32)
    run_id = np.where(stone, remap[lab], 0).astype(np.int32)

    # --- 2. 竖向 union-find ---
    parent = list(range(n_runs))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:                # 路径压缩
            parent[x], x = root, parent[x]
        return root

    up_run = run_id[:-1, 1:-1].ravel()
    dn_run = run_id[1:, 1:-1].ravel()
    pairs = np.flatnonzero(
        (up_run >= 0) & (dn_run >= 0) & (up_run != dn_run)
        & (g[:-1, 1:-1].ravel() == g[1:, 1:-1].ravel()))
    for i in pairs.tolist():
        a, b = find(int(up_run[i])), find(int(dn_run[i]))
        if a != b:
            parent[max(a, b)] = min(a, b)
    root_of = np.array([find(i) for i in range(n_runs)], dtype=np.int32)

    # --- 3. 分组 ---
    chain_of = np.where(stone, root_of[run_id], -1).astype(np.int32).ravel()
    stone_idx = np.flatnonzero(stone.ravel())
    ci = chain_of[stone_idx]
    n_chains = int(ci.max()) + 1 if ci.size else 0
    order = np.argsort(ci, kind='stable')
    ci_s, si_s = ci[order], stone_idx[order]
    cut = np.searchsorted(ci_s, np.arange(n_chains + 1))
    ch_stones = [tuple(si_s[cut[k]:cut[k + 1]].tolist()) for k in range(n_chains)]

    # 气：每个空点的石子邻居都把该空点算作自己块的气，按 (块, 气点) 去重。
    empty_idx = np.flatnonzero(colors == C_EMPTY)
    le, lc = [], []
    for d in nb:
        nbr = empty_idx + d
        cn = colors[nbr]
        ok = (cn == BLACK) | (cn == WHITE)          # 见上：不能写 `cn > 0`
        # 记账方向别搞反：**空点是气，石子邻居决定气归哪个块**。
        sel = np.flatnonzero(ok)
        le.append(empty_idx[sel])
        lc.append(chain_of[nbr[sel]])
    if le and any(a.size for a in le):
        le = np.concatenate(le)
        lc = np.concatenate(lc)
        o = np.lexsort((le, lc))
        le, lc = le[o], lc[o]
        keep = np.ones(le.size, dtype=bool)
        keep[1:] = (lc[1:] != lc[:-1]) | (le[1:] != le[:-1])
        le, lc = le[keep], lc[keep]
    else:
        le = lc = np.zeros(0, dtype=np.int64)
    if le.size:
        o = np.argsort(lc, kind='stable')
        le, lc = le[o], lc[o]
        cutl = np.searchsorted(lc, np.arange(n_chains + 1))
        ch_libs = [frozenset(le[cutl[k]:cutl[k + 1]].tolist())
                   for k in range(n_chains)]
    else:
        ch_libs = [frozenset() for _ in range(n_chains)]
    return chain_of, ch_stones, ch_libs


def assert_consistent(bd):
    """自检：``colors`` / ``chain_of`` / ``ch_stones`` / ``ch_libs`` 必须互相对得上。

    只给测试用（**不在热路径**）：增量落子这类"局部记账"的代码，账目一旦在某
    个角落错位，症状是**远处**的一个数字偏了（某块气数不对 → 梯子判定翻转），
    极难定位。落子后自检能把它变成一个就地报错。

    做法：从 ``colors`` 重算一遍参考结果（:func:`_label_chains`）并逐项比对。
    """
    n = bd.n
    arr = np.asarray(bd.colors, dtype=np.int8).reshape(n + 2, n + 2)
    ref_cof, ref_stones, ref_libs = _label_chains(arr.reshape(-1), n)
    stone = (arr[1:-1, 1:-1] == BLACK) | (arr[1:-1, 1:-1] == WHITE)
    assert np.array_equal(np.asarray(bd.chain_of).reshape(n + 2, n + 2)[
        1:-1, 1:-1] >= 0, stone), 'chain_of 与 colors 不一致'
    # ⚠ 比**划分**而不是比块 id：增量落子每次都分配新 id（不复用槽位），id 本身
    # 与「从 colors 重算」的紧凑编号没有任何对应关系，只有「谁跟谁同块」要对。
    got = {frozenset(bd.ch_stones[cid]): bd.ch_libs[cid]
           for cid in range(len(bd.ch_stones)) if bd.ch_stones[cid]}
    want = {frozenset(s): l for s, l in zip(ref_stones, ref_libs) if s}
    assert set(got) == set(want), '块的划分不对'
    for k in want:
        assert got[k] == want[k], \
            f'块 {sorted(k)} 的气不对：{sorted(got[k])} vs {sorted(want[k])}'


def _pack(board):
    b = np.asarray(board, dtype=np.int8)
    n = b.shape[-1]
    m = b.shape[-2]
    if m != n:
        raise ValueError(f'只支持方阵盘面，收到 {m}x{n}')
    g = np.full((m + 2, n + 2), WALL, dtype=np.int8)
    g[1:-1, 1:-1] = np.where(b > 0, BLACK, np.where(b < 0, WHITE, C_EMPTY))
    return g.reshape(-1)


def _unpack(colors, n):
    """带墙扁平盘面 → ``(n,n)`` int8（−1/0/1）。"""
    g = colors.reshape(n + 2, n + 2)[1:-1, 1:-1]
    return np.where(g == BLACK, 1, np.where(g == WHITE, -1, 0)).astype(np.int8)


# ---------------------------------------------------------------------------
# 假设落子的盘面（增量 play / 快照 undo）
# ---------------------------------------------------------------------------
class _Board:
    """带增量假设落子的假设盘面。

    只在 :func:`_iter_ladders` 内部当"搜索用的临时盘"用：官方 C++ 也是
    ``Board copy(board)`` + 一路 ``play`` / ``undo``，搜索返回时盘面必须复原。

    ``play`` 只重算受影响的一小块：新增块（含与友方块的合并）、4 邻块的气的
    增减、提子后提子点四周的气的增加——与 C++ ``playMoveAssumeLegal`` 的
    记账逐项对应（``removeChain`` 里的 ``changeSurroundingLiberties(cur, opp, +1)``
    只给**提子方**颜色加气是对的：同色且相邻的块本来就是同一块）。
    ``undo`` 是快照回滚（C++ 的 undo 明确「不保证还原成原来的链表形状」，
    而本实现每次都从 `colors` 重建等价信息，所以无需逐项逆操作）。

    ⚠ **``colors`` 是被「接管并原地修改」的**：第一次 :meth:`play` 就会往调用方
    传进来的那个数组里写子。调用方若还要拿一份**未被搜索污染的根盘**
    （``iterLadders`` 正是要：它要在整个扫描过程中按根盘的颜色判断
    ``colors[loc] == opp``），必须自己留副本。

    ⚠ **内部一律用 Python list 存盘面与块表，不用 numpy 数组。**
    假想落子每次要读写好几十个点，``numpy`` 的标量索引（每次 ~150 ns）比 list
    索引（~40 ns）贵 3~4 倍，而这条路径**每个盘面要走几万次**。实测（306 个
    stdata 盘面）这一项占掉约一半的 ladder 耗时。转换只在 ``__init__`` 发生一次。
    """

    __slots__ = ('n', 'w', 'nb', 'colors', 'chain_of', 'ch_stones', 'ch_libs',
                 'n_chains', 'ko_loc')

    def __init__(self, colors, n):
        self.n = n
        self.w = n + 2
        self.nb = _grid(n)
        chain_of, stones, libs = _label_chains(colors, n)
        self.colors = colors.tolist()
        self.chain_of = chain_of.tolist()
        self.ch_stones = stones
        self.ch_libs = libs
        self.n_chains = len(stones)
        self.ko_loc = -1

    # -- 查询 ---------------------------------------------------------------
    def num_liberties(self, loc):
        return len(self.ch_libs[self.chain_of[loc]])

    def liberties(self, loc):
        return self.ch_libs[self.chain_of[loc]]

    def chain_size(self, loc):
        return len(self.ch_stones[self.chain_of[loc]])

    def num_immediate_liberties(self, loc):
        col = self.colors
        k = 0
        for d in self.nb:
            if col[loc + d] == C_EMPTY:
                k += 1
        return k

    def is_legal(self, loc, pla):
        """``Board::isLegal(loc, pla, isMultiStoneSuicideLegal=false)``。
        梯子搜索里 suicide 永远不相关（官方注释），故 multistone-suicide=false。"""
        if self.colors[loc] != C_EMPTY or loc == self.ko_loc:
            return False
        return not self._is_illegal_suicide(loc, pla)

    def _is_illegal_suicide(self, loc, pla):
        opp = BLACK + WHITE - pla
        col, cof, libs = self.colors, self.chain_of, self.ch_libs
        for d in self.nb:
            c = col[loc + d]
            if c == C_EMPTY:
                return False
            if c == pla:
                if len(libs[cof[loc + d]]) > 1:
                    return False
            elif c == opp:
                if len(libs[cof[loc + d]]) == 1:
                    return False
            # 墙（C_WALL）既不是空、也不是 pla/opp ⇒ 按 C++ 一样什么都不做
        return True

    def has_liberty_gaining_captures(self, loc):
        """``Board::hasLibertyGainingCaptures``：块旁是否有 1 气的对方块。"""
        opp = BLACK + WHITE - self.colors[loc]
        col, cof, libs = self.colors, self.chain_of, self.ch_libs
        for cur in self.ch_stones[cof[loc]]:
            for d in self.nb:
                adj = cur + d
                if col[adj] == opp and len(libs[cof[adj]]) == 1:
                    return True
        return False

    def bound_num_liberties_after_play(self, loc, pla):
        """``Board::getBoundNumLibertiesAfterPlay`` → ``(lowerBound, upperBound)``。

        ⚠ **两处上游写法照抄，改动会直接让梯子判多**：
        - ``potentialLibsFromCaps`` 是「被提块的大小**按重数累加**」；
        - ``numCaps`` / ``numConnectionLibs`` **都按方向数、不对块去重** ——
          同一个块从两个方向相邻就加两遍。
        （对照：``findLibertyGainingCaptures`` 是**去重**的。这两处不一致正是
        上游的原文，本移植保持一致。）
        """
        opp = BLACK + WHITE - pla
        col, cof, libs = self.colors, self.chain_of, self.ch_libs
        num_immediate = num_caps = pot_from_caps = 0
        num_conn = max_conn = 0
        for d in self.nb:
            adj = loc + d
            c = col[adj]
            if c == C_EMPTY:
                num_immediate += 1
            elif c == opp:
                n = len(libs[cof[adj]])
                if n == 1:
                    num_caps += 1
                    pot_from_caps += len(self.ch_stones[cof[adj]])
            elif c == pla:
                conn = len(libs[cof[adj]]) - 1
                num_conn += conn
                if conn > max_conn:
                    max_conn = conn
        lower = num_caps + max(max_conn, num_immediate)
        upper = num_immediate + pot_from_caps + num_conn
        return lower, upper

    def count_heuristic_connection_liberties_x2(self, loc, pla):
        """``Board::countHeuristicConnectionLibertiesX2``。

        ⚠ 这里**不去重**块（C++ 直接 ``FOREACHADJ`` 累加），照抄。
        """
        col, cof, libs = self.colors, self.chain_of, self.ch_libs
        total = 0
        for d in self.nb:
            adj = loc + d
            if col[adj] == pla:
                nl = len(libs[cof[adj]])
                if nl > 1:
                    total += nl * 2 - 3
        return total

    def would_be_ko_capture(self, loc, pla):
        """``Board::wouldBeKoCapture``。"""
        if self.colors[loc] != C_EMPTY:
            return False
        opp = BLACK + WHITE - pla
        col, cof, libs = self.colors, self.chain_of, self.ch_libs
        cap_loc = -1
        for d in self.nb:
            adj = loc + d
            c = col[adj]
            if c != WALL and c != opp:
                return False
            if c == opp and len(libs[cof[adj]]) == 1:
                if cap_loc != -1:
                    return False
                cap_loc = adj
        if cap_loc == -1:
            return False
        return len(self.ch_stones[cof[cap_loc]]) == 1

    def num_liberties_after_play(self, loc, pla, cap):
        """``Board::getNumLibertiesAfterPlay(loc, pla, max)``。

        ⚠ 返回值与遍历顺序无关：返回的是 ``min(去重后的真实气数, cap)``，
        所以本移植不还原 C++ 的 ``next_in_chain`` 环形顺序也不影响结果。
        """
        opp = BLACK + WHITE - pla
        col, cof, libs = self.colors, self.chain_of, self.ch_libs
        found = set()
        captured_heads = set()
        num = 0
        for d in self.nb:
            adj = loc + d
            c = col[adj]
            if c == C_EMPTY:
                found.add(adj)
                num += 1
                if num >= cap:
                    return cap
            elif c == opp and len(libs[cof[adj]]) == 1:
                found.add(adj)
                num += 1
                if num >= cap:
                    return cap
                captured_heads.add(cof[adj])

        def would_be_empty(lc):
            c = col[lc]
            return c == C_EMPTY or (c == opp and cof[lc] in captured_heads)

        seen_conn = set()
        for d in self.nb:
            adj = loc + d
            if col[adj] == pla:
                cid = cof[adj]
                if cid in seen_conn:
                    continue
                seen_conn.add(cid)
                for cur in self.ch_stones[cid]:
                    for d2 in self.nb:
                        pl = cur + d2
                        if pl != loc and would_be_empty(pl) and pl not in found:
                            found.add(pl)
                            num += 1
                            if num >= cap:
                                return cap
        return num

    # -- 落子 ---------------------------------------------------------------
    def play(self, loc, pla):
        """假设落子合法地落一手，返回可交给 :meth:`undo` 的快照记录。"""
        # ⚠ ``ch_stones`` / ``ch_libs`` 必须**拷贝列表头**：_play 会原地 append
        # 新块并覆写槽位，若只存引用，undo 之后列表会残留被撤销那一手新建的槽位
        # （长度/内容都多于盘面实际有的块），后续 `_find_liberties` 就会按着
        # 一个不存在的块去数气 ⇒ 结果静默错。槽里的 tuple / frozenset 都是
        # 不可变的，共享是安全的。
        rec = (self.colors[:], self.chain_of[:],
               list(self.ch_stones), list(self.ch_libs),
               self.n_chains, self.ko_loc)
        self._play(loc, pla)
        return rec

    def undo(self, rec):
        colors, chain_of, stones, libs, n_chains, ko = rec
        self.colors = colors
        self.chain_of = chain_of
        self.ch_stones = stones
        self.ch_libs = libs
        self.n_chains = n_chains
        self.ko_loc = ko

    def _play(self, loc, pla):
        nb = self.nb
        col = self.colors
        cof = self.chain_of
        opp = BLACK + WHITE - pla
        col[loc] = pla

        merged = [loc]
        merged_cids = set()
        opp_cids, opp_seen = [], set()
        for d in nb:
            adj = loc + d
            c = col[adj]
            if c == pla:
                cid = cof[adj]
                if cid not in merged_cids:
                    merged_cids.add(cid)
                    merged.extend(self.ch_stones[cid])
            elif c == opp:
                cid = cof[adj]
                if cid not in opp_seen:
                    opp_seen.add(cid)
                    opp_cids.append(cid)

        nid = self.n_chains
        self.n_chains = nid + 1
        ch_libs = self.ch_libs
        for q in merged:
            cof[q] = nid
        while len(self.ch_stones) <= nid:
            self.ch_stones.append(())
            ch_libs.append(frozenset())
        self.ch_stones[nid] = tuple(merged)

        # 4 邻的对方块：吃掉 loc 这口气；气归零 ⇒ 提子
        captured = []
        for cid in opp_cids:
            s = set(ch_libs[cid])
            s.discard(loc)
            ch_libs[cid] = frozenset(s)
            if not s:
                captured.append(cid)

        num_captured = 0
        for cid in captured:
            stones = self.ch_stones[cid]
            num_captured += len(stones)
            for q in stones:
                col[q] = C_EMPTY
                cof[q] = -1

        # 新块的气 = 提子之后、与之相邻的空点（含刚被提掉的点）
        libs = set()
        for q in merged:
            for d in nb:
                a = q + d
                if col[a] == C_EMPTY:
                    libs.add(a)
        ch_libs[nid] = frozenset(libs)

        # 提子点四周的、非新块的块，各多一口气（仅提子方颜色，见 _Board docstring）。
        # ⚠ 判据用 `chain_of >= 0` 而不是「颜色非空」：**墙点也是非空的**
        # （C_WALL=3），而它的 chain_of 是 -1 ⇒ 会写到 ch_libs[-1] 去，
        # 把最后一个块的气的集合改坏，而且只在提子发生在**边线**时才触发。
        for cid in captured:
            for q in self.ch_stones[cid]:
                for d in nb:
                    other = cof[q + d]
                    if 0 <= other < nid:
                        ch_libs[other] = ch_libs[other] | {q}

        # 劫点：提且只提 1 子、且落下的子是孤子且只有 1 气
        self.ko_loc = -1
        if (num_captured == 1 and len(merged) == 1
                and len(ch_libs[nid]) == 1):
            # C++ 的 possible_ko_loc 是「被提块里与落点相邻的那个点」
            for cid in captured:
                for q in self.ch_stones[cid]:
                    if abs(q - loc) in (1, self.w):
                        self.ko_loc = q


# ---------------------------------------------------------------------------
# 两个 ladder 判定函数（逐字移植 board.cpp）
# ---------------------------------------------------------------------------
def _put(buf, k, q):
    """把 ``q`` 写到 ``buf[k]``（按**下标**写，不是 append）。

    ⚠ 必须按 C++ 的语义按下标写：``iterLadders`` 对 1 气的块**不清** ``buf``，
    而 DFS 的根层又总是从下标 0 开始写。若这里用 ``append``，上一条链留下的
    长度会让根层写歪，链的结果直接错。
    """
    while len(buf) <= k:
        buf.append(-1)
    buf[k] = q


def _find_liberties(bd, loc, buf, buf_start, buf_idx):
    """``Board::findLiberties``：块的气（去重）写入 ``buf[buf_idx:]``，返回个数。

    去重范围是 ``buf[buf_start : buf_idx + 已写个数]``（与 C++ 的
    ``for(j = bufStart; j < bufIdx+numFound; j++)`` 一致）。
    """
    col = bd.colors
    nb = bd.nb
    seen = set(buf[buf_start:buf_idx])
    added = 0
    for cur in bd.ch_stones[bd.chain_of[loc]]:
        for d in nb:
            q = cur + d
            if col[q] == C_EMPTY and q not in seen:
                seen.add(q)
                _put(buf, buf_idx + added, q)
                added += 1
    return added


def _find_liberty_gaining_captures(bd, loc, buf, buf_start, buf_idx):
    """``Board::findLibertyGainingCaptures``：块旁 1 气对方块的气（即提子点）。"""
    opp = BLACK + WHITE - bd.colors[loc]
    col, cof, libs, nb = bd.colors, bd.chain_of, bd.ch_libs, bd.nb
    seen_cap = set()
    seen_pt = set(buf[buf_start:buf_idx])
    added = 0
    for cur in bd.ch_stones[cof[loc]]:
        for d in nb:
            adj = cur + d
            if col[adj] != opp:
                continue
            ocid = cof[adj]
            if ocid in seen_cap or len(libs[ocid]) != 1:
                continue
            seen_cap.add(ocid)
            for q in libs[ocid]:
                if q not in seen_pt:
                    seen_pt.add(q)
                    _put(buf, buf_idx + added, q)
                    added += 1
    return added


def search_is_ladder_captured(bd, loc, defender_first, buf):
    """``Board::searchIsLadderCaptured`` 的逐字移植（含那两个上限）。

    ⚠ 与 C++ 一致地保留两处上游写法：
    - 栈满（``stackSize``）⇒ 直接**判赢**（``returnValue = true``）；
    - 25000 节点预算耗尽 ⇒ 判**不赢**，并且**不还原** ``ko_loc``（上游漏了
      ``ko_loc = ko_loc_saved``）。两个上限在真实盘面上都不可达；照抄是为了
      万一触及时行为仍然一致。
    """
    if bd.colors[loc] not in (BLACK, WHITE):
        return False
    nl = bd.num_liberties(loc)
    if nl > 2 or (defender_first and nl > 1):
        return False

    pla = bd.colors[loc]
    opp = BLACK + WHITE - pla
    ko_saved = bd.ko_loc
    if defender_first:
        bd.ko_loc = -1

    w = bd.w
    stack_size = bd.n * bd.n * 3 // 2 + 1
    move_starts = [0] * stack_size
    move_lens = [0] * stack_size
    move_cur = [0] * stack_size
    records = [None] * stack_size
    stack_idx = 0
    node_count = 0
    return_value = False
    from_deeper = False
    move_cur[0] = -1

    while True:
        if stack_idx <= -1:
            bd.ko_loc = ko_saved
            return return_value
        if stack_idx >= stack_size - 1:
            return_value, from_deeper = True, True
            stack_idx -= 1
            continue
        if node_count >= MAX_LADDER_SEARCH_NODE_BUDGET:
            stack_idx -= 1
            while stack_idx >= 0:
                bd.undo(records[stack_idx])
                stack_idx -= 1
            return False

        is_defender = ((stack_idx % 2) == 0) if defender_first \
            else ((stack_idx % 2) == 1)

        if move_cur[stack_idx] == -1:
            libs = bd.num_liberties(loc)
            # base cases（逐条照抄）
            if (not is_defender) and libs <= 1:
                return_value, from_deeper = True, True
                stack_idx -= 1
                continue
            if (not is_defender) and libs >= 3:
                return_value, from_deeper = False, True
                stack_idx -= 1
                continue
            if is_defender and libs >= 2:
                return_value, from_deeper = False, True
                stack_idx -= 1
                continue
            if is_defender and bd.ko_loc != -1:
                return_value, from_deeper = False, True
                stack_idx -= 1
                continue

            start = move_starts[stack_idx]
            if is_defender:
                mlen = _find_liberty_gaining_captures(bd, loc, buf, start, start)
                mlen += _find_liberties(bd, loc, buf, start, start + mlen)
                # C++ 注释：list 最后一个元素恒为防守块那口唯一的气
                lb, ub = bd.bound_num_liberties_after_play(buf[start + mlen - 1], pla)
                if lb >= 3:
                    return_value, from_deeper = False, True
                    stack_idx -= 1
                    continue
                if mlen == 1 and ub <= 1:
                    return_value, from_deeper = True, True
                    stack_idx -= 1
                    continue
            else:
                mlen = _find_liberties(bd, loc, buf, start, start)
                libs0 = bd.num_immediate_liberties(buf[start])
                libs1 = bd.num_immediate_liberties(buf[start + 1])

                # 双劫必死特例
                if (libs0 == 0 and libs1 == 0
                        and bd.would_be_ko_capture(buf[start], opp)
                        and bd.would_be_ko_capture(buf[start + 1], opp)):
                    if (bd.num_liberties_after_play(buf[start], pla, 3) <= 2
                            and bd.num_liberties_after_play(buf[start + 1], pla, 3) <= 2):
                        if not bd.has_liberty_gaining_captures(loc):
                            return_value, from_deeper = True, True
                            stack_idx -= 1
                            continue

                delta = buf[start + 1] - buf[start]
                adjacent = delta in (-w, -1, 1, w)
                if not adjacent:
                    if libs0 >= 3 and libs1 >= 3:
                        return_value, from_deeper = False, True
                        stack_idx -= 1
                        continue
                    elif libs0 >= 3:
                        mlen = 1
                    elif libs1 >= 3:
                        buf[start] = buf[start + 1]
                        mlen = 1
                if mlen > 1:
                    libs0 = libs0 * 2 + bd.count_heuristic_connection_liberties_x2(
                        buf[start], pla)
                    libs1 = libs1 * 2 + bd.count_heuristic_connection_liberties_x2(
                        buf[start + 1], pla)
                    if libs1 > libs0:
                        buf[start], buf[start + 1] = buf[start + 1], buf[start]
            move_lens[stack_idx] = mlen
            move_cur[stack_idx] = 0
        else:
            if from_deeper:
                bd.undo(records[stack_idx])
            if is_defender and not return_value:
                from_deeper = True
                stack_idx -= 1
                continue
            if (not is_defender) and return_value:
                from_deeper = True
                stack_idx -= 1
                continue
            move_cur[stack_idx] += 1

        if move_cur[stack_idx] >= move_lens[stack_idx]:
            return_value = is_defender
            from_deeper = True
            stack_idx -= 1
            continue

        move = buf[move_starts[stack_idx] + move_cur[stack_idx]]
        p = pla if is_defender else opp
        if not bd.is_legal(move, p):
            # 非法着法与"失败"同等处理：不回一层，直接试下一手
            return_value = is_defender
            from_deeper = False
            continue

        records[stack_idx] = bd.play(move, p)
        node_count += 1
        stack_idx += 1
        move_cur[stack_idx] = -1
        move_starts[stack_idx] = move_starts[stack_idx - 1] + move_lens[stack_idx - 1]
        move_lens[stack_idx] = 0


def search_is_ladder_captured_attacker_first_2_libs(bd, loc, buf, working_moves):
    """``Board::searchIsLadderCapturedAttackerFirst2Libs`` 的逐字移植。

    Returns:
        ``bool``（是否进攻方先手就能提死）。为 ``True`` 时 ``working_moves`` 被
        填成进攻方成功的那几手（ch17 的来源）。
    """
    if bd.colors[loc] not in (BLACK, WHITE):
        return False
    if bd.num_liberties(loc) != 2:
        return False
    pla = bd.colors[loc]
    opp = BLACK + WHITE - pla
    _find_liberties(bd, loc, buf, 0, 0)
    move0, move1 = buf[0], buf[1]

    def attempt(m):
        if not bd.is_legal(m, opp):
            return False
        rec = bd.play(m, opp)
        try:
            return search_is_ladder_captured(bd, loc, True, buf)
        finally:
            bd.undo(rec)

    move0_works = attempt(move0)
    move1_works = attempt(move1)
    if move0_works or move1_works:
        working_moves.clear()
        if move0_works:
            working_moves.append(move0)
        if move1_works:
            working_moves.append(move1)
        return True
    return False


# ---------------------------------------------------------------------------
# iterLadders
# ---------------------------------------------------------------------------
def _flat_to_padded(flat, n):
    """盘面**紧凑**扁平下标（stride = n，本仓 ``ko`` 列的口径）→ 带墙下标。

    ⚠ 两套下标不能混：外部（npz 的 ``ko`` 列、``pos_hash``、``to_v7_labels``）用的
    都是 ``r*n + c`` 的紧凑下标，而本模块内部（与 C++ 的 ``Loc`` 对齐）用的是
    带墙的 ``(r+1)*(n+2) + (c+1)``。19 路的 ``ko`` 点 ``(13,17)`` 紧凑下标是 264，
    带墙下标是 312 —— 传错的话劫禁判定会打在**完全无关的点上**，而且不报错。
    """
    if flat is None or flat < 0:
        return -1
    r, c = divmod(int(flat), n)
    if not (0 <= r < n and 0 <= c < n):
        return -1
    return (r + 1) * (n + 2) + (c + 1)


def _iter_ladders(board, opp, ko_loc=-1):
    """``iterLadders`` + 官方 ``addLadderFeature`` 的合并移植。

    Args:
        board: ``(n,n)`` int8（−1/0/1）。
        opp: **对手方颜色**（``+1`` 表示对手是黑）。官方 ``addLadderFeature``
            的 ch17 条件是 ``colors[loc] == opp && libs > 1``，而 ch15/ch16 的
            lambda 明确把它 ``(void)`` 掉了 ⇒ **只有当前盘那一路用 opp**。
        ko_loc: 当前盘面自带的 simple-ko 点，**紧凑**扁平下标（``r*n + c``，
            −1 = 无）。内部会转成带墙下标，见 :func:`_flat_to_padded`。

    Returns:
        ``(ladder, working)`` 两个 ``(n,n) bool``：ch14 与 ch17 的内容。
    """
    n = board.shape[-1]
    w = n + 2
    colors = _pack(board)
    # `_Board` 会原地改 `colors`，所以给它一份私有副本；`colors` 保持是
    # **未被任何假想落子污染的根盘**（下面整段扫描都要按它判 `c == opp`）。
    bd = _Board(colors.copy(), n)
    bd.ko_loc = _flat_to_padded(ko_loc, n)

    # 输出也用带墙的扁平下标（与 loc/working_moves 同一套坐标），最后切掉边框。
    ladder = np.zeros(w * w, dtype=bool)
    working = np.zeros(w * w, dtype=bool)
    buf, working_moves = [], []

    # ---- 只遍历「候选块」，不遍历 19×19 的每一个点 ----
    #
    # 官方的写法是扫 361 个点、每点查一次块表，但一个盘面上**气数 ∈ {1,2} 的块
    # 只有十来块**（实测 306 个 stdata 盘面：3214/306 ≈ 10.5 块/盘）。按块迭代
    # 把 361 次 Python 循环降到 ~10 次。
    #
    # ⚠ **按块迭代不会改变结果**，已逐条核对（这也是「块表一次算好、整盘共享」
    #   的前提）：
    #   1. 同块所有子共享 `libs` 与 `colors`，所以 `addLadderFeature` 的两个条件
    #      在块内逐位相同 ⇒ ch14 置位等价于「整块置位」，幂等。
    #   2. `workingMoves` 的跨块残留**只在 1 气的块上可能泄进 ch17**，而那里有
    #      官方那个 `libs > 1` 挡着（见下）；2 气的块在求解前都先
    #      `workingMoves.clear()`。
    #   3. 每个块只解一次（官方的 `chainHeadsSolved` 记忆化在这里变成「每块只
    #      循环一次」）。
    #   仍按「块内最小扁平下标」排序 = 官方 y-major 扫描里这个块**第一次**被碰到
    #   的次序，逐位对齐上游。
    #
    # ⚠⚠⚠ **循环里必须每步重新读 `bd.ch_libs`，不能把它缓存成局部变量。**
    # `_play` 是**原地**改 `ch_libs` 这个 list 的，而 `undo` 是把 `bd.ch_libs`
    # **重新绑定**到一份快照 —— 于是缓存下来的那个局部变量从此指向一份
    # 「被搜索改过、且再也回不去」的表。第二条链读到的气数就是第一手假想落子
    # 留下的值，`libs` 判错 ⇒ 分支走错。症状极隐蔽：**候选筛选是在循环前做的，
    # 所以筛出来的块是对的，只是每块的结论错了**，而且只在「某些盘面上恰好发生」
    # 才显形（实测 250 行里错 30 行）。`_Board` 的 docstring 已写明它是
    # 「快照回滚」，这条是从那儿直接推出来的。
    ch_libs = bd.ch_libs
    cand = [cid for cid, s in enumerate(bd.ch_stones)
            if s and (len(ch_libs[cid]) == 1 or len(ch_libs[cid]) == 2)]
    cand.sort(key=lambda cid: min(bd.ch_stones[cid]))

    for cid in cand:
        stones = bd.ch_stones[cid]
        libs = len(bd.ch_libs[cid])
        loc = stones[0]
        c = colors[loc]
        if libs == 1:
            # 官方此处**不清** workingMoves（沿用上一次的值）。真正的挡板是下面
            # 那个 `libs > 1`，所以这处上游的不清空对本通道无影响，但分支结构
            # 仍照抄，免得日后有人"顺手修正"掉语义。
            laddered = search_is_ladder_captured(bd, loc, True, buf)
        else:
            del buf[:]
            working_moves.clear()
            laddered = search_is_ladder_captured_attacker_first_2_libs(
                bd, loc, buf, working_moves)
        if laddered:
            ladder[np.asarray(stones, dtype=np.int64)] = True
            # 官方 addLadderFeature 的**两个**条件，逐字照抄：
            #   if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1)
            # ⚠ `libs > 1` 这半个条件**不能省**。1 气的块走的是
            # `searchIsLadderCaptured` 分支，而官方在那里**故意不清**
            # workingMoves（沿用上一条链算出来的值）——真正的挡板就是这个
            # `> 1`。省掉它，一个「1 气、颜色 == opp」的梯子块会把**上一条
            # 2 气链的残留 working move** 泄进 ch17（实测凭空多出约 4% 的错行）。
            # 这与 to_play 无关，是纯粹的移植漏项。
            if c == opp and libs > 1:
                for m in working_moves:
                    working[m] = True
    inner = (slice(1, -1), slice(1, -1))
    return ladder.reshape(w, w)[inner], working.reshape(w, w)[inner]


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------
def ladder_channels(boards, to_play, prev_board=None, prev_prev_board=None,
                    ko=None):
    """给定当前盘 + 前一手盘 + 前二手盘，产出 ch14/15/16/17。

    Args:
        boards: ``(B,19,19)`` int8，取值 −1/0/1（−1 = 白 / +1 = 黑）。
        to_play: ``(B,)`` int8 ±1，轮到谁落子（``+1`` = 黑先行）。
            ⚠ **只有 ch17 依赖它**（官方条件 ``colors[loc] == opp && libs > 1``）；
            ch14/15/16 完全与它无关。
        prev_board: ``(B,19,19)`` int8 或 **None**。None = 历史不足一手 ⇒
            按官方语义回退**复制当前盘**（于是 ch15 == ch14）。
        prev_prev_board: 同上。None = 历史不足二手 ⇒ 回退复制 ``prev_board``
            解析后的结果（history=0 ⇒ ch16 == ch15；history=1 ⇒ ch16 == ch15）。
            ⚠ 官方是**按行**判定 ``numTurnsOfHistoryIncluded`` 的；本签名用
            ``None`` 表示**整批**不足。逐行混合的场景由调用方（B6 的邻行 gather）
            自行把「历史不足」的行填成该行当前盘再整批传进来。
        ko: 可选 ``(B,)`` int16，当前盘面**自带的** simple-ko 点（−1 = 无）。
            官方 ``iterLadders`` 用的就是盘面自己的 ``ko_loc``，它只在一处生效：
            ``searchIsLadderCapturedAttackerFirst2Libs`` 里对进攻方那两手的
            ``isLegal`` 劫禁判定。**不给（None）一律按「无劫」处理**，唯一后果是
            「2 气的块的气恰好是劫点」的行可能分叉（走 defenderFirst 根节点的
            那条路会把 ko 清掉，所以不受影响）。

    Returns:
        ``(B,4,19,19)`` **float16**（``LADDER_DTYPE``），依次是
        ch14 / ch15 / ch16 / ch17，取值 {0.0, 1.0}。
        ⚠ dtype 选 float16 是为了与 ``GoBoard.feature_planes`` /
        ``feature_planes_batched`` 逐位对齐（``src/game/go_rules.py`` 里那句
        「两路必须一致，否则 RL/推理走单图与训练走批量会拿到不同精度的 planes」）。
        fp16 对 {0,1} 逐位无损。若下游想要 bool，用 ``!= 0`` 即可。

    示例:
        >>> ch = ladder_channels(boards, to_play)          # ch[:,0]==ch[:,1]==ch[:,2]
        >>> ch = ladder_channels(boards, to_play, prev)    # 仅 ch16 == ch15
    """
    b = np.asarray(boards, dtype=np.int8)
    if b.ndim != 3 or b.shape[1] != b.shape[2]:
        raise ValueError(f'boards 形状不符：{b.shape}，期望 (B,19,19)')
    nb = b.shape[0]
    if b.shape[1] != BOARD_SIZE:
        raise ValueError(f'只支持 {BOARD_SIZE}x{BOARD_SIZE}，收到 {b.shape[1]}')
    tp = np.asarray(to_play, dtype=np.int8).reshape(-1)
    if tp.size != nb:
        raise ValueError(f'to_play 长度 {tp.size} != B {nb}')

    # 官方历史门控：不足时**回退复制**，不是置 0。
    pb = b if prev_board is None else np.asarray(prev_board, dtype=np.int8)
    p2b = pb if prev_prev_board is None else np.asarray(prev_prev_board, dtype=np.int8)
    for name, arr in (('prev_board', pb), ('prev_prev_board', p2b)):
        if arr.shape != b.shape:
            raise ValueError(f'{name} 形状不符：{arr.shape}，期望 {b.shape}')

    out = np.zeros((nb, 4, BOARD_SIZE, BOARD_SIZE), dtype=LADDER_DTYPE)
    kos = _ko_list(ko, nb)

    # ch15/ch16 只要 ch14（官方那两个 lambda 把 workingMoves (void) 掉了），
    # ch14/ch17 一起出。历史不足时 pb / p2b 就是**同一个数组对象** ⇒ 直接抄已经
    # 算好的那一份，省掉一轮完整的 ladder 搜索（history=0 时三块盘面退化成
    # 一个，这正是本函数最常见的调用形态）。
    cur = _current_row(b, tp, kos)
    out[:, 0] = cur[:, 0]
    out[:, 3] = cur[:, 1]
    out[:, 1] = cur[:, 0] if pb is b else _ladder_row(pb, kos)
    # ⚠⚠ ch16 的兜底是 **out[:,1]（即 ch15 的内容）**，不是 `cur[:,0]`（ch14 的）。
    # 官方第二行回退到的是 `prevBoard`：
    #     prevPrevBoard = (... < 2) ? prevBoard : hist.getRecentBoard(2);
    # 而 history=1 时 `pb` 正是**真正的上一手盘**、不等于当前盘 ⇒ 此处抄 ch14 会
    # 给出「当前盘」的 ch16 而不是「上一手盘」的，与官方正好差一手。
    #
    # ⚠ 别把这个判断**第二次**从历史参数推一遍（`if prev_prev_board is None: 抄 ch15
    # else: 抄 ch16`）。上面 `pb` / `p2b` 已经按官方原文解析过一次历史门控，这里只
    # 认「两个盘面是不是同一个数组对象」这一个事实。**同一口径写两遍必然发散** ——
    # `src/game/go_rules.py:170-177` 记的就是这类（批量侧的 liberty 口径与单图侧各写
    # 一份，U 形块两个 bucket 同时漏）。上一版把兜底抄成了 `cur[:,0]`，正是第二遍
    # 推导出的错。
    out[:, 2] = out[:, 1] if p2b is pb else _ladder_row(p2b, kos)
    return out


def _ko_list(ko, nb):
    """``ko`` kwarg → 长度 ``nb`` 的 ``int`` 列表（None ⇒ 全 −1）。"""
    if ko is None:
        return [-1] * nb
    k = np.asarray(ko).reshape(-1)
    if k.size == 1:
        return [int(k[0])] * nb
    if k.size != nb:
        raise ValueError(f'ko 长度 {k.size} != B {nb}')
    return [int(v) for v in k]


def _current_row(boards, to_play, kos):
    """``(B,19,19)`` → ``(B,2,19,19) bool``，分别是 ch14 与 ch17。

    ch17 需要**逐行**的 ``to_play``（官方条件 ``colors[loc] == opp``），所以这里
    一行一行地调 :func:`_iter_ladders`。
    """
    out = np.zeros((boards.shape[0], 2, boards.shape[1], boards.shape[2]),
                   dtype=bool)
    for i in range(boards.shape[0]):
        # 输入用 ±1，内部用 C_BLACK/C_WHITE ⇒ 这里做一次换色。
        opp = BLACK if int(to_play[i]) < 0 else WHITE
        lad, work = _iter_ladders(boards[i], opp, ko_loc=kos[i])
        out[i, 0] = lad
        out[i, 1] = work
    return out


def _ladder_row(boards, kos):
    """``(B,19,19)`` → ``(B,19,19) bool``：只要 ch14。

    ⚠ ``opp`` 在 ch15/ch16 上**没有作用**（官方那两个 lambda 明确把
    ``workingMoves`` 与颜色条件一起 ``(void)`` 掉了），故这里传 `-1` 占位 ——
    传什么都不会改变输出，但传一个固定值能省掉逐行读 to_play。
    """
    out = np.zeros((boards.shape[0], boards.shape[1], boards.shape[2]),
                   dtype=bool)
    for i in range(boards.shape[0]):
        out[i] = _iter_ladders(boards[i], 0, ko_loc=kos[i])[0]
    return out