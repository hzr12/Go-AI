"""P4.3 — 17 通道特征平面补齐 + `n_channels` 参数化。

被测对象：`src/game/go_rules.py` 的「特征平面」段（`feature_planes` /
`feature_planes_batched` / 模块级 `_neighbor_all` / `_check_n_channels`）与
`src/data/dataset.py` 的 `SupervisedDataset(n_channels=...)`。

⚠ **本文件不共享 `tests/test_go_rules_parity.py` / `test_go_action_api.py` 的任何
   夹具或语料**：同一姿势、各自重建（手工摆子 + `resync_hash()` 显式接管）。

**为什么期望值要自己写一遍**：本文件里凡是「通道该等于什么」的断言，期望值都由
本文件内的参考实现 `_ref_planes_0_11` / `_ref_eye` / `_ref_liberties` / `_ref_legal`
给出，**不调任何生产函数当期望**。参考实现刻意选了与生产不同的**算法**
（union-find vs 栈式 flood fill；「落子-提子-查气」的字面模拟 vs 生产用的
邻接块代数变形），否则两边同错同源、测试只会绿。唯一允许引用生产函数的对照是
「通道 8 = `get_legal_moves()`」这条**已声明的契约**（见 `test_channel8_*`），
它比的是两个不同的生产函数（合法性 API vs 特征构造器），不是自己跟自己比。

覆盖面（12 条，5 路 / 9 路，CPU、秒级）：
  1. `test_plane_count_and_indices`              17 通道 / float32 / (17,n,n) / 边界值
  2. `test_channels_0_to_11_unchanged`           前 12 通道 == 本文件自写的参考实现
  3. `test_n_channels_prefix_is_invariant`       12 通道的输出 == 17 通道的前 12 格
  4. `test_eye_planes`                           真单点眼 / 边 / 角 / 有敌邻 / 两格眼 /
                                                  双方眼 / 空盘 —— 8 个子情形
  5. `test_liberty_two_planes`                   气=1 / 气=2 / 气≥3 × 双方 + **块口径**
  6. `test_ko_plane`                             有/无 ko_point + `play()` 造的真劫 +
                                                  `clone()` 后仍在
  7. `test_batched_matches_scalar`               逐位相同的通道（0-7/9-15，含 10/11/14/15）
  8. `test_channel8_*`                           通道 8 的契约 + 禁自杀分歧
  9. `test_batched_liberty_planes_match_scalar_on_random_boards`
                                                  块气四通道在随机局面上**无条件**一致
 10. `test_u_shaped_group_liberty_two`           U 形块：去重气数=2 → 14/15 标、10/11 不标
 11. `test_shared_liberty_between_groups`        一个气点被**两个不同块**共享 → 各算 1
 12. `test_liberty_count_is_distinct_not_incidence`
                                                  直接钉死口径：取「去重数」不取「入射数」
 13. `test_dataset_n_channels_*` / `test_n_channels_12_does_not_compute_the_new_planes`

**块气的口径（本文件第 10-12 条用例守的东西）**：
`气数 = 与该块相邻的「去重」空点个数`。一个空点被**同一块**的两颗子夹住时，对该块
**只算 1 气**；被**不同块**共享时，对每块各算 1 气。
⚠ 批量路径曾用「入射计数」（对块内每颗子数自己的空邻点再求和），U 形块会被多算，
  于是通道 10/11 与 14/15 都可能漏标 —— 那是 bug 已在 P4.3-fix 订正（修批量侧，
  标量侧的 `set` 去重一直是对的）。旧行为的复现板与新行为的复现板**同一个**，
  见 `test_u_shaped_group_liberty_two`。

**已文档化的分歧**（两条，逐条用测试钉住，见 `test_channel8_*` / `test_ko_plane` ④）：
  - 通道 8：单图走 `get_legal_moves()`（含禁自杀 + PSK），批量版 = 空点 ∧ 排除 ko。
  - 通道 16：依赖调用方传 `ko`；`ko=None` → 恒全零（单图版永远读得到 `ko_point`）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import SupervisedDataset  # noqa: E402
from src.game.go_rules import GoBoard, _check_n_channels, _neighbor_all  # noqa: E402

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------- #
# 参考实现（**不调任何生产函数**）                                            #
# --------------------------------------------------------------------------- #

def _flood(board, r0, c0):
    """与生产 `_group_liberty_count` 不同源的连通块枚举：广度优先 + 队列。"""
    n = board.shape[0]
    color = board[r0, c0]
    out = set()
    frontier = [(r0, c0)]
    out.add((r0, c0))
    while frontier:
        nxt = []
        for r, c in frontier:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in out \
                        and board[nr, nc] == color:
                    out.add((nr, nc))
                    nxt.append((nr, nc))
        frontier = nxt
    return out


def _ref_libs(board, seed):
    """气点**集合**（去重口径）。空集合 = 该块 0 口气 = 已被提。"""
    n = board.shape[0]
    libs = set()
    for r, c in _flood(board, *seed):
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == 0:
                libs.add((nr, nc))
    return libs


def _ref_liberties(board, seed):
    """气数 = **去重后**的气点个数（与生产的 set 口径一致，但枚举算法不同）。"""
    return len(_ref_libs(board, seed))


def _ref_incidence_liberties(board, seed):
    """气数 = **入射计数**（旧批量路径的口径，本文件唯一的「反面参考」）。

    对块内每颗子数自己的空邻点个数，再对整块求和 —— 也就是数「(子, 气) 关联
    **次数**」。**这是错的**口径（共享气点会多算），存在的唯一目的，是让
    `test_liberty_count_is_distinct_not_incidence` 能断言「去重数 != 入射数」，
    从而证明那条用例**不是空转**。算法与 `_ref_liberties` 共用 `_flood`，
    但累加的对象完全不同（`set.add` vs 计数器自增）。
    """
    n = board.shape[0]
    total = 0
    for r, c in _flood(board, *seed):
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == 0:
                total += 1
    return total


def _ref_lib_masks(board, to_play):
    """通道 10/11 的参考实现：**union-find**（生产是栈式 flood fill）。"""
    n = board.shape[0]
    parent = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # 路径压缩
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    stones = [(r, c) for r in range(n) for c in range(n) if board[r, c] != 0]
    for s in stones:
        parent.setdefault(s, s)
    for r, c in stones:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == board[r, c]:
                union((r, c), (nr, nc))

    libs = {}
    for r, c in stones:
        root = find((r, c))
        libs.setdefault(root, set())
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == 0:
                libs[root].add((nr, nc))

    my = np.zeros((n, n), bool)
    op = np.zeros((n, n), bool)
    for r, c in stones:
        root = find((r, c))
        if len(libs[root]) != 1:
            continue
        (my if board[r, c] == to_play else op)[r, c] = True
    return my, op


def _ref_lib_masks_all(board, to_play):
    """通道 **10/11/14/15** 的参考实现（去重口径），返回 4 张掩码。

    枚举用本文件的 `_flood`（广度优先 + 队列）+ `_ref_libs`（`set`），
    与生产的栈式 flood fill / 4 方向链式去重是不同算法。通道 0-9 / 12/13 / 16
    不在这里（各自另有参考实现）。
    """
    n = board.shape[0]
    out = tuple(np.zeros((n, n), bool) for _ in range(4))
    seen = set()
    for r in range(n):
        for c in range(n):
            if board[r, c] == 0 or (r, c) in seen:
                continue
            grp = _flood(board, r, c)
            seen |= grp
            nlibs = len(_ref_libs(board, (r, c)))
            if nlibs not in (1, 2):
                continue
            slot = (0 if nlibs == 1 else 2) + (0 if board[r, c] == to_play else 1)
            for x, y in grp:
                out[slot][x, y] = True
    return out


def _ref_eye(board, color):
    """通道 12/13 的参考实现：**字面三行判定**（生产是移位比较）。"""
    n = board.shape[0]
    out = np.zeros((n, n), bool)
    for r in range(1, n - 1):            # 严格内点，边界/角落直接不进循环
        for c in range(1, n - 1):
            if board[r, c] != 0:
                continue
            if all(board[r + dr, c + dc] == color for dr, dc in DIRS):
                out[r, c] = True
    return out


def _ref_legal(board, to_play=1):
    """通道 8 的参考实现：**字面模拟**（落子 → 提对方无气块 → 查己方块气）。

    与生产 `get_legal_moves()` 的代数变形（「邻接敌块气==1 或 邻接同色块气>=2 或
    有空邻点」）是两个不同算法。`to_play` 决定落子的颜色 —— 白方执子时
    「填真眼」的自杀点与黑方**不同**，写死成黑子会让参考实现系统性放行。

    ⚠ PSK 在本参考里**恒不触发**，所以调用方必须保证盘面是用
    `resync_hash()` 接管的（历史 = {当前局面}，而任何一手落子都会改变染色，
    不可能命中）。语料里凡是走 `play()` 造出来的局面**不满足**这个前提，
    不能拿本函数当期望 —— 用例注释逐条标了。
    """
    n = board.shape[0]
    foe = -to_play
    out = np.zeros((n, n), bool)
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            trial = board.copy()
            trial[r, c] = to_play
            for dr, dc in DIRS:          # 提掉邻接的对方无气块
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and trial[nr, nc] == foe:
                    grp = _flood(trial, nr, nc)
                    if not _ref_libs(trial, (nr, nc)):
                        for x, y in grp:
                            trial[x, y] = 0
            if _ref_libs(trial, (r, c)):
                out[r, c] = True
    return out


def _ref_planes_0_11(board, my_hist, op_hist, to_play):
    """通道 0-11 的参考实现，形状 (12, n, n) float32。"""
    n = board.shape[0]
    ref = np.zeros((12, n, n), np.float32)
    ref[0] = (board == to_play)
    ref[4] = (board == -to_play)
    for src, base in ((my_hist, 1), (op_hist, 5)):
        for k in range(3):
            mv = src[k] if k < len(src) else -1
            if mv is not None and mv >= 0:
                ref[base + k][mv // n, mv % n] = 1.0
    ref[8] = _ref_legal(board, to_play)
    ref[9] = float(to_play)
    my, op = _ref_lib_masks(board, to_play)
    ref[10] = my
    ref[11] = op
    return ref


# --------------------------------------------------------------------------- #
# 夹具                                                                       #
# --------------------------------------------------------------------------- #

def _board_from(stones, n=5, to_play=1):
    """手工摆子 + `resync_hash()` 显式接管（PSK 因此恒不触发，可作通道 8 期望）。"""
    b = GoBoard(n)
    for r, c, v in stones:
        b.board[r, c] = v
    b.current_player = to_play
    b.resync_hash()
    return b


def _random_board(rng, n=9, density=0.45, to_play=1):
    arr = rng.integers(-1, 2, size=(n, n)).astype(np.int8)
    arr[rng.random((n, n)) > density] = 0
    b = GoBoard(n)
    b.board = arr
    b.current_player = to_play
    b.resync_hash()
    return b


def _has_shared_liberty(board):
    """本盘面是否存在「同一块内两个子共享一个气点」。

    存在 → 批量版的气数会**多算一**（见模块 docstring 的分歧条）。用它把
    「通道 10/11/14/15 应当逐位相同」这个断言的前提**显式化**，而不是靠随机
    局面碰运气：谓词由本文件的 union-find 参考实现算出。
    """
    n = board.shape[0]
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    stones = [(r, c) for r in range(n) for c in range(n) if board[r, c] != 0]
    for s in stones:
        parent.setdefault(s, s)
    for r, c in stones:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == board[r, c]:
                union((r, c), (nr, nc))
    libs = {}
    for r, c in stones:
        libs.setdefault(find((r, c)), set())
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == 0:
                libs[find((r, c))].add((nr, nc))
    counted = {}
    for r, c in stones:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == 0:
                counted[(nr, nc)] = counted.get((nr, nc), 0) + 1
    for (r, c), cnt in counted.items():
        if cnt >= 2:
            owners = set()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and board[nr, nc] != 0:
                    owners.add(find((nr, nc)))
            if len(owners) == 1:
                return True
    del libs
    return False


# --------------------------------------------------------------------------- #
# 1. 通道数 / 形状 / 边界                                                       #
# --------------------------------------------------------------------------- #

def test_plane_count_and_indices():
    """17 通道、float32、(17,n,n)，且 `n_channels` 越界必须抛而不是静默裁。"""
    for n in (5, 9, 19):
        b = GoBoard(n)
        my_mv, op_mv = 3, 7 % (n * n)
        p = b.feature_planes([my_mv, -1, -1], [-1, -1, op_mv], 1)
        assert p.shape == (17, n, n), f"{n} 路形状错: {p.shape}"
        assert p.dtype == np.float32, f"{n} 路 dtype 错: {p.dtype}"
        # 值域只允许 {0,1}（通道 9 是 ±1 的常数，单独断言）
        for idx in list(range(8)) + list(range(10, 17)):
            vals = set(np.unique(p[idx]).tolist())
            assert vals <= {0.0, 1.0}, f"通道 {idx} 出现意外取值 {vals}"
        assert set(np.unique(p[9]).tolist()) == {1.0}
        # 历史通道的下标：己方最近一手在 idx1、对手最近一手在 idx5；
        # 这里另给一手「次新」在 idx2 / idx6，把顺序也钉住
        p2 = b.feature_planes([my_mv, 1, -1], [op_mv, 1, -1], 1)
        assert p2[1, my_mv // n, my_mv % n] == 1.0 and p2[2, 1 // n, 1 % n] == 1.0
        assert p2[3].sum() == 0.0
        assert p2[5, op_mv // n, op_mv % n] == 1.0 and p2[6, 1 // n, 1 % n] == 1.0
        assert p2[7].sum() == 0.0

    # 批量版
    pb = GoBoard.feature_planes_batched(
        np.zeros((3, 5, 5), np.int8), np.full((3, 3), -1, np.int16),
        np.full((3, 3), -1, np.int16), np.array([1, -1, 1], np.int8))
    assert pb.shape == (3, 17, 5, 5) and pb.dtype == np.float32

    # 越界
    for bad in (11, 18, 0, -1):
        with pytest.raises(ValueError):
            GoBoard(5).feature_planes([], [], 1, n_channels=bad)
        with pytest.raises(ValueError):
            GoBoard.feature_planes_batched(
                np.zeros((1, 5, 5), np.int8), [[-1] * 3], [[-1] * 3], [1],
                n_channels=bad)
    _check_n_channels(12)     # 不抛
    _check_n_channels(17)


# --------------------------------------------------------------------------- #
# 2. 通道 0-11 与本文件自写的参考实现逐位相同                                   #
# --------------------------------------------------------------------------- #

def test_channels_0_to_11_unchanged():
    """前 12 通道必须与**本文件内**的参考实现逐位相同（参考实现不调生产函数）。"""
    cases = [
        # 空盘
        ([], [-1, -1, -1], [-1, -1, -1], 1),
        # 单子：气=4 → 10/11 全零；0/4 生效
        ([(2, 2, 1)], [-1, -1, -1], [-1, -1, -1], 1),
        # 气=1 的块（(1,1) 黑，四邻里 (1,2) 空、其余白）
        ([(1, 1, 1), (0, 1, -1), (2, 1, -1), (1, 0, -1)],
         [-1, -1, -1], [-1, -1, -1], 1),
        # 双方各有子块 + 历史 + **白方执子**（通道 0/4/9/10/11 的「己方/对方」会翻转）
        ([(0, 0, 1), (1, 1, 1), (4, 4, -1), (3, 4, -1), (0, 4, -1)],
         [0, 1, -1], [24, -1, -1], -1),
        # 4×4 满盘（无空点 → 通道 8 全零，且必然出现 0 气的死块）
        ([(r, c, 1 if (r + c) % 2 == 0 else -1)
          for r in range(4) for c in range(4)], [-1, -1, -1], [-1, -1, -1], 1),
    ]
    for stones, mh, oh, tp in cases:
        b = _board_from(stones, n=5, to_play=tp)
        p = b.feature_planes(mh, oh, tp)
        ref = _ref_planes_0_11(b.board, mh, oh, tp)
        for idx in range(12):
            assert np.array_equal(p[idx], ref[idx]), (
                f"通道 {idx} 与参考实现不符（stones={stones}, to_play={tp}）\n"
                f"实得:\n{p[idx].astype(int)}\n期望:\n{ref[idx].astype(int)}")

    # 随机局面再扫 20 局 9 路（PSK 恒不触发：resync_hash 接管）
    rng = np.random.default_rng(20260927)
    for k in range(20):
        b = _random_board(rng, n=9, to_play=1 if k % 2 else -1)
        mh = [int(x) for x in rng.integers(-1, 81, size=3)]
        oh = [int(x) for x in rng.integers(-1, 81, size=3)]
        p = b.feature_planes(mh, oh)
        ref = _ref_planes_0_11(b.board, mh, oh, b.current_player)
        assert np.array_equal(p[:12], ref), f"随机局面 #{k} 的 0-11 通道与参考不符"


# --------------------------------------------------------------------------- #
# 3. `n_channels` 是「前缀」，12 通道 = 17 通道的前 12 格                          #
# --------------------------------------------------------------------------- #

def test_n_channels_prefix_is_invariant():
    """12 通道的输出必须**逐位等于** 17 通道的前 12 格（两条构造路径都是）。

    ⚠ 这是「默认 12 保零回归」在 feature 侧的核心不变式：一旦有人把新通道插到
      0-11 **中间**（而不是追加到尾部），本用例立刻红。
    """
    rng = np.random.default_rng(11)
    for n in (5, 9, 19):
        b = _random_board(rng, n=n)
        mh = [int(x) for x in rng.integers(-1, n * n, size=3)]
        oh = [int(x) for x in rng.integers(-1, n * n, size=3)]
        full = b.feature_planes(mh, oh)
        for k in range(12, 18):
            cut = b.feature_planes(mh, oh, n_channels=k)
            assert cut.shape == (k, n, n)
            assert np.array_equal(cut, full[:k]), f"{n} 路 n_channels={k} 不是前缀"

    # 批量版
    boards = rng.integers(-1, 2, size=(6, 9, 9)).astype(np.int8)
    mh = rng.integers(-1, 81, size=(6, 3)).astype(np.int16)
    oh = rng.integers(-1, 81, size=(6, 3)).astype(np.int16)
    tp = np.array([1, -1] * 3, np.int8)
    ko = np.array([0, -1, 40, -1, 80, -1], np.int16)
    full = GoBoard.feature_planes_batched(boards, mh, oh, tp, ko)
    for k in (12, 13, 14, 15, 16, 17):
        cut = GoBoard.feature_planes_batched(boards, mh, oh, tp, ko, n_channels=k)
        assert cut.shape == (6, k, 9, 9)
        assert np.array_equal(cut, full[:, :k]), f"批量版 n_channels={k} 不是前缀"


# --------------------------------------------------------------------------- #
# 4. 眼位通道 12/13                                                             #
# --------------------------------------------------------------------------- #

def test_eye_planes():
    """8 个子情形：真单点眼 / 边 / 角 / 有敌邻 / 两格眼 / 双方眼 / 空盘 / 非空点。

    ⚠ 每个子情形都同时对**通道 12 与 13** 断言，并额外与本文件的 `_ref_eye`
      参考实现（字面三行判定）对拍 —— 生产用移位比较、参考用 `all(...)`，
      两个算法同错同源的概率极低。
    """
    n = 5
    # ① 真单点黑眼 (2,2)：四邻 (1,2)(3,2)(2,1)(2,3) 全黑，(2,2) 空
    stones = [(1, 2, 1), (3, 2, 1), (2, 1, 1), (2, 3, 1)]
    b = _board_from(stones, n)
    p = b.feature_planes([], [], 1)
    assert p[12, 2, 2] == 1.0, "真单点黑眼 (2,2) 必须标出来"
    assert p[12].sum() == 1.0, f"只应有一个黑眼，实得 {int(p[12].sum())} 个"
    assert p[13].sum() == 0.0

    # ② 边点 (0,2)：盘内三邻 (1,2)(0,1)(0,3) 全黑，(0,2) 空 —— **不是眼**
    stones = [(1, 2, 1), (0, 1, 1), (0, 3, 1)]
    p = _board_from(stones, n).feature_planes([], [], 1)
    assert p[12, 0, 2] == 0.0, "第一行的点永远不是眼（严格内点）"

    # ③ 角点 (0,0)：盘内两邻 (1,0)(0,1) 全黑 —— **不是眼**
    stones = [(1, 0, 1), (0, 1, 1)]
    p = _board_from(stones, n).feature_planes([], [], 1)
    assert p[12, 0, 0] == 0.0, "角落点永远不是眼"

    # ④ 内点但有一个敌邻：(1,2) 的四邻 (0,2)黑(2,2)黑(1,1)黑(1,3)白 —— **不是眼**
    stones = [(0, 2, 1), (2, 2, 1), (1, 1, 1), (1, 3, -1)]
    p = _board_from(stones, n).feature_planes([], [], 1)
    assert p[12, 1, 2] == 0.0, "有一个敌邻就不是己方眼"
    assert p[13].sum() == 0.0, "白方只一颗子，撑不出眼"

    # ⑤ 两格眼：黑围出 (2,1)(2,2) 两个相邻空点 —— **两个都不是眼**
    stones = [(1, 1, 1), (1, 2, 1), (1, 3, 1), (2, 0, 1), (2, 3, 1),
              (3, 1, 1), (3, 2, 1), (3, 3, 1)]
    p = _board_from(stones, n).feature_planes([], [], 1)
    assert p[12, 2, 1] == 0.0 and p[12, 2, 2] == 0.0, "两格眼不算单点眼"
    assert p[12].sum() == 0.0

    # ⑥ 双方眼：7 路盘上黑眼 (3,3) + 白眼 (1,1)（两个都是真内点眼）
    m = 7
    stones = [(2, 3, 1), (4, 3, 1), (3, 2, 1), (3, 4, 1),      # 黑围 (3,3)
              (0, 1, -1), (2, 1, -1), (1, 0, -1), (1, 2, -1),  # 白围 (1,1)
              (5, 5, 1)]                                       # 一颗无关黑子
    p = _board_from(stones, m).feature_planes([], [], 1)
    assert p[12, 3, 3] == 1.0, "(3,3) 应是黑方的真眼"
    assert p[13, 1, 1] == 1.0, "(1,1) 应是白方的真眼"
    assert p[12].sum() == 1.0 and p[13].sum() == 1.0
    # 5 路盘里白方**不可能**有眼（唯一的内点 (2,2) 必须被黑子占住）
    p5 = _board_from([(1, 3, 1), (3, 3, 1), (3, 1, 1), (3, 3, 1), (2, 2, 1)],
                     5).feature_planes([], [], 1)
    assert p5[13].sum() == 0.0, "5 路盘的唯一内点是黑子，白眼必须全零"

    # ⑦ 空盘：12/13 全零
    p = GoBoard(5).feature_planes([], [], 1)
    assert p[12].sum() == 0.0 and p[13].sum() == 0.0

    # ⑧ 非空点永远不是眼：把 (2,2) 填上自己的子，四邻仍全黑
    stones = [(1, 2, 1), (3, 2, 1), (2, 1, 1), (2, 3, 1), (2, 2, 1)]
    p = _board_from(stones, n).feature_planes([], [], 1)
    assert p[12, 2, 2] == 0.0, "有子的点不是眼位"

    # 与参考实现全盘对拍（含随机局面）
    rng = np.random.default_rng(7)
    for k in range(20):
        bb = _random_board(rng, n=9, to_play=1 if k % 2 else -1)
        tp = bb.current_player
        pp = bb.feature_planes([], [], tp)
        assert np.array_equal(pp[12], _ref_eye(bb.board, tp)), f"随机 #{k} 通道12"
        assert np.array_equal(pp[13], _ref_eye(bb.board, -tp)), f"随机 #{k} 通道13"

    # `_neighbor_all` 的结构性契约：任意盘口下四边恒为 False
    for n2 in (1, 2, 3, 4, 7, 19):
        msk = np.ones((n2, n2), bool)
        got = _neighbor_all(msk)
        assert got[0, :].sum() == 0 and got[-1, :].sum() == 0
        assert got[:, 0].sum() == 0 and got[:, -1].sum() == 0
        msk3 = np.ones((3, n2, n2), bool)
        g3 = _neighbor_all(msk3)
        assert g3[:, 0, :].sum() == 0 and g3[:, :, -1].sum() == 0


# --------------------------------------------------------------------------- #
# 5. 气=2 通道 14/15（块口径）                                                  #
# --------------------------------------------------------------------------- #

def test_liberty_two_planes():
    """气=1 / 气=2 / 气>=3 × 双方，并证明 14/15 是**块**掩码而不是点掩码。

    ⚠ 关键子情形 ⑥：两个黑子各自只有 1 口气、但**整块**有 2 口气 ——
      若 14/15 被写成「按点数气」就会红。这是裁定 ② 的可执行形式。
    """
    n = 5
    # ① 气=1（黑单子 (1,1)：(0,1)(1,0)(1,2) 白、(2,1) 空）
    b = _board_from([(1, 1, 1), (0, 1, -1), (1, 0, -1), (1, 2, -1)], n)
    p = b.feature_planes([], [], 1)
    assert p[10, 1, 1] == 1.0 and p[10].sum() == 1.0
    assert p[14].sum() == 0.0

    # ② 气=2（黑单子 (1,1)：(0,1)(1,0) 白、(1,2)(2,1) 空）
    b = _board_from([(1, 1, 1), (0, 1, -1), (1, 0, -1)], n)
    p = b.feature_planes([], [], 1)
    assert p[14, 1, 1] == 1.0, f"气=2 的黑子必须落进通道 14，实得\n{p[14].astype(int)}"
    assert p[10].sum() == 0.0, "气=2 的块不该同时落进通道 10"
    assert _ref_liberties(b.board, (1, 1)) == 2

    # ③ 气>=3：黑单子只被一个白子挡
    b = _board_from([(1, 1, 1), (0, 1, -1)], n)
    p = b.feature_planes([], [], 1)
    assert p[10].sum() == 0.0 and p[14].sum() == 0.0

    # ④ 白方对称：气=2 的白块落进通道 15
    b = _board_from([(1, 1, -1), (0, 1, 1), (1, 0, 1)], n)
    p = b.feature_planes([], [], 1)
    assert p[15, 1, 1] == 1.0, "气=2 的白块必须落进通道 15"
    assert p[11].sum() == 0.0

    # ⑤ 一个块里同时有气=1 和气=2 的两个独立块，掩码不能串
    stones = [(1, 1, 1), (0, 1, -1), (1, 0, -1), (1, 2, -1),          # 气=1
              (3, 3, 1), (2, 3, -1), (3, 2, -1)]                        # 气=2
    b = _board_from(stones, n)
    p = b.feature_planes([], [], 1)
    assert p[10, 1, 1] == 1.0 and p[10, 3, 3] == 0.0
    assert p[14, 3, 3] == 1.0 and p[14, 1, 1] == 0.0

    # ⑥ **块口径**：黑块 {(1,2),(1,3),(2,3)}，整块只有 (2,2) 一口气，
    #    而 (2,2) 同时贴着 (1,2) 与 (2,3) 两颗子
    stones = [(0, 2, -1), (0, 3, -1), (1, 1, -1), (1, 4, -1), (2, 4, -1),
              (3, 3, -1),
              (1, 2, 1), (1, 3, 1), (2, 3, 1)]
    b = _board_from(stones, n)
    p = b.feature_planes([], [], 1)
    assert _ref_liberties(b.board, (1, 2)) == 1, "参考实现也必须认定整块只有 1 口气"
    assert p[10, 1, 2] == 1.0 and p[10, 1, 3] == 1.0 and p[10, 2, 3] == 1.0, (
        "整块气=1 → 三颗子都要进通道 10（按点数的写法会漏）")
    assert p[14].sum() == 0.0, "整块气=1，不该有任何子落进通道 14"

    # ⑦ 整块气=2 的多子块：黑 {(1,1),(1,2)}，(1,1) 的 2 口气与 (1,2) 的 2 口气
    #    有重叠，整块仍只有 2 口
    stones = [(0, 1, -1), (0, 2, -1), (1, 0, -1), (1, 3, -1),
              (1, 1, 1), (1, 2, 1)]
    b = _board_from(stones, n)
    p = b.feature_planes([], [], 1)
    assert _ref_liberties(b.board, (1, 1)) == 2
    assert p[14, 1, 1] == 1.0 and p[14, 1, 2] == 1.0, "整块气=2 → 两颗子都进通道 14"
    assert p[10].sum() == 0.0, "单子各有 2 口气，整块也是 2 口气，不该进通道 10"

    # ⑧ 双方对称的白块：白 {(1,1),(1,2)} 气=2 → 通道 15；黑墙气=5 → 10/14 都空
    stones = [(0, c, 1) for c in range(5)] + [(1, 0, 1), (1, 3, 1),
                                              (1, 1, -1), (1, 2, -1)]
    b = _board_from(stones, n)
    p = b.feature_planes([], [], 1)
    assert _ref_liberties(b.board, (1, 1)) == 2
    assert p[15, 1, 1] == 1.0 and p[15, 1, 2] == 1.0
    assert p[11].sum() == 0.0, "白块气=2，不该进通道 11"
    assert p[14].sum() == 0.0, "气=2 的白块不该落进通道 14（那是己方的槽）"
    assert p[10].sum() == 0.0, "黑墙气=5，不该进通道 10"


# --------------------------------------------------------------------------- #
# 6. 劫禁点通道 16                                                             #
# --------------------------------------------------------------------------- #

def test_ko_plane():
    """有/无 ko_point、`play()` 造出的**真劫**、`clone()` 后仍在。"""
    n = 5
    # ① 无劫 → 通道 16 全零
    b = GoBoard(n)
    b.resync_hash()
    assert b.ko_point == -1
    assert b.feature_planes([], [], 1)[16].sum() == 0.0
    # 落子（非劫）后仍然全零
    b.play(2 * n + 2)
    assert b.ko_point == -1
    assert b.feature_planes([], [], 1)[16].sum() == 0.0

    # ② 有劫 → 恰有一个 1，落在 ko_point 的扁平坐标上
    b = GoBoard(n)
    b.ko_point = 2 * n + 1
    b.resync_hash()
    p = b.feature_planes([], [], 1)
    assert p[16].sum() == 1.0
    assert p[16].flat[b.ko_point] == 1.0
    assert p[16, 2, 1] == 1.0
    # 克隆体仍然带着这个信息位（clone 逐字段拷贝 ko_point）
    c = b.clone()
    assert c.ko_point == b.ko_point
    assert np.array_equal(c.feature_planes([], [], 1), p), "clone() 后 17 通道必须逐位相同"
    # 落子后 ko_point 被 play() 改写 → 通道 16 跟着变
    b.play(4 * n + 4)
    assert b.feature_planes([], [], 1)[16].sum() == 0.0

    # ③ `play()` 造出的**真劫**：白 (1,1) 的唯一气是 (1,2)；黑下 (1,2) 提子成劫
    ko_board = GoBoard(n)
    for r, c in [(0, 1), (2, 1), (1, 0)]:
        ko_board.board[r, c] = 1        # 黑的墙，把白 (1,1) 憋成 1 口气
    for r, c in [(1, 1), (0, 2), (2, 2), (1, 3)]:
        ko_board.board[r, c] = -1       # 白：(1,1) 待提，(0,2)(2,2)(1,3) 让黑子落子后只剩 1 气
    ko_board.resync_hash()
    assert _ref_liberties(ko_board.board, (1, 1)) == 1, "白 (1,1) 必须恰好 1 口气"
    assert ko_board.ko_point == -1
    mv = 1 * n + 2
    assert ko_board.play(mv), "提子的一手必须合法"
    assert ko_board.ko_point == 1 * n + 1, \
        f"成劫判据应给出 (1,1)，实得 {ko_board.ko_point}"
    assert ko_board.board[1, 1] == 0, "白 (1,1) 应被提掉"
    p = ko_board.feature_planes([], [], 1)
    assert p[16].sum() == 1.0 and p[16, 1, 1] == 1.0
    assert ko_board.clone().feature_planes([], [], 1)[16].sum() == 1.0

    # ④ 通道 16 与通道 8 相互独立：把 ko 点排除出通道 8 后，16 仍然标着它
    pb = GoBoard.feature_planes_batched(
        ko_board.board[None], [[-1] * 3], [[-1] * 3], [1], [ko_board.ko_point])[0]
    assert pb[16, 1, 1] == 1.0, "批量版通道 16 必须标出传入的 ko 点"
    assert pb[8, 1, 1] == 0.0, "同一个点还被排除在通道 8 之外（遗留近似）"


# --------------------------------------------------------------------------- #
# 7. 批量版 vs 单图版：应当逐位相同的通道                                       #
# --------------------------------------------------------------------------- #

# 两条路径**必须逐位相同**的通道。P4.3-fix 起块气四通道（10/11/14/15）也在内 ——
# 它们共用同一份「去重空点数」口径（批量 `_distinct_liberty_counts` ↔ 单图
# `_group_liberty_count`），所以这里**不挂任何前提**（旧版曾用 `_has_shared_liberty`
# 当门，现在那条门已经不需要了）。
_ALWAYS_IDENTICAL = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)


def _pair(b, mh, oh, to_play, ko=None):
    """跑两条路径并返回 (单图 planes, 批量 planes[0])。ko 缺省用 board.ko_point。"""
    if ko is None:
        ko = b.ko_point
    one = b.feature_planes(mh, oh, to_play)
    many = GoBoard.feature_planes_batched(
        b.board[None], [list(mh)], [list(oh)], [to_play], [ko])[0]
    return one, many


def test_batched_matches_scalar():
    """通道 0-7 / 9-15 两条路径**必须逐位相同**（随机局面 + 手工局面）。

    ⚠ **P4.3-fix 起 10/11/14/15 也在逐位比较之列**（旧版只比 12/13，因为批量侧的
      块气口径当时是入射计数、会在共享气点处分叉 —— 那是 bug，已修）。
      通道 8 / 16 的**结构性**分歧见 §「已文档化的分歧」。
    """
    rng = np.random.default_rng(4242)
    cases = []
    # 手工局面：单点眼 / 双方眼 / 满盘 / 单子 / 边上活棋
    cases.append(_board_from([(1, 2, 1), (3, 2, 1), (2, 1, 1), (2, 3, 1)], 5, 1))
    cases.append(_board_from([(1, 3, 1), (3, 3, 1), (3, 1, 1), (3, 5, 1),
                              (5, 1, -1), (5, 5, -1), (4, 2, -1), (4, 4, -1)], 7, 1))
    cases.append(_board_from([(r, c, 1 if (r + c) % 2 == 0 else -1)
                              for r in range(4) for c in range(4)], 5, -1))
    cases.append(_board_from([(0, 0, 1)], 5, 1))
    cases.append(_board_from([], 5, -1))                      # 空盘
    cases.append(_board_from([(2, 2, 1), (2, 3, 1)], 9, 1))   # 9 路
    for k in range(12):
        cases.append(_random_board(rng, n=9, to_play=1 if k % 3 else -1))

    for bi, b in enumerate(cases):
        n = b.board_size
        tp = b.current_player
        mh = [int(x) for x in rng.integers(-1, n * n, size=3)]
        oh = [int(x) for x in rng.integers(-1, n * n, size=3)]
        one, many = _pair(b, mh, oh, tp)
        assert one.shape == many.shape == (17, n, n)
        for idx in _ALWAYS_IDENTICAL:
            assert np.array_equal(one[idx], many[idx]), (
                f"局面 #{bi} 通道 {idx} 两条路径不一致\n"
                f"单图:\n{one[idx].astype(int)}\n批量:\n{many[idx].astype(int)}")
        # 通道 16 也应当相同（ko 默认取 ko_point；本文件这些局面多数 ko=-1）
        assert np.array_equal(one[16], many[16]), f"局面 #{bi} 通道 16 不一致"


# --------------------------------------------------------------------------- #
# 8. 通道 8 与 10/11/14/15 的已文档化分歧                                        #
# --------------------------------------------------------------------------- #

def test_channel8_batch_is_empty_minus_ko():
    """批量版通道 8 = 空点 ∧ 排除 ko（**独立**参考，不是生产函数当期望）。"""
    n = 5
    b = GoBoard(n)
    b.ko_point = 1 * n + 1
    b.resync_hash()
    one, many = _pair(b, [-1, -1, -1], [-1, -1, -1], 1)
    empty = (b.board == 0)
    ref = empty.astype(np.float32)
    ref[1, 1] = 0.0
    assert np.array_equal(many[8], ref), "批量通道 8 应恰好是「空点减 ko」"
    # 单图侧**不读** ko_point（PSK 独立裁决），所以这一格可以不同
    assert one[8].sum() >= many[8].sum(), "单图通道 8 不会比批量版更宽"

    # ko=None → 批量通道 8 退化成「全部空点」
    plain = GoBoard.feature_planes_batched(
        b.board[None], [[-1] * 3], [[-1] * 3], [1], None)[0]
    assert np.array_equal(plain[8], empty.astype(np.float32))

    # 契约：单图通道 8 == get_legal_moves()（比的是两个不同的生产函数）
    rng = np.random.default_rng(99)
    for k in range(8):
        bb = _random_board(rng, n=9, to_play=1 if k % 2 else -1)
        p = bb.feature_planes([-1, -1, -1], [-1, -1, -1], bb.current_player)
        expect = bb.get_legal_moves().reshape(9, 9).astype(np.float32)
        assert np.array_equal(p[8], expect), f"随机 #{k}: 通道 8 必须等于合法掩码"


def test_channel8_batch_ignores_suicide_while_scalar_forbids_it():
    """已文档化的分歧①：批量版**有意不查禁自杀**，单图版查。

    这是「选择」而不是「算不出来」，所以必须能被钉住：一旦有人给批量版补上禁自杀
    （路线图里明确说「可以直接向量化补上」），本用例会红并要求同步更新文档。
    """
    n = 5
    # 白围出一个单点眼 (2,2)，轮到黑 —— 黑填 (2,2) 是自杀，TT 侧禁
    stones = [(1, 2, -1), (3, 2, -1), (2, 1, -1), (2, 3, -1)]
    b = _board_from(stones, n, to_play=1)
    assert not _ref_legal(b.board, 1)[2, 2], "参考实现也认定填自己的真眼是自杀"
    one, many = _pair(b, [-1, -1, -1], [-1, -1, -1], 1)
    assert one[8, 2, 2] == 0.0, "单图通道 8 禁自杀（填自己的真眼）"
    assert many[8, 2, 2] == 1.0, "批量通道 8 有意不查禁自杀 → 这里是分歧点"
    # 而通道 12（本盘黑方一个子都没有）不受影响
    assert one[12].sum() == 0.0 and many[12].sum() == 0.0


def test_batched_liberty_planes_match_scalar_on_random_boards():
    """通道 10/11/14/15 在随机盘面上必须**无条件**逐位相同。

    ⚠ P4.3-fix：旧版这条用例挂着 `_has_shared_liberty` 当门（只在「无共享气点」的
      盘面上比），因为批量侧当时是入射计数。现在口径统一了，那道门**删掉** ——
      留着会让「共享气点处仍然一致」这件事无人看守。
    """
    rng = np.random.default_rng(31337)
    checked = 0
    with_shared = 0
    for k in range(40):
        b = _random_board(rng, n=9, density=0.30, to_play=1 if k % 2 else -1)
        checked += 1
        with_shared += 1 if _has_shared_liberty(b.board) else 0
        one, many = _pair(b, [-1, -1, -1], [-1, -1, -1], b.current_player)
        for idx in (10, 11, 14, 15):
            assert np.array_equal(one[idx], many[idx]), (
                f"随机 #{k} 通道 {idx} 两条路径不一致\n"
                f"单图:\n{one[idx].astype(int)}\n批量:\n{many[idx].astype(int)}")
    assert checked == 40, f"样本太少（{checked}），用例失去意义"
    assert with_shared >= 5, (
        f"40 个随机局面里只有 {with_shared} 个含共享气点 —— 去重逻辑几乎没被考到，"
        f"换更密的盘面（density 0.45）")


# --------------------------------------------------------------------------- #
# 8b. 块气的口径：去重坐标数（不是入射计数）                                  #
# --------------------------------------------------------------------------- #

def test_u_shaped_group_liberty_two():
    """**U 形块**（两颗子夹住同一个气点）：去重气数=2 → 14/15 标、10/11 **不**标。

    板面（5 路，黑执；`.` = 空）：
        B B B W W
        . . B W W
        B B B W W
        W W W W .
        . W W W .

    黑块 U = {(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0)}，它**唯一**的两个气是
    (1,0) 与 (1,1)（U 的两个内角）。但 (1,0) 同时贴着 (0,0) 与 (2,0)、(1,1) 同时
    贴着 (0,1)(1,2)(2,1) —— **入射计数 = 2 + 3 = 5**。
    所以旧批量口径（入射计数）会把这块算成 5 气 → 10/14 **两个 bucket 都漏标**；
    正确的去重口径是 2 气 → 只标 14。白块的气是 {(3,4),(4,0),(4,4)} = 3（入射 4），
    双方都不进 11/15。

    ⚠ 这块板就是旧 `test_batched_liberty_planes_diverge_on_shared_liberty` 用的
      **同一块**（三子拐角块，唯一气 (2,2) 被 (1,2) 与 (2,3) 共享）：那边断言批量
      算成 2 气、这边断言批量算成 1 气。**同一个复现板，取值相反** ——
      这就是本用例能红的全部理由。
    """
    n = 5
    stones = [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 2, 1), (2, 2, 1), (2, 1, 1),
              (2, 0, 1),
              (0, 3, -1), (0, 4, -1), (1, 3, -1), (1, 4, -1), (2, 3, -1), (2, 4, -1),
              (3, 0, -1), (3, 1, -1), (3, 2, -1), (3, 3, -1),
              (4, 1, -1), (4, 2, -1), (4, 3, -1)]
    b = _board_from(stones, n, to_play=1)
    seed = (0, 0)
    # 前提：本盘面确实存在「同块两子共享一个气点」，且去重数 != 入射数
    assert _has_shared_liberty(b.board), "本局面的前提就是存在共享气点"
    assert _ref_liberties(b.board, seed) == 2, "U 形块的去重气数必须是 2"
    assert _ref_incidence_liberties(b.board, seed) == 5, "U 形块的入射计数是 5"

    one, many = _pair(b, [-1, -1, -1], [-1, -1, -1], 1)
    u = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    for tag, p in (("单图", one), ("批量", many)):
        # 通道 14 **标出**整块（块掩码，不是点掩码）；通道 10 一个都不标
        for r, c in u:
            assert p[14, r, c] == 1.0, f"{tag}: U 形块 (r={r},c={c}) 必须落进通道 14"
        assert p[14].sum() == float(len(u)), f"{tag}: 通道 14 只应标这 7 颗子"
        assert p[10].sum() == 0.0, f"{tag}: 气=2 的块不该落进通道 10"
        # 白块气=3 → 11/15 都空
        assert p[11].sum() == 0.0 and p[15].sum() == 0.0, f"{tag}: 白块气=3"
    # 两条路径在 10/11/14/15 上**逐位相同**（P4.3-fix 的核心断言）
    for idx in (10, 11, 14, 15):
        assert np.array_equal(one[idx], many[idx]), f"通道 {idx} 两条路径不一致"


def test_shared_liberty_between_groups():
    """一个气点被**两个不同块**共享时，对**每块各算 1 气**（不得串味）。

    板面（5 路，黑执）：
        W W B B W
        W W . B W
        W W B B W
        W W W W W
        W W W W W

    唯一空点 (1,2) 同时贴着 3 颗黑子（(0,2)(1,3)(2,2)）与 1 颗白子 (1,1)：
      - 黑块 5 子，去重气数 = 1、入射计数 = 3 → 必须落进通道 **10**；
      - 白块（其余 19 颗白子连成一块），去重气数 = 1、入射计数 = 1 → 通道 **11**。
    所以那一个空点**对两块各算 1**。若去重按「空点」而不是「(块, 空点)」做（或者
    干脆全局去重），其中一块会变成 0 气 → 10/11 里少一块 → 本用例红。
    """
    n = 5
    black = {(0, 2), (0, 3), (1, 3), (2, 2), (2, 3)}
    white = {(r, c) for r in range(5) for c in range(5)} - black - {(1, 2)}
    b = _board_from([(r, c, 1) for r, c in sorted(black)]
                    + [(r, c, -1) for r, c in sorted(white)], n, to_play=1)
    assert _ref_liberties(b.board, (0, 2)) == 1
    assert _ref_liberties(b.board, (0, 0)) == 1, "白块（(0,0) 起）的去重气数也是 1"
    assert _ref_incidence_liberties(b.board, (0, 2)) == 3, "黑块的入射计数是 3"
    assert (0, 2) in black and (1, 1) in white and (1, 2) not in black | white

    one, many = _pair(b, [-1, -1, -1], [-1, -1, -1], 1)
    for tag, p in (("单图", one), ("批量", many)):
        for r, c in sorted(black):
            assert p[10, r, c] == 1.0, f"{tag}: 黑块 ({r},{c}) 气=1，必须落进通道 10"
        assert p[10].sum() == float(len(black)), f"{tag}: 通道 10 只应标黑块 5 颗子"
        for r, c in sorted(white):
            assert p[11, r, c] == 1.0, f"{tag}: 白块 ({r},{c}) 气=1，必须落进通道 11"
        assert p[11].sum() == float(len(white)), f"{tag}: 通道 11 只应标白块"
        assert p[14].sum() == 0.0 and p[15].sum() == 0.0, f"{tag}: 两块都只有 1 气"
    ref = _ref_lib_masks_all(b.board, 1)
    for i, idx in enumerate((10, 11, 14, 15)):
        assert np.array_equal(one[idx], ref[i]), f"单图通道 {idx} 与参考实现不符"
        assert np.array_equal(many[idx], ref[i]), f"批量通道 {idx} 与参考实现不符"


def test_liberty_count_is_distinct_not_incidence():
    """**直接钉死口径**：气数取「去重坐标数」，不取「入射计数」。

    四个构造形状，刻意让 (去重数, 入射数) 成对出现，且**两个口径落在不同的 bucket
    上**（而不是恰好同桶、断言空转）：

    | 形状 | 去重 | 入射 | 正确落点 | 取错口径会落到 |
    |---|---|---|---|---|
    | 黑 {(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0)}（U 形，内两空） | 2 | 5 | 14 | 都不标 |
    | 黑 {(0,0),(0,1),(0,2),(1,1),(2,0),(2,1),(2,2)}（实心 3×3 挖两空） | 2 | 6 | 14 | 都不标 |
    | 黑 {(1,2),(2,1),(2,2)}（拐角，唯一气被两子共享） | 1 | 2 | 10 | 14 |
    | 黑 {(1,2),(2,1),(2,2)} + 空 (0,2)(1,1)（**跨位撞号**） | 2 | 3 | 14 | 都不标 |

    第 ③ 行是「反过来」的方向：取入射计数会把它误送进**通道 14**。
    第 ④ 行专打「**只与紧邻的前一个方向比**」那个错法：唯一的气 (1,1) 同时是
    (2,1)（方向 0）与 (1,2)（方向 2）的邻点，而方向 1（上方 (0,1)）是白子。链式
    逐位比（比方向 0、1）能消掉这条重复；只比「紧邻的前一个」就消不掉 → 算成 3 气
    → 通道 14 空。生产 docstring 里那条 ⚠ 说的就是这个形状。
    """
    n = 5
    white_all = [(r, c, -1) for r in range(n) for c in range(n)]
    cases = [
        # (黑子, 额外留空的点, 期望去重气数, 期望入射计数, 期望落点通道)
        ([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)],
         [(1, 0), (1, 1)], 2, 5, 14),
        ([(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)],
         [(1, 0), (1, 2)], 2, 6, 14),
        ([(1, 2), (1, 3), (2, 3)], [(2, 2)], 1, 2, 10),
        # ④ 跨位撞号：唯一的气 (1,1) 是方向 0 与方向 2 的邻点，方向 1 是白子
        ([(1, 2), (2, 1), (2, 2)], [(0, 2), (1, 1)], 2, 3, 14),
    ]
    for stones, holes, want_distinct, want_inc, want_ch in cases:
        black = {tuple(s) for s in stones}
        holes = {tuple(h) for h in holes}
        layout = [p for p in white_all if (p[0], p[1]) not in (black | holes)]
        b = _board_from([(r, c, v) for r, c, v in layout] + [(r, c, 1) for r, c in black],
                        n, to_play=1)
        got_d = _ref_liberties(b.board, stones[0])
        got_i = _ref_incidence_liberties(b.board, stones[0])
        assert (got_d, got_i) == (want_distinct, want_inc), (
            f"夹具自检失败：去重 {got_d} / 入射 {got_i}，期望 "
            f"{want_distinct} / {want_inc}（stones={stones}）")
        assert got_d != got_i, "两个口径必须真的不同，否则本用例空转"

        one, many = _pair(b, [-1, -1, -1], [-1, -1, -1], 1)
        wrong = 14 if want_ch == 10 else 10
        for tag, p in (("单图", one), ("批量", many)):
            for r, c in sorted(black):
                assert p[want_ch, r, c] == 1.0, (
                    f"{tag}: ({r},{c}) 去重气数={want_distinct}，必须落进通道 {want_ch}")
            assert p[want_ch].sum() == float(len(black)), f"{tag}: 通道 {want_ch} 掩码范围"
            assert p[wrong].sum() == 0.0, (
                f"{tag}: 入射计数={want_inc} ≠ 去重={want_distinct}，"
                f"取错口径就会落进通道 {wrong}")
        # 双方四通道整块对拍参考实现（也顺带覆盖白块：夹具里白块的气数各不相同）
        ref = _ref_lib_masks_all(b.board, 1)
        for i, idx in enumerate((10, 11, 14, 15)):
            assert np.array_equal(one[idx], ref[i]), f"单图通道 {idx} 与参考实现不符"
            assert np.array_equal(many[idx], ref[i]), f"批量通道 {idx} 与参考实现不符"
            assert np.array_equal(one[idx], many[idx]), f"通道 {idx} 两条路径不一致"


# --------------------------------------------------------------------------- #
# 9. dataset 的 `n_channels`                                                     #
# --------------------------------------------------------------------------- #

def _fake_data(N=24, n=9, seed=5):
    rng = np.random.default_rng(seed)
    boards = rng.integers(-1, 2, size=(N, n, n)).astype(np.int8)
    return {
        'boards': boards,
        'my_hist': rng.integers(-1, n * n, size=(N, 3)).astype(np.int16),
        'op_hist': rng.integers(-1, n * n, size=(N, 3)).astype(np.int16),
        'ko': rng.integers(-1, n * n, size=N).astype(np.int16),
        'moves': rng.integers(0, n * n, size=N).astype(np.int16),
        'values': rng.integers(-1, 2, size=N).astype(np.int8),
        'to_play': rng.choice([-1, 1], size=N).astype(np.int8),
    }


def test_dataset_n_channels_default_is_12():
    """默认值保旧行为：默认 12 通道，且 12 == 17 的前 12 格（增强路径上也成立）。"""
    data = _fake_data()
    ds = SupervisedDataset(data)
    assert ds.n_channels == 12, "默认必须是 12（C7：默认 12 保零回归）"
    idxs = np.arange(16)

    # ① 形状
    states, moves, values = ds.sample_batch_numpy(idxs, augment=False)
    assert states.shape == (16, 12, 9, 9) and states.dtype == np.float32
    assert moves.shape == (16,) and values.shape == (16, 1)

    # ② augment=False 的契约：states **原样**就是 feature_planes_batched 的输出
    ref = GoBoard.feature_planes_batched(
        data['boards'][idxs], data['my_hist'][idxs], data['op_hist'][idxs],
        data['to_play'][idxs], data['ko'][idxs], n_channels=12)
    assert np.array_equal(states, ref), "augment=False 必须零拷贝原样返回"

    # ③ 17 通道数据集的前 12 格与 12 通道数据集**逐位相同**
    ds17 = SupervisedDataset(data, n_channels=17)
    s17, m17, v17 = ds17.sample_batch_numpy(idxs, augment=False)
    assert s17.shape == (16, 17, 9, 9)
    assert np.array_equal(s17[:, :12], states)
    assert np.array_equal(m17, moves) and np.array_equal(v17, values)

    # ④ **增强路径**上也成立：同一个 rng → 同一批对称变换 → 前 12 格仍逐位相同
    r12 = np.random.default_rng(1234)
    r17 = np.random.default_rng(1234)
    a12, _, _ = ds.sample_batch_numpy(idxs, rng=r12, augment=True)
    a17, _, _ = ds17.sample_batch_numpy(idxs, rng=r17, augment=True)
    assert np.array_equal(a17[:, :12], a12), "对称增强后前 12 格也必须一致"
    assert not np.array_equal(a12, states), "本夹具确实触发了增强（否则断言空转）"

    # ⑤ 12 通道数据集里 12-16 通道**根本不存在**（不是「算了再切」）
    raw = GoBoard.feature_planes_batched(
        data['boards'][idxs], data['my_hist'][idxs], data['op_hist'][idxs],
        data['to_play'][idxs], data['ko'][idxs], n_channels=12)
    assert raw.shape[1] == 12

    # ⑥ 越界参数必须抛
    for bad in (11, 18, 0):
        with pytest.raises(ValueError):
            SupervisedDataset(data, n_channels=bad)
    assert SupervisedDataset(data, n_channels=12).n_channels == 12
    assert SupervisedDataset(data, n_channels=17).n_channels == 17


def test_n_channels_12_does_not_compute_the_new_planes(monkeypatch):
    """**行为性红**：12 通道路径**不许**碰通道 12-16 的计算核。

    ⚠ 纯值断言抓不到这一类错误：「先造 17 通道再切前 12 格」在**取值上完全等价**，
      只有成本不同（多 5 通道的移位比较 + 一次气数比较 + 一次 scatter，而这条路径
      在 MCTS 每个叶子展开与训练预取时都调）。所以这里数 `_neighbor_all` 的调用
      次数，把「12 通道成本零回归」钉成可执行的不变式。
    """
    import src.game.go_rules as gr

    calls = {'n': 0}
    real = gr._neighbor_all

    def counting(mask):
        calls['n'] += 1
        return real(mask)

    monkeypatch.setattr(gr, '_neighbor_all', counting)
    boards = np.zeros((4, 9, 9), np.int8)
    boards[:, 4, 4] = 1
    mh = np.full((4, 3), -1, np.int16)
    tp = np.ones(4, np.int8)

    # 批量版：12 → 0 次；13/14 → 1/2 次；17 → 2 次
    for k, expect in ((12, 0), (13, 1), (14, 2), (15, 2), (16, 2), (17, 2)):
        calls['n'] = 0
        GoBoard.feature_planes_batched(boards, mh, mh, tp, np.full(4, -1, np.int16),
                                       n_channels=k)
        assert calls['n'] == expect, f"批量版 n_channels={k} 调了 {calls['n']} 次眼位核"
    # 单图版同理
    b = GoBoard(9)
    b.board[4, 4] = 1
    b.resync_hash()
    for k, expect in ((12, 0), (13, 1), (14, 2), (17, 2)):
        calls['n'] = 0
        b.feature_planes([], [], 1, n_channels=k)
        assert calls['n'] == expect, f"单图 n_channels={k} 调了 {calls['n']} 次眼位核"
    # 默认 12 的 dataset 一路到底也不该碰眼位核
    data = _fake_data(N=8)
    calls['n'] = 0
    SupervisedDataset(data).sample_batch_numpy(np.arange(8), augment=False)
    assert calls['n'] == 0, "默认 12 通道的 dataset 路径不该算眼位"
    calls['n'] = 0
    SupervisedDataset(data, n_channels=17).sample_batch_numpy(
        np.arange(8), augment=False)
    assert calls['n'] == 2, "17 通道的 dataset 路径应算两个颜色的眼位"
