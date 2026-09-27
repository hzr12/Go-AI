"""P2.7a — 增量棋块/气表的**等价性与生命周期**测试。

被测对象：`GoBoard` 的棋块表 `_groups`（gid → (颜色, 气集合, 棋子集合)）与
`_gid`（每点所属块）。它把热路径上的「邻块除落点外还有没有气」从 O(大气块)
flood fill 换成 O(1) 查表，是 MCTS 每候选 `clone()+play()` 的成本关键。

⚠ 本文件**不测性能**（性能在 P2.7 报告里单独量），只测**等价性**：
增量表必须在**任何**状态变更路径之后都与「按盘面全量 flood fill」逐块一致。
等价性 oracle = 本文件内独立实现的 `_reference_groups()`，它只读盘面、
不碰生产代码的任何一行。

覆盖：
  1. 随机对弈逐步：每手之后表 == 参考实现
  2. 提子之后：被提块消失、相邻块的气正确增长
  3. play/undo 往返：撤销后表回到之前那一刻的精确状态
  4. clone() 隔离：克隆体落子/撤销不污染源棋盘，且各自表自洽
  5. 外部写盘面 + resync_hash()：表作废并按新盘面重建
  6. **预测哈希 == 真实落子后的哈希**（逐步）：这块 oracle 曾抓出惰性重建版
     实现的三个真 bug（漏扫块内其余子 / 同块重复异或 / 记错棋子位置），
     是增量表最灵敏的守门人，必须常驻。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.go_rules import GoBoard  # noqa: E402


# --------------------------------------------------------------------------- #
# 参考实现：只读盘面，全量 flood fill，与生产代码无共享
# --------------------------------------------------------------------------- #
def _reference_groups(board):
    """返回 {frozenset(棋子扁平坐标): (颜色, frozenset(气扁平坐标))}。

    刻意**不**复用生产代码的任何函数：gid 由「扫描到的第一个点」决定，
    因此结果与生产的 gid 编号无关 —— 比较用「棋子集合」做键，天然与 gid 无关。
    """
    n = board.board_size
    seen = [[False] * n for _ in range(n)]
    out = {}
    for r0 in range(n):
        for c0 in range(n):
            color = int(board.board[r0, c0])
            if color == 0 or seen[r0][c0]:
                continue
            stones = set()
            libs = set()
            stack = [(r0, c0)]
            seen[r0][c0] = True
            while stack:
                r, c = stack.pop()
                stones.add(r * n + c)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < n and 0 <= nc < n):
                        continue
                    v = int(board.board[nr, nc])
                    if v == 0:
                        libs.add(nr * n + nc)
                    elif v == color and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            out[frozenset(stones)] = (color, frozenset(libs))
    return out


def _canonical(board):
    """把生产表转成与 `_reference_groups()` 同形状的规范形式。"""
    board._ensure_groups()
    return {
        frozenset(rec[2]): (rec[0], frozenset(rec[1]))
        for rec in board._groups.values()
    }


def assert_table_matches(board, ctx=""):
    got = _canonical(board)
    want = _reference_groups(board)
    assert got == want, (
        f"增量棋块表与参考实现不一致 @ {ctx}\n"
        f"块数 got={len(got)} want={len(want)}\n"
        f"仅 got 有: {sorted(map(sorted, set(got) - set(want)))[:4]}\n"
        f"仅 want 有: {sorted(map(sorted, set(want) - set(got)))[:4]}\n"
        f"键相同但内容不同: "
        f"{[sorted(s) for s in set(got) & set(want) if got[s] != want[s]][:4]}"
    )
    # _gid 数组必须与块表自洽：非空点必有 gid，且 gid 指向含该点的块
    n = board.board_size
    gid = board._gid
    for r in range(n):
        for c in range(n):
            v = int(board.board[r, c])
            g = int(gid[r, c])
            if v == 0:
                assert g == -1, f"({r},{c}) 空点却有 gid={g} @ {ctx}"
            else:
                assert g >= 0, f"({r},{c}) 石头却没有 gid @ {ctx}"
                assert r * n + c in board._groups[g][2], (
                    f"({r},{c}) 的 gid={g} 指向的块里没有它 @ {ctx}")


def _random_game(seed, n=9, moves=80):
    rng = np.random.default_rng(seed)
    b = GoBoard(n, komi=7.5)
    return b, rng


def _step(b, rng):
    pts = [int(i) for i in np.flatnonzero(b.get_legal_moves())]
    if not pts:
        return None
    mv = int(rng.choice(pts))
    assert b.play(mv), f"掩码放行的 {mv} 被 play 拒绝"
    return mv


# --------------------------------------------------------------------------- #
# 1. 随机对弈逐步对拍
# --------------------------------------------------------------------------- #
def test_table_matches_reference_every_random_move():
    for seed in range(6):
        b, rng = _random_game(seed)
        assert_table_matches(b, f"seed={seed} 空盘")
        for t in range(80):
            if _step(b, rng) is None:
                break
            assert_table_matches(b, f"seed={seed} 第 {t} 手之后")


def test_table_matches_reference_on_19x19():
    b, rng = _random_game(11, n=19, moves=60)
    for t in range(60):
        if _step(b, rng) is None:
            break
        assert_table_matches(b, f"19路 第 {t} 手之后")


# --------------------------------------------------------------------------- #
# 2. 提子：被提块消失、相邻块长气
# --------------------------------------------------------------------------- #
def _ko_board(n=9):
    """经典单子劫形（黑提白一子，白回提即 PSK 非法）。

        . B W .
        B . B W      <- (1,1) 是刚落的白子，只剩 (2,1) 一口气
        . B W .
    """
    b = GoBoard(n, komi=7.5)
    setup = {
        (0, 1): 1, (0, 2): -1,
        (1, 0): 1, (1, 2): 1, (1, 3): -1,
        (2, 1): 1, (2, 2): -1,
    }
    for (r, c), v in setup.items():
        b.board[r, c] = v
    b.current_player = -1
    b.resync_hash()
    return b


def test_capture_removes_group_and_grows_neighbours():
    b = _ko_board()
    assert_table_matches(b, "劫形初始")
    # 白 (1,1) 落子：四个邻点 (0,1)/(1,0)/(1,2)/(2,1) 全是黑，其中只有 (1,2)
    # 的气恰好只剩 (1,1) -> 只提这一子，(1,0) 还有 (0,0) 那口气所以活着。
    mv = 1 * 9 + 1
    assert b.play(mv), "白 (1,1) 应合法（提子）"
    assert_table_matches(b, "提子之后")
    # 恰好被提的那子真的不在盘上，而**不该被提的**那子必须还在
    assert int(b.board[1, 2]) == 0, "(1,2) 的唯一一口气被 (1,1) 占掉，应被提"
    for alive in ((1, 0), (0, 1), (2, 1)):
        assert int(b.board[alive]) == 1, f"{alive} 还有别的气，不该被提"
    # 落点所在块的气必须与盘面一致（表自洽已查），再确认块表里没有残留黑块
    g = int(b._gid[1, 1])
    assert b._groups[g][0] == -1, "落子后 (1,1) 应属白块"
    for gid_, rec in b._groups.items():
        for flat in rec[2]:
            assert int(b.board[flat // 9, flat % 9]) != 0, "块表里出现了空点"


def test_multi_stone_capture_keeps_table_consistent():
    """一口气提掉 2 子（不是劫）—— 惰性重建版曾在这里漏扫块内其余子。"""
    b = GoBoard(9, komi=7.5)
    # 黑两子 (3,4)(4,4) 只剩一口气：黑落 (2,4) 提掉白两子 (3,4)? 反过来构形：
    # 白两子 (3,3)(3,4)，其气 = (2,3)(2,4)(3,2)(4,3)(4,4)(3,5) 太多，改用窄口。
    # 直接手工摆一个「白两子只被黑封死一口气」的形状：
    # 白 (4,4)(4,5)；黑 (3,4)(5,4)(3,5)(5,5)(4,6) -> 白两子唯一的气是 (4,3)
    for (r, c), v in {
        (4, 4): -1, (4, 5): -1,
        (3, 4): 1, (5, 4): 1, (3, 5): 1, (5, 5): 1, (4, 6): 1,
    }.items():
        b.board[r, c] = v
    b.current_player = 1
    b.resync_hash()
    assert_table_matches(b, "多子提子构形")
    assert b.play(4 * 9 + 3), "黑 (4,3) 应合法（提掉两子）"
    assert int(b.board[4, 4]) == 0 and int(b.board[4, 5]) == 0, "两子都应被提"
    assert_table_matches(b, "提两子之后")
    g = int(b._gid[4, 3])
    assert len(b._groups[g][2]) == 1, "落点自成一块（没有邻接黑子）"
    assert (4, 4) not in [tuple(np.unravel_index(f, (9, 9))) for f in b._groups[g][2]]


# --------------------------------------------------------------------------- #
# 3. play / undo 往返
# --------------------------------------------------------------------------- #
def test_undo_restores_table_exactly():
    b, rng = _random_game(5)
    for _ in range(30):
        if _step(b, rng) is None:
            break
    before = _canonical(b)
    mv = _step(b, rng)
    assert mv is not None
    assert_table_matches(b, "落子后")
    assert b.undo(), "undo 应成功"
    assert_table_matches(b, "撤销后")
    assert _canonical(b) == before, "撤销后表必须与落子前**逐块相同**"


def test_undo_without_record_is_rejected():
    # 用**空撤销栈**的棋盘：否则 undo() 会弹走上一手 record=True 的记录并返回
    # True，测的就不是「record=False 的着法不可撤销」这件事了。
    b, rng = _random_game(6)
    mv = int(np.flatnonzero(b.get_legal_moves())[0])
    b.play(mv, record=False)
    assert b.undo() is False, "record=False 的着法不可撤销"
    assert_table_matches(b, "record=False 落子后（表须仍然自洽）")


def test_pass_does_not_disturb_table():
    b, rng = _random_game(7)
    for _ in range(20):
        if _step(b, rng) is None:
            break
    before = _canonical(b)
    assert b.play(-1)
    assert_table_matches(b, "pass 之后")
    assert _canonical(b) == before, "pass 不改盘面，表必须原样"
    assert b.undo()
    assert_table_matches(b, "pass 撤销之后")


# --------------------------------------------------------------------------- #
# 4. clone 隔离
# --------------------------------------------------------------------------- #
def test_clone_table_is_isolated_and_self_consistent():
    b, rng = _random_game(8)
    for _ in range(40):
        if _step(b, rng) is None:
            break
    src_snapshot = _canonical(b)
    cb = b.clone()
    assert_table_matches(b, "源棋盘 clone 前")
    assert_table_matches(cb, "克隆体 clone 后")
    # 克隆体落子若干手：源棋盘的表必须一动不动
    for t in range(20):
        pts = [int(i) for i in np.flatnonzero(cb.get_legal_moves())]
        if not pts:
            break
        assert cb.play(int(rng.choice(pts)))
        assert_table_matches(cb, f"克隆体第 {t} 手")
    assert _canonical(b) == src_snapshot, "克隆体落子污染了源棋盘的表"
    assert_table_matches(b, "源棋盘（克隆体走完之后）")
    # 撤销回到 clone 那一刻
    while cb.undo():
        assert_table_matches(cb, "克隆体撤销中")
    assert _canonical(cb) == src_snapshot, "撤销到底必须回到 clone 时刻的表"


def test_clone_shares_immutable_values_but_not_the_dict():
    """clone 必须**浅拷贝**表容器：值是不可变元组，可安全共享。"""
    b, rng = _random_game(9)
    for _ in range(30):
        if _step(b, rng) is None:
            break
    cb = b.clone()
    assert cb._groups is not b._groups, "表容器必须各自一份（否则互相污染）"
    assert cb._gid is not b._gid, "_gid 数组必须各自一份"
    shared = [k for k, v in cb._groups.items() if b._groups.get(k) is v]
    assert shared, "应存在被共享的不可变块记录（写时复制的意义所在）"


# --------------------------------------------------------------------------- #
# 5. 外部写盘面 → resync
# --------------------------------------------------------------------------- #
def test_external_write_then_resync_rebuilds_table():
    b, rng = _random_game(12)
    for _ in range(20):
        if _step(b, rng) is None:
            break
    b.board[0, 0] = 1
    b.board[0, 1] = 1
    b.board[1, 0] = -1
    b.resync_hash()
    assert_table_matches(b, "外部写盘面 + resync 之后")


def test_external_board_swap_adopts_and_rebuilds():
    b, rng = _random_game(13)
    for _ in range(20):
        if _step(b, rng) is None:
            break
    nb = np.zeros_like(b.board)
    nb[3, 3] = 1
    nb[3, 4] = 1
    b.board = nb                      # 换数组对象 -> _ensure_hash 自动采纳
    # 走一次**公共读入口**：增量表与增量哈希共用同一条采纳路径（都在
    # `_adopt_as_new_game()` 里失效/作废），所以「换盘面后」的第一条规矩是
    # 「先调一个公共查询」，而不是直接读内部表。绕过入口去断言表本身就是
    # 越过了契约 —— 那不是产品缺陷，是这个测试的前置条件写错了。
    b.get_legal_moves()
    assert_table_matches(b, "换 board 数组后（经公共读入口自动采纳）")


# --------------------------------------------------------------------------- #
# 6. 预测哈希 == 真实落子后的哈希（P2.7a 最灵敏的守门人）
# --------------------------------------------------------------------------- #
def test_forecast_hash_matches_actual_play_every_move():
    for seed in range(4):
        for n, moves in ((9, 90), (13, 60)):
            b, rng = _random_game(seed, n=n)
            for t in range(moves):
                pts = [int(i) for i in np.flatnonzero(b.get_legal_moves())]
                if not pts:
                    break
                mv = int(rng.choice(pts))
                pred = b.position_hash_after_move(mv)   # 落子后的染色键（预测）
                assert b.play(mv)
                assert b.position_hash() == pred, (
                    f"预测哈希与真实落子不符：seed={seed} n={n} 第 {t} 手 move={mv}")
                assert_table_matches(b, f"seed={seed} n={n} 第 {t} 手之后")


def test_forecast_hash_uses_group_table_for_captures():
    """提子预测必须走整块棋子（惰性重建版在这里错过多子块）。"""
    for seed in range(6):
        b = GoBoard(9, komi=7.5)
        rng = np.random.default_rng(100 + seed)
        for _ in range(400):
            pts = [int(i) for i in np.flatnonzero(b.get_legal_moves())]
            if not pts:
                break
            mv = int(rng.choice(pts))
            pred = b.position_hash_after_move(mv)
            ok = b.play(mv)
            if not ok:
                continue
            assert b.position_hash() == pred, f"seed={seed} 提子预测错 @ {mv}"
