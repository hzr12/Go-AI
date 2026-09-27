"""掩码**向量化实现**的等价性测试（P2.7b 之后追加）。

`GoBoard.get_legal_moves()` 的生产路径是 `_legal_masks_vectorized()`（整盘 numpy），
而 `_legal_masks_reference()`（逐候选 Python 循环 + 查表）保留为**语义 oracle**。
两条实现互为对照，本文件在随机对弈的**每一手**上断言它们**逐位相同**。

⚠ 为什么一个向量化改动要有这样一份测试：两条实现风格完全不同（一条整盘聚集、
一条逐点循环 + 集合查询），任何一条 drifted 都不会被另一条的实现细节掩盖。
这不是「重复劳动式」的防御性测试，而是**唯一**能抓住「两条路径悄悄分叉」的手段 ——
实际已抓到一次真 bug：`_lib_count`（气数查找表）的刷新被放在了「被提子让邻居长气」
**之前**，于是提子后气数偏小，掩码把一个 2 气的块误判成打吃，放行了一手自杀，
而 `play()` 用块表真值正确拒绝 —— 症状是「掩码放行、play 拒绝」。

覆盖面：
  1. 随机对弈逐步逐位对拍（9/13/19 路，含提子、劫、终局 pass）
  2. `_lib_count` 与 `_groups` 的**恒等不变量**（每手之后；这是 1 的根因守护）
  3. clone 隔离：克隆体落子不改变源棋盘的气数表
  4. play/undo 往返：撤销后气数表回到精确状态
  5. 外部写盘面 + resync 后重建
  6. 脱钩守卫：不就地 resync 就改 board，向量化掩码**抛**而不是静默算错
  7. 补边语义：盘面四角/边上的点，越界邻居对判据零贡献（等价于标量的越界 continue）
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.go_rules import GoBoard, _groups_desync_error  # noqa: E402


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def assert_lib_count_consistent(b, ctx=""):
    """`_lib_count[g]` 必须恒等于块 g 的气数；已删的块号必须归 0。

    这是掩码向量化**唯一的气数来源**。它一旦与 `_groups` 分叉，两条真相源就各说各话
    （实测症状：掩码放行一手自杀，`play()` 正确拒绝）。
    """
    b._ensure_groups()      # resync 之后表是失效的，先按当前盘面重建再检查
    bad = []
    for g, rec in b._groups.items():
        want = len(rec[1])
        got = int(b._lib_count[g])
        if got != want:
            bad.append((g, got, want))
    assert not bad, f"气数表漂移 @ {ctx}: (gid, 表里的气数, 真实气数) = {bad[:6]}"
    n2 = b.board_size ** 2
    dead = [g for g in range(n2)
            if g not in b._groups and int(b._lib_count[g]) != 0]
    assert not dead, f"已删除的块号没有归零 @ {ctx}: {dead[:6]}"
    assert int(b._lib_count[n2]) == 0, "哨兵槽必须恒为 0"


def assert_masks_agree(b, ctx=""):
    vec = b.get_legal_moves()
    ref = b._legal_masks_reference()
    assert vec.shape == ref.shape, f"形状不同 @ {ctx}: {vec.shape} vs {ref.shape}"
    if not np.array_equal(vec, ref):
        diff = np.flatnonzero(vec != ref)
        n = b.board_size
        where = [f"{int(i)}({int(i) // n},{int(i) % n})"
                 f" 向量化={bool(vec[i])} 标量={bool(ref[i])}" for i in diff[:6]]
        raise AssertionError(f"两条掩码实现分叉 @ {ctx}（{diff.size} 个点）: {where}")


def assert_all(b, ctx=""):
    assert_masks_agree(b, ctx)
    assert_lib_count_consistent(b, ctx)


def _play_random(b, rng, moves, ctx):
    for t in range(moves):
        pts = [int(i) for i in np.flatnonzero(b.get_legal_moves())]
        if not pts:
            break
        mv = int(rng.choice(pts))
        assert b.play(mv), (
            f"掩码放行的 {mv} 被 play 拒绝 @ {ctx} 第 {t} 手 —— "
            f"两条真相源（掩码 vs play）已经分叉")
        assert_all(b, f"{ctx} 第 {t} 手之后")


# --------------------------------------------------------------------------- #
# 1. 随机对弈逐步对拍
# --------------------------------------------------------------------------- #
def test_vectorized_mask_matches_reference_every_move():
    for seed in range(4):
        _play_random(GoBoard(9, komi=7.5), np.random.default_rng(seed), 80,
                     f"9路 seed={seed}")


def test_vectorized_mask_matches_reference_on_bigger_boards():
    for n, seed, moves in ((13, 21, 60), (19, 22, 50)):
        _play_random(GoBoard(n, komi=7.5), np.random.default_rng(seed), moves,
                     f"{n}路 seed={seed}")


def test_vectorized_mask_matches_reference_around_captures():
    """专门盯提子：补边聚集 + 气数表最容易在「块消失/块长气」时出错。"""
    for seed in range(6):
        b = GoBoard(9, komi=7.5)
        rng = np.random.default_rng(200 + seed)
        captures = 0
        for t in range(300):
            pts = [int(i) for i in np.flatnonzero(b.get_legal_moves())]
            if not pts:
                break
            before = int((b.board != 0).sum())
            mv = int(rng.choice(pts))
            assert b.play(mv)
            if int((b.board != 0).sum()) < before + 1:
                captures += 1
            assert_all(b, f"seed={seed} 第 {t} 手（累计提子 {captures}）")
        assert captures > 0, f"seed={seed} 一子都没提，夹具空转了？"


def test_vectorized_mask_matches_reference_on_full_and_terminal_boards():
    """满盘（无空点）与终局（两次 pass）这两条边界路径也要走对。"""
    b = GoBoard(5, komi=5.5)
    for r in range(5):
        for c in range(5):
            if (r + c) % 2 == 0:
                b.board[r, c] = 1
            else:
                b.board[r, c] = -1
    b.current_player = 1
    b.resync_hash()
    assert_all(b, "手工满盘（可能有填眼点）")
    b2 = GoBoard(5)
    assert b2.play(-1) and b2.play(-1)
    assert_all(b2, "两次 pass 之后")
    assert b2.get_legal_moves().any(), "终局后掩码仍要能算出来（不因为终局而短路）"


# --------------------------------------------------------------------------- #
# 2. 角/边：越界邻居必须零贡献
# --------------------------------------------------------------------------- #
def test_border_points_match_reference():
    """四角与四边的点：向量化靠 1 格补边处理越界，标量靠 continue —— 两者必须一致。

    补边把越界邻居的 board 填 2、gid 填哨兵，于是「除落点外的气数」= -1，
    既不 >0（非自杀）也不 ==0（提子）。这条用例专门盯这个哨兵约定。
    """
    b, rng = GoBoard(9, komi=7.5), np.random.default_rng(31)
    _play_random(b, rng, 60, "9路（角与边）")
    for r, c in ((0, 0), (0, 4), (4, 0), (4, 4), (0, 8), (8, 0), (8, 8), (4, 8)):
        i = r * 9 + c
        if b.board[r, c] == 0:
            assert b.is_legal(i) == bool(b.get_legal_moves()[i]), (
                f"边角 ({r},{c}) 的 is_legal 与掩码不一致")


# --------------------------------------------------------------------------- #
# 3-5. 生命周期
# --------------------------------------------------------------------------- #
def test_clone_lib_count_is_isolated():
    b, rng = GoBoard(9, komi=7.5), np.random.default_rng(41)
    _play_random(b, rng, 40, "源棋盘")
    snapshot = b._lib_count.copy()
    cb = b.clone()
    assert cb._lib_count is not b._lib_count, "气数表必须各自一份"
    for t in range(20):
        pts = [int(i) for i in np.flatnonzero(cb.get_legal_moves())]
        if not pts:
            break
        assert cb.play(int(rng.choice(pts)))
        assert_lib_count_consistent(cb, f"克隆体第 {t} 手")
    assert np.array_equal(b._lib_count, snapshot), "克隆体落子改动了源棋盘的气数表"
    assert_all(b, "源棋盘（克隆体走完之后）")


def test_undo_restores_lib_count_exactly():
    b, rng = GoBoard(9, komi=5.5), np.random.default_rng(42)
    _play_random(b, rng, 40, "undo 前")
    snapshot = b._lib_count.copy()
    pts = [int(i) for i in np.flatnonzero(b.get_legal_moves())]
    assert pts, "需要至少一个可落点"
    mv = int(rng.choice(pts))
    assert b.play(mv)
    assert_all(b, "落子之后")
    assert b.undo()
    assert_all(b, "撤销之后")
    assert np.array_equal(b._lib_count, snapshot), (
        "撤销后气数表必须与落子前**逐槽相同**")


def test_resync_rebuilds_lib_count():
    b, rng = GoBoard(9, komi=7.5), np.random.default_rng(43)
    _play_random(b, rng, 20, "resync 前")
    b.board[4, 4] = 1
    b.board[4, 5] = 1
    b.board[5, 4] = -1
    b.resync_hash()
    assert_lib_count_consistent(b, "外部写盘面 + resync 之后")
    assert_masks_agree(b, "外部写盘面 + resync 之后")


# --------------------------------------------------------------------------- #
# 6. 脱钩守卫
# --------------------------------------------------------------------------- #
def test_vectorized_mask_raises_on_desync_instead_of_silently_wrong():
    """不就地 resync 就改盘面：向量化路径必须**抛**，不能静默算出一个错的掩码。

    向量化靠哨兵槽把 gid=-1 当成「空点」处理，所以脱钩时它**不会**像标量那样
    自然地抛 `KeyError` —— 不加守卫就会静默放行/误禁。这里锁住「大声失败」。
    """
    b, rng = GoBoard(9, komi=7.5), np.random.default_rng(44)
    _play_random(b, rng, 12, "脱钩前")
    empty = int(np.flatnonzero(b.board == 0)[0])
    r, c = divmod(empty, 9)
    b.board[r, c] = 1                      # 就地写，不 resync
    b._legal_cache = None                  # 逼它真的算
    with pytest.raises(RuntimeError, match="resync_hash"):
        b.get_legal_moves()
    # 标量实现同样抛（两条路径的失败方式一致）
    with pytest.raises(RuntimeError, match="resync_hash"):
        b._legal_masks_reference()
    b.resync_hash()
    assert_all(b, "resync 之后")
