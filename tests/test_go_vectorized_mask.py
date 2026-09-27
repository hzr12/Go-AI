"""掩码**两条实现**的等价性测试（生产路径按盘口分派，两条都是生产路径）。

`GoBoard.get_legal_moves()` 按盘口面积分派：
  - `board_size² < 256`（5/9/13 路）-> `_legal_masks_reference()`（逐候选 Python 循环）
  - `board_size² >= 256`（19 路）  -> `_legal_masks_vectorized()`（整盘 numpy）

所以本文件的断言**必须直接调两条实现互相比**，而不是「拿 `get_legal_moves()` 和
reference 比」—— 在小盘口上那等于**标量比标量**，什么也证明不了（这个坑真的踩过：
第一版测试就是这么写的，于是 23 个变异里 15 个存活，而唯一真正跑向量化路径的
19 路夹具里 0 次提子、0 次 PSK 命中）。

覆盖面：
  1. 随机对弈逐步逐位对拍（5/9/13/19 路），**两条实现都直接调**
  2. 提子密集夹具（向量化路径的 `rest == 0` 两个分支 + 提子候选的 PSK 推演）
  3. **真 superko**（三劫循环 6 手）—— 逼出 PSK 段的两个分支：命中历史即置非法
  4. 派生状态的两条不变量：`_lib_count` 对 `_groups`、`_pos_sorted` 对 `_pos_hash_counts`
     （**任一漂移的症状都是「掩码放行、play 拒绝」或「静默放行 superko」**）
  5. 分派本身：`get_legal_moves()` 真的按盘口选路，且结果与直接调实现一致
  6. 生命周期：clone 隔离、undo 精确往返、resync 重建
  7. 脱钩守卫：**成对**改写（石头->空 + 空->石头，计数不变）也必须被抓住
  8. 边界：四角/边的点、满盘、两次 pass 终局
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.go_rules import GoBoard  # noqa: E402

N9 = 9


# --------------------------------------------------------------------------- #
# 派生状态的不变量
# --------------------------------------------------------------------------- #
def assert_lib_count_consistent(b, ctx=""):
    """`_lib_count[g]` 必须恒等于块 g 的气数；已删的块号必须归 0。

    向量化掩码**唯一**的气数来源。漂移的症状是「掩码放行一手自杀、`play()` 正确拒绝」
    —— 这个症状真的出现过（刷新被放在「被提子让邻居长气」之前）。
    """
    b._ensure_groups()      # resync 之后表是失效的，先按当前盘面重建再检查
    bad = [(g, int(b._lib_count[g]), len(rec[1]))
           for g, rec in b._groups.items() if int(b._lib_count[g]) != len(rec[1])]
    assert not bad, f"气数表漂移 @ {ctx}: (gid, 表里, 真实) = {bad[:6]}"
    n2 = b.board_size ** 2
    dead = [g for g in range(n2) if g not in b._groups and int(b._lib_count[g]) != 0]
    assert not dead, f"已删除的块号没有归零 @ {ctx}: {dead[:6]}"
    assert int(b._lib_count[n2]) == 0, "哨兵槽必须恒为 0"


def assert_pos_sorted_consistent(b, ctx=""):
    """`_pos_sorted` 必须与 `_pos_hash_counts` 的**键集**逐个相同。

    它是 PSK 批量查重的唯一数据源。漂移的症状更隐蔽：静默**放行**一手 superko
    （而标量掩码与 `play()` 都会正确拒绝）。所以这条不变量与 `_lib_count` 那条
    同等重要 —— 别因为「它只是个缓存」就省掉。
    """
    if not b._vectorized_mask_worth_it():
        return          # 小盘口不维护它（维护要 np.insert ~10 µs/手），由临时构造覆盖
    assert set(int(x) for x in b._pos_sorted) == set(b._pos_hash_counts), (
        f"_pos_sorted 与历史键集不一致 @ {ctx}: "
        f"多 {sorted(set(int(x) for x in b._pos_sorted) - set(b._pos_hash_counts))[:4]} "
        f"少 {sorted(set(b._pos_hash_counts) - set(int(x) for x in b._pos_sorted))[:4]}")
    arr = [int(x) for x in b._pos_sorted]
    assert arr == sorted(arr), f"_pos_sorted 必须有序 @ {ctx}"


# --------------------------------------------------------------------------- #
# 核心断言：两条实现**直接**互比
# --------------------------------------------------------------------------- #
def assert_both_impls_agree(b, ctx=""):
    """向量化 vs 标量，逐位。**直接调两条实现**，绕过分派。"""
    vec = b._legal_masks_vectorized()
    ref = b._legal_masks_reference()
    assert vec.shape == ref.shape, f"形状不同 @ {ctx}: {vec.shape} vs {ref.shape}"
    if not np.array_equal(vec, ref):
        n = b.board_size
        diff = np.flatnonzero(vec != ref)
        where = [f"{int(i)}({int(i) // n},{int(i) % n}) "
                 f"向量化={bool(vec[i])} 标量={bool(ref[i])}" for i in diff[:6]]
        raise AssertionError(f"两条掩码实现分叉 @ {ctx}（{diff.size} 个点）: {where}")
    # 分派结果也必须与两者一致（分派本身是第三条路径）
    assert np.array_equal(b.get_legal_moves(), ref), f"分派结果与实现不一致 @ {ctx}"


def assert_all(b, ctx=""):
    assert_both_impls_agree(b, ctx)
    assert_lib_count_consistent(b, ctx)
    assert_pos_sorted_consistent(b, ctx)


def _step_and_check(b, rng, ctx, stats):
    """走一手并全量核对；顺带统计提子数（用来证明夹具真的踩到了分支）。"""
    pts = [int(i) for i in np.flatnonzero(b.get_legal_moves())]
    if not pts:
        return False
    before = int((b.board != 0).sum())
    mv = int(rng.choice(pts))
    assert b.play(mv), (
        f"掩码放行的 {mv} 被 play 拒绝 @ {ctx} —— 两条真相源已经分叉")
    if int((b.board != 0).sum()) < before + 1:
        stats["captures"] += 1
    assert_all(b, f"{ctx} 之后（累计提子 {stats['captures']}）")
    return True


# --------------------------------------------------------------------------- #
# 1. 随机对弈逐步逐位对拍（全部盘口，两条实现直接对拍）
# --------------------------------------------------------------------------- #
def test_both_impls_agree_every_random_move_all_board_sizes():
    for n, seed, moves in ((5, 1, 40), (9, 0, 80), (13, 21, 60), (19, 22, 50)):
        b = GoBoard(n, komi=7.5)
        rng = np.random.default_rng(seed)
        stats = {"captures": 0}
        for t in range(moves):
            if not _step_and_check(b, rng, f"{n}路 seed={seed} 第 {t} 手", stats):
                break
        assert b.move_number > 10, f"{n}路 seed={seed} 夹具空转了？"


# --------------------------------------------------------------------------- #
# 2. 提子密集夹具（向量化路径的两个 rest==0 分支）
# --------------------------------------------------------------------------- #
def test_both_impls_agree_on_capture_heavy_games():
    """19 路跑满 400 手：随机对局在 19 路上前 ~180 手**一次提子都没有**，
    所以必须跑够长、并且断言真的提了子（否则夹具空转，测试变成自比自）。
    """
    for n, seed, moves in ((19, 5, 400), (9, 200, 300), (5, 7, 200)):
        b = GoBoard(n, komi=7.5)
        rng = np.random.default_rng(seed)
        stats = {"captures": 0}
        for t in range(moves):
            if b.is_terminal(max_moves=10 ** 9) and False:
                break
            if not _step_and_check(b, rng, f"{n}路 seed={seed} 第 {t} 手", stats):
                break
        assert stats["captures"] > 0, (
            f"{n}路 seed={seed} 一子都没提，夹具空转（本文件的价值全在提子分支上）")


def test_both_impls_agree_on_hand_built_capture_shapes():
    """手工摆几个提子形状，覆盖「提一子 / 提多子 / 提子旁边还有敌块」。

    随机局很难摆出「落点四周全是敌块且其中一个被打吃」的形状，而这正是
    `rest == 0` 的提子分支最容易错的地方。
    """
    b = GoBoard(9, komi=7.5)
    # 白两子 (4,4)(4,5) 只剩一口气 (4,3)：黑落 (4,3) 一次提两子
    for (r, c), v in {
        (4, 4): -1, (4, 5): -1,
        (3, 4): 1, (5, 4): 1, (3, 5): 1, (5, 5): 1, (4, 6): 1,
    }.items():
        b.board[r, c] = v
    b.current_player = 1
    b.resync_hash()
    assert_all(b, "多子提子构形")
    assert b.play(4 * 9 + 3), "黑 (4,3) 应合法（提两子）"
    assert int(b.board[4, 4]) == 0 and int(b.board[4, 5]) == 0, "两子都应被提"
    assert_all(b, "提两子之后")


# --------------------------------------------------------------------------- #
# 3. 真 superko：三劫循环 6 手（逼出 PSK 段的两个分支）
# --------------------------------------------------------------------------- #
_TRIPLE_KO_STONES = {
    (2, 2): 1, (2, 3): -1, (3, 1): 1, (3, 2): -1, (3, 4): -1, (4, 2): 1, (4, 3): -1,
    (2, 6): -1, (2, 7): 1, (3, 5): -1, (3, 6): 1, (3, 8): 1, (4, 6): -1, (4, 7): 1,
    (6, 2): 1, (6, 3): -1, (7, 1): 1, (7, 2): -1, (7, 4): -1, (8, 2): 1, (8, 3): -1,
}
_TRIPLE_KO_SEQUENCE = [
    (1, 3 * N9 + 3), (-1, 3 * N9 + 7), (1, 7 * N9 + 3),
    (-1, 3 * N9 + 2), (1, 3 * N9 + 6), (-1, 7 * N9 + 2),
]


def _triple_ko_board():
    b = GoBoard(N9)
    for (r, c), v in _TRIPLE_KO_STONES.items():
        b.board[r, c] = v
    b.current_player = 1
    b.resync_hash()
    return b


def test_both_impls_agree_on_real_superko():
    """6 手三劫循环：末手闭合成手 -> 落子后的染色命中历史 -> **必须被判非法**。

    这是唯一能稳定触发 PSK 段的夹具（随机局里 superko 一次都不出现）。
    两条实现都必须把它判成 False，且 `play()` 必须拒绝。
    """
    b = _triple_ko_board()
    assert_all(b, "三劫棋形")
    for i, (who, mv) in enumerate(_TRIPLE_KO_SEQUENCE[:-1]):
        assert b.current_player == who, f"第 {i} 手行棋方不符"
        assert b._legal_masks_vectorized()[mv], f"第 {i} 手 {mv} 应合法"
        assert b.play(mv) is True, f"第 {i} 手 {mv} 应被接受"
        assert_all(b, f"三劫第 {i} 手之后")
    # 末手：闭合成手
    who, last = _TRIPLE_KO_SEQUENCE[-1]
    assert b.current_player == who
    ref = b._legal_masks_reference()
    vec = b._legal_masks_vectorized()
    assert not ref[last], "标量实现必须把闭合成手判为非法（PSK）"
    assert not vec[last], "向量化实现必须把闭合成手判为非法（PSK）"
    assert bool(ref[last]) == bool(vec[last])
    assert b.play(last) is False, "play() 也必须拒绝闭合成手"
    assert_all(b, "闭合成手被拒之后")


def test_both_impls_agree_on_superko_at_19():
    """同一套三劫棋形**平移到 19 路的一角** -> 真正被分派到的那条路上也走一遍 PSK 段。

    5/9/13 路上向量化不是生产路径（分派给标量），所以「9 路的三劫」只验证了
    直接调用；这里必须让**生产路径**也真的判一次 superko。
    平移而不是铺 4 份：棋形本身自足（周围都是空点），平移后仍是同一个闭环。
    """
    b = GoBoard(19)
    off_r, off_c = 2, 2
    for (r, c), v in _TRIPLE_KO_STONES.items():
        b.board[r + off_r, c + off_c] = v
    b.current_player = 1
    b.resync_hash()
    assert b._vectorized_mask_worth_it(), "19 路必须走向量化路径，否则本用例空转"
    assert_all(b, "19 路三劫棋形（平移后）")

    played = 0
    for who, mv in _TRIPLE_KO_SEQUENCE[:-1]:
        r, c = divmod(mv, N9)
        mv19 = (r + off_r) * 19 + (c + off_c)
        assert b.current_player == who, f"第 {played} 手行棋方不符"
        assert bool(b._legal_masks_vectorized()[mv19]), f"第 {played} 手应合法"
        assert b.play(mv19) is True, f"第 {played} 手 {mv19} 应被接受"
        played += 1
        assert_all(b, f"19 路三劫第 {played} 手之后")
    assert played == len(_TRIPLE_KO_SEQUENCE) - 1, "三劫序列没走完"

    who, last = _TRIPLE_KO_SEQUENCE[-1]
    r, c = divmod(last, N9)
    last19 = (r + off_r) * 19 + (c + off_c)
    assert b.current_player == who
    ref = b._legal_masks_reference()
    vec = b._legal_masks_vectorized()
    assert not ref[last19], "标量实现必须判非法"
    assert not vec[last19], "**向量化生产路径**必须判非法（PSK）"
    assert b.play(last19) is False, "play() 也必须拒绝"
    assert_all(b, "闭合成手被拒之后")


def test_both_impls_agree_on_simple_ko_at_19():
    """19 路单劫：闭合成手是**提子**候选 -> 走 `position_hash_after_move()` 那一支。

    三劫（平移到 19 路）覆盖的是**复现整个棋盘**的 superko，落子那一手同时也是提子，
    所以它其实也在提子分支上；真正只靠「纯算术 + searchsorted」那一支的判据是
    `test_plain_candidate_psk_hit_is_rejected_by_both`（因为「不提子的复现染色」在
    真实对局里几乎不出现）。两个夹具各管一段，别把它们当成互相覆盖。
    """
    b = GoBoard(19, komi=7.5)
    # 最小可证的单劫形状：
    #   白 (9,9) 被黑 (8,9)(10,9)(9,8) 封到只剩一口气 (9,10)；
    #   黑落 (9,10) 提掉它，而 (9,10) 这颗黑子被白 (8,10)(10,10)(9,11) 围住，
    #   提子后只剩 (9,9) 一口气 —— 白立刻回提会复现提子前的染色 -> PSK 拒绝。
    for (r, c), v in {
        (8, 9): 1, (10, 9): 1, (9, 8): 1,
        (8, 10): -1, (10, 10): -1, (9, 11): -1, (9, 9): -1,
    }.items():
        b.board[r, c] = v
    b.current_player = 1
    b.resync_hash()
    assert b._vectorized_mask_worth_it(), "19 路必须走向量化路径，否则本用例空转"
    assert_all(b, "19 路单劫棋形")
    # 黑提白 (9,9)
    assert b.play(9 * 19 + 10), "黑 (9,10) 应合法（提子成劫）"
    assert int(b.board[9, 9]) == 0, "白 (9,9) 应被提"
    assert b.ko_point == 9 * 19 + 9, "应形成单劫（ko_point 指回被提点）"
    assert_all(b, "提子成劫之后")
    # 白回提 (9,9) -> 复现提子前的染色 -> PSK 必须拒
    assert b.current_player == -1
    vec = b._legal_masks_vectorized()
    ref = b._legal_masks_reference()
    assert bool(vec[9 * 19 + 9]) == bool(ref[9 * 19 + 9]), "回提点两条实现必须同答案"
    assert not ref[9 * 19 + 9], "标量实现必须把劫回提判非法（PSK）"
    assert not vec[9 * 19 + 9], "**向量化生产路径**必须把劫回提判非法"
    assert b.play(9 * 19 + 9) is False, "劫回提必须被 PSK 拒绝"
    assert_all(b, "劫回提被拒之后")


def test_plain_candidate_psk_hit_is_rejected_by_both():
    """**不提子**的候选键命中历史 -> 两条实现都必须判非法（纯算术 + searchsorted 那一支）。

    为什么必须注入历史而不能等一个真实局面：
      - PSK 段的两个分支里，「提子候选」那一支由单劫夹具覆盖（真的会拦，39 次）；
      - 而「不提子候选」那一支要求一个**不提子**的复现染色 —— 随机局与三劫/单劫
        夹具**一次都没产生**（实测该分支执行 1546 次、命中 0 次）。所以那一支
        等不到，只能把它的输入直接构造出来。

    做法：挑一个合法且**非提子**的点 p，把它「落子后的候选键」塞进历史
    （dict 与 `_pos_sorted` 一起塞，保持两者一致），于是两条实现都必须把 p 判非法。
    变异测试证据：把 `flat[plain[hist[pos] == cand]] = False` 换成 `pass`
    （即这一支永不拦）时，本用例是唯一抓住它的断言。
    """
    from src.game.go_rules import _zkey_array, _zobrist_key  # noqa: PLC0415

    b = GoBoard(19, komi=7.5)
    rng = np.random.default_rng(77)
    for t in range(40):
        pts = [int(i) for i in np.flatnonzero(b.get_legal_moves())]
        if not pts:
            break
        assert b.play(int(rng.choice(pts)))
    color = b.current_player
    vec0 = b._legal_masks_vectorized()
    # 找一个「合法且不是提子」的点
    plain_pts = []
    for i in np.flatnonzero(vec0).tolist():
        c = b.clone()
        before = int((c.board != 0).sum())
        if c.play(i) and int((c.board != 0).sum()) == before + 1:
            plain_pts.append(i)
    assert plain_pts, "夹具空转：这一局没有非提子的合法点"

    checked = 0
    for p in plain_pts[:12]:
        r, c = divmod(p, 19)
        cand = b._pos_zobrist ^ _zobrist_key(r, c, color)
        # 只对「本来合法」的 p 才有意义（合法的 p 其候选键此刻必然不在历史里）
        assert cand not in b._pos_hash_counts
        b._pos_hash_counts[cand] = 1
        b._pos_sorted = np.insert(b._pos_sorted,
                                  int(np.searchsorted(b._pos_sorted,
                                                      np.uint64(cand))),
                                  np.uint64(cand))
        b._legal_cache = None
        ref = b._legal_masks_reference()
        vec = b._legal_masks_vectorized()
        assert bool(vec[p]) == bool(ref[p]), f"点 {p}: 两条实现对注入的 PSK 命中同答案"
        assert not ref[p], f"标量实现必须判非法 @ 点 {p}"
        assert not vec[p], f"**向量化**必须判非法（纯算术 + searchsorted 那一支）@ 点 {p}"
        # 复原历史
        del b._pos_hash_counts[cand]
        pos = int(np.searchsorted(b._pos_sorted, np.uint64(cand)))
        b._pos_sorted = np.delete(b._pos_sorted, pos)
        b._legal_cache = None
        checked += 1
    assert checked >= 3, f"只检查了 {checked} 个点，夹具可能太窄"


def test_zkey_array_agrees_with_scalar_zobrist_key():
    """`_zkey_array(n, color)` 必须逐项等于 `_zobrist_key(r, c, color)`。

    向量化掩码的 PSK 段用前者，标量用后者 —— 两者是**同一张表的两种表示**。
    变异测试证据：把 `_zkey_array` 的白槽索引写成黑槽（`+ (0 if color > 0 else 1)`
    -> `+ 0`）时，语义断言抓不住（白方永远等不到一次「非提子的 PSK 命中」），
    只有这条直接的结构断言能抓。
    """
    from src.game.go_rules import _zkey_array, _zobrist_key  # noqa: PLC0415

    for n in (5, 9, 13, 19):
        for color in (1, -1):
            arr = _zkey_array(n, color)
            assert arr.dtype == np.uint64, "必须是 uint64（钥匙可 >= 2**63）"
            assert arr.size == n * n
            for r in range(n):
                for c in range(n):
                    assert int(arr[r * n + c]) == _zobrist_key(r, c, color), (
                        f"{n} 路 color={color} 的 ({r},{c}) 两种表示不一致")
    # 黑白两色必须**不同**（否则上面的循环在 color=-1 时也会通过）
    assert (int(_zkey_array(9, 1)[8]) != int(_zkey_array(9, -1)[8])), \
        "同一交叉点的黑/白钥匙不能相同"


# --------------------------------------------------------------------------- #
# 5. 分派本身
# --------------------------------------------------------------------------- #
def test_dispatch_picks_the_expected_implementation():
    for n, want_vec in ((5, False), (9, False), (13, False), (19, True)):
        b = GoBoard(n)
        assert b._vectorized_mask_worth_it() is want_vec, (
            f"{n}路分派错了：走向量={b._vectorized_mask_worth_it()}")
    # 阈值本身也要钉住（改它之前先重测两条实现的交叉点）
    assert 13 * 13 < 256 <= 19 * 19, "阈值必须落在 13 路与 19 路之间"


# --------------------------------------------------------------------------- #
# 6. 生命周期
# --------------------------------------------------------------------------- #
def test_clone_isolation_of_both_derived_caches():
    b, rng = GoBoard(9, komi=7.5), np.random.default_rng(41)
    stats = {"captures": 0}
    for t in range(40):
        if not _step_and_check(b, rng, f"源棋盘第 {t} 手", stats):
            break
    lc_snapshot = b._lib_count.copy()
    cb = b.clone()
    assert cb._lib_count is not b._lib_count, "气数表必须各自一份"
    for t in range(20):
        pts = [int(i) for i in np.flatnonzero(cb.get_legal_moves())]
        if not pts:
            break
        assert cb.play(int(rng.choice(pts)))
        assert_all(cb, f"克隆体第 {t} 手")
    assert np.array_equal(b._lib_count, lc_snapshot), "克隆体落子改动了源棋盘的气数表"
    assert_all(b, "源棋盘（克隆体走完之后）")


def test_undo_restores_both_caches_exactly():
    """play/undo 之后两个派生缓存都要**逐项**回到落子前的状态。

    ⚠ 这一段必须在 **19 路**上也做：`_pos_sorted` 只在走向量化的盘口上被维护
    （小盘口维护它要 np.insert ~10 µs/手），而 `assert_pos_sorted_consistent` 在
    小盘口是**直接 return** 的 —— 只有一个 9 路 undo 用例时，那条不变量等于没测。
    变异测试证据：把 `_rollback_position` 里的 `np.delete` 去掉（回滚后 `_pos_sorted`
    留一个陈旧键）时，只有这里的 19 路分支能抓住。
    """
    for n, seed, moves in ((9, 42, 40), (19, 43, 40)):
        b, rng = GoBoard(n, komi=5.5), np.random.default_rng(seed)
        stats = {"captures": 0}
        for t in range(moves):
            if not _step_and_check(b, rng, f"{n}路 undo 前第 {t} 手", stats):
                break
        lc_snapshot = b._lib_count.copy()
        ps_snapshot = (b._pos_sorted.copy()
                       if b._pos_sorted is not None else None)
        pts = [int(i) for i in np.flatnonzero(b.get_legal_moves())]
        assert pts, "需要至少一个可落点"
        assert b.play(int(rng.choice(pts)))
        assert_all(b, f"{n}路 落子之后")
        assert b.undo()
        assert_all(b, f"{n}路 撤销之后")
        assert np.array_equal(b._lib_count, lc_snapshot), (
            "撤销后气数表必须与落子前**逐槽相同**")
        if ps_snapshot is not None and b._vectorized_mask_worth_it():
            assert np.array_equal(
                b._pos_sorted.view(np.uint64), ps_snapshot.view(np.uint64)), (
                "撤销后 _pos_sorted 必须与落子前**逐项相同**")
        # 多撤几步：历史容器的删除路径必须一直对
        for _ in range(3):
            if not b.undo():
                break
            assert_all(b, f"{n}路 连续撤销之后")


def test_resync_rebuilds_derived_caches():
    b, rng = GoBoard(19, komi=7.5), np.random.default_rng(43)
    stats = {"captures": 0}
    for t in range(30):
        if not _step_and_check(b, rng, f"resync 前第 {t} 手", stats):
            break
    b.board[9, 9] = 1
    b.board[9, 10] = 1
    b.board[10, 9] = -1
    b.resync_hash()
    assert_all(b, "外部写盘面 + resync 之后")


# --------------------------------------------------------------------------- #
# 7. 脱钩守卫：包括**成对**改写（计数不变那种）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [9, 19])
def test_vectorized_mask_raises_on_desync(n):
    """不就地 resync 就改盘面：向量化必须**抛**，不能静默算错。

    ⚠ 关键在**成对**改写：一次「石头 -> 空」加一次「空 -> 石头」让空点数
    **保持不变**，所以任何「计数相等即认为没脱钩」的守卫都会放行。第一版守卫正是
    计数代理，被 review 抓到（9 路静默返回掩码、标量抛 —— 两条路径对同一个契约违反
    给出不同答案）。现在守卫是逐点比较，这里两种改写方式都要能抓住。
    """
    b, rng = GoBoard(n, komi=7.5), np.random.default_rng(44)
    stats = {"captures": 0}
    for t in range(12):
        if not _step_and_check(b, rng, f"脱钩前第 {t} 手", stats):
            break

    # (a) 单点：空 -> 石。⚠ 必须**先把表建起来**（上面那 12 手已经建了），
    #     否则 `_ensure_groups()` 会按当前盘面重建 —— 那是「自愈」，不是脱钩。
    single = b.clone()
    single_empty = int(np.flatnonzero(single.board == 0)[0])
    single.board[single_empty // n, single_empty % n] = 1
    single._legal_cache = None
    with pytest.raises(RuntimeError, match="resync_hash"):
        single._legal_masks_vectorized()

    # (b) 成对：空 -> 石 且 石 -> 空（空点数不变）
    empty = int(np.flatnonzero(b.board == 0)[0])
    er, ec = divmod(empty, n)
    stone = int(np.flatnonzero(b.board != 0)[0])
    sr, sc = divmod(stone, n)
    b.board[er, ec] = 1
    b.board[sr, sc] = 0
    b._legal_cache = None
    assert int((b.board == 0).sum()) == n * n - int((b.board != 0).sum()), "构造有误"
    with pytest.raises(RuntimeError, match="resync_hash"):
        b._legal_masks_vectorized()
    # 标量实现同样抛（两条路径的失败方式必须一致）
    with pytest.raises(RuntimeError, match="resync_hash"):
        b._legal_masks_reference()
    b.resync_hash()
    assert_all(b, "resync 之后")


# --------------------------------------------------------------------------- #
# 8. 边界
# --------------------------------------------------------------------------- #
def test_border_points_and_terminal_shapes():
    """四角/四边（补边的越界邻居）、满盘、两次 pass 终局。"""
    b, rng = GoBoard(9, komi=7.5), np.random.default_rng(31)
    stats = {"captures": 0}
    for t in range(60):
        if not _step_and_check(b, rng, f"角与边第 {t} 手", stats):
            break
    for r, c in ((0, 0), (0, 4), (4, 0), (4, 4), (0, 8), (8, 0), (8, 8), (4, 8)):
        if b.board[r, c] == 0:
            assert b.is_legal(r * 9 + c) == bool(b.get_legal_moves()[r * 9 + c]), (
                f"边角 ({r},{c}) 的 is_legal 与掩码不一致")

    full = GoBoard(5, komi=5.5)
    for r in range(5):
        for c in range(5):
            full.board[r, c] = 1 if (r + c) % 2 == 0 else -1
    full.current_player = 1
    full.resync_hash()
    assert_all(full, "手工满盘")

    done = GoBoard(5)
    assert done.play(-1) and done.play(-1)
    assert_all(done, "两次 pass 之后")
    assert done.get_legal_moves().any(), "终局后掩码仍要算得出来"
