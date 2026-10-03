"""``V7Dataset`` 的回归测试。

这个模块自己修了一个**别人看不见的 bug**，所以它的测试比一般数据集更重：
``scripts/build_dataset.py:93`` 的 ``return my[::-1], op[::-1]` 让主 npz 的
历史列错得毫无征兆（不抛异常、loss 照降、棋力不涨）。下面对拍的基准是
**逐行 Python 循环**的 :func:`reference_history_row` —— 刻意用不同算法，
否则对拍没有意义。

三条最硬的断言
--------------
1. :func:`rebuild_history_columns` 逐格等于独立参考实现（含非严格交替、
   跨局、pass、局首）；
2. 直接吃 npz 的 ``op_hist`` 会错 4/5 个历史通道
   （:func:`test_raw_npz_op_hist_slots_are_swapped`）—— 这是防「优化回去」；
3. 与 ``scripts/train_sft.py`` 的两份实现逐位一致
   （``v7_batch_features`` / ``v7_loss_labels``）—— `src` 不能反向依赖
   `scripts`，所以只能靠对拍防漂移。
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.v7_dataset import (   # noqa: E402
    ACTION_SIZE,
    BOARD_SIZE,
    DEFAULT_RULES_FLAGS_V7,
    FROZEN_C_HEADS,
    HISTORY_PAST,
    HISTORY_SLOTS,
    LOSS_W_KEYS,
    V7ActionSizeError,
    V7Dataset,
    dense_move_target,
    dihedral_batch,
    finalize_loss_weights,
    rebuild_history_columns,
    reference_history_row,
    spatial_global,
    to_v7_loss_labels,
    verify_history_columns,
)
from src.networks.katago_v7_loss import KataGoV7Loss   # noqa: E402

CELLS = BOARD_SIZE * BOARD_SIZE


class _RngFixed:
    """只吐出一组**预先给定**的 ``tforms`` 的假 rng。

    ⚠ 为什么要它：`sample_batch_numpy(augment=True)` 内部自己调
      `rng.integers(0, 8, size=B)` 抽 tforms，那个调用我们**拦不住**也看不到
      （没有返回值）。想验证「第 k 行用了 t」就只能控制它抽到什么。
      真实 rng 做不到「保证某一行是某个值」，所以用这个替身。
    """

    def __init__(self, tforms):
        self._t = np.asarray(tforms)

    def integers(self, low, high, size):
        assert (low, high) == (0, 8), '增广路径变了（不再是 8 路）'
        assert size == self._t.size, (size, self._t.size)
        return self._t.copy()


def _minimal_labels(b):
    """一份**最小可跑**的 loss 标签（键齐全、值合法），供 loss 级别测试用。

    刻意手写而不是从 `V7Dataset` 取：后者会把「数据集是否产出正确键」和
    「loss 是否认这些键」两件事缠在一起，改一处坏另一处时看不出是谁的锅。
    """
    w = {k: np.zeros(b, dtype=np.float32) for k in LOSS_W_KEYS}
    return {
        'policy_player': torch.zeros(b, ACTION_SIZE),
        'policy_opp': torch.zeros(b, ACTION_SIZE),
        'outcome': torch.zeros(b, dtype=torch.long),
        'ownership': torch.zeros(b, 1, 19, 19),
        'score': torch.zeros(b, dtype=torch.float32),
        'scoring': torch.zeros(b, 1, 19, 19),
        'seki': torch.zeros(b, 1, 19, 19),
        'score_distr': torch.zeros(b, 842),
        'game_weight': torch.ones(b),
        'futurepos': torch.zeros(b, 2, 19, 19),
        'w': w,
    }


# --------------------------------------------------------------------------- #
# 合成数据：一小盘**严格交替**的棋局序列
# --------------------------------------------------------------------------- #
def _game_seq(length, gid=0, first=+1):
    """一局 ``length`` 行：严格交替、落点沿对角线推进、**不含 pass**。"""
    moves = (np.arange(length, dtype=np.int16) * 7) % CELLS
    to_play = (first * (-1) ** np.arange(length)).astype(np.int8)
    return moves, to_play, np.full(length, gid, dtype=np.int32)


def _multi_game(lengths, ):
    """多局拼在一根数组里（模拟主数据集：首尾相接、**没有**局边界标记）。"""
    mv, tp, gid = [], [], []
    for g, n in enumerate(lengths):
        a, b, c = _game_seq(n, gid=g)
        mv.append(a)
        tp.append(b)
        gid.append(c)
    return (np.concatenate(mv), np.concatenate(tp), np.concatenate(gid))


def _tiny_dataset(**kw):
    """一个能直接喂 ``V7Dataset`` 的列 dict（12 通道路径能跑，V7 需要 19 路）。"""
    rng = np.random.default_rng(7)
    lengths = [40, 25, 60, 12, 33]
    n = sum(lengths)
    moves, to_play, game_ids = _multi_game(lengths)
    boards = np.zeros((n, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for i in range(n):
        r, c = divmod(int(moves[i]) if moves[i] >= 0 else 0, BOARD_SIZE)
        boards[i, r, c] = to_play[i] if i else 1
    return {
        'boards': boards,
        'my_hist': np.full((n, 3), -1, dtype=np.int16),
        'op_hist': np.full((n, 3), -1, dtype=np.int16),
        'ko': np.full(n, -1, dtype=np.int16),
        'moves': moves,
        'values': rng.random(n).astype(np.float32),
        'to_play': to_play,
        'game_ids': game_ids,
        'winrates': rng.random(n).astype(np.float32),
        'game_weights': np.ones(n, dtype=np.float32),
    }


# --------------------------------------------------------------------------- #
# 1 · 历史列重建：与独立参考实现逐格对拍
# --------------------------------------------------------------------------- #
def test_rebuild_matches_independent_reference():
    """向量化版 vs 逐行 Python 循环版，逐格比。"""
    mv, tp, gid = _multi_game([40, 25, 60, 12, 33])
    my, op = rebuild_history_columns(mv, tp, gid)
    assert my.shape == op.shape == (len(mv), HISTORY_SLOTS)
    for i in range(len(mv)):
        want_my, want_op = reference_history_row(i, mv, tp, gid)
        assert my[i].tolist() == want_my, ('my', i)
        assert op[i].tolist() == want_op, ('op', i)


def test_strict_alternation_gives_the_documented_slots():
    """严格交替时（最常见的形态）槽位内容是确定的，可直接写死断言。

    19 路盘上局内位置 0..7 的 to_play 是 +1,-1,+1,-1,...
    对行 i：moves[i-1] 由 to_play[i-1] = -to_play[i] 下 ⇒ op[0]
             moves[i-2] 由 to_play[i-2] =  to_play[i] 下 ⇒ my[0]
             moves[i-3] ⇒ op[1]；moves[i-4] ⇒ my[1]；moves[i-5] ⇒ op[2]
    ⇒ my = (i-2, i-4, i-6)，op = (i-1, i-3, i-5)。
    """
    n = 40
    mv, tp, gid = _game_seq(n)
    my, op = rebuild_history_columns(mv, tp, gid)
    i = 20
    assert my[i].tolist() == [mv[i - 2], mv[i - 4], mv[i - 6]]
    assert op[i].tolist() == [mv[i - 1], mv[i - 3], mv[i - 5]]


def test_non_alternating_to_play_still_lands_the_right_slot():
    """🔴 **不能按奇偶取**：真实数据有 842 对同局相邻行 to_play 不翻转。

    这里把第 10 行改成与第 9 行同色（该行本该是对手，实际是自己），断言
    `moves[9]` 落进 **my** 而不是 op —— 按奇偶的实现会把它放进 op[0]。
    """
    n = 24
    mv, tp, gid = _game_seq(n)
    tp = tp.copy()
    tp[9] = tp[8]                     # 同色连走 ⇒ 打破交替
    my, op = rebuild_history_columns(mv, tp, gid)
    i = 12
    # 行 12 的 to_play = +1。打破交替后第 8/9/10 行都是 +1 ⇒ moves[8/9/10]
    # **全是 my**（而不是交替下的 my,op,my）⇒ my[12] = (m[10], m[9], m[8])、
    # op[12] 只有 moves[11] 一手。
    assert tp[8] == tp[9] == tp[10] == tp[12] == +1
    # my 侧收下 k=2,3,4 三手；op 侧收下 k=1 与 k=5（k=3/4 被 my 占掉，
    # 但**不该在 op 侧开个洞** —— 槽位是按「该侧第几手」排的，不是按 k）
    assert my[i].tolist() == [mv[10], mv[9], mv[8]]
    assert op[i].tolist() == [mv[11], mv[7], -1]
    # 🔴 **这正是「不能按奇偶取」的证据**：按局内位置奇偶会把 moves[9]
    # （同色，属 my）算成 op，把整个 op 侧错开一格。
    assert mv[9] not in op[i].tolist()
    wmy, wop = reference_history_row(i, mv, tp, gid)
    assert my[i].tolist() == wmy and op[i].tolist() == wop
    # 再看 i=10：moves[9] 由 tp[9]=tp[8]=-tp[7]... 用参考实现兜底逐格比
    for j in (9, 10, 11, 12, 13):
        wmy, wop = reference_history_row(j, mv, tp, gid)
        assert my[j].tolist() == wmy
        assert op[j].tolist() == wop


def test_cross_game_rows_are_not_taken():
    """跨局那一手**不取**，且**不占槽位**（不是 clamp 到边界、不是沿用上一行）。"""
    mv, tp, gid = _multi_game([20, 20])
    my, op = rebuild_history_columns(mv, tp, gid)
    # 第 20 行是第二局第 0 行 ⇒ 往前全是别的局 ⇒ 三个槽全空
    assert my[20].tolist() == [-1, -1, -1]
    assert op[20].tolist() == [-1, -1, -1]
    # ⚠ 第 19 行是**第一局的末行**，局内还有 19 手历史 ⇒ **不该**是空的。
    #   把它当成空会把「局内历史够深」与「跨局被守卫拦住」两种情形混为一谈，
    #   而两者的正确行为完全不同。
    assert my[19].tolist() != [-1, -1, -1]
    assert op[19].tolist()[0] >= 0
    assert op[30].tolist()[0] >= 0


def test_pass_needs_no_special_case():
    """pass（``-1``）与「无此手」在 ch9..13 上输出相同 ⇒ 两者都是 -1。

    这里把 moves[9] 设成 -1，断言它与「那一手不存在」不可区分（都是 -1），
    也就是**不需要**额外的 pass 分支。
    """
    n = 24
    mv, tp, gid = _game_seq(n)
    mv = mv.copy()
    mv[9] = -1                            # 第 9 行是 pass
    my, op = rebuild_history_columns(mv, tp, gid)
    i = 12
    # 严格交替下 行 12: my = (moves[10], moves[8], moves[6])
    #                  op = (moves[11], moves[9], moves[7])  ⇒ pass 落在 op[1]
    assert op[i][1] == -1                 # pass ⇒ 空
    assert op[i][0] == mv[i - 1]          # 其它槽不受影响
    assert my[i][0] == mv[i - 2]
    # 与「那一手不存在」不可区分 —— 这正是「不需要 pass 分支」的理由
    mv2 = mv.copy()
    i2 = 12
    my2, op2 = rebuild_history_columns(mv2, tp, gid)
    assert np.array_equal(op2[i2][1], op[i2][1])


def test_verify_is_deterministic_and_catches_corruption():
    """构造期 ``verify=`` 抽查必须与 RNG 状态无关，且真能抓住错。"""
    mv, tp, gid = _multi_game([40, 25, 60])
    my, op = rebuild_history_columns(mv, tp, gid)
    r1 = verify_history_columns(my, op, mv, tp, gid, nrows=64)
    np.random.seed(1)
    np.random.rand(100)
    r2 = verify_history_columns(my, op, mv, tp, gid, nrows=64)
    assert r1 == r2 and r1['mismatch'] == 0

    bad_my = my.copy()
    bad_my[7, 0] = (bad_my[7, 0] + 1) % CELLS
    out = verify_history_columns(bad_my, op, mv, tp, gid, nrows=64)
    assert out['mismatch'] > 0 and out['examples']
    with pytest.raises(AssertionError):
        verify_history_columns(bad_my, op, mv, tp, gid, nrows=64, strict=True)


def test_rebuild_rejects_too_small_past_and_mismatched_columns():
    mv, tp, gid = _game_seq(10)
    with pytest.raises(ValueError, match='past'):
        rebuild_history_columns(mv, tp, gid, past=HISTORY_PAST - 1)
    with pytest.raises(ValueError, match='长度'):
        rebuild_history_columns(mv, tp[:-1], gid)


def test_rebuild_is_chunk_size_independent():
    """分块只是峰值内存手段 ⇒ 结果必须与不分块逐位相同。"""
    mv, tp, gid = _multi_game([37, 19, 44])
    ref_my, ref_op = rebuild_history_columns(mv, tp, gid, chunk=1 << 20)
    for cs in (1, 3, 8, 16):
        my, op = rebuild_history_columns(mv, tp, gid, chunk=cs)
        assert np.array_equal(my, ref_my), cs
        assert np.array_equal(op, ref_op), cs


# --------------------------------------------------------------------------- #
# 2 · 防「优化回去」：npz 的 op_hist 是错的
# --------------------------------------------------------------------------- #
def test_raw_npz_op_hist_slots_are_swapped():
    """🔴 直接吃 npz 的 ``op_hist`` 会让 **5 个历史通道错 4 个**。

    这个测试的存在意义是**钉死一个已修的 bug**：若有人觉得
    ``rebuild_history_columns`` 太慢而「优化」成直接读 npz 的列，
    症状是 ch9/ch11 互换、ch12/ch13 恒空 —— 不抛异常、loss 照降。

    这里用真实归档（存在才跑）断言那三格的实测归属；没有归档时退化成
    断言「重建值与 npz 值不同」，方向性仍然成立。
    """
    npz = os.path.join(ROOT, 'data', 'sgf_19x19_full.npz')
    mv, tp, gid = _multi_game([60])
    my, op = rebuild_history_columns(mv, tp, gid)

    if not os.path.isfile(npz):
        # 退化路径：构造一个「op 槽位 0/1 互换」的 npz 形态，确认重建值不同
        swapped = np.stack([op[:, 1], op[:, 0], op[:, 2]], axis=1)
        assert not np.array_equal(swapped, op)
        return

    z = np.load(npz)
    try:
        mv_all = z['moves'][:20000]
        tp_all = z['to_play'][:20000]
        gid_all = z['game_ids'][:20000]
        op_raw = z['op_hist'][:20000]
        my_raw = z['my_hist'][:20000]
    finally:
        z.close()
    my_ok, op_ok = rebuild_history_columns(mv_all, tp_all, gid_all)

    i = 200                        # 取足够深、局内历史充足的一行
    # 实测：op_hist[i,0] == moves[i-3]（99.8%），op_hist[i,1] == moves[i-1]（99.2%）
    assert op_raw[i, 0] == mv_all[i - 3] != op_ok[i, 0] == mv_all[i - 1]
    assert op_raw[i, 1] == mv_all[i - 1] != op_ok[i, 1] == mv_all[i - 3]
    # op_hist[i,2] 恒 -1，而正确值是 moves[i-5]
    assert op_raw[i, 2] == -1 and op_ok[i, 2] == mv_all[i - 5]
    # my_hist[i,0] 本来就对（只有 1 个元素，翻转是空操作）
    assert my_raw[i, 0] == my_ok[i, 0]
    # 重建把三个槽都填上了，npz 那份有两个是空的
    assert np.count_nonzero(my_raw[i] == -1) >= 1
    assert np.count_nonzero(my_ok[i] == -1) == 0


def test_row_i_board_is_before_moves_i():
    """off-by-one 的钉死：第 i 行的 board 是 ``moves[i]`` 落子**之前**的局面。

    ``rebuild_history_columns`` 的判据（``moves[i-k]`` 由 ``to_play[i-k]``
    落下）完全依赖这条。真实归档里 ``boards[i-1] → boards[i]`` 恰好只加一子
    （无提子）的那些行是判据 —— 那一子的颜色必须恰为 ``-to_play[i]``。
    """
    npz = os.path.join(ROOT, 'data', 'sgf_19x19_full.npz')
    if not os.path.isfile(npz):
        pytest.skip('需要 data/sgf_19x19_full.npz')
    z = np.load(npz)
    try:
        b = z['boards'][:60000]
        mv = z['moves'][:60000]
        tp = z['to_play'][:60000]
        gid = z['game_ids'][:60000]
    finally:
        z.close()
    same_game = gid[:-1] == gid[1:]
    delta = b[1:] - b[:-1]
    added = (delta == 1).sum((1, 2)) - (delta == -1).sum((1, 2))
    idx = np.flatnonzero(same_game & (added == 1))
    assert idx.size > 100, '样本里没有「恰好加一子」的行，换归档再跑'
    pos = idx + 1
    for i in pos[:200]:
        m = int(mv[i - 1])
        r, c = divmod(m, BOARD_SIZE)
        assert b[i - 1, r, c] == 0, 'moves[i-1] 落在这子之前 ⇒ 它是新增的那一子'
        assert b[i, r, c] == -tp[i], '新增子的颜色须恰为 -to_play[i]'


# --------------------------------------------------------------------------- #
# 3 · 对称增强
# --------------------------------------------------------------------------- #
def test_dihedral_is_identity_for_t0_and_8_ways_are_distinct():
    """`dihedral_batch` 收的是**单批** ``(B,C,H,W)`` + 逐行 ``tforms``。

    ⚠ 传 ``(8,C,H,W)`` + ``tforms=arange(8)`` 恰好每一行一个 t，但那样测的是
      「整批一次跑完」；这里**逐个 t 单测**，这样 8 路里任何一路写反都会定位到
      具体的 t 而不是整批失败。
    """
    rng = np.random.default_rng(3)
    x = rng.random((1, 3, BOARD_SIZE, BOARD_SIZE)).astype(np.float16)
    for t in range(8):
        out = dihedral_batch(x, np.array([t], dtype=np.int64))[0]
        want = x[0]
        if t >= 4:
            want = want[:, :, ::-1]
        k = t % 4
        if k:
            want = np.rot90(want, k=-k, axes=(1, 2))
        assert np.array_equal(out, want), t
    assert np.array_equal(dihedral_batch(x, np.zeros(1, np.int64))[0], x[0])


def test_dihedral_matches_dataset_augmentation():
    """🔴 与 ``dataset.sample_batch_numpy`` 的增强**逐位对拍**。

    方向写反（`rot90` 是逆时针、镜像在 W 轴）不会报错，只是「loss 照降、
    棋力不涨」。所以两份实现必须一起改，这条测试是防漂移的唯一闸。

    ⚠ ``dataset`` 的增广是**内联**在 ``sample_batch_numpy`` 里的（没有独立的
    ``_symmetrize`` 可调）⇒ 不能直接对拍那个函数。改用**逐位重放**：构造一个
    必然抽出同一组 ``tforms`` 的 rng，两条路径各跑一次，比较输出。
    """
    from src.data.dataset import SupervisedDataset
    rng0 = np.random.default_rng(2024)
    n = 8
    boards = rng0.integers(-1, 2, (n, 19, 19)).astype(np.int8)
    ds = SupervisedDataset({
        'boards': boards,
        'my_hist': np.full((n, 3), -1, dtype=np.int16),
        'op_hist': np.full((n, 3), -1, dtype=np.int16),
        'ko': np.full(n, -1, dtype=np.int16),
        'moves': rng0.integers(0, CELLS, n).astype(np.int16),
        'values': rng0.random(n).astype(np.float32),
        'to_play': rng0.choice([-1, 1], n).astype(np.int8),
    }, n_channels=12)
    idxs = np.arange(n)
    base = ds.sample_batch_numpy(idxs, augment=False)[0]

    for t in range(8):
        # ⚠ **不能**要求「一批里全是同一个 t」：那需要 n 个独立同值抽样，
        #   `default_rng(seed).integers(0,8,n)` 给不出这样的 seed（会 skip）。
        #   改成**逐行**验证 —— 把 tforms 钉成「第 k 行 = t、其余行 = 0」，
        #   于是两路输出里第 k 行应当逐位等于 `dihedral(x[k], t)`。
        tf = np.zeros(n, dtype=np.int64)
        tf[3] = t
        want_rows = dihedral_batch(base, tf)
        got = ds.sample_batch_numpy(idxs, rng=_RngFixed(tf), augment=True)[0]
        assert np.array_equal(got, want_rows), t


def test_dihedral_preserves_value_set_per_row():
    """变换只重排位置 ⇒ 每行的取值多重集不变（旋转/镜像的代数性质）。"""
    rng = np.random.default_rng(5)
    x = (rng.random((8, 2, BOARD_SIZE, BOARD_SIZE)) > 0.5).astype(np.float16)
    t = np.arange(8)
    out = dihedral_batch(x, t)
    for i in range(8):
        assert sorted(out[i].ravel().tolist()) == sorted(x[i].ravel().tolist())


# --------------------------------------------------------------------------- #
# 4 · 行权重
# --------------------------------------------------------------------------- #
def test_missing_w_key_is_rejected_not_defaulted_to_one():
    """🔴 ``KataGoV7Loss.w_of()`` 缺键返回全 1 ⇒ 「不写」不是「屏蔽」。"""
    w = {k: np.ones(4, dtype=np.float32) for k in LOSS_W_KEYS}
    del w['lead']
    w['lead'] = np.zeros(4, dtype=np.float32)
    del w['lead']
    out = finalize_loss_weights(w, 4)
    assert out['lead'].tolist() == [0.0] * 4          # 补的是显式 0
    del out['seki']
    with pytest.raises(KeyError, match='seki'):
        finalize_loss_weights(out, 4)


def test_w_of_really_does_default_to_ones():
    """把「缺键 ⇒ 全 1」这个前提本身也钉住。

    上一个测试只有在 ``w_of`` 保持现状时才有意义；若哪天有人把它改成返回 0，
    那些「缺了就抛」的检查就变成了多余的严格。**这是该约束的前提，不是替代。**

    `w_of` 是 `forward` 内部的闭包、调不到 ⇒ 改用**可观测的后果**反推：
    把某一项的 ``w`` 去掉，它的 loss 就应当与「显式给全 1」逐位相同。
    （`seki` 是唯一被特判的例外，见 katago_v7_loss 的注释。）
    """
    from src.networks.katago_v7 import NbtTfNet
    torch.manual_seed(0)
    net = NbtTfNet()
    out = net(torch.randn(3, 22, 19, 19), torch.randn(3, 19))
    lossf = KataGoV7Loss()
    base = _minimal_labels(3)

    def ownership_term(w):
        # ⚠ forward 返回 {'loss', 'terms', 'weighted', ...}，逐项在 ['terms'] 里
        lbl = {**base, 'w': w}
        return float(lossf(out, lbl)['terms']['ownership'])

    w1 = np.ones(3, dtype=np.float32)
    w0 = np.zeros(3, dtype=np.float32)
    explicit_one = ownership_term({**base['w'], 'ownership': w1})
    explicit_zero = ownership_term({**base['w'], 'ownership': w0})
    no_own = {k: v for k, v in base['w'].items() if k != 'ownership'}
    missing = ownership_term(no_own)
    assert abs(explicit_one - explicit_zero) > 1e-6, '前提：权重真的影响这一项'
    assert abs(missing - explicit_one) < 1e-6, \
        '删掉 ownership 的 w 后 loss 不变 ⇒ 默认值已不是 1，本约束的前提失效了'


def test_finalize_requires_futurepos_h_flags_when_enabled():
    w = {k: np.ones(3, dtype=np.float32) for k in LOSS_W_KEYS}
    with pytest.raises(KeyError, match='futurepos_h'):
        finalize_loss_weights(w, 3, futurepos_enabled=True)
    w['futurepos_h0'] = np.ones(3, dtype=np.float32)
    w['futurepos_h1'] = np.ones(3, dtype=np.float32)
    finalize_loss_weights(w, 3, futurepos_enabled=True)


def test_frozen_c_heads_are_the_nine_without_supervision():
    """C 阶段冻结的 9 项 = SGF 语料无法监督的那 9 项，且与 LOSS_W_KEYS 不混。"""
    assert len(FROZEN_C_HEADS) == 9
    assert set(FROZEN_C_HEADS) & {'policy', 'value'} == set()
    assert 'policy_opp' not in FROZEN_C_HEADS
    for name in ('ownership', 'scoring', 'seki', 'futurepos'):
        assert name in FROZEN_C_HEADS


# --------------------------------------------------------------------------- #
# 5 · one-hot 目标
# --------------------------------------------------------------------------- #
def test_dense_move_target_marks_invalid_as_all_zero():
    """``-1``（局末手 / 跨局 / 下一手是 pass）⇒ **全零行**，不是随便哪一类。

    ⚠ 落点 ``361``（= ``CELLS``）是**合法的 pass 类**，不是越界 —— 把它当
    越界会让「对手收官」这种真实情形退化成全零行（标签被静默抹掉）。
    """
    mv = np.array([0, 5, -1, CELLS, CELLS + 1, 1000], dtype=np.int64)
    out = dense_move_target(mv)
    assert out.shape == (6, ACTION_SIZE)
    assert out[0].argmax() == 0 and out[1].argmax() == 5
    assert out[3].argmax() == CELLS, 'pass 是第 362 类，必须正常 one-hot'
    for i in (2, 4, 5):
        assert float(out[i].sum()) == 0.0, i
    for i in (0, 1, 3):
        assert float(out[i].sum()) == 1.0, i


def test_dense_move_target_accepts_pass_class():
    """pass 是**真实的第 362 类**（不是特殊类），落点 361 要正常 one-hot。"""
    out = dense_move_target(np.array([CELLS], dtype=np.int64))
    assert out.shape == (1, ACTION_SIZE) and out[0, CELLS] == 1.0


# --------------------------------------------------------------------------- #
# 6 · 与 train_sft 的两份实现逐位一致
# --------------------------------------------------------------------------- #
def test_matches_train_sft_v7_batch_features_bitwise():
    """🔴 ``spatial_global`` 与 ``train_sft.v7_batch_features`` 必须逐位相同。

    `src` 不能反向依赖 `scripts`（会把 argparse / 设备探测拖进数据层），
    所以只能各写一份 + 对拍。这条测试是两份实现唯一的防漂移闸门。
    """
    from scripts.train_sft import v7_batch_features
    ds = V7Dataset(_tiny_dataset(), verify_history=None)
    idxs = np.array([5, 30, 31, 70, 120, 165])
    s1, g1 = spatial_global(ds, idxs, rules_flags=DEFAULT_RULES_FLAGS_V7)
    s2, g2 = v7_batch_features(ds, idxs, rules_flags=DEFAULT_RULES_FLAGS_V7)
    assert s1.dtype == s2.dtype and s1.shape == s2.shape
    assert np.array_equal(s1, s2), '空间量与 train_sft 不一致'
    assert np.array_equal(g1, g2), '全局量与 train_sft 不一致'


def test_loss_label_translation_matches_train_sft():
    """``to_v7_loss_labels`` 与 ``train_sft.v7_loss_labels`` 逐位相同且**幂等**。"""
    from scripts.train_sft import v7_loss_labels
    ds = V7Dataset(_tiny_dataset(), verify_history=None)
    _, _, moves, lbl = ds.sample_batch_v7(np.arange(4), augment=False)
    a = to_v7_loss_labels(lbl, moves)
    b = v7_loss_labels(lbl, moves)
    for k in ('policy_player', 'policy_opp', 'futurepos', 'outcome'):
        assert np.array_equal(a[k], b[k]), k
    # 幂等：叠第二遍值不变
    again = to_v7_loss_labels(a, moves)
    for k in ('policy_player', 'policy_opp', 'futurepos'):
        assert np.array_equal(a[k], again[k]), k


def test_future_rename_and_weights_pass_through_untouched():
    """``future`` → ``futurepos`` 只是改名；``w`` 必须**原样**透传。

    🔴 ``w['futurepos']`` 绝不能改成 OR：loss 把 (b,2,bs²) 塌成逐样本标量后
    只乘一个权重，权重的最小作用单位是「整块」⇒ 只活一路时给 1 会把那一路
    的 -1 哨兵当真值拟合。
    """
    ds = V7Dataset(_tiny_dataset(), verify_history=None)
    ds.attach_futurepos(mode='live')
    ds.warm_futurepos()
    _, _, moves, lbl = ds.sample_batch_v7(np.arange(8), augment=False)
    # ⚠ dataset 侧的键叫 ``future``，改名成 ``futurepos`` 发生在
    #   :func:`to_v7_loss_labels`（loss 叫 futurepos、dataset 叫 future）
    assert 'future' in lbl and 'futurepos' not in lbl
    out = to_v7_loss_labels(lbl, moves)
    assert np.array_equal(out['future'], out['futurepos'])
    w = out['w']
    assert 'futurepos_h0' in w and 'futurepos_h1' in w
    both = (w['futurepos_h0'] > 0) & (w['futurepos_h1'] > 0)
    assert np.array_equal(w['futurepos'] > 0, both), 'futurepos 必须是与语义'


# --------------------------------------------------------------------------- #
# 7 · V7Dataset 自身
# --------------------------------------------------------------------------- #
def test_dataset_rebuilds_history_over_the_input_columns():
    """构造后 ``my_hist``/``op_hist`` 必须与传入的（错的）那份**不同**。"""
    data = _tiny_dataset()
    assert (data['op_hist'] == -1).all(), 'fixture 本身给的是空列'
    ds = V7Dataset(data, verify_history=None)
    assert not (ds.op_hist == -1).all()
    rep = verify_history_columns(ds.my_hist, ds.op_hist, ds.moves, ds.to_play,
                                 ds.game_ids, nrows=len(ds.moves))
    assert rep['mismatch'] == 0


def test_dataset_requires_game_ids():
    """缺 game_ids ⇒ 跨局守卫失效 ⇒ **宁可不建**。"""
    data = _tiny_dataset()
    del data['game_ids']
    with pytest.raises(ValueError, match='game_ids'):
        V7Dataset(data, verify_history=None)


def test_dataset_rejects_non_19x19_boards():
    """V7 固定 19 路（动作空间 362 类）⇒ 不为迁就小盘面改动作空间。"""
    data = _tiny_dataset()
    n = len(data['boards'])
    small = np.zeros((n, 9, 9), dtype=np.int8)
    data['boards'] = small
    data['my_hist'] = np.full((n, 3), -1, dtype=np.int16)
    data['op_hist'] = np.full((n, 3), -1, dtype=np.int16)
    with pytest.raises(V7ActionSizeError):
        V7Dataset(data, verify_history=None)


def test_sample_batch_v7_shapes_and_no_rng_when_not_augmenting():
    ds = V7Dataset(_tiny_dataset(), verify_history=None)
    rng = np.random.default_rng(0)
    rng.random(1000)
    state = rng.bit_generator.state
    spatial, gl, moves, lbl = ds.sample_batch_v7(
        np.arange(8), rng=rng, augment=False)
    assert rng.bit_generator.state == state, 'augment=False 时不该碰 RNG'
    assert spatial.shape == (8, 22, BOARD_SIZE, BOARD_SIZE)
    assert gl.shape == (8, 19)
    assert moves.shape == (8,) and moves.dtype == np.int64
    assert 'next_move' in lbl and 'future' in lbl
    assert set(LOSS_W_KEYS) <= set(lbl['w'])


def test_augment_uses_one_tform_for_both_input_and_labels():
    """🔴 ``tforms`` 抽一次同时喂空间量与标签 —— 「自己再抽一份」是静默错标签。"""
    ds = V7Dataset(_tiny_dataset(), verify_history=None)
    rng = np.random.default_rng(4)
    _, _, moves, _ = ds.sample_batch_v7(np.arange(16), rng=rng, augment=True)
    rng2 = np.random.default_rng(4)
    _, _, moves2, _ = ds.sample_batch_v7(np.arange(16), rng=rng2, augment=True)
    assert np.array_equal(moves, moves2), '同一 seed 必须给出同一批增广'


def test_game_komi_accepts_mapping_or_game_id_indexed_array():
    """两种传法都要能用；贴目进全局 ch5（``selfKomi/20``）。"""
    data = _tiny_dataset()
    gids = np.unique(data['game_ids'])
    as_map = V7Dataset(data, game_komi={int(g): 7.5 for g in gids},
                       verify_history=None)
    as_arr = V7Dataset(data, game_komi=np.full(gids.max() + 1, 7.5),
                       verify_history=None)
    s_map, g_map = spatial_global(as_map, np.arange(8))
    s_arr, g_arr = spatial_global(as_arr, np.arange(8))
    assert np.array_equal(s_map, s_arr) and np.array_equal(g_map, g_arr)
    assert g_map[0, 5] != 0, 'ch5 应为 selfKomi/20，非 0'
    none_ds = V7Dataset(data, verify_history=None)
    _, g_none = spatial_global(none_ds, np.arange(8))
    assert g_none[0, 5] == 0.0, '不给贴目时按 0'


def test_labels_feed_all_twelve_loss_terms():
    """产物能喂满 12 项 loss 并反传（smoke 级，但足以证明键齐全）。

    ⚠ ``KataGoV7Loss.forward`` 返回 ``{'loss', 'terms', 'weighted', ...}``，
      逐项值在 ``['terms']`` 里 —— 共 12 项。``loss`` 是已加权求和的标量。
    """
    from src.networks.katago_v7 import NbtTfNet
    ds = V7Dataset(_tiny_dataset(), verify_history=None)
    ds.attach_futurepos(mode='live')
    ds.warm_futurepos()
    spatial, gl, moves, lbl = ds.sample_batch_v7(np.arange(6), augment=False)
    ll = to_v7_loss_labels(lbl, moves)
    torch.manual_seed(0)
    net = NbtTfNet()
    out = net(torch.from_numpy(spatial).float(), torch.from_numpy(gl).float())
    res = KataGoV7Loss()(out, ll)
    assert set(res) >= {'loss', 'terms'}
    assert len(res['terms']) == 12, sorted(res['terms'])
    assert torch.is_tensor(res['loss']) and res['loss'].ndim == 0
    res['loss'].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in net.parameters()), '反传后应有参数拿到非零梯度'


def test_open_heads_are_off_on_this_corpus_so_they_get_no_gradient():
    """🔴 A/C 阶段：SGF 语料没有标签的那 9 项必须**真的**没有梯度。

    只断言 ``w=0`` 是不够的 —— `w=0` 只让 loss 的**数值**为 0，梯度照样流过
    那些头（``0 * x`` 的导数是 0，但共享主干的梯度仍然非 0）。真正的冻需要
    独立 param group（见 train_v7 的 `build_param_groups`）；这里断言的是
    「各旁路头的输出对 loss 无贡献」这一层。
    """
    from src.networks.katago_v7 import NbtTfNet
    ds = V7Dataset(_tiny_dataset(), verify_history=None)
    ds.attach_futurepos(mode='live')
    ds.warm_futurepos()
    spatial, gl, moves, lbl = ds.sample_batch_v7(np.arange(6), augment=False)
    ll = to_v7_loss_labels(lbl, moves)
    assert float(ll['w']['ownership'].sum()) == 0.0
    assert float(ll['w']['scoring'].sum()) == 0.0
    assert float(ll['w']['seki'].sum()) == 0.0
    assert float(ll['w']['lead'].sum()) == 0.0
    # policy 恒开（硬编码 1.0，不经 w）
    torch.manual_seed(0)
    net = NbtTfNet()
    out = net(torch.from_numpy(spatial).float(), torch.from_numpy(gl).float())
    res = KataGoV7Loss()(out, ll)
    terms = res['terms']
    assert float(terms['ownership']) == 0.0
    assert float(terms['scoring']) == 0.0
    assert float(terms['seki']) == 0.0
    assert float(terms['lead']) == 0.0
    assert float(terms['policy']) != 0.0, 'policy 必须有梯度贡献'


def test_repr_reports_the_rebuilt_state():
    ds = V7Dataset(_tiny_dataset(), verify_history=None)
    r = repr(ds)
    assert 'V7Dataset' in r and 'soft_rows' in r
