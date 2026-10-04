"""stdata（KataGo 分布式训练 npz）读取器的约定测试。

这个读取器里有**四条从真实数据反解出来的约定**（bit-packed 布局、stride=19 的
策略索引、棋盘尺寸反推、64/80 列布局分派）。它们都不是「看起来显然」的东西，
而且错了之后**不报错**——只会静默地把 9 盘的行当 19 盘、把权重列读错位。
所以每条都必须有单测。

 依赖 `katago/stdata/*.tgz`（1.5 GB）的那几条用 `pytest.mark.skipif` 跳过；
纯合成的几条（布局分派、topk、越界检查）**永远跑** —— 那些是最容易写错的部分。
"""

import io
import os
import sys
import tarfile

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.katago_npz import (  # noqa: E402
    ACTION_SIZE, BOARD_CELLS, BOARD_STRIDE, GLOBAL_TARGET_LAYOUT, PACKED_BYTES,
    SPATIAL_CHANNELS, KatagoNpzLayoutError, board_size_from_packed,
    policy_index_is_in_board, read_katago_npz, to_v7_labels, topk_policy,
    unpack_binary_input,
)
from src.networks.katago_v7_loss import (  # noqa: E402
    KataGoV7Loss, policy_dense_from_sparse,
)

STDATA = os.path.join(ROOT, 'katago', 'stdata')
ARCHIVE_0825 = os.path.join(STDATA, '2026-08-25npzs.tgz')
ARCHIVE_B28 = os.path.join(STDATA, 'zzb28c512nfd4-s8264801024-d4596884264.tar')

needs_stdata = pytest.mark.skipif(
    not (os.path.isfile(ARCHIVE_0825) and os.path.isfile(ARCHIVE_B28)),
    reason='需要 katago/stdata 下的 tgz（不在仓库里）')


# --------------------------------------------------------------------------- #
# 合成 fixture
# --------------------------------------------------------------------------- #
def _pack_spatial(spatial):
    """``(N,22,19,19) bool`` → ``(N,22,46) uint8``（打包的逆运算）。"""
    n, c, h, w = spatial.shape
    assert (h, w) == (BOARD_STRIDE, BOARD_STRIDE)
    flat = spatial.reshape(n, c, BOARD_CELLS).astype(np.uint8)
    pad = np.zeros((n, c, PACKED_BYTES * 8), dtype=np.uint8)
    pad[:, :, :BOARD_CELLS] = flat
    return np.packbits(pad, axis=-1, bitorder='big')


def _fake_npz(n=4, board=19, cols=80, has_q=True, net='kata1-tf3-b11c768'):
    sp = np.zeros((n, SPATIAL_CHANNELS, BOARD_STRIDE, BOARD_STRIDE), dtype=bool)
    sp[:, 0, :board, :board] = True
    sp[:, 1, :3, :3] = True                       # pla 石头
    g = np.zeros((n, cols), dtype=np.float32)
    g[:, 0] = 1.0                                 # 硬 outcome = 胜
    g[:, 3] = 5.0                                 # scoreMean
    g[:, 20] = 3.0                                # 实际终局分差
    g[:, 21] = 2.0                                # lead
    g[:, 25] = 1.0                                # global_weight
    g[:, 26] = 1.0                                # w_policy_player
    g[:, 27] = 1.0                                # w_ownership
    g[:, 28] = 1.0                                # w_policy_opp
    g[:, 29] = 1.0                                # w_lead
    g[:, 33] = 1.0                                # w_futurepos
    g[:, 34] = 1.0                                # w_scoring
    g[:, 47] = 7.5                                # komi
    vt = np.zeros((n, 5, BOARD_STRIDE, BOARD_STRIDE), dtype=np.int8)
    vt[:, 0] = 1                                   # ownership 全 +1
    vt[:, 4] = 30                                  # scoring
    pol = np.zeros((n, 2, ACTION_SIZE), dtype=np.int16)
    pol[:, 0, 7] = 100
    pol[:, 0, 100] = 50
    pol[:, 1, 7] = 80
    sb = np.zeros((n, 842), dtype=np.int8)
    sb[:, 421] = 30
    sb[:, 422] = 70
    d = {
        'binaryInputNCHWPacked': _pack_spatial(sp),
        'globalInputNC': np.zeros((n, 19), dtype=np.float32),
        'policyTargetsNCMove': pol,
        'globalTargetsNC': g,
        'scoreDistrN': sb,
        'valueTargetsNCHW': vt,
    }
    if has_q:
        d['qValueTargetsNCMove'] = np.zeros((n, 3, ACTION_SIZE), dtype=np.int16)
    return d


class _Npz(dict):
    """最小 NpzFile 替身：只需要 ``__getitem__`` 与 ``.files``。"""
    @property
    def files(self):
        return list(self)


# --------------------------------------------------------------------------- #
# 约定 1：bit-packed 布局
# --------------------------------------------------------------------------- #
def test_pack_unpack_roundtrip_is_lossless():
    rng = np.random.default_rng(0)
    sp = rng.random((3, SPATIAL_CHANNELS, BOARD_STRIDE, BOARD_STRIDE)) < 0.4
    out = unpack_binary_input(_pack_spatial(sp))
    assert out.shape == (3, SPATIAL_CHANNELS, BOARD_STRIDE, BOARD_STRIDE)
    assert np.array_equal(out, sp), 'pack/unpack 不是互逆的'


def test_unpack_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match='通道数不符'):
        unpack_binary_input(np.zeros((2, 7, PACKED_BYTES), dtype=np.uint8))


def test_unpack_rejects_wrong_packed_width():
    """ `np.unpackbits` 的 axis 默认 None 会**静默展平**，形状检查必须自己做。"""
    with pytest.raises(ValueError, match='packed 形状不符'):
        unpack_binary_input(np.zeros((2, SPATIAL_CHANNELS, 40), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# 约定 2：棋盘尺寸从 ch0 反推
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('board', [9, 11, 13, 15, 17, 18, 19])
def test_board_size_recovered_from_channel0(board):
    d = _fake_npz(n=2, board=board)
    got = board_size_from_packed(d['binaryInputNCHWPacked'])
    assert got.tolist() == [board, board]


def test_non_square_mask_yields_zero_not_a_nearest_size():
    """非方阵必须标 0（丢掉），不许「开方取整」凑成邻边长。"""
    sp = np.zeros((1, SPATIAL_CHANNELS, BOARD_STRIDE, BOARD_STRIDE), dtype=bool)
    sp[0, 0, :10, :11] = True                      # 110 格，非完全平方
    got = board_size_from_packed(_pack_spatial(sp))
    assert got.tolist() == [0]


def test_policy_index_in_board_uses_stride_19_not_s_times_c():
    """stride=19：9 盘的合法索引是 ``r*19+c``（r,c ≤ 8），最大 160。

     注意 81 是**合法**的（= 4*19+5），因为 stride=19 下 0..160 并非连续 ——
    把它当反例会让人误以为「索引 > s² 就是小盘」，而真正的判据是
    ``r<9 且 c<9``。pass 槽 361 不算盘内索引。
    """
    idx = np.array([0, 80, 81, 152, 156, 160, 161, 200, 361])
    got = policy_index_is_in_board(idx, 9)
    assert got.tolist() == [True, True, True, True, True, True,
                            False, False, False]
    # 同一个 156 在 stride=9 下会被判为盘外 —— 这正是两个口径的分歧点
    r9, c9 = 156 // 9, 156 % 9
    assert not (r9 < 9 and c9 < 9), '本用例对 stride=9 应当不成立'
    assert policy_index_is_in_board(np.array([156]), 19).tolist() == [True]


# --------------------------------------------------------------------------- #
# 约定 3：top-K policy
# --------------------------------------------------------------------------- #
def test_topk_policy_keeps_largest_and_is_stable_on_ties():
    v = np.zeros((2, ACTION_SIZE), dtype=np.int16)
    v[0, 5] = 10
    v[0, 9] = 10                       # 与 5 同分
    v[0, 2] = 30
    v[1, 361] = 7
    idx, val = topk_policy(v, k=3)
    assert idx[0].tolist() == [2, 5, 9], '同分时应按索引升序（稳定）'
    assert val[0].tolist() == [30, 10, 10]
    assert 361 in idx[1].tolist(), 'pass 槽必须能被 topk 选中'


def test_topk_policy_rejects_wrong_width():
    with pytest.raises(ValueError, match='形状不符'):
        topk_policy(np.zeros((2, 361), dtype=np.int16), k=4)


def test_policy_dense_from_sparse_renormalizes_and_accepts_numpy():
    idx = np.array([[2, 5, 9]])
    val = np.array([[30.0, 10.0, 10.0]])
    d = policy_dense_from_sparse(idx, val, ACTION_SIZE)
    assert isinstance(d, torch.Tensor)
    assert float(d.sum()) == pytest.approx(1.0)
    assert float(d[0, 2]) == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# 约定 4：64/80 列布局分派（纯合成，永远跑）
# --------------------------------------------------------------------------- #
def test_layout_dispatch_accepts_both_column_counts():
    for net, cols, has_q in (('kata1-tf3-b11c768', 80, True),
                             ('kata1-zhizi-b40c768nbt', 64, True),
                             ('zzb28c512nfd4', 64, False)):
        d = read_katago_npz(_Npz(_fake_npz(cols=cols, has_q=has_q)), net)
        assert d['_layout']['cols'] == cols
        assert d['_layout']['has_qvalue'] is has_q
        assert d['_layout']['network'] in GLOBAL_TARGET_LAYOUT


def test_unknown_network_raises_instead_of_guessing():
    """未登记网络必须**报错**。硬猜会让权重列静默错位。"""
    with pytest.raises(KatagoNpzLayoutError, match='未登记的网络'):
        read_katago_npz(_Npz(_fake_npz()), 'kata1-tf9-b99c999')


def test_column_count_mismatch_raises():
    with pytest.raises(KatagoNpzLayoutError, match='登记为 80 列'):
        read_katago_npz(_Npz(_fake_npz(cols=64)), 'kata1-tf3-b11c768')


def test_qvalue_presence_mismatch_raises():
    with pytest.raises(KatagoNpzLayoutError, match='Q 值存在性'):
        read_katago_npz(_Npz(_fake_npz(cols=64, has_q=False)),
                        'kata1-zhizi-b40c768nbt')


def test_network_can_be_inferred_only_when_unambiguous():
    d = read_katago_npz(_Npz(_fake_npz(cols=80)))
    assert d['_layout']['network'] == 'kata1-tf3-b11c768'
    with pytest.raises(KatagoNpzLayoutError, match='无法唯一反推'):
        read_katago_npz(_Npz(_fake_npz(cols=64)))     # 两个网络都是 64 列


# --------------------------------------------------------------------------- #
# 标签契约 → 12 项 loss（合成数据，永远跑）
# --------------------------------------------------------------------------- #
def _fake_out(b):
    g = torch.Generator().manual_seed(3)
    return {
        'policy_logits': torch.randn(b, 2, ACTION_SIZE, generator=g),
        'outcome_logits': torch.randn(b, 3, generator=g),
        'score_mean': torch.randn(b, generator=g),
        'score_stdev': torch.rand(b, generator=g) * 10 + 1,
        'lead': torch.randn(b, generator=g),
        'ownership_pretanh': torch.randn(b, 1, 19, 19, generator=g),
        'scoring': torch.randn(b, 1, 19, 19, generator=g),
        'futurepos': torch.randn(b, 2, 19, 19, generator=g),
        'seki_logits': torch.randn(b, 4, 19, 19, generator=g),
        'scorebelief_logits': torch.randn(b, 842, generator=g),
    }


def test_to_v7_labels_feeds_the_12_item_loss():
    """合成的 stdata → 标签 dict → 12 项 loss 端到端跑通。"""
    d = _fake_npz(n=6, board=19)
    lb = to_v7_labels(read_katago_npz(_Npz(d), 'kata1-tf3-b11c768'))
    assert lb['_kept'] == 6 and lb['_dropped'] == 0
    assert lb['spatial'].shape == (6, SPATIAL_CHANNELS, 19, 19)
    assert lb['board_mask'].all(), '19×19 的 on-board 掩码应全 True'
    assert lb['ownership'].shape == (6, 1, BOARD_STRIDE, BOARD_STRIDE)
    assert lb['futurepos'].shape == (6, 2, BOARD_STRIDE, BOARD_STRIDE)
    assert float(lb['score_distr'].sum(1).mean()) == pytest.approx(1.0, abs=1e-6)
    res = KataGoV7Loss()(_fake_out(6), lb)
    assert torch.isfinite(res['loss'])
    assert len(res['terms']) == 12


def test_to_v7_labels_drops_non_19x19_and_reports_the_count():
    """混了小盘时必须**丢掉并记账** —— 静默丢是最难查的一类数据问题。"""
    sp = np.zeros((4, SPATIAL_CHANNELS, BOARD_STRIDE, BOARD_STRIDE), dtype=bool)
    sp[0, 0, :19, :19] = True
    sp[1, 0, :9, :9] = True
    sp[2, 0, :13, :13] = True
    sp[3, 0, :10, :11] = True                       # 非方阵
    d = _fake_npz(n=4, board=19)
    d['binaryInputNCHWPacked'] = _pack_spatial(sp)
    lb = to_v7_labels(read_katago_npz(_Npz(d), 'kata1-tf3-b11c768'))
    assert lb['_kept'] == 1 and lb['_dropped'] == 3
    assert lb['outcome'].tolist() == [0], 'outcome 0 = 胜（fake 里 g[:,0]=1）'
    assert lb['spatial'].shape[0] == 1


def test_seki_weight_defaults_to_zero_not_ownership():
    """ 没有 w_seki 时必须按 0 处理：stdata 的 seki 极稀有，复用 w_ownership
    等于让 seki 头在 99.99% 的行上被推向「全中性」，还被 #12 的 ×8 放大。"""
    d = _fake_npz(n=4)
    lb = to_v7_labels(read_katago_npz(_Npz(d), 'kata1-tf3-b11c768'))
    assert 'seki' not in lb['w']
    res = KataGoV7Loss()(_fake_out(4), lb)
    assert float(res['terms']['seki']) == 0.0


def test_explicit_w_seki_is_honoured():
    d = _fake_npz(n=4)
    lb = to_v7_labels(read_katago_npz(_Npz(d), 'kata1-tf3-b11c768'))
    lb['seki'] = np.zeros((4, 1, BOARD_CELLS), dtype=np.float32)
    lb['w']['seki'] = np.ones(4, dtype=np.float32)
    res = KataGoV7Loss()(_fake_out(4), lb)
    assert float(res['terms']['seki']) > 0.0


def test_all_zero_outcome_maps_to_noresult():
    """三列全 0（和棋）时 argmax 会落到 0=胜，必须显式改判成「无结果」。"""
    d = _fake_npz(n=2)
    d['globalTargetsNC'][:, 0:3] = 0.0
    lb = to_v7_labels(read_katago_npz(_Npz(d), 'kata1-tf3-b11c768'))
    assert lb['outcome'].tolist() == [2, 2]


# --------------------------------------------------------------------------- #
# 真实官方数据（需要 1.5 GB 的 tgz）
# --------------------------------------------------------------------------- #
def _first_npz(archive, limit=1):
    """流式取前 ``limit`` 个 npz。

     **按行取，不要按成员取**：`getmembers()` 要把 1.5 GB 的 tgz 整个解一遍
    才能列出成员，实测要几分钟；而我们需要的是「前 N 个里有 19×19 的行」，
    顺序流式取通常几十个就够。
    """
    out = []
    t = tarfile.open(archive, 'r|gz')
    try:
        for m in t:
            if m.name.endswith('.npz'):
                out.append(np.load(io.BytesIO(t.extractfile(m).read())))
                if len(out) >= limit:
                    break
    finally:
        t.close()
    return out


def _first_with_19x19(archive, max_files=40):
    """返回第一个含 19×19 行的 npz（实测头几个文件常常整批是小盘）。"""
    t = tarfile.open(archive, 'r|gz')
    try:
        for i, m in enumerate(t):
            if not m.name.endswith('.npz'):
                continue
            d = np.load(io.BytesIO(t.extractfile(m).read()))
            if int((board_size_from_packed(d['binaryInputNCHWPacked']) == 19).sum()):
                return d, i + 1
            if i + 1 >= max_files:
                break
    finally:
        t.close()
    return None, max_files


@needs_stdata
def test_real_stdata_score_distr_matches_official_bin_offset():
    """真实数据上的 scorebelief 桶偏移：与 `build_score_distr_target` 同口径。

    取 stdata 里 score≠0 的行，按 col20 算出应有桶位，再核对 npz 里那两个
    非零桶**逐位相等**。这比任何合成 fixture 都强。
    """
    from src.networks.katago_v7_loss import build_score_distr_target
    # 前几个文件常常整批是小盘（实测 2026-08-25 的头几个文件 19×19 占比 0），
    # 取太少会永远 skip 掉这条 —— 那是本文件里最强的一条断言，不能靠运气。
    raw, scanned = _first_with_19x19(ARCHIVE_0825)
    if raw is None:
        pytest.skip(f'前 {scanned} 个文件里没有 19×19 的行')
    d = read_katago_npz(_Npz(raw), 'kata1-tf3-b11c768')
    lb = to_v7_labels(d)
    assert lb['_kept'] > 0
    nz_rows = np.nonzero(lb['score'] != 0.0)[0]
    if nz_rows.size == 0:
        pytest.skip('抽到的局没有非和棋的行')
    for i in nz_rows[:8]:
        center = int(np.rint(lb['score'][i]))
        lam = float(lb['score'][i]) - (center - 0.5)
        upper = int(np.clip(round(lam * 100), 0, 100))
        want = build_score_distr_target(torch.tensor([center]),
                                        torch.tensor([upper]))[0].numpy()
        assert np.allclose(want, lb['score_distr'][i], atol=1e-6), \
            f'score={lb["score"][i]} 的桶分布与官方不一致'


@needs_stdata
def test_real_stdata_unpack_gives_square_board_mask_for_19x19():
    d = read_katago_npz(_Npz(_first_npz(ARCHIVE_0825, 1)[0]),
                        'kata1-tf3-b11c768')
    sp = d['_spatial']
    bs = d['_board_size']
    for i in range(min(20, sp.shape[0])):
        s = int(bs[i])
        if s == 0:
            continue
        cells = int(sp[i, 0].sum())
        assert cells == s * s, f'ch0 掩码与反推边长不符：{cells} vs {s}²'
        assert int(sp[i, 1, s:, :].sum()) == 0, '盘外不该有 pla 棋子'
        assert int(sp[i, 2, s:, :].sum()) == 0, '盘外不该有 opp 棋子'
