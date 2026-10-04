"""V7 预算特征分片的加载器（``src/data/v7_packed_dataset.py``）。

背景（2026-10-04）
------------------
``--v7 1`` 跑真实数据时崩在 ``KeyError: 'boards'`` ——
``SupervisedDataset`` 只吃 board 级布局（``boards`` / ``my_hist`` / ``moves`` …），
而 ``data/stdata_v7_s*.npz`` 是另一种布局：22 通道已**预算好并位打包**
（``spatial_packed``）。读写两端一直没接上。

本文件钉住这个适配层的每条约定 —— 尤其���那几条「形状完全合法、不报错，
但监督信号整体错位」的：

1. ``next_move`` **不能**走 ``moves[idxs+1]``（board 级语义），分片每行自带答案
2. ``policy_player_prob`` 是**访问计数**不是概率，必须归一化
3. ``sb_center`` / ``sb_upper`` 是**整数对**不是两个桶的概率
4. dihedral 增强必须对空间平面与所有平面标签**同步**施加
5. ``var_time_left`` 缺席时**跳过**而非喂 0
"""
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.v7_packed_dataset import (  # noqa: E402
    ACTION_SIZE,
    POLICY_TOPK,
    V7PackedDataError,
    V7PackedDataset,
    _permute_move_scalar,
    _permute_plane,
    load_v7_packed,
    resolve_shards,
)

BS = 19
N_ROWS = 24
TINY = ROOT / 'tmp' / 'coding' / 'tiny_v7.npz'


# --------------------------------------------------------------------------- #
# 合成夹具（不依赖 340MB 真实分片）
# --------------------------------------------------------------------------- #
def _make_shard(path, n=N_ROWS, *, with_vtl=False, seed=0):
    rng = np.random.default_rng(seed)
    packed = np.zeros((n, 22, 46), np.uint8)
    packed[:, 0, :] = 0xFF           # ch0「on board」= 361 bit 全 1
    packed[:, 1, :] = 0xFF
    # 真实分片里 `policy_player_prob` 是**按降序排好**的访问计数，
    # 所以 `rank[:,0]` 必然是 argmax。夹具必须复现这个不变量，
    # 否则「argmax == rank[:,0]」这条断言测的是夹具的随机性而不是代码。
    prob = np.sort(rng.integers(1, 500, size=(n, POLICY_TOPK)).astype(np.float32),
                   axis=1)[:, ::-1].copy()
    rank = np.stack([rng.permutation(BS * BS)[:POLICY_TOPK] for _ in range(n)]
                    ).astype(np.int16)
    out = {
        'spatial_packed': packed,
        'global': rng.normal(size=(n, 19)).astype(np.float32),
        'policy_player_rank': rank,
        'policy_player_prob': prob,
        'policy_opp_rank': rank.copy(),
        'policy_opp_prob': prob.copy(),
        'outcome': rng.integers(0, 3, size=n).astype(np.int64),
        'game_weight': np.ones(n, np.float32),
        'game_ids': np.arange(n, dtype=np.int32),
        'score': rng.normal(scale=20, size=n).astype(np.float32),
        'ownership': rng.random((n, 1, BS, BS)).astype(np.float32),
        'futurepos': rng.random((n, 2, BS, BS)).astype(np.float32),
        'scoring': rng.random((n, 1, BS, BS)).astype(np.float32),
        'seki': rng.random((n, 1, BS, BS)).astype(np.float32),
        'w_score': np.ones(n, np.float32),
        'w_lead': (rng.random(n) > 0.5).astype(np.float32),
        'w_ownership': np.ones(n, np.float32),
        'w_policy_opp': np.ones(n, np.float32),
        'w_futurepos': np.ones(n, np.float32),
        'w_scoring': np.ones(n, np.float32),
    }
    if with_vtl:
        out['var_time_left'] = rng.random(n).astype(np.float32) * 30.0
    np.savez_compressed(str(path), **out)
    return str(path)


@pytest.fixture(scope='module')
def shard(tmp_path_factory):
    d = tmp_path_factory.mktemp('v7packed')
    return _make_shard(d / 's0.npz')


@pytest.fixture(scope='module')
def ds(shard):
    return V7PackedDataset([shard])


# --------------------------------------------------------------------------- #
# 布局
# --------------------------------------------------------------------------- #
def test_packed_spatial_is_unpacked_to_22_channels(ds):
    sp = ds.sample_spatial(np.arange(4, dtype=np.int64))
    assert sp.shape == (4, 22, BS, BS)
    assert sp.dtype == np.bool_
    # ch0 造的全是 1
    assert sp[:, 0].all()


def test_global_is_19_dims(ds):
    gl = ds.sample_global(np.arange(4, dtype=np.int64))
    assert gl.shape == (4, 19)
    assert gl.dtype == np.float32


def test_missing_required_key_raises(tmp_path):
    p = tmp_path / 'bad.npz'
    np.savez_compressed(str(p), spatial_packed=np.zeros((4, 22, 46), np.uint8))
    with pytest.raises(V7PackedDataError, match='缺少必需键'):
        V7PackedDataset([str(p)])


def test_wrong_spatial_shape_raises(tmp_path):
    p = tmp_path / 'bad2.npz'
    # ⚠ `global` 是 Python 关键字，不能当 kwarg 传 ⇒ 用 dict 形式。
    #   （分片里的键**就叫** `global`。）
    np.savez_compressed(
        str(p),
        **{
            'spatial_packed': np.zeros((4, 12, 46), np.uint8),
            'global': np.zeros((4, 19), np.float32),
            'policy_player_rank': np.zeros((4, POLICY_TOPK), np.int16),
            'policy_player_prob': np.zeros((4, POLICY_TOPK), np.float32),
            'outcome': np.zeros(4, np.int64),
            'game_weight': np.ones(4, np.float32),
        })
    with pytest.raises(V7PackedDataError, match='22,46'):
        V7PackedDataset([str(p)])


# --------------------------------------------------------------------------- #
# 🔴 语义陷阱（形状都合法，不报错但监督信号错）
# --------------------------------------------------------------------------- #
def test_next_move_is_this_row_not_the_next(ds):
    """🔴 分片每行自带答案；走 board 级的 ``moves[idxs+1]`` 会整体错位一行。"""
    idxs = np.arange(8, dtype=np.int64)
    lbl = ds._build_labels(idxs, np.zeros(8, np.int64), False)
    want = ds._gather('policy_player_rank', idxs)[:, 0]
    assert np.array_equal(lbl['next_move'], want)


def test_moves_property_matches_next_move(ds):
    idxs = np.arange(8, dtype=np.int64)
    lbl = ds._build_labels(idxs, np.zeros(8, np.int64), False)
    assert np.array_equal(ds.moves[idxs], lbl['next_move'])


def test_soft_policy_is_normalized_not_raw_counts(ds):
    """🔴 ``policy_player_prob`` 是**访问计数**（和可达数百），必须归一化。"""
    idxs = np.arange(8, dtype=np.int64)
    lbl = ds._build_labels(idxs, np.zeros(8, np.int64), False)
    raw = ds._gather('policy_player_prob', idxs).astype(np.float64)
    raw_sum = raw.sum(axis=1)
    assert raw_sum.min() > 10.0, '夹具的计数不该已经是概率（应是访问次数）'
    np.testing.assert_allclose(lbl['soft'].sum(axis=1), 1.0, atol=1e-5)
    # 且 argmax 必须落在 rank[:,0] 上（计数最大 ⇒ top-1 命中）
    assert np.array_equal(lbl['soft'].argmax(axis=1),
                          ds._gather('policy_player_rank', idxs)[:, 0])


def test_soft_only_touches_the_topk_slots(ds):
    idxs = np.arange(4, dtype=np.int64)
    lbl = ds._build_labels(idxs, np.zeros(4, np.int64), False)
    nz = np.nonzero(lbl['soft'][0])[0]
    assert len(nz) <= POLICY_TOPK
    assert nz.max() < BS * BS          # 只落在棋盘上，不含 pass 索引


def test_scorebelief_is_integer_pair_not_probabilities(ds):
    """🔴 ``sb_center`` 是 ``round(score)``、``sb_upper`` 是 ``round(λ·100)``。

    喂浮点概率会被 ``build_score_distr_target`` 的 ``scatter_`` 静默算错。
    """
    idxs = np.arange(8, dtype=np.int64)
    lbl = ds._build_labels(idxs, np.zeros(8, np.int64), False)
    score = ds._gather('score', idxs).astype(np.float64)
    assert np.array_equal(lbl['sb_center'], np.rint(score).astype(np.int64))
    lam = score - (np.rint(score) - 0.5)
    assert np.allclose(lbl['sb_upper'], np.rint(np.clip(lam, 0, 1) * 100))
    assert lbl['sb_upper'].min() >= 0.0 and lbl['sb_upper'].max() <= 100.0


def test_scorebelief_pair_feeds_the_loss_builder(ds):
    """端到端：产出的 (center, upper) 必须能被 loss 侧的展开函数吃下。"""
    import torch
    from src.networks.katago_v7_loss import build_score_distr_target

    idxs = np.arange(8, dtype=np.int64)
    lbl = ds._build_labels(idxs, np.zeros(8, np.int64), False)
    tgt = build_score_distr_target(
        torch.as_tensor(lbl['sb_center']), torch.as_tensor(lbl['sb_upper']))
    assert tgt.shape == (8, 842)
    np.testing.assert_allclose(tgt.sum(-1).numpy(), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# dihedral 增强的一致性
# --------------------------------------------------------------------------- #
def test_permute_move_scalar_is_a_permutation():
    allm = np.arange(BS * BS)
    for t in range(8):
        moved = _permute_move_scalar(allm, t, BS)
        assert sorted(moved.tolist()) == allm.tolist(), f'tform={t} 不是置换'


def test_permute_plane_keeps_values():
    rng = np.random.default_rng(0)
    p = rng.random((2, BS, BS)).astype(np.float32)
    for t in range(8):
        q = _permute_plane(p, t)
        assert q.shape == p.shape
        np.testing.assert_allclose(np.sort(q.ravel()), np.sort(p.ravel()), atol=1e-6)


def test_augment_moves_soft_and_next_move_consistently(ds):
    """🔴 增强必须让「空间平面 / soft / next_move」指向**同一格**。

    若只转空间平面而标签没转（或反之），形状全对、不报错，但监督信号指向
    错误的格点 —— 这类 bug 能把 top1 训到接近 0。
    """
    idxs = np.arange(6, dtype=np.int64)
    tforms = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    plain = ds._build_labels(idxs, np.zeros(6, np.int64), False)
    aug = ds._build_labels(idxs, tforms, True)
    for b, t in enumerate(tforms):
        want = _permute_move_scalar(
            np.asarray([plain['next_move'][b]]), int(t), BS)[0]
        assert aug['next_move'][b] == want, f'行 {b} tform={t} 的 next_move 没跟着转'
        # soft 的 argmax 必须落在变换后的落点上
        assert int(aug['soft'][b].argmax()) == want


def test_augment_applies_to_all_plane_labels(ds):
    idxs = np.arange(4, dtype=np.int64)
    tforms = np.array([1, 2, 3, 5], dtype=np.int64)
    plain = ds._build_labels(idxs, np.zeros(4, np.int64), False)
    aug = ds._build_labels(idxs, tforms, True)
    for key in ('ownership', 'scoring', 'seki', 'future'):
        for b, t in enumerate(tforms):
            np.testing.assert_allclose(
                np.sort(aug[key][b].ravel()), np.sort(plain[key][b].ravel()),
                atol=1e-6, err_msg=f'{key} 行 {b} 的取值集合变了')


# --------------------------------------------------------------------------- #
# var_time_left
# --------------------------------------------------------------------------- #
def test_var_time_left_absent_is_skipped_not_zero_filled(shard):
    """⚠ 缺席时**不产出该键** —— loss 侧整项跳过。

    喂 0 会把这一路往「方差恒 0」硬拉，比不训练更糟。
    """
    ds = V7PackedDataset([shard])
    assert not ds._has('var_time_left')
    lbl = ds._build_labels(np.arange(4, dtype=np.int64), np.zeros(4, np.int64), False)
    assert 'var_time_left' not in lbl
    assert 'var_time_left' in ds.describe(), 'describe() 应提示缺这一项'


def test_var_time_left_present_is_passed_through(tmp_path):
    p = _make_shard(tmp_path / 'vt.npz', with_vtl=True)
    ds = V7PackedDataset([p])
    idxs = np.arange(4, dtype=np.int64)
    lbl = ds._build_labels(idxs, np.zeros(4, np.int64), False)
    assert 'var_time_left' in lbl
    np.testing.assert_allclose(
        lbl['var_time_left'], ds._gather('var_time_left', idxs))


# --------------------------------------------------------------------------- #
# 多分片
# --------------------------------------------------------------------------- #
def test_multi_shard_gather_is_correct(tmp_path):
    a = _make_shard(tmp_path / 'a.npz', n=10, seed=1)
    b = _make_shard(tmp_path / 'b.npz', n=10, seed=2)
    ds = V7PackedDataset([a, b])
    assert len(ds) == 20
    # 跨分片取行：每行都应能取到，且与单分片读同一行一致
    single_b = V7PackedDataset([b])
    idxs = np.arange(10, 20, dtype=np.int64)
    np.testing.assert_array_equal(
        ds._gather('outcome', idxs), single_b._gather('outcome', np.arange(10)))
    assert ds._gather('spatial_packed', idxs).shape[0] == 10


def test_game_ids_are_globally_unique_across_shards(tmp_path):
    """🔴 分片内 ``game_ids`` 是**块内局部行号**，分片间会重复。

    不去重就会把第 0 片和第 1 片的 id=0 当成「同一局」——
    按棋局切 train/eval 时同一局会同时落在两侧，eval 指标虚高。
    """
    a = _make_shard(tmp_path / 'ga.npz', n=10, seed=3)
    b = _make_shard(tmp_path / 'gb.npz', n=10, seed=3)   # 同 seed ⇒ 局部 id 相同
    ds = V7PackedDataset([a, b])
    gid = ds.game_ids
    assert gid is not None
    assert len(np.unique(gid)) == 20, '跨分片的 game_ids 必须全局唯一'


def test_resolve_shards_accepts_file_and_dir(tmp_path, shard):
    assert resolve_shards(shard) == [shard]
    d = tmp_path / 'shards'
    d.mkdir()
    for i in range(3):
        _make_shard(d / ('s%d.npz' % i), n=4)
    got = resolve_shards(str(d))
    assert len(got) == 3 and got == sorted(got)


def test_resolve_shards_rejects_empty_dir(tmp_path):
    d = tmp_path / 'empty'
    d.mkdir()
    with pytest.raises(V7PackedDataError, match=r'没有 \*\.npz'):
        resolve_shards(str(d))


# --------------------------------------------------------------------------- #
# 真实夹具（存在才跑）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(TINY), reason='需要 tmp/coding/tiny_v7.npz')
def test_real_shard_fixture_loads():
    ds = load_v7_packed(str(TINY))
    assert len(ds) > 0
    sp = ds.sample_spatial(np.arange(2, dtype=np.int64))
    assert sp.shape == (2, 22, BS, BS)
    lbl = ds._build_labels(np.arange(2, dtype=np.int64), np.zeros(2, np.int64), False)
    np.testing.assert_allclose(lbl['soft'].sum(axis=1), 1.0, atol=1e-5)