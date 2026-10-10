"""`pos_hash` 与 `kata_labels.npz` join 的测试。

两条要害：
1. **散列必须与 `probe0_join.py` / `label_sgf.py` 逐位一致** —— 已经落盘的
   `kata_labels.npz` 里的 `pos_hash` 列是用旧实现算的。改种子 / 改公式都会
   让 join 静默变空，而症状看不出原因。
2. **join 必须是多对多里可控的那种** —— 同一局面在多局里出现
   （transposition），`np.intersect1d` 会把对应关系丢掉。
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.kata_label_join import (  # noqa: E402
    REQUIRED_KEYS, build_sidecar, duplicate_factor, join_by_hash,
    load_kata_labels, scan_dataset_hashes,
)
from src.data.pos_hash import (  # noqa: E402
    BOARD, BOARD_CELLS, pos_hash_block, pos_hash_one,
)

LABELS = os.path.join(ROOT, 'data', 'labels', 'kata_labels.npz')
DATASET = os.path.join(ROOT, 'data', 'sgf_19x19_full.npz')
needs_labels = pytest.mark.skipif(
    not os.path.isfile(LABELS), reason='需要 data/labels/kata_labels.npz（不在仓库里）')


# --------------------------------------------------------------------------- #
# 散列
# --------------------------------------------------------------------------- #
def _rand_positions(n, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.integers(-1, 2, (n, BOARD, BOARD)).astype(np.int8)
    tp = rng.integers(-1, 2, n).astype(np.int8)
    ko = rng.integers(-1, 20, n).astype(np.int16)
    return b, tp, ko


def test_pos_hash_is_deterministic():
    b, tp, ko = _rand_positions(16)
    assert np.array_equal(pos_hash_block(b, tp, ko), pos_hash_block(b, tp, ko))


def test_pos_hash_is_uint64_and_sensitive_to_every_component():
    """盘面 / to_play / ko **三者任一**变化都必须改散列。

     全程必须在 uint64 下算。int64 乘法溢出后再 `astype(uint64)` 会触发
    `RuntimeWarning` 且高位置换逻辑不可控 —— 首版就是这么坏的，唯一性与
    单格敏感性全失，而症状是「join 偶尔对上」。这里断言没有 overflow 警告。
    """
    b, tp, ko = _rand_positions(8, seed=1)
    base = pos_hash_block(b, tp, ko)
    assert base.dtype == np.uint64

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        b1 = b.copy()
        b1[0, 0, 0] = np.int8(int(b1[0, 0, 0]) * -1 if b1[0, 0, 0] else 1)
        assert pos_hash_block(b1, tp, ko)[0] != base[0], '改一格盘面散列未变'
        tp2 = tp.copy()
        tp2[0] = np.int8(-tp2[0]) if tp2[0] else np.int8(1)
        assert pos_hash_block(b, tp2, ko)[0] != base[0], '改 to_play 散列未变'
        ko2 = ko.copy()
        ko2[0] = np.int16(int(ko2[0]) + 1)
        assert pos_hash_block(b, tp, ko2)[0] != base[0], '改 ko 散列未变'


def test_pos_hash_one_matches_block():
    b, tp, ko = _rand_positions(4, seed=2)
    blk = pos_hash_block(b, tp, ko)
    one = np.array([pos_hash_one(b[i], int(tp[i]), int(ko[i])) for i in range(4)])
    assert np.array_equal(one, blk), '单点版与批量版不一致'


def test_probe0_join_reexport_is_bit_identical():
    """`probe0_join` 改成 re-export 后，散列必须**逐位不变** ——
    `label_sgf.py` 现有的 `_p0.pos_hash_block` 调用与已落盘的 `pos_hash` 列
    都依赖这一点。"""
    import importlib
    m = importlib.import_module('scripts.probe0_join')
    b, tp, ko = _rand_positions(6, seed=3)
    assert np.array_equal(m.pos_hash_block(b, tp, ko), pos_hash_block(b, tp, ko))


def test_empty_board_and_single_stone_differ():
    empty = np.zeros((1, BOARD, BOARD), dtype=np.int8)
    one = empty.copy()
    one[0, 3, 4] = 1
    assert pos_hash_one(empty, 1, -1) != pos_hash_one(one, 1, -1)


def test_pos_hash_rejects_wrong_board_shape():
    with pytest.raises(ValueError, match='每行应有 361 格'):
        pos_hash_block(np.zeros((2, 18, 18), dtype=np.int8),
                       np.ones(2, np.int8), np.zeros(2, np.int16))


# --------------------------------------------------------------------------- #
# join
# --------------------------------------------------------------------------- #
def test_join_matches_exact_positions():
    b, tp, ko = _rand_positions(10, seed=4)
    h = pos_hash_block(b, tp, ko)
    rows, labs = join_by_hash(h, h[[3, 7]], max_repeats=1)
    assert rows.tolist() == [3, 7]
    assert labs.tolist() == [0, 1]
    assert np.array_equal(h[rows], h[[3, 7]])


def test_join_with_transposition_takes_first_match_only():
    """同一局面出现两次时，`max_repeats=1` 只挂第一个 —— 否则高频布局变例
    会按重复次数压过别处。"""
    b, tp, ko = _rand_positions(6, seed=5)
    b[3] = b[0]
    tp[3] = tp[0]
    ko[3] = ko[0]
    h = pos_hash_block(b, tp, ko)
    rows, _ = join_by_hash(h, h[[0]], max_repeats=1)
    assert rows.tolist() == [0]
    rows2, _ = join_by_hash(h, h[[0]], max_repeats=2)
    assert rows2.tolist() == [0, 3], 'max_repeats=2 应多挂一个'


def test_join_is_sorted_by_row_index():
    b, tp, ko = _rand_positions(20, seed=6)
    h = pos_hash_block(b, tp, ko)
    rows, labs = join_by_hash(h, h[[15, 2, 9, 0]], max_repeats=1)
    assert rows.tolist() == sorted(rows.tolist())
    assert rows.tolist() == [0, 2, 9, 15]
    assert labs.tolist() == [3, 1, 2, 0], '标签下标必须跟着行走'


def test_join_keeps_only_the_first_duplicate_on_the_label_side():
    b, tp, ko = _rand_positions(5, seed=7)
    h = pos_hash_block(b, tp, ko)
    lab = np.concatenate([h[[1]], h[[1]]])       # 标签侧同一局面标了两次
    rows, labs = join_by_hash(h, lab, max_repeats=1)
    assert rows.tolist() == [1]
    assert labs.tolist() == [0], '应保留标签侧首次出现的那一行'


def test_join_with_no_overlap_returns_empty_not_error():
    b, tp, ko = _rand_positions(5, seed=8)
    other = _rand_positions(5, seed=9)
    rows, labs = join_by_hash(pos_hash_block(b, tp, ko),
                              pos_hash_block(*other))
    assert rows.size == 0 and labs.size == 0
    assert rows.dtype == np.int64


def test_join_handles_empty_inputs():
    e = np.empty(0, dtype=np.uint64)
    rows, labs = join_by_hash(e, np.array([1, 2], dtype=np.uint64))
    assert rows.size == 0 and labs.size == 0
    rows, labs = join_by_hash(np.array([1, 2], dtype=np.uint64), e)
    assert rows.size == 0 and labs.size == 0


def test_duplicate_factor_counts_dataset_occurrences():
    b, tp, ko = _rand_positions(9, seed=10)
    b[5] = b[1]
    tp[5] = tp[1]
    ko[5] = ko[1]
    b[8] = b[1]
    tp[8] = tp[1]
    ko[8] = ko[1]
    h = pos_hash_block(b, tp, ko)
    counts, uniq = duplicate_factor(h, h[[1, 2]])
    lut = dict(zip(uniq.tolist(), counts.tolist()))
    assert lut[int(h[1])] == 3
    assert lut[int(h[2])] == 1


# --------------------------------------------------------------------------- #
# sidecar
# --------------------------------------------------------------------------- #
def _fake_labels_npz(path, dataset_hashes, n_per=1):
    idx = np.arange(4)
    n = len(idx)
    pol = np.zeros((n, 362), dtype=np.float16)
    pol[np.arange(n), idx] = 1.0          # 每行**只**放一个 1，行和才是 1
    np.savez(path,
             policy=pol,
             pos_hash=dataset_hashes[idx],
             root_win=np.full(n, 0.6, np.float32),
             score_mean=np.full(n, 2.0, np.float32),
             score_stdev=np.full(n, 7.0, np.float32),
             visits=np.full(n, 36, np.int16),
             game_idx=np.arange(n, dtype=np.int32),
             move_idx=np.arange(n, dtype=np.int32))
    return idx


def _fake_dataset_npz(path, b, tp, ko):
    np.savez(path, boards=b.astype(np.int8), to_play=tp, ko=ko)
    return b, tp, ko


def test_sidecar_roundtrip_and_diagnostics(tmp_path):
    b, tp, ko = _rand_positions(12, seed=11)
    h = pos_hash_block(b, tp, ko)
    _fake_dataset_npz(str(tmp_path / 'ds.npz'), b, tp, ko)
    _fake_labels_npz(str(tmp_path / 'lb.npz'), h)
    out = str(tmp_path / 'side.npz')
    diag = build_sidecar(str(tmp_path / 'ds.npz'), str(tmp_path / 'lb.npz'), out)
    assert diag['matched_rows'] == 4
    assert diag['dataset_rows'] == 12
    assert diag['hit_rate_vs_labels'] == 1.0
    z = np.load(out)
    assert z['row_index'].shape == (4,)
    assert z['policy'].shape == (4, 362)
    assert z['row_index'].tolist() == [0, 1, 2, 3], '应指向被标到的前 4 行'
    assert np.array_equal(z['pos_hash'], h[:4]), '回写的散列要能对上源行'
    assert np.allclose(z['policy'].sum(1), 1.0, atol=2e-3)


def test_sidecar_writes_nothing_when_no_overlap(tmp_path):
    """「无交集」是要立刻看见的结论，不能静默不写文件。"""
    b, tp, ko = _rand_positions(6, seed=12)
    _fake_dataset_npz(str(tmp_path / 'ds2.npz'), b, tp, ko)
    other = _rand_positions(6, seed=13)
    oh = pos_hash_block(*other)
    np.savez(str(tmp_path / 'lb2.npz'),
             policy=np.zeros((6, 362), np.float16),
             pos_hash=oh, root_win=np.zeros(6, np.float32),
             score_mean=np.zeros(6, np.float32),
             score_stdev=np.zeros(6, np.float32),
             visits=np.zeros(6, np.int16), game_idx=np.arange(6, dtype=np.int32),
             move_idx=np.arange(6, dtype=np.int32))
    out = str(tmp_path / 'side2.npz')
    diag = build_sidecar(str(tmp_path / 'ds2.npz'), str(tmp_path / 'lb2.npz'), out)
    assert diag['matched_rows'] == 0
    assert 'out' not in diag
    assert not os.path.exists(out)


def test_load_labels_rejects_missing_key(tmp_path):
    """少一个键 = 那一路监督静默退化成全 0 标签，必须报错。"""
    p = str(tmp_path / 'bad.npz')
    np.savez(p, policy=np.zeros((3, 362), np.float16),
             pos_hash=np.zeros(3, np.uint64))
    with pytest.raises(KeyError, match='缺字段'):
        load_kata_labels(p)


def test_load_labels_rejects_wrong_policy_width(tmp_path):
    p = str(tmp_path / 'narrow.npz')
    d = {k: np.zeros(3) for k in REQUIRED_KEYS if k != 'policy'}
    d['policy'] = np.zeros((3, 361), np.float16)
    np.savez(p, **d)
    with pytest.raises(ValueError, match=r'\(N, 362\)'):
        load_kata_labels(p)


def test_scan_dataset_hashes_limit(tmp_path):
    b, tp, ko = _rand_positions(20, seed=14)
    _fake_dataset_npz(str(tmp_path / 'ds3.npz'), b, tp, ko)
    h, n = scan_dataset_hashes(str(tmp_path / 'ds3.npz'), limit=7, chunk=3)
    assert n == 7 and h.size == 7
    assert np.array_equal(h, pos_hash_block(b[:7], tp[:7], ko[:7]))


# --------------------------------------------------------------------------- #
# 真实数据
# --------------------------------------------------------------------------- #
@needs_labels
def test_real_sidecar_join_hits():
    """真实 `kata_labels.npz` 与真实主数据集的 join 命中率（只扫前 300 万行，
    免得单测跑几分钟）。"""
    diag = build_sidecar(DATASET, LABELS, str(os.devnull), limit=3_000_000)
    assert diag['label_rows'] > 0
    assert diag['matched_rows'] > 0, (
        f'join 命中 0 行：labels={diag["label_rows"]}，'
        f'扫了 {diag["dataset_rows"]} 行。若标签确实落在 move 100~200，'
        f'而 --limit 截到的前 300 万行还没走到那儿，就会这样。')
