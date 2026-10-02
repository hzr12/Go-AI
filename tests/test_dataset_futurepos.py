"""futurepos（未来 2 手位置）接进 `labels_dict`（A6）。

`labels_dict['future']` 的契约（完整版见 `SupervisedDataset.attach_futurepos`）
--------------------------------------------------------------------------
``future[b, h, r*bs+c]`` = 「点 (r,c) 上有**行 b 那一手该走子那一方的对手**的子」，
h=0 取第 ``i+8`` 行盘面、h=1 取第 ``i+32`` 行盘面（``FUTUREPOS_OFFSETS=(8,32)``）。

* 有效格取值域 ``{0,1}``；
* **无效格恒为 ``-1.0``**（`FUTUREPOS_SENTINEL`）—— 0 恰好是「对手一颗子都没占」
  这个合法标签，所以不能用 0 当哨兵；
* ``w['futurepos']`` = 两路**都**有效才 1（loss 的作用单位是整块 2×bs²，
  `_weighted_mean` 又不除 Σw）；
* ``w['futurepos_h0']`` / ``w['futurepos_h1']`` = 单路有效性。

本文件用**小规模合成数据**（9×9、几局几十行、落成真的 `.npy` 走
`mmap_mode='r'`，与生产口径一致），不碰真实 34.2M。

三种「不可用」情形各自有独立用例，因为它们是最容易静默出错的地方：
越界（局尾）/ 跨局 / ``gather_neighbors.valid`` 为 False。

跑：pytest tests/test_dataset_futurepos.py -v
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import (  # noqa: E402
    FUTUREPOS_SENTINEL, SupervisedDataset, _permute_future, permute_soft,
)
from src.data.feature_v7_gather import FUTUREPOS_OFFSETS  # noqa: E402

BS = 5
N_SQ = BS * BS
P8, P32 = FUTUREPOS_OFFSETS          # (8, 32)


# --------------------------------------------------------------------------- #
# 合成数据：三局，行 0..9 / 10..19 / 20..29。盘面用**每行一份随机的 ±1 图案**
# 乘上一个按行交替的颜色。
#
# ⚠ 图案必须同时满足三件事，缺一个测试就测不出东西：
#   ① **两种颜色都出现**（否则「对手占位」可能是全 0，测不出方向）；
#   ② **不对称**（棋盘格那种 (r+c) 奇偶图案在 8 个二面体对称下不变，会把
#      `permute_soft` 的方向 bug 整个掩盖掉）；
#   ③ 形状与 dtype 与生产一致（int8, (N,bs,bs)）。
# --------------------------------------------------------------------------- #
def _toy(lens=(10, 10, 10), seed=7, board_size=BS):
    n = int(sum(lens))
    bs = board_size
    rng = np.random.default_rng(seed)
    # ±1 随机图案（不是棋盘格）：两种颜色都在，且不对称。
    pat = (rng.integers(0, 2, size=(n, bs, bs)) * 2 - 1).astype(np.int8)
    color = np.array([1 if (i // 3) % 2 == 0 else -1 for i in range(n)],
                     dtype=np.int8)
    boards = (pat * color[:, None, None]).astype(np.int8)
    gids, tps, kos = [], [], []
    i = 0
    for gi, L in enumerate(lens):
        for _ in range(L):
            gids.append(gi)
            tps.append(1 if i % 2 == 0 else -1)
            kos.append(-1)
            i += 1
    return {
        'boards': boards,
        'my_hist': np.full((n, 3), -1, dtype=np.int16),
        'op_hist': np.full((n, 3), -1, dtype=np.int16),
        'ko': np.array(kos, dtype=np.int16),
        'moves': rng.integers(0, bs * bs, size=n).astype(np.int16),
        'values': rng.choice([-1, 1], size=n).astype(np.int8),
        'to_play': np.array(tps, dtype=np.int8),
        'game_ids': np.array(gids, dtype=np.int32),
    }


def _write_npy(path, arr):
    """落成**真的** `.npy`（走 `open_memmap`，与 `materialize_dataset` 同口径）。"""
    mm = np.lib.format.open_memmap(path, mode='w+', dtype=arr.dtype, shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm
    return path


def _materialized(tmp_path, data, keys=('boards', 'to_play', 'ko', 'game_ids')):
    return {k: _write_npy(str(tmp_path / f'{k}.npy'), data[k]) for k in keys}


def _expected_occupancy(data, row, offset):
    """手算「第 row+offset 行盘面上，对手的子」= 期望的 (N_SQ,) 0/1 掩码。"""
    b = data['boards'][row + offset].astype(np.int8)
    return (b * int(data['to_play'][row]) < 0).reshape(-1)


def _validity(data, row):
    """(h0, h1) 两个 bool：那一行第 h 路该不该有效（独立于被测代码算一遍）。"""
    n = len(data['boards'])
    gid = data.get('game_ids')
    out = []
    for off in (P8, P32):
        j = row + off
        out.append(bool(j < n and (gid is None or gid[j] == gid[row])))
    return tuple(out)


#: 三局各 70 行：0..69 / 70..139 / 140..209。
#: 在这上面可以挑出「两路都活」「只活一路」「两路都死」「界内但跨局」四类行：
LENS = (70, 70, 70)
BOTH_OK = 0        # 0/8/32 全在第 0 局
H0_ONLY = 40       # 48 在第 0 局（活）；72 在第 1 局（跨局，死）
BOTH_DEAD = 209    # 末行：+8 与 +32 双双越界


def _ds(data, source=None, **kw):
    """造一个**启用** futurepos 的 dataset。

    ``source`` 可以是 `_materialized()` 的返回（dict ⇒ 取其中的 boards.npy）、
    单个路径，或 None（走 `self.boards`）。
    """
    if isinstance(source, dict):
        source = source['boards']
    ds = SupervisedDataset(data, n_channels=12)
    ds.attach_futurepos(source, **kw)
    return ds


# --------------------------------------------------------------------------- #
# 1. 形状 / dtype / 取值域
# --------------------------------------------------------------------------- #
def test_shape_dtype_and_value_domain(tmp_path):
    data = _toy(LENS)
    ds = _ds(data, _materialized(tmp_path, data))
    _, _, _, d = ds.sample_batch_numpy(np.arange(6), augment=False, labels=True)
    assert d['future'].shape == (6, 2, N_SQ), d['future'].shape
    assert d['future'].dtype == np.float32, d['future'].dtype
    for k in ('futurepos', 'futurepos_h0', 'futurepos_h1'):
        assert d['w'][k].shape == (6,), f"w['{k}'] 形状 {d['w'][k].shape}"
        assert d['w'][k].dtype == np.float32
    # 有效格的取值域必须是 {0,1}（不许出现别的数）
    v = d['w']['futurepos'].astype(bool)
    assert v.all(), '本批 6 行两路都该有效'
    assert set(np.unique(d['future'][v]).tolist()) <= {0.0, 1.0}


def test_labels_dict_and_w_keep_their_existing_keys(tmp_path):
    """启用 futurepos **只加键、且只动 `futurepos` 这一个既有键**。

    🔴 `w['futurepos']` 正是 A6 要改的那个键（恒 0 → 有效性掩码），所以它是
    **唯一**允许变的既有键；其余每个键必须与未启用时**逐字节相同**。
    """
    data = _toy(LENS)
    off = SupervisedDataset(data, n_channels=12)
    on = _ds(data, _materialized(tmp_path, data))
    idxs = np.arange(6)
    _, _, _, d_off = off.sample_batch_numpy(idxs, augment=False, labels=True)
    _, _, _, d_on = on.sample_batch_numpy(idxs, augment=False, labels=True)
    assert set(d_off) == set(d_on), '既有键不许少也不许多'
    assert set(d_off['w']) < set(d_on['w']), 'w 只应新增键'
    assert set(d_on['w']) - set(d_off['w']) == {'futurepos_h0', 'futurepos_h1'}
    changed = set()
    for k in d_off:
        if k == 'w':
            for wk in d_off[k]:
                if d_on[k][wk].tobytes() != d_off[k][wk].tobytes():
                    changed.add(f"w['{wk}']")
        elif d_on[k].tobytes() != d_off[k].tobytes():
            changed.add(k)
    assert changed <= {'w[\'futurepos\']', 'future'}, \
        f'启用 futurepos 改了不该改的既有键：{sorted(changed)}'
    assert d_off['w']['futurepos'].sum() == 0.0
    assert d_on['w']['futurepos'].tolist() == [1.0] * 6


# --------------------------------------------------------------------------- #
# 2. 取值正确（手算，不用真 34.2M）
# --------------------------------------------------------------------------- #
def test_values_match_hand_computed_opponent_occupancy(tmp_path):
    data = _toy(LENS)
    ds = _ds(data, _materialized(tmp_path, data))
    idxs = np.array([0, 1, 2, 40, 69])
    _, _, _, d = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    for h, off in enumerate((P8, P32)):
        for b, i in enumerate(idxs):
            want_valid = _validity(data, int(i))[h]
            assert d['w'][f'futurepos_h{h}'][b] == (1.0 if want_valid else 0.0), \
                f'行 {i} 第 {h} 路有效性判错（独立算出 {want_valid}）'
            if not want_valid:
                continue
            got = d['future'][b, h] > 0
            assert np.array_equal(got, _expected_occupancy(data, int(i), off)), (
                f'行 {i} 第 {h} 路取值与手算不符')
            assert got.sum() > 0, '这一路至少得有一个对手子，否则测不出方向'


def test_opponent_is_relative_to_row_i_not_the_future_row(tmp_path):
    """🔴 「对手」按**行 i 的 to_play** 定，不是按未来行的 to_play。

    真实数据里 +8/+32 都是偶数偏移 ⇒ 两口径碰巧一致，所以**本例故意把
    ``to_play[i+8]`` 翻掉**来把两种口径拉开：不翻的话这条断言测不出任何东西。
    """
    data = _toy(LENS)
    data['to_play'][P8] = -int(data['to_play'][0])       # 制造口径分歧
    ds = _ds(data, _materialized(tmp_path, data))
    i = 0
    _, _, _, d = ds.sample_batch_numpy(np.array([i]), augment=False, labels=True)
    fb = data['boards'][i + P8].astype(np.int8)
    by_row_i = (fb * int(data['to_play'][i]) < 0).reshape(-1)
    by_future_row = (fb * int(data['to_play'][i + P8]) < 0).reshape(-1)
    assert not np.array_equal(by_row_i, by_future_row), \
        '本例两口径应不同（否则这条断言测不出任何东西）'
    assert np.array_equal(d['future'][0, 0] > 0, by_row_i), '必须按行 i 的对手'


# --------------------------------------------------------------------------- #
# 3. 三种「不可用」情形（最容易静默出错的地方）
# --------------------------------------------------------------------------- #
def _assert_not_silent_empty(d, b, h, why):
    """🔴 无效格**不许**是全 0（全 0 = 「对手一颗子都没占」这个合法标签）。"""
    v = d['future'][b, h]
    assert np.all(v == FUTUREPOS_SENTINEL), f'{why}：应整块为 {FUTUREPOS_SENTINEL}'
    assert not np.any(v == 0.0), f'{why}：哨兵不能是全 0 —— 那是合法标签'


def test_case1_out_of_range_gets_sentinel_not_empty_board(tmp_path):
    """情形① 越界（接近局尾）。"""
    data = _toy(LENS)
    n = len(data['boards'])
    ds = _ds(data, _materialized(tmp_path, data))
    assert _validity(data, BOTH_DEAD) == (False, False)
    _, _, _, d = ds.sample_batch_numpy(np.array([BOTH_DEAD]),
                                       augment=False, labels=True)
    for h in (0, 1):
        _assert_not_silent_empty(d, 0, h, f'越界行第 {h} 路')
        assert d['w'][f'futurepos_h{h}'][0] == 0.0
    assert d['w']['futurepos'][0] == 0.0
    # 只 +8 越界（+32 早已越界）：行 n-3 ⇒ n+5
    i = n - 3
    assert _validity(data, i) == (False, False)
    _, _, _, d2 = ds.sample_batch_numpy(np.array([i]), augment=False, labels=True)
    _assert_not_silent_empty(d2, 0, 0, f'行 {i} 第 0 路')


def test_case2_cross_game_gets_sentinel_and_no_clamping(tmp_path):
    """🔴 情形② 跨局：`game_ids[j] != game_ids[i]`。

    断言的重点是「**不许 clamp 到别局的行**」：`j` 在界内、盘面完全合法，
    只是不属于这一手。
    """
    data = _toy((20, 20, 20))          # 局界在 20/40：+8 与 +32 都够得着下一局
    ds = _ds(data, _materialized(tmp_path, data))
    i = 15                             # i+8=23 在第 2 局、i+32=47 在第 3 局（都界内）
    assert 0 <= i + P32 < len(data['boards']), '前提：本例要靠界内跨局而非越界'
    assert data['game_ids'][i] == 0, '前提：i 属第 1 局'
    assert data['game_ids'][i + P8] == 1 and data['game_ids'][i + P32] == 2, '前提：确为跨局'
    assert _validity(data, i) == (False, False)
    _, _, _, d = ds.sample_batch_numpy(np.array([i]), augment=False, labels=True)
    for h in (0, 1):
        _assert_not_silent_empty(d, 0, h, f'跨局行第 {h} 路')
        assert d['w'][f'futurepos_h{h}'][0] == 0.0
    assert d['w']['futurepos'][0] == 0.0
    # 反证：那盘面本身是完全合法的（能算出一个非空的占用图），
    # 所以「没给」只可能来自 valid 判定，而不是「数据里没有」。
    assert _expected_occupancy(data, i, P8).sum() > 0


def test_case3_gather_valid_false_is_the_single_gate(tmp_path):
    """🔴 情形③ `gather_neighbors` 的 `valid[offset]` 为 False。

    做法：monkeypatch 一层，**把 gather 算出来的 valid 强制清掉一部分**，
    证明 `future` 的哨兵与权重**只由 `valid` 决定**——即使
    ``boards[offset]`` 在那些行上装着完全合法的盘面（所以「判据是 valid」
    这件事无法用「盘面恰好是空的」蒙过去）。
    """
    data = _toy(LENS)
    ds = _ds(data, _materialized(tmp_path, data))
    idxs = np.array([0, 1, 2, 3])
    _, _, _, base = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    assert base['w']['futurepos'].tolist() == [1.0] * 4, '前提：默认四路全有效'
    for b in range(4):
        for h in (0, 1):
            assert base['future'][b, h].sum() > 0, '前提：盘面非空'

    from src.data import dataset as dsmod
    orig = dsmod.gather_neighbors

    def _fake(boards, indices, offsets=(), **kw):
        g = orig(boards, indices, offsets, **kw)
        for off in list(g.valid):
            g.valid[off] = g.valid[off].copy()
            g.valid[off][:2] = False       # 强行让第 0、1 行两路都不可用
        return g

    dsmod.gather_neighbors = _fake
    try:
        _, _, _, d = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    finally:
        dsmod.gather_neighbors = orig
    # 前置：那些行上 gather 拿到的盘面**完全合法**（不是空盘），所以唯一的
    # 判据只能是 `valid` —— 哨兵若没生效，这里就会看到「对手占满整盘」。
    for b in (0, 1):
        _assert_not_silent_empty(d, b, h=0, why=f'valid=False 的第 {b} 行第 0 路')
        _assert_not_silent_empty(d, b, h=1, why=f'valid=False 的第 {b} 行第 1 路')
        assert d['w'][f'futurepos_h{h}'][b] == 0.0
        assert d['w']['futurepos'][b] == 0.0
    # 未被清掉的行不受影响
    for b in (2, 3):
        assert d['w']['futurepos'][b] == 1.0


# --------------------------------------------------------------------------- #
# 4. 只活一路 ⇒ 整块权重必须为 0（loss 不除 Σw，见 katago_v7_loss）
# --------------------------------------------------------------------------- #
def test_one_surviving_horizon_gives_zero_block_weight(tmp_path):
    """🔴 单路存活：**`w['futurepos'] == 0` 但 `w['futurepos_h*']` 有一个 == 1**。

    为什么：`katago_v7_loss.py:362-368` 把 (b,2,bs²) 压成**一个**逐样本标量、
    只乘**一个**权重；`_weighted_mean` 又不除 Σw。给 w=1 会让另一路的 -1 哨兵
    被当成真值拟合（头会学出一个恒 tanh⁻¹(-1) 的假平面）。
    """
    data = _toy(LENS)
    ds = _ds(data, _materialized(tmp_path, data))
    i = H0_ONLY                     # 48 在第 0 局（活）；72 在第 1 局（跨局，死）
    assert _validity(data, i) == (True, False), '前提：本例恰有一路活'
    _, _, _, d = ds.sample_batch_numpy(np.array([i]), augment=False, labels=True)
    assert d['w']['futurepos_h0'][0] == 1.0, 'h0 应有效'
    assert d['w']['futurepos_h1'][0] == 0.0, 'h1 应无效'
    _assert_not_silent_empty(d, 0, 1, '死掉的那一路')
    assert d['future'][0, 0].sum() > 0, '活的那一路必须仍有真值（不只是权重）'
    # 🔴 整块权重必须是 0，尽管有一路活着
    assert d['w']['futurepos'][0] == 0.0, (
        '只活一路时 w["futurepos"] 必须为 0：loss 的作用单位是整块 2×bs²')


def test_both_horizons_dead_gives_zero_everywhere(tmp_path):
    data = _toy(LENS)
    ds = _ds(data, _materialized(tmp_path, data))
    assert _validity(data, BOTH_DEAD) == (False, False)
    _, _, _, d = ds.sample_batch_numpy(np.array([BOTH_DEAD]),
                                       augment=False, labels=True)
    assert d['w']['futurepos_h0'][0] == 0.0
    assert d['w']['futurepos_h1'][0] == 0.0
    assert d['w']['futurepos'][0] == 0.0
    _assert_not_silent_empty(d, 0, 0, '全死行第 0 路')
    _assert_not_silent_empty(d, 0, 1, '全死行第 1 路')


# --------------------------------------------------------------------------- #
# 5. 对称增广
# --------------------------------------------------------------------------- #
def test_augmented_future_equals_permute_soft_and_keeps_sentinel(tmp_path):
    """🔴 `future` 的增广必须**恰好**等于 `permute_soft` 的效果，
    且哨兵 -1 增广后仍是 -1（不会被当成下标、不会被 clamp、不会变成 0）。"""
    data = _toy(LENS)
    ds = _ds(data, _materialized(tmp_path, data))
    idxs = np.array([0, 5, H0_ONLY, BOTH_DEAD, 69])
    tforms = np.array([0, 3, 5, 7, 1])

    _, _, _, plain = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    got = _permute_future(plain['future'].copy(), tforms, BS)

    # 独立实现：把 (B,2,bs²) 拆成两路，逐路按 SYMMETRIES 逆置换手算
    rr, cc = np.divmod(np.arange(N_SQ), BS)
    from src.game.go_rules import SYMMETRIES
    invs = []
    for t in range(8):
        tr, tc = SYMMETRIES[t](rr, cc, BS)
        inv = np.empty(N_SQ, dtype=np.int64)
        inv[tr * BS + tc] = np.arange(N_SQ)
        invs.append(inv)
    for b, t in enumerate(tforms):
        for h in (0, 1):
            assert np.array_equal(got[b, h], plain['future'][b, h][invs[t]]), (
                f'行 {b}（tform={t}）第 {h} 路：增广结果与逆置换手算不符')
    # 哨兵必须原样留下
    assert (got == FUTUREPOS_SENTINEL).sum() > 0, '本例应至少有一处哨兵'
    assert np.array_equal(got == FUTUREPOS_SENTINEL,
                          plain['future'] == FUTUREPOS_SENTINEL)
    # 有效格的取值域仍然只有 {0,1}
    for h in (0, 1):
        keep = plain['w'][f'futurepos_h{h}'].astype(bool)
        assert set(np.unique(got[keep, h]).tolist()) <= {0.0, 1.0}


def test_augment_true_path_applies_the_same_tform_to_both_horizons(tmp_path):
    """`augment=True` 的热路径也必须增广 `future`，且两路共用同一个 tform。"""
    data = _toy(LENS)
    mat = _materialized(tmp_path, data)
    ds = _ds(data, mat)
    idxs = np.array([0, 1, 2, 3])
    rng1 = np.random.default_rng(99)
    _, _, _, aug = ds.sample_batch_numpy(idxs, rng=rng1, augment=True, labels=True)
    _, _, _, plain = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    # 恒等 tform ⇒ 增广后的 future 必须与未增广的逐位相同
    ident = _permute_future(plain['future'].copy(), np.zeros(4, dtype=int), BS)
    assert np.array_equal(ident, plain['future'])
    # 同种子重跑必须逐位一致（含 future 与 w 的新键）
    ds2 = _ds(data, mat)
    _, _, _, aug2 = ds2.sample_batch_numpy(
        idxs, rng=np.random.default_rng(99), augment=True, labels=True)
    assert aug['future'].tobytes() == aug2['future'].tobytes(), '同种子应逐位一致'
    assert aug2['w']['futurepos'].tolist() == [1.0] * 4
    assert 'futurepos_h0' in aug2['w'] and 'futurepos_h1' in aug2['w']


def test_permute_future_reuses_permute_soft_contract():
    """`_permute_future` 走的是 `permute_soft` 的哑格技巧 ⇒ 两者必须一致。"""
    rng = np.random.default_rng(3)
    f = rng.integers(0, 2, (4, 2, N_SQ)).astype(np.float32)
    tforms = np.array([1, 4, 6, 2])
    got = _permute_future(f.copy(), tforms, BS)
    for h in (0, 1):
        padded = np.zeros((4, N_SQ + 1), dtype=np.float32)
        padded[:, :N_SQ] = f[:, h]
        want = permute_soft(padded, tforms, BS)[:, :N_SQ]
        assert np.array_equal(got[:, h], want), f'h={h} 与 permute_soft 不一致'


# --------------------------------------------------------------------------- #
# 6. 向后兼容
# --------------------------------------------------------------------------- #
def test_disabled_is_byte_identical_to_pre_a6_payload(tmp_path):
    """未启用 ⇒ `future` 全 0、`w['futurepos']` 恒 0、**且 `w` 键集合不变**。"""
    data = _toy(LENS)
    ds = SupervisedDataset(data, n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(6), augment=False, labels=True)
    assert set(d['w']) == {'policy', 'policy_opp', 'ownership', 'score',
                           'scoring', 'seki', 'futurepos'}, sorted(d['w'])
    assert not d['future'].any()
    assert d['future'].shape == (6, 2, N_SQ)
    assert d['w']['futurepos'].sum() == 0.0
    for k in ('futurepos_h0', 'futurepos_h1'):
        assert k not in d['w'], f'未启用时不该出现 {k}'
    assert ds.futurepos_status() == {'enabled': False}


def test_labels_false_never_touches_the_gather(tmp_path):
    """🔴 `labels=False`（热路径）不得触发任何 gather / 物化。"""
    data = _toy(LENS)
    mat = _materialized(tmp_path, data)
    ds = SupervisedDataset(data, n_channels=12)
    ds.attach_futurepos(mat['boards'])          # live 模式，但 boards 还没解析
    assert ds.futurepos_status()['resolved'] is False

    boom = {'n': 0}
    from src.data import dataset as dsmod
    orig = dsmod.gather_neighbors

    def _boom(*a, **k):
        boom['n'] += 1
        raise AssertionError('labels=False 不该 gather')

    dsmod.gather_neighbors = _boom
    try:
        out = ds.sample_batch_numpy(np.arange(6), augment=False, labels=False)
    finally:
        dsmod.gather_neighbors = orig
    assert len(out) == 3, 'labels=False 必须是三元组'
    assert boom['n'] == 0
    assert ds.futurepos_status()['resolved'] is False, '也不该触发惰性解析'


def test_enable_and_disable_toggle_cleanly(tmp_path):
    data = _toy(LENS)
    mat = _materialized(tmp_path, data)
    ds = SupervisedDataset(data, n_channels=12)
    idxs = np.arange(6)
    _, _, _, off = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    ds.attach_futurepos(mat['boards'])
    _, _, _, on = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    assert on['future'].any(), '启用后应有真值'
    # 关掉（attach 一个关闭态）⇒ 回到占位
    ds._fp = None
    _, _, _, back = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    assert back['future'].tobytes() == off['future'].tobytes()
    assert set(back['w']) == set(off['w'])


# --------------------------------------------------------------------------- #
# 7. mmap 来源与惰性解析
# --------------------------------------------------------------------------- #
def test_live_gather_reads_the_npy_mmap_not_the_in_memory_column(tmp_path):
    """🔴 boards 来源取 `.npy` 的 mmap，且**不回退**到 `self.boards`。

    做法：把 `self.boards` 换成一份**不同的**数据，label 必须仍按 `.npy` 那份算
    —— 这同时证明「真的走了 source」和「没有静默用 self.boards」。
    """
    data = _toy(LENS)
    mat = _materialized(tmp_path, data)
    ds = _ds(data, mat['boards'])
    idxs = np.arange(4)
    _, _, _, d = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    for b, i in enumerate(idxs):
        assert np.array_equal(d['future'][b, 0] > 0,
                              _expected_occupancy(data, int(i), P8))
    # 把内存列换成全 -1（= 「对手占满全部点」的反面）
    ds.boards = np.full_like(ds.boards, -1)
    _, _, _, d2 = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    assert d2['future'].tobytes() == d['future'].tobytes(), \
        'source 的 mmap 没被用上（回退到了 self.boards）'


def test_npz_source_is_rejected_with_a_pointing_error(tmp_path):
    """`.npz` 直接当 boards 来源必须报错并指出正确做法。"""
    data = _toy(LENS)
    p = str(tmp_path / 'ds.npz')
    np.savez_compressed(p, **{k: data[k] for k in data})
    ds = SupervisedDataset(data, n_channels=12)
    ds.attach_futurepos(p)
    with pytest.raises(TypeError, match='materialize_dataset|mmap'):
        ds.sample_batch_numpy(np.arange(3), augment=False, labels=True)


def test_lazy_resolution_happens_once_and_warm_is_idempotent(tmp_path):
    data = _toy(LENS)
    mat = _materialized(tmp_path, data)
    ds = _ds(data, mat['boards'])
    assert ds.futurepos_status()['resolved'] is False
    ds.sample_batch_numpy(np.arange(3), augment=False, labels=True)
    assert ds.futurepos_status()['resolved'] is True
    first = ds._fp['boards']
    ds.warm_futurepos()
    assert ds._fp['boards'] is first, '解析结果必须复用'
    st = ds.futurepos_status()
    assert st['enabled'] and st['mode'] == 'live' and st['offsets'] == (8, 32)


def test_materialize_path_writes_npy_once_from_the_npz(tmp_path):
    """给了 `dataset_npz` + `materialized_dir` ⇒ 惰性 materialize 后 mmap。"""
    data = _toy(LENS)
    npz = str(tmp_path / 'ds.npz')
    np.savez_compressed(npz, **data)
    mdir = tmp_path / 'mat'
    ds = SupervisedDataset(data, n_channels=12)
    ds.attach_futurepos(dataset_npz=npz, materialized_dir=str(mdir))
    assert not mdir.exists(), 'attach 阶段不该写盘（惰性）'
    idxs = np.array([0, 5])
    _, _, _, d = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    assert mdir.exists() and (mdir / 'boards.npy').exists()
    assert (mdir / 'game_ids.npy').exists(), '跨局守卫要的那一列必须落盘'
    for b, i in enumerate(idxs):
        assert np.array_equal(d['future'][b, 0] > 0,
                              _expected_occupancy(data, int(i), P8))
    # 第二次不再写盘（内容不变 = 没重算）
    before = sorted((p.name, p.stat().st_size) for p in mdir.iterdir())
    ds.warm_futurepos()
    after = sorted((p.name, p.stat().st_size) for p in mdir.iterdir())
    assert before == after


def test_boards_source_shape_mismatch_is_rejected(tmp_path):
    """行数/盘面尺寸不符必须报错（偏移的语义变了，不能猜）。"""
    data = _toy(LENS)
    bad = _write_npy(str(tmp_path / 'boards.npy'),
                     np.zeros((len(data['boards']) - 1, BS, BS), dtype=np.int8))
    ds = SupervisedDataset(data, n_channels=12)
    ds.attach_futurepos(bad)
    with pytest.raises(ValueError, match='形状|不符'):
        ds.sample_batch_numpy(np.arange(2), augment=False, labels=True)


# --------------------------------------------------------------------------- #
# 8. mode='index'：从预存索引读（评估 / 自战）
# --------------------------------------------------------------------------- #
def test_index_mode_reads_the_precomputed_table(tmp_path):
    data = _toy(LENS)
    n = len(data['boards'])
    table = np.zeros((n, 2, N_SQ), dtype=bool)
    valid = np.ones((n, 2), dtype=bool)
    for i in (0, 5, 65):
        table[i, 0] = _expected_occupancy(data, i, P8)
        table[i, 1] = _expected_occupancy(data, i, P32)
    valid[200] = False                          # 手工标一行全死
    table[200] = True                           # 故意留真值：valid 才是裁判
    ds = SupervisedDataset(data, n_channels=12)
    ds.attach_futurepos(mode='index', table=table, table_valid=valid)
    idxs = np.array([0, 5, 65, 200])
    _, _, _, d = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    for b in (0, 1, 2):
        assert d['w']['futurepos'][b] == 1.0
    assert d['w']['futurepos'][3] == 0.0
    _assert_not_silent_empty(d, 3, 0, 'index 模式里 valid=False 的行')
    _assert_not_silent_empty(d, 3, 1, 'index 模式里 valid=False 的行')


def test_index_mode_does_not_gather(tmp_path):
    data = _toy(LENS)
    n = len(data['boards'])
    ds = SupervisedDataset(data, n_channels=12)
    ds.attach_futurepos(mode='index',
                        table=np.zeros((n, 2, N_SQ), dtype=bool),
                        table_valid=np.ones((n, 2), dtype=bool))
    from src.data import dataset as dsmod
    orig = dsmod.gather_neighbors

    def _boom(*a, **k):
        raise AssertionError('mode=index 不该 gather')

    dsmod.gather_neighbors = _boom
    try:
        ds.sample_batch_numpy(np.arange(3), augment=False, labels=True)
    finally:
        dsmod.gather_neighbors = orig
    assert ds.futurepos_status()['mode'] == 'index'


def test_index_mode_augments_too(tmp_path):
    data = _toy(LENS)
    n = len(data['boards'])
    table = np.zeros((n, 2, N_SQ), dtype=bool)
    table[:10, 0] = True                      # 前 10 行的 h0 是全 1
    ds = SupervisedDataset(data, n_channels=12)
    ds.attach_futurepos(mode='index', table=table)
    idxs = np.arange(4)
    _, _, _, d = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    assert np.all(d['future'][:, 0] == 1.0), 'h0 前提：全 1'
    assert np.all(d['w']['futurepos_h0'] == 1.0)
    got = _permute_future(d['future'].copy(), np.array([3, 3, 3, 3]), BS)
    assert np.all(got[:, 0] == 1.0), '全 1 的那路增广后仍是全 1'
    # h1 全 0 且**有效**（表里就是 0）⇒ 不许被误当成哨兵
    assert np.all(d['future'][:, 1] == 0.0)
    assert np.all(d['w']['futurepos_h1'] == 1.0)
    assert not np.any(got == FUTUREPOS_SENTINEL), \
        'index 模式的有效 0 必须原样保留（0 是合法标签，不是哨兵）'


# --------------------------------------------------------------------------- #
# 9. 参数校验
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('kw,exc', [
    (dict(mode='nope'), ValueError),
    (dict(mode='live', offsets=(8,)), ValueError),
    (dict(mode='live', offsets=(0, 8)), ValueError),
    (dict(mode='index'), ValueError),
])
def test_attach_futurepos_rejects_bad_configs(kw, exc):
    ds = SupervisedDataset(_toy(), n_channels=12)
    with pytest.raises(exc):
        ds.attach_futurepos(None, **kw)


def test_index_mode_validates_table_shapes():
    data = _toy(LENS)
    n = len(data['boards'])
    ds = SupervisedDataset(data, n_channels=12)
    with pytest.raises(ValueError, match='table'):
        ds.attach_futurepos(mode='index', table=np.zeros((n, 2, N_SQ + 1), bool))
    with pytest.raises(ValueError, match='table_valid'):
        ds.attach_futurepos(mode='index', table=np.zeros((n, 2, N_SQ), bool),
                            table_valid=np.ones((n,), bool))


def test_futurepos_target_requires_enable():
    ds = SupervisedDataset(_toy(), n_channels=12)
    with pytest.raises(RuntimeError, match='attach_futurepos'):
        ds._futurepos_target(np.arange(2))


def test_permute_future_rejects_wrong_shape():
    with pytest.raises(ValueError, match='_permute_future'):
        _permute_future(np.zeros((3, 2, N_SQ + 1), np.float32), np.zeros(3), BS)


# --------------------------------------------------------------------------- #
# 10. 无 game_ids（跨局守卫不可用）⇒ 一律不给标签，绝不猜
# --------------------------------------------------------------------------- #
def test_without_game_ids_everything_is_sentinel(tmp_path):
    """与 `next_move` 同一口径：无 `game_ids` ⇒ 无法验证同局 ⇒ **一律不给**。

    不能退化成「相邻行大概就是同局」——那正是要防的串局。
    """
    data = _toy(LENS)
    data.pop('game_ids')
    mat = _materialized(tmp_path, data, keys=('boards', 'to_play', 'ko'))
    ds = _ds(data, mat)
    _, _, _, d = ds.sample_batch_numpy(np.arange(6), augment=False, labels=True)
    assert d['w']['futurepos'].sum() == 0.0
    assert d['w']['futurepos_h0'].sum() == 0.0
    assert d['w']['futurepos_h1'].sum() == 0.0
    for b in range(6):
        _assert_not_silent_empty(d, b, 0, '无 game_ids 的行')
        _assert_not_silent_empty(d, b, 1, '无 game_ids 的行')
    # 连 boards 来源都不该被解析（省掉整次 gather 与可能的物化）
    assert ds.futurepos_status()['resolved'] is False
