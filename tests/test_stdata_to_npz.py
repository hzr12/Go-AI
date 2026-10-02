# -*- coding: utf-8 -*-
"""`scripts/stdata_to_npz.py` 的约定测试。

这个转换器里有**五件错了之后不报错、或者只在换批次时才炸**的事：

1. **19×19 过滤** —— 9/11/13 路与非方阵混在同一个成员里，漏过滤就是静默错位；
2. 🔴 **列布局按成员分派** —— 实测归档里 64 列与 80 列**混着**（见
   `stdata_to_npz.resolve_member_network` 的 docstring 的实测表）；
   「列数不同的归档经映射后得到同一组全局目标」是最容易错的地方；
3. **top-16 稀疏 policy 的打包 / 解包往返** —— 稠密化在 362 维上做，
   打包错了不会报错，只会悄悄改变分布；
4. **跨局守卫** —— stdata 没有着法序列，行与行无关；`game_ids` 必须让
   `gather_neighbors` 恒判不可用；
5. **ch18/ch19 与不可比通道按原样带入** —— 谁在转换期"顺手修一下"或
   "填 0"，就是在制造第三种偏差。

前四条用**合成数据**钉死（永远跑）；最后一条需要 28 MB 的 zzb 归档，
用 `skipif`。
"""

import io
import json
import os
import sys
import tarfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import stdata_to_npz as s2n   # noqa: E402
from src.data.feature_v7_gather import gather_neighbors   # noqa: E402
from src.data.katago_npz import (   # noqa: E402
    ACTION_SIZE,
    BOARD_STRIDE,
    COL_FINAL_SCORE,
    COL_GLOBAL_WEIGHT,
    COL_KOMI,
    COL_LEAD,
    COL_LOSS,
    COL_NORESULT,
    COL_SCORE_MEAN,
    COL_WIN,
    COL_W_FUTUREPOS,
    COL_W_LEAD,
    COL_W_OWNERSHIP,
    COL_W_POLICY_OPP,
    COL_W_POLICY_PLAYER,
    COL_W_SCORING,
    COL_W_VALUE,
    GLOBAL_TARGET_LAYOUT,
    PACKED_BYTES,
    SCORE_DISTR_BINS,
    SPATIAL_CHANNELS,
    unpack_binary_input,
)
from src.networks.katago_v7_loss import policy_dense_from_sparse  # noqa: E402

STDATA = os.path.join(ROOT, 'katago', 'stdata')
ARCHIVE_0825 = os.path.join(STDATA, '2026-08-25npzs.tgz')
ARCHIVE_B28 = os.path.join(STDATA, 'zzb28c512nfd4-s8264801024-d4596884264.tar')

needs_stdata = pytest.mark.skipif(
    not os.path.isfile(ARCHIVE_B28),
    reason='需要 katago/stdata 下的 .tar（不在仓库里）')

#: 80 列（b11c768）与 64 列（b40c768nbt / b28c512）两个布局族。
NET_80 = 'kata1-tf3-b11c768'
NET_64 = 'kata1-zhizi-b40c768nbt'
NET_64B = 'zzb28c512nfd4'


# --------------------------------------------------------------------------- #
# 合成 fixture
# --------------------------------------------------------------------------- #
def _pack_spatial(spatial):
    """``(N,22,19,19) bool`` → ``(N,22,46) uint8``（打包的逆运算）。"""
    n, c, h, w = spatial.shape
    assert (h, w) == (BOARD_STRIDE, BOARD_STRIDE)
    flat = spatial.reshape(n, c, BOARD_STRIDE * BOARD_STRIDE).astype(np.uint8)
    pad = np.zeros((n, c, PACKED_BYTES * 8), dtype=np.uint8)
    pad[:, :, :BOARD_STRIDE * BOARD_STRIDE] = flat
    return np.packbits(pad, axis=-1, bitorder='big')


def _semantic_values(n, offset=0.0):
    """每个语义列一个**可辨认**的值（越界/错位一眼能看出来）。

    ⚠ `outcome_hard` 那三列是**合法的概率行**（和为 1，三个值互不相同）
      —— 否则会被 `check_plausibility` 判成 SUSPECT，而一个 fixture 不该
      靠"检测器闭嘴"才通过。
    """
    return {
        'win': 0.6 + offset, 'loss': 0.3 + offset, 'noresult': 0.1,
        'score_mean': 11.0 + offset, 'final_score': 12.0 + offset,
        'lead': 13.0 + offset, 'global_weight': 1.0,
        'komi': 7.5 + offset, 'w_policy_player': 1.0, 'w_ownership': 1.0,
        'w_policy_opp': 0.5, 'w_lead': 0.25, 'w_futurepos': 0.75,
        'w_scoring': 0.125, 'w_value': 0.0,
    }


def _fake_npz(n=6, board=19, net=NET_80, policy_support=40, seed=0):
    """一个形状完整的合成 stdata 成员。

    ⚠ `globalTargetsNC` 里**每一列**都填了「列号当值」的哨兵
    （`col i == 1000 + i`），只有本仓 `COL_*` 指到的列放真值
    ⇒ 任何一处列号写错，读出来的就是 1000+某数，一眼可见。
    """
    rng = np.random.default_rng(seed)
    cols = GLOBAL_TARGET_LAYOUT[net]['cols']
    sp = np.zeros((n, SPATIAL_CHANNELS, BOARD_STRIDE, BOARD_STRIDE), dtype=bool)
    sp[:, 0, :board, :board] = True
    sp[:, 1, :3, :3] = True                        # pla 棋子
    sp[:, 2, 4, 4] = True                          # opp 棋子
    sp[:, 18, :5, :5] = True                       # ch18：area（批次特定，按原样带）
    sp[:, 19, 5:9, 5:9] = True                     # ch19

    g = np.tile(np.arange(cols, dtype=np.float32) + 1000.0, (n, 1))
    v = _semantic_values(n)
    g[:, COL_WIN] = v['win']
    g[:, COL_LOSS] = v['loss']
    g[:, COL_NORESULT] = v['noresult']
    g[:, COL_SCORE_MEAN] = v['score_mean']
    g[:, COL_FINAL_SCORE] = v['final_score']
    g[:, COL_LEAD] = v['lead']
    g[:, COL_GLOBAL_WEIGHT] = v['global_weight']
    g[:, COL_KOMI] = v['komi']
    g[:, COL_W_POLICY_PLAYER] = v['w_policy_player']
    g[:, COL_W_OWNERSHIP] = v['w_ownership']
    g[:, COL_W_POLICY_OPP] = v['w_policy_opp']
    g[:, COL_W_LEAD] = v['w_lead']
    g[:, COL_W_FUTUREPOS] = v['w_futurepos']
    g[:, COL_W_SCORING] = v['w_scoring']
    g[:, COL_W_VALUE] = v['w_value']

    # policy：`policy_support` 个动作有 visit，权重互不相同 ⇒ top-K 可验证
    pol = np.zeros((n, 2, ACTION_SIZE), dtype=np.int16)
    for i in range(n):
        w = rng.integers(1, 500, size=policy_support).astype(np.int16)
        pol[i, 0, :policy_support] = w
        pol[i, 1, :policy_support] = w[::-1]
    vt = np.zeros((n, 5, BOARD_STRIDE, BOARD_STRIDE), dtype=np.int8)
    vt[:, 0] = 1                                    # ownership
    vt[:, 1, 0, 0] = -1                             # seki 三值
    vt[:, 2, 1, 1] = 5                              # futurepos ch0
    vt[:, 3, 2, 2] = -5                             # futurepos ch1
    vt[:, 4] = 30                                   # scoring
    sb = np.zeros((n, SCORE_DISTR_BINS), dtype=np.int8)
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
    if GLOBAL_TARGET_LAYOUT[net]['has_qvalue']:
        d['qValueTargetsNCMove'] = np.zeros((n, 3, ACTION_SIZE), dtype=np.int16)
    return d


class _Npz(dict):
    """最小 NpzFile 替身：只需要 ``__getitem__`` 与 ``.files``。"""
    @property
    def files(self):
        return list(self)


def _write_archive(path, members, net=NET_80):
    """把若干合成成员写成**未压缩**的 tar（``r|*`` 流式读它）。"""
    with tarfile.open(path, 'w') as tf:
        for name, d in members:
            buf = io.BytesIO()
            np.savez(buf, **d)
            info = tarfile.TarInfo(name)
            info.size = buf.tell()
            tf.addfile(info, io.BytesIO(buf.getvalue()))
    return path


# --------------------------------------------------------------------------- #
# 1 · 19×19 过滤
# --------------------------------------------------------------------------- #
def test_keeps_only_19x19_and_drops_small_and_non_square(tmp_path):
    """9/13 路与**非方阵**都必须丢掉，且丢掉的数量要记账。

    ⚠ 非方阵（10×11）是 `board_size_from_packed` 记 0 的那种行：静默凑成
      邻近边长的话，18 路的行会被当成 19 路喂进模型。
    """
    n = 5
    d = _fake_npz(n=n, board=19)
    sp = np.zeros((n, SPATIAL_CHANNELS, BOARD_STRIDE, BOARD_STRIDE), dtype=bool)
    sp[0, 0, :19, :19] = True                      # 19 路 ✔
    sp[1, 0, :9, :9] = True                        # 9 路
    sp[2, 0, :13, :13] = True                      # 13 路
    sp[3, 0, :10, :11] = True                      # 非方阵
    sp[4, 0, :19, :19] = True                      # 19 路 ✔（带特征）
    sp[4, 18, :5, :5] = True                       # 唯一有内容的行
    d['binaryInputNCHWPacked'] = _pack_spatial(sp)

    arch = _write_archive(str(tmp_path / 'a.tar'), [('a/x.npz', d)])
    out = str(tmp_path / 'o.npz')
    meta = s2n.convert([(arch, NET_80)], out, log_every=0)

    assert meta['rows'] == 2, '只该留 2 行 19×19'
    z = np.load(out)
    try:
        assert z['spatial_packed'].shape == (2, SPATIAL_CHANNELS, PACKED_BYTES)
        # 留下的必须正好是第 0 行与第 4 行 —— 逐位比对，不靠"数量对了就算"
        want = np.concatenate([sp[0:1], sp[4:5]])
        assert np.array_equal(unpack_binary_input(z['spatial_packed']), want)
        assert z['source_id'].tolist() == [0, 0]
        assert z['game_ids'].tolist() == [0, 1]
        # 丢了多少必须出现在报告里，不能静默
        assert meta['count_pass'][0]['rows_raw'] == n
    finally:
        z.close()


def test_whole_archive_of_small_boards_is_reported_not_silently_empty(tmp_path):
    """一行 19×19 都没有时必须**报错**，不是写一个 0 行 npz。"""
    arch = _write_archive(str(tmp_path / 'small.tar'),
                          [('a/x.npz', _fake_npz(n=4, board=9))])
    with pytest.raises(s2n.ConversionError, match='一行 19×19 都没有'):
        s2n.convert([(arch, NET_80)], str(tmp_path / 'o.npz'), log_every=0)


# --------------------------------------------------------------------------- #
# 2 · 列布局：64 列与 80 列必须映射到同一组全局目标
# --------------------------------------------------------------------------- #
def test_same_semantics_land_on_the_same_columns_in_both_layouts():
    """🔴 最容易错的一处：两个列族的**同名列号一致**，映射结果必须逐位相同。

    `_fake_npz` 把「列号当值」的哨兵（1000+i）铺满每一列，只有 `COL_*`
    指到的列放真值 ⇒ 任何一处列号写错，这里立刻炸。
    """
    v = _semantic_values(1)
    g80 = _fake_npz(n=3, net=NET_80)['globalTargetsNC']
    g64 = _fake_npz(n=3, net=NET_64)['globalTargetsNC']
    a = s2n.extract_global_targets(g80, NET_80)
    b = s2n.extract_global_targets(g64, NET_64)

    assert a['_cols'] == 80 and b['_cols'] == 64, '两个归档的列数确实不同'
    assert a['_network'] != b['_network']
    assert set(a) == set(b) and a['_network'] != b['_network']
    for key in a:
        if key.startswith('_'):
            continue
        assert np.array_equal(a[key], b[key]), f'{key} 在两个列族上不一致'
    # 逐项点名，避免"集合相等但值都是哨兵"这种假绿
    # ⚠ 用 approx：`globalTargetsNC` 是 float32，0.6 存不精确，`==` 会假失败
    assert a['komi'].tolist() == pytest.approx([v['komi']] * 3)
    assert a['final_score'].tolist() == pytest.approx([v['final_score']] * 3)
    assert a['lead'].tolist() == pytest.approx([v['lead']] * 3)
    assert a['global_weight'].tolist() == pytest.approx([v['global_weight']] * 3)
    assert a['w_futurepos'].tolist() == pytest.approx([v['w_futurepos']] * 3)
    assert a['w_value'].tolist() == pytest.approx([v['w_value']] * 3)
    assert a['outcome_hard'].shape == (3, 3)
    assert a['outcome_hard'][:, 0].tolist() == pytest.approx([v['win']] * 3)
    assert a['outcome_hard'][:, 1].tolist() == pytest.approx([v['loss']] * 3)
    assert a['outcome_hard'][:, 2].tolist() == pytest.approx([v['noresult']] * 3)
    # 合法概率行 ⇒ 检测器必须闭嘴（一个 fixture 不该靠"检测器闭嘴"才通过）
    assert s2n.suspect_columns(
        s2n.check_plausibility(a)) == {}


def test_every_semantic_column_is_a_tuple_of_column_numbers():
    """🔴 `GLOBAL_TARGET_COLUMNS` 的每个值都必须是**元组**。

    写成 `'w_policy_opp': (COL_W_POLICY_OPP)`（少一个逗号）就是一个 int，
    `max(max(v) ...)` 会当场抛 `TypeError`，而 `g[:, 47]` 会静默取错一列。
    这条断言把「少一个逗号」这种 typo 变成测试失败。
    """
    for name, cols in s2n.GLOBAL_TARGET_COLUMNS.items():
        assert isinstance(cols, tuple) and cols, f'{name} 不是非空元组'
        assert all(isinstance(c, (int, np.integer)) for c in cols), name
        assert len(set(cols)) == len(cols), f'{name} 有重复列号'


def test_column_count_mismatch_raises_instead_of_guessing():
    with pytest.raises(s2n.ConversionError, match='登记为 80 列'):
        s2n.extract_global_targets(np.zeros((2, 64), np.float32), NET_80)
    with pytest.raises(s2n.ConversionError, match='未登记的网络'):
        s2n.resolve_network_key('kata1-tf9-b99c999')


def test_network_is_required_only_when_column_count_is_ambiguous():
    """80 列唯一 ⇒ 光看列数就能定；64 列有两个网络 ⇒ 必须靠批次名消歧。"""
    name80 = 'x/kata1-tf3-b11c768-s11001M-d1/0A.npz'
    assert s2n.resolve_member_network(name80, 80)[0] == NET_80
    assert s2n.resolve_member_network(name80, 80)[1] == 'cols'
    # 64 列：批次名消歧
    name64 = 'x/kata1-zhizi-b40c768nbt-s11472M-d1/0B.npz'
    assert s2n.resolve_member_network(name64, 64)[0] == NET_64
    assert s2n.resolve_member_network(name64, 64)[1] == 'cols+batch'
    name64b = 'x/zzb28c512nfd4-s8264801024-d4596884264/0C.npz'
    assert s2n.resolve_member_network(name64b, 64)[0] == NET_64B
    # 批次名定不下来 ⇒ 只能靠 --network；再定不下来就报错（**不猜**）
    with pytest.raises(s2n.ConversionError, match='多个候选网络'):
        s2n.resolve_member_network('x/opaque/0D.npz', 64)
    assert s2n.resolve_member_network('x/opaque/0D.npz', 64, NET_64B)[1] \
        == 'cols+default'
    with pytest.raises(s2n.ConversionError, match='列布局未登记'):
        s2n.resolve_member_network('x/opaque/0E.npz', 126_000)


def test_mixed_layouts_in_one_archive_are_dispatched_per_member(tmp_path):
    """🔴 实测两个 `.tgz` 都混着 64/80 列 ⇒ 逐成员分派，不按归档名分派。

    这条用合成归档复现那个实测事实：同一个 tar 里先 80 列成员、后 64 列成员，
    两种都要进同一个 npz，且 `meta` 里要分别记账。
    """
    a80 = ('m/kata1-tf3-b11c768-s1/aaa.npz', _fake_npz(n=3, net=NET_80, seed=1))
    a64 = ('m/kata1-zhizi-b40c768nbt-s1/bbb.npz',
           _fake_npz(n=4, net=NET_64, seed=2))
    arch = _write_archive(str(tmp_path / 'mixed.tar'), [a80, a64])
    out = str(tmp_path / 'o.npz')
    # ⚠ 故意给一个**只对得上其中一种布局**的 --network：逐成员分派不许被它带偏
    meta = s2n.convert([(arch, NET_80)], out, log_every=0)

    assert meta['rows'] == 7
    per = meta['archives'][0]['per_network']
    assert set(per) == {NET_80, NET_64}, per
    assert per[NET_80]['cols'] == 80 and per[NET_80]['rows'] == 3
    assert per[NET_64]['cols'] == 64 and per[NET_64]['rows'] == 4
    z = np.load(out)
    try:
        assert z['global'].shape == (7, 19)
    finally:
        z.close()


def test_column_count_different_archives_produce_identical_targets(tmp_path):
    """两个**列数不同**的归档合进一个 npz，各自映射出的全局目标逐位相同。

    这是任务书点名的验收项：64 列与 80 列的归档，喂给下游的必须是同一组
    19 维全局目标 + 同一批语义标签。
    """
    arch80 = _write_archive(str(tmp_path / 'a80.tar'),
                            [('m/kata1-tf3-b11c768-s1/a.npz',
                              _fake_npz(n=5, net=NET_80, seed=7))])
    arch64 = _write_archive(str(tmp_path / 'a64.tar'),
                            [('m/kata1-zhizi-b40c768nbt-s1/b.npz',
                              _fake_npz(n=5, net=NET_64, seed=7))])
    out = str(tmp_path / 'o.npz')
    meta = s2n.convert([(arch80, NET_80), (arch64, NET_64)], out, log_every=0)

    assert meta['rows'] == 10
    assert [a['global_target_cols'] for a in meta['archives']] == [[80], [64]]
    z = np.load(out)
    try:
        a = {k: z[k][:5] for k in ('komi', 'score', 'lead_hint', 'game_weight',
                                   'w_score', 'w_futurepos', 'outcome')}
        b = {k: z[k][5:] for k in a}
        for k in a:
            assert np.array_equal(a[k], b[k]), f'{k} 在两个列族上不一致'
        assert a['komi'].tolist() == [7.5] * 5
        assert a['score'].tolist() == [12.0] * 5
        assert a['w_score'].tolist() == [1.0] * 5
        assert a['outcome'].tolist() == [0] * 5       # win 0.6 最大 ⇒ argmax 0
    finally:
        z.close()


# --------------------------------------------------------------------------- #
# 3 · top-16 稀疏 policy：打包 / 解包往返
# --------------------------------------------------------------------------- #
def test_top16_policy_roundtrips_through_the_sparse_format(tmp_path):
    """存 top-16、用 `policy_dense_from_sparse` 还原 ⇒ **恰好等于那个 top-16 子分布**。

    ⚠ 往返一致的前提是「rank + 权重」两列都没错位。⚠ 注意断言的对象是
      **截断后的 16 个动作**重新归一化的结果（`renormalize=True` 的语义），
      **不是**原始 40 个动作的全分布 —— 后者按定义就还原不出来，而丢掉的
      那部分质量正是 `policy_*_resid` 记的账。
    """
    d = _fake_npz(n=4, net=NET_80, policy_support=40, seed=11)
    arch = _write_archive(str(tmp_path / 'a.tar'), [('m/x.npz', d)])
    out = str(tmp_path / 'o.npz')
    meta = s2n.convert([(arch, NET_80)], out, policy_topk=16, log_every=0)
    assert meta['policy_topk'] == 16

    z = np.load(out)
    try:
        assert z['policy_player_rank'].shape == (4, 16)
        assert z['policy_player_prob'].shape == (4, 16)
        assert z['policy_player_resid'].shape == (4,)
        for i in range(4):
            for who, pol_axis in (('player', 0), ('opp', 1)):
                src = d['policyTargetsNCMove'][i, pol_axis].astype(np.float64)
                order = np.argsort(-src, kind='stable')[:16]
                want = np.zeros(ACTION_SIZE, np.float64)
                want[order] = src[order]
                want /= want.sum()
                # ⚠ `policy_dense_from_sparse` 收 (B,K) —— 单行也要留 2 维
                got = policy_dense_from_sparse(
                    z[f'policy_{who}_rank'][i:i + 1],
                    z[f'policy_{who}_prob'][i:i + 1], ACTION_SIZE).numpy()[0]
                assert np.allclose(got, want, atol=1e-6), f'第 {i} 行 {who}'
                # rank 必须是「按 visit 降序」的前 16 个，且权重与之对应
                assert z[f'policy_{who}_rank'][i].tolist() == order.tolist()
                assert z[f'policy_{who}_prob'][i].tolist() == src[order].tolist()
                # 丢掉的尾部质量由 resid 记账（`topk_policy` docstring 的要求）
                resid = 1.0 - src[order].sum() / src.sum()
                assert 0.0 < resid < 1.0, '40 个动作截到 16 必须有残差'
                assert float(z[f'policy_{who}_resid'][i]) == pytest.approx(
                    resid, abs=1e-6)
                # 截断之外的动作在稠密化后必须是 0（不能凭空长出来）
                assert np.allclose(got[np.setdiff1d(np.arange(ACTION_SIZE),
                                                    order)], 0.0)
    finally:
        z.close()


def test_topk_truncation_resid_is_zero_when_support_fits_in_k(tmp_path):
    """动作数 ≤ K 时不该有截断：`resid == 0`（哨兵的意义就是能判「K 够不够」）。"""
    d = _fake_npz(n=3, net=NET_80, policy_support=8, seed=13)
    arch = _write_archive(str(tmp_path / 'a.tar'), [('m/x.npz', d)])
    out = str(tmp_path / 'o.npz')
    s2n.convert([(arch, NET_80)], out, policy_topk=16, log_every=0)
    z = np.load(out)
    try:
        assert np.allclose(z['policy_player_resid'], 0.0)
        assert np.allclose(z['policy_opp_resid'], 0.0)
    finally:
        z.close()


def test_policy_resid_never_returns_nan_for_an_all_zero_row():
    """全零 visit 行（占位样本）⇒ `resid = 0`，不是 nan。

    ⚠ nan 会在报告里伪装成「K 太小」，是最难查的那类假信号。
    """
    got = s2n.policy_resid(np.zeros((1, ACTION_SIZE)), np.zeros((1, 16)))
    assert got.tolist() == [0.0]
    assert np.isfinite(got).all()


# --------------------------------------------------------------------------- #
# 4 · 跨局守卫
# --------------------------------------------------------------------------- #
def test_emitted_game_ids_are_unique_so_every_neighbor_is_rejected(tmp_path):
    """stdata 没有着法序列 ⇒ `game_ids` 逐行唯一 ⇒ `gather_neighbors` 全不可用。

    ⚠ 越界也判不可用，但**跨局**这条是独立的一层：这里所有 `j` 都在界内，
      判不可用**只能**来自 `game_ids` 不相等。
    """
    d = _fake_npz(n=12, net=NET_80, seed=3)
    arch = _write_archive(str(tmp_path / 'a.tar'), [('m/x.npz', d)])
    out = str(tmp_path / 'o.npz')
    s2n.convert([(arch, NET_80)], out, log_every=0)

    z = np.load(out)
    try:
        gid = z['game_ids']
        assert gid.tolist() == list(range(12))
        assert len(np.unique(gid)) == 12, 'game_ids 必须逐行不同'
        boards = unpack_binary_input(z['spatial_packed'])[:, 1].astype(np.int8) \
            - unpack_binary_input(z['spatial_packed'])[:, 2].astype(np.int8)
        g = gather_neighbors(np.ascontiguousarray(boards),
                             np.arange(4), offsets=(-1, 1, 8), game_ids=gid)
        for off in (-1, 1, 8):
            assert not g.valid[off].any(), f'offset {off:+d} 不该有可用行'
            assert g.all_valid(off) is False
            assert not np.any(g[off]), '不可用行必须填 0'
    finally:
        z.close()


def test_cross_game_guard_rejects_the_boundary_and_keeps_the_interior():
    """守卫本身的行为（用手写的 `game_ids` 钉住，不依赖转换器）。

    行 0,1,2 属同一局，行 3 起是下一局 ⇒ ``+1`` 在行 2（跨局）与行 4（越界）
    上必须被判不可用，行 0/1/3 必须可用。**两种不可用要分开看**：守卫只管前者。
    """
    boards = (np.arange(5 * 19 * 19).reshape(5, 19, 19) % 3 - 1).astype(np.int8)
    gid = np.array([7, 7, 7, 9, 9], dtype=np.int32)
    g = gather_neighbors(np.ascontiguousarray(boards),
                         np.array([0, 1, 2, 3, 4]), offsets=(1,),
                         game_ids=gid)
    assert g.valid[1].tolist() == [True, True, False, True, False]
    # 跨局那一行拿到的盘面必须被清 0（不是"看起来合法的别人的盘面"）
    assert np.array_equal(g[1][2], np.zeros((19, 19), np.int8))
    assert np.array_equal(g[1][0], boards[1])
    assert np.array_equal(g[1][3], boards[4])


# --------------------------------------------------------------------------- #
# 5 · 通道按原样带入 + 输出结构
# --------------------------------------------------------------------------- #
def test_spatial_stays_bit_packed_and_known_divergent_channels_are_untouched(
        tmp_path):
    """🔴 ch18/ch19（死子口径差）必须**逐位原样**带入，不许在转换期"修"。

    同时钉住输入的存储格式：`(N,22,46) uint8`，不是解包后的
    `(N,22,19,19)`（那会让体积翻 8 倍）。
    """
    d = _fake_npz(n=4, net=NET_80, seed=5)
    arch = _write_archive(str(tmp_path / 'a.tar'), [('m/x.npz', d)])
    out = str(tmp_path / 'o.npz')
    s2n.convert([(arch, NET_80)], out, log_every=0)

    z = np.load(out)
    try:
        sp = z['spatial_packed']
        assert sp.dtype == np.uint8, '输入必须保持 bit-packed uint8'
        assert sp.shape == (4, SPATIAL_CHANNELS, PACKED_BYTES)
        src = unpack_binary_input(d['binaryInputNCHWPacked'])
        assert np.array_equal(unpack_binary_input(sp), src), '逐位原样'
        for ch in (18, 19):
            assert np.array_equal(unpack_binary_input(sp)[:, ch], src[:, ch])
        # 不可比通道（ch7/9-13/15/16/20/21）同样不许被填 0 或改动
        for ch in (7, 9, 10, 11, 12, 13, 15, 16, 20, 21):
            assert np.array_equal(unpack_binary_input(sp)[:, ch], src[:, ch])
        # ch7/9-13/15/16 在这个 fixture 里本来就是 0 ⇒ 断言"没被改动"要有意义
        assert not np.array_equal(src[:, 7], unpack_binary_input(sp)[:, 7]) \
            or not src[:, 7].any()
    finally:
        z.close()


def test_output_schema_is_complete_and_self_describing(tmp_path):
    """npz 必须自带 `meta_json` / `schema_version` / `target_model`。

    下游要能只读这一个文件就判断"它是不是我认识的 schema" ⇒ 不靠文件名约定。
    """
    d = _fake_npz(n=2, net=NET_80, seed=9)
    arch = _write_archive(str(tmp_path / 'a.tar'), [('m/x.npz', d)])
    out = str(tmp_path / 'o.npz')
    meta = s2n.convert([(arch, NET_80)], out, log_every=0)

    z = np.load(out)
    try:
        expect = {k for k, _, _ in s2n.output_spec()}
        assert expect <= set(z.files)
        assert int(z['schema_version']) == s2n.SCHEMA_VERSION
        assert str(z['target_model']) == s2n.TARGET_MODEL
        info = json.loads(str(z['meta_json']))
        assert info['schema_version'] == s2n.SCHEMA_VERSION
        assert info['target_model'] == s2n.TARGET_MODEL
        assert info['rows'] == 2
        assert info['policy_topk'] == s2n.POLICY_TOPK
        # 通道资格表逐项来自 crosscheck_stdata，不重抄
        assert info['known_divergent']['spatial'] == [18, 19]
        assert info['aligned_channels'] == [0, 1, 2, 3, 4, 5, 6, 8, 14, 17]
        assert set(info['not_comparable']['spatial']) == \
            {7, 9, 10, 11, 12, 13, 15, 16, 20, 21}
        assert '逐行' in info['game_ids_semantics']
        for key, shape in info['columns'].items():
            assert z[key].shape == tuple(shape), f'{key} 形状与 meta 不符'
        assert meta['file_bytes'] == os.path.getsize(out)
    finally:
        z.close()


def test_komi_out_of_physical_domain_is_flagged_not_silently_written(tmp_path):
    """🔴 `COL_KOMI=47` 实测是 selfKomi，值域可以到 ±94 ⇒ 必须**打标记**。

    这条锁死处理方式：**按原样写入 + 逐批标进 `suspect_columns`**，
    既不改值也不静默 —— 凭空发明一个贴目比留一个已知错位的值更难查。
    """
    d = _fake_npz(n=4, net=NET_80, seed=17)
    d['globalTargetsNC'][:, COL_KOMI] = [-94.5, 94.5, -41.5, 7.5]
    arch = _write_archive(str(tmp_path / 'a.tar'), [('m/x.npz', d)])
    out = str(tmp_path / 'o.npz')
    meta = s2n.convert([(arch, NET_80)], out, log_every=0)

    assert meta['archives'][0]['suspect_columns'] == ['komi']
    assert meta['suspect_columns']
    z = np.load(out)
    try:
        assert z['komi'].tolist() == [-94.5, 94.5, -41.5, 7.5], '不许改值'
        gaps = json.loads(str(z['meta_json']))['known_gaps']
        assert any('selfKomi' in g for g in gaps), 'komi 的口径问题要写进 meta'
    finally:
        z.close()


def test_soft_outcome_triples_pass_the_plausibility_check():
    """软 outcome（行和恒为 1、`max < 0.999`）**不得**被判成 SUSPECT。

    ⚠ 这条是「检测器不许假报警」的约束：实测三个归档上
      `globalTargetsNC[:,0:3]` 都是软分布，早先按「必须 one-hot」写的那版
      检测器对着真实数据一路狂响 —— 那样的检测器比没有更糟。
    """
    g = np.zeros((4, 64), np.float32)
    g[:, COL_WIN] = [0.82, 0.55, 1.0, 0.0]
    g[:, COL_LOSS] = [0.18, 0.45, 0.0, 0.0]
    g[:, COL_KOMI] = 7.5
    for c in (COL_GLOBAL_WEIGHT, COL_W_POLICY_PLAYER, COL_W_OWNERSHIP):
        g[:, c] = 1.0
    rep = s2n.check_plausibility(s2n.extract_global_targets(g, NET_64))
    assert rep['outcome_hard']['verdict'] == 'ok'
    assert rep['outcome_hard']['frac_soft'] == pytest.approx(0.75, abs=1e-6)
    assert s2n.suspect_columns(rep) == {}


def test_failed_conversion_leaves_no_partial_npz(tmp_path, monkeypatch):
    """写到一半失败时**不留半截文件**（约束 5：行数对不上的 npz 比没有更糟）。

    ⚠ 故障注入在 ``_write_member`` 上而不是主循环里：主循环失败时文件还没
      建（`close()` 才建），那种"失败"证明不了任何东西。这里让 zip 已经
      建好、写第 3 个成员时炸 ⇒ ``discard()`` 必须把它删掉。
    """
    good = ('m/kata1-tf3-b11c768-s1/a.npz', _fake_npz(n=3, seed=21))
    arch = _write_archive(str(tmp_path / 'a.tar'), [good])
    out = str(tmp_path / 'o.npz')

    real = s2n.NpzChunkedWriter._write_member
    seen = {'n': 0}

    def boom(self, key, arr):
        seen['n'] += 1
        if seen['n'] == 3:
            raise OSError('磁盘满了（注入的故障）')
        return real(self, key, arr)

    monkeypatch.setattr(s2n.NpzChunkedWriter, '_write_member', boom)
    with pytest.raises(OSError, match='注入的故障'):
        s2n.convert([(arch, NET_80)], out, log_every=0)
    assert seen['n'] == 3, '故障注入点没生效，测试本身失效了'
    assert not os.path.exists(out), '失败后不能留下半截 npz'


def test_writer_rejects_more_rows_than_the_header_declares(tmp_path):
    """缓冲的行数超过 ``.npy`` 头声明的行数 ⇒ 报错（第 0/1 遍对不上）。"""
    spec = s2n.output_spec()
    w = s2n.NpzChunkedWriter(str(tmp_path / 'o.npz'), spec, 2)
    chunk = {k: np.zeros((1,) + t, dtype=np.dtype(t_))
             for k, t_, t in spec}
    w.write(chunk)
    w.write(chunk)
    with pytest.raises(s2n.ConversionError, match='头里声明'):
        w.write(chunk)
    w.discard()


def test_reader_roundtrips_into_the_12_item_loss(tmp_path):
    """🔴 「产出结构正确」的可执行定义：读回来的标签能直接喂 `KataGoV7Loss`。

    没有这条，「键名对不对」就只能靠人读键名猜；而**拼错键名不报错**，
    只会静默少算一项 loss（训练照跑，曲线看着"正常"）。
    """
    import torch
    from src.networks.katago_v7_loss import KataGoV7Loss

    d = _fake_npz(n=6, net=NET_80, policy_support=40, seed=23)
    arch = _write_archive(str(tmp_path / 'a.tar'), [('m/x.npz', d)])
    out = str(tmp_path / 'o.npz')
    s2n.convert([(arch, NET_80)], out, log_every=0)

    lb = s2n.load_v7_npz(out)
    assert lb['spatial'].shape == (6, SPATIAL_CHANNELS, 19, 19)
    assert lb['spatial'].dtype == np.float32
    assert lb['global'].shape == (6, 19)
    assert lb['policy_player_sparse'][0].shape == (6, 16)
    assert set(lb['w']) == {'policy_opp', 'ownership', 'score', 'lead',
                            'futurepos', 'scoring'}

    # 与 `to_v7_labels` 直接产出的标签逐项相等（读回来不许有任何口径漂移）
    from src.data.katago_npz import read_katago_npz, to_v7_labels
    ref = to_v7_labels(read_katago_npz(_Npz(d), NET_80))
    for key in ('outcome', 'score', 'score_mean_hint', 'lead_hint', 'komi',
                'score_distr', 'ownership', 'seki', 'futurepos', 'scoring',
                'game_weight', 'global', 'spatial'):
        assert np.array_equal(lb[key], ref[key]), f'{key} 读回来变了'
    for who in ('player', 'opp'):
        assert np.array_equal(lb[f'policy_{who}_sparse'][0],
                              ref[f'policy_{who}_sparse'][0])

    # 端到端：模型输出（随机）→ 12 项 loss 全部有限
    g = torch.Generator().manual_seed(3)
    res = KataGoV7Loss()({
        'policy_logits': torch.randn(6, 2, ACTION_SIZE, generator=g),
        'outcome_logits': torch.randn(6, 3, generator=g),
        'score_mean': torch.randn(6, generator=g),
        'score_stdev': torch.rand(6, generator=g) * 10 + 1,
        'lead': torch.randn(6, generator=g),
        'ownership_pretanh': torch.randn(6, 1, 19, 19, generator=g),
        'scoring': torch.randn(6, 1, 19, 19, generator=g),
        'futurepos': torch.randn(6, 2, 19, 19, generator=g),
        'seki_logits': torch.randn(6, 4, 19, 19, generator=g),
        'scorebelief_logits': torch.randn(6, SCORE_DISTR_BINS, generator=g),
    }, lb)
    assert len(res['terms']) == 12
    assert torch.isfinite(res['loss'])


def test_reader_refuses_a_foreign_schema_version(tmp_path):
    """schema 对不上必须**拒绝读**，不许"字段差不多就凑合读"。"""
    d = _fake_npz(n=2, net=NET_80, seed=29)
    arch = _write_archive(str(tmp_path / 'a.tar'), [('m/x.npz', d)])
    out = str(tmp_path / 'o.npz')
    s2n.convert([(arch, NET_80)], out, log_every=0)

    import zipfile
    with zipfile.ZipFile(out) as zf:            # 只改 schema_version 那一个成员
        items = [(i, zf.read(i.filename)) for i in zf.infolist()]
    bad = str(tmp_path / 'bad.npz')
    with zipfile.ZipFile(bad, 'w', zipfile.ZIP_DEFLATED) as zf:
        for info, blob in items:
            if info.filename == 'schema_version.npy':
                buf = io.BytesIO()
                np.lib.format.write_array(
                    buf, np.array(s2n.SCHEMA_VERSION + 1, np.int32),
                    allow_pickle=False)
                blob = buf.getvalue()
            zf.writestr(info.filename, blob)
    with pytest.raises(s2n.ConversionError, match='schema_version'):
        s2n.load_v7_npz(bad)


# --------------------------------------------------------------------------- #
# 6 · 真实归档（需要 katago/stdata）
# --------------------------------------------------------------------------- #
@needs_stdata
def test_real_archive_converts_and_matches_the_official_packing(tmp_path):
    """真跑一个归档的前若干行：结构对、且逐位等于官方的 packed 行。"""
    out = str(tmp_path / 'real.npz')
    meta = s2n.convert([(ARCHIVE_B28, NET_64B)], out, limit=600, log_every=0)
    assert meta['rows'] == 600
    assert meta['archives'][0]['per_network'][NET_64B]['cols'] == 64

    # 独立复核：从归档里再读一遍前 600 行 19×19，比对 packed
    want, seen = [], 0
    for _, npz in s2n.iter_npz_members(ARCHIVE_B28):
        packed = npz['binaryInputNCHWPacked']
        keep = np.flatnonzero(
            (packed[:, 0] != 0).sum(1) > 0)     # 占位，真正判尺寸在下面
        from src.data.katago_npz import board_size_from_packed
        keep = np.flatnonzero(board_size_from_packed(packed) == 19)
        if keep.size:
            want.append(packed[keep])
            seen += keep.size
            npz.close()
            if seen >= 600:
                break
        npz.close()
    want = np.concatenate(want)[:600]

    z = np.load(out)
    try:
        assert np.array_equal(z['spatial_packed'], want), 'packed 必须逐位一致'
        assert np.array_equal(unpack_binary_input(z['spatial_packed']),
                              unpack_binary_input(want))
        # 21×19×19 的平面确实被原样带进来了（ch18/19 的口径差不在转换期"修"）
        assert z['ownership'].shape == (600, 1, 19, 19)
        assert z['futurepos'].shape == (600, 2, 19, 19)
        assert z['score_distr'].shape == (600, SCORE_DISTR_BINS)
        assert np.allclose(z['score_distr'].sum(1), 1.0, atol=1e-6)
    finally:
        z.close()


@needs_stdata
def test_real_b28_komi_is_flagged_as_suspect(tmp_path):
    """🔴 真实 zzb 批上 `komi` 必须被标成 SUSPECT（实测 col47 到 ±41.5）。

    这条把上面那条合成测试钉在**真实数据**上：检测器不是理论摆设。
    """
    out = str(tmp_path / 'real2.npz')
    meta = s2n.convert([(ARCHIVE_B28, NET_64B)], out, limit=400, log_every=0)
    assert 'komi' in meta['archives'][0]['suspect_columns']
    z = np.load(out)
    try:
        # selfKomi = globalInputNC[:,5] × 20 ⇒ 与 global 的 ch5 逐行相等
        assert np.allclose(z['komi'], z['global'][:, 5] * 20.0, atol=1e-3)
    finally:
        z.close()
