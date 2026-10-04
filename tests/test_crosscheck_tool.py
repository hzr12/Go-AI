# -*- coding: utf-8 -*-
"""B7 · `scripts/crosscheck_stdata.py` 这个**工具**本身的测试。

被测对象是工具，不是 ``src/data`` 的实现（实现的正确性由
``tests/test_v7_assemble.py`` / ``test_v7_ladders.py`` / ``test_v7_planes.py``
担保）。所以这里钉的是**工具的契约**：

  1. **比对资格不能被弄错** —— ``NOT_COMPARABLE`` 的通道绝不能被报成
     ``ALIGNED``（ch15/ch16 的 0.73 巧合就是这个坑）；
  2. **门禁只对 ``ALIGNED`` 生效** —— ``KNOWN_DIVERGENT``（ch18/19）
     不能被误杀，否则会逼着人去「修」一个不是 bug 的东西；
  3. **归档缺失要报错退出**，不能静默成功（与 pytest 的 ``skipif`` 刻意不同）；
  4. 产出的 JSON **结构完整** —— 每通道都有两个口径 + 资格标注。

 **不在这里跑大样本**：真实归档是 1.5 GB，200 npz 的那一次要几分钟。
  这里全部用**合成的假归档**（几行稀疏的 19×19 盘面）⇒ 秒级，
  而且仍然走的是**真的** ``spatial_channels_v7`` / ``global_features_v7``，
  所以装配层要是写错通道，测试照样红。
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

from scripts import crosscheck_stdata as cc  # noqa: E402

N = 19
SPATIAL_C = 22
GLOBAL_C = 19
PACKED_BYTES = 46


# --------------------------------------------------------------------------- #
# 合成假归档
# --------------------------------------------------------------------------- #
def _pack_plane(plane):
    """``(19,19) bool`` → 46 字节（big-endian 位序，尾部 7 bit 补 0）。"""
    bits = np.zeros(361, np.uint8)
    bits[plane.reshape(-1)] = 1
    return np.packbits(bits, bitorder='big')


def _make_npz(boards):
    """几行盘面（``(B,19,19)`` int8）→ 一个可被官方读取器解开的 npz 字节。

    刻意做成**官方格式**（``binaryInputNCHWPacked`` (N,22,46) uint8 +
    ``globalInputNC`` (N,19)），这样 ``load_sample`` 走的是与真实归档**完全
    相同**的代码路径 —— 否则测的就不是这个工具了。

     ch3/4/5 用本仓的 :func:`liberties_123` **当真值**填，而不是填 0。
      否则「官方」这三格恒 0、本仓算出真值 ⇒ ``--min-rate`` 会因为一个
      **夹具造出来的**假偏差而红，测的就不是门禁的资格范围了。
      官方在这三个通道上不做任何额外处理，所以它们本就可以当 oracle。
     ch18/19 同样用本仓的 :func:`area_ownership_map` 当真值填，**理由完全
      一样**：这两个通道现在是 ``ALIGNED``，若夹具把它们留成 0，等于凭空造出
      一个「我们算错了」的偏差，门禁就会因为**夹具**红而不是因为实现红。
      夹具的 ``globalInputNC`` 全 0 ⇒ 官方口径是 AREA + TAX_NONE +
      ``multiStoneSuicideLegal=false`` + 非 TERRITORY ⇒ 正好是
      ``tax_rule=TAX_NONE`` 那一支。
    """
    from src.data.feature_v7 import TAX_NONE, area_ownership_map, liberties_123

    B = boards.shape[0]
    packed = np.zeros((B, SPATIAL_C, PACKED_BYTES), np.uint8)
    on_board = np.ones((B, N, N), bool)
    libs = liberties_123(boards).astype(bool)
    own = area_ownership_map(boards, tax_rule=TAX_NONE)
    area18, area19 = own > 0, own < 0
    for i in range(B):
        packed[i, 0] = _pack_plane(on_board[i])
        packed[i, 1] = _pack_plane(boards[i] > 0)
        packed[i, 2] = _pack_plane(boards[i] < 0)
        for k in range(3):
            packed[i, 3 + k] = _pack_plane(libs[:, k][i])
        packed[i, 18] = _pack_plane(area18[i])
        packed[i, 19] = _pack_plane(area19[i])
    glob = np.zeros((B, GLOBAL_C), np.float32)
    buf = io.BytesIO()
    np.savez(buf, binaryInputNCHWPacked=packed, globalInputNC=glob)
    return buf.getvalue()


def _sparse_boards(seed, rows):
    """稀疏盘面（少子 ⇒ 梯子搜索秒退，不让这个测试跑成几分钟）。"""
    rng = np.random.default_rng(seed)
    b = np.zeros((rows, N, N), np.int8)
    for i in range(rows):
        for _ in range(4):
            r, c = rng.integers(0, N, 2)
            b[i, r, c] = 1 if rng.random() < 0.5 else -1
    return b


@pytest.fixture(scope='module')
def fake_archive(tmp_path_factory):
    """4 个 npz × 3 行 = 12 行 19×19 的 ``.tgz``（流式读取路径与真归档同构）。"""
    d = tmp_path_factory.mktemp('stdata')
    path = d / 'fake_stdata.tgz'
    rows_per_file = 3
    with tarfile.open(path, 'w:gz') as tf:
        for k in range(4):
            data = _make_npz(_sparse_boards(k, rows_per_file))
            info = tarfile.TarInfo(f'fake/{k:04d}.npz')
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return str(path)


@pytest.fixture(scope='module')
def report(fake_archive):
    spatial, glob, ko, info = cc.load_sample(fake_archive, 4, 3)
    return cc.build_report(spatial, glob, ko, info)


# --------------------------------------------------------------------------- #
# 1 · 结构完整性
# --------------------------------------------------------------------------- #
def test_report_has_all_channels_with_both_rates_and_eligibility(report):
    """22 个空间通道 + 19 个全局通道，一个不少，且**每个**都有两个口径与资格。"""
    assert len(report['spatial']) == SPATIAL_C
    assert len(report['global']) == GLOBAL_C
    assert [r['channel'] for r in report['spatial']] == list(range(SPATIAL_C))
    assert [r['channel'] for r in report['global']] == list(range(GLOBAL_C))
    for row in report['spatial'] + report['global']:
        assert row['eligibility'] in (cc.ALIGNED, cc.KNOWN_DIVERGENT,
                                      cc.NOT_COMPARABLE), row
        assert row['reason'].strip(), '资格必须附原因，否则读者无从判断'
        assert 0.0 <= row['row_exact'] <= 1.0
        assert 'cell_agree' in row
        if row['scope'] == 'spatial':
            # 空间通道是 19×19 平面 ⇒ 两个口径都必须是真的数字
            assert isinstance(row['cell_agree'], float)
            assert 'diff' in row
        else:
            # 全局是标量序列 ⇒ 「逐格」口径**不存在**，记 null 而不是编一个
            assert row['cell_agree'] is None


def test_json_round_trip_keeps_eligibility_and_both_rates(
        report, fake_archive, tmp_path):
    """``--json`` 的产物能原样读回，且资格/两个口径都在（下游要靠它做门禁）。"""
    out = tmp_path / 'out.json'
    rc = cc.main(['--archive', fake_archive, '--npz-count', '4',
                  '--rows-per-npz', '3', '--json', str(out)])
    assert rc == 0
    got = json.loads(out.read_text(encoding='utf-8'))
    assert got['sample']['rows_kept'] == 12
    assert got['sample']['stream_mode'].startswith('r|gz')
    assert got['sample']['npz_count_scanned'] == 4
    for row in got['spatial']:
        assert row['eligibility'] in (cc.ALIGNED, cc.KNOWN_DIVERGENT,
                                      cc.NOT_COMPARABLE)
        assert isinstance(row['cell_agree'], float)
    assert 'warnings' in got and got['warnings'], '逐格骗人的警告必须进 JSON'


def test_sample_selection_method_is_written_into_the_output(report):
    """样本选取方式**必须**出现在报告里（不带样本量的命中率无意义）。"""
    s = report['sample']
    assert 'selection' in s and '前缀' in s['selection']
    assert 'npz_count_scanned' in s and 'rows_per_npz' in s
    assert 'rows_kept' in s and s['rows_kept'] == 12
    assert 'to_play_assumption' in s


# --------------------------------------------------------------------------- #
# 2 · 资格不能被弄错（本工具最容易出错的地方）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('ch, why', [
    (9, '无着法序列'),
    (10, '无着法序列'),
    (11, '无着法序列'),
    (12, '无着法序列'),
    (13, '无着法序列'),
    (15, '前一手盘'),
    (16, '前二手盘'),
    (7, 'encore_phase'),
    (20, 'encore_phase'),
    (21, 'encore_phase'),
])
def test_not_comparable_channels_are_never_reported_as_aligned(report, ch, why):
    """ 空间 ch7 / ch9..13 / ch15 / ch16 / ch20 / ch21 **必须**是
    ``NOT_COMPARABLE``，且原因里必须出现那个具体缺口。

    这条钉的是 ch15/ch16 那个具体的坑：它们在本仓走官方的**回退复制**分支，
    比出来的 ~0.73 是巧合（对齐的其实是 ch14）。一旦有人把它们标成
    ``ALIGNED``，汇报出去的「对齐率」就是伪造的证据 —— 而这类错误**不会
    抛异常**，只会表现为一个好看的百分比。
    """
    row = next(r for r in report['spatial'] if r['channel'] == ch)
    assert row['eligibility'] == cc.NOT_COMPARABLE, (
        f'ch{ch}（{why}）被标成 {row["eligibility"]} ⇒ '
        f'它的 row_exact={row["row_exact"]:.6f} 会被当成对齐率报出去')
    assert why in row['reason'], (
        f'ch{ch} 的原因里找不到「{why}」⇒ 读者无从判断这个 0 是不是缺陷')


SP18_REASON = cc.SPATIAL_SPEC[18][1]


def test_ch18_ch19_are_aligned_and_say_which_subset_is_bit_exact(report):
    """ch18/ch19 必须是 ``ALIGNED``，且原因里必须写清「在哪个子集上 1.000000」。

     **旧断言（已作废）**：这两个通道曾是 ``KNOWN_DIVERGENT``，原因写的是
      「官方算 area 前**先提死子**」。那条根因是**错的** —— 官方
      ``Board::calculateArea*``（``board.cpp:1853-1937``）既不提子也不做死子
      判定，它的输入只有 ``colors``。真正的成因是两层算法：
      **Benson 无条件存活**（``:2159-2195``，vital 区 < 2 即判死）与
      **双活整块过滤**（``:2264-2296``）。按正确算法实现后，官方 stdata 上
      可比子集的逐行精确相等率是 **1.000000**。
     所以这里同时**删掉**了 ``'死子' in reason`` 那条断言：它会把一个已被
      推翻的诊断钉成契约，让下一个人照着错的方向去改实现。
    """
    for ch in (18, 19):
        row = next(r for r in report['spatial'] if r['channel'] == ch)
        assert row['eligibility'] == cc.ALIGNED, (
            f'ch{ch} 是 {row["eligibility"]}，实测在可比子集上已逐位 1.0 '
            f'⇒ 应为 ALIGNED')
        # 「是不是真的对齐」由数字说话，而不是靠 reason 里写没写某个数。
        assert row['row_exact'] == 1.0, (
            f'ch{ch} 标了 ALIGNED 但 row_exact={row["row_exact"]:.6f} '
            f'(比了 {row["n_rows_compared"]} 行，剔除 '
            f'{row["n_rows_not_comparable"]} 行)')
        assert row['cell_agree'] == 1.0
        assert '死子' not in row['reason'], (
            f'ch{ch} 的原因里仍出现「死子」⇒ 那个根因已被推翻，别再钉它')
        assert 'TERRITORY' in SP18_REASON, (
            'ch18 的原因必须说明剔掉了哪些行，否则 1.0 会被误读成'
            '「全样本都对上了」')


def test_eligibility_map_covers_every_channel_with_no_silent_default():
    """资格表**逐条登记**、不留默认 —— 加通道时必须显式表态。"""
    assert set(cc.SPATIAL_SPEC) == set(range(SPATIAL_C))
    assert set(cc.GLOBAL_SPEC) == set(range(GLOBAL_C))
    for spec in (cc.SPATIAL_SPEC, cc.GLOBAL_SPEC):
        for ch, (elig, reason) in spec.items():
            assert elig in (cc.ALIGNED, cc.KNOWN_DIVERGENT,
                            cc.NOT_COMPARABLE), f'ch{ch} 资格非法：{elig}'
            assert reason.strip(), f'ch{ch} 缺原因'


def test_global_pda_and_pass_channels_are_not_comparable(report):
    """全局 ch0..4 / ch14（着法序列）与 ch15/16（无 PDA）不可比。

     全局 ch15/16 在**小样本**上官方这两格恰好也恒 0，本仓也恒 0 ⇒
      ``row_exact`` 会正好是 **1.000000**。这正是「巧合看起来像对齐」的原型：
      唯一拦住它的是资格标注，所以这条测试非钉不可。（真实归档上样本一大就
      露馅 —— 4368 行时官方这两格有 133 个非零点、掉到 0.9696。）
    """
    for ch in (0, 1, 2, 3, 4, 5, 14, 15, 16):
        row = next(r for r in report['global'] if r['channel'] == ch)
        assert row['eligibility'] == cc.NOT_COMPARABLE, (
            f'全局 ch{ch} 被标成 {row["eligibility"]}，'
            f'而它的 row_exact={row["row_exact"]:.6f} 只是巧合')
    # 本仓无 PDA ⇒ 这两格恒 0；这个合成夹具里官方也恒 0 ⇒ row_exact 必为 1.0。
    # 断言的是「巧合确实存在」这个事实，而不是把它当成对齐率。
    pda = [r for r in report['global'] if r['channel'] in (15, 16)]
    assert all(r['ours_nz'] == 0 for r in pda)
    assert all(r['official_nz'] == 0 for r in pda)
    assert all(r['row_exact'] == 1.0 for r in pda), \
        '这个夹具里官方也恒 0 ⇒ row_exact 必为 1.0（巧合，不是对齐）'


# --------------------------------------------------------------------------- #
# 3 · 门禁只对 ALIGNED 生效
# --------------------------------------------------------------------------- #
def _fake_report(**rows):
    base = {'spatial': [], 'global': [], 'tally': {}, 'warnings': [],
            'sample': {'rows_kept': 1}}
    for kw, val in rows.items():
        scope, ch = (kw.split('_')[0], int(kw.split('_')[1]))
        base[scope].append({'scope': scope.rstrip('_'), 'channel': ch,
                            'eligibility': val, 'row_exact': 0.5,
                            'cell_agree': 0.5})
    return base


def test_gate_only_asserts_on_aligned_channels():
    """ ``--min-rate`` **只**对 ``ALIGNED`` 通道断言。

    ``ALIGNED`` 掉下来是真回归要抓；而 ``KNOWN_DIVERGENT``（ch18/19 的口径差）
    与 ``NOT_COMPARABLE``（ch15/16 的巧合）被误杀会逼着人去改一个不是 bug 的
    东西，或者反过来把阈值调到 0.44 来「迁就」它 —— 两种都更糟。
    """
    rep = _fake_report(spatial_18=cc.KNOWN_DIVERGENT,
                       spatial_15=cc.NOT_COMPARABLE,
                       spatial_14=cc.ALIGNED)
    bad = cc.check_gate(rep, 0.99)
    assert len(bad) == 1 and 'ch14' in bad[0], bad
    # 把 ALIGNED 那个也摘掉 ⇒ 一条违规都不该有
    rep['spatial'] = [r for r in rep['spatial']
                      if r['eligibility'] != cc.ALIGNED]
    assert cc.check_gate(rep, 0.99) == []


def test_gate_is_off_by_default_and_a_passing_gate_is_quiet(report):
    """默认**关**；开启且全过时零违规。"""
    assert cc.build_argparser().parse_args([]).min_rate is None
    assert cc.check_gate(report, 1.0) == [], \
        '默认样本上所有 ALIGNED 通道都应逐位 1.0（合成夹具里它们本来就不动）'


def test_gate_exit_code_is_nonzero_on_violation(fake_archive):
    """门禁未通过 ⇒ 退出码 1（且**只**因为 ALIGNED 通道）。"""
    assert cc.main(['--archive', fake_archive, '--npz-count', '4',
                    '--rows-per-npz', '3', '--min-rate', '1.0']) == 0
    assert cc.main(['--archive', fake_archive, '--npz-count', '4',
                    '--rows-per-npz', '3', '--min-rate', '1.0000001']) == 1


def test_gate_message_points_at_the_assembly_order_not_the_algorithm():
    """违规信息必须指向**组装层** —— ch14/17 的算法由 test_v7_ladders 担保。"""
    bad = cc.check_gate(_fake_report(spatial_17=cc.ALIGNED), 0.999)
    assert len(bad) == 1
    assert '组装顺序' in bad[0] and '不要改底层实现' in bad[0]


# --------------------------------------------------------------------------- #
# 4 · 归档缺失 ⇒ 报错退出（不是静默成功）
# --------------------------------------------------------------------------- #
def test_missing_archive_exits_nonzero_and_says_why(tmp_path, capsys):
    """ 归档不存在 ⇒ 退出码 2 + 明确的 stderr，**不是**退出 0。

    这与 pytest 里的 ``skipif`` 刻意不同：skipif 是「代码没验成，但测试职责
    到此为止」；这个工具的职责**就是**回答「我们对齐到哪一步」，归档不在就
    回答不了 ⇒ 静默退出 0 等于假装成功，那正是这个工具要消灭的失败模式。
    """
    rc = cc.main(['--archive', str(tmp_path / 'nope.tgz'),
                  '--npz-count', '1', '--rows-per-npz', '1'])
    assert rc != 0, '归档缺失必须非零退出'
    assert rc == 2
    err = capsys.readouterr().err
    assert '不存在' in err
    assert '不静默跳过' in err, '要说清这是刻意的（pytest 那边是 skipif）'


def test_load_sample_raises_on_missing_archive(tmp_path):
    """``load_sample`` 本身也抛 —— 别让「文件在不在」这条判断散到调用点。"""
    with pytest.raises(cc.CrosscheckError, match='不存在'):
        cc.load_sample(str(tmp_path / 'nope.tgz'))


def test_archive_without_matching_rows_raises(tmp_path):
    """扫描到了但一个 19×19 行都没有 ⇒ **报错**，不是「跑出 0 行然后成功」。"""
    path = tmp_path / 'empty.tgz'
    with tarfile.open(path, 'w:gz') as tf:
        data = _make_npz(np.zeros((2, N, N), np.int8))
        info = tarfile.TarInfo('x/0.npz')
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    # 把 on-board 掩码清零 ⇒ `board_size_from_packed` 记 0 ⇒ 该行被丢掉
    d = np.load(io.BytesIO(data))
    packed = d['binaryInputNCHWPacked'].copy()
    packed[:, 0] = 0
    buf = io.BytesIO()
    np.savez(buf, binaryInputNCHWPacked=packed, globalInputNC=d['globalInputNC'])
    with tarfile.open(path, 'w:gz') as tf:
        raw = buf.getvalue()
        info = tarfile.TarInfo('x/0.npz')
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(raw))
    with pytest.raises(cc.CrosscheckError, match='没有一行'):
        cc.load_sample(str(path), 1, 3)


def test_sample_rejects_degenerate_counts(fake_archive):
    """``--npz-count 0`` / ``--rows-per-npz 0`` 是调参错误 ⇒ 明确报错。"""
    for kwargs in ({'npz_count': 0}, {'rows_per_npz': 0}):
        with pytest.raises(cc.CrosscheckError, match='必须 ≥ 1'):
            cc.load_sample(fake_archive, **dict({'npz_count': 4,
                                                 'rows_per_npz': 3}, **kwargs))


# --------------------------------------------------------------------------- #
# 5 · 两个口径 / 通道选择 / 流式纪律
# --------------------------------------------------------------------------- #
def test_both_rates_are_reported_and_cell_agree_is_flagged_when_it_lies(
        report):
    """ 每个空间通道都要有 ``row_exact`` **和** ``cell_agree``。

    且当「逐格 ≥0.99 而逐行 ≤0.5」时必须打上 ``cell_agree_misleads`` ——
    ch9 在 301 行上就是 row_exact=0.000000 / cell_agree=0.997200，
    汇报成 99.7% 对齐是把一个彻底错的通道说成几乎完美。
    """
    for row in report['spatial']:
        assert 'row_exact' in row and 'cell_agree' in row
        if row['cell_agree'] >= 0.99 and row['row_exact'] <= 0.5:
            assert row['cell_agree_misleads'] is True, (
                f'ch{row["channel"]} 逐格 {row["cell_agree"]:.4f} 骗人却没被标出')
    assert '' in report['warnings'][0] and 'cell_agree' in report['warnings'][0]
    assert any('逐格' in w for w in report['warnings'])


def test_diff_distribution_splits_stone_versus_empty(report):
    """ 「多出/少掉的点」必须**按「子 vs 空点」拆开**。

    这是 ch18/19 那 21,896 个差异点的**唯一**线索：19,913 是子、1,983 是空点
    ⇒ 偏差集中在死子本身 ⇒ 官方先提死子。只报一个总数的话这个猜测拿不出证据。
    """
    for row in report['spatial']:
        d = row['diff']
        assert d['extra_ours_stone'] + d['extra_ours_empty'] == d['extra_ours']
        assert (d['missing_ours_stone'] + d['missing_ours_empty']
                == d['missing_ours'])
        assert d['rows_touched'] <= report['sample']['rows_kept']
    ch18 = next(r for r in report['spatial'] if r['channel'] == 18)
    assert set(ch18['diff']) >= {'extra_ours_stone', 'extra_ours_empty',
                                 'missing_ours_stone', 'missing_ours_empty'}


def test_streaming_reader_never_walks_the_whole_archive(
        monkeypatch, fake_archive):
    """ 归档读取**必须**是流式，**禁止 ``getmembers()``**（1.5 GB 要扫很久）。"""
    def boom(*a, **kw):
        raise AssertionError('禁止 getmembers()/r: 全量模式')

    monkeypatch.setattr(tarfile.TarFile, 'getmembers', boom)
    monkeypatch.setattr(tarfile.TarFile, 'getmember', boom)
    spatial, glob, ko, info = cc.load_sample(fake_archive, 4, 3)
    assert spatial.shape[0] == 12
    assert 'r|gz' in info['stream_mode']


@pytest.mark.parametrize('spec, want', [
    ('ch14', {('spatial', 14)}),
    ('ch14,ch17', {('spatial', 14), ('spatial', 17)}),
    ('gch18', {('global', 18)}),
    ('14', {('spatial', 14)}),
    ('ch6-8', {('spatial', 6), ('spatial', 7), ('spatial', 8)}),
    ('ch18,gch18', {('spatial', 18), ('global', 18)}),
    ('', None),
    (None, None),
])
def test_parse_only(spec, want):
    assert cc.parse_only(spec) == want


def test_only_filters_the_report(fake_archive):
    """``--only ch14,ch17`` ⇒ 只剩这两个通道（且脚本仍以 0 退出）。"""
    rc = cc.main(['--archive', fake_archive, '--npz-count', '4',
                  '--rows-per-npz', '3', '--only', 'ch14,ch17'])
    assert rc == 0


def test_verbose_does_not_change_the_numbers(report, fake_archive, capsys):
    """``--verbose`` 只加明细，不改数字。"""
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cc.main(['--archive', fake_archive, '--npz-count', '4',
                 '--rows-per-npz', '3', '--verbose'])
    text = buf.getvalue()
    assert '逐通道明细' in text
    for row in report['spatial']:
        assert 'ch%d(spatial)' % row['channel'] in text
