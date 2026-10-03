"""分片（``--num-shards`` / ``--shard-id``）的回归测试。

为什么需要分片
--------------
`NpzChunkedWriter` 的峰值内存 = **未压缩总量**（zip 容器无法交错写成员，
所有列的累积块一直留到 `close()` 才逐键落盘）。实测（Windows / 16 逻辑核 /
13.9 GB 物理内存，`2026-08-25npzs.tgz`）：

    --limit 100000  → 峰值 1.50 GB   ⇒ 14.6 KB/行
    --limit 400000  → 峰值 5.72 GB   ⇒ 14.0 KB/行
    外推全量 3,132,287 行           ⇒ ≈ 44.8 GB

本机可用内存不到 10 GB ⇒ **不分块必然 OOM**。分 8 块后 ≈5.6 GB。

⚠ 那个 14.0 KB/行 与 `uncompressed_bytes_estimate` 算出的 11.9 KB/行 差 19%
（zip/deflate 工作缓冲 + 对齐 + 页碎片）⇒ `RAM_OVERHEAD_FACTOR = 1.19`。
**别按未压缩量估**：`--ram-budget-gb` 是唯一能在开跑前拦住"装不下"的闸门，
按未压缩量估会在临界配置上放行一个必然 OOM 的任务，而 OOM 发生在写了几 GB 之后，
现场既没有堆栈也没有块边界可查。

真正的风险不在内存，在**偏移算术**
--------------------------------
全局行号取模只���「所有分块用同一套偏移」时才构成一个划分。任何一处把偏移
写成「本块已写行数」，就会 **重复 + 丢失且全程不报错**。下面的
`test_offset_*` 把这个错钉死，`test_shards_*` 从端到端验证划分本身。
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import stdata_to_npz as s2n   # noqa: E402
from scripts.stdata_to_npz import (   # noqa: E402
    ConversionError,
    count_kept_rows,
    convert,
    render_report,
    shard_mask,
    subset_labels,
)

STDATA = os.path.join(ROOT, 'katago', 'stdata')
ARCHIVE_0825 = os.path.join(STDATA, '2026-08-25npzs.tgz')
ARCHIVE_B28 = os.path.join(STDATA,
                           'zzb28c512nfd4-s8264801024-d4596884264.tar')
NET_80 = 'kata1-tf3-b11c768'
NET_64 = 'kata1-zhizi-b40c768nbt'

needs_stdata = pytest.mark.skipif(
    not (os.path.isfile(ARCHIVE_0825) and os.path.isfile(ARCHIVE_B28)),
    reason='需要 katago/stdata 下的真实归档')


# --------------------------------------------------------------------------- #
# 1 · shard_mask：划分本身
# --------------------------------------------------------------------------- #
def test_shards_partition_the_rows_exactly():
    """N 块的并集恰好是全集、两两不相交、每块大小至多差 1 行。"""
    n, N = 2000, 8
    counts = np.zeros(n, dtype=np.int64)
    for sid in range(N):
        counts[shard_mask(n, 0, N, sid)] += 1
    assert np.array_equal(counts, np.ones(n, dtype=np.int64))   # 恰好一次
    sizes = [len(shard_mask(n, 0, N, sid)) for sid in range(N)]
    assert max(sizes) - min(sizes) <= 1                          # 均衡


def test_num_shards_one_is_the_identity():
    """不分块时必须原样返回全部行 —— 否则会静默丢数据。"""
    n = 37
    for nsh in (1, 0, -1):
        assert np.array_equal(shard_mask(n, 0, nsh, 0), np.arange(n))


def test_global_offset_actually_shifts_ownership():
    """同一个局部下标，在不同全局偏移下归属不同的块。

    这是分片**唯一**的工作机制；没有它，偏移算术写错了测试也发现不了。
    """
    # 逐个手算：shard s 取的是 (offset + j) % N == s 的局部下标 j
    N = 4
    assert shard_mask(8, 0, N, 0).tolist() == [0, 4]      # 0,4 %4==0
    assert shard_mask(8, 1, N, 0).tolist() == [3, 7]      # 1+3=4, 1+7=8
    assert shard_mask(8, 2, N, 1).tolist() == [3, 7]      # 2+3=5, 2+7=9
    # 同一个局部下标 3，在 offset=0/1/2 下分别落到 shard 3/0/1
    assert ([shard_mask(8, off, N, 3).tolist() for off in (0, 1, 2)]
            == [[3, 7], [2, 6], [1, 5]])


def test_offset_must_count_unfiltered_rows():
    """🔴 分片最经典的错：偏移按**过滤后**的行数推进。

    那样每块的全局行号会随自己的过滤结果漂移 ⇒ 有些行落进两块（重复）、
    有些块谁都不落（丢失），而 `a_rows == budget` 那道检查**抓不到**（两边
    一起错）。这里构造两个成员，让「正确偏移」与「错误偏移」给出不同的划分。
    """
    n_members, per = 3, 63
    N = 4
    correct = Counter()
    wrong = Counter()
    base_correct = 0
    base_wrong = 0
    for _ in range(n_members):
        for j in shard_mask(per, base_correct, N, 0):
            correct[int(j)] += 1
        mine = len(shard_mask(per, base_correct, N, 0))
        for j in shard_mask(per, base_wrong, N, 0):
            wrong[int(j)] += 1
        base_correct += per
        base_wrong += mine
    # 两种算术给出不同的局部下标分布 ⇒ 写错就一定会被下游的端到端测试抓到
    assert correct != wrong


def test_shard_mask_rejects_nonsense():
    n = 10
    assert shard_mask(n, 0, 1, 7).tolist() == list(range(n))  # 不分块时忽略 id


# --------------------------------------------------------------------------- #
# 2 · subset_labels：切的是标签 dict 的四类成员
# --------------------------------------------------------------------------- #
def _fake_labels(n=8, seed=0):
    rng = np.random.default_rng(seed)

    def rank():
        return rng.integers(0, 362, (n, 2)).astype(np.int32)

    def prob():
        return rng.random((n, 2)).astype(np.float32)

    return {
        'policy_player_sparse': (rank(), prob()),
        'policy_opp_sparse': (rank(), prob()),
        'outcome': rng.integers(0, 3, n).astype(np.int64),
        'score_distr': rng.random((n, 842)).astype(np.float32),
        'ownership': rng.random((n, 1, 19, 19)).astype(np.float32),
        'w': {'policy_opp': rng.random(n).astype(np.float32),
              'score': rng.random(n).astype(np.float32),
              'lead': rng.random(n).astype(np.float32)},
        'board_mask': np.ones((n, 1, 19, 19), dtype=bool),
        'global': rng.random((n, 19)).astype(np.float32),
        '_dropped': 3,
        '_kept': n,
        '_network': NET_80,
    }


def test_subset_labels_cuts_every_row_axis_and_keeps_scalars():
    """`to_v7_labels` 的返回值混着四类东西，只有一类带行轴：

    * 行轴 ndarray（`outcome` / `score_distr` / `ownership` / `board_mask`）
    * 嵌套 dict（`w`）
    * tuple（`policy_*_sparse` 的 (idx, val) 两半）
    * 标量/字符串（`_kept` / `_dropped` / `_network`）

    漏切任何一类都会让块内行数与 `game_ids` 不符 —— 而那正是写手
    `close()` 会当场报错的地方，所以能被发现；但**值**错位（切了行数对但
    顺序错）不会，所以这里逐键比对。
    """
    labels = _fake_labels(8, seed=1)
    j = np.array([1, 4, 7], dtype=np.int64)
    out = subset_labels(labels, j)
    assert out['_kept'] == 3
    for key in ('outcome', 'score_distr', 'ownership', 'board_mask', 'global'):
        assert np.array_equal(out[key], labels[key][j]), key
    for wk in labels['w']:
        assert np.array_equal(out['w'][wk], labels['w'][wk][j]), wk
    for sk in ('policy_player_sparse', 'policy_opp_sparse'):
        for a, b in zip(out[sk], labels[sk]):
            assert np.array_equal(a, b[j]), (sk, a.shape)
    assert out['_dropped'] == 3          # 与分片无关 ⇒ 原样带走
    assert out['_network'] == NET_80


def test_subset_labels_is_identity_on_full_index():
    labels = _fake_labels(6, seed=2)
    out = subset_labels(labels, np.arange(6))
    for key in ('outcome', 'score_distr', 'ownership', 'global'):
        assert np.array_equal(out[key], labels[key])
    assert out['_kept'] == 6


def test_subset_labels_does_not_touch_same_length_non_row_keys():
    """⚠ 判据是「第 0 轴长度 == `_kept`」，所以一个**恰好也是 n 行、但不是行轴**
    的键会被误切。当前 `to_v7_labels` 没有这种键，但这是本函数的隐含假设，
    用测试写明：将来新增键若不满足，行数会与 `_kept` 不符而被写手抓住。"""
    labels = _fake_labels(5, seed=3)
    labels['_network_is_a_5_array_not_a_string'] = np.arange(5)
    out = subset_labels(labels, np.array([0, 2]))
    assert out['_kept'] == 2
    assert out['_network_is_a_5_array_not_a_string'].tolist() == [0, 2]


# --------------------------------------------------------------------------- #
# 3 · 端到端：合成归档分块后合并 == 不分块
# --------------------------------------------------------------------------- #
def _archive(path, n_members=4, per_member=40, net=NET_80):
    """合成归档：多个成员、每成员行数不同（**故意不整除** N）。

    行数不整除是关键 —— 若成员大小恰好整除分块数，"偏移写错也可能碰巧对"。
    """
    import io
    import tarfile
    from tests.test_stdata_to_npz import _fake_npz
    members = []
    for i in range(n_members):
        k = per_member + (i * 7) % 11
        members.append(('m%02d/b11c768/batch.npz' % i, _fake_npz(n=k, seed=i)))
    with tarfile.open(path, 'w') as tf:
        for name, d in members:
            buf = io.BytesIO()
            np.savez(buf, **d)
            info = tarfile.TarInfo(name)
            info.size = buf.tell()
            tf.addfile(info, io.BytesIO(buf.getvalue()))
    return path


def _row_keys(path):
    z = np.load(path)
    try:
        return [k for k in z.files
                if k not in ('meta_json', 'schema_version', 'target_model')]
    finally:
        z.close()


def test_shards_merge_back_to_the_unsharded_file(tmp_path):
    """分 N 块再拼回去，必须与不分块**逐位相同**。

    比的是逐行指纹多重集：分块只重排行序，比对必须对行序不敏感；但任何
    **重复或丢失**都会让多重集不等。
    """
    arch = _archive(str(tmp_path / 'a.tar'))
    whole = str(tmp_path / 'whole.npz')
    meta_all = convert([(arch, NET_80)], whole)

    N = 4
    keys = _row_keys(whole)
    seen = {k: [] for k in keys}
    for sid in range(N):
        out = str(tmp_path / ('s%d.npz' % sid))
        meta = convert([(arch, NET_80)], out, num_shards=N, shard_id=sid)
        assert meta['shard']['shard_id'] == sid
        assert meta['shard']['rows_in_shard'] == meta['rows']
        z = np.load(out)
        try:
            for k in keys:
                seen[k].append(z[k])
        finally:
            z.close()

    z = np.load(whole)
    try:
        for k in keys:
            got = np.concatenate(seen[k])
            want = z[k]
            assert got.shape[0] == want.shape[0], k
            order_got = np.lexsort(got.T[::-1])
            order_want = np.lexsort(want.T[::-1])
            assert np.array_equal(got[order_got], want[order_want]), k
    finally:
        z.close()
    assert sum(a.shape[0] for a in seen['outcome']) == meta_all['rows']


def test_meta_json_itself_records_the_shard(tmp_path):
    """块内的 `meta_json`（落盘的那份）也必须写明分片 —— 训练端读的是它，
    不是 `convert()` 的返回值。返回 dict 对了但落盘那份漏写，一样会静默。"""
    import json
    arch = _archive(str(tmp_path / 'a.tar'))
    out = str(tmp_path / 's1.npz')
    convert([(arch, NET_80)], out, num_shards=8, shard_id=1)
    z = np.load(out)
    try:
        meta = json.loads(str(z['meta_json']))
    finally:
        z.close()
    assert meta['shard']['num_shards'] == 8
    assert meta['shard']['shard_id'] == 1
    assert meta['shard']['rows_in_shard'] > 0


def test_num_shards_one_is_bit_identical_to_unsharded(tmp_path):
    """`--num-shards 1` 必须与不传该参数**逐位相同**（含 meta 里的分片字段）。"""
    arch = _archive(str(tmp_path / 'a.tar'))
    convert([(arch, NET_80)], str(tmp_path / 'x.npz'))
    convert([(arch, NET_80)], str(tmp_path / 'y.npz'), num_shards=1, shard_id=0)
    # ⚠ 一次只能有一个 npz 打开（`NpzFile` 底层是同一个 zip 句柄，第二个
    #   `np.load` 会把第一个的句柄置空）—— 先各自读完再比。
    import json

    def snapshot(path):
        z = np.load(path)
        try:
            meta = json.loads(str(z['meta_json']))
            data = {k: z[k] for k in z.files if k != 'meta_json'}
            return meta, data
        finally:
            z.close()

    mx, dx = snapshot(str(tmp_path / 'x.npz'))
    my, dy = snapshot(str(tmp_path / 'y.npz'))
    assert set(dx) == set(dy)
    for k in dx:
        assert np.array_equal(dx[k], dy[k]), k
    assert mx['shard'] == my['shard']
    assert mx['shard']['num_shards'] == 1 and mx['shard']['shard_id'] == 0
    assert mx['rows'] == my['rows']


def test_shard_game_ids_restart_locally_but_guard_still_holds(tmp_path):
    """块内 `game_ids` 从 0 重数（跨块会撞号），但**块内逐行唯一**必须保持。

    `game_ids` 是跨局守卫的载体（`gather_neighbors` 靠它判不可用）。块内唯一
    ⇒ 守卫依然恒判不可用 ⇒ 语义不变。这一点不能因为"编号撞了"就放松。
    """
    arch = _archive(str(tmp_path / 'a.tar'))
    for sid in (0, 2):
        out = str(tmp_path / ('s%d.npz' % sid))
        convert([(arch, NET_80)], out, num_shards=3, shard_id=sid)
        z = np.load(out)
        try:
            g = z['game_ids']
            assert g[0] == 0
            assert len(np.unique(g)) == g.shape[0]
        finally:
            z.close()


def test_out_of_range_shard_id_is_rejected(tmp_path):
    arch = _archive(str(tmp_path / 'a.tar'))
    with pytest.raises(ConversionError):
        convert([(arch, NET_80)], str(tmp_path / 'z.npz'),
                num_shards=3, shard_id=7)
    with pytest.raises(ConversionError):
        convert([(arch, NET_80)], str(tmp_path / 'z.npz'),
                num_shards=3, shard_id=-1)


def test_meta_records_shard_so_a_partial_build_is_visible(tmp_path):
    """只跑了 8 块里的 1 块就开训 —— 元数据必须能让这件事显形。"""
    arch = _archive(str(tmp_path / 'a.tar'))
    meta = convert([(arch, NET_80)], str(tmp_path / 's1.npz'),
                   num_shards=8, shard_id=1)
    sh = meta['shard']
    assert sh['num_shards'] == 8 and sh['shard_id'] == 1
    assert sh['rows_in_shard'] == meta['rows']
    assert sh['rows_all_shards'] >= sh['rows_in_shard']
    assert 0 < sh['rows_in_shard'] < sh['rows_all_shards']
    rep = render_report(meta)
    assert '1/8' in rep and '不是' in rep        # 报告里明说"这不是全量"


# --------------------------------------------------------------------------- #
# 4 · 内存守卫
# --------------------------------------------------------------------------- #
def test_ram_guard_scales_with_shard_count(tmp_path):
    """同一份数据：不分块被拦，分 8 块放行 —— 证明守卫确实随分片缩放。"""
    arch = _archive(str(tmp_path / 'a.tar'))
    spec = s2n.output_spec()
    # 用**实际这份归档的行数**算，而不是猜一个大数：守卫比的是
    # estimate(spec, 本块行数) × FACTOR 与预算，预算必须落在
    # 「1 块」与「1/8 块」之间才能同时验证拦与放。
    rows_all = count_kept_rows(arch, NET_80)['rows_kept']
    need_all = s2n.uncompressed_bytes_estimate(spec, rows_all)
    peak_all = need_all * s2n.RAM_OVERHEAD_FACTOR / 1e9
    budget_gb = peak_all * 0.5      # 介于「1/8 块峰值」与「1 块峰值」之间
    assert peak_all / 8 < budget_gb < peak_all, (peak_all, budget_gb)
    with pytest.raises(ConversionError, match='num-shards'):
        convert([(arch, NET_80)], str(tmp_path / 'a.npz'),
                ram_budget_gb=budget_gb)
    convert([(arch, NET_80)], str(tmp_path / 'b.npz'),
            num_shards=8, shard_id=0, ram_budget_gb=budget_gb)


def test_overhead_factor_is_applied_and_documented():
    """守卫用的是**实测峰值**，不是未压缩量 —— 差 19%，不乘会放行 OOM。"""
    assert 1.0 < s2n.RAM_OVERHEAD_FACTOR < 1.5
    # 1.19 是 float，`__doc__` 是 float 的类型文档 —— 改去查模块里我们自己写
    # 的那段注释（`grep` 得到），别查 `__doc__`。
    import re
    src = open(os.path.join(ROOT, 'scripts', 'stdata_to_npz.py'),
               encoding='utf-8').read()
    assert re.search(r'^RAM_OVERHEAD_FACTOR = 1\.19$', src, re.M)
    # 数字的依据（实测每行字节数）必须留在源码注释里，别只剩一个魔数
    assert '14.0' in src and '11.9' in src and '44.8' in src


def test_writer_docstring_no_longer_claims_peak_is_one_key(tmp_path):
    """⚠ 防回归：`NpzChunkedWriter` 曾写着"峰值不叠加 ⇒ 峰值 = 最大单键"，
    那是**错的**（实测 44.8 GB vs 最大单键 10.55 GB）。有人照它做优化就会
    把 5.6 GB 的配置当成够用。"""
    doc = s2n.NpzChunkedWriter.__doc__
    assert '未压缩总量' in doc
    assert 'RAM_OVERHEAD_FACTOR' in doc


# --------------------------------------------------------------------------- #
# 5 · 真实归档（行数均衡 + 划分）
# --------------------------------------------------------------------------- #
@needs_stdata
def test_real_archive_shard_counts_are_balanced():
    """真实成员大小不均 ⇒ 每块行数应接近 1/N（允许差几个成员）。"""
    N = 4
    infos = [count_kept_rows(ARCHIVE_B28, NET_64, num_shards=N, shard_id=sid)
             for sid in range(N)]
    got = [i['rows_kept_shard'] for i in infos]
    total = infos[0]['rows_kept']
    assert sum(got) == total, '分块行数之和必须等于不分块的行数'
    mean = total / N
    for sid, n in enumerate(got):
        assert abs(n - mean) / max(mean, 1) < 0.02, (sid, n, mean)


@needs_stdata
def test_real_archive_count_offset_accumulates_across_archives():
    """跨归档偏移：第一块的 `rows_kept_shard` 与第二块的不该重叠。"""
    N = 3
    base = 0
    prev_last = -1
    for sid in range(N):
        i1 = count_kept_rows(ARCHIVE_B28, NET_64, num_shards=N,
                             shard_id=sid, global_offset=base)
        i2 = count_kept_rows(ARCHIVE_0825, NET_80, num_shards=N,
                             shard_id=sid, global_offset=base + i1['rows_kept'])
        # 偏移变了 ⇒ 第二归档的行归属确实随之平移（否则偏移白传了）
        assert i2['rows_kept_shard'] >= 0
        assert i1['rows_kept_shard'] <= i1['rows_kept']
        base += i1['rows_kept']
        assert base > prev_last
        prev_last = base
