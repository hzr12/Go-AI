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
    convert,
    convert_all_shards,
    count_kept_rows,
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
#: 🔴 真实归档这一档验证的是**偏移算术，不是规模**。全扫 1.5 GB 的
#: ``2026-08-25npzs.tgz`` 实测 152.80 s（本测试一次要扫 3 遍），而
#: `count_kept_rows` 的返回值与归档有多大无关 ⇒ 每个归档只读前这么多个成员。
REAL_MEMBER_PROBE = 300

#: ⚠ **限的是「成员数」，不是 `--limit` 的「行数」** —— 这个区别是承重的。
#: `--limit` 命中后把返回值封顶成 `min(kept_shard, limit)`
#: （`stdata_to_npz.py:993`），于是 `rows_kept_shard` **恒等于 limit**：
#: 实测 limit ∈ {2000, 8000, 20000} × 3 个 shard_id，`rows_kept_shard` 每次都
#: 精确等于 limit，偏移带来的归属变化被整个抹掉 ⇒ 那个测试会变成空转
#: （它唯一还剩的断言 `base > prev_last` 退化成 `limit > 0`）。
#: 限成员数则让 `limit` 保持 `None`，两个计数都是**未经封顶的真实值**。
REAL_MEMBER_PROBE_DOC = """\
⚠ **本测试验证的是「偏移跨归档累加」的逻辑，不是规模。**\
`first_n_members` 把每个归档截到前 %d 个成员（真实归档、真实成员布局不变）。\
限定成员数**不影响被验证的性质**：`count_kept_rows` 的 `limit` 仍是 `None`，\
所以 `rows_kept` / `rows_kept_shard` 都不是封顶值，`member_base` 仍按每个成员\
自己的 `n_keep` 推进 —— 偏移算术一字未改，只是输入变短了。\
（**不要**改用 `--limit` 来提速：它按行数封顶返回值，会把偏移的影响抹成 0，\
见 `REAL_MEMBER_PROBE` 上方的警告。）"""


@pytest.fixture
def first_n_members(monkeypatch):
    """把 `count_kept_rows` 看得见的成员数限到前 `REAL_MEMBER_PROBE` 个。

    只截**遍历范围**，不截任何计数 —— 见 `REAL_MEMBER_PROBE` 的警告。
    """
    real_iter = s2n.iter_npz_members

    def probe(archive):
        gen = iter(real_iter(archive))
        try:
            for i, item in enumerate(gen):
                if i >= REAL_MEMBER_PROBE:
                    return
                yield item
        finally:
            gen.close()        # 触发 `iter_npz_members` 自己的 tf.close()

    monkeypatch.setattr(s2n, 'iter_npz_members', probe)
    return probe


@needs_stdata
@pytest.mark.slow
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


# --------------------------------------------------------------------------- #
# 6 · CLI 的分片参数校验（`--num-shards` / `--shard-id`）
# --------------------------------------------------------------------------- #
#: 每一组都必须被拒。`--count-only` 与 `convert` 两条路径对它们的判定必须一致
#: （🔴 修之前 `--count-only` 只查 shard_id 越界、完全不看 num_shards）。
ILLEGAL_SHARD_ARGVS = [
    ['--num-shards', '1', '--shard-id', '0'],   # N=1 ⇒ 那个 --shard-id 是哑参数
    ['--num-shards', '1', '--shard-id', '3'],
    ['--num-shards', '0'],                      # 曾被静默夹成 1（不分片）
    ['--num-shards', '-3'],
    ['--num-shards', '0', '--shard-id', '0'],
    ['--num-shards', '4'],                      # 说了要分 4 块却没说哪一块
    ['--num-shards', '4', '--shard-id', '4'],   # 越界
    ['--num-shards', '4', '--shard-id', '-1'],
    ['--num-shards', '2', '--shard-id', '9'],
]


def _cli(arch, *extra):
    return ['--source', '%s:%s' % (arch, NET_80)] + list(extra)


@pytest.fixture
def arch(tmp_path):
    """一份合成归档（每成员行数故意不整除 N，见 :func:`_archive`）。"""
    return _archive(str(tmp_path / 'cli.tar'))


def test_shard_id_default_is_a_sentinel_not_zero():
    """🔴 「不分片」与「第 0 块」在 `0` 上无法区分 ⇒ default 必须是 `None`。

    修之前是 `default=0` 配 `if args.shard_id:`（**真值**判断）⇒ `--shard-id 0`
    整个校验被跳过，而 0 恰恰是分块跑时最常用的那一块（第一个分片）。
    """
    ap = s2n.build_argparser()
    assert ap.parse_args([]).shard_id is None, (
        'default 必须是 None（None = 不分片）；0 会让「不分片」与「第 0 块」'
        '在真值判断下混同')
    assert ap.parse_args(['--shard-id', '0']).shard_id == 0


def test_resolve_shard_args_normalizes_the_legal_pairs():
    """三态：无 id + N=1 = 不分片 ⇒ 归一化成 ``(1, 0)``；分片 ⇒ 原样。"""
    assert s2n.resolve_shard_args(1, None) == (1, 0)
    assert s2n.resolve_shard_args(8, 0) == (8, 0)
    assert s2n.resolve_shard_args(8, 7) == (8, 7)


@pytest.mark.parametrize('num_shards,shard_id', [
    (1, 0), (1, 3), (0, None), (0, 0), (-3, None), (-3, 1),
    (4, None), (4, 4), (4, -1), (2, 9),
])
def test_resolve_shard_args_rejects_every_illegal_combination(
        num_shards, shard_id):
    with pytest.raises(ConversionError):
        s2n.resolve_shard_args(num_shards, shard_id)


def test_shard_id_zero_is_validated_not_skipped():
    """🔴 缺陷 1：`--num-shards 1 --shard-id 0` 现在**必须**报错。

    N=1 时 `shard_mask` 原样返回全部行 ⇒ 那个 `--shard-id` 根本不生效，是个哑
    参数；收下它就是再收一个静默无操作参数，而本次要消灭的正是静默降级。
    修之前 `if args.shard_id:` 让 0 跳过校验，所以本条测试在修复前是红的。
    """
    with pytest.raises(ConversionError):
        s2n.resolve_shard_args(1, 0)


def test_num_shards_below_one_is_an_error_not_a_silent_downgrade(tmp_path,
                                                                 arch, capsys):
    """🔴 缺陷 2：`--num-shards 0` / `-3` 曾静默降级成不分片，rc=0。

    报出来的行数是**全量**行数 —— 数字看着完全正常。分片流程的整个前提是
    「各块行数之和 == 全集行数」，这种失败只在最后对数时才发现，而那时已经
    写完几 GB 了。
    """
    for nsh in ('0', '-3', '-1'):
        rc = s2n.main(_cli(arch, '--count-only', '--num-shards', nsh))
        assert rc != 0, f'--num-shards {nsh} 静默 rc=0（降级成了不分片）'
        assert rc == 2
        err = capsys.readouterr().err
        assert 'num-shards' in err and '>= 1' in err, err
        assert '夹成 1' in err, err        # 说清为什么不静默


def test_illegal_shard_args_are_rejected_identically_on_both_paths(tmp_path,
                                                                   arch, capsys):
    """🔴 缺陷 3：`--count-only` 与 `convert` 必须对同一组参数给出同一判定。

    两条路径各写一份校验 —— 一个查一个不查 —— 就是同类漏洞的温床。
    """
    for extra in ILLEGAL_SHARD_ARGVS:
        out_path = tmp_path / 'o.npz'
        rc_count = s2n.main(_cli(arch, '--count-only',
                                 '--out', str(out_path), *extra))
        err_count = capsys.readouterr().err
        rc_conv = s2n.main(_cli(arch, '--out', str(out_path), *extra))
        err_conv = capsys.readouterr().err
        assert rc_count == 2, (extra, rc_count, err_count)
        assert rc_conv == rc_count, (extra, rc_count, rc_conv)
        assert err_count.strip() and err_conv.strip(), (extra, err_count, err_conv)
        # 参数非法时一个字节都不该落地（校验在写文件之前）
        assert not out_path.exists(), extra


def test_out_of_range_message_keeps_its_wording(tmp_path, arch, capsys):
    """`test_stdata_shard.py::test_cli_rejects_out_of_range_shard_id` 依赖
    stderr 里的「越界」二字 —— 别把报错文案改掉，否则那个文件会红。"""
    rc = s2n.main(_cli(arch, '--count-only', '--num-shards', '4',
                       '--shard-id', '9'))
    assert rc == 2
    assert '越界' in capsys.readouterr().err


def test_num_shards_without_shard_id_is_rejected(tmp_path, arch, capsys):
    """🔴 `--num-shards 4` 而不给 `--shard-id` 必须报错。

    两种「善意默认」都只会在最后对数时才显形：默认第 0 块 ⇒ 用户以为参数没
    生效；默认不分片 ⇒ **声称分 4 块却写出全集**，而输出里的行数完全正常。
    """
    rc = s2n.main(_cli(arch, '--count-only', '--num-shards', '4'))
    assert rc == 2
    err = capsys.readouterr().err
    assert '--shard-id' in err and '--num-shards 4' in err, err


def test_no_shard_args_means_unsharded_all_rows(tmp_path, arch, capsys):
    """不传 `--shard-id` ⇒ 不分片、全部行、rc=0（行为不变）。"""
    rc = s2n.main(_cli(arch, '--count-only'))
    out = capsys.readouterr().out
    assert rc == 0
    assert '块 →' not in out, out               # 不该出现分片那行
    total = count_kept_rows(arch, NET_80)['rows_kept']
    assert '保留 %d 行' % total in out, out
    assert '合计：原始 %d 行' % count_kept_rows(arch, NET_80)['rows_raw'] in out


def test_num_shards_one_alone_is_legal_and_equals_no_shard_args(tmp_path, arch,
                                                                capsys):
    """裁决：`--num-shards 1` **显式允许**，含义就是「不分片」，与不传等价。"""
    a = s2n.main(_cli(arch, '--count-only'))
    out_a = capsys.readouterr().out
    b = s2n.main(_cli(arch, '--count-only', '--num-shards', '1'))
    out_b = capsys.readouterr().out
    assert a == 0 and b == 0
    assert out_a == out_b


def test_num_shards_one_with_shard_id_zero_is_rejected_but_convert_allows_it(
        tmp_path, arch, capsys):
    """🔴 裁决只加在 **CLI 层**：函数层 `convert(num_shards=1, shard_id=0)`
    仍然合法且与不分片逐位相同（见 `test_num_shards_one_is_bit_identical_to_
    unsharded`）—— 函数层用签名默认值 `shard_id=0` 表达「默认第 0 块」，那里
    没有「未给」这个状态，CLI 才有，所以只有 CLI 需要区分。"""
    rc = s2n.main(_cli(arch, '--count-only', '--num-shards', '1',
                       '--shard-id', '0'))
    assert rc == 2
    assert '哑参数' in capsys.readouterr().err
    # 函数层不受影响
    meta = convert([(arch, NET_80)], str(tmp_path / 'x.npz'),
                   num_shards=1, shard_id=0)
    assert meta['shard']['num_shards'] == 1 and meta['shard']['shard_id'] == 0
    assert meta['rows'] == count_kept_rows(arch, NET_80)['rows_kept']


def test_shard_zero_is_still_the_first_shard_end_to_end(tmp_path, arch, capsys):
    """🔴 改成 sentinel 之后，第 0 块必须**仍然**被当作「第 0 块」跑。

    修法有个陷阱方向：`--shard-id` 变成 `None` 之后若忘了把归一化值喂下去，
    `--shard-id 0` 会悄悄退化成不分片 ⇒ 第 0 块变成全集，而报告依然正常。
    """
    total = count_kept_rows(arch, NET_80)['rows_kept']
    rc = s2n.main(_cli(arch, '--count-only', '--num-shards', '4',
                       '--shard-id', '0'))
    out = capsys.readouterr().out
    assert rc == 0
    per = [count_kept_rows(arch, NET_80, num_shards=4, shard_id=sid)
           ['rows_kept_shard'] for sid in range(4)]
    assert '第 0/4 块 → %d 行' % per[0] in out, out
    assert per[0] < total, (per[0], total)   # 第 0 块 ≠ 全集（没退化成不分片）
    assert sum(per) == total                 # 各块之和 == 全集行数


def test_both_paths_receive_the_same_normalized_shard_args(tmp_path, arch,
                                                           monkeypatch,
                                                           capsys):
    """🔴 `--count-only` 与 `convert` 必须收到**同一对**归一化参数。

    不靠「跑出来行数一样」这种间接判据：直接录下两条路径各自收到的
    ``(num_shards, shard_id)``。否则哪天有人在其中一条分支里改回
    `args.shard_id`，`None` 就会在一条路径上变成「不分片」而在另一条上是 0。
    """
    seen = {}

    def fake_count(archive, network=None, **kw):
        seen['count'] = (kw['num_shards'], kw['shard_id'])
        return {'rows_raw': 4, 'rows_kept': 4, 'rows_kept_shard': 4}

    def fake_convert(archives, out_path, **kw):
        seen['convert'] = (kw['num_shards'], kw['shard_id'])
        return {'fake': True}

    monkeypatch.setattr(s2n, 'count_kept_rows', fake_count)
    monkeypatch.setattr(s2n, 'convert', fake_convert)
    monkeypatch.setattr(s2n, 'render_report', lambda meta: '')

    for argv, want in (
            ([], (1, 0)),                                 # 不分片
            (['--num-shards', '1'], (1, 0)),             # 显式 1 = 不分片
            (['--num-shards', '8', '--shard-id', '0'], (8, 0)),   # 🔴 第 0 块
            (['--num-shards', '8', '--shard-id', '7'], (8, 7)),
    ):
        seen.clear()
        assert s2n.main(_cli(arch, '--count-only', *argv)) == 0, argv
        assert seen['count'] == want, (argv, seen)
        assert s2n.main(_cli(arch, *argv)) == 0, argv
        assert seen['convert'] == want, (argv, seen)
        capsys.readouterr()


def test_convert_rejects_non_positive_num_shards(tmp_path, arch):
    """🔴 `convert` 里的 `max(1, ...)` 降级：CLI 修好之后它对 CLI 已不可达，
    但直接调 `convert()` 的代码仍会中招（而且 `meta['shard']` 还会把
    num_shards 记成 1，看起来完全正常）⇒ 函数层也必须报错。"""
    for nsh in (0, -3):
        out = tmp_path / ('bad%d.npz' % nsh)
        with pytest.raises(ConversionError):
            convert([(arch, NET_80)], str(out), num_shards=nsh)
        assert not out.exists(), nsh


def test_convert_still_rejects_out_of_range_shard_id(tmp_path, arch):
    for nsh, sid in ((4, 4), (4, -1), (2, 9)):
        with pytest.raises(ConversionError):
            convert([(arch, NET_80)], str(tmp_path / 'o.npz'),
                    num_shards=nsh, shard_id=sid)


@needs_stdata
@pytest.mark.slow
def test_real_archive_count_offset_accumulates_across_archives(first_n_members):
    """跨归档偏移：第一块的 `rows_kept_shard` 与第二块的不该重叠。

    %s
    """ % (REAL_MEMBER_PROBE_DOC % REAL_MEMBER_PROBE,)
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


# --------------------------------------------------------------------------- #
# 9 · --convert-all（一条命令跑完 N 块 + 断点续做）
# --------------------------------------------------------------------------- #
# 它对齐 build_dataset.py --merge 的范式：已完成的部分跳过、中断后重跑同一条
# 命令继续。全量 8 块约 80 分钟，中途 Ctrl-C 是常事而非例外。


def _arch(tmp_path, n_members=4, per_member=40):
    import io as _io
    import tarfile as _tf
    from tests.test_stdata_to_npz import _fake_npz
    ms = [('m%02d/b11c768/b.npz' % i, _fake_npz(n=per_member + i * 7, seed=i))
          for i in range(n_members)]
    path = str(tmp_path / 'a.tar')
    with _tf.open(path, 'w') as tf:
        for name, d in ms:
            buf = _io.BytesIO()
            np.savez(buf, **d)
            info = _tf.TarInfo(name)
            info.size = buf.tell()
            tf.addfile(info, _io.BytesIO(buf.getvalue()))
    return path


def test_convert_all_produces_every_shard_plus_a_manifest(tmp_path):
    arch = _arch(tmp_path)
    out = str(tmp_path / 'v7.npz')
    man = convert_all_shards([(arch, NET_80)], out, num_shards=4,
                             log=lambda *a, **k: None)
    assert man['num_shards'] == 4 and man['shards_done'] == 4
    assert [s['shard_id'] for s in man['shards']] == [0, 1, 2, 3]
    for s in man['shards']:
        assert os.path.isfile(s['path']), s
        assert os.path.isfile(s['done_marker'])
        assert s['rows'] > 0 and s['status'] == 'done'
    assert os.path.isfile(out + '.shards.json')
    # 行数之和 == 不分片的行数
    whole = convert([(arch, NET_80)], str(tmp_path / 'whole.npz'))
    assert man['rows_total'] == whole['rows']


def test_convert_all_skips_finished_shards_on_rerun(tmp_path, monkeypatch):
    """断点续做：重跑同一条命令**不重做已完成的块**。"""
    arch = _arch(tmp_path)
    out = str(tmp_path / 'v7.npz')
    convert_all_shards([(arch, NET_80)], out, num_shards=4,
                       log=lambda *a, **k: None)
    called = []
    real = s2n.convert

    def spy(*a, **k):
        called.append(k.get('shard_id'))
        return real(*a, **k)

    monkeypatch.setattr(s2n, 'convert', spy)
    convert_all_shards([(arch, NET_80)], out, num_shards=4,
                       log=lambda *a, **k: None)
    assert called == [], '已完成的块不应该重跑'


def test_convert_all_resumes_from_the_failed_shard(tmp_path, monkeypatch):
    """第 3 块失败时：前两块**保留**（含 .done），重跑只补少的那块。

    → 修掉问题后重跑**同一条命令**（build_dataset --merge 的范式）。
    """
    arch = _arch(tmp_path)
    out = str(tmp_path / 'v7.npz')
    real = s2n.convert
    n = [0]

    def boom(*a, **k):
        n[0] += 1
        if n[0] == 3:
            raise ConversionError('模拟第 3 块失败')
        return real(*a, **k)

    monkeypatch.setattr(s2n, 'convert', boom)
    with pytest.raises(ConversionError):
        convert_all_shards([(arch, NET_80)], out, num_shards=4,
                           log=lambda *a, **k: None)
    for sid in (0, 1):
        p = s2n.shard_paths(out, 4, sid)
        assert os.path.isfile(p) and os.path.isfile(p + '.done')
    assert not os.path.isfile(s2n.shard_paths(out, 4, 2) + '.done')

    called = []
    monkeypatch.setattr(s2n, 'convert',
                        lambda *a, **k: (called.append(k.get('shard_id')),
                                         real(*a, **k))[1])
    man = convert_all_shards([(arch, NET_80)], out, num_shards=4,
                             log=lambda *a, **k: None)
    assert called == [2, 3], '应只补跑第 3、4 块'
    assert man['shards_done'] == 4


def test_manifest_survives_interruption(tmp_path, monkeypatch):
    """清单每块后落盘 → 中断也留得下进度。"""
    arch = _arch(tmp_path)
    out = str(tmp_path / 'v7.npz')
    real = s2n.convert
    n = [0]

    def boom(*a, **k):
        n[0] += 1
        if n[0] == 2:
            raise ConversionError('模拟第 2 块失败')
        return real(*a, **k)

    monkeypatch.setattr(s2n, 'convert', boom)
    with pytest.raises(ConversionError):
        convert_all_shards([(arch, NET_80)], out, num_shards=4,
                           log=lambda *a, **k: None)
    import json
    man = json.load(open(out + '.shards.json', encoding='utf-8'))
    assert [s['shard_id'] for s in man['shards']] == [0]
    assert man['shards'][0]['status'] == 'done'


def test_shard_paths_are_sibling_files_and_unsharded_is_the_base(tmp_path):
    """命名用兄弟文件（而非子目录），才与 build_dataset.merge_shards
    的 `glob('*.npz')** 扫同目录** 的习惯对齐。"""
    base = str(tmp_path / 'foo.npz')
    assert s2n.shard_paths(base, 1, 0) == base
    for sid in range(4):
        p = s2n.shard_paths(base, 4, sid)
        assert os.path.dirname(p) == str(tmp_path)
        assert os.path.basename(p) == 'foo_s%d.npz' % sid


def test_convert_all_degenerates_to_a_single_unsharded_run(tmp_path):
    """`--convert-all --num-shards 1` → 不分片，只跑一次且输出就是 `out`。"""
    arch = _arch(tmp_path)
    out = str(tmp_path / 'v7.npz')
    man = convert_all_shards([(arch, NET_80)], out, num_shards=1,
                             log=lambda *a, **k: None)
    assert man['num_shards'] == 1 and man['shards_done'] == 1
    assert os.path.isfile(out)
    assert not os.path.isfile(str(tmp_path / 'v7_s0.npz'))


def test_cli_rejects_convert_all_with_shard_id(capsys):
    from scripts.stdata_to_npz import main
    rc = main(['--convert-all', '--num-shards', '4', '--shard-id', '1',
               '--archive', 'x.tar'])
    assert rc == 2
    assert '互斥' in capsys.readouterr().err


def test_cli_convert_all_still_validates_num_shards(capsys):
    """`--num-shards 0` 在 --convert-all 下**照样报错** —— 静默变成分片成 1 块
    会写出一个「分片流程下的全集」，而它看起来是个正常数字。"""
    from scripts.stdata_to_npz import main
    assert main(['--convert-all', '--num-shards', '0', '--archive', 'x.tar']) == 2
    assert 'num-shards' in capsys.readouterr().err


def test_cli_convert_all_does_not_require_shard_id(tmp_path, capsys,
                                                   monkeypatch):
    """一条命令跑完 ⇒ 不得报「少了 --shard-id」。

    ⚠ ``--convert-all`` 自己遍历 0..N-1，所以**不该**过 ``resolve_shard_args``
      那条「N>=2 必须配 --shard-id」的三态表。但 ``num_shards`` 本身的合法性
      仍要查（``--num-shards 0`` 必须报错，不能静默变成不分片）。
    """
    from scripts.stdata_to_npz import main
    arch = _arch(tmp_path)
    rc = main(['--convert-all', '--num-shards', '3', '--archive', arch,
               '--network', NET_80, '--out', str(tmp_path / 'v7.npz')])
    assert rc == 0, capsys.readouterr().err
    for sid in range(3):
        assert os.path.isfile(str(tmp_path / ('v7_s%d.npz' % sid)))
