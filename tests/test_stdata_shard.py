# -*- coding: utf-8 -*-
"""`scripts/stdata_to_npz.py` 的**分片**契约测试。

被测对象是 :func:`scripts.stdata_to_npz.shard_mask`（6 行代码）加上它的两个
调用点 :func:`count_kept_rows`（第 0 遍）与 :func:`subset_labels`。

为什么这 6 行值得单独一份测试
---------------------------
`shard_mask` 是「N 块分别转换、峰值内存 ÷ N」这套做法的**全部**正确性依据，
而它自己的 docstring 里点名了一个**全程不报错**的经典错：

    ``global_offset`` 必须按「分片过滤**之前**」的行数推进。误用过滤后的行数
    ⇒ 从第 2 个成员起，同一行会同时落进两块（重复），另一些行谁都不落（丢失）。

两遍（`count_kept_rows` / `convert`）会**一起错** ⇒ `convert` 的
``a_rows != budget`` 检查抓不到 ⇒ 只会在训练数据里悄悄多出重复样本、少掉
一些样本。而 ``--num-shards`` 的转换期只看得到**每块的行数**，行数仍然"看着
合理" ⇒ 唯一的抓手就是「哪些全局行号落进哪块」。所以这份测试钉的正是
docstring 承诺的三条可证性质 + 与成员边界无关 + 那个经典错的对照。

本文件钉的东西
--------------
1. 三条可证性质：并集 == 全集 / 两两不相交 / 大小至多差 1（大块数 == n % N）；
2. ``num_shards <= 1`` 或 ``None`` ⇒ 原样返回全部行（以及 ``shard_id`` 越界
   仍然返回全部行这个反直觉角落）；
3. 🔴 回归：多个成员（大小不均，模拟实测 ~63 行/成员）× 多个 shard，按
   **分片过滤前**的行数推进偏移 ⇒ 收集起来恰好是全集、无重复无丢失；并与
   **故意写错的参考实现**（按过滤后行数推进）对照，证明它真的会重复 + 丢失；
4. 与成员边界无关：同样 200 行拆成 1×200 / 2×100 / 4×50 / 7×~29，
   同一 shard 拿到**同一个**全局行号集合；
5. ``shard_id`` 越界 / 负数的**实际行为**（当前实现：静默返回空数组；校验在
   ``convert`` / CLI 那一层，不在 ``shard_mask`` 里）；
6. ``n_rows == 0``；
7. ``subset_labels``：``to_v7_labels`` 返回值里四类东西（行轴 ndarray /
   嵌套 dict / tuple / 标量）都切对、``_kept`` 改成子集长度、``_dropped`` 刻意不动；
8. 第 0 遍真归档：各 shard 行数之和 == 未分片总行数，且**逐块行数**与
   「独立算一遍的期望值」逐个相等（能把「偏移按原始行数推进」和
   「偏移按过滤后行数推进」两种错都区分出来）。

⚠ **本文件不断言任何吞吐量 / 耗时** —— 分片要买的是峰值内存，测耗时会写出一条
  在别人机器上随机红的测试。

⚠「各块 ``game_ids`` 区间重叠」**不能**用来判断分片算错了：`labels_to_chunk`
  给的是 ``arange(written, written + n)``，即**本块内**的行号，每块各自从 0 起
  ⇒ 跨块重叠是设计如此（``meta.shard.game_ids_note`` 明说了）。``shard_mask`` 的
  docstring 那么写是不准确的（见报告）；可判定的等价说法是「同一行被两块同时
  领走」，本文件用的就是它。
"""

import collections
import io
import os
import sys
import tarfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import stdata_to_npz as s2n   # noqa: E402
from src.data.katago_npz import (   # noqa: E402
    BOARD_STRIDE,
    PACKED_BYTES,
    SCORE_DISTR_BINS,
    SPATIAL_CHANNELS,
)

NET_80 = 'kata1-tf3-b11c768'


# --------------------------------------------------------------------------- #
# 1 · 三条可证性质
# --------------------------------------------------------------------------- #
#: ``(n_rows, num_shards)``：含 ``n < N``（有的块必须为空）、``N == 1``、
#: 整除与不整除，以及实测的 ~63 行/成员尺度。
SHARD_CASES = [
    (0, 1), (1, 1), (1, 8), (2, 5), (5, 2), (7, 3), (63, 4), (100, 7),
    (200, 8), (341, 3), (1000, 7),
]

#: 含非零 ``global_offset`` 的偏移组合（偏移只旋转归属，不改三条性质）。
OFFSET_CASES = [0, 1, 63, 64, 341, 5000]


@pytest.mark.parametrize('n_rows,num_shards', SHARD_CASES)
def test_union_of_all_shards_is_the_whole_set(n_rows, num_shards):
    """性质 ①：N 块的**并集恒等于全集**。"""
    got = sorted(np.concatenate(
        [s2n.shard_mask(n_rows, 0, num_shards, s) for s in range(num_shards)]
    ).tolist())
    assert got == list(range(n_rows))


@pytest.mark.parametrize('n_rows,num_shards', SHARD_CASES)
def test_shards_are_pairwise_disjoint(n_rows, num_shards):
    """性质 ②：**两两不相交**（同一行绝不同时落进两块）。"""
    sets = [set(s2n.shard_mask(n_rows, 0, num_shards, s).tolist())
            for s in range(num_shards)]
    for a in range(num_shards):
        for b in range(a + 1, num_shards):
            assert not sets[a] & sets[b], f'shard {a} 与 shard {b} 抢了同一行'


@pytest.mark.parametrize('n_rows,num_shards', SHARD_CASES)
@pytest.mark.parametrize('global_offset', OFFSET_CASES)
def test_shard_sizes_differ_by_at_most_one(n_rows, num_shards, global_offset):
    """性质 ③：各块大小 ∈ ``{q, q+1}``（``q = n // N``），且大块数 == ``n % N``。

    ⚠ 这条正是「与成员边界无关」的**可测形式**：成员大小不均也不会让某块偏大，
      所以偏移换成什么值都只影响「哪一块多拿那一行」，不影响大小分布。
    """
    sizes = [s2n.shard_mask(n_rows, global_offset, num_shards, s).size
             for s in range(num_shards)]
    q, rem = divmod(n_rows, num_shards)
    assert set(sizes) <= {q, q + 1}, f'块大小 {sizes} 超出 {{q, q+1}} = {q, q + 1}'
    assert sizes.count(q + 1) == rem, f'大块数 {sizes.count(q + 1)} != n % N = {rem}'
    assert sum(sizes) == n_rows, '并集大小必须等于全集大小'


@pytest.mark.parametrize('global_offset', OFFSET_CASES)
@pytest.mark.parametrize('shard_id', [0, 1, 3])
def test_selection_is_exactly_the_modulo_rule(global_offset, shard_id):
    """归属判据就是 docstring 写的那一条：``(global_offset + j) % N == shard_id``。

    🔴 这条**逐点**核对，而不是只核对三条性质 —— 换成切连续区间（同样满足三条
      性质、同样与成员边界无关）时它会立刻红，而三条性质不会。
    """
    n_rows, num_shards = 341, 7
    j = np.arange(n_rows, dtype=np.int64)
    want = j[((global_offset + j) % num_shards) == shard_id]
    assert np.array_equal(
        s2n.shard_mask(n_rows, global_offset, num_shards, shard_id), want)


@pytest.mark.parametrize('global_offset', OFFSET_CASES)
def test_result_is_ascending_int64_local_indices(global_offset):
    """返回值是**局部**下标、升序、``int64``（docstring 的签名承诺）。"""
    j = s2n.shard_mask(341, global_offset, 7, 2)
    assert j.dtype == np.int64
    assert j.ndim == 1
    assert np.array_equal(j, np.unique(j)), '不是升序 / 有重复'
    assert 0 <= int(j[0]) and int(j[-1]) < 341, '不是本成员内的局部下标'


# --------------------------------------------------------------------------- #
# 2 · 不分片 / 越界 id / 0 行
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('num_shards', [None, 1, 0, -1, -8])
@pytest.mark.parametrize('global_offset', [0, 63])
def test_num_shards_le_one_returns_every_row_verbatim(num_shards, global_offset):
    """``num_shards`` 为 ``None`` 或 ``<= 1`` ⇒ 原样返回 ``np.arange(n)``。

    ⚠ ``convert`` 的承诺是「``num_shards<=1`` 时行为与不分片**逐位相同**」——
      所以这里连偏移都不该看一眼。
    """
    n_rows = 65
    got = s2n.shard_mask(n_rows, global_offset, num_shards, 0)
    assert got.dtype == np.int64
    assert np.array_equal(got, np.arange(n_rows, dtype=np.int64))


@pytest.mark.parametrize('shard_id', [1, 4, 99, -1])
def test_single_shard_ignores_shard_id_entirely(shard_id):
    """⚠ **反直觉角落**（当前实现）：``num_shards<=1`` 时 ``shard_id`` **完全不看**，
    越界 / 负数也照样返回**全部**行。

    ⇒ ``shard_mask`` 自己不校验 ``shard_id``（校验在 `convert` 与 CLI 那一层），
    直接调它时一个越界 id 不报错、反而给回全部行。把 docstring 的「升序 int64
    局部下标」当契约用的代码必须自己保证 ``0 <= shard_id < num_shards``。
    """
    assert np.array_equal(s2n.shard_mask(6, 0, 1, shard_id),
                          np.arange(6, dtype=np.int64))


@pytest.mark.parametrize('shard_id', [4, 99, -1, -7])
def test_out_of_range_shard_id_silently_returns_empty(shard_id):
    """真的在分片时（``N>1``），越界 / 负数 ``shard_id`` **不报错**，返回**空数组**。

    这样也说得过去：空数组让调用方一行都拿不到，转换期立刻以「第 0 块一行都没
    分到」报错退出 ⇒ 不会静默写错。但直接调 `shard_mask` 的代码必须自己校验。
    """
    j = s2n.shard_mask(70, 0, 4, shard_id)
    assert isinstance(j, np.ndarray)
    assert j.dtype == np.int64
    assert j.size == 0


@pytest.mark.parametrize('num_shards,shard_id', [(1, 0), (4, 0), (4, 3), (4, 9)])
def test_zero_rows(num_shards, shard_id):
    """``n_rows == 0``：空成员必须给出空下标，而不是报错或返回一行。"""
    j = s2n.shard_mask(0, 0, num_shards, shard_id)
    assert isinstance(j, np.ndarray) and j.dtype == np.int64 and j.size == 0


def test_convert_rejects_out_of_range_shard_id(tmp_path):
    """校验在 ``convert`` 那一层（``shard_mask`` 自己不管）—— 这里钉住它还在。"""
    with pytest.raises(s2n.ConversionError, match='越界'):
        s2n.convert([], str(tmp_path / 'o.npz'), num_shards=4, shard_id=4)
    with pytest.raises(s2n.ConversionError, match='越界'):
        s2n.convert([], str(tmp_path / 'o.npz'), num_shards=4, shard_id=-1)


# --------------------------------------------------------------------------- #
# 3 · 🔴 回归钉子：偏移必须按「分片过滤**前**」的行数推进
# --------------------------------------------------------------------------- #
#: 实测每个成员 ~63 行且大小不均；这里刻意不整除任何常见的分片数。
MEMBER_ROWS = [63, 41, 63, 7, 63, 28, 12, 63]
TOTAL_ROWS = sum(MEMBER_ROWS)          # 340
NUM_SHARDS = 4


def _select(member_rows, num_shards, *, offset_step='pre'):
    """逐（块, 成员）照抄 `convert` / `count_kept_rows` 的偏移记账。

    对应的源码是这两行::

        j = shard_mask(n_local, global_seen, num_shards, shard_id)
        global_seen += n_local            # ⚠ 按分片过滤前的行数推进

    Args:
        member_rows: 每个成员**进入分片**的行数（实测 ~63，大小不均）。
        num_shards: 块数。
        offset_step: ``'pre'`` = 正确；``'post'`` = 🔴 **故意写错**的那个经典错
            （偏移按**分片过滤后**的行数 ``j.size`` 推进）。

    Returns:
        ``{shard_id: [(该成员的 global_offset, 局部下标 j), ...]}``。
    """
    out = {}
    for shard_id in range(num_shards):
        base = 0
        per_member = []
        for n_local in member_rows:
            member_base = base
            j = s2n.shard_mask(n_local, member_base, num_shards, shard_id)
            base = member_base + (j.size if offset_step == 'post' else n_local)
            per_member.append((member_base, j))
        out[shard_id] = per_member
    return out


def _collect(member_rows, num_shards, *, offset_step='pre'):
    """把 :func:`_select` 的结果摊成「每块实际写出去的东西」。

    ``game_ids`` 按 `labels_to_chunk` 的做法给：``arange(written, written + n)``
    —— 本块内从 0 起的行号，``written`` 随写出推进。
    """
    out = {}
    for shard_id, per_member in _select(member_rows, num_shards,
                                       offset_step=offset_step).items():
        written = 0
        rows, gids, spans = [], [], []
        for member_base, j in per_member:
            if j.size == 0:
                continue
            gid = np.arange(written, written + j.size, dtype=np.int64)
            written += j.size
            rows.append(member_base + j)
            gids.append(gid)
            spans.append((int(gid[0]), int(gid[-1]) + 1))
        empty = np.zeros(0, dtype=np.int64)
        out[shard_id] = (np.concatenate(rows) if rows else empty,
                         np.concatenate(gids) if gids else empty,
                         spans)
    return out


def _claim_counts(got):
    """每个全局行号被几块领走（``{行号: 块数}``）。"""
    c = collections.Counter()
    for shard_id in sorted(got):
        c.update(got[shard_id][0].tolist())
    return c


def test_pre_filter_offset_partitions_every_member_exactly_once():
    """🔴 核心回归：跨**全部成员 × 全部 shard** 收集起来的行恰好是全集。

    这是那个经典错**唯一**能被抓到的形态 —— 它不抛异常、每块的行数也仍然
    "看着合理"（见 `test_the_classic_bug_is_invisible_in_the_row_counts`），
    只有「哪一行落进哪一块」会错。
    """
    got = _collect(MEMBER_ROWS, NUM_SHARDS)
    all_rows = np.concatenate([got[s][0] for s in range(NUM_SHARDS)])
    assert all_rows.size == TOTAL_ROWS, '收集到的行数与全集不符'
    assert sorted(all_rows.tolist()) == list(range(TOTAL_ROWS)), '有行重复或丢失'
    for s in range(NUM_SHARDS):
        assert got[s][0].tolist() == [r for r in range(TOTAL_ROWS)
                                      if r % NUM_SHARDS == s], \
            f'shard {s} 拿到的不是「全局行号取模 == {s}」那一组'


def test_no_row_is_claimed_by_two_shards():
    """等价说法（也是唯一可判定的那个）：**同一行不会被两块同时领走**。

    把 ``(块号, game_ids)`` 当成行在输出文件里的地址，「重复 / 丢失」就成了一句
    可断言的话 —— ``game_ids`` 是这里唯一的溯源列。
    """
    counts = _claim_counts(_collect(MEMBER_ROWS, NUM_SHARDS))
    assert max(counts.values()) == 1, '同一行被多块领走（重复）'
    assert sorted(counts) == list(range(TOTAL_ROWS)), '有行谁都不落（丢失）'


def test_each_shard_game_ids_is_a_local_running_counter():
    """本块内 ``game_ids`` 必须逐行唯一、逐 chunk 区间不重叠（跨局守卫的载体）。

    ⚠ **不能**断言「各块 game_ids 区间不重叠」：``arange(written, written+n)``
      每块各自从 0 起，跨块重叠是**设计如此**（``meta.shard.game_ids_note``
      明说「分块后各块的游戏编号不再全局唯一」）。`shard_mask` 的 docstring 把
      「各块 game_ids 区间重叠」写成那个经典错的症状，是不准确的（见报告）。
    """
    for shard_id, (_, gids, spans) in _collect(MEMBER_ROWS, NUM_SHARDS).items():
        assert gids.tolist() == list(range(gids.size)), \
            f'shard {shard_id} 的 game_ids 不是逐行唯一的 0..n-1'
        prev_hi = 0
        for lo, hi in spans:
            assert lo == prev_hi, \
                f'shard {shard_id} 的 chunk 区间 [{lo},{hi}) 与前一块重叠'
            prev_hi = hi


def test_the_classic_bug_really_duplicates_and_loses_rows():
    """🔴 拿**故意写错的参考实现**在测试里现算一遍，证明它真的会坏。

    参考实现在 ``_select(offset_step='post')`` 里：偏移按**分片过滤后**的行数
    推进（``base += j.size``）。`stdata_to_npz.py` 一行都不动 —— 这条测试要
    证明的是「这个错会怎样坏」，而不是「别这么写」。
    """
    good = _select(MEMBER_ROWS, NUM_SHARDS, offset_step='pre')
    bad = _select(MEMBER_ROWS, NUM_SHARDS, offset_step='post')

    counts = _claim_counts(_collect(MEMBER_ROWS, NUM_SHARDS, offset_step='post'))
    dup = {r: k for r, k in counts.items() if k > 1}
    lost = sorted(set(range(TOTAL_ROWS)) - set(counts))
    assert dup, '这个实现竟然没有产生重复 ⇒ 参考实现写错了？'
    assert lost, '这个实现竟然没有丢行 ⇒ 参考实现写错了？'
    assert sum(counts.values()) != TOTAL_ROWS, '总行数竟然对上了？'

    # 症状的形状：第 1 个成员还对（偏移还没开始漂），从第 2 个成员起全错
    for shard_id in range(NUM_SHARDS):
        assert np.array_equal(good[shard_id][0][1], bad[shard_id][0][1]), \
            '第 1 个成员本该不受影响（这正是它隐蔽的原因）'
        assert any(not np.array_equal(gb, bb)
                   for (_, gb), (_, bb) in zip(good[shard_id][1:],
                                               bad[shard_id][1:])), \
            f'shard {shard_id} 从第 2 个成员起居然还和正确实现一样'


def test_the_classic_bug_is_invisible_in_the_row_counts():
    """🔴 为什么这个错必须靠「行号归属」而不是「行数」来抓：错的实现给出的块大小
    **仍然看着合理**（都落在理想值的 ±2 以内），转换期不会报任何错。
    """
    bad = _collect(MEMBER_ROWS, NUM_SHARDS, offset_step='post')
    ideal = TOTAL_ROWS / NUM_SHARDS
    for shard_id, (rows, _, _) in bad.items():
        assert abs(rows.size - ideal) <= 2, \
            f'块 {shard_id} 的行数 {rows.size} 离理想值 {ideal} 太远，' \
            '这条测试的前提（错得看不出）不成立'


# --------------------------------------------------------------------------- #
# 4 · 与成员边界无关
# --------------------------------------------------------------------------- #
#: 同样 200 行，四种切法（含 7 块的不均匀切法：4×29 + 3×28）。
LAYOUTS = [[200], [100, 100], [50, 50, 50, 50], [29, 29, 29, 29, 28, 28, 28]]


@pytest.mark.parametrize('layout', LAYOUTS)
def test_layouts_cover_the_same_two_hundred_rows(layout):
    assert sum(layout) == 200, '切法本身必须覆盖同一批行'
    assert all(n > 0 for n in layout), '成员不能是空的'


@pytest.mark.parametrize('shard_id', range(4))
def test_partition_does_not_depend_on_member_boundaries(shard_id):
    """「与成员边界无关」的直接检验：同样 200 行、同样的 shard，**同一个**全局
    行号集合，与成员怎么切无关。"""
    want = [r for r in range(200) if r % 4 == shard_id]
    for layout in LAYOUTS:
        got = _collect(list(layout), 4)[shard_id][0].tolist()
        assert got == want, f'成员切法 {layout} 下 shard {shard_id} 拿到的是 {got}'


def test_member_size_imbalance_does_not_inflate_one_shard():
    """成员大小不均（实测 ~63 行/成员，但不是每个都 63）也不会让某块偏大。"""
    for num_shards in (2, 3, 4, 5, 7, 8):
        q, rem = divmod(TOTAL_ROWS, num_shards)
        sizes = [rows.size
                 for rows, _, _ in _collect(MEMBER_ROWS, num_shards).values()]
        assert set(sizes) <= {q, q + 1}
        assert sizes.count(q + 1) == rem
        assert sum(sizes) == TOTAL_ROWS


# --------------------------------------------------------------------------- #
# 5 · subset_labels：把 shard_mask 的局部下标落到 labels 上
# --------------------------------------------------------------------------- #
#: `_labels()` 里所有**带行轴**的键（第 0 轴长度 == ``_kept``）。
ROW_AXIS_KEYS = (
    'outcome', 'score', 'score_distr', 'ownership', 'futurepos', 'spatial',
    'board_mask', 'q_value', 'legacy_probs',
)
#: 带行轴的 tuple（``policy_*_sparse`` = ``(idx, val)``）：两半都要是 0 行。
ROW_AXIS_TUPLES = ('policy_player_sparse',)


def _labels(n):
    """一份与 `to_v7_labels` **同构**的 labels dict：四类东西各来几个。

    * 行轴 ndarray（1-D / 2-D / 5-D）—— 判据是「第 0 轴长度 == ``_kept``」，
      所以尾轴形状无关紧要（真数据里是 ``(N,22,19,19)`` / ``(N,1,1,19,19)``）。
    * 嵌套 dict（``w``：6 个权重列）＋ 一个非行轴的嵌套值（该原样带走）。
    * tuple（``policy_*_sparse`` = ``(idx, val)``）＋ 一个半行轴半标量的 tuple。
    * 标量诊断（``_kept`` / ``_dropped`` / ``_network``）。

    ⚠ ``outcome`` 列装的是**行号**（真数据是 0/1/2）⇒ 切完能逐位核对切的是
      哪几行，而不只是「长度对了」。
    ⚠ ``legacy_probs`` 是**长度恰好等于 ``_kept`` 的非行轴数组** —— 刻意加的
      诱饵，见 `test_subset_labels_uses_the_kept_length_criterion`。
    """
    rows = np.arange(n, dtype=np.int64)
    return {
        'outcome': rows.copy(),                                   # 行轴 1-D
        'score': (rows * 10).astype(np.float32),
        'score_distr': np.tile(rows[:, None], (1, SCORE_DISTR_BINS)).astype(np.float32),
        'ownership': rows[:, None, None, None].astype(np.float32),
        'futurepos': np.stack([rows, rows + 1], 1)[:, :, None, None].astype(np.float32),
        'spatial': rows[:, None, None, None].astype(np.float32),
        'board_mask': np.ones((n, 1, 1, 2, 2), dtype=bool),
        'q_value': rows[:, None, None].astype(np.float32),
        'legacy_probs': rows[:, None].astype(np.float32),         # 诱饵
        'policy_player_sparse': (rows[:, None].astype(np.int64),
                                  (rows * 100)[:, None].astype(np.float32)),
        'policy_meta': ('topk=16', np.arange(3, dtype=np.int64)),
        'w': {
            'policy_opp': rows.astype(np.float32),
            'ownership': rows.astype(np.float32),
            'score': rows.astype(np.float32),
            'lead': rows.astype(np.float32),
            'futurepos': rows.astype(np.float32),
            'scoring': rows.astype(np.float32),
            'note': 'not-a-row-axis',
        },
        '_kept': n,
        '_dropped': 3,
        '_network': NET_80,
    }


def test_subset_labels_slices_every_kind_of_row_axis():
    """四类返回值里**两类带行轴**的都必须按 ``j`` 切，且切的是值不是长度。"""
    n, j = 64, np.array([0, 5, 63, 2, 31], dtype=np.int64)
    lb = _labels(n)
    out = s2n.subset_labels(lb, j)

    for key in ('outcome', 'score', 'score_distr', 'ownership', 'futurepos',
                'spatial', 'board_mask', 'q_value', 'legacy_probs'):
        assert out[key].shape[0] == j.size, f'{key} 的行数没跟着子集走'
        assert np.array_equal(out[key], lb[key][j]), f'{key} 的值不是 lb[key][j]'
    assert out['outcome'].tolist() == j.tolist(), \
        '载荷必须逐位是原行号（不能只对长度）'

    pi, pv = out['policy_player_sparse']            # tuple：两半都切
    assert np.array_equal(pi, lb['policy_player_sparse'][0][j])
    assert np.array_equal(pv, lb['policy_player_sparse'][1][j])

    for key in ('policy_opp', 'ownership', 'score', 'lead',   # 嵌套 dict：逐键切
                'futurepos', 'scoring'):
        assert np.array_equal(out['w'][key], lb['w'][key][j]), f'w[{key}] 没切对'


def test_subset_labels_keeps_non_row_axis_values():
    """标量与「非行轴」的值原样带走 —— `to_v7_labels` 的返回值里混着这些，
    漏掉会让块内行数与 ``game_ids`` 对不上（正是写手 ``close()`` 报错的地方）。"""
    n, j = 20, np.array([19, 0, 7], dtype=np.int64)
    lb = _labels(n)
    out = s2n.subset_labels(lb, j)

    assert out['_dropped'] == lb['_dropped'], \
        '_dropped 记的是 19×19 过滤丢了多少，与分片无关 ⇒ 刻意不动'
    assert out['_network'] == lb['_network']
    assert out['w']['note'] == lb['w']['note'], '非行轴的嵌套值必须原样带走'
    assert out['policy_meta'][0] == lb['policy_meta'][0]
    assert np.array_equal(out['policy_meta'][1], lb['policy_meta'][1]), \
        'tuple 里非行轴的那半必须原样带走'


def test_subset_labels_updates_kept_but_never_dropped():
    """``_kept`` 必须跟着子集走（下游 ``labels_to_chunk`` 的行数就是它）。"""
    lb = _labels(40)
    for j in (np.array([0, 39], dtype=np.int64),
              s2n.shard_mask(40, 0, 4, 1),
              np.zeros(0, dtype=np.int64)):
        out = s2n.subset_labels(lb, j)
        assert out['_kept'] == int(j.size)
        assert out['_dropped'] == lb['_dropped']
        assert set(out) == set(lb), '键集合不能变（少一个键 = 静默少一项 loss）'


def test_subset_labels_uses_the_kept_length_criterion():
    """统一判据是「第 0 轴长度 == ``_kept``」⇒ 长度恰好等于 ``_kept`` 的**非行轴**
    数组也会被切。

    这是**刻意**的取舍：宁可多切一个（那种键本来就没人按行取）也不能漏切一个真
    行轴键（漏切会让块内行数与 ``game_ids`` 不符）。``legacy_probs`` 就是诱饵。
    """
    n, j = 12, np.array([11, 1], dtype=np.int64)
    lb = _labels(n)
    out = s2n.subset_labels(lb, j)
    assert out['legacy_probs'].shape == (2, 1)
    assert np.array_equal(out['legacy_probs'], lb['legacy_probs'][j])


def test_subset_labels_on_empty_index_gives_zero_row_arrays():
    """空子集（``shard_mask`` 一行都没分到）：行轴键变 0 行，标量照旧。"""
    lb = _labels(50)
    j = s2n.shard_mask(50, 3, 5, 4)[:0]
    out = s2n.subset_labels(lb, j)
    assert out['_kept'] == 0
    for key in ROW_AXIS_KEYS:
        assert out[key].shape[0] == 0, f'{key} 在空子集上还有行'
    for key in ROW_AXIS_TUPLES:
        for half in out[key]:
            assert half.shape[0] == 0, f'{key} 的某一半在空子集上还有行'
    assert out['_dropped'] == lb['_dropped']
    assert out['_network'] == lb['_network']


def test_subset_labels_accepts_a_python_list():
    """``j`` 是 Python list 也要能用（``np.asarray`` 兜住了）。"""
    out = s2n.subset_labels(_labels(8), [7, 0])
    assert out['_kept'] == 2
    assert out['outcome'].tolist() == [7, 0]


def test_shard_mask_then_subset_labels_partitions_the_payload():
    """端到端：``shard_mask`` 给局部下标 → ``subset_labels`` 按它取子集。

    照 `convert` 的形状走一遍「成员 × 块」：每个成员有**自己的** labels，
    ``outcome`` 列装的是局部行号 ⇒ 每块的 ``_kept`` 必须与 ``shard_mask``
    的返回一致，而拼起来的全局行号必须恰好是全集、无重复无丢失。
    """
    member_rows, num_shards = [63, 41, 63, 7], 4
    base = 0
    payloads = []
    for n_local in member_rows:
        member_base = base
        base += n_local
        mlb = _labels(n_local)
        for shard_id in range(num_shards):
            j = s2n.shard_mask(n_local, member_base, num_shards, shard_id)
            sub = s2n.subset_labels(mlb, j)
            assert sub['_kept'] == j.size, 'subset_labels 与 shard_mask 对不上'
            assert sub['outcome'].tolist() == j.tolist(), '载荷必须是这一块的局部下标'
            assert np.array_equal(sub['policy_player_sparse'][0][:, 0], j)
            assert np.array_equal(sub['w']['scoring'], j.astype(np.float32))
            payloads.append(j + member_base)
    everything = np.concatenate(payloads)
    assert everything.size == sum(member_rows)
    assert sorted(everything.tolist()) == list(range(sum(member_rows))), \
        '各块拼起来不是全集（有重复或丢失）'


# --------------------------------------------------------------------------- #
# 6 · 第 0 遍（`count_kept_rows`）的分片侧
# --------------------------------------------------------------------------- #
#: ``(原始行数, 该成员里 19×19 的行数)``：刻意让「原始行数 != 保留行数」，
#: 这样「偏移按原始行数推进」这种错会立刻显形。
ARCHIVE_MEMBERS = [(40, 37), (17, 12), (63, 63), (20, 20)]
KEPT_TOTAL = 37 + 12 + 63 + 20          # 132
RAW_TOTAL = 40 + 17 + 63 + 20           # 140


def _mini_member(n_rows, n_19x19):
    """最小成员：第 0 遍只解 ``binaryInputNCHWPacked`` 与 ``globalTargetsNC``。

    ch0 的 on-board 掩码是**逐行**的（1 填 b×b 个 bit），
    ``board_size_from_packed`` 据此 sqrt 反推盘面尺寸 ⇒ 只有前 ``n_19x19``
    行做成 19 路（361 个 bit），其余行 0 个 bit（反推成 0 ⇒ 非 19 路）。
    """
    flat = np.zeros((n_rows, BOARD_STRIDE * BOARD_STRIDE), dtype=np.uint8)
    flat[:n_19x19] = 1
    packed = np.zeros((n_rows, SPATIAL_CHANNELS, PACKED_BYTES), dtype=np.uint8)
    packed[:, 0] = np.packbits(flat, axis=-1, bitorder='big')
    return {
        'binaryInputNCHWPacked': packed,
        'globalTargetsNC': np.zeros((n_rows, 80), dtype=np.float32),
    }


@pytest.fixture(scope='module')
def mini_archive(tmp_path_factory):
    """一个含 4 个成员（保留行数 37/12/63/20）的**未压缩** tar。"""
    path = str(tmp_path_factory.mktemp('shard') / 'mini.tar')
    with tarfile.open(path, 'w') as tf:
        for i, (raw, kept) in enumerate(ARCHIVE_MEMBERS):
            buf = io.BytesIO()
            np.savez(buf, **_mini_member(raw, kept))
            info = tarfile.TarInfo('%s-s11001M-d1/%02d.npz' % (NET_80, i))
            info.size = buf.tell()
            tf.addfile(info, io.BytesIO(buf.getvalue()))
    return path


def _expected_shard_counts(member_kept, num_shards, global_offset=0):
    """独立算一遍期望值：**不调 ``shard_mask``**。

    全局行号 = 保留行（19×19 过滤后）依次编号；归属 = ``r % num_shards``。
    拿实现验实现是自证 ⇒ 这里只用纯 Python 计数。
    """
    counts = [0] * num_shards
    base = int(global_offset)
    for kept in member_kept:
        for r in range(base, base + kept):
            counts[r % num_shards] += 1
        base += kept
    return counts


def _shard_counts(archive, num_shards, global_offset=0):
    return [s2n.count_kept_rows(archive, NET_80, num_shards=num_shards,
                                shard_id=s,
                                global_offset=global_offset)['rows_kept_shard']
            for s in range(num_shards)]


def test_count_pass_shards_sum_to_the_unsharded_total(mini_archive):
    """第 0 遍（``count_kept_rows``）：各块行数之和 == 未分片的总保留行数。"""
    whole = s2n.count_kept_rows(mini_archive, NET_80)
    assert whole['rows_raw'] == RAW_TOTAL, 'fixture 的原始行数不对'
    assert whole['rows_kept'] == KEPT_TOTAL, 'fixture 的保留行数不对'
    for num_shards in (2, 3, 4, 5, 7, 8):
        per = _shard_counts(mini_archive, num_shards)
        assert sum(per) == KEPT_TOTAL, \
            f'{num_shards} 块之和 {sum(per)} != 总保留 {KEPT_TOTAL}'
        q, rem = divmod(KEPT_TOTAL, num_shards)
        assert set(per) <= {q, q + 1}
        assert per.count(q + 1) == rem


@pytest.mark.parametrize('num_shards', [3, 7, 8])
def test_count_pass_offsets_advance_by_kept_rows(mini_archive, num_shards):
    """🔴 第 0 遍的偏移必须按**保留**行数推进（``member_base += n_keep``）。

    逐块行数与独立算出的期望值**逐个**相等 —— 这条能同时区分两种偏移推进：
    「按原始行数」（本fixture 里 140 ≠ 132）与「按分片过滤后行数」
    （`shard_mask` 的 docstring 点名的经典错）。``N=3/7/8`` 是三种错都显形的
    最小规模（``N=2`` 两种错碰巧同解）。
    """
    want = _expected_shard_counts([kept for _, kept in ARCHIVE_MEMBERS],
                                  num_shards)
    got = _shard_counts(mini_archive, num_shards)
    assert got == want, f'{num_shards} 块：期望 {want}，实得 {got}'


def test_count_pass_honours_global_offset(mini_archive):
    """``global_offset``（前一归档累计的保留行数）只旋转归属，不改总和。"""
    base = 10
    for num_shards in (3, 8):
        want = _expected_shard_counts([k for _, k in ARCHIVE_MEMBERS],
                                      num_shards, global_offset=base)
        got = _shard_counts(mini_archive, num_shards, global_offset=base)
        assert got == want, f'{num_shards} 块 + 偏移 {base}：期望 {want}，实得 {got}'
        assert sum(got) == KEPT_TOTAL


def test_cli_count_only_wires_shard_args_through(mini_archive, capsys):
    """CLI 也把 ``--num-shards/--shard-id`` 透传到第 0 遍（``--count-only``）。"""
    rc = s2n.main(['--count-only', '--source', '%s:%s' % (mini_archive, NET_80),
                   '--num-shards', '8', '--shard-id', '3'])
    out = capsys.readouterr().out
    assert rc == 0
    want = _expected_shard_counts([k for _, k in ARCHIVE_MEMBERS], 8)[3]
    assert '第 3/8 块 → %d 行' % want in out, out


def test_cli_rejects_out_of_range_shard_id(mini_archive, capsys):
    rc = s2n.main(['--count-only', '--source', '%s:%s' % (mini_archive, NET_80),
                   '--num-shards', '4', '--shard-id', '4'])
    assert rc == 2
    assert '越界' in capsys.readouterr().err