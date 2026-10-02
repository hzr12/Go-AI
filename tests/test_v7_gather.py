# -*- coding: utf-8 -*-
"""B6 · 邻行 gather：``boards[i±k]`` 的按批 mmap 随机访问 + 越界/跨局守卫。

被测对象 `src/data/feature_v7_gather.py`：
  · `gather_neighbors` —— 一次 fancy-index 取邻行盘面（+ `to_play` / `ko` 小列）
  · `NeighborGather`   —— 按 offset 索引的返回值（``g[offset]`` = 盘面）

运行：
    python -m pytest tests/test_v7_gather.py -q

----
**为什么这条命令不许被"优化"成逐行循环**
本模块唯一的性能契约是「每个偏移一次 gather」。主数据集的 `boards` 是
`34.2M × 361 B = 12.3 GB`，预取器每批要取 ``B × len(offsets)`` 行；逐行
Python 循环会把 2635 行/s 的吞吐目标（spec §5.2）直接打穿，而症状是
「训练变慢」而不是报错。
"""

import os
import time

import numpy as np
import pytest

from src.data.feature_v7_gather import (
    DEFAULT_OFFSETS,
    FUTUREPOS_OFFSETS,
    LADDER_OFFSETS,
    NeighborGather,
    gather_neighbors,
)

N = 19
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# 夹具：在临时目录里造一份**真的 `.npy`**（走 `open_memmap`，与生产一致）
# --------------------------------------------------------------------------- #
def make_board(row):
    """一行盘面，**任意两行都不同、且任何一行都不是全空盘**。

    两条性质各有用处：
      · **互不相同** ⇒ 「取错行」必然被抓到，不靠运气；
      · **不出现全 0** ⇒ 「不可用行填 0」这件事不会被全空盘行伪装过去。
        （首版整盘填 ``[-1,0,1][row % 3]``，第 200 行正好落在全空盘上，于是
        「越界行填 0」与「clamp 到最后一行」在张量上完全一样，断言直接失效。）

    行号编码在**盘面第 0 行**的 19 个点上（±1 二进制），其余点填
    ``[-1,0,1][row % 3]``（非 0 时保证不空）。
    """
    a = np.full((N, N), (-1, 0, 1)[row % 3], dtype=np.int8)
    a[0, :] = np.array([1 if (row >> b) & 1 else -1 for b in range(N)],
                       dtype=np.int8)
    return a


def build_dataset(tmp_path, n_rows, n_games=None, keys=('boards', 'to_play', 'ko',
                                                       'game_ids')):
    """落成 ``{key}.npy``，返回路径 dict。形状与主数据集的列完全一致。"""
    if n_games is None:
        n_games = max(1, n_rows // 10)
    gid = np.minimum(np.arange(n_rows) * n_games // n_rows, n_games - 1)
    gid = gid.astype(np.int32)
    cols = {
        'boards': np.stack([make_board(i) for i in range(n_rows)]),
        # to_play = +1 / -1 交替；ko = 行号（越界值另测）—— 都取自源行号，
        # 所以「offset k 的一列是否等于源第 i+k 行」是一句可断言的话。
        'to_play': (np.arange(n_rows) % 2 * 2 - 1).astype(np.int8),
        'ko': (np.arange(n_rows) % (N * N)).astype(np.int16),
        'game_ids': gid,
    }
    out = {}
    for k in keys:
        p = os.path.join(str(tmp_path), f'{k}.npy')
        arr = cols[k]
        mm = np.lib.format.open_memmap(p, mode='w+', dtype=arr.dtype,
                                       shape=arr.shape)
        mm[:] = arr
        mm.flush()
        del mm
        out[k] = mm_of(p)
    return out


def mm_of(path):
    """按需分页打开 —— **生产口径**（`np.load(..., mmap_mode='r')`）。"""
    return np.load(path, mmap_mode='r')


@pytest.fixture()
def ds(tmp_path):
    return build_dataset(tmp_path, n_rows=200)


# --------------------------------------------------------------------------- #
# 取对行
# --------------------------------------------------------------------------- #
def test_each_offset_reads_its_own_source_row(ds):
    """四个偏移各取各的行：``g[off][j] == boards[i[j] + off]``。

    逐个偏移单独断言（而不是「四个一起比」）—— 一次比完只会告诉我们「有一个
    偏移错了」，而逐个断言会指名道姓。偏移取自**具名常量**，顺带把
    「ladder 往后看、futurepos 往前看」这条契约钉住。
    """
    idx = np.array([50, 51, 100, 150], dtype=np.int64)
    off_all = LADDER_OFFSETS + FUTUREPOS_OFFSETS
    g = gather_neighbors(ds['boards'], idx, offsets=off_all,
                         game_ids=None)
    for off in off_all:
        got = g[off]
        assert got.shape == (idx.size, N, N)
        assert got.dtype == np.int8
        for j, i in enumerate(idx):
            assert np.array_equal(got[j], ds['boards'][i + off]), (
                f'offset {off:+d} 在行 i={i} 上取错了：拿到的是第 {i + off} 行，'
                f'但内容不等于它')
        assert g.valid[off].all(), f'offset {off:+d} 明明全部在界内'


def test_offset_is_a_signed_row_delta_and_the_documented_default_reads_forward(
        ds):
    """**offset 是带符号的行增量** ``j = i + offset``；默认 ``(1,8,32)`` 读**未来**。

    这条专门钉住任务书里被写成混着方向的那处（`offsets=(1,8,32)` 与「offset 1
    = i−1」同时出现）。默认元组保持不动、解释成 ``i+1/i+8/i+32``；
    ladder 通道用的是 :data:`LADDER_OFFSETS` = ``(-1,-2)``。
    """
    assert DEFAULT_OFFSETS == (1, 8, 32)
    idx = np.array([40, 41, 42], dtype=np.int64)
    g = gather_neighbors(ds['boards'], idx)          # 用签名默认值
    assert g.offsets == DEFAULT_OFFSETS
    for j, i in enumerate(idx):
        assert np.array_equal(g[1][j], ds['boards'][i + 1])
        assert np.array_equal(g[8][j], ds['boards'][i + 8])
        assert np.array_equal(g[32][j], ds['boards'][i + 32])
    # 方向搞反的话：i-1 与 i+1 的内容必然不同（make_board 保证任意两行不同）。
    assert not np.array_equal(ds['boards'][40], ds['boards'][41])
    back = gather_neighbors(ds['boards'], idx, offsets=LADDER_OFFSETS)
    assert np.array_equal(back[-1][0], ds['boards'][39])


def test_offset_one_carries_to_play_and_ko_from_the_same_row(ds):
    """**offset ±1 必须同时给出该行的 `to_play` 与 `ko`**（散列 / 合法性需要）。

    `src/data/pos_hash.py` 的散列口径是 ``pos_hash(board 361B, to_play 1B,
    ko 2B)`` —— 少了后两项，同一盘面在不同行棋方/劫点下会撞成一个散列，
    而症状是「join 结果为空」，看不出是数据取错了。
    """
    idx = np.array([77, 78, 120], dtype=np.int64)
    g = gather_neighbors(ds['boards'], idx, offsets=(-1, 1, 8),
                         game_ids=None, to_play=ds['to_play'], ko=ds['ko'])
    for off in (-1, 1, 8):
        assert g.to_play[off].shape == idx.shape
        assert g.ko[off].shape == idx.shape
        for j, i in enumerate(idx):
            assert g.to_play[off][j] == ds['to_play'][i + off]
            assert g.ko[off][j] == ds['ko'][i + off]
    # ko 的口径：紧凑扁平下标 r*n+c，-1 = 无（与 GoBoard.ko_point 同口径）。
    assert g.ko[-1][0] == ds['ko'][76] and ds['ko'][76] >= 0


def test_small_columns_are_none_when_not_requested(ds):
    """没传列 ⇒ 属性是 ``None``，**不是**全 ``-1``。

    全 ``-1`` 与「无劫」「无行棋方」无法区分 —— 这正是 `pos_hash.py` 的
    docstring 警告过的那类假信号。
    """
    g = gather_neighbors(ds['boards'], np.array([10, 11]), offsets=(-1,))
    assert g.to_play is None
    assert g.ko is None
    with pytest.raises(TypeError):
        g.to_play[-1][0]


# --------------------------------------------------------------------------- #
# 越界守卫
# --------------------------------------------------------------------------- #
def test_row_zero_backward_offset_is_unusable_and_does_not_borrow_another_board(
        ds):
    """``i-1 < 0`` ⇒ 标不可用；**不能**静默 clamp 到别处、也不能借最后一行的盘面。

    两种「看起来更友好」的做法都是错的：clamp 会把别的 ply 的盘面当成本行数据，
    而它**看起来完全合法**（它就是某个真实盘面），只是不属于这一手。
    """
    idx = np.array([0, 1, 5], dtype=np.int64)
    g = gather_neighbors(ds['boards'], idx, offsets=(-1,),
                         game_ids=None, to_play=ds['to_play'], ko=ds['ko'])
    assert g.valid[-1].tolist() == [False, True, True]
    assert not g.valid[-1][0]
    # 既不是源 boards[0] 自己，也不是「clamp 后」的 boards[-1]（= 最后一行）
    assert not g.boards[-1][0].any(), '越界行必须填 0，不能填任何真实盘面'
    assert not np.array_equal(g.boards[-1][0], ds['boards'][-1])
    assert not np.array_equal(g.boards[-1][0], ds['boards'][0])
    assert g.to_play[-1][0] == 0 and g.ko[-1][0] == 0
    # 可用行不受影响
    assert np.array_equal(g.boards[-1][1], ds['boards'][0])


def test_row_past_the_end_forward_offset_is_unusable(tmp_path):
    """正向偏移越过末行同样标不可用（futurepos 的 ``i+32`` 靠近终局时必然发生）。"""
    d = build_dataset(tmp_path, n_rows=40)
    idx = np.array([8, 35, 39], dtype=np.int64)
    g = gather_neighbors(d['boards'], idx, offsets=FUTUREPOS_OFFSETS)
    assert g.valid[8].tolist() == [True, False, False]      # 16 / 43 / 47
    assert g.valid[32].tolist() == [False, False, False]     # 40 / 67 / 71
    assert g.boards[32][0].sum() == 0
    assert np.array_equal(g.boards[8][0], d['boards'][16])   # 8 + 8


# --------------------------------------------------------------------------- #
# 跨局守卫
# --------------------------------------------------------------------------- #
def test_cross_game_row_is_marked_unusable(tmp_path):
    """``game_ids[i+k] != game_ids[i]`` ⇒ 标不可用，**不得**取到下一局的盘面。

    主数据集的 `boards` 是把 162,298 局首尾相接排成的一根大数组、**没有局的
    边界标记** ⇒ 行号 ``i`` 与 ``i+1`` 完全可以分属两局。这与 P0 的
    「id 复用 bug」是同一类风险：取到的数据**看起来完全合法**。
    """
    d = build_dataset(tmp_path, n_rows=100, n_games=10)   # 每 10 行一局
    gid = np.asarray(d['game_ids'])
    assert gid[9] != gid[10] and gid[49] != gid[50]
    idx = np.array([8, 9, 10, 11], dtype=np.int64)
    g = gather_neighbors(d['boards'], idx, offsets=(1,),
                         game_ids=d['game_ids'])
    # 只有 i=9（局末）跨界；i=10 是**下一局的第一行**，它往后看仍在同一局里
    assert g.valid[1].tolist() == [True, False, True, True]
    assert not g.boards[1][1].any(), '跨局行必须填 0'
    assert np.array_equal(g.boards[1][2], d['boards'][11])
    assert np.array_equal(g.boards[1][3], d['boards'][12])


def test_without_game_ids_the_cross_game_guard_is_off(tmp_path):
    """不传 `game_ids` 就不启用跨局守卫 —— 这条把「守卫是可选的」钉成事实，
    同时提醒调用方：**主数据集上必须传**，否则上面那条测试里的跨局行会被取到。"""
    d = build_dataset(tmp_path, n_rows=100, n_games=10)
    idx = np.array([9], dtype=np.int64)
    guarded = gather_neighbors(d['boards'], idx, offsets=(1,),
                               game_ids=d['game_ids'])
    unguarded = gather_neighbors(d['boards'], idx, offsets=(1,))
    assert not guarded.valid[1][0]
    assert unguarded.valid[1][0]
    assert np.array_equal(unguarded[1][0], d['boards'][10])


def test_both_guards_combine_and_usable_rows_survive(tmp_path):
    """越界与跨局同时存在时，两者都不能放过；且**可用行一个都不能被牵连**。"""
    d = build_dataset(tmp_path, n_rows=100, n_games=10)
    idx = np.array([0, 5, 9, 10, 50, 99], dtype=np.int64)
    g = gather_neighbors(d['boards'], idx, offsets=LADDER_OFFSETS,
                         game_ids=d['game_ids'], to_play=d['to_play'],
                         ko=d['ko'])
    # 每 10 行一局 ⇒ 边界在 i=10/20/…/90。局首的 i-1 既是越界（i=0）又是跨局。
    assert g.valid[-1].tolist() == [False, True, True, False, False, True]
    assert g.valid[-2].tolist() == [False, True, True, False, False, True]
    for j, i in enumerate(idx):
        for off in LADDER_OFFSETS:
            if not g.valid[off][j]:
                assert g.boards[off][j].sum() == 0
            else:
                assert np.array_equal(g.boards[off][j], d['boards'][i + off])
                assert g.to_play[off][j] == d['to_play'][i + off]


# --------------------------------------------------------------------------- #
# API 形状与守卫的报错
# --------------------------------------------------------------------------- #
def test_mapping_interface_yields_the_board_arrays():
    """``NeighborGather`` 就是 ``dict[int, ndarray]``（任务书钦定的返回类型）。"""
    g = NeighborGather({-1: np.zeros((2, N, N), np.int8)},
                       {-1: np.array([True, False])})
    assert isinstance(g, dict) is False and len(g) == 1
    assert g[-1].shape == (2, N, N)
    assert list(g) == [-1]
    assert dict(g)[-1] is g[-1]
    assert g.all_valid(-1) is False
    assert NeighborGather({-1: np.zeros((2, N, N), np.int8)},
                          {-1: np.array([True, True])}).all_valid(-1) is True


def test_missing_valid_mask_is_rejected():
    with pytest.raises(ValueError, match='valid'):
        NeighborGather({-1: np.zeros((2, N, N), np.int8)}, {})


def test_offset_zero_is_rejected(ds):
    """``offset 0`` 会让每个偏移都等于「自己」—— gather 存在的意义就没了。"""
    with pytest.raises(ValueError, match='offset 0'):
        gather_neighbors(ds['boards'], np.array([5]), offsets=(0,))


def test_out_of_range_indices_are_rejected_not_silently_dropped(ds):
    """**不为越界的 `indices` 兜底**：那是调用方取错了行号，静默丢行会让样本数
    与 batch 形状对不上。"""
    with pytest.raises(ValueError, match='越界'):
        gather_neighbors(ds['boards'], np.array([5, 10 ** 6]), offsets=(1,))


def test_npz_source_is_rejected_with_the_materialize_hint(tmp_path):
    """``.npz``（zip 容器）**必须报错**，而不是静默退化成「整份解压进内存」。

    `np.load('x.npz', mmap_mode='r')` 不报错，但访问 `.z[k]` 会把整个成员解压：
    `boards` 是 12.3 GB 而本机 13.9 GB ⇒ 任何一次 `z['boards'][s:e]` 先吃 12.3 GB。
    正确做法是 `kata_label_join.materialize_dataset()` 落成 `.npy`。
    """
    p = os.path.join(str(tmp_path), 'x.npz')
    np.savez(p, boards=np.zeros((4, N, N), np.int8))
    z = np.load(p, allow_pickle=False)
    try:
        with pytest.raises(TypeError, match='materialize_dataset'):
            gather_neighbors(z, np.array([0, 1]), offsets=(1,))
    finally:
        z.close()


def test_column_row_count_mismatch_is_rejected(tmp_path):
    """列与盘面必须来自同一份数据集 —— 行数对不上说明拿错了源。"""
    d = build_dataset(tmp_path, n_rows=20)
    bad = np.zeros(19, dtype=np.int32)
    with pytest.raises(ValueError, match='行数'):
        gather_neighbors(d['boards'], np.array([1, 2]), offsets=(1,),
                         game_ids=bad)


def test_duplicate_and_unsorted_indices_stay_row_aligned(tmp_path):
    """预取器按 shuffle 取行 ⇒ **重复 + 无序**都很常见。返回必须逐行对齐 indices。"""
    d = build_dataset(tmp_path, n_rows=60)
    idx = np.array([40, 3, 40, 0, 59, 3], dtype=np.int64)
    g = gather_neighbors(d['boards'], idx, offsets=(-1, 1),
                         game_ids=d['game_ids'], to_play=d['to_play'])
    for j, i in enumerate(idx):
        if g.valid[-1][j]:
            assert np.array_equal(g.boards[-1][j], d['boards'][i - 1])
        else:
            assert g.boards[-1][j].sum() == 0
        if g.valid[1][j]:
            assert np.array_equal(g.boards[1][j], d['boards'][i + 1])
            assert g.to_play[1][j] == d['to_play'][i + 1]


def test_empty_batch_returns_well_formed_empty_arrays(ds):
    """空批不炸，且形状/dtype 与非空时一致 —— 下游不必特判。"""
    g = gather_neighbors(ds['boards'], np.zeros(0, np.int64), offsets=(1,),
                         to_play=ds['to_play'])
    assert g.boards[1].shape == (0, N, N)
    assert g.valid[1].shape == (0,)
    assert g.to_play[1].shape == (0,)


# --------------------------------------------------------------------------- #
# 批量契约
# --------------------------------------------------------------------------- #
def test_gather_is_one_fancy_index_per_offset_not_a_python_loop(tmp_path):
    """**每个偏移一次 gather**，不随 ``B`` 增长地逐行循环。

    上界故意放得很宽（这台机器上实测是毫秒级，界限给到秒级）：这条抓的不是
    「快」，是「**没有退化成逐行**」—— 退化的症状是慢到某个数量级，而阈值再
    收紧也抓不住「慢 3 倍」那种退化。与 `test_v7_planes.py` 里那条同构。
    """
    rows = 20_000
    d = build_dataset(tmp_path, n_rows=rows)
    rng = np.random.default_rng(0)
    idx = rng.integers(200, rows - 200, size=4_000).astype(np.int64)
    offs = LADDER_OFFSETS + FUTUREPOS_OFFSETS

    t0 = time.perf_counter()
    g = gather_neighbors(d['boards'], idx, offsets=offs, game_ids=d['game_ids'])
    dt = time.perf_counter() - t0

    assert g.boards[8].shape == (4_000, N, N)
    assert dt < 5.0, (
        f'4000 行 × 4 个偏移用了 {dt:.2f}s。这条路径必须是批量 fancy-index；'
        f'退化成逐行 Python 循环会把 spec §5.2 的 2635 行/s 目标打穿。')


def test_gather_reads_only_the_touched_rows_of_a_real_mmap(tmp_path):
    """确认走的是**按需分页**：两次 gather 的读集合之并 ⊆ 触碰的行数 + 少量页。

    这条不测速度、测**访问模式** —— 如果哪天有人图省事改成
    ``np.load('x.npz')['boards'][a:b]``（整份解压 12.3 GB），这里会先炸。
    """
    rows = 4_000
    d = build_dataset(tmp_path, n_rows=rows, n_games=100)   # 每 40 行一局
    boards = d['boards']
    assert isinstance(boards, np.memmap), '夹具前提：必须是 memmap'

    # 整段落在**同一局**里（1005..1024 ⇒ 第 25 局，边界在 1000 与 1040），
    # 所以 i−1 / i+8 全部可用，「读到的行集合」可以与「应该读到的行集合」相等。
    idx = np.arange(1_005, 1_025, dtype=np.int64)
    g = gather_neighbors(boards, idx, offsets=(-1, 8), game_ids=d['game_ids'])
    assert g.valid[-1].all() and g.valid[8].all()
    want = {(int(i) - 1) for i in idx} | {(int(i) + 8) for i in idx}
    got = set()
    for off in (-1, 8):
        for j in np.flatnonzero(g.valid[off]):
            got.add(int(idx[j] + off))
    assert got == want, (
        f'读到的行与应读的行不一致：多读 {sorted(got - want)} / '
        f'漏读 {sorted(want - got)}')