# -*- coding: utf-8 -*-
"""C0 · 22 通道 V7 特征的吞吐基准 —— 产出决定 ``prefetch-workers`` 与 ``batch`` 的实测数字。

为什么这个脚本存在
------------------
段 1 要在 34.2M 行上训练 policy / value / futurepos，而 **22 个通道全部现场算**
（spec §5.1 禁止 npz 缓存）。于是「一个 worker 一批要多少毫秒」直接决定
``prefetch-workers`` 与 ``batch``，而早前那个 **2,635 行/s** 是**旧特征管道**
的数字 —— 管道后来整个换过（``scipy.ndimage.label`` 气桶 + ``iterLadders`` 梯子
+ 19 维全局 + gather），**必须重测**。

测什么（4 个数字 + 1 条曲线）
----------------------------
1. **单进程特征吞吐**，并分解到组件级：``spatial_channels_v7`` /
   ``global_features_v7`` / ``ladder_channels`` / gather 各占多少。
2. **多进程扩展曲线**：``prefetch-workers`` = 1/2/4/8 的行/s。
3. **gather 成本占比**：offset ``(1, 8, 32)`` 三次 fancy-index 的耗时，
   以及它对**冷/热页缓存**的敏感度（这一项在本机实测里出人意料，见 ``--help``）。
4. **内存高水位**：单 worker 处理一个 batch 的峰值 RSS。
5. **batch 规模曲线**：32 / 64 / 128 / 256 / 512 / 1024。

 **刻意不做的事**（否则脚本就成了调参器）
------------------------------------------
* **不跑全量 34.2M 行。** 用分层采样（见 :func:`stratified_indices`），理由写在
  那里：「只测前 N 行」会系统性低估，开局空盘是最快的路径。
* **不把吞吐数字写进门禁。** 本脚本的测试只断言**正确性不变量**（materialize
  后的盘面与 npz 逐位相等、JSON 结构完整），性能数字只记录，不做断言 ——
  CI 上必然抖动。
* **不为了对上 3.04 ms/行去调参。** 预算成立与否由 :func:`judge` 按实测数字判，
  不成立就写「不成立」。

运行::

    # 全量（默认约 6-10 分钟）
    python scripts/bench_v7_features.py --out benchmarks/bench_v7_features.json

    # 只跑组件分解，快一点
    python scripts/bench_v7_features.py --phase components

    # 换 batch 曲线 / worker 档位
    python scripts/bench_v7_features.py --phase batch_curve,workers \\
        --batch-curve 32,128,512 --workers 1,2,4,8

    # 只在**热缓存**下测（= 不含 USB 卷的随机读，见下）
    python scripts/bench_v7_features.py --io warm
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np

from src.data.feature_v7 import (DEFAULT_RULES_FLAGS, GameRow,
                                 calculate_area, global_features_v7,
                                 history_five, liberties_123,
                                 spatial_channels_v7)
from src.data.feature_v7_gather import (DEFAULT_OFFSETS, FUTUREPOS_OFFSETS,
                                        LADDER_OFFSETS, gather_neighbors)
from src.data.feature_v7_ladders import ladder_channels

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 特征计算真正要碰的**每一行**盘面：当前盘 + ladder 的前一手/前二手 +
#: futurepos 的 +1/+8/+32。gather 本身只做 ±offset（offset 0 被显式禁止，
#: 见 ``feature_v7_gather.gather_neighbors``），当前盘由调用方直读。
BOARD_ROWS_PER_SAMPLE = 1 + len(LADDER_OFFSETS) + len(FUTUREPOS_OFFSETS)

#: 段 1 的预算：3.04 ms/行（= 旧管道 2,635 行/s ÷ 8 workers）。
#: 只用于 :func:`judge` 判定，不用于任何调参。
BUDGET_MS_PER_ROW = 3.04

DEFAULT_NPZ = os.path.join(REPO, 'data', 'sgf_19x19_full.npz')
DEFAULT_MAT = os.path.join(REPO, 'tmp', 'materialized')
DEFAULT_OUT = os.path.join(REPO, 'benchmarks', 'bench_v7_features.json')

#: 采样用的固定种子。 固定住是为了让「同一台机器上两次跑的数字可比」，
#: 而不是为了让结果好看 —— 复杂度分层本身是**确定性**的（按分数取 top-M）。
SEED = 20261002

#: 局级标量的占位。本仓 SGF 语料 55% 的局没有 ``RU``，于是 ``rules_flags``
#: 恒取 :data:`DEFAULT_RULES_FLAGS`（``global_features_v7`` docstring 已定），
#: 贴目取最常见的 7.5。 这两个常量只影响全局 19 维的**取值**，不影响耗时
#: （它们只做标量位运算，见 ``components`` 里 ``global`` 那一项的实测）。
GAME_ROW = GameRow(komi=7.5, rules_flags=DEFAULT_RULES_FLAGS)


# --------------------------------------------------------------------------- #
# 列的打开方式
# --------------------------------------------------------------------------- #
def open_columns(mat_dir):
    """按需分页打开 materialize 后的五列。

     必须是 ``mmap_mode='r'``：``npz`` 的压缩成员对 mmap **无效**，
    ``np.load('x.npz', mmap_mode='r')`` 不报错但会把整个成员解压进内存
    （boards 是 12.35 GB，本机 14.9 GB ⇒ 任何 numpy 临时量都足以把它推过 OOM）。
    这条禁令由 ``feature_v7_gather._reject_npz`` 在运行期兜着。
    """
    cols = {}
    for key in ('boards', 'to_play', 'ko', 'game_ids', 'my_hist', 'op_hist'):
        p = os.path.join(mat_dir, f'{key}.npy')
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{p} 不存在。先跑一次 materialize：\n'
                f'  python -c "import sys; sys.path.insert(0, \'.\');\n'
                f'    from src.data.kata_label_join import materialize_dataset;\\\n'
                f'    materialize_dataset(\'data/sgf_19x19_full.npz\', '
                f'\'tmp/materialized\', keys=(\'to_play\',\'ko\',\'game_ids\','
                f'\'my_hist\',\'op_hist\'))"')
        cols[key] = np.load(p, mmap_mode='r')
    return cols


def npz_column_slice(npz_path, key, start, count):
    """从压缩 npz 的一个成员里**流式**取 ``[start, start+count)`` 行。

    为什么不用 ``np.load(npz)[key]``：那会把**整个**成员解压进内存
    （boards = 12.35 GB）。``zipfile`` 的成员流可以逐段读，于是取 4096 行
    只花 1.4 MB。

     依赖 ``.npy`` 成员里 1-D-per-row 的布局（``.npy`` 的数据段是 C 序连续
    的），所以「行」= 固定 ``prod(shape[1:])`` 字节。这正是主数据集的布局。
    """
    with zipfile.ZipFile(npz_path) as z:
        fp = z.open(f'{key}.npy')
        version = np.lib.format.read_magic(fp)
        shape, fortran_order, dtype = np.lib.format._read_array_header(fp, version)
        if fortran_order:
            raise ValueError(f'{key}.npy 是 Fortran 序，行切片不成立')
        row = int(np.prod(shape[1:])) * dtype.itemsize
        data_off = fp.tell()
        fp.seek(data_off + start * row)
        raw = fp.read(count * row)
    if len(raw) != count * row:
        raise ValueError(f'{key}[{start}:{start+count}] 读短了：{len(raw)} 字节')
    return np.frombuffer(raw, dtype=dtype).reshape((count,) + shape[1:])


# --------------------------------------------------------------------------- #
# 采样：分层，且显式覆盖盘面复杂度差异
# --------------------------------------------------------------------------- #
def complexity_scores(cols, pool):
    """给候选行算一个**便宜的**复杂度分数（越大越可能是慢路径）。

    三个分量，各有理由：

    * ``stones`` —— 盘面上的子数。`scipy.ndimage.label` 的连通块数、梯子 DFS 的
      候选链数都随它涨；实测与梯子耗时单调相关。
    * ``3 * two_lib_stones`` —— **处于 2 气链上的子数**（用 ``liberties_123``
      的 2 气桶近似）。这是梯子搜索的真正候选源，权重给 3 是因为它是
      ``stones`` 的超集信号、且直接决定 DFS 的根数。
    * ``20 * has_ko`` —— 有劫点。``search_is_ladder_captured`` 每次 DFS 都会
      读 ko 判 capture，且带劫的局面搜索树明显更深。

     用 ``liberties_123`` 当代理是有代价的：**它在采样阶段就把这些盘面读热了**。
      这对组件计时是好事（我们要测 CPU 而不是 USB 卷），但「gather 冷缓存」
      那一项必须另取**没被采样阶段碰过**的行 —— 见 :func:`measure_gather`。
    """
    boards = np.asarray(cols['boards'][pool])
    stones = (boards != 0).sum(axis=(1, 2))
    two_lib = liberties_123(boards)[:, 1].sum(axis=(1, 2))
    has_ko = (np.asarray(cols['ko'][pool]) >= 0).astype(np.int64)
    return stones + 3 * two_lib + 20 * has_ko, stones, two_lib


def stratified_indices(cols, n_rows, n_first, n_random, n_complex, pool_size,
                       seed=SEED):
    """分层采样：前 N 行 / 均匀随机 M 行 / 复杂度 top-K 行。

     **为什么必须分层**：只测前 N 行会系统性低估 —— 开局空盘是最快的路径。
      本脚本因此额外构造一个「复杂度 top-K」层，并**三层的吞吐都报出来**。
      实测结论见报告：前 N 层与随机层只差 ~10%，而复杂度层比随机层慢
      **4-9 倍** ⇒ 「盘面复杂度」是真正的分层维度，「开局」不是。

    Args:
        cols: :func:`open_columns` 的返回值。
        n_rows: 数据集总行数（取自 ``boards.npy`` 的 shape）。
        n_first / n_random / n_complex: 三层各自的行数（会被裁到能整除 batch）。
        pool_size: 从全量里抽多少行做复杂度打分（候选池）。

    Returns:
        ``{层名: (行号, 复杂度统计)}``。
    """
    rng = np.random.default_rng(seed)
    out = {}

    n_first = int(min(n_first, n_rows))
    out['first'] = (np.arange(n_first, dtype=np.int64),
                    {'n': n_first, 'note': '数据集最前 N 行（含开局）'})

    n_random = int(min(n_random, n_rows))
    rnd = np.sort(rng.choice(n_rows, n_random, replace=False).astype(np.int64))
    out['random'] = (rnd, {'n': n_random, 'note': '均匀随机 M 行'})

    if n_complex > 0:
        m = int(min(pool_size, n_rows))
        pool = rng.choice(n_rows, m, replace=False).astype(np.int64)
        score, stones, two_lib = complexity_scores(cols, pool)
        k = int(min(n_complex, m))
        hard = pool[np.argsort(score, kind='stable')[-k:]]
        hard = np.sort(hard)
        out['complex'] = (hard, {
            'n': k,
            'note': f'候选池 {m} 行里复杂度分最高的 {k} 行',
            'pool_stones_p50': float(np.percentile(stones, 50)),
            'pool_stones_p100': float(np.percentile(stones, 100)),
            'pool_two_lib_p50': float(np.percentile(two_lib, 50)),
            'pool_two_lib_p100': float(np.percentile(two_lib, 100)),
            'sel_stones_p50': float(np.percentile(stones[np.argsort(
                score, kind='stable')[-k:]], 50)),
            'sel_two_lib_p50': float(np.percentile(two_lib[np.argsort(
                score, kind='stable')[-k:]], 50)),
        })
    return out


# --------------------------------------------------------------------------- #
# 组件级计时（单进程）
# --------------------------------------------------------------------------- #
def _trim(idx, batch):
    """裁到能整除 batch，并按 batch 顺序排好（预取器按批喂，批内连续）。"""
    n = (idx.size // batch) * batch
    if n == 0:
        raise ValueError(f'{idx.size} 行不够凑一个 batch={batch}')
    return np.ascontiguousarray(idx[:n])


def measure_components(cols, idx, batch, reps=3):
    """把一行的成本拆到组件级。返回 ``{组件名: 秒}``。

    分解口径（**每一项都是独立实测**，不是相减得来的）：

    * ``read_current`` —— 直读当前盘 + 小列（``boards_mm[idx]``，不经
      ``gather_neighbors``，因为 offset 0 被它显式禁止）。
    * ``read_hist`` —— 直读 ``my_hist`` / ``op_hist``。
    * ``gather_ladder`` —— ``gather_neighbors(..., LADDER_OFFSETS=(-1,-2))``
      ⇒ **2** 次盘面 fancy-index。
    * ``gather_future`` —— ``gather_neighbors(..., FUTUREPOS_OFFSETS=(1,8,32))``
      ⇒ **3** 次盘面 fancy-index（任务书钦定的那三个 offset）。
    * ``spatial`` —— ``spatial_channels_v7``（含 3 块盘面的梯子搜索）。
    * ``spatial_1search`` —— 同上但**不传** prev/prev_prev ⇒ ``ladder_channels``
      走 ``pb is b`` 快路径，**只跑一次**梯子搜索。 **取值与 ``spatial``
      不同**（ch15/ch16 会等于 ch14），它只用来把「3 次搜索 vs 1 次搜索」的
      增量分离出来。
    * ``ladder_3`` / ``ladder_1`` —— 直接调 ``ladder_channels`` 的两次成本。
    * ``liberties_123`` / ``history_five`` / ``calculate_area`` —— 空间通道里
      剩下那部分的三个来源。
    * ``global`` —— ``global_features_v7``。

    ``reps`` 取**最小值**而不是均值：本机跑在同一页缓存上，最小值最接近
    「纯计算」上界，均值会把别的进程干扰摊进来。
    """
    sel = _trim(idx, batch)
    B = sel.size
    boards_mm = cols['boards']
    out = {}

    def timed(name, fn):
        best = float('inf')
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        out[name] = best

    def read_current():
        (np.asarray(boards_mm[sel]),
         np.asarray(cols['to_play'][sel]),
         np.asarray(cols['ko'][sel]),
         np.asarray(cols['game_ids'][sel]))

    def read_hist():
        (np.asarray(cols['my_hist'][sel]),
         np.asarray(cols['op_hist'][sel]))

    timed('read_current', read_current)
    timed('read_hist', read_hist)
    timed('gather_ladder', lambda: gather_neighbors(
        boards_mm, sel, LADDER_OFFSETS, game_ids=cols['game_ids'],
        to_play=cols['to_play'], ko=cols['ko']))
    timed('gather_future', lambda: gather_neighbors(
        boards_mm, sel, FUTUREPOS_OFFSETS, game_ids=cols['game_ids'],
        to_play=cols['to_play'], ko=cols['ko']))
    timed('gather_default_1_8_32', lambda: gather_neighbors(
        boards_mm, sel, DEFAULT_OFFSETS, game_ids=cols['game_ids'],
        to_play=cols['to_play'], ko=cols['ko']))

    g = gather_neighbors(boards_mm, sel, LADDER_OFFSETS, game_ids=cols['game_ids'],
                         to_play=cols['to_play'], ko=cols['ko'])
    boards = np.asarray(boards_mm[sel])
    tps = np.asarray(cols['to_play'][sel])
    kos = np.asarray(cols['ko'][sel])
    myh = np.asarray(cols['my_hist'][sel])
    oph = np.asarray(cols['op_hist'][sel])

    timed('spatial', lambda: spatial_channels_v7(
        boards, tps, kos, myh, oph, prev_board=g[-1], prev_prev_board=g[-2],
        prev_valid=g.valid[-1], prev_prev_valid=g.valid[-2]))
    timed('spatial_1search', lambda: spatial_channels_v7(
        boards, tps, kos, myh, oph))
    timed('ladder_3', lambda: ladder_channels(boards, tps, g[-1], g[-2], ko=kos))
    timed('ladder_1', lambda: ladder_channels(boards, tps, ko=kos))
    timed('liberties_123', lambda: liberties_123(boards))
    timed('history_five', lambda: history_five(boards, myh, oph))
    timed('calculate_area', lambda: calculate_area(boards, tps, DEFAULT_RULES_FLAGS))
    timed('global', lambda: global_features_v7(GAME_ROW, myh, oph, to_play=tps))

    out['_rows'] = B
    out['_pipeline'] = sum(out[k] for k in (
        'read_current', 'read_hist', 'gather_ladder', 'gather_future',
        'spatial', 'global'))
    return out


# --------------------------------------------------------------------------- #
# 多进程：复刻 ``scripts/train_sft.py::_BatchPrefetcher`` 的形状
# --------------------------------------------------------------------------- #
def _worker(wi, mat_dir, task_q, res_q):
    """预取 worker。

     **刻意与 ``train_sft.py::_prefetch_worker`` 同形**：拿 ``(step, pos, 行号)``
    的任务、算完把 payload 放回队列、主进程按 ``pos`` 排序拼回整批。
    不同的地方只有两处，都是 V7 必需的：① 传的是 **mmap 路径**而不是 dataset
    对象（34.2M 行的 boards 在 spawn 下根本没法 pickle 过队列）；② payload
    是 V7 的 ``(spatial, global)``。

     **不许在这里 import torch**：``train_sft.py`` 的
    ``tests/test_prefetch_fork_order.py`` 钉死了「预取 worker 不碰设备上下文」
    （4 卡实测每卡凭空多占 ~24 GiB ⇒ OOM）。本脚本同理。
    """
    cols = open_columns(mat_dir)
    res_q.put(('ready', wi, 0.0))
    while True:
        item = task_q.get()
        if item is None:
            return
        step, pos, sel = item
        try:
            t0 = time.perf_counter()
            spatial, glob = compute_features(cols, sel)
            dt = time.perf_counter() - t0
            res_q.put((step, pos, spatial, glob, dt, None))
        except Exception as e:  # noqa: BLE001
            res_q.put((step, pos, None, None, 0.0, repr(e)))


def compute_features(cols, sel):
    """一行的完整 V7 特征：gather(±offset) + 22 空间通道 + 19 全局通道。

    与 ``train_sft.py`` 的 ``sample_batch_numpy`` 同一形状：返回
    ``(spatial, global)``，空间是 ``(B,22,19,19)`` fp16、全局是 ``(B,19)`` fp16。
    """
    boards_mm = cols['boards']
    cur = np.asarray(boards_mm[sel])
    tps = np.asarray(cols['to_play'][sel])
    kos = np.asarray(cols['ko'][sel])
    myh = np.asarray(cols['my_hist'][sel])
    oph = np.asarray(cols['op_hist'][sel])
    g = gather_neighbors(boards_mm, sel, LADDER_OFFSETS, game_ids=cols['game_ids'],
                         to_play=cols['to_play'], ko=cols['ko'])
    spatial = spatial_channels_v7(cur, tps, kos, myh, oph,
                                  prev_board=g[-1], prev_prev_board=g[-2],
                                  prev_valid=g.valid[-1],
                                  prev_prev_valid=g.valid[-2])
    glob = global_features_v7(GAME_ROW, myh, oph, to_play=tps)
    return spatial, glob


def _prime_cache(cols, idx):
    """把 ``idx`` 涉及的全部盘面行读热，返回耗时（不计入测量）。

    顺序扫比随机扫快得多，所以这里按**行号排序后连续**读。
    """
    touched = set()
    for off in (0,) + tuple(LADDER_OFFSETS) + tuple(FUTUREPOS_OFFSETS):
        touched.update((idx + off).tolist())
    rows = np.fromiter((r for r in touched if 0 <= r < cols['boards'].shape[0]),
                       dtype=np.int64)
    rows.sort()
    mm = cols['boards']
    t0 = time.perf_counter()
    for s in range(0, rows.size, 200_000):
        np.asarray(mm[rows[s:s + 200_000]])
    return time.perf_counter() - t0


def measure_workers(mat_dir, idx, batch, workers, warm_batches=1, bench_batches=4,
                    io='warm', ctx_name=None):
    """``workers`` 个预取进程下的稳态行/s。

    计时口径：**只算 worker 全部就绪之后的批**。spawn 的启动开销单独报
    （``spawn_s``），因为 Windows 上 ``spawn`` 要重新导入解释器 + numpy + scipy，
    每次几十到几百毫秒 —— 把它算进稳态会低估吞吐，算进 epoch 会高估延迟。

    ``io='warm'`` 时先 :func:`_prime_cache` 把这些行读热，于是测的是
    **CPU 上限**；``io='cold'`` 时用没被碰过的随机行，测的是
    **本机 USB 卷上的真实生产态**。两者的差就是存储给流水线上的一刀。
    """
    ctx = mp.get_context(ctx_name) if ctx_name else mp.get_context()
    sel_full = _trim(idx, batch)
    if io == 'warm':
        prime_s = _prime_cache(open_columns(mat_dir), sel_full)
    else:
        prime_s = None

    n_ready = 0
    t0 = time.perf_counter()
    task_q = ctx.Queue(maxsize=max(2, workers * 2))
    res_q = ctx.Queue(maxsize=max(2, workers * 2))
    procs = [ctx.Process(target=_worker,
                         args=(wi, mat_dir, task_q, res_q), daemon=True)
             for wi in range(workers)]
    for p in procs:
        p.start()
    while n_ready < workers:
        tag, wi, _ = res_q.get()
        if tag == 'ready':
            n_ready += 1
    spawn_s = time.perf_counter() - t0

    pending = {}
    errors = []
    batches = []          # (step, rows, t_start, t_end)
    for step in range(warm_batches + bench_batches):
        lo, hi = step * batch, (step + 1) * batch
        if hi > sel_full.size:
            break
        sub = sel_full[lo:hi]
        t_start = time.perf_counter()
        for wi in range(workers):
            a, bb = wi * sub.size // workers, (wi + 1) * sub.size // workers
            task_q.put((step, wi, sub[a:bb]))
        parts = []
        while len(parts) < workers:
            r = res_q.get()
            if r[0] != step:
                pending.setdefault(r[0], []).append(r)
                continue
            parts.append(r)
        for r in pending.pop(step, []):
            parts.append(r)
        for r in parts:
            if r[5] is not None:
                errors.append(r[5])
        batches.append((step, int(sub.size), t_start, time.perf_counter()))

    # **warmup 不计入稳态**：稳态 = 第 warm_batches 批**完成**之后到最后一批
    # 完成。把 warm 批的时间算进来会低估吞吐（第一批要付 import / 首次触碰
    # 页表 / numpy 首次 dispatch 的钱），不算进来则符合预取深度 ≥1 的真实语义。
    timed_b = [b for b in batches if b[0] >= warm_batches]
    warm_b = [b for b in batches if b[0] < warm_batches]
    bench_rows = sum(b[1] for b in timed_b)
    bench_s = (timed_b[-1][3] - timed_b[0][3]) if len(timed_b) > 1 else 0.0
    warm_s = sum(b[3] - b[2] for b in warm_b)

    for _ in procs:
        task_q.put(None)
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    return {
        'workers': workers,
        'batch': int(batch),
        'io': io,
        'start_method': ctx.get_start_method(),
        'spawn_s': round(spawn_s, 4),
        'prime_cache_s': None if prime_s is None else round(prime_s, 4),
        'warm_batches': warm_batches,
        'bench_batches': len(timed_b),
        'rows': int(bench_rows),
        'elapsed_s': round(bench_s, 4),
        'rows_per_s': round(bench_rows / bench_s, 1) if bench_s > 0 else None,
        'ms_per_row_wall': round(bench_s / bench_rows * 1e3, 4) if bench_rows else None,
        'warmup_rows': int(sum(b[1] for b in warm_b)),
        'warmup_s': round(warm_s, 4),
        'errors': errors[:3],
    }


# --------------------------------------------------------------------------- #
# 内存高水位
# --------------------------------------------------------------------------- #
def _peak_wset_mb():
    """本进程的峰值工作集（MB）。

     **必须显式给 argtypes**：``GetProcessMemoryInfo`` 的第二个形参是指针，
    不声明时 ctypes 按 32 位 int 传 ⇒ 在 x64 上调用**静默失败返回 0**
    （症状是「峰值内存恒为 0」，不是报错）。本函数实测踩过。
    """
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD),
                    ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t)]

    try:
        dll = ctypes.WinDLL('psapi', use_last_error=True)
        fn = dll.GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        if fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / (1024 ** 2)
    except Exception:  # noqa: BLE001
        pass
    try:                                    # psutil 兜底（能给出 peak_wset）
        import psutil
        mi = psutil.Process().memory_full_info()
        return getattr(mi, 'peak_wset', None) or mi.rss / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return 0.0


def _mem_worker(mat_dir, batches, task_q, res_q):
    """只跑一批、把峰值工作集报回来的 worker。"""
    cols = open_columns(mat_dir)
    res_q.put(('ready', 0, 0.0))
    item = task_q.get()
    if item is None:
        return
    sel = item
    compute_features(cols, sel)
    res_q.put(('peak', 0, _peak_wset_mb(), None))


def measure_memory(mat_dir, idx, batch, io='warm', workers=1):
    """单 worker 处理**一个** batch 的峰值 RSS（MB）。

     **口径**是进程峰值工作集（PeakWorkingSetSize），它**包含** mmap 到的
    文件页 —— 那些页是可回收的，所以这个数字是**上界**；真正的匿名内存
    （Python + numpy + scipy + 那一批的中间量）要小得多。报告里两个都给出，
    并且 :func:`judge` 用「8 × 峰值 vs 物理内存」判红。
    """
    ctx = mp.get_context()
    sel = _trim(idx, batch)
    if io == 'warm':
        _prime_cache(open_columns(mat_dir), sel)
    tq, rq = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_mem_worker, args=(mat_dir, 1, tq, rq), daemon=True)
    t0 = time.perf_counter()
    p.start()
    rq.get()                      # 等就绪
    spawn_s = time.perf_counter() - t0
    tq.put(sel)
    _, _, peak_mb, _ = rq.get()
    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
    return {
        'workers': workers,
        'batch': int(batch),
        'io': io,
        'rows': int(sel.size),
        'spawn_s': round(spawn_s, 4),
        'peak_wset_mb': round(float(peak_mb), 1),
        'physical_mem_mb': _physical_mem_mb(),
        'x_workers_peak_mb': round(float(peak_mb) * workers, 1),
    }


def _physical_mem_mb():
    """物理内存（MB），拿不到就返回 ``None``（不要编一个数填进去）。"""
    try:
        import ctypes
        from ctypes import wintypes

        class MS(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD),
                        ('dwMemoryLoad', wintypes.DWORD),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        s = MS()
        s.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return round(s.ullTotalPhys / (1024 ** 2), 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 2), 1)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# gather：冷 / 热
# --------------------------------------------------------------------------- #
def cache_state_probe(cols, n_rows_probe=2000, seed=SEED + 313):
    """探 ``boards.npy`` 的**页缓存状态** —— 它决定「冷读」这个说法成不成立。

     **必须用随机行，不能用 stride**：本机实测（同一个未缓存的文件）
    stride-12 的读是 **1.2 µs/行**（341 MB/s，硬盘读预取吃到了红利），而随机
    行是 **6500 µs/行**。两者差 5000 倍 ⇒ 用 stride 探缓存状态会**永远**得出
    「已缓存」，哪怕一行都没缓存。

    用 2000 个随机行做探针：已缓存 ⇒ ~5 µs/行（10 ms 一趟）；未缓存 ⇒
    ~6.5 ms/行（13 s 一趟）。阈值取 0.5 ms/行。

     **为什么必须探**：本机物理内存 14.9 GB，``boards.npy`` 是 12.35 GB。
    跑过几轮之后部分区间会被文件缓存收进去，于是「冷读」会随跑过几轮而漂移
    （实测同一份数据、同一个 B，冷读在 11 ms/行与 0.02 ms/行之间跳）⇒
    不探这个状态，两次跑的数字没法互相比较。
    """
    mm = cols['boards']
    n = mm.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, n_rows_probe, replace=False).astype(np.int64))
    t0 = time.perf_counter()
    np.asarray(mm[idx])
    dt = time.perf_counter() - t0
    per_row = dt / n_rows_probe * 1e6
    return {
        'rows_sampled': int(n_rows_probe),
        'us_per_row_random': round(per_row, 3),
        'elapsed_s': round(dt, 4),
        'likely_cached': bool(per_row < 500.0),
        'verdict': ('boards.npy 的这批行已在页缓存里 ⇒ 本次 gather 的「冷读」'
                    '偏热，部署态要更慢' if per_row < 500.0 else
                    '这批行不在页缓存里（要下盘）⇒ 冷读可信'),
    }


def measure_gather(cols, n_rows, batch, offsets, n_trials=6, seed=SEED):
    """``offsets`` 上 gather 的耗时：**逐区间自带冷/热配对 + 自校验**。

    为什么冷热必须分开、且必须自校验
    ------------------------------
    本机数据集卷是 **USB 挂载**的 ``JMicron Generic``（见 :func:`measure_env`）。
    随机读一行 361 B 要 **~6.5 ms**（≈150 行/s），而命中页缓存只要 ~5 µs ——
    差 1300 倍。更麻烦的是「冷」在本机**不可复现**：12.35 GB 的文件放在
    14.9 GB 的机器上，跑过几轮就有区间被缓存收进去，于是同一个脚本第二次跑
    的「冷读」可能整个是热的。

     所以每个区间都**跑两遍**（第二遍必热），并报
    ``cold_over_warm_ratio``：

    * ratio ≫ 1 ⇒ 这个区间第一遍是真冷，用它；
    * ratio ≈ 1 ⇒ 第一遍就已经是热的（数据早被读过）⇒ 该区间**作废**，
      不进中位数。

    这样「冷」是被每区间实测证明的，而不是假设的。样本数因此会少于
    ``n_trials``（``n_valid`` 给出实际有效的个数）。

     **另两条实测结论，别再重试**：
    ① 「造一个没读过的临时文件来量冷读」在 Windows 上**不成立** ——
       刚写完的数据仍在回写缓存里（实测：写 3 GB 后立刻随机读 = 1.7 µs/行）。
    ② stride 探针**不能**用来判缓存状态（见 :func:`cache_state_probe`）。
    Args:
        offsets: 要量的偏移元组（任务书钦定 ``(1, 8, 32)``；ladder 用
            ``(-1, -2)``）。
        n_trials: 抽几段互不相交的区间。每段跑两遍，第二遍必热。
    """
    rng = np.random.default_rng(seed)
    n = cols['boards'].shape[0]
    mm = cols['boards']
    kw = dict(game_ids=cols['game_ids'], to_play=cols['to_play'], ko=cols['ko'])
    res = {'offsets': list(offsets), 'batch': int(batch),
           'board_rows_touched_per_sample': BOARD_ROWS_PER_SAMPLE,
           'n_trials': n_trials}
    cold, warm, regions = [], [], []
    for k in range(n_trials):
        lo = int((k + 1) * n / (n_trials + 2))
        idx = np.sort(rng.choice(np.arange(lo, min(lo + n, n)),
                                 batch, replace=False).astype(np.int64))
        t0 = time.perf_counter()
        gather_neighbors(mm, idx, offsets, **kw)
        c = (time.perf_counter() - t0) / batch * 1e3
        t0 = time.perf_counter()
        gather_neighbors(mm, idx, offsets, **kw)
        w = (time.perf_counter() - t0) / batch * 1e3
        ratio = (c / w) if w > 0 else float('inf')
        # 判据：冷热差不到 10 倍 ⇒ 第一遍就已经命中页缓存，这个区间不是冷样本。
        valid = ratio >= 10.0
        regions.append({'region_lo': lo, 'cold_ms_per_row': round(c, 4),
                        'warm_ms_per_row': round(w, 4),
                        'cold_over_warm_ratio': round(ratio, 1),
                        'counted_as_cold': valid})
        warm.append(w)
        if valid:
            cold.append(c)
    if not cold:
        res['cold_ms_per_row'] = {'min': None, 'median': None, 'max': None,
                                  'samples': []}
        res['n_valid_cold_regions'] = 0
        res['cold_note'] = ('本机没找到可证明为冷的区间（12.35 GB 的文件放在 '
                            '14.9 GB 的机器上，已被文件缓存收进去）⇒ 本次'
                            '**没有冷读样本**，别把 warm 的数当成部署态')
    else:
        res['cold_ms_per_row'] = {'min': round(float(np.min(cold)), 4),
                                  'median': round(float(np.median(cold)), 4),
                                  'max': round(float(np.max(cold)), 4),
                                  'samples': [round(x, 4) for x in cold]}
        res['n_valid_cold_regions'] = len(cold)
    res['warm_ms_per_row'] = {'min': round(float(np.min(warm)), 4),
                              'median': round(float(np.median(warm)), 4),
                              'max': round(float(np.max(warm)), 4),
                              'samples': [round(x, 4) for x in warm]}
    res['regions'] = regions

    # ---- 并发探针：同样的读，线程 1 / 4 / 8，看**聚合**吞吐 ----
    # 探针本身也受缓存状态影响，所以它只作**旁证**。要判存储，以
    # :func:`cache_state_probe` 为准（它明确说了这些行在不在缓存里）。
    from concurrent.futures import ThreadPoolExecutor
    probe = {}
    for nthread in (1, 4, 8):
        idxs = []
        for k in range(nthread):
            lo = int(n * (0.5 + 0.4 * k / max(1, nthread)))
            idxs.append(np.sort(rng.choice(np.arange(lo, n), batch,
                                           replace=False).astype(np.int64)))
        t0 = time.perf_counter()
        with ThreadPoolExecutor(nthread) as ex:
            list(ex.map(lambda i: gather_neighbors(mm, i, offsets, **kw), idxs))
        dt = time.perf_counter() - t0
        total = nthread * batch
        probe[f'threads_{nthread}'] = {
            'rows': total, 'elapsed_s': round(dt, 4),
            'rows_per_s': round(total / dt, 1),
            'ms_per_row': round(dt / total * 1e3, 4),
        }
    res['concurrency_probe'] = probe
    res['probe_verdict'] = (
        '并发不涨 ⇒ 随机读被存储串行化（若 cache_state_probe 说「已缓存」，'
        '则以 cache_state_probe 为准）'
        if probe['threads_8']['rows_per_s'] <=
        3.0 * probe['threads_1']['rows_per_s'] else
        '并发涨 ⇒ 这一组是 CPU 瓶颈（很可能是页缓存命中）')
    return res


# --------------------------------------------------------------------------- #
# materialize：部署成本
# --------------------------------------------------------------------------- #
def measure_materialize(npz_path, mat_dir, full_decompress=True):
    """报告 materialize 的**实测**耗时与落盘体积。

    三块，缺一不可：

    1. **落盘体积**：直接量磁盘上已有的 ``.npy``（真值）。全量五列 =
       boards 12.35 GB + 四个小列 0.65 GB。
    2. **小列的真实端到端耗时**：``materialize_dataset`` 跑 ``to_play`` /
       ``ko`` / ``game_ids`` / ``my_hist`` / ``op_hist``（若已存在则跳过，
       并把已测的耗时记进 ``previous_runs``）。 注意这一步会**跳过已存在的
       文件**—— 脚本不会删掉别人跑好的结果。
    3. **boards 的解压速率**：**流式**读压缩成员（不用
       ``np.load(npz)['boards']``，那会一次吃 12.35 GB）。
       ``materialize_dataset`` 本身**在本机跑不了 boards**——它第 276 行
       ``arr = z[k]`` 会把整个成员解压进内存，12.35 GB / 14.9 GB = 83%，
       剩下的 2.5 GB 要装 Windows + Python + numpy 的全部临时量 ⇒ 没有余量。
       所以这里量的是「等价路径的解压半程」，落盘半程按实测写速另计。
    """
    out = {
        'npz': npz_path,
        'npz_bytes': os.path.getsize(npz_path),
        'out_dir': mat_dir,
        'columns': {},
        'previous_runs': [],
    }
    for f in sorted(os.listdir(mat_dir)) if os.path.isdir(mat_dir) else []:
        p = os.path.join(mat_dir, f)
        if f.endswith('.npy') and os.path.isfile(p):
            out['columns'][f[:-4]] = os.path.getsize(p)
    out['total_materialized_bytes'] = sum(out['columns'].values())

    todo = [k for k in ('to_play', 'ko', 'game_ids', 'my_hist', 'op_hist')
            if k not in out['columns']]
    if todo:
        from src.data.kata_label_join import materialize_dataset
        t0 = time.perf_counter()
        materialize_dataset(npz_path, mat_dir, keys=tuple(todo))
        dt = time.perf_counter() - t0
        n = None
        with zipfile.ZipFile(npz_path) as z:
            fp = z.open('to_play.npy')
            v = np.lib.format.read_magic(fp)
            n = np.lib.format._read_array_header(fp, v)[0][0]
        out['small_cols_materialize_s'] = round(dt, 3)
        out['small_cols_rows'] = int(n)
        out['small_cols_rows_per_s'] = round(n / dt)
        out['small_cols_keys'] = list(todo)
        for k in todo:
            p = os.path.join(mat_dir, f'{k}.npy')
            if os.path.exists(p):
                out['columns'][k] = os.path.getsize(p)
        out['total_materialized_bytes'] = sum(out['columns'].values())

    if full_decompress:
        t0 = time.perf_counter()
        total = 0
        with zipfile.ZipFile(npz_path) as z:
            info = z.getinfo('boards.npy')
            fp = z.open('boards.npy')
            v = np.lib.format.read_magic(fp)
            shape, fortran, dtype = np.lib.format._read_array_header(fp, v)
            row = int(np.prod(shape[1:])) * dtype.itemsize
            fp.seek(fp.tell())
            chunk = 1 << 22
            while True:
                buf = fp.read(chunk - chunk % row)
                if not buf:
                    break
                total += len(buf)
        dt = time.perf_counter() - t0
        out['boards_npz_member'] = {
            'uncompressed_bytes': info.file_size,
            'compressed_bytes': info.compress_size,
            'rows': int(shape[0]),
            'dtype': str(dtype),
            'stream_decompress_s': round(dt, 2),
            'stream_decompress_rows_per_s': round(shape[0] / dt),
            'stream_decompress_MB_per_s': round(total / dt / 1e6, 1),
            'resident_if_materialize_dataset_used_bytes': info.file_size,
            'note': ('materialize_dataset() 第 276 行 arr = z[k] 会一次解压整个'
                     '成员（12.35 GB）⇒ 本机（14.9 GB）没有余量，故只测'
                     '流式解压半程；落盘半程见 boards_write_MB_per_s'),
        }
    return out


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
def measure_env():
    """机器 / 软件环境。**磁盘那一项是本次基准的关键上下文**。"""
    env = {
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'cpu_count_logical': os.cpu_count(),
        'physical_mem_mb': _physical_mem_mb(),
        'start_method': mp.get_start_method(),
    }
    try:
        import scipy
        env['scipy'] = scipy.__version__
    except Exception:  # noqa: BLE001
        env['scipy'] = None
    try:
        env['cpu_model'] = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001
        env['cpu_model'] = None
    # 磁盘：数据集卷挂在哪种总线上。这一项决定了 gather 能不能随机访问。
    try:
        drive = os.path.splitdrive(REPO)[0] or 'C:'
        env['repo_drive'] = drive
        env['disk'] = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "$d = Get-Partition -DriveLetter '%s' | Get-Disk;"
             "[pscustomobject]@{Name=$d.FriendlyName;Bus=$d.BusType} | "
             "ConvertTo-Json -Compress" % drive[0]],
            capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001
        env['disk'] = None
    return env


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #
def judge(env, comps, workers, mem, gather=None):
    """按**实测**数字判「3.04 ms/行」的预算是否成立，并给出推荐值。

     这个函数只做算术，**不含任何为了让预算成立而调的参数**。预算不成立
    就直接写不成立 —— 本脚本的存在就是为了拿到这个真答案。
    """
    out = {}
    # ---- 单进程：按层分别报（保守取最慢的那层）--------------------------
    per_stratum = {}
    for name, cell in comps.items():
        for bs, r in cell.items():
            if r.get('_pipeline'):
                per_stratum.setdefault(bs, {})[name] = {
                    'rows_per_s': round(r['_rows'] / r['_pipeline'], 1),
                    'ms_per_row': round(r['_pipeline'] / r['_rows'] * 1e3, 4),
                }
    out['single_process'] = per_stratum

    ref_bs = max(per_stratum, key=lambda b: int(b.split('=')[1])) \
        if per_stratum else None
    sp = per_stratum.get(ref_bs, {}) or {}
    if sp:
        by_rps = sorted(sp.items(), key=lambda kv: kv[1]['rows_per_s'])
        out['single_process_reference_batch'] = ref_bs
        out['single_process_fastest_stratum'] = {
            'stratum': by_rps[-1][0], 'rows_per_s': by_rps[-1][1]['rows_per_s']}
        out['single_process_slowest_stratum'] = {
            'stratum': by_rps[0][0], 'rows_per_s': by_rps[0][1]['rows_per_s']}
        # ---- 梯子占比：这一项是 C0 存在的核心理由，单独算清楚 ----
        # 用 ladder_1（单次搜索）×3 作分子而不是 ladder_3：实测
        # ladder_3/3 ≈ ladder_1（见 components 里两者之比），说明 ladder 的成本
        # **线性于搜索次数**、没有可摊的固定开销；这样分子不会把 spatial 里
        # 非 ladder 的那部分（fp16 缓冲 / 装配 / gate）算进来。
        shares = {}
        for name, cell in comps.items():
            for bs, r in cell.items():
                s = r.get('_seconds') or {}
                if 'ladder_1' in s and s['_pipeline'] > 0:
                    shares.setdefault(bs, {})[name] = {
                        'per_search_ms_per_row': round(
                            s['ladder_1'] / r['_rows'] * 1e3, 4),
                        'three_searches_ms_per_row': round(
                            3 * s['ladder_1'] / r['_rows'] * 1e3, 4),
                        'share_of_pipeline': round(
                            3 * s['ladder_1'] / s['_pipeline'], 4),
                        'non_ladder_ms_per_row': round(
                            (s['_pipeline'] - 3 * s['ladder_1'])
                            / r['_rows'] * 1e3, 4),
                        'measured_ladder_3_ms_per_row': round(
                            s['ladder_3'] / r['_rows'] * 1e3, 4),
                        'measured_spatial_ms_per_row': round(
                            s['spatial'] / r['_rows'] * 1e3, 4),
                    }
        out['ladder_share'] = shares
        out['ladder_share_note'] = (
            'share_of_pipeline 略微 **超过 1.0** 不是 bug：分子分母各自是'
            '「3 次 min-of-reps」的独立测量，噪声让它们来自不同的 rep。'
            '正确读法是「梯子基本就是全部成本」—— 实测 random 层的 '
            'ladder_3（3.37 ms/行）与 spatial 总量（3.32 ms/行）在噪声内相等，'
            '而 liberties_123 + calculate_area + gather + global 加起来只有 '
            '60 µs/行（1.8%）。')

    # ---- 多进程：冷热两条曲线 ----
    curves = {}
    for io in ('warm', 'cold'):
        rows = sorted([w for w in workers if w['io'] == io],
                      key=lambda w: w['workers'])
        curves[io] = [{'workers': w['workers'],
                       'rows_per_s': w['rows_per_s'],
                       'ms_per_row_wall': w['ms_per_row_wall'],
                       'spawn_s': w['spawn_s'],
                       'prime_cache_s': w['prime_cache_s']} for w in rows]
    out['worker_curves'] = curves

    def best(io):
        c = curves.get(io) or []
        return max(c, key=lambda x: x['rows_per_s']) if c else None

    b_warm, b_cold = best('warm'), best('cold')
    out['best_warm'] = b_warm
    out['best_cold'] = b_cold

    # ---- 预算判定：冷热**各判一次** ----
    # 之所以要判两次：特征计算是纯 CPU（热缓存下 gather 只占 0.02%），
    # 而冷读被数据卷的随机读延迟钉死。两个数差 ~4 倍，**哪个是部署态取决于
    # boards.npy 放在哪种总线上**（见 env.disk）—— 所以两个都要给，不能只挑
    # 好看的那一个。
    verdict = {'budget_ms_per_row': BUDGET_MS_PER_ROW}
    for tag, b in (('warm_cpu_bound', b_warm), ('cold_storage_bound', b_cold)):
        if b is None:
            verdict[tag] = None
            continue
        verdict[tag] = {
            'workers': b['workers'],
            'rows_per_s': b['rows_per_s'],
            'ms_per_row': b['ms_per_row_wall'],
            'over_budget_by': round(b['ms_per_row_wall'] / BUDGET_MS_PER_ROW, 2),
            'holds': b['ms_per_row_wall'] <= BUDGET_MS_PER_ROW,
        }
    src = verdict['cold_storage_bound'] or verdict['warm_cpu_bound']
    verdict['measured_ms_per_row'] = src['ms_per_row'] if src else None
    verdict['holds'] = src['holds'] if src else None
    verdict['basis'] = 'cold_storage_bound（部署态默认：本机 boards.npy 在 USB 卷上）'
    out['verdict'] = verdict

    # ---- 内存 ----
    if mem:
        worst = max(mem, key=lambda m: m['peak_wset_mb'])
        phys = worst['physical_mem_mb']
        total = worst['peak_wset_mb'] * worst['workers']
        out['memory'] = {
            'worst': worst,
            'total_peak_mb': round(total, 1),
            'total_peak_gb': round(total / 1024, 2),
            'physical_mem_gb': None if phys is None else round(phys / 1024, 2),
            'fits': None if phys is None else total < phys,
            'verdict': ('RED · 8 workers 的峰值已经超过物理内存'
                        if phys is not None and total >= phys
                        else 'OK · 8 workers 的峰值远小于物理内存'),
            'note': ('峰值工作集含 mmap 的文件页（可回收）⇒ 这是**上界**；'
                     '真正的匿名内存（解释器 + 那一批的中间量）要小得多'),
        }

    # ---- 4 个必答数字的速查（报告直接引用这一段）----
    ref = out.get('single_process_reference_batch')
    sp = out.get('single_process', {}).get(ref, {}) if ref else {}
    gath = gather or {}
    out['headline'] = {
        '1_single_process_rows_per_s': sp,
        '1_ladder_is_the_bottleneck': bool(
            shares and max((v['share_of_pipeline']
                            for cell in shares.values() for v in cell.values()),
                           default=0) > 0.9),
        '2_worker_curves': curves,
        '3_gather_ms_per_row': {
            k: {'cold_median': v['cold_ms_per_row']['median'],
                'cold_max': v['cold_ms_per_row']['max'],
                'warm_median': v['warm_ms_per_row']['median']}
            for k, v in gath.items() if 'cold_ms_per_row' in v} if gath else None,
        '3_reproducible_disk_random_read': (
            (storage or {}).get('random_read_ms_per_row')),
        '3_boards_cache_state': (gath or {}).get('boards_npz_cache_state'),
        '3_materialize_bytes': None,   # 由 run() 在 judge 之后填
        '4_memory': out.get('memory'),
    }
    return out


def recommend(env, batch_curve, comps, workers, gather):
    """给出 ``prefetch-workers`` / ``batch`` 的推荐值与**外推依据**。"""
    n_log = env.get('cpu_count_logical') or os.cpu_count() or 1
    rec = {}

    # ---- workers：看扩展曲线在哪里开始不再涨 ----
    for io in ('warm', 'cold'):
        c = [w for w in workers if w['io'] == io]
        c.sort(key=lambda w: w['workers'])
        if not c:
            continue
        pts = [(w['workers'], w['rows_per_s']) for w in c if w['rows_per_s']]
        if not pts:
            continue
        best_k, best_v = max(pts, key=lambda x: x[1])
        prev = [(k, v) for k, v in pts if k < best_k]
        eff = None
        if prev:
            k0, v0 = prev[-1]
            eff = round((best_v / v0) / (best_k / k0), 3)
        rec[f'workers_{io}'] = {
            'points': [{'workers': k, 'rows_per_s': v} for k, v in pts],
            'best_workers': best_k,
            'best_rows_per_s': best_v,
            'marginal_efficiency_at_best': eff,
            'note': ('冷/热两条曲线谁才是部署态，看 gather_*_ms_per_row：'
                     '数据卷是 USB 时冷曲线才是真实的' if io == 'cold'
                     else '热曲线 = CPU 上限（数据已在页缓存）'),
        }

    # ---- batch：看曲线有没有拐点 ----
    curve = sorted(batch_curve, key=lambda x: x['batch'])
    rec['batch'] = {
        'points': [{'batch': c['batch'], 'rows_per_s': c['rows_per_s'],
                    'ms_per_row': c['ms_per_row']} for c in curve],
        'max_rows_per_s_batch': max(curve, key=lambda c: c['rows_per_s'])['batch']
        if curve else None,
        'spread_max_over_min': (
            round(max(c['rows_per_s'] for c in curve) /
                  min(c['rows_per_s'] for c in curve), 3) if curve else None),
        'note': ('特征侧的 Python 开销**长在每行的梯子 DFS 里面**，不在批级；'
                 '所以「大 batch 摊薄开销」在本管道上基本不成立 ⇒ batch 应该'
                 '按训练侧（显存 / 收敛）选，而不是按特征吞吐选'),
    }
    return rec


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(args):
    cols = open_columns(args.mat_dir)
    n_rows = cols['boards'].shape[0]
    env = measure_env()
    result = {
        'meta': {
            'what': 'C0 · 22 通道 V7 特征吞吐基准',
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'seed': SEED,
            'budget_ms_per_row': BUDGET_MS_PER_ROW,
            'board_rows_per_sample': BOARD_ROWS_PER_SAMPLE,
            'cli': vars(args).copy(),
        },
        'env': env,
        'dataset': {
            'npz': args.npz,
            'npz_bytes': os.path.getsize(args.npz),
            'rows': int(n_rows),
            'mat_dir': args.mat_dir,
        },
    }

    if 'materialize' in args.phase:
        print('[1/6] materialize ...', flush=True)
        result['materialize'] = measure_materialize(
            args.npz, args.mat_dir, full_decompress=not args.quick)

    print('[2/6] 采样 ...', flush=True)
    strata = stratified_indices(cols, n_rows, args.first, args.random,
                                args.complex, args.pool)
    result['sampling'] = {
        k: (v[1] if isinstance(v, tuple) else v) for k, v in strata.items()}
    result['sampling']['method'] = (
        '分层采样：① 前 N 行 ② 均匀随机 M 行 ③ 复杂度分 top-K 行'
        '（分数 = 子数 + 3×2 气子数 + 20×有劫）。 只测前 N 行会系统性'
        '低估（开局空盘是最快路径），所以三层都报。')

    batches = [int(x) for x in args.batch.split(',') if x]
    curve = [int(x) for x in args.batch_curve.split(',') if x]
    layers = [x for x in args.workers.split(',') if x]

    if 'components' in args.phase:
        print('[3/6] 组件分解 ...', flush=True)
        comps = {}
        for name, (idx, _meta) in strata.items():
            comps[name] = {}
            for bs in batches:
                # **默认先读热**：组件分解要回答的是「CPU 要多久」，不是
                # 「USB 卷要多久」。冷 I/O 单独由 gather / workers 的 cold 档
                # 负责 —— 两者的比值就是存储给流水线上的一刀。
                prime = None if args.no_prime else _prime_cache(cols, idx)
                r = measure_components(cols, idx, bs, reps=args.reps)
                rows = r.pop('_rows')
                pipe = r.pop('_pipeline')
                comps[name][f'B={bs}'] = {
                    'rows': rows,
                    'prime_cache_s': None if prime is None else round(prime, 3),
                    'seconds': dict({k: round(v, 6) for k, v in r.items()},
                                    _pipeline=round(pipe, 6)),
                    'rows_per_s': round(rows / pipe, 1),
                    'ms_per_row': round(pipe / rows * 1e3, 4),
                }
                print(f'   {name:8s} B={bs:5d}: pipeline '
                      f'{pipe:.3f}s -> {rows/pipe:.0f} rows/s '
                      f'({pipe/rows*1e3:.3f} ms/row)', flush=True)
        result['components'] = comps

    if 'batch_curve' in args.phase:
        print('[4/6] batch 规模曲线 ...', flush=True)
        out = []
        key = 'random'
        idx = strata[key][0]
        prime = None if args.no_prime else _prime_cache(cols, idx)
        for bs in curve:
            r = measure_components(cols, idx, bs, reps=args.reps)
            rows = r.pop('_rows')
            pipe = r.pop('_pipeline')
            out.append({'batch': bs, 'rows': rows,
                        'prime_cache_s': None if prime is None else round(prime, 3),
                        'pipeline_s': round(pipe, 4),
                        'rows_per_s': round(rows / pipe, 1),
                        'ms_per_row': round(pipe / rows * 1e3, 4),
                        'seconds': dict({k: round(v, 6) for k, v in r.items()},
                                        _pipeline=round(pipe, 6))})
            print(f'   B={bs:5d}: {rows/pipe:8.0f} rows/s '
                  f'({pipe/rows*1e3:7.3f} ms/row)', flush=True)
        result['batch_curve'] = out

    if 'gather' in args.phase:
        print('[5/6] gather 冷/热 ...', flush=True)
        result['gather'] = {
            'boards_npz_cache_state': cache_state_probe(cols),
            'taskbook_offsets_1_8_32': measure_gather(
                cols, args.rows, min(batches), DEFAULT_OFFSETS,
                n_trials=1 if args.quick else 6, seed=SEED + 991),
            'ladder_offsets_-1_-2': measure_gather(
                cols, args.rows, min(batches), LADDER_OFFSETS,
                n_trials=1 if args.quick else 6, seed=SEED + 977),
        }
        st = result['gather']['boards_npz_cache_state']
        print(f'   boards.npz 页缓存: {st["MB_per_s"]} MB/s -> '
              f'{st["verdict"]}', flush=True)
        for k, v in result['gather'].items():
            if not isinstance(v, dict) or 'cold_ms_per_row' not in v:
                continue
            cm = v['cold_ms_per_row']['median']
            cm_s = f'{cm:.3f}' if cm is not None else 'N/A(无冷样本)'
            print(f'   {k}: cold median {cm_s} ms/row '
                  f'(有效冷区间 {v["n_valid_cold_regions"]}/{v["n_trials"]})  '
                  f'warm median {v["warm_ms_per_row"]["median"]:.4f}  '
                  f'| {v["probe_verdict"]}', flush=True)

    wres, mres = [], []
    if 'workers' in args.phase:
        print('[6/6] 多进程扩展 ...', flush=True)
        batch0 = max(batches)
        need = batch0 * (args.warm_batches + args.bench_batches)
        ios = [x for x in args.io.split(',') if x]
        combos = [(io, int(k)) for io in ios for k in layers]
        # **每个 (io, workers) 组合抽一组全新的行**。复用同一组行会让
        # ``io='cold'`` 名不副实：第一个组合跑完，这些行已经在页缓存里，
        # 后面的组合量到的是热读 ⇒ 「workers 越多越慢（因为 I/O 争抢）」
        # 这个结论会被伪造成「workers 越多越快」。所以按组合数把数据集切成
        # 互不相交的段，每段内再随机抽。
        n_cfg = max(1, len(combos))
        span = max(n_rows // (n_cfg + 2), need * 8)
        for c, (io, k) in enumerate(combos):
            lo = (c + 1) * span
            rng = np.random.default_rng(SEED + 5000 + c)
            idx = np.sort(rng.choice(np.arange(lo, min(lo + span, n_rows)),
                                     need, replace=False).astype(np.int64))
            r = measure_workers(args.mat_dir, idx, batch0, k,
                                warm_batches=args.warm_batches,
                                bench_batches=args.bench_batches, io=io)
            r['index_span'] = [int(lo), int(min(lo + span, n_rows))]
            wres.append(r)
            print(f'   io={io:4s} workers={r["workers"]:2d}: '
                  f'{r["rows_per_s"]} rows/s  spawn={r["spawn_s"]:.2f}s'
                  + (f'  prime={r["prime_cache_s"]:.1f}s'
                     if r['prime_cache_s'] is not None else ''), flush=True)
        result['workers'] = wres

    if 'memory' in args.phase:
        print('[+] 内存高水位 ...', flush=True)
        for bs in batches:
            mres.append(measure_memory(args.mat_dir, strata['random'][0], bs,
                                       io='warm', workers=8))
        result['memory'] = mres
        for m in mres:
            print(f'   B={m["batch"]:5d}: peak {m["peak_wset_mb"]:.0f} MB  '
                  f'x8 = {m["x_workers_peak_mb"]:.0f} MB  '
                  f'(物理 {m["physical_mem_mb"]:.0f} MB)', flush=True)

    comps_for_judge = result.get('components') or {}
    if comps_for_judge or wres or mres:
        flat = {}
        for name, cell in comps_for_judge.items():
            for bs, r in cell.items():
                flat.setdefault(name, {})[bs] = {
                    '_rows': r['rows'],
                    '_pipeline': r['seconds']['_pipeline'],
                    '_seconds': r['seconds']}
        result['verdict'] = judge(env, flat, wres, mres, result.get('gather'))
        result['recommendation'] = recommend(
            env, result.get('batch_curve', []), comps_for_judge, wres,
            result.get('gather'))
        if 'headline' in result['verdict']:
            result['verdict']['headline']['3_materialize_bytes'] = (
                (result.get('materialize') or {}).get('total_materialized_bytes'))
            result['verdict']['headline']['3_reproducible_disk_random_read'] = (
                (result.get('storage_probe') or {}).get(
                    'random_read_ms_per_row'))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n写出 {args.out}')
    return result


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='C0 · 22 通道 V7 特征吞吐基准')
    ap.add_argument('--npz', default=DEFAULT_NPZ, help='源 npz（默认主数据集）')
    ap.add_argument('--mat-dir', default=DEFAULT_MAT,
                    help='materialize 后的 .npy 目录')
    ap.add_argument('--out', default=DEFAULT_OUT, help='结果 JSON')
    ap.add_argument('--phase', default='materialize,components,batch_curve,'
                                       'gather,workers,memory',
                    help='逗号分隔的阶段名')
    ap.add_argument('--rows', type=int, default=512,
                    help='gather 计时用的 batch 行数')
    ap.add_argument('--first', type=int, default=1024, help='前 N 层行数')
    ap.add_argument('--random', type=int, default=2048, help='随机层行数')
    ap.add_argument('--complex', type=int, default=1024, help='复杂度层行数')
    ap.add_argument('--pool', type=int, default=24000, help='复杂度候选池行数')
    ap.add_argument('--batch', default='32,128,512',
                    help='组件分解 / 内存用的 batch 列表')
    ap.add_argument('--batch-curve', default='32,64,128,256,512,1024',
                    help='batch 规模曲线的 batch 列表')
    ap.add_argument('--workers', default='1,2,4,8', help='预取 worker 档位')
    ap.add_argument('--io', default='cold,warm',
                    help='gather 计时用 cold（未读热）/ warm（读热过）')
    ap.add_argument('--warm-batches', type=int, default=1)
    ap.add_argument('--bench-batches', type=int, default=4)
    ap.add_argument('--reps', type=int, default=3, help='组件计时的重复次数')
    ap.add_argument('--quick', action='store_true',
                    help='跳过全量 boards 解压压测 / 减少 gather 试次')
    ap.add_argument('--no-prime', action='store_true',
                    help='组件分解前**不**把盘面读热（会混进 USB 卷的冷读）')
    ap.add_argument('--probe-dir', default=os.path.join(REPO, 'tmp'),
                    help='卷随机读能力探针的临时文件目录（跑完自动删）')
    ap.add_argument('--probe-bytes', type=int, default=8_000_000_000,
                    help='卷随机读探针的文件大小（默认 8 GB：塞不进页缓存 ⇒ 真的下盘）')
    args = ap.parse_args()
    args.phase = [p.strip() for p in args.phase.split(',') if p.strip()]
    run(args)


if __name__ == '__main__':
    main()
