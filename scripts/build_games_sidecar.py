"""生成局级 sidecar `games.npz`（spec §5.3 / pipeline D0）。

为什么需要它
------------
主数据集 `data/sgf_19x19_full.npz`（34,202,713 行 / 162,298 局）里有 11 项 V7 需要的
量，但**唯独缺 6 个局级标量**：贴目 `KM`、分差 `RE`、规则 `RU`、是否认输。这 4 个
必须**逐局**存 —— 实测贴目分布是 7.5/6.5/5.5/3.8/**0**/2.8/4.5 混着 2.22% 缺失，
而 **81.1%** 的局 `RE` 是认输（无分差；spec §5.3.3 记的 75.2% 偏低，见
`sgf_parser.parse_result` 的实测表）。

 **主 npz 一个字节都不动。** 本脚本只读它，产物是独立的 `games.npz`（≈1.2 MB）。
逐行存会是 3.6 B/行 ⇒ 123 GB，所以**局级**而非逐行。

怎么把 sidecar 的第 `g` 行对到正确的 SGF（§5.3.4 哈希锚点法）
-------------------------------------------------------------
**不能用「重新枚举 SGF 按同样顺序」**：`build_dataset.build()` 有一串过滤（棋盘过大 /
让子棋 / 非法坐标 / 缺 `RE`）实测丢掉 45% 的 tgz 成员，而 `tarfile` 成员顺序 ≠ glob
顺序；更糟的是 `build_dataset.py:314-316` 的 **id 复用 bug**（`board.play()` 失败时
`return 0,1` 但已追加的行留在 `cur` 里、`game_id_counter` 未自增）让**同一个
`game_id` 对应两段不连续的行**，且 `game_ids` 本身**不是升序**（实测是 0..162297 的一个
置换）。

⇒ 锚点只依赖**盘面内容**，对枚举顺序与 id 复用都免疫：

1. `contiguous_runs` 切出每段行区间 `[start, end)`
2. 每段取锚点行 `r = start + min(20, (end-start)//2)`
3. 独立扫语料：每个 SGF 重放到**第 1..20 手之后**，把每个前缀的 `pos_hash` 存进排序表
4. 二分匹配 ⇒ `sidecar[game_id] = 那个 SGF 的元数据`

 **为什么存 20 个前缀而不是只存 ply-20。** `min(20, L//2)` 在 `L < 40` 时**不是 20**：
实测主数据集有 **134 局 `L < 40`、2 局 `L < 20`**，硬编码 ply 20 会漏掉这 136 局。
存 1..20 全部前缀（169,878 局 × 20 = 3.4M 个散列 = 27 MB）就把它们全兜住了。
代价：每个 SGF 一次重放到第 20 手，20 次 `pos_hash`（批量算，摊薄后 ~µs/个）。

 **同 id 多段行的裁决。** id 被复用时，一个 `game_ids` 值对应两段行，且 sidecar 是
**按 id 索引**的（训练侧 `sidecar[game_ids[idxs]]`），所以两段必须给出同一个值。
本脚本取**行数更多的那段**（完整的局，而非 `play()` 失败留下的残段），并把冲突数打进
覆盖报告 —— 不静默取第一个。

只读主 npz 的两条硬规矩
----------------------
* `np.load('x.npz')['boards']` 会把**整个成员**解压进内存（12.3 GB，本机 13.9 GB），
  任何切片都会 OOM。所以先用 `kata_label_join.materialize_dataset` 落成 `.npy`，
  再 `mmap_mode='r'` 随机取锚点行。
* 读 `game_ids` / 校验 shape 时**只读 zip 成员头**（`np.lib.format.read_magic` +
  `read_array_header_*`），不解压成员本体。

用法
----
    python scripts/build_games_sidecar.py                       # 全量
    python scripts/build_games_sidecar.py --limit-games 50 \
        --out tmp/games_probe.npz                              # 小规模探针
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tarfile
import time
import zipfile
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.kata_label_join import materialize_dataset  # noqa: E402
from src.data.pos_hash import pos_hash_block  # noqa: E402
from src.data.sgf_parser import (  # noqa: E402
    RULES_DEFAULT, SGFParser, parse_rules, recognized_rules,
)
from src.game.go_rules import GoBoard  # noqa: E402

BOARD = 19
#: spec §5.3.4 的锚点 ply。真实上限是 `min(20, L//2)`，见模块 docstring。
ANCHOR_PLY = 20
#: 锚点前缀的下界。**从 1 开始而不是 0**：offset 0 是空枰（`to_play=1, ko=-1`），
#: 17 万局全同 —— 存进去会让任何 `L < 2` 的残段锚到**随机一局**上。
#: 而 `build_dataset` 拒收 `len(moves) < 2`，故 `min(20, L//2) ≥ 1` 恒成立。
ANCHOR_MIN_OFFSET = 1

#: 需要 `.npy` 化的列。`game_ids` 也要 —— 它是 137 MB，一次性读进内存是可接受的，
#: 但 mmap 让 `--limit-games` 的小规模路径不必付这个代价。
NEEDED_KEYS = ('boards', 'to_play', 'ko', 'game_ids')

#: `KM` 离群阈值。实测 `data/games/games` 里 fox 语料把贴目写成 **×100**：
#: `KM[650]` / `KM[750]` / `KM[550]` / `KM[450]` / `KM[375]` / `KM[325]`
#: —— 除以 100 正好是 6.5 / 7.5 / 5.5 / 4.5 / 3.75 / 3.25 这一组**像样贴目**，
#: 而除以 2 得到 325/375/275/225 全都不是。共 2,163 局（1.62%）。
#:
#: spec §5.3.1 的 `KM` 分布表**没有列这一族**（那张表的前 8 项合计 96.5%，
#: 剩下的 3.5% 装得下它们），所以这不是与 spec 冲突，而是 spec 未覆盖。
#: 不修正的话 `g_komi=650` 会把全局 ch5（`currentSelfKomi/20`）顶到 32.5。
KOMI_OUTLIER_ABS = 100.0


def _log(msg: str = '') -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# 只读 npz 成员头
# --------------------------------------------------------------------------- #
def npz_member_meta(npz_path: str, key: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """取 npz 里某成员的 `(shape, dtype)`，**不解压成员本体**。

     为什么不能用 `np.load(npz)[key].shape`：`np.load` 的 NpzFile 是**懒解压**的，
    但 `__getitem__` 会把整个成员读进内存才返回数组 —— 对 `boards` 就是 12.3 GB。
    这里直接读 zip 成员里的 `.npy` 文件头（几十字节）。
    """
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open(f'{key}.npy') as f:
            ver = np.lib.format.read_magic(f)
            if ver == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif ver == (2, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f'{npz_path}:{key} 的 .npy 版本 {ver} 不支持')
    return tuple(shape), dtype


def npy_meta(path: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """取 `.npy` 的 `(shape, dtype)`。走 mmap ⇒ **只读文件头**。"""
    a = np.load(path, mmap_mode='r')
    return tuple(a.shape), a.dtype


def ensure_materialized(dataset: str, out_dir: str, keys: Sequence[str] = NEEDED_KEYS,
                        force: bool = False, log: Callable[[str], None] = _log) -> Dict[str, str]:
    """保证 `<out_dir>/<key>.npy` 存在且 shape/dtype 与 npz 一致，返回路径表。

    复用已存在的 `.npy` 是**必须的**而不是优化：`boards.npy` 是 12.3 GB，重写一遍
    要几十分钟，而它只依赖一个**不 rebuild** 的主 npz ⇒ 天然是稳定缓存。
    一致性用 shape + dtype 判定；不一致（换了数据集）就重做。

     仍然复用 `kata_label_join.materialize_dataset` 这个**已测过的**落盘函数
    （它内部用 `open_memmap` 建完整形状再分块填 —— 文档里记着「先写头再 append」
    会产出前段全 0 的坏文件），本脚本不重写一遍。
    """
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}
    for k in keys:
        want_shape, want_dtype = npz_member_meta(dataset, k)
        p = os.path.join(out_dir, f'{k}.npy')
        if os.path.isfile(p) and not force:
            try:
                got_shape, got_dtype = npy_meta(p)
            except (ValueError, OSError):
                got_shape, got_dtype = None, None
            if got_shape == want_shape and np.dtype(got_dtype) == np.dtype(want_dtype):
                log(f'[sidecar] 复用 {p}  {got_shape} {got_dtype}')
                out[k] = p
                continue
            log(f'[sidecar] {p} 与数据集不一致（{got_shape}/{got_dtype} vs '
                f'{want_shape}/{want_dtype}），重新落盘')
        t0 = time.time()
        materialize_dataset(dataset, out_dir, keys=(k,))
        log(f'[sidecar] 落盘 {p}  {want_shape} {want_dtype}  用时 {time.time()-t0:.0f}s')
        out[k] = p
    return out


# --------------------------------------------------------------------------- #
# 行区间与锚点
# --------------------------------------------------------------------------- #
def contiguous_runs(game_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """切出 `game_ids` 的每段**极大连续同值区间**，返回 `(ids, starts, ends)`。

    半开区间 `[starts[i], ends[i])`，`ids[i]` 是该段的 `game_ids` 值。

     **为什么按「连续段」而不是按「distinct id」建表**：实测 `game_ids` 不是升序
    （是 0..162297 的一个置换，`np.diff` 里有负数），而且 `build_dataset.py:314-316`
    的 id 复用 bug 会让**同一个 id 出现在两段不连续的行**上。后者一旦按 distinct id
    建表就会把两段行的锚点算成一个，锚到错误的局面。

    实现用相邻比较而不是 `np.diff`：int32 上 `np.diff` 会产出一个 137 MB 的临时数组，
    直接比较连临时量都不需要，结果完全等价。
    """
    g = np.asarray(game_ids)
    n = int(g.size)
    if n == 0:
        e = np.empty(0, np.int64)
        return e, e, e
    brk = np.flatnonzero(g[1:] != g[:-1]).astype(np.int64) + 1
    starts = np.concatenate((np.zeros(1, np.int64), brk))
    ends = np.concatenate((brk, np.full(1, n, np.int64)))
    return g[starts].astype(np.int64), starts, ends


def anchor_rows(starts: np.ndarray, ends: np.ndarray, ply: int = ANCHOR_PLY) -> np.ndarray:
    """锚点行号 `start + min(ply, (end-start)//2)`（spec §5.3.4）。

    行 `i`（局内偏移）记录的是**已走满 `i` 手**的局面 —— 见 `build_dataset.py:303`
    「先 append 盘面再 `play()`」。所以 `r - start` 就是 SGF 侧要重放到的手数。
    """
    return starts + np.minimum(int(ply), (ends - starts) // 2)


def anchor_rows_cross(starts: np.ndarray, ends: np.ndarray, ply: int = ANCHOR_PLY
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """**交叉校验锚点**：与 `anchor_rows` 不同偏移、且仍落在段内的第二个行号。

    Returns:
        `(rows, valid)`；`valid` 为 False 的段没有可用的第二锚点（`rows` 记 -1）

    为什么需要它
    ------------
    `build_dataset.py:314-316` 的 id 复用 bug 让**一个 `game_id` 对应两段相邻的行**
    （残段 + 完整局），`contiguous_runs` 会把它们并成**一段**。此时
    `min(20, L//2)` 可能落在残段里，于是元数据取自**残段那一局** —— 而按 id 索引的
    训练侧会把这份元数据用到完整局的那几十行上。这是锚点法在 id 复用下的**已知边界**。

    本仓主数据集实测没有这个问题（162,298 段 ↔ 162,298 个 distinct id，一一对应），
    所以这里**不改变匹配结果**，只加一道计数：两个锚点解析到不同的 SGF 就打
    `n_anchor_disagree` 进覆盖报告 —— 让这个边界**可见**，而不是静默取一个。

     **这道校验是部分检测，不是证明。** 散列表只存到第 20 手，所以能查的偏移上界
    是 20。可检出区间是「残段长度 ∈ [a1//2, a1)」（两个探针一个落在残段、一个落在
    完整局里）；残段 ≥ `a1` 时两个探针都在残段里，查不出来 —— 此时该段的元数据
    归属**真的有歧义**，只能靠「distinct id 数 == 段数」这条全局不变式来排除
    （本数据集实测成立）。
    """
    L = ends - starts
    a1 = np.minimum(int(ply), L // 2)
    a2 = a1 // 2
    bad = a2 < 1
    alt = a1 + 1
    take_alt = bad & (alt < L)
    a2 = np.where(take_alt, alt, a2)
    bad = bad & ~take_alt
    a2 = np.where(bad, -1, a2)
    return starts + a2, ~bad


def dataset_anchor_hashes(mat: Dict[str, str], rows: np.ndarray,
                          chunk: int = 8192,
                          log: Callable[[str], None] = _log) -> np.ndarray:
    """读 `boards/to_play/ko` 的 `.npy` mmap，算给定行的 `pos_hash`。

    行号先升序再分块取 —— memmap 的随机行读最终仍要落到页上，**顺序访问能让
    预读生效**。取完再散回原顺序。
    """
    boards = np.load(mat['boards'], mmap_mode='r')
    to_play = np.load(mat['to_play'], mmap_mode='r')
    ko = np.load(mat['ko'], mmap_mode='r')
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind='stable')
    srt = rows[order]
    out = np.empty(srt.size, dtype=np.uint64)
    t0 = time.time()
    for s in range(0, srt.size, chunk):
        e = min(s + chunk, srt.size)
        sel = srt[s:e]
        out[s:e] = pos_hash_block(boards[sel], to_play[sel], ko[sel])
        if s and (s // chunk) % 64 == 0:
            log(f'[sidecar]   锚点 {s}/{srt.size}  {time.time()-t0:.0f}s')
    res = np.empty_like(out)
    res[order] = out
    log(f'[sidecar] 锚点散列 {srt.size} 个，用时 {time.time()-t0:.0f}s')
    return res


# --------------------------------------------------------------------------- #
# 语料枚举
# --------------------------------------------------------------------------- #
def plan_sources(sgf_dirs: Sequence[str],
                 log: Callable[[str], None] = _log) -> Tuple[List[str], List[str]]:
    """枚举语料，返回 `(archives, files)`，两者都按 realpath 去重 + 排序。

     **必须去重**：默认的 `--sgf-dirs` 是 `["data", "data/games/games"]`，而
    `data/games/games` 就在 `data` 下面 —— 递归 glob 会把 `data/games/games` 的
    133,604 个文件**扫两遍**。同一个 SGF 出现两次会在 `pos_hash` 表里产生重复项，
    匹配率报告随之失真。

    去重用 `realpath`（跟着符号链接回到同一份文件），而不是 `abspath`。
    """
    archives: Dict[str, str] = {}
    files: Dict[str, str] = {}
    for d in sgf_dirs:
        if not os.path.exists(d):
            log(f'[sidecar] 语料路径不存在，跳过：{d}')
            continue
        pats = (os.path.join(d, '**', '*.tgz'), os.path.join(d, '**', '*.tar.gz'))
        for pat in pats:
            for p in glob.glob(pat, recursive=True):
                if os.path.isfile(p):
                    rp = os.path.realpath(p)
                    archives[rp] = p
        for p in glob.glob(os.path.join(d, '**', '*.sgf'), recursive=True):
            if os.path.isfile(p):
                rp = os.path.realpath(p)
                files[rp] = p
    arch = sorted(archives.values())
    fl = sorted(files.values())
    log(f'[sidecar] 语料：{len(arch)} 个 tgz + {len(fl)} 个目录内 SGF '
        f'= {len(arch) + len(fl)} 个来源')
    return arch, fl


def iter_sgf_bytes(archives: Sequence[str], files: Sequence[str]) -> Iterable[Tuple[str, bytes]]:
    """产出 `(标签, 原始字节)`。tgz 走 `r|gz` **流式**，不解包到磁盘。"""
    for a in archives:
        try:
            tf = tarfile.open(a, 'r|gz')
        except (tarfile.TarError, OSError) as e:
            _log(f'[sidecar] 打不开 {a}：{e}')
            continue
        with tf:
            for m in tf:
                if not m.isfile() or not m.name.lower().endswith('.sgf'):
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                yield f'{os.path.basename(a)}!{m.name}', fh.read()
    for p in files:
        try:
            with open(p, 'rb') as f:
                yield p, f.read()
        except OSError:
            continue


# --------------------------------------------------------------------------- #
# SGF 侧：重放 + 散列表
# --------------------------------------------------------------------------- #
def replay_anchor_positions(game, max_offset: int = ANCHOR_PLY
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """重放到第 `ANCHOR_MIN_OFFSET..max_offset` 手之后，返回四个数组。

    Returns:
        `(boards (K,19,19) int8, to_play (K,) int8, ko (K,) int16, offsets (K,) int8)`

     **落子/提子/判罚必须与 `build_dataset._emit` 逐步一致**，否则散列全不同而
    join 结果为空 —— 且这个症状看不出原因：
    * 用同一个 `GoBoard`（`go_rules.py`，其 `play()` 走 TT 判罚）；
    * 坐标 `target = -1 if pass else r*19 + c`，棋盘大小固定 19（主数据集只有 19 路，
      小枰在 `build_dataset` 里被拒于 `board_size > board_size` 之前）；
    * `play()` 返回 False（非法走法）就**停在这里**。这与 `build_dataset.py:314-316`
      一致：那一局不进 sidecar 的候选，但**它已追加的行留在 `cur` 里**且带着同一个
      `game_id` —— 所以要保留失败前算出的那几个前缀（它们是真在数据集里的局面），
      而不是整局丢掉。
    """
    b = GoBoard(BOARD, komi=game.komi)
    boards: List[np.ndarray] = []
    tps: List[int] = []
    kos: List[int] = []
    offs: List[int] = []
    for mv in game.moves:
        r, c = mv.position
        target = -1 if (r, c) == (-1, -1) else r * BOARD + c
        if not b.play(target):
            break
        off = len(boards) + ANCHOR_MIN_OFFSET
        boards.append(b.board.copy())
        tps.append(int(b.current_player))
        kos.append(int(b.ko_point))
        offs.append(off)
        if off >= max_offset:
            break
    if not boards:
        return _empty_positions()
    return (np.stack(boards).astype(np.int8),
            np.asarray(tps, np.int8),
            np.asarray(kos, np.int16),
            np.asarray(offs, np.int8))


def _empty_positions() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.empty((0, BOARD, BOARD), np.int8), np.empty(0, np.int8),
            np.empty(0, np.int16), np.empty(0, np.int8))


def parse_komi(props: Dict[str, str], fix_outlier: bool = True) -> Tuple[float, bool]:
    """从 SGF 属性取 `KM`，返回 `(贴目, 是否被离群修正)`。

     **缺失填 0.0**（spec §5.3.1 的表里「缺」占 2.22%），但**不能靠
    `komi != 7.5` 判缺失** —— `GameRecord.komi` 的默认值就是 7.5，判不出。
    所以查属性本身（`has_komi` 同源）。

    `fix_outlier` 处理 fox 语料的 `KM[650]`（= 6.5 贴目，×100），见 `KOMI_OUTLIER_ABS`。
    """
    raw = props.get('KM')
    if raw is None:
        return 0.0, False
    try:
        v = float(raw.strip())
    except (TypeError, ValueError):
        return 0.0, False
    if fix_outlier and abs(v) > KOMI_OUTLIER_ABS:
        return v / 100.0, True
    return v, False


class AnchorIndex:
    """SGF 侧的 `pos_hash` 排序表，支持**边扫边查**（供 `--limit-games` 提前收工）。

    不用 Python dict：169,878 局 × 20 前缀 = 3.4M 项，dict 会吃掉几百 MB 且
    每次查询是 Python 级循环。这里是 `(uint64 排序数组 + 二分 searchsorted)`，
    3.4M 项 27 MB，查询全向量化。
    """

    def __init__(self) -> None:
        self._h: List[np.ndarray] = []
        self._s: List[np.ndarray] = []
        self._o: List[np.ndarray] = []
        self._sh: Optional[np.ndarray] = None
        self._ss: Optional[np.ndarray] = None
        self._so: Optional[np.ndarray] = None
        self.n_sgf = 0

    def add(self, hashes: np.ndarray, sgf_idx: np.ndarray, offsets: np.ndarray) -> None:
        if hashes.size:
            self._h.append(np.asarray(hashes, np.uint64))
            self._s.append(np.asarray(sgf_idx, np.int32))
            self._o.append(np.asarray(offsets, np.int8))

    @property
    def n_positions(self) -> int:
        return int(sum(a.size for a in self._h))

    def _finalize(self) -> None:
        if self._sh is not None:
            return
        if not self._h:
            self._sh = np.empty(0, np.uint64)
            self._ss = np.empty(0, np.int32)
            self._so = np.empty(0, np.int8)
            return
        h = np.concatenate(self._h)
        s = np.concatenate(self._s)
        o = np.concatenate(self._o)
        order = np.argsort(h, kind='stable')
        self._sh, self._ss, self._so = h[order], s[order], o[order]

    def lookup(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """二分匹配。

        Returns:
            `(sgf_idx int64 (-1 = 未匹配), 每查询的匹配位置数 int64, 命中掩码 bool)`

         同一 `pos_hash` 可能对应多个 SGF（语料里有重复棋谱；同一个常见布局也能被
        不同局走到 —— 实测 `--limit-games 50` 的探针里，**每一个**锚点都有副本）。
        取**排序后第一条**（= 扫描顺序里最早的那个）。因此 `sgf_idx` 是**单值抽样**
        而不是集合 —— 要判「这两个锚点是不是同一局」必须走
        `cross_consistency()`，不能比 `sgf_idx`。
        """
        self._finalize()
        q = np.asarray(queries, dtype=np.uint64)
        if self._sh.size == 0 or q.size == 0:
            z = np.zeros(q.size, np.int64)
            return np.full(q.size, -1, np.int64), z, np.zeros(q.size, bool)
        lo = np.searchsorted(self._sh, q, side='left')
        hi = np.searchsorted(self._sh, q, side='right')
        counts = (hi - lo).astype(np.int64)
        hit = counts > 0
        safe = np.clip(lo, 0, self._sh.size - 1)
        res = np.where(hit, self._ss[safe].astype(np.int64), -1)
        return res, counts, hit

    def ranges(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """每个查询在排序表里的命中区间 `[lo, hi)`（半开，可为空）。"""
        self._finalize()
        q = np.asarray(queries, dtype=np.uint64)
        if self._sh.size == 0:
            z = np.zeros(q.size, np.int64)
            return z, z.copy()
        return (np.searchsorted(self._sh, q, side='left').astype(np.int64),
                np.searchsorted(self._sh, q, side='right').astype(np.int64))


def _flat_keys(lo: np.ndarray, hi: np.ndarray, sgf_of_pos: np.ndarray,
               n_sgf: int) -> Tuple[np.ndarray, np.ndarray]:
    """把「每查询一个命中区间」摊平成 `(复合键, 查询号)`。

    复合键 `qid * n_sgf + sgf_idx` 唯一标识「第 qid 个查询命中了第 sgf_idx 局」。
    """
    cnt = (hi - lo).astype(np.int64)
    tot = int(cnt.sum())
    if tot == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    base = (np.repeat(lo, cnt)
            + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)))
    qid = np.repeat(np.arange(cnt.size, dtype=np.int64), cnt)
    return qid * np.int64(max(n_sgf, 1)) + sgf_of_pos[base].astype(np.int64), qid


def distinct_signature_counts(index: 'AnchorIndex', queries: np.ndarray,
                              sig: np.ndarray, chunk: int = 32768) -> np.ndarray:
    """每个查询的命中里，有多少个**不同的内容签名**（= 不同的棋）。

     这是「元数据会不会挂错」的直接度量：`counts > 1` 只说明有副本（无害），
    而**签名 > 1** 说明**不同的局**走到了同一个局面 —— 此时 `lookup` 取的
    「最早那份」的 `KM/RE/RU` 未必属于本局。实测探针里这类情况存在（重复棋谱）。

    签名先压成稠密 id 再与查询号做复合键（`qid * S + sig_id`），避免 64 位相乘溢出。
    """
    index._finalize()
    q = np.asarray(queries, dtype=np.uint64)
    dense = np.zeros(int(sig.size) + 1, np.int64)
    if sig.size:
        _u, dense[:sig.size] = np.unique(sig, return_inverse=True)
    n_sgf = max(int(sig.size), 1)
    lo, hi = index.ranges(q)
    out = np.zeros(q.size, np.int64)
    for s in range(0, q.size, chunk):
        e = min(s + chunk, q.size)
        cnt = (hi[s:e] - lo[s:e]).astype(np.int64)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        base = (np.repeat(lo[s:e], cnt)
                + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)))
        qid = np.repeat(np.arange(e - s, dtype=np.int64), cnt)
        g = index._ss[base].astype(np.int64)
        pair = np.unique(qid * np.int64(n_sgf) + dense[np.clip(g, 0, dense.size - 1)])
        qq = pair // np.int64(n_sgf)
        counts = np.bincount(qq, minlength=e - s)
        out[s:e] = counts
    return out


def cross_consistency(index: 'AnchorIndex', primary: np.ndarray,
                      cross: np.ndarray, chunk: int = 32768
                      ) -> Tuple[int, int, int, int]:
    """主锚点与交叉锚点的**命中集合是否相交**。

    Returns:
        `(不一致数, 无法判定数, 主锚点歧义数, 交叉锚点歧义数)`

     **必须比集合，不能比「第一条命中」。** 实测探针：`--limit-games 50` 里 5 个段
    的交叉锚点（偏移 10）命中 **4~10 个不同的局** —— 同一个常见布局被很多局走到。
    拿第一条比就会把这些**布局歧义**误报成「这一段跨了两局」。真集合相交就不误报。

    「不一致」= 两边都命中、但没有任何一局同时出现在两边 ⇒ 该段确实跨了局
    （`game_ids` 被复用）。「无法判定」= 只有一边命中。

    分块算：常见布局能命中几百局，162K 个查询一次性摊平会吃掉几百 MB。
    """
    index._finalize()
    n = int(np.asarray(primary).size)
    lo_p, hi_p = index.ranges(primary)
    lo_c, hi_c = index.ranges(cross)
    hit_p, hit_c = hi_p > lo_p, hi_c > lo_c
    n_bad = n_inconclusive = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        kp, _ = _flat_keys(lo_p[s:e], hi_p[s:e], index._ss, index.n_sgf)
        kc, _ = _flat_keys(lo_c[s:e], hi_c[s:e], index._ss, index.n_sgf)
        share = np.zeros(e - s, dtype=bool)
        if kp.size and kc.size:
            qs = np.unique(np.intersect1d(kp, kc)) // np.int64(max(index.n_sgf, 1))
            share[qs] = True
        n_bad += int((hit_p[s:e] & hit_c[s:e] & ~share).sum())
        n_inconclusive += int((hit_p[s:e] ^ hit_c[s:e]).sum())
    return (n_bad, n_inconclusive,
            int(((hi_p - lo_p) > 1).sum()), int(((hi_c - lo_c) > 1).sum()))


#: 散列表里「这行是签名而不是锚点前缀」的哨兵。必须**小于 0**（合法偏移是 1..20）。
SIG_OFFSET = -1


class CorpusScan:
    """一次语料扫描的产物与计数。"""

    def __init__(self) -> None:
        self.komi = np.empty(0, np.float16)
        self.score = np.empty(0, np.float16)
        self.rules = np.empty(0, np.int8)
        self.resign = np.empty(0, np.bool_)
        self.re = np.empty(0, np.int8)     # 0 未识别/无结果 1 数值 2 和棋 3 认输
        #: 每局**是谁认输**：0 = 黑 / 1 = 白 / -1 = 非认输。
        #: 用 -1 而不是 None/0 占位，因为 0 在这里有实义（黑认输）——
        #: 拿 0 当「无信息」的默认值会让「白认输」与「什么都不知道」混成一类。
        #: `resign` 只是 bool，**恢复不出方向**，所以这一列必须自己落盘。
        self.resign_side = np.empty(0, np.int8)
        self.has_komi = np.empty(0, np.bool_)
        #: 每局的**内容签名** = 它最后一个被记录前缀（偏移 `min(20, 手数)`）的散列。
        #:
        #: **交叉校验必须比签名，不能比局号。** 语料里有重复棋谱（spec §5.0 说
        #: 「36,274 **唯一** SGF」⇒ tgz 成员里本来就有重复），同一个局面在散列表里
        #: 因此对应**多个局号**。拿局号去比会把「同一局的两个副本」误判成
        #: 「这一段跨了两局」。
        self.sig = np.empty(0, np.uint64)
        self.stats: Dict[str, int] = {}


#: `CorpusScan.re` 的取值
RE_CLASS_UNKNOWN, RE_CLASS_SCORE, RE_CLASS_DRAW, RE_CLASS_RESIGN = 0, 1, 2, 3
#: `CorpusScan.stats` 里 `ru_*` 的取值
RU_CLASS_DEFAULT, RU_CLASS_AREA, RU_CLASS_TERRITORY, RU_CLASS_UNRECOGNIZED = 0, 1, 2, 3

#: `derive_outcome` 的取值（下游 A 阶段的硬 outcome 标签）。
OUTCOME_BLACK, OUTCOME_WHITE, OUTCOME_DRAW = 0, 1, 2


def derive_outcome(g_re, g_score, g_resign_side) -> np.ndarray:
    """把 sidecar 的三列局级标签压成 A 阶段的硬 outcome。返回 `int64`、形状同输入。

    ============  ==================================  ================
    `g_re`       判据                                 结果
    ============  ==================================  ================
    `RE_CLASS_SCORE`  `g_score` 的符号（黑−白）        黑胜 0 / 白胜 1
    `RE_CLASS_RESIGN` 谁认输谁输（`g_resign_side`）    黑认输→1 / 白认输→0
    `RE_CLASS_DRAW` / `RE_CLASS_UNKNOWN`  ——          2
    ============  ==================================  ================

     **SCORE 分支不减贴目。** `g_score` 是 SGF 的最终分差，符号已经是**黑−白**
      且**含贴目**（`ResultInfo.score` 的约定），再减一次 `g_komi` 会把
      `B+2.5 / KM[7.5]` 这类局翻成白胜 —— 这是这个契约里最容易写错的一步。
     **按 `g_re` 分派，不看另一列。** 复用旧 `.scan.npz` 缓存时两列可能来自不同
      扫描批次，一个「RESIGN 却残留了分差」的样本仍按认输判。
     两个「理论上不可达」的退化情形，按上面那张表**字面**取值、不额外兜底，
      因为它们一旦发生就说明上游坏了，静默修正只会把坏数据藏起来：
      `SCORE` 但 `g_score` 是 `NaN`（`NaN > 0` 为假 ⇒ 判白胜）、
      `RESIGN` 但 `g_resign_side == -1`（≠ 0 ⇒ 判黑胜）。
    """
    re_arr = np.asarray(g_re)
    score_arr = np.asarray(g_score, np.float32)
    side_arr = np.asarray(g_resign_side)
    out = np.full(re_arr.shape, OUTCOME_DRAW, np.int64)

    is_score = re_arr == RE_CLASS_SCORE
    is_resign = re_arr == RE_CLASS_RESIGN
    # SCORE：符号已是黑−白，> 0 即黑胜
    out[is_score] = np.where(score_arr[is_score] > 0, OUTCOME_BLACK, OUTCOME_WHITE)
    # RESIGN：side==0 是**黑**认输 ⇒ 白胜；side==1 是白认输 ⇒ 黑胜
    out[is_resign] = np.where(side_arr[is_resign] == 0, OUTCOME_WHITE, OUTCOME_BLACK)
    return out


def classify_rules(ru: Optional[str]) -> int:
    """`RU` 的四分类（供覆盖报告用；实际存进 `g_rules` 的仍是 `parse_rules`）。"""
    if ru is None or not ru.strip():
        return RU_CLASS_DEFAULT
    if not recognized_rules(ru):
        return RU_CLASS_UNRECOGNIZED
    return RU_CLASS_TERRITORY if parse_rules(ru) & 1 else RU_CLASS_AREA


def scan_corpus(sgf_dirs: Sequence[str] = (), parser: Optional[SGFParser] = None,
                max_offset: int = ANCHOR_PLY, position_chunk: int = 400_000,
                komi_fix: bool = True, archives: Optional[Sequence[str]] = None,
                files: Optional[Sequence[str]] = None,
                stop_hashes: Optional[np.ndarray] = None,
                log: Callable[[str], None] = _log) -> Tuple[CorpusScan, AnchorIndex]:
    """扫全部 SGF，取 `KM/RE/RU` 并建 `pos_hash` 排序表。

    `archives` / `files` 是 `plan_sources` 的产物；给了它们就不必再枚举 `sgf_dirs`。

    `stop_hashes` 是待命中的锚点散列（uint64）；每刷一批就查一次，命中即剔除，
    全空则提前收工。**只给 `--limit-games` 用**：小规模探针时 50 个锚点通常在前几百
    个 SGF 就全命中了，不必扫完 17 万局。调用方**不要**指望看到它被就地缩短 ——
    缩短靠的是重新绑定，所以它必须是本函数的局部视图（调用方传一份副本）。

     这里**必须由 `scan_corpus` 自己查**：散列表是它的局部产物，调用方在
    `scan_corpus` 返回之前拿不到它 —— 让调用方闭包捕获 `index` 会永远拿到 `None`。

    全量模式传 `None` ⇒ 结果与扫描顺序无关。
    """
    parser = parser or SGFParser()
    if archives is None or files is None:
        archives, files = plan_sources(sgf_dirs, log=log)

    scan = CorpusScan()
    komi_l: List[float] = []
    score_l: List[Optional[float]] = []
    rules_l: List[int] = []
    resign_l: List[bool] = []
    resign_side_l: List[int] = []
    recls_l: List[int] = []
    has_komi_l: List[bool] = []
    ru_cls = [0, 0, 0, 0]

    index = AnchorIndex()
    bh: List[np.ndarray] = []
    tp_buf: List[np.ndarray] = []
    ko_buf: List[np.ndarray] = []
    idx_buf: List[np.ndarray] = []
    off_buf: List[np.ndarray] = []
    sig = np.empty(0, np.uint64)
    pending = 0

    stats = dict(n_files=0, n_parse_fail=0, n_not_19x19=0, n_no_anchor=0,
                 n_komi_missing=0, n_komi_fixed=0, n_score=0, n_draw=0,
                 n_resign=0, n_re_unknown=0)
    t0 = time.time()

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        h = pos_hash_block(np.concatenate(bh), np.concatenate(tp_buf),
                           np.concatenate(ko_buf))
        gi = np.concatenate(idx_buf)
        go = np.concatenate(off_buf)
        is_sig = go < 0                      # 见 SIG_OFFSET
        if is_sig.any():
            # 按**局号的最大值 +1** 定尺寸，不能按批内行数 —— 批里每个局贡献
            # 21 行（20 前缀 + 1 签名），按行数会把 sig 撑成 21 倍长，尾部全是
            # 未初始化的垃圾（症状是交叉校验随机报「不一致」）。
            need = int(gi[is_sig].max()) + 1
            if sig.size < need:
                sig.resize(need, refcheck=False)
            sig[gi[is_sig]] = h[is_sig]
        index.add(h[~is_sig], gi[~is_sig], go[~is_sig])
        for lst in (bh, tp_buf, ko_buf, idx_buf, off_buf):
            lst.clear()
        pending = 0

    for label, raw in iter_sgf_bytes(archives, files):
        stats['n_files'] += 1
        try:
            text = raw.decode('utf-8', 'ignore')
            game = parser.parse_string(text)
        except Exception:
            game = None
        if game is None:
            stats['n_parse_fail'] += 1
            continue
        if game.board_size != BOARD:
            stats['n_not_19x19'] += 1
            continue

        props = game.properties
        km, fixed = parse_komi(props, fix_outlier=komi_fix)
        if fixed:
            stats['n_komi_fixed'] += 1
        if not props.get('KM'):
            stats['n_komi_missing'] += 1
        ri = game.result_info
        if ri.is_resign:
            cls = RE_CLASS_RESIGN
            stats['n_resign'] += 1
        elif ri.is_draw:
            cls = RE_CLASS_DRAW
            stats['n_draw'] += 1
        elif ri.score is not None:
            cls = RE_CLASS_SCORE
            stats['n_score'] += 1
        else:
            cls = RE_CLASS_UNKNOWN
            stats['n_re_unknown'] += 1
        ru = props.get('RU')
        rc = classify_rules(ru)
        ru_cls[rc] += 1

        boards, tps, kos, offs = replay_anchor_positions(game, max_offset=max_offset)
        me = len(komi_l)
        if boards.shape[0] == 0:
            stats['n_no_anchor'] += 1
        else:
            bh.append(boards)
            tp_buf.append(tps)
            ko_buf.append(kos)
            idx_buf.append(np.full(offs.size, me, np.int32))
            off_buf.append(offs)
            # 末前缀再挂一次，哨兵标记为「这一行是签名」（见 CorpusScan.sig）
            bh.append(boards[-1:])
            tp_buf.append(tps[-1:])
            ko_buf.append(kos[-1:])
            idx_buf.append(np.full(1, me, np.int32))
            off_buf.append(np.array([SIG_OFFSET], np.int8))
            pending += int(offs.size) + 1

        komi_l.append(km)
        score_l.append(ri.score)
        rules_l.append(int(parse_rules(ru)))
        resign_l.append(bool(ri.is_resign))
        # 方向只在这一行能拿到（`ResultInfo.resign_side` 是 None/0/1），
        # 而落盘侧用 -1 表示「非认输」—— 因为 0 在那边有实义（黑认输）。
        resign_side_l.append(-1 if ri.resign_side is None else int(ri.resign_side))
        recls_l.append(cls)
        has_komi_l.append('KM' in props)

        if pending >= position_chunk:
            flush()
            if stop_hashes is not None:
                got, _cnt, hit_h = index.lookup(stop_hashes)
                stop_hashes = stop_hashes[hit_h < 0]   # 重新绑定（局部视图）
                if stop_hashes.size == 0:
                    log(f'[sidecar] 锚点已全部命中，提前收工'
                        f'（扫了 {stats["n_files"]} 个 SGF）')
                    break

        if stats['n_files'] % 20000 == 0:
            log(f'[sidecar] 已扫 {stats["n_files"]} 个 SGF，{time.time()-t0:.0f}s')

    flush()

    scan.komi = np.asarray(komi_l, np.float16)
    scan.score = np.asarray([np.nan if s is None else s for s in score_l], np.float16)
    scan.rules = np.asarray(rules_l, np.int8)
    scan.resign = np.asarray(resign_l, np.bool_)
    scan.resign_side = np.asarray(resign_side_l, np.int8)
    scan.re = np.asarray(recls_l, np.int8)
    scan.has_komi = np.asarray(has_komi_l, np.bool_)
    # 提前收工时尾部几局没进过 flush ⇒ 没有签名。补零占位（它们也没有任何锚点
    # 命中过，读 `sig` 只会取到 0，但数组形状必须与 `komi` 对齐才不出错）。
    if sig.size < len(komi_l):
        sig = np.concatenate([sig, np.zeros(len(komi_l) - sig.size, np.uint64)])
    scan.sig = sig
    scan.stats = stats
    scan.stats.update(ru_default=ru_cls[RU_CLASS_DEFAULT], ru_area=ru_cls[RU_CLASS_AREA],
                      ru_territory=ru_cls[RU_CLASS_TERRITORY],
                      ru_unrecognized=ru_cls[RU_CLASS_UNRECOGNIZED],
                      n_sgf=len(komi_l), n_positions=index.n_positions)
    index.n_sgf = len(komi_l)
    log(f'[sidecar] 语料扫描完成：{stats["n_files"]} 个 SGF / {len(komi_l)} 局入库 / '
        f'{index.n_positions} 个锚点前缀 / {time.time()-t0:.0f}s')
    return scan, index


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def build_sidecar(dataset: str, sgf_dirs: Sequence[str], out: str,
                  materialized_dir: str, limit_games: Optional[int] = None,
                  anchor_ply: int = ANCHOR_PLY, komi_fix: bool = True,
                  position_chunk: int = 400_000, force_rematerialize: bool = False,
                  reuse_scan: bool = False,
                  log: Callable[[str], None] = _log) -> Dict[str, object]:
    """三步走完并落盘 `games.npz`，返回覆盖报告 dict。

    Steps（都可断点）:
      1. 切分行区间 → `<materialized_dir>/game_spans.npz`（每次重算，很便宜，
         写出来是为了中途被 kill 时能看见切分结果）；
      2. 扫语料 → `<out>.scan.npz`（**始终写**，`reuse_scan=True` 时才读；
         换语料后必须关掉这个开关，否则旧散列表会静默挂错元数据）；
      3. 锚点匹配 → `<out>`。
    """
    t_start = time.time()
    log(f'[sidecar] 主数据集（只读）：{dataset}')
    mat = ensure_materialized(dataset, materialized_dir, force=force_rematerialize, log=log)

    # ---- step 1: 行区间（可断点） ----
    spans_path = os.path.join(materialized_dir, 'game_spans.npz')
    gids = np.load(mat['game_ids'], mmap_mode='r')
    run_ids, starts, ends = contiguous_runs(np.asarray(gids))
    n_rows_total = int(gids.size)
    if limit_games:
        keep = min(int(limit_games), run_ids.size)
        run_ids, starts, ends = run_ids[:keep], starts[:keep], ends[:keep]
        log(f'[sidecar] --limit-games {limit_games}：只取行序最前的 {keep} 段')
    np.savez(spans_path, run_ids=run_ids, starts=starts, ends=ends)
    log(f'[sidecar] step1 行区间：{run_ids.size} 段 / {n_rows_total} 行 → {spans_path}')

    uniq_ids, slot_of_run = np.unique(run_ids, return_inverse=True)
    slot_of_run = slot_of_run.astype(np.int64)
    G = int(uniq_ids.size)
    log(f'[sidecar] distinct game_ids = {G}（sidecar 行数）')

    rows = anchor_rows(starts, ends, anchor_ply)
    want = dataset_anchor_hashes(mat, rows, log=log)

    # ---- step 2: 语料扫描（可断点） ----
    archives, files = plan_sources(sgf_dirs, log=log)
    scan_cache = os.path.splitext(out)[0] + '.scan.npz'
    scan = None
    index = None
    if reuse_scan and os.path.isfile(scan_cache):
        scan, index = _load_scan_cache(scan_cache)
        if scan is not None and index is not None:
            log(f'[sidecar] step2 复用扫描缓存 {scan_cache}（{scan.stats.get("n_sgf")} 局）')
        else:
            scan = index = None
    if scan is None:
        scan, index = scan_corpus(
            archives=archives, files=files, max_offset=anchor_ply,
            position_chunk=position_chunk, komi_fix=komi_fix,
            stop_hashes=want.copy() if limit_games else None, log=log)
        _save_scan_cache(scan_cache, scan, index)

    # ---- step 3: 匹配 ----
    matched_sgf, counts, hit = index.lookup(want)
    n_runs = int(run_ids.size)
    n_hit_run = int(hit.sum())
    run_len = ends - starts

    # 交叉校验：主锚点与第二个锚点的**命中集合**是否相交 ⇒ 该段是否跨了局
    # （`game_ids` 被复用）。比集合而不是比「第一条命中」，见 cross_consistency。
    cross_rows, cross_ok = anchor_rows_cross(starts, ends, anchor_ply)
    n_disagree = n_inconclusive = n_amb_p = n_amb_c = 0
    if bool(cross_ok.any()):
        want_c = dataset_anchor_hashes(mat, cross_rows[cross_ok], log=log)
        n_disagree, n_inconclusive, n_amb_p, n_amb_c = cross_consistency(
            index, want[cross_ok], want_c)
    # 元数据挂错的直接度量：主锚点的局面被**不同的棋**走到了几次
    sig_counts = distinct_signature_counts(index, want, scan.sig)
    n_multi_sig = int(((sig_counts > 1) & hit).sum())

    # id 被复用时，多段行映射到同一个 slot。sidecar 按 id 索引 ⇒ 必须给同一个值。
    # 取**行数更多**的那段（完整的局，而非 `play()` 失败留下的残段），并报冲突数。
    cand = np.flatnonzero(hit)
    sgf_of_best = np.full(G, -1, np.int64)      # slot -> 语料侧的局号
    if cand.size:
        # 先按 slot 升序、行数降序排，再对每个 slot 取第一条命中的
        key = np.lexsort((-run_len[cand], slot_of_run[cand]))
        cand = cand[key]
        slot_sorted = slot_of_run[cand]
        first = np.flatnonzero(np.concatenate(([True], slot_sorted[1:] != slot_sorted[:-1])))
        sgf_of_best[slot_sorted[first]] = matched_sgf[cand[first]]
    n_conflict = int(n_hit_run - int((sgf_of_best >= 0).sum()))

    g_komi = np.zeros(G, np.float16)
    g_score = np.full(G, np.nan, np.float16)
    g_rules = np.full(G, RULES_DEFAULT, np.int8)
    g_resign = np.zeros(G, np.bool_)
    g_re = np.full(G, RE_CLASS_UNKNOWN, np.int8)
    # 未匹配的局回落到 `-1`（「非认输」），**不是** 0 —— 0 有实义（黑认输）。
    # 回落成 0 会让 `derive_outcome` 把它们判成黑认输，凭空造出一批标签。
    g_resign_side = np.full(G, -1, np.int8)
    # 左索引是 **slot**（`sidecar[game_ids]` 的下标），右索引才是**语料局号**。
    # 两者不是一回事：语料 17 万局、sidecar 16 万局，且 slot 只按升序排。
    # 写反了就是 `IndexError`，或更糟 —— 在大小恰好相同时静默串味。
    sel = sgf_of_best >= 0
    slots = np.flatnonzero(sel)
    src = sgf_of_best[sel]
    g_komi[slots] = scan.komi[src]
    finite = np.isfinite(scan.score[src])
    g_score[slots[finite]] = scan.score[src[finite]]
    g_rules[slots] = scan.rules[src]
    g_resign[slots] = scan.resign[src]
    g_re[slots] = scan.re[src]
    g_resign_side[slots] = scan.resign_side[src]

    n_matched = int(sel.sum())
    n_unmatched = G - n_matched

    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    np.savez(out, g_komi=g_komi, g_score=g_score, g_rules=g_rules, g_resign=g_resign,
             g_re=g_re, g_resign_side=g_resign_side)

    report: Dict[str, object] = dict(
        out=out, n_games=G, n_rows=n_rows_total, n_runs=n_runs,
        n_matched=n_matched, n_unmatched=n_unmatched,
        match_rate=(n_matched / G if G else 0.0),
        n_runs_hit=n_hit_run, n_conflict_id_reuse=n_conflict,
        n_anchor_ambiguous=n_amb_p, n_anchor_ambiguous_cross=n_amb_c,
        n_anchor_multi_game=n_multi_sig,
        n_anchor_disagree=n_disagree, n_anchor_inconclusive=n_inconclusive,
        n_sgf=scan.stats.get('n_sgf', 0),
        n_sgf_files=scan.stats.get('n_files', 0),
        n_sgf_positions=scan.stats.get('n_positions', 0),
        n_komi_missing=scan.stats.get('n_komi_missing', 0),
        n_komi_outlier_fixed=scan.stats.get('n_komi_fixed', 0),
        n_re_score=scan.stats.get('n_score', 0),
        n_re_draw=scan.stats.get('n_draw', 0),
        n_re_resign=scan.stats.get('n_resign', 0),
        n_re_unknown=scan.stats.get('n_re_unknown', 0),
        ru_default=scan.stats.get('ru_default', 0), ru_area=scan.stats.get('ru_area', 0),
        ru_territory=scan.stats.get('ru_territory', 0),
        ru_unrecognized=scan.stats.get('ru_unrecognized', 0),
        seconds=time.time() - t_start,
    )
    print_report(report, log=log)
    return report


def print_report(rep: Dict[str, object], log: Callable[[str], None] = _log) -> None:
    """覆盖报告。 **未匹配的局不许静默**：数字必须打在 stdout 上。"""
    G = int(rep['n_games'])  # type: ignore[arg-type]
    m, u = int(rep['n_matched']), int(rep['n_unmatched'])  # type: ignore[arg-type]
    log('')
    log('=' * 66)
    log('局级 sidecar 覆盖报告')
    log('=' * 66)
    log(f'  产物            {rep["out"]}   G={G}')
    log(f'  主数据集        {rep["n_rows"]} 行 / {rep["n_runs"]} 段连续行区间')
    log(f'  语料            {rep["n_sgf_files"]} 个 SGF 文件 → {rep["n_sgf"]} 局入库'
        f' / {rep["n_sgf_positions"]} 个锚点前缀')
    log(f'  匹配            {m} / {G}  ({100.0 * m / max(G, 1):.2f}%)')
    log(f'  未匹配          {u} / {G}  ({100.0 * u / max(G, 1):.2f}%)'
        f'   → g_komi=0, g_score=NaN, g_rules={RULES_DEFAULT:#04x}, g_resign=False, '
        f'g_re={RE_CLASS_UNKNOWN}（无结果）, g_resign_side=-1（非认输）')
    log(f'  锚点段命中      {rep["n_runs_hit"]} 段')
    log(f'                  布局歧义：主锚点 {rep["n_anchor_ambiguous"]} 段 / '
        f'交叉锚点 {rep["n_anchor_ambiguous_cross"]} 段（歧义时取最早那份）')
    log(f' 元数据有风险：{rep["n_anchor_multi_game"]} 段的局面被'
        f'**不同的棋**走到 ⇒ 取的未必是本局的 KM/RE/RU')
    log(f'  锚点交叉校验    跨局 {rep["n_anchor_disagree"]} 段 / '
        f'无法判定 {rep["n_anchor_inconclusive"]} 段')
    log('                  （跨局 ⇒ 该段含两局的行，即 `game_id` 被复用；本数据集应为 0）')
    log(f'  id 复用冲突     {rep["n_conflict_id_reuse"]} 段（取行数更多的那一段）')
    log('  ---- 语料的 RE 形态（分母是入库局数） ----')
    ns = max(int(rep['n_sgf']), 1)  # type: ignore[arg-type]
    for key, lab in (('n_re_score', '有分差'), ('n_re_resign', '认输'),
                     ('n_re_draw', '和棋'), ('n_re_unknown', '无结果/未识别')):
        v = int(rep[key])  # type: ignore[arg-type]
        log(f'    {lab:<12} {v:>8}  ({100.0*v/ns:.2f}%)')
    log('  ---- 语料的 RU 形态 ----')
    for key, lab in (('ru_default', '无 RU → 默认'), ('ru_area', 'Chinese/AREA'),
                     ('ru_territory', 'Japanese/TERRITORY'),
                     ('ru_unrecognized', '未识别 → 默认')):
        v = int(rep[key])  # type: ignore[arg-type]
        log(f'    {lab:<22} {v:>8}  ({100.0*v/ns:.2f}%)')
    log('  ---- KM ----')
    log(f'    属性缺失      {rep["n_komi_missing"]:>8}')
    log(f'    ×100 离群修正 {rep["n_komi_outlier_fixed"]:>8}   (KM[650]→6.5 等)')
    log(f'  用时            {float(rep["seconds"]):.0f}s')  # type: ignore[arg-type]
    log('=' * 66)


def _save_scan_cache(path: str, scan: CorpusScan, index: AnchorIndex) -> None:
    index._finalize()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    keys = sorted(scan.stats)
    np.savez(path,
             komi=scan.komi, score=scan.score, rules=scan.rules, resign=scan.resign,
             re=scan.re, resign_side=scan.resign_side,
             has_komi=scan.has_komi, sig=scan.sig,
             pos_hash=index._sh, sgf_of_pos=index._ss, offset_of_pos=index._so,
             stat_keys=np.array(keys), stat_vals=np.array([scan.stats[k] for k in keys],
                                                          np.int64))


def _load_scan_cache(path: str) -> Tuple[Optional[CorpusScan], Optional[AnchorIndex]]:
    try:
        z = np.load(path, allow_pickle=False)
        scan = CorpusScan()
        scan.komi = z['komi']
        scan.score = z['score']
        scan.rules = z['rules']
        scan.resign = z['resign']
        scan.re = z['re']
        # 这一列**不能**从旧的 `resign`（bool）推出来 —— bool 没有方向。
        # 缺键就抛 KeyError，被下面的 except 吞成 `(None, None)` ⇒ 调用方**重扫**。
        # 宁可重扫一遍 133,604 局，也不能静默填一个方向错乱的默认值。
        scan.resign_side = z['resign_side']
        scan.has_komi = z['has_komi']
        scan.sig = z['sig']
        scan.stats = {str(k): int(v) for k, v in zip(z['stat_keys'], z['stat_vals'])}
        index = AnchorIndex()
        index.add(z['pos_hash'], z['sgf_of_pos'], z['offset_of_pos'])
        index.n_sgf = int(scan.komi.size)
    except (OSError, ValueError, KeyError):
        return None, None
    return scan, index


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='生成局级 sidecar games.npz（spec §5.3 / pipeline D0）',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default=os.path.join('data', 'sgf_19x19_full.npz'),
                    help='主数据集 npz（只读）')
    ap.add_argument('--sgf-dirs', nargs='+',
                    default=[os.path.join('data'), os.path.join('data', 'games', 'games')],
                    help='语料根目录（递归找 *.sgf 与 *.tgz，两者都会覆盖）')
    ap.add_argument('--out', default=os.path.join('data', 'labels', 'games.npz'))
    ap.add_argument('--materialized-dir', default=os.path.join('tmp', 'materialized'),
                    help='主数据集列的 .npy 落地目录（boards 有 12.3 GB，务必与源同盘）')
    ap.add_argument('--limit-games', type=int, default=0,
                    help='只取行序最前的 N 局（调试用；同时允许扫语料提前收工）')
    ap.add_argument('--anchor-ply', type=int, default=ANCHOR_PLY)
    ap.add_argument('--position-chunk', type=int, default=400_000,
                    help='散列表每多少个前缀刷一次盘（越大峰值内存越高）')
    ap.add_argument('--no-komi-outlier-fix', action='store_true',
                    help=f'不把 |KM| > {KOMI_OUTLIER_ABS:g} 的 ×100 离群值除以 100')
    ap.add_argument('--force-rematerialize', action='store_true',
                    help='忽略已有的 .npy 缓存，重新从 npz 落盘（12.3 GB，很慢）')
    ap.add_argument('--reuse-scan', action='store_true',
                    help='复用 `<out>.scan.npz` 里的语料扫描结果（换语料后别用）')
    args = ap.parse_args(argv)

    build_sidecar(
        dataset=args.dataset, sgf_dirs=args.sgf_dirs, out=args.out,
        materialized_dir=args.materialized_dir,
        limit_games=args.limit_games or None, anchor_ply=args.anchor_ply,
        komi_fix=not args.no_komi_outlier_fix,
        position_chunk=args.position_chunk,
        force_rematerialize=args.force_rematerialize,
        reuse_scan=args.reuse_scan,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
