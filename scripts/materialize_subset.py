#!/usr/bin/env python3
"""**流式**抽取 npz 的子集并落成 `.npy`（供 memmap 随机读）—— 不占全量磁盘。

为什么需要它
------------
`bench_v7_features.py` 的组件分解要读 `boards` 等列做**随机读**（邻行 gather）。
`src/data/kata_label_join.py::materialize_dataset` 会把**整个** `boards` 列物化
成 `.npy` ⇒ **12.3 GB** 磁盘（34.2M 行 × 361 B）。本机/云端都不一定放得下。

而组件分解其实**不需要全量数据**：它要回答的是「一行要多少 CPU 时间」，
一个**跨全量等距抽样**的子集就足够（等距 ⇒ 复杂度分布与全量一致，不像
「取前 N 行」那样偏简单开局）。

本脚本的做法
------------
`np.load(npz)['boards']` 会把整个成员解压进内存（12.3 GB）—— 那才是真正的
内存杀手。这里改用 `zipfile` **流式**读成员：
  · 峰值内存 = 一个 chunk（默认 64 MB），与总行数无关；
  · 边解压边按 stride 抽样，只把**抽中的行**写进 `.npy`；
  · 磁盘 = 抽中行数 × 行宽（1M 行 boards ≈ 361 MB）。

用法
----
    # 跨全量等距抽 100 万行（stride 自动算），落成 6 个 .npy
    python scripts/materialize_subset.py \
        --npz data/sgf_19x19_full.npz --out-dir tmp/mat_subset --rows 1000000

    # 然后直接跑组件分解（跳过 materialize 阶段）
    python scripts/bench_v7_features.py --phase components \
        --mat-dir tmp/mat_subset --out benchmarks/bench_v7_subset.json
"""
from __future__ import annotations

import argparse
import ast
import os
import struct
import sys
import zipfile

import numpy as np

#: 组件分解需要的列（与 bench_v7_features 的 open_columns 一致）
DEFAULT_KEYS = ('boards', 'to_play', 'ko', 'game_ids', 'my_hist', 'op_hist')


def _read_npy_header(f):
    """从 npy 流的**开头**读出 (dtype, shape)；返回后流位置就在数据区起点。"""
    magic = f.read(6)
    if magic != b'\x93NUMPY':
        raise ValueError('不是 npy 格式')
    ver = f.read(2)
    major = ver[0]
    if major == 1:
        hlen = struct.unpack('<H', f.read(2))[0]
    else:
        hlen = struct.unpack('<I', f.read(4))[0]
    header = f.read(hlen).decode('latin1')
    d = ast.literal_eval(header.strip())   # npy 头是字面量 dict
    return np.dtype(d['descr']), tuple(d['shape']), d.get('fortran_order', False)


def _stream_subset_member(zf, name, out_path, keep_idx, chunk_bytes):
    """把 npz 成员**流式**读出来，只写 keep_idx 里的行。"""
    with zf.open(name + '.npy') as f:
        dtype, shape, fortran = _read_npy_header(f)
        if fortran:
            raise ValueError('%s 是 Fortran 序，暂不支持' % name)
        if len(shape) == 0:
            raise ValueError('%s 是 0 维' % name)
        total = shape[0]
        row_items = 1
        for s in shape[1:]:
            row_items *= int(s)
        row_bytes = row_items * dtype.itemsize
        n_keep = len(keep_idx)
        out_shape = (n_keep,) + shape[1:]
        mm = np.lib.format.open_memmap(out_path, mode='w+', dtype=dtype,
                                       shape=out_shape)
        # keep_idx 必须升序（我们按 stride 抽，本来就是升序）
        ki = 0
        pos = 0            # 已读到的行号
        written = 0
        rows_per_chunk = max(1, chunk_bytes // max(1, row_bytes))
        while ki < n_keep and pos < total:
            want = keep_idx[ki]
            if want >= total:
                break
            # 丢掉 [pos, want) 这些不需要的行
            if want > pos:
                skip = (want - pos) * row_bytes
                while skip > 0:
                    step = min(skip, 1 << 24)
                    f.read(step)
                    skip -= step
                pos = want
            # 读这一行（或连续若干行）
            take = 1
            while ki + take < n_keep and keep_idx[ki + take] == want + take:
                take += 1
            buf = f.read(take * row_bytes)
            if len(buf) < take * row_bytes:
                break
            arr = np.frombuffer(buf, dtype=dtype).reshape((take,) + shape[1:])
            mm[written:written + take] = arr
            written += take
            pos += take
            ki += take
        mm.flush()
        del mm
        return written, out_shape


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', required=True, help='源 npz（默认用主数据集时给 data/...）')
    ap.add_argument('--out-dir', required=True, help='.npy 输出目录')
    ap.add_argument('--rows', type=int, default=1_000_000,
                    help='要抽多少行（跨全量等距）')
    ap.add_argument('--stride', type=int, default=0,
                    help='抽样步长（0 = 按 --rows 自动算）')
    ap.add_argument('--keys', default=','.join(DEFAULT_KEYS),
                    help='要抽的列（逗号分隔）')
    ap.add_argument('--chunk-mb', type=int, default=64,
                    help='流式解压的 chunk 大小（MB）—— 决定峰值内存')
    args = ap.parse_args(argv)

    keys = [k for k in args.keys.split(',') if k]
    os.makedirs(args.out_dir, exist_ok=True)

    # 先看总行数（读 boards 的头即可，不用解压）
    with zipfile.ZipFile(args.npz) as zf:
        names = set(zf.namelist())
        total = None
        with zf.open('boards.npy') as f:
            _dt, shape, _fo = _read_npy_header(f)
            total = shape[0]
    print('源 npz: %s  boards 总行数 = %d' % (args.npz, total), flush=True)

    stride = args.stride or max(1, total // max(1, args.rows))
    keep = np.arange(0, total, stride, dtype=np.int64)
    print('等距抽样：stride=%d ⇒ 取 %d 行（%.3f%%）'
          % (stride, len(keep), len(keep) / total * 100), flush=True)

    out = {}
    with zipfile.ZipFile(args.npz) as zf:
        for k in keys:
            if k + '.npy' not in names:
                print('  [跳过] 源 npz 无此列: %s' % k, flush=True)
                continue
            p = os.path.join(args.out_dir, '%s.npy' % k)
            n, shp = _stream_subset_member(
                zf, k, p, keep, args.chunk_mb * 1024 * 1024)
            mb = os.path.getsize(p) / 1e6
            print('  %-10s → %-28s %8d 行  %7.1f MB' % (k, p, n, mb), flush=True)
            out[k] = p

    total_mb = sum(os.path.getsize(p) for p in out.values()) / 1e6
    print('\n合计磁盘 %.1f MB（对比全量物化 boards 单独就要 12.3 GB）' % total_mb)
    print('下一步：')
    print('  python scripts/bench_v7_features.py --phase components \\')
    print('      --mat-dir %s --out benchmarks/bench_v7_subset.json' % args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
