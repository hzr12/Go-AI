"""把 KataGo 软标签 join 到数据集行号，产出训练侧可直接吃的索引缓存。

为什么要缓存
------------
join 要算 `full.npz` 全部 3420 万行的位置 hash，**实测 127 秒**。每次启动训练
都重算太浪费；而 `pos_hash` 只依赖 `boards/to_play/ko` 三列（数据不变则结果不变），
所以结果可以落盘复用。

产出
----
`<out>.npz`：
    idx     int64  (M,)   数据集行号（可重复：同一局面多行时每个行号各占一条）
    policy  float16(M,362) 对应的 KataGo 分布
    n_hit   int64  ()     命中的标签数（去重前）
    n_rows  int64  ()     命中的行数（去重后）
    hit_rate float64()    n_hit / 标签总数

用法
----
    python scripts/build_soft_index.py \
        --data data/sgf_19x19_full.npz \
        --labels tmp/bench/verify.npz \
        --out data/labels/soft_index_verify.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pos_hash_block(boards, to_play, ko):
    """位置 hash（与 scripts/probe0_join.py 同口径：board+to_play+ko）。

    ⚠ 权重必须是**固定种子**的常量：改种子会让此前存下的所有索引缓存失效。
    向量化线性散列，3420 万行约 60~130 秒（实测 127s）。
    """
    B = boards.shape[0]
    idx = (boards.reshape(B, -1).astype(np.int8) + np.int8(1)).astype(np.uint64)
    h1 = idx.dot(_W1)
    h2 = idx.dot(_W2)
    out = h1 * np.uint64(0x9E3779B97F4A7C15) + h2 + _SEED
    tp = to_play.astype(np.int8).astype(np.uint64)
    ko16 = ko.astype(np.int16).astype(np.uint64)
    out = out + tp * np.uint64(0xC2B2AE3D27D4EB4F) + \
        (ko16 + np.uint64(1)) * np.uint64(0x165667B19E3779F9)
    return out


_HASH_SEED = 0x5EED19191919
_W1 = np.random.default_rng(_HASH_SEED).integers(1, 2**63, 361, dtype=np.uint64)
_W2 = np.random.default_rng(_HASH_SEED + 1).integers(1, 2**63, 361, dtype=np.uint64)
_SEED = np.uint64(0x243F6A8885A308D3)


def main():
    ap = argparse.ArgumentParser(description="构建软标签 -> 数据集行号 的 join 索引")
    ap.add_argument("--data", default=os.path.join("data", "sgf_19x19_full.npz"))
    ap.add_argument("--labels", required=True, help="打标签产物 npz（verify.npz 等）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()

    lab = np.load(args.labels)
    if "policy" not in lab or "pos_hash" not in lab:
        raise SystemExit(f"{args.labels} 缺 policy/pos_hash 字段")
    lab_h = lab["pos_hash"]
    lab_p = lab["policy"].astype(np.float16)
    print(f"标签: {lab_p.shape[0]} 个局面, policy {lab_p.shape}")
    # hash -> 标签槽位（同一 hash 出现多次取第一条）
    want = {}
    for k, h in enumerate(lab_h.tolist()):
        want.setdefault(int(h), k)
    print(f"去重后待查 hash: {len(want)}")

    z = np.load(args.data)
    N = z["boards"].shape[0]
    print(f"数据: {N} 行, 扫描中 ...")
    found = {}          # hash -> 该 hash 命中的所有行号
    t0 = time.time()
    for s in range(0, N, args.chunk):
        e = min(s + args.chunk, N)
        h = pos_hash_block(z["boards"][s:e], z["to_play"][s:e], z["ko"][s:e])
        for i in range(h.size):
            hi = int(h[i])
            if hi in want:
                found.setdefault(hi, []).append(s + i)
        if (s // args.chunk) % 25 == 0:
            el = time.time() - t0
            print(f"  {e}/{N}  {el:.0f}s  命中 {len(found)}/{len(want)}", flush=True)

    idx_list, pol_list = [], []
    for hi, rows in found.items():
        k = want[hi]
        for r in rows:
            idx_list.append(r)
            pol_list.append(lab_p[k])
    idx = np.asarray(idx_list, dtype=np.int64)
    pol = (np.stack(pol_list) if pol_list
            else np.zeros((0, lab_p.shape[1]), np.float16))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, idx=idx, policy=pol,
             n_hit=np.int64(len(found)), n_rows=np.int64(idx.size),
             hit_rate=np.float64(len(found) / max(len(want), 1)),
             n_label=np.int64(len(want)))
    el = time.time() - t0
    print(f"\n写入 {args.out}")
    print(f"  标签 {len(want)}  命中局面 {len(found)} "
          f"({100*len(found)/max(len(want),1):.2f}%)  覆盖数据行 {idx.size}")
    print(f"  用时 {el:.0f}s")
    if idx.size:
        s = pol.astype(np.float32).sum(1)
        print(f"  policy 每行和 {s.min():.4f}~{s.max():.4f}（应≈1）")


if __name__ == "__main__":
    main()
