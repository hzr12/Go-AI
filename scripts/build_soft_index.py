"""把 KataGo 软标签 join 到数据集行号，产出训练侧可直接吃的索引缓存。

为什么要缓存
------------
join 要算 `full.npz` 全部 3420 万行的位置 hash，**实测 127 秒**。每次启动训练
都重算太浪费；而 `pos_hash` 只依赖 `boards/to_play/ko` 三列（数据不变则结果不变），
所以结果可以落盘复用。

🔴 **缓存 key 必须覆盖全部四个输入**（A5，实现在 `src/data/kata_label_join.py`）：

    (主 npz 的 行数 / distinct game_ids / mtime / size,
     标签 npz 的 行数 / distinct pos_hash / mtime / size,
     max_repeats,
     散列口径版本号 + 从 `pos_hash` 活常量现算的指纹)

任何一项变了都必须重建。旧缓存被复用时不会报错，只会**静默挂错标签** ——
症状是「命中率 0」，与「这批局面真的没标签」长得一模一样，是本项目最危险的
失败模式。所以命中陈旧缓存时**明确告警并逐项列出差异**，绝不静默复用。

产出
----
`<out>.npz`：
    idx     int64  (M,)   数据集行号（可重复：同一局面多行时每个行号各占一条）
    policy  float16(M,362) 对应的 KataGo 分布
    n_hit   int64  ()     命中的**局面**数（去重前）
    n_rows  int64  ()     命中的**行**数
    hit_rate float64()    n_hit / 待查局面数
    cache_key / cache_meta  缓存指纹（缺这两项的旧文件一律当没有缓存）

用法
----
    python scripts/build_soft_index.py \
        --data data/sgf_19x19_full.npz \
        --labels data/labels/kata_labels.npz \
        --materialized-dir tmp/materialized \
        --out data/labels/soft_index.npz

⚠ **`--materialized-dir` 在小内存机器上是必需的，不是可选优化。**
  `data/sgf_19x19_full.npz` 磁盘上只有 447 MB，但**解压后 13.4 GB**
  （`boards` 一列就 12.3 GB）—— 它是 `savez_compressed`，**压缩 npz 无法
  memmap**，不给 materialized_dir 就得整份进内存。本机实测总内存 13.9 GB，
  整份解压必 OOM。

  先造一次（只需一次，之后 memmap 复用）：
      python -c "import sys; sys.path.insert(0,'.'); \\
        from src.data.kata_label_join import materialize_dataset; \\
        materialize_dataset('data/sgf_19x19_full.npz','tmp/materialized')"
  产物约 11.7 GiB（`boards.npy` 占 11.2 GiB），`tmp/materialized/` 已在仓库里备好。

  ⚠ materialize 必须**完整**：`.npy` 被截断时 `np.load(mmap_mode='r')` 照样
    成功，只是行数偏少 ⇒ join 静默漏掉尾部所有命中。所以跑完后核对行数等于
    主 npz 的行数（34,202,713）。

value 列不可用
--------------
`kata_labels.npz` 还有 `root_win` / `score_mean` / `score_stdev` 三列，
**本脚本刻意不输出它们**：实测 `score_mean` 在 21.9 万行上**全为 0**、
`score_stdev` 高达 82 分（19 路盘的目数不确定度不可能这么大）——
`maxVisits=36` 这个预算下 KataGo 的目数估计根本没校准。
要目数标签得把 visits 提到几百到几千（算力 15~50 倍）。
只有 `policy` 是可用的，故训练侧目前是 policy-only 蒸馏。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 散列唯一真相（与打标签端共用，见 pos_hash_block 的 docstring）
from src.data.pos_hash import pos_hash_block as _pos_hash_block  # noqa: E402
from src.data.kata_label_join import (  # noqa: E402
    build_soft_index,
    hash_spec_fingerprint,
)

CACHE_SCHEMA_HINT = 1


def pos_hash_block(boards, to_play, ko):
    """位置 hash —— **直接复用 `src.data.pos_hash` 的唯一实现**。

    ⚠ 不要在这里重抄一份散列实现：口径一旦与打标签端（`label_sfg.py` →
      `probe0_join` → `pos_hash`）漂移，join 会**静默变成空**，而症状看起来
      像「这批局面真的没标签」。单点真相在 `src/data/pos_hash.py`。
    """
    return _pos_hash_block(boards, to_play, ko)


def main():
    ap = argparse.ArgumentParser(description="构建软标签 -> 数据集行号 的 join 索引")
    ap.add_argument("--data", default=os.path.join("data", "sgf_19x19_full.npz"))
    ap.add_argument("--labels", required=True, help="打标签产物 npz（verify.npz 等）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--max-repeats", type=int, default=None,
                    help="同一局面最多挂到几个数据行；默认（None）= 全部挂，"
                         "即历史口径：同一局面在语料里出现几次就挂几次")
    ap.add_argument("--no-cache", action="store_true",
                    help="忽略已有产物的缓存指纹，强制重算（默认按 key 命中复用）")
    ap.add_argument("--materialized-dir", default=None,
                    help="materialize_dataset() 的输出目录：给了它才不会把 12.3 GB 的 "
                         "boards 整份解压进内存（强烈建议给）")
    args = ap.parse_args()

    print(f"散列口径: version={CACHE_SCHEMA_HINT} fingerprint="
          f"{hash_spec_fingerprint()[:12]}（改了 pos_hash 的种子/权重会自动变）")

    t0 = time.time()
    n_print = [0]

    def progress(e, hi):
        n_print[0] += 1
        if n_print[0] % 25 == 0:
            el = time.time() - t0
            print(f"  [scan] {e}/{hi}  {el:.0f}s", flush=True)

    diag = build_soft_index(
        args.data, args.labels, args.out,
        max_repeats=args.max_repeats, chunk=args.chunk,
        cache=not args.no_cache, materialized_dir=args.materialized_dir,
        progress=progress, log=lambda m: print(m, flush=True))

    el = time.time() - t0
    if not diag.get("out"):
        print(f"\n⚠ 未写出 {args.out}（命中 0 行）。用时 {el:.0f}s")
        raise SystemExit(1)

    print(f"\n写入 {diag['out']}")
    print(f"  标签 {diag['label_rows']}  命中局面 {diag.get('n_hit', -1)} "
          f"({100 * diag['hit_rate_vs_labels']:.2f}%)  覆盖数据行 {diag['matched_rows']}")
    print(f"  缓存命中={diag['cache_hit']}  用时 {el:.0f}s")
    z = np.load(diag["out"])
    if z["policy"].shape[0]:
        s = z["policy"].astype(np.float32).sum(1)
        print(f"  policy 每行和 {s.min():.4f}~{s.max():.4f}（应≈1）")


if __name__ == "__main__":
    main()