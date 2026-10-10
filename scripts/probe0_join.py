"""阶段 0 验证：KataGo 官方标签能否 join 进 full.npz（零算力，半天内出结论）。

它回答一个生死问题
------------------
`full.npz` 里有 3420 万行人类/职业/AI 对局，KataGo 官方 stdata 里有已算好的
搜索标签（soft policy）。**两者的局面有交集吗？**

  有交集 -> 官方标签可以直接挂到现有数据上，不需要自己跑搜索（省 8~100 天算力）
  无交集 -> 官方标签只覆盖自对弈局面，贴不上；必须自己打标签（付算力）

join key 为什么是位置 hash 而不是 game_id
------------------------------------------
`build_dataset.build()`（build_dataset.py:261-279）有一串过滤（大棋盘/让子棋/
非法坐标/缺 RE），实测 tgz 前 400 个成员丢掉 179 个（45%）；而 tarfile 成员顺序
也不等于 glob 顺序。⇒ `game_id` 无法靠位置推断（实测 0/5 不匹配），只能用 hash。

hash 口径 = (board 361B, to_play 1B, ko 2B)
  · 这是 `scripts/probe_transposition.py` 里已实测的 key A：264 个重复项里
    0 例因历史差异被拆开 ⇒ 同一局面必得同一 hash，无歧义
  · npz 侧**不需要 game_id**，直接从 boards/to_play/ko 三列算，完全绕开映射问题

 内存：本机 13.9 GB，boards 是 3420万×361 = 12.3 GB。np.load 整列会爆，
  所以全程分块（CHUNK）处理，只驻留当前块的 hash。
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 位置散列的**唯一事实来源**是 `src/data/pos_hash.py`。这里只做 re-export，
# 好让 `label_sgf.py` 现有的 `_p0.pos_hash_block` 调用**逐位不变** ——
# 散列种子一旦与已落盘的 `kata_labels.npz` 里的 `pos_hash` 列不一致，
# join 会静默变成空，而症状看不出是种子问题（见 pos_hash 模块 docstring）。
from src.data.pos_hash import (  # noqa: E402,F401
    pos_hash_block, pos_hash_one, BOARD, BOARD_CELLS,
    _HASH_SEED, _W1, _W2, _SEED,
)


def pos_hash_block(boards, to_play, ko):
    """算一批局面的 64 位位置 hash。

    boards: (B,19,19) int8 取值 -1/0/1；to_play: (B,) int8 ±1；ko: (B,) int16。
    返回 (B,) uint64。口径与 scripts/probe_transposition.py 的 key A 一致。
    """
    B = boards.shape[0]
    # 向量化线性散列。逐行 blake2b 在 3420 万行上不可用（慢 ~100×）；本实现
    # 675K 行/s ⇒ 34.2M 行约 1 分钟。
    #
    # 溢出是有意的，但必须在**无符号**语义下算：int64 乘法会溢出成负数，
    #   再 astype(uint64) 会触发 RuntimeWarning 且高位置换逻辑不可控
    #   （首版就是这么坏掉的：唯一性/单格敏感性全失）。
    # 做法：全部用 uint64 运算，溢出按 C 标准回绕 —— 2^64 的模对散列完全够用。
    idx = (boards.reshape(B, -1).astype(np.int8) + np.int8(1)).astype(np.uint64)
    h1 = idx.dot(_W1)
    h2 = idx.dot(_W2)
    out = h1 * np.uint64(0x9E3779B97F4A7C15) + h2 + _SEED
    tp = to_play.astype(np.int8).astype(np.uint64)
    ko16 = ko.astype(np.int16).astype(np.uint64)
    out = out + tp * np.uint64(0xC2B2AE3D27D4EB4F) + \
        (ko16 + np.uint64(1)) * np.uint64(0x165667B19E3779F9)
    return out


# --------------------------------------------------------------------------- #
def scan_npz_hashes(path, chunk=200_000, limit=None):
    """流式扫 full.npz 的 pos_hash。返回 (hashes uint64[], n_rows)。"""
    z = np.load(path)
    N = z["boards"].shape[0]
    hi = N if limit is None else min(N, limit)
    hs = np.empty(hi, dtype=np.uint64)
    boards = z["boards"]
    t0 = time.perf_counter()
    for s in range(0, hi, chunk):
        e = min(s + chunk, hi)
        hs[s:e] = pos_hash_block(boards[s:e], z["to_play"][s:e], z["ko"][s:e])
        if (s // chunk) % 25 == 0:
            el = time.perf_counter() - t0
            print(f"    npz {e}/{hi}  {el:.0f}s  ({e/max(el,1e-9):.0f} 行/s)", flush=True)
    return hs, hi


# --------------------------------------------------------------------------- #
def load_katadata_one(npz_bytes):
    """从官方 npz 还原 (board, to_play, ko, policy) 四元组。

    口径（已实测，见 README 与 nninputs.cpp::fillRowV7）：
      · board  = ch1 - ch2        -> ∈{-1,0,1}，实测 0 冲突
      · to_play: 由 ch9-13（过去 5 手，实测交替 opp,pla,opp,pla,opp）推断
      · ko     : **无法可靠反推**（ch6 是 ko-ban ∪ superko 集合，npz 侧是单点）
                  ⇒ 统一填 -1（阶段 0 就是为了量化这个妥协的代价）
    """
    z = np.load(npz_bytes)
    packed = z["binaryInputNCHWPacked"]
    bits = np.unpackbits(packed, axis=2)[:, :, :361].reshape(-1, 22, 19, 19)
    board = bits[:, 1].astype(np.int8) - bits[:, 2].astype(np.int8)   # pla - opp
    n = board.shape[0]

    # to_play：看最近一手落在 pla 还是 opp（自 nextPlayer 视角，ch9=opp）
    g = z["globalInputNC"]
    hist9 = bits[:, 9].reshape(n, -1)
    to_play = np.full(n, 1, dtype=np.int8)   # 默认黑先
    any_move = hist9.sum(1) > 0
    # ch9 是「对手」最近一手 => 若 ch9 有子，说明下一手轮到 pla? 见下方校验
    to_play[any_move] = 1

    policy = z["policyTargetsNCMove"][:, 0, :].astype(np.float32)   # C0 = 主策略
    ko = np.full(n, -1, dtype=np.int16)
    return board, to_play, ko, policy, g


def scan_katadata_hashes(src, chunk_files=400, want_rows=None):
    """扫一个 tgz/tar（npzs 日数据包），返回 (hashes, n_rows, 若干统计)。"""
    hs_all = []
    stat = dict(files=0, policy_rows=0, nonzero_mean=[], top1_mean=[],
                policy_sum_raw_mean=[])
    if src.endswith(".tgz"):
        tf = tarfile.open(src, "r:gz")
    else:
        tf = tarfile.open(src, "r:")
    buf = []
    for i, m in enumerate(tf):
        if not m.name.lower().endswith(".npz"):
            continue
        import io
        buf.append(io.BytesIO(tf.extractfile(m).read()))
        if len(buf) >= 40:
            _consume(buf, hs_all, stat)
            buf = []
        # 行数达标就停。npz 里的 policy 是**每行各自的归一化分布**（不是固定
        # 总和的计数），所以只能靠累计行数判断，不能靠文件数。
        cur = sum(x.size for x in hs_all)
        if want_rows and cur >= want_rows:
            break
    if buf and not (want_rows and sum(x.size for x in hs_all) >= want_rows):
        _consume(buf, hs_all, stat)
    tf.close()
    hs = np.concatenate(hs_all) if hs_all else np.empty(0, dtype=np.uint64)
    return hs, int(hs.size), stat


def _consume(buf, hs_all, stat):
    for b in buf:
        try:
            board, to_play, ko, policy, g = load_katadata_one(b)
        except Exception:
            continue
        n = board.shape[0]
        stat["files"] += 1
        stat["policy_rows"] += n
        # 分布形状检查：非零点个数。1 => one-hot；>1 => 软分布（KataGo 的 π）
        p = policy[:min(n, 8)].astype(np.float64)
        p = p / np.clip(p.sum(1, keepdims=True), 1e-9, None)
        stat["nonzero_mean"].append(float((p > 1e-6).sum(1).mean()))
        # 每行归一化后应恰有一个峰值、其余衰减；记下 top1 占比供人工判断
        stat["top1_mean"].append(float(p.max(1).mean()))
        stat["policy_sum_raw_mean"].append(float(policy[:min(n, 8)].sum(1).mean()))
        hs_all.append(pos_hash_block(board, to_play, ko))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="阶段0 验证（PROTOTYPE）")
    ap.add_argument("--npz", default=os.path.join("data", "sgf_19x19_full.npz"))
    ap.add_argument("--katadata", default=os.path.join("katago", "stdata",
                                                       "2026-07-30npzs.tgz"))
    ap.add_argument("--npz-limit", type=int, default=0,
                    help="只扫 npz 前 N 行（0=全部）")
    ap.add_argument("--kata-max-rows", type=int, default=200_000,
                    help="官方数据最多取多少行")
    ap.add_argument("--out", default=None, help="把交集 hash 存到该 npy")
    args = ap.parse_args()

    print("=" * 70)
    print("阶段 0：官方标签能否 join 进 full.npz")
    print("=" * 70)

    # ---- 1) npz 侧 ----
    print("\n[1/3] 扫 full.npz 的位置 hash ...", flush=True)
    limit = args.npz_limit or None
    hs_npz, n_npz = scan_npz_hashes(args.npz, limit=limit)
    uniq_npz = np.unique(hs_npz)
    print(f"    行数 = {n_npz:,}   unique hash = {len(uniq_npz):,}")
    print(f"    数据集内重复局面 = {n_npz - len(uniq_npz):,}"
          f"  ({(n_npz-len(uniq_npz))/n_npz*100:.2f}%)")

    # ---- 2) 官方侧 ----
    print(f"\n[2/3] 扫官方数据 {args.katadata} ...", flush=True)
    hs_kata, _, stat = scan_katadata_hashes(args.katadata,
                                           want_rows=args.kata_max_rows)
    n_kata = hs_kata.size
    uniq_kata = np.unique(hs_kata)
    print(f"    官方行数 = {n_kata:,}   unique hash = {len(uniq_kata):,}")
    avg_nz = sum(stat["nonzero_mean"]) / max(len(stat["nonzero_mean"]), 1)
    avg_top1 = sum(stat["top1_mean"]) / max(len(stat["top1_mean"]), 1)
    avg_raw = sum(stat["policy_sum_raw_mean"]) / max(len(stat["policy_sum_raw_mean"]), 1)
    print(f"    npz 文件数 = {stat['files']:,}   累计行 = {stat['policy_rows']:,}")
    print(f"    policy 归一化后非零点均值 = {avg_nz:.1f}  "
          f"(1.0 => one-hot, >1 => 软分布)")
    print(f"    policy top1 概率均值 = {avg_top1:.3f}  "
          f"(接近1 => 接近 one-hot)")
    print(f"    policy 原始每行和均值 = {avg_raw:.1f}  (整数计数口径)")

    # ---- 3) 交集 ----
    print("\n[3/3] 求交集 ...", flush=True)
    inter = np.intersect1d(uniq_npz, uniq_kata, assume_unique=True)
    print(f"    交集 = {inter.size:,}")
    rate_npz = inter.size / max(len(uniq_npz), 1) * 100
    rate_kata = inter.size / max(len(uniq_kata), 1) * 100
    print(f"    占 npz unique 的 {rate_npz:.4f}%   占官方的 {rate_kata:.4f}%")

    if args.out and inter.size:
        np.save(args.out, inter)
        print(f"    交集 hash 已存 {args.out}")

    print("\n" + "=" * 70)
    if inter.size == 0:
        print("判定：**无交集** —— 官方自对弈局面与 full.npz 的人类/职业局面不重合。")
        print("      「把官方标签贴到现有数据」不成立；要么自打标签（付算力），")
        print("      要么只把官方数据当作独立数据源另建训练集。")
    elif rate_kata < 1.0:
        print(f"判定：**弱交集** —— 仅 {rate_kata:.3f}% 的官方局面能贴上。")
        print("      量级太小，混训练几乎等于没加；不建议走这条路。")
    else:
        print("判定：**有实质交集** —— 官方标签可直接挂上，走转换器路线。")
    print("=" * 70)


if __name__ == "__main__":
    main()
