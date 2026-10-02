"""PROTOTYPE — 搜索速度基准（一次性，答完即弃）。

它回答一个问题
--------------
「MCTS 太慢」到底贵在**前向调用次数**，还是贵在**样本数**？

- `MCTS.search(sims=S)` 的批处理把叶子/子节点跨模拟合并，所以调用次数 ≈ 2，
  但样本数 ≈ S × (expand_topk+1)。
- `lookahead(depth=D)` 的调用次数 = depth+1（线性），但样本数随 depth **超线性**。

只有把「墙钟 / 调用数 / 样本数」三个数并排看，才知道该走哪条路：
  走 MCTS 回采集路径？ 还是加深 lookahead？ 还是两者都不划算、去改 shaping？

跑法
----
    python scripts/bench_search.py                  # 全部四段
    python scripts/bench_search.py --only fwd       # 只跑前向吞吐曲线（决定性数字）
    python scripts/bench_search.py --device npu     # 4×910A 上跑（决策看这个）
    python scripts/bench_search.py --model none     # 随机权重（只验脚本，不看棋力）

⚠ 本机是 torch+CPU：跑出来的曲线**只能验证脚本本身**。决策用的数字必须在
  4×910A 上取，因为 NPU 还有 `_NPU_BATCH_BUCKETS` 归桶与算子编译缓存的影响。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402  (须先补 sys.path)

from src.game.go_rules import GoBoard  # noqa: E402
from src.inference import GoAI  # noqa: E402
from src.search.lookahead import lookahead as lookahead_fn  # noqa: E402
from src.search.mcts import MCTS  # noqa: E402
from src.search.policy_sampler import sample_move  # noqa: E402

DEFAULT_MODEL = os.path.join("models", "sft_19x19_v12.pth")


# --------------------------------------------------------------------------- #
# 统计前向：调用次数 / 累计样本 / batch 尺寸分布
# --------------------------------------------------------------------------- #
class ForwardProbe:
    """包住 `GoAI.predict_batch`，量出**调用次数与样本数**。

    为什么要它：「MCTS sims=48 每手约 48 次前向」这类结论，靠读代码是推不出来的
    （批处理把调用合并了）。这个探针把它变成实测数字。

    ⚠ 通过实例属性遮蔽类方法，故必须在**构造 MCTS / 调 lookahead 之前**安装；
      装晚了 MCTS 已经抓走了原方法，探针就接不到。
    """

    def __init__(self, ai):
        self._orig = ai.predict_batch
        self.reset()
        ai.predict_batch = self._wrapped

    def _wrapped(self, states):
        self.calls += 1
        self.samples += len(states)
        self.batches.append(len(states))
        return self._orig(states)

    def reset(self):
        self.calls = 0
        self.samples = 0
        self.batches = []

    def stats(self, n_iter):
        """返回**每次落子**的均值。`batch_hist` 是**原始累计**（未除 n_iter）——
        要看每次落子的分布请自己除以 `n_iter`；两者对不上不是 bug。
        刻意保留原始值：取整后的分桶再相除会把 0.5 这类计数塌掉。
        """
        if n_iter <= 0 or self.calls == 0:
            return {"calls": 0.0, "samples": 0.0, "max_batch": 0,
                    "batch_hist": {}, "n_iter": n_iter}
        return {
            "calls": round(self.calls / n_iter, 2),
            "samples": round(self.samples / n_iter, 1),
            "max_batch": max(self.batches),
            "batch_hist": dict(sorted(Counter(self.batches).items())),
            "n_iter": n_iter,
        }


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
def sync(device):
    """NPU/CUDA 上前向是异步的，不同步会把时间记到下一次调用头上。"""
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif d.startswith("npu") and hasattr(torch, "npu"):
        try:
            torch.npu.synchronize()
        except Exception:
            pass


def timeit(fn, device, *, warmup=3, reps=6, budget=8.0, probe=None):
    """计时 fn 若干次（受 budget 截断），返回 (median_s, min_s, n_iter, stable)。

    probe 在 warmup **之后**才 reset —— 否则探针统计里混进 warmup 的调用。

    `stable` 是自检结果：min/median < 0.7 说明这一轮**还没进稳态**（oneDNN 图调优、
    算子编译、CPU 降频都在这时污染计时），于是补加热再量一次。本机实测同一配置
    两次跑出 32 ms vs 8.5 ms 的 3.8 倍漂移，就是因为没进稳态 —— 比率结论会因此翻面。
    """

    def _run(n):
        ts, t0 = [], time.perf_counter()
        for _ in range(n):
            a = time.perf_counter()
            fn()
            sync(device)
            ts.append(time.perf_counter() - a)
            if time.perf_counter() - t0 > budget:
                break
        return ts

    for _ in range(warmup):
        fn()
    sync(device)
    if probe is not None:
        probe.reset()
    ts = _run(reps)

    stable = True
    if len(ts) >= 3:
        med = statistics.median(ts)
        if med > 0 and min(ts) / med < 0.7:
            stable = False
            for _ in range(max(3, warmup)):   # 补加热
                fn()
            sync(device)
            if probe is not None:
                probe.reset()
            ts = _run(reps)
            if len(ts) >= 3:
                med2 = statistics.median(ts)
                stable = med2 <= 0 or min(ts) / med2 >= 0.7

    return statistics.median(ts), min(ts), len(ts), stable


# --------------------------------------------------------------------------- #
# 确定性局面
# --------------------------------------------------------------------------- #
def make_positions(board_size, move_counts, seed):
    """造出给定手数的确定性局面（开局 / 中盘 / 收官）。

    历史约定与 `scripts/eval_elo.py:184-214` 逐字一致：`hists = [黑最近3手, 白最近3手]`，
    初值 `[-1,-1,-3]`，轮到谁就 pop/append 谁那一套。feature 平面只认 `>= 0`，
    所以初值里的 -1/-3 都是「无此手」。
    """
    rng = np.random.default_rng(seed)
    board = GoBoard(board_size)
    hists = [[-1, -1, -3], [-1, -1, -3]]
    pass_m = board_size * board_size
    want = sorted(set(int(x) for x in move_counts))
    out = {}
    i = 0
    while i < len(want):
        to_play = board.current_player
        legal = board.get_legal_moves()
        if not legal.any():
            mv = pass_m
        else:
            mv = int(rng.choice(np.flatnonzero(legal)))
        pmv = -1 if mv == pass_m else mv
        if not board.play(pmv):
            pmv = -1
            board.play(-1)
        if pmv >= 0:
            h = hists[0] if to_play == 1 else hists[1]
            h.pop(0)
            h.append(pmv)
        # play 之后 move_number 才推进，所以在推进后对齐目标手数
        if i < len(want) and board.move_number >= want[i]:
            out[want[i]] = (board.clone(),
                            list(hists[0]), list(hists[1]), to_play)
            i += 1
        if board.passes >= 2:
            break
    return [out[k] for k in sorted(out)]


# --------------------------------------------------------------------------- #
# 第 1 段：前向吞吐曲线（M1）
# --------------------------------------------------------------------------- #
def run_fwd(ai, positions, args, results):
    board, mh, oh, tp = positions[0]
    n = ai.board_size * ai.board_size + 1
    rows = []
    print("\n=== 1. 前向吞吐曲线（predict_batch，给定 batch 尺寸 B）===")
    print("    nn  = 只算网络（平面预计算，5 元组）——MCTS 与 lookahead 都走这条")
    print("    tot = 网络 + 现算特征（4 元组）—— 未预取时的真实代价")
    print("    feat= 只算 feature_planes_batched，不含网络")
    print("    eff = 吞吐相对 B=最小 的倍数（1.00 = 批量毫无收益）")
    print("    ms/samp = ms/B —— **决策看这一列**：它随 B 下降才说明大批量摊薄了")
    print("             开销；持平就是算力已饱和，墙钟 ≈ 样本数，批处理优化无效")
    hdr = (f"{'B':>6} {'nn ms':>9} {'nn samples/s':>13} {'ms/samp':>9} "
           f"{'tot ms':>9} {'feat ms':>9} {'feat%':>7} {'eff':>7}")
    print("    " + hdr)
    print("    " + "-" * len(hdr))

    ref_nn = None
    unstable = []
    for B in args.batch_sizes:
        base_planes = board.feature_planes_batched(
            np.repeat(board.board[None], B, axis=0),
            [list(mh)] * B, [list(oh)] * B, [tp] * B,
            [board.ko_point] * B, n_channels=ai.in_channels)
        pre_states = [(None, list(mh), list(oh), tp, base_planes[i])
                      for i in range(B)]
        raw_states = [(board, list(mh), list(oh), tp) for _ in range(B)]

        def call_nn(states=pre_states):
            return ai.predict_batch(states)

        def call_tot(states=raw_states):
            return ai.predict_batch(states)

        def call_feat(mh=mh, oh=oh, tp=tp, b=board, B=B):
            b.feature_planes_batched(
                np.repeat(b.board[None], B, axis=0),
                [list(mh)] * B, [list(oh)] * B, [tp] * B,
                [b.ko_point] * B, n_channels=ai.in_channels)

        # 大 batch 单次就慢，reps 自适应往下砍，避免 B=768 跑满预算
        reps = max(1, min(args.reps, 32))
        m_nn, _, _, ok_nn = timeit(call_nn, args.device, warmup=args.warmup,
                                   reps=reps, budget=args.budget)
        m_tot, _, _, _ = timeit(call_tot, args.device, warmup=args.warmup,
                                reps=max(1, reps // 2), budget=args.budget)
        m_ft, _, _, _ = timeit(call_feat, args.device, warmup=args.warmup,
                               reps=max(1, reps // 2), budget=args.budget)
        if not ok_nn:
            unstable.append(B)
        # 单位是 samples/秒（m_nn 本身已是秒），**不要再除 1000**
        nn_per_s = B / m_nn if m_nn > 0 else 0.0
        if ref_nn is None:
            ref_nn = nn_per_s
        rel = (nn_per_s / ref_nn) if ref_nn else 1.0
        feat_pct = (m_ft / m_nn * 100.0) if m_nn > 0 else 0.0
        ms_samp = (m_nn * 1e3 / B) if B else 0.0
        row = dict(B=B, nn_ms=m_nn * 1e3, nn_per_s=nn_per_s, ms_samp=ms_samp,
                   tot_ms=m_tot * 1e3, tot_per_s=(B / m_tot if m_tot else 0.0),
                   feat_ms=m_ft * 1e3, feat_pct=feat_pct, eff=rel)
        rows.append(row)
        print(f"    {B:>6} {row['nn_ms']:>9.2f} {nn_per_s:>13.1f} "
              f"{ms_samp:>9.3f} {row['tot_ms']:>9.2f} {row['feat_ms']:>9.2f} "
              f"{feat_pct:>6.1f}% {rel:>7.2f}")

    # ---- 判定：批量到底能不能把开销摊薄 ----
    # 基线取**最小 B**（单条前向）。用 B≈48 当基线没有意义：那既不回答「批处理
    # 有没有用」，也依赖扫描点里恰好有 48。
    # ⚠ 相除必须用**未取整**的值：0.04/0.11 这种小数被四舍五入到 1 位后会塌成
    #   0.36 之类毫无意义的数（本脚本第一版的真 bug）。
    xs = sorted((r["B"], r["nn_per_s"], r["ms_samp"]) for r in rows)
    b_lo, t_lo, ms_lo = xs[0]
    b_hi, t_hi, ms_hi = xs[-1]
    verdict = None
    if t_lo > 0:
        ratio = t_hi / t_lo
        if ratio >= 1.5:
            verdict = ("A", f"吞吐 B={b_lo}→{b_hi}: {t_lo:.1f}→{t_hi:.1f} samples/s "
                            f"（{ratio:.2f}×）⇒ **批量摊薄开销**：大 batch 单样本更便宜，"
                            "谁凑得出大 batch 谁赢 —— MCTS 的子节点批、lookahead 的深层"
                            "节点批都会变便宜")
        elif ratio <= 1.25:
            verdict = ("B", f"吞吐 B={b_lo}→{b_hi} 仅 {ratio:.2f}× ⇒ **算力已饱和、"
                            f"批量无收益**：单样本恒为 {ms_hi:.3f} ms，墙钟 ≈ 样本数 ⇒ "
                            "样本少的一方直接赢；优化方向是「少算样本」，把调用合并、"
                            "提高 batch 都不会更快")
        else:
            verdict = ("C", f"吞吐 B={b_lo}→{b_hi} = {ratio:.2f}× ⇒ 批量收益中等，"
                            "别按单一结论下判断，逐档看 ms/samp 再选")
        print(f"\n    判定 {verdict[0]}：{verdict[1]}")

    if unstable:
        print(f"\n    ⚠ 未进稳态：B={unstable} 补加热后 min/median 仍 < 0.7，"
              f"\n      这几档的数值不可信，**不要据此下比率结论**（重跑 / 加大 "
              f"--reps / 确认机器空载）")

    results["fwd"] = {"rows": rows,
                      "verdict": verdict[0] if verdict else None,
                      "unstable": unstable}
    return results


# --------------------------------------------------------------------------- #
# 第 2 段：MCTS 每手成本（M2）
# --------------------------------------------------------------------------- #
def run_mcts(ai, positions, args, results, probe):
    print("\n=== 2. MCTS 每手成本（MCTS.search）===")
    print("    calls/samples = 每次落子的 predict_batch 调用数与累计样本数（探针实测）")
    print("    avgB = samples/calls，平均 batch 尺寸 —— 批处理到底凑没凑起来看这一列")
    print("    B=1% = **batch=1 的调用占比**：这部分完全没吃到批量，是结构性浪费")
    hdr = (f"{'sims':>6} {'thr':>4} {'ms/move':>9} {'calls':>7} {'samples':>9} "
           f"{'avgB':>7} {'maxB':>6} {'B=1%':>7}")
    print("    " + hdr)
    print("    " + "-" * len(hdr))

    rows = []
    unstable = []
    for sims in args.sims:
        for thr in args.threads:
            mcts = MCTS(ai, board_size=ai.board_size, num_threads=thr,
                        expand_topk=args.expand_topk,
                        expand_chunk=args.expand_chunk,
                        leaf_ab_depth=args.leaf_ab_depth,
                        spec_prefetch=thr >= 4)

            def one(mcts=mcts, sims=sims):
                for board, mh, oh, tp in positions:
                    mcts.search(board, mh, oh, tp, simulations=sims)

            med, mn, nit, ok = timeit(one, args.device, warmup=args.warmup,
                                      reps=args.reps, budget=args.budget,
                                      probe=probe)
            if not ok:
                unstable.append(f"sims={sims}/thr={thr}")
            st = probe.stats(nit * len(positions))
            per = med * 1e3
            sm = st["samples"]
            avg_b = (sm / st["calls"]) if st["calls"] else 0.0
            hist = st["batch_hist"]
            tot_c = sum(hist.values())
            pct_b1 = (hist.get(1, 0) / tot_c * 100.0) if tot_c else 0.0
            row = dict(sims=sims, threads=thr, ms=round(per, 1),
                       calls=st["calls"], samples=sm,
                       max_batch=st["max_batch"], avg_batch=round(avg_b, 2),
                       pct_b1=round(pct_b1, 1),
                       samp_per_ms=round(sm / per, 2) if per else 0.0,
                       batch_hist=hist, n_iter=st.get("n_iter"))
            rows.append(row)
            print(f"    {sims:>6} {thr:>4} {per:>9.1f} {row['calls']:>7.2f} "
                  f"{sm:>9.0f} {avg_b:>7.2f} {row['max_batch']:>6} "
                  f"{row['pct_b1']:>6.1f}%")

    if unstable:
        print(f"\n    ⚠ 未进稳态（min/median < 0.7，补加热后仍如此）：{unstable}"
              f"\n      这几档数值不可信")

    results["mcts"] = {"rows": rows, "unstable": unstable}
    return results


# --------------------------------------------------------------------------- #
# 第 3 段：lookahead / sample_move 每手成本（M3）
# --------------------------------------------------------------------------- #
def run_la(ai, positions, args, results, probe):
    print("\n=== 3. lookahead 每手成本（当前 RL 采集走的是它）===")
    print("    预期调用数 = depth+1（线性），样本数超线性；探针用来验这两句")
    hdr = (f"{'depth':>6} {'width':>6} {'kind':>7} {'ms/move':>9} {'calls':>7} "
           f"{'samples':>9} {'avgB':>7} {'maxB':>6} {'samples/ms':>11}")
    print("    " + hdr)
    print("    " + "-" * len(hdr))

    n = ai.board_size * ai.board_size + 1
    rows = []
    unstable = []
    for depth in args.depths:
        for width in args.widths:
            def one(depth=depth, width=width):
                for board, mh, oh, tp in positions:
                    lookahead_fn(ai, board, mh, oh, tp, n, ai.in_channels,
                                 topk=args.topk, width=width, depth=depth)

            med, mn, nit, ok = timeit(one, args.device, warmup=args.warmup,
                                      reps=args.reps, budget=args.budget,
                                      probe=probe)
            if not ok:
                unstable.append(f"depth={depth}/w={width}")
            st = probe.stats(nit * len(positions))
            per = med * 1e3
            avg_b = (st["samples"] / st["calls"]) if st["calls"] else 0.0
            row = dict(depth=depth, width=width, kind="lookahead", ms=round(per, 1),
                       calls=st["calls"], samples=st["samples"],
                       max_batch=st["max_batch"], avg_batch=round(avg_b, 2),
                       samp_per_ms=round(st["samples"] / per, 2) if per else 0.0,
                       batch_hist=st["batch_hist"])
            rows.append(row)
            print(f"    {depth:>6} {width:>6} {'LA':>7} {per:>9.1f} "
                  f"{row['calls']:>7.2f} {row['samples']:>9.0f} "
                  f"{avg_b:>7.2f} {row['max_batch']:>6} "
                  f"{row['samp_per_ms']:>11.2f}")

    # 真实 RL 采集路径：sample_move = 平面 + lookahead + 温度采样 + 混合
    def one_sm():
        for board, mh, oh, tp in positions:
            sample_move(ai, board, mh, oh, tp,
                        topk=args.topk, width=args.widths[0],
                        depth=args.depths[0])

    med, _, nit, ok = timeit(one_sm, args.device, warmup=args.warmup,
                             reps=args.reps, budget=args.budget, probe=probe)
    if not ok:
        unstable.append("sample_move")
    st = probe.stats(nit * len(positions))
    per = med * 1e3
    avg_b = (st["samples"] / st["calls"]) if st["calls"] else 0.0
    sm = dict(depth=args.depths[0], width=args.widths[0], kind="sample_move",
              ms=round(per, 1), calls=st["calls"], samples=st["samples"],
              max_batch=st["max_batch"], avg_batch=round(avg_b, 2),
              samp_per_ms=round(st["samples"] / per, 2) if per else 0.0,
              batch_hist=st["batch_hist"])
    rows.append(sm)
    print(f"    {sm['depth']:>6} {sm['width']:>6} {'RL实际':>7} {per:>9.1f} "
          f"{sm['calls']:>7.2f} {sm['samples']:>9.0f} {avg_b:>7.2f} "
          f"{sm['max_batch']:>6} {sm['samp_per_ms']:>11.2f}")

    if unstable:
        print(f"\n    ⚠ 未进稳态（min/median < 0.7，补加热后仍如此）：{unstable}"
              f"\n      这几档数值不可信")

    results["la"] = {"rows": rows, "unstable": unstable}
    return results


# --------------------------------------------------------------------------- #
# 第 4 段：等墙钟对比（M4）
# --------------------------------------------------------------------------- #
def run_cmp(results, args):
    print("\n=== 4. 等墙钟对比 ===")
    if "mcts" not in results or "la" not in results:
        print("    （跳过：需要先跑 --only mcts 与 --only la）")
        return results

    all_rows = []
    for r in results["mcts"]["rows"]:
        all_rows.append(("MCTS", f"sims={r['sims']} thr={r['threads']}", r, True))
    for r in results["la"]["rows"]:
        tag = "RL" if r["kind"] == "sample_move" else "LA"
        all_rows.append((tag, f"depth={r['depth']} w={r['width']}", r, False))

    all_rows.sort(key=lambda x: x[2]["ms"])
    hdr = (f"{'kind':>4} {'config':>22} {'ms/move':>9} {'calls':>7} "
           f"{'samples':>9} {'ms/samp':>9} {'avgB':>7} {'访问分布':>9}")
    print("    " + hdr)
    print("    " + "-" * len(hdr))
    for kind, cfg, r, has_visits in all_rows:
        mark = "有 (N)" if has_visits else "无"
        psp = (r["ms"] / r["samples"]) if r.get("samples") else 0.0
        print(f"    {kind:>4} {cfg:>22} {r['ms']:>9.1f} {r['calls']:>7.2f} "
              f"{r['samples']:>9.0f} {psp:>9.2f} {r.get('avg_batch', 0.0):>7.2f} "
              f"{mark:>9}")

    # ---- 关键判读：每样本成本是否一致 ----
    psp = [r["ms"] / r["samples"] for _, _, r, _ in all_rows if r.get("samples")]
    spread = (max(psp) / min(psp)) if psp and min(psp) > 0 else None
    print()
    if spread is not None and len(psp) >= 2:
        lo, hi = min(psp), max(psp)
        if spread <= 1.25:
            print(f"    ⇒ 每样本成本一致（{lo:.1f}~{hi:.1f} ms/样本，离散 {spread:.2f}×）：")
            print("      **等墙钟 ⇔ 等样本数** ⇒ 选配置 = 选「单位时间买到多少搜索节点」。")
            print("      而且**调用次数不影响成本**：MCTS 的调用数是 lookahead 的数倍，")
            print("      在这台机器上毫无区别 —— 批量结构的差异已被算力饱和完全吃掉。")
            print("      ⇒ 该比的是 **samples**，不是 calls。")
        else:
            print(f"    ⇒ 每样本成本不一致（{lo:.1f}~{hi:.1f}，离散 {spread:.2f}×）：")
            print("      大 batch 确实更便宜 ⇒ MCTS 的大 batch 结构 / lookahead 的深层")
            print("      节点批有真实优势，**调用结构本身值得优化**（见 §2 avgB）。")

    # ---- 同成本换算：直接在**实测 ms** 上插值，不要用 samples 折算 ----
    # （samples 折算只在「成本 ∝ 样本」成立时才对；上面刚判过离散度，可能 >1.25×）
    mcts_rows = [x for x in all_rows if x[3]]
    la_rows = [x for x in all_rows if not x[3] and x[0] == "LA"]
    if mcts_rows and la_rows:
        pts = sorted(((r["sims"], r["ms"]) for _, _, r, _ in mcts_rows),
                     key=lambda kv: kv[0])

        def sims_at(target):
            """按实测 (sims, ms) 插值出等成本的 sims；越界则按过原点比例外推。"""
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if target <= ys[0]:
                return (xs[0] * target / ys[0]) if ys[0] else None, True
            if target >= ys[-1]:
                return (xs[-1] * target / ys[-1]) if ys[-1] else None, True
            for i in range(len(xs) - 1):
                if ys[i] <= target <= ys[i + 1]:
                    span = ys[i + 1] - ys[i]
                    f = ((target - ys[i]) / span) if span else 0.0
                    return xs[i] + f * (xs[i + 1] - xs[i]), False
            return None, True

        print(f"\n    同成本换算（在实测 (sims, ms) 上插值，* = 越界外推）：")
        print(f"    {'LA 配置':>18} {'ms/move':>9} {'samples':>9} "
              f"{'= 等成本 MCTS sims':>20} {'π':>4}")
        print("    " + "-" * 66)
        for _, cfg_l, rl, _ in la_rows:
            eq, ext = sims_at(rl["ms"])
            if eq is None:
                continue
            star = "*" if ext else " "
            print(f"    {cfg_l:>18} {rl['ms']:>9.1f} {rl['samples']:>9.0f} "
                  f"{eq:>19.1f}{star} {'有':>4}")
        print("    ⇒ 读法：同样一笔墙钟，可以买「更浅的 minimax（无 π）」")
        print("      或「等量的 MCTS 节点（有 π）」。**π 是副产品，不额外计费。**")
        print("      ⚠ 这张表只回答成本；**棋力谁强得靠 eval_elo 实测**，"
              "本脚本给不出。")

    results["cmp"] = {"n_rows": len(all_rows),
                      "ms_per_sample_spread": round(spread, 3) if spread else None}
    return results


# --------------------------------------------------------------------------- #
def parse_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="搜索速度基准（PROTOTYPE，答完即弃）")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="checkpoint 路径；传 none 用随机权重（只验脚本）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--use-amp", action="store_true", default=True)
    ap.add_argument("--attn-mode", default="window")
    ap.add_argument("--attn-window", type=int, default=7)
    ap.add_argument("--only", default="all",
                    choices=["all", "fwd", "mcts", "la", "cmp"])
    ap.add_argument("--batch-sizes", default="1,8,16,32,48,64,96,128,192,256,384,768")
    ap.add_argument("--sims", default="16,32,48,64")
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--depths", default="2,3,4")
    ap.add_argument("--widths", default="4,8")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--expand-topk", type=int, default=32)
    ap.add_argument("--expand-chunk", type=int, default=0)
    ap.add_argument("--leaf-ab-depth", type=int, default=0)
    ap.add_argument("--positions", default="10,120,300",
                    help="用来取局面的手数（开局/中盘/收官）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3,
                    help="预热轮数；不足稳态时 timeit 会自动补加热")
    ap.add_argument("--budget", type=float, default=8.0,
                    help="单条测量最多跑多少秒（自动截断）")
    ap.add_argument("--json", default=None, help="把结果写进这个 JSON 文件")
    args = ap.parse_args()

    args.batch_sizes = parse_list(args.batch_sizes)
    args.sims = parse_list(args.sims)
    args.threads = parse_list(args.threads)
    args.depths = parse_list(args.depths)
    args.widths = parse_list(args.widths)
    args.positions = parse_list(args.positions)

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch, "npu") and torch.npu.is_available():
            args.device = "npu"
        else:
            args.device = "cpu"

    model = None if str(args.model).lower() == "none" else args.model
    if model and not os.path.exists(model):
        print(f"[warn] checkpoint 不存在: {model} —— 改用随机权重（只验脚本）")
        model = None

    print("=" * 78)
    print("搜索速度基准 PROTOTYPE")
    print(f"  device={args.device}  torch={torch.__version__}  "
          f"board={args.board_size}")
    print(f"  model={model or '(随机权重，仅验脚本)'}")
    print(f"  attn_mode={args.attn_mode} window={args.attn_window} "
          f"amp={args.use_amp}")
    print(f"  positions={args.positions}  reps={args.reps}  "
          f"warmup={args.warmup}  budget={args.budget}s")
    print("  ⚠ 本脚本报的是**单次运行**的中位数。同配置跨运行实测波动可达 1.7×"
          "\n    （本机 MCTS sims=8 出过 1201 ms 与 2029 ms），所以："
          "\n    · 下比率结论前**至少跑 2 次**；"
          "\n    · 绝对值以 min 更稳，别拿单次中位数当真值；"
          "\n    · 前向吞吐曲线若报「未进稳态」，那一档直接作废。")
    print("=" * 78)

    ai = GoAI(model_path=model, board_size=args.board_size,
              device=args.device, use_amp=args.use_amp,
              attn_mode=args.attn_mode, attn_window=args.attn_window)
    args.board_size = ai.board_size

    positions = make_positions(args.board_size, args.positions, args.seed)
    if not positions:
        raise SystemExit("没能构造出局面（对局在目标手数前就结束了）")
    print(f"局面：{args.board_size} 路，{len(positions)} 个，手数="
          f"{[b.move_number for b, _, _, _ in positions]}")

    probe = ForwardProbe(ai)
    results = {}

    if args.only in ("all", "fwd"):
        results = run_fwd(ai, positions, args, results)
    if args.only in ("all", "mcts"):
        results = run_mcts(ai, positions, args, results, probe)
    if args.only in ("all", "la"):
        results = run_la(ai, positions, args, results, probe)
    if args.only in ("all", "cmp"):
        results = run_cmp(results, args)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json}")


if __name__ == "__main__":
    main()
