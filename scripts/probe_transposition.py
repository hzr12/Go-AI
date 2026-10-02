"""PROTOTYPE — 换位率测量（一次性，答完即弃）。

回答的问题
----------
**MCGS 值得做吗？**

MCTS 的搜索结构是树：同一局面若由不同落子顺序到达，会长出两个**各自独立评估**
的节点 —— 同一个盘面白白多付一次 `predict_batch`。MCGS（Monte Carlo Graph Search）
把这些节点在**图**上合并。但围棋里「同一局面」有两种口径，差别很大：

  key A = board + to_play + ko          → MCGS 真·合并口径（游戏论上的同一局面）
  key B = A + my_hist + op_hist         → 前向输入口径（`feature_planes` 通道
                                          1-3/5-7 编码最近 3 手，历史不同 ⇒
                                          平面不同 ⇒ 网络输入不同）

A−B 的差 = 「真 DAG」相对「只做前向缓存」的额外收益。

判定阈值（A 口径换位率）
------------------------
  < 5%    叫停，MCGS 不值得做
  5~15%   只做 B 口径的前向共享（低风险：树/统计/虚拟损失/线程全不动）
  > 15%   才考虑真 DAG（共享 visits，需重做虚拟损失与树复用）

为什么不改 mcts.py
------------------
`mcts.py:960` 在 `_expand` 返回后立刻 `leaf.board = None` 释放盘面，事后遍历树
拿不到局面。所以用子类在 `_expand` **入口**抓那一瞬 —— 此刻 board 还活着，
而 my_hist/op_hist/to_play 本来就不会被释放。`mcts.py` 一行不动。

已知偏差（方向：**低估**）
-------------------------
* `dynamic_topk`（`mcts.py:159-172`）早期把 topk 压到 8/16，采样浅 ⇒ 换位自然少
* `expand_chunk` / `solver_thresh` 早停会砍掉大量候选
* 所以低 sims 下测出的换位率偏低 ⇒ 报告 `mergeA` **随 sims 的增长趋势**，
  趋势比绝对值可靠

跑法
----
    python scripts/probe_transposition.py --device cpu
    python scripts/probe_transposition.py --device npu --sims 100,400
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from src.data.sgf_parser import SGFParser  # noqa: E402
from src.game.go_rules import GoBoard  # noqa: E402
from src.inference import GoAI  # noqa: E402
from src.search.mcts import MCTS  # noqa: E402

DEFAULT_MODEL = os.path.join("models", "sft_19x19_v12.pth")
DEFAULT_SGF_DIR = os.path.join("data", "games", "games")


# --------------------------------------------------------------------------- #
# 历史切分：与 build_dataset.py:71-93 逐字一致（否则测出来的局面分布不是训练分布）
# --------------------------------------------------------------------------- #
def pad3(seq):
    """补齐到 3 个元素，最近一手在索引 0，不足的尾部填 -1。（build_dataset.py:71）"""
    seq = list(seq[-3:])
    while len(seq) < 3:
        seq.append(-1)
    return seq


def split_hist(recent, to_play):
    """把 recent 拆成 my/op 各 3 手，索引 0 是最近一手。（build_dataset.py:78）"""
    my, op = [], []
    cur = -to_play                       # 最近一手由对手下的
    for mv in reversed(recent):
        if cur == to_play:
            my.append(mv)
        else:
            op.append(mv)
        cur = -cur
        if len(my) >= 3 and len(op) >= 3:
            break
    return my, op


# --------------------------------------------------------------------------- #
# 局面重放
# --------------------------------------------------------------------------- #
def replay_to_move(text, board_size, target_move):
    """把 SGF 重放到第 target_move 手，返回 (board, my_hist, op_hist, to_play)。

    约定与 `build_dataset.py:288-315` 一致：小棋盘坐标居中映射到大棋盘，
    pass 记 -1；非法走法直接放弃该局面。
    """
    parser = SGFParser()
    game = parser.parse_string(text)
    if game is None or game.board_size > board_size:
        return None
    if len(game.moves) < 2:
        return None
    # 让子棋：首两子同色（黑先单双）跳过，与 build_dataset 一致
    if game.moves[0].color == game.moves[1].color:
        return None

    off = (board_size - game.board_size) // 2 if game.board_size != board_size else 0
    board = GoBoard(board_size, komi=game.komi)
    history = []
    for i, mv in enumerate(game.moves):
        to_play = board.current_player
        if i == target_move:
            recent = history[-3:] if len(history) >= 3 else history
            my_h, op_h = split_hist(recent, to_play)
            return board, pad3(my_h), pad3(op_h), to_play
        r, c = mv.position
        target = -1 if (r, c) == (-1, -1) else (r + off) * board_size + (c + off)
        if not board.play(target):
            return None
        history.append(target)
    return None


# --------------------------------------------------------------------------- #
# 探针 MCTS：只在 _expand 入口记 key，不改 mcts.py
# --------------------------------------------------------------------------- #
class ProbeMCTS(MCTS):
    """记录每次**子节点前向**的两种 key。

    ⚠ 为什么钩 `_eval_children` 而不是 `_expand`：`_expand` 每次模拟只调用一次
      （mcts.py:956 在循环里逐 path 调），它记的是「被展开的叶子」——那些点几乎
      落在同一条搜索路径上、天然不重复，量出来永远是 0%。真正的量在**子节点**：
      每个叶子展开出 `expand_topk` 个孩子并一起前向（mcts.py:470），那是换位发生
      的地方。`_eval_children` 里 child_boards/to_plays/kos/my_hs/op_hs 全在手
      （mcts.py:442-470），正是我们要的 key 原料，且此刻盘面还没被释放。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.keys_A = []
        self.keys_B = []

    def _eval_children(self, board, to_play, leaf, moves, priors=None):
        """拦下这一批**子节点**的建局面数据，记下两种 key。

        为什么这里能拿到盘面：`_eval_children` 在 `mcts.py:465-470` 先
        `np.stack(child_boards)` 再调 `board.feature_planes_batched(...)`，
        那一批 `boards/to_plays/kos/my_hs/op_hs` 与**实际要前向的子节点一一对应**。
        `boards[i]` 本身就是子局面的裸盘面数组（`board.board.copy()`），
        此刻还活着 —— 所以实例属性遮蔽 `feature_planes_batched` 就能截住它，
        不必复制 `_eval_children` 的循环体（复制会和主线漂移）。

        `_expand_chunked`（`mcts.py:506`）也走 `_eval_children`，所以开了
        `expand_chunk` 同样抓得到，只是不含被 solver 早停砍掉的那部分。
        """
        orig = board.feature_planes_batched          # 绑定的是 staticmethod
        had_own = "feature_planes_batched" in getattr(board, "__dict__", {})

        def spy(boards, my_hs, op_hs, tps, kos=None, n_channels=17):
            for i in range(len(tps)):
                k = int(kos[i]) if kos is not None else -1
                ka = (np.asarray(boards[i]).tobytes(), int(tps[i]), k)
                self.keys_A.append(ka)
                self.keys_B.append(ka + (tuple(my_hs[i]), tuple(op_hs[i])))
            return orig(boards, my_hs, op_hs, tps, kos, n_channels=n_channels)

        board.feature_planes_batched = spy           # 实例属性遮蔽类 staticmethod
        try:
            return super()._eval_children(board, to_play, leaf, moves, priors)
        finally:
            if had_own:
                board.__dict__["feature_planes_batched"] = orig
            else:
                del board.feature_planes_batched


# --------------------------------------------------------------------------- #
def _sync(device):
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif d.startswith("npu") and hasattr(torch, "npu"):
        try:
            torch.npu.synchronize()
        except Exception:
            pass


def pick_sgf_paths(root, n_games, seed):
    """从语料里随机抽 n_games 个 19 路 SGF（尽量各源均衡）。"""
    if os.path.isfile(root):
        return [root]
    subs = [d for d in sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(d)]
    pools = []
    for d in subs:
        fs = sorted(glob.glob(os.path.join(d, "**", "*.sgf"), recursive=True))
        if fs:
            pools.append((os.path.basename(d), fs))
    if not pools:
        fs = sorted(glob.glob(os.path.join(root, "**", "*.sgf"), recursive=True))
        pools = [("all", fs)] if fs else []
    if not pools:
        return []

    rng = random.Random(seed)
    per = max(1, n_games // max(len(pools), 1))
    out = []
    for name, fs in pools:
        take = fs if len(fs) <= per else rng.sample(fs, per)
        for f in take:
            out.append((name, f))
    return out


def measure_one(ai, text, board_size, target_move, sims, num_threads,
                expand_topk):
    """跑一次搜索，返回 (nodes, uniqA, uniqB, elapsed_s) 或 None。"""
    pos = replay_to_move(text, board_size, target_move)
    if pos is None:
        return None
    board, my_h, op_h, to_play = pos
    m = ProbeMCTS(ai, board_size=board_size, num_threads=num_threads,
                  expand_topk=expand_topk, spec_prefetch=False,
                  temperature=1.0)
    t0 = time.perf_counter()
    m.search(board, my_h, op_h, to_play, simulations=sims)
    _sync(ai.device)
    el = time.perf_counter() - t0
    return len(m.keys_A), len(set(m.keys_A)), len(set(m.keys_B)), el


def main():
    ap = argparse.ArgumentParser(
        description="换位率测量（PROTOTYPE）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--attn-mode", default="window")
    ap.add_argument("--attn-window", type=int, default=7)
    ap.add_argument("--sgf-dir", default=DEFAULT_SGF_DIR)
    ap.add_argument("--n-games", type=int, default=4,
                    help="抽多少局（每局只取一个手数，故手数由 --moves 决定）")
    ap.add_argument("--moves", default="40,120,240")
    ap.add_argument("--sims", default="100,400")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--expand-topk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    moves_list = [int(x) for x in args.moves.split(",")]
    sims_list = [int(x) for x in args.sims.split(",")]

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch, "npu") and torch.npu.is_available():
            args.device = "npu"
        else:
            args.device = "cpu"

    model = None if str(args.model).lower() == "none" else args.model
    if model and not os.path.exists(model):
        print(f"[warn] 找不到 {model}，改用随机权重（换位率与权重无关，OK）")
        model = None

    print("=" * 78)
    print("换位率测量 PROTOTYPE —— 决定 MCGS 值不值得做")
    print(f"  device={args.device}  board={args.board_size}  "
          f"model={model or '(随机权重)'}")
    print(f"  语料={args.sgf_dir}  局数={args.n_games}  "
          f"手数={moves_list}  sims={sims_list}  threads={args.threads}")
    print("  ⚠ 换位率是**位置与深度相关**的估计，且 dynamic_topk / solver 早停会"
          "\n    低估它（见文件头）。看趋势，别只看单个数。")
    print("=" * 78)

    paths = pick_sgf_paths(args.sgf_dir, args.n_games, args.seed)
    if not paths:
        raise SystemExit(f"没在 {args.sgf_dir} 找到 SGF")
    print(f"抽到 {len(paths)} 局："
          f"{ {n: sum(1 for x, _ in paths if x == n) for n, _ in paths} }")

    ai = GoAI(model_path=model, board_size=args.board_size,
              device=args.device, use_amp=True,
              attn_mode=args.attn_mode, attn_window=args.attn_window)

    # 每手数收集：agg[(move, sims)] = (sum_nodes, sum_uniqA, sum_uniqB, n_ok, sum_ms)
    agg = {}
    skipped = 0
    t_start = time.perf_counter()
    for move_i, move_no in enumerate(moves_list):
        src, path = paths[move_i % len(paths)]
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            skipped += 1
            continue
        for sims in sims_list:
            r = measure_one(ai, text, args.board_size, move_no, sims,
                            args.threads, args.expand_topk)
            if r is None:
                skipped += 1
                continue
            nodes, uniqA, uniqB, el = r
            k = (move_no, sims)
            n, ua, ub, ok, sec = agg.get(k, (0, 0, 0, 0, 0.0))
            agg[k] = (n + nodes, ua + uniqA, ub + uniqB, ok + 1, sec + el)
            print(f"  [{src}] move={move_no:<3} sims={sims:<4} "
                  f"nodes={nodes:<5} uniqA={uniqA:<5} uniqB={uniqB:<5} "
                  f"mergeA={1 - uniqA / max(nodes, 1):.1%} "
                  f"mergeB={1 - uniqB / max(nodes, 1):.1%}  {el:.2f}s")

    # ---- 汇总 ----
    print("\n=== 汇总（跨局取和后算率，等价于按节点数加权）===")
    hdr = (f"{'move':>6} {'sims':>6} {'nodes':>8} {'uniqA':>8} {'uniqB':>8} "
           f"{'mergeA':>8} {'mergeB':>8} {'ms/move':>9} {'可省前向':>9}")
    print("  " + hdr)
    print("  " + "-" * (len(hdr) + 2))

    rows = []
    for (move_no, sims) in sorted(agg):
        n, ua, ub, ok, sec = agg[(move_no, sims)]
        if n == 0:
            continue
        mA = 1 - ua / n
        mB = 1 - ub / n
        ms = sec / ok * 1e3
        # 可省前向：若按 B 口径共享，最多省掉 mB 的前向；A 口径则是 mA
        print(f"  {move_no:>6} {sims:>6} {n:>8} {ua:>8} {ub:>8} "
              f"{mA:>7.1%} {mB:>7.1%} {ms:>9.0f} {mB:>8.1%}")
        rows.append(dict(move=move_no, sims=sims, nodes=n, uniqA=ua,
                         uniqB=ub, mergeA=round(mA, 4), mergeB=round(mB, 4),
                         ms_per_move=round(ms)))

    # ---- 趋势：mergeA 是否随深度/手数增长 ----
    if rows:
        deep = [r for r in rows if r["sims"] == max(sims_list)]
        shallow = [r for r in rows if r["sims"] == min(sims_list)]
        if deep and shallow:
            g_deep = statistics.mean(r["mergeA"] for r in deep)
            g_shal = statistics.mean(r["mergeA"] for r in shallow)
            print(f"\n  mergeA 平均：sims={min(sims_list)} → {g_shal:.1%}，"
                  f"sims={max(sims_list)} → {g_deep:.1%}"
                  f"（{'随深度上升' if g_deep > g_shal else '未随深度上升 ⚠'}）")
        meanA = statistics.mean(r["mergeA"] for r in rows)
        meanB = statistics.mean(r["mergeB"] for r in rows)
        print(f"  全部：mergeA 均值 {meanA:.1%}，mergeB 均值 {meanB:.1%}"
              f"，A−B = {meanA - meanB:.1%}")
        verdict = ("叫停", "MCGS 不值得（<5%）") if meanA < 0.05 else (
            "只做 B", "前向共享即可（5~15%）") if meanA < 0.15 else (
            "考虑真 DAG", "可考虑共享 visits（>15%）")
        print(f"  ⇒ 判定：**{verdict[0]}** —— {verdict[1]}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(rows=rows, skipped=skipped,
                           device=args.device,
                           elapsed_s=round(time.perf_counter() - t_start, 1)),
                      f, ensure_ascii=False, indent=2)
        print(f"\n结果写入 {args.json}")
    if skipped:
        print(f"\n注：跳过 {skipped} 个（该局在该手数无合法局面，如让子棋/太短）")


if __name__ == "__main__":
    main()