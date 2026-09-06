#!/usr/bin/env python3
"""ELO 评测：同一模型的不同对弈配置（策略 / 混合 / MCTS）round-robin 对弈评分。

每个"棋手"是同一份权重 + 不同选点策略，两两配对打循环赛（执黑执白各半），
用逻辑回归 MLE（梯度下降）拟合 Elo 分差。平局记 0.5 分。

用法:
    python scripts/eval_elo.py --model models/sft_19x19_v3.pth --board-size 9 \
        --games 10 \
        --player "random=random" \
        --player "policy=policy" \
        --player "hyb32=hybrid,sims=32,blend=0.5,leaf=1" \
        --player "mcts100=mcts,sims=100"

棋手规格: label=type[,key=val]*
    type:  policy | hybrid | mcts | random
    key:   sims=N（模拟数） blend=F（策略权重，hybrid） temp=F（采样温度，policy）
           leaf=0/1（priors_leaf：叶子自身 policy 作先验，少模拟推荐 1）
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.inference import GoAI
from src.game.go_rules import GoBoard
from src.search.mcts import MCTS


# --------------------------------------------------------------------------- #
# 棋手
# --------------------------------------------------------------------------- #
class PolicyPlayer:
    """纯策略：单次前向 argmax（temp>0 时按温度采样）。"""

    def __init__(self, ai, board_size, temp=0.0, topk=0):
        self.ai = ai
        self.n_actions = board_size * board_size + 1
        self.temp = temp
        self.topk = topk

    def new_game(self):
        pass

    def select(self, board, h_black, h_white, to_play, legal, path_moves):
        policy, _ = self.ai.predict(board, h_black, h_white, to_play)
        masked = np.asarray(policy).reshape(-1).astype(np.float64).copy()
        masked[:len(legal)][~legal] = 0.0
        s = masked.sum()
        if s <= 0:
            return self.n_actions - 1
        if self.temp <= 0:
            return int(np.argmax(masked))
        probs = masked / s
        if self.topk and self.topk < len(probs):
            idx = np.argsort(probs)[::-1][:self.topk]
            p2 = np.zeros_like(probs)
            p2[idx] = probs[idx]
            probs = p2 / p2.sum()
        logp = np.log(probs + 1e-12)
        probs = np.exp(logp / max(self.temp, 1e-3))
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))


class MctsPlayer:
    """MCTS（blend=0）/ 策略+少量MCTS（blend>0，score=blend*policy+(1-blend)*visits）。"""

    def __init__(self, ai, board_size, sims=100, blend=0.0, priors_leaf=False,
                 expand_topk=32, expand_chunk=8, num_threads=1,
                 leaf_ab_depth=0, leaf_ab_width=4, policy_depth=1,
                 policy_width=4, policy_topk=12):
        self.ai = ai
        self.board_size = board_size
        self.n_actions = board_size * board_size + 1
        self.sims = sims
        self.blend = blend
        self.priors_leaf = priors_leaf
        self.expand_topk = expand_topk
        self.expand_chunk = expand_chunk
        self.num_threads = num_threads
        self.leaf_ab_depth = leaf_ab_depth
        self.leaf_ab_width = leaf_ab_width
        self.policy_depth = policy_depth
        self.policy_width = policy_width
        self.policy_topk = policy_topk
        self.mcts = None

    def new_game(self):
        # 每局新建搜索树：旧局的树对新局是错误先验（webui reset 也同理）
        self.mcts = MCTS(self.ai, board_size=self.board_size, num_threads=self.num_threads,
                         expand_topk=self.expand_topk, expand_chunk=self.expand_chunk,
                         priors_leaf=self.priors_leaf,
                         leaf_ab_depth=self.leaf_ab_depth, leaf_ab_width=self.leaf_ab_width)
        self.path_moves = []

    def select(self, board, h_black, h_white, to_play, legal, path_moves):
        self.path_moves = path_moves
        visits, _probs, root_value = self.mcts.search(
            board, h_black, h_white, to_play,
            simulations=self.sims, path_moves=self.path_moves)
        if self.blend <= 0.0:
            return int(np.argmax(visits))
        if self.policy_depth >= 2:
            # 策略分量 = 2 步批量推演价值分布（softmax(T=0.2) 伪概率）
            vals, _p, _bm, _bv = self.mcts.lookahead(
                board, h_black, h_white, to_play,
                topk=self.policy_topk, width=self.policy_width,
                depth=self.policy_depth)
            n = self.mcts.n_actions
            ks = np.full(n, -np.inf)
            for m, v in vals.items():
                ks[m] = v
            fin = np.isfinite(ks)
            pol_n = np.zeros(n)
            if fin.any():
                ex = np.exp((ks[fin] - ks[fin].max()) / 0.2)
                pol_n[fin] = ex / ex.sum()
        else:
            policy, _ = self.ai.predict(board, h_black, h_white, to_play)
            pol = np.asarray(policy).reshape(-1).astype(np.float64).copy()
            pol[:len(legal)][~legal] = 0.0
            ps = pol.sum()
            pol_n = pol / ps if ps > 0 else np.zeros_like(pol)
        vs = visits.astype(np.float64)
        vs_n = vs / vs.sum() if vs.sum() > 0 else np.zeros_like(vs)
        score = self.blend * pol_n + (1.0 - self.blend) * vs_n
        self.last_root_value = root_value
        return int(np.argmax(score))


class RandomPlayer:
    def __init__(self, ai, board_size):
        self.n_actions = board_size * board_size + 1

    def new_game(self):
        pass

    def select(self, board, h_black, h_white, to_play, legal, path_moves):
        return int(np.random.choice(np.where(legal)[0]))


def make_player(spec, ai, board_size, expand_topk, expand_chunk):
    """解析 label=type[,key=val]*。"""
    if "=" not in spec:
        raise ValueError(f"棋手规格需为 label=type[,key=val]*: {spec}")
    label, body = spec.split("=", 1)
    parts = [p for p in body.split(",") if p]
    ptype = parts[0]
    kv = {}
    for p in parts[1:]:
        k, v = p.split("=", 1)
        kv[k.strip()] = v.strip()
    if ptype == "random":
        return label, RandomPlayer(ai, board_size)
    if ptype == "policy":
        return label, PolicyPlayer(ai, board_size, temp=float(kv.get("temp", 0.0)),
                                   topk=int(kv.get("topk", 0)))
    if ptype in ("hybrid", "mcts"):
        return label, MctsPlayer(
            ai, board_size, sims=int(kv.get("sims", 100)),
            blend=float(kv.get("blend", 0.5 if ptype == "hybrid" else 0.0)),
            priors_leaf=kv.get("leaf", "0") in ("1", "true", "True"),
            expand_topk=expand_topk, expand_chunk=expand_chunk,
            leaf_ab_depth=int(kv.get("abdepth", 0)),
            leaf_ab_width=int(kv.get("abwidth", 4)),
            policy_depth=int(kv.get("pdepth", 1)),
            policy_width=int(kv.get("pwidth", 4)),
            policy_topk=int(kv.get("ptopk", 12)))
    raise ValueError(f"未知棋手类型: {ptype}")


# --------------------------------------------------------------------------- #
# 对弈
# --------------------------------------------------------------------------- #
def play_game(black, white, board_size, max_moves):
    """返回黑-白得分（>0 黑胜）。"""
    board = GoBoard(board_size)
    hists = [[-1, -1, -3], [-1, -1, -3]]  # [黑方最近3手, 白方最近3手]
    players = {1: black, -1: white}  # GoBoard.current_player: 1=黑, -1=白
    passes = 0
    mc = 0
    path_moves = []
    black.new_game()
    white.new_game()
    while passes < 2 and mc < max_moves:
        to_play = board.current_player
        legal = board.get_legal_moves()
        if len(legal) == 0:
            mv = board_size * board_size  # pass
        else:
            mv = players[to_play].select(board, hists[0], hists[1], to_play,
                                         legal, path_moves)
        pmv = -1 if mv == board_size * board_size else mv
        if not board.play(pmv):
            pmv = -1
            board.play(-1)
        path_moves.append(pmv)
        if pmv >= 0:
            h = hists[0] if to_play == 1 else hists[1]
            h.pop(0)
            h.append(pmv)
            passes = 0
        else:
            passes += 1
        mc += 1
    return board.score()


# --------------------------------------------------------------------------- #
# Elo 拟合（逻辑回归 MLE，全批量梯度下降；平局=0.5）
# --------------------------------------------------------------------------- #
def compute_elo(pair_stats, epochs=3000, lr=8.0, base=1500.0):
    """pair_stats: {(a,b): {"games": n, "wins": {a: wa, b: wb}, "draws": d}}，a<b。"""
    players = sorted({p for pair in pair_stats for p in pair})
    r = {p: 0.0 for p in players}
    for _ in range(epochs):
        grad = {p: 0.0 for p in players}
        for (a, b), st in pair_stats.items():
            n = st["games"]
            s = (st["wins"][a] + 0.5 * st["draws"]) / n  # a 的得分率
            s = min(max(s, 1e-4), 1 - 1e-4)
            E = 1.0 / (1.0 + 10.0 ** ((r[b] - r[a]) / 400.0))
            g = n * (s - E)
            grad[a] += g
            grad[b] -= g
        for p in players:
            r[p] += lr * grad[p] / max(1, len(pair_stats))
    shift = base - sum(r.values()) / len(r)
    return {p: r[p] + shift for p in players}


def main():
    ap = argparse.ArgumentParser(description="ELO 评测：多配置循环赛")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--games", type=int, default=10, help="每对棋手总对局数（黑白各半）")
    ap.add_argument("--max-moves", type=int, default=None,
                    help="单局手数上限（默认 3×棋盘点数）")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--expand-topk", type=int, default=32)
    ap.add_argument("--expand-chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--player", action="append", required=True,
                    help='棋手规格 label=type[,key=val]*，可多次传入')
    ap.add_argument("--save", type=str, default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ai = GoAI(model_path=args.model, board_size=args.board_size, device=args.device,
              use_amp=True,   # cuda/npu 启用 amp，cpu 内部自动忽略
              attn_mode="window", attn_window=7)
    args.board_size = ai.board_size  # 权重自带棋盘大小时自动校正
    max_moves = args.max_moves or 3 * args.board_size * args.board_size

    players = {}
    for spec in args.player:
        label, p = make_player(spec, ai, args.board_size, args.expand_topk, args.expand_chunk)
        players[label] = p
    labels = list(players)
    if len(labels) < 2:
        print("至少需要 2 个棋手")
        return

    # ---- 循环赛 ----
    import itertools
    pair_stats = {}
    t0 = time.perf_counter()
    for pa, pb in itertools.combinations(labels, 2):
        pair = (pa, pb) if pa < pb else (pb, pa)
        st = pair_stats.setdefault(pair, {"games": 0, "wins": {pair[0]: 0, pair[1]: 0}, "draws": 0})
        half = args.games // 2
        plan = [(pa, pb)] * (args.games - half) + [(pb, pa)] * half
        for i, (bk, wt) in enumerate(plan):
            score = play_game(players[bk], players[wt], args.board_size, max_moves)
            st["games"] += 1
            if score > 0:
                st["wins"][bk] += 1
            elif score < 0:
                st["wins"][wt] += 1
            else:
                st["draws"] += 1
            elapsed = time.perf_counter() - t0
            done = sum(s["games"] for s in pair_stats.values())
            total = sum(1 for _ in ()) or (len(list(itertools.combinations(labels, 2))) * args.games)
            print(f"\r[{done}/{total}] {bk}(黑) vs {wt}(白): score={score:+.1f}  "
                  f"{elapsed:.0f}s", end="", flush=True)
    print()

    # ---- Elo ----
    elo = compute_elo(pair_stats)
    games_played = {p: 0 for p in labels}
    for (a, b), st in pair_stats.items():
        games_played[a] += st["games"]
        games_played[b] += st["games"]

    print("\n===== ELO 排名 =====")
    for p in sorted(labels, key=lambda x: -elo[x]):
        print(f"  {p:24s} {elo[p]:7.1f}   ({games_played[p]} 局)")

    print("\n===== 两两战绩（行 vs 列，行方得分率）=====")
    hdr = " " * 26 + "".join(f"{q:>12s}" for q in labels)
    print(hdr)
    for a in labels:
        row = f"  {a:24s}"
        for b in labels:
            if a == b:
                row += f"{'—':>12s}"
                continue
            pair = (a, b) if a < b else (b, a)
            st = pair_stats.get(pair)
            if st is None:
                row += f"{'n/a':>12s}"
                continue
            sa = (st["wins"][a] + 0.5 * st["draws"]) / st["games"]
            row += f"{sa:>11.0%} " if a < b else f"{(1-sa):>11.0%} "
        print(row)

    if args.save:
        import json
        out = {"elo": elo, "games": games_played,
               "pairs": {f"{a} vs {b}": st for (a, b), st in pair_stats.items()}}
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.save}")


if __name__ == "__main__":
    main()
