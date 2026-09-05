#!/usr/bin/env python3
"""AlphaZero 式自我对弈无监督训练（自对弈生成数据 → 训练 → 循环）。

每轮迭代：
  1. 自对弈 --games 局（MCTS visit 分布当 policy 标签，终局胜负当 value 标签，
     根先验混 Dirichlet 噪声 + 温度采样保证探索）
  2. 数据 8 对称增强（4 旋转 × 2 镜像）入 replay buffer
  3. 训练 --epochs 遍（policy 软标签交叉熵 + value MSE），覆盖式保存权重

CPU 上建议 9 路小规模验证流程；19 路正式训练请上 GPU/NPU 并加大 --sims。
每个迭代保存 models/az_<size>_iter<N>.pth，可用 eval_elo.py 对比新旧棋力。

用法:
    python scripts/selfplay_train.py --board-size 9 --iters 5 --games 4 --sims 32 \
        --model models/sft_19x19_v3.pth --out models/az
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.inference import GoAI
from src.game.go_rules import GoBoard
from src.search.mcts import MCTS


# --------------------------------------------------------------------------- #
# 自对弈数据生成
# --------------------------------------------------------------------------- #
def self_play_game(ai, board_size, sims, max_moves, temperature,
                   expand_topk, expand_chunk, priors_leaf=True,
                   dir_alpha=0.3, dir_eps=0.25):
    """一局自对弈。返回 [(planes, visit_target, player), ...], score(黑-白)。"""
    mcts = MCTS(ai, board_size=board_size, num_threads=1,
                expand_topk=expand_topk, expand_chunk=expand_chunk,
                priors_leaf=priors_leaf, temperature=temperature,
                dirichlet_alpha=dir_alpha, dirichlet_eps=dir_eps)
    board = GoBoard(board_size)
    hists = [[-1, -1, -3], [-1, -1, -3]]  # [黑方, 白方] 最近3手
    n_actions = board_size * board_size + 1
    passes = 0
    mc = 0
    path_moves = []
    data = []
    while passes < 2 and mc < max_moves:
        to_play = board.current_player
        legal = board.get_legal_moves()
        if len(legal) == 0:
            board.play(-1)
            path_moves.append(-1)
            passes += 1
            mc += 1
            continue
        visits, probs, _rv = mcts.search(
            board, hists[0], hists[1], to_play,
            simulations=sims, path_moves=path_moves)
        # 记录训练样本：当前局面特征 + visit 分布（policy 目标）+ 执子方
        planes = np.ascontiguousarray(board.feature_planes(hists[0], hists[1], to_play))
        vt = np.zeros(n_actions)
        vs = visits.sum()
        if vs > 0:
            vt[:n_actions - 1] = visits[:n_actions - 1] / vs
            vt[n_actions - 1] = visits[n_actions - 1] / vs
        data.append((planes, vt, to_play))
        # 按温度分布采样落子（探索）
        p = np.asarray(probs).reshape(-1).astype(np.float64)
        p[-1] = max(p[-1], 0.0)  # pass 概率
        s = p.sum()
        if s <= 0:
            mv = n_actions - 1
        else:
            mv = int(np.random.choice(n_actions, p=p / s))
        pmv = -1 if mv == n_actions - 1 else mv
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
    return data, board.score()


def augment8(plane, target, n):
    """8 对称增强：4 旋转 × 2 镜像（棋盘特征与 visit 目标同步变换）。"""
    board_t = target[:n * n].reshape(n, n)
    pass_t = target[n * n]
    out = []
    for k in range(4):
        for fl in (False, True):
            pl = np.rot90(plane, k, axes=(1, 2))
            tb = np.rot90(board_t, k)
            if fl:
                pl = pl[:, ::-1]
                tb = tb[:, ::-1]
            tv = np.concatenate([tb.reshape(-1), [pass_t]])
            out.append((np.ascontiguousarray(pl), np.ascontiguousarray(tv)))
    return out


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train_epochs(ai, buffer, args, device):
    """在 replay buffer 上训练若干遍。返回平均 loss。"""
    model = ai.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    losses = []
    if not buffer:
        return 0.0
    n = len(buffer)
    steps_per_epoch = max(1, n // args.batch_size)
    for _ in range(args.epochs):
        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, n, size=args.batch_size)
            batch = [buffer[i] for i in idx]
            planes = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(device)
            pi_t = torch.from_numpy(np.stack([b[1] for b in batch])).float().to(device)
            z = torch.tensor([b[2] for b in batch], dtype=torch.float32,
                             device=device).unsqueeze(1)
            policy, value = model(planes)
            logq = torch.log_softmax(policy, dim=-1) + 1e-10
            loss_pi = -(pi_t * logq).sum(dim=1).mean()
            loss_v = ((value.squeeze(-1) - z) ** 2).mean()
            loss = loss_pi + loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    model.eval()
    return sum(losses) / max(len(losses), 1)


def main():
    ap = argparse.ArgumentParser(description="AlphaZero 式自对弈无监督训练")
    ap.add_argument("--model", type=str, default=None, help="初始权重（如 SFT 预训练）")
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--iters", type=int, default=5, help="迭代轮数")
    ap.add_argument("--games", type=int, default=4, help="每轮自对弈局数")
    ap.add_argument("--sims", type=int, default=32, help="自对弈每步 MCTS 模拟数")
    ap.add_argument("--max-moves", type=int, default=None, help="单局手数上限（默认 3×点数）")
    ap.add_argument("--temperature", type=float, default=1.0, help="自对弈采样温度")
    ap.add_argument("--buffer-size", type=int, default=20000, help="replay buffer 容量（样本数，增强后）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2, help="每轮迭代训练遍数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--expand-topk", type=int, default=16)
    ap.add_argument("--expand-chunk", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no-augment", action="store_true", help="关闭 8 对称增强")
    ap.add_argument("--out", type=str, default="models/az", help="权重输出目录")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    ai = GoAI(model_path=args.model, board_size=args.board_size, device=device,
              attn_mode="window", attn_window=7)
    bs = ai.board_size
    n_actions = bs * bs + 1
    max_moves = args.max_moves or 3 * bs * bs
    os.makedirs(args.out, exist_ok=True)

    buffer = []   # [(planes(12,n,n), target(n²+1), z)]
    total_games = 0
    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        for g in range(args.games):
            data, score = self_play_game(
                ai, bs, args.sims, max_moves, args.temperature,
                args.expand_topk, args.expand_chunk)
            total_games += 1
            # z: 终局胜负（执子方视角）
            for planes, vt, player in data:
                if score > 0:
                    z = 1.0 if player == 1 else -1.0
                elif score < 0:
                    z = -1.0 if player == 1 else 1.0
                else:
                    z = 0.0
                samples = augment8(planes, vt, bs) if not args.no_augment else \
                    [(planes, vt)]
                for pl, tv in samples:
                    buffer.append((pl, tv, z))
            while len(buffer) > args.buffer_size:
                buffer.pop(0)
            print(f"  [iter {it} game {g + 1}/{args.games}] score={score:+.1f} "
                  f"moves={len(data)} buffer={len(buffer)}", flush=True)

        avg_loss = train_epochs(ai, buffer, args, device)
        dt = time.perf_counter() - t0
        out_path = os.path.join(args.out, f"az_{bs}_iter{it}.pth")
        torch.save({"model": ai.model.state_dict(), "iter": it,
                    "board_size": bs, "args": vars(args)}, out_path)
        print(f"[iter {it}/{args.iters}] loss={avg_loss:.4f} buffer={len(buffer)} "
              f"games={total_games} {dt:.0f}s -> {out_path}", flush=True)

    print("训练完成。用 scripts/eval_elo.py 对比不同迭代权重棋力。")


if __name__ == "__main__":
    main()
