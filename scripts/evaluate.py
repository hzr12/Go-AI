#!/usr/bin/env python3
"""评估 SFT 围棋 AI：自对弈胜率 / 对随机策略 / 推理速度基准 / MCTS 对比。

基于 src.inference.GoAI（12 通道特征 + GoBoard 规则），不依赖已删除的旧 MuZero 代码。
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


def load_ai(args):
    device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    ai = GoAI(
        model_path=args.model, board_size=args.board_size, device=device,
        use_amp=args.use_amp, attention_mode=args.attention_mode,
        num_attention_layers=args.num_attention_layers, num_heads=args.num_heads,
        attn_mode=args.attn_mode, attn_window=args.attn_window, compile=args.compile,
    )
    return ai, device


def evaluate_vs_random(ai, board_size, num_games=100, use_mcts=False, simulations=400, num_threads=4):
    """模型（黑）对随机策略（白），返回胜率（模型视角）。"""
    wins = losses = draws = 0
    for g in range(num_games):
        board = GoBoard(board_size)
        my_hist = [[-1, -1, -3], [-1, -1, -3]]
        passes = 0
        mc = 0
        while passes < 2 and mc < 400:
            to_play = board.current_player
            legal = board.get_legal_moves()
            if len(legal) == 0:
                board.play(-1); passes += 1; mc += 1; continue
            if to_play == 1:  # 模型执黑
                if use_mcts:
                    mv, _, _ = ai.choose_move_mcts(board, my_hist[0], my_hist[1], to_play,
                                                  legal, simulations=simulations, num_threads=num_threads)
                else:
                    mv, _, _ = ai.choose_move(board, my_hist[0], my_hist[1], to_play, legal, topk=p.topk)
                if mv == board_size * board_size:
                    board.play(-1)
                else:
                    board.play(mv)
                    h = my_hist[0]; h.pop(0); h.append(mv)
                passes = 0 if mv != board_size * board_size else passes + 1
            else:  # 随机执白
                mv = int(np.random.choice(legal))
                board.play(mv)
                h = my_hist[1]; h.pop(0); h.append(mv)
                passes = 0
            mc += 1
        score = board.score()
        if score > 0:
            wins += 1
        elif score < 0:
            losses += 1
        else:
            draws += 1
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / num_games}


def benchmark(ai, board_size, num_positions=200):
    """推理速度基准（单样本 vs 批量）。"""
    board = GoBoard(board_size)
    my_hist = [[-1, -1, -3], [-1, -1, -3]]
    to_play = 1
    # 单样本
    t0 = time.perf_counter()
    for _ in range(num_positions):
        ai.predict(board, my_hist[0], my_hist[1], to_play)
    single = (time.perf_counter() - t0) / num_positions
    # 批量（模拟 MCTS 叶子评估）
    states = [(board, list(my_hist[0]), list(my_hist[1]), to_play)] * 32
    t0 = time.perf_counter()
    for _ in range(num_positions // 32 + 1):
        ai.predict_batch(states)
    batched = (time.perf_counter() - (t0)) / max(num_positions // 32, 1)
    return {"avg_time_single": single, "avg_time_batch32": batched,
            "throughput_single": 1 / single, "throughput_batch": 32 / batched}


def main():
    parser = argparse.ArgumentParser(description="评估围棋 SFT AI")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--attention-mode", default="mix", choices=["none", "mix", "all"])
    parser.add_argument("--num-attention-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-mode", default="global", choices=["global", "window", "axial"])
    parser.add_argument("--attn-window", type=int, default=7)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "benchmark", "selfplay"])
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--use-mcts", action="store_true")
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--topk", type=int, default=10)
    p = parser.parse_args()

    ai, device = load_ai(p)
    if p.mode == "random":
        res = evaluate_vs_random(ai, p.board_size, num_games=p.num_games,
                                 use_mcts=p.use_mcts, simulations=p.simulations,
                                 num_threads=p.num_threads)
        tag = "MCTS" if p.use_mcts else "policy-argmax"
        print(f"[{tag}] vs 随机 {p.num_games} 局: 胜 {res['wins']} 负 {res['losses']} 平 {res['draws']} "
              f"胜率 {res['win_rate']:.1%}")
    elif p.mode == "benchmark":
        res = benchmark(ai, p.board_size)
        print(f"单样本前向: {res['avg_time_single']*1000:.2f} ms "
              f"({res['throughput_single']:.1f} pos/s)")
        print(f"批量32前向: {res['avg_time_batch32']*1000:.2f} ms "
              f"({res['throughput_batch']:.1f} pos/s)  [MCTS 叶子评估]")
    elif p.mode == "selfplay":
        res = ai.self_play(num_games=p.num_games, use_mcts=p.use_mcts,
                           simulations=p.simulations, num_threads=p.num_threads)
        wr = sum(1 for r in res if r > 0) / max(len(res), 1)
        print(f"自对弈 {len(res)} 局 黑方胜率 {wr:.2%}")


if __name__ == "__main__":
    main()
