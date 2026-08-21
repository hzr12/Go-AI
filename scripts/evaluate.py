#!/usr/bin/env python3
"""
评估脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.networks.resnet import MuZeroNet
from src.evaluation.evaluator import Evaluator


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估围棋AI')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--board-size', type=int, default=9, help='棋盘大小')
    parser.add_argument('--search-depth', type=int, default=10, help='搜索深度')
    parser.add_argument('--top-k', type=int, default=5, help='候选着法数量')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--num-games', type=int, default=100, help='评估局数')
    parser.add_argument('--mode', type=str, default='elo', 
                       choices=['random', 'elo', 'benchmark'],
                       help='评估模式')
    
    args = parser.parse_args()
    
    # 创建网络
    model = MuZeroNet(
        in_channels=19,
        channels=64,
        num_res_blocks=4,
        action_size=args.board_size * args.board_size
    )
    
    # 加载模型（如果提供）
    if args.model:
        checkpoint = torch.load(args.model, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {args.model}")
    
    # 创建评估器
    evaluator = Evaluator(
        model=model,
        board_size=args.board_size,
        search_depth=args.search_depth,
        top_k=args.top_k,
        device=args.device
    )
    
    if args.mode == 'random':
        # 与随机玩家对弈
        print(f"Evaluating against random player ({args.num_games} games)...")
        results = evaluator.evaluate_against_random(num_games=args.num_games)
        print(f"\nResults:")
        print(f"  Wins: {results['wins']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Draws: {results['draws']}")
        print(f"  Win rate: {results['win_rate']:.2%}")
    
    elif args.mode == 'elo':
        # ELO评分
        print(f"Evaluating ELO rating ({args.num_games} games)...")
        results = evaluator.evaluate_elo(num_games=args.num_games)
        print(f"\nResults:")
        print(f"  ELO rating: {results['elo']:.0f}")
        print(f"  Number of games: {results['num_games']}")
    
    elif args.mode == 'benchmark':
        # 性能基准测试
        print(f"Running benchmark ({args.num_games} positions)...")
        results = evaluator.benchmark(num_positions=args.num_games)
        print(f"\nResults:")
        print(f"  Average time: {results['avg_time']:.4f}s")
        print(f"  Max time: {results['max_time']:.4f}s")
        print(f"  Min time: {results['min_time']:.4f}s")
        print(f"  Positions per second: {results['positions_per_second']:.2f}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
