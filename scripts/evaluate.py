#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.networks.alphanet import AlphaGoNet
from src.evaluation.alpha_evaluator import AlphaEvaluator


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='评估AlphaGo围棋AI')
    parser.add_argument('--model', type=str, required=True, help='模型路径(.pth)')
    parser.add_argument('--board-size', type=int, default=9, help='棋盘大小')
    parser.add_argument('--backbone-channels', type=int, default=64)
    parser.add_argument('--backbone-res-blocks', type=int, default=4)
    parser.add_argument('--policy-channels', type=int, default=32)
    parser.add_argument('--value-channels', type=int, default=16)
    parser.add_argument('--fast-channels', type=int, default=72)
    parser.add_argument('--fast-res-blocks', type=int, default=3)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num-games', type=int, default=100, help='评估局数')
    parser.add_argument('--mode', type=str, default='random',
                       choices=['random', 'elo', 'benchmark'],
                       help='评估模式: random=vs随机, elo=ELO评分, benchmark=推理速度')
    parser.add_argument('--opponent-elo', type=float, default=1000, help='ELO模式对手评分')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    action_size = args.board_size * args.board_size
    
    model = AlphaGoNet(
        in_channels=19,
        backbone_channels=args.backbone_channels,
        backbone_res_blocks=args.backbone_res_blocks,
        policy_channels=args.policy_channels,
        value_channels=args.value_channels,
        fast_channels=args.fast_channels,
        fast_res_blocks=args.fast_res_blocks,
        action_size=action_size
    )
    
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded: {args.model}")
    else:
        print(f"Warning: {args.model} not found, using random weights")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1000:.1f}K)")
    print(f"Device: {device}")
    
    evaluator = AlphaEvaluator(model=model, board_size=args.board_size, device=device)
    
    if args.mode == 'random':
        print(f"\nEvaluating vs random ({args.num_games} games)...")
        results = evaluator.evaluate_against_random(num_games=args.num_games)
        print(f"\nResults:")
        print(f"  Wins:   {results['wins']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Draws:  {results['draws']}")
        print(f"  Win rate: {results['win_rate']:.1%}")
    
    elif args.mode == 'elo':
        print(f"\nELO evaluation ({args.num_games} games)...")
        results = evaluator.evaluate_elo(num_games=args.num_games, opponent_elo=args.opponent_elo)
        print(f"\nResults:")
        print(f"  ELO rating: {results['elo']:.0f}")
        print(f"  Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")
    
    elif args.mode == 'benchmark':
        print(f"\nBenchmark ({args.num_positions if hasattr(args, 'num_positions') else args.num_games} positions)...")
        n = args.num_games
        results = evaluator.benchmark(num_positions=n)
        print(f"\nResults:")
        print(f"  Avg time:     {results['avg_time']*1000:.1f}ms")
        print(f"  Min time:     {results['min_time']*1000:.1f}ms")
        print(f"  Max time:     {results['max_time']*1000:.1f}ms")
        print(f"  Throughput:   {results['positions_per_second']:.1f} positions/s")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
