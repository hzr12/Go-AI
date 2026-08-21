#!/usr/bin/env python3
"""
推理脚本 - AlphaGo多网络版本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import GoAI


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaGo围棋AI推理')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--board-size', type=int, default=9, help='棋盘大小')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--use-amp', action='store_true', help='使用自动混合精度')
    parser.add_argument('--use-value', action='store_true', default=True, help='使用价值网络')
    parser.add_argument('--mode', type=str, default='play', 
                       choices=['play', 'eval', 'analyze'],
                       help='运行模式')
    
    args = parser.parse_args()
    
    # 创建AI
    ai = GoAI(
        model_path=args.model,
        board_size=args.board_size,
        device=args.device,
        use_amp=args.use_amp,
        use_value=args.use_value
    )
    
    if args.mode == 'play':
        # 人机对弈
        ai.play_against_human()
    elif args.mode == 'eval':
        # 评估当前局面
        eval_result = ai.evaluate_position()
        print(f"最佳着法: {eval_result['best_move']}")
        print(f"置信度: {eval_result['best_prob']:.4f}")
        print(f"价值: {eval_result['value']:.4f}")
    elif args.mode == 'analyze':
        # 分析着法
        eval_result = ai.evaluate_position()
        print(f"策略网络Top-5:")
        import numpy as np
        top5 = np.argsort(eval_result['policy'])[-5:][::-1]
        for i, move in enumerate(top5):
            row, col = move // ai.board_size, move % ai.board_size
            print(f"  {i+1}. {row} {col}: {eval_result['policy'][move]:.4f}")
        
        print(f"\n快速策略Top-5:")
        top5_fast = np.argsort(eval_result['fast_policy'])[-5:][::-1]
        for i, move in enumerate(top5_fast):
            row, col = move // ai.board_size, move % ai.board_size
            print(f"  {i+1}. {row} {col}: {eval_result['fast_policy'][move]:.4f}")


if __name__ == '__main__':
    main()
