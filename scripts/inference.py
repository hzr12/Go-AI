#!/usr/bin/env python3
"""
推理脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import GoAI


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='围棋AI推理')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--board-size', type=int, default=9, help='棋盘大小')
    parser.add_argument('--search-depth', type=int, default=10, help='搜索深度')
    parser.add_argument('--top-k', type=int, default=5, help='候选着法数量')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--use-amp', action='store_true', help='使用自动混合精度')
    parser.add_argument('--mode', type=str, default='play', 
                       choices=['play', 'eval', 'analyze'],
                       help='运行模式')
    
    args = parser.parse_args()
    
    # 检测是否使用AMP
    use_amp = args.use_amp or args.device == 'cuda'
    
    # 创建AI
    ai = GoAI(
        model_path=args.model,
        board_size=args.board_size,
        search_depth=args.search_depth,
        top_k=args.top_k,
        device=args.device,
        use_amp=use_amp
    )
    
    if args.mode == 'play':
        # 人机对弈
        ai.play_against_human()
    elif args.mode == 'eval':
        # 评估当前局面
        eval_result = ai.evaluate_position()
        print(f"最佳着法: {eval_result['best_move']}")
        print(f"置信度: {eval_result['best_prob']:.4f}")
        print(f"Top-5候选: {eval_result['top_moves']}")
    elif args.mode == 'analyze':
        # 分析着法（示例：分析位置0）
        analysis = ai.analyze_move(0)
        print(f"着法分析: {analysis}")


if __name__ == '__main__':
    main()
