#!/usr/bin/env python3
"""
Demo脚本 - 快速体验围棋AI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.inference import GoAI
from src.utils.helpers import print_board


def demo_basic_usage():
    """演示基本用法"""
    print("=" * 50)
    print("围棋AI Demo")
    print("=" * 50)
    
    # 创建AI
    ai = GoAI(
        board_size=9,
        search_depth=3,  # 使用较浅的搜索以加快演示
        top_k=3,
        device='cpu'
    )
    
    print("\n1. 创建AI...")
    print(f"   棋盘大小: {ai.board_size}")
    print(f"   搜索深度: {ai.search.search_depth}")
    print(f"   候选着法: {ai.search.top_k}")
    
    # 显示空棋盘
    print("\n2. 空棋盘:")
    print_board(ai.board)
    
    # 获取AI着法
    print("\n3. AI思考...")
    move, score, info = ai.get_move()
    row, col = move // ai.board_size, move % ai.board_size
    print(f"   AI选择: {row} {col} (评估值: {score:.4f})")
    
    # 显示Top-5候选
    print("\n4. Top-5候选着法:")
    suggestions = ai.suggest_moves(num_moves=5)
    for i, s in enumerate(suggestions):
        row, col = s['position']
        print(f"   {i+1}. {row} {col} (概率: {s['probability']:.4f})")
    
    # 执行着法
    print("\n5. 执行着法...")
    ai.make_move(move)
    print_board(ai.board)
    
    # 评估新局面
    print("\n6. 评估新局面...")
    eval_result = ai.evaluate_position()
    print(f"   最佳着法: {eval_result['best_move']}")
    print(f"   置信度: {eval_result['best_prob']:.4f}")


def demo_game():
    """演示简单对弈"""
    print("\n" + "=" * 50)
    print("简单对弈演示")
    print("=" * 50)
    
    ai = GoAI(
        board_size=9,
        search_depth=2,  # 非常浅的搜索
        top_k=3,
        device='cpu'
    )
    
    print("\n进行5步对弈...")
    
    for step in range(5):
        print(f"\n--- 第{step+1}步 ---")
        
        # 获取AI着法
        move, score, info = ai.get_move()
        row, col = move // ai.board_size, move % ai.board_size
        
        print(f"玩家{'黑' if ai.current_player == 1 else '白'}着法: {row} {col}")
        print(f"评估值: {score:.4f}")
        
        # 执行着法
        ai.make_move(move)
        
        # 显示棋盘
        print_board(ai.board)
    
    print("\n对弈演示完成！")


def main():
    """主函数"""
    print("围棋AI Demo")
    print("=" * 50)
    
    # 演示基本用法
    demo_basic_usage()
    
    # 演示对弈
    demo_game()
    
    print("\n" + "=" * 50)
    print("Demo完成！")
    print("=" * 50)
    print("\n要开始人机对弈，请运行:")
    print("  python -m src.inference --mode play")
    print("\n要训练模型，请运行:")
    print("  python scripts/train.py --num-games 100")


if __name__ == '__main__':
    main()
