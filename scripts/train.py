#!/usr/bin/env python3
"""
训练脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.networks.resnet import MuZeroNet
from src.training.trainer import Trainer
from src.config.config import Config, get_config


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练围棋AI')
    parser.add_argument('--config', type=str, default='default', 
                       help='配置名称 (default/fast/accurate/cpu/gpu)')
    parser.add_argument('--board-size', type=int, default=9, help='棋盘大小')
    parser.add_argument('--channels', type=int, default=64, help='网络通道数')
    parser.add_argument('--num-res-blocks', type=int, default=4, help='残差块数量')
    parser.add_argument('--search-depth', type=int, default=10, help='推理搜索深度')
    parser.add_argument('--search-depth-self-play', type=int, default=3, help='自我对弈搜索深度（越小越快）')
    parser.add_argument('--top-k', type=int, default=5, help='候选着法数量')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--batch-size', type=int, default=64, help='批量大小')
    parser.add_argument('--num-games', type=int, default=5000, help='训练局数')
    parser.add_argument('--games-per-batch', type=int, default=256, help='每批对弈局数')
    parser.add_argument('--save-path', type=str, default='models/model.pth', help='模型保存路径')
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto/cpu/cuda)')
    parser.add_argument('--use-amp', action='store_true', help='使用自动混合精度训练')
    
    args = parser.parse_args()
    
    # 获取配置
    if args.config != 'default':
        config = get_config(args.config)
    else:
        config = Config(
            board_size=args.board_size,
            channels=args.channels,
            num_res_blocks=args.num_res_blocks,
            search_depth=args.search_depth,
            top_k=args.top_k,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_games=args.num_games,
            device=args.device
        )
    
    # 检测是否使用AMP
    use_amp = args.use_amp or config.device == 'cuda'
    
    print(f"Configuration: {args.config}")
    print(f"Board size: {config.board_size}")
    print(f"Channels: {config.channels}")
    print(f"Res blocks: {config.num_res_blocks}")
    print(f"Search depth (inference): {config.search_depth}")
    print(f"Search depth (self-play): {args.search_depth_self_play}")
    print(f"Top K: {config.top_k}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num games: {config.num_games}")
    print(f"Device: {config.device}")
    print(f"Use AMP: {use_amp}")
    
    # 创建网络
    model = MuZeroNet(
        in_channels=config.in_channels,
        channels=config.channels,
        num_res_blocks=config.num_res_blocks,
        action_size=config.action_size
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        board_size=config.board_size,
        search_depth=config.search_depth,
        search_depth_self_play=args.search_depth_self_play,
        top_k=config.top_k,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        device=config.device,
        use_amp=use_amp
    )
    
    # 开始训练
    print(f"\nStarting training...")
    print(f"Save path: {args.save_path}")
    
    trainer.train(
        num_games=config.num_games,
        games_per_batch=args.games_per_batch,
        save_interval=config.save_interval,
        save_path=args.save_path
    )
    
    print(f"\nTraining completed!")


if __name__ == '__main__':
    main()
