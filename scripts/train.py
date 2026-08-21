#!/usr/bin/env python3
"""
训练脚本 - AlphaGo多网络版本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.networks.alphanet import AlphaGoNet
from src.training.trainer import AlphaGoTrainer
from src.data.game_loader import GameLoader


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练AlphaGo围棋AI')
    
    # 基本参数
    parser.add_argument('--board-size', type=int, default=9, help='棋盘大小')
    parser.add_argument('--backbone-channels', type=int, default=64, help='骨干网络通道数')
    parser.add_argument('--backbone-res-blocks', type=int, default=4, help='骨干网络残差块数量')
    parser.add_argument('--policy-channels', type=int, default=32, help='策略网络通道数')
    parser.add_argument('--value-channels', type=int, default=16, help='价值网络通道数')
    parser.add_argument('--fast-channels', type=int, default=72, help='快速策略网络通道数')
    parser.add_argument('--fast-res-blocks', type=int, default=3, help='快速策略网络残差块数量')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--batch-size', type=int, default=256, help='批量大小')
    parser.add_argument('--num-games', type=int, default=5000, help='训练局数')
    parser.add_argument('--games-per-batch', type=int, default=256, help='每批对弈局数')
    parser.add_argument('--save-path', type=str, default='models/alphago_model.pth', help='模型保存路径')
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto/cpu/cuda)')
    parser.add_argument('--use-amp', action='store_true', help='使用自动混合精度')
    parser.add_argument('--policy-weight', type=float, default=1.0, help='策略损失权重')
    parser.add_argument('--value-weight', type=float, default=1.0, help='价值损失权重')
    parser.add_argument('--fast-weight', type=float, default=0.5, help='快速策略损失权重')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    parser.add_argument('--top-k', type=int, default=5, help='候选着法数量')
    
    # 稀疏注意力参数
    parser.add_argument('--use-sparse-attention', action='store_true', help='启用稀疏注意力')
    parser.add_argument('--attention-window-size', type=int, default=3, help='局部窗口大小')
    parser.add_argument('--attention-num-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--attention-num-global-tokens', type=int, default=9, help='全局token数量')
    
    # SGF预训练参数
    parser.add_argument('--pretrain-data', type=str, default='', help='棋谱数据目录')
    parser.add_argument('--pretrain-epochs', type=int, default=10, help='预训练轮数')
    parser.add_argument('--pretrain-augment', action='store_true', default=True, help='预训练数据增强')
    
    args = parser.parse_args()
    
    # 自动检测设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 检测是否使用AMP
    use_amp = args.use_amp or device == 'cuda'
    
    print(f"Configuration:")
    print(f"  Board size: {args.board_size}")
    print(f"  Backbone: {args.backbone_channels}ch × {args.backbone_res_blocks}rb")
    print(f"  Policy: {args.policy_channels}ch")
    print(f"  Value: {args.value_channels}ch")
    print(f"  Fast: {args.fast_channels}ch × {args.fast_res_blocks}rb")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num games: {args.num_games}")
    print(f"  Device: {device}")
    print(f"  Use AMP: {use_amp}")
    print(f"  Loss weights: P={args.policy_weight}, V={args.value_weight}, F={args.fast_weight}")
    print(f"  Sparse attention: {args.use_sparse_attention}")
    if args.use_sparse_attention:
        print(f"    Window size: {args.attention_window_size}")
        print(f"    Num heads: {args.attention_num_heads}")
        print(f"    Global tokens: {args.attention_num_global_tokens}")
    if args.pretrain_data:
        print(f"  Pretrain data: {args.pretrain_data}")
        print(f"  Pretrain epochs: {args.pretrain_epochs}")
    
    # 创建网络
    model = AlphaGoNet(
        in_channels=19,
        backbone_channels=args.backbone_channels,
        backbone_res_blocks=args.backbone_res_blocks,
        policy_channels=args.policy_channels,
        value_channels=args.value_channels,
        fast_channels=args.fast_channels,
        fast_res_blocks=args.fast_res_blocks,
        action_size=args.board_size * args.board_size,
        use_sparse_attention=args.use_sparse_attention,
        attention_window_size=args.attention_window_size,
        attention_num_heads=args.attention_num_heads,
        attention_num_global_tokens=args.attention_num_global_tokens
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    policy_params = sum(p.numel() for p in model.policy.parameters())
    value_params = sum(p.numel() for p in model.value.parameters())
    fast_params = sum(p.numel() for p in model.fast_policy.parameters())
    
    print(f"\nNetwork parameters:")
    print(f"  Total: {total_params:,} ({total_params/1000:.1f}K)")
    print(f"  Backbone: {backbone_params:,} ({backbone_params/1000:.1f}K)")
    print(f"  Policy: {policy_params:,} ({policy_params/1000:.1f}K)")
    print(f"  Value: {value_params:,} ({value_params/1000:.1f}K)")
    print(f"  Fast: {fast_params:,} ({fast_params/1000:.1f}K)")
    
    # 创建训练器
    trainer = AlphaGoTrainer(
        model=model,
        board_size=args.board_size,
        lr=args.lr,
        weight_decay=1e-4,
        batch_size=args.batch_size,
        buffer_size=10000,
        device=device,
        use_amp=use_amp,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        fast_weight=args.fast_weight,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    # SGF预训练
    if args.pretrain_data:
        print(f"\nLoading pretrain data from: {args.pretrain_data}")
        loader = GameLoader(args.board_size)
        if os.path.isfile(args.pretrain_data):
            game_records = loader.load_tgz(args.pretrain_data)
        else:
            game_records = loader.load_directory(args.pretrain_data)
        print(f"Loaded {len(game_records)} games")
        
        if game_records:
            trainer.pretrain_on_games(
                game_records=game_records,
                epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
                augment=args.pretrain_augment
            )
    
    # 开始训练
    print(f"\nStarting training...")
    print(f"Save path: {args.save_path}")
    
    trainer.train(
        num_games=args.num_games,
        games_per_batch=args.games_per_batch,
        save_interval=100,
        save_path=args.save_path
    )
    
    print(f"\nTraining completed!")


if __name__ == '__main__':
    main()
