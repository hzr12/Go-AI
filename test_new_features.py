#!/usr/bin/env python3
"""测试新功能"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def test_sparse_attention():
    """测试稀疏注意力模块"""
    print("Testing sparse attention...")
    
    from src.networks.sparse_attention import (
        LocalWindowAttention, GlobalTokenAttention, 
        HybridSparseAttention, SparseAttentionBlock
    )
    
    # 测试局部窗口注意力
    local_attn = LocalWindowAttention(channels=64, window_size=3)
    x = torch.randn(1, 64, 9, 9)
    out = local_attn(x)
    assert out.shape == (1, 64, 9, 9), f"Local attention output shape: {out.shape}"
    print("  LocalWindowAttention: OK")
    
    # 测试全局token注意力
    global_attn = GlobalTokenAttention(channels=64, num_heads=4, num_global_tokens=9)
    out = global_attn(x)
    assert out.shape == (1, 64, 9, 9), f"Global attention output shape: {out.shape}"
    print("  GlobalTokenAttention: OK")
    
    # 测试混合稀疏注意力
    hybrid_attn = HybridSparseAttention(channels=64, window_size=3, num_heads=4, num_global_tokens=9)
    out = hybrid_attn(x)
    assert out.shape == (1, 64, 9, 9), f"Hybrid attention output shape: {out.shape}"
    print("  HybridSparseAttention: OK")
    
    # 测试稀疏注意力块
    block = SparseAttentionBlock(channels=64, window_size=3, num_heads=4, num_global_tokens=9)
    out = block(x)
    assert out.shape == (1, 64, 9, 9), f"SparseAttentionBlock output shape: {out.shape}"
    print("  SparseAttentionBlock: OK")
    
    print("Sparse attention tests passed!\n")


def test_19x19_support():
    """测试19×19棋盘支持"""
    print("Testing 19×19 support...")
    
    from src.networks.alphanet import AlphaGoNet
    
    # 测试9×9
    model_9x9 = AlphaGoNet(
        in_channels=19,
        backbone_channels=64,
        backbone_res_blocks=4,
        action_size=81
    )
    x_9x9 = torch.randn(1, 19, 9, 9)
    policy_9x9, value_9x9, fast_9x9 = model_9x9(x_9x9)
    assert policy_9x9.shape == (1, 81), f"9x9 policy shape: {policy_9x9.shape}"
    print("  9×9: OK")
    
    # 测试19×19
    model_19x19 = AlphaGoNet(
        in_channels=19,
        backbone_channels=64,
        backbone_res_blocks=4,
        action_size=361
    )
    x_19x19 = torch.randn(1, 19, 19, 19)
    policy_19x19, value_19x19, fast_19x19 = model_19x19(x_19x19)
    assert policy_19x19.shape == (1, 361), f"19x19 policy shape: {policy_19x19.shape}"
    print("  19×19: OK")
    
    # 测试19×19 + 稀疏注意力
    model_19x19_sparse = AlphaGoNet(
        in_channels=19,
        backbone_channels=64,
        backbone_res_blocks=4,
        action_size=361,
        use_sparse_attention=True,
        attention_window_size=3,
        attention_num_heads=4,
        attention_num_global_tokens=9
    )
    policy_19x19_sparse, value_19x19_sparse, fast_19x19_sparse = model_19x19_sparse(x_19x19)
    assert policy_19x19_sparse.shape == (1, 361), f"19x19 sparse policy shape: {policy_19x19_sparse.shape}"
    print("  19×19 + Sparse Attention: OK")
    
    print("19×19 support tests passed!\n")


def test_sgf_parser():
    """测试SGF解析器"""
    print("Testing SGF parser...")
    
    from src.data.sgf_parser import SGFParser, GameRecord
    
    parser = SGFParser()
    
    # 测试解析SGF字符串
    sgf_string = "(;FF[4]SZ[19]RE[B+2.5]PB[Black]PW[White]KM[6.5];B[pd];W[dp];B[pp])"
    game = parser.parse_string(sgf_string)
    
    assert game is not None, "Failed to parse SGF string"
    assert game.board_size == 19, f"Board size: {game.board_size}"
    assert game.result == "B+2.5", f"Result: {game.result}"
    assert game.black_player == "Black", f"Black player: {game.black_player}"
    assert game.white_player == "White", f"White player: {game.white_player}"
    assert len(game.moves) == 3, f"Number of moves: {len(game.moves)}"
    
    print("  SGF parsing: OK")
    
    # 测试策略目标生成
    policy = parser.get_policy_target(game, 19, 0)
    assert policy is not None, "Failed to get policy target"
    assert len(policy) == 361, f"Policy length: {len(policy)}"
    assert sum(policy) == 1.0, f"Policy sum: {sum(policy)}"
    
    print("  Policy target: OK")
    
    print("SGF parser tests passed!\n")


def test_game_loader():
    """测试棋谱加载器"""
    print("Testing game loader...")
    
    from src.data.game_loader import GameLoader, GoGameSimulator
    from src.data.sgf_parser import GameRecord, Move
    
    loader = GameLoader(19)
    
    # 创建测试棋谱
    game = GameRecord(board_size=19)
    game.moves = [
        Move(color='B', position=(3, 3)),  # 星位
        Move(color='W', position=(15, 15)),  # 星位
        Move(color='B', position=(3, 15)),  # 星位
        Move(color='W', position=(15, 3)),  # 星位
    ]
    game.result = "B+1.5"
    
    # 转换为训练数据
    training_data = loader.game_to_training_data(game)
    assert len(training_data) == 4, f"Training data length: {len(training_data)}"
    
    # 验证训练样本
    example = training_data[0]
    assert example.state.shape == (19, 19, 19), f"State shape: {example.state.shape}"
    assert example.policy.shape == (361,), f"Policy shape: {example.policy.shape}"
    assert example.value == 1.0, f"Value: {example.value}"
    
    print("  Game to training data: OK")
    
    # 测试游戏模拟器
    simulator = GoGameSimulator(19)
    simulator.reset()
    assert simulator.make_move((3, 3)), "Failed to make move"
    assert simulator.board[3, 3] == 1, "Move not recorded"
    
    print("  Game simulator: OK")
    
    print("Game loader tests passed!\n")


def test_trainer_pretrain():
    """测试训练器预训练"""
    print("Testing trainer pretrain...")
    
    from src.networks.alphanet import AlphaGoNet
    from src.training.trainer import AlphaGoTrainer
    from src.data.sgf_parser import GameRecord, Move
    
    # 创建模型
    model = AlphaGoNet(
        in_channels=19,
        backbone_channels=32,  # 小模型用于测试
        backbone_res_blocks=2,
        action_size=81
    )
    
    # 创建训练器
    trainer = AlphaGoTrainer(
        model=model,
        board_size=9,
        lr=1e-3,
        batch_size=16,
        buffer_size=100,
        device='cpu'
    )
    
    # 创建测试棋谱
    games = []
    for _ in range(5):
        game = GameRecord(board_size=9)
        game.moves = [
            Move(color='B', position=(i, i)) for i in range(9)
        ]
        game.result = "B+1.5"
        games.append(game)
    
    # 预训练
    result = trainer.pretrain_on_games(games, epochs=2, batch_size=16)
    assert 'pretrain_loss' in result, "Missing pretrain_loss"
    assert result['epochs'] == 2, f"Epochs: {result['epochs']}"
    
    print("  Pretrain: OK")
    
    print("Trainer pretrain tests passed!\n")


if __name__ == '__main__':
    print("=" * 50)
    print("Testing new features")
    print("=" * 50)
    print()
    
    test_sparse_attention()
    test_19x19_support()
    test_sgf_parser()
    test_game_loader()
    test_trainer_pretrain()
    
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
