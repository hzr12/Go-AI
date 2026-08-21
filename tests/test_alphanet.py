import pytest
import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.networks.backbone import SharedBackbone, ResBlock
from src.networks.policy_network import PolicyNetwork
from src.networks.value_network import ValueNetwork
from src.networks.fast_network import FastPolicyNetwork
from src.networks.alphanet import AlphaGoNet
from src.training.trainer import GoGame, ReplayBuffer, AlphaGoTrainer
from src.inference import GoAI


class TestSharedBackbone:
    """测试共享骨干网络"""
    
    def test_backbone_initialization(self):
        """测试骨干网络初始化"""
        backbone = SharedBackbone(in_channels=19, channels=64, num_res_blocks=4)
        assert backbone.channels == 64
    
    def test_backbone_forward(self):
        """测试骨干网络前向传播"""
        backbone = SharedBackbone(in_channels=19, channels=64, num_res_blocks=4)
        x = torch.randn(1, 19, 9, 9)
        out = backbone(x)
        assert out.shape == (1, 64, 9, 9)


class TestPolicyNetwork:
    """测试策略网络"""
    
    def test_policy_initialization(self):
        """测试策略网络初始化"""
        policy = PolicyNetwork(in_channels=64, hidden_channels=32, action_size=81)
        assert policy.action_size == 81
    
    def test_policy_forward(self):
        """测试策略网络前向传播"""
        policy = PolicyNetwork(in_channels=64, hidden_channels=32, action_size=81)
        x = torch.randn(1, 64, 9, 9)
        out = policy(x)
        assert out.shape == (1, 81)


class TestValueNetwork:
    """测试价值网络"""
    
    def test_value_initialization(self):
        """测试价值网络初始化"""
        value = ValueNetwork(in_channels=64, hidden_channels=16)
        assert value is not None
    
    def test_value_forward(self):
        """测试价值网络前向传播"""
        value = ValueNetwork(in_channels=64, hidden_channels=16)
        x = torch.randn(1, 64, 9, 9)
        out = value(x)
        assert out.shape == (1, 1)


class TestFastPolicyNetwork:
    """测试快速策略网络"""
    
    def test_fast_initialization(self):
        """测试快速策略网络初始化"""
        fast = FastPolicyNetwork(in_channels=19, channels=72, num_res_blocks=3, action_size=81)
        assert fast.action_size == 81
    
    def test_fast_forward(self):
        """测试快速策略网络前向传播"""
        fast = FastPolicyNetwork(in_channels=19, channels=72, num_res_blocks=3, action_size=81)
        x = torch.randn(1, 19, 9, 9)
        out = fast(x)
        assert out.shape == (1, 81)


class TestAlphaGoNet:
    """测试AlphaGoNet"""
    
    def test_alphanet_initialization(self):
        """测试AlphaGoNet初始化"""
        model = AlphaGoNet(
            in_channels=19,
            backbone_channels=64,
            backbone_res_blocks=4,
            policy_channels=32,
            value_channels=16,
            fast_channels=72,
            fast_res_blocks=3,
            action_size=81
        )
        assert model.action_size == 81
    
    def test_alphanet_forward(self):
        """测试AlphaGoNet前向传播"""
        model = AlphaGoNet(
            in_channels=19,
            backbone_channels=64,
            backbone_res_blocks=4,
            policy_channels=32,
            value_channels=16,
            fast_channels=72,
            fast_res_blocks=3,
            action_size=81
        )
        x = torch.randn(1, 19, 9, 9)
        policy, value, fast_policy = model(x)
        assert policy.shape == (1, 81)
        assert value.shape == (1, 1)
        assert fast_policy.shape == (1, 81)
    
    def test_alphanet_individual_outputs(self):
        """测试AlphaGoNet单独输出"""
        model = AlphaGoNet(
            in_channels=19,
            backbone_channels=64,
            backbone_res_blocks=4,
            policy_channels=32,
            value_channels=16,
            fast_channels=72,
            fast_res_blocks=3,
            action_size=81
        )
        x = torch.randn(1, 19, 9, 9)
        
        policy = model.get_policy(x)
        value = model.get_value(x)
        fast_policy = model.get_fast_policy(x)
        
        assert policy.shape == (1, 81)
        assert value.shape == (1, 1)
        assert fast_policy.shape == (1, 81)


class TestGoGame:
    """测试围棋游戏环境"""
    
    def test_game_initialization(self):
        """测试游戏初始化"""
        game = GoGame(board_size=9)
        assert game.board_size == 9
        assert game.board.shape == (9, 9)
        assert game.current_player == 1
        assert game.move_count == 0
    
    def test_get_legal_moves(self):
        """测试获取合法着法"""
        game = GoGame(board_size=9)
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) == 81
    
    def test_make_move(self):
        """测试执行着法"""
        game = GoGame(board_size=9)
        result = game.make_move(0)
        assert result == True
        assert game.board[0, 0] == 1
        assert game.current_player == -1
        assert game.move_count == 1
    
    def test_game_over(self):
        """测试游戏结束"""
        game = GoGame(board_size=9)
        assert game.is_game_over() == False
    
    def test_state_tensor(self):
        """测试状态张量"""
        game = GoGame(board_size=9)
        tensor = game.get_state_tensor()
        assert tensor.shape == (19, 9, 9)


class TestReplayBuffer:
    """测试经验回放缓冲区"""
    
    def test_buffer_initialization(self):
        """测试缓冲区初始化"""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0
    
    def test_push_and_sample(self):
        """测试存储和采样"""
        buffer = ReplayBuffer(capacity=100)
        state = np.random.randn(19, 9, 9)
        action = 0
        reward = 1.0
        next_state = np.random.randn(19, 9, 9)
        done = False
        policy = np.random.randn(81)
        
        buffer.push(state, action, reward, next_state, done, policy)
        assert len(buffer) == 1
        
        states, actions, rewards, next_states, dones, policies = buffer.sample(1)
        assert states.shape == (1, 19, 9, 9)
        assert actions.shape == (1,)
        assert rewards.shape == (1,)


class TestAlphaGoTrainer:
    """测试训练器"""
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        model = AlphaGoNet(
            in_channels=19,
            backbone_channels=64,
            backbone_res_blocks=4,
            policy_channels=32,
            value_channels=16,
            fast_channels=72,
            fast_res_blocks=3,
            action_size=81
        )
        trainer = AlphaGoTrainer(model=model, board_size=9)
        assert trainer.board_size == 9
    
    def test_board_to_tensor(self):
        """测试棋盘转张量"""
        model = AlphaGoNet(
            in_channels=19,
            backbone_channels=64,
            backbone_res_blocks=4,
            policy_channels=32,
            value_channels=16,
            fast_channels=72,
            fast_res_blocks=3,
            action_size=81
        )
        trainer = AlphaGoTrainer(model=model, board_size=9, device='cpu')
        board = np.zeros((9, 9), dtype=np.int8)
        tensor = trainer.board_to_tensor(board, 1)
        assert tensor.shape == (1, 19, 9, 9)


class TestGoAI:
    """测试围棋AI"""
    
    def test_ai_initialization(self):
        """测试AI初始化"""
        ai = GoAI(board_size=9, device='cpu')
        assert ai.board_size == 9
    
    def test_get_move(self):
        """测试获取着法"""
        ai = GoAI(board_size=9, device='cpu')
        move, info = ai.get_move()
        assert 0 <= move < 81
        assert 'policy' in info
        assert 'value' in info
    
    def test_make_move(self):
        """测试执行着法"""
        ai = GoAI(board_size=9, device='cpu')
        result = ai.make_move(0)
        assert result == True
        assert ai.board[0, 0] == 1
    
    def test_evaluate_position(self):
        """测试评估局面"""
        ai = GoAI(board_size=9, device='cpu')
        result = ai.evaluate_position()
        assert 'best_move' in result
        assert 'best_prob' in result
        assert 'value' in result
    
    def test_suggest_moves(self):
        """测试推荐着法"""
        ai = GoAI(board_size=9, device='cpu')
        suggestions = ai.suggest_moves(num_moves=5)
        assert len(suggestions) <= 5
        assert all('move' in s for s in suggestions)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
