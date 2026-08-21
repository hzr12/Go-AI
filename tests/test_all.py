import pytest
import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.networks.resnet import MuZeroNet, ResBlock, RepresentationNetwork, DynamicsNetwork, PredictionNetwork
from src.search.minimax import MinimaxSearch
from src.training.trainer import GoGame, PrioritizedReplayBuffer as ReplayBuffer, Trainer
from src.evaluation.evaluator import Evaluator
from src.inference import GoAI
from src.config.config import Config, get_config


class TestResBlock:
    """测试残差块"""
    
    def test_residual_block(self):
        """测试残差块前向传播"""
        block = ResBlock(64)
        x = torch.randn(1, 64, 9, 9)
        out = block(x)
        assert out.shape == x.shape
    
    def test_residual_connection(self):
        """测试残差连接"""
        block = ResBlock(64)
        x = torch.randn(1, 64, 9, 9)
        out = block(x)
        # 输出应该与输入不同但形状相同
        assert out.shape == x.shape
        assert not torch.allclose(out, x)


class TestRepresentationNetwork:
    """测试表示网络"""
    
    def test_representation_network(self):
        """测试表示网络前向传播"""
        net = RepresentationNetwork(in_channels=19, channels=64, num_res_blocks=4)
        x = torch.randn(1, 19, 9, 9)
        out = net(x)
        assert out.shape == (1, 64, 9, 9)


class TestDynamicsNetwork:
    """测试动态网络"""
    
    def test_dynamics_network(self):
        """测试动态网络前向传播"""
        net = DynamicsNetwork(channels=64, num_res_blocks=4)
        state = torch.randn(1, 64, 9, 9)
        action = torch.randn(1, 1, 9, 9)
        next_state, reward = net(state, action)
        assert next_state.shape == (1, 64, 9, 9)
        assert reward.shape == (1, 1)


class TestPredictionNetwork:
    """测试预测网络"""
    
    def test_prediction_network(self):
        """测试预测网络前向传播"""
        net = PredictionNetwork(channels=64, action_size=81)
        x = torch.randn(1, 64, 9, 9)
        policy, value = net(x)
        assert policy.shape == (1, 81)
        assert value.shape == (1, 1)


class TestMuZeroNet:
    """测试MuZero网络"""
    
    def test_initial_inference(self):
        """测试初始推理"""
        net = MuZeroNet(in_channels=19, channels=64, num_res_blocks=4, action_size=81)
        observation = torch.randn(1, 19, 9, 9)
        state, policy, value = net.initial_inference(observation)
        assert state.shape == (1, 64, 9, 9)
        assert policy.shape == (1, 81)
        assert value.shape == (1, 1)
    
    def test_recurrent_inference(self):
        """测试递归推理"""
        net = MuZeroNet(in_channels=19, channels=64, num_res_blocks=4, action_size=81)
        state = torch.randn(1, 64, 9, 9)
        action = torch.randint(0, 81, (1,))
        next_state, reward, policy, value = net.recurrent_inference(state, action)
        assert next_state.shape == (1, 64, 9, 9)
        assert reward.shape == (1, 1)
        assert policy.shape == (1, 81)
        assert value.shape == (1, 1)


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
        assert len(legal_moves) == 81  # 9x9棋盘
    
    def test_make_move(self):
        """测试执行着法"""
        game = GoGame(board_size=9)
        result = game.make_move(0)  # 下在位置0
        assert result == True
        assert game.board[0, 0] == 1  # 黑棋
        assert game.current_player == -1  # 切换到白棋
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


class TestMinimaxSearch:
    """测试Minimax搜索"""
    
    def test_search_initialization(self):
        """测试搜索初始化"""
        model = MuZeroNet(in_channels=19, channels=64, num_res_blocks=4, action_size=81)
        search = MinimaxSearch(model=model, board_size=9, search_depth=5, top_k=3)
        assert search.board_size == 9
        assert search.search_depth == 5
        assert search.top_k == 3
    
    def test_get_legal_moves(self):
        """测试获取合法着法"""
        model = MuZeroNet(in_channels=19, channels=64, num_res_blocks=4, action_size=81)
        search = MinimaxSearch(model=model, board_size=9, search_depth=5, top_k=3)
        board = np.zeros((9, 9), dtype=np.int8)
        legal_moves = search.get_legal_moves(board)
        assert len(legal_moves) == 81
    
    def test_get_candidate_moves(self):
        """测试获取候选着法"""
        model = MuZeroNet(in_channels=19, channels=64, num_res_blocks=4, action_size=81)
        search = MinimaxSearch(model=model, board_size=9, search_depth=5, top_k=3)
        board = np.zeros((9, 9), dtype=np.int8)
        candidates = search.get_candidate_moves(board, 1)
        assert len(candidates) <= 3
    
    def test_search(self):
        """测试搜索"""
        model = MuZeroNet(in_channels=19, channels=64, num_res_blocks=4, action_size=81)
        search = MinimaxSearch(model=model, board_size=9, search_depth=3, top_k=3)
        board = np.zeros((9, 9), dtype=np.int8)
        move, score = search.search(board, 1)
        assert 0 <= move < 81
        assert isinstance(score, float)


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
        
        states, actions, rewards, next_states, dones, policies, indices, weights = buffer.sample(1)
        assert states.shape == (1, 19, 9, 9)
        assert actions.shape == (1,)
        assert rewards.shape == (1,)


class TestEvaluator:
    """测试评估器"""
    
    def test_evaluator_initialization(self):
        """测试评估器初始化"""
        model = MuZeroNet(in_channels=19, channels=64, num_res_blocks=4, action_size=81)
        evaluator = Evaluator(model=model, board_size=9, search_depth=5, top_k=3)
        assert evaluator.board_size == 9
    
    def test_evaluate_position(self):
        """测试评估局面"""
        model = MuZeroNet(in_channels=19, channels=64, num_res_blocks=4, action_size=81)
        evaluator = Evaluator(model=model, board_size=9, search_depth=3, top_k=3)
        board = np.zeros((9, 9), dtype=np.int8)
        result = evaluator.evaluate_position(board, 1)
        assert 'best_move' in result
        assert 'score' in result


class TestGoAI:
    """测试围棋AI"""
    
    def test_ai_initialization(self):
        """测试AI初始化"""
        ai = GoAI(board_size=9)
        assert ai.board_size == 9
    
    def test_get_move(self):
        """测试获取着法"""
        ai = GoAI(board_size=9)
        move, info = ai.get_move()
        assert 0 <= move < 81
        assert 'policy' in info
    
    def test_make_move(self):
        """测试执行着法"""
        ai = GoAI(board_size=9)
        result = ai.make_move(0)
        assert result == True
        assert ai.board[0, 0] == 1
    
    def test_evaluate_position(self):
        """测试评估局面"""
        ai = GoAI(board_size=9)
        result = ai.evaluate_position()
        assert 'best_move' in result
        assert 'best_prob' in result
    
    def test_suggest_moves(self):
        """测试推荐着法"""
        ai = GoAI(board_size=9)
        suggestions = ai.suggest_moves(num_moves=5)
        assert len(suggestions) <= 5
        assert all('move' in s for s in suggestions)


class TestConfig:
    """测试配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = Config()
        assert config.board_size == 9
        assert config.channels == 64
        assert config.num_res_blocks == 4
    
    def test_get_config(self):
        """测试获取配置"""
        config = get_config('default')
        assert config.board_size == 9
        
        fast_config = get_config('fast')
        assert fast_config.search_depth == 5
    
    def test_config_paths(self):
        """测试配置路径"""
        config = Config()
        model_path = config.get_model_path('test')
        assert 'test.pth' in model_path


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
