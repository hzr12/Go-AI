import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np

from src.networks.backbone import SharedBackbone, ResBlock
from src.networks.policy_network import PolicyNetwork
from src.networks.value_network import ValueNetwork
from src.networks.alphanet import AlphaGoNet
from src.game.go_rules import GoBoard
from src.inference import GoAI


class TestSharedBackbone:
    def test_backbone_initialization(self):
        backbone = SharedBackbone(in_channels=12, channels=128, num_res_blocks=12)
        assert backbone.channels == 128

    def test_backbone_forward(self):
        backbone = SharedBackbone(in_channels=12, channels=128, num_res_blocks=12)
        x = torch.randn(1, 12, 9, 9)
        out = backbone(x)
        assert out.shape == (1, 128, 9, 9)

    @pytest.mark.parametrize("mode", ["none", "mix", "all"])
    def test_backbone_attention_modes(self, mode):
        backbone = SharedBackbone(in_channels=12, channels=64, num_res_blocks=6,
                                  attention_mode=mode, num_attention_layers=3, num_heads=4)
        x = torch.randn(2, 12, 9, 9)
        out = backbone(x)
        assert out.shape == (2, 64, 9, 9)


class TestHeads:
    def test_policy_forward(self):
        policy = PolicyNetwork(in_channels=128, hidden_channels=64, action_size=82)
        x = torch.randn(1, 128, 9, 9)
        out = policy(x)
        assert out.shape == (1, 82)

    def test_value_forward(self):
        value = ValueNetwork(in_channels=128, hidden_channels=32)
        x = torch.randn(1, 128, 9, 9)
        out = value(x)
        assert out.shape == (1, 1)


class TestAlphaGoNet:
    """新架构：输入 12 通道，输出 (policy, value)。"""

    def _make(self, bs=9, attention_mode="mix"):
        return AlphaGoNet(
            in_channels=12,
            backbone_channels=128,
            backbone_res_blocks=12,
            attention_mode=attention_mode,
            num_attention_layers=4,
            num_heads=4,
            policy_channels=64,
            value_channels=32,
            action_size=bs * bs + 1,
        )

    def test_alphanet_initialization(self):
        model = self._make(9)
        assert model.action_size == 82

    def test_alphanet_forward(self):
        model = self._make(9)
        x = torch.randn(1, 12, 9, 9)
        policy, value = model(x)
        assert policy.shape == (1, 82)
        assert value.shape == (1, 1)

    def test_alphanet_value_range(self):
        model = self._make(9)
        x = torch.randn(1, 12, 9, 9)
        _, value = model(x)
        # Tanh -> [-1, 1]
        v = value.detach()
        assert float(v.min()) >= -1.0
        assert float(v.max()) <= 1.0


class TestGoAISmoke:
    """推理引擎冒烟测试（CPU，随机权重，仅验证流程不报错）。"""

    def test_ai_init_no_model(self):
        ai = GoAI(board_size=9, device="cpu")
        assert ai.board_size == 9
        assert ai.model is not None

    def test_self_play_runs(self):
        ai = GoAI(board_size=9, device="cpu")
        results = ai.self_play(num_games=1, max_moves=60, temperature=1.0, verbose=False)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0] in (1.0, -1.0)

    def test_predict_returns_policy_value(self):
        ai = GoAI(board_size=9, device="cpu")
        board = GoBoard(9)
        policy, value = ai.predict(board, [-1, -1, -1], [-1, -1, -1], 1)
        assert policy.shape == (9 * 9 + 1,)
        assert abs(policy.sum() - 1.0) < 1e-4
        assert -1.0 <= value <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
