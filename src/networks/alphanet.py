import torch
import torch.nn as nn
from typing import Tuple, Optional

from .backbone import SharedBackbone
from .policy_network import PolicyNetwork
from .value_network import ValueNetwork


class AlphaGoNet(nn.Module):
    """
    监督学习用的策略-价值网络（AlphaGoZero 风格）。

    相比原版删除了 fast_policy 头：在 9x9 自我对弈里 fast_policy 占约 48% 前向算力
    却从未被 MCTS/rollout 使用，纯监督训练更不需要它。

    输入通道从 19 降到 12（原 19 通道里有 13 个恒为 0）。12 通道布局见
    src/data/dataset.py 的 build_state_tensor / GoBoardFeature 文档。
    """

    def __init__(self,
                 in_channels: int = 12,
                 backbone_channels: int = 128,
                 backbone_res_blocks: int = 12,
                 attention_mode: str = "mix",
                 num_attention_layers: int = 4,
                 num_heads: int = 4,
                 attention_dropout: float = 0.0,
                 policy_channels: int = 32,
                 value_channels: int = 16,
                 action_size: int = 361):
        """
        Args:
            attention_mode:      主干中注意力的使用方式
                                 "none"(纯卷积) | "mix"(卷积+注意力混合) | "all"(全注意力)
            num_attention_layers: mix 模式下注意力块数量
            num_heads:           多头注意力头数
            attention_dropout:   注意力 dropout
        """
        super(AlphaGoNet, self).__init__()
        self.action_size = action_size

        self.backbone = SharedBackbone(
            in_channels=in_channels,
            channels=backbone_channels,
            num_res_blocks=backbone_res_blocks,
            attention_mode=attention_mode,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )

        self.policy = PolicyNetwork(
            in_channels=backbone_channels,
            hidden_channels=policy_channels,
            action_size=action_size,
        )

        self.value = ValueNetwork(
            in_channels=backbone_channels,
            hidden_channels=value_channels,
        )

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observation: (batch, in_channels, H, W)
        Returns:
            policy: (batch, action_size) logits
            value:  (batch, 1) 黑方视角胜率（Tanh 输出 [-1,1]）
        """
        shared_state = self.backbone(observation)
        policy = self.policy(shared_state)
        value = self.value(shared_state)
        return policy, value

    def get_policy(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy(self.backbone(observation))

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(self.backbone(observation))
