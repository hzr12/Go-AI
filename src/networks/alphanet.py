import torch
import torch.nn as nn
from typing import Tuple, Optional
from .backbone import SharedBackbone
from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from .fast_network import FastPolicyNetwork


class AlphaGoNet(nn.Module):
    """AlphaGo风格的多网络组合"""
    
    def __init__(self, 
                 in_channels=19,
                 backbone_channels=64,
                 backbone_res_blocks=4,
                 policy_channels=32,
                 value_channels=16,
                 fast_channels=72,
                 fast_res_blocks=3,
                 action_size=81):
        """
        初始化AlphaGoNet
        
        Args:
            in_channels: 输入通道数
            backbone_channels: 骨干网络通道数
            backbone_res_blocks: 骨干网络残差块数量
            policy_channels: 策略网络通道数
            value_channels: 价值网络通道数
            fast_channels: 快速策略网络通道数
            fast_res_blocks: 快速策略网络残差块数量
            action_size: 动作空间大小
        """
        super(AlphaGoNet, self).__init__()
        self.action_size = action_size
        
        # 共享骨干网络
        self.backbone = SharedBackbone(
            in_channels=in_channels,
            channels=backbone_channels,
            num_res_blocks=backbone_res_blocks
        )
        
        # 策略网络
        self.policy = PolicyNetwork(
            in_channels=backbone_channels,
            hidden_channels=policy_channels,
            action_size=action_size
        )
        
        # 价值网络
        self.value = ValueNetwork(
            in_channels=backbone_channels,
            hidden_channels=value_channels
        )
        
        # 快速策略网络 (独立，不共享骨干)
        self.fast_policy = FastPolicyNetwork(
            in_channels=in_channels,
            channels=fast_channels,
            num_res_blocks=fast_res_blocks,
            action_size=action_size
        )
    
    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            observation: 棋盘状态 (batch, in_channels, 9, 9)
            
        Returns:
            policy: 策略网络输出 (batch, action_size)
            value: 价值网络输出 (batch, 1)
            fast_policy: 快速策略网络输出 (batch, action_size)
        """
        # 共享骨干网络
        shared_state = self.backbone(observation)
        
        # 策略网络
        policy = self.policy(shared_state)
        
        # 价值网络
        value = self.value(shared_state)
        
        # 快速策略网络 (独立推理)
        fast_policy = self.fast_policy(observation)
        
        return policy, value, fast_policy
    
    def get_policy(self, observation: torch.Tensor) -> torch.Tensor:
        """获取策略网络输出"""
        shared_state = self.backbone(observation)
        return self.policy(shared_state)
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """获取价值网络输出"""
        shared_state = self.backbone(observation)
        return self.value(shared_state)
    
    def get_fast_policy(self, observation: torch.Tensor) -> torch.Tensor:
        """获取快速策略网络输出"""
        return self.fast_policy(observation)
    
    def get_all_outputs(self, observation: torch.Tensor) -> dict:
        """获取所有网络输出"""
        policy, value, fast_policy = self.forward(observation)
        return {
            'policy': policy,
            'value': value,
            'fast_policy': fast_policy
        }
