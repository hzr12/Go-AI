import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """策略网络：预测最佳着法"""
    
    def __init__(self, in_channels=64, hidden_channels=32, action_size=81):
        """
        初始化策略网络
        
        Args:
            in_channels: 输入通道数 (来自backbone)
            hidden_channels: 隐藏通道数
            action_size: 动作空间大小 (9x9=81)
        """
        super(PolicyNetwork, self).__init__()
        self.action_size = action_size
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_channels * 81, action_size)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 隐藏状态 (batch, in_channels, 9, 9)
            
        Returns:
            policy: 着法概率分布 (batch, action_size)
        """
        policy = self.policy_head(x)
        return policy
