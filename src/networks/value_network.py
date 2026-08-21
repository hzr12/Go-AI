import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """价值网络：评估局面胜率"""
    
    def __init__(self, in_channels=64, hidden_channels=16):
        """
        初始化价值网络
        
        Args:
            in_channels: 输入通道数 (来自backbone)
            hidden_channels: 隐藏通道数
        """
        super(ValueNetwork, self).__init__()
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_channels * 81, 1),
            nn.Tanh()  # 价值范围[-1, 1]
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 隐藏状态 (batch, in_channels, 9, 9)
            
        Returns:
            value: 局面评估值 (batch, 1) 范围[-1, 1]
        """
        value = self.value_head(x)
        return value
