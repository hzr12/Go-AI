import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class FastPolicyNetwork(nn.Module):
    """快速策略网络：用于快速推理和搜索"""
    
    def __init__(self, in_channels=19, channels=72, num_res_blocks=3, action_size=81):
        """
        初始化快速策略网络
        
        Args:
            in_channels: 输入通道数 (19: 原始棋盘状态)
            channels: 隐藏通道数
            num_res_blocks: 残差块数量
            action_size: 动作空间大小 (9x9=81)
        """
        super(FastPolicyNetwork, self).__init__()
        self.action_size = action_size
        
        # 输入层
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        
        # 残差块
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 81, action_size)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 棋盘状态 (batch, in_channels, 9, 9)
            
        Returns:
            policy: 着法概率分布 (batch, action_size)
        """
        # 输入层
        out = F.relu(self.bn_in(self.conv_in(x)))
        
        # 残差块
        out = self.res_blocks(out)
        
        # 策略头
        policy = self.policy_head(out)
        
        return policy
