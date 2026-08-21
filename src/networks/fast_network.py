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
        super(FastPolicyNetwork, self).__init__()
        self.action_size = action_size
        
        # 输入层
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        
        # 残差块
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        # 策略头（使用AdaptiveAvgPool2d支持任意棋盘大小）
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化到1×1
            nn.Flatten(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x)))
        out = self.res_blocks(out)
        policy = self.policy_head(out)
        return policy
