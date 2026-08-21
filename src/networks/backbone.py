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


class SharedBackbone(nn.Module):
    """共享表示网络：将棋盘状态编码为隐藏状态"""
    
    def __init__(self, in_channels=19, channels=64, num_res_blocks=4):
        """
        初始化共享骨干网络
        
        Args:
            in_channels: 输入通道数 (19: 8步历史×2玩家 + 当前玩家标记 + 合法着法标记 + 棋盘大小标记)
            channels: 隐藏通道数
            num_res_blocks: 残差块数量
        """
        super(SharedBackbone, self).__init__()
        self.channels = channels
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 残差块
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        # 输出层
        self.conv_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 棋盘状态张量 (batch, 19, 9, 9)
            
        Returns:
            隐藏状态 (batch, channels, 9, 9)
        """
        # 第一层卷积
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 残差块
        out = self.res_blocks(out)
        
        # 输出层
        out = F.relu(self.bn_out(self.conv_out(out)))
        
        return out
