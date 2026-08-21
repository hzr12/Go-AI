import torch
import torch.nn as nn
import torch.nn.functional as F
from .sparse_attention import HybridSparseAttention, SparseAttentionBlock


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
    
    def __init__(self, in_channels=19, channels=64, num_res_blocks=4,
                 use_sparse_attention=False, attention_window_size=3,
                 attention_num_heads=4, attention_num_global_tokens=9):
        """
        初始化共享骨干网络
        
        Args:
            in_channels: 输入通道数
            channels: 隐藏通道数
            num_res_blocks: 残差块数量
            use_sparse_attention: 是否使用稀疏注意力
            attention_window_size: 局部窗口大小
            attention_num_heads: 注意力头数
            attention_num_global_tokens: 全局token数量
        """
        super(SharedBackbone, self).__init__()
        self.channels = channels
        self.use_sparse_attention = use_sparse_attention
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 残差块
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        # 稀疏注意力（可选）
        if use_sparse_attention:
            self.sparse_attention = HybridSparseAttention(
                channels=channels,
                window_size=attention_window_size,
                num_heads=attention_num_heads,
                num_global_tokens=attention_num_global_tokens
            )
        else:
            self.sparse_attention = nn.Identity()
        
        # 输出层
        self.conv_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 棋盘状态张量 (batch, in_channels, H, W)
            
        Returns:
            隐藏状态 (batch, channels, H, W)
        """
        # 第一层卷积
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 残差块
        out = self.res_blocks(out)
        
        # 稀疏注意力
        out = self.sparse_attention(out)
        
        # 输出层
        out = F.relu(self.bn_out(self.conv_out(out)))
        
        return out
