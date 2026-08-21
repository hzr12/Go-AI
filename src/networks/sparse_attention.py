import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class LocalWindowAttention(nn.Module):
    """局部窗口注意力：使用深度卷积捕获局部关系"""
    
    def __init__(self, channels: int, window_size: int = 3):
        super(LocalWindowAttention, self).__init__()
        self.channels = channels
        self.window_size = window_size
        
        # 深度可分离卷积捕获局部关系
        self.local_conv = nn.Conv2d(channels, channels, window_size, 
                                     padding=window_size//2, groups=channels, bias=False)
        self.norm = nn.LayerNorm(channels)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, H, W = x.shape
        residual = x
        
        # 归一化
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        # 局部特征聚合
        local_features = self.local_conv(x_norm)
        
        # 门控
        gate = self.gate(x_norm)
        gated_features = local_features * gate
        
        # 输出投影
        out = self.proj(gated_features)
        
        return residual + out


class GlobalTokenAttention(nn.Module):
    """全局特殊位置注意力：角、边、中心关注全局"""
    
    def __init__(self, channels: int, num_heads: int = 4, num_global_tokens: int = 9):
        super(GlobalTokenAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.num_global_tokens = num_global_tokens
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # 全局token
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, channels))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)
        
        # 图像QKV
        self.q_img = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_img = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_img = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_img = nn.Conv2d(channels, channels, 1, bias=False)
        
        # 全局token QKV
        self.qkv_global = nn.Linear(channels, channels * 3)
        self.out_global = nn.Linear(channels, channels)
        
        self.norm = nn.LayerNorm(channels)
        self.norm_global = nn.LayerNorm(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, H, W = x.shape
        residual = x
        
        # 归一化
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        # 图像QKV: (batch, channels, H, W)
        q_img = self.q_img(x_norm)
        k_img = self.k_img(x_norm)
        v_img = self.v_img(x_norm)
        
        # 展平空间维度: (batch, channels, H*W) -> (batch, H*W, channels)
        q_flat = q_img.reshape(batch, channels, H * W).permute(0, 2, 1)
        k_flat = k_img.reshape(batch, channels, H * W).permute(0, 2, 1)
        v_flat = v_img.reshape(batch, channels, H * W).permute(0, 2, 1)
        
        # 重塑为多头: (batch, H*W, channels) -> (batch, num_heads, H*W, head_dim)
        q_flat = q_flat.reshape(batch, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_flat = k_flat.reshape(batch, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_flat = v_flat.reshape(batch, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 全局token
        global_tokens = self.global_tokens.expand(batch, -1, -1)
        global_tokens = self.norm_global(global_tokens)
        
        # 全局token QKV
        qkv_g = self.qkv_global(global_tokens)
        qkv_g = qkv_g.reshape(batch, self.num_global_tokens, 3, self.num_heads, self.head_dim)
        qkv_g = qkv_g.permute(2, 0, 3, 1, 4)
        q_g, k_g, v_g = qkv_g[0], qkv_g[1], qkv_g[2]
        
        # 全局token关注所有位置
        attn_g = torch.matmul(q_g, k_flat.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_g = F.softmax(attn_g, dim=-1)  # (batch, num_heads, num_global_tokens, H*W)
        out_g = torch.matmul(attn_g, v_flat)  # (batch, num_heads, num_global_tokens, head_dim)
        
        # 所有位置关注全局token
        attn_x = torch.matmul(q_flat, k_g.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_x = F.softmax(attn_x, dim=-1)  # (batch, num_heads, H*W, num_global_tokens)
        out_x = torch.matmul(attn_x, v_g)  # (batch, num_heads, H*W, head_dim)
        
        # 合并: (batch, num_heads, H*W, head_dim) -> (batch, H*W, channels) -> (batch, channels, H, W)
        out_x = out_x.permute(0, 2, 1, 3).reshape(batch, H * W, channels)
        out_x = out_x.permute(0, 2, 1).reshape(batch, channels, H, W)
        out_x = self.out_img(out_x)
        
        return residual + out_x


class HybridSparseAttention(nn.Module):
    """混合稀疏注意力：融合局部和全局"""
    
    def __init__(self, channels: int, window_size: int = 3, num_heads: int = 4, 
                 num_global_tokens: int = 9):
        super(HybridSparseAttention, self).__init__()
        
        self.local_attention = LocalWindowAttention(channels=channels, window_size=window_size)
        self.global_attention = GlobalTokenAttention(
            channels=channels, num_heads=num_heads, num_global_tokens=num_global_tokens
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = self.local_attention(x)
        global_out = self.global_attention(x)
        combined = torch.cat([local_out, global_out], dim=1)
        return self.fusion(combined)


class SparseAttentionBlock(nn.Module):
    """稀疏注意力块"""
    
    def __init__(self, channels: int, window_size: int = 3, num_heads: int = 4,
                 num_global_tokens: int = 9):
        super(SparseAttentionBlock, self).__init__()
        
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        
        self.sparse_attention = HybridSparseAttention(
            channels=channels, window_size=window_size,
            num_heads=num_heads, num_global_tokens=num_global_tokens
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = self.sparse_attention(out)
        return out
