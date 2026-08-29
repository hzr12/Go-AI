import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """纯卷积残差块（保持原有结构，用于浅层局部特征提取）。"""

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


class MultiHeadSelfAttention(nn.Module):
    """把棋盘 (B, C, H, W) 视为 N=H*W 个 token 的多头自注意力。

    采用 pre-LayerNorm 的 Transformer 风格：
        y = x + Attn(LN(x))
        z = y + FFN(LN(y))
    """

    def __init__(self, channels, num_heads=4, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        assert channels % num_heads == 0, "channels 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.ln1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        # 转成序列: (B, N, C)
        seq = x.flatten(2).transpose(1, 2)

        # 自注意力
        residual = seq
        h = self.ln1(seq)
        qkv = self.qkv(h)  # (B, N, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, Hh, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, Hh, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, Hh, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        seq = residual + out

        # 前馈
        residual = seq
        seq = residual + self.ffn_drop(self.ffn(self.ln2(seq)))

        # 还原为 (B, C, H, W)
        return seq.transpose(1, 2).view(B, C, H, W)


class AttentionResBlock(nn.Module):
    """卷积残差 + 多头自注意力 的混合块。

    顺序：卷积残差 -> 自注意力（均带残差）。注意力负责捕捉棋盘上的
    长程依赖（如整条大龙的死活、全局厚薄），卷积负责局部形状。
    """

    def __init__(self, channels, num_heads=4, dropout=0.0):
        super(AttentionResBlock, self).__init__()
        self.conv = ResBlock(channels)
        self.attn = MultiHeadSelfAttention(channels, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return x


class SharedBackbone(nn.Module):
    """共享表示网络：将棋盘状态编码为隐藏状态。

    支持三种堆叠模式（由 attention_mode 控制）：
        - "none" : 全部用纯卷积 ResBlock（最快，局部性好）
        - "mix"  : 在 num_res_blocks 个块中穿插 num_attention_layers 个
                   AttentionResBlock（推荐：卷积打底 + 注意力提质）
        - "all"  : 全部使用 AttentionResBlock
    """

    def __init__(self, in_channels=12, channels=128, num_res_blocks=12,
                 attention_mode="mix", num_attention_layers=4,
                 num_heads=4, attention_dropout=0.0):
        """
        Args:
            in_channels: 输入通道数（12）
            channels: 隐藏通道数
            num_res_blocks: 总块数量（卷积 + 注意力混合块之和）
            attention_mode: "none" | "mix" | "all"
            num_attention_layers: mix 模式下注意力块的数量（<= num_res_blocks）
            num_heads: 注意力头数
            attention_dropout: 注意力 dropout
        """
        super(SharedBackbone, self).__init__()
        self.channels = channels
        self.attention_mode = attention_mode

        # 第一层卷积：把 12 通道投影到主干通道
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # 构建块序列
        blocks = self._build_blocks(
            num_res_blocks, attention_mode, num_attention_layers, channels, num_heads, attention_dropout
        )
        self.blocks = nn.Sequential(*blocks)

        # 输出层
        self.conv_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)

    @staticmethod
    def _build_blocks(num_res_blocks, mode, num_attn, channels, num_heads, dropout):
        if mode == "none" or num_attn <= 0:
            return [ResBlock(channels) for _ in range(num_res_blocks)]
        if mode == "all":
            return [AttentionResBlock(channels, num_heads, dropout)
                    for _ in range(num_res_blocks)]
        # mix：均匀地把 num_attn 个注意力块插入到卷积块之间
        num_attn = min(num_attn, num_res_blocks)
        attn_idx = set(
            int(round(i * (num_res_blocks - 1) / max(num_attn - 1, 1)))
            for i in range(num_attn)
        )
        blocks = []
        for i in range(num_res_blocks):
            if i in attn_idx:
                blocks.append(AttentionResBlock(channels, num_heads, dropout))
            else:
                blocks.append(ResBlock(channels))
        return blocks

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = F.relu(self.bn_out(self.conv_out(out)))
        return out
