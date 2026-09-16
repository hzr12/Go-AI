import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization。"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _ValueResBlock(nn.Module):
    """Value head 专用残差块（3×3 卷积，BN + ReLU，zero-init）。"""

    def __init__(self, channels):
        super(_ValueResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ConvNeXtValueBlock(nn.Module):
    """ConvNeXt 风格价值块：5x5 深度卷积 + LayerNorm + PWConv。"""

    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 5, padding=2,
                                groups=channels, bias=False)
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.pwconv2 = nn.Conv2d(channels * 4, channels, 1, bias=False)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        return x + residual


class ValueNetwork(nn.Module):
    """Value head：下采样 → 残差块提取全局特征 → GAP → 标量。

    AlphaZero 风格：下采样让感受野快速覆盖全局棋盘，
    残差块提取厚势/死子等全局特征，最后 GAP + Linear 输出 value。
    """

    def __init__(self, in_channels=64, hidden_channels=32, arch="resnet"):
        super(ValueNetwork, self).__init__()
        self.arch = arch

        # 下采样：19×19 → 10×10（stride=2, padding=1, kernel=3）
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # 残差块
        if arch == "convnext":
            self.res1 = ConvNeXtValueBlock(hidden_channels)
            self.res2 = ConvNeXtValueBlock(hidden_channels)
            self.res3 = ConvNeXtValueBlock(hidden_channels)
            self.norm_out = LayerNorm2d(hidden_channels)
        else:
            self.res1 = _ValueResBlock(hidden_channels)
            self.res2 = _ValueResBlock(hidden_channels)
            self.res3 = _ValueResBlock(hidden_channels)

        # 全局池化 + 输出
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        if self.arch == "convnext":
            x = self.norm_out(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return x
