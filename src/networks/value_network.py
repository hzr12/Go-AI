import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ValueNetwork(nn.Module):
    """Value head：下采样 → 残差块提取全局特征 → GAP → 标量。

    AlphaZero 风格：下采样让感受野快速覆盖全局棋盘，
    残差块提取厚势/死子等全局特征，最后 GAP + Linear 输出 value。
    """

    def __init__(self, in_channels=64, hidden_channels=32):
        super(ValueNetwork, self).__init__()

        # 下采样：19×19 → 10×10（stride=2, padding=1, kernel=3）
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        # 残差块：在缩小后的特征图上提取全局特征
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
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return x
