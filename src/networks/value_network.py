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

    def __init__(self, in_channels=64, hidden_channels=32, num_res_blocks=3, arch="resnet"):
        super(ValueNetwork, self).__init__()
        self.arch = arch
        self.num_res_blocks = num_res_blocks

        # 下采样：19×19 → 10×10（stride=2, padding=1, kernel=3）
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # 残差块
        if arch == "convnext":
            self.res_blocks = nn.ModuleList([
                ConvNeXtValueBlock(hidden_channels) for _ in range(num_res_blocks)
            ])
            self.norm_out = LayerNorm2d(hidden_channels)
        else:
            self.res_blocks = nn.ModuleList([
                _ValueResBlock(hidden_channels) for _ in range(num_res_blocks)
            ])

        # 全局池化 + 输出
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.downsample(x)
        for block in self.res_blocks:
            x = block(x)
        if self.arch == "convnext":
            x = self.norm_out(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return x


class FCValueHead(nn.Module):
    """v21 的全连接价值头（P4.1），逐项 **142,017** 参数。

    结构（权威表 §2 的 value 行）：
        1×1 Conv 184→32 无bias + BN    5,952
        GAP                            —（无参数）
        FC   32→128 **带**bias         4,224
        FC  128→512 **带**bias        66,048
        FC  512→128 **带**bias        65,664
        FC  128→  1 **带**bias           129
        合计                          142,017

    末尾是 `nn.Tanh()`（值域 [-1,1]），与 P4.5 的 `value_t ∈ [-1,1]` 回归口径
    以及推理/Brier 侧「按 tanh/[-1,1] 解释 value」的既有约定一致。
    ⚠ 这是**有意偏离**既有 `ValueNetwork`：后者的 `forward`（:92-100）是裸线性
    输出、无 Tanh，head 与消费端差了一个 tanh。Tanh 零参数，预算表区分不出来。
    **用户已裁决：Tanh 保留**（见 report §5 的裁决记录）—— 不是待定项，P4.2
    之后的任务不必再为此往返提问。
    ✅ 下游复核：`src/` 全量 grep 没有任何 `sigmoid(value)` 调用点
    （`inference.py:6` 已按「tanh 后落在 [-1,1]」解释 value），二次压缩的风险不存在。
    """

    def __init__(self, in_channels=184, hidden_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)  # 5,888
        self.bn = nn.BatchNorm2d(hidden_channels)                            # 64
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(hidden_channels, 128)      # 4,224
        self.fc2 = nn.Linear(128, 512)                  # 66,048
        self.fc3 = nn.Linear(512, 128)                  # 65,664
        self.fc4 = nn.Linear(128, 1)                    # 129
        self.out_tanh = nn.Tanh()

    def forward(self, x):
        # x: (B, in_channels, H, W) -> (B, 1)，值域 (-1, 1)
        x = F.relu(self.bn(self.conv(x)))
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out_tanh(self.fc4(x))
