import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=32, action_size=81, num_layers=2):
        super(PolicyNetwork, self).__init__()
        self.action_size = action_size
        self.num_layers = num_layers

        if num_layers == 2:
            # 原始结构：1x1 -> 1x1
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.conv2 = nn.Conv2d(hidden_channels, 1, 1, bias=False)
            self.pass_bias = nn.Parameter(torch.zeros(1))
        else:
            # 3层结构：1x1 -> 3x3 -> 1x1
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channels)
            self.conv3 = nn.Conv2d(hidden_channels, 1, 1, bias=False)
            self.pass_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        if self.num_layers >= 3:
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.conv3(x)
        else:
            x = self.conv2(x)
        x = x.view(B, -1)
        return torch.cat([x, self.pass_bias.expand(B, 1)], dim=1)


class FCPolicyHead(nn.Module):
    """v21 代遗留的全连接策略头（P4.1），逐项 **2,366,730** 参数。

     v21 架构已随其顶层构建函数一并删除，本类**没有现役架构调用方**；
    保留是因为它仍是「生产头」规格的参照（tests/test_grad_checkpointing.py、
    tests/test_huber_loss.py 直接拿它当被测对象）。输入通道与 in_channels=
    184 的 v21 中间层配套，默认值沿用当时的规格，不随现役 12 通道输入改。

    结构（权威表 §2 的 policy 行）：
        1×1 Conv 184→96 无bias + BN   17,856
        1×1 Conv  96→48 无bias + BN    4,704
        Flatten 48×19×19 = 17,328
        FC 17328→128 **带**bias     2,218,112
        FC   128→256 **带**bias        33,024
        FC   256→362 **带**bias        93,034
        合计                        2,366,730

    输出 362 = 19×19 + pass（动作空间含 pass 位），与既有 `PolicyNetwork` 的
    `torch.cat([x, pass_bias])` 拼出同一维度的做法等价，但 pass 位改成由 FC 学出来。

     表里 policy 分项那一行有笔误：FC 128→256 被写成 32,896，正确值是
    32,768 + 256 = **33,024**。brief §3(b) 因此把合计当成「2,366,602」并以分项
    为准，但 2,366,602 是**任何**按表建出来的模块都达不到的数（nn.Linear(128,256)
    的 bias 恒为 256 个，去掉 bias 则是 32,768，合计变成 2,366,474）。表头自己写的
    2,366,730 反而是对的。详见 report §4(b)。

    与既有 `PolicyNetwork` 的区别（有意偏离）：不再有 `pass_bias` 手工参数，
    也不再支持 3 层 3×3 版本 —— 本类结构由 v21 代的权威表写死。
    """

    def __init__(self, in_channels=184, board_size=19, action_size=None):
        super().__init__()
        self.board_size = board_size
        self.action_size = (board_size * board_size + 1) if action_size is None else action_size

        self.conv1 = nn.Conv2d(in_channels, 96, 1, bias=False)   # 17,664
        self.bn1 = nn.BatchNorm2d(96)                            # 192
        self.conv2 = nn.Conv2d(96, 48, 1, bias=False)            # 4,608
        self.bn2 = nn.BatchNorm2d(48)                            # 96
        flat = 48 * board_size * board_size                      # 17,328
        self.flat_size = flat
        self.fc1 = nn.Linear(flat, 128)                          # 2,218,112
        self.fc2 = nn.Linear(128, 256)                           # 33,024
        self.fc3 = nn.Linear(256, self.action_size)              # 93,034

    def forward(self, x):
        # x: (B, in_channels, board, board) -> (B, action_size) logits
        if x.shape[-1] != self.board_size or x.shape[-2] != self.board_size:
            raise ValueError(
                'FCPolicyHead 的 Flatten 维被 board_size={} 钉死（48×{}×{}={}），'
                '收到输入 {}'.format(self.board_size, self.board_size, self.board_size,
                                      self.flat_size, tuple(x.shape)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
