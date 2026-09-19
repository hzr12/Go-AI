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
