import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=32, action_size=81):
        super(PolicyNetwork, self).__init__()
        self.action_size = action_size

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, 1, 1, bias=False)
        self.pass_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)              # (B, 1, H, W)
        x = x.view(B, -1)              # (B, H*W)
        return torch.cat([x, self.pass_bias.expand(B, 1)], dim=1)  # (B, H*W+1)
