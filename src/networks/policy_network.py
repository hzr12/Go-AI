import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=32, action_size=81):
        super(PolicyNetwork, self).__init__()
        self.action_size = action_size
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, action_size)
        )
    
    def forward(self, x):
        policy = self.policy_head(x)
        return policy
