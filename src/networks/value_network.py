import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=16):
        super(ValueNetwork, self).__init__()
        
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        value = self.value_head(x)
        return value
