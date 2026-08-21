import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块"""
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


class RepresentationNetwork(nn.Module):
    """表示网络：将棋盘状态编码为隐藏状态"""
    def __init__(self, in_channels=19, channels=64, num_res_blocks=4):
        super(RepresentationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.res_blocks(out)
        return out


class DynamicsNetwork(nn.Module):
    """动态网络：预测下一个状态和奖励"""
    def __init__(self, channels=64, num_res_blocks=4, action_planes=1):
        super(DynamicsNetwork, self).__init__()
        # 动作平面：将动作编码为特征图
        self.action_conv = nn.Conv2d(1, action_planes, 1)
        
        # 状态转移网络
        self.conv1 = nn.Conv2d(channels + action_planes, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Flatten(),
            nn.Linear(81, 1),
            nn.Tanh()  # 奖励范围[-1, 1]
        )
    
    def forward(self, state, action):
        # 将动作编码为特征图
        action_plane = self.action_conv(action)
        
        # 拼接状态和动作
        x = torch.cat([state, action_plane], dim=1)
        
        # 状态转移
        out = F.relu(self.bn1(self.conv1(x)))
        next_state = self.res_blocks(out)
        
        # 奖励预测
        reward = self.reward_head(next_state)
        
        return next_state, reward


class PredictionNetwork(nn.Module):
    """预测网络：输出策略和价值"""
    def __init__(self, channels=64, action_size=81):
        super(PredictionNetwork, self).__init__()
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 81, action_size)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(81, 1),
            nn.Tanh()  # 价值范围[-1, 1]
        )
    
    def forward(self, x):
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class MuZeroNet(nn.Module):
    """MuZero网络：组合表示、动态和预测网络"""
    def __init__(self, in_channels=19, channels=64, num_res_blocks=4, action_size=81):
        super(MuZeroNet, self).__init__()
        self.representation = RepresentationNetwork(in_channels, channels, num_res_blocks)
        self.dynamics = DynamicsNetwork(channels, num_res_blocks)
        self.prediction = PredictionNetwork(channels, action_size)
        self.action_size = action_size
    
    def initial_inference(self, observation):
        """初始推理：从观察得到策略和价值"""
        state = self.representation(observation)
        policy, value = self.prediction(state)
        return state, policy, value
    
    def recurrent_inference(self, state, action):
        """递归推理：给定状态和动作，预测下一个状态、奖励、策略和价值"""
        # 将动作转换为one-hot编码
        action_one_hot = torch.zeros(state.shape[0], 1, 9, 9, device=state.device)
        action_one_hot.view(-1, 81).scatter_(1, action.unsqueeze(1), 1)
        
        next_state, reward = self.dynamics(state, action_one_hot)
        policy, value = self.prediction(next_state)
        return next_state, reward, policy, value
