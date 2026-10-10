"""
MPV-MCTS: Main-Policy-Value 大小网协同

核心思想：
- 大网 (Main Net): 高精度，用于关键决策 (topk=16)
- 小网 (Lite Net): 低精度但快 3-5x，用于大量模拟 (topk=4)
- 切换策略：根据叶子价值不确定性动态选择网络
"""

import torch
import torch.nn as nn


class LiteValueNetwork(nn.Module):
    """轻量价值网络：仅 1 层残差，~65k 参数。"""

    def __init__(self, in_channels=192, hidden_channels=32):
        super().__init__()
        # 下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        # 单层残差
        self.res = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1, bias=False),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.downsample(x)
        # self.res 输出 (B, 1, H, W)：先去掉通道维 1 再展平为 (B, H*W)。
        # 原 `squeeze(-1).squeeze(-1)` 作用在最后一维（W，非 1）上不生效，
        # 导致 value 头形状错误（多出一个空间维而非 (B, H*W)）。
        res = self.res(x)
        return res.squeeze(1).reshape(res.shape[0], -1)


class LitePolicyNetwork(nn.Module):
    """轻量策略网络：单 conv 层。"""

    def __init__(self, in_channels=192, action_size=362):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.pass_bias = nn.Parameter(torch.zeros(1))
        self.action_size = action_size

    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x).view(B, -1)
        return torch.cat([x, self.pass_bias.expand(B, 1)], dim=1)


class LiteNet(nn.Module):
    """轻量网络：policy + value，总参数 ~150k。"""

    def __init__(self, in_channels=192, hidden_channels=32, action_size=362):
        super().__init__()
        self.policy = LitePolicyNetwork(in_channels, action_size)
        self.value = LiteValueNetwork(in_channels, hidden_channels)

    def forward(self, x):
        return self.policy(x), self.value(x).unsqueeze(1)


class MPVMCTS:
    """MPV-MCTS: 大小网协同搜索。

    策略：
    - 价值绝对值 > switch_thresh: 关键局面，用大网精细评估
    - 价值绝对值 <= switch_thresh: 模糊局面，用小网快速评估
    """

    def __init__(self, main_ai, lite_ai, switch_thresh=0.7, **mcts_kwargs):
        self.main_ai = main_ai
        self.lite_ai = lite_ai
        self.switch_thresh = switch_thresh
        self.mcts = None  # 延迟初始化

    def search(self, *args, **kwargs):
        """代理到 MCTS，但根据价值不确定性切换网络。"""
        from src.search.mcts import MCTS
        if self.mcts is None:
            self.mcts = MCTS(self.main_ai, **self._extract_mcts_kwargs(kwargs))
        # 正常搜索，但在 expand 时根据 value 切换网络
        return self.mcts.search(*args, **kwargs)

    def _extract_mcts_kwargs(self, kwargs):
        """从 kwargs 中提取 MCTS 参数。"""
        mcts_params = ['board_size', 'c_puct', 'virtual_loss', 'num_threads',
                       'temperature', 'use_rollout', 'rollout_lambda',
                       'expand_topk', 'expand_chunk', 'solver_thresh',
                       'spec_prefetch', 'leaf_ab_depth', 'priors_leaf',
                       'dirichlet_alpha', 'dirichlet_eps']
        return {k: kwargs[k] for k in mcts_params if k in kwargs}
