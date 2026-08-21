import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """配置类"""
    
    # 棋盘配置
    board_size: int = 9
    
    # 网络配置
    in_channels: int = 19
    channels: int = 64
    num_res_blocks: int = 4
    action_size: int = 81  # board_size * board_size
    
    # 搜索配置
    search_depth: int = 10
    top_k: int = 5
    use_alpha_beta: bool = True
    
    # 训练配置
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 1000
    num_games: int = 5000
    games_per_batch: int = 256
    save_interval: int = 100
    
    # 损失权重
    policy_weight: float = 1.0
    value_weight: float = 1.0
    reward_weight: float = 0.5
    state_weight: float = 0.5
    
    # 正则化
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    
    # 设备配置
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    
    # 路径配置
    data_dir: str = 'data'
    model_dir: str = 'models'
    log_dir: str = 'logs'
    
    def __post_init__(self):
        """初始化后处理"""
        # 自动检测设备
        if self.device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 更新动作大小
        self.action_size = self.board_size * self.board_size
        
        # 创建目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def get_model_path(self, name: str = 'model') -> str:
        """获取模型路径"""
        return os.path.join(self.model_dir, f'{name}.pth')
    
    def get_data_path(self, name: str = 'data') -> str:
        """获取数据路径"""
        return os.path.join(self.data_dir, f'{name}.npz')
    
    def get_log_path(self, name: str = 'log') -> str:
        """获取日志路径"""
        return os.path.join(self.log_dir, f'{name}.log')


# 默认配置
default_config = Config()


# 配置字典
CONFIGS = {
    'default': default_config,
    'fast': Config(
        search_depth=5,
        top_k=3,
        batch_size=32,
        num_games=1000
    ),
    'accurate': Config(
        search_depth=10,
        top_k=5,
        batch_size=64,
        num_games=5000
    ),
    'cpu': Config(
        device='cpu',
        batch_size=64,
        num_games=5000
    ),
    'gpu': Config(
        device='cuda',
        batch_size=256,
        num_games=5000
    )
}


def get_config(config_name: str = 'default') -> Config:
    """获取配置"""
    if config_name in CONFIGS:
        return CONFIGS[config_name]
    else:
        raise ValueError(f"Unknown config: {config_name}")


def load_config_from_dict(config_dict: dict) -> Config:
    """从字典加载配置"""
    return Config(**config_dict)
