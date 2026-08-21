# MuZero-Lite 围棋AI

基于MuZero思想的围棋AI，使用有限深度Minimax搜索替代MCTS，针对CPU推理优化。

## 特性

- **MuZero架构**：表示网络 + 动态网络 + 预测网络
- **Minimax搜索**：深度10步，Top-5候选，Alpha-Beta剪枝
- **可调参数**：搜索深度、候选宽度、是否使用Alpha-Beta剪枝
- **CPU/GPU兼容**：自动检测设备，支持AMP混合精度训练
- **自我对弈**：从零开始学习，无需棋谱数据

## 架构

```
输入: 棋盘状态 (19通道 × 9×9)
      ↓
表示网络 (ResNet: 4块, 通道64)
      ↓
隐藏状态 h₀
      ↓
动态网络 (预测 h_{t+1}, r_t)
      ↓
预测网络 (输出策略π, 价值v)
      ↓
Minimax搜索 (深度10步, Top-5, α-β剪枝)
      ↓
输出: 最佳着法
```

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd Go-AI

# 安装依赖
pip install torch numpy pytest
```

## 使用

### 推理（人机对弈）

```bash
# 基本用法
python -m src.inference --mode play

# 使用训练好的模型
python -m src.inference --model models/model.pth --mode play

# 调整搜索参数
python -m src.inference --search-depth 10 --top-k 5 --mode play
```

### 训练

```bash
# 使用默认配置训练
python -m src.training.trainer

# 使用自定义配置
python -c "
from src.config.config import Config
from src.training.trainer import Trainer
from src.networks.resnet import MuZeroNet

config = Config(search_depth=10, top_k=5, num_games=5000)
model = MuZeroNet(in_channels=19, channels=64, num_res_blocks=4, action_size=81)
trainer = Trainer(model=model, board_size=9, search_depth=10, top_k=5, device=config.device)
trainer.train(num_games=5000, save_path='models/model.pth')
"
```

### 评估

```bash
# 评估模型
python -m src.evaluation.evaluator
```

## 配置

### 默认配置

| 参数 | 值 | 说明 |
|------|-----|------|
| board_size | 9 | 棋盘大小 |
| channels | 64 | 网络通道数 |
| num_res_blocks | 4 | 残差块数量 |
| search_depth | 10 | 搜索深度 |
| top_k | 5 | 候选着法数量 |
| use_alpha_beta | True | 是否使用Alpha-Beta剪枝 |
| learning_rate | 1e-3 | 学习率 |
| batch_size | 64 | 批量大小 |
| num_games | 5000 | 训练局数 |

### 预设配置

```python
from src.config.config import get_config

# 快速配置（搜索深度5，Top-3）
config = get_config('fast')

# 精确配置（搜索深度10，Top-5）
config = get_config('accurate')

# CPU配置
config = get_config('cpu')

# GPU配置
config = get_config('gpu')
```

## 性能

### 推理速度

- **CPU**：< 800ms（10步Minimax，Top-5）
- **GPU**：< 100ms（使用AMP）

### 训练时间

- **CPU**：5000局约24-48小时
- **GPU**（V100-32G）：5000局约2-4小时

### 棋力目标

- **9x9棋盘**：业余5段-职业初段

## 文件结构

```
Go-AI/
├── config/              # 配置文件
│   └── config.py
├── src/
│   ├── networks/        # 网络组件
│   │   ├── resnet.py
│   │   └── __init__.py
│   ├── search/          # 搜索算法
│   │   ├── minimax.py
│   │   └── __init__.py
│   ├── training/        # 训练流程
│   │   ├── trainer.py
│   │   └── __init__.py
│   ├── evaluation/      # 评估方法
│   │   ├── evaluator.py
│   │   └── __init__.py
│   ├── inference.py     # 推理代码
│   ├── utils/           # 工具函数
│   │   ├── helpers.py
│   │   └── __init__.py
│   └── __init__.py
├── tests/               # 测试
│   └── test_all.py
├── scripts/             # 脚本
├── docs/                # 文档
├── data/                # 数据
└── README.md
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_all.py::TestMuZeroNet -v
```

## 扩展

### 添加新功能

1. 实现新功能
2. 添加测试
3. 更新文档

### 性能优化

1. 调整网络架构（通道数、残差块数量）
2. 优化搜索算法（调整搜索深度、候选宽度）
3. 使用更高效的训练策略

## 参考

- [MuZero](https://arxiv.org/abs/1911.08265)
- [AlphaZero](https://arxiv.org/abs/1712.01815)
- [AlphaGo](https://www.nature.com/articles/nature24270)

## 许可证

MIT License
