# 围棋AI

基于AlphaGo思想的围棋AI，使用多网络架构（策略网络 + 价值网络 + 快速策略网络），无搜索，纯网络推理。

## 特性

- **多网络架构**：策略网络 + 价值网络 + 快速策略网络并行推理
- **无搜索**：直接从神经网络输出着法，推理速度快
- **CPU/GPU兼容**：自动检测设备，支持AMP混合精度训练
- **自我对弈**：从零开始学习，无需棋谱数据
- **可扩展**：支持9×9到19×19棋盘

## 架构

```
输入: 棋盘状态 (19通道 × 9×9)
      ↓
┌─────┴─────┐
│           │
↓           ↓
骨干网络    快速策略网络
(64ch×4残差) (72ch×3残差)
│           │
├───┐       │
↓   ↓       ↓
策略 价值   快速策略
网络 网络   网络
(32ch)(16ch) (72ch×3)
│    │       │
↓    ↓       ↓
策略 价值   快速策略
logits v   logits
```

## 网络参数量

| 网络 | 参数量 | 用途 |
|------|--------|------|
| 骨干网络 | ~150K | 特征提取 |
| 策略网络 | ~21K | 精确着法预测 |
| 价值网络 | ~8K | 局面评估 |
| 快速策略网络 | ~300K | 快速着法建议 |
| **总计** | **~1M** | - |

## 安装

```bash
# 克隆仓库
git clone https://github.com/hzr12/Go-AI.git
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
python -m src.inference --model models/alphago_model.pth --mode play

# 评估当前局面
python -m src.inference --model models/alphago_model.pth --mode eval

# 分析着法概率
python -m src.inference --model models/alphago_model.pth --mode analyze
```

### 训练

```bash
# 基本训练
python scripts/train.py --num-games 5000

# GPU训练（推荐）
python scripts/train.py --num-games 5000 --device cuda --use-amp --batch-size 256

# 自定义参数
python scripts/train.py --num-games 10000 \
    --backbone-channels 128 \
    --backbone-res-blocks 6 \
    --lr 5e-4 \
    --batch-size 512 \
    --save-path models/custom_model.pth
```

### 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 仅测试网络
python -m pytest tests/test_alphanet.py -v

# 仅测试训练
python -m pytest tests/test_alphanet.py::TestAlphaGoTrainer -v
```

## 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--board-size` | 9 | 棋盘大小 |
| `--backbone-channels` | 64 | 骨干网络通道数 |
| `--backbone-res-blocks` | 4 | 骨干网络残差块数量 |
| `--policy-channels` | 32 | 策略网络通道数 |
| `--value-channels` | 16 | 价值网络通道数 |
| `--fast-channels` | 72 | 快速策略网络通道数 |
| `--fast-res-blocks` | 3 | 快速策略网络残差块数量 |
| `--lr` | 1e-3 | 学习率 |
| `--batch-size` | 256 | 批量大小（GPU推荐256-512） |
| `--num-games` | 5000 | 训练局数 |
| `--policy-weight` | 1.0 | 策略损失权重 |
| `--value-weight` | 1.0 | 价值损失权重 |
| `--fast-weight` | 0.5 | 快速策略损失权重 |
| `--temperature` | 1.0 | 温度参数 |
| `--use-amp` | 自动 | 使用混合精度 |

## 训练流程

1. **自我对弈**：策略网络生成着法，收集游戏数据
2. **训练**：同时训练三个网络
   - 策略损失：KL散度（从自我对弈策略学习）
   - 价值损失：MSE（从游戏结果学习）
   - 快速策略损失：KL散度（从策略网络蒸馏）
3. **保存**：定期保存模型检查点

## 项目结构

```
Go-AI/
├── src/
│   ├── networks/
│   │   ├── backbone.py      # 共享骨干网络
│   │   ├── policy_network.py # 策略网络
│   │   ├── value_network.py  # 价值网络
│   │   ├── fast_network.py   # 快速策略网络
│   │   └── alphanet.py       # AlphaGoNet主网络
│   ├── training/
│   │   └── trainer.py        # 训练器
│   └── inference.py          # 推理引擎
├── scripts/
│   ├── train.py              # 训练脚本
│   └── inference.py          # 推理脚本
├── tests/
│   └── test_alphanet.py      # 测试用例
└── models/                   # 模型保存目录
```

## 性能

- **推理速度**：~1000 moves/sec (CPU)
- **训练速度**：~0.3s/game (CPU), ~0.05s/game (GPU)
- **内存占用**：~500MB (CPU), ~1GB (GPU)

## 依赖

- Python 3.8+
- PyTorch 1.9+
- NumPy
- pytest (开发依赖)

## License

MIT
