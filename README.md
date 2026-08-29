# Go-AI — 基于监督学习（SFT）的围棋 AI

一个使用 **监督学习** 从 SGF 棋谱训练围棋策略/价值网络的轻量级项目。
网络结构借鉴 AlphaGo Zero 的「共享骨干 + 策略头 + 价值头」，输入为统一的
**12 通道特征平面**，输出为 `(policy, value)`。

- **policy**：棋盘每点 + 虚着(pass) 的概率分布（softmax）
- **value**：当前执子方视角的局面胜率，tanh 映射到 `[-1, 1]`

## 项目结构

```
src/
  game/go_rules.py      围棋规则引擎（提子 / 打劫 / 禁自杀 / 虚着 / 数子）
                        + feature_planes() 统一生成 12 通道特征
  data/
    sgf_parser.py       SGF 棋谱解析（含让子 AB/AW、贴目 KM、虚着 []）
    dataset.py          监督学习数据集（紧凑 npz + 随机对称增广）
  networks/
    backbone.py         共享骨干（卷积残差 + 可选多头自注意力混合）
    policy_network.py   策略头（1×1 卷积 + 全局平均池化 + 线性）
    value_network.py    价值头（Tanh 输出 [-1,1]）
    alphanet.py         AlphaGoNet：串联上述三者的总模型
  inference.py          对弈 / 自对弈 / 局面分析引擎（GoAI）
scripts/
  build_dataset.py      SGF -> 复局 -> 紧凑 npz 数据集
  train_sft.py          监督训练（AMP / AdamW / Cosine 调度 / 留出集 top1）
tests/
  test_go_rules.py      规则引擎单测（提子 / 劫 / 自杀 / 数子）
  test_sgf_parser.py    SGF 解析单测
  test_alphanet.py      网络与推理引擎冒烟测试
```

## 特征平面（12 通道，单一数据源）

见 `src/game/go_rules.py::GoBoard.feature_planes`，训练（`dataset.py`）与推理
（`inference.py`）共用同一实现，避免训练/推理特征不一致：

| 通道 | 含义 |
|------|------|
| 0    | 己方棋子 |
| 1..3 | 己方前 1/2/3 手落子 |
| 4    | 对手棋子 |
| 5..7 | 对手前 1/2/3 手落子 |
| 8    | 合法点掩码（已排除劫禁着点） |
| 9    | 执子方常数（to_play，±1） |
| 10   | 己方气数=1 的块掩码 |
| 11   | 对手气数=1 的块掩码 |

> 视角以「当前执子方」为「己方」，便于网络专注相对态势。

## 依赖

```bash
pip install -r requirements.txt
```

## 数据准备

将 SGF 棋谱（`.sgf` 或打包的 `.tar.gz` / `.tgz`）放入 `data/` 目录，然后生成训练集：

```bash
# 9 路
python scripts/build_dataset.py --board-size 9 --max-games 2000 --output data/sgf_9x9.npz
# 19 路
python scripts/build_dataset.py --board-size 19 --max-games 30000 --output data/sgf_19x19.npz
```

`build_dataset.py` 会：解析 SGF → 用 `GoBoard` 复局 → 每手收集
`(棋盘, 双方近 3 手历史, 劫点, 着法, 胜负标签, 执子方)`，并做坐标越界 /
让子 / 结果缺失等过滤，最终保存为紧凑 `npz`。

## 训练

```bash
# CPU 快速验证
python scripts/train_sft.py --device cpu --board-size 9 --batch-size 128 --epochs 3

# GPU 全量（19 路，混合精度）
python scripts/train_sft.py --device cuda --use-amp \
    --board-size 19 --batch-size 512 --epochs 5 \
    --data data/sgf_19x19.npz --save-path models/sft_19x19.pth

# 注意力配置（默认 mix 模式：卷积打底 + 注意力提质）
#   --attention-mode {none,mix,all}   无注意力 / 混合 / 全注意力
#   --num-attention-layers N          mix 模式下注意力块数量
#   --num-heads H                     多头注意力头数
python scripts/train_sft.py --device cuda --board-size 19 \
    --attention-mode mix --num-attention-layers 6 --num-heads 8 \
    --data data/sgf_19x19.npz --save-path models/sft_attn_19x19.pth
```

主干注意力：把棋盘 `(B, C, H, W)` 视为 `N=H*W` 个 token 做多头自注意力
（pre-LayerNorm + FFN），捕捉长程依赖（大龙死活、全局厚薄）；卷积残差块
负责局部形状。三种模式由 `--attention-mode` 控制：
- `none`：纯卷积（最快，局部性最好）
- `mix`（默认）：在 `num_res_blocks` 个块中均匀穿插 `num_attention_layers`
  个注意力混合块
- `all`：全部使用「卷积残差 + 注意力」混合块

损失：`L = CrossEntropy(policy, move) + MSELoss(value, z)`，其中 `z ∈ {+1,-1}`
来自棋谱 `RE` 结果（当前执子方视角）。虚着作为最后一类的专用类别
`board_size²`。

## 推理 / 对弈

```bash
# 模型自对弈若干局，输出黑方胜率
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --mode selfplay --games 4 --temperature 0.8

# 人机对弈（人类执黑先手）
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --mode human --human-color 1

# 交互式局面分析（展示 AI 的 top-k 候选着法）
python src/inference.py --model models/sft_19x19.pth --board-size 19 --mode analyze
```

人机/分析模式下，坐标用字母输入，例如 `ce` 表示第 5 列(e) 第 3 行(c)
（a=第1列/行，依此类推），输入 `pass` 虚着，`q` 退出分析模式。

## 测试

```bash
pytest tests/ -q
```

## 设计说明

- **为什么是监督学习而非强化学习**：本项目目标是「从人类棋谱学习合理着法」，
  规则引擎保证复局合法，网络在每一步预测与人类相同的着法，并回归终局胜负。
- **单一特征来源**：`feature_planes` 同时服务训练与推理，杜绝特征漂移。
- **紧凑数据集**：原始棋局以 `int8` 棋盘 + 历史索引存储，运行时再展开为
  12 通道并随机施加 8 向对称增广，兼顾内存与数据多样性。
