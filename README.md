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
#   --attention-mode {none,mix,all}   无注意力 / 混合 / 全注意力（主干堆叠）
#   --attn-mode {global,window,axial} 注意力计算方式（见下）
#   --num-attention-layers N          mix 模式下注意力块数量
#   --num-heads H                     多头注意力头数
#   --attn-window W                   window 模式窗口边长（默认 7）
python scripts/train_sft.py --device cuda --board-size 19 \
    --attention-mode mix --num-attention-layers 6 --num-heads 8 \
    --attn-mode window --attn-window 7 \
    --data data/sgf_19x19.npz --save-path models/sft_attn_19x19.pth
```

主干注意力：把棋盘 `(B, C, H, W)` 视为 `N=H*W` 个 token 做多头自注意力
（pre-LayerNorm + FFN），捕捉长程依赖（大龙死活、全局厚薄）；卷积残差块
负责局部形状。主干堆叠方式由 `--attention-mode` 控制：
- `none`：纯卷积（最快，局部性最好）
- `mix`（默认）：在 `num_res_blocks` 个块中均匀穿插 `num_attention_layers`
  个注意力混合块
- `all`：全部使用「卷积残差 + 注意力」混合块

### 注意力计算模式（`--attn-mode`，提速关键）⚡

注意力内部计算方式决定速度与显存，由 `--attn-mode` 选择：

| 模式 | 复杂度 | 显存 | 长程建模 | 说明 |
|------|--------|------|----------|------|
| `global` | O(N²·C) | O(N²) | 最强 | 标准全配对，走 SDPA/FlashAttention |
| `window` | O(N·w²) | O(N·w²) | 局部为主 | 滑动窗口（w=7 时约 7× 提速），GPU 上最佳性价比 |
| `axial` | O(2N·√N) | O(N) | 整行/整列 | 先按行后按列两次 1D 注意力，省显存且保长程 |

**实现要点（均已落地）**：
- 统一使用 `torch.nn.functional.scaled_dot_product_attention`，在 Ampere+
  GPU 上自动走 **FlashAttention / Memory-Efficient** 路径，**不物化 N×N 矩阵**，
  显存从 O(N²) 降到 O(N)，速度显著提升；CPU 自动回退到手写注意力。
- `window` 模式把窗口维并入 batch 后调用 SDPA，GPU 上每个窗口仅算 `w²×w²`，
  19 路 w=7 时注意力算力约为全局的 `1/7`。
- `axial` 模式把二维拆成两次 1D SDPA，复杂度约 `2N·max(H,W)`，速度与 global
  持平但显存更低，适合显存受限又要长程的场景。

**推荐组合**：
- 追求最快且显存友好：`--attention-mode mix --attn-mode window --attn-window 7`
- 既要长程又要省显存：`--attention-mode mix --attn-mode axial`
- 最强建模（显存充足）：`--attention-mode mix --attn-mode global`

**额外加速：`torch.compile` 算子融合** ⚡
在支持的机器（GPU + 可用的 C++ 编译器）上，加 `--compile` 让模型走
`torch.compile` 图优化，融合卷积/注意力/BN 等算子，**约 20–40% 提速**：
```bash
python scripts/train_sft.py --device cuda --use-amp --compile \
    --board-size 19 --attn-mode window --attn-window 7 \
    --data data/sgf_19x19.npz --save-path models/sft_attn_19x19.pth
# 推理引擎同样支持
python src/inference.py --model models/sft_attn_19x19.pth --compile --mode selfplay
```
实现已做安全回退：编译是惰性的，错误在首次前向才暴露，因此会在
compile 后用 dummy 输入做一次 warmup；若环境缺编译器（如未装 MSVC `cl`
的 Windows / 无 CUDA 的 CPU），自动回退到 eager 模式并提示，不影响训练推理。

**推荐组合**（叠加 `--use-amp --compile` 收益最大）：
- 追求最快且显存友好：`--attention-mode mix --attn-mode window --attn-window 7`
- 既要长程又要省显存：`--attention-mode mix --attn-mode axial`
- 最强建模（显存充足）：`--attention-mode mix --attn-mode global`

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

## MCTS 搜索（棋力跃升的关键）⚡

纯监督网络直接 argmax 策略 = 0 步前瞻，局部好≠全局好（死活/对杀/收官会崩）。
`isearch/mcts.py` 用 AlphaGoZero 风格的 **PUCT + 批量叶子评估 + 虚拟损失多线程**
把网络当作「先验 P + 叶子价值 v」，每步做 N 次模拟，按访问次数选点：

```bash
# 9 路 MCTS 自对弈（N=200，CPU 也快）
python src/inference.py --model models/sft_9x9.pth --board-size 9 \
    --mode selfplay --use-mcts --simulations 200 --num-threads 4

# 19 路 V100S 全加速组合（推荐）
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --device cuda --use-amp --compile --tf32 \
    --attention-mode mix --attn-mode window --attn-window 7 \
    --mode selfplay --use-mcts --simulations 400 --num-threads 4

# 开启 LightPLS：叶子价值融合轻量 rollout（低价提升搜索深度/棋力）
python src/inference.py --model models/sft_9x9.pth --board-size 9 \
    --mode selfplay --use-mcts --simulations 400 --num-threads 4 \
    --use-rollout --rollout-lambda 0.25
```

### 加速手段（已全部落地）
| 加速项 | 状态 | 说明 |
|--------|------|------|
| 批量叶子评估 | ✅ | `GoAI.predict_batch` 把同一层节点拼 batch 一次前向（GPU 上吞吐 ×10+）|
| **跨线程合并 batch（生产者-消费者）** | ✅ | worker 线程只选路径（廉价），主线程统一 `deepcopy`+`feature_planes`+`predict_batch`，每个叶子只算一次特征 |
| 增量特征 | ✅ | `predict_batch` 接受预计算 12 通道 planes，省去重复 `feature_planes` 计算 |
| TF32 matmul | ✅ | `--tf32`，V100/Amp 上 fp32 矩阵乘约 2–4×，精度损失可忽略 |
| `channels_last` 内存布局 | ✅ | CUDA 上 conv 走 NHWC，conv 友好提速 |
| 虚拟损失 + 多线程 | ✅ | `MCTS.num_threads` 并行选路径，利用多核 |
| `torch.compile` 算子融合 | ✅ | `--compile`，GPU 上约 20–40% |
| `--use-amp` 混合精度 | ✅ | V100 走 fp16（Volta 无 bf16）|
| `window` / `axial` 注意力 | ✅ | `--attn-mode window` 在 GPU 上约 7× 注意力提速 |
| **LightPLS 轻量 rollout** | ✅ | `--use-rollout --rollout-lambda`：叶子价值融合 Tromp-Taylor 快数子，低价提升棋力 |

### 速度预期（V100S 16GB）
- 单次前向 ~1–2 ms；批量前向（一层 64 叶）~5–10 ms
- MCTS N=400：单步 **~1–2 秒**（19 路）；N=800 ~3–4 秒；9 路 N=200 单步 <1 秒
- 相比纯 argmax 策略，棋力是**数量级**提升（无搜索→有搜索）

### 评估脚本
```bash
# 对随机策略胜率（对比 policy-argmax vs MCTS）
python scripts/evaluate.py --board-size 9 --mode random --use-mcts --simulations 200
# 推理速度基准（单样本 vs 批量32）
python scripts/evaluate.py --board-size 19 --mode benchmark --device cuda --use-amp --compile
```

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
