# Go-AI：监督学习围棋 AI（策略-价值网络 + MCTS + LightPLS）

> 一个从 SGF 棋谱做监督学习（SFT）的围棋 AI。主线为 **AlphaGoZero 风格的 12 通道策略-价值网络**，
> 推理阶段用 **MCTS（PUCT + 批量叶子评估 + 虚拟损失多线程）** 选点，并可叠加 **LightPLS 轻量 rollout**
> 提升无强 RL 时的棋力。规则引擎为自建 `GoBoard`（Tromp-Taylor 数子、中国规则基础）。

> **状态说明**：当前为**纯监督学习**路线（从人类棋谱学着法），不是 AlphaGo/MuZero 的自我对弈 RL。
> 已删除与 12 通道 SFT 冲突的旧 19 通道死代码（`resnet.py` / `minimax.py` / `evaluator.py` / `alpha_evaluator.py` / `config.py`）。

---

## 目录

- [1. 环境要求](#1-环境要求)
- [2. 安装](#2-安装)
- [3. 项目结构](#3-项目结构)
- [4. 核心概念](#4-核心概念)
  - [4.1 棋盘与规则引擎 `GoBoard`](#41-棋盘与规则引擎-goboard)
  - [4.2 特征平面（12 通道）](#42-特征平面12-通道)
  - [4.3 网络架构 `AlphaGoNet`](#43-网络架构-alphagonet)
  - [4.4 MCTS 搜索](#44-mcts-搜索)
  - [4.5 LightPLS 轻量 rollout](#45-lightpls-轻量-rollout)
- [5. 数据准备](#5-数据准备)
  - [5.1 SGF 解析规则](#51-sgf-解析规则)
  - [5.2 构建训练集 `build_dataset.py`](#52-构建训练集-build_datasetpy)
  - [5.3 训练集内存布局](#53-训练集内存布局)
- [6. 训练 `train_sft.py`](#6-训练-train_sftpy)
- [7. 推理与对弈 `inference.py`](#7-推理与对弈-inferencepy)
- [8. 评估 `evaluate.py`](#8-评估-evaluatepy)
- [9. 加速手段总览](#9-加速手段总览)
- [10. 完整工作流示例（云端 V100S）](#10-完整工作流示例云端-v100s)
- [11. API 速查](#11-api-速查)
- [12. 常见问题与排错](#12-常见问题与排错)
- [13. 已知死代码 / 待办](#13-已知死代码--待办)

---

## 1. 环境要求

| 组件 | 最低 | 推荐（训练/推理） |
|------|------|-------------------|
| Python | 3.8 | 3.10+ |
| PyTorch | ≥ 1.9 | ≥ 2.0（用上 `torch.compile`）|
| 算力 | 任意 CPU | NVIDIA GPU（V100S / Ampere），CUDA 11.8+ |
| 磁盘 | 几百 MB | 棋谱数据 + npz（19 路全量可能数十 GB）|
| 内存 | 4 GB | 16 GB+（npz 全量加载到内存）|

依赖仅 `torch` / `numpy` / `pytest`（见 `requirements.txt`）。**无** `tensorflow`、无额外围棋库。

```text
torch>=1.9.0
numpy>=1.19.0
pytest>=6.0.0
```

---

## 2. 安装

```bash
# 1) 克隆（假设已在仓库根目录）
cd f:/AI/Go-AI

# 2) 创建虚拟环境（可选但推荐）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3) 安装依赖
pip install -r requirements.txt

# 4) 装 GPU 版 PyTorch（云端 V100S，CUDA 12.1 示例）
pip install torch --index-url https://download.pytorch.org/whl/cu121
#   本地仅 CPU 调试：
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> Windows 上若要用 `torch.compile`，需有 MSVC 工具链（`vcvars64.bat` / `cl.exe`）。
> 本项目此前已在 `D:\MSVC` 验证：`torch 2.12.0+cpu` + `torch.compile` 在 CPU 上可用。

---

## 3. 项目结构

```text
Go-AI/
├── README.md                     # 本文件
├── requirements.txt              # 依赖
├── data/
│   └── games/games/              # SGF 棋谱（36648 个 .sgf，多层目录）
│       └── Aizu/01/1.sgf ...
├── models/                       # 训练产出权重（*.pt / *.pth），默认不存在需自训练
├── src/
│   ├── __main__.py
│   ├── inference.py              # GoAI 推理入口 + CLI（selfplay / human / analyze）
│   ├── game/
│   │   └── go_rules.py           # GoBoard 规则引擎 + 12 通道 feature_planes
│   ├── networks/
│   │   ├── alphanet.py           # AlphaGoNet（策略+价值双头）
│   │   ├── backbone.py           # SharedBackbone（ResBlock + 注意力）
│   │   ├── policy_network.py     # PolicyNetwork 头
│   │   └── value_network.py      # ValueNetwork 头
│   ├── search/
│   │   ├── mcts.py               # MCTS（批量叶子评估 + 虚拟损失 + LightPLS 融合）
│   │   └── light_rollout.py      # FastPolicy + light_rollout（Tromp-Taylor 数子）
│   ├── data/
│   │   ├── dataset.py            # SupervisedDataset（紧凑存储 + 随机对称增广）
│   │   └── sgf_parser.py         # SGFParser（解析棋谱）
│   └── utils/
│       └── helpers.py            # print_board 等
├── scripts/
│   ├── train_sft.py              # 监督学习训练（CE + MSE）
│   ├── build_dataset.py          # SGF 目录/tgz -> npz 训练集
│   ├── evaluate.py               # 评估（vs 随机 / 自对弈 / 速度基准）
│   └── demo.py                   # ⚠️ 死代码（旧 API，勿运行，见 §13）
└── tests/
    ├── test_mcts.py              # MCTS + LightPLS 单测
    └── ...                       # 其他单测
```

> 入口统一用 `src.inference.GoAI`，所有脚本通过 `sys.path.insert(0, 仓库根)` 以 `src.xxx` 方式导入。

---

## 4. 核心概念

### 4.1 棋盘与规则引擎 `GoBoard`

文件：`src/game/go_rules.py`

- 棋盘状态：`board` 为 `int8 (n, n)`，取值 `-1`=白、`0`=空、`1`=黑。
- 当前执子方：`current_player`（`1`=黑 / `-1`=白），**注意不是** `to_play`（MCTS 内部用 1/2 表示，二者不等价）。
- 核心方法：
  - `play(mv)` → `bool`：落子 `mv`；`mv == n*n` 表示**虚着(pass)**；非法着法返回 `False`（不抛异常）。
  - `get_legal_moves()` → `bool (n*n,)` 一维掩码（`True`=合法）。
  - `feature_planes(my_hist, op_hist, to_play)` → `(12, n, n)` 特征（见 §4.2）。
  - `score()` → `float`：Tromp-Taylor 数子，**黑 − 白** 目数（终局判定）。
  - `parse_move_str(s, color)` → `(ok, mv)`：坐标串（如 `"ce"`）转扁平索引。
  - `to_string()`：文本化棋盘（供 CLI 展示）。
- 规则：气(liberties)计算、提子、`ko_point` 单劫禁着、双 pass 终局。贴目 `komi` 默认 6.5（中国规则常用）。
- 克隆：无 `clone()` 方法，MCTS 用 `copy.deepcopy(board)` 复制局面。

### 4.2 特征平面（12 通道）

由 `GoBoard.feature_planes(my_hist, op_hist, to_play)` 产生，形状 `(12, n, n)`，单一真相来源（训练/推理/评估共用）：

| 通道 | 含义 |
|------|------|
| 0 | 当前执子方(to_play)的棋子 |
| 1 | 己方最近第 1 手（扁平坐标置 1，其余 0）|
| 2 | 己方最近第 2 手 |
| 3 | 己方最近第 3 手 |
| 4 | 对手棋子 |
| 5 | 对手最近第 1 手 |
| 6 | 对手最近第 2 手 |
| 7 | 对手最近第 3 手 |
| 8 | 合法着法掩码（1=合法）|
| 9 | 常量平面，值 = `to_play`（全 1 若黑 / 全 -1 若白）|
| 10 | 己方「气=1」棋子掩码（送吃预警）|
| 11 | 对手「气=1」棋子掩码 |

> 历史手用「最近 3 手」环形填充（不足 3 手用 `-1` 表示无）。`my_hist`/`op_hist` 为长度 3 的扁平坐标列表。

### 4.3 网络架构 `AlphaGoNet`

文件：`src/networks/alphanet.py`

```
输入 (B, 12, H, W)
      │
SharedBackbone(in_ch=12, ch=128, res_blocks=12, 注意力模式)
      │  -> (B, 128, H, W) 共享表征
      ├─ PolicyNetwork(ch=128 -> 64, action_size) -> (B, A) logits
      └─ ValueNetwork(ch=128 -> 32, 1)             -> (B, 1) Tanh[-1,1]（黑方视角胜率）
```

- `forward(x)` → `(policy_logits, value)`；`policy_logits` 在推理时经 `softmax` 得概率。
- `action_size = n*n + 1`，**多出的 1 类是 pass**。
- 注意力（可选，默认 `mix`）：`global`（全配对）/ `window`（滑动窗口，`--attn-window`）/ `axial`（轴向）；
  `attention_mode`：`none`（纯卷积）/ `mix`（卷积+注意力混合）/ `all`（全注意力）。
  V100 上推荐 `--attn-mode window --attn-window 7`（注意力约 7× 提速）。
- 相比原版：已删除 `fast_policy` 头（9x9 占 ~48% 算力却未被使用）；输入从 19 通道降到 12（原 19 通道里 13 个恒为 0）。

### 4.4 MCTS 搜索

文件：`src/search/mcts.py`，类 `MCTS`

- 算法：PUCT（AlphaGoZero 风格），`score = Q + U`，`U = c_puct * prior * sqrt(parent_visits) / (1 + child_visits)`。
- 节点：`MCTSNode(board, my_hist, op_hist, to_play, move_int, prior, visit, value_sum, virtual_loss)`。
- 展开：对叶子的每个合法着法（含 pass），`deepcopy(board)` 走一步，批量拼成 `states` 一次 `GoAI.predict_batch` 得到
  `(policies, values)`，写入子节点先验与价值。
- **加速（生产者-消费者并行）**：`num_threads` 个 worker 线程只负责「选路径」（纯 CPU 估算 PUCT，极廉价，加虚拟损失占位），
  主线程从队列**批量取出叶子**，统一 `deepcopy`+`feature_planes`+`predict_batch`（昂贵部分一次大 batch 前向），
  每个叶子只构造一次特征、只前向一次。`--num-threads 4` 即可。
- 虚拟损失（virtual loss）：并行模拟时对路径占位，避免多线程反复选同一条路径。
- 输出：`best_move(...)` → `(move_int, is_pass, root_value)`；或 `search(...)` → `(visits, probs, root_value)`。
- 选点：温度 `temperature>0` 按访问次数分布采样；`temperature=0` 贪心取访问最高。

### 4.5 LightPLS 轻量 rollout

文件：`src/search/light_rollout.py`

- `FastPolicy`：纯 numpy 启发式轻量走子策略（邻边奖励 + 避免送吃）；采样走子到终局。
- `light_rollout(board, policy, max_steps, rng)`：从当前局面用 `FastPolicy` 随机走子到双 pass 终局，
  用 `board.score()`（Tromp-Taylor）得**发起方视角**胜率 `+1/-1/0`。
- **价值融合**：叶子最终价值 `v = (1-λ)·v_net + λ·v_rollout`（`--rollout-lambda`，默认 0.25）。
- 意义：在不强 RL 的前提下，用一次低价随机推演补充「全局胜负」信号，显著提升搜索深度与棋力；
  单次 rollout 成本 << 一次网络前向（尤其 9 路）。可用 `--use-rollout` 开启。

---

## 5. 数据准备

### 5.1 SGF 解析规则

文件：`src/data/sgf_parser.py`，类 `SGFParser`

- 支持属性：`SZ`(棋盘大小)、`RE`(结果)、`PB`/`PW`(对局者)、`DT`(日期)、`KM`(贴目，默认 6.5)。
- 着法提取：正则 `(?:;|\A)(AB|AW|[BW])((\[[^\]]*\])+)`，**必须以 `;` 或开头锚定**，避免把 `BR`/`WR`/`PB`/`PW`/`KM`
  里的 `B`/`W` 误当落子。
- **pass** 用空坐标 `B[]` / `W[]` 表示（此前被丢弃导致黑白错位，已修复）。
- **让子** `AB[..]`（黑）/ `AW[..]`（白）作为开局先行子，按其出现顺序并入着法序列。
- 坐标：`a..s` → 0..18；`tt` 或超范围按 pass 处理。

> **棋盘尺寸兼容（居中 pad）**：`build_dataset.build` 只跳过**大于** `--board-size` 的棋谱；
> 小于目标尺寸的棋谱（如 9x9 喂到 19x19）会**居中填充**到目标棋盘——偏移量
> `off = (board_size - game.board_size)//2`，落子坐标 `(r,c)` 映射为 `(r+off, c+off)`。
> 例如 9x9 棋谱在 19x19 上落在 `(5..13, 5..13)` 居中区域，棋形对称不偏，可直接混合训练扩充数据。
> 大于目标尺寸的棋谱无法放入小棋盘，照常跳过。
- `parse_result_to_value(RE)`：`B+`→`+1`（黑胜）、`W+`→`-1`（白胜）、`0`→`0`、其余→`None`（丢弃该样本）。

### 5.2 构建训练集 `build_dataset.py`

```bash
python scripts/build_dataset.py --src <目录或 .tgz> --out data/sgf_19x19.npz \
    --board-size 19 [--max-games N]
```

参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--src` | 必填 | SGF **目录**（递归 `**/*.sgf`）或 `.tgz`/`.tar.gz`（内含 .sgf）；也支持单个 .sgf 文件 |
| `--out` | `data/sft_dataset.npz` | 输出 npz 路径（父目录自动 `makedirs`）|
| `--board-size` | `19` | 只保留该尺寸的棋谱（其余跳过）|
| `--max-games` | `None`（全部）| 最多处理的棋局数（先小批量跑通用）|
| `--chunk-size` | `50000` | **流式分片落盘阈值**：每攒够这么多样本就 flush 成一个临时 npz 分片，最后合并成单个 npz。峰值内存仅约一个 chunk（几十 MB），避免全量常驻内存 OOM。设为 `0` 退回旧的全量内存模式 |

行为：
- 对每局：解析 → 校验 `board_size` 与 `RE`（无效则 `skip`）→ 逐手重放（用 `GoBoard.play`）得到局面 + 监督目标。
- 每个落子位置产生 **1 个样本**；监督目标 `move` = 该手扁平坐标，`value` = 该局 `RE` 的胜负标签（全样本同值）。
- 历史 `my_hist`/`op_hist` 用 `split_hist` 从「最近落子序列」拆成各 3 手。
- **流式模式（`--chunk-size > 0`，默认开启）**：不把全部样本常驻内存，而是每满一个 chunk 就写入临时分片（落盘到系统临时目录），全部解析完成后 `np.concatenate` 各分片再 `np.savez_compressed` 合并输出、并清理临时文件。运行时打印每个分片进度。
- 输出统计：`有效局 / 跳过` 与最终样本数。

```bash
# 示例：先用 200 局跑通，看 有效/跳过 比例
python scripts/build_dataset.py --src data/games/games/ --out data/sgf_9x9.npz \
    --board-size 9 --max-games 200

# 全量 19 路
python scripts/build_dataset.py --src data/games/games/ --out data/sgf_19x19.npz --board-size 19
```

> 注意：你的数据目录是 `data/games/games/`（双层 games），sgf 实际就在这下面。直接传该目录即可。
> 9 路/19 路棋谱混在一起时，脚本按 `--board-size` 自动过滤。

### 5.3 训练集内存布局

`SupervisedDataset`（`src/data/dataset.py`）以紧凑 numpy 保存，**运行时**才展开 12 通道 + 随机对称增广（等效 8× 静态增强，内存仅 1/8）：

| 字段 | dtype | shape | 含义 |
|------|-------|-------|------|
| `boards` | int8 | (N, n, n) | 局面，-1/0/1 |
| `my_hist` | int16 | (N, 3) | 己方前 3 手扁平坐标（-1 填充）|
| `op_hist` | int16 | (N, 3) | 对手前 3 手 |
| `ko` | int16 | (N,) | 劫禁着点（-1 无）|
| `moves` | int16 | (N,) | 监督目标着法（0..n*n-1，pass=n*n）|
| `values` | int8 | (N,) | 胜负标签（+1 黑 / -1 白）|
| `to_play` | int8 | (N,) | 该样本轮到谁（1 黑 / -1 白）|

- `sample_batch(idxs, device)` → `(state(B,12,H,W) fp32, move(B,) int64, value(B,1) fp32)`。
- 对称增广：每样本随机选 8 种变换之一（旋转 0/90/180/270 × 翻转），同时作用于特征平面与着法坐标（`SYMMETRIES`）。

---

## 6. 训练 `train_sft.py`

```bash
python scripts/train_sft.py --data data/sgf_19x19.npz --out models/sft_19x19.pth \
    --device cuda --use-amp --compile \
    --board-size 19 --batch-size 512 --epochs 5 \
    --attention-mode mix --attn-mode window --attn-window 7
```

参数表：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | 必填 | **单个 `.npz` 训练集，或包含多个 `.tgz`/`.tar.gz`/`.npz` 的目录**（目录模式自动递归扫描并合并所有分片，无需先 `build_dataset` 成单个 npz）|
| `--max-games-per-tgz` | `0`（全部）| 仅目录模式生效：每个 tgz 最多解析的棋局数，用于子采样控制内存 |
| `--out` | `models/sft.pt` | 权重输出路径（父目录自动建）|
| `--device` | `auto` | `cuda` / `cpu`；`auto`=有 GPU 用 cuda |
| `--use-amp` | 关 | **在 cuda 上默认强制开启**（fp16，V100 无 bf16）|
| `--batch-size` | `512` | 每步批量 |
| `--epochs` | `5` | 训练轮数 |
| `--lr` | `2e-3` | 学习率（AdamW）|
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--board-size` | `19` | 棋盘大小（必须与数据一致）|
| `--save-every` | `2000` | 每 N 步存 `.latest` |
| `--eval-every` | `5000` | 每 N 步在 2% 留出集上打印 `eval_top1` |
| `--attention-mode` | `mix` | `none`/`mix`/`all` |
| `--num-attention-layers` | `4` | mix 模式注意力块数 |
| `--num-heads` | `4` | 多头头数 |
| `--attention-dropout` | `0.0` | 注意力 dropout |
| `--attn-mode` | `global` | `global`/`window`/`axial` |
| `--attn-window` | `7` | window 模式窗口边长 |
| `--compile` | 关 | `torch.compile` 算子融合（GPU +20~40%，首次迭代较慢）|

训练细节：
- 损失：`L = CrossEntropy(policy_logits, move) + MSELoss(value, z)`，其中 `z` 为棋谱胜负标签。
- 优化器：AdamW + `CosineAnnealingLR`。
- 数据划分：98% 训练 / 2% 留出（`eval_top1` 监控泛化）。
- `cudnn.benchmark=True`；CPU 线程数限制 `min(8, cpu_count)`。
- 最佳模型自动覆盖 `--out`；定期存 `.latest` 便于续训/回滚。

> 先小跑通：`--device cpu --epochs 1 --batch-size 128 --board-size 9`，确认 `eval_top1` 有输出、模型能存盘。

---

## 7. 推理与对弈 `inference.py`

`GoAI` 类（`src/inference.py`）封装模型加载、单/批量前向、MCTS 选点、自对弈、人机对弈，并提供 CLI。

构造参数（节选）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `model_path` | `None` | 权重路径；`None` 时随机初始化（未训练，仅测管线）|
| `board_size` | `19` | 棋盘大小 |
| `device` | `auto` | `cuda`/`cpu` |
| `use_amp` | `False` | fp16（cuda 生效）|
| `compile` | `False` | torch.compile |
| `tf32` | `False` | CUDA TF32 matmul（V100/Amp 上 fp32 约 2~4×）|
| `channels_last` | `True`(cuda) | conv 走 NHWC 内存布局 |
| `attention_mode` / `attn_mode` / `attn_window` 等 | 同训练 | 必须与训练时一致，否则权重形状不匹配 |

关键方法：
- `predict(board, my_hist, op_hist, to_play)` → `(policy(B,A), value(B,1))` 单样本。
- `predict_batch(states)` → 批量；`states` 为 `[(board, my_hist, op_hist, to_play)]`，**或** `[(None, my_hist, op_hist, to_play, planes)]`（第 5 项预计算 12 通道 planes，省去重复 `feature_planes`）。
- `choose_move(...)` → 纯策略 argmax（无搜索）。
- `choose_move_mcts(..., simulations, num_threads, use_rollout, rollout_lambda)` → MCTS 选点。
- `self_play(...)` / `play_against_human(...)` → 对弈。

CLI：

```bash
python -m src.inference --help
```

常用模式：

```bash
# 自对弈（MCTS）
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --device cuda --use-amp --compile --tf32 \
    --attn-mode window --attn-window 7 \
    --mode selfplay --games 10 --use-mcts --simulations 400 --num-threads 4

# 人机对弈（终端输入坐标，如 ce；pass/resign）
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --device cuda --use-amp --compile --tf32 \
    --mode human --human-color 1 --use-mcts --simulations 400

# 开启 LightPLS 轻量 rollout
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --device cuda --use-amp --compile --tf32 \
    --attn-mode window --attn-window 7 \
    --mode selfplay --use-mcts --simulations 800 --num-threads 4 \
    --use-rollout --rollout-lambda 0.25
```

CLI 参数（`--mode` ∈ {selfplay, human, analyze}）：`--model`、`--board-size`、`--device`、`--use-amp`、
`--compile`、`--tf32`、`--attention-mode`、`--num-attention-layers`、`--num-heads`、`--attention-dropout`、
`--attn-mode`、`--attn-window`、`--temperature`、`--topk`、`--games`、`--use-mcts`、`--simulations`、
`--num-threads`、`--use-rollout`、`--rollout-lambda`、`--human-color`。

---

## 8. 评估 `evaluate.py`

```bash
python scripts/evaluate.py --model models/sft_9x9.pth --board-size 9 \
    --mode random --num-games 100 --use-mcts --simulations 200
```

`--mode`：

| 模式 | 说明 |
|------|------|
| `random` | 模型（黑）对随机策略（白），输出胜率 |
| `benchmark` | 推理速度基准：单样本 vs 批量 32 前向吞吐 |
| `selfplay` | 模型自对弈，输出黑方胜率 |

其他参数：`--device`、`--use-amp`、`--compile`、`--attention-*`、`--attn-*`、`--use-mcts`、`--simulations`、`--num-threads`、`--topk`、`--num-games`。

> `evaluate_vs_random` 里模型执黑；`benchmark` 用 32 批量模拟 MCTS 叶子评估吞吐。

---

## 9. 加速手段总览

| 加速项 | 状态 | 说明 / 开启方式 |
|--------|------|----------------|
| 批量叶子评估 | ✅ | `GoAI.predict_batch` 同层拼 batch 一次前向（GPU 吞吐 ×10+）|
| 跨线程合并 batch（生产者-消费者）| ✅ | worker 只选路径，主线程统一 `deepcopy`+`feature_planes`+`predict_batch`，每个叶子只算一次特征 |
| 增量特征 | ✅ | `predict_batch` 接受预计算 planes，省重复计算 |
| TF32 matmul | ✅ | `--tf32`，V100/Amp 上 fp32 约 2~4×，精度损失可忽略 |
| `channels_last` | ✅ | CUDA 上 conv 走 NHWC（默认开）|
| 虚拟损失 + 多线程 | ✅ | `--num-threads 4`，并行选路径利用多核 |
| `torch.compile` | ✅ | `--compile`，GPU 上约 20~40% |
| `--use-amp` fp16 | ✅ | cuda 上默认开（V100 走 fp16，无 bf16）|
| `window`/`axial` 注意力 | ✅ | `--attn-mode window --attn-window 7`，GPU 上约 7× 注意力提速 |
| LightPLS 轻量 rollout | ✅ | `--use-rollout --rollout-lambda`，叶子价值融合 Tromp-Taylor 快数子 |

**V100S 全加速组合（推荐）**：

```bash
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --device cuda --use-amp --compile --tf32 \
    --attention-mode mix --attn-mode window --attn-window 7 \
    --mode selfplay --use-mcts --simulations 400 --num-threads 4
```

---

## 10. 完整工作流示例（云端 V100S）

```bash
# ① 造数据（9 路先小批量跑通）
python scripts/build_dataset.py --src data/games/games/ \
    --out data/sgf_9x9.npz --board-size 9 --max-games 5000

# ② 训练（9 路 CPU 验证 -> 19 路 V100S）
python scripts/train_sft.py --device cpu --board-size 9 \
    --batch-size 128 --epochs 3 --data data/sgf_9x9.npz --out models/sft_9x9.pth

python scripts/train_sft.py --device cuda --use-amp --compile \
    --board-size 19 --batch-size 512 --epochs 5 \
    --attention-mode mix --attn-mode window --attn-window 7 \
    --data data/sgf_19x19.npz --out models/sft_19x19.pth

# ②' 直接喂「装着多个分片 tgz 的文件夹」（无需先 build_dataset 成单个 npz）
#    自动递归扫描 data/games/games/ 下所有 .tgz 解析并合并成一个训练集
python scripts/train_sft.py --device cuda --use-amp --compile \
    --board-size 19 --batch-size 512 --epochs 5 \
    --attention-mode mix --attn-mode window --attn-window 7 \
    --data data/games/games/ --out models/sft_19x19.pth
#    子采样控制内存：每个 tgz 最多 3000 局
# python scripts/train_sft.py --device cuda --board-size 19 --data data/games/games/ \
#     --max-games-per-tgz 3000 --out models/sft_19x19.pth

# ③ 评估棋力信号（对随机策略胜率）
python scripts/evaluate.py --model models/sft_9x9.pth --board-size 9 \
    --mode random --use-mcts --simulations 200

# ④ 推理 / 对弈（加 LightPLS 提棋力）
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --device cuda --use-amp --compile --tf32 \
    --attn-mode window --attn-window 7 \
    --mode selfplay --use-mcts --simulations 800 --num-threads 4 \
    --use-rollout --rollout-lambda 0.25
```

> 数据规模提醒：19 路全量 36648 局可能很大（npz 内存/磁盘），先 `--max-games 30000` 试产出体积。
> 提升棋力上限需「MCTS 自我对弈 + 策略迭代」（AlphaGo 式），当前 `self_play` 只产出胜率、不存数据；
> 若要真正变强，可加「selfplay 产出 npz → 再训练」迭代脚本（待办，见 §13）。

---

## 11. API 速查

```python
from src.inference import GoAI
from src.search.mcts import MCTS
from src.game.go_rules import GoBoard
from src.search.light_rollout import FastPolicy, light_rollout

# 推理
ai = GoAI(board_size=9, device="cpu", compile=False)        # model_path=None 即随机权重
board = GoBoard(9)
h = [[-1, -1, -3], [-1, -1, -3]]                            # 己方/对手 各 3 手历史
to_play = 1                                                 # 1 黑 / -1 白
policy, value = ai.predict(board, h[0], h[1], to_play)      # 单样本
states = [(board, list(h[0]), list(h[1]), to_play)] * 32
pol_batch, val_batch = ai.predict_batch(states)             # 批量

# MCTS
mcts = MCTS(ai, board_size=9, num_threads=4, temperature=0.0,
            use_rollout=True, rollout_lambda=0.25)
move_int, is_pass, root_value = mcts.best_move(
    board, h[0], h[1], to_play, simulations=400, return_value=True)

# LightPLS
fp = FastPolicy(9)
v = light_rollout(board, fp, max_steps=60, rng=np.random.default_rng(0))   # 发起方视角 [-1,1]
```

---

## 12. 常见问题与排错

| 现象 | 原因 / 解决 |
|------|------------|
| `ImportError: src.xxx` | 必须在**仓库根目录**运行（`sys.path` 以根为基准）；`cd f:/AI/Go-AI` 后执行 |
| 权重加载形状不匹配 | `attention_mode`/`attn_mode`/`board_size` 必须与训练时一致；`action_size=n*n+1` |
| `torch.compile` 首次很慢 | 正常（编译开销）；CPU 也支持，需 MSVC（`vcvars64.bat`/`cl.exe`）|
| `play()` 返回 `False` | 着法非法（落子撞禁着/自杀/劫）；调用方需判断返回值，不要假设成功 |
| `board` 无 `to_play` | 引擎用 `current_player`（1/-1），MCTS 内部 `to_play`（1/2）仅用于特征构造，二者不等价 |
| 数据 `有效/跳过` 比例低 | 棋谱 `RE` 缺失或为和棋、坐标越界；检查 `--board-size` 是否匹配；先 `--max-games 100` 看统计 |
| npz 太大 / 内存不足 | 用 `--chunk-size`（默认 50000）流式分片落盘，峰值内存仅约一个 chunk；或减小 `--max-games` 分批生成多个 npz 再用 `train_sft.py --data <目录>` 合并训练 |
| V100 上 bf16 报错 | V100 是 Volta，**无 bf16**，已默认走 fp16（`--use-amp`）|
| MCTS 选到非法着法 | 极端情况下回退到 `choose_move`（纯策略 argmax）；见 `choose_move_mcts` |

---

## 13. 已知死代码 / 待办

- ⚠️ **`scripts/demo.py` 是死代码**：调用 `GoAI(search_depth=...)`、`ai.search`、`ai.get_move()`、
  `ai.suggest_moves()`、`ai.make_move()`、`ai.evaluate_position()` 等**旧 API**，与当前 `inference.py`
  （`choose_move` / `choose_move_mcts` / `predict` / `predict_batch`）不匹配。**不要运行**，也不要据此理解接口。
- 已删除旧 19 通道死代码：`src/networks/resnet.py`、`src/search/minimax.py`、
  `src/evaluation/evaluator.py`、`src/evaluation/alpha_evaluator.py`、`src/evaluation/__init__.py`、
  `src/config/config.py`、`src/config/__init__.py`（与 12 通道 SFT 冲突）。
- 待办：去 `deepcopy` 用增量 `play`（需给 `GoBoard` 加 `undo` 接口）；
  「selfplay 产出 npz → 再训练」迭代脚本（策略提升级路径）。

---

## 许可 / 参考

监督学习路线参考 AlphaGoZero 的 12 通道特征与策略-价值网络设计；规则引擎为自建中国规则基础实现。
```

