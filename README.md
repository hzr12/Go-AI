# Go-AI：监督学习围棋 AI（策略-价值网络 + MCTS + LightPLS）

> 一个从 SGF 棋谱做监督学习（SFT）的围棋 AI。主线为 **AlphaGoZero 风格的 12 通道策略-价值网络**，
> 推理阶段用 **MCTS（PUCT + 批量叶子评估 + 虚拟损失多线程）** 选点，并可叠加 **LightPLS 轻量 rollout**
> 提升无强 RL 时的棋力。规则引擎为自建 `GoBoard`（Tromp-Taylor 数子、中国规则基础）。
> 支持 **ONNX Runtime** 推理后端（含 int8 动态量化），NPU/CPU 场景下显著加速。

> **状态说明**：当前为**纯监督学习**路线（从人类棋谱学着法），不是 AlphaGo/MuZero 的自我对弈 RL。
> 已删除与 12 通道 SFT 冲突的旧 19 通道死代码（`resnet.py` / `minimax.py` / `evaluator.py` / `alpha_evaluator.py` / `config.py`）。

---

## 目录

- [1. 环境要求](#1-环境要求)
- [2. 安装](#2-安装)
- [3. 项目结构](#3-项目结构)
- [4. 核心概念](#4-核心概念)
  - [4.1 棋盘与规则引擎 GoBoard](#41-棋盘与规则引擎-goboard)
  - [4.2 特征平面（12 通道）](#42-特征平面12-通道)
  - [4.3 网络架构 AlphaGoNet](#43-网络架构-alphagonet)
  - [4.4 MCTS 搜索](#44-mcts-搜索)
  - [4.5 LightPLS 轻量 rollout](#45-lightpls-轻量-rollout)
- [5. 数据准备](#5-数据准备)
- [6. 训练](#6-训练)
- [7. 推理与对弈](#7-推理与对弈)
- [8. 评估](#8-评估)
- [9. 加速手段总览](#9-加速手段总览)
- [10. 自对弈训练（AlphaZero）](#10-自对弈训练alphazero)
- [11. 完整工作流示例（4 卡 NPU 910A）](#11-完整工作流示例4-卡-npu-910a)
- [12. API 速查](#12-api-速查)
- [13. 常见问题与排错](#13-常见问题与排错)

---

## 1. 环境要求

| 组件 | 最低 | 推荐（训练/推理） |
|------|------|-------------------|
| Python | 3.8 | 3.10+ |
| PyTorch | ≥ 1.9 | ≥ 2.0（用上 `torch.compile`）|
| 算力 | 任意 CPU | NVIDIA GPU（V100S / Ampere），CUDA 11.8+；Ascend NPU（910B）|
| 磁盘 | 几百 MB | 棋谱数据 + npz（19 路全量可能数十 GB）|
| 内存 | 4 GB | 16 GB+（npz 全量加载到内存）|

依赖仅 `torch` / `numpy` / `pytest`（见 `requirements.txt`）。**无** `tensorflow`、无额外围棋库。
可选：`onnxruntime`（ONNX 推理后端，CPU 推理场景推荐）、`torchao`（GPU INT4 量化）。

```text
torch>=1.9.0
numpy>=1.19.0
pytest>=6.0.0
onnxruntime>=1.17.0    # 可选，ONNX 推理
torchao>=0.1.0         # 可选，GPU weight-only INT4 量化
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
├── run.txt                       # 训练/推理/评估命令（4 卡 NPU 910A 优化版）
├── data/
│   └── games/games/              # SGF 棋谱（36648 个 .sgf，多层目录）
│       └── Aizu/01/1.sgf ...
├── models/                       # 训练产出权重（*.pt / *.pth），默认不存在需自训练
├── src/
│   ├── __main__.py
│   ├── inference.py              # GoAI 推理入口 + CLI（selfplay / human / analyze / ONNX 导出）
│   ├── game/
│   │   └── go_rules.py           # GoBoard 规则引擎 + 12 通道 feature_planes（ThreadPool 并行标注）
│   ├── networks/
│   │   ├── alphanet.py           # AlphaGoNet（策略+价值双头）
│   │   ├── backbone.py           # SharedBackbone（ResBlock + 注意力）
│   │   ├── policy_network.py     # PolicyNetwork 头
│   │   └── value_network.py      # ValueNetwork 头（3 层残差）
│   ├── search/
│   │   ├── mcts.py               # MCTS（批量叶子评估 + 跨叶子批量 leaf_ab + 虚拟损失 + 特征缓存）
│   │   └── light_rollout.py      # FastPolicy + light_rollout（Tromp-Taylor 数子）
│   ├── data/
│   │   ├── dataset.py            # SupervisedDataset（紧凑存储 + 随机对称增广）
│   │   └── sgf_parser.py         # SGFParser（解析棋谱）
│   └── utils/
│       └── helpers.py            # print_board 等
├── scripts/
│   ├── train_sft.py              # 监督学习训练（CE + BCE，支持 DDP / NPU）
│   ├── build_dataset.py          # SGF 目录/tgz -> npz 训练集
│   ├── evaluate.py               # 评估（vs 随机 / 自对弈 / 速度基准）
│   ├── eval_elo.py               # ELO 评分
│   ├── selfplay_train.py         # AlphaZero 自对弈训练（并行生成 + 流式训练 + DDP）
│   ├── webui.py                  # Web UI（Flask，浏览器对弈 + 实时 MCTS 可视化）
│   ├── cli_play.py               # 终端人机对弈
│   ├── fetch_games.py            # 从在线平台抓取棋谱
│   └── bench_train.py            # 训练速度基准
└── tests/
    ├── test_mcts.py              # MCTS + LightPLS 单测
    └── ...                       # 其他单测
```

> 入口统一用 `src.inference.GoAI`，所有脚本通过 `sys.path.insert(0, 仓库根)` 以 `src.xxx` 方式导入。

---

## 4. 核心概念

### 4.1 棋盘与规则引擎 GoBoard

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

### 4.3 网络架构 AlphaGoNet

文件：`src/networks/alphanet.py`

```
输入 (B, 12, H, W)
      │
SharedBackbone(in_ch=12, ch=192, res_blocks=17, 注意力模式)
      │  -> (B, 192, H, W) 共享表征
      ├─ PolicyNetwork(ch=192 -> 64, action_size) -> (B, A) logits
      └─ ValueNetwork(ch=192 -> 32, 3 res_blocks, 1)  -> (B, 1) Tanh[-1,1]（黑方视角胜率）
```

- `forward(x)` → `(policy_logits, value)`；`policy_logits` 在推理时经 `softmax` 得概率。
- `action_size = n*n + 1`，**多出的 1 类是 pass**。
- 注意力（可选，默认 `mix`）：`global`（全配对）/ `window`（滑动窗口，`--attn-window`）/
  `window_global`（窗口 + 全局 token）/ `axial`（轴向）；
  `attention_mode`：`none`（纯卷积）/ `mix`（卷积+注意力混合）/ `all`（全注意力）。
  V100 上推荐 `--attn-mode window --attn-window 7`（注意力约 7× 提速）；
  `window_global` 比 `sparse` 快约 35%，比 `global` 快约 14%，综合最优。
- Value Head：3 层残差块（`res1`, `res2`, `res3`），预训练权重 `res3` 为随机初始化（norm=4.6 vs 其他层=77.4），自对弈训练会自动校准。
- RMSNorm 替代 LayerNorm（手写实现，兼容 PyTorch 2.1）；Policy head 改为 1x1 conv + pass_bias。

### 4.4 MCTS 搜索

文件：`src/search/mcts.py`，类 `MCTS`

- 算法：PUCT（AlphaGoZero 风格），`score = Q + U`，`U = c_puct * prior * sqrt(parent_visits) / (1 + child_visits)`。
- 节点：`MCTSNode(board, my_hist, op_hist, to_play, move_int, prior, visit, value_sum, virtual_loss)`。
- 展开：对叶子的每个合法着法（含 pass），`deepcopy(board)` 走一步，批量拼成 `states` 一次 `GoAI.predict_batch` 得到
  `(policies, values)`，写入子节点先验与价值。
- **加速（生产者-消费者并行）**：`num_threads` 个 worker 线程只负责「选路径」（纯 CPU 估算 PUCT，极廉价，加虚拟损失占位），
  主线程从队列**批量取出叶子**，统一 `deepcopy`+`feature_planes`+`predict_batch`（昂贵部分一次大 batch 前向），
  每个叶子只构造一次特征、只前向一次。`--num-threads 4` 即可。
- **加速（跨叶子批量 leaf_ab）**：`_batch_leaf_ab` 将 N 个叶子的浅层 negamax 合并为 depth+1 次 predict（而非 N×(depth+1) 次），
  CPU/ONNX 场景下 predict 调用降低约 3×。
- **加速（特征平面 LRU 缓存）**：`_planes1` 缓存 16384 个局面的 12 通道特征（key = board bytes + to_play + history + ko），
  带 30 秒 TTL，MCTS 同一叶子深度路径中相同局面可复用，避免重复 flood-fill 计算。
- **spec_prefetch 加速**：仅 `num_threads >= 4` 时启用，worker 线程异步预评估疑似叶子。
- **leaf_ab 智能降级**：`sims < 64` 时自动 `leaf_ab_depth = 1`，避免低模拟数下过度展开。
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
  CUDA/NPU 自动 `pin_memory` + `non_blocking=True`，加速传输。
- 对称增广：每样本随机选 8 种变换之一（旋转 0/90/180/270 × 翻转），同时作用于特征平面与着法坐标（`SYMMETRIES`）。

---

## 6. 训练

### 6.1 监督训练 `train_sft.py`

```bash
python scripts/train_sft.py --data data/sgf_19x19.npz --out models/sft_19x19.pth \
    --device cuda --use-amp 1 --compile 1 \
    --board-size 19 --batch-size 512 --epochs 5 \
    --backbone-channels 192 --backbone-res-blocks 17 \
    --attention-mode mix --attn-mode window --attn-window 7 \
    --value-loss-weight 5.0 --value-lr-mult 2.0
```

**NPU 4 卡训练**（推荐）：

```bash
torchrun --nproc_per_node=4 scripts/train_sft.py \
  --data data/sgf_19x19_full.npz \
  --device npu --board-size 19 \
  --backbone-channels 192 --backbone-res-blocks 17 \
  --res-blocks 8 --convnext-blocks 4 --attn-blocks 5 \
  --value-channels 96 --value-res-blocks 11 \
  --policy-channels 128 --policy-layers 3 \
  --batch-size 3200 --epochs 1 \
  --lr 0.002 --weight-decay 0.0001 \
  --attention-mode mix --num-attention-layers 4 --num-heads 4 \
  --attn-mode window_global --attn-window 5 \
  --attention-dropout 0.1 --label-smoothing 0.1 \
  --gradient-accumulation-steps 1 \
  --use-amp 1 --use-ema 1 \
  --prefetch-workers 16 --prefetch-depth 16 \
  --log-every 50 --eval-every 500 --save-every 500 \
  --early-stop 1 --early-stop-patience 3 \
  --out models/sft_19x19_v17.pth \
  --swanlab 1 --project go-ai --name sft_v17_npu4
```

> **纯 Python 调用**（无需命令行环境）：
> ```bash
> python run.py                    # 默认 SFT 训练
> python run.py sft                # 同上
> python run.py sft -- --epochs 2 --batch-size 64  # 覆盖参数
> python run.py selfplay           # 自对弈训练
> ```
> 详见 `run.py`。

关键参数（**所有布尔开关使用 0/1**）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | 必填 | 单个 `.npz` 或包含多个 `.npz` 的目录 |
| `--out` | `models/sft.pt` | 权重输出路径（父目录自动建）|
| `--device` | `auto` | `cuda` / `npu` / `cpu`；`auto`=有 GPU 用 cuda |
| `--use-amp` | `0` | 启用混合精度训练（0=关闭, 1=开启）|
| `--batch-size` | `512` | 每步批量 |
| `--epochs` | `5` | 训练轮数 |
| `--lr` | `2e-3` | 学习率（AdamW）|
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--backbone-channels` | `128` | 主干通道数（推荐 192，12.4M 参数）|
| `--backbone-res-blocks` | `12` | 主干残差块数（推荐 17）|
| `--res-blocks` | `0` | ResBlock 数量（0=使用默认 mix 模式）|
| `--convnext-blocks` | `0` | ConvNeXtBlock 数量 |
| `--attn-blocks` | `0` | AttentionResBlock 数量 |
| `--value-channels` | `64` | Value head 通道数（推荐 96）|
| `--value-res-blocks` | `3` | Value head 残差块数（推荐 11）|
| `--policy-channels` | `32` | Policy head 通道数（推荐 128）|
| `--policy-layers` | `2` | Policy head 层数（推荐 3）|
| `--use-ema` | `0` | 指数移动平均权重（0=关闭, 1=开启）|
| `--compile` | `0` | `torch.compile` 算子融合（GPU +20~40%）|
| `--flash-attn` | `0` | 启用 flash-attn（A100 最快，需 pip install）|
| `--export-onnx` | `0` | 训练结束后导出 ONNX |
| `--swanlab` | `0` | 启用 SwanLab 实验跟踪 |
| `--early-stop` | `0` | 启用早停机制 |
| `--early-stop-patience` | `3` | 早停耐心值 |
| `--gradient-accumulation-steps` | `1` | 梯度累积步数（等效 batch = batch_size × steps）|
| `--prefetch-depth` | `4` | 预取流水深度（提前造好数据，控制内存/吞吐）|
| `--ver` | `v17` | 模型版本号（用于 swanlab name 和 --out 默认值）|
| `--c2net` | `0` | 启用 C2NET（OpenI 启智平台）支持 |

训练细节：
- 损失：`L = CrossEntropy(policy_logits, move, label_smoothing) + value_loss_weight × BCEWithLogitsLoss(value, z)`
- 优化器：AdamW + `CosineAnnealingLR`；warmup 10%；价值网络头用 `value_lr_mult × base_lr`。
- EMA（`--use-ema 1`）：指数移动平均权重，评估/保存时自动使用 EMA 参数。
- NPU：`torch_npu` + HCCL 后端，fp16 autocast。

### 6.2 自对弈训练 `selfplay_train.py`（AlphaZero 风格）

详见 [§10 自对弈训练](#10-自对弈训练alphazero)。

---

## 7. 推理与对弈

`GoAI` 类（`src/inference.py`）封装模型加载、单/批量前向、MCTS 选点、自对弈、人机对弈，并提供 CLI。

常用模式：

```bash
# 自对弈（MCTS）
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --device cuda --use-amp 1 --compile 1 --tf32 1 \
    --attn-mode window --attn-window 7 \
    --mode selfplay --games 10 --use-mcts 1 --simulations 400 --num-threads 4

# 人机对弈（终端输入坐标，如 ce；pass/resign）
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --device cuda --use-amp 1 --compile 1 --tf32 1 \
    --mode human --human-color 1 --use-mcts 1 --simulations 400

# 开启 LightPLS 轻量 rollout
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --device cuda --use-amp 1 --compile 1 --tf32 1 \
    --attn-mode window --attn-window 7 \
    --mode selfplay --use-mcts 1 --simulations 800 --num-threads 4 \
    --use-rollout 1 --rollout-lambda 0.25

# ONNX 导出
python src/inference.py --model models/sft_19x19.pth --board-size 19 \
    --mode analyze --onnx models/sft_19x19.onnx
```

---

## 8. 评估

```bash
python scripts/evaluate.py --model models/sft_19x19.pth --board-size 19 \
    --mode random --num-games 100 --use-mcts 1 --simulations 400
```

`--mode`：

| 模式 | 说明 |
|------|------|
| `random` | 模型（黑）对随机策略（白），输出胜率 |
| `benchmark` | 推理速度基准：单样本 vs 批量 32 前向吞吐 |
| `selfplay` | 模型自对弈，输出黑方胜率 |

---

## 9. 加速手段总览

| 加速项 | 状态 | 说明 / 开启方式 |
|--------|------|----------------|
| 批量叶子评估 | ✅ | `GoAI.predict_batch` 同层拼 batch 一次前向（GPU 吞吐 ×10+）|
| 跨线程合并 batch（生产者-消费者）| ✅ | worker 只选路径，主线程统一 `deepcopy`+`feature_planes`+`predict_batch` |
| 增量特征 | ✅ | `predict_batch` 接受预计算 planes，省重复计算 |
| **跨叶子批量 leaf_ab** | ✅ | N 个叶子浅层 negamax 合并为 depth+1 次 predict |
| **特征平面 LRU 缓存** | ✅ | 16384 局面缓存（19 路），30 秒 TTL |
| **spec_prefetch** | ✅ | `num_threads >= 4` 时自动启用，worker 异步预评估 |
| **leaf_ab 智能降级** | ✅ | `sims < 64` 时自动 `depth=1`，避免低模拟数过度展开 |
| ONNX Runtime | ✅ | `--onnx models/xxx.onnx`，含 int8 动态量化 |
| TF32 matmul | ✅ | `--tf32`，V100/Amp 上 fp32 约 2~4× |
| `channels_last` | ✅ | CUDA 上 conv 走 NHWC（默认开）|
| 虚拟损失 + 多线程 | ✅ | `--num-threads 4`，并行选路径 |
| `torch.compile` | ✅ | `--compile 1`，GPU 上约 20~40% |
| `--use-amp` fp16 | ✅ | cuda/npu 上 `--use-amp 1` |
| `window_global` 注意力 | ✅ | `--attn-mode window_global --attn-window 7` |
| LightPLS rollout | ✅ | `--use-rollout 1 --rollout-lambda` |
| pin_memory + non_blocking | ✅ | CUDA/NPU 数据传输自动重叠 |
| GPU weight-only INT4 | ✅ | `ai.quantize_int4_torchao()` |
| **并行自对弈** | ✅ | `--parallel-games N`，多进程并行生成 |
| **流式训练** | ✅ | `--streaming`，不写临时文件，内存队列传递 |
| **多 GPU DDP** | ✅ | `--ddp` + `torchrun --nproc_per_node=N` |

---

## 10. 自对弈训练（AlphaZero）

`scripts/selfplay_train.py`：纯自我对弈生成数据 → 训练 → 循环（无外部棋谱）。

### 流程

```
自对弈 N 局 → 8 对称增强 → replay buffer → 训练 → 保存最佳权重 → 下一轮
```

- **数据标签**：MCTS visit 分布 → policy 标签；终局胜负 → value 标签
- **探索**：根先验混 Dirichlet 噪声 + 温度采样
- **价值软化**：按手数位置 tanh 软化（开局弱信号、终局强信号），z = tanh(z_raw × α)，α∈[0.3, 1.0]
- **TD 学习（C+E 混合，`--td 1` 默认开启）**：v_td = sign·root_values[t+td_steps]（sign 按执子方奇偶，越界回退终局值），z = clip(α_td·v_td + (1-α_td)·r_soft, -1, 1)，α_td 从 `--td-alpha-init`（0.2）线性升到 `--td-alpha-end`（0.9）；`--td 0` 时逐位退回旧 tanh 软化（回归保证）

### 50GB 磁盘约束优化

- **流式处理**：自对弈数据不写临时 npz，用共享内存队列传递
- **只保存最佳权重**：只保留 `latest` + `best` 两个文件
- **紧凑 buffer**：内存中最多保留 500 局数据（~80MB）

### 快速开始

```bash
# CPU 9 路验证
python scripts/selfplay_train.py --board-size 9 --iters 5 --games 4 --sims 64

# NPU 19 路正式训练（4 进程并行 + 流式 + rollout）
python scripts/selfplay_train.py \
  --board-size 19 --iters 20 --games 16 --sims 400 \
  --parallel-games 4 --streaming 1 \
  --model models/sft_19x19_v17.pth \
  --out models/az_best.pth \
  --use-rollout 1 --leaf-ab-depth 2 --num-threads 8 \
  --c-puct 2.0 --virtual-loss 8.0

# NPU 4 卡 DDP 训练
torchrun --nproc_per_node=4 scripts/selfplay_train.py \
  --board-size 19 --iters 20 --games 16 --sims 400 \
  --parallel-games 4 --ddp 1 --streaming 1 \
  --model models/sft_19x19_v17.pth --out models/az_best.pth
```

关键参数（**所有布尔开关使用 0/1**）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--sims` | `400` | 每步 MCTS 模拟数（越高越强，越慢）|
| `--parallel-games` | `1` | 并行自对弈局数（多进程，推荐 4-8）|
| `--c-puct` | `2.0` | PUCT 探索系数 |
| `--num-threads` | `8` | MCTS 多线程数 |
| `--use-rollout` | `0` | LightPLS rollout 价值融合（0=关闭, 1=开启）|
| `--leaf-ab-depth` | `2` | 叶内 α-β 搜索深度 |
| `--buffer-size` | `500` | replay buffer 容量（局数）|
| `--streaming` | `0` | 流式训练（不写临时文件，0=关闭, 1=开启）|
| `--ddp` | `0` | 多 GPU 数据并行（torchrun 启动，0=关闭, 1=开启）|
| `--temperature` | `1.0` | 自对弈采样温度 |
| `--use-ema` | `0` | EMA 权重（0=关闭, 1=开启）|
| `--async-pipeline` | `0` | 异步流水线（生成与训练并行，0=关闭, 1=开启）|
| `--swanlab` | `0` | 启用 SwanLab 实验跟踪 |
| `--c2net` | `0` | 启用 C2NET（OpenI 启智平台）支持 |
| `--ver` | `rl` | 模型版本号（SwanLab name 后缀）|
| `--td` | `1` | TD (C+E) 价值标签（0=旧 tanh 软化, 1=开启）|
| `--td-steps` | `3` | TD n-step 前看步数（数据下标空间）|
| `--td-alpha-init` | `0.2` | TD α 调度初值（开局偏 r_soft）|
| `--td-alpha-end` | `0.9` | TD α 调度终值（残局偏 v_td）|
| `--batch-size` | `256` | 训练 batch（NPU 甜点）|
| `--spec-prefetch` | `1` | worker 推测预评估（0=关闭, 1=开启）|

---

## 11. 完整工作流示例（4 卡 NPU 910A）

详见 `run.txt`。

```bash
# ① 构建数据集
python scripts/build_dataset.py \
  --src data/games/games/ --out data/sgf_19x19_full.npz \
  --board-size 19 --chunk-size 50000 --tmp-dir data/tmp

# ② 监督训练（4 卡 NPU）
torchrun --nproc_per_node=4 scripts/train_sft.py \
  --data data/sgf_19x19_full.npz --device npu --board-size 19 \
  --backbone-channels 192 --backbone-res-blocks 17 \
  --res-blocks 8 --convnext-blocks 4 --attn-blocks 5 \
  --value-channels 96 --value-res-blocks 11 \
  --policy-channels 128 --policy-layers 3 \
  --batch-size 3200 --epochs 1 --use-amp 1 --use-ema 1 \
  --early-stop 1 --out models/sft_19x19_v17.pth

# ③ 自对弈训练（AlphaZero）
python scripts/selfplay_train.py \
  --board-size 19 --iters 20 --sims 48 \
  --parallel-games 8 --streaming 1 \
  --model models/sft_19x19_v17.pth --out models/az_best.pth

# ④ 评估
python scripts/evaluate.py --model models/sft_19x19_v17.pth \
  --board-size 19 --mode random --num-games 100 --use-mcts 1

# ⑤ WebUI
python scripts/webui.py --port 7860 --device cpu --priors-leaf 1 \
    --mode hybrid --hybrid-sims 32 --hybrid-blend 0.5 \
    --expand-topk 16 --num-threads 8
```

---

## 12. API 速查

```python
from src.inference import GoAI
from src.search.mcts import MCTS
from src.game.go_rules import GoBoard
from src.search.light_rollout import FastPolicy, light_rollout

# 推理
ai = GoAI(model_path="models/sft_19x19_v17.pth", board_size=19, device="npu")
board = GoBoard(19)
h = [-1, -1, -1]
oh = [-1, -1, -1]
to_play = 1
policy, value = ai.predict(board, h, oh, to_play)            # 单样本
states = [(board, list(h), list(oh), to_play)] * 32
pol_batch, val_batch = ai.predict_batch(states)              # 批量

# MCTS
mcts = MCTS(ai, board_size=19, num_threads=4, temperature=0.0,
            use_rollout=True, rollout_lambda=0.25)
move_int, is_pass, root_value = mcts.best_move(
    board, h, oh, to_play, simulations=400, return_value=True)

# ONNX 导出
ai.export_onnx("model.onnx", board_size=19, quantize_int8=False)

# LightPLS
fp = FastPolicy(19)
v = light_rollout(board, fp, max_steps=60, rng=np.random.default_rng(0))
```

---

## 13. 纯 Python 调用（`run.py`）

无需命令行环境，直接通过 Python 调用训练脚本：

```python
# 方式一：命令行风格
import subprocess
subprocess.run(["python", "run.py", "sft", "--", "--epochs", "2"])

# 方式二：直接 import（适合测试/Jupyter）
import sys
sys.argv = ["train_sft.py", "--data", "data/test.npz", "--epochs", "1", "--device", "cpu"]
from scripts.train_sft import main
main()
```

```bash
python run.py                    # 默认 SFT 训练
python run.py sft                # 同上
python run.py sft -- --epochs 2  # 覆盖参数
python run.py selfplay           # 自对弈训练
```

---

## 14. 常见问题与排错

| 现象 | 原因 / 解决 |
|------|------------|
| `ImportError: src.xxx` | 必须在**仓库根目录**运行（`sys.path` 以根为基准）|
| 权重加载形状不匹配 | `attention_mode`/`attn_mode`/`board_size` 必须与训练时一致 |
| `play()` 返回 `False` | 着法非法（落子撞禁着/自杀/劫）|
| MCTS 选到非法着法 | 极端情况下回退到 `choose_move`（纯策略 argmax）|
| V100 上 bf16 报错 | V100 无 bf16，已默认走 fp16 |
| ONNX 导出 ShapeInferenceError | `export_onnx` 已自动回退 legacy exporter |
| 自对弈 res3 未加载 | 预训练 v12 无 `value.res3`，`strict=False` 自动跳过，随机初始化 |
| NPU 训练报错 | 确认 `torch_npu` 已安装，`torch.npu.is_available()` 返回 `True` |

---

## 许可 / 参考

监督学习路线参考 AlphaGoZero 的 12 通道特征与策略-价值网络设计；规则引擎为自建中国规则基础实现。
