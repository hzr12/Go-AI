# Go-AI —— 对标 KataGo 的 19 路围棋 AI

从 16 万局职业 SGF 语料出发，重建成**官方 KataGo V7 语义**的 19×19 围棋 AI：
输入 = 22 空间通道 + 19 全局特征，主干 = `nbt`(nested bottleneck) + transformer，
四个头（policy / value / ownership / scorebelief），12 项可构造监督目标。

> **当前是「重建管线」阶段，不是「训出强棋」阶段。** 模型、特征、标签三条流都已
> 落地或有实测结论，**训练侧已接到 22 通道**：`scripts/train_sft.py --v7 1` 按数据
> 布局分派到 `V7Dataset`（board 级语料，训练时实时算 22 平面）或 `V7PackedDataset`
> （`stdata_v7` 分片，预算好的 22 平面），两者喂给**同一个** `NbtTfNet`
> （22 通道、5,562,121 参数）⇒ **A→B→C 三段共用一个模型**、权重逐段承接。
> 进度见 [§4](#4-当前进度)。

---

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 架构：数据 → 特征 → 模型 → 训练 → 推理](#2-架构数据--特征--模型--训练--推理)
- [3. 三条流与三个训练段](#3-三条流与三个训练段)
- [4. 当前进度](#4-当前进度)
- [5. 会静默出错的 10 个坑](#5-会静默出错的-10-个坑)
- [6. 关键决策与理由](#6-关键决策与理由)
- [7. 已知限制](#7-已知限制)

---

## 1. 快速开始

**所有可跑命令在 [`run.txt`](run.txt)**（一屏：环境 → 数据 → 打标签 → 训练 → 推理）。
本文件不复述命令，只讲**为什么这样设计**、**踩过什么坑**、**哪里会静默出错**。

设计文档（**注意：`docs/` 已被 `.gitignore` 排除，不在版本控制内**）：

| 文件 | 内容 |
|---|---|
| `docs/superpowers/specs/2026-10-01-katago-nbt-tf-design.md` | 模型 / 22 通道 / 12 项 loss 设计 + §9 实测附录 |
| `docs/superpowers/specs/2026-10-02-goai-training-pipeline-design.md` | 三段训练 + A/B/C/D 工作组 |
| `docs/superpowers/specs/2026-10-01-ddp-instead-of-fsdp-design.md` | 为什么 4 卡用 DDP 而不是 FSDP |

---

## 2. 架构：数据 → 特征 → 模型 → 训练 → 推理

### 2.1 数据层

```
data/games/games/{agz,foxpro,leela_zero,yenw_pro}/   133,604 个 SGF
data/*.tgz            (5 个)                         36,274 个 SGF
                                    └── 零重叠，合计 169,878，完整覆盖 162,298 局
data/sgf_19x19_full.npz   34,202,713 行 / 162,298 局 / 10 列（压缩 447 MB，解压 12.3 GB）
data/labels/games.npz      局级 sidecar（4 键，约 1.2 MB，⚠️ 待生成）
data/labels/kata_labels.npz  KataGo 访问分布软标签（157.5 MB，✅ 已生成）
data/labels/soft_index.npz   软标签 → 主数据集行号的 join（160.5 MB，✅ 已生成）
katago/stdata/*.tgz         官方分布式训练数据，4.67M 行（19×19 可用 ≈3.10M）
```

主数据集的 10 列：`boards / my_hist / op_hist / ko / to_play / moves / winrates /
game_ids / game_weights` 等。V7 需要的 13 项里**11 项**能从这 10 列推出或实时算，
只有 6 个局级标量（`KM` / `RE` / `RU` / 是否认输）需要 sidecar。

🔴 **三条硬约束**：

1. **`data/sgf_19x19_full.npz` 绝对不可 rebuild。** 重建要重放 169,878 个 SGF，
   而 `build_dataset.py:314-316` 有 **id 复用 bug**（`board.play()` 失败时
   `return 0,1` 但已追加的行留在 `cur`、`game_id_counter` 未自增）——
   `game_ids` 本身对不上局，所有 join 链一起废。
2. **它是 12.3 GB，不要整列读。** `np.load('x.npz')['boards']` 会把整个成员
   解压进内存（本机 13.9 GB ⇒ 任何分块读都先吃掉 12.3 GB，余量 1.6 GB）。
   必须经 `kata_label_join.materialize_dataset()` 落成 `.npy` 后
   `mmap_mode='r'`（落盘 152 s，实测 478K 行/s）。
3. **`game_ids` 是置换不是升序**，同一个 id 可能出现在不相邻的行区间。

### 2.2 特征层 —— 22 空间 + 19 全局

**训练时由预取器在 CPU 上实时算，不落盘。** 端口函数在 `src/data/feature_v7.py`
与 `src/data/feature_v7_ladders.py`（**刻意不放 `go_rules.py`**，避免污染
`feature_planes` / `feature_planes_batched` 所在的旧 17 通道实现）。

空间 22 通道：`on-board / pla / opp / 1-2-3 气 / ko-ban / 历史 5 手 / 梯子
(ch14–17) / 当前区域(ch18,19)`。全局 19 通道：历史 pass 标志、`currentSelfKomi/20`、
ko 规则、计分制度、tax、encore、`passWouldEndPhase`、komi 奇偶三角波。

**8 路恒 0 通道（保留不裁剪）**：空间 `7, 8, 20, 21` + 全局 `12, 13, 15, 16`。
保留理由是严格复刻官方张量布局，未来接官方 checkpoint 零改形状；裁剪只省 0.3% 参数。

**性能闸门**：训练侧（GPU）需 ~2635 行/s，`--prefetch-workers 8` 下每行预算 **3.04 ms**，
现有 12 通道单盘基线 1.78 ms ⇒ **1.7× 余量**。贵的是 `iterLadders`（3 块盘面）与
`calculateArea`（1 块）。

### 2.3 模型层

**A→B→C 三段共用同一个模型**（`--v7 1` ⇒ `NbtTfNet`，22 通道，5,562,121 参数）。
`KATAGO_SE_CFG` 那套 12 通道结构（9,112,005 参数）**只在不带 `--v7` 时**才建，
不再是训练主路。

| | 12 通道（`KATAGO_SE_CFG`，现役默认） | **V7（`NbtTfNet`，A/B/C 共用）** |
|---|---|---|
| 输入通道 | 12 | 22 空间 + 19 全局 |
| 主干 | 240 宽 · 13×SEBottleneck + 4×Attention | `C=256 M=128 H=4 F=384 B=11` nbt2 块 |
| 归一化 | — | `fson` 固定方差标量 + `rsnh` 末端 RMSNorm，**全网无 BN** |
| 头 | policy + value | policy(K=2) + value(3 分类 + scoremean/stdev/lead) + ownership + scorebelief(842 桶) |
| 参数量 | 9,112,005 | **5,562,121**（硬预算 5,850,000，余量 4.9%） |
| 代码 | `scripts/train_sft.py:444` `KATAGO_SE_CFG` | `src/networks/katago_v7.py` |

`NbtTfNet` 已实现并通过预算测试（`tests/test_katago_v7_budget.py` 断言精确值
5,562,121），`src/inference.py:177` 已 `register_in_channels_builder(22, ...)`。

**一份架构、两种数据源**（`load_from_path(--v7 1)` 按**数据布局**分派）：

| 数据 | 数据集类 | 22 平面从哪来 |
|---|---|---|
| `data/sgf_19x19_full.npz`（A/B 段） | `V7Dataset` | 训练时实时算（`feature_v7.spatial_channels_v7`） |
| `data/stdata_v7/`（C 段） | `V7PackedDataset` | 转换时算好并位打包（`spatial_packed`） |

判据是「npz 里有没有 `spatial_packed`」，不是文件名 —— 判错的后果很隐蔽：
A/B 段会静默退回 12 通道路径，V7 的 22 个平面**根本没参与训练**，
loss 照降、指标照报，而模型学的是另一个结构。
`tests/test_v7_single_model.py` 钉住分派结果、22 通道输出、以及
A→B→C 的 `load_state_dict` 承接（strict 零缺失）。

⚠️ eval / early-stop / best-model 走 `evaluate_metrics_v7`（与 12 通道版同名同义）；
`--export-onnx` 对 V7 **无效**（走 `GoAI` 的 12 通道推理链），导出见 [§2.5](#25-导出到-kataGo-引擎)。

### 2.4 训练层

- **段 A/B/C SFT**：`scripts/train_sft.py --v7 1`，三段同一个 22 通道模型、
  `--model` 逐段承接（DDP + NCCL + BF16；无 BF16 的旧卡如 V100 / sm_70 才走 fp16+GradScaler）。
  ⚠️ shell/ 里的 `train_sft_npu_4card_katago_se.sh` 是**NPU 专用**的旧入口
  （不带 `--v7` 的 12 通道），非 NPU 环境别拿它跑；`run.txt` 里给的是直连 `torchrun` 命令。
- **⚙️ 调优开关一律是 CLI 参数，没有环境变量**（2026-08 起陆续登记；`GOAI_PROFILE`
  与 `GOAI_PROFILE_STEPS` 是仅有的例外——两个都是诊断开关，与训练配置无关，语义是
  「从第 N 步抓 M 步 kernel 表」，M 默认 50，`M=0` = 起点即终点、在起点后的第一个
  打点步出表）：
  - `--use-sdpa 0/1`（默认 1）：注意力走融合 SDPA（CUDA SDPA / FlashAttn）；0 = 全后端手写 math。
  - `--attn-query-chunk 0|64`（默认 64）/ `--attn-chunk-ckpt 0/1`（默认 1）：手写
    math 注意力的分块与逐块检查点（只影响 math 路径，SDPA 融合路径不生效）。
  这几个都是**只慢不坏**的总闸（数值口径不变），出问题置 0 即回到已验证配置。
- **V7 tracer bullet**：`scripts/smoke_train_v7.py` —— 302 行 / batch 8 / 40 步，
  直接吃 stdata，用来回答「12 项 loss 每项到底降不降」。

### 2.5 导出到 KataGo 引擎

`.bin.gz` 里存的不只是权重，还带一份**结构描述**，引擎完全按它去读每个数组。
翻译逻辑在 `src/data/katago_export.py`，CLI 做包装 + **强制读回自检**：

```bash
python scripts/export_katago_bin.py export --checkpoint models/ours.pth \
    --out tmp/coding/ours.bin.gz
# 用引擎验证。注意仓库里**没有** analysis.cfg —— 现成的是
#   analysis_batch.cfg / analysis_example.cfg / default_gtp.cfg
# 且 logDir 是**相对引擎 CWD** 的，父目录不存在时引擎会直接
#   Uncaught exception: Error creating directory 而退出。
katago/katago-v1.18.1-opencl-windows-x64/katago.exe analysis \
    -model tmp/coding/ours.bin.gz -config analysis_batch.cfg
```

**两条引擎验收已自动化**（2026-10-05 用真实 250 步权重
`tmp/coding/a_v7_npu4.pth` 复核通过）：

```bash
python tmp/coding/smoke_katago_analysis.py   # run.txt 的验收口径
python tmp/coding/smoke_katago_gtp.py        # 完整 GTP 命令序列
```

`analysis` 实测：`Model version 17` / `nbt transformer, 5545737 params`
（与期望逐位相符），在**真实 OpenCL 设备**上推理（AMD `gfx90c`），非 CPU 回落。

`GTP` 实测 **19/19 通过**：`protocol_version` / `name` / `version`（模型名往返
正确）/ `list_commands` / `boardsize` / `clear_board` / `komi` / `showboard`（19×19
渲染正确）/ `play` / **`genmove` 3/3 全部合法着点** / `undo` / `set_position` /
`final_score` / `quit`。

⚠ 写 GTP 冒烟脚本时踩过一个**给出全绿假象**的坑：GTP 响应是「若干行 + 空行终止」，
只读一行会让后续每条命令**错位**（`version` 读到 `list_commands` 的内容），而汇总
仍打「成功 19 / 失败 0」—— 因为错位拿到的也全是 `=` 开头。**务必读到空行。**
另外 KataGo **没有** `place_free`，自由落子要用 `set_position`。

导出 **5,545,737** 参数（V7 的 5,562,121 减去四个无对应物的自研头）。已实测：
引擎报 `Model version 17` / `Model name: goai_v7 (nbt transformer, 5545737 params)`，
`rootInfo` 里 sv3 六通道全部正确读出并后处理，GTP `genmove` 正常走子。

**`modelVersion` 取 17**（官方当前最新）。**不能取 16** —— 16 给 policy head 加了
Q 值通道，`p2Conv.outChannels` 被钉成 4，加载时直接失败：

```
Uncaught exception: Error loading or parsing model file:
model.policy_head: p2Conv.outChannels (2) != 4
```

我们没有 Q 值目标（`policy_outputs=2` 是 π 与 π_opp），17（= 16 去掉 Q）才匹配。

⚠ 15 与 17 的输出差异**不是语义差异** —— `nneval.cpp` 里搜不到 `modelVersion >= 17`
的后处理分支，block kind 也没有版本门控。真正原因是两份 OpenCL tuning 选了不同的
kernel tiling（`ATTN_BLOCK_Q` 256 vs 128、`CHANNELSTRIDE` 2 vs 1），fp16 累加顺序
不同，在**随机权重**下被放大。

⚠ 附带发现（**与我们的导出无关**）：在这台 AMD iGPU 上，v17 路径**逐次运行结果不同**
（v15 路径完全确定）：

| 模型 | 4 次运行的 `rawLead` |
|---|---|
| 我们的 v17 导出 | 0.597 / 0.590 / 0.548 / 0.565 |
| **官方 b10c384（v17）** | 5.548 / 5.668 / 5.953 |

官方模型表现一致 ⇒ 这是 v17 kernel 路径 + 该驱动 fp16 的性质，不是导出缺陷。
影响面仅限自对弈数据引入 fp16 噪声，不影响训练正确性。

三处被吸收的结构差异（详见 `katago_export.py` 模块 docstring）：

1. **trunk 宽度不同** —— 我们 C=256/11 块，官方 b10c384 是 C=384/10 块，描述按我们自己的 cfg 生成。
2. **attn/ffn 堆叠粒度不同** —— 官方 `BlockStack` 是 attn/ffn **交替**堆叠，
   我们把两者**融合**在一个 `TransformerBlock` 里 ⇒ 一个融合单元展开成官方两个 block。
3. **四个自研头**（`scoring` / `futurepos` / `seki` / `scorebelief`）在官方 `.bin`
   里没有对应字段，剥离而非硬塞。

⚠ `sv3Mul` 六通道里后两路（`shorttermWinlossError` / `shorttermScoreError`）
**没有训练标签**（已核对 `trainingwrite.cpp` 全部 `rowGlobal[n]=` 赋值，官方 stdata
的 64/80 列布局里不存在这两列），导出后恒为常量；第 4 路 `varTimeLeft` 是真训练的。
- **RL**：`scripts/selfplay_train.py`（PPO + lookahead，**MCTS 已在 2026-09-30 归档**，
  采集换成 N 步 minimax 推演）。

### 2.6 推理层

`GoAI`（`src/inference.py`）按权重 stem 的形状读 `in_channels`，
再查构建器注册表：`22 → NbtTfNet`、`12 → AlphaGoNet`。
**读不到可识别的 stem 键直接 `RuntimeError`，不静默回退 12**
（否则会拿随机 12ch 模型装一份陌生架构的权重）。
搜索侧 MCTS（PUCT + 批量叶子评估 + 虚拟损失）保留给 `webui` / `cli_play` /
`evaluate` / `eval_elo`；`feature_planes` 取通道数时向所挂模型要，缺失即报错。

---

## 3. 三条流与三个训练段

### 3.1 三条实现流

| 流 | 内容 |
|---|---|
| **软标签流** | `permute_soft` / `soft_cross_entropy`（掩码 `0` 的行贡献恰好 0）/ `labels=True` 的 4 元组 + dict 契约 / `games.npz` sidecar 生成器（ply-20 锚点）|
| **特征流** | `src/data/feature_v7.py`（气桶 / 历史 5 手 / `calculateArea`）+ `feature_v7_ladders.py`（ch14–17）+ `feature_v7_gather.py`（邻行 gather）**全部完成**，ch0–6/ch8/ch14/ch17 对官方 stdata 逐位 1.000000 |
| **模型流** | `katago_v7.py` + `katago_v7_loss.py` + 22ch builder 已闭环，`train_sft.py --v7 1` 已接：**A/B/C 共用同一个模型**（见 [§2.3](#23-模型层)） |

### 3.2 三个训练段

**三段共用一个模型**（`--v7 1` ⇒ 22 通道 `NbtTfNet`，5,562,121 参数），
只换数据、不换架构，权重用 `--model` 逐段承接：

| 段 | 数据 | 量 | policy 目标 | 目的 |
|---|---|---:|---|---|
| **A** | 自有 34.2M 语料（SGF 派生标签） | 34,202,713 | 人类着法 one-hot | 通路基线；学会读 22 通道 |
| **B** | 自有语料的 1%，用 KataGo 标注 | ~342,000 | 搜索分布（软 CE） | 域匹配桥梁（人类局面 + 搜索答案） |
| **C** | `katago/stdata`（19×19 部分） | ≈3,100,000 | 搜索分布 | 棋力主体；**唯一学全 value 的段** |

B 与 C 的 policy 目标同为搜索分布，但**价值标签不同**：C 段才有真正的
ownership / scorebelief / varTimeLeft。B 段的软标签要 `--soft-index`；
**C 段的软标签内建在数据行里**（`policy_player_prob` / `policy_opp_prob`，
转换时归一化），不需要 `--soft-index`。

⚠️ B 段必须加 `--soft-only-sampling 1`：软 CE 是**逐行二选一**（mask=0 的行贡献
恰好 0，不退化成 one-hot）。不加 ⇒ 只有约 1% 的行吃软标签，policy 去折中而不是
学搜索，且 `--soft-weight 0` **关不掉**这个问题。

### 3.3 段 A 只训 policy 系

段 A 启用：`#1 policy`、`#2 π_opp`、`#3 value` 3 类、`#11 futurepos`。

**不启用**：score 系（`#5/#6` scorebelief、`#8` scoremean、`#9` lead、`#10` scoring）
与 `#4 ownership`、`#12 seki`。理由见 [§6.2](#62-段-1-为何不训-score)。

### 3.4 fp16 下 V7 的 LR 稳定边界（**别用 run.txt 那条 LR 公式**）

2026-10-05 实测（4× 多卡，段 A，`--batch-size 4000`/卡）：

| step | lr | loss | scale | skip |
|---:|---:|---|---:|---:|
| 400 | 7.07e-03 | 12.3146 | 81920 | **0** |
| 419 | ← warmup 结束（= 10% × 4198），LR 到顶 | | | |
| 430~440 | ≈7.37e-03 | **nan** | 峰值 163840 | |
| 450 | 7.37e-03 | nan | **10** | 13 |

**从健康到爆炸，LR 只差 4.2%**（7.07e-3 健康、7.37e-3 即炸）。`7.07e-3` 不是
「安全值」，只是**爬升段的顶端** —— 离峰值仅 19 步。

**两层 fp16 天花板，只有一层被处理**：

| 层 | 谁在管 | 后果 |
|---|---|---|
| 缩放后的**梯度**（65504） | `GradScaler` 跳步 + 减半 | 可恢复，每减半赔一步 |
| 未缩放的**前向激活**（65504） | **无人管** | NaN，**不可恢复** |

那次 `scale` 从 81920 峰值 163840 一路减到 10（**14 次减半，一次也没救回来**），
而 `step 400` 时梯度侧还有巨大余量（163840 都没溢出）⇒ **瓶颈在前向**。
诊断指纹也对得上：`inf=0 nan=全量`（每个参数张量的 nan 数等于它的元素数）。
若是反向算子溢出则会是 `inf`；NaN 铺到 `stem`/`global_fc` 只可能来自前向。

⚠ **`LR = 0.00356 × 有效batch / 2500` 那条公式对 V7 未标定**，照它算 4000×4=16000
会得到 **0.0228**，比炸掉的值还高 2.3 倍。用 run.txt 里 4 卡的**实测**值
**`0.00637`**（≈ 已知健康点的 65%，余量是观测余量的 8 倍）；仍不稳就把 warmup
加到总步数的 15~20%。

**前向 NaN 现在是硬失败**（`train_sft.py`，判据是 `total_finite is False` ——
「被优化的那个标量非有限」，而**不是**「某项坏了」，否则段 A 那九项系数为 0、
恒被净化的项会每步都炸）。抛出前带上逐项点名 / 坏在操作数 / 被净化的坏行 / 当前
scale。没有这道闸门时，「训练死了但还在跑」—— 除 `loss=nan` 外日志一切正常
（速度、显存、耗时都稳），能白烧 12 小时。

---

## 4. 当前进度

| 组件 | 状态 | 备注 |
|---|---|---|
| 主数据集 `sgf_19x19_full.npz` | ✅ 已建 | 34,202,713 行 / 162,298 局；**不可 rebuild** |
| 语料完整性 | ✅ 已核实 | 169,878 SGF 零重叠，完整覆盖；100% 的局 ≥20 手 |
| KataGo 权重 + stdata | ✅ 就位 | `katago/`（含 204 MB 权重）；**已被 `.gitignore` 排除** |
| `pos_hash.py`（唯一散列口径） | ✅ | 478,603 行/s；全量 34.2M ≈72 s |
| `katago_npz.py`（stdata 读取） | ✅ | 27 项测试，含两条真实 stdata 对拍 |
| `kata_label_join.py` | ✅ | `materialize_dataset` + join + `REQUIRED_KEYS` + **防陈旧缓存**（含 `hash_spec_fingerprint()`）|
| `permute_soft` | ✅ | 8 项测试；方向极易搞反，见 [§5.1](#51-散列口径只有一份) |
| 软 CE（`soft_cross_entropy`） | ✅ | 掩码 `mask=0` 的行贡献**恰好 0**（**不是**退化成 one-hot CE）；分母恒为 B ⇒ 不开 `--soft-only-sampling` 就只有 ~1% 的行贡献，policy 项缩小约 100× |
| `labels=True` 4 元组 + dict 契约 | ✅ | 5 个假 dataset 已补 `labels=False` 形参 |
| `games.npz` sidecar **生成器** | ✅ | 小规模实测 50/50 匹配；`g_rules`/`g_resign`/`g_resign_side` 4 键 |
| `games.npz` **产物** | ⚠️ **未生成** | 段 A 的 komi/score/rules 来源；需跑 `build_games_sidecar.py`（几十分钟） |
| `kata_labels.npz`（段 B 蒸馏） | ✅ **已生成** | 157.5 MB，跑 `label_sgf.py` |
| `soft_index.npz` | ✅ **已生成** | 160.5 MB，跑 `build_soft_index.py`；`--soft-index` 已接入训练侧 |
| `feature_v7.py`（气桶 / 历史 5 手 / `calculateArea`） | ✅ | ch0–6/ch8 对 stdata 逐位 1.000000；**ch18/19 已知对不齐**（官方先提死子，本仓 `score()` 没有） |
| `feature_v7_ladders.py`（ch14–17） | ✅ | 对官方 stdata **逐位 1.000000**（4,368 行）；梯子占特征耗时 99.8% |
| `katago_v7.py`（5,562,121 参数） | ✅ | `tests/test_katago_v7_budget.py` 精确断言 |
| 22ch builder 注册 | ✅ | `src/inference.py:177` |
| 12 项 loss 装配 | ✅ | `tests/test_katago_v7_loss.py` 24 项 |
| V7 端到端冒烟 | ✅ 跑过 | 40 步，11/12 项下降；`score_stdev` **−0.0%** |
| **`train_sft.py` 切 22 通道** | ✅ **已做** | `--v7 1` 走 `NbtTfNet`；4 卡实跑（2026-10） |
| 软标签 CLI 接线 | ✅ 已做 | `--soft-index/--soft-weight/--soft-every/--soft-only-sampling/--policy-loss soft_ce` |
| 邻行 gather（5 偏移）接线 | ✅ 已做 | ch14–17 与 futurepos 共用，纯 `boards` 索引 |
| A/B/C 共用一个 V7 模型 | ✅ 已做 | `load_from_path(v7=1)` 按布局分派 + `--model` 承接，`tests/test_v7_single_model.py` |
| V7 软 CE（含 π_opp 分离） | ✅ 已做 | `tests/test_v7_soft_ce.py`，逐位对齐 12 通路口径 |
| 段 A 权重 → `.bin.gz` → 引擎 | ✅ **已打通** | 250 步真实权重；analysis `Model version 17` / 5545737 params；**GTP 19/19**、genmove 3/3 |
| C 段（stdata）整轮正式训练 | ⬜ 未做 | 只有 64 行 CPU 冒烟跑通 |
| A 段整轮正式训练 | ⚠ **跑到 step 400 后炸** | `--lr 9.77e-3` 越过 fp16 前向上限；见 [§3.4](#34-fp16-下-v7-的-lr-稳定边界别用-runtxt-那条-lr-公式) |
| B 段正式训练 | ⬜ 未做 | 通路已通 |
| RL（段外） | ✅ 可跑 | 但 MCTS 已归档，采集走 lookahead |
| NPU 融合注意力探针 | — 已废弃 | NPU 支持已移除（详见 [§7](#7-已知限制) 条目 10） |

---

## 5. 会静默出错的 10 个坑

这一节是本文件最有价值的部分。以下每一条**错了都不报错**，只会静默地算错。

### 5.1 散列口径只有一份

`pos_hash_block` = **棋盘(361B) + to_play(1B) + ko(2B)**，三样都进哈希。
**唯一事实来源 = `src/data/pos_hash.py`**；`probe0_join.py` / `label_sgf.py` /
`build_soft_index.py` 全部 import 它。

🔴 `to_play` 或 `ko` 错一个 ⇒ 探针实测 **0/5 匹配**，join **静默变成空**，
症状看起来像「这批局面真的没标签」。这不是数据问题，是你重写了散列。
改了种子会让 `kata_labels.npz` 里已落盘的 `pos_hash` 列全部失效，而症状同样是
「join 变空」⇒ **永远不要重写这个文件**。

（`game_id` 与 `(sgf 路径, ply)` 都不能当 join 键：前者被 id 复用 bug + 45% 过滤率
毁掉；后者要求两侧独立枚举 SGF 且顺序完全一致，枚举规则动一行就静默错位，
而「join 不上」和「本来就不该 join」长得一模一样。）

### 5.2 `KM` 有 ×100 家族

`foxpro` 语料里 `KM[650]` 表示 **6.5**（2,163 局 / 1.62%）。
照字面存 650 会让全局特征 `currentSelfKomi/20` 变成 **32.5**。

处理：`scripts/build_games_sidecar.py::parse_komi` 在 `fix_outlier=True` 时把
`|v| > KOMI_OUTLIER_ABS (=100)` 的值除以 100。命令行开关是 `--no-komi-outlier-fix`
（**默认是修的**，别顺手加上）。

### 5.3 历史门控是回退复制，不是置 0

官方 `fillRowV7`：

```cpp
prevBoard     = (numTurnsOfHistoryIncluded < 1) ? board     : hist.getRecentBoard(1);
prevPrevBoard = (numTurnsOfHistoryIncluded < 2) ? prevBoard : hist.getRecentBoard(2);
```

**history=0 ⇒ ch15=ch14、ch16=ch15**；history=1 ⇒ ch16=ch15。
写成置 0 不崩，只让开局几个位置的输入偏掉。

### 5.4 `labels=True` 必须是 4 元组

```python
sample_batch_numpy(idxs, rng=None, augment=True, labels=False)
```

- `labels=False`（**默认**）→ **3 元组** `(states, moves_out, values)`
- `labels=True` → **4 元组** `(states, moves_out, values, labels_dict)`

🔴 **不是 5 元组**。5 元组 `(…, soft, mask)` 没有 dict 的位置，而 V7 的 loss 需要那个
dict ⇒ 最后只会变成 6 元组或 fork 两条路。4 元组让预取器只需搬运**一种** payload 形状。
（实现过程里确实一度做成 5 元组，已订正回 4 元组。）

### 5.5 软 CE 掩码逐行二选一，不是混合

`soft_mask` 是**逐行二选一**：1 = 软 CE，0 = one-hot CE，**不做混合**。

```python
soft_cross_entropy(...) = (per_row * mask).mean()   # mask=0 的行贡献 0
```

理由：同一个 head 同时收到「搜索分布」与「人类 one-hot」会去**折中**而不是学搜索。
`mask=0` 的行贡献 0，不是「退化成 one-hot CE」。段 B 用独立采样器**只取软行**。

🔴 两条容易被忽略的对齐口径（V7 与 12 通道**逐位相同**，由
`tests/test_v7_soft_ce.py` 钉住）：

- **分母恒为 B**，不是 `Σmask`。若改成 `sum/mask.sum()`，软项量级会随 batch
  组成漂移，`--soft-weight` 这个旋钮就失去意义。
- **`soft_mask` 全 0 ⇒ 退回 one-hot**（含义是「本批无软标签」）。否则 policy 拿到
  **恰好 0** 的梯度：不报错、loss 照降、policy 根本没学。
  ⚠️ 所以 `--policy-loss soft_ce` 在「无软标签来源」时必须报错；唯一例外是
  V7 的 stdata 分片，它的 `soft` / `soft_mask` **内建在数据行里**、不需要
  `--soft-index`。

### 5.6 `RE` 认输类不能当成分差

`B+R` / `W+Resign` / `B+T` / `W+Forfeit` ⇒ **认输填 `NaN`，不是 0**。
`g_score` 为 `NaN` 时 `w_score = 0`。

🔴 **真实语料里约 3/4 的 SGF 是认输**：`games.npz` 全量实测认输率 **76.49%**
（`g_resign=True`，与 `g_re==RESIGN` 的 124,137 局一致），有数值分差的
**23.18%**（`g_score` 非 NaN，37,613 局）。
⚠️ 早期抽样曾写 75.2% / 81.09%，run.txt 曾写 77.51% / 22.20% —— 这几个数
在 `g_komi` / `g_score` / `g_resign` / `g_re` 四种口径下**都算不出来**，已按实测纠正。
引用覆盖率时必须写清是哪个字段算的。
照字面把认输当 0 分会让 scoremean / lead 被系统性地教成「双方刚好一样」。

### 5.7 ch18/19 无法与官方 stdata 对齐

官方 `calculateArea` **先提死子**，本仓 `GoBoard.score()` 没有死子判定
（其 docstring 自陈「没有死子判定，对局双方应在 pass 认输前实际提掉对方的死子」）。

🔴 实测 1254 行**零行完全相同**（总点多 21,896 点）。
⇒ **stdata 只能当 ch3/4/5 与 ch9–13 的 oracle**，不能拿它判 ch18/19 的对错。

ladder 通道的可验证性因此被拆成两半（`katago/` 下无 `cpp/` 源码，stdata 是唯一 oracle）：

| 通道 | 依赖盘面 | 能否对 stdata 直接验证 |
|---|---|---|
| ch14 | 当前盘 | ✅ |
| ch17 | 当前盘（working move） | ✅ |
| ch15 | 前一手盘 | ❌ stdata 是逐行独立样本 |
| ch16 | 前二手盘 | ❌ 同上 |

⇒「**算法**」由 ch14 对 stdata 逐位担保，「**盘面**」由 SGF 重放担保。

### 5.8 `test_no_new_cli_params` 把 61 个 CLI flag 整个冻结

`tests/test_huber_loss.py::test_no_new_cli_params` 断言「D1 零新增 / 零删除 /
零改名」。**加任何新 flag 都要同步改它**，否则 CI 红。
同理，日志三个键的键名也被 `test_log_keys_unchanged` 钉死。

`loss` / `policy_loss` / `opt_loss` 的**键名没变但含义变过**：
`loss = policy_loss + w·value_loss + c‖θ‖²`（含报告用 L2），
`opt_loss = policy_loss + w·value_loss`（唯一被 backward 的量）。
⇒ **跨新旧 run 的 loss 曲线不可直接比**；`top1` 仍可比。

### 5.9 真实语料里一定有一部分坏 SGF

曾整批 13 万局卡死于 `Illegal move 72: F2`（某局没有 `AB/AW/HA`，文件名以数字开头）。

`label_sgf.py` 已改为收集 `(resps, errors)` 由调用方隔离，**不再因单局抛异常**。
判定用 `_BAD_MOVE_RE = /Illegal move\s+(\d+)/i`，并且只有 `out` + `errors` 一起
凑满目标数量才停（否则坏局会被静默跳过、循环提前结束）。

### 5.10 `requirements.txt` 漏列 `scipy`（记账缺口）

`src/data/feature_v7.py` 直接 `from scipy.ndimage import label, binary_dilation`，
`src/game/go_rules.py` 另有 8 处 import，但 **`requirements.txt` 里没有 `scipy`**
（同样漏了 `onnxruntime`）。新环境按 `pip install -r requirements.txt` 装完会
在 import 时炸。

> 已知缺口，**故意不在此处修**（`requirements.txt` 不在本文件的改动范围内）。
> `run.txt` 里显式补了 `pip install scipy`。

---

## 6. 关键决策与理由

### 6.1 22 通道为何现场算，不落盘

| 理由 | 说明 |
|---|---|
| **主数据集不可 rebuild** | §2.1 第 1 条。加列 = 重建 = 毁掉 join 链 |
| **11/13 项不需要新数据** | 现有 10 列能推出或实时算；只有 6 个局级标量要 sidecar |
| **空间开销不划算** | bit-packed 22 通道 × 34.2M 行 ≈ 940 MB；实时算是「零新增空间存储」 |
| **性能有余量** | 3.04 ms/行预算 vs 1.78 ms 基线 = 1.7×；贵的是 `iterLadders`（3 块盘面） |
| **降级成本为零** | 若 `iterLadders` 超预算 ⇒ 降到 **18 通道**（ch14–17 恒 0），主干 / 四头 / 12 项 loss / 参数量 5,562,121 **一行不改** |

未来pos 几乎免费：每局行数 p50=208 / mean=210.7，`+8` 有效 96.2%、`+32` 有效 84.8%
⇒ **gather 全部行是免费的**（`boards` 常驻内存，就是索引），只有靠近终局的那
**4%** 行需要真跑一次 area 的 flood fill。

### 6.2 段 A 为何不训 score

三条理由，按重要性：

1. **覆盖率**：约 76.5% 的局是认输（实测 `g_resign` 76.49%，见 [§5.6](#56-re-认输类不能当成分差)）
   ⇒ 没有分差 ⇒ `g_score = NaN` ⇒ `w_score = 0`。score 系 5 项
   （`#5/#6` scorebelief、`#8` scoremean、`#9` lead、`#10` scoring）在
   **约 76% 的局上权重为 0**，有监督的只约
   `23.18% × 162,298 × 210 ≈ 7.9M 行`（34.2M 的 **23%**）。跑一个 23% 覆盖的
   score 头，不如等 B/C 段。
2. **信号质量**：认输局即使重放到终局，`GoBoard.score()` 没有死子判定 ⇒
   从认输棋谱反推的终局归属**不可靠**。`ownership` / `scoring` / `seki` 正是这类。
3. **有更好的来源**：stdata 的 `valueTargetsNCHW` 直接带 ownership / seki /
   futurepos / scoring（值域 `{−1,0,+1}` 与 `±120`），来自**搜索器自己的评分器**，
   不需要移植任何东西。

⇒ **训练责任从 34.2M 挪到 ~3.4M**，而那 3.4M 上这些信号质量高得多。
⇒ ownership / scoring / seki **不在段 A**，交给 B/C 段。

### 6.3 sidecar 为何只存 4 键

| 决策 | 理由 |
|---|---|
| **局级而非逐行** | 逐行存是 3.6 B/行 ⇒ 34.2M 行 = **123 GB**；局级是 **1.2 MB** |
| **只留 `g_komi` / `g_score` / `g_rules` / `g_resign`** | 砍掉 `g_ownership` / `g_scoring` / `g_seki`（原 176 MB → **1.2 MB**）；墙钟从 8–28 min 降到几分钟。理由见 §6.2 |
| **哈希锚点对齐，不能靠顺序** | `build_dataset.build()` 一串过滤实测丢掉 **45%** 的 tgz 成员，`tarfile` 顺序 ≠ glob 顺序；且有 id 复用 bug。锚点法（ply-20）对枚举顺序与该 bug **完全免疫** |
| **必须打印覆盖报告** | 匹配率 <100% 时未匹配的局权重置 0，**不静默填垃圾** |

`g_komi` **不能假设常数**：实测 7.5 占 37.7%、6.5 占 18.0%、5.5 占 15.1%、
3.8 占 10.5%、**0 占 5.9%**、2.8/4.5/缺 各若干 ⇒ 必须逐局存。

### 6.4 policy 保持 K=2

B/C 段的目标同型（搜索分布）；人类 one-hot 已在 A 段学进去，
B/C 段覆盖它**正是蒸馏的目的**。K=3 只在「同一部位两个 policy 目标同时训」时才需要，
而那会带来权重调参负担且收益未经验证。K=2 恰好对上 stdata 的两路输出
（`policyTargetsNCMove[0]` 本方 / `[1]` 对手）。

⚠️ 两路目标必须**真的不同**。V7 曾把 `policy_opp` 的目标也填成 player 的
`rank[0]`（两路同源）⇒ `π_opp` 白训，而 loss 曲线完全看不出异常。
现在 B 段的 `soft` / `soft_opp`、C 段的 `policy_player_prob` / `policy_opp_prob`
各自独立，`tests/test_v7_soft_ce.py::test_policy_and_policy_opp_use_different_targets`
逐位钉住。

---

## 7. 已知限制

1. **无 KataGo C++ 源码。** `katago/` 下**只有 Windows 二进制**（`katago.exe` +
   dll），**没有 `cpp/`** ⇒ `nninputs.cpp::fillRowV7`、`iterLadders`、`calculateArea`
   的官方实现**不可读**。stdata 是唯一可执行 oracle。
2. **ch18/19 无法与官方 stdata 对齐**（详见 [§5.7](#57-ch1819-无法与官方-stdata-对齐)）。
3. **`docs/` 被 `.gitignore` 排除**（`:46 /docs`）⇒ 两份 spec 与 sidecar 生成脚本
   都不在版本控制内，无设计历史。
4. **`katago/` 也在 `.gitignore` 里**（含 204 MB 权重）⇒ 换机器要重新准备权重与 stdata。
5. **`requirements.txt` 漏 `scipy`**（见 [§5.10](#510-requirementstxt-漏列-scipy记账缺口)）。
6. **`scorestdev` 的 `beta` 已裁决为 `1.0`** ✅ 原 spec 写 `SoftPlus(x, 0.05)`，
   预测初值高达 277，而 loss #7 的目标量级只有 5~20（Huber δ=10）⇒ 初期梯度被
   常数偏差完全支配（40 步冒烟实测该项 **−0.0%**，即**根本没在学**）。
实测改 `beta=1.0` 后 `score_stdev.mean()` **13.8599**、loss #7 **8.83**，
    **落在 δ=10 附近**；参数量不变（5,562,121），段 A 四目标**逐位不变**
    （段 A 权重为 0，此改动是为 B/C 段生效的）。
    常量在 `katago_v7.py::SCORE_STDEV_SOFTPLUS_BETA`，**只改这一行即可**。
7. **`policy_entropy` 曾报负熵** ✅ 已改成真熵 `H`。原值是 `Σ p·log p = −H`，
   取值 `[−log 362, 0]`、随策略变锐而**上升** ⇒ 与指标名方向相反，读图必反。
   现取值 `[0, log 362]`（均匀 ≈ 5.89，锐化后下降），**换算关系 `新 = −旧`**。
   ⚠ **历史 run 的这条曲线符号会翻转**。RL 侧 `selfplay_train.py` 本来就算真熵，
   这次是让 SFT 与 RL **对齐**。
7. **`scoring` 开局就占 83/104**（随机预测 vs ±120 目标）⇒ 梯度范数在 step 20
   冲到 365、被 clip 到 5。真实跑要相应加 warmup，或先确认 `±120` 缩放是否该归一。
8. **stdata 布局随网络版本变。** `globalTargetsNC` 是 **64**（b40c768nbt / b28c512）
   vs **80**（tf3-b11c768）列；`qValueTargetsNCMove` 只有前两批有
   ⇒ **绝不能硬编码列号**，必须按网络名查 `GLOBAL_TARGET_LAYOUT`，未登记的网络报错不猜。
   `zzb28c512` 那批是 **seki 富集**（131,706 行里 104,905 行含 seki 格子）⇒ 当作
   seki 专项单独用，不混进主训练。
9. **`col3` / `col22` 的精确语义未定**（stdata）⇒ 目前**不作为任何 loss 的目标**，只记录。
10. **NPU 融合注意力已移除。** NPU 支持于 2026-10 彻底下线，相关融合注意力探针、
    `head_dim` 支持集、`force_math` 等仅针对 NPU 的判断不再适用。CUDA 走 SDPA /
    FlashAttn，`head_dim` 不受 {16,32,64} 限制；V7 注意力显存按 CUDA SDPA 实测标定。
11. **`run.py` 的 `default_argv` 是旧代残留** —— 仍带 `--compile-mode reduce-overhead`
    （走 CUDA Graphs，维持不归还的私有内存池，临界 batch 下直接 OOM）。
    **别照抄 `python run.py sft` 的默认值。**
12. **显存结论都标着「旧代」。** run.txt 里那张 4 卡账本是 v21 时代
    （184 通道）量的；V7 是 256 通道，换硬件前必须重新标定并先跑 50 步 smoke。
    ⚠ **2026-10-05 实测：`v7_batch_memory_advice` 低估约 3.2 倍** —— 它预测
    batch=1900 约 8.8 GB，实测 **28.28 GB**（64 GiB 卡的 44%；4000/卡时更到
    91%）。后果是**那道启动前预检没能拦住占满卡的 batch**。在真机复测之前，
    别把它当成放行依据。
13. **多卡 fp16 必须自己归约溢出**（`train_sft.py::_scaler_step_global`）。
    `GradScaler` 的 `found_inf` 是**纯本地**的：某 rank 有 inf 时它跳过、其余 rank
    照常 `optimizer.step()` ⇒ **各 rank 权重从此永久不同**，之后每次 `all_reduce`
    都在混合**四个不同模型**的梯度 —— 不是「少训几步」，是**训练从此无效**。
    以 14% 的边际溢出率算，「四张卡同一步一起溢出」的概率极低 ⇒ 分叉几乎立刻发生。
    RL 侧（`selfplay_train.py`）目前靠「缩放值有没有下降」判跳步，**只对单卡成立**；
    将来上多卡 RL 必须换成同一个全局归约。
14. **溢出诊断必须在 `clip_grad_norm_` 之前。** 它在 `total_norm = inf` 时算出
    `clip_coef = 0` 并 `grad.mul_(0)` ⇒ `inf × 0 = NaN`，而 `inf ⇒ clip_coef < 1`
    这个分支**一定会进** ⇒ clip 之后「inf 个数」**结构上恒为 0**，看到的 nan 全是
    clipper 造的。排在 clip 之后就只能打出「溢出可能发生在已被释放的中间张量里」
    —— **那是工具的盲区，不是关于计算的结论**，而且它会把排查带偏到反向与 loss
    （2026-10-05 就被带偏过一次）。