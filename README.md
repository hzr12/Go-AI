# Go-AI —— 对标 KataGo 的 19 路围棋 AI

从 16 万局职业 SGF 语料出发，重建成**官方 KataGo V7 语义**的 19×19 围棋 AI：
输入 = 22 空间通道 + 19 全局特征，主干 = `nbt`(nested bottleneck) + transformer，
四个头（policy / value / ownership / scorebelief），12 项可构造监督目标。

> **当前是「重建管线」阶段，不是「训出强棋」阶段。** 模型、特征、标签三条流都已
> 落地或有实测结论，但**训练侧还没接到 22 通道**：`scripts/train_sft.py` 至今建的是
> `KATAGO_SE_CFG`（12 通道、9,112,005 参数）。进度见 [§4](#4-当前进度)。

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
data/labels/games.npz      局级 sidecar，4 键，约 1.2 MB（待生成）
data/labels/kata_labels.npz  KataGo 访问分布软标签（待生成）
data/labels/soft_index.npz   软标签 → 主数据集行号的 join（待生成）
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

**性能闸门**：NPU 侧需 ~2635 行/s，`--prefetch-workers 8` 下每行预算 **3.04 ms**，
现有 12 通道单盘基线 1.78 ms ⇒ **1.7× 余量**。贵的是 `iterLadders`（3 块盘面）与
`calculateArea`（1 块）。

### 2.3 模型层

| | 现行 SFT（`KATAGO_SE_CFG`） | 目标 V7（`NbtTfNet`） |
|---|---|---|
| 输入通道 | 12 | 22 空间 + 19 全局 |
| 主干 | 240 宽 · 13×SEBottleneck + 4×Attention | `C=256 M=128 H=4 F=384 B=11` nbt2 块 |
| 归一化 | — | `fson` 固定方差标量 + `rsnh` 末端 RMSNorm，**全网无 BN** |
| 头 | policy + value | policy(K=2) + value(3 分类 + scoremean/stdev/lead) + ownership + scorebelief(842 桶) |
| 参数量 | 9,112,005 | **5,561,832**（硬预算 5,850,000，余量 4.9%） |
| 代码 | `scripts/train_sft.py:444` `KATAGO_SE_CFG` | `src/networks/katago_v7.py` |

`NbtTfNet` 已实现并通过预算测试（`tests/test_katago_v7_budget.py` 断言精确值
5,561,832），`src/inference.py:177` 已 `register_in_channels_builder(22, ...)`。
**但 `train_sft.py` 尚未切过来** —— 见 [§4](#4-当前进度)。

### 2.4 训练层

- **段 1 SFT**：`scripts/train_sft.py`（4 卡 910A 入口 `shell/train_sft_npu_4card_katago_se.sh`，
  DDP + HCCL + fp16/GradScaler）。
- **V7 tracer bullet**：`scripts/smoke_train_v7.py` —— 302 行 / batch 8 / 40 步，
  直接吃 stdata，用来回答「12 项 loss 每项到底降不降」。
- **RL**：`scripts/selfplay_train.py`（PPO + lookahead，**MCTS 已在 2026-09-30 归档**，
  采集换成 N 步 minimax 推演）。

### 2.5 推理层

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
| **软标签流** | `permute_soft` / `soft_cross_entropy`（掩码二选一）/ `labels=True` 的 4 元组 + dict 契约 / `games.npz` sidecar 生成器（ply-20 哈希锚点，小规模实测 50/50 匹配） |
| **特征流** | `src/data/feature_v7.py`（气桶 / 历史 5 手 / `calculateArea`）已完成；`src/data/feature_v7_ladders.py`（ch14–17）**仍在收尾** |
| **模型流** | `katago_v7.py` + `katago_v7_loss.py` + 22ch builder 已接线；`train_sft.py` 切 22 通道未做 |

### 3.2 三个训练段

| 段 | 数据 | 量 | policy 目标 | 目的 |
|---|---|---:|---|---|
| **1** | 自有 34.2M 语料（SGF 派生标签） | 34,202,713 | 人类着法 one-hot | 通路基线；学会读 22 通道 |
| **2** | 自有语料的 1%，用 KataGo 标注 | ~342,000 | 访问分布 | 域匹配桥梁（人类局面 + 搜索答案） |
| **3** | `katago/stdata`（19×19 部分） | ≈3,100,000 | 访问分布 | 棋力主体 |

段 2 与段 3 的 policy 目标同为访问分布 ⇒ **可以合并成一轮跑 ~3.44M**。
段 2 的价值是**桥梁**：stdata 是自对弈局，自有语料是职业对局。

### 3.3 段 1 只训 policy 系

段 1 启用：`#1 policy`、`#2 π_opp`、`#3 value` 3 类、`#11 futurepos`。

**不启用**：score 系（`#5/#6` scorebelief、`#8` scoremean、`#9` lead、`#10` scoring）
与 `#4 ownership`、`#12 seki`。理由见 [§6.2](#62-段-1-为何不训-score)。

---

## 4. 当前进度

| 组件 | 状态 | 备注 |
|---|---|---|
| 主数据集 `sgf_19x19_full.npz` | ✅ 已建 | 34,202,713 行 / 162,298 局；**不可 rebuild** |
| 语料完整性 | ✅ 已核实 | 169,878 SGF 零重叠，完整覆盖；100% 的局 ≥20 手 |
| KataGo 权重 + stdata | ✅ 就位 | `katago/`（含 204 MB 权重）；**已被 `.gitignore` 排除** |
| `pos_hash.py`（唯一散列口径） | ✅ | 478,603 行/s；全量 34.2M ≈72 s |
| `katago_npz.py`（stdata 读取） | ✅ | 27 项测试，含两条真实 stdata 对拍 |
| `kata_label_join.py` | ✅ | `materialize_dataset` + join + `REQUIRED_KEYS` |
| `permute_soft` | ✅ | 8 项测试；方向极易搞反，见 [§5.1](#51-散列口径只有一份) |
| 软 CE（`soft_cross_entropy`） | ✅ | 掩码**逐行二选一** |
| `labels=True` 4 元组 + dict 契约 | ✅ | 5 个假 dataset 已补 `labels=False` 形参 |
| `games.npz` sidecar **生成器** | ✅ | 小规模实测 50/50 匹配 |
| `games.npz` **产物** | ⬜ 未生成 | `data/labels/` 目前是空的 |
| `kata_labels.npz`（段 2 标签） | ⬜ 未生成 | 需跑 `label_sgf.py` |
| `soft_index.npz` | ⬜ 未生成 | 需跑 `build_soft_index.py` |
| `feature_v7.py`（气桶 / 历史 5 手 / `calculateArea`） | ✅ | |
| `feature_v7_ladders.py`（ch14–17） | 🔄 **收尾中** | |
| `katago_v7.py`（5,561,832 参数） | ✅ | `tests/test_katago_v7_budget.py` 精确断言 |
| 22ch builder 注册 | ✅ | `src/inference.py:177` |
| 12 项 loss 装配 | ✅ | `tests/test_katago_v7_loss.py` 24 项 |
| V7 端到端冒烟 | ✅ 跑过 | 40 步，11/12 项下降；`score_stdev` **−0.0%** |
| **`train_sft.py` 切 22 通道** | ⬜ **未做** | 现状仍建 `KATAGO_SE_CFG`（12ch / 9.11M） |
| 软标签 CLI 接线 | ⬜ 未做 | 无 `--soft-labels/--soft-weight/--soft-index/--soft-only-sampling` |
| 邻行 gather（5 偏移）接线 | ⬜ 未做 | B6，纯 `boards` 索引，几乎免费 |
| stdata 训练（段 3）整轮 | ⬜ 未做 | 只有 smoke |
| 段 1 正式训练（新模型） | ⬜ 未做 | 被上一行阻塞 |
| RL（段外） | ✅ 可跑 | 但 MCTS 已归档，采集走 lookahead |
| 910A 融合注意力探针 | ⬜ 未做 | R1，见 [§7](#7-已知限制) |

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
`mask=0` 的行贡献 0，不是「退化成 one-hot CE」。段 2 用独立采样器**只取软行**。

### 5.6 `RE` 认输类不能当成分差

`B+R` / `W+Resign` / `B+T` / `W+Forfeit` ⇒ **认输填 `NaN`，不是 0**。
`g_score` 为 `NaN` 时 `w_score = 0`。

🔴 **真实语料里 81.09% 的 SGF 是认输**（spec 早期抽样写的是 75.2%，
`build_games_sidecar.py` 在 162,298 局上实测是 **81.09%**），只有约 **18.5%** 有数值分差。
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
| **降级成本为零** | 若 `iterLadders` 超预算 ⇒ 降到 **18 通道**（ch14–17 恒 0），主干 / 四头 / 12 项 loss / 参数量 5,561,832 **一行不改** |

未来pos 几乎免费：每局行数 p50=208 / mean=210.7，`+8` 有效 96.2%、`+32` 有效 84.8%
⇒ **gather 全部行是免费的**（`boards` 常驻内存，就是索引），只有靠近终局的那
**4%** 行需要真跑一次 area 的 flood fill。

### 6.2 段 1 为何不训 score

三条理由，按重要性：

1. **覆盖率**：81.09% 的局是认输 ⇒ 没有分差 ⇒ `g_score = NaN` ⇒
   `w_score = 0`。score 系 5 项（`#5/#6` scorebelief、`#8` scoremean、`#9` lead、
   `#10` scoring）在 **81% 的局上权重为 0**，有监督的只约
   `18.5% × 162,298 × 210 ≈ 6.3M 行`（34.2M 的 **19%**）。跑一个 19% 覆盖的
   score 头，不如等段 2/3。
2. **信号质量**：认输局即使重放到终局，`GoBoard.score()` 没有死子判定 ⇒
   从认输棋谱反推的终局归属**不可靠**。`ownership` / `scoring` / `seki` 正是这类。
3. **有更好的来源**：stdata 的 `valueTargetsNCHW` 直接带 ownership / seki /
   futurepos / scoring（值域 `{−1,0,+1}` 与 `±120`），来自**搜索器自己的评分器**，
   不需要移植任何东西。

⇒ **训练责任从 34.2M 挪到 3.4M**，而那 3.4M 上这些信号质量高得多。
⇒ ownership / scoring / seki **不在段 1**，交给段 2/3。

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

段 2/3 的目标同型（访问分布）可合并成一轮；人类 one-hot 已在段 1 学进去，
段 2/3 覆盖它**正是蒸馏的目的**。K=3 只在「同一部位两个 policy 目标同时训」时才需要，
而那会带来权重调参负担且收益未经验证。K=2 恰好对上 stdata 的两路输出
（`policyTargetsNCMove[0]` 本方 / `[1]` 对手）。

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
   **落在 δ=10 附近**；参数量不变（5,561,832），段 1 四目标**逐位不变**
   （段 1 权重为 0，此改动是为段 2/3 生效的）。
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
10. **910A 融合注意力未探针。** 现有 `head_dim=46 ∉ 支持集 {16,32,64}`
    （推测是 `force_math` 的根因）。V7 的 `head_dim=32` 落在支持集内，
    若融合可用 ⇒ 注意力显存 6.2–9.3 GiB → **~0.7 GiB**，总峰值 21–25 → 15–19 GiB。
    最坏情形就是维持现在的 `force_math`，所以这只是「可能更好」，不是阻塞项。
11. **`run.py` 的 `default_argv` 是旧代残留** —— 仍带 `--compile-mode reduce-overhead`
    （走 CUDA Graphs，维持不归还的私有内存池，临界 batch 下直接 OOM）。
    **别照抄 `python run.py sft` 的默认值。**
12. **显存结论都标着「旧代」。** run.txt 里那张 4 卡 910A 账本是 v21 时代
    （184 通道）量的；V7 是 256 通道，换硬件前必须重新标定并先跑 50 步 smoke。