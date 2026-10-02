# Go-AI 训练流水线设计（分段 · 软标签 · 22 通道实时算）

> **状态**：设计定稿（2026-10-02，经 20+ 轮逐节问答批准），待转 `writing-plans`。
> **配套**：`2026-10-01-katago-nbt-tf-design.md`（模型 + 22 通道输入 + 12 项 loss）。
> 本 spec 只管**训练流水线**：分段策略、软标签接入、局级 sidecar、22 通道实时算的接线。
> **一手来源**：`https://github.com/lightvector/KataGo`；stdata 数据的实测反解见配套 spec §9。
> **硬约束**：**不 rebuild** `data/sgf_19x19_full.npz`（34,202,713 行 / 162,298 局）。

> 🔴 **2026-10-02 实现期回写**：本 spec 已按实跑结果订正，**被推翻的抽样值一律保留原文
> 并就地标注「原以为是 X / 实测是 Y」**，不静默删除 ——
> 想知道「为什么改」的直接看**§2.6.1**（认输率 / 贴目 ×100 家族 / 两个 `RE` 形式 /
> 哈希锚点三个坑）与 **§3 B 组**（ch18/19 不可对齐、ch14/ch17 已达 `1.000000`）。
> 最关键的三条：**认输率 81.09% 不是 75.2%**、**ch18/19 不能与 stdata 对齐**、
> **CLI 的 flag 被整体冻结（61 个，A4 之后是 65 个）**。
>
> 🔴 **2026-10-02 第二轮回写（本轮，以代码为准）**：
> ① **「CLI 的 61 个 flag」→ 65，`soft_ce` 已接进 CLI** ⇒
>    「段 2 无法从命令行启动」**作废**（§3 A4 / §5 / §6 第 5 行）；
> ② **`soft_mask = 0` 是「贡献恰好 0」，不是「one-hot CE」** ⇒ 分母恒为 `B`
>    ⇒ **段 2 必须配 `--soft-only-sampling`**，否则 1% 软行把 policy 项缩小 ~100×（§1.2 / §3 A2）；
> ③ **C0 已实跑**：3.04 ms/行预算**在本机不成立**（冷读 3.66 ms/行），
>    且 **batch 与 `--prefetch-workers` 解耦**（§2.3）；
> ④ **ch9–13 / ch7 / ch20-21 / ch15-16 与官方 stdata 结构上不可比**
>    ⇒ B7 的覆盖面比 §3 原写的小得多（§3 B7）；
> ⑤ 🔴 **A6（`futurepos`）在本 spec 里原本完全不存在**，已补进 §3 A 组（§3 A6 段）；
> ⑥ 三个测试文件名不存在、另四个实际存在的文件没列（§3 B 组 / §7）。

---

## §1 目标与分段策略

### 1.1 三段式

| 段 | 数据 | 量 | policy 目标 | 目的 |
|---|---|---:|---|---|
| **1** | 自有 34.2M 语料（SGF 派生标签） | **34,202,713 行** / 162,298 局 | 人类着法 one-hot | 通路基线；学会 22 通道输入的读法 |
| **2** | 自有语料的 **1%**，用 KataGo 标注 | **~342,000 行** ≈ **1,623 局** | **访问分布** | 域匹配的蒸馏：同样的职业对局，但换成搜索标签 |
| **3** | `katago/stdata`（19×19 部分） | **≈3,100,000 行** | **访问分布** | 棋力主体 |

🔴 **口径钉死（2026-10-02 复核）**：段 2 的 **`~342,000` 是**行**数，不是局数**。
两种「1%」在**行**的口径上恰好重合，因为语料平均段长是常数：

```
1% × 34,202,713 行                      = 342,027 行
1% × 162,298 局 × 210.74 行/局（实测） = 342,027 行   ← 同一个数
⇒ 342,027 行 ÷ 210.74 行/局 = 1,623 局
```

**段长实测**（`tmp/materialized/game_ids.npy`，34,202,713 行全量切段）：
min 12 / **p50 208** / p95 302 / max 2068 / **mean 210.74**，共 **162,298 段**。
⚠ 配套 spec §5.7 里那句「约 316 行/局 ⇒ 34.2M 行约覆盖 108,000 局」是错的
（混用了语料 SGF 列表下标与主 npz 行号），**已作废**：
**34.2M 行 = 全部 162,298 局**，不是 108,000 局。
⇒ 后面凡出现「段 2 有多少数据」，**行**写 342,000，**局**写 1,623，
两者不要混着加。

**段 2 与段 3 的 policy 目标同为访问分布 ⇒ 可以合并成一轮跑 ~3.44M 行**
（342,027 + ≈3,100,000 = 3,442,000）。
段 2 的价值是**域匹配桥梁**：stdata 是自对弈局，我们语料是职业对局，
1% 那一小段让模型见到「人类局面 + 搜索答案」这种配对。

### 1.2 三条关键决定与被否决的替代

| 决定 | 否决的替代 | 理由 |
|---|---|---|
| **policy 保持 K=2**（不加第 3 个输出） | K=3（访问分布与人类 one-hot 各占一路） | 段 2/3 目标同型、可合并；人类 one-hot 已在段 1 学进去，段 2/3 覆盖它**正是蒸馏的目的**。K=3 只在「同一部位两个 policy 目标同时训」时才需要，而那会带来权重调参负担而收益未经验证。且 K=2 恰好对上 stdata 的两路输出（`policyTargetsNCMove[0]` 本方 / `[1]` 对手；spec 的 `π_opp` = 下一手，在自对弈里与「对手走的那手」是同一个东西） |
| **`soft_mask` 逐行二选一**（1 = 走软 CE，**0 = 该行贡献恰好 0**），不做混合 | 软硬线性混合（`--soft-weight` 控比例） | 同一个 head 同时收到「搜索分布」与「人类 one-hot」会**去折中**而不是学搜索。混合多一个旋钮且无证据支撑。⚠ 原表里写的「0 = one-hot CE」**从来不存在** —— 见下方订正 |
| **段 2 只训软行**（`--soft-only-sampling`） | 全量混训 / 靠 `--soft-weight 0` 关掉软行 | 混训 = 上面的折中问题；`--soft-weight 0` 等于「开了软标签但 99% 样本还在教 one-hot」的**静默半吊子**。独立采样器与 Phase 4 item 3 的窗口**正交**，两者可叠加。🔴 **这一条现在是承重的、不只是「更干净」** —— 见下方订正 |

> 🔴 **订正（2026-10-02 复核，以代码为准）：「0 = one-hot CE」这半句是错的。**
> `scripts/train_sft.py::soft_cross_entropy` 的实现是
>
> ```python
> return float(soft_weight) * (per_row * mask).mean()   # per_row = −Σ soft·log_softmax
> ```
>
> ⇒ **`mask = 0` 的行既不进分子也不进分母，贡献恰好 0**，
> **不退化成 one-hot CE**、也不与软项插值。
> 函数自己的 docstring 就写着这句话，
> `tests/test_soft_ce.py::test_mask_is_not_a_mix_with_one_hot_ce` 把它钉死。
>
> **后果（本 spec 最重要的一条操作纪律）**：分母**恒为 batch 大小 `B`，不是 `Σmask`**
> ⇒ 软项量级随「本批软行占比」线性变化。
> 在 34.2M 行上开 `--soft-index` 而**不开** `--soft-only-sampling`
> ⇒ 只有约 **1%** 的行贡献 ⇒ **policy 项被缩小约 100×**，
> 等于「以为在蒸馏，其实在用 1% 的学习率训 one-hot」。
> ⇒ **段 2 的正确命令行 = `--soft-index` + `--soft-only-sampling 1`**。
> `--soft-weight` 是**全局量级旋钮**，不采样时调它等于在调「软行占比」。

### 1.3 不在范围内

- 非 19×19 泛化（stdata 里 63~75% 是 19×19，其余丢弃）
- 融合注意力的深度优化（`sdpa_force_math` 先保留）
- `scorestdev` 的 `beta` 裁决（配套 spec §4.2 标为待裁决；🔴 **实测该项在 40 步内 −0.0%，
  等效于没有学习信号，但「改 `1.0`」尚未获批准** ⇒ 见 §6 第 1 行）
- selfplay RL（仓库已有 P3-C PPO；本管线的终点是「像 KataGo」，超过教师要靠 RL）
- 🔴 **ch18/19 的死子判定**（配套 spec §2.5）：本轮实测确认官方**无法对齐**
  （1,254 行 0 行相同），因此**不补**判定 —— 补了要猜判据，猜错会得到第三种偏差。
  这不是「留待以后」，是**已决的不做**，见 §6 第 6 行

---

## §2 已核实的事实基线

以下每个数字都是实测的，方案的形状由它们决定。

### 2.1 语料与数据集

| 项 | 值 |
|---|---|
| 主 npz | 34,202,713 行 / **162,298 局**（`game_ids` 稠密 `0..162297`）/ 10 列 |
| 每局行数 | min 12 / p50 208 / p95 302 / max 2068 / **mean 210.74**（34,202,713 行全量切段实测） |
| ~~≥20 手的局~~ | ~~162,296 / 162,298 = 100% ⇒ ply-20 锚点对所有局安全~~ 🔴 **措辞作废**：那个分子**就是**实测的 `L ≥ 20` 段数，但它是 **99.9988%** 不是 100%；真正的风险在**坑 3**（`min(20, L//2)` 对 134 段 `L<40` ≠ 20），见 §2.6.1 ④ |
| **段长长尾（实测）** | **134 段 `L < 40`**、**2 段 `L < 20`** ⇒ 锚点必须用**偏移 1..20 的全前缀** |
| `game_ids` 形状 | 162,298 段对 162,298 个 **distinct** id ⇒ **是置换，不是升序** |
| 语料：tgz 5 个 | 36,274 唯一 SGF |
| 语料：目录 4 个 | 133,604 唯一 SGF |
| **两者重叠** | **0** |
| **并集** | **169,878** ≥ 162,298（接受率 **95.6%**）⇒ **语料完整，无缺失局** |

### 2.2 十三项 V7 需要里，11 项不需要 rebuild

配套 spec §5.1 有完整表格。两处**白捡**必须记在这里，因为它们改变了工作量：

> 🎁 **`value` 3 类能从 `winrates` 精确还原。**
> `build_dataset.py:299-309`：`winrates = tanh(value·to_play·α)`，
> `value ∈ {−1,0,1}`、`α = 0.3+0.7·frac ∈ [0.3,1.0]`。
> `fv=0` ⇒ `winrates=0` **精确**；`fv=±1` ⇒ `|winrates| = tanh(α) ∈ [0.291,0.762]`，
> **永远不为 0**。⇒ **`winrates==0` ⟺ 和棋**；`sign(winrates)` = to_play 视角胜负。
> ⚠ 但 `|winrates|` 是 α 污染过的，**还原不出真实分差** ⇒ 分差仍需 sidecar。

> 🎁 **`calculateArea` 不是从零移植。**
> `go_rules.py:2261 score()` **已经是区域计分 + flood fill**，且 docstring 声明与
> Tromp-Taylor 计分等价。移植 = **在它之上暴露逐点归属图** + 对齐 KataGo 的
> 区域分类语义（`SCORING_AREA`/`TAX_NONE` 条件化、seki 区域）。
> 底层 flood fill（`_neighbor_groups:1446`、`_group_liberty_count:1551`）已在。

### 2.3 22 通道实时算的性能闸门 —— 🔴 **C0 已实跑，原表整张作废**

**原表（作废）**：

| 项 | 原写数值 | 现状 |
|---|---:|---|
| NPU 侧吞吐需求 | ~**2635** 行/s | ⚠ 原写 `run.txt:700`（v18, 12ch）—— **`run.txt` 已改成「一屏」版、只剩 55 行，该行号悬空**；数字本身保留 |
| 预取 worker | **8** | ✅ 8 是实测最优点（见下） |
| **每行时间预算** | **3.04 ms** | 预算不变，但**本机没达到** |
| 现有 12ch 单盘基线 | 1.78 ms | ⚠ 原写 `go_rules.py:2479`，该行号已漂移 |
| ~~**22ch 余量 1.7×**~~ | ~~1.7×~~ | 🔴 **实测是 −0.2×（超预算 1.2×）** |

**实测**（工具 `scripts/bench_v7_features.py`，产物 `benchmarks/bench_v7_features.json`，
`generated_at = 2026-10-02T19:27:30`）。
**机器**：`AMD Ryzen 7 5800H`（8 物理核 / 16 逻辑）、物理内存 **14.2 GB**、
**数据盘 = USB（`JMicron Generic`）**、`boards.npy` = **12.35 GB**。
分层采样（前 1024 / 随机 2048 / 复杂度 top 1024），每层 × `B ∈ {32,128,512}`，reps=3。

**(a) 成本几乎全在 `iterLadders`**（`first` 层 `B=32`，1024 行）：

| 组件 | 秒 | 占 pipeline |
|---|---:|---:|
| `ladder_3` | **3.379** | **99.80%** |
| `_pipeline`（22 通道总计） | 3.386 | 100% |
| `liberties_123` | 0.0319 | 0.94% |
| `calculate_area` | 0.0137 | 0.40% |
| `global`（19 维） | 0.00040 | 0.012% |

⇒ 下面那句定性判断**完全成立**，现在有了数量：3 气桶 / 历史交错 / parity ≈ 0、
`calculateArea` 中、**`iterLadders` 贵 —— 且它就是全部成本**。

**(b) 🔴 batch 规模对特征吞吐几乎无影响 ⇒ 与 `--prefetch-workers` 解耦**
（`batch_curve`，随机 2048 行）：

| B | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| 行/s | **311.4** | 306.4 | 309.7 | 306.1 | 306.6 | **307.1** |

`max/min = 1.017` ⇒ **离散度 1.7%**（32 倍 batch 跨度只换来 1.7%）。
原因：Python 开销**长在每行的梯子 DFS 里面**，不在批级。
⇒ **这条解掉了 §4 门禁里原本的一个耦合**：`--batch-size` 纯按 NPU 显存 / 收敛选，
不必与 worker 数联动。

**(c) worker 扩展（`B=512`、2048 行、`spawn`）**：

| worker | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| **热读** 行/s | 341.2 | 594.7 | 827.8 | **1161.4** |
| **冷读** 行/s | 175.8 | **155.2** | 217.8 | **273.1** |

- 热曲线 8 worker **边际效率 0.701**；1→8 总加速 3.40×。
- 🔴 **冷/热差 4.3×**（8 worker：273.1 vs 1161.4），且
  **冷读 2 worker 比 1 worker 还慢**（155.2 < 175.8）—— 随机读被磁盘串行化。
- **gather 自身**：冷读 **10.4–24.7 ms/行** vs 热读 **0.004–0.008 ms/行**
  （两组 offset 的中位数 11.08/11.15 vs 0.0067/0.0050 ⇒ **1654× / 2230×**）。
  线程探针（1/4/8 线程）聚合吞吐 91.7→109.5 行/s 几乎不涨 ⇒ 脚本判「**存储瓶颈**」。
  ⇒ **fork 之前必须 warm。**

**(d) 单进程**（参考 `B=512`）：`first` 289.4 / `random` 301.5 /
**`complex` 178.7 行/s = 5.60 ms/行**。
🔴 **复杂盘面超 3.04 ms 预算 1.8×**。
⚠ **只测前 N 行会系统性低估**（开局空盘最快，`first` 与 `complex` 差 **1.6×**）。

**(e) 内存**：8 worker 单进程峰值工作集 **306.5–307.2 MB**
（B=32/128/512 几乎不变）⇒ **8 × 307 MB = 2457 MB ≈ 2.4 GB** ≪ 14.2 GB。

**(f) 判定 —— 🔴 不成立，且脚本不替我们圆场**：

```
budget 3.04 ms/行 | measured 3.6612（8 worker，冷读，部署态）| over_budget_by 1.2 | holds: false
```

**部署参数**：

| 参数 | 值 | 依据 |
|---|---|---|
| `--prefetch-workers` | **8**（本机拐点，边际效率 0.701） | (c)。⚠ 云端 24 核按「worker ≈ 物理核数」外推到 **12** —— 🔴 **未实测**（脚本只跑 1/2/4/8），引用必须带「外推」 |
| `--batch-size` | 与 worker **解耦** | (b) |
| **fork 前必须 warm** | 1 个 warmup batch | (c) |
| 采样 | **必须分层**，禁止只测前 N 行 | (d) |

新增五类通道的成本：3 气桶（≈0，复用已在跑的 flood）、历史 5 手交错（≈0）、
parity 三角波（≈0）、`calculateArea`（0.40%）、**`iterLadders`（99.8%）**。

> ⚠ **C0 的角色是标定，不是闸门。** 22 通道已定（§1.2）；C0 用来定
> `--prefetch-workers` 与 batch size。🔴 **但 (f) 说明它同时也是闸门**：
> 本机没过 ⇒ §4 的退路表里「C0 严重超预算」那一行**已触发**，
> 处置顺序是 **① `boards.npy` 挪到本地 NVMe（冷/热差 4.3×）→ ② 才降 B / 启用 18 通道**。
> ⚠ **不要**靠加 worker —— 冷读 2 worker 比 1 worker 还慢。

### 2.4 邻行 gather：5 个偏移，且几乎免费

| 偏移 | 用途 | 守卫 |
|---|---|---|
| `i−2` / `i−1` | ch16 / ch15（前一**二**手 / 前一**手**盘上的梯子） | `game_ids[i−k]==game_ids[i]` |
| `i+8` / `i+32` | futurepos 通道 0 / 1 | `game_ids[i+k]==game_ids[i]` |

**gather 本身几乎不花钱**（`boards` 常驻内存，就是索引）。

> 🎁 **futurepos 的实际 area 计算只有 4%。** 每局行数 p50=208 / mean=210.7：
> `+8` 有效 96.2%（`[0, L−9]`）、`+32` 有效 84.8%（`[0, L−33]`）。
> ⇒ 全部行都 gather，但**只有靠近终局的那 4% 行需要真跑一次 area 的 flood fill**。
> 实现上按行掩码分派：`valid8`/`valid32` 子集才算，其余 target 填 0、`w_futurepos` 置 0。

⚠ **ply 0/1 落回退复制分支**（`prevBoard = board`、`prevPrevBoard = prevBoard`），
**不是置 0**。写错不崩，只让开局几个位置的输入偏掉。

⚠ **`np.load('x.npz')['boards']` 会把整个成员解压进内存**（12.35 GB），不是按需。
本机 13.9 GB ⇒ 必须经 `kata_label_join.materialize_dataset()` 落成 `.npy`
后 `mmap_mode='r'`（⚠ 原写「落盘 152s」—— **`build_soft_index.py` 与 `dataset.py`
的 docstring 都记的是 72 s / 478K 行/s，以 72 s 为准**）。
🔴 **且 materialize 绝不能发生在 fork 之后**（会变成 12.3 GB × worker 数）
⇒ `warm_futurepos()` 是父进程 fork 前的入口，见 §3 A6 ⑥。

### 2.5 stdata 可用量

| 批次 | 网络 | 总行数 | 19×19 占比 | 19×19 可用 |
|---|---|---:|---:|---:|
| 2026-07-30 | `kata1-zhizi-b40c768nbt` | 1.09M | 75.3% | 0.82M |
| 2026-08-25 | `kata1-tf3-b11c768` | 3.45M | 63.1% | 2.18M |
| zzb28c512 | `zzb28c512nfd4` | 131K | 74.0% | 0.10M |
| **合计** | | **4.67M** | | **≈3.10M** |

⚠ **`globalTargetsNC` 列数随网络版本变**（64 vs 80），`qValueTargetsNCMove` 只有
前两批有 ⇒ 读取器**按网络名分派布局**，未登记的网络报错不猜（配套 spec §9.1）。
⚠ **`zzb28c512` 是 seki 富集批**（131,706 行里 104,905 行含 seki 格子，
另两批只有几十行）⇒ 当作 **seki 专项**单独用，不混进主训练。

### 2.6 SGF 语料的实测性质 ⇒ **决定哪些 loss 能在段 1 训**

抽样 133,604 个 SGF（**早期抽样，下两表的部分数字已被全量实测推翻，见 §2.6.1**）：

| `KM` | 占比 | | `RU` | 占比 | 计分 |
|---|---:|---|---|---:|---|
| 7.5 | 37.7% | | 无 RU | **55.0%** | 默认 AREA |
| 6.5 | 18.0% | | `Chinese` | 37.6% | AREA |
| 5.5 | 15.1% | | `Japanese` | **7.3%** | **TERRITORY** |
| 3.8 | 10.5% | | 空 `RU[]` | 0.0% | 默认 AREA |
| **0** | **5.9%** | | | | |
| 2.8 / 4.5 / 缺 | 9.3% | | | | |

| `RE` 模式 | 局数 | 占比 |
|---|---:|---:|
| `B+R` / `W+R` / `B+Resign` / `W+Resign` | 98,352 | **75.2%（认输，无分差）** 🔴 **已被全量实测推翻 → 81.09%** |
| `B<N>` / `W<N>`（如 `B+2.5`） | 24,748 | 18.5% |
| `B+T` / `W+T` | 271 | 0.2% |
| `draw` / `B+` / `W+F` 等 | 211 | 0.2% |

🔴 **两条硬结论：**

1. **75% 的局是认输** ⇒ 配套 spec §4.5 里 **5 项**（#5/#6 scorebelief、
   #8 scoremean、#9 lead、#10 scoring）吃 `w_score`，而 `g_score` 为 `NaN` 时
   `w_score=0` ⇒ **这 5 项在 75% 的局上权重为 0**。有分差监督的约
   `24.8% × 162,298 × 210 ≈ 6.6M 行`（34.2M 的 **19%**）。
   🔴 **订正见 §2.6.1：真实值是 81.09%，不是 75.2%。**
2. **贴目五花八门** ⇒ `g_komi` 必须逐局存，**不能假设常数 7.5**；`KM=0` 占 5.9%。

#### 2.6.1 🔴 订正（2026-10-02 实现期全量实测）：三条数字 + 锚点三个坑

> 本节把「当时以为是 A、实测是 B」的结论集中记一遍。
> 抽样数据保留在上面（它记录的是「抽样时看到了什么」），**以本节为准**。

##### ① `RE` 认输率 **75.2% → 81.09%**

`scripts/build_games_sidecar.py` 在**全部 162,298 局**上实跑（分母 = 入库局数）：

```
认输    81.09%  = B+R 21.94 + W+Resign 21.85 + W+R 21.75 + B+Resign 15.55
有分差  ≈18.5%  （g_score 可用）
```

**差 5.9 个百分点，方向是「比原以为的更糟」。**
原因：上面的抽样只扫 `data/games/games/` 的 4 个目录，
**漏掉了 5 个 tgz 里的 36,274 个 SGF**（§2.1 已核实重叠 = 0）。

⇒ 后果：score 系 5 项在 **~81%** 的行上权重为 0，
有监督的只有 `18.5% × 162,298 × 210 ≈ 6.3M 行`（34.2M 的 **~19%**）。

##### ② `KM` 有一个 **×100 家族**，原表未列

`data/games/games/foxpro` 写的是：

```
KM[650] [750] [550] [450] [375] [325]
```

÷100 得 6.5 / 7.5 / 5.5 / 4.5 / 3.75 / 3.25（全在合理区间）；÷2 得乱数
⇒ **×100 是唯一自洽的解释**。**2,163 局 = 1.62%**。

⚠ 照字面存 ⇒ 全局 ch5 `currentSelfKomi/20` = `650/20` = **`32.5`**，
比任何真实贴目大一个量级，模型没有任何机制能拉回来。
⇒ `build_games_sidecar.py::parse_komi` 在 `fix_outlier=True`（**默认开**）时
把 `|v| > KOMI_OUTLIER_ABS (=100.0)` 除以 100；
关闭开关是 `--no-komi-outlier-fix`（**别顺手加上**）。
报告里 `×100 离群修正` 一行打印条数；
钉住：`tests/test_games_sidecar_alignment.py::test_komi_outlier_fix_is_opt_out`。

##### ③ 两个**未列**的 `RE` 形式（各 ≤2 局，但都有显式分支）

| `RE` | 局数 | 语法 | 分支 |
|---|---:|---|---|
| `W+3 zi` | **2** | 单位后缀（`zi` = 子数） | 剥后缀取数值 |
| `W+0,25` | **1** | 欧洲逗号小数 | 逗号改点取数值 |

样本量极小，但两者都会静默落到「未识别 ⇒ `NaN`」分支，而它们**本该有分差**
⇒ 方向性错误，**必须有显式分支**。

##### ④ 哈希锚点的三个坑（D0c 会踩，见 §3 的 D0c 订正）

1. **`game_ids` 是置换不是升序** —— 162,298 段对 162,298 个 distinct id，
   但**段的顺序 ≠ id 顺序** ⇒ 切段**必须用相邻比较**
   （`game_ids[1:] != game_ids[:-1]`），**不能假设 `np.diff` 的符号**。
2. **偏移必须从 1 起** —— 偏移 0 是**空盘**，**169,878 个 SGF 完全相同**
   ⇒ 任何 `L < 2` 的残局会锚到**任意一局**，且「锚到的也是残局」时看不出来。
   `build_dataset` 拒绝 `< 2` 手的局 ⇒ `min(20, L//2) ≥ 1` **恒成立**。
3. **`min(20, L//2)` 对 134 局 ≠ 20** —— **134 段 `L<40`**、**2 段 `L<20`**
   ⇒ 硬编码 ply-20 会**静默丢掉**它们
   ⇒ 改为对每个 SGF 存**偏移 1..20 的全部前缀**散列（**3.4M 散列 = 27 MB**）。

### 2.7 🔴 语料性质决定 loss 的启用顺序

认输局即使重放到终局，`go_rules.py:2269` 的 docstring 自陈「**没有死子判定**，
对局双方应在 pass 认输前实际提掉对方的死子，否则死子所在点会被判为双方共邻的
中立区域」⇒ **从认输棋谱反推的终局归属不可靠**。

⇒ **段 1 的 loss 启用范围**（不是「全都开、让权重自己决定」）：

| 启用 | 项 | 依据 |
|---|---|---|
| ✅ | #1 policy、#2 π_opp | 目标来自 `moves` one-hot，100% 覆盖 |
| ✅ | #3 value 3 类 | `winrates` 免费给（§2.2），100% 覆盖 |
| ✅ | #11 futurepos | `boards[i+8]/[i+32]` 实时算，96.2% / 84.8% 覆盖 |
| ⚠ **不作为主目标** | #5/#6/#8/#9/#10（score 系） | 🔴 **`g_score` 覆盖率仅 ≈18.5%**（§2.6.1）⇒ **段 1 整组关闭**，而不是跑 18.5% 覆盖 |
| ❌ | #4 ownership、#10 scoring、#12 seki | **不从 SGF 造**（§3 D0b 已砍），段 2/3 由 stdata 提供 |

⇒ **段 2/3（1% 标注 + stdata 3.1M）才开全部 12 项** ——
那 3.4M 上这些信号来自**搜索器自己的评分器**（`valueTargetsNCHW`），
质量比从认输棋谱反推高得多。

> 这条把 ownership/scoring/seki 的训练责任从 **34.2M 挪到 3.4M**，
> 是本 spec 相对「照搬 12 项全开」最重要的修正。

🔴 **实测支撑（2026-10-02 补）**：这个范围此前只写在 decision 里，spec 里只有
「75%」这个**抽样**数字。换成全量实测的 **81.09%** 之后，三条结论都变硬了：

| 项 | 原以为（75.2% 抽样） | 实测（81.09% 全量） |
|---|---|---|
| score 系（#5/#6/#8/#9/#10）的 `w_score=0` 覆盖面 | 75% 的局 | **~81% 的行** |
| `g_score` 覆盖率 | ≈24.8% | **≈18.5%**（6.3M / 34.2M 行 = ~19%） |
| 段 1 要不要训 score | 「19% 覆盖，偏少」 | **不训**（差 5.9 个百分点，方向是更差） |

⇒ **段 1 只训 policy、π_opp、value、futurepos 四项**；
score 系**不作为段 1 主目标**；ownership / scoring / seki **不在段 1**，
交给段 2/3 从 stdata 的 `valueTargetsNCHW` 取（那 3.4M 上 100% 有覆盖，
且标签来自搜索器自己的评分器）。

---

## §3 四组工作

```
A 组 · 软标签 ────────┐
C0  · 特征基准 ───────┼─ 零依赖，可同时推进
D0  · 局级 sidecar ───┘
                          ↓
B 组 · 22 通道实时算（依赖 C0 + D0）
```

### A 组 · 软标签（与通道数无关）

```
A1  labels=True → 4 元组 + dict（soft / soft_mask 放进 dict）✅
    + 3 个假 dataset 补 labels=False 形参
       tests/test_eval_coverage.py:75 / test_eval_determinism.py:111
       / test_eval_no_augment.py:219
    → tests/test_dataset_soft_labels.py（28 个 test 函数）✅
A2  compute_policy_loss 软 CE 分支 ✅ —— 🔴 掩码语义见下方订正
    → tests/test_soft_ce.py（26 个 test 函数）✅
A3  预取器 / worker 透传 dict ✅
    → tests/test_prefetch_labels.py::test_soft_mask_and_future_survive_
             concatenation_of_uneven_blocks
      （⚠ 原写的 test_soft_prefetch.py 不存在）
A4  CLI 四旗 ✅ —— 🔴 名字见下方订正
    --soft-index / --soft-weight / --soft-only-sampling / --soft-every
    → tests/test_soft_cli.py（28 个 test 函数）✅
A5  join 缓存 —— 🔴 实现在 kata_label_join.py，不在脚本里（见下方订正）
    key = (主 npz 行数/distinct game_ids/mtime/size,
           标签 npz 行数/distinct pos_hash/mtime/size,
           max_repeats,
           散列口径版本号 + 从 pos_hash 活常量现算的指纹)
    → tests/test_join_cache_freshness.py（25 个 test 函数）✅
       （⚠ 原写的 test_soft_cache.py 不存在）
A6  futurepos 邻行 gather —— 🔴 **本 spec 原本完全没有这一项**，见下方 A6 段
    → tests/test_dataset_futurepos.py（28 个 test 函数）✅
```

> 🔴 **订正一：A4 的旗名与形态（清单里写错了两个地方）。**
> 原写 `--soft-labels / --soft-weight / --soft-index`，实际是
> **`--soft-index` / `--soft-weight` / `--soft-only-sampling` / `--soft-every`**
> （`scripts/train_sft.py:2362-2390`）—— **没有 `--soft-labels`**。
> ⚠ `--soft-only-sampling` 用**本仓统一形态** `type=int, choices=[0,1]`、
> `default=0`（与 `--use-amp` / `--swanlab` 同），**不是 `store_true`**
> ⇒ 判断它必须写 `if args.soft_only_sampling:` 而不是 `is True`。
> `--soft-every` 用 `type=_at_least_one`（**取值必须 ≥ 1**，
> 写 `--soft-every 0` 会在 argparse 阶段就报错，而不是到训练里
> 变成 `ZeroDivisionError`）。

> 🔴 **订正二：A4 已做完，「CLI 冻结」这条约束已经履行，不是待办。**
> `tests/test_huber_loss.py::test_no_new_cli_params` 以 **D1「零新增 / 零删除 /
> 零改名」**冻结 `scripts/train_sft.py` 的 flag —— 🔴 **冻结集已从 61 扩到 65**
> （61 + A4 的 4 个）。它还有第二道关：全模块扫 `add_argument`，
> **调用数必须恰好 = 65**，且**每个首参必须是字面字符串**。
> `test_policy_loss_default_is_ce` 现在钉**两件事**：
> ① `--policy-loss` 的 **default 仍是 `'ce'`**；② choices `== {huber, ce, soft_ce}`。
> ⚠ 原写的「choices 钉死成 `['huber','ce']`」**已作废**（A4 把逐字比较改掉了）。
>
> ✅ **`soft_ce` 已接进 CLI。** 接线方式是**派生**而不是新增旗：
> `scripts/train_sft.py::resolve_policy_loss_kind` —— 给了 `--soft-index`
> 就把生效口径派生成 `soft_ce`；没给就原样返回 `--policy-loss`（段 1 数值逐位不变）。
> 两条硬报错：① 显式 `--policy-loss soft_ce` 而**无** `--soft-index`；
> ② `--policy-loss huber` + `--soft-index`（huber 不消费软标签）。
>
> ⇒ **「段 2 现在无法从命令行启动」这句话作废。**
> 🔴 **但段 2 的正确命令行是 `--soft-index` + `--soft-only-sampling 1`**
> （§1.2 的分母论证：`mask=0` 贡献恰好 0 ⇒ 分母恒为 `B`
> ⇒ 不采样就只有 1% 的行贡献，policy 项被缩小 ~100×）。
>
> ⚠ **同类约束仍在**：日志的三个键名被 `test_log_keys_unchanged` 钉死；
> 后续再加旗仍必须同步登记冻结集，否则 CI 红且看不出原因。

> ⚠ **A1 是契约变更，必须最先做。** 现状（`47336ce`）是 5 元组
> `(states, moves, values, soft, mask)`，**没有 dict 的位置**；而 V7 的 loss
> 需要配套 spec §5.5 的那个 dict ⇒ 最后只会变 6 元组或 fork 两条路。
> 4 元组让预取器只需搬运**一种** payload 形状。

> ⚠ **`permute_soft` 是软标签接入里唯一会静默出错的地方**（已实现并测，8/8）。
> `states` 施加增强、`moves_out` 施加重映射，软策略不同步重排则**训练不报任何错**，
> 只是标签指向错误的点 ⇒ 表现为「loss 正常下降、指标好看，但棋力不涨」。
> 已踩过的坑：方向极易搞反，`out[:, perms[t]] = soft[:, :]`（gather）得到的是
> **逆变**；正确是显式求逆 `inv[perms[t][k]] = k` 再取 `out[:, :A-1] = soft[:, inv[t]]`。

> ⚠ **A5 的缓存 key 必须防陈旧。** 换一批 SGF 或换标签文件后旧缓存会**静默挂错
> 标签**，而症状是「命中率 0」，看不出是缓存陈旧。
> 🔴 **落点订正**：本 spec 把 A5 记成「`scripts/build_soft_index.py`」——
> 那个脚本里**原本一个字节缓存都没有**，它每次都全量重扫（**实测 127 s**）。
> 带防陈旧的实现落在 **`src/data/kata_label_join.py`**（**join 的唯一实现处**，
> `build_soft_index` + `hash_spec_fingerprint`），脚本**只做 CLI 委派**。
> ⇒ 判定「缓存是否新鲜」的逻辑**只有一处**，不要在脚本里再抄一份。

##### 🔴 A6 · `futurepos` 邻行 gather（本 spec 原本完全缺失这一项）

§2.4 写了「futurepos 的 area 计算只有 4%」，但**契约一条都没写**。
实现期定下的六条，每一条都是「不写下来就会被静默搞错」的那类
（完整论证见配套 spec §9.12.1；`tests/test_dataset_futurepos.py` 28 个 test 函数钉住）：

| # | 契约 | 为什么承重 |
|---|---|---|
| ① | `future[b,h,:] ∈ {0,1}` = 该点是否被**行 i 的 `to_play` 的对手**所占；**「对手」按 `to_play[i]` 定，不是 `to_play[j]`** | +8/+32 **都是偶数** ⇒ 两种读法在真实数据上**恰好一致** ⇒ 不主动构造反例的测试是**空断言**。只能靠把定义写死 |
| ② | 🔴 **无效行哨兵 = `-1.0`（不是 0）** | 「取值域 ∈ {0,1}」与「不能静默全 0」在纯 `{0,1}` 下**不可兼得** —— 全 0 恰好是「对手一颗子都不占」这个**合法标签** ⇒ 哨兵必须落在域外 |
| ③ | 两个 horizon 走**同一个** `tforms` | 否则 `states` 转 90° 而 h1 没转 = 标签与输入不同源，**不报任何错** |
| ④ | 🔴 **`w['futurepos']` = h0 AND h1** | loss 把 `(b,2,bs²)` 压成**一个标量**且 `_weighted_mean` **不除 `Σw`** ⇒ 权重的最小作用单位是**整块**；只活一路时给 1，死路的 `-1` 会被当真值拟合 ⇒ 头学出恒 `−0.76` 的假平面。单路有效性**只能**由 `w['futurepos_h0/h1']` 表达 |
| ⑤ | **opt-in only**；关闭时 payload **逐字节相同** | 是 `tests/test_dataset_soft_labels.py:510` 与 `test_prefetch_labels.py:464-466` 两条**既有**测试逼出来的（否则 A6 一落地 CI 就红，而红的原因与「A6 坏了」长得一样） |
| ⑥ | 🔴 **12.3 GB 的 materialize 绝不能发生在 fork 之后** | 会变成 12.3 GB × worker 数（还会同时写同一路径互踩）⇒ **`warm_futurepos()` 是父进程 fork 前的入口**，`_futurepos_boards()` 里的惰性解析**只是兜底**（幂等） |

⚠ **A6 的接线还差一步**：`scripts/train_sft.py` **还没调 `warm_futurepos()`**
（契约已写在 `dataset.py` 的 docstring 里，调用方缺席）。
⚠ 顺带修正 §2.4 末尾：落盘实测是 **72 s / 478K 行/s**（不是 152 s）——
`build_soft_index.py` 的 docstring 与 `dataset.py` 都记的是 72 s。

### C0 · 特征基准（标定） —— 🔴 **已实跑，四个数都在 §2.3 (a)–(f)**

测四个数，输出到文档：

1. `iterLadders` 单盘 ms/行 → **3.30 ms/行（99.80% 的成本）** ✅
2. `calculateArea` 单盘 ms/行 → **0.013 ms/行（0.40%）** ✅
3. 22ch 完整 vs 1.78 ms 基线 → **3.31 ms/行，🔴 基线被推翻** ✅
4. 8 进程并行的有效行/s → **热 1161.4 / 冷 273.1** ✅

工具 `scripts/bench_v7_features.py`，产物 **`benchmarks/bench_v7_features.json`**
（⚠ 原 spec 的文件清单里**两个都没有**）。
⇒ `--prefetch-workers` = 8；**batch 与 worker 解耦**；**fork 前必须 warm**。
完整数字与推导见 **§2.3**。

### D0 · 局级 sidecar `games.npz`（≈1.2 MB，不碰主 npz）

```
D0a memmap game_ids → G = 162,298 个行区间（⚠ **相邻比较**切段，见 §2.6.1 ④）
D0b **只扫 KM / RU / RE 三项**（不做终局重放）
    · g_komi    ← SGF KM（已解析）——⚠ 贴目分布很散 + ⚠ **×100 家族**，见 §2.6 / §2.6.1
    · g_score   ← RE 数值解析（新增）——🔴 **81.09% 是认输**，见 §2.6.1 / §2.7
    · g_rules   ← 新增 RU 解析；无 RU 按 AREA/none/simple/false/0 默认
    · g_resign  ← RE 形如 B+R/W+R/B+Resign/W+Resign/B+T
D0c 哈希锚点匹配（⚠ **偏移 1..20 全前缀**，不是 ply-20）→ games.npz
    → test_games_sidecar_alignment.py（匹配率必须打印）
```

> 🔴 **D0b 原本还要重放到终局算 `g_ownership`/`g_scoring`/`g_seki`
> （8–28 min + 两个最难移植的算法），2026-10-02 决定砍掉。** 理由三条：
>
> 1. **性价比低**：那笔开销只买到 **2.5 项** loss，而 #10 已因分差缺失在
>    **~81%** 的行上废了（**实测**，原写 75%）。
> 2. **信号质量反而更差**：`go_rules.py:2269` 自陈「没有死子判定，对局双方应在
>    pass 认输前实际提掉对方的死子」⇒ 从**认输**棋谱反推的终局归属不可靠。
> 3. **段 2/3 有更好的来源**：`stdata` 的 `valueTargetsNCHW` 直接带
>    ownership / seki / futurepos / scoring，**不需要移植任何东西**。
>
> ⇒ sidecar 从 **176 MB 降到 1.2 MB**，D0b 从「两次 SGF 扫描 + 终局重放」
> 降到「一次 SGF 扫描取三个字段」，墙钟从 8–28 min 降到**几分钟**。

⚠ **D0c 的哈希锚点法是「不 rebuild」唯一的技术风险点。**
不能用「重新枚举 SGF 按同样顺序」：`build_dataset.py:261-279` 的过滤实测丢掉
**45%** 成员，`tarfile` 顺序 ≠ glob 顺序；且 `:314-316` 的 **id 复用 bug**
（`board.play()` 失败时 `return 0,1` 但已追加的行留在 `cur`、
`game_id_counter` 未自增）让「局数」本身对不上。
锚点法**对枚举顺序与该 bug 完全免疫**。

🔴 **订正：「100% 的局 ≥20 手保证锚点可用」这句话的**结论**对（ply-20 不安全），
但**诊断是错的** —— 「162,296 / 162,298 = 100%」那个分子**就是**实测的
`L ≥ 20` 段数（**99.9988%**，把 99.9988% 写成了 100%）。
真正的风险**不是**「有 2 段 `L<20`」，而是坑 3：`min(20, L//2)` 对
**134 段 `L<40`** ≠ 20 —— **即使 100% 的局都 `L ≥ 20`，坑 3 照样存在**
（`L ∈ [20,40)` 的那 132 段取不到 20）。
全量实测把它换成**三个具体的坑**（详见 §2.6.1 ④，这里只列落地要求）：

| # | 坑 | 落地要求 |
|---|---|---|
| 1 | `game_ids` 是**置换**不是升序 | 切段用**相邻比较** `game_ids[1:] != game_ids[:-1]`；**不能假设 `np.diff` 的符号** |
| 2 | 偏移 **0 是空盘**，169,878 个 SGF 完全相同 | 偏移**从 1 起**；`build_dataset` 拒绝 `<2` 手 ⇒ `min(20, L//2) ≥ 1` 恒成立 |
| 3 | `min(20, L//2)` 对 **134 段 `L<40` / 2 段 `L<20`** ≠ 20 | 存**偏移 1..20 的全前缀**散列（**3.4M = 27 MB**），硬编码 ply-20 会静默丢段 |

⚠ **必须输出覆盖报告**（且比原来要求的更细）。匹配率 < 100% 时，未匹配的局
按 `scripts/build_games_sidecar.py` 的兜底值填：**`g_komi = 0`、
`g_score = NaN`**（⇒ `w_score = 0`）、`g_rules = RULES_DEFAULT`、`g_resign = False`，
**不静默填垃圾**。⚠ 原写的「相关权重置 0」指向 `w_ownership`/`w_seki`/`w_scoring`，
而那三个字段**已被 D0b 砍掉**（sidecar 现在只有 4 个键）⇒ 已按实际兜底值改写。
报告还须打印**歧义**（一段命中多个前缀）、
**跨局命中**（同一局面被不同的棋走到 ⇒ 取的未必是本局的 `KM/RE/RU`）与
**id 复用冲突**（本数据集应为 0，非 0 说明坑 1 的切段写错了）。

### B 组 · 22 通道实时算 —— 🔴 **已全部落地**（✅ 除 B8）

> ⚠ **两个文件名在这一节原写错了**：模块落点是 **`src/data/feature_v7.py`**
> （+ `feature_v7_ladders.py` + `feature_v7_gather.py` 三个模块，
> **不是**原设计的单个 `src/game/go_features_v7.py`）——
> 因为 `iterLadders` / 邻行 gather / 装配三件事各有独立的**纯函数可测性边界**。

```
B1  _check_n_channels 放行 22 + feature_planes_batched 加 V7 分发 ✅
B2  feature_v7：3 气桶 + 历史 5 手交错        ← 纯函数  test_v7_planes.py ✅
B3  feature_v7：calculateArea → ch18/19       ← 纯函数  test_v7_area.py ✅
B4  feature_v7_ladders：iterLadders → ch14/17 ← 纯函数  test_v7_ladders.py ✅
B5  19 维全局装配 + futurepos 掩码分派（只算 4%）  test_v7_globals.py ✅
B6  feature_v7_gather：邻行 gather（5 偏移，批量，绝不逐样本循环）
    ⚠ 原写 test_v7_neighbor_gather.py **不存在** ⇒ tests/test_v7_gather.py（19 个 test 函数）✅
B7  交叉校验 —— 🔴 **覆盖面比原写的小得多**（见下方 B7 三段）
    tools/scripts/crosscheck_stdata.py + tests/test_crosscheck_tool.py
    tests/test_v7_assemble.py（_PINNED_EXACT）
    ⚠ 原写 test_v7_vs_official.py **不存在**
B8  train_sft.py 切 22 通道 —— ❌ 未做（软标签 A4 已接）
```

**B2–B4 三个纯函数在 B6 接线之前就能各自验**；B6 只是把三块盘面喂进去 +
处理回退复制。这是把 B 组内部再切一刀的原因：`iterLadders` 是唯一
「既慢又可能移植错」的项，不该和接线耦合。

⚠ **B4 的验证必须拆成两半**（`katago/` 下只有 Windows 二进制，**无 `cpp/`**，
`nninputs.cpp` 不可读 ⇒ stdata 是**ladder 通道**唯一可执行 oracle；
⚠ 但它**不是** ch18/19 的 oracle，见下）：

| 通道 | 依赖盘面 | 能否对 stdata 直接验证 | **实测结果** |
|---|---|---|---|
| ch14 | 当前盘 | ✅ | ✅ **1.000000**（200 npz / **4,368** 行；另有 12 / 301、40 / 776 两档全对；非零格 ours **22,528** == official **22,528**） |
| ch17 | 当前盘（working-move） | ✅ | ✅ **1.000000**，**但只在 `to_play = +1` 下**（`−1` 假设实测 **0.186813** / 200 npz / 4,368 行）⇒ **stdata 隐含 `to_play` 恒 +1 得到证实** |
| ch15 | 前一手盘 | ❌ **stdata 是逐行独立样本，没有前一盘的盘面** | — |
| ch16 | 前二手盘 | ❌ 同上 | — |

⇒ 「**算法**」由 ch14 对 stdata 逐位担保（≈3.1M 个 19×19 行，样本量足够；
**实测已达 1.000000**）；「**盘面**」由 SGF 重放担保（纯本地测试）。
ch15/16 复用同一份已验证代码，只换喂进去的盘面。
**不需要 KataGo 源码也能把 4 个通道的可信度拆开验证。**

> 🔴 **ch15/16 的归类是 `NOT_COMPARABLE`，而且有个「巧合」必须记下来。**
> 对拍工具在 301 行上给 ch15 / ch16 报出 **`0.7276` / `0.7076`** ——
> 🔴 **那不是对齐率**：我们在这两格上走的是**官方自己的回退复制分支**
> （`prevBoard = board`），所以拿它去比官方 ch15，**实际对齐的是 ch14**
> （ch14 的实测是 `1.000000`）。
> ⇒ 归 `NOT_COMPARABLE`、**不归 `ALIGNED`**：归 ALIGNED 会让 `--min-rate 1.0`
> **永远过不了**（它到不了 1.0），等于把一个永不满足的门槛写进门禁。

⚠ **ch14 与 `to_play` 无关是实测，不是假设** —— 两种假设下**都**实测 `1.000000`。
⇒ 引用「ch14 不看 `to_play`」时**不要**再加「假设」二字。

##### ✅ ladder 算法的两个**反直觉**性质（B4 的测试会撞上）

1. **远处的子会改变 ladder 判定。** 2 气的 ladder 会跑**完整追逃**
   （遍历棋盘）才提子 ⇒ 盘面上任何看似无关的子，只要落在**逃跑延长线**上，
   就**真的**改变结论。⚠ 测试里放「远处无关的诱饵子」会让预期落空 ——
   **那不是泄漏，是算法的真实性质。**
   实测：白被追到 **(16,17)** 才被提掉（**34 个搜索节点**），而诱饵表里那颗
   (16,17) 白子**正好在逃跑延长线上** ⇒ **把 (16,17) 从诱饵里挪走**，
   **不是**调阈值迁就实现（诱饵的唯一职责是「不在被验对象自己的路径上」，
   落在线上是**诱饵表写错了**）。挪走后候选块仍是 12 块。
2. **保留的 C++ 上游怪癖（故意不「修」）**：
   `boundNumLibertiesAfterPlay` **不去重** 而 `findLibertyGainingCaptures`
   **去重**（上游自己就不一致，`lowerBound` 的语义依赖这个不一致）；
   1 气分支**不清** `workingMoves`。
   ⇒ **`libs > 1` 的守卫是 ch17 真正的兜底**，不是冗余判断。

##### 🔴 B7 之外：**ch18/19 不能与 stdata 对拍**（B3 的验收口径要改）

逐位对拍 **1,254 行**真实 19×19 局实测：

| 指标 | 实测 |
|---|---:|
| 我们比官方**多**的点 | **21,896**（**19,913 子** + **1,983 空点**） |
| **完全相同的行** | **0 / 1,254** |
| 受影响的行 | **295 / 1,254** |

最小反例：一个被黑子四面围住的**单个空点**，按 Tromp-Taylor 归黑，**官方给中立**；
且缺失的子**不集中在 1 气的块上**（否掉了「只是没提死子」这个单一解释）。

⇒ 🔴 **订正：stdata 的覆盖面比这行原话小得多。**
原写「stdata 只能当 ch3/4/5、**ch9–13**、ch14/ch17 的 oracle」。
`scripts/crosscheck_stdata.py` 给每个通道打**三档资格**，
**ch9–13 归 `NOT_COMPARABLE`** —— 官方数据是**逐行独立样本、无着法序列**，
重建「过去 5 手落点」需要序列本身 ⇒ **结构上到不了**。
真正的 oracle 只有 **空间 ch0–6 / ch8 / ch14 / ch17**（+ 全局规则位那几个）；
**ch18/19 是 `KNOWN_DIVERGENT`**（判据已知：官方先提死子）。
官方很可能在算 area 前**先提掉死子**，而本仓 `GoBoard.score()` 没有死子判定；
判据要猜，猜错会得到**第三种**偏差 ⇒ **按对齐口径优先，没有偷偷补死子判定。**
完整三档表见配套 spec §9.12.2。

**B3 的验收因此不是「与 stdata 逐位相等」，而是**：本地口径
（`SCORING_AREA && TAX_NONE` 下与 Tromp-Taylor 等价 + 逐条对照
`fillRowV7` 源码引文），**明确没有可执行 oracle**。

**B7 已实测达成的部分**（`tests/test_v7_planes.py`）：

| 通道 | 样本 | 结果 | 资格 |
|---|---|---|---|
| 空间 ch3 / ch4 / ch5 | 1,254 行 | **1.000000** ⇒ 气桶**不分色**、桶是 **`==1/2/3`**（不是 `>=3`）、整块去重口径正确 | `ALIGNED` ✅ |
| 空间 ch0–2 / ch6 / ch8 | `test_v7_assemble.py::_PINNED_EXACT` 钉住 | `ALIGNED`（ch1/ch2 是绝对色，`to_play ≡ +1` 时与官方 pla/opp 同解） | `ALIGNED` ✅ |
| 空间 ch9–13 | 1,254 行 | ch9 **100%** 落在 opp 子上、ch10 **99.4%** 落在 pla 子上、ch11/12/13 = **99.4 / 97.9 / 96.7%**；差额是**落子被提掉**（该点现在为空，两边都判 0）⇒ 顺序 `opp, pla, opp, pla, opp` 属实（**对调会掉到个位数**） | 🔴 **`NOT_COMPARABLE`** —— 这是**落点身份统计**，**不是对齐率**，验收上不得当作已对齐 |
| 空间 ch7 / ch20 / ch21 | 301 行里恰好 **1 行** `encorePhase=2` | 只在那 1 行不一致 | 🔴 **`NOT_COMPARABLE`** —— 官方**只在 encore 行**置位，而 `spatial_channels_v7` **没有 `encore_phase` 入参** ⇒ 结构上到不了 |
| 全局 15 / 16（PDA） | 🔴 **301 行上官方恰好恒 0 ⇒ `row_exact = 1.000000`（巧合）**；200 npz / **4,368** 行上官方有 **133 个非零点 ⇒ 0.9696** | 本仓无 PDA ⇒ 恒 0 | 🔴 **`NOT_COMPARABLE`** |

⚠ **由此得到两条纪律**（`scripts/crosscheck_stdata.py` 的 docstring 原文）：

1. 🔴 **`cell_agree` 会骗人**：ch9 的 `row_exact = 0.000000` 而
   `cell_agree = 0.9972`（每行只差 1 个点 / 361）⇒
   **汇报 ch9 = 99.7% 对齐 = 把一个彻底错的通道说成几乎完美。**
   脚本两个都印，但**门禁只看 `row_exact`** 且**只对 `ALIGNED` 断言**。
2. 🔴 **「出现了 `1.000000`」不等于「对齐了」**：全局 15/16 在 301 行上就是
   `1.000000`，那只是小样本巧合。**任何 `1.000000` 都必须连样本量和资格一起引用。**
3. `_PINNED_EXACT = (0,1,2,3,4,5,6,8,14,17)` **逐元素等于**对拍工具的空间
   `ALIGNED` 集合 —— 同一个判据的两处落地（一条在测试、一条在工具）。

⚠ **归档缺失时脚本「报错退出」，而 pytest 里是 `skipif`「跳过」** ——
刻意的差异：工具的职责是「回答一个问题」，归档不在就回答不了，静默退出 0
等于假装成功；测试的职责是验代码、不是保证归档存在。
`tests/test_crosscheck_tool.py` 用**合成假归档** + **真的**
`spatial_channels_v7` / `global_features_v7`，秒级且装配写错照样红。

⚠ **B2/B7 的历史门控另有一条实现纪律**（见配套 spec §2.2 的警示）：
`feature_v7_ladders.py` 曾把门控**解析两次** —— 函数内算对了，输出行独立重算
并抄了 `cur[:,0]`（ch14）而不是 `out[:,1]`（ch15），**`history=1` 时恰好差一手**。
**门控只能解析一次，只调一处实现**；测试断言的是**通道间的相等关系**，不是绝对内容。

---

## §4 门禁

每组单独过（沿用 `v21-roadmap.md:364`）：

```
python -m pytest tests -q && bash -n shell/*.sh
```

**当前基线：`1011 passed / 2 skipped`，`bash -n` exit 0。**

执行顺序：

```
1. A1  契约（最小、解锁一切）                      ✅
2. A2  软 CE + 单测                                ✅
3. C0  特征基准 → 记录四个数                       ✅ 已实跑（§2.3）
4. D0  sidecar（写脚本后尽早起跑）                  ✅ scripts/build_games_sidecar.py
5. A3 / A4 / A5  预取 / CLI / 缓存                ✅（🔴 段 2 可从命令行启动）
6. B1–B7  22 通道                                  ✅
   B8  train_sft.py 切 22 通道                      ❌ 未做
   A6  futurepos 的 warm_futurepos 接线             ❌ 未做（契约已定，§3 A6）
```

---

## §5 退路表

| 风险 | 退路 | 代价 |
|---|---|---|
| 🔴 **C0 严重超预算 —— 已触发**（§2.3 (f)：8 worker 冷读 **3.66 ms/行** vs 预算 3.04） | **顺序不能反**：① `boards.npy` 挪到**本地 NVMe**（冷/热差 **4.3×**，热态 0.86 ms/行 有 3.5× 余量）；② 才降 B；③ 才启用 18 通道降级。⚠ **不要**只提 worker —— 冷读 2 worker 比 1 worker **还慢** | 步时增加，结构不改 |
| 🔴 **`prefetch-workers` 外推到 12（云端 24 核）** | 🔴 **未实测**（脚本只跑了 1/2/4/8），16 核上 8 已到边际效率 0.701 的拐点 | 引用时必须带「外推」二字，别当实测 |
| ~~ch14 对 stdata 对不上~~ → **已排除** | 实测 **1.000000**（200 npz / 4,368 行）；退路（降到 **18 通道**）**保留未启用** | 若真触发：主干 / 四头 / 12 项 loss / 参数量 **5,561,832 一行不改** |
| 锚点匹配率 < 100% | 输出覆盖报告；未匹配局 `g_komi`/`g_score` 置 `NaN`（⇒ `w_score=0`）。⚠ 原来写的 `w_ownership`/`w_seki`/`w_scoring` **已不存在**（§3 D0b 砍掉了这三个字段） | 那部分行只损失 score 系 |
| ~~`calculateArea` 语义有差 → B7 暴露~~ 🔴 **已实测：确实有差，且不可对齐** | **ch18/19 已确认不能与 stdata 对齐**（1,254 行 0 行相同、多 21,896 点）⇒ 验收口径改为本地（Tromp-Taylor 等价 + 源码引文复核），**无 oracle**；**不补死子判定** | territory 局 ch18/19 恒 0（对齐官方，非缺陷）；模型少一个可用 oracle |
| 预取跟不上 | 提 worker 数或降 B | 步时增加 |
| 段 2 混合训练（未加 `--soft-only-sampling` 就开跑） | 立即停 —— 🔴 **不只是「index 0 向两种语义折中」**：`mask=0` 的行贡献**恰好 0**、分母**恒为 `B`** ⇒ 只有 1% 的行贡献 ⇒ **policy 项被缩小约 100×** | 不学搜索；而且是**静默**的（loss 照降、曲线正常） |
| 🔴 **加 flag 让 CI 红** | `test_no_new_cli_params` 把 `train_sft.py` 的 flag 整个冻结（D1 零新增/零删除/零改名，**冻结集现为 65**）；`test_policy_loss_default_is_ce` 钉 `--policy-loss` 的 **default = `ce`** 且 choices `== {huber, ce, soft_ce}` | ✅ **A4 的 4 个旗已登记**（`tests/test_soft_cli.py` 27 项）。⚠ 后续新旗仍必须同步登记，否则 CI 红且看不出原因。⚠ 原写的「choices == `['huber','ce']`」**已作废** |
| 🔴 **`scripts/train_sft.py` 没调 `warm_futurepos()`** | A6 开着 futurepos 训练时，12.3 GB 的 materialize 可能落在 **fork 之后** ⇒ **12.3 GB × worker 数**。先在父进程调 `warm_futurepos()`（幂等） | 内存爆掉；症状是 OOM 而不是任何语义错误 |
| 🔴 **run.txt 里的数字/flag 与实测值失去自动比对** | `test_run_txt_sync.py`（整文件）与 `test_param_budget.py::test_run_txt_records_are_consistent` **已删**（用户决定；后者要求 run.txt 保留 `# v18 架构参数` 小节，而 v18 已退役，现役是 V7 / `KATAGO_SE_CFG` 12ch） | 补参数时**必须人工核对** run.txt。这两条当初正是**为了抓「argparse 22 → 60 个 flag 无人同步」**而写的 |

---

## §6 悬而未决

| # | 事项 | 影响 | 状态 |
|---|---|---|---|
| 1 | **`scorestdev` 的 `beta`**：配套 spec §4.2 的 `SoftPlus(x₁, 0.05)` 让预测初值落在 277，而 loss #7 的目标是 5~20 量级、δ=10。冒烟训练 40 步该项 **−0.0%**，硬算需 ~9 万步才挪到位 ⇒ **等效于没有学习信号**。改成 `beta=1.0`（初值 13.86）是**一行**，但那是改已批准的 spec | 配套 spec 的 12 项之一失效 | 🔴 **未决，且建议未获批准**。当前代码**逐字实现 `0.05`**（`src/networks/katago_v7.py:534`）。已提为具名常量 `SCORE_STDEV_SOFTPLUS_BETA`，改一行不影响结构/参数/预算。**不得写成「已定」或「已修」，也不得把 #7 排进可训练项** |
| 2 | **`col3` / `col22` 的精确语义**（stdata 的 `globalTargetsNC`） | 目前**不作为任何 loss 的目标**，只记录 | 需查 KataGo `dataio.cpp` 写入顺序 |
| 3 | **2D RoPE 的 `(H,16,2)` 里那个 `2`**：配套 spec §3.2 只给了形状没给语义。本实现取「二维频率」（`[…,0]` 乘行坐标、`[…,1]` 乘列坐标） | 参数形状与预算不变；只影响该模块的 `forward` | 已记录，核对 `model_pytorch.py` 后可改一处 |
| 4 | **`docs/` 被 `.gitignore` 排除**（`:46 /docs`）⇒ 两份 spec 与 `games.npz` 生成脚本都不在版本控制内 | 设计记录无版本历史 | **已提请注意，未擅自改动该规则** |
| 5 | ~~**`soft_ce` 的 CLI 入口**~~ —— 🔴 **已解决，移出「悬而未决」** | 原以为「函数已实现但没接 CLI ⇒ 段 2 无法启动」 | ✅ **已接**。`--policy-loss` 的 choices 含 `soft_ce`，给 `--soft-index` 即派生为 `soft_ce`（`scripts/train_sft.py::resolve_policy_loss_kind`）。A4 的四个旗（`--soft-index` / `--soft-weight` / `--soft-only-sampling` / `--soft-every`）已登记进**65 个**的冻结集。⚠ **新的操作纪律**（不是新问题）：段 2 必须 `--soft-index` + `--soft-only-sampling 1`，否则 policy 项缩小 ~100×（§1.2） |
| 6 | **ch18/19 的死子判定要不要补** | 若补，须猜判据；猜错得到**第三种**偏差，比现在这个已知偏差更难诊断 | **本轮决定不补**（按对齐口径优先）。不是「待定」，是**已决的不做** —— 重启这个话题需要新的可执行 oracle |
| 7 | 🔴 **`warm_futurepos()` 还没接进 `main()`** | A6 的六条契约已定（§3 A6），但**父进程 fork 前的那次强制解析没有调用方** ⇒ 惰性兜底可能在 fork 之后触发 ⇒ **12.3 GB × worker 数** | **未决**（接线活，不需裁决）。缓解手段：`mp` 的 start method 在本机是 `spawn` 而非 `fork`，但契约不变，仍应显式调 |
| 8 | 🔴 **`prefetch-workers` 到底几个** | 本机实测 1/2/4/8，**8 是拐点**（边际效率 0.701）；「云端 24 核 ⇒ 12」是**外推，没有实测** | **未钉死**。落地前在目标机上跑一次 `scripts/bench_v7_features.py --phase workers --workers 8,12,16,24` |

---

## §7 实际落地的文件清单（🔴 原 spec 一个文件清单都没有，这是补的）

**新增**
| 文件 | 谁用 | 状态 |
|---|---|:--:|
| `scripts/build_games_sidecar.py` | 🔴 **D0 的全部**：4 个键 + `_AnchorIndex` 哈希锚点 + 覆盖报告 + `parse_komi` 的 ×100 修正 | ✅ |
| `scripts/bench_v7_features.py` | 🔴 **C0 的全部** | ✅ |
| `benchmarks/bench_v7_features.json` | C0 的产物（`generated_at 2026-10-02T19:27:30`） | ✅ |
| `scripts/crosscheck_stdata.py` | 🔴 **B7 的全部**（三档资格 + `--min-rate` 门禁） | ✅ |
| `scripts/smoke_train_v7.py` | 端到端冒烟（配套 spec §4.2 / §9.7） | ✅ |
| `src/data/feature_v7.py` | 🔴 **B2 / B3 / B5**（原设计写的是单个 `src/game/go_features_v7.py`） | ✅ |
| `src/data/feature_v7_ladders.py` | **B4**（`iterLadders` + §2.2 的回退复制门控） | ✅ |
| `src/data/feature_v7_gather.py` | **B6** + A6 的 `FUTUREPOS_OFFSETS` / `LADDER_OFFSETS` | ✅ |
| `src/data/kata_label_join.py` | 🔴 **A5 的全部**（`build_soft_index` + `materialize_dataset` + `hash_spec_fingerprint`）+ §2.4 的 materialize | ✅ |
| `src/data/katago_npz.py` | 段 3 的 stdata 读取器（配套 spec §9.1） | ✅ |
| `src/networks/katago_v7_loss.py` | 12 项 loss 装配 + `build_score_distr_target` | ✅ |
| `tests/test_dataset_futurepos.py` | 🔴 **A6 的全部**（28 个 test 函数） | ✅ |
| `tests/test_crosscheck_tool.py` | 🔴 **B7 工具的契约**（资格不得被弄错、门禁只对 `ALIGNED`） | ✅ |
| `tests/test_soft_ce.py` / `test_soft_cli.py` / `test_join_cache_freshness.py` / `test_games_sidecar_alignment.py` | A2 / A4 / A5 / D0 | ✅ |
| `shell/train_sft_npu_4card_katago_v7.sh` | 训练脚本 | ❌ **未写**（B8 未落地） |

**修改**：`src/data/sgf_parser.py`（`RU` + `RE` 数值，含 `W+3 zi` / `W+0,25`
两条显式分支）、`src/data/dataset.py`（4 元组 + dict、`attach_soft`、
`attach_futurepos` / `warm_futurepos` / `FUTUREPOS_SENTINEL`）、
`scripts/train_sft.py`（V7 loss 装配、`labels=True` 接线、A4 四旗）、
`src/inference.py`（`register_in_channels_builder(22, ...)`，`:177`）。

**不动**：`data/sgf_19x19_full.npz`、`scripts/build_dataset.py`（不 rebuild）、
`src/game/go_rules.py` 的 `feature_planes` / `feature_planes_batched`。

⚠ **原 spec 引用的三个测试文件不存在**：
`tests/test_soft_prefetch.py`、`tests/test_soft_cache.py`、
`tests/test_v7_neighbor_gather.py`、`tests/test_v7_vs_official.py`
（四个）⇒ 实际落点见 §3 的 A 组与 B 组各条。
⚠ **原 spec 漏列的、实际存在且关键的文件**：`scripts/build_games_sidecar.py`
（**D0 的地基**）、`scripts/crosscheck_stdata.py`、`scripts/bench_v7_features.py`、
`benchmarks/bench_v7_features.json`。
