# Go-AI 训练流水线设计（分段 · 软标签 · 22 通道实时算）

> **状态**：设计定稿（2026-10-02，经 20+ 轮逐节问答批准），待转 `writing-plans`。
> **配套**：`2026-10-01-katago-nbt-tf-design.md`（模型 + 22 通道输入 + 12 项 loss）。
> 本 spec 只管**训练流水线**：分段策略、软标签接入、局级 sidecar、22 通道实时算的接线。
> **一手来源**：`https://github.com/lightvector/KataGo`；stdata 数据的实测反解见配套 spec §9。
> **硬约束**：**不 rebuild** `data/sgf_19x19_full.npz`（34,202,713 行 / 162,298 局）。

---

## §1 目标与分段策略

### 1.1 三段式

| 段 | 数据 | 量 | policy 目标 | 目的 |
|---|---|---:|---|---|
| **1** | 自有 34.2M 语料（SGF 派生标签） | 34,202,713 | 人类着法 one-hot | 通路基线；学会 22 通道输入的读法 |
| **2** | 自有语料的 **1%**，用 KataGo 标注 | ~342,000 | **访问分布** | 域匹配的蒸馏：同样的职业对局，但换成搜索标签 |
| **3** | `katago/stdata`（19×19 部分） | **≈3,100,000** | **访问分布** | 棋力主体 |

**段 2 与段 3 的 policy 目标同为访问分布 ⇒ 可以合并成一轮跑 ~3.44M。**
段 2 的价值是**域匹配桥梁**：stdata 是自对弈局，我们语料是职业对局，
1% 那一小段让模型见到「人类局面 + 搜索答案」这种配对。

### 1.2 三条关键决定与被否决的替代

| 决定 | 否决的替代 | 理由 |
|---|---|---|
| **policy 保持 K=2**（不加第 3 个输出） | K=3（访���分布与人类 one-hot 各占一路） | 段 2/3 目标同型、可合并；人类 one-hot 已在段 1 学进去，段 2/3 覆盖它**正是蒸馏的目的**。K=3 只在「同一部位两个 policy 目标同时训」时才需要，而那会带来权重调参负担而收益未经验证。且 K=2 恰好对上 stdata 的两路输出（`policyTargetsNCMove[0]` 本方 / `[1]` 对手；spec 的 `π_opp` = 下一手，在自对弈里与「对手走的那手」是同一个东西） |
| **`soft_mask` 逐行二选一**，不做混合 | 软硬线性混合（`--soft-weight` 控比例） | 同一个 head 同时收到「搜索分布」与「人类 one-hot」会**去折中**而不是学搜索。混合多一个旋钮且无证据支撑 |
| **段 2 只训软行**（`--soft-only-sampling`） | 全量混训 / 靠 `--soft-weight 0` 关掉软行 | 混训 = 上面的折中问题；`--soft-weight 0` 等于「开了软标签但 99% 样本还在教 one-hot」的**静默半吊子**。独立采样器与 Phase 4 item 3 的窗口**正交**，两者可叠加 |

### 1.3 不在范围内

- 非 19×19 泛化（stdata 里 63~75% 是 19×19，其余丢弃）
- 融合注意力的深度优化（`sdpa_force_math` 先保留）
- `scorestdev` 的 `beta` 裁决（配套 spec §4.2 标为待裁决）
- selfplay RL（仓库已有 P3-C PPO；本管线的终点是「像 KataGo」，超过教师要靠 RL）

---

## §2 已核实的事实基线

以下每个数字都是实测的，方案的形状由它们决定。

### 2.1 语料与数据集

| 项 | 值 |
|---|---|
| 主 npz | 34,202,713 行 / **162,298 局**（`game_ids` 稠密 `0..162297`）/ 10 列 |
| 每局行数 | min 12 / p50 208 / p95 302 / max 2068 / mean 210.7 |
| **≥20 手的局** | **162,296 / 162,298 = 100%** ⇒ ply-20 哈希锚点对所有局安全 |
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

### 2.3 22 通道实时算的性能闸门

| 项 | 数值 | 来源 |
|---|---:|---|
| NPU 侧吞吐需求 | ~**2635** 行/s | `run.txt:700` 4 卡 910A 实测（v18, 12ch） |
| 预取 worker | **8** | `--prefetch-workers` 默认 |
| **每行时间预算** | **3.04 ms** | 2635 / 8 |
| 现有 12ch 单盘基线 | 1.78 ms | `go_rules.py:2479` 实测 |
| **22ch 余量** | **1.7×** | |

新增五类通道的成本：3 气桶（≈0，复用已在跑的 flood）、历史 5 手交错（≈0）、
parity 三角波（≈0）、`calculateArea`（中，1 块盘面）、**`iterLadders`（贵，3 块盘面）**。

> ⚠ **C0 的角色是标定，不是闸门。** 22 通道已定（§1.2），C0 只用来定
> `--prefetch-workers`（8→16？）与 batch size。

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

⚠ **`np.load('x.npz')['boards']` 会把整个成员解压进内存**（12.3 GB），不是按需。
本机 13.9 GB ⇒ 必须经 `kata_label_join.materialize_dataset()` 落成 `.npy`
后 `mmap_mode='r'`（落盘 152s，实测 478K 行/s）。

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

抽样 133,604 个 SGF：

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
| `B+R` / `W+R` / `B+Resign` / `W+Resign` | 98,352 | **75.2%（认输，无分差）** |
| `B<N>` / `W<N>`（如 `B+2.5`） | 24,748 | 18.5% |
| `B+T` / `W+T` | 271 | 0.2% |
| `draw` / `B+` / `W+F` 等 | 211 | 0.2% |

🔴 **两条硬结论：**

1. **贴目五花八门** ⇒ `g_komi` 必须逐局存，**不能假设常数 7.5**；`KM=0` 占 5.9%。
2. **75% 的局是认输** ⇒ 配套 spec §4.5 里 **5 项**（#5/#6 scorebelief、
   #8 scoremean、#9 lead、#10 scoring）吃 `w_score`，而 `g_score` 为 `NaN` 时
   `w_score=0` ⇒ **这 5 项在 75% 的局上权重为 0**。有分差监督的约
   `24.8% × 162,298 × 210 ≈ 6.6M 行`（34.2M 的 **19%**）。

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
| ⚠ 部分 | #5/#6/#8/#9/#10（score 系） | 仅 19% 的行有分差；**段 1 建议整组关闭**而不是跑 19% 覆盖 |
| ❌ | #4 ownership、#10 scoring、#12 seki | **不从 SGF 造**（§3 D0b 已砍），段 2/3 由 stdata 提供 |

⇒ **段 2/3（1% 标注 + stdata 3.1M）才开全部 12 项** ——
那 3.4M 上这些信号来自**搜索器自己的评分器**（`valueTargetsNCHW`），
质量比从认输棋谱反推高得多。

> 这条把 ownership/scoring/seki 的训练责任从 **34.2M 挪到 3.4M**，
> 是本 spec 相对「照搬 12 项全开」最重要的修正。

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
A1  labels=True → 4 元组 + dict（soft / soft_mask 放进 dict）
    + 3 个假 dataset 补 labels=False 形参
       tests/test_eval_coverage.py:75 / test_eval_determinism.py:111
       / test_eval_no_augment.py:219
    → test_dataset_soft_labels.py
A2  compute_policy_loss 软 CE 分支（掩码二选一）
    → test_soft_ce.py（同 batch 内 mask=1/0 两行各走对分支）
A3  预取器 / worker 透传 dict              → test_soft_prefetch.py
A4  CLI --soft-labels / --soft-weight / --soft-index
    + --soft-only-sampling（行索引空间收窄到 soft_row >= 0）
    → test_soft_cli.py
A5  join 缓存（scripts/build_soft_index.py）
    key = (dataset size+mtime, labels size+mtime, max_repeats, pos_hash 版本)
    → test_soft_cache.py
```

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

### C0 · 特征基准（标定）

测四个数，输出到文档：

1. `iterLadders` 单盘 ms/行
2. `calculateArea` 单盘 ms/行
3. 22ch 完整 vs 1.78 ms 基线
4. 8 进程并行的有效行/s

⇒ 定 `--prefetch-workers` 与 batch size。

### D0 · 局级 sidecar `games.npz`（≈1.2 MB，不碰主 npz）

```
D0a memmap game_ids → G = 162,298 个行区间
D0b **只扫 KM / RU / RE 三项**（不做终局重放）
    · g_komi    ← SGF KM（已解析）——⚠ 贴目分布很散，见 §2.6
    · g_score   ← RE 数值解析（新增）——⚠ 75.2% 是认输，见 §2.7
    · g_rules   ← 新增 RU 解析；无 RU 按 AREA/none/simple/false/0 默认
    · g_resign  ← RE 形如 B+R/W+R/B+Resign/W+Resign/B+T
D0c ply-20 哈希锚点匹配 → games.npz
    → test_games_sidecar_alignment.py（匹配率必须打印）
```

> 🔴 **D0b 原本还要重放到终局算 `g_ownership`/`g_scoring`/`g_seki`
> （8–28 min + 两个最难移植的算法），2026-10-02 决定砍掉。** 理由三条：
>
> 1. **性价比低**：那笔开销只买到 **2.5 项** loss，而 #10 已因分差缺失在 75% 的行上废了。
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
锚点法**对枚举顺序与该 bug 完全免疫**，且 100% 的局 ≥20 手保证锚点可用。

⚠ **必须输出覆盖报告。** 匹配率 < 100% 时，未匹配的局相关权重置 0，
**不静默填垃圾**。

### B 组 · 22 通道实时算

```
B1  _check_n_channels 放行 22 + feature_planes_batched 加 V7 分发
B2  feature_v7：3 气桶 + 历史 5 手交错        ← 纯函数，先测  test_v7_planes.py
B3  feature_v7：calculateArea → ch18/19       ← 纯函数，先测  test_v7_area.py
B4  feature_v7：iterLadders → ch14/17         ← 纯函数，先测  test_v7_ladders.py
B5  19 维全局装配 + futurepos 掩码分派（只算 4%）
B6  邻行 gather（5 偏移，批量，绝不逐样本循环）  test_v7_neighbor_gather.py
B7  交叉校验：ch14/ch17 对 stdata 逐位；ch15/16 对 SGF 重放  test_v7_vs_official.py
B8  train_sft.py 切 22 通道（数据侧与模型侧必须同改）
```

**B2–B4 三个纯函数在 B6 接线之前就能各自验**；B6 只是把三块盘面喂进去 +
处理回退复制。这是把 B 组内部再切一刀的原因：`iterLadders` 是唯一
「既慢又可能移植错」的项，不该和接线耦合。

⚠ **B4 的验证必须拆成两半**（`katago/` 下只有 Windows 二进制，**无 `cpp/`**，
`nninputs.cpp` 不可读 ⇒ stdata 是唯一可执行 oracle）：

| 通道 | 依赖盘面 | 能否对 stdata 直接验证 |
|---|---|---|
| ch14 | 当前盘 | ✅ |
| ch17 | 当前盘（working-move） | ✅ |
| ch15 | 前一手盘 | ❌ **stdata 是逐行独立样本，没有前一盘的盘面** |
| ch16 | 前二手盘 | ❌ 同上 |

⇒ 「**算法**」由 ch14 对 stdata 逐位担保（≈3.1M 个 19×19 行，样本量足够）；
「**盘面**」由 SGF 重放担保（纯本地测试）。ch15/16 复用同一份已验证代码，
只换喂进去的盘面。**不需要 KataGo 源码也能把 4 个通道的可信度拆开验证。**

---

## §4 门禁

每组单独过（沿用 `v21-roadmap.md:364`）：

```
python -m pytest tests -q && bash -n shell/*.sh
```

**当前基线：`1011 passed / 2 skipped`，`bash -n` exit 0。**

执行顺序：

```
1. A1  契约（最小、解锁一切）
2. A2  软 CE + 单测
3. C0  特征基准 → 记录四个数
4. D0  sidecar（写脚本后尽早起跑）
5. A3 / A4 / A5  预取 / CLI / 缓存
6. B1–B8  22 通道
```

---

## §5 退路表

| 风险 | 退路 | 代价 |
|---|---|---|
| C0 严重超预算 | `--prefetch-workers` 提到 16 / 降 B | 步时增加，结构不改 |
| ch14 对 stdata 对不上 | 降到 **18 通道**（ch14–17 恒 0） | 主干 / 四头 / 12 项 loss / 参数量 **5,561,832 一行不改** |
| 锚点匹配率 < 100% | 输出覆盖报告；未匹配局 `w_ownership`/`w_seki`/`w_scoring=0` | 那部分行只损失 3 项 |
| `calculateArea` 语义有差 | B7 暴露；必要时只在 AREA/TAX_NONE 下开 | territory 局 ch18/19 恒 0（对齐官方，非缺陷） |
| 预取跟不上 | 提 worker 数或降 B | 步时增加 |
| 段 2 混合训练（未加 `--soft-only-sampling` 就开跑） | 立即停 —— index 0 会向两种语义折中 | 不学搜索 |

---

## §6 悬而未决

| # | 事项 | 影响 | 状态 |
|---|---|---|---|
| 1 | **`scorestdev` 的 `beta`**：配套 spec §4.2 的 `SoftPlus(x₁, 0.05)` 让预测初值落在 277，而 loss #7 的目标是 5~20 量级、δ=10。冒烟训练 40 步该项 **−0.0%**，硬算需 ~9 万步才挪到位 ⇒ **等效于没有学习信号**。改成 `beta=1.0`（初值 13.86）是**一行**，但那是改已批准的 spec | 配套 spec 的 12 项之一失效 | **待裁决**。已提为具名常量 `SCORE_STDEV_SOFTPLUS_BETA`，改一行不影响结构/参数/预算 |
| 2 | **`col3` / `col22` 的精确语义**（stdata 的 `globalTargetsNC`） | 目前**不作为任何 loss 的目标**，只记录 | 需查 KataGo `dataio.cpp` 写入顺序 |
| 3 | **2D RoPE 的 `(H,16,2)` 里那个 `2`**：配套 spec §3.2 只给了形状没给语义。本实现取「二维频率」（`[…,0]` 乘行坐标、`[…,1]` 乘列坐标） | 参数形状与预算不变；只影响该模块的 `forward` | 已记录，核对 `model_pytorch.py` 后可改一处 |
| 4 | **`docs/` 被 `.gitignore` 排除**（`:46 /docs`）⇒ 两份 spec 与 `games.npz` 生成脚本都不在版本控制内 | 设计记录无版本历史 | **已提请注意，未擅自改动该规则** |
