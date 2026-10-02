# KataGo NBT + Transformer 重建设计（V7 输入 22/19）

> **状态**：设计定稿（§1–§8 全部经人工逐节批准），待复核后转 `writing-plans`。
> **取代**：`2026-10-01-katago-se-rebuild-design.md`（见 §8.1）。
> **一手来源**：https://github.com/lightvector/KataGo
> **硬预算**：参数 ≤ **5.85M**，峰值显存 ≤ **31.0 GiB @ batch 3200 / 卡 / 19×19 / 4×Ascend 910A**。

---

## §1 目标与约束

### 1.1 目标

把模型从「自造的 17 通道 SE 卷积」重建为**官方 KataGo 语义**：

1. 输入 = 官方 **V7 布局**（22 空间 + 19 全局），通道语义与 `cpp/neuralnet/nninputs.cpp::fillRowV7` 逐条对齐；
2. 主干 = 官方 **nbt（nested bottleneck）+ transformer** 内块，`fson` 固定方差标量初始化、`rsnh` trunk 末端空间 RMSNorm，**全网无 BatchNorm**；
3. 头 = 官方 policy（K=2：π + π_opp，带 gpool 与 pass 并联支路）、value（3 分类 + scoremean/scorestdev/lead）、ownership、scorebelief；
4. loss = 官方 `python/katago/train/metrics_pytorch.py` 的 **12 项可构造监督目标**（无需搜索）。

### 1.2 硬约束（验收基准）

| # | 约束 | 判定方式 |
|---|---|---|
| C1 | 参数量 ≤ **5,850,000**（目标 **5,561,832**，余量 4.9%） | `tests/test_katago_v7_budget.py` 断言精确值 |
| C2 | `max_memory_allocated ≤ 31.0 GiB @ B=3200/卡` | 远端实测回填预算测试 |
| C3 | `ACTION_SIZE = 362`（361 落点 + pass） | 模型/数据测试 |
| C4 | fp16 + `GradScaler` 训练路径不变 | 现有 `test_fp16_overflow.py` 等 |
| C5 | `pytest tests` 全绿（约 700 项）+ `bash -n shell/*.sh` | 本地必验 |
| C6 | 旧 17 通道路径 `feature_planes` **零回归** | `test_go_feature_planes_v21.py` 18 项全绿 |
| C7 | `dataset.sample_batch_numpy` **默认三元组签名不变** | `labels=False` 为默认 |

### 1.3 范围

- **做**：输入、主干、四头、12 项 loss、数据集标签、训练接线、预算守卫。
- **不做**（Phase 3）：13 项需搜索的 loss（soft/optimistic policy、td-value、td-score、variance time、shortterm error、Q-value）、非 19×19 泛化、深度融合注意力调优。

---

## §2 输入特征（官方 V7：22 空间 + 19 全局）

### 2.1 来源与核验

逐条对照 `cpp/neuralnet/nninputs.cpp::fillRowV7`（2302–2747 行）**全文**核验。下表的「语义」是源码实际行为，不是文档转述。

### 2.2 空间 22 通道

| ch | 语义（源码） | 我们如何得到 |
|---:|---|---|
| 0 | on-board（填充区为 0） | 现有 `boards` 推出 |
| 1 | pla 棋子 | 同上 |
| 2 | opp 棋子 | 同上 |
| 3/4/5 | 棋子的 1/2/3 气 | `boards` 现算 |
| 6 | ko-ban：encore=0 时 `ko_loc` ∪ (`superKoBanned` \ {ko_loc}) | 现有 `ko` 列（**简单局 = 仅 ko_point；不跟踪 superko 集合，见 §2.6 D3**） |
| 7 | `koRecapBlocked` —— **仅 encore>0** | 简单局恒 **0** |
| 8 | 源码注释写「6,7,8」但**从未写入** | 恒 **0** |
| 9–13 | 过去 5 手落点（**pass 不占通道**），顺序为 opp, pla, opp, pla, opp（自 `nextPlayer` 视角） | `my_hist`/`op_hist`（各 3 手）+ `to_play` 交错 |
| 14 | **当前盘**上的梯子（`iterLadders`） | 现算 |
| 15 | **前一手盘**上的梯子 | 邻行 gather `boards[i−1]` |
| 16 | **前二手盘**上的梯子 | 邻行 gather `boards[i−2]` |
| 17 | 当前盘梯子的 working-move 位置（**仅对手方、且 >1 气**） | 现算 |
| 18/19 | 当前区域（pla / opp） | `calculateArea`，**规则条件化**（见 §2.5） |
| 20/21 | second-encore 起始子 —— **仅 encorePhase≥2** | 简单局恒 **0** |

**历史门控的官方语义（易错点）**：

```cpp
prevBoard     = (numTurnsOfHistoryIncluded < 1) ? board        : hist.getRecentBoard(1);
prevPrevBoard = (numTurnsOfHistoryIncluded < 2) ? prevBoard    : hist.getRecentBoard(2);
```

历史不足时 **不是置 0，而是回退复制**：history=0 ⇒ ch15=ch14、ch16=ch15；history=1 ⇒ ch16=ch15。

### 2.3 全局 19 通道

| ch | 语义（源码） | 我们 |
|---:|---|---|
| 0–4 | 过去 5 手中哪些是 pass | `my_hist/op_hist` 含 −1 ⇒ 可判 |
| 5 | `currentSelfKomi(nextPlayer) / 20`，clip 到 `±(bArea + KOMI_CLIP_RADIUS)` | **符号随 `to_play` 翻转**；来自局表 `g_komi` |
| 6 | ko 规则：simple=0,0；positional/spight=1,**0.5**；situational=1,**−0.5** | 局表 `g_rules` |
| 7 | （同上，第二分量） | 同上 |
| 8 | `multiStoneSuicideLegal` | 局表，缺 `ru` 时默认 **false** |
| 9 | territory 计分 ⇒ 1（area ⇒ 0） | 局表，缺 `ru` 时默认 **area** |
| 10/11 | tax：none=0,0；seki=1,0；all=1,1 | 局表，默认 **none** |
| 12/13 | encorePhase >0 / >1 | 简单局恒 **0** |
| 14 | `passWouldEndPhase` | 新函数 `passWouldEndPhase` |
| 15/16 | `playoutDoublingAdvantage` 非 0 ⇒ 1 / 0.5·pda | 无 PDA ⇒ 恒 **0** |
| 17 | `hasButton` | 局表，默认 0 |
| 18 | komi×棋盘奇偶三角波 | 新函数；**该通道即官方 `model.py` 里乘进 scorebelief parity 的那一个**（源码注释：V3/V4=13、V6=15、**V7=18**）——与 §2 通道表互证 |

### 2.4 恒 0 通道（**保留，不裁剪**）

空间 `7, 8, 20, 21` + 全局 `12, 13, 15, 16` 共 **8 路**恒为 0。
**保留理由**：严格复刻官方张量布局，未来接官方 checkpoint 时零改形状；裁剪仅省 0.3% 参数，不值当。

### 2.5 ch18/19 的规则条件化

```cpp
if (SCORING_AREA && TAX_NONE)  calculateArea(...)           // 常规
else if (SCORING_AREA && (SEKI|ALL))  calculateIndependentLifeArea(keepStones=true)
else if (SCORING_TERRITORY)  // 仅 encorePhase >= 2 才置位 ⇒ 正常阶段恒 0
```

⇒ **territory 计分的对局，ch18/19 在正常阶段恒 0 —— 这是对齐官方，不是缺陷**（§5 已批准）。SGF 缺 `RU` 时默认 AREA。

### 2.6 dtype 与有意偏差

| 项 | 官方 | 我们 | 影响 |
|---|---|---|---|
| 空间通道 | `bool`（`rowBin`） | **float16** `{0,1}` | 逐位语义等价 |
| 全局通道 | `float32` | **float16** | 量级差异 < 1e-3，接受 |
| D1 | `superKoBanned` 全集 | 仅 `ko_point` | KO_SIMPLE 下等价 |
| D2 | `encorePhase` 机制 | 无 encore | 简单局等价；ch7/12/13/20/21 正确为 0 |
| D3 | `currentSelfKomi` 含 draw-jitter | 直接用 `g_komi` | 见 §4.6 |

---

## §3 主干结构

**配置名**（沿用官方命名法）：`b11c256h4nbttflrs-fson-silu-rsnh`

| 常量 | 值 | 说明 |
|---|---|---|
| `C` | **256** | trunk 宽度 |
| `M` | **128** | nbt 内宽 = C/2，须被 32 整除（head_dim=32） |
| `H` | **4** | 注意力头数 ⇒ `head_dim = 128/4 = **32**`，落在 CANN 支持集 {16,32,64} |
| `F` | **384** | SwiGLU 隐层 = 1.5C |
| `B` | **11** | nbt 块数，全部为 transformer 内块 |
| `G` | **0** | **无 gpool 块**（官方：注意力已见全局） |
| 输入 | 22 / 19 | init conv `3×3` |

### 3.1 逐层

```
spatial (B,22,19,19) ── Conv2d(22→256, 3×3, padding=1, bias=False) ─┐
global  (B,19)        ── Linear(19→256, bias=False) ───────────────┴─ add → (B,256,19,19)
                                                                          │
   ┌──────────────── 重复 11 次 ────────────────────────────────────────────┘
   │  1. NormAct(C=256)          ← fson: x·(γ·K) + β     无 BN
   │  2. Conv2d(256→128, 1×1, bias=False)                 [p]
   │  3. TransformerBlock @128                            ┐ 内块 ×2
   │  4. TransformerBlock @128                            ┘ 各自带残差
   │  5. NormAct(M=128)           ← fson
   │  6. Conv2d(128→256, 1×1, bias=False)                 [q]
   │  7. + block 输入（外层残差）
   └──────────────────────────────────────────────────────────────────────
                                                                          │
                                     RMSNormMask(spatial=True) + SiLU      │  ← rsnh：全网无 BN
                                                                          ▼
                                                                    trunk 输出 (B,256,19,19)
```

### 3.2 TransformerBlock @ M=128

```
x ─ RMSNorm(128, eps=1e-6) ─ MHSA(q,k,v,out 各 Linear(128→128, bias=False))
                              · 4 头 × dim32，scale = 1/√32
                              · 2D 可学习 RoPE，参数 (H, 16, 2)，init exp(uniform(log 1/50, log 1))·±1
                            ─ + x
   ─ RMSNorm(128, eps=1e-6) ─ SwiGLU：SiLU(x·W_up) ⊗ (x·W_gate)，再 ·W_down
                              · W_up/W_gate: 128→384，W_down: 384→128，全 bias=False
                            ─ +
```

**无 dropout**（对齐官方；现有 `attn_dropout=0.1` 不进入新模型）。
无 dropout 的双重收益：省 3.1 GiB 张量 + **融合注意力更容易通过**（dropout 常是融合的阻碍，见 §6 R1）。

### 3.3 fson 固定方差标量 K（逐字取自 `model_pytorch.py::Model.initialize`）

```python
for i, block in enumerate(self.blocks):
    block.initialize(fixup_scale=1.0 / math.sqrt(i + 1.0))          # trunk 第 i 块首个 NormMask
# 块内：
normactconvp.initialize(scale=1.0, norm_scale=fixup_scale)
for j, inner in enumerate(blockstack):
    inner.initialize(fixup_scale=1.0 / math.sqrt(j + 1.0))           # 内块 j
normactconvq.initialize(scale=1.0, norm_scale=1.0 / math.sqrt(internal_length + 1.0))  # = 1/√3
# trunk 末端：
norm_trunkfinal.set_scale(1.0 / math.sqrt(num_total_blocks + 1.0))   # = 1/√12
```

原理（KataGoMethods.md）：每个残差块给所处栈贡献方差 1，故第 i 块读到的累计方差 = i+1，`K` = 其倒数平方根。
`num_total_blocks` 按「trunk 每块 +1」取 **11**（内块只增加内层流方差，不进 trunk 账）—— 实现时以 `model_pytorch.py::Model.initialize` 实际值为准，列入计划核验项。

### 3.4 权重初始化

truncated normal，`target_std = scale · gain / √fan_in`；
`gain(SiLU) = √2.0`（官方注释：理论值 √2.8108，为兼容保留 √2.0）；
init conv `scale = 0.8`，global linear `scale = 0.6`。

### 3.5 相对现有代码的行为变更（已批准）

1. **全网无 BN** —— `BatchNorm1d/2d` 路径不进入新模型；`fson + rsnh` 只在 trunk 末端留一个 spatial RMSNorm。
2. **不加 dropout** —— `attn_dropout=0.1` 停用；正则靠 weight decay + 数据量。
3. **注意力仍走 `_sdpa` 四站点机制**（`backbone.py:581`），但 `head_dim` 由 **46 → 32**；`train_sft.py:2269` 的 `sdpa_force_math=True` 暂时保留，探针证实可用后再切（§6 R1）。
4. **逐块梯度检查点保留** ⇒ 同一时刻只有 1 个 MHSA 站点重算。

---

## §4 头与 loss

### 4.1 Policy 头（K = 2：π 与 π_opp）

```
trunk ─┬─ Conv2d(256→P=48, 1×1, bias=False) ──────────────→ PP
       └─ Conv2d(256→G=48, 1×1, bias=False) → NormAct → KataGPool → (3G=144)
                                        Linear(144→48, bias=False) ── + ─→ PP
PP → NormAct → Conv2d(48→K=2, 1×1) ──────────────────────→ (B,2,19,19)
                                                          off-board 掩 −5000
pooled(144) → Linear(144→48, bias=True) → Act → Linear(48→2, bias=False) → pass (B,2)
拼接 → policy logits (B, 2, 362)
```

`KataGPool`（policy 用）三个统计量：`mean`、`mean·((√A−14)/10)`、`max`。

**π_opp 的监督代理**：官方 = 下一位置的搜索策略；我们用**实际下一手着法 one-hot**，权重 0.15，**每局末手权重 0**。
pass 是真实的策略索引（`policySize = 361+1 = 362`），不是特殊类。

### 4.2 Value 头

```
trunk → Conv2d(256→V=48, 1×1, bias=False) → NormAct → VV (B,48,19,19)
                                                    ├─ ownership 子头（4.3）
                                                    └─ KataGPool_value → (3V=144)
Linear(144→W=96, bias=True) → SiLU → h (B,96)
   ├─ Linear(96→3, bias=True) → outcome_logits        {胜,负,无结果}
   └─ Linear(96→3, bias=True) → scoremean = 20·x₀
                                 scorestdev = 20·SoftPlus(x₁, 0.05)
                                 lead       = 20·x₂
```

`KataGPool_value` 第三个统计量换成 `mean·(((√A−14)²/100) − 0.1)`（二次 board-size 缩放，**无 max**）。
⚠ **固定 19×19 下 `((√A−14)/10)=0.5`、`(((√A−14)²/100)−0.1)=0.15` 都是常数** ⇒ value 池化实际只是三个缩放的 mean。公式照抄保留，以便将来泛化。

> 🔴 **待裁决：`scorestdev` 的 `SoftPlus(x₁, 0.05)` 与 §4.5 loss #7 自相矛盾。**
>
> 2026-10-02 端到端冒烟训练（`scripts/smoke_train_v7.py`，302 行 / 40 步）实测：
> **这一项 40 步内变化 −0.0%**，是 12 项里唯一不动的。硬算：
>
> ```
> F.softplus(0, 0.05) = log(2)/0.05 = 13.86  ⇒  ×20 = 277   ← 预测初值
> loss #7 的目标 = std(softmax(scorebelief)) = 5~20,  δ=10    ← 初始偏差 260
> 系数 0.001 × AdamW 步长 ~3e-4  ⇒  需 ~9 万步才挪到位，真实训练约 1 万步
> ```
>
> ⇒ 按字面写法，**loss #7 等于没有学习信号**。取默认 `beta=1.0` 则
> `20·softplus(0,1) = 13.86`，与目标同量级、起点即可用。
>
> **现状**：`src/networks/katago_v7.py` **逐字实现 spec 的 0.05**（已批准的 spec
> 是契约，不擅自改），但提成具名常量 `SCORE_STDEV_SOFTPLUS_BETA`，
> 裁决后只改这一行，头部结构 / 参数量 / 预算均不受影响。
> **本行在裁决前视为「已知缺陷」，不得当作可训练项排期。**

### 4.3 Ownership 头

```
VV → Conv2d(V=48→1, 1×1, bias=False) → tanh → (B,1,19,19) ∈ [-1,+1]
```

### 4.4 Scorebelief 头（842 桶）

桶数 `2×(361+60) = **842**`（`EXTRA_SCORE_DISTR_RADIUS = 60`），`mid = 421`。

```
pooled(144) → Linear(144→96, bias=True) ─────────────┐
(1,) → 0.05·(i−mid+0.5) → Linear(1→96, bias=False) ──┤ + → SiLU → h (B,842,96)
(1,) → parity(i)·global[18] → Linear(1→96, bias=False)┘
h → Linear(96→nsb=8, bias=True) ────────────→ comp (B,842,8)
pooled(144) → Linear(144→8, bias=True) ─────→ mix (B,8)
scorebelief_logits = logsumexp_k(mix + comp) → (B,842)
```

`parity(i) = 0.5 − ((i−mid) mod 2)`；`global[18]` 即 §2.3 的奇偶三角波。

### 4.5 Loss：12 项（系数逐条对照 `metrics_pytorch.py`）

| # | 项 | 公式 | 系数 | 行权重 |
|---:|---|---|---:|---|
| 1 | policy | `−Σ π·log_softmax(logits₀)` | **1.0** | 恒 1 |
| 2 | π_opp | `−Σ π_opp·log_softmax(logits₁)` | **0.15** | 每局**末手 = 0** |
| 3 | value | `CE(outcome_logits, {胜,负,无结果})` | **1.20** | 恒 1 |
| 4 | ownership | `Σ BCE_with_logits(2·pretanh, (1+t)/2) / 361` | **1.5** | `w_ownership` |
| 5 | scorebelief pdf | `CE(logits, target)` | **0.020** | `w_score` |
| 6 | scorebelief cdf | `Σ(cumsum(softmax) − cumsum(target))²` | **0.020** | `w_score` |
| 7 | scorestdev 自预测 | `huber(pred, std(softmax(sb)), δ=10)` | **0.001** | 仅 `global_weight`（**无行权重**，与官方一致） |
| 8 | scoremean | `huber(20·x₀, final_score, δ=12)` | **0.0015** | `w_score` |
| 9 | lead | `huber(20·x₂, final_score, δ=8)` | **0.0060** | `w_lead` |
| 10 | scoring | `4(√(0.5·MSE+1) − 1)` 逐格均值 | **0.25** | `w_scoring` |
| 11 | futurepos | `0.25·(tanh(x)−t)²·[1.0,0.25]/√361` | **0.25** | `w_futurepos` |
| 12 | seki | `(CE_sign[3] + 0.5·CE_neutral)/361` × `8·0.005/(0.005+EMA)` | **自适应** | `w_ownership` |

`loss = Σ` 上表。正则走优化器 weight decay（官方 repo 亦如此；论文 `c_L2=3e-5` 不单列）。

**系数位置说明（易错）**：多数系数**内嵌在各自的 `loss_*_samplewise` 里**，不在装配处再乘；唯 `scoring` 的 `0.25` 在装配处。本 spec 的系数即最终有效系数。

**与论文的取舍**：`policy_opt_loss_scale=0.930` 是 version ≥12、6/8 输出模型的重归一化；我们 **K=2 ⇒ version ≤11 分支 ⇒ 系数 1.0**。`value` 用 repo 的 **1.20**（论文 2 分类用 1.5，不适用于 3 分类）。

**明确不做**（13 项，需搜索）：soft policy ×2、optimistic policy ×2、td-value ×3、td-score、variance time、shortterm error ×2、Q-value ×2。

### 4.6 标签构造与权重掩码

| 标签 | 层级 | 构造 | 无权重条件 |
|---|---|---|---|
| π | 行 | 实际着法 one-hot | — |
| π_opp | 行 | **下一手**着法 one-hot | 每局末手 → 0 |
| value 3 类 | 局→行 | SGF `RE`：`B+*/W+*` → 胜/负；`0/Draw/Void/?` → 无结果 | `RE` 缺失 ⇒ **整局丢弃**（沿用 `_emit` 现状） |
| ownership | **局** | 终局盘面 `calculateArea` → ∈{−1,0,+1}（**黑方视角**，读取时按 `to_play` 翻转） | 无结果对局 → 0 |
| scoremean / lead / 分差 | **局** | `RE` 数值（含 komi） | **认输 / 无分差 → 0** |
| scorebelief target | 行内即时 | `center=round(score)`，`λ=score−(center−0.5)`，`upperProp=round(λ·100)∈[0,100]`，写入 `center∓0.5` 两桶，和为 100 | 同上 |
| scoring | **局** | `fillScoring(终局盘面)` | 同 ownership |
| seki | **局** | `calculateIndependentLifeArea` → ∈{−1,0,+1} | 同 ownership |
| futurepos | 行 | 落子后 +8 / +32 手的盘面（2 通道） | 超出对局长度 → 0 |

**认输对局**：有胜负（value 可用）但**无分差** ⇒ #5/#6/#8/#9/#10 权重 0；**#4 ownership 仍计算**（终局盘面 flood fill 与是否认输无关）。
这是与 KataGo 的**有意偏差**（官方 `RE` 认输对局直接不记录），因为官方 `col27` 一列同管 ownership 与分数项，无法区分 —— 我们拆成 `w_ownership` / `w_score` 两列（§5.3），是本设计对官方权重布局的**唯一扩展**。

`global_weight`（官方 `col25`）= 现有 `game_weights` 列（基于棋手等级的 `exp(avg/20)`）。

---

## §5 数据集与标签

> 🔴 **本节于 2026-10-02 整体重写。** 原设计假定「`build_dataset` 加列 + 局表并进
> 主 npz」，实测后**已决定不 rebuild 主数据集**。原 §5.2/§5.3/§5.6/§5.7 的
> 「新增 `next_move` 列」「局表同 npz」「必须先修 game_id bug（阻塞 5.3）」
> 「5.6 B/行增量」**全部作废**，理由与替代方案见下。§5.1 的通道语义与 §5.4 的
> 批契约精神保留但形态改变。

### 5.0 硬前提：**不 rebuild 主数据集**

`data/sgf_19x19_full.npz` = **34,202,713 行 / 162,298 局**（`game_ids` 稠密
`0..162,297`）/ 10 列，**一个字节不动**。V7 需要的 13 项里 **11 项**能从现有
10 列推出或实时算；只有 6 个**局级标量** + 3 个**局级平面**需要新增一个
独立 sidecar。

语料完整性已核实（`data/*.tgz` 5 个 + `data/games/games/` 4 个目录）：

| 项 | 值 |
|---|---|
| tgz 内唯一 SGF | 36,274 |
| 目录内唯一 SGF | 133,604 |
| **两者重叠** | **0** |
| **并集** | **169,878** ≥ 162,298（接受率 95.6%） |

⇒ **语料完整，无缺失局**，sidecar 可覆盖全部 162,298 局。

### 5.1 十三项 V7 需要的逐项来源

| 需要 | 主 npz 里有 | 怎么来 |
|---|---|---|
| 22 通道输入 | ✅ `boards`/`my_hist`/`op_hist`/`ko`/`to_play` | **训练时 CPU 实时算**（§5.2） |
| `next_move`（π_opp 标签） | ✅ `moves` | `moves[i+1]`，守卫 `game_ids[i+1]==game_ids[i]`，局末手 = −1 |
| `π`（policy 标签） | ✅ `moves` | one-hot（KataGo 软标签走 §5.7） |
| `futurepos` | ✅ `boards` | 实时 `boards[i+8]` / `boards[i+32]`（§5.4） |
| `game_weight` | ✅ `game_weights` | 直接用 |
| 全局 ch0–4（pass 标志） | ✅ `my_hist`/`op_hist` | 含 −1 即 pass，实时推 |
| 全局 ch14（`passWouldEndPhase`） | ✅ `boards` | 实时推 |
| **value 3 类** | ✅ **`winrates`** | 见下方「白捡」 |
| 全局 ch5/18（komi）、ch6/7/8/9/10/11/17（规则） | ❌ | **sidecar `games.npz`**（§5.3） |
| `score`（scoremean/lead/scorebelief 目标） | ❌ 幅度被 α 污染 | **sidecar**（⚠ 75% 的局是认输 ⇒ 无值，§5.3.3） |
| ownership / scoring / seki 标签 | ❌ | **不从 SGF 造**（§5.3.0）。段 1 不训这三项；段 2/3 由 `katago/stdata` 的 `valueTargetsNCHW` 提供 |
| scorebelief 目标 | 派生 | 从 sidecar 的 `score` 实时算 center/upper（`NaN` 时权重 0） |
| 软 policy（1% KataGo 标注） | ✅ `kata_labels.npz` | `pos_hash` join → sidecar（§5.7） |

> 🎁 **白捡一：`value` 3 类能从 `winrates` 精确还原，零额外工作。**
> `build_dataset.py:299-309` 里 `winrates = tanh(value·to_play·α)`，其中
> `value ∈ {−1,0,1}`（`parse_result_to_value`）、`α = 0.3+0.7·frac ∈ [0.3,1.0]`：
>
> - `fv = 0` ⇒ `winrates = 0` **精确**
> - `fv = ±1` ⇒ `|winrates| = tanh(α) ∈ [0.291, 0.762]`，**永远不为 0**
>
> ⇒ **`winrates == 0` ⟺ 和棋**；`sign(winrates)` = to_play 视角胜负（`to_play`
> 也是列，要黑方视角就乘 `to_play`）。
> ⚠ 但 `|winrates|` 是 α 污染过的，**还原不出真实分差** —— 那正是 Phase 4
> item 2 要去掉的东西，所以分差仍需 sidecar。

> 🎁 **白捡二：`calculateArea` 不是从零移植。**
> `go_rules.py:2261 score()` **已经是区域计分 + flood fill**（空点连通区域、
> 只与一种颜色相邻则归该色），且 docstring 声明与 Tromp-Taylor 计分等价。
> 移植 = **在它之上暴露逐点归属图** + 对齐 KataGo 的区域分类语义
> （`SCORING_AREA`/`TAX_NONE` 条件化、seki 区域）。底层 flood fill
> （`_neighbor_groups:1446`、`_group_liberty_count:1551`）已在。

### 5.2 22 通道：**训练时实时算**，不落盘

原 §5.1 写「22 通道全部现场算，零新增空间存储」—— 这条**保留并强化**：
明确为「训练时由预取器在 CPU 上实时算」，**不预存 bit-packed**。

- 端口函数放 `src/data/feature_v7.py`（**不放 `go_rules.py`**，避免污染
  17ch 实现所在文件；`go_rules.py` 的 `feature_planes` / `feature_planes_batched`
  **一字不改**）
- 移植项：`iterLadders`（ch14–17）、`calculateArea`（ch18/19）、
  `passWouldEndPhase`（global14）、3 气桶、历史 5 手交错、komi 奇偶三角波
- 通道 7/8/20/21 + 全局 12/13/15/16 共 **8 路恒 0**（encore / 源码从未写），
  **保留不裁剪**（§2.4 的理由不变）

**性能闸门（必须先测，C0）**：

| 项 | 数值 | 来源 |
|---|---:|---|
| NPU 侧吞吐需求 | ~**2635** 行/s | `run.txt:700` 4 卡 910A 实测（v18, 12ch） |
| 预取 worker | **8** | `--prefetch-workers` 默认 |
| **每行时间预算** | **3.04 ms** | 2635 / 8 |
| 现有 12ch 单盘基线 | 1.78 ms | `go_rules.py:2479` 实测 |
| **22ch 余量** | **1.7×** | |

其中 3 气桶 / 历史交错 / parity 三角波几乎免费（前者复用已在跑的
`_group_liberty_count` flood），贵的是 `iterLadders`（**3 块盘面**）与
`calculateArea`（1 块）。

> ⚠ **降级档位（预先约定，不预先实现）**：若 `iterLadders` 使 22ch 严重超预算，
> 降到 **18 通道**（ch14–17 恒 0）。主干 / 四头 / 12 项 loss / 参数量
> **5,561,832 一行都不用改**。

### 5.3 局级 sidecar `games.npz`（新增，≈1 MB）

替代原 §5.3「局表同 npz」。**独立文件，主 npz 不动。**

| 键 | 类型 | 语义 | 来源 |
|---|---|---|---|
| `g_komi` | f16 `(G,)` | SGF `KM` | `sgf_parser`（**已解析**） |
| `g_score` | f16 `(G,)` | `RE` 数值（黑−白，含 komi）；认输 / 无结果 / 未知 → `NaN` | `sgf_parser`（需新增数值解析） |
| `g_rules` | i8 `(G,)` | bit0 计分(0 area/1 territory)、bit1 tax、bit2 ko、bit3 suicide、bit4 button | **需新增 `RU` 解析**（实测覆盖 45%） |
| `g_resign` | bool `(G,)` | `RE` 形如 `B+R`/`W+R`/`B+Resign`/`W+Resign`/`B+T` | `sgf_parser` |

`G = 162,298` ⇒ **≈1.2 MB**（不再是 176 MB —— 见下方「已砍掉的部分」）。

**读取口径**：`sidecar[game_ids[idxs]]`。**局级**而非逐行存储的理由：
逐行存是 3.6 B/行（34.2M ⇒ 123 GB）。

#### 5.3.0 已砍掉的部分：ownership / scoring / seki 不从 SGF 造

原设计让 sidecar 带 `g_ownership` / `g_scoring` / `g_seki`（终局盘面 +
`calculateArea` / `fillScoring` / `calculateIndependentLifeArea`）。
**2026-10-02 决定砍掉**，理由是实测（见 §9.9）：

1. **性价比低**：那 8–28 min 的 SGF 重放 + 两个最难移植的算法
   （`calculateIndependentLifeArea`、`fillScoring`）只买到 **2.5 项** loss
   （#4 ownership、#10 scoring、#12 seki），而 #10 已因分差缺失在 75% 的行上废了。
2. **信号质量反而更差**：`go_rules.py:2269` 的 docstring 自陈「没有死子判定，
   对局双方应在 pass 认输前实际提掉对方的死子，否则死子所在点会被判为双方共邻的
   中立区域」。而实测 **75.2% 的局是认输** ⇒ 从认输棋谱反推的终局归属不可靠。
3. **段 2/3 有更好的来源**：`katago/stdata` 的 `valueTargetsNCHW` 直接带
   ownership / seki / futurepos / scoring（实测值域 {−1,0,+1} 与 ±120），
   来自搜索器自己的评分器，**不需要我们移植任何东西**。

⇒ **训练责任从 34.2M 挪到 3.4M**，而那 3.4M 上这些信号质量高得多。

#### 5.3.1 `g_komi` 不能假设常数

实测 `KM` 分布（133,604 局）：

| KM | 占比 | | KM | 占比 |
|---|---:|---|---|---:|
| 7.5 | 37.7% | | 0 | **5.9%** |
| 6.5 | 18.0% | | 2.8 | 5.3% |
| 5.5 | 15.1% | | 4.5 | 1.8% |
| 3.8 | 10.5% | | 缺 | 2.2% |

⇒ 贴目五花八门，**必须逐局存**。全局 ch5 = `currentSelfKomi/20` 逐局变化。

#### 5.3.2 `g_rules` 的三种制度（`RU` 覆盖仅 45%）

| RU | 局数 | 占比 | 计分 | 本设计处理 |
|---|---:|---:|---|---|
| **无 RU** | 73,521 | **55.0%** | 未知 | 默认 **AREA**（§2.5） |
| `Chinese` | 50,291 | 37.6% | **AREA** | AREA |
| `Japanese` | 9,789 | **7.3%** | **TERRITORY** | **TERRITORY** ⇒ ch18/19 恒 0 |
| 空 `RU[]` | 3 | 0.0% | — | 默认 AREA |

⚠ 按 §2.5，**territory 计分的对局 ch18/19 在正常阶段恒 0 —— 这是对齐官方，
不是缺陷**。所以模型会见到混合分布（92.6% 有 area 平面、7.3% 全 0），而全局
ch9 那个通道正好编码了「是否 territory 计分」，模型有能力自己学。

⚠ **tax / ko 规则 / suicide / button 无来源**（这四个在 `RU` 字符串里，但
实测 55% 的局没有 `RU`）⇒ 一律按 §2.5 默认：tax **none**、ko **simple**、
suicide **false**、button **0**。这与「简单局」的假设自洽。

#### 5.3.3 `g_score`：解析容易但**覆盖率是硬伤**

实测 `RE` 格式（133,604 局）：

| RE 模式 | 局数 | 占比 | 语义 |
|---|---:|---:|---|
| `B+R` / `W+R` / `B+Resign` / `W+Resign` | 98,352 | **75.2%** | **认输，无分差** |
| `B<N>` / `W<N>`（如 `B+2.5`） | 24,748 | **18.5%** | 有分差 |
| `B+T` / `W+T` | 271 | 0.2% | 超时认输 |
| `draw` / `B+` / `W+F` 等 | 211 | 0.2% | 边角 |

🔴 **12 项里有 5 项吃 `w_score`**（#5/#6 scorebelief、#8 scoremean、
#9 lead、#10 scoring），而 §4.6 规定 `g_score` 为 `NaN` 时 `w_score = 0`
⇒ **这 5 项在 75% 的局上权重为 0**。有分差监督的约
`24.8% × 162,298 局 × 210 行 ≈ 6.6M 行`（34.2M 的 **19%**）。

⚠ `parse_result_to_value` 现状只判 `B→+1 / W→−1 / 0→0 / else None`，
**不含数值** ⇒ `g_score` 需要新增一次 `RE` 数值解析（注意 `B<N>` 里的 `<N>`
被我抽样脚本归一成了 `N`，实际是 `B+2.5` 这类浮点，可能带 `+`/`-`/小数）。

#### 5.3.4 `game_ids` 对齐：哈希锚点法

⚠ **不能用「重新枚举 SGF 按同样顺序」**：`build_dataset.build()`
（`build_dataset.py:261-279`）有一串过滤（棋盘过大 / 让子棋 / 非法坐标 /
缺 `RE`），实测 tgz 前 400 个成员丢掉 179 个（**45%**），而 `tarfile` 成员
顺序 ≠ glob 顺序 ⇒ 顺序一变就全错位。更糟的是 `build_dataset.py:314-316`
的 **id 复用 bug**（`board.play()` 失败时 `return 0,1` 但已追加的行留在
`cur`、`game_id_counter` 未自增）让「局数」本身对不上。

**解法：哈希锚点。** 复用 `src/data/pos_hash.py`（**唯一事实来源**，
`probe0_join.py` 与 `label_sfg.py` 都从它 import）：

1. memmap 主 npz 的 `game_ids`（137 MB），切出每局行区间 `[start_g, end_g)`
2. 每局取**中段锚点行**（`start_g + min(20, len//2)`），算其 `pos_hash`
3. 独立扫 169,878 个 SGF：每局重放到**同一 ply**，算 `pos_hash`，
   建 `pos_hash → sgf` 排序表
4. 二分匹配 ⇒ `sidecar[g] = 那个 SGF 的元数据`

锚点可行性已核实：**100%** 的局 ≥20 手（162,296 / 162,298）⇒ ply-20 锚点
对所有局安全。64 位散列碰撞可忽略，且此法**对枚举顺序与 id 复用 bug
完全免疫**。

**必须输出覆盖报告**（匹配率 < 100% 时，未匹配的局
`w_ownership`/`w_seki`/`w_scoring` 置 0，不静默填垃圾）。

### 5.4 邻行 gather：5 个偏移

ladder 通道按定义需要多块盘面；futurepos 标签按定义需要未来盘面。

| 偏移 | 用途 | 守卫 |
|---|---|---|
| `i−2` | ch16（前一**二**手盘上的梯子） | `game_ids[i−2]==game_ids[i]` |
| `i−1` | ch15（前一**手**盘上的梯子） | `game_ids[i−1]==game_ids[i]` |
| `i+8` | futurepos 通道 0 | `game_ids[i+8]==game_ids[i]` |
| `i+32` | futurepos 通道 1 | `game_ids[i+32]==game_ids[i]` |

合法性掩码天然给出官方的 `maxTurnsOfHistoryToInclude` 门：
**ply 0/1 自动落入 §2.2 的回退复制分支**（`prevBoard = board`、
`prevPrevBoard = prevBoard`），**不是置 0**。

⚠ **必须批量做**（把 `idxs` 扩成偏移并集 → 一次 `feature_planes` → gather
回原行），**绝不逐样本循环**，否则批量优势全丢。

> 🎁 **futurepos 的实际成本只有 4%。** 每局行数 p50=208 / mean=210.7：
> `+8` 有效 96.2%（`[0, L−9]`）、`+32` 有效 84.8%（`[0, L−33]`）。
> ⇒ **gather 全部行是免费的**（`boards` 常驻内存，就是索引），
> 而**只有靠近终局的那 4% 行需要真跑一次 area 的 flood fill**。
> 实现上按行掩码分派：`valid8` / `valid32` 的子集才算，其余 target 填 0、
> `w_futurepos` 置 0。

> ⚠ **`np.load('x.npz')['boards']` 会把整个成员解压进内存**（12.3 GB），
> 不是按需。本机 13.9 GB ⇒ 任何分块读都会先吃掉 12.3 GB，余量 1.6 GB。
> ⇒ 必须经 `kata_label_join.materialize_dataset()` 落成 `.npy` 后
> `mmap_mode='r'` 按需分页（落盘 152s，实测 478K 行/s）。

### 5.5 批契约：**4 元组 + dict**（替代 3 元组 / 5 元组）

```python
sample_batch_numpy(idxs, rng=None, augment=True, labels=False)
```
- `labels=False`（**默认**）→ **三元组** `(states, moves_out, values)`
  ⇒ `bench_train.py:81/105`、`train_sft_ms.py:101/214`、
  `train_sft.py:822/886/1042/2920` **逐位不变**
- `labels=True` → **4 元组** `(states, moves_out, values, labels_dict)`，
  `labels_dict` 见 §4.6 的键 + 软标签两项：

```
next_move · outcome · score · sb_center · sb_upper · global
ownership · scoring · seki · future · w{...} · game_weight
soft (B,362) f32        # KataGo 访问分布，仅被标注的行有效
soft_mask (B,) f32      # 1 = 该行有软标签
```

> ⚠ **原设计是 5 元组 `(states, moves, values, soft, mask)`，已改。**
> 5 元组**没有 dict 的位置**，而 V7 的 loss 需要那个 dict ⇒ 最后只会变
> 6 元组或 fork 两条路。4 元组 + dict 让预取器只需搬运**一种** payload 形状。

> ⚠ **三个假 dataset 必须同步补 `labels=False` 形参**：
> `tests/test_eval_coverage.py:75`、`tests/test_eval_determinism.py:111`、
> `tests/test_eval_no_augment.py:219`。默认 `False` 时无害，一旦 `train_sft.py`
> 任何一处传 `labels=True` 就会 `TypeError`。

### 5.6 对称增广

8 路 dihedral 下：`states`、`ownership`/`scoring`/`seki`/`future`、
`soft` 走**同一个** `tforms`；`moves`/`next_move` 按 `SYMMETRIES` 映射
（**pass 不动**）；`outcome`/`score`/`sb_*`/`global`/`w`/`soft_mask` **不变**。

> ⚠ **`permute_soft` 是软标签接入里唯一会静默出错的地方**（已实现并测）。
> `states` 施加增强、`moves_out` 施加重映射，软策略不同步重排则
> **训练不报任何错**，只是标签指向错误的点 ⇒ 表现为「loss 正常下降、
> 指标好看，但棋力不涨」。
> 已踩过的坑：方向极易搞反。`out[:, perms[t]] = soft[:, :]`（gather）得到的
> 是**逆变**；正确是显式求逆 `inv[perms[t][k]] = k` 再取 `out[:, :A-1] = soft[:, inv[t]]`。
> 见 `src/data/dataset.py::permute_soft` 与 `tests/test_soft_labels.py`（8 项）。

### 5.7 软标签（KataGo 访问分布）

`scripts/label_sfg.py` 产出 `kata_labels.npz`（8 键）：

| 键 | 类型 | 语义 |
|---|---|---|
| `policy` | f16 `(N,362)` | 归一化访问分布（361 落点 + pass），行和 ≈ 1.0 |
| `pos_hash` | u64 `(N,)` | **join 键** |
| `root_win` / `score_mean` / `score_stdev` | f32 `(N,)` | 根搜索量 |
| `visits` | i16 `(N,)` | 实际达成 visits |
| `game_idx` / `move_idx` | i32 `(N,)` | 局号 / 手号（**仅诊断，不作 join 键**） |

**join 键必须是 `pos_hash`，不能是 `game_id`**（理由同 §5.3.1）。散列口径 =
`(board 361B, to_play 1B, ko 2B)`，实现在 `src/data/pos_hash.py`
（**唯一事实来源**；`probe0_join.py` 与 `label_sfg.py` 都从它 import，
已验证逐位一致）。

**实测验证**（`tmp/bench/verify.npz`，800 位置 / 10 局 / move 100–199）：
4M 行处命中 100/800，命中行号连续（首局落在第 3,350,229 行），散列**逐位一致**。
散列吞吐 **478,603 行/s**，全量 34.2M 约 72s。

> ⚠ **「前 2M 行命中 0」不表示键不匹配。** `label_sfg.py` 的 `build_queries`
> 用 `rng.sample(paths, N)` **随机抽局**，实测这 10 局在 133,604 局里的下标是
> `[10612, 57263, 67873, 79511, 93860, 100989, 106151, 110250, 124937, 127383]`，
> 全在 2M 行（约 7,800 局）覆盖之外。反推比例：10,612 局 ↔ 3.35M 行
> ⇒ **约 316 行/局**，34.2M 行约覆盖 108,000 局。

**join 缓存**（`scripts/build_soft_index.py`）：key 必须含
`(dataset size+mtime, labels size+mtime, max_repeats, pos_hash 版本号)`，
不匹配即失效重算。⚠ 换一批 SGF 或换标签文件后旧缓存会**静默挂错标签**，
而症状是「命中率 0」，看不出是缓存陈旧。

**掩码语义**：`soft_mask` 是**逐行二选一**（1 = 软 CE，0 = one-hot CE），
**不做混合**。理由：同一个 head 同时收到「搜索分布」与「人类 one-hot」
会去折中而不是学搜索。段 2 用独立采样器**只取软行**
（`--soft-only-sampling`，与 Phase 4 item 3 的窗口**正交**，两者可叠加）。

**policy 保持 K=2**：段 2（1% KataGo 标注）与段 3（stdata）的目标**同为访问
分布**，可合并成一轮；人类 one-hot 已在段 1 学进去，段 2/3 覆盖它正是
蒸馏的目的。K=3 只在「同一部位两个 policy 目标同时训」时才需要。
且 K=2 恰好对上 stdata 的两路输出（`policyTargetsNCMove[0]` 本方、
`[1]` 对手；spec 的 `π_opp` = 下一手，在自对弈里与「对手走的那手」是同一个东西）。

### 5.8 原设计的作废项

| 原条目 | 状态 | 理由 |
|---|---|---|
| §5.2 行表新增 `next_move` 列 | **作废** | 由 `moves[i+1]` + 局守卫实时推出 |
| §5.3 局表并进主 npz | **作废** | 改独立 `games.npz`（§5.3） |
| §5.6 必须先修 game_id 残行 bug（阻塞 5.3） | **降级** | 哈希锚点法对 id 复用免疫，不再阻塞。⚠ 但仍建议修：它污染 `train_sft.py:2396` 的按局切分 |
| §5.7 存储增量 5.6 B/行（≈56 MB） | **作废** | 不 rebuild；sidecar 176 MB（局级） |
| §5.1「零新增空间存储」 | **保留并强化** | 明确为训练时 CPU 实时算，不预存 |

## §6 参数与显存预算

### 6.1 参数明细

| 层 | 形状 | 参数 |
|---|---|---:|
| stem `Conv2d` 3×3 | 22→256, bias=F | 50,688 |
| global `Linear` | 19→256, bias=F | 4,864 |
| **nbt2 块内**（每块） | | |
|  ├ NormAct(C) γ,β | 256 | 512 |
|  ├ `normactconvp` 1×1 | 256→128 | 32,768 |
|  ├ 内块 ×2 · MHSA q/k/v/out | 4 × (128→128) | 65,536 |
|  ├ 内块 ×2 · SwiGLU up/gate/down | 128→384 ×2 + 384→128 | 147,456 |
|  ├ NormAct(M) γ,β ×2 | 2×128 | 512 |
|  ├ RMSNorm(128) ×2(内)×2 | 4×128 | 512 |
|  ├ RoPE (4,16,2) ×2(内) | | 256 |
|  └ `normactconvq` 1×1 | 128→256 | 32,768 |
| **块小计 ×11** | | **5,423,616**（单块 493,056） |
| trunk-end RMSNormMask γ,β | 256 | 512 |
| policy 头（含 2×BiasMask、pass 支路） | 256→48, 144→48, 48→2, 144→48→2 | 38,834 |
| value 头（outcome 3 + 分数 3） | 256→48, 144→96, 96→3 ×2 | 26,886 |
| ownership / scoring / futurepos / seki | 48→1, 48→1, 48→2, 48→4 | 384 |
| scorebelief 头 | 144→96, 1→96 ×2, 96→8, 144→8 | 16,048 |
| **合计** | | **5,561,832** |

**预算 5,850,000 → 余量 288,168（4.9%）** ✅

> ⚠ **本表原有一处行标签错误，2026-10-02 已订正。** 原文把 `5,479,680` 标成
> 「块小计 ×11」，但该数**不是 11 的倍数**（`5,479,680 / 11 = 498,152.7`），
> 不可能是块小计。真相是
> `5,479,680 = stem 50,688 + global 4,864 + 块 5,423,616 + trunk-end 512`，
> 而**块自身是 5,423,616**（单块 493,056）。
>
> **合计 5,561,832 与余量 288,168 自始正确**，且 `src/networks/katago_v7.py`
> 的实测值与本表**逐子模块逐位吻合**
> （`tests/test_katago_v7_budget.py::test_total_param_count_is_exact` 断言
> `delta == 0`；`test_block_subtotal_is_not_the_stem_plus_global_sum` 把这个
> 勘误钉死，防止日后有人照错标签对账）。

> 备选：若探针要求 `head_dim=16`，只需 `H=8`（M=128 不变）—— MHSA 参数量与 `H` **无关**，**零参数代价**。

### 6.2 显存预算（B=3200 / 卡 / fp16）

单位换算自 `2026-10-01-npu-vram-utilization-analysis.md` §1.1（`(2000,184,19,19)` fp16 = 0.247 GiB；`(2000,4,361,361)` fp16 = 1.942 GiB），按 B×C 线性放大：

| 项 | 计算 | GiB |
|---|---|---:|
| 参数 / 梯度 / Adam（AMP） | 5.56M | ~0.4 |
| 输入 `states` (3200,22,361) fp16 | | 0.05 |
| labels（ownership/scoring/seki/future fp32） | | 0.03 |
| **梯度检查点边界 11 × (3200,256,361)** | 11 × 0.551 | **6.06** |
| stem / trunk 输出（寿命更长的 2 张） | 2 × 0.551 | 1.10 |
| stem 3×3 im2col（fp16，输入仅 22ch） | 3200×198×361×2 | 0.43 |
| policy 头中间量 | | 0.55 |
| value / ownership / scoring / seki / future 头 | | 0.35 |
| scorebelief `h` (3200,842,96) fp16 | 0.48 ×2（前向+反向） | 0.96 |
| 损失反传临时量（8 项） | | ~1.5 |
| **注意力重算峰值（单站点）** | 2–3 × 3.114 | **6.2–9.3** |
| **小计** | | **17.2–20.8** |
| CANN workspace / 未归属差额（报告 §1.4 实测差 4 GiB） | | +4.0 |
| **估算峰值** | | **21.2–24.8** |

**预算 31.0 GiB → 余量 6.2–9.8 GiB** ✅【算·待验】

**注意力账的前提**：
- §3.5 已定**无 dropout** ⇒ 少掉 dropout 输出 + 掩码两份张量（报告 §1.2 的上界项）；
- 仍走 `sdpa_force_math=True`（`train_sft.py:2269`）⇒ `q@k^T` 与 softmax 输出必须物化 ⇒ `(3200,4,361,361)` fp16 = **3.114 GiB/份**；
- 逐块检查点 ⇒ 22 个 MHSA 站点**同一时刻只有 1 个在重算**，注意力峰值**不随块数增长**。

### 6.3 三变体对比

| | ① transformer-nbt **(选定)** | ② convnet-nbt | ③ hybrid |
|---|---|---|---|
| 参数 | **5,561,832** | 5.57M | 5.38M |
| 注意力站点 | 22（单活） | 0 | 3（单活） |
| 检查点边界 | 11 × 0.551 = 6.06 | 18 × 0.391 = 7.04 | 15 × 0.445 = 6.68 |
| 3×3 im2col | 0.43（仅 stem） | ~1.7（M=88 内块） | 中 |
| **估算峰值** | **21–25 GiB** | 15–18 GiB | 18–22 GiB |
| 官方对应 | `b10c384h6nbttflrs-fson-silu-rsnh` 缩放 | `b18c176nbt-fson-mish` | 无 |
| 结论 | ✅ 语义最贴官方 + 预算内 | 最省显存但**无注意力**、im2col 最重 | 非官方，工作量最大 |

**② 比 ① 省约 6–7 GiB，但代价是彻底放弃注意力**，且 im2col 峰值最高（报告 §2.2：`(2000,184,361)` fp16 = 2.39 GB 且**不进 torch 显存账本**）。① 在预算内 ⇒ 不为省显存放弃注意力。

### 6.4 风险与降级路径

| # | 风险 | 处置 |
|---|---|---|
| **R1（首要）** | CANN 融合注意力是否对 `head_dim=32` 生效 —— 现有 **46 ∉ 支持集 {16,32,64}**（报告 §O1 判为 `force_math` 根因），**必须实机验证** | **计划 Task 0 = 910A 探针**：查 `npu_fusion_attention_score` / `npu_flash_attention_score`，A/B `sdpa_force_math`。若融合可用且能关 `force_math`：注意力 6.2–9.3 → **~0.7 GiB**，总峰值 **21–25 → 15–19 GiB** |
| R2 | 实测仍 >31.0 GiB | **依次**：a) batch 3200→2800（激活 −12%）；b) 块数 11→9（边界 −1.1 GiB，参数 4.5M）；c) 才考虑退回 ② |
| R3 | 无 dropout 在 SFT 上过拟合 | 正则靠 weight decay + `game_weight` + 数据量；报告 §O1 已列 dropout 为融合障碍，关掉是双重收益 |
| R4 | im2col 未入账（§2.2） | ① 只有 stem 一处 3×3，且输入仅 22ch ⇒ 0.43 GiB，可控；探针时一并记录 |

**结论**：① 在 31.0 GiB 预算内有 **6–10 GiB 余量**，且最大不确定性（R1）**只会让数字更好** —— 最坏情形就是维持现在的 `force_math`。

---

## §7 回归影响与验收

### 7.1 不变量（零改动清单）

| 对象 | 约束 |
|---|---|
| `GoBoard.feature_planes`（`go_rules.py:2420`，17ch）与 `feature_planes_batched` | **一字不改**；`test_go_feature_planes_v21.py` 18 项全绿（含 `p.shape == (17,n,n)`） |
| `feature_planes_katago` / `compute_global_features` / `KataGoPolicyHead` / `KataGoValueHead` / `KataGoGPoolResBlock` / `_build_katago_se_blocks` | 现状 **全部不存在**（已核实）⇒ **纯新增，零迁移** |
| `mcts.py:250-253` transposition `cache_key`、Zobrist `hash()` | 不变（本次不引入新盘面状态字段） |
| `sample_batch_numpy` 默认签名 | 仍是三元组（`labels=False` 为默认） |
| 旧 17ch 推理路径 | `inference.py:501` 白名单、`:567 _build_state`、`:674 _forward_batch` 保留 |
| 12ch「位相同」契约 | `test_katago_se.py:189` 继续通过 |

### 7.2 扩展点（复用既有机制）

- **模型分发**：`src/inference.py::register_in_channels_builder(in_channels, builder)` —— 新增 `register_in_channels_builder(22, builder)`；现有 12/14/16/17 分支的「缺接线」报错与提示（`test_katago_se.py:251/267`）自动适用。
- **预算测试**：照抄 `tests/test_v19_budget.py` 形状（脚本 flags → 实例化 → 断言 + 显存余量 ≥ 4.0 GB）。
- **训练脚本**：照 `shell/train_sft_npu_4card_katago_se.sh` 新增。

### 7.3 文件清单

**新增**
| 文件 | 内容 |
|---|---|
| `src/networks/katago_v7.py` | stem、`Nbt2TransformerBlock`、`TransformerBlock`、fson `NormAct`、`RMSNormMask`、2D RoPE、KataGPool、四头 |
| `src/game/go_features_v7.py` | `feature_planes_v7_batched` + 移植的 `iterLadders` / `calculateArea` / `passWouldEndPhase`（**不放 `go_rules.py`**，避免污染 17ch 实现所在文件） |
| `shell/train_sft_npu_4card_katago_v7.sh` | 新脚本 |
| `tests/test_feature_planes_v7.py` | 22 通道形状/等价性 |
| `tests/test_katago_v7_loss.py` | 逐项系数对照 §4.5 |
| `tests/test_dataset_v7_labels.py` | 局表 join、权重掩码、增广 |
| `tests/test_katago_v7_budget.py` | 参数/显存预算守卫 |

**修改**
| 文件 | 改动 |
|---|---|
| `src/data/sgf_parser.py` | 解析 `RU` 规则、`RE` 分差 |
| `scripts/build_dataset.py` | **不再改动**（§5.8：不 rebuild）。仅**建议**修 `:314-316` 的 game_id 残行 —— 它不再阻塞，但污染 `train_sft.py:2396` 的按局切分 |
| `src/data/dataset.py` | 加载 `games.npz` 局表、邻行 gather（5 偏移）、`labels=True` 返回 **4 元组 + dict**、增广下的标签变换、软标签挂载 |
| `src/data/sgf_parser.py` | 新增 `RU` 解析（`g_rules` 需要；缺 `RU` 时按 §2.5 默认 AREA/none/simple/false/0） |
| `scripts/train_sft.py` | V7 loss 装配、`labels=True` 接线、预取 `pf.next()` 透传 dict、切 22 通道 |
| `src/inference.py` | 注册 22ch builder |

**不动**：`data/sgf_19x19_full.npz`（**不 rebuild**，§5.0）、
`go_rules.py` 的 `feature_planes`/`feature_planes_batched`、`mcts.py`、`search/`、
`test_go_feature_planes_v21.py`、`test_param_budget.py`（钉的是 v18 = 12,858,978，与本模型无关）。

⚠ **`scripts/search_arch.py` 不可直接复用**：其 `ANCHOR`/`project()` 用 **B=2000, C=184 的旧卷积+注意力架构**标定；须**新增投影函数**（按 §6.2 分项表实现）。

⚠ **`docs/` 已被 `.gitignore` 排除**（`:46 /docs`）⇒ 本 spec 与新增的
`games.npz` 生成脚本都不在版本控制内。已提请注意，未擅自改动该规则。

### 7.4 验收标准

**本地必绿**
1. `pytest tests` 全绿（当前基线 **1011 passed / 2 skipped**）
2. `bash -n shell/*.sh`
3. `pytest tests/test_katago_v7_budget.py`
   - 参数量 **`== 5_561_832`**（精确值，非区间）—— 已实现并通过
   - `≤ 5_850_000`
   - 新投影在 `B=3200` 下余量 `≥ 4.0 GiB`（对 31.0 上限）
4. `pytest tests/test_go_feature_planes_v21.py tests/test_katago_se.py` —— 旧路径零回归
5. `register_in_channels_builder(22, ...)` 生效 —— **已实现并通过**
   （22 → `NbtTfNet` 5,561,832；12 → 旧 `AlphaGoNet`；17ch 分支不受影响）
6. 🔴 **验收 #6 已替换**（§5.2 决定不再自己算 ch0–13 ⇒ 「与旧 17ch 逐位相等」
   不再是回归锚点）。**新锚点 = 与官方 stdata 逐位对拍**：
   - `feature_v7(...)` 形状 `(22,19,19)`，`dtype` 与 `feature_planes_batched` 一致（fp16）
   - **ch14、ch17 与 `katago/stdata` 的同名通道逐位相等**（当前盘通道，stdata 可直接验证）
   - **ch15、ch16 对 SGF 重放逐位相等**（stdata 无前一/二盘的盘面，无法直接对拍；
     此处拆成「算法由 ch14 担保 + 盘面由重放担保」两项，见 §9.4）
   - **ch0–6、9–13、18–19 与 stdata 逐位相等**（交叉统计）
7. `pytest tests/test_katago_v7_loss.py`：12 项系数逐条断言 `1.0 / 0.15 / 1.20 / 1.5 / 0.02 / 0.02 / 0.001 / 0.0015 / 0.006 / 0.25 / 0.25 / seki=8·0.005/(0.005+EMA)`
   —— **已实现并通过**（24 项）
8. `labels=False` 时 `sample_batch_numpy` 返回仍可被 3 元组解包（`bench_train.py:81` 不改）
9. `pytest tests/test_dataset_v7_labels.py`：`games.npz` 局表 gather、权重掩码、
   增广下的标签变换、4 元组契约（`soft`/`soft_mask` 在 dict 内）、软标签掩码二选一
10. `pytest tests/test_games_sidecar_alignment.py`：哈希锚点匹配率**必须打印**；
    匹配项的 `g_komi`/`g_score` 与 SGF 直读一致
11. `pytest tests/test_katago_npz.py` + `tests/test_kata_label_join.py`：
    stdata 布局分派（64/80 列、有无 Q 值）、`pos_hash` join、`permute_soft` 双向

**远端必验（Task 0，本地无法替代）**
12. 910A 探针：`npu_fusion_attention_score` / `npu_flash_attention_score` 存在性；`sdpa_force_math` 关闭后前反向数值一致
13. `max_memory_allocated ≤ 31.0 GiB @ B=3200/卡` —— **实测值回填** `test_katago_v7_budget.py` 的估算项
14. **22ch 特征预取吞吐 ≥ NPU 吞吐**（C0 标定 `--prefetch-workers` 与 B）——
    3.04 ms/行 是 8 worker 下的预算，超出则提 worker 或降 B（§5.2）

### 7.5 已识别的冲突

| 冲突 | 处置 |
|---|---|
| `test_grad_checkpointing.py`（32 项）/ `test_attn_dropout_eval.py`（5 项）只覆盖旧模型 | 新模型无 dropout、检查点粒度不同 ⇒ **必须新增** nbt2 块的检查点测试 |
| `test_param_budget.py` 钉 v18 | 不改；v7 用独立测试锁自己的数 |
| `search_arch.py` 标定失配 | 新增投影（§7.3） |
| `test_katago_se.py:13` 引用了不存在的 `test_v21_budget.py` | 顺手补上，否则文档指向悬空 |

---

## §8 分期与旧计划处置

### 8.1 旧交付物处置

| 文件 | 处置 | 理由 |
|---|---|---|
| `docs/superpowers/specs/2026-10-01-katago-se-rebuild-design.md` | 文件头加 **superseded 横幅**，指向本 spec | 输入从 17ch SE 改为官方 **22/19 V7**；主干从卷积 SE 改为官方 **nbt+transformer**；heads/loss 全部对齐官方 |
| `docs/superpowers/plans/2026-10-01-katago-se-phase1.md` | 同上 | Tasks 1–10 **零实现**（相关符号全部缺失，已核实）⇒ **无迁移成本，整份作废不打折** |
| `.gitignore:45` `/docs` 规则 | **不碰**（既定指令） | — |

**新交付物**
- Spec：`docs/superpowers/specs/2026-10-01-katago-nbt-tf-design.md`（本文件）
- Plan：`docs/superpowers/plans/2026-10-01-katago-nbt-tf-phase1.md`（待写）

### 8.2 分期

**Phase 0 — 远端探针（阻塞项，不写模型代码）**
| | 任务 | 产出 |
|---|---|---|
| 0.1 | 910A 查 `npu_fusion_attention_score` / `npu_flash_attention_score`，`head_dim=32` 能否走融合 | 探针报告 |
| 0.2 | `sdpa_force_math` 开/关 **A/B 数值一致性** | 结论：能否关 |
| 0.3 | 单层 MHSA @ `B=3200` 显存实测，校准 §6.2 的 21–25 GiB 估算 | **实测值回填** `test_katago_v7_budget.py` |

Phase 0 **不阻塞** Phase 1/2 —— §6 已按最坏情形（必须 `force_math`）做预算，探针只可能让数字更好。
模板：`docs/superpowers/plans/2026-10-01-npu-vram-observability.md`。

**Phase 1 — 输入与标签（数据侧）**

1.1 `sgf_parser`：解析 `RU` 规则、`RE` 分差
1.2 `build_dataset`：**先修 game_id 残行 bug**（§5.6，阻塞 1.3）
1.3 `build_dataset`：`next_move` 列 + 局表 `g_*`
1.4 `go_features_v7.py`：`feature_planes_v7_batched` 的 **ch0–6、9–13**（与旧 17ch **逐位相等**，验收 #6）
1.5 移植 `calculateArea` → ch18/19；`passWouldEndPhase` → global14
1.6 移植 `iterLadders` → ch14–17（含 §2.2 回退复制语义）
1.7 19 维全局特征（含 global5 的 `currentSelfKomi` 符号约定）
1.8 `dataset.py`：局表 join、邻行 gather、`labels=True` 分支
1.9 对称增广下的标签变换
1.10 测试：`test_feature_planes_v7.py`、`test_dataset_v7_labels.py`

**Phase 2 — 模型与损失（网络侧）**

2.1 `katago_v7.py`：fson `NormAct`、`RMSNormMask`、2D RoPE、`TransformerBlock`
2.2 `Nbt2TransformerBlock` + stem + trunk-end（§3.3 的 K 调度）
2.3 四个头（§4.1–4.4）
2.4 `register_in_channels_builder(22, ...)`
2.5 V7 loss 装配（§4.5 的 12 项系数）
2.6 `train_sft.py` 接线 + `shell/train_sft_npu_4card_katago_v7.sh`
2.7 `test_katago_v7_budget.py` + `test_katago_v7_loss.py`
2.8 顺手补 `test_v21_budget.py`（修 §7.5 的悬空引用）

**Phase 3 — 明确不做（本 spec 范围外）**
- 13 项 search-dependent loss（需 MCTS reanalysis 数据）
- 非 19×19 泛化（`KataGPool` 的 board-size 缩放项目前是常数）
- 融合注意力的深度优化

**沿用的旧知识**：旧计划 Task 4 的 pass bug —— pass **不得**重置 `moves_since_capture`（应 `+=1`）且 `consecutive_passes +=1`。

### 8.3 回到设计的触发条件

| 触发 | 回到 |
|---|---|
| Phase 0 实测 `>31.0 GiB @ B=3200` | §6 R2 降级路径（减 batch → 减块数 → 才换结构） |
| `iterLadders` 移植在 19×19 上性能不可接受 | §2：ch14–17 是否降级为常数 0 |
| SFT 数据量不足以支撑无 dropout | §3：是否恢复 `attn_dropout`（代价是融合受阻 + 3.1 GiB） |
| §5.4 的 `labels` 字典在预取链路上成为瓶颈 | §5.4：批契约重议 |

### 8.4 交接

本 spec 复核通过 → 转 `writing-plans` 产出 `2026-10-01-katago-nbt-tf-phase1.md`（含 Task 0 远端探针、§8.2 的依赖顺序与 blocking edges）。

---

## §9 实测发现附录（2026-10-02）

> 本节记录**从真实数据反解出来的约定**。每一条都不是「看起来显然」的东西，
> 而且错了之后**不报错** —— 只会静默地算错。
> 全部有单测钉住（文件名列在每条末尾）。

### 9.1 `katago/stdata` 的 npz schema

KataGo 官方 distributed training 数据的落盘格式。**输入与全部监督目标都在
同一个 npz 里** ⇒ 本仓**不需要**移植 `iterLadders`/`calculateArea`/
`passWouldEndPhase` 就能用官方输入（这条已被 §5.2 的决定取代，但读 stdata
训练仍然用它）。

三批实测规模：

| 批次 | 网络 | 文件数 | 行/文件 | **总行数** | seki(ch1) 非零行 |
|---|---|---:|---:|---:|---|
| 2026-07-30 | `kata1-zhizi-b40c768nbt` | 16,497 | 66 | **1.09M** | 28 / 2,206 |
| 2026-08-25 | `kata1-tf3-b11c768` | 57,386 | 60 | **3.45M** | 53 / 2,235 |
| zzb28c512 | `zzb28c512nfd4` | 30 | 4,374 | **131K** | **104,905 / 131,706** |
| | | | **合计** | **4.67M** | |

**七个键**（`zzb28c512` 缺 `qValueTargetsNCMove`，是最早的格式）：

```
binaryInputNCHWPacked uint8  (N, 22, 46)    ← 22 空间通道
globalInputNC          float32 (N, 19)      ← 19 全局特征
policyTargetsNCMove    int16  (N, 2, 362)   ← K=2
globalTargetsNC        float32 (N, 64|80)   ← ⚠ 列数随网络版本变
scoreDistrN            int8   (N, 842)
valueTargetsNCHW       int8   (N, 5, 19,19)
qValueTargetsNCMove    int16  (N, 3, 362)
```

⇒ **22 空间 / 19 全局 / K=2 / 842 桶 —— 与本 spec 逐项吻合。**

⚠ **四条实测约定**（`src/data/katago_npz.py`）：

1. **bit-packed 空间通道**：`(N,22,46)` uint8，每通道 46 字节 = 368 bit，
   **只用前 361 bit**；`np.unpackbits(axis=-1, bitorder='big')` 后
   `reshape(19,19)` 得行主序盘面。
   ⚠ `np.unpackbits` 的 `axis` 默认 `None`（**整体展平**），对 `(N,46)` 输入
   会静默算错 —— 必须显式 `axis=-1`。
2. **棋盘尺寸从 ch0（on-board 掩码）反推**，非方阵标 0。
   **实测 19×19 只占 63~75%**（b40c768nbt 75.3% / tf3-b11c768 63.1% /
   b28c512 74.0%），其余是 7~18 盘与 3.7~4.5% 非方阵。
   ⇒ V7 固定 19×19 时**可用量从 4.67M 降到 ≈3.10M**。
3. **策略索引是固定 stride=19**（`r*19+c`），**不是** `r*s+c`。
   实测 9/11/13 盘的非零索引最大到 156/176/200，**全部 > s²**，
   但都 < `(s−1)·19+(s−1)`。
   ⇒ 与本仓 `build_dataset.py:297` 的 `(r+off)·board_size+(c+off)`
   **同一口径**（19×19 局 `off=0`），两边索引可直接对齐。
4. **`globalTargetsNC` 列数随网络版本变**：**64**（b40c768nbt / b28c512）
   vs **80**（tf3-b11c768）；`qValueTargetsNCMove` 只有前两批有。
   ⇒ **绝不能硬编码列号**，必须按网络名查 `GLOBAL_TARGET_LAYOUT`；
   未登记的网络**报错**不猜。

⇒ 实现：`src/data/katago_npz.py`；测试：`tests/test_katago_npz.py`（27 项，
含两条真实 stdata 数据对拍）。

### 9.2 `globalTargetsNC` 的列语义（实测解码）

**已确认**（两个网络版本上位置一致）：

| 列 | 语义 | 证据 |
|---|---|---|
| 0,1,2 | 硬 outcome（胜/负/无结果） | 0/1 稀疏；`sum(0:3)==0` ⟺ 和棋 |
| 3 | scoreMean（搜索侧连续估计） | 值域 ±17.27 / ±31.68 |
| 4–19 | **四组分差分布**，每组 `(P, P, P, 分数)`，**三 P 之和恒为 1.000000** | `sum(cols[4,5,6])` min=max=1.0 |
| **20** | **实际终局分差**（建 scorebelief 目标用；**和棋记 0**） | 见 §9.3 |
| 21 | lead（非零率 39%） | — |
| 22 | 方差量（与分差相关 **−0.036**） | — |
| 25 / 26 | `global_weight` / `w_policy_player` | 恒 1 |
| 27 / 28 | `w_ownership` / `w_policy_opp` | 0/1，各 91% |
| **29** | **`w_lead`** | **非零数与 col21 的 lead 非零数逐行相等**（交叉验证通过） |
| 33 / 34 | `w_futurepos` / `w_scoring` | 0/1 |
| 35 | `w_value` | **恒 0** |
| 47 | komi（±7.5） | — |

**修正一处早期误判**：`col20` 常被误当成「scoreMean 的第二个估计」。
实测它是**实际终局分差**（和棋时恒 0，且 scorebelief 目标恰为中心两桶 50/50）；
`col3` 才是 scoreMean。已在 `katago_npz.py` docstring 记为反面记录。

**仍未确定**：`col20` 之外的 `col3`（连续估计的精确语义）与 `col22`
（方差量的具体身份，疑为 variance time）—— 需查 KataGo `dataio.cpp`
的写入顺序才能定，暂不作为任何 loss 的目标。

### 9.3 scorebelief 的桶偏移（**从官方数据反解**，不是猜）

```
center    = round(score)              整数分差
λ         = score − (center − 0.5)     ∈ [0, 1)
upperProp = round(λ·100)              ∈ [0, 100]
bin_lo    = center + 420              （420 = mid − 1，mid = 421）
bin_hi    = center + 421
target[bin_lo] = (100 − upperProp)/100
target[bin_hi] = upperProp/100
```

从 `2026-08-25npzs.tgz` 取 **400 个「恰有 2 个非零且相邻」**的 `scoreDistr` 行，
与 `globalTargetsNC[:,20]` 配对：

| score | center | upperProp | 实测 bins | 实测 frac_lo | 公式预测 |
|---:|---:|---:|---|---:|---|
| −31.6798 | −32 | 82 | (388, 389) ✓ | 0.18 ✓ | 0.18 |
| −17.2670 | −17 | 23 | (403, 404) ✓ | 0.77 ✓ | 0.77 |
| −6.3197 | −6 | 18 | (414, 415) ✓ | 0.82 ✓ | 0.82 |
| −3.3996 | −3 | 10 | (417, 418) ✓ | 0.90 ✓ | 0.90 |
| +1.4250 | 1 | 92 | (421, 422) ✓ | 0.08 ✓ | 0.08 |
| **0（和棋）** | 0 | 50 | **(420, 421)** ✓ | **0.50** ✓ | 0.50 |

⇒ **和棋自动退化成中心两桶 50/50，不需要任何特判。**

⚠ **spec §4.4 的 `0.05·(i−mid+0.5)` 不是分差映射**，它是网络侧 scorebelief
头的**输入特征坐标**（桶下标自变量）。两者不矛盾。

实现：`katago_v7_loss.py::build_score_distr_target`；
测试：`test_katago_v7_loss.py::test_score_distr_bin_offset_matches_official`
（5 行逐位对拍）+ `test_draw_degenerates_to_two_center_bins`。

### 9.4 ladder 通道的可验证性拆分（**无 KataGo 源码**）

`katago/` 下**只有 Windows 二进制**，没有 `cpp/`，所以
`nninputs.cpp::fillRowV7` **不可读**。⇒ stdata 是唯一可执行 oracle。

而 stdata 是**逐行独立样本**，没有前一/二手的盘面 ⇒ 4 个 ladder 通道里
**只有 2 个能直接对拍**：

| 通道 | 依赖盘面 | 能否对 stdata 直接验证 |
|---|---|---|
| ch14 | 当前盘 | ✅ 可（stdata 的 ch1−ch2 即可还原当前盘） |
| ch17 | 当前盘（working-move） | ✅ 可 |
| ch15 | 前一手盘 | ❌ **不可** |
| ch16 | 前二手盘 | ❌ 同上 |

⇒ **拆成两个各自可测的问题**：

- **算法对不对** ⇒ 拿 stdata 的 ch14 逐位对拍我们的 `iterLadders(current_board)`。
  stdata 有 ≈3.1M 个 19×19 行，样本量足够。
- **喂的盘面对不对** ⇒ 用 SGF 重放验证 `boards[i−1]` 真的等于该局第 i−1 手盘面。
  纯本地测试，不需要任何 oracle。

ch15/ch16 复用**同一份已验证的算法代码**，只换喂进去的盘面 ⇒
「算法」由 ch14 担保，「盘面」由重放担保。**不需要 KataGo 源码也能把
4 个通道的可信度拆开验证。**

⚠ 唯一无法完全验证的是 KataGo 那边 ch15/ch16 的**具体盘面取法**
（取 `getRecentBoard(1)` 还是别的），只能信 §2.2 引的源码记载。

### 9.5 `valueTargetsNCHW` 五通道的实测形态

| ch | 语义 | 值域 | 非零率（tf3 实测） |
|---|---|---|---|
| 0 | ownership | {−1, 0, +1} | 0.35–0.73（按文件变化大） |
| 1 | seki | {−1, 0, +1} | **~1e-4**（极稀有） |
| 2 / 3 | futurepos | {−1, 0, +1} | 0.13 / 0.16 |
| 4 | scoring | **±120** | 同 ownership（印证 spec 的 /120 缩放） |

⚠ **seki 极稀有，且 `zzb28c512` 那批是 seki 富集的**（131,706 行里 104,905 行
含 seki 格子，前两批只有几十行）。

⇒ **设计后果**：spec §4.5 #12 让 seki 复用 `w_ownership`，那是「从 SGF 自造
标签」时的口径。在 stdata 上 `seki` 通道几乎全 0 而 `w_ownership` 在 91% 的行
上是 1 ⇒ 复用等于让 seki 头在 99.99% 的行上被推向「全中性」，再乘 #12 那个
上限 8 的自适应系数放大。

⇒ **已改为独立的 `w_seki`，标签没给时按 0 处理**（宁可不训，不要训错方向）。
`katago_v7_loss.py` 与 `tests/test_katago_npz.py::test_seki_weight_defaults_to_zero_not_ownership`。

### 9.6 `pos_hash` 作为 join 键

**为什么不能用 `game_id`**：见 §5.3.1（45% 过滤率 + id 复用 bug）。
**为什么不用 `(sgf 路径, ply)`**：那要求标签侧与 build 侧各自独立枚举 SGF
且顺序完全一致；只要枚举规则动一行（加一种过滤、换 glob 顺序），历史标签
就静默错位，且「join 不上」与「本来就不该 join」长得一模一样。

**口径** = `(board 361B, to_play 1B, ko 2B)`，只取盘面 + 行棋方 + 打劫点，
不含历史（`probe_transposition.py` 的 key A 已实测：264 个重复项里 0 例因
历史差异被拆开）。

**实测**：吞吐 **478,603 行/s**（34.2M 约 72s）；
`tmp/bench/verify.npz`（800 位置）在 4M 行处命中 100/800，散列逐位一致。

**唯一事实来源** = `src/data/pos_hash.py`；`probe0_join.py` 改为对它的
re-export（**逐位不变**，有单测），因为改了种子会让
`kata_labels.npz` 里已落盘的 `pos_hash` 列全部失效而症状是「join 变空」。

### 9.7 12 项 loss 在真实官方目标上的实测

`scripts/probe_stdata_loss.py`（拿真实 stdata 行跑通，`tests/test_katago_npz.py`
与 `tests/test_katago_v7_loss.py` 各有一组对拍）。标签体检：

```
outcome 分布       [33 胜, 27 负, 0 无结果]      ownership  {−1,0,+1}
scoring 值域       −120 .. 120                 futurepos  {−1,0,+1}
score_distr 行和   1.0，每行恰 2 个非零          komi        {−7.5, 7.5}
policy top16 计数  763 .. 943
```

**端到端冒烟训练**（`scripts/smoke_train_v7.py`，302 行 / batch 8 / 40 步），
逐项 step0 → step39：

| 项 | 变化 | | 项 | 变化 |
|---|---:|---|---|---:|
| scorebelief_cdf | **−48.3%** | | policy_opp | −12.6% |
| lead | **−94.8%** | | scoring | −12.6% |
| value | −15.9% | | futurepos | −12.4% |
| ownership | −14.8% | | policy | −0.5% |
| scorebelief_pdf | −13.7% | | seki | 0.0%（`w_seki` 缺省 0 ✓） |
| | | | **score_stdev** | **−0.0%** ← 见 §4.2 待裁决 |

⚠ **`scoring` 开局就占 83/104**（随机预测 vs ±120 目标，MSE ≈ 1.4e4），
梯度范数在 step 20 冲到 365、被 clip 到 5 ⇒ 早期有效步长被压得很小。
真实跑要相应加 warmup，或先确认 `scoring` 的 ±120 缩放是否该归一。

### 9.9 SGF 语料的实测性质（决定了哪些 loss 能在段 1 训）

抽样 133,604 个 SGF 实测：

| `RU` | 局数 | 占比 | 计分 | | `KM` | 占比 |
|---|---:|---:|---|---|---|---:|
| 无 RU | 73,521 | 55.0% | 默认 AREA | | 7.5 | 37.7% |
| `Chinese` | 50,291 | 37.6% | AREA | | 6.5 | 18.0% |
| `Japanese` | 9,789 | **7.3%** | **TERRITORY** | | 5.5 | 15.1% |
| 空 `RU[]` | 3 | 0.0% | 默认 AREA | | 3.8 | 10.5% |

| `RE` 模式 | 局数 | 占比 | 语义 |
|---|---:|---:|---|
| `B+R` / `W+R` / `B+Resign` / `W+Resign` | 98,352 | **75.2%** | **认输，无分差** |
| `B<N>` / `W<N>`（如 `B+2.5`） | 24,748 | 18.5% | 有分差 |
| `B+T` / `W+T` | 271 | 0.2% | 超时认输 |
| `draw` / `B+` / `W+F` 等 | 211 | 0.2% | 边角 |

🔴 **两条硬结论：**

1. **75% 的局是认输 ⇒ 12 项里 5 项（#5/#6/#8/#9/#10）在 75% 的局上权重为 0。**
   有分差监督的约 `24.8% × 162,298 × 210 ≈ 6.6M 行`（34.2M 的 **19%**）。
2. **贴目五花八门**（7.5/6.5/5.5/3.8/**0**/2.8/4.5/缺）⇒ `g_komi` 必须逐局存，
   **不能假设常数 7.5**；`KM=0` 占 5.9%。

🔴 **更根本的：语料性质决定 loss 的启用顺序。** 认输局即使重放到终局，
`go_rules.py:2269` 的 docstring 自陈「没有死子判定，对局双方应在 pass 认输前
实际提掉对方的死子，否则死子所在点会被判为双方共邻的中立区域」
⇒ 从认输棋谱反推的终局归属不可靠。

⇒ **因此段 1 只训 policy 系 + value 3 类 + futurepos**：
`#1 policy`、`#2 π_opp`、`#3 value`（`winrates` 免费）、`#11 futurepos`；
`#5/#6/#8/#9/#10`（score 系）权重天然为 0 或只覆盖 19%；
`#4 ownership`、`#10 scoring`、`#12 seki` **不在段 1 训**。
⇒ **段 2/3（1% 标注 + stdata 3.1M）才开全部 12 项** ——
那 3.4M 上这些信号来自搜索器自己的评分器（`valueTargetsNCHW`），质量高得多。

### 9.10 本 spec 相对初稿的偏差清单

| # | 位置 | 偏差 | 状态 |
|---|---|---|---|
| 1 | §6.1 | 「块小计 ×11 = 5,479,680」行标签张冠李戴（该数不是 11 的倍数；11 块实为 **5,423,616**） | **已订正**，测试钉住 |
| 2 | §4.2 | `scorestdev` 的 `SoftPlus(x₁, 0.05)` 与 loss #7 的 δ=10 矛盾，实测该项无学习信号 | **待裁决**，提为 `SCORE_STDEV_SOFTPLUS_BETA` |
| 3 | §5 | 原设计（build 加列 + 局表并入主 npz）被「不 rebuild + 实时算 + 独立 sidecar」取代 | **已重写** |
| 4 | §5.4 | 批契约从「4 元组 + dict」被实现成「5 元组」 | **已订正回 4 元组** |
| 5 | §7.4 #6 | 验收锚点从「与旧 17ch 逐位相等」换成「与官方 stdata 逐位对拍」 | **已替换** |
| 6 | §7.4 #12 | seki 的行权重从 `w_ownership` 改成独立 `w_seki` | **已改** |
| 7 | §3.2 | 2D RoPE 的 `(H,16,2)` 里那个 `2` 的语义 spec 未定义；本实现取「二维频率」（`[…,0]` 乘行、`[…,1]` 乘列） | 记录，参数形状与预算不变 |
| 8 | §2.2 | 8 路恒 0 通道在 stdata 上**不全是 0**（encore 开启 ⇒ ch7/20/21 非零率 0.01%/0.35%/0.39%）；我们无 encore 的简单局才为 0 | 记录；stdata 分布与我们不同 |
| 9 | §5.3 | 局级 sidecar 砍掉 `g_ownership`/`g_scoring`/`g_seki`（原 176 MB → **1.2 MB**），只留 `g_komi`/`g_score`/`g_rules`/`g_resign` | **已决定**，理由见 §5.3.0 |
| 10 | §4.5 #4/#10/#12 | 这三项**不在段 1 训**，改由段 2/3 的 stdata 提供 | **已决定**，理由见 §9.9 |
| 11 | §4.5 #5/#6/#8/#9 | 75% 的局是认输 ⇒ 这 5 项在 75% 的局上权重为 0（仅 6.6M / 19% 有监督） | 记录，见 §5.3.3 / §9.9 |
| 12 | §2.5 | `RU` 覆盖仅 45%，其中 `Japanese`（7.3%）是 **TERRITORY** 计分 ⇒ ch18/19 恒 0；另 55% 无 `RU` 按默认 AREA | 记录，见 §5.3.2 |
