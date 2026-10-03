# KataGo NBT + Transformer 重建设计（V7 输入 22/19）

> **状态**：设计定稿（§1–§8 全部经人工逐节批准），待复核后转 `writing-plans`。
> **取代**：`2026-10-01-katago-se-rebuild-design.md`（见 §8.1）。
> **一手来源**：https://github.com/lightvector/KataGo
> **硬预算**：参数 ≤ **5.85M**，峰值显存 ≤ **31.0 GiB @ batch 3200 / 卡 / 19×19 / 4×Ascend 910A**。

> 🔴 **2026-10-02 实现期回写**：§1–§8 的**设计推导保留原样**（那是有价值的），
> 被实跑推翻的结论**就地标注「原以为是 X / 实测是 Y」**而不是删掉。
> 全部订正集中在**新增的 §9.8（哪些通道已实测逐位对齐 / 哪些不能）**
> 与 **§9.11（实现期新发现的长期约束）**、**§9.12（A6 futurepos 契约 /
> B7 三档比对资格 / 行号漂移登记）**，并回填进 §2.2 / §2.3 / §2.4 / §2.5 /
> §2.6 / §3.5 / §4.5 / §5.1 / §5.2 / §5.3.1 / §5.3.3 / §5.3.4 / §5.5 /
> §5.7 / §7.1 / §7.3 / §7.4 / §7.5 / §8.2 / §8.3 / §9.4 / §9.9 / §9.10。
> 最关键的四条：**`RE` 认输率 81.09%（不是 75.2%）**、**`KM` 有 ×100 家族**、
> **ch18/19 无法与官方 stdata 对齐**、**ch14/ch17 已实测 `1.000000`**。
>
> 🔴 **2026-10-02 第二轮回写（本轮）** 又推翻了四条「看起来已定」的东西，
> 全部由**代码为准**判定：
> ① **`soft_mask = 0` 不是「one-hot CE」而是「贡献恰好 0」** ⇒ 分母恒为 `B`
> ⇒ 不配 `--soft-only-sampling` 就有 **1% 软行 ⇒ policy 项缩小 ~100×**（§5.7）；
> ② **ch9–13 / ch7 / ch20-21 / ch15-16 与官方 stdata 结构上不可比**
> ⇒ stdata 的覆盖面远小于 §9.8 第一版写的「20 通道」（§9.8 / §9.12.2）；
> ③ **CLI 冻结集已从 61 扩到 65，`soft_ce` 已接进 CLI**
> ⇒ 「段 2 无法从命令行启动」作废（§9.11.1）；
> ④ **C0 已实跑：3.04 ms/行 预算在本机不成立**（冷读 3.66 ms/行，超 1.2×），
> 且 **batch 与 `--prefetch-workers` 解耦**（§5.2）。

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
| C5 | `pytest tests` 全绿（基线 **1011 passed / 2 skipped**，见 §7.4 #1）+ `bash -n shell/*.sh` | 本地必验 |
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

> 🔴 **警示（2026-10-02 实现期实测）：门控只能解析一次，只调一处实现。**
>
> `src/data/feature_v7_ladders.py` 曾把这段门控**解析了两次**：
> 函数内部（`pb` / `p2b`）算对了，但**输出行**独立重算了一次并抄错了源 ——
> ch16 复制的是 `cur[:,0]`（= **ch14** 的内容）而不是 `out[:,1]`（= **ch15**）。
> ⇒ `history=0` 时两处结果**恰好相同**（`pb is b`，都退化成当前盘），
> **只有 `history=1` 才差**，且差的正好是**一手**。
>
> 这是 `go_rules.py:170-177` 记着的那个教训（「函数内算对了、调用侧又独立重算一遍」
> ⇒ U 形块两个气桶同时漏）在 V7 里的**第二次复刻**，同一个仓库、同一种形状。
> **函数的 docstring 本来就写对了，是代码与 docstring 自相矛盾。**
>
> ⇒ **规则**：门控（及任何「历史不足 ⇒ 回退到上一级」的语义）**只允许有一个
> 实现点**。测试必须覆盖 `history ∈ {0, 1, ≥2}` 三档，且断言的是**通道之间的相等关系**
>（`ch15 == ch14` / `ch16 == ch15`），不是绝对内容 —— 只断言绝对内容的话，
> 整个 ladder 一起算错也能过。

### 2.3 全局 19 通道

| ch | 语义（源码） | 我们 |
|---:|---|---|
| 0–4 | 过去 5 手中哪些是 pass | `my_hist/op_hist` 含 −1 ⇒ 可判 |
| 5 | `currentSelfKomi(nextPlayer) / 20`，clip 到 `±(bArea + KOMI_CLIP_RADIUS)` | **符号随 `to_play` 翻转**；来自局表 `g_komi`。⚠ **`g_komi` 必须先过 ×100 离群修正**（`foxpro` 的 `KM[650]` = 6.5，不修会让本通道变成 **32.5**）—— 见 §5.3.1 |

> 🔴 **同语义随版本换常数：分母 V7 = 20、V3/V4 = 15。**
> `src/data/feature_v7.py:494-497` 把它提成具名常量 `KOMI_SCALE = 20.0`，
> 注释原文：「`globalInputNC[:,5]` 是 komi=7.5 时**恰好 0.375 = 7.5/20**，
> 抄成 /15 就错」（0.375 是**官方自己的实测值**）。
> ⇒ 这不是「抄错一个数」，而是**同一个语义在不同版本用不同常数**；照抄 V3/V4 的
> `15` 会让本通道整体偏离 4/3。任何重读这段的人都要先看 `KOMI_SCALE` 的值，
> 不要凭记忆写分母。clip 半径同理是具名常量 `KOMI_CLIP_MARGIN = 1.0`
> （对应源码 `if(selfKomi > bArea+1.0f)`）。
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

> 🔴 **「恒 0」是对「本仓的简单局」的断言，不是对官方的断言**（2026-10-02 订正）。
> 官方数据上这 8 路**并不恒 0**，且原因分两类，样本量不同：
>
> | 通道 | 官方为什么非 0 | 实测 |
> |---|---|---|
> | 空间 `7` / `20` / `21`（encore 机制） | 官方**只在 encore 行**置位 | 301 行里恰好 **1 行** `encorePhase=2` ⇒ 三通道都只在那 1 行不一致，其余 300 行逐位相同 |
> | 全局 `12` / `13`（encorePhase） | 同上（分组依据就是官方自己的 ch12/ch13） | 同上 |
> | 全局 `15` / `16`（PDA） | 本仓无 PDA ⇒ 恒 0，**但官方有** | 🔴 **小样本上「恰好恒 0」**：301 行上官方这两格全 0 ⇒ 报出来的 `row_exact = 1.000000` **是巧合不是对齐**；样本一大就露馅 —— **200 npz / 4,368 行上官方有 133 个非零点 ⇒ 0.9696** |
> | 空间 `8` | 源码注释写「6,7,8」但**从未写入** | 这一个**真的**恒 0 |
>
> ⇒ 「本仓简单局上恒 0」成立；「这些通道在官方数据上也是 0」**不成立**，
> 尤其**全局 15/16（PDA）在 4,368 行上有 133 个非零**。
> ⚠ 后果：这 4 个通道在 stdata 上被 `scripts/crosscheck_stdata.py` 归
> **`NOT_COMPARABLE`**（而不是 `ALIGNED`），因为
> `spatial_channels_v7` **没有 `encore_phase` 入参**、本仓压根不接 PDA
> ⇒ 结构上到不了那个值。见 §9.12.2。

### 2.5 ch18/19 的规则条件化

```cpp
if (SCORING_AREA && TAX_NONE)  calculateArea(...)           // 常规
else if (SCORING_AREA && (SEKI|ALL))  calculateIndependentLifeArea(keepStones=true)
else if (SCORING_TERRITORY)  // 仅 encorePhase >= 2 才置位 ⇒ 正常阶段恒 0
```

⇒ **territory 计分的对局，ch18/19 在正常阶段恒 0 —— 这是对齐官方，不是缺陷**（§5 已批准）。SGF 缺 `RU` 时默认 AREA。

#### 🔴 订正（2026-10-02 实现期实测）：**ch18/19 无法与官方 stdata 对齐**，不是「待对拍」

上面这段规则本身没错。错的是本 spec 此前把 stdata 当成 ch18/19 的 oracle
（§7.4 验收 #6、§9.4 的表格）。逐位对拍 **1,254 行**真实 19×19 局
（`zzb28c512` 批第一个含 19×19 行的 npz）实测：

| 指标 | 实测（**批次 `zzb28c512`，1,254 行**） |
|---|---:|
| 我们比官方**多**的点 | **21,896**（**19,913 个子** + **1,983 个空点**） |
| **完全相同的行** | **0 / 1,254** |
| 受影响的行 | **295 / 1,254** |

🔴 **这些绝对量是批次特定的，换归档就不复现**（`scripts/crosscheck_stdata.py`
在 `2026-08-25npzs.tgz` 上复现不出来）：

| 归档 / 样本量 | row_exact | 我们多出的点（子 / 空点） |
|---|---:|---|
| `zzb28c512` / 1,254 行 | **0.000000** | 21,896（19,913 / 1,983） |
| `2026-08-25` / 301 行 | **0.445183** | 2,735 / 586 |
| `2026-08-25` / 4,368 行 | **0.352119** | 98,560 / 6,374 |

⇒ **「0 行完全相同」不能跨归档引用**；**稳定的是偏差方向与「约 94% 是子」这个
比例**，不是绝对量。**不带样本量的命中率无意义** —— 这条同样适用于所有
通道的对齐数字。

**最小反例**（`zzb28c512` 批）：一个被黑子四面围住的**单个空点** —— 按
Tromp-Taylor 归黑（空点只与一种颜色相邻 ⇒ 归该色），**官方给中立**。

**排查过的两个假设，都被否掉**：

1. 「缺失的子集中在 1 气的块上」⇒ **实测不成立**（分布打散）。
2. 「我们只是多算了没提的子」⇒ **不完全成立**：1,983 个**空点**的偏差没法用
   「官方提了死子」解释 —— 死子被提掉后该点会归**对方**，而官方给的是**中立**。

⇒ **合理猜测（未证实）**：官方在算 area **之前先提掉死子**，而本仓
`GoBoard.score()` **没有死子判定**（其 docstring 自陈这一点）。
两个**空点**偏差说明「提死子」也不是完整解释，还有别的判据没对上。

🔴 **决定：按「对齐口径优先」处理，没有偷偷补死子判定。**
理由：死子判据（`liberties == 0`？递归救活？superko？）**要猜**，
猜错会得到**第三种**偏差 —— 比现在这个已知偏差更难诊断，而且会让 ch18/19
在没有 oracle 的情况下同时承担两个不确定性。

⇒ 🔴 **订正（2026-10-02 第二轮）：`ch9–13` 也不该被叫「stdata 的 oracle」。**
上面这句话原本写的是「stdata 只是 ch3/4/5、**ch9–13**、ch14/ch17 的 oracle」。
实测（`scripts/crosscheck_stdata.py`）把 **ch9–13 归 `NOT_COMPARABLE`**：
stdata 是**逐行独立样本、没有着法序列**，而 ch9–13 是「过去 5 手落点」，
重建它需要序列本身 ⇒ **结构上到不了**。原来那组 99.x% 是**落点身份统计**
（「ch9 有多少落点落在 opp 子上」），不是对齐率。
⇒ 真正的 oracle 只有 **空间 ch0–6、ch8、ch14、ch17**（+ 全局规则位那几个），
完整三档资格表见 **§9.12.2**。
ch18/19 的正确性口径降级为：
`SCORING_AREA && TAX_NONE` 条件下与 Tromp-Taylor 等价（`go_rules.py` 侧已如此声明）
+ 逐条对照本节那段 `fillRowV7` 源码引文，**没有可执行 oracle**。

### 2.6 dtype 与有意偏差

| 项 | 官方 | 我们 | 影响 |
|---|---|---|---|
| 空间通道 | `bool`（`rowBin`） | **float16** `{0,1}` | 逐位语义等价 |
| 全局通道 | `float32` | **float16** | 量级差异 < 1e-3，接受 |
| D1 | `superKoBanned` 全集 | 仅 `ko_point` | KO_SIMPLE 下等价 |
| D2 | `encorePhase` 机制 | 无 encore | 简单局等价；空间 `7`/`20`/`21` + 全局 `12`/`13` 在**本仓简单局**上正确为 0（⚠ **不是**对官方的断言，见 §2.4） |
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
3. **注意力仍走 `_sdpa` 四站点机制**（`backbone.py:581`，行号仍有效），但 `head_dim` 由 **46 → 32**；`train_sft.py` 里 `sdpa_force_math` 的分派（⚠ 原写 `:2269`，该行号已漂移；实际在 **`:2641-2740`** 的分派块里：默认 `True`，**只有 cuda A100 分支 `:2674` 置 `False`**，910A 走 `:2693/2716/2727` 的 `True`）暂时保留，探针证实可用后再切（§6 R1）。
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
                                 scorestdev = 20·SoftPlus(x₁, 0.05)  ← 字面值，见下方裁决
                                 lead       = 20·x₂
```

`KataGPool_value` 第三个统计量换成 `mean·(((√A−14)²/100) − 0.1)`（二次 board-size 缩放，**无 max**）。
⚠ **固定 19×19 下 `((√A−14)/10)=0.5`、`(((√A−14)²/100)−0.1)=0.15` 都是常数** ⇒ value 池化实际只是三个缩放的 mean。公式照抄保留，以便将来泛化。

> ✅ **已裁决（2026-10-03）：`scorestdev` 的 `beta` 取 `1.0`，不再是未决项。**
>
> 上面的 `SoftPlus(x₁, 0.05)` 是**本 spec 的原始字面值**，保留在正文里不改 ——
> 它是下面这条推导链的一环（推导的正是「字面值为什么错」）。**代码不实现它**：
> `src/networks/katago_v7.py::SCORE_STDEV_SOFTPLUS_BETA = 1.0`。
>
> **推导链**（`F.softplus(x, beta) = log(1+exp(beta·x))/beta`，代入 x=0）：
>
> ```
> softplus(0, beta) = log(2)/beta   ⇒   预测初值 = 20·log(2)/beta
>   beta = 0.05  ⇒  13.86 → ×20 = 277.26    ← spec 字面值
>   beta = 1.0   ⇒   0.69 → ×20 =  13.86    ← 裁决值（PyTorch F.softplus 默认）
> loss #7 的目标 = std(softmax(scorebelief)) = 5~20,  δ=10
> ```
>
> 2026-10-02 端到端冒烟训练（`scripts/smoke_train_v7.py`，302 行 / 40 步）实测
> **字面值那一版这一项 40 步内变化 −0.0%**，是 12 项里唯一不动的。硬算：
> 系数 0.001 × AdamW 步长 ~3e-4 ⇒ 需 ~9 万步才从 277 挪到 10 量级，
> 真实训练约 1 万步 ⇒ 按字面写法 **loss #7 等于没有学习信号**。
>
> **裁决后的实测**（`tests/test_katago_v7_budget.py` 的
> `test_score_stdev_softplus_term_lands_near_huber_delta` 与
> `test_score_stdev_loss_term_is_inside_huber_delta_at_the_ruled_out_beta`；
> seed 0 初始化 + seed 7 输入 / B=64，真实 forward）：
>
> | | 预测初值 `score_stdev.mean()` | loss #7 公式值 | 与 δ=10 |
> |---|---:|---:|---|
> | `beta=0.05`（字面值） | **277.2534** | **272.22** | ✗ 27 倍，梯度被常数偏差支配 |
> | **`beta=1.0`（裁决值）** | **13.8599** | **8.83** | ✓ **落在 δ 以内**（二次段，梯度有效） |
>
> ⚠ **段 1 不受本裁决影响**：`V7_STAGE1_SCORE_TERMS` 8 项系数逐个 0.0，
> `score_stdev` 的 `weighted` 恒 0 ⇒ 段 1 的四个主目标（policy / π_opp /
> value / futurepos）**逐位不变**（有测试钉住）。本裁决是为**段 2/3**
> （接上 sidecar、把 score 系权重打开）生效的。
> ⚠ **纯常量**：头部拓扑、参数量（**5,561,832**）、预算测试全部不受影响
> （同样有测试钉住）。

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
| 12 | seki | `(CE_sign[3] + 0.5·CE_neutral)/361` × `8·0.005/(0.005+EMA)` | **自适应** | `w_seki`（🔴 **不是 `w_ownership`**，见下） |

> 🔴 **订正（2026-10-02 复核）：#12 的行权重是 `w_seki`，本表原先写的 `w_ownership` 已作废。**
> 依据：`src/networks/katago_v7_loss.py:291-293` 在 `w` 里**没有 `seki` 时
> 直接填 0**（宁可不训，不要训错方向），`forward` 的 #12 用 `w_of('seki')`；
> `tests/test_katago_npz.py::test_seki_weight_defaults_to_zero_not_ownership` 钉住。
> §9.5 已记录这个改动，但**本表没跟上** —— 现已对齐。
> ⚠ **只有 #4 ownership 用 `w_ownership`**：那是官方 `globalTargetsNC` 的
> **正式列名**（col27，见 §9.2），在 loss 表与 stdata 目标表里出现都是**正确的**，
> 不要与本条混淆。
>
> ⚠ 顺带钉住装配口径：`_weighted_mean(per_sample, weight)` 是
> `(per_sample * weight).mean()`，**刻意不除 `Σw`**
> （`src/networks/katago_v7_loss.py:195-204`）。⇒ 行权重是**全局量级旋钮**，
> 不是「本批有效行的平均」。这条同时是 §9.12.1 里 `w['futurepos']` 必须取
> **两路的与**的根因。

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
| `score`（scoremean/lead/scorebelief 目标） | ❌ 幅度被 α 污染 | **sidecar**（⚠ **81.09%** 的局是认输 ⇒ 无值，§5.3.3） |
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
| NPU 侧吞吐需求 | ~**2635** 行/s | ⚠ 原写 `run.txt:700`（v18, 12ch）—— **`run.txt` 已改成「一屏」版、只剩 55 行，该行号已悬空**，数字本身保留 |
| 预取 worker | **8**（原写「默认」） | `--prefetch-workers` |
| **每行时间预算** | **3.04 ms** | 2635 / 8 |
| 现有 12ch 单盘基线 | 1.78 ms | 原写 `go_rules.py:2479`（**该行号已漂移**，现为一段无关注释） |
| ~~**22ch 余量**~~ | ~~**1.7×**~~ | 🔴 **已被实测推翻，见下** |

其中 3 气桶 / 历史交错 / parity 三角波几乎免费（前者复用已在跑的
`_group_liberty_count` flood），贵的是 `iterLadders`（**3 块盘面**）与
`calculateArea`（1 块）。

> ⚠ **降级档位（预先约定，不预先实现）**：若 `iterLadders` 使 22ch 严重超预算，
> 降到 **18 通道**（ch14–17 恒 0）。主干 / 四头 / 12 项 loss / 参数量
> **5,561,832 一行都不用改**。

##### 🔴 C0 已实跑（2026-10-02）：上面那张「1.7× 余量」的表**整张作废**

工具 `scripts/bench_v7_features.py`，产物 `benchmarks/bench_v7_features.json`
（`generated_at = 2026-10-02T19:27:30`）。
**机器**：`AMD Ryzen 7 5800H`（8 物理核 / 16 逻辑）、**16 核**、
物理内存 **14.2 GB**、Python 3.12.4、numpy 1.26.4、
**数据盘 = USB（`JMicron Generic`）**、`boards.npy` = **12.35 GB**。
采样：分层（前 1024 / 随机 2048 / 复杂度 top 1024，候选池 24000 行），
每层 × `B ∈ {32,128,512}`，reps=3。⚠ **数据盘是 USB 这件事决定了下面所有数**。

**(a) 成本几乎全在 `iterLadders`**（组件分解，`first` 层 `B=32`，1024 行）：

| 组件 | 秒 | 占 pipeline |
|---|---:|---:|
| `ladder_3`（3 次梯子搜索） | **3.379** | **99.80%** |
| `_pipeline`（22 通道总计） | 3.386 | 100% |
| `liberties_123`（3 气桶） | 0.0319 | 0.94% |
| `calculate_area`（ch18/19） | 0.0137 | 0.40% |
| `global`（19 维全局） | 0.00040 | 0.012% |
| `history_five` / gather 三个偏移 | ≤ 0.0004 each | < 0.1% |

⇒ 「3 气桶 ≈ 0、历史交错 ≈ 0、parity ≈ 0、`iterLadders` 贵、area 中」
这条**定性判断完全成立**，现在有了数量：**梯子占 99.8%，其余全部加起来 < 1.5%**。
（脚本的 `ladder_share_note` 补充：分子分母是各自独立测量的，
`share_of_pipeline` 略超 1.0 是噪声，不是 bug；正确读法是「梯子基本就是全部成本」。）

**(b) 🔴 batch 规模对特征吞吐几乎无影响 ⇒ batch 应按显存选，与特征侧解耦**
（`batch_curve`，随机 2048 行）：

| B | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| 行/s | **311.4** | 306.4 | 309.7 | 306.1 | 306.6 | **307.1** |

`max/min = 1.017` ⇒ **离散度仅 1.7%**（32 倍的 batch 跨度只换来 1.7%）。
原因：特征侧的 Python 开销**长在每行的梯子 DFS 里面**，不在批级
（脚本原文：「『大 batch 摊薄开销』在本管道上基本不成立」）。
⇒ **这条解掉了一个原本的耦合**：`--batch-size` 与 `--prefetch-workers`
**不必联动选**，batch 纯按 NPU 显存 / 收敛选（§6 的显存账按 B 线性放大，
那一半才是 batch 的真正约束）。

**(c) worker 扩展（每点 `B=512`、2048 行，`spawn` 起进程）**：

| worker | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| **热读** 行/s | 341.2 | 594.7 | 827.8 | **1161.4** |
| **冷读** 行/s | 175.8 | **155.2** | 217.8 | **273.1** |

- 热曲线 8 worker 的**边际效率 = 0.701**（`(1161.4/827.8)/2`）⇒ 还没饱和，
  但也没有线性；1→8 总加速 **3.40×**（8 worker 上限）。
- 🔴 **冷/热差 4.3×**（8 worker：273.1 vs 1161.4）。**冷读 2 worker 比 1 worker
  还慢**（155.2 < 175.8）——随机读被磁盘串行化，加并发只会更糟。
- **gather 自身的冷热更极端**：冷读 **10.4–24.7 ms/行**（两组 offset 的中位数
  11.08 / 11.15）vs 热读 **0.004–0.008 ms/行**（中位数 0.0067 / 0.0050）
  ⇒ **中位数之比 1654× / 2230×**。线程并发探针（1/4/8 线程）聚合吞吐
  91.7→109.5 行/s，几乎不涨 ⇒ 脚本判定「**存储瓶颈**」。
  ⇒ **fork 之前必须 warm**：先跑一个 warmup batch 把要读的页压进页缓存，
  否则 4 卡 NPU 会等在 USB 盘上。

**(d) 单进程**（`verdict.single_process`，参考 `B=512`）：

| 层 | 行/s | ms/行 |
|---|---:|---:|
| `first`（开局空盘，最快路径） | 289.4 | 3.455 |
| `random` | 301.5 | 3.317 |
| `complex`（复杂度 top） | 178.7–188.2 | **5.31–5.60** |

🔴 **遇复杂盘面 5.5 ms/行，超 3.04 ms 预算 1.8×**。
⚠ **只测前 N 行会系统性低估**：开局空盘是最快路径（脚本原文如此警告），
`first` 与 `complex` 差 **1.6×**。这就是脚本坚持分层采样的原因。

**(e) 内存**（`memory`，8 worker）：

| B | 32 | 128 | 512 |
|---|---:|---:|---:|
| 单 worker 峰值工作集 | 306.5 MB | **307.2 MB** | 306.4 MB |

**8 worker × 307 MB = 2457 MB ≈ 2.4 GB** ≪ 物理内存 14.2 GB ⇒ `fits: true`。
⚠ 且 batch 32→512 **内存几乎不变**（306.5→306.4），与 (b) 同源。
（脚本注明：峰值含 mmap 的文件页，**这是上界**；真匿名内存小得多。）

**(f) 预算判定 —— 🔴 不成立，且脚本不替我们圆场**：

```
budget_ms_per_row 3.04 | measured 3.6612（8 worker，冷读，部署态） | over_budget_by 1.2 | holds: false
basis = cold_storage_bound（本机 boards.npy 在 USB 卷上）
```

⇒ **§5.2 的性能闸门在本机不成立**（超 1.2×）。降级档位「18 通道」
**现在有了实测触发条件**（§8.3 的触发行「`iterLadders` 性能不可接受」已激活）。

**部署参数（据本表定）**：

| 参数 | 值 | 依据 |
|---|---|---|
| `--prefetch-workers` | **8**（本机已到边际效率 0.701 的拐点） | (c)。⚠ 云端 24 核可按「worker ≈ 物理核数」外推到 **12** —— **该值未实测**（脚本只跑了 1/2/4/8），引用时必须带「外推」二字 |
| `--batch-size` | 与 `--prefetch-workers` **解耦** | (b)，1.7% 离散度 |
| **fork 前必须 warm** | 1 个 warmup batch | (c)，冷/热 4.3× |
| 采样/测量 | **必须分层，禁止只测前 N 行** | (d) |

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
   （#4 ownership、#10 scoring、#12 seki），而 #10 已因分差缺失在 **81.09%** 的行上废了。
2. **信号质量反而更差**：`go_rules.py:2269` 的 docstring 自陈「没有死子判定，
   对局双方应在 pass 认输前实际提掉对方的死子，否则死子所在点会被判为双方共邻的
   中立区域」。而实测 **81.09% 的局是认输** ⇒ 从认输棋谱反推的终局归属不可靠。
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

##### 🔴 订正（2026-10-02 实现期实测）：`KM` 还有一个 **×100 家族**，上表未列

上表是**按字面值**统计的，它漏掉了 `data/games/games/foxpro` 这一族。该目录写的是：

```
KM[650] [750] [550] [450] [375] [325]
```

**÷100 得 6.5 / 7.5 / 5.5 / 4.5 / 3.75 / 3.25** —— 全落在上表的合理区间里；
**÷2 得 325 / 375 / … 全是乱数** ⇒ ×100 是唯一自洽的解释（贴目惯例以「分」为单位，
6.5 贴目写成 `650` 分）。**共 2,163 局 = 1.62%**（分母 133,604）。

⚠ **照字面存的后果是静默的、且很严重**：全局 ch5 = `currentSelfKomi / 20`
⇒ `650 / 20 = ` **`32.5`**，比任何真实贴目大一个量级，而模型没有任何机制能把它拉回来。

⇒ **实现**：`scripts/build_games_sidecar.py::parse_komi` 在
`fix_outlier=True`（**默认开**）时把 `|v| > KOMI_OUTLIER_ABS (=100.0)` 的值**除以 100**；
命令行开关 `--no-komi-outlier-fix` **关闭**该修正（**别顺手加上**）。
修正条数在 sidecar 报告的 `×100 离群修正` 一行打印（`n_komi_outlier_fixed`），
钉住：`tests/test_games_sidecar_alignment.py::test_komi_outlier_fix_is_opt_out`。

> **原以为**：`KM` 只有上表那几种取值，解析即直存。**实测**：存在 ×100 家族，
> 1.62% 的局需要归一化。上表因此是**字面值**分布（它记录「原始 SGF 写了什么」），
> 归一化后的真实分布看 sidecar 报告。

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

实测 `RE` 格式（**早期抽样**，133,604 局 —— ⚠ 已被全量实测推翻，见下方订正）：

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

##### 🔴 订正（2026-10-02 实现期全量实测）：认输率是 **81.09%**，不是 75.2%

上面这张表来自**早期抽样**（133,604 个目录内 SGF 的字面模式统计）。
`scripts/build_games_sidecar.py` 在**全部 162,298 局**上实跑后，认输口径是：

| `RE` 形式 | 占比 |
|---|---:|
| `B+R` | 21.94% |
| `W+Resign` | 21.85% |
| `W+R` | 21.75% |
| `B+Resign` | 15.55% |
| **认输小计** | **81.09%** |
| `g_score` 可用（≈ 有数值分差） | **≈18.5%** |

**差 5.9 个百分点，方向是「比原以为的更糟」。** ⇒ 订正后的结论：

- §4.6 那 **5 个零权重 loss 项**（#5/#6/#8/#9/#10）实际影响 **~81%** 的行，
  有监督的只有 `18.5% × 162,298 × 210 ≈ 6.3M 行`（34.2M 的 **~19%**）。
- 段 1 **不训 score 系**（§9.9）；`ownership`/`scoring`/`seki` 更不在段 1。
- 报告里这两行（`有分差` / `认输`）分母是**入库局数**，不是语料 SGF 数 ——
  交叉引用时不要拿它跟上面的抽样占比混算。

> **原以为**：75.2%（抽样值，写进了本节与 §9.9，并被 §5.1/§5.3.0 引用）。
> **实测**：81.09%（162,298 局全量）。抽样漏掉的那 5.9% 是因为
> 抽样只扫了 `data/games/games/` 的 4 个目录，而 tgz 里的 36,274 个 SGF
> **不在这 4 个目录里**（§5.0 已核实两者重叠 = 0）。

##### 🔴 订正（同一轮）：另有两个**未列**的 `RE` 形式，都有显式分支

| `RE` 形式 | 局数 | 语法 | 解析分支 |
|---|---:|---|---|
| `W+3 zi` | **2** | **单位后缀**（欧洲式：`zi` = 子/子数） | 剥掉单位后缀再取数值 |
| `W+0,25` | **1** | **欧洲逗号小数**（`,` 而非 `.`） | 逗号改点再取数值 |

样本量极小（各 ≤2 局），但**两者都会静默走到「未识别 ⇒ `g_score = NaN`」分支**，
而它们**本该有分差** ⇒ 方向性错误。两条都必须有显式分支，不能靠默认。

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

锚点可行性已核实：**162,296 / 162,298 = 99.9988%** 的局 `L ≥ 20`（主 npz 段长实测）
⇒ ply-20 锚点对**几乎**所有局安全，但**不覆盖全部**（见坑 3）。
64 位散列碰撞可忽略，且此法**对枚举顺序与 id 复用 bug 完全免疫**。

**必须输出覆盖报告**（匹配率 < 100% 时，未匹配的局按
`scripts/build_games_sidecar.py` 的兜底值填：`g_komi = 0`、
**`g_score = NaN`**（⇒ `w_score = 0`）、`g_rules = RULES_DEFAULT`、
`g_resign = False`，**不静默填垃圾**）。

##### 🔴 附带订正：「100% 的局 ≥20 手」这句话**两处都要改**，但原诊断是错的

**原以为**：那句「100% 的局 ≥20 手（162,296 / 162,298）⇒ ply-20 锚点对所有局安全」
站不住，所以「100%」要么是抽样、要么量的是 SGF 侧的手数而不是主 npz 里的段长。

**实测（`tmp/materialized/game_ids.npy`，34,202,713 行全量）**：
段长分布 min 12 / p50 208 / p95 302 / max 2068 / mean **210.74**，
**162,298 段**，`L ≥ 20` 的段**恰好 162,296 段**、`L < 40` **134 段**、
`L < 20` **2 段**。

⇒ 🔴 **「162,296 / 162,298」这个分子是对的**（它就是 `L ≥ 20` 的真实段数），
错的只是把 **99.9988%** 写成「100%」，以及**由此推出「ply-20 对所有局安全」**。
它既不是抽样、也不是 SGF 侧的手数 —— 是**四舍五入 + 过度外推**。
⇒ 真正的风险全在下面坑 3（`min(20, L//2) ≠ 20`），**不在**「有没有 `L < 20` 的局」：
那 2 段 `L < 20`（min = 12）只是让 99.9988% 掉下来的原因，
而**即使 100% 的局都 `L ≥ 20`，坑 3 照样存在**（`L ∈ [20,40)` 的 132 段
在 `min(20, L//2)` 下**取不到 20**）。

##### 🔴 订正（2026-10-02 实现期实测）：锚点法有**三个坑**，上面那版写法每条都踩

> **原以为**：第 1 步「切出每局行区间」= `np.diff(game_ids)` 取非零位置，第 2 步
> 「`start_g + min(20, len//2)`」= 拿第 20 手锚。**实测**：三处都错，且**都不报错**。

**坑 1 — `game_ids` 是**置换**，不是升序。**
162,298 段对 162,298 个 **distinct** id（值域 `0..162,297`、无重复 ⇒ 就是那个置换），
但**段的先后顺序 ≠ id 顺序**。⇒ 切段**必须用相邻比较**（`game_ids[1:] != game_ids[:-1]`
⇒ `np.flatnonzero`），**绝不能**用 `np.diff(game_ids) > 0` 之类假设符号的写法 ——
那样会把同 id 的段漏掉、把换号处的边界算成一整个反号段。

**坑 2 — 偏移必须从 1 起，不能从 0。**
偏移 0 是**空盘**：主 npz 与全部 169,878 个 SGF 的**初始局面完全相同**
⇒ `pos_hash(空盘)` 对每一局都相等 ⇒ 任何 `L < 2` 的残局会锚到**任意一局**，
而**「锚到的那一局恰好也是残局」时完全看不出来**（匹配率仍是 100%）。
⇒ 用 `start_g + 1 + …` 起算；且因为 `build_dataset` 拒绝 `< 2` 手的局，
`min(20, L//2) ≥ 1` **恒成立**，取不到 0。

**坑 3 — `min(20, L//2)` 对 134 局 ≠ 20，硬编码 ply-20 会静默丢掉它们。**
实测段长分布：**134 段 `L < 40`**、**2 段 `L < 20`**。第 2 步若只取 ply-20
（或只取 `L//2`），这 134 段要么越界、要么取到同一个被多局共用的早局面。
⇒ **改为对每个 SGF 存偏移 1..20 的全部前缀散列**（**3.4M 散列 ≈ 27 MB**），
匹配时任取一个命中前缀 ⇒ 短局靠长前缀、长局靠短前缀都能锚上，
也不再需要为短局开特例分支。

> ⇒ 落地形态：`_AnchorIndex` 按 `(pos_hash, sgf, offset)` 建表，
> 段侧用「段内任一 1..20 前缀命中即算匹配」；命中数多于一个的段记为
> **歧义**（`n_anchor_ambiguous`）、跨局命中的记为
> **`n_anchor_multi_game`**，两者都在 sidecar 报告里打印。
> 报告另有一行 `id 复用冲突`（`n_conflict_id_reuse`）—— 本数据集实测应为 0，
> 非 0 说明坑 1 的切段写错了。

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
  `train_sft.py:940/1004/1237/1252` **逐位不变**
  （⚠ 原写 `822/886/1042/2920` —— `train_sft.py` 长了约 900 行，调用点已漂移）
- `labels=True` → **4 元组** `(states, moves_out, values, labels_dict)`，
  `labels_dict` 见 §4.6 的键 + 软标签两项：

```
next_move · outcome · outcome_black · game_weight
score · sb_center · sb_upper · global
ownership · scoring · seki · future
soft   (B,362) f32        # KataGo 访问分布，仅被 soft_mask=1 的行有效
soft_mask (B,) f32       # 1 = 该行有软标签；0 = 该行**不参与**软项
w      (dict)  policy / policy_opp / ownership / score / scoring / seki / futurepos
                        # ① 后五项恒 0（占位，见 §9.9 的 D0b 决定）
                        # ② opt-in 之后额外两键：futurepos_h0 / futurepos_h1（§9.12.1）
                        # ③ w['policy'] 恒 1；w['policy_opp'] = 该行是否有同局下一手
```

🔴 **`outcome_black` 这键原先漏在本 spec 的键表里**（实现里一直有，
`src/data/dataset.py:737-742`）：`outcome` 是 **to_play 视角**的，
黑方视角由它翻转而来。⚠ 翻转**不能**写成 `outcome * to_play`
（`outcome = 0`（to_play 胜）时 `−1` 仍是 0，会把「白胜」报成「黑胜」），
实现走的是 `where(to_play == 1, outcome, 1 - outcome)`。

> ⚠ **原设计是 5 元组 `(states, moves, values, soft, mask)`，已改。**
> 5 元组**没有 dict 的位置**，而 V7 的 loss 需要那个 dict ⇒ 最后只会变
> 6 元组或 fork 两条路。4 元组 + dict 让预取器只需搬运**一种** payload 形状。

> ⚠ **三个假 dataset 必须同步补 `labels=False` 形参**：
> `tests/test_eval_coverage.py:75`、`tests/test_eval_determinism.py:111`、
> `tests/test_eval_no_augment.py:219`。默认 `False` 时无害，一旦 `train_sft.py`
> 任何一处传 `labels=True` 就会 `TypeError`。

> 🔴 **加任何新 flag 前先读 §9.11.1**：
> `tests/test_huber_loss.py::test_no_new_cli_params` 把 `train_sft.py` 的
> **65 个 flag 整个冻结**（D1 零新增/零删除/零改名），
> `test_policy_loss_default_is_ce` 钉住 `--policy-loss` 的 **default = `ce`**
> 且 choices `== {huber, ce, soft_ce}`。
> ⇒ A4 已登记进去的**四个**软标签旗：`--soft-index` / `--soft-weight` /
> `--soft-only-sampling` / `--soft-every`（⚠ **不是** `--soft-labels`，
> 见 §9.11.1）。
> ✅ **`soft_ce` 已接进 CLI**（`--policy-loss` 的 choices 含它，且给
> `--soft-index` 会把生效口径**派生**为 `soft_ce`）
> ⇒ **段 2 的软标签训练现在可以从命令行启动**（原写的「无法启动」已作废）。

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
> 全在 2M 行（约 7,800 局）覆盖之外。
>
> 🔴 **订正（2026-10-02 复核）：这段的「约 316 行/局 ⇒ 34.2M 行约覆盖
> 108,000 局」是错的推断，现在作废。** 它把两个**不同的下标空间**相除了
> ——`10,612` 是**语料 SGF 列表**里的下标，`3,350,229` 是**主 npz 行号**，
> 两者之间没有换算关系。
> **实测**（`tmp/materialized/game_ids.npy`，34,202,713 行全量切段）：
> 段长 **mean 210.74 / p50 208 / p95 302 / min 12 / max 2068**，
> 162,298 段 ⇒ `34,202,713 / 210.74 = 162,298`（就是**全部**局）。
> 另外，前 10,612 段合计 **2,251,471 行 = 212.2 行/局**，
> 而 **3.35M 行对应的是 15,845 段**，不是 10,612 段。
> ⇒ **34.2M 行 = 全部 162,298 局**，不是 108,000 局；语料平均是
> **210.7 行/局**，不是 316。引用「每局多少行」一律用 210.7。

**join 缓存**：实现在 **`src/data/kata_label_join.py::build_soft_index`**
（join 的**唯一实现处**），`scripts/build_soft_index.py` **只做 CLI 委派**。
🔴 本节原先把缓存记在脚本名下 —— 那里**原本一个字节缓存都没有**，
每次都全量重扫（**实测 127 s**）。缓存 key 必须覆盖**四个输入**：

```
(主 npz 的   行数 / distinct game_ids / mtime / size,
 标签 npz 的 行数 / distinct pos_hash  / mtime / size,
 max_repeats,
 散列口径版本号 + 从 pos_hash 活常量现算的指纹)
```

⚠ 命中**陈旧**缓存时**明确告警并逐项列出差异**，绝不静默复用；
缺 `cache_key` / `cache_meta` 的旧产物一律当「没有缓存」重建。
（症状与真缺标签长得一模一样，是本项目最危险的失败模式。）
钉住：`tests/test_join_cache_freshness.py`（25 个 test 函数）、`tests/test_kata_label_join.py`（19 个 test 函数）。

**掩码语义**：`soft_mask` 是**逐行二选一**
（**1 = 该行只算软 CE；0 = 该行对软项贡献恰好 0**），**不做混合**。
理由：同一个 head 同时收到「搜索分布」与「人类 one-hot」
会去折中而不是学搜索。段 2 用独立采样器**只取软行**
（`--soft-only-sampling`，与 Phase 4 item 3 的窗口**正交**，两者可叠加）。

> 🔴 **订正（2026-10-02 复核）：「0 = one-hot CE」这半句从来不存在，删掉。**
> 「二选一」的说法本身没错，错的是把它读成「掩掉的行走另一条 CE」。
> **代码为准**（`scripts/train_sft.py:1601-1647::soft_cross_entropy`）：
>
> ```python
> return float(soft_weight) * (per_row * mask).mean()      # per_row = −Σ soft·log_softmax
> ```
>
> ⇒ **`mask = 0` 的行既不进分子也不进分母，贡献恰好 0**，
> **不退化成 one-hot CE**、也不与软项按比例插值。
> 函数自己的 docstring 就写着这句话，`tests/test_soft_ce.py::
> test_mask_is_not_a_mix_with_one_hot_ce` / `test_masked_row_contributes_exactly_zero`
> / `test_denominator_is_batch_size_not_mask_sum` 三条把它钉死。
>
> **后果（必须写进排期）**：分母**恒为 batch 大小 B，不是 `Σmask`**
> ⇒ 软项的量级随「本批软行占比」线性变化。在 34.2M 行上开 `--soft-index`
> 而**不开** `--soft-only-sampling` ⇒ 只有约 **1%** 的行贡献
> ⇒ **policy 项被缩小约 100×**，等于「以为在蒸馏，其实在用 1% 的学习率训」。
> ⇒ **段 2 必须配 `--soft-only-sampling`**（把占比拉到 ≈1，让 `B == Σmask`）。
> `--soft-weight` 是**全局量级旋钮**，不是占比旋钮：不采样时调它
> 等于在调「软行占比」，同一个值在不同 batch 组成下含义不同。

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
| §5.6 必须先修 game_id 残行 bug（阻塞 5.3） | **降级** | 哈希锚点法对 id 复用免疫，不再阻塞。⚠ 但仍建议修：它污染 `train_sft.py` 的按局切分（⚠ 原写 `:2396` —— 该行号已漂移，实际在 **`:2780-2788`** 的 `unique_games` / `train_idx` 分支） |
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
- 仍走 `sdpa_force_math=True`（`train_sft.py:2641-2740` 的分派，910A 命中 `True`）⇒ `q@k^T` 与 softmax 输出必须物化 ⇒ `(3200,4,361,361)` fp16 = **3.114 GiB/份**；
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
| `mcts.py:250-253` transposition `cache_key`、Zobrist `hash()` | 不变（本次不引入新盘面状态字段）。⚠ 实际路径是 **`src/search/mcts.py`**（不是 `src/game/`） |
| `sample_batch_numpy` 默认签名 | 仍是三元组（`labels=False` 为默认） |
| 旧 17ch 推理路径 | `src/inference.py` 的 `register_in_channels_builder` 分派（22 在 **`:177`**、12 在 `:228`）、`_build_state`（**`:592`/`:622`**）、`_forward_batch`（**`:677`/`:699`**）保留。⚠ 原写 `:501/:567/:674` —— 该文件长了约 300 行，行号已漂移 |
| 12ch「位相同」契约 | `test_katago_se.py:189` 继续通过 |

### 7.2 扩展点（复用既有机制）

- **模型分发**：`src/inference.py::register_in_channels_builder(in_channels, builder)` —— 新增 `register_in_channels_builder(22, builder)`；现有 12/14/16/17 分支的「缺接线」报错与提示（`test_katago_se.py:251/267`）自动适用。
- **预算测试**：照抄 `tests/test_v19_budget.py` 形状（脚本 flags → 实例化 → 断言 + 显存余量 ≥ 4.0 GB）。
- **训练脚本**：照 `shell/train_sft_npu_4card_katago_se.sh` 新增。

### 7.3 文件清单

> 🔴 **本表原先写的是「计划中的文件名」，而实现期的实际落点已经改了。**
> 下表已按**仓库现状**订正（`✅` = 已存在且已接线；括号里是原设计的名字）。
> ⚠ 一条经验：**别在 spec 里提前命名文件** —— 22 通道这一摊最后拆成了
> **3 个模块**而不是原计划的 1 个，因为 `iterLadders` / 邻行 gather / 装配
> 三件事各自有独立的纯函数可测性边界（见配套 spec 的 B2–B4）。

**新增**
| 文件 | 状态 | 内容 |
|---|:--:|---|
| `src/networks/katago_v7.py` | ✅ | stem、`Nbt2TransformerBlock`、`TransformerBlock`、fson `NormAct`、`RMSNormMask`、2D RoPE、KataGPool、四头 |
| `src/data/feature_v7.py` | ✅ | **（原写 `src/game/go_features_v7.py`）** 装配 + 3 气桶 + 历史 5 手交错 + `calculateArea` + `passWouldEndPhase` + 19 维全局（**不放 `go_rules.py`**，避免污染 17ch 实现所在文件） |
| `src/data/feature_v7_ladders.py` | ✅ | `iterLadders` 移植（ch14–17）+ §2.2 的历史回退复制门控 |
| `src/data/feature_v7_gather.py` | ✅ | 邻行 gather（5 偏移）+ `FUTUREPOS_OFFSETS` / `LADDER_OFFSETS` |
| `src/data/katago_npz.py` | ✅ | stdata 读取器（§9.1 的七键 / 位解包 / 按网络名分派布局） |
| `src/networks/katago_v7_loss.py` | ✅ | 12 项 loss 装配 + `build_score_distr_target` |
| `scripts/build_games_sidecar.py` | ✅ | 🔴 **§5.3 整个 sidecar 方案都建在它上面**（原表**完全没列**，是最大的漏项）：`g_komi`/`g_score`/`g_rules`/`g_resign` 四个键、哈希锚点 `_AnchorIndex`、覆盖报告 |
| `scripts/crosscheck_stdata.py` | ✅ | 🔴 逐通道对拍工具（原表没列）：带 `ALIGNED`/`KNOWN_DIVERGENT`/`NOT_COMPARABLE` **三档资格**（§9.12.2） |
| `scripts/bench_v7_features.py` + `benchmarks/bench_v7_features.json` | ✅ | 🔴 C0 特征基准（原表没列）：§5.2 的 (a)–(f) 全部来自它 |
| `scripts/build_soft_index.py` | ✅ | 软标签 join 的 **CLI 委派**（缓存在 `src/data/kata_label_join.py`，§5.7） |
| `src/data/kata_label_join.py` | ✅ | `build_soft_index` + `materialize_dataset` + `hash_spec_fingerprint` |
| `scripts/smoke_train_v7.py` | ✅ | 端到端冒烟（302 行 / 40 步，§4.2 / §9.7） |
| `shell/train_sft_npu_4card_katago_v7.sh` | ❌ **未写** | 训练脚本（**计划中**，仓库里只有 `..._katago_se.sh` / `..._v19.sh` / `..._4card.sh`） |
| `tests/test_katago_v7_loss.py` | ✅ | 逐项系数对照 §4.5（20 个 test 函数） |
| `tests/test_katago_v7_budget.py` | ✅ | 参数/显存预算守卫（**原表没列**，实际存在） |

**测试文件（🔴 原表列的三个名字都不存在，按仓库现状展开）**

| 原表写的 | 实际 | 内容 |
|---|---|---|
| `tests/test_feature_planes_v7.py` | `tests/test_v7_planes.py`（21 个 test 函数） | ch0–6 / ch9–13 形状与等价性 |
| | `tests/test_v7_ladders.py` | `iterLadders`（含 §9.4 的诱饵说明 `:457-474`） |
| | `tests/test_v7_area.py` | `calculateArea` / Tromp-Taylor 等价 |
| | `tests/test_v7_globals.py` | 19 维全局（`KOMI_SCALE` 等） |
| | `tests/test_v7_assemble.py` | 装配 + `_PINNED_EXACT = (0,1,2,3,4,5,6,8,14,17)`（= §9.12.2 的空间 `ALIGNED` 集合） |
| | `tests/test_v7_gather.py`（19 个 test 函数） | 邻行 gather 5 偏移 |
| | `tests/test_crosscheck_tool.py` | 🔴 对拍**工具**的契约（资格标注不得被弄错、门禁只对 `ALIGNED` 生效） |
| `tests/test_dataset_v7_labels.py` | `tests/test_dataset_soft_labels.py`（28 个 test 函数） | 4 元组契约、`soft`/`soft_mask`、增广 |
| | `tests/test_dataset_futurepos.py`（28 个 test 函数） | 🔴 A6 的全部契约（§9.12.1） |
| | `tests/test_prefetch_labels.py` | 预取器透传 dict（占位必须原样搬） |
| | `tests/test_soft_ce.py` | 软 CE + 掩码语义（§5.7） |
| | `tests/test_soft_cli.py` | A4 四旗 + `resolve_policy_loss_kind` |
| | `tests/test_join_cache_freshness.py`（25 个 test 函数） | A5 缓存防陈旧 |
| | `tests/test_games_sidecar_alignment.py` | 锚点匹配率 + ×100 贴目修正 |
| | `tests/test_katago_npz.py` | stdata schema / `w_seki` 缺省 0 |
| | `tests/test_soft_labels.py`（8 项） | `permute_soft` 的 8 种对称 |

**修改**
| 文件 | 改动 |
|---|---|
| `src/data/sgf_parser.py` | 解析 `RU` 规则、`RE` 分差（**含 `W+3 zi` 单位后缀与 `W+0,25` 欧洲逗号小数两条显式分支**，§5.3.3） |
| `src/data/dataset.py` | 4 元组 + dict、`permute_soft`、`attach_soft` / `soft_row_mask`、`attach_futurepos` / `warm_futurepos` / `FUTUREPOS_SENTINEL` |
| `scripts/train_sft.py` | V7 loss 装配、`labels=True` 接线、预取 `pf.next()` 透传 dict、切 22 通道、A4 四旗（**65 个冻结旗里的 4 个**） |
| `src/inference.py` | `register_in_channels_builder(22, _katago_v7_net)`（`:177`） |

**不改**：`data/sgf_19x19_full.npz`（**不 rebuild**，§5.0）、
`go_rules.py` 的 `feature_planes`/`feature_planes_batched`、
`src/search/mcts.py`、`src/search/`、`test_go_feature_planes_v21.py`、
`test_param_budget.py`（钉的是 v18 = 12,858,978，与本模型无关）。

> ⚠ **原表把 `scripts/build_dataset.py` 列在「修改」下又说「不再改动」，
> 且把 `src/data/sgf_parser.py` 列了两次。** 现已合并。

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
   （`src/inference.py:177`；22 → `NbtTfNet` 5,561,832；12 → 旧 `AlphaGoNet`；17ch 分支不受影响）
6. 🔴 **验收 #6 已替换**（§5.2 决定不再自己算 ch0–13 ⇒ 「与旧 17ch 逐位相等」
   不再是回归锚点）。**新锚点 = 与官方 stdata 逐位对拍**：
   - `feature_v7(...)` 形状 `(22,19,19)`，`dtype` 与 `feature_planes_batched` 一致（fp16）
   - **ch14、ch17 与 `katago/stdata` 的同名通道逐位相等**（当前盘通道，stdata 可直接验证）
     —— **实测已达 `1.000000`**（200 npz / 4,368 行），见 §9.8
   - **ch15、ch16 对 SGF 重放逐位相等**（stdata 无前一/二盘的盘面，无法直接对拍；
     此处拆成「算法由 ch14 担保 + 盘面由重放担保」两项，见 §9.4）
   - **空间 ch3/4/5 与 stdata 逐位相等** —— **实测已达** `1.000000`（1,254 行）：
     ⇒ 气桶**不分色**、桶是 **`==1`/`==2`/`==3`**（不是 `>=3`）、整块去重口径正确
   - 🔴 **ch9–13 从这一条里删掉** —— 官方数据**无着法序列**，
     它们在 `scripts/crosscheck_stdata.py` 里归 **`NOT_COMPARABLE`**，
     报出来的对齐率没有意义（见 §9.12.2）。保留的只有「落点身份统计」：
     ch9 **100%** 落在 opp 子上、ch10 **99.4%** 落在 pla 子上、
     ch11/12/13 = 99.4 / 97.9 / 96.7%，差额是**落子被提掉**
     （该点现在为空，两边都判 0）⇒ 顺序 `opp, pla, opp, pla, opp` 属实
     （**对调会掉到个位数**）。⚠ 这**不是**逐位对拍，验收上不得当作已对齐。
   - 🔴 **ch18/19 从这一条里删掉 —— 它不能与 stdata 对齐**（1,254 行里 0 行相同，
     多 21,896 点；判据与证据见 §2.5）。它改为「`SCORING_AREA && TAX_NONE`
     下与 Tromp-Taylor 等价」的本地口径 + 源码引文人工复核，**无可执行 oracle**。
     归类 `KNOWN_DIVERGENT`（不是 `NOT_COMPARABLE` —— 判据已知，只是官方多一步提死子）。
7. `pytest tests/test_katago_v7_loss.py`：12 项系数逐条断言 `1.0 / 0.15 / 1.20 / 1.5 / 0.02 / 0.02 / 0.001 / 0.0015 / 0.006 / 0.25 / 0.25 / seki=8·0.005/(0.005+EMA)`
   —— **已实现并通过**（20 个 test 函数，parametrize 展开后更多）
8. `labels=False` 时 `sample_batch_numpy` 返回仍可被 3 元组解包（`bench_train.py:81/105` 不改）
9. 🔴 **原写的 `pytest tests/test_dataset_v7_labels.py` 不存在**，按仓库现状拆成两个文件：
   - `pytest tests/test_dataset_soft_labels.py`：`games.npz` 局表 gather、权重掩码、
     增广下的标签变换、4 元组契约（`soft`/`soft_mask` 在 dict 内）、
     **`outcome_black` 的翻转口径**（§5.5）、占位标签恒 0 + 对应权重恒 0
   - `pytest tests/test_dataset_futurepos.py`：A6 的全部契约（§9.12.1）——
     哨兵 `-1.0`、单路存活 ⇒ 整块权重 0、两个 horizon 共用同一个 `tform`、
     **关闭时 payload 逐字节相同**、`labels=False` 时**完全不碰** gather
   - ⚠ **软标签掩码那条的措辞已订正**：`soft_mask = 0` 是「该行**不参与**软项」
     （贡献恰好 0），**不是**「走 one-hot CE」。见 §5.7。
10. `pytest tests/test_games_sidecar_alignment.py`：哈希锚点匹配率**必须打印**；
      匹配项的 `g_komi`/`g_score` 与 SGF 直读一致
      —— ⚠ 该测试还须覆盖 §5.3.4 的**三个坑**：`game_ids` 是置换（相邻比较切段）、
      偏移从 1 起（空盘不可作锚）、`L < 40` 的 134 段用**偏移 1..20 全前缀**（3.4M 散列 / 27 MB）；
      以及 §5.3.1 的 ×100 贴目修正（`test_komi_outlier_fix_is_opt_out`）
      —— ✅ 全部已实现并通过
11. `pytest tests/test_katago_npz.py` + `tests/test_kata_label_join.py` +
    `tests/test_join_cache_freshness.py` + `tests/test_crosscheck_tool.py`：
    stdata 布局分派（64/80 列、有无 Q 值）、`pos_hash` join、`permute_soft` 双向、
    join 缓存防陈旧（§5.7）、对拍工具的三档资格标注（§9.12.2）
    —— ⚠ 原写的 `test_v7_vs_official.py` **不存在**，其职责落在
    `test_v7_assemble.py`（`_PINNED_EXACT`）+ `test_crosscheck_tool.py`

**远端必验（Task 0，本地无法替代）**
12. 910A 探针：`npu_fusion_attention_score` / `npu_flash_attention_score` 存在性；`sdpa_force_math` 关闭后前反向数值一致
13. `max_memory_allocated ≤ 31.0 GiB @ B=3200/卡` —— **实测值回填** `test_katago_v7_budget.py` 的估算项
14. 🔴 **22ch 特征预取吞吐 ≥ NPU 吞吐 —— 本机实测「不成立」**：
    3.04 ms/行 是 8 worker 下的预算，C0 实测 **3.66 ms/行（冷读）超 1.2×**
    ⇒ 触发的处置按 §5.2 (f) 与 §8.3：要么把 `boards.npy` 放到**本地 NVMe**
    （冷热差 4.3×，那 4.3× 里有 3.04 ms 的全部余量），要么启用
    **降级档位（18 通道，ch14–17 恒 0）**。**不要**靠加 worker 解决冷读
    （实测冷读 2 worker 比 1 worker 还慢）。

### 7.5 已识别的冲突

| 冲突 | 处置 |
|---|---|
| `test_grad_checkpointing.py`（32 项）/ `test_attn_dropout_eval.py`（5 项）只覆盖旧模型 | 新模型无 dropout、检查点粒度不同 ⇒ **必须新增** nbt2 块的检查点测试 |
| `test_param_budget.py` 钉 v18 | 不改；v7 用独立测试锁自己的数 |
| `search_arch.py` 标定失配 | 新增投影（§7.3） |
| `test_katago_se.py:13` 引用了不存在的 `test_v21_budget.py` | 顺手补上，否则文档指向悬空 |
| 🔴 **`test_no_new_cli_params` 把 `train_sft.py` 的 65 个 flag 整个冻结**（D1 零新增/零删除/零改名）；`test_policy_loss_default_is_ce` 钉住 `--policy-loss` 的 **default = `ce`** 且 choices `== {huber, ce, soft_ce}` | ✅ **A4 已登记那 4 个软标签旗**（`--soft-index` / `--soft-weight` / `--soft-only-sampling` / `--soft-every`）。**段 2 现在可以从命令行启动**（原写的「`soft_ce` 未接 CLI ⇒ 无法启动」已作废）。后续任何新旗仍需同步登记。详见 **§9.11.1** |
| ⚠ **`tests/test_run_txt_sync.py`（整文件）与 `test_param_budget.py::test_run_txt_records_are_consistent` 已删**（用户决定，v18 已退役） | 不恢复。🔴 **随之失去的保证：run.txt 里的数字/flag 与实测值的自动比对** ⇒ 补参数时**必须人工核对** run.txt。🔴 **且它已经咬了一次**：`run.txt` 被改成「一屏」版后**只剩 55 行**，而 §5.2 引的 `run.txt:700` 直接悬空（§5.2 已就地标注）。详见 **§9.11.2** |
| 🔴 **ch18/19 与官方 stdata 无法对齐**（1,254 行 0 行相同，多 21,896 点） | 验收 #6 把它从 stdata 对拍里**删掉**（§7.4 #6）；改为本地口径（Tromp-Taylor 等价 + 源码引文复核）。**按对齐口径优先，没补死子判定** —— 判据要猜，猜错会得到第三种偏差。归 `KNOWN_DIVERGENT`。详见 §2.5 / §9.8 / §9.12.2 |
| 🔴 **ch9–13 / ch7 / ch20-21 / ch15-16 与官方 stdata 结构上不可比** | 从验收 #6 的「逐位相等」里**删掉**；`scripts/crosscheck_stdata.py` 归 **`NOT_COMPARABLE`**。⚠ **老陷阱：ch9 的 `cell_agree = 0.9972` 但 `row_exact = 0.000000`**（每行只差 1 个点），报 `cell_agree` 等于把一个彻底错的通道说成几乎完美。详见 **§9.12.2** |

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

**Phase 1 — 输入与标签（数据侧）** ✅ **已全部落地**（实际文件清单见 §7.3）

1.1 `sgf_parser`：解析 `RU` 规则、`RE` 分差（含 `W+3 zi` 单位后缀与 `W+0,25` 欧洲逗号小数两条**显式**分支，§5.3.3）
1.2 ~~`build_dataset`：先修 game_id 残行 bug~~ —— ❌ **未做，但不再阻塞**
     （哈希锚点法对 id 复用免疫，§5.3.4）
1.3 ~~`build_dataset`：`next_move` 列 + 局表 `g_*`~~ —— **作废**（不 rebuild：`next_move` 改由 `moves[i+1]` + 局守卫实时推；局表走独立 `games.npz`，§5.3）
1.4 ~~`go_features_v7.py`~~ → **`src/data/feature_v7.py`**：`feature_planes_v7_batched` 的 **ch0–6、9–13** ✅
1.5 移植 `calculateArea` → ch18/19；`passWouldEndPhase` → global14 ✅
1.6 移植 `iterLadders` → ch14–17（含 §2.2 回退复制语义）✅（§9.4 的两个反直觉性质已记）
1.7 19 维全局特征（含 global5 的 `currentSelfKomi` 符号约定）
1.8 `dataset.py`：局表 join、邻行 gather、`labels=True` 分支
1.9 对称增广下的标签变换
1.10 🔴 **测试（原写 `test_feature_planes_v7.py` / `test_dataset_v7_labels.py` —— 这两个名字都不存在）** ✅ → `test_v7_planes.py` / `test_v7_ladders.py` / `test_v7_area.py` / `test_v7_globals.py` / `test_v7_assemble.py` / `test_v7_gather.py` / `test_dataset_soft_labels.py` / `test_dataset_futurepos.py`

**Phase 2 — 模型与损失（网络侧）** ✅ 除 2.6 / 2.8 外已落地

2.1 `katago_v7.py`：fson `NormAct`、`RMSNormMask`、2D RoPE、`TransformerBlock`
2.2 `Nbt2TransformerBlock` + stem + trunk-end（§3.3 的 K 调度）
2.3 四个头（§4.1–4.4）
2.4 `register_in_channels_builder(22, ...)` ✅（`src/inference.py:177`）
2.5 V7 loss 装配（§4.5 的 12 项系数）✅（含 #12 改用独立 `w_seki`，见 §4.5）
2.6 **部分**：`train_sft.py` 侧的 A4 四旗已接 ✅；❌ `shell/train_sft_npu_4card_katago_v7.sh` **还不存在**
2.7 `test_katago_v7_budget.py` + `test_katago_v7_loss.py` ✅
2.8 顺手补 `test_v21_budget.py`（修 §7.5 的悬空引用）—— ❌ **仍未补**

**Phase 3 — 明确不做（本 spec 范围外）**
- 13 项 search-dependent loss（需 MCTS reanalysis 数据）
- 非 19×19 泛化（`KataGPool` 的 board-size 缩放项目前是常数）
- 融合注意力的深度优化

**沿用的旧知识**：旧计划 Task 4 的 pass bug —— pass **不得**重置 `moves_since_capture`（应 `+=1`）且 `consecutive_passes +=1`。

### 8.3 回到设计的触发条件

| 触发 | 回到 |
|---|---|
| Phase 0 实测 `>31.0 GiB @ B=3200` | §6 R2 降级路径（减 batch → 减块数 → 才换结构） |
| 🔴 **`iterLadders` 移植在 19×19 上性能不可接受 —— 已触发**（§5.2 (f)：8 worker 冷读 **3.66 ms/行**，超 3.04 预算 **1.2×**；梯子占管道 **99.8%**） | §2：**ch14–17 降级为常数 0（18 通道）**。⚠ 本仓可走的处置有两条，**优先第一条**：① `boards.npy` 挪到本地 NVMe（冷/热差 **4.3×**，热态 0.86 ms/行 有 3.5× 余量）；② 才启用 18 通道降级。**不要**靠加 worker —— 冷读 2 worker 比 1 worker 还慢 |
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

⇒ 实现：`src/data/katago_npz.py`；测试：`tests/test_katago_npz.py`（21 个 test 函数，含两条真实 stdata 数据对拍）。

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

> 🔴 **补充（2026-10-02 复核）：ch15/ch16 的 `NOT_COMPARABLE` 归类 + 「巧合」这两个数要一起记。**
> `scripts/crosscheck_stdata.py` 在 301 行上给 ch15 / ch16 报出
> **`0.7276` / `0.7076`** —— 🔴 **那不是对齐率**。原因：我们在这两格上走的是
> **官方自己的回退复制分支**（`prevBoard = board`），所以拿它去比官方 ch15，
> **实际对齐的是 ch14**（ch14 的实测是 `1.000000`）。
> ⇒ 归 `NOT_COMPARABLE`，**不归 `ALIGNED`**：归 ALIGNED 会让 `--min-rate 1.0`
> 永远过不了（它到不了 1.0），等于把一个永不满足的门槛写进门禁。
> 同理上表的 ch9–13 —— 官方数据无着法序列，比出来的任何数都没有意义。
> 完整三档资格见 **§9.12.2**。

##### ✅ 订正：ch14 / ch17 **实测已逐位对齐**（`1.000000`）

上表写的是「能否对拍」，暗示还没对。**实测已完成**：

| 通道 | 样本 | 逐位相等率 |
|---|---|---:|
| ch14 | 12 npz / 301 行 → 40 / 776 → **200 / 4,368** | **1.000000**（三档全对） |
| ch14 非零格 | 4,368 行上 ours **22,528** == official **22,528** | 逐格对上 |
| ch17（`to_play = +1`） | 12 / 40 / **200** npz → **4,368** 行 | **1.000000** |
| ch17（`to_play = −1`，对照假设） | 200 npz / **4,368** 行（816 行相等） | **0.186813** |

⇒ 三个结论被实测**钉死**，不再是推断：

1. **ch14 与 `to_play` 无关** —— 两种假设下**都**实测 `1.000000`。
   ⇒ spec 此前把它当「假设」，**实测是事实**（函数签名保留 `to_play` 是为了
   ch17 同批出，不是 ch14 需要它）。
2. **ch17 只在 `to_play = +1` 下对齐** ⇒ **stdata 隐含 `to_play` 恒 +1**
   （等价于「恒黑先行」的自对弈数据），得到证实。
3. ⚠ `0.186813` **必须连样本量一起引用** —— 它随样本量明显漂移（归档按 npz 走），
   孤零零写一个 `0.186813` 会误导。

##### ✅ 订正：ladder 算法的两个**反直觉**性质（不是泄漏，别当 bug 修）

1. **远处的子会改变 ladder 判定。**
   2 气的 ladder 会跑**完整追逃**（遍历棋盘）才提子 ⇒ 盘面上任何一颗看似无关的
   子，只要落在逃跑延长线上，就**真的**改变结论。
   ⚠ 测试里放「远处无关的诱饵子」会让预期落空 —— **那不是泄漏，是算法的真实性质**。
   实例：白被追到 **(16,17)** 才被提掉（**34 个搜索节点**），而诱饵表里那颗
   (16,17) 白子**正好在逃跑延长线上**。处理办法是**把 (16,17) 从诱饵里挪走**
   （**不是**调阈值迁就实现 —— 诱饵的唯一职责是「不在被验对象自己的路径上」，
   落在线上是**诱饵表写错了**）。挪走后候选块仍是 12 块。
   `tests/test_v7_ladders.py:457-474` 有完整说明。
2. **保留的 C++ 上游怪癖（故意不「修」）**：
   - `boundNumLibertiesAfterPlay` **不去重**（数的是关联次数），
     而 `findLibertyGainingCaptures` **去重** —— 上游自己就不一致，
     `lowerBound` 的语义依赖这个不一致（`lowerBound = 3` ⇒ 防守方直接脱逃）。
   - 1 气分支**不清** `workingMoves`，沿用上一次的值。
   ⇒ **`libs > 1` 的守卫是 ch17 真正的兜底**，不是冗余判断；去掉它，
     `workingMoves` 的跨块残留会泄进 ch17。

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
| | | | **score_stdev** | **−0.0%** ← 该列是 `beta=0.05`（spec 字面值）那一版的实测；**已裁决改 `1.0`**，见 §4.2 |

⚠ **`scoring` 开局就占 83/104**（随机预测 vs ±120 目标，MSE ≈ 1.4e4），
梯度范数在 step 20 冲到 365、被 clip 到 5 ⇒ 早期有效步长被压得很小。
真实跑要相应加 warmup，或先确认 `scoring` 的 ±120 缩放是否该归一。

### 9.8 🔴 哪些通道**已实测逐位对齐**，哪些**不能**（2026-10-02）

> 这一节是 §7.4 验收 #6 的实测结论表。**之前 spec 只写了「要去对拍」，
> 没写对拍的结果**，导致 §2.5 / §9.4 把 ch18/19 也当成了可对拍的通道。

#### ✅ 可当 oracle 且实测 `1.000000` 的通道

| 通道 | 样本量 | 结论 |
|---|---|---|
| **空间 ch3 / ch4 / ch5**（1/2/3 气桶） | 1,254 行（`zzb28c512` 首个含 19×19 的 npz） | **1.000000** |
| **空间 ch14**（当前盘梯子） | 200 npz / **4,368** 行（另有 12 / 301、40 / 776 两档） | **1.000000** |
| **空间 ch17**（working move，`to_play=+1`） | 200 npz / **4,368** 行 | **1.000000** |
| 空间 ch0–2 / ch6 / ch8 | 由 `_PINNED_EXACT` 钉住 | `ALIGNED`（ch1/ch2 的理由见 §9.12.2：`to_play ≡ +1`） |

这些是**好消息**：它们把三条原本靠推理的假设变成了实测事实 ——
① 气桶**不分色**（只数气，不看是谁的子）；② 气桶判据是 **`==1` / `==2` / `==3`**
（**不是 `>=3`**）；③ **整块去重**口径正确（一个空点被同块两子共享时只算 1 气）。

**ch14 与 `to_play` 无关，是实测不是假设**；**ch17 只在 `to_play = +1` 下对齐**
（`−1` 假设实测 **0.186813** / 200 npz / 4,368 行）⇒ **stdata 隐含 `to_play` 恒 +1
得到证实**。细节见 §9.4。

**ch9–13 的 99.x%（⚠ 这不是对齐率）**：实测（1,254 行）ch9 **100%** 落在 opp 子上、
ch10 **99.4%** 落在 pla 子上、ch11/ch12/ch13 = **99.4 / 97.9 / 96.7%**。
剩下的 0.6%~3.3% 是**落子被提掉**的情形 —— 那时该点现在是空的，通道本来就该是 0，
我们和官方**都**判 0（所以那是「统计口径」而不是「不一致」）。
⇒ **顺序 `opp, pla, opp, pla, opp` 属实**：若把它对调成 `pla, opp, pla, opp, pla`，
这组数会掉到**个位数**（反向检验，不是正向证据 —— 正向证据仍是 ch9/ch10 的落点身份）。
🔴 **但 ch9–13 归 `NOT_COMPARABLE`**（官方数据无着法序列）⇒ **不能**把它们
和上面那张 `1.000000` 的表并排放。完整三档资格见 **§9.12.2**。

#### 🔴 不能对齐 / 不可比的通道

| 通道 | 归类 | 证据 |
|---|---|---|
| **ch18 / ch19**（当前区域） | `KNOWN_DIVERGENT` | 1,254 行：**0 行完全相同**；我们**多 21,896 点**（19,913 子 + 1,983 空点）；**295/1254** 行受影响 |
| **ch9–ch13**（历史 5 手交错） | `NOT_COMPARABLE` | stdata 是**逐行独立样本、无着法序列**，重建这 5 格需要序列本身 ⇒ 结构上到不了。⚠ 且 ch9 的 `cell_agree = 0.9972` / `row_exact = 0.000000`（每行只差 1 个点）—— **报 `cell_agree` 等于把一个彻底错的通道说成几乎完美** |
| **ch15 / ch16** | `NOT_COMPARABLE` | stdata 无前一/二盘的盘面 ⇒ 我们走官方**回退复制**分支。301 行上的 `0.7276` / `0.7076` **是巧合**（对齐的其实是 ch14），不是对齐率 |
| **ch7 / ch20 / ch21**（encore） | `NOT_COMPARABLE` | 官方**只在 encore 行**置位，而 `spatial_channels_v7` **没有 `encore_phase` 入参** ⇒ 结构上到不了。实测 301 行里恰好 **1 行** `encorePhase=2`，三个通道都只在那 1 行不一致 |
| 全局 `0–4` / `14`（pass 标志、`passWouldEndPhase`） | `NOT_COMPARABLE` | 同「无着法序列」 |
| 全局 `15` / `16`（PDA） | `NOT_COMPARABLE` | 本仓无 PDA ⇒ 恒 0。🔴 **301 行上官方恰好也恒 0 ⇒ `row_exact` 报 `1.000000`，那是巧合**；200 npz / **4,368** 行上官方有 **133 个非零点 ⇒ `0.9696`** |
| 全局 `5`（`currentSelfKomi`） | `NOT_COMPARABLE` | 官方那一列含 **draw-jitter**，本仓喂的是 `game_row.komi` ⇒ **不是同一个量** |

**ch18/19 的完整排查**（最小反例、否掉的两个假设、以及「不偷偷补死子判定」的决定）
在 **§2.5**。一句话版：

> 🔴 **stdata 只能当「空间 ch0–6、ch8、ch14、ch17」+「全局规则位那几个」的 oracle。
> ch18/19 是 `KNOWN_DIVERGENT`（判据已知，不是「还没查出来」）；其余上表是
> `NOT_COMPARABLE`（连判据都不该去找）。**
> 官方很可能在算 area 前**先提掉死子**，本仓 `GoBoard.score()` 没有死子判定；
> 判据要猜，猜错会得到**第三种**偏差 ⇒ **按对齐口径优先，没补死子判定**。

#### 为什么这一节重要

| | 之前 spec 的假设 | 实测 |
|---|---|---|
| stdata 的覆盖面 | 22 通道全部可对拍 | **可当 oracle 的只有空间 ch0–6/ch8/ch14/ch17**（+ 全局规则位）；ch18/19 已知有差；其余结构上不可比 |
| 「对齐了」的含义 | 通道集合逐位相等 | ch9–13 是**落点身份统计**（且不可比）；ch18/19 **永久不是** |
| ch18/19 的验收 | 与 stdata 逐位 | 本地口径（Tromp-Taylor 等价 + 源码引文复核），**无 oracle** |
| 「1.000000 出现了」 | 就是对齐了 | 🔴 **可能只是巧合**（全局 15/16 在 301 行上就是 `1.000000`）⇒ 必须先看资格标注 |

### 9.9 SGF 语料的实测性质（决定了哪些 loss 能在段 1 训）

抽样 133,604 个 SGF 实测（**早期抽样** —— ⚠ 下表的 `RE` 认输率与 `KM` 分布都已被
全量实测修正，见本节末尾的订正小节）：

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

##### 🔴 订正（2026-10-02）：上表是**早期抽样**，全量实测是 **81.09%**

`scripts/build_games_sidecar.py` 在**全部 162,298 局**上实跑（分母 = 入库局数）：

```
认输    81.09%   = B+R 21.94 + W+Resign 21.85 + W+R 21.75 + B+Resign 15.55
有分差  ≈18.5%   （g_score 可用）
```

差 **5.9 个百分点**，方向是**更糟**。原因：抽样只扫 `data/games/games/` 的
4 个目录，**漏掉了 5 个 tgz 里的 36,274 个 SGF**（§5.0 已核实重叠 = 0）。
⇒ 本节与 §5.3.3 的 **75.2% 全部按 81.09% 读**。

**另有两个未列的 `RE` 形式**（各 ≤2 局，但**都有显式分支**，不能靠默认）：
`W+3 zi`（单位后缀）与 `W+0,25`（欧洲逗号小数）。

**另有一个未列的 `KM` 形态**：`foxpro` 的 **×100 家族**
（`KM[650] [750] [550] [450] [375] [325]`，**2,163 局 = 1.62%**）。
照字面存会让 `currentSelfKomi/20 = 32.5` ⇒ 已做成**默认开启**的
`--no-komi-outlier-fix` 反向开关。细节见 **§5.3.1**。

🔴 **订正后的第 1 条**：**81.09%** 的局是认输 ⇒ 12 项里 5 项
（#5/#6/#8/#9/#10）在 **~81%** 的行上权重为 0；有监督的约
`18.5% × 162,298 × 210 ≈ 6.3M 行`（34.2M 的 **~19%**）。

🔴 **更根本的：语料性质决定 loss 的启用顺序。** 认输局即使重放到终局，
`go_rules.py:2269` 的 docstring 自陈「没有死子判定，对局双方应在 pass 认输前
实际提掉对方的死子，否则死子所在点会被判为双方共邻的中立区域」
⇒ 从认输棋谱反推的终局归属不可靠。

⇒ **因此段 1 只训 policy、π_opp、value、futurepos**：
`#1 policy`、`#2 π_opp`、`#3 value`（`winrates` 免费）、`#11 futurepos`；
**score 系（#5/#6 scorebelief、#8 scoremean、#9 lead、#10 scoring）不作为
段 1 的主目标**（81.09% 的行权重天然为 0，只覆盖 ~19%）；
`#4 ownership`、`#10 scoring`、`#12 seki` **不在段 1 训**。
⇒ **段 2/3（1% 标注 + stdata 3.1M）才开全部 12 项** ——
那 3.4M 上这些信号来自搜索器自己的评分器（`valueTargetsNCHW`），质量高得多。

> **实测支撑（本轮补）**：这条「段 1 训什么」的决定此前只写在 decision 里、
> spec 里没有数字。支撑就是上面那条 **81.09%**：它把 score 系的覆盖率从
> 「75% 的局上权重 0」推到「~81% 的行上权重 0」，
> **`g_score` 覆盖率只剩 ≈18.5%** ⇒ 段 1 跑一个 18.5% 覆盖的 score 头
> **不如等段 2/3**（后者 3.4M 行 100% 有覆盖，且标签来自搜索器评分器）。
> 同理 ownership/scoring/seki 更是**完全不在段 1** —— 它们依赖终局盘面归属，
> 而 81.09% 的局根本没有可靠的终局（§2.5）。

### 9.10 本 spec 相对初稿的偏差清单

| # | 位置 | 偏差 | 状态 |
|---|---|---|---|
| 1 | §6.1 | 「块小计 ×11 = 5,479,680」行标签张冠李戴（该数不是 11 的倍数；11 块实为 **5,423,616**） | **已订正**，测试钉住 |
| 2 | §4.2 | `scorestdev` 的 `SoftPlus(x₁, 0.05)`（**spec 字面值**）与 loss #7 的 δ=10 矛盾，实测该项无学习信号 | ✅ **已裁决为 `1.0`**（2026-10-03）。推导链与实测数字见 §4.2；`spec` 正文保留 `0.05` 作为字面值记录，**代码不实现它**（`SCORE_STDEV_SOFTPLUS_BETA = 1.0`） |
| 3 | §5 | 原设计（build 加列 + 局表并入主 npz）被「不 rebuild + 实时算 + 独立 sidecar」取代 | **已重写** |
| 4 | §5.4 | 批契约从「4 元组 + dict」被实现成「5 元组」 | **已订正回 4 元组** |
| 5 | §7.4 #6 | 验收锚点从「与旧 17ch 逐位相等」换成「与官方 stdata 逐位对拍」 | **已替换** |
| 6 | §7.4 #12 | seki 的行权重从 `w_ownership` 改成独立 `w_seki` | **已改** |
| 7 | §3.2 | 2D RoPE 的 `(H,16,2)` 里那个 `2` 的语义 spec 未定义；本实现取「二维频率」（`[…,0]` 乘行、`[…,1]` 乘列） | 记录，参数形状与预算不变 |
| 8 | §2.2 | 8 路恒 0 通道在 stdata 上**不全是 0**（encore 开启 ⇒ ch7/20/21 非零率 0.01%/0.35%/0.39%）；我们无 encore 的简单局才为 0 | 记录；stdata 分布与我们不同 |
| 9 | §5.3 | 局级 sidecar 砍掉 `g_ownership`/`g_scoring`/`g_seki`（原 176 MB → **1.2 MB**），只留 `g_komi`/`g_score`/`g_rules`/`g_resign` | **已决定**，理由见 §5.3.0 |
| 10 | §4.5 #4/#10/#12 | 这三项**不在段 1 训**，改由段 2/3 的 stdata 提供 | **已决定**，理由见 §9.9 |
| 11 | §4.5 #5/#6/#8/#9 | 75% 的局是认输 ⇒ 这 5 项在 75% 的局上权重为 0（仅 6.6M / 19% 有监督） | 🔴 **被第 13 行取代**（真实值 81.09% / 6.3M）；见 §5.3.3 / §9.9 |
| 12 | §2.5 | `RU` 覆盖仅 45%，其中 `Japanese`（7.3%）是 **TERRITORY** 计分 ⇒ ch18/19 恒 0；另 55% 无 `RU` 按默认 AREA | 记录，见 §5.3.2 |
| 13 | §5.3.3 / §9.9 / §5.1 / §5.3.0 | **认输率 75.2% → 81.09%**（75.2% 是早期抽样，漏掉 5 个 tgz 的 36,274 个 SGF）。`scripts/build_games_sidecar.py` 在 **162,298 局**上实测：`B+R` 21.94 + `W+Resign` 21.85 + `W+R` 21.75 + `B+Resign` 15.55 ⇒ `g_score` 覆盖率 ≈ **18.5%** | **已订正**（§5.3.3、§9.9 保留原表并标「原以为是 X」） |
| 14 | §5.3.1 / §9.9 | **`KM` 还有一个 ×100 家族**（`foxpro` 的 `KM[650] [750] [550] [450] [375] [325]`，**2,163 局 / 1.62%**）。照字面存 ⇒ `currentSelfKomi/20 = 32.5` | **已订正**；÷100 修正做成**默认开**（`--no-komi-outlier-fix` 关闭），报告里计数 |
| 15 | §5.3.4 | 哈希锚点法三个坑：`game_ids` 是**置换**不是升序（切段必须**相邻比较**，不能假设 `np.diff` 符号）；**偏移必须从 1 起**（偏移 0 是空盘，169,878 个 SGF 完全相同）；`min(20, L//2)` 对 **134 段 `L<40` / 2 段 `L<20`** ≠ 20 ⇒ 改为存**偏移 1..20 全前缀**（3.4M 散列 = 27 MB） | **已订正** |
| 16 | §2.5 / §7.4 #6 / §9.4 / §9.8 | **ch18/19 无法与官方 stdata 对齐**：逐位对拍 **1,254 行**，我们**多 21,896 点**（19,913 子 + 1,983 空点），**0 行完全相同**，295/1254 受影响 ⇒ stdata 只能当 **ch3/4/5、ch9–13、ch14/ch17** 的 oracle | **已订正**；**按对齐口径优先，没有偷偷补死子判定** |
| 17 | §7.4 #6 / §9.4 / §9.8 | **ch3/4/5 = 1.000000**、**ch14/ch17 = 1.000000**（200 npz / 4,368 行；ch14 非零格 22,528 == 22,528）⇒ 气桶不分色 / 桶是 `==1/2/3` 而非 `>=3` / 整块去重 / ch14 与 `to_play` 无关（**实测非假设**）均已证实 | **已订正**（好消息，补进 spec） |
| 18 | §2.2 | 历史门控曾**被解析两次**：函数内算对，输出行独立重算并抄了 `cur[:,0]`（ch14）而非 `out[:,1]`（ch15），`history=1` 时**恰好差一手**。是 `go_rules.py:170-177` 的教训在 V7 里的**第二次复刻**（第一次是 17 通道 U 形块两个 bucket 同时漏） | **已修**；spec 加警示：**门控只能解析一次，只调一处实现** |
| 19 | §5.3.3 / §9.9 | 两个**未列**的 `RE` 形式：`W+3 zi`（单位后缀，**2 局**）、`W+0,25`（欧洲逗号小数，**1 局**） | **已订正**；两者都有显式分支 |
| 20 | §4.5 #7 / §4.2 | `SCORE_STDEV_SOFTPLUS_BETA` 原为 `0.05`（**逐字实现 spec 字面值**），预测初值 277.26、冒烟 40 步该项 **−0.0%** | ✅ **已裁决为 `1.0`**（2026-10-03）。实测预测初值 **13.86**、loss #7 公式值 **8.83**（落在 δ=10 以内）⇒ #7 **恢复为可训练项**。⚠ 段 1 的 8 项 score 系数仍逐个 0.0 ⇒ 段 1 四个主目标**逐位不变**；纯常量 ⇒ 结构/参数（5,561,832）/预算不变 |
| 21 | §7.5 / 新增 §9.11 | `tests/test_huber_loss.py::test_no_new_cli_params` 把 `train_sft.py` 的 flag 整个冻结（D1 零新增/零删除/零改名）；`test_policy_loss_default_is_ce` 钉死 `--policy-loss` 的 **default = `ce`**。🔴 **A4 已把它从 61 扩到 65**（`--soft-index` / `--soft-weight` / `--soft-only-sampling` / `--soft-every`），且 **`soft_ce` 已接进 CLI**（choices == `{huber, ce, soft_ce}`）⇒ 「段 2 无法从命令行启动」**已作废** | **已订正**（冻结集本身保留，价值在「新增必须一次留痕」） |
| 22 | §7.5 / 新增 §9.11 | **两个文档校验测试已删**（用户决定）：`tests/test_run_txt_sync.py`（整文件）与 `tests/test_param_budget.py::test_run_txt_records_are_consistent`（要求 run.txt 保留 `# v18 架构参数` 小节，而 v18 已退役） | **已记录**；⚠ **随之失去的保证：run.txt 里的数字/flag 与实测值的自动比对**（🔴 **已咬过一次**：run.txt 缩到 55 行，`run.txt:700` 悬空） |
| 23 | §5.7 / §5.2 / 配套 spec §1.2·§3 | **`soft_mask` 语义**：`0` 被写成「one-hot CE」。**代码为准**（`train_sft.py::soft_cross_entropy`）：`mask=0` 贡献**恰好 0**，不退化、不插值；分母恒为 `B` | 🔴 **已订正**。「二选一」保留（那句没错），错的是「0 = one-hot CE」。⇒ **段 2 必须配 `--soft-only-sampling`**，否则 1% 软行把 policy 项缩小 ~100× |
| 24 | §2.5 / §7.4 #6 / §9.8 / §9.4 | **ch9–13（及 ch7/ch20-21/ch15-16、全局 pass 位 / PDA）不是 stdata 的 oracle** —— 官方数据**无着法序列 / 无前一二手盘 / 无 `encore_phase`**，结构上到不了 ⇒ 归 **`NOT_COMPARABLE`** | 🔴 **已订正**。§9.8 第一版写的「stdata 覆盖 20 通道」**作废**。⚠ 附带发现：ch9 的 `cell_agree = 0.9972` / `row_exact = 0.000000` ⇒ **逐格口径会骗人**。见 §9.12.2 |
| 25 | §4.5 #12 / §9.5 | seki 的行权重：§4.5 表里写 `w_ownership`，§9.5 说「已改为独立 `w_seki`」—— **表没跟上**。代码为准（`katago_v7_loss.py:291-293`：`w` 里没有 `seki` 就填 0） | 🔴 **已订正**（§4.5 现在写 `w_seki`）。⚠ **`w_ownership` 在 #4 与 stdata 列名里出现都是正确的**（官方 `globalTargetsNC` col27）—— 别顺手一起改掉 |
| 26 | §5.3.4 | 「100% 的局 ≥20 手」⇒「162,296 / 162,298」的**诊断错了**。实测该分子**就是** `L ≥ 20` 的真实段数（99.9988%）；错在把 99.9988% 写成 100% 并外推出「ply-20 对所有局安全」 | 🔴 **已订正**。真正的风险全在坑 3（`min(20, L//2) ≠ 20`，134 段 `L<40`），与「有没有 `L<20` 的局」无关 |
| 27 | §5.7 | 「反推：10,612 局 ↔ 3.35M 行 ⇒ 316 行/局 ⇒ 34.2M 行覆盖 108,000 局」 | 🔴 **作废**：混用了两个下标空间。实测 **mean 210.74 行/局**、162,298 段 ⇒ **34.2M 行 = 全部局**。前 10,612 段 = 2,251,471 行（212.2/局）；3.35M 行对应 **15,845** 段 |
| 28 | §5.2 | C0 **已实跑**：① 梯子占管道 **99.8%**；② **batch 32→1024 只差 1.7%** ⇒ batch 与 worker **解耦**；③ 热 8 worker **1161 行/s**、冷 **273**（差 4.3×，且冷读 2 worker 比 1 worker 慢）；④ 复杂盘面 **5.5 ms/行**（超预算 1.8×）；⑤ 8 worker × 307 MB = **2.4 GB** | 🔴 **「22ch 余量 1.7×」作废**，`holds: false`（3.66 vs 3.04 ms/行）。⚠ **`prefetch-workers = 12` 是外推、未实测**（脚本只跑 1/2/4/8），引用须带「外推」 |
| 29 | §7.3 / §8.2 / §7.4 #9 / 配套 spec §3 B6-B7 | **三个文件名不存在**：`src/game/go_features_v7.py`、`tests/test_feature_planes_v7.py`、`tests/test_dataset_v7_labels.py`。另**四个实际存在但清单没列**：`scripts/build_games_sidecar.py`（§5.3 的地基）、`scripts/crosscheck_stdata.py`、`scripts/bench_v7_features.py`、`benchmarks/bench_v7_features.json` | 🔴 **已按仓库现状重写 §7.3**，§8.2 / §7.4 同步。⚠ `test_katago_v7_loss.py` / `test_katago_v7_budget.py` **确实存在，没动** |
| 30 | §5.5 | `labels_dict` 键表漏了 **`outcome_black`**（实现里一直有）。且 `outcome` 是 **to_play 视角**，翻转**不能**写成 `outcome * to_play` | 🔴 **已补**（§5.5）。⚠ 顺带修了 `train_sft.py:822/886/1042/2920` 的调用点行号（实为 `940/1004/1237/1252`） |
| 31 | §1.2 C5 | 「约 700 项」与 §7.4 #1 的「1011 passed / 2 skipped」自相矛盾 | 🔴 **已统一为 1011 / 2 skipped** |
| 32 | §2.3 / §2.4 | 「恒 0」是对**本仓简单局**的断言，不是对官方的。空间 `7`/`20`/`21` 在 301 行里有 **1 行**不一致；🔴 **全局 `15`/`16`（PDA）在 200 npz / 4,368 行上官方有 133 个非零点（`0.9696`）**，301 行上的 `1.000000` 是巧合 | 🔴 **已订正**（§2.3 记 `KOMI_SCALE`：V7=20 / V3-V4=15，komi 7.5 ⇒ 0.375） |
| 33 | 全篇 | 源文件都长了几百行，spec 里的 `file:line` **大面积漂移**（`run.txt:700`、`go_rules.py:2479`、`train_sft.py:2269/2396/822…`、`inference.py:501/567/674`、`mcts.py` 的路径） | **已就地改正能确认的**；⚠ 登记在 **§9.12.3**，并给出纪律：**长期引用写符号名、不写行号** |
| 34 | §5（§5.5 / §5.6 / §5.7 全无） | 🔴 **A6 `futurepos` 契约整段缺失**：取值域与「谁是对手」、**哨兵 `-1.0` 的不可兼得论证**、两个 horizon 共用一个 `tform`、**`w['futurepos']` = h0 AND h1 的承重理由**、opt-in + 关闭时逐字节相同、**12.3 GB materialize 绝不能发生在 fork 之后** | 🔴 **已补 §9.12.1**。⚠ 顺带发现：`train_sft.py` **还没调 `warm_futurepos()`** —— 接线是 A6 的剩余工作 |

### 9.11 🔴 实现期新发现的**长期约束**（给后来的改动人）

这一节记的两条都不是「本设计的取舍」，而是**改动本仓库任何东西之前必须知道的硬约束**。

#### 9.11.1 CLI flag 被**整个冻结**：**65** 个（61 + A4 登记的 4 个），加一个就要登记

`tests/test_huber_loss.py::test_no_new_cli_params` 断言的是 **D1「零新增 /
零删除 / 零改名」**，也就是它把 `scripts/train_sft.py` 的 flag
**整个列出来冻结**了。它同时有**第二道**关：全模块扫 `add_argument`，
**调用数必须恰好等于冻结集大小**（65），且**每个首参必须是字面字符串**
—— 堵死 `ap.add_argument(*flags)` / f-string / 循环注册这几种写法。

🔴 **2026-10-02 更新：冻结集已从 61 扩到 65。** A4 的**四个**软标签旗已登记进去：

```
--soft-index            default=None
--soft-weight           type=float  default=1.0
--soft-only-sampling    type=int  choices=[0,1]  default=0
--soft-every            type=_at_least_one  default=1
```

⚠ **`--soft-only-sampling` 用本仓统一的 `type=int, choices=[0,1]` 形态**
（与 `--use-amp` / `--swanlab` 同），**不是 `store_true`** ——
所以判断它必须写 `if args.soft_only_sampling:` 而不是 `is True`。
⚠ **没有 `--soft-labels`**（原 spec §5.5 / §9.11.1 都写过这个名字，是错的）。

同一个文件里的 `test_policy_loss_default_is_ce` 现在钉的是**两件事**：
① `--policy-loss` 的 **default 仍是 `'ce'`**；② choices `== {huber, ce, soft_ce}`。
⚠ 原写的「choices 钉死成 `['huber','ce']`」**已作废** —— 逐字比较被 A4 改掉了。
为什么不弱化成「choices 非空」：那条弱化等于丢掉断言的全部内容
（choices 可以被清空）。现在是**两条分别断言**，覆盖面净增不减
（「多了 soft_ce 之后『默认仍是 ce』反而更重要」—— 它多了一个默认值漂移面）。

⇒ **后果（不变）**：任何「加个 flag」的活都会**先让 CI 红**，
而且红的原因看不出是设计问题。**必须同步登记进那个冻结集**，
否则要么改实现、要么改冻结集 —— 后者要写明理由。
✅ **A4 已按这个流程做完**（`tests/test_soft_cli.py` 28 个 test 函数）。

✅ **`soft_ce` 已接进 CLI**（原写的「没接进 CLI ⇒ 段 2 无法从命令行启动」
**已作废**）。接线方式是**派生**而不是新增旗：
`scripts/train_sft.py::resolve_policy_loss_kind` —— 给了 `--soft-index`
就把**生效口径**派生成 `soft_ce`；没给就原样返回 `--policy-loss`
（段 1 的旧路径数值逐位不变）。两条硬报错：
① 显式 `--policy-loss soft_ce` 而**无** `--soft-index` ⇒ 启动时报错
（否则 `soft_mask` 全 0，软项恒 0，**训练照跑但什么也没学到**）；
② `--policy-loss huber` + `--soft-index` ⇒ 互斥报错
（huber 分支**不消费**软标签，组合起来等于「以为在蒸馏、其实在回归 one-hot 概率」）。

🔴 **但「能启动」不等于「配置对」** —— §5.7 记的那条仍然成立：
分母恒为 `B` ⇒ 不配 `--soft-only-sampling` 就有 **1% 软行 ⇒ policy 项缩小 ~100×**。
**段 2 的正确命令行 = `--soft-index` + `--soft-only-sampling 1`**。

同类约束：日志的三个键名也被 `test_log_keys_unchanged` 钉死。
⚠ 且 `loss` / `policy_loss` / `opt_loss` **键名没变但含义变过**
（`opt_loss = policy_loss + w·value_loss`，是唯一被 `backward` 的量）
⇒ **跨新旧 run 的 loss 曲线不可直接比**，`top1` 仍可比。

#### 9.11.2 ⚠ 两个**文档校验测试已删**，随之失去的保证要自己补

按用户决定删除：

| 已删 | 原本保证什么 | 为什么删 |
|---|---|---|
| `tests/test_run_txt_sync.py`（**整文件**） | run.txt 与 `shell/*.sh`、`run.py`、日志键的**双向齐全**断言 | 它是**旧代**（v18/184 通道）的文档契约，对现役 V7 已无意义 |
| `tests/test_param_budget.py::test_run_txt_records_are_consistent` | run.txt 保留 `# v18 架构参数` 小节并逐项写**实测值** | 它要求 run.txt 保留 v18 小节，而 **v18 已退役**（现役是 **V7 / `KATAGO_SE_CFG` 12ch**）；run.txt 已改成「简洁」的一屏版 |

🔴 **随之失去的保证（记账）**：
**run.txt 里的数字 / flag 与实测值的自动比对没有了。**
这两条当初正是**为了抓「argparse 从 22 个涨到 60 个 flag 而无人同步」**而写的
（同一个坑的两条腿：`test_no_new_cli_params` 还在，`test_run_txt_sync` 已经没了）。

⇒ **补参数时必须人工核对** run.txt。
若日后 run.txt 再次变长、值得机器校验，
**应该校验「run.txt 实际写了哪些数字」而不是「必须写 v18 的数字」**
（`tests/test_param_budget.py:84-85` 已把这条写进注释）。

> 🔴 **它已经咬过一次了（2026-10-02 复核发现）**：`run.txt` 被改成「一屏」版后
> **只剩 55 行**，而 §5.2 引的 `run.txt:700`（4 卡 910A / 2635 行/s 的来源）
> **直接悬空**。§5.2 已就地标注。
> ⇒ 这就是「失去机器校验」的真实代价，不是理论风险。

### 9.12 🔴 **三节**实现期新增的契约（原 spec 完全没有 §9.12.1 / §9.12.2）

#### 9.12.1 A6 · `futurepos` 的完整契约

§4.5 #11 / §5.4 只写了「2 通道 / +8 与 +32」三句话。实现期定下的契约有六条，
**每一条都是「不写下来就会被静默搞错」的那类**：

**① 取值域与「谁」**

`future[b, h, r*bs+c] ∈ {0,1}` = 「点 `(r,c)` 上有**行 i 时该走子那一方的
对手**的子」，`h=0` 取第 `i+8` 行盘面、`h=1` 取第 `i+32` 行盘面
（`FUTUREPOS_OFFSETS = (8, 32)`，8/32 是实测选出的「足够远」的间隔）。
「对手」按 **`to_play[i]`** 定，**不是** `to_play[j]`：i+1、i+2 轮到的正是
`-to_play[i]` ⇒ 两个 horizon 天然共用同一套「谁是对手」口径。
⚠ +8 / +32 **都是偶数** ⇒ 「按 `to_play[j]` 读」在真实数据上**恰好一致**，
所以不主动构造反例的话测试就是**空断言**。这不是可以靠测出来的性质，
只能靠把定义写死。钉住：`test_opponent_is_relative_to_row_i_not_the_future_row`。

**② 🔴 无效行哨兵 = `-1.0`（不是 0）**

```python
FUTUREPOS_SENTINEL = np.float32(-1.0)      # src/data/dataset.py:40
```
**为什么不能是 0**：`0/1` 恰好是「对手一颗子都不占」这个**完全合法的标签**
⇒ 在纯 `{0,1}` 下，「盘面丢了」与「对手没占点」**不可区分且不可逆**
（与 `src/data/pos_hash.py` 警告的「假信号」同一类病）。
⇒ 「取值域 ∈ {0,1}」与「不能静默全 0」这两个要求**在纯 `{0,1}` 下不可兼得**，
必须让哨兵**落在域外**。`-1.0` 同时满足三重保险：
① 不是「无占用」；② 也不是「对方占满」（那是全 1，同样看起来无害）；
③ 任何 `0.5` 阈值的消费方都不会把它判成有效占用。第四道是权重（见 ④）。
⚠ 对应的守卫：`gather_neighbors` 的不可用行被填 **0**，而 0 是「合法空盘」
⇒ **不能单独消费 `g.boards[offset]`**，必须先过 `valid`（越界 + 跨局已合并进
同一个 `valid`），无效行写哨兵。**三种「不可用」都在 `_futurepos_target` 收口**。

**③ 对称增广：两个 horizon 必须走**同一个** `tforms`**

两路 h 是「同一局面的两个不同未来时刻」，所以必须一起镜像。
⚠ 若将来两个 horizon 用了**不同**的 `tforms`，batch 内同一行就不存在
「一个统一的镜像」了 —— `states` 转 90° 而 h1 没转，等于把 h1 当成
「另一个镜像下的未来」在学，**标签与输入不同源且不报任何错**。
⇒ **这条是契约不是实现细节**。实现复用 `permute_soft` 的契约
（`test_permute_future_reuses_permute_soft_contract`）。

**④ 🔴 `w['futurepos']` = h0 AND h1（承重，不是可选）**

`src/networks/katago_v7_loss.py:362-368` 把 `future` reshape 成 `(b,2,bs²)`
之后**压成单个逐样本标量**，再乘**一个**权重；而 `_weighted_mean` 是
`(per_sample * weight).mean()`，**刻意不除 `Σw`**（见 §4.5 的引用）。
⇒ **权重的最小作用单位是「整块 2×bs²」，不是单路。**
所以单路存活时若给整块权重 1，那一路的 `-1` 哨兵会被当真值去拟合 `tanh`
⇒ 头学出一个恒 `−0.76` 的假平面。
⇒ 单路有效性**只能**由 `w['futurepos_h0']` / `w['futurepos_h1']` 单独表达，
**整块权重必须为 0**。钉住：`test_one_surviving_horizon_gives_zero_block_weight`、
`test_both_horizons_dead_gives_zero_everywhere`。

**⑤ opt-in only，且关闭时 payload 逐字节相同**

不调 `attach_futurepos()` ⇒ **不 gather、不加 `w` 的新键**、
`future` 是全 0 占位、`w['futurepos'] = 0`。
⇒ 这不是洁癖：`tests/test_dataset_soft_labels.py:510` 与
`tests/test_prefetch_labels.py:464-466` 这**两条既有测试**在 A6 之前就断言
「`w['futurepos']` 与 `future` 恒 0」—— 它们把 opt-in 逼了出来
（否则 A6 一落地 CI 就红，而红的原因与「A6 坏了」长得一模一样）。
钉住：`test_disabled_is_byte_identical_to_pre_a6_payload`、
`test_labels_false_never_touches_the_gather`、`test_enable_and_disable_toggle_cleanly`。

**⑥ 🔴 12.3 GB 的 materialize 绝不能发生在 fork 之后**

`boards` 落 `.npy` 是 **34,202,713 × 19 × 19 = 12.35 GB**（+ `to_play` 34 MB
+ `ko` 68 MB + `game_ids` 137 MB，合计 ≈ 12.99 GB）。
`sample_batch_numpy` 是在 `_prefetch_worker`（`mp.Process` fork 出来的子进程）里调的
⇒ 惰性解析若发生在 fork 之后，**每个 worker 各解析一次**
⇒ 12.3 GB × worker 数（还可能同时写同一个路径互相踩坏）。
⇒ **`warm_futurepos()` 是父进程 fork 前的入口**，由调用方显式调；
`_futurepos_boards()` 里的惰性解析**只是兜底**（「构造期就开了 futurepos
但忘了 warm」），幂等。
⚠ **现状：这条已经写在 `dataset.py` 的 docstring 里，但 `scripts/train_sft.py`
**还没有调用 `warm_futurepos()`** —— 接线是 A6 的剩余工作。**
（`mp` 的 start method 在本机是 **`spawn`** 不是 fork，但契约不变：spawn 也会
重新构造 dataset 状态，只是不会复制已建的 mmap 句柄。）

#### 9.12.2 B7 · 逐通道的**三档比对资格**（`scripts/crosscheck_stdata.py`）

B7 的对拍工具给**每个通道**打一个 `eligibility` 标签，共三档。
**这不是装饰** —— 不许跳过这一列：

| 档 | 含义 | 后果 |
|---|---|---|
| `ALIGNED` | 本仓与官方**逐位一致**，实测就是 `1.000000` | 可当 oracle；`--min-rate` 门禁**只对这一档断言** |
| `KNOWN_DIVERGENT` | **已知**不一致，且**原因已定位**（不是「还没查出来」） | 门禁**绝不能误杀**它（否则会逼着人去「修」一个不是 bug 的东西） |
| `NOT_COMPARABLE` | 🔴 **根本不可比**：官方那份数据里**缺少重建这个通道所需的输入** ⇒ 比出来的任何数都是**伪造证据** | 门禁不看它；**汇报时也不许报它的对齐率** |

**逐条（不是区间推断 —— 写 `9<=ch<=13` 的话，加通道时会静默继承错误的资格）**：

| 通道 | 资格 | 为什么 |
|---|---|---|
| 空间 `0,1,2,3,4,5,6,8,14,17` | `ALIGNED` | ch1/ch2 是**绝对色**，官方是 pla/opp；`to_play ≡ +1` 时同解 |
| 空间 `7,20,21` | `NOT_COMPARABLE` | 官方**只在 encore 行**置位；`spatial_channels_v7` **没有 `encore_phase` 入参** ⇒ 结构上到不了 |
| 空间 `9,10,11,12,13` | `NOT_COMPARABLE` | stdata 是**逐行独立样本、无着法序列** ⇒ 重建历史 5 手无从下手 |
| 空间 `15,16` | `NOT_COMPARABLE` | 无前一手/二手盘 ⇒ 本仓走**官方回退复制**分支；301 行上的 `0.7276` / `0.7076` **是巧合**（对齐的其实是 ch14） |
| 空间 `18,19` | `KNOWN_DIVERGENT` | 官方算 area 前**先提死子**，本仓 `GoBoard.score()` 不做（§2.5 刻意保留）⇒ **不是 bug** |
| 全局 `0–4, 14` | `NOT_COMPARABLE` | 依赖着法序列（pass 标志 / `passWouldEndPhase`） |
| 全局 `5` | `NOT_COMPARABLE` | 官方那一列含 **draw-jitter**，本仓喂 `game_row.komi` ⇒ 不是同一个量 |
| 全局 `15,16` | `NOT_COMPARABLE` | 本仓无 PDA。🔴 **301 行上官方恰好也恒 0 ⇒ `row_exact = 1.000000` 是巧合**；200 npz / 4,368 行上官方有 **133 个非零点 ⇒ `0.9696`** |
| 全局 `6,7,8,9,10,11,17` | `ALIGNED` | 规则位 → 通道的映射；喂进去的 `rules_flags` 由**官方自己的列**反解 |
| 全局 `12,13` | `ALIGNED` | 分组依据就是官方自己的 ch12/ch13 |
| 全局 `18` | `ALIGNED` | `selfKomi` 由官方 ch5 × 20 反推 ⇒ 精确代入 |

🔴 **两条由此得到的纪律**：

1. **`ALIGNED` 的空间集合 == `tests/test_v7_assemble.py::_PINNED_EXACT`
   = `(0,1,2,3,4,5,6,8,14,17)`** —— 逐个元素相等。
   这不是巧合，是同一个判据的两处落地（一条在测试、一条在工具）。
2. **`cell_agree` 会骗人，而且骗得很厉害**：ch9 的 `row_exact = 0.000000`
   而 `cell_agree = 0.9972`（每行只差 1 个点 / 361）。
   ⇒ 脚本两个都印，但**门禁只看 `row_exact`**，且只对 `ALIGNED` 断言。
   **汇报 ch9 = 99.7% 对齐 = 把一个彻底错的通道说成几乎完美。**

**归档缺失时脚本「报错退出」，而 pytest 里是 `skipif`「跳过」** ——
刻意的差异：工具的存在意义是「回答一个问题」，归档不在就回答不了，
静默退出 0 等于假装成功；测试的职责是验代码、不是保证归档存在。
钉住：`tests/test_crosscheck_tool.py`（用**合成假归档** + **真的**
`spatial_channels_v7` / `global_features_v7`，秒级且装配写错照样红）。

#### 9.12.3 ⚠ 代码行号漂移登记（本 spec 里的 `file:line` 是**快照**，不是契约）

实现期源文件都长了几百行，spec 里的行号引用**大面积漂移**。
**已经就地改正的**（§2.3 `KOMI_SCALE`、§3.5 / §6.2 `sdpa_force_math`、
§5.5 `sample_batch_numpy` 调用点、§5.8 `train_sft.py` 按局切分、
§7.1 `inference.py` 三处、§8.2 Phase 2.4）。**仍然有效、可放心引用的**：

| 引用 | 现状 |
|---|---|
| `build_dataset.py:261-279` / `:297` / `:299-309` / `:314-316` | ✅ 全部仍有效（**本 spec 引用得最准的一批**） |
| `bench_train.py:81/105`、`train_sft_ms.py:101/214` | ✅ 有效 |
| `backbone.py:581`（`def _sdpa`） | ✅ 有效 |
| `src/search/mcts.py:250-253`（`cache_key`） | ✅ 有效（⚠ 路径是 **`src/search/`** 不是 `src/game/`） |
| `go_rules.py:170-177` / `:1446` / `:1551` / `:2261` / `:2269` / `:2420` | ✅ 有效 |
| `kata_v7.py::SCORE_STDEV_SOFTPLUS_BETA = 1.0`（spec 字面值是 `0.05`，已裁决改） | ✅ 有效 |
| `kata_v7_loss.py:195-204`（`_weighted_mean` 不除 `Σw`）/ `:291-293` / `:362-368` | ✅ 有效 |
| `dataset.py:40`（`FUTUREPOS_SENTINEL`）/ `:278-323`（权重契约）/ `:737-742`（`outcome_black`） | ✅ 有效 |
| `train_sft.py:1601-1647`（`soft_cross_entropy`）/ `:767-791`（`resolve_policy_loss_kind`）/ `:2379`（`--soft-only-sampling`）/ `:2805-2816`（收窄训练行） | ✅ 有效 |
| `inference.py:177`（`register_in_channels_builder(22, ...)`） | ✅ 有效 |

⇒ **纪律**：spec 里写 `file:line` 只在**同一天**有效。要长期引用，
写**符号名**（`train_sft.py::resolve_policy_loss_kind`）而不是行号 ——
本 spec 里已经有不少 `:func:` / `::` 形式的引用，那些不会漂。
