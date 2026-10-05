# Handoff — Go-AI v21 (feat/v21-roadmap)

日期：2026-09-27
仓库：`F:\AI\Go-AI`，分支 `feat/v21-roadmap`
上一次全量门：`702 passed, 2 failed, 1 warning`（475s）+ `bash -n shell/*.sh` 全 ok
（warning 是既有的 `test_run_txt_sync.py` GBK 解码，与本轮无关）

---

## 0. 本文件的用途

上一个 agent 中断，5 个并行 subagent 调用**全部被取消/失败**，没有任何 agent 正在运行。
接手者需要：先读 §1 和 §2，再按 §5 的顺序推进。

---

## 1. 当前工作树（未提交，全部是本轮产出）

```
 M scripts/train_sft.py            ← 仅 LF/CRLF 行尾差异，`git diff --ignore-all-space`
                                      为空，内容 == HEAD。无工作丢失（见 §1 末尾）。
 M scripts/webui.py                ← P4.8b
 M src/networks/backbone.py        ← 梯度检查点重构：删除 block 级 GC（GradCheckpointMixin/run_segment/逐 kind 开关/compile 守卫），改用标准 checkpoint_module 包裹各段
 M src/search/mcts.py              ← P4.8b 加只读属性 `MCTS.n_channels`
 M tests/test_attn_dropout_eval.py ← T2
 M tests/test_swanlab_logging.py   ← T1
~~ tests/test_grad_checkpointing.py     ← 已删除（block 级 GC 机制移除，BN guard 机制测试并入 tests/test_se_bottleneck.py）
?? tests/test_webui_rollout_channels.py ← P4.8b 新增，7 测试
?? .codegraph/                           ← 不要提交
```

**每个任务的实现都完成了**，缺的是 report + 提交。

已实现内容速览（读代码为准，此处仅导航）：

- **P4.6b（已重构）** `src/networks/backbone.py`：
  原 block 级 GC（`GradCheckpointMixin` + `run_segment` 的逐 kind 状态机、compile 互斥守卫）
  已**删除**，改用标准 `torch.utils.checkpoint.checkpoint` 直接包裹各段（`checkpoint_module`），
  保留使混合精度 checkpoint 正确的必要辅助：`_autocast_like`（重算恢复精度）+
  `_BatchNormStatGuard`/`_collect_batchnorms`（BN 统计还原，determinism 不污染）。
  开关是模型上的普通 bool 属性 `use_checkpoint`（训练态启用、eval/推理/图编译包裹路径自动关闭）。
  原 GC 专用测试 `test_grad_checkpointing.py` 已删除，BN guard 机制测试并入
  `tests/test_se_bottleneck.py`。
- **P4.8b**：`MCTS.n_channels` property（委托 `_in_channels()`，保留缺失即 raise）；
  `webui.py:1319-1326` 用 `n_channels=session.mcts.n_channels` 重建 `FastPolicy`，
  无 `or 12` 回退；新测试 7 个用 AST/exec 真实 `if args.use_rollout:` 节点验证 17/12 通道。
- **T1**：`test_swanlab_logging.py` 改 house-style AST 定位，**顺带修了同文件另外 4 处**
  同类隐患；注入证据表在 `task-t1-report.md`。
- **T2**：`test_attn_dropout_eval.py` 的文件级 `==4` 计数锁改为按 `Class.method`
  的语义锁，无任何计数断言；注入 4 项转红 + 合法新增通路仍绿（反向检查通过）；
  报告 `task-t2-report.md`。

⚠ T1 的 agent 曾用 `git cat-file HEAD` 还原 `scripts/train_sft.py`，**毁掉过一份未提交改动**
（152059B / 2646 行，哈希 `a4081548…`，内容不可恢复）。现已确认该文件内容 == HEAD
（只剩行尾差异）。**教训：本仓库任何还原动作都禁止用 `git checkout/cat-file/restore/stash`，
必须先 COPY 再从 COPY 还原并 SHA-256 校验。** 需向业主确认那份改动是否真的有工作丢失。

---

## 2. 阻塞全量门的 2 个失败（已诊断，未修）

```
FAILED tests/test_search_arch.py::test_attention_window_affects_memory
  assert mems[3] != mems[19]  →  983608 == 983608
  （attn_window=3/5/19 的实测显存完全相同 ⇒ 计量失灵）

FAILED tests/test_search_arch.py::test_calibration_is_self_consistent
  assert 0.7 < k < 0.95  →  k = 1.3491838252840322
  （标定因子从「checkpoint 高估 ~21%」变成 1.349 ⇒ 显存计量模型整体被抬高）
```

**根因推断（接手者需验证）**：`tests/test_search_arch.py` 自身与
`scripts/search_arch.py` 都**没有被修改**（`git status` 干净），失败是 HEAD 之后的
工作树改动引入的。667→704 的增量 = 新增 37 个测试（30 grad ckpt + 7 webui），
所以这 2 个是**新回归**。最大嫌疑是 **P4.6b 给 `backbone.py` 默认开启的梯度检查点
改变了 `S.measure()` 探测到的 saved-tensor / 显存计数**（训练态 `use_checkpoint` 默认开，
标准 `checkpoint_module` 包裹整段 blocks），而 `search_arch.py` 的显存计量正是拿实际前向/反向的
saved tensors 做估计 —— 这与「标定因子从 0.83 涨到 1.35」方向一致。

验证方法建议：在探针里临时把主干 `use_checkpoint` 置 False（或对应构造参数）
后重跑这 2 个测试。若转绿 ⇒ 根因确认，需要决定：让 `search_arch.py` 的计量显式
关闭检查点（推荐：**计量该反映哪一侧需要业主裁决**），而不是放宽断言区间。

**不要直接把 `0.7 < k < 0.95` 放宽成 0.5~1.5 了事** —— 那是掩盖。

---

## 3. 上轮 5 个 agent 调用全部失败，以下任务**未启动**

| 任务 | 独占文件 | 备注 |
|---|---|---|
| P4.6b report + 实测 | `task-p4-6b-report.md` | 实现已完成，缺报告与实测表（B=1/B=8、分块类型、ResBlock 前提裁决、P4.2 接线契约） |
| P3-C PPO | `scripts/selfplay_train.py` + tests | brief 未写；spec 在 roadmap P3-C 节 |
| P4.6 fused 优化器 | `scripts/train_sft.py` | brief 未写；必须保住 P4.5b 的 `opt_loss`/`log_loss` 解耦 |
| P4.12 文档 | `run.txt` + `README.md` | brief 未写；含 README:316 遗留 `--value-loss-weight 5.0` |
| 搜索测试修复 | `tests/test_search_arch.py` / `scripts/search_arch.py` | 见 §2，已诊断未修 |

失败原因：3 次 `Task cancelled`（外部中断），1 次是同一响应里 `task` 调用
JSON 解析失败（prompt 含未转义换行导致截断）。**下次派发时把多个 `task` 调用
放在同一条消息里重发即可。**

---

## 4. 关键裁决与硬约束（不可回退）

- **架构已定稿**：17 路通道；stem `Conv3×3(17→184)+BN` = 28,520（3×3 是业主裁决，
  7×7 已否决）；块 1-8 ResBlock×8；块 9-12 Mamba-2×4；块 13-14 TransformerBlock×2
  （FFN 184→240→184）；块 15-16 CrossAttnRes×2；backbone **6,558,696**，
  全网 **9,067,443**；`FCPolicyHead` 2,366,730（撤回了 128→120）；
  `FCValueHead` 142,017（保留 Tanh）。
- **Mamba-2**：P=4 head × 46 通道，`N=64`、`d_conv=4`、`dt_rank=4`、`expand=2`
  仅指 `in_proj` 扇出、`A_log` **只有 4 个 per-head scalar**、单块 128,252、
  四块 513,008。生产路径**只有** chunked/SSD 向量化扫描（`MAMBA_CHUNK_SIZE=32`），
  `MambaLTI._sequential_scan_oracle` 是测试专用 oracle，运行时 `scan='auto'`
  分派已删。实测（B=1/2/4/8/32）fwd chunked 93.7/194.5/452.0/1007.5/4075.0 ms vs
  oracle 161.2/260.1/325.1/550.0/1774.9 ms；fwd+bwd 快 10–12×，前向交叉点 B≈3。
  ⚠ 这意味着 **selfplay MCTS 推理（B=1）的前向会比顺序路径慢 1.8×**，
  是已接受的代价。
- **D1**：v21 不加任何 CLI 参数。**D5**：`shell/train_sft_npu_4card.sh` 与
  `shell/train_sft_npu_4card_v19.sh` 禁止修改。
- **Loss 口径（P4.5b，已提交）**：policy 默认 `CE`，value 默认 `Huber`
  （`--value-loss-weight=1.0`、`--huber-beta=0.5`），**无 `--l2-coef`**；
  AdamW `--weight-decay=1e-4` 是唯一实际正则；`compute_l2_report` 只把
  `c‖θ‖²` 加进日志不进 backward；`opt_loss` 与 `log_loss` 分开；日志键
  `loss`/`policy_loss`/`value_loss`。`tests/test_huber_loss.py` 24 个测试钉住。
- **P3.0 已提交**：selfplay 参数 **53→56**（不是路线图说的 34→37，那要等 P3-B
  删掉 19 个 MCTS 参数），新增 `--ppo-clip=0.2`、`--kl-coef=0.01`、
  `--kl-target=0.01`，`--epochs` 语义已改为 PPO epoch。
- **P4.13b 已提交**：`MCTS._in_channels()` 是搜索侧通道数唯一真相源，
  缺失即 raise，无 `or 12`。

### 过程约束

- 每任务：先写 `.superpowers/sdd/2026-09-25-v21-roadmap/task-<id>-brief.md`
  → agent 实现 → `task-<id>-report.md` → 独立 review → **控制器统一 git add/commit**，
  绝不 push。subagent 不跑全量、不 git add、不 git commit。
- **并行时文件所有权必须完全不重叠**；单条命令 timeout ≤ 8 分钟。
- 全量门：`python -m pytest tests -q && bash -n shell/*.sh`。
- 当前 702/704 的计数：667（HEAD 基线）+ 37 新测试（30 grad ckpt + 7 webui）。

---

## 5. 接手者的执行顺序（建议）

1. **修 §2 的 2 个失败**（先验证「检查点默认开启」这个根因假设）。
   这是唯一阻塞提交门的东西。
2. **跑全量门**，然后**按任务分 5 个 commit**：
   `T1` / `T2` / `P4.8b` / `P4.6b 梯度检查点` / `搜索测试修复`。
   （若 §2 的修复动了 `backbone.py`，它并入 `P4.6b` 那个 commit。）
3. **并行派 5 个 agent**（同一条消息里发多个 `task` 调用，避免重蹈 §3 的 JSON 截断）：
   - `task-p4-6b-report.md` — 只写 report，backbone 冻结（发现 bug 只报告不改）
   - P3-C — 独占 `scripts/selfplay_train.py`
   - P4.6 fused — 独占 `scripts/train_sft.py`
   - P4.12 — 独占 `run.txt` + `README.md`
   - （第 5 个可选）P4.7b 编译热点 — 独占 `src/networks/backbone.py`，
     但**必须等 P4.6b report 落地**，否则同文件冲突
     （`task-p4-7b-brief.md` 已写好待派）
4. 之后按依赖串行：**P4.2**（`V21_CFG` + 16 块布局 + 释放
   `selfplay_train.py`/`async_pipeline.py` 两处 `n_channels=12` pin +
   `test_v21_budget`；接线契约等 P4.6b report 给出）→ **P3-D** →
   **P4.4**（export_onnx dummy 12 → `V21_CFG['in_channels']`）→
   **P4.10**（`shell/train_sft_npu_4card_v21.sh`，新文件，不碰 D5 禁改的两个）。
5. 长期阻塞：P4.11 MindSpeed、P5 云端 910A 实跑（需云端/用户提供环境）。

## Suggested skills

- `dispatching-parallel-agents` — §5 第 3 步派发前调用
- `verification-before-completion` — 跑全量门并提交前调用
- `systematic-debugging` — 修 §2 的失败前调用（不要放宽断言区间）
- `receiving-code-review` / `requesting-code-review` — 每个 report 之后
- `writing-plans` — P4.2 动手前

---

## 6. 相关文件索引

- 路线图（spec 真相源）：`docs/superpowers/plans/2026-09-25-v21-roadmap.md`
- ledger：`.superpowers/sdd/2026-09-25-v21-roadmap/progress.md`
- 本轮已存在的 brief：`task-p4-6b-brief.md`（梯度检查点，详细）、
  `task-p4-7b-brief.md`（编译热点，已写好待派）、`task-p4-8b-brief.md`、
  `task-t1-brief.md`、`task-t2-brief.md`
- 已存在的 report：`task-t1-report.md`、`task-t2-report.md`、`task-p4-8b-report.md`
- 关键代码：`src/networks/backbone.py`（P4.6b）、`src/search/mcts.py` +
  `scripts/webui.py`（P4.8b）、`tests/test_search_arch.py`（2 个失败）、
  `scripts/selfplay_train.py`（P3-C 待做）、`scripts/train_sft.py`（P4.6 fused 待做）
- 需业主确认：`scripts/train_sft.py` 那份被误删的未提交改动（§1 末尾）
