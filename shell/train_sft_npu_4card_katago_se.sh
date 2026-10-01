#!/usr/bin/env bash
# SFT 四卡 910A —— KataGo SE-bottleneck 档（DDP / HCCL）
#
# 本脚本取代 shell/train_sft_npu_4card_v21.sh（v21 整体退役，连同它的 17 通道
# Mamba-2/Transformer/CrossAttn 块栈）。除「网络结构」与随之而来的存档名外，
# torchrun 写法、FP16+GradScaler、预取、C2NET、导出 ONNX 与 v19 档完全一致。
#
# 用法:
#   python run.py --sh shell/train_sft_npu_4card_katago_se.sh
#   python run.py --sh shell/train_sft_npu_4card_katago_se.sh --epochs 2
#
# 末尾的 "$@" 让追加参数覆盖默认值（argparse 取最后一个）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- SwanLab（可选依赖）----
# 云端环境不保证已装 swanlab。装不上也不中断训练（set -e 下用 || 兜底）。
python -m pip install swanlab -q || \
  echo "[warn] swanlab 安装失败（多为无外网），将仅用 stdout 记录指标"

# ---- 训练结构：唯一真相源是 scripts/train_sft.py 的 KATAGO_SE_CFG ----
# 结构参数**不再**由命令行决定（CLI 结构 flag 全部归档、不参与建网）。下面
# KATAGO_SE_CFG 那一段仍然照传这些 flag，是为了 CLI 契约、run.py 追加参数覆盖，
# 以及守护测试（tests/test_run_py_sh.py 会解析整条命令行并按 flag 推算显存）；
# **它们的取值已经对齐新架构的形状**，但真正决定建网的是那张表。
#
# 实测参数量（sum(p.numel())，19×19 / action_size=362 / 12 通道）：
#     主干 8,392,995 + 头 719,010 = 全网 9,112,005
#   主干 = stem 26,400 + 13 × SEBottleneck(240) 195,375
#          + 4 × AttentionResBlock(240) 1,442,160 + out 58,080
#   头   = ValueNetwork(96 宽 / 2 残差块) 540,193 + PolicyNetwork(128 宽 / 3 层)
#          178,817
# fp32 权重 36.4 MB、参数张量 226 个（启动时那一次权重广播按这两个数计）。
#
# ⚠ 显存口径**尚未实测**。9.11M 是按参数量定的宽度，而 240 通道 × 19×19 的激活
# 比 v21 的 184 通道大 1.7 倍。下面 BATCH 段那张表是 v21 时代在 184 通道上量的，
# 直接照搬有 OOM 风险 ⇒ **首跑请用 shell/train_sft_npu_1card.sh 或 _2card.sh 先
# 冒烟**，或者按本脚本末尾「BATCH 怎么调」降档。

# ---- 现行：分布式是 DDP ----
# scripts/train_sft.py 现在用 DistributedDataParallel 包裹模型，构造是
# `DistributedDataParallel(model, device_ids=[local_rank])` —— **除 device_ids
# 之外全部默认**，且**没有新增任何 CLI flag**（依据：
# docs/superpowers/specs/2026-10-01-ddp-instead-of-fsdp-design.md §5.1）。
# 这些不变量由 tests/test_dist_wrap.py 钉住（只传 device_ids；包裹点在 EMA、
# _sync_init_weights_from_rank0、torch.compile 之后，在训练循环之前）。
#
# 读这个脚本时对分布式只需知道四件事，都是 DDP「完整复制 + 每次反向一次
# 梯度 all-reduce」的直接后果：
#   · **参数不分片**：每 rank 持有完整模型（params + grads + Adam 两矩 ≈ 146 MB，
#     按上面 9,112,005 参数算）。所以「卡数越多、每卡模型越省」这个直觉**从来没
#     有成立过** —— 4 卡分的是 batch，不是模型状态。
#   · **每次反向只同步梯度**：1 次 all-reduce（fp32 梯度 ≈ 36.4 MB）+ 1 次 buffer
#     broadcast。⚠ **不是「每个 optimizer step 1 次」**：train_sft.py 全文件没有
#     `no_sync()`，每个 micro-batch 都 backward()，而梯度 all-reduce 挂在**每一次**
#     backward 的收尾上 ⇒ 本脚本 `GRAD_ACCUM=2` 下每个 optimizer step 是
#     **2 次梯度 all-reduce + 2 次 buffer broadcast**（payload ≈ 72.9 MB/步）。
#   · **存档无需汇聚**：每 rank 的 `state_dict()` 本身就是完整权重，键形如
#     `module.backbone.…`，save_model 剥掉 `module.` 之后与**包裹前同形**
#     ⇒ **旧 checkpoint 继续可读**。
#   · **启动方式与 v19 档逐字相同**：`--nproc_per_node=N` 照旧注入
#     RANK / WORLD_SIZE / LOCAL_RANK，train_sft.py 读的是同一套环境变量契约。
#     ⚠ 正因为逐字相同，**别用 torchrun 行去判断当前用的是哪种包裹层**。

# ---- 可调参数 ----
WORLD_SIZE=4
# ⚠ 2026-10-01（BATCH 段的来历，历史）：v21 在 4 卡 / 每卡 2000 时两次 OOM，数字
#   几乎一样：
#       Tried to allocate 2.81 GiB | 20.1 GiB allocated | 27.0 GiB reserved
#   完整账本（从报错直接算出来）：
#       32.00 GiB 总量 = 20.09 活数据(63%) + 6.93 碎片(22%) + 4.36 CANN/HCCL(14%)
#                       + 0.62 空闲(2%) ⇒ 容器可用容量只有 ~27.6 GiB
#   激活量与每卡 batch 成正比，累积步不进激活 ⇒ 降到每卡 1000 + 累积 2，
#   **有效 batch 与 LR 都不变**（1000×4×2 = 8000 ⇒ LR 仍是 0.00637）。
#   降的只是「每卡 batch」这一格。
#
# 曾经被误判为「根因」的，留档以免重犯：
#   1. 「加大 BATCH」或「分布式省显存」能找空间 —— 错。DDP 每 rank 完整复制模型
#      状态（≈ 146 MB，相对 20GB 级激活约 0.7%），要省显存只能动激活 ⇒ 只能动
#      BATCH。
#   2. 「面板 94% + AICore 0% = 预取 worker 继承 CANN 上下文」——错。面板是节点级
#      视图且分母是容器可用容量（27.6 GiB，非 32），AICore 0% 只是采样在算子编译
#      期。预取 fork 顺序仍按安全侧提前了（有运行时护栏），但它不是 94% 的原因。
#
# 每卡 BATCH  累积  有效batch  LR=0.00356×√(有效/2500)
#   1000        2       8000        0.00637   ★本脚本默认（沿用 v21 那一档，未实测）
#   500         4       8000        0.00637   保守档：新结构首跑建议从这里起
# ⚠ 想换更高有效 batch：动 BATCH 与 GRAD_ACCUM 的乘积，**LR 必须同步**
#   （tests/test_run_py_sh.py 的平方根律断言会拦，它已把 ACCUM 计入有效 batch）。
BATCH=1000
GRAD_ACCUM=2          # 有效 batch = BATCH × WORLD_SIZE × GRAD_ACCUM = 8000
LR=0.00637            # 0.00356 × √(8000/2500) 平方根缩放律（累积计入有效 batch）
PREFETCH_W=6          # 每 rank 6 个，4 卡合计 24（对齐 24 核）
PREFETCH_D=16         # 在途 batch 数；每个约 49MB，16→约0.78GB/rank
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_katago_se_npu4.pth
C2NET=1

# GradScaler：910A 走 FP16。从 65536 起步要靠减半搜平衡点（每次溢出白扔一个
# batch），实测本模型平衡于 512~2048。直接给 1024 并关掉回涨，避免震荡偷步。
SCALER_INIT=1024
SCALER_GROWTH=100000

# NPU TorchAir 图编译（受控实验，默认关）。1=开启，失败自动回退 eager。
# **建议先短跑几十步验证再决定是否长训。**
# ⚠ 顺带一条耦合：梯度检查点与图编译**互斥**（P4.6b §8.2④），开成 1 时
#   train_sft.py 会自动把检查点关掉并打 warning（显存回升是预期内）。
NPU_GRAPH_COMPILE=0

# ---- 其它注意 ----
# · 910A 无 BF16 → 代码自动走 FP16 + GradScaler
# · 不传 --compile：NPU 无 inductor，代码自动禁用（D2 只影响 TorchAir 那条路）
#   · 若将来开 D2 的 Linear-only 图编译：它**作用在裸模块上，且发生在 DDP 包裹
#     之前**，这个相对顺序由 tests/test_dist_wrap.py::test_wrapping_happens_
#     after_compile 钉住。
# · 不传 --flash-attn：NPU 上自动禁用，注意力走手写 math
# · NPU 的 pin_memory 关闭 → 训练侧双缓冲 H2D 不生效（那只对 CUDA 有效）
# · 4 卡 DDP 的梯度 all-reduce 仍是每步硬同步，任一卡通信抖动都会拖住全部
#   rank。此前记录过 4×910A 卡间网络不稳，若吞吐骤降或 hang，
#   先用 shell/train_sft_npu_1card.sh 或 _2card.sh 验证通信。
# · 输入通道是 **12**（与 AlphaGoNet 默认、数据集默认、GoAI 注册的 12 通道构建器
#   一致）。17 通道的 v21 模型随构建器一起从 src/inference.py 注销；`n_channels=17`
#   的特征平面路径在 go_rules/数据侧仍保留（白名单 12..17，测试钉住），
#   但已没有模型消费那一档。
# · C2NET：prepare() 与 --data 覆盖**故意**在所有 rank 上执行（每个 rank 都要
#   独立加载数据集，而 --data 是 required=True，非 rank 0 拿不到路径会直接退出）。
#   日志已按 is_main 过滤。若 c2net 的 prepare() 有写盘副作用，正确修法是挪到
#   init_process_group 之后由 rank 0 调用再广播路径。
# · --model / --resume 是 strict load，与旧 12 通道的 resnet/convnext 权重
#   **不兼容**（块类不同）；旧权重的唯一入口是 GoAI 的双代加载。
# · 首跑请与旧 12 通道档的 loss/top1 轨迹对比，确认收敛正常后再长训。
#
# 工具：python scripts/inspect_ckpt.py <ckpt> --emit-flags

torchrun --nproc_per_node="$WORLD_SIZE" scripts/train_sft.py \
  --data "$DATA" \
  --device npu --board-size 19 \
  --backbone-channels 240 --backbone-res-blocks 17 \
  --res-blocks 13 --convnext-blocks 0 --attn-blocks 4 \
  --value-channels 96 --value-res-blocks 2 \
  --policy-channels 128 --policy-layers 3 \
  --batch-size "$BATCH" --epochs 1 \
  --lr "$LR" --weight-decay 0.0001 \
  --attention-mode mix --num-attention-layers 4 --num-heads 4 \
  --attn-mode window_global --attn-window 5 \
  --attention-dropout 0.1 --label-smoothing 0.1 \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --use-amp 1 --use-ema 1 \
  --npu-graph-compile "$NPU_GRAPH_COMPILE" \
  --scaler-init-scale "$SCALER_INIT" --scaler-growth-interval "$SCALER_GROWTH" \
  --prefetch-workers "$PREFETCH_W" --prefetch-depth "$PREFETCH_D" \
  --log-every 50 --swanlab-every 10 --eval-every 500 --save-every 250 \
  --early-stop 1 --early-stop-patience 3 \
  --out "$OUT" \
  --export-onnx 1 --use-checkpoint 1 \
  --swanlab 1 --ver katago_se_npu4 \
  --c2net "$C2NET" \
  "$@"
