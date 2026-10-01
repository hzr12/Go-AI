#!/usr/bin/env bash
# SFT 四卡 910A —— v21：17 通道 Mamba-2/Transformer/CrossAttn 档（DDP / HCCL）
#
# ⚠ 分布式包裹层换过两次轨，读下面的注释前先看这一行：
#     2026-09-30  DDP → FSDP1（当时模型更大、真实参数量尚未测出）
#     2026-10-01  FSDP1 → DDP（EMA 崩溃 + 规模测算；现行）
#   两段论证都保留在本文件里（历史不删），现行口径见「2026-10-01 的换轨」一节。
#
# 用法:
#   python run.py --sh shell/train_sft_npu_4card_v21.sh
#   python run.py --sh shell/train_sft_npu_4card_v21.sh --epochs 2
#
# 末尾的 "$@" 让追加参数覆盖默认值（argparse 取最后一个）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- SwanLab（可选依赖）----
# 云端环境不保证已装 swanlab。装不上也不中断训练（set -e 下用 || 兜底）。
python -m pip install swanlab -q || \
  echo "[warn] swanlab 安装失败（多为无外网），将仅用 stdout 记录指标"

# ---- ⚠ 与 v19 的关系（先读这一段）----
# v21 只改两件事，其余（torchrun 写法、FP16+GradScaler、预取、C2NET、导出 ONNX）
# 与 shell/train_sft_npu_4card_v19.sh 完全一致：
#   1. 网络结构换成 v21（17 通道，Mamba-2 ×4 + TransformerBlock ×2 +
#      CrossAttnRes ×2 + ResBlock ×8，184 通道），权威参数锚点见
#      scripts/search_arch.py 的 ANCHOR_V21（9,067,443 参数）。
#   2. 分布式包裹层。**历史：2026-09-30 从 v19 的 DDP 换成 FSDP1；2026-10-01 又
#      换回 DDP**（现行 = DistributedDataParallel）。两次换轨的理由分别记在下面
#      「2026-09-30 的决策」与「2026-10-01 的换轨」两节，一节都没删。

# ---- 现行：分布式是 DDP（2026-10-01 换轨后的口径）----
# scripts/train_sft.py 现在用 DistributedDataParallel 包裹模型，构造是
# `DistributedDataParallel(model, device_ids=[local_rank])` —— **除 device_ids
# 之外全部默认**，且**没有新增任何 CLI flag**（依据：
# docs/superpowers/specs/2026-10-01-ddp-instead-of-fsdp-design.md §5.1）。
# 这些不变量由 tests/test_dist_wrap.py 钉住（只传 device_ids；包裹点在 EMA、
# _sync_init_weights_from_rank0、torch.compile 之后，在训练循环之前）。
#
# 读这个脚本时对分布式只需知道四件事，都是 DDP「完整复制 + 一步 all-reduce」的
# 直接后果：
#   · **参数不分片**：每 rank 持有完整模型（params + grads + Adam 两矩 ≈ 145 MB，
#     算术见「2026-10-01 的换轨」一节）。所以「卡数越多、每卡模型越省」这个
#     直觉**从来没有成立过** —— 4 卡分的是 batch，不是模型状态。
#   · **每步只同步梯度**：1 次 all-reduce（约 36 MB/步，fp32 梯度）。
#   · **存档无需汇聚**：每 rank 的 `state_dict()` 本身就是完整权重，键形如
#     `module.backbone.…`，save_model 剥掉 `module.` 之后与**包裹前同形**
#     ⇒ **旧 checkpoint 继续可读，新 checkpoint 的键名与 FSDP1 时代一致**
#     （spec §5.5）。脚本侧无需任何额外动作。
#   · **启动方式与 v19 逐字相同**：`--nproc_per_node=N` 照旧注入
#     RANK / WORLD_SIZE / LOCAL_RANK，train_sft.py 读的是同一套环境变量契约。
#     本文件的 torchrun 行与 v19 逐字相同，这不是遗漏。
#     ⚠ 正因为逐字相同，**别用 torchrun 行去判断当前用的是哪种包裹层**。

# ---- 2026-09-30 的决策：为什么当时选 FSDP1（历史，已退役；保留不删）----
# ⚠ 以下整段描述的是**当时**的代码状态（scripts/train_sft.py 用
#   FullyShardedDataParallel 包裹，即已删除的 _wrap_fsdp1），配置是
#   sharding=SHARD_GRAD_OP、use_orig_params=True、auto_wrap 按**块类**切
#   （ResBlock / CrossAttnRes / MambaLTI / TransformerBlock）。
#   **2026-10-01 已换回 DDP**（理由见下一节）—— 别按这段推算今天的显存，
#   但也别删：它记的是「当时那个决定为什么站得住」。
#   ⚠ 这一层耦合到**类名**：_fsdp_wrap_policy 逐个 getattr 取，取不到就**静默跳过**
#     （不报错）。而 ANCHOR_V21 / 路线图已把 MambaLTI 改称 Mamba2（旧称作废）。
#     若 P4.2 落地时把类真改名成 Mamba2 而没同步那个名字元组，4 个 Mamba 块会**不再
#     单独切**，粒度退化成「整模型一个 unit」→ 激活峰值反弹（正是该函数 docstring
#     里点名的失败模式）。改名前请核 `_fsdp_wrap_policy` 的类名元组。
# 当时的论证是「与 DDP 的差别，都是正确性/容量问题，不是风格问题」：
#   · **参数、梯度、优化器状态三项全部分片**。所以「每卡显存 = DDP 那套算法」
#     的直觉是错的：DDP 只同步梯度，FSDP1 连参数和 Adam 动量都切。
#     ⚠ **2026-10-01 复盘结论：三条逐条检查后发现，它们全是 FSDP 自己制造的
#     问题，不是 DDP 的缺陷** ——
#       (1) use_orig_params=True 是硬需求：否则参数被换成 FlatParameter，
#           EMA 的 shadow 逐参对照与 no_decay 判据 p.ndim==1 都会错位；
#       (2) 分片后 state_dict() 只含本 rank 切片，必须靠 FullStateDictConfig
#           （offload_to_cpu=True, rank0_only=True）在 rank0 汇聚，否则存出来是
#           1/world_size 的碎片（当时由 train_sft.py 的 _fsdp_full_state_dict
#           处理，脚本侧无需额外动作）；
#       (3) optimizer.state 分片后 torch.save 会被 rank0 那份覆盖，必须靠
#           FullOptimStateDictConfig 汇聚。
#     换 DDP 之后这三条的**成因直接消失**（DDP 从不扁平化参数；每 rank 持有完整
#     模型与完整且一致的 optimizer state）。⇒ 原决策在当时是站得住的，它解决的
#     是「FSDP 自带的三个坑」；换 DDP 不是绕过论证，而是让那三个坑不再存在。
#   · 但**激活不分片** —— 每 rank 仍按自己那份 local batch 常驻全部激活。
#     分片省的是模型状态（当时估「约 0.5%，可忽略」；2026-10-01 按真实参数量
#     重算过，见下一节），每卡显存的**大头仍然是激活**，所以它仍由 BATCH 决定。
#     ⚠ **这一条今天依然成立，且是 2000 会 OOM 的根本原因。**
#   · run.txt 头部曾记同一件事（"FSDP1：参数+梯度+优化器状态全分片,
#     sharding=SHARD_GRAD_OP"）—— 那是历史记述，2026-10-01 已同步改成 DDP。
#     两份文档以代码为准、互为交叉引用。

# ---- 2026-10-01 的换轨：FSDP1 → DDP ----
# 触发事件（2026-10-01，云端 4 卡 910A 实测）：SFT 崩在**第一个 step**
#     ema.update() → KeyError: 'backbone.stem_bn._fsdp_wrapped_module.weight'
# 根因（torch 源码层面确认，不是推断）：FSDP1 在
# fully_sharded_data_parallel.py:500 有一句 `self._fsdp_wrapped_module = module`，
# 把原模块注册进了 `_modules`。于是
#   · EMA 构造时（**包裹前**）shadow 的键 = `backbone.stem_bn.weight`
#   · ema.update() 时（**包裹后**，第一个 step）遍历拿到的键 =
#     `backbone.stem_bn._fsdp_wrapped_module.weight`
#   · 而 `_ema_key` 只剥 `_orig_mod.`，不剥这一段 ⇒ **第一个 step 必崩**
# 业主裁决：**全换 DDP**，不采用「只给 `_ema_key` 加一行」的最小修法。
# 换 DDP 后这一类崩溃整类消失（EMA 建的 shadow 键与 update 遍历的键同形）。
#
# 规模测算（这才是决定性理由）：ANCHOR_V21 = 9,067,443 参数（fp32 = 36.27 MB）
#   · DDP 每 rank：params + grads + Adam 两矩 ≈ **145 MB**
#     （4 份 36.27 MB；EMA 的 shadow 是**另一份** 36.27 MB，不在这个口径里）
#   · FSDP1 分片后 ≈ **36.3 MB/rank**
#   · 差额 **+109 MB/rank** = 32 GiB 的 **0.33%**
#   FSDP 的适用区间是「32 GB 卡上 params + grads + optimizer 超过约 20 GB」。
#   本模型 145 MB 离那个门槛差**两个数量级** ⇒ **这个规模就是 DDP 的场景。**
#   ⇒ 换轨前 BATCH 段那套量级算术（分片也只省 0.2~0.5%）方向是对的，但它的
#     结论应该是「分片本来就没帮上忙」，而不是「所以该开 FSDP」。
#
# 通信量：DDP 每 step **1 次** all-reduce（约 36 MB）；FSDP1 是 16 个 wrap 单元
# × 4 rank，每单元前向 1 次 all-gather + 反向 1 次 all-gather + 1 次
# reduce-scatter。
#
# 附带收益（**预期，尚未验证 —— 别当结论引用**）：那 109 MB 分片换来的是分配器
# 里额外的 all-gather 缓存。去掉后 `reserved − allocated` 的差额（实测
# 6.85 GiB，占卡 22%）**应当变小**，但这需要配套的显存记账（上游
# npu-vram-utilization 的 Task 2）才能量化，**目前没有任何数据**。
#
# 换轨后**没有变**的东西（与 BATCH 段直接相关，别混为一谈）：
#   · 每卡显存的**大头仍然是激活**，仍然只由 BATCH 决定 ⇒ 下面那张
#     1000 / 1500 / 2000 的表与「2000 实测 OOM」的结论**全部照旧有效**。
#   · checkpoint 键名（save_model 剥 `module.` 后与 FSDP1 时代一致）。
#   · `broadcast_buffers=True` 是 DDP 与 FSDP1 的**共同默认** ⇒ BN 的
#     running_mean / running_var 跨卡一致这个语义不变，这块通信量持平。
#   · **最大的未知**：DDP 同步路径在 HCCL 上**从未验证过**（本仓库 4 卡一直是
#     FSDP1）。本地测试全绿不构成证据，必须上云冒烟（spec §9 风险表第 1 条）。

# ---- 可调参数 ----
WORLD_SIZE=4
# ⚠ 2026-10-01：BATCH 从 2000 降到 **1000 + GRAD_ACCUM=2**，**有效 batch 与 LR
# 都不变**（1000×4×2 = 8000 ⇒ LR 仍是 0.00637）。降的只是「每卡 batch」。
#
# 为什么必须降（**这一段是云端实测，不是投影**）：
#   4 卡 / 每卡 2000 的两次 OOM 数字几乎一样（修前/修后）：
#       Tried to allocate 2.81 GiB | 20.1 GiB allocated | 27.0 GiB reserved
#   完整账本（从报错直接算出来）：
#       32.00 GiB 总量 = 20.09 活数据(63%) + 6.93 碎片(22%) + 4.36 CANN/HCCL(14%)
#                       + 0.62 空闲(2%) ⇒ 容器可用容量只有 ~27.6 GiB
#   ⇒ 就算碎片全回收也要 24.5 GiB，只剩 3 GiB 余量；步内无法回收碎片 ⇒ 必挂。
#   激活量与每卡 batch 成正比（4096 实验：减半 ⇒ 减半），累积步不进激活。
#
# 三个曾经被误判的「根因」，留档以免重犯：
#   1. MambaLTI 的 drive 整条物化 = 一次 31.67 GiB 申请（真实存在，已修，
#      那个报错消失了）。但它**不是**下面这个 20 GiB 峰值的成因。
#   2. 扫描块内 u/M/h 活到反向：块内检查点确已生效（本地实测 96 次调用全部
#      use_ckpt=True，52.6 → 4.9 GiB/块），但 20.09 GiB 里主角不是它 ⇒
#      CPU fp32 投影低估了约 35%（漏了 autocast 的 fp16 副本、BN 保存的输入
#      平面、NPU 注意力 math 后端瞬时量）。**别再拿 CPU 投影当实测。**
#   3. 「面板 94% + AICore 0% = 预取 worker 继承 CANN 上下文」——**错**。面板是
#      节点级视图且分母是容器可用容量（27.6 GiB，非 32），AICore 0% 只是采样
#      在算子编译期。预取 fork 顺序仍按安全侧提前了（有运行时护栏），但它不是
#      94% 的原因。
#
# 每卡 BATCH  累积  有效batch  活数据(实测口径)  合计(~27.6可用)  LR=0.00356×√(有效/2500)
#   1000        2       8000        ~10 GiB          ~18.4 GiB        0.00637   ★本脚本默认
#   1500        2      12000        ~15 GiB          ~23.5 GiB        0.00771   可试
#   2000        1       8000        ~20 GiB          ~31.4 GiB        0.00637   ✗ OOM
#
# ⚠ 想换更高有效 batch：动 BATCH 与 GRAD_ACCUM 的乘积，**LR 必须同步**
#   （tests/test_run_py_sh.py 的平方根律断言会拦，它已把 ACCUM 计入有效 batch）。
# ⚠ 不要再靠「加大 BATCH」或「分布式省显存」找空间：**换回 DDP（2026-10-01）之后
#   模型状态是每 rank 完整复制的**（params + grads + Adam 两矩 ≈ 145 MB），相对
#   20GB 级激活仍然可忽略（约 0.7%）—— 也就是说这条禁令在换轨前后**同样成立**，
#   只是理由从「分片也只省 0.2%」换成「DDP 连分片都没有，而 145 MB 本来就不算
#   什么」。要省显存只能动激活 ⇒ 只能动 BATCH。
#   （历史：FSDP1 时代这里写的是「模型状态分片后每 rank 约 45MB / 0.2%」；
#   2026-10-01 按 ANCHOR_V21 的 9,067,443 参数重算，分片后是 36.3 MB、
#   差额 +109 MB/rank —— 结论不变。）
BATCH=1000
GRAD_ACCUM=2          # 有效 batch = BATCH × WORLD_SIZE × GRAD_ACCUM = 8000
LR=0.00637            # 0.00356 × √(8000/2500) 平方根缩放律（累积计入有效 batch）
PREFETCH_W=6          # 每 rank 6 个，4 卡合计 24（对齐 24 核）
PREFETCH_D=16         # 在途 batch 数；每个约 49MB，16→约0.78GB/rank
DATA=data/sgf_19x19_full.npz
OUT=models/sft_19x19_v21_npu4.pth
C2NET=1

# GradScaler：910A 走 FP16。从 65536 起步要靠减半搜平衡点（每次溢出白扔一个
# batch），实测本模型平衡于 512~2048。直接给 1024 并关掉回涨，避免震荡偷步。
SCALER_INIT=1024
SCALER_GROWTH=100000

# NPU TorchAir 图编译（受控实验，默认关）。1=开启，失败自动回退 eager。
# D2 之后是 **Linear-only** 图编译（只递归替换 nn.Linear，其余 eager），
# 不是整模型 torch.compile。开启前请注意：常驻显存已 26.7~31.1GB/32GB，
# 图模式的 workspace 可能再炸；且本环境为 torch 2.1.0 / torch_npu 2.1.0.post3 /
# CANN 8.0.RC1（2023 年代），图编译功能成熟度存疑。
# **建议先短跑几十步验证再决定是否长训。**
NPU_GRAPH_COMPILE=0

# ---- 下面这组结构参数是「归档参数」，D1 之后不参与构造 ----
# D1：训练全换 v21，配置写死为 src/networks/alphanet.py 的 V21_CFG 常量，
# **没有 arch=='v21' 分支选择器，也没有新增 CLI 参数**。真正决定 v21 形状的是
# V21_CFG，不是下面这些。所以：
#   · 它们**保留定义**（argparse 里仍在，脚本也照传），是为了 CLI 契约与旧脚本
#     一致、让 run.py 的追加参数能覆盖、也让守护测试能解析整个命令行；
#   · 它们的**取值沿用 v19**，不是在描述 v21。改它们对 v21 没有任何作用。
# 留一句给下一个读到这里的人：**别拿这组数字去推算 v21 的显存或参数量**，
# 那是 ANCHOR_V21 的事（9,067,443 参数 / 16 块 @184ch / in_channels=17）。
# v21 的真实形状一旦与下面这组不一致，是**预期内**的，不是 bug。

# ---- 其它注意 ----
# · 910A 无 BF16 → 代码自动走 FP16 + GradScaler
# · 不传 --compile：NPU 无 inductor，代码自动禁用（D2 只影响 TorchAir 那条路）
#   · 若将来开 D2 的 Linear-only 图编译：它**作用在裸模块上，且发生在 DDP 包裹
#     之前**（train_sft.py 里 torch.compile 在 :2506、DDP 构造在 :2570），这个
#     相对顺序由 tests/test_dist_wrap.py::test_wrapping_happens_after_compile
#     钉住 —— 换轨前后这个顺序**没变**，图编译看到的是未包裹的 nn.Linear。
#   · 顺带一条常被忘的耦合：v21 的检查点开关 = V21_CFG['grad_checkpoint']=1
#     ∧ **未开** --compile / --npu-graph-compile（互斥时关检查点并打 warning）
#     ⇒ 开图编译会同时把块内检查点关掉。
# · 不传 --flash-attn：NPU 上自动禁用，注意力走手写 math
# · NPU 的 pin_memory 关闭 → 训练侧双缓冲 H2D 不生效（那只对 CUDA 有效）
# · 4 卡 DDP 的那 1 次 all-reduce 仍是每步硬同步，任一卡通信抖动都会拖住全部
#   rank。此前记录过 4×910A 卡间网络不稳，若吞吐骤降或 hang，
#   先用 shell/train_sft_npu_1card.sh 或 _2card.sh 验证通信。
#   （FSDP1 时代这里是「16 个单元 × all-gather/reduce-scatter」，同步点更多；
#   2026-10-01 换 DDP 后减到每步 1 次 all-reduce，但**仍然是硬同步**。）
# · 输入通道从 12 变 17（P4.3 补齐 ko 通道）。数据侧必须能出 17 通道，
#   否则数据集构造会与网络 in_channels 对不上。
# · C2NET：prepare() 与 --data 覆盖**故意**在所有 rank 上执行（每个 rank 都要
#   独立加载数据集，而 --data 是 required=True，非 rank 0 拿不到路径会直接退出）。
#   日志已按 is_main 过滤。若 c2net 的 prepare() 有写盘副作用，正确修法是挪到
#   init_process_group 之后由 rank 0 调用再广播路径。
# · --model / --resume 是 strict load，D1 之后与旧 12 通道权重**不兼容**，
#   旧权重的唯一入口是 GoAI 双代加载（4.16）。
# · 首跑请与 v19 的 loss/top1 轨迹对比，确认收敛正常后再长训。
#
# 工具：python scripts/search_arch.py --preset v21   （当前只印参数锚点，无显存投影）
#      python scripts/inspect_ckpt.py <ckpt> --emit-flags

torchrun --nproc_per_node="$WORLD_SIZE" scripts/train_sft.py \
  --data "$DATA" \
  --device npu --board-size 19 \
  --backbone-channels 192 --backbone-res-blocks 17 \
  --res-blocks 4 --convnext-blocks 2 --attn-blocks 5 \
  --value-channels 96 --value-res-blocks 3 \
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
  --swanlab 1 --ver v21_npu4 \
  --c2net "$C2NET" \
  "$@"
