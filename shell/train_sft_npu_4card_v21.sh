#!/usr/bin/env bash
# SFT 四卡 910A —— v21：17 通道 Mamba-2/Transformer/CrossAttn 档（FSDP1 / HCCL）
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
#   2. 分布式从 DDP 换成 **FSDP1**（见下面「FSDP1 vs DDP」一节）。
#
# ---- FSDP1 vs DDP：不要按 DDP 的直觉读这个脚本 ----
# scripts/train_sft.py 现在用 FullyShardedDataParallel 包裹模型
# （_wrap_fsdp1），配置是 sharding=SHARD_GRAD_OP、use_orig_params=True、
# auto_wrap 按**块类**切（ResBlock / CrossAttnRes / MambaLTI / TransformerBlock）。
#   ⚠ 这一层耦合到**类名**：_fsdp_wrap_policy 逐个 getattr 取，取不到就**静默跳过**
#     （不报错）。而 ANCHOR_V21 / 路线图已把 MambaLTI 改称 Mamba2（旧称作废）。
#     若 P4.2 落地时把类真改名成 Mamba2 而没同步那个名字元组，4 个 Mamba 块会**不再
#     单独切**，粒度退化成「整模型一个 unit」→ 激活峰值反弹（正是该函数 docstring
#     里点名的失败模式）。改名前请核 `_fsdp_wrap_policy` 的类名元组。
# 与 DDP 的差别，都是正确性/容量问题，不是风格问题：
#   · **参数、梯度、优化器状态三项全部分片**。所以「每卡显存 = DDP 那套算法」
#     的直觉是错的：DDP 只同步梯度，FSDP1 连参数和 Adam 动量都切。
#   · 但**激活不分片** —— 每 rank 仍按自己那份 local batch 常驻全部激活。
#     分片省的是模型状态（见下面 BATCH 段的量级算术：约 0.5%，可忽略），
#     每卡显存的**大头仍然是激活**，所以它仍由 BATCH 决定。
#   · 存档必须在 rank0 汇聚成完整权重（FullStateDictConfig, offload_to_cpu=True,
#     rank0_only=True），否则存出来的是 1/world_size 的碎片。train_sft.py 内部
#     已经处理（_fsdp_full_state_dict），脚本侧无需额外动作。
#   · torchrun 调用方式**没变**：--nproc_per_node=N 照旧注入
#     RANK / WORLD_SIZE / LOCAL_RANK，train_sft.py 读的是同一套环境变量契约。
#     本文件的 torchrun 行与 v19 逐字相同，这不是遗漏。
# run.txt 头部也记了同一件事（"FSDP1：参数+梯度+优化器状态全分片,
# sharding=SHARD_GRAD_OP"），两份文档以代码为准、互为交叉引用。

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
# ⚠ 不要再靠「加大 BATCH」或「FSDP 省显存」找空间：模型状态分片后每 rank 约
#   45MB，相对 20GB 级激活可忽略（0.2%）。
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
# · 不传 --flash-attn：NPU 上自动禁用，注意力走手写 math
# · NPU 的 pin_memory 关闭 → 训练侧双缓冲 H2D 不生效（那只对 CUDA 有效）
# · 4 卡 FSDP1 的 all-gather/reduce-scatter 仍是每步硬同步，任一卡通信抖动都会
#   拖住全部 rank。此前记录过 4×910A 卡间网络不稳，若吞吐骤降或 hang，
#   先用 shell/train_sft_npu_1card.sh 或 _2card.sh 验证通信。
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
