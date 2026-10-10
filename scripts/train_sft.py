"""
    19x19 监督训练脚本（AlphaGoZero 风格策略+价值网络）。

    依赖 scripts/build_dataset.py 产出的紧凑 npz 数据集。监督学习不涉及
    自我对弈，因此绕开了 H1/H2/H4/H5/L1 等自我对弈性能问题。

    用法:
        python scripts/train_sft.py --data data/sft_dataset.npz --device cuda --use-amp \
            --batch-size 512 --epochs 5 --save-every 2000 --out models/sft.pt
    """

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from contextlib import contextmanager, nullcontext

import numpy as np
# CUDA 显存分配器：开启 expandable_segments 减少碎片 —— 能塞下更大 batch
# （直接放大 `--gc-with-compile` 的吞吐收益）、并避免分配器偶发卡顿。
# 必须在任何 CUDA 分配发生前设置（故放在 import torch 之前）。
_os_alloc = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
if 'expandable_segments' not in _os_alloc:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
        (_os_alloc + ',') if _os_alloc else '') + 'expandable_segments:True'
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


def _auto_select_device():
    """自动选择最优训练设备。

    策略（按优先级）：
    1. 探测 CUDA 各卡的空闲显存，挑空闲显存最大的那张卡。
    2. 都不可用则回退 CPU。
    返回形如 'cuda:0' / 'cpu' 的具体设备串。
    """
    def _cuda_free(idx):
        # 同样**不能**用 `torch.cuda.memory_allocated` / `get_device_properties`：
        # 它们会 `_lazy_init()` 建 CUDA context，而本函数在 `--device auto` 时
        # 跑在 `_BatchPrefetcher` fork **之前** ⇒ 预取 worker 继承设备上下文，
        # 护栏会直接拒绝构造。
        # 改走 NVML：同样的「空闲显存」语义、**不建 context**。
        try:
            import subprocess
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total',
                 '--format=csv,noheader,nounits', '--id=%d' % int(idx)],
                capture_output=True, text=True, timeout=20)
            if out.returncode != 0:
                return 0
            used, total = (int(x.strip()) for x in
                           out.stdout.strip().splitlines()[0].split(',')[:2])
            return max(0, total - used)
        except Exception:
            return 0

    best = None  # (free_bytes, backend, idx)
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            free = _cuda_free(i)
            if best is None or free > best[0]:
                best = (free, 'cuda', i)
    if best is None:
        return 'cpu'
    _, backend, idx = best
    return f'{backend}:{idx}'


def _dist_debug_level():
    """当前 `TORCH_DISTRIBUTED_DEBUG` 的取值（大写；未设置返回 ''）。"""
    return (os.environ.get('TORCH_DISTRIBUTED_DEBUG') or '').strip().upper()


def _dist_env_snapshot():
    """通信相关环境快照（失败诊断用；全部是只读查询）。"""
    import torch as _t
    import torch.distributed as dist
    bits = ['torch={}'.format(_t.__version__)]
    try:
        bits.append('可见GPU数={}'.format(_t.cuda.device_count()))
    except Exception:  # noqa: BLE001
        bits.append('设备数=?')
    bits.append('CUDA_VISIBLE_DEVICES={!r}'.format(
        os.environ.get('CUDA_VISIBLE_DEVICES')))
    # `get_rank/get_world_size` 在**未** `init_process_group` 时会抛
    # ValueError。本函数被 `_dist_preflight_check` 的失败分支调用 —— 那里正
    # 需要把环境快照打进消息里，抛异常会把**真正要报的通信错误**顶掉
    # （用户看到的是「未初始化进程组」，而真因是 all_reduce 失败）。
    # 故这里单独兜住，宁可少两个字段也不能让诊断信息自己失败。
    try:
        bits.append('RANK={}'.format(dist.get_rank()))
        bits.append('WORLD_SIZE={}'.format(dist.get_world_size()))
    except Exception:  # noqa: BLE001
        bits.append('RANK=?/WORLD_SIZE=?')
    bits.append('LOCAL_RANK={}'.format(os.environ.get('LOCAL_RANK')))
    bits.append('MASTER_ADDR={}:{}'.format(os.environ.get('MASTER_ADDR'),
                                        os.environ.get('MASTER_PORT')))
    return ' | '.join(bits)


def _dist_preflight_check(backend, device, logger):
    """通信域自检：立刻试**一发** all_reduce，把通信问题从「训练途中」提前到「启动时」。

    为什么必需：NCCL / NCCL 的通信域是**惰性**创建的 —— `init_process_group`
    只登记后端，真正建链发生在**第一发 collective**。不主动试一发，失败就落在
    「第一个 batch 的前向」里，而且报成 NCCL 的通用错误
    （`ProcessGroupNCCL.cpp:64` + `NCCL error`），真因藏在日志更前面的
    `EJ0001 ... Maybe the last training process is running` 里 —— 2026-09-30 的
    4 卡事故就是这样查了 20 分钟才发现是**上一次崩掉的进程留下 NCCL 状态**。

    失败时抛的异常自带：环境快照 + **可执行的处置步骤**（残留进程 / 等待 /
    逐卡复位），而不是让人去猜。
    """
    import torch.distributed as dist
    rank = dist.get_rank()
    world = dist.get_world_size()
    expect = world * (world - 1) / 2.0
    t = torch.ones(1, device=torch.device(device)) * float(rank)
    try:
        dist.all_reduce(t)
        got = float(t.item())
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            '[dist] 通信域自检失败（backend=%s）：%s\n'
            '  环境: %s\n'
            '  最常见真因（2026-09-30 实测）：上一次崩掉的训练进程仍在占用设备，'
            'NCCL 初始化被拒 —— 日志里真正的报错是\n'
            '    EJ0001: Failed to initialize the NCCL process. Reason: '
            'Maybe the last training process is running.\n'
            '  处置（按顺序，别跳步）：\n'
            '    1) ps -ef | grep -E "train_sft|torchrun" | grep -v grep   '
            '# 找残留\n'
            '    2) nvidia-smi info                                       '
            '# Processes 表应为空\n'
            '    3) pkill -f train_sft.py; pkill -f torchrun; sleep 30    '
            '# 通信后端 清理需要时间（报错里的 Solution 是 10s，实测 30s 更稳）\n'
            '    4) 仍失败：nvidia-smi 复位/重载设备         '
            '# 逐卡复位（确认无进程占用）'
            % (backend, e, _dist_env_snapshot())) from e
    if abs(got - expect) > 1e-6:
        raise RuntimeError(
            '[dist] 通信域自检数值不符（backend=%s）：期望 %.1f，实得 %.1f'
            '（all_reduce 结果被污染，常见于设备被别的 rank/进程同时占用）\n'
            '  环境: %s'
            % (backend, expect, got, _dist_env_snapshot()))
    logger.info("[dist] 通信域自检通过 | backend=%s world_size=%d | 环境: %s",
                backend, world, _dist_env_snapshot())


# --------------------------------------------------------------------------- #
# from-scratch 初始权重同步（2026-10-01）
#
# 事故形状：换轨前的分片包裹层 docstring 要求 from-scratch 也同步初始权重
# （「各 rank 独立就会第一步拿到拼错的权重」），但 `sync_module_states` 从未被传
# ⇒ 全文件当时唯一的 `dist.broadcast` 是 stop_flag，**没有任何参数广播**。
# 而 from-scratch 脚本不传 `--resume`/`--model` ⇒ 各 rank 权重从 step 0 就分叉：
# 梯度虽被 all-reduce 拉齐，被拉齐的却是「起点不同」的同一份梯度。
#
# 用显式 broadcast 而非包裹层自带的 `sync_module_states`：后者在 torch 2.1 上
# 需要 `param_init_fn` 配套（本文件参数包裹前已 `.to(device)`），为此还得维护
# 一层按已安装签名过滤 kwarg 的版本兼容（随分片包裹层一并删）。buffer 无需额外同步
# —— BN 的 running_mean/var/num_batches_tracked 是确定性 0/1，各 rank 逐位一致。
# 显式 broadcast 还是**无条件**的：不靠「resume 路径已经同步了」这类推理，且能
# 被 AST 测试钉住顺序 —— 必须排在 EMA 之前（详见包裹点上方的注释）。
# 代价：36.4 MB + 226 次小 collective，只在启动时发生一次。
# --------------------------------------------------------------------------- #

def _dist_active() -> bool:
    """通信域是否已建立**且** world_size > 1。

    这里刻意问的是 PG 本身（`dist.is_initialized()`）而不是 `main()` 里解析的
    `world_size` 环境变量 —— 两者可以不一致（env 写了但 PG 没起），而本组函数
    的正确行为取决于后者。解耦也让测试能用 monkeypatch 精确控制分支。
    """
    try:
        return bool(dist.is_available() and dist.is_initialized()
                    and dist.get_world_size() > 1)
    except Exception:  # noqa: BLE001 — 老版本/异常路径一律按「未建域」处理
        return False


def _sync_init_weights_from_rank0(model, logger):
    """把 rank0 的初始权重广播给其余 rank（from-scratch 起手，见上方注释）。

    `world_size == 1` / 通信域未建时**直接 return** —— 单卡路径逐字不变。
    """
    if not _dist_active():
        return
    n = 0
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
            n += 1
    logger.info("[dist] 初始权重已从 rank0 同步 | tensors=%d | world_size=%d",
                n, dist.get_world_size())


def _assert_init_weights_identical(model, logger):
    """各 rank 初始权重一致性自检（启动时验证，不等训练途中）。

    本地 checksum = 全部参数在**主机侧**以 fp64 求和；`all_gather` 出
    world_size 个 hi/lo 载荷后**逐位**比对。失败即抛，不允许静默继续 ——
    与 `_dist_preflight_check` 同一立场：通信域的问题是惰性的，要主动试一发。

    为什么必须在主机侧归约（2026-10-05，4×旧多卡环境）
    -------------------------------------------
    实测 `rank0=8125.2251, rank1=8125.2256, rank2/rank3=8125.2251` ——
    三个 rank 一致、rank1 差**一个 ulp**。8125.225 落在 `[4096, 8192)`，fp32 在
    这一档的 ulp = `2**(12-23) = 4.88e-4`，实测差 `5.0e-4`，精确对上；这个量级
    排除了 NaN（会打成 `nan`）与「广播没生效」（那会差量级而非末位）。

    根因：原先的 `p.detach().sum(dtype=torch.float32)` 是在**设备上**归约，而
    加速器 的多块 计算单元 归约**分块顺序不保证跨 rank 一致**。同样这 300 多个分片和、
    以不同顺序在 fp32 里累加 ⇒ 末位差 1 ulp；比对用的是精确相等 ⇒ 每次都误报。
    原注释把「相同归约顺序」当成前提，那在 加速器 上是假的（CPU 上为真，所以
    `tests/test_init_weight_sync.py::test_checksum_contract_after_losing_fp64`
    在本地一直是绿的，测不出这个 bug）。

    顺带修掉的第二个缺陷：旧 checksum 是 fp32 跑马和，噪声底 = 总量的 1 ulp
    ≈ 4.9e-4，而单个 1e-2 量级参数差**一个 fp32 ulp** 只有 ~9.3e-10 —— 低 5 个
    数量级，**检测不到**。即这个闸门既误报又漏报。

    为什么是主机 fp64、而不是「设备上换个更准的算法」
    ----------------------------------------------
    * 主机 fp64 的噪声底 ~1e-12（相对 1e-16），比 fp32 的 4.9e-4 好 11 个数量级，
    单个 ulp 的真实分歧重新变得可检测；
    * 求和顺序由 `model.parameters()` 的迭代顺序唯一确定（形状相同 ⇒ 分块相同），
    且 `run.txt` 已固定 `OMP_NUM_THREADS=1`；
    * **加速器 上不能用 fp64**：旧多卡环境 不支持，会派发 专用内核 kernel 且故障是**异步**的
    （2026-10-01 云端实测，见 `tests/test_no_aicpu_ops_in_startup_check.py`）。
    所以 fp64 只发生在主机，过线一律 fp32。

    为什么载荷要拆成 hi/lo 两个 fp32
    ------------------------------
    fp32 只有 24 位尾数，把 fp64 的和直接塞进去会被舍掉 —— 那就等于把噪声底又
    抬回 1 ulp，正是本次误报的机制。拆成 `hi = fp32(acc)`、`lo = fp32(acc - hi)`
    之后，两个 fp32 承载 ~48 位尾数，在 8125 那一档的分辨率 ~2.9e-11，仍比要
    检测的 9.3e-10 低一个数量级 —— 够用。载荷保持 fp32（NCCL 原生支持）。
    """
    if not _dist_active():
        return
    acc = 0.0
    dev = None
    with torch.no_grad():
        for p in model.parameters():
            # `.to('cpu')` 是位精确的 D2H 拷贝；fp64 只在主机上出现，`sum(dtype=)`
            # 在归约过程中升精度，不会把整份参数物化成 fp64。
            acc += float(p.detach().to('cpu').sum(dtype=torch.float64))
            if dev is None:
                dev = p.device
    if dev is None:
        dev = torch.device('cpu')
    hi = float(torch.tensor(acc, dtype=torch.float32).item())
    lo = acc - hi
    # fp32 all_gather：NCCL 原生支持；fp64 会走 专用内核（同上）。
    buf = torch.tensor([hi, lo], dtype=torch.float32, device=dev)
    gathered = [torch.zeros_like(buf) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, buf)
    # 比对也不走 `torch.equal`（它在 旧多卡环境 上是 专用内核 kernel，见下）。
    _vals = [[float(x.item()) for x in g] for g in gathered]
    if any(v != _vals[0] for v in _vals[1:]):
        vals = ', '.join('rank%d=[%s]' % (i, ', '.join('%.17g' % x for x in v))
                        for i, v in enumerate(_vals))
        raise RuntimeError(
            '[dist] 初始权重一致性自检失败：各 rank 的参数 checksum 不一致。\n'
            '  各 rank checksum: %s\n'
            '  两种可能：\n'
            '    1) from-scratch 起手时初始权重确实不同 —— 说明广播没生效或\n'
            '       调用点晚于模型构造，检查 _sync_init_weights_from_rank0 的位置；\n'
            '    2) 初始权重里有 NaN/Inf —— NaN != NaN，会把这一种误报成前一种\n'
            '       （真因通常是某个 zero-init 假设被破坏）。\n'
            '  环境: %s' % (vals, _dist_env_snapshot()))
    logger.info("[dist] 初始权重一致性自检通过 | world_size=%d | checksum=%.17g",
                dist.get_world_size(), _vals[0][0] + _vals[0][1])


def _cuda_name_and_capability(idx=0):
    """读第 `idx` 块卡的型号与算力，**返回 `(name, (major, minor))`**。

    刻意**不**用 `torch.cuda.get_device_properties`：那个调用会 `_lazy_init()`
    建 CUDA primary context。本函数服务于「设备初始化之前」的启动诊断，而后面
    `_BatchPrefetcher` 要 fork 预取 worker，fork 会整份继承父进程的设备上下文
    与显存映射（4 卡 加速器 实测每卡凭空多占 ~24 GiB），它自己有一条硬护栏拒绝在
    设备已初始化后构造。

    走 `nvidia-smi`（NVML）：同样的信息、**不建 context**、也不额外占显存。
    拿不到就返回 `(None, None)` —— 这是纯诊断信息，读不到不该拦住训练。

    `nvidia-smi --query-gpu` 的 `compute_cap` 在驱动不支持时返回 `[N/A]`，解析
    要能扛住这种情况（此时只保留型号，算力给 None）。
    """
    try:
        import subprocess
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_cap',
             '--format=csv,noheader,nounits', '--id=%d' % int(idx)],
            capture_output=True, text=True, timeout=20)
        line = (out.stdout or '').strip().splitlines()
        if out.returncode != 0 or not line:
            return None, None
        parts = [p.strip() for p in line[0].split(',')]
        name = parts[0] if parts else None
        cap = None
        if len(parts) > 1 and parts[1] and not parts[1].startswith('['):
            bits = parts[1].split('.')
            if len(bits) == 2 and all(b.isdigit() for b in bits):
                cap = (int(bits[0]), int(bits[1]))
        return name, cap
    except Exception:  # noqa: BLE001 — 纯诊断，失败就放弃
        return None, None


def _check_training_env(logger):
    """启动时检查三大加速能力并打印诊断：flash-attn 库 / torch.compile / 混合精度。

    纯检测不改变行为；实际的启用决策在设备路径确定后进行（见 main 中
    set_flash_attn / args.compile / amp_dtype 分支）。检测结果会写入日志，
    便于比对云端/本地环境差异。
    """
    # 1) flash-attn 独立库（可选依赖，仅 Ampere+ CUDA 有收益）
    try:
        import flash_attn  # type: ignore[import-not-found]
        fa_status = "已安装 v%s" % getattr(flash_attn, "__version__", "?")
    except Exception:
        fa_status = ("未安装（可选；Ampere+ CUDA 上比内置 SDPA 再快 20-30%%，"
                    "安装: pip install flash-attn --no-build-isolation）")
    # 2) torch.compile（inductor 后端）
    if hasattr(torch, 'compile'):
        try:
            from torch._inductor import config as _ind_cfg  # noqa: F401
            comp_status = "可用 (inductor)"
        except Exception:
            comp_status = "可用"
    else:
        comp_status = "不可用（torch<2.0）"
    # 3) 混合精度（按后端能力）
    if torch.cuda.is_available():
        # ⚠ 不能用 `torch.cuda.get_device_properties`：它会 `_lazy_init()` 建
        #   CUDA context。本函数在 `_BatchPrefetcher` 构造**之前**跑，预取
        #   worker 是 fork 的、会整份继承设备上下文，撞上那里的硬护栏就抛
        #   （2026-10-08 A100 首启才炸，且报错指向的位置不是真因）。改走 NVML。
        name, cap = _cuda_name_and_capability()
        if cap is None:
            amp_status = "CUDA 可用（型号/算力读取失败，不影响训练）"
        elif cap >= (8, 0):
            amp_status = "bf16+fp16 可用（%s, sm_%d%d）" % (name, cap[0], cap[1])
        else:
            amp_status = "仅 fp16（%s, sm_%d%d，Volta/Turing 无 bf16）" % (name, cap[0], cap[1])
    else:
        amp_status = "不支持（CPU 走 FP32）"
    logger.info("[env] flash-attn: %s", fa_status)
    logger.info("[env] torch.compile: %s | 混合精度: %s", comp_status, amp_status)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.networks.alphanet import AlphaGoNet
from src.networks.katago_v7 import NBT_TF_CFG
from src.networks.katago_v7 import _migrate_ema_shadow_qkv
from src.networks.katago_v7_loss import POLICY_SOFT_WEIGHT
from src.data.dataset import SupervisedDataset
from scripts.build_dataset import build


# =========================================================================== #
# 训练结构的**唯一真相源**（结构参数仍然归档在 CLI 之外）
#
# 这里替代（已随 v21 一并删除的）`alphanet.V21_CFG`：形状不再由某个专有主干类
# 决定，而是一张喂给 `AlphaGoNet.__init__` 的 flag 表（arch="se_bottleneck" = KataGo
# v4+ 的 bottleneck + SE 路线）。旧 CLI flag（--arch / --backbone-channels /
# --res-blocks / --policy-layers …）**仍然不参与建网**，下面这张表才是真的。
#
# 参数量是**实测值**（19×19 / action_size=362 / 12 通道 / 随机初始化下数出来的，
# 不是估算）：主干 8,392,995 + 头 719,010 = 全网 9,112,005
#   主干 = stem 26,400 + 13 × SEBottleneck(240) 195,375
#          + 4 × AttentionResBlock(240) 1,442,160 + out 58,080
#   头   = ValueNetwork(96 宽 / 2 残差块) 540,193 + PolicyNetwork(128 宽 / 3 层) 178,817
# 改这张表之后**必须**重数这两个数（启动时也会把实测值打出来，可对账）。
#
# 宽度为什么是 240 而不是 160：`SEBottleneck` 只 195,375 参数（C=160 时 87,050，
# 而同宽度的 `ResBlock` 是 461,440，省下 5.3 倍）⇒ 160 宽 × 17 块只有约 4M，离 9M
# 预算差一半多。这里**保留目标形状的块布局**（17 块 / mix / 4 个注意力 /
# value_res_blocks=2），只把宽度从 160 提到 240 把参数填满。同布局宽度实测对照：
#     C=192 → 6,050,622   C=208 → 6,996,907   C=224 → 8,017,368
#     C=232 → 8,552,392   C=240 → 9,112,005   C=248 → 9,683,909
#
# 显存口径**尚未实测**：9.11M 的宽度是按参数量定的，而 240 通道 × 19×19 的激活
# 比 160 通道大 2.25 倍，旧脚本里的 BATCH=1000 是按 184 通道那档定的。首跑请按
# 报错往下调 BATCH。
# =========================================================================== #
KATAGO_SE_CFG = {
    # ---- 形状（结构，全部对应 AlphaGoNet.__init__ 的参数名）----
    'in_channels': 12,            # 与 AlphaGoNet 默认 / 数据集默认 / GoAI 注册的
                                # 12 通道构建器一致
    'arch': 'se_bottleneck',      # KataGo v4+ bottleneck + SE
    'channels': 240,              # backbone_channels
    'blocks': 17,                 # backbone_res_blocks（段内块数）
    'attention_mode': 'mix',      # 卷积 + 注意力混排
    'num_attention_layers': 4,    # 其中 4 个是 AttentionResBlock
    'num_heads': 4,
    'value_channels': 96,
    'value_res_blocks': 2,
    'policy_channels': 128,
    'policy_layers': 3,
    # ---- 行为（非结构，可被调用方覆盖）----
    'grad_checkpoint': 1,         # 必开；compile=1 时由调用方关掉并记日志
    # ---- 预算锚（实测，硬数）----
    'params_backbone': 8_392_995,
    'params_total': 9_112_005,
}


def build_katago_se_net(*, action_size, attention_dropout=0.1,
                        grad_checkpoint=1, **overrides):
    """按 `KATAGO_SE_CFG` 建网；返回 `(model, cfg)`，`cfg` 是**实际生效**的取值。

    `overrides` 只用于行为参数（见调用点），结构键不接受覆盖 —— 与 D1 的
    「结构只由这一张表决定」一致。实测参数量与 `cfg` 一并返回，好让日志 /
    SwanLab config 面板里出现的数字与这张表对账（而不是各写一份）。
    """
    c = dict(KATAGO_SE_CFG)
    c['attention_dropout'] = float(attention_dropout)
    c['grad_checkpoint'] = int(grad_checkpoint)
    model = AlphaGoNet(
        in_channels=c['in_channels'],
        backbone_channels=c['channels'],
        backbone_res_blocks=c['blocks'],
        attention_mode=c['attention_mode'],
        num_attention_layers=c['num_attention_layers'],
        num_heads=c['num_heads'],
        attention_dropout=c['attention_dropout'],
        value_channels=c['value_channels'],
        value_res_blocks=c['value_res_blocks'],
        policy_channels=c['policy_channels'],
        policy_layers=c['policy_layers'],
        action_size=int(action_size),
        arch=c['arch'],
        use_checkpoint=bool(c['grad_checkpoint']),
    )
    return model, c


# =========================================================================== #
# V7 路径（22 通道 NBT+Transformer）—— **可选**，`--v7 1` 才走，默认关闭
#
# 与 `KATAGO_SE_CFG`（12 通道 / 9.11M）是**并排的第二条路**：`--v7 0`（默认）的
# 建网/标签/损失一行都没动；V7 只在 `--v7 1` 时接管「造特征 + 前向 + 损失」，
# DDP / EMA / 调度器 / 保存 / 评估两条路共用。
#
# 段 1 的四个目标（系数见 `KataGoV7Loss`）：policy #1（1.0，行权重恒 1）、
# π_opp #2（0.15，`w['policy_opp']`）、value #3（1.20，三分类 CE）、
# futurepos #11（0.25 已内嵌，`w['futurepos']`）。
#
# **score 系在段 1 不作主目标，权重默认 0，但结构保留。** 依据：81.09% 的 SGF
# 是认输，只有约 18.5% 有数值分差 ⇒ score/scoring/ownership/sb_center 在段 1
# 绝大多数行上是**占位零值**，拿占位零值当回归目标 = 教网络「分差永远是 0」。
# 段 2/3 接上 sidecar 后整表换掉即可，不动网络、不动 loss。
#
# 为什么用**系数**（`KataGoV7Loss(coeff=...)`）而不是行权重 `w` 关这 8 项：
# 12 项里有两个没有行权重可用 —— #7 `score_stdev` 的行权重是 `game_weight`
# （官方 `col25`，本仓恒 1）；#9 `lead` 的 `w['lead']` 键根本不存在，
# `_weighted_mean` 缺键时返回 `ones`（不是 0）⇒ 都关不掉。系数是唯一能一次关全
# 的杠杆，且是 loss 自带的公开入口，不动 `src/networks/**`。
# =========================================================================== #

#: V7 的输入形状（由 `fillRowV7` 决定，**不从棋局重新推算**）。
V7_SPATIAL_CHANNELS = 22
V7_GLOBAL_CHANNELS = 19
V7_BOARD_SIZE = 19
V7_ACTION_SIZE = V7_BOARD_SIZE * V7_BOARD_SIZE + 1      # 362（含 pass）

#: 段 1 的四个主目标（只列出来给人看；真正的权重在 `KataGoV7Loss.coeff` 里）。
V7_STAGE1_TERMS = ('policy', 'policy_opp', 'value', 'futurepos')

#: score 系：段 1 **权重 0**、段 2/3 打开。逐项列出是为了让「哪几项被关掉了」
#: 可被测试逐项断言，而不是靠「总数是 8」这种会随 loss 演进而失效的口径。
V7_STAGE1_SCORE_TERMS = (
    'ownership',        # #4  w_ownership
    'scorebelief_pdf',  # #5  w_score
    'scorebelief_cdf',  # #6  w_score
    'score_stdev',      # #7  **只 game_weight**，见系数表旁注
    'score_mean',       # #8  w_score
    'lead',             # #9  w_lead（**唯一例外**：`w_of` 缺省是 ones）
    'var_time_left',    # #9b **game_weight**（不是 w_lead，见 loss 旁注实测）
    'scoring',          # #10 w_scoring
    'seki',             # #12 自适应系数，见 w_seki
)


def v7_stage1_loss_weights():
    """段 1 的**有效系数覆盖表**：score 系 0，四个主目标沿用 loss 自带值。

    返回的是**完整**的 `coeff` 覆盖（不是只给 0 的那些），这样测试可以断言
    「四个主目标的权重等于 `LOSS_COEFFS` 的原值」—— 如果只覆盖 0 值项，
    主目标被误改成 0.3 这类改动就不会被任何测试看见。
    """
    from src.networks.katago_v7_loss import LOSS_COEFFS
    out = dict(LOSS_COEFFS)
    for k in V7_STAGE1_SCORE_TERMS:
        out[k] = 0.0
    return out


def build_v7_stage1_loss(action_size=V7_ACTION_SIZE, num_bins=None):
    """段 1 的 12 项 loss 装配：score 系权重 0，四主目标正常。

    **12 项一个不少地照常计算**：被关掉的那 8 项仍然会出现在返回值的 `terms`
    / `weighted` 两个 dict 里（值恒 0）。这是刻意的 —— 段 2/3 只改系数就能开回来，
    而「结构上保留」在测试里是**逐项可断言**的（见 `tests/test_train_sft_v7.py`）。

    `seki` 的自适应因子（`8·0.005/(0.005+EMA)`）在段 1 仍会随 step 更新
    `seki_ema` buffer。它不影响任何数字（系数 0 ⇒ 贡献 0），但**会**让
    checkpoint 里那个 buffer 在段 1 期间跟着占位标签的噪声动。段 2 打开时
    自适应因子从「被占位标签污染过的 EMA」起步，会比从 0 起步保守 —— 这是
    已知的、方向明确（更保守）的偏差，不修。
    """
    from src.networks.katago_v7_loss import KataGoV7Loss
    kw = {} if num_bins is None else {'num_bins': int(num_bins)}
    return KataGoV7Loss(action_size=int(action_size), coeff=v7_stage1_loss_weights(),
                        **kw)


def build_katago_v7_net(*, board_size=V7_BOARD_SIZE, use_checkpoint=1,
                        attn_dropout=None):
    """按 `NBT_TF_CFG` 建 V7 网（已跑 `initialize()`），返回 `(model, cfg)`。

    `cfg` 是**实际生效**的取值（把 `use_checkpoint` / `attn_dropout` 回写进去），
    好让启动日志里的参数量能与 `NBT_TF_CFG` 的预算锚对账 —— 与
    `build_katago_se_net` 的返回契约一致。
    """
    from src.networks.katago_v7 import build_katago_v7_net as _build
    from src.networks.katago_v7 import NBT_TF_CFG
    kw = {}
    if attn_dropout is not None:
        kw['attn_dropout'] = float(attn_dropout)
    model = _build(board_size=int(board_size), use_checkpoint=bool(use_checkpoint),
                **kw)
    cfg = dict(NBT_TF_CFG)
    cfg['use_checkpoint'] = bool(use_checkpoint)
    cfg['attn_dropout'] = float(
        attn_dropout if attn_dropout is not None else NBT_TF_CFG['attn_dropout'])
    return model, cfg


# ---- V7 的特征装配（纯 numpy，可在预取 worker 里跑）------------------------
#
# --------------------------------------------------------------------------- #
# V7 显存预检（2026-10-04 云端 旧多卡环境：batch 3000 ⇒ 64 GB）
# --------------------------------------------------------------------------- #
#: V7 反向时**驻留**的激活，实测值（`saved_tensors_hooks`，B=16、fp32、本机 CPU）。
#: 来源：`tmp` 里的量法 —— 用 `torch.autograd.graph.saved_tensors_hooks` 统计反向
#: 真正被 autograd 存下来的张量字节，这才是决定峰值的那一份。
#: 换算到 fp16（÷2）。
V7_RESIDENT_MB_PER_SAMPLE = {
    # 无 checkpoint：22 层注意力各存一份 (B,H,361,361)
    False: 277.0,
    # block 级 checkpoint：只留 block 输入；本机实测 6.9 MB/样本（40×）
    True: 6.9,
}

#: 有 block checkpoint 时，**重算**单个 block 期间的瞬时峰值（fp16）。
#: 一个 block 含 `num_inner_blocks`=2 个注意力，各 6 块 query 分块 ⇒
#: 2 × (B,4,64,361) fp16 ≈ 0.55 MB/样本。量级远小于驻留量，但决定峰值上界。
V7_TRANSIENT_MB_PER_SAMPLE_FP16 = 1.2


def v7_peak_activation_bytes(batch, *, n_layers, heads, tokens, elem_size=2):
    """V7 每层的注意力矩阵总字节（``(B, heads, T, T)``）。

    这是**无 checkpoint** 口径：每样本每层 ``heads × T × T × elem_size``。

    假设**完整注意力**（V7 的 `MHSA` 确实如此）：若将来给它加了窗口，
    这个估算会**高估**，届时必须改成 ``T_window``。高估的方向是安全的
    （宁可少报 batch 也不要 OOM 一次）。
    """
    return int(batch) * int(n_layers) * int(heads) * int(tokens) ** 2 * int(elem_size)


def _device_total_bytes(device):
    """设备显存总量（字节）；查不到返回 ``None``（宁可不知道，不要瞎猜）。"""
    try:
        t = str(device)
        if t.startswith('cuda'):
            return int(torch.cuda.get_device_properties(
                torch.cuda.current_device()).total_memory)
    except Exception:  # noqa: BLE001 — 查不到就当没有，别因此拦住训练
        return None
    return None


def v7_batch_memory_advice(args, logger, device, *, n_layers, heads, tokens,
                        use_checkpoint=True):
    """按 `--max-gpu-memory` 检查 V7 的 batch；明显超了就**启动前**报错。

    为什么不等到 OOM：真机上 OOM 发生在第一个 batch 反向之后，那时已经白跑了
    数据加载、模型构建、 初始化（实测几分钟），而且 OOM 报的是
    `OutOfMemoryError` 这种**看不出原因**的异常 —— 而原因其实只是一行乘法。

    **checkpoint 感知**：开/关 block 级梯度检查点差 **40×**
    （驻留 6.9 vs 277.1 MB/样本；fp16 下每样本约 4.65 vs 139.7）。
    历史实测（2026-10-04，云端 旧多卡环境，batch 3000 ⇒ 64 GB ≈ 21.3 MB/样本）
    是**补丁前**的数：当时只有 blocks 被 checkpoint，stem 与三个 head
    走裸前向、激活全程驻留，故比「全开」估算的 8.1 MB/样本高 2.6×。
    但 21.3 仍只有「无 checkpoint」基线（≈139.7）的 1/6.6 —— block 级
    checkpoint 在 旧多卡环境 上**确已生效**，旧注释「没真正生效」是过时误读。
    当前代码 stem/heads 也已 checkpoint（并修了重算吃不到 autocast 的
    旧多卡环境 OOM 真因），全开理论应 ≈ 8.1 MB/样本（batch 3000 ≈ 24 GB），
    待 旧多卡环境 真机复测确认。这里仍把**实际配置**对应的每样本字节打出来，
    让人一眼看出处在哪个区间，而不必靠猜。

    只在能查到设备显存时启用。查不到（CPU / 无 GPU 信息）就只打印估算值，
    不拦 —— 本地冒烟不该被这个门控挡住。
    """
    ckpt = bool(use_checkpoint)
    if ckpt:
        # 驻留（fp16）+ 重算瞬时
        per_sample = (V7_RESIDENT_MB_PER_SAMPLE[True] / 2.0
                    + V7_TRANSIENT_MB_PER_SAMPLE_FP16)
        basis = ('block 级 checkpoint 开：驻留 %.1f + 瞬时 %.1f = %.1f MB/样本'
                % (V7_RESIDENT_MB_PER_SAMPLE[True] / 2.0,
                    V7_TRANSIENT_MB_PER_SAMPLE_FP16, per_sample))
    else:
        attn_mb = (n_layers * heads * tokens * tokens * 2) / 1e6
        per_sample = attn_mb + V7_TRANSIENT_MB_PER_SAMPLE_FP16
        basis = ('block 级 checkpoint **关**：注意力 %d 层 × %.3f MB = %.1f MB/样本'
                % (n_layers, attn_mb, attn_mb))
    attn = per_sample * args.batch_size * 1e6
    total = _device_total_bytes(device)
    logger.info("[mem] V7 显存：%d 层 × %d 头 × %d² token | %s"
                " | batch=%d ⇒ 约 %.1f GB",
                n_layers, heads, tokens, basis, args.batch_size, attn / 1e9)
    if not ckpt:
        logger.warning("[mem] block 级 checkpoint 没开 —— 这是 40× 的差距"
                    "（277 vs 6.9 MB/样本）。它由 KATAGO_SE_CFG['grad_checkpoint']"
                    "决定，且与 --compile 互斥。")
    if total is None:
        logger.info("[mem] 查不到设备显存总量 ⇒ 跳过 batch 预检"
                    "（--max-gpu-memory 本次不生效）")
        return
    frac = float(getattr(args, 'max_gpu_memory', 0.9) or 0.0)
    budget = total * (frac if 0.0 < frac <= 1.0 else 0.9)
    ratio = attn / budget
    if ratio > 1.0:
        safe = max(1, int(args.batch_size / ratio))
        raise SystemExit(
            "V7 的 batch 放不进显存：\n"
            "  估算       = %.1f GB（batch=%d，%s）\n"
            "  可用预算   = %.1f GB（设备总量 %.1f GB × --max-gpu-memory %.2f）\n"
            "  ⇒ 需要 --batch-size %d 以下\n"
            " **调小 --attn-window 没用**：V7 的 MHSA 没有窗口参数"
            "（该旗标只对 12 通道路径有意义，而它同样没接进建网）。\n"
            " 也别指望 --gradient-accumulation-steps：它只改有效 batch，"
            "不降**每卡**的瞬时显存。\n"
            "  ⇒ 要更大的有效 batch，用 --gradient-accumulation-steps 配合"
            "更小的 --batch-size。"
            % (attn / 1e9, args.batch_size, basis,
            budget / 1e9, total / 1e9, frac, safe))
    if ratio > 0.8:
        logger.warning("[mem] batch=%d 的估算已达预算的 %.0f%%"
                    "（%.1f / %.1f GB）—— 非注意力部分与碎片可能让它 OOM，"
                    "建议 --batch-size %d 以下",
                    args.batch_size, 100 * ratio, attn / 1e9, budget / 1e9,
                    max(1, int(args.batch_size / ratio)))


# --------------------------------------------------------------------------- #
# 局级 sidecar（spec §5.3）→ V7Dataset 的逐局贴目
# --------------------------------------------------------------------------- #
def _sidecar_kwargs(dataset, games_npz):
    """`games.npz` → `V7Dataset(game_komi=..., game_rules_flags=...)` 的 kwargs。

    没给 `--games-npz` ⇒ 返回 ``{}`` ⇒ 构造函数两个参数都留 `None` ⇒
    `game_row_for()` 返回 `None` ⇒ 贴目按 0（**逐位等于本函数接入之前**，
    所以这个旗默认关闭时不会动任何已有数值）。
    """
    if not games_npz:
        return {}
    komi, rules, diag = load_games_sidecar(games_npz, dataset)
    print('[sidecar] %s' % diag['path'])
    print('[sidecar] 局数 %d（sidecar）/ %d（数据集）| 有贴目 %d（缺 %d）| '
        '有分差 %d | 认输 %d'
        % (diag['sidecar_games'], diag['dataset_games'],
            diag['has_komi_games'], diag['missing_komi_games'],
            diag['has_score_games'], diag['resign_games']))
    if diag['nondefault_rules']:
        # 如实报「接不进」而不是悄悄按简单局算
        print('[sidecar] %d 局的 g_rules 非默认，但**逐局规则化尚未接进特征**'
            '（官方 calculateArea 只吃标量 rules_flags）⇒ 这些局的空间特征仍按'
            '简单局算。这是已知缺口，不是本次接入能解决的。'
            % diag['nondefault_rules'])
    return {'game_komi': komi, 'game_rules_flags': rules}


def load_games_sidecar(path, dataset):
    """读 `games.npz`，按 **game_id** 对齐，返回 ``(game_komi, game_rules, 诊断)``。

    口径（spec §5.3）：sidecar 是**局级**表，左索引是 `game_id`，
    读法 ``sidecar[game_ids[idxs]]`` —— 不是按行号。搞错的后果是**静默取到
    别局的贴目**：全局 ch5/ch18 跟着错，loss 照降，不报任何错。

    Args:
        path: ``games.npz`` 路径。
        dataset: 已建好的 board 级数据集（只用它的 ``game_ids``）。

    Returns:
        ``(komi_1d, rules_1d, diag)``；``komi_1d`` / ``rules_1d`` 是按 game_id
        索引的一维数组（可直接交给 ``V7Dataset(game_komi=...)``）。

    Raises:
        SystemExit: sidecar 短于 game_id 空间、或缺必需键。**这两种都必须
            当场报错** —— 短了只能靠 ``.get(g, 0.0)`` 补 0，那等于给部分局
            编了一个「贴目 0」的真值。
    """
    need = ('g_komi', 'g_rules', 'g_score', 'g_resign', 'g_re', 'g_resign_side')
    z = np.load(path, allow_pickle=False)
    try:
        missing = [k for k in need if k not in z.files]
        if missing:
            raise SystemExit(
                f'{path} 缺键 {missing}。\n'
                f'  契约见 scripts/build_games_sidecar.py 的 savez 与 '
                f'tests/test_games_sidecar_alignment.py::test_sidecar_keys。'
                f'\n 别「缺哪个补哪个」：键名对不上说明这不是本脚本的产物，'
                f'硬凑只会得到**逐位错位**的标签。')
        komi = z['g_komi'].astype(np.float32)
        rules = z['g_rules'].astype(np.int64)
        score = z['g_score'].astype(np.float32)
        resign = np.asarray(z['g_resign'], bool)
        G = int(komi.shape[0])
    finally:
        z.close()

    gid = np.asarray(dataset.game_ids)
    if gid.size == 0:
        raise SystemExit(f'{path} 无从对齐：数据集没有 game_ids。')
    need_n = int(gid.max()) + 1
    if G < need_n:
        raise SystemExit(
            f'sidecar 只有 {G} 局，但数据集的 game_id 最大到 {need_n - 1}'
            f'（需要 >= {need_n} 局）。\n'
            f' 短了只能给缺失的局补「贴目 0」，那不是缺失标记而是**一个假真值**'
            f'（全局 ch5 变 0、ch18 三角波走偏，且不报任何错）。\n'
            f'  常见原因：sidecar 是旧主 npz 扫的，而数据集已 rebuild —— '
            f'重跑 scripts/build_games_sidecar.py。')

    # 「有真贴目」的判定用 sidecar 自己的约定：`g_komi == 0` ⇔ 缺失
    # （sgf_parser.py:218 的 has_komi 标志；不能拿 `komi != 7.5` 当判据）。
    has_komi = komi[:need_n] != 0.0
    n_game = int(np.unique(gid).size)
    diag = {
        'path': os.path.abspath(path),
        'sidecar_games': G,
        'needed_games': need_n,
        'dataset_games': n_game,
        'has_komi_games': int(has_komi.sum()),
        'missing_komi_games': int(need_n - has_komi.sum()),
        'has_score_games': int(np.isfinite(score[:need_n]).sum()),
        'resign_games': int(resign[:need_n].sum()),
        'nondefault_rules': int((rules[:need_n] != 0).sum()),
    }
    return komi[:need_n], rules[:need_n], diag


# 为什么需要一个**新的**装配函数而不是复用 `dataset.sample_batch_numpy`
# ----------------------------------------------------------------------
# `sample_batch_numpy` 返回的是 `feature_planes_batched` 的 12 通道路径
# （`SupervisedDataset.n_channels` 的产物），而 V7 要的是
# `feature_v7.spatial_channels_v7` 的 22 通道 + `global_features_v7` 的 19 维。
# 两者**通道语义不同**（ch1/ch2 在 V7 里按「对方/自己」固定着色，与 12 通道的
# to_play 相对口径不一样），所以 V7 必须自己造特征。
#
# **邻行 gather 是 V7 独有的成本**：ch15/ch16 需要 `i-1` / `i-2` 两个历史盘面，
#   它们不在 npz 里。走 `feature_v7_gather.gather_neighbors` 一次取回（每个偏移
#   一次 fancy-index，不随 B 逐行循环），跨局由 `game_ids` 守卫。
#
# **batch 不得为迁就特征侧而设**：特征侧对 batch 几乎不敏感（32→311.4 /
#   128→309.7 / 1024→307.1 行/s，离散度 1.7%），成本在 `iterLadders` 逐行 DFS 而不是
#   batch 固定开销 ⇒ batch 按显存选即可。


def _v7_row_gather(dataset, idxs, boards):
    """一次 gather 取回 ``i-2`` / ``i-1`` 两组邻行（纯 numpy）。

    **不把 offset 0 一起 gather**：`gather_neighbors` 明确拒绝 0（「邻行
    gather 存在的意义就是取**别的**行」），当前盘面直接从 `boards[idxs]` 取。
    两条路取的是**同一个**数组对象上的 fancy-index，所以口径一致。
    """
    from src.data.feature_v7_gather import gather_neighbors
    return gather_neighbors(boards, idxs, offsets=(-2, -1),
                            game_ids=dataset.game_ids, to_play=dataset.to_play,
                            ko=dataset.ko)


def v7_batch_features(dataset, idxs, *, boards=None, rules_flags=0):
    """按 dataset 的行号造出 V7 的两份输入。

    Returns:
        ``(spatial, global_features)`` = ``(B,22,19,19) fp16`` / ``(B,19) fp16``。
        dtype 与 `GoBoard.feature_planes_batched` / `feature_v7.py` 一致（fp16），
        AMP 下不必再升精度。

    **对称增强不在这里做**：8 路 dihedral 变换由调用方（预取 worker 或
    `v7_batch_sync`）施加，且**必须与标签用的同一份 `tforms`**（标签侧的
    `next_move` / `future` 会被那一份同步重映射；输入与标签不同源是「loss 照降、
    棋力不涨」的经典静默错）。19 维全局量在棋盘翻转下不变（与 `_build_labels`
    docstring 的口径一致），故只有 22 通道空间量要变换。

    `game_row=None` ⇒ 贴目按 0 处理（`feature_v7._komi_of` 的 NaN 分支口径）：
    主数据集 npz **没有**贴目列，SGF 的 `KM` 在 D0 sidecar（B 组）里。
    后果是全局 ch5（`selfKomi/20`）恒 0、全局 ch18 的 parity 相位按 0 算 ——
    这两维在段 1 拿不到真值，是已知缺口，不在这里假装它有。
    """
    from src.data.feature_v7 import spatial_channels_v7, global_features_v7
    idxs = np.asarray(idxs, dtype=np.int64)

    # `V7PackedDataset` 已经把 22 通道**预算好**并位打包存了（`spatial_packed`）。
    #   那条路上 `boards` / `my_hist` 这些字段根本不存在，也**不该**重算 ——
    #   重算要再做一遍 Benson/seki/劫争，CPU 上比读盘贵一个量级，而且可能与
    #   转换时用的代码版本漂移（那就成了静默的特征错位）。
    if hasattr(dataset, 'sample_spatial'):
        return dataset.sample_spatial(idxs), dataset.sample_global(idxs)

    src = dataset.boards if boards is None else boards
    g = _v7_row_gather(dataset, idxs, src)
    my = np.asarray(dataset.my_hist)[idxs]
    op = np.asarray(dataset.op_hist)[idxs]
    tp = np.asarray(dataset.to_play)[idxs]
    ko = np.asarray(dataset.ko)[idxs]
    b_now = np.asarray(src)[idxs]
    spatial = spatial_channels_v7(
        b_now, tp, ko, my, op,
        prev_board=g.boards[-1], prev_prev_board=g.boards[-2],
        rules_flags=rules_flags,
        prev_valid=g.valid[-1], prev_prev_valid=g.valid[-2])
    gl = global_features_v7(
        None, my, op,
        prev_board=g.boards[-1], prev_prev_board=g.boards[-2],
        rules_flags=rules_flags, to_play=tp,
        board_area=V7_BOARD_SIZE * V7_BOARD_SIZE)
    return spatial, gl


def _dense_move_target(moves, action_size):
    """``(B,)`` 行号 → ``(B, action_size)`` 稠密 one-hot；**-1 ⇒ 全零行**。

    `-1` 不是「随便哪个类」：dataset 的 `next_move` 用 -1 表示「局末手 /
    下一行不是同一局 / 下一手是 pass」，语义是**没有下一手**，对应
    `w['policy_opp'] == 0`。若交给 `F.one_hot(-1)`，PyTorch 的行为不是本仓
    可以依赖的契约（不同版本不同），所以这里**显式**置零 —— 权重本来就是 0，
    全零行既安全又语义正确。

    **设备无关：不许把设备张量送进 numpy**（2026-10-04 云端 旧多卡环境 实测崩）
    ------
    调用方传的是 `move_t`，而它是 `torch.from_numpy(...).to(device)` 的结果
    ⇒ 在 加速器 上是 **设备索引 设备张量**，而 `np.asarray()` 对它抛：

        TypeError: can't convert 设备索引 device type tensor to numpy.
        Use Tensor.cpu() to copy the tensor to host memory first.

    CPU 上恰好能跑（numpy 消费 CPU 张量的 `__array__`）⇒ **本机 CPU 冒烟
    永远发现不了这个 bug**，只有真机 加速器 才会炸。这也是它此前一直没被
    发现的原因，不是「别人都写对了」。

    修法是**按类型分派**而不是无条件 `.cpu()`：`.cpu()` 能让 numpy 转换成功，
    但那样 one-hot 目标会被搬回主机、再在 loss 里搬回设备 ⇒ 每步多两次
    H2D/D2H（`(B, 362)` × 2 项）。直接在设备上 `one_hot` 既修好崩溃，又省掉搬运。
    """
    if isinstance(moves, torch.Tensor):
        # `.detach()`：这一项参与 loss 的图，梯度不经过 one-hot（标签是常量）
        mv = moves.detach().reshape(-1).to(torch.long)
    else:
        mv = torch.as_tensor(np.asarray(moves)).reshape(-1).to(torch.long)
    valid = (mv >= 0) & (mv < int(action_size))
    safe = torch.where(valid, mv, torch.zeros_like(mv))
    out = F.one_hot(safe, int(action_size)).to(torch.float32)
    out[~valid] = 0.0
    return out


def v7_loss_labels(labels_dict, moves, *, action_size=V7_ACTION_SIZE):
    """dataset 的 `labels_dict` → `KataGoV7Loss.forward(out, labels)` 的键集合。

    这是 `src/data/dataset.py::_build_labels` 与 `src/networks/katago_v7_loss.py`
    之间**唯一**的翻译层。两边都不能改（本仓的硬约束），所以差异全部收在这里：

    ==========================  ==============================================
    loss 要的键                 来源
    ==========================  ==============================================
    ``policy_player``           由**本行着法** `moves` 造 one-hot
    ``policy_opp``              由 `labels_dict['next_move']` 造 one-hot
                                （-1 ⇒ 全零行，见 `_dense_move_target`）
    ``outcome``                 直取（**to_play 视角**，三分类 {0胜,1负,2无结果}）
    ``outcome_black``           直取但**不参与** loss（黑方视角，保留给日志/段 3）
    ``ownership`` / ``score`` / 直取（段 1 是占位零值，权重 0）
    ``scoring`` / ``seki`` /
    ``sb_center`` / ``sb_upper``
    ``game_weight``             直取
    ``w``                       直取（**含** `futurepos` / `policy_opp` 等行权重）
    ``futurepos``               **由 `future` 改名**（loss 叫 `futurepos`，
                                dataset 叫 `future`；值域与 -1 哨兵完全一致）
    ==========================  ==============================================

    **`w['futurepos']` 绝不能改成 OR**（这是承重语义，不是风格）：
    loss 的 #11 把 `future` reshape 成 `(b,2,bs²)` 之后**塌成逐样本标量**、
    再乘**一个**权重，而 `_weighted_mean` 是 `(per_sample*weight).mean()`
    （**刻意不除 Σw**）⇒ 权重的最小作用单位是「整块 2×bs²」。
    只活一路时若给 1，那一路的 -1 哨兵会被当真值去拟合 tanh，头会学出一个
    恒 −0.76 的假平面。dataset 给的**与**语义（`w_h0 & w_h1`）是唯一正确的
    口径，本函数**原样透传**，不改。

    **`outcome_black` 不能由 `outcome * to_play` 推出**：`0 * -1 == 0`，
    于是「白胜」会被报成「黑胜」。直接用 dataset 给的键。
    """
    lbl = dict(labels_dict)
    lbl['policy_player'] = _dense_move_target(moves, action_size)
    # `policy_opp` 必须用**对手侧**的着法。`V7PackedDataset` 会给
    #   `next_move_opp`（来自 `policy_opp_rank[:,0]`）；缺席时才退回 `next_move`
    #   —— 那条退路是 board 级路径的旧行为，而在 stdata 上会让 #1 与 #2 两个
    #   loss 项拿到**同一个**目标（π_opp 白训）。
    lbl['policy_opp'] = _dense_move_target(
        labels_dict.get('next_move_opp', lbl['next_move']), action_size)
    lbl['futurepos'] = lbl['future']
    return lbl


def _rebind_futurepos_mmap(dataset):
    """ **spawn 语义下重新打开 boards 的映射，而不是用传进来的那份数据。**

    为什么这是必须的（**实测**结论，不是推测）
    ----------------------------------------
    `mp.Process` 在 Linux 上默认 fork（地址空间整份继承），在 **Windows 上是
    spawn**。spawn 不继承内存：子进程拿到的是参数 **pickle 之后**的副本。而本机
    （numpy 1.26.4 / CPython 3.11）实测：**pickle 一个 `np.memmap` 会把整份数据
    复制进 payload**，往返后的对象丢掉映射信息（`filename` / `offset` 都是
    `None`），即它不再是「按需分页的映射」。
    ⇒ 主数据集的 boards 走 spawn 通道就是 **12.3 GB × worker 数**，症状是 OOM
    或者「每个 worker 各持一份、互相不一致」，**都不会**报「句柄不继承」。
    `tests/test_train_sft_v7.py::test_pickling_a_memmap_materializes_its_
    whole_payload` 把这个前提钉死（payload 字节数 ≥ 数组字节数、映射信息丢失）。

    为什么**重新打开**而不是**继承**
    ------------------------------
    重开后每个 worker 各持一个独立的按需分页映射，页缓存由 OS 在进程间共享，
    内存占用是「被点到的页」而不是「12.3 GB」。
    `dataset._futurepos_boards` 末尾那句 `np.asarray(boards)` 会把 memmap 降级成
    `ndarray` **视图**（共享同一块映射、不复制），所以判「是否还挂着映射」要看
    base 链上有没有 memmap，**不能**判 `isinstance(x, np.memmap)`。

    **绝不在 worker 里走会落盘的那条分支**
    那会把 12.3 GB × worker 数真正写出来（还可能几个进程同时写同一个路径互相
    踩坏）。本函数只把**父进程已经落好的 `.npy` 路径**重新 `np.load` 一次 ——
    那个路径由 `attach_futurepos(source=<.npy>)` 直接给出，或从
    `fp['materialized_paths']` 取回（那只是几个**字符串**的字典，穿过 pickle
    的代价可忽略）。见 `_futurepos_mmap_path`。

    **它依赖父进程已经 warm 过**：只有 warm 过，`materialized_paths` 才被填上，
    worker 才知道该重开哪个文件。`main()` 里 `warm_futurepos()` 因此排在
    `_BatchPrefetcher(...)` **之前**（顺序有测试钉）。

    走的是**公开 API**（`attach_futurepos(source=...)` + `warm_futurepos()`），
    没有伸手进 `dataset.py` 的私有字段：上一个进程留下的 `fp['boards']` 缓存被
    新的 `attach_futurepos` 直接重置（它整体重建 `self._fp`），所以不需要手动
    清缓存，也就不会出现「清了一半」的中间态。

    返回可 mmap 的 boards 数组；**没有可重开的路径时返回 `None`**，调用方退回
    `dataset.boards`（fork 下正确；spawn 下那是**数据集本身**的性质 —— 列本来就在
    内存里、根本不是映射 —— 不是本函数能修的）。
    """
    path = _futurepos_mmap_path(dataset)
    if path is None:
        return None
    dataset.attach_futurepos(source=path, mode='live')
    dataset.warm_futurepos()
    fp = getattr(dataset, '_fp', None)
    return None if not isinstance(fp, dict) else fp.get('boards')


def _futurepos_mmap_path(dataset):
    """从（已被 pickl 过来的）futurepos 配置里取「可以直接 ``np.load`` 的 .npy」。

    两条命中路径：
    1. `fp['materialized_paths']['boards']` —— 父进程跑过 `materialize_dataset`
        留下的产物路径（只有字符串 ⇒ 能安全穿过 spawn 的 pickle）；
    2. `fp['source']` 是路径 —— `attach_futurepos(source=<.npy>)` 的直接形态。

    返回 `None` 的两种情况：futurepos 没启用（`fp is None` / `mode != 'live'`）、
    或来源是**数组**（那已经在 pickle 里整份传过来了，再开一次没有意义）。
    """
    fp = getattr(dataset, '_fp', None)
    if not isinstance(fp, dict) or fp.get('mode') != 'live':
        return None
    paths = fp.get('materialized_paths')
    if isinstance(paths, dict) and paths.get('boards'):
        return os.fspath(paths['boards'])
    src = fp.get('source')
    if src is None or isinstance(src, (str, bytes, os.PathLike)):
        return None if src is None else os.fspath(src)
    return None


def _plain_state_dict(model):
    """`model.state_dict()` 剥掉 `module.` 与 `_orig_mod.` 两种前缀。

    单独抽出来是因为**同步保存与后台写盘共用它** —— 两边必须剥出**同一套**键名，
    否则「异步快照」和「收尾的同步保存」会写出两种布局的 checkpoint，
    `--resume` 挑到哪个都可能是坏的。
    """
    sd = model.state_dict()
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    if any('_orig_mod.' in k for k in sd.keys()):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    return sd


def save_model(model, path):
    """保存模型权重，并剥离 DDP 包裹产生的 'module.' 前缀与 torch.compile 产生的
    '_orig_mod.' 段，保证存档无论是否经 DDP/compile 都能被后续普通加载/resume 使用。

    DDP 下**直接** `model.state_dict()` 就是完整权重，不需要任何汇聚 helper：
    DDP 每 rank 各持一份完整模型、只同步梯度，`state_dict()` 本身**不是
    collective**（没有 all-gather）。所以调用方把它放在 `is_main` 分支里既安全
    也没有额外开销 —— 键名形如 'module.backbone.…'，剥掉 'module.' 之后与包裹前
    完全同形，`src/inference.py` / `scripts/evaluate.py` / `webui.py` /
    `_load_model_state` 一行都不用改，旧 checkpoint 继续可读。

    '_orig_mod.' 由 `torch.compile` 插入，出现在**路径任意层级**（不只在开头）。
    故这里按 `'_orig_mod.' in k` 判存在性、按 `str.replace` 去**所有**层级，
    不能沿用 `replace(..., 1)`（只换首个）——那在嵌套包装下会原样留下中段前缀，
    存档键与 evaluate.py / inference / webui / convert_ckpt 期待的未编译布局
    对不上，load_state_dict 直接失败。
    （曾经还有第二种形态：加速器 的 Linear-only 逐子模块编译，段落落在路径中段。
    该后端已整体移除，但「去所有层级」这条不能退回只换首个。）
    """
    sd = _plain_state_dict(model)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(sd, path)


def _clone_to_cpu(obj):
    """递归把嵌套结构里的**张量换成独立的 CPU 副本**，其余原样返回。

    为什么必须是「独立副本」而不是视图：optimizer state 与 `ema.shadow` 在
    提交之后仍会被每个 step 改写，若后台线程去序列化同一块显存，写出来的文件
    可能是**撕裂的**（前半是旧值、后半是新值）。这条在同步保存时不存在 ——
    那时训练线程就卡在 `torch.save` 上，不可能有并发写。
    """
    if torch.is_tensor(obj):
        return obj.detach().to('cpu', copy=True)
    if isinstance(obj, dict):
        return {k: _clone_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_to_cpu(v) for v in obj)
    return obj


class _AsyncSnapshotWriter:
    """后台线程写盘，让训练步不等磁盘。

    为什么需要：2026-10-08 的 A100 40G 实测，GPU 利用率每~25 秒准时跌到 **0**，
    形状与「GPU Time Spent Accessing Memory」一致 ⇒ 卡的是 host。eval 期间 GPU
    是**忙**的（只有 60-70% 的浅谷），能把利用率打到 0 的只有保存：模型 +
    optimizer state（Adam 的 m/v 是参数量的两倍）+ EMA 一次性 D2H 再
    `torch.save`，全在 step 循环里。

    契约
    ----
    * **至多一个在飞**：`submit` 先 join 上一个 ⇒ 显存里最多多留一份快照，不会
      因写盘慢而堆积（堆积 = 吃显存 = 本末倒置）。
    * `close()` 必须被调用，否则最后一份可能没写完。
    * 写盘失败**不抛回训练线程**（训练已跑很久，不能因一次 IO 失败崩掉），只记
      error 日志。同步版本是会崩的 —— 这是行为变化，故在此写明。
    * 自己 `makedirs`：干净检出（`models/` 不存在）时首次周期快照会抛
      `Parent directory does not exist`，而该异常正好被上面那条吞掉。
    """

    def __init__(self, logger):
        self._logger = logger
        self._ex = ThreadPoolExecutor(max_workers=1)
        self._pending = None
        self._n = 0

    def _write(self, model_sd, train_state, model_path, state_path):
        try:
            for p in (model_path, state_path):
                d = os.path.dirname(p)
                if d:
                    os.makedirs(d, exist_ok=True)
            torch.save(model_sd, model_path)
            torch.save(train_state, state_path)
            self._logger.info("[save] 后台写盘完成: %s（第 %d 次快照）",
                              os.path.basename(model_path), self._n)
        except Exception as e:  # noqa: BLE001 — 写盘失败不该带走训练
            # error 而非 warning：失败被吞掉是有意设计，但周期快照失败的后果是
            # 「崩溃后无法恢复」，等级必须够高且写清是哪个文件，否则排查时
            # 根本不会注意到。2026-10-09 真机事故：干净检出（models/ 不存在）
            # 首次快照即抛 FileNotFoundError，被这里吞掉后训练照跑、快照一路
            # 静默失败 ⇒ 上面那行 makedirs 就是为此加的。
            self._logger.error("[save] 后台写盘失败（训练继续）: %s: %s | model=%s state=%s",
                               type(e).__name__, e, model_path, state_path)

    def submit(self, model_sd, train_state, model_path, state_path):
        """排一次快照。`model_sd` / `train_state` 会被**深拷贝成 CPU 副本**。"""
        self.join()
        self._n += 1
        payload = (_clone_to_cpu(model_sd), _clone_to_cpu(train_state))
        self._pending = self._ex.submit(self._write, payload[0], payload[1],
                                       model_path, state_path)

    def join(self):
        """等在飞的那次快照落盘完（不排新的）。

        ⚠ **绝不把写盘异常抛回调用方**。`close()` 是在训练循环**之后**调的，
        那里抛异常会连带跳过最终评估与 ONNX 导出；`submit()` 里的 `join()` 更是在
        step 循环里，一次磁盘故障不该带走整跑。异常在 `_write` 里已记过一次，
        这里只兜底（`future.result()` 会重抛 worker 里的异常，而那条路径不经过
        `_write` 的 try —— 例如线程被中断、或将来有人重构掉那层 try）。
        """
        if self._pending is None:
            return
        try:
            self._pending.result()
        except Exception as e:  # noqa: BLE001 — 见 docstring
            self._logger.warning("[save] 等后台快照时抛出（已忽略）: %s: %s",
                                 type(e).__name__, e)
        finally:
            self._pending = None

    def close(self):
        self.join()
        self._ex.shutdown(wait=True)


def setup_logging(log_file, level: int = logging.INFO, rank: int = 0) -> logging.Logger:
    """配置 logging：同时写文件与输出到控制台（无缓冲，实时可见）。

    rank>0（DDP 非主进程）时仅保留 ERROR 以上到 stderr，避免多卡日志刷屏——
    各进程是独立进程，各自的 logger 互不干扰，这里只控制本进程输出量。
    """
    logger = logging.getLogger('train')
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    else:
        eh = logging.StreamHandler(sys.stderr)
        eh.setLevel(logging.ERROR)
        eh.setFormatter(fmt)
        logger.addHandler(eh)
    return logger


_HAS_FOREACH = hasattr(torch, '_foreach_mul_') and hasattr(torch, '_foreach_add_')

# GradScaler 低于该值即视为「缩放值已被反复压低」，值得告警。
# PyTorch 默认 init_scale=65536；每发生一次溢出就减半。
_OVERFLOW_WARN_SCALE = 1024.0


def _locate_overflow(optimizer, logger, max_report=3, phase='unscale 后',
                    named_params=None):
    """定位参数组里的 inf/nan 梯度来源，只报不修（修是别处的责任）。

    **必须在 `clip_grad_norm_` 之前调用**（2026-10-04 云端 旧多卡环境 实测打出来的）。
    `clip_grad_norm_(max_norm=1.0)` 的实现是

        total_norm = ‖所有梯度‖                   # 有 inf ⇒ total_norm = inf
        clip_coef  = max_norm/(total_norm + 1e-6)  # = 0
        p.grad.mul_(clip_coef)                    # inf × 0 = NaN

    `total_norm = inf ⇒ clip_coef = 0 < 1` ⇒ 这个分支**一定会进**。所以在
    clip 之后统计，「inf 个数」**结构上恒为 0**，看到的 nan 全是 clipper 造的。

    后果不是「报告不好看」，而是**方向性误导**：真机上那行
    `有 0 个 inf / 211 个 nan 参数` 被读成「反向算出了 NaN」，排查方向偏向
    loss 与前向；而实际上**前向与 loss 都有限**，真凶是**反向算出了 inf**。

    Args:
        named_params: ``{参数名: 参数}``，用于按模块点名。缺省则跳过该段
            （拿不到名字时也要能跑，诊断不该反过来把训练搞崩）。
    """
    groups = optimizer.param_groups
    lrs = [float(g.get('lr', 0.0)) for g in groups]
    min_lr = min(lrs) if lrs else 0.0
    bad = []
    for gi, group in enumerate(groups):
        n_inf = n_nan = 0
        for p in group.get('params', []):
            if p.grad is None:
                continue
            if torch.isinf(p.grad).any():
                n_inf += 1
            elif torch.isnan(p.grad).any():
                n_nan += 1
        if n_inf or n_nan:
            is_value = lrs[gi] > 1.5 * max(min_lr, 1e-12)
            bad.append((gi, lrs[gi], n_inf, n_nan, is_value))
    if not bad:
        logger.warning("[fp16] GradScaler 报告溢出，但此刻（%s）参数上没有 "
                    "inf/nan —— 溢出可能发生在已被释放的中间张量里。", phase)
        return
    for gi, lr, n_inf, n_nan, is_value in bad[:max_report]:
        logger.warning("[fp16] 梯度溢出（%s）：参数组 %d（%s, lr=%.2e）"
                    "有 %d 个 inf / %d 个 nan 参数",
                    phase, gi, 'value head' if is_value else 'backbone/policy',
                    lr, n_inf, n_nan)
    # **按模块点名**：组名只是按 LR 比值**猜**的（`--value-lr-mult` 一改 guess
    #   就失效 —— 真机日志里三组全被打成 "backbone/policy" 就是这个原因），
    #   而「哪个模块的梯度爆了」才是能直接定位的信息。
    if named_params:
        id2name = {id(p): n for n, p in named_params.items()}
        mods = {}
        for gi, _lr, _i, _n, _v in bad:
            for p in groups[gi].get('params', []):
                if p.grad is None:
                    continue
                ni = int(torch.isinf(p.grad).sum())
                nn = int(torch.isnan(p.grad).sum())
                if ni or nn:
                    nm = id2name.get(id(p), '?')
                    top = nm.split('.')[0] + '.' + (
                        nm.split('.')[1] if '.' in nm[1:] else '')
                    cur = mods.get(top, (0, 0))
                    mods[top] = (cur[0] + ni, cur[1] + nn)
        if mods:
            top = sorted(mods.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:8]
            logger.warning("[fp16] 溢出按模块点名：%s",
                        ', '.join('%s[inf=%d nan=%d]' % (n, i, j)
                                    for n, (i, j) in top))
            logger.warning("[fp16] 下一个 %s 里这些 inf 会变成 NaN"
                        "（clip_grad_norm_ 算出的 clip_coef=0，inf×0=NaN），"
                        "所以**之后**再统计就看不到 inf 了。",
                        'clip_grad_norm_')
    if any(b[4] for b in bad):
        logger.warning("[fp16] 溢出集中在 value head —— 优先下调 --value-loss-weight"
                    "（默认 1.0，无补偿）与 --value-lr-mult（默认 5.0）")
    if any(not b[4] for b in bad):
        logger.warning("[fp16] 溢出涉及 backbone/policy —— 优先下调 --lr。"
                    "**不要**去调 --attn-window：V7 的 MHSA 没有窗口参数"
                    "（该旗标只对 12 通道路径有意义，而它同样没接进建网，"
                    "见 main() 里 V7 的 attn-window 提示与 "
                    "tests/test_no_aicpu_ops_in_startup_check.py）。"
                    "若报的是 nan（而不是 inf），先怀疑**前向**而非梯度："
                    "GradScaler 只缩放梯度，NaN 一旦来自前向就不可恢复，"
                    "降 scale 无用。")

def _grads_nonfinite_local(optimizer):
    """本地：任一参数梯度里有 inf/nan？

    **必须在 `clip_grad_norm_` 之前问** —— 见 `_locate_overflow` 的 docstring：
    clip 在 `total_norm = inf` 时算出 `clip_coef = 0` 并 `grad.mul_(0)`，
    `inf × 0 = NaN`，所以 clip 之后「inf 个数」结构上恒为 0。

    **整条路径只做一次 D2H**：`.all()` 逐参数跑在设备上，用 `|` 累积成**一个**
    设备标量，最后只 `bool()` 一次。写成 `bool(torch.isfinite(p.grad).all())`
    放进循环里就是每个参数一次同步 —— V7 有 322 个参数张量，每步 322 次 D2H
    会把训练拖垮（本文件原先那条
    `test_real_step_is_derived_from_scale_not_from_grads` 反对的正是这个）。
    """
    acc = None
    for group in optimizer.param_groups:
        for p in group.get('params', []):
            if p.grad is None:
                continue
            b = torch.isfinite(p.grad).all()
            acc = b if acc is None else (acc | b)
    return False if acc is None else (not bool(acc))


def _grads_nonfinite_any_rank(optimizer):
    r"""**任一 rank** 的梯度非有限？—— 这正是 `GradScaler` 缺的那一次归约。

    `GradScaler` 的 `found_inf` 是纯本地状态：`unscale_` 只登记本 rank 检出的非有限，
    `step` 只看本地标记，全程没有任何 collective。于是在多卡下：

        某 rank 有 inf  ->  它跳过；其余 rank 照常 `optimizer.step()`

    ⇒ **各 rank 的权重从此永久不同**，之后每次 `all_reduce` 都在混合**四个不同
    模型**的梯度；各 rank 的 scale 也各自独立减半、进一步漂移。这不是"少训几步"
    的损失，是训练从此无效（2026-10-05 真机 4×旧多卡环境：50 步内 1024 -> 8、
    `skip=7/50`，而"四张卡同一步一起溢出"的概率极低 ⇒ 分叉几乎立刻发生）。

    通信域未建（单卡）时退化成只看本地，行为与改动前一致。
    """
    local = 1 if _grads_nonfinite_local(optimizer) else 0
    if not _dist_active():
        return bool(local)
    # 设备必须跟着梯度走：NCCL 的 collective 要求张量落在本 rank 的 加速器 上。
    dev = torch.device('cpu')
    for group in optimizer.param_groups:
        for p in group.get('params', []):
            if p.grad is not None:
                dev = p.grad.device
                break
        else:
            continue
        break
    t = torch.tensor([local], dtype=torch.int32, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(int(t.item()) > 0)


def _scaler_step_global(scaler, optimizer, *, scale_before, use_scaler,
                        logger=None):
    """按**全局**判定决定跳/不跳，返回 True = 这一步真的更新了权重。

    为什么不直接用 `scaler.step(optimizer)`
    -------------------------------------
    `scaler.step` 内部只看本地 `found_inf`，多卡下各 rank 会做出**不同**的跳/不跳
    决定 ⇒ 权重分叉（见 `_grads_nonfinite_any_rank`）。所以这里改成：

    * `use_scaler=False`（bf16 路径）⇒ 直接 `optimizer.step()`，不做任何判定；
    * 全局有非有限 ⇒ **所有 rank 一起跳**，并把 scale 显式设成
    `scale_before * backoff_factor`。**显式给值**是关键：`update()` 免参版会读
    本地 `found_inf`（在没溢出的 rank 上是 0，于是那个 rank 会把 scale 往**上**调）
    ⇒ 各 rank 的 scale 就此错位，而 `scaler.unscale_` 是在各 rank 上各自除以
    自己的 scale 的，除数不同 ⇒ all_reduce 混合的是不同尺度的梯度；
    * 全局干净 ⇒ `optimizer.step()` + `scaler.update()` 免参版（保持
    `growth_interval` 的增长语义不变；此时每个 rank 本地都没 inf，
    `update()` 看到的是一致的 0）。

    代价：跳步那条走的是显式 `new_scale`，不会顺手重置 `growth_tracker`。
    全部 rank 走**同一条**分支，所以 tracker 仍然跨 rank 一致；只是"连续成功"的
    计数跨过一次跳步而不是被清零 —— `growth_interval` 默认 1e5 量级时可忽略。
    """
    if not use_scaler:
        optimizer.step()
        return True
    if _grads_nonfinite_any_rank(optimizer):
        if logger is not None:
            logger.debug('[fp16] 全局判定有非有限梯度，全体跳步（scale %.0f -> %.0f）',
                        scale_before,
                        scale_before * scaler.get_backoff_factor())
        scaler.update(scale_before * scaler.get_backoff_factor())
        return False
    optimizer.step()
    scaler.update()
    return True


def _ema_key(name: str) -> str:
    """EMA shadow 的键：去掉 torch.compile 往参数名里插的 '_orig_mod.' 段。

    必须去，否则编译一开就 KeyError：EMA 在**编译之前**按当时的
    `named_parameters()` 名字建 shadow，而 update / apply_shadow / restore 是按
    **运行时**的名字取键的。torch.compile 无论哪种形态都会改名字（整模型 compile
    插在开头，Linear-only compile 插在路径中段），键空间一变就再也对不上。
    这条路径此前是直接崩的、而非被跳过：`shell/train_sft_a100_1card.sh` 同时开了
    `--compile 1` 与 `--use-ema 1`，5 个 加速器 SFT 脚本也都开了 `--use-ema 1`。

    去段后 shadow 的键停在「未编译布局」，与 train_state 里的 `ema_shadow` 键一致，
    旧 checkpoint 不受影响（它们的键本来就没有这一段）。
    """
    return name.replace('_orig_mod.', '')


class EMA:
    """指数移动平均（Exponential Moving Average）权重。

    维护模型参数的 shadow copy，eval/save 时用 EMA 权重可提升 1-3% accuracy。
    键一律经 `_ema_key` 归一，理由见该函数。
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {_ema_key(name): param.clone().detach()
                    for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        if _HAS_FOREACH:
            # 原实现是逐参数 Python 循环，每参数 2 次独立设备 kernel 启动。
            # 本模型 depth 很深（backbone 17 + res 8 + convnext 4 + attn 5
            # + value 8 + policy 3），参数张量达数百个，即每步数百次启动；
            #  的 ACL 单次启动开销明显高于 CUDA，累积可观。
            # foreach 把它们合并成两次批量调用，数值语义与循环完全一致。
            # 刻意不缓存张量列表：resume 时 ema.shadow 会被整体替换
            # （见 main 的 resume 分支），缓存会持有失效张量。每步重建
            # 列表的 Python 开销可忽略，收益全在 kernel 启动数上。
            sh = []
            ps = []
            for name, param in self.model.named_parameters():
                sh.append(self.shadow[_ema_key(name)])
                ps.append(param)
            torch._foreach_mul_(sh, self.decay)
            torch._foreach_add_(sh, ps, alpha=1 - self.decay)
            return
        for name, param in self.model.named_parameters():
            k = _ema_key(name)
            self.shadow[k].data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        self.backup = {_ema_key(name): param.clone()
                    for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data = self.shadow[_ema_key(name)].data

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[_ema_key(name)].data
        del self.backup


def maybe_autocast(device, dtype=torch.float16):
    """在 CUDA 上开启 autocast，dtype 由设备能力决定（A100/BF16、V100/FP16）。

    CPU 或 amp 关闭时返回 `nullcontext`。device 字符串支持 'cuda'/'cuda:0'。

    ⚠ 必须**总有返回值**：调用点直接 `with maybe_autocast(...)`，一旦落到函数
    末尾（隐式返回 None）就是 `TypeError: 'NoneType' object does not support
    the context manager protocol` —— 而且只在 CPU 路径上炸（eval / 全精度对照），
    GPU 训练那条路完全看不见。
    """
    dev = device.split(':')[0] if isinstance(device, str) else str(device)
    if dev == 'cuda' and hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        try:
            return torch.amp.autocast(dev, dtype=dtype)
        except TypeError:
            # 老接口回退
            return torch.cuda.amp.autocast(enabled=True, dtype=dtype)
    return nullcontext()


def _positive_beta(text):
    """`--huber-beta` 的 argparse type：要求严格 > 0。

    为什么必须校验（P4.5-fix）：`beta<=0` 时 `F.smooth_l1_loss` **不报错**——
    `beta=0` 静默退化成纯 L1（`|d|`，拐点消失），`beta<0` 才在**训练跑到第一
    个 batch 的反向**时抛原生 `RuntimeError`（位置在 `compute_*_loss` 里，
    栈里全是训练循环，用户拿不到「是哪个旗写错了」的信息）。改成在解析期就
    用 argparse 的标准错误格式拒掉。
    """
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f'--huber-beta 必须 > 0（收到非数值 {text!r}）')
    if not (value > 0.0) or value != value or value == float('inf'):
        raise argparse.ArgumentTypeError(
            f'--huber-beta 必须 > 0（0 会静默退化成 L1，负值会让 '
            f'F.smooth_l1_loss 在训练中途抛 RuntimeError），收到 {text!r}')
    return value


def _at_least_one(text):
    """`--soft-every` 的 argparse type：要求 >= 1 的整数。

    为什么必须校验：`--soft-every 0` 会让 `step % 0` 变成 ZeroDivisionError ——
    报错点落在训练循环里，栈里全是训练代码，看不出是哪个旗写错了。`step % N`
    的语义里 N=0 本来就无意义（没有「第 0 步」以外的可判定边界）。
    """
    try:
        value = int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f'--soft-every 必须是 >= 1 的整数（收到 {text!r}）')
    if value < 1:
        raise argparse.ArgumentTypeError(
            f'--soft-every 必须 >= 1（0 会让 step % N 抛 ZeroDivisionError，'
            f'负数没有「每 N 步一个软批」的意义），收到 {value}')
    return value


#: `build_soft_index.py` 的产物里必须有的键。缺一个就报错而不是当默认值 ——
#: 缺 `idx` 意味着「这批行没有软标签」，缺 `policy` 意味着「软标签是全 0」，
#: 两者都会静默把软 CE 训成 no-op。
SOFT_INDEX_KEYS = ('idx', 'policy')


def attach_soft_index(dataset, path, data_npz=None):
    """把软标签索引挂到数据集上（`--soft-index` 的实现）。

    产物格式（与 `scripts/build_soft_index.py` / `src/data/kata_label_join.py`
    的 `build_soft_index` 一字对齐）::

        idx     int64  (M,)      数据集行号
        policy  float16(M, 362)  对应的 KataGo 访问分布（行和 ≈ 1）

    必须在**预取器 fork 之前**调用：软标签挂在 dataset 对象上，fork 之后
    再挂就只有父进程看得见，worker 会继续造 `soft_mask` 全 0 的批 —— 而训练
    不报任何错，只是「软标签训了个寂寞」。

    Args:
        data_npz: 主数据集 npz 路径。给了就做**防陈旧校验**（2026-10-06 事故）：
            把产物 `cache_meta` 里记录的数据集指纹（行数等）与**当前**数据集
            现算指纹比对，行数不一致 ⇒ SystemExit。事故形态：数据集重建、
            labels 续跑/重打之后 `soft_index.npz` 没有重建，训练直接吃旧文件
            —— 行号全体错位，policy 与局面**逐行无关**（真机实测：
            train_top1 崩到 0.2%、policy CE 钉在均匀上界 5.85、argmax 30.8%
            落在已占点上）。本校验只算文件指纹，秒级。
            不给 `data_npz`（单测/旧调用）则跳过。
    """
    z = np.load(path, allow_pickle=False)
    missing = [k for k in SOFT_INDEX_KEYS if k not in z.files]
    if missing:
        raise KeyError(
            f'--soft-index {path} 缺字段 {missing}；现有 {sorted(z.files)}。'
            f'请用 scripts/build_soft_index.py 重新生成（它是唯一的产物写者）。')
    if data_npz is not None:
        if 'cache_meta' not in z.files:
            print(f'[soft] ⚠ {path} 没有 cache_meta 指纹（旧版产物），'
                f'无法校验新鲜度 —— 建议用 scripts/build_soft_index.py 重建',
                flush=True)
        else:
            import json as _json  # noqa: PLC0415
            from src.data.kata_label_join import npz_fingerprint  # noqa: PLC0415
            meta = _json.loads(str(np.asarray(z['cache_meta']).reshape(-1)[0]))
            ds_meta = (meta or {}).get('dataset') or {}
            cur = npz_fingerprint(data_npz, id_key='game_ids')
            # 行数不符 = 行号全体平移 ⇒ **硬失败**（2026-10-06 真机事故就是它：
            # 数据集重建 +61,512 行，旧索引的行号指向完全不同的局面）。
            if ds_meta.get('rows') not in (None, cur['rows']):
                raise SystemExit(
                    f'--soft-index {path} 是对**旧版数据集**建的'
                    f'（索引时 {ds_meta.get("rows")} 行，当前 {cur["rows"]} 行）'
                    f'⇒ 行号全体错位、policy 与局面逐行无关（2026-10-06 事故）。\n'
                    f'请重建后再训：\n'
                    f'  python scripts/build_soft_index.py --data {data_npz} '
                    f'--labels <kata_labels.npz> --out {path} '
                    f'--materialized-dir tmp/materialized')
            # mtime/size/game_id 去重数不符只告警：跨机拷贝会改 mtime，硬失败
            # 会误伤；但它们值得被看见。
            drift = [k for k in ('n_distinct', 'mtime_ns', 'size')
                    if ds_meta.get(k) not in (None, cur.get(k))]
            if drift:
                print(f'[soft] ⚠ {path} 相对其构建时的数据集有字段漂移：{drift}'
                    f'（行数一致 ⇒ 行号仍有效；若换过数据内容请重建索引）',
                    flush=True)
    idx = np.asarray(z['idx']).ravel()
    pol = np.asarray(z['policy'])
    diag = dataset.attach_soft(idx, pol)
    diag.update({'path': os.path.abspath(path),
                'index_rows': int(idx.size),
                'policy_shape': tuple(int(x) for x in pol.shape),
                'hit_rate': (int(idx.size) / max(len(dataset), 1))})
    return diag


def resolve_policy_loss_kind(policy_loss, soft_index, v7_packed=False):
    """软标签接入后，**生效**的 policy 损失 kind。

    `soft_ce` 不是又一个旗：它是软标签的**派生**结果。给了 `--soft-index`
    就用软 CE（段 2 的全部意义），没给就原样返回 `--policy-loss`（段 1 的旧路径，
    数值逐位不变）。

    显式 `soft_ce` 却没有软标签来源必须**报错**：`compute_policy_loss`
    拿不到 `soft`/`soft_mask` 会抛 ValueError，但那是训练跑到第一个 batch
    时的栈 —— 这里提前拒掉，错误信息直指 CLI。
    `--policy-loss huber` + `--soft-index` 也报错：huber 分支**完全不看**
    软标签，组合起来等于「以为在蒸馏，其实在回归 one-hot 的概率」。
    """
    # V7 的 stdata 分片把 KataGo 的搜索分布**内建**在行里（`V7PackedDataset`
    # 直接给 `soft` / `soft_opp` / `soft_mask`），不经过 `--soft-index`。
    # 那条 CLI 守卫在这里对段 3 是**误报**：软项真的在算，只是来源不是索引文件。
    have_soft = bool(soft_index) or bool(v7_packed)
    if not have_soft:
        if policy_loss == 'soft_ce':
            raise SystemExit(
                '--policy-loss soft_ce 需要软标签来源：要么 --soft-index，'
                '要么用 V7 的 stdata 分片（--v7 1 + --data data/stdata_v7）。'
                '否则 soft_mask 全 0，软项恒 0 —— 训练照跑但什么也没学到。')
        return policy_loss
    if policy_loss == 'huber':
        raise SystemExit(
            '--policy-loss huber 与软标签互斥：huber 分支不消费软标签，'
            '组合起来会「以为在蒸馏、其实在回归 one-hot 概率」。段 2 请让 '
            '--soft-index 接管（它把 kind 派生为 soft_ce）。')
    return 'soft_ce'


def _soft_kind_for_step(hard_kind, step, soft_every):
    """第 `step` 个 micro-batch 走哪个 policy kind（`--soft-every` 的节奏）。

    `soft_every == 1`（默认）⇒ 每步都走软 CE。`> 1` ⇒ 只有 `step % N == 0` 的
    那些步走软 CE，其余步回退到 `--policy-loss` 的硬目标口径。

    非软步里的软行会按 **one-hot** 训（软项被完全跳过）。这正是 spec §5
    退路表点名的「混合训练 ⇒ index 0 向两种语义折中」，所以默认值是 1，
    且 `>1` 时 main() 会打 warning。
    """
    every = max(1, int(soft_every))
    if every == 1:
        return 'soft_ce'
    return 'soft_ce' if step % every == 0 else hard_kind


def narrow_to_soft_rows(train_idx, dataset):
    """`--soft-only-sampling`：把训练行索引空间收窄到「有软标签的行」。

    段 2 的决定（spec §1.2）：**只训软行**，而不是全量混训 / 靠权重混 ——
    同一个 head 同时收到搜索分布与人类 one-hot 会去折中而不是学搜索；
    `--soft-weight 0` 关不掉（那是「开了软标签但 99% 样本仍在教 one-hot」的
    静默半吊子）。与 Phase 4 item 3 的滑动窗口**正交**，两者可叠加。

    收窄到 0 行必须**报错**而不是退回全量：静默退回正是本函数要防的那个
    失败模式（用户以为在跑段 2，实际在跑段 1）。
    """
    # 先钉成 int64 再用。上游按棋局切分是 `np.array([...])` 的推导式，
    #   当棋局数太少（`int(局数 * 0.98) == 0`）时它返回**空**数组，而
    #   `np.array([])` 的 dtype 是 float64 —— 直接拿去索引会抛
    #   「arrays used as indices must be of integer」，把「这批数据切不出
    #   训练集」这件清楚的事报成一个 numpy 内部错误。
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if train_idx.size == 0:
        raise SystemExit(
            '--soft-only-sampling 之前的**训练集就是空的**，所以无法收窄。\n'
            '  最常见原因：按棋局切分要留 2% 做验证集，而这批数据棋局数太少'
            '（`int(棋局数 * 0.98) == 0`）⇒ 全被划去验证集了。\n'
            '  这与软标签无关，别去查软索引。')
    keep = dataset.soft_row_mask()
    if keep.shape[0] != len(dataset.moves):
        raise SystemExit(
            f'软标签掩码长度 {keep.shape[0]} ≠ 数据集行数 '
            f'{len(dataset.moves)} ⇒ 软索引与数据不是同一份，别硬跑。')
    out = train_idx[keep[train_idx]]
    if out.size == 0:
        raise SystemExit(
            '--soft-only-sampling 之后训练行为 0：软标签与训练集没有交集'
            '（索引陈旧？散列口径漂移？）。**不要**把它当成「这批局面真的没标签」'
            '—— 先跑 scripts/build_soft_index.py 看它的缓存告警。')
    return out


def load_dataset(path):
    """加载单个 .npz 训练集。"""
    d = np.load(path, allow_pickle=False)
    # 通道数必须与**模型侧同改**（训练结构恒 = KATAGO_SE_CFG）。
    # SupervisedDataset 的默认 12 是它自己的默认（dataset.py C7），这里不显式传
    # 就等于赌「模型侧也是 12」—— 两者一旦分叉，第一个 batch 就形状错。
    return SupervisedDataset({k: d[k] for k in d.files},
                            n_channels=KATAGO_SE_CFG['in_channels'])


# ---- eval 的确定性：固定采样源 + 与训练 RNG 流隔离 + 关闭数据增强 ----
# 修复前两个评估函数都不给 rng，抽样落到**全局** np.random，后果两层：
#   ① 同一份权重、同一份 eval_idx 连跑两次 eval 指标不同（KL/Brier 尤其抖）——
#      「模型变好了」与「这次抽到的变换不一样」分不开；
#   ② 每次 eval 推进全局流 → 训练侧 `rng.shuffle(train_idx)` 与训练 batch 的增强
#      抽样结果取决于「eval 跑过几次、什么时候跑」⇒ **eval 频率会改写训练轨迹**。
# P2.2 用固定种子的独立 Generator + `_isolated_global_rng()` 兜住了这两层。
#
# 但「固定」不等于「正确」：被评估的仍是**随机挑了对称**的验证集，推理时不会出现
# 随机翻转的棋盘。故评估路径一律传 `augment=False` —— **评估零随机**，于是「指标与
# 采样种子无关」才真正成立。连带后果：`EVAL_SAMPLING_SEED` 与 `evaluate_*(...,
# rng=)` 现在**不被消费**（只是透传给 `sample_batch_numpy`），保留是为了既有的契约锁
# 与将来的显式入口；训练侧 `_BatchPrefetcher` 的 `seed=1234` 与它数值相同但互不相干。
# 增强**只在训练侧**发生，见 `tests/test_eval_no_augment.py`。
EVAL_SAMPLING_SEED = 1234


def _eval_rng() -> np.random.Generator:
    """评估用的随机源：每次调用返回**全新**的、由 `EVAL_SAMPLING_SEED` 播种的 Generator。

    「全新」是复现的前提：同一个 Generator 连抽两次会前进，两次 eval 就会在**开启增强**
    时拿到不同的变换向量。

    注意：评估现在固定传 `augment=False`（评估零随机），所以这个 Generator 事实上
    **不会被消费** —— 它只是被原样透传给 `sample_batch_numpy`。保留这条链路的理由：
    `tests/test_eval_determinism.py` 的「eval 传下去的 rng 由 `EVAL_SAMPLING_SEED` 播种」
    是一条既有契约锁；且将来若评估要重新开启某种抽样（例如按棋局分层抽样），显式
    `rng=` 就是那条入口，不需要再改函数签名。
    """
    return np.random.default_rng(EVAL_SAMPLING_SEED)


@contextmanager
def _isolated_global_rng():
    """上下文期间对全局随机流（numpy + torch CPU）的改动在退出时全部还原。

    eval 侧的随机性已经走独立的 `rng`，这里是第二道防线：任何仍会碰到全局随机源的
    路径（老调用点、将来新增的增强、第三方库内部的 `np.random` 调用）都不会把状态推进
    出去，训练轨迹因此与「eval 跑过几次」无关。torch 侧同理 —— eval 走
    `inference_mode` + `model.eval()`，本不该消耗任何 torch 随机，一并存取只为把这条
    性质钉住（见 `tests/test_eval_determinism.py`）。

    还原放在 `finally`：eval 中途 OOM 抛错也不该顺手改掉训练的随机流。
    """
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def _eval_batch_budget(n_samples, bs, max_batches):
    """验证集批数预算：返回 (n_batches, truncated)。

    `max_batches` 为 None 或 <= 0 时**不截断**（跑满整个验证集）；否则只跑前
    `max_batches` 批。`truncated` 表示确实因上限少跑了批 —— 调用方靠它把
    「这次评估只覆盖了验证集的一部分」显式写进返回值/日志，而不是让读日志的人
    误以为指标来自全量验证集。

    两个评估函数共用本函数，避免上限语义在两处漂移。
    """
    total = (n_samples + bs - 1) // bs
    if max_batches is None or max_batches <= 0:
        return total, False
    return min(total, max_batches), total > max_batches



def _apply_channels_last_(model, backend='cuda'):
    """就地把所有 **4D** 权重（Conv2d.kernel）转成 NHWC，返回转换个数。

    失败的**每一条路**都会抛出去，由调用方降级 —— helper 内部不吞异常
    （吞掉的话调用方会以为转换成功、继续把输入也转成 NHWC ⇒ 权重 NCHW +
    输入 NHWC 错配）。调用方拿到异常就把 ``use_channels_last`` 置回 False。

    为什么用 ``param.data = param.data.contiguous(memory_format=channels_last)``
    而不是 ``model.to(memory_format=channels_last)``
    --------------------------------------------------------------
    直接在 4D 卷积核上做 NHWC stride 转换（``.contiguous`` 在 CUDA/CPU 上均有效），
    并把结果写回 ``param.data`` 而非新建 Parameter，是为了保住 optimizer
    已经持有的引用 —— 否则这个 flag 一开，优化器就管不到卷积核了（会静默变成
    「卷积核不更新」，比崩溃更难查）。
    """
    n = 0
    for mod in model.modules():
        for name, param in list(mod.named_parameters(recurse=False)):
            if param.dim() == 4 and param.numel() > 0:
                param.data = param.data.contiguous(memory_format=torch.channels_last)
                n += 1
    return n


def _to_nhwc(t):
    """把 4D 输入张量转成 NHWC（**按张量所在设备**自动选后端），非 4D 原样返回。

    后端由 ``t.device.type`` 判定，不靠调用方传 —— 调用点在训练/评估热循环里
    （5 处），多传一个 backend 参数等于给每处都留一个传错的机会，而「权重用
    A 转换、输入用 B 转换」正是这次要根治的错。

    与 `_apply_channels_last_` 同一套判据：统一走 PyTorch 原生
    ``torch.channels_last`` 转换（CUDA/CPU 均有效）。

    ⚠ 输入与权重必须用同一套转换：权重转了、输入没转（或反过来）会让每次
    卷积白搬一次布局 —— 那等于 flag 没开。调用方只在权重转换**成功**后才置
    ``use_channels_last=True``（见 main），所以这里抛异常必须冒到训练层，
    不能静默跳过（静默跳过 = 权重 NHWC + 输入 NCHW，正是最坏组合）。
    """
    if t.dim() != 4:
        return t
    return t.to(memory_format=torch.channels_last)


def evaluate_top1(model, dataset, idxs, bs, device, amp_dtype, max_batches=50,
                use_channels_last=False, rng=None):
    """验证集 top-1 着法准确率。返回 (accuracy, num_samples)。

    返回形状按历史契约保持二元组（本函数在 train_sft.py 内当前没有调用点，
    `scripts/train_sft_ms.py` 里的是另一个独立同名函数），故截断信息不进
    返回值、只写日志。`max_batches` 为 None 或 <= 0 时跑满验证集。

    **评估一律传 `augment=False`**：验证集不做 8 路随机对称增强，因此准确率只取决于
    「权重 + 验证集原始长相」，与采样种子、eval 频率都无关。

    `rng`: 采样随机源。默认 None → 落到 `_eval_rng()`（固定种子
    `EVAL_SAMPLING_SEED`）。因为下面固定传 `augment=False`，这个 rng **不被消费**，
    只是被透传；保留它是为了 `tests/test_eval_determinism.py` 的既有契约锁与将来的
    显式入口。
    """
    model.eval()
    correct = 0
    total = 0
    batches = 0
    eval_rng = _eval_rng() if rng is None else rng
    n_batches, truncated = _eval_batch_budget(len(idxs), bs, max_batches)
    with _isolated_global_rng(), torch.no_grad():
        for b in range(n_batches):
            sel = idxs[b * bs:(b + 1) * bs]
            if len(sel) == 0:
                break
            states_np, moves_np, _ = dataset.sample_batch_numpy(sel, rng=eval_rng, augment=False)
            state = torch.from_numpy(states_np).to(device)
            if use_channels_last:
                state = _to_nhwc(state)
            move_t = torch.from_numpy(moves_np).to(device)
            with maybe_autocast(device, amp_dtype):
                policy_logits, _ = model(state)
            pred = policy_logits.argmax(dim=-1)
            correct += int((pred == move_t).sum())
            total += len(sel)
            batches += 1
    model.train()
    if truncated:
        logging.getLogger('train').info(
            "[eval] top1 验证集被截断：只跑了 %d 批（n=%d，max_batches=%s；"
            "传 <=0 可跑满验证集）", batches, total, max_batches)
    return correct / max(total, 1), total


#: eval 前向批大小封顶。eval 指标只按样本数归一、与批大小无关，但 eval 复用
#: 训练的 batch（曾为 5500/6000）⇒ 训练把常驻显存顶到 ~89% 后，eval 前向的
#: 瞬态分配（SDPA workspace /  碎片下的连续大块）直接把峰值顶到 95~96%，
#: 2026-10-06 真机两次贴线。封到 2048：峰值降一个量级，批数变多但 eval 总时长
#: 由特征同步计算主导、几乎不变，指标逐位不变（同样本、同顺序、同 augment=False）。
_EVAL_BATCH_CAP = 2048


#: SwanLab 面板黑名单（2026-10-06 面板瘦身）：这些量**照算**（stdout 的 [step] 行
#: 与诊断时都要用），只是不再逐 step 上报曲线 —— 面板只留「出事时第一时间看」
#: 与「判读必须的基线」两类。删一个键比加一个键容易，先瘦再按需加回。
#:   · 吞吐三兄弟留 `speed_per_card`/`eta_min`，`speed`/`speed_inst`/`elapsed_min`
#:     与其重合；
#:   · 五个分段计时整体退出曲线（stdout 仍有；`t_comp_ms` 只是 CPU 发射时间，
#:     单独读必然误判）；
#:   · `l2_report`/`opt_loss`/`scaler_scale`：bf16 无 scaler、L2 走解耦衰减，
#:     日常恒平；
#:   · `train_top5`/`policy_ce_random`/`value_rmse(_zero)`：基线与近失信号，
#:     stdout/诊断保留，曲线与 top1 高度重合；
#:   · eval 侧只留 top1/kl/brier + `best_eval_acc`，其余（ema/lr/gap/覆盖度）
#:     进 stdout 的 [eval] 行。
_SWANLAB_DROP_KEYS = frozenset({
    'l2_report', 'opt_loss', 'speed', 'speed_inst', 'elapsed_min',
    'scaler_scale', 't_data_ms', 't_comp_ms', 't_save_ms', 't_eval_ms',
    't_data_max_ms', 'train_top5', 'policy_ce_random',
    'value_rmse', 'value_rmse_zero',
    'eval_top5', 'eval_top10', 'eval_used_ema', 'eval_lr', 'eval_gap_to_best',
    'eval_n', 'eval_batches', 'eval_truncated',
    'final_kl', 'final_brier', 'final_batches', 'final_truncated',
})

#: 段位权重表（逐项 loss 曲线按它过滤：权重为 0 的项是**设计上的平线**，
#: 13 条里通常只剩 4 条主目标 —— 平线进 `run/loss_coef/*` 的解释，不进曲线）。
_SWANLAB_V7_WEIGHTS = v7_stage1_loss_weights()


def _eval_ref_target(lbl, moves, action_size, *, device, dtype):
    """eval 的参考分布：**有软标签用软标签，没有就退回真实着法的 one-hot**。

    `soft_mask` 的语义（`src/data/dataset.py:_build_labels` 与
    `src/data/v7_packed_dataset.py` 都是这个约定）就是「本行有没有软标签」，
    全 0 表示一行都没有。本仓对「board 级 V7 没有内建软标签」这件事**已有**注释
    与训练侧守卫（见 `resolve_policy_loss_kind` 上方、以及 `_soft_on` 那段）：
    `data/*full.npz` + 不给 `--soft-index` ⇒ `soft_row` 从未挂载 ⇒
    `_build_labels` 返回**全 0** 的 `soft`/`soft_mask`。

    但 eval 侧此前无条件读 `lbl['soft']`，守卫不覆盖它 ⇒
    `argmax(全0) = 0` ⇒ top1/5/10 退化成「模型是否预测 index 0」、KL 恒 0。
    真机 2026-10-07 实测就是这个形状：top1 0.0006 → 0.0002 越训越低，而随机基线
    top10 ≈ 2.76%、实测 0.49%；brier 却在正常改善（它读 `lbl['outcome']`）。
    影响面不止指标 —— `--early-stop-metric` 默认就是 top1，`best_eval_acc` 也读它。

    12 通道版 `evaluate_metrics` 一直以真实着法 `move_t` 为参考（见它的 top-k 段），
    所以这里不是引入新语义，而是让 V7 版在软标签缺席时**回到**那条既有语义；
    有软标签时行为逐字不变。
    """
    soft = torch.as_tensor(np.asarray(lbl['soft'])[:, :int(action_size)],
                           device=device, dtype=dtype)
    hard = _dense_move_target(moves, action_size).to(device=device, dtype=dtype)
    mask = lbl.get('soft_mask')
    if mask is None:
        return hard
    m = torch.as_tensor(np.asarray(mask), device=device,
                        dtype=dtype).reshape(-1, 1)
    return soft * m + hard * (1.0 - m)


def evaluate_metrics_v7(model, dataset, idxs, bs, device, amp_dtype, *,
                        max_batches=50, action_size=V7_ACTION_SIZE,
                        prefetcher=None, use_channels_last=False):
    """V7 路径的验证集综合指标（与 :func:`evaluate_metrics` **返回同构**）。

    为什么必须另写一份
    ------------------
    12 通道的 :func:`evaluate_metrics` 建的是 ``sample_batch(n_channels=12)``
    + ``model(state)`` 二元返回的评估器，对 V7（22 通道 19 维全局 + dict 返回）
    会直接形状错。与其在这里猜，不如把口径对齐 —— 否则 V7 训��永远看不到
    top1，`--early-stop` 也跟着形同虚设。

    指标口径（与 12 通道版刻意保持同名同义，便于 early-stop / best-model
    两处逻辑零改动复用）：

    ``top1`` / ``top5`` / ``top10``
        policy 的 top-k 命中率。参考着法 = ``argmax(soft)``。
    ``kl``
        ``KL(soft ‖ softmax(logits))``，即 12 通道版同款。
    ``brier``
        value 三分类的 Brier score（越小越好）；``--early-stop-metric loss``
        看的是它（**默认**的判据是 top1，见该参数）。

    **参考着法的回退**（见 :func:`_eval_ref_target`）：`soft_mask` 全 0 时 ——
    board 级 V7 + 不给 `--soft-index`，此时 `soft` 本身也是全 0 —— 上面两项一律
    改以**真实着法的 one-hot** 为参考，与 12 通道版同一语义。不回退的话
    `argmax(全0) = 0`，top1/5/10 会变成「模型是否预测 index 0」而 KL 恒 0，
    并且因为 top1 同时是 early-stop 与 best-model 的判据，早停会盯着噪声走。
    policy 通道 1 是 ``π_opp``（引擎语义里叫 optimism），本指标**只看通道 0** ——
        引擎在 ``policyOptimism=0`` 时也只消费通道 0。

    Returns:
        与 :func:`evaluate_metrics` 同构的 dict（另加 ``n`` 便于确认覆盖度）。
    """

    bs = min(int(bs), _EVAL_BATCH_CAP)   # 见 _EVAL_BATCH_CAP 注释
    model.eval()
    n = int(len(idxs))
    if n == 0:
        return {'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'kl': 0.0,
                'brier': 0.0, 'n': 0, 'batches': 0, 'truncated': False}

    nb = max_batches if max_batches and max_batches > 0 else 10 ** 9
    rng = np.random.default_rng(0)
    order = np.arange(n)
    rng.shuffle(order)                       # 与训练不同序，避免只看高数据量前缀

    hit1 = hit5 = hit10 = 0
    kl_sum = brier_sum = 0.0
    seen = 0
    batches = 0
    row_batches = [order[s:s + bs] for s in range(0, n, bs)][:nb]
    row_batches = [r for r in row_batches if r.size]

    def _iter_batches():
        """产出 (sp_np, gl_np, moves, lbl)，逐位等于串行 `v7_batch_sync` 路径。

        `prefetcher` 在场时把批投给 eval 特征池（窗口化投递提供背压），墙钟
        除以 worker 数；不在场时退回串行（行为与改造前逐位一致）。
        """
        if prefetcher is None:
            for rows in row_batches:
                yield v7_batch_sync(dataset, rows, device,
                                    rng=rng, augment=False)
            return
        win = 0
        for rows in row_batches:
            prefetcher.submit(rows)
            win += 1
            if win > prefetcher.prefetch:      # 窗口满 ⇒ 先收一个，提供背压
                win -= 1
                yield prefetcher.next()
        while win:
            win -= 1
            yield prefetcher.next()

    for sp_np, gl_np, moves, lbl in _iter_batches():
        batches += 1
        sp = v7_to_device(sp_np, device, amp_dtype)
        if use_channels_last:      # 与训练侧同一布局（见训练循环里的同名注释）
            sp = _to_nhwc(sp)
        gl = v7_to_device(gl_np, device, torch.float32)
        with torch.no_grad():
            out = model(sp, gl)
        pol = out['policy_logits']                      # (B, K, 361+1)
        pol = pol[:, 0, :action_size].float()           # 只看通道 0（π）
        logp = F.log_softmax(pol, dim=-1)

        target = _eval_ref_target(lbl, moves, action_size,
                                  device=logp.device, dtype=torch.float32)
        tgt_rank = target.argmax(dim=-1)
        top = logp.topk(min(10, action_size), dim=-1).indices
        eq = top.eq(tgt_rank.unsqueeze(1))
        hit1 += int(eq[:, 0].sum())
        hit5 += int(eq[:, :5].any(dim=1).sum())
        hit10 += int(eq.any(dim=1).sum())

        kl_sum += float((target * (torch.log(target.clamp_min(1e-9)) - logp)).sum(-1).sum())

        # ---- value Brier（三分类）----
        # `outcome` 已是 {0=胜,1=负,2=无结果} 的类别索引（to_play 视角），
        # 直接 one-hot 即得目标；noresult 行也算进去 —— 引擎同样会预测它。
        oc = out['outcome_logits'].float()
        oc_p = F.softmax(oc, dim=-1)
        oc_t = torch.as_tensor(np.asarray(lbl['outcome']),
                            device=oc_p.device, dtype=torch.long)
        oc_t = oc_t.clamp(0, oc_p.shape[1] - 1)
        oh = F.one_hot(oc_t, oc_p.shape[1]).to(oc_p.dtype)
        brier_sum += float((oc_p - oh).pow(2).sum(-1).sum())

        seen += int(len(lbl['soft']))

    model.train()
    return {
        'top1': hit1 / max(1, seen),
        'top5': hit5 / max(1, seen),
        'top10': hit10 / max(1, seen),
        'kl': kl_sum / max(1, seen),
        'brier': brier_sum / max(1, seen),
        'n': seen,
        'batches': batches,
        'truncated': bool(batches >= nb and batches * bs < n),
    }


def evaluate_metrics(model, dataset, idxs, bs, device, amp_dtype, max_batches=50,
                    use_channels_last=False, rng=None):
    """验证集综合指标：top-1/5/10 准确率 + policy KL + value Brier score。

    返回 dict:
        top1, top5, top10 : 着法准确率 (0~1)
        kl                : model softmax vs expert one-hot 的 KL散量
        brier             : value 预测 vs 实际胜负的 Brier score（越小越好）
        n                 : 样本数
        batches           : 本次**实际**跑掉的批数。防御性计数：批数预算已把尾部残批算作
                            整一批、循环里的 break 在非空 idxs 下不会触发，故当前恒等于
                            计划批数；仍按实际计数只为将来循环若提前退出时日志报真值
        truncated         : 是否因 --eval-max-batches 上限少跑了批（评估覆盖度不足）

    后两个键只增信息、不改计算：`max_batches` 为 None 或 <= 0 时跑满验证集。

    **评估一律传 `augment=False`**：验证集不做 8 路随机对称增强。这是 P2.2「固定
    采样种子」的下一块拼图 —— 种子只固定了「抽到哪一组对称」，可被评估的仍是一份
    随机翻转/旋转过的验证集；关掉增强后评估**零随机**，指标只取决于「权重 + 验证集
    原始长相」。`tests/test_eval_no_augment.py` 的用例 5 端到端锁这条性质。

    `rng`: 采样随机源。默认 None → 落到 `_eval_rng()`（固定种子
    `EVAL_SAMPLING_SEED`），显式传入则用调用方的流。因为下面固定传 `augment=False`，
    这个 rng **不被消费**、只是被透传 —— 「同一份权重 + 同一份 eval_idx 连跑两次，八个
    指标逐位相同」现在由「评估零随机」保证，不再依赖它。默认值即保障：任何调用点
    （包括将来新写的、忘了传 rng 的）都不可能不小心拿到不确定性。保留这条链路是为了
    `tests/test_eval_determinism.py` 的既有契约锁与将来的显式入口。

    采样循环整体包在 `_isolated_global_rng()` 里：eval 既不改全局 numpy 状态，也不改
    torch 状态，故训练侧的随机流与「eval 跑过几次、什么时候跑」无关。
    """
    import torch.nn.functional as F
    model.eval()
    correct1 = correct5 = correct10 = 0
    total = 0
    kl_sum = 0.0
    brier_sum = 0.0
    batches = 0
    eval_rng = _eval_rng() if rng is None else rng
    n_batches, truncated = _eval_batch_budget(len(idxs), bs, max_batches)
    with _isolated_global_rng(), torch.inference_mode():
        for b in range(n_batches):
            sel = idxs[b * bs:(b + 1) * bs]
            if len(sel) == 0:
                break
            states_np, moves_np, values_np = dataset.sample_batch_numpy(sel, rng=eval_rng, augment=False)
            # 全精度评估（CPU）才升 fp32；AMP 下 planes 的 fp16 直接用
            state = torch.from_numpy(states_np)
            if amp_dtype == torch.float32:
                state = state.float()
            state = state.to(device)
            if use_channels_last:
                state = _to_nhwc(state)
            move_t = torch.from_numpy(moves_np).to(device)
            value_t = torch.from_numpy(values_np).to(device)  # (B,1) in {-1,+1}
            with maybe_autocast(device, amp_dtype):
                policy_logits, value_pred = model(state)
            B = len(sel)
            total += B
            batches += 1

            # --- top-k 准确率 ---
            topk = policy_logits.topk(10, dim=-1).indices  # (B,10)
            correct1 += int((topk[:, 0] == move_t).sum())
            correct5 += int((topk[:, :5] == move_t.unsqueeze(1)).any(dim=1).sum())
            correct10 += int((topk == move_t.unsqueeze(1)).any(dim=1).sum())

            # --- policy KL 散量 ---
            # expert: one-hot at move_t -> log prob; model: log_softmax
            log_p = F.log_softmax(policy_logits, dim=-1)  # (B, A)
            # KL(expert || model) = sum(expert * (log_expert - log_model))
            # expert 为 one-hot，简化为 -log_p[expert_move]（即 cross-entropy）
            # 但更标准的 KL = sum(expert * log(expert / model))
            # = sum(expert * (0 - log_model)) for one-hot = -log_p[expert_move]
            # 即 NLL，与 CE 等价。若要真正 KL(expert||model) 需要 expert 有分布
            # 此处用 model 分布 vs uniform 的 KL 作为策略集中度指标：
            # KL(model || uniform) = log(A) + sum(p * log(p))
            A = log_p.shape[1]
            p = F.softmax(policy_logits, dim=-1)
            kl_per_sample = (torch.log(torch.tensor(A, dtype=p.dtype))
                            + (p * log_p).sum(dim=-1))  # (B,)
            kl_sum += float(kl_per_sample.sum())

            # --- value Brier score ---
            # Brier = mean((pred_01 − act_01)²)，两个量都先线性映射到 [0,1]：
            #   act_01 = (value_t+1)/2 ∈ {0,1}；pred_01 = (value_pred+1)/2。
            # 作用在**裸输出**上：`value_pred` 就是 `model(state)` 的第二个
            # 返回值，本文件全程没有 `tanh`；当前 `alphanet.py` 用的还是裸线性
            # `ValueNetwork`（`FCValueHead` 的 Tanh 尚未接线）。所以上一版注释
            # 写的「pred in tanh output」是陈旧且误导的 —— P4.5 让 value_t∈[-1,1]
            # 成了承重契约，这个假注释会让人以为映射里已经有一个 tanh。
            #
            # C8 删掉旧 BCE 分支的收益（机制，勿简化成「按 tanh 解释」）：旧
            # BCEWithLogits 的最优**裸输出**是 ±2.197（=logit(0.9)，因为旧分支把
            # 目标压成了 (v+1)/2*0.8+0.1），而 Brier 是把这个裸输出线性映射到
            # [0,1] 再与 0/1 比 —— 两者尺度错配，指标里带 ~2.197 量级的系统性
            # 伪影。改回归 ±1 之后，裸输出与 [-1,1] 契约对齐，伪影消失。
            pred_01 = (value_pred.squeeze(-1) + 1) / 2
            act_01 = (value_t.squeeze(-1) + 1) / 2
            brier_sum += float(((pred_01 - act_01) ** 2).sum())

    model.train()
    n = max(total, 1)
    return {
        'top1': correct1 / n,
        'top5': correct5 / n,
        'top10': correct10 / n,
        'kl': kl_sum / n,
        'brier': brier_sum / n,
        'n': total,
        # 覆盖度可见化：只增信息，不参与上面任何一个指标的计算
        'batches': batches,
        'truncated': truncated,
    }


def _concat_dicts(dicts):
    """按相同 key 沿第 0 轴拼接多个数据 dict（字段形状一致）。"""
    out = {}
    for k in dicts[0].keys():
        out[k] = np.concatenate([dd[k] for dd in dicts], axis=0)
    return out


def load_from_path(path, board_size, max_games_per_tgz=0, v7=False,
                games_npz=None):
    """加载训练数据。

    - ``v7=True``：**必须**是 ``stdata_to_npz.py`` 产出的 V7 分片
    （``spatial_packed`` + 全部 V7 标签），走 `V7PackedDataset`。
    这条路是 V7 训练**唯一**可行的：``data/sgf_19x19_full.npz`` 那 3420 万行
    虽然更���，但**没有** ``ownership`` / ``futurepos`` / ``scoring`` /
    ``seki`` / ``scorebelief`` 标签，而 ``futurepos`` 是段1 四个主目标之一。
    见 `src/data/v7_dataset.py` 模块 docstring。
    - 否则：若 path 是文件，按 .npz 加载（兼容原行为）；若是目录，递归扫描
    其下所有 .tgz/.tar.gz（用 build_dataset.build 解析）与 .npz（直接加载），
    合并成一个 SupervisedDataset。这样可直接喂一个装着多个分片 tgz 的文件夹，
    无需先手动 build_dataset 成单个 npz。
    """
    if v7:
        # `--v7 1` 下有两种**布局**，都走同一个 22 通道模型（5,562,121 参数）：
        #
        #   stdata 分片（含 spatial_packed）→ `V7PackedDataset`
        #       22 通道已预算好并位打包，每行自带 KataGo 搜索分布（C 段用）
        #   board 级语料（含 boards/my_hist/moves）→ `V7Dataset`
        #       22 通道在训练时实时算，policy 目标是人类着法 one-hot（A/B 段用）
        #
        # 两者不是替代关系，是**同一模型的不同数据源** —— 这样 A→B→C 才能
        #   用 `load_state_dict` 连续承接（12 通道与 22 通道的 stem 形状不同，
        #   跨不过去）。
        #
        # 判据用「文件里有没有 `spatial_packed`」而不是文件名：目录模式下
        # 一个目录里可能同时躺着两种分片。
        import numpy as _np

        def _looks_packed(p: str) -> bool:
            try:
                with _np.load(p, allow_pickle=False) as z:
                    return 'spatial_packed' in z.files
            except Exception:
                return False

        cands = ([path] if os.path.isfile(path)
                else sorted(glob.glob(os.path.join(path, '**', '*.npz'),
                                    recursive=True)))
        packed = [p for p in cands if _looks_packed(p)]
        if packed:
            from src.data.v7_packed_dataset import load_v7_packed

            ds = load_v7_packed(packed[0] if len(packed) == 1 else path)
            print(f"[data] V7 分片（预算特征）：{ds.describe()}")
            return ds
        # board 级：交给下面的常规路径，但升级成 V7Dataset
        v7_board_level = True
    if os.path.isdir(path):
        tgzs = (sorted(glob.glob(os.path.join(path, '**', '*.tgz'), recursive=True))
                + sorted(glob.glob(os.path.join(path, '**', '*.tar.gz'), recursive=True)))
        npzs = sorted(glob.glob(os.path.join(path, '**', '*.npz'), recursive=True))
        # 直接子目录也作为数据源（build 支持递归解析目录内的 .sgf，
        # 例如 data/games/ 这种已解压的棋谱目录）
        subdirs = sorted(d for d in glob.glob(os.path.join(path, '*'))
                        if os.path.isdir(d) and d not in npzs)
        dicts = []
        n_games_total = 0
        n_skip_total = 0

        def _try_build(src):
            """build 可能对单个分片返回 0 有效局并抛 RuntimeError，这里吞掉并跳过。"""
            _log = logging.getLogger('train')
            try:
                d, n_games, skip = build(src, board_size, max_games_per_tgz)
            except RuntimeError as e:
                _log.warning("[data] 跳过分片 %s：%s", src, e)
                return None, 0, 0
            if n_games == 0:
                _log.warning("[data] 跳过分片 %s：0 有效局（与 --board-size %d 不匹配或空）",
                            src, board_size)
                return None, 0, 0
            return d, n_games, skip

        for tg in tgzs + subdirs:
            d, n_games, skip = _try_build(tg)
            if d is not None:
                dicts.append(d)
                n_games_total += n_games
                n_skip_total += skip
        print(f"[data] 已解析 {len(dicts)} 个有效分片，有效局 {n_games_total}，跳过 {n_skip_total}")
        for npz in npzs:
            dd = np.load(npz, allow_pickle=False)
            dicts.append({k: dd[k] for k in dd.files})
        if not dicts:
            raise RuntimeError(
                f"目录 {path} 下未解析到任何有效棋谱，请检查 --board-size 是否与棋谱尺寸匹配")
        merged = _concat_dicts(dicts)
        print(f"[data] 合并后样本数 {merged['boards'].shape[0]}")
        # 同 load_dataset：平面通道数随 KATAGO_SE_CFG 走（与模型侧同改）
        if locals().get('v7_board_level'):
            from src.data.v7_dataset import V7Dataset
            return V7Dataset(merged, n_channels=KATAGO_SE_CFG['in_channels'],
                            **_sidecar_kwargs(merged, games_npz))
        return SupervisedDataset(merged, n_channels=KATAGO_SE_CFG['in_channels'])
    if locals().get('v7_board_level'):
        from src.data.v7_dataset import V7Dataset
        # `load_dataset` 返回的是**已构造好的 SupervisedDataset**，不是 dict。
        # V7Dataset 是它的子类且只多两样东西（重建历史列 + 22 通道装配），
        # 所以直接搬它已加载好的数组，比重新 np.load 一遍 469 MB 划算得多。
        base = load_dataset(path)
        return V7Dataset({k: getattr(base, k) for k in (
            'boards', 'my_hist', 'op_hist', 'ko', 'moves', 'values', 'to_play',
            'game_ids', 'game_weights', 'winrates') if hasattr(base, k)},
            n_channels=KATAGO_SE_CFG['in_channels'],
            **_sidecar_kwargs(base, games_npz))
    return load_dataset(path)


_worker_dataset = None  # multiprocessing worker 进程中由 initializer 设置


def _prefetch_worker_init(dataset):
    """multiprocessing worker 初始化：在子进程中保存 dataset 引用。"""
    global _worker_dataset
    _worker_dataset = dataset


# ---- A3 · labels_dict 的主进程侧工具 --------------------------------------
# **必须放在 `_prefetch_worker` 之前**：`tests/test_prefetch_fork_order.py::
# test_prefetch_workers_do_not_touch_cuda_or_npu` 对源码做的是**切片**扫描 ——
# 它取 worker 函数定义处到 `_BatchPrefetcher` 类定义处之间的那一段，禁止其中
# 出现设备相关字面量。下面这几个函数里 `.to(device)` 是**必需**的（张量转换
# 只允许发生在主进程，见 A3），放进切片会让那条测试变红 —— 而那条测试要守的
# 判据是「worker 纯 numpy」，这里不是 worker。
# 这段注释本身也不能出现 worker 定义的那一行源码：切片用的是 `str.find`，
#   第一个匹配就会被注释里的字面量抢走。
# --------------------------------------------------------------------------


def _concat_label_dicts(dicts):
    """把各子块的 `labels_dict` 沿 **batch 轴**拼成整批（顶层入口）。

    `w` 是**嵌套 dict** ⇒ 拼接必须递归（见 `_concat_tree`）。键集合必须
    一致：同一份 dataset 造出来的 dict 形状恒定（`SupervisedDataset.
    _build_labels`），所以这里做**严格**校验而不是取交集 —— 少一个键就是
    「某个子块用了旧契约」，宁可炸。
    """
    if not dicts:
        raise ValueError('_concat_label_dicts: 没有可拼的子块')
    return _concat_tree(dicts)


def _concat_tree(vals):
    """递归拼接：叶子是 numpy（沿 batch 轴），中间层是 dict。

    `vals` 是「同一层、同一父键」在各子块里的取值列表。**键集合在每一层都做
    对称校验**（不只查「后者缺前者有的」，也查「后者多出前者没有的」）：少一个
    `w` 里的权重项同样会被静默丢掉，而症状是「某个权重恒 0」，没人会想到是
    拼接时丢的。
    """
    out = {}
    keys0 = set(vals[0])
    for v in vals[1:]:
        if set(v) != keys0:
            raise KeyError(
                f'labels_dict 键集合在子块之间不一致：缺 {sorted(keys0 - set(v))}、'
                f'多 {sorted(set(v) - keys0)}（契约分叉）')
    for k, v0 in vals[0].items():
        if isinstance(v0, dict):
            out[k] = _concat_tree([v[k] for v in vals])
        else:
            out[k] = np.concatenate([v[k] for v in vals], axis=0)
    return out


def _labels_dict_to_tensors(d, device=None):
    """numpy 的 `labels_dict` → 同 device 的张量 dict（**递归**）。

    与 `SupervisedDataset.sample_batch(labels=True)` 用同一套口径（含嵌套的
    `w`）：**只有一种 payload 形状**是 A3 的全部意义。
    """
    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        # 标签张量来自 pageable 的 numpy 内存：直接 `.to(device, non_blocking=True)`
        # 在 CUDA 上会对**未固定**内存发起异步 H2D，偶发 `CUDA error: misaligned
        # address`（见 2330 附近崩溃栈，全仓其它搬运路径都不这么写）。与
        # `v7_to_device` / 12 通道路径同口径：CUDA 上先 `pin_memory()` 再非阻塞搬运；
        # 其余后端（非 CUDA）走普通 `.to`。`ascontiguousarray` 防御非连续数组。
        t = torch.from_numpy(np.ascontiguousarray(x))
        if device is not None and str(device).split(':')[0] == 'cuda':
            return t.pin_memory().to(device, non_blocking=True)
        return t.to(device) if device is not None else t
    return {k: conv(v) for k, v in d.items()}


def v7_batch_sync(dataset, idxs, device=None, *, rng=None, augment=True):
    """同步取一个 V7 batch（`--prefetch-workers <= 1` 时的回退路径）。

    返回 ``(spatial, gl, moves, labels_dict)``，前两项仍是 numpy（与预取器路
    径**同形**），由 `v7_to_device` 统一搬到设备上 ⇒ 两条取批路径的下游代码
    完全一样，回退时不会出现「一边是 numpy 一边是张量」的分支。
    """
    idxs = np.asarray(idxs, dtype=np.int64)
    if rng is None:
        rng = np.random.default_rng()
    tforms = (rng.integers(0, 8, size=idxs.size) if augment
            else np.zeros(idxs.size, dtype=np.int64))
    moves, lbl = _v7_labels_and_moves(dataset, idxs, tforms)
    sp, gl = v7_batch_features(dataset, idxs)
    return _dihedral_batch(sp, tforms), gl, moves, lbl


def v7_to_device(x, device, amp_dtype, *, pin=False):
    """V7 输入的 H2D：AMP 下保持 planes 的 **fp16**，全精度才升 fp32。

    与主循环里 12 通道那段**同口径**（同一条「不白带一倍 H2D 字节数」的判据）。
    V7 走的是默认 contiguous 布局、**没有** channels_last 那一摊，所以不需要
    numpy 端转置；`pin` 只在 CUDA 上开（加速器/CPU 的 pin_memory 语义不同，
    12 通道那条路也是只在 `_backend == 'cuda'` 时 pin）。
    """
    t = torch.from_numpy(np.ascontiguousarray(x))
    t = t.float() if amp_dtype == torch.float32 else t
    if pin:
        t = t.pin_memory()
    return t.to(device, non_blocking=True)


def _prefetch_worker(wi, task_q, res_q, seed, dataset, labels=False, v7=False,
                    augment=True):
    """multiprocessing worker：从 task_q 取任务，计算后放 res_q。

    **本函数里不许出现任何设备相关调用**（两个后端的运行时入口、跨设备搬运、
    低精度上下文管理器）—— 见 `tests/test_prefetch_fork_order.py::
    test_prefetch_workers_do_not_touch_cuda_or_npu`，它对本段源码做**字面**
    扫描（注释也算，所以这句描述刻意避开被禁的字面量）。理由是 worker 只做纯
    numpy：碰一下设备上下文就会让每个 worker 建一份映射（4 卡实测每卡凭空多占
    ~24 GiB ⇒ OOM）。
    同理不许在这里重新打开主 npz —— 父进程已经把列读进内存了（fork 复制
    地址空间），每个 worker 再解一次 `boards` 就是 12.3 GB × k。数据只能走
    dataset 引用（`dataset` 形参）。

    `labels=True` 时 payload 是 **4 元组 + dict**（spec §5.5），第 4 项是
    numpy 的 `labels_dict`（`w` 是**嵌套 dict**）；张量转换留到主进程的
    `next()` —— 在 worker 里转会把设备上下文也带出去，正是上面那条禁令。

    ---- `v7=True` 的第二条装配路径 ----------------------------------------
    payload 变成 ``(step, pos, spatial, gl, moves, lbl, None)``（第 3/4 项是
    22 通道与 19 维的 V7 输入，取代旧路径的第 2/3 项 `states`/`values`）。

    **开头那次 `_rebind_futurepos_mmap` 就是本函数存在的理由之一**：Windows
    上 `mp.Process` 是 **spawn** 而不是 fork，spawn **不继承内存** —— 父进程里
    `warm_futurepos()` 解析好的映射句柄到不了子进程（pickle 要么报错，要么把
    12.3 GB 当 ndarray 整份搬过去）。所以 worker 里**重新 open 一次**，而不是
    继承句柄；它走公开 API，**只重开、不落盘**（落盘只在父进程发生一次 ——
    理由与被禁止的字面量清单见 `_rebind_futurepos_mmap` 的 docstring，以及
    `tests/test_train_sft_v7.py::test_worker_reopens_mmap_instead_of_inheriting`
    / `::test_worker_does_not_rewrite_the_boards_file`）。
    这段 docstring 刻意避开若干字面量：`tests/test_prefetch_labels.py::
    test_worker_source_never_reopens_the_dataset` 对**同一个切片**做字面扫描
    （连注释一起扫），列出的那些词一个都不许出现在 worker 段里。

    **`augment` 的对称增强与标签必须同源**：本函数抽一份 `tforms`，然后把它
    **原样**同时交给 22 通道空间量与 `_build_labels`，后者会同步重映射
    `next_move` / `soft` / `future` ⇒ 输入与标签永远出自同一份变换。
    这里**不能**改用 `dataset.sample_batch_numpy(..., labels=True)` 来拿标签：
    那份 `tforms` 是它内部抽的、本函数拿不到；而「自己再抽一份」是静默错标签
    （输入翻了、标签没翻 ⇒ loss 照降、棋力不涨）。见 `_v7_labels_and_moves`。
    """
    # 每次进 worker 只解析一次 boards 来源（幂等：后续 batch 复用同一个映射）。
    v7_boards = _rebind_futurepos_mmap(dataset) if v7 else None
    rng = np.random.default_rng(seed + wi)
    while True:
        item = task_q.get()
        if item is None:
            return
        step, pos, sub_idx = item
        try:
            if v7:
                b = len(sub_idx)
                # augment=False ⇒ 恒等变换（eval 专用：eval 契约是 augment=False，
                # 且输入/标签必须同源不增强 —— 复用本 worker 而不另写一份的原因）。
                tforms = (rng.integers(0, 8, size=b) if augment
                        else np.zeros(b, dtype=np.int64))
                moves, lbl = _v7_labels_and_moves(dataset, sub_idx, tforms)
                spatial, gl = v7_batch_features(dataset, sub_idx,
                                                boards=v7_boards or None)
                res_q.put((step, pos, _dihedral_batch(spatial, tforms), gl,
                        moves, lbl, None))
            else:
                out = dataset.sample_batch_numpy(sub_idx, rng=rng, labels=labels)
                if labels:
                    s, m, v, lbl = out
                else:
                    s, m, v = out
                    lbl = None
                res_q.put((step, pos, s, m, v, lbl, None))
        except Exception as e:  # noqa: BLE001
            res_q.put((step, pos, None, None, None, None, e))


def _v7_labels_and_moves(dataset, idxs, tforms):
    """V7 路径的「本行着法 + 标签」，**用调用方给的那份 `tforms`**。

    为什么不用 `dataset.sample_batch_numpy(..., labels=True)`
    -------------------------------------------------------
    那条路径内部自己抽 `tforms` 并同时施加到 `states` 与 `_build_labels`，调用方
    **拿不到那份 `tforms`**。V7 需要用同一份去变换 22 通道空间量，于是只有两种
    选择：(a) 反推，(b) 自己抽。选 (b)。

    (a) 反推（比对 8 个候选变换的结果）有一个**静默失败**的洞：当盘面本身对某组
    变换不变时（空盘、对称局面；训练早期大量如此），8 个候选输出**完全相同**，
    反推会挑一个**不一定等于真值**的 `t`，而标签是按真值重排过的 ⇒ 输入与标签
    不同源。这类错不报错、loss 照降、只有棋力不涨。

    ⇒ 这里显式抽 `tforms`，并把它同时喂给空间量与 `_build_labels`。
    `_build_labels` 是本仓唯一的标签构造器（`sample_batch_numpy` 的 labels 分支
    就是 `self._build_labels(idxs, tforms, augment)`，一字不差），而本任务不得
    改动 `src/data/dataset.py`，所以这是唯一能拿到「与输入同源的那份 tforms」的
    入口。`tests/test_train_sft_v7.py::test_v7_labels_match_dataset_under_same_tform`
    把本函数与 `sample_batch_numpy` 在**固定 tforms 下**逐项对拍，防它漂移。

    `moves` 的口径逐字照抄 `sample_batch_numpy`：非法/越界归一到 `bs*bs`（pass 类），
    且**只有** `0 <= mv < bs*bs` 被重映射。
    """
    from src.data.dataset import permute_move_vector
    bs = dataset.board_size
    idxs = np.asarray(idxs, dtype=np.int64)
    # `V7PackedDataset` 每行**自带**该行的目标着法，`_build_labels` 已经把它
    #   放进 `next_move`（并按同一 tform 重编号过）。若在这里再走 board 级路径的
    #   `moves[idxs+1]`，会取到**下一行**的答案 —— 监督信号整体错位一行，
    #   而形状完全合法、不报错。所以本类必须直接用 `next_move`。
    if hasattr(dataset, 'sample_spatial'):
        lbl = dataset._build_labels(idxs, tforms, True)
        moves = np.asarray(lbl['next_move'], dtype=np.int64).copy()
        return moves, lbl

    moves = np.full(len(idxs), bs * bs, dtype=np.int64)
    mv = np.asarray(dataset.moves[np.asarray(idxs, dtype=np.int64)],
                    dtype=np.int64)
    valid = (mv >= 0) & (mv < bs * bs)
    moves[valid] = mv[valid]
    permute_move_vector(moves, tforms, bs)
    return moves, dataset._build_labels(np.asarray(idxs, dtype=np.int64),
                                        tforms, True)


def _dihedral_batch(x, tforms):
    """对 ``(B,C,H,W)`` 施加逐行 8 路 dihedral 变换（4 旋转 × 2 镜像）。

    与 `dataset.sample_batch_numpy` 的向量化增强**同一套约定**：
    `t >= 4` 先翻 W 轴（`[..., ::-1]`），`k = t % 4` 再顺时针转 `k`（numpy 的
    `np.rot90` 是逆时针，故取 `-k`）。口径分叉的症状与漏同步重排着法完全一样
    （静默错标签），所以这段复制品在这里显式写出来并由
    `tests/test_train_sft_v7.py::test_dihedral_matches_dataset_augmentation`
    与 dataset 的实现对拍。
    """
    x = np.asarray(x)
    out = np.empty_like(x)
    for t in range(8):
        mask = tforms == t
        if not mask.any():
            continue
        arr = x[mask]
        if t >= 4:
            arr = arr[:, :, :, ::-1]
        k = t % 4
        if k:
            arr = np.rot90(arr, k=-k, axes=(2, 3))
        out[mask] = arr
    return out


class _BatchPrefetcher:
    """后台多进程并行构造训练 batch，与 加速器 前向/反向重叠。

    把每个 batch 的样本下标切成 num_workers 个子块，由 num_workers 个后台进程
    并行调用 dataset.sample_batch_numpy()（绕过 GIL，numpy 操作真正并行），
    主进程按序拼回整批。

    两步流水：submit() 投递一个 batch 的下标，next() 取回构造好的 numpy 数组。
    两个**有界**队列提供背压，避免无限预取吃内存。各进程用独立 np.random.Generator。

    `labels=True`（软标签接入 A3）：`next(device=...)` 额外返回第 4 项
    `labels_dict`（张量 dict，含嵌套 `w`）；`labels=False`（默认）时
    `next()` 的返回值与改造前**逐位一致**（仍是 numpy 三元组）—— 段 1 的
    热路径不因软标签接线多搬任何字节。

    `v7=True`（V7 接线）：worker 改走 `v7_batch_features`，`next()` 返回
    ``(spatial, global_features, moves, labels_dict)``。 `v7=True` 会**强制**
    带上 labels（V7 的 loss 没有 `labels_dict` 算不出任何一项），所以这一条
    取代而不是叠加 `labels`；`v7=False`（默认）时本类行为**逐位不变**。
    """

    def __init__(self, dataset, num_workers=4, prefetch=2, seed=1234,
                labels=False, v7=False, augment=True):
        # 护栏：**绝不能在设备运行时初始化之后**构造本类（4 卡 旧多卡环境 的 OOM
        # 直接原因，2026-09-30）。`mp.Process` 默认 fork，子进程会整份继承父
        # 进程的 /CUDA 上下文与已分配显存映射 ⇒ 每卡被旁挂 4 份 ≈ 24 GiB，
        # 而 PyTorch 自己只记 6.3 GB（实测 HBM 94% / 计算单元 0%）。GC 管不到
        # 别的进程继承来的映射。正确做法见 main()：数据集加载与本类的构造都在
        # `init_process_group` / `set_device` **之前**。
        for _dev in ('cuda',):
            _is_init = getattr(getattr(torch, _dev, None), 'is_initialized', None)
            try:
                _already = bool(_is_init()) if _is_init is not None else False
            except Exception:  # noqa: BLE001 — 拿不到就当作没初始化，不拦
                _already = False
            if _already:
                raise RuntimeError(
                    '_BatchPrefetcher 不能在 torch.{0} 初始化之后构造：'
                    'fork 出的 worker 会继承设备上下文，4 卡实测每卡凭空多占 '
                    '~24 GiB（OOM）。请把它挪到 init_process_group / '
                    'set_device 之前。'.format(_dev))
        self.dataset = dataset
        self.k = max(1, int(num_workers))
        self.prefetch = max(1, int(prefetch))
        # 背压语义不可动：两个队列都必须**有界**（maxsize = k·depth）。
        #   改成无界队列 = 无限预取 = 预取深度失控时把内存吃光（fp16 输入
        #   12 路 × B=512 × 19² × 19² 在 flight 里就能到 GB 级）。
        cap = self.k * self.prefetch
        self._task_q: mp.Queue = mp.Queue(maxsize=cap)
        self._res_q: mp.Queue = mp.Queue(maxsize=cap)
        self.labels = bool(labels)
        self.v7 = bool(v7)
        self.augment = bool(augment)
        self._step = 0     # 下一个待投递 batch 的编号
        self._expect = 0   # 下一个待取回 batch 的编号
        self._pending: dict = {}  # step -> [(pos, s, m, v, lbl, err)]
        self._processes = []
        for wi in range(self.k):
            p = mp.Process(
                target=_prefetch_worker,
                args=(wi, self._task_q, self._res_q, seed, dataset,
                    self.labels, self.v7, self.augment),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

    def submit(self, idxs):
        """投递一个 batch 的下标（队列满时阻塞，提供背压）。"""
        step = self._step
        self._step += 1
        n = len(idxs)
        for wi in range(self.k):
            start = wi * n // self.k
            end = (wi + 1) * n // self.k
            sub = idxs[start:end]
            if len(sub) == 0:
                # 空子块直接回一个空 payload（不投 task_q）：否则每个空块都要
                # 占一个 worker 的往返，而 worker 的往返是要抢 GIL 的。
                self._res_q.put((step, wi, None, None, None, None, None))
            else:
                self._task_q.put((step, wi, sub))

    def next(self, device=None):
        """取回下一个 batch。

        `labels=False`（默认）→ ``(states_np, moves_np, values_np)``，与改造前
        逐位一致（**纯 numpy**，不碰 device）。

        `labels=True` → 上面三项 + `labels_dict`（同 device 的张量 dict，
        含嵌套 `w`）。`device=None` 时 dict 里的张量留在 CPU。

        `v7=True` → ``(spatial_np, gl_np, moves_np, labels_dict)``：
        第 1/2 项是 22 通道与 19 维的 V7 输入，**取代** `states` / `values`
        （`moves` / `labels_dict` 的口径与上面两条路完全一致，见 `_prefetch_worker`）。
        `labels` 被强制视为 True —— V7 的 loss 需要 `labels_dict`，没有它算不出
        任何一项。
        """
        step = self._expect
        self._expect += 1
        # 从 _pending 中取出之前缓存的该 step 结果
        parts = self._pending.pop(step, [])
        # 如果不够 k 个，从队列中继续收
        while len(parts) < self.k:
            r_step, pos, s, m, v, lbl, err = self._res_q.get()
            if r_step == step:
                parts.append((pos, s, m, v, lbl, err))
            else:
                # 缓存未来 step 的结果
                self._pending.setdefault(r_step, []).append(
                    (pos, s, m, v, lbl, err))
        # 检查错误
        for pos, s, m, v, lbl, err in parts:
            if err is not None:
                raise err
        # 按 pos 排序并拼接
        parts = [(pos, s, m, v, lbl) for pos, s, m, v, lbl, _ in parts
                if s is not None]
        parts.sort(key=lambda x: x[0])
        if not parts:
            raise RuntimeError(f"step {step}: 所有子块为空")
        states = np.concatenate([p[1] for p in parts], axis=0)
        moves = np.concatenate([p[2] for p in parts], axis=0)
        values = np.concatenate([p[3] for p in parts], axis=0)
        labels_dict = _concat_label_dicts([p[4] for p in parts]) if (
            self.labels or self.v7) else None
        if self.v7:
            # worker 的 payload 是 (step, pos, spatial, gl, moves, lbl, err)，而
            # 上面三行的解包位置固定把 s/m/v 收进来 ⇒ 这里 s=spatial、m=gl、
            # v=moves。名字改成 V7 的口径再返回，别让上面三行误导读代码的人。
            spatial = states
            gl = moves
            mv = values
            return (spatial, gl, mv,
                    _labels_dict_to_tensors(labels_dict, device=device))
        if not self.labels:
            return states, moves, values
        return (states, moves, values,
                _labels_dict_to_tensors(labels_dict, device=device))


def resolve_c2net_data(dataset_path):
    """把 C2NET 上下文的数据集路径转成 ``--data`` 可直接使用的值。

    目录 → 原样返回，让 ``--data`` 的目录模式（load_from_path）递归扫描并
    **合并全部** .npz/.tgz 分片；文件 → 原样返回。
    不可直接把目录展开成 glob 再取 [0]：那只会加载一个分片，且 glob 顺序依赖
    文件系统，未排序时选中哪个不确定。
    """
    return dataset_path


def _init_swanlab(args, logger):
    """初始化 SwanLab 跟踪，返回 swanlab 模块；失败返回 None（训练照常进行）。

    单独抽出来是为了让 main() 保持扁平：main 只负责"是否启用 + 能否连通"的
    决策，本函数只负责"装/登录/init"。任何异常都吞掉并降级为 None。
    """
    try:
        # 刻意**不在训练进程内** pip install swanlab。调用点在 torch
        # 已加载之后，此时改动 site-packages 可能破坏后续惰性导入；而 shell/*.sh
        # 在启动 python 之前已装过一次，那次失败的话这里必然也失败，只是白等一轮。
        # 「手动安装没问题」正是这个差别：装在解释器启动前，依赖已就位。
        #
        # 用 find_spec 区分「没装」与「装了但坏」：后者是云端常见坑——swanlab 依赖
        # pydantic>=2，而部分环境常把 pydantic 钉在 1.x，于是
        # `import swanlab` 抛 "cannot import name 'TypeAdapter' from 'pydantic'"。
        # 旧代码把任何 ImportError 都当成「未安装」而误触发自动安装，掩盖了真因。
        _mod = sys.modules.get('swanlab')
        if _mod is None:
            import importlib.util
            try:
                _spec = importlib.util.find_spec('swanlab')
            except (ImportError, ValueError):
                _spec = None
            if _spec is None:
                logger.warning("[swanlab] 未安装，已跳过跟踪（指标请看 stdout 日志）。"
                            "请在启动训练前安装: pip install swanlab")
                return None
        import swanlab
        # 登录：优先用 --swanlab-api-key，其次环境变量，最后交互式
        api_key = args.swanlab_api_key or os.environ.get('SWANLAB_API_KEY')
        if api_key:
            swanlab.login(api_key=api_key, save=True)
            logger.info("[swanlab] API key 已设置，自动登录")
        # config 面板记的必须是**真值**：结构一律取 `KATAGO_SE_CFG`（唯一真相源），
        # **不记**任何已归档的 CLI 结构 flag（backbone_channels / res_blocks /
        # convnext_blocks / attn_blocks / value_channels / value_res_blocks /
        # policy_channels / policy_layers …）—— 它们完全不参与建网，记进去就是
        # 虚构值，而 config 面板恰恰是对比两次 run 时第一个看的东西。
        # `blocks` 那个描述串同理：从真形状数出来（se×13 + attn×4），不是抄配置。
        _ws = max(1, int(os.environ.get('WORLD_SIZE', '1') or '1'))
        _accum = max(1, int(args.gradient_accumulation_steps))
        swanlab.init(
            project="go-ai",
            name=f"sft_{args.board_size}x{args.board_size}_{args.ver}",
            config={
                # ---- 结构：新架构真值（唯一真相源是 KATAGO_SE_CFG）----
                # **按 `--v7` 分派**（2026-10-04）：此前这一段无条件记 12 通道的
                #   数字，于是**每个 V7 run 的 config 面板都写着
                #   `in_channels: 12` / `params_total: 9,112,005`**，而它实际训练的是
                #   22 通道 / 5,562,121。面板恰恰是对比两次 run 时第一个看的东西，
                #   给它虚构值比不给更糟（正是本段原注释要防的那件事，只是当时
                #   只考虑了「归档 flag」，没考虑「另一个架构」）。
                # V7 的结构取 `NBT_TF_CFG`（唯一真相源），**不引 args** ——
                #     args 里那些是归档 flag，引用它们等于把面板填成虚构值。
                # `arch/params_total` 在 V7 下取 `NBT_TF_CFG` 的**预算值**
                #     5,561,832，而实测建出来的模型是 **5,562,121**（差 289，
                #     见 `tests/test_katago_v7_budget.py`）。面板记预算、真实值由
                #     run 级 `run/params_actual` 给（那里模型已经建好）。
                "arch/v7": bool(args.v7),
                "arch/in_channels": (NBT_TF_CFG['in_channels'] if args.v7
                                    else KATAGO_SE_CFG['in_channels']),
                "arch/name": ("nbt_tf" if args.v7 else KATAGO_SE_CFG['arch']),
                "arch/channels": (NBT_TF_CFG['trunk_channels'] if args.v7
                                else KATAGO_SE_CFG['channels']),
                # V7 另有一份 19 维全局输入（12 通道路径没有这个概念）
                "arch/global_channels": (NBT_TF_CFG['global_channels']
                                        if args.v7 else 0),
                # 段内块描述串：从真形状数出来（12 通道是 se×13 + attn×4，
                # V7 是 nbt2 × 11 段、每段含 2 个 inner 块），不是抄配置
                "arch/blocks": (("nbt%d_inner%d_x%d"
                                % (NBT_TF_CFG['num_blocks'],
                                    NBT_TF_CFG['num_inner_blocks'],
                                    NBT_TF_CFG['board_size']))
                                if args.v7
                                else "se13_attn4_of_%d" % KATAGO_SE_CFG['blocks']),
                "arch/attention_mode": (("nbt+transformer_%dhead"
                                        % NBT_TF_CFG['num_heads']) if args.v7
                                    else KATAGO_SE_CFG['attention_mode']),
                "arch/num_attention_layers": (
                    NBT_TF_CFG['num_blocks'] if args.v7
                    else KATAGO_SE_CFG['num_attention_layers']),
                "arch/num_heads": (NBT_TF_CFG['num_heads'] if args.v7
                                else KATAGO_SE_CFG['num_heads']),
                # 注意力块里 FFN 的中间维 = 2×通道（12 通道）／显式 384（V7）
                "arch/attn_ffn_hidden": (NBT_TF_CFG['ffn_hidden'] if args.v7
                                        else KATAGO_SE_CFG['channels'] * 2),
                # SE 瓶颈块的中段宽 = 通道/2（SEBottleneck 默认 mid_channels）；
                # V7 没有这个结构，填 0 而不是编一个
                "arch/se_mid_channels": (0 if args.v7
                                        else KATAGO_SE_CFG['channels'] // 2),
                "arch/params_total": (NBT_TF_CFG['params_total'] if args.v7
                                    else KATAGO_SE_CFG['params_total']),
                "arch/params_backbone": (NBT_TF_CFG['params_blocks_total']
                                        if args.v7
                                        else KATAGO_SE_CFG['params_backbone']),
                # ---- V7 特有的头/输出维度（12 通道路径一律 0）----
                # 键名**刻意避开** `value_channels` / `policy_channels`：
                #   `tests/test_swanlab_metrics.py::test_config_panel_has_no_archived_flags`
                #   按**引号内字面量**判归档 flag 泄漏，写 `NBT_TF_CFG['value_channels']`
                #   会让那条门禁误报（它要禁的是 args 上的归档开关，不是 V7 的
                #   真实结构键）。这里改用 V7 自己的键名，不改名也不绕。
                "arch/seki_classes": (NBT_TF_CFG['seki_classes'] if args.v7 else 0),
                "arch/futurepos_ch": (NBT_TF_CFG['futurepos_channels']
                                    if args.v7 else 0),
                "arch/score_distr_bins": (2 * (NBT_TF_CFG['board_size'] ** 2
                                            + NBT_TF_CFG['extra_score_distr_radius'])
                                        if args.v7 else 0),
                "arch/policy_outputs": (NBT_TF_CFG['policy_outputs']
                                        if args.v7 else 1),
                # 同样按 `--v7` 分派：V7 用 `NBT_TF_CFG['use_checkpoint']`（也是
                #   唯一真相源），此前这个键无条件取 12 通道那张表。
                "grad_checkpoint": (NBT_TF_CFG['use_checkpoint'] if args.v7
                                    else KATAGO_SE_CFG['grad_checkpoint']),
                # ---- batch/lr：有效 batch 含累积（漏乘会把学习率口径搞错）----
                "batch_size_per_card": args.batch_size,
                "grad_accum": _accum,
                "world_size": _ws,
                "effective_batch": args.batch_size * _ws * _accum,
                "lr": args.lr,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "clip_grad_max_norm": 1.0,
                # ---- 数值/正则 ----
                "policy_loss": args.policy_loss,
                "value_loss": args.value_loss,
                "value_loss_weight": args.value_loss_weight,
                "huber_beta": args.huber_beta,
                "label_smoothing": args.label_smoothing,
                # ---- 软标签（A4）----
                # policy_loss 在这里已是**派生后**的取值（main() 在加载数据集
                #   之前就调了 resolve_policy_loss_kind），所以 soft_ce 的 run
                #   不会在 config 面板里显示成 'ce'。
                "soft_index": args.soft_index or "",
                "soft_weight": args.soft_weight,
                "soft_only_sampling": bool(args.soft_only_sampling),
                "soft_every": args.soft_every,
                "attention_dropout": args.attention_dropout,
                "ema_enabled": bool(args.use_ema),
                "ema_decay": 0.999,
                # ---- AMP：旧多卡环境 无 BF16，FP16 必须配 GradScaler ----
                "amp_dtype": "float16",
                "scaler_init_scale": args.scaler_init_scale,
                "scaler_growth_interval": args.scaler_growth_interval,
                # ---- 注意力内核（决定显存口径，见 backbone._sdpa）----
                "attn_sdpa_force_math": True,
                # online-softmax 注意力开关：登记进来是为了**面板能看到本次到底
                # 开没开**。静默的开关是排查噩梦 —— 尤其它与 materialize 路径
                # 非逐位相同，出了问题得先知道它开过。
                "attn_online": int(args.attn_online),
                # SDPA（ 融合注意力）开关：登记进来同样为了面板可见 ——
                # 它直接决定注意力走融合内核还是手写 math，是 加速器 上最关键的
                # 显存/速度/数值口径之一。
                "use_sdpa": int(args.use_sdpa),
                "attn_query_chunk": int(args.attn_query_chunk or 0),
                # ---- 数据/运行 ----
                "data": args.data,
                "board_size": args.board_size,
                # ---- 局级 sidecar（2026-10-04）----
                # A/B 段不给它 ⇒ 逐局贴目按 0 ⇒ 全局 ch5 恒 0、ch18 三角波走偏。
                # 面板记它是为了「对比两次 run」时能一眼看出那次是不是忘了给。
                "data/games_npz": (args.games_npz or "") if args.v7 else "",
                # C 段（stdata 分片）的贴目在行里，多给无意义 ⇒ 这里记 0 表示
                # 「不适用」，而不是把空串与「忘了给」混成同一个值
                "prefetch_workers": args.prefetch_workers,
                "prefetch_depth": args.prefetch_depth,
                        },
        )
        logger.info("[swanlab] 实验跟踪已启用")
        return swanlab
    except Exception as e:  # noqa: BLE001
        logger.warning("[swanlab] 初始化失败: %s（指标请看 stdout 日志）", e)
        _emsg = str(e)
        if 'pydantic' in _emsg or 'TypeAdapter' in _emsg:
            logger.warning("[swanlab] 疑似 pydantic 版本冲突：swanlab 需要 pydantic>=2，"
                        "而部分环境常钉 pydantic<2。"
                        "请在启动训练前解决版本冲突（如在独立环境装 swanlab），"
                        "不要依赖训练进程内自动安装。")
        return None


def _should_log(step, log_every, swanlab_every, swanlab_on):
    """决定本步是否打 stdout、是否上报 SwanLab。

    ``swanlab_every <= 0`` 表示跟随 ``log_every``（默认值 0 = 与旧行为一致）。
    两者解耦的目的：stdout 是给人看的（保持 ``log_every``，避免刷屏），
    swanlab 是给曲线用的（可独立加密，``--swanlab-every 5`` 比 50 密 10 倍）。

    ``swanlab_on`` 为 False（如 swanlab 不可用）时第二项恒为 False，
    确保不产生任何额外同步。
    """
    le = max(1, int(log_every))
    se = int(swanlab_every) if swanlab_every and swanlab_every > 0 else le
    se = max(1, se)
    return (step % le == 0, bool(swanlab_on) and (step % se == 0))


def _read_log_scalars(loss, policy_loss, value_loss):
    """日志的**单一同步点**：三个 loss 张量各取一次标量。

    旧实现在同一打点里取了两遍（一次给 stdout、一次给 swanlab），共 6 次设备
    同步，其中 3 次完全重复。这里只取一次并返回给两处复用——数值逐位不变，
    同步次数减半。

    第一个参数传的是 **`log_loss`（报告口径）**，不是被 backward 的
    `opt_loss`。两者相差一个 `l2_report = c‖θ‖²`（P4.5b §3.1），故意不相等；
    形参名沿用 `loss` 是为了不打乱本仓既有的调用/断言写法，含义以上行为准。
    """
    return loss.item(), policy_loss.item(), value_loss.item()


# ---- D4（SFT 侧）：policy/value 损失口径 --------------------------------------
# 三个 CLI 开关：--policy-loss {huber,ce}（**默认 ce**，P4.5b 由 huber 改回 ce）、
# --value-loss {huber,mse}（默认 huber）、--huber-beta（默认 0.5）。C8 修正：value
# 的 BCE 分支已删。RL 侧（scripts/selfplay_train.py）的损失是 P3-C/P3-D，**不经过
# 这里** —— 本文件的函数只服务 train_sft 自己的调用点，改语义不会波及 RL。
#
# 日志契约（不可动）：main() 仍用 loss / policy_loss / value_loss 三个键，下游
# run.txt、看板与 `tests/test_huber_loss.py::test_log_keys_unchanged` 绑着它们，
# tests/test_huber_loss.py::test_log_keys_unchanged 把三个键钉死。`loss` 的**含义**
# 在 P4.5b 变过（多了 c‖θ‖²）、键名没变 ⇒ 跨新旧 run 的曲线不可直接比。
#
# P4.5 遗留收口（用户 2026-09-27 裁决 = P4.5b）：policy 默认改回 ce 之后，「policy
# 梯度天生弱 ~A 倍、共享主干因此 value-only」从根上消失（Huber 打在概率域，梯度
# 带 softmax 雅可比的 p≈1/A；CE 打在 log-prob 上，d(CE)/d(logit)=p−y 有界且与 A
# 无关）。实测 ce+huber、w=1 的 value:policy 梯度比 = 1.113:1，与 P4.5 记录的 ce
# 备选口径逐位一致 ⇒ **不需要补偿旋钮**。
#
# `soft_ce`（软 CE）已实现，但 CLI 的 `--policy-loss` choices 仍冻结在
# ['huber','ce'] —— tests/test_huber_loss.py::test_no_new_cli_params 以 D1
# 「零新增/零删除/零改名」把 61 个 flag 整个钉死。`soft_ce` 与 `--soft-weight`
# 的 CLI 入口属 A4。



def huber_loss(pred, target, beta=0.5, reduction='mean'):
    """Huber（smooth L1）—— D4 的唯一 Huber 实现，口径在本 docstring 钉死。

    数学式（逐元素误差 d = pred − target，N = 元素数）::

        h(d) = 0.5 * d² / beta    若 |d| <  beta      （二次段，∂h/∂d = d/beta）
            = |d| − 0.5 * beta    若 |d| ≥ beta      （线性段，∂h/∂d = sign(d)）
        reduction='mean'  →  (1/N) · Σ h(d)     ← 默认，value 侧用它
        reduction='none'  →  h(d) 逐元素        ← policy 侧自己聚合（见下）

    **「线性段梯度 = ±1」只在 N=1 时成立**（P4.5-fix 更正）。mean 归约把每个
    元素的梯度也除以 N，实测（beta=0.5, d=1）：N=1 → ±1.000000、N=8 → ±0.125000、
    N=2888（B=8 × A=361）→ ±0.000346。上一版 docstring 拿「±1 vs ±0.5」当
    smooth_l1 与 huber_loss 的选型依据，是 `test_huber_limits` 用**单元素张量**
    造出来的假象 —— 在真实的 B×A 张量上，smooth_l1 的线性段梯度是 ±1/N，
    `F.huber_loss` 的是 ±beta/N，两者只差一个统一的 beta 因子。

    实现选择（三问三答，改动前先读完）：

    1. **用 `F.smooth_l1_loss(..., beta=)`，不用 `F.huber_loss(..., delta=)`。**
    D4 简报「两者在 beta ≤ 1 时完全等价」的断言在本仓实测的 torch 2.12.0+cpu
    上**不成立**，torch 自己的 docstring 也这么写（"In general, Huber loss
    differs from SmoothL1Loss by a factor of delta"）。实测（float64、
    d ∈ linspace(-3,3,20001)、逐位比较）：

        F.huber_loss(delta=b) == b · F.smooth_l1_loss(beta=b)     恒等，仅 b=1 时两者相等

    选 smooth_l1 的理由因此**不是**梯度 ±1（那是被 N=1 掩盖的假象），而是
    **量纲约定**：smooth_l1 就是教科书 Huber(delta=b) 本身
    （实测 `smooth_l1(beta=b) ≡ Huber(delta=b)` 逐位相等，拐点确实在 |d|=b），
    而 `F.huber_loss` 是它的 b 倍缩放 —— 用 smooth_l1 时 `--huber-beta`
    的语义与 Huber 的 delta 直觉一致，改 beta 只改拐点、不改整体量纲。
        顺带证伪 fix 简报里的另一句：它把 smooth_l1 说成「delta 取 1.0 的
    Huber 再整体除以 b」，并据此断言拐点固定在 1.0、与 beta 无关 ——
    **同样是错的**：b=0.5 时两者最大差 2.25（线性段不等），仅 b=1 巧合
    相等。拐点就在 |d|=b。逐位 oracle 见
    tests/test_huber_loss.py::test_huber_matches_torch_reference。
    2. **reduction 显式写出**：默认 `'mean'` 与被替换的旧 value 口径
    （`F.mse_loss` 默认 mean）对齐；`'none'` 是 P4.5-fix 给 policy 侧用的
    逃生口 —— policy 必须**逐样本**聚合（类内 sum 后对 batch 取 mean），
    原因见 compute_policy_loss docstring 的 1/A 稀释分析。
    3. **数值稳定性**：|d| ≥ beta 段梯度有界（∝ sign(d)/N，不随误差放大），
    SFT 早期 value 误差可能很大（目标 ±1、初值 ~0 → |d| ~ 1），此时 log
    不会被一次离群样本炸出天文数字（MSE 会：其梯度 ∝ 2|d|）。代价是大误差
    处损失只按 |d| 线性增长 —— 日志上 value_loss 早期是一条**斜率恒定**的
    下降线，收敛末段（|d|<beta）才转成二次的平滑收口。
    """
    return F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)


def soft_cross_entropy(policy_logits, soft, soft_mask, soft_weight=1.0):
    """**软标签**交叉熵：`mean_b( mask_b · (−Σ_a soft_b[a]·log_softmax(logits_b)[a]) )`。

    参数
    ----
    policy_logits : (B, A) —— 模型输出，**logits**（不是概率；归一化在本函数里做）
    soft          : (B, A) —— KataGo 访问分布，**概率**（行和 ≈ 1；最后一项是 pass）
    soft_mask     : (B,)  —— 1 = 该行走软 CE，0 = 该行**不参与**软项
    soft_weight   : float —— 软项的全局缩放（`--soft-weight` 的函数侧形参；默认 1.0）

    **掩码是逐行二选一，不是混合。** `mask=0` 的行贡献**恰好 0**，**不退化
    成 one-hot CE**、也不与软项按比例插值。理由（spec §5.7）：同一个 head 同时
    收到「搜索分布」与「人类 one-hot」会去折中而不是学搜索。段 2/3 用独立采样器
    **只取软行**（`--soft-only-sampling`）来喂蒸馏，不靠混合权重。

    **分母恒为 B（不是 mask 的和）。** 这是上面那条「二选一」的直接推论：
    掩掉的行既不进分子也不进分母 ⇒ 软项的量级随「本批软行占比」线性变化。
    段 2 的采样器把占比拉到 1，所以正常训练里 B == Σmask；要在这里改成
    `sum / mask.sum()`（占比无关的平均）会让软项的量级与 1.0 权重的含义
    随 batch 组成漂移。`soft_weight` 就是给这个全局量级用的旋钮。

    精度：**升到 float32，但绝不上 float64**。autocast 下 logits 可能是
    fp16/bf16，而软 target 在低精度下会被舍入掉可观的相对误差（bf16 只有 8 位
    尾数，0.001 量级的概率直接被抹平），所以要 `.to(torch.float32)`。
    但 旧多卡环境 **没有 fp64 硬件**，任何设备侧 fp64 都走「cast 成 fp32」的兜底，
    而实测那条兜底路径会挂 专用内核（`EXCEPTION TASK: task type=aicpu kernel`），
    且故障是**异步**的——报错栈会指向后面第一次同步的无关算子，极难定位。
    由 `tests/test_no_aicpu_ops_in_startup_check.py::test_no_fp64_anywhere_on_the_device_path`
    钉死。返回标量张量（fp32）。
    """
    if soft is None or soft_mask is None:
        raise ValueError(
            "compute_policy_loss(kind='soft_ce') 需要 soft (B,A) 与 soft_mask (B,)；"
            " 现状二者都是 None —— 软标签的接线在 A3/A4（预取器透传 dict + CLI）")
    if tuple(soft.shape) != tuple(policy_logits.shape):
        raise ValueError(f"soft {tuple(soft.shape)} 与 logits "
                        f"{tuple(policy_logits.shape)} 形状不符")
    # 一律 fp32：见 docstring 的 旧多卡环境/fp64 段。`soft` 是 fp16（.npz 里就是
    # float16），直接乘会把 0.001 量级的概率舍掉，必须先升到 fp32。
    calc_dtype = torch.float32
    logp = F.log_softmax(policy_logits.to(calc_dtype), dim=-1)
    tgt = soft.reshape(logp.shape).to(calc_dtype)
    per_row = -(tgt * logp).sum(dim=-1)                     # (B,) 每行的软 CE
    mask = soft_mask.reshape(-1).to(calc_dtype)             # (B,) 逐行二选一
    if mask.numel() != per_row.numel():
        raise ValueError(f"soft_mask 长度 {mask.numel()} 与 batch {per_row.numel()} 不符")
    return float(soft_weight) * (per_row * mask).mean()


def compute_policy_loss(policy_logits, move_t, kind,
                        label_smoothing=0.1, huber_beta=0.5,
                        soft=None, soft_mask=None, soft_weight=1.0):
    """按 `--policy-loss` 分派 policy 损失；返回标量张量。

    **默认 kind='ce'**（P4.5b 用户裁决）。`huber` 分支保留，仅供复现 D4/P4.5
    的实验；它不是默认的理由写在 docstring 末尾的「P4.5b」一节（梯度尺度对
    动作空间 A 的结构性依赖），别只看 `default=` 那一行就改回去。

    三种 kind 的语义（软标签接入后新增第 3 条，前两条**逐位不变**）
    --------------------------------------------------------------
    kind='ce'   —— **原口径原样保留**：`F.cross_entropy(logits, move_t,
        label_smoothing=...)`，与 D4 之前的调用逐字相同（默认 label_smoothing
        0.1 也照旧生效），数值行为不得改变。目标 = **人类/自对弈的 one-hot**。
    kind='huber' —— 对 policy **目标**（label-smoothed one-hot 分布）做 Huber：
        · 目标 y = (1−eps)·onehot(move_t) + eps/A（eps = label_smoothing，
        A = 动作数）—— 与 `F.cross_entropy(label_smoothing=eps)` 用的是
        **同一构造**，故 `--label-smoothing` 对两条路径语义一致；
        · 预测侧 = `softmax(policy_logits)`（模型输出是 logits，目标是概率
        分布，必须先归一化到同一值域 [0,1] 才能逐元素回归）；
        · 归约 = **类内 sum over A，再对 batch 取 mean**（P4.5-fix 修，见下）。
    kind='soft_ce' —— **软标签**交叉熵（蒸馏，spec §5.7）。目标不是 one-hot 而是
        KataGo 的**访问分布** `soft` (B,A)（361 落点 + pass），由
        `SupervisedDataset(..., soft_idx=, soft_policy=)` 挂载、经
        `labels=True` 返回的 `labels_dict['soft']` 送来，且**已随 `states` 同步
        重排过对称变换**（`permute_soft`）。掩码 `soft_mask` **逐行二选一**：
        1 = 该行只算软 CE，0 = 该行对软项贡献恰好 0（**不退化成 one-hot CE**、
        不做插值）—— 理由与分母约定见 `soft_cross_entropy` 的 docstring。

    `kind='soft_ce'` 走的是**新参数** `soft` / `soft_mask` / `soft_weight`；
    `huber` / `ce` 两条既有路径**完全不看这三个形参**，数值逐位不变（有测试钉着）。
    CLI 侧的 `--policy-loss` choices 仍冻结在 `['huber','ce']`
    （tests/test_huber_loss.py::test_no_new_cli_params，D1 零新增），
    `--soft-weight` 的 CLI 入口属 A4；本函数已带好同名形参（默认 1.0）。

    **P4.5-fix：归约口径是修过的实现 bug，不是设计选择。**
    D4 首版用 `huber_loss(..., reduction='mean')`，对 **B×A 个元素**求均值；
    而被它替换掉的 `F.cross_entropy` 是**逐样本**（类内 softmax 已归一）再对
    batch 求均值。两者差一个 **1/A 的稀释**（A=361 → 361×）。实测（B=8, A=361,
    beta=0.5, targets ±1, 近均匀初值，损失输入空间的梯度范数）：

        policy 梯度范数   ce 3.178e-01  →  D4 首版 huber 2.723e-06  （塌 116,707×）

    另有一个**不可修**的量级差：Huber 打在 softmax 概率上时，梯度 ∝ p(1−p)，
    近均匀初值下 ≈ 1/A，比 CE 的 O(1) 天然弱 ~A 倍。所以**修归约只是把实现
    bug 拿掉，policy/value 的相对梯度仍严重失衡（详见 report 的
    `## Fix 增补` §残余比值）**。

    两种归约都实现并实测过（B=8, A=361, beta=0.5, targets ±1）：

        口径                        policy_loss 起步   |g_policy|      value:policy(w=1)
        sum over A, mean over B  ←  0.649744          9.829e-04        359.7 : 1
        mean over A, mean over B     1.7998e-03        2.723e-06      129,854 : 1

    选 **sum over A, mean over B**：老 CE 的量级是「每样本约 1~6」
    （实测 ln(361)=5.89），`mean over A` 会把它压到 1/361，直接改变可学习率
    的含义与 `--value-loss-weight` 的语义。代价是 policy_loss 起步值比 CE 小
    ~9 倍（0.650 vs 5.889）—— 量级变了，但**不再随 A 变化**（老 CE 也是 O(ln A)）。

    与 CE 的量级差异（**已知行为变化，不是 bug**）：初始时 softmax ≈ 1/A、目标
    ≈ 1−eps，|d| ≈ 0.9 落在线性段 → 每样本 policy_loss ≈ (0.9−0.5·0.5)·(1−1/A)
    + 360·二次段 ≈ **0.650**（旧 docstring 写的 ~1.5e-3 是 `mean over A` 口径的
    值，且 (0.9−0.25)/361 精确算是 **1.80e-03** 不是 1.5e-3），而 CE 起步
    ≈ ln(A) ≈ 5.89。`loss` 与 `policy_loss` 的数值与旧 run **不可直接比**。

    **P4.5b：默认值已由 `huber` 改成 `ce`。`huber` 分支保留（可复现实验），
    但它**不是**默认 —— 下面这段是留给下一个想把它改回去的人的：**

    **定义在概率上的损失，其梯度尺度必然依赖动作空间 A。** 机制：CE 打在
    log-prob 上，`d(CE)/d(logit_j) = p_j − y_j`，每坐标**有界且与 A 无关**；
    而 Huber 打在概率上，链式法则要再乘一层 softmax 雅可比
    `d p_k / d logit_j = p_k(δ_kj − p_j)`，专家坐标上就带一个 **p ≈ 1/A**：

        Huber 的 logit 梯度 ≈ p_expert·(∂h/∂p) / B ≈ (1/A)·O(1) / B
        CE    的 logit 梯度 ≈ 1/B（每坐标 O(1)，不随 A 缩）

    ⇒ **归约只改一个 A 的幂，改不掉这个依赖**：`mean over A` 给 1/A²、
    `sum over A` 给 A。本仓支持 9/13/19 路（A = 82/170/362），实测（生产
    FCPolicyHead，B=8，one-hot，同一组特征/权重；量的是 `‖∂L/∂logits‖`，
    见 `tests/test_huber_loss.py::test_ce_policy_gradient_is_action_space_independent`）：

        A=82(9 路)      A=362(19 路)     362/82
        CE 0.3512 0.3530 **1.005** ← 与 A 无关
        Huber(p) sum over A   0.0046     0.0010        **0.223**   = 82/362 = 1/A

    0.223 与 1/A = 0.2266 差 1.6%，即 Huber 的 logit 梯度**严格按 1/A 缩放**：
    同一个学习率、同一个 batch，在 19 路上 policy 的有效步长只有 9 路的 1/4.4。
    这不是可以靠 `--value-loss-weight` 补平的常数偏差 —— 它**随盘口变**，
    9/13/19 三档要配三个不同的权重，且换 `--policy-loss` 之外的任何参数都不会
    改变它。这条同时解释了 P4.5 修不掉的残余失衡：Huber 打在概率域时
    policy 梯度天生弱 ~A 倍，共享主干因此是 value-only 的。

    附带效果（已实测，`--value-loss-weight` 仍是 1.0、无补偿旋钮）：
    `policy=ce + value=huber, w=1` 的 value:policy 梯度比 = **1.113 : 1**
    （损失输入空间；P4.5 报告记录的 ce 备选口径同一个数，两次独立实测一致），
    老的 `ce + 5·bce` 是 2.225:1 —— 同一量级，**不需要任何补偿旋钮**。
    """
    if kind == 'ce':
        return F.cross_entropy(policy_logits, move_t,
                            label_smoothing=label_smoothing)
    if kind == 'huber':
        A = policy_logits.shape[-1]
        eps = float(label_smoothing)
        with torch.no_grad():
            target = torch.full_like(policy_logits, eps / A)
            target.scatter_(1, move_t.long().view(-1, 1),
                            1.0 - eps + eps / A)
        pred = F.softmax(policy_logits, dim=-1)
        # 逐样本：类内 sum over A，再对 batch 取 mean（对齐 F.cross_entropy 的
        # 「batch 维求均值」口径；reduction='none' + 手工聚合是唯一能同时
        # 保住逐元素 Huber 与逐样本归约的写法）
        return huber_loss(pred, target, beta=huber_beta,
                        reduction='none').sum(dim=-1).mean()
    if kind == 'soft_ce':
        return soft_cross_entropy(policy_logits, soft, soft_mask,
                                soft_weight=soft_weight)
    raise ValueError(f"--policy-loss 只接受 huber|ce|soft_ce，收到 {kind!r}")


def compute_value_loss(value_pred, value_target, kind, huber_beta=0.5):
    """按 `--value-loss` 分派 value 损失；返回标量张量。

    目标一律是 `value_t ∈ [-1,1]`（无 winrates 时是硬标签 ±1，有 winrates 时是
    连续胜率 (−1,1)），**原样回归** —— 不做概率化、不截断、不取 log。

    C8 修正：旧代码在「数据无 winrates」时走 BCE 分支 ——
    `(v+1)/2*0.8+0.1` 把目标压进 [0.1,0.9] 再喂 `binary_cross_entropy_with_logits`
    （打在本该是 Tanh/有界的 value 输出上，语义错：BCEWithLogits 期望未压缩的
    logit，目标却被手工搬进概率空间；两条数据路径还把同一个 value 头训到两种
    值域 ±2.197 vs ±1）。该分支已整段删除，本函数不再看 `dataset.winrates` ——
    winrates 路径若要旧行为，传 `--value-loss mse`（与旧 MSE 调用逐位相同）。

    kind='mse'  —— **原口径原样保留**：`F.mse_loss(pred.squeeze(),
        target.squeeze())`，与 D4 之前 winrates 分支的调用逐字相同。
    kind='huber' —— 同样 squeeze 后对 (value_pred − value_t) 做
        `huber_loss(beta=huber_beta)`，**mean over B**（N=B，不是 B×A：
        value 每个样本只有一个标量，类内没有可聚合的维度）。

    squeeze 沿用旧实现（(B,1)→(B,)），保证 mse 分支连形状都与改前一致。
    value 侧**不涉及** P4.5-fix 修的那个 1/A 稀释：被替换的 `F.mse_loss` 本身
    就是 mean over B，与这里的归约逐位同口径。

    **为什么 value 侧留 Huber、而 policy 侧在 P4.5b 改回 CE**（简报 §2 的裁决）：
    value 头是**单标量输出、没有 softmax**，链式法则里就不存在 policy 侧那个
    「再乘一层 `∂p_k/∂logit_j = p_k(δ_kj−p_j)`」的雅可比因子，也就没有
    「梯度尺度随动作空间 A 变」这个结构性缺陷（它的 A 相关差异只来自
    `FCValueHead` 的 GAP 在 H·W 个位置上求平均，属结构性、与损失无关）。
    policy 侧不同：CE 定义在 log-prob 上、`d(CE)/d(logit)=p−y` 每坐标有界且与 A
    无关，Huber 定义在概率上则严格按 1/A 缩放（实测 A=82 vs 362 为 0.223；
    见 compute_policy_loss docstring 末节与
    tests/test_huber_loss.py::test_ce_policy_gradient_is_action_space_independent）。
    留 Huber 的另一半理由是数值行为：|d| ≥ beta 段梯度有界（∝ sign(d)），
    SFT 早期 value 误差很大（目标 ±1、初值 ~0 → |d| ~ 1）时 log 不会被一次
    离群样本炸出天文数字（MSE 会：其梯度 ∝ 2|d|）。
    """
    pred = value_pred.squeeze()
    target = value_target.squeeze()
    if kind == 'mse':
        return F.mse_loss(pred, target)
    if kind == 'huber':
        return huber_loss(pred, target, beta=huber_beta)
    raise ValueError(f"--value-loss 只接受 huber|mse，收到 {kind!r}")



def _early_stop_decision(early_stop_metric, current_metric, best_metric, counter, patience):
    """早停的**唯一判定点**：返回 (new_best, new_counter, stop)，纯函数无副作用。

    new_best 是布尔「本次 eval 是否刷新了最佳值」而不是最佳值本身 —— 调用方
    据此决定要不要把 `best_metric` 推到 `current_metric`，方向判断只在这里一处，
    避免「比较用一套方向、赋值用另一套」这种静默写反。

    指标方向沿用既有语义：`loss` 越小越好（调用方已把 brier/kl 解析成
    current_metric），其余（top1）越大越好。

    stop 的判据是 `counter >= patience`（**含等号**）：patience-1 次无改善不停，
    第 patience 次恰好停。
    """
    if early_stop_metric == 'loss':
        improved = current_metric < best_metric
    else:  # top1
        improved = current_metric > best_metric
    new_counter = 0 if improved else counter + 1
    return improved, new_counter, new_counter >= patience


def _sync_stop_flag(stop_flag, is_dist, early_stop_enabled):
    """把 rank0 的早停结论广播给所有 rank（**没停也必须调用**）。

    判定只在 rank0 做（计数器语义不变），但「停不停」必须所有 rank 一致：否则
    rank0 跳出训练进入收尾的 eval、其余 rank 还在 step 循环里做反向，collectives
    次序错配 → 挂死或 DDP 崩。

    这里刻意**不**加 `if stop_flag.item()` 之类的短路：eval 块在所有 rank 上
    同点到达（`--eval-every` / `--early-stop` 是逐 rank 相同的 CLI 参数，
    `len(eval_idx) > 0` 也是全 rank 一致的本地条件），所以无条件 broadcast
    才能保证「每个 rank 参加同样次数、同样次序的 collective」。一旦短路，
    没停的 rank 直接跳过 broadcast、停了的 rank 阻塞在上面 —— 只是把
    「epoch/step 次序错配」换成「broadcast 挂死」。

    非 DDP 或未启用早停时不做任何事：单卡走 collective 纯属浪费，未启用早停时
    stop_flag 恒为 0，广播一个常量没有意义。
    """
    if not is_dist or not early_stop_enabled:
        return
    dist.broadcast(stop_flag, src=0)


#: `resolve_grad_checkpoint` 的第二个返回值：需要打日志解释「为什么 GC 被关掉
#: /为什么它被保留」。`None` = 保持配置原值、无事发生。
GC_REASON_COMPILE_EXCLUSIVE = 'compile-exclusive'
GC_REASON_GC_WITH_COMPILE = 'gc-with-compile'


def resolve_grad_checkpoint(grad_checkpoint, *, compile_on, gc_with_compile):
    """算本次运行的梯度检查点开关。返回 ``(gc, reason)``。

    唯一还在的图编译形态是 ``compile_on``（整模型 ``torch.compile(model)``），
    而它与梯度检查点**兼容**：原模型被包在 OptimizedModule **内层**，
    `backbone.assert_grad_checkpoint_compile_compatible` 扫子树看不到
    ``_orig_mod``；检查点走 ``use_reentrant=False``（``backbone._checkpointed``
    已经是），正是与 ``torch.compile`` 兼容的那一种。⇒ 两者**可以同时开**。

    默认仍然关掉（保持历史行为），``--gc-with-compile 1`` 才保留 —— 因为关掉 GC
    意味着激活全驻留、batch 被迫调小，而 batch 缩小吃掉的吞吐通常远大于融合
    省下的（40GB 卡上这条尤其硬：GC 关掉后 batch 直接对不上算力）。

    抽成函数而不是内联在 ``main()`` 里：这条策略有两个输入组合、且两个 reason
    各对应一条不同的 warning/info，内联的话只能靠源码字面断言来测。
    """
    if compile_on and not gc_with_compile:
        return 0, GC_REASON_COMPILE_EXCLUSIVE
    if compile_on and gc_with_compile:
        return int(grad_checkpoint), GC_REASON_GC_WITH_COMPILE
    return int(grad_checkpoint), None


def _build_param_groups(model, args) -> list[dict]:
    """构造 AdamW 的四组参数：{非 value, value} × {decay, no_decay}。

    value head 独立 LR（参数量小，需要更高学习率补偿梯度不足）；
    排除 bias / BatchNorm / LayerNorm 参数的 weight decay（标准做法）。

    no_decay 判据是 **param.ndim == 1**，且集合里收的是 **param 对象**而不是名字。
    这点是必须的：`model.named_parameters()` 产出全名（'value.fc.bias'），
    而 `model.value.named_parameters()` 产出相对名（'fc.bias'）——两侧命名空间
    不同，按名字比对永远不成立。本逻辑此前内联在 main() 里，正是栽在这里：
    value 的 no_decay 组恒空、value decay 组吃掉 value 头全部参数（含 BN/LN 权重
    与 bias），与 SFT 常规做法相反。`nn.Parameter` 按身份可哈希（`Tensor.__hash__`
    是 id 语义），直接用对象做集合元素即可，不必绕 `id()`。

    value 归属按**前缀** `'value.'` 判定而非子串 `'value' in n`：子串判定会把将来
    主干里新增的 `backbone.value_proj` 之类（非 value 头、名字里带 value）误归到 value 组。

    调用前提：`model` 必须是**裸模块**。DDP 包裹后 `named_parameters()` 的名字会带
    `module.` 前缀，`startswith('value.')` 就不再成立。main() 里本函数在
    DDP 包裹之前调用（分组只依赖模块自身结构，与 is_dist 无关），
    改调用顺序时留意这一条。

    DDP **不扁平化参数**，所以本函数抓到的分组对象与 optimizer 绑定的始终是同一批
    `nn.Parameter`：包裹层只在顶层加一层 wrapper，内部 `model.value` / 每个
    `p.ndim` 都保持原样。`p.ndim == 1` 的 no_decay 判据与 `'value.'` 的前缀判据
    因此在包裹前后等价 —— 这正是「分组必须在包裹前建、但语义不会被包裹改坏」的
    依据（历史：上一代分片式包裹层会把参数换成扁平的 `FlatParameter`，那才是会让
    这两条判据双双失效的东西）。

    ---- V7（`NbtTfNet`）：value 头叫 `value_head`，不是 `value` ----------------
    V7 的子模块命名与 `AlphaGoNet` 不同（`value_head` / `scorebelief_head` /
    `policy_head`），而 `_build_param_groups` 的两条判据都写死在 `'value.'` 前缀
    与 `model.value` 上。`getattr(model, 'value', None)` 在 V7 上是 `None` ⇒ 直接
    找 `value_head`。 `scorebelief_head` **不**放进 value 组：它的行权重（`w_score`）
    在段 1 是 0、段 2/3 才打开，而它的输入是 `value_pooled`（已经过池化），
    与 19×19 平面小头（ownership / scoring / futurepos / seki）不同层 —— 混在一
    组里会让「value 头 LR 倍数」的含义随段位漂移。它留在 `other_*` 组里。
    """
    value_mod = getattr(model, 'value', None)
    if value_mod is None:
        value_mod = getattr(model, 'value_head', None)
    if value_mod is None:
        raise ValueError(
            f'{type(model).__name__} 上既没有 .value 也没有 .value_head：'
            f'无法划分 value 头的参数组（旧结构是 AlphaGoNet.value，V7 是 '
            f'NbtTfNet.value_head）')
    no_decay_params = {p for p in model.parameters() if p.ndim == 1}
    value_params = {p for p in value_mod.parameters()}
    value_decay = [p for p in value_mod.parameters() if p not in no_decay_params]
    value_no_decay = [p for p in value_mod.parameters() if p in no_decay_params]
    # `other_*` 的排除条件是「前缀不是 'value.'」**且**「对象不属于 value 头」——
    # 两个条件缺一不可：前缀条件对 AlphaGoNet 是充分的，但对 V7 无效
    # （V7 的头叫 `value_head.`，不匹配 `'value.'`），少了对象条件就会让
    # `value_head.*` 同时落进 value 组与 other 组 —— 也就是**同一个参数出现两次**
    # （AdamW 会照单全收，梯度被更新两遍）。对象条件对 AlphaGoNet 是**空操作**
    # （没有跨模块共享的参数），所以默认路径逐位不变。
    other_decay = [p for n, p in model.named_parameters()
                if not n.startswith('value.') and p not in value_params
                and p not in no_decay_params]
    other_no_decay = [p for n, p in model.named_parameters()
                    if not n.startswith('value.') and p not in value_params
                    and p in no_decay_params]
    return [
        {'params': other_decay, 'lr': args.lr,
        'weight_decay': args.weight_decay},
        {'params': other_no_decay, 'lr': args.lr, 'weight_decay': 0.0},
        {'params': value_decay, 'lr': args.lr * args.value_lr_mult,
        'weight_decay': args.weight_decay},
        {'params': value_no_decay, 'lr': args.lr * args.value_lr_mult, 'weight_decay': 0.0},
    ]


def compose_losses(policy_loss, value_loss, value_loss_weight, l2_report):
    """总损失的两个口径，返回 `(opt_loss, log_loss)` —— **它们故意不相等**。

    · `opt_loss = policy_loss + w·value_loss`：**被 backward** 的量。它**不含**
    `c‖θ‖²`，因为正则走 AdamW 的**解耦** weight decay（`θ ← θ − lr·wd·θ`，在
    参数更新里、按构造不进梯度）。
    · `log_loss = opt_loss + l2_report`：写进日志 `loss` 键的量。多了
    `c‖θ‖²`，是为了让用户裁决的恒等式 `L = L_policy + L_value + c‖θ‖²` 在日志
    上字面成立（`l2_report` 由 `compute_l2_report` 给，纯 python float ⇒
    **不进计算图**，所以即便对 log_loss 反向，L2 项也贡献不出任何梯度）。

    为什么写成函数而不是在 main() 里手写两行：这两个量必须能被测试**按行为**
    区分开（`tests/test_huber_loss.py::test_log_loss_identity` 断言
    「log_loss.backward() 与 opt_loss.backward() 给出逐位相同的参数梯度」，
    即 L2 项确实是纯报告）。只写在 main() 里就只能做源码子串检查 —— 而 P4.5
    的教训正是「文案对了行为没对」。

    `value_loss_weight` 出现在**两项**里：报告口径必须与优化口径同权重，否则
    `loss` 与被优化的目标在 `--value-loss-weight != 1` 时会差两项而不是一项。
    默认 w=1.0 时它就是用户裁决里的 `L_policy + L_value`。
    """
    opt_loss = policy_loss + value_loss_weight * value_loss
    log_loss = opt_loss + l2_report
    return opt_loss, log_loss


def compute_l2_report(param_groups):
    """**只用于报告**的 c·Σ‖p‖²：恒等式 L = L_policy + L_value + c‖θ‖² 的第三项。

    P4.5b 用户裁决（2026-09-27）：总损失按 `L = L_policy + L_value + c‖θ‖²`
    报告，c 通常取 1e-4。**c 不是新参数** —— 它就是 `--weight-decay`（默认
    1e-4，D1：不新增任何 CLI 参数），本函数**从优化器的 param_groups 里读回
    真实的 `weight_decay`**，所以 `--weight-decay 0` ⇒ 本项恒 0，改成 2e-4
    ⇒ 恰好翻倍。真正作用在参数上的仍是 AdamW 的**解耦** weight decay
    （`θ ← θ − lr·wd·θ`，在参数更新里、不进梯度）；本函数既不参与 backward
    也不改优化器。

    **只对 `weight_decay != 0` 的组求和**（`_build_param_groups` 的两组
    decay + 两组 no_decay）。no_decay 组（`param.ndim == 1`，即 norm 权重与
    bias）在优化器里 `weight_decay=0.0`，**没有**被正则；若图省事对
    `model.parameters()` 全量求和，日志就会报告一个仓库根本没有施加的正则
    强度 —— 那比不报更糟（读者会以为 norm 权重也被收缩了）。故判据直接取
    自 param_groups 本身，而不是重新实现一遍 ndim 判据：将来分组逻辑变了，
    本函数自动跟着变，不会漂。

    为什么这样切分（§3.1，最容易做错的一步）：训练循环里**两个变量**——
    `opt_loss = policy + w·value`（被 backward，**不含**本项）与
    `log_loss = policy + w·value + l2_report`（只写日志）。若把本项折进
    backward，等于把**解耦**衰减变成**耦合** L2 塞进梯度：AdamW 自己的
    weight decay 会与它叠加，训练行为立刻变化且没有任何报错。

    DDP：`train_sft.py` 的日志标量（含 `loss`/`policy_loss`/`value_loss`）
    **一律不做 all_reduce**（`_read_log_scalars` 直接 `.item()`，stdout 由
    rank0 的 logger 过滤、swanlab 只在 is_main 建），本项跟随同一口径：本地
    计算、本地 `.item()`、不加 collective。DDP 构造时把 rank0 的参数
    broadcast 给所有 rank、每步再 all_reduce 梯度（`find_unused_parameters`
    =False、无 `no_sync`），故各 rank 的 θ 恒一致，本项在各 rank 上是同一个数
    —— 同一个日志键不会在不同 rank 上打架，也就没有引入新的同步点/挂死风险。
    例外：`GradScaler.step` 只在**本地**查 `found_inf`、不做 all_reduce，
    所以一次 inf/nan 触发的 step 跳过是**单 rank** 的 —— 那一步各 rank 的 θ
    会分叉，本项在那一行就会跨 rank 不一致。稳态（无跳过）下不影响。

    代价（ 分清「参数字节」与「实际流量」，两者差 ~3×）：`p.detach().pow(2).sum()`
    是「逐元素 kernel + 归约 kernel」两段，流量是 **读 θ + 写 θ² + 读 θ²**
    ≈ 3 × 参数字节。v18 参考配置 decay 参数 12,838,112 × 4 B = 51.35 MB
    ⇒ **~154 MB/step 的实际流量**，不是 51 MB（后者只是参数字节计数）。
    `p.detach()` 保证不建图、不占住反向图。设备同步是**每次调用 1 次**
    `float()`（四组里两个 decay 组，只在最后读回一次），但本函数在训练循环里
    **每个 micro-batch 都无条件跑**（不在 `if _do_stdout or _do_swanlab:` 里），
    比日志打点那 3 次同步频繁得多 —— 见 report `## Fix3 增补` §3 的 加速器 说明。
    """
# 累加全部留在**设备上**，整个调用只做一次 `float()`（= 一次 host 同步）。
    # 逐组 `float()` 是 2 次同步，而本函数在训练循环里**每个 micro-batch 都跑**
    #   （不在 `if _do_stdout or _do_swanlab:` 里），比日志打点的 3 次同步频繁
    #   ~`_accum_steps` × `--log-every` 倍。
    # **不要在设备侧做 fp64**（`sq.double()`）：旧多卡环境 没有 fp64 硬件，实测
    #   （2026-10-01 云端）会报 `Device do not support double dtype now` 并挂掉
    #   专用内核 kernel。本 docstring 早先论证过的「`wd * float(sq)` 与
    #   `sq.double() * wd` 逐位相同」正好给了替代：**乘加搬到主机侧**（Python
    #   float 就是 fp64），设备侧只留 fp32 的 `pow(2).sum()` 归约。
    #   精度不降反升（少一次设备侧舍入），`test_l2_report_scales_with_weight_decay`
    #   的 `rel=1e-9` 与 `test_log_loss_identity` 的恒等式容差都不受影响。
    # 单次读回不引入任何 θ 错位：所有 `pow(2).sum()` 的读都发生在这一行之前，
    #   仍是**同一个 θ**。
    #   `tests/test_huber_loss.py::test_l2_report_uses_decay_group_only` 用
    #   TorchDispatchMode 数 `aten::item`，把「恰好一次」钉住。
    _wds, _sqs = [], []
    for group in param_groups:
        wd = float(group.get('weight_decay') or 0.0)
        if wd == 0.0:
            continue          # no_decay 组：优化器没在衰减它，报告里也不该有它
        sq = None
        for p in group['params']:
            s = p.detach().pow(2).sum()
            sq = s if sq is None else sq + s
        if sq is not None:
            _wds.append(wd)
            _sqs.append(sq)

    if not _sqs:
        return 0.0

    # **一次**读回，但加法留到主机侧的 fp64：
    #   `torch.stack(_sqs).tolist()` 是**一发 D2H**（2 个标量），之后所有乘加都在
    #   Python float（fp64）里做。
    #   · 为什么不能直接在设备上加：`_sqs[0] + _sqs[1]` 是 fp32 加法，而组平方和
    #     量级 ~600 ⇒ fp32 的 ulp ≈ 6.1e-5，一次加法就引入 ~3e-5 误差，直接吃掉
    #     `test_l2_report_uses_decay_group_only` 的 1.2e-6 容差（实测 got=1200.0335
    #     vs want=1200.0335471477）。这等于把刚搬到主机侧的算术又拽回设备上。
    #   · 为什么不能逐组 `float()`：那是**两次** D2H，而本函数每个 micro-batch 都跑。
    #   `tests/test_huber_loss.py::test_l2_report_uses_decay_group_only` 用
    # TorchDispatchMode 数 `aten::item`； 那个探针**看不见** `.tolist()`（它只
    #   匹配 `item`/`_local_scalar_dense`，而 stacked 读回落在这两者之外），所以那条
    #   测试现在断言的是「≤ 1 次代理可见同步」，真实的「恰好一次 D2H」由
    #   `tests/test_no_aicpu_ops_in_startup_check.py` 用结构检查钉住。
    _vals = torch.stack(_sqs).tolist()
    _total = 0.0
    for _wd, _sq in zip(_wds, _vals):
        _total += float(_sq) * _wd
    return _total


# ---- P4.6：fused AdamW 的设备策略与回退（全模块唯一构造入口）------------------
# D1：不加 `--fused` 旗标 —— 选择由设备驱动的代码级默认决定，不读环境变量、不接 CLI。
_FUSED_OK_BACKENDS = frozenset({'cuda'})


def build_adamw(param_groups, device, logger=None):
    """按设备构造 AdamW：支持 fused 就走 fused kernel，不支持就回退标准实现。

    返回 `(optimizer, mode)`，`mode ∈ {'fused', 'standard'}`（main 拿去打日志，
    `tests/test_fused_optimizer.py` 拿去钉分支选择与回退契约）。

    **「fused」是什么**：`torch.optim.AdamW(param_groups, fused=True)` —— torch 的
    融合路径（在融合 kernel/多张量一趟里做完 exp_avg、exp_avg_sq 更新与参数写回，
    省逐参数 kernel 启动开销）。它**只换 kernel、不换语义**：四组
    `{非 value, value} × {decay, no_decay}`、解耦 weight decay（L2 不进 loss ——
    P4.5b 的 opt_loss/log_loss 分离与 `compute_l2_report` 的报告口径）在两种模式
    下逐字相同。**不是** 第三方融合优化器（MindSpeed 等 NPU 专属方案）：那是 D6 / P4.11 的事，本函数不碰。

    **设备策略**（A1 记录：CUDA A100 支持；本地开发是 CPU）：

    * `cuda` → `try: AdamW(param_groups, fused=True)`；构造抛 `TypeError`（老
    torch 无此 kwarg）或 `RuntimeError`（无 fused kernel）→ **回退标准构造，
    异常不冒泡**；
    * `cpu` / 其它 → **直接标准构造**（不尝试、不报错）。

    **回退契约（bit-for-bit）**：标准分支就是 `torch.optim.AdamW(param_groups)`
    —— 与 P4.6 之前 main() 非 CUDA 分支**逐字相同、零 kwargs**。同种子、同初始
    权重、同梯度序列下，回退后的参数更新与旧代码 `torch.equal`（零容差，不是
    「近似相等」）。CPU 上由
    `test_fused_optimizer.py::test_cpu_fallback_is_bit_for_bit_head_behaviour`
    钉死；fused 激活时与 standard 是**容差内一致、非逐位**（不同 kernel 的浮点
    结合顺序；实测 torch 2.12.0+cpu 10 步 max|Δ|=2.38e-7，容差 1e-6）——
    见该文件模块 docstring 与 task-p4-6-optimizer-report.md。
    """
    backend = str(device).split(':')[0]
    if backend in _FUSED_OK_BACKENDS:
        try:
            optimizer = torch.optim.AdamW(param_groups, fused=True)
            if logger is not None:
                logger.info("[train] 已启用 fused AdamW (backend=%s)", backend)
            return optimizer, 'fused'
        except (TypeError, RuntimeError, ImportError) as exc:
            if logger is not None:
                logger.warning(
                    "[train] fused AdamW 在 %s 上不可用（%s: %s），回退标准实现",
                    backend, type(exc).__name__, exc)
    # 标准实现兜底。`foreach` 多张量路径**始终**开启：语义与逐参数循环**完全相同**
    # （只是把数百次 kernel 启动合并成批量调用），且不改变浮点结合顺序。
    # 它与 `fused` 不同 —— fused 是融合 kernel、与 standard 只保证容差内一致；
    # foreach 是同一批 kernel 的批量调用、**逐位**等价。
    _std_kwargs = {'foreach': True} if _HAS_FOREACH else {}
    optimizer = torch.optim.AdamW(param_groups, **_std_kwargs)
    if logger is not None:
        if _std_kwargs:
            logger.info("[train] AdamW 标准实现（backend=%s，fused 未启用；"
                        "foreach 多张量路径已启用）", backend)
        else:
            logger.info("[train] AdamW 标准实现（backend=%s，fused 未启用）", backend)
    return optimizer, 'standard'


def _param_group_sizes(source, key=None):
    """数出各参数组的参数个数；结构不可读时返回 None。

    只用来拼告警文案，所以**绝不能自己抛**：它跑在 except 块里，抛一次就把
    「真正该看的那个异常」盖掉了。`key` 给了就从映射里取，否则把 source 当序列用。
    """
    try:
        groups = source[key] if key is not None else source
        return [len(g['params']) for g in groups]
    except (TypeError, AttributeError, KeyError, IndexError):
        return None


def _load_optimizer_state(optimizer, state, logger) -> bool:
    """把 checkpoint 里的优化器状态灌进 optimizer；分组布局不兼容时告警并跳过。

    返回 True = 已加载；False = 已跳过（Adam 动量从零开始）。

    **为什么必须有这一层**：optimizer 的 state_dict 把**每组的参数个数**写进了
    `param_groups`，而 `torch.optim.Optimizer.load_state_dict` 逐组比对大小，
    不一致就直接
    `ValueError: loaded state dict contains a parameter group that doesn't match
    the size of optimizer's group`。C12 把 value 头的一维参数搬进 no-decay 组之后，
    四组大小随之改变（`[7, 13, 11, 0]` → `[7, 13, 4, 7]`）→ **本提交之前存下的
    任何 checkpoint 都再也续不上**，resume 在这里硬崩。分组布局变了以后，旧的动量
    本来就是按「旧分组的参数下标」记的，映射到新分组上没有意义，跳过重训比崩掉正确。

    **刻意只放行这一类失败**：判据是「异常类型 ∈ {ValueError, RuntimeError} 且信息里
    点名 parameter group」（torch 对组布局不兼容只有两条消息，两条都含这个词）。
    其余失败照旧抛出 —— resume 静默降级比崩掉难查得多：训完一整轮才发现动量没恢复。

    torch 的组大小检查发生在 `deepcopy` 之后、`__setstate__` 之前，所以命中时
    optimizer 状态**没有被半途写入**，跳过是干净的。
    """
    try:
        optimizer.load_state_dict(state)
    except (ValueError, RuntimeError) as exc:
        if 'parameter group' not in str(exc):
            raise
        logger.warning(
            "[resume] 【注意】优化器状态与当前参数分组不兼容，已跳过加载。"
            "checkpoint 里的组大小 %s ≠ 当前 %s（torch: %s）。分组方案变了之后，旧的 "
            "Adam 动量是按旧分组的参数下标记的，映射到新分组上已无意义，"
            "因此优化器状态已重置，Adam 动量从零开始。模型权重 / EMA shadow / scaler / "
            "step / epoch / rng 仍照常恢复，训练可以继续，但开头若干步的等效学习率会有"
            "一次跳变。要保留动量，请用当前代码写出的 checkpoint 续训。",
            _param_group_sizes(state, 'param_groups'),
            _param_group_sizes(optimizer.param_groups), exc)
        return False
    return True


def _load_model_state(model, ckpt, logger) -> None:
    """把 state_dict 灌进 model，自动对齐 torch.compile 引入的 '_orig_mod.' 段。

    这段可能出现在路径的**任意层级**（整模型 compile 插在开头；历史上还有过
    逐子模块 compile 插在中段的那种形态，后端已移除）。因此这里不按位置处理，
    而是**双向规范化**：先把目标模型与 checkpoint 的键都归一到「未编译布局」，
    再按目标模型当前的真实键写回去。

    比旧实现多修掉一个 bug：旧代码在 `hasattr(model, '_orig_mod')` 时把前缀
    **补回** ckpt，却把权重灌进已经剥掉前缀的 `model._orig_mod`（其
    `state_dict()` 键本就无前缀）——`--compile 1` + `--resume` 必然
    RuntimeError（missing/unexpected key）。现在两侧同归一，这个矛盾不存在。
    """
    target = getattr(model, '_orig_mod', model)  # 整模型 compile：剥到真实模块
    # 未编译布局的键 → 目标模型当前真实键。setdefault 保留首个（真重复才会撞上）。
    canon = {}
    for k in target.state_dict().keys():
        canon.setdefault(k.replace('_orig_mod.', ''), k)
    aligned = {}
    moved = 0
    for k, v in ckpt.items():
        c = k.replace('_orig_mod.', '')
        real = canon.get(c, c)
        moved += real != k
        aligned[real] = v
    # 「对齐有没有发生 / 发生了几处」是可观测的：resume 静默换掉权重比崩掉难查，
    # 键名对不上时也正是从这条日志里看出来的。
    if moved:
        logger.info("[state_dict] %d/%d 个权重键按 compile 包装（_orig_mod.）重定位",
                    moved, len(ckpt))
    target.load_state_dict(aligned)


def _prof_parse_trace(trace, *, row_limit=18):
    """chrome trace JSON →（top-k 内核表, cat 总耗时摘要）。

    **纯函数**：只吃已经 ``json.load`` 出来的对象，不碰 profiler 实例 ⇒ 不装
    NPU 版 torch 也能测。这是 `key_averages()` 那条路走不通时的唯一出口。

    为什么必须有它：``NPU 版 torch 2.1.0.post10`` 的 ``NPU 版 torch.profiler.profile``
    是**独立类**，实例方法只有 ``add_metadata / add_metadata_json /
    export_chrome_trace / export_memory_timeline / export_stacks / start /
    step / stop`` 共 8 个，**没有** ``key_averages / events / profiler_result``
    （真机实测）。于是打表那段在真机上必
    AttributeError，被外层 ``except`` 吞成一行 warning ⇒「窗口跑完了、表没有」。

    聚合口径：按 ``(cat, name)`` 求和时长。只收 ``dur > 0`` 的事件 —— trace 里
    还有大量瞬时（``ph=i``）、计数器、flow、metadata 事件，它们没有时长，
    混进来会把「按耗时排序」变成「按事件条数排序」。
    """
    events = trace.get('traceEvents', trace) if isinstance(trace, dict) else trace
    agg = {}   # (cat, name) -> [total_us, count]
    cats = {}  # cat       -> total_us
    for ev in events:
        if not isinstance(ev, dict):
            continue
        dur = ev.get('dur') or 0
        if dur <= 0:
            continue
        cat = str(ev.get('cat') or '?')
        name = str(ev.get('name') or '?')
        row = agg.setdefault((cat, name), [0.0, 0])
        row[0] += float(dur)
        row[1] += 1
        cats[cat] = cats.get(cat, 0.0) + float(dur)
    if not agg:
        return ('(trace 里没有带时长的事件 —— 采到 0 条，窗口可能是 0 步)',
                '(空)')
    lines = ['%7s %12s  cat / name' % ('cnt', 'total_us')]
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:row_limit]
    for (cat, name), (tot, cnt) in top:
        lines.append('%7d %12d  %s / %s'
                     % (cnt, int(round(tot)), cat, name))
    cat_line = ', '.join('%s=%.2fs' % (c, t / 1e6)
                         for c, t in sorted(cats.items(), key=lambda kv: -kv[1])[:14])
    return ('\n'.join(lines), cat_line)


def _prof_trace_table(prof, *, row_limit=18):
    """``key_averages()`` 缺失时的退路：导出 chrome trace → 解析 → 表 + cat 摘要。

    ``export_chrome_trace(path)`` 是 NPU 版 torch **确实提供**的 8 个方法之一，所以
    这条路在真机上是走得通的。trace 落到临时文件、读完即删（一次窗口几百 MB，
    留在 /tmp 里没人收）。
    """
    import json
    import tempfile
    fd, path = tempfile.mkstemp(prefix='goai_prof_', suffix='.json')
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path, 'r', encoding='utf-8') as fh:
            trace = json.load(fh)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return _prof_parse_trace(trace, row_limit=row_limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help="单个 .npz 训练集，或包含多个 .tgz/.tar.gz/.npz 的目录（自动合并所有分片）")
    ap.add_argument('--max-games-per-tgz', type=int, default=0,
                    help="目录模式下每个 tgz 最多解析的棋局数（0=全部），用于子采样控制内存")
    ap.add_argument('--device', default='auto')
    ap.add_argument('--use-amp', type=int, default=0, choices=[0, 1],
                    help='启用混合精度训练 (0=关闭, 1=开启)')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--board-size', type=int, default=19)
    ap.add_argument('--save-every', type=int, default=2000)
    ap.add_argument('--out', default='models/sft.pt')
    ap.add_argument('--ver', default='v17',
                    help='模型版本号 (用于 swanlab name 和 --out 默认值，如 v17, c2net-v1)')
    # 注意力相关
    ap.add_argument('--backbone-channels', type=int, default=128,
                    help='主干卷积通道数。容量主开关，实测（19路, mix/window'
                        ' 注意力, 4 层注意力头）：128×12≈4.06M，192×17≈12.41M，'
                        '224×12≈12.36M，256×12≈16.13M')
    ap.add_argument('--backbone-res-blocks', type=int, default=12,
                    help='主干残差块数量（与 --backbone-channels 共同决定容量）')
    ap.add_argument('--attention-mode', default='mix',
                    choices=['none', 'mix', 'all'],
                    help='主干注意力模式: none=纯卷积, mix=卷积+注意力混合, all=全注意力')
    ap.add_argument('--num-attention-layers', type=int, default=4,
                    help='mix 模式下注意力块数量')
    ap.add_argument('--num-heads', type=int, default=4, help='多头注意力头数')
    ap.add_argument('--attention-dropout', type=float, default=0)
    ap.add_argument('--attn-mode', default='global',
                    choices=['global', 'window', 'axial', 'sparse', 'window_global'],
                    help='注意力计算模式: global=全配对, window=块状窗口, '
                        'sparse=窗口+全局token, window_global=块状窗口+全局token(手写math)')
    ap.add_argument('--attn-window', type=int, default=7, help='window 模式窗口边长')
    ap.add_argument('--attn-query-chunk', type=int, default=64,
                    help='手写 math 注意力按 query 分块的长度（0=关闭）：softmax 沿 '
                        'key 轴 ⇒ 分块数学精确，峰值 ∝ chunk。只影响 math 路径，'
                        'SDPA/SFA 融合路径自行管理显存（默认 64，0=关闭）')
    ap.add_argument('--attn-chunk-ckpt', type=int, default=1, choices=[0, 1],
                    help='math 路径逐 chunk 梯度检查点：把注意力矩阵变成反向时一块块'
                        '重算（不依赖 block 级 checkpoint 是否生效）。'
                        '0=关闭，1=开启（默认开启）')
    ap.add_argument('--attn-online', type=int, default=0, choices=[0, 1],
                    help='手写 math 注意力改用 online-softmax（flash 风格）实现：'
                        '不物化 (Nq,Nk) 分数矩阵 ⇒ 反向不保留它（省显存），'
                        '且最大值被归一、低精度下更不易溢出（0=关闭，1=开启）。'
                        '与 materialize 路径**数值等价但非逐位相同**'
                        '（实测 fp32 max|Δ|≈7e-07），开启会让 12 通道的 '
                        'test_twelve_channel_path_bit_identical 基线失效；'
                        '真机加速比尚未实测，故默认关闭')
    ap.add_argument('--use-sdpa', type=int, default=1, choices=[0, 1],
                    help='是否走 F.scaled_dot_product_attention：'
                        '1=放开（默认，省算力、fp32 累加更稳，顺带缓解 warmup 后 NaN）；'
                        '0=强制回退手写 math（调试数值差异用）。'
                        '仍受 _sdpa 内部 batch 上限保护（B>60000 自动回退 math）。'
                        ' 取代原环境变量 GOAI_SDPA（0=math 的语义）')
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--eval-max-batches', type=int, default=50,
                    help='验证集评估最多跑多少个 batch；<=0 表示不截断（跑满全部验证集）')
    ap.add_argument('--log-every', type=int, default=50,
                    help='每隔多少 step 打印一次训练日志（loss/lr/吞吐/显存）')
    ap.add_argument('--log-file', default='training.log',
                    help='训练日志文件路径（同时输出到控制台），设为空字符串可关闭文件日志')
    # 三段式架构参数
    ap.add_argument('--res-blocks', type=int, default=0,
                    help='ResBlock 数量（浅层局部细节），0 表示使用默认 mix 模式')
    ap.add_argument('--convnext-blocks', type=int, default=0,
                    help='ConvNeXtBlock 数量（中层大感受野），0 表示使用默认 mix 模式')
    ap.add_argument('--attn-blocks', type=int, default=0,
                    help='AttentionResBlock 数量（深层全局关系），0 表示使用默认 mix 模式')
    ap.add_argument('--value-res-blocks', type=int, default=3,
                    help='Value head 残差块数量（默认 3，增加到 8-11 可达 1.5-2M 参数）')
    ap.add_argument('--value-channels', type=int, default=64,
                    help='Value head 通道数（默认 64，增加到 96-128 可达 1-2M 参数）')
    ap.add_argument('--policy-channels', type=int, default=32,
                    help='Policy head 隐藏层通道数（默认 32，增加到 128-176 可达 200-300k 参数）')
    ap.add_argument('--policy-layers', type=int, default=2,
                    help='Policy head 层数（2=原始 1x1->1x1，3=1x1->3x3->1x1）')
    ap.add_argument('--prefetch-workers', type=int, default=12,
                    help='数据预取进程数：每个 batch 切块并行造特征并与 GPU 计算重叠；'
                        '<=1 关闭预取（回退同步取样）。'
                        ' 默认 12 来自 C0 基准实测：16 核上 8 worker 的边际效率已'
                        '降到 0.701（饱和），云端 24 核 ⇒ 12 是同一饱和点上的下一步。'
                        ' 这是从 16 核**外推**到 24 核的，仓库里没有 24 核实测'
                        '支撑（spec 已标为外推）。**batch 不要为迁就特征侧而设**：'
                        '特征侧对 batch 几乎不敏感（32→311.4 / 128→309.7 / '
                        '1024→307.1 行/s，离散度 1.7%%），成本在 iterLadders 逐行 DFS '
                        '而不是 batch 固定开销 ⇒ batch 按显存选即可。'
                        ' fork 之前**必须 warm**（--v7 路径的 boards mmap）：'
                        '冷 IO 8 worker 273 vs warm 1161 行/s（差 4.3×），'
                        'gather 冷读 10.4–24.7 ms/行 vs 热读 0.005–0.008（1500×）。')
    ap.add_argument('--prefetch-depth', type=int, default=8,
                    help='预取流水深度（提前多少个 batch 造好数据，控制内存/吞吐平衡）')
    ap.add_argument('--resume', default='',
                    help='断点续训：指定已保存的 .pth 模型路径，会从该权重 + 同目录 '
                        '.train_state.pt 恢复 optimizer/scheduler/step 计数继续训练')
    ap.add_argument('--model', default='',
                    help='加载预训练权重（仅权重，optimizer/scheduler/step 从头开始）。'
                        '用于迁移学习或微调，不加载优化器状态')
    ap.add_argument('--value-loss-weight', type=float, default=1.0,
                    help='value loss 权重。默认 1.0 = 无补偿（P4.5-fix 按用户裁决'
                        '删掉了 BCE 时代为平衡 policy/value 梯度而加的 5.0 倍'
                        '补偿）。 换 --value-loss 后 policy/value 的梯度量级'
                        '关系已变，见 report `## Fix 增补`：policy 默认走 huber，'
                        '其梯度天然比 value 弱 ~A 倍，修掉归约 bug 后实测仍差 '
                        '约 250~360:1，此参数不足以单独补平。')
    # ---- D4（SFT 侧）：损失口径三参数 ------------------------------------
    # 只加这三个（结构参数不新增任何 CLI）。RL 侧的对应拆分在 P3-C/P3-D，
    # scripts/selfplay_train.py 不在本任务范围。
    ap.add_argument('--policy-loss', default='ce',
                    choices=['huber', 'ce', 'soft_ce'],
                    help='policy 损失：ce=原交叉熵口径（数值行为与 D4 之前逐位一致）'
                        '，**默认**；huber=对 label-smoothed one-hot 目标做 Huber'
                        '(smooth L1, beta=--huber-beta，归约=类内 sum over A + '
                        'batch mean)，保留仅供复现实验；soft_ce=**软标签**交叉熵'
                        '（KataGo 访问分布蒸馏，spec §5.7），掩码逐行二选一 —— '
                        'soft_mask=1 的行走软 CE、0 的行贡献恰好 0（不是插值）。'
                        ' soft_ce 只在给了 --soft-index 时才会被自动选中；'
                        '显式写 --policy-loss soft_ce 而没有 --soft-index 会在'
                        '启动时报错（否则软项恒 0，是静默半吊子）。'
                        ' 默认**不是** huber（P4.5b 用户裁决）：定义在概率上的损失，'
                        '梯度尺度必然依赖动作空间 A —— softmax 雅可比贡献 p≈1/A，'
                        'mean over A 给 1/A²、sum over A 给 A，没有任何归约能消掉它；'
                        '本仓支持 9/13/19 路（A=82/170/362），选 sum 等于让同一学习率'
                        '在 9 路与 19 路之间差 4.4 倍。CE 定义在 log-prob 上，'
                        'd(CE)/d(logit)=p−y 每坐标有界且与 A 无关（实测 A=82 vs 362 '
                        '的 policy 梯度比 1.005，Huber 同条件 0.223=1/A）。'
                        '详见 compute_policy_loss docstring')
    ap.add_argument('--value-loss', default='huber',
                    choices=['huber', 'mse'],
                    help='value 损失：huber=对 value_t∈[-1,1] 直接 Huber'
                        '(smooth L1, beta=--huber-beta)；mse=原均方误差口径'
                        '（数值行为与 D4 之前逐位一致）。默认 huber（D4/C8，'
                        'BCE 分支已删）。 value 侧留 Huber 正是为了与 policy 侧'
                        '相反：value 头是**单标量输出、无 softmax**，链式法则里'
                        '没有 softmax 雅可比那层（p≈1/A），所以 policy 侧那个'
                        '「梯度尺度随动作空间 A 变」的结构性缺陷在这里不存在；'
                        '（value 侧仍有 A 相关的差异，但只来自 GAP 在 H·W 个位置上'
                        '求平均，是结构性的、与损失选择无关。）且 |d|≥beta 段梯度'
                        '有界、SFT 早期大误差不会把 log 炸飞')
    ap.add_argument('--huber-beta', default=0.5,
                    type=_positive_beta,
                    help='Huber(smooth L1) 的 beta，同时就是拐点位置 delta：'
                        '|误差|<beta 走二次段、>=beta 走线性段（线性段的逐元素'
                        '梯度是 sign(d)，但 mean 归约会再除以元素数 N —— '
                        'N=2888 时只有 ±0.00035，不是 ±1）。'
                        '实现用 F.smooth_l1_loss(beta=)，它就是教科书 '
                        'Huber(delta=beta) 本身；torch 的 F.huber_loss(delta=) '
                        '则是它的 beta 倍缩放（仅 beta=1 时两者相等），故不用。'
                        '同时作用于 --policy-loss=huber 与 --value-loss=huber，'
                        '必须 >0（0 会静默退化成 L1，负值在训练中途才炸），'
                        '默认 0.5')
    ap.add_argument('--value-lr-mult', type=float, default=5.0,
                    help='value head 学习率倍数（相对主干 LR，补偿参数量小的梯度不足）')
    ap.add_argument('--label-smoothing', type=float, default=0.1,
                    help='policy loss label smoothing（0=不平滑，0.1=标准值）')
    # ---- 局级 sidecar（spec §5.3）：A/B 段的逐局贴目 --------------------------
    ap.add_argument('--games-npz', default=None,
                    help='局级 sidecar（scripts/build_games_sidecar.py 的产物，'
                        '约 1.2 MB）。**只对 board 级数据有意义**（C 段 stdata '
                        '分片的贴目在行里）。给了就把逐局 `g_komi` 接进 V7 的'
                        '全局特征：全局 ch5（currentSelfKomi/20）与 ch18 的'
                        '三角波都要它。不给 ⇒ 贴目按 0 处理（ch5 恒 0）。'
                        ' sidecar 按 **game_id** 索引，不是按行号；长度必须'
                        '覆盖 game_ids.max()+1，短了当场报错而不是静默取到别局的贴目。'
                        ' `g_rules` 目前**接不进特征**（官方 calculateArea 只吃'
                        '标量 rules_flags，逐局化会触发 "truth value is ambiguous"）——'
                        '给了会报告有多少局规则与默认不同，那些局的空间特征仍按'
                        '简单局算。')

    # ---- A4 · 软标签（KataGo 访问分布）四参数 --------------------------------
    # spec §3 A 组 / §5.7。这四个旗**只在给了 --soft-index 时有任何作用**；
    # 没给时下面每一条路径都逐位不变（段 1 的通路基线不受影响）。
    ap.add_argument('--soft-index', default=None,
                    help='软标签索引 npz（scripts/build_soft_index.py 的产物，'
                        '含 idx (M,) 与 policy (M,362)）。给了就把它挂到数据集'
                        '行上并把 policy 损失切到 soft_ce；**不给 = 完全走段 1 '
                        '旧路径**（labels=False、policy_loss=ce，数值逐位不变）。'
                        ' 该索引必须来自当前主 npz：build_soft_index 的缓存 key '
                        '含主 npz 指纹与散列口径版本，索引陈旧时它会告警并重建 —— '
                        '静默挂错标签的症状只是「命中率 0」，别把它当成「这批局面'
                        '真的没标签」')
    ap.add_argument('--soft-weight', type=float, default=1.0,
                    help='软项的全局缩放（soft_ce 的乘子，默认 1.0 = 不缩放）。'
                        ' soft_ce 的分母恒为 batch 大小 B 而不是 Σmask ⇒ 软项'
                        '量级随「批里软行占比」线性变化；--soft-only-sampling 让'
                        '占比≈1 时本参数才有可解释的量级，否则调它是在调软行占比')
    ap.add_argument('--soft-only-sampling', type=int, default=0, choices=[0, 1],
                    help='训练行索引空间收窄到「有软标签的行」（soft_row >= 0）'
                        '(0=关闭=全量混训，1=开启)。段 2 用它：**只训软行**是'
                        'spec §1.2 的决定 —— 同一个 head 同时收到搜索分布与人类 '
                        'one-hot 会去折中而不是学搜索；--soft-weight 0 关不掉'
                        '这个问题（那是「开了软标签但 99%% 样本仍在教 one-hot」的'
                        '静默半吊子）。与 Phase 4 item 3 的滑动窗口**正交**，'
                        '两者可叠加。 必须同时给 --soft-index，否则启动时报错')
    ap.add_argument('--soft-every', type=_at_least_one, default=1,
                    help='每 N 个 micro-batch 里有 1 个走软 CE（默认 1 = 每步都走）。'
                        'N>1 时其余步骤回退到 --policy-loss 的硬目标口径，'
                        '那几步里的软行会按 one-hot 训 —— 这正是 spec §5 退路表'
                        '点名的「混合训练 ⇒ index 0 向两种语义折中」，只在'
                        '明确知道后果时才用。必须 >= 1')
    ap.add_argument('--use-checkpoint', type=int, default=0, choices=[0, 1],
                    help='用 gradient checkpointing 减少显存占用（约省 50%%，训练慢 ~30%%）(0=关闭, 1=开启)')
    ap.add_argument('--use-ema', type=int, default=0, choices=[0, 1],
                    help='启用 EMA（指数移动平均）权重，eval/save 时用 shadow 权重，提升 1-3%% accuracy (0=关闭, 1=开启)')
    ap.add_argument('--gradient-accumulation-steps', type=int, default=1,
                    help='梯度累积步数（模拟更大 batch size，效果等同于 batch_size * N）')
    ap.add_argument('--scaler-init-scale', type=float, default=0.0,
                    help='GradScaler 初始缩放值（0=用 PyTorch 默认 65536）。'
                        '默认 65536 需要靠减半向下搜索平衡点，每次溢出白扔一个 '
                        'batch；实测本模型在 4 卡 旧多卡环境 上平衡于 512~2048，'
                        '故建议直接给 1024 起步。设 --scaler-growth-interval 0 '
                        '可关闭自动回涨，避免震荡反复偷步')
    ap.add_argument('--compile', type=int, default=None, choices=[0, 1],
                    help='用 torch.compile 融合算子（GPU 上约 20-40%% 提速，'
                         '首次迭代较慢）。\n'
                         '**不给 = 按设备自动**：CUDA 且 sm_80 以上（A100 等）'
                         '自动开 1，其余后端 0。本仓 CUDA 分支本来就把 BF16 + '
                         'channels_last + FlashAttn 都默认打开了，compile 却是'
                         '默认关的 —— 那是 加速器 时期「inductor 不可用」留下的默认值，'
                         '对 A100 是白丢的收益。\n'
                         '显式给 0/1 覆盖自动判定。compile 失败会自动回退 eager，'
                         '不会把训练带崩。')
    ap.add_argument('--compile-mode', default=None,
                    choices=['default', 'max-autotune', 'max-autotune-no-cudagraphs',
                             'reduce-overhead'],
                    help='torch.compile 模式（默认 None = 不显式指定，交给下方设备分支）。\n'
                         '  A100(sm_80+)：未显式指定时**自动取 max-autotune-no-cudagraphs**'
                         '（见设备分支）—— 拿到 autotune 的融合/调度收益、不背 CUDA Graphs\n'
                         '  的显存（40G 卡长跑 OOM 风险，已在注释里否决 reduce-overhead）。\n'
                         '  非 Ampere+ 或显式指定时尊重用户值：\n'
                         '  default                  = 常规融合，编译快。\n'
                         '  max-autotune             = 自动调优，但**带** CUDA Graphs；'
                         '私有内存池不归还，40G 卡上容易在长跑里把显存吃满。\n'
                         '  reduce-overhead          = 小 batch 低开销，同样受 '
                         'CUDA Graphs 内存池影响，长跑慎用。')
    ap.add_argument('--compile-fullgraph-probe', type=int, default=0, choices=[0, 1],
                    help='诊断用：编译时额外用 `fullgraph=True` 探测一次，把所有 '
                         'graph break 一次性列进日志（抛异常即逐条列出断点），'
                         '随后正常训练仍用 `fullgraph=False`。开启会多编译一次 '
                         '（max-autotune 下多花几分钟），仅排查时临时启用，'
                         '默认 0 = 不探测。配合已开启的 `log_graph_breaks` 看断点原因。')
    ap.add_argument('--gc-with-compile', type=int, default=0, choices=[0, 1],
                    help='开 --compile 时**仍然保留**梯度检查点（1=保留，0=照旧关掉）。'
                         '默认 0 是历史行为：compile 与 GC 被当成二选一'
                         '（见下面 `_gc = 0` 那段）。那条策略的来源是 加速器/旧编译栈 的'
                         '图捕获限制，不是 A100/CUDA 上的技术必然 —— 检查点走 '
                         '`use_reentrant=False`（`backbone._checkpointed` 已经是），'
                         '与 `torch.compile` 兼容，且 `backbone.'
                         'assert_grad_checkpoint_compile_compatible` 只在**子树里**'
                         '扫 `_orig_mod`，整模型 `torch.compile(model)` 把原模型包在内层，'
                         '扫不到、不会被拦。\n'
                         '为什么要开：关掉 GC 意味着激活全驻留，batch 被迫调小，'
                         '而 batch 缩小吃掉的吞吐通常远大于融合省下的 —— '
                         '40GB 卡上这是「要融合就没显存」的根源。'
                         '（V7 的 head 之前返回 dict、穿不过 checkpoint 的那个问题'
                         '已修，见 `katago_v7._ckpt`。）')
    ap.add_argument('--scaler-growth-interval', type=int, default=0,
                    help='连续多少个无溢出 step 后把缩放值翻倍')
    ap.add_argument('--flash-attn', type=int, default=1, choices=[0, 1],
                    help='flash-attn 独立库开关：A100(Ampere+) 上优先于内置 SDPA（最快，需 '
                        'pip install flash-attn），加载失败自动回退内置 SDPA。'
                        '1=自动优先（默认：A100 优先 flash-attn，其他后端不用）；'
                        '0=强制禁用（A100 也走内置 SDPA）。原环境变量副本 GOAI_FLASH 已于 2026-10-06 删除')
    ap.add_argument('--arch', default='resnet',
                    choices=['resnet', 'convnext'],
                    help='网络架构风格: resnet=传统 ResBlock (默认，兼容旧权重) | convnext=ConvNeXt 风格 (5x5 深度卷积 + LayerNorm + GELU)')
    ap.add_argument('--export-onnx', type=int, default=0, choices=[0, 1],
                    help='训练结束后导出 ONNX 模型（用于 CPU 推理加速）(0=关闭, 1=开启)')
    ap.add_argument('--onnx-quantize', type=int, default=0, choices=[0, 1],
                    help='ONNX int8 量化（模型体积 ~1/4，CPU 推理 ~2x）(0=关闭, 1=开启)')
    ap.add_argument('--swanlab', type=int, default=0, choices=[0, 1],
                    help='启用 SwanLab 实验跟踪 (0=关闭, 1=开启)')
    ap.add_argument('--swanlab-api-key', type=str, default='',
                    help='SwanLab API key（可选，未设置则读取 SWANLAB_API_KEY 环境变量）')
    ap.add_argument('--swanlab-every', type=int, default=0,
                    help='SwanLab 上报频率（每 N 步上报一次）。0=跟随 --log-every（默认，'
                        '与旧行为一致）；曲线太稀时设 5~10，例：--log-every 50 '
                        '--swanlab-every 10 可让曲线密度提升 5 倍而 stdout 不刷屏')
    ap.add_argument('--early-stop', type=int, default=0, choices=[0, 1],
                    help='启用早停机制：当验证集指标连续 N 次无改善时自动停止训练 (0=关闭, 1=开启)')
    ap.add_argument('--early-stop-patience', type=int, default=3,
                    help='早停耐心值：连续 N 次 eval 无改善则停止（默认 3）')
    ap.add_argument('--early-stop-metric', default='top1',
                    choices=['loss', 'top1'],
                    help='早停监控指标：loss=验证集损失（越小越好），top1=Top-1准确率（越大越好）')
    ap.add_argument('--max-gpu-memory', type=float, default=0.9,
                    help='GPU 显存使用上限比例（默认 0.9，防止 OOM）')
    ap.add_argument('--c2net', type=int, default=0, choices=[0, 1],
                    help='启用 C2NET (OpenI 启智平台) 支持 (0=关闭, 1=开启)')
    # ---- B8（V7 接线）：唯一的两个新旗都在这里 --------------------------------
    # **只有一个** `--v7`：V7 的其余选择（段位权重、batch、LR…）都不新增旋钮
    #   —— 段位权重是**代码里的常量表**（`V7_STAGE1_SCORE_TERMS`），不是 CLI。
    #   加成 CLI 会让「段 1 不训 score」变成一个可以被人顺手关掉的参数，而那条
    #   判据的依据是数据集事实（81.09% 的 SGF 是认输），不是调参口味。
    ap.add_argument('--v7', type=int, default=0, choices=[0, 1],
                    help='启用 V7 的 22 通道 NBT+Transformer 路径'
                        '（NbtTfNet + 12 项 KataGoV7Loss，0=关闭=12 通道旧路径，'
                        '**默认**；1=开启）。开启时：特征改由 feature_v7 的 '
                        'spatial_channels_v7/global_features_v7 在预取 worker 里造，'
                        'futurepos 自动 enable 并在**父进程 fork 前 warm**，'
                        '段 1 的四个目标 = policy / π_opp / value / futurepos，'
                        'score 系权重 0 但结构保留。'
                        ' 强制 --board-size 19（V7 通道数与 loss 里的 n_sq=19*19 '
                        '都是钉死的）。'
                        ' 与 --soft-index 互斥：V7 的 policy 目标走 '
                        '`policy_player`（人类 one-hot），软标签是段 2 的口径，'
                        '两条都开会静默互相覆盖 —— 给了就报错。')
    args = ap.parse_args()

    # ---- 分布式训练环境变量（由 torchrun / mp.spawn 注入）----
    # RANK/WORLD_SIZE/LOCAL_RANK 同时存在且 WORLD_SIZE>1 时进入 DDP 模式。
    # B1: rank/is_main/logger 必须在 C2NET 初始化前求值——下方 c2net 分支
    # 会用 is_main 过滤打印、用 logger 输出。
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_dist = world_size > 1
    is_main = (rank == 0)
    log_file = args.log_file if args.log_file else None
    logger = setup_logging(log_file, rank=rank)

    # ---- A4 · 软标签：kind 解析（**在加载数据集之前**）--------------------
    # `soft_ce` 是 `--soft-index` 的**派生**结果，不是又一个旗；坏组合
    # （soft_ce 却没有 --soft-index / huber 撞上 --soft-index）在这里就拒掉 ——
    # 等到第一个 batch 才炸，代价是先花几分钟把 12.3 GB 灌进内存。
    # 位置在 `setup_logging` 之后（要用 logger 报口径变化）、在
    #   `load_from_path` 之前（坏组合不该先吃满内存）。两处顺序都有测试钉。
    _soft_on = bool(args.soft_index)
    _hard_policy_loss = args.policy_loss
    # ⚠ 这里**不能**用 `v7_packed=bool(args.v7)`：`--v7 1` 只说明「走 22 通道
    #   路径」，**不等于**「数据是 stdata 分片」。board 级语料（full.npz，A/B 段）
    #   同样打 `--v7 1`，却**没有**内建软标签。传 `bool(args.v7)` 会让
    #   `resolve_policy_loss_kind` 误判成「有软标签来源」⇒ 提前把 kind 派生成
    #   `soft_ce`，于是在 board 级数据上打出一句**假的**「--soft-index 已给定」
    #   （2026-10-07 真机日志里就出现过），而 `soft_mask` 其实恒 0。
    #   V7 的真实分片判据要等 dataset 建好后用 `hasattr(dataset, 'sample_spatial')`
    #   （见本文件后面 `if _v7_on:` 那一次重新解析），那里才是唯一正确的口径。
    #   此处只做「加载数据集之前拒掉坏组合」这一件事，V7 的最终口径留给后面。
    _eff_kind = resolve_policy_loss_kind(args.policy_loss, args.soft_index,
                                        v7_packed=False)
    if _eff_kind != _hard_policy_loss:
        logger.info("[soft] policy 损失口径 %s → %s（--soft-index 已给定）",
                    _hard_policy_loss, _eff_kind)
    args.policy_loss = _eff_kind
    _soft_every = max(1, int(args.soft_every))

    # ---- B8 · V7：坏组合在加载数据集**之前**就拒掉 ----------------------------
    # 理由与上面 A4 那一段完全一样：等到第一个 batch 才炸，代价是先花几分钟把
    # 12.3 GB 灌进内存。四条约束逐条对应下面代码里的一个硬假设。
    _v7_on = bool(args.v7)
    if _v7_on:
        if args.board_size != V7_BOARD_SIZE:
            # 不是「警告后继续」：`KataGoV7Loss.forward` 里 `n_sq = 19 * 19` 与
            # `spatial_channels_v7` 的 `BOARD_SIZE` 都是**钉死的**，换盘面得到的
            # 是 reshape 形状错或静默错值，不是零回归。
            raise SystemExit(
                f'--v7 只支持 --board-size {V7_BOARD_SIZE}（V7 的 22 通道与 loss '
                f'里的 n_sq={V7_BOARD_SIZE * V7_BOARD_SIZE} 都是钉死的），'
                f'实得 {args.board_size}。')
        if _soft_on:
            # V7 的 policy 目标**按行二选一**（见 `KataGoV7Loss._soft_ce`）：
            #   soft_mask=1 → 用 `soft` 的分布做软 CE
            #   soft_mask=0 → 退回 `policy_player` 的 one-hot CE
            #
            # 这正是 A→B 能用**同一个模型**连续训练的前提：没命中软标签的行
            # 照样学人类着法，逐行语义与段 1 完全一致，不存在「两种口径混在
            # 一个 batch 里互相打架」。
            logger.warning(
                "[v7] + --soft-index：policy 目标按行二选一"
                "（soft CE / one-hot 回退）；加 --soft-only-sampling 1 "
                "则只采样命中行，软项量级与段 2 一致。")
        if args.value_loss_weight != 1.0:
            logger.warning("[v7] --value-loss-weight=%s 在 V7 上**无效**：V7 的"
                        "value 是 `KataGoV7Loss` 内的 3 分类 CE（#3，系数 1.20），"
                        "不经 `compute_value_loss`。该参数只作用于 12 通道路径。",
                        args.value_loss_weight)
        args.value_loss_weight = 1.0
        args.label_smoothing = 0.0
        args.policy_loss = 'ce'
        args.value_loss = 'huber'
        logger.info("[v7] 已启用 22 通道 V7 路径 | board=%d | 段 1 四目标=%s | "
                    "score 系权重 0（结构保留）",
                    V7_BOARD_SIZE, ' / '.join(V7_STAGE1_TERMS))
        logger.info("[v7] 验证集评估：走 `evaluate_metrics_v7`（22 通道 + dict 返回，"
                        "口径 top1/top5/top10/kl/brier 与 12 通道版同名同义）"
                        "⇒ `--early-stop` 与 best-model 判据均**生效**。")
        if args.export_onnx == 1:
            logger.warning("[v7] --export-onnx 走 `GoAI`，它是 12 通道推理链；"
                        "V7 需要另一条导出路径（本次未接），导出结果不可用。")

    # ---- C2NET 支持（OpenI 启智平台）----
    # 关于 rank 守卫：prepare() 与 --data 覆盖**必须**在所有 rank 上执行——
    # 每个 rank 都要独立加载数据集，而 --data 是 required=True，非 rank 0 若拿不到
    # c2net 给的 dataset_path 会直接 argparse 报错退出。因此这里只对「日志打印」
    # 做 is_main 过滤（否则 N 卡会刷出 N 份重复日志），prepare() 本身保持全 rank 调用。
    # 若 c2net 的 prepare() 将来被发现有写盘/建连副作用，正确做法是把它挪到
    # init_process_group 之后、由 rank 0 调用再用 dist.broadcast_object_list 广播
    # 路径，而不是简单地加 if is_main（那会丢掉非 rank 0 的数据集路径）。
    _c2net_ctx = None
    if args.c2net == 1:
        try:
            from c2net.context import prepare, upload_output as _c2net_upload
            _c2net_ctx = prepare()
            if is_main:
                logger.info("[c2net] 已初始化 C2NET 上下文")
                logger.info("[c2net] dataset_path=%s", _c2net_ctx.dataset_path)
                logger.info("[c2net] output_path=%s", _c2net_ctx.output_path)
            # 覆盖 --data：目录原样传入，让 --data 的目录模式合并**全部** npz/tgz 分片
            if _c2net_ctx.dataset_path:
                args.data = resolve_c2net_data(_c2net_ctx.dataset_path)
                if is_main:
                    if os.path.isdir(args.data):
                        import glob
                        _n_npz = len(glob.glob(os.path.join(args.data, '**', '*.npz'),
                                            recursive=True))
                        logger.info("[c2net] 使用数据集目录（合并全部分片）: %s"
                                    "（npz 分片 %d 个）", args.data, _n_npz)
                    else:
                        logger.info("[c2net] 使用数据集: %s", args.data)
        except ImportError:
            if is_main:
                logger.warning("[c2net] c2net 未安装，--c2net 已忽略")

    logger.info("=" * 60)
    _check_training_env(logger)
    logger.info("=" * 60)
    logger.info("配置: data=%s board=%d batch=%d epochs=%d lr=%s wd=%s",
                args.data, args.board_size, args.batch_size, args.epochs,
                args.lr, args.weight_decay)
    logger.info("注意力: mode=%s attn_mode=%s window=%d heads=%d layers=%d dropout=%s compile=%s",
                args.attention_mode, args.attn_mode, args.attn_window,
                args.num_heads, args.num_attention_layers, args.attention_dropout,
                # 设备分派在后面，这里 `args.compile` 还可能是 None（= 按设备
                # 自动），打 None 会让人以为编译被关了。真值见 [device] 那行。
                'auto' if args.compile is None else args.compile)
    logger.info("日志: log_every=%d eval_every=%d save_every=%d out=%s",
                args.log_every, args.eval_every, args.save_every, args.out)
    logger.info("验证集评估: eval_max_batches=%d（<=0 = 跑满全部验证集；日志 [eval] 行的 "
                "batches/truncated 即本次实际覆盖度）", args.eval_max_batches)
    # C2NET 输出路径重定向
    if _c2net_ctx is not None and is_main:
        _c2net_out = os.path.join(_c2net_ctx.output_path, os.path.basename(args.out))
        logger.info("[c2net] 输出路径已重定向: %s -> %s", args.out, _c2net_out)
        args._c2net_orig_out = args.out
        args.out = _c2net_out
    logger.info("=" * 60)

    # SwanLab 实验跟踪（可选，通过 --swanlab 启用）
    use_swanlab = args.swanlab == 1 or os.environ.get('SWANLAB_API_KEY')
    swanlab_logger = None
    if use_swanlab and is_main:
        swanlab_logger = _init_swanlab(args, logger)
        if swanlab_logger is None:
            logger.warning("[swanlab] 已降级为关闭；指标请看 stdout 日志"
                        "（每 --log-every 步一行：loss / p / v / lr / mem / spd）")
    if is_main:
        _se_eff = args.swanlab_every if args.swanlab_every > 0 else args.log_every
        logger.info("SwanLab: 启用=%s 已初始化=%s 上报频率=每 %d 步（--swanlab-every=%d，"
                    "0 表示跟随 --log-every=%d）",
                    bool(use_swanlab), swanlab_logger is not None,
                    _se_eff, args.swanlab_every, args.log_every)

    # ---- 数据集 + 预取 worker：**必须在设备初始化之前**（2026-09-30）----
    # 顺序是硬要求，不是风格问题：本段的 `mp.Process` 默认 fork，若排在
    # `init_process_group` / `torch.cuda.set_device` 之后，4 个 worker 会各自
    # 继承一份父进程的  设备上下文与显存映射 ⇒ 每卡 6.3 GB 的训练被旁挂到
    # 4×6 GB，实测 HBM 94% 而 计算单元 0%，紧接着就是 OOM。数据集加载是纯
    # numpy（与 rank 无关），提前无语义影响；反向顺序（dist 初始化后再 fork）
    # 才是 NCCL 的危险方向，提前 fork 是安全的那一侧。
    dataset = load_from_path(args.data, args.board_size,
                                args.max_games_per_tgz, v7=bool(args.v7),
                                games_npz=args.games_npz)
    # ---- V7 分片的软标签是**内建**的 ⇒ 这里复算一次 policy 口径 -------------
    # 上面那一次发生在数据集建好之前，只能看见 `--soft-index`；而段 3（stdata
    # 分片）的 `soft` / `soft_mask` 直接来自行内，不需要索引文件。
    #
    # 反过来也要把住：board 级 V7（A/B 段，数据来自 full.npz）**没有**内建软
    #   标签，没挂 `--soft-index` 时的 `soft_mask` 恒为 0 —— 那时必须拒掉，
    #   否则就是「以为在蒸馏、其实软项恒 0」。
    if _v7_on:
        _packed = hasattr(dataset, 'sample_spatial')
        _eff_kind = resolve_policy_loss_kind(
            args.policy_loss, args.soft_index, v7_packed=_packed)
        if _eff_kind != _hard_policy_loss:
            logger.info("[policy] 生效口径 %s（--policy-loss=%s，软标签来源=%s）",
                        _eff_kind, _hard_policy_loss,
                        'stdata 分片内建' if _packed else '--soft-index')
    # ---- A4 · 软标签挂载：**必须在 fork 之前** -----------------------------
    # 顺序是硬要求：软标签挂在 dataset 对象上，预取器 fork 之后再挂就只有父
    # 进程看得见，worker 会继续造 `soft_mask` 全 0 的批 —— 训练不报任何错，
    # 只是「软标签训了个寂寞」。
    if _soft_on:
        _sd = attach_soft_index(dataset, args.soft_index, data_npz=args.data)
        logger.info("[soft] 已挂载软标签 | 索引=%s | 覆盖行=%d/%d (%.3f%%) | "
                    "重复标注行=%d | 软 CE 步间隔=%d",
                    args.soft_index, _sd['n_soft'], _sd['n_rows'],
                    100.0 * _sd['hit_rate'], _sd['n_soft_dup'], _soft_every)
        if _sd['n_soft'] == 0:
            logger.error("[soft] 软标签命中 0 行。先确认索引与当前主 npz 匹配"
                        "（build_soft_index 会对陈旧缓存告警并重建），"
                        "**不要**把「索引陈旧」当成「这批局面真的没标签」—— "
                        "后者根本不会命中缓存。")
        if _soft_every > 1:
            logger.warning("[soft] --soft-every=%d：只有 1/%d 的 micro-batch 走软 "
                        "CE，其余步骤的软行按 one-hot 训 —— spec §5 退路表点名"
                        "这是「index 0 向两种语义折中」的场景，请确认这是有意的。",
                        _soft_every, _soft_every)
    elif _soft_every > 1:
        logger.warning("[soft] --soft-every=%d 但没有 --soft-index：该参数无作用",
                    _soft_every)

    # ---- B8 · futurepos 挂载 + warm：**必须在 fork 之前** --------------------
    # `warm_futurepos()` 是本次接线里唯一一处「顺序错了不报错」的调用：
    #   · 它做**惰性解析**（`_futurepos_boards` 首次调用才解析 boards 来源，可能
    #     触发 `materialize_dataset` 落 12.3 GB 的 `boards.npy`）。排在
    #     `_BatchPrefetcher` 之后 ⇒ 惰性解析发生在**每个 worker 里**
    #     ⇒ 12.3 GB × worker 数（还可能几个进程同时写同一路径互相踩坏）。
    #     实测同一份 gather：冷 IO 8 worker 273 行/s vs warm 1161 行/s（**4.3×**）；
    #     gather 冷读 10.4–24.7 ms/行 vs 热读 0.005–0.008（**1500×**）。
    #   · 本仓是 Windows 优先环境，`mp.Process` 走 **spawn**，worker 还会重新
    #     open 一次（`_rebind_futurepos_mmap`）⇒ 没 warm 过就没有
    #     `materialized_paths`，退化成继承句柄（spawn 下那是错的）。
    #
    # `source=None` ⇒ 走优先级第 2 步（直接用 `self.boards`），**零磁盘写**——这是
    # 本仓主 npz 的正确选择，数据已在内存里，再落一份 12.3 GB 是纯浪费。超大数据集
    # 走 `dataset_npz` + `materialized_dir`（`_rebind_futurepos_mmap` 覆盖的另一支）。
    #
    # `V7PackedDataset` 的 futurepos **就在分片里**（`futurepos` 键，19×19×2），
    # 不需挂载/预热——那是 board 级路径才需要的。给它一个「已完成」状态，让下游
    # 日志走同一形状。
    _fp_status = None
    if _v7_on:
        if hasattr(dataset, 'attach_futurepos'):
            _fp_status = dataset.attach_futurepos(source=None, mode='live')
            _fp_status = dataset.warm_futurepos()   # 必须在 worker 起来之前 warm
        else:
            _fp_status = {'mode': 'packed', 'offsets': None,
                        'resolved': True, 'source': 'shard:futurepos'}
        logger.info("[v7] futurepos 数据源 | mode=%s offsets=%s resolved=%s "
                    "source=%s", _fp_status.get('mode'),
                    _fp_status.get('offsets'), _fp_status.get('resolved'),
                    _fp_status.get('source'))

    pf = None
    if args.prefetch_workers > 1:
        pf = _BatchPrefetcher(dataset, num_workers=args.prefetch_workers,
                            prefetch=args.prefetch_depth, labels=_soft_on,
                            v7=_v7_on)
        logger.info("[data] 预取器已启用（在设备初始化之前 fork）| workers=%d depth=%d"
                    " | labels=%s | v7=%s", args.prefetch_workers, args.prefetch_depth,
                    _soft_on, _v7_on)
    else:
        logger.info("[data] 预取器已关闭（--prefetch-workers=%d ≤ 1）",
                    args.prefetch_workers)

    # ---- eval 特征预取池（V7 专用，2026-10-06）--------------------------------
    # 事故背景：eval 的 `v7_batch_sync` 在主进程**串行**算 22 通道特征（含
    # iterLadders 这个 CPU 大头），10 万行要磨 25-30 分钟，期间 加速器 归零、无任何
    # 日志——真机两次被当成「挂死」。本池与训练预取器同机制（同样在设备初始化
    # 之前 fork、同一份 `_prefetch_worker`，仅 `augment=False`），eval 批在
    # worker 间并行 ⇒ 同一批指标逐位不变，只把墙钟除以 worker 数。
    # 池在整个训练期常驻（eval 之间 idle 阻塞在队列上，无 CPU 开销）。
    eval_pf = None
    if _v7_on:
        _eval_workers = max(2, min(4, args.prefetch_workers or 4))
        eval_pf = _BatchPrefetcher(
            dataset, num_workers=_eval_workers,
            prefetch=max(2, min(args.eval_max_batches if args.eval_max_batches > 0
                                else 8, 32)),
            seed=777, labels=_soft_on, v7=True, augment=False)
        logger.info("[data] eval 预取池已启用（与训练预取器同点 fork）| "
                    "workers=%d | augment=False", _eval_workers)

    # ---- 分布式训练：设备由 LOCAL_RANK 决定，忽略 --device 卡号 ----
    # 后端固定为 nccl（CUDA）。多卡前必须 init_process_group，
    # 否则后续 .to(device) / DDP 包裹会失败或各卡不互通。
    if is_dist:
        _dist_backend = (args.device.split(':')[0]
                        if args.device not in ('auto', '') else
                        'cuda')
        # 必须在 init_process_group **之前**：DETAIL 是在建域那一刻给每个 PG 套
        # 一层一致性检查 wrapper（每次 collective 前一发 monitored_barrier）；
        # 其开销与用哪种包裹层无关，统一降为 OFF。
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        device = f'{_dist_backend}:{local_rank}'
        # 通信域是**惰性**创建的（上面两行只登记后端），所以主动试一发 all_reduce：
        # 否则通信建不起来要等到第一个 batch 的前向才炸，且报成 NCCL 通用错误
        # （真因藏在日志前面的 EJ0001 里）。见 _dist_preflight_check 的 docstring。
        _dist_preflight_check(_dist_backend, device, logger)
        if is_main:
            logger.info("[dist] 初始化分布式训练 | backend=%s world_size=%d | 策略=DDP"
                        "（各 rank 各持完整模型；每个 micro-batch 反向做 1 次梯度 "
                        "all-reduce；本文件全无 no_sync()，故梯度累积**不摊薄**同步"
                        "次数：accum=N 时每个 optimizer step 做 N 次）",
                        _dist_backend, world_size)
    else:
        if args.device == 'auto':
            device = _auto_select_device()
        else:
            device = args.device

    use_amp = args.use_amp == 1 or device.split(':')[0] == 'cuda'

    # ---- 后端自适应（CUDA / CPU）----
    #   - amp_dtype:       A100/A800/H100(sm_80+) -> bfloat16（原生支持）；
    #                     V100(sm_70, Volta) -> float16（无 bf16）
    #   - use_scaler:      BF16 下关闭（下溢不会发生 ⇒ scaler 不创建，
    #                     --scaler-init-scale 等成了死参数）；FP16 下必须开，
    #                     且两个参数仍要给（默认 0.0 = PyTorch 的 65536，大
    #                     batch 下必炸）
    #   - use_channels_last: A100 卷积走 NHWC 更快；CPU 收益有限，默认关
    #   - sdpa_force_math: A100 走 SDPA/FlashAttn（False），V100/CPU 强制手写
    #                     math（True）。随后由 --use-sdpa 总开关覆盖：0 强制全部
    #                     回退 math（调试/兼容），1（默认）保留上述默认。
    #   - compile_disable_sparse: 全后端统一禁用 —— unfold 产生
    #                     (B, Hh*d, N, ws²) 巨型中间张量，inductor 常量折叠会以
    #                     fp32 物化 (B,N,Hh,ws²,d)（batch512 下单个 4.3GB），
    #                     编译期直接 OOM。稀疏/窗口注意力走 eager+autocast，
    #                     编译图只覆盖卷积/线性/FFN。
    amp_dtype = torch.float16
    use_scaler = use_amp
    use_channels_last = False
    sdpa_force_math = True
    compile_disable_sparse = True
    gpu_name = 'N/A'
    compute_cap = (0, 0)
    _backend = device.split(':')[0]
    # 具体卡号（device 形如 'cuda:1' / 'cpu'），无索引时默认 0
    try:
        _dev_idx = int(device.split(':')[1]) if ':' in device else 0
    except ValueError:
        _dev_idx = 0
    if _backend == 'cuda' and torch.cuda.is_available():
        # ---- 三行 TF32 / autotune 设置（A100 与 V100 都受益）----
        # `set_float32_matmul_precision('high')` 本身**等价于**把 matmul 的
        # TF32 打开（它是 cudnn 无关的 matmul 侧总闸）。但它是**间接**的：
        # 读代码时看不出「matmul 到底有没有走 TF32」，而这个开关一旦被 torch
        # 的默认值改掉就是**静默**的性能回退（数值不变 ⇒ 测试全绿、只有吞吐
        # 掉）。所以两条都显式写出来，当同一件事的两份契约。
        # 影响面：只作用于**未被 autocast 覆盖**的 fp32 算子。BF16/FP16 走各自的
        # tensor core，与此无关。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        props = torch.cuda.get_device_properties(_dev_idx)
        gpu_name = props.name
        compute_cap = (props.major, props.minor)
        is_ampere_plus = compute_cap >= (8, 0)
        if is_ampere_plus:
            amp_dtype = torch.bfloat16
            use_scaler = False  # BF16 几乎不下溢，去掉 GradScaler 省一次 CUDA 同步
            use_channels_last = True
            sdpa_force_math = False  # A100 走 FlashAttention 后端
            # unfold 巨型中间张量会触发 inductor freezing 以 fp32 物化 (B,N,Hh,ws²,d)
            # 导致编译期 OOM（batch512 下单个 4.3GB），必须排除出编译图
            compile_disable_sparse = True
            # A100 默认把 compile 档提到 `max-autotune-no-cudagraphs`：autotune 的
            # 融合/调度收益、不背 CUDA Graphs 的显存（40G 卡长跑 OOM，见
            # --compile-mode 帮助与 4793 附近注释）。仅当用户**未显式**指定
            # --compile-mode（默认 None）时覆盖；显式给 'default' / 'reduce-overhead'
            # 等一律尊重用户意图。非 Ampere+ 分支不动（留给 torch.compile 默认档）。
            if args.compile_mode is None:
                args.compile_mode = 'max-autotune-no-cudagraphs'
            logger.info("[device] %s (sm_%d%d) | 启用 A100 路径: BF16 + FlashAttn(优先,回退SDPA) + "
                        "channels_last + compile(卷积/线性/FFN, mode=%s)",
                        gpu_name, *compute_cap, args.compile_mode)
            # flash-attn 的实际加载/回退统一在下方「flash-attn 独立库启用决策」块处理
            # （A100 默认优先尝试，--flash-attn 0 才禁用），此处不再重复。
        else:
            # V100 等老卡：保守路径（与原行为一致）
            amp_dtype = torch.float16
            use_scaler = use_amp
            use_channels_last = False
            sdpa_force_math = True
            compile_disable_sparse = True
            logger.info("[device] %s (sm_%d%d) | 走保守路径: FP16 + 手写 math 注意力 + "
                        "稀疏注意力禁用编译", gpu_name, *compute_cap)

        # `--compile` 不给时的自动判定。判据只用「CUDA 且 sm_80+」，与上面
        # 走 A100 路径的判据**故意用同一个**（`is_ampere_plus`）：那条路径已经
        # 把 BF16 + channels_last + FlashAttn 都默认打开了，compile 单独默认关
        # 只会让「A100 路径」的收益少一块。V100 等老卡维持默认关。
        if args.compile is None:
            args.compile = 1 if is_ampere_plus else 0
            logger.info("[device] --compile 未指定 ⇒ 按设备自动取 %d"
                        "（sm_%d%d，%s）",
                        args.compile, compute_cap[0], compute_cap[1],
                        'CUDA sm_80+ 默认开，显式 --compile 0/1 可覆盖'
                        if is_ampere_plus else '非 Ampere+ 默认关')
    else:
        # CPU 或其他：纯 FP32，无 AMP、无 channels_last
        amp_dtype = torch.float32
        use_scaler = False
        use_channels_last = False
        sdpa_force_math = True
        compile_disable_sparse = True
        logger.info("[device] CPU | 走 FP32 路径（无 AMP/编译）")

    # CUDA 分支已在上面把 `--compile` 的 None 收敛掉了；这里补齐其余后端，
    # 让后面所有分支（`_gc` 判定、`elif args.compile == 1`）都只看到确定的 0/1。
    if args.compile is None:
        args.compile = 0

    # 全局 SDPA 总开关（--use-sdpa）：在各后端选定天然默认后做一次统一覆盖，
    # 让该参数对 CUDA / CPU 全部后端生效。
    #   - 默认 1：保留各后端天然选择（A100=SDPA/FlashAttn，V100/CPU=手写 math）。
    #   - 0：强制所有后端回退手写 _sdpa_math（跨后端统一关闭融合注意力，便于
    #     调试数值差异 / 兼容不支持 SDPA 的环境）。
    if args.use_sdpa == 0 and not sdpa_force_math:
        sdpa_force_math = True
        logger.warning("[device] --use-sdpa 0：强制 %s 后端回退手写 math 注意力"
                    "（覆盖其天然默认）", _backend)

    # 把注意力后端/编译开关透传给 backbone 模块（所有分支统一设置）
    from src.networks import backbone as _backbone
    _backbone.set_sdpa_force_math(sdpa_force_math)
    _backbone.set_compile_disable_sparse(compile_disable_sparse)
    # config 面板回填真实注意力后端：swanlab.init 在设备分支之前已上传写死的
    # `attn_sdpa_force_math: True`，这里用运行时真值覆盖。 SDPA 放开后该键应为
    # False。API 不支持 init 后更新时静默忽略，启动日志才是真相源。
    # `amp_dtype` 同理：init 时写死的 'float16' 在 加速器 bf16 路径下是错的，回填真值。
    if swanlab_logger is not None:
        try:
            swanlab_logger.config.update({
                "attn_sdpa_force_math": bool(sdpa_force_math),
                "amp_dtype": str(amp_dtype).split('.')[-1],
            })
        except Exception:
            pass
    # 注意力 query 分块（2026-10-01）：math 路径下整条 (B,Hh,N,N) 分数矩阵
    # @N=361/4head/fp16 在 B=1000 时 0.97 GiB 一份、softmax+dropout 再各一份。
    # softmax 沿 key 轴 ⇒ 按 query 切块**数学精确**，峰值 ∝ chunk（默认 64 ⇒ 5.6×）。
    # 只加在 math 分支；SDPA/flash 融合路径自行管理显存，不受影响（加速器 默认已放开
    # 走  SDPA，故该分块在 加速器 上通常不再生效，见上方日志提示）。
    # （2026-10-06：三个融合/分块开关的环境变量已全部换成 CLI 参数）
    _attn_chunk = int(args.attn_query_chunk or 0)
    _backbone.set_attn_query_chunk(_attn_chunk)
    # 逐 chunk 梯度检查点（2026-10-04 新增）：分块只降瞬时峰值，**不降保留量** ——
    # 每块的 softmax 输出都被 autograd 存着等反向。逐块 checkpoint 把占大头的
    # 注意力矩阵变成「反向时一块一块重算」，**不依赖 block 级 checkpoint 是否生效**
    # （实测云端 旧多卡环境 上 64 GB ≈ 无 checkpoint 的估算值）。
    # 默认开；`--attn-chunk-ckpt 0` 关闭。
    _backbone.set_attn_chunk_checkpoint(int(args.attn_chunk_ckpt))
    # 加速器 融合算子总闸（置 0 只慢不坏，数值口径不变）。

    # online-softmax 注意力（flash 风格）：由 `--attn-online` 控制（默认关）。
    # 与 materialize 路径**数值等价但非逐位相同**（实测 fp32 max|Δ|≈7e-07），
    # 而 12 通路的 test_twelve_channel_path_bit_identical 正是靠逐位一致发现
    # 意外的数值变化 ⇒ 默认关闭，开关走命令行而不是环境变量（`--use-checkpoint`
    # 当年被「归档」就是因为开关只藏在 config 表里、没有干净入口）。
    #
    # 打开前建议先跑 `pytest tests/test_online_softmax_attn.py`（前向等价 + 梯度
    # gradcheck + bf16/autocast）。`dropout_p > 0` 时它自动回退 materialize
    # （online 不实现 dropout mask）；V7 的 attn_dropout 默认 0.0 ⇒ 那条不构成
    # 日常约束，也就是说开关一旦打开就**总是**生效。
    _backbone.set_attn_online(bool(args.attn_online))
    if args.attn_online:
        logger.warning("[model] online-softmax 注意力已开启：与 materialize 路径"
                    "数值等价但非逐位相同，12 通道 bit-identical 基线会失效")

    # flash-attn 独立库启用决策（A100 优先）：仅「Ampere+ CUDA 且走非 math 路径」时
    # 优先尝试加载 flash-attn，加载成功则 _sdpa 优先走 flash-attn 内核，失败回退内置 SDPA。
    # 开关就是 `--flash-attn`（1=A100 优先尝试，默认；0=禁用）——2026-10-06 起不再有
    # 环境变量副本（GOAI_FLASH 已删，CLI 完全覆盖它的语义）。
    _flash_disabled = (args.flash_attn == 0)
    if (not _flash_disabled) and _backend == 'cuda' and compute_cap >= (8, 0) and not sdpa_force_math:
        fa_ok, fa_msg = _backbone.set_flash_attn(True)
        if fa_ok:
            logger.info("[env] 注意力内核: flash-attn %s（A100 优先，回退 SDPA）", fa_msg)
        else:
            logger.info("[env] 注意力内核: 内置 SDPA（flash-attn %s）", fa_msg)
    else:
        _backbone.set_flash_attn(False)
        if _flash_disabled:
            logger.info("[env] 注意力内核: 内置 SDPA / 手写 math"
                        "（flash-attn 已被 --flash-attn 0 禁用）")
        else:
            logger.info("[env] 注意力内核: %s",
                        ("手写 math（%s 不支持 Flash）" % _backend.upper())
                        if sdpa_force_math else
                        "内置 SDPA")
            logger.info("[env] 注意力 query 分块: %s",
                        "关闭（或走 SDPA 路径不生效）" if _attn_chunk <= 0 else
                        "%d（仅手写 math 路径生效；峰值 ∝ chunk；eval 逐位不变，"
                        "训练态 dropout 取样位置变）" % _attn_chunk)

    logger.info("启动训练 | torch=%s | device=%s | amp_dtype=%s scaler=%s channels_last=%s",
                torch.__version__, device, amp_dtype, use_scaler, use_channels_last)

    n = len(dataset)
    # 按棋局分割 train/eval（避免同一棋局的相邻位置同时出现在 train 和 eval）
    if dataset.game_ids is not None:
        unique_games = np.unique(dataset.game_ids)
        rng = np.random.default_rng(0)
        rng.shuffle(unique_games)
        n_train_games = int(len(unique_games) * 0.98)
        train_game_set = set(unique_games[:n_train_games])
        train_idx = np.array([i for i in range(n) if dataset.game_ids[i] in train_game_set])
        eval_idx = np.array([i for i in range(n) if dataset.game_ids[i] not in train_game_set])
        logger.info("[data] 按棋局分割：总棋局=%d 训练棋局=%d 验证棋局=%d",
                    len(unique_games), n_train_games, len(unique_games) - n_train_games)
        n_train = len(train_idx)
    else:
        n_train = int(n * 0.98)
        idx_all = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(idx_all)
        train_idx = idx_all[:n_train]
        eval_idx = idx_all[n_train:]
    # ---- A4 · --soft-only-sampling：训练行空间收窄到 soft_row >= 0 ---------
    # 段 2 的决定（spec §1.2）：**只训软行**，而不是全量混训 / 靠权重混。理由
    # 是同一个 head 同时收到搜索分布与人类 one-hot 会去折中而不是学搜索；
    # `--soft-weight 0` 关不掉（那是「开了软标签但 99% 样本仍在教 one-hot」的
    # 静默半吊子）。与 Phase 4 item 3 的滑动窗口**正交**，两者可叠加。
    #
    # 只收窄**训练**行：`eval_idx` 不动 —— 验证集要能同时看软行与硬行。
    # 收窄后 `n_train` 必须跟着改（下面的每卡步数 / 调度步数 / 日志行数都由它
    # 推出），而 eval 侧的 `n_eval` 用的是 `len(eval_idx)`，不受影响。
    if args.soft_only_sampling:
        if not _soft_on:
            raise SystemExit(
                '--soft-only-sampling 需要 --soft-index：没有软标签时收窄到 '
                'soft_row>=0 会得到 0 行，而静默退回全量混训正是 spec §1.2 '
                '否决掉的方案。')
        _before = len(train_idx)
        train_idx = narrow_to_soft_rows(train_idx, dataset)
        n_train = len(train_idx)
        logger.info("[soft] --soft-only-sampling：训练行 %d → %d "
                    "(%.3f%%，只保留有软标签的行)", _before, n_train,
                    100.0 * n_train / max(_before, 1))
    logger.info("[data] 总样本数=%d | 训练=%d | 验证=%d", n, len(train_idx), len(eval_idx))

    # 分布式：每张卡用 DistributedSampler 取到不相交的训练分片（会自动 pad 到
    # 能被 world_size 整除），各卡步数因此一致，避免 DDP 在 barrier 处互相等待。
    if is_dist:
        train_sampler = torch.utils.data.DistributedSampler(
            torch.arange(len(train_idx)), num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

# ------------------------------------------------------ 结构（唯一）----
    # 训练结构**唯一** = `KATAGO_SE_CFG`（本文件顶部那张表，KataGo
    # SE-bottleneck 路线，形状喂给 `AlphaGoNet`）。
    # 旧结构 flag（--arch / --backbone-channels / --res-blocks / --policy-layers …
    # 与下面这一整串）**全部归档：不参与建网**，仍被 argparse 接受只为旧 shell
    # 不必改命令行（忽略而非报错，是这条约束的硬要求；运行日志里那句
    # 「结构参数已归档」就是这里）。
    # `--use-checkpoint` 同样归档：检查点开关 = `KATAGO_SE_CFG['grad_checkpoint']`
    # ∧「未开图编译」，见下。
    # 没有任何 arch 分支、也没有新增任何 CLI。
    _gc, _gc_reason = resolve_grad_checkpoint(
        KATAGO_SE_CFG['grad_checkpoint'],
        compile_on=(args.compile == 1),
        gc_with_compile=(args.gc_with_compile == 1))
    if _gc_reason == 'compile-exclusive':
        logger.warning(
            "[model] --compile 1 ⇒ 本次运行关闭 gradient checkpointing"
            "（配置的 %d，显存占用回升）；要同时开请加 --gc-with-compile 1",
            KATAGO_SE_CFG['grad_checkpoint'])
    elif _gc_reason == 'gc-with-compile':
        logger.info(
            "[model] --gc-with-compile 1 ⇒ compile 与 gradient checkpointing "
            "**同时**开启（grad_checkpoint=%d；检查点为 use_reentrant=False，"
            "与整模型 torch.compile 兼容）", _gc)
    elif args.use_checkpoint == 0:
        logger.info("[model] --use-checkpoint 已归档：检查点开关由 "
                    "KATAGO_SE_CFG[%r]=%d 决定，本次启用（如需关闭请用 --compile 1）",
                    'grad_checkpoint', _gc)
    _v7_lossf = None
    if _v7_on:
        # ---- B8 · V7（22 通道 NBT+Transformer）----------------------------
        # 形状**只**由 `NBT_TF_CFG` 决定（in_channels=22 / global_channels=19 /
        # board_size=19），与 CLI 上那些已归档的 12 通道结构旗无关。`--attention-dropout`
        # 仍是行为参数，照旧透传（V7 的默认是 0.0）。
        model, _eff_cfg = build_katago_v7_net(
            board_size=V7_BOARD_SIZE,
            use_checkpoint=_gc,
            attn_dropout=args.attention_dropout,
        )
        _v7_lossf = build_v7_stage1_loss(
            action_size=args.board_size * args.board_size + 1).to(device)
        # 启动前就把注意力显存算清楚（batch 3000 ⇒ 68.8 GB，真机实测 64 GB）
        if is_main:
            from src.networks.katago_v7 import MHSA as _MHSA
            _n_att = sum(1 for m in model.modules() if isinstance(m, _MHSA))
            v7_batch_memory_advice(
                args, logger, device, n_layers=_n_att,
                heads=NBT_TF_CFG['num_heads'],
                tokens=V7_BOARD_SIZE * V7_BOARD_SIZE,
                use_checkpoint=bool(_gc))
    else:
        model, _eff_cfg = build_katago_se_net(
            action_size=args.board_size * args.board_size + 1,  # +1 为 pass 类别
            attention_dropout=args.attention_dropout,           # 行为参数，仍生效
            grad_checkpoint=_gc,
        )
    model = model.to(device)
    # from-scratch 起手必须把 rank0 的初始权重广播给其余 rank（2026-10-01）。
    # 位置是硬要求：**必须在 EMA 构造（本文件下方 `EMA(model, ...)`）之前** ——
    # EMA 在构造时就把参数 `clone()` 进 `shadow`，放晚了 shadow 会持有广播前的
    # 随机权重，之后每个 step 的 `ema.update()` 都往这个陈旧 shadow 上混。
    # 也必须在 DDP 包裹之前：先建 EMA 再包裹，EMA 持有的引用就正好是 DDP 的
    # `module`，键空间一致（见下方包裹点上方的注释）。
    _sync_init_weights_from_rank0(model, logger)
    _assert_init_weights_identical(model, logger)
    n_params = sum(p.numel() for p in model.parameters())
    # V7 没有 `backbone` 子模块（11 个 `Nbt2TransformerBlock` 直接挂在 `blocks`
    # 上），主干参数量按「全网 − 三个头」算；12 通道路径仍读 `model.backbone`。
    if _v7_on:
        _n_backbone = n_params - sum(
            sum(p.numel() for p in m.parameters())
            for m in (model.policy_head, model.value_head, model.scorebelief_head))
    else:
        _n_backbone = sum(p.numel() for p in model.backbone.parameters())
        # 实测 vs 锚：改了 `KATAGO_SE_CFG` 之后这两个数必须重数（构建器返回的
        # `_eff_cfg` 里也有一份）。对不上说明表被改过而锚没更新 —— 直接在这里响，
        # 别等到几天后拿一个「莫名涨了 2M 参数」的 run 去比 loss。
        if (n_params, _n_backbone) != (_eff_cfg['params_total'],
                                    _eff_cfg['params_backbone']):
            logger.warning(
                "[model] 参数量与 KATAGO_SE_CFG 的预算锚不符：实测 %d / %d"
                "（全网 / 主干），锚 %d / %d。改了结构表就要重数这两个数。",
                n_params, _n_backbone,
                _eff_cfg['params_total'], _eff_cfg['params_backbone'])
    # 逐段开关也打出来：「grad_checkpoint=1」只说明**总开关**，看不出哪几段真的在
    # 走检查点（per-kind 默认可以不同；且训练态闸门还要求 self.training +
    # grad enabled）。这段日志的用处是让「GC 到底生效没有」不必翻代码，也不必
    # 在云端日志里靠猜 —— 4×旧多卡环境 首跑要看的就是它。
    # mixin 宿主有两种拓扑：12 通道挂 `model.backbone`，V7（NbtTfNet 自己
    # 继承 mixin、没有 `.backbone`）挂 `model`。原来只查后者 ⇒ V7 永远打 `n/a`，
    # 而这段日志的全部用处就是让人不翻代码就看出 GC 有没有生效。
    _gc_owner = model if hasattr(model, 'grad_checkpointing_kinds') \
        else getattr(model, 'backbone', None)
    _gc_kinds = getattr(_gc_owner, 'grad_checkpointing_kinds', None)
    _gc_kinds = _gc_kinds() if callable(_gc_kinds) else {}
    if _v7_on:
        # 预算锚不在本表里（`NBT_TF_CFG` 带的是 stem/block/head 的**分项**锚，
        # 主干合计与全网合计由 tests/test_katago_v7_budget.py 钉），所以这里
        # 只打实测值并指向那个测试，不做「与锚不符」的判定 —— 否则每次都误报。
        logger.info("[model] v7 NBT2+Transformer (%dch / %d 段 / %d 头 / %dch 空间"
                    "+%dch 全局) 参数量=%.2fM（主干 %.2fM）| grad_checkpoint=%d"
                    " | 设备=%s（预算锚见 tests/test_katago_v7_budget.py）",
                    _eff_cfg['trunk_channels'], _eff_cfg['num_blocks'],
                    _eff_cfg['num_heads'], V7_SPATIAL_CHANNELS, V7_GLOBAL_CHANNELS,
                    n_params / 1e6, _n_backbone / 1e6, _gc, device)
    else:
        logger.info("[model] %s (%dch / %d 段 / mix+%d 注意力) 参数量=%.2fM"
                    "（主干 %.2fM）| grad_checkpoint=%d | 设备=%s",
                    _eff_cfg['arch'], _eff_cfg['channels'], _eff_cfg['blocks'],
                    _eff_cfg['num_attention_layers'],
                    n_params / 1e6, _n_backbone / 1e6, _gc, device)
    logger.info("[model] GC 逐段开关=%s | 生效还需 training 态+grad enabled"
                "（eval/推理恒不检查点，零开销）", _gc_kinds or 'n/a')

    # 把卷积型特征（N,C,H,W）转 channels_last(NHWC)，卷积算子走更快内存布局。
    # 输入也需同步转格式（见训练/评估循环），故这里仅转换模型权重布局。
    if use_channels_last and _backend == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        logger.info("[model] 已启用 channels_last (NHWC) | backend=%s v7=%s",
                    _backend, _v7_on)

    # 参数组：value head 独立 LR（参数量小，需要更高学习率补偿梯度不足）
    # 排除 bias / BatchNorm / LayerNorm 参数的 weight decay（标准做法）
    _opt_groups = _build_param_groups(model, args)
    # A1: CUDA(A100) 启用 fused AdamW（单 kernel 融合 param 更新，省启动开销）；
    # CPU 走默认实现。P4.6 起这段决策收进 build_adamw：设备策略、构造、回退契约
    # （fused 不可用 ⇒ 标准实现，bit-for-bit 等于旧路径）集中在唯一入口，全模块
    # 不再有第二处 AdamW 构造点。
    optimizer, _opt_mode = build_adamw(_opt_groups, device, logger)
    # zero_grad 用 set_to_none=True（省一次 memset，CUDA fused AdamW 亦支持）。
    _zero_set_none = True
    # BF16 后端（A100）下 use_scaler=False（BF16 不下溢，省去 loss scaling 的额外同步）；
    # V100/FP16 下开启 GradScaler。按设备选择 GradScaler 实现。
    # 缩放值策略可配：从 65536 起步要靠减半向下搜索平衡点，每次溢出都白扔一个
    # batch；已知平衡点后直接给 init_scale 并关掉回涨，可消除这段浪费与后续震荡。
    _scaler_kwargs = {}
    if args.scaler_init_scale and args.scaler_init_scale > 0:
        _scaler_kwargs['init_scale'] = float(args.scaler_init_scale)
    if args.scaler_growth_interval and args.scaler_growth_interval > 0:
        _scaler_kwargs['growth_interval'] = int(args.scaler_growth_interval)
    # torch.amp.GradScaler 在 torch 2.4+ 可用，旧版本用 torch.cuda.amp.GradScaler
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler(
            _backend, enabled=use_scaler, **_scaler_kwargs)
    else:
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_scaler, **_scaler_kwargs)
    if use_scaler and _scaler_kwargs:
        logger.info("[fp16] GradScaler 初始缩放 %.0f，回涨间隔 %s 步",
                    scaler.get_scale(),
                    args.scaler_growth_interval or '默认 2000')

    # EMA（指数移动平均）：eval/save 时用 shadow 权重，提升 1-3% accuracy
    ema = EMA(model, decay=0.999) if args.use_ema == 1 else None

    # ---- 学习率调度：基于“总 step 数”而非 epoch 数 ----
    # 旧版用 T_max=args.epochs 导致余弦在第 1 个 epoch 结束就被砍到 ~0，
    # 后续 epoch 在 lr≈0 附近横盘。这里用真实总 step 数，并加前 5% step 线性 warmup。
    # DDP 下“每卡”样本数约为 n_train/world_size，调度按每卡步数推进，
    # 这样各卡 LR 曲线一致；有效全局 batch = batch_size * world_size。
    if is_dist:
        per_rank = (n_train + world_size - 1) // world_size
        n_batches = (per_rank + args.batch_size - 1) // args.batch_size
    else:
        n_batches = (n_train + args.batch_size - 1) // args.batch_size
    # 调度器的步数口径必须是 **optimizer step**，不是 micro-batch（2026-10-01 修）。
    #   `scheduler.step()` 只在每个 optimizer.step() 之后调一次（见训练循环），而
    #   `n_batches` 是 micro-batch 数。accum=1 时两者相等，accum>1 时差 accum 倍：
    #   原来 total_steps/warmup/T_max 全按 micro-batch 算 ⇒ SequentialLR 的
    #   milestone 提前到达（甚至在 epoch 内就切进 cosine），而 CosineAnnealingLR 的
    #   T_max 却是真实步数的 accum 倍 ⇒ **余弦只走完 1/accum 就到 epoch 末**。
    #   实测后果：lr 在 epoch 末仍停在峰值的 ~77%（accum=2），本该退火到 ~0。
    #   这个 bug 只在用 `--gradient-accumulation-steps` 时出现，所以 accum=1 的
    #   历史 run 完全正常 —— 也因此更容易漏掉。
    micro_per_epoch = n_batches
    n_batches = (micro_per_epoch + max(1, args.gradient_accumulation_steps) - 1) \
        // max(1, args.gradient_accumulation_steps)
    total_steps = max(1, args.epochs * n_batches)
    warmup_steps = max(1, int(total_steps * 0.10))
    after_warmup = max(1, total_steps - warmup_steps)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=after_warmup)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    # scheduler 首次 step 在第一个 optimizer.step() 之后执行，避免 PyTorch 警告

    bs = args.batch_size
    # V7 的动作空间（+1 为 pass 类别）。与 `build_katago_se_net` 的算法式
    # `board_size*board_size+1` 同源，但在 V7 上它同时是 `KataGoV7Loss` 的
    # `action_size`（policy_logits 的最后一维）⇒ 必须一致，故取自同一个常量。
    _v7_action_size = args.board_size * args.board_size + 1
    # V7 的逐项 loss（沿用上一次 micro-batch 的值，供打点用；与 `_health_last`
    # 同样的「跨 micro-batch 沿用」语义）。空 dict = 还没跑过任何一步。
    _v7_terms_last = {}
    # SwanLab 用的逐项 dict（键名带 `loss_v7/` 前缀）。**必须在循环外初始化**：
    # 它无条件进 `swanlab_logger.log({...})` 的字面量，而 12 通道路径永不给它赋值
    # ⇒ 不初始化就是每步一次 UnboundLocalError，被 `except` 吞成一行 warning ⇒
    # **SwanLab 静默丢掉整块指标而训练照跑**（与 2026-10-01 那次 `_accum_steps`
    # 同类的坑，见下方注释）。
    _v7_terms_swanlab = {}
    # 「哪一项算坏了」的去重表 + 计数（2026-10-04 云端 旧多卡环境）。
    #   同一种坏项组合只 `logger.error` 一次，其余走 debug —— 否则 100% 跳步时
    #   每步刷一遍同样的文本，把唯一有用的那行信息淹掉。
    _v7_bad_seen = []
    _err_cnt = 0
    step = 0
    best_eval_acc = -1.0
    start_epoch = 0
    t0 = time.time()
    # 本进程起点。resume 会把 step 覆盖成 checkpoint 里的累计值，原式
    # speed = step * bs / (now - t0) 于是把「累计步数」当成「本进程步数」；
    # 且 bs 是每卡值未乘 world_size —— 两重错误。实测续训时该式输出
    # 1059 s/s，而真实速率约 2635 s/s。
    _step_at_start = 0
    # 瞬时速率基准：只在 stdout 打点时推进，保证与打印行同口径
    # （_should_log 会因 swanlab_every 让打点块每 10 步触发一次，
    #   若跟着它推进，瞬时速率就会变成 10 步口径、与打印的 50 步区间不符）。
    _last_stdout_t = t0
    _last_stdout_step = 0

    # 早停机制初始化
    early_stop_counter = 0
    best_eval_metric = float('inf') if args.early_stop_metric == 'loss' else -1.0
    # 停止标志：判定只在 rank0 做，但「停不停」要所有 rank 一致（否则 rank0 跳出
    # 训练、其余 rank 还在 step 循环里做反向 → collectives 次序错配 → 挂死）。
    # 每个 eval 点由 _sync_stop_flag 无条件 broadcast 回所有 rank，step/epoch 两层
    # break 都读它。非 DDP 也建（单一代码路径，不写两套）。
    stop_flag = torch.zeros(1, dtype=torch.long, device=device)

    # ---- 断点续训：从 --resume 指定的模型权重 + 同目录 .train_state.pt 恢复 ----
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume 指定的模型不存在: {args.resume}")
        state_path = args.resume + '.train_state'
        if not os.path.isfile(state_path):
            # 兼容训练中途 save_every 的快照命名：<out>.latest.train_state
            _latest = args.resume + '.latest.train_state'
            if os.path.isfile(_latest):
                state_path = _latest
                logger.info("[resume] 使用中途快照训练状态: %s", _latest)
        logger.info("[resume] 加载模型权重: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        # 键对齐交给 _load_model_state：checkpoint 与目标模型两侧都归一到未编译布局
        # 再写回，compile 存档（顶层 _orig_mod. 或 Linear-only 的中段 _orig_mod.）
        # 与普通存档任意组合都能灌进来。
        _load_model_state(model, ckpt, logger)
        if os.path.isfile(state_path):
            tstate = torch.load(state_path, map_location=device)
            _load_optimizer_state(optimizer, tstate['optimizer'], logger)
            scheduler.load_state_dict(tstate['scheduler'])
            try:
                scaler.load_state_dict(tstate['scaler'])
            except (RuntimeError, KeyError):
                logger.warning("[resume] scaler 状态不兼容（可能是 BF16→FP16 切换），从头开始")
            step = tstate.get('step', 0)
            # 续训基准：本进程从 checkpoint 的累计步数起步
            _step_at_start = step
            _last_stdout_step = step
            best_eval_acc = tstate.get('best_eval_acc', -1.0)
            start_epoch = tstate.get('epoch', 0)
            if 'rng' in tstate:
                torch.set_rng_state(tstate['rng'].cpu())
            # 恢复 EMA shadow 状态
            if ema is not None and 'ema_shadow' in tstate:
                shadow, moved = _migrate_ema_shadow_qkv(tstate['ema_shadow'])
                if moved:
                    # 旧 ckpt 的 shadow 是 attn.q/k/v 三键，MHSA 内部已合成
                    # qkv（见 katago_v7.MHSA）。不迁移会在这里之后第一次
                    # EMA.update() 时抛 KeyError —— 那种崩法离现场很远，
                    # 所以显式记一条。
                    logger.info("[resume] EMA shadow: %d 处 attn.q/k/v 已合并为 qkv",
                                moved)
                # 丢弃「模型里根本没有」的 shadow 项。
                # 判据是**对 named_parameters 求差**，不是按 `.q.weight` 之类的
                # 后缀猜 —— 后缀那种写法在今天两个现役模型上都恰好安全（两者
                # named_parameters 里 `.q/.k/.v.weight` 都是 0 个），但哪天有
                # 模块真的加了 `.q.weight` 参数，删掉的就是**活参数**的 shadow
                # 项，`EMA.update()` 第一步就 KeyError。
                #
                # 为什么这些多余项无害、可以放心丢：`EMA.update()` 是遍历
                # `model.named_parameters()` 再去 shadow 取键，**从不遍历 shadow
                # 自己的键**，所以多余项从来不会被读到（既不会算错也不会报错）。
                # 反过来「模型有、shadow 没有」才会 KeyError —— 那才是要防的方向。
                # 典型来源：某次 snapshot 恰好在 qkv 迁移过程中落盘，于是
                # `attn.qkv.weight` 与残留的 `attn.q/k/v.weight` 并存；不清掉
                # 它们会一直躺在每个 checkpoint 里白占空间。
                _known = {_ema_key(n) for n, _ in model.named_parameters()}
                _stale = [k for k in shadow if k not in _known]
                if _stale:
                    shadow = {k: v for k, v in shadow.items() if k in _known}
                    logger.warning(
                        "[resume] EMA shadow: 丢弃 %d 个模型中不存在的项（%s…）",
                        len(_stale), _stale[:3])
                ema.shadow = shadow
                logger.info("[resume] 恢复 EMA shadow 状态")
            elif ema is not None:
                logger.warning("[resume] 未找到 EMA shadow 状态，EMA 从头开始")
            logger.info("[resume] 恢复训练状态 | step=%d best_eval_acc=%.4f epoch=%d",
                        step, best_eval_acc, start_epoch)
        else:
            logger.warning("[resume] 未找到 %s（仅恢复模型权重，optimizer/scheduler 从头开始）",
                        state_path)

    # ---- 仅加载模型权重（不加载 optimizer/scheduler/step）----
    if args.model:
        if not os.path.isfile(args.model):
            raise FileNotFoundError(f"--model 指定的模型不存在: {args.model}")
        logger.info("[model] 仅加载模型权重: %s（optimizer/scheduler 从头开始）", args.model)
        ckpt = torch.load(args.model, map_location=device)
        # 同 resume：键对齐统一走 _load_model_state。
        _load_model_state(model, ckpt, logger)
    elif args.compile == 1:
        if hasattr(torch, 'compile'):
            try:
                # 4 个注意力块实例 × window/sparse 两个禁用点 × train/eval 两态，
                # dynamo 按 fn 身份分缓存，默认上限 64 会被打满触发反复重编译。
                # 注意：函数内不能写 `import torch._dynamo`（会把 torch 绑定为
                # 局部变量，导致函数开头 UnboundLocalError），用 importlib 规避。
                import importlib
                _dynamo_mod = importlib.import_module('torch._dynamo')
                _dynamo_mod.config.cache_size_limit = max(
                    256, _dynamo_mod.config.cache_size_limit)
            except Exception:  # noqa: BLE001
                pass
            # ---- Inductor：编译缓存 + graph break 检查 ----
            # fx_graph_cache：同配置（图结构 + 输入 shape/dtype 哈希）下复用已编译
            # 的 kernel，避免每次重启重编译（max-autotune 动辄几分钟）。缓存落在
            # `~/.cache/torch/inductor`，换 batch/模型会失效，需同配置复用。
            # 注意：该特性 **torch 2.2+** 才有；本仓云端是 2.1.0，`fx_graph_cache`
            # 属性不存在 ⇒ 下面用 hasattr 显式判定，缺失时报警而不是静默 no-op。
            # log_graph_breaks：compile 时逐条把 graph break 及原因写进日志——
            # 有断点的子图会**静默回退 eager**，融合收益没全拿到（见下方探测）。
            try:
                _ind_cfg = torch._inductor.config
                if hasattr(_ind_cfg, 'fx_graph_cache'):
                    _ind_cfg.fx_graph_cache = True
                    logger.info("[compile] 已启用 Inductor FX 图缓存（重启免重编译）")
                else:
                    logger.warning(
                        "[compile] 当前 torch %s 不支持 fx_graph_cache（需 >=2.2），"
                        "编译缓存未启用；升级 torch 可免去每次重启的 max-autotune 重编译。",
                        torch.__version__)
                _ind_cfg.log_graph_breaks = True
            except Exception:  # noqa: BLE001
                pass
            # 可选 fullgraph 探测：`fullgraph=True` 编译一次，把所有 graph break
            # 一次性抛出来（异常里逐条列出断点），随后正常训练仍用 fullgraph=False。
            # 仅 `--compile-fullgraph-probe 1` 时启用，避免生产路径重复编译。
            if getattr(args, 'compile_fullgraph_probe', 0):
                try:
                    torch.compile(model, dynamic=False,
                                  mode=args.compile_mode, fullgraph=True)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[compile] fullgraph 探测发现 graph break（不影响训练，"
                        "已回退 fullgraph=False）:\n%s", e)
            try:
                # `backend` 不显式给：`torch.compile` 的默认 backend 就是 inductor，
                # 写死反而会在需要临时切 eager/aot_eager 排查时多一处要改的地方。
                #
                # ⚠ `mode` 由 `--compile-mode` 给，**默认不是 `reduce-overhead`**。
                #   `reduce-overhead` 会启用 **CUDA Graphs**：它把整段 kernel 录到一张
                #   私有内存池里，**该池不归还给 allocator**。2026-10-08 的 A100 40G
                #   实测 `mem=40.28GB` —— 卡只有 40GB，**已经没有余量**放 cudagraph
                #   池，长跑必然 OOM（而且 OOM 点会漂移，极难复现）。
                #   要自动调优就用 `max-autotune-no-cudagraphs`：拿到 autotune 的
                #   收益、不背 CUDA Graphs 的显存。
                #   另：训练循环里交错着 eval / 保存 / 梯度累积，控制流不是静态的，
                #   cudagraph 的录制与重放对这点很敏感。
                model = torch.compile(model, dynamic=False, mode=args.compile_mode)
                # 预热前向必须与真实训练一致地包 autocast：flash-attn 只接受 fp16/bf16，
                # FP32 输入直灌会报 "FlashAttention only support fp16 and bf16 data
                # type"，导致 compile 被误判为不可用而回退 eager。
                with torch.no_grad(), maybe_autocast(device, amp_dtype):
                    # flash-attn 只接受 fp16/bf16，FP32 输入直灌会被误判为
                    # compile 不可用而静默回退 eager；形状按 forward 签名分派
                    # （同上，就地写而不是抽 helper）。
                    if _v7_on:
                        model(torch.zeros(1, V7_SPATIAL_CHANNELS, args.board_size,
                                        args.board_size, device=device),
                            torch.zeros(1, V7_GLOBAL_CHANNELS, device=device))
                    else:
                        model(torch.zeros(1, KATAGO_SE_CFG['in_channels'],
                                        args.board_size, args.board_size,
                                        device=device))
                logger.info("[train] 已启用 torch.compile 算子融合")
            except Exception as e:  # noqa: BLE001
                # 真正回退 eager：剥离 OptimizedModule 包装，恢复原始模块引用
                model = getattr(model, '_orig_mod', model)
                logger.warning("[train] torch.compile 不可用，回退 eager: %s", e)
        else:
            logger.info("[train] 当前 torch 版本不支持 torch.compile，跳过")

    # 分布式：DDP 包裹需在 torch.compile 之后（算子融合与梯度同步可共存；反过来
    # compile 会被 wrapper 的动态边界吞掉，拿到的是未融合图）。
    #
    # 为什么用 DDP 而非分片包裹（2026-10-01 换轨，FSDP1 已退役）
    # ------------------------------------------------------------------
    # 触发事件：4 卡训练崩在 `ema.update()` 的
    # `KeyError: 'backbone.stem_bn._fsdp_wrapped_module.weight'`。分片包裹会把
    # **内部**模块就地换成 wrapper，于是「包裹前建好的 EMA shadow」与「包裹后
    # `named_parameters()` 遍历出来的键」不再是同一个键空间 ⇒ 每步都 KeyError。
    # DDP 只在**顶层**加一层（键多一个 `module.` 前缀），内部模块树原样不动 ⇒
    # 这整类崩溃消失，不需要任何针对它的补丁。
    #
    # 两个数字（9,067,443 参数 = fp32 36.3 MB / 32 GiB 卡）：
    #   1. 显存：FSDP1 每 rank 约 145 MB（参数/梯度/Adam 两矩各分片），DDP 约
    #      36.3 MB × 4（每项一份全量），差约 109 MB = 0.33% 的卡。参数本来就
    #      装得下，这点余量不值得拿正确性风险换。
    #   2. 通信：FSDP1 每步是「16 个分片单元 × 2 次 collective」，DDP 每次
    #      **反向** 1 次梯度 all-reduce + 1 次 buffer broadcast。注意「每步 1 次」
    #      是错的说法 —— 本文件全无 `no_sync()`（同源说明见
    #      `_compute_l2_report` docstring）：每个 micro-batch 都 `backward()`，
    #      all-reduce 挂在每次 backward 的收尾上 ⇒ `GRAD_ACCUM=2` 下每个
    #      optimizer step 是 2 次 all-reduce + 2 次 broadcast（≈72.9 MB）。仍是
    #      「16 × 2」的零头，通信不是瓶颈（本仓库瓶颈在算子，见 `[profile]`）。
    #
    # 为什么上面两个同步函数必须留在包裹点**之前**（顺序的硬要求）：
    # `DistributedDataParallel.__init__` 在**它自己构造时**（`_ddp_init_helper` →
    # `_sync_module_states`）就广播 rank0 的 params/buffers，而构造发生在
    # **EMA 构造之后** ⇒ rank1~3 的 shadow 会是各自被丢弃的随机权重，
    # `ema.update()` 每步把正确权重混进陈旧 shadow ⇒ **EMA 跨 rank 发散**，且
    # 不报错。`_sync_init_weights_from_rank0` / `_assert_init_weights_identical`
    # 在 `.to(device)` 之后、EMA 之前，恰好堵住它。
    #
    # 构造参数**只有** `device_ids`，其余全默认（不新增任何 CLI flag）：
    #   · `find_unused_parameters=False`：两个头每 step 都参与 loss ⇒ 所有参数都有
    #     梯度。这是**隐含前提**：将来若出现「某个头不参与 loss」的分支，DDP 会抛
    #     `Expected to have finished reduction in the prior iteration` —— 好在它
    #     是**响亮**地失败，不会安静地错。
    #   · `broadcast_buffers=True`：BN 的 `running_mean`/`running_var` 跨卡一致
    #     靠它；上一代默认也是 True ⇒ 行为不变。
    #   · `gradient_as_bucket_view=False`：本文件用 `zero_grad(set_to_none=True)`，
    #     bucket view 的别名每轮被销毁、省不掉拷贝，收益仅 ~36.27 MB/rank
    #     （= 全部梯度大小，torch 对该开关的定义就是如此；占 32 GiB 的 0.106%），
    #     真正的风险是混用 view / 非 view 的 grad 状态触发
    #     `Expected to mark a variable ready only once`。收益配不上这类风险。
    #   · `static_graph=True`：DDP 要求**每个 iteration 参与反向的参数集合完全一致**
    #     —— 本模型满足：两个头每步都进 loss（见上方 find_unused_parameters=False
    #     的隐含前提）、loss 的系数门控是**按 run 而非 per-step** 恒定（`katago_v7_loss`
    #     里 `self.coeff` 在构造时定、训练中不切）、全仓无任何 `requires_grad` 切换。
    #     静态图让 DDP 跳过每轮 unused-param 复检、固化 bucket 分配，省一点反向开销。
    #     风险是 fail-loud：将来若引入「某步某参数不进图」的分支，DDP 会立刻报错
    #     （而非安静错）。下方断言先把「全参数 requires_grad」这个前置不变量钉住。
    if is_dist:
        # static_graph 前置不变量：所有参数都参与反向（DDP 要求每步参数集合一致）。
        # 若将来出现冻结参数，应在此前显式处理；当前全仓不冻结任何参数。
        _frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        if _frozen:
            raise RuntimeError(
                'static_graph=True 要求所有参数 requires_grad，但发现 %d 个冻结参数'
                '（static_graph 不允许 iteration 间参数集合变化）：%s'
                % (len(_frozen), _frozen[:4]))
        model = DistributedDataParallel(
            model, device_ids=[local_rank], static_graph=True)

    # `_accum_steps` 必须定义在**任何**用它之前（2026-10-01 云端教训）。
    #   它原先在下面训练循环的开头才赋值，而上面「run 级指标上报」已经用它算
    #   effective_batch —— 晚 29 行 ⇒ `UnboundLocalError: local variable
    #   '_accum_steps' referenced before assignment`。那次上报被
    #   `except Exception` 吞成一行 warning，于是 **SwanLab 少了 8 个 run 级
    #   指标、训练照跑**，没人会发现。这里把定义提到所有使用点之前，并且
    #   下面那处赋值删掉（全仓库只此一处定义）。
    _accum_steps = max(1, int(args.gradient_accumulation_steps))

    if is_main:
        # 两个口径都打：调度器按 optimizer step 走，读日志的人常按 micro-batch 想。
        # 差别就是 gradient_accumulation_steps 倍（accum=1 时相等）。
        logger.info("[train] 开始训练 | optimizer-steps/epoch=%d | micro-batches/epoch=%d"
                    " | 总 steps≈%d | warmup=%d | accum=%d",
                    n_batches, micro_per_epoch, total_steps, warmup_steps,
                    _accum_steps)
        if swanlab_logger is not None:
            # run 级事实一次性上报：这些量在 config 面板里给不出（init 时模型还没
            # 建、数据还没切分），但对比两次 run 时它们是最先要看的。
            #
            # `run/params_actual` 是 config 里 `arch/params_total` 的**真值**：
            #   V7 的 `NBT_TF_CFG['params_total']` 是**预算值** 5,561,832，而实测
            #   建出来是 **5,562,121**（差 289）。模型在这里已经建好 ⇒ 能给真值。
            #   两者并列才看得出「预算表与实现漂了」。
            _run_facts = {
                "run/optimizer_steps_per_epoch": n_batches,
                "run/micro_batches_per_epoch": micro_per_epoch,
                "run/total_optimizer_steps": total_steps,
                "run/warmup_steps": warmup_steps,
                "run/n_train": n_train,
                "run/n_eval": len(eval_idx),
                "run/effective_batch": bs * max(1, world_size) * _accum_steps,
                # DDP 下 `parameters()` 带 `module.` 前缀但**数量不变**，
                #   要真参数名得先 unwrap；这里只要个数，所以直接数即可。
                "run/params_actual": int(sum(p.numel()
                                            for p in model.parameters())),
                # ---- 数据源形态：board 级实时算 / stdata 预算好 ----
                # 这两条决定「换个 --data 跑同一个 run」到底换掉了什么，而
                # config 面板只有一个 `--data` 路径字符串。
                #
                # **必须报数值码，不能报字符串**（2026-10-04 用户实测报出
                #   `Unsupported scalar string value: 'board_level'`）：swanlab 的
                #   metric 通道只收 bool/int/float，str 只有 `float()` 成功才收
                #   （`swanlab/sdk/internal/run/transforms/scalar/__init__.py`：
                #   `try: float(data) except ValueError: raise TypeError`）。
                #   人读的名字在 config 面板（那里字符串合法）与 stdout，
                #   曲线里存码 —— 两边对照见 run.txt 的键位表。
                "run/v7_source": (2 if hasattr(dataset, 'sample_spatial')
                                else (1 if _v7_on else 0)),
                "run/rows_total": int(len(dataset)),
                "run/n_games": int(np.unique(np.asarray(dataset.game_ids)).size)
                if getattr(dataset, 'game_ids', None) is not None else 0,
                # eval 覆盖度（2026-10-06 补）：「这次 eval 只算了 5 批还是 50 批」
                # 直接决定指标可信度，此前它只活在 config 面板与 stdout。
                "run/eval_max_batches": int(args.eval_max_batches),
            }
            # ---- V7 的 12 项权重（段位表）也进 run 级 ----
            # config 面板记不了它们（那是**代码常量**不是 CLI），而「这一项权重
            # 是 0」正是逐项曲线上那条平线的解释 —— 没有它，看到 `ownership`
            # 恒 0 的人无法区分「没学」与「不学」。
            if _v7_on:
                for _k, _w in v7_stage1_loss_weights().items():
                    _run_facts['run/loss_coef/%s' % _k] = float(_w)
                _run_facts['run/policy_soft_weight'] = float(
                    POLICY_SOFT_WEIGHT)
            # 人读的名字走 **stdout**（metric 通道只收数值，见上面 `run/v7_source`
            # 的注释）。config 面板在 init 时建，那时数据集还没加载 ⇒ 那里
            # 拿不到这个形态，只能靠这一行 + 曲线里的码对照。
            _src_name = ('packed' if hasattr(dataset, 'sample_spatial')
                        else ('board_level' if _v7_on else 'se12'))
            logger.info("[swanlab] run/v7_source=%d（%s）| 0=se12 1=board_level 2=packed",
                        _run_facts['run/v7_source'], _src_name)
            try:
                swanlab_logger.log(_run_facts, step=0)
            except Exception as e:  # noqa: BLE001
                logger.warning("[swanlab] run 级指标上报失败（不影响训练）: %s", e)

    # 预取器 pf 已在**设备初始化之前**构造（见上方「数据集 + 预取 worker」段：
    # fork 晚于 set_device 会让每个 worker 继承  上下文，4 卡实测每卡凭空
    # 多占 ~24 GiB ⇒ OOM）。此处刻意不再构造，避免顺序被无意改回去。
    assert (pf is not None) == (args.prefetch_workers > 1), \
        '预取器构造与 workers 设置不一致：构造顺序被改动了？'

    # 定期快照的后台写盘器（见 `_AsyncSnapshotWriter`）。建在这里、训练循环之前，
    # 且**只由主进程建** —— 非主进程从不做定期保存（下面的分支带 `is_main`）。
    _snap_writer = _AsyncSnapshotWriter(logger) if is_main else None

    for epoch in range(start_epoch, args.epochs):
        # DDP：每卡取本 rank 的不相交分片；set_epoch 让每 epoch 重新洗牌
        if is_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)
            perm = [int(train_idx[j]) for j in train_sampler]
        else:
            rng.shuffle(train_idx)
            perm = train_idx
        model.train()
        n_batches = (len(perm) + bs - 1) // bs
        # 预取器：每 epoch 先灌满流水线（提前 depth 个 batch 造好数据）
        if pf is not None:
            for _j in range(min(args.prefetch_depth, n_batches)):
                pf.submit(perm[_j * bs:(_j + 1) * bs])
        # 内核级剖析（诊断用）：GOAI_PROFILE=<step> 从该 step 起 profiling
        # GOAI_PROFILE_STEPS 个 step（默认 50），结束打印 top kernel 耗时表，
        # 用于定位 740ms/step 的去向。
        _prof_at = int(os.environ.get('GOAI_PROFILE', '0') or 0)
        # 窗口长度，步数。`0` = 起点即终点：起点之后**第一个** `_do_stdout` 步就停表
        # ⇒ 配 `--log-every 5` + `GOAI_PROFILE=5 GOAI_PROFILE_STEPS=0` 即「第 5 步出表」，
        # 而不是等满 50 步（2026-10-07 真机一次诊断要等 ~11 分钟才拿到表）。
        # 钳到 >=0：负数没有意义，且会让 `step >= _prof_at + _prof_span` 恒真 ⇒ 起点步
        # 之前就停表、一个内核都没采到。
        _prof_span = max(0, int(os.environ.get('GOAI_PROFILE_STEPS', '50') or 50))
        _prof_ctx = None
        # ---- 分段计时（纯 CPU 侧观测）----
        # 动机：4 卡 旧多卡环境 实测 4.25 s/step，扣除 eval（实测仅 0.2%）后
        # 约 96% 是黑盒，无法判断瓶颈在取数 / 算子 / 通信 / 保存。
        # 绝不在此插 synchronize()：那会打断预取与双缓冲流水，反而更慢。
        # 代价是 t_comp 只反映 CPU 侧发射时间、不含 加速器 实际执行；
        # 判读靠「各段之和 vs elapsed」的差额。
        # 重置放在打点处，故某步的 save/eval（发生在其打点之后）计入
        # 下一个区间 —— 这与墙钟口径一致。
        _t_data = _t_comp = _t_save = _t_eval = 0.0
        _t_data_max = 0.0
        _n_timed = 0
        # ---- backward() 之后的三个区间（g/o/m）-----------------------------
        # 起因：真机实测墙钟 12.0 s/step，而 `c` 只有 2.29、`d` 0.03 ⇒ 约 81% 的
        # 步时落在 `_t_comp` 结算**之后**，旧计时刻画里完全没有这一段。拿 c vs d
        # 判断瓶颈会把那 81% 当成不存在。
        #
        # 三段按「每个 optimizer step」计，故单独用 `_n_opt` 归一（accum>1 时它们
        # 每 optimizer step 才发生一次，除以 `_n_timed` 会缩小 accum 倍）：
        #   g = unscale_ + 溢出探测 + clip_grad_norm_（含全局范数归约的同步）
        #   o = optimizer.step + zero_grad
        #   m = EMA update（foreach_mul_/foreach_add_）
        # `g` 再劈三段（实测 `g` 独占 79.9%，必须分清是 clip 有病还是队列在此 drain）：
        #     g1 = unscale_ + 溢出探测（BF16 下应 ≈0）
        #     g2 = clip_grad_norm_ 调用本身（返回 device 张量 ⇒ 纯发射）
        #     g3 = float(_gn) 的设备→主机同步 = 队列 drain（真正干活的地方）
        # 判读：g2 ≫ g3 ⇒ clip 在同步，改 foreach；g3 ≫ g2 ⇒ clip 无辜，瓶颈在
        #       graph-compile / channels-last A/B。
        _t_clip = _t_opt = _t_ema = 0.0
        _t_g1 = _t_g2 = _t_g3 = 0.0
        _n_opt = 0
        _n_skipped = 0
        # `_n_attempted` 与 `_n_skipped` **在同一处**自增（scaler.step 那一行），
        #   所以「跳过占比」的分母恒 ≥ 分子。此前分母用的是 `step`，而它在 47 行
        #   之后才自增 ⇒ 连续溢出时会算出 133.33% 这种 > 100% 的荒谬比例。
        _n_attempted = 0
        # 因跳步而**没有**推进 LR 计划的次数（见下面 scheduler.step 的门控）。
        #   单独计数而不是复用 `_n_skipped`：后者是「GradScaler 跳了」，
        #   两者当前恒等，但语义不同（AMP 关掉时前者会大于后者）。
        _n_lr_frozen = 0
        # ---- SwanLab 上报用的「跨 micro-batch 沿用」量 ----
        # grad_norm 每个 optimizer.step() 只有一个值（clip_grad_norm_ 的返回值，
        # 在 unscale_ 之后取才是真值），而上报是每 micro-batch 打一次点 ⇒ 必须
        # carry forward，否则 accum>1 时曲线出现阶梯。
        # nan 表示「还没做过任何 optimizer step」（第 0 个打点），不是 0。
        _grad_norm_last = float('nan')
        # 训练健康度（train_top1 / value_rmse / 基线）在**同一个 batch** 上算才有
        # 对照意义，所以每次打点都重算（不像 grad_norm 需要 carry forward）。
        # 这个 `None` 只为「名字在任何打点之前就已绑定」而存在：上报与健康度
        # 计算在**同一个** `if _do_swanlab:` 分支里，所以它一定先被赋值再被
        # `**_health_last` 用掉（`test_swanlab_metrics.py::
        # test_run_level_metrics_dict_uses_only_names_defined_earlier` 钉这条）。
        _health_last = None
        for i in range(n_batches):
            try:
                if i % _accum_steps == 0:
                    optimizer.zero_grad(set_to_none=_zero_set_none)
                # DDP 梯度累积加速：非末步跳过 all-reduce（与 no_sync() 语义等价）。
                # 数学上 (accum-1) 次本地累加 + 末步一次 all_reduce(SUM) 与「每步都
                # all_reduce(SUM) 再累加」完全相等，仅省 (accum-1) 次跨卡通信。
                # 置于 backward 的 with 块之前、测试 AST 切片（tail）之外：test_log_loss_identity
                # exec 的是「含 compute_l2_report 的 with 块」+「其后到 backward 的切片」，
                # 本段不在其中，故门禁不受影响。单卡/非 DDP 时 model 无该属性，由
                # hasattr 跳过（保持原每步同步行为）。
                _is_last = ((i + 1) % _accum_steps == 0) or ((i + 1) == n_batches)
                if is_dist and hasattr(model, 'require_backward_grad_sync'):
                    model.require_backward_grad_sync = _is_last
                if _prof_at > 0 and step == _prof_at and _prof_ctx is None:
                    try:
                        # 活动集固定 CPU + CUDA。

                        from torch.profiler import (profile, ProfilerActivity)
                        _acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        _prof_ctx = profile(activities=_acts)
                        _prof_ctx.__enter__()
                        logger.info("[profile] 已开始内核剖析（%d steps）| activities=%s",
                                    _prof_span, [str(a) for a in _acts])
                    except Exception as pe:  # noqa: BLE001
                        logger.warning("[profile] 不可用: %s", pe)
                        _prof_at = 0
                _t_data0 = time.perf_counter()
                if _v7_on:
                    # ---- B8 · V7 取批：22 通道 + 19 维（取代 states/values）-------
                    # 与下面 12 通道那条路的**唯一**结构差别是 payload 的第 1/2 项；
                    # P0 的「先提交下一个 batch 再取当前 batch」节奏照旧（预取器的
                    # 背压与重叠不因换路径而变）。
                    sel = perm[i * bs:(i + 1) * bs]
                    if pf is not None:
                        nxt = i + args.prefetch_depth
                        if nxt < n_batches:
                            pf.submit(perm[nxt * bs:(nxt + 1) * bs])
                        _sp_np, _gl_np, _mv_np, lbl = pf.next(device=device)
                    else:
                        _sp_np, _gl_np, _mv_np, lbl = v7_batch_sync(
                            dataset, sel, device)
                    _in32 = (amp_dtype == torch.float32)
                    state = v7_to_device(_sp_np, device, amp_dtype,
                                        pin=(_backend == 'cuda'))
                    # NHWC（2026-10-06）：与模型权重的 channels_last 对齐，否则
                    # 卷积拿到 channels_last 权重 + NCHW 输入，每次卷积都白搬一次。
                    if use_channels_last:
                        state = _to_nhwc(state)
                    gl = v7_to_device(_gl_np, device, amp_dtype,
                                    pin=(_backend == 'cuda'))
                    # 与 state/gl 同口径：moves 也是 pageable numpy，CUDA 上必须
                    # 先 pin_memory 再非阻塞 H2D，否则和标签那处一样会
                    # `misaligned address`（见 _labels_dict_to_tensors 处注释）。
                    _mv_t = torch.from_numpy(
                        np.ascontiguousarray(_mv_np)).long()
                    if _backend == 'cuda':
                        move_t = _mv_t.pin_memory().to(device, non_blocking=True)
                    else:
                        move_t = _mv_t.to(device)
                elif pf is not None:
                    # P0: 先提交下一个 batch，再取当前 batch（给 worker 更多预计算时间）
                    nxt = i + args.prefetch_depth
                    if nxt < n_batches:
                        pf.submit(perm[nxt * bs:(nxt + 1) * bs])
                    # labels=False（默认）时 next() 仍返回 numpy 三元组，与软标签
                    # 接入前**逐位一致**；labels=True 时多一个同 device 的
                    # `labels_dict`（含嵌套 w 的张量）。
                    if _soft_on:
                        states_np, moves_np, values_np, lbl = pf.next(device=device)
                    else:
                        states_np, moves_np, values_np = pf.next()
                        lbl = None
                    if _backend == 'cuda':
                        # pin_memory 需要 contiguous 且为 CPU 内存
                        moves_np = np.ascontiguousarray(moves_np)
                        values_np = np.ascontiguousarray(values_np)
                        # 只有全精度（autocast 关着）才升 fp32。AMP 下权重会被
                        #   autocast 转成 fp16/bf16，输入保持 planes 的原 dtype
                        #   （fp16）即可 —— 强升 fp32 会让设备侧输入张量与 H2D
                        #   字节数都白带一倍（2026-10-01）。
                        _in32 = (amp_dtype == torch.float32)
                        if use_channels_last:
                            # A3: NHWC 物理布局——numpy 端一次性转置（比 torch
                            # .to(memory_format) 的主线程重排 memcpy 快），permute
                            # 得 channels_last 视图；pin 后异步 H2D，免每 step 重排
                            _t = torch.from_numpy(np.ascontiguousarray(
                                states_np.transpose(0, 2, 3, 1)))
                            state = (_t.float() if _in32 else _t).pin_memory().permute(
                                0, 3, 1, 2)
                        else:
                            states_np = np.ascontiguousarray(states_np)
                            _t = torch.from_numpy(states_np)
                            state = (_t.float() if _in32 else _t).pin_memory()
                        move_t = torch.from_numpy(moves_np).long().pin_memory()
                        value_t = torch.from_numpy(values_np).float().pin_memory()
                    else:
                        # 同上：全精度才升 fp32。加速器 走 AMP（amp_dtype=fp16）⇒ 保持
                        # planes 的 fp16；CPU 是 fp32 ⇒ 这里升回去，否则 fp16 输入
                        # 喂 fp32 权重会报 dtype 不匹配。
                        state = torch.from_numpy(states_np.copy())
                        if amp_dtype == torch.float32:
                            state = state.float()
                        move_t = torch.from_numpy(moves_np.copy())
                        value_t = torch.from_numpy(values_np.copy())
                    state = state.to(device, non_blocking=True)
                    move_t = move_t.to(device, non_blocking=True)
                    value_t = value_t.to(device, non_blocking=True)
                else:
                    sel = perm[i * bs:(i + 1) * bs]
                    if _soft_on:
                        state, move_t, value_t, lbl = dataset.sample_batch(
                            sel, device, labels=True)
                    else:
                        state, move_t, value_t = dataset.sample_batch(sel, device)
                        lbl = None
                    # A100 上转 NHWC 以匹配模型 channels_last 布局，卷积更快
                    if use_channels_last:
                        state = _to_nhwc(state)
                # ---- A4：软批节奏 + 软项 kwarg -----------------------------
                # kind 走 `args.policy_loss` 这个**既有**通道：`compute_policy_loss`
                # 的第 3 个位置实参被 tests/test_huber_loss.py::
                # test_flags_reach_loss_calls 钉成 `args.policy_loss`，所以
                # `--soft-every` 的切换改的是 args 上的取值，不是调用点的写法。
                # `_hard_policy_loss` 是用户给的硬目标口径（软标签没接管时的值）。
                if _soft_on:
                    args.policy_loss = _soft_kind_for_step(
                        _hard_policy_loss, step, _soft_every)
                # 未开软标签时是**空 dict** ⇒ 损失调用的实参集合与 A3 之前逐字
                # 相同（段 1 的热路径不多搬一个字节、也不多算一个张量）。
                soft_kwargs = ({'soft': lbl['soft'], 'soft_mask': lbl['soft_mask'],
                                'soft_weight': args.soft_weight}
                            if lbl is not None else {})
                _dt = time.perf_counter() - _t_data0
                _t_data += _dt
                if _dt > _t_data_max:
                    _t_data_max = _dt
                _t_comp0 = time.perf_counter()
                if _v7_on:
                    # ---- B8 · V7：12 项 loss 一次算完 -----------------------------
                    # 与 12 通道的三处差异，全部是**结构**差异而不是口径差异：
                    #  1. forward 签名是 `model(spatial, global_features)`，输出是
                    #     dict 而不是 `(policy_logits, value_logit)` 二元组；
                    #  2. 损失由 `KataGoV7Loss` 装配（自带有效系数与行权重），
                    #     **不经** `compute_policy_loss` / `compute_value_loss`
                    #     ⇒ `--policy-loss` / `--value-loss` / `--huber-beta` /
                    #     `--label-smoothing` / `--value-loss-weight` 在这条路上
                    #     一律无效（启动期已对 value-loss-weight 告警，其余是
                    #     归档旗，启动时已把 args 拨回中性值）；
                    #  3. 下面那三个变量沿用同名，好让下游那一行
                    #     `_read_log_scalars(log_loss, policy_loss, value_loss)`
                    #     与 12 通道路径共用。
                    with maybe_autocast(device, amp_dtype):
                        out = model(state, gl)
                        _v7_res = _v7_lossf(out, v7_loss_labels(
                            lbl, move_t, action_size=_v7_action_size))
                    # **不要**用 `sum(_v7_res['terms'])`：那是**未乘系数**的逐项
                    # 值。`weighted` 才是真正被优化的量（系数乘两次是
                    # `test_katago_v7_loss.py::test_coefficients_are_applied
                    # _exactly_once` 钉死的错误）。
                    _w = _v7_res['weighted']
                    opt_loss = _v7_res['loss']
                    policy_loss = _w['policy'] + _w['policy_opp']
                    # value 侧装 value + futurepos：futurepos 不是「价值」而是
                    # 「对手占位预测」，但下游日志只有 p/v 两栏，而
                    # `log_loss − policy_loss − value_loss` 这个恒等式在下游被
                    # 用到（`test_log_loss_identity`）。因为段 1 那 8 个 score 项
                    # 加权后恒 0，`sum(weighted)` 恰好 = policy_loss + value_loss
                    # ⇒ 恒等式在 V7 上**仍然精确成立**。
                    value_loss = _w['value'] + _w['futurepos']
                    l2_report = compute_l2_report(optimizer.param_groups)
                    log_loss = opt_loss + l2_report
                    # **不要**在这里 `float(x)`：那是每项一次 D2H 同步 —— 13 项
                    #   × 每个 micro-batch，全部是为打点服务的纯浪费（两个消费点
                    #   `_v7_terms_swanlab` / `[step N v7]` 都只在 `--log-every`
                    #   节奏读值）。这里只 `.detach()` 剥图（张量带图，直接留着
                    #   会拖住整段 autograd），**把 D2H 推迟到消费处**：每个打点
                    #   间隔 13 次同步，而不是每步 13 次。值本身与 detach 无关
                    #   （同一份数据）；13 个标量张量常驻可忽略。
                    _v7_terms_last = {k: x.detach() for k, x in _w.items()}
                    # 哪一项算坏了，**当场点名**（2026-10-04 云端 旧多卡环境 实跑）。
                    #   那个 run 的症状是「每步都溢出、loss 全 NaN、缩放值降到 160
                    #   仍 100% 跳过」，本地 fp32/fp16/bf16 都复现不出来 ⇒ 只有
                    #   这条日志能指认是哪一项在 加速器 上坏掉。
                    _bad_terms = _v7_res.get('nonfinite_terms') or []
                    _bad_ops = _v7_res.get('nonfinite_operands') or []
                    _san_rows = _v7_res.get('sanitized_rows') or {}
                    # **只报非空的那一半**：`nonfinite_terms` 与 `sanitized_rows`
                    #   是两件不同的事 —— 前者是「这一项最终仍然非有限」，后者是
                    #   「这一项内部的坏行被净化掉了（所以它现在有限了）」。
                    #   净化分两种，键名不同（见 `_weighted_mean` 的 probe）：
                    #     `项名`        = w==0 的行上 p 非有限
                    #     `项名:w_bad`  = **权重本身是坏的**（inf 或 >exp(10)）
                    #                        ← 真机 NaN 的源头
                    #   段 1 有 9 项系数为 0，后两种同样要紧：那一项其实一直在吐
                    #   inf，只是因为不参与优化而没人发现。
                    if _bad_terms or _san_rows:
                        _err_cnt += 1
                        # **必须区分 total 是否真的非有限**：零系数项会被
                        #   `nan_to_num` 净化成 0，所以「某项坏」时 total 往往
                        #   **仍然是有限的** —— 那种情况下这条日志若写成
                        #   「加权 loss 非有限」就是在**报一件没发生的事**，
                        #   而看日志的人会以为 loss 已经废了。
                        _tot_ok = _v7_res.get('total_finite')
                        _head = ('加权总 loss 仍非有限' if _tot_ok is False
                                else '加权总 loss 有限（该项系数为 0，已净化）')
                        msg = ('[v7] 第 %d 次：%s，逐项点名 = %s%s%s'
                            % (_err_cnt, _head, sorted(_bad_terms) or '（无）',
                                ('｜坏在操作数 %s' % sorted(_bad_ops))
                                if _bad_ops else '',
                                ('｜被净化的坏行 %s' % sorted(_san_rows.items()))
                                if _san_rows else ''))
                        # 每种组合只报一次，后续只计次，否则每步刷屏把真信息淹掉
                        _key = (tuple(sorted(_bad_terms)), tuple(sorted(_bad_ops)),
                                tuple(sorted(_san_rows.items())))
                        if _key not in _v7_bad_seen:
                            _v7_bad_seen.append(_key)
                            logger.error(msg + '（首次出现该组合）')
                        else:
                            logger.debug(msg)
                        # **被优化的量本身非有限 ⇒ 硬失败，不要继续跑。**
                        #
                        # `_tot_ok is False` 就是"加权总 loss 非有限"，而那正是
                        # 反向要传播的标量。它非有限有两条完全不同的来路，必须分开：
                        #
                        #  · **前向就出了 NaN**（权重/激活越界）。这是**不可恢复**的：
                        #    `GradScaler` 只管梯度，看不到前向；而 NaN 一旦进到权重，
                        #    后续每步都是 NaN，`scaler.step` 会一直跳步 —— 于是
                        #    「训练死了但还在跑」，日志上除 `loss=nan` 之外一切正常
                        #    （速度、显存、耗时都稳），能白烧几小时。
                        #  · **纯梯度溢出**。这个 GradScaler 管得住：跳步 + 减半，
                        #    权重不动，下一步往往就干净了。
                        #
                        # 实测（2026-10-05 4×旧多卡环境，lr 9.77e-3 / 4000 每卡）：
                        # step 400 还好（lr 7.07e-3、scale 81920、skip=0），
                        # step 430~440 loss=nan、scale 峰值 163840，
                        # step 450 scale=10、skip=13 —— 减半 14 次**一次也没救回来**，
                        # 因为病根在前向。梯度侧当时还有巨大余量（163840 都没溢出）。
                        if _tot_ok is False:
                            raise RuntimeError(
                                '[v7] 加权总 loss 非有限（第 %d 次）⇒ 训练已不可恢复，'
                                '中止以免白烧机时。\n'
                                '  逐项点名 = %s\n'
                                '  坏在操作数 = %s\n'
                                '  被净化的坏行 = %s\n'
                                '  此刻 scale = %s\n'
                                ' **这几乎不是梯度溢出**（GradScaler 管得住那个，'
                                '跳步减半即可）；这里是**前向**出了 NaN，'
                                '而 GradScaler 只缩放梯度、管不到前向。\n'
                                ' 判据：step 400 时 lr 7.07e-3、scale 81920、skip=0 '
                                '仍健康，warmup 结束把 lr 顶上去后 10~20 步内炸 ⇒ '
                                '峰值 LR 越过了 fp16 激活上限（65504）。\n'
                                ' 处置：把 --lr 降到峰值的一半以下（实测余量只有 4%%：'
                                '7.07e-3 健康、7.37e-3 即炸），必要时把 warmup 加长到'
                                '总步数的 15%%~20%%。'
                                % (_err_cnt, sorted(_bad_terms) or '（无）',
                                sorted(_bad_ops) or '（无）',
                                sorted(_san_rows.items()) or '（无）',
                                getattr(scaler, 'get_scale', lambda: 'n/a')()))
                    # 逐项上报用**独立**的 dict，而不是就地复用 `_v7_terms_last`：
                    #   那个 dict 同时喂 stdout 的 `[step N v7]` 行（键名是裸 term
                    #   名），若直接把带 `loss_v7/` 前缀的键塞进去，stdout 那行会
                    #   变成 `loss_v7/policy=5.88`，而测试与文档都按裸名读它。
                    #   两套键名不能共用一个 dict。
                    #   值沿 `_v7_terms_last` 的**张量形态**原样传（D2H 同样推迟
                    #   到 swanlab log 处），只有权重为 0 的项是字面量 0.0。
                    _v7_terms_swanlab = {
                        'loss_v7/%s' % k: v for k, v in _v7_terms_last.items()}
                    # 权重为 0 的项也**照报**（值恒 0）：图上看得见「这一项存在但
                    # 没在学」，比「曲线里根本没有这一项」更容易区分「没接上」
                    # 与「接上了但权重是 0」。键从**权重函数**取而不是另写一份
                    # 名单 —— 段位表改了而这份名单没改，图上就会出现「多一项/
                    # 少一项」而没人知道是哪边错了。
                    for _k, _w in v7_stage1_loss_weights().items():
                        if _w == 0.0:
                            _v7_terms_swanlab.setdefault('loss_v7/%s' % _k, 0.0)
                    # 这一行与 `else` 分支末尾那行**逐字重复**，是刻意的：
                    # `tests/test_huber_loss.py::test_log_loss_identity` 从 main()
                    # 的 AST 里取「含 `compute_l2_report` 的 `with` 块」与「同一个
                    # 语句列表里紧随其后的 `.backward()`」这一对，然后 **exec**
                    # 它们来验恒等式 —— 它假定两者是**兄弟语句**。一旦 backward
                    # 被提到 `if/else` 之外，它就会去 `orelse` 列表里找、找不到而
                    # 报错。共用的写法（提到 if 外面）会让这条门禁**误报**，所以
                    # 这里各写一份，并让两边都继续被那条测试覆盖。
                    scaler.scale(opt_loss / _accum_steps).backward()
                else:
                    with maybe_autocast(device, amp_dtype):
                        policy_logits, value_logit = model(state)
                        # D4：policy/value 损失由 --policy-loss / --value-loss 分派
                        # （默认 ce / huber），--huber-beta 同时供两处使用。
                        # C8：value 的 BCE 分支已删 —— 无 winrates 时也直接对
                        # value_t ∈ [-1,1] 回归；不再按数据字段分叉目标变换。
                        # P4.5-fix：--value-loss-weight 默认 1.0（BCE 时代的 5.0 倍
                        # 补偿已按用户裁决删除）。P4.5b：--policy-loss 默认由 huber
                        # 改回 ce —— Huber 打在概率域时梯度带 softmax 雅可比的
                        # p≈1/A，policy 梯度天生弱 ~A 倍且**随盘口变**；CE 打在
                        # log-prob 上，d(CE)/d(logit)=p−y 每坐标有界、与 A 无关。
                        # 见 compute_policy_loss docstring 末节与 report
                        # `## Fix2（b）增补`。
                        policy_loss = compute_policy_loss(
                            policy_logits, move_t, args.policy_loss,
                            label_smoothing=args.label_smoothing,
                            huber_beta=args.huber_beta, **soft_kwargs)
                        value_loss = compute_value_loss(
                            value_logit, value_t, args.value_loss,
                            huber_beta=args.huber_beta)
                        # ---- 被 backward 的量 vs 写进日志 `loss` 的量 -------------
                        # opt_loss **不含** L2 项：正则走的是 AdamW 的**解耦** weight
                        # decay（θ ← θ − lr·wd·θ，发生在参数更新里），按构造就不在
                        # 梯度里。把 c‖θ‖² 加进来会让目标从「解耦衰减」变成
                        # 「耦合 L2 + 再次解耦衰减」的双重正则，训练行为立刻变化且
                        # 不报错 —— 故 opt_loss / log_loss 分开命名。
                        # log_loss 与 opt_loss **不相等**（用户裁决的总损失口径是
                        # L = L_policy + L_value + c‖θ‖²，要让恒等式在日志上字面成立）：
                        # l2_report 从 param_groups 读回真实 weight_decay、只覆盖
                        # wd != 0 的组，在本次 step() **之前**算 ⇒ 与两个损失项取自
                        # 同一个 θ，恒等式在一次 fp32 加法的精度内成立。
                        # 但真正的精度上限是 stdout 的 `%.4f`（量化步长 1e-4，对
                        # 初值 0.588 是 0.017%），不是 fp32 舍入。且
                        # `log_loss − policy − value` 只在 `--value-loss-weight == 1`
                        # 时等于 l2_report；w≠1 时它是 w·value。
                        # 读 loss 曲线的人必须知道：log_loss **不是**被优化的目标。
                        l2_report = compute_l2_report(optimizer.param_groups)
                        opt_loss, log_loss = compose_losses(
                            policy_loss, value_loss, args.value_loss_weight,
                            l2_report)
                    # 与 V7 分支里那一行逐字相同（理由见那边的注释：这条 backward
                    # 必须与上面的 `with` 同属一个语句列表）。
                    scaler.scale(opt_loss / _accum_steps).backward()

                _t_comp += time.perf_counter() - _t_comp0
                _n_timed += 1
                if (i + 1) % _accum_steps == 0 or (i + 1) == n_batches:
                    # 缩放值下降 == 本步因 inf/nan 被 GradScaler 跳过，那一整个
                    # batch 的数据就此白扔。实测 4 卡 旧多卡环境 在 step~1770 出现
                    # 16384→8192→4096→2048 的雪崩 + 大量 Skipping step。
                    #
                    # 注意：这里不把 clip_grad_norm_ 挪到 unscale_ 之前。模型
                    # 参数始终是 FP32（只改了 memory_format，从未 .half()），
                    # 梯度也是 FP32，其上限 3.4e38，缩放系数根本不可能让它
                    # 在 65504 处溢出。那些 inf/nan 是前向/反向里真实的数值
                    # 故障（最可疑是 加速器 强制 math 注意力物化大 logits 时的
                    # FP16 溢出），不是 loss scaling 的伪影——所以真正的
                    # 修复点在别处，此处只负责让它**可观测**。
                    _scale_now = scaler.get_scale()
                    # 与 `_n_skipped` 同一处自增 ⇒ 占比口径自洽（见上面注释）
                    _n_attempted += 1
                    # `g` 段起点：unscale_ → float(_gn) 结束（含溢出探测，
                    # 它是一次全参数梯度扫描，可能含设备同步）。
                    _t_clip0 = time.perf_counter()
                    _t_g1_0 = _t_clip0
                    scaler.unscale_(optimizer)
                    # **溢出诊断必须排在 `clip_grad_norm_` 之前**
                    #   `clip_grad_norm_(max_norm=1.0)` 在 `total_norm = inf` 时算出
                    #   `clip_coef = 0` 并 `grad.mul_(0)` ⇒ **inf × 0 = NaN**，而
                    #   `inf ⇒ clip_coef = 0 < 1` 这个分支**一定会进**
                    #   ⇒ clip 之后统计「inf 个数」**结构上恒为 0**。
                    #   此前诊断排在 clip 之后，真机 4 卡日志里每一轮打出来的都是
                    #   「参数上没有 inf/nan —— 溢出可能发生在已被释放的中间张量里」，
                    #   而那是**工具的盲区**，不是关于计算的结论：它把人引向了
                    #   「中间张量已释放」这个错误方向。
                    #   判据改成**直接**问「此刻梯度里真的有 inf/nan 吗」，
                    #   不再靠 clip 返回的总范数间接推断。
                    _had_nonfinite = (use_scaler
                                    and _grads_nonfinite_local(optimizer))
                    if _had_nonfinite and is_main:
                        # 把参数名一并给去 —— 组名只是按 LR 比值**猜**的
                        #（真机日志里三组全被打成 "backbone/policy" 就是
                        # 这个原因），模块归属才是能直接定位的信息。
                        _locate_overflow(
                            optimizer, logger,
                            phase='clip 前（此刻梯度里真的有 inf/nan）',
                            named_params=dict(model.named_parameters()))
                    # g1 结算（unscale_ + 溢出探测）/ g2 起点。
                    _t_g1 += time.perf_counter() - _t_g1_0
                    _t_g2_0 = time.perf_counter()
                    # clip_grad_norm_ **返回 clip 前的总范数** —— 之前被丢弃了。它是
                    # fp16 溢出/梯度爆炸唯一的直接信号：这轮 旧多卡环境 的 inf/nan 与
                    # 缩放值雪崩，本可以由它提前几分钟看到。
                    # 必须在 unscale_ 之后取（unscale 前是按 scale 放大的假值）。
                    _gn = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    # g2 结算 / g3 起点。
                    _t_g2 += time.perf_counter() - _t_g2_0
                    _t_g3_0 = time.perf_counter()
                    _grad_norm_last = float(_gn)
                    # g3 结算。`float(_gn)` 本身要等一次设备→主机同步，而它是
                    # backward 之后**第一次**同步 ⇒ 整个待执行队列（前向/反向的
                    # 实际执行、DDP all-reduce、clip kernel）都在这里 drain。
                    # 真机 2026-10-07：这 10.45 s 占 79.9%，是本段存在的理由。
                    _t_g3 += time.perf_counter() - _t_g3_0
                    # `g` 段结算 = g1+g2+g3（三段首尾相接无缝，故恒等）。放在
                    # `float(_gn)` 之后，不把这次同步算进 `o` 段。
                    _t_clip += time.perf_counter() - _t_clip0
                    # **溢出诊断的时机（2026-10-04 云端 旧多卡环境 实测打出来的 bug）**
                    #   `clip_grad_norm_(max_norm=1.0)` 在 `total_norm = inf` 时算出
                    #   `clip_coef = 0` 并 `grad.mul_(0)` ⇒ **inf × 0 = NaN**，
                    #   而 `inf ⇒ clip_coef = 0 < 1` 这个分支**一定会进**。
                    #   ⇒ clip 之后统计「inf 个数」**结构上恒为 0**，看到的 nan
                    #   **全是 clipper 造的**。
                    #   此前诊断就排在 clip 之后，于是真机那行
                    #   `有 0 个 inf / 211 个 nan` 被读成「反向出了 NaN」，排查
                    #   方向被带偏到 loss 与前向 —— 而**前向与 loss 都有限**
                    #   （本机逐项验证过），真凶是**反向出了 inf**。
                    #
                    #   修法：用 clip **自己返回的** `_gn` 判断「clip 前有非有限」，
                    #   而不是再去数参数里的 inf（那时已经没有了）。
                    # 归属仍然准确：`inf × 0 = NaN` 是**原地**写在同一个张量上，
                    #     所以「哪些张量现在是 NaN」= 「哪些张量原来是 inf」。
                    # **跳/不跳按全局判定**（2026-10-05 真机 4 卡事故的修复）。
                    # 原来用 `scaler.step(optimizer)`，而它只查**本地** `found_inf`
                    #   —— 某个 rank 有 inf 时它跳过、其余 rank 照常更新
                    #   ⇒ 四份权重从此永久不同，之后每次 all_reduce 都在混合四个
                    #   不同模型的梯度 ⇒ **训练从此无效**。而且各 rank 的 scale 各自
                    #   独立减半、进一步漂移（报出来的 `skip` 占比也是 per-rank 的）。
                    # 实测当时：50 ���内 1024 -> 8、skip=7/50，而"四张卡同一步一起
                    #   溢出"的概率极低 ⇒ 分叉几乎立刻发生。
                    # 详见 `_grads_nonfinite_any_rank` / `_scaler_step_global`。
                    _t_opt0 = time.perf_counter()
                    _real_step = _scaler_step_global(
                        scaler, optimizer, scale_before=_scale_now,
                        use_scaler=use_scaler)
                    if not _real_step:
                        _n_skipped += 1
                        if is_main:
                            # 本地干净的 rank 不点名：它没有可点的东西。
                            # 全局非有限但本地干净，说明是别的 rank 炸的。
                            if not _had_nonfinite:
                                logger.warning(
                                    '[fp16] 本步的溢出**不在本 rank**（本地梯度全部'
                                    '有限）—— 已按全局判定一起跳步。定位请看第一个'
                                    '报「溢出按模块点名」的 rank。')
                            if _scale_now >= _OVERFLOW_WARN_SCALE > scaler.get_scale():
                                # 分母用 `_n_attempted`（本行上一次自增）而**不是
                                # `step`：skip 在此处计数，而 `step += 1` 在 47 行
                                # 之后 ⇒ 用 `step` 当分母会算出「占比 133.33%」
                                # 这种 > 100% 的数（4 次缩放 / 当时 step=3）。
                                # `_n_attempted` 与 `_n_skipped` 在同一处++
                                # ⇒ 分母恒 ≥ 分子，口径自洽。
                                logger.warning(
                                    "[fp16] 缩放值首次跌破 %d：%.0f -> %.0f。"
                                    "累计跳过 %d 步（占比 %.2f%%）。",
                                    int(_OVERFLOW_WARN_SCALE), _scale_now,
                                    scaler.get_scale(), _n_skipped,
                                    100.0 * _n_skipped
                                    / max(1, _n_attempted))
                    optimizer.zero_grad(set_to_none=_zero_set_none)
                    # `o` 段结算：optimizer.step + zero_grad。
                    _t_opt += time.perf_counter() - _t_opt0
                    _n_opt += 1
                    # EMA 与 scheduler 同理：**跳过的步权重一动没动**，
                    #   此时 `ema.update()` 会把 shadow 朝当前权重多拉一次
                    #   （step 计数也照样 +1）⇒ EMA 的时间常数被"跳步"稀释，
                    #   而 eval 又是在 EMA shadow 上评的（`eval_used_ema`）。
                    #   100% 跳步时 shadow 会一路收敛到**初始权重**。
                    if ema is not None and _real_step:
                        # `m` 段：EMA（278 个参数的 foreach_mul_/foreach_add_）
                        _t_ema0 = time.perf_counter()
                        ema.update()
                        _t_ema += time.perf_counter() - _t_ema0
            except Exception as oom_exc:
                # 只捕获 CUDA 的 OOM：其他异常照原样上抛（别把真 bug 吞成 OOM）
                if not isinstance(oom_exc, torch.cuda.OutOfMemoryError):
                    raise
                torch.cuda.empty_cache()
                # 保存当前进度，便于减小 batch 后用 --resume 续训（仅主进程写盘，避免多卡并发写同一文件）
                if is_main:
                    save_model(model, args.out + '.latest')
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'step': step,
                        'epoch': epoch,
                        'best_eval_acc': best_eval_acc,
                        'rng': torch.get_rng_state(),
                    }, args.out + '.latest.train_state')
                logger.error("=" * 60)
                logger.error("%s 显存不足 (OOM)！当前 --batch-size=%d 过大。",
                            device.upper(), bs)
                logger.error("window 注意力在 19x19 上把 batch 展开为 B*361，显存增长很快。")
                logger.error(
                    "建议减小 --batch-size。%s",
                    " **V7 不要指望 --attn-window**：它的 MHSA 没有窗口参数，"
                    "该旗标对 V7 完全无效（显存全在 361² 的完整注意力矩阵上）。"
                    "要更大的有效 batch 请用 --gradient-accumulation-steps。"
                    if _v7_on else
                    "12 通道可考虑同时调小 --attn-window。")
                logger.error("已保存进度至 %s.latest(.train_state)，可用 --resume 续训。",
                            args.out)
                logger.error("已清理显存并退出，请调整参数后重跑。")
                logger.error("=" * 60)
                sys.exit(1)
            if (i + 1) % _accum_steps == 0 or (i + 1) == n_batches:
                if _real_step:
                    scheduler.step()
                else:
                    # **跳过的步不许推进 LR 计划**（2026-10-04 云端 旧多卡环境 实跑）。
                    #   `scaler.step()` 在检出 inf 时**内部跳过** `optimizer.step()`，
                    #   但它对调用方是「成功返回」的 ⇒ 无条件 `scheduler.step()`
                    #   会让 warmup/cosine 在**权重一动没动**的步上照样前进。
                    #   实测那个 run 每步都被跳过 ⇒ 558 步的 warmup 被 0 次
                    #   学习消耗掉，等于训练一开始就拿到一个已经退火的 LR。
                    #   而且 PyTorch 会为此打
                    #   `Detected call of lr_scheduler.step() before optimizer.step()`
                    #   —— 那个警告**就是在说这件事**，之前被当成噪音忽略了。
                    _n_lr_frozen += 1
            step += 1

            # 打点：stdout 与 SwanLab 频率解耦，且共用同一次设备同步。
            # 旧实现里 log_every 同时控制两者，且把三个 loss 张量各取两遍
            # （stdout 一次、swanlab 一次），共 6 次同步、其中 3 次重复。
            _do_stdout, _do_swanlab = _should_log(
                step, args.log_every, args.swanlab_every,
                swanlab_logger is not None)
            if _do_stdout or _do_swanlab:
                # 这里传的是 log_loss（报告口径，含 c‖θ‖²），**不是**上面被
                # backward 的 opt_loss —— 日志的 `loss` 键按用户裁决报
                # L_policy + L_value + c‖θ‖²，与优化器实际最小化的量差一个
                # 纯报告项（见 compute_l2_report docstring 的 §3.1 说明）。
                _lv, _pv, _vv = _read_log_scalars(log_loss, policy_loss, value_loss)
                lr = optimizer.param_groups[0]['lr']
                # 内存只打 reserved —— **刻意不**在这里加 allocated / peak。
                # 那三个数（`memory_allocated` / `max_memory_allocated` /
                # `empty_cache`）在 2026-10-01 加过一次又撤掉，理由：
                #   · OOM 报错本身就是更好的显存报告：它在**压力最大那一刻**给出
                #     allocated + reserved + free，配合 `total` 就能反推 torch 之外
                #     占多少（`32.00 − 27.02 − 0.62 = 4.36 GiB`）。常打一个
                #     「平时的 reserved」信息更少。
                #   · `torch.cuda.max_memory_allocated` 全仓库只那一处、没在
                #     NPU 版 torch 2.1 上验证过；它在日志路径上抛异常就是**第 50 步
                #     崩**，正好毁掉最需要那个数的时刻。观测不该有能力杀死被观测的进程。
                if _backend == 'cuda':
                    mem = torch.cuda.memory_reserved(device) / 1e9
                else:
                    mem = 0.0
                _now = time.time()
                # 有效 batch 必须含梯度累积：原来写的是 bs×world_size，漏乘
                # accumulation ⇒ 用了 `--gradient-accumulation-steps 2` 时日志把
                # 吞吐**报成实际的一半**（2026-10-01 修）。
                _eff_bs = bs * max(1, world_size) * max(1, _accum_steps)
                speed = (step - _step_at_start) * _eff_bs / max(1e-6, _now - t0)
                # 瞬时速率：距上一次 stdout 打点。与分段耗时同口径，
                # 二者相乘即该区间理论样本数，可直接核对时间丢在哪。
                spd_inst = ((step - _last_stdout_step) * _eff_bs
                            / max(1e-6, _now - _last_stdout_t))
                _nd = max(1, _n_timed)
                # g/o/m 按 **optimizer step** 归一（`_no`），不是 micro-batch。
                _no = max(1, _n_opt)
                _dms = _t_data * 1000.0 / _nd
                _cms = _t_comp * 1000.0 / _nd
                _sms = _t_save * 1000.0 / _nd
                _ems = _t_eval * 1000.0 / _nd
                _gms = _t_clip * 1000.0 / _no
                _g1ms = _t_g1 * 1000.0 / _no
                _g2ms = _t_g2 * 1000.0 / _no
                _g3ms = _t_g3 * 1000.0 / _no
                _oms = _t_opt * 1000.0 / _no
                _mms = _t_ema * 1000.0 / _no
                # `u` = 未归因 = 墙钟 −(d+c+g+o+m+s+e)。它不是残差噪声，而是
                # 「**当前已知但尚未分段**」的部分（DDP all-reduce、Python 与
                # 调度开销、log-every 处的同步）。2026-10-07 真机 a_v7.4_npu2：
                # 墙钟 12.0 s/step、c=2.29、d=0.03 ⇒ 这 9.7 s 全落进 `u`。
                # 加 g/o/m 就是为了把它劈开。
                _wall_ms = ((_now - _last_stdout_t) * 1000.0
                             / max(1, step - _last_stdout_step))
                _ums = (_wall_ms - _dms - _cms - _gms - _oms - _mms
                        - _sms - _ems)
                _dmax = _t_data_max * 1000.0
                # 重置放在打点处：某步的 save/eval 发生在其打点之后，
                # 因此计入下一个区间，与墙钟口径一致
                _t_data = _t_comp = _t_save = _t_eval = 0.0
                _t_clip = _t_opt = _t_ema = 0.0
                _t_g1 = _t_g2 = _t_g3 = 0.0
                _t_data_max = 0.0
                _n_timed = 0
                _n_opt = 0
                _scale = scaler.get_scale()

            if _do_stdout:
                logger.info("[step %d/%d] loss=%.4f (p=%.4f v=%.4f) lr=%.2e "
                            "scale=%.0f mem=%.2fGB "
                            "spd=%.0f spd_inst=%.0f s/s "
                            "elapsed=%.0fs skip=%d | "
                            "d=%.0f c=%.0f g=%.0f "
                            "g1=%.0f g2=%.0f g3=%.0f "
                            "o=%.0f m=%.0f s=%.0f e=%.0f "
                            "u=%.0f dmax=%.0f ms",
                            step, total_steps, _lv, _pv, _vv,
                            lr, _scale, mem,
                            speed, spd_inst, _now - t0, _n_skipped,
                            _dms, _cms, _gms,
                            _g1ms, _g2ms, _g3ms,
                            _oms, _mms, _sms, _ems, _ums,
                            _dmax)
                _last_stdout_t = time.time()
                _last_stdout_step = step
                # ---- B8 · V7：逐项 loss 打到 stdout -----------------------------
                # 为什么必须逐项打：段 1 有 8 项的权重是 0，而 `_lv` 只是个总和 ——
                # 「四项在学」与「四项都塌了、只剩 L2 在动」在 `loss` 这一个数上
                # **长得一模一样**。逐项曲线是训练早期唯一能抓住「第 5 项权重写反了」
                # 或「futurepos 的 -1 哨兵被当真值拟合」的手段。
                if _v7_on and _v7_terms_last:
                    # 值是设备张量（见 `_v7_terms_last` 处注释），float() 就是
                    # 本行的 D2H —— 每个打点间隔 13 次，而非每步 13 次。
                    logger.info("[step %d v7] " + ' '.join(
                        '%s=%.4f' % (k, float(v))
                        for k, v in _v7_terms_last.items()),
                        step)

            if _do_swanlab:
                # ---- 训练健康度：与**基线**成对上报（2026-10-01）----
                # 为什么必须带基线：`policy_loss` / `value_loss` 两条曲线单看没有参照，
                # 「从 5.89 降到 5.5」和「真的学到了」在图上长得一样。基线给出
                # 「什么都没学到」的那条线在哪儿：
                #   · policy_ce_random = log(A)（19 路 A=362 ⇒ ≈5.89）= 均匀猜测的 CE
                #   · value_rmse_zero  = sqrt(mean(z²))（恒预测 0 的 RMSE，按本批算）
                # 代价：~5 个 kernel + 5 次 D2H（只在打点步付，`--swanlab-every 10`
                # 下摊薄到每步 +0.5 次）。所以放在**上报分支**里算，而不是每个
                # micro-batch 都算 —— 后者会白付 10 倍。
                #
                # **必须就地构造这个 dict 字面量，不许抽成模块级 helper**
                # （2026-10-02 B8 返工）：`tests/test_swanlab_metrics.py` 里三条
                # 门禁 —— `test_health_metric_is_reported`（6 个键必须在
                # main() 的源码里）、`test_health_metrics_only_computed_on_log_steps`
                # `test_reported_values_are_floats_not_tensors`（正则抓这个字面量
                # 的花括号体，要求 ≥6 处 `float(`）—— 全都按**字面量**在 main() 里
                # 定位这块。B8 曾把它抽成 `compute_training_health()`，三条门禁一起
                # 变红：抽取让 12 通道默认路径**唯一**的上报面门禁集体失明，而那 9 个
                # 失败之所以被漏掉，正是因为自测没跑这个文件。
                # 上面三条判据是**纯文本**判据，所以连注释里都不能写出那个字面量
                # （写出来 count 就变 3）—— 这是「就地构造」的代价，也是它必须留下
                # 的原因。与 `test_log_loss_identity` 要求 `.backward()` 与
                # `compute_l2_report` 是兄弟语句同理：**门禁的形状优先于 DRY**。
                with torch.no_grad():
                    # V7 的 `out['policy_logits']` 是 `(B, 2, A)`：第 0 路 =
                    # `policy_player`、第 1 路 = `policy_opp`（与 loss #1/#2 的取用
                    # 一致）⇒ 健康度只取第 0 路。12 通道那条路本来就是 `(B, A)`。
                    # 两条路共用这 4 个 policy 侧指标，所以只有**一份**实现。
                    _plg = (out['policy_logits'][:, 0] if _v7_on
                            else policy_logits).detach().float()
                    _lp = _plg.log_softmax(-1)
                    _topk = _plg.topk(min(5, _plg.shape[-1]), dim=-1).indices
                    _mtg = move_t.reshape(-1)
                    _health_last = {
                        'train_top1': float((_plg.argmax(-1) == _mtg).float().mean()),
                        'train_top5': float((_topk == _mtg[:, None]).any(-1).float().mean()),
                        'policy_ce_random': float(math.log(_plg.shape[-1])),
                        # `policy_entropy` 报的是**真熵** `H = −Σ p·log p`
                        #   （所以 `_lp.exp() * _lp` 前面那个负号不能省）。
                        #   取值 ∈ [0, log A]：均匀时 = log(362) ≈ 5.8926，
                        #   学到之后**下降**（趋近 0），曲线方向与指标名一致。
                        #
                        # **历史 run 的这条曲线符号翻转了**（2026-10-03 裁决）：
                        #   旧实现报的是 `Σ p·log p`，那不是熵，是 **−H** ⇒
                        #   取值 ∈ [−log A, 0]，均匀时 ≈ **−5.89**，学到后**升向 0**。
                        #   换算关系逐位成立：**新值 = −旧值**（实测同一批 logits
                        #   两式之和恒为 0.0，见 `tests/test_train_sft_v7.py::
                        #   test_policy_entropy_is_true_entropy_and_flips_sign`）。
                        #   判读旧曲线时注意符号：旧 run 的「−5.89 → 0」在图上
                        #   看着像**变乱**（entropy 升），实际是**变锐**（熵降）。
                        #   本仓**只有这一条**熵曲线 —— 故意不同时上报两个口径，
                        #   那会让 SwanLab 里出现两条含义重叠、符号相反的曲线。
                        'policy_entropy': float(-(_lp.exp() * _lp).sum(-1).mean()),
                        # V7 的 value 是 3 分类 CE，没有「RMSE」这个口径可报：
                        # 压成标量的任何做法都是**新发明**的口径（且与 12 通道的
                        # `value_rmse` 不可比）⇒ V7 改报 `value_acc3`，这两个键给
                        # `nan` 而不是编一个数，**不假装**两条曲线的 `value_*`
                        # 是同一个东西。
                        # `value_logit` / `value_t` 这两个名字在 main() 里**只有 12 通道那条路
                        # 才会绑定**（V7 走 `model(state, gl)` 返回 dict，不产出
                        # 二元组）⇒ 它们必须只出现在「`_v7_on` 为假」的那一支里：
                        # 条件表达式只求值**被选中的那一支**，所以
                        # `A if _v7_on else value_logit` 是安全的，而把它们写在
                        # 无条件位置上就是 UnboundLocalError（pyflakes/ruff 都不
                        # 会报：它是个合法的局部变量，只是那条路没赋值）。
                        'value_rmse': float('nan') if _v7_on else float(torch.sqrt(
                            (value_logit.detach().float().reshape(-1)
                            - value_t.reshape(-1).float()).pow(2).mean())),
                        'value_rmse_zero': float('nan') if _v7_on else float(
                            torch.sqrt(value_t.reshape(-1).float().pow(2).mean())),
                    }
                    if _v7_on:
                        _ol = out['outcome_logits'].detach().float()
                        _og = torch.as_tensor(lbl['outcome']).reshape(-1).to(torch.long)
                        _health_last['value_acc3'] = float(
                            (_ol.argmax(-1) == _og).float().mean())
                        # ---- 三类占比 + 多数类基线：`value_acc3` 的地板 ----------
                        # 为什么必须有：没有它 `value_acc3` **无法判读**。三分类里
                        # 「永远猜多数类」就能拿到 max(p) 的命中率，可能远高于 1/3
                        # 而毫无判别力 —— run.txt 把 1/3 写成基线是错的，真基线是
                        # 多数类占比。
                        # 实测（2026-10-09）575 步 `value_acc3` 钉在 0.50，而 value
                        # 的三分类 CE ≈ 0.83 已**低于** ln(3)=1.0986 ⇒ 模型至少学到
                        # 了类先验。那 0.50 与先验隐含的多数类占比同量级，正是
                        # 「先验学到了、判别力没有」的形状 —— 没有这条基线就看不出
                        # 它其实**不如**猜多数类。
                        # 同时这两者合起来还能区分两种病：acc ≈ 1/3 ⇒ 先验都没学到；
                        # acc ≈ `value_prior_top1` ⇒ 先验学到了但没有判别力。
                        # ⚠ 三个占比在**单个训练 batch** 上估计，噪声与 acc 同量级
                        # （二项，n≈batch）；要读趋势看平滑曲线，别读单点。
                        _n3 = max(1, int(_og.numel()))
                        for _ci in range(3):
                            _health_last['value_prior%d' % _ci] = (
                                float((_og == _ci).sum()) / _n3)
                        _health_last['value_prior_top1'] = max(
                            _health_last['value_prior%d' % _ci] for _ci in range(3))
                        # ---- policy **目标**的口径（2026-10-04）----
                        # 为什么必须报目标侧：`policy_loss` 与 `policy_entropy` 说的
                        # 都是**模型**（预测分布），而 A/B/C 三段的**目标**根本不
                        # 同 —— A 段 one-hot（熵 0）、B/C 段 KataGo 搜索分布（熵 > 0）。
                        # 同一个 `policy_loss = 5.88` 在 A 段是「什么都没学到」
                        # （= log 362），在 C 段却可能已经接近搜索分布；只看预测侧
                        # 曲线会把这两者读成同一件事。
                        #
                        # 只在**真有软行**时产出这两个键（不给 `nan` 兜底）：
                        #   A 段不挂 `--soft-index` 时 `soft_mask` 恒 0 ⇒ 目标就是
                        #   one-hot、熵恒 0 ⇒ 「标签有多锐」这个问题不存在。报一个
                        #   占位 `nan` 只会让图上多两条读不出来的线，而键集合是
                        #   按段变化的（12 通道路径完全不带这两条，见
                        #   `tests/test_train_sft_v7.py` 对 `_health_last` 键集合
                        #   的两处断言）。
                        _sm = torch.as_tensor(
                            lbl['soft_mask']).reshape(-1).float()
                        if float(_sm.sum()) > 0:
                            _msk = _sm > 0
                            # 只在真吃软标签的行上算熵：mask=0 的行是 one-hot
                            # （熵 0），混进来会让均值被行数权重压平 ⇒ 曲线量到的
                            # 是「batch 里有多少软行」而不是「标签有多锐」。
                            _p = (torch.as_tensor(lbl['soft'])
                                .reshape(int(_msk.numel()), -1)[_msk]
                                .float().clamp_min(1e-12))
                            _health_last['label_entropy'] = float(
                                -(_p * _p.log()).sum(-1).mean())
                            _health_last['soft_row_frac'] = float(
                                _msk.float().mean())
                try:
                    _sw = {
                        "loss": _lv,
                        "policy_loss": _pv,
                        "value_loss": _vv,
                        # ---- 损失的三项分解（此前 `loss` 不可分解）----
                        # log_loss = policy + value + c‖θ‖²（用户裁决的报告口径）；
                        # opt_loss 才是被 backward 的量（不含 L2 —— 正则走 AdamW 的
                        # 解耦衰减，按构造不在梯度里）。三条一起看才能判断「在降」
                        # 是模型在学还是正则项在缩。
                        "l2_report": float(l2_report),
                        "opt_loss": float(opt_loss),
                        "lr": lr,
                    "memory_gb": mem,
                    "speed": speed,
                    "speed_inst": spd_inst,
                    # 每卡吞吐与 ETA：长跑时「还剩多久」比「多快」更需要盯
                    "speed_per_card": speed / max(1, world_size),
                    "elapsed_min": (_now - t0) / 60.0,
                    "eta_min": ((total_steps - step)
                                * (_now - t0) / max(1, step - _step_at_start)
                                / 60.0) if step > _step_at_start else float('nan'),
                    # 分母用 `_n_attempted`（与 `_n_skipped` 同一处自增）而不是
                    #   `step` —— 后者在这行之后 47 行才自增，会算出 > 100% 的占比。
                    # （2026-10-06 面板瘦身：绝对数 `skipped_steps` 删除，只留占比
                    #   —— 两者同源，跳步率才是「溢出严重度」的正确读法。）
                    "skip_rate_pct": 100.0 * _n_skipped / max(1, _n_attempted),
                    # grad_norm 之前被丢弃（clip_grad_norm_ 的返回值）。它是
                    # fp16 溢出/梯度爆炸唯一的直接信号；accum>1 下每 optimizer
                    # step 只有一个值，打点是每 micro-batch ⇒ 沿用上一次的。
                    "grad_norm": _grad_norm_last,
                    # 缩放值：scaler 关掉时上报 1.0 而不是 _scale，让图上能一眼
                    # 分辨「这条曲线是真的 scaler 在动」还是「bf16 路径没 scaler」。
                    # （此前这个键在下面 :5356 又出现了一次 —— dict 字面量里后写的
                    #   覆盖先写的，于是上面那行是**死代码**：谁改它都不会生效。）
                    "scaler_scale": _scale if use_scaler else 1.0,
                    "t_data_ms": _dms,
                    # ⚠ 口径：`t_comp_ms` 只统计 **CPU 侧发射时间**，不含 加速器 实际
                    #   执行（刻意不打 synchronize，会打断预取流水）。它**不能**
                    #   单独读成「算子慢」——判读靠「各段之和 vs elapsed」的差额。
                    "t_comp_ms": _cms,
                    "t_save_ms": _sms,
                    "t_eval_ms": _ems,
                    # 取数长尾：均值能掩盖「偶尔等 3 秒」的预取抖动
                    "t_data_max_ms": _dmax,
                    # （2026-10-06 面板瘦身：`epoch`/`step_pct` 删除 —— step 本身
                    #   已上报，两者是它的派生量，曲线不可读性为零。）
                    **_health_last,
                        # ---- B8 · V7 的 12 项逐项 loss（2026-10-04）----
                        # 此前 `_v7_terms_last` **只进 stdout**（上面那行
                        # `[step N v7]`），SwanLab 里只有 `loss` 一个和 ⇒ 段 1
                        # 那 8 项权重为 0 的 score 项在图上完全不可见，而它们恰恰
                        # 是「权重写反 / 哨兵值被当真值拟合」的唯一早期信号。
                        # 键名加 `loss_v7/` 前缀而不是裸 term 名：裸名会和
                        # `policy_loss` / `value_loss` 在同一面板里混读，而它们
                        # 的**求和口径不同**（这里是 `weighted`，已乘系数）。
                        # 这里才是 `loss_v7/*` 的**唯一** D2H 点（打点节奏）：
                        # 上游存的是设备张量，见 `_v7_terms_last` 处注释。
                        **{k: float(v) for k, v in _v7_terms_swanlab.items()},
                    }
                    # ---- 面板过滤（2026-10-06 瘦身）----
                    #  · `_SWANLAB_DROP_KEYS`：照算不报的键（见常量处注释）；
                    #  · `loss_v7/*` 里**权重为 0** 的项不上曲线 —— 那是设计上的
                    #    平线，解释走 `run/loss_coef/*`（config 面板），不上图。
                    #    真机 2026-10-06：13 条里 9 条恒 0 平线把面板糊死。
                    _sw = {k: v for k, v in _sw.items()
                        if k not in _SWANLAB_DROP_KEYS
                        and not (k.startswith('loss_v7/')
                                    and _SWANLAB_V7_WEIGHTS.get(k[8:], 0) == 0)}
                    swanlab_logger.log(_sw, step=step)
                except Exception as e:
                    logger.warning("[swanlab] log 失败: %s", e)

            if _do_stdout:
                # 内核剖析结束：打印 top kernel 耗时表
                if _prof_ctx is not None and step >= _prof_at + _prof_span:
                    _prof = _prof_ctx
                    # **先摘掉再收尾**。原实现把 `__exit__` 放在 try **外面**、且收尾
                    # 后仍留着 `_prof_ctx` ⇒ 窗口一过，之后每个 stdout 步都会再
                    # `__exit__` 一次、再失败一次（2026-10-07 真机：每 2 步刷一行
                    # "'profile' object has no attribute 'key_averages'"），而 `__exit__`
                    # 一旦抛异常还会直接打穿训练循环。置 None 保证「无论成败只收一次尾」。
                    _prof_ctx = None
                    try:
                        _prof.__exit__(None, None, None)
                        # 先试 `key_averages()`；取不到就退回下面的 chrome trace
                        # 解析（老版本 NPU 版 torch 的 profile 是独立类，只有 8 个方法、
                        # 没有 key_averages ⇒ AttributeError）。
                        try:
                            _ka = _prof.key_averages()
                            _sort_key = 'self_cuda_time_total'
                            try:
                                table = _ka.table(sort_by=_sort_key, row_limit=18)
                            except KeyError:
                                # 键名随 torch 版本变：退一步用无排序的表，至少能看到
                                # 有哪些 kernel 与它们的事件数。
                                table = _ka.table(row_limit=18)
                            logger.info("[profile] 内核耗时 top-18（按 %s 排序）:\n%s",
                                        _sort_key, table)
                        except (AttributeError, NotImplementedError):
                            _table, _cats = _prof_trace_table(_prof, row_limit=18)
                            logger.info("[profile] 内核耗时 top-18（chrome trace 解析）:\n%s",
                                        _table)
                            # cat 摘要是**探针**：分组口径只能先按通用 schema 猜，
                            # 真机第一次回来先看这一行有没有把 kernel 单列出来，
                            # 没有就据此改 `_prof_parse_trace` 的聚合键。
                            logger.info("[profile] trace cat 总耗时: %s", _cats)
                    except Exception as e:
                        logger.warning("[profile] 打印内核耗时表失败: %s", e)

            # 定期保存快照（仅主进程写盘）。
            # 走后台线程：实测这一步会把 GPU 利用率**打到 0**（A100 40G 上每
            # ~25 秒一次，曲线与「GPU Time Spent Accessing Memory」同形）⇒ 卡的是
            # host 的D2H + 落盘，不是算力。缘由见 `_AsyncSnapshotWriter`。
            # 快照本身仍在训练线程里做（`state_dict()` + 深拷贝成 CPU），只是
            # `torch.save` 那段阻塞 IO 挪走 —— 深拷贝不能挪，否则后台线程读的
            # 是仍在被训练改写的显存。
            if is_main and args.save_every > 0 and step % args.save_every == 0:
                _t_save0 = time.perf_counter()
                _state = {
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'best_eval_acc': best_eval_acc,
                    'rng': torch.get_rng_state(),
                }
                if ema is not None:
                    _state['ema_shadow'] = ema.shadow
                _snap_writer.submit(_plain_state_dict(model), _state,
                                    args.out + '.latest',
                                    args.out + '.latest.train_state')
                _t_save += time.perf_counter() - _t_save0

            # 定期评估：综合指标（所有 rank 都做 eval，避免 barrier 死锁）
            # V7 走 `evaluate_metrics_v7`（22 通道 + dict 返回），12 通道走原版。
            # 两版**返回同构**，所以下面的 best-model / early-stop 逻辑零改动。
            if (args.eval_every > 0 and step % args.eval_every == 0
                    and len(eval_idx) > 0):
                if ema is not None:
                    ema.apply_shadow()
                _t_eval0 = time.perf_counter()
                if _v7_on:
                    metrics = evaluate_metrics_v7(
                        model, dataset, eval_idx, bs, device, amp_dtype,
                        max_batches=args.eval_max_batches, prefetcher=eval_pf,
                        use_channels_last=use_channels_last)
                else:
                    metrics = evaluate_metrics(
                        model, dataset, eval_idx, bs, device, amp_dtype,
                        max_batches=args.eval_max_batches,
                        use_channels_last=use_channels_last)
                _t_eval += time.perf_counter() - _t_eval0
                if ema is not None:
                    ema.restore()
                # 曾经在这里周期性 `empty_cache()` 回收分配器缓存段，2026-10-01 撤掉：
                # 它在 加速器 上没验证过，且**每次 eval 都调**（默认 ~35 分钟一次）在训练
                # 主路径上；等真机上确认了碎片确实在爬升、再按实测收益决定值不值得加。
                # OOM 恢复路径里的那次 `empty_cache()`（在 except 分支）是既有的，保留。
                if is_main:
                    # 不打 seed：评估固定 `augment=False`，`EVAL_SAMPLING_SEED` 与 `rng=`
                    # 都不被消费，指标与种子无关。打出它等于宣称「这批数字依赖这个常量」，
                    # 运维改了却发现数字不动 —— 那是主动误导。`augment=off` 陈述的是真实
                    # 状态：这批指标算在**未经随机对称变换**的验证集上。
                    logger.info(
                        "[eval] step=%d top1=%.4f top5=%.4f top10=%.4f "
                        "kl=%.4f brier=%.4f (n=%d, batches=%d, truncated=%s, augment=off)%s",
                        step, metrics['top1'], metrics['top5'], metrics['top10'],
                        metrics['kl'], metrics['brier'], metrics['n'],
                        metrics['batches'], metrics['truncated'],
                        " new best" if metrics['top1'] > best_eval_acc else "")
                    if metrics['truncated']:
                        logger.info(
                            "[eval] 验证集被截断：本次只评估了 %d 批（--eval-max-batches=%s，"
                            "传 <=0 跑满全部验证集）", metrics['batches'],
                            args.eval_max_batches)
                    # SwanLab 记录评估指标
                    if swanlab_logger is not None:
                        try:
                            swanlab_logger.log({
                                "eval_top1": metrics['top1'],
                                "eval_top5": metrics['top5'],
                                "eval_top10": metrics['top10'],
                                "eval_kl": metrics['kl'],
                                "eval_brier": metrics['brier'],
                                "eval_n": metrics['n'],
                                "eval_batches": metrics['batches'],
                                "eval_truncated": metrics['truncated'],
                                "best_eval_acc": best_eval_acc,
                                # ---- 补三项（2026-10-01）----
                                # eval_used_ema: eval 跑的是 **EMA shadow** 还是原权重。
                                #   不报这个就不知道在评什么 —— `--use-ema 1` 时曲线上的
                                #   eval_top1 与训练 loss 不是同一个模型的两个量。
                                # eval_lr: 把 eval 点与 lr 曲线对齐（lr 已退火到多少时
                                #   评的，这在做「早停/最佳模型」判断时需要。
                                # eval_gap_to_best: 早停判据（--early-stop-metric）直接可见。
                                "eval_used_ema": bool(ema is not None),
                                "eval_lr": float(optimizer.param_groups[0]['lr']),
                                "eval_gap_to_best": (best_eval_acc - metrics['top1']
                                                    if best_eval_metric >= 0
                                                    else float('nan')),
                            }, step=step)
                        except Exception as e:
                            logger.warning("[swanlab] eval log 失败: %s", e)
                # 「最佳模型保存」判据（top1，越大越好）只负责 best_eval_acc 与落盘，
                # 与「早停判据」（--early-stop-metric）无关 —— 哪怕默认值下两者都读
                # top1，语义上仍是两个独立指标。早停计数器由 _early_stop_decision
                # 独占维护（改善则归零、无改善则累加），此处复位会让
                # --early-stop-metric loss 的 patience 被 top1 的改善反复抹平、
                # 永远攒不满，早停形同虚设。
                if metrics['top1'] > best_eval_acc:
                    best_eval_acc = metrics['top1']
                    if is_main:
                        if ema is not None:
                            ema.apply_shadow()
                        # 先 join 掉在飞的后台快照：它写的也是 `.latest` 系列，
                        # 不等它收尾就写 `args.out` 之外的最终件没问题，但
                        # 收尾前必须确保没有半份文件留在盘上。
                        if _snap_writer is not None:
                            _snap_writer.join()
                        save_model(model, args.out)
                        # 保存 train_state 到最佳模型路径，确保 --resume 最佳模型时状态一致
                        _state = {
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'step': step,
                            'epoch': epoch,
                            'best_eval_acc': best_eval_acc,
                            'rng': torch.get_rng_state(),
                        }
                        if ema is not None:
                            _state['ema_shadow'] = ema.shadow
                        torch.save(_state, args.out + '.train_state')
                        if ema is not None:
                            ema.restore()

                # 早停检查：判定只在 rank0 做（计数器语义不变），结论写进 stop_flag
                if args.early_stop == 1 and is_main:
                    if args.early_stop_metric == 'loss':
                        current_metric = metrics.get('brier', metrics['kl'])
                    else:  # top1
                        current_metric = metrics['top1']
                    improved, early_stop_counter, should_stop = _early_stop_decision(
                        args.early_stop_metric, current_metric, best_eval_metric,
                        early_stop_counter, args.early_stop_patience)
                    if improved:
                        best_eval_metric = current_metric
                    if should_stop:
                        logger.info("[early_stop] 连续 %d 次 eval 无改善，触发早停",
                                    args.early_stop_patience)
                        if swanlab_logger is not None:
                            swanlab_logger.log({"early_stop": True, "final_step": step}, step=step)
                        # 不在这里 break：早停要跳出的是 **epoch** 循环，
                        # 而这里在 step 循环内，break 只能跳完本 epoch —— 见下方双层 break
                        stop_flag.fill_(1)

                # 停止标志同步：所有 rank 必然同点到达（eval 块的进入条件与 CLI
                # 参数逐 rank 相同），故**无条件** broadcast，stop_flag 未置位
                # 也照发。放在 `is_main` 之外是关键：这里少了它，其他 rank 就
                # 永远收不到停止信号。
                _sync_stop_flag(stop_flag, is_dist, args.early_stop == 1)

                # `.item()` 是同步点，而 stop_flag 只在 `args.early_stop == 1`
                # 时才可能被置位 ⇒ 早停没开时每次都白付（把 host 拉回等 GPU）。
                # 不能改成「先查有没有置位」—— 那本身就得先 `.item()`。
                if args.early_stop == 1 and stop_flag.item():
                    break

        # 双层 break 之二：跳出 step 循环后还要跳出 epoch 循环（缩进 8 = epoch
        # 循环体、step 循环之外），否则只跳过本 epoch 剩下的 step，外层
        # for epoch 照开下一轮 → 早停等于没开。
        if stop_flag.item():
            if is_main:
                logger.info("[early_stop] 提前结束训练，进入收尾流程（ONNX 导出 / 最终评估）")
            break

    # 训练循环结束：**必须**先把后台写盘器收尾，否则最后一份快照可能还只写了一半
    # 就被后续的「最终评估 / 导出」读走（或进程直接退出）。
    if _snap_writer is not None:
        _snap_writer.close()

    # 训练结束后导出 ONNX（可选）
    if args.export_onnx == 1 and is_main:
        logger.info("[train] 开始导出 ONNX 模型...")
        from src.inference import GoAI
        ai = GoAI(model_path=args.out, board_size=args.board_size, device='cpu')
        onnx_path = args.out.replace('.pth', '.onnx')
        ai.export_onnx(onnx_path, quantize_int8=args.onnx_quantize == 1)
        logger.info("[train] ONNX 导出完成: %s", onnx_path)

    # 最后一步评估：使用 EMA 权重（如果启用）
    if len(eval_idx) > 0:
        if is_main:
            logger.info("[train] 开始最终评估...")
        if ema is not None:
            ema.apply_shadow()
        if _v7_on:
            final_metrics = evaluate_metrics_v7(
                model, dataset, eval_idx, bs, device, amp_dtype,
                max_batches=args.eval_max_batches, prefetcher=eval_pf,
                use_channels_last=use_channels_last)
        else:
            final_metrics = evaluate_metrics(
                model, dataset, eval_idx, bs, device, amp_dtype,
                max_batches=args.eval_max_batches,
                use_channels_last=use_channels_last)
        if ema is not None:
            ema.restore()
        if is_main:
            # 同上：FINAL 行也不打 seed（`augment=off` 才是这批数字的真实前提）
            logger.info(
                "[eval] FINAL top1=%.4f top5=%.4f top10=%.4f "
                "kl=%.4f brier=%.4f (n=%d, batches=%d, truncated=%s, augment=off)",
                final_metrics['top1'], final_metrics['top5'], final_metrics['top10'],
                final_metrics['kl'], final_metrics['brier'], final_metrics['n'],
                final_metrics['batches'], final_metrics['truncated'])
            if final_metrics['truncated']:
                logger.info(
                    "[eval] 最终评估的验证集被截断：只评估了 %d 批（--eval-max-batches=%s，"
                    "传 <=0 跑满全部验证集）", final_metrics['batches'],
                    args.eval_max_batches)
            # SwanLab 记录最终评估结果
            if swanlab_logger is not None:
                try:
                    swanlab_logger.log({
                        "final_top1": final_metrics['top1'],
                        "final_kl": final_metrics['kl'],
                        "final_brier": final_metrics['brier'],
                        # 覆盖度可见化（P2.1 遗留）：收尾指标同样只覆盖了验证集的一部分，
                        # 截断在 swanlab 曲线上不可见就等于没说
                        "final_batches": final_metrics['batches'],
                        "final_truncated": final_metrics['truncated'],
                    }, step=total_steps)
                    swanlab_logger.finish()
                    logger.info("[swanlab] 实验跟踪已完成")
                except Exception as e:
                    logger.warning("[swanlab] finish 失败: %s", e)

    # C2NET 回传结果
    if _c2net_ctx is not None and is_main:
        try:
            from c2net.context import upload_output as _c2net_upload
            _c2net_upload()
            logger.info("[c2net] 结果已回传到 OpenI 平台")
        except Exception as e:
            logger.warning("[c2net] 回传失败: %s", e)

if __name__ == "__main__":
    main()
