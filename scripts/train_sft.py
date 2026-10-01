"""
19x19 监督训练脚本（AlphaGoZero 风格策略+价值网络）。

依赖 scripts/build_dataset.py 产出的紧凑 npz 数据集。监督学习不涉及
自我对弈，因此绕开了 H1/H2/H4/H5/L1 等自我对弈性能问题。

用法:
    python scripts/train_sft.py --data data/sft_dataset.npz --device cuda --use-amp \
        --batch-size 512 --epochs 5 --save-every 2000 --out models/sft.pt
"""

import argparse
import logging
import multiprocessing as mp
import os
import queue
import random
import sys
import threading
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


# ---- NPU(Ascend/CANN) 后端兼容辅助 ----
# torch_npu 是可选依赖，未安装时不应让静态分析/运行时报错。所有 NPU 访问都经
# 过这里统一守卫；未安装 torch_npu 的环境（纯 CUDA 开发机）这些函数返回安全默认值。
def npu_is_available() -> bool:
    if not hasattr(torch, 'npu'):
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def npu_get_device_name(idx: int = 0) -> str:
    try:
        return str(torch.npu.get_device_name(idx))
    except Exception:
        return 'Ascend-NPU'


def npu_memory_reserved(device) -> float:
    try:
        return float(torch.npu.memory_reserved(device))
    except Exception:
        return 0.0


def npu_empty_cache() -> None:
    try:
        torch.npu.empty_cache()
    except Exception:
        pass


def npu_grad_scaler(enabled: bool, **kwargs):
    return torch.npu.amp.GradScaler(enabled=enabled, **kwargs)


def npu_out_of_memory_error_type():
    # 未装 torch_npu 的环境（纯 CUDA 开发机）torch.npu 属性不存在，
    # 必须先 hasattr 守卫，否则 OOM 异常处理路径本身会抛 AttributeError
    if not hasattr(torch, 'npu'):
        return None
    return getattr(torch.npu, 'OutOfMemoryError', None)


def _auto_select_device():
    """自动选择最优训练设备。

    策略（按优先级）：
      1. 探测 CUDA 与 NPU 各卡的空闲显存，挑空闲显存最大的那张卡。
      2. 若两后端都可用，选「空闲显存更大」的后端（A100 通常 > 910B，但按实测）。
      3. 都不可用则回退 CPU。
    返回形如 'cuda:0' / 'npu:1' / 'cpu' 的具体设备串。
    """
    def _cuda_free(idx):
        try:
            torch.cuda.synchronize(idx)
            total = torch.cuda.get_device_properties(idx).total_memory
            alloc = torch.cuda.memory_allocated(idx)
            return max(0, total - alloc)
        except Exception:
            return 0

    def _npu_free(idx):
        try:
            total = torch.npu.get_device_properties(idx).total_memory
            alloc = torch.npu.memory_allocated(idx)
            return max(0, total - alloc)
        except Exception:
            return 0

    best = None  # (free_bytes, backend, idx)
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            free = _cuda_free(i)
            if best is None or free > best[0]:
                best = (free, 'cuda', i)
    if npu_is_available():
        try:
            n = torch.npu.device_count()
        except Exception:
            n = 0
        for i in range(n):
            free = _npu_free(i)
            if best is None or free > best[0]:
                best = (free, 'npu', i)
    if best is None:
        return 'cpu'
    _, backend, idx = best
    return f'{backend}:{idx}'


def _dist_debug_level():
    """当前 `TORCH_DISTRIBUTED_DEBUG` 的取值（大写；未设置返回 ''）。"""
    return (os.environ.get('TORCH_DISTRIBUTED_DEBUG') or '').strip().upper()


def _downgrade_npu_dist_debug(logger):
    """NPU 后端下把 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 降为 `OFF`。

    为什么（NPU 专属，2026-09-30 建立、2026-10-01 改写理由）：`DETAIL` 会给通信域
    套一层**一致性检查 wrapper**（torch 的 `_create_process_group_wrapper` →
    `_ProcessGroupWrapper`，顺带另建一条 gloo 辅助 PG），而这层 wrapper 在
    **每一次 collective 之前**都要跑一发 `monitored_barrier` —— 官方文档
    `docs/source/distributed.md` 的 `TORCH_DISTRIBUTED_DEBUG` 一节就是这么写的
    （"consistency and synchronization checks **on every collective call** …
    creating a **wrapper process group** … include a `monitored_barrier`"）。
    ⚠ 关键点：这层 wrapper 是 **`init_process_group` 建域时按 debug level 挂上去的**，
    **与用哪种包裹层无关**（FSDP1 / DDP 都一样）。而本仓库从未在 NPU 上验证过
    它的开销（910A + HCCL 下 DETAIL 的实测数据缺失）⇒ 在 910A 上仍降为 OFF。

    （历史，已退役：这条降级最初的理由是当时那套包裹层的「执行顺序自检」——
    FSDP1 的 `fsdp/_exec_order_utils.py` 里 `_checking_order = debug_level ==
    DETAIL`，它在**每个被包裹模块的每次前向**都额外做一次 `all_gather_into_tensor`
    跨 rank 比对参数句柄，在 HCCL 上是纯负担。那套包裹层在 2026-10-01 被 DDP
    取代，该理由随之失效；「保持 OFF」的决定不变。）

    ⚠ 换轨时（2026-10-01）曾经把上面那条历史理由换成「DDP 下 DETAIL 只加 reducer
    bookkeeping、不额外发 collective」—— **那是错的**：额外 collective 来自上面
    那层 PG wrapper（每次 collective 一发 `monitored_barrier`），与 DDP 无关。
    当前这版表述才是有据的，而且比原理由更强：原理由只覆盖「FSDP1 这一种包裹层」，
    现在覆盖**所有**包裹层。

    ⚠ **实测更正**：曾怀疑 DETAIL 是 4 卡 HCCL 报错的元凶（后来查到的真因是
    上一次崩掉的进程留下 HCCP 状态，`EJ0001 ... Maybe the last training process
    is running`）。降级仍然保留，理由只剩「DETAIL 的 PG wrapper 代价与包裹层无关
    且在 NPU 上未验证、保持 OFF」，但**它不是修那个错的原因**，别再拿它当根因。

    在 `init_process_group` **之前**调用：那时通信域还没建立。CUDA/
    CPU 后端不动（inductor/nccl 上 DETAIL 的开销可接受，且它是排查 nccl hang 的
    正统手段）。
    """
    if _dist_debug_level() != 'DETAIL':
        return
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
    try:
        import torch.distributed as dist
        setter = getattr(dist, 'set_debug_level', None)
        if setter is not None:
            setter(dist.DebugLevel.OFF)
    except Exception as e:  # noqa: BLE001 — 老版本没有这个 setter，不该因此拦住启动
        logger.warning("[dist] 降级 debug level 失败（忽略）：%s", e)
    logger.info("[dist] NPU 后端：TORCH_DISTRIBUTED_DEBUG DETAIL → OFF"
                "（DETAIL 会给每个 PG 套一层一致性检查 wrapper，每次 collective "
                "前跑一次 monitored_barrier；该代价与用哪种包裹层无关，本仓库未在 "
                "NPU 上验证过其开销，保持 OFF）")


def _dist_env_snapshot():
    """通信相关环境快照（失败诊断用；全部是只读查询）。"""
    import torch as _t
    import torch.distributed as dist
    bits = ['torch={}'.format(_t.__version__)]
    try:
        import torch_npu  # noqa: F401
        bits.append('torch_npu={}'.format(getattr(torch_npu, '__version__', '?')))
        bits.append('可见NPU数={}'.format(_t.npu.device_count()))
    except Exception:  # noqa: BLE001
        try:
            bits.append('可见GPU数={}'.format(_t.cuda.device_count()))
        except Exception:  # noqa: BLE001
            bits.append('设备数=?')
    bits.append('ASCEND_RT_VISIBLE_DEVICES={!r}'.format(
        os.environ.get('ASCEND_RT_VISIBLE_DEVICES')))
    bits.append('RANK={}'.format(dist.get_rank()))
    bits.append('WORLD_SIZE={}'.format(dist.get_world_size()))
    bits.append('LOCAL_RANK={}'.format(os.environ.get('LOCAL_RANK')))
    bits.append('MASTER_ADDR={}:{}'.format(os.environ.get('MASTER_ADDR'),
                                           os.environ.get('MASTER_PORT')))
    return ' | '.join(bits)


def _dist_preflight_check(backend, device, logger):
    """通信域自检：立刻试**一发** all_reduce，把通信问题从「训练途中」提前到「启动时」。

    为什么必需：HCCL / NCCL 的通信域是**惰性**创建的 —— `init_process_group`
    只登记后端，真正建链发生在**第一发 collective**。不主动试一发，失败就落在
    「第一个 batch 的前向」里，而且报成 HCCL 的通用错误
    （`ProcessGroupHCCL.cpp:64` + `HCCL error`），真因藏在日志更前面的
    `EJ0001 ... Maybe the last training process is running` 里 —— 2026-09-30 的
    4 卡事故就是这样查了 20 分钟才发现是**上一次崩掉的进程留下 HCCP 状态**。

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
            'HCCP 初始化被拒 —— 日志里真正的报错是\n'
            '    EJ0001: Failed to initialize the HCCP process. Reason: '
            'Maybe the last training process is running.\n'
            '  处置（按顺序，别跳步）：\n'
            '    1) ps -ef | grep -E "train_sft|torchrun" | grep -v grep   '
            '# 找残留\n'
            '    2) npu-smi info                                       '
            '# Processes 表应为空\n'
            '    3) pkill -f train_sft.py; pkill -f torchrun; sleep 30    '
            '# HCCP 清理需要时间（报错里的 Solution 是 10s，实测 30s 更稳）\n'
            '    4) 仍失败：npu-smi info -t reset -i <0..3> -c 0         '
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
# 事故形状：当时的分布式包裹层（**历史：FSDP1，已退役**）的 docstring 要求
# from-scratch 也必须同步初始权重 ——「各 rank 独立、各自不同，不同步的话第一步
# 拿到的就是拼错的权重」。但它的 `sync_module_states=True` **从未被传**
# （当时的构造参数白名单里也没有），**修复前**全文件唯一的 `dist.broadcast`
# 是 `_sync_stop_flag` 的 stop_flag —— **没有任何参数广播**。
# `shell/train_sft_npu_4card_v21.sh` 不传 `--resume`/`--model` ⇒ from-scratch
# ⇒ 第一步之后各 rank 的权重就永久分叉：梯度虽然被 all-reduce 拉齐，但被拉齐的
# 是「起点不同」的同一份梯度，从 step 0 起两份权重就不是同一个模型了。
#
# 为什么用显式 broadcast 而不是包裹层自带的 `sync_module_states`
# --------------------------------------------------------
# 1. 后者在 torch 2.1 上需要 `param_init_fn` 配套（未初始化的参数会留在 CPU），
#    而本文件所有参数在包裹前已 `.to(device)`；当时为绕这条还要维护一层
#    「按已安装签名过滤未知 kwarg」的版本兼容（**历史：随 FSDP1 包裹层一并删除的
#    `_drop_unsupported_kwargs`**）。
# 2. buffer 不需要额外同步：`BatchNorm2d` 的 `running_mean`/`running_var`/
#    `num_batches_tracked` 是确定性 0/1 初始化，各 rank 本来就逐位一致。
# 3. 显式 broadcast 是**无条件**的 —— 不依赖「resume 路径已经同步了」这类推理，
#    运行时可验证，且能被 AST 测试直接钉住顺序。
#
# ⚠ 与 DDP 自带同步的关系（**顺序的硬理由**，详见 main() 里包裹点上方的注释）：
#   `DistributedDataParallel.__init__` 在**它自己构造时**也会把 rank0 的
#   params/buffers 广播出去（`_ddp_init_helper` → `_sync_module_states`），但那时
#   EMA 已经构造完、把各 rank 自己的随机权重 clone 进了 shadow。所以必须由本组
#   函数把广播放在 EMA 之前，包裹层自带的那次只能当第二道保险。
#
# 代价：36.3 MB 一次性广播 + 167 次小 collective（= `build_v21_net` 的参数张量数，
# 实测 167 个 / 9,067,443 参数），**只在启动时发生一次**。
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

    本地 checksum = 全部参数 fp64 求和；`all_gather` 出 world_size 个标量后
    逐位比对。失败即抛，不允许静默继续 —— 与 `_dist_preflight_check` 同一立场：
    通信域的问题是惰性的，要主动试一发。
    """
    if not _dist_active():
        return
    acc = None
    with torch.no_grad():
        for p in model.parameters():
            # `sum(dtype=torch.float64)` 而不是 `p.detach().double().sum()`：两者
            # 结果**逐位相同**（本地实测），但后者把**每个参数整份物化成 fp64** ——
            # v21 有 167 个张量 / 9.07M 参数，峰值临时显存 = 最大单参数 ×8，且每个
            # 张量多一发 cast kernel。本文件唯一的既有先例是 `term = sq.double()`，
            # 那是 **0 维标量** 的 cast，**不能**当作「整张量 fp64 cast 在 910A 上可用」
            # 的证据（开发机纯 CPU，全量测试通过不构成 NPU 证据）。这段在启动期
            # **无条件**执行，一旦 910A 不支持就是 4 卡 run 立刻死 —— 所以宁可走
            # 「只 reduce、不 cast」那条路，把风险面从 cast+reduce 缩到只 reduce。
            s = p.detach().sum(dtype=torch.float64)
            acc = s if acc is None else acc + s
    buf = acc.reshape(1).to(torch.float64)
    gathered = [torch.zeros_like(buf) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, buf)
    if not all(torch.equal(gathered[0], g) for g in gathered):
        vals = ', '.join('rank%d=%.8g' % (i, g.item()) for i, g in enumerate(gathered))
        raise RuntimeError(
            '[dist] 初始权重一致性自检失败：各 rank 的参数 checksum 不一致。\n'
            '  各 rank checksum: %s\n'
            '  两种可能：\n'
            '    1) from-scratch 起手时初始权重确实不同 —— 说明广播没生效或\n'
            '       调用点晚于模型构造，检查 _sync_init_weights_from_rank0 的位置；\n'
            '    2) 初始权重里有 NaN/Inf —— `torch.equal` 对 NaN 返回 False，\n'
            '       会把这一种误报成前一种。（真因通常是某个 zero-init 假设被破坏）\n'
            '  环境: %s' % (vals, _dist_env_snapshot()))
    logger.info("[dist] 初始权重一致性自检通过 | world_size=%d | checksum=%.8g",
                dist.get_world_size(), gathered[0].item())


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
    try:
        import torch_npu  # noqa: F401 — 确保 torch.npu 可用
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        cap = (p.major, p.minor)
        if cap >= (8, 0):
            amp_status = "bf16+fp16 可用（%s, sm_%d%d）" % (p.name, cap[0], cap[1])
        else:
            amp_status = "仅 fp16（%s, sm_%d%d，Volta/Turing 无 bf16）" % (p.name, cap[0], cap[1])
    elif npu_is_available():
        amp_status = "bf16(910B)/fp16(910A) 可用（Ascend NPU，运行时按型号选择）"
    else:
        amp_status = "不支持（CPU 走 FP32）"
    logger.info("[env] flash-attn: %s", fa_status)
    logger.info("[env] torch.compile: %s | 混合精度: %s", comp_status, amp_status)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.networks.alphanet import V21_CFG, build_v21_net
from src.data.dataset import SupervisedDataset
from scripts.build_dataset import build


def save_model(model, path):
    """保存模型权重，并剥离 DDP 包裹产生的 'module.' 前缀与 torch.compile 产生的
    '_orig_mod.' 段，保证存档无论是否经 DDP/compile 都能被后续普通加载/resume 使用。

    ⚠ DDP 下**直接** `model.state_dict()` 就是完整权重，不需要任何汇聚 helper：
    DDP 每 rank 各持一份完整模型、只同步梯度，`state_dict()` 本身**不是
    collective**（没有 all-gather）。所以调用方把它放在 `is_main` 分支里既安全
    也没有额外开销 —— 键名形如 'module.backbone.…'，剥掉 'module.' 之后与包裹前
    完全同形，`src/inference.py` / `scripts/evaluate.py` / `webui.py` /
    `_load_model_state` 一行都不用改，旧 checkpoint 继续可读。

    ⚠ '_orig_mod.' 出现在**路径任意层级**，不只在开头。两种 compile 形态落点不同：
    整模型 compile（CUDA `--compile 1`）把**顶层**包成 OptimizedModule
    （'_orig_mod.backbone.xxx.weight'），而 Linear-only compile（D2，
    `--npu-graph-compile 1`，见 `_compile_linear_submodules`）顶层仍是原模型、
    只把每个 nn.Linear 包起来，段落在**路径中段**（'backbone.qkv._orig_mod.weight'）。
    故这里按 `'_orig_mod.' in k` 判存在性、按 `str.replace` 去**所有**层级，
    不能沿用旧实现的 `replace(..., 1)`（只换首个）——那在 Linear-only 形态下
    会原样留下中段前缀，存档键与 evaluate.py / inference / webui / convert_ckpt
    期待的未编译布局对不上，load_state_dict 直接失败。
    """
    sd = model.state_dict()
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    if any('_orig_mod.' in k for k in sd.keys()):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(sd, path)


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


def _locate_overflow(optimizer, logger, max_report=3):
    """定位梯度 inf/nan 的来源参数组，只在溢出当步调用（开销可忽略）。

    FP16 训练里「哪些参数在溢出」直接决定该调什么：value head 溢出通常指向
    value_loss_weight / value_lr_mult 过大；backbone 溢出则更可能是注意力
    logits 或 LR 本身。没有这一步就只能靠猜。

    参数组按 **LR 比值**分类而非写死下标——opt_groups 的顺序一旦调整
    （例如增删 no_decay 组），按下标判断就会误报。value 组的 LR 是基准的
    value_lr_mult 倍（本项目默认 5.0），故取「LR 明显高于最低组」作为判据。
    ⚠ `--value-loss-weight` 的默认已在 P4.5-fix 由 5.0 改为 1.0（删补偿），
    下面的告警文案同步改了；`--value-lr-mult` 仍是 5.0（那是 LR 倍数、不是
    损失补偿，不在本次裁决范围内）。
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
        logger.warning("[fp16] GradScaler 报告溢出，但未在参数组中找到 inf/nan"
                       "（可能出现在已被释放的中间张量里）")
        return
    for gi, lr, n_inf, n_nan, is_value in bad[:max_report]:
        logger.warning("[fp16] 梯度溢出：参数组 %d（%s, lr=%.2e）"
                       "有 %d 个 inf / %d 个 nan 参数",
                       gi, 'value head' if is_value else 'backbone/policy',
                       lr, n_inf, n_nan)
    if any(b[4] for b in bad):
        logger.warning("[fp16] 溢出集中在 value head —— 优先下调 --value-loss-weight"
                       "（默认 1.0，无补偿）与 --value-lr-mult（默认 5.0）")
    if any(not b[4] for b in bad):
        logger.warning("[fp16] 溢出涉及 backbone/policy —— 优先下调 --lr；"
                       "NPU 上注意力被强制走 math 并物化 logits，"
                       "可考虑调小 --attn-window 降低 logits 幅度")


def _ema_key(name: str) -> str:
    """EMA shadow 的键：去掉 torch.compile 往参数名里插的 '_orig_mod.' 段。

    ⚠ 必须去，否则编译一开就 KeyError：EMA 在**编译之前**按当时的
    `named_parameters()` 名字建 shadow，而 update / apply_shadow / restore 是按
    **运行时**的名字取键的。torch.compile 无论哪种形态都会改名字（整模型 compile
    插在开头，Linear-only compile 插在路径中段），键空间一变就再也对不上。
    这条路径此前是直接崩的、而非被跳过：`shell/train_sft_a100_1card.sh` 同时开了
    `--compile 1` 与 `--use-ema 1`，5 个 NPU SFT 脚本也都开了 `--use-ema 1`。

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
            # Ascend 的 ACL 单次启动开销明显高于 CUDA，累积可观。
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
    """在 CUDA/NPU 上开启 autocast，dtype 由设备能力决定（A100/BF16、V100/FP16、NPU/BF16）。
    CPU 或 amp 关闭时返回 nullcontext。device 字符串支持 'cuda'/'cuda:0'/'npu'/'npu:0' 等。"""
    dev = device.split(':')[0] if isinstance(device, str) else str(device)
    if dev in ('cuda', 'npu'):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            try:
                return torch.amp.autocast(dev, dtype=dtype)
            except TypeError:
                # 老接口回退
                if dev == 'cuda':
                    return torch.cuda.amp.autocast(enabled=True, dtype=dtype)
                return torch.npu.amp.autocast(enabled=True, dtype=dtype)
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


def load_dataset(path):
    """加载单个 .npz 训练集。"""
    d = np.load(path, allow_pickle=False)
    # 通道数必须与**模型侧同改**（D1：train_sft 的模型恒 v21 = 17 路）。
    # SupervisedDataset 的默认 12 是给旧权重回归留的（dataset.py C7），这里
    # 不显式传就会拿 12 路平面去喂 17 路 stem —— 第一个 batch 就形状错。
    return SupervisedDataset({k: d[k] for k in d.files},
                             n_channels=V21_CFG['in_channels'])


# ---- eval 的确定性：固定采样源 + 与训练 RNG 流隔离 + 关闭数据增强 ----
# 验证集评估本来要走 8 路对称增强抽样（`SupervisedDataset.sample_batch_numpy`）。修复前
# 两个评估函数都不给 rng，抽样便落到**全局** np.random 上，后果有两层：
#   ① 同一份权重、同一份 eval_idx 连跑两次 eval，指标不同（KL/Brier 尤其抖）——
#      「模型变好了」与「这次抽到的变换不一样」分不开；
#   ② 每次 eval 都推进全局流 → 训练侧 `rng.shuffle(train_idx)` 与训练 batch 的增强抽样
#      结果取决于「eval 跑过几次、什么时候跑」→ **eval 频率会改写训练轨迹**。
# P2.2 用下面这个固定种子的独立 Generator + `_isolated_global_rng()` 兜底解决了这两层。
#
# 但「固定」不等于「正确」：被评估的仍然是**随机挑了对称**的验证集，推理时不会出现
# 随机翻转的棋盘，顶-1 的分母里还混进了「对称等价但标签已同步」的一致性。故评估路径
# 一律传 `augment=False` —— **评估零随机**，于是「指标与采样种子无关」才真正成立。
# 连带后果：`EVAL_SAMPLING_SEED` 与 `evaluate_*(..., rng=)` 现在**不被消费**（只是被
# 透传给 `sample_batch_numpy`），保留这条链路是为了既有的契约锁与将来的显式入口 ——
# 训练侧 `_BatchPrefetcher` 的 `seed=1234` 与它数值相同但依旧互不相干。
# 增强**只在训练侧**发生（`sample_batch` / 预取器 worker / `train_sft_ms.py` 走默认
# `augment=True`），见 `tests/test_eval_no_augment.py`。
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
                state = state.to(memory_format=torch.channels_last)
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
                state = state.to(memory_format=torch.channels_last)
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
            with torch.inference_mode():
                target = torch.zeros_like(log_p).scatter_(
                    1, move_t.unsqueeze(1).clamp(
                        max=log_p.shape[1] - 1), 1.0)
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
            # ⚠ 作用在**裸输出**上：`value_pred` 就是 `model(state)` 的第二个
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


def load_from_path(path, board_size, max_games_per_tgz=0):
    """加载训练数据。

    - 若 path 是文件：按 .npz 加载（兼容原行为）。
    - 若 path 是目录：递归扫描其下所有 .tgz/.tar.gz（用 build_dataset.build 解析）
      与 .npz（直接加载），合并成一个 SupervisedDataset。这样可直接喂一个装着
      多个分片 tgz 的文件夹，无需先手动 build_dataset 成单个 npz。
    """
    if os.path.isdir(path):
        import glob
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
        # 同 load_dataset：平面通道数随 V21_CFG 走（与模型侧同改）
        return SupervisedDataset(merged, n_channels=V21_CFG['in_channels'])
    return load_dataset(path)


_worker_dataset = None  # multiprocessing worker 进程中由 initializer 设置


def _prefetch_worker_init(dataset):
    """multiprocessing worker 初始化：在子进程中保存 dataset 引用。"""
    global _worker_dataset
    _worker_dataset = dataset


def _prefetch_worker(wi, task_q, res_q, seed, dataset):
    """multiprocessing worker：从 task_q 取任务，计算后放 res_q。"""
    rng = np.random.default_rng(seed + wi)
    while True:
        item = task_q.get()
        if item is None:
            return
        step, pos, sub_idx = item
        try:
            s, m, v = dataset.sample_batch_numpy(sub_idx, rng=rng)
            res_q.put((step, pos, s, m, v, None))
        except Exception as e:  # noqa: BLE001
            res_q.put((step, pos, None, None, None, e))


class _BatchPrefetcher:
    """后台多进程并行构造训练 batch，与 NPU 前向/反向重叠。

    把每个 batch 的样本下标切成 num_workers 个子块，由 num_workers 个后台进程
    并行调用 dataset.sample_batch_numpy()（绕过 GIL，numpy 操作真正并行），
    主进程按序拼回整批。

    两步流水：submit() 投递一个 batch 的下标，next() 取回构造好的 numpy 数组。
    两个有界队列提供背压，避免无限预取吃内存。各进程用独立 np.random.Generator。
    """

    def __init__(self, dataset, num_workers=4, prefetch=2, seed=1234):
        # ⚠ 护栏：**绝不能在设备运行时初始化之后**构造本类（4 卡 910A 的 OOM
        # 直接原因，2026-09-30）。`mp.Process` 默认 fork，子进程会整份继承父
        # 进程的 CANN/CUDA 上下文与已分配显存映射 ⇒ 每卡被旁挂 4 份 ≈ 24 GiB，
        # 而 PyTorch 自己只记 6.3 GB（实测 HBM 94% / AICore 0%）。GC 管不到
        # 别的进程继承来的映射。正确做法见 main()：数据集加载与本类的构造都在
        # `init_process_group` / `set_device` **之前**。
        for _dev in ('npu', 'cuda'):
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
        cap = self.k * self.prefetch
        self._task_q: mp.Queue = mp.Queue(maxsize=cap)
        self._res_q: mp.Queue = mp.Queue(maxsize=cap)
        self._step = 0     # 下一个待投递 batch 的编号
        self._expect = 0   # 下一个待取回 batch 的编号
        self._pending: dict = {}  # step -> [(pos, s, m, v, err)] 已收到但还没收集完的
        self._processes = []
        for wi in range(self.k):
            p = mp.Process(
                target=_prefetch_worker,
                args=(wi, self._task_q, self._res_q, seed, dataset),
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
                self._res_q.put((step, wi, None, None, None, None))
            else:
                self._task_q.put((step, wi, sub))

    def next(self):
        """取回下一个 batch，返回 (states_np, moves_np, values_np)。"""
        step = self._expect
        self._expect += 1
        # 从 _pending 中取出之前缓存的该 step 结果
        parts = self._pending.pop(step, [])
        # 如果不够 k 个，从队列中继续收
        while len(parts) < self.k:
            r_step, pos, s, m, v, err = self._res_q.get()
            if r_step == step:
                parts.append((pos, s, m, v, err))
            else:
                # 缓存未来 step 的结果
                self._pending.setdefault(r_step, []).append((pos, s, m, v, err))
        # 检查错误
        for pos, s, m, v, err in parts:
            if err is not None:
                raise err
        # 按 pos 排序并拼接
        parts = [(pos, s, m, v) for pos, s, m, v, _ in parts if s is not None]
        parts.sort(key=lambda x: x[0])
        if not parts:
            raise RuntimeError(f"step {step}: 所有子块为空")
        states = np.concatenate([p[1] for p in parts], axis=0)
        moves = np.concatenate([p[2] for p in parts], axis=0)
        values = np.concatenate([p[3] for p in parts], axis=0)
        return states, moves, values


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
        # 刻意**不在训练进程内** pip install swanlab。调用点在 torch / torch_npu
        # 已加载之后，此时改动 site-packages 可能破坏后续惰性导入；而 shell/*.sh
        # 在启动 python 之前已装过一次，那次失败的话这里必然也失败，只是白等一轮。
        # 「手动安装没问题」正是这个差别：装在解释器启动前，依赖已就位。
        #
        # 用 find_spec 区分「没装」与「装了但坏」：后者是云端常见坑——swanlab 依赖
        # pydantic>=2，而 MindSpore / torch_npu 常把 pydantic 钉在 1.x，于是
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
        swanlab.init(
            project="go-ai",
            name=f"sft_{args.board_size}x{args.board_size}_{args.ver}",
            config={
                "backbone_channels": args.backbone_channels,
                "backbone_res_blocks": args.backbone_res_blocks,
                "res_blocks": args.res_blocks,
                "convnext_blocks": args.convnext_blocks,
                "attn_blocks": args.attn_blocks,
                "value_channels": args.value_channels,
                "value_res_blocks": args.value_res_blocks,
                "policy_channels": args.policy_channels,
                "policy_layers": args.policy_layers,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
            },
        )
        logger.info("[swanlab] 实验跟踪已启用")
        return swanlab
    except Exception as e:  # noqa: BLE001
        logger.warning("[swanlab] 初始化失败: %s（指标请看 stdout 日志）", e)
        _emsg = str(e)
        if 'pydantic' in _emsg or 'TypeAdapter' in _emsg:
            logger.warning("[swanlab] 疑似 pydantic 版本冲突：swanlab 需要 pydantic>=2，"
                           "而 MindSpore / torch_npu 常钉 pydantic<2。"
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

    ⚠ 第一个参数传的是 **`log_loss`（报告口径）**，不是被 backward 的
    `opt_loss`。两者相差一个 `l2_report = c‖θ‖²`（P4.5b §3.1），故意不相等；
    形参名沿用 `loss` 是为了不打乱本仓既有的调用/断言写法，含义以上行为准。
    """
    return loss.item(), policy_loss.item(), value_loss.item()


# ---- D4（SFT 侧）：policy/value 损失口径 ------------------------------------------------
# 三个 CLI 开关：--policy-loss {huber,ce}（**默认 ce**，P4.5b 由 huber 改回 ce）、
# --value-loss {huber,mse}（默认 huber）、--huber-beta（默认 0.5，既是 smooth L1
# 的 beta 也是拐点 delta）。C8 修正：value 的 BCE 分支已删（见
# compute_value_loss docstring）。RL 侧（scripts/selfplay_train.py）的损失是
# P3-C/P3-D，**不经过这里** —— 本文件的函数只服务 train_sft 自己的调用点，改语义
# 不会波及 RL。
#
# 日志契约（不可动）：main() 打点仍用 loss / policy_loss / value_loss 三个键，
# 下游 run.txt、看板与 tests/test_run_txt_sync.py 的消费者绑着它们，
# tests/test_huber_loss.py::test_log_keys_unchanged 把三个键钉死。
# `loss` 这个键的**含义**在 P4.5b 变过（多了 c‖θ‖²），键名没变 —— 跨新旧 run
# 的曲线不可直接比，见 compute_l2_report docstring 与 report `## Fix2（b）增补`。
#
# P4.5 遗留问题的收口（用户 2026-09-27 裁决 = P4.5b）：policy 默认从 huber 改回
# ce 之后，「policy 梯度天生弱 ~A 倍、共享主干因此 value-only」这个结构性缺陷
# 从根上消失（Huber 打在概率域，梯度带 softmax 雅可比的 p≈1/A；CE 打在 log-prob
# 上，d(CE)/d(logit)=p−y 每坐标有界且与 A 无关）。实测 ce+huber、w=1 的
# value:policy 梯度比 = 1.113:1（与 P4.5 报告记录的 ce 备选口径逐位一致），
# 与老的 ce+5·bce（2.225:1）同一量级，**不需要补偿旋钮**。



def huber_loss(pred, target, beta=0.5, reduction='mean'):
    """Huber（smooth L1）—— D4 的唯一 Huber 实现，口径在本 docstring 钉死。

    数学式（逐元素误差 d = pred − target，N = 元素数）::

        h(d) = 0.5 * d² / beta    若 |d| <  beta      （二次段，∂h/∂d = d/beta）
             = |d| − 0.5 * beta    若 |d| ≥ beta      （线性段，∂h/∂d = sign(d)）
        reduction='mean'  →  (1/N) · Σ h(d)     ← 默认，value 侧用它
        reduction='none'  →  h(d) 逐元素        ← policy 侧自己聚合（见下）

    ⚠ **「线性段梯度 = ±1」只在 N=1 时成立**（P4.5-fix 更正）。mean 归约把每个
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
       ⚠ 顺带证伪 fix 简报里的另一句：它把 smooth_l1 说成「delta 取 1.0 的
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


def compute_policy_loss(policy_logits, move_t, kind,
                        label_smoothing=0.1, huber_beta=0.5):
    """按 `--policy-loss` 分派 policy 损失；返回标量张量。

    **默认 kind='ce'**（P4.5b 用户裁决）。`huber` 分支保留，仅供复现 D4/P4.5
    的实验；它不是默认的理由写在 docstring 末尾的「P4.5b」一节（梯度尺度对
    动作空间 A 的结构性依赖），别只看 `default=` 那一行就改回去。

    kind='ce'   —— **原口径原样保留**：`F.cross_entropy(logits, move_t,
        label_smoothing=...)`，与 D4 之前的调用逐字相同（默认 label_smoothing
        0.1 也照旧生效），数值行为不得改变。
    kind='huber' —— 对 policy **目标**（label-smoothed one-hot 分布）做 Huber：
        · 目标 y = (1−eps)·onehot(move_t) + eps/A（eps = label_smoothing，
          A = 动作数）—— 与 `F.cross_entropy(label_smoothing=eps)` 用的是
          **同一构造**，故 `--label-smoothing` 对两条路径语义一致；
        · 预测侧 = `softmax(policy_logits)`（模型输出是 logits，目标是概率
          分布，必须先归一化到同一值域 [0,1] 才能逐元素回归）；
        · 归约 = **类内 sum over A，再对 batch 取 mean**（P4.5-fix 修，见下）。

    ⚠ **P4.5-fix：归约口径是修过的实现 bug，不是设计选择。**
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

    ⚠⚠ **P4.5b：默认值已由 `huber` 改成 `ce`。`huber` 分支保留（可复现实验），
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
        CE                    0.3512     0.3530        **1.005**   ← 与 A 无关 ✓
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
    raise ValueError(f"--policy-loss 只接受 huber|ce，收到 {kind!r}")


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

    ⚠ **为什么 value 侧留 Huber、而 policy 侧在 P4.5b 改回 CE**（简报 §2 的裁决）：
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

    value 归属按**前缀** `'value.'` 判定而非子串 `'value' in n`：子串判定会把未来
    v21 新增的 `backbone.value_proj` 之类（非 value 头、名字里带 value）误归到 value 组。

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
    """
    no_decay_params = {p for p in model.parameters() if p.ndim == 1}
    value_decay = [p for p in model.value.parameters() if p not in no_decay_params]
    value_no_decay = [p for p in model.value.parameters() if p in no_decay_params]
    other_decay = [p for n, p in model.named_parameters()
                   if not n.startswith('value.') and p not in no_decay_params]
    other_no_decay = [p for n, p in model.named_parameters()
                      if not n.startswith('value.') and p in no_decay_params]
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

    ⚠ `value_loss_weight` 出现在**两项**里：报告口径必须与优化口径同权重，否则
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

    ⚠⚠ **只对 `weight_decay != 0` 的组求和**（`_build_param_groups` 的两组
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
    ⚠ 例外：`GradScaler.step` 只在**本地**查 `found_inf`、不做 all_reduce，
    所以一次 inf/nan 触发的 step 跳过是**单 rank** 的 —— 那一步各 rank 的 θ
    会分叉，本项在那一行就会跨 rank 不一致。稳态（无跳过）下不影响。

    代价（⚠ 分清「参数字节」与「实际流量」，两者差 ~3×）：`p.detach().pow(2).sum()`
    是「逐元素 kernel + 归约 kernel」两段，流量是 **读 θ + 写 θ² + 读 θ²**
    ≈ 3 × 参数字节。v18 参考配置 decay 参数 12,838,112 × 4 B = 51.35 MB
    ⇒ **~154 MB/step 的实际流量**，不是 51 MB（后者只是参数字节计数）。
    `p.detach()` 保证不建图、不占住反向图。设备同步是**每次调用 1 次**
    `float()`（四组里两个 decay 组，只在最后读回一次），但本函数在训练循环里
    **每个 micro-batch 都无条件跑**（不在 `if _do_stdout or _do_swanlab:` 里），
    比日志打点那 3 次同步频繁得多 —— 见 report `## Fix3 增补` §3 的 NPU 说明。
    """
    # 累加全部留在**设备上**，整个调用只做一次 `float()`（= 一次 host 同步）。
    # ⚠ 逐组 `float()` 是 2 次同步，而本函数在训练循环里**每个 micro-batch 都跑**
    #   （不在 `if _do_stdout or _do_swanlab:` 里），比日志打点的 3 次同步频繁
    #   ~`_accum_steps` × `--log-every` 倍。
    # ⚠ `sq.double()` 不是可有可无的：它让乘加保持 **float64** 精度，与旧的
    #   `wd * float(sq)`（python double 乘加）**逐位相同**；若留在 float32 上
    #   累加，wd 的 float32 舍入会让 `l2_report ∝ --weight-decay` 这条线性律
    #   只剩 ~1e-8 的相对精度（`test_l2_report_scales_with_weight_decay` 的
    #   `rel=1e-9` 会红）。代价只是每个 decay 组多两个 0 维标量 kernel。
    # ⚠ 单次读回不引入任何 θ 错位：所有 `pow(2).sum()` 的读都发生在这一行之前，
    #   仍是**同一个 θ**。
    #   `tests/test_huber_loss.py::test_l2_report_uses_decay_group_only` 用
    #   TorchDispatchMode 数 `aten::item`，把「恰好一次」钉住。
    total = None
    for group in param_groups:
        wd = float(group.get('weight_decay') or 0.0)
        if wd == 0.0:
            continue          # no_decay 组：优化器没在衰减它，报告里也不该有它
        sq = None
        for p in group['params']:
            s = p.detach().pow(2).sum()
            sq = s if sq is None else sq + s
        if sq is not None:
            term = sq.double() * wd
            total = term if total is None else total + term
    return 0.0 if total is None else float(total)


# ---- P4.6：fused AdamW 的设备策略与回退（全模块唯一构造入口）------------------
# D1：不加 `--fused` 旗标 —— 选择由设备驱动的代码级默认决定，不读环境变量、
# 不接 CLI。D6/P4.11 MindSpeed 落地后若出现 NPU fused kernel，扩展点是这个常量。
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
    下逐字相同。**不是** MindSpeed 的融合优化器：那是 D6 / P4.11 的事，本函数不碰。

    **设备策略**（A1 记录：CUDA A100 支持、910A 不支持；本地开发是 CPU）：

    * `cuda` → `try: AdamW(param_groups, fused=True)`；构造抛 `TypeError`（老
      torch 无此 kwarg）或 `RuntimeError`（无 fused kernel）→ **回退标准构造，
      异常不冒泡**；
    * `npu` / `cpu` / 其它 → **直接标准构造**（不尝试、不报错）。D6/P4.11
      MindSpeed 落地后若拿到 NPU fused kernel，扩展点 = 把后端加进
      `_FUSED_OK_BACKENDS` 并配能力探针。

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
        except (TypeError, RuntimeError) as exc:
            if logger is not None:
                logger.warning(
                    "[train] fused AdamW 在 %s 上不可用（%s: %s），回退标准实现",
                    backend, type(exc).__name__, exc)
    optimizer = torch.optim.AdamW(param_groups)
    if logger is not None:
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


# ---- NPU Linear-only 图编译（D2）------------------------------------------------
def _compile_linear_submodules(model, backend, rollback=None):
    """递归把 model 下每个 nn.Linear 子模块就地替换为 torch.compile(子模块, …)。

    返回回滚表 `[(parent_module, attr_name, original_module), ...]`，按替换顺序追加；
    传入 `rollback` 则复用调用方持有的表（main 靠它在预热失败时回滚）。

    **为什么只编 Linear、其余仍 eager**（D2）
    整模型 `torch.compile` 在 4 卡 910A 上把常驻显存从 26.7~31.1GB 顶到 OOM
    边缘，而实测收益只落在 Linear 那一小段：子模块图
    `chunk + Linear2d + silu`，invoke=1 / frames=2 / break=0，graph=10.779ms
    vs eager=14.453ms ≈ **1.34x**。卷积/注意力主体留在 eager，图缓冲与 workspace
    也就不必为整张网络付。代价是图数量变多（每 Linear 一张），换来「关掉也不会
    慢回去以外的东西」——没开图编译时代码路径与原来完全一致。

    **只替换「子」模块、不包顶层**：顶层保持裸模块，参数对象不变，于是
    `_build_param_groups` 早前抓到的 param 引用、EMA 的 shadow 张量、DDP 的
    `module.` 前缀逻辑都不受影响。⚠ 但名字会变：被包的 Linear 在
    `named_parameters()` / `state_dict()` 里多出 '_orig_mod.' 段
    （中段，见 `save_model` / `_ema_key` / `_load_model_state` 三处对齐）。

    **先收集再替换**：`torch.compile` 返回的 OptimizedModule 本身也是 nn.Module，
    而 `model.modules()` 是惰性生成器。边遍历边替换实测直接爆
    `RecursionError: maximum recursion depth exceeded`（995 层重复）——生成器走进
    新包进去的 `_orig_mod`，那个 Linear 立刻又满足 `isinstance(..., nn.Linear)`，
    于是一层套一层。先收集一遍就没有这个问题。
    """
    rollback = [] if rollback is None else rollback
    targets = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, torch.nn.Linear):
                targets.append((parent, name, child))
    if not targets:
        # 宁可抛出去走「回退 eager + 告警」，也不要静默什么都不做：开了开关却
        # 没有图，用户会以为已经在跑图模式。
        raise RuntimeError('模型里没有 nn.Linear 子模块，Linear-only 图编译无处可施')
    try:
        for parent, name, child in targets:
            # setattr 走 nn.Module.__setattr__ → 落进 _modules。nn.Sequential /
            # nn.ModuleList 的子模块名是 '0'/'1'…（字符串），同一条路径同样有效，
            # 故不必为容器类型分支。
            setattr(parent, name, torch.compile(child, backend=backend, dynamic=False))
            # 记在 setattr 之后：torch.compile / setattr 任一抛异常时这条并未生效，
            # 不该被记进回滚表。此处通常也不会抛——torch.compile 是惰性的，
            # 真正的编译错误在首次前向（预热）时才浮出来，由 main 的 except 兜。
            rollback.append((parent, name, child))
    except Exception:
        # 半途失败必须还原：只留「一半 Linear 被编译」的混合模型不会报错，
        # 只会静默变慢且极难查（逐个 verify 才知道少了哪层图）。
        _rollback_linear_submodules(rollback)
        raise
    return rollback


def _rollback_linear_submodules(rollback) -> int:
    """按回滚表把 (parent, name, orig) 逐条写回，返回还原条数。

    **幂等**：同一条记录重复还原结果相同（写回的是同一个 orig 对象），
    所以「替换中途失败」（`_compile_linear_submodules` 内部已还原一次）与
    「预热失败」（main 的 except 再还原一次）两条路径都调它也不会互相踩。
    """
    n = 0
    for parent, name, orig in rollback:
        setattr(parent, name, orig)
        n += 1
    return n


def _load_model_state(model, ckpt, logger) -> None:
    """把 state_dict 灌进 model，自动对齐 torch.compile 引入的 '_orig_mod.' 段。

    两种 compile 形态把这一段放在**不同位置**：整模型 compile（CUDA `--compile 1`）
    插在开头，Linear-only compile（`--npu-graph-compile 1`）插在路径中段。
    因此这里不按位置处理，而是**双向规范化**：先把目标模型与 checkpoint 的键
    都归一到「未编译布局」，再按目标模型当前的真实键写回去。

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
    ap.add_argument('--attention-dropout', type=float, default=0.1)
    ap.add_argument('--attn-mode', default='global',
                    choices=['global', 'window', 'axial', 'sparse', 'window_global'],
                    help='注意力计算模式: global=全配对, window=块状窗口, '
                         'sparse=窗口+全局token, window_global=块状窗口+全局token(手写math)')
    ap.add_argument('--attn-window', type=int, default=7, help='window 模式窗口边长')
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
    ap.add_argument('--prefetch-workers', type=int, default=8,
                    help='数据预取线程数：每个 batch 切块并行造特征并与 GPU 计算重叠；'
                         '<=1 关闭预取（回退同步取样）。')
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
                         '补偿）。⚠ 换 --value-loss 后 policy/value 的梯度量级'
                         '关系已变，见 report `## Fix 增补`：policy 默认走 huber，'
                         '其梯度天然比 value 弱 ~A 倍，修掉归约 bug 后实测仍差 '
                         '约 250~360:1，此参数不足以单独补平。')
    # ---- D4（SFT 侧）：损失口径三参数 ------------------------------------
    # 只加这三个（D1：v21 不新增其他 CLI 参数）。RL 侧的对应拆分在 P3-C/P3-D，
    # scripts/selfplay_train.py 不在本任务范围。
    ap.add_argument('--policy-loss', default='ce',
                    choices=['huber', 'ce'],
                    help='policy 损失：ce=原交叉熵口径（数值行为与 D4 之前逐位一致）'
                         '，**默认**；huber=对 label-smoothed one-hot 目标做 Huber'
                         '(smooth L1, beta=--huber-beta，归约=类内 sum over A + '
                         'batch mean)，保留仅供复现实验。'
                         '⚠ 默认**不是** huber（P4.5b 用户裁决）：定义在概率上的损失，'
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
                         'BCE 分支已删）。⚠ value 侧留 Huber 正是为了与 policy 侧'
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
    ap.add_argument('--use-checkpoint', type=int, default=0, choices=[0, 1],
                    help='用 gradient checkpointing 减少显存占用（约省 50%%，训练慢 ~30%%）(0=关闭, 1=开启)')
    ap.add_argument('--use-ema', type=int, default=0, choices=[0, 1],
                    help='启用 EMA（指数移动平均）权重，eval/save 时用 shadow 权重，提升 1-3%% accuracy (0=关闭, 1=开启)')
    ap.add_argument('--gradient-accumulation-steps', type=int, default=1,
                    help='梯度累积步数（模拟更大 batch size，效果等同于 batch_size * N）')
    ap.add_argument('--scaler-init-scale', type=float, default=0.0,
                    help='GradScaler 初始缩放值（0=用 PyTorch 默认 65536）。'
                         '默认 65536 需要靠减半向下搜索平衡点，每次溢出白扔一个 '
                         'batch；实测本模型在 4 卡 910A 上平衡于 512~2048，'
                         '故建议直接给 1024 起步。设 --scaler-growth-interval 0 '
                         '可关闭自动回涨，避免震荡反复偷步')
    ap.add_argument('--scaler-growth-interval', type=int, default=0,
                    help='连续多少个无溢出 step 后把缩放值翻倍'
                         '（0=用 PyTorch 默认 2000；设一个大值如 100000 即'
                         '相当于关闭回涨，让缩放值稳定在 init-scale 附近）')
    ap.add_argument('--npu-graph-compile', type=int, default=0, choices=[0, 1],
                    help='NPU 上用 TorchAir 图编译替代 eager（受控实验，D2 Linear-only：'
                         '只递归编译各 nn.Linear 子模块，其余仍 eager，不包整张网络）。'
                         'NPU 上 inductor 不可用，必须显式传 torchair backend；'
                         'torch_npu 须先于 torchair 导入，否则图模式会静默降级为'
                         'eager 而不报错。失败自动整表回滚并回退 eager。'
                         '⚠ 显存风险：本项目 4 卡 910A 常驻已达 26.7~31.1GB/32GB，'
                         '图模式的 workspace 与图缓冲可能再炸；且实测环境为 '
                         'torch 2.1.0 / torch_npu 2.1.0.post3 / CANN 8.0.RC1，'
                         '属 2023 年代组合，功能成熟度存疑。故默认关闭，'
                         '建议先短跑验证(0=关闭, 1=开启)')
    ap.add_argument('--compile', type=int, default=0, choices=[0, 1],
                    help='用 torch.compile 融合算子（GPU 上约 20-40%% 提速，首次迭代较慢）(0=关闭, 1=开启)')
    ap.add_argument('--compile-mode', default='default',
                    choices=['default', 'max-autotune', 'reduce-overhead'],
                    help='torch.compile 模式: default=常规融合, max-autotune=A100 上进一步 '
                         '自动调优提速（编译更久）, reduce-overhead=小 batch 低开销')
    ap.add_argument('--flash-attn', type=int, default=0, choices=[0, 1],
                    help='启用 flash-attn 独立库（A100 上最快，需 pip install flash-attn）(0=关闭, 1=开启)')
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
                args.num_heads, args.num_attention_layers, args.attention_dropout, args.compile)
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
    # `init_process_group` / `torch.npu.set_device` 之后，4 个 worker 会各自
    # 继承一份父进程的 CANN 设备上下文与显存映射 ⇒ 每卡 6.3 GB 的训练被旁挂到
    # 4×6 GB，实测 HBM 94% 而 AICore 0%，紧接着就是 OOM。数据集加载是纯
    # numpy（与 rank 无关），提前无语义影响；反向顺序（dist 初始化后再 fork）
    # 才是 HCCL 的危险方向，提前 fork 是安全的那一侧。
    dataset = load_from_path(args.data, args.board_size, args.max_games_per_tgz)
    pf = None
    if args.prefetch_workers > 1:
        pf = _BatchPrefetcher(dataset, num_workers=args.prefetch_workers,
                              prefetch=args.prefetch_depth)
        logger.info("[data] 预取器已启用（在设备初始化之前 fork）| workers=%d depth=%d",
                    args.prefetch_workers, args.prefetch_depth)
    else:
        logger.info("[data] 预取器已关闭（--prefetch-workers=%d ≤ 1）",
                    args.prefetch_workers)

    # ---- 分布式训练：设备由 LOCAL_RANK 决定，忽略 --device 卡号 ----
    # 后端选择：NPU 走 hccl，CUDA 走 nccl。多卡前必须 init_process_group，
    # 否则后续 .to(device) / DDP 包裹会失败或各卡不互通。
    if is_dist:
        _dist_backend = (args.device.split(':')[0]
                         if args.device not in ('auto', '') else
                         ('npu' if npu_is_available() else 'cuda'))
        # ⚠ 必须在 init_process_group **之前**：DETAIL 是在建域那一刻给每个 PG 套
        # 一层一致性检查 wrapper（每次 collective 前一发 monitored_barrier），NPU
        # 上未验证过其开销 ⇒ 降为 OFF（该代价与用哪种包裹层无关，换轨后依旧存在）。
        # 见 _downgrade_npu_dist_debug 的 docstring。
        if _dist_backend == 'npu':
            _downgrade_npu_dist_debug(logger)
        if _dist_backend == 'npu':
            import torch_npu  # noqa: F401 — 注册 HCCL 后端
            dist.init_process_group('hccl')
            torch.npu.set_device(local_rank)
        else:
            dist.init_process_group('nccl')
            torch.cuda.set_device(local_rank)
        device = f'{_dist_backend}:{local_rank}'
        # 通信域是**惰性**创建的（上面两行只登记后端），所以主动试一发 all_reduce：
        # 否则通信建不起来要等到第一个 batch 的前向才炸，且报成 HCCL 通用错误
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

    use_amp = args.use_amp == 1 or (device.split(':')[0] in ('cuda', 'npu'))

    # ---- 多后端自适应路径（CUDA / NPU / CPU）----
    # 各后端能力差异很大，逐后端决定：
    #   - amp_dtype:       A100/A800/H100(sm_80+) -> bfloat16（原生支持）
    #                     Ascend 910B/910A -> float16（NPU autocast 仅支持 FP16）
    #                     V100(sm_70, Volta) -> float16（无 bf16）
    #   - use_scaler:      BF16 下关闭 GradScaler（不下溢）；FP16 下开启
    #   - use_channels_last: A100 卷积走 NHWC 更快；NPU/CPU 收益有限默认关
    #   - sdpa_force_math: NPU 上 FlashAttention 后端不稳（CANN SDPA 与 CUDA 不同），
    #                     强制走手写 math 注意力最稳；V100 同样强制 math；A100 走 Flash
    #   - compile_disable_sparse: 所有后端统一禁用——unfold 产生 (B, Hh*d, N, ws²) 巨型
    #                      中间张量，inductor freezing 常量折叠会以 fp32 物化
    #                      (B,N,Hh,ws²,d)（batch512 下单个 4.3GB）直接编译期 OOM；
    #                      稀疏/窗口注意力走 eager+autocast（V100 验证过的稳定路径），
    #                      编译图仅覆盖卷积/线性/FFN。NPU 上 inductor 本身不可用
    amp_dtype = torch.float16
    use_scaler = use_amp
    use_channels_last = False
    sdpa_force_math = True
    compile_disable_sparse = True
    gpu_name = 'N/A'
    compute_cap = (0, 0)
    _backend = device.split(':')[0]
    # 具体卡号（device 形如 'cuda:1' / 'npu:0' / 'cpu'），无索引时默认 0
    try:
        _dev_idx = int(device.split(':')[1]) if ':' in device else 0
    except ValueError:
        _dev_idx = 0
    if _backend == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # A100+ 上开启 TF32：未被 autocast 覆盖的 fp32 matmul/conv 走 TF32 tensor core
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
            logger.info("[device] %s (sm_%d%d) | 启用 A100 路径: BF16 + FlashAttn + "
                        "channels_last + compile(卷积/线性/FFN)", gpu_name, *compute_cap)
            # 尝试加载 flash-attn
            if args.flash_attn == 1:
                from src.networks import backbone as _backbone
                fa_ok, fa_msg = _backbone.set_flash_attn(True)
                if fa_ok:
                    logger.info("[env] flash-attn %s", fa_msg)
                else:
                    logger.warning("[env] flash-attn %s，回退内置 SDPA", fa_msg)
        else:
            # V100 等老卡：保守路径（与原行为一致）
            amp_dtype = torch.float16
            use_scaler = use_amp
            use_channels_last = False
            sdpa_force_math = True
            compile_disable_sparse = True
            logger.info("[device] %s (sm_%d%d) | 走保守路径: FP16 + 手写 math 注意力 + "
                        "稀疏注意力禁用编译", gpu_name, *compute_cap)
    elif _backend == 'npu' and npu_is_available():
        # Ascend 910B / 910A：CANN + torch_npu 后端
        gpu_name = npu_get_device_name(_dev_idx)
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        # 关键：按芯片型号选精度。910B/910Pro 原生 BF16；910A 无 BF16，必须走
        # FP16 + GradScaler（与 V100 路径一致）。CANN 上 FlashAttention 后端不稳，
        # 强制手写 math 注意力；channels_last 对 NPU 卷积无明确收益，关闭；
        # torch.compile(inductor) 在 NPU 不可用，禁用。
        if '910B' in gpu_name or '910Pro' in gpu_name or '910-2' in gpu_name:
            amp_dtype = torch.float16  # NPU autocast 仅支持 FP16
            use_scaler = True  # FP16 需要 GradScaler 防下溢
            logger.info("[device] %s (NPU/CANN) | 910B 路径: FP16 + GradScaler + 手写 math "
                        "注意力 + 禁用 torch.compile(inductor)", gpu_name)
        else:
            amp_dtype = torch.float16
            use_scaler = use_amp  # 910A 无 BF16，FP16 必须开 GradScaler
            logger.info("[device] %s (NPU/CANN) | 910A 路径: FP16 + GradScaler + 手写 math "
                        "注意力 + 禁用 torch.compile(inductor)", gpu_name)
        use_channels_last = False
        sdpa_force_math = True
        compile_disable_sparse = True
        if args.compile == 1:
            logger.warning("[device] NPU 上 torch.compile(inductor) 不可用，已忽略 --compile；"
                           "如需图编译请改用 --npu-graph-compile 1（TorchAir 后端）。")
            args.compile = False
    else:
        # CPU 或其他：纯 FP32，无 AMP、无 channels_last
        amp_dtype = torch.float32
        use_scaler = False
        use_channels_last = False
        sdpa_force_math = True
        compile_disable_sparse = True
        logger.info("[device] CPU | 走 FP32 路径（无 AMP/编译）")

    # NPU 图编译（TorchAir）开关。与 --compile 互斥：NPU 上 inductor 不可用，
    # 两条路径都需要显式指定 backend，不能同时开。
    _npu_graph = (args.npu_graph_compile == 1 and _backend == 'npu')
    if args.npu_graph_compile == 1 and _backend != 'npu':
        logger.warning("[device] --npu-graph-compile 仅对 NPU 生效，当前后端为 %s，"
                       "已忽略。", _backend)

    # 把注意力后端/编译开关透传给 backbone 模块（所有分支统一设置）
    from src.networks import backbone as _backbone
    _backbone.set_sdpa_force_math(sdpa_force_math)
    _backbone.set_compile_disable_sparse(compile_disable_sparse)
    # 注意力 query 分块（2026-10-01）：math 路径下整条 (B,Hh,N,N) 分数矩阵
    # @N=361/4head/fp16 在 B=1000 时 0.97 GiB 一份、softmax+dropout 再各一份。
    # softmax 沿 key 轴 ⇒ 按 query 切块**数学精确**，峰值 ∝ chunk（默认 64 ⇒ 5.6×）。
    # 只加在 math 分支（NPU 恒走 math），SDPA/flash 路径不受影响。
    _attn_chunk = int(os.environ.get('GOAI_ATTN_QUERY_CHUNK', '64') or 0)
    _backbone.set_attn_query_chunk(_attn_chunk)

    # flash-attn 独立库启用决策：仅「Ampere+ CUDA 且走非 math 路径」时尝试加载。
    # 加载失败自动回退内置 SDPA，不影响训练启动。
    # 环境变量 GOAI_FLASH=0 可强制禁用（A/B 实测用：稀疏注意力分块 seq 仅 ~30，
    # flash 小 kernel 密集发射在部分配置下比手写 math 更慢，需实测决定）。
    _flash_wanted = os.environ.get('GOAI_FLASH', '1') != '0'
    if _flash_wanted and _backend == 'cuda' and compute_cap >= (8, 0) and not sdpa_force_math:
        fa_ok, fa_msg = _backbone.set_flash_attn(True)
        if fa_ok:
            logger.info("[env] 注意力内核: flash-attn %s（优先于内置 SDPA）", fa_msg)
        else:
            logger.info("[env] 注意力内核: 内置 SDPA（flash-attn %s）", fa_msg)
    else:
        _backbone.set_flash_attn(False)
        if not _flash_wanted:
            logger.info("[env] 注意力内核: 手写 math（GOAI_FLASH=0 已禁用 flash）")
        else:
            logger.info("[env] 注意力内核: %s",
                        "手写 math（%s 不支持 Flash）" % (_backend.upper(),)
                        if sdpa_force_math else "内置 SDPA")
            logger.info("[env] 注意力 query 分块: %s",
                        "关闭" if _attn_chunk <= 0 else
                        "%d（峰值 ∝ chunk；eval 逐位不变，训练态 dropout 取样位置变）"
                        % _attn_chunk)

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
    logger.info("[data] 总样本数=%d | 训练=%d | 验证=%d", n, len(train_idx), len(eval_idx))

    # 分布式：每张卡用 DistributedSampler 取到不相交的训练分片（会自动 pad 到
    # 能被 world_size 整除），各卡步数因此一致，避免 DDP 在 barrier 处互相等待。
    if is_dist:
        train_sampler = torch.utils.data.DistributedSampler(
            torch.arange(len(train_idx)), num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    # ---------------------------------------------------------------- D1 v2 ----
    # 训练结构**唯一** = v21（结构硬编码在 src.networks.alphanet.V21_CFG）。
    # 旧结构 flag（--arch / --backbone-channels / --res-blocks / --policy-layers …
    # 与下面这一整串）**全部归档：不参与建网**，仍被 argparse 接受只为旧 shell
    # 不必改命令行（忽略而非报错，是 D1 的硬约束 C11；运行日志里那句
    # 「结构参数已归档」就是这里）。
    # `--use-checkpoint` 同样归档：v21 的检查点开关 = V21_CFG['grad_checkpoint']
    # （用户裁决：ResBlocks 必开）∧「未开图编译」，见下。
    # 没有 `arch=='v21'` 分支、没有新增任何 CLI（D1）。
    _v21_gc = V21_CFG['grad_checkpoint']
    if args.compile == 1 or args.npu_graph_compile == 1:
        # grad checkpointing 与 torch.compile / TorchAir 图编译互斥
        # （P4.6b §8.2④：训练态 GC 会在第一次前向撞断言）。两者都要是
        # **显式**决策：这里让图编译赢、检查点让位并打 warning，绝不静默
        # 丢掉任何一边（显存会回升，日志必须能看出原因）。
        _v21_gc = 0
        logger.warning(
            "[model] compile/npu-graph-compile=1 ⇒ 本次运行关闭 gradient "
            "checkpointing（V21_CFG 的 %d 与图编译互斥，显存占用回升）",
            V21_CFG['grad_checkpoint'])
    elif args.use_checkpoint == 0:
        logger.info("[model] --use-checkpoint 已归档（D1）：v21 的检查点开关由 "
                    "V21_CFG[%r]=%d 决定，本次启用（如需关闭请用 --compile 1）",
                    'grad_checkpoint', _v21_gc)
    model = build_v21_net(
        in_channels=V21_CFG['in_channels'],
        action_size=args.board_size * args.board_size + 1,  # +1 为 pass 类别
        attention_dropout=args.attention_dropout,           # 行为参数，仍生效
        grad_checkpoint=_v21_gc,
    ).to(device)
    # from-scratch 起手必须把 rank0 的初始权重广播给其余 rank（2026-10-01）。
    # ⚠ 位置是硬要求：**必须在 EMA 构造（本文件下方 `EMA(model, ...)`）之前** ——
    # EMA 在构造时就把参数 `clone()` 进 `shadow`，放晚了 shadow 会持有广播前的
    # 随机权重，之后每个 step 的 `ema.update()` 都往这个陈旧 shadow 上混。
    # 也必须在 DDP 包裹之前：先建 EMA 再包裹，EMA 持有的引用就正好是 DDP 的
    # `module`，键空间一致（见下方包裹点上方的注释）。
    _sync_init_weights_from_rank0(model, logger)
    _assert_init_weights_identical(model, logger)
    n_params = sum(p.numel() for p in model.parameters())
    # 逐段开关也打出来：「grad_checkpoint=1」只说明**总开关**，看不出哪几段真的在
    # 走检查点（per-kind 默认可以不同；且训练态闸门还要求 self.training +
    # grad enabled）。这段日志的用处是让「GC 到底生效没有」不必翻代码，也不必
    # 在云端日志里靠猜 —— 4×910A 首跑要看的就是它。
    _gc_kinds = getattr(getattr(model, 'backbone', None),
                        'grad_checkpointing_kinds', None)
    _gc_kinds = _gc_kinds() if callable(_gc_kinds) else {}
    logger.info("[model] v21 (V21_CFG) 参数量=%.2fM | grad_checkpoint=%d | 设备=%s",
                n_params / 1e6, _v21_gc, device)
    logger.info("[model] GC 逐段开关=%s | 生效还需 training 态+grad enabled"
                "（eval/推理恒不检查点，零开销）", _gc_kinds or 'n/a')

    # A100 上把卷积型特征（N,C,H,W）转 channels_last(NHWC)，卷积算子走更快内存布局。
    # 输入 state 也需同步转格式（见训练/评估循环），故这里仅转换模型权重布局。
    if use_channels_last and _backend == 'cuda':
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        logger.info("[model] 已启用 channels_last (NHWC) 内存格式（A100 卷积加速）")

    # 参数组：value head 独立 LR（参数量小，需要更高学习率补偿梯度不足）
    # 排除 bias / BatchNorm / LayerNorm 参数的 weight decay（标准做法）
    _opt_groups = _build_param_groups(model, args)
    # A1: CUDA(A100) 启用 fused AdamW（单 kernel 融合 param 更新，省启动开销）；
    # NPU/CPU 走默认实现（910A 不支持 fused）。P4.6 起这段决策收进 build_adamw：
    # 设备策略、构造、回退契约（fused 不可用 ⇒ 标准实现，bit-for-bit 等于旧路径）
    # 集中在唯一入口，全模块不再有第二处 AdamW 构造点。
    optimizer, _opt_mode = build_adamw(_opt_groups, device, logger)
    # BF16 后端（A100/NPU）下 use_scaler=False（BF16 不下溢，省去 loss scaling 的额外同步）；
    # V100/FP16 下开启 GradScaler。按设备选择 GradScaler 实现。
    # 缩放值策略可配：从 65536 起步要靠减半向下搜索平衡点，每次溢出都白扔一个
    # batch；已知平衡点后直接给 init_scale 并关掉回涨，可消除这段浪费与后续震荡。
    _scaler_kwargs = {}
    if args.scaler_init_scale and args.scaler_init_scale > 0:
        _scaler_kwargs['init_scale'] = float(args.scaler_init_scale)
    if args.scaler_growth_interval and args.scaler_growth_interval > 0:
        _scaler_kwargs['growth_interval'] = int(args.scaler_growth_interval)
    if _backend == 'npu':
        scaler = npu_grad_scaler(enabled=use_scaler, **_scaler_kwargs)
    else:
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
                ema.shadow = tstate['ema_shadow']
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

    # torch.compile 融合算子（GPU 上约 20-40%% 提速）。必须在 resume 加载之后再做。
    # NPU TorchAir 图编译（受控实验，D2 Linear-only）。与 --compile 互斥：NPU 上
    # inductor 不可用，两条路径都必须显式指定 backend，不能同时开。位置同样在
    # resume 加载之后（--compile 会把顶层包成 OptimizedModule，先编译再灌权重
    # 会因键名不匹配而失败；Linear-only 形态灌得进去，但一样放后面更不容易忘）。
    #
    # 背景：4 卡 910A 实测 NPU 报 Aicore Usage Rate 85~100%，而实际只有
    # 659 samples/s/卡。AICore 一直「忙」但产出极低，是 eager 小算子的典型形态。
    # 图编译是唯一可能带来量级提升的路径。
    #
    # 三条硬约束（都有测试守护）：
    #   1. torch_npu 必须先于 torchair 导入——否则图模式**静默降级为 eager**
    #      且不报错，等于白开还白付预热代价；
    #   2. 不传 mode/options——昇腾 NPU 不支持，且 reduce-overhead 正是 A100
    #      上 OOM 的元凶（CUDA Graphs 私有内存池不归还），NPU 上同样避开；
    #   3. 任何失败都回退 eager——常驻已 26.7~31.1GB/32GB，图模式的 workspace
    #      与图缓冲极易再炸，绝不能让整个训练崩掉。
    #
    # D2：**不**再 `torch.compile(model, …)` 包整张网络，改成逐个 nn.Linear
    # 递归替换（`_compile_linear_submodules`）。理由、实测数字与代价见该函数
    # docstring。回滚靠回滚表逐条写回，不再靠 `getattr(model, '_orig_mod', model)`
    # 剥顶层包装——顶层现在压根没被包，这条已随 D2 作废。
    if _npu_graph:
        _npu_rollback = []
        try:
            import torch_npu  # noqa: F401  顺序要求：必须先于 torchair
            import torchair
            _tacfg = torchair.CompilerConfig()
            _npui = torchair.get_npu_backend(compiler_config=_tacfg)
            # 逐 Linear 就地替换；回滚表由本函数持有，预热失败时整表写回
            _npu_rollback = _compile_linear_submodules(model, _npui, _npu_rollback)
            # 预热前向必须与真实训练一致地包 autocast：FP32 输入干灌会报
            # flash-attn 只接受 fp16/bf16，导致图编译被误判为不可用而回退 eager，
            # 且这个误判极难排查。torch.compile 是惰性的，编译错误在这里才浮出来。
            with torch.no_grad(), maybe_autocast(device, amp_dtype):
                # 通道数走 V21_CFG 常量（C9 / P4.4）：D1 之后这里建的永远是 v21，
                # 硬编码 12 会让图编译预热在 stem 上直接形状错 → 整条编译路径
                # 静默回退 eager。
                _dummy = torch.zeros(1, V21_CFG['in_channels'], args.board_size,
                                      args.board_size, device=device)
                model(_dummy)
            logger.info("[train] NPU TorchAir Linear-only 图编译已启用（已编译 %d 个 "
                        "nn.Linear，其余子模块仍 eager）", len(_npu_rollback))
        except Exception as e:  # noqa: BLE001
            # 真正回退 eager：按回滚表把每个 nn.Linear 换回原对象。幂等，
            # 「替换中途失败」（helper 内部已还原过一次）与「预热失败」都适用。
            _restored = _rollback_linear_submodules(_npu_rollback)
            logger.warning("[train] NPU TorchAir 图编译失败，已还原 %d 个 nn.Linear 子模块，"
                           "回退 eager: %s", _restored, e)
            if isinstance(e, ImportError):
                logger.warning("[train] 常见原因：torchair 随 torch_npu 附带，"
                               "不可单独 pip install；同时需确认 CANN 的 ATC/ACL "
                               "路径可见（镜像内通常已 source set_env.sh）")
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
            try:
                model = torch.compile(model, dynamic=False, mode=args.compile_mode)
                # 预热前向必须与真实训练一致地包 autocast：flash-attn 只接受 fp16/bf16，
                # FP32 输入直灌会报 "FlashAttention only support fp16 and bf16 data
                # type"，导致 compile 被误判为不可用而回退 eager。
                with torch.no_grad(), maybe_autocast(device, amp_dtype):
                    # 同上：预热输入通道 = V21_CFG['in_channels']（C9 / P4.4）
                    dummy = torch.zeros(1, V21_CFG['in_channels'], args.board_size,
                                        args.board_size, device=device)
                    model(dummy)
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
    # 为什么现在用 DDP（2026-10-01 换轨；**历史：换轨前用的是 FSDP1，已退役**）
    # ------------------------------------------------------------------
    # 触发事件：4 卡 910A 训练崩在 `ema.update()` 的
    # `KeyError: 'backbone.stem_bn._fsdp_wrapped_module.weight'`。那套分片式
    # 包裹层会把**内部**模块就地换成 wrapper（wrapper 又把自己的 `_fsdp_wrapped_module`
    # 注册进父模块的 `_modules`），于是「包裹前建好的 EMA shadow」与
    # 「包裹后 named_parameters() 遍历出来的键」不再是同一个键空间 ⇒ 每步都
    # KeyError。DDP 只在**顶层**加一层 wrapper（键只是多一个 `module.` 前缀），
    # 内部模块树原样不动 ⇒ 这整类崩溃消失，不需要任何针对它的补丁。
    #
    # 两个数字（9,067,443 参数 = fp32 36.3 MB / 32 GiB 卡）：
    #   1. 显存：FSDP1 每 rank 约 145 MB（参数分片 + 梯度分片 + Adam 两矩分片），
    #      DDP 每 rank 约 36.3 MB × 4（参数 + 梯度 + Adam 两矩各一份全量），
    #      差约 109 MB = 0.33% 的卡。参数本来就装得下，分片省下的这点余量
    #      不值得拿正确性风险换。
    #   2. 通信：FSDP1 每步是「16 个分片单元 × 2 次 collective」（前向 all-gather
    #      参数、反向 reduce-scatter 梯度），DDP 每次**反向**只有 1 次梯度
    #      all-reduce +（`broadcast_buffers=True`）1 次 buffer broadcast。
    #      ⚠⚠ **「每步 1 次」是错的说法，本文件全无 `no_sync()`**（见本文件
    #      `_compute_l2_report` docstring 里那条同源的说明）：每个 micro-batch 都
    #      `backward()`，梯度累积只在 `_accum_steps` 满了才 `optimizer.step()`，
    #      而 DDP 的梯度 all-reduce 挂在**每一次 backward 的收尾**上 ⇒ v21 的
    #      `GRAD_ACCUM=2` 下**每个 optimizer step 是 2 次梯度 all-reduce + 2 次
    #      buffer broadcast**（payload ≈ 2 × 36.3 MB ≈ 72.5 MB）。数量仍是
    #      「16 个分片单元 × 2」的零头，通信量小到不像瓶颈，而本仓库的实际瓶颈
    #      在算子（见 `[profile]` 日志）。
    #
    # 为什么 Task 1 的两个同步函数必须留在包裹点**之前**（顺序的硬要求）：
    # `DistributedDataParallel.__init__` 在**它自己构造时**（`_ddp_init_helper` →
    # `_sync_module_states`）把 rank0 的 params/buffers 广播出去，而构造发生在
    # **EMA 构造之后**。所以若依赖 DDP 自带的同步：rank0 的 EMA shadow = rank0
    # 自己的随机权重（正确），rank1~3 的 shadow = 各自被丢弃的随机权重（陈旧）
    # ⇒ `ema.update()` 每步都把正确权重混进陈旧 shadow ⇒ **EMA 跨 rank 发散**，
    # 且不报错。`main()` 里 `_sync_init_weights_from_rank0` /
    # `_assert_init_weights_identical` 在 `.to(device)` 之后、EMA 之前，恰好堵住它。
    #
    # 构造参数**只有** `device_ids`，其余全默认（不新增任何 CLI flag）：
    #   · `find_unused_parameters=False`（默认）：两个头（policy/value）每个 step
    #     都参与 loss ⇒ 所有参数都有梯度。⚠ 这是**隐含前提**：将来若出现「某个头
    #     不参与 loss」的分支，DDP 会抛
    #     `Expected to have finished reduction in the prior iteration`
    #     —— 好在它是**响亮**地失败，不会安静地错。
    #   · `broadcast_buffers=True`（默认）：BN 的 `running_mean`/`running_var`
    #     跨卡一致靠它；上一代分片式包裹层默认也是 True ⇒ 行为不变。
    #   · `gradient_as_bucket_view=False`（默认）：本文件用
#     `optimizer.zero_grad(set_to_none=True)`，bucket view 的别名每轮被销毁，
#     省不掉拷贝，收益仅 ~36.27 MB/rank（= **全部梯度**的大小，torch 对该开关
#     的定义就是"saved memory size will be equal to the total gradients
#     size"；占 32 GiB 的 0.106%）；真正的风险是混用 view / 非 view 的 grad
#     状态触发 `Expected to mark a variable ready only once`。收益配不上这类风险。
    #   · `static_graph=False`（默认）：打开会禁止「iteration 边界内参数集合
    #     变化」，收益未验证。
    if is_dist:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    if is_main:
        logger.info("[train] 开始训练 | steps/epoch=%d | 总 steps≈%d | warmup=%d",
                    n_batches, total_steps, warmup_steps)

    # 预取器 pf 已在**设备初始化之前**构造（见上方「数据集 + 预取 worker」段：
    # fork 晚于 set_device 会让每个 worker 继承 CANN 上下文，4 卡实测每卡凭空
    # 多占 ~24 GiB ⇒ OOM）。此处刻意不再构造，避免顺序被无意改回去。
    assert (pf is not None) == (args.prefetch_workers > 1), \
        '预取器构造与 workers 设置不一致：构造顺序被改动了？'

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
        # 内核级剖析（诊断用）：GOAI_PROFILE=<step> 从该 step 起 profiling 50 个
        # step，结束打印 top CUDA kernel 耗时表，用于定位 740ms/step 的去向。
        _prof_at = int(os.environ.get('GOAI_PROFILE', '0') or 0)
        _prof_ctx = None
        _accum_steps = args.gradient_accumulation_steps
        # ---- 分段计时（纯 CPU 侧观测）----
        # 动机：4 卡 910A 实测 4.25 s/step，扣除 eval（实测仅 0.2%）后
        # 约 96% 是黑盒，无法判断瓶颈在取数 / 算子 / 通信 / 保存。
        # 绝不在此插 synchronize()：那会打断预取与双缓冲流水，反而更慢。
        # 代价是 t_comp 只反映 CPU 侧发射时间、不含 NPU 实际执行；
        # 判读靠「各段之和 vs elapsed」的差额。
        # 重置放在打点处，故某步的 save/eval（发生在其打点之后）计入
        # 下一个区间 —— 这与墙钟口径一致。
        _t_data = _t_comp = _t_save = _t_eval = 0.0
        _t_data_max = 0.0
        _n_timed = 0
        _n_skipped = 0
        for i in range(n_batches):
            try:
                if i % _accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                if _prof_at > 0 and step == _prof_at and _prof_ctx is None:
                    try:
                        from torch.profiler import (profile, ProfilerActivity)
                        _prof_ctx = profile(
                            activities=[ProfilerActivity.CPU,
                                        ProfilerActivity.CUDA])
                        _prof_ctx.__enter__()
                        logger.info("[profile] 已开始内核剖析（50 steps）...")
                    except Exception as pe:  # noqa: BLE001
                        logger.warning("[profile] 不可用: %s", pe)
                        _prof_at = 0
                _t_data0 = time.perf_counter()
                if pf is not None:
                    # P0: 先提交下一个 batch，再取当前 batch（给 worker 更多预计算时间）
                    nxt = i + args.prefetch_depth
                    if nxt < n_batches:
                        pf.submit(perm[nxt * bs:(nxt + 1) * bs])
                    states_np, moves_np, values_np = pf.next()
                    if _backend == 'cuda':
                        # pin_memory 需要 contiguous 且为 CPU 内存
                        moves_np = np.ascontiguousarray(moves_np)
                        values_np = np.ascontiguousarray(values_np)
                        # ⚠ 只有全精度（autocast 关着）才升 fp32。AMP 下权重会被
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
                        # 同上：全精度才升 fp32。NPU 走 AMP（amp_dtype=fp16）⇒ 保持
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
                    state, move_t, value_t = dataset.sample_batch(sel, device)
                    # A100 上转 NHWC 以匹配模型 channels_last 布局，卷积更快
                    if use_channels_last:
                        state = state.to(memory_format=torch.channels_last)
                _dt = time.perf_counter() - _t_data0
                _t_data += _dt
                if _dt > _t_data_max:
                    _t_data_max = _dt
                _t_comp0 = time.perf_counter()
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
                        huber_beta=args.huber_beta)
                    value_loss = compute_value_loss(
                        value_logit, value_t, args.value_loss,
                        huber_beta=args.huber_beta)
                    # ---- 被 backward 的量：不含 c‖θ‖² ------------------------
                    # 为什么 opt_loss **不含** L2 项：正则走的是 AdamW 的**解耦**
                    # weight decay（θ ← θ − lr·wd·θ，发生在参数更新里），按构造就
                    # 不在梯度里。若把 c‖θ‖² 加进来，优化目标会从「解耦衰减」变成
                    # 「耦合 L2 + 再次解耦衰减」的双重正则，训练行为立刻变化且不报错
                    # —— 这是 P4.5b 最容易踩的坑，故 opt_loss / log_loss 分开命名。
                    #
                    # ---- 被写进日志 `loss` 键的量：加上 c‖θ‖² ----------------
                    # 为什么 log_loss 与 opt_loss **不相等**：用户裁决的总损失口径
                    # 是 L = L_policy + L_value + c‖θ‖²，要让恒等式在日志上字面成立。
                    # l2_report 是**报告口径**的量（compute_l2_report：从
                    # optimizer.param_groups 读回真实的 weight_decay，只覆盖
                    # weight_decay != 0 的组，与优化器实际衰减同一批参数），在本次
                    # optimizer.step() **之前**算，故与两个损失项取自同一个 θ。
                    # 恒等式在一次 fp32 加法的精度内成立；⚠ 反过来用
                    # `log_loss − policy − value` 反推 l2_report 时要记得 fp32
                    # 舍入：两项都是 O(1)~O(10)，差值只剩 ~1e-7 的绝对精度
                    # （test_log_loss_identity 按这个容差断言）。
                    # ⚠ 但**真正的**精度上限是 stdout 的 `%.4f`（下面 logger.info
                    # 里 loss/p/v 都是 4 位小数 ⇒ 量化步长 1e-4，对初值
                    # l2_report=0.588 而言是 0.017%），不是 fp32 舍入。且
                    # `log_loss − policy − value` 只在 `--value-loss-weight == 1`
                    # 时等于 l2_report；w≠1 时它是 w·value。
                    # ⚠ 读 loss 曲线的人必须知道：log_loss **不是**被优化的目标。
                    l2_report = compute_l2_report(optimizer.param_groups)
                    opt_loss, log_loss = compose_losses(
                        policy_loss, value_loss, args.value_loss_weight, l2_report)
                scaler.scale(opt_loss / _accum_steps).backward()

                _t_comp += time.perf_counter() - _t_comp0
                _n_timed += 1
                if (i + 1) % _accum_steps == 0 or (i + 1) == n_batches:
                    # 缩放值下降 == 本步因 inf/nan 被 GradScaler 跳过，那一整个
                    # batch 的数据就此白扔。实测 4 卡 910A 在 step~1770 出现
                    # 16384→8192→4096→2048 的雪崩 + 大量 Skipping step。
                    #
                    # 注意：这里不把 clip_grad_norm_ 挪到 unscale_ 之前。模型
                    # 参数始终是 FP32（只改了 memory_format，从未 .half()），
                    # 梯度也是 FP32，其上限 3.4e38，缩放系数根本不可能让它
                    # 在 65504 处溢出。那些 inf/nan 是前向/反向里真实的数值
                    # 故障（最可疑是 NPU 强制 math 注意力物化大 logits 时的
                    # FP16 溢出），不是 loss scaling 的伪影——所以真正的
                    # 修复点在别处，此处只负责让它**可观测**。
                    _scale_now = scaler.get_scale()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if use_scaler and scaler.get_scale() < _scale_now:
                        _n_skipped += 1
                        _locate_overflow(optimizer, logger)
                        if _scale_now >= _OVERFLOW_WARN_SCALE > scaler.get_scale():
                            logger.warning("[fp16] 缩放值首次跌破 %d：%.0f -> %.0f。"
                                           "累计跳过 %d 步（占比 %.2f%%）。",
                                           int(_OVERFLOW_WARN_SCALE), _scale_now,
                                           scaler.get_scale(), _n_skipped,
                                           100.0 * _n_skipped / max(1, step))
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update()
            except Exception as oom_exc:
                # 同时捕获 CUDA 与 NPU 的 OOM（两后端异常类型不同）
                _oom_types = [torch.cuda.OutOfMemoryError]
                _npu_oom = npu_out_of_memory_error_type()
                if _npu_oom is not None:
                    _oom_types.append(_npu_oom)
                if not isinstance(oom_exc, tuple(_oom_types)):
                    raise
                if _backend == 'npu':
                    npu_empty_cache()
                else:
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
                logger.error("建议减小 --batch-size（如 128/96/64），或调小 --attn-window。")
                logger.error("已保存进度至 %s.latest(.train_state)，可用 --resume 续训。",
                             args.out)
                logger.error("已清理显存并退出，请调整参数后重跑。")
                logger.error("=" * 60)
                sys.exit(1)
            if (i + 1) % _accum_steps == 0 or (i + 1) == n_batches:
                scheduler.step()   # 每个 optimizer step 推进一步
            step += 1

            # 打点：stdout 与 SwanLab 频率解耦，且共用同一次设备同步。
            # 旧实现里 log_every 同时控制两者，且把三个 loss 张量各取两遍
            # （stdout 一次、swanlab 一次），共 6 次同步、其中 3 次重复。
            _do_stdout, _do_swanlab = _should_log(
                step, args.log_every, args.swanlab_every,
                swanlab_logger is not None)
            if _do_stdout or _do_swanlab:
                # ⚠ 这里传的是 log_loss（报告口径，含 c‖θ‖²），**不是**上面被
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
                #   · `torch.npu.max_memory_allocated` 全仓库只那一处、没在
                #     torch_npu 2.1 上验证过；它在日志路径上抛异常就是**第 50 步
                #     崩**，正好毁掉最需要那个数的时刻。观测不该有能力杀死被观测的进程。
                if _backend == 'cuda':
                    mem = torch.cuda.memory_reserved(device) / 1e9
                elif _backend == 'npu':
                    mem = npu_memory_reserved(device) / 1e9
                else:
                    mem = 0.0
                _now = time.time()
                # ⚠ 有效 batch 必须含梯度累积：原来写的是 bs×world_size，漏乘
                # accumulation ⇒ 用了 `--gradient-accumulation-steps 2` 时日志把
                # 吞吐**报成实际的一半**（2026-10-01 修）。
                _eff_bs = bs * max(1, world_size) * max(1, _accum_steps)
                speed = (step - _step_at_start) * _eff_bs / max(1e-6, _now - t0)
                # 瞬时速率：距上一次 stdout 打点。与分段耗时同口径，
                # 二者相乘即该区间理论样本数，可直接核对时间丢在哪。
                spd_inst = ((step - _last_stdout_step) * _eff_bs
                            / max(1e-6, _now - _last_stdout_t))
                _nd = max(1, _n_timed)
                _dms = _t_data * 1000.0 / _nd
                _cms = _t_comp * 1000.0 / _nd
                _sms = _t_save * 1000.0 / _nd
                _ems = _t_eval * 1000.0 / _nd
                _dmax = _t_data_max * 1000.0
                # 重置放在打点处：某步的 save/eval 发生在其打点之后，
                # 因此计入下一个区间，与墙钟口径一致
                _t_data = _t_comp = _t_save = _t_eval = 0.0
                _t_data_max = 0.0
                _n_timed = 0
                _scale = scaler.get_scale()

            if _do_stdout:
                logger.info("[step %d/%d] loss=%.4f (p=%.4f v=%.4f) lr=%.2e "
                            "scale=%.0f mem=%.2fGB "
                            "spd=%.0f spd_inst=%.0f s/s "
                            "elapsed=%.0fs skip=%d | "
                            "d=%.0f c=%.0f s=%.0f e=%.0f dmax=%.0f ms",
                            step, total_steps, _lv, _pv, _vv,
                            lr, _scale, mem,
                            speed, spd_inst, _now - t0, _n_skipped,
                            _dms, _cms, _sms, _ems, _dmax)
                _last_stdout_t = time.time()
                _last_stdout_step = step

            if _do_swanlab:
                try:
                    swanlab_logger.log({
                        "loss": _lv,
                        "policy_loss": _pv,
                        "value_loss": _vv,
                        "lr": lr,
                    "memory_gb": mem,
                    "speed": speed,
                    "speed_inst": spd_inst,
                    "skipped_steps": _n_skipped,
                    "skip_rate_pct": 100.0 * _n_skipped / max(1, step),
                    "scaler_scale": _scale,
                    "t_data_ms": _dms,
                    "t_comp_ms": _cms,
                    "t_save_ms": _sms,
                    "t_eval_ms": _ems,
                        "epoch": epoch,
                        "step_pct": step / total_steps,
                        "scaler_scale": _scale if use_scaler else 1.0,
                    }, step=step)
                except Exception as e:
                    logger.warning("[swanlab] log 失败: %s", e)

            if _do_stdout:
                # 内核剖析结束：打印 top CUDA kernel 耗时表
                if _prof_ctx is not None and step >= _prof_at + 50:
                    _prof_ctx.__exit__(None, None, None)
                    try:
                        table = _prof_ctx.key_averages().table(
                            sort_by='cuda_time_total', row_limit=18)
                        logger.info("[profile] 内核耗时 top-18（CUDA 时间排序）:\n%s",
                                    table)
                    except Exception as e:
                        logger.warning("[profile] 打印内核耗时表失败: %s", e)

            # 定期保存快照（仅主进程写盘）
            if is_main and args.save_every > 0 and step % args.save_every == 0:
                _t_save0 = time.perf_counter()
                save_model(model, args.out + '.latest')
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
                torch.save(_state, args.out + '.latest.train_state')
                _t_save += time.perf_counter() - _t_save0

            # 定期评估：综合指标（所有 rank 都做 eval，避免 barrier 死锁）
            if args.eval_every > 0 and step % args.eval_every == 0 and len(eval_idx) > 0:
                if ema is not None:
                    ema.apply_shadow()
                _t_eval0 = time.perf_counter()
                metrics = evaluate_metrics(
                    model, dataset, eval_idx, bs, device, amp_dtype,
                    max_batches=args.eval_max_batches,
                    use_channels_last=use_channels_last)
                _t_eval += time.perf_counter() - _t_eval0
                if ema is not None:
                    ema.restore()
                # 曾经在这里周期性 `empty_cache()` 回收分配器缓存段，2026-10-01 撤掉：
                # 它在 NPU 上没验证过，且**每次 eval 都调**（默认 ~35 分钟一次）在训练
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
                        " ★ new best" if metrics['top1'] > best_eval_acc else "")
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

                if stop_flag.item():
                    break

        # 双层 break 之二：跳出 step 循环后还要跳出 epoch 循环（缩进 8 = epoch
        # 循环体、step 循环之外），否则只跳过本 epoch 剩下的 step，外层
        # for epoch 照开下一轮 → 早停等于没开。
        if stop_flag.item():
            if is_main:
                logger.info("[early_stop] 提前结束训练，进入收尾流程（ONNX 导出 / 最终评估）")
            break

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
                        "final_top5": final_metrics['top5'],
                        "final_top10": final_metrics['top10'],
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
