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
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F


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


def npu_grad_scaler(enabled: bool):
    return torch.npu.amp.GradScaler(enabled=enabled)


def npu_out_of_memory_error_type():
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
        p = torch.cuda.get_device_properties(0)
        cap = (p.major, p.minor)
        if cap >= (8, 0):
            amp_status = "bf16+fp16 可用（%s, sm_%d%d）" % (p.name, cap[0], cap[1])
        else:
            amp_status = "仅 fp16（%s, sm_%d%d，Volta/Turing 无 bf16）" % (p.name, cap[0], cap[1])
    elif npu_is_available():
        amp_status = "bf16 可用（Ascend NPU）"
    else:
        amp_status = "不支持（CPU 走 FP32）"
    logger.info("[env] flash-attn: %s", fa_status)
    logger.info("[env] torch.compile: %s | 混合精度: %s", comp_status, amp_status)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.networks.alphanet import AlphaGoNet
from src.data.dataset import SupervisedDataset
from scripts.build_dataset import build


def save_model(model, path):
    """保存模型权重，并剥离 torch.compile 包装产生的 '_orig_mod.' 前缀，
    保证存档无论是否经 compile 都能被后续普通加载/resume 使用。"""
    sd = model.state_dict()
    if any(k.startswith('_orig_mod.') for k in sd.keys()):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    torch.save(sd, path)


def setup_logging(log_file, level: int = logging.INFO) -> logging.Logger:
    """配置 logging：同时写文件与输出到控制台（无缓冲，实时可见）。"""
    logger = logging.getLogger('train')
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
    return logger


def maybe_autocast(device, dtype=torch.float16):
    """在 CUDA/NPU 上开启 autocast，dtype 由设备能力决定（A100/BF16、V100/FP16、NPU/BF16）。
    CPU 或 amp 关闭时返回 nullcontext。device 字符串 'cuda'/'npu' 直接传给 torch.amp.autocast。"""
    if device in ('cuda', 'npu'):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            try:
                return torch.amp.autocast(device, dtype=dtype)
            except TypeError:
                # 老接口回退
                if device == 'cuda':
                    return torch.cuda.amp.autocast(enabled=True, dtype=dtype)
                return torch.npu.amp.autocast(enabled=True, dtype=dtype)
    return nullcontext()


def load_dataset(path):
    """加载单个 .npz 训练集。"""
    d = np.load(path, allow_pickle=False)
    return SupervisedDataset({k: d[k] for k in d.files})


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
            try:
                d, n_games, skip = build(src, board_size, max_games_per_tgz)
            except RuntimeError as e:
                logger.warning("[data] 跳过分片 %s：%s", src, e)
                return None, 0, 0
            if n_games == 0:
                logger.warning("[data] 跳过分片 %s：0 有效局（与 --board-size %d 不匹配或空）",
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
        return SupervisedDataset(merged)
    return load_dataset(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help="单个 .npz 训练集，或包含多个 .tgz/.tar.gz/.npz 的目录（自动合并所有分片）")
    ap.add_argument('--max-games-per-tgz', type=int, default=0,
                    help="目录模式下每个 tgz 最多解析的棋局数（0=全部），用于子采样控制内存")
    ap.add_argument('--device', default='auto')
    ap.add_argument('--use-amp', action='store_true')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--board-size', type=int, default=19)
    ap.add_argument('--save-every', type=int, default=2000)
    ap.add_argument('--out', default='models/sft.pt')
    # 注意力相关
    ap.add_argument('--attention-mode', default='mix',
                    choices=['none', 'mix', 'all'],
                    help='主干注意力模式: none=纯卷积, mix=卷积+注意力混合, all=全注意力')
    ap.add_argument('--num-attention-layers', type=int, default=4,
                    help='mix 模式下注意力块数量')
    ap.add_argument('--num-heads', type=int, default=4, help='多头注意力头数')
    ap.add_argument('--attention-dropout', type=float, default=0.0)
    ap.add_argument('--attn-mode', default='global',
                    choices=['global', 'window', 'axial', 'sparse'],
                    help='注意力计算模式: global=全配对, window=滑动窗口, axial=轴向')
    ap.add_argument('--attn-window', type=int, default=7, help='window 模式窗口边长')
    ap.add_argument('--window-chunk', type=int, default=128,
                    help='window 注意力沿窗口维分块大小（越小越省显存，过大易 OOM）')
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--log-every', type=int, default=50,
                    help='每隔多少 step 打印一次训练日志（loss/lr/吞吐/显存）')
    ap.add_argument('--log-file', default='training.log',
                    help='训练日志文件路径（同时输出到控制台），设为空字符串可关闭文件日志')
    ap.add_argument('--resume', default='',
                    help='断点续训：指定已保存的 .pth 模型路径，会从该权重 + 同目录 '
                         '.train_state.pt 恢复 optimizer/scheduler/step 计数继续训练')
    ap.add_argument('--compile', action='store_true',
                    help='用 torch.compile 融合算子（GPU 上约 20-40%% 提速，首次迭代较慢）')
    ap.add_argument('--compile-mode', default='default',
                    choices=['default', 'max-autotune', 'reduce-overhead'],
                    help='torch.compile 模式: default=常规融合, max-autotune=A100 上进一步 '
                         '自动调优提速（编译更久）, reduce-overhead=小 batch 低开销')
    args = ap.parse_args()

    # 配置日志（控制台 + 文件），统一用 logger 输出便于事后排查
    log_file = args.log_file if args.log_file else None
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    _check_training_env(logger)
    logger.info("=" * 60)
    logger.info("配置: data=%s board=%d batch=%d epochs=%d lr=%s wd=%s",
                args.data, args.board_size, args.batch_size, args.epochs,
                args.lr, args.weight_decay)
    logger.info("注意力: mode=%s attn_mode=%s window=%d heads=%d layers=%d dropout=%s compile=%s chunk=%d",
                args.attention_mode, args.attn_mode, args.attn_window,
                args.num_heads, args.num_attention_layers, args.attention_dropout, args.compile,
                args.window_chunk)
    logger.info("日志: log_every=%d eval_every=%d save_every=%d out=%s",
                args.log_every, args.eval_every, args.save_every, args.out)
    logger.info("=" * 60)

    if args.device == 'auto':
        device = _auto_select_device()
    else:
        device = args.device
    use_amp = args.use_amp or (device.split(':')[0] in ('cuda', 'npu'))

    # ---- 多后端自适应路径（CUDA / NPU / CPU）----
    # 各后端能力差异很大，逐后端决定：
    #   - amp_dtype:       A100/A800/H100(sm_80+) 与 Ascend 910B -> bfloat16（原生支持）
    #                     V100(sm_70, Volta) -> float16（无 bf16）
    #   - use_scaler:      BF16 下关闭 GradScaler（不下溢）；FP16 下开启
    #   - use_channels_last: A100 卷积走 NHWC 更快；NPU/CPU 收益有限默认关
    #   - sdpa_force_math: NPU 上 FlashAttention 后端不稳（CANN SDPA 与 CUDA 不同），
    #                     强制走手写 math 注意力最稳；V100 同样强制 math；A100 走 Flash
    #   - compile_disable_sparse: V100 inductor 对 unfold 触发 PolynomialError 需禁用；
    #                      A100 可编译；NPU 上 torch.compile(inductor) 不可用，直接整体禁用
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
            compile_disable_sparse = False  # A100 上 unfold 可被 inductor 编译
            logger.info("[device] %s (sm_%d%d) | 启用 A100 路径: BF16 + FlashAttn + "
                        "channels_last + 全量 compile", gpu_name, *compute_cap)
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
        # Ascend 910B / 910Pro：CANN + torch_npu 后端
        gpu_name = npu_get_device_name(_dev_idx)
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        # 910B 原生 BF16；但 FlashAttention 后端在 CANN 上不稳，强制手写 math 注意力
        # channels_last 对 NPU 卷积无明确收益，关闭；torch.compile(inductor) 不可用，禁用
        amp_dtype = torch.bfloat16
        use_scaler = False  # BF16 不下溢
        use_channels_last = False
        sdpa_force_math = True
        compile_disable_sparse = True
        logger.info("[device] %s (NPU/CANN) | NPU 路径: BF16 + 手写 math 注意力 + "
                    "禁用 torch.compile(inductor)", gpu_name)
        if args.compile:
            logger.warning("[device] NPU 上 torch.compile(inductor) 不可用，已忽略 --compile；"
                           "如需图编译请用 torchair (torch_npu.experimental_config)。")
            args.compile = False
    else:
        # CPU 或其他：纯 FP32，无 AMP、无 channels_last
        amp_dtype = torch.float32
        use_scaler = False
        use_channels_last = False
        sdpa_force_math = True
        compile_disable_sparse = True
        logger.info("[device] CPU | 走 FP32 路径（无 AMP/编译）")

    # 把注意力后端/编译开关透传给 backbone 模块（所有分支统一设置）
    from src.networks import backbone as _backbone
    _backbone.set_sdpa_force_math(sdpa_force_math)
    _backbone.set_compile_disable_sparse(compile_disable_sparse)

    # flash-attn 独立库启用决策：仅「Ampere+ CUDA 且走非 math 路径」时尝试加载。
    # 加载失败自动回退内置 SDPA，不影响训练启动。
    if _backend == 'cuda' and compute_cap >= (8, 0) and not sdpa_force_math:
        fa_ok, fa_msg = _backbone.set_flash_attn(True)
        if fa_ok:
            logger.info("[env] 注意力内核: flash-attn %s（优先于内置 SDPA）", fa_msg)
        else:
            logger.info("[env] 注意力内核: 内置 SDPA（flash-attn %s）", fa_msg)
    else:
        _backbone.set_flash_attn(False)
        logger.info("[env] 注意力内核: %s",
                    "手写 math（%s 不支持 Flash）" % (_backend.upper(),)
                    if sdpa_force_math else "内置 SDPA")

    logger.info("启动训练 | torch=%s | device=%s | amp_dtype=%s scaler=%s channels_last=%s",
                torch.__version__, device, amp_dtype, use_scaler, use_channels_last)

    dataset = load_from_path(args.data, args.board_size, args.max_games_per_tgz)
    n = len(dataset)
    logger.info("[data] 总样本数=%d | 训练=%d | 验证=%d", n, int(n * 0.98), n - int(n * 0.98))
    # 留出 held-out 集用于 top-1 准确率监控
    n_train = int(n * 0.98)
    idx_all = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx_all)
    train_idx = idx_all[:n_train]
    eval_idx = idx_all[n_train:]

    model = AlphaGoNet(
        in_channels=12,
        backbone_channels=128,
        backbone_res_blocks=12,
        attention_mode=args.attention_mode,
        num_attention_layers=args.num_attention_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        attn_mode=args.attn_mode,
        attn_window=args.attn_window,
        action_size=args.board_size * args.board_size + 1,  # +1 为 pass 类别
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[model] 参数量=%.2fM | 设备=%s", n_params / 1e6, device)

    # A100 上把卷积型特征（N,C,H,W）转 channels_last(NHWC)，卷积算子走更快内存布局。
    # 输入 state 也需同步转格式（见训练/评估循环），故这里仅转换模型权重布局。
    if use_channels_last and _backend == 'cuda':
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        logger.info("[model] 已启用 channels_last (NHWC) 内存格式（A100 卷积加速）")

    # 将 window_chunk（分块大小）透传到所有 window 注意力层，控制显存峰值
    if args.window_chunk > 0:
        for _m in model.modules():
            if hasattr(_m, 'window_chunk'):
                _m.window_chunk = args.window_chunk
        logger.info("[model] window_chunk=%d（滑动窗口注意力分块大小）", args.window_chunk)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # BF16 后端（A100/NPU）下 use_scaler=False（BF16 不下溢，省去 loss scaling 的额外同步）；
    # V100/FP16 下开启 GradScaler。按设备选择 GradScaler 实现。
    if _backend == 'npu':
        scaler = npu_grad_scaler(enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ---- 学习率调度：基于“总 step 数”而非 epoch 数 ----
    # 旧版用 T_max=args.epochs 导致余弦在第 1 个 epoch 结束就被砍到 ~0，
    # 后续 epoch 在 lr≈0 附近横盘。这里用真实总 step 数，并加前 5% step 线性 warmup。
    n_batches = (n_train + args.batch_size - 1) // args.batch_size
    total_steps = max(1, args.epochs * n_batches)
    warmup_steps = max(1, int(total_steps * 0.05))
    after_warmup = max(1, total_steps - warmup_steps)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=after_warmup)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    bs = args.batch_size
    step = 0
    best_eval_acc = -1.0
    start_epoch = 0
    t0 = time.time()

    # ---- 断点续训：从 --resume 指定的模型权重 + 同目录 .train_state.pt 恢复 ----
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume 指定的模型不存在: {args.resume}")
        state_path = args.resume + '.train_state'
        logger.info("[resume] 加载模型权重: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        # 统一前缀：checkpoint 可能带（来自 compile 存档）或不带 "_orig_mod." 前缀，
        # 目标模型也可能被 compile 包成 OptimizedModule（内部为 _orig_mod）。
        # 先全部规范化成不带前缀，再按需补回，保证任意组合都能匹配。
        ckpt = {k.replace('_orig_mod.', '', 1): v for k, v in ckpt.items()}
        target = getattr(model, '_orig_mod', model)  # compile 后为真实模块
        if hasattr(model, '_orig_mod'):
            ckpt = {'_orig_mod.' + k: v for k, v in ckpt.items()}
            logger.info("[resume] 模型已 torch.compile 包装，权重按 _orig_mod. 前缀对齐")
        target.load_state_dict(ckpt)
        if os.path.isfile(state_path):
            tstate = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(tstate['optimizer'])
            scheduler.load_state_dict(tstate['scheduler'])
            scaler.load_state_dict(tstate['scaler'])
            step = tstate.get('step', 0)
            best_eval_acc = tstate.get('best_eval_acc', -1.0)
            start_epoch = tstate.get('epoch', 0)
            if 'rng' in tstate:
                torch.set_rng_state(tstate['rng'].cpu())
            logger.info("[resume] 恢复训练状态 | step=%d best_eval_acc=%.4f epoch=%d",
                        step, best_eval_acc, start_epoch)
        else:
            logger.warning("[resume] 未找到 %s（仅恢复模型权重，optimizer/scheduler 从头开始）",
                           state_path)

    # torch.compile 融合算子（GPU 上约 20-40%% 提速）。必须在 resume 加载之后再做，
    # 否则模型会被包成 OptimizedModule，其 state_dict 带 "_orig_mod." 前缀，与
    # checkpoint 的 "backbone.xxx" 不匹配导致 load 失败。
    if args.compile:
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, dynamic=False, mode=args.compile_mode)
                with torch.no_grad():
                    dummy = torch.zeros(1, 12, args.board_size, args.board_size,
                                        device=device)
                    model(dummy)
                logger.info("[train] 已启用 torch.compile 算子融合")
            except Exception as e:  # noqa: BLE001
                logger.warning("[train] torch.compile 不可用，回退 eager: %s", e)
        else:
            logger.info("[train] 当前 torch 版本不支持 torch.compile，跳过")

    logger.info("[train] 开始训练 | steps/epoch=%d | 总 steps≈%d | warmup=%d",
                n_batches, total_steps, warmup_steps)

    for epoch in range(start_epoch, args.epochs):
        rng.shuffle(train_idx)
        model.train()
        epoch_loss = 0.0
        n_batches = (len(train_idx) + bs - 1) // bs
        for i in range(n_batches):
            try:
                sel = train_idx[i * bs:(i + 1) * bs]
                state, move_t, value_t = dataset.sample_batch(sel, device)
                # A100 上转 NHWC 以匹配模型 channels_last 布局，卷积更快
                if use_channels_last:
                    state = state.to(memory_format=torch.channels_last)
                with maybe_autocast(device, amp_dtype):
                    policy_logits, value_pred = model(state)
                    policy_loss = F.cross_entropy(policy_logits.float(), move_t)
                    value_loss = F.mse_loss(value_pred.float().squeeze(), value_t.squeeze())
                    loss = policy_loss + value_loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
                # 保存当前进度，便于减小 batch 后用 --resume 续训
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
            scheduler.step()   # 每个 step 推进一步（warmup+cosine 调度依赖逐 step）
            step += 1
            epoch_loss += loss.item()

            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                if _backend == 'cuda':
                    mem = torch.cuda.memory_reserved(device) / 1e9
                elif _backend == 'npu':
                    mem = npu_memory_reserved(device) / 1e9
                else:
                    mem = 0.0
                speed = step * bs / max(1e-6, time.time() - t0)
                logger.info("[step %d/%d] loss=%.4f (p=%.4f v=%.4f) lr=%.2e "
                            "scale=%.0f mem=%.2fGB spd=%.0f s/s elapsed=%.0fs",
                            step, total_steps,
                            loss.item(), policy_loss.item(), value_loss.item(),
                            lr, scaler.get_scale(), mem, speed, time.time() - t0)

            if step % args.eval_every == 0 and len(eval_idx) > 0:
                acc = evaluate(model, dataset, eval_idx, bs, device, amp_dtype, use_channels_last)
                logger.info("[eval] step=%d train_loss=%.4f eval_top1=%.4f scale=%.0f elapsed=%.0fs",
                            step, epoch_loss / max(1, (i + 1)), acc,
                            scaler.get_scale(), time.time() - t0)
                if acc > best_eval_acc:
                    best_eval_acc = acc
                    save_model(model, args.out)
                    logger.info("  -> 保存最佳模型至 %s", args.out)
            if step % args.save_every == 0:
                # 同时保存完整训练状态，供 --resume 断点续训
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
        logger.info("epoch %d/%d done, loss=%.4f", epoch + 1, args.epochs,
                    epoch_loss / max(1, n_batches))

    # 最终保存（含训练状态，供 --resume 续训）
    save_model(model, args.out)
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step,
        'epoch': args.epochs,
        'best_eval_acc': best_eval_acc,
        'rng': torch.get_rng_state(),
    }, args.out + '.train_state')
    logger.info("训练完成。最佳 eval_top1=%.4f，模型已保存至 %s", best_eval_acc, args.out)
    logger.info("总耗时 %.0fs", time.time() - t0)


@torch.no_grad()
def evaluate(model, dataset, eval_idx, bs, device, amp_dtype, use_channels_last):
    model.eval()
    correct = 0
    total = 0
    for i in range(0, len(eval_idx), bs):
        sel = eval_idx[i:i + bs]
        state, move_t, _ = dataset.sample_batch(sel, device)
        if use_channels_last:
            state = state.to(memory_format=torch.channels_last)
        with maybe_autocast(device, amp_dtype):
            policy_logits, _ = model(state)
        pred = policy_logits.argmax(dim=-1)
        correct += int((pred.cpu() == move_t.cpu()).sum())
        total += len(sel)
    return correct / max(1, total)


if __name__ == '__main__':
    main()
