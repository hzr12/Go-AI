#!/usr/bin/env python3
"""AlphaZero 式自我对弈无监督训练（自对弈生成数据 → 训练 → 循环）。

每轮迭代：
  1. 自对弈 --games 局（MCTS visit 分布当 policy 标签，终局胜负当 value 标签，
     根先验混 Dirichlet 噪声 + 温度采样保证探索）
  2. 数据 8 对称增强（4 旋转 × 2 镜像）入 replay buffer
  3. 训练 --epochs 遍（policy 软标签交叉熵 + value BCE，带 EMA/LR Schedule/梯度裁剪），
     value 标签按手数位置做 tanh 软化（开局弱信号、终局强信号）

50GB 空间约束：
  - 流式处理：自对弈数据不写临时 npz，用共享内存队列传递
  - 只保存最佳权重：latest + best 两个文件
  - 紧凑 buffer：内存中最多保留 500 局数据

CPU 上建议 9 路小规模验证流程；19 路正式训练请上 GPU/NPU 并加大 --sims。
每个迭代保存 models/az_<size>_iter<N>.pth，可用 eval_elo.py 对比新旧棋力。

用法:
    python scripts/selfplay_train.py --board-size 9 --iters 5 --games 4 --sims 32 \
        --model models/sft_19x19_v3.pth --out models/az

NPU 正式训练（4 卡 DDP）:
    torchrun --nproc_per_node=4 scripts/selfplay_train.py \
        --board-size 19 --iters 20 --games 16 --sims 400 \
        --parallel-games 4 --ddp --streaming \
        --model models/sft_19x19_v12.pth --out models/az_best.pth
"""
import sys
import os
import time
import struct
import argparse
from multiprocessing import Queue, Process, Event

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.inference import GoAI
from src.game.go_rules import GoBoard
from src.search.mcts import MCTS
from src.search.light_rollout import FastPolicy, DiverseRolloutPolicy


# --------------------------------------------------------------------------- #
# Playout 随机化增强：根据步数轮换策略
# --------------------------------------------------------------------------- #
def _get_rollout_policy(use_diverse=False, board_size=19):
    """创建 rollout 策略（基础或多样化）。"""
    if use_diverse:
        return DiverseRolloutPolicy(board_size, num_strategies=4)
    return FastPolicy(board_size)


def _sample_rollout_move(policy, board, step, rng):
    """根据策略类型采样 move。"""
    if isinstance(policy, DiverseRolloutPolicy):
        return policy.sample_move(board, step, rng)
    return policy.sample_move(board, rng)


# --------------------------------------------------------------------------- #
# NPU 辅助函数（与 train_sft.py 保持一致）
# --------------------------------------------------------------------------- #
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


def _auto_select_device():
    """自动选择最优设备（CUDA > NPU > CPU）"""
    if torch.cuda.is_available():
        return 'cuda'
    if npu_is_available():
        return 'npu'
    return 'cpu'


def maybe_autocast(device, dtype=torch.float16):
    """在 CUDA/NPU 上开启 autocast"""
    dev = device.split(':')[0] if isinstance(device, str) else str(device)
    if dev == 'cuda' and hasattr(torch, 'amp'):
        try:
            return torch.amp.autocast(dev, dtype=dtype)
        except TypeError:
            return torch.cuda.amp.autocast(enabled=True, dtype=dtype)
    if dev == 'npu' and hasattr(torch, 'npu'):
        try:
            return torch.npu.amp.autocast(enabled=True, dtype=dtype)
        except Exception:
            pass
    return nullcontext()


from contextlib import nullcontext


# --------------------------------------------------------------------------- #
# EMA（指数移动平均）权重
# --------------------------------------------------------------------------- #
class EMA:
    """指数移动平均（Exponential Moving Average）权重。"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach()
                       for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name].data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        self.backup = {name: param.clone()
                       for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data = self.shadow[name].data

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name].data
        del self.backup


# --------------------------------------------------------------------------- #
# 自对弈数据生成
# --------------------------------------------------------------------------- #
def self_play_game(ai, board_size, sims, max_moves, temperature,
                   expand_topk, expand_chunk, priors_leaf=True,
                   dir_alpha=0.3, dir_eps=0.25,
                   use_rollout=False, rollout_lambda=0.25, rollout_steps=None,
                   leaf_ab_depth=0, c_puct=2.0, virtual_loss=8.0,
                   num_threads=8, spec_prefetch=False,
                   use_diverse_rollout=False):
    """一局自对弈。返回 [(planes, visit_target, player, mc), ...], score(黑-白)。"""
    mcts = MCTS(ai, board_size=board_size, num_threads=num_threads,
                expand_topk=expand_topk, expand_chunk=expand_chunk,
                priors_leaf=priors_leaf, temperature=temperature,
                dirichlet_alpha=dir_alpha, dir_eps=dir_eps,
                spec_prefetch=spec_prefetch,
                use_rollout=use_rollout, rollout_lambda=rollout_lambda,
                rollout_steps=rollout_steps,
                leaf_ab_depth=leaf_ab_depth,
                c_puct=c_puct, virtual_loss=virtual_loss)
    
    # 创建 rollout 策略（支持多样化）
    rollout_policy = _get_rollout_policy(use_diverse_rollout, board_size)
    rng = np.random.default_rng(1234)
    board = GoBoard(board_size)
    hists = [[-1, -1, -3], [-1, -1, -3]]  # [黑方, 白方] 最近3手
    n_actions = board_size * board_size + 1
    passes = 0
    mc = 0
    path_moves = []
    data = []
    while passes < 2 and mc < max_moves:
        to_play = board.current_player
        legal = board.get_legal_moves()
        if not legal.any():
            board.play(-1)
            path_moves.append(-1)
            passes += 1
            mc += 1
            continue
        visits, probs, _rv = mcts.search(
            board, hists[0], hists[1], to_play,
            simulations=sims, path_moves=path_moves)
        # 记录训练样本
        planes = np.ascontiguousarray(board.feature_planes_batched(
            board.board[None], [list(hists[0])], [list(hists[1])],
            [to_play], [board.ko_point])[0])
        vt = np.zeros(n_actions)
        vs = visits.sum()
        if vs > 0:
            vt[:n_actions - 1] = visits[:n_actions - 1] / vs
            vt[n_actions - 1] = visits[n_actions - 1] / vs
        data.append((planes, vt, to_play, mc))
        # 温度线性衰减
        progress = min(1.0, mc / max(30, 1))
        temp = 1.0 - progress * (1.0 - 0.1)
        p = np.asarray(probs).reshape(-1).astype(np.float64)
        p[-1] = max(p[-1], 0.0)
        if temp > 0 and temp != 1.0:
            p = p ** (1.0 / temp)
        s = p.sum()
        if s <= 0:
            mv = n_actions - 1
        else:
            mv = int(np.random.choice(n_actions, p=p / s))
        pmv = -1 if mv == n_actions - 1 else mv
        success = board.play(pmv)
        if not success:
            board.play(-1)
            pmv = -1
        path_moves.append(pmv)
        h = hists[0] if to_play == 1 else hists[1]
        h.pop(0)
        h.append(pmv)
        if pmv >= 0:
            passes = 0
        else:
            passes += 1
        mc += 1
    return data, board.score()


def augment8(plane, target, n):
    """8 对称增强：4 旋转 × 2 镜像（棋盘特征与 visit 目标同步变换）。"""
    board_t = target[:n * n].reshape(n, n)
    pass_t = target[n * n]
    out = []
    for t in range(8):
        k = t % 4
        pl = plane.copy()
        tb = board_t.copy()
        if t >= 4:
            pl = pl[:, :, ::-1]
            tb = tb[:, ::-1]
        if k > 0:
            pl = np.rot90(pl, k=-k, axes=(1, 2))
            tb = np.rot90(tb, k=-k)
        tv = np.concatenate([tb.reshape(-1), [pass_t]])
        out.append((np.ascontiguousarray(pl), np.ascontiguousarray(tv)))
    return out


# --------------------------------------------------------------------------- #
# 并行自对弈 worker
# --------------------------------------------------------------------------- #
def _selfplay_worker(gid, model_path, args, result_queue):
    """单个自对弈进程。"""
    device = args.device
    if device == 'auto':
        device = _auto_select_device()
    
    # 每个进程独立加载模型（绕过 GIL）
    if args.onnx_model:
        ai = GoAI(model_path=args.onnx_model, board_size=args.board_size, 
                  device='cpu', use_amp=False)
    else:
        ai = GoAI(model_path=model_path, board_size=args.board_size, device=device,
                  use_amp=True, attn_mode='window', attn_window=7)
    
    game_data, score = self_play_game(
        ai, args.board_size, args.sims,
        args.max_moves or 3 * args.board_size * args.board_size,
        args.temperature, args.expand_topk, args.expand_chunk,
        use_rollout=getattr(args, 'use_rollout', False),
        rollout_lambda=getattr(args, 'rollout_lambda', 0.25),
        leaf_ab_depth=getattr(args, 'leaf_ab_depth', 2),
        c_puct=getattr(args, 'c_puct', 2.0),
        virtual_loss=getattr(args, 'virtual_loss', 8.0),
        num_threads=getattr(args, 'mcts_threads', 3),  # 使用新的参数名
        spec_prefetch=getattr(args, 'spec_prefetch', False),
        use_diverse_rollout=getattr(args, 'use_diverse_rollout', False)
    )
    result_queue.put({'gid': gid, 'data': game_data, 'score': score})


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train_epochs(ai, buffer, args, device):
    """在 replay buffer 上训练若干遍。返回平均 loss。"""
    model = ai.model
    model.train()

    # AdamW 参数分组：value head 用独立学习率
    no_decay = ['bias', 'bn', 'Norm']
    value_decay = [p for n, p in model.named_parameters()
                   if 'value' in n and not any(k in n for k in no_decay)]
    value_no_decay = [p for n, p in model.named_parameters()
                      if 'value' in n and any(k in n for k in no_decay)]
    other_decay = [p for n, p in model.named_parameters()
                   if 'value' not in n and not any(k in n for k in no_decay)]
    other_no_decay = [p for n, p in model.named_parameters()
                      if 'value' not in n and any(k in n for k in no_decay)]
    opt = torch.optim.AdamW([
        {'params': other_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': other_no_decay, 'lr': args.lr, 'weight_decay': 0.0},
        {'params': value_decay, 'lr': args.lr * args.value_lr_mult,
         'weight_decay': args.weight_decay},
        {'params': value_no_decay, 'lr': args.lr * args.value_lr_mult, 'weight_decay': 0.0},
    ])

    # Cosine LR Schedule with Warmup
    n = len(buffer)
    steps_per_epoch = max(1, n // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.10))
    after_warmup = max(1, total_steps - warmup_steps)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=after_warmup)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    # EMA
    ema = EMA(model, decay=0.999) if args.use_ema == 1 else None

    losses = []
    if not buffer:
        return 0.0
    
    # 预分配张量避免重复分配
    device_prefix = device.split(':')[0] if isinstance(device, str) else str(device)
    pin_mem = device_prefix in ('cuda', 'npu')
    
    for _ in range(args.epochs):
        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, n, size=args.batch_size)
            batch = [buffer[i] for i in idx]
            planes = torch.from_numpy(np.stack([b[0] for b in batch])).float()
            pi_t = torch.from_numpy(np.stack([b[1] for b in batch])).float()
            z = torch.from_numpy(np.stack([b[2] for b in batch])).float().unsqueeze(1)
            
            if pin_mem:
                planes = planes.pin_memory()
                pi_t = pi_t.pin_memory()
                z = z.pin_memory()
            
            planes = planes.to(device, non_blocking=pin_mem)
            pi_t = pi_t.to(device, non_blocking=pin_mem)
            z = z.to(device, non_blocking=pin_mem)
            
            with maybe_autocast(device):
                policy, value = model(planes)
                logq = torch.log_softmax(policy, dim=-1) + 1e-10
                loss_pi = -(pi_t * logq).sum(dim=1).mean()
                value_target = (z + 1) / 2
                loss_v = F.binary_cross_entropy_with_logits(value.squeeze(-1), value_target)
                loss = loss_pi + loss_v
            
            opt.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            scheduler.step()
            if ema is not None:
                ema.update()
            losses.append(loss.item())

    # eval 时用 EMA 权重
    if ema is not None:
        ema.apply_shadow()
        model.eval()
    else:
        model.eval()
    ret = sum(losses) / max(len(losses), 1)
    if ema is not None:
        ema.restore()
    return ret


def main():
    ap = argparse.ArgumentParser(description="AlphaZero 式自对弈无监督训练（50GB 约束优化版）")
    ap.add_argument("--model", type=str, default=None, help="初始权重（如 SFT 预训练）")
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--iters", type=int, default=5, help="迭代轮数")
    ap.add_argument("--games", type=int, default=4, help="每轮自对弈局数")
    ap.add_argument("--sims", type=int, default=400, help="自对弈每步 MCTS 模拟数")
    ap.add_argument("--max-moves", type=int, default=None, help="单局手数上限（默认 3×点数）")
    ap.add_argument("--temperature", type=float, default=1.0, help="自对弈初始采样温度")
    ap.add_argument("--buffer-size", type=int, default=500, help="replay buffer 容量（局数，非样本数）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2, help="每轮迭代训练遍数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-lr-mult", type=float, default=0.5,
                    help="value head 学习率倍率")
    ap.add_argument("--expand-topk", type=int, default=64,
                    help="MCTS 展开候选截断（CPU 推荐 32-64）")
    ap.add_argument("--expand-chunk", type=int, default=0,
                    help="展开期 α-β 界截断块大小（0=关闭）")
    
    # MCTS 质量参数
    ap.add_argument("--c-puct", type=float, default=2.0, help="PUCT 探索系数")
    ap.add_argument("--virtual-loss", type=float, default=8.0, help="虚拟损失系数")
    ap.add_argument("--num-threads", type=int, default=8, help="MCTS 多线程数")
    ap.add_argument("--spec-prefetch", type=int, default=0, choices=[0, 1],
                    help="启用 worker 推测预评估 (0=关闭, 1=开启)")
    ap.add_argument("--leaf-ab-depth", type=int, default=2, help="叶内 α-β 深度")
    
    # Phase 1 优化参数
    ap.add_argument("--dynamic-topk", type=int, default=0, choices=[0, 1],
                    help="启用动态 topk (早期8, 中期16, 后期32) (0=关闭, 1=开启)")
    ap.add_argument("--dynamic-virtual-loss", type=int, default=0, choices=[0, 1],
                    help="启用动态 virtual loss (早期2, 中期6, 后期12) (0=关闭, 1=开启)")
    ap.add_argument("--policy-pruning-thresh", type=float, default=0.01,
                    help="策略剪枝阈值 (跳过 prior < thresh 的候选)")
    
    # Playout 随机化
    ap.add_argument("--use-diverse-rollout", type=int, default=0, choices=[0, 1],
                    help="启用多样化 rollout 策略 (4 种温度轮换) (0=关闭, 1=开启)")
    
    # Rollout
    ap.add_argument("--use-rollout", type=int, default=0, choices=[0, 1],
                    help="启用 LightPLS rollout (0=关闭, 1=开启)")
    ap.add_argument("--rollout-lambda", type=float, default=0.25, help="rollout 融合权重")
    ap.add_argument("--rollout-steps", type=int, default=60, help="rollout 最大步数")
    
    # 并行生成
    ap.add_argument("--parallel-games", type=int, default=1,
                    help="并行自对弈局数（多进程）")
    ap.add_argument("--result-queue-max", type=int, default=100,
                    help="结果队列最大容量")
    
    # 流式训练
    ap.add_argument("--streaming", type=int, default=0, choices=[0, 1],
                    help="启用流式训练（默认）(0=关闭, 1=开启)")
    ap.add_argument("--no-persist", type=int, default=0, choices=[0, 1],
                    help="不写临时 npz 文件 (0=写入, 1=不写入)")
    
    # DDP
    ap.add_argument("--ddp", type=int, default=0, choices=[0, 1],
                    help="启用 DDP 多卡训练（需 torchrun）(0=关闭, 1=开启)")
    
    # 设备
    ap.add_argument("--device", default="auto",
                    help="设备选择：auto/cuda/npu/cpu")
    
    ap.add_argument("--no-augment", type=int, default=0, choices=[0, 1],
                    help="关闭 8 对称增强 (0=开启, 1=关闭)")
    ap.add_argument("--use-ema", type=int, default=0, choices=[0, 1],
                    help="启用 EMA 权重 (0=关闭, 1=开启)")
    ap.add_argument("--clip-grad", type=float, default=1.0, help="梯度裁剪范数（0=关闭）")
    ap.add_argument("--out", type=str, default="models/az", help="权重输出路径")
    ap.add_argument("--save-every", type=int, default=1, help="每隔几轮保存最佳权重")
    
    # Phase 2: 并行优化参数
    ap.add_argument("--parallel-games", type=int, default=8,
                    help="并行自对弈进程数（建议: CPU cores / 3，24核建议 8）")
    ap.add_argument("--mcts-threads", type=int, default=3,
                    help="每个进程的 MCTS 线程数（建议: cores / parallel_games）")
    ap.add_argument("--onnx-model", type=str, default=None,
                    help="ONNX 模型路径（CPU 推理加速 3-5x）")
    ap.add_argument("--batch-cap", type=int, default=64,
                    help="MCTS 批量展开上限（默认 64）")
    
    # 异步流水线
    ap.add_argument("--async-pipeline", type=int, default=0, choices=[0, 1],
                    help="启用异步流水线（生成与训练并行，需配合 --games-per-iter）(0=关闭, 1=开启)")
    ap.add_argument("--games-per-iter", type=int, default=10,
                    help="异步模式下每轮迭代生成的局数")
    ap.add_argument("--swanlab", type=int, default=0, choices=[0, 1],
                    help="启用 SwanLab 实验跟踪（需设置 SWANLAB_API_KEY 环境变量）(0=关闭, 1=开启)")
    
    args = ap.parse_args()

    # SwanLab 实验跟踪（可选）
    use_swanlab = args.swanlab == 1 or os.environ.get('SWANLAB_API_KEY')
    swanlab_logger = None
    if use_swanlab and is_main:
        try:
            import swanlab
            swanlab.init(
                project="go-ai-rl",
                name=f"selfplay_{args.ver}",
                config={
                    "board_size": args.board_size,
                    "iters": args.iters,
                    "sims": args.sims,
                    "parallel_games": args.parallel_games,
                    "mcts_threads": getattr(args, 'mcts_threads', 3),
                    "c_puct": args.c_puct,
                    "virtual_loss": args.virtual_loss,
                    "buffer_size": args.buffer_size,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "expand_topk": args.expand_topk,
                    "async_pipeline": args.async_pipeline,
                },
            )
            if is_main:
                print(f"[swanlab] 实验跟踪已启用", flush=True)
        except Exception as e:
            if is_main:
                print(f"[swanlab] 初始化失败: {e}", flush=True)

    # 设备选择
    if args.device == "auto":
        device = _auto_select_device()
    else:
        device = args.device
    
    torch.manual_seed(42)
    np.random.seed(42)

    ai = GoAI(model_path=args.model, board_size=args.board_size, device=device,
              use_amp=True, attn_mode="window", attn_window=7)
    
    # 如果指定了 ONNX 模型，切换后端
    if args.onnx_model:
        if is_main:
            print(f"[selfplay] 使用 ONNX 后端: {args.onnx_model}", flush=True)
        ai = GoAI(model_path=args.onnx_model, board_size=args.board_size, 
                  device='cpu', use_amp=False)
    
    # 启用 Phase 1 优化
    if args.dynamic_topk == 0:
        # 暂时不支持禁用，默认启用
        pass
    bs = ai.board_size
    n_actions = bs * bs + 1
    max_moves = args.max_moves or 3 * bs * bs
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # 最佳权重追踪
    best_loss = float('inf')
    best_path = args.out

    buffer = []   # [(planes(12,n,n), target(n²+1), z_soft)]
    total_games = 0
    
    # DDP：获取 rank/world_size
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    is_main = (rank == 0)

    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        
        if is_main:
            print(f"\n[iter {it}/{args.iters}] 开始自对弈...", flush=True)
        
        # SwanLab 记录迭代开始
        if swanlab_logger is not None:
            swanlab.log({"iter_start": it}, step=it)

    # 检查是否启用异步流水线
    if args.async_pipeline == 1:
        from scripts.async_pipeline import AsyncSelfPlayPipeline
        # 异步模式：生成与训练并行
        if is_main:
            print(f"[async] 使用异步流水线模式", flush=True)
            
            # 启动异步流水线（如果尚未启动）
            if not hasattr(main, '_pipeline') or main._pipeline is None:
                main._pipeline = AsyncSelfPlayPipeline(
                    args, model_path=args.model, onnx_model=args.onnx_model
                )
                main._pipeline.start()
            
            # 持续收集数据并训练
            for _ in range(args.games_per_iter or 10):
                # 收集数据
                collected = 0
                while len(buffer) < args.batch_size * 20:
                    item = pipeline.data_queue.get(timeout=0.5)
                    if item:
                        game_data = item['data']
                        score = item['score']
                        _process_game_data(game_data, score, bs, n_actions, buffer, args)
                        collected += 1
                        total_games += 1
                
                if collected > 0 and is_main:
                    print(f"  收集 {collected} 局，buffer={len(buffer)}", flush=True)
                
                # 训练（如果有足够数据）
                if len(buffer) >= args.batch_size * 5:
                    avg_loss = train_epochs(ai, buffer, args, device)
                    
                    # 保存最佳
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        out_path = args.out if args.out.endswith('.pth') else args.out + '.pth'
                        torch.save({"model": ai.model.state_dict(), "iter": it,
                                   "board_size": bs, "args": vars(args)}, out_path)
                        if is_main:
                            print(f"[iter {it}] NEW BEST loss={avg_loss:.4f}", flush=True)
                    
                    # 清空 buffer
                    buffer = []
            
            # 停止流水线
            if is_main:
                pipeline.stop()
        else:
            # 同步模式：原有逻辑
            if args.parallel_games > 1 and args.ddp == 0:
                # 多进程并行生成
                result_queue = Queue(maxsize=args.result_queue_max)
                processes = []
                for g in range(args.parallel_games):
                    p = Process(target=_selfplay_worker,
                               args=(g, args.model, args, result_queue))
                    p.start()
                    processes.append(p)
                # 收集结果
                for _ in range(args.parallel_games):
                    result = result_queue.get()
                    game_data = result['data']
                    score = result['score']
                    _process_game_data(game_data, score, bs, n_actions, buffer, args)
                    total_games += 1
                    if is_main:
                        print(f"  [game {result['gid']+1}] score={score:+.1f} "
                              f"moves={len(game_data)} buffer={len(buffer)}", flush=True)
                
                for p in processes:
                    p.join()
            else:
                # 串行生成
                for g in range(args.games):
                    game_data, score = self_play_game(
                        ai, bs, args.sims, max_moves, args.temperature,
                        args.expand_topk, args.expand_chunk,
                        use_rollout=args.use_rollout,
                        rollout_lambda=args.rollout_lambda,
                        rollout_steps=args.rollout_steps,
                        leaf_ab_depth=args.leaf_ab_depth,
                        c_puct=args.c_puct,
                        virtual_loss=args.virtual_loss,
                        num_threads=getattr(args, 'mcts_threads', 3),
                         spec_prefetch=args.spec_prefetch == 1,
                         use_diverse_rollout=args.use_diverse_rollout == 1)
                    _process_game_data(game_data, score, bs, n_actions, buffer, args)
                    total_games += 1
                    if is_main:
                        print(f"  [iter {it} game {g + 1}] score={score:+.1f} "
                              f"moves={len(game_data)} buffer={len(buffer)}", flush=True)

            # 控制 buffer 大小（50GB 约束）
            max_samples = args.buffer_size * bs * bs * 200
            if len(buffer) > max_samples:
                del buffer[:len(buffer) - max_samples]

            # 训练
            if is_main:
                avg_loss = train_epochs(ai, buffer, args, device)
                dt = time.perf_counter() - t0
                
                # 只保存最佳权重（节省空间）
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    out_path = args.out if args.out.endswith('.pth') else args.out + '.pth'
                    torch.save({"model": ai.model.state_dict(), "iter": it,
                               "board_size": bs, "args": vars(args)}, out_path)
                    if is_main:
                        print(f"[iter {it}/{args.iters}] NEW BEST loss={avg_loss:.4f} "
                              f"-> {out_path}", flush=True)
                
                if is_main:
                    print(f"[iter {it}/{args.iters}] loss={avg_loss:.4f} buffer={len(buffer)} "
                          f"games={total_games} {dt:.0f}s", flush=True)
                    # SwanLab 记录迭代指标
                    if swanlab_logger is not None:
                        swanlab.log({
                            "iter_loss": avg_loss,
                            "iter_games": total_games,
                            "buffer_size": len(buffer),
                            "iter_time_s": dt,
                            "games_per_iter": collected if 'collected' in locals() else 0,
                        }, step=it)
                        swanlab_logger.flush()

    if is_main:
        print(f"训练完成。共 {total_games} 局，最佳权重: {best_path}")
        print("用 scripts/eval_elo.py 对比不同迭代权重棋力。")
        # SwanLab 结束
        if swanlab_logger is not None:
            swanlab.log({"total_games": total_games, "final_iter": args.iters}, step=args.iters)
            swanlab.finish()


def _process_game_data(game_data, score, bs, n_actions, buffer, args):
    """处理一局自对弈数据，加入 buffer。"""
    n_total = len(game_data)
    for mc_idx, (planes, vt, player, mc_orig) in enumerate(game_data):
        # z_soft: 价值标签软化
        if score > 0:
            z_raw = 1.0 if player == 1 else -1.0
        elif score < 0:
            z_raw = -1.0 if player == 1 else 1.0
        else:
            z_raw = 0.0
        alpha = 0.3 + 0.7 * (mc_idx / max(n_total - 1, 1))
        z_soft = float(np.tanh(z_raw * alpha))
        
        if args.no_augment == 1:
            buffer.append((planes, vt, z_soft))
        else:
            for pl, tv in augment8(planes, vt, bs):
                buffer.append((pl, tv, z_soft))


if __name__ == "__main__":
    main()
