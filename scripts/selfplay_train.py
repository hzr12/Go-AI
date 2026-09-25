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
import multiprocessing as mp

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
                   use_diverse_rollout=False, vector_backup=True):
    """一局自对弈。返回 [(planes, visit_target, player, mc, root_value), ...], score(黑-白)。

    root_value: 该步 MCTS 根节点价值（当前 to_play 视角, [-1,1]，见 mcts.search 返回
    第 3 项），供 TD 价值标签做 n-step bootstrap（越界时回退终局值）。
    """
    mcts = MCTS(ai, board_size=board_size, num_threads=num_threads,
                expand_topk=expand_topk, expand_chunk=expand_chunk,
                priors_leaf=priors_leaf, temperature=temperature,
                dirichlet_alpha=dir_alpha, dirichlet_eps=dir_eps,
                spec_prefetch=spec_prefetch,
                use_rollout=use_rollout, rollout_lambda=rollout_lambda,
                rollout_steps=rollout_steps,
                leaf_ab_depth=leaf_ab_depth,
                c_puct=c_puct, virtual_loss=virtual_loss,
                vector_backup=vector_backup)
    
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
        visits, probs, root_value = mcts.search(
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
        data.append((planes, vt, to_play, mc, float(root_value)))
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
    """8 对称增强：4 旋转 × 2 镜像（全向量化，复用 GoBoard.apply_symmetry_batch）。

    旧版逐 t 8 次 np.rot90 + 8 次 Python 循环；新版把平面与目标棋盘各走一次
    分组向量化变换，尾部一次批量构造 target，消除全部逐样本循环。
    返回 8 个 (plane, target) 元组（顺序与旧版 t=0..7 一致）。
    """
    plane = np.asarray(plane)
    ids = np.arange(8, dtype=np.int64)
    tfs = np.full(8, -1, dtype=np.int64)  # 着法索引不用（pass），仅驱动 transform_ids
    planes_in = np.broadcast_to(plane, (8, plane.shape[0], n, n))
    planes_out, _ = GoBoard.apply_symmetry_batch(planes_in, tfs, ids, n)

    board_t = np.asarray(target[:n * n]).reshape(n, n)
    tb_in = np.broadcast_to(board_t, (8, 1, n, n))
    tb_out, _ = GoBoard.apply_symmetry_batch(tb_in, tfs, ids, n)  # (8,1,n,n)
    pass_t = target[n * n]

    # 一次批量构造 target：(8, n²+1) = 棋盘部分 flatten + 末尾 pass 列
    tv = np.concatenate(
        [tb_out[:, 0].reshape(8, n * n),
         np.full((8, 1), pass_t, dtype=tb_out.dtype)], axis=1)
    return [(np.ascontiguousarray(planes_out[i]), np.ascontiguousarray(tv[i]))
            for i in range(8)]


# --------------------------------------------------------------------------- #
# TD (C+E) 价值标签
# --------------------------------------------------------------------------- #
def compute_td_target(players, root_values, score, t,
                      td, td_steps, td_alpha_init, td_alpha_end):
    """计算数据下标 t 处的价值标签 z（纯函数，selfplay/async 两处共享）。

    players:     (n_total,) 每个数据位置执子方（±1）
    root_values: (n_total,) 每个数据位置 MCTS 根价值（该位置 to_play 视角）
    score:       终局分（黑-白，>0 黑胜）
    t:           当前数据下标

    返回 (z, z_raw, alpha)：
      z_raw  = 终局胜负（t 处 player 视角, ±1/0）
      r_soft = 旧位置软化标签 tanh(z_raw·(0.3+0.7·t/T))
      td 关闭 → z = r_soft（与旧公式逐位一致，回归保证）
      td 开启 → v_td = sign·root_values[t+td_steps]（越界回退 z_raw；sign 按
                 players[t] vs players[t+td_steps]，兼容被跳过的无气 pass）
                 alpha = td_alpha_init + (td_alpha_end-td_alpha_init)·t/T
                 z = clip(alpha·v_td + (1-alpha)·r_soft, -1, 1)
    """
    player = players[t]
    n_total = len(players)
    if score > 0:
        z_raw = 1.0 if player == 1 else -1.0
    elif score < 0:
        z_raw = -1.0 if player == 1 else 1.0
    else:
        z_raw = 0.0
    T = max(n_total - 1, 1)
    r_soft = float(np.tanh(z_raw * (0.3 + 0.7 * (t / T))))
    if not td:
        return r_soft, z_raw, 0.0
    t2 = t + td_steps
    if t2 >= n_total:
        v_td = z_raw
    else:
        sign = 1.0 if players[t2] == player else -1.0
        v_td = sign * float(root_values[t2])
    alpha = td_alpha_init + (td_alpha_end - td_alpha_init) * (t / T)
    z = float(np.clip(alpha * v_td + (1.0 - alpha) * r_soft, -1.0, 1.0))
    return z, z_raw, alpha


# --------------------------------------------------------------------------- #
# 并行自对弈 worker
# --------------------------------------------------------------------------- #
def _resolve_selfplay_device(args):
    """解析自对弈 worker 的推理设备，并拦住会崩的组合。

    根因（云端实测）：worker 原先直接沿用训练设备（--device 默认 auto → npu），
    而全仓库没有 torch.npu.set_device()。于是 --parallel-games N 会产生 N+1 个
    进程（父进程训练 + N 个 worker），每个持有独立 CANN 上下文却全部落在
    NPU 0 上，并发创建 matmul primitive 时报
    "could not create a primitive descriptor for a matmul primitive"。

    因此：自对弈默认走 CPU，父进程独占加速器训练。NN 仅占搜索约 17%，而
    树/规则/rollout 本就是 CPU 密集，24 个 CPU 核本来闲置。
    """
    device = getattr(args, 'selfplay_device', None) or 'cpu'
    if device == 'auto':
        device = _auto_select_device()
    is_accel = device.startswith('npu') or device.startswith('cuda')
    if is_accel and getattr(args, 'parallel_games', 1) > 1:
        raise RuntimeError(
            '自对弈设备 {} 与 --parallel-games {} 组合会崩溃：每个 worker 会各自'
            '建立一个加速器运行时上下文，且仓库不做设备绑定，全部落在同一张卡上'
            '并发创建算子 primitive，实测报 "could not create a primitive '
            'descriptor for a matmul primitive"。请改用 --selfplay-device cpu'
            '（推荐，父进程独占 {} 训练），或把 --parallel-games 降到 1。'
            .format(device, args.parallel_games, getattr(args, 'device', 'auto')))
    return device


def _selfplay_worker(gid, model_path, args, result_queue):
    """单个自对弈进程。"""
    # torch 默认用 os.cpu_count() 个 intra-op 线程；N 个 worker 会变成
    # N×核数 个线程抢核，必须钉为 1（理由同 async_pipeline 的 worker 入口）。
    try:
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    device = _resolve_selfplay_device(args)
    
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
        rollout_steps=getattr(args, 'rollout_steps', None),
        leaf_ab_depth=getattr(args, 'leaf_ab_depth', 2),
        c_puct=getattr(args, 'c_puct', 2.0),
        virtual_loss=getattr(args, 'virtual_loss', 8.0),
        num_threads=getattr(args, 'mcts_threads', 3),  # 使用新的参数名
        spec_prefetch=getattr(args, 'spec_prefetch', False),
        use_diverse_rollout=getattr(args, 'use_diverse_rollout', False),
        vector_backup=getattr(args, 'mcts_vector_backup', 1) == 1
    )
    result_queue.put({'gid': gid, 'data': game_data, 'score': score})


def _on_worker_game(worker_id, game_data, score):
    """异步 worker 每完成一局时的回调。

    必须是**模块级**函数：worker 现以 spawn 启动，spawn 会 pickle 整个
    worker 实例（含 progress_cb），局部闭包无法序列化，会在启动时炸掉。

    此前 progress_cb 一直传 None，加上收集循环也不打印，worker 侧与主
    进程双双毫无输出——用户无法区分"在算"与"已死"。
    """
    try:
        n = len(game_data)
    except TypeError:
        n = -1
    print("  [worker {}] 完成一局：{} 样本 score={:+.1f}".format(
        worker_id, n, score), flush=True)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train_epochs(ai, buffer, args, device):
    """在 replay buffer 上训练若干遍。返回平均 loss。

    N1: 910A 无 BF16，FP16 autocast 配 GradScaler 防下溢（对齐 train_sft 的
        npu_grad_scaler）；CUDA 旧卡 FP16 同样需要。
    C3: optimizer/EMA/GradScaler 跨迭代挂在 ai 上复用（保留 Adam 动量与 EMA
        shadow 轨迹），LR scheduler 每轮按当前 buffer 大小重建。
    N4: pin_memory 收窄为仅 CUDA（NPU 直传，对齐 train_sft）。
    """
    model = ai.model
    model.train()

    device_prefix = device.split(':')[0] if isinstance(device, str) else str(device)

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
    opt_groups = [
        {'params': other_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': other_no_decay, 'lr': args.lr, 'weight_decay': 0.0},
        {'params': value_decay, 'lr': args.lr * args.value_lr_mult,
         'weight_decay': args.weight_decay},
        {'params': value_no_decay, 'lr': args.lr * args.value_lr_mult, 'weight_decay': 0.0},
    ]
    # C3: 首轮创建 optimizer/scaler/EMA 并缓存到 ai，后续迭代复用
    if getattr(ai, '_opt', None) is None:
        ai._opt = torch.optim.AdamW(opt_groups)
        if device_prefix == 'npu' and hasattr(torch, 'npu'):
            try:
                ai._scaler = torch.npu.amp.GradScaler(enabled=True)
            except Exception:
                ai._scaler = None
        elif device_prefix == 'cuda':
            if hasattr(torch.amp, 'GradScaler'):
                ai._scaler = torch.amp.GradScaler('cuda', enabled=True)
            else:
                ai._scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            ai._scaler = None
        if args.use_ema == 1:
            ai._ema = EMA(model, decay=0.999)
    opt = ai._opt
    scaler = getattr(ai, '_scaler', None)
    ema = getattr(ai, '_ema', None) if args.use_ema == 1 else None

    # Cosine LR Schedule with Warmup（每轮按 buffer 重建）
    n = len(buffer)
    # 梯度累积：accum 个 micro-batch 才做一次 opt.step，有效 step 数相应变少
    accum = max(1, int(getattr(args, 'grad_accum_steps', 1) or 1))
    steps_per_epoch = max(1, n // (args.batch_size * accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.10))
    after_warmup = max(1, total_steps - warmup_steps)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=after_warmup)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    losses = []
    if not buffer:
        return 0.0

    # N4: pin_memory 仅 CUDA（NPU 直传，对齐 train_sft 的 cuda-only pin 策略）
    pin_mem = device_prefix == 'cuda'

    # G1: 双缓冲 H2D —— 两个预分配 pinned 槽交替使用：槽 A 异步搬运到 device 期间，
    # CPU 侧填充槽 B。覆写某槽前**无条件**等待该槽自己的 CUDA Event——它是「本槽
    # H2D 拷贝已完成」的唯一可靠信号。
    #   · 不能用 current_stream().synchronize()：那会连同计算一起等，重叠收益归零；
    #   · 不能用「计算是否已消费」之类的标志短路：device 张量被 forward/backward
    #     读完后，pinned 源的 H2D 拷贝仍可能在途，短路会导致覆写仍在搬运的源缓冲。
    # 仅动训练循环的数据搬运，不触碰 self-play 数据生成 / 数据集。
    _bs = min(args.batch_size, n) if n > 0 else 0
    _slots = []
    if pin_mem and _bs > 0:
        _n = int(np.asarray(buffer[0][0]).shape[-1])
        _acts = int(np.asarray(buffer[0][1]).shape[0])
        for _k in range(2):
            _ev = torch.cuda.Event()
            _ev.record()   # 显式置为已完成，首次复用该槽时等待立即返回
            _slots.append((
                torch.empty((_bs, 12, _n, _n), dtype=torch.float32, pin_memory=True),
                torch.empty((_bs, _acts), dtype=torch.float32, pin_memory=True),
                torch.empty((_bs, 1), dtype=torch.float32, pin_memory=True),
                _ev,
            ))
    _slot_i = 0

    accum_counter = 0
    for _ in range(args.epochs):
        opt.zero_grad()  # 每轮开始清零（原先每 batch 清零，会丢弃首 batch 梯度）
        for _ in range(steps_per_epoch * accum):
            idx = np.random.randint(0, n, size=min(args.batch_size, n))
            batch = [buffer[i] for i in idx]
            batch = [b for b in batch if b is not None]
            if not batch:
                continue

            if _slots:
                m = len(batch)
                sp, spi, sz, ev = _slots[_slot_i]
                ev.synchronize()   # 只等本槽 H2D，不阻塞计算
                torch.from_numpy(np.stack([b[0] for b in batch])).copy_(sp[:m])
                torch.from_numpy(np.stack([b[1] for b in batch])).copy_(spi[:m])
                torch.from_numpy(np.stack([b[2] for b in batch])
                                .reshape(m, 1)).copy_(sz[:m])
                planes = sp[:m].to(device, non_blocking=True)
                pi_t = spi[:m].to(device, non_blocking=True)
                z = sz[:m].to(device, non_blocking=True)
                ev.record()
                _slot_i ^= 1
            else:
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
                value_target = (z.squeeze(-1) + 1) / 2
                loss_v = F.binary_cross_entropy_with_logits(value.squeeze(-1), value_target)
                loss_raw = loss_pi + loss_v

            # 梯度累积：仅 backward，累积满 accum 才 step
            loss = loss_raw / accum if accum > 1 else loss_raw
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_counter += 1
            if accum_counter < accum:
                continue

            accum_counter = 0
            if scaler is not None:
                if args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
            else:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                opt.step()
            scheduler.step()
            if ema is not None:
                ema.update()
            losses.append(float(loss_raw.item()))

    # 尾部不足一个累积周期的残留梯度：丢弃（不 step），避免半成品梯度污染权重
    if accum_counter > 0:
        opt.zero_grad()

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


def resolve_c2net_model(pretrain_model_path):
    """从 C2NET 上下文的预训练目录确定性地挑一个 ``.pth`` 权重。

    必须 ``sorted`` 后再取第一个：glob 的返回顺序依赖文件系统，直接取
    ``[0]`` 会导致同一目录多次运行可能选中不同权重。目录为空 / 无 ``.pth`` /
    路径为假值时返回 None（调用方据此不覆盖 --model）。
    """
    if not pretrain_model_path:
        return None
    import glob
    pth_files = sorted(glob.glob(os.path.join(pretrain_model_path, '*.pth')))
    return pth_files[0] if pth_files else None


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
    ap.add_argument("--batch-size", type=int, default=256,
                    help="训练 batch（C2: NPU 甜点 256，显存约 2-3x 旧 64）")
    ap.add_argument("--epochs", type=int, default=2, help="每轮迭代训练遍数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-lr-mult", type=float, default=0.5,
                    help="value head 学习率倍率")
    ap.add_argument("--expand-topk", type=int, default=8,
                    help="MCTS 展开候选截断。默认 8（实测工作点）：每次模拟要为"
                         "每个候选各跑一次 rollout，该值直接乘在每模拟成本上。"
                         "注意 dynamic-topk 生效时实际取 min(阶段上限, 本值)，"
                         "故本值为 8 时全程恒为 8，搜索后期的精细化阶段被取消；"
                         "若要保留部分后期爬升可设 16（8→16→16）。"
                         "调大会显著变慢：64 时 9 路实测仅 2.72 sims/s")
    ap.add_argument("--expand-chunk", type=int, default=0,
                    help="展开期 α-β 界截断块大小（0=关闭）")
    
    # MCTS 质量参数
    ap.add_argument("--c-puct", type=float, default=2.0, help="PUCT 探索系数")
    ap.add_argument("--virtual-loss", type=float, default=8.0, help="虚拟损失系数")
    ap.add_argument("--num-threads", type=int, default=8, help="MCTS 多线程数")
    ap.add_argument("--spec-prefetch", type=int, default=1, choices=[0, 1],
                    help="启用 worker 推测预评估 (0=关闭, 1=开启；C7 默认 1)")
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
    ap.add_argument("--rollout-steps", type=int, default=30,
                    help="rollout 最大步数。默认 30（实测工作点）：rollout 是纯 "
                         "Python 规则推演，开销随步数线性增长，且与 expand-topk "
                         "相乘。旧默认 60；不传时 MCTS 会退化为 2*N*N"
                         "（19 路 722 步，超支约 12 倍）")
    
    # 并行生成
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
                    help="训练设备选择：auto/cuda/npu/cpu")
    ap.add_argument("--selfplay-device", default="cpu",
                    help="自对弈 worker 的推理设备：默认 cpu。父进程用 --device "
                         "独占加速器训练，worker 走 CPU——NPU 只占搜索约 17%，"
                         "而树/规则/rollout 本就 CPU 密集。⚠ 不要设为 npu/cuda "
                         "配合 --parallel-games>1：每个 worker 会各建一个加速器"
                         "运行时上下文且仓库不做设备绑定，全部挤同一张卡并发创建"
                         "算子 primitive，实测报 'could not create a primitive "
                         "descriptor for a matmul primitive'")
    
    ap.add_argument("--no-augment", type=int, default=0, choices=[0, 1],
                    help="关闭 8 对称增强 (0=开启, 1=关闭)")
    ap.add_argument("--use-ema", type=int, default=0, choices=[0, 1],
                    help="启用 EMA 权重 (0=关闭, 1=开启)")
    ap.add_argument("--clip-grad", type=float, default=1.0, help="梯度裁剪范数（0=关闭）")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="梯度累积 micro-batch 数（1=每批即更新，默认 1 保持现状；"
                         ">1 时等效 batch×N、step÷N，建议同时把 --lr 调高 1.4~2 倍）")
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
    ap.add_argument("--mcts-vector-backup", type=int, default=1, choices=[0, 1],
                    help="MCTS 回传 visit/value_sum 走 numpy 批量更新"
                         "(1=默认，与逐层循环数值等价；0=回退原实现)")
    
    # 异步流水线
    ap.add_argument("--async-pipeline", type=int, default=0, choices=[0, 1],
                    help="启用异步流水线（生成与训练并行，需配合 --games-per-iter）(0=关闭, 1=开启)")
    ap.add_argument("--games-per-iter", type=int, default=10,
                    help="异步模式下每轮迭代生成的局数")
    ap.add_argument("--swanlab", type=int, default=0, choices=[0, 1],
                    help="启用 SwanLab 实验跟踪 (0=关闭, 1=开启)")
    ap.add_argument("--swanlab-api-key", type=str, default="",
                    help="SwanLab API key（可选，未设置则读取 SWANLAB_API_KEY 环境变量）")
    ap.add_argument("--ver", default="rl",
                    help="模型版本号（SwanLab name 后缀）")
    # TD Learning（C+E 混合价值标签）
    ap.add_argument("--td", type=int, default=1, choices=[0, 1],
                    help="TD 价值标签 (0=旧 tanh 软化, 1=开启)")
    ap.add_argument("--td-steps", type=int, default=3,
                    help="TD n-step 前看步数（数据下标空间）")
    ap.add_argument("--td-alpha-init", type=float, default=0.2,
                    help="TD α 调度初值（开局，偏 r_soft）")
    ap.add_argument("--td-alpha-end", type=float, default=0.9,
                    help="TD α 调度终值（残局，偏 v_td）")
    ap.add_argument("--c2net", type=int, default=0, choices=[0, 1],
                    help="启用 C2NET (OpenI 启智平台) 支持 (0=关闭, 1=开启)")

    args = ap.parse_args()

    # DDP/多卡：rank/world_size/is_main（c2net、swanlab 块均引用 is_main，须先定义）
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    is_main = (rank == 0)

    # ---- C2NET 支持（OpenI 启智平台）----
    # 关于 rank 守卫：prepare() 与 --model 覆盖**必须**在所有 rank 上执行——
    # 每个 rank（含各 _selfplay_worker 子进程所在的 rank）都要独立解析初始权重路径。
    # 故这里只对「日志打印」做 is_main 过滤（避免 N 卡刷出 N 份重复日志）。
    # 若 c2net 的 prepare() 未来被发现有写盘/建连副作用，正确做法是挪到
    # init_process_group 之后由 rank 0 调用再广播路径，而非简单加 if is_main。
    _c2net_ctx = None
    if args.c2net == 1:
        try:
            from c2net.context import prepare, upload_output as _c2net_upload
            _c2net_ctx = prepare()
            if is_main:
                print("[c2net] 已初始化 C2NET 上下文", flush=True)
                print(f"[c2net] output_path={_c2net_ctx.output_path}", flush=True)
                print(f"[c2net] pretrain_model_path={_c2net_ctx.pretrain_model_path}", flush=True)
            # 覆盖 --model：从 c2net 预训练模型目录加载（仅在未显式指定 --model 时）
            # 直接运行 selfplay_train.py 时 --model 默认为 None，此分支会生效，
            # 故必须用 resolve_c2net_model 做确定性选择。
            if not args.model:
                _c2net_model = resolve_c2net_model(_c2net_ctx.pretrain_model_path)
                if _c2net_model:
                    args.model = _c2net_model
                    if is_main:
                        print(f"[c2net] 使用预训练模型: {args.model}", flush=True)
        except ImportError:
            if is_main:
                print("[c2net] c2net 未安装，--c2net 已忽略", flush=True)

    # SwanLab 实验跟踪（可选）
    use_swanlab = args.swanlab == 1 or os.environ.get('SWANLAB_API_KEY')
    swanlab_logger = None
    if use_swanlab and is_main:
        # 刻意**不在训练进程内** pip install swanlab：调用点在 torch / torch_npu
        # 已加载之后，此时改动 site-packages 可能破坏后续惰性导入；且 shell/*.sh
        # 在启动 python 之前已装过一次，那次失败的话这里必然也失败，只是白等一轮。
        # 「手动安装没问题」正是这个差别：装在解释器启动前，依赖已就位。
        #
        # 用 find_spec 区分「没装」与「装了但坏」：后者是云端常见坑——swanlab 依赖
        # pydantic>=2，而 MindSpore / torch_npu 常把 pydantic 钉在 1.x，于是
        # `import swanlab` 抛 "cannot import name 'TypeAdapter' from 'pydantic'"。
        # 旧代码把任何 ImportError 都当成「未安装」而误触发自动安装，掩盖了真因。
        _mod = sys.modules.get('swanlab')
        _have = _mod is not None
        if not _have:
            import importlib.util
            try:
                _have = importlib.util.find_spec('swanlab') is not None
            except (ImportError, ValueError):
                _have = False
        if not _have:
            print("[swanlab] 未安装，已跳过跟踪（指标请看 stdout 日志）。"
                  "请在启动训练前安装: pip install swanlab", flush=True)
        else:
            try:
                import swanlab
                # 登录：优先用 --swanlab-api-key，其次环境变量，最后交互式
                api_key = args.swanlab_api_key or os.environ.get('SWANLAB_API_KEY')
                if api_key:
                    swanlab.login(api_key=api_key, save=True)
                    print("[swanlab] API key 已设置，自动登录", flush=True)
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
                        "td": args.td,
                        "td_steps": args.td_steps,
                        "td_alpha_init": args.td_alpha_init,
                        "td_alpha_end": args.td_alpha_end,
                    },
                )
                swanlab_logger = swanlab
                print("[swanlab] 实验跟踪已启用", flush=True)
            except Exception as e:
                print(f"[swanlab] 初始化失败: {e}", flush=True)
                _emsg = str(e)
                if 'pydantic' in _emsg or 'TypeAdapter' in _emsg:
                    print("[swanlab] 疑似 pydantic 版本冲突：swanlab 需要 pydantic>=2，"
                          "而 MindSpore / torch_npu 常钉 pydantic<2。"
                          "请在启动训练前解决版本冲突（如在独立环境装 swanlab），"
                          "不要依赖训练进程内自动安装。", flush=True)

    # 设备选择
    if args.device == "auto":
        device = _auto_select_device()
    else:
        device = args.device
    
    torch.manual_seed(42)
    np.random.seed(42)

    # C2NET 输出路径重定向
    if _c2net_ctx is not None and is_main:
        _c2net_out = os.path.join(_c2net_ctx.output_path, os.path.basename(args.out))
        print(f"[c2net] 输出路径已重定向: {args.out} -> {_c2net_out}", flush=True)
        args._c2net_orig_out = args.out
        args.out = _c2net_out

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

    buffer = []   # [(planes(12,n,n), target(n²+1), z)]
    total_games = 0

    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()

        if is_main:
            print(f"\n[iter {it}/{args.iters}] 开始自对弈...", flush=True)

        # SwanLab 记录迭代开始
        if swanlab_logger is not None:
            swanlab.log({"iter_start": it}, step=it)

        # 检查是否启用异步流水线（块必须在 for it 循环体内逐迭代执行；
        # 旧版此块在循环外只跑一轮且 buffer 不清空，是"共 0 局"的根因之一）
        if args.async_pipeline == 1:
            from scripts.async_pipeline import AsyncSelfPlayPipeline
            # 异步模式：生成与训练并行（仅主进程驱动流水线；非主进程等待）
            if is_main:
                print(f"[async] 使用异步流水线模式", flush=True)

                # 启动异步流水线（如果尚未启动）
                if not hasattr(main, '_pipeline') or main._pipeline is None:
                    main._pipeline = AsyncSelfPlayPipeline(
                        args, model_path=args.model, onnx_model=args.onnx_model,
                        progress_cb=_on_worker_game)
                    main._pipeline.start()

                # 持续收集数据并训练
                for _ in range(args.games_per_iter or 10):
                    # 收集数据
                    collected = 0
                    # 收集目标只需够训练一轮。原先是 batch_size*20，比训练实际
                    # 需要的 batch_size*5 多等 4 倍数据才起步；而 19 路自对弈
                    # 一局约 250 样本，5120 样本需 15~25 局，CPU MCTS 下首次
                    # 输出前可能静默十几分钟——看起来完全像卡死。
                    _need = max(1, args.batch_size * 5)
                    _t_wait = time.time()
                    _last_report = _t_wait
                    while len(buffer) < _need:
                        item = main._pipeline.data_queue.get(timeout=0.5)
                        if item:
                            game_data = item['data']
                            score = item['score']
                            _process_game_data(game_data, score, bs, n_actions, buffer, args)
                            collected += 1
                            total_games += 1
                            continue
                        # 没拿到数据：必须区分「还在算」与「worker 全死了」。
                        # 原先这里没有任何检查，worker 全死也会无限静默空转。
                        _alive = sum(1 for w in main._pipeline.workers
                                     if w.is_alive())
                        if _alive == 0:
                            from scripts.async_pipeline import _describe_exitcode
                            _errs = main._pipeline.data_queue.stats['errors'].value
                            _why = ', '.join(
                                '#{} {}'.format(i, _describe_exitcode(w.exitcode))
                                for i, w in enumerate(main._pipeline.workers))
                            raise RuntimeError(
                                '所有自对弈 Worker 均已退出（存活 {}/{}，'
                                '累计 worker 错误 {} 次）。\n'
                                '各 worker 终态: {}\n'
                                'Python 异常会打印 traceback 到 stderr 并累加'
                                '上面的错误计数；若计数为 0 而终态是「被信号杀死」，'
                                '则是原生崩溃，请查子进程 stderr 上的 faulthandler 输出。'
                                .format(_alive, len(main._pipeline.workers),
                                        _errs, _why))
                        _now = time.time()
                        if _now - _last_report >= 15.0:
                            _st = main._pipeline.data_queue.stats
                            print("  [collect] buffer={}/{} 队列={} "
                                  "已产/已取={}/{} 丢弃={} 存活Worker={}/{} "
                                  "等待={:.0f}s".format(
                                      len(buffer), _need,
                                      main._pipeline.data_queue.qsize(),
                                      _st['produced'].value,
                                      _st['consumed'].value,
                                      _st['errors'].value,
                                      _alive, len(main._pipeline.workers),
                                      _now - _t_wait), flush=True)
                            _last_report = _now

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

                # 停止流水线（本轮内停止，下轮迭代重新按需启动）
                main._pipeline.stop()
        else:
            # 同步模式：原有逻辑
            if args.parallel_games > 1 and args.ddp == 0:
                    # 多进程并行生成
                    # N2: Linux 下 fork 会复制主进程已初始化的 NPU/CUDA 上下文导致挂死，
                    # 显式用 spawn context（Windows 本就是 spawn，无行为变化）
                    ctx = mp.get_context('spawn')
                    result_queue = ctx.Queue(maxsize=args.result_queue_max)
                    processes = []
                    for g in range(args.parallel_games):
                        p = ctx.Process(target=_selfplay_worker,
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
                        use_diverse_rollout=args.use_diverse_rollout == 1,
                        vector_backup=args.mcts_vector_backup == 1)
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
                        z_arr = np.asarray([b[2] for b in buffer[:min(len(buffer), 4096)]])
                        swanlab.log({
                            "iter_loss": avg_loss,
                            "iter_games": total_games,
                            "buffer_size": len(buffer),
                            "iter_time_s": dt,
                            "games_per_iter": collected if 'collected' in locals() else 0,
                            "td/z_mean": float(z_arr.mean()),
                            "td/z_std": float(z_arr.std()),
                            "td/enabled": args.td,
                        }, step=it)

                # 同步模式：每轮训练完成后重置 buffer（训完即清，避免旧局
                # 与新网络视角的 TD 标签混用；与 async 分支清空语义一致）
                buffer.clear()

    if is_main:
        print(f"训练完成。共 {total_games} 局，最佳权重: {best_path}")
        print("用 scripts/eval_elo.py 对比不同迭代权重棋力。")
        # SwanLab 结束
        if swanlab_logger is not None:
            swanlab.log({"total_games": total_games, "final_iter": args.iters}, step=args.iters)
            swanlab.finish()

    # C2NET 回传结果
    if _c2net_ctx is not None and is_main:
        try:
            from c2net.context import upload_output as _c2net_upload
            _c2net_upload()
            print("[c2net] 结果已回传到 OpenI 平台", flush=True)
        except Exception as e:
            print(f"[c2net] 回传失败: {e}", flush=True)


def _process_game_data(game_data, score, bs, n_actions, buffer, args):
    """处理一局自对弈数据（TD 价值标签 + 8 对称增强），加入 buffer。"""
    td = getattr(args, 'td', 0) == 1
    td_steps = getattr(args, 'td_steps', 3)
    td_ai = getattr(args, 'td_alpha_init', 0.2)
    td_ae = getattr(args, 'td_alpha_end', 0.9)
    players = np.asarray([row[2] for row in game_data])
    # 兼容旧 4 元组数据（无 root_value）：缺省 0.0
    root_values = np.asarray([row[4] if len(row) > 4 else 0.0
                              for row in game_data])
    for mc_idx, row in enumerate(game_data):
        planes, vt = row[0], row[1]
        z, _z_raw, _alpha = compute_td_target(
            players, root_values, score, mc_idx,
            td, td_steps, td_ai, td_ae)
        if args.no_augment == 1:
            buffer.append((planes, vt, z))
        else:
            for pl, tv in augment8(planes, vt, bs):
                buffer.append((pl, tv, z))


if __name__ == "__main__":
    main()
