#!/usr/bin/env python3
"""AlphaZero 式自我对弈无监督训练（自对弈生成数据 → 训练 → 循环）。

每轮迭代：
  1. 自对弈 --games 局（MCTS visit 分布当 policy 标签，终局胜负当 value 标签，
     根先验混 Dirichlet 噪声 + 温度采样保证探索）
  2. 数据 8 对称增强（4 旋转 × 2 镜像，动作↔掩码同一置换）入 replay buffer
  3. PPO 更新 --epochs 轮（P3-C：裁剪代理目标 + k1 KL 惩罚 + β 自适应 + 信任域
     提前中止，读 --ppo-clip/--kl-coef/--kl-target；value 侧仍 BCE，语义错归 P3-D），
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
import math
import time
import queue
import struct
import traceback
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
def _temperature_sample(probs, mc):
    """温度衰减采样（P3-C 单一真相源）。返回 (mv, logp_old, p_norm)。

    p_norm = 温度作用后的归一化行为分布：mcts.search 返回的 probs 已含 MCTS
    构造时传入的 --temperature 级，这里再作用随手数线性衰减 1.0→0.1 的第二级，
    然后 np.random.choice(p_norm) 采样（全局 np.random，保持既有 RNG 序列不变）。

    logp_old = log p_norm[mv] —— P3-C 的 logp_old **必须**取这个两级温度后
    分布的 log-prob：记未加温度的 visit 分布会让 KL/ratio 记账系统性偏高
    （P3.0 报告 §5.5-1）。调用方 play 拒绝回退 pass 时，用返回的 p_norm 重算
    pass 的 log-prob（动作与 logp_old 必须锚定同一次采样分布）。

    退化分支：s <= 0（probs 全零，实践中不可达）→ 强制 pass、不消耗
    np.random.choice 的 RNG、log-prob 无定义记 0.0（与旧代码逐位一致）。
    """
    n_actions = len(probs)
    progress = min(1.0, mc / max(30, 1))
    temp = 1.0 - progress * (1.0 - 0.1)
    p = np.asarray(probs).reshape(-1).astype(np.float64)
    p[-1] = max(p[-1], 0.0)
    if temp > 0 and temp != 1.0:
        p = p ** (1.0 / temp)
    s = p.sum()
    if s <= 0:
        return n_actions - 1, 0.0, np.zeros(n_actions, dtype=np.float64)
    p_norm = p / s
    mv = int(np.random.choice(n_actions, p=p_norm))
    logp_old = float(np.log(p_norm[mv])) if p_norm[mv] > 0 else 0.0
    return mv, logp_old, p_norm


def self_play_game(ai, board_size, sims, max_moves, temperature,
                   expand_topk, expand_chunk, priors_leaf=True,
                   dir_alpha=0.3, dir_eps=0.25,
                   use_rollout=False, rollout_lambda=0.25, rollout_steps=None,
                   leaf_ab_depth=0, c_puct=2.0, virtual_loss=8.0,
                   num_threads=8, spec_prefetch=False,
                   use_diverse_rollout=False, vector_backup=True):
    """一局自对弈。返回 (data, score)，score = 终局分（黑-白）。

    data 行（P3-C 契约，7 元组）：
      (planes, action, logp_old, to_play, mc, root_value, mask)
      · planes:   (12,n,n) n_channels=12（见下方 ⚠ 注释）
      · action:   本手**实际落子**的动作（采样 mv，play 拒绝时回退 pass=n²），
                  是行为策略真正执行的动作 —— logp_old 的支撑点必须是它
      · logp_old: 两级温度后行为分布在 action 上的 log-prob（常量，停止梯度）
      · to_play:  执子方 ±1；mc: 手数；root_value: MCTS 根价值（to_play 视角）
      · mask:     合法动作掩码 (n²+1,)：get_legal_moves() 只有 n² 个（**无** pass
                  槽，见 go_rules.py），拼上恒合法的 pass 槽到 n²+1，与网络
                  logits 同宽 —— 路线图说的「bs*bs+1」由此而来
    旧 5 元组 (planes, vt, to_play, mc, root_value) 是 PPO 之前的布局，
    采集端统一为 7 元组；异步端见 async_pipeline._play_one_game（同一契约）。

    root_value 供 TD 价值标签做 n-step bootstrap（越界时回退终局值）。
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
        _visits, probs, root_value = mcts.search(
            board, hists[0], hists[1], to_play,
            simulations=sims, path_moves=path_moves)
        # 记录训练样本
        # ⚠ n_channels=12：自对弈采集的 buffer 喂 `in_channels=12` 的旧模型（见
        #   train_sft.py 的构造）。P4.3 起 feature_planes* 默认 17 通道，不显式钉住
        #   会让 buffer 里的 planes 与模型 in_channels 不匹配（形状错，不是静默错值）。
        #   ⚠ v21 训练要 17 通道时**这一格必须同步切**（P4.2/P4.8 的遗留项）。
        planes = np.ascontiguousarray(board.feature_planes_batched(
            board.board[None], [list(hists[0])], [list(hists[1])],
            [to_play], [board.ko_point], n_channels=12)[0])
        # 合法掩码必须在落子**前**取：n² 来自 get_legal_moves()（无 pass 槽），
        # 拼上恒合法的 pass 槽 → n²+1，与网络 logits 同宽（P3-C-a）
        mask = np.concatenate((legal, np.array([True])))
        # P3-C：温度衰减采样挪到 append 之前 —— action/logp_old 只有采样并
        # 落子之后才知道。RNG 与旧序一致（append 无 RNG，_temperature_sample
        # 内的 np.random.choice 即旧版同一调用）。
        mv, logp_old, p_norm = _temperature_sample(probs, mc)
        pmv = -1 if mv == n_actions - 1 else mv
        success = board.play(pmv)
        if not success:
            board.play(-1)
            pmv = -1
        # 实际走的着法才是行为策略「执行」的动作：play 拒绝时回退 pass，
        # logp_old 跟着重算 pass 的 log-prob，否则 action 与 logp_old 错配
        action = n_actions - 1 if pmv < 0 else pmv
        if action != mv:
            logp_old = float(np.log(p_norm[action])) if p_norm[action] > 0 else 0.0
        data.append((planes, int(action), logp_old, to_play, mc,
                     float(root_value), mask))
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


# --------------------------------------------------------------------------- #
# 参数映射：并行 worker 与串行共用的唯一真相源
# --------------------------------------------------------------------------- #
def _selfplay_kwargs(args, bs):
    """把 argparse Namespace 映射成自对弈一局所需的**全部**关键字参数。

    并行 worker（_selfplay_worker）与 main() 串行分支曾各手写一份参数映射，
    两者不一致且是**静默**分叉：worker 漏传 rollout_steps，落回本模块的函数签名
    默认 None（MCTS 语义 = 2*N*N 步），而 --rollout-steps 的 argparse 默认是 60
    —— 同一份 args，并行与串行跑出不同的对局（9 盘 162 步 vs 60 步，19 盘 722 步
    vs 60 步）。集中到本函数后，增删改参数只改一处。

    取值规则：全部经 getattr + **argparse 真实默认值**兜底（逐项对齐 main() 的
    argparse 默认）。两层理由：
      - getattr 兜底保留 worker 侧对残缺 Namespace 的容错（旧代码就有这层防御，
        子进程拿到的 args 若被上游裁剪过，不该直接 AttributeError）；
      - 兜底值与 argparse 默认保持一致，避免「残缺 Namespace 跑出的对局又与
        命令行不同」——那等于把刚合并的分叉换个地方复活。
    四个 0/1 标志统一 bool 归一化，两条路径类型一致。

    刻意**不**传 priors_leaf / dir_alpha / dir_eps：D10 已把它们钉死为自对弈
    函数的签名默认（True / 0.3 / 0.25），P3-B 会连同其他 MCTS 参数一起删除，
    现在不接线是刻意的，不是漏。
    """
    return {
        'board_size': bs,
        'sims': getattr(args, 'sims', 400),
        # --max-moves 默认 None → 3×点数（与旧两处写法逐字一致）
        'max_moves': getattr(args, 'max_moves', None) or 3 * bs * bs,
        'temperature': getattr(args, 'temperature', 1.0),
        'expand_topk': getattr(args, 'expand_topk', 64),
        'expand_chunk': getattr(args, 'expand_chunk', 0),
        'use_rollout': bool(getattr(args, 'use_rollout', 0)),
        'rollout_lambda': getattr(args, 'rollout_lambda', 0.25),
        'rollout_steps': getattr(args, 'rollout_steps', 60),
        'leaf_ab_depth': getattr(args, 'leaf_ab_depth', 2),
        'c_puct': getattr(args, 'c_puct', 2.0),
        'virtual_loss': getattr(args, 'virtual_loss', 8.0),
        # 实际生效的是 --mcts-threads；--num-threads（默认 8）是死参数
        'num_threads': getattr(args, 'mcts_threads', 3),
        'spec_prefetch': bool(getattr(args, 'spec_prefetch', 1)),
        'use_diverse_rollout': bool(getattr(args, 'use_diverse_rollout', 0)),
        'vector_backup': bool(getattr(args, 'mcts_vector_backup', 1)),
    }


def augment8(plane, target, n, action=None):
    """8 对称增强：4 旋转 × 2 镜像（全向量化，复用 GoBoard.apply_symmetry_batch）。

    旧版逐 t 8 次 np.rot90 + 8 次 Python 循环；新版把平面与目标棋盘各走一次
    分组向量化变换，尾部一次批量构造 target，消除全部逐样本循环。
    返回 8 个 (plane, target) 元组（顺序与旧版 t=0..7 一致）—— 传了 action
    时返回 8 个 (plane, target, action)，见下。

    P3-C：策略侧改成 PPO 后，增强必须**同时**置换动作与掩码（同一 D4 变换，
    路线图 P3-C-b「置换引擎只一个真相源」）。action 为 int 时走这条：
      · 棋盘动作（0..n²-1）：直接复用 apply_symmetry_batch 自带的 moves 通道
        （flat 坐标，按 SYMMETRIES[t] 变换）—— 与平面/掩码同一次调用、同一
        t，不另写变换代码；变换后动作天然仍落在变换后的合法掩码内；
      · pass 槽（n²）：进引擎前夹成 -1（引擎对 -1 不做任何变换），出引擎再
        还原 n²；
      · action=None：返回 2 元组，与旧行为逐位一致（legacy 对照测试钉死）。
    """
    plane = np.asarray(plane)
    ids = np.arange(8, dtype=np.int64)
    # moves 占位（-1 = pass 不参与变换）；给了 action 时 8 路同入一个动作，
    # 引擎按各自 t 变换后返回 moves_out
    tfs = np.full(8, -1, dtype=np.int64)
    if action is not None and int(action) < n * n:
        tfs[:] = int(action)
    planes_in = np.broadcast_to(plane, (8, plane.shape[0], n, n))
    planes_out, moves_out = GoBoard.apply_symmetry_batch(planes_in, tfs, ids, n)

    board_t = np.asarray(target[:n * n]).reshape(n, n)
    tb_in = np.broadcast_to(board_t, (8, 1, n, n))
    tb_out, _ = GoBoard.apply_symmetry_batch(tb_in, tfs, ids, n)  # (8,1,n,n)
    pass_t = target[n * n]

    # 一次批量构造 target：(8, n²+1) = 棋盘部分 flatten + 末尾 pass 列
    tv = np.concatenate(
        [tb_out[:, 0].reshape(8, n * n),
         np.full((8, 1), pass_t, dtype=tb_out.dtype)], axis=1)

    if action is None:
        return [(np.ascontiguousarray(planes_out[i]), np.ascontiguousarray(tv[i]))
                for i in range(8)]
    return [(np.ascontiguousarray(planes_out[i]), np.ascontiguousarray(tv[i]),
             (n * n if int(moves_out[i]) < 0 else int(moves_out[i])))
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
def _selfplay_worker(gid, model_path, args, result_queue):
    """单个自对弈进程。

    主体包 try/except BaseException：任何异常（模型加载失败 / OOM / board 断言）
    都回传 {'gid', 'error'}，父进程据此立即报错，而不是静默退出让父进程
    result_queue.get() 永久挂起（表现为「训练卡住」）。
    """
    try:
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
            ai, **_selfplay_kwargs(args, args.board_size))
    except BaseException:
        result_queue.put({'gid': gid, 'error': traceback.format_exc()})
        return
    result_queue.put({'gid': gid, 'data': game_data, 'score': score})


# --------------------------------------------------------------------------- #
# 多进程结果收集与子进程收尾
# --------------------------------------------------------------------------- #
def _reap_processes(processes, join_timeout=5.0):
    """收尾全部子进程：先 join(timeout)，仍存活的 terminate 后再无超时 join。

    保证任何异常路径都不泄漏子进程，也不会被 join() 无限阻塞。
    """
    for p in processes:
        p.join(timeout=join_timeout)
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()


def _poll_worker_results(result_queue, processes, on_result, gids=None,
                         poll_timeout=0.5, dead_grace=1.0):
    """带 timeout 轮询收集全部 worker 结果，交给 on_result 逐个处理。

    processes 必须**全部已 start**。on_result 必填，每收一局调用一次。
    gids 省略时按位置取 range(len(processes))；显式传入时长度必须与 processes
    一致，否则直接 ValueError —— zip 会静默截断并让末尾 worker 永远不被监控。

    get 超时后检查进程存活：只要有 worker「已退出却没交付结果」，先给队列
    dead_grace 秒的宽限（容忍「进程已退出」与「管道数据可见」之间的竞态；
    这是有界等待，最坏多等 dead_grace 秒，**不保证**挽回已丢失的 put），
    仍无结果即判定崩溃并抛 RuntimeError —— 绝不无限阻塞。
    """
    if gids is None:
        gids = range(len(processes))
    else:
        gids = list(gids)
        if len(gids) != len(processes):
            raise ValueError(
                f"gids 长度 {len(gids)} 与进程数 {len(processes)} 不一致，"
                f"拒绝静默截断（否则末尾 worker 永远不被监控）")
    total = len(processes)
    results = []
    delivered = set()

    while len(results) < total:
        try:
            result = result_queue.get(timeout=poll_timeout)
        except queue.Empty:
            # 还没死过（或已死的都交付过了）→ 继续等，绝不误报崩溃
            dead = next(((p, gid) for p, gid in zip(processes, gids)
                         if gid not in delivered and not p.is_alive()), None)
            if dead is None:
                continue
            p, dead_gid = dead
            try:
                result = result_queue.get(timeout=dead_grace)
            except queue.Empty:
                raise RuntimeError(
                    f"[selfplay] worker gid={dead_gid} 已退出（exitcode={p.exitcode}）"
                    f"但未交付结果，判定为崩溃；已收集 {len(results)}/{total} 局"
                ) from None
        if 'error' in result:
            raise RuntimeError(
                f"[selfplay] worker gid={result.get('gid')} 内部异常：\n"
                f"{result['error']}")
        results.append(result)
        delivered.add(result.get('gid'))
        on_result(result)
    return results


def _run_parallel_workers(result_queue, processes, on_result, gids=None,
                          poll_timeout=0.5, join_timeout=5.0, dead_grace=1.0):
    """start → 带 timeout 轮询收集 → finally 收尾，整段不泄漏子进程。

    只把**已成功 start** 的进程交给 _reap_processes：未 start 的进程 _popen
    为 None，join() 会触发 stdlib 断言「can only join a started process」，
    把真正的失败原因（高 --parallel-games 下的 EAGAIN / fd 耗尽）盖掉。
    """
    started = []
    try:
        for p in processes:
            p.start()
            started.append(p)
        return _poll_worker_results(result_queue, started, on_result, gids,
                                    poll_timeout, dead_grace)
    finally:
        _reap_processes(started, join_timeout)


# --------------------------------------------------------------------------- #
# 异步流水线队列收集
# --------------------------------------------------------------------------- #
def _pipeline_has_live_worker(pipeline):
    """判定异步流水线是否还有可能继续产出：stop_event 未置位且任一 worker 存活。

    只看 is_alive() 不够：stop() 置位 stop_event 后，仍在跑长对局的 worker 会继续
    is_alive() 好一阵（run() 的 while 只在每局之间检查 stop_event），但它绝不会再 put
    —— 继续等就是白等。两个条件任一不满足即视为「不再产出」。
    """
    if pipeline.stop_event.is_set():
        return False
    return any(w.is_alive() for w in pipeline.workers)


def _ensure_async_pipeline(holder, args, model_path=None, onnx_model=None):
    """按需构造并启动异步流水线，挂在 holder._pipeline 上，返回该流水线。

    holder 是持有 `_pipeline` 属性的对象（生产传 main）。`getattr(..., None) is None`
    才构造 —— 判据是「**本轮迭代还没有活着的流水线**」，不是「这个进程有没有建过」。

    为什么必须能重建：stop() 会置位 `stop_event` 并回收 worker，而
    `AsyncSelfPlayPipeline.start()` 只往 `self.workers` 里 append（不重置
    `stop_event`、不清空列表）。所以同一个对象**无法**二次 start()：新 worker
    一进 run() 的 while 就看到 stop_event 已置位而秒退。因此每轮迭代 stop 之后
    必须由 `_shutdown_async_pipeline` 把引用置 None，这里在下一轮重新构造。
    """
    from scripts.async_pipeline import AsyncSelfPlayPipeline

    pipeline = getattr(holder, '_pipeline', None)
    if pipeline is None:
        pipeline = AsyncSelfPlayPipeline(
            args, model_path=model_path, onnx_model=onnx_model)
        pipeline.start()
        holder._pipeline = pipeline
    return pipeline


def _shutdown_async_pipeline(holder):
    """停止流水线并解挂，使下一轮迭代能重建。无流水线时为空操作。"""
    pipeline = getattr(holder, '_pipeline', None)
    if pipeline is None:
        return
    try:
        pipeline.stop()
    finally:
        # 解挂是「下一轮重新构建」的前置条件：即使 stop() 自身抛错也必须置
        # None，否则残留的半死流水线会被下一轮当成「已启动」而复用。
        holder._pipeline = None


def _drain_queue_into_buffer(data_queue, buffer, args, bs, n_actions, max_stall=120,
                             alive_fn=None, max_wait=1800.0):
    """从异步流水线数据队列收集到 buffer 达标，返回本轮 collected 局数。

    AsyncDataQueue.get 超时会自行吞掉 queue.Empty 并返回 None。

    max_stall 度量的是「连续无产出 **且** 无存活 worker 的空轮询次数」，不是「慢」。
    仓库自己推荐的 19x19 / 400 sims 配置（见本文件 docstring）首局常需数分钟，
    加上各 worker 首次加载模型，慢而活着是常态 —— 纯「队列空」预算会误杀
    数小时的健康训练。

    - alive_fn 给了且返回 True（还有 worker 存活）→ 空轮询计数**重置**并继续等；
    - 只有计数达到 max_stall **且**（alive_fn is None 或 alive_fn() 为 False）
      才抛 RuntimeError，并带 collected / len(buffer) / 阈值 / max_stall 诊断；
    - alive_fn 为 None 时退化为纯「队列空」预算（仅适合已知无 worker 存活的场景，
      如单测，或调用点在流水线 start() 之前）。真实调用点传
      `_pipeline_has_live_worker`。

    max_wait 是与存活判定**正交**的最后一道兜底（绝对无产出预算，默认 30 分钟）：
    `SelfPlayWorker.run()` 的 `except Exception` 吞掉一切异常并无限重试
    （`_play_one_game` 与 MCTS 构造都在该 try 内），所以「每局必失败」的错误
    （--sims/--dir-alpha 等 MCTS 配置错误、规则层断言、MCTS 内 MemoryError）会让
    worker 永远 is_alive()、永远不 put —— 此时 stall 恒被重置、上一条判据永远
    不触发，等待退化成静默无限进行。只要有产出（哪怕很稀疏）就归零重计；
    超过 max_wait 即抛 RuntimeError，与 worker 是否存活无关。传 None 关闭该上限。
    """
    target = args.batch_size * 20
    collected = 0
    stall = 0
    silent_since = time.monotonic()
    while len(buffer) < target:
        item = data_queue.get(timeout=0.5)
        if item:
            _process_game_data(item['data'], item['score'], bs, n_actions, buffer, args)
            collected += 1
            stall = 0
            silent_since = time.monotonic()
            continue
        # 还有活着的 worker → 视为「慢而非死」，重置空轮询预算继续等
        if alive_fn is not None and alive_fn():
            stall = 0
        else:
            stall += 1
            if stall >= max_stall:
                raise RuntimeError(
                    f"[selfplay] 异步流水线连续 {stall} 次（≈{stall * 0.5:.0f}s）无产出且"
                    f"无存活 worker，判定已停止：collected={collected} "
                    f"len(buffer)={len(buffer)} / 目标 {target}（max_stall={max_stall}）。"
                    f"请检查自对弈 worker 是否崩溃（模型加载失败/OOM/board 断言）"
                    f"或流水线 stop() 是否已调用。")
        silent = time.monotonic() - silent_since
        if max_wait is not None and silent > max_wait:
            raise RuntimeError(
                f"[selfplay] 异步流水线已连续 {silent:.1f}s 一局都没产出"
                f"（max_wait={max_wait}s）：worker 存活但无产出，判定已卡死 —— "
                f"collected={collected} len(buffer)={len(buffer)} / 目标 {target}。"
                f"存活信号可信不代表会出数据：SelfPlayWorker.run() 的 except "
                f"Exception 会吞掉每局必失败的异常并无限重试（async_pipeline.py），"
                f"典型成因是 MCTS 配置错误（--sims/--dir-alpha 等）、规则层断言或 "
                f"MCTS 内 MemoryError。请核对这些参数与 worker 的 stderr。")
    return collected


# --------------------------------------------------------------------------- #
# 权重落盘
# --------------------------------------------------------------------------- #
def _save_checkpoint(ai, args, bs, it):
    """按训练格式把当前权重落盘，返回**实际写入的路径**。

    沿用既有命名规则：`args.out` 不以 `.pth` 结尾时补后缀。所以真实路径在
    `--out models/az` 时是 `models/az.pth` —— 与 `args.out` 不是同一个文件，
    返回真实路径而不是让调用方复用 `args.out`，收尾打印的「最佳权重」才是
    磁盘上真实存在的那个。
    """
    out_path = args.out if args.out.endswith('.pth') else args.out + '.pth'
    torch.save({"model": ai.model.state_dict(), "iter": it,
                "board_size": bs, "args": vars(args)}, out_path)
    return out_path


def _finalize_checkpoint(ai, args, bs, it, best_path, saved_any):
    """收尾保证「最佳权重」一定落盘，返回真实存在的路径。

    buffer 从未攒到训练阈值（`train_epochs` 一次都没跑）时一个权重文件都不存在，
    收尾却无条件打印「最佳权重: {best_path}」—— 下游 eval_elo.py 会拿到一个
    打不开的路径。宁可在这种情况下把当前权重（此时是初始权重）存下来，也不能
    宣称一个不存在的产物。

    saved_any 为真说明途中已经存过，原样返回 `best_path`（不再写盘，避免覆盖
    真正的最佳权重 —— 收尾这一刻的权重未必比最佳权重好）。
    """
    if saved_any:
        return best_path
    return _save_checkpoint(ai, args, bs, it)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
# β 自适应的硬夹紧边界（P3-C-c；路线图只给了更新式，未给边界值 —— 夹紧是防
# exp 溢出的必要护栏，报告里标注为规格歧义，两值经测试钉死方向与生效）
_KL_BETA_MIN = 1e-4
_KL_BETA_MAX = 10.0


def _standardize_advantage(adv, std_eps=1e-6):
    """A 按 minibatch 标准化：(A - mean) / max(std, std_eps)（P3-C-c）。

    std 夹紧防 0 除（全同批次 → 中心化后全 0 → 损失只剩 KL 项，不炸）。
    std 用总体标准差（ddof=0）：样本标准差（ddof=1）在 batch=1 时是 NaN，
    夹紧救不了 NaN。grad_accum>1 时优势标准化按**每个 micro-batch** 独立做
    （数据加载是逐 micro-batch 切的），这是显式取舍，报告里说明。
    """
    return (adv - adv.mean()) / torch.clamp(adv.std(unbiased=False), min=std_eps)


def _compute_advantage(z, v_old):
    """A = z - v_old，再按 minibatch 标准化（D13④ / 路线图 P3-C-c）。

    单独抽出来是为了让「减没减 v_old」这一格可被单测钉住：标准化对**仿射**
    变换不变，z 与 z-常数 只差一个常数会被吸收，但 v_old 逐样本不同 → 减掉
    它是实质性的，写在 train_epochs 内联时没有任何测试能证明它被读了。
    """
    return _standardize_advantage(z.reshape(-1) - v_old.reshape(-1))


def _ppo_policy_loss(logits, mask, action, logp_old, adv, clip_eps, kl_coef):
    """P3-C 策略侧损失：L = -E[min(r·A, clip(r,1-ε,1+ε)·A)] - β·KL_k1。

    参数：
      logits   (B, A_logits) 网络原始策略 logits（内部升 fp32 再掩码，
                             防 autocast fp16 下 -inf/softmax 精度坑）
      mask     (B, A_logits) bool 合法动作掩码（False处置 -inf → 概率 0）
      action   (B,) long      采集时实际落子的动作
      logp_old (B,)           行为策略在 action 上的 log-prob（两级温度后，
                             buffer 常量 / 停止梯度 —— ratio 与 KL 都锚定它，
                             **不是**当前网络输出）
      adv      (B,)           已标准化优势（_standardize_advantage）
      clip_eps / kl_coef      --ppo-clip / β（当前自适应值，非 argparse 初值）

    返回 (loss, stats)，stats = {'kl', 'clip_frac', 'entropy'}（detach 后的 float）。

    KL 用 k1 估计：KL_k1 = mean(logp_old - logp_new)[action] —— 采样式 k1，
    old 侧是常量，new 侧带梯度（惩罚把当前策略往行为策略拉）。mean 不是 sum
    （sum 会按 batch 放大 β 一个量级 —— 测试用封闭形式 KL 钉死）。
    """
    logits = logits.float()
    mask = mask.to(torch.bool)
    logp_new = F.log_softmax(logits.masked_fill(~mask, float('-inf')), dim=-1)
    logp_new_a = logp_new.gather(1, action.view(-1, 1)).squeeze(1)
    ratio = torch.exp(logp_new_a - logp_old)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -torch.min(surr1, surr2).mean() - kl_coef * (logp_old - logp_new_a).mean()
    with torch.no_grad():
        probs = logp_new.exp()
        # 掩码位置 logp=-inf → 先置 0 再乘（0·(-inf)=NaN），p 本身是 0，熵不受影响
        logp_ent = torch.where(mask, logp_new, torch.zeros_like(logp_new))
        stats = {
            'kl': float((logp_old - logp_new_a).mean().item()),
            'clip_frac': float(((ratio < 1.0 - clip_eps)
                                | (ratio > 1.0 + clip_eps)).float().mean().item()),
            'entropy': float((-(probs * logp_ent).sum(dim=-1).mean()).item()),
        }
    return loss, stats


def _adapt_kl_coef(beta, kl_obs, kl_target):
    """β 自适应（P3-C-c）：β ← clamp(β·exp((KL_obs - kl_target)/kl_target))。

    KL_obs > target → β↑（收紧），KL_obs < target → β↓（放松），clamp 到
    [_KL_BETA_MIN, _KL_BETA_MAX]。kl_target <= 0 时目标无意义，原样返回
    （防除零；--kl-target 默认 0.01 > 0）。指数先双边夹 ±700 再取 exp：
    (KL_obs-1e-2)/1e-2 可达数千，直接 exp 会溢出告警。
    """
    if kl_target <= 0:
        return float(beta)
    x = max(min((kl_obs - kl_target) / kl_target, 700.0), -700.0)
    return float(min(max(beta * math.exp(x), _KL_BETA_MIN), _KL_BETA_MAX))


def train_epochs(ai, buffer, args, device):
    """在 replay buffer 上训练 PPO 更新 --epochs 轮。返回平均 loss。

    P3-C：策略侧 = 裁剪代理目标 + k1 KL 惩罚（--ppo-clip / --kl-coef /
    --kl-target 三个 P3.0 参数在此真正被消费）；value 侧保持现状（BCE，
    语义错归 P3-D）。buffer 行 = 6 元组 (planes, action, logp_old, z, v_old, mask)。

    - A = z - v_old，按 minibatch 标准化（_standardize_advantage）；
    - 每次 optimizer step 后：β 自适应（窗口均值 KL 朝 --kl-target 走，状态挂
      ai._kl_beta 跨迭代持续，**不**每轮从 --kl-coef 重读），再检查
      running-KL（本轮累计均值）> 2×kl-target → 打印截断日志并 break **本轮**
      剩余 minibatch（外层 epochs 继续）；
    - 统计挂 ai._ppo_stats = {kl, clip_frac, entropy, kl_coef, early_stop, steps}
      供 main() 写 swanlab（缺省不动任何既有键）。

    N1: 910A 无 BF16，FP16 autocast 配 GradScaler 防下溢（对齐 train_sft 的
        npu_grad_scaler）；CUDA 旧卡 FP16 同样需要。
    C3: optimizer/EMA/GradScaler 跨迭代挂在 ai 上复用（保留 Adam 动量与 EMA
        shadow 轨迹），LR scheduler 每轮按当前 buffer 大小重建。
    N4: pin_memory 收窄为仅 CUDA（NPU 直传，对齐 train_sft）。
    """
    if buffer and len(buffer[0]) != 6:
        raise ValueError(
            f"buffer 行应为 P3-C 6 元组 (planes, action, logp_old, z, v_old, mask)，"
            f"实得 {len(buffer[0])} 元组 —— 3 元组 (planes, vt, z) 是 PPO 之前的布局，"
            f"请检查采集端是否还是旧 self_play_game")
    model = ai.model
    model.train()

    # P3-C：β 状态挂 ai（与 ai._opt/_ema 同生命周期）；--kl-coef 只是初值，
    # 首次创建后不再重读（接线契约 §5.2）
    if getattr(ai, '_kl_beta', None) is None:
        ai._kl_beta = float(getattr(args, 'kl_coef', 0.01))
    beta = float(ai._kl_beta)
    ppo_clip = float(getattr(args, 'ppo_clip', 0.2))
    kl_target = float(getattr(args, 'kl_target', 0.01))

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
        # 动作数取自掩码（buffer 行 index 5）：n²+1 与网络 logits 同宽；
        # 旧版从 vt（index 1）取，vt 已随 PPO 移除
        _acts = int(np.asarray(buffer[0][5]).shape[0])
        for _k in range(2):
            _ev = torch.cuda.Event()
            _ev.record()   # 显式置为已完成，首次复用该槽时等待立即返回
            _slots.append((
                torch.empty((_bs, 12, _n, _n), dtype=torch.float32, pin_memory=True),
                torch.empty((_bs, _acts), dtype=torch.bool, pin_memory=True),
                torch.empty((_bs,), dtype=torch.long, pin_memory=True),
                torch.empty((_bs,), dtype=torch.float32, pin_memory=True),
                torch.empty((_bs, 1), dtype=torch.float32, pin_memory=True),
                torch.empty((_bs, 1), dtype=torch.float32, pin_memory=True),
                _ev,
            ))
    _slot_i = 0

    accum_counter = 0
    # P3-C 统计（本调用累计 → ai._ppo_stats；空 buffer 提前返回时不动旧值）
    stat_kl = stat_clip = stat_ent = 0.0
    stat_steps = 0
    early_stop_any = False
    for epoch in range(args.epochs):
        opt.zero_grad()  # 每轮开始兜底清零（正常路径 step 后已清，防外部残留）
        # running-KL 逐 PPO epoch 重置：提前中止后下一轮仍有机会在 β 收紧后
        # 跑满整轮（截断是「本轮剩余」，不是整个 train_epochs 调用）
        epoch_kl = 0.0
        epoch_kl_n = 0
        win_kl = 0.0   # 自上次 β 自适应以来的窗口均值
        win_kl_n = 0
        for _ in range(steps_per_epoch * accum):
            idx = np.random.randint(0, n, size=min(args.batch_size, n))
            batch = [buffer[i] for i in idx]
            batch = [b for b in batch if b is not None]
            if not batch:
                continue

            if _slots:
                m = len(batch)
                sp, sm, sa, sl, sz, sv, ev = _slots[_slot_i]
                ev.synchronize()   # 只等本槽 H2D，不阻塞计算
                torch.from_numpy(np.stack([b[0] for b in batch])).copy_(sp[:m])
                torch.from_numpy(np.stack([b[5] for b in batch])).copy_(sm[:m])
                torch.from_numpy(np.asarray([b[1] for b in batch],
                                            dtype=np.int64)).copy_(sa[:m])
                torch.from_numpy(np.asarray([b[2] for b in batch],
                                            dtype=np.float32)).copy_(sl[:m])
                torch.from_numpy(np.asarray([b[3] for b in batch],
                                            dtype=np.float32).reshape(m, 1)).copy_(sz[:m])
                torch.from_numpy(np.asarray([b[4] for b in batch],
                                            dtype=np.float32).reshape(m, 1)).copy_(sv[:m])
                planes = sp[:m].to(device, non_blocking=True)
                mask_t = sm[:m].to(device, non_blocking=True)
                action_t = sa[:m].to(device, non_blocking=True)
                logp_old_t = sl[:m].to(device, non_blocking=True)
                z = sz[:m].to(device, non_blocking=True)
                v_old = sv[:m].to(device, non_blocking=True)
                ev.record()
                _slot_i ^= 1
            else:
                planes = torch.from_numpy(np.stack([b[0] for b in batch])).float()
                mask_t = torch.from_numpy(np.stack([b[5] for b in batch]))
                action_t = torch.from_numpy(np.asarray([b[1] for b in batch],
                                                       dtype=np.int64))
                logp_old_t = torch.from_numpy(np.asarray([b[2] for b in batch],
                                                         dtype=np.float32))
                z = torch.from_numpy(np.asarray([b[3] for b in batch],
                                                dtype=np.float32)).unsqueeze(1)
                v_old = torch.from_numpy(np.asarray([b[4] for b in batch],
                                                    dtype=np.float32)).unsqueeze(1)
                if pin_mem:
                    planes = planes.pin_memory()
                    mask_t = mask_t.pin_memory()
                    action_t = action_t.pin_memory()
                    logp_old_t = logp_old_t.pin_memory()
                    z = z.pin_memory()
                    v_old = v_old.pin_memory()
                planes = planes.to(device, non_blocking=pin_mem)
                mask_t = mask_t.to(device, non_blocking=pin_mem)
                action_t = action_t.to(device, non_blocking=pin_mem)
                logp_old_t = logp_old_t.to(device, non_blocking=pin_mem)
                z = z.to(device, non_blocking=pin_mem)
                v_old = v_old.to(device, non_blocking=pin_mem)

            with maybe_autocast(device):
                policy, value = model(planes)
                # P3-C-c：A = z - v_old，按 minibatch 标准化（std 夹紧防 0 除）
                adv = _compute_advantage(z, v_old)
                loss_pi, ppo = _ppo_policy_loss(
                    policy, mask_t, action_t, logp_old_t, adv, ppo_clip, beta)
                value_target = (z.squeeze(-1) + 1) / 2
                loss_v = F.binary_cross_entropy_with_logits(value.squeeze(-1), value_target)
                loss_raw = loss_pi + loss_v

            # 梯度累积：仅 backward，累积满 accum 才 step
            loss = loss_raw / accum if accum > 1 else loss_raw
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            stat_kl += ppo['kl']
            stat_clip += ppo['clip_frac']
            stat_ent += ppo['entropy']
            stat_steps += 1
            epoch_kl += ppo['kl']
            epoch_kl_n += 1
            win_kl += ppo['kl']
            win_kl_n += 1

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
            opt.zero_grad()  # 每次 step 后立即清零，防止跨 step 陈旧梯度叠加污染
            scheduler.step()
            if ema is not None:
                ema.update()
            losses.append(float(loss_raw.item()))

            # ---- P3-C：optimizer step 边界 —— 先 β 自适应，再信任域硬约束 ----
            kl_obs = win_kl / max(win_kl_n, 1)
            win_kl = 0.0
            win_kl_n = 0
            beta = _adapt_kl_coef(beta, kl_obs, kl_target)
            ai._kl_beta = beta
            running_kl = epoch_kl / max(epoch_kl_n, 1)
            if running_kl > 2.0 * kl_target:
                print(f"[ppo] 信任域提前中止：running-KL {running_kl:.6f} > "
                      f"2×kl-target {2.0 * kl_target:.6f}，跳过本轮剩余 minibatch"
                      f"（epoch {epoch + 1}/{args.epochs}，β={beta:g}）", flush=True)
                early_stop_any = True
                break

    # 尾部不足一个累积周期的残留梯度：丢弃（不 step），避免半成品梯度污染权重
    if accum_counter > 0:
        opt.zero_grad()

    # P3-C：统计挂 ai 供 main() 写 swanlab（缺省不动任何既有键）
    ai._ppo_stats = {
        'kl': stat_kl / max(stat_steps, 1),
        'clip_frac': stat_clip / max(stat_steps, 1),
        'entropy': stat_ent / max(stat_steps, 1),
        'kl_coef': beta,
        'early_stop': early_stop_any,
        'steps': stat_steps,
    }

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
    ap.add_argument("--epochs", type=int, default=2,
                    help="PPO 更新轮数：每轮迭代把 replay buffer 走几遍 PPO 更新"
                         "（原「每轮迭代训练遍数」；语义改写于 P3.0，"
                         "PPO 裁剪+KL 由 P3-C 接入，value 侧由 P3-D 接入）")
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
    ap.add_argument("--rollout-steps", type=int, default=60, help="rollout 最大步数")
    
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
                    help="设备选择：auto/cuda/npu/cpu")
    
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

    # ---- PPO / KL（路线图 D13）----
    # P3.0 **只声明不消费**：本组三个参数在损失里的精确读法、β 自适应状态挂在哪、
    # 2×target 提前中止读哪个值，都写在
    # .superpowers/sdd/2026-09-25-v21-roadmap/task-p3-0-report.md 的「接线契约」一节。
    # 策略侧（PPO 裁剪代理目标 + KL 惩罚）由 P3-C 接入，value 侧（value clipping）由
    # P3-D 接入；**在它们落地前这三个参数不改变任何数值行为**。
    ap.add_argument("--ppo-clip", type=float, default=0.2,
                    help="PPO 裁剪范围 ε：r=exp(logp_new-logp_old)，代理目标 "
                         "L_pi=-min(r·A, clip(r,1-ε,1+ε)·A)。"
                         "P3-D 的 value clipping 复用同一个 ε，不另设参数")
    ap.add_argument("--kl-coef", type=float, default=0.01,
                    help="KL 惩罚系数 β 的**初值**（不是常数）：L_pi -= β·KL_k1，"
                         "P3-C 另做 β 自适应朝 --kl-target 走，"
                         "自适应状态挂 ai 上（与 optimizer/EMA 同生命周期），"
                         "不每轮从本参数重读")
    ap.add_argument("--kl-target", type=float, default=0.01,
                    help="目标 KL。P3-C 两处都读它：① β 自适应的调节目标；"
                         "② 每次更新后 running-KL > 2×本值 时提前中止本轮剩余 "
                         "minibatch（信任域硬约束，2× 是写死的因子，不另设参数）")

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
                        # P3-C：三个 PPO/KL 旋钮（kl_coef 是 β 初值，实际 β 随
                        # 自适应走，逐 iter 见日志 kl_coef…仅以 kl 三键落盘）
                        "ppo_clip": args.ppo_clip,
                        "kl_coef": args.kl_coef,
                        "kl_target": args.kl_target,
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
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # 最佳权重追踪
    best_loss = float('inf')
    best_path = args.out      # 首次落盘后由 _save_checkpoint 的返回值覆盖
    saved_any = False         # 是否真的存过盘 —— 收尾兜底的条件
    it = 0                    # --iters 0 时 for 循环不执行，收尾兜底仍要用到迭代号

    buffer = []   # [(planes(12,n,n), action, logp_old, z, v_old, mask(n²+1))] P3-C 6 元组
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
            # 异步模式：生成与训练并行（仅主进程驱动流水线；非主进程等待）
            if is_main:
                print(f"[async] 使用异步流水线模式", flush=True)

                # 每轮迭代按需（重新）构建流水线：上一轮 stop() 已解挂并置位
                # stop_event，同一个对象无法二次 start()
                pipeline = _ensure_async_pipeline(
                    main, args, model_path=args.model, onnx_model=args.onnx_model)

                # 持续收集数据并训练
                for _ in range(args.games_per_iter or 10):
                    # 收集数据（无产出 **且** 无存活 worker 连续 max_stall 次 → 报错；
                    # worker 还活着就继续等，慢启动的 19x19/400 sims 不会被误杀；
                    # 另有绝对无产出上限 max_wait=1800s 兜底「活着却一局都不出」）
                    collected = _drain_queue_into_buffer(
                        pipeline.data_queue, buffer, args, bs, n_actions,
                        alive_fn=lambda: _pipeline_has_live_worker(pipeline))
                    total_games += collected

                    if collected > 0 and is_main:
                        print(f"  收集 {collected} 局，buffer={len(buffer)}", flush=True)

                    # 训练（如果有足够数据）
                    if len(buffer) >= args.batch_size * 5:
                        # P3-C：async 模式的 logp_old 不是「当前策略」的采样分布 ——
                        # worker 用 args.model 在本轮迭代构造时冻结，保存新权重不
                        # 回灌（P3.0 报告 §5.5-5）。PPO 容忍 off-policy 起点，但
                        # ratio 起点≠1、2×KL 可能首步即触发 → 一次性告警，不静默。
                        if not getattr(main, '_ppo_async_warned', False):
                            main._ppo_async_warned = True
                            print("[ppo] 警告：async 流水线下 logp_old 来自冻结的"
                                  "生成策略（off-policy），PPO 信任域与 KL 记账"
                                  "口径退化；建议 --async-pipeline 0", flush=True)
                        avg_loss = train_epochs(ai, buffer, args, device)

                        # 保存最佳
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            best_path = _save_checkpoint(ai, args, bs, it)
                            saved_any = True
                            print(f"[iter {it}] NEW BEST loss={avg_loss:.4f}", flush=True)

                        # 清空 buffer
                        buffer = []

                # 停止流水线并解挂（下轮迭代由 _ensure_async_pipeline 重新构建）
                _shutdown_async_pipeline(main)
        else:
            # 同步模式：原有逻辑
            if args.parallel_games > 1 and args.ddp == 0:
                    # 多进程并行生成
                    # N2: Linux 下 fork 会复制主进程已初始化的 NPU/CUDA 上下文导致挂死，
                    # 显式用 spawn context（Windows 本就是 spawn，无行为变化）
                    ctx = mp.get_context('spawn')
                    result_queue = ctx.Queue(maxsize=args.result_queue_max)
                    processes = [ctx.Process(target=_selfplay_worker,
                                            args=(g, args.model, args, result_queue))
                                 for g in range(args.parallel_games)]

                    def _on_worker_result(result):
                        """单局结果入 buffer。worker 崩溃由 _poll_worker_results 抛。"""
                        nonlocal total_games
                        game_data = result['data']
                        score = result['score']
                        _process_game_data(game_data, score, bs, n_actions, buffer, args)
                        total_games += 1
                        if is_main:
                            print(f"  [game {result['gid']+1}] score={score:+.1f} "
                                  f"moves={len(game_data)} buffer={len(buffer)}", flush=True)

                    # 带 timeout 轮询收集；worker 崩溃立即报错，finally 收尾不泄漏子进程
                    _run_parallel_workers(result_queue, processes, _on_worker_result)
            else:
                # 串行生成
                for g in range(args.games):
                    game_data, score = self_play_game(
                        ai, **_selfplay_kwargs(args, bs))
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
                    best_path = _save_checkpoint(ai, args, bs, it)
                    saved_any = True
                    print(f"[iter {it}/{args.iters}] NEW BEST loss={avg_loss:.4f} "
                          f"-> {best_path}", flush=True)

                print(f"[iter {it}/{args.iters}] loss={avg_loss:.4f} buffer={len(buffer)} "
                      f"games={total_games} {dt:.0f}s", flush=True)
                # SwanLab 记录迭代指标
                if swanlab_logger is not None:
                    z_arr = np.asarray([b[3] for b in buffer[:min(len(buffer), 4096)]])
                    swanlab_log = {
                        "iter_loss": avg_loss,
                        "iter_games": total_games,
                        "buffer_size": len(buffer),
                        "iter_time_s": dt,
                        "games_per_iter": collected if 'collected' in locals() else 0,
                        "td/z_mean": float(z_arr.mean()),
                        "td/z_std": float(z_arr.std()),
                        "td/enabled": args.td,
                    }
                    # P3-C：策略侧统计（缺省 ai 上没有 _ppo_stats 时不加键，
                    # 任何既有键都不动）
                    _ppo_stats = getattr(ai, '_ppo_stats', None)
                    if _ppo_stats:
                        swanlab_log.update({
                            "kl": _ppo_stats["kl"],
                            "clip_frac": _ppo_stats["clip_frac"],
                            "entropy": _ppo_stats["entropy"],
                        })
                    swanlab.log(swanlab_log, step=it)

                # 同步模式：每轮训练完成后重置 buffer（训完即清，避免旧局
                # 与新网络视角的 TD 标签混用；与 async 分支清空语义一致）
                buffer.clear()

    if is_main:
        # 兜底：一个权重都没存过（buffer 从未攒到训练阈值）时也必须落一个文件，
        # 否则下面这句「最佳权重」指向磁盘上不存在的东西
        best_path = _finalize_checkpoint(ai, args, bs, it, best_path, saved_any)
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
    """处理一局自对弈数据（TD 价值标签 + 8 对称增强），加入 buffer。

    P3-C 行契约：
      输入行 7 元组 (planes, action, logp_old, to_play, mc, root_value, mask) ——
        采集端 self_play_game / async _play_one_game 共用（旧 5 元组
        (planes, vt, to_play, mc, root_value) 直接拒收，提示跑旧采集端）；
      输出行 6 元组 (planes, action, logp_old, z, v_old, mask) ——
        z 由 compute_td_target 现算（index 3），v_old = root_value（index 5
        采集行原值），action/logp_old/mask 原样。
    增强时动作与掩码走 augment8 的同一 D4 置换（动作亦被置换，见 augment8）。
    """
    if game_data and len(game_data[0]) != 7:
        raise ValueError(
            f"采集行应为 P3-C 7 元组 (planes, action, logp_old, to_play, mc, "
            f"root_value, mask)，实得 {len(game_data[0])} 元组 —— 旧 5 元组是 "
            f"PPO 之前的采集布局（self_play_game / async _play_one_game 需同步升级）")
    td = getattr(args, 'td', 0) == 1
    td_steps = getattr(args, 'td_steps', 3)
    td_ai = getattr(args, 'td_alpha_init', 0.2)
    td_ae = getattr(args, 'td_alpha_end', 0.9)
    players = np.asarray([row[3] for row in game_data])
    root_values = np.asarray([row[5] for row in game_data])
    for mc_idx, row in enumerate(game_data):
        planes, action, logp_old, mask = row[0], int(row[1]), float(row[2]), row[6]
        z, _z_raw, _alpha = compute_td_target(
            players, root_values, score, mc_idx,
            td, td_steps, td_ai, td_ae)
        v_old = float(root_values[mc_idx])
        if getattr(args, 'no_augment', 0) == 1:
            buffer.append((planes, action, logp_old, z, v_old, mask))
        else:
            for pl, mk, ac in augment8(planes, mask, bs, action=action):
                buffer.append((pl, int(ac), logp_old, z, v_old, mk))


if __name__ == "__main__":
    main()
