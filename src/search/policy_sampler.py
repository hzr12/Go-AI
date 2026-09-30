"""RL 自对弈的走子器：**N 步 minimax 推演采样**，不构造 MCTS。

背景（2026-09-30 的 RL 改造）
------------------------------
RL 采集原来每手跑 `MCTS.search(simulations=sims)`：1 卡 910A 上 `--sims 48`
意味着**每手约 48 次前向**，占 RL 算力的绝大部分，而 8 进程 × 3 线程的 CPU 搜索
还要互相抢核。改造的方向是把 RL 的落子器换成「一次根前向 + 一次批量推演」，
**MCTS 引擎保留给 webui / cli_play / evaluate / eval_elo**（那些入口照旧）。

本模块承担三件事，恰好是原来 `MCTS.search` 在采集侧提供的三件
------------------------------------------------------------------
1. **给出一个覆盖全部合法动作的行为分布 q**（推演价值派生 + 与根策略混合）；
2. **从 q 采样**，并同时给出 PPO/B2 需要的两个 log-prob：
   - `logq`     = log q(action)      —— 重要性权重 w = π_θold/q 的分母；
   - `logp_old` = log π_θold(action) —— ratio 的分母（**网络策略族**，不是 q）。
   两者不同源是形态 B 的固有事实（MCTS 时代同样如此，只是错配更大），B2 的
   重要性权重就是为这件事存在的；
3. **返回该局面的网络估值**（根局面的 value，to_play 视角）= 标准 PPO 里的
   `v_old = V_θ_old(s)`，写进 buffer 第 5 列，替代原来的 MCTS 根价值。

为什么 q 必须**满支撑**（不能只在 top-K 候选上有质量）
---------------------------------------------------
若 q 只覆盖推演出的 top-K，其余合法点概率为 0，那么**策略永远学不到那些点**
（梯度里永远不出现）—— 采样器能走的集合会随模型变弱而单调缩小，早期剪掉的
坏点再也回不来。所以 q = (1-mix)·推演分布 + mix·根策略：推演信号主导，同时给
全部合法点留一条非零通道。`mix` 默认 0.1，可按探索强度调。

温度
----
两级（与搬迁前的 `_temperature_sample` 逐字相同）：先随手数线性衰减
1.0→0.1 的第二级温度，再作用在 q 上。它在这个模块里而不是调用方，因为
「采样分布 = 温度作用后的分布」是本模块的契约，而 `logq` 必须取**同一次**
采样分布上的 log-prob（动作与 logq 锚定同一个分布，否则重要性权重无意义）。
"""
from typing import NamedTuple, Optional

import numpy as np

from src.search import lookahead as lookahead_mod

#: q 里分给「根策略」的混合质量（保证满支撑）。见模块 docstring。
DEFAULT_MIX = 0.1


class MoveSample(NamedTuple):
    """一次采样的全部产物（字段名即契约）。

    planes   (C,n,n) float32，本手的特征平面 —— **与前瞻共用同一次计算**，
             调用方直接把它写进 buffer，不必再算一遍
    action   实际采到的动作（含 pass = n_actions-1）
    logq     log q(action)，q = 温度作用后的行为分布
    logp_old log π_θold(action)，π_θold = 根 masked policy（**不带温度**：
             ratio 的两侧必须同族，温度只作用在采样分布上）
    value    根局面的网络估值（to_play 视角）= V_θ_old(s)
    mask     (n_actions,) bool 合法动作掩码（n² 个点 + 恒合法的 pass 槽）
    probs    (n_actions,) 温度作用后的 q，调用方要复查分布时用
    policy   (n_actions,) **不带温度**的根 masked policy π_θold。必须单独给出：
             play 拒绝把动作回退成 pass 时，调用方要按**同一个 π** 重算
             `log π(pass)` —— 拿 `probs`（那是 q）算就等于把 B2 的重要性权重
             悄悄从 π/q 退化成 1，PPO 直接有偏。
    """

    planes: np.ndarray
    action: int
    logq: float
    logp_old: float
    value: float
    mask: np.ndarray
    probs: np.ndarray
    policy: np.ndarray


def temperature_sample(probs, mc, rng=None):
    """温度衰减采样（搬迁前的 `_temperature_sample`，逐字未改）。

    返回 (mv, logp_old, p_norm)：
      - `p_norm` 是**温度作用后**的归一化行为分布；
      - `logp_old` = log p_norm[mv]，必须是这个分布上的 log-prob；
      - 退化分支（p 全零，实践中不可达）强制 pass、不消耗 RNG、log-prob 记 0.0。

    默认仍用全局 `np.random`（保持既有 RNG 序列不变）；传 `rng` 给测试一个确定的
    采样源（`np.random.Generator` 或 `RandomState` 都有 `.choice`）。
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
    chooser = np.random if rng is None else rng
    mv = int(chooser.choice(n_actions, p=p_norm))
    logp_old = float(np.log(p_norm[mv])) if p_norm[mv] > 0 else 0.0
    return mv, logp_old, p_norm


def behavior_distribution(values, masked_policy, n_actions, lookahead_temp,
                          mix=DEFAULT_MIX):
    """由推演价值构造**满支撑**的行为分布 q（温度作用之前）。

    Args:
        values: {根候选着法: 根玩家视角价值}（`lookahead` 的返回值）。
        masked_policy: 根 masked policy（非法点 0、pass 恒合法）。
        n_actions: bs*bs+1。
        lookahead_temp: 价值 → 概率的 softmax 温度（τ_v）。0.2 是 webui hybrid
            沿用的档位（`webui.py` 的 `softmax((ks - ks.max()) / 0.2)`）。
        mix: 分给根策略的质量（见模块 docstring 的「满支撑」）。

    Returns:
        (n_actions,) float64，已归一；非法点恒为 0。
    """
    pi = np.asarray(masked_policy, dtype=np.float64).reshape(-1).copy()
    pi[pi < 0] = 0.0
    tot = pi.sum()
    pi = pi / tot if tot > 0 else np.full(n_actions, 1.0 / max(n_actions, 1))

    if values:
        ks = np.full(n_actions, -np.inf)
        for m, v in values.items():
            ks[m] = float(v)
        finite = np.isfinite(ks)
        if finite.any():
            ex = np.exp((ks[finite] - ks[finite].max()) / max(lookahead_temp, 1e-6))
            q = np.zeros(n_actions)
            q[finite] = ex / ex.sum()
            mix = float(min(max(mix, 0.0), 1.0))
            q = (1.0 - mix) * q + mix * pi
            s = q.sum()
            return q / s if s > 0 else pi
    return pi


def sample_move(ai, board, my_hist, op_hist, to_play, topk=12, width=4, depth=2,
                lookahead_temp=0.2, mix=DEFAULT_MIX, mc=0, rng=None) -> MoveSample:
    """推演 + 采样一步。返回 `MoveSample`（字段见其 docstring）。

    Args:
        ai: `GoAI`（只用 `predict_batch` 与 `in_channels`）。
        board / my_hist / op_hist / to_play: 当前局面（`board` 只 play/undo）。
        topk / width / depth: 推演参数（与 webui 的 `--policy-topk/width/depth`
            同一套语义，默认 12/4/2）。
        lookahead_temp: 价值 → 概率的 τ_v（默认 0.2）。
        mix: q 里分给根策略的质量（满支撑，默认 0.1）。
        mc: 当前手数（温度衰减用）。
        rng: 采样源（默认全局 np.random；测试传 Generator）。

    ⚠ 调用方负责：把 `sample.action` 落到盘上，若 `board.play()` 拒绝（理论上
    不会：q 只在合法点上有质量，但 pass 槽与 ko 规则仍可能拒绝），要回退 pass
    并**用同一个分布**重算 logq（既有约定，见 `selfplay_train` 的采样段）。
    """
    n = board.board_size * board.board_size + 1
    n_channels = getattr(ai, 'in_channels', 12)
    planes = np.ascontiguousarray(board.feature_planes_batched(
        board.board[None], [list(my_hist)], [list(op_hist)], [to_play],
        [board.ko_point], n_channels=n_channels)[0], dtype=np.float32)
    legal = board.get_legal_moves()
    mask = np.concatenate((legal, np.array([True])))

    res = lookahead_mod.lookahead(
        ai, board, my_hist, op_hist, to_play, n_actions=n, n_channels=n_channels,
        topk=topk, width=width, depth=depth, planes=planes)

    q0 = behavior_distribution(res.values, res.policy, n, lookahead_temp, mix)
    mv, _, p_norm = temperature_sample(q0, mc, rng=rng)

    pi = np.asarray(res.policy, dtype=np.float64).reshape(-1)
    pis = pi.sum()
    pi = pi / pis if pis > 0 else np.full(n, 1.0 / max(n, 1))
    logp_old = float(np.log(pi[mv])) if pi[mv] > 0 else 0.0
    logq = float(np.log(p_norm[mv])) if p_norm[mv] > 0 else 0.0

    return MoveSample(planes=planes, action=int(mv), logq=logq,
                      logp_old=logp_old, value=float(res.root_value),
                      mask=mask, probs=p_norm, policy=pi)