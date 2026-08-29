"""LightPLS：轻量策略 + 轻量价值 + 快速 rollout。

在 MCTS 叶子节点上，除了主网络给出的 (prior P, value v) 之外，再用一个
**轻量策略网络 (FastPolicy)** 做短程随机走子（rollout），用 Tromp-Taylor
快速数子得到终局胜率，作为叶子的补充价值。这能在不强 RL 的前提下显著提升
搜索深度与棋力，而单次 rollout 成本远低于一次网络前向（尤其 9 路）。

LightPLS 含义：
- Light Policy：FastPolicy 轻量启发式（纯 numpy），作为 rollout 的走子策略
- Light Value：rollout 终局数子得到的确定性胜负信号
- 融合：叶子最终 value = (1-λ)·v_net + λ·v_rollout
"""

import numpy as np

from src.game.go_rules import GoBoard


class FastPolicy:
    """极轻量走子策略：对合法点做档位打分，按 softmax 采样。

    纯 numpy 启发式（靠近已有棋子、避免送吃），速度极快，可作为 rollout 的
    轻量策略。若传入 weights（长度 12 或 12 个 channel 的线性权重）则叠加使用。
    """

    def __init__(self, board_size: int, weights: np.ndarray = None,
                 temperature: float = 1.0):
        self.n = board_size
        self.weights = weights.astype(np.float32) if weights is not None else None
        self.temperature = float(temperature)
        self._atari_penalty = 1.2

    def logits(self, board: GoBoard) -> np.ndarray:
        n = self.n
        a = n * n
        logit = np.zeros(a, dtype=np.float32)
        b = board.board
        legal = board.get_legal_moves()
        # 已有棋子邻接奖励（靠近战斗）：对 4 邻接做 1 次膨胀求和
        occ = (b != 0).astype(np.float32)
        neigh = np.zeros((n, n), dtype=np.float32)
        neigh[1:, :]   += occ[:-1, :]    # 上方
        neigh[:-1, :]  += occ[1:, :]     # 下方
        neigh[:, 1:]   += occ[:, :-1]    # 左方
        neigh[:, :-1]  += occ[:, 1:]     # 右方
        my_atari = board.feature_planes([], [])[10].reshape(-1)
        for i in range(a):
            if not legal[i]:
                logit[i] = -1e9
                continue
            logit[i] = 0.3 * neigh.reshape(-1)[i] - self._atari_penalty * my_atari[i]
        if self.weights is not None:
            fp = board.feature_planes([], []).reshape(12, -1)
            if self.weights.ndim == 1 and self.weights.shape[0] == 12:
                logit = logit + np.tensordot(self.weights, fp, axes=([0], [0])).reshape(-1).astype(np.float32)
        # 追加 pass 着法（索引 n*n），logit=0
        logit = np.append(logit, 0.0)
        return logit

    def sample_move(self, board: GoBoard, rng: np.random.Generator) -> int:
        logit = self.logits(board) / max(self.temperature, 1e-3)
        logit = logit - logit.max()
        p = np.exp(logit)
        s = p.sum()
        if s <= 0 or not np.isfinite(s):
            return board.board_size * board.board_size  # pass
        p = p / s
        return int(rng.choice(board.board_size * board.board_size + 1, p=p))


def light_rollout(board: GoBoard, policy: "FastPolicy",
                  max_steps: int = None,
                  rng: np.random.Generator = None) -> float:
    """从当前局面用轻量策略随机走子到终局，返回**发起方 (current_player) 视角**
    的 Tromp-Taylor 胜率 (+1 胜 / -1 负 / 0 平)。
    """
    if rng is None:
        rng = np.random.default_rng()
    n = board.board_size
    if max_steps is None:
        max_steps = n * n * 2
    # 只读推演：深拷贝，避免污染调用方棋盘
    cur = GoBoard(n)
    cur.board = board.board.copy()
    cur.current_player = board.current_player
    cur.ko_point = board.ko_point
    cur.passes = board.passes
    cur.move_history = list(board.move_history)
    initiator = cur.current_player  # 发起方（黑=1/白=-1）
    passes = 0
    steps = 0
    while steps < max_steps and passes < 2:
        mv = policy.sample_move(cur, rng)
        if mv == n * n:  # pass
            cur.play(-1)
            passes += 1
        else:
            ok = cur.play(mv)
            if not ok:
                cur.play(-1)
                passes += 1
            else:
                passes = 0
        steps += 1
    score = cur.score()  # 黑 - 白 目数
    return _tt_value(score, initiator)


def _tt_value(score: float, initiator: int) -> float:
    """Tromp-Taylor 数子结果（黑-白目数）转为发起方视角胜率。"""
    if score > 0:
        return 1.0 if initiator > 0 else -1.0
    if score < 0:
        return -1.0 if initiator > 0 else 1.0
    return 0.0
