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

import logging

import numpy as np

from src.game.go_rules import GoBoard

logger = logging.getLogger(__name__)


def _fast_atari_mask(board: GoBoard, player) -> np.ndarray:
    """返回 (n*n,) bool：player 方「仅 1 口气」的棋子位置（打吃）。

    用单次稀疏 BFS 对每个同色连通块计数自由点，替代完整 feature_planes
    （后者每步都做多层 Python flood-fill）。rollout 中每步调用也极快。
    player: 棋子值（1 或 -1，= board.current_player）。
    """
    n = board.board_size
    b = board.board
    occ = (b == player)
    atari = np.zeros((n, n), dtype=bool)
    visited = np.zeros((n, n), dtype=bool)
    seeds = np.argwhere(occ)
    if seeds.size == 0:
        return atari.reshape(-1)
    nb = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for (y, x) in seeds:
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        group = []
        libs = set()
        while stack:
            cy, cx = stack.pop()
            group.append((cy, cx))
            for dy, dx in nb:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < n and 0 <= nx < n:
                    v = b[ny, nx]
                    if v == 0:
                        libs.add((ny, nx))
                    elif v == player and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(libs) == 1:
            for (gy, gx) in group:
                atari[gy, gx] = True
    return atari.reshape(-1)


class FastPolicy:
    """极轻量走子策略：对合法点做档位打分，按 softmax 采样。

    纯 numpy 启发式（靠近已有棋子、避免送吃），速度极快，可作为 rollout 的
    轻量策略。若传入 weights（长度等于 `n_channels` 的线性权重）则叠加使用。

    `n_channels` 是**特征构造的通道数**（P4.13b）：默认 12 保住了「没有模型的
    独立调用方」（webui / selfplay_train / light_rollout.__main__）手里那份
    12 维权重向量的旧布局；MCTS 则把自己的（模型驱动的）通道数显式传进来。
    """

    def __init__(self, board_size: int, weights: np.ndarray = None,
                 temperature: float = 1.0, n_channels: int = 12):
        self.n = board_size
        self.weights = weights.astype(np.float32) if weights is not None else None
        self.temperature = float(temperature)
        self.n_channels = int(n_channels)
        self._atari_penalty = 1.2
        # P4.13b 起 n_channels 由调用方（MCTS 按所挂模型）传入，权重维数与它
        # 不一致是契约违约：权重项会被跳过（见 logits），特征仍按 n_channels
        # 造。这种不一致必须**响亮**——本任务的整条 thesis 就是「没有静默的
        # 错通道行为」，故构造期告警一次（不抛：抛会打断「权重项不参与」的
        # 既有契约，且仓库内暂无任何调用方传 weights，属潜在而非现实风险）。
        if self.weights is not None and self.weights.shape[0] != self.n_channels:
            logger.warning(
                "FastPolicy 权重维数 %d 与 n_channels=%d 不一致：权重项不参与 "
                "logits（特征仍按 %d 通道造，绝不回退 12）",
                self.weights.shape[0], self.n_channels, self.n_channels)

    def logits(self, board: GoBoard) -> np.ndarray:
        n = self.n
        b = board.board
        legal = board.get_legal_moves()
        # 已有棋子邻接奖励（靠近战斗）：对 4 邻接做 1 次膨胀求和
        occ = (b != 0).astype(np.float32)
        neigh = np.zeros((n, n), dtype=np.float32)
        neigh[1:, :]   += occ[:-1, :]    # 上方
        neigh[:-1, :]  += occ[1:, :]     # 下方
        neigh[:, 1:]   += occ[:, :-1]    # 左方
        neigh[:, :-1]  += occ[:, 1:]     # 右方
        # 打吃惩罚：用廉价 BFS 计算「当前方处于打吃（仅 1 口气）的棋子」，
        # 替代 board.feature_planes([], [])[10]——后者每步都做完整 Python flood-fill
        # 特征，是 rollout 的主要性能杀手（每局 60 步 × 每步 2 次特征 ≈ 万次调用）。
        my_atari = _fast_atari_mask(board, board.current_player)
        # 向量化：合法点 = 0.3*邻子数 - penalty*打吃；非法点 = -1e9。
        # 旧实现为 n² 次 Python 标量循环，且循环内反复 reshape，rollout 每步都调用。
        logit = (0.3 * neigh.reshape(-1)
                 - self._atari_penalty * my_atari).astype(np.float32)
        logit = np.where(legal, logit, np.float32(-1e9)).astype(np.float32)
        if self.weights is not None:
            # 仅在显式提供线性权重时才计算完整特征（默认 FastPolicy 不触发）
            # ⚠ `n_channels=nc`、下面的 `reshape(nc, -1)` 与
            #   `weights.shape[0] == nc` 是同一个契约的三个面（P4.13b 起三者
            #   同为 `self.n_channels`，由 MCTS 按所挂模型传入）。三个面必须一起
            #   改：只有它们一致，`tensordot` 才落在合法的 (nc,)×(nc,n*n) 上。
            #   通道数与权重维数不一致时权重项不参与，但特征仍按
            #   `self.n_channels` 造——绝不为此回退到 12；这种不一致在构造期
            #   已告警（见 __init__ 的 logger.warning），不再完全静默。
            nc = self.n_channels
            fp = board.feature_planes_batched(
                b[None], [[-1, -1, -3]], [[-1, -1, -3]],
                [abs(board.current_player)], [board.ko_point],
                n_channels=nc).reshape(nc, -1)
            if self.weights.ndim == 1 and self.weights.shape[0] == nc:
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


# --------------------------------------------------------------------------- #
# Playout 随机化增强：多种策略轮换
# --------------------------------------------------------------------------- #
class DiverseRolloutPolicy:
    """多种 rollout 策略轮换，增加价值估计的多样性。"""

    def __init__(self, board_size: int, num_strategies: int = 4):
        self.n = board_size
        self.strategies = [
            FastPolicy(board_size, temperature=1.0),           # 基础策略
            FastPolicy(board_size, temperature=0.5),           # 保守（更贪婪）
            FastPolicy(board_size, temperature=2.0),           # 激进（更随机）
            FastPolicy(board_size, temperature=5.0),           # 非常随机（探索）
        ]
        self.num_strategies = len(self.strategies)
        self._rng = np.random.default_rng()

    def sample_move(self, board: GoBoard, step: int, rng: np.random.Generator) -> int:
        """根据步数轮换策略。"""
        strategy_idx = (step // 5) % self.num_strategies
        return self.strategies[strategy_idx].sample_move(board, rng)


def light_rollout_diverse(board: GoBoard, diverse_policy: "DiverseRolloutPolicy",
                          max_steps: int = None,
                          rng: np.random.Generator = None) -> float:
    """使用多样化策略的 rollout，返回发起方视角胜率。"""
    if rng is None:
        rng = np.random.default_rng()
    n = board.board_size
    if max_steps is None:
        max_steps = n * n * 2
    cur = GoBoard(n)
    cur.board = board.board.copy()
    cur.current_player = board.current_player
    cur.ko_point = board.ko_point
    cur.passes = board.passes
    cur.move_history = list(board.move_history)
    initiator = cur.current_player
    passes = 0
    steps = 0
    while steps < max_steps and passes < 2:
        mv = diverse_policy.sample_move(cur, steps, rng)
        if mv == n * n:
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
    score = cur.score()
    return _tt_value(score, initiator)


def _tt_value(score: float, initiator: int) -> float:
    """Tromp-Taylor 数子结果（黑-白目数）转为发起方视角胜率。"""
    if score > 0:
        return 1.0 if initiator > 0 else -1.0
    if score < 0:
        return -1.0 if initiator > 0 else 1.0
    return 0.0
