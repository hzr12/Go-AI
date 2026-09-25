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

from scipy.ndimage import label as _ndi_label

# 2D 四连通结构（go_rules._STRUCT3 是 3D，供批量 (B,n,n) 标注用）
_STRUCT2 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _fast_atari_mask(board: GoBoard, player) -> np.ndarray:
    """返回 (n*n,) bool：player 方「仅 1 口气」的棋子位置（打吃）。

    向量化实现：连通块用 scipy.ndimage.label，气数用邻域移位加法 + bincount。
    旧实现对每个己方棋子跑一次 Python BFS，实测占整个搜索 52.8%，是
    rollout 阶段的主要开销。

    ⚠ 语义要点：这里统计的是**互不相同**的气。别直接照抄
    go_rules.feature_planes_batched 的 `bincount(weights=neigh_empty)` ——
    那种求和会把「同时邻接同块两颗子」的空点重复计一次。一个空点邻接同块的
    多颗子时只能算 1 气（旧实现用 set，恰好是去重的），所以下面先对每个空点
    的至多 4 个邻接块号排序去重，只对「首次出现」的块号贡献 1。
    """
    n = board.board_size
    b = board.board
    occ = (b == player)
    if not occ.any():
        return np.zeros((n * n,), dtype=bool)

    labelled, num = _ndi_label(occ, structure=_STRUCT2)
    if num == 0:
        return np.zeros((n * n,), dtype=bool)

    # 每个空点的 4 个邻接块号（越界/非本方棋子记 0）
    nbr = np.zeros((4, n, n), dtype=np.int32)
    nbr[0, 1:, :] = labelled[:-1, :]
    nbr[1, :-1, :] = labelled[1:, :]
    nbr[2, :, 1:] = labelled[:, :-1]
    nbr[3, :, :-1] = labelled[:, 1:]

    srt = np.sort(nbr, axis=0)
    # 只在「该块号在本空点的邻接中首次出现」时贡献 1
    is_first = np.empty((4, n, n), dtype=bool)
    is_first[0] = srt[0] > 0
    is_first[1:] = (srt[1:] != srt[:-1]) & (srt[1:] > 0)

    empty = (b == 0)
    contrib = (is_first & empty[None, :, :]).astype(np.int64).reshape(4, -1)
    lib_counts = np.bincount(
        srt.reshape(4, -1).ravel(), weights=contrib.ravel(),
        minlength=num + 1).astype(np.int64)

    atari = (lib_counts[labelled] == 1) & occ
    return atari.reshape(-1)


def _sample_categorical(p, rng) -> int:
    """按离散分布 p 采样下标（逆 CDF 法）。

    等价于 rng.choice(len(p), p=p)，但快得多：choice 每次都要重新校验概率
    并构建 CDF，而这里只需一次 cumsum + 一次 searchsorted。rollout 里每步都要
    采样，实测 rng.choice 占整个搜索 8.2%。

    注意：抽样**分布**完全相同，但消耗随机数的序列不同——固定种子下走出的
    具体棋谱会与旧实现不同。这不影响任何统计性质。
    """
    cdf = np.cumsum(p)
    cdf[-1] = 1.0  # 消除浮点累积误差，保证 searchsorted 不越界
    return int(np.searchsorted(cdf, rng.random(), side='right'))


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
            fp = board.feature_planes_batched(
                b[None], [[-1, -1, -3]], [[-1, -1, -3]],
                [abs(board.current_player)], [board.ko_point]).reshape(12, -1)
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
        return _sample_categorical(p, rng)


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
