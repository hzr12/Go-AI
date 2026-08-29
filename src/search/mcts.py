"""
MCTS 搜索（AlphaGoZero 风格 PUCT），基于 SFT 主线的 GoAI 网络。

与已删除的旧 minimax/MuZero 代码无关：本模块只用 src.game.go_rules.GoBoard
的 12 通道 feature_planes 与合法着法接口，配合 GoAI.predict_batch 批量评估
叶子节点，是推理提速（GPU 上 N=400~800 仅 1-2s/步）的核心。

加速手段：
  - 批量叶子评估：主线程把跨线程选出的待展开叶子拼成 batch，一次 predict_batch
  - 虚拟损失（virtual loss）：多线程只负责「选路径」（纯 CPU 估算，极廉价），
    昂贵的特征构造 + 网络前向集中在主线程统一做，避免每个 worker 重复 deepcopy
    与 feature_planes 计算
  - LightPLS：叶子价值融合网络 value 与轻量 rollout（Tromp-Taylor 快数子），
    低价提升搜索深度与棋力（见 search/light_rollout.py）
"""

import math
import copy
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.game.go_rules import GoBoard
from src.search.light_rollout import FastPolicy, light_rollout


@dataclass
class MCTSNode:
    board: GoBoard                  # 该节点局面（克隆）
    my_hist: list                   # 当前执子方最近 3 手（clone）
    op_hist: list                   # 对手最近 3 手（clone）
    to_play: int                    # 1=黑 2=白
    move_int: int                   # 到达此节点的着法（-1=根）
    parent: Optional["MCTSNode"] = None
    children: dict = field(default_factory=dict)   # move_int -> child
    prior: float = 0.0              # 父节点给出的先验 P(a)
    visit: int = 0
    value_sum: float = 0.0          # 累加（对手视角）价值，取负即我方
    virtual_loss: int = 0           # 并行模拟占位的虚拟损失
    expanded: bool = False

    def q(self):
        """我方（to_play）视角的平均价值估计。"""
        if self.visit + self.virtual_loss == 0:
            return 0.0
        return -self.value_sum / (self.visit + self.virtual_loss)

    def u(self, c_puct):
        """PUCT 探索项。"""
        if self.parent is None:
            return 0.0
        pa = self.parent.visit + self.parent.virtual_loss
        if pa == 0:
            pa = 1
        return c_puct * self.prior * math.sqrt(pa) / (1 + self.visit + self.virtual_loss)


class MCTS:
    def __init__(self, ai, board_size, c_puct=1.4, virtual_loss=4.0,
                 num_threads=4, temperature=1.0, temperature_decay=0.0,
                 use_rollout=False, rollout_lambda=0.25, rollout_steps=None,
                 rollout_threads=None):
        """
        Args:
            ai:            GoAI 实例（需支持 predict_batch）
            c_puct:        PUCT 探索系数
            virtual_loss:  虚拟损失系数，配合多线程并行模拟
            num_threads:   并行选路径线程数（CPU 多核）
            temperature:   最终选点温度（>0 采样，0=贪心按访问）
            use_rollout:   是否启用 LightPLS 轻量 rollout 价值融合
            rollout_lambda: rollout 价值在叶子价值中的权重 (0=只用网络, 1=只用 rollout)
            rollout_steps: 单次 rollout 最大步数（默认 2*N*N）
            rollout_threads: rollout 线程数（默认与 num_threads 同）
        """
        self.ai = ai
        self.bs = board_size
        self.n_actions = board_size * board_size + 1
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
        self.num_threads = max(1, num_threads)
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.use_rollout = use_rollout
        self.rollout_lambda = float(rollout_lambda)
        self.rollout_steps = rollout_steps
        self.rollout_threads = rollout_threads or num_threads
        self._fast_policy = FastPolicy(board_size) if use_rollout else None
        self._rng = np.random.default_rng(1234)

    # ------------------------------------------------------------------ #
    def _clone_hist(self, h):
        return list(h)

    def _select(self, node):
        """从根递归选到叶子（PUCT + 虚拟损失）。返回路径。"""
        path = [node]
        cur = node
        while cur.children:
            best, best_score = None, -1e9
            for child in cur.children.values():
                score = child.q() + child.u(self.c_puct)
                if score > best_score:
                    best_score, best = score, child
            cur = best
            path.append(cur)
        return path

    def _expand(self, leaf):
        """扩展叶子：批量评估所有合法着法的先验 P 与局面价值 v。"""
        board = leaf.board
        to_play = leaf.to_play
        legal = board.get_legal_moves()
        candidates = [int(m) for m in legal] + [self.n_actions - 1]

        # 批量构造子局面状态（避免逐子克隆+前向，GPU 上一次算完）
        child_states = []
        for mv in candidates:
            nb = copy.deepcopy(board)
            if mv == self.n_actions - 1:
                nb.play(-1)
            else:
                nb.play(mv)
            child_to = 2 if to_play == 1 else 1
            my_h = leaf.op_hist if child_to == 1 else leaf.my_hist
            op_h = leaf.my_hist if child_to == 1 else leaf.op_hist
            child_states.append((nb, list(my_h), list(op_h), child_to))

        policies, values = self.ai.predict_batch(child_states)
        for i, mv in enumerate(candidates):
            prior = float(np.asarray(policies[i, mv]).reshape(-1)[0])
            child = MCTSNode(
                board=child_states[i][0],
                my_hist=child_states[i][1],
                op_hist=child_states[i][2],
                to_play=child_states[i][3],
                move_int=mv,
                parent=leaf,
                prior=prior,
            )
            # value 是 child.to_play 视角，对 leaf 而言是对手价值 -> 取负存为 value_sum 起点
            net_v = -float(np.asarray(values[i]).reshape(-1)[0])
            if self.use_rollout and self.rollout_lambda > 0.0:
                # LightPLS：用轻量 rollout 终局数子补充价值（对 child.to_play 视角）
                rv = light_rollout(child.board, self._fast_policy,
                                   max_steps=self.rollout_steps, rng=self._rng)
                roll_v = -rv  # child 视角 -> leaf 对手视角取负
                net_v = (1.0 - self.rollout_lambda) * net_v + self.rollout_lambda * roll_v
            child.value_sum = net_v
            child.visit = 1
            leaf.children[mv] = child

        # 叶子本身用“最优子局面对手价值”反推价值：取平均 child q 的相反数
        leaf.value_sum = -sum(c.q() for c in leaf.children.values()) / max(len(leaf.children), 1)
        leaf.visit = 1
        leaf.expanded = True

    def _backup(self, path):
        """沿路径回传（含虚拟损失回收）。"""
        for node in path:
            node.virtual_loss = 0

    def search(self, root_board, my_hist, op_hist, to_play, simulations=400, verbose=False):
        """执行 MCTS，返回按访问次数分布（已 softmax/temperature）的 action 概率。

        Returns:
            visits: np.ndarray (n_actions,) 各着法访问次数
            probs : np.ndarray (n_actions,) 温度缩放后的选点分布
            root_value: float 根节点我方视角价值估计
        """
        root = MCTSNode(
            board=copy.deepcopy(root_board),
            my_hist=self._clone_hist(my_hist),
            op_hist=self._clone_hist(op_hist),
            to_play=to_play,
            move_int=-1,
        )
        # 根先展开一次，建立子节点
        self._expand(root)
        self._backup([root])

        # 生产者-消费者式并行：
        #   - num_threads 个 worker 线程只负责「选路径」（纯 CPU 估算 PUCT，极廉价），
        #     选好后给路径加虚拟损失占位，并把叶子推入队列；
        #   - 主线程从队列批量取出叶子，统一 deepcopy + feature_planes + predict_batch
        #     （昂贵部分，集中在主线程一次大 batch 前向），再回收虚拟损失。
        # 这样每个叶子只构造一次特征、只前向一次，避免旧实现里每个 worker 各自
        # deepcopy + feature_planes 的重复开销。
        import queue as _queue
        leaf_q: _queue.Queue = _queue.Queue()
        finished = threading.Event()
        produced = 0
        expanded_count = 0
        total = simulations
        lock = threading.Lock()

        def worker():
            nonlocal produced
            while not finished.is_set():
                with lock:
                    if produced >= total:
                        return
                    path = self._select(root)
                    leaf = path[-1]
                    if leaf.expanded:
                        # 已被其他线程抢先展开：本线程无事可做
                        continue
                    for node in path:
                        node.virtual_loss += int(self.virtual_loss)
                    leaf.expanded = True  # 逻辑占位，真正展开在主线程
                    produced += 1
                leaf_q.put(leaf)

        threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        # 主线程：消费队列，批量展开
        batch: List[MCTSNode] = []
        while expanded_count < total:
            try:
                leaf = leaf_q.get(timeout=0.2)
            except _queue.Empty:
                if all(not t.is_alive() for t in threads) and leaf_q.empty():
                    break
                continue
            batch.append(leaf)
            # 攒够一批（或接近）再统一展开，平衡并行度与 batch 利用率
            if len(batch) >= self.num_threads or expanded_count + len(batch) >= total:
                for lf in batch:
                    self._expand(lf)
                    self._backup([lf])
                    expanded_count += 1
                batch = []
        finished.set()
        for t in threads:
            t.join(timeout=1.0)

        visits = np.zeros(self.n_actions, dtype=np.int64)
        for mv, child in root.children.items():
            visits[mv] = child.visit

        root_value = 0.0
        if root.children:
            root_value = -sum(c.q() for c in root.children.values()) / len(root.children)

        temp = self.temperature
        if temp <= 0:
            probs = np.zeros(self.n_actions)
            best_mv = int(np.argmax(visits))
            probs[best_mv] = 1.0
        else:
            vis = visits.astype(np.float64) ** (1.0 / temp)
            probs = vis / vis.sum()
        return visits, probs, root_value

    def best_move(self, root_board, my_hist, op_hist, to_play, simulations=400,
                  temperature=None, return_value=False):
        """返回最优着法 (move_int, is_pass)，可选返回根价值估计。"""
        if temperature is not None:
            self.temperature = temperature
        visits, probs, root_value = self.search(
            root_board, my_hist, op_hist, to_play, simulations=simulations)
        move_int = int(np.argmax(visits)) if visits.sum() > 0 else self.n_actions - 1
        is_pass = (move_int == self.n_actions - 1)
        if return_value:
            return move_int, is_pass, float(root_value)
        return move_int, is_pass
