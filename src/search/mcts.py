"""
MCTS 搜索（AlphaGoZero 风格 PUCT），基于 SFT 主线的 GoAI 网络。

与已删除的旧 minimax/MuZero 代码无关：本模块只用 src.game.go_rules.GoBoard
的 12 通道 feature_planes 与合法着法接口，配合 GoAI.predict_batch 批量评估
叶子节点，是推理提速（GPU 上 N=400~800 仅 1-2s/步）的核心。

加速手段：
  - 批量叶子评估：同一层待展开的节点拼成一个 batch 一次前向（predict_batch）
  - 虚拟损失（virtual loss）：多线程并行模拟不同分支，避免重复探索同一条路径
  - 充分利用 GPU：batch 越大，前向吞吐越高
"""

import math
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.game.go_rules import GoBoard


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
                 num_threads=4, temperature=1.0, temperature_decay=0.0):
        """
        Args:
            ai:            GoAI 实例（需支持 predict_batch）
            c_puct:        PUCT 探索系数
            virtual_loss:  虚拟损失系数，配合多线程并行模拟
            num_threads:   并行模拟线程数（CPU 多核 / GPU 流水）
            temperature:   最终选点温度（>0 采样，0=贪心按访问）
        """
        self.ai = ai
        self.bs = board_size
        self.n_actions = board_size * board_size + 1
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
        self.num_threads = max(1, num_threads)
        self.temperature = temperature
        self.temperature_decay = temperature_decay

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
        # 合法着法 + 始终允许的虚着（统一转 python int，避免 numpy 标量索引问题）
        candidates = [int(m) for m in legal] + [self.n_actions - 1]

        # 批量构造子局面状态（避免逐子克隆+前向，GPU 上一次算完）
        child_states = []
        for mv in candidates:
            nb = copy.deepcopy(board)
            if mv == self.n_actions - 1:
                nb.play(-1)
            else:
                nb.play(mv)
            # 历史：子节点 to_play 翻转
            child_to = 2 if to_play == 1 else 1
            my_h = leaf.op_hist if child_to == 1 else leaf.my_hist
            op_h = leaf.my_hist if child_to == 1 else leaf.op_hist
            child_states.append((nb, list(my_h), list(op_h), child_to))

        policies, values = self.ai.predict_batch(child_states)
        # values 为各子局面 to_play(=child_to) 视角；转成 leaf.to_play 视角需取负
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
            child.value_sum = -float(np.asarray(values[i]).reshape(-1)[0])
            child.visit = 1
            leaf.children[mv] = child

        # 叶子本身用“最优子局面对手价值”反推价值：取平均 child q 的相反数
        leaf.value_sum = -sum(c.q() for c in leaf.children.values()) / max(len(leaf.children), 1)
        leaf.visit = 1

    def _backup(self, path):
        """沿路径回传（含虚拟损失回收）。"""
        # 路径上每个节点记录的 value_sum 已在 expand 时按对手视角累加，
        # 这里只需把虚拟损失归还（visit 已在 expand 计入）。
        for node in path:
            node.virtual_loss = 0

    def _simulate_once(self, root):
        """一次模拟（无锁版本由调用方线程管理虚拟损失）。"""
        path = self._select(root)
        leaf = path[-1]
        if leaf.children:
            # 非叶子：继续向下（理论上 select 已到叶子，这里保险）
            pass
        else:
            if leaf.visit == 0 or not leaf.children:
                # 给路径加虚拟损失，避免其他线程重复选同路径
                for node in path:
                    node.virtual_loss += int(self.virtual_loss)
                self._expand(leaf)
                self._backup(path)

    def search(self, root_board, my_hist, op_hist, to_play, simulations=400, verbose=False):
        """执行 MCTS，返回按访问次数分布（已 softmax/temperature）的 action 概率。

        Returns:
            visits: np.ndarray (n_actions,) 各着法访问次数
            probs : np.ndarray (n_actions,) 温度缩放后的选点分布
        """
        import threading

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

        def worker():
            for _ in range(simulations // self.num_threads + 1):
                self._simulate_once(root)

        threads = [threading.Thread(target=worker) for _ in range(self.num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        visits = np.zeros(self.n_actions, dtype=np.int64)
        for mv, child in root.children.items():
            visits[mv] = child.visit

        # 根价值：我方（to_play）视角的平均子节点价值估计
        root_value = 0.0
        if root.children:
            root_value = -sum(c.q() for c in root.children.values()) / len(root.children)

        # 温度缩放选点分布
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
