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
    board: Optional[GoBoard]        # 节点局面。仅根/待展开叶子临时持有（展开后置 None 释放）
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
        """扩展叶子：批量评估所有合法着法的先验 P 与局面价值 v。

        2026-09 重写（去 deepcopy）：
          - 旧实现对每个候选着法 copy.deepcopy(整盘) 再 play，19 路中盘
            ~250 候选 × 每叶子 → 每叶子 25-75ms 纯 deepcopy，是搜索最大瓶颈；
          - 新实现：在叶子的棋盘上「play → 构造 planes → (rollout) → undo」
            串行推进（undo 完整恢复棋盘/提子/劫/历史），特征用 predict_batch
            的 5 元组 planes 模式传入；子节点不再持有棋盘（需要时从根重放）。
        """
        board = leaf.board
        to_play = leaf.to_play
        legal = board.get_legal_moves()
        # ⚠ 2026-09 修复根因级 bug：legal 是 bool 掩码，旧写法 [int(m) for m in legal]
        #   迭代出 True/False → int 后只有 0/1，candidates 退化为 {(0,0),(0,1),pass}
        #   三个着法（children dict key 反复覆盖）——MCTS 自项目创建起从未真正
        #   搜索过其他位置。改用 np.where 取真实合法坐标。
        candidates = [int(m) for m in np.where(legal)[0]] + [self.n_actions - 1]  # 末位为 pass

        states = []       # (None, my_h, op_h, child_to, planes) —— 增量特征模式
        child_meta = []   # (mv, child_to, my_h, op_h, rollout_value|None)
        for mv in candidates:
            pmv = -1 if mv == self.n_actions - 1 else mv
            if not board.play(pmv):
                continue  # 理论不应发生（候选来自合法掩码）
            child_to = 2 if to_play == 1 else 1
            my_h = leaf.op_hist if child_to == 1 else leaf.my_hist
            op_h = leaf.my_hist if child_to == 1 else leaf.op_hist
            planes = board.feature_planes(my_h, op_h, child_to)
            rv = None
            if self.use_rollout and self.rollout_lambda > 0.0:
                # LightPLS 在 play 后、undo 前的子局面推演（内部自带只读拷贝）
                rv = light_rollout(board, self._fast_policy,
                                   max_steps=self.rollout_steps, rng=self._rng)
            states.append((None, list(my_h), list(op_h), child_to, planes))
            child_meta.append((mv, child_to, my_h, op_h, rv))
            board.undo()

        policies, values = self.ai.predict_batch(states)
        for i, (mv, child_to, my_h, op_h, rv) in enumerate(child_meta):
            prior = float(np.asarray(policies[i, mv]).reshape(-1)[0])
            # value 是 child.to_play 视角，对 leaf 而言是对手价值 -> 取负存为 value_sum 起点
            net_v = -float(np.asarray(values[i]).reshape(-1)[0])
            if rv is not None:
                roll_v = -rv  # child 视角 -> leaf 对手视角取负
                net_v = (1.0 - self.rollout_lambda) * net_v + self.rollout_lambda * roll_v
            child = MCTSNode(
                board=None,
                my_hist=list(my_h),
                op_hist=list(op_h),
                to_play=child_to,
                move_int=mv,
                parent=leaf,
                prior=prior,
            )
            child.value_sum = net_v
            child.visit = 1
            leaf.children[mv] = child

        # 叶子本身用“最优子局面对手价值”反推价值：取平均 child q 的相反数
        leaf.value_sum = -sum(c.q() for c in leaf.children.values()) / max(len(leaf.children), 1)
        leaf.visit = 1
        leaf.expanded = True

    def _replay_path(self, path):
        """从根局面沿 path 重放着法，返回 leaf 局面的独立棋盘副本。

        重放只做 play（不 undo），历史/劫等由叶子节点自身保存的字段提供，
        这里只需要正确的盘面。单叶子成本 = 1 次 deepcopy + 路径深度次 play，
        相比旧实现每叶子 ~250 次 deepcopy 降低两个数量级。
        """
        board = copy.deepcopy(self._cur_root_board)
        for node in path[1:]:
            mv = node.move_int
            if not board.play(-1 if mv == self.n_actions - 1 else mv):
                break  # 理论不应发生（着法来自展开时的合法集）
        return board

    def _backup(self, path, v_leaf=None):
        """沿路径回传：祖先节点 visit+=1、value_sum 累加（对手视角），并回收虚拟损失。

        v_leaf: leaf.to_play 我方视角价值。旧实现只清虚拟损失、从不更新
            visit/value_sum——搜索统计不随模拟演化，PUCT 的 Q 项恒为 expand
            初值，visits 分布退化为 prior 贪心。补标准回传。
        2026-09 注意：path[-1]（叶子）的 visit/value_sum 已由 _expand 的首次
            网络评估设置（visit=1），此处只更新祖先并逐层清虚拟损失。
        """
        v = v_leaf if v_leaf is not None else 0.0
        for idx in range(len(path) - 1, -1, -1):
            node = path[idx]
            if idx != len(path) - 1:
                v = -v                    # 逐层翻转视角（父节点=对手）
                node.visit += 1
                node.value_sum += -v      # value_sum 存对手视角，q() 取负即我方
            node.virtual_loss = 0

    def search(self, root_board, my_hist, op_hist, to_play, simulations=400,
               verbose=False, path_moves=None):
        """执行 MCTS，返回按访问次数分布（已 softmax/temperature）的 action 概率。

        path_moves: 可选，从**上一次 search 的根局面**到当前局面的着法序列
            （GoBoard 编码，pass 为 -1）。提供且上一次的树仍在时，沿已有子树
            下潜复用（访问/价值统计继承），否则从零建树。

        Returns:
            visits: np.ndarray (n_actions,) 各着法访问次数
            probs : np.ndarray (n_actions,) 温度缩放后的选点分布
            root_value: float 根节点我方视角价值估计
        """
        self._cur_root_board = copy.deepcopy(root_board)

        # ---- 树复用：沿 path_moves 下潜到上次搜索的子树 ----
        root = self._reuse_root(path_moves)
        if root is None:
            root = MCTSNode(
                board=copy.deepcopy(root_board),
                my_hist=self._clone_hist(my_hist),
                op_hist=self._clone_hist(op_hist),
                to_play=to_play,
                move_int=-1,
            )
        else:
            # 复用子树：盘面用当前真实局面刷新（path_moves 已把它推进到同一局面）
            root.board = copy.deepcopy(root_board)
            root.my_hist = self._clone_hist(my_hist)
            root.op_hist = self._clone_hist(op_hist)
            root.to_play = to_play
        self._prev_root = root

        # 根先展开一次，建立子节点（树复用时根已展开，跳过以免重置统计）
        if not root.expanded:
            self._expand(root)
            self._backup([root], v_leaf=-root.value_sum)  # root 我方视角 = 对手视角取负
        root.board = None  # 根展开完即释放盘面（重放源在 _cur_root_board）

        # 生产者-消费者式并行：
        #   - num_threads 个 worker 线程只负责「选路径」（纯 CPU 估算 PUCT，极廉价），
        #     选好后给路径加虚拟损失占位，并把整条路径推入队列；
        #   - 主线程从队列批量取出路径，按路径重放叶子盘面（1 次 deepcopy + 深度次
        #     play，替代旧实现每叶子 ~250 次 deepcopy），再统一 planes + predict_batch，
        #     最后回收虚拟损失。
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
                leaf_q.put(path)

        threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        # 主线程：消费队列，批量展开
        batch: List[list] = []
        while expanded_count < total:
            try:
                path = leaf_q.get(timeout=0.2)
            except _queue.Empty:
                if all(not t.is_alive() for t in threads) and leaf_q.empty():
                    break
                continue
            batch.append(path)
            # 攒够一批（或接近）再统一展开，平衡并行度与 batch 利用率
            if len(batch) >= self.num_threads or expanded_count + len(batch) >= total:
                for pth in batch:
                    leaf = pth[-1]
                    leaf.board = self._replay_path(pth)
                    self._expand(leaf)
                    leaf.board = None  # 展开完成即释放盘面
                    self._backup(pth, v_leaf=-leaf.value_sum)  # leaf 我方视角
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

    def _reuse_root(self, path_moves):
        """沿 path_moves 从上次搜索根下潜，返回可复用的子树根（或 None）。

        中间节点必须有子树才能继续下潜；**末节点允许未展开**（children 为空
        也可复用——其 visit/value 统计仍然继承，search 会按需展开它）。
        """
        prev = getattr(self, '_prev_root', None)
        if not path_moves or prev is None:
            return None
        cur = prev
        for i, mv in enumerate(path_moves):
            if cur is None:
                return None
            key = self.n_actions - 1 if mv == -1 else mv  # 对局编码 -1=pass -> 树内编码
            if i < len(path_moves) - 1 and not cur.children:
                return None  # 还需继续下潜但无子树
            cur = cur.children.get(key)
            if cur is None:
                return None
        cur.parent = None
        cur.move_int = -1
        return cur

    def best_move(self, root_board, my_hist, op_hist, to_play, simulations=400,
                  temperature=None, return_value=False, path_moves=None):
        """返回最优着法 (move_int, is_pass)，可选返回根价值估计。

        path_moves: 从上次 best_move 根局面到当前的着法序列（GoBoard 编码，
            pass=-1），用于搜索树复用；不传则每次从零建树。
        """
        if temperature is not None:
            self.temperature = temperature
        visits, probs, root_value = self.search(
            root_board, my_hist, op_hist, to_play, simulations=simulations,
            path_moves=path_moves)
        move_int = int(np.argmax(visits)) if visits.sum() > 0 else self.n_actions - 1
        is_pass = (move_int == self.n_actions - 1)
        if return_value:
            return move_int, is_pass, float(root_value)
        return move_int, is_pass
