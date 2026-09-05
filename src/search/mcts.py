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
import time
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
    proved: int = 0                 # MCTS-Solver：+1=to_play 必胜, -1=to_play 必败, 0=未知
    prefetch: Optional[tuple] = None  # worker 推测性预评估缓存 (policy_np, value_float)

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
                 rollout_threads=None, expand_topk=0, expand_chunk=0,
                 solver_thresh=0.9, spec_prefetch=True,
                 leaf_ab_depth=0, leaf_ab_width=4, leaf_ab_weight=0.5,
                 leaf_ab_uncertain=0.85, priors_leaf=False,
                 dirichlet_alpha=0.0, dirichlet_eps=0.0):
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
            expand_topk:   展开候选截断（0=全部 ~N²+1 个）。>0 时先前向叶子本身，
                按 policy 取 top-K 合法候选再评估子节点——每次模拟批量从 ~N²+1
                降到 K+1，CPU 推理提速一个数量级；低 prior 点几乎不影响棋力。
            expand_chunk:  展开期 α-β 界截断的评估块大小（0=一块评完，GPU 推荐 0）。
                >0 时按 prior 降序分块评估子节点，一旦发现叶子视角 ≥ solver_thresh
                的必胜着立即停止评估剩余块（省前向）。
            solver_thresh: MCTS-Solver 必胜判定阈值；叶子下全部候选 ≤ -该值时判必败。
                proven 结果作为 ±1 绝对值回传并在选路径时优先/回避。
            spec_prefetch: worker 选路径占位后推测性预评估叶子（与主线程 batch 前向
                重叠），_expand 直接复用，省掉每模拟 1 次叶子前向。
            leaf_ab_depth: 叶子内浅层 α-β 深度（0=关闭）。>0 且叶子净价值不确定
                （|v| < leaf_ab_uncertain）时，在叶子下用 value 头静态评估、policy
                排序走 depth 层 negamax α-β（speculative 深化，棋力换时间）。
            leaf_ab_width: 叶内 α-β 每节点最多扩展的着法数（policy 排序）。
            leaf_ab_weight: AB 结果与净价值的融合权重（0=只用净价值）。
            leaf_ab_uncertain: 净价值绝对值低于该值才触发叶内 α-β。
            priors_leaf: 子节点先验改用叶子自身 policy 在该着法上的值（标准
                AlphaZero 方案）。默认 False=沿用旧方案（子局面自身 policy 在
                同一着法上的值，非标准、少模拟时偏离策略网络）；True 时少量
                模拟的 visits 分布更贴近策略网络（策略+少量MCTS 的基础）。
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
        self.expand_topk = max(0, int(expand_topk))
        self.expand_chunk = max(0, int(expand_chunk))
        self.solver_thresh = float(solver_thresh)
        self.spec_prefetch = bool(spec_prefetch)
        self.leaf_ab_depth = max(0, int(leaf_ab_depth))
        self.leaf_ab_width = max(1, int(leaf_ab_width))
        self.leaf_ab_weight = float(leaf_ab_weight)
        self.leaf_ab_uncertain = float(leaf_ab_uncertain)
        self.priors_leaf = bool(priors_leaf)
        # 自对弈探索：根展开时先验混入 Dirichlet 噪声（0=关闭，对弈不受影响）。
        # 注意：树复用复用根不重新展开，故仅每局首手有噪声——训练管线每局新建树即可。
        self.dir_alpha = float(dirichlet_alpha)
        self.dir_eps = float(dirichlet_eps)
        self._fast_policy = FastPolicy(board_size) if use_rollout else None
        self._rng = np.random.default_rng(1234)

    # ------------------------------------------------------------------ #
    def _clone_hist(self, h):
        return list(h)

    def _select(self, node):
        """从根递归选到叶子（PUCT + 虚拟损失 + proven 剪枝）。返回路径。

        MCTS-Solver 选择规则：存在 proven loss 子节点（其 to_play 必败）→
        立即选它（我方必胜着，无需再探）；proven win 子节点（其 to_play 必胜）
        优先跳过，仅当全部子节点均 proven win 时才回退选择。
        """
        path = [node]
        cur = node
        while cur.children:
            best, best_score = None, -1e9
            proven_loss = None
            proven_win = []
            for child in cur.children.values():
                if child.proved == -1:
                    proven_loss = child
                    break
                if child.proved == 1:
                    proven_win.append(child)
                    continue
                score = child.q() + child.u(self.c_puct)
                if score > best_score:
                    best_score, best = score, child
            if proven_loss is not None:
                cur = proven_loss
            else:
                cur = best if best is not None else proven_win[0]
            path.append(cur)
        return path

    def _expand(self, leaf):
        """扩展叶子：评估合法着法的先验 P 与局面价值 v。

        2026-09 二期（Speculative MCTS+Alpha-Beta，目标 CPU 提速）：
          - expand_topk：先前向叶子本身（优先复用 worker 的推测性预评估
            prefetch，与主线程 batch 前向重叠），按叶子 policy 截断 top-K 候选；
          - expand_chunk：按 prior 降序分块评估子节点并维护 α 界，出现叶子视角
            ≥ solver_thresh 的必胜着立即截断（MCTS-Solver，Winands 2008 风格），
            剩余低 prior 候选不再评估/建节点；全部候选 ≤ -solver_thresh 判必败；
          - leaf_ab_depth>0：叶子净价值不确定时在叶子下做浅层 negamax α-β
            （value 头静态评估 + policy 排序），结果按 leaf_ab_weight 融合；
          - proven 结果（±1）经 _backup 逐层回传，_select 优先必胜着/回避必败着。
        """
        board = leaf.board
        to_play = leaf.to_play
        legal = board.get_legal_moves()
        # ⚠ 2026-09 修复根因级 bug：legal 是 bool 掩码，旧写法 [int(m) for m in legal]
        #   迭代出 True/False → int 后只有 0/1，candidates 退化为 {(0,0),(0,1),pass}
        #   三个着法（children dict key 反复覆盖）——MCTS 自项目创建起从未真正
        #   搜索过其他位置。改用 np.where 取真实合法坐标。
        candidates = [int(m) for m in np.where(legal)[0]] + [self.n_actions - 1]  # 末位为 pass

        leaf_v = None  # leaf.to_play 视角
        if self.expand_topk and self.expand_topk < len(candidates):
            # 叶子前向：优先复用 worker 推测性预评估（已与主线程 batch 重叠）
            if leaf.prefetch is not None:
                lp, leaf_v = leaf.prefetch
                leaf.prefetch = None
            else:
                leaf_planes = board.feature_planes(leaf.my_hist, leaf.op_hist, to_play)
                lp, lv = self.ai.predict_batch(
                    [(None, list(leaf.my_hist), list(leaf.op_hist), to_play, leaf_planes)])
                leaf_v = float(np.asarray(lv[0]).reshape(-1)[0])

            masked = np.zeros(self.n_actions, dtype=np.float64)
            masked[candidates] = np.asarray(lp).reshape(-1)[candidates]
            # prior 降序 = α-β 的走法排序
            order = [int(m) for m in np.argsort(-masked)[:self.expand_topk]]
            # priors_leaf：叶子自身 policy 作子节点先验（标准 AlphaZero 方案）
            priors = ({int(m): float(masked[m]) for m in order}
                      if self.priors_leaf else None)

            # speculative 叶内 α-β（可选）：净价值不确定时深化
            if self.leaf_ab_depth > 0 and abs(leaf_v) < self.leaf_ab_uncertain:
                ab_v = self._leaf_ab(board, to_play, leaf.my_hist, leaf.op_hist,
                                     self.leaf_ab_depth, self.leaf_ab_width)
                leaf_v = (1.0 - self.leaf_ab_weight) * leaf_v + self.leaf_ab_weight * ab_v
                if ab_v >= self.solver_thresh:
                    leaf.proved = 1
                elif ab_v <= -self.solver_thresh:
                    leaf.proved = -1

            if self.expand_chunk > 0 and len(order) > self.expand_chunk:
                all_vals = self._expand_chunked(board, to_play, leaf, order, priors)
            else:
                all_vals = self._eval_children(board, to_play, leaf, order, priors)
            if leaf.proved == 0 and all_vals and \
                    max(all_vals) <= -self.solver_thresh:
                leaf.proved = -1  # 全部候选都输 → to_play 必败
            leaf.value_sum = (-1.0 if leaf.proved == 1 else
                              1.0 if leaf.proved == -1 else -leaf_v)
        else:
            # 全量路径（GPU）：保持原行为，一批评完
            self._eval_children(board, to_play, leaf, candidates)
            # 叶子价值用“子节点平均 q”反推（对手视角取负）
            leaf.value_sum = -sum(c.q() for c in leaf.children.values()) / max(len(leaf.children), 1)
        self._apply_root_noise(leaf)
        leaf.visit = 1
        leaf.expanded = True

    def _apply_root_noise(self, leaf):
        """根展开后混入 Dirichlet 噪声（自对弈探索）。"""
        if self.dir_eps > 0 and self.dir_alpha > 0 and leaf.parent is None and leaf.children:
            noise = np.random.default_rng().dirichlet(
                [self.dir_alpha] * len(leaf.children))
            for c, en in zip(leaf.children.values(), noise):
                c.prior = (1.0 - self.dir_eps) * c.prior + self.dir_eps * en

    def _eval_children(self, board, to_play, leaf, moves, priors=None):
        """评估一批候选着法并创建子节点。返回各子节点叶子视角价值列表。

        在叶子的棋盘上「play → 构造 planes → (rollout) → undo」串行推进
        （undo 完整恢复棋盘/提子/劫/历史），子节点不持有棋盘。
        priors: 可选 {mv: prior}，提供时用叶子 policy 先验（priors_leaf），
        否则用子局面自身 policy（旧方案）。
        """
        states = []       # (None, my_h, op_h, child_to, planes) —— 增量特征模式
        child_meta = []   # (mv, child_to, my_h, op_h, rollout_value|None)
        for mv in moves:
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

        if not states:
            return []
        policies, values = self.ai.predict_batch(states)
        leaf_view_vals = []
        for i, (mv, child_to, my_h, op_h, rv) in enumerate(child_meta):
            prior = (priors[mv] if priors is not None
                     else float(np.asarray(policies[i, mv]).reshape(-1)[0]))
            # value 是 child.to_play 视角，取负即叶子（child 对手）视角
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
            leaf_view_vals.append(net_v)
        return leaf_view_vals

    def _expand_chunked(self, board, to_play, leaf, order, priors=None):
        """α-β 界截断：按 prior 降序分块评估子节点。

        每评完一块检查「叶子视角最优值」：一旦 ≥ solver_thresh（必胜着）立即
        截断并标记 proven win，剩余候选不再评估/建节点——被截掉的候选多为
        低 prior 尾部，对搜索几乎无影响，前向量显著下降。
        """
        kept_vals = []
        chunk = max(1, self.expand_chunk)
        for i in range(0, len(order), chunk):
            part = order[i:i + chunk]
            vals = self._eval_children(board, to_play, leaf, part, priors)
            kept_vals.extend(vals)
            if vals and max(kept_vals) >= self.solver_thresh:
                leaf.proved = 1
                break
        return kept_vals

    def _leaf_ab(self, board, to_play, my_hist, op_hist, depth, width,
                 alpha=-1.0, beta=1.0):
        """叶子内浅层 negamax α-β（speculative 深化）。

        每个访问节点做 1 次前向：value 头当静态评估、policy top-width 当走法
        排序。depth=0 时即静态评估。返回 to_play 视角价值。在传入棋盘上
        play/undo（undo 完整恢复，不破坏调用方的展开流程）。
        """
        legal = [int(m) for m in np.where(board.get_legal_moves())[0]] + [self.n_actions - 1]
        planes = board.feature_planes(my_hist, op_hist, to_play)
        pol, val = self.ai.predict_batch(
            [(None, list(my_hist), list(op_hist), to_play, planes)])
        v_static = float(np.asarray(val[0]).reshape(-1)[0])
        if depth <= 0:
            return v_static
        p = np.asarray(pol).reshape(-1)
        order = sorted(legal, key=lambda m: -p[m])[:width]
        best = -2.0
        for mv in order:
            pmv = -1 if mv == self.n_actions - 1 else mv
            if not board.play(pmv):
                continue
            child_to = 2 if to_play == 1 else 1
            if child_to == 1:
                my_h, op_h = op_hist, my_hist
            else:
                my_h, op_h = my_hist, op_hist
            v = -self._leaf_ab(board, child_to, my_h, op_h, depth - 1, width,
                               -beta, -alpha)
            board.undo()
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break  # β 截断
        return best if best > -2.0 else v_static

    def lookahead2(self, board, my_hist, op_hist, to_play, topk=12, width=4):
        """策略 2 步批量推演：我方候选 → 对手最佳应手，价值回传。

        流程（共 3 次批量前向，无递归）：
          1. 根前向 → policy top-K 候选；
          2. K 个候选位置拼一个 batch 前向（对手视角 policy+value）；
          3. 每个候选取对手 top-W 应手，全部拼一个 batch 前向（回到我方视角）。

        我走 mv 的价值 = min_j V(mv → 对手第 j 应手)（对手挑对我最差的）。
        仅在传入棋盘上 play/undo，不破坏调用方状态。
        返回 ({mv: 我方视角价值}, 根 masked policy)。
        """
        n = self.n_actions
        legal = board.get_legal_moves()
        cands = [int(m) for m in np.where(legal)[0]] + [n - 1]  # 末位 pass

        # 1) 根前向
        planes = board.feature_planes(my_hist, op_hist, to_play)
        pol, _ = self.ai.predict_batch(
            [(None, list(my_hist), list(op_hist), to_play, planes)])
        p = np.asarray(pol).reshape(-1)
        masked = np.zeros(n)
        masked[cands] = p[cands]
        order = [int(m) for m in np.argsort(-masked)[:max(1, topk)]]

        child_to = 2 if to_play == 1 else 1
        my_h = op_hist if child_to == 1 else my_hist
        op_h = my_hist if child_to == 1 else op_hist

        # 2) 第一层批量：K 个候选位置（对手视角）
        states1 = []
        kept = []
        for mv in order:
            pmv = -1 if mv == n - 1 else mv
            if not board.play(pmv):
                continue
            states1.append((None, list(my_h), list(op_h), child_to,
                            board.feature_planes(my_h, op_h, child_to)))
            board.undo()
            kept.append(mv)
        if not kept:
            return {}, masked, n - 1, 0.0
        pols, _vals1 = self.ai.predict_batch(states1)

        # 3) 第二层批量：每个候选下对手 top-W 应手（回到我方视角）
        states2 = []
        owner = []   # 每个 state2 对应 kept 的下标 i
        for i, mv in enumerate(kept):
            pmv = -1 if mv == n - 1 else mv
            if not board.play(pmv):
                continue
            pi = np.asarray(pols[i]).reshape(-1)
            opp_legal = [int(m) for m in np.where(board.get_legal_moves())[0]] + [n - 1]
            opp = sorted(opp_legal, key=lambda m: -pi[m])[:max(1, width)]
            for omv in opp:
                opmv = -1 if omv == n - 1 else omv
                if not board.play(opmv):
                    continue
                states2.append((None, list(my_hist), list(op_hist), to_play,
                                board.feature_planes(my_hist, op_hist, to_play)))
                owner.append(i)
                board.undo()   # 撤对手应手
            board.undo()       # 撤我方候选
        out = {}
        if states2:
            _, vals2 = self.ai.predict_batch(states2)
            worst = {}   # i -> 对手最佳应手后的我方最差价值
            for i, v in zip(owner, vals2):
                v = float(np.asarray(v).reshape(-1)[0])  # 我方视角
                worst[i] = min(worst.get(i, 2.0), v)
            for i, mv in enumerate(kept):
                if i in worst:
                    out[mv] = worst[i]
                else:
                    # 对手无应手记录（罕见）：退化为候选位置价值取负
                    out[mv] = -float(np.asarray(_vals1[i]).reshape(-1)[0])
        else:
            for i, mv in enumerate(kept):
                out[mv] = -float(np.asarray(_vals1[i]).reshape(-1)[0])
        best_mv = max(out, key=lambda m: out[m])
        return out, masked, best_mv, out[best_mv]

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
                # MCTS-Solver 证明回传：
                #   子节点 proven loss（其 to_play 必败）→ 本方有必胜着 → 本节点 proven win
                #   本节点全部子节点均 proven win（对手到哪都赢）→ 本节点 proven loss
                if node.proved == 0:
                    if path[idx + 1].proved == -1:
                        node.proved = 1
                    elif node.children and all(
                            c.proved == 1 for c in node.children.values()):
                        node.proved = -1
            node.virtual_loss = 0

    def search(self, root_board, my_hist, op_hist, to_play, simulations=400,
               verbose=False, path_moves=None, progress_cb=None):
        """执行 MCTS，返回按访问次数分布（已 softmax/temperature）的 action 概率。

        path_moves: 可选，从**上一次 search 的根局面**到当前局面的着法序列
            （GoBoard 编码，pass 为 -1）。提供且上一次的树仍在时，沿已有子树
            下潜复用（访问/价值统计继承），否则从零建树。

        progress_cb: 可选，progress_cb(sims_done, root)，每完成一次模拟调用一次
            （sims_done 为已完成模拟数，root 为当前根节点）。用于 UI 实时渲染；
            回调异常会被吞掉，不影响搜索。

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
                        # 已被其他线程抢先展开：本线程无事可做。
                        # sleep 释放 GIL，避免自旋挤占主线程的 torch 推理。
                        pass
                    else:
                        for node in path:
                            node.virtual_loss += int(self.virtual_loss)
                        leaf.expanded = True  # 逻辑占位，真正展开在主线程
                        produced += 1
                        leaf_q.put(path)
                        # 推测性预评估：占位后立刻在 worker 线程前向叶子
                        #（torch 前向释放 GIL，与主线程 batch 评估重叠），
                        # 主线程 _expand 命中即复用，省掉每模拟 1 次叶子前向。
                        # 竞态安全：_expand 消费时置 None，未命中则主线程自行前向。
                        if self.spec_prefetch and leaf.board is None \
                                and leaf.prefetch is None:
                            try:
                                pb = copy.deepcopy(self._cur_root_board)
                                for nd in path[1:]:
                                    pmv = -1 if nd.move_int == self.n_actions - 1 \
                                        else nd.move_int
                                    if not pb.play(pmv):
                                        break
                                planes = pb.feature_planes(
                                    leaf.my_hist, leaf.op_hist, leaf.to_play)
                                pol, val = self.ai.predict_batch(
                                    [(None, list(leaf.my_hist), list(leaf.op_hist),
                                      leaf.to_play, planes)])
                                leaf.prefetch = (
                                    np.asarray(pol[0]).reshape(-1),
                                    float(np.asarray(val[0]).reshape(-1)[0]))
                            except Exception:  # noqa: BLE001
                                leaf.prefetch = None
                        continue
                time.sleep(0.001)

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
                    if progress_cb is not None:
                        try:
                            progress_cb(expanded_count, root)
                        except Exception:  # noqa: BLE001
                            pass
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
