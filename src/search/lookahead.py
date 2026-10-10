"""策略 N 步 minimax 推演（**纯函数**，不依赖 MCTS 实例）。

这份实现是从 `src/search/mcts.py::MCTS.lookahead` 原样搬出来的（2026-09-30），
搬的理由只有一个：**RL 采集路径要用它，但不该为此构造一个 MCTS**（RL 已去掉搜索，
而 MCTS 引擎要保留给 webui / cli_play / evaluate / eval_elo）。

搬而不是复制，是为了单一真相源：两个实现一旦分叉，就会出现「webui 看到的推演
结果与 RL 采到的动作不是同一套逻辑」这种极难查的矛盾。`MCTS.lookahead` 现在是
**薄转发**，webui 的调用点与行为一字不变（它额外用自己的 `_planes1` LRU/TTL 缓存
提供根特征，见转发处注释）。

数学口径（与搬迁前逐字相同，勿"顺手优化"）
------------------------------------------------
* 根取 policy 的 top-K 候选；之后每层取 top-W；`depth` 控制层数
  （2 = 根候选 → 对手最佳应手 → 评估，共 depth+1 次批量前向）。
* 逐层批量：每层的所有节点拼一个 batch 一次前向（吃到 ONNX 大 batch 吞吐），
  只在传入棋盘上 play/undo，子节点持深拷贝。
* 价值统一折算到「根玩家视角」（子节点 to_play 与根相同时取正、否则取负），
  然后自底向上传播：奇数层（我方）取 max、偶数层（对手）取 min。
* 抽头式的 `kept` 过滤必须留着：`_child_states` 会跳过 `play()` 拒绝的着法，
  不先过滤就会越界（实测在对局第 131 手触发过 IndexError）。

新增（搬迁时一起带出来的，RL 需要而 webui 不需要）
--------------------------------------------------
返回值多了一个 `root_value`：**根局面的网络估值（根 to_play 视角）**。
搬迁前它被 `_` 丢掉了。RL 侧要把它写进 buffer 的第 5 列（替代 MCTS 根价值，
见 `scripts/selfplay_train.py` 的 P3-C 契约），所以这里一并返回。
`MCTS.lookahead` 的转发层把它丢掉，保持 4 元组返回 —— webui 不改。
"""
from typing import Dict, NamedTuple, Optional

import numpy as np


class LookaheadResult(NamedTuple):
    """推演结果。字段名比元组下标好读，下标顺序与旧 4 元组兼容。"""

    values: Dict[int, float]      # {根候选着法: 根玩家视角价值}
    policy: np.ndarray            # 根 masked policy（非法点 0，pass 恒合法）
    best_move: int                # 价值最大的根候选（全 0 时 = pass）
    best_value: float             # best_move 的价值
    root_value: float             # 根局面的网络估值（根 to_play 视角）


def child_states(board, moves, my_hist, op_hist, to_play, *, with_prev=False,
                 prev_board=None, prev_prev_board=None):
    """给定局面与候选着法，返回各子局面的 (GoBoard, my_h, op_h, to_play)。

    子节点持有完整棋盘深拷贝，供后续「按层批量前向 / 再展开」复用；
    在传入棋盘上 play/undo（调用方状态不破坏）。

    ``with_prev=True``（**仅 V7**）时改为返回 6 元组，末尾两项是
    ``(prev_board, prev_prev_board)`` —— 子局面的**前一手 / 前二手盘面**快照
    ``(n,n) int8``（或 None）。

    为什么 V7 需要它：ch15/ch16 的梯子要在**前手盘**上重算，缺了就退化成
    「整批缺失」路径，模型拿到的是训练里从未出现过的输入。12 通道不经过这里取
    前手盘（`feature_planes` 通道 12..17 不含这一路），所以默认关闭。

    ⚠ 存的是 **ndarray 快照而不是 GoBoard**：搜索侧每步都产生新盘面，持有整个
    board 对象会把 undo 栈一起钉住，内存无界增长（`v7_leaf_features` 同一理由）。
    """
    # 父局面自己的前一手/前二手，作为子局面的 prev_prev / 更前一手。
    # None 一律原样传下去（= 历史不足），**不要**拿空盘冒充。
    parent_prev = None if prev_board is None else np.asarray(prev_board, dtype=np.int8)

    out = []
    for mv in moves:
        # 快照必须在 apply_action **之前**取：子局面的「前一手」就是下这手之前的盘面
        cur = np.asarray(board.board, dtype=np.int8).copy() if with_prev else None
        if not board.apply_action(mv):
            continue
        child_to = -to_play
        cmy = list(op_hist)
        cop = list(my_hist)
        cb = board.clone()
        if with_prev:
            out.append((cb, cmy, cop, child_to, cur, parent_prev))
        else:
            out.append((cb, cmy, cop, child_to))
        board.undo()
    return out


def forward_level(ai, nodes, n_channels, n_actions, feature_fn=None):
    """对一批节点批量前向，返回 (policies(B,A), values(B,))。

    一次性组装整批特征（通道数 = 模型 `in_channels`）+ 单次 predict，最大化
    CPU/ONNX 吞吐。`n_channels` 由调用方给（MCTS 侧是 `self._in_channels()`，
    RL 侧是 `ai.in_channels`）—— 不在这里问 ai 是为了保持本模块无状态。

    `feature_fn`：**仅 V7 传**，形如 ``f(nodes) -> (spatial, global_features)``
    （见 `src/search/v7_features.py::v7_batch_features`）。给了它就走 V7 特征并把
    `predict_batch` 的输入从 5 元组换成 6 元组（多带 19 维全局输入）；不给则走
    原来的 `feature_planes_batched` + 5 元组。**两条路都保持"一次前向"，**
    没有为了兼容 V7 给 12 通道加任何分支开销。
    """
    if not nodes:
        return np.zeros((0, n_actions)), np.zeros(0)
    my_hs = [n[1] for n in nodes]
    op_hs = [n[2] for n in nodes]
    tps = [n[3] for n in nodes]
    if feature_fn is not None:
        spatial, gfeat = feature_fn(nodes)
        states = [(None, my_hs[i], op_hs[i], tps[i], spatial[i], gfeat[i])
                  for i in range(len(nodes))]
        return ai.predict_batch(states)
    arrays = np.stack([n[0].board for n in nodes])
    kos = [n[0].ko_point for n in nodes]
    planes = nodes[0][0].feature_planes_batched(arrays, my_hs, op_hs, tps, kos,
                                                 n_channels=n_channels)
    states = [(None, my_hs[i], op_hs[i], tps[i], planes[i])
              for i in range(len(nodes))]
    return ai.predict_batch(states)


def lookahead(ai, board, my_hist, op_hist, to_play, n_actions, n_channels,
              topk=12, width=4, depth=2, planes: Optional[np.ndarray] = None,
              feature_fn=None,
              ) -> LookaheadResult:
    """策略 N 步批量推演（minimax 展开 top-K/width 着法树，价值回传）。

    Args:
        ai: `GoAI`（只用 `predict_batch` 与 `in_channels`，不碰 MCTS 的任何东西）。
        board / my_hist / op_hist / to_play: 当前局面（`board` 只会 play/undo）。
        n_actions: `bs*bs+1`（pass 是最后一类）。
        n_channels: 特征平面通道数（= `ai.in_channels`）。
        topk / width / depth: 根候选数 / 每层展开宽度 / 推演层数。
        planes: 根局面**已算好**的特征。RL 侧本来就要把 planes 写进 buffer，
            传进来可以**只算一次**（根前向与 buffer 行共用同一份）。
        feature_fn: **仅 V7 传**，见 `forward_level`。给了它则根前向与逐层前向都走
            V7 特征（6 元组），且子局面**携带前两手盘面**供 ch15/ch16 梯子重算；
            不给则与 12 通道完全同路径（`planes` 会被照旧透传给根前向）。

    Returns:
        `LookaheadResult`（见其 docstring；比搬迁前多一个 `root_value`）。
    """
    n = n_actions
    legal = [int(m) for m in np.where(board.get_legal_moves())[0]] + [n - 1]

    # 根前向取 policy、**根估值**与 top-K 候选
    if feature_fn is not None:
        rplanes, rgfeat = feature_fn(
            [(board, list(my_hist), list(op_hist), int(to_play))])
        pol, val = ai.predict_batch(
            [(None, list(my_hist), list(op_hist), int(to_play),
              rplanes[0], rgfeat[0])])
    elif planes is None:
        planes = board.feature_planes_batched(
            board.board[None], [list(my_hist)], [list(op_hist)], [to_play],
            [board.ko_point], n_channels=n_channels)[0]
        pol, val = ai.predict_batch(
            [(None, list(my_hist), list(op_hist), to_play, planes)])
    else:
        pol, val = ai.predict_batch(
            [(None, list(my_hist), list(op_hist), to_play, planes)])
    root_value = float(np.asarray(val).reshape(-1)[0])
    root_value = float(np.asarray(val).reshape(-1)[0])
    p = np.asarray(pol).reshape(-1)
    masked = np.zeros(n)
    masked[legal] = p[legal]
    order = [int(m) for m in np.argsort(-masked)[:max(1, topk)]]
    if not order:
        return LookaheadResult({}, masked, n - 1, 0.0, root_value)

    # IndexError 修复（实测对局 131 手触发）：child_states 会跳过 play() 拒绝的
    # 着法，因此 levels[0] 可能比 order 短，下方 `out[kept[i]] = ...` 的对位索引
    # 就会越界。先用 play/undo 过滤出真正可下的 kept，与 levels[0] 一一对应。
    kept = []
    for mv in order:
        if board.apply_action(mv):
            board.undo()
            kept.append(mv)
    if not kept:
        return LookaheadResult({}, masked, n - 1, 0.0, root_value)

    # 逐层生成子树（每层节点 = 上层节点按 policy 选出的 top-W 孩子）
    levels = [child_states(board, kept, my_hist, op_hist, to_play,
                           **({'with_prev': True} if feature_fn is not None else {}))]
    if not levels[0]:
        return LookaheadResult({}, masked, n - 1, 0.0, root_value)
    child_ranges = []
    all_pols, all_vals = [], []
    pol0, val0 = forward_level(ai, levels[0], n_channels, n, feature_fn)
    all_pols.append(pol0)
    all_vals.append(val0)
    for _ in range(1, depth):
        prev = levels[-1]
        pols = all_pols[-1]
        nxt, ranges = [], []
        for i, nd in enumerate(prev):
            cb, cmy, cop, cto = nd[0], nd[1], nd[2], nd[3]
            pp = np.asarray(pols[i]).reshape(-1)
            lleg = [int(m) for m in np.where(cb.get_legal_moves())[0]] + [n - 1]
            porder = sorted(lleg, key=lambda m: -pp[m])[:max(1, width)]
            s = len(nxt)
            if feature_fn is not None:
                nxt.extend(child_states(cb, porder, cmy, cop, cto,
                                        with_prev=True,
                                        prev_board=nd[4], prev_prev_board=nd[5]))
            else:
                nxt.extend(child_states(cb, porder, cmy, cop, cto))
            ranges.append((s, len(nxt)))
        levels.append(nxt)
        child_ranges.append(ranges)
        if not nxt:
            break
        pnxt, vnxt = forward_level(ai, nxt, n_channels, n, feature_fn)
        all_pols.append(pnxt)
        all_vals.append(vnxt)

    # 各层静态价值转到「根玩家视角」（forward 返回的是该节点 to_play 视角）
    node_val = []
    for L, lvl in enumerate(levels):
        node_val.append([
            (float(all_vals[L][i]) if lvl[i][3] == to_play
             else -float(all_vals[L][i]))
            for i in range(len(lvl))
        ])

    # 自底向上传播：奇数层（我方）取 max，偶数层（对手）取 min
    for L in reversed(range(len(levels) - 1)):
        ranges = child_ranges[L]
        is_max = (L % 2 == 1)
        for i in range(len(levels[L])):
            s, e = ranges[i]
            ch = node_val[L + 1][s:e]
            if ch:  # 有孩子 → 按层性质聚合；无孩子 → 保留自身静态价值
                node_val[L][i] = (max if is_max else min)(ch)

    # 根每个候选的最终价值 = 其对应子节点（对手层）的根视角价值
    out = {kept[i]: node_val[0][i] for i in range(len(kept))}
    best_mv = max(out, key=lambda m: out[m]) if out else (n - 1)
    best_v = out[best_mv] if out else 0.0
    return LookaheadResult(out, masked, best_mv, best_v, root_value)
