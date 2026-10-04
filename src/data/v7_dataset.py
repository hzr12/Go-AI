# -*- coding: utf-8 -*-
"""``V7Dataset`` —— 22 通道 V7 训练的数据侧（特征装配 + 标签 + 行权重）。

它是什么
--------
:class:`V7Dataset` 是 :class:`src.data.dataset.SupervisedDataset` 的**子类**，
不是第二个数据集类。除了「多两样东西」之外行为逐位继承：

  1. 构造期**重建历史列** ``my_hist`` / ``op_hist``（见下面那条 bug）；
  2. :meth:`sample_batch_v7` 造出 V7 的两份输入与 ``labels_dict``。

做成子类而不是组合，是为了**直接复用** ``scripts/train_sft.py`` 里那套已经
调好的多进程装配路径：``_prefetch_worker(v7=True)`` → ``v7_batch_features(dataset,
idxs)`` → ``dataset.my_hist[idxs]``。只要本类把 ``my_hist`` / ``op_hist``
**就地产物**覆写成正确的值，那条路径一个字都不用改就会拿到正确特征
（``tests/test_v7_dataset.py::test_matches_train_sft_v7_batch_features_bitwise``
把这件事钉住）。

 重建历史列的原因：npz 的 ``op_hist`` 槽位是**错的**
------------------------------------------------------------
实测 ``data/sgf_19x19_full.npz``（34,202,713 行）：

=============================  ==========================  ==========
槽位                          实测等于（匹配率）           应然等于
=============================  ==========================  ==========
``op_hist[i, 0]``             ``moves[i-3]``  99.8%      ``moves[i-1]``
``op_hist[i, 1]``             ``moves[i-1]``  99.2%      ``moves[i-3]``
``op_hist[i, 2]``             恒 ``-1``                  ``moves[i-5]``
``my_hist[i, 0]`` ``moves[i-2]`` 99.9% ``moves[i-2]``
``my_hist[i, 1] / [2]``       恒 ``-1``                  ``moves[i-4/6]``
=============================  ==========================  ==========

根因在 ``scripts/build_dataset.py:93`` 的 ``return my[::-1], op[::-1]``：
``reversed(recent)`` 产出的 ``op`` **已经是新→旧**，那句 ``[::-1]`` 把它翻反了；
``my`` 只拿到 1 个元素所以翻转是空操作。

⇒ 后果（这就是为什么必须重建）：``feature_v7.history_five`` 的
``_HISTORY_SLOTS = ('op',0,'my',0,'op',1,'my',1,'op',2)`` 若直接吃 npz 的列，

  ==============  ==========================  ==========================
  通道            实际内容                       应然内容
  ==============  ==========================  ==========================
  ch9  (op 0)     ``moves[i-3]``               ``moves[i-1]``
  ch10 (my 0) ``moves[i-2]`` ``moves[i-2]``
  ch11 (op 1)     ``moves[i-1]``               ``moves[i-3]``
  ch12 (my 1)     空                            ``moves[i-4]``
  ch13 (op 2)     空                            ``moves[i-5]``
  ==============  ==========================  ==========================

**5 个历史通道错 4 个**，且「错」得毫无征兆（不抛异常、loss 照降）。
全局特征 ch0..ch4（5 个 pass 标志）与 ch14（``passWouldEndPhase``）同源同错。

 本模块**绕开**那个 bug，**不改** ``build_dataset.py``（裁定：不修、不重建
npz —— 重建 34.2M 行要几小时，而 A 阶段已改走 ``V7Dataset``）。
``tests/test_v7_dataset.py::test_raw_npz_op_hist_slots_are_swapped`` 把这个坑
钉成可执行断言，防止后人「优化」回去直接吃 ``op_hist``。

重建的口径
----------
不走「按局内位置奇偶取第 k 手」，而是**逐手问「这手是谁下的」**：

    ``moves[i-k]`` 的执子方 == ``to_play[i-k]``（按定义：行 ``i-k`` 的盘面是
    ``moves[i-k]`` 落子**之前**的局面，所以 ``moves[i-k]`` 由 ``to_play[i-k]``
    落下 —— 已实测确认，见 ``test_row_i_board_is_before_moves_i``），
    等于 ``to_play[i]`` 的是 ``my``，否则是 ``op``。

这样**不假设严格交替**。真实数据里 34,040,415 对同局相邻行中有 842 对
``to_play`` 不翻转（0.0025%），按奇偶取会把这批的槽位整体错一格。

 **pass 不需要特判**：``-1`` 在 ``my_hist`` / ``op_hist`` 里同时表示「pass」
与「无此手」，而这两者在 ch9..13 上输出**相同**（空），在全局 ch0..4 上也都是
「算作 pass 标志」—— 与 ``feature_v7.history_five`` / ``_resolve_history_length``
的既有约定一致。全量 ``moves`` 里 pass 只有 2,182 行（0.0064%）且局内位置全部
≥60，更不会落进「最近 5 手」的窗口。

off-by-one（本轮实测钉死的结论）
----------------------------------
``scripts/build_dataset.py`` 是**先 append 当前 board、再记录 target move**
（L303 先 ``cur['boards'].append(board.board.copy())``，L307 才
``cur['moves'].append(target)``）。实测 7443/7443 行（``boards[i-1]``→
``boards[i]`` 恰好只加一子、无提子的行）里那一子就在 ``moves[i-1]`` 且颜色恰为
``-to_play[i]``；反向看 ``boards[i]`` 上 ``moves[i]`` 有 7943 行为空、0 行为对方
色、仅 3 行为 to_play 色。⇒ **第 ``i`` 行的 board 是 ``moves[i]`` 落子之前的
局面**。于是

  * ``boards[i-1]`` / ``boards[i-2]`` 就是 ladder 三通道要的 ch15 / ch16 盘面
    （与 ``feature_v7_gather.LADDER_OFFSETS = (-1, -2)`` 一致）；
  * 过去 5 手是 ``moves[i-1]`` … ``moves[i-5]``；
  * 监督目标是 ``moves[i]``。

12 项 loss 在本语料上的消费状态
------------------------------
SGF 自打语料**没有**逐点 ownership、终局分差、seki 标签 ⇒ 这些头只能
``w=0``。``:data:`LOSS_W_KEYS``` 的每一项都**显式**给值（见下面那条警告）。

=================================  =========================  ====================
#     项                            行权重                     本语料
=================================  =========================  ====================
1     policy                        硬编码 1.0                 **开**（``moves[i]`` one-hot）
2     policy_opp                    ``w['policy_opp']``       **开**（``moves[i+1]``，跨局守卫）
3     value（3 分类 CE）             **硬编码 None ⇒ 恒开**     **开**（``winrates`` 符号）
4     ownership                     ``w['ownership']``         关（无标签）
5     scorebelief pdf               ``w['score']``             关
6     scorebelief cdf               ``w['score']``             关
7     scorestdev 自预测             **只有 game_weight**       **恒开**（见下）
8     scoremean                     ``w['score']``             关
9     lead                          ``w['lead']``              关
10    scoring                       ``w['scoring']``           关
11    futurepos                     ``w['futurepos']``         **开**（``boards[i+8/i+32]``）
12    seki                          ``w['seki']``              关
=================================  =========================  ====================

 **#3 与 #7 无法用 ``w`` 掩掉。** ``KataGoV7Loss.forward`` 里
``terms['value']`` 的权重写死成 ``None``（⇒ ``_weighted_mean`` 走 ``.mean()``），
``terms['score_stdev']`` 的权重是 ``labels['game_weight']``（**没有** ``w`` 键）。
⇒ ① A/C 阶段必须提供**硬三分类** ``outcome``（本类给：``winrates`` 的符号，
0 胜 / 1 负 / 2 无结果）；② #7 是**无标签的自洽项**（目标
``std(softmax(scorebelief))`` 由预测自己产生），本语料下它会一直开着 —— 这是
唯一在没有任何 scorebelief 监督的情况下仍有梯度的项，语义正确，但要注意它会
在没有 pdf/cdf 目标的情况下单独塑造 scorebelief 的分布尺度。

 **键缺失 ⇒ 权重全 1。** ``KataGoV7Loss.w_of()`` 在 ``w`` 里找不到键时返回
``torch.ones``。所以「屏蔽某头」必须**显式写 0**，不是「不写」——
:func:`finalize_loss_weights` 把这条变成硬约束：``:data:`LOSS_W_KEYS`` 里每一项
缺了就直接抛错。

 ``_build_labels`` 不给 ``w['lead']``，而 ``w_of('lead')`` 在缺键时返回全 1
⇒ 直接用它的 ``w`` 会拿 ``labels['score']`` 的**零占位**去训 lead 头，把头推向
恒 0。:func:`finalize_loss_weights` 显式补上这一项。
"""

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import SupervisedDataset, permute_move_vector
from src.data.feature_v7_gather import LADDER_OFFSETS, gather_neighbors

__all__ = [
    'HISTORY_SLOTS',
    'HISTORY_PAST',
    'LOSS_W_KEYS',
    'FROZEN_C_HEADS',
    'ACTION_SIZE',
    'BOARD_SIZE',
    'DEFAULT_RULES_FLAGS_V7',
    'V7ActionSizeError',
    'rebuild_history_columns',
    'reference_history_row',
    'verify_history_columns',
    'dihedral_batch',
    'spatial_global',
    'dense_move_target',
    'to_v7_loss_labels',
    'finalize_loss_weights',
    'V7Dataset',
]

#: ``my_hist`` / ``op_hist`` 各 3 格（与 ``feature_v7.HISTORY_MOVES``、``go_rules``
#: 的 ``pad3`` 一致）。V7 的 5 手交错用 3（op）+ 2（my）就够，剩下 2 格是余量。
HISTORY_SLOTS = 3

#: 往回看多少手。**至少** ``2 * HISTORY_SLOTS = 6``：严格交替时 my 拿偶数偏移
#: （2/4/6）、op 拿奇数偏移（1/3/5），少一格就会让某一侧的历史少一手。
HISTORY_PAST = 2 * HISTORY_SLOTS

#: V7 的动作空间：19×19 落点 + **pass 是真实的第 362 类**（不是特殊类）。
ACTION_SIZE = 19 * 19 + 1

#: V7 固定盘面（``feature_v7`` 的 ladder 移植本身就是固定盘面）。
BOARD_SIZE = 19

#: :attr:`V7Dataset.rules_flags` 的默认值 = 简单局（AREA + 无税 + simple ko +
#: 单子禁着 + 无 button），逐位等于 ``feature_v7.DEFAULT_RULES_FLAGS``。
#: 这里**自己定义**而不是 ``from ... import DEFAULT_RULES_FLAGS`` 再转出：
#: 这个值是本模块的**公开契约**（:attr:`V7Dataset.rules_flags`），值变了必须在这里
#: 看得见；``feature_v7.DEFAULT_RULES_FLAGS`` 是那边的实现细节。
#: 逐局规则化（``games.npz`` sidecar 的 ``g_rules``）**没有**接进来，原因见
#: :func:`spatial_global` 的标量约束：`calculate_area` 里的 `if` 收不了数组。
DEFAULT_RULES_FLAGS_V7 = 0

#: ``KataGoV7Loss.forward`` 会 ``w_of(...)`` 的**全部**键。
#:
#: **这份清单就是 ``w_of`` 键缺失返回全 1 的那个漏洞的封口**。清单之外的键
#: loss 不读；清单之内**缺一个**就意味着那一项被静默地按权重 1 训练。核对依据是
#: ``katago_v7_loss.py`` 里每一处 ``w_of('...')`` 的字面量。
LOSS_W_KEYS = ('policy_opp', 'ownership', 'score', 'lead', 'scoring',
               'futurepos', 'seki')

#: C 阶段训练 ``{policy, policy_opp, value}``、冻结其余 **9** 项对应的头。
#:
#: **这 9 项是「必须冻」，不是「省算力才冻」。** 它们是 SGF 自打语料**没有
#: 标签**的头（见模块 docstring 的 12 项表）。用零占位或 garbage 标签训，会把
#: B 阶段从 stdata 学到的东西**毁掉**——而毁掉的症状与「没训过」完全一样
#: （loss 曲线平、指标不动），事后分不清是「阶段没跑」还是「跑了但被 C 阶段
#: 拉回零」。A 阶段同理：用 ``w=0`` 屏蔽掉，而不是喂零标签。
#:
#: **显式清单，不由 ``w`` 表反推**：反推会随 ``w`` 表变动而漂移，而且语义
#: 不同 —— ``w`` 表说的是「这一批有没有标签」，这里说的是「这个阶段的训练
#: 目标里有没有它」。两个问题碰巧重合，但改一处不该动另一处。
#:
#: **9 项 ≠ 9 个参数组**：``score_stdev`` / ``score_mean`` / ``lead`` 共用
#: ``value_head.scores`` 这**一个** ``Linear(96→3)``，``scorebelief_pdf`` 与
#: ``scorebelief_cdf`` 共用 ``scorebelief_head``。参数级的映射见
#: ``scripts/train_v7.py::LOSS_TERM_TO_PARAM_PREFIX``（**只有那里**一份）。
FROZEN_C_HEADS = ('ownership', 'scorestdev', 'scorebelief_pdf', 'scorebelief_cdf',
                  'score_mean', 'lead', 'futurepos', 'scoring', 'seki')


class V7ActionSizeError(ValueError):
    """动作空间与 V7 的 362 类不符（盘面不是 19 路，或标签越界）。"""


# --------------------------------------------------------------------------- #
# 历史列重建
# --------------------------------------------------------------------------- #
def rebuild_history_columns(moves, to_play, game_ids, *, past=HISTORY_PAST,
                            chunk=1 << 21, verify=None):
    """从 ``moves`` / ``to_play`` / ``game_ids`` 重建 ``(my_hist, op_hist)``。

    语义（= ``feature_v7.history_five`` 期望的口径）
    --------------------------------------------------
        ``my_hist[i, s]`` = 局内往回数第 ``s+1`` 个**由 ``to_play[i]`` 落下**的手
        ``op_hist[i, s]`` = 局内往回数第 ``s+1`` 个**由对手落下**的手

    两列的 index 0 都是**最近一手**（与 ``go_rules`` 的 ``pad3`` /
    ``feature_v7._HISTORY_SLOTS`` 一致）。

    判据是**逐手的执子方**而不是局内位置的奇偶：``moves[i-k]`` 由
    ``to_play[i-k]`` 落下（按定义，行 ``i-k`` 的盘面是它落子**之前**的局面），
    等于 ``to_play[i]`` 就是 ``my``。 这条不能用奇偶代替：真实数据里有 842 对
    同局相邻行的 ``to_play`` 不翻转。

    跨局守卫
    --------
    ``game_ids[i-k] != game_ids[i]`` 的那一手**不取**，且不占槽位。
     不能 clamp 到边界行、也不能沿用上一行 —— 取到的盘面/着法**看起来完全
    合法**（它就是某个真实落点），只是不属于这一手。

    Args:
        moves: ``(N,)`` 整数，``0 <= m < 361`` 的落点、``-1`` = pass。
        to_play: ``(N,)`` ``±1``。
        game_ids: ``(N,)`` 整数，局号。**必须给** —— 缺它就没法判同局，
            而跨局取到的东西不报错。
        past: 往回看多少手，**必须 ≥ ``2 * HISTORY_SLOTS``**。
        chunk: 分块行数。 不是为了省时间，是为了**峰值内存**：全量 34.2M 行
            一次算需要 ``(N, past)`` 的中间量（int16 + 若干 bool 掩码），
            34.2M×6×(2+1+1+1+1) ≈ 1.2 GB；分块后降到 ``chunk×past×6`` B。
        verify: ``None`` / ``int``。给行数则在算完后调
            :func:`verify_history_columns` 抽查这么多行（默认 4096）。

    Returns:
        ``(my_hist, op_hist)``，各 ``(N, 3) int16``，历史不足处为 ``-1``。
    """
    past = int(past)
    if past < 2 * HISTORY_SLOTS:
        raise ValueError(
            f'past={past} < 2*{HISTORY_SLOTS}={2 * HISTORY_SLOTS}：严格交替时 '
            f'偶数偏移全归 my、奇数偏移全归 op，少一格就会有一侧历史少一手。')
    mv_all = np.asarray(moves)
    tp_all = np.asarray(to_play)
    gid_all = np.asarray(game_ids)
    n = int(mv_all.shape[0])
    if not (tp_all.shape[0] == gid_all.shape[0] == n):
        raise ValueError(
            f'列长度不一致：moves={n} / to_play={tp_all.shape[0]} / '
            f'game_ids={gid_all.shape[0]}；重建依赖三列同行。')
    if gid_all.dtype.kind not in 'iu':
        raise ValueError(f'game_ids dtype 必须是整数（实得 {gid_all.dtype}），'
                         f'否则跨局守卫失效')

    my = np.full((n, HISTORY_SLOTS), -1, dtype=np.int16)
    op = np.full((n, HISTORY_SLOTS), -1, dtype=np.int16)
    if n == 0:
        return my, op

    tp = tp_all.astype(np.int16, copy=False)
    offs = np.arange(1, past + 1, dtype=np.int64)
    for lo in range(0, n, max(1, int(chunk))):
        hi = min(lo + int(chunk), n)
        rows = np.arange(lo, hi, dtype=np.int64)
        j = rows[:, None] - offs[None, :]                  # (b, past)，可能为负
        ok = j >= 0
        jsafe = np.clip(j, 0, n - 1)
        # 跨局守卫。与 ``gather_neighbors`` 同一口径：先判越界再比 game_ids。
        ok &= gid_all[jsafe] == gid_all[rows][:, None]
        mv = np.where(ok, mv_all[jsafe], -1).astype(np.int16)
        same_player = (np.where(ok, tp[jsafe], np.int16(0))
                       == tp[lo:hi].astype(np.int16)[:, None])
        for take, dst in ((same_player & ok, my), (ok & ~same_player, op)):
            slot = np.cumsum(take, axis=1) - 1             # 该手在本侧是第几手
            for s in range(HISTORY_SLOTS):
                sel = take & (slot == s)
                hit = sel.any(axis=1)
                if not hit.any():
                    continue
                # 每行在每个 slot 上至多一手（cumsum 单调），所以 argmax 取
                # 第一个 True 就是那一手。
                k = sel.argmax(axis=1)
                dst[lo:hi, s][hit] = mv[hit, k[hit]]

    if verify is not None:
        verify_history_columns(my, op, mv_all, tp_all, gid_all,
                               nrows=int(verify), strict=True)
    return my, op


def reference_history_row(i, moves, to_play, game_ids, slots=HISTORY_SLOTS,
                          past=HISTORY_PAST):
    """**逐行 Python 循环**的参考实现 —— 与向量化版刻意用不同的算法。

    它只作为对拍基准存在：:func:`rebuild_history_columns` 的正确性由
    ``tests/test_v7_dataset.py::test_rebuild_matches_independent_reference``
    拿它逐格比对保证。 不要把这个函数「优化」成向量化 —— 那会让对拍失去意义。

     ``past`` 必须与 :func:`rebuild_history_columns` 的那个一致（默认
      :data:`HISTORY_PAST`）。 两侧若不一致，这里会在**某一侧的历史被填满**
      时提前停下（``while`` 条件里有 ``len(my) < slots and len(op) < slots``），
      于是少看了几手、把 ``-1`` 当成真值 —— 而 ``-1`` 在 ch9..13 上与 pass
      不可区分，**不报错**。往回看多少手不是「够填满槽位就行」：槽填满之后
      还要继续看，才能把 ``my[1]`` / ``op[2]`` 填上。
    """
    my, op = [], []
    k = 1
    past = int(past)
    while k <= past and k <= i:
        j = i - k
        if int(game_ids[j]) != int(game_ids[i]):
            break
        if int(to_play[j]) == int(to_play[i]):
            if len(my) < slots:
                my.append(int(moves[j]))
        else:
            if len(op) < slots:
                op.append(int(moves[j]))
        k += 1
    return (my + [-1] * slots)[:slots], (op + [-1] * slots)[:slots]


def verify_history_columns(my_hist, op_hist, moves, to_play, game_ids, *,
                           idxs=None, nrows=4096, strict=False):
    """按**独立参考实现**核对历史列；不符就报出具体行。

    Returns:
        ``{'checked': 命中数, 'mismatch': 不符数, 'examples': [(行, 侧, 期望, 实得)]}``

    Args:
        strict: ``True`` 时不符即抛 ``AssertionError``（生产构造走这条）。
    """
    my_hist = np.asarray(my_hist)
    op_hist = np.asarray(op_hist)
    mv_all = np.asarray(moves)
    tp_all = np.asarray(to_play)
    gid_all = np.asarray(game_ids)
    n = int(my_hist.shape[0])
    if idxs is None:
        # **固定取样而不是随机**：这个函数常在构造期调（verify=...），
        # 取样必须与随机数状态无关，否则同一份数据两次构造得到不同结论。
        step = max(1, n // max(1, int(nrows)))
        idxs = np.arange(0, n, step, dtype=np.int64)[:int(nrows)]
    bad = []
    for i in (int(x) for x in idxs):
        if i < 0 or i >= n:
            continue
        want_my, want_op = reference_history_row(i, mv_all, tp_all, gid_all,
                                                 past=HISTORY_PAST)
        for side, want, got in (('my', want_my, my_hist[i]),
                                ('op', want_op, op_hist[i])):
            if not np.array_equal(np.asarray(want, dtype=np.int16),
                                  np.asarray(got, dtype=np.int16)):
                bad.append((i, side, [int(x) for x in want],
                            [int(x) for x in got]))
                if len(bad) >= 8:
                    break
        if len(bad) >= 8:
            break
    out = {'checked': int(len(idxs)), 'mismatch': len(bad), 'examples': bad}
    if strict and bad:
        raise AssertionError(
            f'重建的历史列与独立参考实现不符（{len(bad)} 处，前几处 {bad[:4]}）。'
            f' 这不是「取样太严」—— 重建口径只有一种可能出错的地方就是槽位分配。')
    return out


# --------------------------------------------------------------------------- #
# 对称增强
# --------------------------------------------------------------------------- #
def dihedral_batch(x, tforms):
    """对 ``(B,C,H,W)`` 施加逐行 8 路 dihedral 变换（4 旋转 × 2 镜像）。

     **与 ``dataset.sample_batch_numpy`` 的增强是同一套约定**（``t >= 4`` 先翻
    W 轴 ``[..., ::-1]``，``k = t % 4`` 再顺时针转 ``k`` —— numpy 的 ``np.rot90``
    是逆时针，故取 ``-k``）。抄一份是因为 V7 的 22 通道空间量不走那条路径；
    方向写反不会报错，只是「loss 照降、棋力不涨」，所以
    ``tests/test_v7_dataset.py::test_dihedral_matches_dataset_augmentation``
    拿 dataset 的实现逐位对拍。

     **对称增广不在 :func:`spatial_global` 里做**：它必须与标签用的是**同一份**
    ``tforms``（见 :meth:`V7Dataset.sample_batch_v7`）。19 维全局量在棋盘翻转下
    不变，故只有 22 通道空间量要变换。
    """
    x = np.asarray(x)
    out = np.empty_like(x)
    for t in range(8):
        mask = tforms == t
        if not mask.any():
            continue
        arr = x[mask]
        if t >= 4:
            arr = arr[:, :, :, ::-1]
        k = t % 4
        if k:
            arr = np.rot90(arr, k=-k, axes=(2, 3))
        out[mask] = arr
    return out


# --------------------------------------------------------------------------- #
# 输入装配
# --------------------------------------------------------------------------- #
class _BatchGameRow:
    """``global_features_v7(game_row=...)`` 的逐批替身。

     **它是逐批的，不能是逐局的**：``feature_v7._komi_of`` 接受任何有 ``.komi``
    属性的对象并把它 reshape 成 ``(B,)``，所以一个装着 ``(B,) float32`` 数组的
    薄壳就够，不必为每行建一个 :class:`feature_v7.GameRow`。
    """

    __slots__ = ('komi',)

    def __init__(self, komi):
        self.komi = np.asarray(komi, dtype=np.float32).reshape(-1)


def spatial_global(dataset, idxs, *, boards=None, rules_flags=0):
    """按 dataset 的行号造出 V7 的两份输入 ``(spatial, global_features)``。

    Returns:
        ``(B,22,19,19) float16`` / ``(B,19) float16``。dtype 与
        ``feature_v7`` 一致，AMP 下不必再升精度。

     **与 ``scripts/train_sft.py::v7_batch_features`` 逐位相同**
    （``tests/test_v7_dataset.py::test_matches_train_sft_v7_batch_features_bitwise``
    钉住）。之所以这里要有一份而不能直接 import 那个：``src`` 反向依赖 ``scripts``
    会把训练脚本的 argparse / 设备探测拖进数据层（本仓 ``katago_v7_loss.huber``
    注释里记着同一条纪律）。**两份的实现必须一起改**，测试是防漂移的那道闸。

     **对称增强不在这里做** —— 见 :func:`dihedral_batch` 的说明。

     ``rules_flags`` 是**整批一个标量**，不是逐行。原因：``feature_v7.calculate_area``
    里是 ``if scoring == SCORING_TERRITORY:``，传数组会在那一行炸成
    「truth value is ambiguous」。逐局规则化要等 ``games.npz`` sidecar 的规则来源
    可靠之后再做（那时按 ``rules_flags`` 分组跑）。
    """
    from src.data.feature_v7 import global_features_v7, spatial_channels_v7

    idxs = np.asarray(idxs, dtype=np.int64)
    src = dataset.boards if boards is None else boards
    # ``game_ids`` **必须给**：主数据集是 162,298 局首尾相接的一根大数组、
    # **没有局的边界标记**，i 与 i-1/i-2 可以分属两局，而跨局取到的盘面
    # **看起来完全合法**（它就是某个真实盘面）只是不属于这一手。
    g = gather_neighbors(src, idxs, offsets=LADDER_OFFSETS,
                         game_ids=dataset.game_ids, to_play=dataset.to_play,
                         ko=dataset.ko)
    my = np.asarray(dataset.my_hist)[idxs]
    op = np.asarray(dataset.op_hist)[idxs]
    tp = np.asarray(dataset.to_play)[idxs]
    ko = np.asarray(dataset.ko)[idxs]
    b_now = np.asarray(src)[idxs]
    spatial = spatial_channels_v7(
        b_now, tp, ko, my, op,
        prev_board=g.boards[-1], prev_prev_board=g.boards[-2],
        rules_flags=rules_flags,
        prev_valid=g.valid[-1], prev_prev_valid=g.valid[-2])
    gl = global_features_v7(
        dataset.game_row_for(idxs), my, op,
        prev_board=g.boards[-1], prev_prev_board=g.boards[-2],
        rules_flags=rules_flags, to_play=tp,
        board_area=BOARD_SIZE * BOARD_SIZE)
    return spatial, gl


# --------------------------------------------------------------------------- #
# 标签
# --------------------------------------------------------------------------- #
def dense_move_target(moves, action_size=ACTION_SIZE):
    """``(B,)`` 着法 → ``(B, action_size)`` 稠密 one-hot；**``-1`` ⇒ 全零行**。

     ``-1`` 不是「随便哪个类」：``next_move`` 用 ``-1`` 表示「局末手 /
    跨局 / 下一手是 pass」，语义是**没有下一手**，对应 ``w['policy_opp'] == 0``。
    交给 ``F.one_hot(-1)`` 的行为不是本仓可以依赖的契约（不同 PyTorch 版本不同），
    所以**显式**置零 —— 权重本来就是 0，全零行既安全又语义正确。
    """
    mv = torch.as_tensor(np.asarray(moves)).reshape(-1).to(torch.long)
    valid = (mv >= 0) & (mv < int(action_size))
    safe = torch.where(valid, mv, torch.zeros_like(mv))
    out = F.one_hot(safe, int(action_size)).to(torch.float32)
    out[~valid] = 0.0
    return out


def to_v7_loss_labels(labels_dict, moves, *, action_size=ACTION_SIZE):
    """``SupervisedDataset`` 的 ``labels_dict`` → ``KataGoV7Loss`` 的键集合。

    这是 ``src/data/dataset.py::_build_labels`` 与
    ``src/networks/katago_v7_loss.py`` 之间**唯一**的翻译层：

    ==========================  ==========================================
    loss 要的键                 来源
    ==========================  ==========================================
    ``policy_player``           由**本行着法** ``moves`` 造 one-hot
    ``policy_opp``              由 ``labels_dict['next_move']`` 造 one-hot
    ``outcome``                 直取（**to_play 视角**，三分类
                                {0 胜, 1 负, 2 无结果}）
    ``ownership`` / ``score`` / 直取（本语料是占位零值，权重 0）
    ``scoring`` / ``seki`` /
    ``score_distr``             直取（842 桶；不传就走 ``sb_center/sb_upper``）
    ``game_weight``             直取
    ``w``                       :func:`finalize_loss_weights` 的产物
    ``futurepos``               **由 ``future`` 改名**（loss 叫 ``futurepos``，
                                dataset 叫 ``future``；值域与 -1 哨兵一致）
    ==========================  ==========================================

     **幂等**：对本函数已经产出的 dict 再调一次，值不变。所以
    ``scripts/train_sft.py::v7_loss_labels`` 叠在 ``V7Dataset`` 的 payload 上
    也是安全的（``tests/test_v7_dataset.py::test_loss_label_translation_matches_train_sft``
    用逐位对拍把这一点钉住）。

     **``w['futurepos']`` 绝不能改成 OR**（承重语义，不是风格）：loss 的 #11 把
    ``future`` reshape 成 ``(b,2,bs²)`` 之后**塌成逐样本标量**再乘**一个**权重，
    而 ``_weighted_mean`` 是 ``(per_sample*weight).mean()``（**刻意不除 Σw**）⇒
    权重的最小作用单位是「整块 2×bs²」。只活一路时若给 1，那一路的 -1 哨兵会被
    当真值拟合 tanh，头会学出一个恒 −0.76 的假平面。dataset 给的**与**语义
    （``w_h0 & w_h1``）是唯一正确的口径，本函数**原样透传**。

     **``outcome_black`` 不能由 ``outcome * to_play`` 推出**：``0 * -1 == 0``，
    于是「白胜」会被报成「黑胜」。直接用 dataset 给的键。
    """
    lbl = dict(labels_dict)
    lbl['policy_player'] = dense_move_target(moves, action_size)
    lbl['policy_opp'] = dense_move_target(lbl['next_move'], action_size)
    lbl['futurepos'] = lbl['future']
    return lbl


def finalize_loss_weights(w, batch, *, futurepos_enabled=False):
    """把 ``_build_labels`` 给的 ``w`` 补成 **loss 认得的、无缺键** 的形状。

    做三件事
    --------
    1. ``LOSS_W_KEYS`` 里**缺一个就抛**（缺键 ⇒ ``w_of`` 返回全 1 ⇒ 那一项被
       静默地按权重 1 训练，这是本仓最难查的一类静默错）；
    2. 补上 ``_build_labels`` 不给的 ``w['lead']``（**显式 0**，见模块 docstring）；
    3. futurepos 启用时要求 ``futurepos_h0`` / ``futurepos_h1`` 也在（它们不参与
       loss，但**缺了就没法区分「只活一路」与「两路都死」**）。

     **就地修改并返回同一个 dict** —— ``_build_labels`` 每次都新建，这个对象是
    本函数独占的；但调用方若跨 batch 复用同一个 ``w`` 会踩到，故 docstring 明写
    「返回值才是权威」。
    """
    if 'lead' not in w:
        # 本函数存在的**首要**理由，且必须在缺键检查**之前**补：
        # _build_labels 的 w 只有 policy/policy_opp/ownership/score/scoring/seki/
        # futurepos(+h0/h1)，唯独没有 lead，而 loss 的 #9 读 w_of('lead')。
        # 缺键时 w_of 返回全 1 ⇒ 拿 labels['score'] 的**零占位**训 lead 头，
        # 把头推向恒 0（不报错、loss 照降）。
        w['lead'] = np.zeros(int(batch), dtype=np.float32)

    missing = [k for k in LOSS_W_KEYS if k not in w]
    if missing:
        raise KeyError(
            f'w 缺 {missing}。 KataGoV7Loss.w_of() 在键缺失时返回 torch.ones —— '
            f'「不写」不是「屏蔽」，那一项会被静默地按权重 1 训练。要屏蔽就显式给 0。'
            f'（现有键：{sorted(w)}）')
    if futurepos_enabled:
        miss_fp = [k for k in ('futurepos_h0', 'futurepos_h1') if k not in w]
        if miss_fp:
            raise KeyError(
                f'futurepos 已启用但 w 缺 {miss_fp}。这两项不参与 loss，缺了不会报错，'
                f'只是「只活一路」与「两路都死」变得不可分 —— 而这两者的正确处理'
                f'完全不同（见 attach_futurepos 的「权重契约」段）。')
    return w


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class V7Dataset(SupervisedDataset):
    """22 通道 V7 训练的数据集（drop-in 替身，不是第二个数据集类）。

    与 :class:`SupervisedDataset` 的差别只有两处：

      1. ``my_hist`` / ``op_hist`` 在 ``super().__init__`` 之后被**重建**成从
         ``moves`` 现算的正确值（根因与实测数字见模块 docstring）；
      2. 多一个 :meth:`sample_batch_v7`，产出 V7 的两份输入 + 标签。

    继承来的、一行都没动的：``attach_soft`` / ``attach_futurepos`` /
    ``warm_futurepos`` / ``_futurepos_target`` / ``_build_labels`` /
    ``sample_batch_numpy``。⇒ 12 通道路径、以及 stdata 的邻行 gather 与跨局守卫
    全部沿用既有实现，**不写第二份**。

    Args:
        data: 与 ``SupervisedDataset`` 同构的列 dict（``boards`` / ``my_hist`` /
            ``op_hist`` / ``ko`` / ``moves`` / ``values`` / ``to_play``，可选
            ``game_ids`` / ``winrates`` / ``game_weights``）。
        n_channels: 只影响继承来的 12 通道路径；V7 固定 22，见 :data:`ACTION_SIZE`。
        game_komi: **可选**的逐局贴目。SGF 的 ``KM`` 在 sidecar ``games.npz`` 里，
            主 npz 没有这一列 ⇒ 不给时贴目按 0（``feature_v7._komi_of`` 的 NaN 分支
            口径），后果是全局 ch5（``selfKomi/20``）恒 0、ch18 的三角波按 0 算。
            可以给 ``{game_id: komi}`` 的 mapping，或**按 game_id 索引**的一维数组。
        verify_history: 构造后抽查多少行（``None`` = 不抽查）。
            小数据集上建议给 ``None``（测试逐格对拍即可）；34.2M 行上默认抽查
            4096 行，代价是几千次 Python 循环，可忽略。
    """

    def __init__(self, data, n_channels=12, *, soft_idx=None, soft_policy=None,
                 game_komi=None, game_rules_flags=None, verify_history=4096):
        super().__init__(data, n_channels=n_channels, soft_idx=soft_idx,
                         soft_policy=soft_policy)
        if self.board_size != BOARD_SIZE:
            raise V7ActionSizeError(
                f'V7 固定 {BOARD_SIZE}x{BOARD_SIZE}（动作空间 {ACTION_SIZE} 类），'
                f'实得 {self.board_size}x{self.board_size}。'
                f' 不要为了迁就小盘面去改动作空间 —— 官方 checkpoint 的 policy 头是'
                f'{ACTION_SIZE} 路，改了就接不上。')
        if self.game_ids is None:
            raise ValueError(
                'V7Dataset 必须有 game_ids 列。 邻行 gather（ladder 的 ch15/ch16、\n'
                '  futurepos 的 i+8/i+32）的跨局守卫靠它：主数据集是 162,298 局首尾\n'
                '  相接的一根大数组、没有局的边界标记，i 与 i±k 可以分属两局，而跨局\n'
                '  取到的盘面**看起来完全合法**（它就是某个真实盘面）只是不属于这一手\n'
                '  ⇒ 静默错特征/错标签。宁可没有这个数据集，也不要没有这根守卫。')
        if self.moves.dtype.kind not in 'iu':
            raise ValueError(f'moves dtype 必须是整数（实得 {self.moves.dtype}）；'
                             f'重建历史列要按整数坐标取值')
        self._game_komi = _normalize_game_scalar(game_komi, 'game_komi', float)
        self._game_rules = _normalize_game_scalar(game_rules_flags,
                                                  'game_rules_flags', int)

        # 覆写历史列。放在 super().__init__ 之后：父类构造期会把 data 里那份
        #   （错的）赋给 self.my_hist / self.op_hist。
        self.my_hist, self.op_hist = rebuild_history_columns(
            self.moves, self.to_play, self.game_ids,
            verify=verify_history)

    # ---- 局级标量 ----
    def game_row_for(self, idxs):
        """``(B,)`` 行号 → ``global_features_v7(game_row=...)`` 认的逐批对象。

        逐局查表而不是预先展开成 ``(N,)``：展开要多 34.2M × 4 B = 137 MB，
        而每批只需要 B 个数（一个 B 长度的 Python 循环，512 行时是几十微秒）。
        """
        if self._game_komi is None:
            return None            # ⇒ feature_v7._komi_of(None) ⇒ 贴目 0.0
        gid = np.asarray(self.game_ids)[np.asarray(idxs, dtype=np.int64)]
        komi = np.array([self._game_komi.get(int(g), 0.0) for g in gid],
                        dtype=np.float32)
        return _BatchGameRow(komi)

    @property
    def rules_flags(self):
        """整批规则位（见 :func:`spatial_global` 的标量约束）。默认 0 = 简单局。"""
        return DEFAULT_RULES_FLAGS_V7

    # ---- 取批 ----
    def sample_batch_v7(self, idxs, *, rng=None, augment=True, boards=None):
        """同步取一个 V7 batch：``(spatial, global_features, moves, labels_dict)``.

         **``labels_dict`` 是 dataset 形状**（``next_move`` / ``future`` /
        ``outcome`` / ``w`` …），不是 loss 形状。翻译由
        :func:`to_v7_loss_labels` 做 —— 这样本函数与
        ``scripts/train_sft.py::_prefetch_worker(v7=True)`` 的 payload
        **逐位同形**，两条取批路径可以互换。

         **``tforms`` 在这里抽一次，同时喂空间量与 ``_build_labels``。**
        不能改用 ``sample_batch_numpy(..., labels=True)``：那份 ``tforms`` 是它
        内部抽的、本方法拿不到；而「自己再抽一份」是**静默错标签**（输入翻了、
        标签没翻 ⇒ loss 照降、棋力不涨）。也不能反推 —— 盘面本身对某组变换不变时
        （空盘、对称局面；训练早期大量如此）8 个候选输出完全相同，反推会挑一个
        **不一定等于真值**的 ``t``。

         **不抽 ``tforms``（``augment=False``）时不碰 RNG**，与
        ``sample_batch_numpy`` 的评估路径同一口径。
        """
        idxs = np.asarray(idxs, dtype=np.int64)
        b = int(idxs.size)
        if rng is None:
            rng = np.random.default_rng()
        tforms = (rng.integers(0, 8, size=b) if augment
                  else np.zeros(b, dtype=np.int64))

        # ---- 标签：逐字走 dataset 唯一的标签构造器 ----
        lbl = self._build_labels(idxs, tforms, True)
        finalize_loss_weights(lbl['w'], b,
                              futurepos_enabled=self._fp is not None)

        # ---- 本行着法：口径逐字照抄 sample_batch_numpy ----
        #   非法/越界归一到 bs*bs（pass 类），且**只有** ``0 <= mv < bs*bs``
        #   被重映射。
        bs = self.board_size
        moves = np.full(b, bs * bs, dtype=np.int64)
        mv = np.asarray(self.moves[idxs], dtype=np.int64)
        valid = (mv >= 0) & (mv < bs * bs)
        moves[valid] = mv[valid]
        if augment:
            permute_move_vector(moves, tforms, bs)

        spatial, gl = spatial_global(self, idxs, boards=boards,
                                     rules_flags=self.rules_flags)
        return dihedral_batch(spatial, tforms), gl, moves, lbl

    def __repr__(self):
        return (f'V7Dataset(N={self.N}, board_size={self.board_size}, '
                f'futurepos={self.futurepos_status()}, '
                f'soft_rows={self.n_soft})')


#: 简单局默认（= ``feature_v7.DEFAULT_RULES_FLAGS``）。见 :data:`DEFAULT_RULES_FLAGS_V7`。

def _normalize_game_scalar(value, name, cast):
    """把「逐局标量」归一成 ``{game_id: value}`` 或 ``None``。

    接受 mapping（键会被强转成 ``int``）或**按 game_id 索引**的一维数组。
     **拒绝**「按行号索引」的数组：那样两种传法的下标语义不同，而数组没有
    自带说明，猜错的后果是**静默取到别局的贴目**（全局 ch5/ch18 跟着错，
    不报任何错）。所以数组必须比 `game_ids.max()+1` 短或等长，语义唯一。
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {int(k): cast(v) for k, v in value.items()}
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f'{name} 应是 mapping 或一维数组（按 game_id 索引），'
                         f'实得 {arr.shape}')
    return {i: cast(arr[i]) for i in range(arr.shape[0])}
