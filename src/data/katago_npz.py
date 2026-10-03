"""KataGo 分布式训练 npz（`katago/stdata/*.npz`）的读取器。

这个格式是什么
--------------
KataGo 官方 distributed training 数据的落盘格式，也是**官方模型能直接吃、
能直接训练**的完整张量集：输入（22 空间 + 19 全局）与全部监督目标都在同一个
npz 里。本模块把它翻译成本仓 V7 模型与 `KataGoV7Loss` 需要的形状。

⚠ **它同时也是「用官方特征平面」这条路线的根据**：因为输入已经是
`fillRowV7` 的产物，本仓**不需要**移植 `iterLadders` / `calculateArea` /
`passWouldEndPhase`（spec §5.1 的三个移植项）。

四条实测得来的约定（都不是猜的，每条都有单测）
------------------------------------------------
1. **bit-packed 空间通道**：`binaryInputNCHWPacked` 是 ``(N, 22, 46) uint8``，
   每通道 46 字节 = 368 bit，**只用前 361 bit**，``np.unpackbits(axis=-1,
   bitorder='big')`` 后 ``reshape(19, 19)`` 即得行主序盘面。
   ⚠ `np.unpackbits` 的 `axis` 默认是 ``None``（把输入**整体展平**），
   对 ``(N,46)`` 的输入会静默算错 —— 必须显式 ``axis=-1``。
2. **策略索引是固定 stride=19**：`index = r*19 + c`，**不是** `r*s+c`。
   实测 9/11/13 盘的非零索引最大分别到 156/176/200，全部 > s²，
   但都 < ``(s-1)*19+(s-1)`` ⇒ 确认是 stride=19。
   ⇒ 这与本仓 `build_dataset.py:297` 的 ``(r+off)*board_size+(c+off)``
   **同一口径**（19×19 局 off=0），两边索引可直接对齐。
3. **棋盘尺寸要从 ch0 反推**：`ch0` 是 on-board 掩码，19×19 局全 1，
   11×11 局只有 121 个 1。实测 stdata 里 **19×19 只占 63~75%**，
   其余是 7~18 盘以及 3.7~4.5% 的**非方阵**（计数不是完全平方）。
   V7 是固定 19×19（spec §1.3 明确不做其他尺寸）⇒ `to_v7_labels` 默认
   ``drop_non_19x19=True``。
4. **``globalTargetsNC`` 的列数随网络版本变**：实测 **64**（b40c768nbt、
   b28c512）vs **80**（tf3-b11c768）；``qValueTargetsNCMove`` 只有前两批有。
   ⇒ **绝不能硬编码列号**，必须按网络名查 `GLOBAL_TARGET_LAYOUT`。

已确认的列语义（实测，spec §5.3 权重组与之一致）
------------------------------------------------
=====  ==================================================================
0,1,2  硬 outcome（胜/负/无结果）
3      scoreMean（搜索侧的连续估计）
4-19   四组分差分布，每组 ``(P, P, P, 分数)``，**三 P 之和恒为 1.000000**
20     **实际终局分差**（建 scorebelief 目标用；和棋记 0）
    21     lead（非零率 39%）
    22     **varTimeLeft** —— 官方「winloss 期望到达时间」，见 `COL_VAR_TIME_LEFT`
    23     恒0（源码 `//Unused`）
    25/26  ``global_weight`` / ``w_policy_player``（恒 1）
    27/28  ``w_ownership`` / ``w_policy_opp``
    29     ``w_lead`` —— **非零数与 col21 的 lead 非零数逐行相等**，交叉验证通过
    30/31/32  ``policySurprise`` / ``policyEntropy`` / ``searchEntropy``
             （⚠ 曾被误判为 shortterm 两列，见下方注释）
    33/34  ``w_futurepos`` / ``w_scoring``
    35     ``w_value``（实测恒 0）
    47     komi（±7.5）
=====  ==================================================================
"""

import numpy as np

#: 盘面 stride / 落点数 / 动作数（361 落点 + pass）。stride 见模块 docstring 第 2 条。
BOARD_STRIDE = 19
BOARD_CELLS = BOARD_STRIDE * BOARD_STRIDE      # 361
ACTION_SIZE = BOARD_CELLS + 1                  # 362
PASS_INDEX = BOARD_CELLS                       # 361
SPATIAL_CHANNELS = 22
GLOBAL_CHANNELS = 19
#: 每通道的 packed 字节数：ceil(361/8) = 46，尾部 7 bit 不用。
PACKED_BYTES = 46
SCORE_DISTR_BINS = 842


class KatagoNpzLayoutError(KeyError):
    """遇到未登记的网络（列布局未知）。**必须报错，不许猜。**"""


#: 网络名 → ``globalTargetsNC`` 列数 + 是否有 Q 值。列布局随网络版本变
#: （实测 64 vs 80），硬编码列号会在换批次时静默错位。
GLOBAL_TARGET_LAYOUT = {
    'kata1-zhizi-b40c768nbt': {'cols': 64, 'has_qvalue': True},
    'kata1-tf3-b11c768': {'cols': 80, 'has_qvalue': True},
    'zzb28c512nfd4': {'cols': 64, 'has_qvalue': False},
}

#: 已确认的列号（两个网络版本上位置一致，见上表）。
COL_WIN, COL_LOSS, COL_NORESULT = 0, 1, 2
COL_SCORE_MEAN = 3
COL_FINAL_SCORE = 20            # 建 scorebelief 目标用；和棋 = 0
COL_LEAD = 21
COL_GLOBAL_WEIGHT = 25
COL_W_POLICY_PLAYER = 26
COL_W_OWNERSHIP = 27
COL_W_POLICY_OPP = 28
COL_W_LEAD = 29
COL_W_FUTUREPOS = 33
COL_W_SCORING = 34
COL_W_VALUE = 35
COL_KOMI = 47

#: ``varTimeLeft``（=官方 ``sv3Mul`` 六通道的**第 3 路**，下标 3）。
#:
#: 🔴 **2026-10-03 已由源码定案**（此前是「语义未定」的猜测列）。权威依据是
#: ``cpp/dataio/trainingwrite.cpp:603-616``：
#:
#:     // Expected time of arrival of winloss variance, in turns
#:     {
#:       double sum = 0.0;
#:       for(int i = whiteValueTargetsIdx+1; i<whiteValueTargets.size(); i++) {
#:         ...
#:         double variance = (nextWL - prevWL) * (nextWL - prevWL);
#:         sum += turnsFromNow * variance;
#:       }
#:       rowGlobal[22] = (float)sum;
#:     }
#:
#: 即「winloss 期望到达时间」，训练侧存的**已经是最终物理量**，不是 raw。
#: 实测（``zzb28c512nfd4`` 两个成员、9211 行）交叉验证：
#:   · 全非负（负值 0 个）⇒ softplus 类输出 ✔
#:   · 范围 [0, 465.53]、均值 10.95 ⇒ 与 KataGo 典型 5~20 同量级 ✔
#:   · 与 ``col3 scoreMean`` 相关 −0.003、与 ``col20 终局分差`` 相关 −0.003
#:     ⇒ 它是**方差/时间**量，不是分差 ✔
#:   · 非零率 87.1%
#:
#: ⚠ 注意它**不是** ``varianceTimeMultiplier=40`` 那个换算的输入 ——
#:   官方 nneval.cpp 读的是网络 raw ``sv3[3]`` 再乘 40，而 col22 是训练数据里
#:   已经算好的目标值。用它做监督时直接回归该值即可，不要再乘 40。
COL_VAR_TIME_LEFT = 22

#: 🔴 **官方 stdata 里不存在 shorttermWinlossError / shorttermScoreError 两列。**
#:
#: 已逐一核对 ``trainingwrite.cpp`` 里**全部** ``rowGlobal[n] =`` 赋值
#: （col 21~69全覆盖），结论：
#:
#:   · col 22 = varTimeLeft（见上）
#:   · col 23 = 恒 0（源码注释就写 ``//Unused``）
#:   · col 30 / 31 / 32 = ``policySurprise`` / ``policyEntropy`` / ``searchEntropy``
#:     —— ⚠ 这三列**曾经**被统计特征误判成 shortterm 两列（分布相近、中位数
#:     0.705 vs 0.708），查源码后被推翻。它们是搜索统计量，与 ``sv3[4:6]`` 无关。
#:
#: 原因：``shorttermX = sqrt(softplus(raw)² · mult)`` 需要**NN 的 raw 输出**，
#: 而训练数据是搜索侧生成的、当时还没有 NN 输出可依；KataGo 自己训这几路时
#: 也是先训主干、再用自对弈重解析补的（见 ``reanalysisData``）。
#:
#: ⇒ **后果**：``ValueHead.scores`` 扩到 6 通道后，后两路
#:   （``shorttermWinlossError`` / ``shorttermScoreError``）**没有对应标签**，
#:   无法监督。若强行训练只能靠自对弈阶段重新生成带 raw 输出的 stdata。
#:   目前这两路权重必须保持 0，导出到引擎后恒为 ``sqrt(softplus(bias)·mult)``
#:   的常量 —— 对 MCTS 无害（引擎只在 ``useUncertainty`` 时才读，且只影响
#:   pruning 启发式），但**不能声称已训练**。
UNRESOLVED_COLUMNS = (COL_SCORE_MEAN,)


def _unpack_bits(packed):
    """``(N, C, 46) uint8`` → ``(N, C, 19, 19) bool``，不校验通道数。

    通道数留给 `unpack_binary_input` 校验，因为本函数也被
    `board_size_from_packed` 以**单通道**（只要 ch0）的方式复用。
    """
    if packed.ndim != 3 or packed.shape[2] != PACKED_BYTES:
        raise ValueError(f'packed 形状不符：{packed.shape}，'
                         f'期望 (N, C, {PACKED_BYTES})')
    bits = np.unpackbits(packed, axis=-1, bitorder='big')   # (N,C,368)
    return bits[:, :, :BOARD_CELLS].reshape(-1, packed.shape[1],
                                            BOARD_STRIDE, BOARD_STRIDE) \
        .astype(bool)


def unpack_binary_input(packed):
    """``(N, 22, 46) uint8`` → ``(N, 22, 19, 19) bool``。

    只取每通道的前 361 bit（368 − 361 = 7 bit 尾部不用）。
    """
    if packed.ndim != 3 or packed.shape[1] != SPATIAL_CHANNELS:
        raise ValueError(f'binaryInputNCHWPacked 通道数不符：{packed.shape}，'
                         f'期望 (N, {SPATIAL_CHANNELS}, {PACKED_BYTES})')
    return _unpack_bits(packed)


def board_size_from_packed(packed):
    """由 ch0（on-board 掩码）的 1 的个数反推棋盘边长；**非方阵记 0**。

    ⚠ 不能用「开方后取整」：非方阵会静默变成某个邻近边长，把 18 盘的行
    当成 19 盘喂进模型。宁可直接标 0 丢掉。
    """
    n = packed.shape[0]
    cells = _unpack_bits(np.ascontiguousarray(packed[:, 0:1]))[:, 0] \
        .reshape(n, -1).sum(1)
    root = np.sqrt(cells.astype(np.float64))
    rounded = np.rint(root).astype(np.int64)
    ok = (rounded * rounded == cells) & (rounded > 0) & (rounded <= BOARD_STRIDE)
    return np.where(ok, rounded, 0)


def policy_index_is_in_board(idx, board_size):
    """索引是否落在该棋盘内：``r = idx // 19``、``c = idx % 19``，两者都 < s。"""
    r = idx // BOARD_STRIDE
    c = idx % BOARD_STRIDE
    return (r < board_size) & (c < board_size)


def topk_policy(policy_targets, k=16, action_size=ACTION_SIZE):
    """``(N, A)`` visit 计数 → ``((N,k) int64 索引, (N,k) float32 计数)``。

    按 visit 数降序取前 k。**同值时用小索引优先**（`np.argsort` 稳定排序的
    副作用，这里显式声明以免日后换排序实现导致 top-k 在平局处抖动、
    破坏「同一批数据每次读出同样的标签」）。

    ⚠ 这是**截断**：丢掉的尾部质量由调用方用 ``1 - Σtopk/Σall`` 记账。
    存稠密 ``(N,362)`` int16 在 34M 行下要 49.5 GB，top-16 只要 8.8 GB。
    """
    v = np.asarray(policy_targets)
    if v.ndim != 2 or v.shape[1] != action_size:
        raise ValueError(f'policy 目标形状不符：{v.shape}，'
                         f'期望 (N, {action_size})')
    kk = min(int(k), action_size)
    # 稳定降序：先取 -v 的 argsort（kind='stable'），平局保原顺序（索引小在前）
    order = np.argsort(-v, axis=1, kind='stable')[:, :kk]
    return order.astype(np.int64), np.take_along_axis(v, order, axis=1) \
        .astype(np.float32)


def read_katago_npz(npz, network=None):
    """读一个 stdata npz，补上派生量并校验布局。

    Args:
        npz: ``np.lib.npyio.NpzFile`` 或已 ``np.load`` 的对象。
        network: 网络名（含或不含 ``kata1-`` 前缀）。**不给就从
            ``globalTargetsNC`` 的列数反推**，但反推只能区分 64/80 两族，
            分不出同族的两个网络 ⇒ 生产路径请显式传。

    Returns:
        dict，额外含 ``_board_size`` / ``_layout`` / ``_spatial``。
    """
    d = {k: npz[k] for k in npz.files}
    cols = d['globalTargetsNC'].shape[1]
    if network is None:
        cand = [n for n, v in GLOBAL_TARGET_LAYOUT.items() if v['cols'] == cols]
        if len(cand) != 1:
            raise KatagoNpzLayoutError(
                f'globalTargetsNC 是 {cols} 列，无法唯一反推网络（候选 {cand}）；'
                f'请显式传 network=。硬猜会让权重列静默错位。')
        network = cand[0]
    key = next((n for n in GLOBAL_TARGET_LAYOUT if n in network), None)
    if key is None:
        raise KatagoNpzLayoutError(
            f'未登记的网络 {network!r}；已知 {sorted(GLOBAL_TARGET_LAYOUT)}。'
            f'新增网络前请先用 metrics_pytorch.py / dataio.cpp 确认其列布局。')
    layout = GLOBAL_TARGET_LAYOUT[key]
    if layout['cols'] != cols:
        raise KatagoNpzLayoutError(
            f'{network} 登记为 {layout["cols"]} 列，实测 {cols} 列。')
    has_q = 'qValueTargetsNCMove' in d
    if has_q != layout['has_qvalue']:
        raise KatagoNpzLayoutError(
            f'{network} 的 Q 值存在性与登记不符：登记 {layout["has_qvalue"]}，'
            f'实测 {has_q}')
    packed = d['binaryInputNCHWPacked']
    d['_layout'] = dict(layout, network=key)
    d['_board_size'] = board_size_from_packed(packed)
    d['_spatial'] = unpack_binary_input(packed)
    d['_n'] = int(packed.shape[0])
    return d


def to_v7_labels(d, network=None, policy_topk=16, drop_non_19x19=True):
    """stdata 一行批次 → `KataGoV7Loss` 的标签 dict。

    做的事：
    - 按网络名分派列布局（`GLOBAL_TARGET_LAYOUT`）；
    - 解包 22 通道 → ``(N,22,19,19)`` bool，并给出 19×19 的 on-board 掩码；
    - visit 计数 → top-K 稀疏 policy 目标；
    - 逐行 ownership / seki / futurepos / scoring；
    - 权重列按 spec §5.3 拆成 ``w`` 字典；
    - ``scoreDistrN``（每行和 100、恰 2 个非零）归一化成概率。

    Args:
        drop_non_19x19: 丢掉非 19×19 的行。V7 固定 19×19（spec §1.3），
            而 stdata 里 19×19 只占 63~75% ⇒ 默认丢。**返回的 dict 里
            ``_dropped`` 记录丢了多少**，别让它静默发生。
    """
    if network is not None or '_layout' not in d:
        d = read_katago_npz(d, network=network)
    n = d['_n']
    keep = np.ones(n, dtype=bool)
    if drop_non_19x19:
        keep = d['_board_size'] == BOARD_STRIDE
    idx = np.nonzero(keep)[0]

    def sel(arr):
        return np.asarray(arr)[idx]

    pol = d['policyTargetsNCMove']
    pi_idx, pi_val = topk_policy(sel(pol[:, 0, :]), k=policy_topk)
    po_idx, po_val = topk_policy(sel(pol[:, 1, :]), k=policy_topk)

    g = d['globalTargetsNC']
    vt = d['valueTargetsNCHW']
    # 硬 outcome：win / loss / noResult 三列取 argmax（和棋时三者皆 0，
    # argmax 会落到 0=胜 —— 显式判「全 0 ⇒ 无结果」）
    hard = sel(g[:, 0:3])
    outcome = hard.argmax(1).astype(np.int64)
    outcome[hard.sum(1) == 0] = 2                      # 2 = 无结果

    sb = sel(d['scoreDistrN']).astype(np.float32)
    sb_sum = sb.sum(1, keepdims=True)
    sb = np.divide(sb, np.where(sb_sum > 0, sb_sum, 1.0))

    labels = {
        'policy_player_sparse': (pi_idx, pi_val),
        'policy_opp_sparse': (po_idx, po_val),
        'outcome': outcome,
        'score': sel(g[:, COL_FINAL_SCORE]).astype(np.float32),
        'score_mean_hint': sel(g[:, COL_SCORE_MEAN]).astype(np.float32),
        'lead_hint': sel(g[:, COL_LEAD]).astype(np.float32),
        # varTimeLeft —— 官方 `sv3Mul` 六通道的下标 3（语义见 COL_VAR_TIME_LEFT）。
        # ⚠ 存的是**最终物理量**，不是 raw：官方训练侧算好才落盘
        #   （trainingwrite.cpp:603-616），而 `ValueHead` 输出的
        #   `var_time_left` 已乘过 VARIANCE_TIME_MULTIPLIER，两端口径一致。
        'var_time_left': sel(g[:, COL_VAR_TIME_LEFT]).astype(np.float32),
        'score_distr': sb,
        'ownership': sel(vt[:, 0]).astype(np.float32)[:, None, :],
        'seki': sel(vt[:, 1]).astype(np.float32)[:, None, :],
        'futurepos': sel(vt[:, 2:4]).astype(np.float32),
        'scoring': sel(vt[:, 4]).astype(np.float32)[:, None, :],
        'game_weight': sel(g[:, COL_GLOBAL_WEIGHT]).astype(np.float32),
        'w': {
            'policy_opp': sel(g[:, COL_W_POLICY_OPP]).astype(np.float32),
            'ownership': sel(g[:, COL_W_OWNERSHIP]).astype(np.float32),
            'score': sel(g[:, COL_W_POLICY_PLAYER]).astype(np.float32),
            'lead': sel(g[:, COL_W_LEAD]).astype(np.float32),
            'futurepos': sel(g[:, COL_W_FUTUREPOS]).astype(np.float32),
            'scoring': sel(g[:, COL_W_SCORING]).astype(np.float32),
        },
        'global': sel(d['globalInputNC']).astype(np.float32),
        'komi': sel(g[:, COL_KOMI]).astype(np.float32),
        # ---- 附带的输入侧张量（模型直接吃，不重算）----
        'spatial': d['_spatial'][idx].astype(np.float32),
        'board_mask': (d['_board_size'][idx] == BOARD_STRIDE)[:, None, None, None]
        * np.ones((1, 1, BOARD_STRIDE, BOARD_STRIDE), dtype=bool),
        # ---- 诊断 ----
        '_dropped': int(n - idx.size),
        '_kept': int(idx.size),
        '_network': d['_layout']['network'],
    }
    if 'qValueTargetsNCMove' in d:
        q = sel(d['qValueTargetsNCMove'])
        labels['q_value'] = q.astype(np.float32)
    return labels
