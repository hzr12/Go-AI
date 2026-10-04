# -*- coding: utf-8 -*-
"""V7 空间通道 B2：ch3/4/5（气 1/2/3）+ ch9–13（过去 5 手）+ 历史门控回退。

被测对象 `src/data/feature_v7.py`：
  · `liberties_123`   —— ch3/4/5
  · `history_five`    —— ch9/10/11/12/13
  · `history_gated`   —— ch15/ch16 的取盘面规则（ladder 通道用）

**并且包含与官方 stdata 的对拍**（最后两条，`skipif`）：spec §9.4 说 stdata 是本仓
**唯一**的可执行 oracle。ch3/4/5 那一组实测**逐位相等**（400 行），
ch9–13 那一组实测颜色分布与 spec §2.2 的「opp, pla, opp, pla, opp」吻合
（ch9 100% 落在 opp 子上、ch10 99.4% 落在 pla 子上，……）。
 **ch18/19 不可对拍**（官方先提死子），理由与实测数字见
`src/data/feature_v7.py::area_ownership_map` 的 docstring。

复用 `tests/test_go_feature_planes_v21.py` 的组织方式（逐条钉一个行为 + 说清
「为什么这条断言结构上不可省」），但**不复用它的判据**：那些钉的是标量 vs 批量的
两条路径，而这里钉的是「V7 通道 vs 官方 stdata」。

运行：
    python -m pytest tests/test_v7_planes.py -q
"""

import io
import os
import tarfile
import time

import numpy as np
import pytest

from src.data.feature_v7 import (
    LADDER_CH_BASE,
    history_five,
    history_gated,
    liberties_123,
)

N = 19
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STDATA_TAR = os.path.join(REPO, 'katago', 'stdata',
                          'zzb28c512nfd4-s8264801024-d4596884264.tar')
needs_stdata = pytest.mark.skipif(
    not os.path.isfile(STDATA_TAR),
    reason='需要 katago/stdata 下的 tar（不在仓库里）')


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def board(rows):
    """ASCII 夹具 → `(1,19,19)` int8。`X`=+1(黑/pla)、`O`=−1(白/opp)、`.`=空。

    短行**右侧补 `.` 到 19**（左对齐）。为什么值得这样而不是每行手写 19 个字符：
    这些形状（U 形块、四面包围）的正确性全靠「一眼能看出这个形状」，
    写成 `X`/`O`/`.` 的小块后断言可以逐字符对照，人和测试看的是同一张图；
    补零是机械的，写 19 遍只会让人不再看它。
    """
    assert len(rows) <= N, f'最多 {N} 行，实得 {len(rows)}'
    a = np.zeros((1, N, N), dtype=np.int8)
    for r, line in enumerate(rows):
        assert len(line) <= N, f'第 {r} 行超过 {N} 个字符'
        for c, ch in enumerate(line):
            if ch == 'X':
                a[0, r, c] = 1
            elif ch == 'O':
                a[0, r, c] = -1
            elif ch != '.':
                raise ValueError(f'非法字符 {ch!r}')
    return a


def pts(channels, ch):
    """返回 `(C,n,n)` 里第 `ch` 个通道上为真的点（按行主序排序），便于逐点比。"""
    rr, cc = np.nonzero(channels[ch])
    return sorted(zip(rr.tolist(), cc.tolist()))


def bucket_at(out, r, c):
    """`(3,n,n)` 输出里点 (r,c) 落在哪个桶（0/1/2 = ch3/ch4/ch5）；不在任何桶则 −1。"""
    for k in range(3):
        if out[0, k, r, c]:
            return k
    return -1


def _blank():
    return '.' * N


# 6 颗子的整块（rows 2-3 × cols 1-3），唯一的气是 (1,2)：气数 1 ⇒ 6 个点全进 ch3。
_SIX_STONE = [
    '',                # row0 全空（不与该块相邻）
    '.O.O',            # row1: (1,2) 是唯一的气，其余邻点封死
    'OXXXO',           # row2
    'OXXXO',           # row3
    '.OOO',            # row4
] + [_blank()] * (N - 5)

# U 形（杯形）块：5 颗子围住唯一的空点 (2,2)，而该空点同时邻接 **3 颗子**。
#   · 按「去重坐标数」⇒ 气数 = 1 ⇒ 5 个点全进 ch3；
#   · 按「入射次数」   ⇒ 气数 = 3 ⇒ 5 个点一个都不进 ch3。
# 这条形状就是判别「口径」的那把尺（与 go_rules 的 U 形回归同源）。
_U_SHAPE = [
    '.OOO',            # row0
    'OXXXO',           # row1
    'OX.XO',           # row2: (2,2) 是那个被 3 颗子围住的气
    '.OOO',            # row3: 封住下方
] + [_blank()] * (N - 4)

# 单子三种气数：黑子恒在 (1,1)，用白子封掉四个邻点中的 1/2/3 个。
# 这些夹具里的**白子自己也会落进某个桶**（封边用的白子通常有 2~4 气），
#   所以对它们一律用「只断言那颗黑子」的 `bucket_at`，不用整张图比较 ——
#   整张图比较会把「白子的桶对不对」混进来，而白子的归属由上面两条夹具负责。
_ONE_STONE_1_LIB = board([
    '.O',              # (0,1)=O
    'OXO',             # (1,1)=X  被三面包住 ⇒ 只剩 (2,1) 一个气
    '..',
])
_ONE_STONE_2_LIB = board([
    '.O',
    'OX.',             # (1,2) 与 (2,1) 两个气
    '..',
])
_ONE_STONE_3_LIB = board([
    '.O',
    '.X',              # (2,1)/(1,0)/(1,2) 三个气
])

# 开放盘面中段的 2×2 黑块：8 气 ⇒ 三个桶一个都不进。
_OPEN_2X2 = board([
    '',
    '',
    '.XX.',
    '.XX.',
])


# --------------------------------------------------------------------------- #
# B2-1 · liberties_123
# --------------------------------------------------------------------------- #
def test_liberties_123_three_buckets_are_disjoint_and_cover_nothing_else():
    """三个桶**互斥**、dtype bool，且非子点永不点亮。

    「非子点永不点亮」是最容易在向量化实现里漏的一条：`per = lib[labelled]`
    这类物化会把非子点也带上一个气数（那里其实是 `lib[0] == 0`），一旦去重核
    的索引口径改了，`==1/==2/==3` 就可能在**空点**上命中。这条断言把它钉住。
    """
    b = board(['', '', '.X.X.', '.XXX.'])
    out = liberties_123(b)
    stones = b != 0
    assert out.shape == (1, 3, N, N)
    assert out.dtype == np.bool_
    assert not (out[:, 0] & out[:, 1]).any(), 'ch3/ch4 必须互斥'
    assert not (out[:, 0] & out[:, 2]).any(), 'ch3/ch5 必须互斥'
    assert not (out[:, 1] & out[:, 2]).any(), 'ch4/ch5 必须互斥'
    assert not (out & ~stones).any(), '气桶只能点亮棋子'


def test_liberties_123_single_stone_each_bucket():
    """手工构造的单子：**逐桶**断言 1/2/3 气分别落进 ch3/ch4/ch5。

    只断言那颗黑子（`bucket_at`）而不是整张图 —— 封边用的白子自己也有 2~4 气、
    必然落进某个桶，把它们一起比较等于把另一件事混进这条断言。
    「落在桶 k 且不落在其他两桶」这条逐桶判定，比只断言「落在桶 k」更能抓住
    「两个桶被同一个块同时点亮」的错。
    """
    for fixture, bucket, name in (
            (_ONE_STONE_1_LIB, 0, '1 气 → ch3'),
            (_ONE_STONE_2_LIB, 1, '2 气 → ch4'),
            (_ONE_STONE_3_LIB, 2, '3 气 → ch5')):
        out = liberties_123(fixture)
        got = bucket_at(out, 1, 1)
        assert got == bucket, f'{name}：(1,1) 应落在 ch{3 + bucket}，实得 ch{3 + got}'
        assert sum(int(out[0, k, 1, 1]) for k in range(3)) == 1, \
            f'{name}：一颗子不该同时落进两个桶'


def test_liberties_123_group_with_more_than_three_liberties_is_in_no_bucket():
    """气 ≥ 4 的整块**一个桶都不进**（2×2 黑块在开放盘面上是 8 气）。

    「桶只有 1/2/3」这条很容易被写成 `else: 进 ch5` 或 `>=3`。
    `>=3` 会让 4 气的块污染 ch5（官方 ch5 的语义是**恰好** 3 气 ——
    与 stdata 的逐位相等就是靠这条），所以显式钉住。
    """
    out = liberties_123(_OPEN_2X2)
    for k in range(3):
        assert pts(out[0], k) == [], f'2×2 黑块有 8 气，不该进 ch{3 + k}'


def test_liberties_123_two_stones_shared_liberty_is_deduplicated():
    """U 形块：唯一的气 (2,2) 同时邻接 **3 颗子** ⇒ 按去重坐标数是 **1 气**。

    这是口径的判别式（brief 点名的那个坑）：
      · 去重坐标（官方 = `_group_liberty_count` 的 `set`）⇒ 气数 1 ⇒ 进 ch3；
      · 入射次数（同块三颗子围住一个气点时各算一次）⇒ 气数 3 ⇒ 进 ch5。
    `go_rules.py:170-177` 记着上一次入射计数口径造成的真 bug（U 形块两个 bucket
    同时漏），所以这条形状必须在本模块再钉一次 —— 两份实现将来若有一份改口径，
    这里立刻红。
    """
    out = liberties_123(board(_U_SHAPE))
    cells = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3)]
    for r, c in cells:
        assert bucket_at(out, r, c) == 0, \
            f'({r},{c}) 是 U 形块的一员，去重后气数 1 ⇒ 应在 ch3；实得 ' \
            f'ch{3 + bucket_at(out, r, c)}（若报 ch5 说明按入射次数数的）'
    assert len(pts(out[0], 0)) == 5, f'ch3 应恰好点亮 5 颗子，实得 {pts(out[0], 0)}'


def test_liberties_123_six_stone_group_shares_one_liberty():
    """6 颗子的整块，**唯一的气**被整块共享 ⇒ 气数 1 ⇒ 6 个点全进 ch3。

    brief 点名的形状。取值上它已经证明「气是整块共享的」：若实现按颗子各自算气，
    (2,1)/(2,2)/(2,3)/(3,1)/(3,2)/(3,3) 的气数各不相同、落点就会散开，
    六个点不会同时落进同一个桶。
    """
    out = liberties_123(board(_SIX_STONE))
    expect = [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    for r, c in expect:
        assert bucket_at(out, r, c) == 0, \
            f'({r},{c}) 应落在 ch3（整块气数 1），实得 ch{3 + bucket_at(out, r, c)}'
    lit = {(r, c) for k in range(3) for r, c in pts(out[0], k)} & set(expect)
    assert len(lit) == 6, f'6 颗子应全部落桶，实得 {sorted(lit)}'


def test_liberties_123_dedup_nucleus_called_once_per_colour_regardless_of_group_size(
        monkeypatch):
    """**成本不变式（行为性）**：去重核算法的调用次数**与块数、块的大小都无关**。

    为什么这条断言有意义、而纯值断言结构上抓不到：
      · 按 k² 的退化实现（每颗子重扫整块）**取值完全正确** —— 只会变慢，
        所以上面所有值断言都抓不到它。
      · 退化实现的形态是「每个 (子, 气) 关联都重算一次」，把它换成批量实现后
        本函数只应调用 `_distinct_liberty_counts` **每色一次**（批内所有块一次算完）。
      · 这里数调用次数，并钉死「6 颗子的 1 块」与「6 个单子块」以及「空盘」
        三种规模**次数相同**（都是 2 = 两个颜色各一次）。任何按块 / 按子分派的
        实现都会让这三个数发散。
     这条**只**约束本模块的分派粒度；`_distinct_liberty_counts` 内部的 O(B·n²)
      由 `go_rules` 自己的测试负责（spec §7.1：那边一字不改）。
    """
    import src.data.feature_v7 as fv

    calls = {'n': 0}
    real = fv._distinct_liberty_counts

    def counting(labelled, num, empty):
        calls['n'] += 1
        return real(labelled, num, empty)

    monkeypatch.setattr(fv, '_distinct_liberty_counts', counting)

    six_one_group = board(_SIX_STONE)
    six_one_group[0, 0, 18] = -1            # 加一颗白子，让两个颜色都非空
    six_groups = np.zeros((1, N, N), dtype=np.int8)
    for r in range(0, 18, 3):
        six_groups[0, r, r] = 1
    six_groups[0, 0, 18] = -1

    per_shape = []
    for name, b in (('6 颗子 / 1 块', six_one_group),
                    ('6 个单子块', six_groups)):
        calls['n'] = 0
        liberties_123(b)
        per_shape.append((name, calls['n']))
    assert [c for _, c in per_shape] == [2, 2], (
        f'去重核的调用次数应恒为 2（两个颜色各一次），与块数/块大小无关；'
        f'实得 {per_shape} —— 按块或按子分派都说明退化成了 O(k²)')

    calls['n'] = 0
    liberties_123(np.zeros((1, N, N), dtype=np.int8))
    assert calls['n'] == 0, '空盘上不该调用去重核（颜色掩码为空 ⇒ 整段短路）'


def test_liberties_123_large_single_group_is_not_quadratic():
    """**成本上界（行为性）**：60 颗子的**单块**不许比空盘贵出一个数量级。

    为什么用「比值」而不是绝对毫秒数：向量化下两者都落在 numpy 的固定开销里
    （~50 µs），绝对阈值会被机器噪声/库版本搞红；比值只对「量级退化」敏感。
    60 颗子按 O(k²) 逐颗子重扫是 3600 次 Python 级邻接判定（约 3–4 ms），
    相对空盘的 ~50 µs 已经是 60× 以上；向量化实测在同一量级（≈2×）。
    阈值取 20×：既能抓住 k² 退化，又留足了库版本差异的余量。
     这是一条**上界**测试，所以它证不了「确实线性」，只证「没有二次退化」。
      线性由上面那条调用次数不变式 + `_distinct_liberty_counts` 的 O(B·n²)
      共同承担（那条是可证的，计时这条是防回归的粗网）。
    """
    blank = np.zeros((1, N, N), dtype=np.int8)
    big = np.zeros((1, N, N), dtype=np.int8)
    big[0, 4:10, 4:10] = 1                      # 6×10 = 60 颗子的整块

    def median_cost(fn, b, reps=25):
        for _ in range(5):
            fn(b)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(b)
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[reps // 2]

    base = median_cost(liberties_123, blank)
    cost = median_cost(liberties_123, big)
    assert cost < 20 * max(base, 1e-6), (
        f'60 颗子单块耗时 {cost * 1e3:.2f} ms，是空盘（{base * 1e3:.3f} ms）的 '
        f'{cost / max(base, 1e-6):.1f}× —— 超过 20× 说明按颗子重扫整块了')

    out = liberties_123(big)
    # 6×10 整块在开放盘面上：上/下各 10 + 左/右各 6 = 32 气 ⇒ 三个桶都不进。
    for k in range(3):
        assert pts(out[0], k) == [], f'32 气的整块不该进 ch{3 + k}'


def test_liberties_123_matches_17ch_union():
    """V7 的 ch3/ch4 == 17 通道 ch10|ch11 与 ch14|ch15 的**逐位**并集。

    这条钉的是**跨模块的口径一致性**，不是单点取值：
    17 通道的 10/11（气 1）与 14/15（气 2）是**分色**的两份，V7 的 3/4/5 是
    **不分色**的三份，但底层量是同一个「整块去重气数」。若将来有人在本模块里
    另写一份气数（而不是复用 `_distinct_liberty_counts`），两条路径就会各算各的，
    而这种漂移**不会**让任何一个模块自己的测试变红 —— 只有这条跨路径对拍会红。
    （`feature_planes_batched` 通道 8 有一处已知口径差，**与 10/11/14/15 无关**，
     那条见它的 docstring。）
    """
    from src.game.go_rules import GoBoard

    rng = np.random.default_rng(20261002)
    for trial in range(6):
        x = rng.random((1, N, N))
        boards = np.where(x < 0.24, 1, np.where(x < 0.48, -1, 0)).astype(np.int8)
        to_play = np.array([1 if trial % 2 == 0 else -1], dtype=np.int8)
        planes = GoBoard.feature_planes_batched(
            boards, np.full((1, 3), -1, np.int16), np.full((1, 3), -1, np.int16),
            to_play, n_channels=17)
        mine = liberties_123(boards, to_play)
        lib1 = (planes[:, 10].astype(bool) | planes[:, 11].astype(bool))
        lib2 = (planes[:, 14].astype(bool) | planes[:, 15].astype(bool))
        assert np.array_equal(mine[:, 0], lib1), \
            f'第 {trial} 个盘面：ch3 != ch10|ch11'
        assert np.array_equal(mine[:, 1], lib2), \
            f'第 {trial} 个盘面：ch4 != ch14|ch15'


def test_liberties_123_rejects_wrong_rank():
    with pytest.raises(ValueError, match='boards 形状不符'):
        liberties_123(np.zeros((N, N), dtype=np.int8))


# --------------------------------------------------------------------------- #
# B2-2 · history_five
# --------------------------------------------------------------------------- #
def test_history_five_channel_order_is_opp_pla_opp_pla_opp():
    """通道顺序 = `opp, pla, opp, pla, opp`（spec §2.2，自 nextPlayer 视角）。

    钉死**具体是哪一格**而不是只钉「有 5 个点」：
    因为上一手永远是对手落的，所以**最新的一格属于 opp**（ch9）而不是 pla。
    写反了（`pla, opp, ...`）会让整组历史通道的语义左右互换，而这种错不会抛
    异常、只会表现为训练分布偏移。
    """
    my = np.array([[11, 22, 33]], dtype=np.int16)   # pla 的手，0 = 最近
    op = np.array([[44, 55, 66]], dtype=np.int16)   # opp 的手，0 = 最近
    out = history_five(np.zeros((1, N, N), np.int8), my, op,
                       np.array([1], np.int8))
    want = [[(44 // N, 44 % N)], [(11 // N, 11 % N)], [(55 // N, 55 % N)],
            [(22 // N, 22 % N)], [(66 // N, 66 % N)]]
    for ch in range(5):
        assert pts(out[0], ch) == want[ch], \
            f'ch{9 + ch} 应落在 {want[ch]}（源 = {"opp" if ch % 2 == 0 else "pla"} ' \
            f'第 {ch // 2 + 1} 手），实得 {pts(out[0], ch)}'


def test_history_five_pass_occupies_no_channel_and_does_not_shift():
    """**pass 不占通道**，而且**不把更早的手往前挤**（spec §2.2）。

    这两件事必须一起说，因为它们是不同的错：
      · 只钉「pass 那格为空」的话，一个「跳过 pass、其余顺序前移」的实现
        同样能过 —— 但它会把历史的时间顺序弄错 1 手；
      · 这里断言**每一格具体落谁**，前移实现立刻红（ch9 会变成空的、
        ch10 会拿到 opp 的次近一手）。
    `-1` 同时表示 pass 与「历史不足」，两者在落点通道上的输出相同（空），
    所以这里不必、也不能区分 —— 区分是全局 ch0-4 的活。
    """
    my = np.array([[-1, 22, 33]], dtype=np.int16)   # pla 最近一手是 pass
    op = np.array([[44, -1, 66]], dtype=np.int16)   # opp 第二手是 pass
    out = history_five(np.zeros((1, N, N), np.int8), my, op,
                       np.array([1], np.int8))
    want = [[(44 // N, 44 % N)],         # ch9  = op[0]
            [],                         # ch10 = my[0] = pass → 空
            [],                         # ch11 = op[1] = pass → 空
            [(22 // N, 22 % N)],        # ch12 = my[1] = 22，**没有被前移**
            [(66 // N, 66 % N)]]        # ch13 = op[2] = 66
    for ch in range(5):
        assert pts(out[0], ch) == want[ch], \
            f'ch{9 + ch} 应为 {want[ch]}（pass 占住自己那一格、但不点亮、也不前移），' \
            f'实得 {pts(out[0], ch)}'


def test_history_five_short_history_leaves_earlier_channels_empty():
    """只有 2 手历史 ⇒ 只占 2 个通道，**更早的不占位**（不是补 0、也不是复制）。

    「补 0」与「复制最近的」都是很容易写出来的实现（一个用 `np.where(valid, …, 0)`
    之后又回填、一个用 `last_valid` 前向填充），两者都会把 ch11-13 填成真值。
    """
    my = np.array([[11, -1, -1]], dtype=np.int16)
    op = np.array([[44, -1, -1]], dtype=np.int16)
    out = history_five(np.zeros((1, N, N), np.int8), my, op,
                       np.array([1], np.int8))
    non_empty = [ch for ch in range(5) if pts(out[0], ch)]
    assert non_empty == [0, 1], f'只有 2 手历史时应只占 ch9/ch10，实得 {non_empty}'
    assert int(out.sum()) == 2, '整组通道的非零点总数应恰好是 2'


def test_history_five_all_pass_gives_all_empty_channels():
    my = np.full((1, 3), -1, np.int16)
    op = np.full((1, 3), -1, np.int16)
    out = history_five(np.zeros((1, N, N), np.int8), my, op,
                       np.array([-1], np.int8))
    assert out.shape == (1, 5, N, N) and out.dtype == np.bool_
    assert int(out.sum()) == 0, '开局（无历史）时 5 个通道应全空'


def test_history_five_batch_is_elementwise_and_ignores_to_play_value():
    """逐样本独立（不串批），且 `to_play` 不参与取值。

    「不参与取值」是要钉的：直觉上历史通道会随执子方翻转，**但本仓的
    `my_hist`/`op_hist` 本来就是 to_play 相对的**（与 `feature_planes_batched`
    的通道 1-3/5-7 同口径），再乘一次 `to_play` 会把两者对调。
    """
    boards = np.zeros((2, N, N), np.int8)
    my = np.array([[11, 22, 33], [-1, -1, -1]], dtype=np.int16)
    op = np.array([[44, 55, 66], [77, -1, -1]], dtype=np.int16)
    a = history_five(boards, my, op, np.array([1, 1], np.int8))
    b = history_five(boards, my, op, np.array([-1, -1], np.int8))
    assert np.array_equal(a, b), 'to_play 换了值而输出变了 ⇒ 说明它参与了取值'
    assert pts(a[1], 0) == [(77 // N, 77 % N)], '第 2 个样本应独立'
    assert int(a[1].sum()) == 1, f'第 2 个样本只有 1 手历史，实得 {int(a[1].sum())}'


def test_history_five_rejects_too_short_histories():
    with pytest.raises(ValueError, match='需要每人最近 3 手'):
        history_five(np.zeros((1, N, N), np.int8),
                     np.zeros((1, 2), np.int16), np.zeros((1, 3), np.int16),
                     np.array([1], np.int8))


# --------------------------------------------------------------------------- #
# B2-3 · history_gated
# --------------------------------------------------------------------------- #
def _blank_board():
    return np.zeros((1, N, N), dtype=np.int8)


def test_history_gated_zero_history_copies_current_board():
    """**history=0 ⇒ ch15 == ch14、ch16 == ch15**（回退复制，**不是置 0**）。

    官方原文（spec §2.2 照录）::

        prevBoard     = (numTurnsOfHistoryIncluded < 1) ? board     : getRecentBoard(1);
        prevPrevBoard = (numTurnsOfHistoryIncluded < 2) ? prevBoard : getRecentBoard(2);

    「置 0」是最容易写出来的版本（`prev = boards if ok else None` → 后面 `None` 被
    当成空盘/零张量）。它产出全 0 的 ch15/ch16，**不抛异常**，与「这块盘上确实没有
    梯子」在张量上无法区分 —— 所以必须钉住「等于当前盘」。
    """
    cur = board([_blank(), '.X.', '...'] + [_blank()] * (N - 3))
    prev, prev_prev = history_gated(None, None, cur)
    assert np.array_equal(prev, cur), 'history=0 时 ch15 必须复制当前盘'
    assert np.array_equal(prev_prev, prev), 'history=0 时 ch16 必须复制 ch15'


def test_history_gated_one_history_second_fallback_is_prev_not_current():
    """**history=1 ⇒ ch16 == ch15**（复制的对象是 ch15，**不是当前盘**）。

    这是这条规则**唯一**的坑，也是纯值断言最容易漏的地方：
      · `ch16 == ch15` 在「两块盘面碰巧一样」的夹具上会假绿，所以夹具必须让
        **当前盘 / 前一手盘 / 前二手盘三块互不相同**；
      · 写错成 `(n < 2) ? board : ...` 时，`ch16 == ch15` 这条断言仍然成立
        （两边都等于 prev_board），只有额外钉住「`ch16 != 当前盘`」才抓得到。
     官方第一行回退到 `board`、**第二行回退到 `prevBoard`** —— 不是都回退到
      `board`。所以 ch16 在 history=1 时该是 prev 的副本。
    """
    cur = board(['.' * N] * N)
    cur[0, 0, 0] = 1
    prev = np.zeros((1, N, N), dtype=np.int8)
    prev[0, 1, 1] = 1
    prev_prev = np.zeros((1, N, N), dtype=np.int8)
    prev_prev[0, 2, 2] = 1

    g = history_gated(prev, None, cur)
    assert np.array_equal(g.prev, prev), 'history=1 时 ch15 应取前一手盘'
    assert np.array_equal(g.prev_prev, prev), \
        'history=1 时 ch16 应复制 ch15（官方回退到 prevBoard），不是当前盘'
    assert not np.array_equal(g.prev_prev, cur), '三块盘面必须互不相同，否则本测试假绿'


def test_history_gated_two_history_uses_both_boards():
    """**history≥2 ⇒ 正常取两块盘面**（门控不介入），且两块互不相同。

    这条钉的是「门控在历史充足时是**恒等**的」—— 加一个无条件回退的实现
    （比如永远 `prev or cur`）在 history≥2 上取值仍然正确，只有这条抓得到。
    """
    cur = np.zeros((1, N, N), np.int8)
    prev = np.zeros((1, N, N), np.int8)
    prev_prev = np.zeros((1, N, N), np.int8)
    cur[0, 0, 0] = 1
    prev[0, 1, 1] = -1
    prev_prev[0, 2, 2] = 1
    g = history_gated(prev, prev_prev, cur)
    assert np.array_equal(g.prev, prev)
    assert np.array_equal(g.prev_prev, prev_prev)


def test_history_gated_returns_named_fields_and_rejects_wrong_shapes_and_offset():
    """返回可按字段取（ladder agent 的调用点依赖它），且错形状 / 错 offset 报错。

    `offset` 不参与取值，只把「这三块盘面对应 ch14/15/16」钉在签名上
    （见 `history_gated` 的 docstring）。传错必须报错而不是照算 ——
    「算对了却写进了错的通道」是静默的。
    """
    cur = np.zeros((1, N, N), np.int8)
    g = history_gated(None, None, cur, offset=LADDER_CH_BASE)
    assert g.prev is cur and g.prev_prev is g.prev, 'history=0 时应直接复用同一块盘'
    assert (g.prev, g.prev_prev) == (g.prev, g.prev_prev)     # 可当元组解包

    with pytest.raises(ValueError, match='不是 ladder 块的基通道号'):
        history_gated(None, None, cur, offset=13)
    with pytest.raises(ValueError, match='prev_board 形状不符'):
        history_gated(np.zeros((2, N, N), np.int8), None, cur)


# --------------------------------------------------------------------------- #
# 与官方 stdata 的对拍（spec §9.4：stdata 是本仓唯一可执行 oracle）
# --------------------------------------------------------------------------- #
def _first_stdata_19x19(limit_rows=1200):
    """从 stdata 的 tar 里流式取**第一批 19×19 行**的 22 空间通道。

     **按行取、不要按成员取**：`getmembers()` 要把整个 tar 走一遍。
     19×19 由 ch0（on-board 掩码）的 1 的个数反推（约定见 spec §9.1 第 2 条）：
      同一个文件里混着 9/11/13/18 路与非方阵，直接喂 19×19 的重建盘面会静默错位。
    """
    bits_all = []
    with tarfile.open(STDATA_TAR, 'r|') as t:
        for member in t:
            if not member.name.endswith('.npz'):
                continue
            d = np.load(io.BytesIO(t.extractfile(member).read()))
            packed = d['binaryInputNCHWPacked']
            bits = np.unpackbits(packed, axis=-1, bitorder='big')[:, :, :361] \
                .reshape(-1, packed.shape[1], 19, 19).astype(bool)
            on_board = bits[:, 0].sum(axis=(1, 2))
            bits_all.append(bits[on_board == 361])
            if sum(b.shape[0] for b in bits_all) >= limit_rows:
                break
    return np.concatenate(bits_all, axis=0)[:limit_rows]


def _boards_from_stdata(spatial):
    """stdata 的 ch1/ch2 → 本仓约定的 `(B,19,19)` int8 盘面（pla=+1、opp=−1）。"""
    boards = np.zeros((spatial.shape[0], 19, 19), dtype=np.int8)
    boards[spatial[:, 1]] = 1
    boards[spatial[:, 2]] = -1
    return boards


@needs_stdata
def test_stdata_ch345_liberty_channels_match_official_bit_for_bit():
    """**ch3/4/5 与官方 stdata 逐位相等**（本模块最强的断言）。

    它一次性钉住四件事：
      1. ch3/4/5 是**不分色**的气桶（一组三格，不是六格）—— 若按 `to_play` 分色，
         与官方不会逐位相等；
      2. 桶是 `==1/==2/==3`（**恰好**）而不是 `>=3`；
      3. 气数是**整块去重**口径（与 `_group_liberty_count` 的 `set` 同义）——
         U 形块会把这条打出来；
      4. 4 邻域的连通性与盘面边界处理（结构元 `_STRUCT3` 逐样本独立）。
     与 ch18/19 不同，这三个通道**官方不做任何额外处理**，所以可以当 oracle。
      （ch18/19 官方先提死子 ⇒ 不可对拍，见 `area_ownership_map` 的 docstring。）
    """
    spatial = _first_stdata_19x19()
    boards = _boards_from_stdata(spatial)
    mine = liberties_123(boards)
    assert not (spatial[:, 1] & spatial[:, 2]).any(), '夹具前提：ch1/ch2 不该重叠'
    for k in range(3):
        assert np.array_equal(mine[:, k], spatial[:, 3 + k]), (
            f'ch{3 + k} 与官方不等：'
            f'我们 {int(mine[:, k].sum())} 点 / 官方 {int(spatial[:, 3 + k].sum())} 点')


@needs_stdata
def test_stdata_ch9_ch13_history_channel_order_is_opp_pla_opp_pla_opp():
    """**ch9–13 的颜色分布证实 spec §2.2 的通道顺序**（自 nextPlayer 视角）。

    stdata 是逐行独立样本、没有着法序列，所以**不能**直接对拍
    `history_five` 的输出；但可以检验它**蕴含的强性质**：
      · 每个通道**至多一个点**（一手一个落点；"每格恰好 1 点" 那条已实测）——
        若 pass 的处理是「跳过后把更早的手前移」，点数仍可能对，这条抓的是
        **顺序**；
      · 奇数号通道（ch9/11/13）的点**压倒性地落在 opp 子（官方 ch2）上**，
        偶数号通道（ch10/12）落在 pla 子（官方 ch1）上。落子若已被提掉则该点
        现在是空的 —— 所以断言用「≥95% 的非空通道点落在对应颜色的子上」，
        而不是「100% 落在子上」。
      · 实测（1254 行）：ch9 100% / ch10 99.4% / ch11 99.4% / ch12 97.9% /
        ch13 96.7%。若顺序写成 `pla, opp, pla, opp, pla`，这两个数会**对调**，
        测试立刻红。
    """
    spatial = _first_stdata_19x19()
    pla_stones = spatial[:, 1]
    opp_stones = spatial[:, 2]
    for ch in range(9, 14):
        plane = spatial[:, ch]
        counts = plane.sum(axis=(1, 2))
        assert counts.max() <= 1, f'ch{ch} 出现了多于一个落点（max={counts.max()}）'
        own_colour = opp_stones if (ch - 9) % 2 == 0 else pla_stones
        hit = (plane & own_colour).sum()
        ratio = hit / max(1, int(counts.sum()))
        assert ratio >= 0.95, (
            f'ch{ch} 只有 {ratio:.1%} 的落点在「该有的那一色」的子上（阈值 95%）；'
            f'顺序若对调（pla, opp, pla, opp, pla）这个数会掉到个位数')
