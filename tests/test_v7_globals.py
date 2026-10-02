# -*- coding: utf-8 -*-
"""B5 · V7 的 19 维全局特征：逐维语义 + 规则条件化 + 缺 ``RU`` 的默认值。

被测对象 `src/data/feature_v7.py`：
  · `global_features_v7`      —— 组装 ``(B,19)``
  · `self_komi`               —— ch5 的分子（符号随 to_play 翻转 + clip）
  · `komi_parity_wave`        —— ch18 的三角波（官方 ``nninputs.cpp`` 原文）
  · `pass_would_end_phase`    —— ch14
  · `rules_flags_from`` / ``rules_flags_from_sidecar`` —— 规则位布局

运行：
    python -m pytest tests/test_v7_globals.py -q

----
**为什么这份表以 spec §2.3 为准，而不是任何二手转述**
任务书里给过另一串 19 维的语义（「禁贴/禁入/气/提子数/自手贴/ko 计数/上次吃子数/
距上次吃子回合数/simple-ko 阻塞合法性/局内经过回合/对局结果」）。那份转述**不是**
V7 的全局输入。三条判据各自都足以排除它：

  1. KataGo ``nninputs.h`` 明写 ``NUM_FEATURES_GLOBAL_V7 = 19``；
  2. 官方 stdata 的 ``globalInputNC (N,19)`` 实测逐项吻合 spec §2.3（本文档
     每个维度的取值域都是从它反解的）；
  3. ch18 的三角波从官方源码取到原文，并在**真实官方数据**上验算命中
     （见 ``test_ch18_parity_wave_matches_seven_official_values``）。

而「对局结果」根本不在输入侧（它在 ``globalTargetsNC``），「提子数 / 距上次吃子
回合数」也不在这 19 格里 —— 这两条最容易被误当成输入特征。
"""

import numpy as np
import pytest

from src.data.feature_v7 import (
    DEFAULT_RULES_FLAGS,
    FLAG_HAS_BUTTON,
    FLAG_KO_MASK,
    FLAG_MULTISTONE_SUICIDE,
    FLAG_SCORING_TERRITORY,
    FLAG_TAX_MASK,
    GLOBAL_CHANNELS,
    GLOBAL_DTYPE,
    KO_POSITIONAL,
    KO_SIMPLE,
    KO_SITUATIONAL,
    KOMI_SCALE,
    SCORING_AREA,
    SCORING_TERRITORY,
    TAX_ALL,
    TAX_NONE,
    TAX_SEKI,
    GameRow,
    global_features_v7,
    komi_parity_wave,
    pass_would_end_phase,
    rules_flags_from,
    rules_flags_from_sidecar,
    self_komi,
)

N = 19
AREA = N * N
#: 实测官方 ``globalInputNC`` 上 ch5 的分母是 **20**（V3/V4 是 15）。
assert KOMI_SCALE == 20.0
#: fp16 只有 ~11 位有效位 ⇒ 归一化后的 ch5 只能比到 ~1e-3。
FP16_TOL = 2e-3


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def hists(rows):
    """把 ``[[op0, my0, op1], ...]`` 这样的**物理手序**写成交错的
    ``(my_hist, op_hist)``（`my` 收偶数位、`op` 收奇数位）。

    物理手序第 1 手永远是对手落的 ⇒ op[0]；第 2 手是自己的 ⇒ my[0]；……
    与 ``feature_v7._HISTORY_SLOTS`` 同一条顺序。
    """
    out_my, out_op = [], []
    for r in rows:
        r = list(r)
        r += [-1] * (6 - len(r))
        my, op = [-1, -1, -1], [-1, -1, -1]
        for k in range(5):
            who = 'op' if k % 2 == 0 else 'my'
            slot = k // 2
            (op if who == 'op' else my)[slot] = r[k]
        out_my.append(my)
        out_op.append(op)
    return np.array(out_my, dtype=np.int16), np.array(out_op, dtype=np.int16)


def gfeat(rows, *, komi=7.5, to_play=None, rules_flags=None, **kw):
    """跑一遍 ``global_features_v7`` 的便利封装。``rows`` 是物理手序的
    「最近一手在前」的 6 个坐标（``-1`` = pass/无）。"""
    my, op = hists(rows)
    B = len(my)
    tp = np.full(B, 1, dtype=np.int8) if to_play is None else np.asarray(to_play)
    gr = GameRow(komi=komi, rules_flags=DEFAULT_RULES_FLAGS)
    return global_features_v7(gr, my, op, rules_flags=rules_flags,
                              to_play=tp, **kw)


# --------------------------------------------------------------------------- #
# 形状 / dtype / 逐维取值域
# --------------------------------------------------------------------------- #
def test_shape_dtype_and_which_channels_are_always_zero():
    """``(B,19)`` float16；**ch15 / ch16 恒 0**（无 PDA，spec §2.2 / §2.4）。

    保留这两格不裁剪的理由：严格复刻官方张量布局，接官方 checkpoint 时零改形状。
    官方 stdata 上实测 ch15/ch16 **非零率 0%**（252 行全 0）。
    """
    out = gfeat([[0, 1, 2, 3, 4, 5]], komi=6.5)
    assert out.shape == (1, GLOBAL_CHANNELS)
    assert out.dtype == GLOBAL_DTYPE
    assert out[0, 15] == 0.0 and out[0, 16] == 0.0


def test_all_channels_are_finite_and_in_declared_domain():
    """每一格的取值域都要站得住（域错了说明写成了另一种特征）。"""
    rng = np.random.default_rng(7)
    for _ in range(20):
        rows = rng.integers(-1, AREA, size=(4, 6))
        out = gfeat(rows, komi=float(rng.choice([0.0, 6.5, 7.5, 22.5, -7.5])),
                    to_play=rng.choice([-1, 1], size=4),
                    rules_flags=int(rng.integers(0, 128)))
        assert np.isfinite(out).all(), '全局特征里不能出现 nan/inf'
        assert set(np.unique(out[:, 0:5])).issubset({0.0, 1.0})
        assert set(np.unique(out[:, 6])).issubset({0.0, 1.0})
        assert set(np.unique(out[:, 7])).issubset({-0.5, 0.0, 0.5})
        assert set(np.unique(out[:, 8])).issubset({0.0, 1.0})
        assert set(np.unique(out[:, 9])).issubset({0.0, 1.0})
        assert set(np.unique(out[:, 10:12])).issubset({0.0, 1.0})
        assert set(np.unique(out[:, 12:15])).issubset({0.0, 1.0})
        assert set(np.unique(out[:, 17])).issubset({0.0, 1.0})
        assert out[:, 5].max() <= (AREA + 1.0) / KOMI_SCALE
        assert out[:, 18].min() >= -0.5001 and out[:, 18].max() <= 0.5001


# --------------------------------------------------------------------------- #
# ch0-4 · 过去 5 手各自是不是 pass
# --------------------------------------------------------------------------- #
def test_ch0_to_ch4_are_pass_flags_in_physical_turn_order():
    """ch0 = 第 1 手（最近）是不是 pass，……，ch4 = 第 5 手。

    ⚠ 官方是 `PASS_LOC` / `NULL_LOC` **两个不同**的 loc ⇒ 「历史不足」在官方那里
    是 0 而不是 1。本测试用一条「5 手全非 pass」的历史（全部有坐标），
    这样 pass 标志与顺序都不受历史不足干扰。
    """
    out = gfeat([[10, 11, 12, 13, 14, 15],
                 [-1, -1, -1, -1, -1, -1],
                 [-1, 20, -1, 30, -1, 40],
                 [1, -1, 2, -1, 3, -1]], history_length=5)
    assert out[0, 0:5].tolist() == [0, 0, 0, 0, 0]
    assert out[1, 0:5].tolist() == [1, 1, 1, 1, 1]
    # 第 1/3/5 手（对手的）全是 pass
    assert out[2, 0:5].tolist() == [1, 0, 1, 0, 1]
    # 第 2/4 手（自己的）全是 pass
    assert out[3, 0:5].tolist() == [0, 1, 0, 1, 0]


def test_history_beyond_history_length_is_not_reported_as_pass():
    """``history_length`` 之外的历史**不许**报成 pass。

    本仓的 ``-1`` 同时表示「pass」与「无此手」，而官方对后者给 0 —— 所以
    ply 0（六个槽全 −1）必须整行 ch0..4 全 0，否则整段开局的全局特征都是
    「上一手是 pass」的假信号。
    """
    # ply 0：没有任何历史
    ply0 = gfeat([[-1, -1, -1, -1, -1, -1]])
    assert ply0[0, 0:5].tolist() == [0, 0, 0, 0, 0]
    assert ply0[0, 14] == 0.0, 'ply 0 的 passWouldEndPhase 也必须是 0'

    # 显式 history_length=2 ⇒ 只有 ch0 / ch1 算数，其余三个槽即使是 −1 也不报
    partial = gfeat([[-1, -1, -1, -1, -1, -1]], history_length=2)
    assert partial[0, 0:5].tolist() == [1, 1, 0, 0, 0]

    # history_length=0 ⇒ 全 0（哪怕全是 pass）
    zero = gfeat([[-1, -1, -1, -1, -1, -1]], history_length=0)
    assert zero[0, 0:5].tolist() == [0, 0, 0, 0, 0]


# --------------------------------------------------------------------------- #
# ch5 · 自手贴目
# --------------------------------------------------------------------------- #
def test_ch5_is_self_komi_over_20_and_flips_sign_with_to_play():
    """ch5 = ``currentSelfKomi(nextPlayer) / 20``，**符号随 to_play 翻转**。

    实测官方：komi 7.5 的黑走行是 **0.375**（= 7.5/20；若是 /15 会是 0.5）。
    """
    for komi in (0.0, 6.5, 7.5, 5.5, 3.8, -7.5, 2.8):
        black = gfeat([[1, 2, 3, 4, 5, 6]], komi=komi, to_play=1)
        white = gfeat([[1, 2, 3, 4, 5, 6]], komi=komi, to_play=-1)
        expect = komi / KOMI_SCALE
        assert black[0, 5] == pytest.approx(expect, abs=FP16_TOL)
        assert white[0, 5] == pytest.approx(-expect, abs=FP16_TOL)


def test_ch5_is_clipped_to_plus_minus_area_plus_one():
    """clip 的是 **selfKomi 本身**，官方原文 ``if(selfKomi > bArea+1.0f)``。"""
    lim = AREA + 1.0
    out = gfeat([[1, 2, 3, 4, 5, 6]], komi=100000.0, to_play=1)
    # ⚠ fp16 的**相对**精度 ~2^-11 ≈ 4.9e-4 ⇒ 18.1 上绝对误差可达 ~0.009，
    #   所以这里必须用相对容差，写绝对容差会逼出「把 clip 值改小」的假修复。
    assert out[0, 5] == pytest.approx(lim / KOMI_SCALE, rel=2e-3)
    out = gfeat([[1, 2, 3, 4, 5, 6]], komi=-100000.0, to_play=1)
    assert out[0, 5] == pytest.approx(-lim / KOMI_SCALE, rel=2e-3)
    # 未 clip 时 self_komi 就是 komi * to_play（这一步是 float32，不经 fp16）
    assert self_komi(np.array([7.5, -7.5]), np.array([1, -1]), AREA) \
        == pytest.approx(np.array([7.5, 7.5]))


def test_missing_komi_becomes_zero_instead_of_poisoning_ch5_and_ch18():
    """``komi`` 为 ``nan``（SGF 缺 ``KM``，实测 2.2%）⇒ 取 0.0。

    不能让 ``nan`` 顺着 clip 污染 ch5 **和** ch18 —— 后者是三角波的输入，
    一个 nan 会毁掉整条曲线，而症状只是「训练不收敛」，看不出是数据源缺字段。
    """
    out = gfeat([[1, 2, 3, 4, 5, 6]], komi=float('nan'))
    assert out[0, 5] == 0.0
    assert np.isfinite(out[0, 18])


# --------------------------------------------------------------------------- #
# ch6 / ch7 · ko 规则（规则条件化）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('ko_rule,expect', [
    (KO_SIMPLE, (0.0, 0.0)),
    (KO_POSITIONAL, (1.0, 0.5)),
    (KO_SITUATIONAL, (1.0, -0.5)),
])
def test_ch6_ch7_encode_the_three_ko_rules(ko_rule, expect):
    """simple = ``(0, 0)`` / positional = ``(1, +0.5)`` / situational = ``(1, −0.5)``。

    实测官方：``ch7 != 0 ⟺ ch6 != 0``（252 行上成立）—— 这条不变量在本测试里
    由三组取值逐个覆盖。
    """
    out = gfeat([[1, 2, 3, 4, 5, 6]], rules_flags=rules_flags_from(ko=ko_rule))
    assert (out[0, 6], out[0, 7]) == expect


def test_each_ko_rule_flag_moves_only_its_own_channels():
    """改 ko 规则**只**动 ch6 / ch7 —— 别的维必须逐位不变。

    这条抓的是「位布局写串了」：比如把 ko 误放进 ``FLAG_TAX_MASK`` 覆盖的位，
    只断言 ch6/ch7 的值是抓不到的（值仍然对，只是搭在了别的位的通道上）。
    """
    base = gfeat([[1, 2, 3, 4, 5, 6]])
    for ko_rule in (KO_POSITIONAL, KO_SITUATIONAL):
        cur = gfeat([[1, 2, 3, 4, 5, 6]],
                    rules_flags=rules_flags_from(ko=ko_rule))
        diff = np.flatnonzero(base != cur)
        assert diff.tolist() == [6, 7], (
            f'ko={ko_rule} 改了不该改的通道：{diff.tolist()}')
    # 位不重叠：ko 的两位不在 tax / scoring / suicide / button 的位里
    assert FLAG_KO_MASK & FLAG_TAX_MASK == 0
    assert FLAG_KO_MASK & FLAG_SCORING_TERRITORY == 0
    assert FLAG_KO_MASK & FLAG_MULTISTONE_SUICIDE == 0
    assert FLAG_KO_MASK & FLAG_HAS_BUTTON == 0


# --------------------------------------------------------------------------- #
# ch8 / ch9 / ch10 / ch11 / ch17 · 规则条件化
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('flag,ch,expect', [
    ('multi_stone_suicide', 8, (0.0, 1.0)),
    ('has_button', 17, (0.0, 1.0)),
])
def test_boolean_rule_flags_move_exactly_one_channel(flag, ch, expect):
    """``multiStoneSuicideLegal`` ⇒ ch8；``hasButton`` ⇒ ch17。**各只动一格。**"""
    base = gfeat([[1, 2, 3, 4, 5, 6]])
    cur = gfeat([[1, 2, 3, 4, 5, 6]], rules_flags=rules_flags_from(**{flag: True}))
    assert base[0, ch] == expect[0] and cur[0, ch] == expect[1]
    assert np.flatnonzero(base != cur).tolist() == [ch]


def test_ch9_is_territory_scoring_flag():
    """ch9：territory 计分 ⇒ 1，area ⇒ 0（spec §2.3）。

    实测官方：非零率 0.4%（252 行里 1 行），且那些行正是 ch5 异常大贴目那批 ——
    与 spec §5.3.2 的 `Japanese` 占 7.3% 是不同批数据的分布。
    """
    area = gfeat([[1, 2, 3, 4, 5, 6]],
                 rules_flags=rules_flags_from(scoring=SCORING_AREA))
    terr = gfeat([[1, 2, 3, 4, 5, 6]],
                 rules_flags=rules_flags_from(scoring=SCORING_TERRITORY))
    assert area[0, 9] == 0.0 and terr[0, 9] == 1.0
    assert np.flatnonzero(area != terr).tolist() == [9, 18], (
        'territory 计分还会把 ch18 的三角波关掉（官方 if 的另一半）')


@pytest.mark.parametrize('tax,expect', [
    (TAX_NONE, (0.0, 0.0)),
    (TAX_SEKI, (1.0, 0.0)),
    (TAX_ALL, (1.0, 1.0)),
])
def test_ch10_ch11_encode_the_three_tax_rules(tax, expect):
    """tax：none = ``(0,0)`` / seki = ``(1,0)`` / all = ``(1,1)``。

    实测官方 ``globalInputNC[:,10:12]`` 只出现这三种组合。
    """
    out = gfeat([[1, 2, 3, 4, 5, 6]], rules_flags=rules_flags_from(tax=tax))
    assert (out[0, 10], out[0, 11]) == expect


# --------------------------------------------------------------------------- #
# ch12 / ch13 · encorePhase（简单局恒 0）
# --------------------------------------------------------------------------- #
def test_ch12_ch13_are_encore_phase_flags_and_zero_for_simple_games():
    """``encorePhase > 0`` / ``> 1``。本仓无 encore（spec §2.6 D2）⇒ 默认全 0。

    实测官方：非零率各 0.4%（252 行里 1 行）。
    """
    base = gfeat([[1, 2, 3, 4, 5, 6]])
    assert base[0, 12] == 0.0 and base[0, 13] == 0.0
    p1 = gfeat([[1, 2, 3, 4, 5, 6]], encore_phase=1)
    p2 = gfeat([[1, 2, 3, 4, 5, 6]], encore_phase=2)
    assert (p1[0, 12], p1[0, 13]) == (1.0, 0.0)
    assert (p2[0, 12], p2[0, 13]) == (1.0, 1.0)
    assert np.flatnonzero(p2[0] != base[0]).tolist() == [12, 13]


# --------------------------------------------------------------------------- #
# ch14 · passWouldEndPhase
# --------------------------------------------------------------------------- #
def test_ch14_is_true_exactly_when_the_previous_move_was_a_pass():
    """区域计分只有一个阶段、连续两手 pass 结束 ⇒ 「这一手 pass 会结束阶段」
    ⟺「**上一手已经是 pass**」。上一手永远是对手落的 ⇒ 看 ``op_hist[:,0]``。"""
    assert gfeat([[7, 1, 2, 3, 4, 5]])[0, 14] == 0.0, '上一手不是 pass'
    assert gfeat([[-1, 1, 2, 3, 4, 5]])[0, 14] == 1.0, '上一手是 pass'


def test_pass_would_end_phase_helper_respects_history_length():
    """直接测 helper：历史不足（``history_length == 0``）必须给 False。"""
    my, op = hists([[-1, 1, 2, 3, 4, 5]])
    assert pass_would_end_phase(my, op, [5]).tolist() == [True]
    assert pass_would_end_phase(my, op, [0]).tolist() == [False]
    assert pass_would_end_phase(my, op, [1]).tolist() == [True]
    my, op = hists([[1, 2, 3, 4, 5, 6]])
    assert pass_would_end_phase(my, op, [5]).tolist() == [False]


# --------------------------------------------------------------------------- #
# ch18 · 贴目 × 棋盘奇偶三角波（**官方 oracle**）
# --------------------------------------------------------------------------- #
def test_ch18_parity_wave_has_all_three_segments_of_the_triangle():
    """ch18 的三角波三段全测到，且**数值是手算的精确值**。

    官方原文（``nninputs.cpp``）::

        komiFloor = (area 偶) ? floor(k/2)*2 : floor((k-1)/2)*2 + 1;
        delta = clip(k - komiFloor, 0, 2);
        wave  = delta < 0.5 ? delta : (delta < 1.5 ? 1-delta : delta-2);

    19 路（361）⇒ 奇 ⇒ ``komiFloor = floor((k-1)/2)*2 + 1``：

    =================  ==========  =========  =========  ==============
    selfKomi          komiFloor  delta      分支        wave
    =================  ==========  =========  =========  ==============
    7.5               7          0.5        下行首端点   0.5
    7.25              7          0.25       **上行段**   0.25
    6.0               5          1.0        **下行段**   0.0
    5.424             5          0.424      上行段       0.424
    6.68              5          1.68       **过波谷**   −0.32
    9.618             9          0.618      下行段       0.382
    =================  ==========  =========  =========  ==============

    最后两行的**负值**是这条曲线的关键：三角波不是「对称的钟形」，波谷在 0
    附近被穿透，所以 wave ∈ [−0.5, 0.5]。把公式写成 ``|delta − 0.5|`` 之类的
    常见替代会在过波谷那两行上符号相反。

    🔴 **与真实官方数据的成批对拍不在这里**，在
    ``tests/test_v7_assemble.py::test_global_ch18_matches_official_batch`` ——
    那里直接读官方 ``globalInputNC`` 的**原始 float32** ch5/ch18（不经打印
    截断），所以能逐位命中；这里刻意用手算值，避免「把打印出来的 4 位小数
    再当精确输入」这种自欺。
    """
    cases = [
        # (selfKomi, 期望 wave, 这行在钉什么)
        (7.5, 0.5, '上行段的终点（delta 恰好 0.5）'),
        (7.25, 0.25, '上行段内部'),
        (6.0, 0.0, '下行段中点（唯一取 0 的点）'),
        (5.424, 0.424, '上行段内部，非整数贴目'),
        (6.68, -0.32, '过波谷 ⇒ 负值'),
        (9.618, 0.382, '下行段内部'),
    ]
    for sk, want, why in cases:
        got = komi_parity_wave(np.array([sk], np.float32), AREA)[0]
        assert got == pytest.approx(want, abs=1e-6), (
            f'selfKomi={sk}（{why}）：我们 {got}、手算 {want}')

    # 波谷两侧符号相反（同一个「和棋点」附近）
    assert komi_parity_wave(np.array([6.6], np.float32), AREA)[0] < 0
    assert komi_parity_wave(np.array([5.4], np.float32), AREA)[0] > 0


def test_parity_wave_is_bounded_by_half_and_period_is_two_komi_points():
    """值域 ``[−0.5, 0.5]``、周期 **2 贴目**（官方的原始注释：period of 2 komi
    points）—— 这两条是「三角波」这个描述本身的可检验形式。"""
    k = np.linspace(-30.0, 30.0, 4001, dtype=np.float32)
    w = komi_parity_wave(k, AREA)
    assert w.min() >= -0.5 - 1e-6 and w.max() <= 0.5 + 1e-6
    assert w.min() == pytest.approx(-0.5, abs=1e-3)
    assert w.max() == pytest.approx(0.5, abs=1e-3)
    # 周期 2：selfKomi 与 selfKomi+2 给同样的值
    a = komi_parity_wave(np.array([6.3, 8.3, -4.1, 12.7], np.float32), AREA)
    b = komi_parity_wave(np.array([8.3, 10.3, -2.1, 14.7], np.float32), AREA)
    assert np.allclose(a, b, atol=1e-5)


def test_parity_wave_is_zero_for_territory_scoring_outside_encore():
    """官方 ``if(scoringRule == SCORING_AREA || encorePhase >= 2)`` 的门控。

    ⇒ territory 计分且 ``encorePhase < 2`` ⇒ 恒 0（与 spec §2.5 的
    「territory 对局 ch18/19 恒 0」同族）。
    """
    sk = np.array([7.5, 6.68, 5.424], dtype=np.float32)
    assert komi_parity_wave(sk, 361, SCORING_TERRITORY, 0).tolist() == [0, 0, 0]
    # encorePhase >= 2 时门控放行 ⇒ 又开始算
    assert komi_parity_wave(sk, 361, SCORING_TERRITORY, 2)[0] != 0.0


def test_even_board_area_takes_the_other_branch_of_the_wave():
    """奇偶盘走不同的 ``komiFloor`` 公式 —— 这条抓「奇偶没被读」。

    19 路（361）与 11 路（121）都走 ``floor((k-1)/2)*2+1``，所以要测另一支
    得用**偶数**盘面积（例如 20×20 = 400）。取 selfKomi = 7.4：

      * 奇数面积：``floor(6.4/2)*2+1 = 7`` ⇒ delta 0.4 ⇒ wave **+0.4**
      * 偶数面积：``floor(7.4/2)*2   = 6`` ⇒ delta 1.4 ⇒ wave **−0.4**

    ⇒ 面积奇偶不只改数值，**连符号都改**。只写一支的话这条会静默通过。
    """
    odd = komi_parity_wave(np.array([7.4], np.float32), 361)[0]
    even = komi_parity_wave(np.array([7.4], np.float32), 400)[0]
    assert odd == pytest.approx(0.4, abs=1e-6)
    assert even == pytest.approx(-0.4, abs=1e-6)
    assert odd != even
    # 121（11 路）也是奇数 ⇒ 与 361 同支
    assert komi_parity_wave(np.array([7.4], np.float32), 121)[0] == pytest.approx(0.4)


def test_ch18_uses_self_komi_so_its_sign_follows_to_play():
    """ch18 的输入是 **selfKomi**（不是原始 komi）⇒ 符号也随 to_play 翻转。"""
    black = gfeat([[1, 2, 3, 4, 5, 6]], komi=7.5, to_play=1)
    white = gfeat([[1, 2, 3, 4, 5, 6]], komi=7.5, to_play=-1)
    assert black[0, 18] == pytest.approx(0.5, abs=1e-3)
    assert white[0, 18] == pytest.approx(-0.5, abs=1e-3)


# --------------------------------------------------------------------------- #
# 缺 RU 时的默认值
# --------------------------------------------------------------------------- #
def test_default_flags_give_the_documented_simple_game_defaults():
    """缺 ``RU``（实测 **55% 的局**）⇒ 这四位取默认值。

    全局 ch6/7 = ``(0,0)``、ch8 = 0、ch10/11 = ``(0,0)``、ch17 = 0，
    ch9 = 0（area 计分）。**扩展位留着**：见 ``rules_flags_from_sidecar``。
    """
    assert DEFAULT_RULES_FLAGS == 0
    out = gfeat([[1, 2, 3, 4, 5, 6]])
    assert out[0, 6:9].tolist() == [0.0, 0.0, 0.0]
    assert out[0, 9] == 0.0
    assert out[0, 10:12].tolist() == [0.0, 0.0]
    assert out[0, 17] == 0.0


def test_default_flags_are_what_a_sidecar_row_without_ru_produces():
    """sidecar 里 ``g_rules`` 只有 bit0（=0 ⇒ AREA）的那批局，转出来必须是 0。

    这是「无 RU ⇒ 默认」这条规则的**端到端**检查：不是「默认值碰巧是 0」，
    而是「sidecar 的 0 号局确实经过转换函数落到 0」。
    """
    for g_rules in (0b00000, 0b10000):     # 只带 button / 只带 suicide 位
        fl = rules_flags_from_sidecar(g_rules)
        if g_rules & 0b10000:
            assert fl & FLAG_HAS_BUTTON
        else:
            assert fl == DEFAULT_RULES_FLAGS


def test_sidecar_mapping_is_explicitly_lossy_for_tax_and_ko():
    """spec §5.3 的 1 位 tax / 1 位 ko 表达不了全部组合 ⇒ 本函数写明丢了什么。

    ⚠ 这是**有损兼容垫片**，不是最终映射：sidecar 想支持全部 8 种组合，得先把
    它的打包改成 ``rules_flags_from`` 的布局。
    """
    # tax 的 1 位 = 1 ⇒ 只能按 TAX_SEKI 解读（TAX_ALL 在 1 位下不可区分）
    assert rules_flags_from_sidecar(0b00010) == rules_flags_from(tax=TAX_SEKI)
    # ko 的 1 位 = 1 ⇒ 只能按 KO_POSITIONAL 解读（situational 不可区分）
    assert rules_flags_from_sidecar(0b00100) == rules_flags_from(ko=KO_POSITIONAL)
    # 五个位都要能读出来
    assert rules_flags_from_sidecar(0b11111) == rules_flags_from(
        SCORING_TERRITORY, TAX_SEKI, KO_POSITIONAL, True, True)


def test_sidecar_tax_seki_makes_calculate_area_raise_instead_of_returning_zero(
        ds_row):
    """🔴 有损映射的**后果必须显式**：tax≠NONE 时 ``calculate_area`` 抛错。

    静默返回全 0 与「TERRITORY 恒 0」在张量上逐位相同，会被当成「无归属」
    污染训练（``calculate_area`` 的 docstring 记着这个理由）。
    """
    from src.data.feature_v7 import calculate_area, spatial_channels_v7
    boards = np.zeros((1, N, N), dtype=np.int8)
    boards[0, 0, 0] = 1
    fl = rules_flags_from_sidecar(0b00010)          # tax 位 = 1
    with pytest.raises(NotImplementedError, match='TAX_SEKI'):
        calculate_area(boards, np.array([1], np.int8), rules_flags=fl)
    # AREA + TAX_NONE 正常走通
    ok = spatial_channels_v7(boards, np.array([1], np.int8), np.array([-1]),
                             np.full((1, 3), -1), np.full((1, 3), -1),
                             rules_flags=DEFAULT_RULES_FLAGS)
    assert ok.shape == (1, 22, N, N)


def test_rules_flags_from_is_the_inverse_of_the_shared_bit_layout(ds_row):
    """``rules_flags_from`` 的每一个字段都能在 ``calculate_area`` 的布局里读回去。"""
    for scoring in (0, SCORING_TERRITORY):
        for tax in (TAX_NONE, TAX_SEKI, TAX_ALL):
            fl = rules_flags_from(scoring=scoring, tax=tax)
            assert (fl & FLAG_SCORING_TERRITORY != 0) == bool(scoring)
            assert (fl & FLAG_TAX_MASK) >> 1 == tax


# --------------------------------------------------------------------------- #
# 守卫
# --------------------------------------------------------------------------- #
def test_to_play_is_required_because_ch5_and_ch18_depend_on_it(ds_row):
    """不给 ``to_play`` 就报错，而不是「猜一半」。

    症状对照：猜错 ch5/ch18 的符号**不会抛异常**，只是让白走的一行拿到了
    「黑方的自手贴目」—— 而这两维是 komi 条件化与 scorebelief parity 的输入。
    """
    my, op = hists([[1, 2, 3, 4, 5, 6]])
    with pytest.raises(ValueError, match='to_play'):
        global_features_v7(GameRow(komi=7.5), my, op)


def test_rejects_short_history_and_mismatched_batch(ds_row):
    my = np.full((2, 2), -1, dtype=np.int16)
    op = np.full((2, 3), -1, dtype=np.int16)
    with pytest.raises(ValueError, match='3 手'):
        global_features_v7(GameRow(komi=7.5), my, op,
                           to_play=np.ones(2, np.int8))
    with pytest.raises(ValueError, match='批大小'):
        global_features_v7(GameRow(komi=7.5), np.full((3, 3), -1), op,
                           to_play=np.ones(3, np.int8))


def test_rejects_bad_to_play_and_missing_komi_key(ds_row):
    my, op = hists([[1, 2, 3, 4, 5, 6]])
    with pytest.raises(ValueError, match=r'±1'):
        global_features_v7(GameRow(komi=7.5), my, op, to_play=np.zeros(1))
    with pytest.raises(TypeError, match='komi'):
        global_features_v7(object(), my, op, to_play=np.ones(1, np.int8))


def test_prev_boards_are_shape_checked_but_semantically_unused(ds_row):
    """``prev_board`` / ``prev_prev_board`` 按任务书签名保留，但 spec §2.3 的
    19 维**没有一维**读它们 ⇒ 形状对不对有影响、值没有任何影响。

    钉住「不读」这件事是为了让将来有人补「提子数」一类特征时知道这两个形参
    现在是空的，而不是以为它已经接上了。
    """
    my, op = hists([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    tp = np.ones(2, np.int8)
    gr = GameRow(komi=7.5)
    a = global_features_v7(gr, my, op, to_play=tp)
    b = global_features_v7(gr, my, op,
                           prev_board=np.ones((2, N, N), np.int8),
                           prev_prev_board=np.zeros((2, N, N), np.int8),
                           to_play=tp)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match='prev_board'):
        global_features_v7(gr, my, op, prev_board=np.ones((2, 5, 5), np.int8),
                           to_play=tp)


def test_batch_is_row_independent(ds_row):
    """同一行放在批里的不同位置，结果必须逐位相同（装配层没有跨行串扰）。"""
    rows = [[1, 2, 3, 4, 5, 6], [-1, -1, -1, -1, -1, -1], [7, -1, 8, -1, 9, -1]]
    komis = [7.5, 0.0, -6.5]
    tps = [1, -1, 1]
    # ⚠ 两边都要传 history_length：否则「全 −1」那行在 single 里被推成 ply 0
    #   而在 batched 里被当成 5 手齐全，ch0..4 就会不一致（推不出来的差异）。
    single = np.vstack([gfeat([r], komi=k, to_play=[tp], history_length=5)
                        for r, k, tp in zip(rows, komis, tps)])
    my, op = hists(rows)
    gr = GameRow(komi=np.array(komis, dtype=np.float32))
    batched = global_features_v7(gr, my, op, history_length=5,
                                 to_play=np.array(tps, np.int8))
    assert np.array_equal(single, batched)


# --------------------------------------------------------------------------- #
# 夹具占位（保持上面几处 pytest 夹具参数可用）
# --------------------------------------------------------------------------- #
@pytest.fixture()
def ds_row():
    """这些测试全是纯函数级的，不需要磁盘数据集；这个夹具只为让签名自解释。"""
    return None