# -*- coding: utf-8 -*-
"""锁定 V7 全局特征里「读源码极易看错」的三处语义。

本文件的存在理由不是覆盖率，而是**防止再次把这三处读错**：

1. ``ch15`` / ``ch17`` 看着像「恒 1」，实际被 ``if`` 包住 ⇒ 是条件写入。
   （曾据此误改过一次，被官方 ``globalInputNC`` 的实测非零率当场否掉。）
2. ``ch0..4`` 的闸门是**递归嵌套**的前缀性质，不是逐格独立。
3. ``ch5`` 的分母是 **20**（不是 V3/V4 的 15）。
"""
import numpy as np
import pytest

from src.data import feature_v7 as fv


BS = 19


def _gf(my=None, op=None, hl=None, komi=7.5, rules_flags=0, encore=0, to_play=1):
    mh = np.full((1, 3), -1, np.int16) if my is None else my
    oh = np.full((1, 3), -1, np.int16) if op is None else op
    g = fv.global_features_v7(
        {'komi': np.array([komi], np.float32)}, mh, oh, None, None, hl,
        rules_flags=rules_flags, to_play=np.array([to_play], np.int8),
        board_area=np.array([BS * BS], np.float32), encore_phase=encore)
    return np.asarray(g, np.float64)[0]


# --------------------------------------------------------------------------- #
# 1. ch15 / ch16 / ch17 是「条件写入」，不是常量
# --------------------------------------------------------------------------- #

def test_ch15_ch16_are_zero_without_pda():
    """官方 ``nninputs.cpp:2673-2676`` 整块被 ``if(pda != 0)`` 包住。

     只看 ``rowGlobal[15] = 1.0`` 那一行会以为它恒 1 —— 实测官方
    ``globalInputNC`` 的 ch15 非零率只有 3.85%，与 ch16 完全同步
    （同一条件写入）。本仓无 PDA ⇒ 恒 0。
    """
    v = _gf()
    assert v[15] == 0.0
    assert v[16] == 0.0


def test_ch17_tracks_has_button_not_a_constant():
    """官方 ``:2679-2680`` 是 ``if(hist.hasButton) rowGlobal[17] = 1.0;``。

    ⇒ 它标的是「这局规则里开了 button」，**不是**「机制存在」。
    实测官方非零率 22.33%（不是 100%）⇒ 恒 1 是错的。
    """
    assert _gf(rules_flags=0)[17] == 0.0
    assert _gf(rules_flags=fv.FLAG_HAS_BUTTON)[17] == 1.0


def test_ch15_ch17_are_not_constants():
    """反向钉死：默认规则（无 PDA / 无 button）下这两格必须是 0。

    写这条是为了让「有人再把它们改成恒 1」时立刻红。
    """
    v = _gf(rules_flags=0)
    assert v[15] == 0.0 and v[17] == 0.0


# --------------------------------------------------------------------------- #
# 2. ch0..4 的前缀闸门
# --------------------------------------------------------------------------- #

def test_history_pass_flags_follow_the_official_slot_order():
    """ch0=op[0]、ch1=my[0]、ch2=op[1]、ch3=my[1]、ch4=op[2]（``_HISTORY_SLOTS``）。"""
    mh = np.array([[-1, -1, -1]], np.int16)
    oh = np.array([[-1, -1, -1]], np.int16)
    v = _gf(my=mh, op=oh, hl=5)
    assert list(v[:5]) == [1.0] * 5


def test_history_pass_flag_respects_history_length():
    """``history_length`` 截断：只有落在长度内的槽位才判 pass。

     造这个夹具时必须**把 ``my_hist`` 也填上落点** —— 只改 ``op_hist`` 而让
    ``my_hist`` 留全 −1，会连带让 ``my[0]`` / ``my[1]`` 变成 pass，
    于是 ch1/ch3 被点亮，测的就不是截断而是别的东西（我第一版就踩了）。
    """
    oh = np.array([[-1, -1, -1]], np.int16)     # 第1手=op[0] 是 pass
    mh = np.array([[7, 8, 9]], np.int16)         # 第2/4手=my[0]/my[1] 有落子
    assert list(_gf(my=mh, op=oh, hl=1)[:5]) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert list(_gf(my=mh, op=oh, hl=2)[:5]) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert list(_gf(my=mh, op=oh, hl=3)[:5]) == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_history_prefix_gate_is_prefix_not_per_slot():
    """ 官方是**递归嵌套**：第 k 手要求前 k-1 手都成立（``:2511-2556``）。

    构造「第 1 手存在、第 2 手缺失」：此时 ch1（=第 2 手）必须是 0，
    且**不能**因为后面某手有值就把中间的格子点亮。
    """
    # op[0] 有一手（⇒ hl 至少 1），但 my[0] 与 op[1] 都缺失。
    oh = np.array([[7, -1, -1]], np.int16)     # 第1手=op[0]=7(有落子)
    mh = np.array([[-1, -1, -1]], np.int16)    # 第2手=my[0] 缺失
    v = _gf(my=mh, op=oh, hl=5)
    # 第 1 手不是 pass ⇒ ch0 = 0；第 2 手「缺失」在 -1 语义下与 pass 同形，
    # 但它落在有效前缀内（hl=5）⇒ 按官方语义 ch1 = 1。
    assert v[0] == 0.0
    assert v[1] == 1.0


def test_non_pass_moves_do_not_light_the_pass_flag():
    """有落点的手 ⇒ 对应的 pass 标志为 0（``loc != PASS_LOC`` 分支，``:2516``）。

    两侧历史都给**满的落点**（``op`` 3 个 + ``my`` 2 个共 5 手），
    这样 5 格都不该亮。
    """
    oh = np.array([[7, 8, 9]], np.int16)         # 第1/3/5手
    mh = np.array([[10, 11, -1]], np.int16)     # 第2/4手
    v = _gf(my=mh, op=oh, hl=5)
    assert list(v[:5]) == [0.0] * 5


# --------------------------------------------------------------------------- #
# 3. ch5 的分母是 20，且符号随 to_play 翻转
# --------------------------------------------------------------------------- #

def test_ch5_uses_denominator_twenty_not_fifteen():
    """``rowGlobal[5] = selfKomi/20.0f``（``:2624``），**不是** V3/V4 的 15。"""
    assert _gf(komi=7.5)[5] == pytest.approx(7.5 / 20.0)
    assert _gf(komi=7.5)[5] == pytest.approx(0.375)
    assert _gf(komi=7.5)[5] != pytest.approx(7.5 / 15.0)


def test_ch5_sign_flips_with_to_play():
    """``currentSelfKomi(nextPlayer)`` ⇒ 黑先与白先互为相反数。"""
    assert _gf(komi=7.5, to_play=1)[5] == pytest.approx(0.375)
    assert _gf(komi=7.5, to_play=-1)[5] == pytest.approx(-0.375)


# --------------------------------------------------------------------------- #
# 4. ch18 三角波与门控
# --------------------------------------------------------------------------- #

def test_ch18_wave_on_19x19():
    """19 路 ⇒ boardArea 奇 ⇒ ``komiFloor = floor((sk-1)/2)*2 + 1``。

    komi 7.5 ⇒ floor(6.5/2)*2+1 = 7、delta 0.5 ⇒ wave 0.5。
    """
    assert _gf(komi=7.5)[18] == pytest.approx(0.5)
    assert _gf(komi=6.5)[18] == pytest.approx(1.0 - 1.5)


def test_ch18_is_zero_for_territory_without_encore():
    """官方门控 ``SCORING_AREA || encorePhase >= 2``（``:2711``）。"""
    terr = fv.FLAG_SCORING_TERRITORY
    assert _gf(rules_flags=terr, encore=0)[18] == 0.0
    assert _gf(rules_flags=terr, encore=2)[18] != 0.0