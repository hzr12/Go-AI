# -*- coding: utf-8 -*-
r"""`build_dataset.parse_player_rating` 的回归：棋手名里的等级 vs. 垃圾。

根因（2026-10-04 实测 20,000 局 / 40,000 个棋手名）
----------------------------------------------------
旧实现的两条正则都**只找「数字 + d/k」**，既不要求词边界，也不校验那个数字
是不是真实存在的级/段：

    re.search(r'(\d+)([dkDK])', s)

于是它把 Leela Zero 名字里的**十六进制 commit hash** 当成了等级：

    'Leela Zero 0.17 74387dbc'  -> 74387d + bc      -> 743890
    'Leela Zero 0.17 c23d983a'  -> 23d              -> 250
    'Leela Zero 0.17 88f40f0d'  -> 0d（尾巴正好是 d）-> 20

实测后缀分支 3054 次命中里 **2913 次（95.4%）是这类 hash**；连看起来正常的
20/30/50/80/110 也全是 hash 碰巧拼出来的假段位。这些 rating 一路喂进
`compute_game_weight` 的 `exp(avg/20)`，就是 `game_weights` 里 6.06% 的 inf
与 1.84% 的巨值的来源。

两条约束缺一不可
----------------
只加词边界挡不住 ``1234567d``（整串数字、左边是空格）；只加范围挡不住
``23d``（rank=23 落在合法区间）。必须**同时**要求：

  1. 数字与字母的**两侧都不是字母数字**（把 hash 拦在词中间）；
  2. 数字是**真实存在的级/段**（30k..1k 与 1d..9d 之上再留余量）。
"""
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_dataset import (  # noqa: E402
    compute_game_weight, parse_player_rating)

# --------------------------------------------------------------------------- #
# 真实语料里抓到的字符串（不是编的）
# --------------------------------------------------------------------------- #
#: 十六进制 hash 拼出假段位的实测样本。旧实现会给出括号里的 rating。
HEX_HASH_NAMES = [
    ('Leela Zero 0.17 74387dbc', 743890),   # 74387d + bc
    ('Leela Zero 0.17 b0a6605d', 66070),    # 6605d
    ('Leela Zero 0.17 a7585deb', 75870),    # 7585d
    ('Leela Zero 0.17 344d1584', 3460),     # 344d
    ('Leela Zero 0.17 3ac165d1', 1670),     # 165d
    ('Leela Zero 0.17 d041d677', 430),      # 041d
    ('Leela Zero 0.17 b28dd6bb', 300),      # 28d   <- rank 落在合法区间，只有词边界拦得住
    ('Leela Zero 0.17 c23d983a', 250),      # 23d   <- 同上
    ('Leela Zero 0.17 1d571b7b', 30),       # 1d
    ('Leela Zero 0.17 88f40f0d', 20),       # 0d（名字正好以 d 结尾）
    ('Leela Zero 0.17 75ef0dc0', 20),       # 0d
]

#: 等级数字大得离谱 —— 只有范围那条约束拦得住。
OVERSIZED_RANK = [
    ('108K棋力', 1),        # 18k 被 clamp 到 1，但 108 级根本不存在
    ('200d', 2020),         # 词边界齐备，只有范围拦得住
]

#: 真实存在的级/段 —— 修复后必须**逐位不变**。
LEGIT_RANKS = [
    ('9d', 110),            # 20 + 9*10
    ('6d', 80),
    ('3k', 14),             # max(1, 20 - 3*2)
    ('15k', 1),             # 20 - 30 < 0 -> clamp 到 1
    ('武林高手3D', 50),
    ('KGS:9', 18),          # 无后缀 -> rating*2
    ('KGS:15d', 170),       # 有后缀 -> 20 + rating*10
]


@pytest.mark.parametrize('name,old_rating', HEX_HASH_NAMES,
                         ids=[n for n, _ in HEX_HASH_NAMES])
def test_hex_hash_is_not_a_rank(name, old_rating):
    """十六进制 commit hash 不是等级 —— 必须回落到默认 10。

    括号里的 `old_rating` 是**旧实现的错误输出**，写在这里是为了让这条测试
    明确「我在钉哪个 bug」：它当前会让测试失败。
    """
    assert old_rating != 10, '样本失效：这条不再体现旧 bug'
    assert parse_player_rating(name) == 10, (
        '%r 被当成了等级（旧实现给 %d）—— hash 里的 d 不是段位后缀'
        % (name, old_rating))


@pytest.mark.parametrize('name,old_rating', OVERSIZED_RANK,
                         ids=[n for n, _ in OVERSIZED_RANK])
def test_oversized_rank_is_not_a_rank(name, old_rating):
    """等级数字超出真实范围（>30）⇒ 不是等级，回落默认 10。"""
    assert parse_player_rating(name) == 10, (
        '%r 的等级数字 %d 不是真实存在的级/段' % (name, old_rating))


@pytest.mark.parametrize('name,want', LEGIT_RANKS,
                         ids=[n for n, _ in LEGIT_RANKS])
def test_real_dan_and_kyu_still_parse(name, want):
    """修复必须是**收窄**而不是误伤：真实级/段逐位不变。"""
    assert parse_player_rating(name) == want, (
        '%r 应得 %d，实得 %d' % (name, want, parse_player_rating(name)))


def test_empty_falls_back_to_default():
    assert parse_player_rating('') == 10
    assert parse_player_rating(None) == 10
    assert parse_player_rating('   ') == 10


def test_no_rank_also_falls_back_to_default():
    """连一个 d/k 都没有的普通名字 = 语料里 92% 的情况。"""
    for name in ('AlphaGo Zero', 'Cho Hun-hyeon', '三目中林', 'tdcq'):
        assert parse_player_rating(name) == 10, name


def test_every_real_name_yields_a_finite_sane_weight():
    """端到端：任意真实棋手名经 `compute_game_weight` 都不许溢出或越界。

    这是本 bug 的**实际杀伤面** —— rating 大了不报错，只有 `exp(avg/20)`
    溢出成 inf 才会顺着反向把整个模型的梯度打脏。
    """
    names = [n for n, _ in HEX_HASH_NAMES + OVERSIZED_RANK + LEGIT_RANKS]
    names += ['AlphaGo Zero', 'Leela Zero 0.17 e6e2e99b', 'ruo18k']
    ceiling = math.exp(10.0)
    for i, a in enumerate(names):
        for b in names[i:]:
            w = compute_game_weight(parse_player_rating(a),
                                    parse_player_rating(b))
            assert math.isfinite(w), '%r / %r -> %r' % (a, b, w)
            assert 0.0 < w <= ceiling, (
                '%r / %r -> weight %r 越界（上界 exp(10)=%r）' % (a, b, w, ceiling))
