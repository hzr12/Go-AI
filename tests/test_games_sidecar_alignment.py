"""局级 sidecar `games.npz` 的对齐测试（spec §5.3 / pipeline D0）。

本文件只覆盖「**对不对得上**」，不碰大文件：主数据集 34.2M 行跑一次要几十分钟，
所以全部用**小规模合成数据**（`build_dataset.build` 从几个手写 SGF 真跑出来的
npz），把 `build_games_sidecar` 当被测对象。

三条要害
--------
1. **`RE` / `RU` 解析必须每个实测形态都有显式分支** —— 包括 `B+` / `W+F` /
   `?` / `Void` 这些边角。靠 `except` 判别未识别形态，会把「解析 bug」与
   「语料里的新形态」混成同一个症状。
2. **锚点匹配要对 `game_id` 复用免疫**（`build_dataset.py:314-316`）。sidecar 是
   **按 id 索引**的，一个 id 对应两段行时必须给同一个值，且不能取残段。
3. **未匹配的局不许静默填垃圾** —— `g_score` 必须是 `NaN`，且覆盖报告里看得见。
"""

import hashlib
import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from src.data.sgf_parser import (  # noqa: E402
    RULES_BIT_SCORING, RULES_DEFAULT, ResultInfo, parse_result, parse_rules,
    parse_sgf_string, recognized_rules,
)

import build_games_sidecar as B  # noqa: E402

BOARD = 19


# --------------------------------------------------------------------------- #
# 1. RE 数值解析
# --------------------------------------------------------------------------- #
#: `(RE 串, score, is_draw, is_resign)`。分母是 `data/games/games` 133,604 局实测。
RE_CASES = [
    # 有分差 18.5%：`score` 是**黑−白**，含贴目
    ('B+2.5', 2.5, False, False),
    ('W+3.5', -3.5, False, False),
    ('B+7.5', 7.5, False, False),
    ('W+1.25', -1.25, False, False),
    ('B+0.25', 0.25, False, False),
    # 认输 81.1%：无分差
    ('B+R', None, False, True),
    ('W+R', None, False, True),
    ('B+Resign', None, False, True),
    ('W+Resign', None, False, True),
    ('B+T', None, False, True),          # 超时 0.20%
    ('W+T', None, False, True),
    ('B+Forfeit', None, False, True),
    ('W+F', None, False, True),          # 实测 26 局
    ('B+F', None, False, True),          # 实测 19 局
    # 和棋 0.07%：`score=0.0` **且** `is_draw=True`（要与「无结果」分开）
    ('0', 0.0, True, False),
    ('draw', 0.0, True, False),
    ('Draw', 0.0, True, False),
    # 空分差 / 无结果：**都不是**认输、**都不是**和棋
    ('B+', None, False, False),          # 实测 43 局
    ('W+', None, False, False),          # 实测 52 局
    ('B', None, False, False),
    ('?', None, False, False),
    ('Void', None, False, False),
    ('', None, False, False),
    ('B+Void', None, False, False),
    # 欧陆逗号小数 / 带单位（各 1~2 局，但分支必须存在）
    ('W+0,25', -0.25, False, False),
    ('W+3 zi', -3.0, False, False),
    ('W+1 zi', -1.0, False, False),
]


@pytest.mark.parametrize('re_str,score,is_draw,is_resign', RE_CASES,
                         ids=[c[0] if c[0] else '<empty>' for c in RE_CASES])
def test_parse_result(re_str, score, is_draw, is_resign):
    got = parse_result(re_str)
    assert isinstance(got, ResultInfo)
    assert got.score == score, f'RE[{re_str!r}] score: {got.score!r} != {score!r}'
    assert got.is_draw is is_draw, f'RE[{re_str!r}] is_draw'
    assert got.is_resign is is_resign, f'RE[{re_str!r}] is_resign'


def test_parse_result_distinguishes_draw_from_absent():
    """⚠ 和棋与「无结果」**必须**能分开 —— 这是 `ResultInfo` 存在的全部理由。

    两者的 `score` 都是 0.0 / None 的组合不同：`is_draw` 是唯一判据。若实现退回
    「score=None 表示一切未知」，训练侧就无法把和棋（合法终局）从残局里摘出来。
    """
    d = parse_result('0')
    n = parse_result('?')
    r = parse_result('B+R')
    assert (d.is_draw, d.is_resign, d.score) == (True, False, 0.0)
    assert (n.is_draw, n.is_resign, n.score) == (False, False, None)
    assert (r.is_draw, r.is_resign, r.score) == (False, True, None)
    # 三者两两不同
    assert len({(d.is_draw, d.is_resign), (n.is_draw, n.is_resign),
                (r.is_draw, r.is_resign)}) == 3


def test_parse_result_sign_is_black_minus_white():
    """符号约定：黑胜为正、白胜为负（sidecar 的 `g_score` 直接用这个符号）。"""
    assert parse_result('B+2.5').score > 0
    assert parse_result('W+2.5').score < 0
    assert parse_result('B+2.5').score == -parse_result('W+2.5').score


def test_parse_result_never_raises_on_garbage():
    """未识别形态必须走显式分支，**不靠 `except`** —— 也不许抛。"""
    for junk in ('???', 'B+???', 'B++', 'B+ ', '   ', 'b+r', 'B+R extra',
                 'Draw?', 'B++2.5', 'B+2.5.5', 'B+1e3', 'Void+1'):
        r = parse_result(junk)
        assert isinstance(r, ResultInfo)
        assert r.is_draw is False, junk
        assert r.is_resign is False or r.is_resign is True, junk


def test_game_record_exposes_result_info_and_rules():
    """`GameRecord` 上新增的三个字段与 `properties` 同源。"""
    g = parse_sgf_string('(;SZ[19]KM[0]RU[Japanese]RE[B+2.5];B[dd];W[pp])')
    assert g.has_komi is True and g.komi == 0.0
    assert g.rules_flags == RULES_BIT_SCORING
    assert g.result_info.score == 2.5


def test_game_record_komi_missing_is_distinguishable_from_7_5():
    """⚠ `GameRecord.komi` 默认值是 7.5，所以「没有 KM」与「KM[7.5]」在 `komi`
    上不可区分。sidecar 的 `g_komi` 约定「缺失填 0」，靠的必须是 `has_komi`。"""
    a = parse_sgf_string('(;SZ[19];B[dd];W[pp])')
    b = parse_sgf_string('(;SZ[19]KM[7.5];B[dd];W[pp])')
    assert a.komi == b.komi == 7.5
    assert a.has_komi is False
    assert b.has_komi is True


# --------------------------------------------------------------------------- #
# 2. RU 解析
# --------------------------------------------------------------------------- #
def test_parse_rules_chinese_is_area():
    """`Chinese` → 区域计分 ⇒ `bit0 = 0` ⇒ 恰好等于 TT 默认值。"""
    assert parse_rules('Chinese') == RULES_DEFAULT == 0
    assert parse_rules('Chinese') & 1 == 0
    assert recognized_rules('Chinese') is True


def test_parse_rules_japanese_is_territory():
    """`Japanese` → 数目计分 ⇒ **只翻 bit0**，其余四位仍取默认。"""
    assert parse_rules('Japanese') == RULES_BIT_SCORING == 1
    assert recognized_rules('Japanese') is True


def test_parse_rules_missing_defaults_to_area():
    """无 `RU`（实测 55.03%）/ 空 `RU[]`（3 局）/ 未识别串 → 全部 TT 默认。"""
    for ru in (None, '', '   ', 'SomethingNew'):
        assert parse_rules(ru) == RULES_DEFAULT, ru
    assert recognized_rules('') is False


def test_parse_rules_only_bit0_is_ever_set():
    """⚠ **tax / ko / suicide / button 这四位在当前语料里无来源**，必须恒为 0。

    不是「碰巧没测到」而是结构性的：实测 45% 有 `RU` 的局里只有 `Chinese` 与
    `Japanese` 两个取值，`RU` 是单个词，压根没提到税/劫/自杀/还子。所以 `g_rules`
    只可能在 `0x00` 与 `0x01` 之间取值 —— 这条断言把「将来别偷偷往高位塞东西」
    钉死，bit 5..7 的扩展位（`RULES_BITS_USED = 0x1F` 之外）留白也是有意的。
    """
    for ru in ('Chinese', 'Japanese', '', None):
        v = parse_rules(ru)
        assert v in (0, 1), f'RU[{ru!r}] -> {v:#x}，不应超出 bit0'
        assert v & ~1 == 0


def test_game_record_rules_from_string():
    g = parse_sgf_string('(;SZ[19]RU[Japanese]KM[7.5]RE[W+R];B[dd])')
    assert g.rules_flags == RULES_BIT_SCORING
    h = parse_sgf_string('(;SZ[19]KM[7.5]RE[W+R];B[dd])')
    assert h.rules_flags == RULES_DEFAULT


# --------------------------------------------------------------------------- #
# 3~6. 锚点匹配（小规模合成数据）
# --------------------------------------------------------------------------- #
def _lattice_points():
    """50 个**两两不相邻**的 19 路点位（行 0,2,...,18 × 列 0,4,8,12,16）。

    互不相邻 ⇒ 空枰上逐个落子全部合法（每子 4 口气），不必为「合法性」写特例。
    """
    return [(r, c) for r in range(0, 19, 2) for c in range(0, 19, 4)]


def _coords(pt):
    r, c = pt
    return f'{chr(ord("a") + c)}{chr(ord("a") + r)}'


def _sgf(n_moves, km=None, re_str='B+R', ru=None, seed=0, illegal_last=False,
         pts=None):
    """造一局 19 路 SGF。

    ⚠ **`seed` 必须让每局的落子序列真的不同**：同一条确定性序列造出来的多局在每个
    手数上都落在同一局面，于是 ply-20 锚点**互相匹配** —— 匹配率 100% 但元数据
    全串到同一局上，测试反而测不出「对上正确的 SGF」。所以按 seed 从格点里随机取序
    （仍然两两不相邻 ⇒ 全部合法）。

    `pts` 显式给定落子序列（用来造「同一个布局被多个局走到」的情形）。
    `illegal_last=True` 在末尾追加一手**落在第 1 手已占的点上**的着法，用来触发
    `build_dataset.py:314-316` 的 `play()` 失败路径。
    """
    if pts is None:
        pts = random.Random(seed).sample(_lattice_points(), len(_lattice_points()))
    body = ''.join(f';{"B" if i % 2 == 0 else "W"}[{_coords(pts[i])}]'
                   for i in range(n_moves))
    if illegal_last:
        body += f';W[{_coords(pts[0])}]'
    head = '(;GM[1]FF[4]SZ[19]'
    if km is not None:
        head += f'KM[{km}]'
    if ru is not None:
        head += f'RU[{ru}]'
    head += f'RE[{re_str}]PB[p]PW[q]' + '\n'
    return head + body + ')'


def _write_sgf(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def _build_npz(sgf_dir, out_npz):
    """用**真实的** `build_dataset.build` 从 SGF 目录造一个小 npz。

    ⚠ 必须走真代码：`contiguous_runs` / `anchor_rows` 的语义（行 `i` = 已走满
    `i` 手、id 可以复用）都是 `build_dataset` 决定的，自己手搓 npz 会把
    待测契约一起改掉。
    """
    from scripts.build_dataset import build
    data, n_games, skip = build(sgf_dir, 19, 0, chunk_size=0)
    np.savez(out_npz, **data)
    return data, n_games, skip


@pytest.fixture()
def tiny(tmp_path):
    """3 局、贴目/结果/规则各不相同，主数据集 + 语料目录。"""
    sgf_dir = tmp_path / 'sgf'
    _write_sgf(str(sgf_dir / 'a.sgf'), _sgf(40, km='7.5', re_str='B+2.5', ru='Chinese', seed=1))
    _write_sgf(str(sgf_dir / 'b.sgf'), _sgf(40, km='0', re_str='W+R', ru=None, seed=2))
    _write_sgf(str(sgf_dir / 'c.sgf'), _sgf(40, km='5.5', re_str='W+3.5', ru='Japanese', seed=3))
    npz = tmp_path / 'ds.npz'
    data, n_games, skip = _build_npz(str(sgf_dir), str(npz))
    assert n_games == 3 and skip == 0, (n_games, skip)
    return tmp_path, npz, sgf_dir, data


def test_anchor_matches_each_game_to_the_right_sgf(tiny):
    """锚点能把 sidecar 的每一行对上**正确的那个 SGF**。"""
    tmp_path, npz, sgf_dir, data = tiny
    out = str(tmp_path / 'games.npz')
    rep = B.build_sidecar(str(npz), [str(sgf_dir)], out,
                          str(tmp_path / 'mat'), log=lambda *_: None)

    z = np.load(out)
    ids = np.unique(data['game_ids'])
    assert z['g_komi'].shape[0] == ids.size == 3
    assert rep['n_matched'] == 3 and rep['n_unmatched'] == 0
    assert rep['match_rate'] == 1.0

    # 每一行的元数据必须来自**同一局**，不能串味
    expect = {7.5: (2.5, RULES_DEFAULT, False), 0.0: (None, RULES_DEFAULT, True),
              5.5: (-3.5, RULES_BIT_SCORING, False)}
    for g in range(z['g_komi'].shape[0]):
        km = float(z['g_komi'][g])
        assert km in expect, f'g={g} 贴目 {km} 不是三局里任何一个'
        score, rules, resign = expect[km]
        if score is None:
            assert bool(np.isnan(z['g_score'][g]))
        else:
            assert float(z['g_score'][g]) == score
        assert int(z['g_rules'][g]) == rules
        assert bool(z['g_resign'][g]) is resign


def test_sidecar_row_count_equals_distinct_game_ids(tiny):
    """`G` 必须等于主 npz 的 distinct `game_ids` 数（读取口径是 `sidecar[game_ids]`）。"""
    tmp_path, npz, sgf_dir, data = tiny
    out = str(tmp_path / 'games.npz')
    rep = B.build_sidecar(str(npz), [str(sgf_dir)], out,
                          str(tmp_path / 'mat'), log=lambda *_: None)
    z = np.load(out)
    G = int(np.unique(data['game_ids']).size)
    assert z['g_komi'].shape == (G,)
    assert z['g_score'].shape == (G,)
    assert z['g_rules'].shape == (G,)
    assert z['g_resign'].shape == (G,)
    assert rep['n_games'] == G


def test_sidecar_dtypes_match_spec(tiny):
    tmp_path, npz, sgf_dir, _ = tiny
    out = str(tmp_path / 'games.npz')
    B.build_sidecar(str(npz), [str(sgf_dir)], out, str(tmp_path / 'mat'),
                    log=lambda *_: None)
    z = np.load(out)
    assert z['g_komi'].dtype == np.float16
    assert z['g_score'].dtype == np.float16
    assert z['g_rules'].dtype == np.int8
    assert z['g_resign'].dtype == np.bool_
    assert set(z.files) == {'g_komi', 'g_score', 'g_rules', 'g_resign'}


def test_anchor_is_immune_to_game_id_reuse(tmp_path):
    """复现 `build_dataset.py:314-316` 的 id 复用 bug，验证锚点仍按**盘面内容**选局。

    构造（`build_dataset` 按 glob 排序，文件名 `broken < good_a < good_c`）：
      1. `broken.sgf` —— 5 手合法 + 第 6 手落在已占点 ⇒ `play()` 失败 ⇒
         `return 0,1`。**已追加的 6 行留在 `cur` 里**，`game_id_counter` 未自增；
      2. `good_a.sgf` —— 完整 40 手局，**拿到 broken 留下的同一个 id 0**；
      3. `good_c.sgf` —— 完整 40 手局，id 1。

    ⇒ `game_ids = [0]*6 + [0]*40 + [1]*40`：**id 0 的 46 行来自两个不同的 SGF**
    （`broken` 的残段 + `good_a` 的完整局）。锚点 `min(20, 46//2) = 20` 落在
    `good_a` 那一侧（残段只占 0..5 行）⇒ 元数据取自 `good_a`，**不是残段那一局**。

    这就是「按 id 建表 / 按枚举顺序对齐」会错的地方：id 0 在两边指的不是同一局。
    """
    sgf_dir = tmp_path / 'sgf'
    _write_sgf(str(sgf_dir / 'good_a.sgf'),
               _sgf(40, km='7.5', re_str='B+2.5', ru='Chinese', seed=31))
    # 第 6 手下在第 1 手已经占住的点上 ⇒ 非法
    _write_sgf(str(sgf_dir / 'broken.sgf'),
               _sgf(5, km='99', re_str='W+R', ru='Japanese', seed=32, illegal_last=True))
    _write_sgf(str(sgf_dir / 'good_c.sgf'),
               _sgf(40, km='6.5', re_str='W+1.5', ru=None, seed=33))
    npz = tmp_path / 'ds.npz'
    data, n_games, skip = _build_npz(str(sgf_dir), str(npz))
    assert skip == 1 and n_games == 2, (n_games, skip)

    gids = data['game_ids']
    assert gids.size == 6 + 40 + 40, gids.size
    assert np.unique(gids).tolist() == [0, 1]
    # 前提自检：**id 0 的 46 行来自两个 SGF** —— 复用 bug 确实被造出来了
    ids, starts, ends = B.contiguous_runs(gids)
    assert ids.tolist() == [0, 1] and (ends - starts).tolist() == [46, 40]

    out = str(tmp_path / 'games.npz')
    rep = B.build_sidecar(str(npz), [str(sgf_dir)], out,
                          str(tmp_path / 'mat'), log=lambda *_: None)
    z = np.load(out)
    assert rep['n_games'] == 2
    assert z['g_komi'].shape == (2,)
    # id 0 → good_a（KM=7.5 / RE=B+2.5 / Chinese → AREA），**不是** broken 残段
    assert float(z['g_komi'][0]) == 7.5
    assert float(z['g_score'][0]) == 2.5
    assert int(z['g_rules'][0]) == RULES_DEFAULT
    assert bool(z['g_resign'][0]) is False
    # id 1 → good_c（KM=6.5 / RE=W+1.5 / 无 RU → 默认 AREA）
    assert float(z['g_komi'][1]) == 6.5
    assert float(z['g_score'][1]) == -1.5
    # 残段的元数据（KM=99 / Japanese → TERRITORY / 认输）**一个都不许出现**
    assert 99.0 not in z['g_komi'].astype(np.float32).tolist()
    assert int(z['g_rules'][0]) != RULES_BIT_SCORING
    assert not bool(z['g_resign'][0])
    # 本构造下两个锚点都落在 good_a 一侧 ⇒ 交叉校验一致
    assert rep['n_anchor_disagree'] == 0, rep


def test_anchor_crosscheck_flags_a_run_that_spans_two_games(tmp_path):
    """⚠ id 复用的**危险变体**：该段跨了两局，必须**报出来**。

    `broken` 留下 15 行（走满 14 手后第 15 手非法 ⇒ 15 行已 append），`good_a` 有
    40 行 ⇒ 同一个 `game_id` 的段长 55。主锚点 `a1 = min(20, 55//2) = 20` 落在
    **完整局**那一侧（残段只占 0..14 行），交叉锚点 `a2 = 10` 落在**残段**里 ⇒
    两个锚点解析到不同的 SGF，正是「这一段跨了局」的证据。

    元数据取自哪一局在这里**真的有歧义**（残段占 15/55 行），所以本脚本不改匹配
    结果（实测主数据集没有这种段），而是让 `n_anchor_disagree` 把它喊出来。

    这条断言钉的是「不静默」：**边界要可见**，而不是悄悄挑一个。
    """
    sgf_dir = tmp_path / 'sgf'
    _write_sgf(str(sgf_dir / 'broken.sgf'),
               _sgf(14, km='99', re_str='W+R', ru='Japanese', seed=41, illegal_last=True))
    _write_sgf(str(sgf_dir / 'good_a.sgf'),
               _sgf(40, km='7.5', re_str='B+2.5', ru='Chinese', seed=42))
    npz = tmp_path / 'ds.npz'
    data, n_games, skip = _build_npz(str(sgf_dir), str(npz))
    assert skip == 1 and n_games == 1
    ids, starts, ends = B.contiguous_runs(data['game_ids'])
    assert ids.tolist() == [0] and (ends - starts).tolist() == [55]
    assert B.anchor_rows(starts, ends).tolist() == [20]
    assert B.anchor_rows_cross(starts, ends)[0].tolist() == [10]

    rep = B.build_sidecar(str(npz), [str(sgf_dir)], str(tmp_path / 'games.npz'),
                          str(tmp_path / 'mat'), log=lambda *_: None)
    assert rep['n_games'] == 1
    assert rep['n_anchor_disagree'] == 1, rep


def test_anchor_crosscheck_offset_is_valid_and_distinct():
    """第二锚点必须**落在段内且与主锚点不同**，否则这道校验是假的。"""
    starts = np.array([0, 100, 200, 300], np.int64)
    ends = np.array([210, 130, 203, 301], np.int64)     # L = 210 / 30 / 3 / 1
    primary = B.anchor_rows(starts, ends)
    cross, ok = B.anchor_rows_cross(starts, ends)
    assert primary.tolist() == [20, 115, 201, 300]
    assert ok.tolist() == [True, True, True, False]      # L=1 没有第二锚点
    assert cross[:3].tolist() != primary[:3].tolist()
    for s, e, c, o, k in zip(starts, ends, cross, ok, primary):
        if not o:
            continue
        assert s <= c < e, '第二锚点越出该段'
        assert c != k


def test_unmatched_game_is_nan_not_garbage(tmp_path):
    """语料里缺一局 ⇒ `g_score` 必须是 `NaN`、**不抛异常**，且报告里看得见。"""
    full = tmp_path / 'sgf_all'
    _write_sgf(str(full / 'a.sgf'), _sgf(40, km='7.5', re_str='B+2.5', ru='Chinese', seed=11))
    _write_sgf(str(full / 'b.sgf'), _sgf(40, km='5.5', re_str='W+R', ru='Japanese', seed=12))
    _write_sgf(str(full / 'c.sgf'), _sgf(40, km='6.5', re_str='W+3.5', ru=None, seed=13))
    npz = tmp_path / 'ds.npz'
    data, n_games, skip = _build_npz(str(full), str(npz))
    assert n_games == 3

    # 只把 a、b 交给 sidecar；c 的 SGF 不在语料里 ⇒ 必然匹配不上
    part = tmp_path / 'sgf_part'
    _write_sgf(str(part / 'a.sgf'), _sgf(40, km='7.5', re_str='B+2.5', ru='Chinese', seed=11))
    _write_sgf(str(part / 'b.sgf'), _sgf(40, km='5.5', re_str='W+R', ru='Japanese', seed=12))

    out = str(tmp_path / 'games.npz')
    rep = B.build_sidecar(str(npz), [str(part)], out,
                          str(tmp_path / 'mat'), log=lambda *_: None)   # 不抛
    z = np.load(out)
    assert rep['n_matched'] == 2
    assert rep['n_unmatched'] == 1
    assert rep['match_rate'] == 2 / 3
    assert rep['n_games'] == 3

    # ⚠ **`g_score` 是 NaN 有两个来源，必须分开看**：认输（matched 但无分差）与
    # 未匹配（连贴目都不知道）。这里三局的贴目都非 0，所以 `g_komi == 0` 唯一地
    # 标出未匹配的那一局。
    a, b = (int(np.flatnonzero(z['g_komi'] == v)[0]) for v in (7.5, 5.5))
    miss = int(np.flatnonzero(z['g_komi'] == 0.0)[0])
    assert len({a, b, miss}) == 3

    # matched + 有分差
    assert float(z['g_score'][a]) == 2.5 and bool(z['g_resign'][a]) is False
    # matched + 认输 ⇒ 分差 NaN，但**贴目和规则仍在**（不能与未匹配混为一谈）
    assert np.isnan(z['g_score'][b]) and float(z['g_komi'][b]) == 5.5
    assert bool(z['g_resign'][b]) is True
    assert int(z['g_rules'][b]) == RULES_BIT_SCORING
    # 未匹配 ⇒ 四项全部回落到「无信息」的约定值，不是垃圾
    assert np.isnan(z['g_score'][miss])
    assert float(z['g_komi'][miss]) == 0.0
    assert int(z['g_rules'][miss]) == RULES_DEFAULT
    assert bool(z['g_resign'][miss]) is False


def test_report_prints_unmatched_count(tmp_path, capsys):
    """⚠ 覆盖报告**必须**把未匹配数打在 stdout 上（「不静默填垃圾」）。"""
    full = tmp_path / 'sgf_all'
    _write_sgf(str(full / 'a.sgf'), _sgf(40, km='7.5', re_str='B+2.5', seed=21))
    _write_sgf(str(full / 'b.sgf'), _sgf(40, km='5.5', re_str='W+R', seed=22))
    npz = tmp_path / 'ds.npz'
    _build_npz(str(full), str(npz))
    part = tmp_path / 'sgf_part'
    _write_sgf(str(part / 'a.sgf'), _sgf(40, km='7.5', re_str='B+2.5', seed=21))
    B.build_sidecar(str(npz), [str(part)], str(tmp_path / 'games.npz'),
                    str(tmp_path / 'mat'))
    out = capsys.readouterr().out
    assert '未匹配' in out
    assert '1 / 2' in out.replace('  ', ' ') or '1/2' in out.replace(' ', '')


def test_sidecar_slot_order_follows_game_id_not_corpus_order(tmp_path):
    """⚠ **slot 下标与语料局号不是一回事**，写反了会静默串味。

    这里让两者**故意错开**：语料里多一个 `aa_extra.sgf`，它的 `RE[]` 让
    `parse_result_to_value` 返回 `None` ⇒ `build_dataset` 拒收它 ⇒ 它进了语料
    散列表却**不在数据集里**。于是语料局号整体右移一位（extra=0, a=1, b=2），
    而 sidecar 的 slot 仍是 0/1。

    若散射的左索引错用语料局号，要么 `IndexError`，要么在两边大小恰好相同时
    静默把 extra 的贴目（1.25）挂到某一局上。
    """
    full = tmp_path / 'sgf_all'
    _write_sgf(str(full / 'aa_extra.sgf'),
               _sgf(40, km='1.25', re_str='', seed=51))
    _write_sgf(str(full / 'b.sgf'), _sgf(40, km='7.5', re_str='B+2.5', seed=52))
    _write_sgf(str(full / 'c.sgf'), _sgf(40, km='6.5', re_str='W+3.5', seed=53))
    npz = tmp_path / 'ds.npz'
    data, n_games, skip = _build_npz(str(full), str(npz))
    assert n_games == 2 and skip == 1, (n_games, skip)

    out = str(tmp_path / 'games.npz')
    rep = B.build_sidecar(str(npz), [str(full)], out,
                          str(tmp_path / 'mat'), log=lambda *_: None)
    z = np.load(out)
    assert rep['n_games'] == 2
    # glob 顺序 b → c ⇒ game_id 0 = b(7.5)，1 = c(6.5)
    assert z['g_komi'].astype(np.float32).tolist() == [7.5, 6.5]
    assert 1.25 not in z['g_komi'].astype(np.float32).tolist(), '被拒收的 SGF 混进来了'


def test_corpus_scan_covers_tgz_and_plain_files(tmp_path):
    """两种语料来源都要覆盖：`.tgz`（流式 `r|gz`）与目录里的裸 `.sgf`。"""
    import tarfile
    plain = tmp_path / 'plain'
    _write_sgf(str(plain / 'p1.sgf'), _sgf(40, km='6.5', re_str='W+1.5', seed=61))
    _write_sgf(str(plain / 'p2.sgf'), _sgf(40, km='5.5', re_str='B+R', seed=62))
    arc = tmp_path / 'arch.tgz'
    with tarfile.open(arc, 'w:gz') as tf:
        for name, km, re_str, seed in (('t1.sgf', '0', 'W+2.5', 63),
                                       ('t2.sgf', '7.5', 'B+7.5', 64)):
            data = _sgf(40, km=km, re_str=re_str, seed=seed).encode('utf-8')
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, __import__('io').BytesIO(data))

    npz = tmp_path / 'ds.npz'
    data, n_games, skip = _build_npz(str(tmp_path), str(npz))
    assert n_games == 4 and skip == 0, (n_games, skip)

    archives, files = B.plan_sources([str(tmp_path)])
    assert archives == [str(arc)]
    assert len(files) == 2

    out = str(tmp_path / 'games.npz')
    rep = B.build_sidecar(str(npz), [str(tmp_path)], out,
                          str(tmp_path / 'mat'), log=lambda *_: None)
    z = np.load(out)
    assert rep['n_sgf'] == 4 and rep['n_matched'] == 4
    got = sorted(z['g_komi'].astype(np.float32).tolist())
    assert got == [0.0, 5.5, 6.5, 7.5], got


def test_plan_sources_dedupes_overlapping_dirs(tmp_path):
    """⚠ 默认的 `--sgf-dirs` 是 `["data", "data/games/games"]`，后者就在前者底下。

    不去重就会把 133,604 个 SGF 扫两遍 ⇒ 散列表出现重复项、匹配率报告失真。
    """
    inner = tmp_path / 'data' / 'games' / 'games'
    _write_sgf(str(inner / 'x.sgf'), _sgf(40, seed=71))
    archives, files = B.plan_sources([str(tmp_path / 'data'),
                                      str(inner)])
    assert archives == []
    assert files == [str(inner / 'x.sgf')]


def test_anchor_crosscheck_does_not_false_alarm_on_a_shared_opening(tmp_path):
    """⚠ 交叉锚点命中**多个局**不是「跨局」，是**布局歧义** —— 两者必须分开。

    真实语料里同一个常见布局（实测 `--limit-games 50` 探针的偏移 10）被 **4~10 个
    不同的局**走到。若拿「第一条命中」去比主锚点，就会把这些**布局歧义**误报成
    「这一段跨了两局」，报告里凭空冒出 5/50。

    构造：A、B 前 10 手**完全相同**（⇒ 偏移 10 的局面两者共有，交叉锚点有歧义），
    第 11 手起分道扬镳（⇒ 偏移 20 不同，主锚点唯一）。断言 `n_anchor_disagree == 0`
    且两局都匹上。
    """
    lattice = _lattice_points()
    common = random.Random(99).sample(lattice, 10)
    taken = set(common)
    rest = [p for p in lattice if p not in taken]
    a_tail = random.Random(101).sample(rest, 30)
    b_tail = random.Random(202).sample(rest, 30)
    assert set(a_tail) != set(b_tail)
    sgf_dir = tmp_path / 'sgf'
    _write_sgf(str(sgf_dir / 'a.sgf'),
               _sgf(40, km='7.5', re_str='B+2.5', pts=common + a_tail))
    _write_sgf(str(sgf_dir / 'b.sgf'),
               _sgf(40, km='6.5', re_str='W+3.5', pts=common + b_tail))
    npz = tmp_path / 'ds.npz'
    _data, n_games, skip = _build_npz(str(sgf_dir), str(npz))
    assert n_games == 2 and skip == 0

    out = str(tmp_path / 'games.npz')
    rep = B.build_sidecar(str(npz), [str(sgf_dir)], out,
                          str(tmp_path / 'mat'), log=lambda *_: None)
    assert rep['n_matched'] == 2
    assert rep['n_anchor_disagree'] == 0, rep
    assert rep['n_anchor_ambiguous_cross'] == 2, '两局的交叉锚点都该是歧义的'
    assert rep['n_anchor_ambiguous'] == 0, '主锚点（第 20 手）应当唯一'
    z = np.load(out)
    assert sorted(z['g_komi'].astype(np.float32).tolist()) == [6.5, 7.5]


def test_distinct_signature_counts_separates_duplicates_from_different_games():
    """⚠ `counts > 1`（有副本，无害）与「签名 > 1」（**不同的棋**，元数据可能挂错）
    是两件事，必须能分开数 —— 否则报告里那个数没有行动价值。"""
    idx = B.AnchorIndex()
    idx.n_sgf = 4
    # 局面 A 被「同一局的两个副本」(局 0/1) 与「另一局」(局 2) 走到；
    # 局面 B 只被局 3 走到。
    idx.add(np.array([11, 11, 11, 22], np.uint64),
            np.array([0, 1, 2, 3], np.int32),
            np.array([20, 20, 20, 20], np.int8))
    idx._finalize()
    sig = np.array([777, 777, 888, 999], np.uint64)     # 局 0/1 同签，局 2 异签
    got = B.distinct_signature_counts(idx, np.array([11, 22, 33], np.uint64), sig)
    assert got.tolist() == [2, 1, 0]                    # 11→2 个不同的棋；33 未命中


def test_dataset_npz_is_opened_read_only(tiny, tmp_path):
    """⛔ 主数据集必须纯只读。比对运行前后的 sha256。"""
    _tmp, npz, sgf_dir, _ = tiny
    before = hashlib.sha256(open(npz, 'rb').read()).hexdigest()
    B.build_sidecar(npz, [str(sgf_dir)], str(tmp_path / 'games.npz'),
                    str(tmp_path / 'mat'), log=lambda *_: None)
    after = hashlib.sha256(open(npz, 'rb').read()).hexdigest()
    assert before == after, '主数据集被改动了'


# --------------------------------------------------------------------------- #
# 行区间 / 锚点行号本身
# --------------------------------------------------------------------------- #
def test_contiguous_runs_handles_shuffled_and_reused_ids():
    """`game_ids` 实测**不是升序**（是 0..N-1 的置换），且 id 可被复用 ⇒
    按 distinct id 建表会把两段不连续的行算成一个锚点。"""
    g = np.array([2, 2, 2, 0, 0, 2, 1], dtype=np.int32)
    ids, starts, ends = B.contiguous_runs(g)
    assert ids.tolist() == [2, 0, 2, 1]
    assert starts.tolist() == [0, 3, 5, 6]
    assert ends.tolist() == [3, 5, 6, 7]
    assert np.unique(ids).size == 3          # distinct id 数，与段数 4 不同


def test_anchor_rows_formula():
    """`r = start + min(20, L//2)`；`L ≥ 40` 时恒为 20。"""
    starts = np.array([0, 100, 200], np.int64)
    ends = np.array([210, 130, 206], np.int64)      # L = 210 / 30 / 6
    assert B.anchor_rows(starts, ends).tolist() == [20, 115, 203]


def test_komi_outlier_fix_is_opt_out():
    """fox 语料把贴目写成 ×100（`KM[650]` = 6.5）。默认修，可关。"""
    assert B.parse_komi({'KM': '650'}) == (6.5, True)
    assert B.parse_komi({'KM': '650'}, fix_outlier=False) == (650.0, False)
    assert B.parse_komi({'KM': '7.5'}) == (7.5, False)
    assert B.parse_komi({}) == (0.0, False)              # 缺失填 0
    assert B.parse_komi({'KM': 'junk'}) == (0.0, False)


def test_replay_prefixes_match_dataset_rows(tiny):
    """SGF 侧重放的第 k 个前缀必须**逐位等于**主数据集第 k 行的盘面。

    这是整个方案的地基：两侧的落子/提子/判罚只要有一处不一致，散列就全不同而
    join 结果为空 —— 且症状看不出原因。
    """
    _tmp, npz, sgf_dir, data = tiny
    _ids, starts, _ends = B.contiguous_runs(data['game_ids'])
    boards = data['boards'][starts[0]:starts[0] + 25]
    to_play = data['to_play'][starts[0]:starts[0] + 25]
    ko = data['ko'][starts[0]:starts[0] + 25]

    parser_g = B.SGFParser()
    game = parser_g.parse_file(str(sgf_dir / 'a.sgf'))
    pb, tp, kp, offs = B.replay_anchor_positions(game)
    assert offs.tolist() == list(range(1, 21))
    # 散列逐位相等（比逐行比数组更直接地测「两侧口径一致」）。
    # ⚠ 对齐关系：数据行 `i` = **已走满 i 手**的局面（`build_dataset.py:303` 先
    # append 盘面再 `play()`），所以 SGF 侧的第 k 个前缀对上数据行 **k**，行 0 是空枰。
    h_ds = B.pos_hash_block(boards, to_play, ko)
    h_sgf = B.pos_hash_block(pb, tp, kp)
    assert np.array_equal(h_ds[1:21], h_sgf), 'SGF 侧重放与主数据集行不对齐'