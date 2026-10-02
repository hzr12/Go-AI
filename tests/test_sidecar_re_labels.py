"""局级 sidecar 的 `g_re` / `g_resign_side` 与派生的 outcome 契约。

A 阶段（硬 outcome 标签）只需要三样东西：**这局是什么形态的终局**（`g_re`）、
**有分差时黑赢多少**（`g_score`）、**认输时是谁认的**（`g_resign_side`）。
`derive_outcome` 把这三样压成一个 `int64` 标签，是下游唯一的入口契约。

三条要害
--------
1. ⚠ **`score is None` 单独出现分不清「认输」与「无结果」** —— 必须读
   `resign_side` / `is_resign`。语料里 63.8% 是认输，把它和残局混成一类会让
   A 阶段的标签整体反转。
2. ⚠ **`g_re` / `g_resign_side` 是局级标签**，搬运时左索引是 **slot**
   （`sidecar[game_ids]` 的下标）、右索引才是**语料局号**，两者不是一回事。
   写反了在两边大小恰好相同时会**静默串味**（见
   `test_re_labels_follow_slot_not_corpus_index`）。
3. ⚠ **未匹配的局不许被当成某一类终局** —— 回落到 `RE_CLASS_UNKNOWN` /
   `resign_side = -1`，`derive_outcome` 因此给 2（无信息），而不是白送一个胜负。
"""

import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from src.data.sgf_parser import ResultInfo, parse_result  # noqa: E402

import build_games_sidecar as B  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. ResultInfo.resign_side —— 解析层
# --------------------------------------------------------------------------- #
def _side_label(re_str):
    return f'RE[{re_str!r}]'


#: `(RE 串, resign_side)`。0 = 黑认输 / 1 = 白认输 / None = 非认输。
RESIGN_SIDE_CASES = [
    ('B+R', 0),          # 实测 21.94%
    ('B+Resign', 0),     # 实测 15.55%
    ('B+T', 0),          # 超时
    ('B+F', 0),          # 实测 19 局
    ('B+Time', 0),
    ('W+R', 1),          # 实测 21.75%
    ('W+Resign', 1),     # 实测 21.85%
    ('W+T', 1),
    ('W+F', 1),          # 实测 26 局
    ('W+Timeout', 1),
    # 小写前缀：`_RE_RESULT_RE` 只认大写 `B`/`W` ⇒ 小写串走「未识别」分支，
    # 方向无从得知 ⇒ 必须是 None，不许靠猜。
    ('w+r', None),
    # 非认输 ⇒ 一律 None
    ('B+2.5', None),
    ('W+3.5', None),
    ('B+7.5', None),
    ('W+1,25', None),
    ('0', None),
    ('draw', None),
    ('B+', None),
    ('W+', None),
    ('B', None),
    ('?', None),
    ('Void', None),
    ('B+Void', None),
    ('', None),
    (None, None),
]


@pytest.mark.parametrize('re_str,side', RESIGN_SIDE_CASES,
                         ids=[c[0] if c[0] else '<none>' for c in RESIGN_SIDE_CASES])
def test_parse_result_resign_side(re_str, side):
    assert parse_result(re_str).resign_side == side, _side_label(re_str)


def test_resign_side_is_set_exactly_when_is_resign():
    """⚠ 契约：`resign_side is not None` ⟺ `is_resign`。

    两者必须**同时**成立 —— 只看其中一个都会把「谁认输」和「有没有输」搞混。
    语料里 `B+` / `W+`（空分差，实测 95 局）胜负已定但分差不可知，那**不是认输**，
    `resign_side` 必须是 None，否则 A 阶段会把它们当成「有人认了」。
    """
    for re_str, _side in RESIGN_SIDE_CASES:
        ri = parse_result(re_str)
        assert (ri.resign_side is not None) == ri.is_resign, re_str
        if ri.resign_side is not None:
            assert ri.resign_side in (0, 1), (re_str, ri.resign_side)


def test_resign_side_reveals_who_lost():
    """`resign_side` 的语义是**谁认输**，不是谁赢 —— 名字容易读反，钉死。"""
    # `B+R` = 黑认输 ⇒ 白赢
    assert parse_result('B+R').resign_side == 0
    assert parse_result('W+R').resign_side == 1


def test_result_info_positional_construction_still_works():
    """⚠ 新字段必须**放在末尾且带默认值**，否则全仓 8 处 `ResultInfo(a, b, c)`
    的位置构造会一起炸（`sgf_parser.py` 内部 + `GameRecord` 的 default_factory）。"""
    ri = ResultInfo(2.5, False, False)
    assert (ri.score, ri.is_draw, ri.is_resign) == (2.5, False, False)
    assert ri.resign_side is None, '缺省必须是「非认输」而不是 0（0 = 黑认输）'

    ri2 = ResultInfo(None, False, True, 0)
    assert ri2.resign_side == 0


# --------------------------------------------------------------------------- #
# 2. derive_outcome —— 下游 A 阶段的入口契约
# --------------------------------------------------------------------------- #
OUTCOME_BLACK, OUTCOME_WHITE, OUTCOME_DRAW = 0, 1, 2


def test_derive_outcome_score_uses_black_minus_white_sign():
    """⚠ `g_score` **已经是黑−白分差**（含贴目），所以判胜负**不减 komi**。

    再减一次贴目会把 `B+2.5 / KM[7.5]` 这类局翻成白胜 —— 这是这个契约里最容易
    写错的一步，单独钉一条。
    """
    got = B.derive_outcome(
        np.array([B.RE_CLASS_SCORE] * 4, np.int8),
        np.array([2.5, -2.5, 0.25, -0.25], np.float16),
        np.full(4, -1, np.int8))
    assert got.tolist() == [OUTCOME_BLACK, OUTCOME_WHITE,
                            OUTCOME_BLACK, OUTCOME_WHITE]


def test_derive_outcome_resign_who_resigned_loses():
    """认输局：**谁认输谁输**，与分差无关（认输局本来就没有分差）。"""
    got = B.derive_outcome(
        np.array([B.RE_CLASS_RESIGN] * 4, np.int8),
        np.full(4, np.nan, np.float16),
        np.array([0, 0, 1, 1], np.int8))
    assert got.tolist() == [OUTCOME_WHITE, OUTCOME_WHITE,
                            OUTCOME_BLACK, OUTCOME_BLACK]


def test_derive_outcome_draw_and_unknown_are_both_draw_bucket():
    """和棋与「无结果/未识别」都归 2 —— A 阶段不需要区分它们，只要别硬造胜负。"""
    got = B.derive_outcome(
        np.array([B.RE_CLASS_DRAW, B.RE_CLASS_UNKNOWN], np.int8),
        np.array([0.0, np.nan], np.float16),
        np.array([-1, -1], np.int8))
    assert got.tolist() == [OUTCOME_DRAW, OUTCOME_DRAW]


def test_derive_outcome_class_wins_over_the_other_fields():
    """⚠ 契约是**按 `g_re` 分派**的：SCORE 看分差、RESIGN 看认输方，
    彼此的另一个字段一律不看。

    钉这条是因为 REAL 数据里两列可能来自不同来源（sidecar 复用旧 `.scan.npz`
    缓存时），一个「RESIGN 但残留了分差」的样本必须仍然按认输判。
    """
    got = B.derive_outcome(
        np.array([B.RE_CLASS_RESIGN, B.RE_CLASS_SCORE], np.int8),
        np.array([-99.0, 99.0], np.float16),      # 与认输方矛盾
        np.array([1, 0], np.int8))
    # RESIGN：resign_side=1（白认输）⇒ 黑胜，尽管分差是 -99
    # SCORE ：分差 +99 ⇒ 黑胜，尽管 resign_side=0（若按认输分支就会判成白胜）
    assert got.tolist() == [OUTCOME_BLACK, OUTCOME_BLACK]


def test_derive_outcome_dtype_and_shape():
    got = B.derive_outcome(
        np.zeros(5, np.int8), np.zeros(5, np.float16), np.full(5, -1, np.int8))
    assert got.dtype == np.int64
    assert got.shape == (5,)
    assert got.tolist() == [OUTCOME_DRAW] * 5      # 全 UNKNOWN


# --------------------------------------------------------------------------- #
# 3. 落盘往返 —— 用小规模合成语料跑**真的** build_sidecar
# --------------------------------------------------------------------------- #
def _lattice_points():
    """50 个**两两不相邻**的 19 路点位 —— 空枰上逐个落子全部合法。"""
    return [(r, c) for r in range(0, 19, 2) for c in range(0, 19, 4)]


def _coords(pt):
    r, c = pt
    return f'{chr(ord("a") + c)}{chr(ord("a") + r)}'


def _sgf(n_moves, km=None, re_str='B+R', ru=None, seed=0, pts=None):
    """造一局 19 路 SGF。

    ⚠ `seed` 必须让每局的落子序列真的不同：同一条确定性序列造出来的多局在每个
    手数上都落在同一局面，锚点会**互相匹配** —— 匹配率 100% 但元数据全串到同一局，
    测试反而测不出「对上正确的 SGF」。
    """
    if pts is None:
        pts = random.Random(seed).sample(_lattice_points(), len(_lattice_points()))
    body = ''.join(f';{"B" if i % 2 == 0 else "W"}[{_coords(pts[i])}]'
                   for i in range(n_moves))
    head = '(;GM[1]FF[4]SZ[19]'
    if km is not None:
        head += f'KM[{km}]'
    if ru is not None:
        head += f'RU[{ru}]'
    head += f'RE[{re_str}]PB[p]PW[q]\n'
    return head + body + ')'


def _write_sgf(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def _build_npz(sgf_dir, out_npz):
    """走真代码 `build_dataset.build` 造小 npz —— `contiguous_runs` / `anchor_rows`
    的语义（行 `i` = 已走满 `i` 手）都是它决定的，手搓 npz 会把待测契约改掉。"""
    from scripts.build_dataset import build
    data, n_games, skip = build(sgf_dir, 19, 0, chunk_size=0)
    np.savez(out_npz, **data)
    return data, n_games, skip


#: 四类 `RE` 各一局，认输那两局黑白各一 —— 覆盖「谁认输」的两个方向。
RE_SIDE_TINY = [
    # (文件名, KM, RE, RU) —— 文件名按 glob 升序 ⇒ game_id 0/1/2/3 同序
    ('a.sgf', '7.5', 'B+2.5', 'Chinese'),      # SCORE，黑胜
    ('b.sgf', '0', 'B+R', 'Japanese'),         # RESIGN，**黑**认输
    ('c.sgf', '5.5', 'W+T', None),             # RESIGN，**白**认输（超时）
    ('d.sgf', '6.5', 'W+3.5', 'Japanese'),     # SCORE，白胜
    # ⚠ DRAW 必须写 `0` 而不是 `draw`：`build_dataset.parse_result_to_value`
    #   只收 `B*` / `W*` / 可 float 的 0，`draw` 会被**拒收**（skip += 1）。
    ('e.sgf', '0', '0', None),
]


@pytest.fixture()
def re_tiny(tmp_path):
    sgf_dir = tmp_path / 'sgf'
    # ⚠ seed 必须随**下标**变，不能用 `len(name)`（'a.sgf'..'e.sgf' 全是 5 字符 ⇒
    # 同一个 seed ⇒ 五局落子序列完全相同 ⇒ 锚点互相匹配，元数据全串到同一局上）。
    for i, (name, km, re_str, ru) in enumerate(RE_SIDE_TINY):
        _write_sgf(str(sgf_dir / name),
                   _sgf(40, km=km, re_str=re_str, ru=ru, seed=1000 + i))
    npz = tmp_path / 'ds.npz'
    data, n_games, skip = _build_npz(str(sgf_dir), str(npz))
    assert n_games == len(RE_SIDE_TINY) and skip == 0, (n_games, skip)
    return tmp_path, npz, sgf_dir, data


def _build_sidecar(tmp_path, npz, sgf_dir):
    out = str(tmp_path / 'games.npz')
    B.build_sidecar(str(npz), [str(sgf_dir)], out, str(tmp_path / 'mat'),
                    log=lambda *_: None)
    return out, np.load(out)


def test_sidecar_persists_re_class_and_resign_side(re_tiny):
    """每局两列都要对上**它自己那一局**的 `RE`，认输局还要分清黑白。

    ⚠ 断言按 **slot**（= `game_id` 升序 = 文件名 glob 升序 a..e）写死，而不是按
    贴目反查：b 与 e 的贴目都是 0，按 KM 分不开它们；而「谁的标签挂到哪个 slot 上」
    正是这套搬运最容易错的地方，逐 slot 钉死才测得到。
    """
    tmp_path, npz, sgf_dir, data = re_tiny
    out, z = _build_sidecar(tmp_path, npz, sgf_dir)
    G = int(np.unique(data['game_ids']).size)
    assert G == len(RE_SIDE_TINY)
    assert z['g_re'].shape == (G,)
    assert z['g_resign_side'].shape == (G,)
    assert z['g_re'].dtype == np.int8
    assert z['g_resign_side'].dtype == np.int8

    # slot → (文件名, RE_CLASS_*, resign_side)
    expect = [
        (0, 'a.sgf', B.RE_CLASS_SCORE, -1),      # B+2.5，黑胜
        (1, 'b.sgf', B.RE_CLASS_RESIGN, 0),      # B+R ⇒ **黑**认输
        (2, 'c.sgf', B.RE_CLASS_RESIGN, 1),      # W+T ⇒ **白**认输（超时）
        (3, 'd.sgf', B.RE_CLASS_SCORE, -1),      # W+3.5，白胜
        (4, 'e.sgf', B.RE_CLASS_DRAW, -1),       # RE[0] ⇒ 和棋
    ]
    for g, name, re_cls, side in expect:
        assert (int(z['g_re'][g]), int(z['g_resign_side'][g])) == (re_cls, side), \
            f'slot {g} 应是 {name}，实得 re={z["g_re"][g]} side={z["g_resign_side"][g]}'
    # 认输的两局方向必须相反（b 与 c）
    sides = z['g_resign_side'][z['g_re'] == B.RE_CLASS_RESIGN].tolist()
    assert sorted(sides) == [0, 1], sides
    assert sorted(z['g_re'].tolist()) == [B.RE_CLASS_SCORE, B.RE_CLASS_SCORE,
                                          B.RE_CLASS_DRAW,
                                          B.RE_CLASS_RESIGN, B.RE_CLASS_RESIGN]


def test_sidecar_re_labels_agree_with_derive_outcome(re_tiny):
    """端到端契约：三列落盘 + `derive_outcome` ⇒ 每一局的硬 outcome。

    这是 A 阶段真正要走的那一步，所以在这里而不是在下游接一次。
    """
    tmp_path, npz, sgf_dir, _data = re_tiny
    out, z = _build_sidecar(tmp_path, npz, sgf_dir)
    outcome = B.derive_outcome(z['g_re'], z['g_score'], z['g_resign_side'])

    G = outcome.size
    assert outcome.dtype == np.int64 and outcome.shape == (G,)
    for g in range(G):
        re_cls = int(z['g_re'][g])
        km = float(z['g_komi'][g])
        if re_cls == B.RE_CLASS_SCORE:
            want = B.OUTCOME_BLACK if float(z['g_score'][g]) > 0 else B.OUTCOME_WHITE
        elif re_cls == B.RE_CLASS_RESIGN:
            # 谁认输谁输：resign_side 0 = 黑认输 ⇒ 白胜
            want = B.OUTCOME_WHITE if int(z['g_resign_side'][g]) == 0 else B.OUTCOME_BLACK
        else:
            want = B.OUTCOME_DRAW
        assert int(outcome[g]) == want, (g, re_cls, km)

    # a.sgf = B+2.5 / KM 7.5 ⇒ 黑胜，且**没有**被贴目翻面
    a = int(np.flatnonzero(z['g_komi'] == 7.5)[0])
    assert float(z['g_score'][a]) == 2.5 and int(outcome[a]) == B.OUTCOME_BLACK
    # c.sgf = W+T（白认输）⇒ 黑胜，尽管它 KM=5.5 而 RE 里根本没有分差
    c = int(np.flatnonzero(z['g_komi'] == 5.5)[0])
    assert np.isnan(z['g_score'][c]) and int(z['g_resign_side'][c]) == 1
    assert int(outcome[c]) == B.OUTCOME_BLACK


def test_sidecar_re_labels_survive_npz_roundtrip(re_tiny):
    """落盘 ↔ 读回必须逐位一致（int8 窄类型最容易在这里被静默升成 int64）。"""
    tmp_path, npz, sgf_dir, _data = re_tiny
    out, z = _build_sidecar(tmp_path, npz, sgf_dir)
    re_a, side_a = z['g_re'].copy(), z['g_resign_side'].copy()
    z.close()
    with np.load(out) as z2:
        assert np.array_equal(z2['g_re'], re_a)
        assert np.array_equal(z2['g_resign_side'], side_a)
        assert z2['g_re'].dtype == np.int8
        assert z2['g_resign_side'].dtype == np.int8
    # 复用缓存重建一遍，值必须一致（`.scan.npz` 往返不能改标签）
    out2 = str(tmp_path / 'games2.npz')
    B.build_sidecar(str(npz), [str(sgf_dir)], out2, str(tmp_path / 'mat'),
                    reuse_scan=True, log=lambda *_: None)
    with np.load(out2) as z3:
        assert np.array_equal(z3['g_re'], re_a)
        assert np.array_equal(z3['g_resign_side'], side_a)


def test_scan_cache_roundtrip_keeps_resign_side(re_tiny):
    """`.scan.npz` 缓存的往返：`resign_side` 必须进得去也出得来。

    ⚠ 旧缓存**恢复不出**这一列（`resign` 只是 bool，没有方向）⇒ 缺键时
    `_load_scan_cache` 必须返回 `(None, None)` 让调用方**重扫**，而不是静默
    填一个方向错乱的默认值。
    """
    tmp_path, npz, sgf_dir, _data = re_tiny
    scan, index = B.scan_corpus(sgf_dirs=[str(sgf_dir)], log=lambda *_: None)
    assert scan.re.size == len(RE_SIDE_TINY)
    assert sorted(scan.resign_side[scan.re == B.RE_CLASS_RESIGN].tolist()) == [0, 1]
    assert (scan.resign_side[scan.re != B.RE_CLASS_RESIGN] == -1).all()

    cache = str(tmp_path / 'scan.npz')
    B._save_scan_cache(cache, scan, index)
    scan2, index2 = B._load_scan_cache(cache)
    assert scan2 is not None and index2 is not None
    assert np.array_equal(scan2.resign_side, scan.resign_side)
    assert scan2.resign_side.dtype == np.int8
    assert np.array_equal(scan2.re, scan.re)
    assert np.array_equal(scan2.resign, scan.resign)

    # 砍掉 `resign_side` 键模拟旧缓存 ⇒ 必须触发重扫而不是给出垃圾方向
    z = dict(np.load(cache, allow_pickle=False))
    z.pop('resign_side')
    old = str(tmp_path / 'old.npz')
    np.savez(old, **z)
    assert B._load_scan_cache(old) == (None, None)


def test_re_labels_follow_slot_not_corpus_index(tmp_path):
    """⚠ **slot 与语料局号不是一回事**，搬运写反了会静默串味。

    构造：语料里多一个 `aa_extra.sgf`，它的 `RE[]` 为空 ⇒ `build_dataset` 拒收
    ⇒ 它进了语料散列表却**不在数据集里**。于是语料局号整体右移一位
    （extra=0 / b=1 / c=2 / d=3），而 sidecar 的 slot 仍是 0/1/2。
    把两列写反（`g_re[src] = scan.re[slots]`）时，要么 `IndexError`，要么在两边
    大小恰好相同时把 extra 的标签挂到某一局上。
    """
    full = tmp_path / 'sgf_all'
    _write_sgf(str(full / 'aa_extra.sgf'), _sgf(40, km='1.25', re_str='', seed=501))
    for i, (name, km, re_str, ru) in enumerate(RE_SIDE_TINY):
        _write_sgf(str(full / name),
                   _sgf(40, km=km, re_str=re_str, ru=ru, seed=600 + i))
    npz = tmp_path / 'ds.npz'
    data, n_games, skip = _build_npz(str(full), str(npz))
    assert skip == 1, f'空 RE 的 SGF 应被 build_dataset 拒收，却 skip={skip}'
    assert n_games == len(RE_SIDE_TINY), n_games

    out = str(tmp_path / 'games.npz')
    rep = B.build_sidecar(str(npz), [str(full)], out, str(tmp_path / 'mat'),
                          log=lambda *_: None)
    z = np.load(out)
    G = int(np.unique(data['game_ids']).size)
    assert rep['n_games'] == G == len(RE_SIDE_TINY)
    # glob 顺序：aa_extra, a, b, c, d, e ⇒ game_id 0=a, 1=b, 2=c, 3=d, 4=e
    # 被拒收的 extra（KM=1.25, RE 空 → UNKNOWN）一个标签都不许出现
    assert 1.25 not in z['g_komi'].astype(np.float32).tolist()
    assert B.RE_CLASS_UNKNOWN not in z['g_re'].tolist(), '语料局号 0 的标签串到了 slot 上'
    assert (z['g_resign_side'] == -1).sum() == 3      # a/d 是 SCORE、e 是 DRAW

    # 认输的两局仍然是 b（黑认输）与 c（白认输），方向没被 extra 带偏
    sides = z['g_resign_side'][z['g_re'] == B.RE_CLASS_RESIGN].tolist()
    assert sorted(sides) == [0, 1], sides


def test_unmatched_game_falls_back_to_unknown_not_a_fake_result(tmp_path):
    """⚠ 未匹配的局必须回落到 `RE_CLASS_UNKNOWN` / `resign_side = -1`。

    它**没有**终局信息 —— 若回落成「白认输」或某个具体类别，A 阶段就会凭空造出
    一批标签，而覆盖报告里的未匹配数也就失去了意义。
    """
    full = tmp_path / 'sgf_all'
    _write_sgf(str(full / 'a.sgf'), _sgf(40, km='7.5', re_str='B+2.5', seed=701))
    _write_sgf(str(full / 'b.sgf'), _sgf(40, km='5.5', re_str='B+R', seed=702))
    npz = tmp_path / 'ds.npz'
    _data, n_games, _skip = _build_npz(str(full), str(npz))
    assert n_games == 2

    part = tmp_path / 'sgf_part'
    _write_sgf(str(part / 'a.sgf'), _sgf(40, km='7.5', re_str='B+2.5', seed=701))
    out = str(tmp_path / 'games.npz')
    rep = B.build_sidecar(str(npz), [str(part)], out, str(tmp_path / 'mat'),
                          log=lambda *_: None)
    z = np.load(out)
    assert rep['n_unmatched'] == 1

    miss = int(np.flatnonzero(z['g_komi'] == 0.0)[0])
    assert int(z['g_re'][miss]) == B.RE_CLASS_UNKNOWN
    assert int(z['g_resign_side'][miss]) == -1
    assert int(B.derive_outcome(z['g_re'], z['g_score'], z['g_resign_side'])[miss]) \
        == B.OUTCOME_DRAW
    # 匹配上的那局该有真标签（`B+2.5`）
    hit = 1 - miss
    assert int(z['g_re'][hit]) == B.RE_CLASS_SCORE
    assert int(z['g_resign_side'][hit]) == -1