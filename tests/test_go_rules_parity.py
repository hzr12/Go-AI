"""P2.6b-1 — GoBoard 规则/计分对拍语料。

语料文件 `tests/data/go_parity.json`，每条局面字段：

- `id` / `covers`：用例标识与覆盖点说明；
- `board_size` / `komi` / `to_play`：盘口、贴目、轮到谁（1 黑 / -1 白）；
- `rows`：自上而下的棋盘行字符串，`X`=黑 `O`=白 `.`=空；
- `moves`（可选）：`[r, c]` 序列，**从空盘**按序落子（黑先）。给 PSK/劫类局面用——
  重复判定依赖历史，只有走出来的局面才有意义；
- `expected_score`（可选）：期望分数，约定 **`score = B_area - W_area - komi`（黑视角，>0 黑胜）**。
  `B_area` = 黑子数 + 只接触黑色的空区；`W_area` 同理；其余空点中立。死子不另计、
  也不要求 pass-alive（引擎无死子判定，故语料一律预先无死子）；komi 加给白方；
- `expected_legal_all`（可选）：为真则该局所有落点 + pass 均合法；
- `expected_legal_excludes` / `expected_legal_includes`（可选）：必须非法 / 必须合法的动作下标，
  动作下标 = `r * board_size + c`，pass = `board_size * board_size`。

所有期望值都是**手工推导**后写入语料，测试只负责回放核对；推导过程见
`.superpowers/sdd/2026-09-25-v21-roadmap/task-p2-6b-1-report.md`。

几何约束（写语料时用来校验可行性）：`board_size^2 = B_area + W_area + neutral`，
因此给定目标差值即固定中立点数的奇偶（5 路差值 8 必配奇数中立点，差值 7 必配偶数）。

合法性（禁自杀 / 简单劫 / 超级劫 / 填眼 / snapback）已由 `tests/test_go_rules_legality.py`
与 `tests/test_go_hash.py` 覆盖；本文件只做端到端合订，不重复那些用例。
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.game.go_rules import GoBoard  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), 'data', 'go_parity.json')


def _load_corpus():
    if not os.path.exists(CORPUS):
        pytest.fail(f"对拍语料缺失: {CORPUS}")
    with open(CORPUS, encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:  # 可读失败信息，别裸抛
            pytest.fail(f"对拍语料不是合法 JSON: {CORPUS}\n{e}")
    positions = data.get('positions')
    if not isinstance(positions, list) or not positions:
        pytest.fail(f"对拍语料缺少非空 positions 列表: {CORPUS}")
    return positions


CORPUS_POSITIONS = _load_corpus()
IDS = [p.get('id', f'<第 {i} 条缺 id>') for i, p in enumerate(CORPUS_POSITIONS)]


def _board_from_case(case):
    """按语料构造 GoBoard：`rows` 先摆子（缺省空盘），再在其上回放 `moves`。

    直接摆子属于 in-place 改 `board.board`，之后**必须** `resync_hash()`，
    否则哈希陈旧、重复判定会用到过期历史。`moves` 用 `play()` 走，因此自带历史，
    重复判定（PSK/劫）类局面要表达历史时必须走这条路。
    """
    n = case['board_size']
    b = GoBoard(n, komi=case['komi'])
    if 'rows' in case:
        rows = case['rows']
        assert len(rows) == n, f"语料 {case['id']} 的 rows 行数与 board_size 不符"
        for r, row in enumerate(rows):
            assert len(row) == n, f"语料 {case['id']} 第 {r} 行列数与 board_size 不符"
            for c, ch in enumerate(row):
                if ch == 'X':
                    b.board[r, c] = 1
                elif ch == 'O':
                    b.board[r, c] = -1
                elif ch != '.':
                    pytest.fail(f"语料 {case['id']} 第 {r} 行第 {c} 列出现未知字符 {ch!r}")
        b.resync_hash()
        b.current_player = case['to_play']
    if 'moves' in case:
        for mv in case['moves']:
            r, c = mv
            ok = b.play(r * n + c)
            assert ok, f"语料 {case['id']} 的走子 {mv} 非法，无法回放"
    return b
    assert len(rows) == n, f"语料 {case['id']} 的 rows 行数与 board_size 不符"
    for r, row in enumerate(rows):
        assert len(row) == n, f"语料 {case['id']} 第 {r} 行列数与 board_size 不符"
        for c, ch in enumerate(row):
            if ch == 'X':
                b.board[r, c] = 1
            elif ch == 'O':
                b.board[r, c] = -1
            elif ch != '.':
                pytest.fail(f"语料 {case['id']} 第 {r} 行第 {c} 列出现未知字符 {ch!r}")
    b.resync_hash()
    b.current_player = case['to_play']
    return b


@pytest.mark.parametrize('case', CORPUS_POSITIONS, ids=IDS)
def test_corpus_score(case):
    """语料的期望分数必须与引擎一致（期望值来自规则推导，不是录制）。"""
    if 'expected_score' not in case:
        pytest.skip(f"语料 {case['id']} 未声明 expected_score")
    b = _board_from_case(case)
    got = b.score()
    assert got == pytest.approx(case['expected_score']), (
        f"语料 {case['id']} 期望 {case['expected_score']}，引擎给出 {got}"
    )
    # 面积计分必须落在物理范围内，且满足整数分解 n*n = B + W + neutral
    n = case['board_size']
    assert -(n * n + case['komi']) <= got <= (n * n - case['komi'])


@pytest.mark.parametrize('case', CORPUS_POSITIONS, ids=IDS)
def test_corpus_legality(case):
    """语料的合法点期望必须与掩码、`is_legal()`、`legal_actions()` 三者一致。"""
    b = _board_from_case(case)
    n = case['board_size']
    mask = b.get_legal_moves()
    actions = b.legal_actions()
    assert actions == sorted(actions), f"语料 {case['id']}: legal_actions 未升序"
    assert b.PASS in actions, f"语料 {case['id']}: legal_actions 缺 PASS"
    assert b.num_actions() == n * n + 1
    for a in range(n * n + 1):
        assert b.is_legal(a) == (a == b.PASS or bool(mask[a])), (
            f"语料 {case['id']}: is_legal({a}) 与掩码不一致"
        )
    if case.get('expected_legal_all'):
        assert mask.all(), f"语料 {case['id']}: 期望全部落点合法，实际有非法点"
    for a in case.get('expected_legal_excludes', []):
        assert a < n * n, f"语料 {case['id']}: expected_legal_excludes 含越界动作 {a}"
        assert not mask[a], f"语料 {case['id']}: 动作 {a} 期望非法，但掩码判为合法"
    for a in case.get('expected_legal_includes', []):
        assert a < n * n, f"语料 {case['id']}: expected_legal_includes 含越界动作 {a}"
        assert mask[a], f"语料 {case['id']}: 动作 {a} 期望合法，但掩码判为非法"


@pytest.mark.parametrize('case', CORPUS_POSITIONS, ids=IDS)
def test_corpus_result_matches_score_sign(case):
    """result() 与 score() 的符号口径必须一致。"""
    b = _board_from_case(case)
    s = b.score()
    r = b.result()
    assert r == (1 if s > 0 else (-1 if s < 0 else 0)), f"语料 {case['id']}: result 与 score 口径不符"


def test_corpus_komi_half_point_no_integer_tie():
    """半目贴目（7.5/6.5）下不可能出现整数和棋：score 必为 x.5。"""
    for case in CORPUS_POSITIONS:
        if case.get('expected_score') is None:
            continue
        got = case['expected_score']
        assert abs(got * 2 - round(got * 2)) < 1e-9, (
            f"语料 {case['id']}: 贴目 {case['komi']} 下出现整数分 {got}"
        )
        assert got != 0, f"语料 {case['id']}: 出现和棋分数 0"


def test_corpus_scores_are_board_only_not_history():
    """语料的分数只依赖盘面：把盘面复制成新实例（无历史）后分数不变。"""
    for case in CORPUS_POSITIONS:
        if 'expected_score' not in case:
            continue
        b = _board_from_case(case)
        fresh = GoBoard(case['board_size'], komi=case['komi'])
        fresh.board = np.array(b.board, copy=True)
        fresh.resync_hash()
        assert fresh.score() == pytest.approx(case['expected_score']), (
            f"语料 {case['id']}: 分数依赖了历史状态而非盘面"
        )
