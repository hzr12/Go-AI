"""
GoBoard 规则引擎测试。
核心回归：随机对弈黑胜率不得落在 [0.95, 1.0]（锁死旧 GoGame 奖励恒为 +1 的 bug）。
"""
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.go_rules import GoBoard


def test_basic_capture():
    """被完全包围的对方子必须被提掉。"""
    b = GoBoard(9, komi=6.5)
    b.board[:] = 0
    b.board[1, 1] = -1  # 白
    b.board[0, 1] = 1   # 黑
    b.board[2, 1] = 1   # 黑
    b.board[ 1, 0] = 1  # 黑
    b.current_player = 1
    # 黑下 (1,2) 应提掉 (1,1)
    assert b.play(1 * 9 + 2) is True
    assert b.board[1, 1] == 0, "白子应被提掉"
    print("PASS test_basic_capture")


def test_suicide_illegal():
    """自杀手非法（除非能提子，这里仅测纯自杀）。"""
    b = GoBoard(9)
    # 白占据 (0,1),(1,0),(1,1)，黑下 (0,0) 自身无气且未提子 -> 非法
    b.board[0, 1] = -1
    b.board[1, 0] = -1
    b.board[1, 1] = -1
    b.current_player = 1
    ok = b.play(0 * 9 + 0)
    assert ok is False, "纯自杀应非法"
    assert b.board[0, 0] == 0
    print("PASS test_suicide_illegal")


def test_ko():
    """劫争后立即回提必须非法。构造标准 ko 形：黑下 (1,2) 提白 (1,1) 形成劫。"""
    b = GoBoard(9)
    b.board[:] = 0
    # 行/列 (r,c): (0,1)黑 (0,2)白 (1,0)黑 (1,1)白 (1,3)白 (2,1)黑 (2,2)白
    b.board[0, 1] = 1
    b.board[0, 2] = -1
    b.board[1, 0] = 1
    b.board[1, 1] = -1
    b.board[1, 3] = -1
    b.board[2, 1] = 1
    b.board[2, 2] = -1
    b.current_player = 1
    # 黑下 (1,,2) -> 白 (1,1) 无气被提；黑新子恰为单子且唯一气是 (1,1)，形成劫
    ok = b.play(1 * 9 + 2)
    assert ok is True
    assert b.board[1, 1] == 0, "白子应被提掉"
    assert b.ko_point == 1 * 9 + 1, f"应形成劫，ko_point={b.ko_point}"
    # 白若立即回提于 (1,1) -> 非法（劫）
    b.current_player = -1
    assert b.play(1 * 9 + 1) is False, "劫争回提应非法"
    # 白在别处落子后，劫解除
    b.play(8 * 9 + 8)
    assert b.ko_point == -1, "落子后应解除劫禁"
    print("PASS test_ko")


def test_pass_and_termination():
    """连续两次 pass 终局。"""
    b = GoBoard(9)
    assert b.is_game_over() is False
    b.play(-1)  # 黑 pass
    assert b.is_game_over() is False
    b.play(-1)  # 白 pass
    assert b.is_game_over() is True
    print("PASS test_pass_and_termination")


def test_score_komi():
    """计分正确性：黑占第一行、白占最后一行，中间中立；含贴目。"""
    b = GoBoard(9, komi=6.5)
    for c in range(9):
        b.board[0, c] = 1   # 黑 row0
        b.board[8, c] = -1  # 白 row8
    s = b.score()
    # 中间 7 排空，上下两色各邻 -> 中立不计；黑 9 - (白 9 + 贴目 6.5) = -6.5
    assert s == 9 - (9 + 6.5), f"score 应为 -6.5，实际 {s}"
    assert b.result() == -1
    print("PASS test_score_komi")


def test_random_selfplay_not_constant():
    """
    核心回归：1000 局随机对弈，黑胜率不得落在 [0.95, 1.0]。
    旧 GoGame 填满棋盘后黑恒胜 -> 黑胜率必然 ~1.0，此处必须打破。
    """
    wins = {1: 0, -1: 0, 0: 0}
    n = 9
    for _ in range(1000):
        b = GoBoard(n, komi=6.5)
        steps = 0
        max_steps = n * n * 2
        while not b.is_game_over() and steps < max_steps:
            legal = b.get_legal_moves()
            if legal.any():
                choices = np.where(legal)[0]
                mv = int(choices[random.randrange(len(choices))])
                b.play(mv)
            else:
                b.play(-1)
            steps += 1
        wins[b.result()] += 1
    black_win_rate = wins[1] / 1000
    print(f"随机对弈结果: 黑{wins[1]} 白{wins[-1]} 平{wins[0]} -> 黑胜率 {black_win_rate:.3f}")
    # 核心回归：旧 GoGame 填满棋盘后黑恒胜 -> 黑胜率 ~1.0。
    # 修复后规则有真实方差，黑胜率必须明显低于 0.95。
    assert not (0.95 <= black_win_rate <= 1.0), \
        f"黑胜率 {black_win_rate:.3f} 仍接近常数（旧 bug 未修！）"


def test_both_sides_win_random():
    """随机对弈中黑白都必须出现过胜利（信号存在的基本体现）。"""
    wins = {1: 0, -1: 0}
    n = 9
    for _ in range(300):
        b = GoBoard(n, komi=6.5)
        steps = 0
        while not b.is_game_over() and steps < n * n * 2:
            legal = b.get_legal_moves()
            if legal.any():
                choices = np.where(legal)[0]
                b.play(int(choices[random.randrange(len(choices))]))
            else:
                b.play(-1)
            steps += 1
        r = b.result()
        if r != 0:
            wins[r] += 1
    assert wins[1] > 0 and wins[-1] > 0, f"随机对弈应两性都赢，实际 {wins}"
    print("PASS test_both_sides_win_random")


if __name__ == "__main__":
    test_basic_capture()
    test_suicide_illegal()
    test_ko()
    test_pass_and_termination()
    test_score_komi()
    test_both_sides_win_random()
    test_random_selfplay_not_constant()
    print("\nALL TESTS PASSED")
