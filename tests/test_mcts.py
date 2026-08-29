"""MCTS 搜索与批量前向的单元测试（CPU，无需 GPU）。"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import GoAI
from src.search.mcts import MCTS
from src.game.go_rules import GoBoard


def _make_ai(board_size=9):
    return GoAI(board_size=board_size, device="cpu", compile=False)


def test_predict_batch_shape():
    ai = _make_ai(9)
    board = GoBoard(9)
    hist = [[-1, -1, -3], [-1, -1, -3]]
    states = [(board, list(hist[0]), list(hist[1]), 1) for _ in range(4)]
    policies, values = ai.predict_batch(states)
    assert policies.shape == (4, 9 * 9 + 1)
    assert values.shape == (4,)
    # 策略应接近概率分布
    assert np.allclose(policies.sum(axis=1), 1.0, atol=1e-3)


def test_mcts_runs_and_returns_valid_move():
    ai = _make_ai(9)
    board = GoBoard(9)
    hist = [[-1, -1, -3], [-1, -1, -3]]
    mcts = MCTS(ai, board_size=9, num_threads=2, temperature=0.0)
    move_int, is_pass, value = mcts.best_move(
        board, hist[0], hist[1], to_play=1, simulations=50, return_value=True)
    assert isinstance(move_int, int)
    assert 0 <= move_int <= 9 * 9  # 含虚着
    assert -1.0 <= value <= 1.0


def test_mcts_visits_sum_positive():
    ai = _make_ai(9)
    board = GoBoard(9)
    hist = [[-1, -1, -3], [-1, -1, -3]]
    mcts = MCTS(ai, board_size=9, num_threads=2)
    visits, probs, root_value = mcts.search(
        board, hist[0], hist[1], to_play=1, simulations=50)
    assert visits.sum() > 0
    assert probs.shape == (9 * 9 + 1,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_mcts_prefers_capture_over_random():
    """构造一个能吃子的简单局面，验证 MCTS 选了合法着法且不崩。"""
    ai = _make_ai(9)
    board = GoBoard(9)
    # 摆一个黑棋能吃白子的形状（简化：仅验证流程稳定）
    board.board[2][2] = 1   # 黑
    board.board[2][3] = 2   # 白，气被黑包围
    board.board[1][3] = 1
    board.board[3][3] = 1
    board.board[2][4] = 1
    hist = [[-1, -1, -3], [-1, -1, -3]]
    mcts = MCTS(ai, board_size=9, num_threads=2, temperature=0.0)
    move_int, is_pass = mcts.best_move(
        board, hist[0], hist[1], to_play=1, simulations=30)
    legal = board.get_legal_moves()
    assert move_int in legal or move_int == 9 * 9


if __name__ == "__main__":
    for fn in [test_predict_batch_shape, test_mcts_runs_and_returns_valid_move,
               test_mcts_visits_sum_positive, test_mcts_prefers_capture_over_random]:
        fn()
        print(f"PASS {fn.__name__}")
