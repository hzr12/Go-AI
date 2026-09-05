"""临时 smoke：CliGame 逻辑（mock ai）。用后删。"""
import sys
import numpy as np
sys.path.insert(0, '.')


class MockAI:
    def __init__(self, bs):
        self.board_size = bs

    def predict_batch(self, states):
        B = len(states)
        A = self.board_size ** 2 + 1
        pol = np.zeros((B, A), dtype=np.float32)
        val = np.zeros(B, dtype=np.float32)
        for i, st in enumerate(states):
            planes = st[4]
            p = np.abs(planes[0]).reshape(-1) + 0.01
            pol[i, :len(p)] = p
            pol[i, -1] = 0.05
            pol[i] /= pol[i].sum()
            val[i] = float(planes[9, 0, 0]) * 0.1
        return pol, val


from scripts.cli_play import CliGame, PASS

g = CliGame(MockAI(9), 9, simulations=24, num_threads=2,
            use_rollout=False, rollout_lambda=0.25, human_color=1, topk=5)

g._push_undo()
assert g._apply(40), "human apply"
print("human move ok, last:", g.last_move)

is_pass = g.ai_turn()
print("ai turn ok, is_pass:", is_pass)

assert g.undo(), "undo"
print("undo ok, move_count:", g.move_count, "path len:", len(g.path_moves))
assert g.move_count == 1 and len(g.path_moves) == 1

g.ai_turn()
assert g.move_count == 2
print("ai re-turn ok, move_count:", g.move_count)

assert not g._apply(40), "occupied apply should fail"
print("occupied reject ok")

g._push_undo()
assert g._apply(PASS)
assert not g.over()
g._push_undo()
assert g._apply(PASS)
assert g.over()
print("game over ok, score:", g.board.score())
print("CLI_SMOKE_OK")
