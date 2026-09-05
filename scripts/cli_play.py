"""
CLI 人机对弈：19 路终端棋盘 + MCTS 实时搜索信息（非 WebUI 版）。

启动:
    python scripts/cli_play.py --model models/sft_19x19_v3.pth --device cuda
    python scripts/cli_play.py --model ... --human-color white   # AI 执黑先行

对局中输入:
    ce / jj ...   坐标落子（字母=列 a-s，字母=行 a-s，如 webui）
    pass          虚着
    undo          悔一手（撤销最近一步，含 AI 步）
    resign        认输退出

每步 AI 搜索后打印: top 候选表（着法/visits/prior/AI 胜率）、根价值、耗时、
吞吐，并在棋盘上以数字 1-5 标出访问次数最高的候选点。
"""

import sys
import os
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import GoAI
from src.search.mcts import MCTS
from src.game.go_rules import GoBoard

PASS = -1


class CliGame:
    def __init__(self, ai, board_size, simulations, num_threads,
                 use_rollout, rollout_lambda, human_color, topk):
        self.ai = ai
        self.size = board_size
        self.sims = simulations
        self.topk = topk
        self.human_color = human_color
        self.board = GoBoard(board_size)
        self.hist = [[-1, -1, -1], [-1, -1, -1]]   # [黑方最近3手, 白方最近3手]
        self.path_moves = []                        # MCTS 树复用
        self.last_move = None
        self.move_count = 0
        self.candidates = []
        self.undo_stack = []                        # 悔棋快照
        self.mcts = MCTS(ai, board_size=board_size, num_threads=num_threads,
                         use_rollout=use_rollout, rollout_lambda=rollout_lambda)

    # ---- 视图 ---------------------------------------------------------------

    def _coord_str(self, mv):
        if mv == PASS:
            return "pass"
        r, c = divmod(mv, self.size)
        return f"{chr(ord('a') + c)}{chr(ord('a') + r)}"

    def print_board(self, markers=None):
        lm = self.last_move
        print(self.board.to_string(markers=markers, last_move=lm))
        to_play = self.board.current_player
        who = "人类" if to_play == self.human_color else "AI"
        print(f"手数 {self.move_count} | 轮到 {'黑' if to_play == 1 else '白'}({who}) "
              f"| 连续pass {self.board.passes}")

    def print_candidates(self):
        if not self.candidates:
            return
        print(f"  {'着法':<6}{'visits':>8}{'prior':>9}{'AI胜率':>9}")
        for cd in self.candidates:
            mv_str = "pass" if cd["mv"] == PASS else self._coord_str(cd["mv"])
            print(f"  {mv_str:<6}{cd['visits']:>8}{cd['prior'] * 100:>8.1f}%"
                  f"{cd['ai_winrate'] * 100:>8.0f}%")

    # ---- 动作 ---------------------------------------------------------------

    def _push_undo(self):
        self.undo_stack.append((
            [list(h) for h in self.hist],
            len(self.path_moves),
            self.move_count,
            self.last_move,
            len(self.candidates),
        ))

    def undo(self):
        if not self.undo_stack:
            print("  无法悔棋（没有可撤销的步）")
            return False
        hist, plen, mc, lm, clen = self.undo_stack.pop()
        if not self.board.undo():
            print("  悔棋失败（棋盘栈空）")
            return False
        self.hist = hist
        del self.path_moves[plen:]
        self.move_count = mc
        self.last_move = lm
        del self.candidates[clen:]
        return True

    def _apply(self, mv):
        ok = self.board.play(mv)
        if not ok:
            return False
        self.path_moves.append(mv)
        to_play_before = -self.board.current_player
        if mv >= 0:
            h = self.hist[0] if to_play_before == 1 else self.hist[1]
            h.pop(0)
            h.append(mv)
            self.last_move = (mv // self.size, mv % self.size)
        else:
            self.last_move = "pass"
        self.move_count += 1
        return True

    def ai_turn(self):
        to_play = self.board.current_player
        t0 = time.perf_counter()
        visits, probs, root_value = self.mcts.search(
            self.board, self.hist[0], self.hist[1], to_play,
            simulations=self.sims, path_moves=self.path_moves)
        elapsed = time.perf_counter() - t0

        move_int = int(np.argmax(visits)) if visits.sum() > 0 else len(visits) - 1
        root = self.mcts._prev_root

        # top 候选
        cands = []
        if root is not None:
            for mv, child in root.children.items():
                ai_winrate = 0.5 - child.q() / 2.0   # child 视角=人类 → 取反
                cands.append({"mv": mv, "visits": int(child.visit),
                              "prior": float(child.prior),
                              "ai_winrate": ai_winrate})
            cands.sort(key=lambda x: -x["visits"])
        self.candidates = cands[:self.topk]

        # 棋盘标记：visits 前 5 名（数字 1-5）
        markers = {}
        for rank, cd in enumerate(cands[:5], 1):
            if cd["mv"] == PASS:
                continue
            r, c = divmod(cd["mv"], self.size)
            markers[(r, c)] = str(rank)

        is_pass = (move_int == len(visits) - 1)
        self._push_undo()
        self._apply(PASS if is_pass else move_int)

        ai_winrate = (root_value + 1) / 2.0   # to_play（AI）视角
        print(f"\n[AI] {self._coord_str(PASS if is_pass else move_int)} | "
              f"visits {int(visits.sum())}/{self.sims} | "
              f"AI胜率 {ai_winrate * 100:.1f}% | {elapsed:.2f}s "
              f"({int(self.sims / max(elapsed, 1e-6))} sims/s)")
        self.print_candidates()
        self.print_board(markers=markers)
        return is_pass

    def over(self):
        return self.board.passes >= 2


def main():
    ap = argparse.ArgumentParser(description="Go-AI CLI 人机对弈（MCTS 搜索信息可视化）")
    ap.add_argument("--model", default="models/sft_19x19_v3.pth")
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--simulations", type=int, default=400)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--human-color", default="black", choices=["black", "white"])
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0=贪心取访问最高；>0 按访问分布采样")
    ap.add_argument("--use-rollout", action="store_true")
    ap.add_argument("--rollout-lambda", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=5, help="候选表显示条数")
    args = ap.parse_args()

    model_path = args.model
    if not os.path.isfile(model_path):
        print(f"[warn] 模型 {model_path} 不存在，使用随机权重（仅验证流程）")
        model_path = None
    ai = GoAI(model_path=model_path, board_size=args.board_size, device=args.device,
              use_amp=True, attn_mode="window", attn_window=7,
              channels_last=(args.device.split(":")[0] == "cuda"))

    human_color = 1 if args.human_color == "black" else -1
    game = CliGame(ai, args.board_size, args.simulations, args.num_threads,
                   args.use_rollout, args.rollout_lambda, human_color, args.topk)

    print(f"Go-AI CLI | 模型={args.model if model_path else '随机权重'} | "
          f"设备={ai.device} | sims={args.simulations} | 你执{'黑(先手)' if human_color == 1 else '白'}")
    print("输入: 坐标(如 ce) / pass / undo / resign")

    if human_color == -1:
        game.print_board()
        game.ai_turn()

    while not game.over():
        game.print_board()
        try:
            inp = input("你的着法: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            return
        if inp in ("resign", "quit", "exit"):
            print("你认输。")
            return
        if inp == "undo":
            if game.undo():
                game.print_board()
            continue
        if inp in ("pass", ""):
            mv = PASS
        else:
            ok, mv = game.board.parse_move_str(inp, game.board.current_player)
            if not ok or (mv != PASS and mv not in np.where(game.board.get_legal_moves())[0]):
                print("  非法着法，请重试")
                continue
        game._push_undo()
        if not game._apply(mv):
            print("  落子失败")
            game.undo_stack.pop()
            continue
        if game.over():
            break
        game.ai_turn()

    score = game.board.score()
    human_res = "胜" if (score > 0) == (human_color == 1) else "负"
    print(f"\n对局结束：黑-白 = {score:+.1f}（贴目 {game.board.komi}）→ 你{human_res}")


if __name__ == "__main__":
    main()
