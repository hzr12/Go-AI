"""
围棋 AI 推理引擎（监督学习 / SFT 模型）

兼容 src/networks.alphanet.AlphaGoNet 输出的 (policy, value)：
    - policy: 棋盘上每点 + 虚着(pass) 的概率分布
    - value : 当前执子方视角的局面胜率，tanh 后落在 [-1, 1]

特征由 src.game.go_rules.GoBoard.feature_planes 统一生成（12 通道），
与 src.data.dataset 使用同一套特征工程，避免训练/推理不一致。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import numpy as np
import torch

from src.game.go_rules import GoBoard
from src.networks.alphanet import AlphaGoNet


class GoAI:
    """基于 SFT 模型的围棋对弈 / 分析引擎。

    用法示例::

        ai = GoAI(model_path="models/sft_19x19.pth", board_size=19, device="cuda")
        # 自对弈一局
        ai.self_play(verbose=True, temperature=0.8)
        # 人机对弈（人类执黑先手）
        ai.play_against_human(human_color=1)
        # 分析某个局面的 top-k 候选着法
        ai.analyze()
    """

    def __init__(self, model_path=None, board_size=19, device="auto", use_amp=False,
                 backbone_channels=128, backbone_res_blocks=12, policy_channels=64, value_channels=32,
                 attention_mode="mix", num_attention_layers=4, num_heads=4, attention_dropout=0.0):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_amp = use_amp and self.device.startswith("cuda")
        self.board_size = board_size

        self.model = AlphaGoNet(
            in_channels=12,
            backbone_channels=backbone_channels,
            backbone_res_blocks=backbone_res_blocks,
            attention_mode=attention_mode,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            policy_channels=policy_channels,
            value_channels=value_channels,
            action_size=board_size * board_size + 1,  # +1 = 虚着
        ).to(self.device)
        self.model.eval()

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            # 兼容直接保存的 state_dict 或 {"model": state_dict}
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
            print(f"[GoAI] 已加载模型: {model_path}  ({device})")
        else:
            print(f"[GoAI] 未加载权重（随机初始化），仅用于流程验证。device={device}")

    # ------------------------------------------------------------------ #
    # 特征构造
    # ------------------------------------------------------------------ #
    def _build_state(self, board, my_hist, op_hist, to_play):
        """用统一的 feature_planes 构造 12 通道状态张量。"""
        planes = board.feature_planes(my_hist, op_hist, to_play=to_play)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device).float()
        return x

    # ------------------------------------------------------------------ #
    # 核心：模型前向 + 采样
    # ------------------------------------------------------------------ #
    def predict(self, board, my_hist, op_hist, to_play):
        """返回 (policy_np, value)。

        policy_np: shape=(bs*bs+1,) 概率（已 softmax）
        value    : float, 当前 to_play 视角 [-1,1]
        """
        x = self._build_state(board, my_hist, op_hist, to_play)
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    policy_logits, value = self.model(x)
            else:
                policy_logits, value = self.model(x)
        policy = torch.softmax(policy_logits.squeeze(0), dim=-1).cpu().numpy()
        value = float(value.squeeze().item())
        return policy, value

    def choose_move(self, board, my_hist, op_hist, to_play, legal_mask, temperature=1.0, topk=10):
        """根据策略分布与合法着法掩码，采样一个着法。

        返回 (move_int, is_pass, value)，move_int 为 board_size*board_size 表示虚着。
        """
        policy, value = self.predict(board, my_hist, op_hist, to_play)
        bs = self.board_size
        n_actions = bs * bs + 1

        illegal = np.ones(n_actions, dtype=bool)
        for m in legal_mask:
            illegal[m] = False
        # 始终允许虚着
        illegal[n_actions - 1] = False
        masked = policy.copy()
        masked[illegal] = 0.0
        s = masked.sum()
        if s <= 0:
            return n_actions - 1, True, value

        if temperature <= 0:
            # 贪心
            move_int = int(np.argmax(masked))
        else:
            probs = masked / s
            # 可选 top-k 截断，降低随机性
            if topk and topk < len(probs):
                idx = np.argsort(probs)[::-1][:topk]
                p2 = np.zeros_like(probs)
                p2[idx] = probs[idx]
                probs = p2 / p2.sum()
            # 温度缩放：对 log 缩放后重新 softmax
            if temperature != 1.0:
                logp = np.log(probs + 1e-12)
                probs = np.exp(logp / max(temperature, 1e-3))
                probs = probs / probs.sum()
            move_int = int(np.random.choice(len(probs), p=probs))

        is_pass = (move_int == n_actions - 1)
        return move_int, is_pass, value

    # ------------------------------------------------------------------ #
    # 对局循环
    # ------------------------------------------------------------------ #
    def _move_int_to_coord(self, move_int):
        bs = self.board_size
        if move_int == bs * bs:
            return None  # pass
        r, c = divmod(move_int, bs)
        return (r, c)

    def self_play(self, num_games=1, max_moves=400, temperature=1.0, topk=10, verbose=False):
        """模型自我对弈 num_games 局，返回每局结果（黑方视角 +1/-1）。"""
        results = []
        for g in range(num_games):
            board = GoBoard(self.board_size)
            my_hist = [[-1, -1, -1], [-1, -1, -1]]  # 对当前执子方：最近3手
            passes = 0
            move_count = 0
            while passes < 2 and move_count < max_moves:
                to_play = board.current_player
                legal = board.get_legal_moves()
                if len(legal) == 0:
                    passes += 1
                    board.play(-1)
                    move_count += 1
                    continue
                move_int, is_pass, value = self.choose_move(
                    board, my_hist[0], my_hist[1], to_play, legal,
                    temperature=temperature, topk=topk)
                if is_pass:
                    board.play(-1)
                    passes += 1
                else:
                    r, c = self._move_int_to_coord(move_int)
                    board.play(r * self.board_size + c)
                    passes = 0
                    # 更新历史（最近3手，最新在末尾）
                    hist = my_hist[0] if to_play == 1 else my_hist[1]
                    hist.pop(0)
                    hist.append(r * self.board_size + c)
                move_count += 1
                if verbose and (move_count % 10 == 0):
                    print(f"  game {g} move {move_count} to_play={to_play} "
                          f"value={value:+.3f} pass={is_pass}")
            score = board.score()
            # 黑方(1)视角
            result = 1.0 if score > 0 else -1.0
            results.append(result)
            if verbose:
                print(f"game {g} finished: score(黑-白)={score:+.1f} -> "
                      f"{'黑胜' if result > 0 else '白胜'}")
        return results

    def play_against_human(self, human_color=1, max_moves=400, temperature=0.6):
        """人机对弈，人类通过终端输入坐标（如 'ce' 或 'pass'）。"""
        board = GoBoard(self.board_size)
        my_hist = [[-1, -1, -1], [-1, -1, -1]]
        passes = 0
        move_count = 0
        while passes < 2 and move_count < max_moves:
            to_play = board.current_player
            legal = board.get_legal_moves()
            if to_play == human_color:
                print(board.to_string())
                print(f"合法着法数: {len(legal)}  | 输入坐标(如 ce)，或 pass，或 resign")
                inp = input("你的着法: ").strip().lower()
                if inp in ("pass", ""):
                    board.play(-1)
                    passes += 1
                elif inp in ("resign", "quit"):
                    print("你认输。")
                    break
                else:
                    ok, mv = board.parse_move_str(inp, human_color)
                    if not ok or mv not in legal:
                        print("非法着法，请重试。")
                        continue
                    board.play(mv)
                    passes = 0
                    hist = my_hist[0] if human_color == 1 else my_hist[1]
                    hist.pop(0); hist.append(mv)
            else:
                if len(legal) == 0:
                    board.play(-1); passes += 1
                else:
                    move_int, is_pass, value = self.choose_move(
                        board, my_hist[0], my_hist[1], to_play, legal,
                        temperature=temperature)
                    if is_pass:
                        board.play(-1); passes += 1
                        print(f"AI 虚着 (value={value:+.3f})")
                    else:
                        r, c = self._move_int_to_coord(move_int)
                        mv = r * self.board_size + c
                        board.play(mv); passes = 0
                        hist = my_hist[0] if to_play == 1 else my_hist[1]
                        hist.pop(0); hist.append(mv)
                        print(f"AI 落子 {chr(ord('a')+c)}{chr(ord('a')+r)} (value={value:+.3f})")
            move_count += 1
        print(board.to_string())
        score = board.score()
        print(f"终局 score(黑-白)={score:+.1f} -> {'黑胜' if score > 0 else '白胜'}")

    def analyze(self, max_moves=400, temperature=0.0):
        """交互式分析：从当前局面出发，展示 AI 的 top-k 候选着法。"""
        board = GoBoard(self.board_size)
        my_hist = [[-1, -1, -1], [-1, -1, -1]]
        print("逐步分析（输入坐标落子，pass 虚着，auto 让 AI 走，q 退出）")
        while True:
            to_play = board.current_player
            print(board.to_string())
            legal = board.get_legal_moves()
            policy, value = self.predict(board, my_hist[0], my_hist[1], to_play)
            bs = self.board_size
            ranked = []
            for m in legal:
                ranked.append((m, float(policy[m])))
            ranked.append((bs * bs, float(policy[bs * bs])))  # pass
            ranked.sort(key=lambda t: t[1], reverse=True)
            print(f"当前 to_play={to_play}  value={value:+.3f}")
            for m, p in ranked[:8]:
                if m == bs * bs:
                    print(f"  pass        p={p:.3f}")
                else:
                    r, c = divmod(m, bs)
                    print(f"  {chr(ord('a')+c)}{chr(ord('a')+r)} (idx {m:3d})  p={p:.3f}")
            inp = input("> ").strip().lower()
            if inp in ("q", "quit"):
                break
            if inp == "auto":
                mv = ranked[0][0]
                if mv == bs * bs:
                    board.play(-1)
                else:
                    board.play(mv)
                    hist = my_hist[0] if to_play == 1 else my_hist[1]
                    hist.pop(0); hist.append(mv)
                continue
            if inp in ("pass", ""):
                board.play(-1); continue
            ok, mv = board.parse_move_str(inp, to_play)
            if not ok or int(mv) not in legal.tolist():
                print("非法，重试。"); continue
            board.play(mv)
            hist = my_hist[0] if to_play == 1 else my_hist[1]
            hist.pop(0); hist.append(mv)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="围棋 SFT 模型推理 / 对弈")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--mode", type=str, default="selfplay",
                        choices=["selfplay", "human", "analyze"])
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--human-color", type=int, default=1)
    parser.add_argument("--attention-mode", default="mix", choices=["none", "mix", "all"])
    parser.add_argument("--num-attention-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    args = parser.parse_args()

    ai = GoAI(model_path=args.model, board_size=args.board_size,
              device=args.device, use_amp=args.use_amp,
              attention_mode=args.attention_mode,
              num_attention_layers=args.num_attention_layers,
              num_heads=args.num_heads,
              attention_dropout=args.attention_dropout)
    if args.mode == "selfplay":
        res = ai.self_play(num_games=args.games, temperature=args.temperature, topk=args.topk)
        wr = sum(1 for r in res if r > 0) / max(len(res), 1)
        print(f"自对弈 {len(res)} 局，黑方胜率 {wr:.2%}")
    elif args.mode == "human":
        ai.play_against_human(human_color=args.human_color, temperature=args.temperature)
    elif args.mode == "analyze":
        ai.analyze(temperature=args.temperature)


if __name__ == "__main__":
    main()
