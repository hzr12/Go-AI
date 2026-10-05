# -*- coding: utf-8 -*-
"""webui 走 KataGo 引擎后端（GTP）时的行为契约。

为什么需要这条路径：V7 是 **22 通道** `NbtTfNet`（`src/networks/katago_v7.py`），
而原生 MCTS 那条路的特征是 `go_rules.py::feature_planes` 造的，`_check_n_channels`
硬性只允许 12..17。ch14–17 是梯子、ch18 要贴目，`feature_planes` 根本造不出来。
所以 V7 上 WebUI 只能走后端引擎 —— 引擎按 `.bin.gz` 的结构描述自己造全部 22 通道。

本文件锁三件事：
  1. **AI 着法来自引擎**，且棋盘状态被正确同步（含悔棋后的全量重放）；
  2. **引擎拿不到的量一律留空** —— `visits` / `ai_winrate` / `candidates`。
     这是本文件最重要的断言：普通 GTP 没有这些（`kata-analyze` 才有），填个
     0.5 看起来毫无异常，但那是在 UI 上摆一个**编造的**数字，比报错坏得多；
  3. **引擎不可用时响亮失败**，绝不静默退化成随机走子 —— 后者会让用户以为在
     对弈，实际是在跟随机数下棋。
"""
import ast
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.webui import Session
from src.engine.gtp_client import GTPError

N = 9
REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEBUI_SRC = (REPO / "scripts" / "webui.py").read_text(encoding="utf-8")
PASS = N * N          # GTP 的 pass 槽 = 扁平下标 81


class _FakeEngine:
    """记下 `set_position` 收到什么，并按预设序列回着法。"""

    def __init__(self, replies=None, raises=None):
        self.replies = list(replies or [])
        self.raises = raises
        self.seen = []          # 每次 set_position 的 (moves, to_move)
        self.closed = False

    def set_position(self, moves, to_move='b'):
        self.seen.append((list(moves), to_move))
        if self.raises is not None:
            raise self.raises

    def genmove(self, color):
        if not self.replies:
            raise AssertionError("genmove 次数超出预设")
        return self.replies.pop(0)

    def close(self):
        self.closed = True


def _session(engine, human_color=-1):
    """`ai=None`：引擎后端下不建 torch 模型（这正是要验的行为之一）。

    `human_color=-1`（人类执白）⇒ 轮到黑棋时 `ai_move` 才不会被
    「当前轮到人类」挡掉。要测「人类先手」就传 `human_color=1` 并让人类先落。
    """
    s = Session(None, board_size=N, num_threads=1, engine=engine)
    s.human_color = human_color
    return s


# --------------------------------------------------------------------------- #
# 1. 着法来自引擎
# --------------------------------------------------------------------------- #
def test_engine_move_is_applied_to_the_board():
    eng = _FakeEngine(replies=[40])
    s = _session(eng)
    st = s.ai_move(simulations=32)
    assert eng.seen, "AI 落子前必须先与引擎同步棋盘"
    assert s.board.move_history == [40], "引擎给的着法应落到棋盘上"
    assert st["last_move"] is not None


def test_engine_is_synced_with_the_whole_history_in_order():
    eng = _FakeEngine(replies=[41])
    s = _session(eng, human_color=1)   # 人类执黑先手
    s._apply_move(0, "human")
    s.ai_move(simulations=32)
    moves, to_move = eng.seen[-1]
    # 全量重放：颜色按 b/w 交替，且 to_move 落在最后一个着子之后
    assert [c for c, _ in moves] == ["b", "w"][:len(moves)]
    assert all(isinstance(v, int) for _, v in moves)
    assert to_move == ("b" if len(moves) % 2 == 0 else "w"), \
        "set_position 的 to_move 必须落在最后一个着子之后"


def test_human_move_is_synced_before_the_ai_replies():
    """人类先手时，AI 请求前引擎必须已经看到人类那一手。"""
    eng = _FakeEngine(replies=[41])
    s = _session(eng, human_color=1)
    s._apply_move(0, "human")
    s.ai_move(simulations=32)
    moves, _ = eng.seen[-1]
    assert moves == [("b", 0)], f"引擎看到的应是人类那一手，实际 {moves}"


def test_pass_from_engine_is_recorded_as_a_pass():
    """引擎的 pass 槽 81 必须翻译成 webui 的 -1 约定。

    两套约定不同：GTP 用扁平下标 `N*N`（=81）表示 pass，`_apply_move` 判 pass
    的标准是 `mv < 0`。把 81 原样传下去，`board.play(81)` 会当成越界着点判非法
    **静默丢弃** —— 引擎叫了 pass，棋盘上却什么都没发生。
    """
    eng = _FakeEngine(replies=[PASS])
    s = _session(eng)
    st = s.ai_move(simulations=32)
    assert s.board.move_history == [-1], \
        f"pass 应存成 -1，实际 {s.board.move_history}（若为 [] 说明 81 被当非法着点丢了）"
    assert st["last_move"] == "pass"
    assert s.board.passes == 1


def test_resign_from_engine_ends_the_game():
    eng = _FakeEngine(replies=[-1])
    s = _session(eng)
    st = s.ai_move(simulations=32)
    assert s.game_over, "引擎认输应结束对局"
    assert st["ai_info"]["move"] == "resign"


def test_pass_round_trips_back_to_the_engine_as_gtp_pass():
    """AI pass 之后，下一次同步必须发 `pass`，**不能**把 -1 当成坐标发出去。

    这是最容易出事的一处：`-1` 直接进 `vertex_to_str` 会经 `divmod` 变成
    `S20` 这种**看着合法、实际越界**的记号（`vertex_to_str` 已挡负数，但这里的
    价值在于证明整条往返链真的对）。
    """
    eng = _FakeEngine(replies=[PASS, 44])
    s = _session(eng, human_color=1)
    s._apply_move(0, "human")
    s.ai_move(simulations=32)          # AI pass
    # 人类不能也 pass：连续两 pass 直接终局，第二次 ai_move 会被
    # 「对局已结束」挡掉，就验证不到往返了。
    s._apply_move(5, "human")
    s.ai_move(simulations=32)          # AI 再走一手
    moves, _ = eng.seen[-1]
    assert moves == [("b", 0), ("w", -1), ("b", 5)], \
        f"pass 要以 -1 交给 GTP 客户端翻译成 'pass'，实际 {moves}"


# --------------------------------------------------------------------------- #
# 2. 引擎拿不到的量一律留空（核心断言）
# --------------------------------------------------------------------------- #
def test_unknown_quantities_are_left_empty_not_faked():
    eng = _FakeEngine(replies=[42])
    s = _session(eng)
    info = s.ai_move(simulations=32)["ai_info"]
    for key in ("visits", "simulations", "ai_winrate", "sps"):
        assert info[key] is None, \
            f"{key} 在普通 GTP 下拿不到，必须留 None；填 0 / 0.5 是在 UI 上" \
            f"编造一个看起来正常的数字"
    assert s.candidates == [], "引擎后端没有候选着法（kata-analyze 才有）"
    assert s.analysis is None, "候选为空就不该有可展示的分析"
    assert s.wr_hist == [], "没有胜率就不该往胜率曲线里塞点（会画出一条假平线）"


def test_engine_backend_is_labelled_in_the_info():
    eng = _FakeEngine(replies=[43])
    s = _session(eng)
    info = s.ai_move(simulations=32)["ai_info"]
    assert info["mode"] == "engine"
    assert info["backend"] == "katago-gtp", \
        "前端要能看出这一手是引擎出的，不是原生 MCTS"


# --------------------------------------------------------------------------- #
# 3. 引擎不可用 → 响亮失败
# --------------------------------------------------------------------------- #
def test_engine_failure_is_reported_loudly_instead_of_a_random_move():
    eng = _FakeEngine(raises=GTPError('引擎已退出（returncode=1）'))
    s = _session(eng)
    out = s.ai_move(simulations=32)
    assert "error" in out, "引擎挂了必须报错，不能装作正常"
    assert "引擎不可用" in out["error"]
    assert s.board.move_history == [], "**绝不能**在引擎失败时偷偷下一手随机棋"


# --------------------------------------------------------------------------- #
# 4. 装配：引擎模式下不建 GoAI，rollout 被显式拒绝
# --------------------------------------------------------------------------- #
def test_main_does_not_build_goai_in_engine_mode():
    """`main()` 里 GoAI 构造必须落在「引擎模式」的 else 分支。

    静态锁：V7 权重塞进 `GoAI` 只会撞 `_infer_in_channels` 的 12..17 白名单。
    引擎后端不需要 torch 模型，建它纯属给自己找一个无关的报错。
    """
    tree = ast.parse(WEBUI_SRC)
    main_fn = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "main")

    def _builds_goai(nodes):
        return any(isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Call)
                           and getattr(t.func, "id", "") == "GoAI"
                           for t in ast.walk(n))
                   for n in nodes)

    goai_ifs = [n for n in main_fn.body
                if isinstance(n, ast.If) and _builds_goai(n.orelse)
                and "engine" in ast.dump(n.test)]
    assert goai_ifs, (
        "GoAI 构造必须挂在 `if engine is not None: ... else: <GoAI(...)>` 的 "
        "else 分支上；否则引擎模式下仍会建 torch 模型，V7 会撞 12..17 白名单报错")
    # 反向检查：GoAI 不许在「没有 else」的位置被建（即不许无条件构造）
    assert not _builds_goai(main_fn.body), \
        "GoAI(...) 出现在 main() 顶层无条件路径上"


def test_use_rollout_is_rejected_in_engine_mode():
    assert "--use-rollout 与 --engine-gtp 互斥" in WEBUI_SRC, \
        "引擎搜索在 KataGo 内部，webui 侧再开 rollout 是静默无效加速，必须拒绝"


@pytest.mark.parametrize("flag", ["--engine-gtp", "--engine-gtp-exe",
                                  "--engine-gtp-config", "--engine-komi"])
def test_engine_flags_exist(flag):
    assert flag in WEBUI_SRC, f"CLI 缺少 {flag}"
