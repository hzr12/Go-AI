"""webui 打开 `use_rollout` 时交给 FastPolicy 的通道数必须跟随模型（P4.8b）。

背景：P4.13b（commit 2f48994）把搜索侧的特征通道数改成向模型要
（`MCTS._in_channels()`：读 `ai.in_channels`，**缺失即报错，绝不回退 12**），
并留下一个未修的尾巴：`scripts/webui.py` 的 `main()` 在 `Session` 建好
**之后**才把 `use_rollout` 打开（此时 `MCTS.__init__` 早过去了，它在
`use_rollout=False` 时刻意不造 FastPolicy），于是自己重建了一个
`FastPolicy(args.board_size)`——`n_channels` 掉回签名默认值 12。

今天无害：那个实例 `weights=None`，而 `FastPolicy.logits` 只在
`weights is not None` 时才读 `n_channels`。但它是**潜伏的**静默错配源：
谁一旦在这里挂上权重，P4.13b 刚拆掉的 12-vs-17 就原样回来。

本文件锁三件事：
  1. AST 级：`webui.py` 里**每一个** `FastPolicy(` 调用点都显式传
     `n_channels`，且那个值不是写死的 12、不是 `getattr(..., 12)`；
  2. 行为级：把 `main()` 里那段装配代码**原样 exec 一遍**（不是手抄一份），
     17 通道模型 → 重建出来的 `FastPolicy.n_channels == 17`；
     12 通道模型 → 12（零回归）；
  3. `MCTS.n_channels` 是只读 property，且仍然「问不到就报错」——
     有了它 webui 才不必自己再抄一遍 getattr，也就不必自己发明回退值。
"""
import ast
import os
import pathlib
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.webui import Session
from src.search.mcts import MCTS

N = 9
REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEBUI_SRC = (REPO / "scripts" / "webui.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
class _SpyAI:
    """只声明 `in_channels` 的替身：MCTS 唯一会从模型侧读的通道属性。

    `predict_batch` 永不真跑（本文件只关心「按几个通道造出来」），但形状
    摆对，免得哪天有人在用例里顺手调一次 search 就炸在无关的地方。
    """

    def __init__(self, in_channels, board_size=N):
        self.in_channels = int(in_channels)
        self.n_actions = board_size * board_size + 1
        self.seen = []

    def predict_batch(self, states):
        planes = np.stack([np.asarray(s[4]) for s in states])
        self.seen.append(int(planes.shape[1]))
        if planes.shape[1] != self.in_channels:
            raise RuntimeError(
                f"预计算特征通道数不匹配：期望 {self.in_channels}，实际 {planes.shape[1]}")
        b = planes.shape[0]
        return (np.full((b, self.n_actions), 1.0 / self.n_actions, dtype=np.float32),
                np.zeros((b,), dtype=np.float32))


def _main_fn():
    tree = ast.parse(WEBUI_SRC)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert fn is not None, "webui.py 里找不到 main()"
    return fn


def _rollout_setup_block():
    """`main()` 顶层那个 `if args.use_rollout:` 语句节点（装配 FastPolicy 的那段）。"""
    for node in _main_fn().body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                and node.test.attr == "use_rollout"):
            return node
    pytest.fail("main() 顶层找不到 `if args.use_rollout:` —— rollout 装配段被挪走"
                "或改写了，本用例必须跟着改（不许静默放过）")


def _run_rollout_setup(ai, *, use_rollout=True, board_size=N):
    """原样跑一遍 `main()` 里那段 rollout 装配代码，返回 Session。

    刻意**执行源码**而不是在测试里复刻一遍 `FastPolicy(...)`：复刻的那份
    不会跟着 webui 一起变，于是回归能全绿通过；这里跑的是文件里真实的那个
    `ast.If` 节点，webui 改成什么样就跑什么样（改得解析不了就响亮地红）。
    """
    session = Session(ai, board_size=board_size, num_threads=1)
    assert session.mcts._fast_policy is None, \
        "夹具前提：Session 建 MCTS 时没开 rollout（use_rollout=True 才是本任务的路径）"
    args = types.SimpleNamespace(use_rollout=use_rollout, board_size=board_size)
    block = _rollout_setup_block()
    mod = ast.Module(body=[block], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "webui.py::main()", "exec"), {"args": args, "session": session})
    return session


# --------------------------------------------------------------------------- #
# 1. AST：每个 FastPolicy 调用点都显式传通道数，且没有回退默认值
# --------------------------------------------------------------------------- #
def _fastpolicy_calls(node):
    """webui 里所有 `FastPolicy(...)` 调用点（含 `__import__(...).FastPolicy` 形式）。"""
    out = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if (isinstance(f, ast.Name) and f.id == "FastPolicy") or \
           (isinstance(f, ast.Attribute) and f.attr == "FastPolicy"):
            out.append(n)
    return out


def test_every_fastpolicy_call_site_passes_model_channel_count():
    """webui 里**每一个** `FastPolicy(` 都必须显式传 `n_channels=<模型通道数>`。

    只查 `n_channels` 形参在不在还不够：`n_channels=12` 和
    `n_channels=getattr(ai, "in_channels", 12)` 都是「在」，但一个是写死的
    旧布局、一个自带回退默认值——正是本任务要拆的两种静默错配，故一并禁。
    """
    calls = _fastpolicy_calls(ast.parse(WEBUI_SRC))
    assert calls, "webui 里不再有 FastPolicy 调用点（rollout 装配被挪走？"\
                   "本用例必须跟着改，不许静默放过）"
    for call in calls:
        where = ast.unparse(call)
        kw = {k.arg: k.value for k in call.keywords}
        assert "n_channels" in kw, \
            f"FastPolicy 调用点没传 n_channels（会掉回签名默认 12）: {where}"
        src = ast.unparse(kw["n_channels"])
        assert src != "12", f"FastPolicy 的通道数被写死成 12: {where}"
        assert "getattr" not in src and " or " not in src, \
            f"FastPolicy 的通道数自带回退默认值（必须响亮地问模型）: {where}"
        # 通道数只能来自模型/搜索侧，不能来自本地常量
        assert "mcts" in src or "ai" in src, \
            f"FastPolicy 的通道数既不来自 mcts 也不来自 ai: {where}"


def test_webui_channel_count_reads_mcts_not_a_second_getattr():
    """webui 必须向 MCTS 要通道数，而不是自己再 `getattr(ai, ...)` 抄一遍。

    抄一份 = 契约出现第二个所有者 = 多一个可能悄悄退回 12 的地方，且丢掉
    「问不到就报错」（webui 侧只剩一个含糊的 AttributeError）。所以这里钉住
    数据流的方向：webui → `mcts.n_channels` → `_in_channels()` → `ai.in_channels`。
    """
    block = _rollout_setup_block()
    reads = [ast.unparse(n) for n in ast.walk(block)
             if isinstance(n, ast.Attribute) and n.attr == "n_channels"]
    assert reads, "rollout 装配段没读任何通道数（webui 又退回默认值了？）"
    for r in reads:
        assert r.startswith("session.mcts"), \
            f"webui 的通道数不是从 mcts 读来的: {r}"


# --------------------------------------------------------------------------- #
# 2. 行为：重建出来的 FastPolicy 按模型通道数
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ic", (12, 17))
def test_webui_rollout_policy_uses_model_channel_count(ic):
    """17 通道模型 → `FastPolicy.n_channels == 17`；12 通道模型 → 12（零回归）。

    端到端走 `main()` 里真实的那段装配代码（`_run_rollout_setup`），不是复刻。
    """
    session = _run_rollout_setup(_SpyAI(ic))
    fp = session.mcts._fast_policy
    assert fp is not None, "use_rollout=True 却没装上 FastPolicy"
    assert fp.n_channels == ic, \
        f"in_channels={ic} 的模型却让 webui 的 rollout 按 {fp.n_channels} 通道跑"
    assert fp.n == N and fp.weights is None, \
        "board_size / weights 语义被改动了（超出本任务范围）"
    # 与 MCTS 内部建特征用的是同一个数（不是「碰巧相等」）
    assert fp.n_channels == session.mcts.n_channels
    # 真跑一次 logits：17 通道下特征按 17 通道造得出来
    from src.game.go_rules import GoBoard
    logit = fp.logits(GoBoard(N))
    assert logit.shape == (N * N + 1,) and np.all(np.isfinite(logit))


def test_webui_rollout_policy_matches_mcts_constructed_one(ic=17):
    """webui 重建出来的那份，必须与 `MCTS.__init__` 自己造的那份**完全一致**。

    这是「同一个源」的最强证据：不只是数字相等，而是同参数构造出的同一份
    策略（`temperature` / 权重判等都一样），所以 webui 覆盖掉构造期那份
    不会带来任何行为差异——除了通道数从 12 变成模型的真实值。
    """
    ai = _SpyAI(ic)
    session = _run_rollout_setup(ai)
    webui_made = session.mcts._fast_policy
    mcts_made = MCTS(ai, board_size=N, num_threads=1, use_rollout=True)._fast_policy
    assert mcts_made.n_channels == ic
    assert webui_made.n_channels == mcts_made.n_channels
    assert webui_made.temperature == mcts_made.temperature
    assert webui_made._atari_penalty == mcts_made._atari_penalty
    assert webui_made.weights is mcts_made.weights is None


# --------------------------------------------------------------------------- #
# 3. MCTS.n_channels：只读，且「问不到就报错」没有被架空
# --------------------------------------------------------------------------- #
def test_mcts_n_channels_is_readonly_and_never_falls_back():
    """只读 property；模型问不到通道数时抛错（不是回退 12）。"""
    for ic in (12, 17):
        m = MCTS(_SpyAI(ic), board_size=N, num_threads=1)
        assert m.n_channels == ic
    with pytest.raises(AttributeError):
        MCTS(_SpyAI(12), board_size=N, num_threads=1).n_channels = 99

    class _NoChannelAI:
        def predict_batch(self, states):
            raise AssertionError("本用例不应触发前向")

    m = MCTS(_NoChannelAI(), board_size=N, num_threads=1)
    with pytest.raises(RuntimeError) as ei:
        _ = m.n_channels
    assert "in_channels" in str(ei.value), f"报错没说清缺的是什么: {ei.value}"


def test_webui_rollout_setup_raises_when_model_has_no_channels():
    """模型没有 `in_channels` 时，webui 的装配段必须**响**，不能静默按 12 跑。"""
    class _NoChannelAI:
        def predict_batch(self, states):
            raise AssertionError("本用例不应触发前向")

    with pytest.raises(RuntimeError) as ei:
        _run_rollout_setup(_NoChannelAI())
    assert "in_channels" in str(ei.value)
