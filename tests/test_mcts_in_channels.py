"""MCTS / LightPLS 的特征平面通道数必须跟随所挂模型（P4.8+ 后续）。

背景：P4.8/P4.13 让 GoAI 从 checkpoint 的 stem 形状推断 `in_channels`（只读
property），并让 `GoAI._build_state` / `predict_batch` 的特征通道随模型驱动。
但搜索侧还有 4 处写死的 `n_channels=12`（`src/search/mcts.py`）和 1 处
（`src/search/light_rollout.py`）：12 通道 checkpoint 下它们是**碰巧**对的，
17 通道 v21 模型下则被 P4.8 的守卫炸成「期望 17…实际 12」——响亮但不可用。
本文件把通道数改成向模型要，并锁住三件事：

  1. planes 的通道数 = 挂着的 `ai.in_channels`（12 就是 12，17 就是 17）；
  2. 12 通道路径**逐字节零回归**（对着改前抓下来的 baseline 数值比）；
  3. 真错配仍然响亮（`predict_batch` 的通道数守卫没被本次改动架空），且
     `mcts.py` worker 里那个 `except Exception: prefetch = None` 的吞异常
     不会把「通道数错」藏起来（主线程 `_expand` 会再前向一次并抛出）。

`baseline_in_channels.json` 是**改前**代码在本机抓的（`capture_baseline.py`
的输出，逐位 float32 十六进制）。它与 baseline 里的 fixture 定义（权重
seed、局面、模拟数）绑在一起：改了 fixture 就必须重抓，否则这里转红是
真的回归而不是噪音。
"""
import ast
import hashlib
import json
import os
import sys
import threading

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.go_rules import GoBoard
from src.inference import GoAI
from src.networks.alphanet import AlphaGoNet
from src.search.light_rollout import FastPolicy
from src.search.mcts import MCTS, MCTSNode

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_JSON = os.path.join(HERE, "baseline_in_channels.json")
CHANGED_FILES = [
    os.path.join(HERE, "..", "src", "search", "mcts.py"),
    os.path.join(HERE, "..", "src", "search", "light_rollout.py"),
]
N = 9
SEED = 1234


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _register_17_placeholder():
    """17ch 没有真构建器（v21 代连同 V21_CFG 已删除），占位只证明接缝 + 通道驱动。

    手法与已退役的 `tests/test_goai_dual_generation.py` 相同：17ch 的构建器
    不存在时，用一个「尊重 in_channels 参数」的构建器顶上去，端到端仍走真 GoAI。
    """
    from src import inference as inf

    def builder(*, in_channels, **arch_kwargs):
        return AlphaGoNet(in_channels=in_channels, **arch_kwargs)

    inf.register_in_channels_builder(17, builder)


@pytest.fixture(autouse=True)
def _restore_registry():
    """构建器注册表是进程级全局：用完即还原，别漏给同进程的后续用例。"""
    from src import inference as inf

    saved = dict(inf._IN_CHANNEL_BUILDERS)
    yield
    inf._IN_CHANNEL_BUILDERS.clear()
    inf._IN_CHANNEL_BUILDERS.update(saved)


@pytest.fixture(autouse=True)
def _single_thread():
    """数值取证要求前向的浮点求和顺序固定：否则线程数一变末位就变。"""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _make_ckpt(tmp_path, in_channels, seed=SEED):
    torch.manual_seed(seed)
    m = AlphaGoNet(in_channels=in_channels, backbone_channels=16,
                   backbone_res_blocks=2, action_size=N * N + 1, policy_layers=2)
    p = str(tmp_path / f"tiny_{in_channels}ch.pth")
    torch.save({"model": m.state_dict()}, p)
    return p


class _SpyAI:
    """只记录送进 predict_batch 的 planes 通道数的替身（不跑真网络）。

    `in_channels` 是唯一被 MCTS 读取的模型侧属性：这里能改它，就等于声明
    「我是一个 N 通道模型」。predict_batch 记下每批 planes 的通道数后返回
    形状正确的假结果——本文件只关心「特征按几个通道造出来的」。
    """

    def __init__(self, in_channels, board_size=N):
        self.in_channels = int(in_channels)
        self.n_actions = board_size * board_size + 1
        self.seen = []          # [(batch, channels, h, w), ...]
        self._bs = board_size

    def predict_batch(self, states):
        planes = np.stack([np.asarray(s[4]) for s in states])
        self.seen.append((planes.shape[0], planes.shape[1],
                          planes.shape[2], planes.shape[3]))
        if planes.shape[1] != self.in_channels:
            raise RuntimeError(
                f"预计算特征通道数不匹配：期望 {self.in_channels}"
                f"（模型 in_channels），实际 {planes.shape[1]}")
        b = planes.shape[0]
        pol = np.full((b, self.n_actions), 1.0 / self.n_actions, dtype=np.float32)
        return pol, np.zeros((b,), dtype=np.float32)

    @property
    def channels_seen(self):
        return sorted({c for (_b, c, _h, _w) in self.seen})


class _ThreadSpyAI(_SpyAI):
    """在 `_SpyAI` 基础上额外记录每次 `predict_batch` 来自哪个线程。

    全仓库只有 worker 推测性预取（`mcts.py:933`）从**非主线程**调
    `ai.predict_batch`（其余 5 处都在主线程），故按线程分桶即可证明
    「预取真的 fired」而不是靠主线程的调用糊弄过去。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread_channels = {}   # thread name -> {channels, ...}

    def predict_batch(self, states):
        planes = np.stack([np.asarray(s[4]) for s in states])
        name = threading.current_thread().name
        self.thread_channels.setdefault(name, set()).add(int(planes.shape[1]))
        return super().predict_batch(states)

    @property
    def worker_channels(self):
        main = threading.main_thread().name
        return sorted({c for name, cs in self.thread_channels.items()
                       if name != main for c in cs})


def _board_after(moves, n=N):
    b = GoBoard(n)
    for mv in moves:
        b.play(mv)
    return b


# --------------------------------------------------------------------------- #
# 1. planes 的通道数跟随模型
# --------------------------------------------------------------------------- #
def test_planes_channel_count_follows_model():
    """12 通道模型造出 12 通道 planes，17 通道模型造出 17 通道 planes。

    逐个点名**主线程**三个建特征的位置：(a) `_planes1`（单局面 + 缓存，
    `_expand` 叶子前向 / `lookahead` 根前向 / 主线程批量叶子前向）、
    (b) `_forward_level`（整批节点一次前向）、(c) `_eval_children`（叶子下
    所有子局面的批量特征）。任何一处退回写死 12 都会在这里转红——不是
    「不抛异常」，是**通道数**对不上。

    ⚠ 第四个位置——worker 推测性预取（`mcts.py:929-932`）——**不被本用例
    覆盖**：它在 worker 线程里把 `_cur_root_board` 克隆一份、沿路径再放
    一遍，然后**自己**调 `feature_planes_batched(...,
    n_channels=self._in_channels())`，与 `_planes1` 不是同一次计算。它由
    `test_worker_prefetch_site_follows_model_channels` 专门钉（spy 按线程
    记录，证明预取真的 fired 且通道数跟随模型），AST 规则兜底。
    """
    moves = (20, 30, 40, 50)
    for ic in (12, 17):
        spy = _SpyAI(ic)
        mcts = MCTS(spy, board_size=N, num_threads=1, temperature=0.0)
        board = _board_after(moves)
        mh, oh, tp = [20, 40, -1], [30, 50, -1], 1

        # (a) `_planes1`：单局面 + 缓存（`_expand` 的叶子前向、`lookahead` 根前向、
        #     主线程批量叶子前向都走它）
        p1 = mcts._planes1(board, mh, oh, tp)
        assert p1.shape == (ic, N, N), f"in_channels={ic} 时 _planes1 建出 {p1.shape}"

        # (b) `_forward_level`：整批节点一次前向（`lookahead` / `_batch_leaf_ab` 的逐层批量）
        spy.seen.clear()
        nodes = [(board, list(mh), list(oh), tp),
                 (_board_after(moves + (60,)), [60, 40, 20], [50, 30, -1], -1)]
        mcts._forward_level(nodes)
        assert spy.channels_seen == [ic], \
            f"in_channels={ic} 时 _forward_level 送进模型的通道数是 {spy.channels_seen}"

        # (c) `_eval_children`：叶子下所有子局面的批量特征（搜索里最高频的一处）
        spy.seen.clear()
        leaf = MCTSNode(board=board, my_hist=list(mh), op_hist=list(oh), to_play=tp,
                        move_int=-1)
        mcts._eval_children(board, tp, leaf, [20, 30, 40, N * N])
        assert spy.channels_seen == [ic], \
            f"in_channels={ic} 时 _eval_children 送进模型的通道数是 {spy.channels_seen}"


def test_worker_prefetch_site_follows_model_channels():
    """site #4（`mcts.py:929-932` worker 推测性预取）必须按**模型**通道数建特征。

    ⚠ 与 `_planes1` **不是**同一次计算：预取在 worker 线程里对克隆棋盘
    自己调 `feature_planes_batched(..., n_channels=self._in_channels())`，
    从不经过 `_planes1`——所以 `test_planes_channel_count_follows_model`
    的 (a)(b)(c) 三段覆盖不到这里，本用例专门钉它。

    并且要证明预取**真的 fired**（不是靠主线程的调用糊弄过去）：

      - `num_threads=4 + spec_prefetch=True` 打开预取（gate 见 `mcts.py:133`）；
      - worker 每产出一条路径都会**同步**触发一次预取前向（`leaf_q.put(path)`
        之后无中断地进入预取块），`search` 返回前又 `t.join()` 等所有 worker
        退出——故断言时预取**必然已经发生**；
      - `_ThreadSpyAI` 按线程记录通道数：全仓库只有 site #4 从非主线程调
        `ai.predict_batch`，`worker_channels` 非空即证明预取 fired；
      - 若 site #4 退回写死 12：预取仍会调 `predict_batch`，spy 在抛错
        **之前**就记下 12 通道 → `worker_channels == [12]` → 本用例转红
        （异常随后被 worker 那行 except 咽掉、主线程重前向，search 照样
        「成功」——所以光有端到端用例抓不住这种回归）。
    """
    spy = _ThreadSpyAI(17)
    m = MCTS(spy, board_size=N, num_threads=4, temperature=0.0, spec_prefetch=True)
    assert m.spec_prefetch, "num_threads=4 应打开 spec_prefetch，否则测不到 worker 预取"
    board = _board_after((20, 30, 40, 50))
    visits, probs, root_value = m.search(board, [20, 40, -1], [30, 50, -1], 1,
                                         simulations=24)
    assert spy.channels_seen == [17], \
        f"整场 search 里出现了非 17 通道的 predict_batch: {spy.thread_channels}"
    assert spy.worker_channels == [17], (
        f"worker 预取未按模型通道数建特征（worker 线程见到的通道: "
        f"{spy.worker_channels}；各线程: {spy.thread_channels}）")


def test_rollout_policy_feature_channel_follows_weights_contract():
    """FastPolicy：特征通道、`reshape` 与权重维数三者必须同一个数。

    MCTS 把自己的（模型驱动的）通道数交给 FastPolicy；独立调用方没有模型，
    保留 12 的默认值——但「默认 12」不能渗进 MCTS 的搜索路径。
    """
    w17 = np.linspace(-0.3, 0.3, 17, dtype=np.float32)
    pol = FastPolicy(N, weights=w17, n_channels=17)
    board = _board_after((20, 30))
    logit17 = pol.logits(board)
    assert np.all(np.isfinite(logit17))

    # 通道数与权重维数不一致时，权重项不参与（沿用改前的静默跳过），
    # 但特征不能因此按 12 造出来——把 n_channels 拨回 17 才能对上。
    pol_wrong = FastPolicy(N, weights=w17, n_channels=12)
    assert not np.allclose(pol_wrong.logits(board), logit17)

    # 默认（无模型调用方）：12，零回归
    assert FastPolicy(N).n_channels == 12
    w12 = np.linspace(-0.3, 0.3, 12, dtype=np.float32)
    assert np.allclose(FastPolicy(N, weights=w12).logits(board),
                       FastPolicy(N, weights=w12, n_channels=12).logits(board))


def test_mcts_passes_model_channel_count_to_rollout_policy():
    """`use_rollout=True` 时 MCTS 交给 FastPolicy 的通道数必须来自模型。"""
    for ic in (12, 17):
        m = MCTS(_SpyAI(ic), board_size=N, num_threads=1, use_rollout=True,
                 rollout_lambda=0.25)
        assert m._fast_policy is not None
        assert m._fast_policy.n_channels == ic, \
            f"in_channels={ic} 的模型却让 rollout 按 {m._fast_policy.n_channels} 通道跑"


# --------------------------------------------------------------------------- #
# 2. 12 通道路径逐字节零回归
# --------------------------------------------------------------------------- #
def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()
                          ).hexdigest()


def _numeric_probe(tmp_path):
    """12 通道模型下，把建特征位置的真实输出与真实前向结果取指纹。

    返回的 dict 就是 baseline 文件的内容。**必须在改代码之前抓**（那是
    「12 通道下改前行为本来就对」的证据），改完再抓一次逐项相等即零回归。

    覆盖图（每个键 → 它钉住哪个建特征位置）：

    | 键 | 覆盖 |
    |---|---|
    | `planes1_b1` / `planes1_b2` | #1 `_planes1`（两个局面防缓存串味） |
    | `forward_level_planes` | #2 `_forward_level` |
    | `eval_children_planes` | #3 `_eval_children` |
    | `search50/120_visits`、`*_value` | 端到端 search 输出（真实走过 #1/#2/#3） |
    | `search50/120_probs` | **smoke 键**：`temperature=0` 时 `probs` 是 one-hot（`mcts.py:1070-1079`），两键同 sha、信息量约 1 bit，只证明「没崩」；搜索级取证需 `temperature>0` + 非退化先验 |
    | `fastpolicy_logits` | #5/#6 `FastPolicy.logits` 内部（默认 12 通道 + 12 维权重的旧契约） |

    **不覆盖**：#4 worker 推测性预取（probe 用 `num_threads=1`，`mcts.py:133`
    的 `num_threads >= 4` gate 关掉预取）与 #0 `MCTS.__init__` 的 FastPolicy
    通道交接（probe 用默认 `use_rollout=False`）。这两处分别由
    `test_worker_prefetch_site_follows_model_channels` 与
    `test_mcts_passes_model_channel_count_to_rollout_policy` 钉住。
    """
    torch.manual_seed(SEED)
    ck = _make_ckpt(tmp_path, 12)
    ai = GoAI(model_path=ck, board_size=N, device="cpu", compile=False)
    assert ai.in_channels == 12
    mcts = MCTS(ai, board_size=N, num_threads=1, temperature=0.0)

    b1 = _board_after((20, 30, 40, 50))
    b2 = _board_after((0, 10, 20, 1, 11))
    mh1, oh1, tp1 = [20, 40, -1], [30, 50, -1], 1
    mh2, oh2, tp2 = [0, 20, 1], [10, 40, -1], -1

    out = {"fixture": {"seed": SEED, "board_size": N, "torch": torch.__version__,
                       "numpy": np.__version__}}

    # (a) _planes1 的 planes（两个不同局面，防缓存串味）
    out["planes1_b1"] = _sha(mcts._planes1(b1, mh1, oh1, tp1))
    out["planes1_b2"] = _sha(mcts._planes1(b2, mh2, oh2, tp2))

    # (b) _forward_level 送进 predict_batch 的 planes（记录真实张量）
    captured = []
    real_predict_batch = ai.predict_batch

    def _capture(states):
        captured.append(np.stack([np.asarray(s[4]) for s in states]))
        return real_predict_batch(states)

    nodes = [(b1, list(mh1), list(oh1), tp1), (b2, list(mh2), list(oh2), tp2)]
    ai.predict_batch = _capture
    try:
        mcts._forward_level(nodes)
    finally:
        del ai.predict_batch
    out["forward_level_planes"] = _sha(captured[0])

    # (c) _eval_children 送进 predict_batch 的 planes
    captured.clear()
    leaf = MCTSNode(board=b1, my_hist=list(mh1), op_hist=list(oh1), to_play=tp1,
                    move_int=-1)
    ai.predict_batch = _capture
    try:
        mcts._eval_children(b1, tp1, leaf, [20, 30, 40, 50, N * N])
    finally:
        del ai.predict_batch
    out["eval_children_planes"] = _sha(captured[0])

    # (d) 端到端搜索结果：visits 分布 + 根价值（真正会被下游看到的东西）
    for tag, sims in (("search50", 50), ("search120", 120)):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        m = MCTS(ai, board_size=N, num_threads=1, temperature=0.0)
        visits, probs, root_value = m.search(b1, list(mh1), list(oh1), tp1,
                                             simulations=sims)
        out[tag + "_visits"] = _sha(visits)
        out[tag + "_probs"] = _sha(probs)
        out[tag + "_value"] = repr(float(root_value))

    # (e) FastPolicy 的 rollout 走子 logit（12 通道 + 12 维权重的旧契约）
    pol12 = FastPolicy(N, weights=np.linspace(-0.3, 0.3, 12, dtype=np.float32))
    out["fastpolicy_logits"] = _sha(pol12.logits(b1))
    return out


def test_twelve_channel_path_bit_identical(tmp_path):
    """对着**改前**抓下的 baseline 逐项比指纹：12 通道路径必须一字不差。

    baseline 由 `capture_baseline.py` 用改前的代码抓出（`python
    capture_baseline.py tests/baseline_in_channels.json`）。用 sha256 而不是
    裸数值：只要有任何一个 bit 变了，指纹就变。
    """
    if not os.path.exists(BASELINE_JSON):
        pytest.fail(
            f"缺 baseline: {BASELINE_JSON}。用改前的代码抓一份："
            f"python tests/capture_baseline.py tests/baseline_in_channels.json")
    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        base = json.load(f)
    now = _numeric_probe(tmp_path)

    assert now["fixture"] == base["fixture"], \
        f"取证夹具变了（baseline 记的是 {base['fixture']}，现在 {now['fixture']}）"
    for k in ("planes1_b1", "planes1_b2", "forward_level_planes",
              "eval_children_planes", "search50_visits", "search50_probs",
              "search50_value", "search120_visits", "search120_probs",
              "search120_value", "fastpolicy_logits"):
        assert now[k] == base[k], (
            f"12 通道路径回归：{k} baseline={base[k]} 现在={now[k]}")


# --------------------------------------------------------------------------- #
# 3. 真错配仍然响亮
# --------------------------------------------------------------------------- #
def test_mismatched_planes_still_raise(tmp_path):
    """改完之后，P4.8 的通道数守卫仍然可达、仍然响亮。

    守卫（`GoAI.predict_batch` / `_build_state` 里的 `planes.shape[0] !=
    self.in_channels`）是「12ch 模型吃 17ch 特征」这类静默错配的最后一道防线。
    本次改动让 MCTS 不再自己造错配，但仓库里仍有别的 12 通道调用方
    （selfplay_train / async_pipeline 的采集缓冲），所以这道门不能被架空。
    另附一条：`_SpyAI` 自己实现的那道门也必须在 17 通道模型吃 12 通道 planes
    时抛（它就是 MCTS 侧的直接等价物）。
    """
    _register_17_placeholder()
    ai = GoAI(model_path=_make_ckpt(tmp_path, 17), board_size=N, device="cpu",
              compile=False)
    assert ai.in_channels == 17
    board = _board_after((20, 30))
    mh, oh, tp = [20, -1, -1], [-1, -1, -1], 1

    planes12 = board.feature_planes_batched(
        board.board[None], [list(mh)], [list(oh)], [tp], [board.ko_point],
        n_channels=12)[0]
    with pytest.raises(RuntimeError) as ei:
        ai.predict_batch([(None, mh, oh, tp, planes12)])
    msg = str(ei.value)
    assert "17" in msg and "12" in msg, f"报错未同时给出期望/实际: {msg}"
    assert "期望" in msg and "实际" in msg, f"报错未按契约给期望/实际: {msg}"

    # 对照组：17 通道 planes 走同一条路必须通（证明门是「通道数」不是「5 元组」）
    planes17 = board.feature_planes_batched(
        board.board[None], [list(mh)], [list(oh)], [tp], [board.ko_point],
        n_channels=17)[0]
    pol, val = ai.predict_batch([(None, mh, oh, tp, planes17)])
    assert pol.shape == (1, N * N + 1) and val.shape == (1,)

    # MCTS 侧等价门：改完之后 MCTS 自己不再造错配——17 通道模型经 MCTS 拿到
    # 17 通道 planes，这道门是「开」的（对照组）；但错配一旦真的发生，同一道
    # 门必须照旧响亮。两者缺一不可：只留对照组就成了空转。
    #
    # ⚠ 规格订正（report §「规格冲突」）：本用例原先写成
    #   `MCTS(_SpyAI(17))._forward_level(...)` 必须抛——那断言的恰恰是
    #   **改前**的 bug（MCTS 钉死 12 → 17 通道模型吃 12 格 → 抛）。改完之后
    #   MCTS 造 17 格，门自然开着，所以错配改由直接送 planes 触发：出口
    #   （`predict_batch`）不变，只是喂进去的东西换回 12 格。
    spy = _SpyAI(17)
    mcts = MCTS(spy, board_size=N, num_threads=1)
    spy.seen.clear()
    mcts._forward_level([(board, list(mh), list(oh), tp)])
    assert spy.channels_seen == [17], "对照组失效：MCTS 竟没喂 17 通道"
    with pytest.raises(RuntimeError) as ei2:
        spy.predict_batch([(None, list(mh), list(oh), tp, planes12)])
    assert "17" in str(ei2.value) and "12" in str(ei2.value)


def test_worker_prefetch_swallow_does_not_hide_channel_mismatch_end_to_end():
    """`mcts.py` worker 的 `except Exception: prefetch = None` 不能藏住通道数错。

    改前 worker 用写死的 12 造 planes，17 通道模型下 `predict_batch` 抛的
    「期望 17…实际 12」被这一行咽掉（`prefetch=None`）。安全性来自主线程：
    `_expand` 拿到没有 prefetch 的叶子会自己再前向一次（`_planes1` +
    `predict_batch`），异常在那里抛出，`search` 直接失败。

    ⚠ 规格订正（report §「规格冲突」）：原用例靠「`in_channels=17` 而 MCTS
    仍造 12 通道 planes」触发错配——那正是**改前**的 bug 本身。改完之后 MCTS
    照模型造 17 格，错配不再发生，用例就变成了一个永假的断言（它此前能过，
    恰恰是因为钉死的 12 还在）。要验「吞异常不藏错」，必须让错配真的发生：
    这里用一个「嘴上说 17、真身是 12 通道 stem」的模型——MCTS 按它**声明**的
    `in_channels` 造 17 格（走的是新代码路径），网络守卫按**真身**要 12 格，
    错配于是真实发生，并被 worker 那行 except 咽掉；安全性由主线程兜底。

    两段都是端到端证据（故 `_end_to_end` 后缀），但**都不直接证明 worker
    预取 fired**：第 1 段的抛出点可能在主线程 `_expand`（worker 预取的那次
    已被咽掉），`assert m.spec_prefetch` 只是**前置闸门**（保证预取路径是
   开的），不是「worker 已被走到」的证明；预取本身 fired 的证据由
    `test_worker_prefetch_site_follows_model_channels` 给。第 2 段直接钉住
    兜底机制：`prefetch is None` 的叶子必须由 `_expand` 重前向，错配在
    那里照常逃逸（吞异常不是承重墙）。
    """
    class _BoomAI:
        in_channels = 17          # 声明值：MCTS 据此造 17 通道 planes
        _stem_channels = 12       # 真身：网络只吃 12 通道（旧布局权重）

        def predict_batch(self, states):
            planes = np.stack([np.asarray(s[4]) for s in states])
            if planes.shape[1] != self._stem_channels:
                raise RuntimeError(
                    f"预计算特征通道数不匹配：期望 {self._stem_channels}"
                    f"（模型 in_channels），实际 {planes.shape[1]}")
            return (np.full((planes.shape[0], N * N + 1), 1.0 / (N * N + 1),
                             dtype=np.float32),
                    np.zeros((planes.shape[0],), dtype=np.float32))

    m = MCTS(_BoomAI(), board_size=N, num_threads=4, temperature=0.0,
             spec_prefetch=True)
    # 前置闸门：保证预取路径是开的（不是「worker 已被走到」的证明，见 docstring）
    assert m.spec_prefetch, "num_threads=4 应打开 spec_prefetch，否则测不到 worker 预取"
    board = _board_after((20, 30, 40, 50))
    with pytest.raises(RuntimeError) as ei:
        m.search(board, [20, 40, -1], [30, 50, -1], 1, simulations=24)
    assert "17" in str(ei.value) and "12" in str(ei.value), \
        f"worker 吞掉了通道数错、且没被主线程再抛出来: {ei.value}"

    # 上面那次抛出可能落在根展开上。要害是「worker 咽掉的那一次，主线程
    # `_expand` 会自己再前向并抛出」——直接钉住这个兜底机制：prefetch=None
    # 的叶子必须由 `_expand` 重前向，错配在那里照常逃逸（吞异常不是承重墙）。
    m1 = MCTS(_BoomAI(), board_size=N, num_threads=1, temperature=0.0)
    leaf = MCTSNode(board=_board_after((20, 30, 40, 50)), my_hist=[20, 40, -1],
                    op_hist=[30, 50, -1], to_play=1, move_int=-1)
    assert leaf.prefetch is None, "夹具前提：叶子没有 prefetch（= worker 咽掉后的状态）"
    with pytest.raises(RuntimeError) as ei2:
        m1._expand(leaf)
    assert "17" in str(ei2.value) and "12" in str(ei2.value), \
        f"_expand 没用 prefetch=None 的叶子自己重前向: {ei2.value}"


def test_seventeen_channel_search_runs_end_to_end(tmp_path):
    """17 通道模型（占位 builder 造）走**整条**搜索链路（含 worker 推测性预取）
    必须跑通。

    前面那些用例是把三处建特征拆开点名的；这条把它们串起来：真 `GoAI`
    （17ch，由本文件的占位 builder 造）+ 真 MCTS + `num_threads=4`
    打开 `spec_prefetch`（worker 预取路径真的会被走到）+ 一次 `search`。
    改前这条路必然炸在 F5（「期望 17…实际 12」）。
    """
    _register_17_placeholder()
    for ic, threads in ((12, 1), (17, 4)):
        ai = GoAI(model_path=_make_ckpt(tmp_path, ic), board_size=N,
                  device="cpu", compile=False)
        assert ai.in_channels == ic
        m = MCTS(ai, board_size=N, num_threads=threads, temperature=0.0,
                 spec_prefetch=True, use_rollout=True, rollout_lambda=0.25,
                 rollout_steps=6)
        assert m._fast_policy.n_channels == ic
        board = _board_after((20, 30, 40, 50))
        visits, probs, root_value = m.search(board, [20, 40, -1], [30, 50, -1],
                                             1, simulations=8)
        assert visits.shape == (N * N + 1,) and probs.shape == (N * N + 1,)
        assert visits.sum() > 0, f"in_channels={ic} 时 search 一次模拟都没跑"
        assert np.isfinite(root_value), f"in_channels={ic} 根价值非有限: {root_value}"
        assert probs.sum() > 0, f"in_channels={ic} 选点分布全零"


# --------------------------------------------------------------------------- #
# 4. 源码里不再有写死的 12
# --------------------------------------------------------------------------- #
def test_no_hardcoded_12_remains():
    """AST 级断言：改动过的两个文件里不得再有 `n_channels=12` 之类的写死值。

    规则只针对**通道数**这一件事（规格的原意是「feature_planes 调用点」）：

      - 调用点关键字 `f(..., n_channels=12)` → 违规；
      - 赋值 `n_channels = 12` / `self.n_channels = 12` → 违规；
      - 函数签名默认值 `n_channels: int = 12` → 按**形参名** `n_channels`
        收集，唯一合法例外是 `FastPolicy.__init__` 那一个（给**没有模型**的
        独立调用方 webui / selfplay_train / 本模块 __main__ / tests 保 12 维
        权重旧布局用的），故断言收集结果恰好只有它。

    与通道数无关的 `= 12` 默认值（如 `lookahead(topk=12)` 的搜索宽度）
    **不在本规则管辖内**——别把下一个实现者逼进「改个名绕开规则」的角落
    （初版规则过宽，曾逼出 `DEFAULT_LOOKAHEAD_TOPK` 那个改名 workaround，
    Fix 轮已收窄并回退，见 report §Fix 增补 #7）。
    """
    offenders = []
    allowed_defaults = []
    for path in CHANGED_FILES:
        src = open(path, "r", encoding="utf-8").read()
        tree = ast.parse(src, filename=path)
        rel = os.path.relpath(path, os.path.join(HERE, "..")).replace("\\", "/")
        for node in ast.walk(tree):
            # 形如 f(..., n_channels=12) 的关键字实参
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if (kw.arg == "n_channels" and isinstance(kw.value, ast.Constant)
                            and kw.value.value == 12):
                        offenders.append(f"{rel}:{node.lineno} f(n_channels=12)")
            # 形如 n_channels = 12 / self.n_channels = 12 的赋值
            if isinstance(node, ast.Assign):
                tgts = [t for t in node.targets if isinstance(t, ast.Name)
                        or isinstance(t, ast.Attribute)]
                names = {t.id for t in tgts if isinstance(t, ast.Name)}
                names |= {t.attr for t in tgts if isinstance(t, ast.Attribute)}
                if "n_channels" in names and isinstance(node.value, ast.Constant) \
                        and node.value.value == 12:
                    offenders.append(f"{rel}:{node.lineno} n_channels = 12")
            # 函数签名的默认值：只认形参名为 n_channels 的（规格原意）
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pos = node.args.args
                pos_defaults = node.args.defaults
                offset = len(pos) - len(pos_defaults)  # defaults 对齐到参数表末尾
                for i, d in enumerate(pos_defaults):
                    if isinstance(d, ast.Constant) and d.value == 12 \
                            and pos[offset + i].arg == "n_channels":
                        allowed_defaults.append(
                            f"{rel}:{node.lineno} {node.name}() 的默认参数 = 12")
                for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
                    if d is not None and isinstance(d, ast.Constant) \
                            and d.value == 12 and a.arg == "n_channels":
                        allowed_defaults.append(
                            f"{rel}:{node.lineno} {node.name}() 的默认参数 = 12")
    assert not offenders, "仍有写死的 12 通道特征调用点:\n  " + "\n  ".join(offenders)
    assert allowed_defaults == [
        f"src/search/light_rollout.py:{FastPolicy.__init__.__code__.co_firstlineno} "
        f"__init__() 的默认参数 = 12"
    ], ("n_channels 默认值 = 12 的出现位置变了（合法例外只有 FastPolicy 那一个）: "
        f"{allowed_defaults}")


def test_mcts_does_not_silently_fall_back_to_12():
    """ai 没有 `in_channels` 时必须**报错**，不得回退 12。

    回退 12 就是把 MCTS 重新钉回旧布局——那正是本任务要拆掉的东西，而且
    它会安静地给一个未知通道数的模型喂 12 通道特征。
    """
    class _NoChannelAI:
        def predict_batch(self, states):
            raise AssertionError("本用例不应触发前向")

    m = MCTS(_NoChannelAI(), board_size=N, num_threads=1)
    board = _board_after((20, 30))
    with pytest.raises(RuntimeError) as ei:
        m._planes1(board, [20, -1, -1], [-1, -1, -1], 1)
    assert "in_channels" in str(ei.value), f"报错没说清缺的是什么: {ei.value}"
