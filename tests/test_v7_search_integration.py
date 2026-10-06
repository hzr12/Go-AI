# -*- coding: utf-8 -*-
"""原生 MCTS 支持 V7（22 通道 + 19 维全局输入）的接合处测试。

前面那些单点验证（`test_search_v7_features.py` 验特征装配、`test_v7_forward_*`
验前向口径）都是**单点**。真正的风险全在接合处，而接合处的错误**一个都不
报错**：

  - MCTS 少给一张前手盘面 ⇒ `ladder_channels` 回退到「在当前盘上算」，
    形状照样 22 通道，ch15/ch16 语义已错；
  - 特征缓存 key 漏掉前两张盘面 ⇒ 两个节点撞同一条缓存，拿到别人的梯子通道；
  - 某个调用点还在走 `feature_planes_batched` ⇒ 直接 ValueError（这条会响，
    算是运气好）；
  - policy 取了 `[:,1]`（对手 π_opp）⇒ 搜索方向整个反过来，着法看着像那么回事。

所以本文件用**假 V7 模型**（秒级、可控）把接合处逐个钉住，再用**真权重**
单独验一次口径。假模型刻意返回 V7 的真实输出形状（dict + `(B,2,A)` +
`(B,3)`），这样口径转错会被抓住而不是被假模型掩盖。
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.go_rules import GoBoard  # noqa: E402
from src.inference import GoAI  # noqa: E402
from src.search.mcts import MCTS  # noqa: E402
from src.search.v7_features import V7_GLOBAL_CHANNELS, V7_SPATIAL_CHANNELS  # noqa: E402

N = 19
PASS = N * N
REAL_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tmp", "coding", "a_v7_npu4.pth")


# --------------------------------------------------------------------------- #
# 假 V7 模型
# --------------------------------------------------------------------------- #
class _FakeV7(nn.Module):
    """形状与真 `NbtTfNet` 一致的最小 V7：双输入、dict 输出、2 路策略 + 3 类结果。

    刻意**不复刻**真模型的计算（那不是这里要测的），只保证：
      - `REQUIRES_GLOBAL_FEATURES = True`（GoAI 靠它认双输入）；
      - `forward(spatial, global_features)` 返回 dict；
      - `policy_logits` 是 `(B,2,A)`、`outcome_logits` 是 `(B,3)`。
    这样 policy 取错路、value 口径错，都会被下面那些断言抓住。
    """

    REQUIRES_GLOBAL_FEATURES = True

    def __init__(self, in_channels=22, n_actions=PASS + 1):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.n_actions = n_actions
        self.seen_globals = []          # 供测试断言「第二路真的送进来了」

    def forward(self, spatial, global_features, board_mask=None):
        assert global_features is not None, "假模型也要求第二路输入非空"
        self.seen_globals.append(global_features.detach().clone())
        h = self.stem(spatial).mean(dim=(2, 3))          # (B,4)
        g = global_features[:, :4]
        z = (h + g)[:, :1]                                # (B,1) 让输出随两路输入都变
        B = spatial.shape[0]
        # 策略：两路必须**softmax 之后**仍不同，否则本用例测不出「取错路」——
        # 踩过的坑：先给一个常量向量再 softmax，均匀分布让两路恒等，断言失效。
        ramp = torch.linspace(0.0, 1.0, self.n_actions).unsqueeze(0)
        pl = torch.stack([z.expand(B, self.n_actions) * ramp,      # 本方 π
                          -z.expand(B, self.n_actions) * ramp], dim=1)
        oc = torch.zeros(B, 3)
        oc[:, 0] = 0.6 + 0.2 * z[:, 0]                    # P(win)
        oc[:, 2] = 0.1 * (1 - z[:, 0].clamp(0, 1))        # P(loss)
        return {"policy_logits": pl, "outcome_logits": oc}


@pytest.fixture
def v7_ai(tmp_path):
    """注册 22 通道假 V7 builder，并造一份真 checkpoint 让 `GoAI` 推出 22。

    走「存 checkpoint 再加载」而不是给 `GoAI` 直接传通道数：`GoAI` 无 checkpoint
    时把 `in_channels` 硬编码成 12（那是刻意的零回归默认），所以想测 22 就必须
    让它从 stem 形状**推断**出来 —— 这也顺带覆盖了 `_infer_in_channels` 放行 22
    这条新路径。做法与 `test_mcts_in_channels._make_ckpt` 一致。
    """
    from src import inference as inf
    saved = dict(inf._IN_CHANNEL_BUILDERS)
    inf.register_in_channels_builder(
        22, lambda *, in_channels, **kw: _FakeV7(in_channels=in_channels))
    try:
        torch.manual_seed(7)
        ckpt = tmp_path / "fake_v7.pth"
        torch.save(_FakeV7().state_dict(), ckpt)
        yield GoAI(model_path=str(ckpt), device="cpu", use_amp=False)
    finally:
        inf._IN_CHANNEL_BUILDERS.clear()
        inf._IN_CHANNEL_BUILDERS.update(saved)


def _legal(board, mv):
    """`mv` 是否合法。**pass 槽 (N*N) 不在 `get_legal_moves()` 掩码里**。

    那个掩码是 `size*size` 的 bool（索引 0..360），而 MCTS 的动作空间是
    `size*size+1`（多一个 pass 槽 361）。所以判合法必须把 pass 单独放行 ——
    直接 `mask[mv]` 在引擎叫 pass 时会 IndexError，而那恰恰是最该测通的一手。
    """
    return mv == PASS or bool(board.get_legal_moves()[mv])


@pytest.fixture
def mcts(v7_ai):
    return MCTS(v7_ai, board_size=N, num_threads=1, expand_topk=8, komi=7.5)


def _hist():
    return [-1, -1, -1], [-1, -1, -1]


# --------------------------------------------------------------------------- #
# 1. 识别与特征口径
# --------------------------------------------------------------------------- #
def test_mcts_recognises_the_v7_path(v7_ai, mcts):
    assert v7_ai.in_channels == 22
    assert v7_ai.needs_global_features, "GoAI 必须靠模型声明认双输入，不能看通道数"
    assert mcts._v7, "MCTS 没把 V7 认出来"


def test_feature_inputs_are_22_channels_plus_19_globals(mcts):
    b = GoBoard(N)
    for mv in (61, 300, 62):
        b.play(mv)
    my, op = [62, 61, -1], [300, -1, -1]
    sp, gl = mcts._feature_inputs(b, my, op, b.current_player,
                                  prev_board=b.board.copy())
    assert sp.shape == (V7_SPATIAL_CHANNELS, N, N)
    assert sp.dtype == np.float16
    assert gl is not None and gl.shape == (V7_GLOBAL_CHANNELS,)
    assert gl.dtype == np.float16


def test_planes1_still_returns_only_planes(mcts):
    """`_planes1` 是给 lookahead 等旧调用点的薄封装，不能改变返回契约。"""
    b = GoBoard(N)
    out = mcts._planes1(b, [-1, -1, -1], [-1, -1, -1], 1)
    assert isinstance(out, np.ndarray), type(out)
    assert out.shape == (V7_SPATIAL_CHANNELS, N, N)


# --------------------------------------------------------------------------- #
# 2. 缓存 key 必须区分前手盘面（本文件最重要的一条）
# --------------------------------------------------------------------------- #
def test_cache_key_separates_different_prev_boards(mcts):
    """同一张盘面 + 不同的前手盘面 ⇒ 必须落到**不同的缓存条目**。

    key 里漏掉前两张盘面，两次调用就会共用一条缓存、拿到同一份特征：形状完全
    正常、不报任何错，只是 ch15/ch16 变成了别人的历史。这比崩溃难查得多。

    这里断言**缓存条目数**而不是特征是否不同：4 子盘面上梯子通道全 0，
    「特征是否不同」在那种局面上恒为「相同」，断言会变成摆设（而它照样绿）。
    条目数是直接对「key 有没有算上前手盘面」的检验，不受局面内容影响。
    """
    b = GoBoard(N)
    for mv in (61, 300, 62):
        b.play(mv)
    tmp = GoBoard(N)
    for mv in (61, 300, 62, 299):
        tmp.play(mv)
    my, op = [62, 61, -1], [300, -1, -1]

    mcts._feature_inputs(b, my, op, 1, prev_board=b.board.copy())
    assert len(mcts._plane_cache) == 1
    # 当前盘完全相同，只换前手盘面
    mcts._feature_inputs(b, my, op, 1, prev_board=tmp.board.copy())
    assert len(mcts._plane_cache) == 2, (
        "前手盘面不同却只有一条缓存 —— key 漏了前手盘面，两次调用拿到了同一份"
        "特征，ch15/ch16 静默串了")
    # 前手盘也相同 ⇒ 应当命中缓存（顺带确认缓存本身还工作）
    mcts._feature_inputs(b, my, op, 1, prev_board=tmp.board.copy())
    assert len(mcts._plane_cache) == 2, "同前手盘面重复调用没命中缓存"


def test_cache_key_separates_komi(mcts):
    """贴目是特征的一部分（进 ch5/ch18），同盘面不同贴目不能撞缓存。"""
    b = GoBoard(N)
    b.play(61)
    mcts._feature_inputs(b, [-1, -1, -1], [-1, -1, -1], 1)
    assert len(mcts._plane_cache) == 1
    mcts.komi = 0.5
    mcts._feature_inputs(b, [-1, -1, -1], [-1, -1, -1], 1)
    assert len(mcts._plane_cache) == 2, "不同贴目撞了同一条缓存"


# --------------------------------------------------------------------------- #
# 3. 两路输入必须成对（错一半 = 形状对内容错）
# --------------------------------------------------------------------------- #
def test_giving_only_planes_is_rejected(v7_ai):
    b = GoBoard(N)
    b.play(61)
    with pytest.raises(RuntimeError, match="只给了一路"):
        v7_ai.predict_batch([(b, [-1, -1, -1], [-1, -1, -1], 1,
                              np.zeros((22, N, N), np.float16))])


def test_giving_globals_to_a_non_v7_model_is_rejected():
    """非 V7 模型收到全局输入 = 调用方搞错了对象，必须响亮。"""
    ai = GoAI(model_path=None, device="cpu", use_amp=False)
    assert not ai.needs_global_features
    b = GoBoard(N)
    b.play(61)
    with pytest.raises(RuntimeError, match="没有第二路输入"):
        ai.predict_batch([(b, [-1, -1, -1], [-1, -1, -1], 1,
                           np.zeros((12, N, N), np.float16),
                           np.zeros(19, np.float16))])


def test_second_input_actually_reaches_the_model(v7_ai):
    """断言第二路真的送进了网络，而不是被静默丢掉。"""
    b = GoBoard(N)
    b.play(61)
    v7_ai.predict_batch([(b, [-1, -1, -1], [-1, -1, -1], 1)])
    assert v7_ai.model.seen_globals, "模型没收到任何全局输入"
    assert v7_ai.model.seen_globals[-1].shape == (1, V7_GLOBAL_CHANNELS)


# --------------------------------------------------------------------------- #
# 4. policy / value 口径
# --------------------------------------------------------------------------- #
def test_policy_takes_channel_0_not_the_opponent_channel(v7_ai):
    """`[:,1]` 是对手 π_opp。取它当自己的策略 = 搜索方向整个反过来。"""
    b = GoBoard(N)
    b.play(61)
    pol, _ = v7_ai.predict_batch([(b, [-1, -1, -1], [-1, -1, -1], 1)])
    built = v7_ai._build_state(b, [-1, -1, -1], [-1, -1, -1], 1)
    with torch.no_grad():
        out = v7_ai.model(built[0], built[1])
    p_mine = torch.softmax(out["policy_logits"][0, 0], dim=-1).numpy()
    p_opp = torch.softmax(out["policy_logits"][0, 1], dim=-1).numpy()
    assert not np.allclose(p_mine, p_opp), "假模型两路策略相同，本用例测不出东西"
    assert np.allclose(pol[0], p_mine, atol=1e-5), "用的不是 policy_logits[:,0]"


def test_value_is_p_win_minus_p_loss(v7_ai):
    b = GoBoard(N)
    b.play(61)
    _, val = v7_ai.predict_batch([(b, [-1, -1, -1], [-1, -1, -1], 1)])
    built = v7_ai._build_state(b, [-1, -1, -1], [-1, -1, -1], 1)
    with torch.no_grad():
        out = v7_ai.model(built[0], built[1])
    p = torch.softmax(out["outcome_logits"].float(), dim=-1)[0]
    assert abs(float(val[0]) - float(p[0] - p[2])) < 1e-5
    assert -1.0 <= float(val[0]) <= 1.0


# --------------------------------------------------------------------------- #
# 5. 节点携带前手盘面 + 整盘搜索
# --------------------------------------------------------------------------- #
def test_children_carry_prev_board(mcts):
    b = GoBoard(N)
    my, op = _hist()
    mcts.search(b, my, op, 1, simulations=8)
    root = mcts._prev_root
    assert root is not None and root.children, "没有子节点可查"
    child = next(iter(root.children.values()))
    assert child.prev_board is not None, "子节点没带 prev_board ⇒ ch15 静默退化"
    assert child.prev_prev_board is None, "根的子节点不该有前二手盘"


def test_full_search_returns_a_legal_move(v7_ai, mcts):
    b = GoBoard(N)
    my, op = _hist()
    to_play = 1
    for ply in range(3):
        visits, probs, root_value = mcts.search(
            b, my, op, to_play, simulations=8)
        mv = int(np.argmax(visits))
        assert _legal(b, mv), "第 %d 手 %d 非法" % (ply, mv)
        assert abs(float(probs.sum()) - 1.0) < 1e-3
        assert -1.0 <= float(root_value) <= 1.0
        b.play(mv)
        my, op = [mv] + my[:2], [op[0], op[1], mv]
        to_play = -to_play


# --------------------------------------------------------------------------- #
# 6. V7 上明确不支持的那条路（响亮拒绝，不静默走错）与现已打通的那条
# --------------------------------------------------------------------------- #
def test_leaf_alpha_beta_is_rejected_on_v7(v7_ai):
    """leaf α-β 仍不支持：它走 `lookahead.forward_level` 那条不携带前手盘的路。

    ⚠ 别把它和 `lookahead()` 混为一谈：后者在 2026-10-06 已经接通 V7
    （`feature_fn` + `v7_batch_features` + `child_states(with_prev=True)`），
    而 leaf α-β 走的是 `MCTS._batch_leaf_ab`，那里的节点仍是 4 元组、
    没有前手盘 ⇒ ch15/ch16 会退化成「整批缺失」，那比报错糟得多，所以继续拒。
    """
    with pytest.raises(ValueError, match="leaf α-β"):
        MCTS(v7_ai, board_size=N, num_threads=1, leaf_ab_depth=1)


def test_lookahead_works_on_v7(mcts):
    """V7 上 `lookahead()`（= webui 的 hybrid / policy_depth>=2 分支）已打通。

    这条在 2026-10-06 之前是「响亮拒绝」，拒绝理由是 `lookahead.py` 只会
    `feature_planes_batched` + 5 元组，给不出 22 通道与 19 维全局输入。现在
    通过 `feature_fn` 参数改走 `v7_batch_features`（批量、逐位等价于单图版），
    并给子局面带上前两手盘面供 ch15/ch16 的梯子重算。
    """
    b = GoBoard(N)
    b.play(4 * N + 4)          # 黑 D5
    b.play(4 * N + 15)         # 白 D16
    b.play(15 * N + 4)         # 黑 Q5

    vals, pol, best, best_val = mcts.lookahead(
        b, [-1, 4 * N + 4, 15 * N + 4], [-1, 4 * N + 15, -1], 1,
        topk=6, width=3, depth=2)

    assert vals, "候选着法不应为空（V7 上 hybrid 分支必须能出着法）"
    assert len(pol) == mcts.n_actions
    # masked policy 是「原始 policy 把非法点清零」，**不重新归一**（lookahead.py:114
    # 只做 `masked[legal] = p[legal]`），所以只能在 [0,1] 且非零，不能要求 ==1。
    assert 0.0 < float(pol.sum()) <= 1.0 + 1e-6, "masked policy 越界"
    assert best in vals, "best_move 必须在候选集合内"
    assert -1.0 <= float(best_val) <= 1.0


def test_lookahead_on_v7_carries_the_prev_boards(mcts):
    """子局面必须带上前两手盘面 —— 否则 ch15/ch16 静默退化成「整批缺失」。

    `lookahead()` 内部是纯函数、不走搜索树，所以根局面的「前一手盘」只能由
    `MCTS._prev_board_arrays` 从 `move_history` 重放取；`clone()+undo()` 取不到
    （`GoBoard.clone()` 不带 undo 栈，实测连撤四次都返回 False）。
    """
    b = GoBoard(N)
    for mv in (4 * N + 4, 4 * N + 15, 15 * N + 4):
        b.play(mv)
    prev, prev_prev = mcts._prev_board_arrays(b)
    assert prev is not None and prev.shape == (N, N), "历史够深却没取到前一手盘"
    assert prev_prev is not None and prev_prev.shape == (N, N), "历史够深却没取到前二手盘"
    assert not np.array_equal(prev, prev_prev), "前一手盘与前二手盘不应相同"


def test_prev_board_arrays_is_none_when_history_is_short(mcts):
    """历史不足时返回 None（= 缺失），**不许拿空盘冒充**。

    拿全 0 盘面冒充会让 ch15/ch16 去算一个空盘上的梯子，得到的是训练里从未
    出现过的输入，而且不报任何错。
    """
    b = GoBoard(N)
    assert mcts._prev_board_arrays(b) == (None, None), "开局不该有前手盘"
    b.play(4 * N + 4)
    prev, prev_prev = mcts._prev_board_arrays(b)
    # 只有一手历史时统一退化成「缺失」：把「一手之前的空盘」当作前一手盘会让
    # ch15/ch16 进入 history=1 门控，而官方口径是 history=0（ch14==ch15==ch16）——
    # 那个空盘并不是一个真实存在过的「前一手」。
    assert prev is None and prev_prev is None, "一手历史应当两格都缺（=整批缺失）"


def test_komi_reaches_the_v7_global_features(v7_ai):
    """贴目必须真的进 V7 全局输入（ch5 / ch18），且随取值变化。

    2026-10-06 之前 webui 的 `Session` 构造 `MCTS(...)` 时**没传 komi**，
    于是原生 MCTS 路径恒用 MCTS 默认 7.5、且没有任何 CLI 开关（`--engine-komi`
    只喂 GTP 后端）。V7 的 ch5=`currentSelfKomi/20`、ch18=贴目三角波都由它算，
    填错就是模型拿到训练里没见过的输入 —— 且**不报任何错**。
    """
    b = GoBoard(N)
    b.play(4 * N + 4)
    seen = {}
    for k in (7.5, 0.5):
        m = MCTS(v7_ai, board_size=N, num_threads=1, komi=k)
        assert m.komi == k, "komi 没有存进 MCTS"
        _sp, gl = m._feature_inputs(b, [-1, -1, 4 * N + 4], [-1, -1, -1], 1)
        seen[k] = (float(gl[5]), float(gl[18]))
        # 全局特征是 float16（与训练、12 通道 planes 同一 dtype），所以按 fp16
        # 精度比而不是 float64 —— 0.025 在 fp16 里就是 0.024993896484375。
        assert abs(seen[k][0] - k / 20.0) < 1e-4, \
            "ch5 必须是 currentSelfKomi/20，实得 %r" % seen[k][0]
    assert seen[7.5][1] != seen[0.5][1], \
        "ch18 贴目三角波应随贴目变化，实得两者都是 %r" % (seen[7.5][1],)


def test_search_features_use_the_v7_training_history_order(v7_ai):
    """搜索侧产出的 ch9..13 必须与 V7 **训练**口径一致（index 0 = 最近一手）。

    推理侧所有对局驱动都用 `hist.pop(0); hist.append(mv)`（最老在前），
    12 通道也正是这个方向；但 V7 训练数据（`v7_dataset` 换过槽位）是「最近在前」。
    不对齐的话 ch9..13 与全局 ch0..4 会整体时间倒序 —— 形状全对、不报错、
    着法看着正常，所以只能靠测试钉住。
    """
    b = GoBoard(N)
    for mv in (4 * N + 4, 4 * N + 15, 15 * N + 4, 15 * N + 15):
        b.play(mv)
    to_play = b.current_player
    # webui 口径：黑 [-1, D5, Q5]、白 [-1, D16, Q16]，即最老在前
    webui_black = [-1, 4 * N + 4, 15 * N + 4]
    webui_white = [-1, 4 * N + 15, 15 * N + 15]
    if to_play == 1:
        my_h, op_h = webui_black, webui_white
    else:
        my_h, op_h = webui_white, webui_black

    m = MCTS(v7_ai, board_size=N, num_threads=1)
    sp, _gl = m._feature_inputs(b, my_h, op_h, to_play)

    def _pt(ch):
        idx = np.argwhere(sp[ch] > 0)
        return (int(idx[0][0]), int(idx[0][1])) if idx.size else None

    # 走完 4 手后，最近一手 = Q16。ch9 = opp 最近手 / ch10 = pla 最近手。
    # 具体哪一边是「pla」取决于轮到谁，但**两个必须分别是 D5 与 Q16**，
    # 而不是 (Q5, Q16) 那种被倒序的结果。
    assert _pt(9) == (15, 15), "ch9 应是对方最近一手 Q16，实得 %r" % (_pt(9),)
    assert _pt(10) in ((4, 4), (15, 4)), \
        "ch10 应是我方最近一手 D5 或 Q5，实得 %r" % (_pt(10),)


# --------------------------------------------------------------------------- #
# 7. 12 通道路径零回归
# --------------------------------------------------------------------------- #
def test_12_channel_path_is_untouched():
    """12 通道模型仍走 feature_planes，且**不**被要求提供全局输入。"""
    ai = GoAI(model_path=None, device="cpu", use_amp=False)
    assert ai.in_channels == 12 and not ai.needs_global_features
    m = MCTS(ai, board_size=N, num_threads=1, expand_topk=8)
    assert not m._v7
    b = GoBoard(N)
    b.play(61)
    sp, gl = m._feature_inputs(b, [-1, -1, -1], [-1, -1, -1], 1)
    assert sp.shape[0] == 12, sp.shape
    assert gl is None, "12 通道不该有全局输入"
    visits, probs, rv = m.search(b, [-1, -1, -1], [-1, -1, -1], 1, simulations=8)
    assert _legal(b, int(np.argmax(visits)))


# --------------------------------------------------------------------------- #
# 8. 真权重复核口径（无权重则跳过）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isfile(REAL_CKPT),
                    reason="真 V7 权重不在（tmp/coding/a_v7_npu4.pth）")
def test_real_checkpoint_uses_the_same_conventions():
    """假模型钉住「代码有没有转对」，真权重钉住「这个 checkpoint 能不能跑」。"""
    ai = GoAI(model_path=REAL_CKPT, device="cpu", use_amp=False, komi=7.5)
    assert ai.in_channels == 22 and ai.needs_global_features
    b = GoBoard(N)
    for mv in (61, 300, 62, 299):
        b.play(mv)
    pol, val = ai.predict_batch([(b, [62, 61, -1], [299, 300, -1],
                                  b.current_player)])
    assert pol.shape == (1, PASS + 1)
    assert abs(float(pol[0].sum()) - 1.0) < 1e-3
    assert _legal(b, int(np.argmax(pol[0]))), "首选着点不合法"
    assert -1.0 <= float(val[0]) <= 1.0
