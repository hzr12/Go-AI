"""P3-C 策略侧 PPO + KL 惩罚单元测试（路线图 P3-C-e；brief 指定文件名
test_rl_ppo_policy —— 即路线图里示例的 test_rl_ppo_clip）。

覆盖（路线图 P3-C-e ①-⑥）＋封闭形式数值验证：
  1. 手算裁剪代理目标（r < 1-ε / 带内 / r > 1+ε × A 正负，参数化 6 例）
  2. logp_new 用掩码后分布（非法动作概率 0、熵 0、KL 不被巨 logit 带偏）
  3. logp_old = 两级温度后采样分布的 log-prob（_temperature_sample 单点验证 +
     未温度化反例 + s<=0 退化分支）
  4. A = z - v_old 的 minibatch 标准化（总体 std、全同批次 / batch=1 不 NaN）
  5. 8 路增强：动作↔掩码同一 D4 置换（对齐 legacy rot90 oracle）、变换后动作
     仍合法、pass 槽还原、旧 2 元组调用不受影响
  6. 源断言：train_epochs 走 _ppo_policy_loss、无 MSE、无 CE 回归、无 pi_t/logq；
     采集端 _temperature_sample + play 回退重算；argparse 三参数仍在
  7. k1 KL = 封闭形式 KL(p_old‖p_new)（分类分布枚举；mean 不是 sum、方向不反）
  8. β 自适应方向 + 上下夹紧 + 溢出安全 + kl-target 单调（路线图 ④）
  9. running-KL > 2×kl-target 提前中止（剩余 minibatch 未处理 + 截断日志 +
     只断本轮、下一轮继续）
 10. ratio/KL 锚定行为 logp_old（用当前网络 logp 顶替 → KL 恒 0 = 变异红）
 11. 全链路冒烟：7 元组行 → _process_game_data → train_epochs → _ppo_stats，
     且 β 状态挂 ai（不从 --kl-coef 重读）
"""
import argparse
import inspect
import math
import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.selfplay_train as st  # noqa: E402
from scripts.selfplay_train import (  # noqa: E402
    _KL_BETA_MAX, _KL_BETA_MIN, _adapt_kl_coef, _compute_advantage,
    _ppo_policy_loss, _process_game_data, _standardize_advantage,
    _temperature_sample, augment8, train_epochs)


class _TinyNet(nn.Module):
    """最小可训练网络：(B,12,n,n) -> (policy (B,A), value (B,1))。"""

    def __init__(self, board=3, actions=5):
        super().__init__()
        self.c = nn.Conv2d(12, 8, 3, padding=1)
        self.p = nn.Linear(8 * board * board, actions)
        self.v = nn.Linear(8 * board * board, 1)

    def forward(self, x):
        h = F.relu(self.c(x))
        h = h.flatten(1)
        return self.p(h), self.v(h)


def _args(**over):
    base = dict(
        batch_size=4, epochs=1, lr=1e-2, weight_decay=1e-4, value_lr_mult=0.5,
        clip_grad=1.0, use_ema=0,
        ppo_clip=0.2, kl_coef=0.01, kl_target=0.01,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _buffer(n=16, board=3, actions=5, seed=0, logp_old=0.0):
    """P3-C 6 元组 buffer。logp_old=0.0（声称 p≈1）→ 与网络 logp 差出大正 KL，
    专供提前中止测试；其他场景传自己的值。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        planes = rng.random((12, board, board), dtype=np.float32)
        action = int(rng.integers(0, actions))
        z = np.float32(rng.uniform(-1, 1))
        v_old = np.float32(rng.uniform(-1, 1))
        mask = np.ones(actions, dtype=bool)
        out.append((planes, action, float(logp_old), z, v_old, mask))
    return out


def _make_ai(seed=0, board=3, actions=5):
    torch.manual_seed(seed)
    return types.SimpleNamespace(model=_TinyNet(board=board, actions=actions))


def _count_steps(buf, args, ai):
    """包一层 AdamW.step 计数跑 train_epochs，返回 optimizer step 次数。"""
    calls = {'n': 0}
    orig = torch.optim.AdamW

    class Counting(orig):
        def step(self, *a, **k):
            calls['n'] += 1
            return super().step(*a, **k)

    torch.optim.AdamW = Counting
    try:
        train_epochs(ai, buf, args, 'cpu')
    finally:
        torch.optim.AdamW = orig
    return calls['n']


# --------------------------------------------------------------------------- #
# 1. 手算裁剪代理目标
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('r,a,expect_loss,expect_clip', [
    (0.5, 2.0, -1.0, 1.0),    # r<1-ε、A>0：surr1=r·A=1.0 被选中（未触发裁剪）
    (0.5, -2.0, 1.6, 1.0),    # r<1-ε、A<0：surr2=0.8·(-2)=-1.6 被裁剪选中
    (1.0, 2.0, -2.0, 0.0),    # 带内：min(r·A, clip·A)=2 → loss=-2
    (0.9, 2.0, -1.8, 0.0),    # 带内 r<1：r·A=1.8
    (1.5, 2.0, -2.4, 1.0),    # r>1+ε、A>0：surr2=1.2·2=2.4 被裁剪选中
    (1.5, -2.0, 3.0, 1.0),    # r>1+ε、A<0：surr1=-3 更小 → loss=+3（不被裁剪）
])
def test_clipped_surrogate_hand_computed(r, a, expect_loss, expect_clip):
    """路线图①：L_pi = -min(r·A, clip(r,1-ε,1+ε)·A)，与手算逐项对齐。"""
    p0 = 0.7
    logits = torch.tensor([[0.0, math.log((1 - p0) / p0)]], dtype=torch.float32)
    mask = torch.ones(1, 2, dtype=torch.bool)
    action = torch.tensor([0])
    # 构造 logp_new(a)=log(p0)，则 ratio = exp(logp_new-logp_old) = r 精确成立
    logp_old = torch.tensor([math.log(p0) - math.log(r)], dtype=torch.float32)
    adv = torch.tensor([a], dtype=torch.float32)
    loss, stats = _ppo_policy_loss(logits, mask, action, logp_old, adv,
                                   clip_eps=0.2, kl_coef=0.0)
    assert loss.item() == pytest.approx(expect_loss, abs=1e-4)
    assert stats['clip_frac'] == pytest.approx(expect_clip)
    # kl_coef=0 → KL 项完全退出；KL 本身 = logp_old - logp_new(a) = -ln r
    assert stats['kl'] == pytest.approx(-math.log(r), abs=1e-5)


# --------------------------------------------------------------------------- #
# 2. 掩码后分布
# --------------------------------------------------------------------------- #
def test_masked_distribution_is_legal_only():
    """路线图②：logp_new 取掩码后分布 —— 非法动作概率 0、熵 0、KL 不被带偏。"""
    mask = torch.tensor([[True, False]])
    action = torch.tensor([0])
    logp_old = torch.tensor([-0.1])
    adv = torch.tensor([1.0])
    loss, stats = _ppo_policy_loss(torch.tensor([[0.0, 1000.0]]), mask,
                                   action, logp_old, adv, 0.2, 0.0)
    assert torch.isfinite(loss)
    # 掩码生效：只剩合法动作 → 确定分布 → 熵 0；logp_new(a)=0 → KL=logp_old
    assert stats['entropy'] == pytest.approx(0.0, abs=1e-6)
    assert stats['kl'] == pytest.approx(-0.1, abs=1e-5)
    # ratio = exp(0-(-0.1)) = 1.105 ∈ [0.8, 1.2] → 不裁剪
    assert stats['clip_frac'] == 0.0
    # 非法动作的 logit 值任意（0 vs 1000）都不该影响任何统计 —— 未掩码时必红
    _, stats_b = _ppo_policy_loss(torch.tensor([[0.0, 50.0]]), mask,
                                  action, logp_old, adv, 0.2, 0.0)
    assert stats_b['entropy'] == pytest.approx(stats['entropy'], abs=1e-6)
    assert stats_b['kl'] == pytest.approx(stats['kl'], abs=1e-6)


# --------------------------------------------------------------------------- #
# 3. logp_old = 温度作用后采样分布
# --------------------------------------------------------------------------- #
def test_logp_old_is_post_temperature_sampling_prob():
    """路线图③：logp_old 必须是两级温度后分布的 log-prob（误记未温度化值 → 红）。"""
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    state = np.random.get_state()
    try:
        np.random.seed(123)
        mv0, logp0, pn0 = _temperature_sample(probs, mc=0)
        assert np.allclose(pn0, probs)                        # temp=1 → 分布不动
        assert logp0 == pytest.approx(math.log(probs[mv0]))   # 与采样分布一致

        np.random.seed(123)
        mv1, logp1, pn1 = _temperature_sample(probs, mc=60)   # temp=0.1 → p^10
        manual = probs ** 10.0
        manual /= manual.sum()
        assert np.allclose(pn1, manual)                       # 二级温度确实作用
        assert logp1 == pytest.approx(math.log(manual[mv1]))
        # 反例：未温度化的 probs[mv] 与本值不同（变异自检）
        assert logp1 != pytest.approx(math.log(probs[mv1]), abs=1e-4)
    finally:
        np.random.set_state(state)
    # s<=0 退化分支：强制 pass、不消耗 RNG、log-prob 记 0.0（旧行为逐位一致）
    mv2, logp2, pn2 = _temperature_sample(np.zeros(4), mc=5)
    assert mv2 == 3 and logp2 == 0.0 and np.all(pn2 == 0)


# --------------------------------------------------------------------------- #
# 4. 优势标准化
# --------------------------------------------------------------------------- #
def test_advantage_standardization_population_std():
    """路线图④：A=z-v_old 按 minibatch 标准化；总体 std、夹紧防 0 除。"""
    out = _standardize_advantage(torch.tensor([1.0, 2.0, 3.0]))
    # (x-2)/sqrt(2/3)：mean=2、总体 std=0.8164966
    expect = torch.tensor([-1.2247449, 0.0, 1.2247449])
    assert torch.allclose(out, expect, atol=1e-6)

    # 全同批次（std=0）→ 夹紧 → 全 0、有限（0 除 → NaN/Inf = 红）
    out2 = _standardize_advantage(torch.full((4,), 7.0))
    assert torch.isfinite(out2).all() and torch.all(out2 == 0)

    # batch=1：样本 std(ddof=1) 是 NaN，夹紧救不了 → 必须有限且为 0
    out3 = _standardize_advantage(torch.tensor([0.5]))
    assert torch.isfinite(out3).all() and out3.item() == pytest.approx(0.0)

    # 随机批次：center 0、scale 1（除错分母 → 红）
    out4 = _standardize_advantage(torch.randn(64))
    assert abs(float(out4.mean())) < 1e-5
    assert abs(float(out4.std(unbiased=False)) - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
# 5. 8 路增强：动作 ↔ 掩码同一置换
# --------------------------------------------------------------------------- #
def test_augment8_permutes_action_with_mask():
    """路线图⑤：动作与掩码走同一 D4 变换；变换后动作仍合法；pass 槽还原。"""
    n = 5
    rng = np.random.default_rng(0)
    plane = rng.normal(size=(12, n, n)).astype(np.float32)
    mask = np.zeros(n * n + 1, dtype=bool)
    mask[1 * n + 2] = True   # (1,2) 非中心点（对称变换下必移动）
    mask[0 * n + 1] = True   # (0,1) 另一非对称点
    mask[n * n] = True       # pass 恒合法

    legacy = augment8(plane, mask, n)      # 旧 2 元组调用不受影响
    assert len(legacy) == 8 and len(legacy[0]) == 2

    for act in (1 * n + 2, 0 * n + 1, n * n):
        out = augment8(plane, mask, n, action=act)
        assert len(out) == 8 and all(len(t) == 3 for t in out)
        for t, (pl, mk, ac) in enumerate(out):
            # 平面/掩码输出与旧 2 元组路径逐位一致（同一引擎同一 t）
            np.testing.assert_array_equal(pl, legacy[t][0])
            np.testing.assert_array_equal(mk, legacy[t][1])
            # 变换后动作仍落在变换后的合法掩码内
            assert mk[ac], f"act {act} → t={t} 得 {ac}，不在变换后掩码内"
            if act == n * n:
                assert ac == n * n, "pass 槽必须原样还原"
            else:
                # 与 legacy rot90 oracle（_legacy_augment8 同式）对齐
                r, c = divmod(act, n)
                g = np.zeros((n, n), np.float32)
                g[r, c] = 1.0
                if t >= 4:
                    g = g[:, ::-1]
                k = t % 4
                if k > 0:
                    g = np.rot90(g, k=-k)
                rr, cc = np.unravel_index(g.argmax(), (n, n))
                assert ac == rr * n + cc, f"act {act} t={t} 与 oracle 不符"
                # 非恒等变换必须真的移动了动作（动作没被置换 → 红）
                if t == 1:
                    assert ac != act


def test_advantage_is_z_minus_v_old():
    """D13④：A = z - v_old（不是 z、也不是 z - 均值）。减 v_old 必须真被读到。

    变异防线：把 train_epochs 里的 `z - v_old` 改成只用 `z`，标准化之后仍是一个
    自洽的分布 → 没有任何统计量会变，只有这一格显式比对能变红。
    """
    z = torch.tensor([1.0, 0.0, -1.0, 0.5])
    v_old = torch.tensor([0.5, 0.0, 0.5, -0.25])
    raw = torch.tensor([0.5, 0.0, -1.5, 0.75])
    adv = _compute_advantage(z, v_old)
    assert torch.allclose(adv, _standardize_advantage(raw), atol=1e-6)
    # 与「只用 z」不同 → 接线被删则红
    assert not torch.allclose(adv, _standardize_advantage(z), atol=1e-3)
    # 与「z 减 v_old 的均值」不同（那是把 v_old 当偏置吸收掉，语义更弱）
    assert not torch.allclose(adv, _standardize_advantage(z - v_old.mean()), atol=1e-3)


# --------------------------------------------------------------------------- #
# 5b. 采集端：数据行里的 logp_old 真的锚定「温度作用后」分布
# --------------------------------------------------------------------------- #
def _post_temperature_pnorm(probs, mc):
    """独立复算两级温度后的行为分布（不复用 _temperature_sample，避免自证）。"""
    p = np.asarray(probs, dtype=np.float64).reshape(-1).copy()
    p[-1] = max(p[-1], 0.0)
    temp = 1.0 - min(1.0, mc / max(30, 1)) * (1.0 - 0.1)
    if temp > 0 and temp != 1.0:
        p = p ** (1.0 / temp)
    return p / p.sum()


def test_self_play_game_rows_carry_post_temperature_logp_old(monkeypatch):
    """采集行契约：logp_old = log(温度作用后分布)[**实际落子**的 action]。

    这是 P3-0 报告 §5.5-1 点名的陷阱：改记「未加温度的 visit 分布」会让 KL/ratio
    系统性偏高且**不报错**。既有测试只钉住 _temperature_sample 这个 helper，
    这里补上**调用点**（self_play_game → data.append）：变异把 append 里的
    logp_old 换成原始 visit 分布 / 换成别的着法的 log-prob，本测试必红。
    """
    bs, n_actions = 5, 5 * 5 + 1
    seen = []

    class _StubMCTS:
        def __init__(self, ai, **kw):
            self.k = 0

        def search(self, board, h0, h1, to_play, simulations, path_moves):
            k = self.k
            self.k += 1
            # 逐步偏斜的固定分布（全部 > 0 → p**10 不下溢、logp 恒有定义）
            probs = np.full(n_actions, 1e-3)
            probs[(k * 3) % (n_actions - 1)] = 0.30
            probs[(k * 7 + 1) % (n_actions - 1)] = 0.15
            probs[-1] = 0.05
            seen.append(probs.copy())
            return np.zeros(n_actions), probs, 0.25

    monkeypatch.setattr(st, 'MCTS', _StubMCTS)
    state = np.random.get_state()
    try:
        np.random.seed(7)
        data, score = st.self_play_game(
            None, board_size=bs, sims=1, max_moves=4, temperature=1.0,
            expand_topk=4, expand_chunk=0)
    finally:
        np.random.set_state(state)

    assert len(data) == 4, f"应记录 4 步，实得 {len(data)}"
    for i, row in enumerate(data):
        assert len(row) == 7, f"采集行应为 7 元组，实得 {len(row)}"
        planes, action, logp_old, to_play, mc, root_value, mask = row
        assert planes.shape == (12, bs, bs)
        assert mask.shape == (n_actions,) and mask.dtype == np.bool_
        assert mask[action], f"第 {i} 步落子 {action} 不在自身掩码内"
        assert 0 <= action < n_actions and to_play in (1, -1)
        p_norm = _post_temperature_pnorm(seen[mc], mc)
        # 核心断言：logp_old 锚定在**实际落子**动作上，不是别的着法、也不是原始 visit 分布
        assert logp_old == pytest.approx(math.log(p_norm[action]), abs=1e-6), (
            f"第 {i} 步 logp_old={logp_old} 与温度后分布 log(p_norm[{action}])="
            f"{math.log(p_norm[action])} 不符")
        # 反例：记成未温度化的 visit 分布（mc>0 时第二级温度才真的作用 → 必不同；
        # mc=0 时 temp≡1，两级温度退化为恒等，该反例不成立，跳过）
        raw = seen[mc] / seen[mc].sum()
        if mc > 0:
            assert logp_old != pytest.approx(math.log(raw[action]), abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. 源断言
# --------------------------------------------------------------------------- #
def test_train_epochs_source_wires_ppo_only():
    """路线图⑥：源级断言 —— 策略侧走 PPO 裁剪+KL，无 MSE/CE 回归残留。"""
    src = inspect.getsource(st.train_epochs)
    assert '_ppo_policy_loss(' in src, "loss 未走 _ppo_policy_loss"
    assert '_compute_advantage(' in src, "A 未走 _compute_advantage（z - v_old）"
    assert '_adapt_kl_coef(' in src, "缺 β 自适应"
    assert '2.0 * kl_target' in src, "缺 2×kl-target 提前中止条件"
    # 旧实现残留 = 变异红
    assert 'F.mse_loss' not in src, "策略侧回退成 MSE 回归"
    assert 'logq' not in src, "旧 CE 回归的 logq 残留"
    assert 'pi_t' not in src, "PPO 后 pi_t 目标已移除"
    # value 侧：P3-D 已把 BCE 换成 MSE + PPO value clipping（helper 实现，见
    # test_rl_ppo_value.py）。这里锁「BCE 已彻底消失」+「loss_v 走新 helper」——
    # 回退成 BCE 会让下面两条同时红（本文件此前锁的是反面：BCE 必须还在）。
    assert 'binary_cross_entropy' not in src, "value 侧回退成 BCE（P3-D 已删）"
    assert '_ppo_value_loss(' in src, "loss_v 未走 _ppo_value_loss"

    msrc = inspect.getsource(st.main)
    for flag in ('--ppo-clip', '--kl-coef', '--kl-target'):
        assert flag in msrc, f"argparse 丢失 {flag}"
    # swanlab 新增三键、既有键不动
    assert '"entropy": _ppo_stats["entropy"]' in msrc
    assert '"ppo_clip": args.ppo_clip' in msrc

    # 采集端：logp_old 取自 _temperature_sample，play 回退后重算
    ssrc = inspect.getsource(st.self_play_game)
    assert '_temperature_sample(' in ssrc
    assert 'if action != mv:' in ssrc, "play 回退 pass 后未重算 logp_old"
    # 回退分支必须重算**同一个 p_norm** 的 log-prob（动作与 logp_old 锚定同一次采样）
    assert 'logp_old = float(np.log(p_norm[action]))' in ssrc
    # append 进 data 的必须是那个 logp_old 变量本身
    assert 'float(root_value), mask))' in ssrc
    # async 端同一契约（路线图要求同步 async_pipeline.py 调用点）
    import scripts.async_pipeline as ap
    asrc = inspect.getsource(ap.SelfPlayWorker._play_one_game)
    assert '_temperature_sample(' in asrc
    assert 'mask = np.concatenate' in asrc
    # async 模式 logp_old 来自冻结生成策略（P3-0 §5.5-5）→ 必须有一次性告警，
    # 否则 ratio 起点≠1 与 KL 记账退化是**静默**的
    assert '_ppo_async_warned' in msrc, "async 模式缺 off-policy logp_old 告警"
    assert 'off-policy' in msrc


# --------------------------------------------------------------------------- #
# 7. k1 KL 封闭形式验证
# --------------------------------------------------------------------------- #
def test_k1_kl_matches_closed_form():
    """k1 估计器 mean(logp_old-logp_new)[a] ≡ 封闭形式 KL(p_old‖p_new)。"""
    p_old = np.array([0.5, 0.25, 0.15, 0.10])
    counts = (p_old * 20).astype(int)              # 10,5,3,2 → 按 p_old 分层
    actions = np.repeat(np.arange(4), counts)
    logits = torch.tensor([[0.3, -0.7, 1.2, 0.0]], dtype=torch.float32)
    logits_b = logits.expand(20, 4).contiguous()
    mask_b = torch.ones(20, 4, dtype=torch.bool)
    logp_old_b = torch.tensor(np.log(p_old[actions]), dtype=torch.float32)
    adv = torch.zeros(20)                          # adv=0 → loss = -β·KL
    loss, stats = _ppo_policy_loss(logits_b, mask_b, torch.tensor(actions),
                                   logp_old_b, adv, clip_eps=0.2, kl_coef=1.0)

    q = F.softmax(logits, dim=-1).squeeze(0).numpy()
    exact = float((p_old * np.log(p_old / q)).sum())
    assert exact > 0, "两个分布必须不同"
    # mean 不是 sum：sum 会放大 20 倍 → 红
    assert loss.item() == pytest.approx(-exact, abs=1e-5)
    assert stats['kl'] == pytest.approx(exact, abs=1e-5)
    # 方向不反：KL(p_new‖p_old) ≠ KL(p_old‖p_new)，实现取反 → 红
    reverse = float((q * np.log(q / p_old)).sum())
    assert abs(exact - reverse) > 1e-3


# --------------------------------------------------------------------------- #
# 8. β 自适应
# --------------------------------------------------------------------------- #
def test_kl_beta_adaptation_direction_and_clamp():
    """β ← clamp(β·exp((KL-target)/target))：方向、夹紧字面值、溢出安全。"""
    b0 = 0.01
    b_up = _adapt_kl_coef(b0, 0.03, 0.01)           # KL > target → 收紧
    assert b_up == pytest.approx(b0 * math.exp(2.0))
    assert b_up > b0
    b_dn = _adapt_kl_coef(b0, 0.005, 0.01)          # KL < target → 放松
    assert b_dn == pytest.approx(b0 * math.exp(-0.5))
    assert b_dn < b0
    # 夹紧边界（字面值钉死；规格未给边界 → 报告标注歧义）
    assert _KL_BETA_MIN == 1e-4 and _KL_BETA_MAX == 10.0
    assert _adapt_kl_coef(b0, 1e6, 0.01) == _KL_BETA_MAX
    assert _adapt_kl_coef(b0, -1e6, 0.01) == _KL_BETA_MIN
    # 溢出安全：(1e300-0.01)/0.01 直接 exp 会 OverflowWarning → 夹指数
    assert math.isfinite(_adapt_kl_coef(0.01, 1e300, 0.01))
    # kl_target<=0 → 防除零，β 不动
    assert _adapt_kl_coef(0.07, 1.0, 0.0) == 0.07


def test_larger_kl_target_smaller_penalty_step():
    """路线图④：--kl-target 变大 → β 自适应步长变小（KL 惩罚项变小），单调。"""
    kl_obs = 0.05
    betas = [_adapt_kl_coef(0.01, kl_obs, t) for t in (0.01, 0.05, 0.2)]
    assert betas[0] > betas[1] > betas[2], betas


# --------------------------------------------------------------------------- #
# 9. 信任域提前中止
# --------------------------------------------------------------------------- #
def test_early_stop_truncates_remaining_minibatches(capsys):
    """running-KL > 2×kl-target → 打印截断并只断本轮剩余 minibatch。"""
    buf = _buffer(n=16, seed=5, logp_old=0.0)      # 行为声称 p≈1 → KL ≫ 0.02
    args = _args(batch_size=4, epochs=1, kl_target=0.01)
    ai = _make_ai(seed=5)
    steps = _count_steps(buf, args, ai)
    out = capsys.readouterr().out
    assert steps == 1, f"首步后应截断剩余 minibatch，实际 {steps} 步（全轮=4）"
    assert '信任域提前中止' in out
    assert ai._ppo_stats['early_stop'] is True
    assert ai._ppo_stats['kl'] > 2 * 0.01


def test_early_stop_control_runs_full_epoch(capsys):
    """对照组：阈值不可达 → 整轮 4 步跑满、无截断日志。"""
    buf = _buffer(n=16, seed=5, logp_old=0.0)
    args = _args(batch_size=4, epochs=1, kl_target=1e9)
    ai = _make_ai(seed=5)
    steps = _count_steps(buf, args, ai)
    assert steps == 4, f"期望整轮 4 步，实际 {steps}"
    assert ai._ppo_stats['early_stop'] is False
    assert '提前中止' not in capsys.readouterr().out


def test_early_stop_only_breaks_current_epoch():
    """截断的是「本轮剩余」：epochs=2 → 每轮各 1 步，共 2 步（不是整个调用）。"""
    buf = _buffer(n=16, seed=5, logp_old=0.0)
    args = _args(batch_size=4, epochs=2, kl_target=0.01)
    ai = _make_ai(seed=5)
    steps = _count_steps(buf, args, ai)
    assert steps == 2, f"期望 2 轮各 1 步 = 2 步，实际 {steps}"


# --------------------------------------------------------------------------- #
# 10. ratio/KL 锚定行为 logp_old
# --------------------------------------------------------------------------- #
def test_ratio_and_kl_anchor_behavior_logp():
    """变异防线：用当前网络 logp 顶替 logp_old → KL 恒 0 → 本测试必红。"""
    p0 = 0.6
    logits = torch.tensor([[0.0, math.log((1 - p0) / p0)]], dtype=torch.float32)
    mask = torch.ones(1, 2, dtype=torch.bool)
    action = torch.tensor([0])
    adv = torch.tensor([0.0])

    # logp_old = 行为值 = log(0.6) → KL=0
    _, st0 = _ppo_policy_loss(logits, mask, action,
                              torch.tensor([math.log(0.6)]), adv, 0.2, 1.0)
    assert st0['kl'] == pytest.approx(0.0, abs=1e-6)
    assert st0['clip_frac'] == 0.0                    # ratio=1 在带内

    # logp_old = log(0.3) → KL = ln0.3 - ln0.6 = -ln2（锚定的是传入的行为值）
    _, st1 = _ppo_policy_loss(logits, mask, action,
                              torch.tensor([math.log(0.3)]), adv, 0.2, 1.0)
    assert st1['kl'] == pytest.approx(-math.log(2.0), abs=1e-5)

    # ratio 同样锚定 logp_old：r = 0.6/0.3 = 2 > 1.2 → A=1 时 loss=-1.2
    loss2, st2 = _ppo_policy_loss(logits, mask, action,
                                  torch.tensor([math.log(0.3)]),
                                  torch.tensor([1.0]), 0.2, 0.0)
    assert loss2.item() == pytest.approx(-1.2, abs=1e-4)
    assert st2['clip_frac'] == 1.0


# --------------------------------------------------------------------------- #
# 10b. buffer 列 → 张量 的接线（logp_old / mask / action 各走对列）
# --------------------------------------------------------------------------- #
def test_buffer_columns_wired_to_right_tensors():
    """变异防线：把 logp_old 读成 z（列错位）必须变红。

    做法：把网络 value head 清零 → v_new ≡ 0、z ≡ 1、v_old ≡ 0、ε=0.2 →
    L_v = max((0-1)², (clip(0, 0±0.2)-1)²) = max(1, 0.64) = 1.0（P3-D 的
    MSE+clip，常数项，与 BCE 时代的 log2 无关）；z 恒 1 → A 标准化后全 0 →
    策略代理项恒 0。于是返回的 loss 精确等于
    `1.0 - β·mean(logp_old - logp_new[a])`，每一列的接线都参与这个数：
    logp_old 读错列 / action 读错列 / mask 读错列 → 立刻对不上。
    """
    n, board, actions = 8, 3, 5
    rng = np.random.default_rng(3)
    logp_old_col = np.log(np.array([0.05, 0.07, 0.11, 0.13, 0.2, 0.3, 0.4, 0.5]))
    buf = []
    for i in range(n):
        buf.append((rng.random((12, board, board), dtype=np.float32),
                    int(rng.integers(0, actions)),   # 列 1: action
                    float(logp_old_col[i]),           # 列 2: logp_old（行为值）
                    np.float32(1.0),                  # 列 3: z → target=1、adv≡0
                    np.float32(0.0),                  # 列 4: v_old → 0
                    np.ones(actions, dtype=bool)))    # 列 5: mask
    args = _args(batch_size=n, epochs=1, kl_coef=0.01, kl_target=1e6)
    ai = _make_ai(seed=3, board=board, actions=actions)
    ai.model.v.weight.data.zero_()      # value ≡ 0 → L_v = max(1, 0.64) = 1.0（P3-D）
    ai.model.v.bias.data.zero_()

    state = np.random.get_state()
    try:
        np.random.seed(3)
        # train_epochs 用 np.random.randint **有放回**抽 minibatch → 必须复现同一次抽
        idx = np.random.randint(0, n, size=n)
        batch = [buf[i] for i in idx]
        planes = torch.from_numpy(np.stack([b[0] for b in batch])).float()
        mask = torch.from_numpy(np.stack([b[5] for b in batch]))
        actions_t = torch.tensor([b[1] for b in batch], dtype=torch.long)
        lp_old = torch.from_numpy(
            np.asarray([b[2] for b in batch], dtype=np.float32))
        with torch.no_grad():
            policy, _ = ai.model(planes)
            lp_new = torch.log_softmax(
                policy.float().masked_fill(~mask, float('-inf')), -1)
            lp_new_a = lp_new.gather(1, actions_t.view(-1, 1)).squeeze(1)
        expect_kl = float((lp_old - lp_new_a).mean().item())
        # 策略项（A≡0 → 代理项恒 0）只剩 -β·KL；value 项 = 1.0（P3-D 的 MSE+clip，
        # 见 docstring 手算）。旧公式的 log2 其实是 BCE 的 value 项，已被 1.0 取代。
        expect_loss = 1.0 - 0.01 * expect_kl

        np.random.seed(3)
        loss = train_epochs(ai, buf, args, 'cpu')
    finally:
        np.random.set_state(state)
    assert loss == pytest.approx(expect_loss, abs=1e-4), (
        f"loss={loss} 与手算 {expect_loss} 不符 → buffer 列接线错位")
    # clip_frac 与手算一致（clip 上下界取自 --ppo-clip=0.2；A≡0 不影响 ratio）
    ratio = torch.exp(lp_new_a - lp_old)
    expect_clip = float(((ratio < 0.8) | (ratio > 1.2)).float().mean().item())
    assert ai._ppo_stats['clip_frac'] == pytest.approx(expect_clip, abs=1e-6)


# --------------------------------------------------------------------------- #
# 11. 全链路冒烟：7 元组行 → buffer → train_epochs → stats + β 持久化
# --------------------------------------------------------------------------- #
def _game_rows(bs=5, steps=4, seed=9):
    n_actions = bs * bs + 1
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(steps):
        planes = rng.normal(size=(12, bs, bs)).astype(np.float32)
        action = int(rng.integers(0, n_actions))
        logp_old = float(np.log(rng.random() + 1e-3))
        player = 1 if i % 2 == 0 else -1
        root_value = float(rng.uniform(-1, 1))
        mask = np.ones(n_actions, dtype=bool)
        cell = int(rng.integers(0, n_actions - 1))   # 不碰恒合法的 pass 槽
        mask[cell] = False
        if not mask[action]:
            mask[action] = True                      # 行为动作必须合法
        rows.append((planes, action, logp_old, player, i, root_value, mask))
    return rows, n_actions


def test_full_chain_7tuple_to_train_and_beta_persists(monkeypatch):
    """7 元组行 → 6 元组 buffer → 训练；β 挂 ai 不从 --kl-coef 重读。"""
    rows, n_actions = _game_rows()
    args = _args(no_augment=0, td=0, td_steps=3, td_alpha_init=0.2,
                 td_alpha_end=0.9)
    buf = []
    _process_game_data(rows, score=1, bs=5, n_actions=n_actions,
                       buffer=buf, args=args)
    assert len(buf) == 8 * len(rows)
    zs = []
    for i, r in enumerate(buf):
        assert len(r) == 6, f"buffer 行应为 6 元组，实得 {len(r)}"
        planes, action, logp_old, z, v_old, mask = r
        assert mask.shape == (n_actions,) and mask.dtype == np.bool_
        assert 0 <= action < n_actions and mask[action], "增强后动作不在掩码内"
        assert np.isfinite(logp_old) and -1.0 <= z <= 1.0
        zs.append(z)
    # 同一行的 8 份增强共享同一 z（z 与增强无关）
    assert len(set(zs)) == len(rows)

    ai = _make_ai(seed=11, board=5, actions=n_actions)
    ai._kl_beta = 0.05                              # 预置 β 状态
    seen = []
    real = st._adapt_kl_coef

    def spy(beta, kl_obs, kl_target):
        seen.append(float(beta))
        return real(beta, kl_obs, kl_target)

    monkeypatch.setattr(st, '_adapt_kl_coef', spy)
    loss = train_epochs(ai, buf, _args(batch_size=4, epochs=1, kl_coef=0.01),
                        'cpu')
    assert np.isfinite(loss)
    # 自适应收到的初值是挂 ai 的 0.05，不是 args.kl_coef=0.01（接线契约 §5.2）
    assert seen and seen[0] == pytest.approx(0.05)
    stats = ai._ppo_stats
    # 自适应结果回写 ai._kl_beta，且与 stats 快照一致（首轮 KL 大 → 朝上走）
    assert ai._kl_beta == stats['kl_coef'], "自适应结果必须回写 ai._kl_beta"
    assert ai._kl_beta != 0.05, "β 应已发生自适应移动"
    for k in ('kl', 'clip_frac', 'entropy', 'kl_coef', 'early_stop', 'steps'):
        assert k in stats, f"_ppo_stats 缺 {k}"
    assert stats['steps'] >= 1
    assert stats['entropy'] >= 0.0
