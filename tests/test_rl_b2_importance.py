"""B2 重要性修正与 8/7 元组链路（2026-09-30 RL 去 MCTS 的收尾部分）。

为什么需要这一整套：去 MCTS 之后，采集的行为分布 q（温度 + minimax 推演派生）
与 PPO ratio 两侧的 π_θold（无温度根 policy）**不再是同一个分布**。不修正就是
拿 off-policy 数据做 on-policy 比值 —— 不报错，只表现为「训不稳」。这组测试把
三个环节各自钉死：

  1. buffer 7 元组与 B2 列（logq 必在、位置固定）
  2. w = clip(π/q, 1/W, W) 的数值与**双向**截断
  3. w 乘在 surrogate 上、**不**乘在 KL 上
"""
import argparse
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.selfplay_train as st  # noqa: E402
from scripts.selfplay_train import (_importance_weight, _ppo_policy_loss,  # noqa: E402
                                    _process_game_data)

BS, N_ACTIONS = 3, 3 * 3 + 1


def _args(**kw):
    d = {'td': 0, 'no_augment': 1, 'td_steps': 3, 'td_alpha_init': 0.2,
         'td_alpha_end': 0.9}
    d.update(kw)
    return argparse.Namespace(**d)


def _game_row(mc=0, logp_old=-1.0, logq=-1.5, v=0.25):
    """一条 8 元组采集行。"""
    return (np.zeros((12, BS, BS), dtype=np.float32), 4, logp_old, 1, mc,
            v, np.ones(N_ACTIONS, dtype=np.bool_), logq)


# --------------------------------------------------------------------------- #
# 1. buffer 7 元组
# --------------------------------------------------------------------------- #
def test_process_game_data_writes_seven_tuple_with_logq():
    buffer = []
    _process_game_data([_game_row(mc=i) for i in range(3)], 1.0, BS, N_ACTIONS,
                       buffer, _args())
    assert len(buffer) == 3
    for row in buffer:
        assert len(row) == 7, f'buffer 行应为 7 元组，实得 {len(row)}'
        planes, action, logp_old, z, v_old, mask, logq = row
        assert planes.shape == (12, BS, BS)
        assert action == 4 and mask[action]
        assert logp_old == pytest.approx(-1.0)
        assert logq == pytest.approx(-1.5), 'logq 必须原样进 buffer（index 6）'
        # 第 5 列 = 采集期 V_θold(s)（改造前是 MCTS 根价值，位置不变语义变）
        assert v_old == pytest.approx(0.25)


def test_process_game_data_rejects_legacy_rows():
    """6 元组（去 MCTS 前）与 5 元组（PPO 前）必须被拒收并说清是哪一代。"""
    for legacy, hint in (
        (tuple(range(7)), 'logq'),
        (tuple(range(5)), 'PPO'),
    ):
        buffer = []
        with pytest.raises(ValueError) as ei:
            _process_game_data([legacy], 1.0, BS, N_ACTIONS, buffer, _args())
        assert hint in str(ei.value), \
            f'报错没指出布局代次（应含 {hint!r}）：{ei.value}'


def test_augment_keeps_logq_and_v_old_per_row():
    """8 对称增强后每条仍带同一 logq（对称不改变概率值）。"""
    buffer = []
    rows = [_game_row(mc=i, logq=-0.5 - i, v=0.1 * i) for i in range(2)]
    _process_game_data(rows, 1.0, BS, N_ACTIONS, buffer, _args(no_augment=0))
    assert len(buffer) == 16, f'2 行 × 8 增强 = 16，实得 {len(buffer)}'
    for k, row in enumerate(buffer):
        assert len(row) == 7
        assert row[6] == pytest.approx(-0.5 - (k // 8)), \
            '增强后的行必须仍带它自己那一行的 logq'
        assert row[4] == pytest.approx(0.1 * (k // 8))


# --------------------------------------------------------------------------- #
# 2. 权重本身
# --------------------------------------------------------------------------- #
def test_weight_is_exp_of_log_difference():
    logp = torch.tensor([math.log(0.4), math.log(0.1)])
    logq = torch.tensor([math.log(0.2), math.log(0.2)])
    w = _importance_weight(logp, logq, 20.0)
    assert torch.allclose(w, torch.tensor([2.0, 0.5]), atol=1e-6), w


def test_weight_is_clipped_on_both_sides():
    """w 顶到 W 与掉到 1/W 都要被截断；单侧截断会留下失控的样本。"""
    W = 4.0
    logp = torch.tensor([0.0, 0.0, 0.0])
    logq = torch.tensor([-10.0, 0.0, 10.0])       # w = e^10, 1, e^-10
    w = _importance_weight(logp, logq, W)
    assert float(w[0]) == pytest.approx(W), '上界未截断'
    assert float(w[1]) == pytest.approx(1.0), 'w=1 的样本不该被改动'
    assert float(w[2]) == pytest.approx(1.0 / W), '下界未截断'


def test_weight_survives_extreme_logq_without_inf():
    """q 的尾部可以低到 exp(-700) 以下：先除会 inf/nan，log 域不会。"""
    logp = torch.tensor([0.0])
    logq = torch.tensor([-800.0])
    w = _importance_weight(logp, logq, 20.0)
    assert torch.isfinite(w).all(), f'权重溢出：{w}'
    assert float(w[0]) == pytest.approx(20.0)


def test_weight_max_one_means_no_correction():
    """W=1 ⇒ w ≡ 1：与「不做修正」逐位等价（这正是回退开关的语义）。"""
    logp = torch.tensor([0.3, -1.0])
    logq = torch.tensor([-0.2, 0.5])
    w = _importance_weight(logp, logq, 1.0)
    assert torch.allclose(w, torch.ones(2), atol=0), w


# --------------------------------------------------------------------------- #
# 3. w 乘 surrogate、不乘 KL
# --------------------------------------------------------------------------- #
def _loss_with_and_without(logq, w_max=20.0):
    torch.manual_seed(5)
    logits = torch.randn(4, N_ACTIONS, requires_grad=True)
    mask = torch.ones(4, N_ACTIONS, dtype=torch.bool)
    action = torch.tensor([0, 1, 2, 3])
    logp_old = torch.tensor([-1.0, -1.2, -0.8, -2.0])
    adv = torch.tensor([1.0, -0.5, 0.25, -0.75])
    return _ppo_policy_loss(logits, mask, action, logp_old, adv, 0.2, 0.01,
                            logq=logq, w_max=w_max)


def test_w_scales_the_surrogate_only():
    """KL 项**不乘 w**：它是「拉回 π_θold」的正则，不是「拉回 q」。"""
    # logp_old ≈ -1.0..-2.0；q = 1e-6 ⇒ logw ≈ +11~+12 ≫ log(20) ⇒ w 顶到上界
    logq = torch.tensor([math.log(1e-6)] * 4)
    loss_w, st_w = _loss_with_and_without(logq, w_max=20.0)
    loss_1, st_1 = _loss_with_and_without(None)
    assert st_w['w_clip_frac'] == 1.0, \
        f'w 应全部触顶（logw≈+11），实得 clip_frac={st_w["w_clip_frac"]}'
    assert st_w['w_mean'] == pytest.approx(20.0, rel=1e-6)
    assert st_1['w_mean'] == pytest.approx(1.0)

    # 手算：loss = -mean(w·min(...)) - β·KL
    torch.manual_seed(5)
    logits = torch.randn(4, N_ACTIONS)
    lp_new = torch.log_softmax(logits, dim=-1)
    lp_a = lp_new.gather(1, torch.tensor([[0], [1], [2], [3]])).squeeze(1)
    logp_old = torch.tensor([-1.0, -1.2, -0.8, -2.0])
    adv = torch.tensor([1.0, -0.5, 0.25, -0.75])
    ratio = torch.exp(lp_a - logp_old)
    w = torch.full((4,), 20.0)
    surr = torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
    expect = -(w * surr).mean() - 0.01 * (logp_old - lp_a).mean()
    assert float(loss_w) == pytest.approx(float(expect), rel=1e-5), \
        f'loss={float(loss_w)} 期望 {float(expect)}（KL 被乘了 w？）'

    # 反例：若 KL 也乘 w，loss 会更负 —— 明确排除
    wrong = -(w * surr).mean() - 0.01 * (w * (logp_old - lp_a)).mean()
    assert abs(float(loss_w) - float(wrong)) > 1e-4, 'KL 项与被乘 w 的版本不可区分'


def test_no_logq_reproduces_the_pre_b2_loss_exactly():
    """logq=None ⇒ 与去 MCTS 之前（2026-09-30 改造前）的 loss 逐位相同。"""
    torch.manual_seed(9)
    logits = torch.randn(6, N_ACTIONS)
    mask = torch.ones(6, N_ACTIONS, dtype=torch.bool)
    action = torch.randint(0, N_ACTIONS, (6,))
    logp_old = -torch.rand(6) - 0.5
    adv = torch.randn(6)
    loss_none, st_none = _ppo_policy_loss(logits, mask, action, logp_old, adv,
                                          0.2, 0.03, logq=None)
    # 同参数、logq ≡ logp_old ⇒ w ≡ 1，结果必须一样
    loss_one, st_one = _ppo_policy_loss(logits, mask, action, logp_old, adv,
                                        0.2, 0.03, logq=logp_old, w_max=20.0)
    assert float(loss_none) == pytest.approx(float(loss_one), rel=1e-6)
    assert st_one['w_mean'] == pytest.approx(1.0, rel=1e-5)


def test_stats_expose_the_weight_distribution():
    """w 的分布必须在 stats 里看得见（w≡1 或全触顶都是需要知道的事）。"""
    _l, s = _loss_with_and_without(torch.tensor([math.log(1e-9)] * 4), 20.0)
    assert s['w_mean'] > 19.0 and s['w_clip_frac'] == 1.0
    # logq ≡ logp_old ⇒ w ≡ 1、无一被截断
    _l, s2 = _loss_with_and_without(
        torch.tensor([-1.0, -1.2, -0.8, -2.0]), 20.0)
    assert s2['w_mean'] == pytest.approx(1.0, rel=1e-5)
    assert s2['w_clip_frac'] == 0.0


# --------------------------------------------------------------------------- #
# 4. train_epochs 接线
# --------------------------------------------------------------------------- #
def test_train_epochs_rejects_six_tuple_buffer():
    import inspect
    src = inspect.getsource(st.train_epochs)
    assert 'len(buffer[0]) != 7' in src, 'buffer 校验必须按 7 元组'
    assert "'v_clip_frac': stat_vclip" in src, '既有统计键不能丢'
    assert "'w_mean'" in src and "'w_clip_frac'" in src, 'B2 统计没接线'
    assert 'logq=logq_t' in src and 'w_max=w_max' in src, \
        'B2 权重没传进 _ppo_policy_loss'


def test_importance_weight_max_argument_exists_and_validated():
    import inspect
    src = inspect.getsource(st.train_epochs)
    assert "importance_weight_max" in src, '缺 --importance-weight-max 读取'
    assert 'w_max < 1.0' in src, 'W < 1 未被拒（W<1 会让截断区间为空）'


def test_async_pipeline_uses_sampler_not_mcts():
    """异步采集必须与串行同一契约：无 MCTS、8 元组、logq 回退按 π 重算。"""
    import inspect
    from scripts.async_pipeline import SelfPlayWorker
    src = inspect.getsource(SelfPlayWorker._play_one_game)
    assert 'sample_move(' in src, '异步采集未走子器'
    assert 'MCTS(' not in src, '异步采集仍在构造 MCTS'
    assert 's.policy[action]' in src and 's.probs[action]' in src, \
        '回退分支必须分别按 π 与 q 重算'
    assert 'mask, logq))' in src, '异步采集行末位必须是 logq（8 元组）'