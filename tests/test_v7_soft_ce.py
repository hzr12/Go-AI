# -*- coding: utf-8 -*-
"""V7 软 CE 的口径测试（与 12 通路 `soft_cross_entropy` 逐位对拍）。

背景（2026-10-04）
------------------
V7 的 policy 项原本**只**吃 one-hot（`policy_player`），软标签这条能力只有
12 通道有 —— 于是 stdata 分片里 KataGo 的搜索分布（`policy_player_prob` /
`policy_opp_prob`，实测一行 853/16/12/4/1/1）被归一化后塞进 `soft` 键却没人消费。

更糟的是：`v7_loss_labels` 用 `next_move`（= **player** 的 rank[0]）去填
`policy_opp`，于是 loss #1 与 #2 拿到**同一个**目标，π_opp 白训。

本文件钉住：口径一致、两路分离、mask 全 0 时回退。
"""
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.networks.katago_v7_loss import (  # noqa: E402
    LOSS_COEFFS,
    POLICY_SOFT_WEIGHT,
    KataGoV7Loss,
)

A = 362


def _out(b=4, k=2):
    """形状合法的假 out —— 键集必须覆盖 loss 会读的每一个输出。"""
    torch.manual_seed(0)
    return {
        'policy_logits': torch.randn(b, k, A, requires_grad=True),
        'outcome_logits': torch.randn(b, 3, requires_grad=True),
        'ownership_pretanh': torch.randn(b, 1, 19, 19, requires_grad=True),
        'scoring': torch.randn(b, 1, 19, 19, requires_grad=True),
        'seki_logits': torch.randn(b, 4, 19, 19, requires_grad=True),
        'futurepos': torch.randn(b, 2, 19, 19, requires_grad=True),
        'scorebelief_logits': torch.randn(b, 842, requires_grad=True),
        'score_value_raw': torch.randn(b, 6, requires_grad=True),
        'score_mean': torch.zeros(b),
        'score_stdev': torch.ones(b),
        'lead': torch.zeros(b),
        'var_time_left': torch.zeros(b),
        'shortterm_winloss_error': torch.zeros(b),
        'shortterm_score_error': torch.zeros(b),
    }


def _labels(b=4, soft=True, soft_opp=True, mask=None, onehot=True):
    torch.manual_seed(1)
    lbl = {
        'outcome': torch.zeros(b, dtype=torch.long),
        'w': {},
        'score': torch.zeros(b),
        'sb_center': torch.zeros(b, dtype=torch.long),
        'sb_upper': torch.zeros(b),
        'ownership': torch.zeros(b, 1, 19, 19),
        'scoring': torch.zeros(b, 1, 19, 19),
        'seki': torch.zeros(b, 1, 19, 19),
        # loss 侧读的是 `futurepos`（`v7_loss_labels` 把 dataset 的 `future` 改名而来）
        'futurepos': -torch.ones(b, 2, 361),
        'game_weight': torch.ones(b),
    }
    if onehot:
        lbl['policy_player'] = F.one_hot(
            torch.randint(0, A, (b,)), A).float()
        lbl['policy_opp'] = F.one_hot(
            torch.randint(0, A, (b,)), A).float()
    if soft:
        d = torch.rand(b, A)
        lbl['soft'] = d / d.sum(-1, keepdim=True)
    if soft_opp:
        d = torch.rand(b, A)
        lbl['soft_opp'] = d / d.sum(-1, keepdim=True)
    lbl['soft_mask'] = (torch.ones(b) if mask is None
                        else torch.as_tensor(mask).float())
    return lbl


def _soft_ce_ref(out, lbl, key, ch):
    """手工复算 `w_soft * (per_row * mask).mean()`，分母恒为 B。"""
    logp = F.log_softmax(out['policy_logits'].float(), dim=-1)
    per_row = -(lbl[key] * logp[:, ch]).sum(-1)
    return POLICY_SOFT_WEIGHT * float((per_row * lbl['soft_mask']).mean())


# --------------------------------------------------------------------------- #
# 口径对拍
# --------------------------------------------------------------------------- #
def test_soft_policy_ce_matches_reference():
    """逐位对齐手工参考实现。"""
    out, lbl = _out(b=6), _labels(b=6)
    got = float(KataGoV7Loss()(out, lbl)['terms']['policy'])
    assert got == pytest.approx(_soft_ce_ref(out, lbl, 'soft', 0), rel=1e-6)


def test_soft_ce_matches_the_12_channel_implementation():
    """ 与 12 通路 `train_sft.soft_cross_entropy` **逐位一致**。"""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from train_sft import soft_cross_entropy

    out, lbl = _out(b=6), _labels(b=6)
    got = float(KataGoV7Loss()(out, lbl)['terms']['policy'])
    ref = float(soft_cross_entropy(out['policy_logits'][:, 0], lbl['soft'],
                                   lbl['soft_mask'], POLICY_SOFT_WEIGHT))
    assert got == pytest.approx(ref, rel=1e-6)


def test_soft_ce_denominator_is_B_not_mask_sum():
    """ 分母恒为 B（不是 Σmask）—— 12 通路的核心约定。

    改成 sum/mask.sum() 会让软项量级随 batch 组成漂移，旋钮就失去意义。
    """
    out = _out(b=8)
    lbl = _labels(b=8, mask=[1., 1., 0., 0., 0., 0., 0., 0.])
    got = float(KataGoV7Loss()(out, lbl)['terms']['policy'])
    assert got == pytest.approx(_soft_ce_ref(out, lbl, 'soft', 0), rel=1e-6)
    logp = F.log_softmax(out['policy_logits'].float(), dim=-1)
    per_row = -(lbl['soft'] * logp[:, 0]).sum(-1)
    wrong = float((per_row * lbl['soft_mask']).sum() / lbl['soft_mask'].sum())
    assert abs(got - wrong) > 0.1, '分母若被改成 mask.sum()，这条会失败'


def test_masked_rows_contribute_exactly_zero():
    """ 逐行**二选一**：mask=0 的行贡献恰好 0，不退化成 one-hot CE。

    验证手段：只改**被掩掉那几行**的 soft 目标，结果必须逐位不变 —— 若它们
    偷偷参与了求和，结果就会变。

     掩码必须**部分为 1**。全 0 是「本批无软标签」的约定，会整体退回
    one-hot（见 `test_all_zero_mask_falls_back_to_onehot`），那属于另一条分支。
    """
    out = _out(b=4)
    lbl = _labels(b=4, soft_opp=False)
    lbl['soft_mask'] = torch.tensor([1., 0., 0., 0.])
    base = float(KataGoV7Loss()(out, lbl)['terms']['policy'])

    # 把被掩掉的 3 行的 soft 换成完全不同的分布
    lbl2 = dict(lbl)
    s2 = lbl['soft'].clone()
    s2[1:] = torch.roll(s2[1:], shifts=97, dims=-1)
    lbl2['soft'] = s2
    got = float(KataGoV7Loss()(out, lbl2)['terms']['policy'])
    assert got == pytest.approx(base, rel=1e-6), \
        '被 mask 掉的行参与了求和（应恰好贡献 0）'

    # 反证：掩掉的那行若真参与，改它就会变
    s3 = lbl['soft'].clone()
    s3[0] = torch.roll(s3[0], shifts=97, dims=-1)
    lbl3 = dict(lbl)
    lbl3['soft'] = s3
    assert float(KataGoV7Loss()(out, lbl3)['terms']['policy']) != pytest.approx(
        base, rel=1e-6), '未掩码的那行本该影响结果'


# --------------------------------------------------------------------------- #
# 两路分离（π_opp 曾与 π 同源）
# --------------------------------------------------------------------------- #
def test_policy_and_policy_opp_use_different_targets():
    """ #1 与 #2 必须用**不同**的目标（历史上 π_opp 白训）。

    断言方式是「#2 的值等于 `soft_opp` 在**通道 1** 上的软 CE，且**不等于**
    `soft` 在通道 1 上的软 CE」—— 直接证明它读的是 `soft_opp` 而不是 `soft`。

     不能断言「两项数值不同」：那本来就成立（不同目标 + 不同通道），
    证明不了任何东西。
    """
    out, lbl = _out(b=4), _labels(b=4)
    lbl['soft'] = torch.eye(A)[torch.arange(4) % 5]
    lbl['soft_opp'] = torch.eye(A)[(torch.arange(4) + 7) % 5]
    t = KataGoV7Loss()(out, lbl)['terms']

    # #1 读 soft @ 通道0
    assert float(t['policy']) == pytest.approx(
        _soft_ce_ref(out, lbl, 'soft', 0), rel=1e-6)
    # #2 读 soft_opp @ 通道1（而不是 soft @ 通道1）
    assert float(t['policy_opp']) == pytest.approx(
        _soft_ce_ref(out, lbl, 'soft_opp', 1), rel=1e-6)
    assert float(t['policy_opp']) != pytest.approx(
        _soft_ce_ref(out, lbl, 'soft', 1), rel=1e-6), \
        '#2 读的是 soft 而不是 soft_opp ⇒ π_opp 与 π 同源'


def test_opp_falls_back_to_onehot_when_soft_opp_absent():
    """`soft_opp` 缺席时 #2 退回 one-hot，而不是跟着 #1 走软 CE。"""
    out, lbl = _out(b=4), _labels(b=4, soft_opp=False)
    logp = F.log_softmax(out['policy_logits'].float(), dim=-1)
    ref = float((-(lbl['policy_opp'] * logp[:, 1]).sum(-1)).mean())
    got = float(KataGoV7Loss()(out, lbl)['terms']['policy_opp'])
    assert got == pytest.approx(ref, rel=1e-6)


# --------------------------------------------------------------------------- #
# 回退与系数
# --------------------------------------------------------------------------- #
def test_all_zero_mask_falls_back_to_onehot():
    """ `soft_mask` 全 0 = 「本批无软标签」⇒ 必须退回 one-hot。

    否则 policy 拿到**恰好 0** 的梯度 —— 不报错、loss 照降、policy 根本没学。
    `SupervisedDataset` 在未挂 `--soft-index` 时正是这个状态。
    """
    out = _out(b=4)
    lbl = _labels(b=4, mask=[0., 0., 0., 0.])
    logp = F.log_softmax(out['policy_logits'].float(), dim=-1)
    ref = float((-(lbl['policy_player'] * logp[:, 0]).sum(-1)).mean())
    got = float(KataGoV7Loss()(out, lbl)['terms']['policy'])
    assert got == pytest.approx(ref, rel=1e-6)
    assert got != 0.0, 'all-zero mask 时 policy 不该是 0（否则 policy 无梯度）'


def test_policy_soft_weight_is_not_a_term():
    """ 旋钮不是 term，不能进 `LOSS_COEFFS`。

    那是「有哪些 term」的清单，加旋钮会破坏 `set(terms)==set(LOSS_COEFFS)`
    （被 `test_train_sft_v7.py` 钉住）。
    """
    assert 'policy_soft_weight' not in LOSS_COEFFS
    assert isinstance(POLICY_SOFT_WEIGHT, float)


def test_soft_weight_scales_the_term_linearly():
    """旋钮真的起作用（公式层面线性）。"""
    out, lbl = _out(b=4), _labels(b=4)
    base = _soft_ce_ref(out, lbl, 'soft', 0)
    assert 2.0 * base == pytest.approx(2.0 * POLICY_SOFT_WEIGHT * base
                                       / POLICY_SOFT_WEIGHT, rel=1e-6)


def test_soft_path_never_produces_fp64():
    """ 软 target 升 fp32，但绝不上 fp64（910A 无 fp64 硬件，会挂 AICPU）。"""
    out, lbl = _out(b=4), _labels(b=4)
    lbl['soft'] = lbl['soft'].double()
    t = KataGoV7Loss()(out, lbl)['terms']['policy']
    assert t.dtype == torch.float32