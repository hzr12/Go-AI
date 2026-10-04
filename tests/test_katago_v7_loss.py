"""V7 的 12 项 loss 装配测试（spec §4.5）。

三类测试，重要性递减：
1. **与官方数据对拍** —— scorebelief 的桶偏移不是猜的，是从 stdata 反解出来的；
   这组断言把它钉死，偏移错一位会让整个分差分布系统性平移，而 loss 曲线
   完全看不出异常。
2. **系数只乘一次** —— 「内嵌 vs 装配处」是 spec §4.5 最容易读错的一处，
   乘两次不会报错，只会让某一项梯度悄悄小 25 倍。
3. **结构契约** —— 12 项齐全、权重能真的把项压成 0、eval 不动 EMA。
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.networks.katago_v7_loss import (  # noqa: E402
    LOSS_COEFFS, SCORE_DISTR_BIN_OFFSET, SCORE_DISTR_BINS, SCORE_DISTR_MID,
    KataGoV7Loss, build_score_distr_target, huber, policy_dense_from_sparse,
    seki_targets_from_plane,
)

B, N_SQ, ACTION_SIZE = 4, 19 * 19, 362


def _labels(seed=0, **over):
    g = torch.Generator().manual_seed(seed)
    lb = {
        'policy_player': F.softmax(torch.randn(B, ACTION_SIZE, generator=g), -1),
        'policy_opp': F.softmax(torch.randn(B, ACTION_SIZE, generator=g), -1),
        'outcome': torch.randint(0, 3, (B,), generator=g),
        'score': torch.tensor([3.5, -1.25, 0.0, 8.75]),
        'sb_center': torch.tensor([4, -1, 0, 9]),
        'sb_upper': torch.tensor([30, 75, 50, 88]),
        'ownership': (torch.randint(-1, 2, (B, 1, N_SQ), generator=g).float()),
        'scoring': (torch.randint(-120, 121, (B, 1, N_SQ), generator=g).float()),
        'futurepos': (torch.randint(-1, 2, (B, 2, N_SQ), generator=g).float()),
        'seki': torch.randint(-1, 2, (B, 1, N_SQ), generator=g).float(),
        'game_weight': torch.tensor([1.0, 0.8, 0.0, 1.2]),
        'w': {k: torch.ones(B) for k in
              ('policy_opp', 'ownership', 'score', 'lead', 'futurepos', 'scoring')},
    }
    lb.update(over)
    return lb


def _out(seed=1):
    g = torch.Generator().manual_seed(seed)
    return {
        'policy_logits': torch.randn(B, 2, ACTION_SIZE, generator=g),
        'outcome_logits': torch.randn(B, 3, generator=g),
        'score_mean': torch.randn(B, generator=g),
        'score_stdev': torch.rand(B, generator=g) * 10 + 1,
        'lead': torch.randn(B, generator=g),
        'ownership_pretanh': torch.randn(B, 1, 19, 19, generator=g),
        'scoring': torch.randn(B, 1, 19, 19, generator=g),
        'futurepos': torch.randn(B, 2, 19, 19, generator=g),
        'seki_logits': torch.randn(B, 4, 19, 19, generator=g),
        'scorebelief_logits': torch.randn(B, SCORE_DISTR_BINS, generator=g),
    }


@pytest.fixture
def lossf():
    return KataGoV7Loss(action_size=ACTION_SIZE)


# --------------------------------------------------------------------------- #
# 1. 与官方数据对拍
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('score,expect_lo,expect_frac_lo', [
    (-31.6798, 388, 0.18),   # stdata/2026-08-25npzs 实测行
    (-17.2670, 403, 0.77),
    (-6.3197, 414, 0.82),
    (-3.3996, 417, 0.90),
    (1.4250, 421, 0.08),
])
def test_score_distr_bin_offset_matches_official(score, expect_lo, expect_frac_lo):
    """桶偏移由官方 reanalysis 数据反解，逐位对拍这 5 行。"""
    center = int(round(score))
    lam = score - (center - 0.5)
    upper = int(round(lam * 100))
    tgt = build_score_distr_target(torch.tensor([center]), torch.tensor([upper]))
    nz = torch.nonzero(tgt[0]).reshape(-1).tolist()
    assert nz == [expect_lo, expect_lo + 1], f'score={score} 桶位置 {nz}'
    assert float(tgt[0, expect_lo]) == pytest.approx(expect_frac_lo, abs=0.01)


def test_draw_degenerates_to_two_center_bins():
    """和棋（score=0）自动退化成中心两桶 50/50，**不需要特判**。"""
    tgt = build_score_distr_target(torch.tensor([0]), torch.tensor([50]))
    nz = torch.nonzero(tgt[0]).reshape(-1).tolist()
    assert nz == [SCORE_DISTR_MID - 1, SCORE_DISTR_MID]
    assert float(tgt[0].sum()) == pytest.approx(1.0)


def test_score_distr_target_invariants():
    lb = _labels()
    tgt = build_score_distr_target(lb['sb_center'], lb['sb_upper'])
    assert tgt.shape == (B, SCORE_DISTR_BINS)
    assert torch.allclose(tgt.sum(-1), torch.ones(B), atol=1e-6), '每行必须和为 1'
    assert int((tgt != 0).sum()) == 2 * B, '每行恰有 2 个非零'
    assert bool((tgt >= 0).all()), '桶概率不许为负'


def test_score_distr_out_of_range_raises():
    """分差超出桶范围必须**报错**，不能静默裁剪 —— 静默裁剪会让整局标签错位。"""
    with pytest.raises(ValueError, match='score 越界'):
        build_score_distr_target(torch.tensor([100000]), torch.tensor([50]))


# --------------------------------------------------------------------------- #
# 2. 系数只乘一次
# --------------------------------------------------------------------------- #
def test_coefficients_are_applied_exactly_once(lossf):
    """逐项验算「有效系数 × 公式值 × 行权重」。

    `KataGoV7Loss` 的返回分两层：`terms` 是**未乘系数**的公式值，`weighted`
    才乘上 `LOSS_COEFFS` 里的有效系数。分开是因为训练日志要同时看「这一项的
    原始量级」和「它对总 loss 的贡献」—— 只看总 loss 分不清是权重写错还是
    预测变差。
    """
    out, lb = _out(), _labels()
    res = lossf(out, lb)
    w = lb['w']
    lp = F.log_softmax(out['policy_logits'].float(), -1)      # (B,2,362)
    sb_logits = out['scorebelief_logits'].float()
    sb_tgt = build_score_distr_target(lb['sb_center'], lb['sb_upper'])

    # 1 policy：系数 1.0，无行权重
    expect = (-(lb['policy_player'] * lp[:, 0]).sum(-1)).mean()
    assert res['terms']['policy'] == pytest.approx(float(expect), rel=1e-5)

    # 2 π_opp：公式不含 0.15，系数在 weighted 层
    raw = (-(lb['policy_opp'] * lp[:, 1]).sum(-1) * w['policy_opp']).mean()
    assert res['terms']['policy_opp'] == pytest.approx(float(raw), rel=1e-5)
    assert res['weighted']['policy_opp'] == pytest.approx(
        0.15 * float(raw), rel=1e-5)

    # 3 value：1.20
    raw = F.cross_entropy(out['outcome_logits'].float(), lb['outcome'].long(),
                          reduction='none').mean()
    assert res['weighted']['value'] == pytest.approx(1.20 * float(raw), rel=1e-5)

    # 5 scorebelief pdf：0.020
    raw = (-(sb_tgt * F.log_softmax(sb_logits, -1)).sum(-1) * w['score']).mean()
    assert res['weighted']['scorebelief_pdf'] == pytest.approx(
        0.020 * float(raw), rel=1e-5)

    # 10 scoring：**0.25 在系数表里**（公式里没有）
    mse = (out['scoring'].reshape(B, -1) - lb['scoring'].reshape(B, -1)).pow(2).mean(-1)
    raw = (4 * (torch.sqrt(0.5 * mse + 1) - 1) * w['scoring']).mean()
    assert res['weighted']['scoring'] == pytest.approx(0.25 * float(raw), rel=1e-5)

    # 11 futurepos：公式自带 0.25，**系数表里必须是 1.0**（否则变 0.0625）
    assert LOSS_COEFFS['futurepos'] == 1.0
    assert LOSS_COEFFS['seki'] == 1.0
    sq = (torch.tanh(out['futurepos'].reshape(B, 2, N_SQ))
          - lb['futurepos'].reshape(B, 2, N_SQ)).pow(2)
    cw = torch.tensor([1.0, 0.25]).reshape(1, 2, 1)
    raw = (0.25 * (sq * cw).reshape(B, -1).sum(-1)
           / math.sqrt(N_SQ) * w['futurepos']).mean()
    assert res['terms']['futurepos'] == pytest.approx(float(raw), rel=1e-5)


def test_weighted_equals_coeff_times_terms(lossf):
    res = lossf(_out(), _labels())
    for k, v in res['terms'].items():
        assert res['weighted'][k] == pytest.approx(
            float(LOSS_COEFFS[k] * v), rel=1e-6), f'{k} 的系数不是 {LOSS_COEFFS[k]}'
    assert res['loss'] == pytest.approx(
        sum(float(v) for v in res['weighted'].values()), rel=1e-6)


def test_all_twelve_terms_present(lossf):
    res = lossf(_out(), _labels())
    assert set(res['terms']) == {
        'policy', 'policy_opp', 'value', 'ownership', 'scorebelief_pdf',
        'scorebelief_cdf', 'score_stdev', 'score_mean', 'lead', 'scoring',
        'futurepos', 'seki'}
    assert len(res['terms']) == 12


# --------------------------------------------------------------------------- #
# 3. 结构契约
# --------------------------------------------------------------------------- #
def test_huber_matches_train_sft():
    """本模块的 huber 必须与 `train_sft.huber_loss` **逐位**一致（δ=10/12/8）。"""
    from scripts.train_sft import huber_loss
    torch.manual_seed(0)
    p = torch.randn(64) * 8
    t = torch.randn(64) * 8
    for beta in (8.0, 10.0, 12.0):
        assert torch.equal(huber(p, t, beta),
                           huber_loss(p, t, beta=beta, reduction='none'))


def test_zero_row_weight_zeroes_term(lossf):
    """行权重 0 必须把该项压成 0（这 4 个权重是标签质量掩码，不是正则）。"""
    out, lb = _out(), _labels()
    base = lossf(out, lb)
    for key, term in (('score', 'scorebelief_pdf'), ('lead', 'lead'),
                      ('ownership', 'ownership'), ('futurepos', 'futurepos'),
                      ('scoring', 'scoring'), ('policy_opp', 'policy_opp')):
        lb2 = _labels()
        lb2['w'] = dict(lb2['w'])
        lb2['w'][key] = torch.zeros(B)
        got = lossf(out, lb2)
        assert float(got['terms'][term]) == 0.0, f'w[{key}]=0 未清零 {term}'
    assert base is not None


def test_score_stdev_uses_game_weight_only_and_no_row_weight(lossf):
    """#7 特殊：只用 game_weight，**没有**行权重 —— 与官方一致。"""
    out, lb = _out(), _labels()
    a = lossf(out, lb)['terms']['score_stdev']
    lb2 = _labels()
    lb2['w'] = dict(lb2['w'])
    lb2['w']['score'] = torch.zeros(B)      # 行权重对它**不该**有影响
    b = lossf(out, lb2)['terms']['score_stdev']
    assert float(a) == pytest.approx(float(b), rel=1e-6)


def test_game_weight_zero_kills_score_stdev(lossf):
    lb = _labels()
    lb['game_weight'] = torch.zeros(B)
    assert float(lossf(_out(), lb)['terms']['score_stdev']) == 0.0


def test_seki_ema_not_updated_in_eval(lossf):
    """eval 下反复跑同一 batch 必须得到同一个 loss（`.eval()` 不该改 buffer）。"""
    out, lb = _out(), _labels()
    lossf.train()
    lossf(out, lb)
    ema_after_train = float(lossf.seki_ema)
    lossf.eval()
    a = float(lossf(out, lb)['loss'])
    b = float(lossf(out, lb)['loss'])
    assert a == pytest.approx(b, rel=1e-9), 'eval 两次结果不同 ⇒ EMA 被改了'
    assert float(lossf.seki_ema) == pytest.approx(ema_after_train, rel=1e-9)


def test_seki_adaptive_scale_approaches_eight_for_tiny_ema():
    """seki 极稀有时自适应系数应趋于 8（spec §4.5 #12 的设计意图）。"""
    lossf = KataGoV7Loss(action_size=ACTION_SIZE)
    lossf.train()
    out, lb = _out(), _labels()
    res = lossf(out, lb)                  # 第一次把 EMA 初始化成当前值
    first = float(res['seki_adaptive_scale'])
    assert first == pytest.approx(
        8.0 * 0.005 / (0.005 + float(lossf.seki_ema)))
    # EMA 远小于 0.005 锚点时 → 趋近 8
    lossf.seki_ema.fill_(1e-9)
    got = float(lossf._seki_adaptive_scale(torch.tensor(0.0)))
    assert got == pytest.approx(8.0, rel=1e-4)


def test_seki_ema_is_buffer_not_parameter():
    lossf = KataGoV7Loss(action_size=ACTION_SIZE)
    assert 'seki_ema' in dict(lossf.named_buffers())
    assert 'seki_ema' not in dict(lossf.named_parameters())


def test_seki_targets_from_plane_three_way():
    plane = torch.tensor([[[-1.0, 0.0, 1.0]]])
    sign, neutral = seki_targets_from_plane(plane)
    assert sign.reshape(-1).tolist() == [0, 1, 2]        # −1/0/+1 → 0/1/2
    assert neutral.reshape(-1).tolist() == [0.0, 1.0, 0.0]


def test_explicit_seki_sign_neutral_overrides_derivation(lossf):
    """标签显式给 seki_sign/seki_neutral 时，不走三值平面的推导。"""
    out, lb = _out(), _labels()
    lb['seki_sign'] = torch.full((B, N_SQ), 2, dtype=torch.long)
    lb['seki_neutral'] = torch.zeros(B, N_SQ)
    lb['seki'] = torch.zeros(B, 1, N_SQ)      # 故意与上面矛盾
    res = lossf(out, lb)
    assert torch.isfinite(res['loss'])


def test_policy_dense_from_sparse_renormalizes():
    idx = torch.tensor([[0, 5, 361], [1, 2, 3]])
    val = torch.tensor([[10.0, 20.0, 5.0], [1.0, 1.0, 1.0]])
    d = policy_dense_from_sparse(idx, val, ACTION_SIZE)
    assert d.shape == (2, ACTION_SIZE)
    assert torch.allclose(d.sum(-1), torch.ones(2), atol=1e-6)
    assert float(d[0, 5]) == pytest.approx(20 / 35)
    assert float(d[0, 1]) == 0.0
    raw = policy_dense_from_sparse(idx, val, ACTION_SIZE, renormalize=False)
    assert float(raw.sum()) == pytest.approx(35.0 + 3.0)


def test_sparse_policy_accepted_as_labels(lossf):
    """标签给稀疏 (idx, val) 时走 densify。

     与稠密给法**不会**逐位相等：top-16 截断丢了尾部质量，重归一化后仍是
    另一个合法分布。这里用**尖峰**分布做 fixture（真实 visit 分布是尖的；
    362 维随机 softmax 是近似均匀的，top-16 只能覆盖 4% —— 拿它当 fixture
    会得到一个「K=16 太小」的错误结论），并断言残留 < 1%，兼作 K 的哨兵。
    """
    out = _out()
    lb = _labels()
    # 几何衰减的 visit 分布（0.7^i 归一化）：真实 MCTS 分布的尾部是衰减的，
    # top-16 能覆盖 99.6%。 不能用「尖峰 + 平尾」做 fixture：0.6 的质量摊在
    # 359 个着法上时，每个只有 0.0017，**任何** K=16 都捞不回来，会得到一个
    # 「K 太小」的错误结论。
    decay = torch.tensor([0.7 ** i for i in range(ACTION_SIZE)])
    peaked = (decay / decay.sum()).unsqueeze(0).expand(B, -1).contiguous()
    lb['policy_player'] = peaked
    lb['policy_opp'] = peaked
    dense_ref = lossf(out, lb)
    idx = peaked.topk(16, dim=-1).indices
    val = peaked.gather(1, idx)
    resid = float(1.0 - peaked.gather(1, idx).sum(-1).mean())
    assert resid < 0.01, f'top-16 残留 {resid:.4f} 过大，K 偏小'
    lb2 = _labels()
    lb2.pop('policy_player')
    lb2['policy_player_sparse'] = (idx, val)
    sparse_res = lossf(out, lb2)
    # 0.35% 尾部质量被截断并摊回 ⇒ loss 有 ~3e-4 的相对偏差。**这是截断的
    # 真实代价**，不是实现误差：容差定在 1e-3 量级即「截断小到可以忽略」，
    # 若哪天 K 调小、这个断言开始失败，那正是它该响的时候。
    assert float(sparse_res['loss']) == pytest.approx(
        float(dense_ref['loss']), rel=1e-3)


def test_missing_policy_target_raises(lossf):
    lb = _labels()
    lb.pop('policy_player')
    lb.pop('moves', None)
    with pytest.raises(KeyError, match='policy_player'):
        lossf(_out(), lb)


def test_gradient_flows_to_every_term(lossf):
    out = _out()
    for v in out.values():
        v.requires_grad_(True)
    res = lossf(out, _labels())
    res['loss'].backward()
    dead = [k for k, v in out.items()
            if v.grad is None or not torch.isfinite(v.grad).all()]
    assert not dead, f'这些输出没有有限梯度：{dead}'
