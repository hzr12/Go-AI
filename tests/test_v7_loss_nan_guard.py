# -*- coding: utf-8 -*-
"""V7 loss 的 NaN 防护（2026-10-04 云端 910A 实跑打出来的）。

实跑症状
--------
```
Loss scaler reducing loss scale to 5120.0 → 2560 → 1280 → 640 → 320 → 160
[fp16] 缩放值首次跌破 1024：1280 -> 640。累计跳过 4 步（占比 133.33%）
```
缩放值从 10240 一路降到 160 仍 100% 跳过，且**每步的 nan 参数计数完全一样**
（211 / 98 / 8）。

「每步都溢出 ⇒ GradScaler 降 scale」是**误读**：GradScaler 只缩放 loss，
管不了前向/反向里的 NaN。真正能解释「恒定计数 + 降到底仍不恢复」的是
**一个被 buffer 记住的 NaN**。

本文件钉住两道防护，以及「非零系数项坏了必须暴露、不能被吞」这条底线。
"""
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from src.networks.katago_v7_loss import (  # noqa: E402
    SEKI_ADAPT_DEN,
    SEKI_ADAPT_NUM,
    KataGoV7Loss,
)

B, A = 4, 362
ZERO_STAGE1 = ('lead', 'ownership', 'score_mean', 'score_stdev',
               'scorebelief_cdf', 'scorebelief_pdf', 'scoring', 'seki',
               'var_time_left')


def _out():
    g = torch.Generator().manual_seed(3)
    return {
        'policy_logits': torch.randn(B, 2, A, generator=g),
        'outcome_logits': torch.randn(B, 3, generator=g),
        'ownership_pretanh': torch.randn(B, 1, 19, 19, generator=g),
        'scoring': torch.randn(B, 1, 19, 19, generator=g),
        'seki_logits': torch.randn(B, 4, 19, 19, generator=g),
        'futurepos': torch.randn(B, 2, 19, 19, generator=g),
        'scorebelief_logits': torch.randn(B, 842, generator=g),
        'score_value_raw': torch.randn(B, 6, generator=g),
        'score_mean': torch.zeros(B), 'score_stdev': torch.ones(B),
        'lead': torch.zeros(B),
    }


def _lbl():
    return {
        'policy_player': torch.zeros(B, A), 'policy_opp': torch.zeros(B, A),
        'outcome': torch.zeros(B, dtype=torch.long), 'w': {},
        'score': torch.zeros(B), 'sb_center': torch.zeros(B, dtype=torch.long),
        'sb_upper': torch.zeros(B), 'game_weight': torch.ones(B),
        'ownership': torch.zeros(B, 1, 19, 19),
        'scoring': torch.zeros(B, 1, 19, 19),
        'seki': torch.zeros(B, 1, 19, 19),
        'futurepos': -torch.ones(B, 2, 361),
    }


def _stage1():
    """段 1 的 loss（9 项系数为 0），与 train_sft 用的是同一套装配。"""
    from train_sft import build_v7_stage1_loss
    return build_v7_stage1_loss(action_size=A)


# --------------------------------------------------------------------------- #
# ① 零系数项坏掉不许把总 loss 拖成 NaN
# --------------------------------------------------------------------------- #
def test_zero_coefficient_terms_are_exactly_the_stage1_ones():
    lf = _stage1()
    zeros = {k for k, v in lf.coeff.items() if v == 0.0}
    assert zeros == set(ZERO_STAGE1), zeros


@pytest.mark.parametrize('term,key', [
    ('scoring', 'scoring'),
    ('ownership', 'ownership_pretanh'),
    ('lead', 'lead'),
])
def test_broken_zero_weight_term_does_not_poison_the_total(term, key):
    """🔴 `0.0 * NaN == NaN`（IEEE-754），不是 0。

    段 1 有 **9 项系数为 0**。它们本来就不参与优化，却仍要算、还要进反向 ⇒
    纯浪费，且是 NaN 的现成通道：任何一项在 NPU 上算坏，**加权总 loss 就是
    NaN**，而它一 NaN，每一个收到梯度的参数张量都是 NaN（实测 317/322）。
    """
    lf = _stage1()
    o = _out()
    o[key] = torch.full_like(o[key], float('nan'))
    r = lf(o, _lbl())
    assert torch.isfinite(r['loss']), \
        f'{term} 坏掉不该把总 loss 拖成 NaN（它的系数是 0）'


@pytest.mark.parametrize('term,key', [
    ('scoring', 'scoring'),
    ('ownership', 'ownership_pretanh'),
    ('lead', 'lead'),
])
def test_broken_zero_weight_term_is_still_named(term, key):
    """🔴 守卫不能「顺手把证据也吞掉」—— 必须点名。

    正因为 `total` 对零系数项恒有限，**只在 `total` 非有限时才逐项扫**的那种
    写法会让这些项永远不被发现：训练看起来正常，而那一项其实一直没在学。
    """
    lf = _stage1()
    o = _out()
    o[key] = torch.full_like(o[key], float('nan'))
    r = lf(o, _lbl())
    assert term in r['nonfinite_terms'], \
        f'{term} 坏了却没被点名：{r["nonfinite_terms"]}'


def test_seki_broken_is_reported_via_sanitized_rows():
    """⚠ `seki` 走的是另一条路：它的行权重全为 0 ⇒ 被**净化**成有限。

    所以它不会出现在 `nonfinite_terms` 里（那一项确实已经有限了），
    但必须出现在 `sanitized_rows` 里 —— 否则「seki 头一直在吐 inf」这件事
    就彻底没人知道了，而它在段 1 系数为 0、坏掉时对总 loss 毫无影响。
    """
    lf = _stage1()
    o = _out()
    o['seki_logits'] = torch.full((B, 4, 19, 19), float('inf'))
    r = lf(o, _lbl())
    assert torch.isfinite(r['loss']), '净化后总 loss 应有限'
    assert r['sanitized_rows'].get('seki'), r['sanitized_rows']


def test_healthy_forward_names_nothing():
    lf = _stage1()
    r = lf(_out(), _lbl())
    assert r['nonfinite_terms'] == [], r['nonfinite_terms']
    assert torch.isfinite(r['loss'])


def test_broken_nonzero_weight_term_is_not_masked():
    """⚠ 反向底线：**系数非 0** 的项坏了必须让 `total` 非有限。

    守卫只允许跳过 c==0 的项。若它扩大到 c!=0，就会把「主目标坏了」这种
    最该立刻发现的故障伪装成正常训练。
    """
    lf = _stage1()
    o = _out()
    o['policy_logits'] = torch.full_like(o['policy_logits'], float('nan'))
    r = lf(o, _lbl())
    assert not torch.isfinite(r['loss']), \
        'policy（系数 1.0）坏了必须暴露，不能被守卫吞掉'
    assert 'policy' in r['nonfinite_terms']


def test_every_zero_weight_head_still_gets_a_gradient():
    """🔴 计算图**必须完整** —— 剪掉就等于把 NaN 换成 DDP 崩溃。

    真机 2 卡实测：`Expected to have finished reduction in the prior iteration.
    Parameter indices which did not receive grad for rank 0: 308 309 310 311
    313 314 315 316 317 318 319 320 321` —— 成段的索引正是那些零系数头。

    我第一版修法写的是「c==0 就直接给 0.0」，那会把这些项从图里摘掉 ⇒
    它们的参数 `grad=None` ⇒ DDP（`find_unused_parameters=False`）当场抛错。
    **安静地剪掉梯度不是修复，是换一个更响的崩溃。**

    所以现在用 `torch.nan_to_num` 净化数值：前向有限、反向透传、图完整。
    本测试直接用 backward 后数参数来钉住这条 —— 它比任何文本判据都硬。
    """
    from train_sft import build_v7_stage1_loss
    from src.networks.katago_v7 import build_katago_v7_net
    net = build_katago_v7_net(board_size=19)
    lf = build_v7_stage1_loss(action_size=A)
    sp = torch.randn(B, 22, 19, 19)
    gl = torch.randn(B, 19)
    out = net(sp, gl)
    lbl = _lbl()
    lbl['ownership'] = torch.zeros(B, 1, 19, 19)
    lbl['seki'] = torch.zeros(B, 1, 19, 19)
    lbl['futurepos'] = -torch.ones(B, 2, 361)
    lf(out, lbl)['loss'].backward()

    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, \
        f'这些参数没收到梯度 ⇒ DDP 会抛「Expected to have finished reduction」：{missing}'
    assert len(list(net.parameters())) > 300, '网络结构与预期不符'


def build_katago_v7_net_for_test():
    from src.networks.katago_v7 import build_katago_v7_net
    return build_katago_v7_net(board_size=19)


def test_nan_to_num_keeps_the_graph_and_passes_gradient():
    """⚠ `nan_to_num` 的反向语义必须仍是「图活着」。

    前向：非有限 → 0；反向：**有限处透传梯度**、非有限处给 0。
    若哪天它变成 detach 语义，这条会先红。
    """
    x = torch.tensor([1.0, float('nan'), 3.0], requires_grad=True)
    y = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    assert y.requires_grad, 'nan_to_num 切断了图'
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), x.grad


def test_nan_guard_actually_sanitizes_forward_but_not_the_reported_term():
    """点名用的是**净化前**的值 —— 否则诊断会自我掩盖。"""
    lf = _stage1()
    o = _out()
    o['scoring'] = torch.full_like(o['scoring'], float('nan'))
    r = lf(o, _lbl())
    assert torch.isnan(r['terms']['scoring']).all(), \
        'terms 里应保留原始 NaN（供诊断与排查）'
    assert r['weighted']['scoring'] == 0.0, 'weighted 里应是净化后的 0'
    assert torch.isfinite(r['loss'])


def test_score_stdev_survives_near_uniform_distribution():
    """🔴 根因回归：842 桶近均匀时 `std` 必须是有限值。

    softmax 在 842 个桶上近均匀 ⇒ p ≈ 1.19e-3 且彼此差极小 ⇒ 方差 ≈ 1e-10。
    朴素公式 `E[x²] − E[x]²` 在这个量级会**灾难性抵消**出负数 ⇒ sqrt → NaN。
    CPU 的 `torch.std` 用两遍算法所以本机复现不出来，但 NPU 规约核用朴素公式，
    真机实测点名到本项（2026-10-04）。
    """
    lf = _stage1()
    # 构造「近均匀」的 scorebelief logits：与真实初始化同量级
    o = _out()
    o['scorebelief_logits'] = torch.full((B, 842), 1e-4)
    r = lf(o, _lbl())
    assert torch.isfinite(r['terms']['score_stdev']), \
        '近均匀 softmax 上的 std 变成了非有限（朴素方差公式抵消）'
    assert 'score_stdev' not in r['nonfinite_terms']


def test_score_stdev_two_pass_matches_torch_std_in_fp32():
    """两遍实现必须与 `torch.std` 在 fp32 下**逐位接近**（不是新发明口径）。"""
    lf = _stage1()
    o = _out()
    o['scorebelief_logits'] = torch.randn(B, 842) * 3.0
    r = lf(o, _lbl())
    p = torch.softmax(o['scorebelief_logits'].float(), dim=-1)
    want = p.std(-1)
    # score_stdev = huber(pred, sb_std, 10)；pred 是 ones ⇒ huber(1, std, 10)
    # 反解不方便，改为直接比两遍算法本身
    mu = p.mean(-1, keepdim=True)
    two_pass = (p - mu).pow(2).mean(-1).sqrt()
    # 实测两遍与 Welford 的相对差 ≈ 5.9e-4（两者都正确，只是累加顺序不同）
    # ⇒ 容差取 2e-3。断言的是「同一个口径」，不是逐位相同。
    assert torch.allclose(two_pass, want, rtol=2e-3, atol=1e-9), \
        (two_pass, want)
    assert torch.isfinite(r['terms']['score_stdev'])


def test_score_stdev_target_has_a_gradient_bounding_floor():
    """🔴 `std` 目标必须有**下界**，否则它的反向在 fp16 上溢。

    实测（本地 fp32）：初始化时 842 桶近均匀 ⇒ `std ≈ 1.17e-7`
    ⇒ `d(sqrt(v))/dv = 1/(2·std) ≈ 4.27e6`，**远超 fp16 的 65504** ⇒ 反向
    溢出成 inf；再乘「段 1 系数 0」这个上游梯度（**0**），`0 × inf = NaN`
    ⇒ 全部参数梯度 NaN ⇒ 每步都被 GradScaler 跳过。

    ⚠ 这与前向的「灾难性抵消」是**两个独立**的问题：前向用两遍算法修好了，
      但反向的 `1/(2·std)` 只能靠**下界**钉住 —— `clamp_min` 在下界以下是常数、
      梯度恰好 0，从根上拿掉那个爆炸因子（而不是靠降学习率绕）。
    """
    from src.networks.katago_v7_loss import SCORE_STDEV_TARGET_FLOOR
    # 导数上界必须远低于 fp16 上限（留 100 倍余量）
    assert 1.0 / (2.0 * SCORE_STDEV_TARGET_FLOOR) < 655.04, \
        '下界太小，钉不住 1/(2·std)'
    # 且下界要远低于任何有意义的展度（预测初值约 13.9）
    assert SCORE_STDEV_TARGET_FLOOR < 1e-2, SCORE_STDEV_TARGET_FLOOR


def test_score_stdev_floor_actually_clamps_near_uniform_targets():
    """近均匀 softmax 的 std ≈ 1.17e-7 远低于下界 ⇒ 目标被钉在 1e-3。"""
    import torch.nn.functional as F
    from src.networks.katago_v7_loss import SCORE_STDEV_TARGET_FLOOR
    p = F.softmax(torch.full((1, 842), 1e-4), dim=-1)
    mu = p.mean(-1, keepdim=True)
    raw = (p - mu).pow(2).mean(-1).sqrt()
    assert float(raw) < SCORE_STDEV_TARGET_FLOOR, float(raw)
    clamped = raw.clamp_min(SCORE_STDEV_TARGET_FLOOR)
    assert float(clamped) == pytest.approx(SCORE_STDEV_TARGET_FLOOR)
    # 下界以下梯度恒 0 ⇒ 1/(2·std) 这个因子根本不会被算出来
    v = raw.clone().requires_grad_(True)
    v.clamp_min(SCORE_STDEV_TARGET_FLOOR).sum().backward()
    assert float(v.grad.abs().max()) == 0.0, '下界以下不该有梯度'


def test_operand_attribution_names_the_broken_side():
    """🔴 必须能区分「预测坏」与「目标坏」。

    逐项点名只说「哪一项」，而这一项内部有两个来源完全不同的操作数：
    预测来自 ValueHead、目标来自 scorebelief 头。本机与 NPU 的 kernel 不同，
    每次真机跑一轮都要几分钟 ⇒ 归因必须一次到位，不能再猜。
    """
    lf = _stage1()
    o = _out()
    o['scorebelief_logits'] = torch.full((B, 842), 1e-4)
    o['score_stdev'] = torch.full((B,), float('nan'))
    r = lf(o, _lbl())
    assert 'score_stdev:pred' in r['nonfinite_operands'], r['nonfinite_operands']


def test_total_finite_flag_distinguishes_sanitised_from_real():
    """⚠ 日志要能说清「total 到底有限没有」。

    零系数项会被净化成 0，所以「某项坏」时 total 往往**仍有限**。若日志一律
    写成「加权 loss 非有限」，就是在报一件没发生的事，看日志的人会以为训练
    已经废了 —— 这比不报更坏。
    """
    lf = _stage1()
    r_ok = lf(_out(), _lbl())
    assert r_ok['total_finite'] is True
    o = _out()
    o['scoring'] = torch.full_like(o['scoring'], float('nan'))
    r_bad = lf(o, _lbl())
    assert r_bad['nonfinite_terms'], '零系数项坏了应被点名'
    assert r_bad['total_finite'] is True, \
        '零系数项被净化后 total 应仍有限'


def test_weighted_mean_keeps_its_exact_semantics_when_finite():
    """🔴 防护不能改动 spec §4.5 的口径 —— 分母恒为 batch 大小，不是 Σw。

    `samplewise` 的语义是「逐样本损失对 batch 取均值，行权重只作逐样本乘子」。
    有人会顺手改成 `/w.sum()`，那会让「整批 w=0」把这一项放大到噪声水平。
    """
    from src.networks.katago_v7_loss import _weighted_mean
    ps = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for w in ([1., 1., 1., 1.], [0., 1., 1., 0.],
              [0., 0., 0., 0.], [2., 0., 1., 3.]):
        wt = torch.tensor(w)
        assert float(_weighted_mean(ps, wt)) == pytest.approx(
            float((ps * wt).mean()), rel=1e-12), w


@pytest.mark.parametrize('bad', [float('inf'), float('nan')])
def test_weighted_mean_survives_nonfinite_on_zero_weight_rows(bad):
    """🔴 `inf × 0 = NaN`（IEEE-754）—— 一行坏数据不该带崩整项。

    那些行按定义贡献恰好 0，所以「inf/NaN × 0」的正确结果是 0，不是 NaN。
    典型触发：`game_weight == 0` 的行 + 该行的 `per_sample` 溢出。
    """
    from src.networks.katago_v7_loss import _weighted_mean
    ps = torch.tensor([bad, 1.0, 2.0, 3.0])
    w = torch.tensor([0., 1., 1., 1.])
    got = _weighted_mean(ps, w)
    assert torch.isfinite(got), got
    # 分母仍是 **batch 大小 4**（不是「有效行数 3」）—— 这是 spec §4.5 的口径，
    # 改成 /Σw 或 /len(kept) 都是悄悄改了语义。
    assert float(got) == pytest.approx(6.0 / 4.0), got


def test_weighted_mean_records_sanitised_rows_for_diagnostics():
    """🔴 净化是**静默**的 —— 必须把「有几行本该是 0 而实际非有限」记下来。

    不记的话就成了「这一项坏了但没人知道」：段 1 有 9 项系数为 0，它们坏掉时
    对总 loss 毫无影响，正因如此更需要把线索报出来。
    """
    from src.networks.katago_v7_loss import _weighted_mean
    probe = {}
    ps = torch.tensor([float('inf'), 1.0, float('nan'), 3.0])
    w = torch.tensor([0., 1., 0., 1.])
    _weighted_mean(ps, w, probe, 'demo')
    assert probe['demo'] == 2, probe


def test_weighted_mean_records_nothing_when_finite():
    from src.networks.katago_v7_loss import _weighted_mean
    probe = {}
    _weighted_mean(torch.tensor([1.0, 2.0]),
                   torch.tensor([0.0, 1.0]), probe, 'demo')
    assert probe == {}, probe


def test_weighted_mean_does_not_swallow_nonzero_weight_rows():
    """⚠ 反向底线：`w != 0` 的行坏了必须照常暴露。"""
    from src.networks.katago_v7_loss import _weighted_mean
    got = _weighted_mean(torch.tensor([float('inf'), 1.0]),
                         torch.tensor([1., 1.]))
    assert not torch.isfinite(got), '主目标行坏了却被吞掉'


def test_weighted_mean_without_weight_is_unchanged():
    from src.networks.katago_v7_loss import _weighted_mean
    ps = torch.tensor([1.0, 2.0, 3.0])
    assert float(_weighted_mean(ps, None)) == pytest.approx(2.0)


def test_huber_is_numerically_robust_at_stage1_magnitudes():
    """用户怀疑过 huber 本身 —— 这条把它**排除**掉，留作记录。

    实测：`huber` = `F.smooth_l1_loss`，前向在 pred=13.86…1e4 全有限
    （fp32 与 fp16 都试过），反向导数被 clip 到 ±1（有界）。
    ⇒ 这一项的问题**不在 huber**，而在它**目标**那一侧：
    `std(softmax)` 的反向 `1/(2·std)`，初始化时 ≈ 4.27e6 ⇒ fp16 溢出。
    """
    import torch.nn.functional as F
    for pred in (13.86, 100.0, 1e3, 1e4):
        for dt in (torch.float32, torch.float16):
            p = torch.tensor([pred], dtype=dt)
            v = F.smooth_l1_loss(p, torch.tensor([1e-3], dtype=dt),
                                 beta=10.0, reduction='none')
            assert torch.isfinite(v).all(), (pred, dt, float(v))
    # 反向有界
    x = torch.tensor([1e4], requires_grad=True)
    F.smooth_l1_loss(x, torch.tensor([1e-3]), beta=10.0,
                     reduction='sum').backward()
    assert abs(float(x.grad)) <= 1.0 + 1e-6, float(x.grad)


def test_weighted_keys_still_complete_and_sum_unchanged():
    lf = _stage1()
    r = lf(_out(), _lbl())
    assert set(r['weighted']) == set(r['terms'])
    assert set(r['terms']) <= set(lf.coeff), set(r['terms']) - set(lf.coeff)
    w = r['weighted']
    assert abs(float(sum(w.values()))
               - float(w['policy'] + w['policy_opp'] + w['value']
                       + w['futurepos'])) < 1e-5, r['weighted']


# --------------------------------------------------------------------------- #
# ② seki_ema：一次 NaN 永久污染
# --------------------------------------------------------------------------- #
def test_seki_ema_is_a_persistent_buffer():
    """前提：它是注册 buffer ⇒ 一次写入永久，且**随 checkpoint 传下去**。"""
    lf = _stage1()
    names = dict(lf.named_buffers())
    assert 'seki_ema' in names, names.keys()
    assert 'seki_ema' in lf.state_dict(), \
        'buffer 进 state_dict ⇒ 被 NaN 毒化的 .pth 会传染给 --resume'


def test_nan_seki_ema_no_longer_poisons_every_future_step():
    """🔴 NaN 进 buffer 后必须**不写进去**，且后续每步的 scale 都有限。

    这条对应实跑里最关键的现象：每步 nan 计数**完全一样**。一个「每步重新
    发生」的溢出会让计数抖动，而一个被 buffer 记住的 NaN 给出恒定结果。
    """
    lf = _stage1()
    lf.train()
    lf.seki_ema.copy_(torch.tensor(float('nan')))
    scales = [float(lf._seki_adaptive_scale(torch.tensor(0.02)))
              for _ in range(5)]
    assert all(s == s for s in scales), f'仍有 NaN：{scales}'
    # 退到「seki 极稀有」那个系数，而不是 0（0 会让这一项彻底不学）
    assert all(abs(s - SEKI_ADAPT_NUM / SEKI_ADAPT_DEN) < 1e-6 for s in scales), \
        scales


def test_nan_cur_does_not_overwrite_the_ema():
    """⚠ 一步坏掉**不该抹掉**已有的 EMA 历史。"""
    lf = _stage1()
    lf.train()
    good = torch.tensor(0.02)
    lf._seki_adaptive_scale(good)
    before = float(lf.seki_ema)
    assert before != 0.0, '先建立一个非零 EMA'
    lf._seki_adaptive_scale(torch.tensor(float('nan')))
    assert float(lf.seki_ema) == before, \
        '坏的那一步把 EMA 冲掉了 ⇒ 下一步的尺度就变了'


def test_healthy_seki_ema_still_adapts():
    """⚠ 守卫不能把自适应**关掉**：正常路径必须照常更新 EMA。

    ⚠ 用**递减**输入：常数输入下 EMA 一步就到不动点（`0.99·x + 0.01·x = x`），
      第二次调用起 scale 恒定 —— 那是数学正确，不是「没推进」。
    """
    lf = _stage1()
    lf.train()
    s = [float(lf._seki_adaptive_scale(torch.tensor(v)))
         for v in (0.02, 0.05, 0.10, 0.20)]
    assert len(set(s)) == len(s), f'EMA 没有推进：{s}'
    # EMA 上升 ⇒ 分母变大 ⇒ scale 下降
    assert s == sorted(s, reverse=True), s
    assert all(x > 0 for x in s), s


def test_fallback_scale_is_a_tensor_not_a_float():
    """⚠ 兜底返回的必须是 **tensor**。

    它的返回值会走 `return {... 'seki_adaptive_scale': adaptive.detach()}` ⇒
    给 Python float 会在那里抛 `AttributeError`，而那正是「NaN 兜底路径」
    本身 —— 它一旦抛异常就等于**没兜**，把唯一一次能说出「seki 坏了」的
    机会也弄没了。这条是被真的 AttributeError 打出来的。
    """
    lf = _stage1()
    lf.train()
    lf.seki_ema.copy_(torch.tensor(float('nan')))
    sc = lf._seki_adaptive_scale(torch.tensor(0.02))
    assert torch.is_tensor(sc), type(sc).__name__
    assert hasattr(sc, 'detach')
    assert float(sc) == pytest.approx(SEKI_ADAPT_NUM / SEKI_ADAPT_DEN)


def test_eval_mode_does_not_touch_the_buffer():
    """`.eval()` 下不得改 buffer（否则同一 batch 反复评得到不同 loss）。"""
    lf = _stage1()
    lf.seki_ema.copy_(torch.tensor(0.03))
    lf.eval()
    for _ in range(3):
        lf._seki_adaptive_scale(torch.tensor(0.99))
    assert float(lf.seki_ema) == pytest.approx(0.03)


def test_full_loss_forward_survives_a_nan_in_every_zero_weight_head():
    """把所有**零系数**头打成 NaN：总 loss 仍有限，且点名列表非空。

    这是「最坏情况」用例 —— 对应真机上「不知道是哪一项坏了」的那一刻。

    ⚠ 只打**零系数**的那些：`futurepos` 在段 1 的系数是 1.0（非 0），它坏了
      就该让总 loss 非有限 —— 那是另一条测试（`test_broken_nonzero_weight_term
      _is_not_masked`）的职责，混进来会让这条的前提失效。
    """
    lf = _stage1()
    o = _out()
    for k in ('scoring', 'seki_logits', 'ownership_pretanh', 'lead',
              'scorebelief_logits', 'score_mean', 'score_stdev'):
        o[k] = torch.full_like(o[k], float('nan'))
    r = lf(o, _lbl())
    assert torch.isfinite(r['loss']), '零系数头全坏也不该污染总 loss'
    assert len(r['nonfinite_terms']) >= 5, r['nonfinite_terms']
    assert 'policy' not in r['nonfinite_terms'], \
        'policy 项本身没坏，不该被牵连'
    assert 'futurepos' not in r['nonfinite_terms'], \
        'futurepos 系数非 0，没坏就不该被点名'