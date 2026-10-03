"""V7 的 12 项 loss 装配（spec §4.5）。

系数为什么大半「内嵌」
----------------------
spec §4.5 的表里每项都有一列「系数」，但同节末尾的说明是关键：

    多数系数**内嵌在各自的 `loss_*_samplewise` 里**，不在装配处再乘；
    唯 `scoring` 的 `0.25` 在装配处。

照字面把表里的系数全在装配处再乘一遍，futurepos 会变成 0.0625、seki 会
被乘两次。本文件把每项的**有效**系数集中在 `LOSS_COEFFS`，并在
`test_katago_v7_loss.py::test_coefficients_are_applied_exactly_once` 里逐项
钉死 —— 系数被乘两次这种错误不会报错，只会让某一项的梯度悄悄小 25 倍。

逐项有效系数
------------
=====  ==================  ==================================  ==============
#      项                   有效系数                             行权重
=====  ==================  ==================================  ==============
1      policy               1.0                                 恒 1
2      π_opp                0.15                                局末手 = 0
3      value（3 分类 CE）    1.20                                恒 1
4      ownership            1.5                                 w_ownership
5      scorebelief pdf      0.020                               w_score
6      scorebelief cdf      0.020                               w_score
7      scorestdev 自预测    0.001                               **仅 game_weight**
8      scoremean            0.0015                              w_score
9      lead                 0.0060                              w_lead
10     scoring              0.25（**装配处**）                    w_scoring
11     futurepos            公式自带 0.25（**不再乘**）            w_futurepos
12     seki                 自适应 8·0.005/(0.005+EMA)            w_ownership
=====  ==================  ==================================  ==============

#7 特殊：**没有行权重**，只用 `game_weight`（官方 `col25`）—— 这与官方
`metrics_pytorch.py` 一致，别给它补一个 `w_*`。

#7 的目标 `std(softmax(scorebelief))` 是 5~20 量级，Huber 的 δ=10 同量级。
⚠ 这与 `katago_v7.SCORE_STDEV_SOFTPLUS_BETA = 0.05`（spec §4.2 的**字面值**，已弃用）
# 矛盾 —— 那个值让 `score_stdev` 预测初值落在 **277.26**、本项公式值 **272.22**
# （δ=10 的 27 倍），初期梯度被常数偏差完全支配。
# ✅ **已裁决为 `SCORE_STDEV_SOFTPLUS_BETA = 1.0`**（2026-10-03）⇒ 实测预测初值
# **13.86**、本项公式值 **8.83**，**落在 δ=10 以内**（Huber 二次段，梯度有效）。
# 推导链与实测数字见该常量上方的注释块。纯常量 ⇒ 结构/参数量/预算不变。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#: 每项的**有效**系数。装配处只乘这里，不再另乘公式里的常数。
LOSS_COEFFS = {
    'policy': 1.0,
    'policy_opp': 0.15,
    'value': 1.20,
    'ownership': 1.5,
    'scorebelief_pdf': 0.020,
    'scorebelief_cdf': 0.020,
    'score_stdev': 0.001,
    'score_mean': 0.0015,
    'lead': 0.0060,
    'scoring': 0.25,
    'futurepos': 1.0,      # ⚠ 0.25 已内嵌在公式里（spec §4.5 #11）
    'seki': 1.0,           # ⚠ 自适应因子，见 `_seki_adaptive_scale`
}

#: scorebelief 的桶数与中心（spec §4.4：桶数 = 2*(361+60) = 842，mid = 421）。
SCORE_DISTR_BINS = 842
SCORE_DISTR_MID = 421

#: 目标写入的两个桶的偏移：`bin_lo = center + 420`、`bin_hi = center + 421`。
#:
#: **这个偏移是从官方 reanalysis 数据反解出来的，不是猜的。** 取
#: `katago/stdata/2026-08-25npzs.tgz` 里 400 个「恰有 2 个非零且相邻」的
#: scoreDistr 行，与 `globalTargetsNC[:,20]`（实际终局分差）配对：
#:
#:     score=−31.6798 → center=−32 → bins (388,389)，frac_lo=0.18
#:     score=−17.2670 → center=−17 → bins (403,404)，frac_lo=0.77
#:     score= +1.4250 → center=  1 → bins (421,422)，frac_lo=0.08
#:     score=  0（和棋）→ center= 0 → bins (420,421)，frac_lo=0.50
#:
#: 四行全部逐位吻合，且和棋那一行**自动**退化成中心两桶 50/50 —— 不需要
#: 任何特判。`test_katago_v7_loss.py::test_score_distr_bin_offset_matches_official`
#: 把这四条钉死。
SCORE_DISTR_BIN_OFFSET = SCORE_DISTR_MID - 1

#: seki 头的 4 个通道：前 3 个是符号类（−1/0/+1），第 4 个是「中性」。
SEKI_SIGN_CHANNELS = 3
SEKI_TOTAL_CHANNELS = 4

#: seki 自适应权重的两个常数（spec §4.5 #12：`8·0.005/(0.005+EMA)`）。
SEKI_ADAPT_NUM = 8.0 * 0.005
SEKI_ADAPT_DEN = 0.005
SEKI_EMA_MOMENTUM = 0.99


def huber(pred, target, beta):
    """Huber，口径与 `train_sft.huber_loss` **逐位一致**（= `smooth_l1`）。

    单独实现而不是 `from scripts.train_sft import huber_loss`：那个 import 会把
    整个训练脚本（torch_npu 探测、argparse、SwanLab…）拖进 loss 模块，而 loss
    要被单测直接引用。`test_katago_v7_loss.py::test_huber_matches_train_sft`
    用两个实现逐位对拍，防止这份复制品漂移。
    """
    return F.smooth_l1_loss(pred, target, beta=beta, reduction='none')


def build_score_distr_target(sb_center, sb_upper, num_bins=SCORE_DISTR_BINS):
    """由 ``(center, upper)`` 展开成 842 桶目标（spec §4.6）。

        center    = round(score)            整数分差
        λ         = score − (center − 0.5)   ∈ [0, 1)
        upperProp = round(λ·100)            ∈ [0, 100]
        写入 center∓0.5 两桶，和恒为 100

    ``sb_upper`` 落在**高**桶（``bin_hi``），``100 − upper`` 落在低桶。
    和棋（``center=0, upper=50``）自动得到中心两桶各 0.5，不需要特判。

    Args:
        sb_center: ``(B,)`` 整数，`round(score)`。
        sb_upper:  ``(B,)`` 整数，``[0, 100]``。

    Returns:
        ``(B, num_bins)`` float32，每行和为 1、恰有 2 个非零。
    """
    b = sb_center.shape[0]
    center = sb_center.reshape(-1).to(torch.long)
    upper = sb_upper.reshape(-1).to(torch.long)
    lo = center + SCORE_DISTR_BIN_OFFSET
    hi = lo + 1
    if int(lo.min()) < 0 or int(hi.max()) >= num_bins:
        raise ValueError(
            f'score 越界：center ∈ [{int(center.min())}, {int(center.max())}] '
            f'映射到 bin ∈ [{int(lo.min())}, {int(hi.max())}]，'
            f'超出 [0, {num_bins - 1}]。分差标签异常或桶数配置不对。')
    out = torch.zeros(b, num_bins, dtype=torch.float32, device=center.device)
    p_hi = (upper.to(torch.float32) / 100.0).unsqueeze(1)
    out.scatter_(1, lo.unsqueeze(1), 1.0 - p_hi)
    out.scatter_(1, hi.unsqueeze(1), p_hi)
    return out


def policy_dense_from_sparse(idx, val, action_size=362, renormalize=True):
    """top-K 稀疏 policy 目标 → 稠密 ``(B, action_size)`` 概率行。

    存储用 top-16（34.2M 行下稠密 int16 要 49.5 GB），**计算用稠密** ——
    spec §4.5 #1 的公式 ``−Σ π·log_softmax(logits)`` 就定义在 362 维分布上，
    截断版本会引入一个 spec 里没有的额外近似。

    截断丢掉的尾部质量默认**按比例摊回**（`renormalize=True`），让每行仍是
    和为 1 的合法分布。丢了多少由标签侧单独记账（``to_v7_labels`` 的
    ``policy_resid``），作为「K=16 够不够大」的哨兵。

    ⚠ 接受 numpy 输入：整条数据链路（`dataset.py` / `katago_npz.py`）产出的
    都是 numpy，在这里 `torch.as_tensor` 比要求上游先转一遍更不容易漏。

    Args:
        idx: ``(B, K)`` int，着法索引（0..action_size−1，含 pass）。
        val: ``(B, K)`` 数值，原始 visit 计数或概率。
    """
    idx = torch.as_tensor(idx)
    val = torch.as_tensor(val, dtype=torch.float32, device=idx.device)
    b, k = idx.shape
    out = torch.zeros(b, action_size, dtype=torch.float32, device=idx.device)
    out.scatter_(1, idx.to(torch.long), val)
    if renormalize:
        s = out.sum(dim=1, keepdim=True)
        out = out / s.clamp_min(1e-12)
    return out


def seki_targets_from_plane(seki):
    """三值平面 ``(B,1,361) ∈ {−1,0,+1}`` → ``(sign (B,361) int64, neutral (B,361))``。

    ⚠ **这一层是 spec 的缺口，需要核对。** loss #12 用的是
    ``CE_sign[3] + 0.5·CE_neutral``，即标签必须同时给「三分类符号」与
    「是否中性」两个量；而 spec §5.4 的标签契约只给了一个 ``seki (B,1,361)``
    的三值平面。

    本函数的默认规则是 **符号 = 平面值，neutral = (平面 == 0)**。这在
    「seki 平面本身就是 KataGo 的 `targetSeki`、0 表示该点未判定」时正确；
    若官方语义是「0 = 中性地（unresolved）」，那么它同时既是符号类之一又是
    中性标记 —— 那就需要标签侧直接给两个量。

    标签契约里**显式提供** ``seki_sign`` / ``seki_neutral`` 时，本函数不被调用
    （见 `KataGoV7Loss.forward`），所以核对结论落地时只需改标签侧，不动本函数。
    """
    b = seki.shape[0]
    flat = seki.reshape(b, -1)
    # 映射 −1/0/+1 → 类 0/1/2：`1 + (正) − (负)`。
    # 写成两段 `+=` 会把 −1 映到 1、+1 映到 2，**丢掉类 0** —— 而类 0 正是
    # 「黑方占这一块」，漏掉它会让 seki 的符号判别整体错一格。
    sign = 1 + (flat > 0.5).long() - (flat < -0.5).long()
    neutral = (flat.abs() <= 0.5).to(flat.dtype)
    return sign, neutral


def _weighted_mean(per_sample, weight):
    """``mean_b(weight_b · per_sample_b)``（spec §4.5 的装配口径）。

    刻意**不**除以 ``Σw``：除以 Σw 会让「该批全是 w=0 的行」把这一项放大到
    噪声水平（分母趋零）。`samplewise` 的语义是逐样本损失对 batch 取均值，
    行权重只作为逐样本的乘子。
    """
    if weight is None:
        return per_sample.mean()
    return (per_sample * weight).mean()


class KataGoV7Loss(nn.Module):
    """12 项 loss 的装配（spec §4.5）。

    `forward` 收 `NbtTfNet.forward` 的输出 dict 与一个标签 dict，返回
    ``{'loss': 标量, 'terms': {每项的已乘系数已乘权重的值}}``。**返回逐项值**
    是刻意的：13 项权重里有 4 个是 0 或自适应，只看总 loss 无法判断是哪一项
    在漂，而逐项曲线是唯一能在训练早期抓住「第 5 项权重写反了」的手段。
    """

    def __init__(self, action_size=362, num_bins=SCORE_DISTR_BINS,
                 coeff=None):
        super().__init__()
        self.action_size = int(action_size)
        self.num_bins = int(num_bins)
        self.coeff = dict(LOSS_COEFFS)
        if coeff:
            self.coeff.update(coeff)
        # seki 的自适应权重要跨 step 记住 EMA（注册成 buffer ⇒ 随 checkpoint 走）
        self.register_buffer('seki_ema', torch.tensor(0.0))

    def extra_repr(self):
        return f'action_size={self.action_size}, num_bins={self.num_bins}'

    # ---- seki 自适应权重（spec §4.5 #12）----
    def _seki_adaptive_scale(self, seki_loss):
        """``8·0.005/(0.005 + EMA)``。

        seki 在自对弈局里**极稀有**，绝对损失长期接近 0，固定系数会让这一项
        永远学不动。分母那个 0.005 是「多大算大」的锚：EMA 远小于它时系数趋于
        ``8·0.005/0.005 = 8``，EMA 追上来后系数回落到 O(1)。

        ⚠ EMA 只在**训练态**更新：`forward` 里用 `self.training` 挡着，
        否则 eval/推理会顺手改 buffer，而 `.eval()` 下反复跑同一个 batch 得到
        不同 loss 是最難查的一类 bug。
        """
        cur = seki_loss.detach()
        if self.training:
            with torch.no_grad():
                if float(self.seki_ema) == 0.0:
                    self.seki_ema.copy_(cur)
                else:
                    self.seki_ema.mul_(SEKI_EMA_MOMENTUM).add_(
                        cur * (1.0 - SEKI_EMA_MOMENTUM))
        return SEKI_ADAPT_NUM / (SEKI_ADAPT_DEN + self.seki_ema)

    # ---- 标签取用（各带一个「有则用、无则由 one-hot / 稀疏目标推出」的退路）----
    def _policy_target(self, labels, key, fallback_move):
        if key in labels and labels[key] is not None:
            t = labels[key]
            if t.dim() == 2 and t.shape[1] == self.action_size:
                return t.to(torch.float32)
        sparse = key + '_sparse'
        if sparse in labels and labels[sparse] is not None:
            pair = labels[sparse]
            return policy_dense_from_sparse(pair[0], pair[1], self.action_size)
        if fallback_move is None:
            raise KeyError(f'标签里既没有稠密 {key}，也没有 {sparse}，'
                           f'也没有可退化的着法列')
        return F.one_hot(fallback_move.reshape(-1).to(torch.long),
                         self.action_size).to(torch.float32)

    def forward(self, out, labels):
        b = out['policy_logits'].shape[0]
        dev = out['policy_logits'].device
        n_sq = 19 * 19
        w = labels.get('w') or {}
        ones = torch.ones(b, device=dev)

        def T(x, dtype=torch.float32):
            """numpy → torch（整条数据链路都是 numpy；要求上游先转一遍
            只会让某个新字段静默留在 numpy 上，而 numpy 与 tensor 的混算
            报错位置离出错点很远）。"""
            return torch.as_tensor(x, dtype=dtype, device=dev)

        def w_of(name):
            v = w.get(name)
            return ones if v is None else T(v).reshape(-1)

        # ⚠ seki 的行权重**不能**默认取 w_ownership（spec §4.5 #12 写的是
        # w_ownership，但那是「从 SGF 自造标签」时的口径）。在 stdata 上
        # seki 通道实测**极稀有**（~1e-4 的格子，前两批几乎全 0），而
        # w_ownership 在 91% 的行上是 1 ⇒ 复用它等于让 seki 头在 99.99%
        # 的行上被推向「全中性」，再乘上 #12 那个上限 8 的自适应系数放大。
        # ⇒ 标签**没给** w_seki 时一律按 0 处理（宁可不训，不要训错方向）。
        if 'seki' not in w:
            w = dict(w)
            w['seki'] = torch.zeros(b, device=dev)

        terms = {}
        logp = F.log_softmax(out['policy_logits'].float(), dim=-1)

        # ---- 1 policy（系数 1.0，行权重恒 1）----
        pi = self._policy_target(labels, 'policy_player', labels.get('moves'))
        terms['policy'] = _weighted_mean(
            -(pi * logp[:, 0]).sum(-1), None)

        # ---- 2 π_opp（系数 0.15，局末手权重 0）----
        pi_opp = self._policy_target(labels, 'policy_opp',
                                     labels.get('next_move'))
        terms['policy_opp'] = _weighted_mean(
            -(pi_opp * logp[:, 1]).sum(-1), w_of('policy_opp'))

        # ---- 3 value 三分类 CE（系数 1.20，行权重恒 1）----
        terms['value'] = _weighted_mean(F.cross_entropy(
            out['outcome_logits'].float(),
            T(labels['outcome'], torch.long).reshape(-1),
            reduction='none'), None)

        # ---- 4 ownership（系数 1.5，w_ownership）----
        own_t = T(labels['ownership']).reshape(b, -1)
        own_logit = 2.0 * out['ownership_pretanh'].reshape(b, -1).float()
        own_bce = F.binary_cross_entropy_with_logits(
            own_logit, (own_t + 1.0) * 0.5, reduction='none')
        terms['ownership'] = _weighted_mean(own_bce.sum(-1) / n_sq,
                                            w_of('ownership'))

        # ---- 5/6 scorebelief pdf + cdf（各 0.020，w_score）----
        sb_logits = out['scorebelief_logits'].float()
        sb_tgt = labels.get('score_distr')
        if sb_tgt is None:
            sb_tgt = build_score_distr_target(
                torch.as_tensor(labels['sb_center']).to(dev),
                torch.as_tensor(labels['sb_upper']).to(dev), self.num_bins)
        sb_tgt = T(sb_tgt)
        terms['scorebelief_pdf'] = _weighted_mean(
            -(sb_tgt * F.log_softmax(sb_logits, dim=-1)).sum(-1),
            w_of('score'))
        cdf_p = F.softmax(sb_logits, dim=-1).cumsum(-1)
        cdf_t = sb_tgt.cumsum(-1)
        terms['scorebelief_cdf'] = _weighted_mean(
            (cdf_p - cdf_t).pow(2).sum(-1), w_of('score'))

        # ---- 7 scorestdev 自预测（系数 0.001，**仅 game_weight，无行权重**）----
        sb_std = F.softmax(sb_logits, dim=-1).std(-1)
        terms['score_stdev'] = _weighted_mean(
            huber(out['score_stdev'].float(), sb_std, 10.0),
            None if labels.get('game_weight') is None
            else T(labels['game_weight']).reshape(-1))

        # ---- 8 scoremean（系数 0.0015，w_score，δ=12）----
        score_t = T(labels['score']).reshape(-1)
        terms['score_mean'] = _weighted_mean(
            huber(out['score_mean'].float(), score_t, 12.0), w_of('score'))

        # ---- 9 lead（系数 0.0060，w_lead，δ=8）----
        terms['lead'] = _weighted_mean(
            huber(out['lead'].float(), score_t, 8.0), w_of('lead'))

        # ---- 10 scoring（**0.25 在系数表里**，w_scoring）----
        sc_t = T(labels['scoring']).reshape(b, -1)
        sc_mse = (out['scoring'].reshape(b, -1).float() - sc_t).pow(2).mean(-1)
        terms['scoring'] = _weighted_mean(
            4.0 * (torch.sqrt(0.5 * sc_mse + 1.0) - 1.0), w_of('scoring'))

        # ---- 11 futurepos（**0.25 已内嵌在公式里**，w_futurepos）----
        fut_t = T(labels['futurepos']).reshape(b, 2, n_sq)
        fut_sq = (torch.tanh(out['futurepos'].reshape(b, 2, n_sq).float())
                  - fut_t).pow(2)
        chan_w = torch.tensor([1.0, 0.25], device=dev).reshape(1, 2, 1)
        terms['futurepos'] = _weighted_mean(
            0.25 * (fut_sq * chan_w).reshape(b, -1).sum(-1) / math.sqrt(n_sq),
            w_of('futurepos'))

        # ---- 12 seki（自适应，w_seki）----
        if 'seki_sign' in labels and 'seki_neutral' in labels:
            sign_t = T(labels['seki_sign'], torch.long).reshape(b, n_sq)
            neu_t = T(labels['seki_neutral']).reshape(b, n_sq)
        else:
            sign_t, neu_t = seki_targets_from_plane(
                torch.as_tensor(labels['seki']).reshape(b, 1, n_sq))
        seki_logits = out['seki_logits'].float()
        # 逐 (样本, 格) 对 3 个符号类做 CE：把类维放中间 ⇒
        # cross_entropy(input=(B,3,N), target=(B,N)) → (B,N)。
        # ⚠ 误写成 reshape(B*3, N) 会让 cross_entropy 把 361 当类维，
        # 报「input batch_size 12 vs target 1444」—— 一眼能看出，但更糟的
        # 写法是 (B,3,N)→(B*N,3) 再转置，形状对而语义错，静默学错。
        ce_sign = F.cross_entropy(
            seki_logits[:, :SEKI_SIGN_CHANNELS].reshape(
                b, SEKI_SIGN_CHANNELS, n_sq),
            sign_t.reshape(b, n_sq), reduction='none')
        ce_neu = F.binary_cross_entropy_with_logits(
            seki_logits[:, SEKI_SIGN_CHANNELS].reshape(b, n_sq), neu_t,
            reduction='none')
        seki_raw = (ce_sign.sum(-1) + 0.5 * ce_neu.sum(-1)) / n_sq
        adaptive = self._seki_adaptive_scale(seki_raw.detach().mean())
        terms['seki'] = _weighted_mean(seki_raw * adaptive, w_of('seki'))

        # ---- 装配：逐项乘**有效**系数（只乘一次）----
        weighted = {k: self.coeff[k] * v for k, v in terms.items()}
        total = sum(weighted.values())
        return {'loss': total, 'terms': terms, 'weighted': weighted,
                'seki_adaptive_scale': adaptive.detach()}
