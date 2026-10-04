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
 这与 `katago_v7.SCORE_STDEV_SOFTPLUS_BETA = 0.05`（spec §4.2 的**字面值**，已弃用）
# 矛盾 —— 那个值让 `score_stdev` 预测初值落在 **277.26**、本项公式值 **272.22**
# （δ=10 的 27 倍），初期梯度被常数偏差完全支配。
# **已裁决为 `SCORE_STDEV_SOFTPLUS_BETA = 1.0`**（2026-10-03）⇒ 实测预测初值
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
    # varTimeLeft（官方 sv3Mul 六通道的下标 3）。系数与 lead 同档：两者都
    # 是「局面不确定性」的标量，量纲都是 0~数百，Huber δ 取 8。
    # 训练数据 col22 里存的**已经是最终物理量**（见 COL_VAR_TIME_LEFT），
    #   而 `out['var_time_left']` 已乘过 VARIANCE_TIME_MULTIPLIER=40，
    #   两者口径一致，可直接回归。
    'var_time_left': 0.0060,
    'scoring': 0.25,
    'futurepos': 1.0, # 0.25 已内嵌在公式里（spec §4.5 #11）
    'seki': 1.0, # 自适应因子，见 `_seki_adaptive_scale`
}

#: scorebelief 的桶数与中心（spec §4.4：桶数 = 2*(361+60) = 842，mid = 421）。
#: 软标签 policy 项的**全局缩放**（对应 12 通路CLI 的 ``--soft-weight``）。
#:
#: 它**不是** ``LOSS_COEFFS`` 里的一项 —— 那是「有哪些 term」的清单，加进去会
#:   破坏 ``set(terms) == set(LOSS_COEFFS)`` 这条被测试钉住的不变量。
#:   它是 policy 项的**量级旋钮**：软 CE 的分母恒为 B（不是 Σmask），所以软项
#:   的量级随「本批软行占比」线性变化，需要一个固定系数把它标定回来。
POLICY_SOFT_WEIGHT = 1.0

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
#: ``score_stdev`` 自预测目标的**数值下界**（见 :meth:`KataGoV7Loss.forward` 第 7 项
#: 的注释）。它唯一能挡住的极端是方差 `v` 下溢成**恰好 0** —— 那时 `sqrt` 的反向
#: 是 `1/(2·0) = inf`。
#:
#: **它不是** fp16 溢出保护（2026-10-04 实测推翻的旧说法）：曾以为
#: `1/(2·std) ≈ 4.27e6`（std≈1.17e-7）会溢出。实测在 B=3000 / 842 桶 / coeff=0.001
#: 下 ``dL/dv`` 只有 **1.53**，因为那个因子乘的是 ``coeff/B ≈ 3.3e-7`` 这个极小的
#: 上游梯度 —— 离 fp16 上限 65504 差四个数量级。见 ``tmp/coding/measure_stdev_grad.py``。
#: 真实模型里 842 桶恰好等概率是零测度，所以这层是兜底，不是已观测到的故障。
SCORE_STDEV_TARGET_FLOOR = 1e-3


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

     接受 numpy 输入：整条数据链路（`dataset.py` / `katago_npz.py`）产出的
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

     **这一层是 spec 的缺口，需要核对。** loss #12 用的是
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


def _weighted_mean(per_sample, weight, probe=None, tag=''):
    r"""``mean_b(weight_b · per_sample_b)``（spec §4.5 的装配口径）。

    刻意**不**除以 ``Σw``：除以 Σw 会让「该批全是 w=0 的行」把这一项放大到
    噪声水平（分母趋零）。`samplewise` 的语义是逐样本损失对 batch 取均值，
    行权重只作为逐样本的乘子。

     **`w==0` 的行必须先摘掉非有限值**（2026-10-04 云端 910A 实测
       `(inf * 0).mean()` = **NaN**）。
       那些行按定义贡献恰好 0，而「inf × 0 = NaN」（IEEE-754）会让**整项**
       变成 NaN ⇒ 系数非 0 的主目标也会被一行坏数据带崩。
       典型触发：`game_weight==0` 的行 + 该行的 `per_sample` 溢出。

     **`w` 自己也可能是 `inf`** —— 这是真机 NaN 的**源头**（2026-10-04 定位）：

       实测 ``data/sgf_19x19_full.npz`` 的 ``game_weights`` 有
       **2,072,682 / 34,202,713 = 6.06% 是 inf**（其余浮点列全干净）。
       来源是 ``build_dataset.compute_game_weight`` 的 ``np.exp(avg/20)``：
       ``parse_player_rating`` 的正则 ``(\d+)([dk])`` 会从棋手名里抓到荒谬的
       大数（如 ``KGS:123456``）⇒ ``exp(61729)`` = inf。

       为什么下面 ``w==0`` 那条防护挡不住它：``inf != 0`` ⇒ ``zero`` 为 False
       ⇒ 根本不进净化分支。而真正的杀伤在**反向**：

           v   = mean(p * inf) = inf                    # 前向
           v   ← nan_to_num ← c=0                       # 段 1 的 c==0 净化
           grad_v   = 0                                  # c=0 乘出来的
           grad_p   = grad_v * w = 0 * inf = **NaN**     # IEEE-754

       即「**净化了前向、没净化反向**」：日志显示
       ``加权总 loss 有限（该项系数为 0，已净化）``、``坏在操作数`` 是空的
       （pred/std 确实都有限），而梯度已经是 NaN。NaN 从 ``score_stdev``
       进 value head、再经 trunk 污染**全部**参数 —— 真机实测
       5,562,121 / 5,562,121（连 ``stem`` 都是），与这条链完全吻合。

       ⇒ 无穷权重在任何口径下都没有意义，按 ``w==0`` 同一条语义把它摘掉。

    Args:
        probe: 可选的诊断累加器（``{项名: 被净化的行数}``）。**净化是静默的**
            —— 不记下来就成了「这一项坏了但没人知道」，而段 1 有 9 项系数为 0，
            它们坏掉时对总 loss 毫无影响，正因如此更需要把线索报出来。
            非有限权重记在 ``'{项名}:w_inf'`` 下，与 ``w==0`` 那条分开报。
    """
    if weight is None:
        return per_sample.mean()
    p = per_sample.reshape(-1)
    w = weight.reshape(-1)
    bad_w = ~torch.isfinite(w)
    if bool(bad_w.any()):
        if probe is not None:
            probe[tag + ':w_inf'] = int(bad_w.sum())
        # 摘掉后这些行的 w 恒为 0 ⇒ 与 w==0 走同一条路（p 也会被下方的
        # `where` 净化），于是 `p * w` 与它的反向都落在有限值上。
        w = w.masked_fill(bad_w, 0.0)
    zero = (w == 0)
    if bool(zero.any()):
        clean = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        if probe is not None and not bool(torch.isfinite(p).all()):
            # 记「有几行本该是 0 而实际是非有限」—— 这是坏数据的直接证据
            probe[tag] = int((zero & ~torch.isfinite(p)).sum())
        # 只在**被丢弃**的那些行上清成 0：w!=0 的行原样保留，
        # 所以 `per_sample` 有限时结果与原来**逐位相同**（分母仍是 batch 大小，
        # 不改变 spec §4.5 的口径）。
        p = torch.where(zero, clean, p)
    return (p * w).mean()


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

         EMA 只在**训练态**更新：`forward` 里用 `self.training` 挡着，
        否则 eval/推理会顺手改 buffer，而 `.eval()` 下反复跑同一个 batch 得到
        不同 loss 是最難查的一类 bug。
        """
        cur = seki_loss.detach()
        if self.training:
            with torch.no_grad():
                # **NaN/Inf 一律不写进 buffer**（2026-10-04 云端 910A 实跑）。
                #   `seki_ema` 是**注册 buffer** ⇒ 一次写入就是**永久**的：之后每一步
                #   的 adaptive scale 都是 NaN ⇒ `terms['seki']` 是 NaN ⇒ 加权总
                #   loss 是 NaN ⇒ **全部**参数梯度 NaN ⇒ GradScaler 永远跳步。
                #
                #   实测症状与这个机制逐条吻合：缩放值 10240→5120→…→160 一路
                #   降到最低仍 100% 跳过，且每步的 nan 参数计数**完全一样**
                #   （211/98/8）—— 一个「每步重新发生」的溢出会让计数抖动，
                #   而一个被 buffer 记住的 NaN 会给出恒定的结果。
                #
                # 更糟的是它**进 state_dict** ⇒ 存下来的 `.pth` 与
                #   `--resume` 都带着毒，换台机器续训照样每步 NaN。
                #
                #   为什么此前没被发现：段 1 的 seki 系数是 1.0 但**该项在 CPU 上
                #   恒为 0**，而 CPU 的 kernel 不产生 NaN ⇒ 只有真机 NPU 才会毒化。
                if not torch.isfinite(cur):
                    # 不 copy_：保留上一个**有限**的 EMA（初值 0.0 ⇒ scale = 8.0，
                    # 即「按 seki 极稀有处理」，正是这个 EMA 初值的语义）。
                    # 刻意不 raise：训练不该因为一个自适应系数而崩，而下一步的
                    # 逐项有限性检查（见 forward 末尾）会把「这一项坏了」报出来。
                    return self._seki_fallback_scale()
                if float(self.seki_ema) == 0.0:
                    self.seki_ema.copy_(cur)
                else:
                    self.seki_ema.mul_(SEKI_EMA_MOMENTUM).add_(
                        cur * (1.0 - SEKI_EMA_MOMENTUM))
        # 兜底：即使 buffer 在别处（加载旧 checkpoint、手工改写）已经是 NaN，
        #   也不能让它进 loss。`float()` 只在**标量** buffer 上调用，代价可忽略。
        if not torch.isfinite(self.seki_ema):
            return self._seki_fallback_scale()
        return SEKI_ADAPT_NUM / (SEKI_ADAPT_DEN + self.seki_ema)

    def _seki_fallback_scale(self):
        """「seki 极少」那个系数（`8·0.005/0.005 = 8`），**以 tensor 返回**。

         必须是 tensor：本函数的返回值会进 `return {... 'seki_adaptive_scale':
          adaptive.detach()}`，给 Python float 会在那里抛
          `AttributeError: 'float' object has no attribute 'detach'` ——
          而那正是「NaN 兜底路径」本身，它一旦抛异常就等于**没兜**，
          反而把唯一一次能说出「seki 坏了」的机会也弄没了。
        """
        return torch.as_tensor(SEKI_ADAPT_NUM / SEKI_ADAPT_DEN)

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
        #: 哪个**操作数**坏了（`项名:操作数`）。逐项点名只说「哪一项」，而
        #: 「这一项」内部往往有两个来源不同的操作数（本轮 `score_stdev` 就是：
        #: 预测 `out['score_stdev']` 来自 ValueHead、目标 `std(softmax)` 来自
        #: scorebelief 头）。不区分就还得再猜一轮 —— 本机与 NPU 的 kernel
        #: 不同，只有真机能回答，而每次真机跑一轮都要几分钟。
        _bad_operands = []
        #: 被 `_weighted_mean` 按「w==0」静默净化掉的行数（见该函数 docstring）
        _sanitized = {}

        def _wm(tag, per_sample, weight):
            return _weighted_mean(per_sample, weight, _sanitized, tag)

        def T(x, dtype=torch.float32):
            """numpy → torch（整条数据链路都是 numpy；要求上游先转一遍
            只会让某个新字段静默留在 numpy 上，而 numpy 与 tensor 的混算
            报错位置离出错点很远）。"""
            return torch.as_tensor(x, dtype=dtype, device=dev)

        def w_of(name):
            v = w.get(name)
            return ones if v is None else T(v).reshape(-1)

        # seki 的行权重**不能**默认取 w_ownership（spec §4.5 #12 写的是
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
        w_soft = POLICY_SOFT_WEIGHT

        def soft_ce(target, ch):
            """某一通道的软 CE，口径与 12 通路 `soft_cross_entropy` 逐位一致。

             逐行**二选一**（`mask=0` 的行贡献恰好 0，**不退化成 one-hot CE**）。
             分母恒为 B（不是 Σmask）—— 软项量级随「本批软行占比」线性变化，
              `policy_soft_weight` 就是标定量级的旋钮。
             一律 fp32、绝不上 fp64（910A 无 fp64 硬件，设备侧 fp64 会挂 AICPU
              且报错栈指向无关算子）。
            """
            tgt = T(target).reshape(-1, self.action_size)
            per_row = -(tgt * logp[:, ch]).sum(-1)            # (B,)
            mask = T(soft_mask).reshape(-1).to(per_row.dtype)
            if mask.shape[0] != per_row.shape[0]:
                raise ValueError(
                    f'soft_mask 长度 {mask.shape[0]} 与 batch {per_row.shape[0]} 不符')
            return w_soft * (per_row * mask).mean()

        # ---- 1 policy（系数 1.0，行权重恒 1）----
        # 软标签（2026-10-04）：`labels['soft']` 形状是 **(B, A)**（只管
        #    **通道 0 = π**），不是 (B,K,A)。stdata 分片里
        #    `policy_player_prob` 存的就是 KataGo 搜索访问分布（实测一行
        #    853/16/12/4/1/1，和 887），归一化后即是软标签。
        #    缺席时退回 one-hot（board 级路径的老行为）。
        soft = labels.get('soft')
        soft_mask = labels.get('soft_mask')
        # `soft_mask` 全 0 是 `SupervisedDataset` 约定的「**本批无软标签**」
        #   （未挂 `--soft-index` 时它就是全 0）。若照字面走软 CE，policy 会拿到
        #   **恰好 0** 的梯度 —— 不报错、loss 照降、policy 根本没学。
        #   所以「有没有软标签」以 mask 是否有命中为准，缺席时退回 one-hot。
        #   `V7PackedDataset` 的 mask 恒为 1（stdata 每行都带 KataGo 搜索分布），
        #   所以 C 段永远走软 CE；board 级路径没挂索引时永远走 one-hot。
        #   两者都不会「时有时无」。
        _has_soft = (soft is not None and soft_mask is not None
                     and bool(torch.as_tensor(soft_mask).ne(0).any()))
        if _has_soft:
            terms['policy'] = soft_ce(soft, 0)
        else:
            pi = self._policy_target(labels, 'policy_player', labels.get('moves'))
            terms['policy'] = _wm('policy', 
                -(pi * logp[:, 0]).sum(-1), None)

        # ---- 2 π_opp（系数 0.15，局末手权重 0）----
        # 必须用**对手侧**的目标：`labels['soft_opp']`（来自分片的
        #    `policy_opp_prob`）。旧实现拿 player 的着法去填这一项，
        #    于是 #1 与 #2 拿到**同一个**目标，π_opp 白训。
        soft_opp = labels.get('soft_opp')
        if _has_soft and soft_opp is not None:
            terms['policy_opp'] = _wm('policy_opp', 
                soft_ce(soft_opp, 1), w_of('policy_opp'))
        else:
            pi_opp = self._policy_target(labels, 'policy_opp',
                                         labels.get('next_move'))
            terms['policy_opp'] = _wm('policy_opp', 
                -(pi_opp * logp[:, 1]).sum(-1), w_of('policy_opp'))

        # ---- 3 value 三分类 CE（系数 1.20，行权重恒 1）----
        terms['value'] = _wm('value', F.cross_entropy(
            out['outcome_logits'].float(),
            T(labels['outcome'], torch.long).reshape(-1),
            reduction='none'), None)

        # ---- 4 ownership（系数 1.5，w_ownership）----
        own_t = T(labels['ownership']).reshape(b, -1)
        own_logit = 2.0 * out['ownership_pretanh'].reshape(b, -1).float()
        own_bce = F.binary_cross_entropy_with_logits(
            own_logit, (own_t + 1.0) * 0.5, reduction='none')
        terms['ownership'] = _wm('ownership', own_bce.sum(-1) / n_sq,
                                            w_of('ownership'))

        # ---- 5/6 scorebelief pdf + cdf（各 0.020，w_score）----
        sb_logits = out['scorebelief_logits'].float()
        sb_tgt = labels.get('score_distr')
        if sb_tgt is None:
            sb_tgt = build_score_distr_target(
                torch.as_tensor(labels['sb_center']).to(dev),
                torch.as_tensor(labels['sb_upper']).to(dev), self.num_bins)
        sb_tgt = T(sb_tgt)
        terms['scorebelief_pdf'] = _wm('scorebelief_pdf', 
            -(sb_tgt * F.log_softmax(sb_logits, dim=-1)).sum(-1),
            w_of('score'))
        cdf_p = F.softmax(sb_logits, dim=-1).cumsum(-1)
        cdf_t = sb_tgt.cumsum(-1)
        terms['scorebelief_cdf'] = _wm('scorebelief_cdf', 
            (cdf_p - cdf_t).pow(2).sum(-1), w_of('score'))

        # ---- 7 scorestdev 自预测（系数 0.001，**仅 game_weight，无行权重**）----
        # `std` **不能**直接调 `F.softmax(...).std(-1)`（2026-10-04 云端 910A
        #   实测点名到本项：加权 loss 非有限，逐项点名 = ['score_stdev']）。
        #
        # 根因：scorebelief 有 **842 个桶**，初始化时 logits 近均匀 ⇒ p ≈ 1/842
        # ≈ 1.19e-3 且彼此相差极小 ⇒ **方差 ≈ 1e-10**。这个量级下
        # 「朴素公式」`E[x²] − E[x]²` 会发生**灾难性抵消**：两个 ~1.4e-6 的数
        # 相减得到一个微小**负数** ⇒ `sqrt(负数)` = **NaN**。
        #
        # CPU 的 `torch.std` 用 Welford / 两遍算法，稳；NPU 的规约核用朴素公式，
        # 于是**只有真机炸** —— 本地 fp32 / fp16 / bf16 全都复现不出来
        # （实测三项梯度 NaN 张量均为 0/322）。
        #
        # 修法：**两遍 + fp32**。`mean((p − μ)²)` 恒非负（每一项都是平方），
        # 在 fp32 下 842 个元素的抵消也远不到出问题的量级。代价是多一个
        # `(B, 842)` 的减法/平方，可忽略（这一项的系数在段 1 是 0）。
        _sb_p = F.softmax(sb_logits, dim=-1).float()
        _sb_mu = _sb_p.mean(dim=-1, keepdim=True)
        sb_std = (_sb_p - _sb_mu).pow(2).mean(dim=-1).sqrt()
        # **目标必须 detach**。
        #
        #   `#7 scorestdev` 是 13 项里**唯一**「目标不是标签、而是模型自己输出的
        #   派生量」的一项：`sb_std = std(softmax(sb_logits))` 来自 `scorebelief_head`。
        #   其余 12 项的目标都是**标签常量**，autograd 不会往它们回传梯度。
        #
        #   detach 的理由是**语义**，不是数值：自预测蒸馏的目标就该是常量。官方
        #   `losses.cpp` 算 `avgStddev` 时走 `forwardEval` / no-grad —— 它要的是
        #   「当前预测的分布宽度」这个**读数**。留着梯度等于让 scorebelief 头同时
        #   被两股方向相反的力撕：CE（#5/#6）要它锐利，这条梯度要它均匀。
        #
        #   **不要把它当成 NaN 修复**（2026-10-04 实测推翻的旧说法）：我曾认为
        #   `d(sqrt v)/dv = 1/(2·std) ≈ 4.27e6`（std≈1.17e-7）会在 fp16 上溢成 inf，
        #   再乘上游 0 变 NaN。实测不成立：`tmp/coding/measure_stdev_grad.py` 在
        #   B=3000 / 842 桶 / coeff=0.001 下量得 dL/dstd=3.33e-07、**dL/dv=1.53**
        #   —— 那个 4.27e6 乘的是 coeff/B 这个极小的上游梯度，不是 0，也不是 1。
        #   峰值 1.53 离fp16 上限 65504 差四个数量级，无 NaN 无 inf。真正的 NaN
        #   源头**仍未定位**，见 :data:`_bad_operands` 的操作数级归因。
        sb_std = sb_std.detach()
        # 下界是**独立的数值兜底**，与 detach 无关（因此这里不需要梯度注释）：
        # 它唯一能挡住的极端是 `v` 下溢成**恰好 0**（`sqrt` 的反向 = 1/(2·0) = inf）。
        # 真实模型里 842 个桶恰好等概率是零测度，所以这层是 belt-and-suspenders，
        # 不是已观测到的故障。取值 1e-3 远低于任何真实分布展度，不改口径。
        sb_std = sb_std.clamp_min(SCORE_STDEV_TARGET_FLOOR)
        # 操作数级归因：下一轮日志能直接看出是「预测」还是「目标」坏掉，
        # 而不必再猜（真机与本机的 std 实现不同，只有真机能回答）。
        # 下面的 `huber(out['score_stdev'].float(), sb_std, ...)` 是
        #   `tests/test_katago_v7_budget.py::test_score_stdev_loss_term_is_inside_
        #   huber_delta_at_the_ruled_out_beta` 按**字面量**钉住的（它要保证本路
        #   的 δ 与测试常量同步）⇒ **不要**把 `out['score_stdev'].float()` 提成一个
        #   中间变量，否则那条门禁会假红。
        if not bool(torch.isfinite(out['score_stdev']).all()):
            _bad_operands.append('score_stdev:pred')
        if not bool(torch.isfinite(sb_std).all()):
            _bad_operands.append('score_stdev:std')
        terms['score_stdev'] = _wm('score_stdev', 
            huber(out['score_stdev'].float(), sb_std, 10.0),
            None if labels.get('game_weight') is None
            else T(labels['game_weight']).reshape(-1))

        # ---- 8 scoremean（系数 0.0015，w_score，δ=12）----
        score_t = T(labels['score']).reshape(-1)
        terms['score_mean'] = _wm('score_mean', 
            huber(out['score_mean'].float(), score_t, 12.0), w_of('score'))

        # ---- 9 lead（系数 0.0060，w_lead，δ=8）----
        terms['lead'] = _wm('lead', 
            huber(out['lead'].float(), score_t, 8.0), w_of('lead'))

        # ---- 9b varTimeLeft（官方 sv3[3]，系数 0.0060，δ=8）----
        # **不能用 `w_of('lead')`**。实测（zzb28c512nfd4 三个成员、12748 行）
        #   各权重列与 col22 非零模式的一致率：
        #     col29 w_lead          19.7% 非零，一致率仅 **27.4%**  ← 错
        #     col25 global_weight  100%  非零，一致率 87.4%（= col22 自身非零率）
        #   `w_lead` 的门控跟着 **lead 有没有值**（col21）走，而 varTimeLeft
        #   在 lead 缺失的 66% 行里照样有值。复用它会白白丢掉三分之二的数据。
        # ⇒ 用 `game_weight`（与 `score_stdev` 一致），即整行有效即参与。
        # 标签或输出任一缺失时**跳过**而不是喂 0 —— 喂 0 会把这一路往
        #   「方差恒 0」的方向硬拉，比不训练更糟。`out.get` 而非 `out[...]`
        #   是为了让只构造了部分输出的测试桩也能跑通（真实模型恒有该键）。
        vtl_t = labels.get('var_time_left')
        vtl_p = out.get('var_time_left')
        if vtl_t is not None and vtl_p is not None:
            terms['var_time_left'] = _wm('var_time_left', 
                huber(vtl_p.float(), T(vtl_t).reshape(-1), 8.0),
                None if labels.get('game_weight') is None
                else T(labels['game_weight']).reshape(-1))

        # ---- 10 scoring（**0.25 在系数表里**，w_scoring）----
        sc_t = T(labels['scoring']).reshape(b, -1)
        sc_mse = (out['scoring'].reshape(b, -1).float() - sc_t).pow(2).mean(-1)
        terms['scoring'] = _wm('scoring', 
            4.0 * (torch.sqrt(0.5 * sc_mse + 1.0) - 1.0), w_of('scoring'))

        # ---- 11 futurepos（**0.25 已内嵌在公式里**，w_futurepos）----
        fut_t = T(labels['futurepos']).reshape(b, 2, n_sq)
        fut_sq = (torch.tanh(out['futurepos'].reshape(b, 2, n_sq).float())
                  - fut_t).pow(2)
        chan_w = torch.tensor([1.0, 0.25], device=dev).reshape(1, 2, 1)
        terms['futurepos'] = _wm('futurepos', 
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
        # 误写成 reshape(B*3, N) 会让 cross_entropy 把 361 当类维，
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
        terms['seki'] = _wm('seki', seki_raw * adaptive, w_of('seki'))

        # ---- 装配：逐项乘**有效**系数（只乘一次）----
        # **净化数值，但绝不切断计算图**（2026-10-04 云端 910A 实跑）。
        #   `0.0 * NaN == NaN`（IEEE-754，不是 0）⇒ 只要 13 项里任何一项算坏，
        #   加权总 loss 就是 NaN，而它一 NaN，**每一个**收到梯度的参数张量都是
        #   NaN —— 实测 317/322，正是这个签名。
        #
        # **不能**用「c==0 就直接给 0.0」那种写法（我先写了这个版本）：
        #     那会把这些项从图里摘掉 ⇒ 它们的参数 `grad=None` ⇒ DDP 抛
        #     `Expected to have finished reduction in the prior iteration`
        #     （真机 2 卡实测，参数索引 308-321）。**安静地剪掉梯度不是修复，
        #     是换一个更响的崩溃。**
        #
        #   净化**只对 c==0 的项**：它们按定义不参与优化，坏掉时的正确行为就是
        #   「这一项当 0」，净化是如实的。而 c!=0 的项（段 1 的
        #   policy / policy_opp / value / futurepos）坏了必须**原样传出 NaN** ——
        #   那是整批数据废掉的信号，净化掉就变成「静默训成 0」：
        #   loss 曲线正常、指标正常，而模型什么也没学。
        weighted = {}
        for k, v in terms.items():
            c = self.coeff[k]
            if c == 0.0:
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            weighted[k] = c * v
        total = sum(weighted.values())
        # 逐项指认「哪一项坏了」。
        #
        # **不能只在 `total` 非有限时才扫**：正因上面保证了 c==0 的项不进加总，
        #   `total` 对这些项是**恒有限**的 ⇒ 那种写法下它们永远不会被点名，
        #   而它们恰恰是唯一「坏了但看不出来」的一类。
        #
        # 成本可控：段 1 的 9 个零系数项经 `_weighted_mean` 后都是 `(b,)` 标量，
        # stack 成一个 `(9, b)` 一次 `isfinite().all()` 即可 ⇒ **一次** D2H
        # 同步，而本循环每步本来就有 `float(_gn)`（clip_grad_norm_ 的返回值）
        # 那一次同步，所以不多付。非段 1（c≠0 的项）坏掉时 `total` 会非有限，
        # 那时再补扫一遍全表。
        _zero_bad = []
        _zero_terms = [v for k, v in terms.items() if self.coeff[k] == 0.0]
        if _zero_terms:
            _stacked = torch.stack([t.reshape(-1) for t in _zero_terms])
            if not bool(torch.isfinite(_stacked).all()):
                _names = [k for k in terms if self.coeff[k] == 0.0]
                _per = torch.isfinite(_stacked).all(dim=1)
                _zero_bad = [n for n, ok in zip(_names, _per.tolist())
                             if not ok]
        if not bool(torch.isfinite(total)):
            _zero_bad = sorted(set(_zero_bad) | {
                k for k, v in terms.items()
                if not bool(torch.isfinite(v).all())})
        bad = sorted(_zero_bad)
        return {'loss': total, 'terms': terms, 'weighted': weighted,
                'nonfinite_terms': bad,
                'nonfinite_operands': sorted(set(_bad_operands)),
                'sanitized_rows': dict(_sanitized),
                'total_finite': bool(torch.isfinite(total)),
                'seki_adaptive_scale': adaptive.detach()}
