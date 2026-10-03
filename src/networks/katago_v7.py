"""KataGo NBT + Transformer 主干与四头（官方 V7 输入 22 空间 / 19 全局）。

设计来源：`docs/superpowers/specs/2026-10-01-katago-nbt-tf-design.md` §3（主干）
与 §4（头）。配置名沿用官方命名法 **`b11c256h4nbttflrs-fson-silu-rsnh`**。

形状常量（spec §3）
------------------
==========  ====  ====================================================
常量         值     约束来源
==========  ====  ====================================================
``C``        256    trunk 宽度
``M``        128    nbt 内宽 = C/2，须被 32 整除
``H``        4      注意力头数 ⇒ ``head_dim = M/H = 32``（CANN 支持集 {16,32,64}）
``F``        384    SwiGLU 隐层 = 1.5C
``B``        11     nbt 块数，全部是 transformer 内块
``G``        0      无 gpool 块
输入          22/19  官方 ``fillRowV7``
==========  ====  ====================================================

三条与本仓既有模型**不同**的语义选择（不是风格偏好，别顺手改回去）
------------------------------------------------------------------
1. **全网无 BatchNorm。** 归一化由两种可学习仿射承担：块内的 ``fson``
   （`NormAct`）与 trunk 末端的 ``rsnh``（`RMSNormMask`）。`fson` 之所以能取代
   BN，靠的是「残差流方差可解析」：第 i 个残差块给栈贡献方差 1，故该块读到的
   累计方差是 ``i+1``，把这个事实写进一个逐通道标量 ``K`` 里即可，见
   ``initialize`` 与 spec §3.3。
2. **无 dropout。** 对齐官方；`AlphaGoNet` 的 ``attn_dropout=0.1`` 不进入本模型。
   双重收益：省激活/掩码两份张量，且 dropout 常是融合注意力的阻碍。
3. **注意力走 ``backbone._sdpa`` 同一个四站点机制**（`backbone.py:581`），
   ``head_dim`` 由 46 变 32。复用它而不是自己调 SDPA，是为了让
   ``_sdpa_force_math`` / flash-attn 开关 / batch 上限对两个模型一致生效 ——
   spec §6.2 的注意力显存账就是按这条路径估的。

⚠ **RoPE 的形状是 spec 唯一未逐字确定的一处**：spec §3.2 只给出参数形状
``(H, 16, 2)``，没有给「2」的语义。本实现取**二维频率**（``[...,0]`` 乘行坐标、
``[...,1]`` 乘列坐标，角度 = 二者加权和），它是 2D 旋转位置编码的直接推广，
且**参数量与 spec 完全一致**。若日后核对 ``model_pytorch.py`` 发现官方是别的
切法，只需改 ``RoPE2D.forward`` 一处 —— 参数形状不变、预算不变。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from src.networks.backbone import RMSNorm, _sdpa

#: spec §3 的形状常量。结构**只由这张表**决定（与 `train_sft.KATAGO_SE_CFG`
#: 同一立场：结构不许由调用方零散覆盖）。
NBT_TF_CFG = {
    # ---- 输入（官方 fillRowV7）----
    'in_channels': 22,
    'global_channels': 19,
    'board_size': 19,
    # ---- 形状 ----
    'trunk_channels': 256,      # C
    'nbt_mid': 128,             # M = C/2
    'num_heads': 4,             # H  ⇒ head_dim = 128/4 = 32
    'ffn_hidden': 384,          # F = 1.5C
    'num_blocks': 11,           # B
    'num_gpool_blocks': 0,      # G
    'num_inner_blocks': 2,      # 每个 nbt 块内的 transformer 块数
    # ---- 头 ----
    'policy_channels': 48,      # P
    'gpool_channels': 48,       # G（policy 头的池化支路宽度）
    'policy_outputs': 2,        # K = 2：π + π_opp
    'value_channels': 48,       # V
    'value_hidden': 96,         # W
    'seki_classes': 4,          # 3 个符号类 + 1 个中性类（spec §4.5 loss #12）
    'futurepos_channels': 2,    # +8 / +32 手
    # ---- scorebelief ----
    'extra_score_distr_radius': 60,   # 桶数 = 2*(361+60) = 842
    'score_distr_components': 8,      # nsb
    # ---- 行为（非结构，可被调用方覆盖）----
    'attn_dropout': 0.0,
    'use_checkpoint': True,
    # ---- 预算锚（spec §6.1；由 tests/test_katago_v7_budget.py 钉死）----
    'params_stem': 50_688,
    'params_global': 4_864,
    'params_block_each': 493_056,
    'params_blocks_total': 5_423_616,
    'params_trunkfinal': 512,
    'params_policy_head': 38_834,
    'params_value_head': 26_886,
    'params_small_heads': 384,
    'params_scorebelief_head': 16_048,
    'params_total': 5_561_832,
}

#: off-board 位置的 logits 惩罚（spec §4.1）。常量、不参与学习 —— 因此
#: **不占参数**，spec §6.1 的 policy 头 38,834 正是按「零参数 BiasMask」记的。
OFF_BOARD_LOGIT = -5000.0

#: ``SiLU`` 的 init gain。官方注释：理论值 √2.8108，为兼容保留 √2.0。
GAIN_SILU = math.sqrt(2.0)

#: stem / global 两条输入投影各自的 init scale（spec §3.4）。
SCALE_INIT_CONV = 0.8
SCALE_GLOBAL_LINEAR = 0.6

#: 2D RoPE 频率的初始化区间：``exp(uniform(log 1/50, log 1))``（spec §3.2）。
ROPE_INIT_MIN = 1.0 / 50.0
ROPE_INIT_MAX = 1.0


def _trunc_normal_(tensor, std):
    """截断正态初始化（±3σ），KataGo 与 spec §3.4 都用它而非 `normal_`。"""
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)


class _ScaledLinear(nn.Linear):
    """带 ``initialize(scale, gain)`` 的 Linear。"""

    def initialize(self, scale=1.0, gain=GAIN_SILU):
        std = scale * gain / math.sqrt(self.in_features)
        _trunc_normal_(self.weight, std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        return self


class _ScaledConv2d(nn.Conv2d):
    """带 ``initialize(scale, gain)`` 的 Conv2d。"""

    def initialize(self, scale=1.0, gain=GAIN_SILU):
        std = scale * gain / math.sqrt(self.in_channels * self.kernel_size[0]
                                       * self.kernel_size[1])
        _trunc_normal_(self.weight, std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        return self


class NormAct(nn.Module):
    """``fson`` 固定方差标量归一化 + 激活（spec §3.1 步骤 1/5）。

        forward:  x → x·(γ·K) + β → SiLU

    **它不做任何归一化统计** —— 这是 ``fson`` 与 BN/BN-free-RMSNorm 的本质区别。
    归一化的「该除以多少」被解析地写死在一个逐通道可学习标量 ``K`` 里，理由见
    模块 docstring 第 1 条。参数量 ``2C``（γ + β），全网**无 running stats**，
    因此与 batch 大小无关，也不会在按 group 采样时漂移。

    ⚠ **γ 的初值是 ``norm_scale / scale``，不是 1。** ``norm_scale`` 就是
    ``K``：第 ``i`` 个 trunk 块取 ``1/√(i+1)``，trunk 末端取 ``1/√(B+1)``。
    忘了乘它，残差流的方差会随深度线性增长，深层激活直接饱和。
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        # 刻意初始化为 0：未经 `initialize()` 的模块不应假装自己已归一化过。
        self.gamma = nn.Parameter(torch.zeros(self.channels))
        self.beta = nn.Parameter(torch.zeros(self.channels))

    def initialize(self, scale=1.0, norm_scale=1.0):
        nn.init.constant_(self.gamma, float(norm_scale) / float(scale))
        nn.init.zeros_(self.beta)
        return self

    def forward(self, x):
        y = torch.addcmul(self.beta.view(1, -1, 1, 1), x, self.gamma.view(1, -1, 1, 1))
        return F.silu(y)


class RMSNormMask(nn.Module):
    """``rsnh``：沿**空间维**的 RMSNorm + 逐通道仿射（spec §3.1 trunk 末端）。

    归一化在 ``(H, W)`` 上求，**通道维独立** —— 与 `backbone.RMSNorm`（在
    通道维上求、用于 token 序列）刻意不同，别混用。参数量 ``2C``。

    数值口径：平方与与 `rsqrt` 一律在 fp32 上做再转回。910A 无 bf16、AMP 走
    fp16，而 fp16 的 ``x²`` 在通道宽 256 时容易溢出到 inf（65504 上限）。
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(self.channels))
        self.beta = nn.Parameter(torch.zeros(self.channels))

    def initialize(self, scale=1.0, norm_scale=1.0):
        nn.init.constant_(self.gamma, float(norm_scale) / float(scale))
        nn.init.zeros_(self.beta)
        return self

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=(2, 3), keepdim=True) + self.eps)
        y = xf * rms * self.gamma.view(1, -1, 1, 1).float() \
            + self.beta.view(1, -1, 1, 1).float()
        return y.to(dtype)


class RoPE2D(nn.Module):
    """2D 可学习旋转位置编码（spec §3.2）。

    参数形状 ``(H, head_dim//2, 2)``：逐**头**、逐**旋转对**、逐**坐标轴**一个
    频率。棋盘上第 ``(r, c)`` 个 token 的第 ``j`` 个旋转对的角度是

        angle = freq[h, j, 0] · r + freq[h, j, 1] · c

    于是 ``q``/``k`` 的第 ``(2j, 2j+1)`` 两个分量被这个角度旋转。注意力分数
    只依赖 ``q·k``，而该内积对 ``(r,c)`` 的依赖化为**相对位移**的函数 ——
    这正是 RoPE 的平移等变性，也正是它比「加一个可学习位置嵌入」更适合
    变长/变尺寸 token 序列的原因。

    频率初值 ``exp(uniform(log 1/50, log 1))``：最长波长跨约 50 格、最短约 1 格，
    覆盖从整盘到邻域的多尺度。乘 ``±1`` 是为了让一部分分量初值为负（对应
    顺/逆时针相反的旋转方向）。
    """

    def __init__(self, num_heads, head_dim):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError('head_dim 必须为偶数（旋转按 (2j, 2j+1) 成对）')
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_pairs = self.head_dim // 2
        self.freq = nn.Parameter(torch.empty(self.num_heads, self.num_pairs, 2))

    def initialize(self):
        log_min, log_max = math.log(ROPE_INIT_MIN), math.log(ROPE_INIT_MAX)
        with torch.no_grad():
            vals = torch.exp(torch.empty(self.num_heads, self.num_pairs, 2)
                             .uniform_(log_min, log_max))
            self.freq.copy_(vals * torch.tensor([1.0, -1.0])
                            .expand_as(vals).contiguous())
        return self

    @staticmethod
    def _rotate(x, cos, sin):
        """把 ``x`` 的相邻分量对按 ``(cos, sin)`` 旋转。"""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos),
                           dim=-1).flatten(-2)

    def forward(self, x, pos):
        """x: ``(B, Hh, N, head_dim)``；``pos``: ``(B, N, 2)`` 整数，**``(行, 列)``**。

        🔴 ``freq[..., 0]`` 乘的是**列**，``freq[..., 1]`` 乘的是**行** ——
        与 ``pos`` 的 ``(行, 列)`` 顺序**相反**，这不是笔误。

        官方 ``desc.cpp`` 的 ``TransformerAttentionDesc::computeRopeCosSin``：

            float freqX = ropeFreqs[(h * numPairs + p) * 2 + 0];
            float freqY = ropeFreqs[(h * numPairs + p) * 2 + 1];
            ...
            for(int y = 0; y < nnYLen; y++)
              for(int x = 0; x < nnXLen; x++) {
                float angle = (float)x * freqX + (float)y * freqY;

        即 **第 0 个频率配 x（列）、第 1 个配 y（行）**。而 ``desc.cpp`` 里紧邻的
        注释说「前 numPairsPerDim 对是 height、接着是 width」—— 那句描述的是
        **非learnable（固定 theta）分支**，其 ``emb = cat([y*freqs, x*freqs])``
        顺序与 learnable 分支**相反**。照那句注释写 learnable 分支就会写反。

        ⚠ **这个错误在任何正方盘面上都测不出来**：19×19 的行列数相同，行列互换
        只是一个有效对称，policy 输出仍然「看着合理」。只有拿官方权重逐位对拍
        才能发现（我们是这么发现的）。
        """
        b, _, n, d = x.shape
        if d != self.head_dim:
            raise ValueError(f'head_dim 不符：传入 {d}，本层 {self.head_dim}')
        pos = pos.to(x.device).reshape(b, n, 2)
        row = pos[:, None, :, 0:1]                     # y
        col = pos[:, None, :, 1:2]                     # x
        f = self.freq.to(x.dtype)                      # (Hh, pairs, 2)
        ang = (f[None, :, :, 0].unsqueeze(2) * col     # freqX * x
               + f[None, :, :, 1].unsqueeze(2) * row)  # freqY * y
        return self._rotate(x, ang.cos(), ang.sin())


class MHSA(nn.Module):
    """多头自注意力，``head_dim = nbt_mid / num_heads``（spec §3.2）。

    ``q`` 在调用 ``_sdpa`` **之前**已乘 ``1/√head_dim``（`_sdpa` 的契约，见
    `backbone.py:584`），这样 flash / mem-efficient 后端与 math 路径拿到的是
    同一份缩放后的输入。
    """

    def __init__(self, dim, num_heads, attn_dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim {dim} 必须被 num_heads {num_heads} 整除'
                             f'（head_dim 要落在 CANN 支持集 {{16,32,64}}）')
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q = _ScaledLinear(self.dim, self.dim, bias=False)
        self.k = _ScaledLinear(self.dim, self.dim, bias=False)
        self.v = _ScaledLinear(self.dim, self.dim, bias=False)
        self.out = _ScaledLinear(self.dim, self.dim, bias=False)
        self.rope = RoPE2D(self.num_heads, self.head_dim)
        self.attn_dropout = float(attn_dropout)

    def initialize(self, scale=1.0, gain=GAIN_SILU):
        for lin in (self.q, self.k, self.v, self.out):
            lin.initialize(scale=scale, gain=gain)
        self.rope.initialize()
        return self

    def _heads(self, t):
        """``(B,N,C)`` → ``(B,Hh,N,head_dim)``。"""
        b, n, _ = t.shape
        return t.reshape(b, n, self.num_heads, self.head_dim) \
                .permute(0, 2, 1, 3).contiguous()

    def forward(self, t, pos):
        """``t``: ``(B, N, C)`` token 序列；``pos``: ``(B, N, 2)`` 行列坐标。"""
        b, n, c = t.shape
        # 🔴 q **不**预乘 scale。scale 作为参数交给 `_sdpa`，由它按所选后端决定
        #    怎么施加（math 手动乘、SDPA 透传 scale=、flash 手动抵消它的写死值）。
        #    旧写法「预乘 q + scale=None」只在 math 路径（V100/910A）正确；
        #    走 SDPA 路径时 SDPA 会再乘一次 1/sqrt(d) ⇒ 注意力 logits 小 32 倍，
        #    softmax 被压平、注意力趋近均值，实测相对误差 82%。
        q = self._heads(self.q(t))
        k = self._heads(self.k(t))
        v = self._heads(self.v(t))
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        ctx = _sdpa(q, k, v, dropout_p=self.attn_dropout, scale=self.scale)
        return self.out(ctx.permute(0, 2, 1, 3).reshape(b, n, c))


class SwiGLU(nn.Module):
    """SwiGLU 前馈（spec §3.2）：``SiLU(x·W_up) ⊗ (x·W_gate)`` → ``·W_down``。

    隐层 ``F = 1.5C``。三个矩阵全 ``bias=False``。
    """

    def __init__(self, dim, hidden):
        super().__init__()
        self.up = _ScaledLinear(dim, hidden, bias=False)
        self.gate = _ScaledLinear(dim, hidden, bias=False)
        self.down = _ScaledLinear(hidden, dim, bias=False)

    def initialize(self, scale=1.0, gain=GAIN_SILU):
        for lin in (self.up, self.gate, self.down):
            lin.initialize(scale=scale, gain=gain)
        return self

    def forward(self, x):
        return self.down(F.silu(self.up(x)) * self.gate(x))


class TransformerBlock(nn.Module):
    """pre-norm transformer 块（spec §3.2），两个残差各自独立。

        x → RMSNorm → MHSA → +x
        x → RMSNorm → SwiGLU → +x

    内部一律用 ``(B, N, C)`` token 序列，块的最后再转回 ``(B, C, H, W)``。

    ⚠ 归一化用 `backbone.RMSNorm`，它在**最后一维**（即 C）上求 RMS，所以
    token 布局必须是 ``(B,N,C)`` 而不是 ``(B,C,N)`` —— 传反了不会报错，只会让
    归一化跨 token 进行，训练能收敛但形状测试全绿、指标莫名其妙。
    它与 `RMSNormMask`（在空间维 ``(H,W)`` 上求）分工不同，两者不可互换。
    """

    def __init__(self, dim, num_heads, ffn_hidden, attn_dropout=0.0):
        super().__init__()
        self.norm_attn = RMSNorm(dim, eps=1e-6)
        self.attn = MHSA(dim, num_heads, attn_dropout=attn_dropout)
        self.norm_ffn = RMSNorm(dim, eps=1e-6)
        self.ffn = SwiGLU(dim, ffn_hidden)

    def initialize(self, scale=1.0, gain=GAIN_SILU, fixup_scale=None):
        self.attn.initialize(scale=scale, gain=gain)
        self.ffn.initialize(scale=scale, gain=gain)
        # `backbone.RMSNorm` 的 weight 初值恒 1：它是**真**归一化（自己除以
        # 实测 RMS），不需要 fson 那种 fixup 缩放。`fixup_scale` 只是为了与
        # `Nbt2TransformerBlock.initialize` 的调用签名对齐。
        return self

    def forward(self, x, pos):
        b, c, h, w = x.shape
        t = x.reshape(b, c, h * w).permute(0, 2, 1)     # (B,N,C)
        t = t + self.attn(self.norm_attn(t), pos)
        t = t + self.ffn(self.norm_ffn(t))
        return t.permute(0, 2, 1).reshape(b, c, h, w)


class Nbt2TransformerBlock(nn.Module):
    """单个 nbt2 块（spec §3.1），``inner_len`` 个 transformer 内块。

        1. NormAct(C)              ← fson
        2. Conv2d(C→M, 1×1)       [p]
        3. inner_len × TransformerBlock @ M
        4. NormAct(M)              ← fson
        5. Conv2d(M→C, 1×1)       [q]
        6. + 块输入                （外层残差）
    """

    def __init__(self, channels, mid_channels, num_heads, ffn_hidden,
                 inner_len=2, attn_dropout=0.0):
        super().__init__()
        self.channels = int(channels)
        self.mid_channels = int(mid_channels)
        self.inner_len = int(inner_len)
        self.normact = NormAct(self.channels)
        self.conv_p = _ScaledConv2d(self.channels, self.mid_channels, 1,
                                    bias=False)
        self.inner = nn.ModuleList([
            TransformerBlock(self.mid_channels, num_heads, ffn_hidden,
                             attn_dropout=attn_dropout)
            for _ in range(self.inner_len)
        ])
        self.normact_mid = NormAct(self.mid_channels)
        self.conv_q = _ScaledConv2d(self.mid_channels, self.channels, 1,
                                    bias=False)

    def initialize(self, scale=1.0, fixup_scale=None):
        """spec §3.3 的 K 调度。

        ``fixup_scale`` 是本 trunk 块的分量（``1/√(i+1)``），由
        `NbtTfNet.initialize` 逐块传入；块内的 ``normact_mid`` 另按内块数
        取 ``1/√(inner_len+1)``，**不**叠乘 fixup_scale —— 内块只增加内层流的
        方差，不进 trunk 账。
        """
        k = 1.0 if fixup_scale is None else float(fixup_scale)
        self.normact.initialize(scale=1.0, norm_scale=k)
        self.conv_p.initialize(scale=scale)
        for blk in self.inner:
            blk.initialize(fixup_scale=k)
        self.normact_mid.initialize(scale=1.0,
                                   norm_scale=1.0 / math.sqrt(self.inner_len + 1.0))
        self.conv_q.initialize(scale=scale)
        return self

    def forward(self, x, pos):
        h = self.conv_p(self.normact(x))
        for blk in self.inner:
            h = blk(h, pos)
        return x + self.conv_q(self.normact_mid(h))


def _board_pos(b, h, w, device):
    """``(B, N, 2)`` 的行/列坐标，供 `RoPE2D` 用。"""
    rows = torch.arange(h, device=device, dtype=torch.float32).reshape(1, h, 1)
    cols = torch.arange(w, device=device, dtype=torch.float32).reshape(1, 1, w)
    pos = torch.cat((rows.expand(1, h, w), cols.expand(1, h, w)), dim=-1)
    return pos.reshape(1, h * w, 2).expand(b, h * w, 2).contiguous()


# --------------------------------------------------------------------------- #
# 池化
# --------------------------------------------------------------------------- #
def gpool_policy(x):
    """policy 头的 `KataGPool`：``mean``、``mean·((√A−14)/10)``、``max``（spec §4.1）。

    第三个统计量取 ``max``（不是缩放 mean）—— policy 需要「最强候选」那个量级，
    value 头才换成缩放 mean（见 `gpool_value`）。
    """
    b, c, h, w = x.shape
    area = float(h) * float(w)
    mean = x.mean(dim=(2, 3))
    scaled = mean * ((math.sqrt(area) - 14.0) / 10.0)
    mx = x.amax(dim=(2, 3))
    return torch.cat((mean, scaled, mx), dim=1)


def gpool_value(x):
    """value 头的 `KataGPool_value`：三次**缩放 mean**，**无 max**（spec §4.2）。

        mean、mean·((√A−14)/10)、mean·(((√A−14)²/100) − 0.1)

    ⚠ 固定 19×19 下两个缩放系数都是常数（0.5 / 0.15），池化实际只是三个缩放的
    mean。**公式照抄保留**，以便将来 board size 可变时不用重推。
    """
    b, c, h, w = x.shape
    area = float(h) * float(w)
    d = math.sqrt(area) - 14.0
    mean = x.mean(dim=(2, 3))
    return torch.cat((mean, mean * (d / 10.0), mean * ((d * d) / 100.0 - 0.1)),
                     dim=1)


# --------------------------------------------------------------------------- #
# 头
# --------------------------------------------------------------------------- #
class PolicyHead(nn.Module):
    """policy 头，K=2（π 与 π_opp）（spec §4.1）。

        trunk ─┬─ Conv2d(C→P) ─────────────────────────────────→ PP
               └─ Conv2d(C→G) → NormAct → gpool(3G) → Linear(3G→P) ─ + ─→ PP
        PP → NormAct → Conv2d(P→K) → (B,K,H,W)     off-board 掩 −5000
        pooled → Linear → Act → Linear → pass (B,K)
        拼接 → (B, K, 361+1)

    pass 是**真实的第 362 个策略索引**，不是特殊类 —— 空间 361 个落点展平后与
    pass 支路直接 concat，不做「两个分布再归一化」。
    """

    def __init__(self, in_channels, channels=48, gpool_channels=48,
                 num_outputs=2):
        super().__init__()
        self.num_outputs = int(num_outputs)
        self.conv = _ScaledConv2d(in_channels, channels, 1, bias=False)
        self.conv_g = _ScaledConv2d(in_channels, gpool_channels, 1, bias=False)
        self.normact_g = NormAct(gpool_channels)
        self.fuse = _ScaledLinear(3 * gpool_channels, channels, bias=False)
        self.normact = NormAct(channels)
        self.out = _ScaledConv2d(channels, self.num_outputs, 1, bias=True)
        self.pass_fc1 = _ScaledLinear(3 * gpool_channels, channels, bias=True)
        self.pass_fc2 = _ScaledLinear(channels, self.num_outputs, bias=False)

    def initialize(self, scale=1.0, gain=GAIN_SILU):
        self.conv.initialize(scale=scale, gain=gain)
        self.conv_g.initialize(scale=scale, gain=gain)
        self.normact_g.initialize()
        self.fuse.initialize(scale=scale, gain=gain)
        self.normact.initialize()
        self.out.initialize(scale=scale, gain=gain)
        self.pass_fc1.initialize(scale=scale, gain=gain)
        self.pass_fc2.initialize(scale=scale, gain=gain)
        return self

    def forward(self, trunk, board_mask=None):
        b, _, h, w = trunk.shape
        pp = self.conv(trunk)
        pooled = gpool_policy(self.normact_g(self.conv_g(trunk)))
        pp = pp + self.fuse(pooled).reshape(b, -1, 1, 1)
        spatial = self.out(self.normact(pp))            # (B,K,H,W)
        spatial = spatial.reshape(b, self.num_outputs, h * w)
        if board_mask is not None:
            # 允许 (H,W) / (1,H,W) / (B,H,W) / (B,1,H,W)，一律规约到 (B,1,H*W)。
            if board_mask.dim() == 2:
                keep = board_mask.reshape(1, 1, h * w)
            else:
                keep = board_mask.reshape(board_mask.shape[0], 1, h * w)
            keep = keep.to(spatial.dtype).expand(b, 1, h * w)
            spatial = spatial.masked_fill(keep <= 0.5, OFF_BOARD_LOGIT)
        # pass 支路输出 (B,K) → (B,K,1)：pass 是第 362 个策略索引，与展平后的
        # 361 个落点**并列**在同一个分布里，不做两个分布再归一化。
        pass_logit = self.pass_fc2(F.silu(self.pass_fc1(pooled))).unsqueeze(-1)
        return torch.cat((spatial, pass_logit), dim=2)


#: ``scorestdev = 20·SoftPlus(x₁, beta)`` 里的 beta。**spec §4.2 的字面写法是
#: `0.05`，已裁决改为 `1.0`**（2026-10-03）—— 下面保留推导链，因为「为什么不能
#: 随手改这个数」正是这段推导本身，而「spec 写的是 0.05」是推导的一环。
#:
#: 🔴 **裁决依据：softplus(beta) 的量纲标定。**
#: ``F.softplus(x, beta) = log(1+exp(beta·x))/beta``，代入 x=0 得
#: ``softplus(0, beta) = log(2)/beta`` ⇒ **预测初值 = 20·log(2)/beta**：
#:
#: ==========  ==========================  ==========================
#: beta        20·softplus(0,beta) 初值    与 loss #7 的目标（5~20，δ=10）
#: ==========  ==========================  ==========================
#: 0.05        **277.26**（spec 字面值）   ✗ 高出 27 倍 ⇒ 初期梯度被常数偏差支配
#: **1.0**     **13.86**（本常量）        ✓ 与 δ=10 同量级，起点即可用
#: ==========  ==========================  ==========================
#:
#: 取默认 ``beta=1.0``（PyTorch ``F.softplus`` 的默认值）后，这一项从「等效于
#: 没有学习信号」恢复为正常可训练项。
#:
#: ⚠ **实测（`tests/test_katago_v7_budget.py::
#: test_score_stdev_softplus_term_lands_near_huber_delta` 与
#: test_score_stdev_loss_term_is_inside_huber_delta_at_the_ruled_out_beta`）**：
#: seed 0 初始化 + seed 7 输入（B=64）跑真实 forward，
#: ``score_stdev.mean()``：beta=0.05 → **277.2534**，beta=1.0 → **13.8599**；
#: 落到 loss #7 的公式值：beta=0.05 → **272.22**，beta=1.0 → **8.83**
#: （Huber δ=10 ⇒ 落在 δ **以内**，此时是二次段、梯度仍有效）。
#:
#: ⚠ **这是纯常量，不改结构**：头部拓扑、参数量（**5,561,832**）、预算测试全部
#: 不受影响（有测试钉住）。另 ⚠ **段 1 不训 score**（8 项系数逐个 0.0），所以本
#: 改动对段 1 的四个主目标（policy / π_opp / value / futurepos）**逐位无影响**，
#: 它是为**段 2/3**（接上 sidecar、把 score 系权重打开）生效的。
SCORE_STDEV_SOFTPLUS_BETA = 1.0


class ValueHead(nn.Module):
    """value 头（spec §4.2/4.3）+ scoring / futurepos / seki 三个 1×1 小头。

    主路径：``gpool_value(3V=144) → Linear(→W=96) → SiLU → h``，再分两支：
    3 类 outcome 与 3 个标量（scoremean / scorestdev / lead）。

    ⚠ **小头全部接在 ``VV``（池化之前）上**，不是接在 ``h`` 上 —— 它们的输出是
    19×19 平面，空间分辨率不能被池化抹掉。

    ⚠ ownership 输出的是 **pretanh**（线性），不是 ``tanh`` 之后的值。loss #4
    要的是 ``BCE_with_logits(2·pretanh, (1+t)/2)``，而 `BCEWithLogits` 吃的是
    无界 logit；``tanh`` 只在**推理/导出**时施加（`ownership()` 方法）。
    futurepos 同理：loss #11 自己带 ``tanh``，所以头是线性的。
    """

    def __init__(self, in_channels, channels=48, hidden=96, seki_classes=4,
                 futurepos_channels=2):
        super().__init__()
        self.conv = _ScaledConv2d(in_channels, channels, 1, bias=False)
        self.normact = NormAct(channels)
        self.fc = _ScaledLinear(3 * channels, hidden, bias=True)
        self.outcome = _ScaledLinear(hidden, 3, bias=True)
        self.scores = _ScaledLinear(hidden, 3, bias=True)
        self.ownership = _ScaledConv2d(channels, 1, 1, bias=False)
        self.scoring = _ScaledConv2d(channels, 1, 1, bias=False)
        self.futurepos = _ScaledConv2d(channels, futurepos_channels, 1, bias=False)
        self.seki = _ScaledConv2d(channels, seki_classes, 1, bias=False)

    def initialize(self, scale=1.0, gain=GAIN_SILU):
        self.conv.initialize(scale=scale, gain=gain)
        self.normact.initialize()
        self.fc.initialize(scale=scale, gain=gain)
        self.outcome.initialize(scale=scale, gain=gain)
        self.scores.initialize(scale=scale, gain=gain)
        for head in (self.ownership, self.scoring, self.futurepos, self.seki):
            head.initialize(scale=scale, gain=gain)
        return self

    def forward(self, trunk):
        vv = self.normact(self.conv(trunk))
        h = F.silu(self.fc(gpool_value(vv)))
        s = self.scores(h)
        return {
            'outcome_logits': self.outcome(h),
            'score_mean': 20.0 * s[:, 0],
            'score_stdev': 20.0 * F.softplus(s[:, 1],
                                             beta=SCORE_STDEV_SOFTPLUS_BETA),
            'lead': 20.0 * s[:, 2],
            'ownership_pretanh': self.ownership(vv),
            'scoring': self.scoring(vv),
            'futurepos': self.futurepos(vv),
            'seki_logits': self.seki(vv),
            'value_pooled': gpool_value(vv),
        }


class ScorebeliefHead(nn.Module):
    """scorebelief 头（spec §4.4），842 桶的**成分混合**分布。

        pooled(3V)      → Linear(→96)  ┐
        0.05·(i−mid+0.5) → Linear(1→96) ├→ + → SiLU → h (B,842,96)
        parity(i)·global[18] → Linear(1→96) ┘
        h → Linear(→8) → comp (B,842,8)
        pooled → Linear(→8) → mix  (B,8)
        logits = logsumexp_k(mix + comp)

    桶下标自变量（``0.05`` 的分差步长与 ``parity``）在 ``__init__`` 里算好注册成
    buffer，**不进 checkpoint**（由 spec §6.1 的 ``1→96 ×2`` 无 bias 可证）。

    ⚠ ``parity(i)·global[18]`` 那一路是 spec §2.3 ch18 的镜像：那个通道是
    「komi × 棋盘奇偶三角波」，乘进分差桶的奇偶性里，让网络在相邻两桶之间
    感知贴不贴和。它是 V7 相对 V3/V4/V6 唯一的全局通道新增（源码注释互证）。
    """

    def __init__(self, pooled_dim, hidden=96, num_bins=842,
                 num_components=8, parity_channel=18):
        super().__init__()
        self.num_bins = int(num_bins)
        self.mid = self.num_bins // 2
        self.num_components = int(num_components)
        idx = torch.arange(self.num_bins, dtype=torch.float32) - self.mid + 0.5
        self.register_buffer('bin_coord', (0.05 * idx).reshape(1, -1, 1))
        parity = 0.5 - (torch.arange(self.num_bins) - self.mid) % 2
        self.register_buffer('bin_parity', parity.reshape(1, -1, 1))
        self.parity_channel = int(parity_channel)
        self.fc_pooled = _ScaledLinear(pooled_dim, hidden, bias=True)
        self.fc_coord = _ScaledLinear(1, hidden, bias=False)
        self.fc_parity = _ScaledLinear(1, hidden, bias=False)
        self.fc_comp = _ScaledLinear(hidden, self.num_components, bias=True)
        self.fc_mix = _ScaledLinear(pooled_dim, self.num_components, bias=True)

    def initialize(self, scale=1.0, gain=GAIN_SILU):
        for lin in (self.fc_pooled, self.fc_coord, self.fc_parity,
                    self.fc_comp, self.fc_mix):
            lin.initialize(scale=scale, gain=gain)
        return self

    def forward(self, value_pooled, global_features):
        b = value_pooled.shape[0]
        coord = self.bin_coord.expand(b, -1, 1).to(value_pooled.dtype)
        par = (self.bin_parity * global_features[:, self.parity_channel:self.parity_channel + 1]
               .unsqueeze(1)).to(value_pooled.dtype)
        h = F.silu(self.fc_pooled(value_pooled).unsqueeze(1)
                   + self.fc_coord(coord) + self.fc_parity(par))
        comp = self.fc_comp(h)                                   # (B,842,K)
        mix = self.fc_mix(value_pooled).unsqueeze(1)             # (B,1,K)
        return torch.logsumexp(mix + comp, dim=2)                # (B,842)


# --------------------------------------------------------------------------- #
# 整网
# --------------------------------------------------------------------------- #
class NbtTfNet(nn.Module):
    """V7 NBT+Transformer 整网（spec §3 + §4）。

    输入 ``(B, 22, H, W)`` 空间 + ``(B, 19)`` 全局，输出 `forward` 里那组张量。
    22 与 19 都由 ``fillRowV7`` 决定，**不从棋局重新推算**。
    """

    def __init__(self, cfg=None, use_checkpoint=None, attn_dropout=None):
        super().__init__()
        c = dict(NBT_TF_CFG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        self.use_checkpoint = (c['use_checkpoint'] if use_checkpoint is None
                               else bool(use_checkpoint))
        self.attn_dropout = (c['attn_dropout'] if attn_dropout is None
                             else float(attn_dropout))
        C = c['trunk_channels']
        M = c['nbt_mid']
        self.stem = _ScaledConv2d(c['in_channels'], C, 3, padding=1, bias=False)
        self.global_fc = _ScaledLinear(c['global_channels'], C, bias=False)
        self.blocks = nn.ModuleList([
            Nbt2TransformerBlock(C, M, c['num_heads'], c['ffn_hidden'],
                                 inner_len=c['num_inner_blocks'],
                                 attn_dropout=self.attn_dropout)
            for _ in range(c['num_blocks'])
        ])
        self.norm_trunkfinal = RMSNormMask(C)
        self.policy_head = PolicyHead(C, channels=c['policy_channels'],
                                      gpool_channels=c['gpool_channels'],
                                      num_outputs=c['policy_outputs'])
        self.value_head = ValueHead(C, channels=c['value_channels'],
                                    hidden=c['value_hidden'],
                                    seki_classes=c['seki_classes'],
                                    futurepos_channels=c['futurepos_channels'])
        # 桶数 = 2*(361 + EXTRA_SCORE_DISTR_RADIUS)，spec §4.4。
        bins = 2 * (c['board_size'] ** 2 + c['extra_score_distr_radius'])
        self.scorebelief_head = ScorebeliefHead(
            3 * c['value_channels'], hidden=c['value_hidden'], num_bins=bins,
            num_components=c['score_distr_components'])

    # ---- 初始化（spec §3.3 / §3.4）----
    def initialize(self):
        """fson 的 K 调度 + 截断正态权重（spec §3.3 / §3.4）。

        K 调度：trunk 第 ``i`` 块 ``1/√(i+1)``；trunk 末端 ``1/√(B+1)``。
        块内的 ``normact_mid`` 按内块数取 ``1/√(inner_len+1)``（在
        `Nbt2TransformerBlock.initialize` 内）。

        gain：``SiLU`` 取 √2.0（官方注释：理论值 √2.8108，为兼容保留 √2.0）。
        stem conv 的 scale = 0.8、global linear 的 scale = 0.6，其余 scale = 1.0。
        """
        self.stem.initialize(scale=SCALE_INIT_CONV, gain=GAIN_SILU)
        self.global_fc.initialize(scale=SCALE_GLOBAL_LINEAR, gain=GAIN_SILU)
        for i, blk in enumerate(self.blocks):
            blk.initialize(scale=1.0, fixup_scale=1.0 / math.sqrt(i + 1.0))
        self.norm_trunkfinal.initialize(
            scale=1.0,
            norm_scale=1.0 / math.sqrt(len(self.blocks) + 1.0))
        self.policy_head.initialize()
        self.value_head.initialize()
        self.scorebelief_head.initialize()
        return self

    def trunk(self, spatial, global_features):
        """stem + 11 个 nbt2 块 + trunk 末端归一化（spec §3.1）。"""
        x = self.stem(spatial)
        g = self.global_fc(global_features).reshape(-1, self.cfg['trunk_channels'],
                                                    1, 1)
        x = x + g
        pos = _board_pos(x.shape[0], x.shape[2], x.shape[3], x.device)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, pos,
                                                      use_reentrant=False)
            else:
                x = blk(x, pos)
        return F.silu(self.norm_trunkfinal(x))

    def forward(self, spatial, global_features, board_mask=None):
        t = self.trunk(spatial, global_features)
        out = {'policy_logits': self.policy_head(t, board_mask=board_mask)}
        vout = self.value_head(t)
        out.update({k: v for k, v in vout.items() if k != 'value_pooled'})
        out['scorebelief_logits'] = self.scorebelief_head(vout['value_pooled'],
                                                          global_features)
        return out

    def ownership(self, out):
        """推理用：ownership 的 ``tanh(pretanh)``，落在 ``[-1,+1]``（spec §4.3）。"""
        return torch.tanh(out['ownership_pretanh'])


def build_katago_v7_net(*, board_size=19, use_checkpoint=None,
                        attn_dropout=None, cfg=None):
    """按 `NBT_TF_CFG` 建 V7 网并跑一次 `initialize()`。

    **返回已初始化的模型**（不是随机权重）—— fson 的 γ 必须带 K，否则残差流
    方差随深度线性增长，深层一开始就饱和，而那种「模型能跑但 loss 离谱」的
    现象很难回溯到这里。`tests/test_katago_v7_budget.py` 也依赖这一点。
    """
    c = dict(cfg or {})
    c.setdefault('board_size', int(board_size))
    net = NbtTfNet(cfg=c, use_checkpoint=use_checkpoint,
                   attn_dropout=attn_dropout)
    return net.initialize()
