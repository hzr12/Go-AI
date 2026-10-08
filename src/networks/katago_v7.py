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
``B``        11     nbt 块数。每个块 2 个内块，纯 nbt（GAU 已关，见 `gau_positions=None`）
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

 **RoPE 的形状是 spec 唯一未逐字确定的一处**：spec §3.2 只给出参数形状
``(H, 16, 2)``，没有给「2」的语义。本实现取**二维频率**（``[...,0]`` 乘行坐标、
``[...,1]`` 乘列坐标，角度 = 二者加权和），它是 2D 旋转位置编码的直接推广，
且**参数量与 spec 完全一致**。若日后核对 ``model_pytorch.py`` 发现官方是别的
切法，只需改 ``RoPE2D.forward`` 一处 —— 参数形状不变、预算不变。
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from src.networks.backbone import (
    GC_LEGACY,
    GC_RES,
    GC_TRANSFORMER,
    GRAD_CHECKPOINT_DEFAULTS,
    RMSNorm,
    GradCheckpointMixin,
    _sdpa,
)

# ---- NPU 融合 SwiGLU（可选加速，带运行时回退）----------------------------
# torch_npu 缺失或 npu_swiglu 不可用/自检不过时，SwiGLU 退化为标准实现
# （F.silu(up(x)) * gate(x) 再 down），数值行为完全不变。
try:
    import torch_npu  # 仅在 NPU 环境可导入
    _HAS_NPU_SWIGLU = hasattr(torch_npu, 'npu_swiglu')
except Exception:
    torch_npu = None
    _HAS_NPU_SWIGLU = False


def set_npu_swiglu(enabled: bool) -> None:
    """NPU 融合 SwiGLU 开关（对应 CLI ``--npu-swiglu``，2026-10-06 从环境变量搬来）。

    `False` = 强制走标准路径（`F.silu(up(x)) * gate(x)` 再 down），数值不变、
    只慢不坏。是「融合路径是不是坏的」这个问题的总闸（2026-10-06 融合首次真跑后
    真机出现过前向 NaN，需要能单独二分它）。
    """
    global _HAS_NPU_SWIGLU
    _HAS_NPU_SWIGLU = bool(enabled) and torch_npu is not None

#: 融合可用性（运行时会被自检/调用异常降级为 False）。模块级 ⇒ **失败只告警一次**
#: —— 原实现每个 SwiGLU 每 step 都 warning 一条（V7 主干 6 处 × 每 step），真机日志
#: 已被 `[SwiGLU] NPU 融合失败` 刷爆（2026-10-06）。
_npu_swiglu_ok = _HAS_NPU_SWIGLU
_npu_swiglu_checked = False


def _npu_swiglu(y, dim=-1):
    """torch_npu.npu_swiglu 薄封装。

    CANN 签名是 ``npu_swiglu(Tensor input, int dim=-1)``：**输入是已沿 dim 拼好
    的 (…, 2H) 张量**，内部自己分半，返回 ``silu(a) * b``（a=前半过 SiLU，b=后半），
    与 Megatron-core / MindSpeed 的 chunk 口径一致。它**不吃**两张权重矩阵 ——
    旧封装 ``npu_swiglu(x, w1, w2)`` 正是因此每次都抛
    ``expected at most 2 argument(s) but received 3``，整体退回标准路径。
    """
    return torch_npu.npu_swiglu(y, dim)


def _swiglu_fusion_selfcheck(dev):
    """分半顺序自检：钉住「silu(前半) * 后半」。

    若某个 CANN 版本的分半语义相反（silu 在后半），融合会给出**错值而非异常** ——
    forward 的 try/except 抓不住，训练会静默学错。故在首个融合调用处用随机小张量
    对拍一次；不匹配则调用方永久关闭融合（退标准路径）。
    """
    y = torch.randn(16, 64, device=dev, dtype=torch.float16)
    a, b = y[:, :32], y[:, 32:]
    fused = torch_npu.npu_swiglu(y, -1).float()
    ref = (F.silu(a) * b).float()
    return bool(torch.allclose(fused, ref, atol=1e-2, rtol=1e-2))


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
    'attn_impl': 'nbt',         # 内块实现。'nbt' = 现有 MHSA+SwiGLU；
                                # 'gau' = GAU 改版（覆盖全部内块）。
    'gau_positions': None,      # 关掉 GAU，退回纯 nbt 参考结构 —— 可保真导出到 KataGo
                                # （`.bin` 只有 `transformer_attention_block`，无 GAU 层）。
                                # 重开 GAU 时改回下标列表（如 [1]）并设置下方 first/last。
    'gau_first_nbt': 0,         # 前 N 块纯 nbt（无 GAU）。GAU 关闭时为 0。
    'gau_last_nbt': 0,          # 末 N 块纯 nbt（无 GAU）。GAU 关闭时为 0。
    'gau_hidden': 384,          # GAU 的 U/V 宽度 e（GAU 已关，此值暂不使用；
                                # 如需重开 GAU 取 384，与 SwiGLU 的 FFN 宽度一致）。
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
    'params_block_nbt': 493_056,       # 每个 nbt 块：[nbt, nbt]（GAU 关闭，全 11 块都是它）
    'params_block_gau': 525_953,       # GAU 块成本（GAU 关闭时为未使用；重开 GAU 时
                                       # [nbt,gau] 块 = 此值，含偏置 b）
    'params_blocks_total': 5_423_616,  # 11×493_056（全 nbt）
    'params_trunkfinal': 512,
    'params_policy_head': 38_832,
    'params_value_head': 27_177,       # value 头本体（不含 4 个小头）
    'params_small_heads': 384,         # 4 个小头（ownership/scoring/futurepos/seki）
    'params_scorebelief_head': 16_048,
    'params_total': 5_562_121,        # 5,562,121（全 11 块纯 nbt；实测）
}

#: off-board 位置的 logits 惩罚（spec §4.1）。常量、不参与学习 —— 因此
#: **不占参数**，spec §6.1 的 policy 头 38,834 正是按「零参数 BiasMask」记的。
OFF_BOARD_LOGIT = -5000.0

#: ``SiLU`` 的 init gain。官方注释：理论值 √2.8108，为兼容保留 √2.0。
GAIN_SILU = math.sqrt(2.0)

#: ``relu²`` 的 init gain。与 `GAIN_SILU` 同取 √2.0：代码库对这些门控激活统一用
#: √2.0（不精确匹配各自方差；relu² 的保方差理论 gain 为 √0.8≈0.894），保持与既有
#: 初始化口径一致即可，避免 GAU 初值偏离其它块太多。
GAIN_RELU2 = math.sqrt(2.0)

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

     **γ 的初值是 ``norm_scale / scale``，不是 1。** ``norm_scale`` 就是
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

         ``freq[..., 0]`` 乘的是**列**，``freq[..., 1]`` 乘的是**行** ——
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

         **这个错误在任何正方盘面上都测不出来**：19×19 的行列数相同，行列互换
        只是一个有效对称，policy 输出仍然「看着合理」。只有拿官方权重逐位对拍
        才能发现（我们是这么发现的）。
        """
        b, _, n, d = x.shape
        if d != self.head_dim:
            raise ValueError(f'head_dim 不符：传入 {d}，本层 {self.head_dim}')
        pos = pos.to(device=x.device, dtype=x.dtype).reshape(b, n, 2)
        row = pos[:, None, :, 0:1]                     # y
        col = pos[:, None, :, 1:2]                     # x
        f = self.freq.to(x.dtype)                      # (Hh, pairs, 2)
        ang = (f[None, :, :, 0].unsqueeze(2) * col     # freqX * x
               + f[None, :, :, 1].unsqueeze(2) * row)  # freqY * y
        return self._rotate(x, ang.cos(), ang.sin())


def _migrate_ema_shadow_qkv(shadow):
    """把旧 ckpt 的 EMA shadow（``...attn.q/k/v.weight``）并成 ``qkv.weight``。

    MHSA 现在内部只存 ``qkv``（见 `MHSA` 类文档），EMA 以
    ``named_parameters()`` 为键 ⇒ 新 shadow 用 ``qkv``。旧 checkpoint 存的是
    三个独立键，直接换上去会让 `EMA.update()` 抛 KeyError。三者按 q,k,v 顺序
    沿 dim0 拼接即可（与 `qkv_to_qkv` 的逆运算，**无损**）。

    返回 ``(新 shadow, 迁移了几处)``。不是 MHSA 的键原样透传。
    """
    out, moved = {}, 0
    i = 0
    keys = list(shadow.keys())
    while i < len(keys):
        k = keys[i]
        if k.endswith('.q.weight'):
            # k[:-len('.q.weight')] 会把 q 前的那个点一起吃掉，所以下面补回。
            base = k[:-len('.q.weight')]
            trip = [shadow.get(base + '.' + n + '.weight') for n in ('q', 'k', 'v')]
            if all(t is not None for t in trip):
                out[base + '.qkv.weight'] = torch.cat(
                    [t.detach().clone() for t in trip], dim=0)
                i += 3
                moved += 1
                continue
        out[k] = shadow[k]
        i += 1
    return out, moved


class MHSA(nn.Module):
    """多头自注意力，``head_dim = nbt_mid / num_heads``（spec §3.2）。

    ``q`` 在调用 ``_sdpa`` **之前**已乘 ``1/√head_dim``（`_sdpa`` 的契约，见
    `backbone.py:584`），这样 flash / mem-efficient 后端与 math 路径拿到的是
    同一份缩放后的输入。

    **q/k/v 合成（训练侧优化）**：内部只存一个 ``qkv`` 权重（一次 GEMM），但
    ``state_dict()`` 仍以 ``q/k/v`` 三个键对外暴露。因此：
      - 旧 checkpoint（``attn.q/k/v.weight``）可直接 ``load_state_dict``；
      - ``katago_export.py`` 读的三个键不变，导出的 .bin.gz 与合成前**逐字节相同**
        ⇒ KataGo 引擎加载零影响；
      - EMA（``named_parameters`` 键）只在训练期新起的 shadow 里是 ``qkv``，
        见 `scripts/train_sft.py` resume 分支的迁移。
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
        # q/k/v 三个 Linear(dim,dim) 合成**一个** Linear(dim,3*dim)：22 处
        # (11 block × 2 inner) 每处省 2 次 ACL 启动。V7 dim=128 时单个 GEMM 是
        # 128×128×N，M 很小、完全被启动开销支配，合成 384×128 明显更划算。
        # ⚠ 合成**只发生在训练时的前向组织**：权重张量本身没变（见
        # `_state_dict_qkv_to_qkv` / `qkv_to_qkv`），所以导出的 .bin.gz 与
        # 合成前**逐字节相同**，KataGo 引擎加载零影响（实测：两组 ckpt 导出的
        # SHA256 相同、引擎都 GTP ready）。
        self.qkv = _ScaledLinear(self.dim, 3 * self.dim, bias=False)
        self.out = _ScaledLinear(self.dim, self.dim, bias=False)
        self.rope = RoPE2D(self.num_heads, self.head_dim)
        self.attn_dropout = float(attn_dropout)

    def initialize(self, scale=1.0, gain=GAIN_SILU):
        # 逐段初始化（而非对 (3dim,dim) 整块 _trunc_normal_）：三段的 std 只跟
        # in_features 有关、与 out_features 无关，所以整块初始化与分段初始化
        # **同分布**。分段写是为了让 q/k/v 三段的 RNG 抽样顺序与旧的三个独立
        # Linear 完全一致 ⇒ 同 seed 下初值与合成前逐位相同，便于回归比对。
        std = scale * gain / math.sqrt(self.dim)
        with torch.no_grad():
            for i, name in enumerate(('q', 'k', 'v')):
                _trunc_normal_(self.qkv.weight[i * self.dim:(i + 1) * self.dim], std)
        self.out.initialize(scale=scale, gain=gain)
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
        # q **不**预乘 scale。scale 作为参数交给 `_sdpa`，由它按所选后端决定
        #    怎么施加（math 手动乘、SDPA 透传 scale=、flash 手动抵消它的写死值）。
        #    旧写法「预乘 q + scale=None」只在 math 路径（V100/910A）正确；
        #    走 SDPA 路径时 SDPA 会再乘一次 1/sqrt(d) ⇒ 注意力 logits 小 32 倍，
        #    softmax 被压平、注意力趋近均值，实测相对误差 82%。
        # 一次 GEMM 出 q/k/v，按行切成三段（chunk 只是 view，不额外拷贝）。
        qkv = self.qkv(t)
        q = self._heads(qkv[..., :c])
        k = self._heads(qkv[..., c:2 * c])
        v = self._heads(qkv[..., 2 * c:])
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        ctx = _sdpa(q, k, v, dropout_p=self.attn_dropout, scale=self.scale)
        return self.out(ctx.permute(0, 2, 1, 3).reshape(b, n, c))

    # ---- state_dict 兼容：内部 qkv，对外 q/k/v ---------------------------- #
    def _split_qkv(self, fused):
        """``(3*dim, dim)`` → 三个 ``(dim, dim)``（view，不拷贝）。"""
        d = self.dim
        return fused[:d], fused[d:2 * d], fused[2 * d:3 * d]

    def state_dict(self, *args, **kwargs):
        """对外吐字典时把 ``qkv.weight`` 展开成三个 ``q/k/v.weight``。

        ⚠ 两个坑，都是真机 Linux / torch 2.1.0 上踩出来的：

        1. **不能用** ``register_state_dict_post_hook``：那是 **torch ≥2.2**
           才有的 API，在 2.1.0 上 AttributeError，整个模型在
           ``MHSA.__init__`` 就建不起来（本地 torch 2.12 有这个 API，
           所以本地测试全绿也照样漏）。
        2. **也不能只覆写** ``_save_to_state_dict``：``nn.Module.state_dict``
           的调用顺序是「先父模块 ``_save_to_state_dict``、**再**递归子模块、
           最后跑 post-hook」。子模块的 ``qkv.weight`` 是在父模块那步**之后**
           才写进去的 ⇒ 那一刻去 ``pop`` 找不到键，展开静默失效。

        ⇒ 只能在 ``state_dict()`` 这一层整体收口：递归结束后
        ``prefix+'qkv.weight'`` 已经存在，这时再展开。``state_dict`` 与
        ``_load_from_state_dict`` 从 torch 1.x 到 2.x 签名都稳定。
        """
        # 本层既可能是最外层调用（无 destination），也可能是父模块带着
        # destination= 递归下来。两种都要处理：本模块的 prefix 在两种情况下
        # 分别来自 kwargs['prefix'] 或位置参数 args[1]。
        destination = super().state_dict(*args, **kwargs)
        prefix = kwargs.get('prefix')
        if prefix is None:
            prefix = args[1] if len(args) > 1 else ''
        key = prefix + 'qkv.weight'
        if key not in destination:
            return destination
        fused = destination.pop(key)
        for name, part in zip(('q', 'k', 'v'), self._split_qkv(fused)):
            # `.clone()` 必要：三个切片共享同一 storage，不拷就会别名同一块
            # 内存，调用方 load_state_dict 进去三者互相污染。
            destination[prefix + name + '.weight'] = part.detach().clone()
        return destination

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """反向：见到旧的 ``q/k/v`` 三键就拼回 ``qkv`` 再交父类加载。

        这样**旧 checkpoint 无需迁移脚本**即可加载（`--resume` / `--model`）。
        """
        d = self.dim
        parts = [state_dict.get(prefix + n + '.weight') for n in ('q', 'k', 'v')]
        qkv_key = prefix + 'qkv.weight'
        if qkv_key not in state_dict and all(p is not None for p in parts):
            state_dict[qkv_key] = torch.cat([p.detach() for p in parts], dim=0)
            for n in ('q', 'k', 'v'):
                state_dict.pop(prefix + n + '.weight', None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


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
        global _npu_swiglu_ok, _npu_swiglu_checked
        # NPU 融合：npu_swiglu 把「SiLU 激活 + 门控相乘」合成一个 kernel（GEMM 仍是
        # up/gate 两次，融合的是逐元素部分），减少 kernel launch / 显存往返。
        # 输入须沿末维拼成 (…, 2H)：silu(up(x)) * gate(x) ⟺ npu_swiglu(cat(up, gate))
        # （前半过 SiLU）。bias=False 的 _ScaledLinear 的缩放已在 initialize 时
        # bake 进 weight，无运行时额外缩放，数学上与下方标准路径等价。
        if x.device.type == 'npu' and _npu_swiglu_ok:
            try:
                if not _npu_swiglu_checked:
                    _npu_swiglu_checked = True
                    if not _swiglu_fusion_selfcheck(x.device):
                        _npu_swiglu_ok = False
                        import logging
                        logging.getLogger(__name__).warning(
                            "[SwiGLU] npu_swiglu 分半顺序自检不匹配"
                            "（非 silu(前半)*后半），本进程内永久退回标准路径")
                        return self.down(F.silu(self.up(x)) * self.gate(x))
                return self.down(
                    _npu_swiglu(torch.cat((self.up(x), self.gate(x)), dim=-1)))
            except Exception as _e:
                # 融合失败（签名/设备异常等）不要静默吞掉，但也**不要每 step 都重试
                # 刷告警**：置 False 永久退回标准路径，只告警一次。
                _npu_swiglu_ok = False
                import logging
                logging.getLogger(__name__).warning(
                    "[SwiGLU] NPU 融合失败，本进程内永久退回标准路径: %s", _e)
        return self.down(F.silu(self.up(x)) * self.gate(x))


class TransformerBlock(nn.Module):
    """pre-norm transformer 块（spec §3.2），两个残差各自独立。

        x → RMSNorm → MHSA → +x
        x → RMSNorm → SwiGLU → +x

    内部一律用 ``(B, N, C)`` token 序列，块的最后再转回 ``(B, C, H, W)``。

     归一化用 `backbone.RMSNorm`，它在**最后一维**（即 C）上求 RMS，所以
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


def _resolve_inner_kinds(attn_impl, gau_positions, inner_len):
    """把内块类型解析成 ``['nbt' | 'gau', ...]``（长度 == ``inner_len``）。

    - ``attn_impl``：所有内块的**默认**实现，``'nbt'``（现有 MHSA+SwiGLU）或
      ``'gau'``（GAU 改版）。
    - ``gau_positions``：要**强制**用 GAU 的内块下标集合（覆盖 ``attn_impl``）。
      ``None`` ⇒ 不强制，全用 ``attn_impl``。下标越界即报错。

    默认（``attn_impl='nbt'`` 且 ``gau_positions=None``）⇒ 全 ``'nbt'``，与改造前
    **逐位相同**：已有 checkpoint 与参数预算锚继续成立。

    「每层分别 1 个 GAU + 1 个 nbt」＝ ``attn_impl='nbt', gau_positions=[0]``
    （``inner_len=2`` ⇒ ``['gau', 'nbt']``）。
    """
    impl = str(attn_impl).strip().lower()
    if impl not in ('nbt', 'gau'):
        raise ValueError(f'attn_impl 只能是 nbt/gau，收到 {attn_impl!r}')
    positions = set()
    if gau_positions is not None:
        positions = {int(p) for p in gau_positions}
    invalid = [p for p in positions if not (0 <= p < int(inner_len))]
    if invalid:
        raise ValueError(f'gau_positions 下标越界（需 0..{int(inner_len) - 1}），'
                         f'收到 {invalid}')
    return ['gau' if i in positions else impl
            for i in range(int(inner_len))]


def _relu2(x):
    """GAU 的门控激活：``relu(x)²``（GAU 论文的 ReLU² 变体，非负、零中心梯度）。

    比 SiLU 更稀疏：``x≤0`` 直接归零（整段梯度为零），且输出非负，门控时不会翻转
    符号。放在 `GatedAttentionUnit` 的 U（GLU 的门）与 V（注意力 value）两处。
    """
    return F.relu(x).square()


#: GAU 门控输出（``o`` 投影之前）的幅值兜底。``relu²`` 门控非负、只下界有界，
#: 叠加 11 层后极端 step 仍可能把门控值推大；这里硬性夹到 ±C，避免 GAU 输出
#: 无限制放大（尤其 fp16 推理对大值敏感）。仅作安全网，正常训练几乎不触顶。
_GAU_GATED_CLAMP = 64.0

#: GAU 手写注意力的 query 分块长度（0/负 = 关闭）。沿用 `_sdpa_math` 的口径：
#: 按 query 切块令峰值显存 ∝ chunk 而非 ∝ N（N=361 下 64 ⇒ 6 块）。
_GAU_Q_CHUNK = 64


class GatedAttentionUnit(nn.Module):
    """GAU（Gated Attention Unit）核心：把注意力与门控前馈融合成**一个**单元。

    论文：Hua et al., *Transformer Quality in Linear Time*（2022）§3.2。

        U = SiLU(x·W_u);   V = SiLU(x·W_v)          # 各 (B, N, e)
        Q = RoPE2D(x·W_q); K = RoPE2D(x·W_k)        # 各 (B, Hg, N, hd)
        A = relu²(QKᵀ·s + b)                         # 多头注意力（无 softmax）
        V̂ = V ⊙ (A @ V_heads)                       # ① 注意力对 V 的门控
        O = (U ⊙ V̂) · W_o                           # ② GLU 门控

    与 ``MHSA + SwiGLU`` 两条子层的差别
    ----------------------------------
    - V **不再**单独投影：注意力直接拿 GLU 的 value 分支当 value；
    - 于是「注意力 + 门控前馈」合成**一个**残差子层，而不是两个（论文实测一个
      GAU ≈ 两个 transformer 子层的质量，参数更省）。

    形状约束（重要）
    --------------
    ``e``（U/V 的宽度）必须能被 ``head_dim`` 整除：V 要切成 ``Hg = e/hd`` 份当
    注意力的 value 头。**这样 value 的 head_dim 与 Q/K 相同** —— 手写注意力
    （``relu²(QKᵀ·s+b)``）要求 q/k/v 的 head_dim 一致，CANN 同理。
    ``head_dim`` 由调用方给的 ``num_heads`` 推出
    （``dim // num_heads``），锁在 CANN 支持集 {16,32,64}。
    """

    def __init__(self, dim, num_heads, hidden=None, attn_dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim {dim} 必须被 num_heads {num_heads} 整除'
                             f'（head_dim 要落在 CANN 支持集 {{16,32,64}}）')
        self.dim = int(dim)
        self.head_dim = self.dim // int(num_heads)
        # U/V 的宽度。默认沿用 `ffn_hidden`，让 GAU 的 FFN 宽度与 `SwiGLU` 一致。
        self.e = int(hidden) if hidden else self.dim
        if self.e % self.head_dim != 0:
            raise ValueError(
                f'GAU 的 hidden {self.e} 必须被 head_dim {self.head_dim} 整除'
                f'（V 要切成整数个 value 头；可改成 '
                f'{self.head_dim * max(1, round(self.e / self.head_dim))}）')
        self.num_heads = self.e // self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.u = _ScaledLinear(self.dim, self.e, bias=False)
        self.v = _ScaledLinear(self.dim, self.e, bias=False)
        self.q = _ScaledLinear(self.dim, self.e, bias=False)
        self.k = _ScaledLinear(self.dim, self.e, bias=False)
        self.o = _ScaledLinear(self.e, self.dim, bias=False)
        self.rope = RoPE2D(self.num_heads, self.head_dim)
        self.attn_dropout = float(attn_dropout)
        # 注意力的可学习标量偏置 b：加到每个 score 元素上（位置无关 ⇒ 平移等变保留）。
        # 配合 relu²，b 控制「多少对 (q,k) 关系能透过门」——b 越大越多项存活。
        # 初值 0（= 退化为纯 relu²(QKᵀ·s)），训练自行学。
        self.b = nn.Parameter(torch.zeros(()))

    def initialize(self, scale=1.0, gain=GAIN_RELU2):
        for lin in (self.u, self.v, self.q, self.k, self.o):
            lin.initialize(scale=scale, gain=gain)
        self.rope.initialize()
        return self

    def _heads(self, t):
        """``(B,N,e)`` → ``(B,Hg,N,hd)``。"""
        b, n, _ = t.shape
        return t.reshape(b, n, self.num_heads, self.head_dim) \
                .permute(0, 2, 1, 3).contiguous()

    def forward(self, t, pos):
        """``t``: ``(B, N, C)`` token 序列；``pos``: ``(B, N, 2)`` 行列坐标。

        注意力用 **relu²(QKᵀ·s + b)** 取代 softmax：``s = 1/√head_dim``（√ 归一），
        ``b`` 为可学习标量偏置；score 经 relu² 后按 **key 维行 L1 归一**（"除以序列
        长度"），保证序列长度不变性、并把 ``A@V`` 压成 ``V`` 的凸组合，避免 11 层
        叠乘放大 / fp16 溢出。门控激活 ``relu²``（见 `_relu2`）。
        """
        b, n, _ = t.shape
        u = _relu2(self.u(t))                        # (B,N,e)  GLU 的门
        v = _relu2(self.v(t))                        # (B,N,e)  兼作注意力的 value
        q = self.rope(self._heads(self.q(t)), pos)   # (B,Hg,N,hd)
        k = self.rope(self._heads(self.k(t)), pos)   # (B,Hg,N,hd)
        # value **不**再投影：直接切 GLU 的 V，其 head_dim 与 q/k 相同（见类文档）。
        vh = v.reshape(b, n, self.num_heads, self.head_dim) \
              .permute(0, 2, 1, 3).contiguous()
        ctx = self._attn(q, k, vh)                    # (B,Hg,N,hd)
        ctx = ctx.permute(0, 2, 1, 3).reshape(b, n, self.e)
        # ① 注意力输出门控 V；② U 门控（GLU）。两层门控都发生在 e 维上。
        # clamp 兜底：防止门控值无限制放大（见 `_GAU_GATED_CLAMP`）。
        gated = (u * (v * ctx)).clamp(-_GAU_GATED_CLAMP, _GAU_GATED_CLAMP)
        return self.o(gated)

    def _attn(self, q, k, v):
        """``relu²(QKᵀ·s + b)`` 注意力，按 query 分块约束显存（沿用 `_sdpa_math` 口径）。

        q,k,v: ``(B, Hg, N, hd)`` → 返回 ``(B, Hg, N, hd)``。
        """
        kt = k.transpose(-2, -1)
        step = _GAU_Q_CHUNK
        nq = q.shape[-2]
        if step and nq > step:
            outs = []
            for i in range(0, nq, step):
                qc = q[..., i:i + step, :]
                a = _relu2((qc @ kt) * self.scale + self.b)      # (B,Hg,chunk,N)
                a = a / (a.sum(-1, keepdim=True) + 1e-6)         # 行 L1 归一
                if self.attn_dropout > 0.0:
                    a = F.dropout(a, p=self.attn_dropout,
                                 training=self.training)
                outs.append(a @ v)
            return torch.cat(outs, dim=-2)
        a = _relu2((q @ kt) * self.scale + self.b)
        a = a / (a.sum(-1, keepdim=True) + 1e-6)
        if self.attn_dropout > 0.0:
            a = F.dropout(a, p=self.attn_dropout, training=self.training)
        return a @ v


class GatedAttentionBlock(nn.Module):
    """GAU 改版的 transformer 内块：``x + GAU(RMSNorm(x))``。

    与 `TransformerBlock` 的接口**完全对齐**（``forward(x, pos)`` 4D→4D、
    ``initialize(scale, gain, fixup_scale)``），所以两种块能混进
    `Nbt2TransformerBlock.inner` 同一个 `ModuleList` —— 这就是「每层 1 个 GAU +
    1 个 nbt」的挂载方式（``attn_impl='nbt', gau_positions=[0]``）。

    这个块**包含**（而非单纯调用外部）：
      - `RoPE2D`：q/k 的旋转位置编码（在 `GatedAttentionUnit` 里）；
      - **手写 `relu²(QKᵀ·s + b)` 注意力**：自己实现、按 query 分块（`_GAU_Q_CHUNK`）
        约束显存，长序列不爆炸；不含 softmax（见 `GatedAttentionUnit`）；
      - **融合 FFN**：GAU 把「注意力 + 门控前馈」融成**一个**残差子层（而非
        `TransformerBlock` 的 MHSA + SwiGLU 两条），容量相当、参数更省；
      - **自己的 `initialize`**：内部权重按 `GAIN_RELU2` 截断正态、RoPE 频率按对数
        均匀初始化，与 `TransformerBlock.initialize` 签名对齐但不依赖其 fixup。

    门控激活用 `relu²`（见 `GatedAttentionUnit`）；Q/K 走与 `MHSA` 同一套 `RoPE2D`
    ⇒ 平移等变性一致。
    """

    def __init__(self, dim, num_heads, ffn_hidden=None, attn_dropout=0.0,
                 hidden=None):
        super().__init__()
        self.norm = RMSNorm(dim, eps=1e-6)
        # `ffn_hidden` 只为与 `TransformerBlock` 的构造签名对齐（工厂按同一组
        # 参数造两种块）；GAU 没有独立的 FFN 隐藏维，容量由 `hidden`（= e）决定，
        # 默认沿用 `ffn_hidden`，好让两种块的 FFN 宽度一致。
        self.gau = GatedAttentionUnit(dim, num_heads,
                                      hidden=hidden or ffn_hidden,
                                      attn_dropout=attn_dropout)

    def initialize(self, scale=1.0, gain=GAIN_RELU2, fixup_scale=None):
        # GAU 用 relu² 门控 ⇒ 内部权重一律按 `GAIN_RELU2` 初始化，忽略传入的 gain。
        self.gau.initialize(scale=scale, gain=GAIN_RELU2)
        # `backbone.RMSNorm` 的 weight 初值恒 1（它是**真**归一化，自己除以实测
        # RMS），不需要 fson 那种 fixup 缩放。`fixup_scale` 只为与
        # `TransformerBlock.initialize` 的签名对齐。
        return self

    def forward(self, x, pos):
        b, c, h, w = x.shape
        t = x.reshape(b, c, h * w).permute(0, 2, 1)      # (B,N,C)
        t = t + self.gau(self.norm(t), pos)
        return t.permute(0, 2, 1).reshape(b, c, h, w)


class Nbt2TransformerBlock(nn.Module):
    """单个 nbt2 块（spec §3.1），``inner_len`` 个 transformer 内块（可混 GAU）。

        1. NormAct(C)              ← fson
        2. Conv2d(C→M, 1×1)       [p]
        3. inner_len × TransformerBlock @ M
        4. NormAct(M)              ← fson
        5. Conv2d(M→C, 1×1)       [q]
        6. + 块输入                （外层残差）
    """

    def __init__(self, channels, mid_channels, num_heads, ffn_hidden,
                 inner_len=2, attn_dropout=0.0, attn_impl='nbt',
                 gau_positions=None, gau_hidden=None):
        super().__init__()
        self.channels = int(channels)
        self.mid_channels = int(mid_channels)
        self.normact = NormAct(self.channels)
        self.conv_p = _ScaledConv2d(self.channels, self.mid_channels, 1,
                                    bias=False)
        # 默认（attn_impl='nbt' 且 gau_positions=None）⇒ 全 'nbt'，与改造前逐位
        # 相同：不传这些参数的老代码 / 老 checkpoint 不受影响。
        self.inner_kinds = _resolve_inner_kinds(attn_impl, gau_positions,
                                                inner_len)
        self.inner = nn.ModuleList([
            self._make_inner(kind, num_heads, ffn_hidden, attn_dropout,
                             gau_hidden)
            for kind in self.inner_kinds
        ])
        self.inner_len = len(self.inner)
        self.normact_mid = NormAct(self.mid_channels)
        self.conv_q = _ScaledConv2d(self.mid_channels, self.channels, 1,
                                    bias=False)

    def _make_inner(self, kind, num_heads, ffn_hidden, attn_dropout,
                    gau_hidden):
        """按类型造一个内块：``'nbt'`` → `TransformerBlock`，``'gau'`` → `GatedAttentionBlock`。

        两者接口对齐（4D→4D 的 ``forward(x, pos)``），所以放进同一个 `ModuleList`
        后 `forward` 的循环与 `initialize` 的逐块调用都不用改。
        """
        if kind == 'nbt':
            return TransformerBlock(self.mid_channels, num_heads, ffn_hidden,
                                    attn_dropout=attn_dropout)
        return GatedAttentionBlock(self.mid_channels, num_heads, ffn_hidden,
                                   attn_dropout=attn_dropout, hidden=gau_hidden)

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
    """value 头的 `poolRowsValueHead`：``mean``、``mean·(√A−14)/10``、``max``。

     第三段是 **max**，不是第三个缩放 mean。官方
    ``eigenbackend.cpp::poolRowsValueHead``：

        (*out)(c, n)                 = mean;
        (*out)(c + in->dim(0), n)    = mean * (sqrtdiv - 14.0f) * 0.1f;
        (*out)(c + 2*in->dim(0), n)  = m;      ← m 是该通道在空间维的最大值

    旧实现把第三段写成 ``mean·((√A−14)²/100 − 0.1)``（三个都是 mean 的
    仿射变换），于是第三段与第一段**线性相关**、不携带任何新信息 —— 相当于
    白白浪费三分之一的池化维度。用官方权重灌入对拍时，value 输出量级差约
    250 倍，正是这个错误的表现之一。

     ``max`` 那一路在官方实现里是带 mask 的：padding 位置先置为
      ``x + (mask − 1)`` 再取 max，保证 padding 永远选不到。由于我们只在
      有效位置（off-board 已被特征置零）上池化，且 padding 位置的激活恒
      ≤ 有效位置，直接 ``amax`` 即可。
    """
    b, c, h, w = x.shape
    area = float(h) * float(w)
    mean = x.mean(dim=(2, 3))
    scaled = mean * ((math.sqrt(area) - 14.0) / 10.0)
    mx = x.amax(dim=(2, 3))
    return torch.cat((mean, scaled, mx), dim=1)


# --------------------------------------------------------------------------- #
# 头
# --------------------------------------------------------------------------- #
class PolicyHead(nn.Module):
    """policy 头，K=2（π 与 π_opp），拓扑**严格对齐官方** ``PolicyHead::apply``。

    官方流程（``eigenbackend.cpp::PolicyHead::apply``）::

        p1Conv(C→P) ─────────────────────────────────────► p1Out
        g1Conv(C→G) → g1BN → poolRowsGPool → g1Concat(3G)
                                                      │
                            gpoolToBiasMul(3G→P) ─────┴─► g1Bias(P)
                                        addNCBiasInplace: p1Out += g1Bias
        p1BN(p1Out) → p2Conv(P→K) ────────────────────► policy (B,K,H,W)

        pass 支路（modelVersion ≥ 15）：
        g1Concat(3G) → gpoolToPassMul(3G→P) → gpoolToPassBias(P)
                    → passActivation → gpoolToPassMul2(P→K) ──► policyPass (B,K)

     **旧实现是错的，两处**：

    1. **bias 加的位置错了**。旧代码先 ``fuse(pooled)`` 再加到 ``pp`` 上，而
       官方是 ``gpoolToBiasMul`` 把 3G 投到 **P** 维、**逐通道**加到 ``p1Out``
       的**空间图**上（``addNCBiasInplace``），**然后**才做 ``p1BN``。也就是说
       bias 要先经过 BN 的仿射变换，顺序反了结果就不同。
    2. **pass 支路多了一层且激活位置错**。旧代码是
       ``pass_fc2(SiLU(pass_fc1(pooled)))``；官方是
       ``passMul2(passActivation(passMul(pooled) + passBias))`` ——
       ``passMul2`` 之前只有**一次**激活，且 ``gpoolToPassBias`` 是加在
       ``passMul`` 之后、``passActivation`` 之前的。

    另：官方 ``gpoolToPassMul`` 的输入是 **3G=144**（g1Concat 全量），
    旧代码同样用 3G，这点是对的。

    pass 是**真实的第 362 个策略索引**，与展平后的 361 个落点直接 concat，
    不做「两个分布再归一化」。
    """

    def __init__(self, in_channels, channels=48, gpool_channels=48,
                 num_outputs=2):
        super().__init__()
        self.num_outputs = int(num_outputs)
        # p1 分支：1×1 conv 直接吐 logits
        self.conv = _ScaledConv2d(in_channels, channels, 1, bias=False)
        self.normact = NormAct(channels)
        self.out = _ScaledConv2d(channels, self.num_outputs, 1, bias=False)
        # g1 分支：1×1 conv → BN → gpool(3G)，再分别投到 bias 与 pass
        self.conv_g = _ScaledConv2d(in_channels, gpool_channels, 1, bias=False)
        self.normact_g = NormAct(gpool_channels)
        self.fuse = _ScaledLinear(3 * gpool_channels, channels, bias=False)
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
        # 整头强制 FP32（推理后端对 FP16 敏感，policy 头输出精度不能压到 FP16）。
        trunk = trunk.float()
        b, _, h, w = trunk.shape
        pooled = gpool_policy(self.normact_g(self.conv_g(trunk)))   # (B,3G)

        # bias 在 **BN 之前**、逐通道加到空间图上（官方 addNCBiasInplace）
        pp = self.conv(trunk) + self.fuse(pooled).reshape(b, -1, 1, 1)
        spatial = self.out(self.normact(pp))                        # (B,K,H,W)
        spatial = spatial.reshape(b, self.num_outputs, h * w)

        if board_mask is not None:
            # 允许 (H,W) / (1,H,W) / (B,H,W) / (B,1,H,W)，一律规约到 (B,1,H*W)。
            if board_mask.dim() == 2:
                keep = board_mask.reshape(1, 1, h * w)
            else:
                keep = board_mask.reshape(board_mask.shape[0], 1, h * w)
            keep = keep.to(spatial.dtype).expand(b, 1, h * w)
            spatial = spatial.masked_fill(keep <= 0.5, OFF_BOARD_LOGIT)

        # pass 支路：passMul → +passBias → 激活 → passMul2（官方 modelVersion≥15）
        # `pass_fc1(bias=True)` 把 bias 与激活合成一步，等价于官方的
        # gpoolToPassMul + gpoolToPassBias + passActivation。
        pass_logit = self.pass_fc2(F.silu(self.pass_fc1(pooled))).unsqueeze(-1)
        return torch.cat((spatial, pass_logit), dim=2)


#: ``scorestdev = 20·SoftPlus(x₁, beta)`` 里的 beta。**spec §4.2 的字面写法是
#: `0.05`，已裁决改为 `1.0`**（2026-10-03）—— 下面保留推导链，因为「为什么不能
#: 随手改这个数」正是这段推导本身，而「spec 写的是 0.05」是推导的一环。
#:
#: **裁决依据：softplus(beta) 的量纲标定。**
#: ``F.softplus(x, beta) = log(1+exp(beta·x))/beta``，代入 x=0 得
#: ``softplus(0, beta) = log(2)/beta`` ⇒ **预测初值 = 20·log(2)/beta**：
#:
#: ==========  ==========================  ==========================
#: beta        20·softplus(0,beta) 初值    与 loss #7 的目标（5~20，δ=10）
#: ==========  ==========================  ==========================
#: 0.05 **277.26**（spec 字面值） 高出 27 倍 ⇒ 初期梯度被常数偏差支配
#: **1.0** **13.86**（本常量） 与 δ=10 同量级，起点即可用
#: ==========  ==========================  ==========================
#:
#: 取默认 ``beta=1.0``（PyTorch ``F.softplus`` 的默认值）后，这一项从「等效于
#: 没有学习信号」恢复为正常可训练项。
#:
#: **实测（`tests/test_katago_v7_budget.py::
#: test_score_stdev_softplus_term_lands_near_huber_delta` 与
#: test_score_stdev_loss_term_is_inside_huber_delta_at_the_ruled_out_beta`）**：
#: seed 0 初始化 + seed 7 输入（B=64）跑真实 forward，
#: ``score_stdev.mean()``：beta=0.05 → **278.2364**，beta=1.0 → **14.8824**；
#: 落到 loss #7 的公式值：beta=0.05 → **273.3450**，beta=1.0 → **9.9922**
#: （Huber δ=10 ⇒ 落在 δ **以内**，此时是二次段、梯度仍有效）。
#:
#: **这是纯常量，不改结构**：头部拓扑、参数量（**5,562,121**）、预算测试全部
#: 不受影响（有测试钉住）。另 **段 1 不训 score**（8 项系数逐个 0.0），所以本
#: 改动对段 1 的四个主目标（policy / π_opp / value / futurepos）**逐位无影响**，
#: 它是为**段 2/3**（接上 sidecar、把 score 系权重打开）生效的。
SCORE_STDEV_SOFTPLUS_BETA = 1.0

#: 官方 ``ModelPostProcessParams``（``desc.cpp``）的六个 multiplier，逐字照抄。
#: 它们不是超参，而是**引擎读 ``sv3Mul`` 六通道时写死的换算系数** ——
#: 导出到 ``.bin.gz`` 后由官方引擎自己做后处理，所以我们的 forward 必须
#: 用**完全相同**的系数，否则同一个 raw 数字在两边解释成不同的物理量。
#:
#: ==============================  ==========  ===========================
#: 字段                              multiplier  后处理
#: ==============================  ==========  ===========================
#: ``scoreMeanMultiplier``           20.0       ``raw × 20``
#: ``scoreStdevMultiplier``          20.0       ``softplus(raw) × 20``
#: ``leadMultiplier``                20.0       ``raw × 20``
#: ``varianceTimeMultiplier``        40.0       ``softplus(raw) × 40``
#: ``shorttermValueErrorMultiplier``  0.25       ``sqrt(softplus(raw)² × 0.25)``
#: ``shorttermScoreErrorMultiplier``  30.0       ``sqrt(softplus(raw)² × 30)``
#: ==============================  ==========  ===========================
SCORE_MEAN_MULTIPLIER = 20.0
SCORE_STDEV_MULTIPLIER = 20.0
LEAD_MULTIPLIER = 20.0
VARIANCE_TIME_MULTIPLIER = 40.0
#: **下面两个通道没有训练标签，权重必须保持 0。**
#:
#: 2026-10-03 已逐一核对官方 ``cpp/dataio/trainingwrite.cpp`` 里**全部**
#: ``rowGlobal[n] =`` 赋值（col 21~69 无遗漏），确认 stdata 的 64/80 列布局里
#: **不存在** shorttermWinlossError / shorttermScoreError：
#:
#:   · col 22 = varTimeLeft（已接标签，见 ``katago_npz.COL_VAR_TIME_LEFT``）
#:   · col 23 = 恒 0（源码就写 ``//Unused``）
#:   · col 30/31/32 = policySurprise / policyEntropy / searchEntropy
#: —— 这三列**曾**因分布相近被统计特征误判成 shortterm 两列
#:       （中位数 0.705 vs 0.708），查源码后推翻。它们是搜索统计量。
#:
#: 根因：``shorttermX = sqrt(softplus(raw)² · mult)`` 需要**网络 raw 输出**，
#: 而训练数据由搜索侧生成、当时还没有 NN 输出可依。KataGo 自己也是先训主干、
#: 再用自对弈重解析补这两路。
#:
#: ⇒ 当前状态：这两路**未被训练**，导出到引擎后恒为
#:   ``sqrt(softplus(bias)·mult)`` 的常量。对 MCTS 无害（仅在
#:   ``useUncertainty`` 时被读，且只影响 pruning 启发式），
#:   但**不得声称已训练**。要真正训练需重新生成带 raw 输出的 stdata。
SHORTTERM_WINLOSS_ERROR_MULTIPLIER = 0.25
SHORTTERM_SCORE_ERROR_MULTIPLIER = 30.0


class ValueHead(nn.Module):
    """value 头：``gpool_value(3V=144) → Linear(→W=96) → SiLU → h``，两支 3+6。

    **主路径与官方 ``ValueHead::apply`` 逐行一致**::

        v1Conv(C→V,1×1) → v1BN → poolRowsValueHead(3V) → v2Mul(3V→W)
                       → v2Bias → v2Activation
                       ├→ v3Mul(W→3) + v3Bias     ⇒ outcome (win/loss/noresult)
                       └→ sv3Mul(W→6) + sv3Bias   ⇒ scoreValue 六通道

     **小头（ownership / scoring / futurepos / seki）全部接在 ``VV``（池化之前）
    上**，不是接在 ``h`` 上 —— 它们的输出是 19×19 平面，空间分辨率不能被池化抹掉。
    这四个是本项目自研，官方没有，导出时丢弃。

     ownership 输出的是 **pretanh**（线性），不是 ``tanh`` 之后的值。loss #4
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
        # 3 → 6：官方 sv3Mul 的 out_channels=6。用户 2026-10-03 明确要求
        # 「真实训练缺失的三个通道」，所以这里补齐结构并接真实标签，而不是
        # 零填充 —— 零填充会让这三项在导出后恒为常量，引擎的 varTimeLeft /
        # shorttermWinlossError / shorttermScoreError 全部失去意义。
        self.scores = _ScaledLinear(hidden, 6, bias=True)
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
        # 整头强制 FP32（推理后端对 FP16 敏感；同时让 softplus/sqrt 链稳定）。
        trunk = trunk.float()
        vv = self.normact(self.conv(trunk))
        h = F.silu(self.fc(gpool_value(vv)))
        s = self.scores(h)
        # **数值敏感计算强制 FP32**（2026-10-05 真机 NaN 根因修复）。
        #   `softplus(x)=log(1+exp(x))` 与 `sqrt(softplus(...))` 在 FP16 下，
        #   当 `s[:,k]` 越过 ~11（warmup 后 value-head LR=3.7e-2 驱动权重漂移）
        #   即 `exp` 上溢成 inf ⇒ `score_stdev/var_time_left/shortterm_*` 全 inf，
        #   进而加权总 loss NaN、毒化全部梯度（实测逐项点名 = score_stdev，
        #   坏操作数 = ['score_stdev:pred','score_stdev:std']）。
        #   参数始终是 FP32（从未 .half()），autocast 只对「fp32 权重 × fp16 输入」
        #   降级；这里把 Linear 输出 `s` 提到 fp32，**整条 softplus/sqrt 链保持
        #   fp32**，开销可忽略（仅 6 个标量通道）。
        s_f = s.float()
        return {
            'outcome_logits': self.outcome(h),
            'score_mean': SCORE_MEAN_MULTIPLIER * s_f[:, 0],
            'score_stdev': SCORE_STDEV_MULTIPLIER * F.softplus(
                s_f[:, 1], beta=SCORE_STDEV_SOFTPLUS_BETA),
            'lead': LEAD_MULTIPLIER * s_f[:, 2],
            # ---- 官方 sv3 的后三个通道（varianceTimeLeft / shortterm×2）----
            # nneval.cpp（modelVersion>=10 分支）：
            #     varTimeLeft = softPlus(raw) * 40.0
            #     shorttermX  = sqrt(softPlus(raw*0.5)^2 * mult)
            # 而 softplus(u)² ≡ softplus(2u)，代入 u=raw/2 得
            #     shorttermX = sqrt(softplus(raw) * mult)
            # 后者少一次平方再开根，数值更稳，等价。
            'var_time_left': VARIANCE_TIME_MULTIPLIER * F.softplus(s_f[:, 3]),
            'shortterm_winloss_error': torch.sqrt(
                F.softplus(s_f[:, 4]) * SHORTTERM_WINLOSS_ERROR_MULTIPLIER),
            'shortterm_score_error': torch.sqrt(
                F.softplus(s_f[:, 5]) * SHORTTERM_SCORE_ERROR_MULTIPLIER),
            # raw 六通道原样保留（fp16，导出用，引擎自己后处理）。
            'score_value_raw': s,
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

     ``parity(i)·global[18]`` 那一路是 spec §2.3 ch18 的镜像：那个通道是
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
        # **数值敏感计算强制 FP32**（2026-10-05 真机 NaN 根因修复）。
        #   `logsumexp(mix+comp)` 的 `mix+comp` 是 (B,842,K) 的 FP16 Linear 输出；
        #   warmup 后该头权重漂移使分量到几万 ⇒ `inf` ⇒ `logsumexp(inf-inf)=nan`
        #   ⇒ `scorebelief_logits` 坏 ⇒ 派生 `sb_std` 坏（实测坏操作数之一）。
        #   参数始终 FP32，autocast 只对「fp32 权重 × fp16 输入」降级；这里把输入
        #   提到 fp32，整条 `fc_* + logsumexp` 链路保持 fp32，不再上溢。
        value_pooled = value_pooled.float()
        global_features = global_features.float()
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
#: V7 在 `backbone` 那三个 kind 之外**自己**的两个段。`backbone` 的 kind 是按块类
#: 命名的（res / transformer / legacy），而 V7 的 blocks 用的正是 `GC_RES`；
#: 剩下两个段是 V7 结构特有的，且都**在 blocks 之外**。
#:
#: 为什么要给它们开检查点：`scripts/train_sft.py::v7_batch_memory_advice` 的解析
#: 模型给 V7 定的预算是 checkpoint 开 ⇒ 8.1 MB/样本，而 910A 实测 batch 3000 用了
#: 64 GB ≈ 21.3 MB/样本。`V7_RESIDENT_MB_PER_SAMPLE[True]` 的注释写明那 6.9 MB
#: 是「**只存 block 边界**」—— 差额只可能落在 stem 与三个 head 上，而它们原来走
#: 裸前向、一次都不重算，中间激活在反向全程驻留。
GC_STEM = 'stem'
GC_HEADS = 'heads'

#: **只有 `ValueHead.forward` 返回 dict**（其余两个 head 返回张量），
#: `NbtTfNet._ckpt` 需要在调用 `torch.utils.checkpoint` 之前知道键序，才能把 dict
#: 摊成 tuple 穿过去、出来再装回 dict（缘由见 `_ckpt` 的 docstring）。
#:
#: 键用**子模块对象**而不是字符串：`self.value_head` 走 `nn.Module.__getattr__`
#: 命中 `self._modules`，返回的是**存着的那个模块对象本身**（不是每次新建的
#: bound method），所以同一个实例上反复取身份恒等、可以直接当 dict 键。
#: 登记表由 `build_katago_v7_net` 在返回模型前填一次（`_register_ckpt_dict_keys`）；
#: 查不到就退回原样返回，行为与改动前完全一致。
_CKPT_DICT_KEYS = {}

#: `ValueHead.forward` 的键序，**必须与 `:1168-1193` 的字面量逐字一致**。
#: 键序错了不会报错、只会让 `out` 的插入序变了 —— 而 `export_katago_bin.py` 与
#: `_dense_move_target` 都按名字取值，插序本身无害；真正的约束是「集合相同」。
_VALUE_HEAD_KEYS = (
    'outcome_logits',
    'score_mean',
    'score_stdev',
    'lead',
    'var_time_left',
    'shortterm_winloss_error',
    'shortterm_score_error',
    'score_value_raw',
    'ownership_pretanh',
    'scoring',
    'futurepos',
    'seki_logits',
    'value_pooled',
)


def _register_ckpt_dict_keys(model):
    """把 `model` 里**返回 dict 的 head** 登记进 `_CKPT_DICT_KEYS`（幂等）。

    必须**晚于** head 的构造：键是子模块对象本身，而对象是在 `NbtTfNet.__init__`
    里创建的。登记挂在 `build_katago_v7_net` 返回之前，保证第一次前向时表已就位。

    只有 `ValueHead` 在册 —— `PolicyHead` / `ScorebeliefHead` 返回的是张量，
    张量本来就能直接穿过 `checkpoint`（aot 的 higher-order-op 检查要的正是
    「纯张量」）。**别把返回张量的 head 也登记进来**：那会让 `_ckpt` 去对张量
    做 `[k]` 下标，直接 `IndexError`。
    """
    _CKPT_DICT_KEYS[model.value_head] = _VALUE_HEAD_KEYS
    return model


class NbtTfNet(GradCheckpointMixin, nn.Module):
    """V7 NBT+Transformer 整网（spec §3 + §4）。

    输入 ``(B, 22, H, W)`` 空间 + ``(B, 19)`` 全局，输出 `forward` 里那组张量。
    22 与 19 都由 ``fillRowV7`` 决定，**不从棋局重新推算**。

    梯度检查点覆盖 stem / blocks / 三个 head 三段（见 `GC_STEM` / `GC_HEADS`）。
    ``use_checkpoint`` 是**总开关**的别名，现在由 `GradCheckpointMixin` 持有 ——
    形状与语义与改造前逐位相同，只是从"内联 if"变成走 `run_segment`，因此重算期间
    也能恢复 autocast（`backbone._checkpointed._recompute_ctx`）。
    """

    #: 本模型是**双输入**网络：``forward(spatial, global_features)``，19 维全局
    #: 特征走独立的 ``global_fc``，不经 stem。
    #:
    #: 声明成类属性而不是让调用方去猜（``in_channels == 22``、或者探 ``forward``
    #: 签名），是因为「要不要第二路输入」是**架构属性**，与空间通道数是两件事：
    #: 将来若出现另一种 22 通道布局，按通道数判断就会认错网络。
    #:
    #: `GoAI.needs_global_features` 与 `MCTS._v7` 都读它 —— 判定链是
    #: 「网络声明 → GoAI 转述 → MCTS 采用」单源，不在下游各自复述一遍。
    REQUIRES_GLOBAL_FEATURES = True

    GRAD_CHECKPOINT_KINDS = (GC_RES, GC_TRANSFORMER, GC_LEGACY,
                             GC_STEM, GC_HEADS)
    #: 三段默认全开。`GC_LEGACY` 保持 False（V7 没有混合 block 列表）。
    #: 粒度（逐块 vs 并段）由 `backbone.GC_PER_BLOCK_DEFAULT[GC_RES]` 决定 = 逐块，
    #: 与 V18 旧机制一致（`git b65c804`：并段是 4 卡 OOM 的直接成因）。
    GRAD_CHECKPOINT_DEFAULTS = dict(GRAD_CHECKPOINT_DEFAULTS,
                                    **{GC_STEM: True, GC_HEADS: True})

    @property
    def use_checkpoint(self):
        """总开关的别名。读写都落到 mixin 的 `_gc_enabled`。

        **不动逐 kind 开关**：只翻这一个位不该顺手把 `set_grad_checkpointing(
        True, heads=False)` 那类逐段覆盖重置回默认值。
        """
        return self.grad_checkpointing

    @use_checkpoint.setter
    def use_checkpoint(self, value):
        self._gc_enabled = bool(value)

    def __init__(self, cfg=None, use_checkpoint=None, attn_dropout=None):
        super().__init__()
        c = dict(NBT_TF_CFG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        # 先起 mixin 的两个属性，再让 `use_checkpoint` 的 setter 有处可落。
        self._gc_kinds = dict(self.GRAD_CHECKPOINT_DEFAULTS)
        self._gc_enabled = False
        self.use_checkpoint = (c['use_checkpoint'] if use_checkpoint is None
                               else bool(use_checkpoint))
        self.attn_dropout = (c['attn_dropout'] if attn_dropout is None
                             else float(attn_dropout))
        C = c['trunk_channels']
        M = c['nbt_mid']
        self.stem = _ScaledConv2d(c['in_channels'], C, 3, padding=1, bias=False)
        self.global_fc = _ScaledLinear(c['global_channels'], C, bias=False)
        # GAU 按块布局：前 `gau_first_nbt` 块与末 `gau_last_nbt` 块是纯 nbt 边缘块，
        # 中间块才挂 GAU（位置由 `gau_positions` 指定）。这样 GAU 只出现在网络中部，
        # 两端保留纯 MHSA+SwiGLU，总参仍守在硬预算内（见 spec §6.1 / 预算测试）。
        gau_first = int(c.get('gau_first_nbt', 0))
        gau_last = int(c.get('gau_last_nbt', 0))
        gau_pos = c.get('gau_positions')
        gau_hidden = c.get('gau_hidden')
        self.blocks = nn.ModuleList([
            Nbt2TransformerBlock(C, M, c['num_heads'], c['ffn_hidden'],
                                 inner_len=c['num_inner_blocks'],
                                 attn_dropout=self.attn_dropout,
                                 attn_impl=c.get('attn_impl', 'nbt'),
                                 gau_positions=(None if (i < gau_first
                                                 or i >= c['num_blocks'] - gau_last)
                                                 else gau_pos),
                                 gau_hidden=gau_hidden)
            for i in range(c['num_blocks'])
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

    def _ckpt(self, fn, *args):
        """按 `GC_HEADS` 这段的开关决定是否走检查点。

        三个 head 是**并联**（都吃 `t`），而 `backbone.run_segment` 走的是
        `_segment_runner` 的**串联**语义（`out = blk(*cur)`），塞不进去；而包一层
        `HeadBank` 又会改掉 `state_dict` 的键（A/B/C 三段权重就互相 load 不上了）。
        所以就地内联，三段各一次 `checkpoint` —— 粒度仍是"每一段一次"，
        与 blocks 的逐块粒度同一口径。

        dict 返回值为什么要在**检查点内部**摊成 tuple
        ------------------------------------------------
        `torch.utils.checkpoint` 在 Dynamo 下被当作 higher-order operator 处理，
        而 aot 的检查只认**纯张量**输出（torch 2.1
        `torch/_functorch/aot_autograd.py` / `torch/_dynamo/variables/higher_order_ops.py`）::

            if not are_tensors(outs):
                raise ...("HigherOrderOperator body's output must consist of "
                          "tensors only")

        `ValueHead.forward` 返回的是 12 键的 dict（`:1168-1193`），于是
        **「开检查点 + 开 torch.compile」必然在前向第一帧抛**——与 batch、与设备、
        与数值都无关。torch 2.12 已经放宽了这条检查，所以这个坑在本地复现不出来，
        修它属于「按源码证据修」而不是「按复现修」。

        做法：被 checkpoint 的函数返回 tuple，**在 `checkpoint` 调用之外**再组装成
        原来的 dict（`_AS_DICT` 给出键顺序）。组装发生在 checkpoint 之外 ⇒ 不参与
        重算、也不进 higher-order op 的输出；两个分支返回的 dict 键序与值逐位相同。
        """
        if self.grad_checkpointing_for(GC_HEADS):
            if fn in _CKPT_DICT_KEYS:
                keys = _CKPT_DICT_KEYS[fn]
                packed = torch.utils.checkpoint.checkpoint(
                    lambda *a: tuple(fn(*a)[k] for k in keys), *args,
                    use_reentrant=False)
                return dict(zip(keys, packed))
            return torch.utils.checkpoint.checkpoint(fn, *args,
                                                     use_reentrant=False)
        return fn(*args)

    def trunk(self, spatial, global_features):
        """stem + 11 个 nbt2 块 + trunk 末端归一化（spec §3.1）。"""
        if self.grad_checkpointing_for(GC_STEM):
            x = torch.utils.checkpoint.checkpoint(self.stem, spatial,
                                                   use_reentrant=False)
        else:
            x = self.stem(spatial)
        g = self.global_fc(global_features).reshape(-1, self.cfg['trunk_channels'],
                                                    1, 1)
        x = x + g
        pos = _board_pos(x.shape[0], x.shape[2], x.shape[3], x.device)
        # `uncapped_last` 不传（= False）：legacy 路径那个开关存在是为了保住 v18 的
        # 31.12 GB 显存锚点，V7 没有那个锚点，而我们现在正在打 OOM
        # （见 `git b65c804` 的裁决）。
        x, _ = self.run_segment(self.blocks, (x, pos), GC_RES)
        return F.silu(self.norm_trunkfinal(x))

    def forward(self, spatial, global_features, board_mask=None):
        t = self.trunk(spatial, global_features)
        out = {'policy_logits': self._ckpt(self.policy_head, t, board_mask)}
        vout = self._ckpt(self.value_head, t)
        out.update({k: v for k, v in vout.items() if k != 'value_pooled'})
        out['scorebelief_logits'] = self._ckpt(
            self.scorebelief_head, vout['value_pooled'], global_features)
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
    # 必须在第一次前向之前登记：`_ckpt` 要靠它把 head 的 dict 返回值摊成 tuple
    # 穿过 `torch.utils.checkpoint`（见 `_ckpt` / `_register_ckpt_dict_keys`）。
    return _register_ckpt_dict_keys(net.initialize())
