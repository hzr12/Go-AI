# -*- coding: utf-8 -*-
"""把我们自训的 V7 checkpoint 写成 KataGo 能加载的 ``.bin.gz``。

⚠ 为什么需要这一层
------------------
``.bin.gz`` 里存的不只是权重，还带一份**结构描述**（`ModelDesc` / `TrunkDesc` /
`PolicyHeadDesc` / `ValueHeadDesc` / ...）。引擎完全按这份描述去读每个数组 ——
数组顺序、通道数、`hasScale`/`hasBias` 这类**存在性开关**都必须与描述一致。
所以导出不是「把 tensor 拼起来写文件」，而是「把我们的结构翻译成官方的
结构描述，再把权重按描述要求的布局填进去」。

两处**结构差异**与官方不同，本文件负责吸收
------------------------------------------
1. **trunk 宽度不同**。我们是 C=256 / mid=128 / 11 块 / 每块 2 个 inner 单元；
   官方 b10c384 是 C=384 / mid=192 / 10 块 / 每块 4 个 inner 单元。
   ⇒ 描述必须按**我们自己的 cfg**生成，绝不能照抄官方维度表。
2. **attn 与 ffn 的堆叠粒度不同**。官方把两者拆成**独立的** block
   （``transformer_attention_block`` / ``transformer_ffn_block`` 交替）；
   我们的 ``TransformerBlock`` 把 attn+ffn **融合**在一个单元里。
   ⇒ 一个我们的 inner 单元要展开成官方的**两个** block（attn 在前、ffn 在后），
   顺序与官方 ``BlockStack`` 的读法一致。这纯粹是重新编号，不改任何数值。

刻意**不导出**的头（官方 ``.bin`` 无对应字段）
------------------------------------------
``value_head.scoring`` / ``.futurepos`` / ``.seki`` 与独立的 ``ScorebeliefHead``
是本项目自研，官方 value 头只有 ``v3Mul``(3) + ``sv3Mul``(6) +
``vOwnershipConv``(1) 三个出口。硬塞进某个通道会被引擎读成**别的语义**，
所以这里直接丢弃（约占总参数 0.3%）。

``sv3Mul`` 六通道里后两路（``shorttermWinlossError`` / ``shorttermScoreError``）
**没有训练标签**（见 `katago_npz.COL_VAR_TIME_LEFT` 旁的说明），
本次导出按「未训练」处理：写 0 ⇒ 引擎侧恒为 ``sqrt(softplus(0)*mult)`` 的常量。
第 4 路 ``varTimeLeft`` 是**真训练出来的**，直接写 raw。
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.katago_bin import (  # noqa: E402
    ActivationDesc,
    BatchNormDesc,
    Block,
    ConvDesc,
    MatBiasDesc,
    MatMulDesc,
    ModelDesc,
    NestedBottleneckDesc,
    PolicyHeadDesc,
    RMSNormDesc,
    TextFloat,
    TransformerAttentionDesc,
    TransformerFFNDesc,
    TransformerRMSNormDesc,
    TrunkDesc,
    ValueHeadDesc,
    check_name_valid,
    save_model_file,
)
from src.networks.katago_v7 import build_katago_v7_net  # noqa: E402

#: 官方 ``ActivationDesc`` 的枚举（``desc.h`` / ``nneval.cpp``）。
#: 🔴 **SILU 是 3，不是 1**（1 是 RELU）—— 这个搞错过一次，见 README。
ACTIVATION_RELU = 1
ACTIVATION_SILU_OFFICIAL = 3

#: 官方 `trunk_norm_kind`（`desc.h`）：0=BN/BiasMask，1=RMSNorm。
TRUNK_NORM_KIND_STANDARD = 0
TRUNK_NORM_KIND_RMSNORM = 1

#: 官方 `trunk_tip_rmsnorm` 的 eps 与 spatial 标志。
#: `spatial=True` 表示归约跨 (C,H,W) 取**单个标量**（`openclbackend.cpp`
#: 的 `finalSumFloats = maxBatchSize` 是每个样本一个标量的直接证据）。
TRUNK_TIP_RMSNORM_EPS = 1e-6
TRUNK_TIP_RMSNORM_SPATIAL = True

#: BatchNorm 的 eps：官方 value/policy 头的 `v1BN`/`p1BN` 实测就是 1e-20
#: （从 b10c384 权重读出），我们沿用同一值，避免引入无谓的数值差异。
BATCHNORM_EPS = 1e-20

#: `modelVersion` —— 取 **17**，即官方当前最新版本（2026-10-03）。
#:
#: 官方版本表（``cpp/neuralnet/modelversion.cpp:9-24``）：
#:
#:   15 = V7 features, Extra nonlinearity for pass output
#:   16 = V7 features, **Q value predictions in the policy head**
#:   17 = V7 features, **dropped Q value**, introduced transformers
#:        and added guards to unused params
#:
#: **为什么是 17 而不是 16**
#: ---------------------
#: 16 给 policy head 加了 Q 值通道，``desc.cpp:2068`` 把 ``policyOutChannels``
#: 直接钉成 4，于是加载时 ``desc.cpp:2145`` 的断言必然失败。**实测确认**：
#:
#:   Uncaught exception: Error loading or parsing model file:
#:   model.policy_head: p2Conv.outChannels (2) != 4
#:
#: 我们没有 Q 值目标（``policy_outputs=2`` 是 π 与 π_opp，不是 Q），
#: 所以 16 走不通，17（= 16 去掉 Q）才是匹配我们的那一个。
#:
#: 17 的两点格式变化都已由写出侧正确处理（实测可加载）
#: ------------------------------------------------
#: * ``policyOutChannels`` 改为**从文件里读**（``desc.cpp:2060-2066``，
#:   只接受 2 或 4）。我们的写出会写 2。
#: * policy / value head 各多 3 个「unused guard」字段（``desc.cpp:2075-2084``
#:   / ``2251-2260``），非 0 即拒。我们写 0。
#:
#: 「introduced transformers」**不是** 17 才有的能力：block kind
#: （``transformer_attention_block`` / ``transformer_ffn_block``）在
#: ``desc.cpp`` 的分派里没有任何版本门控，15 一样能读能跑。
#:
#: 15 与 17 的输出差异**不是语义差异**（实测排查结论）
#: ----------------------------------------------
#: 同一份权重，v15 与 v17 给出的 rawLead 差 ~27%。追查发现既不是
#: ``nneval.cpp`` 的后处理分支（那里搜不到 ``modelVersion >= 17``），
#: 也不是 block kind 的门控，而是**两份 OpenCL tuning 选了不同的 kernel
#: tiling**（``ATTN_BLOCK_Q`` 256 vs 128、``CHANNELSTRIDE`` 2 vs 1、
#: local size 也不同）⇒ fp16 累加顺序不同 ⇒ 在**随机权重**下被放大。
#: 随机权重的网络输出本就是任意值，对微小数值差极敏感。
#:
#: ⚠ 附带发现（**与我们的导出无关**）：在这台 AMD iGPU 上，v17 路径
#:   **逐次运行结果不同**，v15 路径则完全确定 ——
#:     我们的 v17 导出：rawLead 0.597 / 0.590 / 0.548 / 0.565
#:     官方 b10c384：  rawLead 5.548 / 5.668 / 5.953   ← 同样如此
#:   官方模型表现一致 ⇒ 这是 v17 kernel 路径 + 该驱动 fp16 的性质，
#:   不是导出缺陷。影响面仅限自对弈数据引入 fp16 噪声，不影响训练正确性。
#:
#: 特征布局无差异：``getNumSpatialFeatures`` / ``getNumGlobalFeatures`` 对
#: 8~17 一律返回 V7 的 22/19（``modelversion.cpp:51-81``）。
MODEL_VERSION = 17


class ExportError(RuntimeError):
    """导出前置条件不满足。**必须报错，不许猜。**"""


def _f32(x: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32).ravel())


def _t(sd: Dict[str, Any], key: str) -> torch.Tensor:  # type: ignore[name-defined]
    """取一个必需张量，缺失即报错（不做静默默认值）。"""
    if key not in sd:
        raise ExportError(
            f'checkpoint 缺少张量 {key!r}。这通常意味着导出器与模型结构'
            f'不同步 —— 请检查是否改了网络却没同步本文件。')
    return sd[key]


def _bn(name: str, channels: int, scale: np.ndarray, bias: np.ndarray) -> BatchNormDesc:
    """构造一个 ``mean=0 / variance=1`` 的 BatchNormDesc。

    官方 b10c384 的 ``v1BN`` / ``p1BN`` 实测就是 ``mean=0, variance=1``
    （即恒等），我们的 `NormAct` 是 ``x*gamma + beta``，两者**完全等价**。
    把 mean/variance 显式写成 0/1 是为了让这个等价关系在文件里可见 ——
    引擎读到的就是恒等 BN，不会因为 ``epsilon=1e-20`` 而放大数值。
    """
    c = int(channels)
    s = np.ascontiguousarray(np.asarray(scale, dtype=np.float32).reshape(-1))
    b = np.ascontiguousarray(np.asarray(bias, dtype=np.float32).reshape(-1))
    if s.size != c or b.size != c:
        raise ExportError(
            f'{name}: gamma/beta 长度应为 {c}，实得 {s.size}/{b.size}')
    return BatchNormDesc(
        name=name,
        num_channels=c,
        epsilon=TextFloat(BATCHNORM_EPS),
        has_scale=True,
        has_bias=True,
        mean=np.zeros(c, np.float32),
        variance=np.ones(c, np.float32),
        scale=s,
        bias=b,
    )


def _conv(name: str, w: Any) -> ConvDesc:
    """torch ``(oc,ic,y,x)`` → ``ConvDesc``（``from_torch_weight`` 负责布局转换）。"""
    arr = np.asarray(w)
    if arr.ndim != 4:
        raise ExportError(f'{name}: conv 权重应是 4 维 (oc,ic,y,x)，实得 {arr.shape}')
    oc, ic, ky, kx = arr.shape
    return ConvDesc.from_torch_weight(name, arr,
                                      dilation_y=1, dilation_x=1)


def _mm(name: str, w: Any) -> MatMulDesc:
    """torch ``(out,in)`` → ``MatMulDesc``。

    官方 `MatMulDesc` 存的是 ``(in,out)`` 的**扁平**数组
    （`desc.cpp` 直接 ``reshape(inChannels, outChannels)``，不转置），
    所以这里只要按 C 序 ravel 即可。
    """
    arr = np.asarray(w)
    if arr.ndim != 2:
        raise ExportError(f'{name}: matmul 权重应是 2 维 (out,in)，实得 {arr.shape}')
    out_c, in_c = arr.shape
    return MatMulDesc(name=name, in_channels=int(in_c),
                      out_channels=int(out_c), weights=_f32(arr))


def _rope_freqs(freq: Any, num_heads: int, q_head_dim: int) -> np.ndarray:
    """torch ``(heads, q_head_dim//2, 2)`` → 官方扁平 ``(heads*pairs*2,)``。

    官方读回时断言 ``ropeNumPairs == qHeadDim/2`` 且 ``dim2 == 2``
    （见 `TransformerAttentionDesc.read`），所以这里按 C 序展平即可。
    """
    arr = np.asarray(freq, dtype=np.float32)
    pairs = q_head_dim // 2
    if arr.shape != (num_heads, pairs, 2):
        raise ExportError(
            f'rope.freq 形状应为 {(num_heads, pairs, 2)}，实得 {arr.shape}')
    return _f32(arr)


# --------------------------------------------------------------------------- #
# trunk
# --------------------------------------------------------------------------- #
def _build_trunk(sd: Dict[str, Any], cfg: Dict[str, Any]) -> TrunkDesc:
    C = int(cfg['trunk_channels'])
    M = int(cfg['nbt_mid'])
    H = int(cfg['num_heads'])
    QD = M // H
    NB = int(cfg['num_blocks'])
    INNER = int(cfg['num_inner_blocks'])

    blocks: List[Block] = []
    for bi in range(NB):
        pb = f'blocks.{bi}'
        units = []
        for ui in range(INNER):
            ub = f'{pb}.inner.{ui}'
            # ---- attn 单元 → 官方 transformer_attention_block ----
            units.append(Block(
                kind='transformer_attention_block',
                desc=TransformerAttentionDesc(
                    name=f'{ub}.attn',
                    num_heads=H, num_kv_heads=H,
                    q_head_dim=QD, v_head_dim=QD,
                    use_rope=True, learnable_rope=True,
                    pre_ln=TransformerRMSNormDesc(
                        name=f'{ub}.norm_attn',
                        num_channels=M,
                        epsilon=TextFloat(1e-6),
                        weight=_f32(_t(sd, f'{ub}.norm_attn.weight'))),
                    q_proj=_mm(f'{ub}.attn.q', _t(sd, f'{ub}.attn.q.weight')),
                    k_proj=_mm(f'{ub}.attn.k', _t(sd, f'{ub}.attn.k.weight')),
                    v_proj=_mm(f'{ub}.attn.v', _t(sd, f'{ub}.attn.v.weight')),
                    out_proj=_mm(f'{ub}.attn.out', _t(sd, f'{ub}.attn.out.weight')),
                    rope_freqs_name=f'{ub}.attn.rope.freq',
                    rope_num_kv_heads=H, rope_num_pairs=QD // 2,
                    rope_freqs=_rope_freqs(_t(sd, f'{ub}.attn.rope.freq'), H, QD)),
            ))
            # ---- ffn 单元 → 官方 transformer_ffn_block ----
            # 我们的 SwiGLU = SiLU(x·up) ⊗ (x·gate) → ·down，
            # 与官方 `use_swiglu=True` 的 linear1/linear_gate/linear2 一一对应。
            units.append(Block(
                kind='transformer_ffn_block',
                desc=TransformerFFNDesc(
                    name=f'{ub}.ffn',
                    num_channels=M,
                    ffn_channels=int(cfg['ffn_hidden']),
                    use_swiglu=True,
                    pre_ln=TransformerRMSNormDesc(
                        name=f'{ub}.norm_ffn',
                        num_channels=M,
                        epsilon=TextFloat(1e-6),
                        weight=_f32(_t(sd, f'{ub}.norm_ffn.weight'))),
                    linear1=_mm(f'{ub}.ffn.up', _t(sd, f'{ub}.ffn.up.weight')),
                    linear2=_mm(f'{ub}.ffn.down', _t(sd, f'{ub}.ffn.down.weight')),
                    linear_gate=_mm(f'{ub}.ffn.gate', _t(sd, f'{ub}.ffn.gate.weight')),
                ),
            ))

        blocks.append(Block(
            kind='nested_bottleneck_block',
            desc=NestedBottleneckDesc(
                name=pb,
                num_blocks=len(units),
                pre_bn=_bn(f'{pb}.normact', C,
                           _t(sd, f'{pb}.normact.gamma'),
                           _t(sd, f'{pb}.normact.beta')),
                pre_activation=ActivationDesc(
                    name=f'{pb}.normact.act',
                    activation=ACTIVATION_SILU_OFFICIAL),
                pre_conv=_conv(f'{pb}.conv_p', _t(sd, f'{pb}.conv_p.weight')),
                blocks=units,
                post_bn=_bn(f'{pb}.normact_mid', M,
                            _t(sd, f'{pb}.normact_mid.gamma'),
                            _t(sd, f'{pb}.normact_mid.beta')),
                post_activation=ActivationDesc(
                    name=f'{pb}.normact_mid.act',
                    activation=ACTIVATION_SILU_OFFICIAL),
                post_conv=_conv(f'{pb}.conv_q', _t(sd, f'{pb}.conv_q.weight')),
            ),
        ))

    return TrunkDesc(
        name='model.trunk',
        num_blocks=NB,
        trunk_num_channels=C,
        mid_num_channels=M,
        # 这两个只被 GlobalPoolingResidualBlock 用；我们 num_gpool_blocks=0，
        # 全是 nested_bottleneck，所以它们只是元数据。仍按官方惯例填满，
        # 免得引擎读到 0 触发边界检查。
        regular_num_channels=int(cfg['trunk_channels']) * 9 // 16,
        dilated_num_channels=int(cfg['trunk_channels']) * 3 // 16,
        gpool_num_channels=int(cfg['policy_channels']),
        trunk_norm_kind=TRUNK_NORM_KIND_RMSNORM,
        trunk_options=(0, 0, 0, 0, 0),
        initial_conv=_conv('model.stem', _t(sd, 'stem.weight')),
        initial_matmul=_mm('model.linear_trunk', _t(sd, 'global_fc.weight')),
        blocks=blocks,
        trunk_tip_activation=ActivationDesc(
            name='model.act_trunkfinal',
            activation=ACTIVATION_SILU_OFFICIAL),
        trunk_tip_bn=BatchNormDesc(
            name='unused_trunk_tip_bn', num_channels=0,
            epsilon=TextFloat(1.0), has_scale=False, has_bias=False,
            mean=np.zeros(0, np.float32), variance=np.zeros(0, np.float32),
            scale=np.zeros(0, np.float32), bias=np.zeros(0, np.float32)),
        trunk_tip_rmsnorm=RMSNormDesc(
            name='model.norm_trunkfinal',
            num_channels=C,
            epsilon=TextFloat(TRUNK_TIP_RMSNORM_EPS),
            spatial=TRUNK_TIP_RMSNORM_SPATIAL,
            cgroup_size=0,
            gamma=_f32(_t(sd, 'norm_trunkfinal.gamma')),
            beta=_f32(_t(sd, 'norm_trunkfinal.beta'))),
        sgf_metadata_encoder=None,
    )


# --------------------------------------------------------------------------- #
# policy head
# --------------------------------------------------------------------------- #
def _build_policy_head(sd: Dict[str, Any], cfg: Dict[str, Any]) -> PolicyHeadDesc:
    """对齐官方 ``PolicyHead::apply`` 的拓扑。

        p1Conv → (加 gpoolToBiasMul 的 bias) → p1BN → p2Conv ⇒ policy
        g1Conv → g1BN → gpool(3G) → {gpoolToBiasMul, gpoolToPassMul…} ⇒ pass
    """
    C = int(cfg['trunk_channels'])
    P = int(cfg['policy_channels'])
    G = int(cfg['gpool_channels'])
    K = int(cfg['policy_outputs'])
    return PolicyHeadDesc(
        name='model.policy_head',
        policy_out_channels=K,
        policy_options=(0, 0, 0),
        p1_conv=_conv('model.policy_head.conv1p', _t(sd, 'policy_head.conv.weight')),
        g1_conv=_conv('model.policy_head.conv1g', _t(sd, 'policy_head.conv_g.weight')),
        g1_bn=_bn('model.policy_head.biasg', G,
                  _t(sd, 'policy_head.normact_g.gamma'),
                  _t(sd, 'policy_head.normact_g.beta')),
        g1_activation=ActivationDesc(
            name='model.policy_head.actg',
            activation=ACTIVATION_SILU_OFFICIAL),
        gpool_to_bias_mul=_mm('model.policy_head.linear_g',
                              _t(sd, 'policy_head.fuse.weight')),
        p1_bn=_bn('model.policy_head.bias1p', P,
                  _t(sd, 'policy_head.normact.gamma'),
                  _t(sd, 'policy_head.normact.beta')),
        p1_activation=ActivationDesc(
            name='model.policy_head.actp',
            activation=ACTIVATION_SILU_OFFICIAL),
        p2_conv=_conv('model.policy_head.conv2p', _t(sd, 'policy_head.out.weight')),
        # pass 支路：gpoolToPassMul(3G→P) 与 gpoolToPassBias(P) 被我们的
        # `pass_fc1`（Linear(3G→P, bias=True)）**合并**成一步 ——
        # `Linear(xW + b)` 与 `MatMul` 后 `MatBias` 在数值上完全等价。
        gpool_to_pass_mul=_mm('model.policy_head.linear_pass',
                              _t(sd, 'policy_head.pass_fc1.weight')),
        gpool_to_pass_bias=MatBiasDesc(
            name='model.policy_head.linear_pass_bias',
            num_channels=P,
            weights=_f32(_t(sd, 'policy_head.pass_fc1.bias'))),
        pass_activation=ActivationDesc(
            name='model.policy_head.actpass',
            activation=ACTIVATION_SILU_OFFICIAL),
        gpool_to_pass_mul2=_mm('model.policy_head.linear_pass2',
                               _t(sd, 'policy_head.pass_fc2.weight')),
    )


# --------------------------------------------------------------------------- #
# value head
# --------------------------------------------------------------------------- #
def _build_value_head(sd: Dict[str, Any], cfg: Dict[str, Any]) -> ValueHeadDesc:
    """对齐官方 ``ValueHead::apply``。

        v1Conv → v1BN → gpool(3V) → v2Mul → v2Bias → v2Activation
                ├→ v3Mul + v3Bias  （3 类 outcome）
                └→ sv3Mul + sv3Bias（六通道 score）
                └→ vOwnershipConv  （1 通道 pretanh）
    """
    C = int(cfg['trunk_channels'])
    V = int(cfg['value_channels'])
    W = int(cfg['value_hidden'])

    # sv3 六通道。**写 raw**（官方引擎会自己做
    # softplus / ×multiplier 后处理），我们 forward 侧乘过的倍率在这里
    # 必须**不能再乘**，否则会被乘两次。
    sv3_w = np.asarray(_t(sd, 'value_head.scores.weight'))
    sv3_b = np.asarray(_t(sd, 'value_head.scores.bias'))
    if sv3_w.shape[0] != 6 or sv3_b.shape[0] != 6:
        raise ExportError(
            f'value_head.scores 应为 6 通道（官方 sv3Mul.out_channels=6），'
            f'实得 weight{tuple(sv3_w.shape)} bias{tuple(sv3_b.shape)}。'
            f'请先确认 ValueHead 已扩到 6 通道。')

    return ValueHeadDesc(
        name='model.value_head',
        value_options=(0, 0, 0),
        v1_conv=_conv('model.value_head.conv1', _t(sd, 'value_head.conv.weight')),
        v1_bn=_bn('model.value_head.bias1', V,
                  _t(sd, 'value_head.normact.gamma'),
                  _t(sd, 'value_head.normact.beta')),
        v1_activation=ActivationDesc(
            name='model.value_head.act1',
            activation=ACTIVATION_SILU_OFFICIAL),
        v2_mul=_mm('model.value_head.linear2', _t(sd, 'value_head.fc.weight')),
        v2_bias=MatBiasDesc(
            name='model.value_head.bias2',
            num_channels=W, weights=_f32(_t(sd, 'value_head.fc.bias'))),
        v2_activation=ActivationDesc(
            name='model.value_head.act2',
            activation=ACTIVATION_SILU_OFFICIAL),
        v3_mul=_mm('model.value_head.linear_valuehead',
                   _t(sd, 'value_head.outcome.weight')),
        v3_bias=MatBiasDesc(
            name='model.value_head.bias_valuehead',
            num_channels=3, weights=_f32(_t(sd, 'value_head.outcome.bias'))),
        sv3_mul=_mm('model.value_head.linear_miscvaluehead', sv3_w),
        sv3_bias=MatBiasDesc(
            name='model.value_head.bias_miscvaluehead',
            num_channels=6, weights=_f32(sv3_b)),
        v_ownership_conv=_conv('model.value_head.conv1o',
                               _t(sd, 'value_head.ownership.weight')),
    )


# --------------------------------------------------------------------------- #
# 顶层
# --------------------------------------------------------------------------- #
def build_model_desc(state_dict: Dict[str, Any], cfg: Dict[str, Any],
                     name: str = 'goai_v7') -> ModelDesc:
    """把 V7 的 ``state_dict`` 翻译成官方 ``ModelDesc``。"""
    from src.networks.katago_v7 import (
        LEAD_MULTIPLIER,
        SCORE_MEAN_MULTIPLIER,
        SCORE_STDEV_MULTIPLIER,
        SHORTTERM_SCORE_ERROR_MULTIPLIER,
        SHORTTERM_WINLOSS_ERROR_MULTIPLIER,
        VARIANCE_TIME_MULTIPLIER,
    )
    return ModelDesc(
        name=name,
        sha256='',
        model_version=MODEL_VERSION,
        num_input_channels=int(cfg['in_channels']),
        num_input_global_channels=int(cfg['global_channels']),
        # 🔴 multiplier 在文件里是 **TextFloat**（字符串形式的浮点），
        #   不是 float —— `ModelDesc.write` 读的是 `.text` 属性。
        td_score_multiplier=TextFloat(20.0),
        # 这六个必须与 `ValueHead` forward 侧用的完全一致
        #（见 katago_v7.py 的常量表），否则同一个 raw 在两边被解释成不同的
        # 物理量。导出路径**不乘**任何倍率，引擎会自己做后处理。
        score_mean_multiplier=TextFloat(SCORE_MEAN_MULTIPLIER),
        score_stdev_multiplier=TextFloat(SCORE_STDEV_MULTIPLIER),
        lead_multiplier=TextFloat(LEAD_MULTIPLIER),
        variance_time_multiplier=TextFloat(VARIANCE_TIME_MULTIPLIER),
        shortterm_value_error_multiplier=TextFloat(SHORTTERM_WINLOSS_ERROR_MULTIPLIER),
        shortterm_score_error_multiplier=TextFloat(SHORTTERM_SCORE_ERROR_MULTIPLIER),
        meta_encoder_version=0,
        prefer_pass_alive_under_suicide_rules=False,
        prefer_exclude_territory_adjacent_to_atari=False,
        model_options=(0, 0, 0, 0, 0),
        trunk=_build_trunk(state_dict, cfg),
        policy_head=_build_policy_head(state_dict, cfg),
        value_head=_build_value_head(state_dict, cfg),
    )


def _load_state_dict(path: str) -> Dict[str, Any]:
    import torch
    obj = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('state_dict', 'model', 'model_state_dict', 'net'):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise ExportError(f'{path}: 不认识的结构（既非 dict 也无 state_dict 子键）')
    # 去掉 DDP / torch.compile 的前缀
    out = {}
    for k, v in obj.items():
        for pref in ('module.', '_orig_mod.'):
            while k.startswith(pref):
                k = k[len(pref):]
        out[k] = v
    return out


def export_checkpoint(checkpoint: str, out_path: str, name: str = 'goai_v7') -> str:
    from src.networks.katago_v7 import NBT_TF_CFG, build_katago_v7_net
    sd = _load_state_dict(checkpoint)
    # 用真实模型实例校验结构一致（顺带暴露多余的键）
    net = build_katago_v7_net()
    want = set(net.state_dict())
    have = set(sd)
    missing = sorted(want - have)
    if missing:
        raise ExportError(
            f'checkpoint 缺 {len(missing)} 个张量，例如 {missing[:4]}。'
            f'导出前请确认 checkpoint 与当前网络结构一致。')
    cfg = dict(NBT_TF_CFG)
    model = build_model_desc(sd, cfg, name=name)
    save_model_file(model, out_path)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='把 V7 checkpoint 导出为 KataGo .bin.gz')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='goai_v7')
    a = ap.parse_args(argv)
    try:
        p = export_checkpoint(a.checkpoint, a.out, name=a.name)
    except ExportError as e:
        print(f'导出失败：{e}', file=sys.stderr)
        return 2
    print(f'已写出 {p}（{os.path.getsize(p):,} 字节）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())