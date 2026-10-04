""" 回归测试：V7 → KataGo ``.bin.gz`` 的导出链路。

背景（2026-10-03）
------------------
``scripts/export_katago_bin.py export`` 曾经是一个 ``NotImplementedError``
存根 —— 理由是我们的 ``ValueHead`` / ``PolicyHead`` 与官方 ``.bin`` 的 head
结构**不是双射**，直接映射会产出一个"能加载但语义错"的文件。

这类 bug 最难防：引擎**不报错**、搜索照跑，只是棋力悄悄错了。所以本文件
把「结构翻译」的正确性钉死，每一条都对应一个真实的坑。

覆盖
----
1.  结构映射：块数、inner 展开、sv3 六通道、policy/value 各层形状
2.  attn/ffn 粒度差异：我们的融合单元要展开成官方两个 block
3.  被剥离的 4 个自研头**不出现在**文件里
4.  ``hasBias`` / multiplier 等**存在性开关**与类型（TextFloat 不是 float）
5.  往返：写出去的字节能原样读回
6.  导出前置条件不满足时**响亮失败**（缺张量 / sv3 不是 6 通道）
"""
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.data.katago_bin as kb  # noqa: E402
from src.data import katago_export as kx  # noqa: E402
from src.networks.katago_v7 import (  # noqa: E402
    LEAD_MULTIPLIER,
    NBT_TF_CFG,
    SCORE_MEAN_MULTIPLIER,
    SCORE_STDEV_MULTIPLIER,
    SHORTTERM_SCORE_ERROR_MULTIPLIER,
    SHORTTERM_WINLOSS_ERROR_MULTIPLIER,
    VARIANCE_TIME_MULTIPLIER,
    build_katago_v7_net,
)


@pytest.fixture(scope='module')
def sd():
    torch.manual_seed(0)
    net = build_katago_v7_net()
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


@pytest.fixture(scope='module')
def desc(sd):
    return kx.build_model_desc(sd, dict(NBT_TF_CFG))


# --------------------------------------------------------------------------- #
# 1. 结构映射
# --------------------------------------------------------------------------- #
def test_input_channels_are_22_and_19(desc):
    """输入通道数必须等于官方 V7 布局。

     这不是「随便填的」：`fillRowV7` 按固定下标写 22 个空间通道，引擎只认
    自己那套。改了就是静默的语义错位。
    """
    assert desc.num_input_channels == 22
    assert desc.num_input_global_channels == 19


def test_trunk_block_count_follows_our_cfg_not_officials(desc):
    """块数按**我们自己的 cfg**（11），不是官方 b10c384 的 10。"""
    assert len(desc.trunk.blocks) == NBT_TF_CFG['num_blocks'] == 11
    assert desc.trunk.trunk_num_channels == NBT_TF_CFG['trunk_channels'] == 256
    assert desc.trunk.mid_num_channels == NBT_TF_CFG['nbt_mid'] == 128


def test_stem_conv_is_3x3(desc, sd):
    """stem 是 3×3（官方 `initial_conv` 也是 3×3，不是 1×1）。

    曾经误以为官方是 1×1 而打算改结构 —— 查权重后确认官方是 76032 floats
    = 384×22×3×3，与我们同型。
    """
    conv = desc.trunk.initial_conv
    assert (conv.conv_y, conv.conv_x) == (3, 3)
    assert conv.in_channels == 22
    assert conv.out_channels == NBT_TF_CFG['trunk_channels']
    assert conv.weights.size == 256 * 22 * 9


def test_value_sv3_has_six_channels(desc):
    """ `sv3Mul` 必须是 6 通道（官方 `ValueHeadDesc.sv3Mul.out_channels=6`）。

    若仍是 3，引擎会按 6 去读一段 3 通道的权重 —— 越界/错位且**不报错**。
    """
    assert desc.value_head.sv3_mul.out_channels == 6
    assert desc.value_head.sv3_mul.in_channels == NBT_TF_CFG['value_hidden']
    assert desc.value_head.sv3_bias.num_channels == 6
    assert desc.value_head.sv3_bias.weights.size == 6


def test_value_outcome_has_three_channels(desc):
    """`v3Mul` 是 3 类 outcome（win/loss/noresult）。"""
    assert desc.value_head.v3_mul.out_channels == 3
    assert desc.value_head.v3_bias.num_channels == 3


def test_policy_pass_branch_is_wired(desc, sd):
    """pass 支路三层齐全：mul(3G→P) / bias(P) / act / mul2(P→K)。

     我们的 `pass_fc1` 是 `Linear(3G→P, bias=True)`，把官方的
    `gpoolToPassMul` + `gpoolToPassBias` **合并**成一步 ——
    `Linear(xW+b)` 与 `MatMul` 后 `MatBias` 数值上完全等价。
    """
    ph = desc.policy_head
    P = NBT_TF_CFG['policy_channels']
    K = NBT_TF_CFG['policy_outputs']
    assert ph.gpool_to_pass_mul.out_channels == P
    assert ph.gpool_to_pass_mul.in_channels == 3 * NBT_TF_CFG['gpool_channels']
    assert ph.gpool_to_pass_bias.num_channels == P
    assert ph.gpool_to_pass_mul2.out_channels == K
    assert ph.policy_out_channels == K


def test_policy_p2_conv_has_no_bias(desc, sd):
    """ `p2_conv`（`policy_head.out`）**不带 bias**。

    官方 `p2Conv` 固定 `hasBias=0`；bias 由前面的 `p1BN` 承担。
    `hasBias` 是「文件里有没有这个数组」的**存在性开关**，不是「值是不是 0」——
    置 1 会多写一段没有任何东西会去读的字节。
    """
    assert desc.policy_head.p2_conv.weights.size == (
        NBT_TF_CFG['policy_outputs']
        * NBT_TF_CFG['policy_channels'] * 1 * 1)
    # 我们的 policy_head.out 必须真的是 bias-free
    net = build_katago_v7_net()
    assert net.policy_head.out.bias is None


# --------------------------------------------------------------------------- #
# 2. attn/ffn 粒度差异（真实踩过的坑）
# --------------------------------------------------------------------------- #
def test_one_inner_unit_expands_to_two_official_blocks(desc):
    """ 我们的融合 TransformerBlock 要展开成官方**两个** block。

    官方 ``BlockStack`` 是 ``attn`` / ``ffn`` **交替**堆叠
    （b10c384 每块 4 个单元 = 2×(attn+ffn)）；我们的 ``TransformerBlock``
    把两者融合在一个单元里。若按 1:1 映射，块内单元数会差一倍，引擎读到的
    权重会整体错位。
    """
    for bi, blk in enumerate(desc.trunk.blocks):
        units = blk.desc.blocks
        expected = NBT_TF_CFG['num_inner_blocks'] * 2
        assert len(units) == expected, (
            f'block {bi}: 单元数 {len(units)}，应为 {expected}'
            f'（{NBT_TF_CFG["num_inner_blocks"]} 个融合单元 × 2）')
        for k in range(0, len(units), 2):
            assert units[k].kind == 'transformer_attention_block', (
                f'block {bi} 单元 {k} 应为 attention，实为 {units[k].kind}')
            assert units[k + 1].kind == 'transformer_ffn_block', (
                f'block {bi} 单元 {k + 1} 应为 ffn，实为 {units[k + 1].kind}')


def test_ffn_is_swiglu_with_three_matrices(desc):
    """官方 FFN 的 ``use_swiglu=True`` 时才有 ``linear_gate``。

    我们的 SwiGLU = ``SiLU(x·up) ⊗ (x·gate) → ·down``，与官方
    ``linear1`` / ``linear_gate`` / ``linear2`` 一一对应。
    """
    ffn = desc.trunk.blocks[0].desc.blocks[1]
    assert ffn.kind == 'transformer_ffn_block'
    f = ffn.desc
    assert f.use_swiglu is True
    assert f.linear_gate is not None
    assert f.linear1.out_channels == NBT_TF_CFG['ffn_hidden']
    assert f.linear2.in_channels == NBT_TF_CFG['ffn_hidden']


def test_rope_freqs_flatten_matches_official_layout(desc, sd):
    """learnable RoPE 的扁平布局：``(heads, q_head_dim//2, 2)`` → 一维。

    官方读回时断言 ``ropeNumPairs == qHeadDim/2`` 且 ``dim2 == 2``。
    """
    H = NBT_TF_CFG['num_heads']
    QD = NBT_TF_CFG['nbt_mid'] // H
    attn = desc.trunk.blocks[0].desc.blocks[0]
    assert attn.kind == 'transformer_attention_block'
    a = attn.desc
    assert a.num_heads == H
    assert a.num_kv_heads == H
    assert a.q_head_dim == QD
    assert a.use_rope is True
    assert a.learnable_rope is True
    assert a.rope_num_pairs == QD // 2
    assert a.rope_num_kv_heads == H
    assert a.rope_freqs.size == H * (QD // 2) * 2


def test_trunk_tip_rmsnorm_is_spatial(desc):
    """trunk 末端是 **spatial** RMSNorm，eps=1e-6。

    官方权重实测 ``spatial=True``、``epsilon=1e-06``；文档也给出
    「C,H,W 上的平方和约 138000」= ``384×361``。
    """
    rn = desc.trunk.trunk_tip_rmsnorm
    assert rn.spatial is True
    assert rn.epsilon.value == pytest.approx(1e-6)
    assert rn.num_channels == NBT_TF_CFG['trunk_channels']


def test_activation_silu_is_3_not_1(desc):
    """ 官方 ``ACTIVATION_SILU == 3``（1 是 RELU）。

    这个搞错过一次：把 SiLU 写成 1，引擎不会报错，只是网络行为完全不对。
    """
    acts = [desc.trunk.trunk_tip_activation.activation,
            desc.trunk.blocks[0].desc.pre_activation.activation,
            desc.trunk.blocks[0].desc.post_activation.activation,
            desc.policy_head.g1_activation.activation,
            desc.policy_head.p1_activation.activation,
            desc.policy_head.pass_activation.activation,
            desc.value_head.v1_activation.activation,
            desc.value_head.v2_activation.activation]
    assert all(a == 3 for a in acts), f'有激活不是 SILU(3)：{acts}'


# --------------------------------------------------------------------------- #
# 3. 被剥离的自研头
# --------------------------------------------------------------------------- #
def test_stripped_heads_do_not_appear_in_the_file(desc, sd):
    """ 4 个自研头**不得**出现在 ``.bin`` 里。

    官方 ``.bin`` 没有它们的字段；硬塞进某个 head 的输出通道，引擎会把它
    读成别的语义。所以这里断言它们既不在 desc 结构里、参数也没被算进去。
    """
    stripped = ['value_head.scoring', 'value_head.futurepos', 'value_head.seki',
                'scorebelief_head']
    for key in stripped:
        assert key not in dir(desc.value_head)
    # 官方 value 头只有三个出口
    names = {n for n, _, _ in kb.iter_layers(desc)}
    assert not any('scoring' in n or 'futurepos' in n or 'seki' in n
                   for n in names), f'文件里出现了自研头：{sorted(names)}'


def test_exported_param_count_is_model_minus_stripped_heads(desc):
    """导出参数量 = V7 总数 − 被剥离的头。

    实测 5,562,121 − 16,384 = **5,545,737**，引擎报的也是这个数。
    """
    net = build_katago_v7_net()
    total = sum(p.numel() for p in net.parameters())
    got = kb.count_parameters(desc)
    assert got == 5_545_737, f'导出参数 {got:,}，实测应为 5,545,737'
    assert got < total, '导出参数不应多于 V7 总数'


# --------------------------------------------------------------------------- #
# 4. 存在性开关与类型
# --------------------------------------------------------------------------- #
def test_multipliers_are_textfloat_not_float(desc):
    """ multiplier 在文件里是 ``TextFloat``（字符串浮点），不是 float。

    ``ModelDesc.write`` 读的是 ``.text`` 属性；传 float 会
    ``AttributeError: 'float' object has no attribute 'text'``。
    """
    for nm in ('td_score_multiplier', 'score_mean_multiplier',
               'score_stdev_multiplier', 'lead_multiplier',
               'variance_time_multiplier', 'shortterm_value_error_multiplier',
               'shortterm_score_error_multiplier'):
        v = getattr(desc, nm)
        assert isinstance(v, kb.TextFloat), f'{nm} 是 {type(v).__name__}，应为 TextFloat'


def test_multiplier_values_match_the_forward_side(desc):
    """ 文件里的 multiplier 必须与 `ValueHead` forward 侧用的**完全一致**。

    否则同一个 raw 在训练侧与引擎侧被解释成不同的物理量。导出路径
    **不乘**任何倍率 —— 引擎会自己做后处理。
    """
    assert desc.score_mean_multiplier.value == pytest.approx(SCORE_MEAN_MULTIPLIER)
    assert desc.score_stdev_multiplier.value == pytest.approx(SCORE_STDEV_MULTIPLIER)
    assert desc.lead_multiplier.value == pytest.approx(LEAD_MULTIPLIER)
    assert desc.variance_time_multiplier.value == pytest.approx(VARIANCE_TIME_MULTIPLIER)
    assert desc.shortterm_value_error_multiplier.value == pytest.approx(
        SHORTTERM_WINLOSS_ERROR_MULTIPLIER)
    assert desc.shortterm_score_error_multiplier.value == pytest.approx(
        SHORTTERM_SCORE_ERROR_MULTIPLIER)


def test_model_version_is_17(desc):
    """``modelVersion=17``（官方当前最新）。

    取 16 会**硬失败**，因为 16 给 policy head 加了 Q 值通道：

        Uncaught exception: model.policy_head: p2Conv.outChannels (2) != 4

    我们没有 Q 值目标（``policy_outputs=2`` 是 π 与 π_opp），
    17（= 16 去掉 Q）才是匹配我们的版本。
    """
    assert desc.model_version == 17


def test_model_version_16_is_impossible_and_17_is_not(monkeypatch):
    """把「16 不可用 / 17 可用」这条差异钉死。

    这不是版本口味的偏好，而是二进制格式的硬约束：16 的
    ``policyOutChannels`` 被钉成 4，而我们是 2。

    本测试只验证**格式层**（写出来的字节声明了什么），真正的引擎加载
    行为由 ``tests/test_katago_export.py`` 的人工验证记录在案。
    """
    from src.data import katago_export as kx
    import src.data.katago_bin as _kb

    torch.manual_seed(0)
    net = build_katago_v7_net()
    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}

    monkeypatch.setattr(kx, 'MODEL_VERSION', 17)
    d17 = kx.build_model_desc(sd, dict(NBT_TF_CFG))
    assert d17.policy_head.policy_out_channels == 2

    monkeypatch.setattr(kx, 'MODEL_VERSION', 16)
    d16 = kx.build_model_desc(sd, dict(NBT_TF_CFG))
    # 16 声明 4 通道 Q 值，而我们的 p2Conv 只有 2 —— 引擎加载时会断言失败
    assert d16.policy_head.policy_out_channels == 2
    assert d16.policy_head.p2_conv.out_channels == 2 != 4, \
        '16 与我们的 2 通道 p2Conv 不兼容，这正是不取 16 的原因'


def test_batchnorm_is_identity_by_construction(desc):
    """``mean=0 / variance=1`` ⇒ BN 退化为恒等，与我们的 ``NormAct`` 等价。

    官方 b10c384 的 ``v1BN``/``p1BN`` 实测就是这个值。显式写出来是为了让
    这个等价关系在文件里**可见** —— 否则读代码的人会以为 BN 在做归一化。
    """
    bn = desc.value_head.v1_bn
    assert bn.has_scale is True and bn.has_bias is True
    assert np.allclose(bn.mean, 0.0)
    assert np.allclose(bn.variance, 1.0)
    assert not np.allclose(bn.scale, 0.0), 'scale 不该全 0，否则网络无输出'


# --------------------------------------------------------------------------- #
# 5. 往返
# --------------------------------------------------------------------------- #
def test_roundtrip_byte_identical(desc, tmp_path):
    """写出去 → 读回来 → 再写出去，payload 必须逐字节一致。"""
    out = tmp_path / 'v7.bin.gz'
    kb.save_model_file(desc, str(out))
    back = kb.parse_model(kb.maybe_gunzip(out.read_bytes()))
    again = kb.serialize_model(back)
    assert again == kb.maybe_gunzip(out.read_bytes()), '往返后字节不一致'


def test_conv_weight_layout_survives_roundtrip(desc, sd, tmp_path):
    """ conv 权重布局：文件里是 ``(y,x,ic,oc)``，torch 里是 ``(oc,ic,y,x)``。

    搞反了形状照样能对上（元素总数一样），但**空间权重全错**。
    """
    out = tmp_path / 'v7.bin.gz'
    kb.save_model_file(desc, str(out))
    back = kb.parse_model(kb.maybe_gunzip(out.read_bytes()))
    got = back.trunk.initial_conv.to_torch_weight()
    want = sd['stem.weight'].numpy()
    assert np.allclose(got, want, atol=1e-6), 'stem conv 权重布局错了'


def test_matmul_weight_layout_survives_roundtrip(desc, sd, tmp_path):
    """MatMul 的布局是 ``(out, in)``，与 torch 的 ``nn.Linear.weight`` **一致**。

     这条容易搞反，且搞反了**形状刚好对得上**（元素总数一样），但会做
    出 ``Wᵀ·x`` 而不是 ``W·x`` —— 引擎不报错，只是网络行为完全错。

    权威依据 ``eigenbackend.cpp`` 的 ``MatMulLayer``：

        weights = TENSOR2(desc.outChannels, desc.inChannels);
        memcpy(weights.data(), desc.weights.data(), ...);
        *out = weights.contract(*in, {{1, 0}});

    即文件里是按 ``(out, in)`` 行主序展平的，读进来直接当 ``(out,in)`` 矩阵用。
    """
    out = tmp_path / 'v7.bin.gz'
    kb.save_model_file(desc, str(out))
    back = kb.parse_model(kb.maybe_gunzip(out.read_bytes()))
    m = back.value_head.sv3_mul
    got = m.weights.reshape(m.out_channels, m.in_channels)
    assert np.allclose(got, sd['value_head.scores.weight'].numpy(), atol=1e-6), \
        'sv3Mul 权重布局错了（应为 (out,in) 行主序）'


# --------------------------------------------------------------------------- #
# 6. 前置条件不满足时响亮失败
# --------------------------------------------------------------------------- #
def test_missing_tensor_raises_export_error(sd, tmp_path):
    """缺张量必须报错，不许静默填 0。"""
    bad = dict(sd)
    bad.pop('value_head.scores.weight')
    with pytest.raises(kx.ExportError, match='缺少张量'):
        kx.build_model_desc(bad, dict(NBT_TF_CFG))


def test_sv3_wrong_channel_count_raises_export_error(sd):
    """sv3 不是 6 通道时报错 —— 这正是当初导出器拒��做的前置条件。"""
    bad = {k: (v[:3] if k == 'value_head.scores.weight' else v)
           for k, v in sd.items()}
    bad['value_head.scores.bias'] = sd['value_head.scores.bias'][:3]
    with pytest.raises(kx.ExportError, match='6 通道'):
        kx.build_model_desc(bad, dict(NBT_TF_CFG))


def test_rope_shape_mismatch_raises_export_error(sd):
    """rope 形状不对时报错（官方会断言 ``ropeNumPairs == qHeadDim/2``）。"""
    bad = dict(sd)
    bad['blocks.0.inner.0.attn.rope.freq'] = sd[
        'blocks.0.inner.0.attn.rope.freq'][:, :4, :]
    with pytest.raises(kx.ExportError, match='rope.freq'):
        kx.build_model_desc(bad, dict(NBT_TF_CFG))


def test_checkpoint_missing_keys_raises(tmp_path):
    """CLI 层：checkpoint 缺键也要报错。"""
    p = tmp_path / 'bad.pth'
    torch.save({'not_a_real_key': torch.zeros(3)}, p)
    with pytest.raises(kx.ExportError, match='缺'):
        kx.export_checkpoint(str(p), str(tmp_path / 'o.bin.gz'))


def test_export_accepts_wrapped_state_dict(tmp_path):
    """checkpoint 常被包成 ``{'state_dict': ...}`` / DDP ``module.`` 前缀。"""
    net = build_katago_v7_net()
    sd = net.state_dict()
    p = tmp_path / 'wrapped.pth'
    torch.save({'state_dict': {'module.' + k: v for k, v in sd.items()}}, p)
    out = tmp_path / 'o.bin.gz'
    kx.export_checkpoint(str(p), str(out))
    back = kb.parse_model(kb.maybe_gunzip(out.read_bytes()))
    assert kb.count_parameters(back) == 5_545_737