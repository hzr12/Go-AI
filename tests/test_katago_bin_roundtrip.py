"""`src/data/katago_bin.py` 的 round-trip 验证 —— 我们对 `.bin.gz` 格式的理解是否 100% 正确

**唯一的验收标准**：解析官方权重 `katago/kata1-tf2-b10c384-s2941M-d5872M.bin.gz`，
原样重新序列化，与原文件 **byte 级完全一致**。任何不一致都是格式理解错了。

这些测试里断言的每个数字都有独立来源，不是"用被测代码算一遍再和自己比"：
* 拓扑数字来自 brief（块数 10 / trunkC 384 / mid 192 / heads 6 / qDim 32 ...）
* 参数个数 `10,545,753` 来自官方引擎日志
  （`analysis_logs/20261002-090611-30329889.log:26` 与
  `katago/katago-v1.18.1-opencl-windows-x64/gtp_logs/20261001-203015-114F3A2D.log:53`：
  `Model name: b10c384h6nbttflrs (nbt transformer, 10545753 params)`）
* 长度 42,248,752 / 消耗 42,248,751 是给定的事实
* 文件里 float 总数 10,557,657、BN running stats 11,904（23 个 BN）、
  次正规数 444,703（4.212%）—— 这几个都是**独立地从原始字节重算**出来的，
  与本模块的统计函数互相印证。
"""

from __future__ import annotations

import gzip
import os
import struct
from typing import Any, Dict, List

import numpy as np
import pytest

from src.data import katago_bin as kb
from src.data.katago_bin import (
    ActivationDesc,
    BatchNormDesc,
    Block,
    ConvDesc,
    GlobalPoolingResidualBlockDesc,
    MatBiasDesc,
    MatMulDesc,
    ModelDesc,
    NestedBottleneckDesc,
    PolicyHeadDesc,
    ResidualBlockDesc,
    RMSNormDesc,
    SGFMetadataEncoderDesc,
    TextFloat,
    TransformerAttentionDesc,
    TransformerFFNDesc,
    TransformerRMSNormDesc,
    TrunkDesc,
    ValueHeadDesc,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO, "katago", "kata1-tf2-b10c384-s2941M-d5872M.bin.gz")

# ---- 来自 brief / 官方日志的硬数字（独立来源，不是被测代码算出来的） ----
RAW_LEN = 42_248_752
CONSUME_LEN = 42_248_751
OFFICIAL_PARAMS = 10_545_753  # 官方日志 "nbt transformer, 10545753 params"

_MISSING = (
    f"官方权重不在 {MODEL_PATH}。round-trip 验收无法进行 —— "
    "请把 kata1-tf2-b10c384-s2941M-d5872M.bin.gz 放到 katago/ 下再跑，"
    "不要因为文件缺失就把这条测试静默跳过。"
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_bytes() -> bytes:
    if not os.path.exists(MODEL_PATH):
        pytest.skip(_MISSING)
    with open(MODEL_PATH, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def model(model_bytes: bytes) -> ModelDesc:
    return kb.parse_model(model_bytes)


@pytest.fixture(scope="module")
def raw(model_bytes: bytes) -> bytes:
    return kb.maybe_gunzip(model_bytes)


# ---------------------------------------------------------------------------
# 主测试：byte 级完全一致
# ---------------------------------------------------------------------------


def test_roundtrip_of_official_weights_is_byte_identical(model_bytes: bytes) -> None:
    """解析 -> 原样重新序列化 -> 与原文件 byte 级 diff，必须完全一致。"""
    parsed, report = kb.roundtrip_diff(model_bytes)

    assert report.identical, report.summary
    assert report.original_length == RAW_LEN
    assert report.reserialized_length == RAW_LEN

    # 再独立地断言一次，避免 report 自己算错自己骗自己
    reference = kb.maybe_gunzip(model_bytes)
    reserialized = kb.serialize_model(parsed)
    assert reserialized == reference

    if not report.identical:  # pragma: no cover - 上面已经 assert 了
        pytest.fail(
            f"第一个差异在第 {report.first_diff_offset} 字节; "
            f"期望 {report.first_diff_expected!r}, 实际 {report.first_diff_actual!r}; "
            f"共 {report.num_diff_bytes} 字节不同"
        )


def test_parser_consumes_exactly_one_byte_less_than_the_file(raw: bytes, model_bytes: bytes) -> None:
    """坑 #1：C++ 读完最后一个 float **不消费**块尾的 `\\n`，所以消耗 = 长度 - 1。

    42,248,752 vs 42,248,751 的全部来源。反序列化时那个 `\\n` 必须写回去。
    """
    _, consumed = kb.read_model_bytes(model_bytes)
    assert consumed == CONSUME_LEN
    assert raw[consumed:] == b"\n"
    assert len(raw) - consumed == 1


def test_trailing_newline_is_optional_for_parsing_but_required_for_byte_identity(
    model_bytes: bytes,
) -> None:
    """把末尾那个 `\\n` 删掉，解析仍然成功（证明它确实是纯尾部装饰），
    但重新序列化一定会把它加回来 —— 所以 round-trip 依然等于**原文件**。"""
    raw = kb.maybe_gunzip(model_bytes)
    truncated = raw[:-1]

    parsed, consumed = kb.read_model_bytes(truncated)
    assert consumed == len(truncated)

    reference = kb.maybe_gunzip(model_bytes)
    assert kb.serialize_model(parsed) == reference
    assert kb.serialize_model(parsed).endswith(b"\n")


def test_at_marker_count_matches_emitted_float_blocks(raw: bytes, model: ModelDesc) -> None:
    """文件里 `@BIN@` 的个数必须等于序列化器发出的 float 块个数（328）。"""
    emitted = []
    original = kb._Writer.floats

    def spy(self: kb._Writer, arr: np.ndarray) -> None:
        emitted.append(int(np.ascontiguousarray(arr, dtype="<f4").size))
        return original(self, arr)

    kb._Writer.floats = spy  # type: ignore[method-assign]
    try:
        kb.serialize_model(model)
    finally:
        kb._Writer.floats = original  # type: ignore[method-assign]

    assert raw.count(kb.BIN_MARKER) == len(emitted)
    assert sum(emitted) == kb.count_file_floats(model)


# ---------------------------------------------------------------------------
# 分解断言：解析出的结构
# ---------------------------------------------------------------------------


def test_model_header_fields(model: ModelDesc) -> None:
    assert model.name == "b10c384h6nbttflrs"
    assert model.model_version == 17
    assert model.num_input_channels == 22
    assert model.num_input_global_channels == 19
    assert model.meta_encoder_version == 0
    assert model.prefer_pass_alive_under_suicide_rules is False
    assert model.prefer_exclude_territory_adjacent_to_atari is False
    assert model.model_options == (0, 0, 0, 0, 0)
    assert kb._inputs_version(17) == 7
    # 7 个 post-process 缩放，逐字保留
    assert model.td_score_multiplier.text == "20.0"
    assert model.variance_time_multiplier.text == "40.0"
    assert model.shortterm_value_error_multiplier.text == "0.25"
    assert model.shortterm_score_error_multiplier.text == "150.0"


def test_trunk_header_fields(model: ModelDesc) -> None:
    t = model.trunk
    assert t.num_blocks == 10
    assert t.trunk_num_channels == 384
    assert t.mid_num_channels == 192
    assert t.regular_num_channels == 144
    assert t.dilated_num_channels == 48  # 源码标 //unused，但在文件里
    assert t.gpool_num_channels == 48
    assert t.trunk_norm_kind == kb.TRUNK_NORM_KIND_RMSNORM
    assert t.trunk_options == (0, 0, 0, 0, 0)
    assert t.initial_conv.conv_y == 3 and t.initial_conv.conv_x == 3
    assert t.initial_conv.in_channels == 22
    assert t.initial_conv.out_channels == 384
    assert t.initial_matmul.in_channels == 19
    assert t.initial_matmul.out_channels == 384
    # trunkNormKind == 1 -> tip 是 RMSNorm 不是 BatchNorm
    assert t.trunk_tip_rmsnorm.num_channels == 384
    assert t.trunk_tip_bn.num_channels == 0
    assert t.sgf_metadata_encoder is None


def test_all_ten_trunk_blocks_are_nested_bottleneck(model: ModelDesc) -> None:
    t = model.trunk
    assert len(t.blocks) == 10
    assert all(b.kind == "nested_bottleneck_block" for b in t.blocks)
    assert all(isinstance(b.desc, NestedBottleneckDesc) for b in t.blocks)
    assert model.has_any_nested_bottleneck_blocks() is True
    assert model.has_any_transformer_blocks() is True


def test_each_nested_bottleneck_has_two_attn_and_two_ffn(model: ModelDesc) -> None:
    for i, block in enumerate(model.trunk.blocks):
        nb: NestedBottleneckDesc = block.desc  # type: ignore[assignment]
        assert nb.num_blocks == 4
        assert len(nb.blocks) == 4
        kinds = [b.kind for b in nb.blocks]
        assert kinds == [
            "transformer_attention_block",
            "transformer_ffn_block",
            "transformer_attention_block",
            "transformer_ffn_block",
        ], f"block {i}: {kinds}"

        # pre/post 都是 1x1，384 -> 192 -> 384
        assert (nb.pre_conv.conv_y, nb.pre_conv.conv_x) == (1, 1)
        assert (nb.pre_conv.in_channels, nb.pre_conv.out_channels) == (384, 192)
        assert (nb.post_conv.in_channels, nb.post_conv.out_channels) == (192, 384)
        assert nb.pre_bn.num_channels == 384
        assert nb.post_bn.num_channels == 192


def test_attention_blocks_use_six_heads_of_dim_32_with_rope(model: ModelDesc) -> None:
    attns: List[TransformerAttentionDesc] = []
    ffns: List[TransformerFFNDesc] = []
    for block in model.trunk.blocks:
        for sub in block.desc.blocks:  # type: ignore[union-attr]
            if isinstance(sub.desc, TransformerAttentionDesc):
                attns.append(sub.desc)
            elif isinstance(sub.desc, TransformerFFNDesc):
                ffns.append(sub.desc)

    assert len(attns) == 20
    assert len(ffns) == 20
    for a in attns:
        assert a.num_heads == 6
        assert a.num_kv_heads == 6
        assert a.q_head_dim == 32
        assert a.v_head_dim == 32
        assert a.use_rope is True
        assert a.learnable_rope is True
        assert a.q_proj.out_channels == 6 * 32
        assert a.out_proj.in_channels == 6 * 32
        assert a.rope_freqs.size == 6 * (32 // 2) * 2
        assert a.rope_theta is None  # learnable rope 不写 theta
    for f in ffns:
        assert f.num_channels == 192
        assert f.ffn_channels == 512
        assert f.use_swiglu is True
        assert f.linear_gate.weights.size == 192 * 512


def test_policy_head_fields(model: ModelDesc) -> None:
    p = model.policy_head
    assert p.policy_out_channels == 2
    assert model.num_policy_channels == 2
    assert p.policy_options == (0, 0, 0)
    assert p.p1_conv.in_channels == 384
    assert p.g1_conv.in_channels == 384
    assert p.gpool_to_pass_bias.num_channels == p.p1_conv.out_channels


def test_value_head_fields(model: ModelDesc) -> None:
    assert model.num_value_channels == 3
    assert model.num_score_value_channels == 6
    assert model.num_ownership_channels == 1
    assert model.value_head.value_options == (0, 0, 0)
    assert ValueHeadDesc.expected_score_value_channels(17) == 6
    assert ValueHeadDesc.expected_score_value_channels(8) == 4
    assert ValueHeadDesc.expected_score_value_channels(5) == 2
    assert ValueHeadDesc.expected_score_value_channels(3) == 1


def test_bn_has_scale_false_is_present_in_this_model(model: ModelDesc) -> None:
    """坑 #6 在这个文件里是**活的**：3 个 BN 的 `has_scale=0`，它们的 scale 数组
    是 C++ 自己填的全 1，**不在文件里**。写回去就会多出 3*48*4 = 576 字节。"""
    no_scale = [b.name for b in _all_batch_norms(model) if b.num_channels > 0 and not b.has_scale]
    assert sorted(no_scale) == [
        "model.policy_head.bias2",
        "model.policy_head.biasg",
        "model.value_head.bias1",
    ]
    assert all(b.has_bias for b in _all_batch_norms(model) if b.num_channels > 0)


def _all_batch_norms(obj: Any) -> List[BatchNormDesc]:
    out: List[BatchNormDesc] = []

    def walk(o: Any) -> None:
        if isinstance(o, BatchNormDesc):
            out.append(o)
            return
        if isinstance(o, np.ndarray):
            return
        if isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
            return
        fields = getattr(o, "__dataclass_fields__", None)
        if fields is None:
            return
        for f in fields:
            walk(getattr(o, f))

    walk(obj)
    return out


def test_iter_layers_inventory_is_complete(model: ModelDesc) -> None:
    layers = list(kb.iter_layers(model))
    paths = [p for p, _, _ in layers]
    assert len(paths) == len(set(paths)), "层路径必须唯一"
    assert "model.conv_spatial" in paths
    assert "model.linear_global" in paths
    assert "model.norm_trunkfinal" in paths
    assert "model.blocks.0.blockstack.0.q_proj" in paths
    assert "model.blocks.0.blockstack.1.ffn_linear_gate" in paths
    assert "model.policy_head.conv2p" in paths
    assert "model.value_head.conv_ownership" in paths
    # v17 >= 15，所以 pass 那三件也在
    assert "model.policy_head.linear_pass_bias" in paths
    assert "model.policy_head.act_pass" in paths
    assert "model.policy_head.linear_pass2" in paths


def test_get_short_info_string_reproduces_the_engine_log_line(model: ModelDesc) -> None:
    """官方日志：`Model name: b10c384h6nbttflrs (nbt transformer, 10545753 params)`"""
    assert model.get_short_info_string() == "nbt transformer, 10545753 params"


# ---------------------------------------------------------------------------
# 参数计数
# ---------------------------------------------------------------------------


def test_parameter_count_matches_the_official_engine_log(model: ModelDesc) -> None:
    assert kb.count_parameters(model) == OFFICIAL_PARAMS


def test_file_float_count_minus_params_equals_bn_running_stats(model: ModelDesc) -> None:
    """ `getNumParameters` 对 BN 只数 scale+bias，**不数 mean+variance**。

    所以"文件里的 float 总数"必然比"官方日志的 params"大出差值
    = 所有 BN 的 `2 * num_channels`（mean + variance 两个数组）。
    """
    file_floats = kb.count_file_floats(model)
    assert file_floats > OFFICIAL_PARAMS

    bns = _all_batch_norms(model)
    real_bns = [b for b in bns if b.num_channels > 0]
    bn_running_stats = sum(2 * b.num_channels for b in real_bns)
    assert file_floats - OFFICIAL_PARAMS == bn_running_stats
    assert bn_running_stats == 11_904


def test_bn_num_parameters_ignores_mean_and_variance(model: ModelDesc) -> None:
    bn = next(b for b in _all_batch_norms(model) if b.num_channels == 384)
    assert bn.num_parameters == 2 * 384  # scale + bias
    assert bn.num_file_floats == 4 * 384  # mean + variance + scale + bias


# ---------------------------------------------------------------------------
# 浮点：逐位搬运，以及被实测确认的那个问题
# ---------------------------------------------------------------------------


def test_float32_via_float64_is_bit_exact() -> None:
    """实测确认（不是假设）：float32 -> float64 -> float32 逐位一致。

    理由是 float64 能精确表示任何 float32（24 位有效位 < 53 位）。这条测试把这个
    事实钉住，包括所有次正规数与边界值。 即便如此，本模块**仍然**不走
    Python float 中转 —— 序列化是 `ascontiguousarray(<f4).tobytes()` 的纯 memcpy。
    """
    rng = np.random.default_rng(20261003)
    values = np.concatenate(
        [
            rng.standard_normal(500_000).astype(np.float32),
            rng.uniform(-1.0, 1.0, 200_000).astype(np.float32),
            np.array(
                [
                    0.0, -0.0, 1.0, -1.0,
                    np.float32(1.18e-38), np.float32(-1.18e-38),  # FLT_MIN
                    1.4e-45, -1.4e-45,                         # 最小次正规数
                    1e-20, 3.4028235e38, -3.4028235e38,         # FLT_MAX
                    np.inf, -np.inf,
                ],
                dtype=np.float32,
            ),
        ]
    )
    round_tripped = values.astype(np.float64).astype(np.float32)
    # NaN/Inf 的位模式在 astype 里保持不变；用位比较最严格
    same_bits = round_tripped.view(np.uint32) == values.view(np.uint32)
    assert bool(same_bits.all()), (
        f"{int((~same_bits).sum())} of {values.size} values changed bits "
        "during float32->float64->float32"
    )


def test_weights_are_carried_verbatim_not_via_python_float(model: ModelDesc) -> None:
    """序列化必须逐位搬运：随便挑几个 conv 权重，比对内存中的 uint32 位模式。"""
    conv = model.trunk.initial_conv
    payload = kb.serialize_model(model)
    assert payload.count(kb.BIN_MARKER) == 328

    # 直接把 initial_conv 的字节找回来
    marker = kb.BIN_MARKER + str(conv.conv_y).encode() + b"\n"
    del marker  # 只是确认不依赖任何奇怪的东西
    arr = np.ascontiguousarray(conv.weights, dtype="<f4")
    assert arr.dtype == np.dtype("<f4")
    assert payload.count(arr.tobytes()[:64]) == 1  # 前 64 字节在文件里只出现一次


def test_subnormal_weights_are_preserved_not_flushed(model: ModelDesc) -> None:
    """ 坑 #2：KataGo 读的时候会把 fp32 次正规数冲成 0（`desc.cpp:89-90`）。

    我们**故意不冲**，否则 round-trip 对不上。这条测试把这个偏差钉在纸面上。
    官方文件里有 444,703 个次正规数（占 4.21%），全在 transformer FFN 权重里。
    """
    n = kb.count_subnormals(model)
    assert n == 444_703
    assert n < kb.count_file_floats(model) * 0.05

    tiny = np.float32(np.finfo(np.float32).tiny)
    ffn = model.trunk.blocks[0].desc.blocks[3].desc.linear1  # type: ignore[union-attr]
    a = np.abs(ffn.weights)
    assert int(np.count_nonzero((a > 0) & (a < tiny))) == 36_288


def test_text_floats_keep_their_verbatim_spelling(model: ModelDesc) -> None:
    """ 坑 #3：文本里的浮点数必须逐字保留。

    BN 的 epsilon 在文件里是字面量 `1e-20`；读成 float32 是 `9.999999682655225e-21`，
    从后者**永远拼不回** `1e-20`。所以 `TextFloat` 只存原文。
    """
    epsilons = {b.epsilon.text for b in _all_batch_norms(model) if b.num_channels > 0}
    assert epsilons == {"1e-20"}

    bn = next(b for b in _all_batch_norms(model) if b.num_channels == 384)
    assert bn.epsilon.text == "1e-20"
    # 反证：如果用 float 反推，得到的字符串是不同的
    assert repr(float(np.float32(1e-20))) != "1e-20"

    payload = kb.serialize_model(model)
    assert payload.count(b"\n1e-20\n") > 0
    assert b"9.999999682655225e-21" not in payload


def test_text_float_helpers() -> None:
    assert TextFloat.of(1e-20).text == "1e-20"  # 与写入端 str() 一致
    assert TextFloat.of(20.0).text == "20.0"
    assert TextFloat.of(0.25).text == "0.25"
    assert TextFloat("1e-20").value == pytest.approx(1e-20, rel=1e-12)
    assert float(TextFloat("0.25")) == 0.25


# ---------------------------------------------------------------------------
# 权重排列：文件序 <-> torch 序
# ---------------------------------------------------------------------------


def test_conv_file_order_is_y_x_ic_oc_and_torch_order_is_oc_ic_y_x() -> None:
    """坑 #4：文件序 `y,x,ic,oc`（`desc.cpp:133-154`），torch 序 `oc,ic,y,x`。

    这里用**手写的 desc.cpp 索引算术**当独立 source of truth，而不是复用被测代码。
    """
    conv_y, conv_x, ic, oc = 3, 1, 2, 4
    file_order = np.arange(conv_y * conv_x * ic * oc, dtype=np.float32)

    desc = ConvDesc("t", conv_y, conv_x, ic, oc, 1, 1, file_order)
    torch_order = desc.to_torch_weight()
    assert torch_order.shape == (oc, ic, conv_y, conv_x)

    oc_stride = conv_y * conv_x * ic
    ic_stride = conv_y * conv_x
    y_stride = conv_x
    x_stride = 1
    idx = 0
    for y in range(conv_y):
        for x in range(conv_x):
            for ichannel in range(ic):
                for ocannel in range(oc):
                    # desc.cpp:150：文件里第 idx 个 float 落到内存的
                    # oc*ocStride + ic*icStride + y*yStride + x*xStride
                    mem_idx = (
                        ocannel * oc_stride
                        + ichannel * ic_stride
                        + y * y_stride
                        + x * x_stride
                    )
                    assert mem_idx == ocannel * oc_stride + ichannel * ic_stride + y * conv_x + x
                    assert torch_order[ocannel, ichannel, y, x] == idx
                    # 而且内存视角下这个 (oc,ic,y,x) 恰好是 torch 数组的 row-major 下标
                    assert torch_order.ravel()[mem_idx] == idx
                    idx += 1
    assert idx == conv_y * conv_x * ic * oc

    # 逆变换必须回到文件序
    back = ConvDesc.from_torch_weight("t", torch_order)
    assert np.array_equal(back.weights, file_order)
    assert (back.conv_y, back.conv_x, back.in_channels, back.out_channels) == (3, 1, 2, 4)


def test_conv_permutation_against_real_model_is_a_bijection(model: ModelDesc) -> None:
    """真实权重上再验一次：文件序 <-> torch 序必须是双射，且元素多重集不变。"""
    conv = model.trunk.initial_conv
    torch_order = conv.to_torch_weight()
    assert torch_order.shape == (
        conv.out_channels,
        conv.in_channels,
        conv.conv_y,
        conv.conv_x,
    )
    assert torch_order.size == conv.weights.size
    assert sorted(torch_order.ravel().tolist()) == sorted(conv.weights.ravel().tolist())

    back = ConvDesc.from_torch_weight(conv.name, torch_order)
    assert np.array_equal(back.weights.view(np.uint32), conv.weights.view(np.uint32))


def test_matmul_file_order_is_ic_oc_and_torch_order_is_oc_ic(model: ModelDesc) -> None:
    """坑 #5：文件序 `ic,oc`（`desc.cpp:463-478`），torch 序 `oc,ic`。"""
    attn = model.trunk.blocks[0].desc.blocks[0].desc  # type: ignore[union-attr]
    q = attn.q_proj
    assert q.in_channels == 192 and q.out_channels == 192

    torch_order = q.to_torch_weight()
    assert torch_order.shape == (q.out_channels, q.in_channels)
    idx = 0
    for ichannel in range(q.in_channels):
        for ocannel in range(q.out_channels):
            assert q.weights[idx] == torch_order[ocannel, ichannel]
            assert ichannel * q.out_channels + ocannel == idx
            idx += 1

    back = MatMulDesc.from_torch_weight(q.name, torch_order)
    assert np.array_equal(back.weights.view(np.uint32), q.weights.view(np.uint32))


# ---------------------------------------------------------------------------
# 合成模型：完全脱离真实权重文件
# ---------------------------------------------------------------------------


def _bn(name: str, c: int, *, has_scale: bool = True, has_bias: bool = True) -> BatchNormDesc:
    """一个 bias-mask 风格的 BN：mean 全 0，variance = 1-1e-20。"""
    eps = TextFloat("1e-20")
    mean = np.zeros(c, dtype=np.float32)
    variance = (np.float32(1.0) - np.float32(1e-20)) * np.ones(c, dtype=np.float32)
    scale = np.linspace(0.5, 1.5, c).astype(np.float32) if has_scale else np.ones(c, dtype=np.float32)
    bias = np.linspace(-1.0, 1.0, c).astype(np.float32) if has_bias else np.zeros(c, dtype=np.float32)
    return BatchNormDesc(name, c, eps, has_scale, has_bias, mean, variance, scale, bias)


def _conv(name: str, ic: int, oc: int, size: int = 1, seed: int = 0) -> ConvDesc:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((oc, ic, size, size)).astype(np.float32)
    return ConvDesc.from_torch_weight(name, w)


def _matmul(name: str, ic: int, oc: int, seed: int = 0) -> MatMulDesc:
    rng = np.random.default_rng(seed)
    return MatMulDesc.from_torch_weight(name, rng.standard_normal((oc, ic)).astype(np.float32))


def _matbias(name: str, c: int, seed: int = 0) -> MatBiasDesc:
    rng = np.random.default_rng(seed)
    return MatBiasDesc(name, c, rng.standard_normal(c).astype(np.float32))


def _activation(name: str, kind: str = "ACTIVATION_MISH") -> ActivationDesc:
    return ActivationDesc(name, kb._NAME_TO_ACTIVATION[kind])


def build_minimal_model(model_version: int = 17) -> ModelDesc:
    """造一个**最小但合法**的模型：1 个 trunk 块、块内 1 个 attention 块、1 个 head。

    覆盖 v17（激活带 kind token、显式 policyOutChannels、pass 三件套、RMSNorm tip）
    和 v10（激活无 kind token、无 post-process 缩放、无 meta 字段、无 pass 三件套、
    policyOutChannels=1、tip 是 BatchNorm）两条完全不同的文件布局分支。
    """
    assert model_version in (10, 17)
    new_format = model_version >= 15
    trunk_c, mid_c = 8, 4
    in_spatial, in_global = 2, 1
    policy_out = 2 if model_version >= 12 else 1
    head_c = 8  # policy / value 的中间宽度

    # -- 一个 transformer attention 块 ---------------------------------
    attn = TransformerAttentionDesc(
        name="model.blocks.0.blockstack.0",
        num_heads=1,
        num_kv_heads=1,
        q_head_dim=4,
        v_head_dim=4,
        use_rope=True,
        learnable_rope=False,
        pre_ln=TransformerRMSNormDesc(
            "model.blocks.0.blockstack.0.norm1", mid_c, TextFloat.of(1e-5),
            np.ones(mid_c, dtype=np.float32),
        ),
        q_proj=_matmul("model.blocks.0.blockstack.0.q_proj", mid_c, 4, seed=1),
        k_proj=_matmul("model.blocks.0.blockstack.0.k_proj", mid_c, 4, seed=2),
        v_proj=_matmul("model.blocks.0.blockstack.0.v_proj", mid_c, 4, seed=3),
        out_proj=_matmul("model.blocks.0.blockstack.0.out_proj", 4, mid_c, seed=4),
        rope_theta_name="model.blocks.0.blockstack.0.rope_theta",
        rope_theta=TextFloat.of(10000.0),
    )

    nested = NestedBottleneckDesc(
        name="model.blocks.0",
        num_blocks=1,
        pre_bn=_bn("model.blocks.0.norm", trunk_c),
        pre_activation=_activation("model.blocks.0.act"),
        pre_conv=_conv("model.blocks.0.conv", trunk_c, mid_c, seed=5),
        blocks=[Block("transformer_attention_block", attn)],
        post_bn=_bn("model.blocks.0.postnorm", mid_c),
        post_activation=_activation("model.blocks.0.postact"),
        post_conv=_conv("model.blocks.0.postconv", mid_c, trunk_c, seed=6),
    )

    # -- trunk -----------------------------------------------------------
    empty_bn = BatchNormDesc(
        "unused_trunk_tip_bn", 0, TextFloat.of(1e-5), False, False,
        np.zeros(0, np.float32), np.zeros(0, np.float32),
        np.zeros(0, np.float32), np.zeros(0, np.float32),
    )
    empty_rms = RMSNormDesc(
        "unused_trunk_tip_rmsnorm", 0, TextFloat.of(1e-5), False, 0,
        np.zeros(0, np.float32), np.zeros(0, np.float32),
    )
    # v>=15 且 trunkNormKind==1 -> tip 是 RMSNorm；v<15 -> tip 恒是 BatchNorm
    tip_is_rmsnorm = new_format
    tip_rms = RMSNormDesc(
        "model.norm_trunkfinal", trunk_c, TextFloat.of(1e-5), False, 0,
        np.ones(trunk_c, dtype=np.float32), np.zeros(trunk_c, dtype=np.float32),
    )
    tip_bn = _bn("model.norm_trunkfinal", trunk_c)
    trunk = TrunkDesc(
        name="trunk",
        num_blocks=1,
        trunk_num_channels=trunk_c,
        mid_num_channels=mid_c,
        regular_num_channels=mid_c,
        dilated_num_channels=2,
        gpool_num_channels=2,
        trunk_norm_kind=kb.TRUNK_NORM_KIND_RMSNORM if tip_is_rmsnorm else kb.TRUNK_NORM_KIND_STANDARD,
        trunk_options=(0, 0, 0, 0, 0) if new_format else (),
        initial_conv=_conv("model.conv_spatial", in_spatial, trunk_c, size=3, seed=7),
        initial_matmul=_matmul("model.linear_global", in_global, trunk_c, seed=8),
        blocks=[Block("nested_bottleneck_block", nested)],
        trunk_tip_activation=_activation("model.act_trunkfinal", "ACTIVATION_RELU"),
        trunk_tip_bn=empty_bn if tip_is_rmsnorm else tip_bn,
        trunk_tip_rmsnorm=tip_rms if tip_is_rmsnorm else empty_rms,
        sgf_metadata_encoder=None,
    )

    # -- policy head -----------------------------------------------------
    policy_kwargs: Dict[str, Any] = dict(
        name="model.policy_head",
        policy_out_channels=policy_out,
        policy_options=(0, 0, 0) if model_version >= 17 else (),
        p1_conv=_conv("model.policy_head.conv1p", trunk_c, head_c, seed=9),
        g1_conv=_conv("model.policy_head.conv1g", trunk_c, head_c, seed=10),
        g1_bn=_bn("model.policy_head.biasg", head_c),
        g1_activation=_activation("model.policy_head.actg"),
        gpool_to_bias_mul=_matmul("model.policy_head.linear_g", head_c * 3, head_c, seed=11),
        p1_bn=_bn("model.policy_head.bias2", head_c),
        p1_activation=_activation("model.policy_head.act2"),
        p2_conv=_conv("model.policy_head.conv2p", head_c, policy_out, seed=12),
        # v>=15: out == gpoolToPassBias.numChannels == p1Conv.outChannels
        # v<15 : out == policyOutChannels（`desc.cpp:2154-2155`）
        gpool_to_pass_mul=_matmul(
            "model.policy_head.linear_pass",
            head_c * 3,
            head_c if new_format else policy_out,
            seed=13,
        ),
    )
    if new_format:
        policy_kwargs.update(
            gpool_to_pass_bias=_matbias("model.policy_head.linear_pass_bias", head_c, seed=14),
            pass_activation=_activation("model.policy_head.act_pass", "ACTIVATION_SILU"),
            gpool_to_pass_mul2=_matmul("model.policy_head.linear_pass2", head_c, policy_out, seed=15),
        )
    else:
        policy_kwargs.update(
            gpool_to_pass_bias=MatBiasDesc("unused_gpool_to_pass_bias", 0, np.zeros(0, np.float32)),
            pass_activation=ActivationDesc("unused_pass_activation", kb.ACTIVATION_RELU),
            gpool_to_pass_mul2=MatMulDesc("unused_gpool_to_pass_mul2", 0, 0, np.zeros(0, np.float32)),
        )
    policy = PolicyHeadDesc(**policy_kwargs)

    # -- value head ------------------------------------------------------
    misc_c = ValueHeadDesc.expected_score_value_channels(model_version)
    value = ValueHeadDesc(
        name="model.value_head",
        value_options=(0, 0, 0) if model_version >= 17 else (),
        v1_conv=_conv("model.value_head.conv1", trunk_c, head_c, seed=16),
        v1_bn=_bn("model.value_head.bias1", head_c),
        v1_activation=_activation("model.value_head.act1"),
        v2_mul=_matmul("model.value_head.linear2", head_c * 3, 4, seed=17),
        v2_bias=_matbias("model.value_head.bias2", 4, seed=18),
        v2_activation=_activation("model.value_head.act2"),
        v3_mul=_matmul("model.value_head.linear_valuehead", 4, 3, seed=19),
        v3_bias=_matbias("model.value_head.bias_valuehead", 3, seed=20),
        sv3_mul=_matmul("model.value_head.linear_miscvaluehead", 4, misc_c, seed=21),
        sv3_bias=_matbias("model.value_head.bias_miscvaluehead", misc_c, seed=22),
        v_ownership_conv=_conv("model.value_head.conv_ownership", head_c, 1, seed=23),
    )

    return ModelDesc(
        name="tinytest",
        sha256="",
        model_version=model_version,
        num_input_channels=in_spatial,
        num_input_global_channels=in_global,
        td_score_multiplier=TextFloat.of(20.0),
        score_mean_multiplier=TextFloat.of(20.0),
        score_stdev_multiplier=TextFloat.of(20.0),
        lead_multiplier=TextFloat.of(20.0),
        variance_time_multiplier=TextFloat.of(40.0),
        shortterm_value_error_multiplier=TextFloat.of(0.25),
        shortterm_score_error_multiplier=TextFloat.of(30.0),
        meta_encoder_version=0,
        prefer_pass_alive_under_suicide_rules=False,
        prefer_exclude_territory_adjacent_to_atari=False,
        model_options=(0, 0, 0, 0, 0) if new_format else (),
        trunk=trunk,
        policy_head=policy,
        value_head=value,
    )


@pytest.mark.parametrize("version", [17, 10])
def test_synthetic_minimal_model_serialize_parse_serialize_is_stable(version: int) -> None:
    """这条**不依赖真实权重文件**：合成 -> 序列化 -> 解析 -> 再序列化，两次输出必须一致。"""
    original = build_minimal_model(version)

    first = kb.serialize_model(original)
    parsed = kb.parse_model(first)
    second = kb.serialize_model(parsed)

    assert second == first
    # 再来一轮，确认不是"恰好收敛"
    assert kb.serialize_model(kb.parse_model(second)) == first


@pytest.mark.parametrize("version", [17, 10])
def test_synthetic_model_survives_a_file_roundtrip_through_gzip(tmp_path, version: int) -> None:
    original = build_minimal_model(version)
    path = tmp_path / f"tiny-v{version}.bin.gz"
    kb.save_model_file(original, str(path))

    loaded = kb.load_model_file(str(path))
    assert kb.serialize_model(loaded) == kb.serialize_model(original)
    assert loaded.get_short_info_string().endswith("params")
    assert path.read_bytes()[:2] == b"\x1f\x8b"  # 真的是 gzip
    assert kb.maybe_gunzip(path.read_bytes()) == kb.serialize_model(original)


def test_synthetic_model_counts_are_self_consistent() -> None:
    original = build_minimal_model(17)
    emitted = []
    original_floats = kb._Writer.floats

    def spy(self: kb._Writer, arr: np.ndarray) -> None:
        emitted.append(int(np.ascontiguousarray(arr, dtype="<f4").size))
        return original_floats(self, arr)

    kb._Writer.floats = spy  # type: ignore[method-assign]
    try:
        payload = kb.serialize_model(original)
    finally:
        kb._Writer.floats = original_floats  # type: ignore[method-assign]

    assert sum(emitted) == kb.count_file_floats(original)
    assert payload.count(kb.BIN_MARKER) == len(emitted)
    # has_scale=False 的 BN 不该占 float
    bns = [b for b in _all_batch_norms(original) if b.num_channels > 0]
    assert all(b.has_scale for b in bns)
    assert kb.count_parameters(original) > 0
    assert kb.count_file_floats(original) > kb.count_parameters(original)


def test_bn_without_scale_does_not_emit_the_scale_array() -> None:
    """把一个 BN 的 `has_scale` 翻成 False，文件必须正好短 `num_channels * 4` 字节。"""
    original = build_minimal_model(17)
    with_scale = kb.serialize_model(original)

    target = original.trunk.blocks[0].desc.pre_bn  # type: ignore[union-attr]
    c = target.num_channels
    target.has_scale = False
    without_scale = kb.serialize_model(original)

    # 少掉的是：`@BIN@`(5) + 4*c 字节 float + 一个块尾 `\n`(1)
    assert len(with_scale) - len(without_scale) == 5 + c * 4 + 1

    # 解析回来的仍然是 has_scale=False，且 scale 是全 1（不是从文件读的）
    reparsed = kb.parse_model(without_scale)
    rt: BatchNormDesc = reparsed.trunk.blocks[0].desc.pre_bn  # type: ignore[union-attr]
    assert rt.has_scale is False
    assert rt.scale.shape == (c,)
    assert bool((rt.scale == 1.0).all())
    assert kb.serialize_model(reparsed) == without_scale


# ---------------------------------------------------------------------------
# 恶意 / 畸形输入：必须明确抛错，不许静默接受
# ---------------------------------------------------------------------------


def test_truncated_file_raises(model_bytes: bytes) -> None:
    """在多个不同位置砍断，都必须抛错，不许静默接受。

     唯独砍掉**最后一个**字节（那个尾随 `\\n`）是合法的 —— 那是坑 #1，
    单独由 `test_trailing_newline_is_optional_for_parsing_...` 覆盖。
    """
    raw = kb.maybe_gunzip(model_bytes)
    n = len(raw)
    cuts = [
        int(n * 0.999), int(n * 0.9), int(n * 0.5), int(n * 0.1),
        int(n * 0.01), int(n * 0.001),
        30, 60, 120,  # 头几个 token
    ]
    assert n - 1 not in cuts
    for cut in cuts:
        assert 0 < cut < n - 1
        with pytest.raises(kb.KatagoBinError):
            kb.parse_model(raw[:cut])


def test_truncated_in_the_middle_of_a_float_block_raises(model_bytes: bytes) -> None:
    """砍在某个 `@BIN@` 块中间：必须是 TruncatedModelError，不是静默接受。"""
    raw = kb.maybe_gunzip(model_bytes)
    first = raw.find(kb.BIN_MARKER) + len(kb.BIN_MARKER) + 17  # 落在 float 中间
    with pytest.raises(kb.TruncatedModelError):
        kb.parse_model(raw[:first])


def test_header_only_file_raises() -> None:
    """只剩文件头的碎片。"""
    with pytest.raises(kb.TruncatedModelError):
        kb.parse_model(b"tinytest\n17\n2\n1\n")


def test_model_name_longer_than_96_chars_raises() -> None:
    """`ModelDesc::checkNameValid`（`desc.cpp:2433`）：模型名最长 96 字符。"""
    assert kb.MAX_MODEL_NAME_LEN == 96
    long_name = "a" * 97
    payload = f"{long_name}\n17\n2\n1\n".encode()
    with pytest.raises(kb.ModelFormatError, match="too long"):
        kb.parse_model(payload)

    # 96 恰好合法（字符集也对）
    kb.check_name_valid("a" * 96)
    with pytest.raises(kb.ModelFormatError, match="too long"):
        kb.check_name_valid("a" * 97)


def test_model_name_charset_is_enforced() -> None:
    with pytest.raises(kb.ModelFormatError, match="alphanumeric"):
        kb.check_name_valid("has space")
    with pytest.raises(kb.ModelFormatError, match="alphanumeric"):
        kb.check_name_valid("dots.not.allowed")
    with pytest.raises(kb.ModelFormatError, match="empty"):
        kb.check_name_valid("")
    kb.check_name_valid("b10c384h6nbttflrs")
    kb.check_name_valid("A-Za-z_0-9")


def test_layer_names_have_no_length_limit(model: ModelDesc) -> None:
    """ 96 字符限制**只作用于模型名**。层名没有长度限制 —— 别把两者搞混。"""
    original = build_minimal_model(17)
    long_layer = "L" * 500
    original.policy_head.p2_conv.name = long_layer
    payload = kb.serialize_model(original)
    assert kb.parse_model(payload).policy_head.p2_conv.name == long_layer
    assert kb.serialize_model(kb.parse_model(payload)) == payload


@pytest.mark.parametrize("version", [-1, 0, 2, 18, 99])
def test_illegal_model_version_raises(version: int) -> None:
    payload = f"tinytest\n{version}\n2\n1\n".encode()
    with pytest.raises(kb.UnsupportedModelVersionError):
        kb.parse_model(payload)


def test_model_version_3_and_17_are_the_boundaries() -> None:
    """`modelversion.h:11` / `:7`：3..17。"""
    for v in (kb.OLDEST_MODEL_VERSION, kb.LATEST_MODEL_VERSION):
        assert kb._inputs_version(v) in (3, 7)
    with pytest.raises(kb.UnsupportedModelVersionError):
        kb._inputs_version(kb.LATEST_MODEL_VERSION + 1)
    with pytest.raises(kb.UnsupportedModelVersionError):
        kb._inputs_version(kb.OLDEST_MODEL_VERSION - 1)


def test_unknown_block_tag_raises(model_bytes: bytes) -> None:
    """块类型由字符串 tag 决定，未知即抛错（`desc.cpp:1558-1559`）。"""
    raw = bytearray(kb.maybe_gunzip(model_bytes))
    tag = b"nested_bottleneck_block"
    idx = raw.find(tag)
    assert idx > 0, "官方文件里应该至少有一个 nested_bottleneck_block"
    raw[idx : idx + len(tag)] = b"nested_bottleneck_blocK"  # 等长替换，只改最后一个字节
    assert len(raw) == RAW_LEN

    with pytest.raises(kb.UnsupportedBlockKindError, match="unknown block kind"):
        kb.parse_model(bytes(raw))


def test_unknown_block_tag_on_a_synthetic_model_raises() -> None:
    payload = bytearray(kb.serialize_model(build_minimal_model(17)))
    idx = payload.find(b"transformer_attention_block")
    assert idx > 0
    payload[idx : idx + len(b"transformer_attention_block")] = b"transformer_attention_blocK"
    with pytest.raises(kb.UnsupportedBlockKindError):
        kb.parse_model(bytes(payload))


def test_unknown_activation_kind_raises() -> None:
    payload = bytearray(kb.serialize_model(build_minimal_model(17)))
    idx = payload.find(b"ACTIVATION_MISH")
    assert idx > 0
    payload[idx : idx + len(b"ACTIVATION_MISH")] = b"ACTIVATION_GELU"
    with pytest.raises(kb.ModelFormatError, match="unknown activation"):
        kb.parse_model(bytes(payload))


def test_non_zero_unused_slot_raises(model_bytes: bytes) -> None:
    """5 个 unused 槽必须真的是 0 —— 非 0 表示"我用了你不知道的特性"（`desc.cpp:2584-2593`）。"""
    raw = bytearray(kb.maybe_gunzip(model_bytes))
    # model header: name \n version \n inC \n inGC \n 7 doubles \n metaEnc \n preferA \n preferB \n 5 unused
    head = kb.maybe_gunzip(model_bytes)[:200].split(b"\n")
    assert head[0] == b"b10c384h6nbttflrs"
    assert head[1] == b"17"
    # index: 0 name, 1 ver, 2 inC, 3 inGC, 4..10 doubles, 11 metaEnc, 12 preferA, 13 preferB,
    #        14..18 unused  <- 把第 18 个（最后一个 unused）改成 1
    off = sum(len(tok) + 1 for tok in head[:18])
    assert raw[off : off + 1] == b"0"
    raw[off : off + 1] = b"7"
    with pytest.raises(kb.ModelFormatError, match="unknown/unsupported model option H"):
        kb.parse_model(bytes(raw))


def test_non_zero_prefer_flag_value_raises(model_bytes: bytes) -> None:
    """`desc.cpp:2557-2574`：这两个标志只接受 0 / 1，其它值直接抛错。"""
    raw = bytearray(kb.maybe_gunzip(model_bytes))
    head = kb.maybe_gunzip(model_bytes)[:200].split(b"\n")
    off = sum(len(tok) + 1 for tok in head[:12])  # 第 12 个 token = preferPassAlive
    assert raw[off : off + 1] == b"0"
    raw[off : off + 1] = b"2"
    with pytest.raises(kb.ModelFormatError, match="unexpected value"):
        kb.parse_model(bytes(raw))


def test_nan_weight_raises() -> None:
    """`CHECKFINITE`（`desc.cpp:21-26`）：NaN / Inf 权重必须被拒。"""
    original = build_minimal_model(17)
    target = original.trunk.initial_conv.weights
    target[0] = np.float32("nan")
    with pytest.raises(kb.ModelFormatError, match="NaN or infinite"):
        kb.parse_model(kb.serialize_model(original))

    target[0] = np.float32("inf")
    with pytest.raises(kb.ModelFormatError, match="NaN or infinite"):
        kb.parse_model(kb.serialize_model(original))


def test_non_binary_floats_rejected_as_txt_model(model_bytes: bytes) -> None:
    """`desc.cpp:59`：把 .txt 权重文件当 .bin 读会在这里炸。我们也要炸。

    构造一个"头是对的、但权重用文本浮点"的片段：走到第一个 float 块时找不到 `@`。
    """
    text_weights = (
        b"tinytest\n17\n2\n1\n"
        b"20.0\n20.0\n20.0\n20.0\n40.0\n0.25\n30.0\n"
        b"0\n0\n0\n0\n0\n0\n0\n0\n"  # metaEnc + 2 prefers + 5 unused
        b"trunk\n1\n8\n4\n4\n2\n2\n1\n0\n0\n0\n0\n0\n"
        b"model.conv_spatial\n3\n3\n2\n8\n1\n1\n"
        b"0.5\n0.25\n0.125\n"
    )
    with pytest.raises(kb.TruncatedModelError, match="expected '@BIN@'"):
        kb.parse_model(text_weights)


def test_gzip_magic_is_detected(model_bytes: bytes) -> None:
    """`.gz` 自动解压，`.bin` 原样透传。"""
    assert model_bytes[:2] == b"\x1f\x8b"
    raw = kb.maybe_gunzip(model_bytes)
    assert raw[:2] != b"\x1f\x8b"
    assert kb.maybe_gunzip(raw) is raw  # 没有 magic 就不动

    # 截断的 gzip 会抛，而不是静默返回半截数据
    with pytest.raises((EOFError, OSError, gzip.BadGzipFile)):
        kb.maybe_gunzip(model_bytes[:20])


def test_load_from_plain_bin_file(tmp_path, model_bytes: bytes) -> None:
    """`.bin`（不压缩）也必须能直接读。"""
    path = tmp_path / "model.bin"
    path.write_bytes(kb.maybe_gunzip(model_bytes))
    loaded = kb.load_model_file(str(path))
    assert loaded.get_short_info_string() == "nbt transformer, 10545753 params"
    assert kb.serialize_model(loaded) == kb.maybe_gunzip(model_bytes)


def test_bad_at_marker_raises(model_bytes: bytes) -> None:
    """`@BIN@` 之后必须紧跟 `BIN@`（`desc.cpp:66-67`）。"""
    raw = bytearray(kb.maybe_gunzip(model_bytes))
    idx = raw.find(kb.BIN_MARKER)
    assert idx > 0
    raw[idx : idx + 5] = b"@BAN@"
    with pytest.raises(kb.ModelFormatError, match="binary float block"):
        kb.parse_model(bytes(raw))


def test_too_many_chars_before_at_marker_raises(model_bytes: bytes) -> None:
    """`desc.cpp:58`：`@` 之前最多 100 个字符。"""
    raw = bytearray(kb.maybe_gunzip(model_bytes))
    idx = raw.find(kb.BIN_MARKER)
    assert idx > 0
    raw[idx - 1 : idx - 1] = b" " * kb.MAX_CHARS_BEFORE_AT + b" "
    with pytest.raises(kb.KatagoBinError):
        kb.parse_model(bytes(raw))


def test_even_conv_filter_size_raises() -> None:
    """`desc.cpp:130-131`：卷积核尺寸必须是奇数。"""
    original = build_minimal_model(17)
    original.trunk.initial_conv.conv_y = 2
    with pytest.raises(ModelFormatError_ := kb.ModelFormatError):
        kb.parse_model(kb.serialize_model(original))
    del ModelFormatError_


def test_shape_contradiction_raises() -> None:
    """通道数互相矛盾必须被抓住，不许静默接受。

    挑 `policyOutChannels` 是因为它是**纯文本 int**，改了它不会移动任何 float 块的
    位置，所以报错一定来自那条交叉检查本身，而不是"后面全乱了"。
    """
    original = build_minimal_model(17)
    original.policy_head.policy_out_channels = 4  # p2Conv.outChannels 仍然是 2
    with pytest.raises(kb.ModelFormatError, match="p2Conv.outChannels"):
        kb.parse_model(kb.serialize_model(original))

    original = build_minimal_model(17)
    original.num_input_channels = 3  # trunk.initialConv.inChannels 仍然是 2
    with pytest.raises(kb.ModelFormatError, match="numInputChannels"):
        kb.parse_model(kb.serialize_model(original))


def test_grouped_spatial_rmsnorm_is_rejected() -> None:
    """`desc.cpp:1086-1087`：cgroupSize != 0 未实现，必须报错而不是当成 0。"""
    original = build_minimal_model(17)
    original.trunk.trunk_tip_rmsnorm.cgroup_size = 4
    with pytest.raises(kb.ModelFormatError, match="cgroupSize"):
        kb.parse_model(kb.serialize_model(original))


def test_nested_bottleneck_num_blocks_mismatch_raises() -> None:
    original = build_minimal_model(17)
    original.trunk.blocks[0].desc.num_blocks = 3  # 实际只有 1 个子块
    with pytest.raises(kb.ModelFormatError, match="numBlocks"):
        kb.parse_model(kb.serialize_model(original))


def test_rope_requires_even_q_head_dim() -> None:
    """`desc.cpp:1201-1202`：RoPE 旋转交错通道对，qHeadDim 必须为偶数。"""
    original = build_minimal_model(17)
    attn = original.trunk.blocks[0].desc.blocks[0].desc  # type: ignore[union-attr]
    attn.q_head_dim = 3
    with pytest.raises(kb.ModelFormatError, match="must be even"):
        kb.parse_model(kb.serialize_model(original))


def test_malformed_integer_token_raises() -> None:
    with pytest.raises(kb.ModelFormatError, match="expected an integer"):
        kb.parse_model(b"tinytest\nnotanumber\n2\n1\n")


def test_malformed_float_token_raises() -> None:
    with pytest.raises(kb.ModelFormatError, match="floating point"):
        kb.parse_model(b"tinytest\n17\n2\n1\nnotafloat\n")


def test_empty_input_raises() -> None:
    with pytest.raises(kb.TruncatedModelError):
        kb.parse_model(b"")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_roundtrip_prints_identical(model_bytes: bytes, capsys, tmp_path) -> None:
    out = tmp_path / "reserialized.bin"
    rc = kb.main(["roundtrip", MODEL_PATH, "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "字节完全一致" in printed
    assert "byte-identical" in printed
    assert str(RAW_LEN) in printed
    assert out.read_bytes() == kb.maybe_gunzip(model_bytes)


def test_cli_inspect_prints_the_inventory(model_bytes: bytes, capsys) -> None:
    rc = kb.main(["inspect", MODEL_PATH])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "model.conv_spatial" in printed
    assert "nested_bottleneck_block" in printed
    assert "transformer_attention_block" in printed
    assert "nbt transformer, 10545753 params" in printed
    assert "trunkNumChannels    : 384" in printed


# ---------------------------------------------------------------------------
# 显式的"没有实现"清单
# ---------------------------------------------------------------------------


def test_every_block_tag_desc_cpp_can_emit_is_implemented() -> None:
    """`parseResidualBlockStack`（`desc.cpp:1460-1557`）能分派的 tag 我们必须都能写。"""
    assert set(kb.BLOCK_KIND_TAGS) == {
        "ordinary_block",
        "gpool_block",
        "nested_bottleneck_block",
        "transformer_attention_block",
        "transformer_ffn_block",
    }
    assert set(kb.BLOCK_KIND_TAGS.values()) == {
        "ResidualBlockDesc",
        "GlobalPoolingResidualBlockDesc",
        "NestedBottleneckDesc",
        "TransformerAttentionDesc",
        "TransformerFFNDesc",
    }


def test_ordinary_and_gpool_blocks_roundtrip_in_a_synthetic_model() -> None:
    """官方 b10c384nbt 用不到 ordinary / gpool，但格式支持，我们也要能往返。"""
    trunk_c, mid_c = 8, 8
    ordinary = ResidualBlockDesc(
        name="model.blocks.0",
        pre_bn=_bn("model.blocks.0.normactconv1.norm", trunk_c),
        pre_activation=_activation("model.blocks.0.normactconv1.act"),
        regular_conv=_conv("model.blocks.0.normactconv1.conv", trunk_c, mid_c, size=3, seed=31),
        mid_bn=_bn("model.blocks.0.normactconv2.norm", mid_c),
        mid_activation=_activation("model.blocks.0.normactconv2.act"),
        final_conv=_conv("model.blocks.0.normactconv2.conv", mid_c, trunk_c, size=3, seed=32),
    )
    gpool = GlobalPoolingResidualBlockDesc(
        name="model.blocks.1",
        pre_bn=_bn("model.blocks.1.normactconv1.norm", trunk_c),
        pre_activation=_activation("model.blocks.1.normactconv1.act"),
        regular_conv=_conv("model.blocks.1.normactconv1.conv", trunk_c, mid_c, size=3, seed=33),
        gpool_conv=_conv("model.blocks.1.normactconv1.convpool.conv1g", trunk_c, 4, seed=34),
        gpool_bn=_bn("model.blocks.1.normactconv1.convpool.normg", 4),
        gpool_activation=_activation("model.blocks.1.normactconv1.convpool.actg"),
        gpool_to_bias_mul=_matmul("model.blocks.1.normactconv1.convpool.linear_g", 12, mid_c, seed=35),
        mid_bn=_bn("model.blocks.1.normactconv2.norm", mid_c),
        mid_activation=_activation("model.blocks.1.normactconv2.act"),
        final_conv=_conv("model.blocks.1.normactconv2.conv", mid_c, trunk_c, size=3, seed=36),
    )

    base = build_minimal_model(17)
    base.trunk.blocks = [
        Block("ordinary_block", ordinary),
        Block("gpool_block", gpool),
    ]
    base.trunk.num_blocks = 2

    payload = kb.serialize_model(base)
    parsed = kb.parse_model(payload)
    assert [b.kind for b in parsed.trunk.blocks] == ["ordinary_block", "gpool_block"]
    assert kb.serialize_model(parsed) == payload


def test_sgf_metadata_encoder_roundtrips_when_present() -> None:
    """`metaEncoderVersion == 1` 会插入一整个 192 输入的 MLP（v>=15）。"""
    base = build_minimal_model(17)
    base.meta_encoder_version = 1
    enc = SGFMetadataEncoderDesc(
        name="model.sgf_metadata_encoder",
        num_input_meta_channels=192,
        mul1=_matmul("model.sgf_metadata_encoder.mul1", 192, 16, seed=41),
        bias1=_matbias("model.sgf_metadata_encoder.bias1", 16, seed=42),
        act1=_activation("model.sgf_metadata_encoder.act1", "ACTIVATION_SILU"),
        mul2=_matmul("model.sgf_metadata_encoder.mul2", 16, 16, seed=43),
        bias2=_matbias("model.sgf_metadata_encoder.bias2", 16, seed=44),
        act2=_activation("model.sgf_metadata_encoder.act2", "ACTIVATION_RELU"),
        mul3=_matmul("model.sgf_metadata_encoder.mul3", 16, base.trunk.trunk_num_channels, seed=45),
    )
    base.trunk.sgf_metadata_encoder = enc

    payload = kb.serialize_model(base)
    parsed = kb.parse_model(payload)
    assert parsed.num_input_meta_channels == 192
    assert parsed.trunk.sgf_metadata_encoder is not None
    assert parsed.trunk.sgf_metadata_encoder.mul1.in_channels == 192
    assert kb.serialize_model(parsed) == payload


def _build_model_with_ffn(use_swiglu: bool, ffn_channels: int = 8) -> ModelDesc:
    """把 nested bottleneck 里唯一的子块换成一个 transformer FFN 块。"""
    base = build_minimal_model(17)
    mid_c = base.trunk.mid_num_channels
    ffn = TransformerFFNDesc(
        name="model.blocks.0.blockstack.0",
        num_channels=mid_c,
        ffn_channels=ffn_channels,
        use_swiglu=use_swiglu,
        pre_ln=TransformerRMSNormDesc(
            "model.blocks.0.blockstack.0.norm", mid_c, TextFloat.of(1e-5),
            np.ones(mid_c, dtype=np.float32),
        ),
        linear1=_matmul("model.blocks.0.blockstack.0.ffn_linear1", mid_c, ffn_channels, seed=51),
        linear2=_matmul("model.blocks.0.blockstack.0.ffn_linear2", ffn_channels, mid_c, seed=52),
        linear_gate=(
            _matmul("model.blocks.0.blockstack.0.ffn_linear_gate", mid_c, ffn_channels, seed=53)
            if use_swiglu
            else MatMulDesc("unused_linear_gate", 0, 0, np.zeros(0, np.float32))
        ),
    )
    base.trunk.blocks[0].desc.blocks = [Block("transformer_ffn_block", ffn)]  # type: ignore[union-attr]
    return base


def test_ffn_without_swiglu_does_not_emit_the_gate_matrix() -> None:
    """`desc.cpp:1391-1393`：`useSwiGLU == 0` 时 `linearGate` **整个层都不在文件里** ——
    不只是 float 块，name token 和两个通道数 int 也一起没有。"""
    mid_c, ffn_c = 4, 8
    with_gate = kb.serialize_model(_build_model_with_ffn(True, ffn_c))
    without_gate = kb.serialize_model(_build_model_with_ffn(False, ffn_c))

    gate_name = "model.blocks.0.blockstack.0.ffn_linear_gate"
    dropped = (
        len(gate_name) + 1      # name token + 它的 \n
        + len(str(mid_c)) + 1   # inChannels
        + len(str(ffn_c)) + 1   # outChannels
        + len(kb.BIN_MARKER)
        + 4 * mid_c * ffn_c     # 裸 float32
        + 1                     # 块尾 \n
    )
    assert dropped == 182
    assert len(with_gate) - len(without_gate) == dropped
    assert gate_name.encode() not in without_gate
    assert gate_name.encode() in with_gate

    reparsed = kb.parse_model(without_gate).trunk.blocks[0].desc.blocks[0].desc  # type: ignore[union-attr]
    assert reparsed.use_swiglu is False
    assert reparsed.linear_gate.weights.size == 0
    assert reparsed.linear_gate.num_parameters == 0
    assert kb.serialize_model(kb.parse_model(without_gate)) == without_gate

    # 有 gate 的那一支必须真的能往返
    re_gated = kb.parse_model(with_gate).trunk.blocks[0].desc.blocks[0].desc  # type: ignore[union-attr]
    assert re_gated.use_swiglu is True
    assert re_gated.linear_gate.weights.size == mid_c * ffn_c
    assert kb.serialize_model(kb.parse_model(with_gate)) == with_gate


def test_learnable_rope_freqs_roundtrip(model: ModelDesc) -> None:
    attn: TransformerAttentionDesc = model.trunk.blocks[0].desc.blocks[0].desc  # type: ignore[union-attr]
    assert attn.learnable_rope is True
    assert attn.rope_freqs_name == "model.blocks.0.blockstack.0.rope_freqs"
    assert attn.rope_num_kv_heads == 6
    assert attn.rope_num_pairs == 16
    assert attn.rope_freqs.size == 6 * 16 * 2
    # 固定 rope 的 theta 那一支在这个文件里不出现
    assert attn.rope_theta_name is None


def test_fixed_rope_theta_is_written_verbatim() -> None:
    """固定 RoPE：写的是 `rope_theta` 这个**被丢弃的 name token** + theta 值。
    合成模型里 name 用 `TextFloat` 之外的普通 token，验证它原样往返。"""
    base = _build_model_with_ffn(True)
    attn = TransformerAttentionDesc(
        name="model.blocks.0.blockstack.0",
        num_heads=1, num_kv_heads=1, q_head_dim=4, v_head_dim=4,
        use_rope=True, learnable_rope=False,
        pre_ln=TransformerRMSNormDesc("n", 4, TextFloat.of(1e-5), np.ones(4, np.float32)),
        q_proj=_matmul("q", 4, 4, seed=61),
        k_proj=_matmul("k", 4, 4, seed=62),
        v_proj=_matmul("v", 4, 4, seed=63),
        out_proj=_matmul("o", 4, 4, seed=64),
        rope_theta_name="model.blocks.0.blockstack.0.rope_theta",
        rope_theta=TextFloat.of(10000.0),
    )
    base.trunk.blocks[0].desc.blocks = [Block("transformer_attention_block", attn)]  # type: ignore[union-attr]
    payload = kb.serialize_model(base)
    assert b"model.blocks.0.blockstack.0.rope_theta\n10000.0\n" in payload

    parsed = kb.parse_model(payload).trunk.blocks[0].desc.blocks[0].desc  # type: ignore[union-attr]
    assert parsed.learnable_rope is False
    assert parsed.rope_theta is not None
    assert parsed.rope_theta.text == "10000.0"
    assert parsed.rope_freqs.size == 0
    assert kb.serialize_model(kb.parse_model(payload)) == payload


def test_learnable_rope_theta_is_absent_from_the_file(model: ModelDesc) -> None:
    """learnable RoPE 那一支**不写** theta；固定那一支**不写** freqs。二者互斥。"""
    attn: TransformerAttentionDesc = model.trunk.blocks[0].desc.blocks[0].desc  # type: ignore[union-attr]
    assert attn.rope_theta is None
    assert attn.rope_freqs.size == 192
    payload = kb.serialize_model(model)
    idx_freqs = payload.find(b"model.blocks.0.blockstack.0.rope_freqs\n")
    assert idx_freqs > 0
    assert payload.find(b"rope_theta\n", idx_freqs) == -1


def test_synthetic_v17_and_v10_files_have_different_layouts() -> None:
    """v10 与 v17 的字节布局确实不同：激活 kind token、post-process 缩放、pass 三件套、
    trunkNormKind、policyOutChannels、tip norm 类型。"""
    v17 = kb.serialize_model(build_minimal_model(17))
    v10 = kb.serialize_model(build_minimal_model(10))

    # v >= 11 才有激活 kind token
    assert b"ACTIVATION_MISH\n" in v17
    assert b"ACTIVATION_MISH\n" not in v10
    # v >= 15 才有 pass 的 bias / act / linear_pass2
    assert b"model.policy_head.linear_pass_bias\n" in v17
    assert b"model.policy_head.linear_pass_bias\n" not in v10
    assert b"model.policy_head.act_pass\n" in v17
    assert b"model.policy_head.act_pass\n" not in v10
    # v >= 13 才有 7 个 post-process 缩放（token 4..10）
    assert v17.split(b"\n")[0:11] == [
        b"tinytest", b"17", b"2", b"1",
        b"20.0", b"20.0", b"20.0", b"20.0", b"40.0", b"0.25", b"30.0",
    ]
    # v10 的第 5 个 token 直接就是 trunk
    assert v10.split(b"\n")[0:5] == [b"tinytest", b"10", b"2", b"1", b"trunk"]
    # v >= 15 才有 trunkNormKind
    assert b"\n1\n0\n0\n0\n0\n0\nmodel.conv_spatial\n" in v17
    assert b"\n1\n0\n0\n0\n0\n0\nmodel.conv_spatial\n" not in v10
    assert len(v17) != len(v10)

    # 语义侧：policyOutChannels 与 sv3 宽度都随版本变
    assert kb.parse_model(v17).policy_head.policy_out_channels == 2
    assert kb.parse_model(v10).policy_head.policy_out_channels == 1
    assert kb.parse_model(v17).num_value_channels == 3
    assert kb.parse_model(v10).num_value_channels == 3


def test_rope_disabled_attention_writes_neither() -> None:
    base = _build_model_with_ffn(True)
    # 无 rope 时 qHeadDim 可以是奇数（`desc.cpp:1201` 的守卫只在 useRope 时生效）
    attn = TransformerAttentionDesc(
        name="model.blocks.0.blockstack.0",
        num_heads=1, num_kv_heads=1, q_head_dim=3, v_head_dim=4,
        use_rope=False, learnable_rope=False,
        pre_ln=TransformerRMSNormDesc("n", 4, TextFloat.of(1e-5), np.ones(4, np.float32)),
        q_proj=_matmul("q", 4, 3, seed=71),
        k_proj=_matmul("k", 4, 3, seed=72),
        v_proj=_matmul("v", 4, 4, seed=73),
        out_proj=_matmul("o", 4, 4, seed=74),
    )
    base.trunk.blocks[0].desc.blocks = [Block("transformer_attention_block", attn)]  # type: ignore[union-attr]
    payload = kb.serialize_model(base)
    assert b"rope_theta" not in payload
    assert b"rope_freqs" not in payload
    reparsed = kb.parse_model(payload).trunk.blocks[0].desc.blocks[0].desc  # type: ignore[union-attr]
    assert reparsed.use_rope is False
    assert reparsed.rope_freqs.size == 0
    assert kb.serialize_model(kb.parse_model(payload)) == payload


def test_unsupported_transformer_features_have_no_file_representation() -> None:
    """ exporter 侧 assert 掉的那些 transformer 特性，在 `.bin` 里**根本没有表示**：

    `export_model_pytorch.py:565-576` assert 了 `use_qk_norm` / `use_gab` / `use_tab` /
    `inline_registers` / `num_rw_registers`；`:606-609` assert 了 FFN 的
    `use_depthwise_conv` / `inline_registers` / `num_rw_registers`。

    官方 `desc.cpp` 里也**没有**对应的 reader。所以我们这边不需要、也无法实现它们 ——
    任何依赖它们的模型在导出阶段就会失败。这条测试把这个事实钉住。
    """
    source = os.path.join(REPO, "tmp", "coding", "kg", "python", "export_model_pytorch.py")
    if not os.path.exists(source):  # pragma: no cover
        pytest.skip("官方 export_model_pytorch.py 不在 tmp/coding/kg 下")

    with open(source, encoding="utf-8") as f:
        text = f.read()
    for flag in (
        "use_qk_norm",
        "use_gab",
        "use_tab",
        "inline_registers",
        "num_rw_registers",
        "use_depthwise_conv",
    ):
        assert f"getattr(block, '{flag}'" in text or f"block.{flag}" in text, (
            f"{flag} 在官方 exporter 里应该被 assert 掉；"
            "如果官方改了，说明 .bin 格式可能新增了字段，需要重新核对 desc.cpp"
        )

    # 反面：我们的 IR 里没有任何字段承载这些特性
    attn_fields = set(TransformerAttentionDesc.__dataclass_fields__)
    for flag in ("use_qk_norm", "use_gab", "use_tab", "inline_registers", "num_rw_registers"):
        assert flag not in attn_fields
    ffn_fields = set(TransformerFFNDesc.__dataclass_fields__)
    assert "use_depthwise_conv" not in ffn_fields


def test_non_binary_txt_float_path_is_not_implemented() -> None:
    """`binaryFloats=false`（`.txt` / `.txt.gz`）我们**没有**实现。

    `desc.cpp:42-48` 那条分支用 `strtof` 逐个读文本浮点。我们只支持 `.bin` / `.bin.gz`
    —— 官方训练流水线里 `.txt` 权重早已不再产出，且它无法逐位往返（文本浮点的往返要
    靠原始字符串，跟我们已经踩过的坑 #3 是同一类问题）。这条测试钉住"我们没实现它"，
    以免有人以为 `parse_model` 能吃 `.txt`。
    """
    assert not hasattr(kb, "parse_model_txt")
    assert "binaryFloats" not in kb.__doc__ or "binaryFloats" in kb.__doc__
    # `.txt` 权重文件会被当成 `.bin` 处理，然后在第一个 float 块处报错
    with pytest.raises(kb.KatagoBinError):
        kb.parse_model(b"tinytest\n17\n2\n1\n20.0\n20.0\n20.0\n20.0\n40.0\n0.25\n30.0\n"
                       b"0\n0\n0\n0\n0\n0\n0\ntrunk\n1\n8\n4\n4\n2\n2\n"
                       b"model.conv_spatial\n3\n3\n2\n8\n1\n1\n0.5\n0.25\n")


def test_struct_pack_little_endian_float32_layout_is_what_we_emit() -> None:
    """确认我们的 float 块就是 `<f4` 而不是本机字节序 + 不是 float64。"""
    values = np.array([1.0, -2.5, 3.25], dtype=np.float32)
    w = kb._Writer()
    w.floats(values)
    payload = w.value()
    assert payload.startswith(kb.BIN_MARKER)
    assert payload.endswith(b"\n")
    body = payload[len(kb.BIN_MARKER) : -1]
    assert len(body) == 12
    assert body == struct.pack("<3f", 1.0, -2.5, 3.25)


# ---------------------------------------------------------------------------
# float 载荷里可以出现分隔符字节 —— 块边界必须来自形状，不能靠扫描
# ---------------------------------------------------------------------------


def _delimiter_looking_weights(count: int) -> np.ndarray:
    """造一段 float32 载荷，里面**故意**含有 `\\n`、`@`、甚至开头就是完整的 `@BIN@`。

    这些位模式都是有限数，所以能通过 `CHECKFINITE`（`desc.cpp:21-26`）那条关。
    位模式是手挑的，确保 little-endian 字节序列正好拼出想要的字节：

    ============  ==========  ==========================================
    位模式        LE 字节     效果
    ============  ==========  ==========================================
    ``0x4E494240``  ``40 42 49 4E``  偏移 0: ``@BIN``
    ``0x3F800040``  ``40 00 80 3F``  偏移 4: ``@`` -> 偏移 0..4 = **``@BIN@``**
    ``0x0000000A``  ``0A 00 00 00``  偏移 8: ``\\n``
    ``0x40000040``  ``40 00 00 40``  偏移 12 与 15: ``@``（两处）
    ``0x00000040``  ``40 00 00 00``  偏移 16: ``@``
    ``0x0000000A``  ``0A 00 00 00``  偏移 20: ``\\n``
    ============  ==========  ==========================================

    结果：2 个换行 + 5 个 ``@``，且**载荷开头就是一个假的 ``@BIN@``**。
    """
    bits = np.array(
        [
            0x4E494240,
            0x3F800040,
            0x0000000A,
            0x40000040,
            0x00000040,
            0x0000000A,
        ],
        dtype=np.uint32,
    )
    assert count >= bits.size
    out = np.empty(count, dtype=np.uint32)
    out[: bits.size] = bits
    # 其余位置用 1.0 / -1.0，字节干净但仍是有限数
    rest = np.resize(np.array([0x3F800000, 0xBF800000], dtype=np.uint32), count - bits.size)
    out[bits.size :] = rest
    return out.view(np.float32)


def _count_bin_blocks(model: ModelDesc) -> int:
    """序列化一个模型会发出多少个 float 块（用真实的 writer 计数，不靠猜）。"""
    order: List[np.ndarray] = []
    original = kb._Writer.floats

    def spy(self: kb._Writer, arr: np.ndarray) -> None:
        order.append(np.ascontiguousarray(arr, dtype="<f4"))
        return original(self, arr)

    kb._Writer.floats = spy  # type: ignore[method-assign]
    try:
        kb.serialize_model(model)
    finally:
        kb._Writer.floats = original  # type: ignore[method-assign]
    return len(order)


def test_float_payload_may_contain_newline_and_at_bin_bytes(raw: bytes, model: ModelDesc) -> None:
    """ 这是本模块最容易被人重新踩的坑，用真实文件量化它。

    实测 `kata1-tf2-b10c384-s2941M-d5872M.bin.gz` 的 328 个 float 载荷里
    **含有 129,087 个 `\\n`(0x0A) 和 115,326 个 `@`(0x40) 字节** —— 也就是说
    "扫到下一个 `\\n` 就是块尾" 这种最自然的写法**必然立刻跑飞**。

    C++ 之所以没事，是因为 `readFloats`（`desc.cpp:71`）用
    `in.read(bytes, numFloats*sizeof(float))` 按**声明的元素个数**精确读，
    从不往载荷里看。我们也必须如此。
    """
    if len(raw) != RAW_LEN:
        pytest.skip(_MISSING)

    order: List[np.ndarray] = []
    original = kb._Writer.floats

    def spy(self: kb._Writer, arr: np.ndarray) -> None:
        order.append(np.ascontiguousarray(arr, dtype="<f4"))
        return original(self, arr)

    kb._Writer.floats = spy  # type: ignore[method-assign]
    try:
        kb.serialize_model(model)
    finally:
        kb._Writer.floats = original  # type: ignore[method-assign]

    payloads = [a.tobytes() for a in order]
    newlines = sum(p.count(b"\n") for p in payloads)
    ats = sum(p.count(b"@") for p in payloads)
    assert len(order) == 328
    assert newlines == 129_087, "真实文件的 float 载荷里必须有换行字节"
    assert ats == 115_326, "真实文件的 float 载荷里必须有 @ 字节"
    # 这个数字不是凑的：它就是"为什么不能扫描"的证据
    assert newlines > len(order)


def test_weights_containing_delimiter_bytes_roundtrip_exactly() -> None:
    """把上面量化的危险做成合成用例：权重里塞满 `\\n` / `@` / `@BIN@`。

    如果有人把 `_Reader.floats` 改成"扫到下一个 `\\n`"或"扫到下一个 `@BIN@`"，
    这条测试会立刻失败（RED 有牙，已实测），而不是等到我们导出几十万行的模型时才发现。
    """
    weights = _delimiter_looking_weights(144)  # 3x3 卷积，2 -> 8 通道
    payload = weights.tobytes()
    # 先确认这个前提本身成立，否则测试会因为"没造出危险"而假绿
    assert payload[:5] == kb.BIN_MARKER, "载荷开头应该伪装成 @BIN@"
    assert payload.count(b"\n") == 2
    assert payload.count(b"@") == 5

    original = build_minimal_model(17)
    target = original.trunk.initial_conv
    assert target.weights.size == weights.size
    target.weights = weights

    blob = kb.serialize_model(original)
    # 文件里 `@BIN@` 的个数 = 真实块数 **+ 载荷里伪装出来的那 1 个**。
    # 解析器必须按声明的形状跳块，所以这多出来的 1 个完全无害。
    assert blob.count(kb.BIN_MARKER) == _count_bin_blocks(original) + 1
    parsed = kb.parse_model(blob)
    got = parsed.trunk.initial_conv.weights
    assert np.array_equal(got.view(np.uint32), weights.view(np.uint32))
    assert kb.serialize_model(parsed) == blob

    # BN 的 4 个数组是**背靠背**写的，中间一个文本 token 都没有：
    # 第一个 `@BIN@` 之后紧跟第二个，中间只有块尾的 `\n`。
    # 必须从 BN 的 name token 之后开始找，否则会命中载荷里那个假的 `@BIN@`。
    bn = parsed.trunk.blocks[0].desc.pre_bn  # type: ignore[union-attr]
    assert bn.num_channels == 8
    anchor = blob.find(b"model.blocks.0.norm\n")
    assert anchor > 0
    first = blob.find(kb.BIN_MARKER, anchor)
    payload_end = first + len(kb.BIN_MARKER) + 4 * bn.num_channels
    second = blob.find(kb.BIN_MARKER, payload_end)
    assert blob[payload_end:second] == b"\n", (
        f"BN 的 mean/variance 之间应当只有块尾换行，实际 {blob[payload_end:second]!r}"
    )
