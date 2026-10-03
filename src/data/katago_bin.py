"""KataGo `.bin.gz` 神经网络权重格式的**无损**解析 / 反向序列化（round-trip 验证）

这个模块存在的唯一理由：**证明我们真的懂 `.bin.gz` 的字节布局**，而不是"大概对"。
做法是 round-trip —— 解析官方权重 `katago/kata1-tf2-b10c384-s2941M-d5872M.bin.gz`，
原样重新序列化，与原文件做 byte 级 diff，要求**完全一致**。任何不一致都是格式理解错了。

CLI::

    python -m src.data.katago_bin roundtrip katago/kata1-tf2-b10c384-s2941M-d5872M.bin.gz
    python -m src.data.katago_bin inspect   katago/kata1-tf2-b10c384-s2941M-d5872M.bin.gz

权威格式来源（逐条对着写，一个字没改）
------------------------------------------------
* `cpp/neuralnet/desc.h`   —— 所有 `*LayerDesc` 结构与字段顺序
* `cpp/neuralnet/desc.cpp` —— 全部 reader 的确切实现
* `cpp/neuralnet/modelversion.h/.cpp` —— modelVersion 合法区间 (3..17)
* `python/export_model_pytorch.py`       —— **写入端**（`writeln` / `write_weights`）

格式骨架
------------------------------------------------
整个文件是**一段 ASCII 文本 + 若干裸 float32 块**的交替。文本部分由 C++ 的
`operator>>` 读（跳过空白、按空白切 token），float 部分由 `@BIN@` 标记分隔::

    <token>\n<token>\n...@BIN@<4*N 字节 little-endian float32>\n@BIN@...

写入端 `export_model_pytorch.py` 的两个 primitive 决定了**全部**字节布局::

    def writeln(s):    f.write((str(s)+"\n").encode(...))     # 每个文本字段 = str + "\n"
    def write_weights(w):
        writestr("@BIN@")                                     # 标记，前面没有空格
        f.write(struct.pack(f'<{N}f', *w))                     # 裸 little-endian float32
        writestr("\n")                                        # ⚠ 二进制块之后也有 "\n"

⚠ **反直觉之处 / 踩坑清单**（每一条都对应官方源码的一行，不写下来下一个人会重踩）
=============================================================================

1. **`@BIN@` 前面恰好一个 `\n`，后面也恰好一个 `\n`**。
   `readFloats`（`desc.cpp:55-61`）是**逐字符**扫描到 `@` 的，最多允许 100 个字符。
   写入端 `writeln` 已经吐了一个 `\n`，所以正常情况下中间只隔 1 个字符。
   ⇒ C++ 读完最后一个 float **不消费**块尾的 `\n`，它留给下一个 `>>`。
   ⇒ **文件最后一个字节是 `\n`，解析器消耗 `len-1` 字节**。反直觉，但这就是
   `42248752` vs `42248751` 的全部来源。序列化时必须把它写回去（`write_weights` 自然会写）。

2. **C++ 会把 fp32 次正规数（subnormal）冲成 0**（`desc.cpp:87-91`），我们**故意不做**。
   做了 round-trip 就对不上。`count_subnormals()` 单独报告有多少个，让 exporter 决定要不要冲。

3. **文本部分的浮点数必须逐字保留原文，不能从 float 反推字符串**。
   最典型的例子是 BN 的 epsilon：写入端 `writeln(1e-20)` 落盘就是字面量 `1e-20`；
   而 C++ 把它读进 `float` 变成 `9.999999682655225e-21`。从后者**永远拼不回** `1e-20`。
   ⇒ 所以文本里的浮点数一律用 `TextFloat`（只存 `text`，`value` 是派生属性）保存。

4. **卷积权重：文件序 `y,x,ic,oc`，内存序 `oc,ic,y,x` col-major**
   （`desc.cpp:133-154`）。写入端 `torch.permute(w,(2,3,1,0))`。
   ⚠ 本模块把权重**按文件序原样存**（`ConvDesc.weights`），因为那样 round-trip
   是纯 memcpy、零位错风险；permute 由 `to_torch_weight()` / `from_torch_weight()` 负责。

5. **MatMul 权重：文件序 `ic,oc`，内存序 `oc,ic`**（`desc.cpp:463-478`）。
   写入端 `torch.permute(w,(1,0))`。

6. **`hasScale` / `hasBias` 是"文件里有没有这个数组"的开关**，不是"值是不是 1/0"。
   `hasScale==0` 时 C++ **不读文件**、自己填 1.0（`desc.cpp:235-249`）。
   ⇒ 序列化时只在 `has_scale` 为真时写 scale 数组，否则会多写一块字节。

7. **块类型由字符串 tag 决定，未知即抛错**（`desc.cpp:1558-1559`）。
   tag 是**独立的一行**，在块自己的 `name` **之前**。

8. **`nested_bottleneck_block` 是递归的**（`desc.cpp:785-803`）：`preBN/act/preConv` →
   `numBlocks` 个子块（子块自己还带 tag！）→ `postBN/act/postConv`。

9. **`modelVersion` 改变文件布局**，不是只有 v≥13/v≥15 两处：
   * v < 11：`ActivationLayerDesc` **不读** kind token，激活被硬编码成 RELU
   * v < 12：policy head 期望 `policyOutChannels == 1`；v12-15 → 2；v16 → 4
   * v < 13：7 个 post-process 缩放系数**不在文件里**
   * v < 15：trunk 的 `trunkNormKind` / 5 个 unused、model 的 `metaEncoderVersion` /
     `preferPassAlive` / `preferExcludeTerritory` / 5 个 unused **都不在文件里**；
     policy head 的 `gpoolToPassBias` / `passActivation` / `gpoolToPassMul2` **也不在文件里**
   * v ≥ 16：value head 的 `sv3*` 宽度是 4；v ≥ 9 是 6；v ≥ 4 是 2；更早是 1
   * v ≥ 17：policy head 显式存 `policyOutChannels`（2 或 4）+ 3 个 unused；
     value head 3 个 unused

10. **`sha256` 不是文件的一部分**。它是 loader 算出来传给 `ModelDesc` 构造函数的
    （`desc.cpp:2467`），序列化时自然不写 —— 解析器同样不从文件读它。对称。

11. **trunk header 的 6 个通道数**里第 4 个是 `dilatedNumChannels`，源码里标了
    `//unused`（`desc.cpp:1678-1680`）—— 但它**在文件里**，必须原样读回原样写回。
    写入端填的是 `c_gpool`。第 3 个 `regularNumChannels` 写入端填 `c_mid - c_gpool`。

12. **`ropeTheta` / learnable `ropeFreqs` 前面各有一个被 C++ 读掉就丢弃的 name token**
    （`desc.cpp:1226` / `desc.cpp:1246`）。但它在文件里，所以要存、要写回。

13. **模型名有硬限制**：`ModelDesc::checkNameValid`（`desc.cpp:2429-2440`）要求
    1..96 字符且只能 `[A-Za-z0-9_-]`。这个限制**只作用于模型名**，层名没有长度限制。

14. **NaN / Inf 权重会被 C++ 拒绝**（`desc.cpp:21-26` 的 `CHECKFINITE`）。我们也拒绝。

15. **`cgroupSize != 0` 被 C++ 明确拒绝**（`desc.cpp:1086-1087`）：grouped spatial RMSNorm
    没实现。`spatial` 标志本身是支持的。

16. **🔴 千万别用"扫到下一个 `\\n`"或"扫到下一个 `@`"来定位 float 块的结尾** ——
    float32 载荷里**本来就会出现**这些字节。实测官方 b10c384 文件的 328 个载荷里
    含有 **129,087 个 `0x0A`(换行)** 和 **115,326 个 `0x40`('@')**，也就是说
    "扫换行"这种最自然的写法**几乎立刻就会跑飞**（实测：第一个换行就在载荷第 8 字节）。
    C++ 之所以没事，是因为 `readFloats` 用 `in.read(bytes, numFloats*sizeof(float))`
    按**声明的元素个数**精确读（`desc.cpp:71`），从不往载荷里看。
    ⇒ 块边界**只能**来自形状元数据。本模块的 `_Reader.floats` 就是这么做的，
    `tests/test_katago_bin_roundtrip.py::test_weights_containing_delimiter_bytes_roundtrip_exactly`
    用"载荷开头伪装成 `@BIN@`"的权重把这个坑钉住。

17. **同一个 BN 的几个数组是背靠背写的，中间一个文本 token 都没有。**
    `write_bn` 写完 `name / c_in / epsilon / hasScale / hasBias` 之后，
    mean、variance、[scale]、[bias] 是**连续**的 `@BIN@` 块，两个块之间只有块尾 `\n`。
    官方文件里有 **67 个**块的前导文本为空。⇒ 别假设"每个 `@BIN@` 前面都有 token"。

18. ⚠ `_iter_float_arrays`（本模块内部用来数 float / 数次正规数的遍历器）按
    **dataclass 字段声明顺序**递归，而**序列化顺序是另一回事** ——
    最明显的是 `TransformerFFNDesc` 的字段序是 `pre_ln, linear1, linear2, linear_gate`，
    但 writer 发出的是 `pre_ln, linear1, linear_gate, linear2`（`desc.cpp:1391-1393`）。
    所以它只能用于**顺序无关**的统计（求和 / 计数）。需要真实落盘顺序时请用
    `_Writer.floats` 之类的探针。

**没有实现的层类型**（官方 b10c384nbt 用不到，见文件末尾 `UNSUPPORTED_*` 注释）
------------------------------------------------
`GlobalPoolingResidualBlockDesc` 实现了，但 **FFN 里的 depthwise conv**、
**attention 的 QK-norm / GAB / TAB bias / inline registers / RW registers** 这些
exporter 侧的特性在 `.bin` 里**根本没有表示** —— 官方 exporter 直接 `assert` 掉
（`export_model_pytorch.py:565-576`、`:606-609`）。我们这边也就不需要实现。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "KatagoBinError",
    "TruncatedModelError",
    "ModelFormatError",
    "UnsupportedModelVersionError",
    "UnsupportedBlockKindError",
    "TextFloat",
    "ConvDesc",
    "BatchNormDesc",
    "ActivationDesc",
    "MatMulDesc",
    "MatBiasDesc",
    "RMSNormDesc",
    "TransformerRMSNormDesc",
    "TransformerAttentionDesc",
    "TransformerFFNDesc",
    "ResidualBlockDesc",
    "GlobalPoolingResidualBlockDesc",
    "NestedBottleneckDesc",
    "SGFMetadataEncoderDesc",
    "Block",
    "TrunkDesc",
    "PolicyHeadDesc",
    "ValueHeadDesc",
    "ModelDesc",
    "OLDEST_MODEL_VERSION",
    "LATEST_MODEL_VERSION",
    "TRUNK_NORM_KIND_STANDARD",
    "TRUNK_NORM_KIND_RMSNORM",
    "ACTIVATION_NAMES",
    "BLOCK_KIND_TAGS",
    "check_name_valid",
    "fmt_double",
    "maybe_gunzip",
    "read_model_bytes",
    "parse_model",
    "serialize_model",
    "load_model_file",
    "save_model_file",
    "count_parameters",
    "count_file_floats",
    "count_subnormals",
    "iter_layers",
    "summarize",
    "roundtrip_diff",
    "RoundtripReport",
]


# ---------------------------------------------------------------------------
# 常量（全部来自官方源码）
# ---------------------------------------------------------------------------

#: float 块的标记（`desc.cpp:50`）。写入端 `write_weights` 原样吐出。
BIN_MARKER = b"@BIN@"

#: `readFloats` 在扫描到 `@` 之前最多允许跳过多少个字符（`desc.cpp:58`）
MAX_CHARS_BEFORE_AT = 100

OLDEST_MODEL_VERSION = 3   # `modelversion.h:11`
LATEST_MODEL_VERSION = 17  # `modelversion.h:7`

TRUNK_NORM_KIND_STANDARD = 0  # `desc.h:235`
TRUNK_NORM_KIND_RMSNORM = 1   # `desc.h:236`

# `activations.h`
ACTIVATION_IDENTITY = 0
ACTIVATION_RELU = 1
ACTIVATION_MISH = 2
ACTIVATION_SILU = 3
ACTIVATION_MISH_SCALE8 = 12

#: 激活 int -> 文件里的字符串。`desc.cpp:389-397` 只认这四个。
ACTIVATION_NAMES = {
    ACTIVATION_IDENTITY: "ACTIVATION_IDENTITY",
    ACTIVATION_RELU: "ACTIVATION_RELU",
    ACTIVATION_MISH: "ACTIVATION_MISH",
    ACTIVATION_SILU: "ACTIVATION_SILU",
}
_NAME_TO_ACTIVATION = {v: k for k, v in ACTIVATION_NAMES.items()}

#: `metaEncoderVersion` -> 输入通道数（`modelversion.cpp:83-89`）
NUM_INPUT_META_CHANNELS = {0: 0, 1: 192}

#: block kind tag -> 类名标签（`desc.cpp:1460-1559`）
BLOCK_KIND_TAGS = {
    "ordinary_block": "ResidualBlockDesc",
    "gpool_block": "GlobalPoolingResidualBlockDesc",
    "nested_bottleneck_block": "NestedBottleneckDesc",
    "transformer_attention_block": "TransformerAttentionDesc",
    "transformer_ffn_block": "TransformerFFNDesc",
}

#: `ModelDesc::checkNameValid` 的长度上限（`desc.cpp:2433`）
MAX_MODEL_NAME_LEN = 96

_ASCII_MODEL_NAME_RE = re.compile(r"\A[A-Za-z0-9_-]+\Z")
_INT_RE = re.compile(r"\A[+-]?[0-9]+\Z")
_FLOAT_RE = re.compile(
    r"\A[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z"
    r"|\A[+-]?(?:inf(?:inity)?|nan)\Z",
    re.IGNORECASE,
)
_WS = frozenset(b" \t\n\v\f\r")

_EMPTY_F32 = np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class KatagoBinError(Exception):
    """.bin / .bin.gz 的所有解析失败都归到这一族，便于调用方一把抓住。"""


class TruncatedModelError(KatagoBinError):
    """文件提前结束（token 读空、float 块不够长）。"""


class ModelFormatError(KatagoBinError):
    """token 读到了但内容不合法（形状矛盾、unused 槽非 0、未知激活名 ...）。"""


class UnsupportedModelVersionError(ModelFormatError):
    """`modelVersion` 不在 `[3, 17]`。"""


class UnsupportedBlockKindError(ModelFormatError):
    """块 tag 不是那五个之一（`desc.cpp:1558-1559`）。"""


# ---------------------------------------------------------------------------
# 文本里的浮点数：逐字保留
# ---------------------------------------------------------------------------


def fmt_double(value: float) -> str:
    """把 Python float 格式化成写入端 `writeln` 会吐出的样子。

    Python 的 `repr(float)` 和 `str(float)` 完全一样（3.1+），而写入端
    `writeln(epsilon)` 就是 `str(1e-20)` == `'1e-20'`。所以这里就是往返的正解。
    """
    return repr(float(value))


@dataclass(frozen=True)
class TextFloat:
    """文件文本部分里的一个浮点数，**只存原文**。

    为什么不能只存 `float`：见模块 docstring 的坑 #3。BN 的 epsilon 在文件里是
    字面量 `1e-20`，读成 float32 之后是 `9.999999682655225e-21`，拼不回去。
    存原文是唯一可证明无损的做法。
    """

    text: str

    @property
    def value(self) -> float:
        return float(self.text)

    @classmethod
    def of(cls, value: float) -> "TextFloat":
        """从 Python float 构造（合成模型时用），走写入端同样的格式化路径。"""
        return cls(fmt_double(value))

    def __float__(self) -> float:  # pragma: no cover - 便利方法
        return float(self.text)

    def __str__(self) -> str:  # pragma: no cover - 便利方法
        return self.text


def check_name_valid(name: str) -> str:
    """复刻 `ModelDesc::checkNameValid`（`desc.cpp:2429-2440`）。

    只作用于**模型名**：1..96 字符，字符集 `[A-Za-z0-9_-]`。层名不受此限制。
    模型名会被拼进磁盘缓存文件名（比如 TensorRT plan cache），所以必须是文件系统安全的。
    """
    if len(name) == 0:
        raise ModelFormatError("Model name is empty, a nonempty model name is required")
    if len(name) > MAX_MODEL_NAME_LEN:
        raise ModelFormatError(
            f"Model name is too long ({len(name)} chars, max {MAX_MODEL_NAME_LEN}): {name}"
        )
    if not _ASCII_MODEL_NAME_RE.match(name):
        raise ModelFormatError(
            "Model name must contain only alphanumeric characters, underscores, "
            f"and hyphens: {name}"
        )
    return name


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class _Reader:
    """字节流上的 C++ `istream >>` 模拟器。

    `token()` 用 latin-1 解码：任何字节序列都能无损往返（`str.encode('latin-1')`
    是字节透明的）。写入端用 ascii+backslashreplace，那是 exporter 的事，不是格式的事。
    """

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    # -- 基础 token ------------------------------------------------------

    def token(self) -> str:
        data = self.data
        n = len(data)
        i = self.pos
        while i < n and data[i] in _WS:
            i += 1
        if i >= n:
            raise TruncatedModelError(
                f"expected a token at byte offset {i} but the file ends at {n}"
            )
        j = i
        while j < n and data[j] not in _WS:
            j += 1
        self.pos = j
        return data[i:j].decode("latin-1")

    def integer(self, what: str) -> int:
        tok = self.token()
        if not _INT_RE.match(tok):
            raise ModelFormatError(f"{what}: expected an integer, got {tok!r}")
        val = int(tok)
        if not (-(2**31) <= val < 2**31):
            raise ModelFormatError(f"{what}: integer out of 32-bit range: {tok!r}")
        return val

    def boolean(self, what: str) -> bool:
        # C++ 读成 int 再 `!= 0`，所以任何非 0 都算真
        return self.integer(what) != 0

    def text_float(self, what: str) -> TextFloat:
        tok = self.token()
        if not _FLOAT_RE.match(tok):
            raise ModelFormatError(f"{what}: expected a floating point number, got {tok!r}")
        return TextFloat(tok)

    # -- float 块（`desc.cpp:40-92`） ------------------------------------

    def floats(self, count: int, name: str) -> np.ndarray:
        """复刻 `readFloats` 的 `binaryFloats=true` 分支。

        ⚠ 与 C++ 的两处**故意**不同：
        1. C++ 会把 fp32 次正规数冲成 0（`desc.cpp:89-90`）—— 我们不做，见模块
           docstring 坑 #2，否则 round-trip 对不上。
        2. C++ 在 big-endian 上会翻转字节 —— 我们显式按 `<f4` 读，与平台无关。

        ⚠ 坑 #16：这里**必须**按 `count` 精确读 `4*count` 字节。
        float32 载荷里会出现 `\\n`(0x0A) 和 `@`(0x40)（实测官方文件分别有
        129,087 和 115,326 个），扫描分隔符定位块尾一定跑飞。
        """
        data = self.data
        n = len(data)

        chars_before_at = 0
        while True:
            if self.pos >= n:
                raise TruncatedModelError(
                    f"{name}: expected '@BIN@' marker for {count} floats but the file ends"
                )
            ch = data[self.pos]
            self.pos += 1
            if ch == 0x40:  # '@'
                break
            chars_before_at += 1
            if chars_before_at > MAX_CHARS_BEFORE_AT:
                raise ModelFormatError(
                    f"{name}: could not read float weights -- more than "
                    f"{MAX_CHARS_BEFORE_AT} characters before '@'. Invalid model -- "
                    "perhaps you are trying to load a .txt.gz model as a .bin.gz model?"
                )
        if data[self.pos : self.pos + 4] != b"BIN@":
            got = data[self.pos : self.pos + 4]
            raise ModelFormatError(
                f"{name}: did not find expected header for binary float block "
                f"(got {got!r} instead of b'BIN@')"
            )
        self.pos += 4

        end = self.pos + 4 * count
        if end > n:
            raise TruncatedModelError(
                f"{name}: expected {count} floats ({4 * count} bytes) but only "
                f"{n - self.pos} bytes remain"
            )
        arr = np.frombuffer(data, dtype="<f4", count=count, offset=self.pos).copy()
        self.pos = end
        if not np.isfinite(arr).all():
            raise ModelFormatError(f"{name}: NaN or infinite neural net weight or parameter")
        return arr

    def at_eof(self) -> bool:
        return self.pos >= len(self.data)


# ---------------------------------------------------------------------------
# Writer（与 Reader 逐条对称）
# ---------------------------------------------------------------------------


class _Writer:
    """字节流构造器。两个 primitive 对应写入端的 `writeln` / `write_weights`。"""

    __slots__ = ("_parts",)

    def __init__(self) -> None:
        self._parts: List[bytes] = []

    def token(self, value: Any) -> None:
        """`writeln(s)`：文本 + `\\n`（坑 #1：float 块前面就靠这个 `\\n`）。"""
        self._parts.append(str(value).encode("latin-1"))
        self._parts.append(b"\n")

    def floats(self, arr: np.ndarray) -> None:
        """`write_weights(w)`：`@BIN@` + 裸 little-endian float32 + `\\n`。

        `ascontiguousarray(..., dtype='<f4')` 对已经是 `<f4` C 连续 的数组是**零拷贝**，
        所以 `.tobytes()` 是纯 memcpy —— 逐位一致的保证来自这里，不是来自"小心处理"。
        """
        a = np.ascontiguousarray(arr, dtype="<f4")
        self._parts.append(BIN_MARKER)
        self._parts.append(a.tobytes())
        self._parts.append(b"\n")

    def value(self) -> bytes:
        return b"".join(self._parts)


def _as_f32(values: Any) -> np.ndarray:
    """把任意 float 序列变成 C 连续 `<f4`（供合成模型构造用）。"""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32), dtype="<f4")


# ---------------------------------------------------------------------------
# 叶子层
# ---------------------------------------------------------------------------


@dataclass
class ConvDesc:
    """`ConvLayerDesc`（`desc.h:15-41`）。

    ⚠ `weights` 是**文件序** `y,x,ic,oc`（`desc.cpp:133-154`），长度
    `conv_y*conv_x*in_channels*out_channels`。内存 / torch 序是 `oc,ic,y,x`，
    用 `to_torch_weight()` 转换。
    """

    name: str
    conv_y: int
    conv_x: int
    in_channels: int
    out_channels: int
    dilation_y: int
    dilation_x: int
    weights: np.ndarray = field(default_factory=lambda: _EMPTY_F32)

    @property
    def num_weights(self) -> int:
        return self.conv_y * self.conv_x * self.in_channels * self.out_channels

    @property
    def num_parameters(self) -> int:
        return int(self.weights.size)

    @property
    def num_file_floats(self) -> int:
        return int(self.weights.size)

    @property
    def spatial_conv_depth(self) -> float:
        """1x1 -> 0，3x3 -> 1，5x5 -> 2 ...（`desc.cpp:180`）"""
        return (self.conv_y + self.conv_x - 2) / 4.0

    def to_torch_weight(self) -> np.ndarray:
        """文件序 `(y,x,ic,oc)` -> torch / 内存序 `(oc,ic,y,x)`。

        对应写入端的 `torch.permute(w,(2,3,1,0))`（`export_model_pytorch.py:396`）的逆。
        """
        w = self.weights.reshape(self.conv_y, self.conv_x, self.in_channels, self.out_channels)
        return np.ascontiguousarray(w.transpose(3, 2, 0, 1))

    @classmethod
    def from_torch_weight(
        cls,
        name: str,
        weight: Any,
        dilation_y: int = 1,
        dilation_x: int = 1,
    ) -> "ConvDesc":
        """torch / 内存序 `(oc,ic,y,x)` -> 文件序 `y,x,ic,oc`（exporter 用）。"""
        w = np.asarray(weight, dtype=np.float32)
        if w.ndim != 4:
            raise ModelFormatError(f"{name}: conv weight must be 4-D (oc,ic,y,x), got {w.shape}")
        oc, ic, dy, dx = (int(v) for v in w.shape)
        return cls(
            name=name,
            conv_y=dy,
            conv_x=dx,
            in_channels=ic,
            out_channels=oc,
            dilation_y=dilation_y,
            dilation_x=dilation_x,
            weights=_as_f32(w.transpose(2, 3, 1, 0).reshape(-1)),
        )

    @classmethod
    def read(cls, r: _Reader) -> "ConvDesc":
        name = r.token()
        conv_y = r.integer(f"{name}: convYSize")
        conv_x = r.integer(f"{name}: convXSize")
        in_channels = r.integer(f"{name}: inChannels")
        out_channels = r.integer(f"{name}: outChannels")
        dilation_y = r.integer(f"{name}: dilationY")
        dilation_x = r.integer(f"{name}: dilationX")

        # `desc.cpp:124-131`
        if conv_x <= 0 or conv_y <= 0:
            raise ModelFormatError(f"{name}: convolution filter sizes must be positive")
        if in_channels <= 0 or out_channels <= 0:
            raise ModelFormatError(f"{name}: number of in and out channels must be positive")
        if dilation_x <= 0 or dilation_y <= 0:
            raise ModelFormatError(f"{name}: dilation factors must be positive")
        if conv_x % 2 != 1 or conv_y % 2 != 1:
            raise ModelFormatError(
                f"{name}: convolution filter sizes must be odd, found even sizes"
            )

        weights = r.floats(conv_y * conv_x * in_channels * out_channels, name)
        return cls(name, conv_y, conv_x, in_channels, out_channels, dilation_y, dilation_x, weights)

    def write(self, w: _Writer) -> None:
        if self.weights.size != self.num_weights:
            raise ModelFormatError(
                f"{self.name}: conv weight count mismatch: have {self.weights.size}, "
                f"expected {self.num_weights}"
            )
        w.token(self.name)
        w.token(self.conv_y)
        w.token(self.conv_x)
        w.token(self.in_channels)
        w.token(self.out_channels)
        w.token(self.dilation_y)
        w.token(self.dilation_x)
        w.floats(self.weights)


@dataclass
class BatchNormDesc:
    """`BatchNormLayerDesc`（`desc.h:43-75`）。

    ⚠ 坑 #6：`has_scale` / `has_bias` 是**"文件里有没有这个数组"**的开关。
    为假时 C++ 不读文件、自己填 1.0 / 0.0（`desc.cpp:235-249`）。序列化时为假
    就不写那个数组，否则多出 8*num_channels 字节。
    """

    name: str
    num_channels: int
    epsilon: TextFloat
    has_scale: bool
    has_bias: bool
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray
    bias: np.ndarray

    @property
    def num_parameters(self) -> int:
        """⚠ 只数可学习的 scale + bias。mean / variance 是 running stats，不是参数。
        复刻 `desc.cpp:278-282`（含那条 "Count the learnable scale and bias" 注释）。"""
        return (self.num_channels if self.has_scale else 0) + (
            self.num_channels if self.has_bias else 0
        )

    @property
    def num_file_floats(self) -> int:
        """文件里这个 BN 实际占的 float 数（mean + variance [+ scale] [+ bias]）。"""
        return 2 * self.num_channels + (
            self.num_channels if self.has_scale else 0
        ) + (self.num_channels if self.has_bias else 0)

    def compute_merged(self) -> Tuple[np.ndarray, np.ndarray]:
        """复刻 `computeMerged`（`desc.cpp:284-291`），给推理侧对齐用。

        ⚠ 这会**读**权重、不改 IR —— exporter 不该调用它，否则 round-trip 就废了。
        """
        merged_scale = self.scale / np.sqrt(self.variance + float(self.epsilon))
        merged_bias = self.bias - merged_scale * self.mean
        return merged_scale.astype(np.float32), merged_bias.astype(np.float32)

    @classmethod
    def read(cls, r: _Reader) -> "BatchNormDesc":
        name = r.token()
        num_channels = r.integer(f"{name}: numChannels")
        epsilon = r.text_float(f"{name}: epsilon")
        has_scale = r.boolean(f"{name}: hasScale")
        has_bias = r.boolean(f"{name}: hasBias")

        # `desc.cpp:220-223`
        if num_channels < 1:
            raise ModelFormatError(f"{name}: numChannels ({num_channels}) < 1")
        if not (epsilon.value > 0) or not np.isfinite(epsilon.value):
            raise ModelFormatError(
                f"{name}: epsilon ({epsilon.text}) is not positive and finite"
            )

        mean = r.floats(num_channels, name)
        variance = r.floats(num_channels, name)
        if has_scale:
            scale = r.floats(num_channels, name)
        else:
            scale = np.ones(num_channels, dtype=np.float32)
        if has_bias:
            bias = r.floats(num_channels, name)
        else:
            bias = np.zeros(num_channels, dtype=np.float32)

        return cls(name, num_channels, epsilon, has_scale, has_bias, mean, variance, scale, bias)

    def write(self, w: _Writer) -> None:
        w.token(self.name)
        w.token(self.num_channels)
        w.token(self.epsilon.text)
        w.token(1 if self.has_scale else 0)
        w.token(1 if self.has_bias else 0)
        w.floats(self.mean)
        w.floats(self.variance)
        if self.has_scale:
            w.floats(self.scale)
        if self.has_bias:
            w.floats(self.bias)


@dataclass
class ActivationDesc:
    """`ActivationLayerDesc`（`desc.h:77-91`）。

    ⚠ 坑 #9：`model_version < 11` 时文件里**没有** kind token，C++ 硬编码成 RELU
    （`desc.cpp:402-404`）。序列化由 `TrunkDesc.write` 之类按 model_version 决定
    写不写，所以这里不需要额外标志位。
    """

    name: str
    activation: int = ACTIVATION_RELU

    @property
    def activation_name(self) -> str:
        try:
            return ACTIVATION_NAMES[self.activation]
        except KeyError:  # pragma: no cover - 写入前会被校验拦住
            raise ModelFormatError(
                f"{self.name}: activation {self.activation} cannot be serialized "
                f"(only {sorted(ACTIVATION_NAMES)} are writable)"
            ) from None

    @classmethod
    def read(cls, r: _Reader, model_version: int) -> "ActivationDesc":
        name = r.token()
        if model_version >= 11:
            kind = r.token()
            try:
                activation = _NAME_TO_ACTIVATION[kind]
            except KeyError:
                raise ModelFormatError(f"{name}: unknown activation {kind}") from None
            return cls(name, activation)
        return cls(name, ACTIVATION_RELU)

    def write(self, w: _Writer, model_version: int) -> None:
        w.token(self.name)
        if model_version >= 11:
            w.token(self.activation_name)


@dataclass
class MatMulDesc:
    """`MatMulLayerDesc`（`desc.h:93-114`）。

    ⚠ 坑 #5：文件序 `ic,oc`（`desc.cpp:463-478`），torch / 内存序 `oc,ic`。
    """

    name: str
    in_channels: int
    out_channels: int
    weights: np.ndarray = field(default_factory=lambda: _EMPTY_F32)

    @property
    def num_parameters(self) -> int:
        return int(self.weights.size)

    @property
    def num_file_floats(self) -> int:
        return int(self.weights.size)

    def to_torch_weight(self) -> np.ndarray:
        """文件序 `(ic,oc)` -> torch 序 `(oc,ic)`（写入端 `permute((1,0))` 的逆）。"""
        return _as_f32(self.weights.reshape(self.in_channels, self.out_channels).transpose(1, 0))

    @classmethod
    def from_torch_weight(cls, name: str, weight: Any) -> "MatMulDesc":
        w = np.asarray(weight, dtype=np.float32)
        if w.ndim != 2:
            raise ModelFormatError(f"{name}: matmul weight must be 2-D (oc,ic), got {w.shape}")
        oc, ic = (int(v) for v in w.shape)
        return cls(name, ic, oc, _as_f32(w.transpose(1, 0).reshape(-1)))

    @classmethod
    def read(cls, r: _Reader) -> "MatMulDesc":
        name = r.token()
        in_channels = r.integer(f"{name}: inChannels")
        out_channels = r.integer(f"{name}: outChannels")
        # `desc.cpp:460-461`
        if in_channels <= 0 or out_channels <= 0:
            raise ModelFormatError(f"{name}: number of in and out channels must be positive")
        weights = r.floats(in_channels * out_channels, name)
        return cls(name, in_channels, out_channels, weights)

    def write(self, w: _Writer) -> None:
        if self.weights.size != self.in_channels * self.out_channels:
            raise ModelFormatError(
                f"{self.name}: matmul weight count mismatch: have {self.weights.size}, "
                f"expected {self.in_channels * self.out_channels}"
            )
        w.token(self.name)
        w.token(self.in_channels)
        w.token(self.out_channels)
        w.floats(self.weights)


@dataclass
class MatBiasDesc:
    """`MatBiasLayerDesc`（`desc.h:116-135`）：一维 bias，无形状信息以外的排列问题。"""

    name: str
    num_channels: int
    weights: np.ndarray = field(default_factory=lambda: _EMPTY_F32)

    @property
    def num_parameters(self) -> int:
        return int(self.weights.size)

    @property
    def num_file_floats(self) -> int:
        return int(self.weights.size)

    @classmethod
    def read(cls, r: _Reader) -> "MatBiasDesc":
        name = r.token()
        num_channels = r.integer(f"{name}: numChannels")
        if num_channels <= 0:
            raise ModelFormatError(f"{name}: number of channels must be positive")
        return cls(name, num_channels, r.floats(num_channels, name))

    def write(self, w: _Writer) -> None:
        if self.weights.size != self.num_channels:
            raise ModelFormatError(
                f"{self.name}: matbias weight count mismatch: have {self.weights.size}, "
                f"expected {self.num_channels}"
            )
        w.token(self.name)
        w.token(self.num_channels)
        w.floats(self.weights)


@dataclass
class RMSNormDesc:
    """`RMSNormLayerDesc`（`desc.h:238-258`）：trunk tip 的最终 norm（v>=15）。

    ⚠ 坑 #15：`cgroup_size != 0` 被 C++ 明确拒绝（`desc.cpp:1086-1087`），
    grouped spatial RMSNorm 没实现。`spatial` 本身是支持的。
    """

    name: str
    num_channels: int
    epsilon: TextFloat
    spatial: bool
    cgroup_size: int
    gamma: np.ndarray
    beta: np.ndarray

    @property
    def num_parameters(self) -> int:
        return int(self.gamma.size + self.beta.size)

    @property
    def num_file_floats(self) -> int:
        return int(self.gamma.size + self.beta.size)

    @classmethod
    def read(cls, r: _Reader) -> "RMSNormDesc":
        name = r.token()
        num_channels = r.integer(f"{name}: numChannels")
        epsilon = r.text_float(f"{name}: epsilon")
        spatial = r.boolean(f"{name}: spatial")
        cgroup_size = r.integer(f"{name}: cgroupSize")

        # `desc.cpp:1082-1087`
        if not (epsilon.value > 0) or epsilon.value > 1.0:
            raise ModelFormatError(
                f"{name}: rmsnorm epsilon ({epsilon.text}) is not positive or is too large"
            )
        if num_channels < 1:
            raise ModelFormatError(f"{name}: rmsnorm numChannels ({num_channels}) < 1")
        if cgroup_size != 0:
            raise ModelFormatError(
                f"{name}: rmsnorm cgroupSize ({cgroup_size}) != 0, grouped spatial "
                "RMSNorm is not supported"
            )

        gamma = r.floats(num_channels, name)
        beta = r.floats(num_channels, name)
        return cls(name, num_channels, epsilon, spatial, cgroup_size, gamma, beta)

    def write(self, w: _Writer) -> None:
        w.token(self.name)
        w.token(self.num_channels)
        w.token(self.epsilon.text)
        w.token(1 if self.spatial else 0)
        w.token(self.cgroup_size)
        w.floats(self.gamma)
        w.floats(self.beta)


@dataclass
class TransformerRMSNormDesc:
    """`TransformerRMSNormDesc`（`desc.h:261-278`）：transformer 内部的轻量 norm。

    只有 `weight`，没有 bias，没有 spatial 模式（`desc.cpp:1127-1145`）。
    """

    name: str
    num_channels: int
    epsilon: TextFloat
    weight: np.ndarray

    @property
    def num_parameters(self) -> int:
        return int(self.weight.size)

    @property
    def num_file_floats(self) -> int:
        return int(self.weight.size)

    @classmethod
    def read(cls, r: _Reader) -> "TransformerRMSNormDesc":
        name = r.token()
        num_channels = r.integer(f"{name}: numChannels")
        epsilon = r.text_float(f"{name}: epsilon")
        # `desc.cpp:1134-1137`
        if num_channels < 1:
            raise ModelFormatError(f"{name}: transformer rmsnorm numChannels ({num_channels}) < 1")
        if not (epsilon.value > 0) or epsilon.value > 1.0:
            raise ModelFormatError(
                f"{name}: transformer rmsnorm epsilon ({epsilon.text}) is not positive "
                "or is too large"
            )
        return cls(name, num_channels, epsilon, r.floats(num_channels, name))

    def write(self, w: _Writer) -> None:
        w.token(self.name)
        w.token(self.num_channels)
        w.token(self.epsilon.text)
        w.floats(self.weight)


@dataclass
class TransformerAttentionDesc:
    """`TransformerAttentionDesc`（`desc.h:280-321`）。

    ⚠ 坑 #12：learnable RoPE 前面有一个 `rope_freqs_name`、fixed RoPE 前面有一个
    `rope_theta_name`，C++ 读掉就丢弃（`desc.cpp:1226` / `desc.cpp:1246`）。
    它们**在文件里**，所以要存要写回。

    RoPE 旋转的是交错通道对 `(2p, 2p+1)`，所以 `use_rope` 时 `q_head_dim` 必须为偶数
    （`desc.cpp:1201-1202`）。
    """

    name: str
    num_heads: int
    num_kv_heads: int
    q_head_dim: int
    v_head_dim: int
    use_rope: bool
    learnable_rope: bool
    pre_ln: TransformerRMSNormDesc
    q_proj: MatMulDesc
    k_proj: MatMulDesc
    v_proj: MatMulDesc
    out_proj: MatMulDesc
    rope_freqs_name: Optional[str] = None
    rope_num_kv_heads: int = 0
    rope_num_pairs: int = 0
    rope_freqs: np.ndarray = field(default_factory=lambda: _EMPTY_F32)
    rope_theta_name: Optional[str] = None
    rope_theta: Optional[TextFloat] = None

    @property
    def num_parameters(self) -> int:
        # `desc.cpp:1283-1291`：rope_freqs 空数组（fixed / 无 rope）贡献 0
        return (
            self.pre_ln.num_parameters
            + self.q_proj.num_parameters
            + self.k_proj.num_parameters
            + self.v_proj.num_parameters
            + self.out_proj.num_parameters
            + int(self.rope_freqs.size)
        )

    @property
    def num_file_floats(self) -> int:
        return (
            self.pre_ln.num_file_floats
            + self.q_proj.num_file_floats
            + self.k_proj.num_file_floats
            + self.v_proj.num_file_floats
            + self.out_proj.num_file_floats
            + int(self.rope_freqs.size)
        )

    @classmethod
    def read(cls, r: _Reader) -> "TransformerAttentionDesc":
        name = r.token()
        num_heads = r.integer(f"{name}: numHeads")
        num_kv_heads = r.integer(f"{name}: numKVHeads")
        q_head_dim = r.integer(f"{name}: qHeadDim")
        v_head_dim = r.integer(f"{name}: vHeadDim")
        use_rope = r.boolean(f"{name}: useRope")
        learnable_rope = r.boolean(f"{name}: learnableRope")

        # `desc.cpp:1192-1202`
        if num_heads < 1 or num_kv_heads < 1:
            raise ModelFormatError(
                f"{name}: transformer attention numHeads and numKVHeads must be positive"
            )
        if num_heads % num_kv_heads != 0:
            raise ModelFormatError(f"{name}: numHeads must be divisible by numKVHeads")
        if q_head_dim < 1 or v_head_dim < 1:
            raise ModelFormatError(f"{name}: head dims must be positive")
        if use_rope and q_head_dim % 2 != 0:
            raise ModelFormatError(
                f"{name}: qHeadDim ({q_head_dim}) must be even when RoPE is used"
            )

        pre_ln = TransformerRMSNormDesc.read(r)
        q_proj = MatMulDesc.read(r)
        k_proj = MatMulDesc.read(r)
        v_proj = MatMulDesc.read(r)
        out_proj = MatMulDesc.read(r)

        # `desc.cpp:1210-1217`
        if q_proj.out_channels != num_heads * q_head_dim:
            raise ModelFormatError(
                f"{name}: qProj.outChannels ({q_proj.out_channels}) != "
                f"numHeads*qHeadDim ({num_heads * q_head_dim})"
            )
        if k_proj.out_channels != num_kv_heads * q_head_dim:
            raise ModelFormatError(
                f"{name}: kProj.outChannels ({k_proj.out_channels}) != "
                f"numKVHeads*qHeadDim ({num_kv_heads * q_head_dim})"
            )
        if v_proj.out_channels != num_kv_heads * v_head_dim:
            raise ModelFormatError(
                f"{name}: vProj.outChannels ({v_proj.out_channels}) != "
                f"numKVHeads*vHeadDim ({num_kv_heads * v_head_dim})"
            )
        if out_proj.in_channels != num_heads * v_head_dim:
            raise ModelFormatError(
                f"{name}: outProj.inChannels ({out_proj.in_channels}) != "
                f"numHeads*vHeadDim ({num_heads * v_head_dim})"
            )

        rope_freqs_name: Optional[str] = None
        rope_num_kv_heads = 0
        rope_num_pairs = 0
        rope_freqs = _EMPTY_F32
        rope_theta_name: Optional[str] = None
        rope_theta: Optional[TextFloat] = None

        if use_rope:
            if learnable_rope:
                rope_freqs_name = r.token()
                rope_num_kv_heads = r.integer(f"{name}: ropeNumKVHeads")
                rope_num_pairs = r.integer(f"{name}: ropeNumPairs")
                rope_dim2 = r.integer(f"{name}: rope dim2")
                # `desc.cpp:1233-1238`
                if rope_num_kv_heads != num_kv_heads:
                    raise ModelFormatError(
                        f"{name}: ropeNumKVHeads ({rope_num_kv_heads}) != numKVHeads ({num_kv_heads})"
                    )
                if rope_num_pairs != q_head_dim // 2:
                    raise ModelFormatError(
                        f"{name}: ropeNumPairs ({rope_num_pairs}) != qHeadDim/2 ({q_head_dim // 2})"
                    )
                if rope_dim2 != 2:
                    raise ModelFormatError(f"{name}: rope freq dim2 must be 2")
                rope_freqs = r.floats(rope_num_kv_heads * rope_num_pairs * 2, name)
            else:
                rope_theta_name = r.token()
                rope_theta = r.text_float(f"{name}: ropeTheta")
                if not (rope_theta.value > 0) or not np.isfinite(rope_theta.value):
                    raise ModelFormatError(f"{name}: rope theta must be positive and finite")

        return cls(
            name=name,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            q_head_dim=q_head_dim,
            v_head_dim=v_head_dim,
            use_rope=use_rope,
            learnable_rope=learnable_rope,
            pre_ln=pre_ln,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            out_proj=out_proj,
            rope_freqs_name=rope_freqs_name,
            rope_num_kv_heads=rope_num_kv_heads,
            rope_num_pairs=rope_num_pairs,
            rope_freqs=rope_freqs,
            rope_theta_name=rope_theta_name,
            rope_theta=rope_theta,
        )

    def write(self, w: _Writer, model_version: int = 0) -> None:
        w.token(self.name)
        w.token(self.num_heads)
        w.token(self.num_kv_heads)
        w.token(self.q_head_dim)
        w.token(self.v_head_dim)
        w.token(1 if self.use_rope else 0)
        w.token(1 if self.learnable_rope else 0)

        self.pre_ln.write(w)
        self.q_proj.write(w)
        self.k_proj.write(w)
        self.v_proj.write(w)
        self.out_proj.write(w)

        if not self.use_rope:
            return
        if self.learnable_rope:
            if self.rope_freqs_name is None:
                raise ModelFormatError(
                    f"{self.name}: learnable RoPE needs a rope_freqs_name token"
                )
            w.token(self.rope_freqs_name)
            w.token(self.rope_num_kv_heads)
            w.token(self.rope_num_pairs)
            w.token(2)
            w.floats(self.rope_freqs)
        else:
            if self.rope_theta is None:
                raise ModelFormatError(f"{self.name}: fixed RoPE needs a rope_theta value")
            w.token(self.rope_theta_name if self.rope_theta_name is not None else self.name)
            w.token(self.rope_theta.text)


@dataclass
class TransformerFFNDesc:
    """`TransformerFFNDesc`（`desc.h:323-345`）。

    ⚠ `use_swiglu` 时 `linear_gate` 才在文件里（`desc.cpp:1391-1393`）。
    """

    name: str
    num_channels: int
    ffn_channels: int
    use_swiglu: bool
    pre_ln: TransformerRMSNormDesc
    linear1: MatMulDesc
    linear2: MatMulDesc
    linear_gate: MatMulDesc = field(default_factory=lambda: MatMulDesc("", 0, 0, _EMPTY_F32))

    @property
    def num_parameters(self) -> int:
        return (
            self.pre_ln.num_parameters
            + self.linear1.num_parameters
            + self.linear_gate.num_parameters  # 非 swiglu 时是空数组 -> 0
            + self.linear2.num_parameters
        )

    @property
    def num_file_floats(self) -> int:
        return (
            self.pre_ln.num_file_floats
            + self.linear1.num_file_floats
            + self.linear_gate.num_file_floats
            + self.linear2.num_file_floats
        )

    @classmethod
    def read(cls, r: _Reader) -> "TransformerFFNDesc":
        name = r.token()
        num_channels = r.integer(f"{name}: numChannels")
        ffn_channels = r.integer(f"{name}: ffnChannels")
        use_swiglu = r.boolean(f"{name}: useSwiGLU")

        # `desc.cpp:1386-1387`
        if num_channels < 1 or ffn_channels < 1:
            raise ModelFormatError(f"{name}: transformer ffn channels must be positive")

        pre_ln = TransformerRMSNormDesc.read(r)
        linear1 = MatMulDesc.read(r)
        linear_gate = MatMulDesc.read(r) if use_swiglu else MatMulDesc("", 0, 0, _EMPTY_F32)
        linear2 = MatMulDesc.read(r)

        # `desc.cpp:1396-1407`
        if linear1.in_channels != num_channels:
            raise ModelFormatError(
                f"{name}: linear1.inChannels ({linear1.in_channels}) != numChannels ({num_channels})"
            )
        if linear1.out_channels != ffn_channels:
            raise ModelFormatError(
                f"{name}: linear1.outChannels ({linear1.out_channels}) != ffnChannels ({ffn_channels})"
            )
        if use_swiglu:
            if linear_gate.in_channels != num_channels:
                raise ModelFormatError(
                    f"{name}: linearGate.inChannels ({linear_gate.in_channels}) != "
                    f"numChannels ({num_channels})"
                )
            if linear_gate.out_channels != ffn_channels:
                raise ModelFormatError(
                    f"{name}: linearGate.outChannels ({linear_gate.out_channels}) != "
                    f"ffnChannels ({ffn_channels})"
                )
        if linear2.in_channels != ffn_channels:
            raise ModelFormatError(
                f"{name}: linear2.inChannels ({linear2.in_channels}) != ffnChannels ({ffn_channels})"
            )
        if linear2.out_channels != num_channels:
            raise ModelFormatError(
                f"{name}: linear2.outChannels ({linear2.out_channels}) != numChannels ({num_channels})"
            )

        return cls(name, num_channels, ffn_channels, use_swiglu, pre_ln, linear1, linear2, linear_gate)

    def write(self, w: _Writer, model_version: int = 0) -> None:
        w.token(self.name)
        w.token(self.num_channels)
        w.token(self.ffn_channels)
        w.token(1 if self.use_swiglu else 0)

        self.pre_ln.write(w)
        self.linear1.write(w)
        if self.use_swiglu:
            self.linear_gate.write(w)
        self.linear2.write(w)


# ---------------------------------------------------------------------------
# 块
# ---------------------------------------------------------------------------


@dataclass
class ResidualBlockDesc:
    """`ResidualBlockDesc`（`desc.h:137-163`）= `ordinary_block`。

    文件里**没有** tag（tag 由调用方 `parseResidualBlockStack` 读），所以
    `write` 只写自己；tag 由 `Block.write` 补。
    """

    name: str
    pre_bn: BatchNormDesc
    pre_activation: ActivationDesc
    regular_conv: ConvDesc
    mid_bn: BatchNormDesc
    mid_activation: ActivationDesc
    final_conv: ConvDesc

    @property
    def num_parameters(self) -> int:
        return (
            self.pre_bn.num_parameters
            + self.regular_conv.num_parameters
            + self.mid_bn.num_parameters
            + self.final_conv.num_parameters
        )

    @property
    def num_file_floats(self) -> int:
        return (
            self.pre_bn.num_file_floats
            + self.regular_conv.num_file_floats
            + self.mid_bn.num_file_floats
            + self.final_conv.num_file_floats
        )

    @classmethod
    def read(cls, r: _Reader, model_version: int) -> "ResidualBlockDesc":
        name = r.token()
        pre_bn = BatchNormDesc.read(r)
        pre_activation = ActivationDesc.read(r, model_version)
        regular_conv = ConvDesc.read(r)
        mid_bn = BatchNormDesc.read(r)
        mid_activation = ActivationDesc.read(r, model_version)
        final_conv = ConvDesc.read(r)
        # `desc.cpp:580-591`
        if pre_bn.num_channels != regular_conv.in_channels:
            raise ModelFormatError(
                f"{name}: preBN.numChannels ({pre_bn.num_channels}) != "
                f"regularConv.inChannels ({regular_conv.in_channels})"
            )
        if mid_bn.num_channels != regular_conv.out_channels:
            raise ModelFormatError(
                f"{name}: midBN.numChannels ({mid_bn.num_channels}) != "
                f"regularConv.outChannels ({regular_conv.out_channels})"
            )
        if mid_bn.num_channels != final_conv.in_channels:
            raise ModelFormatError(
                f"{name}: midBN.numChannels ({mid_bn.num_channels}) != "
                f"finalConv.inChannels ({final_conv.in_channels})"
            )
        return cls(name, pre_bn, pre_activation, regular_conv, mid_bn, mid_activation, final_conv)

    def write(self, w: _Writer, model_version: int) -> None:
        w.token(self.name)
        self.pre_bn.write(w)
        self.pre_activation.write(w, model_version)
        self.regular_conv.write(w)
        self.mid_bn.write(w)
        self.mid_activation.write(w, model_version)
        self.final_conv.write(w)


@dataclass
class GlobalPoolingResidualBlockDesc:
    """`GlobalPoolingResidualBlockDesc`（`desc.h:165-196`）= `gpool_block`。"""

    name: str
    pre_bn: BatchNormDesc
    pre_activation: ActivationDesc
    regular_conv: ConvDesc
    gpool_conv: ConvDesc
    gpool_bn: BatchNormDesc
    gpool_activation: ActivationDesc
    gpool_to_bias_mul: MatMulDesc
    mid_bn: BatchNormDesc
    mid_activation: ActivationDesc
    final_conv: ConvDesc

    @property
    def num_parameters(self) -> int:
        return (
            self.pre_bn.num_parameters
            + self.regular_conv.num_parameters
            + self.gpool_conv.num_parameters
            + self.gpool_bn.num_parameters
            + self.gpool_to_bias_mul.num_parameters
            + self.mid_bn.num_parameters
            + self.final_conv.num_parameters
        )

    @property
    def num_file_floats(self) -> int:
        return (
            self.pre_bn.num_file_floats
            + self.regular_conv.num_file_floats
            + self.gpool_conv.num_file_floats
            + self.gpool_bn.num_file_floats
            + self.gpool_to_bias_mul.num_file_floats
            + self.mid_bn.num_file_floats
            + self.final_conv.num_file_floats
        )

    @classmethod
    def read(cls, r: _Reader, model_version: int) -> "GlobalPoolingResidualBlockDesc":
        name = r.token()
        pre_bn = BatchNormDesc.read(r)
        pre_activation = ActivationDesc.read(r, model_version)
        regular_conv = ConvDesc.read(r)
        gpool_conv = ConvDesc.read(r)
        gpool_bn = BatchNormDesc.read(r)
        gpool_activation = ActivationDesc.read(r, model_version)
        gpool_to_bias_mul = MatMulDesc.read(r)
        mid_bn = BatchNormDesc.read(r)
        mid_activation = ActivationDesc.read(r, model_version)
        final_conv = ConvDesc.read(r)

        # `desc.cpp:670-700`
        if pre_bn.num_channels != regular_conv.in_channels:
            raise ModelFormatError(
                f"{name}: preBN.numChannels ({pre_bn.num_channels}) != "
                f"regularConv.inChannels ({regular_conv.in_channels})"
            )
        if pre_bn.num_channels != gpool_conv.in_channels:
            raise ModelFormatError(
                f"{name}: preBN.numChannels ({pre_bn.num_channels}) != "
                f"gpoolConv.inChannels ({gpool_conv.in_channels})"
            )
        if gpool_bn.num_channels != gpool_conv.out_channels:
            raise ModelFormatError(
                f"{name}: gpoolBN.numChannels ({gpool_bn.num_channels}) != "
                f"gpoolConv.outChannels ({gpool_conv.out_channels})"
            )
        if gpool_bn.num_channels * 3 != gpool_to_bias_mul.in_channels:
            raise ModelFormatError(
                f"{name}: gpoolBN.numChannels * 3 ({gpool_bn.num_channels * 3}) != "
                f"gpoolToBiasMul.inChannels ({gpool_to_bias_mul.in_channels})"
            )
        if mid_bn.num_channels != regular_conv.out_channels:
            raise ModelFormatError(
                f"{name}: midBN.numChannels ({mid_bn.num_channels}) != "
                f"regularConv.outChannels ({regular_conv.out_channels})"
            )
        if mid_bn.num_channels != gpool_to_bias_mul.out_channels:
            raise ModelFormatError(
                f"{name}: midBN.numChannels ({mid_bn.num_channels}) != "
                f"gpoolToBiasMul.outChannels ({gpool_to_bias_mul.out_channels})"
            )
        if mid_bn.num_channels != final_conv.in_channels:
            raise ModelFormatError(
                f"{name}: midBN.numChannels ({mid_bn.num_channels}) != "
                f"finalConv.inChannels ({final_conv.in_channels})"
            )
        return cls(
            name,
            pre_bn,
            pre_activation,
            regular_conv,
            gpool_conv,
            gpool_bn,
            gpool_activation,
            gpool_to_bias_mul,
            mid_bn,
            mid_activation,
            final_conv,
        )

    def write(self, w: _Writer, model_version: int) -> None:
        w.token(self.name)
        self.pre_bn.write(w)
        self.pre_activation.write(w, model_version)
        self.regular_conv.write(w)
        self.gpool_conv.write(w)
        self.gpool_bn.write(w)
        self.gpool_activation.write(w, model_version)
        self.gpool_to_bias_mul.write(w)
        self.mid_bn.write(w)
        self.mid_activation.write(w, model_version)
        self.final_conv.write(w)


@dataclass
class NestedBottleneckDesc:
    """`NestedBottleneckResidualBlockDesc`（`desc.h:198-232`）= `nested_bottleneck_block`。

    ⚠ 坑 #8：**递归**。`blocks` 里每个子块自己还带 tag（`desc.cpp:799`）。
    """

    name: str
    num_blocks: int
    pre_bn: BatchNormDesc
    pre_activation: ActivationDesc
    pre_conv: ConvDesc
    blocks: List["Block"]
    post_bn: BatchNormDesc
    post_activation: ActivationDesc
    post_conv: ConvDesc

    @property
    def num_parameters(self) -> int:
        return (
            self.pre_bn.num_parameters
            + self.pre_conv.num_parameters
            + sum(b.num_parameters for b in self.blocks)
            + self.post_bn.num_parameters
            + self.post_conv.num_parameters
        )

    @property
    def num_file_floats(self) -> int:
        return (
            self.pre_bn.num_file_floats
            + self.pre_conv.num_file_floats
            + sum(b.num_file_floats for b in self.blocks)
            + self.post_bn.num_file_floats
            + self.post_conv.num_file_floats
        )

    @classmethod
    def read(cls, r: _Reader, model_version: int) -> "NestedBottleneckDesc":
        name = r.token()
        num_blocks = r.integer(f"{name}: numBlocks")
        if num_blocks < 1:
            raise ModelFormatError(
                f"{name}: nested bottleneck res block num blocks must be positive"
            )

        pre_bn = BatchNormDesc.read(r)
        pre_activation = ActivationDesc.read(r, model_version)
        pre_conv = ConvDesc.read(r)

        # `desc.cpp:799`：注意传的是 preConv.outChannels 作为子块的 trunkNumChannels
        blocks = read_block_stack(r, model_version, name, num_blocks, pre_conv.out_channels)

        post_bn = BatchNormDesc.read(r)
        post_activation = ActivationDesc.read(r, model_version)
        post_conv = ConvDesc.read(r)

        # `desc.cpp:805-816`
        if pre_bn.num_channels != pre_conv.in_channels:
            raise ModelFormatError(
                f"{name}: preBN.numChannels ({pre_bn.num_channels}) != "
                f"preConv.inChannels ({pre_conv.in_channels})"
            )
        if post_bn.num_channels != pre_conv.out_channels:
            raise ModelFormatError(
                f"{name}: postBN.numChannels ({post_bn.num_channels}) != "
                f"preConv.outChannels ({pre_conv.out_channels})"
            )
        if post_bn.num_channels != post_conv.in_channels:
            raise ModelFormatError(
                f"{name}: postBN.numChannels ({post_bn.num_channels}) != "
                f"postConv.inChannels ({post_conv.in_channels})"
            )
        return cls(
            name,
            num_blocks,
            pre_bn,
            pre_activation,
            pre_conv,
            blocks,
            post_bn,
            post_activation,
            post_conv,
        )

    def write(self, w: _Writer, model_version: int) -> None:
        w.token(self.name)
        w.token(self.num_blocks)
        self.pre_bn.write(w)
        self.pre_activation.write(w, model_version)
        self.pre_conv.write(w)
        if len(self.blocks) != self.num_blocks:
            raise ModelFormatError(
                f"{self.name}: numBlocks says {self.num_blocks} but {len(self.blocks)} "
                "child blocks are present"
            )
        for b in self.blocks:
            b.write(w, model_version)
        self.post_bn.write(w)
        self.post_activation.write(w, model_version)
        self.post_conv.write(w)


_AnyBlockDesc = Union[
    ResidualBlockDesc,
    GlobalPoolingResidualBlockDesc,
    NestedBottleneckDesc,
    TransformerAttentionDesc,
    TransformerFFNDesc,
]


@dataclass
class Block:
    """块栈里的一项：**tag + 载荷**。

    tag 必须显式存着，因为 `parseResidualBlockStack`（`desc.cpp:1455-1459`）是先读
    tag 再按 tag 分派，序列化时也要先写 tag。未知 tag 直接抛
    `UnsupportedBlockKindError`（`desc.cpp:1558-1559`）。
    """

    kind: str
    desc: _AnyBlockDesc

    @property
    def num_parameters(self) -> int:
        return self.desc.num_parameters

    @property
    def num_file_floats(self) -> int:
        return self.desc.num_file_floats

    def write(self, w: _Writer, model_version: int) -> None:
        w.token(self.kind)
        self.desc.write(w, model_version)  # type: ignore[union-attr]


def read_block(
    r: _Reader,
    model_version: int,
    parent_name: str,
    trunk_num_channels: int,
) -> Block:
    """读**一个**块：先读 tag 字符串，再按 tag 分派（`desc.cpp:1455-1559`）。

    这里刻意写成和 `parseResidualBlockStack` 一样的 if/elif 链，而不是查表 + 动态调用 ——
    一是形状一模一样、方便逐行对照官方源码，二是每个分支各自绑定局部变量，
    类型收窄干净（mypy 不用一串 ignore）。
    """
    kind = r.token()

    if kind == "ordinary_block":
        ordinary = ResidualBlockDesc.read(r, model_version)
        if ordinary.pre_bn.num_channels != trunk_num_channels:
            raise _trunk_mismatch(
                parent_name, ordinary.name, f"preBN.numChannels ({ordinary.pre_bn.num_channels})",
                trunk_num_channels,
            )
        if ordinary.final_conv.out_channels != trunk_num_channels:
            raise _trunk_mismatch(
                parent_name, ordinary.name,
                f"finalConv.outChannels ({ordinary.final_conv.out_channels})", trunk_num_channels,
            )
        return Block(kind, ordinary)

    if kind == "gpool_block":
        gpool = GlobalPoolingResidualBlockDesc.read(r, model_version)
        if gpool.pre_bn.num_channels != trunk_num_channels:
            raise _trunk_mismatch(
                parent_name, gpool.name, f"preBN.numChannels ({gpool.pre_bn.num_channels})",
                trunk_num_channels,
            )
        if gpool.final_conv.out_channels != trunk_num_channels:
            raise _trunk_mismatch(
                parent_name, gpool.name,
                f"finalConv.outChannels ({gpool.final_conv.out_channels})", trunk_num_channels,
            )
        return Block(kind, gpool)

    if kind == "nested_bottleneck_block":
        nested = NestedBottleneckDesc.read(r, model_version)
        if nested.pre_bn.num_channels != trunk_num_channels:
            raise _trunk_mismatch(
                parent_name, nested.name, f"preBN.numChannels ({nested.pre_bn.num_channels})",
                trunk_num_channels,
            )
        if nested.post_conv.out_channels != trunk_num_channels:
            raise _trunk_mismatch(
                parent_name, nested.name,
                f"postConv.outChannels ({nested.post_conv.out_channels})", trunk_num_channels,
            )
        return Block(kind, nested)

    if kind == "transformer_attention_block":
        # transformer 块不带 modelVersion：它们内部没有 ActivationLayerDesc
        attn = TransformerAttentionDesc.read(r)
        if attn.q_proj.in_channels != trunk_num_channels:
            raise _trunk_mismatch(
                parent_name, attn.name, f"qProj.inChannels ({attn.q_proj.in_channels})",
                trunk_num_channels,
            )
        if attn.out_proj.out_channels != trunk_num_channels:
            raise _trunk_mismatch(
                parent_name, attn.name,
                f"outProj.outChannels ({attn.out_proj.out_channels})", trunk_num_channels,
            )
        return Block(kind, attn)

    if kind == "transformer_ffn_block":
        ffn = TransformerFFNDesc.read(r)
        if ffn.num_channels != trunk_num_channels:
            raise _trunk_mismatch(
                parent_name, ffn.name, f"numChannels ({ffn.num_channels})", trunk_num_channels,
            )
        return Block(kind, ffn)

    # `desc.cpp:1558-1559`：未知 tag 直接抛，绝不猜
    raise UnsupportedBlockKindError(f"{parent_name}: found unknown block kind: {kind}")


def _trunk_mismatch(
    parent_name: str, block_name: str, detail: str, trunk_num_channels: int
) -> ModelFormatError:
    return ModelFormatError(
        f"{parent_name}: {block_name}: {detail} != trunkNumChannels ({trunk_num_channels})"
    )


def _unused_bn() -> BatchNormDesc:
    """trunk tip 是 RMSNorm 时，BatchNormDesc 在文件里不存在 —— 造一个"空"的占位。

    它带 `num_channels == 0` 和四个空数组，所以 `num_parameters` / `num_file_floats`
    都是 0，`TrunkDesc.getNumParameters` 那种"两个都加起来也安全"的写法仍然成立
    （`desc.cpp:1907`）。
    """
    return BatchNormDesc(
        "unused_trunk_tip_bn", 0, TextFloat.of(1.0), False, False,
        _EMPTY_F32, _EMPTY_F32, _EMPTY_F32, _EMPTY_F32,
    )


def _unused_rmsnorm() -> RMSNormDesc:
    """trunk tip 是 BatchNorm 时，RMSNormLayerDesc 的占位（同样全空）。"""
    return RMSNormDesc(
        "unused_trunk_tip_rmsnorm", 0, TextFloat.of(1.0), False, 0, _EMPTY_F32, _EMPTY_F32
    )


def read_block_stack(
    r: _Reader, model_version: int, parent_name: str, num_blocks: int, trunk_num_channels: int
) -> List[Block]:
    """`parseResidualBlockStack`（`desc.cpp:1446-1564`）。"""
    return [
        read_block(r, model_version, parent_name, trunk_num_channels) for _ in range(num_blocks)
    ]


# ---------------------------------------------------------------------------
# trunk / heads / model
# ---------------------------------------------------------------------------


@dataclass
class SGFMetadataEncoderDesc:
    """`SGFMetadataEncoderDesc`（`desc.h:347-371`）。

    只在 `meta_encoder_version > 0` 时存在（`desc.cpp:1731`）。写入端把 feature mask
    折进第一个 matmul（`export_model_pytorch.py:667`），所以文件里看不出 mask。
    """

    name: str
    num_input_meta_channels: int
    mul1: MatMulDesc
    bias1: MatBiasDesc
    act1: ActivationDesc
    mul2: MatMulDesc
    bias2: MatBiasDesc
    act2: ActivationDesc
    mul3: MatMulDesc

    @property
    def num_parameters(self) -> int:
        return (
            self.mul1.num_parameters
            + self.bias1.num_parameters
            + self.mul2.num_parameters
            + self.bias2.num_parameters
            + self.mul3.num_parameters
        )

    @property
    def num_file_floats(self) -> int:
        return (
            self.mul1.num_file_floats
            + self.bias1.num_file_floats
            + self.mul2.num_file_floats
            + self.bias2.num_file_floats
            + self.mul3.num_file_floats
        )

    @classmethod
    def read(cls, r: _Reader, model_version: int, meta_encoder_version: int) -> "SGFMetadataEncoderDesc":
        name = r.token()
        num_input_meta_channels = r.integer(f"{name}: numInputMetaChannels")
        # `desc.cpp:1584-1588`
        expected = NUM_INPUT_META_CHANNELS.get(meta_encoder_version)
        if expected is None:
            raise UnsupportedModelVersionError(
                f"{name}: metaEncoderVersion {meta_encoder_version} is not implemented"
            )
        if num_input_meta_channels != expected:
            raise ModelFormatError(
                f"{name}: number of in channels ({num_input_meta_channels}) did not match "
                f"expected ({expected})"
            )

        mul1 = MatMulDesc.read(r)
        bias1 = MatBiasDesc.read(r)
        act1 = ActivationDesc.read(r, model_version)
        mul2 = MatMulDesc.read(r)
        bias2 = MatBiasDesc.read(r)
        act2 = ActivationDesc.read(r, model_version)
        mul3 = MatMulDesc.read(r)

        # `desc.cpp:1601-1618`
        if mul1.out_channels != bias1.num_channels:
            raise ModelFormatError(
                f"{name}: mul1.outChannels ({mul1.out_channels}) != "
                f"bias1.numChannels ({bias1.num_channels})"
            )
        if mul2.in_channels != mul1.out_channels:
            raise ModelFormatError(
                f"{name}: mul2.inChannels ({mul2.in_channels}) != mul1.outChannels ({mul1.out_channels})"
            )
        if mul2.out_channels != bias2.num_channels:
            raise ModelFormatError(
                f"{name}: mul2.outChannels ({mul2.out_channels}) != "
                f"bias2.numChannels ({bias2.num_channels})"
            )
        if mul2.out_channels != mul3.in_channels:
            raise ModelFormatError(
                f"{name}: mul2.outChannels ({mul2.out_channels}) != mul3.inChannels ({mul3.in_channels})"
            )
        return cls(name, num_input_meta_channels, mul1, bias1, act1, mul2, bias2, act2, mul3)

    def write(self, w: _Writer, model_version: int) -> None:
        w.token(self.name)
        w.token(self.num_input_meta_channels)
        self.mul1.write(w)
        self.bias1.write(w)
        self.act1.write(w, model_version)
        self.mul2.write(w)
        self.bias2.write(w)
        self.act2.write(w, model_version)
        self.mul3.write(w)


@dataclass
class TrunkDesc:
    """`TrunkDesc`（`desc.h:380-421`）。

    ⚠ 坑 #9 / #11：通道数共 6 个（numBlocks 之后），第 4 个是源码里标 `//unused` 的
    `dilated_num_channels`，但它在文件里，必须原样往返。写入端填 `c_gpool`
    （`export_model_pytorch.py:682`），第 3 个 `regular_num_channels` 填 `c_mid - c_gpool`。

    ⚠ `trunk_norm_kind` 决定 trunk tip 是 `BatchNormDesc` 还是 `RMSNormDesc`
    （`desc.cpp:1752-1765`），两者在文件里互斥。
    """

    name: str
    num_blocks: int
    trunk_num_channels: int
    mid_num_channels: int
    regular_num_channels: int
    dilated_num_channels: int
    gpool_num_channels: int
    trunk_norm_kind: int
    trunk_options: Tuple[int, ...]
    initial_conv: ConvDesc
    initial_matmul: MatMulDesc
    blocks: List[Block]
    trunk_tip_activation: ActivationDesc
    trunk_tip_bn: BatchNormDesc
    trunk_tip_rmsnorm: RMSNormDesc
    sgf_metadata_encoder: Optional[SGFMetadataEncoderDesc] = None

    @property
    def num_parameters(self) -> int:
        total = self.initial_conv.num_parameters + self.initial_matmul.num_parameters
        if self.sgf_metadata_encoder is not None:
            total += self.sgf_metadata_encoder.num_parameters
        total += sum(b.num_parameters for b in self.blocks)
        # 没用上的那个 tip norm 权重数组是空的，两个都加也安全（`desc.cpp:1907`）
        total += self.trunk_tip_bn.num_parameters + self.trunk_tip_rmsnorm.num_parameters
        return total

    @property
    def num_file_floats(self) -> int:
        total = self.initial_conv.num_file_floats + self.initial_matmul.num_file_floats
        if self.sgf_metadata_encoder is not None:
            total += self.sgf_metadata_encoder.num_file_floats
        total += sum(b.num_file_floats for b in self.blocks)
        total += self.trunk_tip_bn.num_file_floats + self.trunk_tip_rmsnorm.num_file_floats
        return total

    @property
    def spatial_conv_depth(self) -> float:
        """只算普通 / gpool 残差块的卷积深度，transformer 与 nested 容器不计
        （对应 `desc.cpp` 的 `iterConvLayers` 语义）。"""
        total = self.initial_conv.spatial_conv_depth
        for b in self.blocks:
            if isinstance(b.desc, ResidualBlockDesc):
                total += (
                    b.desc.regular_conv.spatial_conv_depth
                    + b.desc.final_conv.spatial_conv_depth
                )
            elif isinstance(b.desc, GlobalPoolingResidualBlockDesc):
                total += (
                    b.desc.regular_conv.spatial_conv_depth
                    + b.desc.final_conv.spatial_conv_depth
                )
        return total

    def has_any_transformer_blocks(self) -> bool:
        return any(_block_has_transformer(b) for b in self.blocks)

    def has_any_nested_bottleneck_blocks(self) -> bool:
        return any(b.kind == "nested_bottleneck_block" for b in self.blocks)

    @classmethod
    def read(cls, r: _Reader, model_version: int, meta_encoder_version: int) -> "TrunkDesc":
        name = r.token()
        num_blocks = r.integer(f"{name}: numBlocks")
        trunk_num_channels = r.integer(f"{name}: trunkNumChannels")
        mid_num_channels = r.integer(f"{name}: midNumChannels")
        regular_num_channels = r.integer(f"{name}: regularNumChannels")
        dilated_num_channels = r.integer(f"{name}: dilatedNumChannels")
        gpool_num_channels = r.integer(f"{name}: gpoolNumChannels")

        trunk_norm_kind = TRUNK_NORM_KIND_STANDARD
        trunk_options: Tuple[int, ...] = ()
        if model_version >= 15:
            # `desc.cpp:1685-1702`
            trunk_norm_kind = r.integer(f"{name}: trunkNormKind")
            labels = "BCDEF"
            options = []
            for label in labels:
                val = r.integer(f"{name}: trunk option {label}")
                if val != 0:
                    raise ModelFormatError(
                        f"{name}: unknown/unsupported trunk option {label}: {val}"
                    )
                options.append(val)
            trunk_options = tuple(options)
            if trunk_norm_kind not in (TRUNK_NORM_KIND_STANDARD, TRUNK_NORM_KIND_RMSNORM):
                raise ModelFormatError(
                    f"{name}: unknown or unsupported trunk norm kind: {trunk_norm_kind}"
                )

        # `desc.cpp:1706-1711`
        if num_blocks < 1:
            raise ModelFormatError(f"{name}: trunk num blocks must be positive")
        if (
            trunk_num_channels <= 0
            or mid_num_channels <= 0
            or regular_num_channels <= 0
            or gpool_num_channels <= 0
        ):
            raise ModelFormatError(f"{name}: all numbers of channels must be positive")

        initial_conv = ConvDesc.read(r)
        if initial_conv.out_channels != trunk_num_channels:
            raise ModelFormatError(
                f"{name}: initialConv.outChannels ({initial_conv.out_channels}) != "
                f"trunkNumChannels ({trunk_num_channels})"
            )
        initial_matmul = MatMulDesc.read(r)
        if initial_matmul.out_channels != trunk_num_channels:
            raise ModelFormatError(
                f"{name}: initialMatMul.outChannels ({initial_matmul.out_channels}) != "
                f"trunkNumChannels ({trunk_num_channels})"
            )

        sgf_metadata_encoder: Optional[SGFMetadataEncoderDesc] = None
        if meta_encoder_version > 0:
            sgf_metadata_encoder = SGFMetadataEncoderDesc.read(
                r, model_version, meta_encoder_version
            )
            expected_meta = NUM_INPUT_META_CHANNELS[meta_encoder_version]
            if expected_meta != sgf_metadata_encoder.mul1.in_channels:
                raise ModelFormatError(
                    f"{name}: sgfMetadataEncoder.mul1.inChannels "
                    f"({sgf_metadata_encoder.mul1.in_channels}) != numInputMetaChannels "
                    f"({expected_meta})"
                )
            if sgf_metadata_encoder.mul3.out_channels != trunk_num_channels:
                raise ModelFormatError(
                    f"{name}: sgfMetadataEncoder.mul3.outChannels "
                    f"({sgf_metadata_encoder.mul3.out_channels}) != trunkNumChannels "
                    f"({trunk_num_channels})"
                )

        blocks = read_block_stack(r, model_version, name, num_blocks, trunk_num_channels)

        if trunk_norm_kind == TRUNK_NORM_KIND_STANDARD:
            trunk_tip_bn = BatchNormDesc.read(r)
            trunk_tip_rmsnorm = _unused_rmsnorm()
            if trunk_tip_bn.num_channels != trunk_num_channels:
                raise ModelFormatError(
                    f"{name}: trunkTipBN.numChannels ({trunk_tip_bn.num_channels}) != "
                    f"trunkNumChannels ({trunk_num_channels})"
                )
        else:
            trunk_tip_bn = _unused_bn()
            trunk_tip_rmsnorm = RMSNormDesc.read(r)
            if trunk_tip_rmsnorm.num_channels != trunk_num_channels:
                raise ModelFormatError(
                    f"{name}: trunkTipRMSNorm.numChannels ({trunk_tip_rmsnorm.num_channels}) != "
                    f"trunkNumChannels ({trunk_num_channels})"
                )

        trunk_tip_activation = ActivationDesc.read(r, model_version)

        return cls(
            name=name,
            num_blocks=num_blocks,
            trunk_num_channels=trunk_num_channels,
            mid_num_channels=mid_num_channels,
            regular_num_channels=regular_num_channels,
            dilated_num_channels=dilated_num_channels,
            gpool_num_channels=gpool_num_channels,
            trunk_norm_kind=trunk_norm_kind,
            trunk_options=trunk_options,
            initial_conv=initial_conv,
            initial_matmul=initial_matmul,
            blocks=blocks,
            trunk_tip_activation=trunk_tip_activation,
            trunk_tip_bn=trunk_tip_bn,
            trunk_tip_rmsnorm=trunk_tip_rmsnorm,
            sgf_metadata_encoder=sgf_metadata_encoder,
        )

    def write(self, w: _Writer, model_version: int, meta_encoder_version: int) -> None:
        w.token(self.name)
        w.token(self.num_blocks)
        w.token(self.trunk_num_channels)
        w.token(self.mid_num_channels)
        w.token(self.regular_num_channels)
        w.token(self.dilated_num_channels)
        w.token(self.gpool_num_channels)
        if model_version >= 15:
            w.token(self.trunk_norm_kind)
            for val in self.trunk_options:
                w.token(val)
        self.initial_conv.write(w)
        self.initial_matmul.write(w)
        if meta_encoder_version > 0:
            if self.sgf_metadata_encoder is None:
                raise ModelFormatError(
                    f"{self.name}: metaEncoderVersion is {meta_encoder_version} but there is "
                    "no sgf_metadata_encoder"
                )
            self.sgf_metadata_encoder.write(w, model_version)
        for b in self.blocks:
            b.write(w, model_version)
        if self.trunk_norm_kind == TRUNK_NORM_KIND_STANDARD:
            self.trunk_tip_bn.write(w)
        else:
            self.trunk_tip_rmsnorm.write(w)
        self.trunk_tip_activation.write(w, model_version)


def _block_has_transformer(block: Block) -> bool:
    """复刻 `blocksContainTransformerRecursive`（`desc.cpp:2736+`）。"""
    if block.kind in ("transformer_attention_block", "transformer_ffn_block"):
        return True
    if block.kind == "nested_bottleneck_block":
        return any(_block_has_transformer(b) for b in block.desc.blocks)  # type: ignore[union-attr]
    return False


@dataclass
class PolicyHeadDesc:
    """`PolicyHeadDesc`（`desc.h:423-457`）。

    ⚠ 坑 #9：`gpool_to_pass_bias` / `pass_activation` / `gpool_to_pass_mul2` 只在
    `model_version >= 15` 时在文件里（`desc.cpp:2096-2105`）。v<15 时 C++ 构造默认对象，
    我们也放"未使用"的空对象，且不写。
    """

    name: str
    policy_out_channels: int
    policy_options: Tuple[int, ...]
    p1_conv: ConvDesc
    g1_conv: ConvDesc
    g1_bn: BatchNormDesc
    g1_activation: ActivationDesc
    gpool_to_bias_mul: MatMulDesc
    p1_bn: BatchNormDesc
    p1_activation: ActivationDesc
    p2_conv: ConvDesc
    gpool_to_pass_mul: MatMulDesc
    gpool_to_pass_bias: MatBiasDesc
    pass_activation: ActivationDesc
    gpool_to_pass_mul2: MatMulDesc

    @property
    def num_parameters(self) -> int:
        return (
            self.p1_conv.num_parameters
            + self.g1_conv.num_parameters
            + self.g1_bn.num_parameters
            + self.gpool_to_bias_mul.num_parameters
            + self.p1_bn.num_parameters
            + self.p2_conv.num_parameters
            + self.gpool_to_pass_mul.num_parameters
            + self.gpool_to_pass_bias.num_parameters
            + self.gpool_to_pass_mul2.num_parameters
        )

    @property
    def num_file_floats(self) -> int:
        return (
            self.p1_conv.num_file_floats
            + self.g1_conv.num_file_floats
            + self.g1_bn.num_file_floats
            + self.gpool_to_bias_mul.num_file_floats
            + self.p1_bn.num_file_floats
            + self.p2_conv.num_file_floats
            + self.gpool_to_pass_mul.num_file_floats
            + self.gpool_to_pass_bias.num_file_floats
            + self.gpool_to_pass_mul2.num_file_floats
        )

    @classmethod
    def read(cls, r: _Reader, model_version: int) -> "PolicyHeadDesc":
        name = r.token()

        # `desc.cpp:2060-2073`
        if model_version >= 17:
            policy_out_channels = r.integer(f"{name}: policyOutChannels")
            if policy_out_channels not in (2, 4):
                raise ModelFormatError(
                    f"{name}: policy head got invalid policyOutChannels {policy_out_channels}"
                )
        elif model_version == 16:
            policy_out_channels = 4
        elif model_version >= 12:
            policy_out_channels = 2
        else:
            policy_out_channels = 1

        policy_options: Tuple[int, ...] = ()
        if model_version >= 17:
            # `desc.cpp:2076-2084`
            options = []
            for label in "ABC":
                val = r.integer(f"{name}: policy option {label}")
                if val != 0:
                    raise ModelFormatError(
                        f"{name}: unknown/unsupported policy option {label}: {val}"
                    )
                options.append(val)
            policy_options = tuple(options)

        p1_conv = ConvDesc.read(r)
        g1_conv = ConvDesc.read(r)
        g1_bn = BatchNormDesc.read(r)
        g1_activation = ActivationDesc.read(r, model_version)
        gpool_to_bias_mul = MatMulDesc.read(r)
        p1_bn = BatchNormDesc.read(r)
        p1_activation = ActivationDesc.read(r, model_version)
        p2_conv = ConvDesc.read(r)
        gpool_to_pass_mul = MatMulDesc.read(r)
        if model_version >= 15:
            gpool_to_pass_bias = MatBiasDesc.read(r)
            pass_activation = ActivationDesc.read(r, model_version)
            gpool_to_pass_mul2 = MatMulDesc.read(r)
        else:
            gpool_to_pass_bias = MatBiasDesc("unused_gpool_to_pass_bias", 0, _EMPTY_F32)
            pass_activation = ActivationDesc("unused_pass_activation", ACTIVATION_RELU)
            gpool_to_pass_mul2 = MatMulDesc("unused_gpool_to_pass_mul2", 0, 0, _EMPTY_F32)

        # `desc.cpp:2110-2156`
        if p1_conv.out_channels != p1_bn.num_channels:
            raise ModelFormatError(
                f"{name}: p1Conv.outChannels ({p1_conv.out_channels}) != "
                f"p1BN.numChannels ({p1_bn.num_channels})"
            )
        if g1_conv.out_channels != g1_bn.num_channels:
            raise ModelFormatError(
                f"{name}: g1Conv.outChannels ({g1_conv.out_channels}) != "
                f"g1BN.numChannels ({g1_bn.num_channels})"
            )
        if gpool_to_bias_mul.in_channels != g1_bn.num_channels * 3:
            raise ModelFormatError(
                f"{name}: gpoolToBiasMul.inChannels ({gpool_to_bias_mul.in_channels}) != "
                f"g1BN.numChannels*3 ({g1_bn.num_channels * 3})"
            )
        if gpool_to_bias_mul.out_channels != p1_bn.num_channels:
            raise ModelFormatError(
                f"{name}: gpoolToBiasMul.outChannels ({gpool_to_bias_mul.out_channels}) != "
                f"p1BN.numChannels ({p1_bn.num_channels})"
            )
        if p2_conv.in_channels != p1_bn.num_channels:
            raise ModelFormatError(
                f"{name}: p2Conv.inChannels ({p2_conv.in_channels}) != "
                f"p1BN.numChannels ({p1_bn.num_channels})"
            )
        if gpool_to_pass_mul.in_channels != g1_bn.num_channels * 3:
            raise ModelFormatError(
                f"{name}: gpoolToPassMul.inChannels ({gpool_to_pass_mul.in_channels}) != "
                f"g1BN.numChannels*3 ({g1_bn.num_channels * 3})"
            )
        if p2_conv.out_channels != policy_out_channels:
            raise ModelFormatError(
                f"{name}: p2Conv.outChannels ({p2_conv.out_channels}) != {policy_out_channels}"
            )
        if model_version >= 15:
            if gpool_to_pass_mul.out_channels != gpool_to_pass_bias.num_channels:
                raise ModelFormatError(
                    f"{name}: gpoolToPassMul.outChannels ({gpool_to_pass_mul.out_channels}) != "
                    f"gpoolToPassBias.numChannels ({gpool_to_pass_bias.num_channels})"
                )
            if gpool_to_pass_mul.out_channels != gpool_to_pass_mul2.in_channels:
                raise ModelFormatError(
                    f"{name}: gpoolToPassMul.outChannels ({gpool_to_pass_mul.out_channels}) != "
                    f"gpoolToPassMul2.inChannels ({gpool_to_pass_mul2.in_channels})"
                )
            if gpool_to_pass_mul.out_channels != p1_conv.out_channels:
                raise ModelFormatError(
                    f"{name}: gpoolToPassMul.outChannels ({gpool_to_pass_mul.out_channels}) != "
                    f"p1Conv.outChannels ({p1_conv.out_channels})"
                )
            if gpool_to_pass_mul2.out_channels != policy_out_channels:
                raise ModelFormatError(
                    f"{name}: gpoolToPassMul2.outChannels ({gpool_to_pass_mul2.out_channels}) "
                    f"!= {policy_out_channels}"
                )
        else:
            if gpool_to_pass_mul.out_channels != policy_out_channels:
                raise ModelFormatError(
                    f"{name}: gpoolToPassMul.outChannels ({gpool_to_pass_mul.out_channels}) "
                    f"!= {policy_out_channels}"
                )

        return cls(
            name=name,
            policy_out_channels=policy_out_channels,
            policy_options=policy_options,
            p1_conv=p1_conv,
            g1_conv=g1_conv,
            g1_bn=g1_bn,
            g1_activation=g1_activation,
            gpool_to_bias_mul=gpool_to_bias_mul,
            p1_bn=p1_bn,
            p1_activation=p1_activation,
            p2_conv=p2_conv,
            gpool_to_pass_mul=gpool_to_pass_mul,
            gpool_to_pass_bias=gpool_to_pass_bias,
            pass_activation=pass_activation,
            gpool_to_pass_mul2=gpool_to_pass_mul2,
        )

    def write(self, w: _Writer, model_version: int) -> None:
        w.token(self.name)
        if model_version >= 17:
            w.token(self.policy_out_channels)
            for val in self.policy_options:
                w.token(val)
        self.p1_conv.write(w)
        self.g1_conv.write(w)
        self.g1_bn.write(w)
        self.g1_activation.write(w, model_version)
        self.gpool_to_bias_mul.write(w)
        self.p1_bn.write(w)
        self.p1_activation.write(w, model_version)
        self.p2_conv.write(w)
        self.gpool_to_pass_mul.write(w)
        if model_version >= 15:
            self.gpool_to_pass_bias.write(w)
            self.pass_activation.write(w, model_version)
            self.gpool_to_pass_mul2.write(w)


@dataclass
class ValueHeadDesc:
    """`ValueHeadDesc`（`desc.h:459-491`）。

    ⚠ `sv3_mul` / `sv3_bias` 的宽度是版本决定的：v>=9 -> 6，v8 -> 4，v>=4 -> 2，更早 -> 1
    （`desc.cpp:2307-2330`）。宽度**不在文件里**，由 `model_version` 推导。
    """

    name: str
    value_options: Tuple[int, ...]
    v1_conv: ConvDesc
    v1_bn: BatchNormDesc
    v1_activation: ActivationDesc
    v2_mul: MatMulDesc
    v2_bias: MatBiasDesc
    v2_activation: ActivationDesc
    v3_mul: MatMulDesc
    v3_bias: MatBiasDesc
    sv3_mul: MatMulDesc
    sv3_bias: MatBiasDesc
    v_ownership_conv: ConvDesc

    @property
    def num_parameters(self) -> int:
        return (
            self.v1_conv.num_parameters
            + self.v1_bn.num_parameters
            + self.v2_mul.num_parameters
            + self.v2_bias.num_parameters
            + self.v3_mul.num_parameters
            + self.v3_bias.num_parameters
            + self.sv3_mul.num_parameters
            + self.sv3_bias.num_parameters
            + self.v_ownership_conv.num_parameters
        )

    @property
    def num_file_floats(self) -> int:
        return (
            self.v1_conv.num_file_floats
            + self.v1_bn.num_file_floats
            + self.v2_mul.num_file_floats
            + self.v2_bias.num_file_floats
            + self.v3_mul.num_file_floats
            + self.v3_bias.num_file_floats
            + self.sv3_mul.num_file_floats
            + self.sv3_bias.num_file_floats
            + self.v_ownership_conv.num_file_floats
        )

    @staticmethod
    def expected_score_value_channels(model_version: int) -> int:
        """`desc.cpp:2307-2330` 那张版本表。"""
        if model_version >= 9:
            return 6
        if model_version >= 8:
            return 4
        if model_version >= 4:
            return 2
        return 1

    @classmethod
    def read(cls, r: _Reader, model_version: int) -> "ValueHeadDesc":
        name = r.token()

        value_options: Tuple[int, ...] = ()
        if model_version >= 17:
            # `desc.cpp:2252-2260`
            options = []
            for label in "ABC":
                val = r.integer(f"{name}: value option {label}")
                if val != 0:
                    raise ModelFormatError(
                        f"{name}: unknown/unsupported value option {label}: {val}"
                    )
                options.append(val)
            value_options = tuple(options)

        v1_conv = ConvDesc.read(r)
        v1_bn = BatchNormDesc.read(r)
        v1_activation = ActivationDesc.read(r, model_version)
        v2_mul = MatMulDesc.read(r)
        v2_bias = MatBiasDesc.read(r)
        v2_activation = ActivationDesc.read(r, model_version)
        v3_mul = MatMulDesc.read(r)
        v3_bias = MatBiasDesc.read(r)
        sv3_mul = MatMulDesc.read(r)
        sv3_bias = MatBiasDesc.read(r)
        v_ownership_conv = ConvDesc.read(r)

        # `desc.cpp:2279-2339`
        if v1_conv.out_channels != v1_bn.num_channels:
            raise ModelFormatError(
                f"{name}: v1Conv.outChannels ({v1_conv.out_channels}) != "
                f"v1BN.numChannels ({v1_bn.num_channels})"
            )
        if v2_mul.in_channels != v1_bn.num_channels * 3:
            raise ModelFormatError(
                f"{name}: v2Mul.inChannels ({v2_mul.in_channels}) != "
                f"v1BN.numChannels*3 ({v1_bn.num_channels * 3})"
            )
        if v2_mul.out_channels != v2_bias.num_channels:
            raise ModelFormatError(
                f"{name}: v2Mul.outChannels ({v2_mul.out_channels}) != "
                f"v2Bias.numChannels ({v2_bias.num_channels})"
            )
        if v2_mul.out_channels != v3_mul.in_channels:
            raise ModelFormatError(
                f"{name}: v2Mul.outChannels ({v2_mul.out_channels}) != "
                f"v3Mul.inChannels ({v3_mul.in_channels})"
            )
        if v3_mul.out_channels != 3:
            raise ModelFormatError(f"{name}: v3Mul.outChannels ({v3_mul.out_channels}) != 3")
        if v3_bias.num_channels != 3:
            raise ModelFormatError(f"{name}: v3Bias.numChannels ({v3_bias.num_channels}) != 3")
        if sv3_mul.in_channels != v2_mul.out_channels:
            raise ModelFormatError(
                f"{name}: sv3Mul.inChannels ({sv3_mul.in_channels}) != "
                f"v2Mul.outChannels ({v2_mul.out_channels})"
            )
        expected_misc = cls.expected_score_value_channels(model_version)
        if sv3_mul.out_channels != expected_misc or sv3_bias.num_channels != expected_misc:
            raise ModelFormatError(
                f"{name}: sv3Mul/sv3Bias channels ({sv3_mul.out_channels}/"
                f"{sv3_bias.num_channels}) != {expected_misc} for model version {model_version}"
            )
        if v_ownership_conv.in_channels != v1_conv.out_channels:
            raise ModelFormatError(
                f"{name}: vOwnershipConv.inChannels ({v_ownership_conv.in_channels}) != "
                f"v1Conv.outChannels ({v1_conv.out_channels})"
            )
        if v_ownership_conv.out_channels != 1:
            raise ModelFormatError(
                f"{name}: vOwnershipConv.outChannels ({v_ownership_conv.out_channels}) != 1"
            )

        return cls(
            name=name,
            value_options=value_options,
            v1_conv=v1_conv,
            v1_bn=v1_bn,
            v1_activation=v1_activation,
            v2_mul=v2_mul,
            v2_bias=v2_bias,
            v2_activation=v2_activation,
            v3_mul=v3_mul,
            v3_bias=v3_bias,
            sv3_mul=sv3_mul,
            sv3_bias=sv3_bias,
            v_ownership_conv=v_ownership_conv,
        )

    def write(self, w: _Writer, model_version: int) -> None:
        w.token(self.name)
        if model_version >= 17:
            for val in self.value_options:
                w.token(val)
        self.v1_conv.write(w)
        self.v1_bn.write(w)
        self.v1_activation.write(w, model_version)
        self.v2_mul.write(w)
        self.v2_bias.write(w)
        self.v2_activation.write(w, model_version)
        self.v3_mul.write(w)
        self.v3_bias.write(w)
        self.sv3_mul.write(w)
        self.sv3_bias.write(w)
        self.v_ownership_conv.write(w)


@dataclass
class ModelDesc:
    """`ModelDesc`（`desc.h:508-597`）—— 整个 `.bin.gz`。

    ⚠ 坑 #10：`sha256` **不是文件的一部分**，是 loader 算出来传进来的
    （`desc.cpp:2467`）。所以它有字段、但不序列化；解析器也不从文件读它。

    后面 4 个 `num_*_channels` 是从 head 推导出来的（`desc.cpp:2608-2611`），
    存在属性里而不是字段 —— 文件里没有它们。
    """

    name: str
    sha256: str
    model_version: int
    num_input_channels: int
    num_input_global_channels: int
    td_score_multiplier: TextFloat
    score_mean_multiplier: TextFloat
    score_stdev_multiplier: TextFloat
    lead_multiplier: TextFloat
    variance_time_multiplier: TextFloat
    shortterm_value_error_multiplier: TextFloat
    shortterm_score_error_multiplier: TextFloat
    meta_encoder_version: int
    prefer_pass_alive_under_suicide_rules: bool
    prefer_exclude_territory_adjacent_to_atari: bool
    model_options: Tuple[int, ...]
    trunk: TrunkDesc
    policy_head: PolicyHeadDesc
    value_head: ValueHeadDesc

    # -- 派生属性（`desc.cpp:2608-2611`） --------------------------------

    @property
    def num_input_meta_channels(self) -> int:
        return NUM_INPUT_META_CHANNELS.get(self.meta_encoder_version, 0)

    @property
    def num_policy_channels(self) -> int:
        return self.policy_head.policy_out_channels

    @property
    def num_value_channels(self) -> int:
        return self.value_head.v3_mul.out_channels

    @property
    def num_score_value_channels(self) -> int:
        return self.value_head.sv3_mul.out_channels

    @property
    def num_ownership_channels(self) -> int:
        return self.value_head.v_ownership_conv.out_channels

    @property
    def num_parameters(self) -> int:
        """复刻 `ModelDesc::getNumParameters`（`desc.cpp:2703-2707`）。"""
        return self.trunk.num_parameters + self.policy_head.num_parameters + self.value_head.num_parameters

    @property
    def num_file_floats(self) -> int:
        return self.trunk.num_file_floats + self.policy_head.num_file_floats + self.value_head.num_file_floats

    @property
    def trunk_spatial_conv_depth(self) -> float:
        return self.trunk.spatial_conv_depth

    def has_any_transformer_blocks(self) -> bool:
        return self.trunk.has_any_transformer_blocks()

    def has_any_nested_bottleneck_blocks(self) -> bool:
        return self.trunk.has_any_nested_bottleneck_blocks()

    def get_short_info_string(self) -> str:
        """复刻 `getShortInfoString`（`desc.cpp:2719-2728`）。

        这是官方日志里 `Model name: b10c384h6nbttflrs (nbt transformer, 10545753 params)`
        那个括号内容的来源。
        """
        is_transformer = self.has_any_transformer_blocks()
        is_nbt = self.has_any_nested_bottleneck_blocks()
        if is_nbt:
            kind = "nbt transformer" if is_transformer else "nbt convnet"
        else:
            kind = "transformer" if is_transformer else "convnet"
        return f"{kind}, {self.num_parameters} params"

    # -- 读 -------------------------------------------------------------

    @classmethod
    def read(cls, r: _Reader, sha256: str = "") -> "ModelDesc":
        name = r.token()
        model_version = r.integer("model version")

        check_name_valid(name)
        # `desc.cpp:2474-2479`
        if model_version < 0:
            raise UnsupportedModelVersionError(
                f"This neural net has an invalid version ({model_version}); you probably "
                "specified the wrong file."
            )
        if model_version < OLDEST_MODEL_VERSION:
            raise UnsupportedModelVersionError(
                "This neural net is from an extremely old version of KataGo and is no "
                f"longer supported by the engine. Model version: {model_version}"
            )
        if model_version > LATEST_MODEL_VERSION:
            raise UnsupportedModelVersionError(
                "This neural net requires a newer KataGo version. Obtain a newer KataGo at "
                f"https://github.com/lightvector/KataGo. Model version: {model_version}"
            )

        num_input_channels = r.integer(f"{name}: numInputChannels")
        if num_input_channels <= 0:
            raise ModelFormatError(f"{name}: model numInputChannels must be positive")
        num_input_global_channels = r.integer(f"{name}: numInputGlobalChannels")
        if num_input_global_channels <= 0:
            raise ModelFormatError(f"{name}: model numInputGlobalChannels must be positive")

        # v>=13: 7 个 double（`desc.cpp:2493-2529`）。v<13 时用默认值，不在文件里。
        mult_names = (
            ("tdScoreMultiplier", "td_score_multiplier"),
            ("scoreMeanMultiplier", "score_mean_multiplier"),
            ("scoreStdevMultiplier", "score_stdev_multiplier"),
            ("leadMultiplier", "lead_multiplier"),
            ("varianceTimeMultiplier", "variance_time_multiplier"),
            ("shorttermValueErrorMultiplier", "shortterm_value_error_multiplier"),
            ("shorttermScoreErrorMultiplier", "shortterm_score_error_multiplier"),
        )
        mults: List[TextFloat] = []
        if model_version >= 13:
            for label, _ in mult_names:
                tf = r.text_float(f"{name}: model {label}")
                if not (tf.value > 0) or not np.isfinite(tf.value):
                    raise ModelFormatError(f"{name}: model {label} must be positive")
                mults.append(tf)
        else:
            # `ModelPostProcessParams()` 的默认值（`desc.cpp:2432` 附近）
            mults = [TextFloat(x) for x in _DEFAULT_POST_PROCESS_MULTIPLIERS]

        # v>=15（`desc.cpp:2534-2596`）
        meta_encoder_version = 0
        prefer_pass_alive = False
        prefer_exclude_territory = False
        model_options: Tuple[int, ...] = ()
        if model_version >= 15:
            meta_encoder_version = r.integer(f"{name}: metaEncoderVersion")
            if meta_encoder_version < 0:
                raise ModelFormatError(
                    f"{name}: model metaEncoderVersion unexpected value: {meta_encoder_version}"
                )
            if meta_encoder_version not in NUM_INPUT_META_CHANNELS:
                raise UnsupportedModelVersionError(
                    f"{name}: model metaEncoderVersion not implemented, you may need a "
                    f"newer KataGo version, value was: {meta_encoder_version}"
                )
            prefer_pass_alive = _read_strict_bool(r, f"{name}: preferPassAliveUnderSuicideRules")
            prefer_exclude_territory = _read_strict_bool(
                r, f"{name}: preferExcludeTerritoryAdjacentToAtari"
            )
            options = []
            for label in "DEFGH":
                val = r.integer(f"{name}: model option {label}")
                if val != 0:
                    raise ModelFormatError(
                        f"{name}: unknown/unsupported model option {label}: {val}"
                    )
                options.append(val)
            model_options = tuple(options)

        trunk = TrunkDesc.read(r, model_version, meta_encoder_version)
        policy_head = PolicyHeadDesc.read(r, model_version)
        value_head = ValueHeadDesc.read(r, model_version)

        # `desc.cpp:2616-2646`
        if num_input_channels != trunk.initial_conv.in_channels:
            raise ModelFormatError(
                f"{name}: numInputChannels ({num_input_channels}) != "
                f"trunk.initialConv.inChannels ({trunk.initial_conv.in_channels})"
            )
        if num_input_global_channels != trunk.initial_matmul.in_channels:
            raise ModelFormatError(
                f"{name}: numInputGlobalChannels ({num_input_global_channels}) != "
                f"trunk.initialMatMul.inChannels ({trunk.initial_matmul.in_channels})"
            )
        if trunk.trunk_num_channels != policy_head.p1_conv.in_channels:
            raise ModelFormatError(
                f"{name}: trunk.trunkNumChannels ({trunk.trunk_num_channels}) != "
                f"policyHead.p1Conv.inChannels ({policy_head.p1_conv.in_channels})"
            )
        if trunk.trunk_num_channels != policy_head.g1_conv.in_channels:
            raise ModelFormatError(
                f"{name}: trunk.trunkNumChannels ({trunk.trunk_num_channels}) != "
                f"policyHead.g1Conv.inChannels ({policy_head.g1_conv.in_channels})"
            )
        if trunk.trunk_num_channels != value_head.v1_conv.in_channels:
            raise ModelFormatError(
                f"{name}: trunk.trunkNumChannels ({trunk.trunk_num_channels}) != "
                f"valueHead.v1Conv.inChannels ({value_head.v1_conv.in_channels})"
            )

        return cls(
            name=name,
            sha256=sha256,
            model_version=model_version,
            num_input_channels=num_input_channels,
            num_input_global_channels=num_input_global_channels,
            td_score_multiplier=mults[0],
            score_mean_multiplier=mults[1],
            score_stdev_multiplier=mults[2],
            lead_multiplier=mults[3],
            variance_time_multiplier=mults[4],
            shortterm_value_error_multiplier=mults[5],
            shortterm_score_error_multiplier=mults[6],
            meta_encoder_version=meta_encoder_version,
            prefer_pass_alive_under_suicide_rules=prefer_pass_alive,
            prefer_exclude_territory_adjacent_to_atari=prefer_exclude_territory,
            model_options=model_options,
            trunk=trunk,
            policy_head=policy_head,
            value_head=value_head,
        )

    # -- 写 -------------------------------------------------------------

    def write(self, w: _Writer) -> None:
        w.token(self.name)
        w.token(self.model_version)
        w.token(self.num_input_channels)
        w.token(self.num_input_global_channels)
        if self.model_version >= 13:
            w.token(self.td_score_multiplier.text)
            w.token(self.score_mean_multiplier.text)
            w.token(self.score_stdev_multiplier.text)
            w.token(self.lead_multiplier.text)
            w.token(self.variance_time_multiplier.text)
            w.token(self.shortterm_value_error_multiplier.text)
            w.token(self.shortterm_score_error_multiplier.text)
        if self.model_version >= 15:
            w.token(self.meta_encoder_version)
            w.token(1 if self.prefer_pass_alive_under_suicide_rules else 0)
            w.token(1 if self.prefer_exclude_territory_adjacent_to_atari else 0)
            for val in self.model_options:
                w.token(val)
        self.trunk.write(w, self.model_version, self.meta_encoder_version)
        self.policy_head.write(w, self.model_version)
        self.value_head.write(w, self.model_version)


#: `ModelPostProcessParams()` 默认值，只在 model_version < 13 时用（那时不在文件里）
_DEFAULT_POST_PROCESS_MULTIPLIERS = ("20.0", "20.0", "20.0", "20.0", "40.0", "0.25", "30.0")


def _read_strict_bool(r: _Reader, what: str) -> bool:
    """`desc.cpp:2557-2574`：这两个标志**只接受 0 或 1**，其它值直接抛错。"""
    val = r.integer(what)
    if val == 0:
        return False
    if val == 1:
        return True
    raise ModelFormatError(f"{what}: unexpected value: {val}")


# ---------------------------------------------------------------------------
# 顶层 API
# ---------------------------------------------------------------------------


def maybe_gunzip(data: bytes) -> bytes:
    """`.bin` 与 `.bin.gz` 一视同仁（gzip magic `\\x1f\\x8b`）。"""
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def read_model_bytes(data: bytes, sha256: str = "") -> Tuple[ModelDesc, int]:
    """解析（自动 gunzip），返回 `(ModelDesc, 消耗的字节数)`。

    ⚠ **消耗的字节数通常是 `len - 1`**：文件最后一个字节是最后一个 `@BIN@` 块的
    尾随 `\\n`，C++ 与我们都不消费它（坑 #1）。这个返回值就是给 round-trip diff 用的。
    """
    raw = maybe_gunzip(data)
    digest = sha256 or hashlib.sha256(raw).hexdigest()
    r = _Reader(raw)
    model = ModelDesc.read(r, digest)
    return model, r.pos


def parse_model(data: bytes) -> ModelDesc:
    """只要模型，不要消耗字节数。"""
    return read_model_bytes(data)[0]


def serialize_model(model: ModelDesc) -> bytes:
    """反向序列化。`parse_model(serialize_model(m))` 必须与 `m` 等价，
    且 `serialize_model(parse_model(x)) == x` 对任何合法 `x` 成立。"""
    w = _Writer()
    model.write(w)
    return w.value()


def load_model_file(path: str) -> ModelDesc:
    """读 `.bin` / `.bin.gz`（`ModelDesc::loadFromFileMaybeGZipped` 的 Python 版）。"""
    with open(path, "rb") as f:
        return parse_model(f.read())


def save_model_file(model: ModelDesc, path: str) -> None:
    """写 `.bin` / `.bin.gz`（后缀是 `.gz` 就压缩；压缩后不可字节级比较，用 roundtrip）。"""
    payload = serialize_model(model)
    if path.lower().endswith(".gz"):
        # mtime=0 让同样内容产出同样字节，便于测试
        payload = gzip.compress(payload, mtime=0)
    with open(path, "wb") as f:
        f.write(payload)


def count_parameters(model: ModelDesc) -> int:
    """KataGo `getNumParameters` 口径（⚠ BN 只数 scale+bias，不数 mean+variance）。"""
    return model.num_parameters


def count_file_floats(model: ModelDesc) -> int:
    """文件里 float32 的**总个数**。

    与 `count_parameters` 的差 = 所有 BN 的 `2 * num_channels`（mean + variance）
    —— 这两个是 running stats，官方日志的 `params` 不算它们。
    """
    return model.num_file_floats


def _iter_float_arrays(obj: Any) -> Iterator[np.ndarray]:
    """结构化遍历 IR 里**每一个**非空 float 数组（含 `rope_freqs` 这种没有独立层的）。

    ⚠ 不能靠 `iter_layers`：那个只吐叶子"层"，attention 块本身不是叶子，
    它的 `rope_freqs` 就漏掉了。本函数按 dataclass 字段递归，与序列化器看到的完全一致。

    ⚠ 坑 #18：这里的顺序是**dataclass 字段声明顺序**，**不是落盘顺序**
    （`TransformerFFNDesc` 是 `linear1, linear2, linear_gate` vs 落盘的
    `linear1, linear_gate, linear2`）。所以本函数只可用于**顺序无关**的统计。
    要真实落盘顺序请用 `_Writer.floats` 探针。
    """
    if isinstance(obj, np.ndarray):
        if obj.size:
            yield obj
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_float_arrays(item)
        return
    fields = getattr(obj, "__dataclass_fields__", None)
    if fields is None:
        return
    for fname in fields:
        if fname in ("name", "kind"):
            continue
        yield from _iter_float_arrays(getattr(obj, fname))


def count_subnormals(model: ModelDesc) -> int:
    """文件里有多少个 fp32 次正规数（`abs(x)` 非零但 `< FLT_MIN`）。

    ⚠ KataGo 读的时候会**把它们冲成 0**（`desc.cpp:89-90`），我们为了 round-trip
    保留原值。这个数字告诉你"如果要做数值等价性比对，需要额外容忍多少差异"。

    实测 `kata1-tf2-b10c384-s2941M-d5872M.bin.gz`：**444,703 / 10,557,657 = 4.21%**，
    而且高度集中在 transformer FFN 的 `linear1` / `linear_gate` / `linear2`
    （个别块高达 37%），是 weight decay 之后死掉的 FFN 通道。
    """
    tiny = np.float32(np.finfo(np.float32).tiny)
    total = 0
    for arr in _iter_float_arrays(model):
        a = np.abs(arr)
        total += int(np.count_nonzero((a > 0) & (a < tiny)))
    return total


# ---------------------------------------------------------------------------
# 层清单（inspect 用）
# ---------------------------------------------------------------------------


def iter_layers(model: ModelDesc) -> Iterator[Tuple[str, str, Any]]:
    """深度优先遍历所有叶子层，产出 `(路径, 层类型名, 层对象)`。

    路径沿用官方命名习惯（`model.blocks.0.blockstack.1.q_proj`），便于和
    `export_model_pytorch.py` / TensorRT 日志对照。
    """

    def emit(path: str, layer: Any) -> Iterator[Tuple[str, str, Any]]:
        yield path, type(layer).__name__, layer

    def blocks(prefix: str, block: Block) -> Iterator[Tuple[str, str, Any]]:
        d = block.desc
        dname = d.name
        if isinstance(d, NestedBottleneckDesc):
            yield from emit(f"{prefix}.norm", d.pre_bn)
            yield from emit(f"{prefix}.act", d.pre_activation)
            yield from emit(f"{prefix}.conv", d.pre_conv)
            for i, sub in enumerate(d.blocks):
                yield from blocks(f"{prefix}.blockstack.{i}", sub)
            yield from emit(f"{prefix}.postnorm", d.post_bn)
            yield from emit(f"{prefix}.postact", d.post_activation)
            yield from emit(f"{prefix}.postconv", d.post_conv)
        elif isinstance(d, TransformerAttentionDesc):
            yield from emit(f"{prefix}.norm1", d.pre_ln)
            yield from emit(f"{prefix}.q_proj", d.q_proj)
            yield from emit(f"{prefix}.k_proj", d.k_proj)
            yield from emit(f"{prefix}.v_proj", d.v_proj)
            yield from emit(f"{prefix}.out_proj", d.out_proj)
        elif isinstance(d, TransformerFFNDesc):
            yield from emit(f"{prefix}.norm", d.pre_ln)
            yield from emit(f"{prefix}.ffn_linear1", d.linear1)
            if d.use_swiglu:
                yield from emit(f"{prefix}.ffn_linear_gate", d.linear_gate)
            yield from emit(f"{prefix}.ffn_linear2", d.linear2)
        elif isinstance(d, ResidualBlockDesc):
            yield from emit(f"{prefix}.normactconv1.norm", d.pre_bn)
            yield from emit(f"{prefix}.normactconv1.act", d.pre_activation)
            yield from emit(f"{prefix}.normactconv1.conv", d.regular_conv)
            yield from emit(f"{prefix}.normactconv2.norm", d.mid_bn)
            yield from emit(f"{prefix}.normactconv2.act", d.mid_activation)
            yield from emit(f"{prefix}.normactconv2.conv", d.final_conv)
        elif isinstance(d, GlobalPoolingResidualBlockDesc):
            yield from emit(f"{prefix}.normactconv1.norm", d.pre_bn)
            yield from emit(f"{prefix}.normactconv1.act", d.pre_activation)
            yield from emit(f"{prefix}.normactconv1.conv", d.regular_conv)
            yield from emit(f"{prefix}.normactconv1.convpool.conv1g", d.gpool_conv)
            yield from emit(f"{prefix}.normactconv1.convpool.normg", d.gpool_bn)
            yield from emit(f"{prefix}.normactconv1.convpool.actg", d.gpool_activation)
            yield from emit(f"{prefix}.normactconv1.convpool.linear_g", d.gpool_to_bias_mul)
            yield from emit(f"{prefix}.normactconv2.norm", d.mid_bn)
            yield from emit(f"{prefix}.normactconv2.act", d.mid_activation)
            yield from emit(f"{prefix}.normactconv2.conv", d.final_conv)
        else:  # pragma: no cover - read_block 保证了种类封闭
            raise ModelFormatError(f"unknown block payload {dname}")

    t = model.trunk
    yield from emit("model.conv_spatial", t.initial_conv)
    yield from emit("model.linear_global", t.initial_matmul)
    if t.sgf_metadata_encoder is not None:
        e = t.sgf_metadata_encoder
        yield from emit(f"{e.name}.mul1", e.mul1)
        yield from emit(f"{e.name}.bias1", e.bias1)
        yield from emit(f"{e.name}.act1", e.act1)
        yield from emit(f"{e.name}.mul2", e.mul2)
        yield from emit(f"{e.name}.bias2", e.bias2)
        yield from emit(f"{e.name}.act2", e.act2)
        yield from emit(f"{e.name}.mul3", e.mul3)
    for i, b in enumerate(t.blocks):
        yield from blocks(f"model.blocks.{i}", b)
    if t.trunk_norm_kind == TRUNK_NORM_KIND_STANDARD:
        yield from emit("model.norm_trunkfinal", t.trunk_tip_bn)
    else:
        yield from emit("model.norm_trunkfinal", t.trunk_tip_rmsnorm)
    yield from emit("model.act_trunkfinal", t.trunk_tip_activation)

    p = model.policy_head
    pn = p.name
    for suffix, layer in (
        (".conv1p", p.p1_conv),
        (".conv1g", p.g1_conv),
        (".biasg", p.g1_bn),
        (".actg", p.g1_activation),
        (".linear_g", p.gpool_to_bias_mul),
        (".bias2", p.p1_bn),
        (".act2", p.p1_activation),
        (".conv2p", p.p2_conv),
        (".linear_pass", p.gpool_to_pass_mul),
        (".linear_pass_bias", p.gpool_to_pass_bias),
        (".act_pass", p.pass_activation),
        (".linear_pass2", p.gpool_to_pass_mul2),
    ):
        if model.model_version >= 15 or not suffix.startswith((".linear_pass_bias", ".act_pass", ".linear_pass2")):
            yield from emit(pn + suffix, layer)

    v = model.value_head
    vn = v.name
    for suffix, layer in (
        (".conv1", v.v1_conv),
        (".bias1", v.v1_bn),
        (".act1", v.v1_activation),
        (".linear2", v.v2_mul),
        (".bias2", v.v2_bias),
        (".act2", v.v2_activation),
        (".linear_valuehead", v.v3_mul),
        (".bias_valuehead", v.v3_bias),
        (".linear_miscvaluehead", v.sv3_mul),
        (".bias_miscvaluehead", v.sv3_bias),
        (".conv_ownership", v.v_ownership_conv),
    ):
        yield from emit(vn + suffix, layer)


@dataclass
class RoundtripReport:
    """round-trip 结果。`identical` 为真时其余字段无意义。"""

    identical: bool
    original_length: int
    reserialized_length: int
    consumed_length: int
    first_diff_offset: Optional[int]
    first_diff_expected: Optional[str]
    first_diff_actual: Optional[str]
    num_diff_bytes: int
    trailing_bytes: bytes

    @property
    def summary(self) -> str:
        if self.identical:
            return (
                f"字节完全一致 (byte-identical): {self.original_length} bytes, "
                f"parser consumed {self.consumed_length}, "
                f"trailing {len(self.trailing_bytes)} byte(s) = {self.trailing_bytes!r}"
            )
        return (
            f"不一致 (MISMATCH): 长度 {self.original_length} vs {self.reserialized_length}, "
            f"第一个差异在第 {self.first_diff_offset} 字节, "
            f"期望 {self.first_diff_expected!r} 实际 {self.first_diff_actual!r}, "
            f"共 {self.num_diff_bytes} 字节不同"
        )


def roundtrip_diff(original_raw: bytes) -> Tuple[ModelDesc, RoundtripReport]:
    """解析 -> 重新序列化 -> byte 级 diff。返回 `(模型, 报告)`。"""
    model, consumed = read_model_bytes(original_raw)
    reserialized = serialize_model(model)
    reference = maybe_gunzip(original_raw)

    if reserialized == reference:
        return model, RoundtripReport(
            identical=True,
            original_length=len(reference),
            reserialized_length=len(reserialized),
            consumed_length=consumed,
            first_diff_offset=None,
            first_diff_expected=None,
            first_diff_actual=None,
            num_diff_bytes=0,
            trailing_bytes=reference[consumed:],
        )

    limit = min(len(reference), len(reserialized))
    first = next((i for i in range(limit) if reference[i] != reserialized[i]), limit)
    num_diff = sum(
        1 for i in range(limit) if reference[i] != reserialized[i]
    ) + abs(len(reference) - len(reserialized))
    lo = max(0, first - 24)
    return model, RoundtripReport(
        identical=False,
        original_length=len(reference),
        reserialized_length=len(reserialized),
        consumed_length=consumed,
        first_diff_offset=first,
        first_diff_expected=repr(reference[lo : first + 24]),
        first_diff_actual=repr(reserialized[lo : first + 24]),
        num_diff_bytes=num_diff,
        trailing_bytes=reference[consumed:],
    )


def summarize(model: ModelDesc) -> str:
    """给 `inspect` 打印的层清单 / 结构摘要。"""
    lines: List[str] = []
    t = model.trunk
    add = lines.append
    add(f"model name            : {model.name}")
    add(f"sha256 (not in file)  : {model.sha256}")
    add(f"modelVersion          : {model.model_version}")
    add(f"inputsVersion         : {_inputs_version(model.model_version)}")
    add(f"numInputChannels      : {model.num_input_channels}")
    add(f"numInputGlobalChannels: {model.num_input_global_channels}")
    add(f"metaEncoderVersion    : {model.meta_encoder_version} "
        f"(numInputMetaChannels={model.num_input_meta_channels})")
    add(f"preferPassAlive       : {int(model.prefer_pass_alive_under_suicide_rules)}")
    add(f"preferExcludeTerritory: {int(model.prefer_exclude_territory_adjacent_to_atari)}")
    add("postProcessParams     : " + ", ".join(
        f"{k}={getattr(model, k).text}"
        for k in (
            "td_score_multiplier",
            "score_mean_multiplier",
            "score_stdev_multiplier",
            "lead_multiplier",
            "variance_time_multiplier",
            "shortterm_value_error_multiplier",
            "shortterm_score_error_multiplier",
        )
    ))
    add("")
    add("trunk")
    add(f"  name                : {t.name}")
    add(f"  numBlocks           : {t.num_blocks}")
    add(f"  trunkNumChannels    : {t.trunk_num_channels}")
    add(f"  midNumChannels      : {t.mid_num_channels}")
    add(f"  regularNumChannels  : {t.regular_num_channels}")
    add(f"  dilatedNumChannels  : {t.dilated_num_channels}   (源码标 //unused，但在文件里)")
    add(f"  gpoolNumChannels    : {t.gpool_num_channels}")
    add(f"  trunkNormKind       : {t.trunk_norm_kind}")
    add(f"  trunk options (5)   : {list(t.trunk_options)}")
    add(f"  trunk tip norm      : {'BatchNorm' if t.trunk_norm_kind == TRUNK_NORM_KIND_STANDARD else 'RMSNorm'}")
    add(f"  sgf metadata encoder: {'yes' if t.sgf_metadata_encoder else 'no'}")
    add(f"  hasAnyTransformer   : {int(t.has_any_transformer_blocks())}")
    add(f"  hasAnyNestedBottle  : {int(t.has_any_nested_bottleneck_blocks())}")

    def dump_blocks(prefix: str, blocks: Sequence[Block], indent: str) -> None:
        for i, b in enumerate(blocks):
            d = b.desc
            head = f"{indent}[{i}] {b.kind} {d.name}"
            if isinstance(d, NestedBottleneckDesc):
                add(f"{head}  (numBlocks={d.num_blocks}, "
                    f"pre {d.pre_conv.conv_y}x{d.pre_conv.conv_x} "
                    f"{d.pre_conv.in_channels}->{d.pre_conv.out_channels}, "
                    f"post {d.post_conv.conv_y}x{d.post_conv.conv_x} "
                    f"{d.post_conv.in_channels}->{d.post_conv.out_channels})")
                dump_blocks(f"{prefix}.{i}", d.blocks, indent + "    ")
            elif isinstance(d, TransformerAttentionDesc):
                add(f"{head}  (heads={d.num_heads}, kvHeads={d.num_kv_heads}, "
                    f"qDim={d.q_head_dim}, vDim={d.v_head_dim}, "
                    f"rope={int(d.use_rope)}, learnableRope={int(d.learnable_rope)})")
            elif isinstance(d, TransformerFFNDesc):
                add(f"{head}  (numChannels={d.num_channels}, ffnChannels={d.ffn_channels}, "
                    f"swiglu={int(d.use_swiglu)})")
            elif isinstance(d, ResidualBlockDesc):
                add(f"{head}  (mid={d.regular_conv.out_channels})")
            elif isinstance(d, GlobalPoolingResidualBlockDesc):
                add(f"{head}  (mid={d.regular_conv.out_channels}, "
                    f"gpool={d.gpool_conv.out_channels})")

    dump_blocks("model.blocks", t.blocks, "  ")
    add("")
    add("policy head")
    add(f"  policyOutChannels   : {model.num_policy_channels}")
    add(f"  policy options (3)  : {list(model.policy_head.policy_options)}")
    add("")
    add("value head")
    add(f"  numValueChannels    : {model.num_value_channels}")
    add(f"  numScoreValueChannels: {model.num_score_value_channels}")
    add(f"  numOwnershipChannels: {model.num_ownership_channels}")
    add(f"  value options (3)   : {list(model.value_head.value_options)}")
    add("")
    add("counts")
    add(f"  floats in file      : {count_file_floats(model)}")
    add(f"  params (KataGo)     : {count_parameters(model)}")
    add(f"  subnormal floats    : {count_subnormals(model)}")
    add(f"  getShortInfoString  : {model.get_short_info_string()}")
    add("")
    add("layers")
    for path, kind, layer in iter_layers(model):
        shape = _layer_shape(layer)
        add(f"  {path:<52} {kind:<28} {shape}")
    return "\n".join(lines)


def _layer_shape(layer: Any) -> str:
    w = getattr(layer, "weights", None)
    if isinstance(w, np.ndarray) and w.size:
        return f"weights[{w.size}]"
    if isinstance(layer, ConvDesc):
        return f"{layer.conv_y}x{layer.conv_x} {layer.in_channels}->{layer.out_channels} dil={layer.dilation_y}"
    if isinstance(layer, MatMulDesc):
        return f"{layer.in_channels}->{layer.out_channels}"
    if isinstance(layer, MatBiasDesc):
        return f"bias[{layer.num_channels}]"
    if isinstance(layer, BatchNormDesc):
        flags = f"scale={int(layer.has_scale)},bias={int(layer.has_bias)}"
        return f"c={layer.num_channels} eps={layer.epsilon.text} {flags}"
    if isinstance(layer, RMSNormDesc):
        return f"c={layer.num_channels} eps={layer.epsilon.text} spatial={int(layer.spatial)}"
    if isinstance(layer, TransformerRMSNormDesc):
        return f"c={layer.num_channels} eps={layer.epsilon.text}"
    if isinstance(layer, ActivationDesc):
        return layer.activation_name
    if isinstance(layer, TransformerAttentionDesc):
        return (
            f"heads={layer.num_heads} kv={layer.num_kv_heads} "
            f"q={layer.q_head_dim} v={layer.v_head_dim} rope={int(layer.use_rope)}"
        )
    if isinstance(layer, TransformerFFNDesc):
        return f"c={layer.num_channels} ffn={layer.ffn_channels} swiglu={int(layer.use_swiglu)}"
    return ""


def _inputs_version(model_version: int) -> int:
    """`NNModelVersion::getInputsVersion`（`modelversion.cpp:35-49`）。"""
    if 8 <= model_version <= 17:
        return 7
    if model_version == 7:
        return 6
    if model_version == 6:
        return 5
    if model_version == 5:
        return 4
    if model_version in (3, 4):
        return 3
    raise UnsupportedModelVersionError(
        f"NNModelVersion: Model version not currently implemented or supported: {model_version}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_any(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.katago_bin",
        description="KataGo .bin.gz 权重的无损解析 / 反向序列化工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_rt = sub.add_parser("roundtrip", help="parse -> reserialize -> byte diff")
    p_rt.add_argument("path", help=".bin or .bin.gz model file")
    p_rt.add_argument(
        "--out", default=None, help="optional: also write the reserialized bytes here"
    )

    p_ins = sub.add_parser("inspect", help="print the layer inventory")
    p_ins.add_argument("path", help=".bin or .bin.gz model file")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "roundtrip":
        model, report = roundtrip_diff(_read_any(args.path))
        print(report.summary)
        print(f"  parsed {len(list(iter_layers(model)))} layers")
        if args.out:
            with open(args.out, "wb") as f:
                f.write(serialize_model(model))
            print(f"  wrote reserialized bytes to {args.out}")
        return 0 if report.identical else 1

    if args.command == "inspect":
        print(summarize(parse_model(_read_any(args.path))))
        return 0

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
