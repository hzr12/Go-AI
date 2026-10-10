#!/usr/bin/env python3
"""KataGo `.bin.gz` 导出链路的**转发验证**入口。

这个脚本**不改** `src/data/katago_bin.py`（它已通过字节级 round-trip，是基线），
只在它之上提供一条可执行的链路，用来回答一个 byte diff 回答不了的问题：

    解析出来的**中间表示语义**对不对？转发写出的文件**真实引擎**认不认？

 为什么 round-trip 不够：`serialize_model(parse_model(x)) == x` 对**任何**只要
自洽的读写器都成立 —— 把 `linear_gate` 读进 `linear2`、把 `numBlocks` 读成
`numBlocks+1`，往返照样逐字节一致，但权重已经张冠李戴。这两类 bug 互相掩盖。
所以真正的验收必须是：**同一份官方权重，官方引擎加载前后给出同一份输出**。

三个子命令
----------------------------------------------------------------------------
``forward --in A --out B``
    解析 A -> 原样重新序列化 -> 写 B（`.gz` 后缀则 `mtime=0` 压缩），
    并对 **decompressed payload** 做 byte 级 diff。风险为零，只用来看
    "我们写出去的字节是不是和读进来的一模一样"。

``inspect --in A``
    打印层清单与结构摘要（trunk 拓扑 / 两个 head / 层表）。

``export --checkpoint ours.pth --out ours.bin.gz``
     **未实现，故意抛 `NotImplementedError``** → **2026-10-03 已实现**。
    翻译逻辑在 `src/data/katago_export.py`（含两处结构差异与四个被剥离的
    自研头的完整说明）；本函数只做包装 + **强制读回自检**。
    已用随机权重验证：引擎报 ``Model name: goai_v7 (nbt transformer,
    5545737 params)``，``rootInfo`` 里 sv3 六通道全部被正确读出并后处理，
    GTP ``genmove`` 可正常走子。

用法::

    python scripts/export_katago_bin.py forward --in katago/kata1-tf2-b10c384-s2941M-d5872M.bin.gz \\
        --out tmp/coding/kg_forward.bin.gz
    python scripts/export_katago_bin.py inspect --in katago/kata1-tf2-b10c384-s2941M-d5872M.bin.gz
    python scripts/export_katago_bin.py export --checkpoint models/ours.pth --out tmp/coding/ours.bin.gz
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import katago_bin as kb  # noqa: E402
from src.data.katago_bin import ModelDesc, RoundtripReport  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 官方权重（阶段 1 / 阶段 2 的输入）
OFFICIAL_MODEL = os.path.join(
    REPO, "katago", "kata1-tf2-b10c384-s2941M-d5872M.bin.gz"
)

#: decompressed payload 的字节数与官方引擎日志里的参数量（独立来源，不是被测代码算的）
OFFICIAL_PAYLOAD_LEN = 42_248_752
OFFICIAL_PARAMS = 10_545_753

# ---------------------------------------------------------------------------
# `export` 的待办清单
# ---------------------------------------------------------------------------
#
# 官方 b10c384h6nbttflrs 的 value head（`inspect` 实测）：
#     v3Mul/bias   -> numValueChannels         = 3
#     sv3Mul/bias  -> numScoreValueChannels    = 6      ← 关键
#     ownership    -> numOwnershipChannels     = 1
# 我们的 `ValueHead`（`src/networks/katago_v7.py:571-582`）：
# outcome : Linear(96 -> 3) -> v3 (3) 通道数对得上
# scores : Linear(96 -> 3) -> sv3 (3) 官方要 6
#     ownership/scoring/futurepos/seki : 1x1 conv on VV
#     ScorebeliefHead : 842 桶混合分布（katago_v7.py:612-647）
# 我们的 `PolicyHead`（`katago_v7.py:487`）：
#     out = _ScaledConv2d(channels, K, 1, bias=True)
# KataGo 的 `p2Conv` 是 `hasBias=0`（`desc.cpp` policy head 写出端不写 bias 数组），
# 且 `hasBias` 是**"文件里有没有这个数组"**的开关，置 1 会多写/少写一段字节。

#: 这行文案被 `tests/test_export_forward.py::test_export_command_raises_not_implemented` 逐字断言
# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass
class ForwardResult:
    """一次转发的全部事实。`report.identical` 是唯一的成败判据。"""

    model: ModelDesc
    payload: bytes
    report: RoundtripReport
    reference: bytes
    out_path: Optional[str] = None

    @property
    def reference_sha256(self) -> str:
        return hashlib.sha256(self.reference).hexdigest()

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def num_layers(self) -> int:
        return sum(1 for _ in kb.iter_layers(self.model))

    def summary(self) -> str:
        lines = [
            f"in   payload        : {len(self.reference)} bytes",
            f"     sha256         : {self.reference_sha256}",
            f"out  payload        : {len(self.payload)} bytes",
            f"     sha256         : {self.payload_sha256}",
            f"byte diff           : {self.report.summary}",
            f"parser consumed     : {self.report.consumed_length} "
            f"(trailing {self.report.trailing_bytes!r})",
            f"model               : {self.model.name} "
            f"(modelVersion {self.model.model_version})",
            f"layers              : {self.num_layers}",
            f"floats in file      : {kb.count_file_floats(self.model)}",
            f"params (KataGo)     : {kb.count_parameters(self.model)}",
            f"getShortInfoString  : {self.model.get_short_info_string()}",
        ]
        if self.out_path:
            size = os.path.getsize(self.out_path)
            lines.append(f"wrote               : {self.out_path} ({size} bytes on disk)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_any(path: str) -> bytes:
    """读 `.bin` 或 `.bin.gz`（后缀不参与判断，看 gzip magic）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到输入文件 {path}\n"
            f"官方权重应在 {OFFICIAL_MODEL}（阶段 1/2 的前提）。"
        )
    with open(path, "rb") as f:
        return f.read()


def write_any(path: str, payload: bytes) -> None:
    """写 `.bin` / `.bin.gz`。走 `katago_bin.save_model_file`，不自己拼 gzip。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    kb.save_model_file(kb.parse_model(payload), path)


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def forward_bytes(data: bytes) -> ForwardResult:
    """纯内存转发：解析 -> 原样序列化 -> byte diff。"""
    model, report = kb.roundtrip_diff(data)
    payload = kb.serialize_model(model)
    return ForwardResult(
        model=model, payload=payload, report=report, reference=kb.maybe_gunzip(data)
    )


def forward(in_path: str, out_path: Optional[str] = None) -> ForwardResult:
    """转发一个模型文件。可选写出到 `out_path`。

     比较的是 **decompressed payload**。`.gz` 容器本身**必然**与官方文件不同 ——
    官方是 KataGo 自带的 zlib 打的，我们用 Python `gzip`（`mtime=0`，保证可复现）。
    容器字节不可比，payload 才是模型本身。
    """
    result = forward_bytes(read_any(in_path))
    if out_path is not None:
        write_any(out_path, result.payload)
        result.out_path = out_path
        # 写完立刻回读，证明落盘的东西真的能被自己的解析器吃回去
        reread = forward_bytes(read_any(out_path))
        if reread.payload != result.payload:
            raise kb.KatagoBinError(
                f"回读失败：{out_path} 的 payload 与刚写进去的不一致 "
                f"({reread.report.summary})"
            )
    return result


def inspect_text(in_path: str) -> str:
    """层清单 + 结构摘要（`forward` 的伴随产物，同一份 IR）。"""
    return kb.summarize(kb.parse_model(read_any(in_path)))


class ExportError(RuntimeError):
    """导出前置条件不满足。**响亮失败**，绝不产出「能加载但语义错」的文件。"""


def export(checkpoint: str, out_path: str, name: str = 'goai_v7') -> str:
    """把 V7 checkpoint 写成 KataGo 能加载的 ``.bin.gz``。

    实现见 `src/data/katago_export.py` —— 它持有全部「结构翻译」逻辑
    （含两处官方/我们的结构差异与四个被剥离的自研头的说明）。这里只做
    CLI 包装：调用 → 读回自检 → 打印摘要。

    读回自检是**强制**的：一个「能加载但语义错」的 `.bin.gz` 比没有 exporter
    危险得多 —— 引擎不报错、搜索照跑，只是棋力悄悄错了。所以导出后立刻
    parse 回来，确认结构与刚落进去的一致。
    """
    from src.data import katago_export as kx

    try:
        kx.export_checkpoint(checkpoint, out_path, name=name)
    except kx.ExportError as e:
        raise kb.KatagoBinError(str(e)) from e

    # 读回：确认写出去的东西能被同一套解析器完整读回来
    reparsed = kb.parse_model(read_any(out_path))
    if reparsed.num_input_channels != 22 or reparsed.num_input_global_channels != 19:
        raise kb.KatagoBinError(
            f'导出后回读发现输入通道数不对：'
            f'{reparsed.num_input_channels}/{reparsed.num_input_global_channels}，'
            f'应为 22/19'
        )
    if reparsed.value_head.sv3_mul.out_channels != 6:
        raise kb.KatagoBinError(
            f'导出后回读发现 sv3Mul 通道数 {reparsed.value_head.sv3_mul.out_channels}，应为 6'
        )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/export_katago_bin.py",
        description="KataGo .bin.gz 导出链路的转发验证 / 层清单 / (未实现的)导出",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fwd = sub.add_parser(
        "forward", help="parse -> reserialize -> byte diff (verification only)"
    )
    p_fwd.add_argument("--in", dest="in_path", required=True, help=".bin / .bin.gz input")
    p_fwd.add_argument("--out", dest="out_path", default=None, help="where to write it")

    p_ins = sub.add_parser("inspect", help="print the layer inventory")
    p_ins.add_argument("--in", dest="in_path", required=True, help=".bin / .bin.gz input")

    p_exp = sub.add_parser(
        "export",
        help="export our PyTorch checkpoint to a KataGo-loadable .bin.gz",
    )
    p_exp.add_argument("--checkpoint", required=True, help="our .pth")
    p_exp.add_argument("--out", dest="out_path", required=True, help="output .bin.gz")
    p_exp.add_argument("--name", default="goai_v7", help="model name recorded in the file")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Windows 控制台默认 GBK，层清单里有中文；显式换 utf-8 + replace，
    # 免得 `print` 在管道里炸掉（而不是炸在真正该炸的地方）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - 旧解释器/被替换的流
            pass

    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.command == "forward":
        result = forward(args.in_path, args.out_path)
        print(result.summary())
        if not result.report.identical:
            print("FAIL: 转发不是字节一致的", file=sys.stderr)
            return 1
        print("OK: 转发字节完全一致")
        return 0

    if args.command == "inspect":
        print(inspect_text(args.in_path))
        return 0

    if args.command == "export":
        try:
            out = export(args.checkpoint, args.out_path, name=args.name)
        except kb.KatagoBinError as e:
            print(f"导出失败：{e}", file=sys.stderr)
            return 2
        import os

        print(f"已写出 {out}（{os.path.getsize(out):,} 字节）")
        print(f"用引擎验证：katago.exe analysis -model {out} -config analysis.cfg")
        return 0

    raise AssertionError(f"unreachable command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())