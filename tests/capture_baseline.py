"""抓 `tests/baseline_in_channels.json` —— 12 通道路径的**改前**数值 baseline。

    python tests/capture_baseline.py tests/baseline_in_channels.json

 顺序是硬要求：这份 baseline 必须在 `src/search/mcts.py` /
`src/search/light_rollout.py` 改动**之前**抓。它要证明的是「12 通道下改前的行为
本来就对」，所以在改完代码之后再抓就只剩自我循环——抓到的永远是当前代码的指纹，
零回归也就无从谈起。

fixture 定义（权重 seed、局面、模拟数）**只有一份**，就在
`tests/test_mcts_in_channels.py::_numeric_probe` 里：本脚本直接 import 它来跑，
不重抄一遍。两边一旦分叉，`test_twelve_channel_path_bit_identical` 里的
`assert now["fixture"] == base["fixture"]` 会先炸出来，不会把差异伪装成「回归」。

数值取证要求前向的浮点求和顺序固定（线程数一变末位就变），故本脚本与测试用
同一个 `_single_thread` 约定：单线程前向。

覆盖边界（Fix 增补 #2）：本 baseline 钉住 #1 `_planes1` / #2 `_forward_level` /
#3 `_eval_children` / #5+#6 `FastPolicy.logits` 与端到端 search 的 visits/value；
**不覆盖** #4 worker 推测性预取（`num_threads=1` 被 `mcts.py:133` 的 gate 关掉）
与 #0 FastPolicy 通道交接（默认 `use_rollout=False`）。`*_probs` 两键是
smoke 键（`temperature=0` 时 one-hot，两键同 sha）。
"""
import importlib.util
import json
import os
import pathlib
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_OUT = os.path.join(HERE, "baseline_in_channels.json")
_PROBE_SRC = os.path.join(HERE, "test_mcts_in_channels.py")


def _load_numeric_probe():
    """从测试文件里取 `_numeric_probe`（baseline 的内容定义就在那）。"""
    spec = importlib.util.spec_from_file_location("_t_in_channels", _PROBE_SRC)
    if spec is None or spec.loader is None:  # pragma: no cover - 文件被移走
        raise RuntimeError(f"找不到测试文件 {_PROBE_SRC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._numeric_probe


def capture():
    """跑一遍 probe，返回 baseline dict（与测试读到的键集完全一致）。"""
    torch.set_num_threads(1)  # 数值取证：浮点求和顺序必须固定
    probe = _load_numeric_probe()
    with tempfile.TemporaryDirectory(prefix="capture_baseline_") as td:
        return probe(pathlib.Path(td))


def _verify(path, expected):
    """回读刚写的文件与内存结果逐项比：写盘这一步自己不出错。"""
    with open(path, "r", encoding="utf-8") as f:
        got = json.load(f)
    assert got == expected, "回读 baseline 与抓取结果不一致，写盘环节有问题"
    return got


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out_path = argv[0] if argv else DEFAULT_OUT

    out = capture()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    _verify(out_path, out)

    fixture = out["fixture"]
    print(f"[capture_baseline] 已写入 {out_path}")
    print(f"[capture_baseline] fixture: seed={fixture['seed']} "
          f"board_size={fixture['board_size']} torch={fixture['torch']} "
          f"numpy={fixture['numpy']}")
    for k in sorted(out):
        if k == "fixture":
            continue
        v = out[k]
        print(f"[capture_baseline]   {k}: {v if isinstance(v, str) else v!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
