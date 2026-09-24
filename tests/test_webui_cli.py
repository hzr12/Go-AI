"""webui 命令行参数解析测试。

回归 bug：scripts/webui.py 的 main() 早期就调用了 ap.parse_args()，而 --port /
--num-threads / --mode / --expand-topk 等参数在其后才被 add_argument()，导致
所有后续 CLI 选项都报 "unrecognized arguments"（只能用无参默认值启动）。

修复方式：把全部 add_argument 收进 build_parser()，main() 只调
build_parser().parse_args()。本测试直接对 build_parser() 传参，验证每个选项
都能被解析。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.webui import build_parser


def test_parser_accepts_model_and_ver():
    p = build_parser()
    a = p.parse_args(["--model", "models/x.pth", "--ver", "v20"])
    assert a.model == "models/x.pth"
    assert a.ver == "v20"


@pytest.mark.parametrize("argv,attr,expect", [
    (["--port", "7899"], "port", 7899),
    (["--num-threads", "4"], "num_threads", 4),
    (["--mode", "mcts"], "mode", "mcts"),
    (["--board-size", "9"], "board_size", 9),
    (["--device", "cpu"], "device", "cpu"),
    (["--hybrid-sims", "8"], "hybrid_sims", 8),
    (["--hybrid-blend", "0.25"], "hybrid_blend", 0.25),
    (["--expand-topk", "16"], "expand_topk", 16),
    (["--expand-chunk", "0"], "expand_chunk", 0),
    (["--solver-thresh", "0.8"], "solver_thresh", 0.8),
    (["--leaf-ab-depth", "2"], "leaf_ab_depth", 2),
    (["--leaf-ab-width", "3"], "leaf_ab_width", 3),
    (["--policy-depth", "2"], "policy_depth", 2),
    (["--cpu-threads", "6"], "cpu_threads", 6),
])
def test_parser_accepts_each_cli_option(argv, attr, expect):
    """逐个验证曾被 parse_args 位置 bug 吞掉的 CLI 选项现在都能解析。"""
    a = build_parser().parse_args(argv)
    assert getattr(a, attr) == expect


def test_parser_accepts_boolean_flags():
    """布尔开关同样必须可用。"""
    a = build_parser().parse_args(["--priors-leaf", "--compile", "--tf32",
                                    "--no-prefetch"])
    assert a.priors_leaf is True
    assert a.compile is True
    assert a.tf32 is True
    assert a.no_prefetch is True


def test_parser_defaults_unchanged():
    """默认值保持原样（回归保护：无参启动行为不变）。"""
    a = build_parser().parse_args([])
    assert a.port == 7860
    assert a.num_threads == 8
    assert a.mode == "hybrid"
    assert a.board_size == 19
    assert a.device == "auto"
    assert a.expand_topk == 32
    assert a.expand_chunk == 8
    assert a.priors_leaf is False
    assert a.compile is False
    assert a.no_prefetch is False
    assert a.cpu_threads == 0


def test_parser_rejects_unknown_option():
    """真正未知的选项仍应报错（不要把校验放得太松）。"""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--definitely-not-an-option", "1"])


def test_mode_choices_enforced():
    """--mode 仍受 choices 约束。"""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "nope"])
