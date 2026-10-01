"""webui 模型加载路径测试（P4.13 阶段门指定文件）。

守 scripts/webui.py main() 的模型构建段：
  1. **每一个** GoAI(...) 调用点都带 model_path，且其值可溯源到 args.model
     （双代加载唯一入口，不许被绕开；只查第一个调用点会漏掉后来新增的第二个）
  2. 按 main() **推导出的**（不是手抄的）同参构造真加载 checkpoint → 推断 / 前向全通
  3. 通道数由权重形状驱动（17ch 走 webui 路径也得是 17），不是处处钉死 12
  4. 权重文件缺失时回退随机权重（model_path=None 分支）不崩

用例 2/4 的构造参数从 main() 的 AST 现场推导：main() 加一个我们看不懂的参数
时是用例响亮地红，而不是悄悄少传一个键、从此不再「同参」。
"""
import ast
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.webui import build_parser
from src.game.go_rules import GoBoard
from src.inference import GoAI
from tests.test_katago_se import _make_ckpt, _register_17_placeholder

N = 9
REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEBUI_SRC = (REPO / "scripts" / "webui.py").read_text(encoding="utf-8")

# main() 的 GoAI(...) 里既不是字面量也不是 args.X 的局部量，由测试侧按 CPU 场景提供
_TEST_SIDE_LOCALS = {"use_compile": False, "channels_last": False}


# --------------------------------------------------------------------------- #
# AST 辅助：从 main() 里现场推导，而不是手抄一份参数列表
# --------------------------------------------------------------------------- #
def _main_fn():
    tree = ast.parse(WEBUI_SRC)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert fn is not None, "webui.py 里找不到 main()"
    return fn


def _goai_calls(node):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "GoAI"]


def _is_args_attr(node, attr):
    return (isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Name) and node.value.id == "args")


def _main_goai_kwargs(argv):
    """按 main() 的 GoAI(...) 调用推导构造参数。

    解析规则：
      - 字面量            → 取值本身
      - args.X            → argparse 解析结果
      - Name('model_path')→ 复刻 main() 的 isfile 守卫（不存在则 None）
      - Name('use_compile')/无法求值的表达式 → 取 _TEST_SIDE_LOCALS（CPU 场景）
      - 其它未登记的 Name  → pytest.fail（main() 改了写法，用例必须响，不能默认放过）
    """
    calls = _goai_calls(_main_fn())
    assert len(calls) == 1, f"main() 里的 GoAI 调用点应恰好 1 个，实为 {len(calls)}"
    ns = build_parser().parse_args(argv)
    locals_ = dict(_TEST_SIDE_LOCALS)
    locals_["model_path"] = ns.model if os.path.isfile(ns.model) else None
    out = {}
    for kw in calls[0].keywords:
        assert kw.arg, f"main() 的 GoAI 调用含位置参数，无法同参复刻: {ast.unparse(calls[0])}"
        v = kw.value
        if isinstance(v, ast.Constant):
            out[kw.arg] = v.value
        elif isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) \
                and v.value.id == "args":
            out[kw.arg] = getattr(ns, v.attr)
        elif isinstance(v, ast.Name) and v.id in locals_:
            out[kw.arg] = locals_[v.id]
        elif kw.arg in _TEST_SIDE_LOCALS:
            # 表达式形如 channels_last=args.device.split(':')[0] == 'cuda'：
            # CPU 场景下恒 False，按参数名取测试侧给定值。
            out[kw.arg] = _TEST_SIDE_LOCALS[kw.arg]
        else:
            pytest.fail(f"main() 的 GoAI 参数 {kw.arg}={ast.unparse(v)} 解析不了，"
                        f"请在 _TEST_SIDE_LOCALS 里补上（不许静默跳过）")
    return out


@pytest.fixture(autouse=True)
def _restore_registry():
    """17ch 占位构建器是进程级注册，17ch 用例用完即还原。"""
    from src import inference as inf
    saved = dict(inf._IN_CHANNEL_BUILDERS)
    yield
    inf._IN_CHANNEL_BUILDERS.clear()
    inf._IN_CHANNEL_BUILDERS.update(saved)


# --------------------------------------------------------------------------- #
# 1. 每一个调用点都走 checkpoint 路径
# --------------------------------------------------------------------------- #
def test_every_webui_goai_call_site_loads_by_checkpoint_path():
    """webui 里**每一个** GoAI(...) 都必须传 model_path，且其值溯源到 args.model。

    只看第一个调用点的话，后来新增的第二个调用点（哪怕完全绕开双代加载）会
    悄悄通过 —— 双代入口是「不许被绕开」的，必须按调用点集合守。
    """
    tree = ast.parse(WEBUI_SRC)
    calls = _goai_calls(tree)
    assert calls, "webui 不再经 GoAI 构建模型（双代入口被绕开）"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        where = ast.unparse(call)
        assert "model_path" in kw, f"GoAI 调用缺少 model_path（双代加载入口）: {where}"
        src = kw["model_path"]
        assert (_is_args_attr(src, "model")
                or (isinstance(src, ast.Name) and src.id == "model_path")), \
            f"model_path 的值既不是 args.model 也不是 main() 的 model_path 局部量: {where}"

    # 溯源链：main() 里 model_path 必须从 args.model 来，并保留不存在→None 的守卫
    fn = _main_fn()
    assigns = [n for n in ast.walk(fn)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "model_path"
                       for t in n.targets)]
    assert any(_is_args_attr(a.value, "model") for a in assigns), \
        "main() 的 model_path 不再来自 args.model（--model 失效了）"
    guards = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "isfile"]
    assert guards, "main() 的 model_path 缺少 isfile 守卫（缺权重时的 None 回退没了）"


# --------------------------------------------------------------------------- #
# 2. 按 main() 同参构造真加载 checkpoint
# --------------------------------------------------------------------------- #
def test_webui_goai_loads_checkpoint_like_main(tmp_path):
    """用 main() 现场推导的参数构造 → 权重逐位进了模型 → 单/批量前向输出合法。"""
    ck = _make_ckpt(tmp_path, 12)
    argv = ["--model", ck, "--board-size", str(N), "--device", "cpu"]
    kwargs = _main_goai_kwargs(argv)

    # 「同参」是推导出来的：参数名集合必须与 main() 里的完全一致
    calls = _goai_calls(_main_fn())
    assert {k.arg for k in calls[0].keywords} == set(kwargs), \
        "推导出的参数名与 main() 不一致"
    assert kwargs["model_path"] == ck, "model_path 没取到 argparse 的 --model"

    ai = GoAI(**kwargs)
    assert ai.in_channels == 12
    assert ai.board_size == N
    # 权重真的进了模型（抓「加载被跳过/状态字典没解包」这类静默随机权重）
    sd = torch.load(ck, map_location="cpu")["model"]
    assert torch.equal(ai.model.state_dict()["backbone.conv1.weight"],
                       sd["backbone.conv1.weight"]), "checkpoint 权重未加载进模型"
    board = GoBoard(N)
    board.play(20)
    pol, val = ai.predict(board, [20, -1, -1], [-1, -1, -1], -1)
    assert pol.shape == (N * N + 1,)
    assert np.isclose(pol.sum(), 1.0, atol=1e-5)
    assert -1.0 <= val <= 1.0
    # 批量路径（webui 的 MCTS 走 predict_batch）
    p_b, v_b = ai.predict_batch([(board, [20, -1, -1], [-1, -1, -1], -1)])
    assert p_b.shape == (1, N * N + 1)
    assert v_b.shape == (1,)
    # 5 元组预计算特征路径（webui MCTS 实际喂的形状）
    planes = board.feature_planes_batched(
        board.board[None], [[20, -1, -1]], [[-1, -1, -1]], [-1], [board.ko_point],
        n_channels=ai.in_channels)[0]
    p_p, v_p = ai.predict_batch([(None, [20, -1, -1], [-1, -1, -1], -1, planes)])
    assert p_p.shape == (1, N * N + 1) and v_p.shape == (1,)


# --------------------------------------------------------------------------- #
# 3. 通道数由权重形状驱动，不是钉死 12
# --------------------------------------------------------------------------- #
def test_webui_goai_infers_in_channels_from_checkpoint(tmp_path):
    """17ch 权重走 webui 的同一入口 → in_channels 必须是 17。

    只测 12ch 默认路径的话，把 GoAI 里的 in_channels 全程写死 12 也能全绿。
    """
    _register_17_placeholder()
    ck = _make_ckpt(tmp_path, 17)
    ai = GoAI(**_main_goai_kwargs(
        ["--model", ck, "--board-size", str(N), "--device", "cpu"]))
    assert ai.in_channels == 17, "webui 路径没有把通道数从权重带进来"
    assert ai.model.state_dict()["backbone.conv1.weight"].shape[1] == 17
    board = GoBoard(N)
    board.play(20)
    pol, val = ai.predict(board, [-1, -1, -1], [-1, -1, -1], 1)
    assert pol.shape == (N * N + 1,)
    assert np.isclose(pol.sum(), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# 4. 权重缺失 → 随机权重分支
# --------------------------------------------------------------------------- #
def test_webui_missing_checkpoint_uses_random_weights(tmp_path):
    """main() 的「文件不存在 → model_path=None」分支：随机权重不崩、默认 12ch。"""
    missing = str(tmp_path / "does_not_exist.pth")
    kwargs = _main_goai_kwargs(
        ["--model", missing, "--board-size", str(N), "--device", "cpu"])
    assert kwargs["model_path"] is None, \
        f"不存在的权重没有回退成 None（main() 的 isfile 守卫没走到）: {kwargs['model_path']}"
    ai = GoAI(**kwargs)
    assert ai.in_channels == 12
    pol, val = ai.predict(GoBoard(N), [-1, -1, -1], [-1, -1, -1], 1)
    assert pol.shape == (N * N + 1,)
