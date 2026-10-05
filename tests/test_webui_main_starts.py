# -*- coding: utf-8 -*-
"""`scripts/webui.py::main()` 必须**真的能起来** —— 两种后端都要。

## 为什么需要这条

`main()` 里有一段「引擎后端 or 原生 MCTS」的双分支装配（item A 引入）。那次改动
里 `_bs = args.board_size` 只写在引擎分支里，而原生分支去读它 ——
Python 于是只在**没走引擎分支时**抛 `UnboundLocalError`。也就是说：

    python scripts/webui.py --model models/sft_19x19_v12.pth

这条**最常用、看起来最正常**的命令直接崩，而整套 2141 条测试全绿。门禁抓不到它，
因为没有任何测试真把 `main()` 跑到那一行 —— 现有测试都是 import `Session` 或
单独 exec 某个 `ast.If` 节点。

所以这里补的不是「再测一遍装配逻辑」，而是**把 `main()` 真的跑起来**：
子进程起服务、轮询到 HTTP 200、再确认端口已释放。判据是「服务真的应答了」，
不是「没抛异常」。

跑法：pytest tests/test_webui_main_starts.py
"""
import os
import socket
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V7_CKPT = os.path.join(ROOT, "tmp", "coding", "a_v7_npu4.pth")
V12_CKPT = os.path.join(ROOT, "models", "sft_19x19_v12.pth")


def _port_open(port, timeout=0.5):
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_and_probe(argv, port, wait_s=75):
    """起 `webui.py`，等它应答；返回 (是否起来, 合并后的输出)。"""
    p = subprocess.Popen([sys.executable, "scripts/webui.py"] + argv,
                         cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    try:
        import time
        import urllib.request
        for _ in range(wait_s):
            if p.poll() is not None:            # 提前退出 = 启动失败
                break
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/" % port,
                                            timeout=1.5) as r:
                    if r.status == 200:
                        return True, ""
            except Exception:
                time.sleep(1.0)
        return False, ""
    finally:
        p.terminate()
        try:
            out, _ = p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
        globals()["_LAST_OUT"] = out or ""


def _diagnostics():
    out = globals().get("_LAST_OUT", "")
    keep = [ln for ln in out.splitlines()
            if any(k in ln for k in ("Traceback", "Error", "error", "Unbound",
                                     "raise", "in_channels", "WebUI:"))]
    return "\n".join(keep[-25:])


@pytest.mark.skipif(not os.path.isfile(V12_CKPT), reason="v12 权重不在")
def test_main_starts_with_the_12_channel_model():
    """原生 MCTS 路径（**默认**，不带 --engine-gtp）必须能起来。"""
    port = _free_port()
    ok, _ = _start_and_probe(
        ["--model", V12_CKPT, "--device", "cpu", "--port", str(port),
         "--num-threads", "1"], port)
    assert ok, "webui 用 12 通道模型起不来：\n%s" % _diagnostics()
    assert not _port_open(port), "进程退出后端口仍被占用（服务没干净收掉）"


@pytest.mark.skipif(not os.path.isfile(V7_CKPT),
                    reason="真 V7 权重不在（tmp/coding/a_v7_npu4.pth）")
def test_main_starts_with_the_22_channel_v7_model():
    """22 通道 V7 走原生 MCTS 也必须能起来（双输入 + 前两手盘面都在这条路上）。"""
    port = _free_port()
    ok, _ = _start_and_probe(
        ["--model", V7_CKPT, "--device", "cpu", "--port", str(port),
         "--num-threads", "1", "--mode", "mcts"], port)
    assert ok, "webui 用 V7 22 通道模型起不来：\n%s" % _diagnostics()
    assert not _port_open(port), "进程退出后端口仍被占用"


@pytest.mark.parametrize("name", ["_bs", "_nt"])
def test_locals_read_after_the_backend_branch_are_defined_on_all_paths(name):
    """静态钉住那个只在默认路径上炸的 `UnboundLocalError`。

    `main()` 里有个「引擎后端 or 原生 MCTS」的双分支装配。那次改动把
    `_bs = args.board_size` 写在**引擎分支内部**，而两条分支之后都要读它 ——
    Python 于是只在**没走引擎分支时**抛 `UnboundLocalError`，也就是默认的
    原生 MCTS 路径直接崩，而 2141 条测试全绿。

    判据是「这个名字在所有路径上都有定义」：顶层赋值，或者 if/else **两条
    分支都**赋值。只写一条分支就不成立 —— 那正是这个 bug 的形状。
    （不写成「赋值序号 < 读取序号」那种顺序推断：那种判据会把合法的
    「先分支后读取」判成有问题，恰恰是它自己先写错过一次。）
    """
    import ast
    src = open(os.path.join(ROOT, "scripts", "webui.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    main_fn = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "main")

    def stores_in(nodes):
        out = set()
        for n in nodes:
            for m in ast.walk(n):
                if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Store):
                    out.add(m.id)
                elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
                    out.add(m.target.id)
        return out

    top = stores_in(main_fn.body)
    if name in top:
        return

    # 找那个按 engine 分叉的 if，两条分支都必须赋值
    branches = [n for n in main_fn.body
                if isinstance(n, ast.If) and "engine" in ast.dump(n.test)]
    assert branches, "main() 里找不到按 engine 分叉的 if —— 装配结构变了，"
    for br in branches:
        assert name in stores_in(br.body), (
            "%s 只在 engine 分支的 body 里赋值，orelse 没赋值 ⇒ 非引擎路径读它"
            "就是 UnboundLocalError" % name)
        assert stores_in(br.orelse), (
            "%s 只在 engine 分支里赋值，orelse 没赋值 ⇒ 非引擎路径读它就是"
            "UnboundLocalError（这正是本用例要钉的那个 bug）" % name)

    assert name in top_level_stores, (
        "%s 没在 main() 顶层赋值 —— 只在某个分支里赋值的话，没走那条分支时"
        "读它就是 UnboundLocalError（默认路径会直接崩）" % name)
