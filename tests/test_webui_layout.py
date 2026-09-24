"""webui 鼠标落子虚影 + 缩放回退守卫。

webui 是内嵌在 scripts/webui.py 里的一段 HTML/CSS/JS 字符串，没有浏览器可跑，
因此本测试做两件事：
  1. 断言落子虚影逻辑存在，并断言棋盘缩放功能已被回退（防止悄悄复活）；
  2. 抽出 <script> 内容交给 `node --check` 做真实 JS 语法校验（node 缺失则跳过）。

测试驱动：先写本测试并确认失败（功能尚未实现），再改 webui.py 使其通过。
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.webui import HTML_PAGE


def _script_body():
    """取出 HTML_PAGE 里的 <script>…</script> 正文。"""
    m = re.search(r"<script>(.*?)</script>", HTML_PAGE, re.DOTALL)
    assert m, "HTML_PAGE 中未找到 <script> 块"
    return m.group(1)


# --------------------------------------------------------------------------- #
# 缩放功能已回退（用户决策：棋盘缩放不要，恢复原样）
# --------------------------------------------------------------------------- #
def test_no_board_zoom_css():
    """#boardWrap 不得再带 zoom（缩放已回退）。"""
    assert not re.search(r"#boardWrap\s*\{[^}]*\bzoom\s*:", HTML_PAGE), \
        "#boardWrap 仍带 CSS zoom，缩放功能未回退"


def test_no_zoom_controls():
    """缩放滑杆 / 百分比 / 自适应按钮必须已移除。"""
    for tag in ('id="zoomRange"', 'id="zoomPct"', 'id="btnAutoZoom"', 'id="zoomRow"'):
        assert tag not in HTML_PAGE, f"{tag} 仍存在，缩放控件未回退"


def test_no_zoom_js():
    """setZoom / autoZoom / autoZoomMode 等缩放 JS 必须已移除。"""
    js = _script_body()
    for sym in ("function setZoom", "function autoZoom", "autoZoomMode", "ZMIN"):
        assert sym not in js, f"{sym} 仍存在，缩放 JS 未回退"


# --------------------------------------------------------------------------- #
# 坐标换算（回退后为原始 rect 相对坐标，不再做 zoom 比例换算）
# --------------------------------------------------------------------------- #
def test_coordinate_conversion_present():
    """click 与 mousemove 共用 canvasPoint（基于 getBoundingClientRect 的相对坐标）。"""
    js = _script_body()
    assert "getBoundingClientRect" in js
    assert "function canvasPoint" in js


# --------------------------------------------------------------------------- #
# 命中判定（click 与 mousemove 共用）
# --------------------------------------------------------------------------- #
def test_hit_test_helper_exists():
    """必须抽出 hitTest(mx,my)，click 与 mousemove 共用以保证判定一致。"""
    js = _script_body()
    assert "function hitTest" in js


# --------------------------------------------------------------------------- #
# 鼠标落子虚影
# --------------------------------------------------------------------------- #
def test_hover_listeners_exist():
    """必须监听 mousemove / mouseleave 来驱动/清除虚影。"""
    js = _script_body()
    assert "mousemove" in js
    assert "mouseleave" in js


def test_ghost_rendered_with_transparency():
    """虚影以半透明绘制（globalAlpha），普通点 0.45、候选点更淡。"""
    js = _script_body()
    assert "globalAlpha" in js
    assert "0.45" in js
    assert "0.28" in js


def test_ghost_uses_human_color():
    """虚影颜色应取人类执子方（S.human_color：1=黑 / -1=白）。"""
    js = _script_body()
    assert re.search(r"drawStone\(\s*S\.human_color", js), \
        "虚影未复用 drawStone 且未取 S.human_color"


def test_ghost_only_when_playable():
    """仅在轮到人走且目标点为空时显示虚影。"""
    js = _script_body()
    assert "to_play" in js and "human_color" in js


# --------------------------------------------------------------------------- #
# JS 语法校验（node 可用时）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("node") is None, reason="未安装 node，跳过 JS 语法校验")
def test_embedded_js_is_syntactically_valid():
    """把 <script> 正文交给 node --check，确保无语法错误。"""
    body = _script_body()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "webui_script.js")
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
        r = subprocess.run(["node", "--check", p], capture_output=True, text=True)
        assert r.returncode == 0, f"JS 语法错误:\n{r.stderr}"


def test_board_paint_constants_untouched():
    """缩放回退后棋盘内部绘制常量保持原样（画布 660 / PAD=34 / N=19）。"""
    assert re.search(r"const N = 19, PAD = 34, CS = \(660 - PAD \* 2\) / \(N - 1\);",
                     HTML_PAGE), "棋盘绘制常量被改动"
    assert '<canvas id="bd" width="660" height="660">' in HTML_PAGE

