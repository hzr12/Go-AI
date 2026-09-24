"""校验 run.txt 里的 RL 参数表与 selfplay_train.py 的真实参数一致。

run.txt ③' 节手工维护了 50 项 RL 参数及默认值。这类文档最容易腐化——代码改了
文档没改，或者写错默认值，读者照抄命令就会踩坑。

本测试从 scripts/selfplay_train.py 的源码里抽出所有 add_argument 的
(选项名, 默认值)，再与 run.txt 表格逐项比对：
  · 选项名必须双向齐全（代码有→文档有，文档有→代码有）
  · 默认值文本必须一致
"""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RUN_TXT = os.path.join(ROOT, 'run.txt')
SFT_SRC = os.path.join(ROOT, 'scripts', 'selfplay_train.py')


def _code_params():
    """从源码抽出 {选项名: 默认值文本}（不实际运行 main，避免起训练）。"""
    src = open(SFT_SRC, encoding='utf-8').read()
    out = {}
    # 匹配 ap.add_argument("--name", ... )  直到该调用的右括号
    for m in re.finditer(r'ap\.add_argument\(\s*[\'"](--[a-z0-9-]+)[\'"]', src):
        name = m.group(1)
        # 从选项名之后截取到下一个 add_argument 或函数结束
        seg = src[m.end():m.end() + 600]
        nxt = seg.find('ap.add_argument')
        if nxt != -1:
            seg = seg[:nxt]
        d = re.search(r'default=([^\n,]+(?:\([^)]*\))?)', seg)
        if d:
            val = d.group(1).strip().rstrip(')')
        elif 'required=True' in seg:
            val = '(required)'
        else:
            val = 'None'
        out[name] = val
    return out


def _doc_params():
    """从 run.txt ③' 表格抽出 {选项名: 默认值文本}。"""
    txt = open(RUN_TXT, encoding='utf-8').read()
    # 只取 ③' 到 ④ 之间的参数表
    start = txt.find("③' RL 完整参数表")
    end = txt.find('④ WebUI')
    assert start != -1 and end != -1 and start < end, "run.txt 缺少 ③' RL 参数表"
    table = txt[start:end]
    out = {}
    # 说明文字里可能含空格（如「初始权重（如 SFT 预训练）」），所以用可跨空格的
    # 惰性匹配；分隔符兼容半角 ; 与全角 ；。
    # 取值只捕获 ASCII 值 token，避免把后面的全角标点/括号一起吃进来
    # （如「默认 0，详见」「默认 1（>1 时…）」）。
    pat = r'#\s+(--[a-z0-9-]+)\b.*?[;；]\s*默认\s+([A-Za-z0-9_.+\-/eE"\']+)'
    for m in re.finditer(pat, table):
        out[m.group(1)] = m.group(2)
    return out


@pytest.fixture(scope='module')
def code_params():
    return _code_params()


@pytest.fixture(scope='module')
def doc_params():
    return _doc_params()


def test_parser_has_expected_option_count(code_params):
    """自检：抽取逻辑没失效（选项数与 --help 的量级一致）。"""
    assert len(code_params) >= 45, f"只抽到 {len(code_params)} 项，抽取逻辑可能失效"


def test_doc_header_count_matches_actual(code_params):
    """run.txt 表头写的项数必须等于真实选项数（防止新增参数忘记更新表头）。"""
    txt = open(RUN_TXT, encoding='utf-8').read()
    m = re.search(r"RL 完整参数表（selfplay_train\.py，(\d+)\s*项", txt)
    assert m, "run.txt 未标注 RL 参数表项数"
    assert int(m.group(1)) == len(code_params), \
        f"表头写 {m.group(1)} 项，实际 {len(code_params)} 项"


def test_every_code_option_is_documented(code_params, doc_params):
    """代码里的每个选项都必须在 run.txt 表格中有对应行。"""
    missing = sorted(set(code_params) - set(doc_params))
    assert not missing, f"run.txt ③' 缺少这些选项: {missing}"


def test_no_documented_option_is_fictional(doc_params, code_params):
    """run.txt 表格不能出现代码里不存在的选项（防臆造）。"""
    extra = sorted(set(doc_params) - set(code_params))
    assert not extra, f"run.txt ③' 多出这些不存在的选项: {extra}"


def test_defaults_match(code_params, doc_params):
    """逐项比对默认值文本，防止文档腐化。"""
    mismatches = []
    for name, code_val in sorted(code_params.items()):
        doc_val = doc_params.get(name)
        if doc_val is None:
            continue
        cv, dv = str(code_val), str(doc_val)
        # 归一化：去空白/引号/尾部标点，便于比较写法差异
        cv = cv.strip('"\'').rstrip('.,;')
        dv = dv.strip('"\'').rstrip('.,;')
        if cv == dv:
            continue
        # 数值等价（1e-3 vs 0.001；空串 vs ""）
        try:
            if abs(float(cv) - float(dv)) < 1e-12:
                continue
        except (ValueError, TypeError):
            pass
        if cv in ('None', 'True', 'False') and dv in ('None', 'True', 'False', ''):
            continue
        mismatches.append(f"{name}: 代码={cv} 文档={dv}")
    assert not mismatches, "默认值不一致:\n  " + "\n  ".join(mismatches)


def test_run_txt_webui_commands_are_valid():
    """run.txt 里的 webui 命令必须真能被 build_parser 解析。

    回归：原先写的是 `--priors-leaf 1`（store_true 后多一个位置参数）与
    不存在的 `--simulations`，两条命令都跑不起来。
    """
    import io
    import contextlib
    from scripts.webui import build_parser

    txt = open(RUN_TXT, encoding='utf-8').read()
    bs = chr(92)
    joined = txt.replace(bs + chr(10), ' ')
    blocks = re.findall(r'python scripts/webui\.py[^\n#]*', joined)
    assert len(blocks) >= 2, "run.txt 应至少含 2 条 webui 命令"
    bad = []
    for b in blocks:
        toks = [t.rstrip(bs) for t in b.split()[2:]]
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                build_parser().parse_args(toks)
        except SystemExit:
            bad.append(' '.join(toks) + ' => ' + err.getvalue().strip().splitlines()[-1])
    assert not bad, "以下 webui 命令无法解析:\n  " + "\n  ".join(bad)
