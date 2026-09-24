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


# --------------------------------------------------------------------------- #
# ⓪ 推荐参数节必须与代码默认值一致
# --------------------------------------------------------------------------- #
def test_recommended_rl_section_exists():
    """run.txt 必须有 ⓪ 推荐参数节。"""
    txt = open(RUN_TXT, encoding='utf-8').read()
    assert '⓪ 推荐训练参数' in txt, "run.txt 缺少 ⓪ 推荐参数节"


def test_sft_recommended_batch_matches_scheme_b():
    """⓪ 节的 SFT batch/lr 必须与方案 B 实际命令一致（防注释与命令脱节）。

    回归：曾出现命令写 --batch-size 3000、而注释与 ⓪ 节都写 2500 的矛盾，
    且 3000 恰是已确认 OOM 的值。
    """
    txt = open(RUN_TXT, encoding='utf-8').read()

    # ⓪ 节声明的 batch
    m = re.search(r'--batch-size (\d+)\s+与 --lr ([\d.]+) 配套', txt)
    assert m, '⓪ 节未声明 SFT batch/lr 配套关系'
    rec_batch, rec_lr = m.group(1), m.group(2)

    # 方案 B 实际命令（锚点用注释头 "# 方案 B"，避免匹配到 ⓪ 节里的引用文字）
    b0 = txt.find('# 方案 B')
    assert b0 != -1, 'run.txt 缺少方案 B'
    b1 = txt.find('# 方案 C')
    assert b1 != -1 and b1 > b0, 'run.txt 方案 B/C 顺序异常'
    block = txt[b0:b1]
    cm = re.search(r'--batch-size (\d+)\s+--epochs', block)
    lm = re.search(r'--lr ([\d.]+)', block)
    assert cm and lm, '方案 B 命令缺少 --batch-size/--lr'
    cmd_batch, cmd_lr = cm.group(1), lm.group(1)

    assert rec_batch == cmd_batch, \
        f"⓪ 节 batch={rec_batch} 与方案 B 命令 batch={cmd_batch} 不一致"
    assert rec_lr == cmd_lr, \
        f"⓪ 节 lr={rec_lr} 与方案 B 命令 lr={cmd_lr} 不一致"


def test_sft_recommended_lr_follows_sqrt_scaling():
    """⓪ 节的 lr 必须符合从 (2500, 0.00356) 出发的平方根缩放律。"""
    import math
    txt = open(RUN_TXT, encoding='utf-8').read()
    m = re.search(r'--batch-size (\d+)\s+与 --lr ([\d.]+) 配套', txt)
    assert m, '⓪ 节未声明 SFT batch/lr 配套关系'
    batch, lr = int(m.group(1)), float(m.group(2))
    expect = 0.00356 * math.sqrt(batch / 2500)
    assert abs(lr - expect) < 5e-5, \
        f"lr={lr} 与平方根缩放预期 {expect:.5f}（batch={batch}）不符"


@pytest.mark.parametrize('opt,expect', [
    # 这些就是代码默认值 —— 推荐"不传也是这个值"
    ('--td', '1'),
    ('--batch-size', '256'),
    ('--mcts-vector-backup', '1'),
    ('--grad-accum-steps', '1'),
])
def test_recommended_defaults_match_code(opt, expect):
    """⓪ 节标 [默认] 的 RL 取值必须等于 selfplay_train.py 的真实默认值。"""
    import inspect
    import scripts.selfplay_train as st
    src = inspect.getsource(st.main)
    m = re.search(r'ap\.add_argument\(\s*[\'"]' + re.escape(opt) +
                  r'[\'"].*?default=([^\n,]+)', src, re.S)
    assert m, f"源码中找不到 {opt} 的默认值"
    code_val = m.group(1).strip().rstrip(')')
    assert code_val == expect, f"{opt} 代码默认={code_val}，⓪ 节推荐={expect}"


@pytest.mark.parametrize('opt,override', [
    ('--sims', '48'),
    ('--expand-topk', '16'),
])
def test_recommended_overrides_are_flagged_in_doc(opt, override):
    """刻意偏离默认值的推荐项，⓪ 节必须显式标注 [刻意下调] 并写出真实默认值。

    这些值不是代码默认（如 --sims 默认 400），靠"不传"是拿不到的，必须显式传。
    若有人误把它们"修正"回默认值，测试会失败。
    """
    import inspect
    import scripts.selfplay_train as st
    src = inspect.getsource(st.main)
    m = re.search(r'ap\.add_argument\(\s*[\'"]' + re.escape(opt) +
                  r'[\'"].*?default=([^\n,]+)', src, re.S)
    code_val = m.group(1).strip().rstrip(')')
    assert code_val != override, \
        f"{opt} 现在默认值就是 {override}，⓪ 节不应再标 [刻意下调]"

    txt = open(RUN_TXT, encoding='utf-8').read()
    start = txt.find('⓪ 推荐训练参数')
    end = txt.find('① 构建数据集')
    sec = txt[start:end if end != -1 else len(txt)]
    line = next((l for l in sec.splitlines() if opt in l), None)
    assert line, f"⓪ 节缺少 {opt} 的说明行"
    assert '[刻意下调]' in line, f"{opt} 是刻意覆盖默认值，必须标注 [刻意下调]"
    assert override in line, f"{opt} 说明行未写出推荐值 {override}"
    assert f'默认 {code_val}' in line, \
        f"{opt} 说明行未写出真实默认值（默认 {code_val}），读者无法分辨"


def test_rl_commands_do_not_use_onnx_model():
    """run.txt 的 RL 命令不得再出现 --onnx-model（当前实现会启动即崩）。

    selfplay_train.py 把 --onnx-model 的值当 torch 权重 torch.load，而 ONNX 是
    protobuf 格式 → UnpicklingError。已在 ⓪ 与 ③ 两处标注。
    """
    txt = open(RUN_TXT, encoding='utf-8').read()
    bs = chr(92)
    joined = txt.replace(bs + chr(10), ' ')
    blocks = re.findall(r'python scripts/selfplay_train\.py[^\n#]*', joined)
    assert blocks, "run.txt 未找到 selfplay_train 命令"
    bad = [b for b in blocks if '--onnx-model' in b]
    assert not bad, "以下 RL 命令仍带 --onnx-model（会崩溃）:\n  " + "\n  ".join(bad)


def test_rl_commands_are_parseable():
    """run.txt 里的 selfplay_train 命令必须真能解析（防止又写出无效参数）。"""
    import subprocess
    txt = open(RUN_TXT, encoding='utf-8').read()
    bs = chr(92)
    joined = txt.replace(bs + chr(10), ' ')
    blocks = re.findall(r'python scripts/selfplay_train\.py[^\n#]*', joined)
    assert blocks, "run.txt 未找到 selfplay_train 命令"
    env = dict(os.environ, PYTHONUTF8='1')
    for b in blocks:
        toks = [t.rstrip(bs) for t in b.split()[2:]]
        r = subprocess.run([sys.executable, SFT_SRC] + toks + ['--help'],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        # --help 会在参数校验后立即退出；非 0 且含 unrecognized 即为无效参数
        assert 'unrecognized arguments' not in r.stderr, \
            f"以下 RL 命令含无效参数:\n  {' '.join(toks)}\n  {r.stderr.strip()[-200:]}"

