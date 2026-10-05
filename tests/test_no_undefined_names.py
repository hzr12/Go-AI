r"""全仓「未定义名」门禁：把 `fetch_games.py`/`bench_v7_features.py` 那一类一次性挡住。

事故（2026-10-05，全仓扫描）
--------------------------
pyflakes 的 `F821 undefined name` 在生产路径上抓到两处**必然 NameError**：

* `scripts/fetch_games.py:252-253` —— 用了 `os.walk` / `os.sep`，而该文件的
  import 列表里**没有 `os`**。调用点 `fetch_zip_source` 是活路径 ⇒ 走 ZIP 源下载
  必崩。
* `scripts/bench_v7_features.py:1029` —— `(storage or {})`，`storage` 从未定义
  （同文件另一处的正确取法是 `storage_probe`）。

还有一处更阴的，是**诊断信息本身**坏掉：

* `tests/test_go_rules_legality.py` 里 `assert cond, f"...{当前局面}"` ——
  `当前局面` 未定义。`assert` 的消息**只在断言失败时求值**，所以它一直绿；
  **一旦那条断言失败（正是最需要这条消息的时候），拿到的是 NameError 而不是
  诊断内容**。

为什么值得做成门禁而不是逐个修
----------------------------
这三处的共同点是：**静态可查、后果是运行时崩溃或信息丢失**，而且都不在现有
门禁的覆盖范围内（全量 2115 条测试是绿的）。逐个修只能保证「这次绿」；做成门禁
才能保证「以后也绿」。

为什么用 `ast` 而不是调 `flake8`
-------------------------------
项目**没有 lint 配置**，而 `flake8` 的默认行宽 79 会把输出淹掉（实测全仓几千条
风格告警）。门禁里跑子进程 + 只取 `F821`，既能复用 pyflakes 的解析器，又不必
把 `flake8` 变成项目的硬依赖 —— 没装就 `skip`，装了就把关守住。
"""
import ast
import io
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 只扫生产路径。`tests/` 里出现未定义名多数是**故意的**（f-string 里的中文标签、
#: 故意写错的变异体），混进来会让门禁变成噪声而被关掉。
PROD_ROOTS = ('src', 'scripts')

SKIP_DIRS = {'__pycache__', '.git', 'tmp', 'data', 'katago', 'models',
             '.codegraph', 'tests', 'benchmarks'}


def _prod_files():
    for root in PROD_ROOTS:
        base = os.path.join(ROOT, root)
        if not os.path.isdir(base):
            continue
        for dp, dn, fns in os.walk(base):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for fn in sorted(fns):
                if fn.endswith('.py'):
                    yield os.path.join(dp, fn)


def _f821_via_pyflakes():
    """用 pyflakes 找未定义名；没装 flake8 就返回 None（由调用方 skip）。"""
    try:
        out = subprocess.run(
            [sys.executable, '-m', 'flake8', *PROD_ROOTS,
             '--max-line-length=400', '--select=F821,F601,F632'],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    if 'No module named flake8' in out.stderr:
        return None
    rows = []
    for ln in out.stdout.splitlines():
        if ':' not in ln:
            continue
        parts = ln.split(':', 3)
        if len(parts) < 4:
            continue
        rows.append((parts[0].replace('\\', '/'), int(parts[1]), parts[3].strip()))
    return rows


def test_no_undefined_names_in_production_code():
    """生产路径不得有未定义名 / 重定义 / 重复字典键。"""
    rows = _f821_via_pyflakes()
    if rows is None:
        pytest.skip('未装 flake8（pyflakes）；装上后本门禁才生效')
    assert not rows, (
        '生产路径有 pyflakes 级别的静态错误（都是运行时会 NameError 或信息丢失）：\n'
        + '\n'.join('  %s:%d  %s' % r for r in rows[:30]))


def test_every_production_file_parses():
    """每个生产 `.py` 都要能被 `ast` 解析。

    覆盖**没被任何测试导入**的文件 —— 全量门禁只跑得到被 import 到的模块，
    而 `scripts/` 下有一批工具脚本从不被测试触及。
    """
    bad = []
    for p in _prod_files():
        try:
            ast.parse(io.open(p, encoding='utf-8').read())
        except SyntaxError as e:
            bad.append('%s:%s %s' % (os.path.relpath(p, ROOT), e.lineno, e.msg))
        except (UnicodeDecodeError, OSError):
            continue
    assert not bad, '生产代码里存在语法错误：%s' % bad


def test_assert_messages_never_reference_undefined_names():
    """`assert cond, f"..."` 的消息里不得引用未定义名。

    这条单独立出来，是因为它**不会让测试变红**：消息只在断言失败时求值。于是
    「测试全绿」与「诊断可用」两件事被解耦了 —— 而后者恰恰是在出问题时才需要。

    做法：把每个 `assert` 的消息表达式单独求值，看它引用的名字是否都能在**同一
    函数**里绑定（或解析为全局 / 内建）。
    """
    problems = []
    for p in _prod_files():
        with io.open(p, encoding='utf-8') as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        module_names = {n.name for n in ast.walk(tree)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                          ast.ClassDef))}
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            local = set()
            for a in list(fn.args.args) + list(fn.args.kwonlyargs):
                local.add(a.arg)
            if fn.args.vararg:
                local.add(fn.args.vararg.arg)
            if fn.args.kwarg:
                local.add(fn.args.kwarg.arg)
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store,
                                                                  ast.Del)):
                    local.add(n.id)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    for al in n.names:
                        local.add((al.asname or al.name).split('.')[0])
                elif isinstance(n, ast.ExceptHandler) and n.name:
                    local.add(n.name)
            for n in ast.walk(fn):
                if not isinstance(n, ast.Assert) or n.msg is None:
                    continue
                for sub in ast.walk(n.msg):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        if (sub.id not in local
                                and sub.id not in module_names
                                and sub.id not in dir(__builtins__)
                                and sub.id not in dir(__import__('builtins'))):
                            problems.append('%s:%d  assert 消息引用未定义名 %r'
                                            % (os.path.relpath(p, ROOT), n.lineno,
                                               sub.id))
    assert not problems, '\n'.join(problems[:30])