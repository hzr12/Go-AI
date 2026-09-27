"""spd / spd_inst 正确性测试。

原实现 ``speed = step * bs / (now - t0)`` 有两重错误：

1. ``bs`` 是每卡值，未乘 ``world_size`` —— 多卡下少算 world_size 倍；
2. resume 时 ``step`` 被覆盖成 checkpoint 的累计值，却拿它当
   「本进程步数」去算平均速率。

实测 4 卡 910A 续训时该式输出 ``spd=1059 s/s``，而按日志时间戳反推
的真实速率约为 2635 s/s（4.25 s/step × 有效 batch 11200），差 2.5 倍。
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()

# 去掉整行注释：1129 行的注释刻意引用了旧公式来解释它为何错，
# 若不剔除会被下面的「旧公式必须消失」检查误判。
CODE = '\n'.join(
    ln for ln in SRC.splitlines() if not ln.lstrip().startswith('#')
)

# --------------------------------------------------------------------------- #
# AST 定位工具
# --------------------------------------------------------------------------- #
# ⚠ 为什么本文件「基准推进在哪」这类断言必须走 AST、不能走正则：
# 正则是在**整份原文**上扫的，注释与 docstring 和真代码逐字同形。63345c7 给
# compute_l2_report 写的说明里引了一次 `if _do_stdout or _do_swanlab:`，
# `re.search(r'if _do_stdout or _do_swanlab:(.*?)\n            if _do_stdout:')`
# 就从那段 docstring 起扫，一路吞掉 365 行（把 resume 路径里合法的
# `_last_stdout_step = step` 也吞了进去），于是「打点块内不得推进基准」被
# 一段**解释代码的说明文字**判成失败。AST 里注释和 docstring 不是节点，
# 定位对「又有人写了一段解释」彻底免疫。

_BASELINE = {'_last_stdout_step': 'step', '_last_stdout_t': 'time.time()'}


def _uniques(nodes, what):
    assert len(nodes) == 1, \
        f'期望恰好 1 个{what}，实得 {len(nodes)} 个：' \
        + ', '.join(f'第 {n.lineno} 行' for n in nodes)
    return nodes[0]


def _if_nodes(tree, test_src):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.If) and ast.unparse(n.test) == test_src]


def _metrics_if(tree):
    """打点块 `if _do_stdout or _do_swanlab:`（stdout 与 swanlab 频率解耦）。"""
    return _uniques(_if_nodes(tree, '_do_stdout or _do_swanlab'),
                    '打点块 `if _do_stdout or _do_swanlab:`')


def _stdout_print_if(tree):
    """`if _do_stdout:` 里**含 spd_inst 打印行**的那一个。

    main() 里有两处 `if _do_stdout:`（分段耗时行 / 剖析结束提示），靠「第一个」
    定位会随排版漂移，所以按「体内含 spd_inst 的 logger.info」来认。
    """
    return _uniques(
        [n for n in _if_nodes(tree, '_do_stdout')
         if any('spd_inst' in ast.unparse(s) for s in n.body)],
        '含 spd_inst 打印的 `if _do_stdout:` 分支',
    )


def _single_assign(stmt):
    """`x = <值>` 返回 (x, 值源码)；链式赋值/非赋值返回 None。"""
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)):
        return stmt.targets[0].id, ast.unparse(stmt.value)
    return None


def test_spd_uses_effective_batch():
    """spd 必须乘 world_size（有效 batch），而不是每卡 bs。"""
    assert '_eff_bs = bs * max(1, world_size)' in SRC, \
        '缺少有效 batch 计算 _eff_bs = bs * world_size'
    assert re.search(r'speed = \(step - _step_at_start\) \* _eff_bs', SRC), \
        'spd 未使用本进程步数与有效 batch'


def test_resume_syncs_step_baseline():
    """resume 覆盖 step 后必须同步基准，否则 spd 仍被累计步数污染。"""
    assert '_step_at_start = 0' in SRC, '缺少 _step_at_start 初始化'
    assert re.search(
        r"step = tstate\.get\('step',\s*0\)\s*\n\s*#.*\n\s*_step_at_start = step",
        SRC,
    ), 'resume 后未同步 _step_at_start'


def test_speed_instant_exists_and_uses_local_delta():
    """spd_inst 是判断停顿的关键：相邻 stdout 打点间的真实速率。"""
    assert 'spd_inst' in SRC, '缺少 spd_inst'
    assert re.search(
        r'spd_inst = \(\(step - _last_stdout_step\) \* _eff_bs', SRC
    ), 'spd_inst 未按本区间步数增量计算'


def test_instant_baseline_advanced_only_on_stdout():
    """瞬时速率基准只在 stdout 打点时推进，才能与打印行同口径。

    _should_log 会因 swanlab_every 让打点块每 10 步就触发一次；
    若基准跟着它推进，spd_inst 会变成 10 步口径，而打印行是 50 步
    区间，两者对不上、无法交叉验证。
    """
    tree = ast.parse(SRC)
    out = _stdout_print_if(tree)      # `if _do_stdout:`（打印 spd_inst 那行）
    metrics = _metrics_if(tree)       # `if _do_stdout or _do_swanlab:`

    # --- ① 两个基准推进语句必须是 stdout 分支的**直接**语句 -----------------
    # 「直接」= 不许藏在内层 if/try/with 里：藏起来就等于「不一定推进」，
    # 打印行与基准又会跨口径。
    got = {}
    for stmt in out.body:
        pair = _single_assign(stmt)
        if pair and pair[0] in _BASELINE:
            got[pair[0]] = pair[1]
    assert got == _BASELINE, \
        '瞬时速率基准必须在 _do_stdout 分支内直接更新，实际：' + repr(got)

    # --- ② 打点块（_do_stdout or _do_swanlab）内不得推进基准 ----------------
    # 查的是**整个子树里的写目标**（Store 上下文的 Name），而不是某一行的
    # 逐字文本：写在打点块的任何嵌套层里都算违规。
    bad = [w.lineno for w in ast.walk(metrics)
           if isinstance(w, ast.Name) and isinstance(w.ctx, ast.Store)
           and w.id in _BASELINE]
    assert not bad, \
        '瞬时速率基准不能在打点块内推进（会被 swanlab 频率带偏）：第 ' \
        + ', '.join(map(str, bad)) + ' 行'


def test_speed_inst_logged_and_uploaded():
    """瞬时速率要同时进 stdout 与 swanlab，否则无法画图对比。"""
    assert re.search(r'spd=%.0f spd_inst=%.0f s/s', SRC), '日志行缺少 spd_inst'
    assert '"speed_inst": spd_inst' in SRC, 'swanlab 未上报 speed_inst'


def test_old_broken_formula_gone():
    """旧的双重错误公式必须已消失（防止改回）。"""
    assert not re.search(r'speed = step \* bs /', CODE), \
        '旧的 spd 公式仍在：未乘 world_size 且用累计 step'
    assert not re.search(r'speed = step \* bs\b', CODE), '旧 spd 公式残留'
