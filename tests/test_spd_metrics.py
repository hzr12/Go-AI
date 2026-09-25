"""spd / spd_inst 正确性测试。

原实现 ``speed = step * bs / (now - t0)`` 有两重错误：

1. ``bs`` 是每卡值，未乘 ``world_size`` —— 多卡下少算 world_size 倍；
2. resume 时 ``step`` 被覆盖成 checkpoint 的累计值，却拿它当
   「本进程步数」去算平均速率。

实测 4 卡 910A 续训时该式输出 ``spd=1059 s/s``，而按日志时间戳反推
的真实速率约为 2635 s/s（4.25 s/step × 有效 batch 11200），差 2.5 倍。
"""
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
    m = re.search(r'if _do_stdout:(.*?)\n\n', SRC, re.S)
    assert m, '未找到 _do_stdout 分支'
    assert '_last_stdout_step = step' in m.group(1), \
        '_last_stdout_step 必须在 _do_stdout 分支内更新'
    assert '_last_stdout_t = time.time()' in m.group(1), \
        '_last_stdout_t 必须在 _do_stdout 分支内更新'
    # 打点块（_do_stdout or _do_swanlab）内不得推进基准
    m2 = re.search(r'if _do_stdout or _do_swanlab:(.*?)\n            if _do_stdout:',
                   SRC, re.S)
    assert m2, '未找到打点块'
    assert '_last_stdout_step = step' not in m2.group(1), \
        '瞬时速率基准不能在打点块内推进（会被 swanlab 频率带偏）'


def test_speed_inst_logged_and_uploaded():
    """瞬时速率要同时进 stdout 与 swanlab，否则无法画图对比。"""
    assert re.search(r'spd=%.0f spd_inst=%.0f s/s', SRC), '日志行缺少 spd_inst'
    assert '"speed_inst": spd_inst' in SRC, 'swanlab 未上报 speed_inst'


def test_old_broken_formula_gone():
    """旧的双重错误公式必须已消失（防止改回）。"""
    assert not re.search(r'speed = step \* bs /', CODE), \
        '旧的 spd 公式仍在：未乘 world_size 且用累计 step'
    assert not re.search(r'speed = step \* bs\b', CODE), '旧 spd 公式残留'
