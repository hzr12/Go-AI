"""异步自对弈流水线的可观测性（卡死 vs 慢速必须可区分）。

现场：4 卡 NPU 上 `--async-pipeline 1 --parallel-games 8` 启动后
    `[async] 启动 8 个自对弈 Worker` 之后完全静默，CPU 0%、内存 17.8GB，
    看起来像卡死。

排查结论：不是死锁，是**在无任何输出的情况下等数据**。原收集循环是

    while len(buffer) < args.batch_size * 20:
        item = queue.get(timeout=0.5)
        if item: ...            # 只有拿到数据才有任何动作

三个问题叠加：
  1. 目标 batch_size*20 = 5120 样本，而训练只需 batch_size*5 = 1280，
     多等 4 倍数据才起步；
  2. 19 路一局约 250 样本，5120 需 15~25 局，CPU MCTS 下首次输出前
     静默十几分钟属正常；
  3. worker 全死也会无限空转——没有存活检查；且 progress_cb 一直传 None，
     worker 侧同样毫无输出。
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = open(os.path.join(ROOT, 'scripts', 'selfplay_train.py'),
           encoding='utf-8').read()
ASYNC = open(os.path.join(ROOT, 'scripts', 'async_pipeline.py'),
             encoding='utf-8').read()
CODE = '\n'.join(l for l in SRC.splitlines() if not l.lstrip().startswith('#'))


def test_collect_target_matches_training_threshold():
    """收集目标应与训练门槛一致（batch_size*5），不得再是 *20。"""
    m = re.search(r'_need\s*=\s*max\(1,\s*args\.batch_size\s*\*\s*(\d+)\)', CODE)
    assert m, '未找到 _need 定义（收集目标应是可读的变量而非内联表达式）'
    assert int(m.group(1)) == 5, \
        '收集目标应为 batch_size*5（与 selfplay_train.py 的训练门槛一致），实为 *{}'.format(
            m.group(1))
    # 内联的 *20 必须消失
    assert 'args.batch_size * 20' not in CODE, \
        '仍存在 batch_size*20 的收集目标：比训练所需多等 4 倍数据'
    assert re.search(r'while len\(buffer\) < _need:', CODE), \
        '收集循环应使用 _need'


def test_collect_loop_reports_progress_periodically():
    """等待数据期间必须周期性打印进度，否则无法区分「在算」与「死了」。"""
    assert '[collect]' in CODE, '缺少 [collect] 进度输出'
    assert '_last_report' in CODE, '缺少按时间节流（否则每 0.5s 刷一行）'
    m = re.search(r'if _now - _last_report >= ([\d.]+):', CODE)
    assert m, '未见时间节流判断'
    assert float(m.group(1)) >= 5.0, \
        '进度间隔 {}s 过短，会刷屏'.format(m.group(1))
    for field in ('buffer=', '存活Worker='):
        assert field in CODE, '进度输出缺少 {}'.format(field)


def test_collect_loop_surfaces_queue_stats():
    """AsyncDataQueue 早已维护 produced/consumed/errors 计数，但一直没人读。"""
    assert 'data_queue.stats' in CODE, '未读取队列统计'
    assert 'qsize()' in CODE, '未上报队列深度'
    for key in ('produced', 'consumed', 'errors'):
        assert key in CODE, '进度输出缺少 {} 计数'.format(key)


def test_worker_liveness_is_checked():
    """worker 全死必须报错退出，不能无限静默空转。"""
    assert 'is_alive()' in CODE, '缺少 worker 存活检查'
    m = re.search(r'if _alive == 0:(?:.|\n){0,900}?raise RuntimeError', CODE)
    assert m, 'worker 全死时应抛出 RuntimeError（并说明常见原因）'
    # 异常信息应给出可操作提示，而不只是抛一个错
    seg = CODE[m.start():m.start() + 1400]
    assert 'Worker' in seg and 'stderr' in seg, \
        '异常信息应指明是 Worker 退出并提示查看 stderr'


def test_progress_callback_is_wired():
    """progress_cb 一直传 None，worker 侧因此毫无输出。"""
    assert 'progress_cb=_on_worker_game' in CODE, 'progress_cb 未接上'
    assert 'def _on_worker_game(' in CODE, '缺少 worker 完成回调'
    assert '完成一局' in CODE, 'worker 回调应打印完成信息'


def test_callback_is_module_level_not_nested():
    """worker 现以 spawn 启动，会 pickle 整个实例（含 progress_cb）。

    嵌套在 main() 内的闭包无法序列化，worker 会在启动时直接炸掉，
    表现为 0/8 存活。因此回调必须是模块级函数。
    """
    import scripts.selfplay_train as st
    assert st._on_worker_game.__qualname__ == '_on_worker_game', \
        '回调嵌套在别的作用域（{}），spawn 下无法 pickle'.format(
            st._on_worker_game.__qualname__)


def test_callback_signature_matches_worker_call():
    """回调签名须与 worker 侧调用一致，否则运行期 TypeError。"""
    m = re.search(r'self\.progress_cb\(([^)]*)\)', ASYNC)
    assert m, '未找到 progress_cb 调用点'
    args = [a.strip() for a in m.group(1).split(',')]
    assert args == ['self.worker_id', 'game_data', 'score'], \
        '回调调用参数: {}'.format(args)
    d = re.search(r'def _on_worker_game\(([^)]*)\)', SRC)
    assert d, '未找到 _on_worker_game 定义'
    dargs = [a.strip() for a in d.group(1).split(',')]
    assert dargs == ['worker_id', 'game_data', 'score'], \
        '回调定义参数: {}'.format(dargs)


def test_pipeline_exposes_stats_and_qsize():
    """队列必须真的维护这些统计，否则上报是空壳。"""
    for key in ('produced', 'consumed', 'errors'):
        assert "'{}'".format(key) in ASYNC, \
            'AsyncDataQueue 缺少 {} 统计'.format(key)
    assert 'def qsize(' in ASYNC
