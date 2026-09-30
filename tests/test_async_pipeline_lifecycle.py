"""异步流水线生命周期回归测试（跨迭代重建 + stop 收尾 + 绝对无产出兜底）。

P1.2 把「无产出**且**无存活 worker」变成显式报错，留了三个洞：

  A. `--async-pipeline 1 --iters > 1` 直接不可用：main() 只在 `_pipeline is None` 时
     建流水线，轮末 `stop()` 之后**从不清空引用** → 第 2 轮判为「已启动」→ 复用
     `stop_event` 已置位、worker 已退出的流水线，既无产出也无存活 worker。
  B. `AsyncSelfPlayPipeline.stop()` 只 `join(timeout=5.0)` 不 terminate 拖尾 worker
     （正在下长对局的 worker 会活过 stop），且 `AsyncDataQueue` 的 mp.Queue 从不
     close → 每轮迭代泄漏管道/信号量句柄。
  C. `SelfPlayWorker.run()` 的 `except Exception: time.sleep(0.1)` 吞掉一切异常并
     无限重试（`_play_one_game` 与 MCTS 构造都在该 try 内）→ 每局必失败的配置错误
     让 worker 永远 `is_alive()`、永远不 `put` →「存活就继续等」退化成静默无限等待。

覆盖:
  - `_ensure_async_pipeline` / `_shutdown_async_pipeline`：按需构建、迭代内复用、
    解挂后重建、幂等收尾
  - `stop()` 收尾：拖尾 worker terminate 后再 join、队列 close、workers 清空、
    幂等不重复打印/close
  - `_drain_queue_into_buffer` 的绝对无产出上限：默认 1800s、超过即抛（含完整诊断）、
    有产出即归零、`max_wait=None` 显式关闭

全部用鸭子类型流水线 / worker / 队列，**不真起 mp 进程**；假队列带硬轮询上限，
实现若失效会以 AssertionError 失败而不是死循环挂住整个测试进程。
"""
import argparse
import inspect
import os
import re
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.async_pipeline as ap
from scripts.selfplay_train import (
    _drain_queue_into_buffer, _ensure_async_pipeline, _shutdown_async_pipeline)


def _args(**over):
    base = dict(batch_size=1, td=0, td_steps=3, td_alpha_init=0.2,
                td_alpha_end=0.9, no_augment=1)
    base.update(over)
    return argparse.Namespace(**base)


def _game(board=2, moves=10):
    """构造一局自对弈数据。

    采集行契约是 **8 元组**（2026-09-30 去 MCTS 起）：`(planes, action, logp_old,
    to_play, mc, v_collect, mask, logq)`。倒数第二、三代是 7 元组（无 logq）与
    5 元组（PPO 之前的 `(planes, vt, to_play, mc, root_value)`）—— 第二位从
    「visit 分布向量」换成了「实际动作 + 该动作的 log-prob」，因为 PPO 的
    importance ratio 需要策略在**采集时**的 log-prob，而不是事后从 visit 数反推。
    `mask` 是合法着法掩码（n²+1，含恒合法的 pass 槽）；`logq` 是行为分布
    （B2 权重 w = π/q 靠它）。
    """
    n_actions = board * board + 1
    rows = []
    for t in range(moves):
        planes = np.zeros((12, board, board), dtype=np.float32)
        mask = np.ones(n_actions, dtype=bool)
        rows.append((planes, 0, -0.5, 1, t, 0.1, mask, -0.7))
    return rows


class _FakeQueue:
    """鸭子类型数据队列：按预置序列返回，None 表示一次空轮询。

    delay 复刻真实 `AsyncDataQueue.get(timeout=0.5)` 的阻塞节奏（绝对无产出上限
    是墙钟预算，桩若瞬间返回则一轮 0.05s 都不占，测不出计时逻辑）；
    hard_cap 是防挂死保险：预置序列耗尽后本应无限返回 None（模拟「流水线彻底
    不再产出」），若上限逻辑失效，测试以 AssertionError 失败而不是挂住。
    """

    HARD_CAP_MARGIN = 500

    def __init__(self, items, delay=0.0, hard_cap=None):
        self._items = list(items)
        self.polls = 0
        self.delay = delay
        self._hard_cap = hard_cap if hard_cap is not None \
            else len(self._items) + self.HARD_CAP_MARGIN

    def get(self, timeout=None):
        self.polls += 1
        if self.polls > self._hard_cap:
            raise AssertionError(
                f"队列被轮询 {self.polls} 次仍未停止：空轮询预算失效（死循环）")
        if self.delay:
            time.sleep(self.delay)
        if not self._items:
            return None
        return self._items.pop(0)


class _FakeEvent:
    """mp.Event 替身：记录 set()，is_set() 反映当前置位状态。"""

    def __init__(self):
        self.set_calls = 0

    def is_set(self):
        return self.set_calls > 0

    def set(self):
        self.set_calls += 1


class _RecordingWorker:
    """鸭子类型 worker：记录 join/terminate 调用序列。

    exits_on_join=False 复刻拖尾 worker：`run()` 的 while 只在局间检查
    `stop_event`，stop() 置位后它仍在下长对局，join(timeout) 超时也活着 ——
    只能靠 terminate() 收掉。
    survives_terminate=True 复刻连 terminate 都拿不下的极端情况（系统调用阻塞），
    用来验证 stop() 不会谎报「所有 Worker 已停止」。
    """

    def __init__(self, pid, exits_on_join=True, survives_terminate=False):
        self.pid = pid
        self.calls = []
        self.alive = True
        self._exits_on_join = exits_on_join
        self._survives_terminate = survives_terminate

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.calls.append(('join', timeout))
        if self.alive and not self._exits_on_join:
            return                          # join 超时仍未退出
        self.alive = False

    def terminate(self):
        self.calls.append(('terminate', None))
        if not self._survives_terminate:
            self.alive = False


class _FakeDataQueue:
    """鸭子类型 AsyncDataQueue：只实现 stop() 需要的 close()，复刻 closed 标志。"""

    def __init__(self):
        self.close_calls = 0
        self.closed = False

    def close(self):
        self.close_calls += 1
        self.closed = True


class _NoCloseQueue:
    """底层队列没有 close()（getattr 分支）：close() 必须不抛。"""

    def get(self, timeout=None):
        return None


def _pipeline_with(workers, data_queue=None, event=None):
    """用鸭子类型部件组装一个「只够 stop() 用」的流水线实例。

    走 `__new__` 跳过 `__init__`（它会真建 mp.Queue / mp.Event，在测试进程留
    句柄），但被测的 stop() 是真的 AsyncSelfPlayPipeline.stop。
    """
    pipeline = ap.AsyncSelfPlayPipeline.__new__(ap.AsyncSelfPlayPipeline)
    pipeline.data_queue = data_queue if data_queue is not None else _FakeDataQueue()
    pipeline.stop_event = event if event is not None else _FakeEvent()
    pipeline.workers = list(workers)
    return pipeline


def _patch_pipeline_cls(monkeypatch):
    """把 `scripts.async_pipeline.AsyncSelfPlayPipeline` 换成记录型替身。

    `_ensure_async_pipeline` 在函数体内 `from scripts.async_pipeline import
    AsyncSelfPlayPipeline`，调用时才查模块属性 → 打补丁有效，且不需要真起进程。
    """
    built = []

    class _FakePipeline:
        def __init__(self, args, model_path=None, onnx_model=None, progress_cb=None):
            self.args = args
            self.model_path = model_path
            self.onnx_model = onnx_model
            self.progress_cb = progress_cb
            self.start_calls = 0
            self.stop_calls = 0
            built.append(self)

        def start(self):
            self.start_calls += 1

        def stop(self):
            self.stop_calls += 1

    monkeypatch.setattr(ap, 'AsyncSelfPlayPipeline', _FakePipeline)
    return built


class _Holder:
    """生产里 holder 是 main() 函数对象，这里用实例代表「持有 _pipeline 的对象」。"""


# --------------------------------------------------------------------------- #
# 缺陷 A — 流水线跨迭代重建
# --------------------------------------------------------------------------- #
def test_ensure_builds_then_reuses(monkeypatch):
    """按需构建、迭代内复用；`holder._pipeline` 置 None（= 上一轮已 shutdown）后重建。

    缺陷 A 的回归锁：旧代码轮末只 `stop()` 不置 None，第 2 轮迭代的「尚未启动」
    判据为假 → 复用 stop_event 已置位、worker 已退出的流水线。
    """
    built = _patch_pipeline_cls(monkeypatch)
    holder = _Holder()
    args = _args()

    first = _ensure_async_pipeline(holder, args, 'm.pth', 'm.onnx')
    assert len(built) == 1, '首次调用必须构造流水线'
    assert first.start_calls == 1, '构造后必须立即 start'
    assert first.args is args and first.model_path == 'm.pth' and first.onnx_model == 'm.onnx'
    assert holder._pipeline is first

    again = _ensure_async_pipeline(holder, args, 'm.pth', 'm.onnx')
    assert again is first, '同一次迭代内重复调用必须复用，不得再构造/再 start'
    assert len(built) == 1 and first.start_calls == 1

    holder._pipeline = None                      # 模拟 _shutdown_async_pipeline 的解挂
    rebuilt = _ensure_async_pipeline(holder, args, 'm.pth', 'm.onnx')
    assert rebuilt is not first, '解挂后必须重建（新 stop_event + 新 worker）'
    assert len(built) == 2 and rebuilt.start_calls == 1


def test_ensure_reuses_live_pipeline_without_restarting_workers(monkeypatch):
    """复用时不得再 start()：`start()` 只 append workers，复用同一个对象会叠加进程。"""
    built = _patch_pipeline_cls(monkeypatch)
    holder = _Holder()
    args = _args()

    first = _ensure_async_pipeline(holder, args, None, None)
    first.start_calls = 1
    _ensure_async_pipeline(holder, args, None, None)

    assert len(built) == 1
    assert first.start_calls == 1, '复用路径不得重复 start'


def test_shutdown_stops_and_clears(monkeypatch):
    """shutdown：stop() 之后 `holder._pipeline is None`；无流水线时调用不抛。"""
    _patch_pipeline_cls(monkeypatch)
    holder = _Holder()
    pipeline = _ensure_async_pipeline(holder, _args(), None, None)

    _shutdown_async_pipeline(holder)

    assert pipeline.stop_calls == 1, '必须真的停掉流水线（回收 worker 与队列句柄）'
    assert holder._pipeline is None, \
        'stop() 后必须解挂，否则下一轮迭代复用已停的流水线（缺陷 A 的根因）'

    _shutdown_async_pipeline(holder)             # 无流水线：幂等、不抛
    assert pipeline.stop_calls == 1, '解挂后不得重复 stop'


def test_shutdown_clears_even_if_stop_raises(monkeypatch):
    """stop() 自身抛错也必须解挂：否则残留的半死流水线会被下一轮当成「已启动」复用。"""
    _patch_pipeline_cls(monkeypatch)
    holder = _Holder()
    pipeline = _ensure_async_pipeline(holder, _args(), None, None)

    def _boom():
        raise OSError('关闭队列句柄失败')

    pipeline.stop = _boom
    with pytest.raises(OSError):
        _shutdown_async_pipeline(holder)

    assert holder._pipeline is None, 'stop 抛错也必须解挂（否则复用坏流水线）'


# --------------------------------------------------------------------------- #
# 缺陷 B — stop() 收尾
# --------------------------------------------------------------------------- #
def test_stop_terminates_stragglers_and_closes_queue(capsys):
    """stop() 收尾三件事：拖尾 worker terminate 后再 join、队列 close、workers 清空。"""
    quick = _RecordingWorker(101)                             # join 即退出
    straggler = _RecordingWorker(202, exits_on_join=False)    # 拖尾：join 超时仍活着
    data_queue = _FakeDataQueue()
    pipeline = _pipeline_with([quick, straggler], data_queue)

    pipeline.stop()

    assert quick.calls == [('join', 5.0)], \
        '自然退出的 worker 只应 join，不该被 terminate'
    assert straggler.calls == [('join', 5.0), ('terminate', None), ('join', 5.0)], \
        '拖尾 worker 必须 join 超时 → terminate → 再 join'
    assert straggler.is_alive() is False, 'stop() 返回时不得还有活着的 worker'
    assert data_queue.close_calls == 1, 'mp.Queue 句柄必须关闭，否则每轮迭代泄漏'
    assert pipeline.workers == [], 'stop 后清空 workers，避免二次 stop 重复 join'
    assert pipeline.stop_event.is_set(), 'stop_event 必须置位（worker 循环靠它退出）'
    assert '所有 Worker 已停止' in capsys.readouterr().out

    pipeline.stop()                    # 幂等：已 stop 过
    assert data_queue.close_calls == 1, '第二次 stop 不得重复 close 句柄'
    assert straggler.calls == [('join', 5.0), ('terminate', None), ('join', 5.0)], \
        '第二次 stop 不得重复 join/terminate'
    assert capsys.readouterr().out.count('所有 Worker 已停止') == 0, \
        '已 stop 过就不该重复打印「所有 Worker 已停止」'


def test_stop_warns_when_worker_survives_terminate(capsys):
    """连 terminate 都收不掉的 worker 不能被静默吞掉：stop() 得说清，而不是谎报全停。"""
    zombie = _RecordingWorker(303, exits_on_join=False, survives_terminate=True)
    pipeline = _pipeline_with([zombie])

    pipeline.stop()

    out = capsys.readouterr().out
    assert '仍存活' in out and '303' in out, \
        f'terminate 后仍存活的 worker 必须点名报出，实际输出：{out!r}'
    assert pipeline.workers == [], '即使有 worker 收不掉，workers 也要清空（不留悬空引用）'


def test_data_queue_close_is_idempotent_and_tolerates_missing_hook():
    """AsyncDataQueue.close()：重复调用安全；底层队列没有 close() 时也不抛。"""
    dq = ap.AsyncDataQueue.__new__(ap.AsyncDataQueue)
    dq.queue = _NoCloseQueue()          # getattr(self.queue, 'close', None) 为 None
    dq.closed = False                   # 跳过了 __init__，手工补上 close() 依赖的标志

    dq.close()
    assert dq.closed is True
    dq.close()                          # 重复调用安全
    assert dq.closed is True


def test_real_data_queue_close_releases_handle_and_repeats():
    """真实 mp.Queue：close() 不抛、可重复；finally 保证即使断言失败也收掉句柄。"""
    dq = ap.AsyncDataQueue(maxsize=1)
    try:
        dq.close()
        assert dq.closed is True
        dq.close()                      # 重复调用安全（mp.Queue.close 之后不再动句柄）
    finally:
        dq.close()


# --------------------------------------------------------------------------- #
# 缺陷 C — 绝对无产出上限
# --------------------------------------------------------------------------- #
def test_drain_default_max_wait_is_1800():
    """真实调用点不传 max_wait → 默认 30 分钟绝对无产出预算。"""
    default = inspect.signature(_drain_queue_into_buffer).parameters['max_wait'].default
    assert default == 1800.0


def test_drain_absolute_no_output_cap():
    """worker 永远 is_alive() 但一局都下不出来 → 绝对无产出上限必须报错（缺陷 C 兜底）。

    这正是 `SelfPlayWorker.run()` 的 `except Exception: time.sleep(0.1)` 吞掉
    每局必失败的异常（MCTS 配置错误 / 规则断言 / MCTS 内 MemoryError）后的状态：
    存活判定恒为 True、stall 恒被重置 → 没有这道兜底就是静默无限等待。
    """
    q = _FakeQueue([None] * 200, delay=0.05)
    args = _args(batch_size=1)

    with pytest.raises(RuntimeError) as ei:
        _drain_queue_into_buffer(q, [], args, 2, 5, max_stall=2,
                                 alive_fn=lambda: True, max_wait=0.3)

    msg = str(ei.value)
    assert '存活但无产出' in msg, '判词要说明是「worker 存活但无产出」而不是慢启动'
    assert 'max_wait=0.3' in msg
    assert 'collected=0' in msg and 'len(buffer)=0' in msg and '目标 20' in msg
    assert re.search(r'已连续 \d+\.\d+s', msg), f'诊断要带出已等待秒数：{msg}'
    assert q.polls >= 4, \
        f'兜底必须独立于 max_stall：max_stall=2 却继续等到上限（实际 polls={q.polls}）'


def test_drain_cap_resets_on_progress():
    """有产出即归零：两段空转各 0.15s（< max_wait）但合计 0.4s（> max_wait）仍不误报。"""
    game = {'data': _game(), 'score': 1.0}
    q = _FakeQueue([None, None, None, game, None, None, None,
                    {'data': _game(), 'score': -1.0}], delay=0.05)
    args = _args(batch_size=1)
    buffer = []

    collected_t0 = time.monotonic()
    collected = _drain_queue_into_buffer(q, buffer, args, 2, 5, max_stall=2,
                                        alive_fn=lambda: True, max_wait=0.3)
    elapsed = time.monotonic() - collected_t0

    assert collected == 2
    assert len(buffer) == 20, '2 局 × 10 手 = 20 条，正好达标'
    assert q.polls == 8, '必须真的经历了 6 次空轮询才收齐 2 局'
    assert elapsed > 0.3, \
        f'总空转 {elapsed:.2f}s 确实超过 max_wait=0.3s —— 没误报只能靠「有产出即归零」'


def test_drain_waits_past_max_stall_while_workers_alive():
    """「存活就继续等」：max_wait=None（显式关闭绝对上限）时，空轮询预算 max_stall
    必须一直被重置，直到测试自己的硬上限踢掉我们。

    锁住两道上限的分工：绝对上限**默认生效**（见 test_drain_absolute_no_output_cap），
    且只在显式关闭时才允许无限等 —— 存活信号的重置语义不得被兜底逻辑吞掉。
    hard_cap=12 → 若实现忽略存活信号，polls 会停在 2（max_stall）。
    """
    q = _FakeQueue([None] * 12, delay=0.0, hard_cap=12)
    args = _args(batch_size=1)

    with pytest.raises(AssertionError) as ei:
        _drain_queue_into_buffer(q, [], args, 2, 5, max_stall=2,
                                 alive_fn=lambda: True, max_wait=None)

    assert '仍未停止' in str(ei.value)
    assert q.polls >= 6, f'存活 worker 下必须一路等下去（实际 polls={q.polls}）'
