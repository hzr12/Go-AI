"""异步流水线空队列轮询回归测试（selfplay 收集循环无止境空转）。

背景：AsyncDataQueue.get 超时返回 None（自行吞掉 queue.Empty），旧收集循环
`while len(buffer) < batch_size*20` 遇 None 不计数、不退出 —— 流水线一旦停止
产出（worker 崩溃 / stop() 已调用）就永远空转且无任何诊断输出。

覆盖:
  - 模块顶层 import 了 queue（否则 except queue.Empty 直接 NameError）
  - 零星空轮询被容忍，收集行为与阈值语义不变
  - 连续空轮询达 max_stall 即抛 RuntimeError 且带完整诊断
  - stall 计数只算「连续」空轮询
"""
import argparse
import inspect
import os
import queue
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.selfplay_train as st
from scripts.selfplay_train import _drain_queue_into_buffer, _pipeline_has_live_worker


def _args(**over):
    base = dict(batch_size=1, td=0, td_steps=3, td_alpha_init=0.2,
                td_alpha_end=0.9, no_augment=1)
    base.update(over)
    return argparse.Namespace(**base)


def _game(board=2, moves=10):
    """构造一局自对弈数据：每行 (planes, vt, to_play, mc, root_value)。"""
    rows = []
    for t in range(moves):
        planes = np.zeros((12, board, board), dtype=np.float32)
        vt = np.zeros(board * board + 1, dtype=np.float32)
        vt[0] = 1.0
        rows.append((planes, vt, 1, t, 0.1))
    return rows


class _FakeQueue:
    """鸭子类型数据队列：按预置序列返回，None 表示一次空轮询。

    hard_cap 是防挂死保险：预置序列耗尽后本应无限返回 None（模拟「流水线彻底
    不再产出」），但若实现的空轮询预算失效，测试会以 AssertionError 失败而不是
    死循环挂住整个测试进程。
    """

    HARD_CAP_MARGIN = 500

    def __init__(self, items):
        self._items = list(items)
        self.polls = 0
        self._hard_cap = len(self._items) + self.HARD_CAP_MARGIN

    def get(self, timeout=None):
        self.polls += 1
        if self.polls > self._hard_cap:
            raise AssertionError(
                f"队列被轮询 {self.polls} 次仍未停止：实现没有按空轮询预算退出（死循环）")
        if not self._items:
            return None
        return self._items.pop(0)


def test_module_imports_queue():
    """模块必须有 queue 属性：mp 队列的 get 超时分支要 except queue.Empty。"""
    assert hasattr(st, 'queue'), 'scripts.selfplay_train 顶层缺少 import queue'
    assert st.queue is queue


def test_drain_tolerates_empty_polls():
    """先空轮询若干次再给数据：正常收集，不抛异常。"""
    q = _FakeQueue([None, None,
                    {'data': _game(), 'score': 1.0},
                    None,
                    {'data': _game(), 'score': -1.0}])
    args = _args(batch_size=1)          # 阈值 = batch_size*20 = 20 条
    buffer = []

    collected = _drain_queue_into_buffer(q, buffer, args, 2, 5, max_stall=5)

    assert collected == 2
    assert len(buffer) == 20, 'no_augment=1 下 2 局 × 10 手 = 20 条，正好达标'
    assert q.polls == 5


def test_drain_raises_on_stall():
    """永远空轮询：达到 max_stall 即报错，带 collected/buffer/阈值诊断。"""
    q = _FakeQueue([None] * 100)
    args = _args(batch_size=1)
    buffer = []

    with pytest.raises(RuntimeError) as ei:
        _drain_queue_into_buffer(q, buffer, args, 2, 5, max_stall=3)

    msg = str(ei.value)
    assert 'collected=0' in msg
    assert 'len(buffer)=0' in msg
    assert 'max_stall=3' in msg
    assert '20' in msg, '诊断里要带 batch_size*20 阈值'
    assert q.polls == 3, '必须在 max_stall 次后停下，不能无止境空转'


def test_stall_counts_consecutive_polls_only():
    """空轮询累计 4 次但每次都不连续（max_stall=4）→ 不误报。"""
    q = _FakeQueue([None, None,
                    {'data': _game(), 'score': 0.0},
                    None, None, None,
                    {'data': _game(), 'score': 0.0}])
    args = _args(batch_size=1)
    buffer = []

    collected = _drain_queue_into_buffer(q, buffer, args, 2, 5, max_stall=4)

    assert collected == 2
    assert len(buffer) == 20


def test_drain_default_max_stall_is_120():
    """默认 max_stall=120（0.5s × 120 = 60s 无数据即报错）。"""
    default = inspect.signature(_drain_queue_into_buffer).parameters['max_stall'].default
    assert default == 120


def test_drain_waits_past_stall_budget_while_workers_alive():
    """Important 1：worker 还活着时，空轮询预算必须被重置 —— 慢启动不得被误杀。

    桩：前 5 次空轮询期间 alive_fn 为 True（模拟正在加载模型 / 跑 400 sims 首局），
    之后 worker 全死。max_stall=3。
    若实现忽略存活信号，会在第 3 次空轮询就抛错（polls==3）；正确实现要一路等到
    worker 真的死透才在第 8 次空轮询报错。
    """
    q = _FakeQueue([None] * 50)
    args = _args(batch_size=1)

    with pytest.raises(RuntimeError) as ei:
        _drain_queue_into_buffer(q, [], args, 2, 5, max_stall=3,
                                 alive_fn=lambda: q.polls <= 5)

    assert q.polls == 8, '存活期内（5 次）预算被重置，死透后才累计到 max_stall=3'
    assert '无存活 worker' in str(ei.value), '诊断要说清是「无产出且无存活 worker」'


def test_drain_raises_when_no_worker_alive():
    """alive_fn 恒为 False（worker 全死）→ 纯队列空预算，报错带完整诊断。"""
    q = _FakeQueue([None] * 50)
    args = _args(batch_size=1)

    with pytest.raises(RuntimeError) as ei:
        _drain_queue_into_buffer(q, [], args, 2, 5, max_stall=4,
                                 alive_fn=lambda: False)

    msg = str(ei.value)
    assert q.polls == 4, '无存活 worker 时应按 max_stall 及时报错'
    assert 'collected=0' in msg and 'len(buffer)=0' in msg
    assert 'max_stall=4' in msg and '20' in msg


def test_drain_resumes_after_transient_liveness_loss():
    """空轮询期间存活信号有起有伏 → 只有「连续无产出且无存活」才累计到预算上限。

    存活序列（按 poll 序号）：1-2 死、3-9 活、10+ 死。max_stall=3。
    - 修复前（忽略存活信号）：poll 3 就抛错，polls==3；
    - 正确实现：poll 1-2 累计到 2，poll 3 被存活重置为 0，poll 10-12 连续累计到 3
      才抛错，polls==12。
    """
    q = _FakeQueue([None] * 50)
    args = _args(batch_size=1)
    alive = lambda polls: 3 <= polls <= 9

    with pytest.raises(RuntimeError):
        _drain_queue_into_buffer(q, [], args, 2, 5, max_stall=3,
                                 alive_fn=lambda: alive(q.polls))

    assert q.polls == 12, f'实际 polls={q.polls}'


class _FakeWorker:
    def __init__(self, alive):
        self._alive = alive

    def is_alive(self):
        return self._alive


class _FakeEvent:
    def __init__(self, is_set):
        self._is_set = is_set

    def is_set(self):
        return self._is_set


class _FakePipeline:
    """鸭子类型流水线：workers 列表 + mp.Event 替身。"""

    def __init__(self, workers, stopped=False):
        self.workers = workers
        self.stop_event = _FakeEvent(stopped)


def test_pipeline_has_live_worker_uses_stop_event_and_worker_alive():
    """存活判定的两个条件：stop_event 置位即视为不再产出；否则看任一 worker 存活。"""
    assert _pipeline_has_live_worker(_FakePipeline([], stopped=False)) is False, \
        '没有 worker 存活（流水线未 start 或已全灭）→ 不可能再有产出'
    assert _pipeline_has_live_worker(
        _FakePipeline([_FakeWorker(False), _FakeWorker(True)], stopped=False)) is True
    assert _pipeline_has_live_worker(
        _FakePipeline([_FakeWorker(True)], stopped=True)) is False, \
        'stop() 之后仍在跑长对局的 worker 不会再 put，不能继续等'


def test_real_pipeline_exposes_liveness_signal():
    """锁住真实调用点依赖的上游契约（async_pipeline.py:260 的 self.workers、:259 的 stop_event）。

    若上游把 workers/stop_event 改名或删掉，存活判定会静默退化成 False，误杀慢启动的训练。
    构造真实流水线（不 start，不起进程）断言容器形状；元素类型由 SelfPlayWorker
    的 Process 继承关系保证。
    """
    from multiprocessing import Process
    from scripts.async_pipeline import AsyncSelfPlayPipeline, SelfPlayWorker

    assert issubclass(SelfPlayWorker, Process), \
        'worker 必须是 Process 子类才能提供 is_alive() 存活信号'

    pipeline = AsyncSelfPlayPipeline(
        argparse.Namespace(result_queue_max=4, parallel_games=2))
    assert isinstance(pipeline.workers, list)
    assert not pipeline.stop_event.is_set()
    assert _pipeline_has_live_worker(pipeline) is False, \
        '未 start() 的流水线没有存活 worker，调用点在 start() 之后才传 alive_fn'
