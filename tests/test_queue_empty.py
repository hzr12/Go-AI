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
from scripts.selfplay_train import _drain_queue_into_buffer


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
    """鸭子类型数据队列：按预置序列返回，None 表示一次空轮询。"""

    def __init__(self, items):
        self._items = list(items)
        self.polls = 0

    def get(self, timeout=None):
        self.polls += 1
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
