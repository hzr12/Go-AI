"""多进程自对弈 worker 收尾回归测试（父进程 deadlock + 子进程泄漏）。

背景：旧实现对 result_queue.get() 与 p.join() 都不带 timeout，且 spawn→收集→join
整段没有 try/finally：
  - worker 侧任何异常（模型加载失败 / OOM / board 断言）都静默退出且不放结果，
    父进程 get() 永久挂起，用户只看到「训练卡住」；
  - 收集阶段一旦抛异常，末尾的 p.join() 根本不执行 → 子进程泄漏。

覆盖:
  - worker 已退出但没交付结果 → 立刻 RuntimeError（带 gid + exitcode），不阻塞
  - 收集阶段抛异常 → finally 仍 join 全部子进程，活着的 terminate 后再 join
  - worker 侧 {'gid', 'error'} 标记被转成 RuntimeError（带原始 traceback），
    且不被当作正常结果解包
  - worker 主体 try/except 会把异常包成 error 标记回传

全部用鸭子类型进程 + 真实 queue.Queue，不真起 mp 进程、不阻塞挂起。
"""
import argparse
import os
import queue
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.selfplay_train as st
from scripts.selfplay_train import _poll_worker_results, _run_parallel_workers


class _FakeProc:
    """鸭子类型进程：记录调用序列，可控制 join 是否自然退出。"""

    def __init__(self, gid, alive=True, exitcode=None, exits_on_join=False):
        self.gid = gid
        self._alive = alive
        self._exitcode = exitcode
        self._exits_on_join = exits_on_join
        self.calls = []

    @property
    def exitcode(self):
        return self._exitcode

    def is_alive(self):
        return self._alive

    def start(self):
        self.calls.append(('start', None))

    def join(self, timeout=None):
        self.calls.append(('join', timeout))
        if self._alive and not (self._exits_on_join or timeout is None):
            return                      # join 超时仍未退出
        self._alive = False
        if self._exitcode is None:
            self._exitcode = 0

    def terminate(self):
        self.calls.append(('terminate', None))
        self._alive = False
        self._exitcode = -15


def test_dead_worker_fails_fast_instead_of_hanging():
    """空队列 + 已退出（exitcode=1）的 worker → 立刻报错，不无限阻塞。"""
    q = queue.Queue()                    # worker 崩了，一个结果都没交付
    dead = _FakeProc(3, alive=False, exitcode=1)

    with pytest.raises(RuntimeError) as ei:
        _poll_worker_results(q, [dead], gids=[3], poll_timeout=0.01, dead_grace=0.01)

    msg = str(ei.value)
    assert 'gid=3' in msg
    assert 'exitcode=1' in msg
    assert '0/1' in msg, '诊断里要带已收集局数'


def test_poll_survives_early_empty_then_collects_all():
    """worker 尚未交付时先空轮询几次，随后正常收齐全部结果。"""
    q = queue.Queue()
    q.put({'gid': 0, 'data': [], 'score': 1.0})
    q.put({'gid': 1, 'data': [], 'score': -1.0})
    procs = [_FakeProc(0), _FakeProc(1)]
    seen = []

    results = _poll_worker_results(q, procs, on_result=seen.append,
                                   poll_timeout=0.01, dead_grace=0.01)

    assert [r['gid'] for r in results] == [0, 1]
    assert [r['gid'] for r in seen] == [0, 1]


def test_reap_cleans_up_on_exception():
    """收集阶段处理器抛异常 → finally 仍收尾：join 全部，存活者 terminate。"""
    q = queue.Queue()
    q.put({'gid': 0, 'data': [], 'score': 1.0})
    finisher = _FakeProc(0, exits_on_join=True)    # 交完结果后自然退出
    hung = _FakeProc(1)                            # 赖着不退出
    procs = [finisher, hung]

    def _boom(result):
        raise ValueError(f"处理结果失败: {result['gid']}")

    with pytest.raises(ValueError):
        _run_parallel_workers(q, procs, _boom, poll_timeout=0.01, join_timeout=0.01)

    assert finisher.calls == [('start', None), ('join', 0.01)], \
        '自然退出的 worker 只应 join 一次（带 timeout），不该被 terminate'
    assert hung.calls == [('start', None), ('join', 0.01),
                          ('terminate', None), ('join', None)], \
        '仍存活的 worker 必须 terminate 后再无超时 join 兜底'


def test_worker_error_marker_surfaced():
    """worker 侧 error 标记 → RuntimeError（含原始 traceback），不当正常结果解包。"""
    q = queue.Queue()
    q.put({'gid': 2, 'error': 'Traceback (most recent call last):\n  ...boom...'})

    class _ErrProc(_FakeProc):
        # 已退出且拿到的是 error 标记：必须走「error 标记」分支而不是崩溃分支
        def __init__(self):
            super().__init__(2, alive=False, exitcode=1)

    procs = [_FakeProc(0), _FakeProc(1), _ErrProc()]
    seen = []

    with pytest.raises(RuntimeError) as ei:
        _run_parallel_workers(q, procs, seen.append,
                              poll_timeout=0.01, join_timeout=0.01)

    msg = str(ei.value)
    assert 'gid=2' in msg
    assert 'boom' in msg, '必须带出 worker 侧原始 traceback 文本'
    assert seen == [], 'error 标记不能进入正常结果解包路径'
    assert all(any(c[0] == 'join' for c in p.calls) for p in procs), \
        '报错路径同样要收尾全部子进程'


def test_worker_wraps_exception_into_error_marker(monkeypatch):
    """worker 主体 try/except：异常时回传 {'gid', 'error'}，成功结构不变。"""
    def _boom(**kw):
        raise RuntimeError('模型加载失败')

    monkeypatch.setattr(st, 'GoAI', _boom)
    q = queue.Queue()
    args = argparse.Namespace(device='cpu', onnx_model=None, board_size=9)

    st._selfplay_worker(7, 'models/does_not_exist.pth', args, q)

    item = q.get_nowait()
    assert item['gid'] == 7
    assert '模型加载失败' in item['error']
    assert 'Traceback' in item['error']
    assert 'data' not in item and 'score' not in item
