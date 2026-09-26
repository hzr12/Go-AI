"""多进程自对弈 worker 收尾回归测试（父进程 deadlock + 子进程泄漏）。

背景：旧实现对 result_queue.get() 与 p.join() 都不带 timeout，且 spawn→收集→join
整段没有 try/finally：
  - worker 侧任何异常（模型加载失败 / OOM / board 断言）都静默退出且不放结果，
    父进程 get() 永久挂起，用户只看到「训练卡住」；
  - 收集阶段一旦抛异常，末尾的 p.join() 根本不执行 → 子进程泄漏。

覆盖:
  - worker 已退出但没交付结果 → 立刻 RuntimeError（带 gid + exitcode），不阻塞
  - get 空轮询（慢 worker）时走 dead is None → continue，继续等而不误报崩溃
  - 收集阶段抛异常 → finally 仍 join 全部子进程，活着的 terminate 后再 join
  - p.start() 自身抛错 → 抛原始异常，未 start 的进程不进收尾
  - worker 侧 {'gid', 'error'} 标记被转成 RuntimeError（带原始 traceback），
    且不被当作正常结果解包
  - worker 主体 try/except 成功时 put {'gid','data','score'}，异常时包成 error 标记
  - 并行收集按到达顺序逐局交付 on_result，每局恰好一次

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
    """鸭子类型进程：记录调用序列，可控制 join 是否自然退出。

    join() 忠实复刻 stdlib 语义：未 start 的进程 _popen is None，
    multiprocessing.Process.join 会抛 AssertionError("can only join a
    started process")。收尾逻辑若把未 start 的进程传进来，测试就能抓到。
    """

    def __init__(self, gid, alive=True, exitcode=None, exits_on_join=False):
        self.gid = gid
        self._alive = alive
        self._exitcode = exitcode
        self._exits_on_join = exits_on_join
        self._started = False
        self.calls = []

    @property
    def exitcode(self):
        return self._exitcode

    def is_alive(self):
        return self._alive

    def start(self):
        self.calls.append(('start', None))
        self._started = True

    def join(self, timeout=None):
        if not self._started:
            raise AssertionError('can only join a started process')
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


class _SlowQueue:
    """可控队列桩：前 empty_polls 次 get 抛 queue.Empty（模拟慢 worker），之后按序返回。"""

    def __init__(self, items, empty_polls):
        self._items = list(items)
        self._empty = empty_polls
        self.polls = 0

    def get(self, timeout=None):
        self.polls += 1
        if self._empty > 0:
            self._empty -= 1
            raise queue.Empty
        return self._items.pop(0)


def _worker_args(**over):
    """_selfplay_worker 会真实读取的一组 args（缺一个就 AttributeError→error 标记）。"""
    base = dict(device='cpu', onnx_model=None, board_size=9, sims=1, max_moves=None,
                temperature=1.0, expand_topk=8, expand_chunk=0)
    base.update(over)
    return argparse.Namespace(**base)


def test_dead_worker_fails_fast_instead_of_hanging():
    """空队列 + 已退出（exitcode=1）的 worker → 立刻报错，不无限阻塞。"""
    q = queue.Queue()                    # worker 崩了，一个结果都没交付
    dead = _FakeProc(3, alive=False, exitcode=1)
    seen = []

    with pytest.raises(RuntimeError) as ei:
        _poll_worker_results(q, [dead], seen.append, gids=[3],
                             poll_timeout=0.01, dead_grace=0.01)

    msg = str(ei.value)
    assert 'gid=3' in msg
    assert 'exitcode=1' in msg
    assert '0/1' in msg, '诊断里要带已收集局数'
    assert seen == [], '崩溃路径不得把任何东西交给 on_result'


def test_poll_waits_through_empty_polls_without_false_crash():
    """前 3 次 get 抛 queue.Empty（慢 worker）→ 必须走 dead is None -> continue，
    继续轮询并收齐全部结果，不误报崩溃。"""
    q = _SlowQueue([{'gid': 0, 'data': [], 'score': 1.0},
                    {'gid': 1, 'data': [], 'score': -1.0}], empty_polls=3)
    procs = [_FakeProc(0), _FakeProc(1)]
    seen = []

    results = _poll_worker_results(q, procs, seen.append,
                                   poll_timeout=0.01, dead_grace=0.01)

    assert [r['gid'] for r in results] == [0, 1]
    assert [r['gid'] for r in seen] == [0, 1]
    assert q.polls == 5, '必须先经历 3 次空轮询再收 2 条，否则分支其实没被覆盖'


def test_gids_length_mismatch_is_rejected():
    """gids 与进程数不等长 → 入口就 ValueError，不能让 zip 静默截断。

    进程标记为已退出：老实现没有长度校验时 zip 截断后会走到崩溃判定抛
    RuntimeError（干净失败），而不是对一个活着的空队列死循环挂住。
    """
    q = queue.Queue()
    dead = _FakeProc(0, alive=False, exitcode=1)

    with pytest.raises(ValueError) as ei:
        _poll_worker_results(q, [dead], lambda r: None, gids=[0, 1],
                             poll_timeout=0.01, dead_grace=0.01)

    assert '长度' in str(ei.value)


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
    # gid=2 已退出：必须走「error 标记」分支而不是崩溃分支
    procs = [_FakeProc(0), _FakeProc(1), _FakeProc(2, alive=False, exitcode=1)]
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


def test_worker_success_puts_plain_result(monkeypatch):
    """worker 成功分支：put 的是 {'gid','data','score'}，且无 error 键（契约对称性）。

    锁住成功路径：上一轮只测了异常分支，成功 put 若被误加 error 键或漏字段，
    父进程会把正常局当崩溃处理（'error' in result → RuntimeError）。
    """
    game = [('planes', 'vt', 1, 0, 0.5)]
    monkeypatch.setattr(st, 'GoAI', lambda **kw: object())
    monkeypatch.setattr(st, 'self_play_game', lambda *a, **kw: (game, 3.0))
    q = queue.Queue()

    st._selfplay_worker(5, 'models/does_not_exist.pth', _worker_args(), q)

    item = q.get_nowait()
    assert item == {'gid': 5, 'data': game, 'score': 3.0}
    assert 'error' not in item, '成功路径不得混入 error 键'


def test_parallel_collection_preserves_order_count_and_logging(capsys):
    """并行收集按到达顺序逐局交给 on_result、每局恰好一次。

    这是 main() 里 _on_worker_result 的 total_games 计数与每局打印所依赖的契约：
    收集层若改成按 gid 排序、漏交付或重复交付，计数就会漂。
    """
    q = queue.Queue()
    # 故意乱序到达（gid 2 先到），验证「按到达顺序」而非按 gid 排序
    q.put({'gid': 2, 'data': [], 'score': 1.0})
    q.put({'gid': 0, 'data': [], 'score': -1.0})
    q.put({'gid': 1, 'data': [], 'score': 0.0})
    procs = [_FakeProc(i, exits_on_join=True) for i in range(3)]

    total_games = 0
    seen = []

    def _on_result(result):
        nonlocal total_games
        total_games += 1
        seen.append(result['gid'])
        print(f"  [game {result['gid']+1}] score={result['score']:+.1f} "
              f"moves={len(result['data'])}", flush=True)

    results = _run_parallel_workers(q, procs, _on_result,
                                    poll_timeout=0.01, join_timeout=0.01)

    assert [r['gid'] for r in results] == [2, 0, 1], '结果顺序 = 到达顺序'
    assert seen == [2, 0, 1], 'on_result 必须按到达顺序逐局调用'
    assert total_games == 3, '每局恰好计数一次（等价于 total_games 语义）'
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3, '每局恰好打印一次'
    assert out[0].strip() == '[game 3] score=+1.0 moves=0'


def test_start_failure_surfaces_original_error_and_reaps_started():
    """Important 3：start() 在第 2 个进程抛错 → 抛出原始异常（不是 stdlib 断言），
    且只收尾已 start 的进程。"""
    q = queue.Queue()

    class _BoomOnStart(_FakeProc):
        def start(self):
            self.calls.append(('start', None))
            raise OSError(11, 'Resource temporarily unavailable')

    ok = _FakeProc(0, exits_on_join=True)
    bad = _BoomOnStart(1)

    with pytest.raises(OSError) as ei:
        _run_parallel_workers(q, [ok, bad], lambda r: None,
                              poll_timeout=0.01, join_timeout=0.01)

    assert not isinstance(ei.value, AssertionError), \
        '真正的失败原因（EAGAIN）不能被 join 断言盖掉'
    assert 'Resource temporarily unavailable' in str(ei.value)
    assert ok.calls == [('start', None), ('join', 0.01)], \
        '已 start 的进程必须被收尾'
    assert bad.calls == [('start', None)], \
        '未 start 的进程绝不能进收尾（否则 join 触发 "can only join a started process"）'


def test_worker_wraps_exception_into_error_marker(monkeypatch):
    """worker 主体 try/except：异常时回传 {'gid', 'error'}，成功结构不变。"""
    def _boom(**kw):
        raise RuntimeError('模型加载失败')

    monkeypatch.setattr(st, 'GoAI', _boom)
    q = queue.Queue()

    st._selfplay_worker(7, 'models/does_not_exist.pth', _worker_args(), q)

    item = q.get_nowait()
    assert item['gid'] == 7
    assert '模型加载失败' in item['error']
    assert 'Traceback' in item['error']
    assert 'data' not in item and 'score' not in item
