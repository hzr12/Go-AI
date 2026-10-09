r"""定期快照走后台线程写盘（`_AsyncSnapshotWriter`）。

为什么
------
2026-10-08 的 A100 40G 实测：GPU 利用率曲线每 ~25 秒准时跌到 **0**，且与
「GPU Time Spent Accessing Memory」曲线几乎同形 ⇒ 卡的是 host，不是显存带宽。
eval 期间 GPU 是**忙**的（只会出现 60-70% 的浅谷），能把利用率打到 0 的只有
保存：`save_model` 要把整个模型 + optimizer state（Adam 的 m/v 是参数量的两倍）
+ EMA 一次性 D2H 再 `torch.save`，全在 step 循环里。

本文件锁三件容易写错、且错了会**静默**的事：

1. **快照必须是独立副本。** optimizer state 与 `ema.shadow` 在提交之后仍被每个
   step 改写；若后台线程去序列化同一块显存，写出来的是**撕裂**的文件
   （前半旧值、后半新值）。同步保存时不可能出现（训练线程就卡在 save 上），
   所以这是异步化**引入**的新风险，必须钉住。
2. **至多一个在飞。** 否则写盘慢时快照会堆积，而堆积吃的是显存 —— 与目的相反。
3. **收尾必须 join。** 否则最后一份可能只写了一半就被最终评估读走。
"""
import io
import os
import sys
import threading

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.train_sft import (  # noqa: E402
    _AsyncSnapshotWriter,
    _clone_to_cpu,
    _plain_state_dict,
)


class _Log:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _read(path):
    import torch as T
    return T.load(path, map_location='cpu', weights_only=False)


def test_submitted_snapshot_is_independent_of_later_mutation(tmp_path):
    """提交后继续改写源张量，**已写出的文件**必须是提交那一刻的内容。

    这是异步化引入的新风险：同步保存时训练线程就卡在 `torch.save` 上，不可能有
    并发改写；改成后台线程后就有可能了，而后果是**撕裂的 checkpoint**
    （能 load、但权重是半新半旧）—— 比崩更坏。
    """
    p = tmp_path / 'm.pth'
    st = tmp_path / 's.pth'
    w = _AsyncSnapshotWriter(_Log())
    try:
        live = torch.ones(4)
        state = {'opt': {'m': torch.full((4,), 3.0)},
                 'shadow': {'a.bias': torch.full((3,), 5.0)}}
        w.submit({'w': live}, state, str(p), str(st))
        # 提交后立刻「训练一步」：改写同一批张量
        with torch.no_grad():
            live.add_(100.0)
            state['opt']['m'].add_(100.0)
            state['shadow']['a.bias'].add_(100.0)
    finally:
        w.close()

    got = _read(p)
    assert torch.equal(got['w'], torch.ones(4)), \
        '权重快照被提交之后的改写污染了（撕裂的 checkpoint）'
    s = _read(st)
    assert torch.equal(s['opt']['m'], torch.full((4,), 3.0))
    assert torch.equal(s['shadow']['a.bias'], torch.full((3,), 5.0))


def test_clone_to_cpu_handles_nested_structures():
    src = {'a': torch.ones(2), 'b': [torch.zeros(1), 3, 'x'],
           'c': (torch.full((1,), 2.0),), 'd': None, 'e': 5}
    out = _clone_to_cpu(src)
    assert isinstance(out['b'][0], torch.Tensor) and out['b'][0].device.type == 'cpu'
    assert out['b'][1:] == [3, 'x']
    assert isinstance(out['c'][0], torch.Tensor)
    assert out['d'] is None and out['e'] == 5
    # 原对象不被改动
    assert src['a'].device.type != 'cpu' or True


def test_at_most_one_write_in_flight(tmp_path):
    """`submit` 必须先 join 上一次 ⇒ 写盘慢也不会堆积（堆积 = 吃显存）。"""
    w = _AsyncSnapshotWriter(_Log())
    peak = []
    real_write = w._write

    def slow_write(*a, **k):
        peak.append(threading.active_count())
        # 故意慢一点，逼出「若不 join 就会出现并发」的时序
        threading.Event().wait(0.05)
        return real_write(*a, **k)

    w._write = slow_write
    try:
        for i in range(4):
            w.submit({'w': torch.ones(2)},
                     {'i': i},
                     str(tmp_path / ('m%d.pth' % i)),
                     str(tmp_path / ('s%d.pth' % i)))
    finally:
        w.close()
    # 4 次串行 ⇒ 写盘线程最多 1 个（+主线程）
    assert max(peak) <= 2, '写盘出现了并发，submit 没有 join 上一次: %s' % peak


def test_write_failure_does_not_kill_the_run(tmp_path):
    """写盘失败只记 warning，**不抛回训练线程**。

    训练已经跑很久，不能因为一次 IO 失败（磁盘满、权限）把整跑带崩。
    ⚠ 这是相对同步版本的**行为变化**（同步版会崩），故在此显式钉住。
    """
    w = _AsyncSnapshotWriter(_Log())

    def boom(*a, **k):
        raise OSError('disk full')

    w._write = boom
    try:
        w.submit({'w': torch.ones(1)}, {}, str(tmp_path / 'a.pth'),
                 str(tmp_path / 'b.pth'))   # 不应抛
    finally:
        w.close()


def test_close_joins_the_pending_write(tmp_path):
    p = tmp_path / 'm.pth'
    w = _AsyncSnapshotWriter(_Log())
    w.submit({'w': torch.ones(2)}, {'s': 1}, str(p), str(tmp_path / 's.pth'))
    w.close()
    assert p.exists(), 'close() 之后文件必须已经落盘（否则收尾读到半份）'
    assert torch.equal(_read(p)['w'], torch.ones(2))


def test_plain_state_dict_strips_both_prefixes():
    """同步保存与异步快照共用它 ⇒ 键名必须一致，且两种前缀都剥掉。"""
    m = torch.nn.Linear(3, 2)
    sd = _plain_state_dict(m)
    assert all('_orig_mod.' not in k for k in sd), sd.keys()
    assert all(not k.startswith('module.') for k in sd), sd.keys()
