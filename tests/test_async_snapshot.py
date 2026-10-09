r"""定期快照走后台线程写盘（`_AsyncSnapshotWriter`）。

为什么：2026-10-08 的 A100 40G 实测，GPU 利用率每 ~25 秒准时跌到 **0**，且与
「GPU Time Spent Accessing Memory」曲线同形 ⇒ 卡的是 host。eval 期间 GPU 是**忙**的
（只有 60-70% 的浅谷），能把利用率打到 0 的只有保存：模型 + optimizer state
（Adam 的 m/v 是参数量的两倍）+ EMA 一次性 D2H 再 `torch.save`，全在 step 循环里。

三件容易写错、且错了会**静默**的事：

1. 快照必须是**独立副本**。optimizer state 与 `ema.shadow` 在提交后仍被每个 step
   改写，后台线程去序列化同一块显存 ⇒ **撕裂**的 checkpoint（能 load、半新半旧）。
   同步保存时不可能出现（训练线程就卡在 save 上），这是异步化**引入**的新风险。
2. **至多一个在飞**。否则写盘慢时快照堆积，而堆积吃的是显存 —— 与目的相反。
3. **收尾必须 join**。否则最后一份可能只写了一半就被最终评估读走。
"""
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
    return torch.load(path, map_location='cpu', weights_only=False)


def test_submitted_snapshot_is_independent_of_later_mutation(tmp_path):
    """提交后继续改写源张量，已写出的文件必须是提交那一刻的内容（见 docstring 1）。"""
    p, st = tmp_path / 'm.pth', tmp_path / 's.pth'
    w = _AsyncSnapshotWriter(_Log())
    try:
        live = torch.ones(4)
        state = {'opt': {'m': torch.full((4,), 3.0)},
                 'shadow': {'a.bias': torch.full((3,), 5.0)}}
        w.submit({'w': live}, state, str(p), str(st))
        with torch.no_grad():                       # 提交后「训练一步」
            live.add_(100.0)
            state['opt']['m'].add_(100.0)
            state['shadow']['a.bias'].add_(100.0)
    finally:
        w.close()

    assert torch.equal(_read(p)['w'], torch.ones(4)), '权重快照被提交后的改写污染'
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


def test_at_most_one_write_in_flight(tmp_path):
    """`submit` 必须先 join 上一次 ⇒ 写盘慢也不会堆积（见 docstring 2）。

    计数用**本类自己**的并发计数，不是 `threading.active_count()`：后者数的是
    进程内所有线程，并行跑测试时 pytest/xdist 自己的线程会灌进来导致飘红
    （实测过一次并行失败、单跑通过）。
    """
    w = _AsyncSnapshotWriter(_Log())
    real_write = w._write
    state = {'now': 0, 'peak': 0}

    def counted_write(*a, **k):
        state['now'] += 1
        state['peak'] = max(state['peak'], state['now'])
        try:
            threading.Event().wait(0.05)          # 逼出「不 join 就有并发」的时序
            return real_write(*a, **k)
        finally:
            state['now'] -= 1

    w._write = counted_write
    try:
        for i in range(4):
            w.submit({'w': torch.ones(2)}, {'i': i},
                      str(tmp_path / ('m%d.pth' % i)),
                      str(tmp_path / ('s%d.pth' % i)))
    finally:
        w.close()
    assert state['peak'] <= 1, \
        '写盘出现了并发，submit 没有 join 上一次（峰值 %d）' % state['peak']


def test_write_failure_does_not_kill_the_run(tmp_path):
    """写盘失败只记日志、不抛回训练线程。

    ⚠ 这是相对同步版本的**行为变化**（同步版会崩）。训练已经跑很久，不能因为
    一次磁盘故障把整跑带崩。
    """
    blocker = tmp_path / 'not_a_dir'
    blocker.write_text('我是个文件，不是目录')
    w = _AsyncSnapshotWriter(_Log())
    try:
        w.submit({'w': torch.ones(1)}, {}, str(blocker / 'a.pth'),
                 str(blocker / 'b.pth'))           # 不应抛
    finally:
        w.close()


def test_write_creates_missing_parent_directory(tmp_path):
    """父目录不存在时**必须自己建**（2026-10-09 真机事故）。

    `makedirs` 原本只在 `save_model()` 里，后台线程直接 `torch.save` 绕过了它 ⇒
    干净检出（`models/` 不存在）时首次周期快照即
    `RuntimeError: Parent directory models does not exist`。异常又被「写盘失败不
    带走训练」的设计吞掉，训练照跑、快照一路静默失败 —— 崩溃时才发现根本没有可
    恢复的快照。
    """
    target = tmp_path / 'models'                  # 故意**不**创建
    assert not target.exists()
    w = _AsyncSnapshotWriter(_Log())
    try:
        w.submit({'w': torch.ones(2)}, {'s': 1},
                 str(target / 'm.pth'), str(target / 'm.pth.train_state'))
    finally:
        w.close()
    assert (target / 'm.pth').exists(), '快照没落盘：父目录未被创建'
    assert (target / 'm.pth.train_state').exists()


def test_write_failure_is_logged_at_error_level_with_paths(tmp_path):
    """写盘失败必须打 **error** 且带路径，不能是混在 INFO 里的 warning。

    失败被吞掉是有意设计，但周期快照失败的后果是「崩溃后无法恢复」，等级必须够高
    且写清是哪个文件，否则排查时根本不会注意到。

    失败是**真造出来的**（把父路径做成普通文件 ⇒ `makedirs` 必然抛），不是替换
    `_write` —— 替换掉就绕过了内部那条 error 日志，测的就不是它了。
    """
    class _ErrLog(_Log):
        def __init__(self):
            self.errors = []

        def error(self, msg, *a):
            self.errors.append(msg % a if a else msg)

    blocker = tmp_path / 'not_a_dir'
    blocker.write_text('我是个文件，不是目录')
    log = _ErrLog()
    w = _AsyncSnapshotWriter(log)
    try:
        w.submit({'w': torch.ones(1)}, {}, str(blocker / 'a.pth'),
                 str(blocker / 'b.pth'))
    finally:
        w.close()
    assert log.errors, '写盘失败没有 error 级日志（warning 会被漏掉）'
    joined = ' '.join(log.errors)
    assert 'a.pth' in joined and 'b.pth' in joined, \
        'error 日志必须写清是哪个文件：%s' % joined


def test_close_joins_the_pending_write(tmp_path):
    """`close()` 之后文件必须已经落盘，否则收尾读到半份（见 docstring 3）。"""
    p = tmp_path / 'm.pth'
    w = _AsyncSnapshotWriter(_Log())
    w.submit({'w': torch.ones(2)}, {'s': 1}, str(p), str(tmp_path / 's.pth'))
    w.close()
    assert p.exists()
    assert torch.equal(_read(p)['w'], torch.ones(2))


def test_plain_state_dict_strips_both_prefixes():
    """同步保存与异步快照共用它 ⇒ 键名必须一致，且两种前缀都剥掉。"""
    sd = _plain_state_dict(torch.nn.Linear(3, 2))
    assert all('_orig_mod.' not in k for k in sd), sd.keys()
    assert all(not k.startswith('module.') for k in sd), sd.keys()