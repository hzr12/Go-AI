"""预取 worker 必须 fork 在设备初始化之前（4 卡 910A OOM 的第二个原因）。

事故形状（云端截图，2026-09-30 21:39）
----------------------------------------
    [train] 开始训练 | steps/epoch=4191 | 总 steps≈4191 | warmup=419
    [data]  预取器已启用 | workers=4 depth=8          ← 之后立刻
    NPU-4/5/6/7   HBM 94%   AICore 0%

同一时刻 PyTorch 的账本只有 `6.30 GiB already allocated / 6.55 GiB reserved`
⇒ 差额 ≈ 24 GiB/卡 = 4 × 6 GiB = 4 个 worker 各一份继承来的设备映射；
AICore 0% 说明它们只占显存不干活（纯 numpy 取数）。

根因：`_BatchPrefetcher` 用 `mp.Process`（Linux 默认 fork）构造于
`init_process_group` / `torch.cuda.set_device` / 分布式包裹**之后**。fork 复制
地址空间 ⇒ CANN 设备上下文与显存映射被整份继承。GC 只管 torch 张量，管不到
别的进程继承来的映射，所以任何 GC / batch / chunk 调整都治不了它。

 2026-10-01：包裹层从 FSDP1 换成 DDP（FSDP1 已退役）。本文件钉的是「fork 必须
早于**任何**设备初始化与分布式初始化」，与用哪种包裹层无关 ⇒ 断言不变，只有这
句历史指涉改成中性表述。

这两个测试分别钉住「顺序」与「护栏」：顺序靠源码位置（这是**唯一**能证明
fork 时机的手段 —— 运行时真机上没法观察继承），护栏用假设备运行时验证真的会拦。
"""
import ast
import os
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC_PATH = ROOT / 'scripts' / 'train_sft.py'
SRC = SRC_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SRC)


def _line_of(node):
    return node.lineno


def test_prefetcher_is_forked_before_device_initialization():
    """`_BatchPrefetcher(` 的调用行号必须早于 set_device / init_process_group。

    2026-10-06 起构造点为 **2 个**：训练预取器 + eval 特征预取池（V7 eval 的
    梯子特征串行计算曾让每次 eval 磨 25-30 分钟）。两个池都必须在 fork 前。
    """
    calls = [n.lineno for n in ast.walk(TREE)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == '_BatchPrefetcher']
    assert len(calls) == 2, (
        '预取器构造点应为 2（训练 + eval 池）：找到 %d 处 %s' % (len(calls), calls))

    device_lines = [n.lineno for n in ast.walk(TREE)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ('set_device', 'init_process_group')]
    assert device_lines, '没找到 set_device / init_process_group'
    first_device = min(device_lines)
    assert max(calls) < first_device, (
        '_BatchPrefetcher 在 %s 构造，但设备初始化在 L%d —— 晚于它 fork 的 '
        'worker 会继承 CANN 上下文（4 卡实测每卡凭空多占 ~24 GiB ⇒ OOM）'
        % (calls, first_device))


def test_dataset_load_also_precedes_device_init():
    """数据集加载与预取器同段（fork 早 ⇒ 数据集也必须在设备初始化前就绪）。"""
    load_lines = [n.lineno for n in ast.walk(TREE)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name)
                  and n.func.id == 'load_from_path']
    assert len(load_lines) == 1, 'load_from_path 调用点应唯一：%s' % load_lines
    device_lines = [n.lineno for n in ast.walk(TREE)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ('set_device', 'init_process_group')]
    assert load_lines[0] < min(device_lines), \
        '数据集加载（L%d）必须早于设备初始化（L%d）' % (load_lines[0],
                                                     min(device_lines))


def test_guard_rejects_prefetcher_after_device_init():
    """护栏：设备运行时已初始化时构造预取器必须**抛错**，不能安静吃显存。"""
    sys.path.insert(0, str(ROOT / 'scripts'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('_train_sft_probe', SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _FakeCuda:
        @staticmethod
        def is_initialized():
            return True

    real_cuda = getattr(torch, 'cuda', None)
    torch.cuda = _FakeCuda
    try:
        with pytest.raises(RuntimeError, match='继承设备上下文'):
            mod._BatchPrefetcher(object(), num_workers=2, prefetch=1)
    finally:
        torch.cuda = real_cuda


def test_guard_passes_when_no_device_runtime():
    """没有设备运行时时护栏不得误伤（真机首启时 torch.cuda.is_initialized 为 False）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_train_sft_probe2', SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _False:
        @staticmethod
        def is_initialized():
            return False

    real_npu = getattr(torch, 'npu', None)
    torch.npu = _False
    # 不真起进程：这里只验「护栏不误伤」。mp.Process 打桩是因为本仓库的测试
    # 在 Windows 上跑（默认 spawn → 需要 pickle 假 dataset），而真机是 Linux
    # fork；两条路的**参数**一样，护栏只看 torch.<dev>.is_initialized()。
    started = []

    class _StubProc:
        def __init__(self, *a, **kw):
            started.append(kw.get('target'))

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def terminate(self):
            pass

    class _StubMP:
        Process = _StubProc

        class Queue:
            def __init__(self, *a, **kw):
                pass

    real_mp = mod.mp
    mod.mp = _StubMP
    try:
        pf = mod._BatchPrefetcher(object(), num_workers=2, prefetch=1)
        assert len(started) == 2, '护栏放行后应照常起 2 个 worker：%s' % started
    finally:
        mod.mp = real_mp
        if real_npu is not None:
            torch.npu = real_npu
        else:
            delattr(torch, 'npu')


def test_prefetch_workers_do_not_touch_cuda_or_npu():
    """worker 只用 numpy：不许 import/调用任何设备 API（否则又会建一份上下文）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_train_sft_probe3', SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seg = SRC[SRC.find('def _prefetch_worker('):SRC.find('class _BatchPrefetcher')]
    for bad in ('torch.npu', 'torch.cuda', '.to(device', 'autocast'):
        assert bad not in seg, '预取 worker 里出现了设备相关调用：%s' % bad
    assert 'sample_batch_numpy' in seg, \
        'worker 应只调用 dataset.sample_batch_numpy（纯 numpy）'