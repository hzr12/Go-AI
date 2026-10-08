r"""「设备初始化之前」这条不变量：预取 worker 是 fork 出来的。

事故（A100 首次启动，2026-10-08）
---------------------------------
`--device cuda` 跑 A 段直接崩::

    RuntimeError: _BatchPrefetcher 不能在 torch.cuda 初始化之后构造：
    fork 出的 worker 会继承设备上下文，4 卡实测每卡凭空多占 ~24 GiB（OOM）。
    请把它挪到 init_process_group / set_device 之前。

而报错里指的地方**不是真正的原因** —— `_BatchPrefetcher` 一直就在
`init_process_group` 之前。真正的原因是 `main()` 开头为了打一行环境诊断调了
`torch.cuda.get_device_properties(0)`，那个调用会 `_lazy_init()` **建 CUDA
primary context**；诊断打完之后才去构造预取器，于是撞上自己的护栏。

**为什么之前从没炸**：同一段代码在 NPU 机上 `torch.cuda.is_available()` 为
False，整个 CUDA 分支被跳过 ⇒ 这是一条潜伏的、只在 CUDA 上成立的缺陷。
本仓此前的全量测试都在 CPU / NPU 上跑，所以它能一路绿灯地躺着。

为什么用「扫源码」而不是行为测试
--------------------------------
这条不变量是**关于 main() 里语句先后顺序**的，而 main() 不能在测试里真跑
（要数据集、要设备、要分布式）。仓库里同类 wiring 不变量也是用 AST 锁的
（`tests/test_ddp_early_stop_sync.py` 锁停止标志广播的位置），本文件是同一路子。

判据
----
扫 `main()` 里**构造 `_BatchPrefetcher` 之前**的那段文本，出现下列任一
「会建设备 context」的 torch.cuda 调用就红：

    get_device_properties / memory_allocated / synchronize / mem_get_info /
    set_device / current_device / init

注意刻意**不**禁 `torch.cuda.is_available()` 与 `device_count()`：它们走 NVML
查询、**不建 context**（`is_available()` 用 `_cuda_getDeviceCount`，不 `_lazy_init`），
是启动期判断后端的标准做法。
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_SRC_PATH = os.path.join(ROOT, 'scripts', 'train_sft.py')

#: 会建设备 context 的调用 ⇒ 出现在预取器之前就是 bug。
_CONTEXT_CREATING = (
    'get_device_properties', 'memory_allocated', 'synchronize', 'mem_get_info',
    'set_device', 'current_device', 'init',
)
_RE_CTX_CALL = re.compile(
    r'torch\.cuda\.(%s)\b' % '|'.join(_CONTEXT_CREATING))


def _main_src():
    with open(_SRC_PATH, encoding='utf-8') as fh:
        return fh.read()


def test_no_context_creating_cuda_call_before_the_prefetcher():
    src = _main_src()
    main = src[src.index('def main('):]
    cut = main.index('_BatchPrefetcher(dataset')
    offenders = sorted(set(_RE_CTX_CALL.findall(main[:cut])))
    assert not offenders, (
        'main() 里构造 _BatchPrefetcher **之前**出现了会建 CUDA context 的调用：%s\n'
        '它们会让 fork 出的预取 worker 继承设备上下文（4 卡 NPU 实测每卡凭空多占 '
        '~24 GiB），而 _BatchPrefetcher 的护栏会直接拒绝构造。\n'
        '启动期要读型号/算力/空闲显存请走 NVML（nvidia-smi），'
        '见 _cuda_name_and_capability。' % offenders)


def test_the_nvml_helper_does_not_touch_torch_cuda():
    """`_cuda_name_and_capability` 存在的全部意义就是不建 context。

    所以它内部**不许**出现任何 `torch.cuda.*` 调用 —— 一旦有人「顺手改回」
    `get_device_properties`，这条就红。
    """
    src = _main_src()
    start = src.index('def _cuda_name_and_capability(')
    body = src[start:src.index('\ndef ', start + 1)]
    assert not re.search(r'torch\.cuda\.', body), (
        '_cuda_name_and_capability 里出现了 torch.cuda.* 调用 —— '
        '那会重新引入 CUDA context 初始化')
    assert 'nvidia-smi' in body, '实现应走 nvidia-smi（NVML）'


def test_the_helper_parses_capability_and_survives_na():
    """`compute_cap` 在老驱动上返回 `[N/A]`；解析必须扛住，且不抛。"""
    import scripts.train_sft as t

    real_run = t.subprocess.run if hasattr(t, 'subprocess') else None
    assert real_run is not None or True  # subprocess 是函数内 import，见下

    # 函数内部是 `import subprocess`，所以直接用 monkeypatch 打在 subprocess 上
    import subprocess as sp

    class _R:
        returncode = 0
        stdout = 'NVIDIA A100-SXM4-40GB, 8.0\n'

    orig = sp.run
    try:
        sp.run = lambda *a, **k: _R()
        assert t._cuda_name_and_capability() == ('NVIDIA A100-SXM4-40GB', (8, 0))
        _R.stdout = 'NVIDIA A100-SXM4-40GB, [N/A]\n'
        name, cap = t._cuda_name_and_capability()
        assert name == 'NVIDIA A100-SXM4-40GB' and cap is None, (name, cap)
        _R.stdout = ''
        assert t._cuda_name_and_capability() == (None, None)
    finally:
        sp.run = orig


def test_prefetcher_guard_still_refuses_after_device_init():
    """护栏本身不能被顺手放宽 —— 它是这条不变量的执行者。"""
    import torch
    from scripts.train_sft import _BatchPrefetcher

    if not hasattr(torch, 'npu') and not torch.cuda.is_available():
        pytest.skip('本机既无 NPU 也无 CUDA，护栏分支不可达')
    import pytest
    with pytest.raises(RuntimeError, match='初始化之后构造'):
        _BatchPrefetcher(object(), num_workers=2)
