r"""SFT 侧「图编译 × 梯度检查点」的兼容策略（`resolve_grad_checkpoint`）。

为什么单独立一个文件
--------------------
这条策略决定**显存能不能压住**，而 40GB 的 A100 上它就是吞吐的第一杠杆：
关掉 GC 意味着激活全驻留、batch 被迫调小，batch 缩小吃掉的吞吐通常远大于
融合省下的。所以「开 compile 时能不能保住 GC」必须被钉住，而不是靠注释。

两种图编译的兼容性不同，这是本文件的核心事实
------------------------------------------------
* `--npu-graph-compile`（`_compile_linear_submodules` 逐 ``nn.Linear`` 替换）
  的编译产物**落在检查点段内部**，`backbone.
  assert_grad_checkpoint_compile_compatible` 会在第一次前向扫到 `_orig_mod`
  并抛 —— 真的互斥。
* `--compile`（整模型 ``torch.compile(model)``）把原模型包在 OptimizedModule
  **内层**，守卫扫子树看不到 ``_orig_mod``；检查点又是 ``use_reentrant=False``
  （``backbone._checkpointed`` 已经是），与 compile 兼容。

所以「NPU 那条不许被 ``--gc-with-compile`` 放行」是必须守住的不变量：放行了就是
「启动不报错、第一个 step 前向才炸」——最难查的那种失败。

配套的端到端证据在 ``tests/test_katago_v7_grad_checkpointing.py::
test_grad_checkpoint_survives_aot_autograd``（真跑一遍 aot_autograd）。
"""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.train_sft import (  # noqa: E402
    GC_REASON_COMPILE_EXCLUSIVE,
    GC_REASON_GC_WITH_COMPILE,
    resolve_grad_checkpoint,
)

_SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
            encoding='utf-8').read()

#: 整张策略表。`(开关, 期望的 gc, 期望的 reason)`；`grad_checkpoint` 固定传 1。
#:
#: 写成一张表而不是一条条 test，是为了让所有组合在同一处可见。
#: reason 也在同一行被钉住，所以两个常量撞名会让本表红。
#:
#: ⚠ 2026-10-08 表里少了「NPU 逐 Linear 编译」那三行：整个 NPU 后端连同
#:   `--npu-graph-compile` 已删除（硬件迁到 A100 / V100）。原先那条不变量
#:   「NPU 那条不许被 `--gc-with-compile` 放行」随之后端一起消失 —— 现在
#:   **只剩一种编译形态**（整模型 `torch.compile`），而它与 GC 是兼容的，
#:   所以 `resolve_grad_checkpoint` 只剩两个 reason。
_POLICY = [
    ('无图编译', dict(), 1, None),
    ('整模型 compile（默认关 GC）',
     dict(compile_on=True), 0, GC_REASON_COMPILE_EXCLUSIVE),
    ('整模型 compile + gc-with-compile',
     dict(compile_on=True, gc_with_compile=True), 1, GC_REASON_GC_WITH_COMPILE),
]


def test_npu_backend_is_gone_from_the_compile_policy():
    """NPU 已整体移除，`resolve_grad_checkpoint` 不得再留那条互斥分支。

    这条钉的是「删干净了」而不是行为：若有人把 `--npu-graph-compile` 复活，
    签名里就会多回一个维度，而**那一维必须重新决定是否放行 GC**（逐 Linear
    编译的产物落在检查点段内部，真的互斥）。这里提醒的是那条推理，不是禁令。
    """
    import inspect
    import scripts.train_sft as t
    params = inspect.signature(t.resolve_grad_checkpoint).parameters
    assert 'npu_linear_compile' not in params, (
        'resolve_grad_checkpoint 又出现了 npu_linear_compile —— 若 NPU 逐 Linear '
        '编译被复活，必须同时决定它与 --gc-with-compile 的关系（真互斥，不放行）')
    assert not hasattr(t, 'GC_REASON_NPU_LINEAR')
    assert "--npu-graph-compile" not in _SRC, \
        '训练脚本里还有 --npu-graph-compile（后端已删除）'


@pytest.mark.parametrize('flags,exp_gc,exp_reason', [c[1:] for c in _POLICY],
                         ids=[c[0] for c in _POLICY])
def test_policy_table(flags, exp_gc, exp_reason):
    kw = dict(compile_on=False, gc_with_compile=False)
    kw.update(flags)
    assert resolve_grad_checkpoint(1, **kw) == (exp_gc, exp_reason)


def test_configured_zero_is_respected():
    """配置本身是 0 时不要被「无图编译」这条路径抬成 1。"""
    off = dict(compile_on=False, gc_with_compile=False)
    assert resolve_grad_checkpoint(0, **off) == (0, None)
    assert resolve_grad_checkpoint(0, **dict(off, compile_on=True,
                                              gc_with_compile=True)) == (
                                                  0, GC_REASON_GC_WITH_COMPILE)


def test_new_flags_default_to_the_historical_behaviour():
    """两个新旗都不能改变「命令行什么都不给」时的行为。

    `--gc-with-compile` 默认 0（照旧关 GC）；`--compile` 默认 None（按设备自动）
    而不是写死 0/1 —— 写死就分不出「用户显式给了」与「没给」，自动判定无从下手。
    """
    assert "add_argument('--gc-with-compile', type=int, default=0" in _SRC
    assert "add_argument('--compile', type=int, default=None" in _SRC


def test_compile_mode_offers_the_cudagraph_free_autotune():
    """A100 40G 上 `max-autotune` 会带 CUDA Graphs，私有内存池不归还。

    所以必须提供 `max-autotune-no-cudagraphs`，**且它得真的在 `--compile-mode`
    的 `choices` 里** —— 只在别处提一句等于没有。旧的三个值一个都不能丢。
    """
    m = re.search(r"add_argument\('--compile-mode'.*?choices=\[(?P<c>[^\]]*)\]",
                  _SRC, re.S)
    assert m, '找不到 --compile-mode 的 choices'
    choices = m.group('c')
    for want in ('default', 'max-autotune', 'max-autotune-no-cudagraphs',
                 'reduce-overhead'):
        assert "'%s'" % want in choices, \
            '--compile-mode 的 choices 里没有 %r（实际: %s）' % (want, choices)
