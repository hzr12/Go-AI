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
    GC_REASON_NPU_LINEAR,
    resolve_grad_checkpoint,
)

_SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
            encoding='utf-8').read()

#: 整张策略表。`(开关, 期望的 gc, 期望的 reason)`；`grad_checkpoint` 固定传 1。
#:
#: 写成一张表而不是一条条 test，是为了让「NPU 那条优先于整模型那条」这条
#: **顺序不变量**和其余组合在同一处可见 —— 拆成独立 test 时，判据顺序写错
#: 反而可能各自都绿。reason 也在同一行被钉住，所以三个常量撞名会让本表红。
_POLICY = [
    ('无图编译', dict(), 1, None),
    ('整模型 compile（默认关 GC）',
     dict(compile_on=True), 0, GC_REASON_COMPILE_EXCLUSIVE),
    ('整模型 compile + gc-with-compile',
     dict(compile_on=True, gc_with_compile=True), 1, GC_REASON_GC_WITH_COMPILE),
    ('NPU 逐 Linear compile',
     dict(npu_linear_compile=True), 0, GC_REASON_NPU_LINEAR),
    # 下面两行是**不许被放行**的那两条：放行 = 启动不报错、第一个 step 才炸。
    ('NPU 逐 Linear + gc-with-compile（仍须关）',
     dict(npu_linear_compile=True, gc_with_compile=True), 0, GC_REASON_NPU_LINEAR),
    ('两种 compile 同开（逐 Linear 优先）',
     dict(compile_on=True, npu_linear_compile=True, gc_with_compile=True),
     0, GC_REASON_NPU_LINEAR),
]


@pytest.mark.parametrize('flags,exp_gc,exp_reason', _POLICY,
                         ids=[c[0] for c in _POLICY])
def test_policy_table(flags, exp_gc, exp_reason):
    assert resolve_grad_checkpoint(1, compile_on=False, npu_linear_compile=False,
                                   gc_with_compile=False, **flags) == (
                                       exp_gc, exp_reason)


def test_configured_zero_is_respected():
    """配置本身是 0 时不要被「无图编译」这条路径抬成 1。"""
    assert resolve_grad_checkpoint(
        0, compile_on=False, npu_linear_compile=False,
        gc_with_compile=False) == (0, None)
    assert resolve_grad_checkpoint(
        0, compile_on=True, gc_with_compile=True) == (0, GC_REASON_GC_WITH_COMPILE)


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
