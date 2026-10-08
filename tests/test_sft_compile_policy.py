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
* `--compile`（整模型 `torch.compile(model)`）把原模型包在 OptimizedModule
  **内层**，守卫扫子树看不到 `_orig_mod`；检查点又是 `use_reentrant=False`
  （`backbone._checkpointed` 已经是），与 compile 兼容。

所以「NPU 那条不许被 `--gc-with-compile` 放行」是必须守住的不变量：放行了就是
「启动不报错、第一个 step 前向才炸」——最难查的那种失败。

配套的端到端证据在 `tests/test_katago_v7_grad_checkpointing.py::
test_grad_checkpoint_survives_aot_autograd`（真跑一遍 aot_autograd）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.train_sft as t  # noqa: E402
from scripts.train_sft import (  # noqa: E402
    GC_REASON_COMPILE_EXCLUSIVE,
    GC_REASON_GC_WITH_COMPILE,
    GC_REASON_NPU_LINEAR,
    resolve_grad_checkpoint,
)


def _resolve(**kw):
    base = dict(compile_on=False, npu_linear_compile=False, gc_with_compile=False)
    base.update(kw)
    return resolve_grad_checkpoint(1, **base)


def test_no_graph_compile_keeps_the_configured_value():
    assert _resolve() == (1, None)


def test_configured_zero_stays_zero():
    assert resolve_grad_checkpoint(
        0, compile_on=False, npu_linear_compile=False,
        gc_with_compile=False) == (0, None)


def test_whole_compile_disables_gc_by_default():
    """历史行为：`--compile 1` 且没显式要求 ⇒ GC 让位。"""
    assert _resolve(compile_on=True) == (0, GC_REASON_COMPILE_EXCLUSIVE)


def test_whole_compile_keeps_gc_when_asked():
    """`--compile 1 --gc-with-compile 1` ⇒ 两者同时开（A100 40G 的正路）。"""
    assert _resolve(compile_on=True, gc_with_compile=True) == (
        1, GC_REASON_GC_WITH_COMPILE)


def test_npu_linear_compile_always_wins_over_gc_with_compile():
    """逐 Linear 编译 + `--gc-with-compile 1` ⇒ **仍然**关 GC。

    这条是不变量：放行的话启动不报错、第一个 step 前向才抛
    `assert_grad_checkpoint_compile_compatible`。
    """
    for gwc in (False, True):
        assert _resolve(npu_linear_compile=True, gc_with_compile=gwc) == (
            0, GC_REASON_NPU_LINEAR), 'gc_with_compile=%r 竟放行了 NPU 那条' % gwc


def test_npu_linear_compile_wins_even_when_whole_compile_is_also_on():
    """两个都开时仍以逐 Linear 为准 —— 判据顺序不能被「谁先判」左右。"""
    assert _resolve(compile_on=True, npu_linear_compile=True,
                    gc_with_compile=True) == (0, GC_REASON_NPU_LINEAR)


def test_reasons_are_distinct():
    """三个 reason 常量必须互不相同 —— main() 按 reason 分派打日志，
    撞名会让两条消息互相顶掉。"""
    got = {GC_REASON_NPU_LINEAR, GC_REASON_COMPILE_EXCLUSIVE,
           GC_REASON_GC_WITH_COMPILE}
    assert len(got) == 3, got


def test_cli_defaults_preserve_the_historical_behaviour():
    """CLI 默认值：新增的两个开关都不能改变「不传参数时的行为」。

    `--compile` 的 default 是 None（= 按设备自动），所以这里只钉 gc-with-compile：
    它必须是 0，否则所有既有命令行会突然开始「compile + GC 同时开」。
    """
    assert t.GC_REASON_GC_WITH_COMPILE, '常量应可从模块导入'
    src = open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
               encoding='utf-8').read()
    assert "add_argument('--gc-with-compile', type=int, default=0" in src, (
        '--gc-with-compile 的默认值必须是 0（照旧关 GC），否则既有命令行行为突变')
    assert "add_argument('--compile', type=int, default=None" in src, (
        '--compile 必须是 None（按设备自动）而不是写死 0/1 —— '
        '写死就分不出「用户显式给了」与「没给」')


def test_compile_mode_offers_the_cudagraph_free_autotune():
    """A100 40G 上 `max-autotune` 会带 CUDA Graphs，私有内存池不归还。

    所以必须提供 `max-autotune-no-cudagraphs` 这个选项，**且它得真的在
    `--compile-mode` 的 `choices` 里** —— 只在别处提一句等于没有。
    """
    import re
    src = open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
               encoding='utf-8').read()
    m = re.search(
        r"add_argument\('--compile-mode'.*?choices=\[(?P<c>[^\]]*)\]",
        src, re.S)
    assert m, '找不到 --compile-mode 的 choices'
    choices = m.group('c')
    for want in ('default', 'max-autotune', 'max-autotune-no-cudagraphs',
                 'reduce-overhead'):
        assert "'%s'" % want in choices, (
            '--compile-mode 的 choices 里没有 %r（实际: %s）' % (want, choices))
    # 旧的三个值一个都不能丢：既有命令行还指着它们
    assert choices.count('max-autotune') >= 1
