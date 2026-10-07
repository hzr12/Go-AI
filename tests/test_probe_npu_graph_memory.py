"""探针的失败采集：必须显示**内层**原因，并把 `BackendCompilerFailed` 归到 inductor 档。

2026-10-07 的 910C 实测里 `gc-ind` / `gc-ind-all` 报
``BackendCompilerFailed: backend='inductor' raised:``，而探针给出的结论是
「追踪阶段（Dynamo）就挂了 —— 这一层与设备无关，换成 NPU 也会原样复现」。
两处都错，且方向相反：

1. **原因被丢掉了。** `torch._dynamo.exc.BackendCompilerFailed.__init__` 把消息
   拼成两行（torch 源码）::

       msg = f"backend={self.backend_name!r} raised:\\n"
             f"{type(inner_exception).__name__}: {inner_exception}"

   真正的原因在 **第 1 行**（0-based index 1），而 `_err_line` 与 `_trace_tail`
   都只取 ``splitlines()[0]`` ⇒ 报告里只剩 ``backend='inductor' raised:``，
   「到底为什么挂」一个字都没有。

2. **档位归错。** `_fail_stage` 拿 ``torch._dynamo`` 去匹配
   ``torch._dynamo.exc.BackendCompilerFailed`` 于是返回 ``'dynamo'``。但这个异常
   **按定义**意味着 Dynamo 已经追踪成功、是 **backend 抛的** ⇒ 必须归
   ``'inductor'``。归成 ``'dynamo'`` 会推出「与设备无关 ⇒ NPU 也一样挂」，
   而本机恰恰就是 NPU —— 自相矛盾，还会掩盖「triton 没装」这类真因。
"""
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.probe_npu_graph_memory import (  # noqa: E402
    _backend_tag,
    _err_line,
    _fail_stage,
    _trace_tail,
)

INNER = "No module named 'triton'"


def _backend_failed(inner):
    """按**本机**签名构造 `BackendCompilerFailed`（2.1 两个位置参，2.12 多一个）。

    构造不出来就直接抛 —— 用模拟对象会让测试变成自说自话。
    """
    from torch._dynamo.exc import BackendCompilerFailed

    def inductor(*_a, **_k):
        pass

    vals = {'backend_fn': inductor, 'inner_exception': inner,
            'first_useful_frame': None}
    args = [vals[k] for k in inspect.signature(BackendCompilerFailed).parameters]
    return BackendCompilerFailed(*args)


def test_inner_cause_surfaces_in_err_line():
    """`_err_line` 不能只取首行 —— 首行是 `backend='inductor' raised:` 这句废话。"""
    e = _backend_failed(ModuleNotFoundError(INNER))
    line = _err_line(e)
    assert 'inductor' in line, line
    assert 'triton' in line, (
        '内层原因没显示出来，探针报告只会剩下 `backend=\'inductor\' raised:`：\n  %s' % line)


def test_inner_cause_surfaces_in_trace_tail():
    e = _backend_failed(ModuleNotFoundError(INNER))
    tail = _trace_tail(e)
    assert 'triton' in tail, 'trace_tail 同样只取了首行：%r' % tail


def test_backend_compiler_failed_is_inductor_stage_not_dynamo():
    """`BackendCompilerFailed` = 追踪成功、backend 抛 ⇒ 必须归 inductor 档。"""
    e = _backend_failed(ModuleNotFoundError(INNER))
    stage = _fail_stage(_err_line(e), _trace_tail(e))
    assert stage == 'inductor', (
        '归成 %r 会推出「与设备无关、NPU 也一样挂」—— 而本机就是 NPU。' % stage)


def test_real_dynamo_stage_error_still_classified_as_dynamo():
    """回归：真的 Dynamo 追踪期错误（context_fn 限制）仍要归 dynamo。"""
    err = ("NotImplementedError: checkpoint not implemented for "
           "<class 'torch._dynamo.variables.functions.NestedUserFunctionVariable'> "
           "context_fn")
    stage = _fail_stage(err, 'torch._dynamo.exc.InternalTorchDynamoError: ' + err)
    assert stage == 'dynamo', stage


def test_unwrapped_inductor_error_still_classified_as_inductor():
    """回归：torch 2.12 本机跑出来的是**没包壳**的 InductorError，也要归 inductor。"""
    err = 'RuntimeError: Compiler: cl is not found.'
    stage = _fail_stage(err, 'torch._inductor.exc.InductorError: ' + err)
    assert stage == 'inductor', stage


# ---------------------------------------------------------------- TorchAir / CANN

# 2026-10-07 910C 实测：TorchAir 后端的名字是整段 functools.partial 的 repr，
# 300+ 字节，恒排在 `backend=… raised:` 与真原因**之间**。
TORCHAIR_BACKEND_NAME = (
    "functools.partial(<function _npu_backend at 0x7f9e1c0d2f80>, "
    "compiler_config=<torchair.configs.options.Proto2IROptions>, "
    "decompositions={})")


def test_torchair_backend_repr_does_not_shove_out_the_real_cause():
    """TorchAir 的后端 repr 极长 —— 不能把它当成 `_err_line` 的全部内容。"""
    e = _backend_failed(RuntimeError('AHGraphCompileError: unsupported op'))
    # torch 2.1 的 `BackendCompilerFailed` 用 backend_fn 的 repr 当 backend_name
    # （functools.partial 没有 __name__），于是消息里夹着这段 300+ 字节的 repr。
    e.backend_name = TORCHAIR_BACKEND_NAME
    e.args = ('backend=%r raised:\nRuntimeError: AHGraphCompileError: '
              'unsupported op' % TORCHAIR_BACKEND_NAME,)

    line = _err_line(e)
    assert 'unsupported op' in line, (
        '真原因被那段 300+ 字节的 partial repr 挤没了：\n  %s' % line)
    assert len(line) <= 1200, len(line)
    assert _backend_tag(e) == 'torchair', _backend_tag(e)
    assert _fail_stage(line, _trace_tail(e), e) == 'torchair'
    assert 'inductor' not in line, 'TorchAir 的失败不能被显示成 inductor：%s' % line


CANN_MSG = '\n'.join([
    'E19999: Inner Error!',
    'E19999: [PID: 62] 2026-10-07-20:48:26.590.568 [Call][PreRun] Failed, '
    'graph_id:151, session_id:0.[FUNC: CompileGraph][FILE: graph_manager.cc]'
    '[LINE: 4512]',
    'TraceBack (most recent call last):',
    '[Compile][Graph] Compile graph failed. Unsupported operator: _masked_fill',
])


def test_cann_error_shows_the_last_line_not_the_empty_first_one():
    """CANN 的根因在**最后一行**，首行 `E19999: Inner Error!` 零信息量。

    2026-10-07 `block bs=800` 实测只打印了前 200 字节，看不到图编不过什么。
    """
    line = _err_line(RuntimeError(CANN_MSG))
    assert '_masked_fill' in line, (
        '只留首行 ⇒ 报告里只剩 `E19999: Inner Error!`，看不出哪张图编不过：\n  %s' % line)
    assert 'graph_manager.cc' in line, line


def test_cann_error_is_a_backend_stage_not_other():
    """`E19999` 是 CANN 图编译失败 ⇒ 已过追踪、属后端阶段。

    归成 `'other'` 会让结论段漏掉「必须在目标环境重测」这句。
    """
    e = RuntimeError(CANN_MSG)
    assert _fail_stage(_err_line(e), _trace_tail(e), e) == 'torchair'
    # 不传 e 也要能靠消息本身归档 —— 结论段可能只拿到已经截好的字符串。
    assert _fail_stage(_err_line(e), _trace_tail(e)) == 'torchair'
