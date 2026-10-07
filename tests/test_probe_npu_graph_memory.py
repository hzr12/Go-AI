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
import contextlib
import inspect
import io
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.probe_npu_graph_memory import (  # noqa: E402
    _backend_tag,
    _err_line,
    _fail_stage,
    _full_trace,
    _project_frames,
    _trace_tail,
    batch_ceiling,
    print_throughput,
    workspace_pairs,
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


# --------------------------------------------------------------------------- #
# ③ 吞吐段
# --------------------------------------------------------------------------- #
def _row(tier, batch, elapsed, steps=2, **extra):
    r = {'tier': tier, 'batch': batch, 'steps': steps, 'elapsed': elapsed,
         'losses': [1.0, 2.0], 'grad_norms': [123.0]}
    r.update(extra)
    return r


def _throughput(*rows):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print_throughput(list(rows))
    return out.getvalue()


def test_throughput_does_not_depend_on_compile_tiers():
    """**回归**：编译档一档没跑时，③ 也必须打印。

    这段原先整块挂在 `if ok_rows:` 里 —— 而目前每一轮 `gc-torchair` 都在
    backend 阶段抛异常，`ok_rows` 恒空 ⇒ ③ **静默消失**，于是
    「GC 关 vs 开 谁快（samples/s）」这个 spd 本义的问题，在最需要它的那一轮
    反而没有数据。基线档的吞吐跟编译能不能跑通无关。
    """
    text = _throughput(_row('gc', 800, 40.0), _row('eager', 800, 33.0))
    assert '③' in text, '编译档全挂 ⇒ ③ 整段没打印：%r' % text
    assert 'samples/s' in text, text
    # 同 batch 的比值：800*2/33 ÷ 800*2/40 = 1.21
    assert '×1.21 vs gc' in text, text


def test_throughput_skips_rows_that_threw():
    """失败/中断的行没有可比的 elapsed，混进来会把「档位效应」算成噪声。"""
    text = _throughput(_row('gc', 800, 40.0),
                       _row('gc-torchair', 800, 1.0, err='boom'),
                       _row('eager', 800, 1.0, oom=True))
    assert 'gc-torchair' not in text, text
    # 只剩基线 ⇒ 不该出现「每档最优吞吐」（单档比不出档位差异）
    assert '每档最优吞吐' not in text, text
    assert 'eager' not in text, text


def test_throughput_silent_when_nothing_measured():
    out = _throughput()
    assert out == '', out


# --------------------------------------------------------------------------- #
# `--full-trace`
# --------------------------------------------------------------------------- #
def _deep(depth):
    """递归帧**故意两两交替**：Python 的 traceback 会把连续同
    `(file, line, name)` 的帧折叠成 ``[Previous line repeated N more times]``
    （实测 `_deep` 单函数 80 层只打出 8 行），那样就测不到「默认不截断」。
    """
    if depth:
        return _deep_alt(depth - 1)
    raise ValueError('boom')


def _deep_alt(depth):
    if depth:
        return _deep(depth - 1)
    raise AssertionError('unreachable')


def _raises(depth):
    try:
        _deep(depth)
    except ValueError as e:
        return e
    raise AssertionError('unreachable')


def test_full_trace_is_not_truncated_by_default():
    """默认必须**全打**。

    原先默认 `limit_lines=90` ⇒ head 30 + tail 60，**中间被删**。而 FakeTensor
    那条报错「哪个算子从哪来」就在 innermost 栈的中段 —— 砍掉就只能再跑一轮
    （910C 上十几分钟）。日志走 tee，几百行不构成问题。
    """
    e = _raises(40)
    want = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    t = _full_trace(e)
    assert '省略' not in t, '默认仍被截断：%r' % t[:400]
    assert t == want.rstrip(), '默认应当原样输出整条异常链'


def test_full_trace_still_honors_an_explicit_limit():
    t = _full_trace(_raises(40), limit_lines=30)
    assert '中间省略' in t, t[:300]


def test_project_frames_find_repo_frames():
    """本仓库的帧要单独捞出来 —— 它们是最先被截断删掉的那段。"""
    fr = _project_frames(_raises(40))
    # 只比文件名：Windows 上 `path[len(root):].lstrip('/\\')` 给出的是反斜杠。
    assert any('test_probe_npu_graph_memory.py' in x for x in fr), fr
    assert any('in _deep' in x for x in fr), fr


# --------------------------------------------------------------------------- #
# 第 2/3 项
# --------------------------------------------------------------------------- #
def _mrow(tier, batch, peak, n_compiled=1, oom=False, err=None):
    return {'tier': tier, 'batch': batch, 'peak': peak,
            'n_compiled': n_compiled, 'oom': oom, 'err': err}


def test_workspace_reports_every_batch_not_just_the_first():
    """**回归**：原实现在内层循环打完一个就 `break`。

    `sorted(batches)` 升序 ⇒ 永远只报最小那个，bs=800 被整条吞掉
    （2026-10-07 实测只打出 `linear bs=400 +0.02 GB`，bs=800 的 +0.00 不见了）。
    """
    rows = [_mrow('eager', 400, 25.64e9), _mrow('eager', 800, 51.02e9),
            _mrow('linear', 400, 25.66e9), _mrow('linear', 800, 51.02e9)]
    pairs = workspace_pairs(rows, ['eager', 'linear'], [800, 400])
    assert [(p[0], p[1]) for p in pairs] == [('linear', 400), ('linear', 800)], pairs
    assert pairs[1][2] == 0.0, pairs[1]


def test_workspace_pairs_with_the_same_gc_state_baseline():
    """GC-off 减 `eager`、GC-on 减 `gc` —— 拿错基线会把省下的显存算成 workspace。"""
    rows = [_mrow('gc', 800, 6.64e9), _mrow('gc-torchair', 800, 6.70e9)]
    pairs = workspace_pairs(rows, ['gc-torchair'], [800])
    assert len(pairs) == 1, pairs
    tier, bs, delta, base, _n = pairs[0]
    assert (tier, bs, base) == ('gc-torchair', 800, 'gc'), pairs[0]
    assert abs(delta - 0.06e9) < 1e3, delta


def test_workspace_skips_tiers_that_do_not_compile():
    assert workspace_pairs([_mrow('eager', 800, 1.0)], ['eager'], [800]) == []


def test_batch_ceiling_reports_the_largest_success():
    """**回归**：原来取 `min(ok)` ⇒ 测了 800/400 都过却打「≥ 400」，低估一半。"""
    rows = [_mrow('gc', 800, 1.0), _mrow('gc', 400, 1.0),
            _mrow('block', 800, None, err='E19999'),
            _mrow('block', 400, None, err='E19999'),
            _mrow('whole', 1600, None, oom=True)]
    best, oom, fail = batch_ceiling('gc', rows)
    assert best == 800, best
    assert oom == [] and fail == []

    best, oom, fail = batch_ceiling('block', rows)
    assert best is None and fail == [800, 400], (best, fail)

    best, oom, fail = batch_ceiling('whole', rows)
    assert best is None and oom == [1600], (best, oom)


def test_batch_ceiling_does_not_report_skipped_as_oom():
    """被跳过的档位记的是 `oom='skipped'` —— 打成「OOM」就是把没跑说成跑挂。"""
    rows = [_mrow('gc-torchair', 4, None, oom='skipped')]
    best, oom, fail = batch_ceiling('gc-torchair', rows)
    assert (best, oom, fail) == (None, [], []), (best, oom, fail)
