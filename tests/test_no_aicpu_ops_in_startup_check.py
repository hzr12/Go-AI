"""启动期一致性自检不许用 AICPU 算子（2026-10-01 云端实测踩到）。

事故
----
`_assert_init_weights_identical`（DDP 改动新增的一行）用 `torch.equal` 逐位比对
各 rank 的参数 checksum：

    if not all(torch.equal(gathered[0], g) for g in gathered):

在 910A 上 `torch.equal` 派发到 **AICPU kernel**，直接把训练打死在启动期：

    EXCEPTION TASK: TGID=..., task type=aicpu kernel, ... error code=0x2a
    [Error]: The aicpu execution is abnormal.
    EH9999: rtStreamSynchronizeWithTimeout execute failed, ... 507018
    RuntimeError: ACL stream synchronize failed
      File "scripts/train_sft.py", line 341, in _assert_init_weights_identical
        if not all(torch.equal(gathered[0], g) for g in gathered):

讽刺的是这个函数的 docstring 本身就在讲「宁可走只 reduce 不 cast 那条路，把风险面
缩小」—— 原则正确，只是最后落到了另一个 AICPU 算子上。

修法：比对改成 `.item()` + Python float。`.item()` 是 D2H 拷贝、不是 AICPU 算子
（同一份代码里 `_read_log_scalars` 用了上百次，正常）；fp64 → Python float 是无损
fp64，相等判定与 `torch.equal` **逐位等价**。

本文件钉住「训练路径上不出现 `torch.equal`」—— 它在这套栈上是 AICPU。
"""
import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN = ROOT / 'scripts' / 'train_sft.py'
SRC = TRAIN.read_text(encoding='utf-8')
CODE = '\n'.join(ln for ln in SRC.splitlines() if not ln.lstrip().startswith('#'))


def _fn_node(fn):
    tree = ast.parse(SRC)
    return next(f for f in ast.walk(tree)
                if type(f) is ast.FunctionDef and f.name == fn)


def _fn_raw(name):
    return ast.get_source_segment(SRC, _fn_node(name)) or ''


def _fn_src(name):
    """函数源码，剥掉整行注释。

    本文件多处 docstring/注释里**故意**提到 `torch.equal` / `float64`（记录事故
    与理由），带着注释判会自己把自己判红。
    """
    src = _fn_raw(name)
    return '\n'.join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith('#'))


def test_no_torch_equal_anywhere_in_training_path():
    """整个训练脚本不得出现 `torch.equal(` —— 它在 910A 上是 AICPU kernel。

    全仓库只有一处用到过（就是这个自检），所以这里可以一刀切：将来谁为了「比对
    两个张量」写上它，就是把训练打死在启动期的又一步。
    """
    assert 'torch.equal(' not in CODE, \
        '训练路径出现 torch.equal —— 910A 上派发到 AICPU kernel，会在启动期炸掉'


def test_init_weight_check_compares_on_cpu_via_item():
    """自检必须用 `.item()` + Python 比较，且把 fp64 值取到主机侧。"""
    src = _fn_src('_assert_init_weights_identical')
    assert 'torch.equal' not in src
    assert '.item()' in src, '应通过 .item()（D2H 拷贝）把 checksum 取到主机侧比较'
    # 相等判定必须在主机侧的 Python float 上做
    assert 'float(g.item())' in src
    assert any(tok in src for tok in ('!= _vals[0]', '!= vals[0]')), \
        '应在主机侧 float 上做不等判定'


def test_checksum_stays_fp32_because_npu_has_no_fp64():
    """checksum 必须在 **fp32** 上累加 —— 910A 不支持 fp64。

    实测事故（2026-10-01 云端，两轮）：
      · 第一轮崩在 `torch.equal(...)`        ⇒ 我误判成「比较算子」的问题
      · 改用 `.item()` 后崩在 `.item()`      ⇒ 说明故障是**异步**的
      · 真正的源头是上游的 `sum(dtype=torch.float64)`：
            Warning: Device do not support double dtype now, dtype cast ...
            EXCEPTION TASK: task type=aicpu kernel ... error code=0x2a
        AICPU 故障由 fp64 算子排队，在后面第一次同步时才报出来 —— 栈指向哪里，
        错就在**哪里**被误判。
    """
    src = _fn_src('_assert_init_weights_identical')
    assert 'float64' not in src, \
        'checksum 不得用 fp64（910A 不支持；且 fp64 的 AICPU 故障是异步的，' \
        '会在后面某次同步时才报，栈看起来指向无关的算子）'
    assert 'sum(dtype=torch.float32)' in src, 'checksum 应在 fp32 上累加'
    assert '.to(torch.float32)' in src, 'all_gather 的 buf 应是 fp32（HCCL 原生支持）'
    assert '.double().sum()' not in src and '.double(' not in src, \
        '不要走 p.double().sum()（整份 fp64 物化，峰值显存 ×8，且 fp64 不可用）'


def test_no_fp64_anywhere_on_the_device_path():
    """整个训练脚本不得在设备侧用 fp64（`.double()` / `float64` / `torch.float64`）。

    910A 没有 fp64 硬件；任何设备侧 fp64 都会走「cast 成 fp32」的兜底，而实测表明
    这条兜底路径会挂 AICPU。主机侧（numpy / Python float）用 fp64 不受限 ——
    `compute_l2_report` 现在就是走 `float(sq)` 在主机侧算的。
    """
    code = CODE
    offenders = []
    for i, ln in enumerate(code.splitlines(), 1):
        if '.double()' in ln or 'torch.float64' in ln or 'dtype=torch.float64' in ln:
            offenders.append((i, ln.strip()[:70]))
    assert not offenders, \
        '设备路径出现 fp64（910A 不支持，实测会挂 AICPU）：%s' % offenders


def test_l2_report_accumulates_on_host_side():
    """`compute_l2_report` 的乘法/累加必须在主机侧 fp64（Python float）做。

    这不只是躲 AICPU：主机侧 fp64 比「设备侧 fp64 cast 后再乘」**更精确**
    （少一次设备侧舍入），所以 `test_log_loss_identity` 的恒等式容差不受影响。

    ⚠ 关键是加法**也**必须在主机侧。踩过一次：为了让「每次调用只同步一次」，
      一度写成 `torch.stack` 之后再在设备上把两组 fp32 标量相加 ——
      fp32 在组平方和量级（~600）上的 ulp ≈ 6.1e-5，一次加法就吃掉
      `test_l2_report_uses_decay_group_only` 的 1.2e-6 容差
      （实测 got=1200.0335 vs want=1200.0335471477）。等于把刚搬到主机侧的算术
      又拽回设备上，白搬。
    """
    src = _fn_src('compute_l2_report')
    assert 'sq.double()' not in src, \
        '不要用 sq.double()（设备侧 fp64）；先把标量读回主机侧再用 Python float'
    assert 'float(_sq)' in src and 'float(_wd)' not in src, \
        '乘加应写成 float(_sq) * _wd（_sq 已是主机侧 Python float）'
    assert 'float(total)' in src or '_total' in src, '最终仍返回 Python float'


def test_l2_report_reads_device_to_host_exactly_once():
    """`compute_l2_report` 每次调用**恰好一次**设备→主机读回。

    为什么这条是结构检查而不是探针：`torch.stack(_sqs).tolist()` 在 CPU 上**只
    派发 `aten.stack.default`** —— `.tolist()` 连算子都不派发（纯内存读），
    `tests/test_huber_loss.py::_count_host_syncs` 只匹配 `item`/`_local_scalar_dense`
    ⇒ 对它**结构性地看不见**（实测报 0）。所以那条测试只能断言 `<= 1`；
    「恰好一次」得在这里用 AST 数读回点。

    为什么这个不变量值得钉：本函数在训练循环里**每个 micro-batch 都无条件跑**
    （不在 `if _do_stdout or _do_swanlab:` 门控里），在 NPU 上每次读回都是一次
    排空设备流水线的阻塞。**耗时在 CPU 上测不出来，个数测得出来。**
    """
    fn = _fn_node('compute_l2_report')
    # 只数**无歧义**的设备→主机读回：`.tolist()` 与 `.item()`。
    # ⚠ `float(x)` 不静态计数：函数里合法的 `float(group.get('weight_decay'))`
    #   （dict → Python float）和 `float(_sq)`（`_sq` 来自 `.tolist()`，已是
    #   Python float）都是**零同步**，把它们算进去是误报。
    #   静态类型推断在这里不成立（不跑 mypy/pyright），所以留给
    #   `test_l2_report_never_reads_back_individually_per_group` 的结构断言：
    #   它证明 `float()` 的操作数来自 `zip(_wds, _vals)`，而 `_vals` 就是那次
    #   `.tolist()` 的结果 ⇒ 不可能是张量。
    reads = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        name = n.func.attr if isinstance(n.func, ast.Attribute) else ''
        if name in ('tolist', 'item'):
            reads.append((n.lineno, name))
    assert len(reads) == 1, (
        'compute_l2_report 应恰好一个设备→主机读回点，实得 %s —— '
        '多个读回点意味着每个 micro-batch 多次排空设备流水线' % reads)
    assert reads[0][1] == 'tolist', (
        '唯一读回点应是 torch.stack(...).tolist()（一发 D2H 读多个标量）；'
        '若换成逐组 float()/item() 就会变成多次同步，实得 %s' % (reads,))
    assert not any(n.func.attr == 'item' for n in ast.walk(fn)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)), \
        '不要用 .item()（每处都是一次独立同步，且在 NPU 上是排空流水线的阻塞读回）'


def test_l2_report_never_reads_back_individually_per_group():
    """反模式守卫：不得出现「对每个 decay 组各读一次」的写法。

    恒等式：`_build_param_groups` 的两个 decay 组 wd 相同，但**实现不得依赖这一点**
    去合并读回 —— 合并只能发生在**主机侧**（`float(_sq) * _wd` 逐组相乘再累加），
    不能靠 `_all = _sqs[0] + _sqs[1]` 这种设备侧 fp32 加法（见上面那条的实测数据）。
    """
    src = _fn_src('compute_l2_report')
    for bad in ('_sqs[0] + _sqs[1]', '_sq.double()', 'torch.equal('):
        assert bad not in src, 'compute_l2_report 出现反模式：%s' % bad
    assert 'zip(_wds, _vals)' in src, \
        '应把读回的标量与各组 wd 逐组在主机侧相乘（不要依赖 wd 相同而合并）'


def test_error_message_still_explains_both_causes():
    """报错文案必须同时说清两种可能（广播未生效 / NaN），别在修 bug 时丢掉。

    这条查**含注释**的原文：报错文案是给人看的，不该被剥注释。
    """
    src = _fn_raw('_assert_init_weights_identical')
    assert '一致性自检失败' in src
    assert '广播' in src, '应说明「广播没生效」这一可能'
    assert 'NaN' in src, '应说明 NaN/Inf 会被误报成 rank 不一致这一可能'