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
import re
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


def _fn_code(name):
    """函数源码：剥掉整行注释**与 docstring**，只剩可执行代码。

    结构断言（"不许出现某个调用/某个 dtype"）必须只看代码 —— docstring 里为了
    讲清事故会引用旧写法（`torch.equal`、`sum(dtype=torch.float64)`），那是应当
    鼓励的，否则每补一段事故说明就要改一次断言。用 `ast` 定位 docstring 的行区间
    再删，不用正则（正则会跨函数、也会被字符串里的三引号骗到）。
    """
    node = _fn_node(name)
    drop = set()
    first = node.body[0] if node.body else None
    if (first is not None and type(first) is ast.Expr
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        for i in range(first.lineno, (first.end_lineno or first.lineno) + 1):
            drop.add(i)
    # `node.lineno` 是**文件绝对行号**，而 `get_source_segment` 返回的片段是从
    # `def` 那一行开始的 —— 不减这个偏移就会拿片段的第 k 行去比绝对行号 k，
    # 于是 docstring 一行都删不掉（表现为"函数里出现了 docstring 里引用的旧写法"）。
    off = node.lineno - 1
    out = []
    for k, ln in enumerate(_fn_raw(name).splitlines(), 1):
        if (k + off) in drop or ln.lstrip().startswith('#'):
            continue
        out.append(ln)
    return '\n'.join(out)


def test_no_torch_equal_anywhere_in_training_path():
    """整个训练脚本不得出现 `torch.equal(` —— 它在 910A 上是 AICPU kernel。

    全仓库只有一处用到过（就是这个自检），所以这里可以一刀切：将来谁为了「比对
    两个张量」写上它，就是把训练打死在启动期的又一步。
    """
    assert 'torch.equal(' not in CODE, \
        '训练路径出现 torch.equal —— 910A 上派发到 AICPU kernel，会在启动期炸掉'


def test_init_weight_check_compares_on_cpu_via_item():
    """自检必须用 `.item()` + Python 比较，且把值取到主机侧比较。"""
    src = _fn_src('_assert_init_weights_identical')
    assert 'torch.equal' not in src
    assert '.item()' in src, '应通过 .item()（D2H 拷贝）把 checksum 取到主机侧比较'
    # 相等判定必须在主机侧的 Python float 上做（载荷是 hi/lo 两个 fp32，逐元素比）
    assert re.search(r'float\(\w+\.item\(\)\)', src), \
        '应在主机侧 float 上取值（形如 float(x.item())）'
    assert '!= _vals[0]' in src or '!= vals[0]' in src, \
        '应在主机侧 float 上做不等判定'


def test_checksum_reduces_on_the_host_but_ships_fp32():
    """归约在**主机 fp64**、过线一律 **fp32**。

    实测事故（2026-10-01 云端，两轮）：
      · 第一轮崩在 `torch.equal(...)`        ⇒ 我误判成「比较算子」的问题
      · 改用 `.item()` 后崩在 `.item()`      ⇒ 说明故障是**异步**的
      · 真正的源头是上游的 `sum(dtype=torch.float64)`：
            Warning: Device do not support double dtype now, dtype cast ...
            EXCEPTION TASK: task type=aicpu kernel ... error code=0x2a
        AICPU 故障由 fp64 算子排队，在后面第一次同步时才报出来 —— 栈指向哪里，
        错就在**哪里**被误判。

    2026-10-05（4×910A）补的一面：那次把 checksum 改成**设备侧 fp32** 之后，
    每次启动都误报「各 rank checksum 不一致」—— 差**一个 ulp**
    （`8125.2251` vs `8125.2256`；该档 ulp = `2**-11` = 4.88e-4）。根因是设备侧
    多块归约的分块顺序不保证跨 rank 一致。所以正确的形状是**两条都要**：
    fp64 挪到主机（躲开 AICPU，且噪声底从 1 ulp 降到 ~1e-12），
    过线仍用 fp32（HCCL 原生支持）。
    """
    src = _fn_code('_assert_init_weights_identical')
    assert 'dtype=torch.float64, device' not in src, \
        'fp64 不得作为设备张量的 dtype（910A 不支持；且 fp64 的 AICPU 故障是异步的，' \
        '会在后面某次同步时才报，栈看起来指向无关的算子）'
    assert '.sum(dtype=torch.float64)' in src, \
        'checksum 应在**主机侧**以 fp64 累加（先搬到 cpu 再求和）'
    assert 'sum(dtype=torch.float32)' not in src, \
        '不得在设备张量上直接 sum fp32 —— 分块顺序跨 rank 不一致，会末位误报'
    assert 'dtype=torch.float32' in src, 'all_gather 的 buf 应是 fp32（HCCL 原生支持）'
    assert '.double().sum()' not in src and '.double(' not in src, \
        '不要走 p.double().sum()（整份 fp64 物化，峰值显存 ×2）'


def test_no_fp64_anywhere_on_the_device_path():
    """训练脚本不得在**设备侧**用 fp64（`.double()` / `torch.float64`）。

    910A 没有 fp64 硬件；任何设备侧 fp64 都会走「cast 成 fp32」的兜底，而实测表明
    这条兜底路径会挂 AICPU。主机侧（numpy / Python float）用 fp64 不受限 ——
    `compute_l2_report` 现在就是走 `float(sq)` 在主机侧算的，
    `_assert_init_weights_identical` 也是（见下面那条测试的 docstring）。

    判据按行匹配，并在**该行先把张量搬回主机**时放行：搬回主机之后 fp64 只存在于
    CPU，910A 完全看不到它。这不是宽松化 —— 没有主机搬移的 fp64 行照样会被判红
    （`p.double().sum()`、`sum(dtype=torch.float64)` 直接作用在参数上都会被抓住）。
    """
    host_marks = ('.cpu()', "'cpu'", '"cpu"')
    offenders = []
    for i, ln in enumerate(CODE.splitlines(), 1):
        if '.double()' not in ln and 'torch.float64' not in ln \
                and 'dtype=torch.float64' not in ln:
            continue
        if any(mark in ln for mark in host_marks):
            continue                      # 已搬回主机，fp64 不经过 910A
        offenders.append((i, ln.strip()[:70]))
    assert not offenders, \
        '设备路径出现 fp64（910A 不支持，实测会挂 AICPU）：%s' % offenders


def test_l2_report_accumulates_on_host_side():
    """`compute_l2_report` 的乘法/累加必须在主机侧 fp64（Python float）做。

    这不只是躲 AICPU：主机侧 fp64 比「设备侧 fp64 cast 后再乘」**更精确**
    （少一次设备侧舍入），所以 `test_log_loss_identity` 的恒等式容差不受影响。

     关键是加法**也**必须在主机侧。踩过一次：为了让「每次调用只同步一次」，
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
    # `float(x)` 不静态计数：函数里合法的 `float(group.get('weight_decay'))`
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