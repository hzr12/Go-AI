"""分段计时测试。

动机：4 卡 910A 实测 4.25 s/step。逐区间核对日志后确认速率极稳定
（8 个无 eval 区间全在 4.22~4.30 s/step，30 分钟零漂移），而 eval
每次仅约 1.7 秒、占比 0.2% —— 即约 96% 的时间是黑盒，无法判断瓶颈
在取数 / 算子 / 通信 / 保存。

另外：SwanLab 的 NPU Utilization 曲线无法用于归因。npu-smi 采样间隔
10~14s，远粗于振荡周期，2.5 秒与 60 秒的停顿在图上都渲染成同样宽的
缺口，密集折线多为混叠；且 `Aicore Usage Rate` 是内核执行时间占比、
不是效率（实测该值 85~100% 而实际仅 659 samples/s/卡）。

关键不变量：
1. 绝不插 synchronize() —— 会打断预取与双缓冲流水，反而更慢；
2. 累加器必须在打点处统一重置，否则跨区间累加导致数值虚高；
3. save/eval 发生在其打点之后，因此计入下一区间（墙钟口径正确）。
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()

# 去掉整行注释与 docstring 内的说明文字，避免把「解释旧公式为何错」的注释
# 误判成「旧公式仍在」，也避免把「不要插 synchronize」的注释当成调用。
CODE = '\n'.join(
    ln for ln in SRC.splitlines() if not ln.lstrip().startswith('#')
)

# --------------------------------------------------------------------------- #
# AST 定位工具
# --------------------------------------------------------------------------- #
# 为什么本文件的「顺序」类断言必须走 AST、不能走 SRC.index / 正则：
# 顺序类测试要表达的是**代码节点之间的位置关系**，而原始文本里注释与
# docstring 和真代码逐字同形。63345c7 给 compute_l2_report 写的说明里引了
# 一次 `if _do_stdout or _do_swanlab:`，于是 SRC.index 取到的是 docstring 的
# 偏移（1255 行）而非真正的打点块（2375 行），`assert init < blk` 直接变成
# `94572 < 48651` 而红 —— 一段**解释代码的说明文字**把一个检查代码位置的测试
# 打挂了。AST 里注释和 docstring 根本不是节点，所以下面的定位对「又有人写了
# 一段解释」彻底免疫；这正是这两个测试本来想表达的不变量。

_ZERO_CHAIN = ('_t_data', '_t_comp', '_t_save', '_t_eval')


def _tree():
    return ast.parse(SRC)


def _parents(tree):
    return {ch: par for par in ast.walk(tree)
            for ch in ast.iter_child_nodes(par)}


def _stmt_list_of(node, parents):
    """(node 所在的那个语句列表, 它的父节点) —— 即 node 的同级列表。

    从 node 往上找第一个「把 node 放在自己某个 list 字段里」的父节点；
    找不到返回 (None, None)。
    """
    while node in parents:
        par = parents[node]
        for _, val in ast.iter_fields(par):
            if isinstance(val, list) and any(v is node for v in val):
                return val, par
        node = par
    return None, None


def _uniques(nodes, what):
    assert len(nodes) == 1, \
        f'期望恰好 1 个{what}，实得 {len(nodes)} 个：' \
        + ', '.join(f'第 {n.lineno} 行' for n in nodes)
    return nodes[0]


def _assigned_names(node):
    """这条语句赋值给了哪些名字（链式赋值的全部 target 都算）。"""
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)) \
            and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _assign_value(node):
    return ast.unparse(node.value) if isinstance(node, ast.Assign) else None


def _if_nodes(tree, test_src):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.If) and ast.unparse(n.test) == test_src]


def _metrics_if(tree):
    """打点块 `if _do_stdout or _do_swanlab:`（stdout 与 swanlab 频率解耦）。"""
    return _uniques(_if_nodes(tree, '_do_stdout or _do_swanlab'),
                    '打点块 `if _do_stdout or _do_swanlab:`')


def _stdout_print_if(tree):
    """`if _do_stdout:` 里**含 spd_inst 打印行**的那一个。

    main() 里有两处 `if _do_stdout:`（分段耗时行 / 剖析结束提示），靠「第一个」
    定位会随排版漂移，所以按「体内含 spd_inst 的 logger.info」来认。
    """
    return _uniques(
        [n for n in _if_nodes(tree, '_do_stdout')
         if any('spd_inst' in ast.unparse(s) for s in n.body)],
        '含 spd_inst 打印的 `if _do_stdout:` 分支',
    )


def test_no_synchronize_in_training_path():
    """训练路径（main 及其之后）不得出现 synchronize。

    计时必须纯 CPU 侧。插 synchronize 会把预取与双缓冲流水打断，
    测出来的数还比真实更慢。
    注意 _cuda_free/_npu_free 里的 synchronize 属设备选择、在 main 之前，
    是合法且必须保留的，故只检查 main 之后的区域。
    """
    m = re.search(r'^def main\(', CODE, re.M)
    assert m, '未找到 main()'
    body = CODE[m.start():]
    for bad in ('npu.synchronize', 'cuda.synchronize'):
        assert bad not in body, \
            f'main() 内出现 {bad}：会打断预取/双缓冲流水'
    # 设备选择辅助函数在 main 之前，允许存在
    assert 'cuda.synchronize' in CODE[:m.start()], \
        '设备选择的 _cuda_free 应保留 synchronize（用于探测空闲显存）'


def test_all_four_accumulators_initialised():
    for name in ('_t_data', '_t_comp', '_t_save', '_t_eval'):
        assert name in SRC, f'缺少计时累加器 {name}'
    assert re.search(
        r'_t_data = _t_comp = _t_save = _t_eval = 0\.0\s*\n'
        r'\s*_t_data_max = 0\.0\s*\n\s*_n_timed = 0',
        SRC,
    ), '循环入口未初始化四个累加器'


def test_every_segment_is_measured():
    """四段都要有 perf_counter 起止并累加。

    t_data 走中间变量 _dt（因为还要拿它更新 _t_data_max 尖峰），
    其余三段直接内联 perf_counter，两种写法都算合法。
    """
    for tag in ('_t_data0', '_t_comp0', '_t_save0', '_t_eval0'):
        assert f'{tag} = time.perf_counter()' in SRC, f'缺少 {tag} 起点'
    for acc in ('_t_comp', '_t_save', '_t_eval'):
        assert re.search(re.escape(acc) + r' \+= time\.perf_counter\(\) - ', SRC), \
            f'{acc} 未被累加'
    assert re.search(r'_t_data \+= _dt', SRC) or re.search(
        r'_t_data \+= time\.perf_counter\(\) - ', SRC), '_t_data 未被累加'


def test_accumulators_reset_together_at_logging():
    """重置必须在打点处统一进行，且三行齐全；入口初始化在循环之外、之前。"""
    tree = _tree()
    parents = _parents(tree)
    metrics = _metrics_if(tree)       # `if _do_stdout or _do_swanlab:`
    out = _stdout_print_if(tree)      # `if _do_stdout:`（打印分段耗时那行）

    # --- 1) 重置在打点块内，且三行齐全、紧邻 ---------------------------------
    # 链式赋值 `_t_data = _t_comp = _t_save = _t_eval = 0.0` 的 4 个 target
    # 必须一个不少、值是 0.0，且必须是打点块的**直接**语句（不是藏在
    # 某个内层 if/try 里 —— 那样就不保证每次打点都重置了）。
    chains = [s for s in metrics.body
              if _assigned_names(s) == list(_ZERO_CHAIN)
              and _assign_value(s) == '0.0']
    assert len(chains) == 1, \
        f'打点块内应有且仅有一处累加器重置，实得 {len(chains)} 处'
    reset = chains[0]
    j = metrics.body.index(reset)
    # 「三行齐全」= 紧跟其后两条正是 _t_data_max / _n_timed（与原正则的
    # 「三行相邻」同义，但按语句判定，注释与空行不再能插进来）。
    assert j + 2 < len(metrics.body), '重置后缺少 _t_data_max / _n_timed 两行'
    assert _assigned_names(metrics.body[j + 1]) == ['_t_data_max'] \
        and _assign_value(metrics.body[j + 1]) == '0.0', \
        '重置后第 2 行必须是 _t_data_max = 0.0'
    assert _assigned_names(metrics.body[j + 2]) == ['_n_timed'] \
        and _assign_value(metrics.body[j + 2]) == '0', \
        '重置后第 3 行必须是 _n_timed = 0'

    # --- 2) 打点块与 stdout 打印是同级的先后两条 -----------------------------
    # 「同级」很关键：只有同级才能证明 out 紧跟在 metrics 之后，而不是恰好
    # 落在文件更后面的某个无关分支里。
    mlist, _ = _stmt_list_of(metrics, parents)
    assert mlist is not None and out in mlist, \
        'stdout 打印分支与打点块不在同一个语句列表里'
    assert mlist.index(metrics) < mlist.index(out), \
        '打点块必须排在 stdout 打印之前（reset 要覆盖到刚测完的这一段）'
    assert reset.lineno < out.lineno, \
        '重置位置应在打点块内、stdout 打印之前'

    # --- 3) 入口初始化：在**循环之外、循环之前** -----------------------------
    # 只比 lineno 是不够的：初始化若被挪进循环体，打点块之前也能满足
    # `init < blk`，但那样每步都会被清零、分段计时直接失效。所以这里断言
    # 「init 与 for 循环同级、且排在 for 之前」。
    others = [n for n in ast.walk(tree)
              if _assigned_names(n) == list(_ZERO_CHAIN)
              and _assign_value(n) == '0.0' and n is not reset]
    assert len(others) == 1, \
        f'除打点处重置外应还有且仅有一处入口初始化，实得 {len(others)} 处'
    init = others[0]

    node, loop = metrics, None
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.For):
            loop = node
            break
    assert loop is not None, '打点块不在任何 for 循环内，无法定义「循环入口」'
    llist, _ = _stmt_list_of(loop, parents)
    assert init in llist, \
        '入口初始化必须与 for 循环同级（在被它保护的循环体内=每步清零）'
    assert llist.index(init) < llist.index(loop), \
        '循环入口未初始化累加器（初始化必须早于 for）'


def test_data_wait_peak_is_tracked():
    """取数尖峰最能暴露数据管道饥饿，必须单独记 max。"""
    assert 'if _dt > _t_data_max:' in SRC
    assert '_dt = time.perf_counter() - _t_data0' in SRC
    assert '_dmax = _t_data_max * 1000.0' in SRC


def test_segments_logged_and_uploaded():
    assert re.search(r'd=%.0f c=%.0f s=%.0f e=%.0f dmax=%.0f ms', SRC), \
        '日志行缺少分段耗时字段'
    for key in ('"t_data_ms": _dms', '"t_comp_ms": _cms',
                '"t_save_ms": _sms', '"t_eval_ms": _ems'):
        assert key in SRC, f'swanlab 未上报 {key}'


def test_memory_line_reports_reserved_only():
    """`[step]` 行只打 reserved —— **刻意不**加 allocated/peak。

    这三个调用（`memory_allocated` / `max_memory_allocated` / `empty_cache`）在
    2026-10-01 加过一次又撤掉，理由两条：
      · **OOM 报错本身就是更好的报告**：它在压力最大那一刻给出 allocated +
        reserved + free，配合 `total` 就能反推 torch 之外的占用
        （`32.00 − 27.02 − 0.62 = 4.36 GiB`）。常打一个「平时的 reserved」信息更少。
      · `torch.npu.max_memory_allocated` 当时全仓库只有那一处、没在 torch_npu 2.1
        上验证过，而它在日志路径上抛异常就是**第 50 步崩** —— 正好毁掉最需要那个
        数的时刻。观测不该有能力杀死被观测的进程。
    本测试锁住这个「刻意不加」的选择：将来有人看到只有一个数又想把三个加回来时，
    会先撞到这里。
    """
    assert re.search(r'mem=%\.2fGB', SRC), 'mem 行应打 reserved'
    assert 'max_memory_allocated' not in CODE, \
        'max_memory_allocated 全仓库不该出现（当时只有那一处、未在 torch_npu 2.1 验证）'
    # 只在**日志打点附近**禁 memory_allocated：`_auto_select_device` 里选最空的卡
    # 本来就在用 `torch.npu.memory_allocated(idx)`，那是既有且必要的。
    i = CODE.find('mem=%.2fGB')
    window = CODE[max(0, i - 2500):i + 2500]
    assert 'memory_allocated' not in window, \
        '日志打点附近不应再出现 memory_allocated（见 docstring 的两条理由）'
    # OOM 恢复路径里的 empty_cache 是既有的、必须保留
    assert SRC.count('npu_empty_cache()') >= 2, 'OOM 恢复路径的 empty_cache 被误删'


def test_effective_batch_for_throughput_includes_accumulation():
    """吞吐口径的有效 batch 必须含梯度累积，否则日志把速度报成一半。

    `_eff_bs = bs * world_size` 漏乘 accumulation steps ⇒ 用了
    `--gradient-accumulation-steps 2` 时 `spd`/`spd_inst` 只有真实值的一半。
    而 2026-10-01 的降档方案（每卡 batch 减半 + 累积 2）**正是** accum=2，
    所以这条不修就会误导那次降档的判读。
    """
    m = re.search(r'_eff_bs = (.+)', SRC)
    assert m, '未找到 _eff_bs'
    expr = m.group(1)
    for factor in ('bs', 'world_size', '_accum_steps'):
        assert factor in expr, \
            f'_eff_bs 漏乘 {factor}：{expr.strip()}（accum>1 时吞吐会被报成假值）'


def test_eval_timing_wraps_the_dominant_cost():
    """eval 计时必须**只**包住 eval 调用本身（主导开销）。

    2026-10-04：V7 路径接上 ``evaluate_metrics_v7`` 后，周期 eval 变成
    「按 ``_v7_on`` 二选一」的两条调用。原正则要求 `_t_eval0` 后面**紧跟**
    ``metrics = evaluate_metrics(``，会把这条正常分派判成失败。

    放宽的只是「紧跟哪个名字」，**不变量没松**：仍然要求
    ``_t_eval0 = perf_counter()`` 与 ``_t_eval += … - _t_eval0`` 之间
    夹着一个真正的 eval 调用，且**不得**夹进别的大段逻辑（否则计时会
    把重排缩进的开销也算进去 —— 那正是这条测试当初要防的）。
    """
    m = re.search(
        r'_t_eval0 = time\.perf_counter\(\)\s*\n'
        r'(?:.*?\n)??'                       # 可选的 if/else 分派
        r'\s*metrics = (?:\(\s*)?'
        r'(?:evaluate_metrics_v7|evaluate_metrics)\('
        r'.*?\)\s*\n'
        r'\s*_t_eval \+= time\.perf_counter\(\) - _t_eval0',
        SRC, re.S,
    )
    assert m, 'eval 计时未包住 evaluate_metrics / evaluate_metrics_v7'
