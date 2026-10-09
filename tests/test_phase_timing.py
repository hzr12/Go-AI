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
3. save/eval 发生在其打点之后，因此计入下一区间（墙钟口径正确）；
4. g/o/m（backward 之后的 clip / optimizer.step / EMA）按 **optimizer step**
   归一，不能除以 micro-batch 数 `_n_timed` —— accum>1 时会缩小 accum 倍，
   把要找的瓶颈读成「几乎不耗时」；
5. g 还要拆成 g1（unscale_）/ g2（clip 调用）/ g3（float(_gn) 的 D2H 同步）
   —— 不拆就分不清「clip 自己在同步」与「设备执行慢」，两者的药方相反。
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
    注意 _cuda_free 里的 synchronize 属设备选择、在 main 之前，
    是合法且必须保留的，故只检查 main 之后的区域。
    """
    m = re.search(r'^def main\(', CODE, re.M)
    assert m, '未找到 main()'
    body = CODE[m.start():]
    for bad in ('cuda.synchronize',):
        assert bad not in body, \
            f'main() 内出现 {bad}：会打断预取/双缓冲流水'
    # ⚠ 2026-10-08：这条断言的方向**反转**了。
    # 原来要求「main 之前保留 `cuda.synchronize`」，因为设备选择要用它探测空闲
    # 显存。但那个 synchronize（连同 `get_device_properties` /
    # `memory_allocated`）会 `_lazy_init()` 建 CUDA primary context，而设备
    # 选择发生在 **fork 预取 worker 之前** ⇒ worker 整份继承设备上下文
    # （A100 40G 上直接被 `_BatchPrefetcher` 的护栏拒绝启动，报错还指向错的地方）。
    # 现在设备选择一律走 NVML（`nvidia-smi`），所以这里**禁止**再出现。
    # 顺序不变量由 `tests/test_prefetch_fork_ordering.py` 用 AST 锁住。
    assert 'cuda.synchronize' not in CODE, \
        '全仓库不得再出现 cuda.synchronize：它会建 CUDA context，而 fork 出的' \
        '预取 worker 会继承它（A100 首次启动就是这么崩的）。用 nvidia-smi。'


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
    # 「齐全」= 紧跟其后必须依次是 g/o/m 那组、g1/g2/g3、_t_data_max、_n_timed、_n_opt。
    # 2026-10-07 加 g/o/m 后重置点从 3 行变 5 行，再拆 g1/g2/g3 变 6 行；这里按
    # **语句**判定而非正则，注释与空行不会插进来，也不会因「有人多加了一个计时器」
    # 而静默放行。
    _expect_after = [
        (['_t_clip', '_t_opt', '_t_ema'], '0.0'),
        (['_t_g1', '_t_g2', '_t_g3'], '0.0'),
        (['_t_data_max'], '0.0'),
        (['_n_timed'], '0'),
        (['_n_opt'], '0'),
    ]
    assert j + len(_expect_after) < len(metrics.body), \
        '重置后缺少 _t_data_max / _n_timed 等行'
    for k, (names, val) in enumerate(_expect_after, start=1):
        assert _assigned_names(metrics.body[j + k]) == names \
            and _assign_value(metrics.body[j + k]) == val, \
            f'重置后第 {k} 行必须是 {" = ".join(names)} = {val}'

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
    # 2026-10-07：日志行从 `d c s e dmax` 扩成 `d c g o m s e u dmax`。
    # g/o/m 是 backward() 之后的三段、u 是未归因余量 —— 真机 a_v7.4_npu2 上
    # 墙钟 12.0 s/step 而 c 只有 2.29 s，那 81% 正是靠这一行才第一次可见。
    # 同日再把 `g` 拆成 g1/g2/g3（13.09 s/step 里 g 独占 79.9%，但分不清是
    # clip 有病还是队列 drain）。格式串在源码里被拆成多行字面量，故分段断言。
    assert re.search(r'd=%.0f c=%.0f g=%.0f ', SRC), \
        '日志行缺少 d/c/g 字段'
    assert re.search(r'g1=%.0f g2=%.0f g3=%.0f', SRC), \
        '日志行缺少 g 的三段拆分（g1/g2/g3）'
    assert re.search(r'o=%.0f m=%.0f s=%.0f e=%.0f', SRC), \
        '日志行缺少 o/m/s/e 字段'
    assert re.search(r'u=%.0f dmax=%.0f ms', SRC), \
        '日志行缺少未归因 u 与 dmax'
    for key in ('"t_data_ms": _dms', '"t_comp_ms": _cms',
                '"t_save_ms": _sms', '"t_eval_ms": _ems'):
        assert key in SRC, f'swanlab 未上报 {key}'


def test_backward_tail_segments_are_measured():
    """g/o/m 三段必须起止齐全，且按 optimizer step 归一。

    动机见 `_t_clip` 初始化处的注释：旧口径只有 d/c/s/e，2026-10-07 真机
    实测 81% 的步时落在 `c` 结算之后 —— 即 backward 之后的 clip /
    optimizer.step / EMA，旧计时完全没看它们。

    归一口径是这里的**关键不变量**：accum>1 时这三段每个 optimizer step
    才发生一次，若除以 `_n_timed`（micro-batch 数）会缩小 accum 倍、
    被读成「这几段几乎不耗时」，正好把要找的瓶颈藏起来。
    """
    for tag in ('_t_clip0', '_t_opt0', '_t_ema0'):
        assert f'{tag} = time.perf_counter()' in SRC, f'缺少 {tag} 起点'
    for acc in ('_t_clip', '_t_opt', '_t_ema'):
        assert re.search(re.escape(acc) + r' \+= time\.perf_counter\(\) - ', SRC), \
            f'{acc} 未被累加'
    # 入口初始化（g1/g2/g3 紧跟在 g 后面、`_n_opt` 之前）
    assert re.search(
        r'_t_clip = _t_opt = _t_ema = 0\.0\s*\n'
        r'\s*_t_g1 = _t_g2 = _t_g3 = 0\.0\s*\n\s*_n_opt = 0', SRC), \
        '循环入口未初始化 g/o/m 与 g1/g2/g3 计数'
    # 归一必须走 `_n_opt`，不能走 `_n_timed`
    for acc, var in (('_t_clip', '_gms'), ('_t_opt', '_oms'), ('_t_ema', '_mms'),
                     ('_t_g1', '_g1ms'), ('_t_g2', '_g2ms'), ('_t_g3', '_g3ms')):
        assert re.search(
            re.escape(var) + r'\s*=\s*' + re.escape(acc)
            + r' \* 1000\.0 / _no', SRC), \
            f'{acc} 的归一未走 optimizer-step 口径（应为 … / _no，不是 / _nd）'
    # `u` = 墙钟 −(d+c+g+o+m+s+e)：未归因余量必须真的被算出来
    assert re.search(r'_ums = \(_wall_ms', SRC), '缺少未归因 u 的计算'


def test_g_is_split_into_three_contiguous_subsegments():
    """`g` 必须拆成 g1/g2/g3，且首尾相接、顺序固定（和恒等于 g）。

    动机（2026-10-07 真机 a_v7.3_npu2）：13.09 s/step 里 `g` 独占 10.45 s
    （79.9%），而 `u`=1 ms、`c`=2.48 s。但 `g` 内部混着三件性质完全不同的事：

      g1 = `unscale_` + 溢出探测 —— BF16 下 `use_scaler=False`，两者都应 ≈0；
           它 ≫0 就说明本以为跳过的路径其实没跳过。
      g2 = `clip_grad_norm_` 调用本身 —— 返回 device 张量 ⇒ 理论上纯发射；
      g3 = `float(_gn)` 的设备→主机同步 —— backward 之后**第一次**同步，整个
           待执行队列（前/反向实际执行、DDP all-reduce、clip kernel）在此 drain。

    不拆就分不清「clip 自己在同步」和「设备执行慢」，而这两种病的药方相反：
    `g2≫g3` ⇒ 改 `foreach=True`；`g3≫g2` ⇒ clip 无辜，去打 graph-compile
    与 channels-last 的 A/B。所以三段必须**紧邻无缝**——中间被塞进别的代码
    时 `g ≠ g1+g2+g3`，「g 独占 79.9%」就再也对不上账。
    """
    # g1 的起点刻意**就是** g 的起点（`_t_g1_0 = _t_clip0`），两段之间零缝隙；
    # g2/g3 才各自新取一次 perf_counter。
    for tag in ('_t_g2_0', '_t_g3_0'):
        assert f'{tag} = time.perf_counter()' in SRC, f'缺少 {tag} 起点'
    assert '_t_g1_0 = _t_clip0' in SRC, \
        'g1 起点必须与 g 起点同一时刻（否则 g ≠ g1+g2+g3）'
    for acc in ('_t_g1', '_t_g2', '_t_g3'):
        assert re.search(re.escape(acc) + r' \+= time\.perf_counter\(\) - ', SRC), \
            f'{acc} 未被累加'
    # 起点与结算必须严格交错排列。走 AST 而不是 SRC.index：注释里引用一次
    # `_t_g2_0 = ...` 就能让索引测试错位（本文件顶部那段教训的同类问题）。
    want = ['_t_clip0', '_t_g1_0', '_t_g1', '_t_g2_0',
            '_t_g2', '_t_g3_0', '_t_g3', '_t_clip']
    got = []
    for n in ast.walk(_tree()):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id in want:
            got.append((n.lineno, n.targets[0].id))
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) \
                and n.target.id in ('_t_g1', '_t_g2', '_t_g3', '_t_clip'):
            got.append((n.lineno, n.target.id))
    got.sort()
    assert [x[1] for x in got] == want, \
        'g 的起止必须按 clip0 → g1_0 → g1 → g2_0 → g2 → g3_0 → g3 → clip 排列，' \
        f'实际 {["%d:%s" % (ln, nm) for ln, nm in got]}'


def test_memory_line_reports_reserved_only():
    """`[step]` 行只打 reserved —— **刻意不**加 allocated/peak。

    这三个调用（`memory_allocated` / `max_memory_allocated` / `empty_cache`）在
    2026-10-01 加过一次又撤掉，理由两条：
      · **OOM 报错本身就是更好的报告**：它在压力最大那一刻给出 allocated +
        reserved + free，配合 `total` 就能反推 torch 之外的占用
        （`32.00 − 27.02 − 0.62 = 4.36 GiB`）。常打一个「平时的 reserved」信息更少。
      · `max_memory_allocated` 当时全仓库只有那一处、且在部分后端上没验证过，
        而它在日志路径上抛异常就是**第 50 步崩** —— 正好毁掉最需要那个数的
        时刻。观测不该有能力杀死被观测的进程。
    本测试锁住这个「刻意不加」的选择：将来有人看到只有一个数又想把三个加回来时，
    会先撞到这里。
    """
    assert re.search(r'mem=%\.2fGB', SRC), 'mem 行应打 reserved'
    assert 'max_memory_allocated' not in CODE, \
        'max_memory_allocated 全仓库不该出现（理由见 docstring）'
    # 只在**日志打点附近**禁 memory_allocated：`_auto_select_device` 里选最空的卡
    # 本来就在用 `torch.cuda.memory_allocated(idx)`，那是既有且必要的。
    i = CODE.find('mem=%.2fGB')
    window = CODE[max(0, i - 2500):i + 2500]
    assert 'memory_allocated' not in window, \
        '日志打点附近不应再出现 memory_allocated（见 docstring 的两条理由）'
    # OOM 恢复路径里的 empty_cache 是既有的、必须保留
    # OOM 恢复路径（`except` 分支）里那一次 empty_cache 必须还在。
    # 计数从 >=2 收紧成 >=1：NPU 后端下线前这个断言靠 `npu_empty_cache()` 凑够
    # 两处，现在 CUDA 只剩 OOM 分支这一处真实调用（另一处引用在注释里）。
    # 要守的是「它在」，不是「它有几处」。
    assert SRC.count('torch.cuda.empty_cache()') >= 1, \
        'OOM 恢复路径的 empty_cache 被误删'


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
