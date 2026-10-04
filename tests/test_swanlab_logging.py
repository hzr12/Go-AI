"""SwanLab 上报频率与日志同步点测试。

两个问题：
1. `--log-every` 同时控制 stdout 和 swanlab（train_sft.py 同一 if 分支），
   曲线只有 log_every 的密度——log-every 50 时一个 epoch 仅约 200 个点，
   看不出细节。新增 `--swanlab-every`（默认 0 = 跟随 log_every）把两者解耦。
2. 同一打点里 loss/policy_loss/value_loss 各被 `.item()` 取了两次
   （一次给 stdout、一次给 swanlab），共 6 次设备同步，其中 3 次完全重复。
   改为单一同步点后每打点只同步 3 次，且两处用的是同一份值。

本测试覆盖频率决策、单同步点、以及若干结构性不变量。
"""
import ast
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_sft import _read_log_scalars, _should_log

SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()

# --------------------------------------------------------------------------- #
# AST 定位工具
# --------------------------------------------------------------------------- #
# 为什么本文件「打点/上报在哪」这类断言必须走 AST、不能走 find / 正则 / 定长窗口：
# 那三者都是在**整份原文**上扫的，注释与 docstring 和真代码逐字同形，所以「谁先
# 出现」根本不是「谁在执行」。63345c7 给 compute_l2_report 写的说明里引了一次
# `if _do_stdout or _do_swanlab:`，把 22f8f07 修好的两个测试重新打挂；本文件当时
# 被记成「一个真实隐患」而未修。
#
# 本文件比那两个还脆一层：`main()` 里有 **4 处** `swanlab_logger.log(`——打点 /
# eval / 早停 / 收尾——而 `src.find('swanlab_logger.log(')` 取的是**第一个**，
# 今天取对纯属排版运气。往前加一条提到该串的注释，测试就改看别处：要么对着一段
# 说明文字报红，要么把 400 字符窗口挪到不含真正调用体的位置、把真实缺陷放过去。
#
# AST 里注释与 docstring 不是节点，定位对「又有人写了一段解释」彻底免疫；而
# 「4 处里我要哪一处」也必须**指名**，不能靠行序默认——下面的定位器一律
# 「按内容认领 + 要求唯一」，唯一性不成立时直接报红并说清有哪几处、要认谁。

_LOSS_KEYS = ('loss', 'policy_loss', 'value_loss')


def _main_tree():
    """main() 的语法树（只取这一个函数，断言范围与原 inspect.getsource 一致）。"""
    for n in ast.parse(SRC).body:
        if isinstance(n, ast.FunctionDef) and n.name == 'main':
            return n
    raise AssertionError('scripts/train_sft.py 里没有 main()')


def _uniques(nodes, what):
    assert len(nodes) == 1, \
        f'期望恰好 1 个{what}，实得 {len(nodes)} 个：' \
        + ', '.join(f'第 {n.lineno} 行' for n in nodes)
    return nodes[0]


def _parents(tree):
    return {ch: par for par in ast.walk(tree)
            for ch in ast.iter_child_nodes(par)}


def _stmt_list_of(node, parents):
    """(node 所在的语句列表, 父节点) —— 即 node 的同级列表。

    从 node 往上找第一个「把 node 放进自己某个 list 字段里」的父节点；找不到
    返回 (None, None)。用它断言「两条语句同级且有先后」，比文本偏移可靠。
    """
    while node in parents:
        par = parents[node]
        for _, val in ast.iter_fields(par):
            if isinstance(val, list) and any(v is node for v in val):
                return val, par
        node = par
    return None, None


def _if_nodes(tree, test_src):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.If) and ast.unparse(n.test) == test_src]


def _attr_uses(tree, obj, attr):
    """`obj.attr` 形式的全部属性访问节点（不限于是否被调用）。"""
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == attr
            and isinstance(n.value, ast.Name) and n.value.id == obj]


def _calls_attr(tree, obj, attr):
    """`obj.attr(...)` 的全部调用节点。"""
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr and isinstance(n.func.value, ast.Name)
            and n.func.value.id == obj]


def _calls_name(node, name):
    """裸函数名 `name(...)` 的全部调用节点（`node` 可以是语句树，也可以是任一表达式）。"""
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name]


def _assign_names(stmt):
    """这条赋值写入了哪些名字（解包 `a, b = ...` 的每个元素都算进去）。

    注意 `_lv, _pv, _vv = ...` 在 AST 里是**一个** Tuple target，不是三个 target，
    只看 `stmt.targets` 会拿到空列表。
    """
    return [n.id for tgt in stmt.targets
            for n in ast.walk(tgt)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]


def _call_dict(call):
    """调用首个位置实参是 dict 字面量时返回 {key: 值源码}；否则 None。"""
    if not call.args:
        return None
    a0 = call.args[0]
    if not (isinstance(a0, ast.Dict) and a0.keys):
        return None
    return {k.value: ast.unparse(v) for k, v in zip(a0.keys, a0.values)
            if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _train_log_call(tree):
    """**上报打点标量**的那次 `swanlab_logger.log(` —— 字典里带 loss 三项者。

    main() 里有 4 处 `swanlab_logger.log(`：打点 / eval / 早停 / 收尾。只有
    打点那处传 loss、policy_loss、value_loss，也就是「三个张量各被 .item()
    取两次」这条不变式真正约束的那一次。按「带哪些 key」指名认领，不按行序取
    第一个；哪天真的多出第二个带 loss 的上报点，这里会报红要求重新指名，而不会
    静默改看别处。
    """
    return _uniques(
        [c for c in _calls_attr(tree, 'swanlab_logger', 'log')
         if all(k in (_call_dict(c) or {}) for k in _LOSS_KEYS)],
        '带 loss/policy_loss/value_loss 的 swanlab_logger.log(',
    )


def _prof_print_if(tree):
    """`if _do_stdout:` 里**打印内核剖析表**的那一个（体内含 _prof_ctx）。

    main() 里有两处 `if _do_stdout:`（打点行 / 剖析表），靠「第一个」定位会随
    排版漂移，所以按「体内含 _prof_ctx」来认——它就是打点区域的末端。
    """
    return _uniques([n for n in _if_nodes(tree, '_do_stdout')
                     if any('_prof_ctx' in ast.unparse(s) for s in n.body)],
                    '打印 [profile] 剖析表的 `if _do_stdout:` 分支')


# --------------------------------------------------------------------------- #
# 频率决策：swanlab_every=0 跟随 log_every
# --------------------------------------------------------------------------- #
def test_zero_swanlab_every_follows_log_every():
    """默认 0：stdout 与 swanlab 频率完全一致（与改动前行为相同）。"""
    for step in range(1, 201):
        do_out, do_swan = _should_log(step, log_every=50,
                                      swanlab_every=0, swanlab_on=True)
        assert do_out == (step % 50 == 0)
        assert do_swan == do_out, f"step={step}: swanlab 应跟随 stdout"


def test_swanlab_every_decoupled_from_stdout():
    """swanlab_every=5、log_every=50：swanlab 密 10 倍，stdout 不变。"""
    swan_steps = [s for s in range(1, 501)
                  if _should_log(s, 50, 5, True)[1]]
    out_steps = [s for s in range(1, 501)
                 if _should_log(s, 50, 5, True)[0]]
    assert len(out_steps) == 10, f"stdout 应仍为 10 次，实得 {len(out_steps)}"
    assert len(swan_steps) == 100, f"swanlab 应为 100 次，实得 {len(swan_steps)}"
    assert all(s % 50 == 0 for s in out_steps)


def test_swanlab_off_never_logs():
    """swanlab_logger 为 None 时，上报开关恒为 False（零开销）。"""
    for step in range(1, 101):
        assert _should_log(step, 50, 1, False)[1] is False


def test_swanlab_every_one_logs_every_step():
    """swanlab_every=1 时每步都该上报（stdout 仍按 log_every）。"""
    swan = [s for s in range(1, 51) if _should_log(s, 50, 1, True)[1]]
    out = [s for s in range(1, 51) if _should_log(s, 50, 1, True)[0]]
    assert len(swan) == 50
    assert out == [50]


# --------------------------------------------------------------------------- #
# 单一同步点：三个张量各取一次
# --------------------------------------------------------------------------- #
class _CountingTensor:
    """记录 .item() 调用次数的张量替身。"""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def item(self):
        self.calls += 1
        return self.value


def test_read_log_scalars_syncs_each_tensor_exactly_once():
    """每打点 loss/policy_loss/value_loss 各只 .item() 一次（3 次同步，非 6 次）。"""
    a, b, c = _CountingTensor(1.5), _CountingTensor(0.25), _CountingTensor(0.125)
    got = _read_log_scalars(a, b, c)
    assert a.calls == 1, f"loss 被取 {a.calls} 次"
    assert b.calls == 1, f"policy_loss 被取 {b.calls} 次"
    assert c.calls == 1, f"value_loss 被取 {c.calls} 次"
    assert got == (1.5, 0.25, 0.125)


def test_read_log_scalars_returns_plain_floats():
    """返回值是普通 float（供 %-格式化与 swanlab 字典直接使用）。

    真实的 torch 浮点张量 .item() 就返回 python float，这里用 float 替身对齐该语义。
    """
    a, b, c = _CountingTensor(3.0), _CountingTensor(2.0), _CountingTensor(1.0)
    vals = _read_log_scalars(a, b, c)
    assert all(isinstance(v, float) for v in vals)


# --------------------------------------------------------------------------- #
# 结构性不变量
# --------------------------------------------------------------------------- #
def test_main_uses_helpers_and_no_longer_reads_tensors_twice():
    """main() 的打点应走上述两个 helper，stdout 与 swanlab 共用同一份标量。"""
    tree = _main_tree()
    parents = _parents(tree)

    # 两个 helper 必须**真的被调用**。原文是 `'_should_log(' in src` 那种
    # 子串判定，一条提到它的注释就足以满足 —— 断言强度不等于它的字面意思。
    for helper in ('_should_log', '_read_log_scalars'):
        assert _calls_name(tree, helper), f'main() 未调用 {helper}'

    # --- 打点区域 = step 循环体里「频率决策 → 剖析表分支」这一段**同级语句** ----
    # 原文是 `src[find('_should_log('):][:find('if _prof_ctx')]`：起点能被注释
    # 钓走；而且 find 找不到时返回 -1，`[: -1]` 会**静默**砍掉最后一个字符、把
    # 区域悄悄缩到别处。这里改按同级语句切，两端都是认出来的真节点。
    should = _uniques(
        [s for s in ast.walk(tree)
         if isinstance(s, ast.Assign)
         and _calls_name(s.value, '_should_log')],
        '频率决策赋值 `_do_stdout, _do_swanlab = _should_log(...)`',
    )
    assert _assign_names(should) == ['_do_stdout', '_do_swanlab'], \
        '频率决策应同时决定 stdout 与 swanlab 开关'
    prof = _prof_print_if(tree)          # 区域内最后一个 stdout 分支

    lst, _ = _stmt_list_of(should, parents)
    assert lst is not None and prof in lst, \
        '频率决策与剖析表分支不在同一个语句列表里，无法定义「打点区域」'
    lo, hi = lst.index(should), lst.index(prof)
    assert lo < hi, '剖析表分支应排在频率决策之后（它才是打点区域的末端）'
    # 区间取 `lo:hi+1` = 连剖析分支**整条**一起算，比原文的
    # `find('if _prof_ctx')` 多盖住 2443~2453，因此只会更宽、不会更窄。
    region = lst[lo:hi + 1]

    # 区域内不应再有裸 .item()：三个张量的取值必须全部经 _read_log_scalars
    bad = sorted({n.lineno for stmt in region for n in ast.walk(stmt)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == 'item'})
    assert not bad, \
        '打点区域不应再有裸 .item()（应全部经 _read_log_scalars），行：' \
        + ', '.join(map(str, bad))


def test_swanlab_log_uses_cached_scalars():
    """swanlab 上报必须用缓存的标量变量，而不是重新取张量。"""
    tree = _main_tree()
    call = _train_log_call(tree)   # 带 loss/policy_loss/value_loss 的那次上报

    # --- ① 整棵调用子树里不得再取张量 -----------------------------------------
    # 原文是 `src[log_start:log_start + 400]`：起点是一段注释就能钓走的第一个
    # 匹配；那个 400 的常数**连调用体都盖不满**（打点这次从 2421 铺到 2439，
    # 窗口在 2430 就断了），末尾几个 key 从来没被检查过。改成「该调用节点的
    # 整个子树」后两头都不漏，且注释/docstring 因为不是节点而无法参与。
    got_item = sorted({n.lineno for n in ast.walk(call)
                       if isinstance(n, ast.Attribute) and n.attr == 'item'})
    assert not got_item, \
        'swanlab.log 内不得再取 .item()，行：' + ', '.join(map(str, got_item))

    # --- ② 上报的三项必须正是单同步点的返回值 ---------------------------------
    # 这是「用缓存的标量」的正向表述。原文只查了「400 字符内没有 .item()」那
    # 一半，于是 `float(log_loss)` 这种不含 .item()、却照样再同步一次设备的
    # 写法能混过去 —— 它破坏的正是本文件要防的「一个打点只同步 3 次」。
    d = _call_dict(call)
    assert d is not None, 'swanlab.log 的首个实参不是 dict 字面量，无法核对上报内容'
    got = tuple(d[k] for k in _LOSS_KEYS)
    assert got == ('_lv', '_pv', '_vv'), \
        f'上报的 loss 三项必须复用单同步点的 _lv/_pv/_vv，实得 {got}'

    # --- ③ 那三个名字确实来自 _read_log_scalars，且早于本次上报 ---------------
    # 光看名字不够：还得确认它就是那个单同步点的解包结果，且发生在上报之前。
    unpack = _uniques(
        [s for s in ast.walk(tree)
         if isinstance(s, ast.Assign) and _calls_name(s.value, '_read_log_scalars')],
        '把 _read_log_scalars 结果解包成 _lv/_pv/_vv 的赋值',
    )
    assert _assign_names(unpack) == ['_lv', '_pv', '_vv'], \
        f'_lv/_pv/_vv 必须来自 _read_log_scalars 的解包，实得 {ast.unparse(unpack)}'
    assert unpack.lineno < call.lineno, \
        '上报必须晚于单同步点（否则又是一次独立取数）'


def test_epoch_loss_removed():
    """死变量 epoch_loss 已删除（grep 证实原本从未被读取，且注释误导）。"""
    src_path = os.path.join(ROOT, 'scripts', 'train_sft.py')
    src = open(src_path, encoding='utf-8').read()
    assert 'epoch_loss' not in src, 'epoch_loss 死变量仍未删除'


def test_swanlab_every_arg_exists_with_default_zero():
    """--swanlab-every 必须是合法参数且默认 0（保持现状）。"""
    tree = _main_tree()
    flags = {}
    decls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'add_argument' and n.args
             and isinstance(n.args[0], ast.Constant)
             and isinstance(n.args[0].value, str)]
    for call in decls:
        flags.setdefault(call.args[0].value, []).append(call)

    assert '--swanlab-every' in flags, 'train_sft.py 缺少 --swanlab-every'
    # 原文是 `add_argument\(\s*['"]--swanlab-every['"].*?default=([^\s,]+)`（re.S）：
    # `.*?` 配 re.S 会跨行，于是只要有人**在任何位置**（比如 --swanlab-every 的
    # help 文本里就含 `--log-every`）先提到这个旗名，正则就会从那儿起扫、把
    # **后面另一个旗名**的 default 抓过来；反过来，一旦本次调用的 default 被
    # 删掉，`.*?default=` 会继续往前找到下一个旗名的 default 而照样通过。
    # 改按「首个实参就是那个旗名的 add_argument 调用」认领，default 只认它自己
    # 的 keyword。
    call = _uniques(flags['--swanlab-every'], '`--swanlab-every` 的 add_argument 调用')
    kwargs = {k.arg: k.value for k in call.keywords if k.arg}
    assert 'default' in kwargs, \
        f'--swanlab-every 未显式给 default（实得 {ast.unparse(call)}）'
    got = ast.unparse(kwargs['default'])
    assert got == '0', f'--swanlab-every 默认应为 0，实得 {got}'


def test_swanlab_every_logged_in_config():
    """启动日志应打印 swanlab 上报频率，便于确认加密是否生效。"""
    tree = _main_tree()
    # 原文是 `src.find('SwanLab: 启用=')` + 500 字符窗口：起点是一句提到该字样
    # 的注释就能钓走的第一个匹配，而那个 500 的窗口**早已越过了本条语句**、伸进
    # 后面的分布式初始化（见改前窗口内容），里面「出现过这两个名字」不再说明
    # 本条日志打印了它们。改按「首个实参是含该字样的字符串字面量的调用」认领。
    call = _uniques(
        [c for c in ast.walk(tree)
         if isinstance(c, ast.Call) and c.args
         and isinstance(c.args[0], ast.Constant)
         and isinstance(c.args[0].value, str)
         and 'SwanLab: 启用=' in c.args[0].value],
        '打印 SwanLab 配置状态的日志调用',
    )
    fmt = call.args[0].value
    assert '--swanlab-every' in fmt, 'SwanLab 状态日志未包含 --swanlab-every'
    assert '--log-every' in fmt, 'SwanLab 状态日志未说明与 --log-every 的关系'

    # 实参里必须真的把两个频率打出来（只看窗口的话，help 文本里出现的旗名就够
    # 让旧断言绿了，而那两个 %d 其实没喂任何频率）。
    used = [ast.unparse(a) for a in call.args[1:]]
    assert 'args.swanlab_every' in used, \
        f'SwanLab 状态日志未打印实际 swanlab_every，实参={used}'
    assert 'args.log_every' in used, \
        f'SwanLab 状态日志未打印实际 log_every，实参={used}'


# --------------------------------------------------------------------------- #
# swanlab_logger 落地方式（回归：曾因把 import 抽走而留下未定义名）
# --------------------------------------------------------------------------- #
def test_main_logs_through_swanlab_logger_not_bare_swanlab():
    """main() 里的记录必须走 swanlab_logger，不能直接引用 `swanlab`。

    回归：把 `import swanlab` 抽进 _init_swanlab 后，main() 中残留的
    `swanlab.log(...)` 会变成未定义名——平时被 `if swanlab_logger is not None`
    挡住看不出来，但**一旦 swanlab 真能启用，首个打点就会 NameError 崩溃**。
    """
    tree = _main_tree()
    parents = _parents(tree)

    # 不得直接引用未定义的 `swanlab`。查的是**属性访问**而非调用，
    # 因此连 `f = swanlab.log` 这种只取不调也拦得住（比原文的逐字
    # `'swanlab.log(' not in src` 略严），而注释里提到它不再误报。
    for attr in ('log', 'finish'):
        bad = _attr_uses(tree, 'swanlab', attr)
        assert not bad, \
            f'main() 不应直接引用 swanlab.{attr}（swanlab 在此作用域未定义）：' \
            + ', '.join(f'第 {n.lineno} 行' for n in bad)

    assert _calls_attr(tree, 'swanlab_logger', 'log'), \
        'main() 应通过 swanlab_logger 记录'
    _uniques(_calls_attr(tree, 'swanlab_logger', 'finish'),
             'swanlab_logger.finish() 调用')

    # 确认每个调用都在 None 守卫之内。守卫有两种等价写法：
    #   · if swanlab_logger is not None:   —— 直接判空
    #   · if _do_swanlab:                  —— 经 _should_log 判空（内部含
    #                                         `swanlab_on = swanlab_logger is not None`）
    # 后者见 test_swanlab_off_never_logs 验证其确实含判空语义。
    # 原文是「往前数 14 行找守卫字样」：一句提到 `swanlab_logger.log(` 的注释会
    # 凭空造出一条要守卫的调用（假红），而守卫写在 15 行之上又会被漏判（假绿）。
    # 改沿 AST 父链找**真正包围该调用的 If**——那就是判空本身。
    ok_guards = ('swanlab_logger is not None', '_do_swanlab')
    for attr in ('log', 'finish'):
        for call in _calls_attr(tree, 'swanlab_logger', attr):
            node, guard = call, None
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.If) and ast.unparse(node.test) in ok_guards:
                    guard = node
                    break
            assert guard is not None, \
                f'swanlab_logger.{attr} 调用（第 {call.lineno} 行）缺少 None 守卫'


def test_init_returns_none_on_failure():
    """初始化失败必须返回 None（而非抛出），保证训练不被跟踪功能拖垮。"""
    import logging
    import sys as _sys
    import types as _types
    import scripts.train_sft as t

    class _Args:
        swanlab_api_key = ''
        board_size = 19
        ver = 'v1'
        backbone_channels = 1
        backbone_res_blocks = 1
        res_blocks = 1
        convnext_blocks = 1
        attn_blocks = 1
        value_channels = 1
        value_res_blocks = 1
        policy_channels = 1
        policy_layers = 1
        batch_size = 1
        lr = 1.0
        epochs = 1

    fake = _types.ModuleType('swanlab')

    def _boom(*a, **k):
        raise RuntimeError('boom')

    fake.init = _boom
    old = _sys.modules.get('swanlab')
    _sys.modules['swanlab'] = fake
    try:
        assert t._init_swanlab(_Args(), logging.getLogger('test')) is None
    finally:
        if old is None:
            _sys.modules.pop('swanlab', None)
        else:
            _sys.modules['swanlab'] = old


def test_dns_precheck_removed():
    """DNS 预检已按决策移除（不再有 socket 依赖与 getaddrinfo 探测）。"""
    src = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()
    assert '_swanlab_reachable' not in src, 'swanlab DNS 预检仍未移除'


# --------------------------------------------------------------------------- #
# 禁止在训练进程内 pip install swanlab
# --------------------------------------------------------------------------- #
_TRAIN_SCRIPTS = ('train_sft.py', 'selfplay_train.py')


@pytest.mark.parametrize('fname', _TRAIN_SCRIPTS)
def test_no_inprocess_pip_install(fname):
    """训练脚本不得在进程内 `pip install swanlab`。

    原因：调用点位于 torch / torch_npu **已加载之后**，此时改动 site-packages
    可能破坏后续惰性导入；且 shell/*.sh 在启动 python 前已装过一次，那次失败
    的话这里必然也失败，只是白等一轮。手动装（进程启动前）没问题正是此因。
    """
    src = open(os.path.join(ROOT, 'scripts', fname), encoding='utf-8').read()
    # 只匹配真正调用 pip 的 argv 形式，不误伤「请执行 pip install swanlab」这类提示文本
    assert "'-m', 'pip'" not in src, f'{fname} 仍在训练进程内 pip install swanlab'
    assert '"-m", "pip"' not in src, f'{fname} 仍在训练进程内 pip install swanlab'
    assert 'check_call' not in src, f'{fname} 仍在训练进程内 pip install swanlab'


def _swanlab_present(spec_result):
    """把 importlib.util.find_spec('swanlab') 固定为指定返回值。"""
    import importlib.util
    return lambda name: spec_result


def test_absent_swanlab_degrades_with_actionable_message(caplog):
    """swanlab 真的没装时：返回 None，且提示如何安装（而不是自己装）。"""
    import logging
    import types as _types
    import scripts.train_sft as t

    class _Args:
        swanlab_api_key = ''
        board_size = 19
        ver = 'v1'
        backbone_channels = backbone_res_blocks = res_blocks = 1
        convnext_blocks = attn_blocks = value_channels = 1
        value_res_blocks = policy_channels = policy_layers = 1
        batch_size = 1
        lr = 1.0
        epochs = 1

    saved = {m: sys.modules.pop(m) for m in list(sys.modules) if m == 'swanlab'}
    import importlib.util
    real_find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name: None
    logger = logging.getLogger('test_absent')
    try:
        with caplog.at_level(logging.DEBUG, logger='test_absent'):
            assert t._init_swanlab(_Args(), logger) is None
    finally:
        importlib.util.find_spec = real_find_spec
        for m in saved:
            sys.modules[m] = saved[m]
        assert not [m for m in sys.modules if m == 'swanlab']

    text = caplog.text
    assert 'pip install swanlab' in text, f'未给出安装指引: {text!r}'
    assert '自动安装' not in text, f'不应再声称会自动安装: {text!r}'


def test_broken_swanlab_reports_real_cause_not_missing(tmp_path, caplog):
    """装了但坏（pydantic 1.x 缺 TypeAdapter）时，要报真实原因而非「未安装」。

    这是云端实际踩到的坑：swanlab 依赖 pydantic>=2，而 MindSpore / torch_npu
    常把 pydantic 钉在 1.x，于是 `import swanlab` 抛 ImportError。旧代码把
    任何 ImportError 都当成「没装」而误触发 pip install。
    """
    import logging
    import scripts.train_sft as t

    class _Args:
        swanlab_api_key = ''
        board_size = 19
        ver = 'v1'
        backbone_channels = backbone_res_blocks = res_blocks = 1
        convnext_blocks = attn_blocks = value_channels = 1
        value_res_blocks = policy_channels = policy_layers = 1
        batch_size = 1
        lr = 1.0
        epochs = 1

    # 造一个「存在但 import 就炸」的 swanlab
    pkg = tmp_path / 'swanlab'
    pkg.mkdir()
    (pkg / '__init__.py').write_text(
        "raise ImportError(\"cannot import name 'TypeAdapter' from 'pydantic'\")\n",
        encoding='utf-8')

    saved = sys.modules.pop('swanlab', None)
    sys.path.insert(0, str(tmp_path))
    import subprocess
    pip_calls = []
    real_check_call = subprocess.check_call
    subprocess.check_call = lambda *a, **k: pip_calls.append(a)
    logger = logging.getLogger('test_broken')
    try:
        with caplog.at_level(logging.DEBUG, logger='test_broken'):
            assert t._init_swanlab(_Args(), logger) is None
    finally:
        subprocess.check_call = real_check_call
        sys.path.remove(str(tmp_path))
        sys.modules.pop('swanlab', None)
        if saved is not None:
            sys.modules['swanlab'] = saved

    assert not pip_calls, f'不该在进程内 pip install，实际调用了: {pip_calls}'
    text = caplog.text
    assert 'TypeAdapter' in text, f'未报出真实原因，实际日志: {text!r}'
    assert '未安装' not in text, f'把「装了但坏」误报成「未安装」: {text!r}'


@pytest.mark.parametrize('fname', _TRAIN_SCRIPTS)
def test_import_error_never_triggers_autoinstall(fname):
    """结构性不变量：两个训练脚本都不得在 import 失败分支里调 pip。"""
    src = open(os.path.join(ROOT, 'scripts', fname), encoding='utf-8').read()
    assert 'except ImportError:' in src or 'ImportError' in src
    # find_spec 存在 => 能区分「没装」与「装了但坏」
    assert 'find_spec' in src, \
        f'{fname} 未用 find_spec 区分「未安装」与「安装损坏」'
    assert 'getaddrinfo' not in src, '仍残留 DNS 探测调用'
    assert '\nimport socket' not in src, 'socket import 仍未移除'
