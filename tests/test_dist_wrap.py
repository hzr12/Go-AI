"""分布式包裹层的结构与语义不变量（DDP，2026-10-01 取代 FSDP1）。

为什么这些断言长这样（而不是「跑起来没报错」就够）：

DDP 相对「分片式包裹层」只有一处结构差别，但它**恰好**是本次换轨的触发点：
包裹层只允许在**顶层**加一层 wrapper。顶层加 wrapper 时 `named_parameters()`
的键只是多一个 `module.` 前缀，EMA（在包裹前构造、持有裸模块引用）的 shadow
与包裹后的遍历结果仍在同一个键空间里；而分片式包裹层会把**内部**模块就地换成
wrapper，于是每步 `ema.update()` 都抛
`KeyError: 'backbone.stem_bn._fsdp_wrapped_module.weight'`（4 卡 910A 实测）。

这类错误全部**静默或晚爆**：
  · 包裹顺序错（compile 之前 / EMA 之后）⇒ 拿到未融合图 / EMA 跨 rank 发散，
    训练照跑、指标照出，只是不对；
  · 存档忘了走汇聚 helper ⇒ 存出 1/world_size 的碎片，`evaluate.py` /
    `inference` / `webui` / `_load_model_state` 全部 `load_state_dict` 失败；
  · 构造参数被「顺手优化」⇒ 本地 world_size=1 根本不进这条分支，只有云端 4 卡
    才炸（2026-09 的 `MixedPrecision(cast_forward_precision=...)` 就是这么活到
    云端的）。

本文件不启动多进程（单机单卡没有 world_size>1，测不到真分片），所以断言以
**AST / 源码**为主（这些是「必须存在且必须长成某样」的静态不变量，比跑一次前向
更能防住未来的误改），另加**少量行为测试**：gloo + world_size=1 真跑一次包裹
（DDP 支持 CPU），以及 `save_model` 对裸模块 / 包裹模块两种形态的实际落盘结果。

被本文件取代的旧文件：``tests/test_fsdp1_conversion.py``（FSDP1 专属，已删）。
从它迁过来的、与分片特性**无关**的不变量见每条测试的 docstring。
"""
import ast
import io
import pathlib
import subprocess
import sys
import tokenize

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC_PATH = ROOT / 'scripts' / 'train_sft.py'
SRC = SRC_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SRC)

#: 已退役的包裹层在本文件里只允许以「历史记述」的形式出现（见第 7 条）。
RETIRED_MARKERS = ('历史', '已退役')


# --------------------------------------------------------------------------- #
# AST 工具
# --------------------------------------------------------------------------- #

_PARENTS = {}
for _parent in ast.walk(TREE):
    for _child in ast.iter_child_nodes(_parent):
        _PARENTS[_child] = _parent


def _func(name):
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError('train_sft.py 里没有找到函数 {}'.format(name))


def _main_fn():
    return _func('main')


def _calls_named(node, fname):
    """`node` 体内对 `fname(...)` 的调用点（按行号排序）。

    走 AST 而不是文本搜索：文档串与注释里正是在**讲解**同名调用
    （`test_init_weight_sync.py` 记过一次这个坑：文本搜索会把 docstring
    里提到的 `EMA(` 当成真调用点），文本搜索还会被行号漂移误伤。
    """
    out = [n for n in ast.walk(node)
           if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Name)
           and n.func.id == fname]
    return sorted(out, key=lambda n: n.lineno)


def _ddp_construct_calls():
    """全文件 `DistributedDataParallel(...)` 的构造点。

    全文件而不是只在 main() 里找：包裹点必须唯一，多一处构造就意味着有一处
    组件是在错误的位置被包起来的（优化器先建好、EMA 先建好……顺序都是语义）。
    """
    return [n for n in ast.walk(TREE)
            if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == 'DistributedDataParallel')
                 or (isinstance(n.func, ast.Attribute)
                     and n.func.attr == 'DistributedDataParallel'))]


def _body_src(node):
    """只取函数体（剥掉 docstring），避免注释/文档串污染文本断言。"""
    lines = SRC.splitlines()
    body = node.body
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return ''
    return '\n'.join(lines[body[0].lineno - 1:body[-1].end_lineno])


def _comment_lines():
    """全文注释所在的行号集合。"""
    return {tok.start[0] for tok in
            tokenize.generate_tokens(io.StringIO(SRC).readline)
            if tok.type == tokenize.COMMENT}


def _contiguous_comment_block_above(lineno):
    """紧贴 `lineno` 之上、连续的那一段注释（包裹点的说明就在这里）。"""
    comments = _comment_lines()
    out = []
    n = lineno - 1
    while n in comments:
        out.append(n)
        n -= 1
    return sorted(out)


#: 本文件自身的 AST —— 用来**现算**行号（假模型 `_Block` / `_tiny_model` 就定义在
#: 本文件里）。为什么需要它：第 10 条那条身份断言原来把行号写死成 `0`，失败信息
#: 变成「L0 对应参数对象被替换了」，等于让读者去第 0 行找 —— 那是噪声，不是线索。
#: 行为断言的判据是**对象身份**，报错的用途是「告诉人去看哪儿」，两者都得真。
_SELF_TREE = ast.parse(pathlib.Path(__file__).read_text(encoding='utf-8'))


def _self_assign_linenos():
    """`{'fc': L…, 'bn': L…}`：`self.x = <构造调用>` 的行号（只收赋值给属性的）。

    只认 `self.<attr> = Call(...)` 这种「造出子模块」的写法；赋值给局部变量或
    非 Call 右值的行不进这张表（它们不是参数来源）。
    """
    out = {}
    for node in ast.walk(_SELF_TREE):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == 'self'):
                out[tgt.attr] = node.lineno
    return out


_SELF_ASSIGN_LINENO = _self_assign_linenos()


def _param_def_lineno(param_name):
    """`'0.bn.weight'` → 造出 `bn` 的那一行；查不到退到 `_tiny_model` 的构造行。

    命名空间的第一段是序号（`Sequential` 给的），所以取**序号之后**那一段去查表。
    `Sequential(_Block(8), torch.nn.Linear(8, 4))` 里最后那个 Linear 是内联构造的
    位置参数、查不到属性名 ⇒ 退到 `_tiny_model` 里的 `return`（它就是这些参数
    唯一的出处）。查不到时返回 0，但**不会抛**：工具函数自己炸掉会把真正的失败
    原因盖住 —— 而「行号取不到」这件事本身不该让身份断言变成别的错。
    """
    parts = param_name.split('.')
    for i, part in enumerate(parts):
        if part.isdigit() and i + 1 < len(parts):
            return _SELF_ASSIGN_LINENO.get(parts[i + 1]) or _tiny_model_build_lineno()
    return _tiny_model_build_lineno()


def _tiny_model_build_lineno():
    """`_tiny_model()` 里 `return torch.nn.Sequential(...)` 那行（内联参数的兜底定位点）。"""
    for node in ast.walk(_SELF_TREE):
        if not (isinstance(node, ast.FunctionDef) and node.name == '_tiny_model'):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                return stmt.lineno
        if node.body:
            return node.body[0].lineno
    return 0


# --------------------------------------------------------------------------- #
# 1. 构造参数：只有 device_ids
# --------------------------------------------------------------------------- #

def test_ddp_wrapper_passes_only_device_ids():
    """DDP 构造**只**传 `device_ids`，其余全默认 —— 防止「顺手优化」构造参数。

    每个被否决的开关都有一个具体理由（写在包裹点上方那段注释里），收益都抵不上
    风险：
      · `gradient_as_bucket_view` 与 `optimizer.zero_grad(set_to_none=True)`
        （本文件唯一的用法）混用时 bucket view 别名每轮被销毁 ⇒ 省不掉拷贝，
        收益仅 ~36.27 MB/rank（0.106% 的卡），却引入
        `Expected to mark a variable ready only once` 的风险；
      · `find_unused_parameters=True` 只是给「某头不参与 loss」兜底，而那种情况
        现在就会**响亮地**抛 `Expected to have finished reduction in the prior
        iteration` —— 不响亮的失败更危险；
      · `broadcast_buffers=False` 会让 BN 的 running_mean/var 跨卡漂移；
      · `static_graph=True` 会禁止 iteration 边界内的参数集合变化，收益未验证。

    断言顺带钉住「kwarg 必须散装写」：`**d` 展开的键在 AST 里不可见，若哪天把
    参数集挪进 dict 再展开，本文件所有构造参数断言会**集体失明**而不是报错。
    """
    calls = _ddp_construct_calls()
    assert len(calls) == 1, (
        'DistributedDataParallel 的构造点必须**唯一**（多处构造 = 有一处组件在'
        '错误的位置被包裹）：找到 %d 处' % len(calls))
    call = calls[0]
    assert len(call.args) == 1, '只应传模块本身一个位置参数，实得 %d 个' % len(call.args)
    unpacked = [k for k in call.keywords if k.arg is None]
    assert not unpacked, (
        '构造必须散装写 kwarg：`%s` 展开会让本文件的构造参数断言集体失明'
        % '**')
    names = [k.arg for k in call.keywords]
    assert names == ['device_ids'], (
        'DDP 构造只允许传 device_ids（其余全默认），实得 %s' % names)
    assert ast.unparse(call.keywords[0].value) == '[local_rank]', \
        'device_ids 应为 [local_rank]（local_rank 是本文件的权威卡号），实得 %s' \
        % ast.unparse(call.keywords[0].value)


def test_no_new_ddp_cli_flag_was_introduced():
    """DDP 这套构造参数**不新增任何 CLI flag**（也不许有被退役的 FSDP flag）。

    加 flag 等于把「当前不用」包装成「用户可调」，而 spec §5.1 给这四个参数逐条
    写了否决理由 —— 没有一条是需要用户调的。
    """
    import scripts.train_sft as t   # noqa: F401 — 先确认模块能 import（语法门之外）

    parser_actions = set()
    for node in ast.walk(TREE):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and a.value.startswith('--'):
                    parser_actions.add(a.value)
    banned = ('fsdp', 'ddp', 'shard', 'bucket', 'unused-param', 'static-graph')
    bad = sorted(f for f in parser_actions if any(k in f for k in banned))
    assert not bad, (
        '出现了与包裹层相关的 CLI flag %s —— DDP 构造参数全默认，不新增 flag' % bad)
    assert parser_actions, '没解析出任何 add_argument（解析方式失效？）'


# --------------------------------------------------------------------------- #
# 2-4. 包裹顺序（spec §5.3 的硬要求）
# --------------------------------------------------------------------------- #

def _wrap_lineno():
    return _ddp_construct_calls()[0].lineno


def test_wrapping_happens_after_ema_construction():
    """包裹点**必须在 `EMA(` 之后**。

    为什么这是正确性而不是风格：`DistributedDataParallel.__init__` 在
    **它自己构造时**（`_ddp_init_helper` → `_sync_module_states`）把 rank0 的
    params/buffers 广播出去。EMA 构造时就把参数 `clone()` 进 shadow，所以：

      · 包裹在 EMA **之后**（现状）⇒ EMA 先拿到 Task 1 广播过的正确权重；
      · 若把包裹提到 EMA 之前，EMA 会把各 rank **自己那份被丢弃的随机权重**
        clone 进 shadow ⇒ rank0 对、rank1~3 陈旧 ⇒ `ema.update()` 每步把正确
        权重混进陈旧 shadow ⇒ **EMA 跨 rank 发散且不报错**。

    这条与 Task 1 的 `_sync_init_weights_from_rank0`（必须在 EMA 之前）是一对：
    广播由 Task 1 在 EMA 之前做完，包裹层自带的那次同步只能当第二道保险。
    """
    main = _main_fn()
    wrap = _wrap_lineno()
    ema = _calls_named(main, 'EMA')
    assert ema, 'main() 里没找到 EMA( 调用点'
    assert wrap > min(c.lineno for c in ema), (
        '包裹（L%d）必须在 EMA（L%d）**之后**：EMA 构造时就把参数 clone 进 shadow，'
        '放在包裹之前会让 rank1~3 的 shadow 抓住各自被丢弃的随机权重'
        % (wrap, min(c.lineno for c in ema)))


def test_wrapping_happens_after_init_weight_sync():
    """包裹点**必须在 `_sync_init_weights_from_rank0` 之后**（spec §5.3）。

    这一条迁自被删的 `tests/test_fsdp1_conversion.py`（当时叫
    `test_param_groups_built_before_wrapping` 的姊妹条），它与分片特性无关：
    包裹层的 `_sync_module_states` 只在构造那一刻广播一次，之后参数存储归它管；
    本文件自己的广播必须在那之前、由裸模块上的 `model.parameters()` 完成。

    同时钉住 `_assert_init_weights_identical` 也在包裹之前：它是「启动时验证
    状态已一致」的自检，包裹之后就变成验证 DDP 自己刚做的事，语义变了。
    """
    main = _main_fn()
    wrap = _wrap_lineno()
    for fname in ('_sync_init_weights_from_rank0', '_assert_init_weights_identical'):
        calls = _calls_named(main, fname)
        assert len(calls) == 1, \
            '%s 在 main() 里应恰好调用一次（唯一才能证明时机）：%s' % (fname, len(calls))
        assert calls[0].lineno < wrap, (
            '%s（L%d）必须在包裹（L%d）之前' % (fname, calls[0].lineno, wrap))


def test_param_groups_built_before_wrapping():
    """`_build_param_groups` 必须在包裹之前：包裹后名字带 `module.` 前缀。

    迁自被删文件（同名不变）。`startswith('value.')` 判据依赖**裸模块**的
    命名空间；包裹后 `named_parameters()` 产出 'module.value.fc.bias'，
    value 组与 other 组的划分会静默错掉 —— 不抛异常，只是 LR 分组变了。
    """
    main = _main_fn()
    groups = _calls_named(main, '_build_param_groups')
    assert len(groups) == 1, '找不到唯一的 _build_param_groups 调用点：%d' % len(groups)
    assert groups[0].lineno < _wrap_lineno(), \
        '_build_param_groups 必须在包裹之前（分组依赖裸模块的 \'value.\' 前缀）'


def test_wrapping_happens_after_compile():
    """包裹点**必须在 `torch.compile` 之后**（保持现状）。

    迁自被删文件的 `test_fsdp_wrapping_happens_after_compile`。反过来：
    整模型 compile 会被 wrapper 的动态边界吞掉，拿到的是未融合图；P4.7 的
    Linear-only 编译路径（`--npu-graph-compile 1`）尤其敏感。
    """
    main = _main_fn()
    compiles = [n for n in ast.walk(main)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'compile'
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == 'torch']
    assert len(compiles) == 1, (
        'main() 里 torch.compile 的调用点应唯一（当前 %d 处），否则无法判定包裹次序'
        % len(compiles))
    assert _wrap_lineno() > compiles[0].lineno, \
        '包裹必须在 torch.compile（L%d）之后' % compiles[0].lineno


def test_wrapping_happens_before_training_loop():
    """包裹点必须在训练循环（`for epoch in ...`）之前。

    迁自被删文件里「包裹点在训练循环之前」的隐含前提（当时由
    `test_fsdp_wrapping_happens_after_compile` 的位置断言顺带覆盖）。
    漏包 = 4 个 rank 各跑各的、全程无梯度同步，且**不报任何错**。
    """
    main = _main_fn()
    epoch_loops = [n for n in ast.walk(main)
                   if isinstance(n, ast.For)
                   and isinstance(n.target, ast.Name)
                   and n.target.id == 'epoch']
    assert len(epoch_loops) == 1, \
        'main() 里应恰好一个 for epoch 循环（当前 %d 个）' % len(epoch_loops)
    assert _wrap_lineno() < epoch_loops[0].lineno, \
        '包裹必须在训练循环（L%d）之前' % epoch_loops[0].lineno


def test_wrap_site_documents_why_sync_must_precede_ema():
    """包裹点上方那段注释必须写明「Task 1 的同步必须在 EMA 之前」的理由。

    这是最容易被「顺手整理注释」抹掉、又最难靠运行时报错发现的一处知识：
    顺序一旦反过来，EMA 跨 rank 发散且**不报错**。因此钉住注释内容本身。

    注释块是紧贴 `if is_dist:` 之上的那一段（不是紧贴构造调用 —— 中间隔着
    `if is_dist:` 那一行）。

    ⚠ 本条的**意图只是取注释块**（上面那个 `assert stmt is not None` 是取块的
    副产品，不是「包裹点必须在 `if is_dist` 内」这条不变量的守护）。变异测试确实
    观察到：把 `world_size == 1` 那条路径改坏（把包裹挪出 `if is_dist:`）时，**是
    这一句**把它抓住的 —— 纯属偶然，且失败信息指向错的地方（读者会以为要查注释）。
    spec §7 的 I3 由 `test_ddp_wrapping_stays_inside_the_is_dist_branch` 直接钉，
    那里的判据与失败信息都是对着 I3 写的。
    """
    call = _ddp_construct_calls()[0]
    stmt = _PARENTS.get(call)
    while stmt is not None and not isinstance(stmt, ast.If):
        stmt = _PARENTS.get(stmt)
    assert stmt is not None, '包裹点不在任何 if 里（结构被改了？）'
    block = _contiguous_comment_block_above(stmt.lineno)
    assert block, '包裹点上方没有连续注释块：那段理由被删掉了？'
    text = '\n'.join(SRC.splitlines()[n - 1] for n in block)
    for token in ('_sync_init_weights_from_rank0', 'EMA', 'shadow', '_sync_module_states'):
        assert token in text, (
            '包裹点上方的注释必须记下 %r（EMA 与包裹顺序的因果）；当前注释块：\n%s'
            % (token, text))


def test_ddp_wrapping_stays_inside_the_is_dist_branch():
    """spec §7 I3：`world_size == 1` 路径不变 —— 包裹点必须仍在 `if is_dist:` 内。

    `is_dist = world_size > 1`（由 `test_world_size_env_contract_is_unchanged` 钉），
    所以「包裹在 `if is_dist` 内」就是「world_size==1 时这段代码根本不执行」的
    **静态等价物**：判据不看运行时，只看那行 `DistributedDataParallel(...)` 的
    `ast.Call` 祖先里有没有 `ast.If`、以及最内层判据是不是 `is_dist`。

    为什么这条要**单独**钉：2026-10-01 的变异测试确实把「world_size==1 路径被改」
    弄红过，但抓住它的是 `test_wrap_site_documents_why_sync_must_precede_ema` 里
    顺带的一句 `assert stmt is not None`（那条要往上找 `ast.If` 才能取到注释块）——
    **偶然**命中，且失败信息是「包裹点不在任何 if 里（结构被改了？）」，会把读者
    引到错的地方。spec §7 的 I3 当时**没有任何测试直接对应**。

    破坏它会长什么样：包裹被挪到 `if is_dist:` 之外（或判据被换成 `world_size >= 1`
    之类）⇒ 单卡路径也会去构造 DDP，而那条路径上没有 `device_ids` 所依赖的 device
    上下文，且 `shell/train_sft_npu_1card.sh` / `_2card.sh` 本来就不该被分布式代码
    碰到。更隐蔽的一种是**判据被换成别的变量**（例如 `if world_size > 1:` 就地写死）
    —— 此时 `if` 祖先还在、上面那条顺带断言照样绿，所以下面必须比对判据的**内容**。
    """
    call = _ddp_construct_calls()[0]
    stmt = _PARENTS.get(call)
    guards = []
    while stmt is not None:
        if isinstance(stmt, ast.If):
            guards.append(ast.unparse(stmt.test))
        stmt = _PARENTS.get(stmt)
    assert guards, (
        'DDP 包裹点（L%d）不在任何 `if` 里 ⇒ world_size==1 的路径也会构造 DDP，'
        'spec §7 I3 被破坏' % call.lineno)
    assert 'is_dist' in guards[0], (
        'DDP 包裹点（L%d）的最内层判据是 %r 而不是 `is_dist` ⇒ 它不再由 '
        'world_size>1 兜住，I3 被破坏'
        % (call.lineno, guards[0]))


# --------------------------------------------------------------------------- #
# 5-6. 存档：state_dict 直接用，不再有汇聚 helper
# --------------------------------------------------------------------------- #

def test_save_model_uses_plain_state_dict():
    """存档走 `model.state_dict()`：DDP 下每 rank 各持完整模型，无需汇聚。

    与被删文件的 `test_save_model_uses_full_state_dict` 相反：那边守的是
    「不许裸调 state_dict」（分片下会存出碎片），这边守的是「不许绕 helper」。
    DDP 的 `state_dict()` **不是 collective**（只有梯度 all-reduce），所以它在
    `is_main` 分支里调用既安全也无额外开销。
    """
    body = _body_src(_func('save_model'))
    assert 'model.state_dict()' in body, 'save_model 必须直接用 model.state_dict()'
    assert '_fsdp_full_state_dict' not in SRC, '汇聚 helper 已被删除（退役符号）'


def test_no_full_state_dict_vocabulary_remains():
    """全文件不得再有 `full_state_dict` / `StateDictConfig` 之类的分片词汇。

    spec §6.4 第 5 条。这类东西一旦复活，多半是有人以为「分片下要汇聚」而
    重新引入的 —— 而 DDP 下它们要么不存在，要么是无意义的额外 all-gather。
    """
    for banned in ('FullStateDictConfig', 'FullOptimStateDictConfig',
                   'StateDictType', 'OptimStateDictType', 'full_state_dict',
                   'full_optimizer_state', 'state_dict_type'):
        assert banned not in SRC, \
            '退役符号 %r 复活了（DDP 不需要任何分片汇聚）' % banned


def test_all_optimizer_state_saves_use_plain_state_dict_under_is_main():
    """3 处 train_state 存档点全部直接用 `optimizer.state_dict()`，且都在 `is_main` 下。

    两半都要：
      · 直接用 —— DDP 下各 rank 的 optimizer state 完整且一致，绕 helper 只会
        复活退役路径（迁自被删文件的
        `test_all_optimizer_state_dict_saves_go_through_fsdp`，方向相反）；
      · `is_main` 下 —— 非 rank0 写同一个文件会互相覆盖，且 3 个 rank 同时
        `torch.save` 到同一路径是真实发生过的（OOM 兜底、定期保存、最佳模型）。
    """
    sites = []
    for node in ast.walk(TREE):
        if not (isinstance(node, ast.Dict) and node.keys):
            continue
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and k.value == 'optimizer'
                    and isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                    and v.func.attr == 'state_dict'):
                sites.append((node.lineno, v))
    assert len(sites) == 3, (
        '期望 3 处 train_state 存档点（OOM 兜底 / 定期保存 / 最佳模型）都用 '
        "optimizer.state_dict()，实得 %d 处：%s"
        % (len(sites), [s[0] for s in sites]))
    for lineno, call in sites:
        assert ast.unparse(call.func.value) == 'optimizer', \
            'L%d 应直接调 optimizer.state_dict()' % lineno
        node = _PARENTS.get(_PARENTS.get(call))
        guarded = False
        while node is not None:
            if isinstance(node, ast.If) and 'is_main' in ast.unparse(node.test):
                guarded = True
                break
            node = _PARENTS.get(node)
        assert guarded, (
            'L%d 的 optimizer 存档不在 is_main 分支里：多 rank 会同时写同一文件'
            % lineno)


# --------------------------------------------------------------------------- #
# 7. FSDP 零残留：**代码级**全仓库；**标记级**只扫 scripts/train_sft.py
# --------------------------------------------------------------------------- #
# ⚠ 范围（spec §7 I5 的对应写法，2026-10-01 评审收窄）：I5 说的是
#   ① `scripts/train_sft.py` 内零 FSDP **代码**引用；② 该文件里剩下的 FSDP 字样
#     只许出现在带「历史/已退役」标记的注释或文档串里。
# 「零残留」只在**代码级**是全仓库事实：①的前两条对 `train_sft.py` 判 AST，最后
# 一条 `test_no_fsdp_code_reference_anywhere_in_repo` 对整个仓库判 import/名字/
# 属性/def 名。**注释级不是、也不追求**全仓库零：`src/networks/backbone.py` 记着
# 「auto_wrap 按块类切 ⇒ 逐块检查点与 FSDP unit 1:1」这段 2026-09 的局部设计论证，
# 那是**模型代码**里的取舍记录而不是包裹层配置，为它改注释属于范围蔓延。所以下面
# 第三条只扫 `train_sft.py` —— 别把它当成「全仓库都干净了」的证据。

def test_no_fsdp_import_remains():
    """`from torch.distributed.fsdp import ...` 一处都不许剩（含子模块 import）。

    按 AST 判 import 而不是文本搜索：文本搜索会被「历史：随 FSDP1 包裹层一并
    删除」这类注释里的字样误伤（那是被明确允许的，见下一条）。
    """
    bad = []
    for node in ast.walk(TREE):
        if isinstance(node, ast.ImportFrom) and node.module \
                and 'fsdp' in node.module.lower():
            bad.append(('from ' + node.module, node.lineno))
        if isinstance(node, ast.Import):
            for a in node.names:
                if 'fsdp' in a.name.lower():
                    bad.append(('import ' + a.name, node.lineno))
    assert not bad, '仍从分片式包裹层 import：%s' % bad


def test_no_fsdp_symbol_is_referenced():
    """代码里不得引用任何 FSDP 符号（变量名 / 属性名也算）。

    名字里带 fsdp 的局部函数、变量或属性即使当前没被调用，也说明有人正准备把
    那条路径接回来 —— 而它在本仓库里是**已退役**状态，接回来就会重演 2026-10-01
    的 EMA KeyError。
    """
    bad = []
    for node in ast.walk(TREE):
        if isinstance(node, ast.Name) and 'fsdp' in node.id.lower():
            bad.append(('name:' + node.id, node.lineno))
        elif isinstance(node, ast.Attribute) and 'fsdp' in node.attr.lower():
            bad.append(('attr:' + node.attr, node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and 'fsdp' in node.name.lower():
            bad.append(('def:' + node.name, node.lineno))
    assert not bad, '仍引用分片式包裹层的符号：%s' % bad


def test_fsdp_mentions_are_only_marked_as_retired_history():
    """残留的 FSDP 字样只允许出现在**显式标了「历史 / 已退役」的注释或文档串**里。

    ⚠ **扫描范围 = `scripts/train_sft.py`**（`SRC`），**不是全仓库**。这是
    spec §7 I5 收窄后的口径：I5 承诺的是「该文件内零 FSDP 代码引用 + 残留字样
    带退役标记」，而不是「全仓库注释都干净」。已知在范围外的一处：
    `src/networks/backbone.py`（2026-09 的逐块检查点论证里提到 auto_wrap 的切分
    粒度）—— 那是模型代码里的取舍记录，让它为了「FSDP 已退役」而改注释是范围蔓延。
    「代码级零残留」的全仓库版本由下面 `test_no_fsdp_code_reference_anywhere_in_repo`
    单独守。**读这条测试时别把它的绿当成「全仓库没提过 FSDP」。**

    为什么不干脆一个字都不留：包裹点上方的换轨理由（触发事件是
    `KeyError: 'backbone.stem_bn._fsdp_wrapped_module.weight'`）必须留在代码里，
    否则半年后有人会「再优化回」分片。保留历史的代价是它可能被误读成当前方案，
    所以这里要求每处提到它的注释都自带退役标记。

    判粒度是**连续注释块 / 单个 docstring**，不是单行 —— 一段解释天然跨多行，
    要求每行都重复写「历史」只会逼出废话（tokenize 给的 COMMENT token 是逐行的，
    所以要先按行号把连续的 token 并成块）。
    """
    # (lo, hi, text)：注释按行号并块，docstring 各算一块
    spans = []
    for node in ast.walk(TREE):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, 'body', [])
        if body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            spans.append((body[0].lineno, body[0].end_lineno,
                          body[0].value.value))
    comment_blocks = []
    for tok in tokenize.generate_tokens(io.StringIO(SRC).readline):
        if tok.type != tokenize.COMMENT:
            continue
        if comment_blocks and tok.start[0] <= comment_blocks[-1][1] + 1:
            comment_blocks[-1][1] = tok.end[0]
        else:
            comment_blocks.append([tok.start[0], tok.end[0]])
    lines = SRC.splitlines()
    for lo, hi in comment_blocks:
        spans.append((lo, hi, '\n'.join(lines[lo - 1:hi])))

    offenders = []
    for i, line in enumerate(lines, 1):
        if 'fsdp' not in line.lower():
            continue
        covering = [s for s in spans if s[0] <= i <= s[1]]
        if not covering:
            offenders.append('L%d 不在注释/文档串里（是代码）：%s' % (i, line.strip()))
            continue
        for _lo, _hi, text in covering:
            if not any(m in text for m in RETIRED_MARKERS):
                offenders.append(
                    'L%d 所在的注释块/文档串没有「%s」标记：%s'
                    % (i, '/'.join(RETIRED_MARKERS), line.strip()))
    assert not offenders, '以下 FSDP 字样没有被标成历史/已退役：\n  %s' \
        % '\n  '.join(offenders)


#: 扫「全仓库代码级零引用」时必须排除的文件：这两个文件的**守护函数名本身**
#: 带 fsdp（`test_no_fsdp_import_remains` / `test_retired_fsdp_history_is_...`）。
#: 排除它们不是给它们开后门，而是「守护不许因为自己的名字而红」——它们要判的
#: 内容（import / 标记 / 现行段）都在各自断言里。
_FSDP_GUARD_FILES = frozenset({
    pathlib.Path(__file__).resolve(),
    (ROOT / 'tests' / 'test_v21_shell_script.py').resolve(),
})

#: 扫全仓库时跳过的目录（非代码 / 非本仓库产物）。写死而不是靠 gitignore：
#: 守护不该因为「谁在跑它」而改变范围。
_FSDP_SCAN_SKIP_DIRS = ('.git', '.codegraph', '__pycache__', 'tmp',
                        '.superpowers', 'docs')


def test_no_fsdp_code_reference_anywhere_in_repo():
    """**全仓库**代码级零 FSDP 引用（spec §7 I5 的代码级那半，跨文件版本）。

    为什么需要跨文件这一条：`test_no_fsdp_import_remains` /
    `test_no_fsdp_symbol_is_referenced` 走的是**全文件** AST，而 `SRC` 只指
    `scripts/train_sft.py`。若哪天有人在 `src/` 或别的脚本里 import 了
    `torch.distributed.fsdp` 并用它包某个模块（历史上正是 `train_sft.py`
    `_wrap_fsdp1` 干的），那两条守护**一声不响**。跨文件这条把它们兜住：
    任何 `.py` 里出现 fsdp 的 import / 变量名 / 属性名 / 函数与类名即红。

    判的是**代码级**（import / Name / Attribute / def 名），不判注释 ——
    注释级的口径窄到 `train_sft.py` 一个文件（见上一条的 docstring），别混。
    """
    offenders = []
    for path in sorted(ROOT.rglob('*.py')):
        rel = path.relative_to(ROOT)
        if any(part in _FSDP_SCAN_SKIP_DIRS for part in rel.parts):
            continue
        if path.resolve() in _FSDP_GUARD_FILES:
            continue
        text = path.read_text(encoding='utf-8')
        if 'fsdp' not in text.lower():
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            what = None
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names]
                if getattr(node, 'module', None):
                    mods.append(node.module)
                hit = [m for m in mods if m and 'fsdp' in m.lower()]
                what = ('import ' + hit[0]) if hit else None
            elif isinstance(node, ast.Name) and 'fsdp' in node.id.lower():
                what = 'name:' + node.id
            elif isinstance(node, ast.Attribute) and 'fsdp' in node.attr.lower():
                what = 'attr:' + node.attr
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)) and 'fsdp' in node.name.lower():
                what = 'def:' + node.name
            if what:
                offenders.append('%s L%d %s' % (rel.as_posix(), node.lineno, what))
    assert not offenders, (
        '全仓库代码级零 FSDP 引用（I5）被破坏：\n  %s\n'
        '（注释级不在本条范围：train_sft.py 之外的注释性指涉按上一条的口径放行）'
        % '\n  '.join(offenders))


# --------------------------------------------------------------------------- #
# 8. gradient_as_bucket_view 与 set_to_none 的互斥关系
# --------------------------------------------------------------------------- #
# 本文件**当前**没有钉一条「gradient_as_bucket_view 必须为 False」的断言：那是
# 恒假断言（对着一个默认值写断言，等于断言「没人改默认值」），改默认值时它会
# 变成噪音而不是信号。真正值得钉的是**那对互斥关系**，它今天不成立、但一旦有人
# 打开那个开关就会立刻要求另一件事 —— 下面的测试是「条件式」的：只有当构造点
# 真的出现 `gradient_as_bucket_view` 时，它才开始要求 `set_to_none=False`。
#
# 为什么这两个不能混用：`optimizer.zero_grad(set_to_none=True)` 会把每个
# `p.grad` 置为 None，bucket view 只是**别名**那片 bucket 存储，每轮都被销毁
# ⇒ 省不掉拷贝；更糟的是同一进程里若某些轮的 grad 是 view、某些轮不是，
# reducer 会抛 `Expected to mark a variable ready only once`。收益仅
# **~36.27 MB/rank（0.106% 的卡）** —— 数字来自 torch 对该开关的定义「the saved
# memory size will be equal to the **total gradients size**」，本模型即全量梯度
# 9,067,443 × 4 B。⚠ 别把它和「分片下的梯度 + Adam 两矩 = 3 × 9.07 MB」那个量
# 搞混：那是另一处账（显存对比里的 +109 MB/rank），不是本开关省下的量。
# 配上面这类风险不值 ⇒ 保持默认 False。

def test_gradient_as_bucket_view_implies_set_to_none_false():
    """条件式互斥钉：开了 `gradient_as_bucket_view` 就必须同时用 `set_to_none=False`。

    现状（不传那个 kwarg）下这条断言的主体不会被执行 —— 见上面那段说明为什么
    不写成恒假的「必须为 False」。但它不是死代码：有人真去打开那个开关时，这里
    会立刻指出还缺 `set_to_none=False`，而不是等到云端 4 卡报
    `Expected to mark a variable ready only once`。
    """
    call = _ddp_construct_calls()[0]
    names = {k.arg for k in call.keywords}
    if 'gradient_as_bucket_view' not in names:
        pytest.skip('当前未开启 gradient_as_bucket_view（默认 False）；'
                    '一旦开启，本测试会要求 zero_grad(set_to_none=False) 配套出现')
    value = next(k.value for k in call.keywords
                 if k.arg == 'gradient_as_bucket_view')
    assert isinstance(value, ast.Constant) and value.value is False, \
        'gradient_as_bucket_view=True 与 set_to_none=True 互斥，不许打开'
    zero_grads = [n for n in ast.walk(TREE)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == 'zero_grad']
    assert zero_grads, '找不到 zero_grad 调用点'
    for n in zero_grads:
        kw = {k.arg: ast.unparse(k.value) for k in n.keywords}
        assert kw.get('set_to_none') == 'False', (
            'L%d 的 zero_grad 必须改成 set_to_none=False，否则与 '
            'gradient_as_bucket_view 互斥' % n.lineno)


def test_zero_grad_uses_set_to_none_true():
    """现状钉住：本文件唯一的 zero_grad 用 `set_to_none=True`。

    这条本身也是第 8 条的前提（上面那条依赖「现在确实是 set_to_none=True」才谈
    得上互斥）。它同时是性能相关的事实：set_to_none 少一次全量写零。
    """
    zero_grads = [n for n in ast.walk(TREE)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == 'zero_grad']
    assert zero_grads, '找不到 zero_grad 调用点'
    for n in zero_grads:
        kw = {k.arg: ast.unparse(k.value) for k in n.keywords}
        assert kw.get('set_to_none') == 'True', (
            'L%d 的 zero_grad 应为 set_to_none=True（现状，且是 '
            'gradient_as_bucket_view 互斥判定的依据），实得 %s'
            % (n.lineno, kw))


# --------------------------------------------------------------------------- #
# 9. 迁自被删文件、与分片特性无关的不变量
# --------------------------------------------------------------------------- #

def test_stop_flag_broadcast_still_unconditional():
    """早停 broadcast 的「无短路」不变量在 DDP 下同样成立（迁自被删文件）。

    `_sync_stop_flag` 的无条件 broadcast 是防挂死的关键：一旦按
    `stop_flag.item()` 短路，没停的 rank 跳过 collective、停了的 rank 阻塞。
    包裹层换了不换这条 —— DDP 的梯度 all-reduce 让 collective 次序错配的后果
    同样直接是挂死。

    用 AST 判 `If.test` 里是否出现对 stop_flag 的引用，而不是文本搜索 ——
    函数 docstring 里正是在**讲解**这个反例，文本搜索会自我误伤。
    """
    node = _func('_sync_stop_flag')
    assert 'dist.broadcast' in _body_src(node), '_sync_stop_flag 里的 broadcast 不见了'
    for sub in ast.walk(node):
        if isinstance(sub, ast.If):
            test = ast.unparse(sub.test)
            assert 'stop_flag' not in test, \
                '_sync_stop_flag 不得对 stop_flag 做短路求值（会破坏 collective 次序）：%s' \
                % test


def test_world_size_env_contract_is_unchanged():
    """`is_dist` 仍由 `WORLD_SIZE` 环境变量决定（迁自被删文件）。

    torchrun 注入的就是 `WORLD_SIZE`；`is_dist` 必须由**它**判定而不是问通信域，
    因为设备选择 / `init_process_group` / 包裹这些决策全在通信域建立**之前**就要
    拿到答案。换个包裹层不构成改这个契约的理由。
    """
    assert "world_size = int(os.environ.get('WORLD_SIZE', '1'))" in SRC, \
        'WORLD_SIZE 环境变量契约被改动：torchrun 注入的就是它'
    assert 'is_dist = world_size > 1' in SRC, 'is_dist 判定应保持 world_size > 1'


def test_build_param_groups_docstring_records_the_non_flattening_invariant():
    """`_build_param_groups` 的 docstring 必须记下「DDP 不扁平化参数」这条约束。

    迁自被删文件的 `test_build_param_groups_docstring_records_orig_params_requirement`
    （当时记的是上一代包裹层要求 `use_orig_params=True`）。约束的**形状**没变：
    「分组抓到的对象必须就是 optimizer 绑定的对象」，只是满足它的机制从
    `use_orig_params=True` 换成了「DDP 根本不扁平化」。反过来也钉一下：退役的
    `use_orig_params` 说法不得留在 docstring 里冒充现役约束。
    """
    src = ast.get_source_segment(SRC, _func('_build_param_groups')) or ''
    for token in ('DDP', '扁平化', 'nn.Parameter'):
        assert token in src, \
            "_build_param_groups 的 docstring 必须写明 %r（DDP 不扁平化 ⇒ 分组对象 " \
            "与 optimizer 绑定的是同一批 nn.Parameter）" % token
    assert 'use_orig_params' not in src, \
        'use_orig_params 是已退役包裹层的约束，不得留在 docstring 里'


def test_train_sft_still_compiles():
    """语法门（迁自被删文件）。"""
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(SRC_PATH)],
                       capture_output=True, text=True)
    assert r.returncode == 0, 'train_sft.py 语法错误: {}'.format(r.stderr)


# --------------------------------------------------------------------------- #
# 10. 行为测试：真跑一次包裹（gloo + world_size=1 + CPU）
# --------------------------------------------------------------------------- #
# 上面全是 AST —— 而 2026-09 的 `MixedPrecision(cast_forward_precision=...)` 事故
# 之所以一路活到云端，正是因为**包裹在本地从未被执行过**（world_size=1 不进这条
# 分支），AST 又只查结构。下面这几条用 world_size=1 的 gloo 进程组把 DDP 包裹真跑
# 一遍（构造 → 前向/反向 → state_dict → save_model → EMA.update），覆盖只有执行
# 才暴露的事：键名形态、存档可读性、以及**包裹会不会破坏 EMA 已建的键空间**
# ——最后这条正是 2026-10-01 4 卡事故的机理。
@pytest.fixture(scope='module')
def _gloo_pg():
    """world_size=1 的 gloo 进程组；环境不支持时 skip（不假装通过）。"""
    import os
    import socket
    import torch.distributed as dist
    if dist.is_initialized():
        yield dist
        return
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip('本机 torch 无 gloo，跳过 DDP 执行测试')
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = str(port)
    try:
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    except Exception as e:  # noqa: BLE001
        pytest.skip('本机无法初始化 gloo 进程组（跳过 DDP 执行测试）：{}'.format(e))
    yield dist
    dist.destroy_process_group()


class _Block(torch.nn.Module):
    """带 BN 的块：buffer 的存在是必要的（DDP 默认 broadcast_buffers=True）。"""

    def __init__(self, c):
        super().__init__()
        self.fc = torch.nn.Linear(c, c)
        self.bn = torch.nn.BatchNorm1d(c)

    def forward(self, x):
        return self.bn(self.fc(x))


def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(_Block(8), torch.nn.Linear(8, 4))


def test_ddp_wrap_runs_and_keeps_inner_module_identity(_gloo_pg):
    """包裹能真跑通，且**内部**模块对象原样不动（EMA 键空间的前提）。

    顶部一层 wrapper 是允许的（键只多 `module.` 前缀）；把内部模块换成别的东西
    就是本次要消灭的那类崩溃。断言用**对象身份**而不只是名字：名字相同但对象被
    换过（包装 / 展平 / 重建）一样会破坏 EMA 已持有的 param 引用。
    """
    from torch.nn.parallel import DistributedDataParallel

    model = _tiny_model()
    before = {n: p for n, p in model.named_parameters()}
    wrapped = DistributedDataParallel(model)  # CPU + gloo ⇒ device_ids 必须为 None
    assert wrapped.module is model, 'DDP 应持有**原对象**，不替换'
    after = dict(wrapped.module.named_parameters())
    assert set(after) == set(before), '包裹后内部参数名集合变了'
    for n in after:
        assert after[n] is before[n], (
            'L%d 处创建的参数 %s 的对象被替换了（DDP 只应在顶层加 wrapper，内部'
            '模块树必须原样不动）' % (_param_def_lineno(n), n))


def test_ddp_state_dict_is_module_prefixed_and_save_model_strips_it(_gloo_pg, tmp_path):
    """存档：包裹后键是 `'module.' + 裸键`，`save_model` 剥掉后与未包裹时**逐键相同**。

    这条锁住 spec §5.5 的结论：存档格式与包裹前同形 ⇒ `src/inference.py` /
    `scripts/evaluate.py` / `webui.py` / `_load_model_state` 一行都不用改，
    旧 checkpoint 继续可读。若哪天 `save_model` 的 `'module.'` 剥离被「优化」掉，
    这里立刻红。
    """
    from torch.nn.parallel import DistributedDataParallel
    from scripts.train_sft import save_model

    model = _tiny_model()
    reference_keys = set(model.state_dict())

    bare_path = tmp_path / 'bare.pt'
    save_model(model, str(bare_path))
    bare = torch.load(str(bare_path), weights_only=True)
    assert set(bare) == reference_keys, '裸模块存档键不该被改动'

    wrapped = DistributedDataParallel(model)
    sd = wrapped.state_dict()
    assert all(k.startswith('module.') for k in sd), \
        'DDP 的 state_dict 键应全部带 module. 前缀，实际：%s' % sorted(sd)[:3]
    assert {k[len('module.'):] for k in sd} == reference_keys, \
        '剥掉 module. 后应与未包裹时同形'

    wrapped_path = tmp_path / 'wrapped.pt'
    save_model(wrapped, str(wrapped_path))
    got = torch.load(str(wrapped_path), weights_only=True)
    assert set(got) == reference_keys, (
        '存档键与未包裹时不一致：多 %s 少 %s'
        % (sorted(set(got) - reference_keys)[:5],
           sorted(reference_keys - set(got))[:5]))
    for k, v in bare.items():
        assert torch.equal(got[k], v), '%s 的值被改动了' % k


def test_ema_built_before_wrapping_still_updates(_gloo_pg):
    """**本次事故的回归测试**：包裹前构造的 EMA，包裹后 `update()` 不得 KeyError。

    事故形状（4 卡 910A，2026-10-01）：`ema.update()` 抛
    `KeyError: 'backbone.stem_bn._fsdp_wrapped_module.weight'` —— 上一代包裹层把
    **内部**模块就地换成 wrapper，于是「EMA 构造时建的 shadow 键」与「update 时
    遍历到的参数名」不再是同一个键空间。DDP 只在顶层包一层 ⇒ 两条路径都停在
    裸模块上。

    这里不模拟多 rank（world_size=1 跑不出跨 rank 发散），钉的是这条崩溃的
    **机理**：键空间必须一致。

    ⚠ **这条断言的牙在哪里**（2026-10-01 评审记录）：原先写的是
    `assert set(ema.shadow) == shadow_before`，而 `EMA.update()` 只**改写已有键的
    值**、从不改键集 ⇒ 那个等式**恒真**，红不了任何东西 —— 读的人却会以为
    「键空间一致」这件事在这里被守着，而实际上本条的全部牙都在
    「`ema.update()` 没抛 `KeyError`」这一行。恒真的断言比没有断言更坏：它把一处
    没被守住的地方装饰成被守住了。所以现在把那条恒真等式删掉，换成**能红的**
    形态：shadow 的键必须与**裸模块**的键同形（尤其不得带 `module.` 前缀）——
    若哪天有人在包裹**之后**构造 EMA，`shadow` 的键就会带上 `module.` 前缀，
    这里立刻红；而那一刻正是事故的机理（键空间分叉）。
    """
    from torch.nn.parallel import DistributedDataParallel
    from scripts.train_sft import EMA

    model = _tiny_model()
    ema = EMA(model, decay=0.999)          # 与 main() 同序：先 EMA，后包裹

    wrapped = DistributedDataParallel(model)
    assert wrapped.module is ema.model, 'EMA 持有的引用必须就是被包裹的那个模块'

    # 反向 + 一步 optimizer，让参数真的变一下，再更新 shadow
    wrapped(torch.randn(4, 8)).sum().backward()
    ema.update()          # 事故点：这里曾经抛 KeyError

    # 能红的断言：键空间必须停在裸模块上（详见 docstring —— 「键集合没变」恒真，
    # 不能用那种写法）。
    prefixed = sorted(k for k in ema.shadow if k.startswith('module.'))
    assert not prefixed, (
        'shadow 的键不该带 module. 前缀（EMA 构造在包裹之前，键空间属于裸模块）：%s'
        % prefixed)
    assert set(ema.shadow) == {n for n, _ in model.named_parameters()}, \
        'shadow 的键应与裸模块 named_parameters() 同键空间'
    assert all(torch.isfinite(v).all() for v in ema.shadow.values()), \
        'update 后 shadow 出现 NaN/Inf'


def test_optimizer_state_dict_needs_no_gathering(_gloo_pg):
    """optimizer 存档：`optimizer.state_dict()` 直接就是完整动量，不需要任何汇聚。

    这是与被删文件里 `test_all_optimizer_state_dict_saves_go_through_fsdp`
    **方向相反**的不变量：那边守「不许裸调」（分片下会丢动量），这边守
    「不许绕」（绕回去只会复活退役路径）。
    """
    from torch.nn.parallel import DistributedDataParallel

    model = _tiny_model()
    wrapped = DistributedDataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    wrapped(torch.randn(4, 8)).sum().backward()
    opt.step()

    saved = opt.state_dict()
    assert saved['state'], 'AdamW 走过一步后 state 不该为空'
    # 键是参数序号（0..n-1）而非分片名：这是 DDP 下「与 _load_optimizer_state
    # 的预期一致」的根据。
    assert all(isinstance(k, int) for k in saved['state']), \
        'optimizer state 的键应是参数序号，实得 %s' % sorted(saved['state'])[:3]
