"""FSDP1 取代 DDP 的结构与语义测试。

为什么这些断言长这样（而不是「跑起来没报错」就够）：

FSDP1 相对 DDP 的三处改动，每一处都能**静默**产出错误训练结果而不抛异常，
所以必须逐条钉住：

1. ``use_orig_params=True`` —— 本文件 ``_build_param_groups`` 与 ``EMA`` 都在
   包裹**之前**持有原始 param 对象。FSDP 默认会把参数换成 ``FlatParameter``，
   于是 optimizer 里的 param 与 EMA 的 shadow 逐张量对照时长度/顺序错位，
 �� ``p.ndim == 1`` 的 no_decay 判据恒为 False（扁平张量是 1 维，但语义是
   "被切成一片的参数"而不是"一个 bias/LN 权重"）。全程无异常，只是分组错。

2. ``_fsdp_full_state_dict`` —— 分片后 ``model.state_dict()`` 只含本 rank 的
   切片。不走汇聚就 ``torch.save``，得到的是 1/world_size 的碎片存档，
   ``evaluate.py`` / ``inference`` / ``webui`` / P4.8 的两代加载全部读不了。
   这同样不抛异常，只在下游 load 时才炸。

3. ``_fsdp_full_optimizer_state`` —— 同理，动量是残缺的，续训等于丢历史。
   AdamW 依然能跑，只是收敛行为悄悄变了。

4. 包裹顺序在 ``torch.compile`` **之后** —— 反过来 compile 会被 FSDP 的
   动态边界吞掉，拿到未融合图；P4.7 的 Linear-only 编译路径尤其敏感。

本文件不启动多进程（单机单卡没有 world_size>1，测不到真分片），
所以断言全部走 **AST / 源码** 层面：这些是「必须存在且必须长成某样」的
静态不变量，比跑一次前向更能防住未来的误改。
"""
import ast
import inspect
import pathlib
import subprocess
import sys

import pytest
import torch

SRC_PATH = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'train_sft.py'
SRC = SRC_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SRC)


def _func(name):
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError('train_sft.py 里没有找到函数 {}'.format(name))


def _calls_in(node, attr):
    """函数体内对 `attr` 的属性调用点。"""
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Attribute) and f.attr == attr:
                out.append(sub)
            elif isinstance(f, ast.Attribute) and f.attr.endswith('.' + attr):
                out.append(sub)
    return out


# ---------------------------------------------------------------- 1. 策略选择

def _body_src(node):
    """只取函数体（剥掉 docstring 与签名），避免注释/文档串污染文本断言。

    用 AST 的 lineno 而不是按行数切：docstring 可能跨多行，删首行会把
    文档的剩余部分当成代码，文本断言就会自我误伤（本文件开发时正是这样
    被 `stop_flag.item()` 出现在 docstring 里、以及 `model.state_dict()`
    出现在 save_model 文档里各坑了一次）。
    """
    lines = SRC.splitlines()
    body = node.body
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # 剥掉 docstring 语句本身
    if not body:
        return ''
    lo = body[0].lineno - 1
    hi = body[-1].end_lineno
    return '\n'.join(lines[lo:hi])


def test_no_ddp_wrapper_remains():
    """DDP 必须被彻底移除：残留一行 DistributedDataParallel 就是漏改的信号。"""
    assert 'DistributedDataParallel' not in SRC, \
        'train_sft.py 仍有 DistributedDataParallel 引用；DDP→FSDP1 是替换不是并存'
    assert 'torch.nn.parallel' not in SRC, \
        'train_sft.py 仍引用 torch.nn.parallel（DDP 命名空间）'


def _kwargs_of(call):
    """取调用点的关键字实参表。"""
    return {k.arg: k.value for k in call.keywords}


def test_fsdp_wrapper_uses_orig_params():
    """`use_orig_params=True` 是硬需求（见模块 docstring 第 1 条）。

    2026-09 起这些断言改读 `_fsdp_ctor_kwargs` 的返回 dict（构造参数的唯一
    真相源），不再读调用点 —— 调用点是 `FSDP(model, **kwargs)`，`**` 的键在
    AST 里不可见。
    """
    node = _func('_fsdp_ctor_kwargs')
    entry = None
    for sub in ast.walk(node):
        if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
            for k, v in zip(sub.value.keys, sub.value.values):
                if isinstance(k, ast.Constant) and k.value == 'use_orig_params':
                    entry = v
    assert entry is not None, 'FSDP 构造漏了 use_orig_params，默认 False 会毁掉 param 分组'
    assert isinstance(entry, ast.Constant) and entry.value is True, \
        'use_orig_params 必须是 True（False 会把参数换成 FlatParameter）'


def test_fsdp_wrapper_targets_fsdp1_class():
    """FSDP1 = FullyShardedDataParallel；FSDP2 是 fully_shard 的 composable API。"""
    node = _func('_wrap_fsdp1')
    names = {c.func.id for c in ast.walk(node)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert 'FullyShardedDataParallel' in names, '必须用 FSDP1 的 FullyShardedDataParallel'
    assert 'fully_shard' not in names, \
        '这是 FSDP1 任务；fully_shard 是 FSDP2 的 composable API，语义与本文件的 ' \
        'param 引用持有方式不兼容'


def test_fsdp_wrapper_uses_shard_grad_op():
    """训练态显存 ∝ 1/world_size 靠的是 SHARD_GRAD_OP。"""
    node = _func('_fsdp_ctor_kwargs')
    strategy = None
    for sub in ast.walk(node):
        if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
            for k, v in zip(sub.value.keys, sub.value.values):
                if isinstance(k, ast.Constant) and k.value == 'sharding_strategy':
                    strategy = ast.get_source_segment(SRC, v) or ''
    assert strategy is not None, 'FSDP 构造缺 sharding_strategy，会退化成默认的 NO_SHARD'
    # 判 AST 属性链，不做文本匹配：函数体里有一段解释「为什么不用 SHARD_OP」的
    # 注释，纯文本搜索会把它当成「用了 SHARD_OP」而误报。
    assert strategy.strip() == 'ShardingStrategy.SHARD_GRAD_OP', \
        '分片策略应为 ShardingStrategy.SHARD_GRAD_OP，实为 {}'.format(strategy.strip())


def test_fsdp_auto_wrap_policy_is_class_based():
    """按块类切，不按 batch 相关的 size 阈值切。

    `size_based_auto_wrap_policy` 的阈值随 batch_size 漂移，同一份配置在
    batch 8 与 2800 下会切出不同形状 —— 与 search_arch.py 的 PROBE_BS 语义
    直接冲突，所以必须是 ModuleWrapPolicy。
    """
    body = _body_src(_func('_fsdp_wrap_policy'))
    assert 'ModuleWrapPolicy' in body, 'auto_wrap_policy 应用 ModuleWrapPolicy（按类）'
    assert 'size_based' not in body, \
        '按 size 切会随 batch 漂移，破坏 search_arch 的标定口径'


def test_fsdp_wrap_policy_resolves_every_block_class():
    """元组里的名字必须**逐个**在 backbone 里解析成 Module 子类。

    实现用 `getattr(_bb, _name, None)` 逐名取类，取不到就跳过。这条静默路径的
    后果是：某天 `MambaLTI` 改名，Mamba 块不再被单独切分片，激活峰值反弹，
    而 FSDP 照常启动、不抛任何异常 —— 正是本文件开头说的那种「静默产出错误
    训练结果」。它还有一条更糟的兜底：`classes` 全空时退化成 `[nn.Module]`，
    于是整模型一个 unit，与 ModuleWrapPolicy 的设计意图相反。

    断言名字集合与实际解析结果，改名任一侧都会转红。
    """
    import ast as _ast
    import torch

    node = _func('_fsdp_wrap_policy')
    names = []
    for sub in _ast.walk(node):
        if isinstance(sub, (ast.Tuple, ast.List)):
            if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in sub.elts) \
                    and sub.elts:
                names = [e.value for e in sub.elts]
                break
    assert names, '没在 _fsdp_wrap_policy 里找到块类名元组'

    sys.path.insert(0, str(SRC_PATH.parent.parent))
    from src.networks import backbone as _bb

    resolved = []
    for name in names:
        cls = getattr(_bb, name, None)
        assert isinstance(cls, type) and issubclass(cls, torch.nn.Module), \
            'backbone 里没有块类 {}（改名了？）—— 静默跳过会让该块不被 FSDP 单独切分'.format(name)
        resolved.append(cls)

    assert len(set(names)) == len(names), '元组里有重复类名'
    expected = {'ResBlock', 'CrossAttnRes', 'MambaLTI', 'TransformerBlock'}
    assert set(names) == expected, \
        '块类集合漂移：当前 {}，期望 {}'.format(sorted(names), sorted(expected))
    assert len(resolved) == len(names)


# ---------------------------------------------------------------- 2. 存档与续训

def test_save_model_uses_full_state_dict():
    """分片下直接 state_dict() = 存碎片。"""
    body = _body_src(_func('save_model'))
    assert '_fsdp_full_state_dict' in body, \
        'save_model 必须走 _fsdp_full_state_dict，否则 FSDP 下存档只有分片'
    assert 'model.state_dict()' not in body, \
        'save_model 里有裸 state_dict() 调用，FSDP 下会存下 1/world_size 的碎片'


def test_all_optimizer_state_dict_saves_go_through_fsdp():
    """4 处 optimizer 存档点必须全部汇聚；漏一处就静默丢动量。"""
    full = _func('_fsdp_full_optimizer_state')
    assert full is not None
    # train_sft.py 里 optimizer.state_dict() 只应出现在 helper 内部（回退分支）
    occurrences = SRC.count("'optimizer': optimizer.state_dict()")
    assert occurrences == 0, \
        '还有 {} 处 train_state 直接存 optimizer.state_dict()，FSDP 下动量是残缺的'.format(
            occurrences)
    assert SRC.count("'optimizer': _fsdp_full_optimizer_state(optimizer, model)") >= 3, \
        '期望至少 3 处 train_state 存档点（latest / 定期 / 最佳模型）都走 FSDP 汇聚'


def test_fsdp_full_state_dict_is_rank0_only():
    """`rank0_only=True` 避免每个 rank 都做一次全量 all-gather 汇聚。"""
    for fn in ('_fsdp_full_state_dict', '_fsdp_full_optimizer_state'):
        src = ast.get_source_segment(SRC, _func(fn)) or ''
        assert 'rank0_only=True' in src, \
            '{} 缺 rank0_only=True：非 rank0 也要汇聚会白白多耗一次全量通信'.format(fn)
        assert 'offload_to_cpu=True' in src, \
            '{} 应 offload_to_cpu：汇聚出的完整权重不再放回 GPU'.format(fn)


def test_fsdp_helpers_degrade_gracefully_for_single_process():
    """单卡（is_dist=False）必须退化为原路径，不能因为没有 process group 而炸。"""
    for fn, fallback in (('_fsdp_full_state_dict', 'model.state_dict()'),
                         ('_fsdp_full_optimizer_state', 'optimizer.state_dict()')):
        body = _body_src(_func(fn))
        assert 'isinstance' in body, '{} 需先判类型再决定是否走 FSDP 汇聚'.format(fn)
        assert fallback in body, '{} 的非 FSDP 回退分支缺失'.format(fn)


# ---------------------------------------------------------------- 3. 顺序与不变式

def test_fsdp_wrapping_happens_after_compile():
    """包裹顺序：FSDP1 必须在 torch.compile 之后。

    反过来 compile 会被 FSDP 的动态边界吞掉，拿到未融合图。
    """
    i_compile = SRC.find('torch.compile(model')
    i_fsdp = SRC.find('model = _wrap_fsdp1(model')
    assert i_compile != -1 and i_fsdp != -1, '找不到 compile / FSDP 包裹点'
    assert i_fsdp > i_compile, \
        'FSDP 包裹必须在 torch.compile 之后（当前顺序反了）'


def test_param_groups_built_before_wrapping():
    """`_build_param_groups` 必须在 FSDP 包裹之前调用（依赖裸模块的 `'value.'` 前缀）。"""
    i_groups = SRC.find('_opt_groups = _build_param_groups(model, args)')
    i_fsdp = SRC.find('model = _wrap_fsdp1(model')
    assert i_groups != -1, '找不到 _build_param_groups 调用点'
    assert i_groups < i_fsdp, \
        '_build_param_groups 必须在 FSDP 包裹之前：包裹后名字带 module. 前缀，' \
        'startswith(\'value.\') 失效'


def test_build_param_groups_docstring_records_orig_params_requirement():
    """`_build_param_groups` 的 docstring 必须记下 FSDP 的 `use_orig_params` 约束。

    这条约束是这个文件里最容易被「顺手调换调用顺序」破坏、又最难在运行时
    看出来的：破坏后不抛异常，只是 value 组与 other 组的划分悄悄变了。
    """
    src = ast.get_source_segment(SRC, _func('_build_param_groups')) or ''
    assert 'use_orig_params' in src, \
        '_build_param_groups 的 docstring 必须写明 FSDP 下的 use_orig_params 约束'


def test_stop_flag_broadcast_still_unconditional():
    """早停 broadcast 的「无短路」不变量在 FSDP 下同样成立。

    `_sync_stop_flag` 的无条件 broadcast 是防挂死的关键：一旦按
    `stop_flag.item()` 短路，没停的 rank 跳过 collective、停了的 rank 阻塞。
    FSDP 的 all-gather/reduce-scatter 让 collective 次序错配的后果比 DDP 更糟。

    用 AST 判 `If.test` 里是否出现对 stop_flag 的调用，而不是文本搜索 ——
    函数 docstring 里正是在**讲解**这个反例，文本搜索会自我误伤。
    """
    node = _func('_sync_stop_flag')
    body = _body_src(node)
    assert 'dist.broadcast' in body, '_sync_stop_flag 里的 broadcast 不见了'
    for sub in ast.walk(node):
        if isinstance(sub, ast.If):
            test = ast.get_source_segment(SRC, sub.test) or ''
            assert 'stop_flag' not in test, \
                '_sync_stop_flag 不得对 stop_flag 做短路求值（会破坏 collective 次序）'


def test_early_stop_sync_guards_multi_rank():
    """`is_dist` 判定仍是 `world_size > 1`（FSDP1 不改环境变量契约）。"""
    assert "world_size = int(os.environ.get('WORLD_SIZE', '1'))" in SRC, \
        'WORLD_SIZE 环境变量契约被改动：torchrun 注入的就是它'
    assert 'is_dist = world_size > 1' in SRC, 'is_dist 判定应保持 world_size > 1'


def test_train_sft_still_compiles():
    """语法门。"""
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(SRC_PATH)],
                       capture_output=True, text=True)
    assert r.returncode == 0, 'train_sft.py 语法错误: {}'.format(r.stderr)


# --------------------------------------------------------------------------- #
# 6. FSDP 调用的 kwarg 必须是**真实 API**（2026-09 云端事故）
# --------------------------------------------------------------------------- #
# 本文件前面的断言全是「源码里必须出现某段结构」，它们查不出**参数名拼错 /
# 根本不属于该类**：kwarg 名字对不对，只有被调用的类自己知道。而 FSDP 包裹
# 只在 world_size>1 时执行 → 单机单卡的本地测试永远走不到，4 卡一启动就崩：
#
#   File "train_sft.py", line 243, in _wrap_fsdp1
#     mp = MixedPrecision(
#   TypeError: __init__() got an unexpected keyword argument
#              'cast_forward_precision'
#
# `cast_forward_precision` / `cast_root_forward_precision` /
# `keep_low_precision_grads` / `cast_forward_inputs` 是
# `FullyShardedDataParallel` 构造参数（或根本不存在），`MixedPrecision` 只收
# `param_dtype` / `reduce_dtype` / `buffer_dtype` / `keep_low_precision_module_wrapper`。
#
# 下面三条断言把「kwarg 名」交给**真实安装的 torch** 去判：AST 取代码里实际
# 传的 kwarg，对 `inspect.signature` 的参数表求差集。⚠ 本地 torch 版本可能
# 比云端新，所以另配一条**版本无关**的白名单断言（只允许三个 dtype 参数），
# 两条合起来才能覆盖「本地过、云端炸」的方向。
def _call_kwargs_by_name(callee):
    """train_sft.py 里对 `callee(...)` 的调用点：{kwarg: 字面值源码}。

    ⚠ FSDP 构造的 kwargs 不走这里 —— `_wrap_fsdp1` 用
    `FullyShardedDataParallel(model, **kwargs)` 展开，`**` 的键在 AST 里看不见。
    那份由 `_dict_literal_keys('_fsdp_ctor_kwargs')` 从**返回的 dict 字面量**
    里读，与运行时读的是同一份源码。
    """
    out = {}
    for sub in ast.walk(TREE):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)
        if name != callee:
            continue
        for kw in sub.keywords:
            if kw.arg is None:      # **kwargs 展开，不参与静态判定
                continue
            out[kw.arg] = ast.get_source_segment(SRC, kw.value)
    return out


def _dict_literal_keys(func_name):
    """取 `func_name` 里 return 的 dict 字面量的键名集合。"""
    node = _func(func_name)
    keys = None
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Return) or not isinstance(sub.value, ast.Dict):
            continue
        got = {k.value for k in sub.value.keys if isinstance(k, ast.Constant)
               and isinstance(k.value, str)}
        assert got, '{} 的 return 不是 dict 字面量或键非字面量，无法静态判定'.format(func_name)
        keys = got if keys is None else (keys & got)
    assert keys, '{} 里找不到 return dict'.format(func_name)
    return keys


def test_mixed_precision_kwargs_are_supported_by_installed_torch():
    from torch.distributed.fsdp import MixedPrecision

    kwargs = _call_kwargs_by_name('MixedPrecision')
    assert kwargs, 'MixedPrecision 调用点不见了（_wrap_fsdp1 应显式构造它）'
    params = set(inspect.signature(MixedPrecision.__init__).parameters) - {'self'}
    unknown = sorted(set(kwargs) - params)
    assert not unknown, (
        '这些 kwarg 不是 MixedPrecision 的参数（云端 4 卡会直接 TypeError）：'
        '{}\n实际签名: {}'.format(unknown, sorted(params)))
    # 真能构造出来：MixedPrecision 是纯配置对象，单机即可实例化
    MixedPrecision(**{k: _literal(v) for k, v in kwargs.items()})


def test_mixed_precision_only_passes_dtype_kwargs():
    """版本无关的白名单：混合精度由 autocast 承担，FSDP 侧只允许「不设」三档。

    这条比签名比对更耐版本差异：`cast_forward_precision` 这类参数即便某个
    torch 版本碰巧加进 `MixedPrecision`，按本仓库的意图（autocast 负责精度，
    FSDP 不做转换 —— 见 `_wrap_fsdp1` 的注释）也**不该**传。
    """
    kwargs = _call_kwargs_by_name('MixedPrecision')
    allowed = {'param_dtype', 'reduce_dtype', 'buffer_dtype'}
    assert set(kwargs) <= allowed, (
        'MixedPrecision 只应传 {}（值取 None = FSDP 不做精度转换），实际传了 {}'
        .format(sorted(allowed), sorted(kwargs)))
    for k, v in kwargs.items():
        assert v.strip() == 'None', \
            'MixedPrecision 的 {} 应保持 None（精度交给 autocast），实际 {}'.format(k, v)


def test_fsdp_ctor_kwargs_are_supported_by_installed_torch():
    from torch.distributed.fsdp import FullyShardedDataParallel

    kwargs = _dict_literal_keys('_fsdp_ctor_kwargs')
    assert kwargs, '读不到 _fsdp_ctor_kwargs 的返回 dict'
    params = set(inspect.signature(FullyShardedDataParallel.__init__).parameters) - {'self'}
    unknown = sorted(kwargs - params)
    assert not unknown, (
        '这些 kwarg 不是 FullyShardedDataParallel 的参数：{}\n实际签名: {}'
        .format(unknown, sorted(params)))


def test_fsdp_wrapper_still_passes_the_expected_kwargs():
    """`_wrap_fsdp1` 必须经 `**kwargs` 展开调用构造（不许散装写 kwarg）。

    这是防「有人把 dict 展开改成直接写在构造调用里」——那样静态守护立刻失明
    （`**d` 的键在 AST 里看不见），只能退回云端炸。判 AST 的 `keyword(arg=None)`
    而不是文本搜 `**`（docstring 里出现 `**` 会造成假绿：变异实测过）。
    """
    node = _func('_wrap_fsdp1')
    assert '_fsdp_ctor_kwargs' in (ast.get_source_segment(SRC, node) or ''), \
        '_wrap_fsdp1 应通过 _fsdp_ctor_kwargs(...) 构造参数'
    calls = [c for c in ast.walk(node)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == 'FullyShardedDataParallel']
    assert calls, '_wrap_fsdp1 里没有 FullyShardedDataParallel 调用'
    unpacked = [k for k in calls[-1].keywords if k.arg is None]
    explicit = [k.arg for k in calls[-1].keywords if k.arg]
    assert unpacked, \
        'FSDP 构造必须以 **kwargs 展开调用（当前改成了散装 kwarg：{}），否则静态守护失明'.format(
            explicit)
    assert not explicit, \
        'FSDP 构造里出现了散装 kwarg {}，参数集就多了一份真相源'.format(explicit)


def _literal(src):
    """把极小的字面量源码求值（只认 None/True/False/数字/字符串）。"""
    import ast as _ast
    return _ast.literal_eval(src.strip())


def test_fsdp_ctor_kwargs_are_whitelisted():
    """传进 FSDP1 构造的 kwarg 必须在白名单里，且白名单 ⊆ 已安装 torch 的签名。

    为什么白名单还要再钉一层：`_wrap_fsdp1` 现在按**已安装签名**过滤未知参数
    （开发机 torch 2.12 / 云端 2.1 两头都不保证），过滤让「云端崩」降级成
    「功能降级 + 告警」。但过滤是**兜底**，不是许可证 —— 白名单保证我们只依赖
    「2.0 起就存在」的那几个参数，不去碰 2.x 中途新增的 API。
    """
    sys.path.insert(0, str(SRC_PATH.parents[1]))
    from scripts.train_sft import FSDP_CTOR_KWARGS_WHITELIST
    from torch.distributed.fsdp import FullyShardedDataParallel

    used = set(_dict_literal_keys('_fsdp_ctor_kwargs'))
    assert used == set(FSDP_CTOR_KWARGS_WHITELIST), (
        '实际传入的 kwarg {} 与白名单 {} 不一致 —— 新增参数前请先确认它在云端 '
        'torch 2.1 上也存在'.format(sorted(used), sorted(FSDP_CTOR_KWARGS_WHITELIST)))
    params = set(inspect.signature(FullyShardedDataParallel.__init__).parameters) - {'self'}
    missing = sorted(set(FSDP_CTOR_KWARGS_WHITELIST) - params)
    assert not missing, '白名单里的 {} 在当前 torch 上不存在'.format(missing)
    # 白名单必须含两个硬需求：漏了任何一条都会静默毁掉训练语义
    assert {'use_orig_params', 'sharding_strategy'} <= set(FSDP_CTOR_KWARGS_WHITELIST)


def test_drop_unsupported_kwargs_filters_and_reports():
    """`_drop_unsupported_kwargs`：不认的丢掉并报告，认得的一个不动。"""
    sys.path.insert(0, str(SRC_PATH.parents[1]))
    from scripts.train_sft import _drop_unsupported_kwargs

    class _New(torch.nn.Linear):
        def __init__(self, in_features, out_features, bias=True,
                     device=None, dtype=None, new_fancy_kwarg=None):
            super().__init__(in_features, out_features, bias=bias,
                             device=device, dtype=dtype)
            self.new_fancy_kwarg = new_fancy_kwarg

    class _Old(torch.nn.Linear):
        def __init__(self, in_features, out_features, bias=True,
                     device=None, dtype=None):
            super().__init__(in_features, out_features, bias=bias,
                             device=device, dtype=dtype)

    kwargs = {'bias': False, 'new_fancy_kwarg': 1}
    dropped = _drop_unsupported_kwargs(_Old, kwargs, 'x', None)
    assert kwargs == {'bias': False}, '未被识别的参数应被移出 kwargs'
    assert dropped == ['new_fancy_kwarg'], dropped

    kwargs = {'bias': False, 'new_fancy_kwarg': 1}
    dropped = _drop_unsupported_kwargs(_New, kwargs, 'x', None)
    assert kwargs == {'bias': False, 'new_fancy_kwarg': 1}, '认得的参数不能被丢掉'
    assert dropped == []


# --------------------------------------------------------------------------- #
# 7. 真跑一次 FSDP1（world_size=1、gloo、CPU）
# --------------------------------------------------------------------------- #
# 前面所有断言都是 AST —— 而 2026-09 那次事故（`MixedPrecision(
# cast_forward_precision=...)` → TypeError）之所以能一路活到云端，正是因为
# **FSDP 包裹在本地从未被执行过**（world_size=1 不进这条分支），AST 又只查
# 「有没有 use_orig_params」这类结构在不在，查不出参数名是不是真 API。
#
# 这里用 world_size=1 的 gloo 进程组把包裹真跑一遍：MixedPrecision 构造 →
# auto_wrap → 前向/反向 → FULL_STATE_DICT 汇聚 → optimizer 状态汇聚。
# 不起多进程（Windows 上 spawn 慢且易挂），但覆盖了「参数集被真 API 接受」
# 「use_orig_params 下原始 param 引用仍可用」「汇聚后的键名与未包裹时一致」
# 这三件只有执行才暴露的事。真正的多卡语义仍由云端实测负责。
@pytest.fixture(scope='module')
def _gloo_pg():
    """world_size=1 的 gloo 进程组；环境不支持时 skip（不假装通过）。"""
    import os
    import socket
    import torch.distributed as dist
    if dist.is_initialized():
        yield
        return
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip('本机 torch 无 gloo，跳过 FSDP 执行测试')
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = str(port)
    try:
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    except Exception as e:  # noqa: BLE001
        pytest.skip('本机无法初始化 gloo 进程组（跳过 FSDP 执行测试）：{}'.format(e))
    yield
    dist.destroy_process_group()


def test_fsdp1_wrap_actually_runs(_gloo_pg):
    """FSDP1 包裹 + 前向/反向 + 完整 state_dict 汇聚，真跑一遍（1 rank）。

    ⚠ 需要**非 CPU 加速器**：torch 2.12 的 FSDP1 在纯 CPU 上直接
    `RuntimeError: FSDP needs a non-CPU accelerator device`（本仓库开发机是
    `torch 2.12.0+cpu`，所以这里如实 skip）。有卡的环境（云端 910A / CI 的
    CUDA 机）上这条会真跑 —— 那正是它存在的意义：AST 断言看不见「参数集被真
    API 接受」这种事。
    """
    if not (torch.cuda.is_available()
            or (hasattr(torch, 'npu') and torch.npu.is_available())):
        pytest.skip('本机无非 CPU 加速器（torch FSDP1 硬性要求），跳过 FSDP 执行测试')
    sys.path.insert(0, str(SRC_PATH.parents[1]))
    import torch.nn as nn
    from scripts.train_sft import _fsdp_ctor_kwargs, _drop_unsupported_kwargs
    from torch.distributed.fsdp import (FullyShardedDataParallel, MixedPrecision,
                                        ShardingStrategy)

    class _Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.fc = nn.Linear(c, c)
            self.bn = nn.BatchNorm1d(c)

        def forward(self, x):
            return self.bn(self.fc(x))

    torch.manual_seed(0)
    model = nn.Sequential(_Block(8), nn.Linear(8, 4))
    reference_keys = set(model.state_dict())

    mp = MixedPrecision(param_dtype=None, reduce_dtype=None, buffer_dtype=None)
    kwargs = _fsdp_ctor_kwargs(model, None, mp)
    assert not _drop_unsupported_kwargs(FullyShardedDataParallel, kwargs,
                                        'FullyShardedDataParallel', None), \
        '本机 torch 上就有构造参数不被支持（先修白名单/过滤器再谈云端）'

    wrapped = FullyShardedDataParallel(model, **kwargs)
    assert isinstance(wrapped, FullyShardedDataParallel)
    assert kwargs['sharding_strategy'] == ShardingStrategy.SHARD_GRAD_OP

    # use_orig_params=True 的核心承诺：optimizer 仍能用**未包裹前**的参数对象
    params = [p for p in model.parameters()]
    assert params, '参数引用为空'
    opt = torch.optim.SGD(params, lr=0.1)
    out = wrapped(torch.randn(4, 8)).sum()
    out.backward()
    opt.step()

    from scripts.train_sft import _fsdp_full_optimizer_state, _fsdp_full_state_dict
    full = _fsdp_full_state_dict(wrapped)
    assert set(full) == reference_keys, \
        '汇聚后的键名与未包裹时不一致：多 {} 少 {}'.format(
            sorted(set(full) - reference_keys), sorted(reference_keys - set(full)))
    assert _fsdp_full_optimizer_state(opt, wrapped) is not None
