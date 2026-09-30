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
import pathlib
import subprocess
import sys

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
    """`use_orig_params=True` 是硬需求（见模块 docstring 第 1 条）。"""
    node = _func('_wrap_fsdp1')
    calls = [c for c in ast.walk(node)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == 'FullyShardedDataParallel']
    assert calls, '_wrap_fsdp1 里没有调用 FullyShardedDataParallel'
    kws = _kwargs_of(calls[0])
    assert 'use_orig_params' in kws, 'FSDP 构造漏了 use_orig_params，默认 False 会毁掉 param 分组'
    val = kws['use_orig_params']
    assert isinstance(val, ast.Constant) and val.value is True, \
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
    node = _func('_wrap_fsdp1')
    strategy = None
    for c in ast.walk(node):
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) \
                and c.func.id == 'FullyShardedDataParallel':
            v = _kwargs_of(c).get('sharding_strategy')
            if v is not None:
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
