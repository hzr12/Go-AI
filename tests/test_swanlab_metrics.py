"""SwanLab 上报面（2026-10-01 扩充）。

这次改了两类东西，都容易「加了变量却没上报」或「上报了但没意义」：

1. **config 面板此前记的是 9 个 D1 已归档的结构 flag** —— 它们完全不参与建网
   （结构恒由 `KATAGO_SE_CFG` 这张表决定，v21 硬删除后旧 `V21_CFG` 已不存在），
   而 config 面板是对比两次 run 时第一个看的东西。记虚构值比不记更糟。
2. **一批「算了但没上报」或「上报了但没有参照」的量**：
   - `grad_norm`：`clip_grad_norm_` 的返回值此前被**丢弃** —— fp16 溢出/梯度爆炸
     唯一的直接信号；
   - `l2_report` / `opt_loss`：此前 `loss` 不可分解；
   - `policy_ce_random` / `value_rmse_zero`：**基线**。没有基线的 loss 曲线分不清
     「学到了」和「从 5.89 降到 5.5」。

本文件钉住「该报的都在报」与「基线成对出现」，防止以后又出现新的「算了没报」。
"""
import ast
import os
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / 'scripts' / 'train_sft.py').read_text(encoding='utf-8')
TREE = ast.parse(SRC)
CODE = '\n'.join(ln for ln in SRC.splitlines() if not ln.lstrip().startswith('#'))


def _fn_src(name):
    fn = next(f for f in ast.walk(TREE)
              if isinstance(f, ast.FunctionDef) and f.name == name)
    return ast.unparse(fn)


MAIN = _fn_src('main')

def _has(hay, key):
    """ast.unparse 会把双引号归一成单引号，两种都认。"""
    return ('"%s"' % key) in hay or ("'%s'" % key) in hay

INIT = _fn_src('_init_swanlab')


# --------------------------------------------------------------------------- #
# 1. 算出来的东西必须真的被上报
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('key', [
    'grad_norm',            # clip_grad_norm_ 的返回值（曾被丢弃）
    'l2_report',            # loss 的第三项
    'opt_loss',             # 真正被 backward 的量
    't_data_max_ms',        # 取数长尾
    'speed_per_card',       # 每卡吞吐
    'eta_min',              # 剩余时间
    'elapsed_min',
])
def test_step_metric_is_reported(key):
    assert _has(MAIN, key), f'{key} 算/定义了但没上报到 swanlab'


@pytest.mark.parametrize('key', [
    'train_top1', 'train_top5', 'value_rmse',
    'policy_ce_random', 'value_rmse_zero', 'policy_entropy',
])
def test_health_metric_is_reported(key):
    assert _has(MAIN, key), f'健康度 {key} 未上报'


@pytest.mark.parametrize('key', [
    'eval_used_ema', 'eval_lr', 'eval_gap_to_best',
])
def test_eval_metric_is_reported(key):
    assert _has(MAIN, key), f'eval 侧 {key} 未上报'


def test_every_loss_curve_has_a_baseline():
    """policy_loss 与 value_loss 各要有一条「什么都没学到」的基线。

    这是这次扩充的核心动机：没有基线时「从 5.89 降到 5.5」和「真的学到了」
    在图上完全一样。所以断言基线与被基线化的量**成对存在**。
    """
    for curve, baseline in (('policy_loss', 'policy_ce_random'),
                            ('value_rmse', 'value_rmse_zero')):
        assert curve in MAIN, f'{curve} 未上报'
        assert baseline in MAIN, \
            f'{curve} 没有基线 {baseline} —— 曲线不可读（分不清学没学到）'


def test_grad_norm_is_taken_after_unscale():
    """`grad_norm` 必须在 `scaler.unscale_()` **之后**取才是真值。

    unscale 之前梯度还乘着 loss scale（默认 1024），量出来的范数大三个数量级 ——
    曲线看着「梯度爆炸」，而那只是缩放的假值。

    ⚠ 判据必须在 **main() 内**取位置，不能用 `CODE.index()` 找全文件第一个：
       `_locate_overflow` 的 docstring 里也会提到 `clip_grad_norm_`（解释
       「为什么诊断必须在 clip 之前」），而它定义在 main 之前 ⇒ 全文件搜索会
       命中那段说明文字，于是这条门禁变成在测文档而不是测代码。
    """
    i_un = MAIN.index('scaler.unscale_(optimizer)')
    i_gn = MAIN.index('clip_grad_norm_')
    assert i_un < i_gn, 'grad_norm 必须在 unscale_ 之后取'


def test_grad_norm_carries_forward_across_micro_batches():
    """accum>1 时每个 optimizer step 只有一个 grad_norm，打点是每 micro-batch。

    所以必须 carry forward（沿用上一次的），否则曲线在非 optimizer step 的打点上
    是 nan / 旧值 / 跳变，accum 越大越难看。
    """
    assert '_grad_norm_last' in MAIN, '缺 grad_norm 的沿用变量'
    assert MAIN.count('_grad_norm_last') >= 3, \
        '应在「初始化 / 赋值 / 上报」三处出现（沿用语义）'


def test_health_metrics_only_computed_on_log_steps():
    """健康度算在**上报分支**里，不是每个 micro-batch。

    每步算会白付 ~5 次 D2H 同步；`--swanlab-every 10` 下打点频率只有 1/10，
    放错位置等于 10 倍浪费。
    """
    # 用原始源码（ast.unparse 不保序）：健康度块必须落在 `if _do_swanlab:` 之后
    i_branch = SRC.index('if _do_swanlab:')
    i_health = SRC.index('_health_last = {')
    assert i_branch < i_health, '健康度必须在 if _do_swanlab 分支内计算'
    # 且不得在训练主循环里每步都算：主循环里不应再出现 _health_last 的构造
    assert SRC.count('_health_last = {') == 1, \
        '健康度只在上报分支构造一次（每 micro-batch 算会白付 10 倍 D2H 同步）'


# --------------------------------------------------------------------------- #
# 2. config 面板：真结构，不是归档 flag
# --------------------------------------------------------------------------- #
def test_config_panel_has_no_archived_flags():
    """9 个 D1 已归档的结构 flag 不得再出现在 config 面板里。

    它们照旧被 argparse 接受（旧命令行不改），但**不参与建网** —— 记进 config
    会让「对比两次 run」这件最常用的事得到错误答案。
    """
    archived = ('backbone_channels', 'backbone_res_blocks', 'res_blocks',
                'convnext_blocks', 'attn_blocks', 'value_channels',
                'value_res_blocks', 'policy_channels', 'policy_layers')
    cfg = INIT[INIT.index('config={'):]
    leaked = [a for a in archived if _has(cfg, a)]
    assert not leaked, f'config 面板仍记归档 flag（对 v21 无效）：{leaked}'


@pytest.mark.parametrize('key', [
    'arch/in_channels', 'arch/channels', 'arch/blocks', 'arch/params_total',
    'grad_checkpoint', 'batch_size_per_card', 'grad_accum', 'world_size',
    'effective_batch', 'lr', 'weight_decay', 'clip_grad_max_norm',
    'label_smoothing', 'attention_dropout', 'ema_enabled', 'ema_decay',
    'amp_dtype', 'scaler_init_scale', 'attn_sdpa_force_math',
    'attn_query_chunk', 'effective_batch',
])
def test_config_panel_records_the_key(key):
    assert _has(INIT, key), f'config 面板缺 {key}'


def test_config_arch_facts_come_from_katago_se_cfg():
    """结构字段必须来自 `KATAGO_SE_CFG` 而不是 args（args 里那些是归档 flag）。

    v21 硬删除后 `V21_CFG` 不复存在，结构唯一真相源是 `scripts/train_sft.py` 里的
    `KATAGO_SE_CFG` —— 断言的对象随之换轨，判据的形状不变：config 面板里的每一个
    结构数字都必须直接引那张表，不得引 args。
    """
    cfg = INIT[INIT.index('config={'):]
    assert "KATAGO_SE_CFG['in_channels']" in cfg
    assert "KATAGO_SE_CFG['channels']" in cfg
    assert "KATAGO_SE_CFG['blocks']" in cfg
    assert "KATAGO_SE_CFG['params_total']" in cfg


def test_effective_batch_includes_accumulation():
    """config 里的 `effective_batch` 必须含 grad_accum —— 漏乘就把 lr 口径搞错。"""
    expr = re.search(r'"effective_batch":\s*([^,\n]+)', SRC).group(1)
    assert 'batch_size' in expr and '_accum' in expr, \
        f'effective_batch 表达式缺累积因子：{expr}'


def test_every_args_attribute_used_in_config_exists():
    """`_init_swanlab` 里引用的每个 `args.X` 都必须是**真实存在的 argparse 参数**。

    这条是被真事件打出来的：2026-10-01 我往 config 面板加了 `"td": args.td`，
    而 `--td`/`--td-steps` 只存在于 RL 侧 `selfplay_train.py`，SFT 的 argparse
    里没有 ⇒ `args.td` 抛 AttributeError ⇒ **整个 swanlab init 失败**
    （`except` 降级成 None，训练照跑但**一条曲线都没有**）。

    那个 except 救的是训练、不是跟踪：静默降级让「跟踪没了」这件事只在日志里
    留一行。所以用测试把「config 里引用的 dest 必须存在」钉死。
    """
    init = SRC[SRC.index('def _init_swanlab'):SRC.index('def _should_log')]
    used = set(re.findall(r'args\.([a-z_0-9]+)', init))

    # argparse 声明的 dest 全集
    dests = set()
    for node in ast.walk(TREE):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            dests.add(node.args[0].value.lstrip('-').replace('-', '_'))
    assert dests, '解析不到任何 add_argument（测试自身失效）'

    missing = sorted(used - dests)
    assert not missing, \
        f'config/swanlab 引用了不存在的 args 属性：{missing} —— 会让整个 ' \
        f'swanlab init 抛异常降级成 None（跟踪全丢，而训练照跑、看不出原因）'


def _namespace_from_argparse():
    """从 main() 的 add_argument 调用重建一个 Namespace（dest + 字面量默认值）。

    parser 是内联构造在 main() 里的（没有独立的 parse_args），所以这里走 AST：
    收集 flag 与能静态求值的 default，其余填占位。**目的不是复现 argparse 语义**
    （required / type 转换都不管），而是给 config 表达式一个「属性名与真实 CLI
    一致」的求值环境 —— 缺一个属性就抛 AttributeError，正是要挡的那个失败。
    """
    import argparse
    fn = next(f for f in ast.walk(TREE)
              if isinstance(f, ast.FunctionDef) and f.name == 'main')
    ns = argparse.Namespace()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant)):
            continue
        flag = node.args[0].value
        dest = flag.lstrip('-').replace('-', '_')
        default = None
        for kw in node.keywords:
            if kw.arg == 'default':
                try:
                    default = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    default = None       # 非字面量（引用模块常量）→ 占位
        setattr(ns, dest, default)
    return ns


def _fn_node(fn):
    tree = ast.parse(SRC)
    return next(f for f in ast.walk(tree)
                if type(f) is ast.FunctionDef and f.name == fn)


import builtins

# ⚠ 必须用 `builtins` 模块，不能用 `dir(__builtins__)`：在**被 import** 的测试
#   模块里 `__builtins__` 是一个 dict，`dir(dict)` 给的是 dict 的方法
#   （keys/values/get…），`len`/`max` 一个都不在里面 ⇒ 顺序检查把它们全误报成
#   「从未赋值」。这不是假设：这个 bug 让本文件第一次跑就红。
_BUILTINS = frozenset(dir(builtins))


def _first_assign_lines(fn_node):
    """{名字: 该函数里第一次赋值的行号}（函数参数不算赋值）。"""
    out = {}
    for n in ast.walk(fn_node):
        if isinstance(n, (ast.For, ast.AsyncFor)):
            # 循环变量也是绑定：`for epoch in range(...)`（L2718）—— 本文件的
            # `epoch` 就是这么来的，漏掉它这条检查会一直误报。
            # ⚠ 不含 `ast.comprehension`：推导式在 Python 3 是**独立作用域**，
            #   它的目标不会成为函数局部变量。
            for nm in _bound_names(n.target):
                out.setdefault(nm, n.lineno)
            continue
        if not isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            continue
        tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
        for t in tgts:
            # ⚠ 必须递归进 Tuple/List：`opt_loss, log_loss = _loss_terms(...)`
            #   是本文件最常见的赋值形式，只认 Name 会把它们全误报成「从未赋值」
            #   —— 让这条检查一上来就红，等于没有检查。
            for nm in _bound_names(t):
                out.setdefault(nm, n.lineno)
    return out


def _bound_names(tgt):
    """赋值目标里绑定的全部名字（穿透 tuple/list 解包与下标/属性目标）。"""
    if isinstance(tgt, ast.Name):
        return [tgt.id]
    if isinstance(tgt, (ast.Tuple, ast.List)):
        return [nm for el in tgt.elts for nm in _bound_names(el)]
    if isinstance(tgt, ast.Starred):
        return _bound_names(tgt.value)
    return []          # Subscript / Attribute：绑定的是容器/字段，不是新名字


def _assigned_before(fn, use_line, names):
    """断言 `names` 里每个名字在 `use_line` 行**之前**已在 `fn` 里被赋值。

    为什么需要这一条（2026-10-01 云端两次踩坑）：
      · `args.td`      —— 属性不存在，AttributeError 被 `except Exception` 吞掉
      · `_accum_steps` —— 名字**确实存在**，只是赋值在 29 行**之后** ⇒
        `UnboundLocalError: local variable ... referenced before assignment`。
        同样被 `except Exception` 吞成一行 warning，于是 SwanLab 少 8 个 run 级
        指标、训练照跑，**没有任何报错**。

    前者靠「属性存在」静态检查能抓到；后者只能靠**顺序**检查 —— Python 自己的
    静态工具（pyflakes/ruff）都不会报：它是个合法局部变量，只是用早了。

    ⚠ 用 `if type(f) is ast.FunctionDef` 而非 `isinstance`：后者会把**嵌套**在
    `main()` 里的 `def`（例如 _build_param_groups 的内层闭包）也算进来，于是内层
    的局部名被误判成「main() 里已赋值」，正好放行我们要抓的那类 bug。
    """
    first = _first_assign_lines(_fn_node(fn))
    names = {nm for nm in names if nm not in _BUILTINS}
    missing = {nm: first.get(nm) for nm in names
               if first.get(nm) is None or first[nm] >= use_line}
    assert not missing, (
        '%s() 里 %s 在使用点（第 %d 行）之前**未赋值** ⇒ 运行时 UnboundLocalError，'
        '又会被 except Exception 吞掉：%s'
        % (fn, sorted(missing), use_line,
           {k: ('从未赋值' if v is None else '第 %d 行才赋值' % v)
            for k, v in missing.items()}))


def test_init_swanlab_defines_ws_and_accum_before_config():
    """`_init_swanlab` 的 config 块用 `_ws`/`_accum`，两者必须在它之前赋值。"""
    line = SRC[:SRC.index('config={')].count('\n') + 1
    _assigned_before('_init_swanlab', line, {'_ws', '_accum'})


def test_run_level_metrics_dict_uses_only_names_defined_earlier():
    """`swanlab_logger.log({...})` 里每个名字都必须在同一函数里**先赋值后使用**。

    这条直接钉死 `_accum_steps` 那次事故：它在 run 级上报里被引用，而赋值在
    29 行之后。`except Exception` 把 UnboundLocalError 变成一行 warning，
    SwanLab 静默少 8 个指标而训练照跑。
    """
    fn_node = _fn_node('main')
    calls = [n for n in ast.walk(fn_node)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'log' and n.args
             and isinstance(n.args[0], ast.Dict)]
    assert calls, 'main() 里找不到 swanlab_logger.log({...}) 调用'
    for call in calls:
        used = {n.id for n in ast.walk(call.args[0]) if isinstance(n, ast.Name)}
        # `args` 是 main() 的参数（不是局部赋值），交给「属性存在」那条检查
        _assigned_before('main', call.lineno, used - {'args'})


def test_config_expression_actually_evaluates():
    """把 config 表达式用**真实 dest 集合**的 Namespace 真 eval 一遍。

    静态检查挡得住「属性不存在」，挡不住别的：表达式里除 args 还引用了
    `KATAGO_SE_CFG` / `os` / `_ws` / `_accum` 等局部名，任何一个拼错或漏定义都会在
    `swanlab.init(...)` 那一步抛异常 —— 而那一步被 `except Exception` 吞掉，
    结果是**跟踪静默全丢、训练照跑**（2026-10-01 真发生过一次：`args.td`）。

    所以这里不满足于静态：把表达式抠出来实跑，并断言它产出预期内容。
    """
    import scripts.train_sft as tsf

    ns = _namespace_from_argparse()
    ns.data = 'x.npz'
    ns.board_size = 19
    ns.batch_size = 1000
    ns.lr = 0.0045
    ns.epochs = 1
    ns.use_ema = 1
    ns.scaler_init_scale = 1024.0
    ns.scaler_growth_interval = 100000

    body = SRC[SRC.index('config={'):]
    expr = body[body.index('{') + 1:body.index('},\n')]
    env = {'args': ns, 'KATAGO_SE_CFG': tsf.KATAGO_SE_CFG, 'os': os,
           '_ws': 2, '_accum': 2, 'NBT_TF_CFG': tsf.NBT_TF_CFG}
    cfg = eval('{' + expr + '}', env)   # noqa: S307 — 测试内显式求值

    assert isinstance(cfg, dict) and len(cfg) >= 25, \
        f'config 键数异常：{len(cfg) if isinstance(cfg, dict) else type(cfg)}'
    for k in ('arch/in_channels', 'arch/blocks', 'effective_batch',
              'grad_accum', 'world_size', 'lr', 'amp_dtype',
              'attn_query_chunk', 'ema_decay'):
        assert k in cfg, f'config 实跑后缺 {k}'
    assert cfg['effective_batch'] == 1000 * 2 * 2, cfg['effective_batch']
    for dead in ('backbone_channels', 'res_blocks', 'policy_layers'):
        assert dead not in cfg, f'归档 flag {dead} 混进了 config'


# --------------------------------------------------------------------------- #
# 3. run 级事实（config 给不出的那些）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('key', [
    'run/optimizer_steps_per_epoch', 'run/micro_batches_per_epoch',
    'run/total_optimizer_steps', 'run/warmup_steps', 'run/n_train',
    'run/n_eval', 'run/effective_batch',
])
def test_run_level_facts_are_logged_once(key):
    assert _has(MAIN, key), f'run 级 {key} 未上报'


def test_reported_values_are_floats_not_tensors():
    """上报值必须是 python float —— swanlab 收到 tensor 的行为不可依赖。"""
    # `float(...)` 出现在关键标量转换处
    for pat in ('float(l2_report)', 'float(opt_loss)', 'float(_gn)'):
        assert pat in SRC, f'缺少 {pat}'
    # `**_health_last` 里的值在构造时已 float()（逐项断言在下面）
    m = re.search(r"_health_last = \{(.*?)\n\s*\}", SRC, re.S)
    assert m, '找不到 _health_last 的构造'
    body = m.group(1)
    assert body.count('float(') >= 6, \
        f'_health_last 的 6 个指标都必须 float()，实得 {body.count("float(")} 处'


# --------------------------------------------------------------------------- #
# 4. V7 段（A/B/C 共用同一个 22 通道模型）的上报面，2026-10-04
# --------------------------------------------------------------------------- #
#
# 这次扩充的动机：V7 训练早已接通，但它的上报面**只覆盖 12 通路的形状** ——
# config 面板无条件记 12 通道的数字（于是每个 V7 run 的面板都写着
# `in_channels: 12` / `params_total: 9,112,005`，而它实际训练的是
# 22 通道 / 5,562,121），逐项 loss 只进 stdout 不进 SwanLab。
#
# 两类错误都属于「面板给了虚构值 / 该报的没报」，与本文件开头写的那两类同源。
@pytest.mark.parametrize('key', [
    'arch/v7', 'arch/global_channels', 'arch/seki_classes',
    'arch/futurepos_ch', 'arch/score_distr_bins', 'arch/policy_outputs',
    'data/games_npz',
])
def test_v7_config_keys_are_recorded(key):
    """V7 特有的结构维度必须进 config 面板（否则面板只剩 12 通道那套）。"""
    assert _has(INIT, key), f'config 面板缺 V7 键 {key}'


def test_config_arch_numbers_follow_the_v7_flag():
    """🔴 结构数字必须**按 `--v7` 分派**，不得无条件取 12 通道那张表。

    这不是洁癖：一个 V7 run 的面板写着 `in_channels: 12` 时，「对比两次 run」
    这件最常用的事会得到错误答案 —— 而面板恰恰是第一个看的东西。

    ⚠ 判据跑在 `ast.unparse` 的输出上（`INIT`），那是一行、无换行的形式 ⇒
      只能按「键名到下一个键名之间」切片，不能按源码的多行排版写正则。
    """
    for key in ('arch/in_channels', 'arch/channels', 'arch/params_total',
                'arch/num_heads', 'grad_checkpoint', 'arch/blocks'):
        k = "'%s'" % key
        assert k in INIT, f'config 里找不到 {key}'
        i = INIT.index(k)
        nxt = INIT.find("', '", i + len(k))
        val = INIT[i + len(k):nxt if nxt > 0 else i + 220]
        assert 'args.v7' in val, \
            f'{key} 未按 --v7 分派（V7 run 的面板会显示 12 通道的数字）：{val!r}'


def test_v7_config_references_nbt_tf_cfg_not_args():
    """V7 的结构取 `NBT_TF_CFG`（唯一真相源），**不引 args**。

    args 里那些是已归档的结构 flag，引它们等于把面板填成虚构值 ——
    与本文件 `test_config_arch_facts_come_from_katago_se_cfg` 同一纪律。
    """
    cfg = INIT[INIT.index('config={'):]
    assert "NBT_TF_CFG['in_channels']" in cfg
    assert "NBT_TF_CFG['trunk_channels']" in cfg
    assert "NBT_TF_CFG['ffn_hidden']" in cfg


def test_v7_config_avoids_the_archived_flag_names():
    """⚠ V7 的键名必须避开 `value_channels` / `policy_channels` 等归档名。

    `test_config_panel_has_no_archived_flags` 是按**引号内字面量**判泄漏的，
    所以写 `NBT_TF_CFG['value_channels']` 会让它误报（它要禁的是 args 上的
    归档开关，不是 V7 的真实结构键）。这条把「改名而不是绕开」钉住。
    """
    cfg = INIT[INIT.index('config={'):]
    for quoted in ("'value_channels'", '"value_channels"',
                   "'policy_channels'", '"policy_channels"'):
        assert quoted not in cfg, \
            f'config 里出现 {quoted} ⇒ 会误触 archived-flag 门禁，且含义与 args 旗同名'


def test_v7_loss_terms_are_uploaded_not_only_printed():
    """🔴 V7 的 12 项逐项 loss 必须进 SwanLab（此前**只**进 stdout）。

    为什么关键：段 1 有 9 项的权重是 0，而 `loss` 只是个总和 ——
    「四项在学」与「四项都塌了」在 `loss` 这一个数上长得一模一样。
    逐项曲线是早期唯一能抓住「第 5 项权重写反了」的手段，而它在图上不可见
    等于这个手段不存在。
    """
    assert "'loss_v7/%s' % k" in MAIN, '逐项 loss 未按 loss_v7/ 前缀上报'
    assert '**_v7_terms_swanlab' in MAIN, '逐项 dict 未进 swanlab.log 的字面量'


def test_v7_zero_weight_terms_are_still_uploaded():
    """⚠ 权重为 0 的项也**照报**（值恒 0）。

    图上看得见「这一项存在但没在学」，比「曲线里根本没有这一项」更容易
    区分「没接上」与「接上了但权重是 0」。键从**权重函数**取而不是另写一份
    名单 —— 段位表改了而名单没改，图上就会多一项/少一项而没人知道哪边错了。
    """
    assert 'v7_stage1_loss_weights().items()' in MAIN, \
        '零权重项的键未从权重函数派生（另写名单会与段位表漂移）'
    assert "_v7_terms_swanlab.setdefault('loss_v7/%s' % _k, 0.0)" in MAIN


def test_v7_terms_dict_is_initialised_outside_the_loop():
    """`_v7_terms_swanlab` 必须在循环外初始化。

    它无条件进 `swanlab.log({...})` 的字面量，而 12 通道路径**永不给它赋值**
    ⇒ 不初始化就是每步一次 UnboundLocalError，被 `except` 吞成一行 warning ⇒
    **SwanLab 静默丢掉整块指标而训练照跑**。与 2026-10-01 那次
    `_accum_steps` 同类，由 `test_run_level_metrics_dict_uses_only_names_defined_earlier`
    之外的第二道闸守住。
    """
    assert '_v7_terms_swanlab = {}' in MAIN, \
        '_v7_terms_swanlab 必须在循环外初始化为 {}（12 通道路径不赋值它）'


def test_terms_prefix_keeps_stdout_reading_bare_names():
    """⚠ stdout 的 `[step N v7]` 行按**裸 term 名**排版，两者不能共用一个 dict。

    共用会把 stdout 变成 `loss_v7/policy=5.88`，而文档与测试都按裸名读它。
    """
    assert "_v7_terms_last = {k: float(x) for k, x in _w.items()}" in MAIN, \
        'stdout 用的裸名 dict 形状被改动'
    assert "'%s=%.4f' % (k, v) for k, v in _v7_terms_last.items()" in MAIN


@pytest.mark.parametrize('key', [
    'run/params_actual', 'run/v7_source', 'run/rows_total', 'run/n_games',
    'run/policy_soft_weight',
])
def test_v7_run_level_keys_are_uploaded(key):
    """模型/数据源的真实形态：config 面板给不出（init 时模型还没建）。"""
    assert _has(MAIN, key), f'run 级缺 {key}'


def test_run_level_reports_actual_params_not_the_budget():
    """🔴 `run/params_actual` 必须是**实测**参数量。

    V7 的 `NBT_TF_CFG['params_total']` 是**预算值** 5,561,832，而实测建出来
    是 **5,562,121**（差 289，`tests/test_katago_v7_budget.py` 钉住后者）。
    只报预算值时，「预算表与实现漂了」这件事在图上完全看不出来。
    """
    assert 'run/params_actual' in MAIN and 'numel' in MAIN, \
        'run/params_actual 必须实测参数量（ast.unparse 会把生成器加括号）'


def test_v7_source_is_a_numeric_code_not_a_string():
    """🔴 `run/v7_source` 必须是**数值码**，不能是 `'board_level'` 这种名字。

    这条是被真报错打出来的（2026-10-04，用户云端 run）：
    `Unsupported scalar string value: 'board_level'`。swanlab 的 metric 通道
    只收 bool/int/float —— 字符串只有 `float()` 成功才收（见
    `swanlab/sdk/internal/run/transforms/scalar/__init__.py` 的
    `try: float(data) except ValueError: raise TypeError`）。
    人读的名字走 stdout + config 面板，曲线里存码。
    """
    i = MAIN.index("'run/v7_source'")
    val = MAIN[i + len("'run/v7_source'"):MAIN.index(',', i)]
    assert 'board_level' not in val and 'packed' not in val, \
        f'run/v7_source 仍在报字符串：{val!r}'
    assert 'hasattr(dataset' in val, '应按数据源分派出数值码'


@pytest.mark.parametrize('src_name', ['_init_swanlab', 'main'])
def test_no_string_reaches_the_metric_channel(src_name):
    """🔴 所有上报字典的值都必须是 bool/int/float —— 一个字符串都不许有。

    直接用 swanlab 自己的 `_transform_tensor_or_array` + 判定规则**实跑**，
    而不是自己重写一份类型白名单（重写的那份会与 SDK 漂）。
    swanlab 没装时跳过：这条钉的是「上报面合乎 SDK 契约」，SDK 不在时
    无从违反。
    """
    fn = _fn_src(src_name)
    # 抠出每个 `swanlab_logger.log({...})` / `swanlab.log({...})` 的字面量
    dicts = re.findall(r"\.log\(\{", fn)
    assert dicts or src_name == '_init_swanlab', '未找到 log 调用'
    if not dicts:
        return

    pytest.importorskip('swanlab', reason='swanlab 未装，无法按 SDK 契约校验')
    from swanlab.sdk.internal.run.transforms.scalar import (  # noqa: PLC0415
        _transform_tensor_or_array,
    )

    def _accepts(v):
        try:
            v = _transform_tensor_or_array(v)
        except TypeError:
            return False
        if isinstance(v, bool) or isinstance(v, (int, float)):
            return True
        if isinstance(v, str):
            try:
                float(v)
            except ValueError:
                return False
            return True
        return False

    # 真跑一次主流程里能取到的字面量值（不含运行时变量 —— 那些由
    # `_run_facts` 那条 `test_*_are_plain_floats` 风格的运行期检查负责）
    literals = {}
    for m in re.finditer(r'"([\w/]+)":\s*("(?:[^"\\]|\\.)*")', fn):
        literals[m.group(1)] = m.group(2)[1:-1]
    bad = []
    for k, v in literals.items():
        # config 面板的字符串是合法的（那是 config 不是 metric），只查 log 侧
        if k in ('soft_index', 'data', 'data/games_npz'):
            continue
        if not _accepts(v):
            bad.append((k, v))
    assert not bad, f'字符串值进了 metric 通道（SDK 会抛 TypeError）：{bad}'


def test_bool_metrics_are_accepted_by_the_sdk():
    """⚠ bool 是**合法** metric（SDK 先判 bool，因为 bool 是 int 的子类）。

    这条把「bool 也要转成 0/1」这类过度修正挡住 —— `eval_truncated` /
    `eval_used_ema` 直接报 bool 才是对的。
    """
    pytest.importorskip('swanlab', reason='swanlab 未装')
    from swanlab.sdk.internal.run.transforms.scalar import (  # noqa: PLC0415
        _transform_tensor_or_array,
    )
    for v in (True, False):
        assert _transform_tensor_or_array(v) in (True, False)


def test_loss_coefficients_are_uploaded_so_flat_lines_are_explicable():
    """逐项曲线上的平线（值恒 0）需要 `run/loss_coef/*` 才能解释。

    看到 `loss_v7/ownership` 恒 0 的人无法区分「没学」与「不学」——
    权重表是**代码常量**不是 CLI，config 面板里没有。
    """
    assert "run/loss_coef/%s' % _k" in MAIN


@pytest.mark.parametrize('key', ['label_entropy', 'soft_row_frac'])
def test_target_side_metrics_are_uploaded(key):
    """policy **目标**侧的口径必须上报。

    `policy_loss` / `policy_entropy` 说的都是**模型**（预测分布），而 A/B/C
    三段的目标根本不同：A 段 one-hot（熵 0）、B/C 段 KataGo 搜索分布（熵 > 0）。
    同一个 `policy_loss = 5.88` 在 A 段是「什么都没学到」，在 C 段却可能已经
    接近搜索分布 —— 只看预测侧会把两者读成同一件事。
    """
    assert _has(MAIN, key), f'目标侧指标 {key} 未上报'


def test_target_metrics_only_appear_when_soft_rows_exist():
    """⚠ 只在**真有软行**时产出，不给 `nan` 兜底。

    A 段不挂 `--soft-index` 时 `soft_mask` 恒 0 ⇒ 目标就是 one-hot、熵恒 0 ⇒
    「标签有多锐」这个问题不存在。报占位 `nan` 只会让图上多两条读不出来的线。
    """
    blk = SRC[SRC.index('_health_last = {'):]
    blk = blk[:blk.index("\n                try:")]
    assert 'if float(_sm.sum()) > 0:' in blk, \
        '目标侧指标未按「真有软行」设门槛（会在 A 段产出无意义的 nan）'


def test_label_entropy_is_computed_on_soft_rows_only():
    """🔴 熵只能在 `mask=1` 的行上算。

    `mask=0` 的行是 one-hot（熵 0），混进来会让均值被行数权重压平 ⇒
    曲线量到的是「batch 里有多少软行」而不是「标签有多锐」。
    """
    blk = SRC[SRC.index('if float(_sm.sum()) > 0:'):]
    blk = blk[:blk.index("_health_last['soft_row_frac']")]
    assert '][_msk]' in blk or '[_msk]' in blk, \
        '熵的取值未按 soft_mask 过滤行'


def test_soft_ce_weight_comes_from_the_loss_module():
    """`run/policy_soft_weight` 必须取自 `katago_v7_loss` 的常量，不另写一份。"""
    assert 'from src.networks.katago_v7_loss import POLICY_SOFT_WEIGHT' in SRC
    assert 'float(\n                    POLICY_SOFT_WEIGHT)' in MAIN or \
        'POLICY_SOFT_WEIGHT)' in MAIN