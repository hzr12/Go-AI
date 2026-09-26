"""train_sft 的 AdamW 参数分组：value 头的一维参数必须落进 no-decay 组。

原实现（scripts/train_sft.py:1299-1311）**两侧用了不同的命名空间**：

    no_decay_params = set()
    for name, param in model.named_parameters():      # 全名 'value.fc.bias'
        if param.ndim == 1:
            no_decay_params.add(name)
    value_no_decay = [p for n, p in model.value.named_parameters()   # 相对名 'fc.bias'
                      if n in no_decay_params]

`'fc.bias'` 永远不在装着全名的集合里 → `value_no_decay` **恒空**，
value 头全部参数（含 BatchNorm/LayerNorm 权重与 bias）都吃了 weight decay，
与 SFT 常规做法相反。

修法（C12）：`no_decay_params` 收 **param 对象**而不是名字，比对用对象身份
（`nn.Parameter` 按 id 可哈希），这样两个命名空间不再需要对齐。
同时把 value 归属判定从子串 `'value' not in n` 收紧成前缀 `not n.startswith('value.')`。

下面这些断言**全部按 ndim 与前缀判定，不依赖任何具体参数名清单**——
v21 会继续改结构，靠名字锁死会立刻失效。
"""
import logging
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_sft import _build_param_groups, _load_optimizer_state  # noqa: E402
from src.networks.alphanet import AlphaGoNet  # noqa: E402

# 刻意取互不相同的数值：value 组与非 value 组靠 lr 就能区分，
# 四个组的 (lr, weight_decay) 签名两两不同 → 测试可以按签名定位组，
# 既不依赖组的下标（_locate_overflow 的注释明确警告过按下标判断很脆），
# 也不依赖参数名。
LR = 2e-3
VALUE_LR_MULT = 5.0
WEIGHT_DECAY = 1e-4


class _Args:
    """只提供 _build_param_groups 真正读到的三个属性。"""

    def __init__(self, lr=LR, value_lr_mult=VALUE_LR_MULT, weight_decay=WEIGHT_DECAY):
        self.lr = lr
        self.value_lr_mult = value_lr_mult
        self.weight_decay = weight_decay


def _tiny_model(arch='resnet'):
    """真实的小型 AlphaGoNet：通道压到 8~16，CPU 上构造是秒级的。

    用真模型而不是同构玩具，才能覆盖 value 头两种归一化
    （resnet → BatchNorm2d，convnext → LayerNorm2d）里的 1-D 参数。
    """
    return AlphaGoNet(
        in_channels=12,
        backbone_channels=16,
        backbone_res_blocks=1,
        attention_mode='none',
        num_attention_layers=0,
        num_heads=2,
        policy_channels=8,
        value_channels=8,
        action_size=10,
        value_res_blocks=1,
        policy_layers=1,
        arch=arch,
    )


def _ids(params):
    return {id(p) for p in params}


def _group_by_signature(groups, args):
    """按 (lr, weight_decay) 把四组索引出来，签名重复即报错。"""
    by_sig = {}
    for g in groups:
        by_sig.setdefault((g['lr'], g['weight_decay']), []).append(g)
    assert len(by_sig) == len(groups), (
        f'组的 (lr, weight_decay) 签名不唯一，无法按签名定位: {sorted(by_sig)}')
    v_lr = args.lr * args.value_lr_mult
    return {
        'other_decay': by_sig[(args.lr, args.weight_decay)],
        'other_no_decay': by_sig[(args.lr, 0.0)],
        'value_decay': by_sig[(v_lr, args.weight_decay)],
        'value_no_decay': by_sig[(v_lr, 0.0)],
    }


def _build(model, args=None):
    args = args or _Args()
    groups = _build_param_groups(model, args)
    return groups, _group_by_signature(groups, args)


# ---------------------------------------------------------------- 1

@pytest.mark.parametrize("arch", ["resnet", "convnext"])
def test_value_head_1d_params_get_no_weight_decay(arch):
    """value 头所有 ndim==1 的参数 → wd=0 且 lr=value_lr_mult*lr；ndim>1 → wd=weight_decay。

    修复前 value_no_decay 恒空、value_decay 吃掉全部 value 参数 → 红。
    """
    model, args = _tiny_model(arch), _Args()
    by_sig = _build(model, args)[1]

    value_1d = _ids(p for p in model.value.parameters() if p.ndim == 1)
    value_nd = _ids(p for p in model.value.parameters() if p.ndim > 1)
    assert value_1d, 'value 头没有任何一维参数，测试前提失效'
    assert value_nd, 'value 头没有任何多维参数，测试前提失效'

    got_no_decay = _ids(by_sig['value_no_decay'][0]['params'])
    got_decay = _ids(by_sig['value_decay'][0]['params'])
    assert got_no_decay == value_1d, (
        f'value no-decay 组与 value 头一维参数集合不符：'
        f'缺 {len(value_1d - got_no_decay)} 个，多 {len(got_no_decay - value_1d)} 个')
    assert got_decay == value_nd, (
        f'value decay 组与 value 头多维参数集合不符：'
        f'缺 {len(value_nd - got_decay)} 个，多 {len(got_decay - value_nd)} 个')


# ---------------------------------------------------------------- 2

def test_value_no_decay_group_is_non_empty():
    """显式断言 value no-decay 组非空，且恰好等于 value 头的一维参数集合。

    只断言「非空」会被「靠别的东西填满了它」的假绿骗过去，
    所以这里同时做集合相等。
    """
    model, args = _tiny_model(), _Args()
    groups = _build_param_groups(model, args)
    by_sig = _group_by_signature(groups, args)

    value_no_decay = by_sig['value_no_decay'][0]['params']
    assert len(value_no_decay) > 0, (
        'value no-decay 组是空的 —— value 头的一维参数（BN/LN 权重与 bias、'
        'fc bias）本应 weight_decay=0，现在全被 value decay 组吃掉了')
    expected = [p for p in model.value.parameters() if p.ndim == 1]
    assert _ids(value_no_decay) == _ids(expected), (
        'value no-decay 组里混进了不该在的组外参数，或漏掉了 value 头的一维参数')


# ---------------------------------------------------------------- 3

@pytest.mark.parametrize("arch", ["resnet", "convnext"])
def test_every_parameter_appears_exactly_once(arch):
    """四组按身份的并集 == model.parameters()，且没有任何参数落进两组。

    重复 = 双重更新（AdamW 动量与衰减各来一遍）；遗漏 = 那个参数完全不更新。
    """
    model, args = _tiny_model(arch), _Args()
    groups = _build_param_groups(model, args)

    flat = [p for g in groups for p in g['params']]
    seen, dup = set(), []
    for p in flat:
        if id(p) in seen:
            dup.append(p)
        seen.add(id(p))
    all_ids = _ids(model.parameters())

    missing = all_ids - seen
    extra = seen - all_ids
    assert not dup, f'{len(dup)} 个参数出现在多个组里（双重更新）'
    assert not missing, f'{len(missing)} 个模型参数没落进任何组（完全不更新）'
    assert not extra, f'{len(extra)} 个组内参数不属于该模型'
    assert len(flat) == len(all_ids), '扁平化后的参数个数与模型参数个数不等'


# ---------------------------------------------------------------- 4

def test_non_value_groups_keep_plain_lr_and_wd():
    """两个非 value 组 lr == args.lr；decay 组 wd == args.weight_decay，no-decay 组 == 0.0。"""
    model, args = _tiny_model(), _Args()
    groups = _build_param_groups(model, args)
    by_sig = _group_by_signature(groups, args)

    for key in ('other_decay', 'other_no_decay', 'value_decay', 'value_no_decay'):
        (g,) = by_sig[key]
        assert g['lr'] in (args.lr, args.lr * args.value_lr_mult), f'{key}: lr 取值不在预期内'
    for key in ('other_decay', 'other_no_decay'):
        (g,) = by_sig[key]
        assert g['lr'] == args.lr, f'{key}: lr 应保持基准 args.lr，实际 {g["lr"]}'
    assert by_sig['other_decay'][0]['weight_decay'] == args.weight_decay
    assert by_sig['other_no_decay'][0]['weight_decay'] == 0.0


# ---------------------------------------------------------------- 5

@pytest.mark.parametrize("where", ["backbone", "value"])
def test_no_decay_criterion_is_ndim_not_name(where):
    """名字里没有 bias / Norm 字样的一维参数也必须进 no-decay 组。

    防的是「有人把判据改回名字子串」（selfplay 侧现在就是这个写法）。
    """
    model, args = _tiny_model(), _Args()
    parent = model.value if where == "value" else model.backbone
    parent.w1 = nn.Parameter(torch.zeros(3))
    p = parent.w1
    assert p.ndim == 1
    # 按身份反查全名（Parameter 的 __eq__ 是逐元素的，不能拿它当 dict 键）
    full_name = {id(q): n for n, q in model.named_parameters()}[id(p)]
    assert 'bias' not in full_name and 'Norm' not in full_name and 'norm' not in full_name, \
        f'测试前提失效：{full_name} 名字里含判据字样'

    by_sig = _build(model, args)[1]
    key = 'value_no_decay' if where == "value" else 'other_no_decay'
    assert id(p) in _ids(by_sig[key][0]['params']), \
        f'{full_name} 是一维参数却没落进 {key}（判据被改成了名字匹配？）'


# ---------------------------------------------------------------- 6

def test_value_membership_is_prefix_not_substring():
    """名字含 'value' 但不属于 value 头的参数（backbone.value_proj）必须进非 value 组。

    修复前用子串判定 `'value' not in n`：这类参数会被两个 other_* 组同时排除，
    从四组里**整体消失**（永不更新）。前缀判定 `'value.'` 才正确。
    """
    model, args = _tiny_model(), _Args()
    model.backbone.value_proj = nn.Linear(4, 4)
    w, b = model.backbone.value_proj.weight, model.backbone.value_proj.bias

    by_sig = _build(model, args)[1]
    assert id(w) in _ids(by_sig['other_decay'][0]['params']), \
        'backbone.value_proj.weight 被误判成 value 头参数（子串判定回潮了？）'
    assert id(b) in _ids(by_sig['other_no_decay'][0]['params']), \
        'backbone.value_proj.bias 被误判成 value 头参数（子串判定回潮了？）'
    for key in ('value_decay', 'value_no_decay'):
        assert id(w) not in _ids(by_sig[key][0]['params'])
        assert id(b) not in _ids(by_sig[key][0]['params'])


# ---------------------------------------------------------------- 7

def test_groups_are_consumed_by_optimizer():
    """四组喂给真实 torch.optim.AdamW 后 param_groups 逐组一致，且 main() 真的调它。"""
    import ast
    model, args = _tiny_model(), _Args()
    groups = _build_param_groups(model, args)

    optimizer = torch.optim.AdamW(groups)
    assert len(optimizer.param_groups) == len(groups)
    for g, og in zip(optimizer.param_groups, groups):
        assert g['lr'] == og['lr']
        assert g['weight_decay'] == og['weight_decay']
        assert _ids(g['params']) == _ids(og['params'])

    # 主流程确实用的是这个函数（而不是另处又抄了一份分组逻辑）
    with open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    called = {c.func.id for c in ast.walk(main)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert '_build_param_groups' in called, \
        'main() 没有调用 _build_param_groups —— 参数分组被留在了 main() 里，测试等于没在测'


# ---------------------------------------------------------------- 8

def test_main_parser_supplies_exactly_the_attributes_the_builder_reads():
    """_build_param_groups 读到的 args.X 必须与 main() argparse 的旗名一一对应。

    两侧对不上就是「测试用 _Args 造了个假前提、真实 main() 却在别处炸」的经典坑：
    测试用的 _Args 只有三个属性，真实 args 有几十个。
    """
    import ast
    with open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())

    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == '_build_param_groups')
    read = sorted({n.attr for n in ast.walk(fn)
                   if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                   and n.value.id == 'args'})

    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    flags = {a.args[0].value.lstrip('-').replace('-', '_')
             for a in ast.walk(main)
             if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute)
             and a.func.attr == 'add_argument' and a.args
             and isinstance(a.args[0], ast.Constant)}

    assert read == ['lr', 'value_lr_mult', 'weight_decay'], \
        f'_build_param_groups 读到的 args 属性变成了 {read}（本测试的 _Args 需要同步）'
    missing = [a for a in read if a not in flags]
    assert not missing, f'argparse 没有提供 {missing}，真实 main() 会在分组处 AttributeError'


# ---------------------------------------------------------------- 9
#
# 以下三个用例针对 resume 路径的**运维断裂**：P2.4 把 value 头一维参数搬进
# no-decay 组，四组大小从 [7, 13, 11, 0] 变成 [7, 13, 4, 7]。改动本身是本任务
# 被要求的修复，但 optimizer 的 state_dict 把每组大小写进了 param_groups，
# torch.optim.Optimizer.load_state_dict 逐组比对大小，不一致就抛
#   ValueError: loaded state dict contains a parameter group that doesn't match
#   the size of optimizer's group
# → **任何在 C12 修复（提交 95652ae）之前存下的 checkpoint 都再也续不上**
# （当时 main() 直接调 load_state_dict，硬崩）。
# 修法是「只对这一类失败告警并跳过优化器状态，其余照旧恢复」，而不是静默吞异常。


def _optimizer_over(model, args):
    groups = _build_param_groups(model, args)
    return torch.optim.AdamW(groups), groups


# torch.optim.Optimizer.load_state_dict 对组布局不兼容只有两条消息，两条都点名
# 'parameter group'（实测 torch 2.12.0+cpu）：
#   ValueError("loaded state dict has a different number of parameter groups")
#   ValueError("loaded state dict contains a parameter group that doesn't match "
#              "the size of optimizer's group")
# 前者组数不同、后者组大小不同，属于同一类断裂。
_TORCH_GROUP_SIZE_MSG = ("loaded state dict contains a parameter group that doesn't "
                         "match the size of optimizer's group")


@pytest.mark.parametrize("exc_type", [None, RuntimeError],
                         ids=["real-torch", "as-runtimeerror"])
def test_incompatible_optimizer_state_is_skipped_with_warning(caplog, monkeypatch, exc_type):
    """组大小对不上时：返回 False、不抛异常、告警里带新旧组大小与「重置」提示。

    同时断言优化器状态仍是空的：torch 的组大小检查发生在 `__setstate__` **之前**
    （先 deepcopy 再比对，命中就 raise），所以「跳过」不会留下半途写入的脏状态。

    `exc_type=None` 走**真实** torch 抛错路径；`RuntimeError` 那个 case 是给
    except 元组里的 RuntimeError 分支上锁的窄特征化用例 —— 只测真实路径的话，
    把 except 收窄成 `except ValueError` 全绿，而 fused / 子类 optimizer 报
    RuntimeError 时就会重新变成硬崩。
    """
    model, args = _tiny_model(), _Args()
    optimizer, _ = _optimizer_over(model, args)
    logger = logging.getLogger('train')

    # 造一个「组大小与当前不符」的旧 state dict：从最大的一组里砍掉一个参数。
    # 刻意不用 [7, 13, 11, 0] 这种写死的旧布局——组大小是会继续变的意图，
    # 锁死它就变成「改常量才红」的检测器；这里要测的是行为：**不匹配就跳过**。
    saved = optimizer.state_dict()
    widest = max(range(len(saved['param_groups'])),
                 key=lambda i: len(saved['param_groups'][i]['params']))
    assert len(saved['param_groups'][widest]['params']) > 1, \
        '测试前提失效：最大的一组只有 1 个参数，砍不掉'
    saved['param_groups'][widest]['params'] = saved['param_groups'][widest]['params'][:-1]

    old_sizes = [len(g['params']) for g in saved['param_groups']]
    new_sizes = [len(g['params']) for g in optimizer.param_groups]
    assert old_sizes != new_sizes, \
        '测试前提失效：构造出的 state dict 布局与当前分组一致'

    if exc_type is not None:
        def _boom(_state, _t=exc_type):
            raise _t(_TORCH_GROUP_SIZE_MSG)
        monkeypatch.setattr(optimizer, 'load_state_dict', _boom)

    with caplog.at_level(logging.WARNING, logger='train'):
        assert _load_optimizer_state(optimizer, saved, logger) is False

    assert not optimizer.state, \
        '跳过后优化器里却出现了状态——加载并非在写入之前就失败（脏状态）'

    text = caplog.text
    assert '组大小' in text, f'告警没有点明是组大小的问题：{text!r}'
    assert '重置' in text, f'告警没有说明优化器状态已被重置：{text!r}'
    assert str(old_sizes) in text, f'告警里找不到旧组大小 {old_sizes}：{text!r}'
    assert str(new_sizes) in text, f'告警里找不到新组大小 {new_sizes}：{text!r}'
    assert 'Adam' in text, f'告警没有说明后果（Adam 动量从头开始）：{text!r}'


def test_compatible_optimizer_state_loads_normally(caplog):
    """布局一致时正常加载：返回 True、Adam 动量真的落到优化器里、且不打那条告警。

    「返回 True」单独不够——一个什么都不做直接 return True 的实现也能骗过它，
    所以这里逐个参数比对 exp_avg 的实际取值。
    """
    model, args = _tiny_model(), _Args()
    source, groups = _optimizer_over(model, args)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    source.step()
    saved = source.state_dict()
    want = {id(p): s['exp_avg'].clone() for p, s in source.state.items() if 'exp_avg' in s}
    assert want, '测试前提失效：step() 之后优化器里没有任何 Adam 动量'

    # 同样按当前分组建、但状态为空的优化器
    fresh = torch.optim.AdamW(groups)
    assert not fresh.state, '测试前提失效：fresh 优化器本该没有状态'

    with caplog.at_level(logging.WARNING, logger='train'):
        assert _load_optimizer_state(fresh, saved, logging.getLogger('train')) is True

    assert '组大小' not in caplog.text, \
        f'布局完全一致却打了组大小告警：{caplog.text!r}'

    got = {id(p): s['exp_avg'] for p, s in fresh.state.items() if 'exp_avg' in s}
    # want/got 已经以 id(p) 为键，直接比键集；套 _ids() 会变成「取 int 的 id」
    assert set(got) == set(want), '加载后优化器里的参数集合与 checkpoint 对不上'
    for pid, exp_avg in got.items():
        assert torch.equal(exp_avg, want[pid]), \
            f'参数 {pid} 的 Adam 动量没被真正加载进来'


@pytest.mark.parametrize("exc_type,msg", [
    (RuntimeError, 'boom'),
    (ValueError, 'unrelated failure'),
])
def test_unrelated_load_error_still_propagates(monkeypatch, exc_type, msg):
    """与分组无关的加载失败必须原样抛出。

    这是「不许写 `except Exception: pass`」的闸门：宽泛的 except 会把**所有**
    resume 失败都变成「优化器状态安静地丢了」，训完一整轮才发现动量没恢复。
    ValueError 与 RuntimeError 都要覆盖——只测 RuntimeError 的话，
    「只 catch RuntimeError」的写法照样能过，但它同样是错分支放行。
    """
    model, args = _tiny_model(), _Args()
    optimizer, _ = _optimizer_over(model, args)

    def _boom(_state):
        raise exc_type(msg)

    monkeypatch.setattr(optimizer, 'load_state_dict', _boom)
    with pytest.raises(exc_type, match=msg):
        _load_optimizer_state(optimizer, optimizer.state_dict(),
                              logging.getLogger('train'))
