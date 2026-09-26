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
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_sft import _build_param_groups  # noqa: E402
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
