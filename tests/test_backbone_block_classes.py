"""现役主干的块类必须在 `src/networks/backbone.py` 里解析成 `nn.Module` 子类。

来历：这条迁自**已删除**的 `tests/test_fsdp1_conversion.py`
（`test_fsdp_wrap_policy_resolves_every_block_class`，见 `git show 777d79a:...`）。

为什么换轨之后它还留着
----------------------
旧版守的是上一代（FSDP1，已退役）包裹层的 **auto_wrap 粒度**：
`_fsdp_wrap_policy` 用 `getattr(_backbone, _name, None)` **逐名取类、取不到就静默跳过**
（不报错）。那条静态白名单路的失效模式很阴：今天把 `MambaLTI` 改个名，明天 Mamba
块就不再被单独切分片 —— 训练**不抛任何异常**，只是粒度退化成「整模型一个 unit」。

2026-10-01 换轨到 DDP 之后，**那个消费方没有了**：DDP 只在顶层加一层 wrapper，
根本不按块类做任何切分。但底层不变量仍然成立、且仍然值得钉 ——「这些名字各自对应
backbone 命名空间里一个真实的 `nn.Module` 子类」。

2026-10-01 v21 硬删除之后的**改版**（判据的形状没变，对象换轨）
--------------------------------------------------------------
`V21Backbone` / `MambaLTI` / `CrossAttnRes` 随 v21 一起从 `src/networks/` 删除 ⇒
本文件原来钉的那条链（alphanet.V21Backbone import 并实例化四个 v21 块）**被测对象
没了**，不能靠改期望值留绿。于是按同一套判据换轨到现役构建链：

  · 被解析的清单 = `SharedBackbone` 的两个 builder（`_build_blocks` /
    `_build_segmented_blocks`）实际实例化的块类 —— 即现役块栈的**全部**成员；
  · 「真的在用」的判据仍然取自**真实构建链**，不取自本文件自己的常量：
    `alphanet` 必须 `from .backbone import SharedBackbone` 并在 `AlphaGoNet.__init__`
    里实例化它；两个 builder 里被实例化的裸名字 ∩ backbone 的类名必须**恰好**等于
    清单（同旧版：是 `==` 不是 `⊆`，多出来的一半同样会被抓住）。

`SEBottleneck` 定义在 `src/networks/se_bottleneck.py`，由 `backbone.py` **绝对**
import 进自己的模块命名空间 —— 本文件的 `_classes_defined_in` 看的就是那个命名空间，
这正是 `backbone.py` 顶部特意写绝对导入的原因（见那里的注释：相对导入会让本文件的
standalone 加载炸掉）。

 `MHSA` / `TransformerBlock` 也在 backbone 里，但**没有现役调用方**、不进任何块栈
（backbone.py 末节注释），故刻意不在清单里 —— 清单只收「真的会被建出来」的块。
"""
import ast
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BACKBONE_PATH = ROOT / 'src' / 'networks' / 'backbone.py'
ALPHANET_PATH = ROOT / 'src' / 'networks' / 'alphanet.py'

#: 现役块栈：`SharedBackbone._build_blocks` / `_build_segmented_blocks` 实际实例化的
#: 全部块类。`SEBottleneck`（KataGo SE 路线的主块）定义在 se_bottleneck.py、经绝对
#: import 进 backbone 的命名空间；其余三个定义在 backbone.py 本体。
BLOCK_CLASSES = ('ResBlock', 'ConvNeXtBlock', 'AttentionResBlock', 'SEBottleneck')


def _classes_defined_in(path):
    """`path` 的模块命名空间里全部类名 → 类对象（真 import，不是文本搜）。

    含**从别处 import 进来**的类（如 backbone 里的 `SEBottleneck`）—— 判据是
    「在 backbone 这个命名空间里解析得到」，与它定义在哪个文件无关。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        '_bb_probe_' + path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return {name: getattr(mod, name) for name in dir(mod)
            if isinstance(getattr(mod, name), type)}


def _resolve_block_classes():
    """逐个解析清单里的名字，返回类对象列表；解析不到就给出**可读**的断言失败。

    单列一个 helper 是因为有两个测试都要解析：若让每个测试各自 `defined[n]`，
    名字漂移时后者会抛 `KeyError` 而不是可读的断言失败（本仓库的规矩是失败消息
    必须能直接告诉人哪儿坏了，见 `test_dist_wrap.py` 里「不要抛 ValueError」那条）。
    """
    defined = _classes_defined_in(BACKBONE_PATH)
    out = []
    for name in BLOCK_CLASSES:
        assert name in defined, (
            'src/networks/backbone.py 的命名空间里没有块类 %r —— 改过名了或 import 断了？'
            '（现役块栈由 SharedBackbone 的 builder 按名字实例化，漏了会在建网时才炸；'
            '这里提前到 import 期把名字漂移抓住）' % name)
        out.append(defined[name])
    return out


def test_block_classes_resolve_to_nn_module_subclasses():
    """清单里每个名字逐个在 `backbone` 的命名空间里解析成 `nn.Module` 子类。

    判定是「解析得到**类**」而不是「解析得到某个东西」：非类型（函数、实例、常量）
    都必须露出来 —— 旧版 docstring 记过一个兜底失败模式
    「`classes` 全空时退化成 `[nn.Module]`」，那正是本条断言挡住的东西。
    """
    resolved = _resolve_block_classes()
    assert len(resolved) == len(BLOCK_CLASSES), \
        '解析结果条数与清单不符：%d vs %d' % (len(resolved), len(BLOCK_CLASSES))
    for name, cls in zip(BLOCK_CLASSES, resolved):
        assert isinstance(cls, type), \
            'backbone.%s 解析到的不是类，是 %r' % (name, type(cls))
        assert issubclass(cls, torch.nn.Module), \
            'backbone.%s 不是 nn.Module 子类（它得是能被 Sequential/ModuleList 装下的块）' % name


def _instantiated_bare_names(fn_node):
    """`fn_node` 体内被**裸名字调用**的全部标识符（`Foo(...)`，不含 `a.b(...)`）。"""
    return {c.func.id for c in ast.walk(fn_node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}


def test_block_stack_actually_uses_these_names():
    """清单里的名字必须在**真实构建链**上被实例化，且 builder 用的恰好是它们。

    没有这一条，上一条就退化成「断言一个常量等于它自己」—— 硬写的元组对硬写的
    元组，恒真。这里把判据接到真实的构建链上，两段：

      1. `alphanet` 必须 `from .backbone import SharedBackbone`，且
         `AlphaGoNet.__init__` 里被实例化的裸名字 ∩ backbone 的类名**恰好**是
         `{SharedBackbone}` —— 主干容器的来源被换掉（或换成别的容器）时这里立刻红；
      2. `SharedBackbone` 的两个 builder（`_build_blocks` / `_build_segmented_blocks`，
         块栈的构建处）里被实例化的裸名字 ∩ backbone 的类名**恰好**等于
         `BLOCK_CLASSES` —— 同旧版是 `==` 不是 `⊆`：只查 `⊆` 的话，将来有人把
         块栈换成另一组类，本测试会安静地跟着清单一起改，而清单是本测试自己写的
         常量，跟着改就等于自证。

    同时钉住 `AlphaGoNet` / `SharedBackbone` 这两个类名 —— 它们是现役建网链的
    两端，改了名就得同步这里。
    """
    tree = ast.parse(ALPHANET_PATH.read_text(encoding='utf-8'))

    imported = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.module == 'backbone'
                and node.level == 1):          # `from .backbone import …`
            imported.update(a.name for a in node.names)
    assert 'SharedBackbone' in imported, (
        'alphanet 不再 from .backbone import SharedBackbone —— 现役主干容器的'
        '块类来源被换掉了，本文件的解析断言也就失去了它要守护的那条链')

    builders = [n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == 'AlphaGoNet']
    assert len(builders) == 1, (
        'alphanet 里应恰好一个 AlphaGoNet（现役建网的入口），实得 %d 个'
        % len(builders))
    inits = [n for n in builders[0].body
             if isinstance(n, ast.FunctionDef) and n.name == '__init__']
    assert len(inits) == 1, 'AlphaGoNet 应恰好一个 __init__，实得 %d 个' % len(inits)

    backbone_classes = set(_classes_defined_in(BACKBONE_PATH))
    used_container = _instantiated_bare_names(inits[0]) & backbone_classes
    assert used_container == {'SharedBackbone'}, (
        'AlphaGoNet.__init__ 里被实例化的 backbone 类是 %s，应恰好是 '
        "{'SharedBackbone'} —— 主干容器被换掉或换了来源，本文件的清单要跟着换轨"
        % sorted(used_container))

    # 块栈的构建处：SharedBackbone 的两个 builder。
    btree = ast.parse(BACKBONE_PATH.read_text(encoding='utf-8'))
    containers = [n for n in btree.body
                  if isinstance(n, ast.ClassDef) and n.name == 'SharedBackbone']
    assert len(containers) == 1, (
        'backbone 里应恰好一个 SharedBackbone（块栈的构建处），实得 %d 个'
        % len(containers))
    fns = [n for n in containers[0].body
           if isinstance(n, ast.FunctionDef)
           and n.name in ('_build_blocks', '_build_segmented_blocks')]
    assert len(fns) == 2, (
        'SharedBackbone 应恰好两个块栈 builder（_build_blocks / '
        '_build_segmented_blocks），实得 %d 个：%s'
        % (len(fns), [n.name for n in fns]))

    stack_names = set()
    for fn in fns:
        stack_names |= _instantiated_bare_names(fn)
    used_blocks = stack_names & backbone_classes
    assert used_blocks == set(BLOCK_CLASSES), (
        'SharedBackbone 的 builder 里被实例化的 backbone 类是 %s，与本文件的清单 %s '
        '不一致 —— 要么块栈换了块类（清单要跟着改），要么有一种块已经从现役建网里消失了'
        % (sorted(used_blocks), sorted(BLOCK_CLASSES)))


def test_block_class_names_are_all_distinct():
    """清单里的名字互不相同（`BLOCK_CLASSES` 是元组，重复项会静默合并）。

    旧版这条是对 `_fsdp_wrap_policy` 的名字元组断言 `len(set(names)) == len(names)`。
    迁移后元组没了，可迁移的是**判据的形状**而不是它的对象：若哪天有人把两个块
    「合并」成同一个类，这里会指出清单与代码实际提供的类不再是 1:1。
    """
    assert len(set(BLOCK_CLASSES)) == len(BLOCK_CLASSES), \
        'BLOCK_CLASSES 里有重复项（这会让「名字 ↔ 类」的对应失真）'

    # 1:1 的实证：backbone 命名空间里这些名字各自解析到**不同的**类对象。
    resolved = _resolve_block_classes()
    assert len({id(c) for c in resolved}) == len(BLOCK_CLASSES), \
        '清单里的名字解析到了同一个类对象（清单与代码不再是 1:1）'
