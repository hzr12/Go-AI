"""v21 的四种块类必须在 `src/networks/backbone.py` 里解析成 `nn.Module` 子类。

来历：这条迁自**已删除**的 `tests/test_fsdp1_conversion.py`
（`test_fsdp_wrap_policy_resolves_every_block_class`，见 `git show 777d79a:...`）。

为什么换轨之后它还留着
----------------------
旧版守的是上一代（FSDP1，已退役）包裹层的 **auto_wrap 粒度**：
`_fsdp_wrap_policy` 用 `getattr(_backbone, _name, None)` **逐名取类、取不到就静默跳过**
（不报错）。那条静态白名单路的失效模式很阴：今天把 `MambaLTI` 改个名，明天 Mamba
块就不再被单独切分片 —— 训练**不抛任何异常**，只是粒度退化成「整模型一个 unit」。

2026-10-01 换轨到 DDP 之后，**那个消费方没有了**：DDP 只在顶层加一层 wrapper，
根本不按块类做任何切分。但底层不变量仍然成立、且仍然值得钉 ——「这四个名字各自对应
backbone 里一个真实的 `nn.Module` 子类」。

迁移时做了一处**必要的判据替换**（不是削弱）
------------------------------------------
旧版从 `_fsdp_wrap_policy` 的名字元组里取待检名字。那个元组已随包裹层删除 ⇒ 判据
的来源没了。若改成把四个名字硬写在本测试里再断言它们存在，那会退化成「断言一个
常量等于它自己」—— 恒真、零信号。

所以判据改为从**真正消费这些名字的地方**取：`src/networks/alphanet.py` 的
`V21Backbone.__init__` 就是 v21 块栈的构建处，它 `from .backbone import` 这四个名字
并逐个实例化。于是本文件断言的是一条**真实的 import + 调用链**：改 backbone 里的类名
而没同步 alphanet（或反过来从 alphanet 里删掉某个 import），这里立刻红；把块栈换成
另一组类，也立刻红。

⇒ 换句话说：判据从「对着一个没人读的元组自证」换成了「对着真实的构建链」。
覆盖面没变窄（类名漂移仍被抓住），多出来的部分是「这些名字确实有人在用」。

⚠ 与包裹层有关的那一半（分片粒度）**没有**迁移过来 —— 它已随旧文件退役，
其残留价值由 `tests/test_dist_wrap.py` 以 DDP 的等价不变量接管
（包裹顺序、存档口径、EMA 键空间）。
"""
import ast
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BACKBONE_PATH = ROOT / 'src' / 'networks' / 'backbone.py'
ALPHANET_PATH = ROOT / 'src' / 'networks' / 'alphanet.py'

#: v21 的四种块。Mamba 那种块的类名历史上被路线图文档写成过 `Mamba2`，而代码里
#: 一直叫 `MambaLTI` —— 这正是这条测试要盯的那类「文档名 vs 代码名」漂移。
V21_BLOCK_CLASSES = ('ResBlock', 'MambaLTI', 'TransformerBlock', 'CrossAttnRes')


def _classes_defined_in(path):
    """`path` 里定义的全部顶层类名 → 类对象（真 import，不是文本搜）。"""
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
    """逐个解析四个名字，返回类对象列表；解析不到就给出**可读**的断言失败。

    单列一个 helper 是因为有两个测试都要解析：若让每个测试各自 `defined[n]`，
    名字漂移时后者会抛 `KeyError` 而不是可读的断言失败（本仓库的规矩是失败消息
    必须能直接告诉人哪儿坏了，见 `test_dist_wrap.py` 里「不要抛 ValueError」那条）。
    """
    defined = _classes_defined_in(BACKBONE_PATH)
    out = []
    for name in V21_BLOCK_CLASSES:
        assert name in defined, (
            'src/networks/backbone.py 里没有块类 %r —— 改过名了？'
            '（旧版包裹层会用 getattr 静默跳过取不到的类名 ⇒ 粒度退化而不报错；'
            '现在 alphanet.V21Backbone 直接 import 它 ⇒ 会在 import 时就炸，'
            '但 rename 只改一半仍会漏）' % name)
        out.append(defined[name])
    return out


def test_block_classes_resolve_to_nn_module_subclasses():
    """四个名字逐个在 `backbone` 里解析成 `nn.Module` 子类。

    判定是「解析得到**类**」而不是「解析得到某个东西」：非类型（函数、实例、常量）
    都必须露出来 —— 旧版 docstring 记过一个兜底失败模式
    「`classes` 全空时退化成 `[nn.Module]`」，那正是本条断言挡住的东西。
    """
    resolved = _resolve_block_classes()
    assert len(resolved) == len(V21_BLOCK_CLASSES), \
        '解析结果条数与清单不符：%d vs %d' % (len(resolved), len(V21_BLOCK_CLASSES))
    for name, cls in zip(V21_BLOCK_CLASSES, resolved):
        assert isinstance(cls, type), \
            'backbone.%s 解析到的不是类，是 %r' % (name, type(cls))
        assert issubclass(cls, torch.nn.Module), \
            'backbone.%s 不是 nn.Module 子类（它得是能被 Sequential/ModuleList 装下的块）' % name


def test_v21_block_stack_actually_uses_these_names():
    """这四个名字必须是 `alphanet` **从 backbone import 并实例化**的，且**只有**它们。

    没有这一条，上一条就退化成「断言一个常量等于它自己」—— 硬写的四元组对硬写的
    四元组，恒真。这里把判据接到真实的构建链上：
      · `from .backbone import (...)` 里必须有这四个名字；
      · `V21Backbone.__init__` 里必须有对它们的**调用**（裸名字 `Call` 节点）。

    ⚠ 判据取的是**恰好相等**（旧版那条也是 `set(names) == expected`，不是 `⊆`）：
    方向上多出来的一半交给「`__init__` 里没有第五个 backbone 类」这条 —— 只查
    `⊆` 的话，将来有人把 v21 块栈换成另一组类，本测试会安静地跟着清单一起改，
    而清单是本测试自己写的常量，跟着改就等于自证。做法是拿
    「`__init__` 里被实例化的裸名字」∩「backbone 定义的类名」，要求它正好等于本清单
    （内建 `int`/`range`/`super` 之类自然被交集排除掉）。

    同时钉住 `V21Backbone` 这个类名 —— 它是 v21 块栈的构建处，改了名就得同步这里。
    """
    tree = ast.parse(ALPHANET_PATH.read_text(encoding='utf-8'))

    imported = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.module == 'backbone'
                and node.level == 1):          # `from .backbone import …`
            imported.update(a.name for a in node.names)
    missing = sorted(set(V21_BLOCK_CLASSES) - imported)
    assert not missing, (
        'alphanet 不再 from .backbone import %s —— v21 块栈的块类来源被换掉了，'
        '本文件的解析断言也就失去了它要守护的那条链' % missing)

    builders = [n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == 'V21Backbone']
    assert len(builders) == 1, (
        'alphanet 里应恰好一个 V21Backbone（v21 块栈的构建处），实得 %d 个'
        % len(builders))
    inits = [n for n in builders[0].body
             if isinstance(n, ast.FunctionDef) and n.name == '__init__']
    assert len(inits) == 1, 'V21Backbone 应恰好一个 __init__，实得 %d 个' % len(inits)
    instantiated = {c.func.id for c in ast.walk(inits[0])
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}

    backbone_classes = set(_classes_defined_in(BACKBONE_PATH))
    used_backbone_classes = instantiated & backbone_classes
    assert used_backbone_classes == set(V21_BLOCK_CLASSES), (
        'V21Backbone.__init__ 里被实例化的 backbone 类是 %s，与本文件的清单 %s 不一致 —— '
        '要么块栈换了块类（清单要跟着改），要么有一种块已经从 v21 里消失了'
        % (sorted(used_backbone_classes), sorted(V21_BLOCK_CLASSES)))


def test_block_class_names_are_all_distinct():
    """四个名字互不相同（`V21_BLOCK_CLASSES` 是元组，重复项会静默合并）。

    旧版这条是对 `_fsdp_wrap_policy` 的名字元组断言 `len(set(names)) == len(names)`。
    迁移后元组没了，可迁移的是**判据的形状**而不是它的对象：若哪天有人把两个块
    「合并」成同一个类，这里会指出清单与代码实际提供的类不再是 1:1。
    """
    assert len(set(V21_BLOCK_CLASSES)) == len(V21_BLOCK_CLASSES), \
        'V21_BLOCK_CLASSES 里有重复项（这会让「四个名字 ↔ 四个类」的对应失真）'

    # 1:1 的实证：backbone 里这四个名字各自解析到**不同的**类对象。
    resolved = _resolve_block_classes()
    assert len({id(c) for c in resolved}) == len(V21_BLOCK_CLASSES), \
        '四个名字解析到了同一个类对象（清单与代码不再是 1:1）'
