"""NPU TorchAir 图编译：**Linear-only**（D2），受控实验，默认关闭，失败必须回退 eager。

背景
----
`train_sft.py` 此前对 NPU 硬禁用 `torch.compile`（inductor 在 NPU 不可用），
全程 eager。实测 4 卡 910A 上 NPU 报 `Aicore Usage Rate` 85~100% 而实际只有
659 samples/s/卡——AICore 一直"忙"但产出极低，典型的 eager 小算子形态。
TorchAir 是昇腾官方的图编译后端，是唯一可能带来量级提升的路径。

环境（本项目实测）：torch 2.1.0 / torch_npu 2.1.0.post3 / CANN 8.0.RC1 /
aarch64。属 2023 年代组合，功能成熟度存疑，故定位为**受控实验**。

D2：为什么是 Linear-only
-----------------------
本提交之前走的是整模型 `torch.compile(model, backend=torchair_backend, dynamic=False)`。
问题有两层：

1. **显存**：整张网络一次性进图，workspace 与图缓冲要按全模型算，而 4 卡 910A
   常驻已到 26.7~31.1GB / 32GB。实测收益却只落在 Linear 那一小段：子模块图
   `chunk + Linear2d + silu`，invoke=1 / frames=2 / break=0，graph=10.779ms
   vs eager=14.453ms ≈ **1.34x**。为 1.34x 去顶爆显存不划算。
2. **可回滚性**：整模型 compile 返回一个 `OptimizedModule`，失败时只能靠
   `getattr(model, '_orig_mod', model)` 剥顶层包装——一条**只有一个**名字的、
   无处核对的假设（剥没剥成功只能看日志）。改成逐个 `nn.Linear` 就地替换后，
   每次替换都记 `(parent, attr_name, original)` 三元组，回滚是**逐条写回**，
   数目对得上就是还原干净了。

代价：图数量从 1 张变成「每个 Linear 1 张」。本模型 Linear 数量是两位数、
单图很小，可接受；换来的是没开开关时代码路径与原来完全一致。

必须守住的安全性质
----------------
1. **失败必须回退 eager**，绝不能让整个训练崩掉；回滚必须**逐条**、且可核对。
2. **导入顺序强制** `torch_npu` 先于 `torchair`，否则图模式**静默降级为
   eager** 而不报错——那样就白开了，还白付了预热代价。
3. **不得传 `mode` / `options`**：昇腾 NPU 不支持；且 `reduce-overhead` 正是
   A100 上 OOM 的元凶（CUDA Graphs 私有内存池不归还），NPU 上同样要避开。
4. **显存是真实风险**：本项目 4 卡 910A 常驻已到 26.7~31.1GB / 32GB，
   图模式额外的 workspace/图缓冲极易再炸。默认关闭即为此。
5. **不得二次包装**： `torch.compile` 返回的 `OptimizedModule` 本身也是
   `nn.Module`，而 `model.modules()` 是惰性生成器。边遍历边替换会让它走进
   新包进去的 `_orig_mod`，那个 Linear 立刻又满足 `isinstance(..., nn.Linear)`
   → 无限套娃（实测 `RecursionError: maximum recursion depth exceeded`，
   995 层重复）。故必须**先收集再替换**（见
   `test_collect_before_replace_never_double_wraps`）。
6. **键名会变**：`state_dict()` 里被包的 Linear 多出 `_orig_mod.` 段，且落在
   **路径中段**（`backbone.qkv._orig_mod.weight`）而不是开头——只有整模型
   compile 才是开头。存档 / 续训 / EMA 三处都必须对此对齐（见
   `test_saved_keys_are_stripped_at_every_level` 等）。

分两类：
  - **结构锁**：读 `train_sft.py` 源码，正则/AST 断言形态（无需 NPU、秒级）。
  - **行为用例**：直接调 `t._compile_linear_submodules` / `t._rollback_linear_submodules`
    与 `save_model` / `EMA`，用一个假 torchair backend 走完「替换 → 预热 →
    回滚」，断言模型没被污染。`torch.compile` 被桩掉（真跑一次 dynamo 要 3~4s，
    而这些用例关心的不是 dynamo），只有 `test_real_compile_artifacts_match_stub`
    跑真的。
"""

import ast
import copy
import os
import re
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.train_sft as t  # noqa: E402

SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()
CODE = '\n'.join(l for l in SRC.splitlines() if not l.lstrip().startswith('#'))
TREE = ast.parse(SRC)
_LINES = SRC.splitlines()
_FULL_LINE_COMMENTS = {i for i, l in enumerate(_LINES, 1) if l.lstrip().startswith('#')}


def _main_fn():
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            return node
    raise AssertionError('train_sft.py 里没有找到 main()')


def _code_span(lo, hi):
    """SRC 坐标的半开区间 `[lo, hi)`，剥掉整行注释但**保留行号对齐**。

    剥注释是为了不让「解释这件事的注释」冒充断言对象；保留行号对齐是为了让这里
    返回的行号仍能与 AST 的 `lineno` 直接比较 —— `CODE`（全文件去注释）做不到这点，
    混用两套坐标会得出看似合理、实际错位的区间边界。
    """
    return '\n'.join('' if i in _FULL_LINE_COMMENTS else _LINES[i - 1]
                     for i in range(lo, hi))


def _dist_wrap_lineno():
    """分布式包裹点的源码行号：`main()` 体内 `DistributedDataParallel(...)` 的调用节点。

     **必须走 AST 的 `Call` 节点，不能在源码文本里搜 `DistributedDataParallel(`**。
    那个字符串在 `train_sft.py` 里有四处，其中三处不是调用点：
      · L27   `from torch.nn.parallel import DistributedDataParallel`（在 `main()` 之前
              ⇒ 拿它当下界会让「包裹点在编译分支之后」这条断言**恒假**）；
      · L263 / L2546  包裹点上方解释换轨理由的 docstring / 注释（⇒ 「出现过就算」
              这类判据**恒真**）。
    限定在 `main()` 子树 + `isinstance(n.func, ast.Name)` 才能把这两类都排除掉；
    这与 `tests/test_init_weight_sync.py` 的 `_main_body_calls`、
    `tests/test_dist_wrap.py` 的 `_ddp_construct_calls` 是同一个坑、同一种解法。
    """
    main = _main_fn()
    hits = [n.lineno for n in ast.walk(main)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == 'DistributedDataParallel']
    assert len(hits) == 1, (
        'main() 里 DistributedDataParallel 的**调用**节点应恰好 1 个（实得 %d：%s）—— '
        '0 个说明分布式包裹被删了，多个说明有一处在错误的位置被包裹。'
        ' 别改成在源码文本里搜这个名字，见本函数 docstring。' % (len(hits), hits))
    return hits[0]


def _cuda_compile_branch_lineno():
    """`elif args.compile == 1:`（CUDA `--compile` 分支）的起始行，源码坐标。

    按 AST 取 `if _npu_graph:` 的 **orelse**，而不是文本搜 `elif args.compile == 1:`：
    `main()` 里至少还有两处 `args.compile == 1`（开头的 `--compile` 总开关、
    NPU 禁用 inductor 的那处），文本搜可能圈到不属于这个分支的区间。
    """
    main = _main_fn()
    for node in ast.walk(main):
        if not (isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == '_npu_graph'):
            continue
        for sub in node.orelse:
            if isinstance(sub, ast.If):
                return sub.lineno
    raise AssertionError('没找到 `if _npu_graph:` 的 elif 分支（结构被改了？）')


def _graph_block():
    m = re.search(r'if _npu_graph:(.*?)\n    elif args\.compile', CODE, re.S)
    assert m, '未找到 _npu_graph 分支'
    return m.group(1)


def _helper_source(name):
    """取 train_sft.py 里某个模块级函数的源码（含 docstring，注释已被 CODE 剔除）。"""
    m = re.search(r'(?m)^def %s\(.*?(?=\n\ndef |\n\nclass )' % name, CODE, re.S)
    assert m, '未找到函数 %s' % name
    return m.group(0)


# --------------------------------------------------------------------------- #
# 行为用例的公共夹具
# --------------------------------------------------------------------------- #
class _StubCompiled(nn.Module):
    """`OptimizedModule` 的最小替身：持有 `_orig_mod`，state_dict 键插一段前缀。

    与真 `OptimizedModule` 的差别只在 forward（直接转发，没有 dynamo）。本组用例
    关心的是「替换 / 回滚 / 键名」三件事，不是 dynamo 本身，故不必真跑一遍
    图编译（3 个 Linear ≈ 3.5s，每个用例都付不起）。
    """

    def __init__(self, mod):
        super().__init__()
        self._orig_mod = mod

    def forward(self, *a, **kw):
        return self._orig_mod(*a, **kw)


class _FakeTorchAirBackend:
    """假 torchair backend：只当身份标记用（桩 compile 不会真调它）。"""

    def __repr__(self):
        return '<fake-torchair-backend>'


class _Net(nn.Module):
    """刻意覆盖三种「Linear 住在哪」的形态：直接属性 / Sequential / ModuleList。"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)                      # 直接子模块
        self.seq = nn.Sequential(nn.Linear(3, 3), nn.ReLU())   # 名字是 '0'
        self.ml = nn.ModuleList([nn.Linear(3, 3)])     # 名字也是 '0'
        self.norm = nn.LayerNorm(3)                    # 非 Linear：必须原样不动
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(self.norm(self.ml[0](self.seq(self.fc(x)))))


def _is_wrapper(m):
    """编译包装体：真 `OptimizedModule` 与桩 `_StubCompiled` 都靠 `_orig_mod` 认。

     断言「替换生效了没有」时**必须**先排除包装体：包装体把原 Linear 挂在
    `_orig_mod` 下，而 `_orig_mod` 又是它自己的直接子模块，于是任何
    `isinstance(child, nn.Linear)` 的遍历在替换后照样能摸到那些 Linear ——
    照直断言「一个都不该剩」会永远为假，看不出替换到底生效没有。
    """
    return isinstance(getattr(m, '_orig_mod', None), nn.Module)


def _direct_children(model):
    """按**替换规则**列模型的直接子模块，不钻进编译包装体内部。"""
    out = []
    for parent in model.modules():
        if _is_wrapper(parent):
            continue
        for name, child in parent.named_children():
            out.append((parent, name, child))
    return out


def _direct_linears(model):
    """会被编译的那些 nn.Linear：`(parent, attr_name, original)` 三元组。"""
    return [(p, n, c) for p, n, c in _direct_children(model)
            if isinstance(c, nn.Linear)]


def _linears(model):
    return [c for _, _, c in _direct_linears(model)]


def _non_linears(model):
    """没被碰过的子模块（只看直接子级，且排除编译包装体本身）。"""
    return [c for _, _, c in _direct_children(model)
            if not isinstance(c, nn.Linear) and not _is_wrapper(c)]


@pytest.fixture
def spy_compile(monkeypatch):
    """把 `torch.compile` 换成记录调用的桩，返回 `[(被编译对象, kwargs), ...]`。

    被编译对象记的是**原 Linear 对象**本身，不是替身 —— 桩在调用时拿到的正是
    `train_sft.py` 传进来的那一个，身份可比。
    """
    calls = []

    def _fake(mod, **kwargs):
        calls.append((mod, kwargs))
        return _StubCompiled(mod)

    monkeypatch.setattr(t.torch, 'compile', _fake)
    return calls


# --------------------------------------------------------------------------- #
# 结构锁
# --------------------------------------------------------------------------- #
def test_flag_exists_and_defaults_off():
    assert '--npu-graph-compile' in SRC
    m = re.search(r"'--npu-graph-compile',\s*type=int,\s*default=(\d)", CODE)
    assert m, '未找到 --npu-graph-compile 定义'
    assert int(m.group(1)) == 0, \
        '图编译必须默认关闭：显存已到 26.7~31.1GB/32GB，且版本组合成熟度存疑'


def test_restricted_to_npu_backend():
    assert re.search(
        r"_npu_graph = \(args\.npu_graph_compile == 1 and _backend == 'npu'\)", CODE
    ), '图编译必须限定 NPU 后端（inductor 路径与 torchair 路径互斥）'


def test_import_order_npu_before_torchair():
    """torch_npu 必须先于 torchair，否则图模式静默降级为 eager 而不报错。"""
    body = _graph_block()
    i_npu = body.find('import torch_npu')
    i_air = body.find('import torchair')
    assert i_npu != -1, '缺少 import torch_npu'
    assert i_air != -1, '缺少 import torchair'
    assert i_npu < i_air, \
        'torch_npu 必须在 torchair 之前导入，否则图模式静默失效'


def test_uses_torchair_backend():
    """TorchAir backend 必须显式传给 `torch.compile`。

    **D2 改写**：原先断言的是 `torch.compile(model, backend=`，即「把 backend 传给
    **整模型** compile」。D2 删掉了整模型编译，那条正则**已作废**（它钉的是本次
    要废掉的那个形态，留着就等于把 D2 钉死不许改）。改为钉同一件事在新形态下的
    载体：backend 取自 `torchair.get_npu_backend(...)`，再交给逐子模块的
    `_compile_linear_submodules(model, _npui, …)`，最终由 helper 转交每次
    `torch.compile`。`dynamic=False` 也随之搬进 helper（下面按函数取源码断言）。
    """
    body = _graph_block()
    assert 'torchair.CompilerConfig()' in body
    assert 'torchair.get_npu_backend(' in body
    assert re.search(r'_compile_linear_submodules\(\s*model,\s*_npui,', body), \
        'torchair 的 backend 必须传给逐子模块编译入口，而不是被整模型 compile 吃掉'
    helper = _helper_source('_compile_linear_submodules')
    assert 'torch.compile(' in helper, 'helper 必须真的调 torch.compile'
    assert 'backend=backend' in helper, \
        '每个子模块的 torch.compile 都必须拿到显式 backend（inductor 路径与 torchair 路径互斥）'
    assert 'dynamic=False' in helper, '固定形状应关掉 dynamic（避免多次重编译）'


def test_linear_only_not_whole_model_compile():
    """D2 核心：`_npu_graph` 分支里不得再出现对**整模型**的 `torch.compile(model, …)`。"""
    body = _graph_block()
    assert not re.search(r'torch\.compile\(\s*model', body), \
        'NPU 路径不得再整模型 compile（D2：只递归替换 nn.Linear，其余 eager）'
    assert 'getattr(model, \'_orig_mod\', model)' not in body, \
        '整模型 compile 没了，顶层就不会再有 _orig_mod 包装；回退必须靠回滚表逐条写回'


def test_cuda_compile_path_untouched():
    """CUDA `--compile` 路径（D2 明确不动）仍是整模型 compile，形态与位置都不变。

    区间下界用 `_dist_wrap_lineno()`（`main()` 体内的 `DistributedDataParallel(...)`
    **调用节点**）作结束标记：这个测试要圈的是「CUDA 分支自身的代码」，而分布式
    包裹紧随其后。

     换轨记录：2026-10-01 之前这里锚的是 `_wrap_fsdp1`（FSDP1 时代），再往前是
    `DistributedDataParallel`（DDP 时代）。**两次都因为同一个原因坏掉**：按源码文本
    找一个「可能消失也可能被注释/docstring 命中」的字符串当下界，在换轨后要么恒假
    （命中 `main()` 之前的 import 行）要么恒真（命中解释性文字）。所以现在两端都
    走 AST：`ast.Call` + `isinstance(n.func, ast.Name)` + 限定 `main()` 子树。
    标记的存在性由 `_dist_wrap_lineno()` 里的 `len(hits) == 1` 守住（不再另立一条
    测试 —— 旧 docstring 里提到的 `test_fsdp_boundary_marker_exists` 从来不存在）。
    """
    assert 'model = torch.compile(model, dynamic=False, mode=args.compile_mode)' in CODE, \
        'CUDA --compile 路径不得被 D2 顺手改成 Linear-only'
    i_npu = CODE.index('if _npu_graph:')
    assert i_npu != -1, '找不到 _npu_graph 分支'
    i_cuda = _cuda_compile_branch_lineno()
    i_end = _dist_wrap_lineno()
    assert i_end > i_cuda, (
        '分布式包裹点（L%d）应在 CUDA compile 分支（L%d）**之后**：compile 必须先于包裹'
        '（反过来整模型 compile 会被 wrapper 的动态边界吞掉，拿到未融合图）'
        % (i_end, i_cuda))
    cuda = _code_span(i_cuda, i_end)
    assert 'torch.compile(model' in cuda, \
        'CUDA 分支里整模型 compile 不见了（Linear-only 只限 backend == npu）'
    assert '_rollback_linear_submodules' not in cuda, \
        'CUDA 分支的失败回退仍是剥 _orig_mod，不该被 D2 的回滚表机制污染'


def test_never_sets_mode_or_options():
    """昇腾 NPU 不支持 mode/options；reduce-overhead 更是 A100 OOM 的元凶。"""
    body = _graph_block()
    assert 'mode=' not in body, 'NPU 图编译不应传 mode（昇腾不支持且有 OOM 前科）'
    assert 'options=' not in body, 'NPU 图编译不应传 options（昇腾不支持）'
    assert 'reduce-overhead' not in body


def test_falls_back_to_eager_on_failure():
    """编译失败必须回退 eager 并告警——不能让整个训练崩掉。

    **D2 改写**：原先断言 `getattr(model, '_orig_mod', model)`（剥顶层包装）。
    D2 之后顶层压根没被包，这条断言钉的是已作废的机制；改为钉新机制：
    **回滚表逐条写回**（`_rollback_linear_submodules(_npu_rollback)`），
    且告警里带上还原条数——数目对得上才叫「还原干净」，比剥一条 getattr 更可核对。
    """
    body = _graph_block()
    assert 'except Exception' in body, '图编译失败必须被捕获'
    assert '_rollback_linear_submodules(_npu_rollback)' in body, \
        '回退时必须按回滚表逐条把 nn.Linear 换回原对象'
    assert '已还原 %d 个 nn.Linear' in body, \
        '回退告警必须报出还原条数（回滚是否干净要能核对，而不是靠剥一条 getattr 猜）'
    assert '回退 eager' in body, '回退必须有明确告警'


def test_warms_up_under_autocast():
    """预热前向必须与真实训练一致地包 autocast。

    否则 FP32 输入干灌会报 flash-attn 只接受 fp16/bf16，导致图编译被误判为
    不可用而回退 eager（且这个误判很难排查）。
    """
    body = _graph_block()
    assert 'maybe_autocast' in body, '预热前向应包 autocast'
    assert 'model(' in body or 'model(' in body.replace(' ', ''), '缺少预热前向'
    assert 'torch.no_grad()' in body, '预热只前向、不训练（no_grad 必须有）'


def test_warns_about_memory_risk():
    """开启时必须就显存风险给出提示（常驻已 26.7~31.1GB / 32GB）。"""
    assert '显存' in SRC, '应就显存风险给出提示'
    # 锚在 add_argument 上：SRC 里 --npu-graph-compile 现在还被函数 docstring
    # 与分支注释提到，用裸 search 会命中那些地方、拿不到 help 文本。
    m = re.search(r"add_argument\('--npu-graph-compile'", SRC)
    assert m, '缺少参数说明'
    # 参数 help 里应提到显存/风险（help 文本随 D2 变长过，窗口给足 1000）
    help_txt = SRC[m.start():m.start() + 1000]
    assert ('显存' in help_txt or '风险' in help_txt), \
        '--npu-graph-compile 的 help 未说明显存风险'


def test_npu_still_disables_inductor_path():
    """NPU 上 inductor 路径仍应被禁用（本项是另一条互斥路径）。"""
    m = re.search(
        r'if args\.compile == 1:(.*?)args\.compile = False', CODE, re.S)
    assert m, 'NPU 上 inductor 路径的禁用逻辑不见了'
    body = m.group(1)
    assert 'inductor' in body and '不可用' in body, \
        'NPU 上应继续禁用 inductor 并说明原因'
    assert '--npu-graph-compile' in body, \
        '禁用告警应把用户引导到 --npu-graph-compile 而不是已废弃的旧说法'


def test_shell_scripts_expose_the_switch():
    """NPU 训练脚本应能方便地开关它，且默认关闭。"""
    import glob
    npu = [f for f in glob.glob(os.path.join(ROOT, 'shell', 'train_sft_npu*.sh'))]
    assert npu, '未找到 NPU SFT 脚本'
    for f in npu:
        txt = open(f, encoding='utf-8').read()
        assert 'NPU_GRAPH_COMPILE' in txt, \
            '{} 未暴露 NPU_GRAPH_COMPILE 开关'.format(os.path.basename(f))
        m = re.search(r'(?m)^NPU_GRAPH_COMPILE=(\d)', txt)
        assert m, '{} 的 NPU_GRAPH_COMPILE 未显式赋 0/1'.format(os.path.basename(f))
        assert int(m.group(1)) in (0, 1)
        assert '--npu-graph-compile "$NPU_GRAPH_COMPILE"' in txt, \
            '{} 未把开关传给 train_sft.py'.format(os.path.basename(f))


# --------------------------------------------------------------------------- #
# 行为用例：递归替换（Linear-only 的「only」到底只到哪）
# --------------------------------------------------------------------------- #
def test_every_linear_is_replaced_exactly_once(spy_compile):
    """递归到**每个** nn.Linear，且只替换一次。"""
    m = _Net()
    expected = _linears(m)
    assert len(expected) == 3, '夹具本身应恰好 3 个 Linear'

    rb = t._compile_linear_submodules(m, _FakeTorchAirBackend())

    assert len(spy_compile) == len(expected), \
        f'应编译 {len(expected)} 个 nn.Linear，实际 {len(spy_compile)} 个'
    assert [c[0] for c in spy_compile] == expected, \
        '编译对象应就是遍历到的那些原 Linear（按遍历顺序）'
    assert len(rb) == len(expected), '回滚表条数应与替换数一致'
    assert _linears(m) == [], '替换后模型里不应再有裸 nn.Linear（都已进图）'


def test_non_linear_submodules_are_left_eager(spy_compile):
    """D2 的「only」：非 Linear 子模块一个都不许动。"""
    m = _Net()
    untouched = _non_linears(m)
    assert {type(x) for x in untouched} >= {nn.LayerNorm, nn.ReLU, nn.Dropout}, \
        '夹具应含多种非 Linear 子模块'

    t._compile_linear_submodules(m, _FakeTorchAirBackend())

    assert _non_linears(m) == untouched, '非 Linear 子模块不得被替换'


def test_container_children_are_replaced_by_attr_name(spy_compile):
    """Sequential / ModuleList 的子模块名是 '0' 这类字符串，setattr 同样要生效。"""
    m = _Net()
    seq_lin, ml_lin = m.seq[0], m.ml[0]

    t._compile_linear_submodules(m, _FakeTorchAirBackend())

    assert isinstance(m.seq[0], _StubCompiled), 'Sequential 内的 Linear 没被替换'
    assert isinstance(m.ml[0], _StubCompiled), 'ModuleList 内的 Linear 没被替换'
    assert m.seq[0]._orig_mod is seq_lin and m.ml[0]._orig_mod is ml_lin, \
        '替换必须把原对象挂在 _orig_mod 上（回滚/键名都依赖它）'


def test_backend_is_passed_to_every_submodule_call(spy_compile):
    """每一次 `torch.compile` 都得拿到同一个显式 backend（不得有漏网的 eager）。"""
    m = _Net()
    backend = _FakeTorchAirBackend()

    t._compile_linear_submodules(m, backend)

    assert spy_compile, '一次都没编译'
    for mod, kwargs in spy_compile:
        assert kwargs.get('backend') is backend, \
            f'{type(mod).__name__} 的 compile 没拿到显式 backend: {kwargs}'
        assert kwargs.get('dynamic') is False, \
            f'{type(mod).__name__} 的 compile 应关掉 dynamic: {kwargs}'


def test_model_with_no_linear_raises_instead_of_silently_doing_nothing():
    """没有 nn.Linear 时宁可抛出去走「回退 eager + 告警」，也不要假装开了图。"""
    m = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU())
    with pytest.raises(RuntimeError, match='nn.Linear'):
        t._compile_linear_submodules(m, _FakeTorchAirBackend())


# --------------------------------------------------------------------------- #
# 行为用例：回滚表与回滚
# --------------------------------------------------------------------------- #
def test_rollback_table_records_parent_name_and_original(spy_compile):
    """回滚表逐条记 `(parent_module, attr_name, original_module)`。"""
    m = _Net()
    seq, ml = m.seq, m.ml
    originals = [c for _, _, c in _direct_linears(m)]
    expected = [(m, 'fc'), (seq, '0'), (ml, '0')]

    rb = t._compile_linear_submodules(m, _FakeTorchAirBackend())

    assert len(rb) == 3
    for (parent, name, orig), (exp_parent, exp_name) in zip(rb, expected):
        assert parent is exp_parent, \
            f'回滚表第 {name} 条的 parent 记错了对象（回滚会写错地方）'
        assert name == exp_name, f'attr_name 应是 {exp_name!r}，实际 {name!r}'
        assert orig in originals, 'original 必须是遍历到的那些原 Linear 之一'
        assert isinstance(orig, nn.Linear), 'original 必须还是 nn.Linear'
    # parent 记的必须是**替换前**的容器对象：m.seq / m.ml 本身没被换掉
    # （只换了它们的子模块），所以回滚 setattr 落在原处。
    assert all(p is not None for p, _, _ in rb)


def test_rollback_restores_model_bitwise(spy_compile):
    """回滚后 model 与替换前**逐位等价**（键集与张量都要一模一样）。

    这是「失败不许污染模型」的硬指标：只查「有没有 `_orig_mod.` 残留」不够——
    万一某个 Linear 被换成了别的权重同样合法（键对得上、数值不同），
    那就是静默的训练结果错误。故按位比。
    """
    m = _Net()
    ref_sd = copy.deepcopy(m.state_dict())
    ref_linears = _linears(m)

    rb = t._compile_linear_submodules(m, _FakeTorchAirBackend())
    t._rollback_linear_submodules(rb)

    sd = m.state_dict()
    assert set(sd) == set(ref_sd), \
        f'回滚后键集不一致，多了 {set(sd) - set(ref_sd)} 少了 {set(ref_sd) - set(sd)}'
    for k in ref_sd:
        assert torch.equal(sd[k], ref_sd[k]), f'回滚后 {k} 的数值与替换前不等（静默污染）'
    assert _linears(m) == ref_linears, '回滚后 Linear 的对象身份应回到替换前'


def test_rollback_is_idempotent(spy_compile):
    """回滚幂等：两条失败路径（替换中途 / 预热失败）都调它也不会互相踩。"""
    m = _Net()
    ref_sd = copy.deepcopy(m.state_dict())
    rb = t._compile_linear_submodules(m, _FakeTorchAirBackend())

    assert t._rollback_linear_submodules(rb) == 3
    assert t._rollback_linear_submodules(rb) == 3, '第二次还原应仍返回条数而不是报错'
    for k in ref_sd:
        assert torch.equal(m.state_dict()[k], ref_sd[k])


def test_failure_midway_rolls_back_everything(monkeypatch):
    """替换到第 2 个 Linear 失败 → 抛出去，且**已替换的那几个**必须全还原。

    留一个「一半 Linear 进图」的混合模型不会报错，只会静默变慢且极难查
    （要逐个 verify 才看得出少了哪层图），所以半途失败必须整表回滚。
    """
    m = _Net()
    ref_sd = copy.deepcopy(m.state_dict())
    ref_linears = _linears(m)
    real_calls = []

    def _boom_on_second(mod, **kwargs):
        real_calls.append(mod)
        if len(real_calls) == 2:
            raise RuntimeError('假的后端炸了')
        return _StubCompiled(mod)

    monkeypatch.setattr(t.torch, 'compile', _boom_on_second)
    table = []
    with pytest.raises(RuntimeError, match='假的后端炸了'):
        t._compile_linear_submodules(m, _FakeTorchAirBackend(), table)

    assert len(real_calls) == 2, '应在第 2 个就炸（不是全跑完才炸）'
    assert len(table) == 1, '炸掉那条不该进回滚表（它并未生效）'
    assert _linears(m) == ref_linears, \
        '第 1 个 Linear 没还原 —— 半途失败留下了混合模型'
    assert set(m.state_dict()) == set(ref_sd)


def test_collect_before_replace_never_double_wraps(spy_compile):
    """ 不得二次包装：每个 Linear 只被编译一次，键里只多**一段** `_orig_mod.`。

    这条锁的是本文件开头第 5 条性质。`torch.compile` 返回的 OptimizedModule 本身
    也是 `nn.Module`；若边遍历边替换，`modules()` 会走进新包进去的 `_orig_mod`，
    那个 Linear 立刻又满足 `isinstance(..., nn.Linear)` → 无限套娃（实测
    RecursionError 995 层重复）。先收集再替换才躲得开。
    """
    m = _Net()
    originals = [id(x) for x in _linears(m)]

    t._compile_linear_submodules(m, _FakeTorchAirBackend())

    assert len(spy_compile) == len(originals), \
        f'每个 Linear 恰好编译一次（{len(originals)} 个），实际 {len(spy_compile)} 次'
    assert [id(c[0]) for c in spy_compile] == originals, '没有把包装后的对象再编一遍'
    sd = m.state_dict()
    assert max(k.count('_orig_mod.') for k in sd) == 1, \
        'state_dict 键里最多只该有一段 _orig_mod.（两段 = 套娃了）'


# --------------------------------------------------------------------------- #
# 行为用例：整条「替换 → 预热 → 回滚」路径（与后端无关，不需要 NPU）
# --------------------------------------------------------------------------- #
def test_replace_warmup_rollback_leaves_no_trace(spy_compile):
    """模拟 main() 的 _npu_graph 分支走一遍：替换 → 预热 → 失败回滚。

    刻意**不碰 NPU、不碰 torchair**：用假 backend + 桩 compile 复现同一段控制流，
    断言 NPU 不可用时不会污染模型（权重逐位不变、键名不带前缀、Linear 身份还原）。
    """
    m = _Net()
    ref_sd = copy.deepcopy(m.state_dict())
    rb = []
    warmup_ok = False
    try:
        rb = t._compile_linear_submodules(m, _FakeTorchAirBackend(), rb)
        with torch.no_grad():                       # 同 main()：只前向、不训练
            m(torch.randn(2, 4))
        warmup_ok = True
    except Exception:                              # noqa: BLE001 —— 复现 except 分支
        t._rollback_linear_submodules(rb)
    assert warmup_ok, '桩 compile 下预热本该成功（否则这条测的就不是回滚路径了）'

    # 成功路径：模型仍可正常前向，且参数是同一个对象（未被换掉）
    out = m(torch.randn(2, 4))
    assert out.shape == (2, 3)
    assert len(rb) == 3

    # 现在强制走回滚：模拟预热失败
    t._rollback_linear_submodules(rb)
    sd = m.state_dict()
    for k in ref_sd:
        assert torch.equal(sd[k], ref_sd[k]), f'回滚后 {k} 数值被污染'
    assert not any('_orig_mod.' in k for k in sd), '回滚后不该残留 _orig_mod. 段'
    assert m(torch.randn(2, 4)).shape == (2, 3), '回滚后模型应仍能前向'


def test_rollback_after_successful_use_keeps_training_identical(spy_compile):
    """编译期间训练过（权重更新）后再回滚，权重应保留**训练后的**值，不是初始值。
    回滚换的是**模块引用**不是权重：编译包装与原模块共享同一批 Parameter 对象。
    若哪天把替换写成了 deepcopy，这里会静默把训练成果扔掉。
    """
    m = _Net()
    ref0 = {k: v.clone() for k, v in m.state_dict().items()}
    rb = t._compile_linear_submodules(m, _FakeTorchAirBackend())

    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    trained = {k.replace('_orig_mod.', ''): v
               for k, v in copy.deepcopy(m.state_dict()).items()}
    assert any(not torch.equal(trained[k], ref0[k]) for k in ref0), \
        '夹具没生效：权重应当确实被加过 1.0'

    t._rollback_linear_submodules(rb)

    sd = m.state_dict()
    for k in trained:
        assert torch.equal(sd[k], trained[k]), \
            f'回滚把 {k} 退回成替换前的权重了 —— 训练成果被静默丢弃'



# --------------------------------------------------------------------------- #
# 键名对齐：存档 / 续训 / EMA（D2 的连带面）
# --------------------------------------------------------------------------- #
def test_saved_keys_are_stripped_at_every_level(spy_compile, tmp_path):
    """`save_model` 必须去掉**任意层级**的 `_orig_mod.`，不是只去开头那一个。

    D2 之前 `_orig_mod.` 只在开头（整模型 compile），故旧实现 `replace(..., 1)`
    够用。D2 之后它落在**路径中段**（`seq.0._orig_mod.weight`），只换首个会原样
    留下 → 存档键与 evaluate.py / inference / webui / convert_ckpt 期待的未编译
    布局对不上，load_state_dict 直接失败。
    """
    m = _Net()
    t._compile_linear_submodules(m, _FakeTorchAirBackend())

    path = tmp_path / 'sd.pth'
    t.save_model(m, str(path))
    sd = torch.load(str(path), weights_only=False)

    ref = {k: v for k, v in copy.deepcopy(m.state_dict()).items()}
    assert all('_orig_mod.' not in k for k in sd), \
        f'存档键还带 _orig_mod.: {[k for k in sd if "_orig_mod." in k]}'
    assert set(sd) == {'fc.weight', 'fc.bias', 'seq.0.weight', 'seq.0.bias',
                       'ml.0.weight', 'ml.0.bias', 'norm.weight', 'norm.bias'}
    for k, v in ref.items():
        assert torch.equal(sd[k.replace('_orig_mod.', '')], v)


def test_stripped_save_loads_into_a_compiled_model(spy_compile, tmp_path):
    """未编译布局的存档要能灌进已编译的模型（Linear-only 形态的 resume）。"""
    m = _Net()
    ref_sd = copy.deepcopy(m.state_dict())
    t._compile_linear_submodules(m, _FakeTorchAirBackend())
    path = tmp_path / 'sd.pth'
    t.save_model(m, str(path))
    sd = torch.load(str(path), weights_only=False)

    other = _Net()
    t._compile_linear_submodules(other, _FakeTorchAirBackend())
    log = _NullLogger()
    t._load_model_state(other, sd, log)

    got = {k.replace('_orig_mod.', ''): v for k, v in other.state_dict().items()}
    for k, v in ref_sd.items():
        assert torch.equal(got[k], v), f'续训后 {k} 不等于存档值'
    # 3 个 Linear × (weight+bias) = 6 个键要重定位；LayerNorm 那 2 个没被编译，
    # 键名不变。可观测性顺带被钉住：对齐发生了几处要能从日志里看出来。
    assert len(log.infos) == 1 and '6/8' in log.infos[0], \
        f'重定位计数应恰是 6 个被编译 Linear 的参数，实际日志 {log.infos}'


def test_ema_keys_survive_compile(spy_compile):
    """EMA 的 shadow 键必须对 compile 免疫，否则第一步 update 就 KeyError。

    EMA 在**编译之前**按当时的 `named_parameters()` 名字建 shadow，update 时却按
    **运行时**的名字取键；torch.compile 一改名键空间就断了。实测这条此前是直接
    崩的：`shell/train_sft_a100_1card.sh` 同时开 `--compile 1` 与 `--use-ema 1`，
    5 个 NPU 脚本也都开 `--use-ema 1`。
    """
    m = _Net()
    ema = t.EMA(m, decay=0.5)
    assert all('_orig_mod.' not in k for k in ema.shadow)

    t._compile_linear_submodules(m, _FakeTorchAirBackend())
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)

    ema.update()          # 编译后仍取得到 shadow
    ema.apply_shadow()
    ema.restore()
    assert all('_orig_mod.' not in k for k in ema.shadow), \
        'shadow 键不得带 _orig_mod.（它要存进 train_state 的 ema_shadow）'


def test_real_compile_artifacts_match_stub():
    """跑一次**真** `torch.compile`（backend='eager'），确认桩没骗人。

    桩 `_StubCompiled` 声称 OptimizedModule 的键名形状与真的一致（`_orig_mod.`
    落在路径中段）。本用例拿真的 OptimizedModule 对一遍，防止「桩与现实不符」
    让上面所有键名断言失去意义。只编 2 个 Linear，约 1s。
    """
    m = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    ref_sd = copy.deepcopy(m.state_dict())
    rb = t._compile_linear_submodules(m, 'eager')   # 'eager' backend = 假 torchair
    assert len(rb) == 2
    assert not _linears(m), '真 compile 后直接子级里不该再有裸 nn.Linear'
    assert '0._orig_mod.weight' in m.state_dict(), \
        f'真 OptimizedModule 的键名形状与桩不一致: {sorted(m.state_dict())}'
    assert all(k.count('_orig_mod.') <= 1 for k in m.state_dict())
    m(torch.randn(2, 4))                            # 真前向：图要真能编出来

    t._rollback_linear_submodules(rb)
    sd = m.state_dict()
    assert set(sd) == set(ref_sd)
    for k in ref_sd:
        assert torch.equal(sd[k], ref_sd[k])


class _NullLogger:
    """`_load_model_state` 只在「键被重定位」时打一条 info；哑对象即可。"""

    def __init__(self):
        self.infos = []

    def info(self, msg, *a, **kw):
        self.infos.append(msg % a if a else msg)

    def warning(self, *a, **kw):
        pass
