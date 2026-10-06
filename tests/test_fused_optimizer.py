"""P4.6：SFT 融合优化器（fused，D6 相邻）。

对应简报：`.superpowers/sdd/2026-09-25-v21-roadmap/task-p4-6-optimizer-brief.md`
（引文与测试清单），规划原文见
`docs/superpowers/plans/2026-09-25-v21-roadmap.md:159`：

    | P4.6 | SFT 融合优化器（fused，D6 相邻） | `test_fused_optimizer` |

「fused」在本仓的准确含义：`torch.optim.AdamW(param_groups, fused=True)` —— torch 的
fused AdamW 路径（融合 exp_avg/exp_avg_sq 更新与参数写回的单 kernel/多张量实现）。
不是 MindSpeed 融合优化器（那是 D6 / P4.11 的事），也不新增任何 CLI 旗标（D1）。

回退契约（测试 1/3/4 从三个方向钉）：
* `cpu` → 标准构造，**bit-for-bit 等于 HEAD**（HEAD 的非 CUDA 分支就是
  `torch.optim.AdamW(_opt_groups)`，见 `train_sft.py` A1 锚点）；
* `npu` → 2026-10-06 曾接入 `torch_npu.optim.NpuFusedAdamW`，但同日真机实测
  （torch_npu 2.1.0.post10）**构造静默挂死**（双 DDP rank 同停 ~605MB RSS、
  无异常无日志，卡在 [device] 之后 [model] 之前），故暂时关闭尝试
  （`_NPU_FUSED_ATTEMPT = False`），一律 standard + foreach（foreach 多张量
  路径语义不变、拿走大部分 kernel-launch 收益）。代码路径保留，
  `_NPU_FUSED_ATTEMPT` 改回 True 即可重试；
* `cuda` 上 fused 构造抛 `TypeError/RuntimeError`（老 torch / 无 kernel）→
  回退标准构造，异常不冒泡。

数值口径（实测 torch 2.12.0+cpu，见 report）：
* fused vs standard **不是逐位相等**：10 步最大差 2.38e-7（fp32 ULP 量级）
  ⇒ 测试 2 用**绝对容差 1e-6**（4 倍余量；超参口径若变，偏差 ≥1e-3，必红）；
* 回退路径与 HEAD **必须逐位相等** ⇒ 测试 1/4 用 `torch.equal`，零容差。

本地无 CUDA：torch 2.12 的 CPU 张量也能构造并 step `fused=True`
（`defaults['fused'] is True`），因此测试 2 用 `device='cuda'` 选中与 A100 相同的
代码分支、参数留在 CPU 上跑**真实** fused 更新 —— 测的是分支与数值契约，不是模拟。
"""
import ast
import copy
import inspect
import logging
import os
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.train_sft as t  # noqa: E402
from scripts.train_sft import (  # noqa: E402
    _build_param_groups,
    _load_optimizer_state,
    build_adamw,
)

SRC_PATH = os.path.join(ROOT, 'scripts', 'train_sft.py')
SRC = open(SRC_PATH, encoding='utf-8').read()

# fused vs standard 的实测上界是 2.38e-7（10 步），留 4 倍余量；任何超参/分组
# 口径的分歧都会把偏差推到 >=1e-3，这个容差对那类 bug 有 3 个数量级的区分力。
_FUSED_TOL = 1e-6


# --------------------------------------------------------------------------- #
# 共用夹具
# --------------------------------------------------------------------------- #
class _Tiny(nn.Module):
    """与 test_huber_loss._Tiny 同构：fc（decay/no_decay 两类）+ value 头。

    `_build_param_groups` 按 `'value.'` 前缀把 value 头单列一组，
    没有 `value` 子模块会直接 AttributeError。
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3, bias=True)
        self.value = nn.Linear(4, 1, bias=True)


class _Args:
    """`_build_param_groups` 读到的三个属性（lr / weight_decay / value_lr_mult）。"""

    lr = 0.1
    weight_decay = 1e-4
    value_lr_mult = 5.0


def _seeded(seed=7):
    torch.manual_seed(seed)
    return _Tiny()


def _grad_seq(steps=10, seed=1234):
    """固定梯度序列（与网络无关的纯生成器产物），每步 {参数名: 梯度张量}。"""
    net = _seeded()
    gen = torch.Generator().manual_seed(seed)
    return [{n: torch.randn(p.shape, generator=gen)
             for n, p in net.named_parameters()} for _ in range(steps)]


def _run(opt, net, grad_seq):
    for g in grad_seq:
        with torch.no_grad():
            for n, p in net.named_parameters():
                p.grad = g[n].clone()   # 缺名即 KeyError，不许静默换随机梯度
        opt.step()


def _named_params(net):
    return [p.detach().clone() for _, p in net.named_parameters()]


def _max_diff(net_a, net_b):
    return max((a - b).abs().max().item()
               for a, b in zip(net_a.parameters(), net_b.parameters()))


def _main_tree():
    return ast.parse(textwrap.dedent(inspect.getsource(t.main)))


def _is_adamw_call(node):
    """`torch.optim.AdamW(...)` 调用节点（AST 判定，不吃源码子串的排版变化）。"""
    f = getattr(node, 'func', None)
    return (isinstance(f, ast.Attribute) and f.attr == 'AdamW'
            and isinstance(f.value, ast.Attribute) and f.value.attr == 'optim'
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id == 'torch')


# --------------------------------------------------------------------------- #
# 1. CPU 回退：bit-for-bit 等于 HEAD
# --------------------------------------------------------------------------- #
def test_cpu_fallback_is_bit_for_bit_head_behaviour():
    """`device='cpu'` → standard 模式，更新后参数与 HEAD 行为**逐位相等**。

    HEAD 参考 = A1 锚点（train_sft.py:1926-1935）非 CUDA 分支的原话
    `torch.optim.AdamW(_opt_groups)` —— 回退契约要求 P4.6 之后这条路径一个字节
    都没变，所以对照组就用这个表达式本身（同种子、同初始权重、同梯度序列、零容差）。
    """
    gs = _grad_seq(5)

    net_ref = _seeded()
    opt_ref = torch.optim.AdamW(_build_param_groups(net_ref, _Args()))  # HEAD 直构

    net_new = _seeded()
    opt_new, mode = build_adamw(_build_param_groups(net_new, _Args()), 'cpu')

    assert mode == 'standard', f'CPU 必须走标准回退，实得 mode={mode!r}'
    assert opt_new.defaults.get('fused') is None, (
        '回退构造的 defaults[fused] 必须与 HEAD 直构一致（None），'
        f"实得 {opt_new.defaults.get('fused')!r}")

    init = _seeded()
    _run(opt_ref, net_ref, gs)
    _run(opt_new, net_new, gs)

    assert _max_diff(init, net_new) > 1e-3, '轨迹没动，本用例失去区分力'
    for (n1, p1), (n2, p2) in zip(net_ref.named_parameters(),
                                  net_new.named_parameters()):
        assert torch.equal(p1, p2), (
            f'回退路径与 HEAD 不再逐位相等：{n1} max|Δ|='
            f'{(p1 - p2).abs().max().item():.3e}（回退契约要求零容差）')

    # 动量状态也逐位比（参数相等但 state 分叉 ⇒ 下一步就会散开）
    s_ref = list(opt_ref.state.values())
    s_new = list(opt_new.state.values())
    assert len(s_ref) == len(s_new) and s_ref, '优化器状态条目数不一致'
    for a, b in zip(s_ref, s_new):
        assert torch.equal(a['exp_avg'], b['exp_avg'])
        assert torch.equal(a['exp_avg_sq'], b['exp_avg_sq'])
        assert a['step'] == b['step']


# --------------------------------------------------------------------------- #
# 2. fused 激活时与 standard 容差内一致（**非逐位**，见模块 docstring）
# --------------------------------------------------------------------------- #
def test_fused_path_matches_standard_within_tolerance():
    """`device='cuda'` 选中 fused 分支；与 standard 同种子同梯度 10 步后
    `max|Δ| <= 1e-6`，且两条轨迹都**真正移动**（防空转断言）。

    本地无 CUDA：torch 2.12 的 CPU 张量可构造/可 step `fused=True`，
    分支选择与 A100 完全相同（`build_adamw` 只看 device 串），
    因此这里跑的是**真实** fused kernel 语义，不是 mock。
    实测（torch 2.12.0+cpu）：10 步 max|Δ| = 2.38e-7 ⇒ 容差 1e-6，**非逐位**。
    """
    gs = _grad_seq(10)

    net_s = _seeded()
    opt_s, mode_s = build_adamw(_build_param_groups(net_s, _Args()), 'cpu')
    net_f = _seeded()
    opt_f, mode_f = build_adamw(_build_param_groups(net_f, _Args()), 'cuda')

    assert mode_s == 'standard'
    assert mode_f == 'fused', (
        "device='cuda' 未选中 fused 分支（老 torch 不支持？本地 torch 2.12 应可构造），"
        f'实得 mode={mode_f!r}')
    assert opt_f.defaults.get('fused') is True, (
        'fused 优化器的 defaults[fused] 必须为 True —— 旗标没生效则本用例测的只是'
        '标准路径的自比较')
    assert opt_s.defaults.get('fused') is None

    init_f = _seeded()
    _run(opt_s, net_s, gs)
    _run(opt_f, net_f, gs)

    moved = _max_diff(init_f, net_f)
    assert moved > 1e-3, f'fused 轨迹没动（max|Δ|={moved:.3e}），断言失去区分力'
    diff = _max_diff(net_s, net_f)
    assert diff <= _FUSED_TOL, (
        f'fused 与 standard 更新不一致：10 步后 max|Δ|={diff:.3e} > {_FUSED_TOL}'
        f'（实测基准 2.38e-7；超此容差说明超参/分组口径分叉，不是 ULP 噪声）')


# --------------------------------------------------------------------------- #
# 3. NPU：一律 standard（+foreach）——NpuFusedAdamW 构造在 post10 真机挂死，
#    2026-10-06 起关闭尝试（_NPU_FUSED_ATTEMPT=False），代码路径保留待重试。
# --------------------------------------------------------------------------- #
def test_npu_falls_back_to_standard():
    """`'npu'` / `'npu:0'` 一律 standard：不尝试 fused、不报错。

    2026-10-06 当天曾接入 `NpuFusedAdamW`，同日真机（post10）构造静默挂死
    （双 DDP rank 同停），故 `_NPU_FUSED_ATTEMPT = False`。本测试钉住：
    即使 torch_npu 在场（桩），npa 分支也**不得**走 fused。
    """
    import types

    # 造一个「torch_npu 在场且带 NpuFusedAdamW」的环境：若门控被随手改回 True，
    # 本测试会抓到（构造期挂死在测试里不可重现，但分支选择可钉）。
    class _FakeNPUFusedAdamW(torch.optim.AdamW):
        pass

    fake_opt = types.ModuleType('torch_npu.optim')
    fake_opt.NpuFusedAdamW = _FakeNPUFusedAdamW
    fake_tnpu = types.ModuleType('torch_npu')
    fake_tnpu.optim = fake_opt
    monkeypatch_this = {'torch_npu': fake_tnpu, 'torch_npu.optim': fake_opt}
    saved = {k: sys.modules.get(k) for k in monkeypatch_this}
    sys.modules.update(monkeypatch_this)
    try:
        for dev in ('npu', 'npu:0'):
            net = _seeded()
            opt, mode = build_adamw(_build_param_groups(net, _Args()), dev)
            assert mode == 'standard', \
                f'{dev} 在 _NPU_FUSED_ATTEMPT=False 下必须 standard，实得 {mode!r}'  # noqa: NKLOC
            assert opt.defaults.get('fused') is None, \
                f'{dev} 的 defaults[fused] 被置位'
            # foreach 多张量路径应已启用（见 build_adamw standard 兜底的注释）
            assert opt.defaults.get('foreach') is True, \
                f'{dev} 的 foreach 路径未启用'
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_npu_fused_import_error_falls_back(monkeypatch):
    """torch_npu 在场但 `optim` 里两种拼写的 fused AdamW 都没有 ⇒ 回退 standard。"""
    import types

    fake_opt = types.ModuleType('torch_npu.optim')
    fake_tnpu = types.ModuleType('torch_npu')
    fake_tnpu.optim = fake_opt
    monkeypatch.setitem(sys.modules, 'torch_npu', fake_tnpu)
    monkeypatch.setitem(sys.modules, 'torch_npu.optim', fake_opt)

    net = _seeded()
    opt, mode = build_adamw(_build_param_groups(net, _Args()), 'npu')
    assert mode == 'standard', f'老 torch_npu 必须回退 standard，实得 {mode!r}'
    assert opt.defaults.get('fused') is None


# --------------------------------------------------------------------------- #
# 4. kernel 不可用（构造抛错）→ 回退标准、不冒泡、结果仍逐位等于 HEAD
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('exc', [RuntimeError, TypeError],
                         ids=['runtimeerror', 'typeerror'])
def test_kernel_unavailable_falls_back_without_error(monkeypatch, exc):
    """CUDA 分支构造 `fused=True` 抛 `RuntimeError`（无 kernel）或
    `TypeError`（老 torch 无此 kwarg）→ 返回 standard、异常不冒泡、
    更新与 HEAD 直构**逐位相等**。

    本地 torch 2.12 的 CPU 张量可构造 fused，所以「kernel 不可用」用
    monkeypatch 模拟（这是构造期错误的忠实再现：HEAD 的 try/except 捕的正是这两类）。
    """
    gs = _grad_seq(5)
    net_ref = _seeded()
    opt_ref = torch.optim.AdamW(_build_param_groups(net_ref, _Args()))  # HEAD 参考

    real_cls = torch.optim.AdamW

    class _NoFused(real_cls):
        def __init__(self, *args, **kwargs):
            if kwargs.get('fused'):
                raise exc('No fused kernels available (simulated)')
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(torch.optim, 'AdamW', _NoFused)

    net_new = _seeded()
    opt_new, mode = build_adamw(_build_param_groups(net_new, _Args()), 'cuda')
    assert mode == 'standard', f'构造失败必须回退 standard，实得 {mode!r}'

    _run(opt_ref, net_ref, gs)
    _run(opt_new, net_new, gs)
    for (n1, p1), (n2, p2) in zip(net_ref.named_parameters(),
                                  net_new.named_parameters()):
        assert torch.equal(p1, p2), (
            f'{exc.__name__} 回退后的更新与 HEAD 不再逐位相等：{n1} '
            f'max|Δ|={(p1 - p2).abs().max().item():.3e}')


# --------------------------------------------------------------------------- #
# 5. 优化器状态 save/resume 往返（fused↔fused 逐位、fused→standard 跨模式可续）
# --------------------------------------------------------------------------- #
def test_optimizer_state_save_resume_roundtrip():
    """`--resume` 契约：state_dict → `_load_optimizer_state` → 续跑轨迹一致。

    (a) 同模式（fused→fused）：加载返回 True、动量真的落进去、续跑**逐位**相等
        —— 与「不中断跑完 6 步」的参考轨迹比。
    (b) 跨模式（fused 存档 → standard 续跑，即 CUDA ckpt 在 CPU 上 resume）：
        加载返回 True，续跑在 §2 同一容差内（状态逐位搬移，只差 kernel 噪声）。
    (c) main() 的 resume 分支确实接了 `_load_optimizer_state(optimizer,
        tstate['optimizer'], …)` —— 否则 (a)(b) 只测了 helper 自己，生产接线断了
        测试仍绿。
    """
    logger = logging.getLogger('train')
    gs = _grad_seq(6)

    net = _seeded()
    opt, mode = build_adamw(_build_param_groups(net, _Args()), 'cuda')
    assert mode == 'fused'
    _run(opt, net, gs[:3])
    # 必须 deepcopy：`Optimizer.state_dict()` 的 state 值是对**活张量**的引用，
    # 不拷贝的话下面参考轨迹继续 step 会把快照里的动量也改掉（快照变成第 6 步的）。
    # 生产路径经 `torch.save` 落盘天然就是拷贝，别名问题只存在于测试内存里。
    sd = copy.deepcopy(opt.state_dict())
    weights3 = copy.deepcopy(net.state_dict())
    assert sd['state'], '3 步之后 state_dict 却是空的（前提失效）'

    _run(opt, net, gs[3:])          # 参考：不中断的 fused 轨迹（6 步）

    # (a) fused → fused 同模式
    # 每次加载都用 `deepcopy(sd)`：`load_state_dict` **不**深拷贝内层动量张量，
    #   加载后的优化器与传入的 state 共享同一批 tensor（实测 torch 2.12），续跑
    #   step 会把快照原地改掉 ⇒ (a) 的续跑会污染 (b) 的快照。生产里 tstate
    #   一次性用完即弃、无影响；别名问题只存在于「同一份快照加载两次」的测试里。
    net2 = _Tiny()
    net2.load_state_dict(weights3)
    opt2, mode2 = build_adamw(_build_param_groups(net2, _Args()), 'cuda')
    assert mode2 == 'fused'
    assert _load_optimizer_state(opt2, copy.deepcopy(sd), logger) is True, \
        '同模式 resume 被跳过'
    assert len(opt2.state) == len(opt.state) and opt2.state, '动量没有落进新优化器'
    _run(opt2, net2, gs[3:])
    assert _max_diff(net, net2) == 0.0, (
        f'同模式续跑与不间断轨迹不逐位相等：max|Δ|={_max_diff(net, net2):.3e}')

    # (b) fused → standard 跨模式（CUDA 存档在 CPU 续跑）
    net3 = _Tiny()
    net3.load_state_dict(weights3)
    opt3, mode3 = build_adamw(_build_param_groups(net3, _Args()), 'cpu')
    assert mode3 == 'standard'
    assert _load_optimizer_state(opt3, copy.deepcopy(sd), logger) is True, \
        '跨模式 resume 被跳过'
    assert opt3.state, '跨模式加载后 state 为空'
    _run(opt3, net3, gs[3:])
    diff = _max_diff(net, net3)
    assert diff <= _FUSED_TOL, (
        f'跨模式续跑偏差 {diff:.3e} > {_FUSED_TOL}：状态没被逐位搬过来？')

    # (c) main() 的接线
    main_fn = _main_tree().body[0]
    calls = [c for c in ast.walk(main_fn)
             if isinstance(c, ast.Call)
             and getattr(c.func, 'id', None) == '_load_optimizer_state']
    assert calls, 'main() 里没有 _load_optimizer_state 调用（resume 接线断了）'
    c = calls[0]
    assert isinstance(c.args[0], ast.Name) and c.args[0].id == 'optimizer'
    a1 = c.args[1]
    assert (isinstance(a1, ast.Subscript)
            and isinstance(a1.value, ast.Name) and a1.value.id == 'tstate'
            and ((isinstance(a1.slice, ast.Constant) and a1.slice.value == 'optimizer')
                 or (isinstance(a1.slice, ast.Index)
                     and getattr(a1.slice, 'value', None) is not None
                     and getattr(a1.slice.value, 'value', None) == 'optimizer'))), (
        f'第二实参必须是 tstate["optimizer"]，实得 {ast.unparse(a1)}')


# --------------------------------------------------------------------------- #
# 6. main() 只经 build_adamw 构造（绕过设备策略 ⇒ 红）
# --------------------------------------------------------------------------- #
def test_main_wiring_uses_build_adamw():
    """main() 的构造入口钉死：

    * 调 `build_adamw(_opt_groups, …)`（分组来源仍是 `_build_param_groups`）；
    * main() 内**没有**直构 `torch.optim.AdamW(` —— 有人绕过设备策略/回退契约
      在 main 里手搓一个优化器，本断言立刻红。
    """
    main_src = inspect.getsource(t.main)
    tree = _main_tree()
    main_fn = tree.body[0]

    called = {c.func.id for c in ast.walk(main_fn)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert 'build_adamw' in called, 'main() 没有调用 build_adamw（P4.6 接线丢失）'

    b_calls = [c for c in ast.walk(main_fn)
               if isinstance(c, ast.Call)
               and getattr(c.func, 'id', None) == 'build_adamw']
    assert b_calls, 'build_adamw 调用找不到'
    a0 = b_calls[0].args[0]
    assert isinstance(a0, ast.Name) and a0.id == '_opt_groups', (
        f'build_adamw 必须吃 _build_param_groups 的产物 _opt_groups，实得 '
        f'{ast.unparse(a0)}')

    direct = [ast.unparse(c) for c in ast.walk(main_fn) if _is_adamw_call(c)]
    assert not direct, (
        f'main() 里出现绕过 build_adamw 的直构 AdamW：{direct}'
        '（设备策略与回退契约只应存在于 build_adamw 一处）')

    assert '_build_param_groups(model, args)' in main_src, \
        'param_groups 的来源被换掉了'


# --------------------------------------------------------------------------- #
# 7. EMA / --compile（_orig_mod.）互不回归
# --------------------------------------------------------------------------- #
def test_ema_and_compile_interplay_unchanged():
    """按 main() 的真实顺序（optimizer/EMA 先建 → 后做 Linear-only compile，D2）
    验证三件事：

    1. 优化器持有的是**参数对象**，包装前后 id 集不变，step 仍推动 net 权重；
    2. EMA 的 shadow 键始终是未编译布局（`_ema_key` 剥 `_orig_mod.`），
       `update/apply_shadow/restore` 三连不炸；
    3. `compute_l2_report(optimizer.param_groups)` 包装前后**逐位相等**
       （报告读的是同一批张量，与名字无关）。
    """
    net = _seeded()
    opt, mode = build_adamw(_build_param_groups(net, _Args()), 'cpu')
    assert mode == 'standard'
    ema = t.EMA(net, decay=0.999)
    shadow_keys_before = set(ema.shadow)
    assert all('_orig_mod.' not in k for k in shadow_keys_before)

    ids_before = {id(p) for g in opt.param_groups for p in g['params']}
    l2_before = t.compute_l2_report(opt.param_groups)

    # 生产的 Linear-only compile（D2）；backend='eager' 免编译器，包装/命名与真实一致
    rollback = t._compile_linear_submodules(net, backend='eager')
    assert rollback, '前提失效：一个 Linear 都没包上'

    names_now = [n for n, _ in net.named_parameters()]
    assert any('_orig_mod.' in n for n in names_now), \
        f'前提失效：包装后名字里没有 _orig_mod. 段：{names_now}'

    ids_now = {id(p) for g in opt.param_groups for p in g['params']}
    assert ids_now == ids_before, (
        'compile 包装改变了参数对象 —— 优化器/EMA 指向失效'
        f'（少了 {len(ids_before - ids_now)} 个旧对象，多了 {len(ids_now - ids_before)} 个新对象）')

    # 3) 报告口径与名字无关，逐位相等
    l2_mid = t.compute_l2_report(opt.param_groups)
    assert l2_mid == l2_before, (
        f'包装改变了 l2_report：{l2_before!r} -> {l2_mid!r}（必须逐位相等）')

    # 1) step 仍推动 net 权重（优化器抓的是同一批对象）
    before = _named_params(net)
    with torch.no_grad():
        for p in net.parameters():
            p.grad = torch.full_like(p, 0.05)
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, _named_params(net))), \
        '包装后 optimizer.step() 没有推动任何参数（优化器指向失效）'

    # 2) EMA 三连：键经 _ema_key 归一，不得 KeyError、不得新增编译段
    ema.update()
    assert set(ema.shadow) == shadow_keys_before, 'shadow 键空间被改写'
    assert all('_orig_mod.' not in k for k in ema.shadow), \
        'shadow 键里混进了 _orig_mod.（_ema_key 剥段失效）'
    ema.apply_shadow()
    ema.restore()

    # 中段形态（Linear-only）与顶层形态（整模型 compile）都得剥
    assert t._ema_key('backbone.qkv._orig_mod.weight') == 'backbone.qkv.weight'
    assert t._ema_key('_orig_mod.backbone.conv.weight') == 'backbone.conv.weight'


# --------------------------------------------------------------------------- #
# 8. D1：fused 没有 CLI 旗标、没有环境变量旋钮（设备驱动的代码级默认）
# --------------------------------------------------------------------------- #
def test_no_fused_cli_or_env_knob():
    """P4.6 不加任何新参数（D1，roadmap:30）；fused 的选择只能是设备驱动的
    代码默认。三处钉死：

    * 模块里没有名字含 `fused` 的 `add_argument`；
    * `build_adamw` 签名固定 `(param_groups, device, logger=None)` —— 加一个
      `fused: bool = None` 形参就是把选择权交回调用方（等价于新旗标的入口）；
    * 函数体不读 `os.environ` / `os.getenv`（配置级旋钮也不许，D1 的精神）。
    """
    tree = ast.parse(SRC)
    flags = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'add_argument'
                and n.args and isinstance(n.args[0], ast.Constant)):
            flags.append(n.args[0].value)
    bad = [f for f in flags if 'fused' in str(f).lower()]
    assert not bad, f'出现了 fused 相关 CLI 旗标（D1 禁止）：{bad}'

    builder = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == 'build_adamw')
    params = [a.arg for a in builder.args.args]
    assert params == ['param_groups', 'device', 'logger'], (
        f'build_adamw 签名被改动（新增形参 = 变相新参数入口）：{params}')

    env_reads = [ast.unparse(n) for n in ast.walk(builder)
                 if (isinstance(n, ast.Attribute)
                     and n.attr in ('environ', 'getenv'))
                 or (isinstance(n, ast.Name) and n.id == 'environ')]
    assert not env_reads, f'build_adamw 读了环境变量（D1：不允许配置级旋钮）：{env_reads}'
