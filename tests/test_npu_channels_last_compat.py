"""NPU `--npu-channels-last` 的跨版本兼容门禁

事故（真机 torch 2.1.0 + torch_npu 2.1.0 / CANN 7.0.RC1，Linux）—— 连续三次
-------------------------------------------------------------------------
1. ``model.to(memory_format=torch.channels_last)``

       RuntimeError: Only contiguous_format or preserve_format is supported.

   ``torch_npu`` 覆写了 ``Module.to``（``torch_npu/utils/_module.py``）。

2. 改逐参数 ``param.data.contiguous(memory_format=channels_last)``

       RuntimeError: NPU contiguous operator only supportted contiguous
       memory format.  [ERROR] ERR01007 OPS feature not supported

3. 改官方文档那套 ``torch_npu.npu_format_cast(t, torch_npu.Format.NHWC)``

       AttributeError: module 'torch_npu' has no attribute 'Format'

   连 ``Format`` 枚举都没有 ⇒ 该 API 在这个版本上整体不存在（它是 CANN 8.x
   才有的 beta API，而本仓库训练环境是 2023-10 的 torch_npu 2.1.0）。

教训与本文件的定位
-----------------
**别再猜第四个 API。** 所以代码改成 `_npu_nhwc_formats()` 运行时探测可用入口，
一个都没有就干净降级；`scripts/probe_npu_layout.py` 负责把那台机器上「到底有
什么」问清楚。

这里钉住的是三条**结构性质**（能不能真跑通只能在真机验）：

  1. NPU 路径不再无条件引用 ``channels_last``；
  2. 输入侧与权重侧共用**同一套**探测/转换入口（否则 flag 等于没开）；
  3. 一个入口都没有 ⇒ 抛 ``NotImplementedError``，由 main 降级成
     ``use_channels_last=False``，训练照常（run.txt 的「只慢不坏」契约）。

⚠ 第 3 点里"helper 要抛、main 要接"这个分工很容易写反，且写反后症状是
**静默的布局错配**（权重 NCHW + 输入 NHWC），loss 照降、指标照报，所以单测
必须钉住。
"""

import importlib.util
import inspect
import pathlib
import sys
import types

import pytest
import torch
import torch.nn as nn


def _load_train_sft():
    """train_sft.py 不是可导入包（import 即跑 argparse），按文件加载取函数。"""
    p = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'train_sft.py'
    spec = importlib.util.spec_from_file_location('_ts_for_cl_test', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def ts():
    return _load_train_sft()


def _fake_torch_npu(*, fmt=True, cast=True, **extra):
    """造一个假的 torch_npu 模块并装进 sys.modules。"""
    m = types.ModuleType('torch_npu')
    if fmt:
        m.Format = types.SimpleNamespace(NHWC=1, ND=2)
    if cast:
        m.npu_format_cast = lambda t, f: t.clone()
    for k, v in extra.items():
        setattr(m, k, v)
    return m


@pytest.fixture
def install(monkeypatch):
    def _do(m):
        monkeypatch.setitem(sys.modules, 'torch_npu', m)
        return m
    return _do


# --------------------------------------------------------------------------- #
# 1. 探测：不同 torch_npu 版本探测出的入口
# --------------------------------------------------------------------------- #
def test_probe_finds_nothing_when_api_absent(ts, install):
    """当前真机的情况：既无 Format 也无 cast ⇒ 探测结果为空。"""
    install(_fake_torch_npu(fmt=False, cast=False))
    assert ts._npu_nhwc_formats() == []


def test_probe_uses_format_enum_when_available(ts, install):
    install(_fake_torch_npu(fmt=True, cast=True))
    got = ts._npu_nhwc_formats()
    assert len(got) == 1
    assert 'Format.NHWC' in got[0][0]


def test_probe_falls_back_to_integer_when_enum_missing(ts, install):
    """有 cast 函数但没枚举 ⇒ 按官方文档的整数值 NHWC=1 传。"""
    install(_fake_torch_npu(fmt=False, cast=True))
    got = ts._npu_nhwc_formats()
    assert len(got) == 1, got
    assert '1' in got[0][0]


def test_probe_must_use_getattr_chain_not_getattr_on_missing_attr(ts, install):
    """⚠ 第 3 个坑的根因：`getattr(mod.Format, 'NHWC', 1)` 不会返回默认值。

    它先取 `mod.Format`，该属性不存在时抛的是 `AttributeError`，根本走不到
    `getattr` 的默认值。所以探测必须写成 `getattr(mod, 'Format', None)`
    —— 先把整条属性链拆成两段。
    """
    install(_fake_torch_npu(fmt=False, cast=True))
    with pytest.raises(AttributeError):
        getattr(sys.modules['torch_npu'].Format, 'NHWC', 1)
    # 正确写法：拿到 None，于是回退到整数入口而不是崩
    assert ts._npu_nhwc_formats()


# --------------------------------------------------------------------------- #
# 2. 转换：权重侧 / 输入侧
# --------------------------------------------------------------------------- #
def test_helper_converts_only_4d_weights(ts, install):
    """只转 4D；Linear/Norm 的 2D、1D 权重必须保持 contiguous。"""
    install(_fake_torch_npu())
    m = nn.Sequential(nn.Conv2d(8, 16, 3), nn.Linear(16, 4), nn.BatchNorm2d(16))
    conv, lin, bn = m[0], m[1], m[2]
    n = ts._apply_channels_last_(m, backend='cpu')     # cpu 走原生 channels_last
    assert n == 1, n                                    # 只有那个 Conv2d
    assert conv.weight.is_contiguous(memory_format=torch.channels_last)
    assert lin.weight.is_contiguous()
    assert bn.weight.is_contiguous()


def test_npu_weight_path_goes_through_probe(ts, install):
    """`backend='npu'` 时必须经由探测出的入口，而不是直接引用 API 名。"""
    m = _fake_torch_npu()
    install(m)
    seq = nn.Sequential(nn.Conv2d(8, 16, 3), nn.Conv2d(16, 16, 3), nn.Linear(16, 4))
    n = ts._apply_channels_last_(seq, backend='npu')
    assert n == 2, n                                     # Linear 不算


def test_conversion_preserves_parameter_identity(ts, install):
    """⚠ 必须 `param.data = ...` 而不是新建 Parameter。

    否则 optimizer 在建好之后拿到的是**旧张量**的引用 ⇒ 卷积核静默不再更新。
    这比崩溃更难查：loss 照降、指标照报，只是模型学的是一份冻结的权重。
    """
    install(_fake_torch_npu())
    m = nn.Sequential(nn.Conv2d(8, 16, 3), nn.Linear(16, 4))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    before = [id(p) for p in m.parameters()]
    held = [id(p) for g in opt.param_groups for p in g['params']]

    ts._apply_channels_last_(m, backend='cpu')

    assert [id(p) for p in m.parameters()] == before, 'Parameter 身份变了'
    assert [id(p) for g in opt.param_groups for p in g['params']] == held, \
        '优化器持有的引用与当前参数脱钩'


def test_v7_net_conversion_keeps_param_count_and_forward(ts, install):
    """V7 上要真的转到卷积核，且转换后前向正常、参数量不变。"""
    install(_fake_torch_npu())
    from src.networks.katago_v7 import build_katago_v7_net, NBT_TF_CFG
    net = build_katago_v7_net(board_size=19)
    total = sum(p.numel() for p in net.parameters())

    n = ts._apply_channels_last_(net, backend='npu')

    four_d = [p for p in net.parameters() if p.dim() == 4]
    assert n == len(four_d) and n > 0, (n, len(four_d))
    assert sum(p.numel() for p in net.parameters()) == total == NBT_TF_CFG['params_total']
    net.eval()
    out = net(torch.randn(2, 22, 19, 19), torch.randn(2, 19))
    assert out['policy_logits'].shape[0] == 2


class _NpuTensor:
    """伪装 device.type == 'npu' 的张量（CPU 上 .device 不可写）。

    透传 ``clone``，好让假 torch_npu 的 `lambda t, f: t.clone()` 能跑通 ——
    这里只关心**分支判定**，不关心真 CANN 行为。
    """

    def __init__(self, t):
        self._t = t

    @property
    def device(self):
        return types.SimpleNamespace(type='npu')

    def dim(self):
        return self._t.dim()

    def clone(self):
        return _NpuTensor(self._t.clone())


def test_input_side_uses_npu_branch(ts, install):
    install(_fake_torch_npu())
    assert ts._to_nhwc(_NpuTensor(torch.randn(2, 8, 4, 4))) is not None


def test_input_non_4d_untouched(ts, install):
    """非 4D（GL 特征、label）不该被转换。"""
    install(_fake_torch_npu())
    t = torch.randn(2, 8)
    assert ts._to_nhwc(t) is t


def test_input_cpu_uses_native_channels_last(ts, install):
    install(_fake_torch_npu())
    t = torch.randn(2, 8, 4, 4)
    assert ts._to_nhwc(t).is_contiguous(memory_format=torch.channels_last)


def test_input_and_weight_share_the_probe(ts, install):
    """⚠ 输入侧与权重侧必须共用同一个探测结果。

    「权重转了、输入没转」（或反过来）会让每次卷积白搬一次布局 —— 那正是接这个
    flag 的目的，等于白开。两边各写一套判据就会漂移。
    """
    install(_fake_torch_npu())
    src = inspect.getsource(ts)
    # 一处定义 + 两处调用（权重侧 / 输入侧），输入侧不能再自己判一遍 npu。
    assert src.count('_cast_nhwc_npu(') == 3, \
        '应当恰好三处：1 处定义 + 权重侧 + 输入侧'
    assert src.count('def _cast_nhwc_npu') == 1


# --------------------------------------------------------------------------- #
# 2b. 转换「真的生效」要被读回校验（假成功是最阴的那种）
# --------------------------------------------------------------------------- #
def _box_like_torch_npu_2_1_0_post10(state):
    """复刻真机探针结果：有 npu_format_cast、**无** Format 枚举、有 get_npu_format。"""
    m = types.ModuleType('torch_npu')
    m.npu_format_cast = lambda t, f: (state.update(fmt=f), t.clone())[1]
    m.get_npu_format = lambda t: state['fmt']
    return m


def test_real_box_shape_picks_integer_entry_and_verifies(ts, install):
    """真机（torch_npu 2.1.0.post10）应走整数入口，且转换后 format 读回 1。"""
    state = {'fmt': 0}
    install(_box_like_torch_npu_2_1_0_post10(state))

    got = ts._npu_nhwc_formats()
    assert len(got) == 1 and 'Format.NHWC' not in got[0][0], got

    out = ts._cast_nhwc_npu(torch.randn(1, 4, 8, 8))
    assert state['fmt'] == ts.ACL_FORMAT_NHWC == 1
    assert out.shape == (1, 4, 8, 8)


def test_silent_no_op_cast_is_rejected(ts, install):
    """⚠ 若 cast 静默不生效（读回仍是 NCHW=0），必须抛 —— 不能让它混过去。

    `npu_format_cast` 返回新张量、不改原张量。某些版本/形状上它可能什么都不做，
    此时若不校验，日志照样打「已启用」，而真实布局没变 ⇒ 权重与输入都白搬，
    等于白开，且没有任何报错。**转换「声称」成功不算数，读回是 1 才算。**
    """
    m = types.ModuleType('torch_npu')
    m.npu_format_cast = lambda t, f: t.clone()      # 假装转换、实际没转
    m.get_npu_format = lambda t: 0                  # 读回仍是 NCHW
    install(m)
    with pytest.raises(RuntimeError, match='未生效'):
        ts._cast_nhwc_npu(torch.randn(1, 4, 8, 8))


def test_verify_can_be_disabled(ts, install):
    state = {'fmt': 0}
    install(_box_like_torch_npu_2_1_0_post10(state))
    out = ts._cast_nhwc_npu(torch.randn(1, 4, 8, 8), verify=False)
    assert out.shape == (1, 4, 8, 8)


def test_missing_get_npu_format_does_not_break(ts, install):
    """没有 get_npu_format（老版本）时跳过校验，而不是崩。"""
    m = types.ModuleType('torch_npu')
    m.npu_format_cast = lambda t, f: t.clone()
    install(m)                                     # 故意不给 get_npu_format
    assert ts._cast_nhwc_npu(torch.randn(1, 4, 8, 8)).shape == (1, 4, 8, 8)


def test_acl_format_constants_are_declared_before_use(ts):
    """常量必须在使用点之前定义（否则 import 就 NameError）。"""
    src = inspect.getsource(ts)
    assert src.index('ACL_FORMAT_NHWC = 1') < src.index('def _npu_nhwc_formats')


# --------------------------------------------------------------------------- #
# 3. 降级契约：一个入口都没有时
# --------------------------------------------------------------------------- #
def test_raises_not_implemented_when_nothing_available(ts, install):
    """一个入口都没有 ⇒ 抛 NotImplementedError，**不能**静默返回原张量。

    静默返回会让权重保持 NCHW 而日志说「已启用」，是假成功。
    """
    install(_fake_torch_npu(fmt=False, cast=False))
    with pytest.raises(NotImplementedError, match='不支持用户侧布局控制'):
        ts._cast_nhwc_npu(torch.randn(1, 4, 8, 8))
    with pytest.raises(NotImplementedError):
        ts._apply_channels_last_(nn.Conv2d(4, 4, 3), backend='npu')


def test_main_catches_and_disables_flag(ts):
    """main 必须 try 包住转换，并在失败时把 use_channels_last 置回 False。

    只置回标志很关键：输入侧（训练/评估循环）也读同一个变量，不回退就会出现
    「权重没转、输入转了」的布局错配 —— 而那种错配**不报错**，只是白搬数据。
    """
    src = inspect.getsource(ts)
    i = src.index('_n_cl = _apply_channels_last_(model, backend=_backend)')
    block = src[max(0, i - 400):i + 1200]      # `try:` 在调用点之前
    assert 'try:' in block
    assert 'use_channels_last = False' in block
    assert 'warning' in block


def test_main_log_names_the_missing_capability(ts):
    """降级日志要说清「本机可用入口 = 什么/无」，否则每次只能靠猜。"""
    src = inspect.getsource(ts)
    i = src.index('_n_cl = _apply_channels_last_(model, backend=_backend)')
    block = src[i:i + 1200]
    assert '_npu_nhwc_formats()' in block, '降级日志未回灌可用入口清单'
    assert 'probe_npu_layout' in block, '降级日志未指向诊断脚本'


# --------------------------------------------------------------------------- #
# 4. 诊断脚本本身
# --------------------------------------------------------------------------- #
def test_probe_script_exists_and_is_readonly():
    """`scripts/probe_npu_layout.py` 是排障入口，不能被改没了。"""
    p = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'probe_npu_layout.py'
    assert p.is_file(), 'probe_npu_layout.py 不在了'
    src = p.read_text(encoding='utf-8')
    ast_ok = importlib.util.spec_from_file_location('probe', p)
    mod = importlib.util.module_from_spec(ast_ok)
    ast_ok.loader.exec_module(mod)          # 只 import，不跑 main
    assert callable(mod.main)
    # 只读：不能出现会改动模型/写文件的调用
    for bad in ('torch.save', 'shutil', 'os.remove', 'rmtree'):
        assert bad not in src, bad
