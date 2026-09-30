"""P4.6b —— 梯度检查点（ResBlocks 必开 / Mamba·Transformer 建议开 / CrossAttnRes 关）。

本文件同时是**测量台**：v21 的主干容器归 P4.2 建（`V21_CFG` / `AlphaGoNet`），
所以这里按 `task-p4-1-report.md` §7.1 的接线契约复刻了一份等价物
`V21BackboneHarness`，既是测试夹具也是 §5 实测的被测对象。
报告见 `.superpowers/sdd/2026-09-25-v21-roadmap/task-p4-6b-report.md`。

只跑本文件 + `tests/test_arch_v21_blocks.py` + `tests/test_npu_graph_compile.py`。
"""
import os
import sys
import threading
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.networks.backbone import (  # noqa: E402
    GC_CROSS_ATTN_RES,
    GC_LEGACY,
    GC_MAMBA,
    GC_PER_BLOCK_DEFAULT,
    GC_RES,
    GC_TRANSFORMER,
    V21_GRAD_CHECKPOINT_DEFAULTS,
    ConvNeXtBlock,
    CrossAttnRes,
    GradCheckpointMixin,
    MambaLTI,
    ResBlock,
    SharedBackbone,
    TransformerBlock,
    _BatchNormStatGuard,
    assert_grad_checkpoint_compile_compatible,
    compiled_module_paths,
    run_grad_segment,
)
from src.networks.policy_network import FCPolicyHead  # noqa: E402
from src.networks.value_network import FCValueHead  # noqa: E402

CH = 184
BOARD = 19
IN_CH = 17
DROPOUT = 0.1
EXPECT_TOTAL = 9_067_443
EXPECT_BACKBONE = 6_558_696
N_RES, N_MAMBA, N_TRANS, N_CROSS = 8, 4, 2, 2

# 抽头下标（0-based，`tap_positions` 取的是**该块的输出**，不是下一个块的输入）：
#   s1 = ResBlock #1 的输出  -> 段内下标 0
#   s5 = ResBlock #5 的输出  -> 段内下标 4
#   s9 = MambaLTI #1 的输出  -> 段内下标 0（P4.1 §7.3 读法 A/B：stem 不计数）
S1_POS, S5_POS, S9_POS = 0, 4, 0
RES_SLICE = slice(0, N_RES)
MAMBA_SLICE = slice(N_RES, N_RES + N_MAMBA)
TRANS_SLICE = slice(N_RES + N_MAMBA, N_RES + N_MAMBA + N_TRANS)
CROSS_SLICE = slice(N_RES + N_MAMBA + N_TRANS, None)


def _n(mod):
    return sum(p.numel() for p in mod.parameters())


class V21BackboneHarness(GradCheckpointMixin, nn.Module):
    """P4.2 将要建的 v21 主干的等价物：stem + 8/4/2/2 段 + out。

    段划分（= 报告 §2 的 (b) 粒度：同类型连续块合并成一段）：
        stem → [ResBlock ×8] → [MambaLTI ×4] → [TransformerBlock ×2]
             → [CrossAttnRes ×2] → out
    三路抽头 `s1/s5/s9` 用 `run_segment(..., tap_positions=...)` 从段内取，
    **不**切开段边界。
    """

    def __init__(self, in_channels=IN_CH, channels=CH, dropout=DROPOUT,
                 n_res=N_RES, n_mamba=N_MAMBA, n_trans=N_TRANS, n_cross=N_CROSS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels))
        blocks = ([ResBlock(channels) for _ in range(n_res)]
                  + [MambaLTI(channels) for _ in range(n_mamba)]
                  + [TransformerBlock(channels, ffn_hidden=240, attn_dropout=dropout)
                     for _ in range(n_trans)]
                  + [CrossAttnRes(channels, tap_channels=(channels,) * 3,
                                  ffn_hidden=240, attn_dropout=dropout)
                     for _ in range(n_cross)])
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels))
        self._init_grad_checkpointing()

    def forward(self, x):
        x = F.relu(self.stem(x))
        x, taps = self.run_segment(self.blocks[RES_SLICE], (x,), GC_RES,
                                   tap_positions=(S1_POS, S5_POS))
        s1, s5 = taps
        x, taps = self.run_segment(self.blocks[MAMBA_SLICE], (x,), GC_MAMBA,
                                   tap_positions=(S9_POS,))
        s9 = taps[0]
        x, _ = self.run_segment(self.blocks[TRANS_SLICE], (x,), GC_TRANSFORMER)
        x, _ = self.run_segment(self.blocks[CROSS_SLICE], (x, (s1, s5, s9)),
                                GC_CROSS_ATTN_RES)
        return F.relu(self.out(x))


class V21Net(GradCheckpointMixin, nn.Module):
    """主干 + 两个 v21 头，全网 9,067,443。"""

    def __init__(self, **kw):
        super().__init__()
        self.backbone = V21BackboneHarness(**kw)
        self.policy = FCPolicyHead(CH, board_size=BOARD)
        self.value = FCValueHead(CH)
        self._init_grad_checkpointing()

    def forward(self, x):
        h = self.backbone(x)
        return self.policy(h), self.value(h)

    def run_segment(self, blocks, args, kind, tap_positions=(), per_block=None,
                    uncapped_last=False):
        return run_grad_segment(blocks, args, use_checkpoint=False,
                                tap_positions=tap_positions, per_block=per_block,
                                uncapped_last=uncapped_last,
                                kind=kind, guard_root=None)

    def grad_checkpointing_for(self, kind):
        return self.backbone.grad_checkpointing_for(kind)

    def set_grad_checkpointing(self, enabled=None, **kinds):
        self.backbone.set_grad_checkpointing(enabled, **kinds)
        super(V21Net, self).set_grad_checkpointing(enabled, **kinds)
        return self


def _x(batch=1, board=BOARD, seed=20260927, ch=IN_CH):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, ch, board, board, generator=g)


def _small(**kw):
    """小尺寸 v21（board=5、通道 24）—— 语义测试用，跑得快。"""
    kw.setdefault('in_channels', 8)
    kw.setdefault('channels', 24)
    return V21BackboneHarness(**kw)


def _reset_bn(mod):
    for m in mod.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = 0.1


def _snap_bn(mod):
    out = {}
    for name, m in mod.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            out[name] = (m.running_mean.clone(), m.running_var.clone(),
                         int(m.num_batches_tracked))
    return out


def _bn_equal(a, b):
    if set(a) != set(b):
        return False, 'BN 模块集合不同: %s vs %s' % (sorted(a), sorted(b))
    for k in a:
        (m0, v0, n0), (m1, v1, n1) = a[k], b[k]
        if n0 != n1:
            return False, '%s: num_batches_tracked %d -> %d' % (k, n0, n1)
        if not torch.equal(m0, m1):
            return False, '%s: running_mean max|Δ|=%.3e' % (k, (m0 - m1).abs().max())
        if not torch.equal(v0, v1):
            return False, '%s: running_var max|Δ|=%.3e' % (k, (v0 - v1).abs().max())
    return True, ''


def _grads(mod, out, seed=0):
    torch.manual_seed(seed)
    loss = (out ** 2).sum() if torch.is_tensor(out) else sum((o ** 2).sum() for o in out)
    params = [p for p in mod.parameters() if p.requires_grad]
    return torch.autograd.grad(loss, params, allow_unused=True), params


def _saved_stats(fn):
    """forward 期间统计「留给反向的张量」个数与字节数。"""
    count = [0]
    nbytes = [0]

    def pack(t):
        count[0] += 1
        nbytes[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, count[0], nbytes[0]


def _fwd_calls(mod):
    """给每个块挂计数 hook，返回 {块名: forward 被调用次数}。"""
    counts = {}

    def mk(name):
        def hook(_m, _i, _o):
            counts[name] = counts.get(name, 0) + 1
        return hook

    handles = [b.register_forward_hook(mk('%s.%d' % (type(b).__name__, i)))
               for i, b in enumerate(mod.blocks)]
    return counts, handles


def _private_bytes():
    """本进程的 private commit（字节）。取不到就返回 None。

    口径说明（报告 §5 必须照抄）：这是**进程级**数字，包含
    「权重 + 梯度 + 留给反向的激活 + 反向的临时缓冲 + 分配器缓存/碎片」，
    所以它是「峰值占用」的**上界**，不是激活显存的精确值。梯度检查点**唯一**
    能精确控制的那一项是「留给反向的张量」，由 `_saved_stats` 数 —— 报告里
    两组数字并列给出。
    """
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes as wt

        class _PMC(ctypes.Structure):
            _fields_ = [
                ('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivateUsage', ctypes.c_size_t),
            ]

        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        psapi = ctypes.WinDLL('psapi')
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        # ⚠ 三个原型都要显式声明：`GetCurrentProcess` 的返回值是 HANDLE(64 位)，
        # 不声明 restype 会被按默认的 32 位 int 截断，句柄变成垃圾值，
        # `GetProcessMemoryInfo` 直接返回 0 —— 于是 `PrivateUsage` 一直是 0，
        # 峰值口径**静默**变成 null。
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
        psapi.GetProcessMemoryInfo.restype = wt.BOOL
        h = k32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return int(c.PrivateUsage)
        return None
    try:
        with open('/proc/self/statm', 'r') as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE')
    except Exception:
        return None


class _PeakMem:
    """采样 private commit 的最大值（相对进入时的基线）。

    Windows 上 `PeakWorkingSetSize` 是**进程生命周期**的峰值、无法重置，
    所以这里自己起线程按 1 ms 轮询当前值取区间内最大值。
    """

    def __init__(self, interval=0.001):
        self.interval = interval
        self.base = None
        self.peak = None
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            v = _private_bytes()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.base = _private_bytes()
        self.peak = self.base
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
        v = _private_bytes()
        if v is not None:
            self.peak = v if self.peak is None else max(self.peak, v)
        return False

    @property
    def delta_mb(self):
        if self.base is None or self.peak is None:
            return None
        return round(max(0, self.peak - self.base) / 2 ** 20, 1)


def _time(fn, warm=1, reps=3):
    for _ in range(warm):
        fn()
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


# ============================================================================ #
# 1. 形状/前向+全部参数梯度在开关两侧**数值一致**
# ============================================================================ #
def test_checkpointing_preserves_output_and_grad():
    m = _small()
    _reset_bn(m)
    x = _x(batch=2, board=5, seed=7, ch=8)

    m.train()
    m.set_grad_checkpointing(False)
    torch.manual_seed(11)
    o_off = m(x)
    g_off, params = _grads(m, o_off)

    m.set_grad_checkpointing(True)
    torch.manual_seed(11)
    o_on = m(x)
    g_on, _ = _grads(m, o_on)

    assert torch.equal(o_off, o_on), \
        '前向不逐位相同：max|Δ|=%.3e' % (o_off - o_on).abs().max()
    named = dict(m.named_parameters())
    assert len(params) > 0
    worst = 0.0
    for p, a, c in zip(params, g_off, g_on):
        nm = named[id(p)] if id(p) in named else '?'
        if (a is None) != (c is None):
            pytest.fail('%s: 梯度一侧为 None 一侧不为 None' % nm)
        if a is None:
            continue
        if not torch.equal(a, c):
            pytest.fail('%s: 梯度不逐位相同 max|Δ|=%.3e' % (nm, (a - c).abs().max()))
        worst = max(worst, float(a.abs().max()))
    # 反向确实走通了（不是全 None 的空转）
    assert worst > 0.0, '所有梯度都是零，测试是空转'


def test_bn_running_stats_not_double_updated():
    """§4.2：`num_batches_tracked` 必须是 1（不是 2），mean/var 逐位相等。"""
    m = _small()
    x = _x(batch=2, board=5, seed=5, ch=8)

    m.train()
    m.set_grad_checkpointing(False)
    _reset_bn(m)
    torch.manual_seed(3)
    o = m(x)
    o.sum().backward()
    ref = _snap_bn(m)

    m.set_grad_checkpointing(True)
    _reset_bn(m)
    for p in m.parameters():
        p.grad = None
    torch.manual_seed(3)
    o2 = m(x)
    o2.sum().backward()
    got = _snap_bn(m)

    ok, why = _bn_equal(ref, got)
    assert ok, 'BN running stats 被污染：' + why
    assert all(n == 1 for _, _, n in got.values()), \
        'num_batches_tracked 应为 1（单次前向），实得 %s' % sorted({n for _, _, n in got.values()})


def test_bn_guard_is_actually_entered_only_on_recompute():
    """守卫必须**只**在重算期间进入（原前向不能进，否则统计会被冻住）。"""
    m = _small()
    m.train()
    m.set_grad_checkpointing(True)
    _reset_bn(m)
    seen = []
    orig = _BatchNormStatGuard.__enter__

    def spy(self):
        seen.append(1)
        return orig(self)
    _BatchNormStatGuard.__enter__ = spy
    try:
        o = m(_x(batch=1, board=5, seed=1, ch=8))
        assert seen == [], '原前向期间就进入了守卫（entries=%d）' % len(seen)
        o.sum().backward()
    finally:
        _BatchNormStatGuard.__enter__ = orig
    n_seg = sum(1 for k in (GC_RES, GC_MAMBA, GC_TRANSFORMER)
                if m.grad_checkpointing_for(k))
    assert len(seen) == n_seg, \
        '重算期间守卫进入次数应为「被检查点的段数」%d，实得 %d' % (n_seg, len(seen))


# ============================================================================ #
# 2. Mamba-2 的 A_log 梯度（§4.1）
# ============================================================================ #
def test_A_log_grad_nonzero_and_equal_under_checkpointing():
    m = _small()
    _reset_bn(m)
    x = _x(batch=2, board=5, seed=13, ch=8)
    a_names = [n for n, _ in m.named_parameters() if n.endswith('A_log')]

    def a_grad(on):
        m.set_grad_checkpointing(on)
        for p in m.parameters():
            p.grad = None
        torch.manual_seed(17)
        o = m(x)
        o.sum().backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()
                if n.endswith('A_log')}

    off = a_grad(False)
    on = a_grad(True)
    assert len(off) == N_MAMBA and len(on) == N_MAMBA, \
        '预期 %d 个 A_log，实得 %d' % (N_MAMBA, len(on))
    for n in a_names:
        assert off[n] is not None and on[n] is not None, '%s 梯度为 None' % n
        assert float(on[n].abs().max()) > 0.0, \
            '%s 梯度恒零 —— A_log 是 Mamba-2 唯一学到衰减的入口，不能丢' % n
        assert torch.equal(off[n], on[n]), \
            '%s: 开/关检查点的梯度不一致 max|Δ|=%.3e' % (n, (off[n] - on[n]).abs().max())


def test_mamba_checkpointed_path_never_calls_the_sequential_oracle():
    """§4.4：检查点路径不得引用测试专用 oracle。"""
    m = _small()
    m.train()
    m.set_grad_checkpointing(True)
    calls = []
    for blk in m.blocks:
        if isinstance(blk, MambaLTI):
            orig = blk._sequential_scan_oracle

            def spy(*a, __o=orig, __b=blk):
                calls.append(__b)
                return __o(*a)
            blk._sequential_scan_oracle = spy
    o = m(_x(batch=1, board=5, seed=2, ch=8))
    o.sum().backward()
    assert calls == [], '检查点路径调用了 _sequential_scan_oracle %d 次' % len(calls)


# ============================================================================ #
# 3. dropout 掩码在重算下可复现（§4.3）
# ============================================================================ #
def test_dropout_mask_reproduced_under_checkpointing():
    """同权重同输入、train 模式、同 seed：开关两侧的**前向与全部梯度**必须逐位相同。

    ⚠ **必须包含反向**：检查点的前向与不开检查点的前向本来就是同一条计算
    （掩码相同），**只有重算那次前向**才会用到 RNG。实测（mutation
    `preserve_rng_state=False`）：只比前向的版本对这个错误改动**完全免疫**，
    28 个测试全绿 —— 所以这里比的是 fwd+bwd，并且先断言「重算确实发生过」
    （否则「相等」可能是没重算换来的）。
    """
    m = V21BackboneHarness(in_channels=8, channels=24, dropout=0.3)
    _reset_bn(m)
    x = _x(batch=2, board=5, seed=23, ch=8)
    assert any(isinstance(b, (TransformerBlock, CrossAttnRes)) for b in m.blocks)

    def fwd_bwd(on):
        m.set_grad_checkpointing(on)
        for p in m.parameters():
            p.grad = None
        counts, handles = _fwd_calls(m)
        try:
            torch.manual_seed(101)
            out = m(x)
            out.sum().backward()
        finally:
            for h in handles:
                h.remove()
        return out, {n: p.grad.detach().clone() for n, p in m.named_parameters()}, counts

    off, g_off, c_off = fwd_bwd(False)
    on, g_on, c_on = fwd_bwd(True)
    assert set(c_off.values()) == {1}, '未开检查点却重算了：%s' % c_off
    assert any(v >= 2 for v in c_on.values()), \
        '开了检查点却没有重算 —— 这条测试对 preserve_rng_state 免疫（空转）'
    assert torch.equal(off, on), \
        '同 seed 下开/关检查点的前向不逐位相同 max|Δ|=%.3e —— 重算没用同一个掩码' \
        % (off - on).abs().max()
    named = {n for n, _ in m.named_parameters()}
    for n in named:
        a, c = g_off[n], g_on[n]
        assert (a is None) == (c is None), '%s: 梯度一侧为 None 一侧不为 None' % n
        if a is None:
            continue
        assert torch.equal(a, c), \
            '%s: 重算的梯度不逐位相同 max|Δ|=%.3e（掩码没被复现）' \
            % (n, (a - c).abs().max())

    # 非空转：train 下两次前向必须不同（掩码确实在动）
    torch.manual_seed(101)
    a = m(x)
    torch.manual_seed(202)
    b = m(x)
    assert not torch.equal(a, b), 'train 下两次前向逐位相同 —— dropout 压根没进路径'


def test_eval_mode_zero_dropout_regardless_of_switch():
    m = V21BackboneHarness(in_channels=8, channels=24, dropout=0.3)
    _reset_bn(m)
    x = _x(batch=2, board=5, seed=29, ch=8)
    m.train()
    torch.manual_seed(1)
    tr = m(x)
    m.eval()
    m.set_grad_checkpointing(False)
    torch.manual_seed(1)
    e_off = m(x)
    m.set_grad_checkpointing(True)
    torch.manual_seed(1)
    e_on = m(x)
    assert torch.equal(e_off, e_on), 'eval 下开关改变了输出'
    assert not torch.equal(tr, e_off), 'eval 与 train 输出相同 —— dropout 闸门失效'


# ============================================================================ #
# 4. eval / 推理零行为变化 + 零开销
# ============================================================================ #
def test_eval_mode_is_unaffected():
    m = _small()
    _reset_bn(m)
    x = _x(batch=2, board=5, seed=31, ch=8)
    bn_before = _snap_bn(m)
    m.eval()
    for kind in (GC_RES, GC_MAMBA, GC_TRANSFORMER, GC_CROSS_ATTN_RES):
        assert not m.grad_checkpointing_for(kind), \
            'eval 下 %s 段仍会走检查点' % kind
    m.set_grad_checkpointing(False)
    with torch.no_grad():
        off = m(x)
    m.set_grad_checkpointing(True)
    with torch.no_grad():
        on = m(x)
    assert torch.equal(off, on), 'eval 下输出不逐位相同 max|Δ|=%.3e' % (off - on).abs().max()
    ok, why = _bn_equal(bn_before, _snap_bn(m))
    assert ok, 'eval 前向动了 BN 统计：' + why


def test_inference_zero_overhead():
    m = _small()
    x = _x(batch=2, board=5, seed=37, ch=8)
    m.eval()

    def once():
        with torch.no_grad():
            m(x)

    m.set_grad_checkpointing(False)
    t_off = _time(once, warm=2, reps=5)
    m.set_grad_checkpointing(True)
    t_on = _time(once, warm=2, reps=5)
    assert t_on <= 1.5 * t_off, \
        'eval 下开关带来 %.1f%% 的耗时变化（%.2f -> %.2f ms），应为零' \
        % ((t_on / t_off - 1) * 100, t_off, t_on)


# ============================================================================ #
# 5. 开关形态
# ============================================================================ #
def test_defaults_match_the_ruling():
    assert V21_GRAD_CHECKPOINT_DEFAULTS == {
        GC_RES: True, GC_MAMBA: True, GC_TRANSFORMER: True,
        GC_CROSS_ATTN_RES: False, GC_LEGACY: False,
    }, '默认值与用户 2026-09-27 裁决不符：%s' % V21_GRAD_CHECKPOINT_DEFAULTS
    m = _small()
    assert m.grad_checkpointing is True
    assert m.grad_checkpointing_kinds() == V21_GRAD_CHECKPOINT_DEFAULTS
    assert m.grad_checkpointing_for(GC_RES) is True
    assert m.grad_checkpointing_for(GC_MAMBA) is True
    assert m.grad_checkpointing_for(GC_TRANSFORMER) is True
    assert m.grad_checkpointing_for(GC_CROSS_ATTN_RES) is False
    with pytest.raises(ValueError):
        m.grad_checkpointing_for('nope')
    with pytest.raises(ValueError):
        m.set_grad_checkpointing(nope=True)


def test_switch_without_rebuilding_the_model():
    m = _small()
    _reset_bn(m)
    x = _x(batch=1, board=5, seed=41, ch=8)
    m.train()
    m.set_grad_checkpointing(False)
    torch.manual_seed(3)
    a = m(x)
    m.set_grad_checkpointing(True)
    torch.manual_seed(3)
    b = m(x)
    m.set_grad_checkpointing(mamba=False)
    torch.manual_seed(3)
    c = m(x)
    assert torch.equal(a, b) and torch.equal(a, c)
    assert m.grad_checkpointing is True, '只给逐类型覆盖时总开关不该被动'
    assert m.grad_checkpointing_for(GC_MAMBA) is False
    assert m.grad_checkpointing_for(GC_RES) is True
    m.set_grad_checkpointing(False)
    assert m.grad_checkpointing is False
    assert m.grad_checkpointing_for(GC_RES) is False


class _Wrapper(GradCheckpointMixin, nn.Module):
    """父级持有 mixin、`forward` 只调子模块 —— 正是 P4.2 的 `AlphaGoNet` 形状。"""

    def __init__(self):
        super().__init__()
        self.backbone = V21BackboneHarness(in_channels=8, channels=24)

    def forward(self, x):
        return self.backbone(x)


def test_switch_owner_must_be_the_module_that_calls_run_segment():
    """接线坑：开关**不向子模块传播**，持有者必须就是调 `run_segment` 的那个模块。

    P4.2 的 `AlphaGoNet` 若持有 mixin 而 `forward` 只调 `self.backbone(x)`，那
    `net.set_grad_checkpointing(True)` 会**静默不生效**（主干里的
    `grad_checkpointing_for` 读的是主干自己那份 mixin 状态）。下面把三种写法
    的行为差异逐个钉住：只在父级开 = 不生效；设在主干上 = 生效；父级显式
    转发 = 生效。
    """
    wrapper = _Wrapper()
    _reset_bn(wrapper)
    x = _x(batch=1, board=5, seed=43, ch=8)
    wrapper.train()

    def fwd_counts():
        counts, handles = _fwd_calls(wrapper.backbone)
        try:
            torch.manual_seed(3)
            wrapper(x).sum().backward()
        finally:
            for h in handles:
                h.remove()
        return counts

    def reset():
        wrapper.set_grad_checkpointing(False)
        wrapper.backbone.set_grad_checkpointing(False)

    reset()
    assert set(fwd_counts().values()) == {1}, '两侧都关时不该有重算'

    # 只在父级开：子模块看不到 —— 静默不生效（P4.2 会踩的坑）
    wrapper.set_grad_checkpointing(True)
    assert wrapper.grad_checkpointing is True
    assert wrapper.backbone.grad_checkpointing is False
    assert set(fwd_counts().values()) == {1}, \
        'mixin 的开关竟然自动传播到了子模块 —— 那 P4.2 就不必转发，契约需重写'

    # 形态 (a)：开关设在真正跑段的主干上 → 重算发生
    reset()
    wrapper.backbone.set_grad_checkpointing(True)
    assert any(v >= 2 for v in fwd_counts().values()), fwd_counts()

    # 形态 (b)：父级显式转发 → 重算发生
    reset()
    wrapper.set_grad_checkpointing(True)
    wrapper.backbone.set_grad_checkpointing(wrapper.grad_checkpointing)
    assert any(v >= 2 for v in fwd_counts().values()), fwd_counts()


def test_state_dict_unaffected_by_switch():
    m = _small()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    keys_before = set(before)
    m.set_grad_checkpointing(False)
    mid = {k: v.clone() for k, v in m.state_dict().items()}
    m.set_grad_checkpointing(True, res=False, mamba=True, transformer=False)
    after = m.state_dict()
    assert set(after) == keys_before, '开关改了 state_dict 的键集合'
    for k in keys_before:
        assert torch.equal(before[k], mid[k]), k
        assert torch.equal(before[k], after[k].clone()), k
    # 开关属性既不是 parameter 也不是 buffer
    assert not any('grad_checkpoint' in n for n, _ in m.named_parameters())
    assert not any('grad_checkpoint' in n for n, _ in m.named_buffers())


def test_param_count_unchanged():
    net = V21Net()
    assert _n(net.backbone) == EXPECT_BACKBONE, \
        '主干 %d != %d' % (_n(net.backbone), EXPECT_BACKBONE)
    assert _n(net.policy) == 2_366_730
    assert _n(net.value) == 142_017
    assert _n(net) == EXPECT_TOTAL, '全网 %d != %d' % (_n(net), EXPECT_TOTAL)
    m2 = V21Net()
    m2.set_grad_checkpointing(False)
    assert _n(m2) == EXPECT_TOTAL
    sd = net.state_dict()
    m2.load_state_dict(sd)
    assert _n(m2) == EXPECT_TOTAL


# ============================================================================ #
# 6. compile 互斥守卫
# ============================================================================ #
def test_compile_and_checkpoint_conflict_still_raises():
    m = _small()
    m.train()
    m.set_grad_checkpointing(False)
    lin = m.blocks[0].conv1
    compiled = torch.compile(lin, backend='eager')
    assert hasattr(compiled, '_orig_mod'), \
        'torch.compile 没产出 _orig_mod（守卫的判据失效）'
    m.blocks[0].conv1 = compiled
    assert compiled_module_paths(m) == ['blocks.0.conv1'], compiled_module_paths(m)

    with pytest.raises(RuntimeError, match='互斥'):
        m.set_grad_checkpointing(True)
    with pytest.raises(RuntimeError, match='互斥'):
        m(_x(batch=1, board=5, seed=3, ch=8))

    m.blocks[0].conv1 = lin
    assert compiled_module_paths(m) == []
    m.set_grad_checkpointing(True)
    torch.manual_seed(1)
    out = m(_x(batch=1, board=5, seed=3, ch=8))
    assert out.shape == (1, 24, 5, 5)


def test_assert_helper_is_public_and_callable_on_a_bare_model():
    m = _small()
    assert_grad_checkpoint_compile_compatible(m)
    wrapped = torch.compile(nn.Linear(4, 4), backend='eager')
    with pytest.raises(RuntimeError, match='互斥'):
        assert_grad_checkpoint_compile_compatible(wrapped)


# ============================================================================ #
# 7. 三种块类型真的在段内 + 分组粒度
# ============================================================================ #
def test_each_block_type_is_covered():
    """用 **forward 调用次数**证明覆盖（不是读代码）。

    被检查点的段在 fwd+bwd 里每块至少跑 2 次（原前向 + 重算）。⚠ 段的**最后
    一块**通常只跑 1 次：`torch.utils.checkpoint` 的 `early_stop=True`（默认）在
    「所有被留的张量都重算完」那一刻就抛 `_StopRecomputationError` 中止重算，
    段尾那几块不再重跑。所以判据是「除末块外每块 == 2、末块 >= 1」，而
    **未开检查点的块一段都不会跑第 2 次** —— 少开一个类型会立刻掉到 0。
    """
    m = _small()
    _reset_bn(m)
    x = _x(batch=1, board=5, seed=51, ch=8)
    m.train()
    segs = {GC_RES: RES_SLICE, GC_MAMBA: MAMBA_SLICE,
            GC_TRANSFORMER: TRANS_SLICE, GC_CROSS_ATTN_RES: CROSS_SLICE}
    enabled = (GC_RES, GC_MAMBA, GC_TRANSFORMER)

    def run(kinds):
        m.set_grad_checkpointing(True, **kinds)
        _reset_bn(m)
        counts, handles = _fwd_calls(m)
        try:
            torch.manual_seed(2)
            m(x).sum().backward()
        finally:
            for h in handles:
                h.remove()
        return counts

    off = run(dict(res=False, mamba=False, transformer=False))
    assert set(off.values()) == {1}, '未开检查点时每块应只跑 1 次：%s' % off

    on = run({})
    for kind, sl in segs.items():
        names = ['%s.%d' % (type(m.blocks[i]).__name__, i)
                 for i in range(sl.start, len(m.blocks) if sl.stop is None else sl.stop)]
        for j, n in enumerate(names):
            if kind in enabled and j < len(names) - 1:
                assert on[n] == 2, '%s 未被检查点覆盖（前向调用 %d 次）' % (n, on[n])
            else:
                assert on[n] == 1, '%s 不该重算，实际前向 %d 次' % (n, on[n])
        if kind in enabled:
            assert sum(1 for n in names if on[n] >= 2) >= len(names) - 1, \
                '%s 段几乎没有块被重算：%s' % (kind, {n: on[n] for n in names})


def test_cross_attn_res_can_be_turned_on_and_is_then_covered():
    m = _small()
    _reset_bn(m)
    x = _x(batch=1, board=5, seed=53, ch=8)
    m.train()
    m.set_grad_checkpointing(True, cross_attn_res=True)
    counts, handles = _fwd_calls(m)
    try:
        torch.manual_seed(4)
        m(x).sum().backward()
    finally:
        for h in handles:
            h.remove()
    assert counts['CrossAttnRes.14'] == 2, counts
    assert counts['CrossAttnRes.15'] >= 1, counts


def test_grouping_is_one_segment_per_run_not_per_block():
    """粒度 = (b)：同类型连续块合并成一段 ⇒ 只留 1 个边界激活（逐块会留 n−1 个）。"""
    m = _small()
    _reset_bn(m)
    stem_out = F.relu(m.stem(_x(batch=2, board=5, seed=57, ch=8)))
    seg = m.blocks[RES_SLICE]
    m.train()

    def saved(**kw):
        _reset_bn(m)
        _, n, b = _saved_stats(lambda: run_grad_segment(seg, (stem_out,), **kw))
        return n, b

    n_off, b_off = saved(use_checkpoint=False)
    n_mrg, b_mrg = saved(use_checkpoint=True, per_block=False)
    n_blk, b_blk = saved(use_checkpoint=True, per_block=True)
    assert b_mrg < b_off, '开检查点没有减少留给反向的字节：%d -> %d' % (b_off, b_mrg)
    assert n_mrg < n_off, '开检查点没有减少留给反向的张量数：%d -> %d' % (n_off, n_mrg)
    elem = stem_out.numel() * stem_out.element_size()
    assert b_mrg < b_blk, '合并段没有比逐块段省：%d vs %d' % (b_mrg, b_blk)
    assert (b_blk - b_mrg) >= (N_RES - 1) * elem * 0.5, \
        ('合并段比逐块段只省了 %d 字节，理论下限约 %d（少留 %d 个边界激活）'
         % (b_blk - b_mrg, (N_RES - 1) * elem, N_RES - 1))


def test_tap_position_is_that_blocks_output():
    """钉住 `tap_positions` 的语义：**该块的输出**，不是下一个块的输入。

    这是给 P4.2 挖的坑：P4.1 §7.3 的「第 1/5/9 号块」在 0-based 下标里对应
    `(0, 4)` 与 `(0,)`；如果实现取的是「下标 i 那个块的**输入**」，那么
    `tap_positions=(0,)` 拿到的是 **stem 的输出**（段入口）而不是 ResBlock #1 的
    输出 —— 语义完全错一个块，而两条张量形状一样、类型一样，`assert t.shape ==
    ...` 之类**抓不到**。所以这里用逐块 forward hook 把「块 i 的输出」录下来做
    逐位对照。
    """
    m = _small()
    _reset_bn(m)
    # ⚠ 必须先打破 He 零初始化：`ResBlock.__init__` 里有 `nn.init.zeros_(bn2.weight)`，
    # 于是**初值处残差块是恒等映射** ⇒ ResBlock #1 的输出与它的输入（stem 输出）
    # 逐位相同。不打破它，下面「抽头不是段入口」那条对照就恒真、整条测试空转。
    for blk in m.blocks:
        if isinstance(blk, ResBlock):
            nn.init.ones_(blk.bn2.weight)
    x0 = _x(batch=1, board=5, seed=63, ch=8)
    stem_out = F.relu(m.stem(x0))
    seg = m.blocks[RES_SLICE]
    m.train()

    outs = []
    handles = [blk.register_forward_hook(
        (lambda n: (lambda _m, _i, o: outs.append((n, tuple(o.shape)))))(i))
        for i, blk in enumerate(seg)]
    try:
        m.set_grad_checkpointing(False)
        torch.manual_seed(2)
        ref, taps = run_grad_segment(seg, (stem_out,), use_checkpoint=False,
                                     tap_positions=(S1_POS, S5_POS))
    finally:
        for h in handles:
            h.remove()
    assert len(outs) == N_RES, outs

    # 不开检查点、逐块手工前向，拿到「块 i 的输出」作参照
    cur = stem_out
    ref_outs = []
    for blk in seg:
        cur = blk(cur)
        ref_outs.append(cur)

    for got, pos, want_i in ((taps[0], S1_POS, 1), (taps[1], S5_POS, 5)):
        assert torch.equal(got, ref_outs[want_i - 1]), \
            ('tap_positions=%d 拿到的不是第 %d 块的输出' % (pos, want_i))
    # 非空转：确实**不是**段入口（否则与 s1 的对照就恒真了）
    assert not torch.equal(taps[0], stem_out), \
        'tap_positions=(0,) 拿到的是段入口（stem 输出），语义错了一个块'


def test_taps_survive_a_checkpointed_segment():
    """§2 的关键事实：抽头用「额外出参」跨段边界，不需要为它切开段。

    非空转判据：**抽头是带计算图的**（不是 detach 出来的快照）—— 从抽头出发的
    梯度必须能回到产生它的 stem 权重。（不能用 ResBlock 自己的 conv1 梯度做判据：
    `bn2.weight` 是 He 零初始化，`ResBlock` 的卷积分支梯度恒为 0。）
    """
    m = _small()
    _reset_bn(m)
    stem_out = F.relu(m.stem(_x(batch=1, board=5, seed=61, ch=8)))
    m.train()
    m.set_grad_checkpointing(True)
    out, taps = m.run_segment(m.blocks[RES_SLICE], (stem_out,), GC_RES,
                              tap_positions=(S1_POS, S5_POS))
    assert len(taps) == 2 and all(t.shape == stem_out.shape for t in taps)
    assert all(t.requires_grad and t.grad_fn is not None for t in taps), \
        '抽头没有保留计算图（被 detach 了？）'
    (out.sum() + sum(t.sum() for t in taps)).backward()
    g_stem = m.stem[0].weight.grad
    assert g_stem is not None and float(g_stem.abs().max()) > 0, \
        '从抽头出发的梯度没有回到 stem —— 抽头的反向路径被段边界切断了'


def test_out_of_range_tap_position_raises():
    m = _small()
    with pytest.raises(ValueError, match='越界'):
        m.run_segment(m.blocks[RES_SLICE], (torch.zeros(1, 24, 5, 5)), GC_RES,
                      tap_positions=(99,))


# ============================================================================ #
# 8. 旧 resnet / convnext 路径：能力可用、默认关
# ============================================================================ #
def test_legacy_path_available_but_default_off():
    m = SharedBackbone(in_channels=12, channels=32, num_res_blocks=4,
                       attention_mode='none', arch='resnet')
    assert m.use_checkpoint is False
    assert m.grad_checkpointing is False
    assert m.grad_checkpointing_for(GC_LEGACY) is False
    x = torch.randn(2, 12, 19, 19)
    m.train()
    m.set_grad_checkpointing(False)
    _reset_bn(m)
    torch.manual_seed(5)
    off = m(x)
    off[0].sum().backward()
    ref_bn = _snap_bn(m)

    m.set_grad_checkpointing(True)
    assert m.use_checkpoint is True, '历史属性 use_checkpoint 没同步'
    _reset_bn(m)
    torch.manual_seed(5)
    on = m(x)
    on[0].sum().backward()
    assert torch.equal(off, on)
    ok, why = _bn_equal(ref_bn, _snap_bn(m))
    assert ok, '旧路径的 BN 统计被重算污染：' + why


def test_legacy_use_checkpoint_attribute_is_live():
    """`use_checkpoint` 必须是**活**的：直接赋值也要真的开/关检查点。

    若它退化成 mixin 之外的普通 bool（`forward` 改读 `_gc_enabled`），那么
    `backbone.use_checkpoint = True` 会变成「读起来是 True、实际没开」的静默
    回退 —— 而这是既有代码里最常见的写法之一。

    判据用**留给反向的张量数**（`_saved_stats`）而不是 forward hook 次数：旧路径
    是**逐块**检查点（`per_block=True`，与 `checkpoint_sequential` 的既有粒度
    一致），于是每块都是「长度为 1 的段」，`early_stop` 会在单个块的前向**内部**
    就抛 `_StopRecomputationError` 中止重算 ⇒ forward hook 在重算那次**根本不
    触发**。实测：逐块粒度下 4 个块的 hook 计数全为 1，与不检查点完全一样。
    这本身是逐块粒度的一个可观测副作用（hook 看不见重算），也说明「用
    forward hook 数重算」只对合并段有效。
    """
    m = SharedBackbone(in_channels=12, channels=32, num_res_blocks=4,
                       attention_mode='none', arch='resnet')
    x = torch.randn(1, 12, 5, 5)
    m.train()

    def saved():
        for p in m.parameters():
            p.grad = None
        _reset_bn(m)
        m(x)[0].sum().backward()
        return _saved_stats(lambda: m(x))[1]

    m.use_checkpoint = False
    assert m.grad_checkpointing is False
    n_off = saved()

    m.use_checkpoint = True            # ← 既有代码的写法，不许静默失效
    assert m.grad_checkpointing is True, 'use_checkpoint 与 mixin 状态分叉了'
    assert m.grad_checkpointing_for(GC_LEGACY) is True
    n_on = saved()
    assert n_on < n_off, \
        'use_checkpoint=True 没有真的开检查点（留给反向的张量 %d -> %d）' % (n_off, n_on)


def test_legacy_default_construction_is_bitwise_unchanged():
    outs = []
    for flag in (False, True):
        torch.manual_seed(0)
        m = SharedBackbone(in_channels=12, channels=32, num_res_blocks=4,
                           attention_mode='none', arch='convnext')
        torch.manual_seed(7)
        outs.append(m(torch.randn(2, 12, 19, 19)))
    assert torch.equal(outs[0], outs[1])


def test_convnext_legacy_path_default_off():
    m = SharedBackbone(in_channels=12, channels=32, num_res_blocks=4,
                       attention_mode='none', arch='convnext')
    assert m.blocks[0].__class__ is ConvNeXtBlock
    assert not any(hasattr(b, 'bn1') for b in m.blocks)
    assert m.grad_checkpointing_for(GC_LEGACY) is False


# ============================================================================ #
# 9. 段内 BN 与 shape 校验
# ============================================================================ #
def test_run_grad_segment_multi_input_signature():
    """`CrossAttnRes` 那种 `(x, taps)` 双输入段。"""
    m = _small()
    _reset_bn(m)
    stem_out = F.relu(m.stem(_x(batch=1, board=5, seed=67, ch=8)))
    seg = m.blocks[CROSS_SLICE]
    m.train()
    m.set_grad_checkpointing(False)
    torch.manual_seed(9)
    a = run_grad_segment(seg, (stem_out, (stem_out,) * 3), use_checkpoint=False)
    torch.manual_seed(9)
    b = run_grad_segment(seg, (stem_out, (stem_out,) * 3), use_checkpoint=True)
    assert torch.equal(a[0], b[0]), \
        '双输入段开/关检查点不逐位相同 max|Δ|=%.3e' % (a[0] - b[0]).abs().max()
    assert a[1] == () and b[1] == ()


def test_no_grad_path_never_checkpoints():
    """`no_grad` 下**连 `torch.utils.checkpoint` 都不许被调用**（mixin 路径）。

    ⚠ P4.6b fix B3 更正了这里原先的自相矛盾：本测试**抓不到**
    `run_grad_segment` 里 `active = bool(use_checkpoint) and torch.is_grad_enabled()`
    的后半段 —— mixin 的 `grad_checkpointing_for()` 在更早的地方就用同一个
    `torch.is_grad_enabled()` 把 `use_checkpoint=False` 传下来了，所以删掉
    `active` 里的那一项，本测试依然全绿（实测：30 个测试无变化）。它守的是
    「经 `run_segment` 的路径在 no_grad 下零调用」这条**契约本身**；`active`
    里那一项是给**绕过 mixin 直接调 `run_grad_segment` 的公开调用者**兜底的，
    由下面的 `test_direct_call_under_no_grad_never_checkpoints` 钉住（那条
    才是能打红该变异的测试）。两个测试各守一条门，别再混为一谈。
    """
    m = _small()
    _reset_bn(m)
    x = _x(batch=1, board=5, seed=71, ch=8)
    m.train()
    m.set_grad_checkpointing(True)
    calls = []
    orig = torch.utils.checkpoint.checkpoint

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    torch.utils.checkpoint.checkpoint = spy
    try:
        with torch.no_grad():
            m(x)
        with torch.inference_mode():
            m(x)
    finally:
        torch.utils.checkpoint.checkpoint = orig
    assert calls == [], 'no_grad / inference_mode 下仍然调用了 checkpoint：%d 次' % len(calls)


def test_direct_call_under_no_grad_never_checkpoints():
    """B3 的**另一半**：绕过 mixin 直接调 `run_grad_segment` 时，
    `active = bool(use_checkpoint) and torch.is_grad_enabled()` 里的
    `torch.is_grad_enabled()` 是唯一的门（mixin 的预门不存在）。

    这是唯一能打红 `active = bool(use_checkpoint)` 变异的测试；上面那条
    `test_no_grad_path_never_checkpoints` 走 mixin 路径，抓不到它（B3）。
    对照组（grad 开着）必须走 checkpoint，否则本测试空转也绿。
    """
    m = _small()
    _reset_bn(m)
    m.train()
    seg = m.blocks[RES_SLICE]
    with torch.no_grad():
        stem_out = F.relu(m.stem(_x(batch=1, board=5, seed=73, ch=8)))
    calls = []
    orig = torch.utils.checkpoint.checkpoint

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    torch.utils.checkpoint.checkpoint = spy
    try:
        with torch.no_grad():
            run_grad_segment(seg, (stem_out,), use_checkpoint=True,
                             per_block=True, kind=GC_RES)
        assert calls == [], \
            '直调 + no_grad 仍调用了 checkpoint：%d 次' % len(calls)
        with torch.enable_grad():
            run_grad_segment(seg, (stem_out.detach().requires_grad_(),),
                             use_checkpoint=True, per_block=True, kind=GC_RES)
        assert calls, '对照组空转：grad 开着时没走 checkpoint，本测试无效'
    finally:
        torch.utils.checkpoint.checkpoint = orig


def test_legacy_granularity_is_pinned_to_checkpoint_sequential():
    """B2/M4：旧路径粒度 = **逐块 + 末块豁免**（= `checkpoint_sequential` 语义）。

    判据 = checkpoint 调用次数：N 个块必须恰好 N-1 次（每次一个块）。
    一次调用同时钉住两件事，两个变异各自都能打红它：
      * `GC_PER_BLOCK_DEFAULT = {}`（粒度被「顺手统一」成并段）→ 只调 1 次；
      * 删掉 `forward` 里的 `uncapped_last=True`（B1 回归）→ 调 N 次。
    这条测试守的是 31.12GB 锚点的显存口径（`search_arch.py:63` 的 k 标定基准）：
    粒度改动会把 v18 形状上「留给反向的字节」改 17×（50,274,304 vs 2,957,312），
    而锚点重测已被业主裁决取消 —— 口径必须逐字锁死。
    """
    m = SharedBackbone(in_channels=12, channels=32, num_res_blocks=4,
                       attention_mode='none', arch='resnet')
    n = len(m.blocks)
    assert n == 4
    m.train()
    m.set_grad_checkpointing(True)
    x = torch.randn(1, 12, 5, 5)
    calls = []
    orig = torch.utils.checkpoint.checkpoint

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    torch.utils.checkpoint.checkpoint = spy
    try:
        m(x)
    finally:
        torch.utils.checkpoint.checkpoint = orig
    assert len(calls) == n - 1, (
        '旧路径的 checkpoint 调用次数 %d != 块数-1=%d（逐块粒度或末块豁免被改）'
        % (len(calls), n - 1))
    # 常量面再钉一层（调用次数是对行为，常量是对默认表）
    assert GC_PER_BLOCK_DEFAULT[GC_LEGACY] is True, \
        'legacy 粒度默认表被改（并段会让保留量差 17×，k 失效）'
    assert V21_GRAD_CHECKPOINT_DEFAULTS[GC_LEGACY] is False, \
        'legacy 默认必须关（D5：不传 use_checkpoint 就是旧行为）'


def test_legacy_retention_bytes_equal_checkpoint_sequential():
    """B1 的口径证明：新 legacy 路径「留给反向的字节」与原实现**逐字节相等**。

    原实现（v19，`b262755~1` 的 forward）：
        torch.utils.checkpoint.checkpoint_sequential(
            self.blocks, len(self.blocks), out, use_reentrant=False)
    它的文档语义是「除最后一段外都检查点」。B1 用 `uncapped_last=True` 恢复
    同一语义 ⇒ 31.12GB 锚点（`search_arch.py:63`）与 k=0.8235 的标定基准继续
    有效（业主已裁决**取消锚点重测**，口径只能靠这条相等断言守住）。
    变异：删 `uncapped_last=True` → 留存字节变多 → 红。
    """
    m = SharedBackbone(in_channels=12, channels=32, num_res_blocks=4,
                       attention_mode='none', arch='resnet')
    m.train()
    x = torch.randn(1, 12, 5, 5)
    with torch.no_grad():
        stem_out = F.relu(m.bn1(m.conv1(x)))

    m.set_grad_checkpointing(True)
    _, c_new, b_new = _saved_stats(
        lambda: run_grad_segment(m.blocks, (stem_out,), use_checkpoint=True,
                                 per_block=True, uncapped_last=True,
                                 kind=GC_LEGACY))
    _, c_old, b_old = _saved_stats(
        lambda: torch.utils.checkpoint.checkpoint_sequential(
            m.blocks, len(m.blocks), stem_out, use_reentrant=False))
    assert (c_new, b_new) == (c_old, b_old), (
        '留存口径与原 checkpoint_sequential 不一致：新 (%d, %d) B vs 旧 (%d, %d) B '
        '—— 锚点/k 标定会失效' % (c_new, b_new, c_old, b_old))


def test_use_checkpoint_property_equals_effective_legacy_state():
    """B4：`use_checkpoint` 的读数必须 == `forward` 真正生效的状态。

    变异（B4 的原始现象）：getter 只读总开关 →
    `set_grad_checkpointing(True, legacy=False)` 时读数 True 而 legacy 段实际
    关着，`if self.use_checkpoint` 与属性读数分叉。
    """
    m = SharedBackbone(in_channels=12, channels=32, num_res_blocks=4,
                       attention_mode='none', arch='resnet')
    m.train()
    m.set_grad_checkpointing(True, legacy=False)
    assert m.grad_checkpointing is True
    assert m.grad_checkpointing_for(GC_LEGACY) is False
    assert m.use_checkpoint is False, \
        'B4 回归：读数 True 而 legacy 段实际不检查点'

    m.set_grad_checkpointing(True, legacy=True)
    assert m.grad_checkpointing_for(GC_LEGACY) is True
    assert m.use_checkpoint is True
    # 写路径往返（既有代码的写法）也必须一致
    m.use_checkpoint = False
    assert m.use_checkpoint is False and m.grad_checkpointing is False
    m.use_checkpoint = True
    assert m.use_checkpoint is True and m.grad_checkpointing_for(GC_LEGACY) is True


def test_mixin_precedes_nn_module_in_mro():
    """B5：mixin 必须排在 `nn.Module` **之前**（`run_segment` 等通用名防劫持）。"""
    for cls in (SharedBackbone, V21BackboneHarness, V21Net):
        mro = cls.__mro__
        assert mro.index(GradCheckpointMixin) < mro.index(nn.Module), \
            '%s 的 MRO 里 mixin 落到 nn.Module 之后（B5 回归）' % cls.__name__
    # 解析到 mixin 的实现（`V21Net` 例外：它**有意**覆写 run_segment 强制
    # use_checkpoint=False，是 switch-owner 测试的夹具，不算 B5）
    for cls in (SharedBackbone, V21BackboneHarness):
        assert cls.run_segment is GradCheckpointMixin.run_segment, \
            '%s.run_segment 没解析到 mixin 的实现' % cls.__name__
    assert 'run_segment' not in vars(nn.Module), \
        'nn.Module 新增了 run_segment —— 必须复查全部 mixin 挂载点（B5 的前提变了）'


def test_batchnorm_stat_guard_ignores_missing_buffers():
    bn = nn.BatchNorm2d(4, track_running_stats=False)
    g = _BatchNormStatGuard([bn])
    with g:
        pass
    assert bn.running_mean is None and bn.num_batches_tracked is None


# ============================================================================ #
# 10. 测量台（报告 §5 的数据来源）
# ============================================================================ #
def measure(batch=1, board=BOARD, channels=CH, in_channels=IN_CH,
            reps=3, warm=1, do_timing=True, do_mem=True, cfg_names=None):
    """返回逐配置的测量行（报告 §5 的数据来源）。

    三个口径，**不要混着比**：
    * `saved_tensors` / `saved_mb`：`torch.autograd.graph.saved_tensors_hooks`
      数「留给反向的张量」。这是梯度检查点**唯一**能精确控制的量，CPU 上比
      RSS 更准 —— 它不含反向的临时缓冲、分配器缓存与碎片。
    * `peak_fwd_mb` / `peak_fwd_bwd_mb`：private commit 的区间峰值增量（见
      `_private_bytes` 的口径说明），是**进程级上界**，含权重 + 梯度 + 临时缓冲。
    * `fwd_ms` / `fwd_bwd_ms`：min-of-`reps`。
    * `cfg_names`：只跑其中若干个配置（B=8 全跑要 ~7 min，用它切成两半，
      每条命令都能压在 8 分钟以内）。
    """
    m = V21BackboneHarness(in_channels=in_channels, channels=channels)
    _reset_bn(m)
    x = _x(batch=batch, board=board, seed=20260927, ch=in_channels)
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    configs = [
        ('off/all', {'res': False, 'mamba': False, 'transformer': False,
                     'cross_attn_res': False}),
        ('on/res', {'res': True, 'mamba': False, 'transformer': False}),
        ('on/mamba', {'res': False, 'mamba': True, 'transformer': False}),
        ('on/trans', {'res': False, 'mamba': False, 'transformer': True}),
        ('on/res+mamba', {'res': True, 'mamba': True, 'transformer': False}),
        ('on/res+trans', {'res': True, 'mamba': False, 'transformer': True}),
        ('on/mamba+trans', {'res': False, 'mamba': True, 'transformer': True}),
        ('on/all', {'res': True, 'mamba': True, 'transformer': True}),
        ('on/all+cross', {'res': True, 'mamba': True, 'transformer': True,
                          'cross_attn_res': True}),
    ]
    rows = []
    for name, kinds in configs:
        if cfg_names is not None and name not in cfg_names:
            continue
        m.set_grad_checkpointing(True, **kinds)
        _reset_bn(m)
        for p in params:
            p.grad = None

        def fwd():
            with torch.no_grad():
                m(x)

        def fwd_bwd():
            for p in params:
                p.grad = None
            torch.manual_seed(0)
            m(x).sum().backward()

        _, ns, bs = _saved_stats(lambda: m(x))
        row = {
            'cfg': name,
            'saved_tensors': ns,
            'saved_mb': round(bs / 2 ** 20, 3),
            'kinds': {k: v for k, v in m.grad_checkpointing_kinds().items()
                      if k != GC_LEGACY},
        }
        if do_timing:
            row['fwd_ms'] = round(_time(fwd, warm, reps), 1)
            row['fwd_bwd_ms'] = round(_time(fwd_bwd, warm, reps), 1)
        if do_mem:
            fwd()                      # 预热：把首次的惰性分配排除在区间外
            with _PeakMem() as pm:
                fwd()
            row['peak_fwd_mb'] = pm.delta_mb
            fwd_bwd()
            with _PeakMem() as pm:
                fwd_bwd()
            row['peak_fwd_bwd_mb'] = pm.delta_mb
        for p in params:
            p.grad = None
        rows.append(row)
    m.set_grad_checkpointing(False, res=False, mamba=False, transformer=False,
                             cross_attn_res=False)
    return rows


def test_measurement_ordering_is_stable():
    """非空转：真实尺寸（B=1 / 19 路 / 184 通道）下 Mamba 段的激活远大于 ResBlock 段。

    这是报告 §5 用来**反驳**「ResBlocks 占 74% 参数 ⇒ 必须开」那条推理的数据：
    参数量与激活量在 v21 里几乎不相关。
    """
    rows = measure(batch=1, board=BOARD, channels=CH, in_channels=IN_CH,
                   do_timing=False, do_mem=False)
    by = {r['cfg']: r for r in rows}
    base = by['off/all']['saved_mb']
    saved = {k: round(base - by['on/' + k]['saved_mb'], 2)
             for k in ('res', 'mamba', 'trans')}
    assert saved['mamba'] > 5 * saved['res'], \
        'Mamba 段省下的激活没有远超 ResBlock 段：%s（比例 %.1f×）' \
        % (saved, saved['mamba'] / max(saved['res'], 1e-9))
    assert min(saved.values()) > 0, saved
    assert by['on/all']['saved_mb'] < by['off/all']['saved_mb']


if __name__ == '__main__':
    import json
    torch.set_num_threads(int(os.environ.get('P46B_THREADS',
                                            max(1, (os.cpu_count() or 4) // 2))))
    only = os.environ.get('P46B_ONLY') or None
    print(json.dumps(measure(batch=int(sys.argv[1]) if len(sys.argv) > 1 else 1,
                             reps=int(sys.argv[3]) if len(sys.argv) > 3 else 3,
                             do_timing=(len(sys.argv) < 3 or sys.argv[2] != 'notime'),
                             do_mem=(len(sys.argv) < 4 or sys.argv[3] != 'nomem'),
                             cfg_names=only.split(',') if only else None),
                      indent=1, ensure_ascii=False), flush=True)


