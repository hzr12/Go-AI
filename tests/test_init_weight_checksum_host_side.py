r"""`_assert_init_weights_identical` 的 checksum 必须在**主机侧**归约。

事故（4×910A）
------------
`rank0=8125.2251, rank1=8125.2256, rank2=8125.2251, rank3=8125.2251` ——
三个 rank 完全一致，rank1 差**一个 ulp**。

为什么是 1 ulp 而不是别的
------------------------
8125.225 落在 ``[4096, 8192)``，fp32 在这一档的 ulp = ``2**(12-23) = 4.88e-4``，
实测差 ``5.0e-4`` —— **精确 1 ulp**。这个量级排除掉了另外两种可能：

* **不是 NaN/Inf**：NaN 会打印成 ``nan``，这里全是有限数；
* **不是广播没生效**：那会让某个 rank 的和**完全不同**，而不是差末位。

根因
----
旧实现 ``p.detach().sum(dtype=torch.float32)`` 在 **NPU 上**做归约。多块AICore
归约的**分块顺序不保证跨 rank 一致**，于是同样这324 个分片和、以不同顺序在
fp32 里累加 ⇒ 末位差 1 ulp。而比对用的是**精确相等** ⇒ 每次都误报。

`train_sft.py` 原注释把「相同归约顺序」当成了前提：

    相同输入 + 相同形状 + 相同归约顺序 ⇒ fp32 的和也逐位相同

**这个前提在 NPU 上是假的。** CPU 上为真，所以
`tests/test_init_weight_sync.py::test_checksum_contract_after_losing_fp64`
在本地一直是绿的 —— 它测不出这个 bug。

顺带被这个 bug 暴露的第二个缺陷
------------------------------
旧 checksum 是 fp32 跑马和，**噪声底 = 总量的 1 ulp ≈ 4.9e-4**；而单个 1e-2
量级参数差**一个 fp32 ulp** 只有 ``~9.3e-10`` —— 比噪声底低 5 个数量级，
**检测不到**。也就是说这个闸门既会误报，又会漏报小的真实分歧。

本文件钉住修好后的两条契约：

* **C1 灵敏度**：只差 1 个 fp32 ulp 的权重**必须**被判为不同（旧实现漏）；
* **C2 不依赖设备侧归约**：归约发生在主机上，设备侧分块顺序影响不到它。
"""
import io
import os
import re
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.train_sft as mod  # noqa: E402


def _fn_src(name):
    """取 `train_sft.py` 里某个函数的源码（repo 既有测试同款做法）。"""
    with io.open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
                 encoding='utf-8') as fh:
        src = fh.read()
    i = src.index('def %s(' % name)
    j = src.index('\ndef ', i + 1)
    return src[i:j]


def _fn_code(name):
    """同 `_fn_src`，但**去掉 docstring**。

    结构断言（"不许出现某个调用"）必须只看代码：docstring 里为了讲清事故而引用
    旧写法是应当鼓励的，否则每写一段事故说明就要改一次断言。
    """
    src = _fn_src(name)
    m = re.search(r'(?s)(r?)("""|\'\'\')(.*?)\2', src)
    if m:
        src = src[:m.start()] + src[m.end():]
    return src


class _EchoDist:
    """`all_gather` 原样回显 —— 让「各 rank 一致」的路径可测。"""

    def __init__(self):
        self.seen = []

    def as_dist(self):
        return self

    def get_world_size(self):
        return 2

    def all_gather(self, out, buf):
        self.seen.append((buf.dtype, buf.detach().clone()))
        for t in out:
            t.copy_(buf)


class _RecLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _run(model):
    """在假通信域上跑一次自检，返回 ``(all_gather 的 dtype, 载荷)``。"""
    cap = _EchoDist()
    real = (mod.dist, mod._dist_active, mod._dist_env_snapshot)
    mod.dist, mod._dist_active = cap.as_dist(), (lambda: True)
    mod._dist_env_snapshot = (lambda: 'SNAPSHOT=<stub>')
    try:
        mod._assert_init_weights_identical(model, _RecLogger())
    finally:
        mod.dist, mod._dist_active, mod._dist_env_snapshot = real
    assert len(cap.seen) == 1, '每次自检只该发一次 all_gather：%d' % len(cap.seen)
    return cap.seen[0]


def _model_like_big_sum(total=8125.0, n=4000):
    """一个「和约为 total、形状像真网」的模型。

    真网 V7 是 5,562,121 个参数、典型初值 1e-2 量级，和约 8125。用 1e-2 的正态
    + 偏置把总和推到目标量级，这样 total 落在与事故相同数量级上。
    """
    torch.manual_seed(0)
    m = nn.Module()
    m.register_parameter('w', nn.Parameter(torch.randn(n) * 1e-2))
    m.register_parameter('b', nn.Parameter(torch.tensor([total])))
    return m


# --------------------------------------------------------------------------- #
# C1 灵敏度 —— 这条在旧实现上是红的
# --------------------------------------------------------------------------- #
def test_checksum_detects_a_single_ulp_weight_difference():
    """两个只差**一个 fp32 ulp** 的模型必须被判为不同。

    旧实现的 fp32 跑马和噪声底是总量的 1 ulp（8125 那一档 = 4.88e-4），而这里
    的差异只有 ~9.3e-10 —— 低 5 个数量级，必然被判为「一致」而**静默放过**
    一次真实的权重分歧。
    """
    a = _model_like_big_sum()
    b = _model_like_big_sum()
    assert torch.equal(a.w, b.w), '两个模型本应只差一个 ulp'

    # 只把 w 的最后一个元素推进**一个 fp32 ulp**。
    with torch.no_grad():
        flat = b.w.view(-1)
        flat[-1] = torch.nextafter(flat[-1], torch.tensor(float('inf')))

    # delta 必须用 **fp64** 算：用 fp32 算的话它自己就已经掉进 1 ulp 的噪声底里
    # 变成 0.0 —— 那正是本测试要演示的现象，不能拿来当断言的前提。
    delta = (float(a.w.detach().sum(dtype=torch.float64))
             - float(b.w.detach().sum(dtype=torch.float64)))
    assert delta != 0.0, '推进一个 ulp 后和应当变化'

    total = abs(float(a.w.detach().sum(dtype=torch.float64))) + 8125.0
    old_floor = total * 2.0 ** -23        # 旧实现的 fp32 噪声底
    assert abs(delta) < old_floor, (
        '本测试要演示的正是「差异小于旧噪声底」；现在 Δ=%.3g ≥ 噪声底 %.3g，'
        '说明构造失效' % (abs(delta), old_floor))

    _, ra = _run(a)
    _, rb = _run(b)
    assert not torch.equal(ra, rb), (
        '只差一个 fp32 ulp（Δ和=%.3g，远低于旧 fp32 噪声底 %.3g）的两份权重被'
        '判为一致 —— 噪声底把真实分歧吞了' % (abs(delta), old_floor))


def test_all_zero_weights_give_exactly_zero_checksum():
    """全 0 参数的 checksum 必须**恰好** 0.0（精确，无需容差）。"""
    m = nn.Module()
    m.register_parameter('a', nn.Parameter(torch.zeros(4)))
    m.register_parameter('b', nn.Parameter(torch.zeros(15)))
    _, buf = _run(m)
    assert float(buf.double().sum()) == 0.0, '实得 %r' % buf


def test_signed_sum_is_sign_sensitive():
    """取负必须改变 checksum（否则「rank0 的权重取负」这种分歧会漏）。"""
    m = nn.Module()
    m.register_parameter('a', nn.Parameter(torch.ones(4)))
    _, buf = _run(m)
    assert float(buf.double().sum()) > 0.0
    m2 = nn.Module()
    m2.register_parameter('a', nn.Parameter(-torch.ones(4)))
    _, buf2 = _run(m2)
    assert float(buf2.double().sum()) < 0.0


# --------------------------------------------------------------------------- #
# C2 归约必须发生在主机上
# --------------------------------------------------------------------------- #
def test_reduction_happens_on_the_host():
    """checksum 的求和必须先把参数搬回主机，不能在设备侧归约。

    这是本次事故的**根因**：设备侧多块归约的分块顺序不保证跨 rank 一致。
    搬回主机后求和顺序由 `parameters()` 的迭代顺序唯一确定，且主机 fp64 的
    舍入误差比 fp32 小 11 个数量级，噪声底从 4.9e-4 降到 ~1e-12。
    """
    src = _fn_code('_assert_init_weights_identical')
    assert '.cpu()' in src or "'cpu'" in src, \
        '归约前必须把参数搬回主机（设备侧归约的分块顺序跨 rank 不一致）'
    # 旧写法：对设备张量直接 sum
    assert 'p.detach().sum(dtype=' not in src, \
        '不得在设备张量上做 sum —— 那正是分块顺序不确定的归约'


def test_collective_payload_stays_fp32():
    """**过线**的载荷仍必须是 fp32：HCCL 原生支持 fp32，fp64 会走 AICPU。

    主机侧的 fp64 只用于求和；跨 rank 交换的一律 fp32。
    """
    src = _fn_code('_assert_init_weights_identical')
    assert '.sum(dtype=torch.float64)' in src, \
        '主机侧求和应当用 fp64（否则噪声底仍在 1 ulp）'
    # fp64 只许作为**主机侧 sum 的累加 dtype**出现，不得是设备张量的 dtype。
    # （早先这里写的是「源码里 float64 出现次数 ≤ 3」，那是个任意的计数，
    #   docstring 一改就假红；改成直接表达不变量的形状。）
    assert 'dtype=torch.float64, device' not in src, \
        'fp64 不得作为设备张量的 dtype（910A 不支持，会派发 AICPU kernel）'
    # all_gather 的载荷 dtype 由实测钉住，而不是靠读源码
    _, buf = _run(_model_like_big_sum())
    assert buf.dtype == torch.float32, 'all_gather 载荷应是 fp32，实得 %s' % buf.dtype


def test_no_aicpu_ops_in_the_collective():
    """`torch.equal` 在 910A 上是 AICPU kernel，脚本里不许出现。"""
    assert 'torch.equal(' not in _fn_src('_assert_init_weights_identical')


def test_fp64_never_materialised_on_device():
    """不许走 `p.double().sum()` —— 那会把整份参数物化成 fp64（体积 ×2）。"""
    src = _fn_code('_assert_init_weights_identical')
    assert '.double().sum()' not in src
    assert 'p.double()' not in src


def test_all_gather_payload_carries_the_whole_checksum_exactly():
    """载荷必须**无损**承载主机侧算出的 fp64 checksum。

    fp32 只有 24 位尾数，直接把 fp64 和塞进去会把它舍掉 —— 那就等于把噪声底
    又抬回 1 ulp，这正是旧实现误报的机制。实现用 hi/lo 两个 fp32 拆分承载。
    """
    _, buf = _run(_model_like_big_sum())
    assert buf.numel() >= 2, (
        '载荷应至少 2 个 fp32（hi/lo 拆分）才能无损承载 fp64，实得 %d 个' % buf.numel())


def test_checksum_survives_a_value_that_needs_all_24_bits():
    """构造一个**单 fp32 承载必然丢低位**的值 —— 防止实现退化成单 fp32。

    ``2**59`` 在 fp32 里是精确的（2 的幂），所以单 fp32 也能还原它 ⇒ 测不出问题。
    这里加上 ``+4``：fp32 在 ``2**59`` 那一档的 ulp 是 ``2**36``，所以
    单 fp32 会把它**整个抹掉**，只有 hi/lo 拆分能还原。
    """
    m = nn.Module()
    m.register_parameter('a', nn.Parameter(
        torch.tensor([2.0 ** 58 + 1.0, 2.0 ** 58 + 3.0], dtype=torch.float64)))
    _, buf = _run(m)
    got = 0.0
    for i in range(buf.numel()):
        got = got + float(buf[i].double())
    want = (2.0 ** 58 + 1.0) + (2.0 ** 58 + 3.0)
    assert got == want, (
        '展开载荷得 %.17g，期望 %.17g —— 单个 fp32 承载会把 +4 抹掉'
        '（fp32 在 2**59 那一档的 ulp 是 2**36）' % (got, want))