"""损失函数的**主机同步预算**：`forward` 里 `bool(tensor)` 的次数。

事故（2026-10-11 A100 实测）
--------------------------------
段 2 训练实测每 optimizer step 墙钟 665 ms，拆解 `c=328 / d=192 / u=146`
（`t_comp_ms` 只统计 CPU 发射时间、不含设备执行）。逐处数下来发现
**`KataGoV7Loss.forward` 每次调用有 28 次 `bool(tensor)`** ——
`_weighted_mean` 里每项两处（`bad_w.any()` / `zero.any()`）× 13 项 = 26，
外加收尾几处。256×8 ⇒ **每 optimizer step 224 次**。

每次 `bool(tensor)` 在设备上都是**把流水线排空**的阻塞读回（A100 实测
~0.6 ms）。这些等待发生在损失 forward 内部 ⇒ 落在 `c` 里，表现为
「发射时间莫名偏长」，而 CPU 每次被阻塞后都没法提前入队下一批算子。

修法（不改变任何数值）
----------------------
1. `_weighted_mean` 的两处净化改成**无条件执行**。依据是一条可验证的事实：
   **mask 全 False 时 `torch.where` 原样返回**，前向逐位相同、反向梯度也
   逐位相同（`where` 反向按 mask 分流，全 False 时等于 `grad * 1.0`）。
2. probe 计数从「就地 `int(...)`」改成**延迟到 forward 末尾统一读回**，
   且每个键存 `[计数, 存在标志]` —— 见 `_resolve_probe` 的 docstring，
   **键在不在由标志位决定，不是「计数 > 0」**。
3. 收尾两处 `isfinite(total)` 合成一次。

实测 37 → 12 次/micro-batch（正常数据 4 次；剩下 12 次在「总损失已非有限」
的慢路径里逐项点名 —— 那条路本来就要响)。

本文件钉三件事
--------------
1. **同步预算**：正常数据每次 forward 的 `bool(tensor)` ≤ 4。
2. **数值逐位不变**：前向、逐项、反向梯度、以及 `sanitized_rows` /
   `nonfinite_terms` / `total_finite`（含**类型**）全部与旧写法一致 ——
   对拍用的是「字节级相等」，因为 `torch.equal` 遇 NaN 返回 False，
   而这些用例里 NaN 恰恰是主要内容。
3. **坏数据路径仍然被点名**：净化/诊断能力没被优化掉。

跑：pytest tests/test_katago_v7_loss_syncs.py -v
"""
import contextlib
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from src.networks.katago_v7_loss import KataGoV7Loss, _resolve_probe  # noqa: E402
from test_katago_v7_loss import _labels, _out                        # noqa: E402

NAN = float('nan')
INF = float('inf')


@contextlib.contextmanager
def _count_bool():
    """数 `bool(tensor)` 的调用次数。

    为什么数个数而不是时间：CPU 上 `bool(tensor)` 只是一次读，几乎免费；
    真机上它是**排空设备流水线**的一次阻塞读回，而那在 CPU 上原理上不可测。
    「次数」是平台无关、可被变异测试证伪的代理量。
    """
    n = {'c': 0}
    raw = torch.Tensor.__bool__

    def counting(s, *a, **k):
        n['c'] += 1
        return raw(s, *a, **k)
    torch.Tensor.__bool__ = counting
    try:
        yield n
    finally:
        torch.Tensor.__bool__ = raw


def _bitwise(a, b):
    """字节级相等。`torch.equal` 对 NaN 返回 False（NaN != NaN）——
    而这里的用例里 NaN 恰恰是被检视的主体，用它比会得到一堆假阳性。"""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().contiguous().reshape(-1).view(torch.uint8))


def test_normal_forward_stays_within_sync_budget():
    """正常（无坏数据）时每次 forward 的 `bool(tensor)` ≤ 4。

    旧实现是 28。阈值给到 4 是因为剩下几处分散在 score_stdev 的目标构造与
    seki EMA 上，各 1 次；真要再压就该单独评估收益，而不是顺手改。
    """
    with _count_bool() as n:
        KataGoV7Loss(action_size=362)(
            {k: v.clone() for k, v in _out().items()}, _labels())
    assert n['c'] <= 4, (
        f'正常数据一次 forward 有 {n["c"]} 次 bool(tensor)（预算 4）。'
        f'每次都是排空设备流水线的一次阻塞读回；256x8 下乘以 8。')


def test_sanitizing_a_bad_weight_does_not_reintroduce_per_item_syncs():
    """坏权重路径也必须在预算内 —— 它是**最需要**诊断的那条路。"""
    lbl = dict(_labels(seed=3))
    w = {k: torch.ones(4) for k in lbl['w']}
    w['policy_opp'] = torch.tensor([1., 0., 1., INF])      # 坏权重
    lbl['w'] = w
    with _count_bool() as n:
        res = KataGoV7Loss(action_size=362)(
            {k: v.clone() for k, v in _out(seed=4).items()}, lbl)
    assert n['c'] <= 4, f'坏权重路径有 {n["c"]} 次 bool(tensor)（预算 4）'
    # ⚠ 诊断没被优化掉：坏权重必须仍被点名
    assert res['sanitized_rows'] == {'policy_opp:w_bad': 1}, res['sanitized_rows']


@pytest.mark.parametrize('p,w,expect', [
    # 有限 p + 全非零 w：什么都不该记
    ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], {}),
    # p 含 NaN 但**不**落在零权重行上 ⇒ 旧守卫 `zero.any() and not
    # isfinite(p).all()` 不成立 ⇒ 不记。这条最容易在重构里写反。
    ([1.0, NAN, 3.0], [1.0, 1.0, 1.0], {}),
    # NaN 落在零权重行上 ⇒ 记 1
    ([NAN, 2.0, 3.0], [0.0, 1.0, 1.0], {'t': 1}),
    # 🔴 **计数为 0 但键必须存在** —— 这一条是「按计数>0 过滤」的唯一判据，
    #    少了它，下面那个变异（`if cnt:` 取代 `if has:`）就抓不到。
    #    形状：存在零权重行（index 1）、p 的 NaN 在**非**零权重行（index 0）
    #    ⇒ 计数 `(zero & bad).sum()` = 0，而守卫 `zero.any() and
    #    not isfinite(p).all()` 成立 ⇒ 键在、值为 0。
    #    这不是构造出来的边角：真机实测就有 `{'score_stdev': 0}`。
    ([NAN, 2.0, 3.0], [1.0, 0.0, 1.0], {'t': 0}),
    # 零权重行存在但 p 有限 ⇒ **不**记（`not isfinite(p).all()` 为假）
    ([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], {}),
    # 坏权重 ⇒ 记 w_bad
    ([1.0, 2.0, 3.0], [1.0, INF, 1.0], {'t:w_bad': 1}),
])
def test_probe_key_presence_matches_the_original_guard(p, w, expect):
    """`sanitized_rows` 的**键存在性**逐条还原旧守卫。

    「键在不在」不等于「计数 > 0」：旧 `probe[tag]` 的条件是
    ``zero.any() and not isfinite(p).all()``，于是
    ``{'score_stdev': 0}``（计数 0 但键存在）是真实可能的返回值。
    按计数过滤会把它悄悄抹掉 —— 而这是训练循环逐项上报的诊断，抹掉等于丢证据。
    """
    from src.networks.katago_v7_loss import _weighted_mean
    probe = {}
    _weighted_mean(torch.tensor(p), torch.tensor(w), probe, 't')
    assert _resolve_probe(probe) == expect


def test_total_finite_is_a_python_bool_not_an_int():
    """`total_finite` 必须是 **bool**。

    它是合并读回时最容易退化的一个：`.tolist()` 返回 0/1，`0 == False` 让
    等值断言全部照过，但任何 `is True` / `type(...) is bool` 的下游就悄悄坏了。
    """
    res = KataGoV7Loss(action_size=362)(
        {k: v.clone() for k, v in _out().items()}, _labels())
    assert type(res['total_finite']) is bool


def test_forward_is_bitwise_identical_to_the_reference_implementation():
    """数值与梯度**字节级**不变（含坏数据路径）。

    参照实现是「净化按旧守卫条件执行 + 就地 int()」。两条容易悄悄坏掉的：
      * 无条件 `torch.where` 改变了前向/反向的图结构（实测逐位相同，但要钉住）；
      * 延迟读回改变了 `sanitized_rows` 的键存在性与 `total_finite` 的类型。
    """
    from src.networks.katago_v7_loss import (
        W_WEIGHT_CEILING, W_WEIGHT_DEFAULT, _weighted_mean)

    def reference(p, w, probe):
        """旧写法：净化只在守卫成立时执行，计数就地 int()。"""
        bad = ~torch.isfinite(w) | (w > W_WEIGHT_CEILING)
        if bool(bad.any()):
            probe['x:w_bad'] = int(bad.sum())
            w = torch.where(bad, torch.full_like(w, W_WEIGHT_DEFAULT), w)
        zero = (w == 0)
        if bool(zero.any()):
            clean = torch.where(torch.isfinite(p), p,
                                torch.zeros((), dtype=p.dtype))
            if not bool(torch.isfinite(p).all()):
                probe['x'] = int((zero & ~torch.isfinite(p)).sum())
            p = torch.where(zero, clean, p)
        return (p * w).mean()

    cases = [([1., 2., 3.], [1., 1., 1.]),
             ([1., NAN, 3.], [1., 1., 1.]),
             ([NAN, 2., 3.], [0., 1., 1.]),
             ([1., 2., 3.], [0., 1., 1.]),
             ([1., 2., 3.], [1., INF, 1.])]
    for p, w in cases:
        pt, wt = torch.tensor(p), torch.tensor(w)
        a = pt.clone().requires_grad_(True)
        b = pt.clone().requires_grad_(True)
        ra, rb = {}, {}
        ref = reference(a, wt.clone(), ra)
        new = _weighted_mean(b, wt, rb, 'x')
        assert _bitwise(ref.detach(), new.detach()), (p, w)
        g = torch.randn(1, generator=torch.Generator().manual_seed(0))
        (ref * g).sum().backward()
        (new * g).sum().backward()
        assert _bitwise(a.grad, b.grad), ('梯度不同', p, w)
        assert ra == _resolve_probe(rb), ('probe 不同', p, w, ra,
                                        _resolve_probe(rb))