"""EMA.update() 必须走 foreach 批量算子。

原实现是逐参数 Python 循环，每参数 2 次独立设备 kernel 启动：

    for name, param in self.model.named_parameters():
        self.shadow[name].data.mul_(d).add_(param.data, alpha=1 - d)

本模型 depth 很深（backbone 17 + res 8 + convnext 4 + attn 5
+ value 8 + policy 3），参数张量达数百个，即每步数百次 kernel 启动。
Ascend 的 ACL 单次启动开销明显高于 CUDA，累积起来不可忽略。
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_sft import EMA


class _Net(torch.nn.Module):
    """刻意含 bias=False / 多层，确保参数张量数量 > 1 且形状各异。"""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 6)
        self.c = torch.nn.Linear(6, 3, bias=False)

    def forward(self, x):
        return self.c(self.b(self.a(x)))


def _all_params(model):
    return dict(model.named_parameters())


def test_foreach_matches_loop_semantics():
    """foreach 路径与逐参数循环必须给出完全相同的 shadow。"""
    torch.manual_seed(0)
    m1, m2 = _Net(), _Net()
    m2.load_state_dict(m1.state_dict())
    decay = 0.9
    e_loop, e_foreach = EMA(m1, decay), EMA(m2, decay)
    # 先手工走循环路径
    for name, param in m1.named_parameters():
        e_loop.shadow[name].data.mul_(decay).add_(param.data, alpha=1 - decay)
    # 再让 e_foreach.update() 自己选路径
    e_foreach.update()
    for n in e_loop.shadow:
        assert torch.allclose(e_loop.shadow[n], e_foreach.shadow[n], atol=1e-7), \
            f'{n}: foreach 与逐参数循环结果不一致'


def test_foreach_actually_used_when_available():
    """有 foreach 时必须走批量路径，否则这项性能改动等于没做。"""
    import scripts.train_sft as t
    assert t._HAS_FOREACH, '当前 torch 应支持 _foreach_mul_/_foreach_add_'
    m = _Net()
    ema = EMA(m, 0.9)
    calls = []
    orig_mul, orig_add = torch._foreach_mul_, torch._foreach_add_
    torch._foreach_mul_ = lambda ts, v: (calls.append('mul'), orig_mul(ts, v))[1]
    torch._foreach_add_ = lambda ts, ps, **kw: (
        calls.append('add'), orig_add(ts, ps, **kw))[1]
    try:
        ema.update()
    finally:
        torch._foreach_mul_, torch._foreach_add_ = orig_mul, orig_add
    assert calls == ['mul', 'add'], f'未走 foreach 批量路径，实际调用 {calls}'


def test_foreach_is_not_a_noop():
    """批量调用后 shadow 必须真的变了。"""
    m = _Net()
    ema = EMA(m, 0.9)
    before = {n: s.detach().clone() for n, s in ema.shadow.items()}
    ema.update()
    unchanged = [n for n, s in ema.shadow.items()
                 if torch.equal(s, before[n])]
    assert not unchanged, f'以下参数的 shadow 未被更新: {unchanged}'


def test_foreach_respects_decay():
    """shadow = decay*old + (1-decay)*param 的语义必须保持。"""
    m = _Net()
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(1.0)
    ema = EMA(m, 0.5)
    with torch.no_grad():
        for s in ema.shadow.values():
            s.fill_(0.0)
    ema.update()
    for n, s in ema.shadow.items():
        assert torch.allclose(s, torch.full_like(s, 0.5)), f'{n}: decay 语义被破坏'


def test_update_under_no_grad():
    """update 必须在 no_grad 下运行，不得污染计算图/参数梯度。"""
    m = _Net()
    ema = EMA(m, 0.9)
    ema.update()
    for n, p in _all_params(m).items():
        assert p.grad is None, f'{n}: update() 意外产生了梯度'
    for n, s in ema.shadow.items():
        assert not s.requires_grad, f'{n}: shadow 不应 requires_grad'


def test_shadow_replacement_still_works():
    """resume 会整体替换 ema.shadow，update 必须跟着用新的 shadow。"""
    m = _Net()
    ema = EMA(m, 0.9)
    new_shadow = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    ema.shadow = new_shadow
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(1.0)
    ema.update()
    for n, s in ema.shadow.items():
        assert torch.allclose(s, torch.full_like(s, 0.1)), \
            f'{n}: 未使用替换后的 shadow（可能缓存了失效张量）'
