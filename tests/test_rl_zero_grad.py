"""RL 训练循环梯度清零回归测试（train_epochs 跨 step 梯度污染）。

覆盖:
  - train_epochs 返回后所有参数 .grad 为 None 或全零（accum=1 与 accum=2）
  - 多 step 训练：对照「每次 step 后清零」的参考轨迹，最终权重必须逐位一致
    （旧实现 step 后从不清零，第 2 个 step 起梯度叠加陈旧残量 → 必红）
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.selfplay_train import train_epochs
from tests.test_grad_accum import _args, _buffer, _make_ai


def _assert_grads_cleared(model):
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        assert torch.count_nonzero(p.grad) == 0, f"参数 {name} 残留非零梯度"


def test_grads_cleared_after_train_epochs():
    """accum=1: n=16/batch=4/epochs=1 共 4 个 step，返回后梯度已清零。"""
    buf = _buffer(n=16, seed=0)
    args = _args(batch_size=4, epochs=1, grad_accum_steps=1)
    ai = _make_ai(seed=0)
    train_epochs(ai, buf, args, 'cpu')
    _assert_grads_cleared(ai.model)


def test_grads_cleared_accum2():
    """accum=2: 同上但每 2 个 micro-batch 才 step，返回后梯度同样已清零。"""
    buf = _buffer(n=16, seed=0)
    args = _args(batch_size=4, epochs=1, grad_accum_steps=2)
    ai = _make_ai(seed=0)
    train_epochs(ai, buf, args, 'cpu')
    _assert_grads_cleared(ai.model)


def test_multi_step_no_stale_accumulation():
    """关键回归：step 后不清单会让第 2 个 step 起的梯度叠加陈旧残量。

    参考轨迹 = 包一层 AdamW.step、每次 step 后手工 zero_grad（即修复后的
    正确语义）。同 seed 下待测路径与参考轨迹的最终 state_dict 必须逐位一致。
    """
    buf = _buffer(n=16, seed=0)
    args = _args(batch_size=4, epochs=1, grad_accum_steps=1)

    # 待测路径：直接跑 train_epochs
    np.random.seed(1234)
    ai1 = _make_ai(seed=0)
    train_epochs(ai1, buf, args, 'cpu')

    # 参考路径：每次 step 后强制清零
    orig_step = torch.optim.AdamW.step

    def _step_then_clear(self, *a, **k):
        out = orig_step(self, *a, **k)
        self.zero_grad()
        return out

    np.random.seed(1234)
    ai2 = _make_ai(seed=0)
    torch.optim.AdamW.step = _step_then_clear
    try:
        train_epochs(ai2, buf, args, 'cpu')
    finally:
        torch.optim.AdamW.step = orig_step

    s1 = ai1.model.state_dict()
    s2 = ai2.model.state_dict()
    assert s1.keys() == s2.keys()
    for k in s1:
        assert torch.equal(s1[k], s2[k]), \
            f"参数 {k} 两次轨迹不一致：存在跨 step 梯度污染"
