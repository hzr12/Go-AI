# -*- coding: utf-8 -*-
"""`_dense_move_target` 的设备无关性（2026-10-04 云端 910A 实测崩）。

崩溃原文
--------
```
File "scripts/train_sft.py", line 845, in v7_loss_labels
    lbl['policy_player'] = _dense_move_target(moves, action_size)
File "scripts/train_sft.py", line 802, in _dense_move_target
    mv = torch.as_tensor(np.asarray(moves)).reshape(-1).to(torch.long)
TypeError: can't convert npu:0 device type tensor to numpy.
```

为什么本机 CPU 冒烟**永远发现不了**
------------------------------------
`np.asarray()` 对 **CPU** 张量是可用的（numpy 消费它的 `__array__`），只有
设备张量（NPU/CUDA）才抛。而调用方传的是 `move_t`，它是
`torch.from_numpy(...).to(device)` 的结果 ⇒ CPU 上是 CPU 张量、恰好能过。

所以「本地 1900+ 测试全绿 + CPU 冒烟全绿」与「真机 910A 一步都走不动」
可以同时成立。这条测试用**打掉 numpy 转换**的方式在 CPU 上复现那个前提，
而不是指望有 NPU。

修法为什么不是 `.cpu()`
----------------------
`.cpu()` 也能让 numpy 转换成功，但 one-hot 目标会被搬回主机、再在 loss 里
搬回设备 ⇒ 每步多两次 H2D/D2H（`(B, 362)` × 2 项）。直接在设备上 `one_hot`
既修好崩溃，又省掉搬运 —— 这也是本文件钉住 `out.device == moves.device` 的原因。
"""
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from train_sft import _dense_move_target, v7_loss_labels  # noqa: E402

A = 362


# --------------------------------------------------------------------------- #
# 核心前提：Tensor 输入不许走 numpy
# --------------------------------------------------------------------------- #
def test_tensor_input_never_touches_numpy(monkeypatch):
    """ 给 Tensor 时**不许**调 `np.asarray` —— 那正是 NPU 崩溃的那一行。

    用「把 numpy 转换打成会炸的」在 CPU 上复现 NPU 的前提：CPU 张量本来
    能被 numpy 消费，不这样就没法在无 NPU 的机器上钉住这条。
    """
    def _boom(x):
        if isinstance(x, torch.Tensor):
            raise TypeError(
                "can't convert device type tensor to numpy "
                "(模拟 npu:0 上的真实失败)")
        return np.asarray(x)

    monkeypatch.setattr(np, 'asarray', _boom)
    moves = torch.tensor([5, 7, -1, 362], dtype=torch.long)
    out = _dense_move_target(moves, A)          # 不许抛
    assert out.shape == (4, A)
    assert out.dtype is torch.float32


def test_numpy_input_still_works(monkeypatch):
    """ 修设备问题不能**弄坏** numpy 输入那条路（dataset 侧给的就是 numpy）。"""
    real = np.asarray

    def _boom(x):
        if isinstance(x, torch.Tensor):
            raise TypeError('numpy 路不许被 tensor 触发')
        return real(x)

    monkeypatch.setattr(np, 'asarray', _boom)
    out = _dense_move_target(np.array([5, 7, -1, 362], dtype=np.int64), A)
    assert out.shape == (4, A)


# --------------------------------------------------------------------------- #
# 逐位一致：两条路必须给出同一个东西
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('moves', [
    [5, 7, -1, 362],
    [0, 0, 0, 0],
    [-1, -1, -1, -1],
    list(range(16)),
])
def test_tensor_and_numpy_paths_are_bitwise_identical(moves):
    """Tensor 与 numpy 两条路必须**逐位相同**（否则标签在两条路上不同）。

    逐位而不是 `approx`：这是 one-hot，值只该是 0.0 / 1.0。
    """
    mv = list(moves)
    a = _dense_move_target(torch.tensor(mv, dtype=torch.long), A)
    b = _dense_move_target(np.array(mv, dtype=np.int64), A)
    assert torch.equal(a, b)


def test_out_of_range_and_sentinel_rows_are_all_zero():
    """ 非法行号与 `-1` 哨兵都必须**全零**，且**不是**随便某个类。

    `-1` 的语义是「没有下一手」（局末 / 跨局 / pass），对应
    `w['policy_opp'] == 0`；越界同理。交给 `F.one_hot(-1)` 的行为不是本仓
    可以依赖的契约（不同 torch 版本不同）⇒ 必须显式置零。
    """
    out = _dense_move_target(torch.tensor([-1, A, A + 5, 7]), A)
    assert out[0].sum().item() == 0.0, '-1 行必须全零'
    assert out[1].sum().item() == 0.0, '越界行必须全零'
    assert out[2].sum().item() == 0.0, '越界行必须全零'
    assert out[3].sum().item() == 1.0
    assert int(out[3].argmax()) == 7


def test_grad_is_not_flowing_through_the_label():
    """one-hot 是**常量标签**，梯度不该经过它。

    传进来的 `moves` 可能带图（复用自别处的张量）；`.detach()` 保证
    「标签侧」不参与反向。
    """
    base = torch.tensor([5, 7], dtype=torch.float32, requires_grad=True)
    mv = base * 1.0                     # 带图
    out = _dense_move_target(mv, A)
    assert not out.requires_grad, '标签张量不该带图'


# --------------------------------------------------------------------------- #
# 设备：结果留在输入所在设备（不要静默搬回主机）
# --------------------------------------------------------------------------- #
def test_result_stays_on_the_input_device():
    """ 输出必须在 `moves` 所在设备 —— 否则每步多两次 H2D/D2H。

    CPU 上这个断言平凡成立，但它是「不许退回 `.cpu()` 修法」的护栏：
    有人为了省事改成 `np.asarray(moves.cpu())` 时，本条在 CPU 上抓不到，
    可它同样会把意图写进注释与函数体，届时由
    `test_tensor_input_never_touches_numpy` 拦下（它连 `.cpu()` 的结果
    也不许送进 numpy）。
    """
    mv = torch.tensor([1, 2, 3], dtype=torch.long)
    out = _dense_move_target(mv, A)
    assert out.device == mv.device


def test_non_contiguous_input_is_handled():
    """ 传进来的可能是**非连续**视图（切片 / 转置的 `moves`）。

    `.reshape(-1)` 而不是 `.view(-1)`：后者对非连续张量抛
    「view size is not compatible with input tensor's size」。
    """
    mv = torch.arange(20, dtype=torch.long)[::2]      # 非连续
    assert not mv.is_contiguous()
    out = _dense_move_target(mv, A)
    assert out.shape == (mv.numel(), A)
    ref = _dense_move_target(mv.contiguous(), A)
    assert torch.equal(out, ref)


def test_float_tensor_moves_is_cast_not_rejected():
    """`moves` 若是 float（某些路径给的是浮点行号），应**转换**而不是报错。

    `moves` 来自 `torch.from_numpy(moves_np)`，而 dataset 的 `moves` 列是
    int16/int64；但 int16 在 NPU 上 `one_hot` 不接受，所以统一 `.to(long)`
    —— 这条钉住「dtype 不是理由去走 numpy」。
    """
    out = _dense_move_target(torch.tensor([5.0, 7.0]), A)
    assert int(out[0].argmax()) == 5
    assert out.dtype is torch.float32


# --------------------------------------------------------------------------- #
# v7_loss_labels 端到端（真实崩溃点）
# --------------------------------------------------------------------------- #
def test_v7_loss_labels_accepts_a_tensor_for_moves():
    """ 真实崩溃点是 `v7_loss_labels(lbl, move_t, ...)`，不是一个孤立函数。

    `move_t` 在训练循环里是 `torch.from_numpy(...).to(device)` 的结果 ⇒ NPU 上
    是设备张量。这里用「打掉 numpy 转换」在 CPU 上复现那个前提。
    """
    np_asarray = np.asarray

    def _boom(x):
        if isinstance(x, torch.Tensor):
            raise TypeError("can't convert npu:0 device type tensor to numpy")
        return np_asarray(x)

    b = 4
    lbl = {
        'next_move': np.array([1, 2, 3, -1], dtype=np.int64),
        'next_move_opp': np.array([4, 5, 6, 7], dtype=np.int64),
        'future': np.zeros((b, 2, 19, 19), np.float32),
        'outcome': np.zeros(b, np.int64),
        'w': {'policy_opp': np.ones(b, np.float32)},
    }
    np.asarray = _boom
    try:
        out = v7_loss_labels(lbl, torch.tensor([10, 11, 12, -1]), action_size=A)
    finally:
        np.asarray = np_asarray
    assert out['policy_player'].shape == (b, A)
    assert out['policy_opp'].shape == (b, A)
    assert int(out['policy_player'][0].argmax()) == 10
    assert out['policy_opp'][3].sum().item() == 1.0, 'policy_opp 用的是 next_move_opp'


def test_policy_opp_falls_back_to_next_move_when_absent():
    """board 级路径没有 `next_move_opp` ⇒ 退回 `next_move`（旧行为）。

     这条退路在 stdata 上会让 #1/#2 拿到**同一个**目标（π_opp 白训），
    所以 packed 路径**必须**给 `next_move_opp` —— 由
    `tests/test_v7_soft_ce.py::test_policy_and_policy_opp_use_different_targets`
    钉住「两条路的标签真的不同」。
    """
    b = 2
    lbl = {
        'next_move': np.array([8, 9], dtype=np.int64),
        'future': np.zeros((b, 2, 19, 19), np.float32),
        'outcome': np.zeros(b, np.int64),
        'w': {},
    }
    out = v7_loss_labels(lbl, torch.tensor([1, 2]), action_size=A)
    assert int(out['policy_opp'][0].argmax()) == 8
    assert int(out['policy_opp'][1].argmax()) == 9


def test_both_targets_differ_when_opp_is_supplied():
    """#1 与 #2 的目标必须不同（历史上 π_opp 与 π 同源 ⇒ 白训）。"""
    b = 2
    lbl = {
        'next_move': np.array([8, 9], dtype=np.int64),
        'next_move_opp': np.array([20, 21], dtype=np.int64),
        'future': np.zeros((b, 2, 19, 19), np.float32),
        'outcome': np.zeros(b, np.int64),
        'w': {},
    }
    out = v7_loss_labels(lbl, torch.tensor([1, 2]), action_size=A)
    assert not torch.equal(out['policy_player'], out['policy_opp'])


def test_one_hot_matches_the_reference_construction():
    """与「先 clamp 再 one_hot、再把非法行置零」的朴素写法逐位一致。"""
    mv = torch.tensor([3, -1, A, 17], dtype=torch.long)
    got = _dense_move_target(mv, A)
    want = F.one_hot(mv.clamp(0, A - 1), A).to(torch.float32)
    want[~((mv >= 0) & (mv < A))] = 0.0
    assert torch.equal(got, want)