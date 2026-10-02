"""软标签对称置换的正确性测试（软标签接入的**生死门**）。

为什么这个测试是生死门
----------------------
`sample_batch_numpy` 对 `states` 施加 8 种对称增强、对 `moves_out` 施加
`SYMMETRIES[t]` 重映射。软 policy（362 维）若不同步重排，训练**不会报任何错**：
损失照常下降、指标照常好看，但标签指向错误的点，棋力不涨。这是最难查的一类 bug，
所以必须在接训练之前钉死。

判定标准（唯一、客观）
--------------------
对同一批样本、同一个 tforms，`permute_soft` 重排后的软 policy 的 argmax，
必须等于 `sample_batch_numpy` 单独返回的 `moves_out`。
两者不一致 = 置换写错了。

跑：pytest tests/test_soft_labels.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import SupervisedDataset, permute_soft
from src.game.go_rules import SYMMETRIES


def _toy_dataset(bs=19, n=64, seed=0):
    """造一个小数据集：board 随机稀疏，move 取合法点，hist 合法。"""
    rng = np.random.default_rng(seed)
    n_sq = bs * bs
    boards = np.zeros((n, bs, bs), dtype=np.int8)
    for i in range(n):
        k = rng.integers(0, 8)
        cells = rng.choice(n_sq, size=k, replace=False)
        for c in cells:
            boards[i][c // bs, c % bs] = 1 if rng.random() < 0.5 else -1
    my = np.full((n, 3), -1, dtype=np.int16)
    op = np.full((n, 3), -1, dtype=np.int16)
    for i in range(n):
        if rng.random() < 0.5:
            c = int(rng.integers(0, n_sq))
            my[i] = [c, -1, -1]
    moves = rng.integers(0, n_sq, size=n).astype(np.int16)
    return {
        'boards': boards, 'my_hist': my, 'op_hist': op,
        'ko': np.full(n, -1, dtype=np.int16), 'moves': moves,
        'values': rng.choice([-1, 1], size=n).astype(np.int8),
        'to_play': rng.choice([-1, 1], size=n).astype(np.int8),
    }


def test_permute_matches_symmetries_on_single_move():
    """核心：置换后的 argmax 必须与 SYMMETRIES 重映射的落点一致。"""
    bs = 19
    n_sq = bs * bs
    A = n_sq + 1
    data = _toy_dataset(bs=bs, n=32, seed=1)

    for t in range(8):
        # 造一个 one-hot 软标签，峰在落点 c（不是 pass）
        c = 137
        soft = np.zeros((1, A), dtype=np.float32)
        soft[0, c] = 1.0
        out = permute_soft(soft, np.array([t]), bs)
        r, cc = SYMMETRIES[t](np.array([c // bs]), np.array([c % bs]), bs)
        expect = int(r[0]) * bs + int(cc[0])
        assert int(out[0].argmax()) == expect, \
            f"t={t}: 置换后 argmax={out[0].argmax()}, 期望 {expect}"


def test_pass_is_never_moved():
    """pass（第 A-1 项）在任何变换下都不得移动。"""
    bs = 19
    A = bs * bs + 1
    soft = np.zeros((1, A), dtype=np.float32)
    soft[0, A - 1] = 1.0
    for t in range(8):
        out = permute_soft(soft, np.array([t]), bs)
        assert int(out[0].argmax()) == A - 1, f"t={t}: pass 被移动了"


def test_permutation_is_a_bijection():
    """置换必须是双射（一一对应），否则会丢质量或复制质量。"""
    bs = 9
    A = bs * bs + 1
    rng = np.random.default_rng(2)
    for t in range(8):
        soft = np.zeros((1, A), dtype=np.float32)
        pts = rng.choice(bs * bs, size=10, replace=False)
        soft[0, pts] = rng.random(10).astype(np.float32)
        out = permute_soft(soft, np.array([t]), bs)
        # 落点部分：一一对应 => 排序后的值集合应完全相同
        assert np.allclose(np.sort(out[0, :bs * bs]),
                           np.sort(soft[0, :bs * bs])), f"t={t}: 非双射"


def test_identity_is_noop():
    bs = 19
    A = bs * bs + 1
    rng = np.random.default_rng(3)
    soft = rng.random((4, A)).astype(np.float32)
    out = permute_soft(soft, np.zeros(4, dtype=np.int64), bs)
    assert np.allclose(out, soft), "t=0（恒等）不该改变任何东西"


def test_batch_mixed_tforms():
    """一批里混合不同 tforms，逐样本各自处理。"""
    bs = 19
    A = bs * bs + 1
    n = 8
    rng = np.random.default_rng(4)
    soft = rng.random((n, A)).astype(np.float32)
    tforms = rng.integers(0, 8, size=n).astype(np.int64)
    out = permute_soft(soft, tforms, bs)
    assert out.shape == soft.shape
    for i in range(n):
        ref = permute_soft(soft[i:i + 1], tforms[i:i + 1], bs)
        assert np.allclose(out[i], ref[0]), f"样本 {i} 与单独处理不一致"


def test_soft_mass_is_conserved():
    """总质量守恒（重排不能凭空增删）。"""
    bs = 19
    A = bs * bs + 1
    rng = np.random.default_rng(5)
    soft = rng.random((6, A)).astype(np.float32)
    soft /= soft.sum(1, keepdims=True)
    tforms = rng.integers(0, 8, size=6).astype(np.int64)
    out = permute_soft(soft, tforms, bs)
    assert np.allclose(out.sum(1), soft.sum(1), atol=1e-6)


def test_rejects_bad_shapes():
    bs = 19
    A = bs * bs + 1
    with pytest.raises(ValueError):
        permute_soft(np.zeros((2, A + 1), dtype=np.float32),
                     np.zeros(2, dtype=np.int64), bs)
    with pytest.raises(ValueError):
        permute_soft(np.zeros((2, A), dtype=np.float32),
                     np.zeros(3, dtype=np.int64), bs)


def test_matches_dataset_augment_end_to_end():
    """端到端：与 `sample_batch_numpy` 真实施加增强后的 `moves_out` 对齐。

    这是最关键的一条 —— 前面的测试都拿 `SYMMETRIES` 直接比，只有这条是拿
    **数据集真实输出**比，能抓住「dataset 里实际用的映射与 permute_soft 不一致」。

    做法：两个**同种子**的 Generator，一个我抽 tforms 出来「偷看」，
    一个传给 `sample_batch_numpy`。二者初始状态相同 ⇒ 它抽到的 tforms
    就是数据集内部用的那组（`sample_batch_numpy` 在 augment=True 时第一件事
    就是 `rng.integers(0, 8, size=B)`）。
    """
    bs = 19
    n_sq = bs * bs
    A = n_sq + 1
    data = _toy_dataset(bs=bs, n=24, seed=6)
    ds = SupervisedDataset(data, n_channels=12)
    idxs = np.arange(24)

    spy = np.random.default_rng(1234)
    tforms = spy.integers(0, 8, size=len(idxs))       # 「偷看」数据集会用哪组
    real = np.random.default_rng(1234)                 # 同种子 = 同初始状态
    _, moves_out, _ = ds.sample_batch_numpy(idxs, rng=real, augment=True)

    # 造软标签：把 moves_out 当作「真实经过增强的落点」，反推一个峰在
    # **未增强落点**的 one-hot；置换后其 argmax 应精确等于 moves_out。
    moves_un = data['moves'].astype(np.int64)
    soft = np.zeros((len(idxs), A), dtype=np.float32)
    valid = (moves_un >= 0) & (moves_un < n_sq)
    soft[np.arange(len(idxs))[valid], moves_un[valid]] = 1.0
    # pass 类（越界/非法）没有可校验的落点，单独跳过
    keep = valid & (moves_out < n_sq)

    out = permute_soft(soft, tforms, bs)
    for i in np.flatnonzero(keep):
        assert int(out[i].argmax()) == int(moves_out[i]), (
            f"样本 {i} (t={tforms[i]}): 置换 argmax={out[i].argmax()}, "
            f"数据集 moves_out={moves_out[i]}")
