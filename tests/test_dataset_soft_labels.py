"""`labels=True` 的**批契约**测试：4 元组 + `labels_dict`（软标签接入 A1）。

契约（spec §5.5）
-----------------
    labels=False（默认）→ **三元组** (states, moves_out, values)
    labels=True          → **4 元组** (states, moves_out, values, labels_dict)

为什么必须是 dict 而不是原来的五元组 `(states, moves, values, soft, mask)`
------------------------------------------------------------------------
五元组**没有 dict 的位置**，而 V7 的 loss 需要那个装满
`next_move`/`outcome`/`score`/`sb_*`/`global`/`ownership`/`scoring`/`seki`/
`future`/`w{...}`/`game_weight` 的 dict ⇒ 再往下走只会变成 6 元组、或 fork
两条路。4 元组让预取器只需搬运**一种** payload 形状。

本文件锁的三件事
--------------
1. **`labels=False` 的返回值逐位不变**（热路径，34M 行 × 每 step 都跑）；
2. `labels=True` 的形状/dtype/键**恒定**（软标签挂不挂都是 4 元组）；
3. **每个能从现有 10 列实时算出来的标签，其值都是对的**：
   - `next_move` 必须**跨局被守卫**（残行会让相邻两行不是同一局）；
   - `outcome` 的三分类从 `winrates` 精确还原；
   - `soft` 必须与 `states` 同步重排（错位不报错，只是棋力不涨）。

跑：pytest tests/test_dataset_soft_labels.py -v
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import SupervisedDataset, permute_soft  # noqa: E402
from src.game.go_rules import GoBoard, SYMMETRIES  # noqa: E402

#: `labels_dict` 必须恒含的键（A1 已实现 + 占位待接线）。
REQUIRED_KEYS = ('next_move', 'outcome', 'outcome_black', 'game_weight',
                 'soft', 'soft_mask', 'w', 'score', 'sb_center', 'sb_upper',
                 'global', 'ownership', 'scoring', 'seki', 'future')


def _toy_dataset(bs=9, game_lens=(5, 4), seed=0, with_winrates=True,
                 with_game_weights=True):
    """造一个小数据集。`game_lens` 决定局的条数与每局行数（用于 next_move 守卫）。"""
    rng = np.random.default_rng(seed)
    n_sq = bs * bs
    n = int(sum(game_lens))
    boards = np.zeros((n, bs, bs), dtype=np.int8)
    for i in range(n):
        k = int(rng.integers(0, 6))
        for c in rng.choice(n_sq, size=k, replace=False):
            boards[i][c // bs, c % bs] = 1 if rng.random() < 0.5 else -1
    data = {
        'boards': boards,
        'my_hist': np.full((n, 3), -1, dtype=np.int16),
        'op_hist': np.full((n, 3), -1, dtype=np.int16),
        'ko': np.full(n, -1, dtype=np.int16),
        'moves': rng.integers(0, n_sq, size=n).astype(np.int16),
        'values': rng.choice([-1, 1], size=n).astype(np.int8),
        'to_play': rng.choice([-1, 1], size=n).astype(np.int8),
        'game_ids': np.concatenate(
            [np.full(L, g, dtype=np.int32) for g, L in enumerate(game_lens)]),
    }
    if with_winrates:
        data['winrates'] = rng.uniform(-0.9, 0.9, size=n).astype(np.float32)
    if with_game_weights:
        data['game_weights'] = rng.uniform(0.5, 2.0, size=n).astype(np.float32)
    return data


def _soft_labels(bs, m, rows, seed=7):
    """造 m 份合法软标签（行和 = 1），挂在给定的 rows 上。返回 (soft_idx, soft_policy)。"""
    rng = np.random.default_rng(seed)
    sp = rng.random((m, bs * bs + 1)).astype(np.float32)
    sp /= sp.sum(axis=1, keepdims=True)
    return np.asarray(rows, dtype=np.int64), sp


# --------------------------------------------------------------------------- #
# 1. labels=False 的热路径：逐位不变
# --------------------------------------------------------------------------- #
def test_labels_false_returns_plain_triple():
    """默认路径仍是**三元组**（bench_train / train_sft / 预取器都按 3 个解包）。"""
    ds = SupervisedDataset(_toy_dataset(), n_channels=12)
    out = ds.sample_batch_numpy(np.arange(9), augment=True)
    assert len(out) == 3, f"labels=False 必须返回三元组，实得 {len(out)} 个"
    states, moves_out, values = out
    assert states.ndim == 4 and states.shape[1] == 12
    assert moves_out.dtype == np.int64 and values.shape == (9, 1)


def test_labels_true_first_three_are_bitwise_identical():
    """**加标签分支不能动热路径**：前三个元素与 labels=False 逐位相等。

    这是「labels=False 逐位不变」在本仓可验证的形式 —— 两条路径共用同一段
    特征构造/增强/规范化代码，唯一差别是末尾追加了一个 dict。
    """
    ds = SupervisedDataset(_toy_dataset(), n_channels=12)
    idxs = np.arange(9)
    a = ds.sample_batch_numpy(idxs, rng=np.random.default_rng(1), augment=True,
                              labels=False)
    b = ds.sample_batch_numpy(idxs, rng=np.random.default_rng(1), augment=True,
                              labels=True)
    assert len(a) == 3 and len(b) == 4
    for x, y, name in zip(a, b[:3], ('states', 'moves_out', 'values')):
        assert x.dtype == y.dtype, f"{name} 的 dtype 变了：{x.dtype} → {y.dtype}"
        assert x.shape == y.shape, f"{name} 的 shape 变了：{x.shape} → {y.shape}"
        assert x.tobytes() == y.tobytes(), f"{name} 不是逐位相等（增强 RNG 同种子）"


def test_augment_false_states_are_raw_feature_planes():
    """`augment=False` 时 `states` 就是 `feature_planes_batched` 的原始输出。"""
    bs = 9
    data = _toy_dataset(bs=bs)
    ds = SupervisedDataset(data, n_channels=12)
    idxs = np.arange(5)
    states, _, _ = ds.sample_batch_numpy(idxs, augment=False)
    ref = GoBoard.feature_planes_batched(
        data['boards'][idxs], data['my_hist'][idxs], data['op_hist'][idxs],
        data['to_play'][idxs], data['ko'][idxs], n_channels=12)
    assert states.tobytes() == ref.tobytes(), "augment=False 不该做任何变换"


# --------------------------------------------------------------------------- #
# 2. labels=True 的形状：恒 4 元组 + dict
# --------------------------------------------------------------------------- #
def test_labels_true_returns_four_tuple_with_dict():
    ds = SupervisedDataset(_toy_dataset(), n_channels=12)
    out = ds.sample_batch_numpy(np.arange(9), augment=False, labels=True)
    assert len(out) == 4, f"labels=True 必须返回 4 元组，实得 {len(out)} 个"
    d = out[3]
    assert isinstance(d, dict), f"第 4 个元素必须是 dict，实得 {type(d)}"
    missing = [k for k in REQUIRED_KEYS if k not in d]
    assert not missing, f"labels_dict 缺键：{missing}"
    assert isinstance(d['w'], dict) and 'policy_opp' in d['w']
    assert d['next_move'].dtype == np.int64
    assert d['outcome'].dtype == np.int64
    assert d['soft'].dtype == np.float32
    assert d['soft_mask'].dtype == np.float32


def test_dict_is_complete_when_no_soft_attached():
    """**没挂软标签也是 4 元组、dict 照样完整**（soft 全 0 / soft_mask 全 0）。

    形状随数据分叉是 bug 温床：预取器的 payload 只有一种形状才不用 fork。
    """
    ds = SupervisedDataset(_toy_dataset(), n_channels=12)   # 不传 soft_idx/soft_policy
    assert ds.soft_row is None
    _, _, _, d = ds.sample_batch_numpy(np.arange(9), augment=False, labels=True)
    missing = [k for k in REQUIRED_KEYS if k not in d]
    assert not missing, f"未挂软标签时 dict 缺键：{missing}"
    assert d['soft'].shape == (9, 9 * 9 + 1)
    assert not d['soft'].any(), "未挂软标签时 soft 必须全 0"
    assert not d['soft_mask'].any(), "未挂软标签时 soft_mask 必须全 0"


def test_labels_and_no_labels_agree_on_tuple_length():
    """挂 / 不挂软标签，`labels=True` 的元组长度必须是同一个（4）。"""
    base = _toy_dataset()
    idxs = np.arange(9)
    plain = SupervisedDataset(base, n_channels=12)
    rows, sp = _soft_labels(9, 4, [0, 3, 5, 8])
    withsoft = SupervisedDataset(base, n_channels=12,
                                 soft_idx=rows, soft_policy=sp)
    assert len(plain.sample_batch_numpy(idxs, augment=False, labels=True)) == 4
    assert len(withsoft.sample_batch_numpy(idxs, augment=False, labels=True)) == 4


def test_sample_batch_labels_true_returns_four_tensors():
    """torch 版 `sample_batch` 同步：第 4 个元素是 dict，值都是张量（含嵌套 w）。"""
    base = _toy_dataset()
    rows, sp = _soft_labels(9, 3, [1, 4, 6])
    ds = SupervisedDataset(base, n_channels=12, soft_idx=rows, soft_policy=sp)
    out = ds.sample_batch(np.arange(9), labels=True)
    assert len(out) == 4
    states, moves_out, values, d = out
    assert isinstance(states, torch.Tensor) and states.shape == (9, 12, 9, 9)
    assert isinstance(moves_out, torch.Tensor) and isinstance(values, torch.Tensor)
    assert isinstance(d, dict)
    for k in REQUIRED_KEYS:
        if k == 'w':            # w 是嵌套 dict（内层才是张量）
            assert isinstance(d[k], dict)
            continue
        assert isinstance(d[k], torch.Tensor), f"{k} 没有转成张量（{type(d[k])}）"
    for k, v in d['w'].items():
        assert isinstance(v, torch.Tensor), f"w['{k}'] 没有转成张量（{type(v)}）"
    assert d['soft'].shape == (9, 82)
    assert len(ds.sample_batch(np.arange(9))) == 3, "labels=False 仍是三元组"


# --------------------------------------------------------------------------- #
# 3. next_move：跨局守卫 + 局末手 = -1
# --------------------------------------------------------------------------- #
def test_next_move_is_minus_one_at_game_end():
    """每局末手 = −1（−1 就是 pass，也是「没有下一手」的哨兵）。"""
    bs = 9
    ds = SupervisedDataset(_toy_dataset(bs=bs, game_lens=(5, 4)), n_channels=12)
    idxs = np.arange(9)
    _, _, _, d = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    nm = d['next_move']
    assert nm.shape == (9,)
    assert nm[4] == -1, f"第 0 局末手（行 4）应为 -1，实得 {nm[4]}"
    assert nm[8] == -1, f"第 1 局末手（行 8）应为 -1，实得 {nm[8]}"
    assert (nm[[0, 1, 2, 3, 5, 6, 7]] >= 0).all(), "非末手不该是 -1"


def test_next_move_does_not_cross_games():
    """**跨局不串**：第 0 局末手的下一行属于第 1 局，绝不能被当成 π_opp。"""
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(3, 3), seed=2)
    # 让每局第一手的落点各不相同且合法，便于确认没有拿到跨界的值
    data['moves'] = np.array([10, 11, 12, 40, 41, 42], dtype=np.int16)
    assert (data['game_ids'] == np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)).all()
    ds = SupervisedDataset(data, n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(6), augment=False, labels=True)
    nm = d['next_move']
    # 行 2 是第 0 局末手 → -1（**不是**行 3 的 40）
    assert nm[2] == -1, f"跨局串了：nm[2]={nm[2]}"
    # 行 0/1 是同局，π_opp 分别是下一行的落点
    assert nm[0] == 11 and nm[1] == 12
    assert nm[4] == 42 and nm[5] == -1
    # 权重也必须跟着归零：局末手对 π_opp 无权重
    assert d['w']['policy_opp'].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]


def test_next_move_ignores_last_row_of_dataset():
    """数据集最后一行没有「下一行」，必须是 −1（不能越界 gather）。"""
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(4,), seed=3)
    ds = SupervisedDataset(data, n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(4), augment=False, labels=True)
    assert d['next_move'][3] == -1
    assert d['w']['policy_opp'][3] == 0.0


def test_next_move_without_game_ids_is_all_sentinel():
    """无 game_ids ⇒ **无法验证同局**，一律 −1（不猜「相邻行就是同局」）。

    猜的那条路正是要防的串局；代价是这些行的 π_opp 权重恒 0（生产数据必带 game_ids）。
    """
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(4,), seed=4)
    del data['game_ids']
    ds = SupervisedDataset(data, n_channels=12)
    assert ds.game_ids is None
    _, _, _, d = ds.sample_batch_numpy(np.arange(4), augment=False, labels=True)
    assert (d['next_move'] == -1).all(), "没有 game_ids 时不该猜着给 next_move"
    assert not d['w']['policy_opp'].any()


def test_next_move_is_permuted_like_moves_out():
    """增强时 `next_move` 必须与 `moves_out` 走**同一个** SYMMETRIES 映射。

    漏掉这一步 = policy_opp 标签指向翻转前的点，训练不报错、只是棋力不涨
    （与 `permute_soft` 同类的静默错误，见 spec §5.6）。
    """
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(5,), seed=5)
    ds = SupervisedDataset(data, n_channels=12)
    idxs = np.arange(5)
    for t in range(8):
        spy = np.random.default_rng(t)          # 「偷看」tforms
        tforms = spy.integers(0, 8, size=len(idxs))
        real = np.random.default_rng(t)
        _, moves_out, _, d = ds.sample_batch_numpy(idxs, rng=real, augment=True,
                                                   labels=True)
        nm = d['next_move']
        want = int(data['moves'][1])             # 行 0 的下一手（未被增强的落点）
        r, c = SYMMETRIES[int(tforms[0])](np.array([want // bs]),
                                          np.array([want % bs]), bs)
        assert nm[0] == int(r[0]) * bs + int(c[0]), \
            f"t={t}: next_move 没跟着 states 一起变换（{nm[0]}）"
        # moves_out 同理（这条已有 test_eval_no_augment 覆盖，这里做对照）
        assert moves_out[0] == nm[0] or int(data['moves'][0]) != want


def test_next_move_sentinel_survives_permutation():
    """−1 哨兵与 pass 在任何变换下**都不动**（与 moves_out 的 pass 一致）。"""
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(2, 3), seed=6)
    ds = SupervisedDataset(data, n_channels=12)
    idxs = np.arange(5)
    for t in range(8):
        _, _, _, d = ds.sample_batch_numpy(idxs, rng=np.random.default_rng(100 + t),
                                           augment=True, labels=True)
        assert d['next_move'][1] == -1, f"t={t}: 局末手哨兵被变换动了"


# --------------------------------------------------------------------------- #
# 4. outcome：三分类从 winrates 精确还原
# --------------------------------------------------------------------------- #
def test_outcome_three_way_from_winrates():
    """`winrates=[+0.5, -0.5, 0.0]` → `outcome=[0, 1, 2]`（胜 / 负 / 无结果）。"""
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(3,), seed=7)
    data['winrates'] = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    ds = SupervisedDataset(data, n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(3), augment=False, labels=True)
    assert d['outcome'].tolist() == [0, 1, 2]
    assert set(np.unique(d['outcome']).tolist()) <= {0, 1, 2}
    # values 的口径也跟着 winrates 走（既有行为，不因加标签而变）
    _, _, values = ds.sample_batch_numpy(np.arange(3), augment=False)
    assert values.reshape(-1).tolist() == [0.5, -0.5, 0.0]


def test_outcome_is_to_play_view_and_black_view_flips():
    """`outcome` 是 **to_play 视角**；`outcome_black` 在白走时两个结论互换。

    ⚠ 不能写 `outcome * to_play`：`0 * -1 == 0`，会把「白胜」也报成「黑胜」。
    """
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(4,), seed=8)
    data['winrates'] = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    data['to_play'] = np.array([1, 1, -1, -1], dtype=np.int8)   # 黑 黑 白 白
    ds = SupervisedDataset(data, n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(4), augment=False, labels=True)
    assert d['outcome'].tolist() == [0, 1, 0, 1], "winrate>0 恒为 to_play 胜"
    # 黑走：to_play 视角 == 黑方视角
    assert d['outcome_black'].tolist()[:2] == [0, 1]
    # 白走：两个结论互换（白胜 ⇒ 黑负，白负 ⇒ 黑胜）
    assert d['outcome_black'].tolist()[2:] == [1, 0]


def test_outcome_no_result_stays_no_result_in_both_views():
    """无结果（winrate == 0）在两个视角下都是 2，不能被「互换」逻辑改成胜负。"""
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(2,), seed=9)
    data['winrates'] = np.array([0.0, 0.0], dtype=np.float32)
    data['to_play'] = np.array([1, -1], dtype=np.int8)
    ds = SupervisedDataset(data, n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(2), augment=False, labels=True)
    assert d['outcome'].tolist() == [2, 2]
    assert d['outcome_black'].tolist() == [2, 2]


def test_outcome_falls_back_to_values_without_winrates():
    """无 winrates 列时回退 `values`（同语义：正=胜、负=负、0=无结果）。"""
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(3,), seed=10, with_winrates=False)
    data['values'] = np.array([1, -1, 0], dtype=np.int8)
    ds = SupervisedDataset(data, n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(3), augment=False, labels=True)
    assert d['outcome'].tolist() == [0, 1, 2]


def test_outcome_is_invariant_under_augment():
    """翻转棋盘不改变胜负（§5.6：outcome 不进 tforms）。"""
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(6,), seed=11)
    ds = SupervisedDataset(data, n_channels=12)
    idxs = np.arange(6)
    _, _, _, no_aug = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    _, _, _, aug = ds.sample_batch_numpy(idxs, rng=np.random.default_rng(3),
                                         augment=True, labels=True)
    assert aug['outcome'].tolist() == no_aug['outcome'].tolist()
    assert aug['outcome_black'].tolist() == no_aug['outcome_black'].tolist()
    assert aug['game_weight'].tolist() == no_aug['game_weight'].tolist()
    assert aug['soft_mask'].tolist() == no_aug['soft_mask'].tolist()


def test_game_weight_defaults_to_one_without_column():
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(4,), seed=12, with_game_weights=False)
    ds = SupervisedDataset(data, n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(4), augment=False, labels=True)
    assert d['game_weight'].shape == (4,)
    assert d['game_weight'].tolist() == [1.0] * 4


def test_game_weight_reads_the_column():
    bs = 9
    data = _toy_dataset(bs=bs, game_lens=(3,), seed=13)
    ds = SupervisedDataset(data, n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(3), augment=False, labels=True)
    assert d['game_weight'].tolist() == pytest.approx(
        data['game_weights'][:3].astype(np.float32).tolist())


# --------------------------------------------------------------------------- #
# 5. soft / soft_mask：形状 + 与 states 同步重排
# --------------------------------------------------------------------------- #
def test_soft_shape_and_mask_follow_attachment():
    """`soft` (B, bs²+1)、`soft_mask` 逐行标 1/0；未命中的行 soft 全 0。"""
    bs = 9
    A = bs * bs + 1
    data = _toy_dataset(bs=bs, game_lens=(9,), seed=14)
    rows, sp = _soft_labels(bs, 3, [1, 4, 7])
    ds = SupervisedDataset(data, n_channels=12, soft_idx=rows, soft_policy=sp)
    _, _, _, d = ds.sample_batch_numpy(np.arange(9), augment=False, labels=True)
    assert d['soft'].shape == (9, A)
    assert d['soft_mask'].dtype == np.float32
    assert d['soft_mask'].tolist() == [0, 1, 0, 0, 1, 0, 0, 1, 0]
    assert d['soft_mask'].sum() == 3.0
    for j, k in enumerate((1, 4, 7)):
        assert np.allclose(d['soft'][k], sp[j]), f"行 {k} 取错了软标签槽位"
    # 未命中的行必须是全 0（不能拿相邻槽位的分布来污染）
    for k in (0, 2, 3, 5, 6, 8):
        assert not d['soft'][k].any(), f"行 {k} 没有软标签却非 0"
    # 软标签必须是合法分布（行和 ≈ 1）
    assert np.allclose(d['soft'].sum(1)[d['soft_mask'] > 0], 1.0, atol=1e-6)


def test_soft_under_augment_equals_permute_soft():
    """增强时 `soft` 的差异**恰好**等于 `permute_soft` 的效果（不多不少）。

    做法：同种子 Generator 先「偷看」tforms，另一个同种子的传给数据集。
    """
    bs = 9
    A = bs * bs + 1
    data = _toy_dataset(bs=bs, game_lens=(9,), seed=15)
    rows, sp = _soft_labels(bs, 3, [0, 5, 8])
    ds = SupervisedDataset(data, n_channels=12, soft_idx=rows, soft_policy=sp)
    idxs = np.arange(9)

    spy = np.random.default_rng(4242)
    tforms = spy.integers(0, 8, size=len(idxs))
    _, _, _, d = ds.sample_batch_numpy(idxs, rng=np.random.default_rng(4242),
                                       augment=True, labels=True)
    # 未增强的原始批（同一个 aug=False 调用给出「没被重排」的基准）
    raw = np.zeros((len(idxs), A), dtype=np.float32)
    for j, r in enumerate((0, 5, 8)):
        raw[r] = sp[j]
    expect = permute_soft(raw, tforms, bs)
    assert d['soft'].tobytes() == expect.tobytes(), "增强后的 soft != permute_soft(raw)"
    assert tforms.any(), "本用例得真的抽到非恒等变换，否则是空转"


def test_soft_is_identical_to_permute_soft_rowwise():
    """逐行等价：每行的 soft 恰好等于该行单独过一遍 permute_soft。"""
    bs = 9
    A = bs * bs + 1
    data = _toy_dataset(bs=bs, game_lens=(12,), seed=16)
    rows = np.arange(12)
    rng = np.random.default_rng(17)
    sp = rng.random((12, A)).astype(np.float32)
    sp /= sp.sum(1, keepdims=True)
    ds = SupervisedDataset(data, n_channels=12, soft_idx=rows, soft_policy=sp)
    idxs = np.arange(12)
    spy = np.random.default_rng(77)
    tforms = spy.integers(0, 8, size=len(idxs))
    _, _, _, d = ds.sample_batch_numpy(idxs, rng=np.random.default_rng(77),
                                       augment=True, labels=True)
    for i in range(len(idxs)):
        ref = permute_soft(sp[i:i + 1], tforms[i:i + 1], bs)[0]
        assert np.allclose(d['soft'][i], ref), f"第 {i} 行与单独 permute 不一致"
    # 软标签重排后仍是合法分布
    assert np.allclose(d['soft'].sum(1), 1.0, atol=1e-6)


def test_soft_untouched_when_augment_false():
    """`augment=False` ⇒ soft 原样返回（tforms 恒等，不该有任何重排）。"""
    bs = 9
    A = bs * bs + 1
    data = _toy_dataset(bs=bs, game_lens=(6,), seed=18)
    rows = np.arange(6)
    rng = np.random.default_rng(19)
    sp = rng.random((6, A)).astype(np.float32)
    sp /= sp.sum(1, keepdims=True)
    ds = SupervisedDataset(data, n_channels=12, soft_idx=rows, soft_policy=sp)
    _, _, _, d = ds.sample_batch_numpy(np.arange(6), augment=False, labels=True)
    assert d['soft'].tobytes() == sp.tobytes()
    assert np.array_equal(d['soft_mask'], np.ones(6, dtype=np.float32))


def test_soft_argmax_agrees_with_moves_out_after_augment():
    """**生死门**（与 test_soft_labels.py 同判据）：one-hot 软标签置换后的 argmax
    必须精确等于数据集同批返回的 `moves_out`（同一组 tforms 下）。

    两者不一致 = 重排方向写反了。软标签只在 argmax 上与 moves 对齐才谈得上其余。
    """
    bs = 9
    A = bs * bs + 1
    n_sq = bs * bs
    data = _toy_dataset(bs=bs, game_lens=(12,), seed=20)
    idxs = np.arange(12)
    # 用**未增强**的 moves 造 one-hot 软标签（峰在真值落点上）
    soft = np.zeros((len(idxs), A), dtype=np.float32)
    mv = data['moves'].astype(np.int64)
    for i, m in enumerate(mv):
        if 0 <= m < n_sq:
            soft[i, m] = 1.0
    ds = SupervisedDataset(data, n_channels=12,
                           soft_idx=idxs, soft_policy=soft)
    _, moves_out, _, d = ds.sample_batch_numpy(idxs, rng=np.random.default_rng(31),
                                               augment=True, labels=True)
    for i in range(len(idxs)):
        if not (0 <= mv[i] < n_sq):
            continue                     # pass / 越界没有可校验的落点
        assert int(d['soft'][i].argmax()) == int(moves_out[i]), (
            f"行 {i}: 软标签 argmax={int(d['soft'][i].argmax())} "
            f"≠ moves_out={int(moves_out[i])}")


# --------------------------------------------------------------------------- #
# 6. 占位标签：恒 0 + 对应权重恒 0（不能被当成「标签就是 0」）
# --------------------------------------------------------------------------- #
def test_placeholder_labels_are_zeros_with_zero_weights():
    """待接线的标签一律恒 0，且 `w` 里对应项也恒 0 —— 语义是「本行没有这个标签」。"""
    ds = SupervisedDataset(_toy_dataset(game_lens=(6,)), n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(6), augment=False, labels=True)
    for k in ('score', 'sb_center', 'sb_upper', 'global',
              'ownership', 'scoring', 'seki', 'future'):
        assert not d[k].any(), f"占位标签 {k} 不是恒 0"
    assert d['w']['ownership'].sum() == 0.0
    assert d['w']['score'].sum() == 0.0
    assert d['w']['scoring'].sum() == 0.0
    assert d['w']['seki'].sum() == 0.0
    assert d['w']['futurepos'].sum() == 0.0
    assert d['w']['policy'].tolist() == [1.0] * 6, "π 本身的权重恒 1"


def test_future_placeholder_shape_is_bs_squared():
    """`future` = (B, 2, bs²)；19 路时即 (B,2,361)，与 V7 的 futurepos 对齐。"""
    bs = 9
    ds = SupervisedDataset(_toy_dataset(bs=bs, game_lens=(4,)), n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(4), augment=False, labels=True)
    assert d['future'].shape == (4, 2, bs * bs)
    assert d['future'].dtype == np.float32


def test_ownership_scoring_seki_shapes():
    bs = 9
    ds = SupervisedDataset(_toy_dataset(bs=bs, game_lens=(4,)), n_channels=12)
    _, _, _, d = ds.sample_batch_numpy(np.arange(4), augment=False, labels=True)
    for k in ('ownership', 'scoring', 'seki'):
        assert d[k].shape == (4, 1, bs, bs), f"{k} 形状 {d[k].shape}"
        assert d[k].dtype == np.float32
    assert d['global'].shape == (4, 19)
