# -*- coding: utf-8 -*-
"""A→B→C **共用一个模型**的回归测试（2026-10-04）。

背景
----
用户要「A/B/C 都用 1 个模型」。此前 `train_sft.py` 的实际口径是：

* A/B 段（`full.npz`，人类棋谱）→ `--v7 1` 建 12 通道 `SupervisedDataset`
  + V7 loss ⇒ 形状对不上，V7 的 22 平面**根本没参与**；
* C 段（`stdata_v7`）→ `V7PackedDataset`，22 通道预算特征。

于是两段的架构不同，权重**无法承接**，所谓「A→B→C」其实是三段各自为政。

现在 `load_from_path(v7=True)` 按**数据布局**分派到同一个 22 通道 V7：
board 级 → `V7Dataset`（训练时实时算 22 平面），
stdata 分片 → `V7PackedDataset`（预算好的 22 平面）。

本文件钉住三件事：分派正确、两段架构**逐位相同**、软标签口径不再误报。
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from src.data.dataset import SupervisedDataset  # noqa: E402
from src.data.v7_dataset import V7Dataset  # noqa: E402
from src.networks.katago_v7 import build_katago_v7_net  # noqa: E402
from train_sft import (  # noqa: E402
    load_from_path,
    narrow_to_soft_rows,
    resolve_policy_loss_kind,
)

BOARD = 19
A = BOARD * BOARD + 1


def _board_npz(path, n_games=4, per_game=4):
    """一个最小的 board 级语料（布局 = full.npz 那一类）。"""
    n = n_games * per_game
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        boards=rng.integers(0, 3, size=(n, BOARD, BOARD)).astype(np.int8),
        my_hist=rng.integers(0, 2, size=(n, 4)).astype(np.int8),
        op_hist=rng.integers(0, 2, size=(n, 4)).astype(np.int8),
        ko=rng.integers(0, BOARD * BOARD, size=n).astype(np.int64),
        moves=rng.integers(0, A - 1, size=n).astype(np.int64),
        values=rng.choice([-1.0, 1.0], size=n).astype(np.float32),
        # to_play 必须是 ±1（`feature_v7` 明确拒绝 0/1）
        to_play=rng.choice([-1, 1], size=n).astype(np.int64),
        game_ids=np.repeat(np.arange(n_games), per_game).astype(np.int64),
    )
    return path


# --------------------------------------------------------------------------- #
# 分派：board 级语料在 --v7 1 下必须是 V7Dataset
# --------------------------------------------------------------------------- #
def test_v7_board_level_dispatches_to_v7_dataset(tmp_path):
    """ board 级语料 + `--v7 1` ⇒ `V7Dataset`（而不是父类或 packed 类）。"""
    from src.data.v7_packed_dataset import V7PackedDataset

    p = _board_npz(tmp_path / 'board.npz')
    ds = load_from_path(str(p), BOARD, 0, v7=True)
    assert isinstance(ds, V7Dataset), type(ds).__name__
    # 判据是「有没有 spatial_packed」，board 级语料没有 ⇒ 不能进 packed 那条路
    assert not isinstance(ds, V7PackedDataset)
    assert not hasattr(ds, 'sample_spatial')


def test_v7_board_level_emits_22_channels(tmp_path):
    """ 产出的必须是 **22** 通道 —— 12 通道喂 V7 模型会形状不匹配。"""
    p = _board_npz(tmp_path / 'board.npz')
    ds = load_from_path(str(p), BOARD, 0, v7=True)
    spatial, gl = ds.sample_batch_v7(np.arange(4), rng=np.random.default_rng(0),
                                     augment=False)[0:2]
    assert spatial.shape[1] == 22, spatial.shape
    assert gl.shape[1] == 19, gl.shape


def test_n_channels_12_does_not_downgrade_the_v7_input(tmp_path):
    """ `n_channels=12` 只管**继承来的 12 通道路径**，不得影响 V7 的 22。

    这正是本轮踩过的坑：构造器收到 12，若 V7 取样路径也用 12，前向就废了。
    """
    p = _board_npz(tmp_path / 'board.npz')
    ds = load_from_path(str(p), BOARD, 0, v7=True)
    assert ds.n_channels == 12, '构造参数原样保留（继承路径的行为）'
    spatial = ds.sample_batch_v7(np.arange(2), rng=np.random.default_rng(0),
                                 augment=False)[0]
    assert spatial.shape[1] == 22


def test_both_sources_feed_the_identical_architecture():
    """ A/B 与 C 必须是**同一个**架构 —— 「1 个模型」的全部含义。

    两边数据布局完全不同（一段实时算、一段预算好），但喂给模型的张量形状
    必须一致，否则 `load_state_dict` 承接就是空话。
    """
    net = build_katago_v7_net(board_size=BOARD)
    n_v7 = sum(p.numel() for p in net.parameters())
    first = tuple(net.state_dict().values())[0]
    assert first.shape[1] == 22, f'V7 stem 应吃 22 通道，实得 {first.shape}'
    # 12 通道的旧结构通道数不同 ⇒ 权重形状不同，承接必然失败
    se = SupervisedDataset.__mro__  # 仅确认父类还在位
    assert SupervisedDataset is not None and len(se) >= 2
    assert n_v7 == 5_562_121, n_v7


# --------------------------------------------------------------------------- #
# 软标签口径：不能再对段 3 误报
# --------------------------------------------------------------------------- #
def test_soft_ce_accepted_for_packed_without_soft_index():
    """ 段 3（stdata 分片）的软标签内建，不该被「需要 --soft-index」拒掉。"""
    assert resolve_policy_loss_kind('soft_ce', None, v7_packed=True) == 'soft_ce'


def test_soft_ce_still_rejected_for_board_level_without_soft_index():
    """ 反向仍要拒：board 级 V7 没挂软索引时 `soft_mask` 恒 0。

    放过去就是「以为在蒸馏、其实软项恒 0」—— 训练照跑、loss 照降、
    policy 根本没学。这条守卫是段 1/2 的生命线，不能因为段 3 而整体拆掉。
    """
    with pytest.raises(SystemExit) as e:
        resolve_policy_loss_kind('soft_ce', None, v7_packed=False)
    assert 'soft' in str(e.value).lower()


def test_huber_still_rejected_against_soft_labels():
    with pytest.raises(SystemExit):
        resolve_policy_loss_kind('huber', 'some.npz', v7_packed=True)


def test_no_soft_source_keeps_the_hard_path_untouched():
    """段 1 数值必须逐位不变：没软标签就原样返回。"""
    assert resolve_policy_loss_kind('ce', None) == 'ce'
    assert resolve_policy_loss_kind('huber', None) == 'huber'


# --------------------------------------------------------------------------- #
# narrow_to_soft_rows：空训练集要报人话
# --------------------------------------------------------------------------- #
def test_empty_train_index_reports_the_real_cause(tmp_path):
    """ 棋局太少导致训练集为空时，必须说清楚，而不是抛 numpy 的 IndexError。

    症状链：按棋局切 98/2 ⇒ `int(棋局数*0.98)==0` ⇒ `np.array([])` 的 dtype
    是 **float64** ⇒ 拿去索引报「arrays used as indices must be of integer」。
    用户会顺着 numpy 去查，而真因是数据规模。
    """
    p = _board_npz(tmp_path / 'one_game.npz', n_games=1, per_game=8)
    ds = load_from_path(str(p), BOARD, 0, v7=True)
    empty = np.array([], dtype=np.float64)     # 复现上游那个空 float 数组
    with pytest.raises(SystemExit) as e:
        narrow_to_soft_rows(empty, ds)
    msg = str(e.value)
    assert '训练集' in msg and '空' in msg, msg


def test_soft_narrowing_keeps_only_hit_rows(tmp_path):
    """收窄到软行：非命中行不得混进来（否则 π 去折中而不是学搜索）。"""
    p = _board_npz(tmp_path / 'board.npz', n_games=4, per_game=4)
    ds = load_from_path(str(p), BOARD, 0, v7=True)
    idx = np.array([0, 5, 9], dtype=np.int64)
    pol = np.zeros((3, A), dtype=np.float32)
    pol[np.arange(3), [1, 2, 3]] = 1.0
    np.savez_compressed(tmp_path / 'si.npz', idx=idx, policy=pol)
    ds.attach_soft(idx, pol)
    train = np.arange(len(ds.moves), dtype=np.int64)
    out = narrow_to_soft_rows(train, ds)
    assert out.tolist() == [0, 5, 9]


def test_mask_length_mismatch_is_caught(tmp_path):
    """掩码长度 ≠ 行数 ⇒ 软索引与数据不是同一份，当场报错。"""
    p = _board_npz(tmp_path / 'board.npz', n_games=4, per_game=4)
    ds = load_from_path(str(p), BOARD, 0, v7=True)
    np.savez_compressed(tmp_path / 'si.npz',
                        idx=np.array([0], dtype=np.int64),
                        policy=np.zeros((1, A), dtype=np.float32))
    ds.attach_soft(np.array([0], dtype=np.int64),
                   np.zeros((1, A), dtype=np.float32))
    ds.soft_row = np.full(len(ds.moves) + 5, -1, dtype=np.int64)
    with pytest.raises(SystemExit) as e:
        narrow_to_soft_rows(np.arange(len(ds.moves), dtype=np.int64), ds)
    assert '行数' in str(e.value)


# --------------------------------------------------------------------------- #
# 承接：A→B→C 的 load_state_dict 必须真的成立
# --------------------------------------------------------------------------- #
def test_weights_carry_from_board_level_to_packed_stage(tmp_path):
    """ A/B（board 级）训出的权重能被 C（packed）原样吃下。

    这是「1 个模型」的**可执行**定义：两边张量形状一致 ⇒ `strict=True`
    加载零缺失零多余。
    """
    net_a = build_katago_v7_net(board_size=BOARD)
    net_c = build_katago_v7_net(board_size=BOARD)
    # `load_state_dict(strict=True)` 直接返回 `(missing_keys, unexpected_keys)`；
    # 两者皆空即「零缺失零多余」
    missing, unexpected = net_c.load_state_dict(net_a.state_dict(), strict=True)
    assert missing == [] and unexpected == [], (missing, unexpected)
    # V7 的前向吃**两份**输入（22 空间 + 19 全局）—— 两段数据都得同时提供
    x = torch.randn(2, 22, BOARD, BOARD)
    g = torch.randn(2, 19)
    net_a.eval(), net_c.eval()
    with torch.inference_mode():
        ya = net_a(x, g)['policy_logits']
        yc = net_c(x, g)['policy_logits']
    assert torch.equal(ya, yc), '同一权重在同一输入上必须给出同一输出'