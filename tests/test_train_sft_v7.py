"""B8：`scripts/train_sft.py` 的 22 通道 V7 接线（spec §3 B 组）。

这份测试要守的四件事
--------------------
1. **V7 路径真的能端到端跑**：`(B,22,19,19) + (B,19)` 造出来 → 喂进
   `NbtTfNet` → 12 项 loss 算得出来 → 反传得动梯度。段 1 的四个目标
   （policy / π_opp / value / futurepos）**逐项**有限。
2. **段 1 不训 score 系**：8 项的权重逐项断言为 0，但 12 项**结构一个不少**地
   出现在返回值里（段 2/3 只改系数就能开回来，不需要动网络或 loss）。
3. **spawn 下 worker 重新 mmap、且不重复落盘**：`mp.Process` 在 Windows 上
   是 spawn 而不是 fork，spawn **不继承内存** —— 父进程 warm 好的映射句柄到不了
   子进程。这两件事都**不报错**，只会表现为 OOM 或「读到的数据莫名其妙」，
   所以必须实测钉住，不能靠读代码。
4. **默认路径逐位不变**：`--v7` 默认 0，12 通路的标签/取批/前向/损失一行都没改；
   `prefetch_workers` 的新默认是 12（C0 基准实测值）。

 **这里没有「断言吞吐 > X」的测试**：CI 机器的核数、页缓存、内存带宽都不同，
   任何吞吐断言都会变成随机红灯，而红灯会训练所有人忽略这个文件里真正重要的
   那几条。
"""

import ast
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sft import (  # noqa: E402
    V7_GLOBAL_CHANNELS, V7_SPATIAL_CHANNELS, V7_STAGE1_SCORE_TERMS,
    V7_STAGE1_TERMS, _dihedral_batch, _v7_labels_and_moves, build_katago_v7_net,
    build_v7_stage1_loss, evaluate_metrics_v7, v7_batch_features,
    v7_batch_sync, v7_loss_labels, v7_stage1_loss_weights,
)
from src.data.dataset import SupervisedDataset  # noqa: E402
from src.networks.katago_v7_loss import LOSS_COEFFS  # noqa: E402

BOARD = 19
ACTION = BOARD * BOARD + 1


# --------------------------------------------------------------------------- #
# 合成数据
# --------------------------------------------------------------------------- #
def make_dataset(n_games=4, game_len=48, seed=7, n_channels=12):
    """一份**小**合成数据集：`n_games × game_len` 行，19 路。

    刻意做成「每 `game_len` 行一局」并带 `game_ids`：futurepos 的 +8 / +32 偏移
    与 `next_move` 都靠 `game_ids` 做跨局守卫，没有它那两项恒为 0（测试会
    因为「权重是 0」而**假绿**）。盘面用确定性伪随机 + 若干连通块，让
    `calculate_area`（ch18/19）与 `iterLadders`（ch14–17）有活可干，而不是在
    空盘上走快路径。
    """
    rng = np.random.default_rng(seed)
    n = n_games * game_len
    boards = np.zeros((n, BOARD, BOARD), dtype=np.int8)
    # 撒一些 ±1 的连通小块（每 3×3 一块），避免整盘全空。
    for g in range(n_games):
        base = g * game_len
        for k in range(0, game_len, 3):
            r0 = int(rng.integers(0, BOARD - 4))
            c0 = int(rng.integers(0, BOARD - 4))
            sign = 1 if (g + k) % 2 == 0 else -1
            boards[base + k, r0:r0 + 3, c0:c0 + 3] = sign
    boards[boards == 0] = rng.choice(
        np.array([0, 1, -1], dtype=np.int8), size=int((boards == 0).sum()))
    to_play = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8)
    my_hist = rng.integers(-1, BOARD * BOARD, size=(n, 3)).astype(np.int16)
    op_hist = rng.integers(-1, BOARD * BOARD, size=(n, 3)).astype(np.int16)
    ko = rng.integers(-1, BOARD * BOARD, size=n).astype(np.int16)
    moves = rng.integers(0, BOARD * BOARD, size=n).astype(np.int16)
    # 胜负**与 to_play 无关地**随机：这样既有「黑胜」也有「白胜」的行。
    # 全部按 to_play 走的话只会出现「谁走谁赢」，`outcome_black` 那条测试里
    # 「白胜（to_play=-1 且 outcome=0）」就一条都没有 —— 断言会空跑而假绿。
    values = np.where((np.arange(n) // 7) % 2 == 0, 1, -1).astype(np.int8)
    game_ids = np.repeat(np.arange(n_games, dtype=np.int32), game_len)
    data = dict(boards=boards, my_hist=my_hist, op_hist=op_hist, ko=ko,
                moves=moves, values=values, to_play=to_play, game_ids=game_ids)
    return SupervisedDataset(data, n_channels=n_channels)


@pytest.fixture(scope='module')
def ds():
    return make_dataset()


@pytest.fixture(scope='module')
def v7_batch(ds):
    """一个真实走完「标签 + 22ch/19ch 特征」装配的 batch。"""
    tforms = np.zeros(8, dtype=np.int64)
    moves, lbl = _v7_labels_and_moves(ds, np.arange(8), tforms)
    sp, gl = v7_batch_features(ds, np.arange(8))
    return sp, gl, moves, lbl


@pytest.fixture(scope='module')
def net():
    return build_katago_v7_net(use_checkpoint=False)[0]


# --------------------------------------------------------------------------- #
# 1. 22 通道接入
# --------------------------------------------------------------------------- #
def test_spatial_and_global_shapes_and_dtypes(ds, v7_batch):
    """造出来的两份输入形状/dtype 必须是 V7 钉死的那一组。"""
    sp, gl, moves, lbl = v7_batch
    assert sp.shape == (8, V7_SPATIAL_CHANNELS, BOARD, BOARD), sp.shape
    assert gl.shape == (8, V7_GLOBAL_CHANNELS), gl.shape
    # fp16：与 `feature_v7` / `GoBoard.feature_planes_batched` 同口径，AMP 下
    # 不必再升精度（强升 fp32 会让 H2D 字节数白带一倍）。
    assert sp.dtype == np.float16 and gl.dtype == np.float16
    assert np.isfinite(sp.astype(np.float32)).all()
    assert np.isfinite(gl.astype(np.float32)).all()
    # ch0 = on-board 掩码（恒 1）—— 这条是 V7 相对 12 通路的口径变更点，
    # 错了整张特征图的语义就反了。
    assert np.all(sp[:, 0] == 1.0)
    assert moves.shape == (8,) and lbl['future'].shape == (8, 2, BOARD * BOARD)


def test_v7_forward_produces_every_loss_head_output(net, v7_batch):
    """`(B,22,H,W) + (B,19)` 必须喂得进 `NbtTfNet`，且四个头的输出齐备。"""
    sp, gl, _moves, _lbl = v7_batch
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(sp.astype(np.float32)),
                  torch.from_numpy(gl.astype(np.float32)))
    # policy 两路（policy_player / policy_opp）、value 三分类、futurepos 两通道。
    assert out['policy_logits'].shape == (8, 2, ACTION)
    assert out['outcome_logits'].shape == (8, 3)
    # futurepos 头是 1×1 卷积 ⇒ 输出**平面** (B,2,19,19)；loss 内部自己
    # reshape 成 (b,2,bs²)（见 `katago_v7_loss.py` #11）。
    assert out['futurepos'].shape == (8, 2, BOARD, BOARD)
    # score 系的头**也必须在**（段 1 权重 0，但结构保留、参数要吃梯度）。
    for k in ('ownership_pretanh', 'scoring', 'seki_logits', 'scorebelief_logits',
              'score_mean', 'score_stdev', 'lead'):
        assert k in out, f'V7 的 out 缺少 {k}：段 2/3 会用到它'


def test_stage1_four_objectives_are_finite_and_backprop_works(net, ds, v7_batch):
    """端到端：装配 → 前向 → 12 项 loss → 反传。

    段 1 的四个目标逐项断言**有限**（不是 0、不是 NaN/Inf）：有限性是「这一项
    真的在算」的最弱判据，而「权重是不是 0」由下一组测试单独钉 —— 两者缺一不可
    （一个能钉住「接错了」，一个能钉住「没接」）。
    """
    sp, gl, moves, lbl = v7_batch
    lossf = build_v7_stage1_loss()
    net.train()
    sp_t = torch.from_numpy(sp.astype(np.float32))
    gl_t = torch.from_numpy(gl.astype(np.float32))
    out = net(sp_t, gl_t)
    res = lossf(out, v7_loss_labels(lbl, moves))
    for term in V7_STAGE1_TERMS:
        assert term in res['terms'], f'段 1 目标 {term} 不在逐项 loss 里'
        assert np.isfinite(float(res['terms'][term].detach())), \
            f'{term} = {res["terms"][term]} 不是有限值'
    assert np.isfinite(float(res['loss'].detach()))
    res['loss'].backward()
    # 四个主目标各自喂到的头必须拿到梯度（policy 两路共用 policy_head）。
    for name in ('policy_head', 'value_head'):
        mod = getattr(net, name)
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in mod.parameters()), f'{name} 没有拿到梯度'


# --------------------------------------------------------------------------- #
# 2. 段 1：score 系权重 0，但结构仍在
# --------------------------------------------------------------------------- #
def test_stage1_score_family_coefficients_are_exactly_zero():
    """段 1 的 8 项 score 系权重**逐项**为 0。

    逐项而不是「共 8 项」：loss 以后增项时，「总数是 8」会悄悄失效，而失效的
    表现是「某个权重写反了却没人发现」。
    """
    w = v7_stage1_loss_weights()
    for k in V7_STAGE1_SCORE_TERMS:
        assert w[k] == 0.0, f'段 1 里 {k} 的权重应为 0，实得 {w[k]}'
    # 四个主目标必须**沿用 loss 自带值**（不是被改成别的数）。
    for k in V7_STAGE1_TERMS:
        assert w[k] == LOSS_COEFFS[k], \
            f'主目标 {k} 的权重被改成了 {w[k]}，应为 {LOSS_COEFFS[k]}'
    # 覆盖表是完整的（不只是 0 值项），否则「主目标被误改」没有任何测试看得见。
    assert set(w) == set(LOSS_COEFFS)


def test_stage1_score_family_contributes_nothing_but_is_still_computed(net, v7_batch):
    """被关掉的 score 系各项：贡献恒 0 **且**仍然出现在 `terms`/`weighted` 里。

     ``var_time_left`` 是**条件项**（2026-10-03）：官方 stdata 里有 col22，
    但老批次 / 老 fixture 的 labels 里可能没这个键，缺标签时**整项跳过**。
    这与「标签为 0」是两种不同的语义 —— 跳过才是对的，喂 0 会把这一路往
    「方差恒为 0」的方向硬拉。所以断言要分两拨。
    """
    sp, gl, moves, lbl = v7_batch
    lossf = build_v7_stage1_loss()
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(sp.astype(np.float32)),
                  torch.from_numpy(gl.astype(np.float32)))
    labels = v7_loss_labels(lbl, moves)
    res = lossf(out, labels)

    # 本 fixture 无 var_time_left 标签 ⇒ 该项被跳过，不该出现在 terms 里。
    assert 'var_time_left' not in labels, \
        'fixture 若已带该标签，本测试的跳过断言需改成「有标签」分支'
    assert 'var_time_left' not in res['terms']

    expect = set(LOSS_COEFFS) - {'var_time_left'}
    assert set(res['terms']) == expect, \
        f'terms 键集不符：多 {set(res["terms"]) - expect}，缺 {expect - set(res["terms"])}'
    for k in V7_STAGE1_SCORE_TERMS:
        if k not in res['terms']:
            continue
        assert np.isfinite(float(res['terms'][k])), \
            f'{k} 的逐项值不是有限值：结构没保住（段 2/3 会拿到垃圾）'
        assert float(res['weighted'][k]) == 0.0, \
            f'{k} 加权后应为 0，实得 {res["weighted"][k]}'
    # 贡献确实为 0：段 1 的总 loss 恰好等于四个主目标之和。
    total = sum(float(res['weighted'][k]) for k in V7_STAGE1_TERMS)
    assert float(res['loss']) == pytest.approx(total, rel=1e-5, abs=1e-7)


def test_var_time_left_is_computed_when_label_present(net, v7_batch):
    """反向对照：`var_time_left` 标签存在时，这一项必须**真的被算出来**。

    钉住「跳过」不等于「永远不算」—— 否则标签接上后该项会静默保持缺失。
    """
    sp, gl, moves, lbl = v7_batch
    lossf = build_v7_stage1_loss()
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(sp.astype(np.float32)),
                  torch.from_numpy(gl.astype(np.float32)))
    labels = v7_loss_labels(lbl, moves)
    assert 'var_time_left' not in labels

    b = sp.shape[0]
    labels['var_time_left'] = np.abs(
        np.random.default_rng(0).normal(10.0, 3.0, size=b)).astype(np.float32)
    res = lossf(out, labels)

    assert 'var_time_left' in res['terms'], '标签齐备却没算这一项'
    assert np.isfinite(float(res['terms']['var_time_left']))
    # 段 1 权重为 0 ⇒ 加权后仍是 0（但逐项值必须是真的）
    assert float(res['weighted']['var_time_left']) == 0.0


def test_score_terms_would_not_be_zero_if_weights_were_on(net, v7_batch):
    """反向对照：把 score 系的系数打开，它们**确实非零**。

    没有这条，上面两条可能在「loss 根本没算这一项」的情况下一起变绿 ——
    「贡献 0」和「没算」在值上无法区分。这条把它们区分开。
    """
    from src.networks.katago_v7_loss import KataGoV7Loss
    sp, gl, moves, lbl = v7_batch
    net.eval()
    labels = v7_loss_labels(lbl, moves)
    with torch.no_grad():
        out = net(torch.from_numpy(sp.astype(np.float32)),
                  torch.from_numpy(gl.astype(np.float32)))
    full = KataGoV7Loss(coeff=dict(LOSS_COEFFS))
    res = full(out, labels)
    assert any(float(res['terms'][k]) != 0.0 for k in V7_STAGE1_SCORE_TERMS), \
        'score 系逐项值全为 0：即使权重打开也不产生损失 ⇒ 这一项其实没接上'


# --------------------------------------------------------------------------- #
# 3. futurepos：承重语义（与 / 单路 / 哨兵）
# --------------------------------------------------------------------------- #
def test_futurepos_weight_is_and_not_or():
    """ `w['futurepos']` 是 h0 **AND** h1，绝不是 OR。

    理由（`katago_v7_loss.py` 的 #11 + `_weighted_mean`）：loss 把
    `(b,2,bs²)` 塌成**一个**逐样本标量再乘**一个**权重，而 `_weighted_mean`
    是 `(per_sample*weight).mean()`（**刻意不除 Σw**）⇒ 权重的最小作用单位是
    「整块 2×bs²」。只活一路时若给 1，那一路的 -1 哨兵会被当真值去拟合 tanh，
    头会学出一个恒 −0.76 的假平面。

    本测试用「靠近局尾」的行（只有 h0 或两路都死）来把这条钉死。
     **「只活 h1」在结构上不可能**：偏移是 +8 与 +32，而 `game_ids` 要求两路
    同局 ⇒ h1 活着必然 h0 也活着。所以 AND 与 OR 的差别**恰好**落在
    「只活 h0」这一种情形上，也就是本测试断言的那个。
    """
    ds = make_dataset(n_games=2, game_len=64, seed=21)
    ds.attach_futurepos(source=None, mode='live')
    try:
        # 每局 64 行：i+32 同局需要 i <= 31。
        idxs = np.array([0 * 64 + 40, 1 * 64 + 40])   # 只活 h0
        _, lbl = _v7_labels_and_moves(ds, idxs, np.zeros(2, dtype=np.int64))
        w = lbl['w']
        assert 'futurepos' in w and 'futurepos_h0' in w and 'futurepos_h1' in w
        assert np.all(w['futurepos_h0'] == 1.0), w['futurepos_h0']
        assert np.all(w['futurepos_h1'] == 0.0), w['futurepos_h1']
        assert np.all(w['futurepos'] == 0.0), \
            '单路存活时 w[futurepos] 给了 1 —— 那一路的 -1 哨兵会被当真值拟合'
        # 无效格是 -1 哨兵（不是 0：0 是「合法空盘」）。
        fut = lbl['future']
        dead1 = ~lbl['w']['futurepos_h1'].astype(bool)
        assert np.all(fut[dead1, 1] == -1.0), fut[dead1, 1]
        alive0 = lbl['w']['futurepos_h0'].astype(bool)
        assert set(np.unique(fut[alive0, 0])) <= {-1.0, 0.0, 1.0}, \
            np.unique(fut[alive0, 0])
        # 两路都活 ⇒ 整块权重才是 1。
        both = np.array([0 * 64 + 10, 1 * 64 + 10])
        _, lbl2 = _v7_labels_and_moves(ds, both, np.zeros(2, dtype=np.int64))
        assert np.all(lbl2['w']['futurepos'] == 1.0)
        # 两路都死（局末）⇒ 0。
        dead = np.array([0 * 64 + 60])
        _, lbl3 = _v7_labels_and_moves(ds, dead, np.zeros(1, dtype=np.int64))
        assert float(lbl3['w']['futurepos'][0]) == 0.0
    finally:
        ds._fp = None


def test_futurepos_defaults_to_disabled_on_a_fresh_dataset():
    """默认构造的 dataset 上 futurepos 是**关**的（占位零 + 权重 0）。

    这不是「没接线」的证据，是 A6 之前的历史契约（`dataset.py::__init__` 的
    说明 + 两条既有测试）。V7 是 `--v7` 才打开的开关之一。
    """
    ds = make_dataset(n_games=1, game_len=32, seed=22)
    _, lbl = _v7_labels_and_moves(ds, np.arange(4), np.zeros(4, dtype=np.int64))
    assert float(lbl['w']['futurepos'].sum()) == 0.0
    assert not np.any(np.asarray(lbl['future']))


def test_outcome_black_is_not_outcome_times_to_play(ds):
    """ `outcome_black` 不能由 `outcome * to_play` 推出。

    `0 * -1 == 0` ⇒ 「白胜」会被报成「黑胜」。这条断言直接构造出那个反例：
    to_play=-1（白在走）且 outcome=0（白胜）时，黑方结果必须是 1（负）。
    """
    idxs = np.arange(len(ds))
    _, lbl = _v7_labels_and_moves(ds, idxs, np.zeros(len(ds), dtype=np.int64))
    # 「白胜（to_play=-1 且 outcome=0）」的行；`outcome` 是 to_play 视角。
    to_play = np.asarray(ds.to_play[idxs])
    outcome = np.asarray(lbl['outcome'])
    ob = np.asarray(lbl['outcome_black'])
    white_wins = (to_play == -1) & (outcome == 0)
    assert white_wins.any(), '合成数据里没有「白胜」的行，这条断言会空跑'
    assert np.all(ob[white_wins] == 1), \
        '白胜被报成黑胜 —— 这正是 `outcome * to_play` 的 0 * -1 == 0'
    # 与 dataset 自己的公式逐行一致（不靠重推）。
    expect = np.where(outcome == 2, 2,
                      np.where(to_play == 1, outcome, 1 - outcome))
    assert np.array_equal(ob, expect)


def test_futurepos_label_is_renamed_not_or_bisected(v7_batch):
    """loss 要的键叫 `futurepos`，dataset 叫 `future`：翻译层只改名、不拆。"""
    _sp, _gl, moves, lbl = v7_batch
    out = v7_loss_labels(lbl, moves)
    assert 'futurepos' in out
    assert np.array_equal(np.asarray(out['futurepos']),
                          np.asarray(out['future']))
    assert out['futurepos'].shape == (8, 2, BOARD * BOARD)


# --------------------------------------------------------------------------- #
# 4. labels dict 逐项接线
# --------------------------------------------------------------------------- #
def test_v7_loss_labels_supplies_every_key_the_loss_needs(v7_batch):
    """翻译层补齐 loss 需要的键，且 dense policy 两路形状正确。"""
    _sp, _gl, moves, lbl = v7_batch
    out = v7_loss_labels(lbl, moves)
    for k in ('policy_player', 'policy_opp', 'outcome', 'ownership', 'score',
              'scoring', 'seki', 'sb_center', 'sb_upper', 'game_weight',
              'futurepos', 'w'):
        assert k in out, f'翻译层缺少 {k}'
    assert out['policy_player'].shape == (8, ACTION)
    assert out['policy_opp'].shape == (8, ACTION)
    # policy_player = 本行着法的 one-hot（行和为 1）。
    assert torch.allclose(out['policy_player'].sum(-1),
                          torch.ones(8), atol=1e-6)
    # `w` 是**同一个** dict 语义（含 futurepos / policy_opp 等行权重）。
    assert 'policy_opp' in out['w']


def test_policy_opp_sentinel_row_is_all_zero(v7_batch):
    """`next_move == -1`（局末手）⇒ `policy_opp` 是**全零行**，不是任意一类。

    交给 `F.one_hot(-1)` 的行为在不同 torch 版本里不一致，所以翻译层显式置零。
    权重本来就是 0，但全零行既安全又语义正确。
    """
    lbl = {'next_move': np.array([-1, 0, 5, -1, 7, -1, 2, 3]),
           'future': np.zeros((8, 2, BOARD * BOARD))}
    out = v7_loss_labels(lbl, np.arange(8), action_size=ACTION)
    rows = out['policy_opp']
    assert torch.all(rows[0] == 0) and torch.all(rows[3] == 0)
    assert torch.all(rows[5] == 0)
    assert int(rows[1].argmax()) == 0
    assert int(rows[4].argmax()) == 7


def test_policy_player_handles_pass_class(v7_batch):
    """`moves` 的 `bs*bs`（=361）是 pass 类，必须落在 one-hot 的第 361 位。"""
    mv = np.array([361, 0, 5])
    lbl = {'next_move': np.array([-1, -1, -1]),
           'future': np.zeros((3, 2, BOARD * BOARD))}
    out = v7_loss_labels(lbl, mv, action_size=ACTION)
    assert int(out['policy_player'][0].argmax()) == BOARD * BOARD
    assert float(out['policy_player'][0, BOARD * BOARD]) == 1.0


# --------------------------------------------------------------------------- #
# 5. 对称增强：输入与标签必须同源
# --------------------------------------------------------------------------- #
def test_dihedral_batch_agrees_with_the_dataset_move_transform():
    """ `_dihedral_batch`（空间侧）与 `permute_move_vector`（标签侧）必须是**同一个**
    dihedral 群。

    这是「输入与标签同源」的全部内容：盘面上的点 `(r,c)` 经空间变换落到
    `(r',c')` 的同时，`next_move` / `moves` 那个下标必须也被映射到 `r'*bs+c'`。
    两者不同源的症状是**静默**的 —— loss 照降、top1 也可能看着正常，只有棋力不涨。

     判据刻意**不**是「和 dataset 那段增强代码逐字比」：那需要把
    `sample_batch_numpy` 内部的 `tforms` 拿出来（它拿不出来），于是只能退化成
    「我自己抄一遍和另一个我自己抄的比」—— 那是**循环论证**，改了 dataset 那边
    它照样绿。
    ⇒ 这里用 **dataset 自己的 `permute_move_vector` / `SYMMETRIES` 当 oracle**：
    在单子上（只有一个 +1 的盘面）跑 8 路变换，看落点与标签侧的映射是否一致。
    两段实现分处不同模块、不同用途，不共享代码。
    """
    from src.data.dataset import permute_move_vector
    b = np.zeros((1, 1, BOARD, BOARD), dtype=np.float16)
    # 8 路各取一个互不相同的起点，并全部跑一遍。
    starts = [(0, 0), (0, 5), (3, 7), (9, 9), (12, 2), (18, 18), (5, 18), (15, 0)]
    for t, (r0, c0) in enumerate(starts):
        b[0, 0] = 0.0
        b[0, 0, r0, c0] = 1.0
        tf = np.array([t], dtype=np.int64)
        moved = _dihedral_batch(b, tf)[0, 0]
        r, c = np.argwhere(moved > 0.5)[0]
        expect = permute_move_vector(
            np.array([r0 * BOARD + c0], dtype=np.int64), tf, BOARD)[0]
        assert int(r) * BOARD + int(c) == int(expect), (
            f't={t}：盘面落点 {int(r)},{int(c)}（{(int(r), int(c))}）与标签映射 '
            f'{int(expect)}（{(int(expect) // BOARD, int(expect) % BOARD)}）不一致 '
            f'⇒ 输入与标签不同源（静默错标签）')
        # 变换必须是「重排」而不是「增删」：一颗子进一颗子出。
        assert int(moved.sum()) == 1


def test_dihedral_transforms_are_the_eight_distinct_images():
    """8 路必须给出 8 个**互不相同**的像（否则其中两路退化成恒等/重复）。"""
    rng = np.random.default_rng(5)
    x = (rng.random((1, 3, BOARD, BOARD)) > 0.7).astype(np.float16)
    seen = {}
    for t in range(8):
        out = _dihedral_batch(x, np.array([t], dtype=np.int64))
        assert out.shape == x.shape and out.dtype == x.dtype
        key = out.tobytes()
        assert key not in seen, f't={t} 与 t={seen.get(key)} 给出了同一个像'
        seen[key] = t
    assert _dihedral_batch(x, np.zeros(1, dtype=np.int64)).tobytes() == \
        x.tobytes(), 't=0 必须是恒等变换'
    # 逐行：tforms 逐批不同 ⇒ 各行各自变换（不是整批共用一个）。
    tf = np.array([1, 5, 3], dtype=np.int64)
    rows = _dihedral_batch(np.repeat(x, 3, axis=0), tf)
    for i, t in enumerate(tf):
        assert rows[i].tobytes() == _dihedral_batch(x, np.array([t])).tobytes()


def test_v7_labels_match_dataset_under_same_tform(ds):
    """给定**同一份** tforms，`_v7_labels_and_moves` 与 dataset 逐项一致。

    `_build_labels` 是本仓唯一的标签构造器，而 worker 里必须自己抽 tforms
    （`sample_batch_numpy` 那份拿不到）⇒ 这段是复制品，用「同一 tforms 下
    两条路必须相等」来防漂移。
    """
    idxs = np.arange(32)
    tforms = np.arange(32, dtype=np.int64) % 8
    mv, lbl = _v7_labels_and_moves(ds, idxs, tforms)
    ref = ds._build_labels(idxs, tforms, True)
    for k in ('next_move', 'outcome', 'outcome_black', 'game_weight'):
        assert np.array_equal(np.asarray(lbl[k]), np.asarray(ref[k])), k
    assert np.array_equal(np.asarray(lbl['future']), np.asarray(ref['future']))
    assert np.array_equal(np.asarray(lbl['soft']), np.asarray(ref['soft']))
    assert set(lbl['w']) == set(ref['w'])
    for k in ref['w']:
        assert np.array_equal(np.asarray(lbl['w'][k]), np.asarray(ref['w'][k])), k
    # moves_out 的口径也逐字一致（非法/越界归一到 bs*bs = pass）。
    import src.data.dataset as dsmod
    ref_moves = np.full(32, BOARD * BOARD, dtype=np.int64)
    raw = np.asarray(ds.moves[idxs], dtype=np.int64)
    ok = (raw >= 0) & (raw < BOARD * BOARD)
    ref_moves[ok] = raw[ok]
    dsmod.permute_move_vector(ref_moves, tforms, BOARD)
    assert np.array_equal(mv, ref_moves)


# --------------------------------------------------------------------------- #
# 6. spawn 下 worker 重新 mmap
# --------------------------------------------------------------------------- #
# 探针跑在**真 spawn** 的子进程里。Windows 上不可能用 fork，所以这两条测试
# **不能**偷用 fork 蒙混过去 —— 整个机制（不继承内存、句柄要重开）只存在于
# spawn 下，用 fork 测等于什么都没测。
#
# 为什么探针是「写成两个真文件」而不是 `python -c`：spawn 的 worker 要能
# **按名字**找到 target 函数（pickle 按引用），所以 target 必须在某个**可导入**
# 的模块里。
_PROBE_MOD = r'''
"""子进程探针：在 spawn 的 worker 里量「句柄有没有被重开」「有没有再落盘」。

输入是**父进程 warm 完之后 pickle 出来的 dataset** —— 那正是 spawn 真正传过去
的东西（Windows 上 `mp.Process` 的 payload 就是参数的可 pickle 表示）。
"""
import pickle
import sys

# 必须在**任何** `src.*` import 之前把仓库根塞进 sys.path：探针文件落在
# tmp_path 下，`python probe.py` 不会把 cwd 放进 sys.path，而 spawn 起的 worker
# 又只继承 `sys.path` 的**值**（`PYTHONPATH` 会被继承，这里靠显式 insert）。
_ROOT = sys.argv[1]
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import src.data.kata_label_join as klj

_REMAT = []
_real = klj.materialize_dataset


def _counting(*a, **k):
    _REMAT.append(1)
    return _real(*a, **k)


def _base_chain_has_memmap(arr, depth=8):
    """`arr` 的 base 链上是否还挂着 `np.memmap`（= 仍是按需分页的映射）。"""
    cur = arr
    for _ in range(depth):
        cur = getattr(cur, 'base', None)
        if cur is None:
            return False
        if isinstance(cur, np.memmap):
            return True
    return False


def run(payload, res_q):
    try:
        # 在**子进程里**给落盘函数套计数器。父进程的 monkeypatch 不会穿过 spawn
        # 的 pickle 边界，只能就地包装模块属性 —— `dataset.py::_futurepos_boards`
        # 是函数内 import，它在调用时从模块上取属性，所以包得住。
        klj.materialize_dataset = _counting
        from scripts.train_sft import _rebind_futurepos_mmap

        # 这就是 spawn 传过来的东西：父进程 warm 之后的 dataset 的 pickle。
        # 里面的 `fp['boards']` 是一个**被完整复制出去的 ndarray**（见
        # `test_pickling_a_memmap_materializes_its_whole_payload`），
        # `fp['materialized_paths']` 则只是几个**字符串**。
        with open(payload['pkl'], 'rb') as fh:
            ds = pickle.load(fh)

        # **必须留住一个强引用**（`_inherited`），理由见
        # `test_worker_reopens_mmap_instead_of_inheriting` 的登记注释：`attach_futurepos`
        # 会整体重建 `self._fp`，那份被 pickle 复制过来的 boards 是**最后一个**
        # 引用它的对象，一重建就被回收，分配器随即把同一块地址交给新加载的映射
        # ⇒ 只比 `id()` 的话，10 次里有 3 次会误报「没有重开」。
        _inherited = ds._fp['boards']
        inherited_shape = tuple(_inherited.shape)
        inherited_is_memmap = isinstance(_inherited, np.memmap)
        inherited_id = id(_inherited)
        reference = np.array(_inherited, copy=True)
        reopened = _rebind_futurepos_mmap(ds)
        res_q.put({
            'ok': True,
            # 有强引用兜着 ⇒ `is not` 与 `id()` 都变成确定性的（不再撞地址）。
            'id_changed': id(reopened) != inherited_id,
            'is_a_different_object': reopened is not _inherited,
            'inherited_still_alive': _inherited.shape == inherited_shape,
            'inherited_was_memmap': inherited_is_memmap,
            # `_futurepos_boards` 末尾有一句 `np.asarray(boards)`，它把 memmap
            # **降级成 ndarray 视图**（共享同一块映射、不复制）。所以这里判的
            # 不是「类型还是不是 memmap」，而是「映射还挂着」—— base 链上能
            # 找到 memmap 才说明是按需分页而不是整份物化。
            # 这一条才是「确实重开了」的**语义**判据：传进来的那份是普通
            # ndarray（`inherited_was_memmap` 为假），所以「重开后 base 链上有
            # memmap」不可能在没重开的情况下成立。
            'mmap_backed': _base_chain_has_memmap(reopened),
            'shape_ok': tuple(reopened.shape) == tuple(reference.shape),
            'dtype_ok': str(reopened.dtype) == str(reference.dtype),
            'values_match': bool(np.array_equal(np.asarray(reopened), reference)),
            'remat_calls': len(_REMAT),
        })
    except Exception as e:  # noqa: BLE001
        import traceback
        res_q.put({'ok': False, 'error': f'{type(e).__name__}: {e}',
                   'traceback': traceback.format_exc()[-2000:]})


def main():
    import json
    import multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    payload = json.loads(sys.argv[2])
    q = mp.Queue()
    p = mp.Process(target=run, args=(payload, q))
    p.start()
    p.join(300)
    try:
        res = q.get_nowait()
    except Exception:  # noqa: BLE001
        res = {'ok': False, 'error': f'子进程没回结果（exitcode={p.exitcode}）'}
    print('PROBE ' + json.dumps(res))


if __name__ == '__main__':
    main()
'''


def _write_probe(tmp_path):
    p = tmp_path / '_spawn_probe.py'
    p.write_text(_PROBE_MOD, encoding='utf-8')
    return p


def _run_probe_spawn(probe_path, payload):
    """在**真 spawn** 子进程里跑探针，返回 ``(returncode, result_dict)``。"""
    out = subprocess.run(
        [sys.executable, str(probe_path), str(ROOT), json.dumps(payload)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=900)
    res = None
    for line in out.stdout.splitlines():
        if line.startswith('PROBE '):
            res = json.loads(line[len('PROBE '):])
    if res is None:
        raise AssertionError(
            'spawn 探针没回结果。\nstdout:\n%s\nstderr:\n%s'
            % (out.stdout[-3000:], out.stderr[-3000:]))
    assert res.get('ok'), \
        f'spawn 子进程内部失败：{res.get("error")}\n{res.get("traceback", "")}'
    return out.returncode, res


def test_pickling_a_memmap_materializes_its_whole_payload(tmp_path):
    """ **前提**本身也要钉：pickle 一个 `np.memmap` 会把**整份数据**塞进去。

    这是「12.3 GB × worker 数」最直接的成因，也是
    `tests/test_train_sft_v7.py::test_worker_reopens_mmap_instead_of_inheriting`
    与 `test_worker_does_not_rewrite_the_boards_file` 两条 spawn 实测要防的那件事。

    实测（numpy 1.26.4 / CPython 3.11，本机）：12 B 的数组过一趟
    `pickle.dumps` 得到 162 B 的 payload，且往返后的对象**丢失了映射信息**
    （`filename is None`、`offset is None`）—— 也就是说它不再是「按需分页的
    映射」，而是一份实打实被复制出去的内存。
    ⇒ **spawn 通道绝不能带着一个已解析的 boards 映射**。所以 worker 里必须重开。
    """
    import pickle
    p = tmp_path / 'b.npy'
    arr = np.arange(4096, dtype=np.int8).reshape(16, 16, 16)
    np.save(p, arr)
    mm = np.load(p, mmap_mode='r')
    assert isinstance(mm, np.memmap) and mm.filename is not None
    payload = pickle.dumps(mm)
    back = pickle.loads(payload)
    # 映射信息没了（真正的映射会带上 filename/offset）。
    assert getattr(back, 'filename', None) is None
    assert getattr(back, 'offset', None) is None
    # payload 里装的是**数据**（≥ 数组字节数），不是偏移量。
    assert len(payload) >= arr.nbytes, \
        (f'pickle payload {len(payload)} B < 数据 {arr.nbytes} B：'
         f'numpy 已经改成传偏移量了 ⇒ 本测试的前提失效，需要重读 '
         f'_rebind_futurepos_mmap 的理由')
    assert np.array_equal(np.asarray(back), arr)


def _dump_small_npz(tmp_path, npz_name, mat_name, seed):
    """落一份小合成 npz + 返回 (npz 路径, 物化目录)。"""
    ds = make_dataset(n_games=2, game_len=16, seed=seed)
    npz = tmp_path / npz_name
    np.savez(npz, boards=ds.boards, my_hist=ds.my_hist, op_hist=ds.op_hist,
             ko=ds.ko, moves=ds.moves, values=ds.values, to_play=ds.to_play,
             game_ids=ds.game_ids)
    return npz, tmp_path / mat_name


def _warmed_parent_with_materialized_boards(tmp_path, npz, mat):
    """父进程：落盘 + warm（**恰好一次**），并把 warm 后的 dataset pickle 出来。

    返回 ``(pickle 路径, boards.npy 路径, 调用计数)``。那个 pickle 就是 spawn
    真正传给 worker 的东西。
    """
    import pickle
    import src.data.kata_label_join as klj
    calls = []
    _real = klj.materialize_dataset

    def counting(*a, **k):
        calls.append(1)
        return _real(*a, **k)

    klj.materialize_dataset = counting
    try:
        parent = _load_npz_dataset(npz)
        parent.attach_futurepos(dataset_npz=str(npz), materialized_dir=str(mat),
                                mode='live')
        status = parent.warm_futurepos()
        assert status['resolved'] is True
        assert len(calls) == 1, f'父进程 warm 应落盘恰好 1 次，实得 {len(calls)}'
        # 幂等：再 warm 一次不得再落盘。这是「只在父进程发生一次」的前半段。
        parent.warm_futurepos()
        assert len(calls) == 1, f'warm 不幂等：第二次又落盘了（共 {len(calls)} 次）'
        # 关键：warm 之后 `materialized_paths` 必须有值 —— worker 侧就是靠这几个
        # **字符串**找到该重开哪个文件。没有它，spawn 的 worker 就只能退回
        # 继承（或整份物化），也就是这里要防的那两种结局。
        assert parent._fp.get('materialized_paths'), \
            'warm 之后没有 materialized_paths：worker 侧将无处可重开'
        pkl = tmp_path / 'warmed.pkl'
        pkl.write_bytes(pickle.dumps(parent))
    finally:
        klj.materialize_dataset = _real
    return pkl, mat / 'boards.npy', len(calls)


def test_worker_reopens_mmap_instead_of_inheriting(tmp_path):
    """ 在**真 spawn** 子进程里：worker 重新 mmap，而不是用继承来的那份数据。

    为什么必须实测：Windows（以及 macOS 3.8+）上 `mp.Process` 是 **spawn**，
    spawn **不继承内存** —— 父进程里 `warm_futurepos()` 解析好的 boards 会随
    dataset 的 pickle **整份被复制**（见
    `test_pickling_a_memmap_materializes_its_whole_payload`）。症状是 OOM 或
    「每个 worker 各持一份」，**都不会**报「句柄没继承」，所以只能靠断言钉。
    （本探针在所有平台上都 `mp.set_start_method('spawn', force=True)`，所以
    「Windows spawn / Linux fork」这个差异**不**是本测试的敏感来源。）

     **登记（2026-10-02）：原 `assert r['id_changed']` 这一条是错的，已改判据**
    ------------------------------------------------------------------------
    原断言用 `id(reopened) != inherited_id` 判定「worker 有没有重开」。这是
    **地址比较**，而 CPython 会回收地址：`attach_futurepos(source=..., mode=
    'live')` 整体重建 `self._fp`，那份被 pickle 复制过来的 boards 是**最后一个**
    引用它的对象 ⇒ 一重建它就被释放 ⇒ 分配器随即把**同一块地址**交给新 `np.load`
    出来的映射。于是 `id()` 相等，而重开其实**成功了**。

    实测（本机 CPython 3.12.4 / numpy 1.26.4，连续 10 次同一条探针）：
        id_changed=False 而 mmap_backed=True —— 3 次
        id_changed=True  而 mmap_backed=True —— 7 次
    `mmap_backed` **10/10 全为 True**，即重开每次都发生了；变的只是地址。
    再加一个强引用（`_inherited` 不被回收）后 `id_changed` 变成 **8/8 全 True**，
    这就是地址复用而非「没重开」的决定性证据。

    所以：实现**没有**问题（`_rebind_futurepos_mmap` 确实重开了），是**判定方式**
    在本平台上不成立。改成两条都成立的判据，并把语义判据放在最前面 ——
    `mmap_backed` 才是「确实重开了」的真正证据：传进来的那份是普通 ndarray
    （下面 `inherited_was_memmap` 已钉），它**不可能**有 memmap base。
    """
    npz, mat = _dump_small_npz(tmp_path, 'ds.npz', 'mat', seed=3)
    probe = _write_probe(tmp_path)
    code, r = _run_probe_spawn(
        probe, {'pkl': str(_warmed_parent_with_materialized_boards(
            tmp_path, npz, mat)[0])})

    assert code == 0, f'spawn 子进程退出码 {code}'
    # 传进来的那份**不是**映射（被 pickle 复制成了普通数据）。
    assert not r['inherited_was_memmap'], \
        'pickle 竟然保住了映射信息：前提变了，_rebind_futurepos_mmap 的理由要重写'
    # ---- 「确实重开了」的语义判据（主判据）--------------------------------
    assert r['mmap_backed'], \
        '重开之后不是映射（base 链上没有 memmap）⇒ 整份物化进了 worker 内存'
    # ---- 身份判据（辅助）：探针已持有强引用，故对象身份比较是确定的 --------
    assert r['inherited_still_alive'], '探针自检：持有的引用没活下来？'
    assert r['is_a_different_object'] and r['id_changed'], \
        'worker 直接用了传进来的那份数据（没有重开）'
    assert r['shape_ok'] and r['dtype_ok'] and r['values_match'], r


def test_worker_does_not_rewrite_the_boards_file(tmp_path):
    """ 落盘**只在父进程发生一次**；worker 里重开映射**绝不**再次落盘。

    这是「12.3 GB × worker 数」最容易写出来的地方：`_futurepos_boards` 的第三条
    分支会调落盘函数，而 worker 是在第一次做邻行 gather 时才惰性走到那里的 ——
    若不显式 warm，那就是每个 worker 各落一份（还可能同时写同一个路径互相踩坏）。
    """
    npz, mat = _dump_small_npz(tmp_path, 'ds2.npz', 'mat2', seed=4)
    pkl, boards_npy, parent_calls = _warmed_parent_with_materialized_boards(
        tmp_path, npz, mat)
    assert parent_calls == 1
    mtime, size = boards_npy.stat().st_mtime_ns, boards_npy.stat().st_size

    probe = _write_probe(tmp_path)
    _code, r = _run_probe_spawn(probe, {'pkl': str(pkl)})
    assert r['remat_calls'] == 0, \
        (f'worker 里又落盘了 {r["remat_calls"]} 次 —— 这就是「12.3 GB × '
         f'worker 数」')
    # 落盘产物没被动过（worker 只读映射，不重写文件）。
    assert boards_npy.stat().st_mtime_ns == mtime
    assert boards_npy.stat().st_size == size


def _load_npz_dataset(npz):
    z = np.load(npz, allow_pickle=False)
    try:
        data = {k: z[k] for k in z.files}
    finally:
        z.close()
    return SupervisedDataset(data, n_channels=12)


# --------------------------------------------------------------------------- #
# 7. 默认路径不变 + CLI
# --------------------------------------------------------------------------- #
def test_prefetch_workers_default_is_twelve():
    """C0 基准实测给出的默认是 **12**（16 核上 8 worker 边际效率已 0.701）。"""
    assert _default_of('--prefetch-workers') == 12


def test_v7_flag_defaults_to_off():
    """`--v7` 默认 0 ⇒ 默认走 12 通道路径。"""
    assert _default_of('--v7') == 0


def test_katago_se_cfg_is_untouched():
    """ `KATAGO_SE_CFG` 没被删也没被改（12 通道 / 9.11M 仍是唯一现役基线）。"""
    from scripts.train_sft import KATAGO_SE_CFG
    assert KATAGO_SE_CFG['in_channels'] == 12
    assert KATAGO_SE_CFG['params_total'] == 9_112_005
    assert KATAGO_SE_CFG['params_backbone'] == 8_392_995
    assert KATAGO_SE_CFG['arch'] == 'se_bottleneck'


def test_default_path_payload_is_unchanged_by_v7(ds):
    """默认路径（v7=False）的取批仍是 numpy 三元组，且形状就是 12 通道。"""
    a = ds.sample_batch_numpy(np.arange(8), rng=np.random.default_rng(0))
    assert len(a) == 3, 'labels=False 时 payload 必须仍是三元组（不因 V7 变成 4 元组）'
    states, moves, values = a
    assert states.shape == (8, 12, BOARD, BOARD)   # 12 通道，不是 22
    assert moves.shape == (8,) and values.shape == (8, 1)


# --------------------------------------------------------------------------- #
# 7. 上报面：健康度**两条路**各自取对的头
# --------------------------------------------------------------------------- #
# 下面这几条 exec 的是 main() 里那段**真实语句**，不是复刻一份 —— 健康度必须
#   **就地**构造在 `if _do_swanlab:` 分支里（不许抽成 helper：
#   `tests/test_swanlab_metrics.py` 的三条门禁按字面量在 main() 里定位它，见
# `scripts/train_sft.py` 上报分支里那段 注释）。做法与
#   `tests/test_huber_loss.py::test_log_loss_identity` 同源：抠 AST → exec。
def _health_block_node():
    """main() 里 `if _do_swanlab:` 分支下那段 `with torch.no_grad():` 的 AST。"""
    _tree, main_fn = _main_tree()
    for node in ast.walk(main_fn):
        if not (isinstance(node, ast.If)
                and ast.unparse(node.test) == '_do_swanlab'):
            continue
        for st in node.body:
            if (isinstance(st, ast.With)
                    and ast.unparse(st.items[0].context_expr) == 'torch.no_grad()'):
                return st
    raise AssertionError(
        'main() 的 `if _do_swanlab:` 分支里没有 `with torch.no_grad():` —— '
        '健康度块被挪走了？（`tests/test_swanlab_metrics.py::'
        'test_health_metrics_only_computed_on_log_steps` 会先红）')


def _run_health_block(env):
    """把 main() 里那段健康度代码**真跑一遍**，返回它构造出的那个 dict。"""
    ns = {'torch': torch, 'math': math, '_v7_on': False}
    ns.update(env)
    exec(compile(ast.Module(body=[_health_block_node()], type_ignores=[]),
                 '<train_sft.health>', 'exec'), ns)  # noqa: S102
    return ns['_health_last']


def _assert_all_plain_floats(h):
    """上报面契约：swanlab 收到的必须是 python float，且本路径上都得是有限值。"""
    for k, v in h.items():
        assert type(v) is float, f'{k} 是 {type(v).__name__}，不是 python float'
        assert math.isfinite(v), f'{k} = {v} 不是有限值'


def test_build_param_groups_partitions_v7_exactly_once(net):
    """V7 的 `value_head.*` 不得同时落进 value 组与 other 组（否则参数被更新两遍）。

    V7 的头叫 `value_head.`，不匹配 `_build_param_groups` 原本的 `'value.'`
    前缀判据 ⇒ 光靠前缀会把同一批参数收进两组。
    """
    from scripts.train_sft import _build_param_groups
    import argparse
    args = argparse.Namespace(lr=1e-3, weight_decay=1e-4, value_lr_mult=5.0)
    groups = _build_param_groups(net, args)
    seen = []
    for g in groups:
        seen.extend(g['params'])
    ids = [id(p) for p in seen]
    assert len(ids) == len(set(ids)), '有参数出现在多组里'
    assert len(ids) == len(list(net.parameters()))
    value_group = {id(p) for g in groups[2:4] for p in g['params']}
    assert {id(p) for p in net.value_head.parameters()} == value_group


def test_health_block_reports_rmse_on_the_default_path():
    """ 12 通道默认路径：6 个键齐、值**可精确复算**、基线真的垫底。

    **为什么这条以前不存在**：B8 把健康度抽成 `compute_training_health()` 时，
    只给 V7 那条路补了运行时断言，12 通道那一侧的
    `pytest.raises(AttributeError)` 只证明「参数不能是 None」，并没有验证过
    任何一个指标算得对 —— 于是一处能把默认路径的 `value_rmse` 算错的改动可以
    全绿通过整个门禁。这与上一轮 9 个失败没被自测发现是同一类漏洞（覆盖面）。
    """
    g = torch.Generator().manual_seed(11)
    b, a = 8, ACTION
    moves = torch.randint(0, a, (b,), generator=g)
    # 构造一份**答案已知**的 logits：前 4 行 argmax 就是真着法（top1 命中），
    # 后 4 行把真着法压在第 5 位、argmax 落在别处（top1 不中、top5 中）。
    # ⇒ top1 必为 4/8 = 0.5、top5 必为 8/8 = 1.0，可以**精确**断言而不是
    # 「看起来在合理范围内」。
    logits = torch.full((b, a), -1.0)
    # 下面**不用** `logits[:4, moves[:4]] = 6.0`：切片与张量索引混用时
    #   高级索引会广播成 4×4 的**笛卡尔积**（实测 torch 2.12：四行四列全被置
    #   成 6.0），于是 top1 变成 1/8 而不是 1/2。用 `arange` 配对索引。
    logits[torch.arange(4), moves[:4]] = 6.0
    for r in range(4, b):
        logits[r, :] = -1.0
        for rank, off in enumerate((1, 2, 3, 4, 0), start=1):
            logits[r, (moves[r] + off) % a] = float(6 - rank)
    # value：`value_t + 小噪声` ⇒ 真的「学到了」的预测，实测 RMSE 必须低于
    # 恒预测 0 的基线 —— 这正是那对基线存在的全部意义。
    tgt = torch.rand(b, 1, generator=g) * 2 - 1
    pred = tgt + 0.1 * torch.randn(b, 1, generator=g)

    h = _run_health_block({'policy_logits': logits, 'value_logit': pred,
                           'move_t': moves, 'value_t': tgt})
    assert set(h) == {'train_top1', 'train_top5', 'value_rmse',
                      'policy_ce_random', 'value_rmse_zero', 'policy_entropy'}, h
    _assert_all_plain_floats(h)
    assert h['train_top1'] == pytest.approx(0.5), h
    assert h['train_top5'] == pytest.approx(1.0), h
    assert h['policy_ce_random'] == pytest.approx(math.log(a)), h
    # `policy_entropy` 的符号：**2026-10-03 已裁决改成真熵**，本断言随之改写。
    #
    #   **旧约定（已作废）**：实现报的是 `Σ p·log p`，那不是熵，是 **−H** ⇒
    #   取值 ∈ [−log A, 0]，均匀时 ≈ **−5.89**，学到后**升向 0**。旧断言写的是
    #   `-log(a) - 1e-6 <= h['policy_entropy'] <= 0.0`。
    #   **为什么必须改**：指标名叫 entropy 而方向是反的 —— 读者看到曲线从 −5.89
    #   升到 0 会以为策略在**变乱**，真实情况是**变锐**（熵在降）。
    # 旧约定当时只是「把既成事实写进了测试」，**没有**任何论证支持它对；
    #   覆盖面上它也只由 `test_health_metric_is_reported` 的「键在不在」间接护住，
    #   符号本身无门禁 ⇒ 要改正是一次显式裁决，而不是静默改数字。
    #   **新约定**：`H = −Σ p·log p`，取值 ∈ [0, log A]，均匀时 = log(362)
    #   ≈ 5.8926，学到后**下降** ⇒ 曲线方向与指标名一致。
    #   **代价**：历史 run 的这条曲线符号翻转，换算关系逐位成立 **新值 = −旧值**
    #   （见下面的 `test_policy_entropy_is_true_entropy_and_flips_sign`）。
    # 故意**不同时上报两个口径** —— 那会让 SwanLab 里出现两条含义重叠、
    #   符号相反的曲线，比一个反了的曲线更难读。
    assert 0.0 <= h['policy_entropy'] <= math.log(a) + 1e-6, h
    # 两个 value 键的值可**逐项复算**（防止有人把两个键写反，或把
    # `value_t` 与 `value_logit` 搞混 —— 那正是 `compute_value_loss` 历史上
    # 咬过人的方向）。
    assert h['value_rmse_zero'] == pytest.approx(
        float(torch.sqrt((tgt ** 2).mean())), abs=1e-6), h
    assert h['value_rmse'] == pytest.approx(
        float(torch.sqrt(((pred - tgt) ** 2).mean())), abs=1e-6), h
    assert h['value_rmse'] < h['value_rmse_zero'], \
        '会预测的模型必须低于恒预测 0 的基线，否则这对比没有意义'


def test_policy_entropy_is_true_entropy_and_flips_sign():
    """ `policy_entropy` 的**口径**与**换算关系**（2026-10-03 裁决的两个锚点）。

    三件事一起钉，缺一件就留了回归的口子：

    1. **均匀分布 ⇒ log(动作数)** —— 19 路 361+1 = 362 个动作，log(362)
       ≈ 5.8926。真熵在均匀时取上界，这是「报的是 H 而不是 −H」最直接的判据：
       旧口径在这里给的是 **−5.89**。
    2. **确定性分布 ⇒ ≈ 0** —— 真熵在下界。
    3. **旧值 = −新值** —— 换算关系逐位成立，历史 run 的曲线才能被平移过来读。
    """
    a = ACTION

    def reported(logits):
        """走 main() 里那段健康度代码本身（不重算公式），返回它报的值。"""
        h = _run_health_block({
            'policy_logits': logits,
            'value_logit': torch.zeros(logits.shape[0], 1),
            'move_t': torch.zeros(logits.shape[0], dtype=torch.long),
            'value_t': torch.zeros(logits.shape[0], 1),
        })
        return h['policy_entropy']

    # (1) 均匀：全零 logits ⇒ softmax 均匀。
    uni = reported(torch.zeros(4, a))
    assert uni == pytest.approx(math.log(a), abs=1e-6), uni
    assert abs(uni - 5.8926) < 1e-3, f'实测 {uni}，与 log(362)≈5.8926 不符'

    # (2) 确定性：一路 +60、其余 −60 ⇒ softmax 已是 one-hot（float32 下
    #     非主项 exp(−120) 已下溢到 0），真熵必须 ≈ 0。
    det_logits = torch.full((4, a), -60.0)
    det_logits[:, 7] = 60.0
    det = reported(det_logits)
    assert det == pytest.approx(0.0, abs=1e-6), det

    # (3) 换算关系：旧口径 `Σ p·log p` 与新口径逐位互为相反数。
    #     用一批**非平凡** logits（真熵严格落在 (0, log A) 内）而不是上面两个
    #     端点 —— 端点上 `new == -old` 是恒真的（0/−logA），证明不了什么。
    g = torch.Generator().manual_seed(23)
    lp = torch.randn(8, a, generator=g).log_softmax(-1)
    new_val = float(-(lp.exp() * lp).sum(-1).mean())      # H
    old_val = float((lp.exp() * lp).sum(-1).mean())       # −H（旧实现）
    assert 0.0 < new_val < math.log(a), new_val            # 真的落在内区间
    assert new_val == pytest.approx(-old_val, abs=1e-6), (new_val, old_val)
    # 且上报的那个数**就是 H 本身**（不是 −H、也不是任何归一化后的东西）：
    # 同一批 logits 走 main() 里的健康度代码，与本地算的 H 必须逐位相等。
    probe = torch.randn(8, a, generator=torch.Generator().manual_seed(31))
    lp2 = probe.log_softmax(-1)
    assert reported(probe) == float(-(lp2.exp() * lp2).sum(-1).mean())


def test_stage1_four_objectives_are_bit_identical_across_score_stdev_betas(net,
                                                                         v7_batch):
    """ `SCORE_STDEV_SOFTPLUS_BETA` 0.05 → 1.0 对**段 1 逐位无影响**。

    为什么这条不能省：段 1 的 8 项 score 系系数逐个 0.0（`test_stage1_score_family_
    coefficients_are_exactly_zero`），而 `score_stdev` 是**唯一没有行权重可用**
    的一项（只有 `game_weight`）⇒ 一旦有人顺手把 beta 改回去、或把系数表动一下，
    段 1 的四个主目标会**静默**跟着变，而曲线看上去仍然「在学」。

    所以这里不是断言「近似相等」，是断言 `float(...)` 的**逐位相等**（`==`，
    不是 `pytest.approx`）—— 近似相等挡不住「值确实动了但动得小」。
    """
    from src.networks import katago_v7 as k7

    sp, gl, moves, lbl = v7_batch
    sp_t = torch.from_numpy(sp.astype(np.float32))
    gl_t = torch.from_numpy(gl.astype(np.float32))
    labels = v7_loss_labels(lbl, moves)

    snap = {}
    orig = k7.SCORE_STDEV_SOFTPLUS_BETA
    try:
        for tag, beta in (('old', 0.05), ('new', 1.0)):   # spec 字面值 / 裁决值
            k7.SCORE_STDEV_SOFTPLUS_BETA = beta
            lossf = build_v7_stage1_loss()
            net.eval()
            with torch.no_grad():
                res = lossf(net(sp_t, gl_t), labels)
            snap[tag] = {
                'terms': {t: float(res['terms'][t]) for t in V7_STAGE1_TERMS},
                'weighted': {t: float(res['weighted'][t]) for t in V7_STAGE1_TERMS},
                # 反证：#7 **确实**变了（否则下面那条「逐位不变」是因为 beta
                # 根本没进到 forward 里，而不是因为段 1 真的不受影响）
                'sd': float(res['terms']['score_stdev']),
            }
    finally:
        k7.SCORE_STDEV_SOFTPLUS_BETA = orig

    assert snap['new']['sd'] != snap['old']['sd'], \
        'beta 改了但 #7 的公式值没变 ⇒ 测的不是「beta 影响段 1」这件事'
    for t in V7_STAGE1_TERMS:
        assert snap['old']['terms'][t] == snap['new']['terms'][t], \
            f'段 1 的 {t} 逐项值随 beta 变了：{snap["old"]["terms"][t]} → ' \
            f'{snap["new"]["terms"][t]}'
        assert snap['old']['weighted'][t] == snap['new']['weighted'][t], \
            f'段 1 的 {t}（加权后）随 beta 变了：{snap["old"]["weighted"][t]} → ' \
            f'{snap["new"]["weighted"][t]}'


def test_health_block_reports_3class_accuracy_on_the_v7_path(net, v7_batch):
    """V7 的 value 是 3 分类 CE ⇒ 报 `value_acc3` + 三类占比/多数类基线，且
    **不假装** value_rmse。

压成标量的任何做法都是新发明的口径（且与 12 通道的 `value_rmse` 不可比）
⇒ 那两个键给 `nan`（键仍在，图上形状不变），而不是编一个数。

`value_prior{0,1,2}` / `value_prior_top1` 是 `value_acc3` 的**地板**：三分类里
「永远猜多数类」就能拿到 max(p)，可能远高于 1/3 而毫无判别力 ⇒ 没有基线时
`value_acc3` 不可判读（2026-10-09 实测：acc 钉在 0.50、CE 已低于 ln(3)，正是
「先验学到了、判别力没有」）。
"""
    sp, gl, moves, lbl = v7_batch
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(sp.astype(np.float32)),
                  torch.from_numpy(gl.astype(np.float32)))
    h = _run_health_block({
        '_v7_on': True, 'out': out,
        'move_t': torch.from_numpy(np.ascontiguousarray(moves)), 'lbl': lbl,
    })
    assert {'train_top1', 'train_top5', 'policy_ce_random',
            'policy_entropy', 'value_rmse', 'value_rmse_zero',
            'value_acc3', 'value_prior0', 'value_prior1', 'value_prior2',
            'value_prior_top1'} == set(h), h
    _assert_all_plain_floats({k: v for k, v in h.items()
                              if k not in ('value_rmse', 'value_rmse_zero')})
    assert 0.0 <= h['value_acc3'] <= 1.0, h
    assert 0.0 <= h['train_top1'] <= 1.0, h
    assert math.isnan(h['value_rmse']) and math.isnan(h['value_rmse_zero']), \
        'V7 下的 value_rmse / value_rmse_zero 必须是 nan（不假装同一个口径）'


def test_value_prior_keys_reconstruct_the_floor(net, v7_batch):
    """三个占比必须**加起来等于 1**，且 `value_prior_top1` 等于其中最大者。

    这条不是形式主义：`value_acc3` 的全部判读价值就在「它与地板比」，
    而地板是这几个键算出来的。若它们不自洽（例如把 `w==0` 的净化行也算进
    分母、或类别索引写错），基线就会**静默错**，`value_acc3` 随之不可读 ——
    且没有任何异常会报出来。
    """
    sp, gl, moves, lbl = v7_batch
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(sp.astype(np.float32)),
                  torch.from_numpy(gl.astype(np.float32)))
    h = _run_health_block({
        '_v7_on': True, 'out': out,
        'move_t': torch.from_numpy(np.ascontiguousarray(moves)), 'lbl': lbl,
    })
    priors = [h['value_prior0'], h['value_prior1'], h['value_prior2']]
    assert abs(sum(priors) - 1.0) < 1e-6, \
        '三个类的占比之和必须为 1（分母算错会让基线静默失真）：%r' % (priors,)
    assert abs(h['value_prior_top1'] - max(priors)) < 1e-12, \
        'value_prior_top1 必须等于三类占比的最大者：%r vs %r' % (
            h['value_prior_top1'], max(priors))
    # 地板必须真的是「永远猜多数类」能做到的水平 —— 即 ≥ 1/3。
    assert h['value_prior_top1'] >= 1.0 / 3.0 - 1e-9, h


def test_health_block_never_touches_the_12ch_names_under_v7():
    """ V7 下 `policy_logits` / `value_logit` / `value_t` 在 main() 里**未被绑定**。

    （V7 走 `model(state, gl)` 返回 dict，不产出 `(policy_logits, value_logit)`
    二元组；`value_t` 那条路也不存在。）所以健康度块里任何一处对它们的**无条件**
    求值都会抛 UnboundLocalError —— 症状是「V7 训练跑到第一次打点才炸」，
    pyflakes/ruff 都不会报（它是个合法的局部变量，只是那条路没赋值）。

    证明分两半：
      1. `test_health_block_reports_3class_accuracy_on_the_v7_path` 把这三个名字
         **故意不提供**给 `_run_health_block` 的命名空间却跑通了 —— 这是实证；
      2. 这里从 AST 上钉住「每个出现都在 `_v7_on` 的 else 侧」。条件表达式**只
         求值被选中的那一支**，所以 `A if _v7_on else <name>` 是安全的；不安全
         的是把 `<name>` 放在无条件下（或 `if not _v7_on` 的**真**侧以外）的位置。
    """
    blk = _health_block_node()
    guarded = [n.orelse for n in ast.walk(blk)
               if isinstance(n, ast.IfExp) and ast.unparse(n.test) == '_v7_on']
    assert guarded, ('健康度块里没有任何 `_v7_on` 的条件表达式 ⇒ 分支判据被改掉了？')

    def _protected(node):
        return any(any(x is node for x in ast.walk(g)) for g in guarded)

    for name in ('policy_logits', 'value_logit', 'value_t'):
        uses = [n for n in ast.walk(blk)
                if isinstance(n, ast.Name) and n.id == name]
        assert uses, f'{name} 在健康度块里根本没用到 —— 分支判据被改掉了？'
        for n in uses:
            assert _protected(n), (
                f'`{name}`（第 {n.lineno} 行）出现在不受 `_v7_on` 保护的位置 ⇒ '
                'V7 下 UnboundLocalError')


# ---- 小工具 --------------------------------------------------------------- #
def _module():
    import scripts.train_sft as m
    return m


def _default_of(flag):
    """从 main() 的 AST 里读某个 flag 的 default（不真跑 argparse）。"""
    import ast
    src = (ROOT / 'scripts' / 'train_sft.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'add_argument':
            if n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == flag:
                for kw in n.keywords:
                    if kw.arg == 'default':
                        return ast.literal_eval(kw.value)
    raise AssertionError(f'main() 里没有 {flag}')


# --------------------------------------------------------------------------- #
# 8. `warm_futurepos()` 必须在父进程、`fork` 之前
# --------------------------------------------------------------------------- #
def _main_tree():
    import ast
    src = (ROOT / 'scripts' / 'train_sft.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    return tree, next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == 'main')


def _calls_named(fn, name):
    """`fn` 里调用了 `name` 的全部行号（`Name(...)` 与 `attr.name(...)` 都算）。"""
    import ast
    out = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        if getattr(n.func, 'id', None) == name:
            out.append(n.lineno)
        elif getattr(n.func, 'attr', None) == name:
            out.append(n.lineno)
    return out


def test_warm_futurepos_is_called_in_main_before_the_prefetcher_fork():
    """ `warm_futurepos()` 的调用点必须在 `_BatchPrefetcher(...)` **之前**。

    这是本次接线里唯一一处「顺序错了不报错」的调用：`warm_futurepos()` 做的是
    **惰性解析**，若它排在预取器之后，解析就发生在**每个 worker 里**
    ⇒ 12.3 GB × worker 数。实测代价（同一份 gather）：冷 IO 8 worker 273 行/s
    vs warm 1161 行/s（**差 4.3×**）；gather 冷读 10.4–24.7 ms/行 vs 热读
    0.005–0.008（**1500×**）。

    这里按**源码行号**钉（`ast` + `lineno`），而不是靠运行时观测 —— 后者在没开
    `--v7` 时根本不会发生。
    """
    _tree, main_fn = _main_tree()
    warm = _calls_named(main_fn, 'warm_futurepos')
    assert len(warm) == 1, (
        f'main() 里有 {len(warm)} 处 warm_futurepos() 调用（{warm}）；必须**恰好一处**，'
        f'否则「哪一处在 fork 前」这件事就没法用行号钉住')
    fork = [n.lineno for n in ast.walk(main_fn)
            if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == '_BatchPrefetcher']
    # 2026-10-06 起是**两个**构造点：训练预取器 + eval 特征预取池（V7 eval 的
    # 梯子特征串行计算曾让每次 eval 磨 25-30 分钟）。两个池都必须在 fork 前。
    assert len(fork) == 2, f'预取器构造点应为 2（训练 + eval 池），实得 {fork}'
    assert all(warm[0] < f for f in fork), (
        f'warm_futurepos() 在 L{warm[0]}，预取器构造在 {fork} —— '
        f'惰性解析会发生在每个 worker 里')
    # attach 也必须在 fork 之前（否则 worker 看到的 dataset 上 futurepos 是关的，
    # 症状是「futurepos 项恒 0」且不报任何错）。
    attach = _calls_named(main_fn, 'attach_futurepos')
    assert len(attach) == 1 and attach[0] < min(fork), (
        f'attach_futurepos 调用在 {attach}，预取器构造在 {fork}')


def test_v7_flag_exists_and_is_the_only_new_one():
    """`--v7` 是本次唯一新增的旗（`--prefetch-workers` 只改了默认值）。"""
    import ast
    src = (ROOT / 'scripts' / 'train_sft.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    flags = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'add_argument':
            if n.args and isinstance(n.args[0], ast.Constant):
                flags.add(n.args[0].value)
    assert '--v7' in flags
    # 已有的旗一个都不许少（D1：零删除）。
    for f in ('--data', '--policy-loss', '--soft-index', '--soft-only-sampling',
              '--prefetch-workers', '--value-loss-weight', '--label-smoothing'):
        assert f in flags, f'{f} 不见了'


# --------------------------------------------------------------------------- #
# 9. 预取器的 V7 payload
# --------------------------------------------------------------------------- #
def test_prefetch_worker_v7_payload_shape():
    """worker 的 V7 payload 形状与标签口径（真跑 worker 函数，纯 numpy）。"""
    m = _module()
    ds = make_dataset(n_games=2, game_len=32, seed=31)
    ds.attach_futurepos(source=None, mode='live')
    import queue
    task_q, res_q = queue.Queue(), queue.Queue()
    idxs = np.arange(6)
    task_q.put((0, 0, idxs))
    task_q.put(None)
    m._prefetch_worker(0, task_q, res_q, 1234, ds, labels=False, v7=True)
    step, pos, sp, gl, mv, lbl, err = res_q.get_nowait()
    assert err is None, err
    assert (step, pos) == (0, 0)
    assert sp.shape == (6, V7_SPATIAL_CHANNELS, BOARD, BOARD)
    assert gl.shape == (6, V7_GLOBAL_CHANNELS)
    assert sp.dtype == np.float16 and gl.dtype == np.float16
    assert mv.shape == (6,) and lbl['future'].shape == (6, 2, BOARD * BOARD)
    ds._fp = None


def test_prefetch_worker_default_payload_is_unchanged():
    """`v7=False`（默认）的 payload 仍是 12 通道三元组 + 可选 labels，逐位不变。"""
    m = _module()
    ds = make_dataset(n_games=1, game_len=16, seed=32)
    import queue
    task_q, res_q = queue.Queue(), queue.Queue()
    task_q.put((0, 0, np.arange(4)))
    task_q.put(None)
    m._prefetch_worker(0, task_q, res_q, 1234, ds, labels=False, v7=False)
    _step, _pos, s, mv, v, lbl, err = res_q.get_nowait()
    assert err is None, err
    assert s.shape == (4, 12, BOARD, BOARD)     # 12 通道，不是 22
    assert mv.shape == (4,) and v.shape == (4, 1)
    assert lbl is None                            # labels=False 时不搬 labels


def test_prefetch_worker_reports_errors_instead_of_dying():
    """worker 抛异常时把 err 放回队列（不能静默吞掉，否则训练 hang 在 next()）。"""
    m = _module()
    import queue

    class _Boom:
        board_size = BOARD

        def sample_batch_numpy(self, *a, **k):
            raise RuntimeError('boom')

    task_q, res_q = queue.Queue(), queue.Queue()
    task_q.put((0, 0, np.arange(4)))
    task_q.put(None)
    # 默认路径（v7=False）：`sample_batch_numpy` 确实被调到，所以能验到那个异常
    # 被**捕获并回传**，而不是让 worker 整个死掉（死掉的话主进程会永远卡在
    # `res_q.get()` 上 —— 训练表现为「hang 住」而不是报错）。
    m._prefetch_worker(0, task_q, res_q, 1234, _Boom(), labels=False, v7=False)
    step, pos, s, mv, v, lbl, err = res_q.get_nowait()
    assert (step, pos) == (0, 0)
    assert s is None and mv is None and v is None and lbl is None
    assert isinstance(err, RuntimeError) and 'boom' in str(err)


def test_prefetcher_next_returns_v7_quadruple(tmp_path):
    """`_BatchPrefetcher(v7=True).next()` 返回 `(spatial, gl, moves, labels)`。

     这里用 `prefetch-workers=1` 之外的方式构造（`_StubMP`）不可行 —— payload
    的拼接逻辑必须真跑，所以这里起**真进程**（默认 start method，Windows 上
    就是 spawn）。数据只有 16 行，进程启动的固定开销可以接受。
    """
    m = _module()
    ds = make_dataset(n_games=1, game_len=16, seed=33)
    ds.attach_futurepos(source=None, mode='live')
    pf = m._BatchPrefetcher(ds, num_workers=2, prefetch=1, labels=False, v7=True)
    try:
        pf.submit(np.arange(8))
        sp, gl, mv, lbl = pf.next(device=None)
        assert sp.shape == (8, V7_SPATIAL_CHANNELS, BOARD, BOARD)
        assert gl.shape == (8, V7_GLOBAL_CHANNELS)
        assert mv.shape == (8,)
        # labels 必须已转成张量（下一段就是喂进 loss 的）。
        assert torch.is_tensor(lbl['next_move'])
        assert torch.is_tensor(lbl['w']['futurepos'])
        # dataset 侧的键名是 `future`；`futurepos` 是翻译层（`v7_loss_labels`）
        # 才补上的别名 —— 预取器只搬运，不做翻译。
        assert lbl['future'].shape == (8, 2, BOARD * BOARD)
    finally:
        for p in pf._processes:
            p.terminate()
            p.join(timeout=5)
        ds._fp = None


def test_v7_batch_sync_matches_worker_features():
    """`--prefetch-workers <=1` 的回退路径造出的特征与 worker 那条**同形**。"""
    ds = make_dataset(n_games=1, game_len=32, seed=34)
    sp, gl, mv, lbl = v7_batch_sync(ds, np.arange(8), None,
                                    rng=np.random.default_rng(5))
    assert sp.shape == (8, V7_SPATIAL_CHANNELS, BOARD, BOARD)
    assert sp.dtype == np.float16 and gl.dtype == np.float16
    assert gl.shape == (8, V7_GLOBAL_CHANNELS)
    assert mv.shape == (8,) and lbl['future'].shape == (8, 2, BOARD * BOARD)
    # augment=False ⇒ 恒等变换（回退路径在评估/对拍时要用）。
    sp0, gl0, _mv0, _l0 = v7_batch_sync(ds, np.arange(8), None,
                                        rng=np.random.default_rng(5),
                                        augment=False)
    ref_sp, ref_gl = v7_batch_features(ds, np.arange(8))
    assert np.array_equal(sp0, ref_sp) and np.array_equal(gl0, ref_gl)


# --------------------------------------------------------------------------- #
# 10. `--v7` 的坏组合必须在加载数据集之前就拒掉
# --------------------------------------------------------------------------- #
def _run_train_argv(argv):
    """跑一次 train_sft 的 argparse 到校验阶段，返回 ``(rc, stdout+stderr)``。"""
    out = subprocess.run(
        [sys.executable, '-c',
         'import sys, runpy; sys.argv=sys.argv[1:];'
         ' runpy.run_path("scripts/train_sft.py", run_name="__main__")'] + argv,
        capture_output=True, text=True, cwd=str(ROOT), timeout=900,
        input='')
    return out.returncode, (out.stdout or '') + (out.stderr or '')


def test_v7_rejects_non_19_board(tmp_path):
    """`--v7` + `--board-size != 19` ⇒ 启动就拒（不加载数据集）。

    `KataGoV7Loss.forward` 里的 `n_sq = 19*19` 与 `spatial_channels_v7` 的
    `BOARD_SIZE` 都是钉死的，换盘面得到的是形状错或静默错值，不是零回归。
    """
    npz, _mat = _dump_small_npz(tmp_path, 'bs.npz', 'bsmat', seed=41)
    rc, out = _run_train_argv(['--data', str(npz), '--v7', '1',
                               '--board-size', '13', '--device', 'cpu'])
    assert rc != 0, '非 19 路的 --v7 被放行了'
    assert 'board-size' in out or '19' in out, out[-800:]


def test_v7_rejects_soft_index(tmp_path):
    """`--v7` + `--soft-index` ⇒ 启动就拒（两条链会静默互相覆盖）。"""
    npz, _mat = _dump_small_npz(tmp_path, 'soft.npz', 'softmat', seed=42)
    # 造一个最小 soft index（build_soft_index 的产物形状：idx + policy）。
    idx_path = tmp_path / 'soft_index.npz'
    A = BOARD * BOARD + 1
    np.savez(idx_path, idx=np.arange(4, dtype=np.int64),
             policy=np.full((4, A), 1.0 / A, dtype=np.float32))
    rc, out = _run_train_argv(['--data', str(npz), '--v7', '1',
                               '--soft-index', str(idx_path),
                               '--device', 'cpu'])
    assert rc != 0, '--v7 与 --soft-index 同时给了却被放行'
    assert '互斥' in out or 'soft' in out.lower(), out[-800:]


def test_soft_ce_end_to_end_with_soft_only_sampling(tmp_path):
    """`--policy-loss soft_ce` + `--soft-only-sampling` 端到端跑通（12 通道路径）。

     **段 2 必须配 `--soft-only-sampling`**：`soft_ce` 的 `mask=0` 的行贡献
    **恰好 0**（不是退化成 one-hot CE）⇒ 34.2M 行上只有约 1% 的行贡献时，
    policy 项被缩小约 100×。这条断言那两件事同时成立：掩码确实生效（贡献 0）、
    且收窄行空间之后软行占比≈1。
    """
    from scripts.train_sft import (compute_policy_loss, narrow_to_soft_rows,
                                   resolve_policy_loss_kind, soft_cross_entropy)

    ds = make_dataset(n_games=1, game_len=16, seed=51)
    A = BOARD * BOARD + 1
    # 给前 8 行挂软标签，其余 8 行没有。
    sd = ds.attach_soft(np.arange(8, dtype=np.int64),
                        np.full((8, A), 1.0 / A, dtype=np.float32))
    assert sd['n_soft'] == 8

    # (a) `mask=0` 的行贡献恰好 0 —— 直接判据：**改掉那一行的目标，loss 不动**。
    #     （若它在参与，loss 必然变；这比「和某个手算值比大小」更直接，也不依赖
    #     `soft_cross_entropy` 的分母口径是 B 还是 Σmask。）
    logits = torch.randn(2, A, requires_grad=True)
    soft = torch.full((2, A), 1.0 / A)
    mask = torch.tensor([1.0, 0.0])
    garbled = soft.clone()
    garbled[1] = 0.0
    garbled[1, 7] = 1.0                     # 给第 2 行一个完全不同的目标
    a = float(soft_cross_entropy(logits, soft, mask).detach())
    b = float(soft_cross_entropy(logits, garbled, mask).detach())
    assert a == pytest.approx(b, rel=1e-6), \
        (f'mask=0 的行改了目标 loss 就变了（{a} vs {b}）⇒ 它在参与损失，'
         f'退化成了 one-hot CE')
    # 反向对照：mask 全开时，改目标**必须**改 loss —— 否则上面那条可能因为别的
    # 原因恒等而假绿。
    c = float(soft_cross_entropy(logits, garbled, torch.ones(2)).detach())
    assert c != pytest.approx(a, rel=1e-6), 'mask 全开时换目标无影响：测试空跑'
    # 分母恒为 batch 大小 B（不是 Σmask）⇒ 掩码全 0 时这一项为 0。
    assert float(soft_cross_entropy(logits, soft, torch.zeros(2)).detach()) == \
        pytest.approx(0.0, abs=1e-9)

    # (b) `resolve_policy_loss_kind` 在给了 soft-index 时把 soft_ce 接上
    assert resolve_policy_loss_kind('soft_ce', 'idx.npz') == 'soft_ce'
    assert resolve_policy_loss_kind('ce', 'idx.npz') == 'soft_ce'
    assert resolve_policy_loss_kind('ce', None) == 'ce'

    # (c) `--soft-only-sampling` 把训练行空间收窄到「有软标签的行」
    train_idx = np.arange(len(ds))
    narrow = narrow_to_soft_rows(train_idx, ds)
    assert len(narrow) == 8 and set(narrow.tolist()) == set(range(8))
    assert len(narrow) / len(train_idx) == pytest.approx(0.5)

    # (d) 端到端：soft_ce 真能算出一个标量并反传
    mv = torch.tensor([3, 4], dtype=torch.long)
    loss = compute_policy_loss(logits, mv, 'soft_ce',
                               soft=torch.full((2, A), 1.0 / A),
                               soft_mask=mask, soft_weight=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # 掩码生效的直接后果：第 2 行的**梯度必须恰好为 0**（它没被教任何东西）。
    # 这一条是段 2 的核心语义：「只训软行」不是靠 `--soft-weight 0` 实现的，
    # 而是靠掩码把那些行的梯度直接掐掉。
    assert torch.allclose(logits.grad[1], torch.zeros(A), atol=1e-8), \
        'mask=0 的行拿到了梯度'


def test_soft_only_sampling_without_soft_index_is_refused(tmp_path):
    """`--soft-only-sampling` 没有 `--soft-index` ⇒ 启动就拒（不许静默退回全量）。"""
    npz, _mat = _dump_small_npz(tmp_path, 'nos.npz', 'nosmat', seed=52)
    rc, out = _run_train_argv(['--data', str(npz), '--soft-only-sampling', '1',
                               '--device', 'cpu'])
    assert rc != 0
    assert 'soft-index' in out or 'soft' in out.lower(), out[-800:]


# --------------------------------------------------------------------------- #
# eval 的参考着法：软标签缺席时必须退回真实着法
# --------------------------------------------------------------------------- #
class _AlwaysPredict:
    """固定预测同一个着法的假模型（eval 里只当打分器用，不需要真网络）。"""

    def __init__(self, move, action_size=ACTION):
        self.move = int(move)
        self.A = int(action_size)

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, sp, gl):
        b = sp.shape[0]
        lg = torch.full((b, 1, self.A), -50.0, dtype=torch.float32)
        lg[:, 0, self.move] = 50.0
        return {'policy_logits': lg,
                'outcome_logits': torch.zeros(b, 3, dtype=torch.float32)}


class _InjectSoft:
    """把一份软标签注入 `_build_labels` 返回值的薄壳，其余属性全部转发。"""

    def __init__(self, base, soft):
        self.__dict__['_base'] = base
        self.__dict__['_soft'] = np.asarray(soft, dtype=np.float32)

    def __getattr__(self, name):
        base = self.__dict__.get('_base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def _build_labels(self, idxs, tforms=None, augment=False):
        lbl = self._base._build_labels(idxs, tforms, augment)
        idxs = np.asarray(idxs, dtype=np.int64)
        lbl['soft'] = self._soft[idxs].copy()
        lbl['soft_mask'] = np.ones(len(idxs), dtype=np.float32)
        return lbl


def _eval_top1(ds, predict_move):
    """跑一次 CPU eval，返回指标 dict。"""
    return evaluate_metrics_v7(_AlwaysPredict(predict_move), ds,
                               np.arange(len(ds.moves)), 8,
                               torch.device('cpu'), torch.float32,
                               max_batches=0, prefetcher=None)


def test_eval_reference_falls_back_to_real_move_when_soft_absent():
    """无软标签时（full.npz + 不给 `--soft-index`），eval 参考必须是**真实着法**。

    线上配置正是这一条：`SupervisedDataset.soft_row` 从未挂载 ⇒
    `src/data/dataset.py:_build_labels` 返回**全 0** 的 `soft`（`soft_mask` 同样
    全 0）。`evaluate_metrics_v7` 此前无条件拿 `lbl['soft']` 当参考 ⇒
    `argmax(全0) = 0` ⇒ top1/5/10 退化成「模型是否预测 index 0」、KL 恒 0。

    真机 2026-10-07 实测：top1 0.0006 → 0.0002 一路走低，而随机基线
    top10 = 10/362 ≈ 2.76%、实测 0.49%（**低于随机 5.6 倍**）—— 只有「参考恒为
    index 0 且模型越来越自信」能同时解释「越训越差」与「KL 打印 0.0000」。
    brier 不受影响（它读 `lbl['outcome']`），与真机「brier 在正常改善」一致。
    """
    ds = make_dataset()
    ds.moves[:] = 100                      # 全部行的真实着法 = 100（≠0，便于分辨）
    assert not hasattr(ds, 'sample_spatial'), '本例要走 board 级那条路径'
    _, lbl = _v7_labels_and_moves(ds, np.arange(8), np.zeros(8, np.int64))
    assert float(np.abs(lbl['soft']).max()) == 0.0, \
        '前置条件不成立：本例要求「无软标签」'
    assert float(lbl['soft_mask'].sum()) == 0.0, \
        '前置条件不成立：本例要求 soft_mask 全 0'

    m = _eval_top1(ds, 100)                # 模型逐行都预测真实着法
    assert m['n'] > 0, '没评估到任何样本'
    assert m['top1'] == 1.0, \
        f'模型逐行都预测真实着法 100，top1 应为 1.0，实得 {m["top1"]}'
    assert m['top5'] == 1.0 and m['top10'] == 1.0, m


def test_eval_reference_still_uses_soft_argmax_when_soft_present():
    """有软标签时参考仍是 `argmax(soft)` —— 退回真实着法不许盖掉软路径。

    与上一条配成对：这里刻意让 `argmax(soft)`（=7）与真实着法（=100）**不同**，
    两条路径才会给出相反的答案，否则两条断言等价、测不出区别。
    """
    base = make_dataset()
    base.moves[:] = 100
    n = len(base.moves)
    soft = np.full((n, ACTION), 0.001, dtype=np.float32)
    soft[:, 7] = 1.0                        # argmax 恒为 7，与 moves=100 不同
    soft /= soft.sum(axis=1, keepdims=True)
    ds = _InjectSoft(base, soft)

    m_soft = _eval_top1(ds, 7)              # 押 argmax(soft) ⇒ 应全中
    assert m_soft['top1'] == 1.0, \
        f'有软标签时参考应为 argmax(soft)=7，实得 top1={m_soft["top1"]}'
    m_hard = _eval_top1(ds, 100)            # 押真实着法 ⇒ 应全不中
    assert m_hard['top1'] == 0.0, \
        f'有软标签时不该以真实着法为参考，实得 top1={m_hard["top1"]}'