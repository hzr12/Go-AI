# -*- coding: utf-8 -*-
"""局级 sidecar `games.npz` → V7 逐局贴目的接入测试（2026-10-04）。

为什么要有这个测试
------------------
`games.npz` 早就生成了（162,298 局、90.30% 有真贴目），但 `load_from_path` 建
`V7Dataset` 时**不传** `game_komi` ⇒ `game_row_for()` 返回 `None` ⇒

* 全局 **ch5**（`currentSelfKomi/20`）**恒 0**；
* 全局 **ch18** 的三角波按 komi=0 算。

全局 19 维里有 **2 维**依赖贴目。这是个**不报任何错**的缺口：loss 照降、
指标照报、训练照跑，只是模型永远不知道这局是贴 7.5 还是贴 0。

钉住四件事：接上、默认关闭时逐位不变、口径是 game_id 而非行号、短 sidecar 当场报错。
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from train_sft import (  # noqa: E402
    _sidecar_kwargs,
    load_from_path,
    load_games_sidecar,
)

BOARD = 19
A = BOARD * BOARD + 1

SIDE_KEYS = ('g_komi', 'g_score', 'g_rules', 'g_resign', 'g_re', 'g_resign_side')


def _board_npz(path, n_games=4, per_game=4):
    n = n_games * per_game
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        boards=rng.integers(0, 3, size=(n, BOARD, BOARD)).astype(np.int8),
        my_hist=rng.integers(0, 2, size=(n, 4)).astype(np.int8),
        op_hist=rng.integers(0, 2, size=(n, 4)).astype(np.int8),
        ko=rng.integers(0, A, size=n).astype(np.int64),
        moves=rng.integers(0, A - 1, size=n).astype(np.int64),
        values=rng.choice([-1.0, 1.0], size=n).astype(np.float32),
        to_play=rng.choice([-1, 1], size=n).astype(np.int64),
        game_ids=np.repeat(np.arange(n_games), per_game).astype(np.int64),
    )
    return path


def _sidecar(path, n_games, komi=None):
    """造一份契约完整的 sidecar（键集与落盘一致，见 test_games_sidecar_alignment）。"""
    komi = np.full(n_games, 7.5, np.float16) if komi is None else np.asarray(komi, np.float16)
    np.savez_compressed(
        path,
        g_komi=komi,
        g_score=np.where(komi > 0, 2.5, np.nan).astype(np.float16),
        g_rules=np.zeros(n_games, np.int8),
        g_resign=np.zeros(n_games, bool),
        g_re=np.where(komi > 0, 1, 3).astype(np.int8),
        g_resign_side=np.full(n_games, -1, np.int8),
    )
    return path


# --------------------------------------------------------------------------- #
# 接上
# --------------------------------------------------------------------------- #
def test_komi_reaches_the_dataset_and_through_to_ch5(tmp_path):
    """🔴 贴目必须真的进到全局特征，不能只是「命令没报错」。"""
    d = _board_npz(tmp_path / 'board.npz')
    s = _sidecar(tmp_path / 'games.npz', 4)
    off = load_from_path(str(d), BOARD, 0, v7=True)
    on = load_from_path(str(d), BOARD, 0, v7=True, games_npz=str(s))

    idx = np.arange(8)
    assert off.game_row_for(idx) is None, '不给 sidecar 时应是 None（贴目按 0）'
    row = on.game_row_for(idx)
    assert row is not None
    # 逐行取到的是**各自局**的贴目
    assert np.asarray(row.komi).tolist() == [7.5] * 8


def test_ch5_is_nonzero_only_with_the_sidecar(tmp_path):
    """🔴 全局 ch5 = currentSelfKomi/20：接 sidecar 后应等于 komi/20。"""
    from src.data.feature_v7 import global_features_v7

    d = _board_npz(tmp_path / 'board.npz')
    s = _sidecar(tmp_path / 'games.npz', 4)
    on = load_from_path(str(d), BOARD, 0, v7=True, games_npz=str(s))

    B = 4
    idx = np.arange(B)
    prev = np.zeros((B, BOARD, BOARD), np.int8)
    gl = global_features_v7(
        on.game_row_for(idx),
        np.ones((B, 5), np.int8), np.zeros((B, 5), np.int8),
        prev_board=prev, prev_prev_board=prev, rules_flags=0,
        to_play=np.ones(B, np.int64), board_area=BOARD * BOARD)
    np.testing.assert_allclose(gl[:, 5], 7.5 / 20.0, rtol=1e-6)


def test_default_off_is_bitwise_unchanged(tmp_path):
    """⚠ 不给 `--games-npz` ⇒ 逐位等于接入前的行为（这个旗默认关闭）。"""
    d = _board_npz(tmp_path / 'board.npz')
    ds = load_from_path(str(d), BOARD, 0, v7=True)
    assert ds._game_komi is None
    assert ds.game_row_for(np.arange(4)) is None
    assert _sidecar_kwargs(ds, None) == {}


# --------------------------------------------------------------------------- #
# 口径：按 game_id 索引，不是按行号
# --------------------------------------------------------------------------- #
def test_indexing_is_by_game_id_not_by_row(tmp_path):
    """🔴 同一局的所有行必须拿到**同一个**贴目。

    若误按行号索引，`per_game > 1` 时同一局的不同行会拿到不同贴目 ——
    而 19×19 语料是 162,298 局**首尾相接**的一根大数组，这条错了极隐蔽。
    """
    d = _board_npz(tmp_path / 'board.npz', n_games=4, per_game=4)  # 16 行 / 4 局
    komi = np.array([0.0, 6.5, 5.5, 0.0], np.float16)
    s = _sidecar(tmp_path / 'games.npz', 4, komi=komi)
    ds = load_from_path(str(d), BOARD, 0, v7=True, games_npz=str(s))

    rows = ds.game_row_for(np.arange(16))
    got = np.asarray(rows.komi)
    # 每 4 行一局 ⇒ 逐局常量
    for g in range(4):
        blk = got[g * 4:(g + 1) * 4]
        assert len(set(blk.tolist())) == 1, f'第 {g} 局 4 行贴目不一致：{blk}'
    assert got[0] == 0.0 and got[4] == 6.5 and got[8] == 5.5 and got[12] == 0.0


def test_komi_zero_means_missing_not_real(tmp_path):
    """⚠ `g_komi == 0` 是 sidecar 的**缺失约定**，不是「这局真的贴 0」。

    两种情况在特征里都表现为 ch5=0，本测试钉住的是「别把 0 当成需要特殊处理
    的异常值」—— 也就是不需要过滤、不需要报错。
    """
    d = _board_npz(tmp_path / 'board.npz', n_games=2, per_game=4)
    s = _sidecar(tmp_path / 'games.npz', 2, komi=[0.0, 7.5])
    ds = load_from_path(str(d), BOARD, 0, v7=True, games_npz=str(s))
    got = np.asarray(ds.game_row_for(np.arange(8)).komi)
    assert got[:4].tolist() == [0.0] * 4
    assert got[4:].tolist() == [7.5] * 4


# --------------------------------------------------------------------------- #
# 失败要早、要响
# --------------------------------------------------------------------------- #
def test_short_sidecar_raises_instead_of_silently_zero_filling(tmp_path):
    """🔴 sidecar 短于 game_id 空间 ⇒ 当场报错。

    短了只能靠 `.get(g, 0.0)` 补 0，而 0 会被当成「真贴目 0」写进 ch5/ch18 ——
    那不是缺失标记，是一个**假真值**，且不报任何错。
    """
    d = _board_npz(tmp_path / 'board.npz', n_games=8, per_game=2)  # game_id 0..7
    s = _sidecar(tmp_path / 'games.npz', 3)                       # 只有 0..2
    with pytest.raises(SystemExit) as e:
        load_from_path(str(d), BOARD, 0, v7=True, games_npz=str(s))
    assert 'sidecar' in str(e.value)


def test_missing_key_raises_rather_than_partial_load(tmp_path):
    """🔴 键名对不上 ⇒ 报错，不逐键补齐（拼错键名不报错，只会静默少一项）。"""
    d = _board_npz(tmp_path / 'board.npz')
    bad = tmp_path / 'games.npz'
    np.savez_compressed(bad, g_komi=np.full(4, 7.5, np.float16))  # 只有 1 个键
    with pytest.raises(SystemExit) as e:
        load_from_path(str(d), BOARD, 0, v7=True, games_npz=str(bad))
    assert 'g_rules' in str(e.value) or '缺键' in str(e.value)


def test_longer_sidecar_is_truncated_to_the_id_space(tmp_path):
    """sidecar 比 game_id 空间长（多扫了几局）⇒ 截断，不报错。"""
    d = _board_npz(tmp_path / 'board.npz', n_games=4, per_game=2)  # game_id 0..3
    s = _sidecar(tmp_path / 'games.npz', 10)                      # 多 6 局
    ds = load_from_path(str(d), BOARD, 0, v7=True, games_npz=str(s))
    got = np.asarray(ds.game_row_for(np.arange(8)).komi)
    assert got.tolist() == [7.5] * 8


# --------------------------------------------------------------------------- #
# 诊断
# --------------------------------------------------------------------------- #
def test_diag_counts_are_reported(tmp_path):
    """诊断数字要能回答「有多少局真能用」，而不只是「读到了」。"""
    d = _board_npz(tmp_path / 'board.npz', n_games=6, per_game=2)
    s = _sidecar(tmp_path / 'games.npz', 6, komi=[0, 0, 7.5, 6.5, 5.5, 0])
    ds = load_from_path(str(d), BOARD, 0, v7=True)
    _, _, diag = load_games_sidecar(str(s), ds)
    assert diag['sidecar_games'] == 6
    assert diag['needed_games'] == 6
    assert diag['has_komi_games'] == 3, diag
    assert diag['missing_komi_games'] == 3, diag
    assert diag['has_score_games'] == 3, diag


def test_nondefault_rules_are_surfaced_not_hidden(tmp_path, capsys):
    """⚠ `g_rules` 接不进特征，但**必须报出来**（不能悄悄按简单局算）。

    官方 `calculateArea` 只吃标量 `rules_flags`，逐局化会抛
    "truth value is ambiguous" ⇒ 已知缺口。缺口可以留，隐瞒不行。
    """
    d = _board_npz(tmp_path / 'board.npz', n_games=4, per_game=2)
    s = _sidecar(tmp_path / 'games.npz', 4)
    z = dict(np.load(str(s), allow_pickle=False))
    z['g_rules'] = np.array([0, 1, 1, 0], np.int8)
    np.savez_compressed(str(s), **z)
    ds = load_from_path(str(d), BOARD, 0, v7=True)
    _, _, diag = load_games_sidecar(str(s), ds)
    assert diag['nondefault_rules'] == 2
    _sidecar_kwargs(ds, str(s))
    out = capsys.readouterr().out
    assert 'g_rules' in out and '尚未接进特征' in out


def test_packed_shards_ignore_the_flag(tmp_path):
    """⚠ `--games-npz` 对 C 段（stdata 分片）无意义：贴目已在行里。"""
    packed = tmp_path / 'stdata_v7_s0.npz'
    n = 2
    # packed 分片有必需键契约（见 V7PackedDataset._check_layout），补齐
    np.savez_compressed(
        packed,
        spatial_packed=np.zeros((n, 22, 46), np.uint8),
        moves=np.array([5, 6], np.int64),
        outcome=np.zeros(n, np.int64),
        policy_player_rank=np.zeros((n, 16), np.int64),
        policy_player_prob=np.full((n, 16), 1.0 / 16, np.float32),
        game_weight=np.ones(n, np.float32),
        **{'global': np.zeros((n, 19), np.float32)},
    )
    ds = load_from_path(str(packed), BOARD, 0, v7=True, games_npz=None)
    # packed 分片走 sample_spatial 分支，不经过 _sidecar_kwargs
    assert hasattr(ds, 'sample_spatial')
    assert getattr(ds, '_game_komi', None) is None