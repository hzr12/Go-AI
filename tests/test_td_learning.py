"""TD (C+E) 价值标签单元测试。

覆盖：
  - n-step 边界 ×3：越界回退终局 / 恰好命中 / 奇偶符号（含跳过的无气 pass）
  - α 调度：两端点 + 单调 + 自定义 init/end
  - 回归：--td 0 ≡ 旧 z_soft（tanh 位置软化）逐位一致
  - 范围 / NaN
  - augment8 向量化 ≡ 旧 8×rot90 实现
  - 合成分布检查（不跑 selfplay）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.selfplay_train as sp
from scripts.selfplay_train import compute_td_target, augment8
from src.game.go_rules import GoBoard


def _z_raw_for(score, player):
    if score > 0:
        return 1.0 if player == 1 else -1.0
    if score < 0:
        return -1.0 if player == 1 else 1.0
    return 0.0


def _old_z_soft(z_raw, mc_idx, n_total):
    """旧 _process_game_data 的 z_soft 公式（回归基准）。"""
    alpha = 0.3 + 0.7 * (mc_idx / max(n_total - 1, 1))
    return float(np.tanh(z_raw * alpha))


def test_nstep_out_of_range_fallback():
    """t + td_steps >= n_total → v_td = z_raw（终局值）。"""
    n_total = 4
    players = np.array([1, -1, 1, -1])
    root_values = np.array([0.5, 0.2, -0.3, 0.1])
    for score in (-2, 0, 3):
        for t in range(n_total):
            if t + 3 >= n_total:
                z, z_raw, a = compute_td_target(
                    players, root_values, score, t,
                    td=True, td_steps=3,
                    td_alpha_init=0.2, td_alpha_end=0.9)
                # 越界：v_td 必等于 z_raw（当前 player 视角）
                assert z_raw == _z_raw_for(score, players[t])
                T = max(n_total - 1, 1)
                alpha = 0.2 + (0.9 - 0.2) * (t / T)
                r_soft = _old_z_soft(z_raw, t, n_total)
                expect = float(np.clip(alpha * z_raw + (1 - alpha) * r_soft, -1, 1))
                assert abs(z - expect) < 1e-12, (score, t, z, expect)
    print("PASS n-step 越界回退终局值")


def test_nstep_exact_hit_sign_same_player():
    """t+td_steps 恰好命中；players 相同 → sign=+1。"""
    n_total = 6
    players = np.array([1, -1, 1, -1, 1, -1])
    root_values = np.array([0.9, 0.4, 0.1, -0.2, 0.3, -0.5])
    score = 2
    t, td = 1, 2          # t2 = 3, players[1]=-1, players[3]=-1 同号
    z, z_raw, a = compute_td_target(players, root_values, score, t,
                                    td=True, td_steps=td,
                                    td_alpha_init=0.2, td_alpha_end=0.9)
    T = max(n_total - 1, 1)
    alpha = 0.2 + 0.7 * (t / T)
    r_soft = _old_z_soft(_z_raw_for(score, players[t]), t, n_total)
    expect = float(np.clip(alpha * root_values[3] + (1 - alpha) * r_soft, -1, 1))
    assert abs(z - expect) < 1e-12, (z, expect)
    print("PASS n-step 恰好命中（同号 +1）")


def test_nstep_sign_flip_and_skipped_pass():
    """players 异号 → sign=-1；含跳过的无气 pass（连续同 player）。"""
    n_total = 5
    # 正常交替 + 最后一位模拟"对手无气 pass 被跳过"：players[3] 与 players[4] 同色
    players = np.array([1, -1, 1, -1, -1])
    root_values = np.array([0.8, -0.4, 0.2, 0.6, -0.7])
    score = 1

    # t=0, td=1 → t2=1: players[0]=1, players[1]=-1 异号 → v_td = -root_values[1]
    z, _, _ = compute_td_target(players, root_values, score, 0,
                                td=True, td_steps=1,
                                td_alpha_init=0.2, td_alpha_end=0.9)
    T = max(n_total - 1, 1)
    alpha = 0.2 + 0.7 * (0 / T)
    r_soft = _old_z_soft(_z_raw_for(score, 1), 0, n_total)
    expect = float(np.clip(alpha * (-root_values[1]) + (1 - alpha) * r_soft, -1, 1))
    assert abs(z - expect) < 1e-12, (z, expect)

    # t=2, td=1 → t2=3: players[2]=1, players[3]=-1 异号
    z, _, _ = compute_td_target(players, root_values, score, 2,
                                td=True, td_steps=1,
                                td_alpha_init=0.2, td_alpha_end=0.9)
    alpha = 0.2 + 0.7 * (2 / T)
    r_soft = _old_z_soft(_z_raw_for(score, 1), 2, n_total)
    expect = float(np.clip(alpha * (-root_values[3]) + (1 - alpha) * r_soft, -1, 1))
    assert abs(z - expect) < 1e-12, (z, expect)

    # t=3, td=1 → t2=4: players[3]=-1, players[4]=-1 同号（跳过的 pass 情形）
    z, _, _ = compute_td_target(players, root_values, score, 3,
                                td=True, td_steps=1,
                                td_alpha_init=0.2, td_alpha_end=0.9)
    alpha = 0.2 + 0.7 * (3 / T)
    r_soft = _old_z_soft(_z_raw_for(score, -1), 3, n_total)
    expect = float(np.clip(alpha * root_values[4] + (1 - alpha) * r_soft, -1, 1))
    assert abs(z - expect) < 1e-12, (z, expect)
    print("PASS n-step 奇偶符号（含跳过的无气 pass）")


def test_alpha_schedule_endpoints_and_monotonic():
    """α(t): t=0→init, t=T→end；单调不减（init<end）。"""
    n_total = 10
    players = np.array([1 if i % 2 == 0 else -1 for i in range(n_total)])
    root_values = np.zeros(n_total)

    _, _, a0 = compute_td_target(players, root_values, 0, 0,
                                 True, 3, 0.2, 0.9)
    _, _, aL = compute_td_target(players, root_values, 0, n_total - 1,
                                 True, 3, 0.2, 0.9)
    assert abs(a0 - 0.2) < 1e-12
    assert abs(aL - 0.9) < 1e-12

    prev = -2.0
    for t in range(n_total):
        _, _, a = compute_td_target(players, root_values, 0, t,
                                    True, 3, 0.2, 0.9)
        assert a >= prev, (t, a, prev)
        prev = a
    print("PASS α 调度两端点 + 单调")


def test_alpha_schedule_custom_init_end():
    n_total = 5
    players = np.array([1, -1, 1, -1, 1])
    root_values = np.zeros(n_total)
    _, _, a0 = compute_td_target(players, root_values, 0, 0,
                                 True, 3, 0.1, 0.8)
    _, _, aL = compute_td_target(players, root_values, 0, 4,
                                 True, 3, 0.1, 0.8)
    assert abs(a0 - 0.1) < 1e-12
    assert abs(aL - 0.8) < 1e-12
    print("PASS α 调度自定义 init/end")


def test_td0_bitwise_regression():
    """--td 0 必须与旧 z_soft 逐位一致（对全部 score / 位置）。"""
    rng = np.random.default_rng(0)
    n_total = 50
    players = np.array([1 if i % 2 == 0 else -1 for i in range(n_total)])
    root_values = rng.uniform(-1, 1, size=n_total)
    for score in (-2, 0, 3):
        for t in range(n_total):
            z, z_raw, a = compute_td_target(players, root_values, score, t,
                                            td=False, td_steps=3,
                                            td_alpha_init=0.2,
                                            td_alpha_end=0.9)
            expect = _old_z_soft(_z_raw_for(score, players[t]), t, n_total)
            assert z == expect, f"score={score} t={t} z={z} expect={expect}"
            assert a == 0.0
    print("PASS td=0 逐位回归（≡ 旧 tanh 软化）")


def test_z_range_and_no_nan():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n_total = int(rng.integers(5, 60))
        players = np.array([1 if i % 2 == 0 else -1 for i in range(n_total)])
        # 随机把某些位置设成与前同色（模拟无气 pass 被跳过）
        for i in range(1, n_total):
            if rng.random() < 0.2:
                players[i] = players[i - 1]
        root_values = rng.uniform(-1, 1, size=n_total)
        score = int(rng.integers(-2, 3))
        t = int(rng.integers(0, n_total))
        z, z_raw, a = compute_td_target(players, root_values, score, t,
                                        True, 3, 0.2, 0.9)
        assert -1.0 <= z <= 1.0, z
        assert -1.0 <= z_raw <= 1.0
        assert np.isfinite(z) and np.isfinite(z_raw) and np.isfinite(a)
        assert 0.0 <= a <= 1.0
    print("PASS z 范围/无 NaN（50 随机局）")


def test_td_distribution_synthetic():
    """合成分布检查（不跑 selfplay）：网络越接近终局越准 → z 的
    均值/标准差/极值应在合理范围，且开局→残局 |z| 不塌缩。"""
    rng = np.random.default_rng(7)
    n_total = 200
    players = np.array([1 if i % 2 == 0 else -1 for i in range(n_total)])
    score = 1  # 黑胜
    z_raw_arr = np.array([_z_raw_for(score, p) for p in players])
    prog = np.linspace(0, 1, n_total)
    # 合成 root_values：随位置逐渐逼近 z_raw（模拟网络变准）+ 噪声
    root_values = np.clip(z_raw_arr * prog + rng.normal(0, 0.2, size=n_total),
                         -1, 1)
    zs = np.array([compute_td_target(players, root_values, score, t,
                                     True, 3, 0.2, 0.9)[0]
                   for t in range(n_total)])
    assert zs.shape == (n_total,)
    assert np.all(np.isfinite(zs))
    assert np.all(np.abs(zs) <= 1.0 + 1e-9)
    assert zs.std() > 0.0, "分布塌缩（std=0），TD 未生效"
    first = zs[:n_total // 4].mean()
    last = zs[-(n_total // 4):].mean()
    print(f"PASS 合成分布检查 mean={zs.mean():.3f} std={zs.std():.3f} "
          f"range=[{zs.min():.3f},{zs.max():.3f}] "
          f"first_quartile_mean={first:.3f} last_quartile_mean={last:.3f}")


def _legacy_augment8(plane, target, n):
    """旧 8×rot90 实现（等价基准）。"""
    board_t = target[:n * n].reshape(n, n)
    pass_t = target[n * n]
    out = []
    for t in range(8):
        k = t % 4
        pl = plane.copy()
        tb = board_t.copy()
        if t >= 4:
            pl = pl[:, :, ::-1]
            tb = tb[:, ::-1]
        if k > 0:
            pl = np.rot90(pl, k=-k, axes=(1, 2))
            tb = np.rot90(tb, k=-k)
        tv = np.concatenate([tb.reshape(-1), [pass_t]])
        out.append((np.ascontiguousarray(pl), np.ascontiguousarray(tv)))
    return out


def test_augment8_matches_legacy():
    """C5 全向量化 augment8 必须与旧 8×rot90 实现逐位一致。"""
    n = 9
    rng = np.random.default_rng(0)
    for trial in range(50):
        plane = rng.normal(size=(12, n, n)).astype(np.float32)
        target = rng.random(size=(n * n + 1,)).astype(np.float64)
        legacy = _legacy_augment8(plane, target, n)
        new = augment8(plane, target, n)
        assert len(new) == 8
        for i in range(8):
            np.testing.assert_array_equal(
                new[i][0], legacy[i][0], err_msg=f"plane t={i} trial={trial}")
            np.testing.assert_array_equal(
                new[i][1], legacy[i][1], err_msg=f"target t={i} trial={trial}")
    print("PASS augment8 向量化 ≡ 旧实现（50 随机样本 × 8 变换）")


def _process_game_data_regression():
    """_process_game_data 在 td=0 时与旧公式一致；td=1 时接入 TD。"""
    import argparse
    n = 5
    bs = 9
    n_actions = bs * bs + 1
    rng = np.random.default_rng(3)
    game_data = []
    for i in range(n):
        planes = rng.normal(size=(12, bs, bs)).astype(np.float32)
        vt = rng.random(size=n_actions).astype(np.float64)
        vt /= vt.sum()
        player = 1 if i % 2 == 0 else -1
        game_data.append((planes, vt, player, i, float(rng.uniform(-1, 1))))

    args0 = argparse.Namespace(td=0, td_steps=3, td_alpha_init=0.2,
                               td_alpha_end=0.9, no_augment=1)
    buf0 = []
    sp._process_game_data(game_data, score=1, bs=bs, n_actions=n_actions,
                          buffer=buf0, args=args0)
    assert len(buf0) == n
    for i, row in enumerate(game_data):
        z_expect = _old_z_soft(_z_raw_for(1, row[2]), i, n)
        assert buf0[i][2] == z_expect, f"td=0 回归 t={i}"

    args1 = argparse.Namespace(td=1, td_steps=3, td_alpha_init=0.2,
                               td_alpha_end=0.9, no_augment=1)
    buf1 = []
    sp._process_game_data(game_data, score=1, bs=bs, n_actions=n_actions,
                          buffer=buf1, args=args1)
    assert len(buf1) == n
    # td=1 时 z 与 root_values 相关（不再恒等于 r_soft）
    players = np.asarray([row[2] for row in game_data])
    rvs = np.asarray([row[4] for row in game_data])
    for i in range(n):
        z_expect, _, _ = compute_td_target(players, rvs, 1, i, True, 3, 0.2, 0.9)
        assert abs(buf1[i][2] - z_expect) < 1e-12
    # 与 td=0 结果应有差异（TD 生效）
    assert any(abs(buf0[i][2] - buf1[i][2]) > 1e-9 for i in range(n))
    print("PASS _process_game_data td=0 回归 + td=1 接入")


def test_apply_symmetry_batch_consistency():
    """apply_symmetry_batch 与单样本 apply_symmetry 逐位一致（兜底 C5 依赖）。"""
    n = 9
    rng = np.random.default_rng(11)
    B = 24
    states = rng.normal(size=(B, 12, n, n)).astype(np.float32)
    moves = rng.integers(-1, n * n, size=B)
    tids = rng.integers(0, 8, size=B)
    out, out_mv = GoBoard.apply_symmetry_batch(states, moves, tids, n)
    for i in range(B):
        ref_pl, ref_mv = GoBoard.apply_symmetry(
            states[i], int(moves[i]), int(tids[i]), n)
        np.testing.assert_array_equal(out[i], ref_pl, err_msg=f"plane i={i}")
        assert int(out_mv[i]) == int(ref_mv), (i, out_mv[i], ref_mv)
    print("PASS apply_symmetry_batch ≡ apply_symmetry（24 样本 × 随机变换）")


if __name__ == "__main__":
    test_nstep_out_of_range_fallback()
    test_nstep_exact_hit_sign_same_player()
    test_nstep_sign_flip_and_skipped_pass()
    test_alpha_schedule_endpoints_and_monotonic()
    test_alpha_schedule_custom_init_end()
    test_td0_bitwise_regression()
    test_z_range_and_no_nan()
    test_td_distribution_synthetic()
    test_augment8_matches_legacy()
    _process_game_data_regression()
    test_apply_symmetry_batch_consistency()
    print("\nAll TD learning tests passed.")
