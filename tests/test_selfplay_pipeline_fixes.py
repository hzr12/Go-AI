"""异步自对弈流水线的四个已定位缺陷。

现场（4 卡 NPU，`--async-pipeline 1 --parallel-games 8`）：启动后
`[async] 启动 8 个自对弈 Worker` 完全静默，151 秒内
`已产/已取=0/0 存活Worker=8/8`，CPU 0%。活着却不产出且不吃 CPU
= 阻塞，不是计算。

四个根因：

1. **fork-after-CANN 死锁**（挂死主因）。`AsyncSelfPlayPipeline.start()`
   用裸 `worker.start()`，Linux 下即 `fork`；而父进程此前已在 NPU 上建
   模型并 warmup（起 CANN 线程），子进程继承被锁死的上下文。同步路径
   早就修过这个问题（`selfplay_train.py` 内注释标记 "N2"，改用
   `mp.get_context('spawn')`），**异步路径漏修**。
2. **跨叶子批处理是死代码**。`MCTSNode.expanded` 被同时用作两个互斥语义：
   worker 线程置 `True` 表示"已被认领"用于去重，主线程又用
   `not leaf.expanded` 表示"尚未展开"以触发批量前向。后者恒为假，
   批量评估整段不可达，每个叶子退化成一次 B=1 前向。
3. **rollout 步数超支约 12 倍**。两条 worker 路径都漏传
   `rollout_steps`，落到默认 `None` → `2*N*N`（19 路为 722），
   而 CLI `--rollout-steps` 默认 60。
4. `args.batch_cap` 已定义但 MCTS 硬编码 `max(threads*4, 32)`，从不读取。
"""

import argparse
import multiprocessing
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.async_pipeline import AsyncSelfPlayPipeline, SelfPlayWorker
from src.game.go_rules import GoBoard
from src.search.mcts import MCTS


def make_args(**over):
    d = dict(
        parallel_games=2, result_queue_max=10, board_size=5, sims=16,
        max_moves=6, temperature=1.0, expand_topk=4, expand_chunk=0,
        mcts_threads=1, spec_prefetch=False, use_rollout=False,
        rollout_lambda=0.25, rollout_steps=60, rollout_steps_cli=60,
        leaf_ab_depth=0, c_puct=2.0, virtual_loss=8.0,
        dynamic_topk=True, dynamic_virtual_loss=True, mcts_vector_backup=1,
        dir_alpha=0.3, dir_eps=0.25, batch_cap=64, device='cpu',
        onnx_model=None,
    )
    d.update(over)
    return argparse.Namespace(**d)


class RecordingAI:
    """只实现 MCTS 真正使用的接口，并记录每次前向的批大小。"""

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.batch_sizes = []

    def predict_batch(self, states):
        b = len(states)
        self.batch_sizes.append(b)
        # 均匀策略：保证 expand_topk 能选出 k 个互异候选，使树分叉。
        # 若把概率全压在单一着法上，会被 policy_pruning_thresh 剪成单链，
        # 树退化、根本无法观测到批处理。
        pol = np.full((b, self.n_actions), 1.0 / self.n_actions,
                      dtype=np.float32)
        return pol, np.zeros((b,), dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. fork-after-CANN 死锁
# --------------------------------------------------------------------------- #

def test_worker_processes_are_created_from_spawn_context():
    """Linux 下裸 start() 即 fork，会复制父进程已初始化的 NPU/CUDA 上下文
    导致子进程挂死——现场表现就是 8/8 存活但永远 0 产出、CPU 0%。"""
    pipeline = AsyncSelfPlayPipeline(args=make_args(), model_path='m.pth')
    worker = pipeline._make_worker(0)
    assert worker._start_method == 'spawn', \
        '异步 worker 必须用 spawn context，实际为 {}'.format(
            worker._start_method)
    assert isinstance(worker, multiprocessing.context.SpawnProcess)


def test_worker_is_daemon_so_stays_shutdownable():
    """daemon=True 保证主进程退出时 worker 不会挂住训练。"""
    pipeline = AsyncSelfPlayPipeline(args=make_args(), model_path='m.pth')
    assert pipeline._make_worker(0).daemon is True


def test_progress_callback_is_importable_and_picklable():
    """spawn 会 pickle worker 实例，局部闭包无法序列化，会在启动时炸掉。"""
    import scripts.selfplay_train as st
    cb = st._on_worker_game
    assert pickle.loads(pickle.dumps(cb)) is cb


def test_callback_accepts_worker_invocation_signature():
    """回调签名须与 worker 侧调用点一致。"""
    import scripts.selfplay_train as st
    assert st._on_worker_game(0, [(0, 0, 0)], 1.5) is None


# --------------------------------------------------------------------------- #
# 2. 跨叶子批处理（expanded 语义冲突）
# --------------------------------------------------------------------------- #

def _run_search(num_threads=8, sims=120, board_size=7, expand_topk=4):
    ai = RecordingAI(n_actions=board_size * board_size + 1)
    mcts = MCTS(ai, board_size=board_size, num_threads=num_threads,
                expand_topk=expand_topk, spec_prefetch=False,
                leaf_ab_depth=0, dynamic_topk=False, temperature=0.0)
    board = GoBoard(board_size)
    hist = [-1, -1, -3]
    mcts.search(board, list(hist), list(hist), 1, simulations=sims)
    return ai


def test_leaf_evaluation_is_batched_across_simulations():
    """叶子前向必须跨模拟合并成一次调用。

    固定 expand_topk=4 时，子节点评估的批恒 <= 4（play() 还可能再拒绝
    几个），叶子自身的评估是 B=1。因此出现 > 4 的批只可能来自跨叶子
    合并——这正是原先不可达的那段代码。
    """
    ai = _run_search()
    assert max(ai.batch_sizes) > 4, (
        '未观测到跨叶子批处理：最大批={}（子节点批上限=4）。'
        'expanded 标志的"已认领/已展开"语义冲突使批量前向不可达，'
        '每个叶子退化为 B=1。'.format(max(ai.batch_sizes)))


def test_claimed_leaf_is_not_yet_expanded():
    """claimed（已认领）与 expanded（已展开）必须是两个独立状态。

    worker 认领叶子时只应置 claimed；若直接置 expanded，主线程就再也
    筛不出待批量评估的叶子。
    """
    from src.search.mcts import MCTSNode
    node = MCTSNode(board=None, my_hist=[], op_hist=[], to_play=1,
                    move_int=-1)
    assert node.claimed is False
    assert node.expanded is False


# --------------------------------------------------------------------------- #
# 3. rollout_steps 传参
# --------------------------------------------------------------------------- #

class _FakeMCTS:
    captured = None

    def __init__(self, ai, board_size=None, **kw):
        type(self).captured = dict(kw)
        self._n = board_size * board_size + 1

    def search(self, board, mh, oh, to_play, simulations=0,
               path_moves=None, **kw):
        n = board.board.shape[0] ** 2 + 1
        return np.ones(n), np.full(n, 1.0 / n), 0.0


@pytest.fixture
def fake_mcts(monkeypatch):
    import src.search.mcts as mod
    _FakeMCTS.captured = None
    monkeypatch.setattr(mod, 'MCTS', _FakeMCTS)
    yield
    assert mod.MCTS is _FakeMCTS


def test_async_worker_mcts_receives_rollout_steps(fake_mcts):
    """漏传会落到 2*N*N（19 路 722 步），比 CLI 默认 60 超支约 12 倍。"""
    args = make_args(rollout_steps=60, use_rollout=True)
    pipeline = AsyncSelfPlayPipeline(args=args, model_path='m.pth')
    worker = SelfPlayWorker(0, 'm.pth', None, args,
                            pipeline.data_queue, pipeline.stop_event)
    worker._play_one_game(RecordingAI(n_actions=26), 26, 6)
    assert _FakeMCTS.captured['rollout_steps'] == 60


def test_sync_worker_passes_rollout_steps(monkeypatch):
    """同步 worker 同样漏传（只有串行路径传了）。"""
    import scripts.selfplay_train as st
    captured = {}

    class _FakeAI:
        def __init__(self, *a, **k):
            pass

    def _fake_self_play_game(*a, **kw):
        captured.update(kw)
        return [], 0.0

    monkeypatch.setattr(st, 'GoAI', _FakeAI)
    monkeypatch.setattr(st, 'self_play_game', _fake_self_play_game)

    class _FakeQueue:
        def put(self, item):
            pass

    st._selfplay_worker(0, 'm.pth', make_args(rollout_steps=60), _FakeQueue())
    assert captured['rollout_steps'] == 60


# --------------------------------------------------------------------------- #
# 4. batch_cap 可配置
# --------------------------------------------------------------------------- #

def test_batch_cap_default_unchanged():
    """默认必须保持 max(threads*4, 32)，否则会重开既有的批量回归。"""
    m = MCTS(RecordingAI(26), board_size=5, num_threads=3)
    assert m.batch_cap == 32
    m8 = MCTS(RecordingAI(26), board_size=5, num_threads=8)
    assert m8.batch_cap == 32


def test_batch_cap_is_configurable():
    m = MCTS(RecordingAI(26), board_size=5, num_threads=3, batch_cap=64)
    assert m.batch_cap == 64


# --------------------------------------------------------------------------- #
# 5. worker 失败必须可见（原先静默吞异常 / 加载失败直接暴死）
# --------------------------------------------------------------------------- #

class _LoadBoomWorker(SelfPlayWorker):
    def _build_ai(self):
        raise RuntimeError('模型加载失败-模拟')


class _GameBoomWorker(SelfPlayWorker):
    def _build_ai(self):
        return object()

    def _play_one_game(self, ai, n_actions, max_moves):
        # 先置 stop_event，让 run() 的下一轮 while 正常退出，避免测试挂死
        self.stop_event.set()
        raise RuntimeError('对局失败-模拟')


def _errors(pipeline):
    return pipeline.data_queue.stats['errors'].value


def _require_build_ai_seam():
    """模型构建必须是可注入的接缝。

    否则 run() 会真的去加载一个完整网络（实测会跑起 conv2d、几十秒不返回），
    既测不到失败路径，也会把 RED 变成超时而不是干净的断言失败。
    """
    assert hasattr(SelfPlayWorker, '_build_ai'), (
        'SelfPlayWorker 必须把模型构建抽成 _build_ai()，'
        '否则无法在不加载真实模型的前提下测试 worker 失败路径')


def test_model_load_failure_is_reported_not_silent(capsys):
    """模型加载在 try 之外，抛异常会直接杀掉子进程 → 父进程只见 0/8 存活，
    却拿不到任何原因。必须上报。"""
    _require_build_ai_seam()
    args = make_args()
    pipeline = AsyncSelfPlayPipeline(args=args, model_path='m.pth')
    worker = _LoadBoomWorker(0, 'm.pth', None, args,
                             pipeline.data_queue, pipeline.stop_event)
    worker.run()  # 不得抛出
    assert _errors(pipeline) == 1, '模型加载失败必须计入共享错误计数'
    assert '模型加载失败-模拟' in capsys.readouterr().err


def test_game_failure_is_reported_and_worker_keeps_going(capsys):
    """单局异常原先被 `except Exception: time.sleep(0.1)` 静默吞掉，
    worker 永远活着却零产出。必须上报。"""
    _require_build_ai_seam()
    args = make_args()
    pipeline = AsyncSelfPlayPipeline(args=args, model_path='m.pth')
    worker = _GameBoomWorker(0, 'm.pth', None, args,
                             pipeline.data_queue, pipeline.stop_event)
    worker.run()
    assert _errors(pipeline) == 1, '对局失败必须计入共享错误计数'
    assert '对局失败-模拟' in capsys.readouterr().err


def test_successful_game_does_not_count_as_error(capsys):
    """正常路径不能被误报为错误。"""
    class _OkWorker(SelfPlayWorker):
        def _build_ai(self):
            return object()

        def _play_one_game(self, ai, n_actions, max_moves):
            self.stop_event.set()
            return [('p', 'v', 1, 0, 0.0)], 1.0

    _require_build_ai_seam()
    args = make_args()
    pipeline = AsyncSelfPlayPipeline(args=args, model_path='m.pth')
    worker = _OkWorker(0, 'm.pth', None, args,
                       pipeline.data_queue, pipeline.stop_event)
    worker.run()
    assert _errors(pipeline) == 0
    assert capsys.readouterr().err == ''


def test_parent_error_message_points_at_worker_stderr():
    """父进程存活检查的提示必须指向真实死因渠道。

    旧文案归因于「fork 复制 CANN 上下文」，而 worker 早已改用 spawn，
    该归因已过时，会把排查引向错误方向。
    """
    import scripts.selfplay_train as st
    src = open(os.path.join(ROOT, 'scripts', 'selfplay_train.py'),
               encoding='utf-8').read()
    assert 'stderr' in src, '提示应指向 worker stderr'
    for stale in ('子进程继承了不可用的 CANN 状态',
                  '主进程已初始化 NPU/CANN 后再 fork'):
        assert stale not in src, (
            '存活检查的文案仍在归因于 fork/CANN（{}），但 worker 已改用 '
            'spawn，该归因会把排查引向错误方向'.format(stale))


# --------------------------------------------------------------------------- #
# 6. probs 零保护：vis.sum()==0 时 0/0 = NaN
# --------------------------------------------------------------------------- #

def test_probs_handles_zero_visits_without_nan():
    """温度 > 0 分支的 `vis / vis.sum()` 在全零 visits 时产出 NaN。

    调用方 self_play_game 的 `s <= 0` 兜底救不了：NaN 与任何值比较都是
    False，判断会直接落进 np.random.choice(p=NaN) → ValueError。
    同一函数的 temp<=0 分支已有 `if visits.sum() > 0` 保护，缺保护的是
    这一侧，属于明确的不对称。
    """
    from src.search.mcts import MCTS as _M
    assert hasattr(_M, '_probs_from_visits'), (
        '概率换算应抽成 _probs_from_visits，才能对全零 visits 直接断言')
    m = _M(RecordingAI(26), board_size=5, num_threads=1, temperature=1.0)
    probs = m._probs_from_visits(np.zeros(26, dtype=np.int64))
    assert not np.isnan(probs).any(), '全零 visits 不得产出 NaN'
    assert np.isfinite(probs).all()
    assert probs.sum() > 0, '必须给出可采样的分布'
    assert np.isclose(probs.sum(), 1.0), '分布必须归一'


def test_probs_zero_guard_also_covers_zero_temperature():
    m = MCTS(RecordingAI(26), board_size=5, num_threads=1, temperature=0.0)
    probs = m._probs_from_visits(np.zeros(26, dtype=np.int64))
    assert not np.isnan(probs).any()
    assert probs.max() == 1.0


def test_probs_normal_case_unchanged():
    """正常 visits 的换算结果不能被零保护改动。"""
    v = np.zeros(26, dtype=np.int64)
    v[3] = 7
    v[9] = 5
    m_t1 = MCTS(RecordingAI(26), board_size=5, num_threads=1, temperature=1.0)
    p = m_t1._probs_from_visits(v)
    assert np.isclose(p[3], 7 / 12) and np.isclose(p[9], 5 / 12)
    m_t0 = MCTS(RecordingAI(26), board_size=5, num_threads=1, temperature=0.0)
    p0 = m_t0._probs_from_visits(v)
    assert p0[3] == 1.0 and p0[9] == 0.0


# --------------------------------------------------------------------------- #
# 7. P0：同步原语必须与 worker 同 context；死因必须可判
# --------------------------------------------------------------------------- #

def _spawn_putter(data_queue, stop_event):
    """模块级 spawn 子进程入口，仅用于验证跨进程原语真的可用。"""
    data_queue.put({'from_child': True})


def test_shared_primitives_survive_spawn_boundary():
    """队列/事件必须能被 spawn 出的子进程真正使用。

    异步流水线原先用模块级 Queue 与 mp.Event()（绑定**默认** context，
    Linux 上即 fork），却把对象交给 spawn context 的进程。这是真正被打破的
    行为：子进程无声死掉，既无 traceback、错误计数也为 0。这里直接跑一次
    真实 spawn 往返来守住它。
    """
    pipeline = AsyncSelfPlayPipeline(args=make_args(), model_path='m.pth')
    assert pipeline.ctx.get_start_method() == 'spawn'
    assert pipeline.data_queue.ctx is pipeline.ctx, \
        '队列必须由 worker 使用的同一个 context 创建'
    assert pipeline._make_worker(0)._start_method == \
        pipeline.ctx.get_start_method(), 'worker 与原语必须同 context'

    proc = mp.get_context('spawn').Process(
        target=_spawn_putter,
        args=(pipeline.data_queue, pipeline.stop_event), daemon=True)
    proc.start()
    proc.join(timeout=180)
    assert proc.exitcode == 0, \
        'spawn 子进程未能通过共享队列回传，exitcode={}'.format(proc.exitcode)
    assert pipeline.data_queue.get(timeout=30) == {'from_child': True}


def test_exitcode_describer_names_the_signal():
    """exitcode 为负 = 被信号杀死。必须能区分 SIGSEGV/SIGKILL 等，
    否则「worker 全死」无从排查。"""
    from scripts.async_pipeline import _describe_exitcode
    assert '仍在运行' in _describe_exitcode(None)
    assert 'SIGSEGV' in _describe_exitcode(-11)
    assert 'SIGKILL' in _describe_exitcode(-9)
    assert 'SIGABRT' in _describe_exitcode(-6)
    assert '正常退出' in _describe_exitcode(0)
    assert '3' in _describe_exitcode(3)


def test_child_entry_enables_faulthandler(monkeypatch):
    """原生崩溃（SIGSEGV/SIGABRT）不会产生 Python traceback，
    faulthandler 才能在子进程里把栈打出来。"""
    import scripts.async_pipeline as ap

    calls = []

    class _StubWorker:
        def __init__(self, *a, **k):
            pass

        def run(self):
            calls.append('run')

    monkeypatch.setattr(ap.faulthandler, 'enable',
                        lambda *a, **k: calls.append('faulthandler'))
    monkeypatch.setattr(ap, 'SelfPlayWorker', _StubWorker)
    ap._async_selfplay_worker(0, 'm', None, make_args(), None, None, None)
    assert 'faulthandler' in calls, '子进程入口必须开启 faulthandler'
    assert 'run' in calls, '开启 faulthandler 后仍应照常执行 worker 主循环'


# --------------------------------------------------------------------------- #
# 8. P1a：_fast_atari_mask 向量化（实测占 rollout 阶段 52.8%）
# --------------------------------------------------------------------------- #

def _reference_fast_atari_mask(board, player):
    """优化前的逐子 Python BFS 实现，作为等价性 oracle。

    保留在测试里而不是生产代码：它是判断"向量化有没有改变语义"的唯一依据。
    与 src/game/go_rules.py:549-577 给特征平面通道 10/11 的向量化算法等价——
    那段代码算的正是"气数恰为 1 的连通块"。
    """
    n = board.board_size
    b = board.board
    occ = (b == player)
    atari = np.zeros((n, n), dtype=bool)
    visited = np.zeros((n, n), dtype=bool)
    seeds = np.argwhere(occ)
    if seeds.size == 0:
        return atari.reshape(-1)
    nb = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for (y, x) in seeds:
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        group = []
        libs = set()
        while stack:
            cy, cx = stack.pop()
            group.append((cy, cx))
            for dy, dx in nb:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < n and 0 <= nx < n:
                    v = b[ny, nx]
                    if v == 0:
                        libs.add((ny, nx))
                    elif v == player and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(libs) == 1:
            for (gy, gx) in group:
                atari[gy, gx] = True
    return atari.reshape(-1)


def _random_positions(board_size, count, seed, fill=0.45):
    """造若干合法中盘局面。

    fill 是落子数占交叉点数的比例。默认取 0.45（中盘密度）而不是稀疏局面：
    旧实现的开销正比于棋子数，新实现正比于 n² 且常数极小，因此稀疏盘面会
    让 numpy 的每次调用固定开销占主导，严重低估真实加速比。
    """
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(count):
        b = GoBoard(board_size)
        for _ in range(int(board_size * board_size * fill)):
            legal = np.where(b.get_legal_moves())[0]
            if legal.size == 0:
                break
            mv = int(rng.choice(legal))
            if not b.play(mv):
                break
        out.append((b, b.current_player))
    return out


def test_fast_atari_mask_matches_reference():
    """向量化不得改变语义：逐元素一致。跨 9 路与 19 路两种尺度。"""
    from src.search.light_rollout import _fast_atari_mask
    positions = (_random_positions(9, 8, seed=1234)
                 + _random_positions(19, 4, seed=4321))
    assert positions, '随机局面构造失败，测试无意义'
    checked = 0
    for board, player in positions:
        got = _fast_atari_mask(board, player)
        want = _reference_fast_atari_mask(board, player)
        assert got.dtype == want.dtype, 'dtype 变了'
        assert got.shape == want.shape, '形状变了'
        assert np.array_equal(got, want), (
            'atari 掩码与原实现不一致：差 {} 个点'.format(
                int(np.count_nonzero(got != want))))
        checked += 1
    assert checked >= 10, '样本过少（{}），不足以覆盖多种块形'.format(checked)


def test_fast_atari_mask_counts_shared_liberty_once():
    """一个空点邻接同块多颗子时，只能算 1 气。

    这是去重语义的核心，也是**不能**照抄 go_rules.feature_planes_batched 的
    `bincount(weights=neigh_empty)` 求和写法的原因：那种写法会把共享气重复
    计入。下面这组黑子 (1,0)/(0,1)/(1,1) 中，空点 (0,0) 同时邻接前两颗子。
    按去重：全组只有 (0,0) 一口气 → 是打吃；按求和：1+1+0=2 气 → 不是打吃。
    """
    from src.search.light_rollout import _fast_atari_mask
    b = GoBoard(9)
    b.board[1, 0] = 1
    b.board[0, 1] = 1
    b.board[1, 1] = 1
    b.board[2, 0] = -1
    b.board[0, 2] = -1
    b.board[2, 1] = -1
    b.board[1, 2] = -1
    m = _fast_atari_mask(b, 1).reshape(9, 9)
    assert m[1, 0] and m[0, 1] and m[1, 1], \
        '共享气必须去重：三颗子同组、仅 (0,0) 一口气，应整体判为打吃'
    # oracle 必须给出同样结论，确保该用例真的能区分两种语义
    assert np.array_equal(m.reshape(-1), _reference_fast_atari_mask(b, 1)), \
        '该构造用例未能区分去重与求和两种语义，测试失去意义'


def test_fast_atari_mask_handles_empty_and_full_board():
    from src.search.light_rollout import _fast_atari_mask
    empty = GoBoard(9)
    assert not _fast_atari_mask(empty, 1).any()
    assert _fast_atari_mask(empty, 1).shape == (81,)


def test_fast_atari_mask_is_faster_than_reference():
    """这是本项优化本身的判据：必须真的更快，而不只是等价。

    旧实现对每个己方棋子跑一次 Python BFS，实测占整个搜索 52.8%。
    阈值取 2× 以避免在慢机器上抖动（实测预期远高于此）。
    """
    import time
    from src.search.light_rollout import _fast_atari_mask
    positions = _random_positions(9, 8, seed=99)
    assert positions

    def _bench(fn, reps=3):
        best = float('inf')
        for _ in range(reps):
            t0 = time.perf_counter()
            for board, player in positions:
                fn(board, player)
            best = min(best, time.perf_counter() - t0)
        return best

    t_new = _bench(_fast_atari_mask)
    t_old = _bench(_reference_fast_atari_mask)
    assert t_new < t_old / 2.0, (
        '向量化未带来足够提速：新 {:.4f}s vs 旧 {:.4f}s（仅 {:.2f}×）'.format(
            t_new, t_old, t_old / max(t_new, 1e-9)))


# --------------------------------------------------------------------------- #
# 9. P1a：rollout 采样改逆 CDF（rng.choice 实测占 8.2%）
# --------------------------------------------------------------------------- #

def test_sample_categorical_matches_target_distribution():
    """逆 CDF 采样必须与 rng.choice 抽自同一个离散分布。"""
    from src.search.light_rollout import _sample_categorical
    p = np.array([0.05, 0.10, 0.30, 0.40, 0.15])
    rng = np.random.default_rng(20260925)
    draws = np.array([_sample_categorical(p, rng) for _ in range(60000)])
    assert draws.min() >= 0 and draws.max() < p.size
    freq = np.bincount(draws, minlength=p.size) / draws.size
    assert np.all(np.abs(freq - p) < 0.01), \
        '采样分布偏离目标分布: got={} want={}'.format(
            np.round(freq, 4).tolist(), p.tolist())


def test_sample_categorical_is_faster_than_rng_choice():
    import time
    from src.search.light_rollout import _sample_categorical
    p = np.full(82, 1.0 / 82)
    rng = np.random.default_rng(7)
    reps = 20000

    t0 = time.perf_counter()
    for _ in range(reps):
        _sample_categorical(p, rng)
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(reps):
        rng.choice(82, p=p)
    t_old = time.perf_counter() - t0

    assert t_new < t_old, (
        '逆 CDF 采样反而更慢：新 {:.4f}s vs rng.choice {:.4f}s'.format(
            t_new, t_old))


def test_sample_move_only_returns_legal_range():
    """采样下标必须落在 [0, n*n]（含 pass）。"""
    from src.search.light_rollout import FastPolicy
    b = GoBoard(9)
    for _ in range(8):
        legal = np.where(b.get_legal_moves())[0]
        if legal.size:
            b.play(int(legal[0]))
    policy = FastPolicy(9)
    rng = np.random.default_rng(3)
    for _ in range(200):
        mv = policy.sample_move(b, rng)
        assert 0 <= mv <= 81, '非法下标 {}'.format(mv)


# --------------------------------------------------------------------------- #
# 10. 自对弈工作点默认值（实测驱动）
# --------------------------------------------------------------------------- #

def _argparse_defaults():
    import ast
    path = os.path.join(ROOT, 'scripts', 'selfplay_train.py')
    with open(path, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == 'add_argument'):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        for kw in node.keywords:
            if kw.arg == 'default' and isinstance(kw.value, ast.Constant):
                out[node.args[0].value] = kw.value.value
    return out


def test_selfplay_workload_defaults_are_reasonable():
    """工作点默认值必须反映实测，而不是拍脑袋。

    旧默认 expand_topk=64 + rollout_steps=60：每次模拟要为 64 个子节点各跑
    60 步 rollout。实测 9 路 sims/s 仅 2.72；而 topk=8 + rollout 30 可达约
    3 倍。该值直接乘在每模拟成本上，是当前最大的单一杠杆。
    """
    d = _argparse_defaults()
    assert d.get('--expand-topk') == 8, \
        '--expand-topk 默认应为 8（实测工作点），实际 {}'.format(
            d.get('--expand-topk'))
    assert d.get('--rollout-steps') == 30, \
        '--rollout-steps 默认应为 30，实际 {}'.format(
            d.get('--rollout-steps'))


def test_async_worker_pins_torch_intraop_threads(monkeypatch):
    """自对弈 worker 必须把 torch intra-op 线程钉为 1。

    selfplay_train.py 从未调用 torch.set_num_threads()，而 torch 默认取
    os.cpu_count()。--parallel-games N 就是 N×核数 个 intra-op 线程抢同样多的
    核：8 worker × 24 核 = 192 线程抢 24 核，过度订阅会同时拖慢吞吐与尾延迟，
    也会让 mcts-threads 的配比失去意义。实测不限线程比钉为 1 快 1.43~1.5x
    （即钉为 1 后单进程变慢，但多进程总体大幅提速）。
    """
    import scripts.async_pipeline as ap

    calls = []
    stop = mp.get_context('spawn').Event()

    class _StubWorker:
        def __init__(self, *a, **k):
            pass

        def run(self):
            calls.append('run')

    def _fake_game(self, ai, n, m):
        # 必须置位，否则 run() 的 `while not stop_event.is_set()` 会死循环
        stop.set()
        return [], 0.0

    monkeypatch.setattr(ap.faulthandler, 'enable', lambda *a, **k: None)
    monkeypatch.setattr(ap.SelfPlayWorker, '_build_ai', lambda self: object())
    monkeypatch.setattr(ap.SelfPlayWorker, '_play_one_game', _fake_game)
    monkeypatch.setattr(ap.torch, 'set_num_threads', lambda n: calls.append(n))
    ap._async_selfplay_worker(0, 'm', None, make_args(), None, stop, None)
    assert 1 in calls, 'worker 入口必须 torch.set_num_threads(1)'


def test_sync_worker_pins_torch_intraop_threads(monkeypatch):
    """同步 worker 同理。"""
    import scripts.selfplay_train as st
    calls = []

    class _FakeAI:
        def __init__(self, *a, **k):
            pass

    class _FakeQueue:
        def put(self, item):
            pass

    monkeypatch.setattr(st, 'GoAI', _FakeAI)
    monkeypatch.setattr(st, 'self_play_game', lambda *a, **k: ([], 0.0))
    monkeypatch.setattr(st.torch, 'set_num_threads', lambda n: calls.append(n))
    st._selfplay_worker(0, 'm.pth', make_args(rollout_steps=30), _FakeQueue())
    assert 1 in calls, '同步 worker 必须 torch.set_num_threads(1)'


def test_selfplay_device_default_is_cpu():
    """自对弈 worker 默认必须用 CPU。

    崩溃根因：worker 沿用训练设备（默认 auto → npu），而全仓库没有
    torch.npu.set_device()，于是 N+1 个进程（父进程训练 + N worker）全部挤在
    NPU 0 上、各自持有独立 CANN 上下文并发创建 matmul primitive，最终报
    "could not create a primitive descriptor for a matmul primitive"。
    """
    d = _argparse_defaults()
    assert d.get('--selfplay-device') == 'cpu', \
        '--selfplay-device 默认应为 cpu（父进程独占加速器），实际 {}'.format(
            d.get('--selfplay-device'))


def test_resolve_selfplay_device_prefers_selfplay_device():
    from scripts.selfplay_train import _resolve_selfplay_device
    args = make_args(device='npu', selfplay_device='cpu', parallel_games=8)
    assert _resolve_selfplay_device(args) == 'cpu'


def test_shared_accelerator_across_workers_is_rejected():
    """加速器 + 多 worker 必须直接报错，而不是走到 CANN 崩溃。"""
    from scripts.selfplay_train import _resolve_selfplay_device
    args = make_args(device='npu', selfplay_device='npu', parallel_games=8)
    with pytest.raises(RuntimeError) as ei:
        _resolve_selfplay_device(args)
    msg = str(ei.value)
    assert 'npu' in msg.lower() or '加速' in msg
    assert 'selfplay-device' in msg, '报错应直接指明用哪个参数绕过'


def test_single_worker_may_use_accelerator():
    """只有 1 个 worker 时不存在多上下文竞争，应允许显式用加速器。"""
    from scripts.selfplay_train import _resolve_selfplay_device
    args = make_args(device='npu', selfplay_device='npu', parallel_games=1)
    assert _resolve_selfplay_device(args) == 'npu'


def test_sync_worker_builds_model_on_selfplay_device(monkeypatch):
    """worker 建模型时必须用自对弈设备，而不是训练设备。"""
    import scripts.selfplay_train as st
    seen = {}

    class _FakeAI:
        def __init__(self, *a, **kw):
            seen['device'] = kw.get('device')

    class _FakeQueue:
        def put(self, item):
            pass

    monkeypatch.setattr(st, 'GoAI', _FakeAI)
    monkeypatch.setattr(st, 'self_play_game', lambda *a, **k: ([], 0.0))
    st._selfplay_worker(0, 'm.pth', make_args(device='npu',
                                               selfplay_device='cpu'),
                        _FakeQueue())
    assert seen['device'] == 'cpu', \
        'worker 仍在用训练设备，实际 {}'.format(seen['device'])


def test_cpu_linear_workaround_is_bitwise_equivalent():
    """垫片必须与 nn.Linear 逐位等价，不能有任何数值差异。"""
    import torch
    import torch.nn as nn
    from src.inference import _cpu_linear_matmul
    torch.manual_seed(0)
    cases = (((5,), 5, 3), ((2, 5), 5, 3), ((2, 8, 5), 5, 3),
             ((3, 2, 4, 5), 5, 3), ((1, 1, 5), 5, 3))
    for shape_in, nin, nout in cases:
        lin = nn.Linear(nin, nout)
        x = torch.randn(*shape_in)
        ref = lin(x)
        got = _cpu_linear_matmul(x, lin.weight, lin.bias)
        assert got.shape == ref.shape, '形状变了: {} vs {}'.format(
            got.shape, ref.shape)
        assert torch.equal(ref, got), (
            '与 nn.Linear 数值不等: maxdiff={:.3e}'.format(
                float((ref - got).abs().max())))
    lin = nn.Linear(5, 3, bias=False)
    x = torch.randn(2, 5)
    assert torch.equal(lin(x), _cpu_linear_matmul(x, lin.weight))


def test_probe_cpu_linear_detects_healthy_path():
    """自检必须能真实跑通一次 nn.Linear——它就是"垫片是否生效"的判据。

    该环境的故障是静默的，只能靠真跑一次确认。健康环境下自检必须为真，
    否则说明探针本身写错了，会给出误导性的 FAIL。
    """
    import torch.nn.functional as F
    from src.inference import probe_cpu_linear
    real_linear = getattr(F.linear, '_goai_original', F.linear)
    try:
        F.linear = real_linear
        ok, detail = probe_cpu_linear()
        assert ok, '健康环境下自检应通过，实际: {}'.format(detail)
        # 装上垫片后也必须通过
        from src.inference import install_cpu_linear_workaround
        install_cpu_linear_workaround(device='cpu', verbose=False)
        ok2, detail2 = probe_cpu_linear()
        assert ok2, '安装垫片后自检应通过，实际: {}'.format(detail2)
    finally:
        F.linear = real_linear


def test_workaround_is_idempotent():
    """重复安装必须是空操作（幂等），避免反复替换包装函数。"""
    import torch.nn.functional as F
    from src.inference import install_cpu_linear_workaround
    real_linear = getattr(F.linear, '_goai_original', F.linear)
    try:
        F.linear = real_linear
        assert install_cpu_linear_workaround(device='cpu') is True
        first = F.linear
        assert install_cpu_linear_workaround(device='cpu') is False
        assert F.linear is first, '重复安装替换了函数对象'
    finally:
        F.linear = real_linear


def test_workaround_does_not_pollute_npu_path(monkeypatch):
    """NPU 路径不得被垫片污染：NPU 上的 linear 是好的（走真实算子）。"""
    import torch.nn.functional as F
    from src.inference import install_cpu_linear_workaround
    real_linear = getattr(F.linear, '_goai_original', F.linear)
    try:
        F.linear = real_linear
        assert install_cpu_linear_workaround(device='npu') is False
        assert F.linear is real_linear, 'NPU 路径不应安装垫片'
        assert install_cpu_linear_workaround(device='cuda') is False
        assert F.linear is real_linear, 'CUDA 路径不应安装垫片'
    finally:
        F.linear = real_linear


def test_cpu_goai_constructor_installs_workaround():
    """只要构造了 CPU GoAI 就必须装垫片——不能只装在 worker 入口。

    垫片原先只在 _selfplay_worker / _async_selfplay_worker 里安装，于是
    **串行路径与主进程自身的 CPU 推理完全没有垫片**：手动指定 --device cpu
    时主进程直接建 GoAI 跑 CPU 前向，照样崩在 nn.Linear。安装点必须在
    GoAI.__init__ 按解析后的设备决定，才能覆盖 worker、串行、evaluate、
    cli_play、WebUI 等全部 CPU 路径。
    """
    import torch.nn.functional as F
    from src.inference import GoAI
    real_linear = getattr(F.linear, '_goai_original', F.linear)
    try:
        F.linear = real_linear
        GoAI(model_path=None, board_size=5, device='cpu', use_amp=True)
        assert getattr(F.linear, '_goai_shim', False) is True, \
            '构造 CPU GoAI 后未安装 F.linear 垫片'
    finally:
        F.linear = real_linear


def test_npu_goai_constructor_does_not_install_workaround():
    """NPU 上的 linear 是好的，构造 NPU GoAI 不得装垫片。"""
    import torch.nn.functional as F
    from src.inference import GoAI
    real_linear = getattr(F.linear, '_goai_original', F.linear)
    try:
        F.linear = real_linear
        ai = GoAI(model_path=None, board_size=5, device='cpu', use_amp=True)
        # 手动把已解析设备改成 npu，模拟 NPU 分支不走安装逻辑
        ai.device = 'npu'
        ai.is_npu = True
        install = getattr(F.linear, '_goai_shim', False)
        assert install is True, '前置：CPU 构造应已安装'
        F.linear = real_linear          # 复位后再验证非 CPU 不装
        from src.inference import install_cpu_linear_workaround
        assert install_cpu_linear_workaround(device=ai.device) is False
        assert F.linear is real_linear
    finally:
        F.linear = real_linear


def test_bench_defaults_match_production_defaults():
    """基准工具必须衡量实际要跑的工作点。

    bench_mcts.py 曾用 expand_topk=32 / rollout_steps=60，与生产默认脱节，
    于是"基准显示没变"而"生产其实更慢"——这类漂移会让人得出错误结论。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_bench_mcts_probe',
        os.path.join(ROOT, 'scripts', 'bench_mcts.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bench_defaults = vars(mod.build_parser().parse_args([]))
    prod = _argparse_defaults()
    for bench_flag, prod_flag in (('expand_topk', '--expand-topk'),
                                  ('rollout_steps', '--rollout-steps')):
        assert bench_defaults[bench_flag] == prod[prod_flag], (
            'bench_mcts.py 的 {}={} 与生产默认 {}={} 不一致'.format(
                bench_flag, bench_defaults[bench_flag],
                prod_flag, prod[prod_flag]))


def test_expand_topk_8_flattens_the_dynamic_ramp():
    """记录一个必须让使用者知道的副作用。

    _dynamic_topk 用 min(阶段上限, expand_topk)：expand_topk=64 时实际被封顶到
    32，且随进度 8→16→32 爬升；改成 8 之后全程恒为 8，搜索后期的精细化阶段
    被取消。这是选择该工作点所付出的棋力代价。
    """
    ai = RecordingAI(50)
    m = MCTS(ai, board_size=7, num_threads=1, expand_topk=8,
             dynamic_topk=True)
    seen = {m._dynamic_topk(s, 100) for s in (0, 20, 50, 80, 99)}
    assert seen == {8}, 'expand_topk=8 时各阶段应恒为 8，实际 {}'.format(seen)

    m64 = MCTS(ai, board_size=7, num_threads=1, expand_topk=64,
               dynamic_topk=True)
    ramp = [m64._dynamic_topk(s, 100) for s in (0, 20, 50, 80, 99)]
    assert ramp == [8, 8, 16, 32, 32], \
        'expand_topk=64 时的爬升行为被改动了，实际 {}'.format(ramp)
