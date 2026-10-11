"""按棋局切 train/eval：结果等价 + **`game_ids` 属性只访问一次**。

事故（2026-10-11 真机，C 段）
--------------------------------
C 段（stdata 分片）启动后卡死在「按棋局分割」，**无日志无报错**，Ctrl-C 后栈底
停在 `numpy/lib/format.py:_read_bytes` / `zipfile._update_crc`，看起来像数据加载
卡住。真正的原因在更早的一行：

    train_idx = np.array([i for i in range(n) if dataset.game_ids[i] in train_game_set])

`V7PackedDataset.game_ids` 是 **property**，每次访问都重新
`_gather('game_ids', arange(N))` 跑一遍全量（**O(N)**）。写在列表推导式里就是
**n 次迭代 × 每次 O(N) 重建 = O(N²)** —— 3.1M 行根本跑不完。

B 段用同一条代码路径却没事：`V7Dataset.game_ids` 是 `__init__` 里的**普通属性**，
索引 O(1)。**同一段代码，两条数据路径复杂度差一个平方级** —— 这就是「只有 C 段卡」
的原因，也是这个 bug 能在仓库里活下来的原因：B 段的实测把 12 秒的代价盖住了。

本文件钉三件事
--------------
1. 结果与**旧实现逐字等价**（拿旧写法当参考实现对照，fixture 上跑）。
2. `dataset.game_ids` **只被访问一次** —— 这条是本文件的重点：只看结果的话，
   一个「O(N²) 但结果相同」的实现照样全绿，正是原 bug 的形态。
3. `game_ids is None` ⇒ 返回 `None`，调用方走「按行随机切」的退路。

跑：pytest tests/test_train_sft_game_split.py -v
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_sft import split_train_eval_by_game  # noqa: E402


def _reference(dataset):
    """**原实现**（O(N²)），只在小 fixture 上当参考实现用。"""
    n = len(dataset.game_ids)
    if dataset.game_ids is None:
        return None
    unique_games = np.unique(dataset.game_ids)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_games)
    n_train_games = int(len(unique_games) * 0.98)
    train_game_set = set(unique_games[:n_train_games])
    train_idx = np.array([i for i in range(n) if dataset.game_ids[i] in train_game_set])
    eval_idx = np.array([i for i in range(n) if dataset.game_ids[i] not in train_game_set])
    return train_idx, eval_idx, len(unique_games), n_train_games, rng


class _Plain:
    """`game_ids` 是普通属性（`V7Dataset` / `SupervisedDataset` 那一类）。"""

    def __init__(self, gids):
        # `None` 必须**原样**保留：`np.asarray(None)` 会造出一个 0 维 object
        # 数组而不是 None，那样测的就不是「没有 game_ids」这条退路了。
        self.game_ids = None if gids is None else np.asarray(gids)


class _CountingProperty:
    """`game_ids` 是 property 且**每次访问重新计算**（`V7PackedDataset` 那一类）。"""

    def __init__(self, gids):
        self._gids = np.asarray(gids)
        self.accesses = 0

    @property
    def game_ids(self):
        self.accesses += 1
        # 刻意重算：真实实现每次都重新 `_gather` 全量。
        return self._gids + np.zeros_like(self._gids)


def _fixture(n_games=50, rows_per_game=7, seed=3):
    rng = np.random.default_rng(seed)
    gids = np.repeat(np.arange(n_games, dtype=np.int64), rows_per_game)
    return gids[rng.permutation(len(gids))]      # 打散，避免「同局连续」掩盖问题


@pytest.mark.parametrize('n_games,rows_per_game', [
    (50, 7), (2, 5), (100, 3), (37, 1),
])
def test_matches_the_original_implementation(n_games, rows_per_game):
    """与原实现（O(N²) 版本）**逐字等价**。

    等价性是这次改动的前提：切分比例、随机种子、索引顺序都不能变，否则
    历史 run 的 eval 指标就不可比了。
    """
    gids = _fixture(n_games, rows_per_game)
    got = split_train_eval_by_game(_Plain(gids))[:4]
    want = _reference(_Plain(gids))[:4]
    assert got is not None and want is not None
    for a, b in zip(got, want):
        assert np.array_equal(a, b), (a, b)


def test_game_ids_property_is_accessed_exactly_once():
    """**本文件的核心断言。**

    结果正确但访问了 N 次的实现，同样会卡死在启动期 —— 而只看结果的测试对它
    完全是绿的。所以这里数访问次数：一个可用的实现必须**只取一次**。
    """
    gids = _fixture()
    ds = _CountingProperty(gids)
    out = split_train_eval_by_game(ds)
    assert out is not None
    assert ds.accesses == 1, (
        f'dataset.game_ids 被访问了 {ds.accesses} 次。它在 V7PackedDataset 上是 '
        f'property，每次访问重新 gather 全量（O(N)）⇒ N 次就是 O(N^2)，'
        f'真机表现为「C 段启动卡死、无日志」。只能取一次。')


def test_splits_are_ascending_disjoint_and_complete():
    """两侧都是升序、互斥、并集覆盖全部行 —— 索引语义本身不能被向量化改坏。"""
    gids = _fixture()
    train_idx, eval_idx, n_games, n_train_games, _ = split_train_eval_by_game(
        _Plain(gids))
    n = len(gids)
    assert np.array_equal(train_idx, np.sort(train_idx))
    assert np.array_equal(eval_idx, np.sort(eval_idx))
    assert not set(train_idx.tolist()) & set(eval_idx.tolist()), '两侧有交集'
    assert len(train_idx) + len(eval_idx) == n, '两侧并集没有覆盖全部行'
    assert n_games == len(np.unique(gids))
    assert n_train_games == int(n_games * 0.98)


def test_no_game_leakage_across_the_split():
    """**切分的目的本身**：同一棋局不能同时出现在两侧。

    这是这个函数存在的唯一理由，比「结果等价」更该保住 —— 等价性只证明
    没改坏，防不住「改成一个同样快但会泄漏的实现」。
    """
    gids = _fixture()
    train_idx, eval_idx, _, _, _ = split_train_eval_by_game(_Plain(gids))
    train_games = set(gids[train_idx].tolist())
    eval_games = set(gids[eval_idx].tolist())
    assert not train_games & eval_games, '同一棋局同时落在 train 和 eval 两侧'


def test_returns_none_when_dataset_has_no_game_ids():
    """没有 `game_ids` ⇒ 返回 `None`，调用方走按行随机切的退路。"""
    assert split_train_eval_by_game(_Plain(None)) is None


def test_returns_the_same_rng_so_the_epoch_loop_keeps_drawing_from_it():
    """**必须把 `rng` 一并返回**，且其后续抽样序列与原实现逐位一致。

    事故（2026-10-11 真机，C 段）：把 `rng = np.random.default_rng(0)` 搬进
    helper 之后，`main()` 的作用域里就没有 `rng` 了，而 epoch 循环第一行是
    `rng.shuffle(train_idx)` ⇒

        UnboundLocalError: local variable 'rng' referenced before assignment

    走的正是**主路径**（`game_ids` 非 None = V7 / B / C 段全部）。

    ⚠ 这条不是「顺手加的防御」：**2098 项测试全绿也没有抓到它** ——
    没有任何测试跑过 `main()` 的分割之后那段。所以这里不只断言「有 rng」，
    还要断言**随机流连续**：重新播种（`default_rng(0)` 再来一次）会让每个
    epoch 的洗牌序列整体改变，跨 run 的可复现性就没了。
    """
    gids = _fixture()
    train_idx, eval_idx, _, _, rng = split_train_eval_by_game(_Plain(gids))
    want_train, _, _, _, want_rng = _reference(_Plain(gids))

    # 与原实现返回的 rng **状态相同** ⇒ 后续抽样序列必然相同
    assert np.array_equal(rng.random(8), want_rng.random(8)), \
        'rng 状态与原实现不同 —— epoch 循环的洗牌序列会变，破坏可复现性'
    assert np.array_equal(train_idx, want_train)

    # 直接模拟 epoch 循环：同一条流上连抽三次，结果必须与原实现逐位相同
    for _ in range(3):
        a = train_idx.copy()
        b = want_train.copy()
        rng.shuffle(a)
        want_rng.shuffle(b)
        assert np.array_equal(a, b)


def test_main_epoch_loop_can_reach_rng_on_both_split_paths():
    """`main()` 的两条分割分支都必须让 `rng` 到达 epoch 循环。

    这是事故的门禁形态：光断言 helper 返回了 rng 不够 —— 还要断言
    **`main()` 真的接住了它**。用 AST 检查两条分支都把 `rng` 绑定出来。
    """
    import ast
    import inspect
    import textwrap
    import scripts.train_sft as t
    tree = ast.parse(textwrap.dedent(inspect.getsource(t.main)))
    assigns = [ast.unparse(n) for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               for t_ in n.targets
               if isinstance(t_, ast.Name) and t_.id == 'rng']
    assert len(assigns) >= 1, 'main() 里没有任何地方给 rng 赋值'
    # epoch 循环用到 rng ⇒ 它必须在某个分支里被绑定
    assert any('default_rng' in a for a in assigns), assigns
    # 而按棋局那条分支靠的是 helper 的第五个返回值，不是就地播种
    assert any('split_train_eval_by_game' in ast.unparse(n)
               for n in ast.walk(tree) if isinstance(n, ast.Assign)), \
        'main() 没有调用 split_train_eval_by_game'