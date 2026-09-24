"""C2NET 上下文解析：数据集加载**全部** npz 分片，预训练权重确定性选择。

回归问题 1（train_sft.py）：C2NET 分支原为
    npz_files = glob.glob(os.path.join(ctx.dataset_path, '*.npz'))
    args.data = npz_files[0]          # ← 只取 glob 顺序的第一个，且未排序
导致上下文目录下有多个分片时只加载其中一个（选中哪个还依赖文件系统顺序）。
修复：把上下文的 dataset_path 原样交给 --data，目录走目录模式合并全部分片。

回归问题 2（selfplay_train.py）：直接运行 `python scripts/selfplay_train.py`
时 --model 默认为 None，该分支会生效，而原实现同样是未排序的 pth_files[0]，
多个 .pth 时选中哪个不确定。修复：sorted 后取第一个（按文件名确定性选择）。

本测试直接验证「全部 npz 都被合并」与「权重选择确定」，而不只是验证路径字符串。
"""
import glob as _glob_mod
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.selfplay_train import resolve_c2net_model
from scripts.train_sft import load_from_path, resolve_c2net_data


def _write_npz(path, n):
    """写一个含 n 条样本的最小 npz（字段与 SupervisedDataset 期望一致）。"""
    rng = np.random.default_rng(abs(hash(path)) % (2**31))
    np.savez(
        path,
        boards=rng.integers(-1, 2, (n, 9, 9)).astype(np.int8),
        my_hist=np.full((n, 3), -1, dtype=np.int16),
        op_hist=np.full((n, 3), -1, dtype=np.int16),
        ko=np.full(n, -1, dtype=np.int16),
        to_play=np.ones(n, dtype=np.int8),
        moves=np.zeros(n, dtype=np.int64),
        values=rng.uniform(-1, 1, (n, 1)).astype(np.float32),
    )


@pytest.fixture()
def shard_dir(tmp_path):
    """一个含 3 个 npz 分片（5+7+11=23 条）的目录。"""
    for name, n in (("a.npz", 5), ("b.npz", 7), ("c.npz", 11)):
        _write_npz(str(tmp_path / name), n)
    return tmp_path


def test_resolve_returns_directory_untouched(shard_dir):
    """上下文给目录 → 原样返回该目录（交由目录模式合并全部分片）。"""
    assert resolve_c2net_data(str(shard_dir)) == str(shard_dir)


def test_resolve_returns_file_untouched(tmp_path):
    """上下文给单文件 → 原样返回该文件。"""
    f = tmp_path / "only.npz"
    _write_npz(str(f), 4)
    assert resolve_c2net_data(str(f)) == str(f)


def test_resolve_handles_none_and_empty():
    """None / 空串原样返回（调用方据此不覆盖 --data）。"""
    assert resolve_c2net_data(None) is None
    assert resolve_c2net_data("") == ""


def test_directory_mode_loads_all_npz_shards(shard_dir):
    """核心回归：目录下 3 个 npz 分片必须**全部**加载并合并（5+7+11=23）。"""
    ds = load_from_path(str(shard_dir), board_size=9)
    assert len(ds) == 23, f"期望合并 23 条，实际 {len(ds)}（只加载了部分分片？）"


def test_resolved_directory_feeds_all_shards(shard_dir):
    """端到端：resolve_c2net_data 的结果直接喂 load_from_path 即可加载全部分片。"""
    resolved = resolve_c2net_data(str(shard_dir))
    assert len(load_from_path(resolved, board_size=9)) == 23


def test_single_npz_still_works(tmp_path):
    """单文件场景不受影响。"""
    f = tmp_path / "one.npz"
    _write_npz(str(f), 6)
    resolved = resolve_c2net_data(str(f))
    assert len(load_from_path(resolved, board_size=9)) == 6


# --------------------------------------------------------------------------- #
# C2NET 预训练权重：确定性选择（直接运行 selfplay_train.py 时会生效）
# --------------------------------------------------------------------------- #
def test_model_none_when_no_pretrain_path():
    """上下文没给预训练目录 → 返回 None（调用方不覆盖 --model）。"""
    assert resolve_c2net_model(None) is None
    assert resolve_c2net_model("") is None


def test_model_none_when_directory_has_no_pth(tmp_path):
    """目录里没有 .pth → None。"""
    assert resolve_c2net_model(str(tmp_path)) is None


def test_model_picks_alphabetically_first(tmp_path):
    """多个 .pth 时按文件名取字典序第一个。"""
    for name in ("z_last.pth", "a_first.pth", "m_mid.pth"):
        (tmp_path / name).write_bytes(b"x")
    assert resolve_c2net_model(str(tmp_path)) == str(tmp_path / "a_first.pth")


def test_model_choice_independent_of_glob_order(monkeypatch, tmp_path):
    """核心回归：即使 glob 返回逆序/乱序，结果也必须是字典序第一个。

    直接运行 selfplay_train.py 时该分支会生效，glob 顺序依赖文件系统，
    未排序会导致每次可能选中不同权重。
    """
    d = str(tmp_path)
    names = ["m_mid.pth", "a_first.pth", "z_last.pth"]
    # 故意让 glob 返回非字典序，模拟文件系统的任意返回顺序
    monkeypatch.setattr(_glob_mod, "glob",
                        lambda pat, **kw: [os.path.join(d, n) for n in reversed(names)])
    assert resolve_c2net_model(d) == os.path.join(d, "a_first.pth")

