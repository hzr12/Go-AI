"""MCTS _backup 向量化（--mcts-vector-backup）等价性测试。

覆盖:
  - vector_backup=False 与 True 两条路径产生完全相同的 visit / value_sum /
    proved / virtual_loss（visit 精确、value_sum 容差 1e-9、proved 严格一致）
  - 默认值是 True（vector_backup 默认开启）
  - 符号翻转（value_sum 存对手视角）语义保持
  - 单节点路径、含 proven 传播的路径均等价
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search.mcts import MCTS, MCTSNode


class _FakeAI:
    """无需网络的占位 AI——测试只直接调用 _backup，不触发推理。"""

    def predict_batch(self, states):
        raise AssertionError("_backup 测试不应触发网络前向")


def _mk_mcts(vector_backup):
    return MCTS(_FakeAI(), board_size=9, num_threads=1, vector_backup=vector_backup)


def _build_chain(depth, priors=None):
    """构造一条 depth 层的链：root -> n1 -> ... -> leaf。"""
    root = MCTSNode(board=None, my_hist=[], op_hist=[], to_play=1, move_int=-1)
    cur = root
    for i in range(depth):
        mv = i + 1
        child = MCTSNode(board=None, my_hist=[], op_hist=[], to_play=-1,
                         move_int=mv, parent=cur, prior=0.1)
        cur.children[mv] = child
        cur = child
    return root, cur


def _snapshot(root):
    """按 move_int 顺序收集整棵树的 (visit, value_sum, proved, virtual_loss)。"""
    out = []

    def walk(n):
        out.append((n.move_int, n.visit, n.value_sum, n.proved, n.virtual_loss))
        for c in n.children.values():
            walk(c)

    walk(root)
    return sorted(out, key=lambda t: (t[0] if t[0] >= 0 else -1))


def _set_virtual_loss(root, vl=3.0):
    def walk(n):
        n.virtual_loss = vl
        for c in n.children.values():
            walk(c)

    walk(root)


@pytest.mark.parametrize("depth", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("v_leaf", [0.0, 0.37, -0.82, 1.0, -1.0])
def test_vector_backup_equivalence(depth, v_leaf):
    """两条路径对同一棵树、同一叶子价值产生完全一致的统计量。"""
    trees = []
    for vb in (False, True):
        m = _mk_mcts(vb)
        root, leaf = _build_chain(depth)
        _set_virtual_loss(root, 3.0)
        path = []
        n = root
        while n is not leaf:
            path.append(n)
            n = next(iter(n.children.values()))
        path.append(leaf)
        m._backup(path, v_leaf=v_leaf)
        trees.append(_snapshot(root))

    assert len(trees[0]) == len(trees[1])
    for (mv, v0, s0, p0, vl0), (mv1, v1, s1, p1, vl1) in zip(trees[0], trees[1]):
        assert mv == mv1
        assert v0 == v1, f"visit 不一致 @move {mv}: {v0} vs {v1}"
        assert s0 == pytest.approx(s1, abs=1e-9), \
            f"value_sum 不一致 @move {mv}: {s0} vs {s1}"
        assert p0 == p1, f"proved 不一致 @move {mv}: {p0} vs {p1}"
        assert vl0 == vl1, f"virtual_loss 未回收 @move {mv}: {vl0} vs {vl1}"


def test_virtual_backup_default_true():
    """vector_backup 默认开启（用户决策）。"""
    m = MCTS(_FakeAI(), board_size=9, num_threads=1)
    assert getattr(m, 'vector_backup', None) is True


def test_backup_clears_virtual_loss_both_paths():
    """virtual_loss 必须在所有节点清零（两条路径都应如此）。"""
    for vb in (False, True):
        m = _mk_mcts(vb)
        root, leaf = _build_chain(4)
        _set_virtual_loss(root, 7.5)
        path = []
        n = root
        while n is not leaf:
            path.append(n)
            n = next(iter(n.children.values()))
        path.append(leaf)
        m._backup(path, v_leaf=0.5)
        for _, _, _, _, vl in _snapshot(root):
            assert vl == 0.0, f"vector_backup={vb} 时 virtual_loss 未清零"


def test_backup_sign_flip_semantics():
    """value_sum 存对手视角：path 长 3 时 root 拿 -v、其子拿 +v（逐层交替）。"""
    m = _mk_mcts(True)
    root, leaf = _build_chain(2)
    path = [root, next(iter(root.children.values())), leaf]
    m._backup(path, v_leaf=0.4)
    # 叶子 visit 由 _expand 设置，此处不更新；自叶向根：
    #   n1（idx=1）先 v=-0.4 → value_sum += +0.4
    #   root（idx=0）再 v=+0.4 → value_sum += -0.4
    assert root.value_sum == pytest.approx(-0.4, abs=1e-9)
    assert next(iter(root.children.values())).value_sum == pytest.approx(0.4, abs=1e-9)
    assert root.visit == 1
    assert next(iter(root.children.values())).visit == 1


def test_backup_proven_propagation_equivalence():
    """MCTS-Solver 的 proved ±1 传播在两条路径上一致。"""
    trees = []
    for vb in (False, True):
        m = _mk_mcts(vb)
        root = MCTSNode(board=None, my_hist=[], op_hist=[], to_play=1, move_int=-1)
        # 三个子节点：leaf 标 proved=-1；另两个标 proved=1
        for i, pr in enumerate([(-1, -1), (1, 1), (2, 1)]):
            c = MCTSNode(board=None, my_hist=[], op_hist=[], to_play=-1,
                         move_int=pr[0], parent=root, prior=0.3)
            c.proved = pr[1]
            root.children[pr[0]] = c
        # 模拟 backup：先访问 proved=-1 的子节点（触发 root.proved=1）
        child = root.children[-1]
        m._backup([root, child], v_leaf=0.0)
        trees.append((root.proved, {k: c.proved for k, c in root.children.items()}))
    assert trees[0] == trees[1]
    assert trees[0][0] == 1
