"""位置散列：**打通「KataGo 标签」与「本仓 34.2M 行数据集」的唯一 join 键**。

为什么不能用 `game_id`
---------------------
`build_dataset.build()`（`build_dataset.py:261-279`）有一串过滤：棋盘过大、
让子棋、非法坐标、缺 `RE`。实测 tgz 前 400 个成员丢掉 179 个（**45%**），
而 `tarfile` 成员顺序也不等于 glob 顺序 ⇒ **`game_id` 无法按位置推断**
（`probe0_join.py` 实测 0/5 匹配）。用 `game_id` join 的方案全部作废。

为什么不用 `(sgf 相对路径, ply)`
-------------------------------
那要求标签侧与 build 侧**各自独立枚举 SGF 且顺序完全一致**。只要枚举规则
动一行（加一种过滤、换 glob 顺序），历史标签就静默错位 —— 而且错位后
「join 不上」和「本来就不该 join」长得一模一样，无法区分。

散列口径
--------
    pos_hash(board 361B, to_play 1B, ko 2B)

只取**盘面 + 行棋方 + 打劫点**，不含历史。`scripts/probe_transposition.py`
的 key A 已实测：264 个重复项里 0 例因历史差异被拆开 ⇒ 同一局面必得同一
散列，无歧义。

 **两侧必须用同一份实现与同一组种子常量**，否则散列全不同 —— 而症状是
「join 结果为空」，看不出是种子不一致。本模块是**唯一事实来源**：
`probe0_join.py` 与 `label_sgf.py` 都从这里 import。

 **to_play 与 ko 不可省略**。`probe0_join.py` 从 KataGo npz 反推时把
`to_play` 恒置 +1、`ko` 恒置 −1（`fillRowV7` 的 ch6 是 ko-ban ∪ superko
**集合**，反推不出单点），而本仓 npz 侧是**真值**。⇒ **stdata ↔ 本仓的
散列口径天然不对齐**，那个探针的「无交集」结论里混着这个偏差。
`label_sgf.py` 侧用 `replay_to` 拿到真 `to_play`/`ko`，**与本仓一致**，
所以自打标签这条路的 join 是可靠的。
"""

import numpy as np

#: 散列种子。**改了它，之前存下的所有 `pos_hash` 全部作废**（`label_sgf.py`
#: 的 `pos_hash` 列会静默对不上）。所以这里是模块级常量、只在首次定义。
_HASH_SEED = 0x5EED19191919
_W1 = np.random.default_rng(_HASH_SEED).integers(1, 2 ** 63, 361, dtype=np.uint64)
_W2 = np.random.default_rng(_HASH_SEED + 1).integers(1, 2 ** 63, 361, dtype=np.uint64)
_SEED = np.uint64(0x243F6A8885A308D3)

BOARD = 19
BOARD_CELLS = BOARD * BOARD


def pos_hash_block(boards, to_play, ko):
    """一批局面的 64 位位置散列。

    Args:
        boards: ``(B,19,19)`` int8，取值 ``-1/0/1``（黑/空/白）。
        to_play: ``(B,)`` int8，``±1``（本手执子方）。
        ko:      ``(B,)`` int16，打劫点；``-1`` 表示无。

    Returns:
        ``(B,)`` uint64。

    性能：向量化线性散列，~675K 行/s ⇒ 34.2M 行约 1 分钟。逐行 `blake2b`
    在这个量级慢约 100×，不可用。

     **溢出是有意的，但必须全程在无符号语义下算。** int64 乘法溢出后
    ``astype(uint64)`` 会触发 `RuntimeWarning` 且高位置换逻辑不可控 ——
    首版就是这么坏的：唯一性与单格敏感性全部失效。这里所有中间量都先
    转 `uint64`，溢出按 C 标准回绕；2⁶⁴ 的模对散列完全够用。
    """
    b = np.asarray(boards)
    b = b.reshape(b.shape[0], -1)
    if b.shape[1] != BOARD_CELLS:
        raise ValueError(f'boards 每行应有 {BOARD_CELLS} 格，实测 {b.shape[1]}')
    n = b.shape[0]
    idx = (b.astype(np.int8) + np.int8(1)).astype(np.uint64)   # -1/0/1 → 0/1/2
    h1 = idx.dot(_W1)
    h2 = idx.dot(_W2)
    out = h1 * np.uint64(0x9E3779B97F4A7C15) + h2 + _SEED
    tp = np.asarray(to_play).reshape(n).astype(np.int8).astype(np.uint64)
    ko16 = np.asarray(ko).reshape(n).astype(np.int16).astype(np.uint64)
    out = out + tp * np.uint64(0xC2B2AE3D27D4EB4F) \
        + (ko16 + np.uint64(1)) * np.uint64(0x165667B19E3779F9)
    return out


def pos_hash_one(board, to_play, ko):
    """单局面散列（`label_sgf.py` 逐局调用）。"""
    b = np.asarray(board, dtype=np.int8).reshape(1, BOARD, BOARD)
    return pos_hash_block(b, np.array([to_play], np.int8),
                          np.array([ko], np.int16))[0]
