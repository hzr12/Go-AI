# -*- coding: utf-8 -*-
"""邻行 gather —— V7 的 ``boards[i±k]`` **按批 mmap 随机访问**（任务 B6）。

为什么单独一个模块（而不是加进 ``feature_v7.py``）
----------------------------------------------------
``src/data/feature_v7.py`` 的模块 docstring 自陈是「**纯函数**部分」，只 numpy、
无 I/O、无全局状态。而本模块做的是**按行号随机访问数据集列**：要 mmap、要越界
守卫、要跨局守卫，还要处理「数据源是 ``.npz``（zip 容器）时 mmap 无效」这件
只在这里才出现的事。依赖（``np.load`` 的 mmap 模式）与生命周期（跨批复用的
memmap 句柄）都不同 ⇒ 混在一起会让「纯函数模块」这个前提失效。

需要邻行做什么
--------------
==============================  ==================  ==========================
用途                             偏移                消费方
==============================  ==================  ==========================
ch15 前一手盘的梯子              ``-1``              ``spatial_channels_v7``
ch16 前二手盘的梯子              ``-2``              ``spatial_channels_v7``
futurepos 标签（§5.1）           ``+8`` / ``+32``    训练标签侧
散列 / 合法性（``pos_hash``）     ``-1`` 的 ``to_play`` / ``ko``
==============================  ==================  ==========================

⚠ **offset 是「带符号的行增量」**：``j = i + offset``。混用方向正是这类 API
最容易静默取错行的地方（brief 里就同时写过 ``offsets=(1,8,32)`` 与「offset 1
= ``i-1``」），所以两组常见取法都给了具名常量，让调用点自解释、让错方向变成
一眼能看出的负数或越界。

⚠ **默认 ``offsets=(1, 8, 32)`` 是占位值，不是任何真实调用点的取法。**
真实调用点只有两个，且都必须显式传参（见上面的两个常量）。

⚠ **本模块不解析「历史不足 ⇒ 回退复制」那套门控**（那是
``feature_v7.py::resolve_ladder_boards`` 的活，且必须**只解析一次**——
``feature_v7_ladders.py`` 曾把同一段门控解析两次、输出行抄错了通道）。本模块
只回答一个问题：``i + offset`` 这一行**能不能用**。

越界与跨局：为什么填 0 而不是 clamp
-----------------------------------
``i + offset`` 越界、或 ``game_ids[j] != game_ids[i]`` 时，本行标记
``valid=False`` 且盘面填 **0**。两种「看起来更友好」的做法都是错的：

  · **clamp 到边界行** —— 会把别局的（或别的 ply 的）盘面当成本行数据喂进
    ``iterLadders``。梯子搜索会照着那张盘面跑出一组「自洽但无意义」的通道，
    **不抛异常、不打日志**，只是训练分布被污染。
  · **沿用上一行的结果** —— 同样静默，且更隐蔽（因为上一行的盘面多半也是合法的）。

⇒ 这与 P0 的「id 复用 bug」是同一类风险：**取到的数据看起来完全合法**。

🔴 **因此 ``NeighborGather.boards[offset]`` 单独消费不安全。** 不可用行是 0，
而 0 是一张合法的空盘 —— 空盘上 ``iterLadders`` 什么也找不到，于是 ch15 变成
全 0，**看起来像「这一手没有梯子」**。必须先看 ``valid[offset]``，再交给
``feature_v7.py::resolve_ladder_boards`` 做官方的回退复制。本模块的
``docstring`` 在返回类型上也写了这条。
"""

from collections.abc import Mapping

import numpy as np

__all__ = [
    'LADDER_OFFSETS',
    'FUTUREPOS_OFFSETS',
    'DEFAULT_OFFSETS',
    'NeighborGather',
    'gather_neighbors',
]

#: ladder 三通道要的两块历史盘面：``i-1`` / ``i-2``（spec §2.2 / §5.4）。
LADDER_OFFSETS = (-1, -2)

#: futurepos 标签要的两块未来盘面：``i+8`` / ``i+32``（spec §5.1 / §5.4）。
FUTUREPOS_OFFSETS = (8, 32)

#: 签名里的默认偏移。⚠ **这是占位值，不是任何真实调用点的取法**
#: （真实取法见上面两个常量）。之所以仍保留，是因为它是任务书钦定的签名，
#: 改掉会让调用方以为「默认值是有用的」。
DEFAULT_OFFSETS = (1, 8, 32)


class NeighborGather(Mapping):
    """``gather_neighbors`` 的返回值：**按 offset 索引的**邻行盘面 + 可用性掩码。

    本类是 ``Mapping[int, np.ndarray]``（``g[-1]`` ⇒ ``(B,19,19)`` int8，与任务书
    钦定的返回类型 ``dict[int, np.ndarray]`` 同形），另外挂三个属性：

    ==================  ==================================================
    属性                 语义
    ==================  ==================================================
    ``.boards[offset]``  ``(B,19,19)`` int8，**不可用行填 0**（见模块 docstring）
    ``.valid[offset]``   ``(B,)`` bool，该行这一偏移**能不能用**
    ``.to_play[offset]`` ``(B,)`` int8，与该偏移**同行**的行棋方
    ``.ko[offset]``      ``(B,)`` int16，与该偏移**同行**的 simple-ko 点
    ==================  ==================================================

    ``ko`` 的口径是 ``GoBoard.ko_point``：**紧凑**扁平下标 ``r*n+c``，``-1`` = 无
    （与主数据集的 ``ko`` 列同口径，见 ``scripts/build_dataset.py:294``）。

    ⚠ ``to_play`` / ``ko`` 在**没传对应列**时是 ``None``，不是全 ``-1`` ——
    全 ``-1`` 与「无劫」「无行棋方」无法区分，正是 ``src/data/pos_hash.py`
    docstring 警告的那类假信号。
    """

    __slots__ = ('_boards', '_valid', '_to_play', '_ko')

    def __init__(self, boards, valid, to_play=None, ko=None):
        self._boards = dict(boards)
        self._valid = dict(valid)
        self._to_play = dict(to_play) if to_play is not None else None
        self._ko = dict(ko) if ko is not None else None
        for off in self._boards:
            if off not in self._valid:
                raise ValueError(f'offset {off:+d} 有盘面却没有 valid 掩码')
            if self._valid[off].shape != (self._boards[off].shape[0],):
                raise ValueError(
                    f'offset {off:+d} 的 valid 形状 {self._valid[off].shape} 与 '
                    f'批大小 {self._boards[off].shape[0]} 不符')

    # ---- Mapping 接口：`g[offset]` 取盘面 ---------------------------------
    def __getitem__(self, offset):
        return self._boards[int(offset)]

    def __iter__(self):
        return iter(self._boards)

    def __len__(self):
        return len(self._boards)

    # ---- 掩码与小列 --------------------------------------------------------
    @property
    def boards(self):
        """``{offset: (B,19,19) int8}``。⚠ **不可用行是 0，必须配 ``valid`` 用。**"""
        return self._boards

    @property
    def valid(self):
        """``{offset: (B,) bool}``。``False`` = 越界或跨局，该行**不可用**。"""
        return self._valid

    @property
    def to_play(self):
        """``{offset: (B,) int8}``，或没传列时的 ``None``。"""
        return self._to_play

    @property
    def ko(self):
        """``{offset: (B,) int16}``，或没传列时的 ``None``。"""
        return self._ko

    @property
    def offsets(self):
        """本批实际 gather 的偏移（``tuple``，保持传入顺序）。"""
        return tuple(self._boards)

    def all_valid(self, offset):
        """该偏移是否**整批**可用。

        给定这个问题的调用点只有一个：``feature_v7.py::resolve_ladder_boards``
        用它决定「能不能传 ``None`` 保住 ``ladder_channels`` 的 ``is`` 快路径」。
        写成方法而不是让调用点自己算全 ``and``，是为了让「整批 vs 逐行」这个
        区别在读代码时一眼可见 —— 逐行混合的历史门控是最容易写错的地方。
        """
        return bool(self._valid[int(offset)].all())


def _reject_npz(obj, name):
    """``np.load('x.npz')`` 返回的 ``NpzFile`` 直接当 memmap 用会**静默退化成
    「整份解压进内存」**，所以明确拦下。"""
    if isinstance(obj, np.lib.npyio.NpzFile):
        raise TypeError(
            f'{name} 是 NpzFile（.npz 的 zip 容器），**mmap 对压缩成员无效**：\n'
            f'  · `np.load("x.npz", mmap_mode="r")` 不会报错，但访问 .z[k] 会把整个\n'
            f'    成员解压进内存 —— boards 是 34.2M × 361 B = 12.3 GB，本机 13.9 GB。\n'
            f'  · 正确做法：先跑 `src.data.kata_label_join.materialize_dataset(npz, out_dir)`\n'
            f'    把列落成 `.npy`，再 `np.load(out_dir + "/boards.npy", mmap_mode="r")`。\n'
            f'  （该函数已存在且有测试，见 `tests/test_kata_label_join.py`。）\n'
            f'这里拦下而不是警告，是因为症状（OOM）与病因（用了错的数据源）相距太远。')


def _gather_column(col, j, valid):
    """按已算好的行号 ``j`` 取一列（带不可用行填 0），返回按行对齐的数组。

    ``boards`` 走这条路径时返回 ``(B,19,19)``、小列走时返回 ``(B,)`` ——
    「先索引再按掩码覆盖 0」对两者是同一段代码。

    ⚠ 不可用行的 ``j`` 可能越界，所以**先 clip 再覆盖 0**。clip 只影响不可用行：
    可用行的 ``j`` 由 :func:`gather_neighbors` 的 ``in_range`` 保证在界内，
    不会被 clip 改动。
    """
    if col is None:
        return None
    arr = np.asarray(col)
    if valid.all():
        return np.asarray(arr[j])
    out = np.array(arr[np.clip(j, 0, arr.shape[0] - 1)], copy=True)
    out[~valid] = 0
    return out


def gather_neighbors(boards_mmap, indices, offsets=DEFAULT_OFFSETS, *,
                     game_ids=None, to_play=None, ko=None):
    """一次 fancy-index 取出 ``indices ± offsets`` 的盘面（小列顺带取）。

    Args:
        boards_mmap: ``(N,19,19)`` int8，**必须是按需分页的 memmap**。
            ``np.load('boards.npy', mmap_mode='r')`` 或
            ``np.lib.format.open_memmap``；``.npz`` 直接传会报错（见
            :func:`_reject_npz` 与模块 docstring）。
        indices: ``(B,)`` 整数行号。**允许重复、允许无序**（预取器按 shuffle
            取行，两条都常见）—— 返回值一律与 ``indices`` **逐行对齐**。
        offsets: 带符号行增量 ``tuple[int, ...]``。``j = i + offset``。
            真实取法见 :data:`LADDER_OFFSETS` / :data:`FUTUREPOS_OFFSETS`；
            默认 :data:`DEFAULT_OFFSETS` 是占位值。
        game_ids: 可选 ``(N,)`` 整数。**给了就启用跨局守卫**：
            ``game_ids[j] != game_ids[i]`` ⇒ 该行不可用。
            ⚠ **必须给**：主数据集的 ``boards`` 是把 162,298 局首尾相接排成的
            一根大数组，**没有局的边界标记**，行号 ``i`` 与 ``i+8`` 可以分属两局。
            跨局取到的盘面**看起来完全合法**（它就是某个真实盘面），只是不属于
            这一手 ⇒ 静默错标签。
        to_play / ko: 可选 ``(N,)`` 列。对**每个**偏移都跟着取，理由见
            :class:`NeighborGather`。

    Returns:
        :class:`NeighborGather`。⚠ **``.boards[offset]`` 的不可用行填 0，
        必须配 ``.valid[offset]`` 消费**；直接喂给 ``ladder_channels`` 会得到
        「ch15 全 0」这种看起来无害、实际是「盘面丢了」的通道。

    性能契约：**每个偏移一次 gather**，不随 ``B`` 增长地逐行循环
    （``tests/test_v7_gather.py`` 用一个宽松的耗时上界把它钉住）。
    ``np.asarray(memmap[j])`` 对 ``j`` 是数组时是一次向量化随机读，numpy 只
    触碰被点到的页。
    """
    _reject_npz(boards_mmap, 'boards_mmap')
    for name, col in (('game_ids', game_ids), ('to_play', to_play), ('ko', ko)):
        if col is not None:
            _reject_npz(col, name)

    boards = np.asarray(boards_mmap)
    if boards.ndim != 3:
        raise ValueError(f'boards_mmap 形状不符：{boards.shape}，期望 (N,n,n)')
    n_rows = boards.shape[0]

    idx = np.asarray(indices)
    if idx.ndim != 1:
        idx = idx.reshape(-1)
    if idx.dtype.kind not in 'iu':
        raise ValueError(f'indices dtype 必须是整数，实得 {idx.dtype}')
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_rows):
        raise ValueError(
            f'indices 越界：[{int(idx.min())}, {int(idx.max())}] 不在 [0, {n_rows})。'
            f'⚠ 本函数**不为越界的 indices 兜底** —— 那是调用方取错了行号，'
            f'静默丢行会让样本数与 batch 形状对不上。')

    for name, col in (('game_ids', game_ids), ('to_play', to_play), ('ko', ko)):
        if col is not None and np.asarray(col).shape[0] != n_rows:
            raise ValueError(f'{name} 行数 {np.asarray(col).shape[0]} != boards '
                             f'{n_rows}；列与盘面必须来自同一份数据集')

    b_idx = idx.astype(np.int64, copy=False)
    gid = None if game_ids is None else np.asarray(game_ids)

    out_b, out_v, out_tp, out_ko = {}, {}, {}, {}
    for off in offsets:
        off = int(off)
        if off == 0:
            raise ValueError('offset 0 会让每个偏移都等于「自己」—— 邻行 gather '
                             '存在的意义就是取**别的**行；要当前盘就直接用 boards。')
        j = b_idx + off
        in_range = (j >= 0) & (j < n_rows)
        # 跨局守卫。⚠ 先按 in_range 取 game_ids 再比 —— 越界的 j 不能读。
        if gid is None:
            valid = in_range
        else:
            valid = np.zeros(idx.size, dtype=bool)
            np.logical_and(valid, in_range, out=valid)
            ok = np.flatnonzero(in_range)
            if ok.size:
                same = gid[j[ok]] == gid[idx[ok]]
                valid[ok[same]] = True

        if not valid.any():
            # 整批都不可用（常见于 ``indices`` 恰好落在各局开头 + 负偏移）。
            # 仍然走一次「填 0 的 gather」以保证返回形状/dtype 与可用时一致 ——
            # 让下游不必为「这一批正好全不可用」写分支。
            out_b[off] = np.zeros((idx.size,) + boards.shape[1:], dtype=boards.dtype)
        else:
            out_b[off] = _gather_column(boards, j, valid)
        out_v[off] = valid
        if to_play is not None:
            out_tp[off] = _gather_column(to_play, j, valid)
        if ko is not None:
            out_ko[off] = _gather_column(ko, j, valid)

    return NeighborGather(out_b, out_v,
                          out_tp if to_play is not None else None,
                          out_ko if ko is not None else None)