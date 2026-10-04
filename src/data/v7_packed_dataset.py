# -*- coding: utf-8 -*-
"""V7 **预算特征**分片的训练视图：读 ``stdata_to_npz.py`` 产出的 npz。

 **不要与 ``src/data/v7_dataset.py`` 混淆**
------------------------------------------------
那个模块里的 ``V7Dataset`` 是 :class:`~src.data.dataset.SupervisedDataset` 的
**子类**，吃 **board 级** 布局（``boards`` / ``my_hist`` / ``moves`` …），
22 通道由 :func:`spatial_global` 在训练时**实时算**。对应数据源是
``data/sgf_19x19_full.npz``。

本模块吃的是**另一种**布局：``data/stdata_v7_s*.npz``，22 通道已经由
``stdata_to_npz.py`` **预算好并位打包**（``spatial_packed``，``(N,22,46) uint8``），
外加全部 V7 标签。两者不是同一个东西，也不是替代关系。

为什么 V7 训练必须走本模块（而不是 3420 万行那条路）
--------------------------------------------------
``data/sgf_19x19_full.npz`` 虽然大 11 倍（34,202,713 vs 3,132,287 行），
但它**没有** V7 专用的标签：``ownership`` / ``futurepos`` / ``scoring`` /
``seki`` / ``scorebelief`` 一个都没有。而 ``futurepos`` 是段1 四个主目标之一。
所以 V7 只能吃 stdata 分片 —— 代价是数据量只有 9%。

与 ``SupervisedDataset._build_labels`` 的对齐
-------------------------------------------
产出的 dict **键集完全一致**（``next_move`` / ``outcome`` / ``outcome_black`` /
``game_weight`` / ``soft`` / ``soft_mask`` / ``w`` / ``score`` / ``sb_center`` /
``sb_upper`` / ``global`` / ``ownership`` / ``scoring`` / ``seki`` / ``future``），
但值从占位零换成**真标签**。三处语义差异必须留意：

1. ``next_move``：board 级路径靠 ``moves[idxs+1]`` 且要求同局；本类的分片
   **每行自带**该行的目标着法（``policy_player_rank[:,0]``），不需要向后看。
2. ``outcome``：board 级路径从 ``winrates`` 的符号推（>0→0，<0→1，=0→2）；
   本类的分片已经存成 {0,1,2} 类别索引，**直取**。
3. ``sb_center`` / ``sb_upper``：是**整数对**（``round(score)`` 与
   ``round(λ·100)``），不是两个桶的概率 —— 权威定义见
   ``katago_v7_loss.build_score_distr_target``。

 ``var_time_left`` 缺席
-----------------------
现有 ``data/stdata_v7_s*.npz`` 生成于本项目把 col22 接进转换器**之前**
（``meta_json.created = 2026-10-03T11:10:01``），因此**没有** ``var_time_left`` 键。
本类对此**显式跳过**（不喂 0），与 ``KataGoV7Loss`` 里的守卫一致。
要用这一项训模型，必须重跑 ``stdata_to_npz.py`` 重新生成全部分片。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.katago_npz import BOARD_STRIDE, unpack_binary_input

#: 分片里 policy target 的稀疏宽度（top-16）。
POLICY_TOPK = 16
ACTION_SIZE = BOARD_STRIDE * BOARD_STRIDE + 1

#: 构造期就跳过、只在标签里用到的元信息键。
_META_KEYS = ('meta_json', 'schema_version', 'target_model')

_REQUIRED = ('spatial_packed', 'global', 'policy_player_rank',
             'policy_player_prob', 'outcome', 'game_weight')


class V7PackedDataError(RuntimeError):
    """分片布局不符合预期。**必须报错，不许猜。**"""


def _permute_move_scalar(moves: np.ndarray, tform: int, bs: int) -> np.ndarray:
    """单个着法编号在 8 个 dihedral 变换下的重编号（与 dataset 侧同源）。

     必须与 ``dataset.permute_move_vector`` / ``_dihedral_batch`` 用**同一套**
    约定（``t >= 4`` 先沿 W 翻转，逆序；``k = t % 4`` 次逆时针 90°），
    否则空间平面转了、标签没转 —— 监督信号会指向错误的格点。
    """
    if tform == 0:
        return moves
    r, c = moves // bs, moves % bs
    if tform >= 4:
        c = bs - 1 - c
    for _ in range(tform % 4):
        r, c = c, bs - 1 - r          # 逆时针 90°
    return r * bs + c


def _permute_plane(plane: np.ndarray, tform: int) -> np.ndarray:
    """``(B,H,W)`` / ``(B,C,H,W)`` 的 dihedral 变换（与 ``_dihedral_batch`` 同约定）。"""
    if tform == 0:
        return plane
    out = plane
    if tform >= 4:
        out = out[..., ::-1]
    for _ in range(tform % 4):
        out = np.swapaxes(out, -1, -2)[..., ::-1]
    return np.ascontiguousarray(out)


class V7PackedDataset:
    """``data/stdata_v7_s*.npz`` 的只读视图，接口对齐 ``SupervisedDataset``。"""

    #: 与 ``SupervisedDataset`` 同名，供下游按现有方式取盘面尺寸。
    board_size = BOARD_STRIDE

    def __init__(self, shard_paths: Sequence[str], board_size: int = BOARD_STRIDE):
        if board_size != BOARD_STRIDE:
            raise V7PackedDataError(
                f'V7 分片只支持 {BOARD_STRIDE}×{BOARD_STRIDE}，实得 {board_size}')
        self.shard_paths = [str(p) for p in shard_paths]
        if not self.shard_paths:
            raise V7PackedDataError('至少要一个分片')

        collected: Dict[str, List[np.ndarray]] = {}
        offsets: List[int] = []
        total = 0
        for p in self.shard_paths:
            z = np.load(p, allow_pickle=True)
            n = int(np.asarray(z['spatial_packed']).shape[0])
            for k in z.files:
                if k in _META_KEYS:
                    continue
                collected.setdefault(k, []).append(np.asarray(z[k]))
            offsets.append(total)
            total += n
        offsets.append(total)

        self._shards = collected
        self._offsets = offsets
        self.N = total
        self.n_rows = total
        self._check_layout()

    # ------------------------------------------------------------------ #
    def _check_layout(self) -> None:
        missing = [k for k in _REQUIRED if k not in self._shards]
        if missing:
            raise V7PackedDataError(
                f'分片缺少必需键 {missing}。现有键：{sorted(self._shards)}')
        sp = self._shards['spatial_packed'][0]
        if sp.ndim != 3 or sp.shape[1] != 22 or sp.shape[2] != 46:
            raise V7PackedDataError(
                f'spatial_packed 形状应为 (N,22,46)，实得 {sp.shape}')
        g = self._shards['global'][0]
        if g.ndim != 2 or g.shape[1] != 19:
            raise V7PackedDataError(f'global 形状应为 (N,19)，实得 {g.shape}')
        r = self._shards['policy_player_rank'][0]
        if r.ndim != 2 or r.shape[1] != POLICY_TOPK:
            raise V7PackedDataError(
                f'policy_player_rank 宽度应为 {POLICY_TOPK}，实得 {r.shape}')

    def _gather(self, key: str, idxs: np.ndarray) -> np.ndarray:
        """跨分片取行（``idxs`` 是**全局**行号）。"""
        shards = self._shards.get(key)
        if shards is None:
            raise V7PackedDataError(
                f'分片没有 {key!r}。现有键：{sorted(self._shards)}')
        if idxs.size == 0:
            return shards[0][:0]
        starts = np.asarray(self._offsets[:-1], dtype=np.int64)
        sid = np.searchsorted(starts, idxs, side='right') - 1
        out = None
        for s in np.unique(sid):
            m = sid == s
            part = shards[int(s)][idxs[m] - starts[s]]
            if out is None:
                out = np.empty((idxs.size,) + part.shape[1:], dtype=part.dtype)
            out[m] = part
        return out  # type: ignore[return-value]

    def _has(self, key: str) -> bool:
        return key in self._shards

    def __len__(self) -> int:
        return self.N

    def describe(self) -> str:
        miss = [k for k in ('var_time_left',) if not self._has(k)]
        extra = ('\n 缺 %s —— 这些分片生成于 col22 接线之前，'
                 '需重跑 `stdata_to_npz.py` 才能训这一项' % miss) if miss else ''
        return ('V7PackedDataset: %d 行 / %d 分片，键 %d 个%s'
                % (self.N, len(self.shard_paths), len(self._shards), extra))

    # ------------------------------------------------------------------ #
    # 特征
    # ------------------------------------------------------------------ #
    def sample_spatial(self, idxs: np.ndarray) -> np.ndarray:
        """``(B,22,19,19) bool`` —— 解开位打包。"""
        return unpack_binary_input(self._gather('spatial_packed', idxs))

    def sample_global(self, idxs: np.ndarray) -> np.ndarray:
        """``(B,19) float32`` —— 19 维全局输入特征。"""
        return self._gather('global', idxs).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # 兼容 board 级路径期望的属性
    # ------------------------------------------------------------------ #
    @property
    def moves(self) -> np.ndarray:
        """本行的目标着法（= ``policy_player_rank[:,0]``）。

         语义与 board 级路径**不同**：那边 ``moves[i]`` 是「第 i 行之后实际走的
        那手」，监督目标要靠 ``moves[idxs+1]`` 取；而本分片每行**自带**该行的
        答案。所以 ``_v7_labels_and_moves`` 对本类改走 ``next_move``
        （见该函数的 ``sample_spatial`` 分支）—— 否则整体错位一行。
        """
        if not self._has('policy_player_rank'):
            raise V7PackedDataError('分片没有 policy_player_rank，无法给出 moves')
        mv = np.asarray(self._gather(
            'policy_player_rank', np.arange(self.N, dtype=np.int64)),
            dtype=np.int64)[:, 0]
        return np.where(mv >= 0, mv, -1)

    @property
    def game_ids(self) -> Optional[np.ndarray]:
        """每行所属棋局。**按棋局切 train/eval 是防泄漏的关键** ——
        同一局相邻位置若同时落在两侧，eval 指标会虚高。

         分片内的 ``game_ids`` 是**块内局部行号**（见 ``meta_json`` 的
        ``game_ids_note``），分片间会重复。这里统一加上分片偏移，造出
        全局唯一 id —— 否则第 0 片与第 1 片都会出现 id=0 的「同一局」。
        """
        if not self._has('game_ids'):
            return None
        gid = np.asarray(self._gather(
            'game_ids', np.arange(self.N, dtype=np.int64)), dtype=np.int64).copy()
        span = self._offsets[-2]
        if span > 0:
            for s in range(1, len(self._offsets) - 1):
                lo, hi = self._offsets[s], self._offsets[s + 1]
                gid[lo:hi] += self._offsets[s]
        return gid

    # ------------------------------------------------------------------ #
    # 标签
    # ------------------------------------------------------------------ #
    def _build_labels(self, idxs: np.ndarray, tforms: Optional[np.ndarray] = None,
                      augment: bool = False) -> Dict[str, Any]:
        """与 ``SupervisedDataset._build_labels`` 同构，但填**真标签**。"""
        idxs = np.asarray(idxs, dtype=np.int64)
        B = idxs.size
        bs = self.board_size
        permuted = bool(augment) and tforms is not None
        tf = (np.asarray(tforms, dtype=np.int64) if tforms is not None
              else np.zeros(B, dtype=np.int64))

        # ---- policy：**两路**都要（player 与 opp）----
        # stdata 的 `policyTargetsNCMove` 是两通道：index 0 = 行棋方（player）、
        # index 1 = 对手（opp），见 `trainingwrite.cpp:552-568`
        # （policyTarget0 → rowGlobal[26] w_policy_player，policyTarget1 → [28]）。
        # 旧实现**只做了 player**，而 `v7_loss_labels` 又用 `next_move`（= player
        #   的 rank[0]）去填 `policy_opp` —— 于是 #1 与 #2 两个 loss 项拿的是
        #   **同一个**目标，π_opp 白训。这两路必须分开。
        def soft_from(prefix: str) -> np.ndarray:
            dense = np.zeros((B, ACTION_SIZE), dtype=np.float32)
            rk = self._gather(prefix + '_rank', idxs).astype(np.int64, copy=False)
            pb = self._gather(prefix + '_prob', idxs).astype(np.float32, copy=False)
            ok = (rk >= 0) & (rk < bs * bs)
            cnt = np.where(ok, pb, 0.0).astype(np.float64)
            cnt[cnt < 0] = 0.0
            den = cnt.sum(axis=1, keepdims=True)
            # `*_prob` 是 KataGo 的**访问计数**（实测一行 853/16/12/4/1/1，和 887），
            #   必须归一化；全零行退回均匀分布而非 NaN。
            nrm = np.where(den > 0, cnt / np.where(den > 0, den, 1.0),
                           np.full_like(cnt, 1.0 / POLICY_TOPK))
            if not permuted:
                rows = np.repeat(np.arange(B), POLICY_TOPK).reshape(B, POLICY_TOPK)
                np.add.at(dense, (rows, np.where(ok, rk, 0)), nrm.astype(np.float32))
            else:
                for b in range(B):
                    moved = _permute_move_scalar(
                        np.where(ok[b], rk[b], 0), int(tf[b]), bs)
                    np.add.at(dense[b], np.where(ok[b], moved, 0),
                              nrm[b].astype(np.float32))
            return dense

        soft = soft_from('policy_player')
        soft_opp = soft_from('policy_opp') if self._has('policy_opp_prob') else None
        soft_mask = np.ones(B, dtype=np.float32)

        rank = self._gather('policy_player_rank', idxs).astype(np.int64, copy=False)
        valid = (rank >= 0) & (rank < bs * bs)
        next_move = np.where(valid[:, 0], rank[:, 0], -1).astype(np.int64)
        next_move_opp = None
        if self._has('policy_opp_rank'):
            ro = self._gather('policy_opp_rank', idxs).astype(np.int64, copy=False)
            ok_o = (ro >= 0) & (ro < bs * bs)
            next_move_opp = np.where(ok_o[:, 0], ro[:, 0], -1).astype(np.int64)
        if permuted:
            for b in range(B):
                if next_move[b] >= 0:
                    next_move[b] = _permute_move_scalar(
                        np.asarray([next_move[b]]), int(tf[b]), bs)[0]
                if next_move_opp is not None and next_move_opp[b] >= 0:
                    next_move_opp[b] = _permute_move_scalar(
                        np.asarray([next_move_opp[b]]), int(tf[b]), bs)[0]

        outcome = self._gather('outcome', idxs).astype(np.int64, copy=False)
        to_play = self._gather('global', idxs)[:, 18]   # ch18 = 当前行棋方符号
        outcome_black = np.where(outcome == 2, 2,
                                 np.where(to_play > 0, outcome, 1 - outcome))
        game_weight = self._gather('game_weight', idxs).astype(np.float32, copy=False)

        def planes(key: str, channels: int) -> np.ndarray:
            if not self._has(key):
                return np.zeros((B, channels, bs, bs), dtype=np.float32)
            arr = self._gather(key, idxs).astype(np.float32, copy=False)
            arr = arr.reshape(B, channels, bs, bs)
            if not permuted:
                return arr
            return np.stack([_permute_plane(arr[b], int(tf[b])) for b in range(B)])

        ownership = planes('ownership', 1)
        scoring = planes('scoring', 1)
        seki = planes('seki', 1)
        future = planes('futurepos', 2)

        score = (self._gather('score', idxs).astype(np.float32, copy=False)
                 if self._has('score') else np.zeros(B, dtype=np.float32))

        # ---- scorebelief：**(center, upper) 整数对**，不是概率 ----
        # 权威定义 `katago_v7_loss.build_score_distr_target`：
        #     center = round(score)；λ = score - (center - 0.5)；upper = round(λ·100)
        # 我最初误以为这是两个桶的概率，喂浮点会被 `scatter_` 静默算错。
        sb_center = np.rint(score).astype(np.int64)
        lam = score - (sb_center - 0.5)
        sb_upper = np.rint(np.clip(lam, 0.0, 1.0) * 100.0).astype(np.float32)

        def wcol(key: str, default: float = 1.0) -> np.ndarray:
            if not self._has(key):
                return np.full(B, default, dtype=np.float32)
            return self._gather(key, idxs).astype(np.float32, copy=False)

        w = {
            'policy': wcol('w_score'),          # policy_player 沿用 w_score
            'policy_opp': wcol('w_policy_opp'),
            'ownership': wcol('w_ownership'),
            'score': wcol('w_score'),
            'lead': wcol('w_lead'),
            'futurepos': wcol('w_futurepos'),
            'scoring': wcol('w_scoring'),
            # loss #11 的 futurepos 是**整体 2×bs²** 的一路权重，dataset 给什么
            # 就透传什么（见 katago_v7_loss 的承重语义注释）。
            'futurepos_h0': wcol('w_futurepos'),
            'futurepos_h1': wcol('w_futurepos'),
        }

        out = {
            'next_move': next_move,
            # `next_move_opp`：对手侧的 top-1。`v7_loss_labels` 优先用它填
            # `policy_opp`；缺席时退回 `next_move`（board 级路径的旧行为）。
            **({'next_move_opp': next_move_opp}
               if next_move_opp is not None else {}),
            'outcome': outcome,
            'outcome_black': outcome_black,
            'game_weight': game_weight,
            'soft': soft,
            # `soft_opp`：对手侧的搜索分布。与 `soft` 同口径（计数归一化）。
            **({'soft_opp': soft_opp} if soft_opp is not None else {}),
            'soft_mask': soft_mask,
            'w': w,
            'score': score,
            'sb_center': sb_center,
            'sb_upper': sb_upper,
            'global': self.sample_global(idxs),
            'ownership': ownership,
            'scoring': scoring,
            'seki': seki,
            'future': future,
        }
        # var_time_left 缺席时**不产出该键** —— loss 侧整项跳过，
        # 而不是拿 0 把这一路往「方差恒 0」硬拉。
        if self._has('var_time_left'):
            out['var_time_left'] = self._gather(
                'var_time_left', idxs).astype(np.float32, copy=False)
        return out


def resolve_shards(path: str) -> List[str]:
    """``--data`` 可给目录或单文件；目录则按文件名排序收集所有 ``*.npz``。"""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise V7PackedDataError(f'{path} 既不是文件也不是目录')
    out = sorted(os.path.join(path, f) for f in os.listdir(path)
                 if f.endswith('.npz'))
    if not out:
        raise V7PackedDataError(f'{path} 下没有 *.npz')
    return out


def load_v7_packed(path: str) -> V7PackedDataset:
    return V7PackedDataset(resolve_shards(path))