"""
监督学习数据集（紧凑内存布局 + 按 batch 随机对称增强）。

紧凑存储（每样本）：
    board   : int8  (board_size, board_size)   取值 -1/0/1
    my_hist : int16 (3,)                       己方前 3 手扁平坐标（-1 填充）
    op_hist : int16 (3,)                       对手前 3 手扁平坐标（-1 填充）
    ko      : int16                            劫禁着点（-1 无）
    move    : int16                            监督目标着法（0..size*size-1，pass=size*size）
    value   : int8                            胜负标签（+1 黑胜 / -1 白胜）
    to_play : int8                            该样本轮到谁落子（1 黑 / -1 白）

特征平面（12 通道）由 GoBoard.feature_planes 统一构造，保证与推理/评估一致。
运行时随机施加 8 种对称变换之一（等价于 8 倍静态增强，内存仅 1/8）。
"""

import numpy as np
import torch

from src.game.go_rules import GoBoard, SYMMETRIES


class SupervisedDataset:
    def __init__(self, data: dict):
        """
        data 必须含：boards(int8), my_hist(int16), op_hist(int16),
                       ko(int16), moves(int16), values(int8), to_play(int8)
        可选含：game_ids(int32) — 每个样本所属棋局 ID，用于 stratified split。
        均为 shape=(N, ...) 的 numpy 数组，N 相同。
        """
        self.boards = data['boards']
        self.my_hist = data['my_hist']
        self.op_hist = data['op_hist']
        self.ko = data['ko']
        self.moves = data['moves']
        self.values = data['values']
        self.to_play = data['to_play']
        self.game_ids = data.get('game_ids', None)
        self.N = self.boards.shape[0]
        self.board_size = self.boards.shape[1]
        self._board = GoBoard(self.board_size)  # 复用实例，避免重复分配

    def __len__(self):
        return self.N

    def sample_batch_numpy(self, idxs, rng=None):
        """
        给定样本下标，返回 numpy 版 (states, moves_out, values)：
            states    : (B, 12, H, W) float32
            moves_out : (B,) int64
            values    : (B, 1) float32

        与 sample_batch 逻辑完全相同（含向量化特征构造与 8 对称增强），只是不转
        torch 张量——供 MindSpore 训练脚本使用（910B 环境无 torch）。
        使用向量化批量特征构造（GoBoard.feature_planes_batched）+ 向量化对称增强，
        避免逐样本 Python 循环，训练吞吐显著更高。

        rng: 可选随机源，供多线程预取时各线程使用独立 Generator，避免竞争全局
            np.random。为 None 时沿用全局 np.random（保持原行为）。
        """
        bs = self.board_size
        B = len(idxs)
        boards = self.boards[idxs]
        my_h = self.my_hist[idxs]
        op_h = self.op_hist[idxs]
        ko = self.ko[idxs]
        to_play = self.to_play[idxs]
        if rng is None:
            tforms = np.random.randint(0, 8, size=B)
        else:
            tforms = rng.integers(0, 8, size=B)  # Generator 用 integers，非 randint

        # 批量构造 12 通道特征（B,12,H,W）
        states = GoBoard.feature_planes_batched(boards, my_h, op_h, to_play, ko)

        # 向量化对称增强：8 种变换（4 旋转 × 2 镜像），与 SYMMETRIES 坐标变换严格对齐。
        # np.rot90 是逆时针旋转，SYMMETRIES 是顺时针，故用 k=-k；
        # _FLIP 翻转列 (W/axis=3)，不是行 (H/axis=2)；
        # _FLIP_ROTxx = _ROTxx ∘ _FLIP，即先翻转再旋转。
        # 注意: np.fliplr 对2D数组翻转 axis=1(W)是正确的，但对4D数组翻转 axis=2(H)
        # 是错误的——此处用显式切片 [:, :, :, ::-1] 精确指定 W 轴。
        out = np.empty_like(states)
        for t in range(8):
            mask = tforms == t
            if not mask.any():
                continue
            arr = states[mask]
            if t >= 4:
                arr = arr[:, :, :, ::-1]            # flip W (axis=3)
            k = t % 4
            if k > 0:
                arr = np.rot90(arr, k=-k, axes=(2, 3))  # CW rotation
            out[mask] = arr
        states = out

        # 向量化坐标对称变换（move）
        moves_out = np.full(B, bs * bs, dtype=np.int64)  # 默认 pass/越界 -> 专用类别
        mv = np.asarray(self.moves[idxs], dtype=np.int64)
        valid = (mv >= 0) & (mv < bs * bs)
        r, c = np.divmod(np.where(valid, mv, 0), bs)
        for t in range(8):
            mask = tforms == t
            if not mask.any():
                continue
            # 只对有效走法做对称变换，pass/越界保留默认标签
            vmask = mask & valid
            if vmask.any():
                rr, cc = SYMMETRIES[t](r[vmask], c[vmask], bs)
                moves_out[vmask] = rr * bs + cc

        values = self.values[idxs].astype(np.float32).reshape(-1, 1)
        return states, moves_out, values

    def sample_batch(self, idxs, device='cpu'):
        """numpy 取批 + 转 torch 张量（等价于 sample_batch_numpy 后再 to(device)）。"""
        states, moves_out, values = self.sample_batch_numpy(idxs)
        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(moves_out).to(device),
            torch.from_numpy(values).to(device),
        )
