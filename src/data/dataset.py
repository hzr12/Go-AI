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
        均为 shape=(N, ...) 的 numpy 数组，N 相同。
        """
        self.boards = data['boards']
        self.my_hist = data['my_hist']
        self.op_hist = data['op_hist']
        self.ko = data['ko']
        self.moves = data['moves']
        self.values = data['values']
        self.to_play = data['to_play']
        self.N = self.boards.shape[0]
        self.board_size = self.boards.shape[1]
        self._board = GoBoard(self.board_size)  # 复用实例，避免重复分配

    def __len__(self):
        return self.N

    def sample_batch(self, idxs, device='cpu'):
        """
        给定样本下标（numpy int array），返回 (state_tensor, move_tensor, value_tensor)。
        state_tensor: (B, 12, H, W) float32
        move_tensor:  (B,) int64
        value_tensor: (B, 1) float32

        使用向量化批量特征构造（GoBoard.feature_planes_batched）+ 向量化对称增强，
        避免逐样本 Python 循环，训练吞吐显著更高。
        """
        bs = self.board_size
        B = len(idxs)
        boards = self.boards[idxs]
        my_h = self.my_hist[idxs]
        op_h = self.op_hist[idxs]
        ko = self.ko[idxs]
        to_play = self.to_play[idxs]
        tforms = np.random.randint(0, 8, size=B)

        # 批量构造 12 通道特征（B,12,H,W）
        states = GoBoard.feature_planes_batched(boards, my_h, op_h, to_play, ko)

        # 向量化对称增强：rot90(k=t%4) 旋转 (H,W) 平面；t>=4 时翻转 H 轴。
        # 注意: 旧逐样本版用 np.fliplr(plane) 对 (12,H,W) 翻转的是 H 轴(axis=1)，
        # 这里用显式切片 x[:, :, ::-1, :] 精确复刻，避免 np.fliplr 在 4D 上的隐式轴歧义。
        rot_k = tforms % 4
        for k in range(1, 4):  # k=0 无需旋转
            mask = rot_k == k
            if mask.any():
                states[mask] = np.rot90(states[mask], k=k, axes=(2, 3))
        for t in range(4, 8):
            mask = tforms == t
            if mask.any():
                states[mask] = states[mask][:, :, ::-1, :]

        # 向量化坐标对称变换（move）
        moves_out = np.full(B, bs * bs, dtype=np.int64)  # 默认 pass/越界 -> 专用类别
        mv = np.asarray(self.moves[idxs], dtype=np.int64)
        valid = (mv >= 0) & (mv < bs * bs)
        r, c = np.divmod(np.where(valid, mv, 0), bs)
        for t in range(8):
            mask = tforms == t
            if not mask.any():
                continue
            rr, cc = SYMMETRIES[t](r[mask], c[mask], bs)
            moves_out[mask] = rr * bs + cc

        values = self.values[idxs].astype(np.float32).reshape(-1, 1)
        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(moves_out).to(device),
            torch.from_numpy(values).to(device),
        )
