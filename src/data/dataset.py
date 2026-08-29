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
        """
        bs = self.board_size
        B = len(idxs)
        states = np.zeros((B, 12, bs, bs), dtype=np.float32)
        moves_out = np.empty(B, dtype=np.int64)

        boards = self.boards[idxs]
        my_h = self.my_hist[idxs]
        op_h = self.op_hist[idxs]
        ko = self.ko[idxs]
        to_play = self.to_play[idxs]
        tforms = np.random.randint(0, 8, size=B)

        for i in range(B):
            b = boards[i]
            tp = int(to_play[i])
            # 复用 GoBoard 构造特征（单一真相来源）
            gb = self._board
            gb.board = b
            gb.ko_point = int(ko[i])
            gb.current_player = tp
            plane = gb.feature_planes(my_h[i].tolist(), op_h[i].tolist(), tp)

            # 随机对称增强（对同一变换同时作用于平面与着法）
            t = int(tforms[i])
            r = t % 4
            if r:
                plane = np.stack([np.rot90(plane[ch], r) for ch in range(12)])
            if t >= 4:
                plane = np.fliplr(plane)
            mv = int(self.moves[idxs[i]])
            if 0 <= mv < bs * bs:
                rr, cc = SYMMETRIES[t](*divmod(mv, bs), bs)
                mv = rr * bs + cc
            else:  # pass 或越界 -> 专用类别 board_size*board_size
                mv = bs * bs
            states[i] = plane
            moves_out[i] = mv

        values = self.values[idxs].astype(np.float32).reshape(-1, 1)
        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(moves_out).to(device),
            torch.from_numpy(values).to(device),
        )
