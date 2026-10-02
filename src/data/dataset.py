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

特征平面由 `GoBoard.feature_planes_batched` 统一构造，保证与推理/评估一致。
**通道数由构造参数 `n_channels` 决定，默认 12**（C7「12↔17，默认 12 保零回归」）：
12 = P4.3 之前的布局，喂 `in_channels=12` 的现役权重与旧数据，输出逐字节不变；
17 = 通道表全表（已退役 v21 代用过的布局，现役无模型消费，但白名单与数据侧
仍按 12..17 放行），通道表见 `src/game/go_rules.py` 的「特征平面」段注释。
训练期运行时随机施加 8 种对称增强之一（等价于 8 倍静态增强，内存仅 1/8）；
**评估期不施加**（`sample_batch_numpy(..., augment=False)`）—— 验证集不该被随机
翻转/旋转污染，详见该方法的 docstring。
"""

import numpy as np
import torch

from src.game.go_rules import GoBoard, SYMMETRIES, _check_n_channels


def permute_soft(soft, tforms, board_size):
    """按每个样本的对称变换重排软策略（362 维）向量。

    ⚠ **这是软标签接入里唯一会「静默出错」的地方**。
    `sample_batch_numpy` 会对 `states` 施加 8 种对称增强、对 `moves_out` 施加
    `SYMMETRIES[t]` 重映射；若软 policy 不同步重排，训练**不会报任何错**，
    只是标签指向错误的点 —— 表现为「loss 正常下降、指标好看，但棋力不涨」。

    约定：
      · 前 `board_size**2` 项是落点，扁平下标 `r * board_size + c`；
      · 最后 1 项是 **pass**（下标 `board_size**2`），**任何变换都不动它**。
        这与 `sample_batch_numpy` 里 `moves_out` 对 pass 的处理一致
        （L147：默认填 `bs*bs`，只对 `0 <= mv < bs*bs` 做重映射）。

    参数
    ----
    soft   : (B, A) float；A = board_size**2 + 1
    tforms : (B,) int，取值 0..7（0 = 恒等）
    返回   : (B, A) float（float64 计算后转回原 dtype）
    """
    bs = board_size
    A = bs * bs + 1
    soft = np.asarray(soft)
    if soft.ndim != 2 or soft.shape[1] != A:
        raise ValueError(f"permute_soft: soft 应为 (B,{A})，实得 {soft.shape}")
    tforms = np.asarray(tforms)
    if tforms.shape[0] != soft.shape[0]:
        raise ValueError(f"permute_soft: tforms 长度 {tforms.shape[0]} "
                         f"与 soft 批 {soft.shape[0]} 不符")

    # 8 种变换的**逆置换**表：inv[t][j] = 「目标位 j 该从哪个源位取值」。
    #
    # ⚠ 方向极易搞反，这里是踩过的坑：若直接写
    #       out[:, perms[t]] = soft[:, :]
    #   或 `out = soft[:, perms[t]]`（gather），得到的都是**逆变换**——
    #   因为 `out[j] = soft[perms[j]]` 意味着「源 perms[j] 移到 j」，
    #   等价于把每个源点送到了 perms 的**像**的反方向。
    #   正解是显式求逆：perms[t] 是「源→目标」，那么目标 j 的来源是
    #   perms[t] 的逆映射，即 inv[perms[t][k]] = k。
    rr_all, cc_all = np.divmod(np.arange(bs * bs), bs)   # 源点 (r,c)
    invs = []
    for t in range(8):
        tr, tc = SYMMETRIES[t](rr_all, cc_all, bs)
        fwd = tr * bs + tc                             # 源 k -> 目标 fwd[k]
        inv = np.empty(bs * bs, dtype=np.int64)
        inv[fwd] = np.arange(bs * bs)                  # 目标 j <- 源 inv[j]
        invs.append(inv)

    out = np.zeros_like(soft, dtype=np.float64)
    out[:, A - 1] = soft[:, A - 1]          # pass 恒等
    for t in range(8):
        m = tforms == t
        if not m.any():
            continue
        out[m, :A - 1] = soft[m][:, invs[t]]   # 目标位取来源位
    return out.astype(soft.dtype)


class SupervisedDataset:
    def __init__(self, data: dict, n_channels: int = 12,
                 soft_idx=None, soft_policy=None):
        """
        data 必须含：boards(int8), my_hist(int16), op_hist(int16),
                       ko(int16), moves(int16), values(int8), to_play(int8)
        可选含：game_ids(int32) — 每个样本所属棋局 ID，用于 stratified split。
               winrates(float32) — 连续胜率标签（手数比例软标签 D），优先于 values。
               game_weights(float32) — 每样本的游戏权重（SGF 元数据加权），用于加权采样。
        均为 shape=(N, ...) 的 numpy 数组，N 相同。

        n_channels: 特征平面通道数，取 12..17 的前缀，**默认 12**。
            - `12`：P4.3 之前的布局。旧调用点与旧数据因此**一个字节都不变**，
              连 `feature_planes_batched` 的成本都不变（尾部 5 格直接不算，
              不是「算了再切」—— 这条路径在 MCTS 每个叶子展开时都调）。
            - `17`：通道表全表（已退役 v21 代 stem `Conv3×3(17→184)` 用过的布局；
              现役无模型消费这一档，白名单仍放行）。新增的 5 格
              （己方/对方眼位、双方块气=2、劫禁点掩码）**全部**从下面已有的
              `boards` / `to_play` / `ko` 三列现算，npz **不需要**新增列
              （键盘点见 P4.3 report）。
        ⚬ 通道数必须在**数据侧与模型侧同时**改（模型侧 = `scripts/train_sft.py`
          的 `KATAGO_SE_CFG['in_channels']`，现为 12），
          只改一侧得到的是形状错或静默错值，不是零回归。
        """
        _check_n_channels(n_channels)
        self.n_channels = n_channels
        self.boards = data['boards']
        self.my_hist = data['my_hist']
        self.op_hist = data['op_hist']
        self.ko = data['ko']
        self.moves = data['moves']
        self.values = data['values']
        self.to_play = data['to_play']
        self.game_ids = data.get('game_ids', None)
        self.winrates = data.get('winrates', None)  # 可选：连续胜率
        self.game_weights = data.get('game_weights', None)  # 可选：游戏权重
        self.N = self.boards.shape[0]
        self.board_size = self.boards.shape[1]
        self._board = GoBoard(self.board_size)  # 复用实例，避免重复分配

        # ---- 软标签（KataGo 搜索分布）——可选，**默认 None = 零回归** ----
        # `soft_idx` 是行下标数组 (M,)，`soft_policy` 是对应的 (M, A) 分布。
        # 两者由调用方（train_sft）用 pos_hash join 算好后传入；本类不自己算 hash
        # （3420 万行算一遍要 ~2 分钟，属于启动期一次性开销，交给脚本层更合适）。
        self.soft_idx = None
        self.soft_policy = None
        self.soft_row = None      # int32 (N,) 行->soft 槽位，-1 = 该行无标签
        self.n_soft = 0
        if soft_idx is not None and soft_policy is not None:
            self._attach_soft(soft_idx, soft_policy)

    def _attach_soft(self, soft_idx, soft_policy):
        """挂载软标签并建行→槽位映射。"""
        A = self.board_size * self.board_size + 1
        soft_idx = np.asarray(soft_idx, dtype=np.int64).ravel()
        sp = np.asarray(soft_policy)
        if sp.ndim != 2 or sp.shape[1] != A:
            raise ValueError(f"soft_policy 应为 (M,{A})，实得 {sp.shape}")
        if sp.shape[0] != soft_idx.size:
            raise ValueError("soft_idx 与 soft_policy 行数不一致")
        if soft_idx.size and (soft_idx.min() < 0 or soft_idx.max() >= self.N):
            raise ValueError("soft_idx 越界（数据集行号）")
        # 同一行可能被多份标签命中（实测 800 标签覆盖 995 行，因为同局内
        # 相同局面在 npz 里有多行）。此时**保留第一条**并计数，不报错——
        # 同一局面的 KataGo 分布本应一致，重复只是同一事实的多次记录。
        row = np.full(self.N, -1, dtype=np.int32)
        n_dup = 0
        for k, r in enumerate(soft_idx):
            if row[r] >= 0:
                n_dup += 1
                continue
            row[r] = k
        self.soft_idx = soft_idx
        self.soft_policy = sp
        self.soft_row = row
        self.n_soft = int((row >= 0).sum())
        self.n_soft_dup = n_dup

    def __len__(self):
        return self.N

    def sample_batch_numpy(self, idxs, rng=None, augment=True, labels=False):
        """
        给定样本下标，返回 numpy 版 (states, moves_out, values)：
            states    : (B, n_channels, H, W) float32
            moves_out : (B,) int64
            values    : (B, 1) float32

        与 sample_batch 逻辑完全相同（含向量化特征构造与 8 对称增强），只是不转
        torch 张量——供 MindSpore 训练脚本使用（910B 环境无 torch）。
        使用向量化批量特征构造（GoBoard.feature_planes_batched）+ 向量化对称增强，
        避免逐样本 Python 循环，训练吞吐显著更高。
        `n_channels` 由构造参数决定（默认 12），`states` 的 C 轴恒等于它。

        rng: 可选随机源，供多线程预取时各线程使用独立 Generator，避免竞争全局
            np.random。为 None 时沿用全局 np.random（保持原行为）。**仅在
            `augment=True` 时被读取**。

        augment: 是否施加 8 路随机对称增强，**默认 True = 训练路径**（`sample_batch`、
            `_BatchPrefetcher` 的 worker、`train_sft_ms.py` 都靠这个默认值吃增强，行为
            逐位不变）。`False` = **评估路径**（`train_sft.py` 的 `evaluate_metrics` /
            `evaluate_top1`）：验证集不该被随机翻转/旋转 —— 推理时不会出现随机翻转的
            棋盘，那层方差与真实输入分布不符，还会把「对称等价但标签已同步」的一致性
            混进 top-1 的分母里。`augment=False` 的精确语义是：
              ① **完全不抽 `tforms`**（连 RNG 都不碰：`rng` 既不被读、传入的 Generator
                 也不被推进，全局 np.random 亦然）→ 评估指标与采样种子彻底无关；
              ② `states` 直接返回 `feature_planes_batched(...)` 的原始输出，不做
                 `out = np.empty_like` 那一遍拷贝/变换；
              ③ `moves` **不做 `SYMMETRIES` 重映射**，但**保留**非法/越界 → `bs*bs` 的
                 标签规范化 —— pass 是动作空间里的第 `bs*bs` 类，这个定义必须与
                 「要不要翻转棋盘」解耦，否则 eval 与训练对 pass 的理解就不一样了。
        """
        bs = self.board_size
        B = len(idxs)
        boards = self.boards[idxs]
        my_h = self.my_hist[idxs]
        op_h = self.op_hist[idxs]
        ko = self.ko[idxs]
        to_play = self.to_play[idxs]
        tforms = None
        if augment:
            if rng is None:
                tforms = np.random.randint(0, 8, size=B)
            else:
                tforms = rng.integers(0, 8, size=B)  # Generator 用 integers，非 randint

        # 批量构造特征（B, n_channels, H, W）；`n_channels` 直接下传而不是
        # 「先造 17 通道再切前 12」—— 12 通道路径要连成本都零回归（MCTS 叶子
        # 展开与训练预取都调这条路径）。
        states = GoBoard.feature_planes_batched(boards, my_h, op_h, to_play, ko,
                                                 n_channels=self.n_channels)

        if augment:
            # 向量化对称增强：8 种变换（4 旋转 × 2 镜像），与 SYMMETRIES 坐标变换严格对齐。
            # np.rot90 是逆时针旋转，SYMMETRIES 是顺时针，故用 k=-k；
            # _FLIP 翻转列 (W/axis=3)，不是行 (H/axis=2)；
            # _FLIP_ROTxx = _ROTxx ∘ _FLIP，即先翻转再旋转。
            # 注意: np.fliplr 对2D数组翻转 axis=1(W)是正确的，但对4D数组翻转 axis=2(H)
            # 是错误的——此处用显式切片 [:, :, :, ::-1] 精确指定 W 轴。
            # 优化：一次性分配输出数组，避免重复分配
            out = np.empty_like(states)
            for t in range(8):
                mask = tforms == t
                if not mask.any():
                    continue
                arr = states[mask].copy()  # copy 避免视图问题
                if t >= 4:
                    arr = arr[:, :, :, ::-1]            # flip W (axis=3)
                k = t % 4
                if k > 0:
                    arr = np.rot90(arr, k=-k, axes=(2, 3))  # CW rotation
                out[mask] = arr
            states = out
        # augment=False：states 直接就是 feature_planes_batched 的原始输出
        # （恒等变换，连 out 缓冲都不分配）

        # 向量化坐标对称变换（move）
        moves_out = np.full(B, bs * bs, dtype=np.int64)  # 默认 pass/越界 -> 专用类别
        mv = np.asarray(self.moves[idxs], dtype=np.int64)
        valid = (mv >= 0) & (mv < bs * bs)
        if augment:
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
        else:
            # 不做 SYMMETRIES 重映射，但保留上面的非法/越界 -> bs*bs 规范化：
            # pass 标签的语义与「要不要翻转棋盘」无关，两条路径必须一致
            moves_out[valid] = mv[valid]

        values = self.values[idxs].astype(np.float32).reshape(-1, 1)
        # 优先使用 winrates（连续胜率），缺失则回退到 values
        if self.winrates is not None:
            values = self.winrates[idxs].astype(np.float32).reshape(-1, 1)

        # ---- 软标签分支：契约保持 ----
        # `labels=False`（默认）→ 仍返回**三元组**，既有调用点逐位不变。
        # `labels=True` 且挂了软标签 → 返回五元组，追加 (soft, mask)。
        if not labels or self.soft_row is None:
            return states, moves_out, values

        B = len(idxs)
        A = self.board_size * self.board_size + 1
        soft = np.zeros((B, A), dtype=np.float32)
        mask = np.zeros((B,), dtype=np.float32)
        slots = self.soft_row[idxs]
        hit = slots >= 0
        if hit.any():
            mask[hit] = 1.0
            soft[hit] = self.soft_policy[slots[hit]].astype(np.float32)
            # ⚠ **软标签必须与 states 用同一个变换重排**。漏了这步不报错，
            #   只会让标签指向错误的点（loss 照降、棋力不涨）。
            #   上面 states 走的是 `tforms`（augment=False 时恒等），故：
            if augment:
                soft = permute_soft(soft, tforms, self.board_size)
        return states, moves_out, values, soft, mask

    def sample_batch(self, idxs, device='cpu', labels=False):
        """numpy 取批 + 转 torch 张量（等价于 sample_batch_numpy 后再 to(device)）。"""
        out = self.sample_batch_numpy(idxs, labels=labels)
        dev_prefix = device.split(':')[0] if isinstance(device, str) else str(device)
        pinned = (dev_prefix in ('cuda', 'npu'))
        conv = ([lambda a: torch.from_numpy(a).pin_memory().to(device, non_blocking=True)]
                if pinned else
                [lambda a: torch.from_numpy(a).to(device)])
        if labels and self.soft_row is not None:
            states, moves_out, values, soft, mask = out
            return (conv[0](states), conv[0](moves_out), conv[0](values),
                    conv[0](soft), conv[0](mask))
        states, moves_out, values = out
        return conv[0](states), conv[0](moves_out), conv[0](values)
