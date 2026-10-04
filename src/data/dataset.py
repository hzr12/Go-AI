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

import os

import numpy as np
import torch

from src.data.feature_v7_gather import FUTUREPOS_OFFSETS, gather_neighbors
from src.game.go_rules import GoBoard, SYMMETRIES, _check_n_channels


#: ``future``（futurepos 标签）里**不可用行**的哨兵值。
#:
#: 为什么不能是 0：0/1 恰好是「对手一颗子都不占」这个**完全合法的标签**，
#: 所以「盘面丢了」与「对手没占点」在纯 {0,1} 下不可区分，且不可逆
#: （这与 `src/data/pos_hash.py` 警告的「假信号」是同一类病）。
#: `-1.0` 落在标签域 {0,1} 之外 ⇒ ① 不是「无占用」；② 也不是「对方占满」
#: （那是全 1，同样看起来无害）；③ 任何 0.5 阈值的消费方都不会把它判成有效占用。
#: 第四道保险是 `w['futurepos_h*']` = 0：哨兵管「看到了什么」，权重管「算不算」。
FUTUREPOS_SENTINEL = np.float32(-1.0)


def permute_soft(soft, tforms, board_size):
    """按每个样本的对称变换重排软策略（362 维）向量。

     **这是软标签接入里唯一会「静默出错」的地方**。
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
    # 方向极易搞反，这里是踩过的坑：若直接写
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


def _permute_future(future, tforms, board_size):
    """按每个样本的对称变换重排 ``future`` 的最后一维（(B,2,bs²) 扁平占用图）。

     **为什么复用 `permute_soft` 而不是另写一份置换**：置换方向极易搞反
    （见 `permute_soft` 上面那段踩坑注释），而 `future` 是**标签**——方向错了
    不报错，只是 loss 照降、棋力不涨。本文件里唯一已被
    `tests/test_dataset_soft_labels.py::test_soft_argmax_agrees_with_moves_out_after_augment`
    钉住方向的置换实现就是 `permute_soft`，所以这里**不加第二份**。

    `permute_soft` 的契约是 (B, A=n²+1) 且**最后一格是 pass、任何变换都不动**。
    `future` 没有 pass 格，于是补一个恒 0 的哑格调用它、再把哑格丢掉 —— 哑格恒 0
    所以「不动它」这条规则对它无影响。顺带白拿 `permute_soft` 的 float64 中间量
    语义（先 float64 算再转回原 dtype）。

    两个 horizon 走**同一个** `tforms`（前提见 `attach_futurepos` 的「对称增广」段）。
    """
    f = np.asarray(future)
    if f.ndim != 3 or f.shape[1] != 2 or f.shape[2] != board_size * board_size:
        raise ValueError(f'_permute_future: 应为 (B,2,{board_size ** 2})，'
                         f'实得 {f.shape}')
    tforms = np.asarray(tforms)
    if tforms.shape[0] != f.shape[0]:
        raise ValueError(f'_permute_future: tforms 长度 {tforms.shape[0]} 与批 '
                         f'{f.shape[0]} 不符')
    bs2 = board_size * board_size
    # (B,2,bs²) -> (B*2, bs²+1)：哑格恒 0
    padded = np.zeros((f.shape[0] * 2, bs2 + 1), dtype=f.dtype)
    padded[:, :bs2] = f.reshape(f.shape[0] * 2, bs2)
    out = permute_soft(padded, np.repeat(tforms, 2), board_size)
    return out[:, :bs2].reshape(f.shape)


def permute_move_vector(moves, tforms, board_size):
    """按每个样本的对称变换重排**一维**着法向量（就地修改并返回）。

    与 `permute_soft` 是同一套 `SYMMETRIES` 约定的两个入口，判据完全一致：
    **只有 `0 <= mv < board_size**2` 的落点被重映射**；pass / 越界 / 本仓的
    `next_move` 用的 −1 哨兵**一律原地不动**（这与 `sample_batch_numpy` 里
    `moves_out` 对 pass 的处理逐字一致：默认填 `bs*bs`，只对
    `0 <= mv < bs*bs` 做重映射）。

     为什么 `next_move` 也必须走这里：`states` 施加增强时，若只重映射 `moves`
    而漏掉 `next_move`，policy_opp 的标签就会指向被翻转前的点 —— 与
    `permute_soft` 同类的静默错误（loss 照降、棋力不涨）。

    参数
    ----
    moves  : (B,) int64，**就地修改**（调用方若需原值须自行 copy）
    tforms : (B,) int，取值 0..7（0 = 恒等）
    """
    bs = board_size
    moves = np.asarray(moves)
    if moves.shape != np.asarray(tforms).shape:
        raise ValueError(f"permute_move_vector: moves {moves.shape} "
                         f"与 tforms {np.asarray(tforms).shape} 形状不符")
    valid = (moves >= 0) & (moves < bs * bs)
    if valid.any():
        r, c = np.divmod(np.where(valid, moves, 0), bs)
        for t in range(8):
            vmask = (tforms == t) & valid
            if not vmask.any():
                continue
            rr, cc = SYMMETRIES[t](r[vmask], c[vmask], bs)
            moves[vmask] = rr * bs + cc
    return moves


class SupervisedDataset:
    def __init__(self, data: dict, n_channels: int = 12,
                 soft_idx=None, soft_policy=None):
        """
        data 必须含：boards(int8), my_hist(int16), op_hist(int16),
                       ko(int16), moves(int16), values(int8), to_play(int8)
        可选含：game_ids(int32) — 每个样本所属棋局 ID，用于 stratified split。
               winrates(float32) — 连续胜率标签（手数比例软标签 D），优先于 values。
               game_weights(float32) — 每样本的游戏权重（来自 SGF 元数据加权）。
               **不是采样概率**：它一路作为标签传给 loss 的 `game_weight`，
               由 `katago_v7_loss._weighted_mean` 逐样本加权求和；
               本仓库没有任何 `.choice(p=...)` 消费它。
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
         通道数必须在**数据侧与模型侧同时**改（模型侧 = `scripts/train_sft.py`
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

        # ---- futurepos（未来 2 手位置）——**可选，默认关闭 = 逐字节零回归** ----
        # 关闭的理由**不是**设计偏好，而是两个既有测试钉死的：
        # `tests/test_dataset_soft_labels.py::test_placeholder_labels_are_zeros_
        # with_zero_weights` 在**默认构造**的 dataset 上断言 `w['futurepos'].sum()
        # == 0`，`tests/test_prefetch_labels.py::test_future_placeholder_stays_
        # zeros_through_the_prefetcher` 断言 `future` 全零。 **不要**为了「让
        # futurepos 默认生效」去改那两条测试——它们钉的是 A6 之前的历史契约。
        # `self._fp` 见 `attach_futurepos`；None = 未启用。
        self._fp = None

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

    def attach_soft(self, soft_idx, soft_policy):
        """公开入口：数据集**建好之后**再挂软标签（`--soft-index` 走这里）。

        为什么需要它：`_attach_soft` 是构造期路径，而 CLI 是在 `load_from_path`
        返回之后才读 `--soft-index` 的。让脚本去调私有方法会把 `_` 前缀的
        约定撕开，于是「构造器传参」与「CLI 传参」两条路各自漂移。

        与构造器传参**完全等价**（同一个 `_attach_soft`），可重复调用（后一次
        覆盖前一次）。
        """
        self._attach_soft(soft_idx, soft_policy)
        return {'n_soft': self.n_soft, 'n_soft_dup': self.n_soft_dup,
                'n_rows': self.N}

    def soft_row_mask(self):
        """``(N,) bool``：该行是否有软标签（`--soft-only-sampling` 收窄行空间用）。

         没挂软标签时返回**全 False**（而不是全 True）—— 收窄到 0 行必须
        让调用方立刻发现，而不是悄悄退回「全量采样」。
        """
        if self.soft_row is None:
            return np.zeros(self.N, dtype=bool)
        return self.soft_row >= 0

    # ---- futurepos（未来 2 手位置）----------------------------------------
    #
    # 语义
    # ----
    # ``future[b, h, r*bs+c] ∈ {0,1}`` = 「点 (r,c) 上有**行 i 时该走子那一方的
    # 对手**的子」，其中 h=0 取第 ``i+8`` 行盘面、h=1 取第 ``i+32`` 行盘面
    # （``FUTUREPOS_OFFSETS = (8, 32)``，主数据集每局约 316 行，8/32 是实测选出
    # 的「足够远」间隔）。
    #
    # 「对手」按**行 i 的 to_play** 定，不是按未来行的 to_play：i+1、i+2 轮到的
    # 正是 ``-to_play[i]``。⇒ 两个 horizon 天然共用同一套「谁是对手」口径。
    #
    # 权重契约
    # --------
    # =============================  ==========================================
    # 键                            形状 / 含义
    # =============================  ==========================================
    # ``future``                    (B,2,bs²) f32；h 有效 ⇒ {0,1}，否则全 -1
    # ``w['futurepos']``            (B,)；**两路都有效**才 1.0
    # ``w['futurepos_h0']``         (B,)；``i+8`` 那一路有效才 1.0
    # ``w['futurepos_h1']``         (B,)；``i+32`` 那一路有效才 1.0
    # =============================  ==========================================
    #
    # **为什么 `w['futurepos']` 是「与」而不是「或」**：
    # `src/networks/katago_v7_loss.py:362-368` 把 `future` reshape 成 (b,2,bs²)
    # 之后压成**一个逐样本标量**，再乘**一个**权重；`_weighted_mean` 又是
    # `(per_sample * weight).mean()`（**刻意不除 Σw**）。⇒ 权重的最小作用单位是
    # 「整块 2×bs²」，不是单路。只活一路时若给 w=1，那一路的 -1 哨兵会被当成
    # 真值去拟合 tanh ⇒ 头学出一个恒 -0.76 的假平面。所以单路有效性只能由
    # `w['futurepos_h*']` 表达，整块权重必须为 0。
    #
    # 不可用行的哨兵
    # --------------
    # ``future[b, h, :] = FUTUREPOS_SENTINEL``（-1.0），**不是** 0。见
    # :data:`FUTUREPOS_SENTINEL` 的论证。
    #
    # 对称增广
    # --------
    # 两路 h 是两个**不同时间点、同一局面**的未来盘面，因此走**同一个**
    # `tforms`（镜像的是行 i 的盘面，两个未来时刻必须跟着一起镜像）。
    # 若将来两个 horizon 用了**不同**的 tforms，batch 内同一行就不存在
    # 「一个统一的镜像」了——`states` 旋转 90° 而 h1 没转，等于把 h1 当成
    # 「另一个镜像下的未来」在学，标签与输入不同源，且不报任何错。**所以两条
    # 路必须共用同一个 `tforms`，这条是契约不是实现细节。**
    def attach_futurepos(self, source=None, *, mode='live',
                         dataset_npz=None, materialized_dir=None,
                         offsets=FUTUREPOS_OFFSETS,
                         table=None, table_valid=None):
        """启用 futurepos 标签（**opt-in**，不调它就完全保持 A6 之前的占位行为）。

        与 ``attach_soft`` 是同一个「构造之后再挂」的约定：构造器签名不加参数，
        CLI 才能在拿到 dataset 之后再决定要不要开这个开关。

        Parameters
        ----------
        mode:
            - ``'live'``（默认，**段 1 训练用**）：每个 batch 走一次
              ``gather_neighbors``，实时取 ``i+8`` / ``i+32`` 的盘面。
            - ``'index'``（评估 / 自战 / 调用方已做过全量扫描）：读调用方预好的
              整表 ``table``，**完全不做邻行 gather**。

        source:
            ``'live'`` 模式下 ``boards`` 的来源，可为
            ``None``（用 ``self.boards``）/ ``.npy`` 路径 / memmap / ndarray。
             传 ``.npz`` 路径**会报错**：``gather_neighbors`` 的 ``_reject_npz``
            会拦下（mmap 对 zip 压缩成员无效，会静默退化成整份 12.3 GB 解压）。
        dataset_npz / materialized_dir:
            ``'live'`` 模式下「先落 .npy 再 mmap」的原料，见
            :meth:`_futurepos_boards`。
        table / table_valid:
            ``mode='index'`` 的两张表：``table`` (N,2,bs²) bool、
            ``table_valid`` (N,2) bool（缺省 = 全 True）。

        Returns
        -------
        ``dict``：本次启用是否改变了 payload（供启动期打日志）。
        """
        bs = self.board_size
        n_sq = bs * bs
        if mode not in ('live', 'index'):
            raise ValueError(f"futurepos mode 应为 'live'/'index'，实得 {mode!r}")
        offsets = tuple(int(o) for o in offsets)
        if len(offsets) != 2:
            raise ValueError(f'futurepos 恒为 2 路（future 的第 1 轴长 2），'
                             f'实得 {len(offsets)} 个偏移：{offsets}')
        if 0 in offsets:
            raise ValueError('futurepos 偏移不能是 0（那就是当前盘面，不是未来）')

        if mode == 'index':
            if table is None:
                raise ValueError("mode='index' 必须给 table（N,2,bs²）")
            tbl = np.asarray(table)
            if tbl.shape != (self.N, 2, n_sq):
                raise ValueError(f'table 应为 {(self.N, 2, n_sq)}，实得 {tbl.shape}')
            if table_valid is None:
                tv = np.ones((self.N, 2), dtype=bool)
            else:
                tv = np.asarray(table_valid, dtype=bool)
                if tv.shape != (self.N, 2):
                    raise ValueError(f'table_valid 应为 {(self.N, 2)}，实得 {tv.shape}')
            self._fp = {'mode': 'index', 'offsets': offsets,
                        'table': tbl, 'valid': tv, 'boards': None}
            return self.futurepos_status()

        self._fp = {'mode': 'live', 'offsets': offsets,
                    'source': source, 'dataset_npz': dataset_npz,
                    'materialized_dir': materialized_dir,
                    'boards': None}          # 惰性解析的缓存位
        return self.futurepos_status()

    def _futurepos_boards(self):
        """解析 ``boards`` 的 mmap 来源，**只做一次**，之后复用。

        解析优先级（命中即止）
        --------------------
        1. ``attach_futurepos(source=...)`` 给了路径/数组 ⇒ 直接用，**零磁盘写**；
        2. ``self.boards`` 本身是 memmap 或已在内存的 ndarray ⇒ 直接用，
           **零磁盘写**。fancy-index 只触碰被点到的 B 行，不会整列读；
        3. 给了 ``dataset_npz`` + ``materialized_dir`` ⇒ 调
           ``kata_label_join.materialize_dataset`` 落 ``.npy`` 再 mmap。

        第 3 步会往磁盘写什么、多大
        ----------------------------
        ``materialize_dataset(npz, dir, keys=('boards','to_play','ko','game_ids'))``：
        ``boards.npy`` = ``N × bs × bs`` B（int8，主数据集 34.2M × 19 × 19 =
        **12.3 GB**）、``to_play.npy`` 34 MB、``ko.npy`` 68 MB、``game_ids.npy``
        137 MB。落盘耗时实测全量约 72s（478K 行/s），**一次落盘、之后反复扫描
        都只按需分页**。

         **为什么这条路径绝不能靠惰性触发**
        ``sample_batch_numpy`` 是在 ``_prefetch_worker``（``mp.Process`` fork 出来
        的子进程）里调的。惰性解析若发生在 fork 之后，**每个 worker 各解析一次**
        ⇒ 12.3 GB × worker 数（还可能同时写同一个路径互相踩坏）。所以：
        :meth:`warm_futurepos` 由调用方在**父进程 fork 之前**显式调，本方法里的
        惰性解析只是「构造期就开了 futurepos 但忘了 warm」的兜底。
        """
        fp = self._fp
        if fp is None:
            raise RuntimeError('futurepos 未启用：请先调 attach_futurepos()')
        if fp['mode'] == 'index':
            raise RuntimeError("mode='index' 不做邻行 gather，不该走到 _futurepos_boards")
        if fp['boards'] is not None:
            return fp['boards']

        boards = None
        src = fp.get('source')
        if src is not None:
            if isinstance(src, (str, bytes, os.PathLike)):
                p = os.fspath(src)
                if p.endswith('.npz'):
                    raise TypeError(
                        f'futurepos source 不能是 .npz（{p}）：mmap 对 zip 压缩成员'
                        f'无效，`np.load(..., mmap_mode="r")` 不报错但会整份解压。'
                        f'先跑 `materialize_dataset(npz, dir)`，或直接传 boards.npy。')
                boards = np.load(p, mmap_mode='r')
            else:
                boards = src
        elif fp.get('dataset_npz') and fp.get('materialized_dir'):
            # 第 3 步：唯一会写磁盘的分支，且**只在调用方显式给了两个路径时**。
            from src.data.kata_label_join import materialize_dataset
            d = fp['materialized_dir']
            paths = materialize_dataset(fp['dataset_npz'], d,
                                        keys=('boards', 'to_play', 'ko', 'game_ids'))
            fp['materialized_paths'] = paths
            boards = np.load(paths['boards'], mmap_mode='r')
            # 三个小列一并换成 mmap 口径：它们只有几百 MB，但保持「同一份
            # 物化产物」比保持「同一次解压」更重要（口径漂移 = 静默错标签）。
            fp['cols'] = {k: np.load(paths[k], mmap_mode='r')
                          for k in ('to_play', 'ko', 'game_ids')}
        else:
            boards = self.boards       # 第 2 步：零磁盘写

        boards = np.asarray(boards)
        if boards.shape != self.boards.shape:
            raise ValueError(
                f'futurepos 的 boards 来源形状 {boards.shape} 与数据集 '
                f'{self.boards.shape} 不符（行数或盘面尺寸不同 ⇒ 偏移的语义变了）')
        fp['boards'] = boards
        return boards

    def warm_futurepos(self):
        """在**父进程**里强制解析 boards 来源（fork 预取 worker 之前调）。

        返回 source 描述 dict。幂等：已解析就直接返回缓存。
        """
        if self._fp is not None and self._fp['mode'] == 'live':
            self._futurepos_boards()
        return self.futurepos_status()

    def futurepos_status(self):
        """当前 futurepos 配置快照（启动期打日志用；不触发任何 IO）。"""
        if self._fp is None:
            return {'enabled': False}
        out = {'enabled': True, 'mode': self._fp['mode'],
               'offsets': self._fp['offsets'],
               'resolved': self._fp.get('boards') is not None}
        if self._fp['mode'] == 'live':
            out['source'] = ('self.boards' if not self._fp.get('source')
                             and not self._fp.get('dataset_npz')
                             else self._fp.get('source') or self._fp['dataset_npz'])
        return out

    def _futurepos_target(self, idxs, tforms=None):
        """算 ``(future, w_h0, w_h1)``；**每个偏移恰好一次** fancy-index。

        三个「不可用」情形都在这里收口，且它们都表现为 ``g.valid[offset]``
        的 False（``gather_neighbors`` 已把「越界」与「跨局」都算进同一个
        ``valid``）：越界（接近局尾）/ ``game_ids[j] != game_ids[i]``（跨局）。

         **不能单独消费 ``g.boards[offset]``**：它的不可用行被填 0，而 0 是
        「合法空盘」—— 会被读成「该点将来没人下」，也就是**把盘面丢了说成
        「对手一颗子都没占」**。所以先过 ``valid``，无效行写哨兵。
        """
        idxs = np.asarray(idxs, dtype=np.int64)
        fp = self._fp
        if fp is None:
            raise RuntimeError('futurepos 未启用：请先调 attach_futurepos()')
        B = idxs.size
        bs = self.board_size
        n_sq = bs * bs

        if fp['mode'] == 'index':
            occ = np.asarray(fp['table'][idxs], dtype=bool).reshape(B, 2, n_sq)
            hvalid = np.asarray(fp['valid'])[idxs].astype(bool).reshape(B, 2)
        else:
            cols = fp.get('cols')
            gid = cols['game_ids'] if cols is not None else self.game_ids
            if gid is None:
                # 无 game_ids ⇒ **无法验证同局**，一律不给标签（与 `next_move`
                # 同一口径，见 `_build_labels` 里那段说明）。`gather_neighbors` 在
                # 缺 game_ids 时只做越界检查（valid = in_range），会把别局的盘面
                # 当成这一手的未来 —— 那正是要防的串局，宁可一行都不给。
                # 连 boards 来源都不解析（连 gather 一起省掉）。
                hvalid = np.zeros((B, 2), dtype=bool)
                occ = np.zeros((B, 2, n_sq), dtype=bool)
            else:
                # 「对手」按**行 i** 的 to_play 定（i+1/i+2 轮到的正是 -to_play[i]），
                # 所以乘的是本行的 to_play，不是未来行的。
                to_play_i = np.asarray(self.to_play[idxs],
                                       dtype=np.int8).reshape(B, 1, 1)
                g = gather_neighbors(
                    self._futurepos_boards(), idxs, offsets=fp['offsets'],
                    # game_ids **必须给**：主数据集是 162,298 局首尾相接的一根
                    # 大数组、没有局边界标记，i 与 i+8/i+32 可以分属两局，跨局
                    # 取到的盘面**看起来完全合法**（它就是某个真实盘面）只是不属于
                    # 这一手 ⇒ 静默错标签。
                    game_ids=gid,
                    to_play=(cols['to_play'] if cols is not None else self.to_play),
                    ko=(cols['ko'] if cols is not None else self.ko))
                hvalid = np.stack([g.valid[off] for off in fp['offsets']]).T.astype(bool)
                occ = np.stack(
                    [(np.asarray(g.boards[off], dtype=np.int8) * to_play_i) < 0
                     for off in fp['offsets']],
                    axis=1).reshape(B, 2, n_sq)   # 对手的子 ⇒ 乘 to_play 后 < 0

        future = occ.astype(np.float32)
        # 哨兵**在置换之后**盖：增广按 tforms 把有效格的占用挪位置，随后一次性
        # 把无效行整块写成 -1，就不必去追「无效行的值被搬到哪去了」。
        if tforms is not None:
            future = _permute_future(future, tforms, bs)
        future[~hvalid] = FUTUREPOS_SENTINEL
        return future, hvalid[:, 0], hvalid[:, 1]

    def __len__(self):
        return self.N

    def sample_batch_numpy(self, idxs, rng=None, augment=True, labels=False):
        """
        给定样本下标，返回 numpy 版批：
            labels=False（默认）→ **三元组** (states, moves_out, values)
            labels=True          → **4 元组** (states, moves_out, values, labels_dict)
        states    : (B, n_channels, H, W) float32
        moves_out : (B,) int64
        values    : (B, 1) float32
        labels_dict : dict（见 `_build_labels` 的 docstring，键与形状都在那里钉死）

         **为什么 `labels=True` 是 4 元组 + dict、不是 5 元组**（spec §5.5）：
        旧形状 `(states, moves, values, soft, mask)` 里**没有 dict 的位置**，
        而 V7 的 loss 需要那个装满 `next_move`/`outcome`/`score`/`sb_*`/`global`/
        `ownership`/`scoring`/`seki`/`future`/`w{...}`/`game_weight` 的 dict ——
        继续下去只会变成 6 元组、或 fork 两条路。4 元组让预取器只需搬运**一种**
        payload 形状。

         **软标签挂不挂，返回的元组长度都一样（都是 4 元组）**：未挂载时
        `soft` 全 0、`soft_mask` 全 0（=「本批没有任何软标签」），而不是退回
        3 元组。理由同上：形状随数据分叉是 bug 的温床。

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
        moves_out[valid] = mv[valid]      # 非法/越界 -> bs*bs 规范化（两条路径都做）
        if augment:
            # 与 `permute_move_vector` 是同一段实现：states 施加增强时着法必须同步
            # 重映射，且只重映射 `0 <= mv < bs*bs`，pass/越界保留默认标签。
            permute_move_vector(moves_out, tforms, bs)

        values = self.values[idxs].astype(np.float32).reshape(-1, 1)
        # 优先使用 winrates（连续胜率），缺失则回退到 values
        if self.winrates is not None:
            values = self.winrates[idxs].astype(np.float32).reshape(-1, 1)

        # ---- 标签分支：契约见本方法 docstring 与 `_build_labels` ----
        # `labels=False`（默认）→ **三元组**，既有调用点（bench_train / train_sft /
        # train_sft_ms 的预取器）逐位不变。
        # `labels=True` → **4 元组** (states, moves_out, values, labels_dict)，
        # **无论有没有挂软标签都是 4 元组**（见 docstring 的「为什么不是 5 元组」）。
        if not labels:
            return states, moves_out, values
        return (states, moves_out, values,
                self._build_labels(idxs, tforms, augment))

    # ---- 标签构造：spec §5.5 的 `labels_dict` ---------------------------------
    def _build_labels(self, idxs, tforms=None, augment=False):
        """构造 `labels=True` 时返回的第 4 个元素（V7 loss 需要的那个 dict）。

        键的来源分三类，**不要混看**：

        1. **本轮真算**（现有 10 列 + 软标签就能算）：
           `next_move` / `outcome` / `outcome_black` / `game_weight` /
           `soft` / `soft_mask` / `w['policy']` / `w['policy_opp']`。
        2. **本轮占位**（恒 0，且对应 `w` 恒 0，消费方必须先看权重）：
           `score` / `sb_center` / `sb_upper` / `global` / `ownership` /
           `scoring` / `seki` —— **待 D0 sidecar（B 组）接线**。
           占位用 `np.zeros`（而非 `None`）是为了让预取器只搬运**一种** payload
           形状（A3）；它们恒 0 的语义与 KataGo「权重 0 = 本行无此标签」一致。
        3. **`future` 是 opt-in 的**：`attach_futurepos()` 调过 ⇒ 填真值
           （`w['futurepos']` / `w['futurepos_h0']` / `w['futurepos_h1']` 才出现）；
           没调过 ⇒ 仍是 `np.zeros` + `w['futurepos']` 恒 0，与 A6 之前**逐字节
           相同**。契约见 `attach_futurepos`。
        4. 软标签未挂载时 `soft`/`soft_mask` 全 0（=「本批没有任何软标签」），
           **不退回 3 元组**——形状随数据变化会让预取器的 payload 分叉。

        形状（bs = board_size，B = len(idxs)）
        --------------------------
        ``next_move`` (B,) int64 · ``outcome`` / ``outcome_black`` (B,) int64 ·
        ``game_weight`` (B,) f32 · ``soft`` (B, bs²+1) f32 ·
        ``soft_mask`` (B,) f32 · ``w`` dict[str → (B,) f32] ·
        ``future`` (B, 2, bs²) f32（19 路时即 (B,2,361)，与 V7 对齐）·
           取值域：有效格 ∈ {0,1}，**无效格恒为 -1.0**（`FUTUREPOS_SENTINEL`）；
          未启用 futurepos 时整块为 0。详见 `attach_futurepos` ·
        ``score``/``sb_center``/``sb_upper`` (B,) f32 · ``global`` (B,19) f32 ·
        ``ownership``/``scoring``/``seki`` (B,1,bs,bs) f32

        对称增强（spec §5.6）
        --------------------
        `states` / `soft` / `ownership` / `scoring` / `seki` / `future` 走**同一个**
        `tforms`；`moves` / `next_move` 按 `SYMMETRIES[t]` 映射（−1 哨兵不动）；
        `outcome` / `score` / `sb_*` / `global` / `w` / `soft_mask` **不变**
        （棋盘翻转不改变胜负与权重）。
        """
        bs = self.board_size
        B = len(idxs)
        A = bs * bs + 1
        idxs = np.asarray(idxs, dtype=np.int64)
        permuted = bool(augment) and tforms is not None

        # ---- next_move：由**下一行**的 moves 推出，跨局必须被守卫 ------------
        # 原设计 §5.2 的「行表新增 next_move 列」已作废（§5.8）：实时推。
        # 守卫不可省：数据里存在 **game_id 残行**（§5.3 的哈希锚点法对 id 复用
        #   免疫，但残行仍在），相邻两行**可能不是同一局**，不守卫就是把上一局的
        #   收官手当成 π_opp —— 又是一个不报错的静默错标签。
        nxt = idxs + 1
        has_next = nxt < self.N
        nxt_safe = np.where(has_next, nxt, idxs)      # 越界行退回自身，只为安全 gather
        if self.game_ids is not None:
            gi = np.asarray(self.game_ids)
            same_game = gi[nxt_safe] == gi[idxs]
        else:
            # 无 game_ids ⇒ **无法验证同局**，一律不给 next_move（而不是猜
            #   「相邻行就是同局」——那正是要防的串局）。代价是这些行的
            #   policy_opp 权重恒 0；生产数据一定带 game_ids。
            same_game = np.zeros(B, dtype=bool)
        mv_next = np.asarray(self.moves[nxt_safe], dtype=np.int64)
        usable = has_next & same_game & (mv_next >= 0) & (mv_next < bs * bs)
        next_move = np.full(B, -1, dtype=np.int64)
        next_move[usable] = mv_next[usable]
        if permuted:
            next_move = permute_move_vector(next_move, tforms, bs)

        # ---- outcome：三分类 {0 胜, 1 负, 2 无结果} ------------------------
        # 来源是 `winrates`（build_dataset 写的是 `tanh(value * to_play * alpha)`），
        # 因此 `sign(winrates[i])` **就是 to_play 视角**的结果。`winrates == 0`
        # 判「无结果」（无分差/和棋）；无 winrates 列时回退 `values`（同语义，
        # 只是被 round 成了 int8 ⇒ 弱位容易取到 0 而被判无结果）。
        wr_src = self.winrates if self.winrates is not None else self.values
        wr = np.asarray(wr_src[idxs], dtype=np.float32)
        outcome = np.where(wr > 0, 0, np.where(wr < 0, 1, 2)).astype(np.int64)
        # 黑方视角：outcome 是 to_play 视角，to_play=1 表示黑在走 ⇒ 直接就是黑方结果；
        # to_play=-1（白在走）时两个结论互换。 不能写 `outcome * to_play`：
        # outcome=0（to_play 胜）乘 -1 仍是 0，会把「白胜」也报成「黑胜」。
        to_play = np.asarray(self.to_play[idxs], dtype=np.int64)
        outcome_black = np.where(outcome == 2, 2,
                                 np.where(to_play == 1, outcome, 1 - outcome)
                                 ).astype(np.int64)

        # ---- 软标签（KataGo 访问分布）----------------------------------------
        # 未挂载（self.soft_row is None）时恒 0：soft_mask 全 0 即「本批无软标签」。
        soft = np.zeros((B, A), dtype=np.float32)
        soft_mask = np.zeros(B, dtype=np.float32)
        if self.soft_row is not None:
            slots = self.soft_row[idxs]
            hit = slots >= 0
            if hit.any():
                soft_mask[hit] = 1.0
                soft[hit] = self.soft_policy[slots[hit]].astype(np.float32)
                # **软标签必须与 states 用同一个变换重排**。漏了这步不报错，
                #   只会让标签指向错误的点（loss 照降、棋力不涨）。
                #   上面 states 走的是 `tforms`（augment=False 时恒等），故：
                if permuted:
                    soft = permute_soft(soft, tforms, self.board_size)

        # ---- 局级权重：game_weights 列（SGF 元数据加权），缺列时恒 1 ----------
        if self.game_weights is None:
            game_weight = np.ones(B, dtype=np.float32)
        else:
            game_weight = np.asarray(self.game_weights[idxs],
                                     dtype=np.float32).reshape(B)

        w = {
            'policy': np.ones(B, dtype=np.float32),
            # π_opp = 下一手的 one-hot：局末手（或下一行是 pass/无效）→ 0（spec §4.6）
            'policy_opp': usable.astype(np.float32),
            # 下面这五个**恒 0**：对应的标签还是占位零值，消费方必须先看权重，
            # 否则会把「没有这个标签」当成「标签是 0」（待 D0 sidecar / B 组接线）
            'ownership': np.zeros(B, dtype=np.float32),
            'score': np.zeros(B, dtype=np.float32),
            'scoring': np.zeros(B, dtype=np.float32),
            'seki': np.zeros(B, dtype=np.float32),
            # 未启用 futurepos 时恒 0。「未启用」不是「碰巧没数据」：opt-in 是
            # `test_dataset_soft_labels.py:510` / `test_prefetch_labels.py:464`
            # 两条既有测试逼出来的（见 `__init__` 里的说明）。
            'futurepos': np.zeros(B, dtype=np.float32),
        }

        def z(*shape):
            """占位标签的统一构造（全 0 float32）。"""
            return np.zeros(shape, dtype=np.float32)

        # ---- futurepos（未来 2 手位置）：opt-in，未启用 ⇒ 保持 A6 之前的占位 ----
        # 未启用时**不 gather、不加 `w` 的新键** ⇒ payload 与今天逐字节相同。
        future = z(B, 2, bs * bs)
        if self._fp is not None:
            future, w_h0, w_h1 = self._futurepos_target(
                idxs, tforms if permuted else None)
            # 整块权重是**与**：loss 的作用单位是整个 (2, bs²) 块（见
            # `attach_futurepos` 的「权重契约」段），只活一路时给 1 会让另一路
            # 的 -1 哨兵被当真值拟合。
            w['futurepos'] = (w_h0 & w_h1).astype(np.float32)
            # 单路有效性**必须**单独暴露：否则「只活一路」与「两路都死」不可分。
            w['futurepos_h0'] = w_h0.astype(np.float32)
            w['futurepos_h1'] = w_h1.astype(np.float32)

        return {
            'next_move': next_move,
            'outcome': outcome,
            'outcome_black': outcome_black,
            'game_weight': game_weight,
            'soft': soft,
            'soft_mask': soft_mask,
            'w': w,
            # ---- 以下为占位（恒 0 + 对应权重恒 0），待 D0 sidecar / B 组接线 ----
            'score': z(B),
            'sb_center': z(B),
            'sb_upper': z(B),
            'global': z(B, 19),
            'ownership': z(B, 1, bs, bs),
            'scoring': z(B, 1, bs, bs),
            'seki': z(B, 1, bs, bs),
            # futurepos = 落子后 +8 / +32 手的**对手**占位图（2 通道）；
            # 启用后由 `attach_futurepos` 填真值，无效行填 FUTUREPOS_SENTINEL。
            'future': future,
        }

    def sample_batch(self, idxs, device='cpu', labels=False):
        """numpy 取批 + 转 torch 张量（等价于 sample_batch_numpy 后再 to(device)）。

        `labels=True` 时第 4 个元素是 `labels_dict`，其 numpy 值（含嵌套的 `w`）
        也一并转成同 device 的张量；`labels=False` 仍是三元组，形状与数值不变。
        """
        out = self.sample_batch_numpy(idxs, labels=labels)
        dev_prefix = device.split(':')[0] if isinstance(device, str) else str(device)
        pinned = (dev_prefix in ('cuda', 'npu'))
        conv = ([lambda a: torch.from_numpy(a).pin_memory().to(device, non_blocking=True)]
                if pinned else
                [lambda a: torch.from_numpy(a).to(device)])

        def _to_t(x):
            if isinstance(x, dict):        # labels_dict 里的 `w` 是嵌套 dict
                return {k: _to_t(v) for k, v in x.items()}
            return conv[0](x)

        if labels:
            states, moves_out, values, labels_dict = out
            return conv[0](states), conv[0](moves_out), conv[0](values), _to_t(labels_dict)
        states, moves_out, values = out
        return conv[0](states), conv[0](moves_out), conv[0](values)
