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


def permute_move_vector(moves, tforms, board_size):
    """按每个样本的对称变换重排**一维**着法向量（就地修改并返回）。

    与 `permute_soft` 是同一套 `SYMMETRIES` 约定的两个入口，判据完全一致：
    **只有 `0 <= mv < board_size**2` 的落点被重映射**；pass / 越界 / 本仓的
    `next_move` 用的 −1 哨兵**一律原地不动**（这与 `sample_batch_numpy` 里
    `moves_out` 对 pass 的处理逐字一致：默认填 `bs*bs`，只对
    `0 <= mv < bs*bs` 做重映射）。

    ⚠ 为什么 `next_move` 也必须走这里：`states` 施加增强时，若只重映射 `moves`
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
        给定样本下标，返回 numpy 版批：
            labels=False（默认）→ **三元组** (states, moves_out, values)
            labels=True          → **4 元组** (states, moves_out, values, labels_dict)
        states    : (B, n_channels, H, W) float32
        moves_out : (B,) int64
        values    : (B, 1) float32
        labels_dict : dict（见 `_build_labels` 的 docstring，键与形状都在那里钉死）

        ⚠ **为什么 `labels=True` 是 4 元组 + dict、不是 5 元组**（spec §5.5）：
        旧形状 `(states, moves, values, soft, mask)` 里**没有 dict 的位置**，
        而 V7 的 loss 需要那个装满 `next_move`/`outcome`/`score`/`sb_*`/`global`/
        `ownership`/`scoring`/`seki`/`future`/`w{...}`/`game_weight` 的 dict ——
        继续下去只会变成 6 元组、或 fork 两条路。4 元组让预取器只需搬运**一种**
        payload 形状。

        ⚠ **软标签挂不挂，返回的元组长度都一样（都是 4 元组）**：未挂载时
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
           `scoring` / `seki` / `future` —— **待 D0 sidecar（B 组）接线**。
           占位用 `np.zeros`（而非 `None`）是为了让预取器只搬运**一种** payload
           形状（A3）；它们恒 0 的语义与 KataGo「权重 0 = 本行无此标签」一致。
        3. 软标签未挂载时 `soft`/`soft_mask` 全 0（=「本批没有任何软标签」），
           **不退回 3 元组**——形状随数据变化会让预取器的 payload 分叉。

        形状（bs = board_size，B = len(idxs)）
        --------------------------
        ``next_move`` (B,) int64 · ``outcome`` / ``outcome_black`` (B,) int64 ·
        ``game_weight`` (B,) f32 · ``soft`` (B, bs²+1) f32 ·
        ``soft_mask`` (B,) f32 · ``w`` dict[str → (B,) f32] ·
        ``future`` (B, 2, bs²) f32（19 路时即 (B,2,361)，与 V7 对齐）·
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
        # ⚠ 守卫不可省：数据里存在 **game_id 残行**（§5.3 的哈希锚点法对 id 复用
        #   免疫，但残行仍在），相邻两行**可能不是同一局**，不守卫就是把上一局的
        #   收官手当成 π_opp —— 又是一个不报错的静默错标签。
        nxt = idxs + 1
        has_next = nxt < self.N
        nxt_safe = np.where(has_next, nxt, idxs)      # 越界行退回自身，只为安全 gather
        if self.game_ids is not None:
            gi = np.asarray(self.game_ids)
            same_game = gi[nxt_safe] == gi[idxs]
        else:
            # ⚠ 无 game_ids ⇒ **无法验证同局**，一律不给 next_move（而不是猜
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
        # to_play=-1（白在走）时两个结论互换。⚠ 不能写 `outcome * to_play`：
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
                # ⚠ **软标签必须与 states 用同一个变换重排**。漏了这步不报错，
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
            'futurepos': np.zeros(B, dtype=np.float32),
        }

        def z(*shape):
            """占位标签的统一构造（全 0 float32）。"""
            return np.zeros(shape, dtype=np.float32)

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
            # futurepos = 落子后 +8 / +32 手的盘面（2 通道）；待 B6 邻行 gather 接线
            'future': z(B, 2, bs * bs),
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
