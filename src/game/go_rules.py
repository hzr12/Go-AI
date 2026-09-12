"""
围棋规则引擎（纯 Python + numpy）。

这是整个项目的规则真相来源（single source of truth）。
同时服务于：
  - 棋谱重放（SGF -> 状态张量）
  - 自我对弈（将来）
  - 评估

规则范围（按诊断结论取舍，聚焦监督学习可用的最小正确集）：
  - 提子（连通块气计算）
  - 打劫（单子劫禁着，标准 ko）
  - 禁自杀（落子后自身无气且未提子则非法）
  - pass（连续两次 pass 终局）
  - 中国规则计分：数子法（区域计分）+ 贴目 7.5
    （计分算法与 Tromp-Taylor 等价：空点归最近同色连通块；差异见 score() 注释）
未实现（监督学习不需要）：超级劫、积攒劫、多劫循环判定、终局死子人工判定。
"""

import numpy as np
from scipy.ndimage import label as _scipy_label

# scipy.ndimage.label 的 4 邻域 3D 结构元素（模块级常量，避免每次调用重建）
_STRUCT3 = np.zeros((3, 3, 3), dtype=bool)
_STRUCT3[1] = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


# 对称变换：8 种（4 旋转 × 2 翻转）。用于数据增强时的坐标重映射。
# 每个变换是一个函数 (r, c) -> (r, c)，作用在 (board_size, board_size) 的平面上。
_ROT0 = lambda r, c, n: (r, c)
_ROT90 = lambda r, c, n: (c, n - 1 - r)
_ROT180 = lambda r, c, n: (n - 1 - r, n - 1 - c)
_ROT270 = lambda r, c, n: (n - 1 - c, r)
_FLIP = lambda r, c, n: (r, n - 1 - c)
_FLIP_ROT90 = lambda r, c, n: (n - 1 - c, n - 1 - r)
_FLIP_ROT180 = lambda r, c, n: (n - 1 - r, c)
_FLIP_ROT270 = lambda r, c, n: (c, r)

SYMMETRIES = [
    _ROT0, _ROT90, _ROT180, _ROT270,
    _FLIP, _FLIP_ROT90, _FLIP_ROT180, _FLIP_ROT270,
]


def transform_coord(r: int, c: int, transform_id: int, board_size: int) -> int:
    """把 (r, c) 按 8 种对称之一变换，返回扁平坐标 r*board_size + c。"""
    r2, c2 = SYMMETRIES[transform_id % 8](r, c, board_size)
    return r2 * board_size + c2


class GoBoard:
    """围棋棋盘。内部棋盘取值：-1=白, 0=空, 1=黑。"""

    def __init__(self, board_size: int = 19, komi: float = 7.5):
        self.board_size = board_size
        self.komi = komi
        self.reset()

    def reset(self):
        n = self.board_size
        self.board = np.zeros((n, n), dtype=np.int8)
        self.current_player = 1  # 1=黑, -1=白
        self.ko_point = -1       # 打劫禁着点（扁平坐标），-1 表示无
        self.passes = 0          # 连续 pass 计数
        self.move_history = []   # 记录每步落子扁平坐标，pass 记为 -1
        self._undo_stack = []    # 撤销栈：每项 (move, captured|None, prev_ko, prev_passes, prev_player)

    def clone(self) -> "GoBoard":
        """轻量克隆：只复制推演所需状态（盘面/执子方/劫/连续 pass 计数），
        **不复制** _undo_stack 与 move_history。

        用于 MCTS 里大量局面推演。原实现用 copy.deepcopy，会把随手数线性增长的
        撤销栈与着法历史整份复制，是叶子批量展开的主要开销之一。克隆出的棋盘
        撤销栈为空，仍可正常 play/undo 自己的后续着法（推演不需要旧历史）。
        """
        nb = GoBoard.__new__(GoBoard)
        nb.board_size = self.board_size
        nb.komi = self.komi
        nb.board = self.board.copy()
        nb.current_player = self.current_player
        nb.ko_point = self.ko_point
        nb.passes = self.passes
        nb.move_history = []
        nb._undo_stack = []
        return nb

    # ---- 基础查询 ----------------------------------------------------------

    def __getitem__(self, idx):
        return self.board[idx]

    def is_on_board(self, r, c):
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def get_legal_moves(self, check_suicide: bool = False) -> np.ndarray:
        """返回长度为 size*size 的 bool 掩码，True 表示该点可落子。

        check_suicide=True 时额外过滤自杀手（需模拟落子，较慢但更准确）。
        """
        n = self.board_size
        legal = (self.board == 0).reshape(-1).copy()
        if self.ko_point >= 0:
            legal[self.ko_point] = False
        if check_suicide:
            color = self.current_player
            for i in range(n * n):
                if not legal[i]:
                    continue
                r, c = divmod(i, n)
                # 模拟落子检查是否为自杀
                self.board[r, c] = color
                has_liberty = self._group_has_liberty(r, c)
                # 检查是否提掉对手子（非自杀）
                if not has_liberty:
                    opponent = -color
                    for nb_r, nb_c in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                        if self.is_on_board(nb_r, nb_c) and self.board[nb_r, nb_c] == opponent:
                            if not self._group_has_liberty(nb_r, nb_c):
                                has_liberty = True
                                break
                self.board[r, c] = 0
                if not has_liberty:
                    legal[i] = False
        return legal

    # ---- 连通块 / 气 -------------------------------------------------------

    def _neighbor_groups(self, r, c):
        """返回 (r,c) 的 4 邻域内不同颜色的连通块列表。"""
        n = self.board_size
        groups = []
        seen = set()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not self.is_on_board(nr, nc):
                continue
            color = self.board[nr, nc]
            if color == 0:
                continue
            if (nr, nc) in seen:
                continue
            # flood fill 同色连通块
            stack = [(nr, nc)]
            comp = []
            seen.add((nr, nc))
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    gnr, gnc = cr + dr2, cc + dc2
                    if not self.is_on_board(gnr, gnc):
                        continue
                    if (gnr, gnc) in seen:
                        continue
                    if self.board[gnr, gnc] == color:
                        seen.add((gnr, gnc))
                        stack.append((gnr, gnc))
            groups.append((color, comp))
        return groups

    def _group_has_liberty(self, seed_r, seed_c) -> bool:
        """判断 (seed_r, seed_c) 所在连通块是否还有气。"""
        n = self.board_size
        color = self.board[seed_r, seed_c]
        stack = [(seed_r, seed_c)]
        seen = {(seed_r, seed_c)}
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not self.is_on_board(nr, nc):
                    continue
                v = self.board[nr, nc]
                if v == 0:
                    return True
                if v == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return False

    def _group_liberty_count(self, seed_r, seed_c) -> int:
        """返回 (seed_r, seed_c) 所在同色连通块的气数。"""
        n = self.board_size
        color = self.board[seed_r, seed_c]
        stack = [(seed_r, seed_c)]
        seen = {(seed_r, seed_c)}
        libs = set()
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not self.is_on_board(nr, nc):
                    continue
                v = self.board[nr, nc]
                if v == 0:
                    libs.add((nr, nc))
                elif v == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return len(libs)

    # ---- 落子 --------------------------------------------------------------

    def play(self, move: int, record: bool = True) -> bool:
        """
        落子。move 为扁平坐标 (0..size*size-1)，或 -1 表示 pass。
        返回是否成功（非法落子返回 False 且不改变状态）。

        record=True（默认）时把撤销信息压入 _undo_stack，可用 undo() 撤销
        本次落子（含提子恢复 / 劫点 / pass 计数 / 历史 / 执子方）。
        """
        n = self.board_size
        if move == -1:
            # pass
            if record:
                self._undo_stack.append((-1, None, self.ko_point, self.passes, self.current_player))
            self.passes += 1
            self.ko_point = -1
            self.move_history.append(-1)
            self.current_player = -self.current_player
            return True

        if move < 0 or move >= n * n:
            return False
        r, c = divmod(move, n)
        if self.board[r, c] != 0:
            return False
        if move == self.ko_point:
            return False

        color = self.current_player
        opponent = -color

        # 试落子
        self.board[r, c] = color
        # 提掉相邻 opponent 块中无气的
        captured = []
        for nb_color, comp in self._neighbor_groups(r, c):
            if nb_color == opponent:
                cr0, cc0 = comp[0]
                if self.board[cr0, cc0] == opponent and not self._group_has_liberty(cr0, cc0):
                    captured.extend(comp)

        # 检查自身是否还有气（禁自杀）
        if not captured and not self._group_has_liberty(r, c):
            self.board[r, c] = 0  # 撤销
            return False

        # 执行提子
        for (cr, cc) in captured:
            self.board[cr, cc] = 0

        # 压撤销信息（此时 ko/passes/player 尚未更新）
        if record:
            self._undo_stack.append(
                (move, captured, self.ko_point, self.passes, self.current_player))

        # 打劫判定：提掉恰好 1 子，且落子子本身恰好只剩 1 气（即被提点）-> 形成劫
        if len(captured) == 1 and self._group_liberty_count(r, c) == 1:
            self.ko_point = captured[0][0] * n + captured[0][1]
        else:
            self.ko_point = -1

        self.passes = 0
        self.move_history.append(move)
        self.current_player = -self.current_player
        return True

    def undo(self) -> bool:
        """撤销最近一次成功 play（须 play(record=True)）。

        完整恢复：棋盘子与被提子、劫禁着点、pass 计数、着法历史、执子方。
        返回是否成功（栈空返回 False）。
        """
        if not self._undo_stack:
            return False
        move, captured, ko, passes, player = self._undo_stack.pop()
        n = self.board_size
        if move != -1:
            r, c = divmod(move, n)
            self.board[r, c] = 0
            if captured:
                cap_color = -player  # 被提子为落子方对手
                for (cr, cc) in captured:
                    self.board[cr, cc] = cap_color
        self.move_history.pop()
        self.ko_point = ko
        self.passes = passes
        self.current_player = player
        return True

    # ---- 终局与计分 --------------------------------------------------------

    def is_game_over(self) -> bool:
        return self.passes >= 2

    def score(self) -> float:
        """
        中国规则（数子法）计分：区域计分，黑分 = 黑子 + 黑围空，白同理 + 贴目 7.5。
        返回黑方视角得分（>0 黑胜，<0 白胜）。

        算法与 Tromp-Taylor 计分等价（空点归仅与一种颜色相邻的空区域）。
        与正式中国规则的差异：正式规则终局需先提净死子再数子；本实现没有
        死子判定，因此对局双方应在 pass 认输前实际提掉对方的死子，否则死子
        所在点会被判为双方共邻的中立区域，不参与计分。
        """
        n = self.board_size
        board = self.board
        empty = (board == 0)
        # 每个空点归属：与相邻同色块判定的简化版本 —— 用 flood fill 连通空区域，
        # 区域若只与一种颜色相邻，则该区域归该颜色。
        visited = np.zeros((n, n), dtype=bool)
        black_territory = 0
        white_territory = 0
        for r in range(n):
            for c in range(n):
                if not empty[r, c] or visited[r, c]:
                    continue
                # flood fill 这片空区域
                stack = [(r, c)]
                region = []
                border_colors = set()
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    region.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if not self.is_on_board(nr, nc):
                            continue
                        if empty[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                        elif board[nr, nc] != 0:
                            border_colors.add(int(board[nr, nc]))
                if len(border_colors) == 1:
                    if 1 in border_colors:
                        black_territory += len(region)
                    elif -1 in border_colors:
                        white_territory += len(region)
                # 否则中立，不计

        black_stones = int((board == 1).sum())
        white_stones = int((board == -1).sum())
        black_score = black_stones + black_territory
        white_score = white_stones + white_territory + self.komi
        return float(black_score - white_score)

    def result(self) -> int:
        """返回 +1 黑胜, -1 白胜, 0 平（理论上贴目 6.5 不会平）。"""
        s = self.score()
        if s > 0:
            return 1
        elif s < 0:
            return -1
        return 0

    # ---- 文本 / 坐标辅助（供推理与交互使用）------------------------------

    def to_string(self, markers=None, last_move=None) -> str:
        """返回可读的棋盘字符串，'X'=黑 'O'=白 '.'=空。

        markers   : 可选 dict {(r,c): char} 在对应点叠加标记（如候选着法）
        last_move : 可选 (r,c) 用 '*' 标记上一手
        """
        n = self.board_size
        coord = " abcdefghijklmnopqrs"
        lines = [f"   {coord[1:n + 1]}"]
        for r in range(n):
            row = []
            for c in range(n):
                if markers and (r, c) in markers:
                    row.append(markers[(r, c)])
                elif last_move is not None and (r, c) == last_move:
                    row.append('*')
                else:
                    v = self.board[r, c]
                    row.append('X' if v == 1 else 'O' if v == -1 else '.')
            lines.append(f"{r + 1:2d} {' '.join(row)}")
        return "\n".join(lines)

    def parse_move_str(self, s: str, color=1):
        """把坐标字符串（如 'ce' 或 SGF 风格 'ce'）解析为整数 move。

        s 为避免歧义统一用小写字母 a-s。返回 (ok, move_int)；pass 返回 -1。
        """
        n = self.board_size
        s = s.strip().lower()
        if s in ("", "pass", "resign"):
            return (True, -1)
        if len(s) != 2:
            return (False, -1)
        c = ord(s[0]) - ord('a')
        r = ord(s[1]) - ord('a')
        if not (0 <= r < n and 0 <= c < n):
            return (False, -1)
        return (True, r * n + c)

    # ---- 特征平面（12 通道）------------------------------------------------
    #
    # 布局（当前执子方视角，to_play 为当前落子方 1/黑 -1/白）：
    #   0      : 己方棋子
    #   1..3   : 己方前 1/2/3 手落子
    #   4      : 对手棋子
    #   5..7   : 对手前 1/2/3 手落子
    #   8      : 合法点掩码（含劫禁着点已排除）
    #   9      : 执子方常数（to_play，±1）
    #   10     : 己方气数=1 的块掩码
    #   11     : 对手气数=1 的块掩码
    #
    # my_hist / op_hist: 长度均为 3 的扁平坐标序列（不足补 -1），最近一手在 index 0。

    def feature_planes(self, my_hist, op_hist, to_play=None):
        n = self.board_size
        if to_play is None:
            to_play = self.current_player
        planes = np.zeros((12, n, n), dtype=np.float32)
        opp = -to_play

        planes[0] = (self.board == to_play)
        planes[4] = (self.board == opp)

        for k, mv in enumerate(my_hist):
            if mv >= 0:
                r, c = divmod(mv, n)
                planes[1 + k][r, c] = 1.0
        for k, mv in enumerate(op_hist):
            if mv >= 0:
                r, c = divmod(mv, n)
                planes[5 + k][r, c] = 1.0

        planes[8] = self.get_legal_moves().reshape(n, n).astype(np.float32)
        planes[9] = float(to_play)

        # 气 = 1 掩码
        my_liberties1 = np.zeros((n, n), dtype=bool)
        op_liberties1 = np.zeros((n, n), dtype=bool)
        seen = np.zeros((n, n), dtype=bool)
        for r in range(n):
            for c in range(n):
                v = self.board[r, c]
                if v == 0 or seen[r, c]:
                    continue
                if self._group_liberty_count(r, c) == 1:
                    stack = [(r, c)]
                    seen[r, c] = True
                    while stack:
                        y, x = stack.pop()
                        if v == to_play:
                            my_liberties1[y, x] = True
                        else:
                            op_liberties1[y, x] = True
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ny, nx = y + dr, x + dc
                            if 0 <= ny < n and 0 <= nx < n and not seen[ny, nx] and self.board[ny, nx] == v:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
        planes[10] = my_liberties1
        planes[11] = op_liberties1
        return planes

    @staticmethod
    def feature_planes_batched(boards, my_hist, op_hist, to_play, ko=None):
        """向量化批量版 feature_planes，语义与单图 feature_planes 完全一致。

        输入:
            boards   : (B, n, n) int8，取值 -1/0/1
            my_hist  : (B, 3) int16，己方前 3 手扁平坐标（-1 填充）
            op_hist  : (B, 3) int16
            to_play  : (B,) int8，轮到谁落子（1 黑 / -1 白）
            ko       : (B,) int16，劫禁着点扁平坐标（-1 无）；可选，用于通道 8 排除
        返回: (B, 12, n, n) float32

        性能: 用 scipy.ndimage.label 一次性标注连通块并向量化计算气数，
            scipy 释放 GIL，两个颜色的标注线程可真正并行。
            multiprocessing prefetcher 提供跨 worker 的真正 CPU 并行。
        """
        boards = np.asarray(boards)
        B, n, _ = boards.shape
        planes = np.zeros((B, 12, n, n), dtype=np.float32)
        to_play = np.asarray(to_play).reshape(B, 1, 1)
        opp = -to_play  # (B,1,1)

        # 通道 0/4: 己方/对手棋子
        planes[:, 0] = (boards == to_play)
        planes[:, 4] = (boards == opp)

        # 通道 1-3 / 5-7: 历史手（向量化 scatter）
        my_hist = np.asarray(my_hist).reshape(B, 3)
        op_hist = np.asarray(op_hist).reshape(B, 3)
        
        for hist, ch_base in ((my_hist, 1), (op_hist, 5)):
            valid = hist >= 0 # (B, 3) bool
            if valid.any():
                # 【修复】获取所有有效位置的扁平化索引
                valid_flat_indices = np.flatnonzero(valid)
                
                # 【修复】根据扁平化索引，分别获取对应的 batch、channel 和棋盘坐标
                b_idx = valid_flat_indices // 3          # 对应的 batch 索引
                c_idx = valid_flat_indices % 3           # 对应的 history 索引 (0, 1, 2)
                r, c = np.divmod(hist[valid], n)         # 对应的棋盘坐标
                
                # 【修复】使用对齐后的索引进行赋值
                planes[b_idx, ch_base + c_idx, r, c] = 1.0

        # 通道 8: 合法点掩码（空点，劫禁着点排除）—— 与单图 get_legal_moves 一致
        legal = (boards == 0).astype(np.float32)
        if ko is not None:
            ko = np.asarray(ko).reshape(B)
            vk = ko >= 0
            if vk.any():
                # 向量化：只处理真正有劫的样本，取代逐样本 Python 循环
                bidx = np.nonzero(vk)[0]
                r, c = np.divmod(ko[bidx].astype(np.int64), n)
                legal[bidx, r, c] = 0.0
        planes[:, 8] = legal

        # 通道 9: 执子方常数
        planes[:, 9] = to_play.astype(np.float32)

        # 通道 10/11: 气数=1 掩码（整批向量化连通块标注 + 邻空计数）
        my_lib1 = np.zeros((B, n, n), dtype=np.float32)
        op_lib1 = np.zeros((B, n, n), dtype=np.float32)
        # 每个棋子点的 4 邻域空点坐标数（整数 0-4），整批一次算，与单图
        # _group_liberty_count 逐点计数语义一致（每个空邻域坐标各算 1 气）。
        empty = (boards == 0)
        neigh_empty = np.zeros((B, n, n), dtype=np.int8)
        neigh_empty[:, :-1, :] += empty[:, 1:, :]
        neigh_empty[:, 1:, :]  += empty[:, :-1, :]
        neigh_empty[:, :, :-1] += empty[:, :, 1:]
        neigh_empty[:, :, 1:]  += empty[:, :, :-1]

        # 并行标注：scipy.ndimage.label 释放 GIL，两个颜色的标注可真正并行
        import threading

        def _label_and_mark(mask, lib_plane, to_play_val):
            if not mask.any():
                return
            labelled, num = _scipy_label(mask, structure=_STRUCT3)
            if num == 0:
                return
            w = np.where(mask, neigh_empty, 0).ravel()
            lib_counts = np.bincount(labelled.ravel(), weights=w,
                                     minlength=num + 1).astype(np.int64)
            lib_plane[(lib_counts[labelled] == 1) & mask] = 1.0

        t_my = threading.Thread(target=_label_and_mark,
                                args=(boards == to_play, my_lib1, to_play))
        t_op = threading.Thread(target=_label_and_mark,
                                args=(boards == -to_play, op_lib1, -to_play))
        t_my.start()
        t_op.start()
        t_my.join()
        t_op.join()

        planes[:, 10] = my_lib1
        planes[:, 11] = op_lib1
        return planes

    @staticmethod
    def apply_symmetry(state_12ch, move, transform_id, board_size):
        planes = np.array(state_12ch)
        k = transform_id % 4
        for ch in range(planes.shape[0]):
            if transform_id >= 4:
                planes[ch] = np.fliplr(planes[ch])       # flip W (axis=1 on 2D)
            if k > 0:
                planes[ch] = np.rot90(planes[ch], k=-k)  # CW rotation
        if move >= 0:
            r, c = divmod(move, board_size)
            rr, cc = SYMMETRIES[transform_id % 8](r, c, board_size)
            move = rr * board_size + cc
        return planes, move
