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
  - Tromp-Taylor 计分 + 贴目（默认 6.5）
未实现（监督学习不需要）：超级劫、积攒劫、多劫循环判定。
"""

import numpy as np

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

    def __init__(self, board_size: int = 19, komi: float = 6.5):
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

    # ---- 基础查询 ----------------------------------------------------------

    def __getitem__(self, idx):
        return self.board[idx]

    def is_on_board(self, r, c):
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def get_legal_moves(self) -> np.ndarray:
        """返回长度为 size*size 的 bool 掩码，True 表示该点可落子。"""
        n = self.board_size
        legal = (self.board == 0).reshape(-1).copy()
        if self.ko_point >= 0:
            legal[self.ko_point] = False
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
        libs = 0
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not self.is_on_board(nr, nc):
                    continue
                v = self.board[nr, nc]
                if v == 0:
                    libs += 1
                elif v == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return libs

    # ---- 落子 --------------------------------------------------------------

    def play(self, move: int) -> bool:
        """
        落子。move 为扁平坐标 (0..size*size-1)，或 -1 表示 pass。
        返回是否成功（非法落子返回 False 且不改变状态）。
        """
        n = self.board_size
        if move == -1:
            # pass
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
        for color, comp in self._neighbor_groups(r, c):
            if color == opponent:
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

        # 打劫判定：提掉恰好 1 子，且落子子本身恰好只剩 1 气（即被提点）-> 形成劫
        if len(captured) == 1 and self._group_liberty_count(r, c) == 1:
            self.ko_point = captured[0][0] * n + captured[0][1]
        else:
            self.ko_point = -1

        self.passes = 0
        self.move_history.append(move)
        self.current_player = -self.current_player
        return True

    # ---- 终局与计分 --------------------------------------------------------

    def is_game_over(self) -> bool:
        return self.passes >= 2

    def score(self) -> float:
        """
        Tromp-Taylor 计分：空点归最近同色连通块；黑分 = 黑子 + 黑围空，白同理 + 贴目。
        返回黑方视角得分（>0 黑胜，<0 白胜）。
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
    # my_hist / op_hist: 长度均为 3 的扁平坐标序列（不足补 -1），最近一手在最后。

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
            避免单图版的双层 Python for 循环（每个非空点一次 flood fill），
            在 B 较大时（训练 batch）提速数倍。
        """
        from scipy.ndimage import label as _label
        boards = np.asarray(boards)
        B, n, _ = boards.shape
        planes = np.zeros((B, 12, n, n), dtype=np.float32)
        to_play = np.asarray(to_play).reshape(B, 1, 1)
        opp = -to_play  # (B,1,1)

        # 通道 0/4: 己方/对手棋子
        planes[:, 0] = (boards == to_play)
        planes[:, 4] = (boards == opp)

        # 通道 1-3 / 5-7: 历史手（高级索引 scatter）
        my_hist = np.asarray(my_hist).reshape(B, 3)
        op_hist = np.asarray(op_hist).reshape(B, 3)
        for k in range(3):
            mv = my_hist[:, k]
            valid = mv >= 0
            r, c = np.divmod(np.where(valid, mv, 0), n)
            planes[np.arange(B)[valid], 1 + k, r[valid], c[valid]] = 1.0
            mv = op_hist[:, k]
            valid = mv >= 0
            r, c = np.divmod(np.where(valid, mv, 0), n)
            planes[np.arange(B)[valid], 5 + k, r[valid], c[valid]] = 1.0

        # 通道 8: 合法点掩码（空点，劫禁着点排除）—— 与单图 get_legal_moves 一致
        legal = (boards == 0).astype(np.float32)
        if ko is not None:
            ko = np.asarray(ko).reshape(B)
            for b in range(B):
                if ko[b] >= 0:
                    r, c = divmod(int(ko[b]), n)
                    legal[b, r, c] = 0.0
        planes[:, 8] = legal

        # 通道 9: 执子方常数
        planes[:, 9] = to_play.astype(np.float32)

        # 通道 10/11: 气数=1 掩码（向量化连通块标注 + 邻空计数）
        my_lib1 = np.zeros((B, n, n), dtype=np.float32)
        op_lib1 = np.zeros((B, n, n), dtype=np.float32)
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)  # 4 邻域
        for b in range(B):
            board_b = boards[b]
            tp = int(to_play[b, 0, 0])
            opp_b = -tp
            empty = (board_b == 0)
            # 每个棋子点的 4 邻域空点坐标数（整数 0-4），与单图
            # _group_liberty_count 逐点计数语义一致（每个空邻域坐标各算 1 气）。
            up = np.zeros_like(empty, dtype=np.int64); up[:-1, :] = empty[1:, :]
            down = np.zeros_like(empty, dtype=np.int64); down[1:, :] = empty[:-1, :]
            left = np.zeros_like(empty, dtype=np.int64); left[:, :-1] = empty[:, 1:]
            right = np.zeros_like(empty, dtype=np.int64); right[:, 1:] = empty[:, :-1]
            neigh_empty = up.astype(np.int64) + down + left + right
            for color, lib_plane in ((tp, my_lib1[b]), (opp_b, op_lib1[b])):
                mask = (board_b == color)
                if not mask.any():
                    continue
                labelled, num = _label(mask, structure=struct)
                if num == 0:
                    continue
                # 每个 group 的邻空数 = 该 group 内点邻域空点坐标总数（不去重空块）
                # 用 np.add.at 仅对棋子点（mask 为真）按 labelled 累加 neigh_empty。
                lib_counts = np.zeros(num + 1, dtype=np.int64)
                np.add.at(lib_counts, labelled[mask], neigh_empty[mask])
                lib1_mask = (lib_counts[labelled] == 1) & mask
                lib_plane[lib1_mask] = 1.0

        planes[:, 10] = my_lib1
        planes[:, 11] = op_lib1
        return planes

    @staticmethod
    def apply_symmetry(state_12ch, move, transform_id, board_size):
        planes = np.array(state_12ch)
        for ch in range(planes.shape[0]):
            planes[ch] = np.rot90(planes[ch], k=transform_id % 4)
            if transform_id >= 4:
                planes[ch] = np.fliplr(planes[ch])
        if move >= 0:
            r, c = divmod(move, board_size)
            rr, cc = SYMMETRIES[transform_id % 8](r, c, board_size)
            move = rr * board_size + cc
        return planes, move
