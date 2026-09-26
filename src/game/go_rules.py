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
  - 位置超级劫（PSK）的**基础设施**：确定性 Zobrist 哈希 + 重复局面历史。
    重复判定的键是 **position-only**（`position_hash()`，只含棋盘染色）——
    Tromp-Taylor / OpenSpiel 口径；`hash()` 是含行棋方的通用局面指纹，**不用于**
    重复判定（见「Zobrist 哈希与重复局面」段与 position_hash 的 docstring）
未实现（监督学习不需要）：积攒劫、多劫循环判定、终局死子人工判定。

⚠ PSK 状态机已就位，但**尚未接入合法性**：superko 重复棋当前仍可落子
（get_legal_moves() 与 play() 的判定逻辑未动）。接入由 v21 路线图 P2.6a-2b 负责；
`tests/test_go_hash.py::test_repetition_not_yet_wired_into_legality` 锁住了这一点。
⚠ 重复**不是终局条件**：Tromp-Taylor 规则 6/8 下重复是非法手，终局是两次连续 pass
（`is_terminal()` 的语义，见其 docstring）。
"""

import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import label as _scipy_label

_label_pool = ThreadPoolExecutor(max_workers=2)

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


# ---- Zobrist 哈希表（位置超级劫的地基）----------------------------------- #
#
# 为什么必须是确定性的：PSK 判定 = 「新局面的哈希是否命中历史集合」。哈希一旦
# 不确定，同一局面两次算出不同值，重复局面就会时灵时不灵 —— 规则引擎会**静默**
# 放行本不该放、或禁掉本不该禁的着法。所以下面三条都必须成立：
#   1) 固定种子（_ZOBRIST_SEED，写死在源码里）；
#   2) 模块加载时一次性预生成（不是首次调用时随机生成）；
#   3) 表项不借用 numpy 的随机流，而是 sha256(种子 || 用途 || 索引) 的前 8 字节
#      —— 任何语言、任何 numpy 版本都能按同一规则复算出同一批常量。
#
# 覆盖范围：每个交叉点 × {黑, 白} 两色（空点不需要表项：空 = 所有点项都不异或上去）
#          + 行棋方一项。
# ⚠ 行棋方那一项**只**属于 `hash()`（通用局面指纹），**不属于** `position_hash()`
#   （重复判定专用）。见 GoBoard.position_hash 的 docstring：用含行棋方的键判重复
#   就是 SSK（situational superko），与 Tromp-Taylor / OpenSpiel 的 PSK 分歧。
_ZOBRIST_SEED = b"Go-AI/v21/GoBoard/Zobrist/v1"
_ZOBRIST_MASK = (1 << 64) - 1
# 按 (行, 列) 索引而非 r*board_size+c：这样同一坐标在任何盘口下都是同一把钥匙，
# 且与盘口大小无关。棋盘边长需 <= 32（覆盖 9/13/19 等全部在用盘口）。
_ZOBRIST_STRIDE = 32


def _zobrist_entry(label: bytes) -> int:
    return int.from_bytes(hashlib.sha256(_ZOBRIST_SEED + b"|" + label).digest()[:8],
                          "little") & _ZOBRIST_MASK


def _build_zobrist_table() -> tuple:
    """每个交叉点两把钥匙：黑 1 把、白 1 把。索引 = 2*(r*STRIDE+c) + (0黑/1白)。"""
    tbl = [0] * (2 * _ZOBRIST_STRIDE * _ZOBRIST_STRIDE)
    for r in range(_ZOBRIST_STRIDE):
        for c in range(_ZOBRIST_STRIDE):
            base = 2 * (r * _ZOBRIST_STRIDE + c)
            tbl[base] = _zobrist_entry(b"p|%d|%d|B" % (r, c))
            tbl[base + 1] = _zobrist_entry(b"p|%d|%d|W" % (r, c))
    return tuple(tbl)


# 模块加载时预生成（确定性要求，见上）
_ZOBRIST_TABLE = _build_zobrist_table()
_ZOBRIST_TO_PLAY = (_zobrist_entry(b"to_play|B"), _zobrist_entry(b"to_play|W"))
# 行棋方翻转的增量 = 两个行棋方钥匙的异或。XOR 是对合运算，所以「落子」与
# 「撤销落子」用的是同一份增量，undo() 只需再异或一次就能精确回到父局哈希。
_ZOBRIST_TO_PLAY_XOR = _ZOBRIST_TO_PLAY[0] ^ _ZOBRIST_TO_PLAY[1]


def _zobrist_key(r: int, c: int, color: int) -> int:
    """交叉点 (r, c) 上颜色 color（1=黑, -1=白）的 Zobrist 钥匙。"""
    return _ZOBRIST_TABLE[2 * (r * _ZOBRIST_STRIDE + c) + (0 if color > 0 else 1)]


class GoBoard:
    """围棋棋盘。内部棋盘取值：-1=白, 0=空, 1=黑。"""

    def __init__(self, board_size: int = 19, komi: float = 7.5):
        if board_size > _ZOBRIST_STRIDE:
            # 超出 Zobrist 坐标表的覆盖范围会静默产生钥匙碰撞（哈希恒失效、
            # PSK 判定形同虚设），因此直接拒绝而不是悄悄算错。
            raise ValueError(
                f"board_size={board_size} 超过 Zobrist 坐标表上限 "
                f"{_ZOBRIST_STRIDE}；请调大 go_rules._ZOBRIST_STRIDE 并同步更新种子")
        self.board_size = board_size
        self.komi = komi
        self.reset()

    def reset(self):
        n = self.board_size
        self.board = np.zeros((n, n), dtype=np.int8)
        self.current_player = 1  # 1=黑, -1=白
        self.ko_point = -1       # 打劫禁着点（扁平坐标），-1 表示无
        # 其余全部字段（对局进度 + 哈希 + 重复局面历史）由 _adopt_as_new_game()
        # 统一重建 —— 「新的一局」在 GoBoard 里只有这一处实现，reset 与外部盘面接管
        # 共用它，两条路径不可能长歪。
        self._adopt_as_new_game()

    def clone(self) -> "GoBoard":
        """轻量克隆：复制推演所需状态（盘面/执子方/劫/连续 pass 计数/落子数/
        **两套 Zobrist 键 + 重复局面历史**）。

        **不复制** _undo_stack 与 move_history。原实现用 copy.deepcopy，会把随手数
        线性增长的撤销栈与着法历史整份复制，是叶子批量展开的主要开销之一。克隆出的
        棋盘撤销栈为空，仍可正常 play/undo 自己的后续着法（推演不需要旧历史）。

        重复局面历史必须深拷贝：PSK 要看的是**整盘**历史，共享可变对象会让
        克隆体落子污染源棋盘的历史（反之亦然），表现为随机的假重复。
        """
        nb = GoBoard.__new__(GoBoard)
        nb.board_size = self.board_size
        nb.komi = self.komi
        nb.board = self.board.copy()
        nb.current_player = self.current_player
        nb.ko_point = self.ko_point
        nb.passes = self.passes
        nb.move_number = self.move_number
        nb.move_history = []
        nb._undo_stack = []
        nb._legal_cache = None
        nb._legal_cache_suicide = None
        nb._zobrist = self._zobrist
        nb._pos_zobrist = self._pos_zobrist
        # 哈希所对应的盘面对象 = 克隆体**自己**的 board 数组（否则 hash() 会把它
        # 当成「外部替换了棋盘」而重新采纳，反倒丢掉刚深拷贝来的历史）
        nb._zobrist_ref = nb.board
        nb._zobrist_player = self._zobrist_player
        nb._pos_hash_history = list(self._pos_hash_history)
        nb._pos_hash_counts = dict(self._pos_hash_counts)
        return nb

    # ---- 基础查询 ----------------------------------------------------------

    def __getitem__(self, idx):
        return self.board[idx]

    def is_on_board(self, r, c):
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def get_legal_moves(self, check_suicide: bool = False) -> np.ndarray:
        """返回长度为 size*size 的 bool 掩码，True 表示该点可落子。

        check_suicide=True 时额外过滤自杀手（需模拟落子，较慢但更准确）。
        结果会被缓存，直到下次 play()/undo()/reset() 时失效。
        """
        if not check_suicide and self._legal_cache is not None:
            return self._legal_cache
        if check_suicide and self._legal_cache_suicide is not None:
            return self._legal_cache_suicide

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
        if not check_suicide:
            self._legal_cache = legal.copy()
        else:
            self._legal_cache_suicide = legal.copy()
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

    # ---- Zobrist 哈希与重复局面（位置超级劫 PSK 的地基）--------------------
    #
    # **两套键，分工不可混用**（P2.6a-2a 写死）：
    #   _zobrist / hash()          棋盘 + 行棋方 -> 通用局面指纹（缓存、诊断、对拍）。
    #                               **不用于重复判定**。
    #   _pos_zobrist / position_hash()
    #                               仅棋盘染色   -> **重复判定专用**（Tromp-Taylor /
    #                               OpenSpiel 的 position：不含「轮到谁」）。
    # 为什么需要两套：通用指纹要能区分「同形但轮到对方」（否则缓存 / 对拍分不清两局），
    # 而重复判定按规则**不能**区分。用错的后果具体是：拿含行棋方的键判重复 = SSK
    # （situational superko），在「同一染色 + 异手方」的局面上与 Tromp-Taylor 分歧 ——
    # 最小可见的一例是 pass（染色不变、行棋方翻转）。
    # 两套键都在 play / undo / clone / reset / 接管路径上同步增量维护。
    #
    # 哈希按落子**增量**维护（提子与行棋方翻转都计入），不做每手全盘重算；
    # undo() 复用同一份增量精确回退。
    #
    # 两个谓词，语义必须分清（P2.6a-2b 接入 PSK 时直接依赖这个区分）：
    #   _would_repeat(candidate_pos_key) —— 「若下一手把**染色**变成 candidate，
    #       会不会命中历史」。**落子前**问，用于合法性判定。候选键必须用
    #       `position_hash_after_move(mv)` 算，不能用 `hash_after_move(mv)`。
    #   is_repetition()                   —— 「当前染色（不含本次落子）此前是否出现过」。
    #       **落子后**自查历史用。
    # 不能互相替代：play() 每次落子后都会把新染色的键追加进历史，所以落子后再问
    # is_repetition() 答案恒为真（自己那一次出现）。

    def _board_coloring_xor(self) -> int:
        """仅棋盘染色的 Zobrist 异或（不含行棋方项）—— position 键的计算核心。

        空点无项：空 = 该点的黑/白钥匙都不异或上去。
        """
        h = 0
        board = self.board
        rows, cols = np.nonzero(board)
        for r, c in zip(rows.tolist(), cols.tolist()):
            h ^= _zobrist_key(r, c, int(board[r, c]))
        return h

    def _hash_from_board(self) -> int:
        """按盘面全量重算**通用指纹**（棋盘 + 行棋方）。

        只在重置 / 侦测到哈希与盘面脱钩时用；正常落子走增量，不做全盘重算。

        ⚠ 前提：board 的取值只可能是 -1/0/1（本类只写这三个值），因此 `_zobrist_key`
        里用 `color > 0` 判黑在这里是等价的。若将来同一个数组里要放别的非零取值
        （例如死活 / 气紧之类的辅助标记），**这里会静默把它们按白子算进哈希** ——
        那时必须改成按取值分派钥匙，而不是继续沿用 `> 0`。
        """
        return self._board_coloring_xor() ^ _ZOBRIST_TO_PLAY[0 if self.current_player > 0 else 1]

    def _position_hash_from_board(self) -> int:
        """按盘面全量重算 **position 键**（仅棋盘染色，不含行棋方）。

        重复判定专用的那把；与 `_hash_from_board()` 的差别就是那一项行棋方。
        """
        return self._board_coloring_xor()

    def _adopt_as_new_game(self) -> None:
        """把当前盘面 + 行棋方**采纳**为「从此刻开局的一局新对局」。唯一实现。

        这是「外部盘面接管」的唯一入口，三处共用同一个函数（所以三条路径不可能长歪）：
          - `reset()`：造完空盘后调用；
          - `_ensure_hash()`：侦测到 board 数组被换掉 / current_player 被改写时自动调用；
          - `resync_hash()`：调用方手搓盘面后的显式入口。

        **归零而不是沿用旧值**：来路不明的旧状态与新盘面之间没有任何因果关系，留着只会
        让「盘面 / 哈希 / 历史」三者互相矛盾。实测过的两种症状：字段拷贝路径上
        move_number 停在上一局的计数（却是 0），而「只改写 current_player」那一支
        （旧实现）只重算键、却保留旧历史，于是当前局面的键根本不是历史的键，
        谓词 `_would_repeat(position_hash())` 对「就是当前局面」的位置返回假。
        ⚠ 这里配的必须是 **position-only** 键：历史里存的是染色键（见 _commit_position），
        拿 `hash()`（含行棋方）去配这个谓词会对**任何**局面都返回假 —— 那就不再是
        「一个症状」，而是谓词静默失效、整条 PSK 判定失灵。
        反过来说也解释了为什么今天仍要重建历史：正因为键只看染色，单改 current_player
        已经影响不到它，上面那条症状如今只能由「陈旧 / 异源的历史」造成 —— 而那正是
        这里选择重建（而不是沿用旧历史）要挡的东西。
        宁可漏判重复，也绝不误禁合法着法。

        归零（**对局进度**，属于上一局）：
          passes = 0        连续 pass 计数
          move_number = 0   已落手数（含 pass）。刻意不从 move_history 派生：clone() 有意
                            不带 move_history（MCTS 叶子只要推演状态），派生会让子局归零。
          move_history = [] 落子扁平坐标序列，pass 记为 -1
          _undo_stack = []  撤销栈，每项 (move, captured|None, prev_ko, prev_passes,
                            prev_player)。必须一起清：栈里存的是上一局的着法，撤销它会把
                            上一局的子恢复到新盘上，并且回滚时会把刚重建的历史弹空。
          _legal_cache / _legal_cache_suicide = None
                            盘面可能被换过，缓存的掩码属于旧盘面。
        保留（**局面描述**，调用方写了什么就是什么）：
          board、current_player、**ko_point**。ko_point 是「盘面 + 上一手」的性质而不是
          对局进度：Tromp-Taylor 规则 6 的 PSK 只看盘面涂色，而简单劫是它「只禁紧邻上一手
          之前那个局面」这一条更弱的限制，所以从该局面开新局时劫禁着照样成立。
          light_rollout 显式拷贝 ko_point 要的就是这个语义，不能在这里清掉。
        重建：按盘面重算**两套**键，历史 = {当前染色的 position 键}（**绝不伪造父局历史**）。
        """
        self.passes = 0
        self.move_number = 0
        self.move_history = []
        self._undo_stack = []
        self._legal_cache = None
        self._legal_cache_suicide = None
        self._zobrist = self._hash_from_board()
        self._pos_zobrist = self._position_hash_from_board()
        self._zobrist_ref = self.board
        self._zobrist_player = self.current_player
        self._pos_hash_history = [self._pos_zobrist]
        self._pos_hash_counts = {self._pos_zobrist: 1}

    def _ensure_hash(self) -> None:
        """O(1) 守卫：侦测「增量哈希与盘面/行棋方脱钩」，并把当前局面**接管成一局新局**。

        入口（与本 docstring 保持一致）：`play()` / `undo()` / `hash()` /
        `position_hash()` / `hash_after_move()` / `position_hash_after_move()` /
        `is_repetition()` / `_would_repeat()` 全部先调它。
        `get_legal_moves()` 不调（它不碰哈希）。

        两种脱钩的处置**完全相同**，都走 `_adopt_as_new_game()`（契约见其 docstring）：
          - board **数组对象**被整体换掉（如 light_rollout 的只读推演逐字段赋值）；
          - 只有 current_player 被直接改写。
        两者都按「以此局面为新局」处理：不保留来路不明的旧历史，也不保留旧的对局计数。
        （就地改写 board 元素、不换数组对象是察觉不到的 —— 那条路径必须由调用方
        显式 resync_hash()，见该方法。）
        """
        if self._zobrist_ref is self.board and self._zobrist_player == self.current_player:
            return
        self._adopt_as_new_game()

    def resync_hash(self) -> None:
        """把当前盘面与行棋方**采纳**为「从此刻开局的一局新对局」（显式入口）。

        外部直接手搓棋盘（测试夹具、导入外部局面、只读推演）后必须调用，否则增量
        哈希与盘面不一致。整体替换 board 数组的场景 `_ensure_hash()` 会自动采纳，
        本方法是给「就地改写」用的显式入口。

        语义与自动采纳**逐字段一致**（共用 `_adopt_as_new_game`）：move_number /
        passes / move_history / _undo_stack 归零，重复局面历史重建为 {当前局面}；
        ko_point 与盘面、行棋方一样按调用方写的保留。
        """
        self._adopt_as_new_game()

    def hash(self) -> int:
        """通用局面指纹（棋盘 + **行棋方**）的 Zobrist 哈希。

        返回 [0, 2**64) 的 Python int。语义稳定：盘面与行棋方相同 => 哈希相同，
        跨实例、跨进程逐位一致（固定种子 + 模块加载时预生成）。

        ⚠ **不用于重复判定**。重复判定必须用 `position_hash()`（仅棋盘染色）。
        拿本方法去查重复历史 = SSK（situational superko），与 Tromp-Taylor /
        OpenSpiel 的 PSK 在「同一染色 + 异手方」的局面上分歧 —— 最小可见的一例是
        pass：染色不变、行棋方翻转，SSK 判它是新局面，PSK 判它重复同一染色。

        只读：不改盘面、行棋方、历史集合，也不失效合法性缓存。唯一可能的写入是
        按当前盘面重算**过期**的增量哈希缓存（见 _ensure_hash）。
        """
        self._ensure_hash()
        return self._zobrist

    def position_hash(self) -> int:
        """**重复判定专用**键：仅棋盘染色的 Zobrist 哈希（不含行棋方）。

        这就是 Tromp-Taylor / OpenSpiel 口径下的 position —— 棋盘涂色本身，
        既不含「轮到谁」，也不含历史着法。与 `hash()` 的差别就是那一项行棋方。

        与 `hash()` 的分工（写死，别用反）：
          - 重复判定（`_pos_hash_history` / `_pos_hash_counts` / `_would_repeat` /
            `is_repetition`）**只**用本方法及其增量版本 `position_hash_after_move()`；
          - `hash()` 是通用指纹（缓存键、诊断、对拍），需要区分「同形但轮到对方」。

        为什么必须两套：规则要求重复判定**不**区分行棋方，而通用指纹**要**区分。
        用错的后果 = SSK = 与 Tromp-Taylor 分歧（见 `hash()` 的 docstring）。

        返回 [0, 2**64) 的 Python int，跨实例 / 跨盘口 / 跨进程确定性：只由
        (行, 列, 颜色) 决定（空盘恒为 0）。

        只读：不改盘面、行棋方、历史集合，也不失效合法性缓存。
        """
        self._ensure_hash()
        return self._pos_zobrist

    def _forecast_delta(self, move: int) -> int:
        """只读推演一手棋对**棋盘染色**的哈希增量（落子点提子），不含行棋方翻转。

        内部会临时落子以判定提子并立即还原（还原的是**原值**，因此即使误传占点
        也不会擦掉棋子），故与 play() 一样不可重入。move == -1（pass）返回 0：
        pass 不改变染色。
        """
        if move == -1:
            return 0
        n = self.board_size
        r, c = divmod(move, n)
        color = self.current_player
        delta = 0
        saved = self.board[r, c]
        try:
            self.board[r, c] = color
            opponent = -color
            for nb_color, comp in self._neighbor_groups(r, c):
                if nb_color == opponent:
                    cr0, cc0 = comp[0]
                    if (self.board[cr0, cc0] == opponent
                            and not self._group_has_liberty(cr0, cc0)):
                        for (cr, cc) in comp:
                            delta ^= _zobrist_key(cr, cc, opponent)
            delta ^= _zobrist_key(r, c, color)
        finally:
            self.board[r, c] = saved
        return delta

    def hash_after_move(self, move: int) -> int:
        """只读推演：若现在下 move（-1 = pass），落子后**通用指纹**的值。不改任何状态。

        与 `position_hash_after_move()` 的差别就是行棋方那一项，所以
        **重复判定不要用本方法**（含行棋方 = SSK 的候选键）。PSK 合法性请用
        `position_hash_after_move()`。

        只对合法着法（含 pass）有定义；占点 / 劫禁着等非法落子语义未定义，调用方须先过
        合法性检查。内部会临时落子以判定提子并立即还原，故与 play() 一样不可重入。
        """
        self._ensure_hash()
        return self._zobrist ^ _ZOBRIST_TO_PLAY_XOR ^ self._forecast_delta(move)

    def position_hash_after_move(self, move: int) -> int:
        """只读推演：若现在下 move（-1 = pass），落子后**position 键**（仅染色）。

        **P2.6a-2b 接线合法性时用本方法**算候选键：
            `board._would_repeat(board.position_hash_after_move(mv))`
        不可改用 `hash_after_move(mv)`（那是含行棋方的通用指纹 = SSK）。

        move == -1（pass）返回**当前**键：pass 不改变染色，因此「pass 之后的染色」
        必然已经在历史里，谓词会返回真。**接线时必须把 pass 排除在 PSK 判定之外**
        （Tromp-Taylor 规则 6 禁的是重复棋，不是 pass；终局由两次连续 pass 表达），
        否则任何 pass 都会被误判非法。

        只对合法着法有定义；占点 / 劫禁着等非法落子语义未定义，调用方须先过合法性检查。
        内部会临时落子以判定提子并立即还原，故与 play() 一样不可重入。
        """
        self._ensure_hash()
        return self._pos_zobrist ^ self._forecast_delta(move)

    def _would_repeat(self, candidate_pos_key: int) -> bool:
        """PSK 谓词：局面**染色**变成 candidate_pos_key 会不会命中历史。

        candidate 必须是 **position-only 键**（`position_hash()` /
        `position_hash_after_move()`）；传 `hash()` 的值几乎必然查不到，表现为
        「重复判定静默失效」。

        「历史」含初始空局面。**落子前**在候选键上问 —— 因为 play() 每次落子后
        都会把新染色的键追加进历史，落子后再自查必然为真。

        同样先过 `_ensure_hash()`：历史若属于另一个盘面，命中判定就毫无意义
        （O(1) 恒等比较，不在热路径上）。
        """
        self._ensure_hash()
        return candidate_pos_key in self._pos_hash_counts

    def is_repetition(self) -> bool:
        """当前**染色**此前是否出现过 —— **不含本次落子**。

        历史记录的是「每一手落子之后」的染色（含初始空局面），所以当前染色的那
        一次出现总在集合里；真正要问的是「除本手之外当前染色此前是否出现过」，
        即该键的出现次数 > 1。undo() 之后当前局面回到父局，语义同样成立。

        注意 pass：pass 不改变染色，所以**任何一手 pass 之后本方法都为真**
        （该染色第二次出现）。这不是 bug，是 PSK 的定义；因此合法性判定必须把
        pass 排除在外（见 `position_hash_after_move()` 的 docstring）。
        """
        self._ensure_hash()
        return self._pos_hash_counts.get(self._pos_zobrist, 0) > 1

    def _board_delta(self, moved_by: int, move: int, captured) -> int:
        """一手棋对**棋盘染色**的哈希增量（落子点 + 提子），不含行棋方翻转。

        pass（move < 0）返回 0：染色没变。

        XOR 是对合运算 => 增量的逆就是它本身，所以「落子」与「撤销落子」用的是**同一条**
        表达式（而不是各存一份父哈希），两套键都能精确回到父局，且两处不可能写歪。
        通用指纹的增量 = 本增量 ^ 行棋方翻转项。
        """
        delta = 0
        if move >= 0:
            n = self.board_size
            r, c = divmod(move, n)
            delta ^= _zobrist_key(r, c, moved_by)
            for (cr, cc) in captured:
                delta ^= _zobrist_key(cr, cc, -moved_by)
        return delta

    def _commit_position(self, moved_by: int, move: int, captured) -> None:
        """增量更新**两套**键，并把新染色的键追加进历史（每次成功落子一次）。

        两套键共用同一条棋盘增量：通用指纹额外异或一次行棋方翻转项，position 键不异或。
        历史集合只收 position 键 —— 重复判定的键只有这一把。
        """
        delta = self._board_delta(moved_by, move, captured)
        self._zobrist ^= delta ^ _ZOBRIST_TO_PLAY_XOR
        self._pos_zobrist ^= delta
        self._zobrist_player = -moved_by
        h = self._pos_zobrist
        self._pos_hash_history.append(h)
        self._pos_hash_counts[h] = self._pos_hash_counts.get(h, 0) + 1

    def _rollback_position(self, moved_by: int, move: int, captured) -> None:
        """回退**两套**键并把历史末尾那一项弹出（undo 一次）。

        注意必须按**出现次数**回退而不是 set.discard()：同一染色可能出现多次
        （例如实着 + 随后的 pass），直接 discard 会把更早那次出现也抹掉，于是
        「已经重复过」被误判成「没重复过」—— 一个静默放行 superko 的 bug。
        """
        # 离场局面（撤销前的当前局面）必须就是历史末项，否则哈希与历史已经脱钩
        leaving = self._pos_zobrist
        delta = self._board_delta(moved_by, move, captured)
        self._zobrist ^= delta ^ _ZOBRIST_TO_PLAY_XOR
        self._pos_zobrist ^= delta
        self._zobrist_player = moved_by
        last = self._pos_hash_history.pop()
        assert last == leaving, (
            f"哈希回退与历史不自洽: 历史末项 {last} != 离场局面 {leaving}")
        remaining = self._pos_hash_counts[last] - 1
        if remaining:
            self._pos_hash_counts[last] = remaining
        else:
            del self._pos_hash_counts[last]

    @property
    def to_play(self) -> int:
        """当前行棋方（1=黑, -1=白）。current_player 的只读别名，不新增真相源。"""
        return self.current_player

    @property
    def num_passes(self) -> int:
        """**连续** pass 计数（任何实着归零）。

        passes 的只读别名；字段名 passes 保留给 scripts/cli_play.py 与
        scripts/webui.py 等既有调用点。
        """
        return self.passes

    # ---- 落子 --------------------------------------------------------------

    def play(self, move: int, record: bool = True) -> bool:
        """
        落子。move 为扁平坐标 (0..size*size-1)，或 -1 表示 pass。
        返回是否成功（非法落子返回 False 且不改变状态）。

        record=True（默认）时把撤销信息压入 _undo_stack，可用 undo() 撤销
        本次落子（含提子恢复 / 劫点 / pass 计数 / 历史 / 执子方 / 两套 Zobrist 键
        （通用指纹 + 重复判定用的 position 键）/ 重复局面历史 / 落子数）。
        """
        n = self.board_size
        self._ensure_hash()
        if move == -1:
            # pass
            if record:
                self._undo_stack.append((-1, None, self.ko_point, self.passes, self.current_player))
            self.passes += 1
            self.ko_point = -1
            self.move_history.append(-1)
            self.move_number += 1
            self._commit_position(self.current_player, -1, None)
            self.current_player = -self.current_player
            self._legal_cache = None
            self._legal_cache_suicide = None
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
        self.move_number += 1
        self._commit_position(color, move, captured)
        self.current_player = -self.current_player
        self._legal_cache = None
        self._legal_cache_suicide = None
        return True

    def undo(self) -> bool:
        """撤销最近一次成功 play（须 play(record=True)）。

        完整恢复：棋盘子与被提子、劫禁着点、pass 计数、着法历史、执子方、落子数、
        两套 Zobrist 键（通用指纹 + 重复判定用的 position 键）、
        重复局面历史（含出现次数）。
        返回是否成功（栈空返回 False）。

        入口同样先过 `_ensure_hash()`：若盘面/行棋方已被外部改写，接管会连带清空
        属于旧局的撤销栈，于是这里返回 False —— 而不是拿旧局的着法去恢复一颗
        新盘上不存在的棋子。
        """
        self._ensure_hash()
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
        self.move_number -= 1
        self.current_player = player
        self._rollback_position(player, move, captured)
        self._legal_cache = None
        self._legal_cache_suicide = None
        return True

    # ---- 终局与计分 --------------------------------------------------------

    def is_terminal(self, max_moves: int = None) -> bool:
        """终局判定：连续两次 pass（对局认输），或达到 move 上限。

        ⚠ **重复局面不是终局条件**：Tromp-Taylor 规则 6/8 下「重复」是**非法手**
        （PSK 由 _would_repeat 判，接线在 P2.6a-2b），终局只由两次连续 pass 表达。
        本方法不查重复历史，不要在这里加。

        max_moves 为 None 时取 2 * board_size ** 2 —— 沿用仓库既有约定
        （tests/test_go_rules.py 随机对弈的步数上限、light_rollout 的
        max_steps 都是 n*n*2），因此取这个默认值不会改变任何既有调用点的行为；
        传参可让调用方按自己的对局长度约定覆盖。
        """
        if self.num_passes >= 2:
            return True
        if max_moves is None:
            max_moves = 2 * self.board_size * self.board_size
        return self.move_number >= max_moves

    def is_game_over(self) -> bool:
        """[保留] is_terminal() 的别名，维持既有调用点可用
        （tests/test_go_rules.py 在用）。新代码请用 is_terminal()，它多了
        move 上限且允许传上限。"""
        return self.is_terminal()

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

        future_my = _label_pool.submit(_label_and_mark,
                                       boards == to_play, my_lib1, to_play)
        future_op = _label_pool.submit(_label_and_mark,
                                       boards == -to_play, op_lib1, -to_play)
        future_my.result()
        future_op.result()

        planes[:, 10] = my_lib1
        planes[:, 11] = op_lib1
        return planes

    @staticmethod
    def apply_symmetry_batch(states, moves, transform_ids, board_size):
        """批量对称增强（全向量化，无逐样本 Python 循环）。

        states: (B, C, H, W)；moves: (B,) 扁平坐标（-1=pass 不变换）；
        transform_ids: (B,)，取值 0..7。
        变换顺序与单样本版一致：先 flip W（transform>=4），
        再顺时针旋转 k=transform%4 次（k=1/2/3 对应 90°/180°/270°）。
        """
        x = np.asarray(states)
        transform_ids = np.asarray(transform_ids)
        out = np.empty_like(x)
        flip_mask = transform_ids >= 4
        if flip_mask.any():
            out[flip_mask] = x[flip_mask][..., :, ::-1]
        keep = ~flip_mask
        if keep.any():
            out[keep] = x[keep]
        # 旋转：按 k 分组，每组一次向量化变换（无逐样本循环）
        # CW90 = 转置后翻转新 W 轴；180 = 双轴翻转；CW270 = 转置后翻转新 H 轴
        for k in (1, 2, 3):
            mask = (transform_ids % 4) == k
            if not mask.any():
                continue
            if k == 1:
                out[mask] = out[mask].transpose(0, 1, 3, 2)[..., :, ::-1]
            elif k == 2:
                out[mask] = out[mask][..., ::-1, ::-1]
            else:
                out[mask] = out[mask].transpose(0, 1, 3, 2)[..., ::-1, :]
        # moves 变换：SYMMETRIES 为纯算术 lambda，天然支持 numpy 数组
        moves_out = np.array(moves, copy=True)
        idxs = np.flatnonzero(moves_out >= 0)
        if idxs.size:
            n = board_size
            mv = moves_out[idxs]
            r = mv // n
            c = mv % n
            ids = transform_ids[idxs]
            for t in range(8):
                m = ids == t
                if not m.any():
                    continue
                rr, cc = SYMMETRIES[t](r[m], c[m], n)
                mv[m] = rr * n + cc
            moves_out[idxs] = mv
        return out, moves_out

    @staticmethod
    def apply_symmetry(state_12ch, move, transform_id, board_size):
        """单样本对称增强（内部走批量实现，保持旧接口兼容）。"""
        planes, moves = GoBoard.apply_symmetry_batch(
            np.asarray(state_12ch)[None],
            np.asarray([move]),
            np.asarray([transform_id]),
            board_size,
        )
        return planes[0], int(moves[0])
