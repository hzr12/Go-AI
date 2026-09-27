"""
围棋规则引擎（纯 Python + numpy）。

这是整个项目的规则真相来源（single source of truth）。
同时服务于：
  - 棋谱重放（SGF -> 状态张量）
  - 自我对弈（将来）
  - 评估

规则范围（按诊断结论取舍，聚焦监督学习可用的最小正确集）：
  - 提子（连通块气计算）
  - 打劫 —— **完全由位置超级劫承载，没有单独的劫禁**。Tromp-Taylor 规则 6 的全文只有
    「A turn is either a pass; or a move that doesn't repeat an earlier grid coloring.」
    一句，**没有** ko-capture 从句（全文见 `GoBoard` 类 docstring），所以本引擎**不设**
    「不能回提劫」的禁令：`play()` 曾经那道 `move == self.ko_point` 的劫禁已于
    P2.6a-2c 删掉，经典单子劫由 PSK 独立覆盖，而「多子提子形成的劫」的立即回提
    按 TT 本就合法。`ko_point` 随之降为「上一手是否形成简单劫」的**只读信息位**。
  - 禁自杀（落子后自身无气且未提子则非法）—— ⚠ **相对 TT 的有意偏离**：TT 规则**本身
    允许自杀**。TT 规则 5 就是「coloring an empty point one's own color; then clearing
    the opponent color, **and then clearing one's own color**」——「清自己的子」那一步
    就是自提（自杀），Tromp-Taylor 并不禁它；Tromp 原话：「Suicide. Neutral; I have a
    slight preference for allowing it.」本引擎选择**禁止**自杀，是向日式 / OpenSpiel
    等多数规则集靠拢的**取舍**，不是 TT 合规要求。**别拿 TT 原文来「纠正」这一条。**
  - pass（连续两次 pass 终局）
  - 中国规则计分：数子法（区域计分）+ 贴目 7.5
    （计分算法与 Tromp-Taylor 等价：空点归最近同色连通块；差异见 score() 注释）
  - 位置超级劫（PSK）：确定性 Zobrist 哈希 + 重复局面历史，**并已接入合法性**。
    重复判定的键是 **position-only**（`position_hash()`，只含棋盘染色）——
    Tromp-Taylor / OpenSpiel 口径；`hash()` 是含行棋方的通用局面指纹，**不用于**
    重复判定（见「Zobrist 哈希与重复局面」段与 position_hash 的 docstring）
未实现（监督学习不需要）：积攒劫、多劫循环判定、终局死子人工判定。

⚠ 合法性判据**只有一套**：`get_legal_moves()` 定义它（那里判**三条**：
空点 / 禁自杀 / PSK；经典单子劫由 PSK 覆盖，**不单独判**），`play()` 逐条照判
（P2.6a-2c 起）。两点必须记住：
  - 重复**不是终局条件**：Tromp-Taylor 规则 6/8 下重复是**非法手**，终局是两次连续
    pass（`is_terminal()` 的语义，见其 docstring）。
  - **`play()` 判的是同一套**（P2.6a-2c 起：占点 / 禁自杀 / PSK，pass 豁免），
    所以两条路径对**任意手**给出相同答案 —— 掩码是合法点集合的**唯一**真相源，
    而 `play()` 是它唯一的执行口，二者不再有分工也没有偏差。
    逐点对拍由 `tests/test_go_rules_legality.py::test_play_and_mask_agree_on_every_move`
    钉住；改动其中任何一侧的判定时，那条测试会红。

---- 动作空间（OpenSpiel 风格，P2.6a-2b-2）----
三套编码并存，**各自的 pass 编号不同**，这是本引擎最容易静默出错的地方：
  - **棋盘方言**：`play()` / `move_history` / `feature_planes` 历史通道 / MCTS 的
    `path_moves` / webui 与 cli_play 的 `PASS = -1` —— pass 记作 **-1**；
  - **动作空间**：`GoBoard.PASS`（= `n*n`）、`num_actions()`（= `n*n+1`）、
    `legal_actions()`、`is_legal()`、MCTS 的树内编码（`n_actions - 1`）——
    pass 记作 **n*n**；
  - **掩码**：长度恒为 n*n，**没有** pass 槽位。
所以 `is_legal(-1)` 为 **False**（越界）而 `play(-1)` 为 **True**（合法 pass）；
`is_legal(PASS)` 为 **True** 而 `play(PASS)` 为 **False**（越界）——
**这两对都不可比**，`is_legal()` 的返回值不能拿去预判 `play()` 的返回值，
要落一个动作编号必须先换算：`play(-1 if a == board.PASS else a)`。
PASS 恒合法且**不查 PSK**：pass 不改变染色，`position_hash_after_move(-1)` 恒等于
当前键（该键必然已在历史里），谓词恒为真，接上去就会把所有 pass 判成非法、
对局永远无法终局。完整说明见「动作空间」段与各方法 docstring。
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


# 4 邻域偏移（模块级常量，避免在逐候选的热循环里每次重建元组）。
# 与 _neighbor_groups / _group_has_liberty / _group_liberty_count 里的内联写法同义。
_NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _groups_desync_error(r, c):
    """棋块/气表与盘面脱钩时的异常（就地改写 board 却没 resync_hash 的后果）。

    P2.7a 之前这个契约违反**不抛**：查气走的是 flood fill，它直接读活盘面，
    所以只有 PSK 那一半用的是陈旧哈希。改成查表之后，一颗没有块号的棋子会变成
    `KeyError: -1` —— 对调用者毫无信息量。这里换成人话，并说清怎么修。

    守卫是热路径上每次查表前的一次整数比较（`g < 0`），只在**真的脱钩**时才
    走到构造异常那条路。
    """
    return RuntimeError(
        f"棋块/气表与盘面脱钩：({r},{c}) 上有棋子却没有块号。"
        "直接就地改写 board.board 之后必须调 resync_hash()"
        "（见 GoBoard 类 docstring 的「局面状态字段的读写契约」）。")


class _PassSlot:
    """`GoBoard.PASS` 的描述符：PASS 的动作值恒等于 `board_size * board_size`。

    为什么是描述符而不是一个普通类属性：PASS 的值**随盘口变**（5 路 25 / 9 路 81 /
    19 路 361），写成 `PASS = 361` 那种类级 int 就等于把「默认盘口」硬编码进引擎 ——
    5 路的 `board.PASS` 会静默变成 361，而 `coord_to_action(5, 0)` 返回 25，
    于是「动作空间」与「扁平坐标」两套编号错位 336，症状是 pass 被当成棋盘上的一个点、
    或者随机走子下出越界动作。宁可让它**大声报错**。

    因此**类上访问（`GoBoard.PASS`）直接抛 AttributeError**，只允许实例访问
    （`board.PASS` / `self.PASS`）。这条报错是设计的一部分：它拦住
    `PASS = GoBoard.PASS` 这类模块级常量（那会在 5 路与 19 路共用一个值）。

    **写入 / 删除也一并封死**（`__set__` / `__delete__`）。只定义 `__get__` 的话本类
    是**非数据描述符**，而实例 `__dict__` 在描述符之前查找 —— 于是
    `board.PASS = 5` 会**静默**在实例上挂一个 int 5，把描述符整个遮蔽掉：此后
    `board.PASS` 恒为 5，与 `board_size * board_size`（5 路 25 / 19 路 361）分叉，
    `num_actions() == PASS + 1` 这条不变式当场失效，症状是「pass 被当成棋盘上的
    一个点」而且**没有任何报错**。补上 `__set__` 之后本类成为数据描述符，
    属性查找永远走它，赋值只能大声失败。
    真想给一个具体盘面钉一个 pass 编号，请写**局部变量** `pass_action = board.PASS` ——
    它的作用域天然不会跨盘口。

    ⚠ 覆盖范围：这两个方法只拦**实例**上的写入。类级重绑定
    （`GoBoard.PASS = 5`）是给类属性赋值、不经过描述符协议，拦不住 —— 要拦它得上
    metaclass，超出本任务范围（那一处至少还是**可见**的：改完之后类上访问不再抛
    AttributeError，单元测试第 1 条会立刻发现）。
    """
    __slots__ = ()

    def __get__(self, obj, objtype=None):
        if obj is None:
            raise AttributeError(
                "GoBoard.PASS 必须从实例读（board.PASS）：它的值是 board_size * "
                "board_size，随盘口变化（5 路 25 / 9 路 81 / 19 路 361），类上没有"
                "「默认盘口」可言。要一个模块级常量就写 "
                "`PASS = GoBoard(board_size).PASS`。")
        return obj.board_size * obj.board_size

    def __set__(self, obj, value):
        raise AttributeError(
            f"GoBoard.PASS 是只读槽位，不得赋值（board.PASS = {value!r}）：它的值"
            f"由 board_size * board_size 唯一决定（{obj.board_size} 路 = "
            f"{obj.board_size * obj.board_size}），写进去就等于凭空造出第二个"
            f"真相源，且会静默遮蔽本描述符。真要一个 pass 编号就写局部变量 "
            f"`pass_action = board.PASS`。")

    def __delete__(self, obj):
        raise AttributeError(
            "GoBoard.PASS 是只读槽位，不得删除（del board.PASS）：它是全类共享的"
            "动作空间约定，删掉之后所有实例访问都会退化成 AttributeError，而调用点"
            "（legal_actions / is_legal / MCTS 的 n_actions-1）会一起崩。")


class GoBoard:
    """围棋棋盘。内部棋盘取值：-1=白, 0=空, 1=黑。

    ---- 局面状态字段的读写契约 ----
    `board` / `current_player` 是**真相源**（也是「外部盘面接管」的唯一入口：
    改写它们之后必须让 `_ensure_hash()` 或 `resync_hash()` 跑一次）。
    `ko_point` 是**只读派生量**，见下面的专门说明。
    `passes` / `move_number` / `move_history` / `_undo_stack` 是**对局进度**，
    由 `play()` / `undo()` / `_adopt_as_new_game()` 维护，消费者只读
    （只读别名：`to_play` / `num_passes`）。

    ---- `ko_point`：只读派生量，**不参与合法性判罚** ----
    语义只剩一句话：「**上一手是否形成了简单劫**」——若上一手恰好提掉一颗子、且落下的
    这颗子自身只剩 1 口气（正好是被提点），`ko_point` 记下那颗被提子的扁平坐标，
    否则为 -1。它由 `play()` 写、`undo()` 回退，**任何消费者都只该拿它当信息位读**
    （`feature_planes_batched` 的 ko 通道就是这种读法）。

    为什么不参与判罚（P2.6a-2b-1 降级、P2.6a-2c 收口，v21 路线图 D14）：**位置超级劫
    覆盖了经典的单子劫**。单子劫的回提会把染色复现成「上一手之前」那个局面，而那个染色
    的 position 键必然记在 PSK 历史里，所以 PSK 判定会独立地把它禁掉 —— 两条判罚叠加
    没有增量信息，却让「合法性由谁裁决」这件事多了一个真相源。
    实测后果：掩码里 `ko_point` 置 -1 与不置，掩码**逐位相同**
    （`tests/test_go_rules_legality.py::test_ko_point_does_not_affect_legality`）。
    ⚠ 「PSK 覆盖简单劫」这句话**只对单子劫成立**：`ko_point` 的成劫判据并不要求落子后
    那块是单子，多子提子形成的劫其回提不复现染色，PSK 判不出来 —— 而按 TT，**那一手
    本来就合法**，PSK 判不出来正是规则要的答案。**两条路径在这一点上永远一致。**

    ---- 与 TT 的对齐：判据只有 PSK 一条（P2.6a-2c）----
    Tromp-Taylor 规则 6 的原文**只有一句**：
    「A turn is either a pass; or a move that doesn't repeat an earlier grid coloring.」
    —— **没有**「禁劫争」那半句。合法性判据因此就是**位置超级劫**（position-only 键），
    「多子提子形成的劫，其立即回提」是**合法手**（回提后的染色与提子前**不同**）。

    曾经存在的偏差**在 `play()` 侧**，已在本任务删除：那里另有一道
    `if move == self.ko_point: return False`，它自己的成劫判据不要求那块是单子，于是
    出现两种棋「掩码放行、`play()` 拒绝」（实测 9 路随机自对弈里约 **1.2%** 的取点撞上）：
      (1) **多子提子形成的劫**（`play()` 的成劫判据是「恰好提 1 子 **且** 落子后己方块
          总气 == 1」，**没有**要求那块是单子）。若那块有 2 颗以上，回提会一次提掉
          整块，落子后的染色与「提子之前」那个染色**不同** -> 不是 PSK 重复，掩码按 TT
          放行是对的，而那道额外的劫禁仍然拒它。
      (2) **外部盘面接管**（`_ensure_hash()` 触发 `_adopt_as_new_game()`）之后：历史被
          重建成 {当前局面}，提子之前那个染色不在历史里，PSK 同样判不出 —— 掩码放行，
          而那道劫禁仍然拒它。

    收口方式：**拿掉 `play()` 的劫禁 + 补上 PSK 检查**。两者必须**同时**做，只删不补
    会让 `play()` 反向变宽松（连经典单子劫的立即回提都接受，因为它压根不查重复历史）。
    现在 `play()` 与 `get_legal_moves()` 判**同一套**：占点 / 禁自杀 / PSK，pass 豁免。
    `ko_point` 在两条路径上都不参与判罚，纯粹是信息位。

    （**另一条同样可行的收法，本任务没选**：把成劫判据收窄成「那块必须是单子」，让
    `ko_point` 只标记经典劫。它与掩码的 PSK 完全重合、不再有任何增量严格性，但会在
    引擎里留下一条 TT 没有的禁令 —— 与「判据只有 PSK 一条」这个更干净的立场冲突。）

    ---- PASS：类级常量，值 = board_size * board_size ----
    `GoBoard.PASS` 是**动作空间**里 pass 的编号（`board_size * board_size`），
    它是描述符而不是普通 int —— 因为值随盘口变，见 `_PassSlot` 的 docstring。
    读走实例、写入删除一律报错（`__get__` 只在实例上给值，`__set__` / `__delete__`
    抛 `AttributeError`），所以「类上取」与「实例上写」两种错法都是**结构性地**
    出声，而不是静默分叉。
    动作空间与 `play()` 的「棋盘方言」（pass = -1）是**两套编号**，
    `legal_actions()` / `is_legal()` 只认动作空间；详见动作空间段的注释。
    """

    # 动作空间的 PASS 槽（描述符：值 = board_size * board_size，见 _PassSlot）
    PASS = _PassSlot()

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
        # 「上一手是否形成简单劫」的只读派生量，**不参与合法性判罚**（PSK 已覆盖），
        # 详见类 docstring 的专门段落。
        self.ko_point = -1       # 上一手形成的简单劫：被提子的扁平坐标，-1 表示无
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
        # 棋块/气表（P2.7a）：**容器**浅拷贝、**值是不可变元组** -> 未修改的块
        # 跨克隆体零拷贝共享，修改走写时复制。这是「MCTS 每个候选一次 clone()」
        # 能付得起的关键：深拷贝整张块表会让每个候选都付 O(#块) 的重建。
        nb._groups = dict(self._groups) if self._groups is not None else None
        nb._gid = self._gid.copy() if self._gid is not None else None
        nb._groups_valid = self._groups_valid
        # 克隆体的撤销栈从空开始，所以「未记录落子」的计数也从 0 起算
        nb._unrecorded_seq = 0
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

    def get_legal_moves(self) -> np.ndarray:
        """返回长度为 size*size 的 bool 掩码，True 表示该点可落子。

        口径 = **Tromp-Taylor 的完整合法点集合，外加一条有意偏离（禁自杀）**：
        TT 规则 6 只有一句「A turn is either a pass; or a move that doesn't repeat an
        earlier grid coloring.」，本方法的第 3 条就是它，一字不多；而第 2 条禁自杀是
        **相对 TT 的有意偏离**（TT 规则本身允许自提，见模块头）。

        **合法性判定唯一的真相源**。判三条（`ko_point` 不再单独判罚，见类 docstring）：

          1. **空点**；
          2. **禁自杀**（⚠ 相对 TT 的有意偏离）：落子后己方连通块（落点 + 邻接同色块
             合并）总气 ≥ 1，**或者**该手提掉了敌子。**绝不原地改 `self.board` 做模拟** ——
             判定只用邻接棋块的气数，见下面「自杀判定的等价变形」；
          3. **位置超级劫（PSK）** = TT 规则 6：落子后的染色若命中历史集合则非法。

        P2.7a 起第 2 条走**增量棋块/气表**（O(1) 查气）而不是对邻块做 flood fill；
        第 3 条的两段式与临时落子的历史见下面那段说明。

        **顺序：先禁自杀，再判 PSK。** 这是正确性要求而不是性能偏好：
        `position_hash_after_move()` 内部会**临时写 `self.board[r, c]`** 来推演提子，
        所以绝不能在「盘上已经放了子」的状态里调它。自杀点压根没有「落子后的染色」，
        对它算候选键也没有意义。

        ✅ **本掩码就是完整的 TT 合法点集合**（除上面那条声明的禁自杀偏离之外），
        而 `play()` 判的**也是同一套**（P2.6a-2c 起：占点 / 禁自杀 / PSK，pass 豁免），
        所以两条路径对**任意手**给出**相同答案**。
        ⚠ 历史上这里有过一处偏差（`play()` 多一道 TT 没有的
        `if move == self.ko_point: return False` 劫禁，导致「多子提子形成的劫」的立即
        回提被掩码放行、却被 `play()` 拒绝，实测 9 路随机自对弈约 1.2% 的取点撞上），
        已随 P2.6a-2c 删除。**别再把任何一条 TT 之外的禁令加回 `play()`**，那会让本方法
        重新变成「两套判据」；逐点对拍的锁是
        `tests/test_go_rules_legality.py::test_play_and_mask_agree_on_every_move`。

        ---- 自杀判定的等价变形（不写盘） ----
        落点 (r, c) 落子**前**是空点，因此它**必然**落在每个相邻块的气里。据此：
          - 邻接**敌**块气数 == 1  ⟹ 它的气全被落点占掉 ⟹ 该手**提子** ⟹ 合法
            （提子后落点旁边必然空出被提子的位置，落子方有气）；
          - 邻接**同色**块气数 ≥ 2  ⟹ 除落点外还有别的气 ⟹ 合并后己方有气 ⟹ 合法；
          - 落点有**空的**正交邻点 ⟹ 落子方有气 ⟹ 合法；
          - 三条都不成立 ⟹ 落子后己方块无气且未提子 ⟹ **自杀，非法**。
        「气数 ≥ 2」而不是「有气」是关键：同色邻块那唯一的一口气**就是落点本身**，
        落上去等于自己把那口气堵死。只判「有气」会把这类点错放成合法。

        ---- PSK 的两段式（掩码被 MCTS 与自对弈每手调用，必须压成本） ----
          1. 廉价判据：该候选是否**提子**（上面已经算出来了，不额外花钱）；
          2. **不提子**的候选走**纯算术**：候选染色 = 落点从「空」变「己方子」，
             不提子就没有别的染色变化，而 position 键只含棋盘染色（连行棋方翻转都
             不影响），所以 `position_hash() ^ _zobrist_key(r, c, color)` 就是候选键 ——
             **O(1)，不动棋盘**；
             （P2.7a 之前这条也是「不动棋盘」的，但**提子**那一支要靠
             `position_hash_after_move()` 临时落子再还原；现在那一支也不落子了。）
          3. **提子**的候选才走 `position_hash_after_move(mv)` 的完整推演（数量极少）；
          4. 命中历史 → 该点置 False。
        ⚠ **不许**用「只有提子才可能违反 PSK」这种剪枝：提子与否与「是否复现历史染色」
        无关，P2.6a-1 已给出理论反例，至今未被证明或推翻。
        ⚠ 候选键必须是 **position-only** 口径。喂 `hash_after_move(mv)`（含行棋方 =
        SSK）会**静默**查不中历史，表现为「超级劫手被判合法」。

        ---- pass ----
        **本掩码没有 PASS 槽**（长度恒为 n*n），所以 pass 天然不进入 PSK 判定 ——
        这就是 TT 规则 6「禁的是重复棋、不是 pass」在这一层的落法，也正是
        简报要求「不要在这里写 `if mv == -1: continue`」的原因：那在长度 n*n 的掩码里
        是死代码。动作空间层（`legal_actions()` / MCTS 的候选表）要把 PASS 作为**永远
        合法**的动作显式加进去（见 P2.6a-2b-2）。

        ---- 状态变更收口 ----
        `_ensure_hash()` 收在**函数入口、缓存检查之前**。它可能触发
        `_adopt_as_new_game()` —— 那是一次状态变更（重置 `move_number` / `passes` /
        `_undo_stack` 并把 `_legal_cache` 置 None）。本方法被文档描述为只读且带缓存，
        绝不能在扫描中途发生这种变更，所以它必须排在扫描之前。

        ⚠ **必须排在缓存检查之前**（顺序反了会拿到过期掩码）：`_adopt_as_new_game()`
        会失效缓存，所以「先缓存检查、后 `_ensure_hash()`」在外部换掉 `board` 数组之后
        会让第一次调用**直接返回属于旧盘面的掩码** —— 哈希已被重置、掩码还是旧的，
        那是最糟的组合（比单纯的过期掩码更难查）。当前顺序下，采纳必然把缓存清掉、
        紧接着的缓存检查必然不命中，于是掩码一定是在接管之后算出来的。

        结果缓存在 `_legal_cache`，由 `play()` / `undo()` / `reset()` /
        `_adopt_as_new_game()` 失效。
        """
        # 入口收口：可能触发 _adopt_as_new_game()（状态变更），必须在扫描开始前发生，
        # 也必须在缓存检查之前 —— 采纳会失效缓存，顺序反了就会返回属于旧盘面的过期掩码。
        self._ensure_hash()

        if self._legal_cache is not None:
            return self._legal_cache

        n = self.board_size
        color = self.current_player
        legal = (self.board == 0).reshape(-1)
        if legal.any():
            # 读路径用 **Python int 列表快照**，不用 numpy 数组：逐候选 4 邻域扫描里
            # `board[nr, nc] == 0` 会造 numpy 标量再取 __bool__，实测 19 路空盘一次
            # 掩码 2.53 ms → 0.26 ms（10x），而 tolist() 本身只要 0.003 ms。
            # 掩码是热路径（MCTS 每个展开节点、自对弈每手都要），这个便宜必须占。
            # 快照与 numpy 数组在本方法内**始终一致**：扫描过程不写盘，而
            # position_hash_after_move() 只读盘面（P2.7a 起它连临时落子都没有了）。
            board = self.board.tolist()
            # 局部绑定：热循环里逐候选调用，属性查找与 _ensure_hash 的重复调用都省掉。
            # PSK 历史集合与 position 键在整轮扫描里不变（扫描不写盘、不换行棋方），
            # 所以 `cand in counts` 与 `_would_repeat(cand)` **完全等价**。
            counts = self._pos_hash_counts
            pos_key = self._pos_zobrist
            # P2.7a：气的查询走增量表（O(1)），不再对每个邻块做一次 flood fill。
            # `_ensure_groups()` 排在最前：adopt 之后表可能是失效的。
            # `_gid.tolist()` 一次（19 路 0.003 ms）换掉内层成千上万次 numpy
            # 标量构造 —— 与上面 `board.tolist()` 同一个理由。
            self._ensure_groups()
            gids = self._gid.tolist()
            groups = self._groups
            excl = self._libs_excluding_count
            zkey = _zobrist_key

            for i in np.flatnonzero(legal).tolist():
                r, c = divmod(i, n)
                # ---- 2. 禁自杀 + 提子判据：一次邻域扫描同时得到两件事 ----
                capture = False
                liberty = False
                for dr, dc in _NB4:
                    nr = r + dr
                    nc = c + dc
                    if nr < 0 or nr >= n or nc < 0 or nc >= n:
                        continue
                    v = board[nr][nc]
                    if v == 0:
                        liberty = True                  # 落点自身的气
                    else:
                        # 落点此刻是空点 -> 它必然在这个邻块的气里，于是
                        # 「除落点外的气数」一个整数就同时回答己方（非自杀）
                        # 与敌方（提子）两个问题。
                        g = gids[nr][nc]
                        if g < 0:
                            raise _groups_desync_error(nr, nc)
                        rest = excl(groups[g][1], i)
                        if v == color:
                            if rest > 0:
                                liberty = True
                        elif rest == 0:
                            capture = True              # -> 提子
                if not capture and not liberty:
                    legal[i] = False
                    continue

                # ---- 3. PSK（两段式；必须在自杀判定之后，见本方法 docstring）----
                if capture:
                    cand = self.position_hash_after_move(i)
                else:
                    cand = pos_key ^ zkey(r, c, color)
                if cand in counts:
                    legal[i] = False

        self._legal_cache = legal.copy()
        return legal

    # ---- 动作空间（OpenSpiel 风格）------------------------------------------
    #
    # ⚠⚠ **三套编码并存，混用会静默出错 —— 动手前先读完这一段** ⚠⚠
    #
    #   | 编码            | 取值                     | pass      | 谁在用                                   |
    #   |-----------------|--------------------------|-----------|------------------------------------------|
    #   | 棋盘方言        | 0 .. n*n-1               | **-1**    | `play()` / `move_history` / `undo_stack` |
    #   |                 |                          |           | `feature_planes` 的历史通道 / MCTS 的    |
    #   |                 |                          |           | `path_moves` / webui 的 `PASS = -1`       |
    #   | 动作空间（本段）| 0 .. n*n-1，**外加 n*n** | **n*n**   | `PASS` / `num_actions()` /               |
    #   |                 |                          |           | `legal_actions()` / `is_legal()` /       |
    #   |                 |                          |           | MCTS 的树内编码（`n_actions - 1`）       |
    #   | 掩码            | 长度恒为 n*n 的 bool 数组 | **无槽位**| `get_legal_moves()` / 通道 8             |
    #
    # 三个必然踩到的坑，全部**由结构保证**而不是靠调用方记得：
    #   - `is_legal(-1)` → **False**（-1 在动作空间里越界），而 `play(-1)` → **True**（合法的
    #     pass）。两者**不可比**，不要拿 `is_legal(x)` 的结果去预判 `play(x)`；
    #   - `is_legal(PASS)` → **True**，而 `play(PASS)` → **False**（n*n 越界，pass 在
    #     `play()` 里是 -1）。要落一个动作必须先换算：`play(-1 if a == PASS else a)`；
    #   - `legal_actions()` 里的 PASS 是**无条件追加**的，不查 PSK（理由见下）。
    #
    # 为什么动作空间要把 PASS 放进编号里：这是 OpenSpiel 的约定（`num_actions()` 覆盖
    # 全部动作，策略网络的输出维度因此是 n*n+1，掩码类接口不需要为 pass 开后门），
    # 本仓库的 MCTS 早就按这个约定写死了 `n_actions = n*n+1`（见 `src/search/mcts.py`），
    # 本段只是把那套**已经存在但只存在于调用点里**的约定收进引擎，给出唯一名字与
    # 换算函数。它**没有**改变任何既有调用点的行为。

    def num_actions(self) -> int:
        """动作空间大小 = `board_size * board_size + 1`（末位是 PASS）。

        与 `PASS` 同源（`num_actions() == PASS + 1`），所以「动作编号上界」永远
        不用在调用点各写一遍 `bs*bs + 1`（`MCTS.n_actions` / `inference.choose_move` /
        webui / rollout 都是同一份）。策略网络的输出维度也等于本值。
        """
        return self.board_size * self.board_size + 1

    def action_to_coord(self, action: int) -> tuple:
        """动作编号 → 棋盘坐标 `(r, c)`。**PASS 不是坐标**，对它抛 `ValueError`。

        只接受 `0 <= action < PASS`（即 `0 .. n*n-1`）。越界（含负数、`PASS`、
        `>= num_actions()`）一律 `ValueError` —— 这里**不**返回 `None`：
        返回 `None` 会让 `r, c = board.action_to_coord(a)` 抛一句
        「cannot unpack non-sequence」或者更糟的 NoneType 栈，调用方根本看不出
        自己漏了 `a == PASS` 的分支；显式异常把「pass 没有坐标」这件事说清楚。

        `PASS` 请走 `action_to_string()`，落子请走 `play(-1)`。
        """
        n = self.board_size
        if action < 0 or action >= n * n:
            raise ValueError(
                f"action={action} 不是落子动作：落子动作范围是 [0, {n * n})，"
                f"而 PASS={self.PASS}（=n*n）没有坐标。pass 用 action_to_string()"
                f"（返回 'pass'），落子用 play(-1)。")
        return divmod(action, n)

    def coord_to_action(self, r: int, c: int) -> int:
        """棋盘坐标 `(r, c)` → 动作编号 `r * board_size + c`。越界抛 `ValueError`。

        与 `action_to_coord` 严格互逆（`coord_to_action(*action_to_coord(a)) == a`
        对全部 `a < PASS` 成立）。编号用 `r * n + c` 而不是 SGF 的 `c * n + r`，
        与 `move_history` / `apply_symmetry_batch` / 掩码下标**同一套**；
        只有**文本**记法是 SGF 风格（列字母在前），见 `action_to_string`。
        """
        n = self.board_size
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError(
                f"坐标 ({r}, {c}) 越界：{n} 路盘的合法坐标是 0..{n - 1}。")
        return r * n + c

    def action_to_string(self, action: int) -> str:
        """动作编号 → 文本记法。`PASS` → `'pass'`，落子 → **两字母** SGF 风格串。

        **记法只有一套，就是 `parse_move_str` 那一套**（先读它，别造第二套）：
        小写字母，**列字母在前、行字母在后**（SGF 约定），`'aa'` = (行 0, 列 0) = 角，
        `'ee'` = 天元。`action_to_string(a) == parse_move_str` 的逆，
        `string_to_action(action_to_string(a)) == a` 对全部合法 a 成立。

        ⚠ **「小写字母」只在 `board_size <= 26` 时为真**：本实现按
        `chr(ord('a') + 索引)` 造字母（无查表，所以没有字母表可越界），而
        `__init__` 允许 `board_size` 到 `_ZOBRIST_STRIDE = 32`（Zobrist 坐标表的
        覆盖上限）—— 于是 27 路起越出 `a..z`：第 27..30 行/列落成 `{ | } ~`，
        31/32 路落成 `\x7f` / `\x80`（**连可打印都不是**，别拿去写 SGF 或 JSON）。
        **往返仍然成立**（`parse_move_str` 用的是同一套算术，两边同进同出，已实测
        26/27/30/31/32 路往返一致），所以这不是错误、只是用词要准确：别拿
        `s.isalpha()` / `s.islower()` 当落子串的合法性判据（判据用
        `string_to_action`），也别照着 `parse_move_str` 自己的 docstring
        （「统一用小写字母 a-s」）去推 26 以上的盘口。
        在用盘口（5/7/9/13/19）都远在 26 以内。

        越界（含 `-1`）抛 `ValueError`：`-1` **不是**动作空间的编码（见段首的表）。
        """
        if action == self.PASS:
            return "pass"
        n = self.board_size
        if action < 0 or action >= n * n:
            raise ValueError(
                f"action={action} 越界：合法动作是 [0, {n * n}]，末位 {self.PASS} "
                f"是 PASS（棋盘方言里的 pass 是 -1，不是动作编号）。")
        r, c = divmod(action, n)
        return chr(ord('a') + c) + chr(ord('a') + r)

    def string_to_action(self, s: str) -> int:
        """文本记法 → 动作编号。`parse_move_str` 的**严格版**（非法输入抛异常）。

        与 `parse_move_str` 的唯一区别是**失败处理**，接受的记法完全相同
        （`'pass'` / `'resign'` / `''` → PASS；两字母串 → `r*n+c`；SGF 风格、列在前）。
        这里抛 `ValueError` 而不是返回 `(ok, mv)`，因为动作空间层的调用方要的是
        「一个动作编号」，`ok=False` 那种 `(False, -1)` 的返回值会把 -1 混进
        动作编号里 —— 那正是段首表里最危险的那一格。

        ⚠ **`'pass'` 映射到 `PASS`（n*n），不是 -1**。`-1` 留在棋盘方言里。

        接受的记法与 `action_to_string` **同一个 26 的天花板**（那里写全了，
        含 31/32 路会落到不可打印字符这件事）：`board_size > 26` 时两字母串的字符
        会越出 `a..z`，两边仍然严格互逆。
        """
        ok, mv = self.parse_move_str(s)
        if not ok:
            raise ValueError(
                f"{s!r} 不是合法的着法串（记法见 action_to_string：'aa'=左上角、"
                f"'pass'=虚着，{self.board_size} 路用 a.."
                f"{chr(ord('a') + self.board_size - 1)}）")
        return self.PASS if mv == -1 else mv

    def legal_actions(self) -> list:
        """全部合法动作的**升序**列表 = 掩码里的真点 + 末位 `PASS`（**无条件**）。

        掩码长度为 n*n 且**没有** PASS 槽，所以 pass 天然不进 PSK 判定；
        本方法把 PASS 作为**永远合法**的动作显式追加到末位。因为 PASS = n*n 大于
        掩码里任何下标，「掩码下标升序 + 末尾追加」**天然就是升序**，不需要再排一次。

        ⚠ **追加是无条件的，且绝不可改成「对 PASS 查一次 PSK」**：
        pass 不改变棋盘染色，所以 `position_hash_after_move(-1)` 恒等于当前键，
        而当前键必然已在重复局面历史里（每次落子都记它）—— 谓词**恒为真**。
        一旦接上，所有 pass 都会被判非法，对局将**永远无法终局**（终局 = 两次连续 pass）。
        `is_legal()` 里对 PASS 的短路与本行的无条件追加是同一件事的两面。

        只读**意图**（不落子、不改行棋方），但入口同样过 `_ensure_hash()`：PSK 判定读
        的是重复局面历史，历史若属于别的盘面，命中判定毫无意义（这与掩码的契约完全
        一致，调用由掩码那一侧代劳）。
        ⚠ **别把它读成「绝不改状态」**：`_ensure_hash()` 可能触发
        `_adopt_as_new_game()`，那是一次实打实的状态变更（重置 `move_number` /
        `passes` / `_undo_stack` **并把 `_legal_cache` 置 None**）。措辞与
        `is_legal` 的同一句保持一致。

        **不适用于 MCTS 热路径**：那里要的是 `np.flatnonzero(mask)` 之后接
        `n_actions - 1`，省掉中间的 Python list。
        """
        mask = self.get_legal_moves()
        return [int(i) for i in np.flatnonzero(mask).tolist()] + [self.PASS]

    def is_legal(self, action: int) -> bool:
        """单个动作是否合法 —— **单点判定，绝不物化全掩码**。

        判据与 `get_legal_moves()` 掩码**逐条同源、同一顺序**（空点 / 禁自杀 / PSK，
        见掩码 docstring 的等价变形与两段式），所以对任意 `0 <= action < n*n`：
        `is_legal(a) == bool(get_legal_moves()[a])`。掩码是合法点集合的**唯一**真相源，
        本方法是它对单个点的投影（O(1) 级邻域扫描 + 至多一次提子推演），
        逐点对拍由 `tests/test_go_action_api.py::test_is_legal_matches_mask` 钉住。

        ⚠ **上面那条等价是有前提的，前提就是类 docstring 的「局面状态字段的读写
        契约」**：**就地改写 `board.board` 之后必须调 `resync_hash()`**。
        `_ensure_hash()` 只侦测「board **数组对象**被换掉」与「current_player 被改写」
        两种脱钩，**察觉不到就地改写**，于是增量哈希仍属于旧盘面：带缓存的
        `get_legal_moves()` / `legal_actions()`（旧掩码）与现算的 `is_legal`（新盘面）
        可能给出不同答案 —— 而**此时 `is_legal` 才是对的那个**（它直接读 `self.board`）。
        所以看到两者不一致时，先查有没有漏 `resync_hash()`，别急着判 `is_legal` 有 bug。
        本方法的 docstring 刻意把这条写在这里而不是只放在 `_ensure_hash` 里：
        契约的**受益方**是这里那条等价声明。

        边界：
          - `action == PASS` → **True**（结构保证；理由见 `legal_actions` 的 ⚠ 段）；
          - `action` 越界（含 `-1` 与 `>= num_actions()`）→ **False**，**不抛**。
            这是「问一个非法编号合不合法」的答案，与 `action_to_coord` 的「把非法编号
            换算成坐标」不同 —— 后者才抛。
          - 占点 → False（不落任何子）。

        ⚠ **绝不写成 `return bool(self.get_legal_moves()[action])`**：那会把 `is_legal`
          退化成「每次调用都物化整张 n*n 掩码」，单点 O(1) 的 API 变成 O(n²)，
          在按候选调用的场景（UI 逐点问、rollout 采样）上是数量级的差别。
          `tests/test_go_action_api.py::test_is_legal_does_not_materialize_the_mask`
          用一个「被调用就抛异常」的 `get_legal_moves` 桩把这条钉死。

        **成本**：与盘面**面积无关**，也**与邻块大小无关**（P2.7a 起）：
        ≤4 次查表（`len(libs) - (落点在不在里面)`）+ 至多一次提子推演，
        全部与 n² 和块的大小无关。
        实测 19 路随机自对弈中盘（180 手后、179 个合法点）：
        一次**冷**掩码 0.24 ms，而本方法逐点判定 ≈ 5.2 µs/点 —— 约 1/46。
        （P2.7a 之前这里是「≤4 个邻块各一次带早退的 flood fill」，19 路 23 µs/点、
        冷掩码 0.66 ms；增量棋块/气表把两者都换成了查表。）
        ⚠ 别拿「掩码有缓存时按位取值只要 0.5 µs」来比：那不是本方法的工作量。
        本方法**不读** `_legal_cache`：复用缓存当然更快，但那样「绝不物化全掩码」
        这条保证就只在缓存冷时成立（依赖调用顺序），而本方法的成本模型要的是
        「与 n² 无关」这个无条件性质。⚠ 这条性质有个前提：表在 adopt 之后是失效的，
        第一次调用要付一次 O(n²) 重建（19 路 0.7 ms，正常对局**每局一次**）——
        那次重建由 `_ensure_hash()` 触发的采纳一并决定，不在本方法的 O(1) 承诺之内，
        因为它属于「换盘面」这个显式动作，而不是「问一个点」。
        ⚠ 同一节开头那句「就地改写 board 后 `is_legal` 才是对的那个」在本表之后要补
        一个前提：违反契约现在会**抛** `RuntimeError`（`_groups_desync_error`）而不是
        给答案 —— 查表需要块号，而陈旧表里没有。修法不变：`resync_hash()`。

        ⚠ **`-1` 在动作空间里越界**，所以 `is_legal(-1) is False`；而 `play(-1)`
          是**合法**的 pass（棋盘方言）。两者**不可比**，别拿本方法的返回值去预判
          `play()` 的返回值 —— `is_legal(PASS)` 为真而 `play(PASS)` 也为假正是同一个
          道理（pass 在 `play()` 里是 -1）。落子请先换算：
          `play(-1 if a == self.PASS else a)`。

        只读，但入口同样过 `_ensure_hash()`：PSK 判定读的是重复局面历史，
        历史若属于别的盘面，命中判定毫无意义（这与掩码的契约完全一致）。
        """
        n = self.board_size
        self._ensure_hash()
        if action == self.PASS:
            return True
        if action < 0 or action >= n * n:
            return False
        r, c = divmod(action, n)
        color = self.current_player
        if self.board[r, c] != 0:
            return False
        # 判据与掩码逐字同源（get_legal_moves docstring 的「自杀判定的等价变形」）：
        # 落点落子前是空点 -> 它必然落在每个相邻块的气里，于是「除落点外还有没有气」
        # 一个布尔同时回答「己方合并后有气吗」与「敌块是不是被打吃」。
        # 这里**不**取 tolist() 快照：单点判定只读 ≤4 个邻点、≤4 个块记录，
        # 整盘快照是 O(n²) 的无用拷贝（掩码需要它是因为要扫全盘）。
        # `_gid` 同理：只逐点索引 4 次，numpy 标量的那点开销远小于一次 O(n²) 拷贝。
        # ⚠ **与掩码的差异是有意的取舍，不是等价，也不是漏改**：掩码走
        # `self.board.tolist()` 是因为它逐候选扫全盘 n*n 次邻域，那里 numpy 标量
        # 每次比较都要造一个标量再取 __bool__，实测 19 路空盘一次掩码
        # 2.53 ms → 0.26 ms（10x，见掩码 docstring）。本方法每点只碰 ≤4 个邻点，
        # 一次 tolist() 的 O(n²) 拷贝**远超**那点省下来的开销，而本方法对外承诺的
        # 是「与 n² 无关」（见上文成本段）—— 换成 tolist() 会把这个无条件性质悄悄
        # 变成 O(n²)。正确性两边一致（`board[nr, nc] == 0` 对 Python int 与
        # np.int8 同义，查表也只做整数比较），所以**别顺手「对齐」**。
        # P2.7a：气的查询走增量表（O(1) 查表，不做 flood fill）。
        # ⚠ 「与 n² 无关」这条对外承诺**照样成立**：`_ensure_groups()` 只在表
        # 失效时才重建（adopt 之后，或撤销链被 record=False 打断之后），
        # 而重建是 O(n²) 但通常一局只发生一次；稳态下本方法只碰 <=4 个邻点。
        self._ensure_groups()
        groups = self._groups
        gid_of = self._gid
        capture = False
        liberty = False
        for dr, dc in _NB4:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n:
                continue
            v = int(self.board[nr, nc])
            if v == 0:
                liberty = True               # 落点自身的气
                continue
            g = int(gid_of[nr, nc])
            if g < 0:
                raise _groups_desync_error(nr, nc)
            rest = self._libs_excluding_count(groups[g][1], action)
            if v == color:
                if rest > 0:                 # 同色邻块除落点外还有气
                    liberty = True
            elif rest == 0:                  # 敌块的气全被落点占掉 -> 提子
                capture = True
        if not capture and not liberty:
            return False                      # 自杀（⚠ 相对 TT 的有意偏离）
        # PSK（TT 规则 6），两段式与掩码相同：提子才走完整推演，不提子走纯算术。
        # 历史集合直接查 `_pos_hash_counts`（与掩码同一个理由，见 `_would_repeat` 的
        # docstring）：入口已过 `_ensure_hash()`，扫描期间不写盘也不换行棋方，
        # 所以两种写法完全等价。
        if capture:
            cand = self.position_hash_after_move(action)
        else:
            cand = self._pos_zobrist ^ _zobrist_key(r, c, color)
        return cand not in self._pos_hash_counts

    # ---- 连通块 / 气 -------------------------------------------------------
    #
    # ⚠ 本节有**两套**棋块查询，别混用：
    #   - `_groups` / `_gid`（P2.7a 增量表）：**热路径**用这个，查气 O(1)；
    #   - `_neighbor_groups` / `_group_has_liberty*` / `_group_liberty_count`
    #     （flood fill）：**参考实现**，留给测试当 oracle，别再挂进热路径。
    # 踩过的坑：「惰性重建表」那版正是把热路径换成查表，却让每个候选
    # `clone()+play()` 都付一次 O(n²) 全盘重建（19 路 87 µs → 2080 µs），
    # 掩码也跟着从 0.66 ms 涨到 2.19 ms。教训全文见
    # `.superpowers/sdd/2026-09-25-v21-roadmap/task-p2-7a-negative-result.md`。

    # ---- 增量棋块/气表（P2.7a） -------------------------------------------
    #
    # 三件套，缺一不可：
    #   self._gid       (n,n) int32 数组，每点的块号；空点 = -1
    #   self._groups    {块号: (颜色, frozenset(气), frozenset(棋子))}
    #   self._group_undo 与 _undo_stack 同步的撤销记录栈
    #
    # **块号 = 该块里那颗「扫描到的第一个点」的扁平坐标**，因此块号天然唯一
    # （一个点同一时刻只属于一个块）且在表被重建后可能改变 —— 这没问题，
    # 因为表内从不留悬挂引用：撤销记录里的块号只活到它自己那次 undo，
    # 而任何重建都发生在 `_adopt_as_new_game()`，它会连 `_undo_stack` 一起清空。
    #
    # **值一律不可变**（tuple + frozenset）：`clone()` 因此只需 `dict(...)`
    # 浅拷贝；任何修改都换新元组（写时复制），共享者永远看不到变化。

    def _ensure_groups(self) -> None:
        """保证增量表与盘面一致（不一致才按盘面全量 flood fill 重建）。

        失效**只有一个来源**：`_adopt_as_new_game()`（reset / 外部换 board 数组 /
        外部改 current_player / resync_hash）。`play()` 与 `undo()` 走增量路径，
        永不置失效。

        ⚠ **惰性重建的陷阱正在这里**：本方法看着安全，但它把成本从「查询时」挪到
        「变更后第一次查询」，而 MCTS 是变更远多于查询的形态。所以表**必须**在
        `play()` 入口就备好 —— 只有 adopt 之后才付这一次全盘重建。
        """
        if self._groups_valid:
            return
        n = self.board_size
        board = self.board.tolist()
        gid = [[-1] * n for _ in range(n)]
        groups = {}
        for r0 in range(n):
            row = board[r0]
            for c0 in range(n):
                v = row[c0]
                if v == 0 or gid[r0][c0] >= 0:
                    continue
                g = r0 * n + c0
                stack = [(r0, c0)]
                gid[r0][c0] = g
                libs = set()
                stones = set()
                while stack:
                    r, c = stack.pop()
                    stones.add(r * n + c)
                    for dr, dc in _NB4:
                        nr = r + dr
                        nc = c + dc
                        if nr < 0 or nr >= n or nc < 0 or nc >= n:
                            continue
                        w = board[nr][nc]
                        if w == 0:
                            libs.add(nr * n + nc)
                        elif w == v and gid[nr][nc] < 0:
                            gid[nr][nc] = g
                            stack.append((nr, nc))
                groups[g] = (v, frozenset(libs), frozenset(stones))
        self._gid = np.array(gid, dtype=np.int32)
        self._groups = groups
        self._groups_valid = True

    def _libs_excluding_count(self, libs, point):
        """`libs`（frozenset）中除 `point` 之外的气数 —— 「这个块除落点外还有气吗」。

        O(1)（集合大小 + 一次成员判定），替代原来的「整块 flood fill 数气」。
        掩码 docstring「自杀判定的等价变形」用的正是这个量：己方邻块 >=1 -> 非自杀，
        敌块 ==0 -> 提子。
        """
        return len(libs) - (1 if point in libs else 0)

    def _update_groups_after_move(self, move, color, own_gids, cap_gids, neighbor_gids):
        """把「落子 + 提子」增量写进棋块表，并返回可回滚的记录。

        必须在**盘面已经改完**（落点放了子、被提子删了）之后调用：新气要算给
        周围**还活着**的块，而被提点的块号要在这一步复位成 -1。

        只碰这些块（与盘面大小**无关**）：
          - 落点所在的新块：气 = 各同色邻块的气的并集，剔掉落点，再加落点自己的空邻点；
          - 未被提的敌邻块：各少一口气（= 落点）；
          - 被提的敌块：整块删除，其每颗子的**存活邻居**各多一口气（= 该被提点）。
        返回 `(old_values, points, old_gids, new_gids)`：
          old_values  {块号: 旧值}，值为 None 表示该块本次才新建（undo 时删掉）
          points/old_gids/new_gids  gid 数组里被改写的点及其改写前后的块号
        """
        n = self.board_size
        r, c = divmod(move, n)
        g_new = move                # 新块号取落点：它就是这块里的一颗子，天然唯一
        old_values = {}
        points = [move]
        old_gids = [int(self._gid[r, c])]
        new_gids = [g_new]

        # ---- 1) 落点新块：并入同色邻块的棋子与气 ----
        libs = set()
        stones = [move]
        for g in own_gids:
            ocol, olibs, ostones = self._groups[g]
            old_values[g] = self._groups[g]
            libs |= olibs
            for flat in ostones:
                stones.append(flat)
                points.append(flat)
                old_gids.append(g)
                new_gids.append(g_new)
            del self._groups[g]
        libs.discard(move)          # 落点不再是自己块的气
        for dr, dc in _NB4:         # 落点自己的空邻点成为新气（含刚被提掉的位置）
            nr = r + dr
            nc = c + dc
            if 0 <= nr < n and 0 <= nc < n and int(self.board[nr, nc]) == 0:
                libs.add(nr * n + nc)
        old_values[g_new] = None    # 本次新建
        self._groups[g_new] = (color, frozenset(libs), frozenset(stones))

        # ---- 2) 未被提的敌邻块：失去落点这口气 ----
        for g in neighbor_gids:
            if g == g_new or g in cap_gids or g not in self._groups:
                continue
            ocol, olibs, ostones = self._groups[g]
            if move not in olibs:
                continue
            if g not in old_values:
                old_values[g] = self._groups[g]
            self._groups[g] = (ocol, olibs - {move}, ostones)

        # ---- 3) 被提块：整块从表里摘掉 ----
        cap_stones = []
        for g in cap_gids:
            rec = self._groups[g]
            old_values[g] = rec
            del self._groups[g]
            cap_stones.extend(rec[2])
            for flat in rec[2]:
                points.append(flat)
                old_gids.append(g)
                new_gids.append(-1)

        # ---- 4) gid 数组先落定（合并后的新块号 / 被提点复位 -1）----
        # ⚠ **顺序是正确性要求**：第 5 步要按「存活邻居的块号」找块加气，
        #   而合并块的旧块号已经被删掉了。数组若还停在旧值，第 5 步查到的是
        #   一个已不存在的块号，于是**整段长气被静默跳过** —— 表现为合并块的
        #   气莫名少掉几个（实测：白方合并后的块少了一口气 (7,6)，
        #   而那口气是被提黑子的位置）。
        self._gid.reshape(-1)[np.array(points, dtype=np.intp)] = np.array(
            new_gids, dtype=np.int32)

        # ---- 5) 被提的每颗子：它现在的空位是周围存活块的新气 ----
        for flat in cap_stones:
            cr, cc = divmod(flat, n)
            for dr, dc in _NB4:
                nr = cr + dr
                nc = cc + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                if int(self.board[nr, nc]) == 0:
                    continue
                ng = int(self._gid[nr, nc])
                if ng < 0 or ng not in self._groups or flat in self._groups[ng][1]:
                    continue
                ncol, nlibs, nstones = self._groups[ng]
                if ng not in old_values:
                    old_values[ng] = self._groups[ng]
                self._groups[ng] = (ncol, nlibs | {flat}, nstones)
        return old_values, points, old_gids, new_gids

    def _restore_groups_after_move(self, record) -> None:
        """回滚一次 `_update_groups_after_move()`：把被碰过的块与块号原样换回去。

        因为块的值是**不可变元组**，「换回去」只是换引用，不需要任何拷贝 ——
        这就是撤销记录能只存旧值的原因。

        ⚠ **只在记录可信时调**（判据在 `undo()` 里）：`play(record=False)` 落子
        会打断「撤销 = 回退一手」的前提（这是**哈希那边早就存在**的同一条限制，
        本表只是继承了它），此时旧记录描述的已经不是当前表，必须改成作废重建。
        """
        old_values, points, old_gids, _new_gids = record
        for g, val in old_values.items():
            if val is None:
                self._groups.pop(g, None)
            else:
                self._groups[g] = val
        self._gid.reshape(-1)[np.array(points, dtype=np.intp)] = np.array(
            old_gids, dtype=np.int32)

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
        """判断 (seed_r, seed_c) 所在连通块是否还有气。

        ⚠ **P2.7a 起不在热路径上**：热路径走 `_groups` 增量表。本方法与
        `_neighbor_groups` / `_group_has_liberty_excluding` / `_group_liberty_count`
        一起**保留为参考实现（oracle）**，供测试对拍，别再挂进 MCTS 路径。
        """
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

    def _group_has_liberty_excluding(self, board, seed_r, seed_c, ex_r, ex_c) -> bool:
        """判断 board 上 (seed_r, seed_c) 所在同色连通块是否还有**除 (ex_r, ex_c) 以外**的气。

        只服务 `get_legal_moves()` 的禁自杀判定。落点 (ex_r, ex_c) 落子**前**是空点，
        所以它必然落在每个相邻块的气里；「这个块除落点外还有没有别的气」于是同时
        回答两个问题（详见 get_legal_moves 的 docstring）：
          - 己方邻块：除落点外还有气 ⟹ 落子合并后己方有气 ⟹ 非自杀；
          - 敌块：除落点外**没有**气 ⟹ 落子把它提掉 ⟹ 非自杀（提子后落点旁必空）。

        为什么**不**用 `_group_liberty_count` 判（那个也能答，判据是 >=2 / ==1）：
        气数必须数**完**所有气，没有早退；而这里要的正是「有没有」这一个布尔。
        实测差别很大：19 路中盘一次掩码 26.6 ms → 6.1 ms（早退把「大气块」压成 O(1)）。

        ⚠ **P2.7a 起本方法已不在热路径上**：`get_legal_moves()` / `is_legal()` /
        `play()` / `_forecast_delta()` 全部改用增量棋块/气表（O(1) 查表，
        19 路 5.2 µs/点），本方法**保留下来当 oracle** ——
        `tests/test_go_incremental_groups.py` 用测试内独立实现的 flood fill
        对拍整张表，另有测试直接拿本方法的结果当期望值。**别再挂进热路径**。

        `board` 是**读快照**（`GoBoard.get_legal_moves` 传的是 `self.board.tolist()`，
        逐格读比 numpy 标量快约 10x）。本方法只读它、不写；传 numpy 数组也能工作
        （`board[nr][nc]` 对两者都成立）。

        早退语义与 `_group_has_liberty` 一致：找到第一个「非落点」的气就返回 True。
        前提：(ex_r, ex_c) 在 board 上**必须仍是空点** —— 本方法只读棋盘、不落子，
        所以从 `get_legal_moves()` 调用时天然满足；绝不能在已落子的盘面上调它。
        """
        n = self.board_size
        color = board[seed_r][seed_c]
        stack = [(seed_r, seed_c)]
        seen = {(seed_r, seed_c)}
        while stack:
            r, c = stack.pop()
            for dr, dc in _NB4:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= n or nc < 0 or nc >= n:
                    continue
                v = board[nr][nc]
                if v == 0:
                    if nr != ex_r or nc != ex_c:
                        return True
                elif v == color and (nr, nc) not in seen:
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
          _legal_cache = None    盘面可能被换过，缓存的掩码属于旧盘面。
        保留（**局面描述**，调用方写了什么就是什么）：
          board、current_player、**ko_point**。ko_point 是「盘面 + 上一手」的性质而不是
          对局进度：它是「上一手是否形成简单劫」的信息位（见类 docstring），从该局面
          开新局时「上一手形成了简单劫」这件事照样成立，所以不能在这里清掉 ——
          light_rollout 显式拷贝 ko_point 要的就是这个语义。
          ⚠ 但要注意：历史被重建成 {当前局面} 之后，**提子之前那个染色不在历史里**，
          PSK 判不出那个回提点 —— 掩码与 `play()` 因此**一致地**放行它（两边同答案，
          不是分歧）。这是「绝不伪造父局历史」这条取舍的已知代价（宁可漏判重复），
          正常对局下历史完整，两者永远一致地拒它。

        重建：按盘面重算**两套**键，历史 = {当前染色的 position 键}（**绝不伪造父局历史**）。
        """
        self.passes = 0
        self.move_number = 0
        self.move_history = []
        self._undo_stack = []
        self._legal_cache = None
        # 棋块/气表（P2.7a）同样作废：这张表是**增量维护**的，旧表属于旧盘面。
        # ⚠ 撤销记录**不存在第二条栈**：它就存放在 `_undo_stack` 的每一项里
        #   （第 6 个元素），而上面刚把 `_undo_stack` 清空，旧记录因此无处可寻、
        #   天然作废 —— 不需要第二处生命周期，也就不可能出现两条栈错位。
        self._groups = None
        self._gid = None
        self._groups_valid = False
        self._unrecorded_seq = 0
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
        `is_repetition()` / `_would_repeat()` / `get_legal_moves()` 全部先调它。

        `get_legal_moves()` 自 P2.6a-2b-1 起也是入口（它要查 PSK 历史）。它虽然是
        「只读且带缓存」的方法，但**必须**把本调用放在函数入口、且排在它的缓存检查
        **之前**：本方法可能触发 `_adopt_as_new_game()`，那是一次状态变更（重置
        `move_number` / `passes` / `_undo_stack` 并把 `_legal_cache` 置 None），绝不能
        发生在一次掩码扫描的中间；而正因为采纳会失效缓存，把调用放在缓存检查之后就会
        让「外部换掉 board 数组后的第一次 `get_legal_moves()`」直接返回属于**旧盘面**
        的过期掩码。

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

        **不改盘面、不改对局状态**（P2.7a）：提子探测走增量棋块/气表，既不临时落子
        也不写棋盘，所以本方法**可重入**（唯一可能的写入是 `_ensure_groups()` 惰性
        重建那张**纯派生**的表，见其 docstring）。move == -1（pass）返回 0：
        pass 不改变染色。

        对**非法**着法（占点 / 自杀 / 重复）返回值是未定义的，调用方须先过合法性检查。
        """
        if move == -1:
            return 0
        n = self.board_size
        r, c = divmod(move, n)
        color = self.current_player
        opponent = -color
        # P2.7a：提子探测走增量表，**全程不写盘**（旧实现要临时落子再还原）。
        # 判据与旧实现逐条同义：落点此刻是空点、必然在被提块的气里，
        # 所以「该块被提掉」⟺「它的气恰好只有落点这一个」。
        self._ensure_groups()
        groups = self._groups
        delta = 0
        seen_gids = set()
        for dr, dc in _NB4:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n:
                continue
            if int(self.board[nr, nc]) != opponent:
                continue
            g = int(self._gid[nr, nc])
            if g < 0:
                raise _groups_desync_error(nr, nc)
            if g in seen_gids:      # 同一块可能被多个邻点触及，必须去重
                continue            # （不去重会把该块的钥匙异或两次、抵消掉）
            seen_gids.add(g)
            _ocol, olibs, ostones = groups[g]
            if len(olibs) == 1 and move in olibs:
                # 整块提掉：必须遍历**整块**的棋子，只扫邻域是不够的
                # （惰性重建那版就错在这里，见负面结论笔记）。
                for flat in ostones:
                    delta ^= _zobrist_key(flat // n, flat % n, opponent)
        delta ^= _zobrist_key(r, c, color)
        return delta

    def hash_after_move(self, move: int) -> int:
        """只读推演：若现在下 move（-1 = pass），落子后**通用指纹**的值。不改任何状态。

        与 `position_hash_after_move()` 的差别就是行棋方那一项，所以
        **重复判定不要用本方法**（含行棋方 = SSK 的候选键）。PSK 合法性请用
        `position_hash_after_move()`。

        只对合法着法（含 pass）有定义；占点 / 自杀 / 重复局面（PSK）等非法落子语义未定义，
        调用方须先过合法性检查（`get_legal_moves()` 掩码或 `play()` 的返回值）。
        **不改盘面**（P2.7a：`_forecast_delta()` 不再临时落子），故可重入。
        """
        self._ensure_hash()
        return self._zobrist ^ _ZOBRIST_TO_PLAY_XOR ^ self._forecast_delta(move)

    def position_hash_after_move(self, move: int) -> int:
        """只读推演：若现在下 move（-1 = pass），落子后**position 键**（仅染色）。

        **算 PSK 候选键只能用本方法**：
            `board._would_repeat(board.position_hash_after_move(mv))`
        不可改用 `hash_after_move(mv)`（那是含行棋方的通用指纹 = SSK；喂进去会
        **静默**查不中历史，表现为「超级劫手被判合法」）。

        move == -1（pass）返回**当前**键：pass 不改变染色，因此「pass 之后的染色」
        必然已经在历史里，谓词会返回真。**调用方必须把 pass 排除在 PSK 判定之外**
        （Tromp-Taylor 规则 6 禁的是重复棋，不是 pass；终局由两次连续 pass 表达），
        否则任何 pass 都会被误判非法。
        ✅ 两处豁免都已就位：`play()` 的 pass 分支在 PSK 判定**之前**就 return 了
        （P2.6a-2c）；`get_legal_moves()` 返回的掩码长度恒为 n*n、**没有 PASS 槽**，
        所以 pass 天然不进 PSK 判定。动作空间层（P2.6a-2b-2 的 `legal_actions()`、
        MCTS 的 `+ [n_actions - 1]` 候选表）负责把 PASS 作为**永远合法**的动作显式加回去
        —— 别在那里对 PASS 调本方法再查 `_would_repeat`，那会把 pass 全禁。
        ⚠ 不要为了「表达豁免」在 `get_legal_moves()` 里加 `if mv == -1: continue`
        —— 那个循环遍历的是 n*n 个点，`mv` 永远取不到 -1，是死代码。

        只对合法着法有定义；占点 / 自杀 / 重复局面（PSK）等非法落子语义未定义，调用方须先过
        合法性检查（`get_legal_moves()` 掩码或 `play()` 的返回值）。
        **不改盘面**（P2.7a：`_forecast_delta()` 不再临时落子，所以「盘面上已经放了子」
        这件事不再有任何技术后果）。但**语义**上仍要求传入的是「本手之前」的盘面 ——
        染色的定义就是如此。`get_legal_moves()` 的「先禁自杀、再判 PSK」顺序保留着，
        它保证推演发生在合法的候选上。
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

        `get_legal_moves()` 在热路径上**不调本方法**，而是直接查 `_pos_hash_counts`
        —— 那里已经在函数入口调过一次 `_ensure_hash()`，且扫描过程不写盘、不换行棋方，
        所以两种写法完全等价。改那段代码时别把这个前提弄丢。

        `play()` **调本方法**（P2.6a-2c 起，它判 PSK），但只在结构检查全部通过之后调用，
        所以它不在热路径的「每个候选」位置上：失败的候选在占点 / 自杀那一步就返回了。
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
        本次落子（含提子恢复 / `ko_point` / pass 计数 / 历史 / 执子方 / 两套 Zobrist 键
        （通用指纹 + 重复判定用的 position 键）/ 重复局面历史 / 落子数）。

        ---- 判定：与 `get_legal_moves()` 掩码**同一套**（P2.6a-2c 起）----
        拒的只有三类，判据、顺序、用的键都与掩码逐条一致：
          1. **占点** / **越界**（含 `move < -1`）；
          2. **禁自杀**（⚠ 相对 TT 的有意偏离）：落子后己方块无气且未提子；
          3. **位置超级劫（PSK）** = TT 规则 6：`position_hash_after_move(move)`
             命中重复局面历史。
        **顺序与掩码相同：先结构（1、2）后 PSK（3）。** 所以对**任意** `move ∈ [0, n*n]`
        本方法与 `bool(get_legal_moves()[move])` 必然同答案；逐点对拍由
        `tests/test_go_rules_legality.py::test_play_and_mask_agree_on_every_move` 钉住。
        ⚠ **没有第四条**：`ko_point` 不参与判罚（见类 docstring）。曾经存在的
        `if move == self.ko_point: return False` **不是 TT 规则**（TT 规则 6 全文只有
        「doesn't repeat an earlier grid coloring」，没有任何 ko-capture 从句），已删除 ——
        它比掩码更严，会让「多子提子形成的劫」的立即回提「掩码放行、本方法拒绝」
        （实测 1.2% 的取点）。⚠ **只删它是不够的**：本方法当时也不查 PSK，删掉之后会
        反向变宽松、连经典单子劫的立即回提都放行。补上的 PSK 检查与那条劫禁**必须同进同退**。

        **pass（`move == -1`）恒合法** —— 它在 PSK 判定**之前**就 return 了。豁免必须是
        显式的：pass 不改变染色，`position_hash_after_move(-1)` 恒等于当前键，而当前键
        必然已在历史里，查下去恒为「重复」。TT 规则 6 禁的是重复**棋**，不是 pass；
        终局由两次连续 pass 表达（`is_terminal()`）。

        ⚠ **本方法说棋盘方言（pass = -1），不是动作空间**（那里 pass = `PASS` = n*n）。
        `play(PASS)` 越界返回 False，`is_legal(-1)` 也是 False 而 `play(-1)` 是 True ——
        落子一个动作编号前必须换算 `play(-1 if a == self.PASS else a)`，
        见模块头「动作空间」段。

        判定失败时**不改变任何状态**（PSK 那一支的试落子已还原）。
        """
        n = self.board_size
        self._ensure_hash()
        if move == -1:
            # pass
            if record:
                # pass 不改盘面，所以第 6 位（表记录）是 None；第 7 位照记，
                # 让 undo() 能对「这条记录是不是可信」用同一把尺子量所有条目。
                self._undo_stack.append(
                    (-1, None, self.ko_point, self.passes, self.current_player,
                     None, self._unrecorded_seq))
            self.passes += 1
            self.ko_point = -1
            self.move_history.append(-1)
            self.move_number += 1
            self._commit_position(self.current_player, -1, None)
            self.current_player = -self.current_player
            self._legal_cache = None
            return True

        if move < 0 or move >= n * n:
            return False
        r, c = divmod(move, n)
        if self.board[r, c] != 0:
            return False

        color = self.current_player
        opponent = -color

        # ---- 结构判定：全走增量表，零 flood fill、**零试落子**（P2.7a）----
        #
        # 落点此刻是**空点**，因此它必然落在每个相邻块的气里 —— 这就是掩码
        # docstring「自杀判定的等价变形」那条不变式。于是一次 4 邻域扫描同时得到：
        #   own_gids  待合并的同色邻块
        #   cap_gids  唯一的气就是落点的敌块 -> 本手提掉它们
        #   liberty   落子后己方块除落点外还有气
        # ⚠ `_ensure_groups()` 排在最前：表失效（adopt 之后）必须先备好，
        # 否则每个候选都要付一次全盘重建 —— 那正是惰性重建版的致命处。
        self._ensure_groups()
        own_gids = []
        cap_gids = []
        seen_gids = set()
        liberty = False
        for dr, dc in _NB4:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n:
                continue
            v = int(self.board[nr, nc])
            if v == 0:
                liberty = True              # 落点自身的气
                continue
            g = int(self._gid[nr, nc])
            if g < 0:
                raise _groups_desync_error(nr, nc)
            if g in seen_gids:              # 同一块可能被多个邻点触及，必须去重
                continue
            seen_gids.add(g)
            rest = self._libs_excluding_count(self._groups[g][1], move)
            if v == color:
                own_gids.append(g)
                if rest > 0:              # 除落点外还有气 -> 合并后己方有气
                    liberty = True
            elif rest == 0:                # 敌块的气全被落点占掉 -> 提子
                cap_gids.append(g)
        # 检查自身是否还有气（禁自杀）
        if not cap_gids and not liberty:
            return False

        # 位置超级劫（TT 规则 6 = TT 规则的全文；与 get_legal_moves() **同一判据、同一键**）。
        # ⚠ 必须排在上面那批结构检查**之后**：`psk` 的历史只有「每一手成功落子之后」的
        #   染色，这一手还不算数；反过来，先判 PSK 就得为一个注定被拒的着法走完整的
        #   提子推演。与掩码「先禁自杀、再判 PSK」是同一条理由。
        # ⚠ 此刻盘面是**本手之前**的（结构判定已全部走表，一个子都没落），
        #   这正是 `position_hash_after_move()` 唯一允许的前置状态 ——
        #   P2.7a 之前这里是靠「落子 -> 判提子 -> 把子拿掉 -> 推演 -> 再放回去」
        #   的手工 dance 达成的，现在那套 dance 整个删掉了。
        if self._would_repeat(self.position_hash_after_move(move)):
            return False        # 盘面自始至终未改，无需还原

        # 到这里才真正改盘：落子 + 提子
        self.board[r, c] = color
        captured = []
        for g in cap_gids:
            for flat in self._groups[g][2]:
                cr, cc = divmod(flat, n)
                captured.append((cr, cc))
                self.board[cr, cc] = 0
        # 增量维护棋块/气表（P2.7a）。必须在提子**已从盘上删掉**之后：新气要算给
        # 周围还活着的块，而被提点的块号在这里复位成 -1。
        group_record = self._update_groups_after_move(
            move, color, own_gids, cap_gids, seen_gids)

        # 压撤销信息（此时 ko/passes/player 尚未更新）。
        # ⚠ 棋块表的撤销记录是**同一条记录里的第 6 个元素**，不是第二条栈 ——
        #   两条栈会错位（`record=False` 的着法只推其中一条），而这里两者
        #   由同一次 `if record:` 一起进出，结构上不可能错位。
        #   第 7 个元素是「此刻的未记录落子计数」：撤销时用它判断这条记录
        #   描述的还是不是当前表（见 undo()）。
        if record:
            self._undo_stack.append(
                (move, captured, self.ko_point, self.passes, self.current_player,
                 group_record, self._unrecorded_seq))
        else:
            # `record=False` 的着法没有撤销记录，而它**改变了盘面与表** ——
            # 于是撤销链上更早的那些记录从这一刻起全部失效。用单调计数标记，
            # 旧记录在弹出时会自己发现「我这一段里有过未记录落子」。
            # ⚠ 这不是本表引入的限制：哈希那边同样要求「撤销 = 回退一手」，
            #   混用 record=False 与 undo 会让增量哈希与盘面脱钩。本表选择
            #   **作废重建**（表是纯派生物，重建永远正确），而不是留在错的状态。
            self._unrecorded_seq += 1

        # 打劫判定：提掉恰好 1 子，且落子子本身恰好只剩 1 气（即被提点）-> 形成劫
        if len(captured) == 1 and len(self._groups[move][1]) == 1:
            self.ko_point = captured[0][0] * n + captured[0][1]
        else:
            self.ko_point = -1

        self.passes = 0
        self.move_history.append(move)
        self.move_number += 1
        self._commit_position(color, move, captured)
        self.current_player = -self.current_player
        self._legal_cache = None
        return True

    def undo(self) -> bool:
        """撤销最近一次成功 play（须 play(record=True)）。

        完整恢复：棋盘子与被提子、`ko_point`（只读信息位）、pass 计数、着法历史、执子方、
        落子数、两套 Zobrist 键（通用指纹 + 重复判定用的 position 键）、
        重复局面历史（含出现次数）。
        返回是否成功（栈空返回 False）。

        入口同样先过 `_ensure_hash()`：若盘面/行棋方已被外部改写，接管会连带清空
        属于旧局的撤销栈，于是这里返回 False —— 而不是拿旧局的着法去恢复一颗
        新盘上不存在的棋子。
        """
        self._ensure_hash()
        if not self._undo_stack:
            return False
        (move, captured, ko, passes, player,
         group_record, unrecorded_at) = self._undo_stack.pop()
        n = self.board_size
        if move != -1:
            r, c = divmod(move, n)
            self.board[r, c] = 0
            if captured:
                cap_color = -player  # 被提子为落子方对手
                for (cr, cc) in captured:
                    self.board[cr, cc] = cap_color
            # 棋块/气表回滚（P2.7a）：只对**实着**回滚（pass 没改盘面，
            # 它的记录里也没有表的部分）。
            # ⚠ 可信性判据：`play(record=False)` 打断过撤销链时，这条记录
            #   描述的已经不是当前表，回滚它会把表恢复到从未存在过的状态。
            #   那就作废等重建 —— 表是盘面的纯派生物，重建永远正确。
            if unrecorded_at == self._unrecorded_seq:
                self._restore_groups_after_move(group_record)
            else:
                self._groups_valid = False
        self.move_history.pop()
        self.ko_point = ko
        self.passes = passes
        self.move_number -= 1
        self.current_player = player
        self._rollback_position(player, move, captured)
        self._legal_cache = None
        return True

    # ---- 终局与计分 --------------------------------------------------------

    def is_terminal(self, max_moves: int = None) -> bool:
        """终局判定：连续两次 pass（对局认输），或达到 move 上限。

        ⚠ **重复局面不是终局条件**：Tromp-Taylor 规则 6/8 下「重复」是**非法手**
        （PSK 已于 P2.6a-2b-1 接入 `get_legal_moves()`、于 P2.6a-2c 接入 `play()`），
        终局只由两次连续 pass 表达。本方法不查重复历史，不要在这里加。

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
    #   8      : 合法点掩码（= get_legal_moves()：空点 ∧ 非自杀 ∧ 非 PSK 重复。
    #             P2.6a-2b-1 起**取值变严** —— 通道序号与含义不变，只是这一格更严；
    #             它仍然只有 n*n 个点、**不含 PASS**）
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

        # 通道 8 = 合法点（TT 口径：禁自杀 + PSK）
        # ⚠ 供 P4.2 / P4.3 引用：这一格的值 = `get_legal_moves()`，即 TT 规则 6 的
        # 完整合法点集合，外加一条**有意偏离**（禁自杀，TT 本身允许自提）。
        # 只有 n*n 个点、**不含 PASS**（动作空间里的 pass 是 n*n，见 `GoBoard.PASS`）。
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
        """向量化批量版 feature_planes，**通道 8 除外**（见下）。

        输入:
            boards   : (B, n, n) int8，取值 -1/0/1
            my_hist  : (B, 3) int16，己方前 3 手扁平坐标（-1 填充）
            op_hist  : (B, 3) int16
            to_play  : (B,) int8，轮到谁落子（1 黑 / -1 白）
            ko       : (B,) int16，劫禁着点扁平坐标（-1 无）；可选，用于通道 8 排除
        返回: (B, 12, n, n) float32

        ⚠ **通道 8 与单图版不再等价（P2.6a-2b-1 起，已知分歧，非回归）**：
        两条路径的通道 8 差在**三件**互相独立的事上，别混成一句「更松」：
          1. **PSK：算不出来**（不是选择）。本方法只拿到裸 `boards` 数组，**没有
             GoBoard、因而没有重复局面历史**，超级劫在这条路径上**物理上无法判**。
             单图 `feature_planes()` 的通道 8 走 `get_legal_moves()`，含 PSK，取值更严。
          2. **禁自杀：算得出来，但本路径有意没做**（是**选择**，不是物理不可能）。
             它是纯局部判定（只看邻接块的气），`boards` 数组本身就够算；不做是为了
             保持热路径成本（每节点一次批量前向的成本敏感）。这个分歧在本任务之前
             就已存在（单图版在 `check_suicide=False` 默认下同样不查自杀），
             P2.6a-2b-1 只是把它拉大。**可以直接向量化补上**（P4.3 的 17 通道）。
          3. **排除 `ko` 点：一条遗留近似，且单图侧已不再有对应判罚**（P2.6a-2b-1/2c）。
             `ko_point` 早已降级为**只读信息位**（见 `GoBoard` 类 docstring），单图掩码
             **不读它** —— 简单劫由 PSK 独立禁掉。所以这里的「排除 ko」不再镜像任何
             规则，只是**碰巧**与单图侧一致：单子劫两边都禁（一个靠 PSK、一个靠字段）；
             而在「多子提子形成的劫」（`ko_point` 的成劫判据不要求那块是单子）与
             「外部盘面接管后无历史」两种情形下，本路径会比规则**更严**（禁掉按 TT 合法的
             着法）。⚠ 改本函数时不要把这个更严当成「安全」：它与单图通道 8 的差
             正是训练分布的来源之一（输入分布登记见路线图 D14）。
        收口需要把重复局面历史一起批量喂进来（P2.6b 的下游语义同步 / P4.3 的 17 通道），
        不要在这里假装两条路径一致。

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

        # 通道 8: 合法点掩码（空点，ko 点排除）。⚠ **与单图 get_legal_moves() 不等价**，
        # 差在三件独立的事上（PSK 算不出 / 禁自杀有意没做 / 排除 ko 是遗留近似，
        # 单图侧已改由 PSK 判简单劫、不再读该字段）—— 见本方法 docstring 的
        # 「通道 8 除外」段。禁自杀是纯局部判定、boards 就够算，本路径**有意没做**
        # （保持热路径成本），是选择不是算不出来；「排除 ko」则不是选择也不是算不出，
        # 它比规则更严，别当成安全边际。
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
