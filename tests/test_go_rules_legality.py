"""
GoBoard 合法性接入测试（P2.6a-2b-1：禁自杀 + 位置超级劫进 `get_legal_moves()`）。

本文件锁住的是**接线之后**的合法性语义。`get_legal_moves()` 判三条：空点 / 禁自杀 /
位置超级劫（PSK = Tromp-Taylor 规则 6 的全文，掩码因此**就是**完整的 TT 合法点集合）。
`play()` 另有一道 **TT 规则之外**的劫禁（`move == self.ko_point`），`ko_point` 随之只是
只读信息位；两者的分叉见 §6b 那个钉住它的用例。

覆盖面（5 路与 9 路各一组，19 路一组专测成本）：
  1. `test_suicide_is_illegal`                     角上单点自杀 / 单点自杀 / 自杀但提子
  2. `test_simple_ko_is_illegal_immediately`       标准单劫：回提点非法，且判罚不靠 ko_point
  3. `test_positional_superko_illegal`             真实三重劫（9 路）：闭合成手非法
  4. `test_no_board_mutation_during_legality_check` 判定过程零状态变更 + 只在提子候选上写盘
  5. `test_eye_filling_stays_legal`                填自己的真眼**合法**（不要误禁）
  6. `test_ko_point_does_not_affect_legality`      掩码与「ko_point 强制置 -1」逐位相同
  7. `test_legality_cost_is_bounded`               19 路掩码生成中位耗时上界
  8. `test_cached_mask_is_invalidated_by_lifecycle` play / undo / reset 与缓存的关系
另有三个配套用例：
  - `test_suicide_boundary_same_colour_group_losing_its_only_liberty`
        自杀判定的关键边界（同色邻块那唯一一口气就是落点本身）
  - `test_psk_candidate_key_ignores_side_to_play`
        PSK 候选键 = position-only（染色，与行棋方无关）
  - `test_known_gap_mask_allows_multi_stone_ko_recapture`
        **钉住两处分歧**：多子提子形成的劫，其立即回提**按 TT 是合法手**（掩码放行是对的），
        但 `play()` 仍按自己那道非 TT 的劫禁拒绝

为什么 3 只在 9 路跑：多劫循环夹具需要 **3 个互不相邻的劫形**，最小足迹就是 9 路
（三劫夹具的子占 (2..8, 1..8)）。已用穷举验证过 **5 路装不下任何双劫循环**（把标准
单劫形状 + 其颜色取反的副本在 5x5 上平移/旋转全部试过，均出现子冲突或「循环不闭合」）
—— 小盘口的循环只能靠反复重演同一批劫，那正是 PSK 要禁的着法。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.go_rules import GoBoard  # noqa: E402

N5 = 5
N9 = 9


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #

def _hand_built(n, stones, to_play=1):
    """按 (r, c) 摆子并显式接管（与 tests/test_go_hash.py 同一姿势，不共享任何符号）。"""
    b = GoBoard(n)
    for (r, c), v in stones.items():
        b.board[r, c] = v
    b.current_player = to_play
    b.resync_hash()
    assert b.ko_point == -1, "夹具开局不得带劫禁着（否则测的就不是 PSK 了）"
    return b


def _idx(n, r, c):
    return r * n + c


# ---- 1. 三个自杀 / 提子 case 同屏 ------------------------------------------
#
# 手工盘（X=黑=1 当行棋方，O=白=-1），5 路：
#     c0 c1 c2 c3 c4
#  r0  .  O  O  .  X
#  r1  O  .  O  X  O      <- (1,4) 是白子，只剩 (2,4) 一口气
#  r2  O  O  .  O  .      <- 黑下 (2,4) 提掉 (1,4)：落子本身无气，但提子 -> 合法
#  r3  .  .  .  .  O
#  r4  .  .  .  .  .
#
# 三个 case：
#   (0,0) 角上单点自杀：两个邻点都是白，且两块白气数都 >= 2（提不掉）
#   (1,1) 单点自杀    ：四邻全是白，同样提不掉
#   (2,4) 自杀但提子  ：落点四邻全是白，己方无气；但 (1,4) 在打吃 -> 提子 -> 合法
#
# 9 路用同一套左上角棋形（上表逐行嵌入 9 路即可，多出来的空点只会让周围白块**气更多**，
# 不会把「提不掉」变成「提得掉」），提子 case 改放**内部**的 (4,4)：5 路那版靠右边界
# 少一个邻点才让 (1,4) 只剩一口气，嵌进 9 路就会多出 (1,5) 这口气，atari 就不成立了。
_SUICIDE_5_STONES = {
    (0, 1): -1, (0, 2): -1, (1, 0): -1, (2, 0): -1, (2, 1): -1, (1, 2): -1,
    (1, 4): -1, (0, 4): 1, (1, 3): 1, (3, 4): -1, (2, 3): -1,
}
_SUICIDE_5_CASES = [((0, 0), False), ((1, 1), False), ((2, 4), True)]

_SUICIDE_9_STONES = {
    (0, 1): -1, (0, 2): -1, (1, 0): -1, (2, 0): -1, (2, 1): -1, (1, 2): -1,
    (4, 3): -1, (3, 3): 1, (5, 3): 1, (4, 2): 1,          # (4,3) 白在打吃，只剩 (4,4)
    (3, 4): -1, (5, 4): -1, (4, 5): -1,                  # (4,4) 的另三个邻点：白且气 >= 2
}
_SUICIDE_9_CASES = [((0, 0), False), ((1, 1), False), ((4, 4), True)]

# ---- 2. 标准单劫（tests/test_go_rules.py::test_ko 的同一棋形，5 路也装得下）----
_KO_STONES = {
    (0, 1): 1, (0, 2): -1,
    (1, 0): 1, (1, 1): -1, (1, 3): -1,
    (2, 1): 1, (2, 2): -1,
}
_KO_CAPTURE = (1, 2)      # 黑下这里 -> 提掉 (1,1)，形成单劫
_KO_RECAPTURE = (1, 1)    # 白立即回提 -> 复现「黑提之前」的染色

# ---- 3. 真实三重劫（与 tests/test_go_hash.py §5 同一局面，此处**重建**不共享符号）----
# 三劫棋形：ko A / ko C 是白子被黑提，ko B 是黑子被白提。
# A: 白(3,2) 被黑(3,3) 提；B: 黑(3,6) 被白(3,7) 提；C: 白(7,2) 被黑(7,3) 提。
_TRIPLE_KO_STONES = {
    (2, 2): 1, (2, 3): -1, (3, 1): 1, (3, 2): -1, (3, 4): -1, (4, 2): 1, (4, 3): -1,
    (2, 6): -1, (2, 7): 1, (3, 5): -1, (3, 6): 1, (3, 8): 1, (4, 6): -1, (4, 7): 1,
    (6, 2): 1, (6, 3): -1, (7, 1): 1, (7, 2): -1, (7, 4): -1, (8, 2): 1, (8, 3): -1,
}
# 6 手：黑提 A -> 白提 B -> 黑提 C -> 白回提 A -> 黑回提 B -> 白回提 C
# 末手之后盘面与开局完全相同 => 位置超级劫成立。
_TRIPLE_KO_SEQUENCE = [
    (1, 3 * N9 + 3),
    (-1, 3 * N9 + 7),
    (1, 7 * N9 + 3),
    (-1, 3 * N9 + 2),
    (1, 3 * N9 + 6),
    (-1, 7 * N9 + 2),
]

# ---- 5. 真眼 / 假眼 ------------------------------------------------------- #
# 真眼：己方环抱的单点，且环抱块另有别的气 -> 填它合法。
_EYE_5_STONES = {(1, 1): 1, (1, 2): 1, (1, 3): 1, (2, 0): 1, (2, 1): 1,
                 (2, 3): 1, (2, 4): 1, (3, 1): 1, (3, 2): 1, (3, 3): 1}
_EYE_5_POINT = (2, 2)
# 假眼：整块只剩这一口气 -> 填它就是自杀，必须被禁。
_LAST_LIB_5_STONES = {(r, c): 1 for r in range(N5) for c in range(N5)
                      if (r, c) != (2, 2)}
_LAST_LIB_5_POINT = (2, 2)
_EYE_9_STONES = {(3, 3): 1, (3, 4): 1, (3, 5): 1, (4, 3): 1, (4, 5): 1,
                 (5, 3): 1, (5, 4): 1, (5, 5): 1}
_EYE_9_POINT = (4, 4)


# --------------------------------------------------------------------------- #
# 1. 禁自杀
# --------------------------------------------------------------------------- #

def test_suicide_is_illegal():
    """纯自杀非法；**自杀但提到对方子**合法。三个 case 同屏，5 路与 9 路各跑一遍。

    这是行为性红：接入前 `get_legal_moves()` 默认 `check_suicide=False`，
    (0,0)/(1,1) 都会被判成合法。
    """
    for n, stones, cases in ((N5, _SUICIDE_5_STONES, _SUICIDE_5_CASES),
                             (N9, _SUICIDE_9_STONES, _SUICIDE_9_CASES)):
        b = _hand_built(n, stones, to_play=1)
        mask = b.get_legal_moves()
        assert mask.shape == (n * n,), f"{n} 路掩码长度必须是 n*n（**没有 PASS 槽**）"
        for (r, c), expect_legal in cases:
            mv = _idx(n, r, c)
            assert bool(mask[mv]) is expect_legal, (
                f"{n} 路 ({r},{c}) 期望 legal={expect_legal}，"
                f"实际 {bool(mask[mv])}\n{b.to_string()}")
        # 与 play() 的结构性判定对拍：掩码说非法的，play() 也必须拒绝；
        # 掩码说合法的（本盘只有提子那个点），play() 必须接受。
        for (r, c), expect_legal in cases:
            mv = _idx(n, r, c)
            assert b.clone().play(mv) is expect_legal, \
                f"{n} 路 ({r},{c}) 掩码与 play() 判定不一致"


def test_suicide_boundary_same_colour_group_losing_its_only_liberty():
    """自杀判定的关键边界：**同色邻块的那唯一一口气就是落点本身**。

    盘上 (0,4)/(1,3) 是黑块且**只**剩 (0,3) 一口气；黑下 (0,3) 会与该块合并，
    合并后整块无气、也提不掉任何白子 -> 必须是自杀。

    这一条专门锁「气数 >= 2」而不是「有气」的判据：只判「同色邻块有气」会把
    (0,3) 错放成合法（那正是落点自己堵死自己唯一一口气的情形）。
    """
    b = _hand_built(N5, _SUICIDE_5_STONES, to_play=1)
    mv = _idx(N5, 0, 3)
    mask = b.get_legal_moves()
    assert not mask[mv], "(0,3) 必须被判自杀"
    assert b.clone().play(mv) is False, "play() 也必须拒绝（结构性自杀检查）"
    # 对照：**同一个点换成白方下是合法的** —— 白子并入白块 A，而 A 除 (0,3) 外还有
    # {(0,0),(1,1),(2,2)} 三口气（并顺手提到只剩这一口气的黑块）。
    # 这条对照证明判据是「**落子方自己**合并后块的气数」，而不是「该点周围有没有气」。
    w = _hand_built(N5, _SUICIDE_5_STONES, to_play=-1)
    assert w.get_legal_moves()[mv], "白方下 (0,3) 必须合法（它并入的是有气的白块）"
    assert w.clone().play(mv) is True


# --------------------------------------------------------------------------- #
# 2. 简单劫
# --------------------------------------------------------------------------- #

def test_simple_ko_is_illegal_immediately():
    """标准单劫：黑提子形成劫后，白**立即**回提的那一点在掩码里必须是 False；
    而且**判罚不靠 `ko_point` 字段**。

    行为性红 —— 关键在后半个用例：把 `ko_point` 清成 -1（「劫禁已解除」的状态），
    掩码**仍然**必须禁掉回提点，因为回提复现的是「黑提之前」的染色，那个染色的
    position 键一直在 PSK 历史里。接入前 `get_legal_moves()` 只认 `ko_point` 字段，
    置 -1 就放行 -> 红；接入后判罚完全由 PSK 承担 -> 绿。

    为什么必须**人工**构造「ko_point = -1 但染色没变」这个状态：黑白交替使得
    「提子方 pass」根本轮不到（提完就轮到被提方），而被提方一旦在别处落子，染色就变了、
    PSK 也不该再禁。所以这个状态在真实对局里不可达。人工置 -1 是把「**判罚来源**」
    这条性质单独取出来测 —— 它就是 P2.6a-2b-1 要求 3（`ko_point` 降级为只读派生量）
    的直接编码，也是 `test_ko_point_does_not_affect_legality` 的姊妹用例：
    那条测「掩码不依赖该字段」，这条测「不依赖它也判得对」。
    """
    for n in (N5, N9):
        b = _hand_built(n, _KO_STONES, to_play=1)
        r, c = _KO_CAPTURE
        mv = _idx(n, r, c)
        assert b.get_legal_moves()[mv], "形成劫的那一手本身必须合法"
        assert b.clone().play(mv) is True, "play() 必须接受提子手"
        assert b.play(mv) is True
        assert b.ko_point == _idx(n, *_KO_RECAPTURE), \
            f"{n} 路夹具必须造出单劫，实际 ko_point={b.ko_point}"

        rr, cc = _KO_RECAPTURE
        recapture = _idx(n, rr, cc)
        assert not b.get_legal_moves()[recapture], \
            f"{n} 路：立即回提 ({rr},{cc}) 必须在掩码里被判非法"
        # 判罚的依据是 PSK（复现「提子之前」的染色），不是 ko_point 字段
        assert b._would_repeat(b.position_hash_after_move(recapture)) is True, \
            "回提必须命中 PSK 历史"
        assert b.clone().play(recapture) is False, "play() 也必须拒绝（简单劫检查）"

        # ---- 判罚不靠 ko_point 字段（行为性红的核心）----
        forced = b.clone()
        forced.ko_point = -1
        forced._legal_cache = None
        assert not forced.get_legal_moves()[recapture], (
            f"{n} 路：ko_point 置 -1（劫禁解除）后回提点仍必须非法 —— "
            "禁它的是 PSK，不是那个字段")
        # play() 那一侧只认字段，所以这里会放行：两条路径的分工被本用例钉住
        assert forced.play(recapture) is True, \
            "play() 只做结构性检查（它读 ko_point，但读不到 PSK）"

        # ---- 在别处落一手之后，劫禁解除，回提重新合法 ----
        far = _idx(n, n - 1, n - 1)
        assert far not in (recapture, mv) and not b.board[n - 1, n - 1]
        assert b.play(far) is True
        assert b.ko_point == -1
        assert b._would_repeat(b.position_hash_after_move(recapture)) is False, \
            "别处落子后染色已变，PSK 不再命中"
        assert b.get_legal_moves()[recapture], "劫禁解除后回提点必须恢复合法"


# --------------------------------------------------------------------------- #
# 3. 位置超级劫
# --------------------------------------------------------------------------- #

def test_positional_superko_illegal():
    """真实三重劫（9 路）：第 6 手闭合成手，掩码必须把它判为非法。

    行为性红：接入前超级劫手仍可落（`tests/test_go_hash.py` 里那条同名断言已翻转）。
    """
    b = _hand_built(N9, _TRIPLE_KO_STONES, to_play=1)
    for i, (who, mv) in enumerate(_TRIPLE_KO_SEQUENCE[:-1]):
        assert b.current_player == who, f"第 {i} 手行棋方不符"
        assert b.get_legal_moves()[mv], f"第 {i} 手 {mv} 必须合法"
        assert b.play(mv) is True, f"第 {i} 手 {mv} 应能落下"

    who, closing = _TRIPLE_KO_SEQUENCE[-1]
    assert b.current_player == who
    # 闭合成手复现的是**开局**的染色
    assert b._would_repeat(b.position_hash_after_move(closing)) is True
    assert not b.get_legal_moves()[closing], \
        "超级劫闭合成手必须在掩码里被判非法"
    # 掩码不能因为「只禁了这一手」而误伤邻点
    assert b.get_legal_moves().sum() > 0, "其余合法点必须照常可用"

    # PSK 判罚随历史走：撤销一手必须把那条染色的记录一并回滚，掩码跟着变；
    # 重演之后闭合成手仍然非法（PSK 判定可复现，不是一次性的临时状态）。
    hist_len = len(b._pos_hash_history)
    mask_before = np.asarray(b.get_legal_moves()).copy()
    assert b.undo() is True
    assert len(b._pos_hash_history) == hist_len - 1, "undo 必须回滚重复局面历史"
    assert b._legal_cache is None, "undo 必须失效掩码缓存"
    assert not np.array_equal(np.asarray(b.get_legal_moves()), mask_before), \
        "撤销一手后掩码必须变化（否则缓存与新规则脱节）"
    assert b.play(_TRIPLE_KO_SEQUENCE[-2][1]) is True
    assert not b.get_legal_moves()[closing], "重演之后闭合成手仍然非法"
    assert b._would_repeat(b.position_hash_after_move(closing)) is True


def test_psk_candidate_key_ignores_side_to_play():
    """PSK 的关键语义：重复判定**只看棋盘染色，与行棋方无关**。

    从两侧钉住它：
      1. **掩码层**（红得住的那条）：三重劫的闭合成手在掩码里必须被判非法。
         若掩码的 PSK 分支改用 `hash_after_move()`（含行棋方），候选键与历史里存的
         染色键永不相等 -> 静默查不中 -> 闭合成手被放行 -> 本行红。
      2. **键层**：同一染色、**相反行棋方**的两个棋盘，`position_hash()` 必须逐位相同，
         而通用指纹 `hash()` 必须不同。

    ⚠ **必须说清这条用例证明到哪一步**：第 2 组断言只证明**那把钥匙**的语义
    （position-only vs 含行棋方），**不证明掩码 PSK 分支的语义** —— 掩码根本不调
    `position_hash()`，它调的是 `position_hash_after_move()`。也就是说，把掩码**整套**
    换成 situational 键（本组断言仍绿、上面那条掩码断言也绿：实着造成的局面重复与上次
    出现的间隔必为偶数，偶数间隔 ⟹ 行棋方相同 ⟹ SSK 与 PSK 在掩码上同判），
    本用例抓不到。已用 288 局随机对弈（5/7/9 路）逐候选扫描确认实着路径上无反例，
    所以这个盲区在实着上不可观察；要抓它需要构造离奇数间隔的循环（pass 不在掩码里）。
    第 1 组断言守的是**现实中最可能发生的回归**：候选键误用 `hash_after_move()` 之后
    键族与历史不匹配，掩码静默失效。
    """
    a = _hand_built(N9, _TRIPLE_KO_STONES, to_play=1)
    for who, mv in _TRIPLE_KO_SEQUENCE[:-1]:
        assert a.play(mv) is True

    # ---- 1. 掩码层：行为性红（候选键族必须与历史里的染色键同族）----
    who, closing = _TRIPLE_KO_SEQUENCE[-1]
    assert not a.get_legal_moves()[closing], \
        "闭合成手必须在掩码里被判非法（候选键若含行棋方就会静默查不中历史）"

    # ---- 2. 键层：position 键只认染色，通用指纹认行棋方 ----
    twin = a.clone()
    twin.current_player = -a.current_player
    twin.resync_hash()
    assert np.array_equal(twin.board, a.board), "两个盘面必须同形"
    assert twin.current_player != a.current_player, "行棋方必须相反"
    assert twin.position_hash() == a.position_hash(), \
        "position 键（= 掩码 PSK 分支的候选键口径）只认棋盘染色"
    assert twin.hash() != a.hash(), "通用指纹含行棋方，必须仍能区分二者"

    key = a.position_hash_after_move(closing)
    assert key in a._pos_hash_counts, "候选键必须真的命中历史（否则测试是空转）"
    assert key == a._pos_hash_history[0], "闭合成手复现的正是开局染色"


# --------------------------------------------------------------------------- #
# 4. 判定过程零状态变更
# --------------------------------------------------------------------------- #

def _write_spy(board):
    """返回 (棋盘的 ndarray 子类视图, 写日志)。只记录**棋盘数组自身**的写入。

    掩码是 `(board == 0)` 派生出来的 ndarray，同属一个子类；用 `id(self)` 把
    掩码上的 `legal[i] = False` 与棋盘上的写入区分开。
    """
    log = []
    holder = {}

    class _Spy(np.ndarray):
        def __setitem__(self, key, value):
            if id(self) == holder.get("id"):
                log.append((key, int(value)))
            return super().__setitem__(key, value)

    spy = board.view(_Spy)
    holder["id"] = id(spy)
    return spy, log


def _state_snapshot(b):
    """把「不该被合法性判定改动」的全部状态拍成一个可逐位比较的快照。"""
    return {
        "board": b.board.tobytes(),
        "current_player": b.current_player,
        "ko_point": b.ko_point,
        "passes": b.passes,
        "move_number": b.move_number,
        "move_history": list(b.move_history),
        "undo_stack": list(b._undo_stack),
        "hash": b._zobrist,
        "pos_zobrist": b._pos_zobrist,
        "pos_hash_history": list(b._pos_hash_history),
        "pos_hash_counts": dict(b._pos_hash_counts),
    }


def test_no_board_mutation_during_legality_check():
    """`get_legal_moves()` 是「只读 + 带缓存」的方法：判定过程不得留下任何状态变更。

    三段断言：
      (a) **入口 `_ensure_hash()` 的效果**：本方法是 `_ensure_hash()` 的首个调用者，
          而它可能触发 `_adopt_as_new_game()`（一次状态变更）。这里锁住「变更被收在
          入口」这件事的可观察后果 —— 掩码算完之后，重复局面历史必须**属于当前盘面**。
          行为性红：接入前 `get_legal_moves()` 根本不调 `_ensure_hash()`，外部换掉
          board 数组后历史仍属于旧盘面，`_would_repeat(position_hash())` 返回假。
      (a2) **入口调用排在缓存检查之前**：接管会失效 `_legal_cache`，所以只要
          `_ensure_hash()` 在缓存检查**之后**，外部换掉 board 数组再只调
          `get_legal_moves()` 就会拿到**属于旧盘面**的过期掩码。行为性红：投毒过的
          缓存会被原样返回。
      (b) **零残留**：在已经自洽的盘面上调用，前后所有状态逐位相同（第二次调用走缓存，
          也必须一样）。
      (c) **零模拟**：判定过程中对 `self.board` 的写入只允许出现在**提子候选**上
          （`position_hash_after_move` 的临时落子/还原），且只写「该点原本是空」的
          位置。这正是两段式「不提子走纯算术」的成本主张的可执行版本。
    """
    # ---- (a) 入口 _ensure_hash ----
    src = _hand_built(N5, _SUICIDE_5_STONES, to_play=1)
    b = GoBoard(N5)
    b.board = src.board.copy()               # 外部整体换盘面 -> _ensure_hash 会接管
    b.current_player = src.current_player
    stale_hist = list(b._pos_hash_history)
    assert len(stale_hist) == 1 and stale_hist == [0], \
        "新棋盘的历史还停在空盘上（本用例要的就是这个「陈旧」前提）"
    b.get_legal_moves()
    assert b._pos_hash_history == [b.position_hash()], \
        "掩码算完后重复局面历史必须属于当前盘面（入口 _ensure_hash 生效）"
    assert b._would_repeat(b.position_hash()) is True, \
        "当前局面必须在历史里 —— 接入前这里会因为历史陈旧而返回假"
    assert b._pos_hash_history != stale_hist, \
        "本用例必须真的换掉了盘面（否则断言空转）"

    # ---- (a2) 入口调用必须在缓存检查之前 ----
    # 走的是「调用方只调 get_legal_moves()、不碰 hash()/play()」这条最朴素的路径：
    # 那种情况下没有任何别的方法会先把缓存清掉。
    poisoned = GoBoard(N5)
    poisoned.get_legal_moves()               # 热缓存：此刻的掩码属于**空盘**（全 True）
    assert poisoned._legal_cache is not None
    poisoned._legal_cache[_idx(N5, 4, 4)] = False   # 人为投毒：把新盘面上合法的一格标非法
    poisoned.board = src.board.copy()        # 外部整体换盘面
    mask = poisoned.get_legal_moves()        # 只调这一个方法
    assert mask[_idx(N5, 4, 4)], \
        "换掉 board 数组后只调 get_legal_moves() 也必须重算（入口 _ensure_hash 排在缓存检查之前）"
    assert not mask[_idx(N5, 0, 0)], \
        "新盘面上 (0,0) 是自杀点，必须为 False —— 拿到的是空盘的旧掩码就说明顺序反了"
    assert poisoned._pos_hash_history == [poisoned.position_hash()], \
        "重算掩码之前必须先完成接管（否则是「哈希已重置、掩码还是旧的」）"

    # ---- (b) 零残留 ----
    for n, stones in ((N5, _SUICIDE_5_STONES), (N9, _SUICIDE_9_STONES)):
        g = _hand_built(n, stones, to_play=1)
        g.get_legal_moves()                  # 先把缓存热起来
        before = _state_snapshot(g)
        first = np.asarray(g.get_legal_moves()).copy()
        after_first = _state_snapshot(g)
        second = np.asarray(g.get_legal_moves()).copy()
        assert before == after_first, f"{n} 路：掩码计算改动了状态"
        assert np.array_equal(first, second), "两次取值必须一致"
        assert _state_snapshot(g) == after_first, f"{n} 路：缓存命中路径也改了状态"

    # ---- (c) 零模拟：写入只出现在提子候选上 ----
    h = _hand_built(N5, _SUICIDE_5_STONES, to_play=1)
    before_bytes = h.board.tobytes()
    empty_before = {(r, c) for r, c in zip(*np.nonzero(h.board == 0))}
    spy, log = _write_spy(h.board)
    h.board = spy                           # 换数组对象 -> 入口会接管（这正是 (a) 的语义）
    h.get_legal_moves()
    capture_r, capture_c = _SUICIDE_5_CASES[2][0]
    for key, value in log:
        assert isinstance(key, tuple) and len(key) == 2, f"写入应逐点进行，实际 {key!r}"
        r, c = key
        assert (r, c) in empty_before, f"判定过程往非空点 {key} 写了 {value}"
        assert value in (0, 1), f"判定过程只应写当前行棋方或 0，实际 {value}"
    touched = {(k[0], k[1]) for k, _ in log}
    assert touched <= {(capture_r, capture_c)}, (
        "只有**提子**候选才允许走完整推演（临时落子），实际被写的点："
        f"{sorted(touched)}")
    assert h.board.tobytes() == before_bytes, "判定过程结束后盘面必须逐字节还原"


# --------------------------------------------------------------------------- #
# 5. 眼位
# --------------------------------------------------------------------------- #

def test_eye_filling_stays_legal():
    """填自己的**真眼**合法；「看着像眼、但整块只剩这一口气」的假眼必须禁。

    反向回归：禁自杀接进来之后最容易出的错就是把真眼一起禁掉（那会让自对弈在收束阶段
    凭空少掉大量合法点，甚至让「无处可下」提前发生）。
    """
    for n, stones, pt in ((N5, _EYE_5_STONES, _EYE_5_POINT),
                          (N9, _EYE_9_STONES, _EYE_9_POINT)):
        b = _hand_built(n, stones, to_play=1)
        mv = _idx(n, *pt)
        assert b.get_legal_moves()[mv], (
            f"{n} 路：填自己的真眼 {pt} 必须合法（不要误禁）\n{b.to_string()}")
        assert b.clone().play(mv) is True

    for n, stones, pt in ((N5, _LAST_LIB_5_STONES, _LAST_LIB_5_POINT),):
        b = _hand_built(n, stones, to_play=1)
        mv = _idx(n, *pt)
        mask = b.get_legal_moves()
        assert not mask[mv], f"{n} 路：整块只剩这一口气，填它是自杀，必须禁"
        assert b.clone().play(mv) is False


# --------------------------------------------------------------------------- #
# 6. ko_point 不再参与判罚
# --------------------------------------------------------------------------- #

def test_ko_point_does_not_affect_legality():
    """掩码的取值不得依赖 `ko_point` 字段：置 -1 重算，掩码必须**逐位相同**。

    行为性红：接入前 `get_legal_moves()` 里有 `if self.ko_point >= 0: legal[ko] = False`
    这一行直接读字段，所以「置 -1 后掩码变宽」—— 两条掩码必然不同。

    这条断言证明的是「掩码的取值不再**依赖**该字段」。它**不能**区分「字段没被读」与
    「字段被读了但冗余」—— 事实上在本夹具里 PSK 会独立地把劫禁着点禁掉，所以两种实现
    都能通过。真正的信息位语义由下面那条断言给出：`ko_point` 仍被 `play()` 记着并读着。
    """
    for n in (N5, N9):
        b = _hand_built(n, _KO_STONES, to_play=1)
        assert b.play(_idx(n, *_KO_CAPTURE)) is True
        ko = b.ko_point
        assert ko >= 0, f"{n} 路夹具必须造出劫禁着（否则本测试是空转）"

        with_ko = np.asarray(b.get_legal_moves()).copy()
        b.ko_point = -1
        b._legal_cache = None
        without_ko = np.asarray(b.get_legal_moves()).copy()
        assert np.array_equal(with_ko, without_ko), (
            f"{n} 路：掩码仍依赖 ko_point（两条掩码不同）")
        assert not with_ko[ko], f"{n} 路：劫禁着点必须仍然非法（由 PSK 判，不是靠字段）"

    # 字段本身仍是活的：play() 记它、undo() 回退它 —— 它是「只读派生量」而不是死字段
    c = _hand_built(N9, _KO_STONES, to_play=1)
    mv = _idx(N9, *_KO_CAPTURE)
    assert c.play(mv) is True
    assert c.ko_point >= 0
    assert c.undo() is True
    assert c.ko_point == -1, "undo 必须回退 ko_point"


# --------------------------------------------------------------------------- #
# 6b. 与 play() 的已知分歧（钉住它，别让它被悄悄忘掉）
# --------------------------------------------------------------------------- #

# A=(4,4) 白；黑下 B=(4,5) 提掉它，且黑块 {B,(4,6)} 提子后**只剩 1 口气** = A。
# A 的其余三邻是黑 -> A 被打吃；黑块外圈是白 -> 黑块的气只有 A。
_MULTI_KO_STONES = {
    (3, 4): 1, (5, 4): 1, (4, 3): 1,        # A 的其余三邻：黑（A 被打吃）
    (4, 6): 1,                              # 黑块第二子
    (4, 4): -1,                             # A：白
    (3, 5): -1, (5, 5): -1, (3, 6): -1, (5, 6): -1, (4, 7): -1,   # 黑块外圈：白
}


def test_known_gap_mask_allows_multi_stone_ko_recapture():
    """**已知分歧**，刻意不在 P2.6a-2b-1 修：**掩码是对的，偏差在 `play()` 那道劫禁。**

    Tromp-Taylor 规则 6 的全文只有一句：「A turn is either a pass; or a move that doesn't
    repeat an earlier grid coloring.」——**没有** ko-capture 从句。所以「多子提子形成的劫」
    的立即回提，在 TT 下**是合法手**，本掩码放行它是**正确**的。

    简报的前提「PSK 已覆盖简单劫」对**单子劫**成立。但 `play()` 的成劫判据是
    「恰好提 1 子 **且** 落子后己方块总气 == 1」，**没有**要求那块是单子。于是一块
    2 颗子的黑棋被打吃时：白回提会一次提掉**整块 2 子**，落子后的染色与「黑提之前」那个
    染色**不同**（那颗 (4,6) 的黑子被一并提掉了）-> 不是 PSK 重复 -> 掩码按 TT 放行；
    而 `play()` 读 `ko_point`、拒掉它。**多出来的禁令在 `play()` 侧，TT 并没有这条。**

    实测频率：9 路随机自对弈 100 局里约 1.2% 的取点撞上这一条（30759 次成功取点中
    373 次被 `play()` 拒绝，全部属于本类）。

    收口方向：**拿掉 `play()` 里 `if move == self.ko_point: return False`**（连带后果见
    `GoBoard` 类 docstring 的「与 `play()` 的两处分歧」段：`play()` 不查 PSK，拿掉之后
    它连经典单子劫的立即回提也会接受）。**不是**把「上一手是否成劫」带进掩码。

    **本测试的用途是钉住这个分歧**：将来 whoever 收口了它（无论是拿掉 `play()` 的劫禁、
    还是把 `play()` 的成劫判据收窄成「必须是单子」），本测试都会红，并在失败信息里
    指出该改哪一处。它不是对当前行为的「背书」。
    """
    b = _hand_built(N9, _MULTI_KO_STONES, to_play=1)
    capture_mv = _idx(N9, 4, 5)
    recapture = _idx(N9, 4, 4)
    assert b.play(capture_mv) is True, "黑提子手必须合法"
    assert b.ko_point == recapture, "夹具必须造出劫禁着"
    assert len(b._undo_stack[-1][1]) == 1, "这一步只提掉 1 子"

    # 提子后黑块有 2 颗子、气恰好 1 —— 这正是「非单子劫」
    assert b._group_liberty_count(4, 5) == 1
    assert sum(1 for r in range(N9) for c in range(N9)
               if b.board[r, c] == 1 and (r, c) in ((4, 5), (4, 6))) == 2

    # 分歧本体：回提不是 PSK 重复 -> 掩码按 TT 放行
    assert b._would_repeat(b.position_hash_after_move(recapture)) is False, \
        "前提：这一手不复现历史染色，所以 PSK（= TT 规则 6）判不出来，也就该放行"
    assert b.get_legal_moves()[recapture], \
        "**当前**行为：掩码放行（这才是 TT 的答案）。收口后本行会红 —— 那是好事。"
    # 而 play() 仍然拒绝（它自己那道 TT 之外的劫禁）
    assert b.clone().play(recapture) is False

    # 收窄成单子劫之后，PSK 就能独立禁掉：这是缺口不存在的情形
    single = _hand_built(N9, _KO_STONES, to_play=1)
    assert single.play(_idx(N9, *_KO_CAPTURE)) is True
    assert not single.get_legal_moves()[_idx(N9, *_KO_RECAPTURE)], \
        "单子劫必须被 PSK 独立禁掉（这正是「PSK 覆盖简单劫」成立的那个形状）"


# --------------------------------------------------------------------------- #
# 7. 成本
# --------------------------------------------------------------------------- #

def test_legality_cost_is_bounded():
    """19 路一次掩码生成的**中位**耗时上界（用中位数，不用单次：单次会被 GC / 调度抖动打穿）。

    上界为什么取 20 ms —— 它要同时满足「抓得住回归」和「不误伤慢机器」：
      - 实测（本任务接入后，19 路随机自对弈中盘 ~0.5 ms、后期最密 ~1.4 ms），
        20 ms 是 **15~40 倍**的余量；
      - 接入前语义最接近的旧路径 `get_legal_moves(check_suicide=True)` 实测
        1.0~1.2 ms，所以 20 ms 并不比「接入前就已经很慢」更宽松；
      - 而它足以抓住**最贵的那一类真回归**：自杀判定退回「每个候选都原地落子模拟」
        或丢掉早退（实测 26.6 ms，直接破线）。

    ⚠ **它抓不住另一类回归，别把这条上界当万能护栏**：PSK 判罚退回「每个候选都走
    `position_hash_after_move` 的完整推演」（实测 ~13~46 µs/候选 × 19 路中盘约 181 个空点
    ≈ **2.4~8.3 ms**）**仍在 20 ms 以内**。抓它要另加断言（例如统计
    `position_hash_after_move` 的调用次数 ≤ 提子候选数），本轮没做。
    """
    b = GoBoard(19)
    rng = np.random.default_rng(20260926)
    for _ in range(19 * 19 // 2):           # 中盘：空点与棋块都多，最接近真实搜索负载
        legal = np.flatnonzero(b.get_legal_moves())
        if not legal.size:
            b.play(-1)
            continue
        b.play(int(rng.choice(legal)))
    b.resync_hash()

    samples = []
    for _ in range(9):
        b._legal_cache = None
        t0 = time.perf_counter()
        b.get_legal_moves()
        samples.append((time.perf_counter() - t0) * 1e3)
    median = float(np.median(samples))
    assert median < 20.0, (
        f"19 路 get_legal_moves() 中位耗时 {median:.2f} ms 超过 20 ms 上界"
        f"（样本={['%.2f' % s for s in samples]}）")


# --------------------------------------------------------------------------- #
# 8. 缓存生命周期
# --------------------------------------------------------------------------- #

def test_cached_mask_is_invalidated_by_lifecycle():
    """掩码缓存必须与对局生命周期同步：play 变、undo 复原、reset 回初始。

    接入 PSK 之后这条更关键：掩码的取值**依赖重复局面历史**，而历史会被
    play / undo / reset 改动。缓存一旦不同步，PSK 判罚就会读到过期历史 ——
    症状是「超级劫手被判合法」或「合法手被误禁」，且只在长对局里偶发。
    """
    b = _hand_built(N9, _TRIPLE_KO_STONES, to_play=1)
    initial = np.asarray(b.get_legal_moves()).copy()
    assert b._legal_cache is not None, "第一次调用后必须填上缓存"
    assert np.array_equal(np.asarray(b.get_legal_moves()), initial), "缓存命中应一致"

    # play -> 掩码变化且缓存被失效
    first = _TRIPLE_KO_SEQUENCE[0][1]
    assert b.get_legal_moves()[first]
    assert b.play(first) is True
    assert b._legal_cache is None, "play() 必须失效掩码缓存"
    after_play = np.asarray(b.get_legal_moves()).copy()
    assert not np.array_equal(after_play, initial), "落子后掩码必须变化"
    assert not after_play[first], "刚下的点不再合法（已被占）"

    # undo -> 完全复原
    assert b.undo() is True
    assert b._legal_cache is None, "undo() 必须失效掩码缓存"
    assert np.array_equal(np.asarray(b.get_legal_moves()), initial), \
        "undo() 之后掩码必须逐位复原（缓存与历史同步回退）"

    # reset -> 初始掩码
    for who, mv in _TRIPLE_KO_SEQUENCE[:3]:
        assert b.play(mv) is True
    b.reset()
    assert b._legal_cache is None, "reset() 必须失效掩码缓存"
    fresh = np.asarray(b.get_legal_moves()).copy()
    assert fresh.all(), "空盘的掩码必须全 True"
    assert np.array_equal(fresh, GoBoard(N9).get_legal_moves()), \
        "reset() 后的掩码必须与全新棋盘一致"

    # 缓存确实被用上了（否则上面几条只是「每次重算恰好一样」）
    b2 = _hand_built(N9, _TRIPLE_KO_STONES, to_play=1)
    b2.get_legal_moves()
    assert b2._legal_cache is not None
    before_id = id(b2._legal_cache)
    b2.get_legal_moves()
    assert id(b2._legal_cache) == before_id, "第二次调用必须命中同一个缓存对象"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nALL TESTS PASSED")
