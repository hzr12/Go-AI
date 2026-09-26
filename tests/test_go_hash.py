"""
GoBoard Zobrist 哈希 / 重复局面（PSK 地基）测试。

覆盖（对应 P2.6a-1 简报）:
  1. 跨实例 / 跨进程确定性
  2. 哈希依赖棋子颜色与行棋方
  3. play/undo 精确往返（增量更新不漏项、不漂移）
  4. clone() 深拷贝
  5. 真重复局面（三劫循环，6 手 superko）+ undo 回滚语义
  6. 初始局面在历史集合内
  7. num_passes / move_number / is_terminal / is_game_over
  8. 小盘口（5 路）维度：divmod 坐标映射 / 跨盘口同一把 Zobrist 钥匙 / 2·n² 上限
  9. **重复判定键 = position-only（PSK）**：只认棋盘染色，与行棋方无关；
     而通用指纹 hash() 含行棋方、**不用于**重复判定（§9）

另锁两条本任务的生命周期契约:
  - 外部盘面接管（整体替换 board 数组、或只改写 current_player）= **以此局面为新局**：
    对局进度归零、重复局面历史重建为 {当前局面}、ko_point 作为局面描述保留
  - 重复局面尚未接入 get_legal_moves()（P2.6a-2b 才接），本任务行为零变化
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game import go_rules  # noqa: E402
from src.game.go_rules import GoBoard  # noqa: E402

N = 9


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #

def _pick(b, rng):
    """从 get_legal_moves() 的真合法点里按固定种子选一手。"""
    legal = np.flatnonzero(b.get_legal_moves())
    assert legal.size > 0, "该局面无合法点"
    return int(rng.choice(legal))


def _naive_key(b):
    """与哈希实现完全无关的朴素局面键：原始盘面字节 + 行棋方。"""
    return (b.board.tobytes(), b.current_player)


def _fresh(n=N):
    return GoBoard(n)


# --------------------------------------------------------------------------- #
# 1. 确定性
# --------------------------------------------------------------------------- #

def test_hash_is_deterministic_across_instances():
    """两个独立实例走同一序列，每步哈希必须相同；reset() 后回到初始哈希。"""
    a, c = _fresh(), _fresh()
    init = a.hash()
    assert a.hash() == c.hash(), "两个新实例的初始哈希必须相同"

    rng_a, rng_c = np.random.default_rng(12345), np.random.default_rng(12345)
    for step in range(12):
        mv_a, mv_c = _pick(a, rng_a), _pick(c, rng_c)
        assert mv_a == mv_c, "固定种子必须选出同一手"
        assert a.play(mv_a) and c.play(mv_c)
        assert a.hash() == c.hash(), f"第 {step} 步哈希不同: {a.hash()} vs {c.hash()}"

    assert a.hash() != init, "走了 12 手后不该还是初始哈希"
    a.reset()
    c.reset()
    assert a.hash() == init, "reset() 必须回到初始空局面哈希"
    assert a.hash() == c.hash()
    assert len(a._pos_hash_history) == 1, "reset() 后历史只应含初始局面"


def test_hash_is_reproducible_across_processes():
    """固定种子 + 模块加载时预生成 => 跨进程可复现（用子进程取同一组哈希比对）。

    这是「确定性是硬要求」的端到端证据：若 Zobrist 表退化为进程内随机，
    子进程返回的数值必然不同。
    """
    probe = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from src.game.go_rules import GoBoard\n"
        "b = GoBoard(9)\n"
        "hs = [b.hash()]\n"
        "ps = [b.position_hash()]\n"
        "for mv in (0, 4, 40, 80, 41):\n"
        "    b.play(mv)\n"
        "    hs.append(b.hash())\n"
        "    ps.append(b.position_hash())\n"
        "print(','.join(str(h) for h in hs))\n"
        "print(','.join(str(h) for h in ps))\n"
    ) % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    b = GoBoard(9)
    mine = [b.hash()]
    mine_pos = [b.position_hash()]
    for mv in (0, 4, 40, 80, 41):
        assert b.play(mv)
        mine.append(b.hash())
        mine_pos.append(b.position_hash())

    outs = set()
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           text=True, timeout=300)
        assert r.returncode == 0, f"子进程失败: {r.stderr[-2000:]}"
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"两个子进程给出不同哈希: {outs}"
    lines = outs.pop().splitlines()
    assert [int(x) for x in lines[0].split(",")] == mine, "跨进程哈希不可复现"
    # 重复判定专用的 position 键同样必须跨进程可复现（PSK 判定完全建立在它上面）
    assert [int(x) for x in lines[1].split(",")] == mine_pos, \
        "跨进程 position 键不可复现"
    assert all(isinstance(h, int) and 0 <= h < 2 ** 64 for h in mine), \
        "hash() 必须返回 [0, 2**64) 的 int"
    assert all(isinstance(h, int) and 0 <= h < 2 ** 64 for h in mine_pos), \
        "position_hash() 必须返回 [0, 2**64) 的 int"


# --------------------------------------------------------------------------- #
# 2. 哈希依赖颜色与行棋方
# --------------------------------------------------------------------------- #

def test_hash_depends_on_color_and_side_to_play():
    # (a) 空盘面，只改行棋方 -> 必须不同（否则 PSK 把「轮到对方」误判重复）
    b1, b2 = _fresh(), _fresh()
    b1.current_player = 1
    b2.current_player = -1
    assert b1.hash() != b2.hash(), "仅行棋方不同，哈希必须不同"

    # (b) 同一点，黑子 vs 白子 -> 必须不同
    b3, b4 = _fresh(), _fresh()
    b3.play(4 * N + 4)
    b4.current_player = -1
    b4.play(4 * N + 4)
    assert b3.board[4, 4] == 1 and b4.board[4, 4] == -1
    assert b3.hash() != b4.hash(), "同点异色哈希必须不同"

    # (c) 同形棋面，黑先 vs 白先 -> 必须不同
    b5, b6 = _fresh(), _fresh()
    b5.play(4 * N + 4)
    b5.play(2 * N + 2)
    b6.board[4, 4] = 1
    b6.board[2, 2] = -1
    b6.current_player = -1
    b6.resync_hash()
    assert np.array_equal(b5.board, b6.board), "两盘面必须同形"
    assert b5.current_player == 1 and b6.current_player == -1, "行棋方必须相反"
    assert b5.hash() != b6.hash(), "同形但行棋方不同，哈希必须不同"

    # (d) 提子与不落子都影响哈希（增量更新必须计入提子）
    b7 = _fresh()
    b7.play(4 * N + 4)
    h_with = b7.hash()
    b7.undo()
    assert b7.hash() != h_with


# --------------------------------------------------------------------------- #
# 3. play / undo 精确往返
# --------------------------------------------------------------------------- #

def test_hash_changes_only_on_play_and_undo():
    b = _fresh()
    h0 = b.hash()

    # 连续 10 轮，每轮下一手**不同**的合法点（覆盖无提子/有提子等不同增量路径）
    rng = np.random.default_rng(7)
    seen_moves = []
    for rnd in range(10):
        legal = np.flatnonzero(b.get_legal_moves())
        mv = int(legal[rnd % len(legal)]) if legal.size else -1
        if rnd == 3 and legal.size:
            mv = int(rng.choice(legal))
        seen_moves.append(mv)
        assert b.play(mv) is True, f"第 {rnd} 轮 play({mv}) 应成功"
        h1 = b.hash()
        assert h1 != h0, f"第 {rnd} 轮 play 后哈希必须变化"
        assert b.undo() is True
        assert b.hash() == h0, f"第 {rnd} 轮 undo 后必须精确回到初始哈希"
    assert len(set(seen_moves)) >= 5, "测试应覆盖多手不同着法"

    # 带提子的往返：黑提掉一颗白子后 undo，哈希必须精确复原
    c = _fresh()
    c.board[0, 1] = 1
    c.board[2, 1] = 1
    c.board[1, 0] = 1
    c.board[1, 1] = -1
    c.resync_hash()
    h_before = c.hash()
    p_before = c.position_hash()        # 重复判定的键 = position-only
    assert c.play(1 * N + 2) is True, "应能提子"
    assert c.board[1, 1] == 0
    h_capture = c.hash()
    p_capture = c.position_hash()
    assert h_capture != h_before
    assert p_capture != p_before
    assert c.undo() is True
    assert c.board[1, 1] == -1, "undo 必须把被提子放回"
    assert c.hash() == h_before, "提子往返后哈希必须精确复原"
    assert c.position_hash() == p_before, "提子往返后 position 键必须精确复原"
    assert c._would_repeat(p_capture) is False, "undo 后该局面已不在历史中"


# --------------------------------------------------------------------------- #
# 4. clone() 深拷贝
# --------------------------------------------------------------------------- #

def test_clone_is_deep():
    a = _fresh()
    rng = np.random.default_rng(3)
    for _ in range(8):
        a.play(_pick(a, rng))

    b = a.clone()
    assert b.hash() == a.hash(), "clone 后哈希必须相同"
    assert b._pos_hash_history == a._pos_hash_history
    assert b._pos_hash_counts == a._pos_hash_counts
    assert b._pos_hash_history is not a._pos_hash_history, "历史列表必须深拷贝"
    assert b._pos_hash_counts is not a._pos_hash_counts, "历史计数必须深拷贝"
    assert b.position_hash() == a.position_hash()
    assert b._would_repeat(a.position_hash()) is True

    # 只在**一个**实例上继续落子
    b.play(_pick(b, rng))
    assert b.hash() != a.hash(), "克隆体落子后哈希应不同"
    assert a.hash() == a.clone().hash(), "原实例不得被克隆体影响"

    n_hist = len(a._pos_hash_history)
    a_counts = dict(a._pos_hash_counts)
    a_hist = list(a._pos_hash_history)
    for _ in range(3):
        b.play(_pick(b, rng))
    assert len(a._pos_hash_history) == n_hist, "原实例历史长度不得变化"
    assert a._pos_hash_counts == a_counts, "原实例历史集合不得变化"
    assert a._pos_hash_history == a_hist

    # 克隆体自己的 undo 不应影响原实例
    h_a = a.hash()
    hb = b.hash()
    b.undo()
    assert a.hash() == h_a
    assert b.hash() != hb


# --------------------------------------------------------------------------- #
# 5. 真重复局面：三劫循环（6 手 positional superko）
# --------------------------------------------------------------------------- #

# 三劫棋形：ko A / ko C 是白子被黑提，ko B 是黑子被白提。
# A: 白(3,2) 被黑(3,3) 提；B: 黑(3,6) 被白(3,7) 提；C: 白(7,2) 被黑(7,3) 提。
_TRIPLE_KO_STONES = {
    (2, 2): 1, (2, 3): -1, (3, 1): 1, (3, 2): -1, (3, 4): -1, (4, 2): 1, (4, 3): -1,
    (2, 6): -1, (2, 7): 1, (3, 5): -1, (3, 6): 1, (3, 8): 1, (4, 6): -1, (4, 7): 1,
    (6, 2): 1, (6, 3): -1, (7, 1): 1, (7, 2): -1, (7, 4): -1, (8, 2): 1, (8, 3): -1,
}

# 6 手：黑提 A -> 白提 B -> 黑提 C -> 白回提 A -> 黑回提 B -> 白回提 C
# 末手之后盘面 + 行棋方与开局完全相同 => 位置超级劫成立。
_TRIPLE_KO_SEQUENCE = [
    (1, 3 * N + 3),
    (-1, 3 * N + 7),
    (1, 7 * N + 3),
    (-1, 3 * N + 2),
    (1, 3 * N + 6),
    (-1, 7 * N + 2),
]


def _triple_ko_board():
    b = GoBoard(N)
    for (r, c), v in _TRIPLE_KO_STONES.items():
        b.board[r, c] = v
    b.current_player = 1
    b.resync_hash()
    assert b.ko_point == -1, "开局不得有劫禁着"
    return b


def test_undo_restores_repetition_history():
    b = _triple_ko_board()
    start_key = _naive_key(b)
    h0 = b.hash()
    p0 = b.position_hash()          # 重复判定的键 = position-only（PSK）

    # 前 5 手都不是重复
    for i, (who, mv) in enumerate(_TRIPLE_KO_SEQUENCE[:-1]):
        assert b.current_player == who, f"第 {i} 手行棋方不符"
        assert b._would_repeat(b.position_hash_after_move(mv)) is False, \
            f"第 {i} 手不应是重复"
        assert b.play(mv) is True, f"第 {i} 手 {mv} 应合法（简单劫不禁止它）"
        assert _naive_key(b) != start_key, f"第 {i} 手后不得已与开局相同"

    # 第 6 手：闭合成手 —— 谓词必须命中
    who, last = _TRIPLE_KO_SEQUENCE[-1]
    assert b.current_player == who
    cand = b.hash_after_move(last)
    assert cand == h0, "候选局面的哈希应与开局相同"
    pcand = b.position_hash_after_move(last)
    assert pcand == p0, "候选局面的 position 键应与开局相同"
    assert b._would_repeat(pcand) is True, "造成真重复的那一手必须命中谓词"

    # 与哈希实现无关的独立确认：这是真重复，不是伪造
    assert b.play(last) is True
    assert _naive_key(b) == start_key, "末手后局面必须与开局逐字节相同"
    assert b.hash() == h0
    assert b.is_repetition() is True, "当前局面此前出现过 -> is_repetition() 为真"

    # ---- undo 必须把历史精确回滚 ----
    assert b.undo() is True
    h5 = b.hash()
    assert h5 != h0, "undo 后应回到父局（不是闭合成手）"
    assert b.is_repetition() is False, "父局此前未出现过 -> is_repetition() 为假"
    assert b._pos_hash_counts[p0] == 1, "开局染色的计数必须回到 1"
    assert len(b._pos_hash_history) == len(_TRIPLE_KO_SEQUENCE), \
        f"历史长度应回到 {len(_TRIPLE_KO_SEQUENCE)}，实际 {len(b._pos_hash_history)}"
    # 幂等：再走一遍闭合成手，谓词仍应命中（历史未被 undo 破坏）
    assert b._would_repeat(b.position_hash_after_move(last)) is True
    assert b.play(last) is True
    assert b.hash() == h0

    # 一路 undo 回开局，哈希与历史都回到初始状态
    while b.move_number > 0:
        assert b.undo() is True
    assert b.hash() == h0
    assert b.position_hash() == p0
    assert b._pos_hash_history == [p0]
    assert b._pos_hash_counts == {p0: 1}
    assert b.is_repetition() is False

    # ---- True -> False 的翻转：撤掉某染色的**首次**出现后，重演它不再是重复 ----
    s = _fresh()
    ps = s.position_hash()
    assert s.play(4 * N + 4) is True
    p1 = s.position_hash()
    assert s._would_repeat(p1) is True
    assert s.undo() is True
    assert s._would_repeat(p1) is False, "撤销首次出现后该染色已不在历史中"
    assert s._would_repeat(ps) is True, "初始染色始终在历史中"


def test_hash_after_move_matches_actual_play():
    """只读推演必须与 play() 的真实增量结果一致，且绝不改动盘面。"""
    b = _fresh()
    rng = np.random.default_rng(2)
    for _ in range(10):
        legal = np.flatnonzero(b.get_legal_moves(check_suicide=True))
        b.play(int(rng.choice(legal)))

    before = b.board.copy()
    h = b.hash()
    p = b.position_hash()

    # 误传占点：语义未定义，但绝不能擦掉棋子或改动任何状态
    occupied = int(np.flatnonzero(b.board.reshape(-1) != 0)[0])
    b.hash_after_move(occupied)
    b.position_hash_after_move(occupied)
    assert np.array_equal(b.board, before), "hash_after_move 不得改动盘面"
    assert b.hash() == h, "hash_after_move 不得改动哈希"
    assert b.position_hash() == p, "position_hash_after_move 不得改动染色键"
    assert b._pos_hash_history[-1] == p, "两个推演方法都不得改动历史"

    # pass 的推演：行棋方变、染色不变
    assert b.hash_after_move(-1) != h, "pass 只翻转行棋方，哈希仍应变化"
    assert b.position_hash_after_move(-1) == p, "pass 不改变染色，position 键必须不变"

    # 合法着法：推演值 == 真实落子后的键
    mv = _pick(b, rng)
    cand = b.hash_after_move(mv)
    pcand = b.position_hash_after_move(mv)
    assert b.play(mv) is True
    assert b.hash() == cand, "hash_after_move 与 play() 的增量结果必须一致"
    assert b.position_hash() == pcand, "position_hash_after_move 同理"
    assert b.undo() is True
    assert b.hash() == h
    assert b.position_hash() == p


def test_repetition_not_yet_wired_into_legality():
    """本任务范围锁：重复局面尚未接入 get_legal_moves()（P2.6a-2b 才接）。

    现在那手 superko 重复棋仍然可落 —— P2.6a-2b 会把本断言翻转。
    """
    b = _triple_ko_board()
    for who, mv in _TRIPLE_KO_SEQUENCE[:-1]:
        assert b.current_player == who
        assert b.play(mv) is True
    who, last = _TRIPLE_KO_SEQUENCE[-1]
    assert b._would_repeat(b.position_hash_after_move(last)) is True
    assert b.play(last) is True, "本任务不得改变落子合法性"
    assert b.get_legal_moves().sum() > 0


# --------------------------------------------------------------------------- #
# 6. 初始局面在历史集合内
# --------------------------------------------------------------------------- #

def test_history_set_contains_initial_position():
    b = _fresh()
    h0 = b.hash()
    p0 = b.position_hash()
    assert b._pos_hash_history == [p0], "初始空局面的染色必须在历史中"
    assert b._would_repeat(p0) is True, "谓词必须认初始局面"
    assert b.is_repetition() is False, \
        "is_repetition() 排除本局面自身的那一次出现，故初始局面为假"

    # 两手 pass 后行棋方翻转回来 => 盘面+行棋方与开局相同（真重复）
    assert b.play(-1) is True
    assert b.play(-1) is True
    assert b.hash() == h0, "两 pass 后局面应与开局相同"
    assert b.position_hash() == p0
    assert b.is_repetition() is True
    # 计数是 3 而不是 2：历史收的是**染色**键，pass 不改变染色，所以每一手 pass
    # 都把开局染色又记一次（开局 1 次 + 两手 pass 各 1 次）。SSK 下 pass 会翻行棋方，
    # 那时只有「回到开局」的那一手才对得上，计数才是 2。
    assert b._pos_hash_counts[p0] == 3, "初始染色出现次数应为 3（开局 + 两手 pass）"

    b.reset()
    assert b._pos_hash_history == [p0]
    assert b._pos_hash_counts == {p0: 1}
    assert b.is_repetition() is False


# --------------------------------------------------------------------------- #
# 7. 状态计数：num_passes / move_number / is_terminal
# --------------------------------------------------------------------------- #

# 3 路（9 个点）上的**普通对局**：18 手全为实着、无 pass、黑白正常交替，
# 走满 18 = 2·3² 手，于是 move 上限的边界（17 假 / 18 真）在 num_passes == 0 下
# 是干净的 —— 终局只可能由上限触发，不受「两次连续 pass」干扰。
#
# 每一手之后的棋盘染色都**没有出现过**（19 个局面 19 种互不相同的染色），
# 因此这 18 手在 P2.6a-2b 把 Tromp-Taylor 的 PSK 接进合法性之后仍然全部合法
# （测试里逐手用 is_repetition() 守着这条前提）。
#
# 为什么不用 9 路走满 162 手：9 路只有 81 个点，要走 82 手实着就必须靠提子腾地方，
# 而「提子 + 回填」在小盘口上会把染色玩回旧形态（§8 的 5 路夹具只敢走到 48 手就是
# 这个原因）。旧夹具干脆用 27 遍三劫循环硬凑 162 手 —— 那些手**全是**超级劫着法，
# PSK 一接线就会全部非法。
#
# 为什么这个文件仍保留三重劫夹具（§5 的 _triple_ko_board）：它测的是重复**判定**
# 本身（test_undo_restores_repetition_history / test_repetition_predicate_is_position_based
# / test_undo_shares_the_adoption_entry_point），与「着法能否落下」无关，那几处不受
# PSK 接线影响。三劫作为**对拍语料**留给 P2.6b。
_CAP_LINE_3X3 = [1, 4, 3, 6, 5, 8, 0, 2, 1, 7, 5, 3, 2, 0, 2, 1, 5, 3]


def test_num_passes_and_is_terminal():
    b = _fresh()
    assert b.to_play == b.current_player, "to_play 应复用 current_player"
    assert b.num_passes == 0
    assert b.move_number == 0
    assert b.is_terminal() is False
    assert b.is_game_over() is False

    assert b.play(-1) is True
    assert b.num_passes == 1
    assert b.move_number == 1
    assert b.is_terminal() is False
    assert b.is_game_over() is False, "老调用点必须继续可用且一致"

    assert b.play(-1) is True
    assert b.num_passes == 2, "连续两 pass"
    assert b.is_terminal() is True
    assert b.is_game_over() is True, "is_game_over 与 is_terminal 必须一致"

    # 中间夹一手实着 -> 连续 pass 归零
    assert b.play(4 * N + 4) is True
    assert b.num_passes == 0, "实着必须清零连续 pass 计数"
    assert b.move_number == 3
    assert b.is_terminal() is False
    assert b.is_game_over() is False

    # move_number 随 undo 回退
    assert b.undo() is True
    assert b.move_number == 2
    assert b.num_passes == 2, "undo 必须恢复连续 pass 计数"
    assert b.is_terminal() is True
    assert b.undo() is True
    assert b.move_number == 1
    assert b.num_passes == 1
    assert b.undo() is True
    assert b.move_number == 0
    assert b.num_passes == 0
    assert b.undo() is False, "栈空应返回 False"

    # clone() 必须带走状态计数
    c = _fresh()
    c.play(-1)
    c.play(4 * N + 4)
    d = c.clone()
    assert d.move_number == c.move_number
    assert d.num_passes == c.num_passes
    assert d.to_play == c.to_play
    assert d.is_terminal() == c.is_terminal()

    # 显式传入上限
    e = _fresh()
    assert e.is_terminal(max_moves=0) is True, "上限 0 立即终局"
    assert e.is_terminal(max_moves=1) is False
    e.play(4 * N + 4)
    assert e.is_terminal(max_moves=1) is True, "达到上限即终局"

    # move 上限：默认 2*n*n。
    # 终局语义提醒：TT 规则下终局 = **两次连续 pass**（上面已验），**重复不是终局条件**
    # ——重复是非法手。所以下面这段必须 num_passes == 0 才能说「终局由 move 上限触发」。
    #
    # 为什么换成 3 路（见 _CAP_LINE_3X3 的说明）：9 路 2n²=162 手在小盘口上基本只能
    # 靠超级劫着法凑出来，旧夹具就是 27 遍三劫循环 —— P2.6a-2b 接入 PSK 后每一手都非法。
    cap3 = 2 * 3 * 3
    e2 = GoBoard(3)
    for i, mv in enumerate(_CAP_LINE_3X3):
        assert e2.play(mv) is True, f"第 {i} 手 {mv} 应合法（夹具是普通对局）"
        # 夹具的前提：每一手之后的染色都是新的。PSK 接入后这些手仍然是合法着法，
        # is_repetition() 就是这条前提的运行期守卫。
        assert e2.is_repetition() is False, \
            f"第 {i} 手 {mv} 之后染色重现了 —— 夹具依赖「PSK 尚未接入」"
    assert e2.move_number == cap3, f"应能走满 {cap3} 手，实际 {e2.move_number}"
    assert e2.num_passes == 0, "本段不含 pass -> 终局只可能由 move 上限触发"
    assert e2.is_terminal() is True, f"3 路默认 move 上限应为 {cap3}"
    assert e2.is_game_over() is True
    assert e2.undo() is True
    assert e2.move_number == cap3 - 1
    assert e2.is_terminal() is False, f"上限前一手（{cap3 - 1}）不应终局"

    # 上限按盘口缩放，不是写死的常数
    for n in (3, 5, 9, 13, 19):
        assert GoBoard(n).is_terminal(max_moves=cap3) is False, \
            f"{n} 路默认上限 2n²={2 * n * n}，{cap3} 手不该终局"


# --------------------------------------------------------------------------- #
# 生命周期契约：外部盘面接管 = 以此局面为新的一局
# --------------------------------------------------------------------------- #

def test_external_board_replacement_is_adopted_as_new_game():
    """light_rollout 用「新建 GoBoard + 逐字段赋值」做只读推演，不走 clone()。

    外部整体替换 board 数组时，hash() 必须给出**该盘面**的正确哈希，
    否则 P2.6a-2 接入 PSK 后 rollout 会拿空盘哈希判重复，静默把整局下成 pass。

    并且整个状态契约必须自洽：接管 = **以此局面为新的一局**（对局进度归零、
    历史重建为 {当前局面}），而不是「哈希换了、计数与历史留着上一局的」。
    """
    src = _fresh()
    rng = np.random.default_rng(11)
    for _ in range(10):
        src.play(_pick(src, rng))
    assert src.move_number == 10
    assert src.play(-1) is True, "让源局面的连续 pass 计数非零，好检验它被归零"
    assert src.num_passes == 1

    cur = GoBoard(N)
    cur.board = src.board.copy()
    cur.current_player = src.current_player
    cur.ko_point = src.ko_point
    cur.passes = src.passes
    cur.move_history = list(src.move_history)
    # 抄来的是陈旧状态，下面逐项断言接管把它清干净
    assert cur.move_number == 0 and len(cur._pos_hash_history) == 1

    h = cur.hash()
    p = cur.position_hash()
    assert h == src.hash(), "被采纳的局面哈希必须与源局面一致"
    assert p == src.position_hash(), "染色键也必须与源局面一致"

    # ---- 契约：以此局面为新的一局 ----
    assert cur.move_number == 0, "move_number 必须归零，不能停在源局面的计数上"
    assert cur.num_passes == 0, "连续 pass 计数必须归零"
    assert cur.move_history == [], "着法历史属于上一局"
    assert cur._undo_stack == [], "撤销栈属于上一局"
    assert cur._pos_hash_history == [p], "历史重建为 {当前局面}"
    assert cur._pos_hash_counts == {p: 1}
    assert cur._would_repeat(p) is True, "当前局面必须在历史里（它是这一局的开局）"
    assert cur.is_repetition() is False, "开局局面此前没出现过"

    # ---- 接管后按新盘面继续走仍自洽 ----
    mv = _pick(cur, rng)
    assert cur.play(mv) is True
    assert cur.hash() != h
    assert cur.position_hash() == cur._pos_hash_history[-1]
    assert cur.is_repetition() is False
    assert cur.undo() is True
    assert cur.hash() == h
    assert cur.position_hash() == p == cur._pos_hash_history[-1]
    assert cur._pos_hash_counts == {p: 1}
    assert cur.move_number == 0


def test_current_player_rewrite_is_adopted_as_new_game():
    """只改写 current_player 的那一支同样按「以此局面为新局」处理。

    旧实现只重算哈希却**保留**旧历史，于是盘面与自己的历史永久脱钩：
    「翻转 + play + undo」之后 position_hash() != _pos_hash_history[-1]，且当前键根本不是
    _pos_hash_counts 的键 —— `_would_repeat(position_hash())` 对「就是当前局面」的位置返回假。
    """
    b = _fresh()
    assert b.play(4 * N + 4) is True
    assert b.play(2 * N + 2) is True
    assert b.current_player == 1, "两手之后又轮到黑，本测试要靠改写触发接管"

    b.current_player = -1                      # 外部改写行棋方
    h = b.hash()
    assert b._zobrist == h and b._zobrist_player == -1
    p = b.position_hash()
    assert b._pos_zobrist == p, "染色键与行棋方无关，但历史仍按新局重建"
    assert b._pos_hash_history == [p], "历史必须重建为 {当前局面}"
    assert b._pos_hash_counts == {p: 1}
    assert b._would_repeat(p) is True, "当前局面必须在历史里"
    assert b.is_repetition() is False
    assert b.move_number == 0, "对局进度归零"
    assert b.num_passes == 0

    # 翻转 + play + undo：全过程自洽（这正是审查实测崩掉的那条路径）
    mv = 3 * N + 3
    assert b.play(mv) is True
    assert b.position_hash() == b._pos_hash_history[-1], "落子后当前键必须就是历史末项"
    assert b.position_hash() in b._pos_hash_counts
    assert b.undo() is True
    assert b.position_hash() == p == b._pos_hash_history[-1]
    assert b._pos_hash_counts == {p: 1}
    assert b._would_repeat(b.position_hash()) is True
    assert b.is_repetition() is False
    assert b.move_number == 0
    assert b.current_player == -1, "undo 必须回到本局的行棋方"


def test_adoption_keeps_ko_ban_but_drops_stale_legal_cache():
    """ko_point 属于**局面描述**而不是对局进度，接管时保留。

    Tromp-Taylor 规则 6 的 PSK 只看盘面涂色，简单劫是它「只禁紧邻上一手之前那个局面」这条
    更弱的限制 —— 与整盘历史无关，所以从该局面开新局时劫禁着照样成立。
    light_rollout 显式拷贝 ko_point 要的就是这个语义，不能在接管时清掉。
    """
    src = _triple_ko_board()
    who, mv = _TRIPLE_KO_SEQUENCE[0]
    assert src.current_player == who
    assert src.play(mv) is True
    assert src.ko_point >= 0, "夹具必须造出劫禁着"

    cur = GoBoard(N)
    cur.get_legal_moves()                  # 先把**旧盘面**的掩码缓存热起来
    assert cur._legal_cache is not None
    cur._legal_cache[0] = False             # 人为投毒：接管若不失效缓存，0 号点就会漏掉
    cur.board = src.board.copy()
    cur.current_player = src.current_player
    cur.ko_point = src.ko_point

    h = cur.hash()
    assert h == src.hash()
    assert cur.ko_point == src.ko_point, "劫禁着属于局面描述，接管时保留"
    assert cur.move_number == 0 and cur.num_passes == 0
    assert cur._pos_hash_history == [cur.position_hash()]
    legal = cur.get_legal_moves()
    assert legal[0], "接管必须失效属于旧盘面的合法性缓存"
    assert not legal[cur.ko_point], "劫禁着点仍不可下"


def test_undo_shares_the_adoption_entry_point():
    """undo() 与 play()/hash() 走同一条入口（代码与 docstring 一致的那一半）。

    外部整体换掉盘面后，撤销必须**干净地失败**，而不是拿上一局的着法去恢复一颗
    新盘上不存在的棋子、并把刚重建的历史弹空。
    """
    b = _triple_ko_board()
    who, mv = _TRIPLE_KO_SEQUENCE[0]
    assert b.current_player == who and b.play(mv) is True
    assert b._undo_stack and b.move_number == 1

    b.board = np.zeros((N, N), dtype=np.int8)      # 外部整体换盘面
    b.current_player = 1
    assert b.undo() is False, "接管已清空上一局的撤销栈，必须返回 False"
    assert not b.board.any(), "撤销不得改动盘面"
    assert b.hash() == _fresh().hash(), "哈希必须与新盘面一致"
    assert b.move_number == 0
    assert b._pos_hash_history == [b.position_hash()]


def test_in_place_board_mutation_needs_resync():
    """就地改写 board（不换数组对象）无法自动察觉 -> 必须显式 resync_hash()。

    锁住这条契约：谁在测试夹具里手搓棋盘，谁就得调用 resync_hash()。
    """
    b = _fresh()
    h_empty = b.hash()
    b.board[0, 0] = 1          # 就地改写，数组对象未变
    assert b.hash() == h_empty, "就地改写不会被自动察觉（这是已知且被锁定的行为）"
    b.resync_hash()
    ref = _fresh()
    ref.board[0, 0] = 1
    ref.resync_hash()
    assert b.hash() == ref.hash(), "resync_hash 后必须按盘面重算"
    assert b.position_hash() == ref.position_hash()
    # 显式接管与自动接管是同一份实现：对局进度同样归零
    b.play(2 * N + 2)
    assert b.move_number == 1
    b.resync_hash()
    assert b.move_number == 0 and b.num_passes == 0
    assert b._pos_hash_history == [b.position_hash()]


# --------------------------------------------------------------------------- #
# 8. 小盘口（5 路）维度：坐标映射、跨盘口同一把钥匙、2·n² 上限
# --------------------------------------------------------------------------- #

SMALL = 5


def _hand_built(n, stones, to_play):
    """按 (r, c) 摆子并显式接管（测试夹具姿势）。"""
    b = GoBoard(n)
    for (r, c), v in stones.items():
        b.board[r, c] = v
    b.current_player = to_play
    b.resync_hash()
    return b


def test_flat_move_index_maps_to_row_col_on_small_board():
    """非 9 路盘口下 divmod(move, n) 的坐标映射（含撤销往返）。"""
    b = GoBoard(SMALL)
    for i, mv in enumerate((0, 6, 8, 12, 24)):
        assert b.play(mv) is True
        r, c = divmod(mv, SMALL)
        color = 1 if i % 2 == 0 else -1      # 黑白交替，所以断言颜色而不只是非空
        assert b.board[r, c] == color, \
            f"move={mv} 在 {SMALL} 路必须落在 ({r}, {c}) 且是 {'黑' if color > 0 else '白'}"
    # 同一个扁平索引在不同盘口上是**不同**的坐标
    a5, a9 = GoBoard(SMALL), GoBoard(N)
    assert a5.play(6) and a9.play(6)
    assert divmod(6, SMALL) == (1, 1) and divmod(6, N) == (0, 6)
    assert a5.board[1, 1] == 1 and a5.board[0, 1] == 0
    assert a9.board[0, 6] == 1 and a9.board[1, 1] == 0
    assert a5.hash() != a9.hash(), "不同坐标不得撞出同一哈希"

    h0 = GoBoard(SMALL).hash()
    assert a5.undo() is True and a5.hash() == h0, "5 路上的增量撤销必须精确复原"


def test_same_coordinates_hash_identically_on_any_board_size():
    """同一坐标在任何盘口下取到同一把 Zobrist 钥匙 —— 报告 §1 声称的设计优势。

    走**增量路径**验证：在 5 路上真落子的局面，与 9/13/19 路上手摆的同坐标局面
    哈希必须逐位相同（既证明 _board_delta 的 divmod 映射，也证明跨盘口不串扰）。
    """
    seq = [0, 6, 12, 18, 24, 3, 9, 7]
    small = GoBoard(SMALL)
    for mv in seq:
        assert small.play(mv) is True
    h_small = small.hash()
    p_small = small.position_hash()
    stones = {(r, c): int(small.board[r, c])
              for r, c in zip(*np.nonzero(small.board))}
    assert len(stones) == len(seq), "夹具这串着法不该有提子"

    for n in (9, 13, 19):
        ref = _hand_built(n, stones, small.current_player)
        assert ref.hash() == h_small, f"{n} 路盘的同坐标局面哈希必须相同"
        assert ref._pos_hash_history == [p_small]

    # 哈希只认坐标 + 行棋方：行棋方不同则哈希不同（跨盘口同样成立）
    flipped = _hand_built(N, stones, -small.current_player)
    assert flipped.hash() != h_small


def test_zobrist_key_depends_only_on_row_col_and_color():
    """直接钉住「钥匙表按 (r, c) 寻址、与盘口无关」这条不变式。

    报告 §1 声称的优势至今没有任何测试锁着；这里直接查表：同一坐标取到的就是那一项，
    且该项等于 sha256(种子|标签) 的前 8 字节（任何语言都能复算）。跨盘口取到同一把钥匙
    这件事本身由上面那条测试（5 路增量局面 == 19 路手摆局面）从行为上证明。
    """
    for (r, c) in ((0, 0), (1, 2), (2, 3), (4, 4)):
        for color in (1, -1):
            idx = 2 * (r * go_rules._ZOBRIST_STRIDE + c) + (0 if color > 0 else 1)
            label = b"p|%d|%d|%s" % (r, c, b"B" if color > 0 else b"W")
            key = go_rules._zobrist_key(r, c, color)
            assert key == go_rules._ZOBRIST_TABLE[idx], "寻址必须是 (行,列) 而非 r*n+c"
            assert key == go_rules._zobrist_entry(label), "表项必须只由标签决定"
    assert go_rules._zobrist_key(2, 3, 1) != go_rules._zobrist_key(2, 3, -1), "两色不同钥匙"
    assert go_rules._zobrist_key(2, 3, 1) != go_rules._zobrist_key(3, 2, 1), "坐标不共用钥匙"
    # 表必须覆盖到 19 路（否则大盘口会静默撞钥匙）
    assert go_rules._ZOBRIST_STRIDE >= 19


def _greedy_new_move(b):
    """确定性取一手「合法、非自杀、且不造成重复局面」的着法（最小索引优先）。

    过滤重复局面是为了让夹具在 P2.6a-2b 接入 PSK 之后**依然合法**：
    过滤用的键必须是 position-only（`position_hash_after_move`），用含行棋方的
    `hash_after_move` 会永远命中不了历史（历史里存的是染色键），过滤静默失效。
    返回 -1 表示无此着（pass）。
    """
    for mv in np.flatnonzero(b.get_legal_moves(check_suicide=True)).tolist():
        if not b._would_repeat(b.position_hash_after_move(mv)):
            return mv
    return -1


def test_small_board_move_cap_is_two_n_squared():
    """5 路默认上限 = 2·n² = 50 手（不是 9 路那个 162）：上限逐盘口生效。

    用一条**真实可达、全程无重复局面**的 5 路走子（24 实着 + 24 pass = 48 手）把它落到
    具体手数上：默认上限仍未触发，而显式传 48 立刻终局 —— 两者一夹，默认上限落在
    (48, ...]；配合下面按盘口核对公式的断言，5 路这条就是 2·5² = 50。

    为什么停在 48 而不是 50（这条事实本身对 P2.6a-2b 有用）：5 路只有 25 个点，
    24 颗子之后剩下那个点已是**自己的眼**（下上去是自杀），所以第 25 手实着必须靠一次
    提子腾地方 —— 而提子必然让某个局面重复（回提的位置与之前逐字节相同），那是 superko
    着法。P2.6a-2b 把 Tromp-Taylor 的 PSK 接上之后，小盘口的 2·n² 上限实际上近乎死规则。
    （原文此处写的是「重复即终局」，是错的：TT 规则下重复是**非法手**，终局是两次连续 pass。
    结论不变，但原因要改对。）同理，§7 那条上限测试原来用 27 遍三劫循环凑满 9 路的 162 手，
    那些手全是 superko 着法，已在 P2.6a-2a 换成 3 路的普通对局（见 _CAP_LINE_3X3）。

    **对「PSK 是否已接入」的依赖**：本夹具靠 `_greedy_new_move` 过滤重复局面，
    所以 PSK 接入后仍然合法（不需要改）。但下面「48 手 / 25 个染色」这两个数字依赖
    **夹具里没有任何重复局面**这一前提：接入 PSK 之前 `_would_repeat` 用哪把钥匙都
    判不出重复（本段仍绿），接入之后若有人把 `_greedy_new_move` 的键改回 `hash_after_move`，
    过滤会静默失效、这个数字就会变 —— 所以下面直接钉住「25 个互不相同的染色」。
    """
    for n in (SMALL, N, 13):
        c = GoBoard(n)
        assert c.is_terminal(max_moves=0) is True, "上限比较是 move_number >= max_moves"
        assert c.is_terminal() is False, f"{n} 路默认上限 2n²={2 * n * n}，开局不该终局"

    b = GoBoard(SMALL)
    for i in range(24):
        mv = _greedy_new_move(b)
        assert mv >= 0, f"第 {i} 手应还有不重复的实着可下"
        assert b.play(mv) is True
        assert b.num_passes == 0, "实着必须清零连续 pass 计数"
        assert b.is_repetition() is False, "第 %d 手的染色必须是新的" % i
        assert b.play(-1) is True
    assert b.move_number == 48
    assert b.num_passes == 1, "末手是 pass，但没有连续两次 -> 终局只可能由上限触发"
    # 历史存的是**染色**键：24 手实着给出 24 个新染色，加上初始空盘共 25 个；
    # 24 次 pass 不改变染色，因此不新增键（每个实着染色各出现 2 次：一次来自实着、
    # 一次来自随后的 pass）。
    assert len(b._pos_hash_counts) == 25, "开局 + 24 手实着 = 25 个互不相同的染色"
    assert max(b._pos_hash_counts.values()) == 2
    # 末手是 pass，染色没变 => 该染色是第二次出现，is_repetition() 为真。
    # 这**不是**「夹具里下了非法手」：pass 不受 PSK 限制（Tromp-Taylor 规则 6 禁的是
    # 重复棋），所以上面循环里每一手**实着**之后必须为假（那里已逐手断言）。
    assert b.is_repetition() is True
    assert b.is_terminal(max_moves=48) is True, "显式上限 48 立刻终局"
    assert b.is_terminal(max_moves=49) is False
    assert b.is_terminal() is False, "48 手时 5 路默认上限（50）未触发"
    assert b.is_game_over() is False


# --------------------------------------------------------------------------- #
# 9. 重复判定键 = position-only（PSK 修正）
# --------------------------------------------------------------------------- #
#
# **两套键，分工写死**（P2.6a-2a）：
#   hash()          = 棋盘 + 行棋方 -> 通用局面指纹（缓存、诊断、对拍），
#                     **不用于重复判定**；
#   position_hash() = 仅棋盘染色 -> **重复判定专用**（Tromp-Taylor / OpenSpiel 的
#                     position 含义：不含「轮到谁」）。
# 用错的后果：把含行棋方的键拿去判重复 = SSK（situational superko），
# 与 Tromp-Taylor 在「同染色、异手方」的局面上分歧（见下面两个用例）。


def test_position_key_ignores_side_to_play():
    """同棋盘、异手方 -> position_hash() 相同，hash() 不同。"""
    b1 = _fresh()
    b1.play(4 * N + 4)
    b1.play(2 * N + 2)

    b2 = _fresh()
    for (r, c), v in {rc: int(b1.board[rc]) for rc in zip(*np.nonzero(b1.board))}.items():
        b2.board[r, c] = v
    b2.current_player = -b1.current_player
    b2.resync_hash()

    assert np.array_equal(b1.board, b2.board), "两盘面必须同形"
    assert b1.current_player != b2.current_player, "行棋方必须相反"

    assert b1.position_hash() == b2.position_hash(), \
        "PSK：position 只指棋盘染色，与轮到谁无关"
    assert b1.hash() != b2.hash(), "通用指纹含行棋方，必须仍能区分二者"


def test_repetition_predicate_is_position_based():
    """「同染色 + 异手方」在 PSK 下**也算**重复 —— 这是 SSK 与 PSK 的分歧点。

    夹具用 §5 的三劫开局（真实对局局面，不是手工摆的空盘）。
    分歧最小可见的一手是 **pass**：pass 不改变染色却翻转行棋方，
    SSK 判它是新局面，PSK 判它重复同一染色。
    """
    b = _triple_ko_board()
    init_pos = b.position_hash()
    assert b.is_repetition() is False, "开局染色此前只出现过一次（就是它自己）"

    assert b.play(-1) is True, "黑 pass：染色不变，行棋方翻成白"
    assert b.position_hash() == init_pos, "pass 不改变棋盘染色"
    assert b._would_repeat(init_pos) is True, "PSK：同一染色已在历史里 -> 命中"
    assert b.is_repetition() is True, "PSK：该染色本次是第二次出现"

    # 同一染色 + **相反手方**的另一个 GoBoard：判定答案必须与上面一致。
    # 旧实现（SSK）在这里会给「不重复」—— 那正是本任务要修的分歧。
    twin = _triple_ko_board()
    twin.current_player = -b.current_player
    twin.resync_hash()
    assert twin.position_hash() == b.position_hash(), "染色相同 -> position 键相同"
    assert twin.hash() != b.hash(), "手方不同 -> 通用指纹不同"
    assert twin.is_repetition() is False, "twin 的历史只含它自己这一条"
    assert twin._would_repeat(b.position_hash()) is True

    # 撤销 pass：同一染色的计数回到 1 -> 不再是重复
    assert b.undo() is True
    assert b.position_hash() == init_pos
    assert b.is_repetition() is False
    assert b._would_repeat(init_pos) is True, "开局染色仍在历史里"

    # 三劫循环本身的闭合成手（真实着法）在 position 键下同样命中
    for who, mv in _TRIPLE_KO_SEQUENCE[:-1]:
        assert b.current_player == who
        assert b._would_repeat(b.position_hash_after_move(mv)) is False
        assert b.play(mv) is True
    who, last = _TRIPLE_KO_SEQUENCE[-1]
    assert b.current_player == who
    assert b._would_repeat(b.position_hash_after_move(last)) is True, \
        "6 手闭合成手：PSK 必须命中（键不含手方也照样命中）"


def test_position_hash_lifecycle():
    """play / undo / clone / reset 对 position 键的维护，与 hash() 逐项对齐。"""
    b = _fresh()
    p0 = b.position_hash()
    assert p0 == _fresh().position_hash(), "position 键同样必须确定"
    assert b._pos_hash_history == [p0], "初始空局面的 position 键在历史里"

    # 提子路径（增量更新最容易漏项的一条）
    cap = _fresh()
    cap.board[0, 1] = 1
    cap.board[2, 1] = 1
    cap.board[1, 0] = 1
    cap.board[1, 1] = -1
    cap.resync_hash()
    cap_before = cap.position_hash()
    assert cap.play(1 * N + 2) is True and cap.board[1, 1] == 0, "应能提子"
    cap_mid = cap.position_hash()
    assert cap_mid != cap_before
    assert cap.undo() is True
    assert cap.position_hash() == cap_before, "提子往返后 position 键必须精确复原"

    # 连续 10 轮 play/undo，每轮精确回到父局键
    for rnd in range(10):
        legal = np.flatnonzero(b.get_legal_moves())
        mv = int(legal[rnd % len(legal)])
        assert b.play(mv) is True
        assert b.position_hash() != p0, f"第 {rnd} 轮 play 后 position 键必须变化"
        assert b._pos_hash_history[-1] == b.position_hash()
        assert b.undo() is True
        assert b.position_hash() == p0, f"第 {rnd} 轮 undo 后必须精确回到父局键"

    # clone() 深拷贝：两边继续落子互不影响
    src = _fresh()
    for _ in range(8):
        src.play(_pick(src, np.random.default_rng(_)))
    cp = src.position_hash()
    clone = src.clone()
    assert clone.position_hash() == cp
    assert clone._pos_hash_history == src._pos_hash_history
    assert clone._pos_hash_history is not src._pos_hash_history, "历史列表必须深拷贝"
    assert clone._pos_hash_counts is not src._pos_hash_counts, "历史计数必须深拷贝"
    src_hist, src_counts = list(src._pos_hash_history), dict(src._pos_hash_counts)
    mv = _pick(clone, np.random.default_rng(99))
    assert clone.play(mv) is True
    assert clone.position_hash() != cp
    assert src.position_hash() == cp and src._pos_hash_history == src_hist
    assert src._pos_hash_counts == src_counts, "克隆体落子不得污染源棋盘的历史"

    # undo 在克隆体上也不得影响源棋盘
    assert clone.undo() is True
    assert clone.position_hash() == cp
    assert src.position_hash() == cp

    # reset() 回到初始 position 键
    b.reset()
    assert b.position_hash() == p0
    assert b._pos_hash_history == [p0]
    assert b._pos_hash_counts == {p0: 1}

    # 两个实例走同一序列 -> position 键逐步相同（确定性）
    a1, a2 = _fresh(), _fresh()
    rng1, rng2 = np.random.default_rng(12345), np.random.default_rng(12345)
    for step in range(12):
        m1, m2 = _pick(a1, rng1), _pick(a2, rng2)
        assert m1 == m2
        assert a1.play(m1) and a2.play(m2)
        assert a1.position_hash() == a2.position_hash(), f"第 {step} 步 position 键不同"


def test_position_key_is_size_independent():
    """position 键只认 (行, 列, 颜色) —— 与盘口、与行棋方都无关。"""
    assert GoBoard(SMALL).position_hash() == GoBoard(N).position_hash() == 0, \
        "空盘染色在所有盘口上是同一个键"

    seq = [0, 6, 12, 18, 24, 3, 9, 7]
    small = GoBoard(SMALL)
    for mv in seq:
        assert small.play(mv) is True
    p_small = small.position_hash()
    stones = {(r, c): int(small.board[r, c])
              for r, c in zip(*np.nonzero(small.board))}
    assert len(stones) == len(seq), "夹具这串着法不该有提子"

    for n in (N, 13, 19):
        ref = _hand_built(n, stones, small.current_player)
        assert ref.position_hash() == p_small, f"{n} 路盘的同坐标局面 position 键必须相同"
        assert ref._pos_hash_history == [p_small]

    # 跨盘口 + 异手方：position 键不变，通用指纹变
    flipped = _hand_built(N, stones, -small.current_player)
    assert flipped.position_hash() == p_small, "position 键与手方无关，跨盘口也一样"
    assert flipped.hash() != small.hash(), "通用指纹含手方，必须不同"

    # 同点异色 -> position 键必须不同（否则 PSK 把换色当重复）
    other = _hand_built(SMALL, {rc: -v for rc, v in stones.items()}, small.current_player)
    assert other.position_hash() != p_small


def test_docstring_distinguishes_the_two_keys():
    """**文档锁**（如标题所述：这是结构锁，不是行为断言）。

    「哪把钥匙用于重复判定」是本段最容易被回归的 knowledge：两套哈希长得几乎一样，
    用错 = SSK = 与 Tromp-Taylor 分歧，而这种用错在功能测试里几乎抓不到
    （只在罕见的奇数染色循环上才出错）。所以直接锁 docstring。
    """
    pos_doc = GoBoard.position_hash.__doc__ or ""
    full_doc = GoBoard.hash.__doc__ or ""
    assert "重复判定" in pos_doc, "position_hash() 必须写明它服务于重复判定"
    assert "hash()" in pos_doc, "position_hash() 必须点名与 hash() 的分工"
    assert "不用于重复判定" in full_doc, \
        "hash() 必须明说自己不用于重复判定（否则后来人会拿它判 PSK）"
    assert "position_hash()" in full_doc, "hash() 必须指向 position_hash()"
    assert pos_doc.strip() != full_doc.strip(), "两把钥匙的 docstring 不能是同一段文本"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nALL TESTS PASSED")
