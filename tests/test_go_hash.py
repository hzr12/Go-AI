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

另锁两条本任务的生命周期契约:
  - 外部整体替换 board 数组（light_rollout 的只读推演）必须被自动采纳
  - 重复局面尚未接入 get_legal_moves()（P2.6a-2 才接），本任务行为零变化
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        "for mv in (0, 4, 40, 80, 41):\n"
        "    b.play(mv)\n"
        "    hs.append(b.hash())\n"
        "print(','.join(str(h) for h in hs))\n"
    ) % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    b = GoBoard(9)
    mine = [b.hash()]
    for mv in (0, 4, 40, 80, 41):
        assert b.play(mv)
        mine.append(b.hash())

    outs = set()
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           text=True, timeout=300)
        assert r.returncode == 0, f"子进程失败: {r.stderr[-2000:]}"
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"两个子进程给出不同哈希: {outs}"
    theirs = [int(x) for x in outs.pop().split(",")]
    assert theirs == mine, f"跨进程哈希不可复现: {theirs} != {mine}"
    assert all(isinstance(h, int) and 0 <= h < 2 ** 64 for h in mine), \
        "hash() 必须返回 [0, 2**64) 的 int"


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
    assert c.play(1 * N + 2) is True, "应能提子"
    assert c.board[1, 1] == 0
    h_capture = c.hash()
    assert h_capture != h_before
    assert c.undo() is True
    assert c.board[1, 1] == -1, "undo 必须把被提子放回"
    assert c.hash() == h_before, "提子往返后哈希必须精确复原"
    assert c._would_repeat(h_capture) is False, "undo 后该局面已不在历史中"


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
    assert b._would_repeat(a.hash()) is True

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

    # 前 5 手都不是重复
    for i, (who, mv) in enumerate(_TRIPLE_KO_SEQUENCE[:-1]):
        assert b.current_player == who, f"第 {i} 手行棋方不符"
        assert b._would_repeat(b.hash_after_move(mv)) is False, \
            f"第 {i} 手不应是重复"
        assert b.play(mv) is True, f"第 {i} 手 {mv} 应合法（简单劫不禁止它）"
        assert _naive_key(b) != start_key, f"第 {i} 手后不得已与开局相同"

    # 第 6 手：闭合成手 —— 谓词必须命中
    who, last = _TRIPLE_KO_SEQUENCE[-1]
    assert b.current_player == who
    cand = b.hash_after_move(last)
    assert cand == h0, "候选局面的哈希应与开局相同"
    assert b._would_repeat(cand) is True, "造成真重复的那一手必须命中谓词"

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
    assert b._pos_hash_counts[h0] == 1, "开局局面的计数必须回到 1"
    assert len(b._pos_hash_history) == len(_TRIPLE_KO_SEQUENCE), \
        f"历史长度应回到 {len(_TRIPLE_KO_SEQUENCE)}，实际 {len(b._pos_hash_history)}"
    # 幂等：再走一遍闭合成手，谓词仍应命中（历史未被 undo 破坏）
    assert b._would_repeat(b.hash_after_move(last)) is True
    assert b.play(last) is True
    assert b.hash() == h0

    # 一路 undo 回开局，哈希与历史都回到初始状态
    while b.move_number > 0:
        assert b.undo() is True
    assert b.hash() == h0
    assert b._pos_hash_history == [h0]
    assert b._pos_hash_counts == {h0: 1}
    assert b.is_repetition() is False

    # ---- True -> False 的翻转：撤掉某局面的**首次**出现后，重演它不再是重复 ----
    s = _fresh()
    hs = s.hash()
    assert s.play(4 * N + 4) is True
    h1 = s.hash()
    assert s._would_repeat(h1) is True
    assert s.undo() is True
    assert s._would_repeat(h1) is False, "撤销首次出现后该局面已不在历史中"
    assert s._would_repeat(hs) is True, "初始局面始终在历史中"


def test_hash_after_move_matches_actual_play():
    """只读推演必须与 play() 的真实增量结果一致，且绝不改动盘面。"""
    b = _fresh()
    rng = np.random.default_rng(2)
    for _ in range(10):
        legal = np.flatnonzero(b.get_legal_moves(check_suicide=True))
        b.play(int(rng.choice(legal)))

    before = b.board.copy()
    h = b.hash()

    # 误传占点：语义未定义，但绝不能擦掉棋子或改动任何状态
    occupied = int(np.flatnonzero(b.board.reshape(-1) != 0)[0])
    b.hash_after_move(occupied)
    assert np.array_equal(b.board, before), "hash_after_move 不得改动盘面"
    assert b.hash() == h, "hash_after_move 不得改动哈希"
    assert b._pos_hash_history[-1] == h, "hash_after_move 不得改动历史"

    # pass 的推演
    assert b.hash_after_move(-1) != h, "pass 只翻转行棋方，哈希仍应变化"

    # 合法着法：推演值 == 真实落子后的哈希
    mv = _pick(b, rng)
    cand = b.hash_after_move(mv)
    assert b.play(mv) is True
    assert b.hash() == cand, "hash_after_move 与 play() 的增量结果必须一致"
    assert b.undo() is True
    assert b.hash() == h


def test_repetition_not_yet_wired_into_legality():
    """本任务范围锁：重复局面尚未接入 get_legal_moves()（P2.6a-2 才接）。

    现在那手 superko 重复棋仍然可落 —— P2.6a-2 会把本断言翻转。
    """
    b = _triple_ko_board()
    for who, mv in _TRIPLE_KO_SEQUENCE[:-1]:
        assert b.current_player == who
        assert b.play(mv) is True
    who, last = _TRIPLE_KO_SEQUENCE[-1]
    assert b._would_repeat(b.hash_after_move(last)) is True
    assert b.play(last) is True, "本任务不得改变落子合法性"
    assert b.get_legal_moves().sum() > 0


# --------------------------------------------------------------------------- #
# 6. 初始局面在历史集合内
# --------------------------------------------------------------------------- #

def test_history_set_contains_initial_position():
    b = _fresh()
    h0 = b.hash()
    assert b._pos_hash_history == [h0], "初始空局面必须在历史中"
    assert b._would_repeat(h0) is True, "谓词必须认初始局面"
    assert b.is_repetition() is False, \
        "is_repetition() 排除本局面自身的那一次出现，故初始局面为假"

    # 两手 pass 后行棋方翻转回来 => 盘面+行棋方与开局相同（真重复）
    assert b.play(-1) is True
    assert b.play(-1) is True
    assert b.hash() == h0, "两 pass 后局面应与开局相同"
    assert b.is_repetition() is True
    assert b._pos_hash_counts[h0] == 2, "开局局面出现次数应为 2"

    b.reset()
    assert b._pos_hash_history == [h0]
    assert b._pos_hash_counts == {h0: 1}
    assert b.is_repetition() is False


# --------------------------------------------------------------------------- #
# 7. 状态计数：num_passes / move_number / is_terminal
# --------------------------------------------------------------------------- #

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
    # 9 路只有 81 个点，靠随机落子凑不满 162 手（会连着 pass 提前终局）；这里复用
    # 三劫循环 27 遍 = 162 手**全为实着**，num_passes 恒为 0 —— 于是终局只可能由
    # move 上限触发，边界（161 假 / 162 真）才是干净的。
    cap = 2 * N * N
    e2 = _triple_ko_board()
    h0 = e2.hash()
    reps = cap // len(_TRIPLE_KO_SEQUENCE)
    for _ in range(reps):
        for who, mv in _TRIPLE_KO_SEQUENCE:
            assert e2.current_player == who
            assert e2.play(mv) is True
    assert e2.move_number == cap, f"应能走满 {cap} 手，实际 {e2.move_number}"
    assert e2.num_passes == 0, "本段不含 pass"
    assert e2.hash() == h0, "27 遍循环后应回到开局局面（哈希须自洽）"
    assert e2.is_terminal() is True, f"默认 move 上限应为 {cap}"
    assert e2.is_game_over() is True
    assert e2._pos_hash_counts[h0] == reps + 1, "开局局面应出现 reps+1 次"
    assert e2.undo() is True
    assert e2.move_number == cap - 1
    assert e2.is_terminal() is False, f"上限前一手（{cap - 1}）不应终局"


# --------------------------------------------------------------------------- #
# 生命周期契约：外部替换 board 数组
# --------------------------------------------------------------------------- #

def test_external_board_replacement_is_adopted():
    """light_rollout 用「新建 GoBoard + 逐字段赋值」做只读推演，不走 clone()。

    外部整体替换 board 数组时，hash() 必须给出**该盘面**的正确哈希，
    否则 P2.6a-2 接入 PSK 后 rollout 会拿空盘哈希判重复，静默把整局下成 pass。
    """
    src = _fresh()
    rng = np.random.default_rng(11)
    for _ in range(10):
        src.play(_pick(src, rng))

    cur = GoBoard(N)
    cur.board = src.board.copy()
    cur.current_player = src.current_player
    cur.ko_point = src.ko_point
    cur.passes = src.passes
    cur.move_history = list(src.move_history)
    assert cur.hash() == src.hash(), "被采纳的局面哈希必须与源局面一致"
    # 采纳后按新盘面继续走仍自洽
    mv = _pick(cur, rng)
    before = cur.hash()
    assert cur.play(mv) is True
    assert cur.hash() != before
    assert cur.undo() is True
    assert cur.hash() == src.hash(), "undo 必须回到被采纳的局面"


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


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nALL TESTS PASSED")
