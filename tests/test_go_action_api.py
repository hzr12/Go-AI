"""
GoBoard 动作空间 API 测试（P2.6a-2b-2，OpenSpiel 风格动作空间 + 消费者同步收尾）。

被测对象是 `src/game/go_rules.py` 的「动作空间」段：`GoBoard.PASS` / `num_actions()` /
`action_to_coord()` / `coord_to_action()` / `action_to_string()` / `string_to_action()` /
`legal_actions()` / `is_legal()`。它们**不重新定义规则**，只是
`get_legal_moves()` 那张掩码的动作空间投影 —— 所以本文件的主轴是
「与掩码、与 `play()` 双向一致」，而不是重新钉一遍规则。

覆盖面（5 路与 9 路各一组，CPU、秒级）：
  1. `test_action_space_contract`                 PASS/num_actions/坐标↔动作/文本往返 +
                                                   非法输入抛 ValueError
  2. `test_legal_actions_sorted_and_includes_pass` 升序、含 PASS、其余与掩码逐位一致
  3. `test_is_legal_matches_mask`                 对**全部**动作编号与掩码一致
  4. `test_is_legal_does_not_materialize_the_mask` **行为性红**：桩掉 get_legal_moves
  5. `test_pass_is_exempt_from_superko`            PASS 恒合法且不查 PSK
  6. `test_suicide_and_ko_stay_illegal_via_is_legal` 新 API 与新规则一致，没另开一套判定
  7. `test_string_roundtrip_matches_parse_move_str` 记法只有一套（不许造第二套）
  8. `test_legal_actions_matches_play_on_mutations` 随机合法着法序列上与 `play()` 对拍
  9. `apply_action()`（P2.7b）: PASS 槽不静默落空 / 与方言换算逐手同答案 /
     越界不抛 / 非法着法一样拒 / record 透传 —— 共 5 个用例

三条最容易写错、因而单独钉住的不变量：
  - **`PASS = n*n`，而棋盘方言里 pass 是 `-1`**。两者**不可比**：
    `is_legal(-1)` 为 False（越界）而 `play(-1)` 为 True；`is_legal(PASS)` 为 True 而
    `play(PASS)` 为 False。用例 1/3/4 把这几格逐个钉住。
  - **PASS 豁免 PSK 必须是结构性的**（用例 5）。pass 不改变染色，
    `position_hash_after_move(-1)` 恒等于当前键、该键必然已在历史里，谓词**恒为真**；
    一旦接上 PSK，所有 pass 都会被判非法，对局永远无法终局。
  - **`is_legal()` 必须是单点判定**（用例 4，行为性红）。写成
    `bool(self.get_legal_moves()[a])` 会把单点 O(1) 的 API 退化成每次调用物化整张
    n*n 掩码（O(n²)），在逐候选调用的场景上是数量级的差别。

夹具与 `tests/test_go_rules_legality.py` **不共享任何符号**（同一姿势、各自重建）：
手工摆子 + `resync_hash()` 显式接管。
"""
import os
import sys

import numpy as np
import pytest

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


# 左上角「双钩」棋形，5 路与 9 路通用（嵌进 9 路只会让周围白块**气更多**，
# 不会把「提不掉」变成「提得掉」）：
#     c0 c1 c2 c3 ...
#  r0  .  O  O  .
#  r1  O  .  O  .
#  r2  O  O  .  .
#  黑下 (1,1)：四邻全是白，且四块白都还有别的气（提不掉）-> 纯自杀。
#  黑下 (0,0)：两个邻点都是白，同样提不掉 -> 也是纯自杀。
_WEDGE_STONES = {(0, 1): -1, (0, 2): -1, (1, 0): -1, (1, 2): -1, (2, 0): -1, (2, 1): -1}
_WEDGE_SUICIDE = [(0, 0), (1, 1)]

# 标准单子劫（`_SUICIDE` 那种角形 + 一手提子），5 路与 9 路通用：
#     c0 c1 c2 c3
#  r0  .  X  O  .
#  r1  O  O  X  O
#  r2  .  X  O  .
#  黑下 (1,2) 提掉 (1,1)：黑子只剩 (1,1) 一口气 -> 成劫；
#  白立即回提 (1,1) 会**复现「黑提之前」的染色** -> 被 PSK 禁掉（与 ko_point 无关）。
_KO_STONES = {(0, 1): 1, (0, 2): -1, (1, 0): 1, (1, 1): -1, (1, 3): -1,
              (2, 1): 1, (2, 2): -1}
_KO_CAPTURE = (1, 2)      # 黑提子成劫
_KO_RECAPTURE = (1, 1)    # 白立即回提（PSK 非法）


def _midgame(n, seed, n_moves):
    """固定种子的随机自对弈中盘（掩码 / play() / 动作空间三方对拍用的通用局面）。"""
    b = GoBoard(n)
    rng = np.random.default_rng(seed)
    for _ in range(n_moves):
        legal = np.flatnonzero(b.get_legal_moves())
        if not legal.size:
            b.play(-1)
            continue
        b.play(int(rng.choice(legal)))
    return b


# --------------------------------------------------------------------------- #
# 1. 动作空间契约
# --------------------------------------------------------------------------- #

def test_action_space_contract():
    """`PASS` / `num_actions()` / 坐标↔动作 / 动作↔文本 的契约与失败模式。

    四组断言：
      1. **`PASS` 是类级常量且值随盘口变**（5 路 25 / 9 路 81）：它必须**不是**逐实例
         赋值的属性（那会出现第二个真相源），也必须**不能**在类上取（`GoBoard.PASS`
         抛 AttributeError）—— 类上没有「默认盘口」可言，让它大声报错好过让 5 路
         静默用 19 路的 361，那会让动作空间与扁平坐标错位 336。**写入与删除同样要
         报错**（`b.PASS = …` / `del b.PASS`）：只定义 `__get__` 的描述符是非数据的，
         实例 `__dict__` 排在它前面，一次赋值就能静默遮蔽掉整个槽位。
      2. 坐标 ↔ 动作严格互逆，且动作编号与掩码下标**同一套**（`r*n+c`）。
      3. 动作 ↔ 文本严格互逆（含 `'pass'` 与四个角），并钉住**列字母在前**的 SGF 记法
         （`(行2, 列3)` 必须是 `"dc"` 而不是 `"cd"`）。
      4. 非法输入抛 `ValueError`：非法文本、越界动作、越界坐标、**以及 `-1`** ——
         最后这一条是「`-1` 不是动作空间编码」的可执行版本。
    """
    for n in (N5, N9):
        b = GoBoard(n)
        n2 = n * n
        assert "PASS" in vars(GoBoard), "PASS 必须是类级常量"
        assert "PASS" not in vars(b), "PASS 不得逐实例赋值（那就有第二个真相源）"
        assert b.PASS == n2, f"{n} 路：PASS 必须等于 n*n"
        assert b.num_actions() == n2 + 1, f"{n} 路：num_actions() 必须等于 n*n+1"
        assert b.num_actions() == b.PASS + 1
        with pytest.raises(AttributeError):
            GoBoard.PASS            # 类上访问必须大声报错（见上）
        # 写入 / 删除同样必须报错：只定义 __get__ 的话 _PassSlot 是**非数据描述符**，
        # 而实例 __dict__ 排在描述符之前 —— `board.PASS = 5` 会**静默**挂上一个 int 5
        # 把描述符整个遮蔽掉，此后 PASS 与 n*n 永久分叉且零报错（症状是 pass 被当成
        # 棋盘上的一个点）。守卫必须是结构性的，不能靠调用方记得别写。
        with pytest.raises(AttributeError):
            b.PASS = n2 + 7
        with pytest.raises(AttributeError):
            del b.PASS
        assert b.PASS == n2, "失败的写入不得留下痕迹（PASS 仍由 board_size 决定）"

        # ---- 坐标 ↔ 动作 ----
        for (r, c) in ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1),
                       (1, 1), (n // 2, n // 2)):
            a = b.coord_to_action(r, c)
            assert a == r * n + c, "动作编号必须与掩码下标同一套（r*n+c）"
            assert b.action_to_coord(a) == (r, c), f"{n} 路：({r},{c}) 往返不一致"
        # 动作编号 < PASS 时与掩码下标完全一致（掩码长度恒为 n*n）
        assert len(b.get_legal_moves()) == n2
        for a in (0, 1, n2 - 1):
            assert b.action_to_coord(a)[0] * n + b.action_to_coord(a)[1] == a

        # ---- 动作 ↔ 文本（含 pass 与四个角）----
        assert b.action_to_string(b.PASS) == "pass"
        assert b.string_to_action("pass") == b.PASS
        assert b.string_to_action("PASS") == b.PASS, "记法大小写不敏感（与 parse_move_str 一致）"
        assert b.string_to_action("") == b.PASS, "'' 在 parse_move_str 里就是 pass"
        assert b.string_to_action("resign") == b.PASS, "resign 在 parse_move_str 里也是 pass"
        for a in range(n2):
            s = b.action_to_string(a)
            assert len(s) == 2 and s.isalpha() and s.islower()
            assert b.string_to_action(s) == a, f"{n} 路：动作 {a} 的文本往返不一致"
        assert b.action_to_string(0) == "aa"
        assert b.action_to_string(n - 1) == chr(ord('a') + n - 1) + "a"
        assert b.action_to_string((n - 1) * n) == "a" + chr(ord('a') + n - 1)
        assert b.action_to_string(n2 - 1) == chr(ord('a') + n - 1) * 2
        # 列字母在前（SGF 风格）：(行 2, 列 3) -> "dc"
        assert b.action_to_string(b.coord_to_action(2, 3)) == "dc"

        # ---- 失败模式 ----
        for bad in ("zz", "abc", "a", "1a", "!!", "e" * 40):
            with pytest.raises(ValueError):
                b.string_to_action(bad)
        for bad_action in (-1, -n2, n2 + 1, n2 * 3):
            with pytest.raises(ValueError):
                b.action_to_string(bad_action)
        # PASS 是**合法**动作（只是没有坐标），所以只有换算坐标时才抛
        assert b.action_to_string(b.PASS) == "pass"
        for bad_action in (-1, n2, n2 + 7):
            with pytest.raises(ValueError):
                b.action_to_coord(bad_action)     # PASS 也没有坐标
        for bad_coord in ((-1, 0), (0, -1), (n, 0), (0, n)):
            with pytest.raises(ValueError):
                b.coord_to_action(*bad_coord)


# --------------------------------------------------------------------------- #
# 2. legal_actions()
# --------------------------------------------------------------------------- #

def test_legal_actions_sorted_and_includes_pass():
    """`legal_actions()` = 升序、含 PASS、其余与掩码**逐位**一致。

    掩码长度恒为 n*n（**没有** PASS 槽），所以 PASS 只能由本方法追加；
    追加是无条件的：哪怕盘面已经无处可下（掩码全 False），列表也仍是 `[PASS]`
    —— 那正是「没有合法落子就 pass」的规则表达（用例会真的造出这种局面）。
    """
    positions = [
        ("空盘", lambda n: GoBoard(n)),
        ("中盘", lambda n: _midgame(n, 20260926, 20)),
        ("假眼（唯一空点是自杀）", lambda n: _hand_built(
            n, {(r, c): 1 for r in range(n) for c in range(n) if (r, c) != (2, 2)})),
    ]
    for n in (N5, N9):
        n2 = n * n
        for label, make in positions:
            b = make(n)
            mask = np.asarray(b.get_legal_moves()).copy()
            la = b.legal_actions()
            assert la == sorted(la), f"{n} 路 {label}：legal_actions() 必须升序"
            assert la[-1] == b.PASS and b.PASS in la, \
                f"{n} 路 {label}：PASS 必须在列表里"
            assert len(set(la)) == len(la), f"{n} 路 {label}：不得有重复项"
            assert len(la) == int(mask.sum()) + 1, \
                f"{n} 路 {label}：长度必须等于掩码真点数 + 1（只有 PASS 是额外的）"
            assert la[:-1] == [int(i) for i in np.flatnonzero(mask)], \
                f"{n} 路 {label}：除 PASS 外必须与掩码逐位一致"
            assert all(0 <= a < n2 for a in la[:-1]), "落子动作必须落在 [0, n*n)"
    # 掩码全 False 的极端局面：仍然只有 PASS，且它**无条件**在里面
    dead = _hand_built(N5, {(r, c): 1 for r in range(N5) for c in range(N5)
                            if (r, c) != (2, 2)})
    assert not dead.get_legal_moves().any(), "夹具前提：这一盘无任何合法落点"
    assert dead.legal_actions() == [dead.PASS], "无处可下时只该剩 PASS"


# --------------------------------------------------------------------------- #
# 3. is_legal() 与掩码逐点一致
# --------------------------------------------------------------------------- #

def test_is_legal_matches_mask():
    """对**全部**动作编号，`is_legal(a)` 与 `bool(get_legal_moves()[a])` 同答案。

    枚举范围刻意比掩码宽两头：`[0, n*n]` 覆盖每个落子动作与 PASS，`n*n+1` 与
    `-1 / -2` 覆盖越界。越界一律 **False**（`is_legal` 是「问合法性」，不抛异常；
    会抛的是 `action_to_coord` —— 「问一个非法编号合不合法」与「把非法编号换算成
    坐标」是两件事）。

    局面覆盖四类：一般中盘（大片空点 + 真实 PSK 历史）、纯自杀点、单子劫回提点、
    掩码全 False 的「无处可下」盘。少一类就有一类分歧抓不到。
    """
    def _ko_board(n):
        b = _hand_built(n, _KO_STONES, to_play=1)
        assert b.play(_idx(n, *_KO_CAPTURE)) is True, "夹具：成劫那一手必须能落下"
        return b

    positions = []
    for n in (N5, N9):
        positions += [
            (n, "空盘", GoBoard(n)),
            (n, "中盘（随机自对弈）", _midgame(n, 20260926, 20)),
            (n, "自杀点所在盘", _hand_built(n, _WEDGE_STONES, to_play=1)),
            (n, "单子劫（提子后）", _ko_board(n)),
            (n, "无处可下（掩码全 False）",
             _hand_built(n, {(r, c): 1 for r in range(n) for c in range(n)
                             if (r, c) != (2, 2)})),
        ]
    for n, label, b in positions:
        n2 = n * n
        mask = np.asarray(b.get_legal_moves()).copy()
        for a in range(n2 + 2):
            expected = bool(mask[a]) if a < n2 else (a == n2)
            got = b.is_legal(a)
            assert got is expected, (
                f"{n} 路 {label}：动作 {a} 掩码说 {expected}，is_legal 说 {got}")
        for bogus in (-1, -2, -n2, n2 + 1, n2 * 2):
            assert b.is_legal(bogus) is False, \
                f"{n} 路 {label}：越界动作 {bogus} 必须恒为 False（不抛）"
        assert b.is_legal(b.PASS) is True, "PASS 恒合法"


# --------------------------------------------------------------------------- #
# 4. is_legal() 必须是单点判定（行为性红）
# --------------------------------------------------------------------------- #

def test_is_legal_does_not_materialize_the_mask(monkeypatch):
    """把 `get_legal_moves` 换成一个「被调用就抛异常」的桩，`is_legal()` 仍必须正常返回。

    **这是本任务唯一的行为性红**：朴素实现 `return bool(self.get_legal_moves()[a])`
    在这个桩下必然抛 AssertionError —— 它证明 `is_legal` 走的是**单点判定**
    （≤4 个邻点 + 至多一次提子推演），而不是「每次调用物化整张 n*n 掩码」。
    代价必须真的差出来：单点 O(1) 的 API 一旦物化掩码就退化成 O(n²)，
    在逐候选调用的场景（UI 逐点问、rollout 采样、推理时过滤）上是数量级的差别。

    桩打在**类**上（不是实例属性）：那样连 `type(self).get_legal_moves(self)`
    这种绕法也照样抓到。

    同时断言答案**仍然正确** —— 只证明「没调掩码」不够，还得证明它算的是同一套规则。
    """
    n = N5
    b = _hand_built(n, _WEDGE_STONES, to_play=1)
    mask = np.asarray(b.get_legal_moves()).copy()          # 先在未污染的盘上取真值
    expected = {a: bool(mask[a]) for a in range(n * n)}
    expected[b.PASS] = True
    before = b.board.tobytes()

    calls = []

    def _boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "is_legal() 物化了全掩码（调用了 get_legal_moves）—— 必须单点判定")

    monkeypatch.setattr(GoBoard, "get_legal_moves", _boom)
    b._legal_cache = None

    checked = 0
    for a in list(expected) + [-1, n * n + 1]:
        assert b.is_legal(a) is expected.get(a, False), \
            f"动作 {a}：单点判定的答案与掩码不一致"
        checked += 1
    assert checked == n * n + 3
    assert not calls, f"get_legal_moves 被调用了 {len(calls)} 次"
    # 状态零变更：单点判定是只读的（PSK 那一支的试落子已还原）
    assert b.board.tobytes() == before, "is_legal() 不得改动盘面"


# --------------------------------------------------------------------------- #
# 5. PASS 豁免 PSK
# --------------------------------------------------------------------------- #

def test_pass_is_exempt_from_superko():
    """PASS 恒合法，且**结构上**不查 PSK —— 否则对局永远无法终局。

    机制上不可能搞错：pass 不改变棋盘染色，所以 `position_hash_after_move(-1)` 恒等于
    当前键，而当前键必然已在重复局面历史里（每次成功落子都记它）—— 查下去**恒为真**。
    Tromp-Taylor 规则 6 禁的是重复**棋**，不是 pass；终局由两次连续 pass 表达。
    所以豁免必须是显式的：`is_legal()` 对 PASS 短路、`legal_actions()` 无条件追加、
    `play(-1)` 在 PSK 判定之前就 return。

    P2.6a-2c 已钉 `play()` 侧那条；本用例钉**动作空间侧**：
      - 连着 4 次 pass 之后当前染色在历史里出现 5 次，谓词仍恒为真，
        而 `is_legal(PASS)` / `legal_actions()` 里的 PASS 照旧合法；
      - `num_passes` 递增、`is_terminal()` 在第二次连续 pass 后变真；
      - PASS 不进落子编号区间（`is_legal(-1)` 为 False 是**另一个**编码的事）。
    """
    b = GoBoard(N9)
    assert b._pos_hash_counts[b.position_hash()] == 1, "初始局面在历史里出现 1 次"
    for i in range(1, 5):
        assert b.play(-1) is True, f"第 {i} 次 pass 必须成功"
        assert b.num_passes == i, "num_passes 必须递增"
        assert b.is_legal(b.PASS) is True, "PASS 恒合法（哪怕染色已重复多次）"
        assert b.PASS in b.legal_actions(), "PASS 恒在 legal_actions() 里"
        assert not b.is_terminal() or i >= 2
    assert b._pos_hash_counts[b.position_hash()] == 5, "4 次 pass 后当前染色出现 5 次"
    assert b._would_repeat(b.position_hash_after_move(-1)) is True, \
        "前提：若 PASS 也查 PSK，谓词在这里恒为真（所以豁免必须是显式的）"
    assert b.is_legal(b.PASS) is True, "染色已重复 5 次，PASS 仍必须合法"
    assert b.is_terminal() is True, "两次连续 pass 即终局（不靠 pass 非法表达）"

    # 落子会清零 pass 计数，所以终局判定与 PSK 无关
    c = GoBoard(N9)
    c.play(-1)
    c.play(_idx(N9, 3, 3))
    assert c.num_passes == 0 and not c.is_terminal()

    # PASS 不在落子编号区间里：动作空间的 pass 是 n*n，不是 -1
    assert b.PASS == N9 * N9
    assert b.is_legal(-1) is False, "-1 在动作空间里越界（棋盘方言的 pass 不算）"
    assert b.play(-1) is True, "…而在棋盘方言里 -1 就是合法 pass：两者不可比"


# --------------------------------------------------------------------------- #
# 6. 新 API 与新规则一致（没另开一套判定）
# --------------------------------------------------------------------------- #

def test_suicide_and_ko_stay_illegal_via_is_legal():
    """自杀点与简单劫回提点在 `is_legal()` 上必须与掩码**同答案**。

    这一条是「`is_legal()` 没有另写一套规则」的证据：它不查 `ko_point`、不查自杀，
    而是复用掩码那三条判据（空点 / 禁自杀 / PSK）。反面对照同时钉住两侧的边界：
      - 「自杀但提子」的点**必须仍合法**（禁自杀的另一半，别把它一起禁掉）；
      - 劫在别处落子一手之后回提**恢复合法**（PSK 判的是染色重复，不是「成没成劫」）。
    """
    for n in (N5, N9):
        # ---- 自杀 ----
        b = _hand_built(n, _WEDGE_STONES, to_play=1)
        for pt in _WEDGE_SUICIDE:
            mv = _idx(n, *pt)
            assert b.is_legal(mv) is False, f"{n} 路：自杀点 {pt} 必须非法"
            assert not b.get_legal_moves()[mv], f"{n} 路：掩码与 is_legal 必须同答案"
            assert b.clone().play(mv) is False, f"{n} 路：play() 也必须拒绝 {pt}"
        # 真眼 / 有气的点照常合法（反向回归：别把禁自杀做成「禁一切」）
        assert b.is_legal(_idx(n, 0, 0)) is False
        assert b.is_legal(_idx(n, 3, 3)) is True, f"{n} 路：空旷处的点必须合法"

        # ---- 单子劫 ----
        k = _hand_built(n, _KO_STONES, to_play=1)
        capture = _idx(n, *_KO_CAPTURE)
        recapture = _idx(n, *_KO_RECAPTURE)
        assert k.is_legal(capture) is True, "形成劫的那一手本身合法"
        assert k.play(capture) is True
        assert k.ko_point == recapture, "夹具必须造出单劫（只读信息位）"
        assert k.is_legal(recapture) is False, \
            f"{n} 路：单子劫回提必须非法（禁它的是 PSK，不是 ko_point 字段）"
        assert not k.get_legal_moves()[recapture]
        assert k.clone().play(recapture) is False
        # 判罚不靠字段：清掉 ko_point 也解禁不了
        forced = k.clone()
        forced.ko_point = -1
        assert forced.is_legal(recapture) is False
        # 在别处落一手之后，染色已变，PSK 不再命中 -> 回提恢复合法
        far = _idx(n, n - 1, n - 1)
        assert k.play(far) is True
        assert k.is_legal(recapture) is True, "别处落子后回提必须恢复合法"


# --------------------------------------------------------------------------- #
# 7. 记法只有一套
# --------------------------------------------------------------------------- #

def test_string_roundtrip_matches_parse_move_str():
    """`string_to_action` 与既有 `parse_move_str` 必须给出一致结果 —— 不造第二套记法。

    `parse_move_str` 是既有公开 API（P2.6a-2b-2 明确不删它），新 API 与它并存；
    两者的差别**只允许**是失败处理（`parse_move_str` 返回 `(ok, mv)`，新 API 抛
    `ValueError`）。接受的记法、字母序、列/行顺序、大小写敏感度必须逐项一致。

    两条断言一起才完整：
      1. 对**每一个**盘内坐标，`string_to_action(action_to_string(a))` 与
         `parse_move_str` 给出**同一个**扁平编号；
      2. 对**盘外**的串（越界坐标、长度不对、非法字符），`parse_move_str` 说
         `ok=False` **且** `string_to_action` 抛 `ValueError` —— 两边同时拒。
    唯一允许的差异是 pass：`parse_move_str("pass")` 返回 -1（棋盘方言），
    `string_to_action("pass")` 返回 `PASS`（动作空间）—— 这正是两套编码的分界。
    """
    for n in (N5, N9):
        b = GoBoard(n)
        for r in range(n):
            for c in range(n):
                s = b.action_to_string(b.coord_to_action(r, c))
                ok, mv = b.parse_move_str(s)
                assert ok is True, f"{n} 路：parse_move_str 必须认 {s!r}"
                assert b.string_to_action(s) == mv == b.coord_to_action(r, c), \
                    f"{n} 路：{s!r} 在两套 API 下必须给出同一个编号"
        # 盘外的串：两边同时拒
        for s in ("zz", "a", "abc", "!!", "1a", "e" * 30):
            if len(s) == 2 and all("a" <= ch <= chr(ord('a') + n - 1) for ch in s):
                continue
            ok, _mv = b.parse_move_str(s)
            if ok:
                continue                       # parse_move_str 认它就必须给出同一个值
            with pytest.raises(ValueError):
                b.string_to_action(s)
        # pass：唯一允许的差异（-1 vs n*n）
        for s in ("pass", "", "resign"):
            ok, mv = b.parse_move_str(s)
            assert ok is True and mv == -1, "parse_move_str 的 pass 是棋盘方言的 -1"
            assert b.string_to_action(s) == b.PASS, "动作空间的 pass 是 n*n"


# --------------------------------------------------------------------------- #
# 8. 动作空间 ↔ play() 的端到端巡检
# --------------------------------------------------------------------------- #

def test_legal_actions_matches_play_on_mutations():
    """随机合法着法序列上，`legal_actions()` 里除 PASS 外的每一项都能被 `play()` 接受。

    这是跨 P2.6a-2c 收口成果的端到端巡检：掩码（`legal_actions` 的来源）与 `play()`
    必须在**真实对局演化**中始终同答案，而不只是人工夹具上同答案。

    **PASS 单独处理**：`play()` 说棋盘方言（pass = -1，`play(PASS)` 越界返回 False），
    动作编号必须先换算。所以本用例检查「每一项落子动作」，pass 那一侧由
    `test_pass_is_exempt_from_superko` 钉。
    """
    import time
    t0 = time.perf_counter()
    checked = 0
    for n, seed, n_moves in ((N5, 7, 14), (N9, 11, 10)):
        rng = np.random.default_rng(seed)
        b = GoBoard(n)
        for _ in range(n_moves):
            la = b.legal_actions()
            base = b.clone()
            for a in la[:-1]:                 # 除 PASS 外每一项
                probe = base.clone()
                assert probe.play(a) is True, (
                    f"{n} 路：legal_actions() 给出 {a}（{a // n},{a % n}），"
                    f"play() 却拒绝它")
                checked += 1
            move = int(rng.choice(la[:-1])) if len(la) > 1 else b.PASS
            assert b.play(-1 if move == b.PASS else move) is True
    elapsed = time.perf_counter() - t0
    assert checked > 100, f"巡检点数太少（{checked}），夹具可能空转"
    assert elapsed < 20.0, f"巡检耗时 {elapsed:.1f}s，太慢（应当秒级）"


# --------------------------------------------------------------------------- #
# apply_action()（P2.7b）：动作空间的落子入口
# --------------------------------------------------------------------------- #
def test_apply_action_pass_slot_does_not_silently_do_nothing():
    """**本方法存在的全部理由**：`play(PASS)` 返回 False（越界），不是 pass。

    漏掉 `a == PASS` 那一支的消费者会写 `play(a)`，于是「想 pass」变成
    「什么都没发生」—— 返回 False 还容易被当成「这手被判非法」，于是 pass
    被静默吞掉、对局无法终局。所以这里逐格钉住 PASS 槽。
    """
    for n in (N5, N9):
        b = GoBoard(n)
        assert b.PASS == n * n
        assert b.play(b.PASS) is False, "棋盘方言里 play(PASS) 必须仍是非法越界"
        assert b.apply_action(b.PASS) is True, "apply_action 必须把 PASS 落成 pass"
        assert b.num_passes == 1, "PASS 槽必须真的记成一次 pass"
        assert b.move_number == 1


def test_apply_action_matches_play_after_dialect_conversion():
    """`apply_action(a)` 必须与 `play(-1 if a == PASS else a)` **逐手同答案**。

    随机合法着法序列上逐步对拍：返回值、盘面、连续 pass 计数、落子数全都要一致。
    """
    for n, seed, n_moves in ((N5, 3, 12), (N9, 4, 10)):
        rng = np.random.default_rng(seed)
        a_board = GoBoard(n)
        b_board = GoBoard(n)
        for step in range(n_moves):
            acts = [a for a in a_board.legal_actions() if a != a_board.PASS]
            if not acts:
                break
            a = int(rng.choice(acts + [a_board.PASS]))
            dialect = -1 if a == a_board.PASS else a
            got = a_board.apply_action(a)
            want = b_board.play(dialect)
            assert got is want, f"{n} 路第 {step} 手 action={a}：apply={got} play={want}"
            assert np.array_equal(a_board.board, b_board.board), (
                f"{n} 路第 {step} 手 action={a} 之后盘面分叉")
            assert a_board.num_passes == b_board.num_passes
            assert a_board.move_number == b_board.move_number
        assert a_board.move_number > 0, "夹具空转了？"


def test_apply_action_rejects_out_of_range_without_raising():
    """越界（含棋盘方言的 -1）→ False，**不抛**；与 is_legal 的边界口径一致。"""
    b = GoBoard(N5)
    for bad in (-1, -7, b.num_actions(), b.num_actions() + 3):
        assert b.apply_action(bad) is False, f"越界动作 {bad} 应返回 False"
        assert b.is_legal(bad) is False, f"越界动作 {bad} 的 is_legal 也应是 False"
    assert b.move_number == 0 and b.num_passes == 0, "被拒不得留下任何状态变更"


def test_apply_action_rejects_illegal_legal_space_moves_like_play():
    """占点 / 自杀 / PSK 重复：apply_action 必须与 play() **一样拒**。

    这里是「动作空间不许比棋盘方言更宽松」的守门人：apply_action 若图省事写成
    `if action == PASS: ... else: self.play(action)` 之后又自己加判据，就会出现
    第二套规则。
    """
    b = GoBoard(N9)
    # 造一个占点：先落一子，再用同一个动作编号落第二次
    mv = int(np.flatnonzero(b.get_legal_moves())[0])
    assert b.apply_action(mv) is True
    assert b.apply_action(mv) is False, "占点必须被拒"
    # 随机巡检：对每个 legal_actions() 里的非 PASS 项，apply_action 的接受集合
    # 必须与 play() 在克隆体上的一致
    base = b.clone()
    checked = 0
    for a in base.legal_actions():
        if a == base.PASS:
            continue
        assert b.clone().apply_action(a) is base.clone().play(a)
        checked += 1
    assert checked > 10, f"巡检点数太少（{checked}），夹具可能空转"


def test_apply_action_is_record_aware():
    """`record=False` 必须透传 —— 推演路径（record=False 后不撤销）依赖它。"""
    b = GoBoard(N9)
    b.apply_action(int(np.flatnonzero(b.get_legal_moves())[0]), record=False)
    assert b._undo_stack == [], "record=False 不得压撤销栈"
    assert b.undo() is False

if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nALL TESTS PASSED")
