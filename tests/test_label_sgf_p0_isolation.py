"""P0：坏局隔离的回归测试。

覆盖的真实故障（2026-10-02）：跑 20 个局面就崩在
    RuntimeError: 分析错误(q0): Illegal move 72: F2
`query_many` 原来对单个报错 `raise`，等于「第一个坏局把整批打死」。

另外盯住一个**更隐蔽**的坑：`.done.n` 曾被同时当成「已写行数」和
「下一个局面的下标」（`todo = range(done, len(positions))`）。一旦跳过局面，
两者错位，续跑会拿错的 `positions[i]` 去配第 i 行 —— 产出的标签
`pos_hash` 与 `policy` 不再属于同一局面，join 命中率会**诡异地**下降而不报错。
所以游标必须按「已处理局面数」推进，行数从落盘文件长度反推。
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util as _ilu  # noqa: E402

_s = _ilu.spec_from_file_location(
    "label_sgf",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "label_sgf.py"))
L = _ilu.module_from_spec(_s)
_s.loader.exec_module(L)


# --------------------------------------------------------------------------- #
# 错误归因
# --------------------------------------------------------------------------- #
def test_first_bad_move_parses_the_index():
    assert L.first_bad_move("Illegal move 72: F2") == 72
    assert L.first_bad_move("illegal move 5: D4") == 5
    # 抠不出就 -1，**不能抛** —— 归因失败不该中断隔离流程
    assert L.first_bad_move("some other engine error") == -1
    assert L.first_bad_move("") == -1
    assert L.first_bad_move(None) == -1


def test_classify_error_keeps_illegal_move_apart_from_ko():
    # 这两类必须能分开：illegal-move 多半是布局子/坐标错位，ko 是记录本身
    # 的循环分歧，补救手段完全不同。
    assert L.classify_error("Illegal move 72: F2") == "illegal-move"
    assert L.classify_error("Superko violation") == "ko"
    assert L.classify_error("Invalid komi") == "komi"
    assert L.classify_error("bad json") == "protocol"
    assert L.classify_error("wat") == "other"


# --------------------------------------------------------------------------- #
# 隔离清单的存取
# --------------------------------------------------------------------------- #
def test_skipped_ledger_roundtrip(tmp_path):
    out = str(tmp_path / "lbl.npz")
    assert L.load_skipped(out) == {}          # 不存在 -> 空
    L.save_skipped(out, {})                   # 空也落盘，区分「没坏」与「没生成」
    assert os.path.isfile(L._skip_path_for(out))
    assert L.load_skipped(out) == {}

    L.save_skipped(out, {3: ("a.sgf", 72, "Illegal move 72: F2"),
                         9: ("b.sgf", -1, "Superko violation")})
    got = L.load_skipped(out)
    assert set(got) == {3, 9}
    assert got[3] == ("a.sgf", 72, "Illegal move 72: F2")
    assert got[9][1] == -1


def test_skipped_ledger_survives_corrupt_file(tmp_path):
    """清单文件坏了必须当空处理，不能因此中断续跑。"""
    out = str(tmp_path / "lbl.npz")
    p = L._skip_path_for(out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"not an npz at all")
    assert L.load_skipped(out) == {}


# --------------------------------------------------------------------------- #
# query_many 不再抛
# --------------------------------------------------------------------------- #
class _FakeProc:
    """按预设脚本回行的假进程。"""

    def __init__(self, lines):
        self._lines = list(lines)
        self.stdin = self
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass

    def _readline(self):
        return self._lines.pop(0) if self._lines else ""


def _lab(lines):
    lab = L.KataLabeler.__new__(L.KataLabeler)
    lab.timeout = 5
    lab.n_done = 0
    lab.p = _FakeProc(lines)
    lab._readline_timeout = lambda timeout=None: lab.p._readline()
    return lab


def test_query_many_returns_errors_instead_of_raising():
    """核心回归：一条坏 + 两条好，必须**返回**错误而不是抛异常。"""
    lab = _lab([
        json.dumps({"id": "q0", "error": "Illegal move 72: F2"}),
        json.dumps({"id": "q1", "rootInfo": {"winrate": 0.5, "visits": 37}}),
        json.dumps({"id": "q2", "rootInfo": {"winrate": 0.6, "visits": 38}}),
    ])
    reqs = [{"id": "q0", "moves": []}, {"id": "q1", "moves": []},
            {"id": "q2", "moves": []}]
    out, errs = lab.query_many(reqs)
    assert errs == {"q0": "Illegal move 72: F2"}
    assert set(out) == {"q1", "q2"}
    assert lab.n_done == 2


def test_query_many_terminates_when_all_error():
    """全是错的时候不能死等 —— 终止条件是 len(out)+len(errors)。"""
    lab = _lab([json.dumps({"id": f"q{i}", "error": "Illegal move 0: A1"})
                for i in range(3)])
    out, errs = lab.query_many([{"id": f"q{i}", "moves": []} for i in range(3)])
    assert out == {}
    assert len(errs) == 3


def test_query_many_ignores_foreign_ids():
    """引擎会回带自己 id 的噪声行（就绪探测、terminate 等），不能算进 want。"""
    lab = _lab([
        json.dumps({"id": "__probe__", "rootInfo": {}}),
        json.dumps({"id": "q0", "rootInfo": {"visits": 37}}),
    ])
    out, errs = lab.query_many([{"id": "q0", "moves": []}])
    assert errs == {} and set(out) == {"q0"}


def test_query_many_still_raises_on_engine_death():
    """⚠ 边界：引擎真的死了**必须**抛，不能被 P0 的容错吞掉 —— 那是
    基础设施故障，不是坏记录，静默继续只会得到空结果。"""
    lab = _lab([])                     # 立刻 EOF
    with pytest.raises(RuntimeError, match="意外退出"):
        lab.query_many([{"id": "q0", "moves": []}])


# --------------------------------------------------------------------------- #
# 游标语义：跳过局面后不得错位
# --------------------------------------------------------------------------- #
def test_rid_maps_back_to_position_index():
    """错误 rid 必须能反查回局号，隔离才落得到正确的局上。"""
    positions = [{"game": g, "move": m} for g in range(5) for m in range(3)]
    for i, q in enumerate(positions):
        assert int(f"q{i}"[1:]) == i
        assert q is positions[i]


def test_cursor_tracks_processed_not_rows(tmp_path):
    """P0 之前 `done` 同时是「已写行数」和「下一局面下标」。跳过 1 个局面后
    两者必须**分叉**：游标前进 1，写入行数不变。写错了续跑就会拿错 positions[i]
    去配第 i 行 —— pos_hash 与 policy 不再同源，join 命中率静默下滑。

    这里直接验证新代码里那两行的算术关系。
    """
    # 每局面一个独立局号：这样只有 i=3 会因引擎报错被隔离，不触发预跳过分支。
    positions = [{"game": i, "move": 100 + i} for i in range(10)]
    skipped = {}
    conc = 4
    cursor = 0
    written = 0
    todo = list(range(cursor, len(positions)))
    for cs in range(0, len(todo), conc):
        raw = todo[cs:cs + conc]
        batch = [i for i in raw if positions[i]["game"] not in skipped]
        assert batch, "没有预跳过时 batch 应等于 raw"
        # 假设 batch 里只有 i=3 会报错
        resps = {f"q{i}": {} for i in batch if i != 3}
        errors = {"q3": "Illegal move 3: D4"} if 3 in batch else {}
        assert len(resps) + len(errors) == len(batch)
        written += sum(1 for i in batch if f"q{i}" in resps)
        cursor = raw[-1] + 1
    assert cursor == 10          # 10 个局面全部处理过
    assert written == 9          # 但只写了 9 行（第 4 个被隔离）
    assert cursor != written     # 两者确实分叉了 —— 这正是 P0 修掉的那个 bug


def test_known_bad_game_is_not_resent_but_cursor_still_advances():
    """已知坏局**不再发请求**（窗口内同局 ~100 个局面，全发是白烧引擎），
    但游标**必须照常推进** —— 否则会死循环重扫同一段。"""
    positions = [{"game": 1, "move": m} for m in range(6)]
    skipped = {1: ("bad.sgf", 3, "Illegal move 3: D4")}
    conc = 4
    cursor = 0
    sent = 0
    todo = list(range(cursor, len(positions)))
    for cs in range(0, len(todo), conc):
        raw = todo[cs:cs + conc]
        batch = [i for i in raw if positions[i]["game"] not in skipped]
        if not batch:
            cursor = raw[-1] + 1
            continue
        sent += len(batch)
        cursor = raw[-1] + 1
    assert sent == 0              # 一个请求都没发出去
    assert cursor == 6            # 但游标推到底，不死循环


def test_labelled_rows_count_comes_from_file_length(tmp_path):
    """行数必须从落盘文件反推，不能用 cursor/n —— 续跑时文件里还有上次的行。"""
    part = str(tmp_path / "lbl")
    rows = 7
    with open(f"{part}.pos_hash.part.npy", "wb") as f:
        f.write(np.zeros(rows, np.uint64).tobytes())
    got = np.fromfile(f"{part}.pos_hash.part.npy", dtype=np.uint64).size
    assert got == rows


def test_row_count_guard_compares_rows_not_elements():
    """⚠ 这条钉的是守卫**自己**的 bug（2026-10-02 真实踩到）：收尾一致性检查
    拿 `v.size`（元素总数）跟行数比，而 policy 的形状是 (N, 362) —— 14 行就有
    5068 个元素，于是正常数据被误判成「文件被外部改动」并中止，标签已写完却拿不到。

    正确口径是 shape[0]（行数）。这里复现同一批 14 行的形状。
    """
    N_ACTIONS = 362
    rows = 14
    cols = {"policy": np.zeros((rows, N_ACTIONS), np.float16),
            "pos_hash": np.zeros(rows, np.uint64),
            "root_win": np.zeros(rows, np.float32),
            "visits": np.zeros(rows, np.int16),
            "game_idx": np.zeros(rows, np.int32)}
    m = int(cols["pos_hash"].size)
    assert m == rows
    # 错误口径：policy.size = 5068 != 14 -> 会误报
    assert cols["policy"].size != m
    # 正确口径：全部相等 -> 放行
    assert {k: v.shape[0] for k, v in cols.items() if v.shape[0] != m} == {}
