"""续跑指纹的回归测试（Phase 0a）。

要防的真实故障（2026-10-02）
--------------------------
`scripts/label_sgf.py` 原来的守卫是**单向**的：

    if cursor > len(positions): raise SystemExit(...)

于是把 `game-frac 0.02` 改成全量时，`positions` 从 219,343 涨到
13,360,400，而游标 219,343 **不比它大** ⇒ 守卫不拦 ⇒ 脚本认为「前
219,343 个局面已完成」而直接跳过 ⇒ **那批标签永久缺失且不报任何错**。

这是「陈旧游标静默丢数据」类故障，和 `build_soft_index.py` 踩过的
「陈旧缓存静默挂错标签」同源。指纹覆盖一切能改变 `positions` 的输入。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util as _ilu  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_s = _ilu.spec_from_file_location(
    "label_sgf", os.path.join(_ROOT, "scripts", "label_sgf.py"))
L = _ilu.module_from_spec(_s)
_s.loader.exec_module(L)


class _Args:
    """只提供指纹用到的字段。"""

    def __init__(self, **kw):
        self.move_lo = 100
        self.move_hi = 200
        self.max_games = 0
        self.game_frac = 0.02
        self.seed = 0
        self.limit = 5000
        for k, v in kw.items():
            setattr(self, k, v)


def _fp(tmp_path, **kw):
    d = tmp_path / "sgf"
    d.mkdir(exist_ok=True)
    return L.positions_fingerprint(_Args(**kw), str(d), 133604, 219343)


# --------------------------------------------------------------------------- #
# 指纹覆盖：每个能改变 positions 的参数都必须进指纹
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,value", [
    ("move_lo", 120),        # 窗口变了 -> 候选局面集合变
    ("move_hi", 180),
    ("max_games", 5000),     # 抽局方式变
    ("game_frac", 0.5), # 就是这个：0.02 -> 全量，静默丢数据的那一步
    ("seed", 7),
    ("limit", 0),
])
def test_fingerprint_changes_with_every_param(tmp_path, field, value):
    assert _fp(tmp_path) != _fp(tmp_path, **{field: value})


def test_fingerprint_covers_corpus_identity(tmp_path):
    """语料换了也必须重建游标 —— 否则「第 i 行」指向的局面已经不是当初那个。"""
    a = _fp(tmp_path)
    d = tmp_path / "sgf"
    os.utime(d, (0, 0))                       # 改 mtime
    b = L.positions_fingerprint(_Args(), str(d), 133604, 219343)
    assert a["corpus"]["path"] == b["corpus"]["path"]
    assert a["corpus"]["mtime_ns"] != b["corpus"]["mtime_ns"]


def test_fingerprint_records_position_count(tmp_path):
    """n_positions 进指纹：它一变，「第 i 行」的定义就变了。"""
    a = _fp(tmp_path)
    b = L.positions_fingerprint(_Args(), str(tmp_path / "sgf"), 133604, 13360400)
    assert a["n_positions"] != b["n_positions"]


def test_fingerprint_is_json_serializable_and_stable(tmp_path):
    """必须能 json 落盘、且同输入两次生成完全相同（否则每次跑都判为不一致）。"""
    a = _fp(tmp_path)
    assert json.loads(json.dumps(a, sort_keys=True)) == a
    assert _fp(tmp_path) == a


# --------------------------------------------------------------------------- #
# 比对行为
# --------------------------------------------------------------------------- #
def test_first_run_has_no_fingerprint_and_is_allowed(tmp_path):
    out = str(tmp_path / "lbl.npz")
    cursor, ok = L.check_resume_fingerprint(out, _fp(tmp_path))
    assert ok is True and cursor == 0        # 首次跑 = 正常起点


def test_matching_fingerprint_resumes(tmp_path):
    out = str(tmp_path / "lbl.npz")
    fp = _fp(tmp_path)
    L.save_resume_fingerprint(out, fp)
    cursor, ok = L.check_resume_fingerprint(out, fp)
    assert ok is True


def test_changed_game_frac_is_rejected(tmp_path):
    """ 本次修复的核心场景：game-frac 变大必须报错，而不是静默续跑。"""
    out = str(tmp_path / "lbl.npz")
    L.save_resume_fingerprint(out, _fp(tmp_path))          # 之前跑的是 0.02
    with pytest.raises(SystemExit) as e:
        L.check_resume_fingerprint(out, _fp(tmp_path, game_frac=1.0))
    msg = str(e.value)
    assert "指纹不一致" in msg and "game_frac" in msg
    assert "静默" in msg                                    # 说明了危害


def test_error_lists_only_the_differing_fields(tmp_path):
    """差异清单必须精准 —— 把全部字段都列出来等于没指向问题。"""
    out = str(tmp_path / "lbl.npz")
    L.save_resume_fingerprint(out, _fp(tmp_path))
    with pytest.raises(SystemExit) as e:
        L.check_resume_fingerprint(out, _fp(tmp_path, move_lo=120))
    msg = str(e.value)
    assert "move_lo" in msg
    assert "game_frac" not in msg                          # 没变的别刷屏


def test_fresh_env_var_allows_restart(tmp_path):
    """SOFT_TAG_FRESH=1 是显式「从头」的出口。"""
    out = str(tmp_path / "lbl.npz")
    L.save_resume_fingerprint(out, _fp(tmp_path))
    old = dict(os.environ)
    os.environ["SOFT_TAG_FRESH"] = "1"
    try:
        cursor, ok = L.check_resume_fingerprint(out, _fp(tmp_path, game_frac=1.0))
        assert ok is False and cursor == 0
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_fresh_on_missing_fingerprint_still_starts_at_zero(tmp_path):
    out = str(tmp_path / "lbl.npz")
    old = dict(os.environ)
    os.environ["SOFT_TAG_FRESH"] = "1"
    try:
        cursor, ok = L.check_resume_fingerprint(out, _fp(tmp_path))
        assert ok is False and cursor == 0
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_corrupt_fingerprint_refuses_to_guess(tmp_path):
    """坏指纹 = 无法判断 positions 是不是同一份 ⇒ 必须报错，不能当「首次跑」。"""
    out = str(tmp_path / "lbl.npz")
    with open(L._fp_path(out), "w", encoding="utf-8") as f:
        f.write("{ 这不是 json")
    with pytest.raises(SystemExit) as e:
        L.check_resume_fingerprint(out, _fp(tmp_path))
    assert "读取失败" in str(e.value)


def test_save_fingerprint_is_atomic(tmp_path):
    """落盘必须是原子的 —— 崩溃留下的半份 JSON 会让下次续跑直接报错。"""
    out = str(tmp_path / "lbl.npz")
    fp = _fp(tmp_path)
    L.save_resume_fingerprint(out, fp)
    assert not os.path.exists(L._fp_path(out) + ".tmp")
    assert json.load(open(L._fp_path(out), encoding="utf-8")) == fp


# --------------------------------------------------------------------------- #
# 旧数据兼容：仓库里已有的 .done.n 没有指纹
# --------------------------------------------------------------------------- #
def test_legacy_done_n_without_fingerprint_still_resumes(tmp_path):
    """ 升级兼容：已经跑过的批次只有 `.done.n`、没有 `.done.json`。

    此时 `check_resume_fingerprint` 返回 `ok=True`（走「首次跑」那支），
    调用方据此**照常读 `.done.n`** ⇒ 已有游标继续有效，不会让用户重跑一遍。

    代价是：升级后的**第一次**续跑拿不到参数级校验（指纹此刻才刚写下）。
    从第二次起就有完整校验了 —— 这是刻意的取舍，不能为了严格而作废既有进度。
    """
    out = str(tmp_path / "lbl.npz")
    part = os.path.splitext(out)[0]
    with open(part + ".done.n", "w") as f:
        f.write("219343")
    cursor, ok = L.check_resume_fingerprint(out, _fp(tmp_path))
    assert ok is True
    # 调用方逻辑：ok=True 时游标取自 .done.n
    assert cursor == 0                          # 本函数不读游标，只表态「可续跑」
