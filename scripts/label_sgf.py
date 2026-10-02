"""用 KataGo analysis 引擎给自有 SGF 局面打搜索标签（阶段 2）。

为什么走引擎而不是自己的 MCTS
------------------------------
· 引擎有 `useGraphSearch`（内置 MCGS），我们实测过自己实现的换位率只有 8.6%，
  且要重做虚拟损失/树复用 —— 引擎里现成且更快。
· 引擎的 `moveInfos[].visits` 就是 KataGo 的 π（访问分布），与论文口径一致。

输出
----
一个 npz，每行一个被搜索的局面：
    pos_hash  uint64  (N,)      位置 hash（与 probe0_join.py 同口径）
    policy    float16 (N,362)  软标签：visits 归一化，362 = 361 点 + pass
    root_win  float32 (N,)      根胜率（0..1，黑方视角）
    score_mean/score_stdev     根目数期望/标准差
    visits    int16   (N,)      实际达成的 visits（可能 < 请求值）
    game_idx  int32   (N,)      来自哪个 SGF（便于回溯）
    move_idx  int32   (N,)      该局第几手

join 契约
---------
`pos_hash` 与 `scripts/probe0_join.py` 的 `pos_hash_block` **完全同口径**
(board+to_play+ko)，因此可与 `full.npz` 直接 join；不可靠的 `game_id` 一律不用。

坐��口径
--------
analysis 引擎回的 `move` 是 GTP 坐标（"Q16"）。**末位字母列 = 从左往右 A.. 跳过 I**
（围棋惯例），行是数字 1..19（从**下往上**）。必须转成 GoBoard 的扁平索引
`(row, col) = (19 - num, letter_idx)`。这一处的错位会让整份标签报废，
故下面 `_gtp_to_idx` 单独钉了测试。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sgf_parser import SGFParser  # noqa: E402
from src.game.go_rules import GoBoard  # noqa: E402

# 位置散列**复用** probe0_join 的实现，而不是在这里重写一遍：两处口径一旦
# 分叉，标签就 join 不上，而这种错不会报错、只会在训练时表现为「标签没生效」。
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "probe0_join", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "probe0_join.py"))
_p0 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_p0)

BOARD = 19
N_ACTIONS = BOARD * BOARD + 1          # 362，末位 pass
_PASS = N_ACTIONS - 1

# GTP 列字母：跳过 I（围棋惯例）
_COLS = "ABCDEFGHJKLMNOPQRST"
_COL2IDX = {c: i for i, c in enumerate(_COLS)}


def gtp_to_idx(gtp):
    """'Q16' -> 扁平索引。行号从下往上（16 行 => row=19-16=3），列 A 在最左。"""
    if gtp in ("pass", "PASS", "resign"):
        return _PASS
    col = _COL2IDX[gtp[0].upper()]
    row = BOARD - int(gtp[1:])
    if not (0 <= row < BOARD and 0 <= col < BOARD):
        raise ValueError(f"GTP 坐标越界: {gtp}")
    return row * BOARD + col


# --------------------------------------------------------------------------- #
class KataLabeler:
    """把 katago.exe analysis 当子进程驱动（stdin/stdout 管道）。"""

    def __init__(self, exe, model, config, cwd=None, timeout=600):
        self.p = subprocess.Popen(
            [exe, "analysis", "-model", model, "-config", config],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=cwd, text=True,
            encoding="utf-8", errors="ignore", bufsize=1,
        )
        self.timeout = timeout
        self.n_done = 0

    def wait_ready(self, deadline=240):
        """就绪探测。

        ⚠ **不能靠日志判断**：config 里 `logDir` 已设，KataGo 把所有启动日志
        （含 "Started, ready to begin handling requests"）写进日志文件，**stdout
        上只有 JSON 响应**。等那句日志会永久阻塞（第一版就是这么挂的）。

        正确做法：stdout 只有 JSON ⇒ 直接发一条最小查询，用它的响应当就绪信号。
        引擎没起来就查不到响应，`query` 会在超时/EOF 上抛出来。
        """
        t0 = time.time()
        # 空局、1 visit：最快的一次往返，只为确认管道通了
        probe = dict(id="__ready__", moves=[], maxVisits=1,
                     boardXSize=BOARD, boardYSize=BOARD,
                     rules="tromp-taylor", komi=7.5)
        last = None
        while time.time() - t0 < deadline:
            try:
                self.query(probe, retries=0)
                return
            except Exception as e:          # 引擎还在加载模型
                last = e
                time.sleep(2)
        raise TimeoutError(f"katago {deadline}s 未就绪: {last}")

    def _readline_timeout(self, timeout):
        """带超时的 readline。

        ⚠ 直接 `readline()` 在引擎挂掉时会**永久阻塞**（管道没有数据也不返回），
          整个脚本就静默卡住。所以用后台线程 + join(timeout) 包装。
        """
        box = {}

        def _rd():
            try:
                box["line"] = self.p.stdout.readline()
            except Exception as e:            # pragma: no cover
                box["err"] = e

        th = threading.Thread(target=_rd, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            raise TimeoutError(f"读 katago 响应超时 {timeout}s（引擎可能已崩）")
        if "err" in box:
            raise box["err"]
        return box.get("line", "")

    def query(self, req, retries=2, timeout=None):
        """发一条分析请求，返回 dict。"""
        timeout = timeout or self.timeout
        for attempt in range(retries + 1):
            self.p.stdin.write(json.dumps(req) + "\n")
            self.p.stdin.flush()
            while True:
                line = self._readline_timeout(timeout)
                if line == "":
                    raise RuntimeError("katago 意外退出（stdout EOF）")
                line = line.strip()
                if not line.startswith("{"):
                    continue                      # 日志行
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # 半行/非 JSON
                if "id" in obj and obj.get("id") == req["id"]:
                    if "error" in obj:
                        raise RuntimeError(f"分析错误: {obj['error']}")
                    self.n_done += 1
                    return obj
        raise RuntimeError("重试耗尽")

    def query_many(self, reqs, timeout=None):
        """**流水线**发一批请求、收一批响应。

        ⚠ 为什么必须有这个：config 里 `numAnalysisThreads=16` 让引擎能同时搜 16 个
          局面，但同步「发一条等一条」同一时刻只有 **1** 个请求在飞，引擎 15/16 的
          能力闲置（实测 0.42 局面/s vs 理论 6.7）。所以这里先一口气把整批都写进
          stdin，再逐条读响应 —— 响应带 id，按 id 归档，与到达顺序无关。

        返回 `(resps, errors)`，errors 是 `{rid: 错误串}`。

        ⚠⚠ **不再因单个请求报错而抛异常**（P0）：真实语料里一定有一部分 SGF 是
          坏记录 —— 让子/布局子被当普通着法塞进 moves、坐标错位、记录本身不合法。
          原实现 `raise`，等于「第一个坏局就把 13 万局的一整批打死」，实测就卡死在
          `Illegal move 72: F2`。改成「收集 + 交调用方隔离」后，坏局只损失它自己。
        """
        timeout = timeout or self.timeout
        want = {r["id"] for r in reqs}
        for r in reqs:
            self.p.stdin.write(json.dumps(r) + "\n")
        self.p.stdin.flush()
        out, errors = {}, {}
        # 终止条件用 len(out) + len(errors)：报错的请求不会进 out，只用它判完成会死等。
        while len(out) + len(errors) < len(want):
            line = self._readline_timeout(timeout)
            if line == "":
                raise RuntimeError("katago 意外退出（stdout EOF）")
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get("id")
            if rid not in want:
                continue
            if "error" in obj:
                errors[rid] = str(obj["error"])
            else:
                out[rid] = obj
                self.n_done += 1
        return out, errors

    def close(self):
        # 合法 action 只有 query_version / query_models / clear_cache /
        # terminate / terminate_all（引擎会明确报 "must be ..."）。
        # 第一版写的 "quit" 会让收尾报参数错误（打标签本身已完成，标签不丢）。
        try:
            self.p.stdin.write(
                json.dumps({"id": "__bye__", "action": "terminate"}) + "\n")
            self.p.stdin.flush()
        except Exception:
            pass
        try:
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


# --------------------------------------------------------------------------- #
# 坏局隔离（P0）
# --------------------------------------------------------------------------- #
# `AB`/`AW`（让子/布局子）在 sgf_parser 里被**并入 moves**（见其
# `_extract_moves` 的 docstring）。这个决定对 GTP 成立，对 KataGo analysis
# 不成立：analysis 协议要用 `initialStones` + `initialPlayer` 表达起始局面，
# 而把布局子当普通着法塞进 moves 会让坐标、颜色、打劫状态整体错位。
# 后果不止是非法记录本身 —— 布局子被当同色连续手落下，KataGo 的
# `GameState::playMove` **不强制颜色交替**（那是 GTP 层的职责），所以错位
# 不会在第 0 手就炸，而是要走到几十手后才由点合法性/打劫分歧暴露
# （实测 `Illegal move 72: F2`）。
#
# 所以这里不做「猜根因」，只做**隔离**：坏局损失它自己那一局，不拖垮整批。
# 隔离清单落盘，跳过率会打印出来 —— 那是决定「要不要动 sgf_parser 协议」的
# 唯一依据（跳过率 <0.5% 可以先不管，>2% 就必须正规化 initialStones 并用
# verify.npz 回归 join 命中率）。

_BAD_MOVE_RE = re.compile(r"Illegal move\s+(\d+)", re.I)


def first_bad_move(err):
    """从引擎错误串里抠出「第一个非法着的下标」；抠不出返回 -1。"""
    m = _BAD_MOVE_RE.search(err or "")
    return int(m.group(1)) if m else -1


def classify_error(err):
    """把错误归成粗类，便于汇总时一眼看出是哪一类坏记录在拖累跳过率。

    ⚠ `komi` 的判断**必须排在 `ko` 前面**：`"komi" in s` 会被裸的 `"ko" in s`
      命中（子串），于是所有棋盘大小/贴目错误都会被误报成打劫分歧，把真正的
      illegal-move 占比掩盖掉。
    """
    s = (err or "").lower()
    if "illegal move" in s:
        return "illegal-move"
    if "komi" in s:
        return "komi"
    if "superko" in s or "ko violation" in s or "repeating ko" in s:
        return "ko"
    if "board" in s and "size" in s:
        return "board-size"
    if "json" in s:
        return "protocol"
    return "other"


def _skip_path_for(out_path):
    return os.path.join(os.path.dirname(out_path) or ".", "skipped_games.npz")


def load_skipped(out_path):
    """读上次的隔离清单（续跑时与本轮合并）。坏文件当空，不因此中断。"""
    p = _skip_path_for(out_path)
    if not os.path.isfile(p):
        return {}
    try:
        z = np.load(p, allow_pickle=True)
        g = z["game_idx"]
        return {int(gi): (str(pa), int(fb), str(er))
                for gi, pa, fb, er in zip(g, z["path"], z["first_bad_move"],
                                          z["error"])}
    except Exception:
        return {}


def save_skipped(out_path, skipped):
    """落盘隔离清单：path / game_idx / first_bad_move / error。"""
    p = _skip_path_for(out_path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    if not skipped:
        # 空清单也要写：让「跑完确实一局没坏」与「清单还没生成」可区分。
        np.savez(p, path=np.array([], dtype=object),
                 game_idx=np.zeros(0, np.int32),
                 first_bad_move=np.zeros(0, np.int32),
                 error=np.array([], dtype=object))
        return p
    ks = sorted(skipped)
    np.savez(p,
             path=np.array([skipped[k][0] for k in ks], dtype=object),
             game_idx=np.array(ks, dtype=np.int32),
             first_bad_move=np.array([skipped[k][1] for k in ks], np.int32),
             error=np.array([skipped[k][2] for k in ks], dtype=object))
    return p


def dump_bad_game(path, err, chars=900):
    """把坏局 SGF 的开头打出来 —— 用来一眼看出有没有 AB/AW/HA 布局子。"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(chars)
    except OSError as e:
        return f"(读不出 {path}: {e})"
    tail = "…" if len(head) >= chars else ""
    return head + tail


# --------------------------------------------------------------------------- #
# 续跑指纹（Phase 0a）
# --------------------------------------------------------------------------- #
# ⚠⚠ **为什么必须有**：`.done.n` 只是一个裸游标，而 `positions` 的内容由一堆参数
#   决定。原来只有一条守卫 `cursor > len(positions)` —— **单向**。于是：
#     · 同样参数重跑        -> 无事可做，正确
#     · 参数变小（局面变少） -> 拦住了
#     · **参数变大**（比如把 game-frac 0.02 改成全量，positions 从 219,343 变成
#       13,360,400）-> `cursor(219,343) > len(13,360,400)` 为假，**不拦** ⇒
#       脚本认为「前 219,343 个局面已完成」而直接跳过 ⇒ **那批标签永久缺失，
#       且不报任何错**，只会看到最终行数少了 219,343。
#   这与 `build_soft_index.py` 踩过的「陈旧缓存静默挂错标签」是同一类故障。
#
# 指纹覆盖**一切能改变 positions 的输入**：抽局参数、窗口参数、语料目录的身份
# （路径 + 文件数 + 总大小 + mtime）。语料变了同样必须重建游标 —— 否则「第 i 行」
# 指向的局面已经完全不是当初那个。

_FP_VERSION = 1


def positions_fingerprint(args, sgf_dir, n_sgf, n_positions):
    """续跑指纹：任何影响 `positions` 的输入都进这里。"""
    try:
        st = os.stat(sgf_dir)
        corpus = {"path": os.path.abspath(sgf_dir), "n_sgf": int(n_sgf),
                  "mtime_ns": int(st.st_mtime_ns)}
    except OSError:
        corpus = {"path": os.path.abspath(sgf_dir), "n_sgf": int(n_sgf),
                  "mtime_ns": None}
    return {
        "version": _FP_VERSION,
        "move_lo": int(args.move_lo), "move_hi": int(args.move_hi),
        "max_games": int(args.max_games), "game_frac": float(args.game_frac),
        "seed": int(args.seed), "limit": int(args.limit),
        "n_positions": int(n_positions),
        "corpus": corpus,
    }


def _fp_path(out_path):
    return os.path.splitext(out_path)[0] + ".done.json"


def check_resume_fingerprint(out_path, fp):
    """比对续跑指纹。

    Returns
    -------
    `(cursor, matched)`：
      `matched=True`  -> `cursor` 是可信游标（从 `.done.n` 读，或 0）
      `matched=False` -> 指纹对不上但用户显式要求重来，`cursor=0` 从头开始

    指纹不一致且**没有**显式重来时直接报错退出 —— 静默续跑会丢掉那批标签。
    """
    path = _fp_path(out_path)
    fresh = os.environ.get("SOFT_TAG_FRESH", "") == "1"
    if not os.path.isfile(path):
        if fresh:
            return 0, False
        # 首次跑（既无指纹也无 .done.n）→ 正常起点
        return 0, True
    try:
        old = json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:                       # 坏文件当「无法判断」而不是「没跑过」
        raise SystemExit(
            f"续跑指纹 {path} 读取失败（{type(e).__name__}: {e}）。"
            f"它损坏时无法判断 positions 是否还是同一份 —— 请删掉该文件"
            f"（或设 SOFT_TAG_FRESH=1 从头重跑）。") from e
    if old == fp:
        return 0, True                           # 指纹一致；真正的游标由调用方读
    if fresh:
        return 0, False
    diff = {k: (old.get(k), fp.get(k)) for k in set(old) | set(fp)
            if old.get(k) != fp.get(k)}
    detail = "\n".join(
        f"    {k}: 之前={v[0]!r} 现在={v[1]!r}" for k, v in sorted(diff.items()))
    raise SystemExit(
        f"续跑指纹不一致 —— 上次的游标**不能**套到这次的 positions 上：\n"
        f"  {detail}\n"
        f"  继续跑会静默丢掉已经处理过的那些局面的标签（不报任何错）。\n"
        f"  两种处理：\n"
        f"    · 从头重跑：删掉 {os.path.splitext(out_path)[0]}.* 后重跑，"
        f"或设 SOFT_TAG_FRESH=1\n"
        f"    · 换输出路径：给 --out 一个新的名字，让两批标签互不干扰")


def save_resume_fingerprint(out_path, fp):
    p = _fp_path(out_path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fp, f, ensure_ascii=False, sort_keys=True, indent=1)
    os.replace(tmp, p)      # 原子：崩溃时不会留下半份指纹
    return p


# --------------------------------------------------------------------------- #
def pos_hash_one(board, to_play, ko):
    """单个局面的位置 hash（与 probe0_join.pos_hash_block 同口径）。"""
    b = np.asarray(board, dtype=np.int8).reshape(1, BOARD, BOARD)
    return _p0.pos_hash_block(b, np.array([to_play], np.int8),
                              np.array([ko], np.int16))[0]


def replay_to(moves):
    """把着法序列重放成 (board, to_play, ko)。

    `moves` 是 build_queries 产出的 `[color, gtp]` 列表，重放到**最后一手之后**的
    局面 —— 即 KataGo 分析的那个局面（分析请求带的就是完整 seq）。
    """
    b = GoBoard(BOARD, komi=7.5)
    for color, gtp in moves:
        idx = gtp_to_idx(gtp)
        mv = -1 if idx == _PASS else idx
        if not b.play(mv):
            b.play(-1)
    return b.board, int(b.current_player), int(b.ko_point)


def visit_distribution(obj, topk_keep=None):
    """从 analysis 响应抽访问分布 -> (362,) float32。

    只取 `moveInfos` 里出现过的点：引擎默认不返回 visits=0 的点，所以未探索的
    361-topk 位置都是 0。这本身就是 KataGo 的软标签形态（top-k 支撑）。
    `topk_keep` 用来裁掉长尾（默认全留，maxVisits 128 时通常只有 10~40 个点）。
    """
    v = np.zeros(N_ACTIONS, dtype=np.float64)
    infos = obj.get("moveInfos", [])
    if topk_keep:
        infos = sorted(infos, key=lambda m: -m.get("visits", 0))[:topk_keep]
    for m in infos:
        try:
            v[gtp_to_idx(m["move"])] += float(m.get("visits", 0))
        except (KeyError, ValueError):
            continue
    s = v.sum()
    return v / s if s > 0 else v


def build_queries(paths, move_lo, move_hi, max_games, seed, game_frac=0.0):
    """从 SGF 产出待打标签的局面列表。

    走 SGF 而不是 npz，因为 KataGo analysis 的请求体是**着法序列**
    （`moves: [["B","Q16"],...]`），不是盘面 —— 必须有原始着法才能表达局面。
    """
    import random
    rng = random.Random(seed)
    if game_frac and game_frac > 0:
        k = max(1, int(round(len(paths) * game_frac)))
        sel = rng.sample(paths, k) if k < len(paths) else list(paths)
    elif max_games <= 0 or len(paths) <= max_games:
        sel = paths
    else:
        sel = rng.sample(paths, max_games)
    out = []
    # ⚠ 解析器在循环**外**构造：原实现每个文件 `SGFParser()` 一次，13 万次
    #   多余构造。复用同一个实例无副作用（`_extract_moves` 等都是纯函数式用法）。
    parser = SGFParser()
    for gi, p in enumerate(sel):
        try:
            g = parser.parse_file(p)
        except Exception:
            continue
        if g is None or g.board_size != BOARD or len(g.moves) < move_lo:
            continue
        seq = []
        for i, mv in enumerate(g.moves):
            r, c = mv.position
            gtp = "pass" if (r, c) == (-1, -1) else f"{_COLS[c]}{BOARD - r}"
            seq.append([mv.color, gtp])
            if move_lo <= i < move_hi:
                out.append(dict(game=gi, move=i, moves=list(seq),
                                path=p))
        # 每个局面独立从零开始，故 seq 需拷贝（上面已 list()）
    return out, len(sel)


# --------------------------------------------------------------------------- #
def find_katago_exe():
    """自动找 katago 可执行文件（跨 Windows/Linux）。

    ⚠ **同名不同平台要挑对的**：soft_tag 包里同时有 `katago/katago`（Linux ELF）
      和（若在 Windows 上测试时）`katago/katago.exe`。按 glob 顺序取第一个会在
      Windows 上选中 ELF，报 `WinError 193 %1 不是有效的 Win32 应用程序`。
      故按平台显式偏好带后缀的那个。
    """
    import glob
    import shutil
    import sys as _sys
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "katago")
    win = _sys.platform.startswith("win")
    # 覆盖两种摆放：`katago/katago[.exe]`（本包）与 `katago/<子目录>/katago[.exe]`
    # （仓库里解压的版本）。Windows 优先 .exe —— 同目录可能并存 Linux ELF 与
    # Windows .exe（冒烟测试时就会），取错会报 WinError 193。
    if win:
        pats = ("katago.exe", "katago*/katago.exe", "*/katago.exe",
                "katago", "katago*/katago", "*/katago")
    else:
        pats = ("katago", "katago*/katago", "*/katago",
                "katago.exe", "katago*/katago.exe", "*/katago.exe")
    if os.path.isdir(base):
        for pat in pats:
            for h in sorted(glob.glob(os.path.join(base, pat))):
                if os.path.isfile(h) and (h.endswith(".exe")
                                         or os.access(h, os.X_OK)):
                    return h
    return shutil.which("katago")


# --------------------------------------------------------------------------- #
# 引擎分发形态与依赖体检
# --------------------------------------------------------------------------- #
# KataGo 官方 Linux zip 里的 `katago` **是 AppImage**（type 2：ELF 头之后
# offset 8 放 b"AI\x02"）。AppImage 靠 FUSE 挂载自己才能跑，而容器里通常没有
# /dev/fuse —— 症状是引擎秒退，但报错要绕到 wait_ready() 才浮出来，变成
# 「240s 未就绪」，极难定位（2026-10-02 实测踩过）。
# `--appimage-extract` 可解包出可执行树，解包后不再需要 FUSE。
_APPIMAGE_MAGIC = b"AI\x02"


def is_appimage(path):
    """`path` 是否是 AppImage（仅识别 type 2 的 ELF 头布局）。

    ⚠ 打包期只校验「是不是 ELF」是不够的：AppImage 本身就是 ELF，只有 offset 8
      的 `AI\\x02` 能把它和真正的裸二进制区分开。漏了这一条就会发出一个在目标
      环境完全跑不起来的包。
    """
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return head[:4] == b"\x7fELF" and head[8:11] == _APPIMAGE_MAGIC


def resolve_katago(exe):
    """把候选引擎解析成真正可跑的路径。

    返回 `(可执行文件, 需追加到 LD_LIBRARY_PATH 的目录元组)`。
      · 原生二进制      -> 原样返回
      · AppImage 已解包 -> `squashfs-root/AppRun`（它会设好 LD_LIBRARY_PATH）
      · AppImage 未解包 -> 抛带指引的错，**不自动解包**

    ⚠ 不自动解包是刻意的：解包会往包目录落 ~100MB 的 `squashfs-root/`，属于用户
      可见的磁盘副作用，不该由脚本默认触发。手工解一次之后本函数直接复用，
      往后无需再介入。
    """
    if not exe or not is_appimage(exe):
        return exe, ()
    base = os.path.dirname(exe)
    root = os.path.join(base, "squashfs-root")
    apprun = os.path.join(root, "AppRun")
    if not os.path.isfile(apprun):
        raise SystemExit(
            f"{exe} 是 AppImage，但 {apprun} 不存在。\n"
            f"  KataGo 官方 Linux zip 里的引擎就是 AppImage，运行时要靠 FUSE\n"
            f"  挂载自己，而本环境没有 /dev/fuse（容器常见）。解包后不再需要：\n\n"
            f"      cd {base}\n"
            f"      ./{os.path.basename(exe)} --appimage-extract\n\n"
            f"  之后重跑本命令即可。")
    # ⚠ 捆绑 so 的目录必须一并交给 ldd：AppRun 会设 LD_LIBRARY_PATH，裸 ldd
    #   看不到，于是把**自带的** libzip 报成「not found」（实测假警报）。
    #   用分段 join 而不是 "usr/lib" 这种字面斜杠 —— 后者在 Windows 上碰巧能过，
    #   但断言与跨平台行为会对不齐。
    _LIB_SUBDIRS = (("usr", "lib"), ("lib",),
                    ("usr", "lib", "x86_64-linux-gnu"), ("usr", "lib64"))
    libs = tuple(d for d in (os.path.join(root, *s) for s in _LIB_SUBDIRS)
                 if os.path.isdir(d))
    return apprun, libs


#: 缺库 -> 可执行的处置建议。把踩过的坑写死成表，别让用户自己猜。
_DEP_HINT = {
    "libcudnn.so.9": "引擎是 cuda*-cudnn9.8.0 构建：装 cuDNN >= 9.8，或换 "
                     "cudnn8.9.7 构建。注意 sm_80 以下两者都走不到 cudnn "
                     "SDPA，差距比文档说的小",
    "libcudnn.so.8": "引擎是 cudnn8.9.7 构建：宿主需 cuDNN 8.x",
    "libcudnn_ops_infer.so.8": "引擎是 cudnn8.9.7 构建：宿主需 cuDNN 8.x",
    "libcudnn_cnn_infer.so.8": "引擎是 cudnn8.9.7 构建：宿主需 cuDNN 8.x",
    "libcuda.so.1": "容器没挂 GPU 驱动：用 --gpus all / nvidia-container-runtime",
    "libzip.so.4": "AppImage 本应捆绑 libzip；仍缺说明 --appimage-extract 没解完",
    "libnvinfer.so.10": "引擎是 TensorRT 构建：宿主需 TensorRT >= 10",
}


def missing_shared_libs(exe, extra_libdirs=()):
    """列出本机解析不到的 so。非 Linux（或无 ldd）返回 None —— 不假装全过。"""
    if not sys.platform.startswith("linux"):
        return None
    env = dict(os.environ)
    if extra_libdirs:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            list(extra_libdirs) + ([prev] if prev else []))
    try:
        r = subprocess.run(["ldd", exe], capture_output=True, text=True, env=env)
    except OSError:
        return None                    # 没有 ldd
    return sorted({ln.split("=>")[0].strip()
                   for ln in r.stdout.splitlines() if "not found" in ln})


def find_config():
    """找 analysis 配置。优先 analysis_batch.cfg（我们调过并发的那份）。"""
    import glob
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "katago")
    for name in ("analysis_batch.cfg",):
        hits = sorted(glob.glob(os.path.join(base, "**", name), recursive=True))
        if hits:
            return hits[0]
    return None


def find_model():
    import glob
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "katago")
    hits = sorted(glob.glob(os.path.join(base, "*.bin.gz")))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description="KataGo 批量打搜索标签")
    ap.add_argument("--sgf-dir", default=None,
                    help="SGF 目录；不给则优先用 <repo>/data/sgf，"
                         "否则 <repo>/data/games/games")
    ap.add_argument("--exe", default=None,
                    help="katago 可执行文件；不给则自动在 katago/ 下找")
    ap.add_argument("--model", default=None,
                    help="KataGo 权重 .bin.gz；不给则自动取 katago/*.bin.gz")
    ap.add_argument("--config", default=None,
                    help="analysis 配置；不给则自动找 analysis_batch.cfg")
    ap.add_argument("--move-lo", type=int, default=100)
    ap.add_argument("--move-hi", type=int, default=200)
    ap.add_argument("--max-games", type=int, default=0, help="0=全部")
    ap.add_argument("--game-frac", type=float, default=0.0,
                    help="按比例随机抽局（0=不抽）。与 --max-games 互斥，"
                         "1.5% ≈ 0.015。抽的是**局**，再在局内取 "
                         "--move-lo~--move-hi 区间")
    ap.add_argument("--max-visits", type=int, default=36,
                    help="每局面的 search 预算。⚠ KataGo 实际达成 = 此值 + 1"
                         "（根节点首次展开也计一次 visit），属引擎口径而非 bug。"
                         "实测吞吐：32→1.39 / 48→0.93 局面/s（visits 越低越快，"
                         "但搜索深度浅、软标签的'搜索知识'变弱）")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="同时在飞的请求数，**须与 config 的 numAnalysisThreads 一致**"
                         "（当前 analysis_batch.cfg = 4）。实测 4/8/16 并发同速，"
                         "瓶颈在 GPU 前向本身，不在并发度")
    ap.add_argument("--limit", type=int, default=500, help="本次最多打多少局面")
    ap.add_argument("--dump-bad-game", action="store_true",
                    help="隔离坏局时把该 SGF 开头 900 字打出来（P0）。"
                         "用来一眼看出有没有 AB/AW/HA 布局子被当成普通着法，"
                         "不必再猜引擎报的是哪一类问题")
    ap.add_argument("--preflight", action="store_true",
                    help="只做引擎/依赖体检并打印运行参数，不打标签。"
                         "run.sh 用它在真正开跑前秒级失败")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("data", "labels", "kata_labels.npz"))
    args = ap.parse_args()

    # SGF 目录：soft_tag 包里是 data/sgf（扁平 + 源名前缀），
    # 仓库里是 data/games/games（分源子目录）。两者都支持（glob 带 **）。
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.sgf_dir is None:
        for cand in (os.path.join(repo, "data", "sgf"),
                     os.path.join(repo, "data", "games", "games")):
            if os.path.isdir(cand):
                args.sgf_dir = cand
                break
        else:
            raise SystemExit("找不到 SGF 目录（试了 data/sgf 与 data/games/games），"
                             "请用 --sgf-dir 指定")
    paths = sorted(glob.glob(os.path.join(args.sgf_dir, "**", "*.sgf"), recursive=True))
    if not paths:
        raise SystemExit(f"没找到 SGF: {args.sgf_dir}")

    # 三个路径全部自动发现（跨平台），显式传入则优先
    exe = args.exe or find_katago_exe()
    model = args.model or find_model()
    config = args.config or find_config()
    for name, v in (("--exe", exe), ("--model", model), ("--config", config)):
        if not v or not os.path.exists(v):
            raise SystemExit(f"{name} 找不到: {v!r}\n"
                             f"  请放到 katago/ 下，或用命令行显式指定。")
    # AppImage 要先解包成 AppRun，否则 Popen 会以 FUSE 失败告终，而症状绕到
    # wait_ready() 变成 240s 超时。放在打印之前 —— 预检要报的是**真正会跑的那个**。
    exe, libdirs = resolve_katago(exe)
    print(f"SGF 池 = {len(paths):,} 局")
    print(f"引擎   = {exe}")
    print(f"权重   = {os.path.basename(model)}")
    print(f"配置   = {os.path.basename(config)}")

    # 动态库体检：缺依赖时引擎在 loader 阶段就死，症状同样绕到 wait_ready() 的
    # 240s 超时。这里秒级失败并指明缺哪个、怎么补。
    missing = missing_shared_libs(exe, libdirs)
    if missing:
        print("\n引擎依赖缺失：", file=sys.stderr)
        for m in missing:
            print(f"  {m}\n    → {_DEP_HINT.get(m, '需在宿主补装该库')}",
                  file=sys.stderr)
        raise SystemExit(1)
    if missing is None:
        print("依赖检查  = 跳过（非 Linux 或无 ldd）")

    if args.preflight:
        print(f"手数窗   = {args.move_lo}~{args.move_hi}")
        print(f"visits   = {args.max_visits}（引擎实际达成 {args.max_visits + 1}）")
        print(f"并发     = {args.concurrency}（须=config 的 numAnalysisThreads）")
        print(f"limit    = {args.limit}（0=全打）")
        print(f"输出     = {args.out}")
        print("\npreflight 通过。")
        return

    positions, n_sel = build_queries(paths, args.move_lo, args.move_hi,
                                     args.max_games, args.seed, args.game_frac)
    print(f"选中 {n_sel} 局 -> {args.move_lo}~{args.move_hi} 手共 "
          f"{len(positions):,} 个候选局面")
    # limit<=0 表示「全打」——不能写成 positions[:0]，那会静默产出空批次
    # （用户以为在跑 1.5%，实际一个标签都没有）。
    n_all = len(positions)
    if args.limit and args.limit > 0:
        positions = positions[: args.limit]
    else:
        print(f"limit<=0 -> 全打 {n_all:,} 个候选局面")
    print(f"本次打标签 {len(positions):,} 个局面，"
          f"maxVisits={args.max_visits}（引擎实际达成 {args.max_visits+1}），"
          f"并发={args.concurrency}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # **边打边落盘**（.npy 追加写），不是攒在内存里最后一次性 save。
    # 长跑（几小时~几天）中途若崩/被kill，已打的标签还在，可续跑；
    # 攒内存的写法一崩就全丢，这也是第一版跑完 8 分钟输出为空的原因之一。
    part = os.path.splitext(args.out)[0]
    # repo 已在上面（SGF 目录探测处）算好：引擎**必须在项目根启动**
    #（KataGo 的 KataGoData/ 按 cwd 解析）
    #
    # ⚠⚠ `cursor` 是「下一个待处理局面的下标」，**不是已写行数**。P0 之前这两者
    #   被混为一谈（`todo = range(done, len(positions))`），一旦跳过局面就错位：
    #   续跑会拿 `positions[i]` 去配第 i 行，而 i 已经不是它当初对应的局面 ——
    #   结果是**静默产出错位的标签**（pos_hash 与 policy 不再属于同一局面，
    #   join 命中率会诡异地下降而不是报错）。行数改为从落盘文件长度反推。
    done_marker = part + ".done.n"
    # 指纹必须**先于**读游标比对：裸游标无法证明「第 i 行」还是当初那个局面
    # （见 positions_fingerprint 的 docstring —— 参数变大时旧守卫是单向的）。
    fp = positions_fingerprint(args, args.sgf_dir, len(paths), len(positions))
    _c, fp_ok = check_resume_fingerprint(args.out, fp)
    cursor = (int(open(done_marker).read().strip())
              if (fp_ok and os.path.exists(done_marker)) else 0)
    # 隔离清单：续跑时与上轮合并，避免同一坏局被反复试、反复打日志
    skipped = load_skipped(args.out) if fp_ok else {}
    if cursor:
        print(f"检测到已完成 {cursor} 个局面，从该处续跑"
              + (f"（上轮已隔离 {len(skipped)} 局）" if skipped else ""))
    if cursor > len(positions):
        raise SystemExit(
            f"续跑游标 {cursor} > 本次候选局面数 {len(positions)} —— "
            f"参数变了（--move-lo/--move-hi/--max-games/--game-frac/--limit "
            f"任一）。继续跑会把上一轮的游标套到新的 positions 上，"
            f"产出错位标签。请删掉 {done_marker} 与 {part}.*.part.npy 后重来。")
    if cursor == 0:
        # 从头跑：把上一轮残留的分片挪走，否则新行会追加在旧数据后面 -> 行数对不上
        stale = [f"{part}.{k}.part.npy" for k in
                 ("policy", "pos_hash", "root_win", "score_mean", "score_stdev",
                  "visits", "game_idx", "move_idx")]
        hit = [p for p in stale if os.path.exists(p)]
        if hit:
            raise SystemExit(
                "检测到无续跑指纹但存在上一轮的分片文件：\n  "
                + "\n  ".join(hit)
                + "\n这些是**上一批**标签的缓冲，追加写入会让行数与内容错位。"
                  "请删掉它们（或换 --out）后重跑。")
    save_resume_fingerprint(args.out, fp)

    def _open(tag):
        return {k: open(f"{part}.{k}.{tag}.npy", "ab")
                for k in ("policy", "pos_hash", "root_win", "score_mean",
                          "score_stdev", "visits", "game_idx", "move_idx")}

    handles = _open("part")
    lab = KataLabeler(os.path.abspath(exe), os.path.abspath(model),
                      os.path.abspath(config), cwd=repo)
    print("等待引擎就绪 ...")
    t_ready = time.perf_counter()
    lab.wait_ready()
    print(f"就绪耗时 {time.perf_counter()-t_ready:.1f}s，开始打标签\n")

    n = 0
    t0 = time.perf_counter()
    try:
        # 流水线：一次发 CONC 条，收齐再发下一批，让 config 的 numAnalysisThreads
        # 个并发槽真正被占满。串行发的话同一刻只有 1 个请求在飞，引擎闲置 5/6。
        conc = args.concurrency
        start_cursor = cursor
        todo = list(range(cursor, len(positions)))
        for cs in range(0, len(todo), conc):
            raw = todo[cs:cs + conc]
            # 已知坏局**不再发请求**：窗口内同局有 ~100 个局面，全发一遍既白烧
            # 引擎又会把日志刷爆。只跳发，但游标照常推进（否则会死循环重扫）。
            batch = [i for i in raw if positions[i]["game"] not in skipped]
            if not batch:
                cursor = raw[-1] + 1
                open(done_marker, "w").write(str(cursor))
                continue

            reqs = []
            for i in batch:
                q = positions[i]
                reqs.append(dict(id=f"q{i}", moves=q["moves"],
                                 maxVisits=args.max_visits,
                                 boardXSize=BOARD, boardYSize=BOARD,
                                 rules="tromp-taylor", komi=7.5))
            resps, errors = lab.query_many(reqs)

            # ---- P0：整局隔离 ----
            # 引擎报「某手非法」时，错的是**这一局**的着法序列（布局子被当普通
            # 手 / 坐标错位 / 记录本身不合法）。同一局在窗口内的其它局面用的是
            # 同一条序列的前缀，必然同样非法 —— 只跳这一个局面是白费，所以按局
            # 拉黑，并把它记进清单。
            for rid, err in errors.items():
                i = int(rid[1:])
                q = positions[i]
                gidx = q["game"]
                if gidx in skipped:
                    continue
                fb = first_bad_move(err)
                skipped[gidx] = (q["path"], fb, err)
                print(f"\n[隔离] {os.path.basename(q['path'])} "
                      f"首非法手={fb}  {err}", flush=True)
                if args.dump_bad_game:
                    print("  ---- SGF 开头 ----")
                    print(dump_bad_game(q["path"], err))
                    print("  ------------------", flush=True)
                if len(skipped) % 20 == 0:
                    sp = save_skipped(args.out, skipped)
                    print(f"[隔离] 累计 {len(skipped)} 局 -> {sp}", flush=True)

            for i in batch:
                obj = resps.get(f"q{i}")
                if obj is None:
                    continue          # 已隔离（引擎报错），不写行
                q = positions[i]
                p = visit_distribution(obj).astype(np.float16)
                b, tp, ko = replay_to(q["moves"])
                ri = obj.get("rootInfo", {})
                handles["policy"].write(p.tobytes())
                handles["pos_hash"].write(
                    np.array([_p0.pos_hash_block(np.asarray(b, np.int8)[None],
                                                  np.array([tp], np.int8),
                                                  np.array([ko], np.int16))[0]],
                             np.uint64).tobytes())
                for k, v in (("root_win", ri.get("winrate", 0.5)),
                             ("score_mean", ri.get("scoreMean", 0.0)),
                             ("score_stdev", ri.get("scoreStdev", 0.0))):
                    handles[k].write(np.array([v], np.float32).tobytes())
                handles["visits"].write(
                    np.array([ri.get("visits", 0)], np.int16).tobytes())
                handles["game_idx"].write(np.array([q["game"]], np.int32).tobytes())
                handles["move_idx"].write(np.array([q["move"]], np.int32).tobytes())
                n += 1
            cursor = raw[-1] + 1        # 游标按「已处理」推进，与写了多少行无关
            for f in handles.values():
                f.flush()
            open(done_marker, "w").write(str(cursor))
            el = time.perf_counter() - t0
            # 速率/ETA 按**已处理局面**算（不是已写行数），否则隔离局会污染 ETA。
            done_pos = cursor - start_cursor
            rate = done_pos / max(el, 1e-9)
            eta = (len(positions) - cursor) / max(rate, 1e-9) / 3600
            print(f"  {cursor}/{len(positions)}  {el/60:.1f}min  "
                  f"{rate:.2f} 局面/s  ETA {eta:.2f}h  "
                  f"[已写 {n:,} 行, 隔离 {len(skipped)} 局]", flush=True)
    finally:
        lab.close()
        for f in handles.values():
            f.close()

    # 拼成最终 npz。
    # ⚠ 行数从**落盘文件长度**反推，不用 `cursor` 也不用 `n`：隔离局不写行，
    #   续跑时 `n` 只统计本次，而文件里还有上次的行。pos_hash 是定长 uint64，
    #   拿它数行最可靠。
    out = {k: np.fromfile(f"{part}.{k}.part.npy", dtype=dt).reshape(-1, *sh)
           for k, dt, sh in (("policy", np.float16, (N_ACTIONS,)),
                             ("pos_hash", np.uint64, ()),
                             ("root_win", np.float32, ()),
                             ("score_mean", np.float32, ()),
                             ("score_stdev", np.float32, ()),
                             ("visits", np.int16, ()),
                             ("game_idx", np.int32, ()),
                             ("move_idx", np.int32, ()))}
    m = int(out["pos_hash"].size)
    # 定长记录写在同一批 flush 下，各列**行数**必然一致；不一致说明文件被外部动过。
    # ⚠ 必须比 shape[0]（行数）而不是 size（元素数）—— policy 是 (N, 362)，
    #   14 行就有 5068 个元素，拿 size 比会把正常数据误判成损坏。
    bad_cols = {k: v.shape[0] for k, v in out.items() if v.shape[0] != m}
    if bad_cols:
        raise SystemExit(f"落盘各列行数不一致（pos_hash={m} 行）：{bad_cols} —— "
                         f"{part}.*.part.npy 已被外部改动，请删掉后重跑。")
    np.savez(args.out, **out)
    el = time.perf_counter() - t0
    print(f"\n写入 {args.out}  行数={m:,}  用时 {el/60:.1f}min  "
          f"({n/max(el,1e-9):.2f} 局面/s)")
    if m:
        ps = out["policy"][:200].astype(np.float32).sum(1)
        print(f"  policy 每行和: min={ps.min():.4f} max={ps.max():.4f}（应≈1）")
        print(f"  visits: min={out['visits'][:200].min()} "
              f"max={out['visits'][:200].max()} 均值={out['visits'][:200].mean():.1f}")

    # ---- P0 收尾：隔离率汇报 ----
    # 跳过率是决定「要不要动 sgf_parser 的 initialStones 协议」的唯一依据：
    # <0.5% 可以先不管；>2% 说明让子/布局子问题普遍，必须正规化并用
    # verify.npz 回归 join 命中率。所以这个数必须打在 stdout 上，不许静默。
    n_games_total = len({p["game"] for p in positions}) or 1
    sp = save_skipped(args.out, skipped)
    print(f"\n隔离局 = {len(skipped)} / {n_games_total:,} 局 "
          f"({len(skipped) / n_games_total * 100:.2f}%)  -> {sp}")
    if skipped:
        by_kind = {}
        for _g, (_p, _fb, err) in skipped.items():
            k = classify_error(err)
            by_kind[k] = by_kind.get(k, 0) + 1
        for k, c in sorted(by_kind.items(), key=lambda x: -x[1]):
            print(f"  {k:12s} {c:6,d} 局")
        rate = len(skipped) / n_games_total
        if rate > 0.02:
            print("  ⚠ 跳过率 >2%：多半是 sgf_parser 把 AB/AW 布局子并入了 moves。"
                  "应改用 KataGo 的 initialStones + initialPlayer 正规化，"
                  "并用 tmp/bench/verify.npz 回归 join 命中率。")
    else:
        print("  本轮无一局被隔离。")


if __name__ == "__main__":
    main()
