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
        """
        timeout = timeout or self.timeout
        want = {r["id"] for r in reqs}
        for r in reqs:
            self.p.stdin.write(json.dumps(r) + "\n")
        self.p.stdin.flush()
        out = {}
        while len(out) < len(want):
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
            if rid in want:
                if "error" in obj:
                    raise RuntimeError(f"分析错误({rid}): {obj['error']}")
                out[rid] = obj
                self.n_done += 1
        return out

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
    for gi, p in enumerate(sel):
        try:
            g = SGFParser().parse_file(p)
        except Exception:
            continue
        if g is None or g.board_size != BOARD or len(g.moves) < move_lo:
            continue
        seq = []
        ok = True
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

    约定：`katago/<平台目录>/katago[.exe]`，或 PATH 里的 `katago`。
    找不到就返回 None，由调用方报明确错误而不是 FileNotFoundError。
    """
    import glob
    import shutil
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "katago")
    if os.path.isdir(base):
        for pat in ("katago*/katago.exe", "katago*/katago",
                    "*/katago.exe", "*/katago", "katago", "katago.exe"):
            hits = sorted(glob.glob(os.path.join(base, pat)))
            # 必须可执行（Linux 上 +x）
            for h in hits:
                if os.access(h, os.X_OK) or h.endswith(".exe"):
                    return h
    return shutil.which("katago")


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
    ap.add_argument("--sgf-dir", default=os.path.join("data", "games", "games"))
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("data", "labels", "kata_labels.npz"))
    args = ap.parse_args()

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
    print(f"SGF 池 = {len(paths):,} 局")
    print(f"引擎   = {exe}")
    print(f"权重   = {os.path.basename(model)}")
    print(f"配置   = {os.path.basename(config)}")

    positions, n_sel = build_queries(paths, args.move_lo, args.move_hi,
                                     args.max_games, args.seed, args.game_frac)
    print(f"选中 {n_sel} 局 -> {args.move_lo}~{args.move_hi} 手共 "
          f"{len(positions):,} 个候选局面")
    positions = positions[: args.limit]
    print(f"本次打标签 {len(positions):,} 个局面，"
          f"maxVisits={args.max_visits}（引擎实际达成 {args.max_visits+1}），"
          f"并发={args.concurrency}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # **边打边落盘**（.npy 追加写），不是攒在内存里最后一次性 save。
    # 长跑（几小时~几天）中途若崩/被kill，已打的标签还在，可续跑；
    # 攒内存的写法一崩就全丢，这也是第一版跑完 8 分钟输出为空的原因之一。
    part = os.path.splitext(args.out)[0]
    # 引擎**必须在项目根启动**（KataGo 的 KataGoData/ 按 cwd 解析）
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done_marker = part + ".done.n"
    done = int(open(done_marker).read().strip()) \
        if os.path.exists(done_marker) else 0
    if done:
        print(f"检测到已完成 {done} 个局面，从该处续跑")

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
        todo = list(range(done, len(positions)))
        for cs in range(0, len(todo), conc):
            batch = todo[cs:cs + conc]
            reqs = []
            for i in batch:
                q = positions[i]
                reqs.append(dict(id=f"q{i}", moves=q["moves"],
                                 maxVisits=args.max_visits,
                                 boardXSize=BOARD, boardYSize=BOARD,
                                 rules="tromp-taylor", komi=7.5))
            resps = lab.query_many(reqs)
            for i in batch:
                q = positions[i]
                obj = resps[f"q{i}"]
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
            for f in handles.values():
                f.flush()
            open(done_marker, "w").write(str(done + n))
            el = time.perf_counter() - t0
            rate = n / max(el, 1e-9)
            eta = (len(positions) - done - n) / max(rate, 1e-9) / 3600
            print(f"  {done+n}/{len(positions)}  {el/60:.1f}min  "
                  f"{rate:.2f} 局面/s  ETA {eta:.2f}h", flush=True)
    finally:
        lab.close()
        for f in handles.values():
            f.close()

    # 拼成最终 npz
    m = done + n
    out = {k: np.fromfile(f"{part}.{k}.part.npy", dtype=dt).reshape(-1, *sh)
           for k, dt, sh in (("policy", np.float16, (N_ACTIONS,)),
                             ("pos_hash", np.uint64, ()),
                             ("root_win", np.float32, ()),
                             ("score_mean", np.float32, ()),
                             ("score_stdev", np.float32, ()),
                             ("visits", np.int16, ()),
                             ("game_idx", np.int32, ()),
                             ("move_idx", np.int32, ()))}
    out = {k: v[:m] for k, v in out.items()}
    np.savez(args.out, **out)
    el = time.perf_counter() - t0
    print(f"\n写入 {args.out}  行数={m:,}  用时 {el/60:.1f}min  "
          f"({n/max(el,1e-9):.2f} 局面/s)")
    if m:
        ps = out["policy"][:200].astype(np.float32).sum(1)
        print(f"  policy 每行和: min={ps.min():.4f} max={ps.max():.4f}（应≈1）")
        print(f"  visits: min={out['visits'][:200].min()} "
              f"max={out['visits'][:200].max()} 均值={out['visits'][:200].mean():.1f}")


if __name__ == "__main__":
    main()
