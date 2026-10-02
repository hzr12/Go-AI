"""扫 KataGo analysis 并发参数，找本机（gfx90c + b10c384）的最高吞吐点。

为什么必须实测
--------------
`kata_analyze` 的吞吐不只取决于线程数，还取决于**每次 GPU 前向的 batch 大小**
（引擎结束时打印 "NN avg batch size"）。batch=1 意味着 GPU 每次只算一个样本，
完全没用上并行度 —— 这时加线程几乎没用。batch 涨上去才可能线性加速。

所以每个配置都要读引擎日志里的 `NN rows / NN batches`，用真实 batch size 佐证，
不能只看墙钟（墙钟会被 CPU 侧树遍历、木桶效应掩盖）。

用法
----
    python scripts/bench_katago.py --configs "4x2,8x1,16x1,8x2,16x2" --visits 80
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ⚠ 临时文件一律放仓库内的 tmp/，**不要用系统 TEMP**：本机 TEMP 在 C 盘且
#   空间/权限不稳，引擎日志被锁时也难以排查。路径集中在一处便于清理。
TMPDIR = os.path.join(REPO, "tmp", "bench")
os.makedirs(TMPDIR, exist_ok=True)
BENCH_LOGS = os.path.join(TMPDIR, "engine_logs")
os.makedirs(BENCH_LOGS, exist_ok=True)
EXE = os.path.join(REPO, "katago", "katago-v1.18.1-opencl-windows-x64", "katago.exe")
MODEL = os.path.join(REPO, "katago", "kata1-tf2-b10c384-s2941M-d5872M.bin.gz")
BOARD = 19
_COLS = "ABCDEFGHJKLMNOPQRST"


def make_cfg(num_analysis, num_search, batch, cache_pow2=23, extra=""):
    """生成一个临时 config。

    ⚠ 键名必须用 `numSearchThreads`（短名）。KataGo 会先自动加载
    `default_gtp.cfg`，其中已有 `numSearchThreads = 5`；若本文件再写
    `numSearchThreadsPerAnalysisThread`（长名），引擎会认为两个别名键都被指定
    而直接退出：`Cannot specify both ... in the same config`。
    """
    return f"""logDir = {BENCH_LOGS.replace(os.sep, "/")}
maxVisits = 500
numAnalysisThreads = {num_analysis}
numSearchThreads = {num_search}
nnMaxBatchSize = {batch}
nnCacheSizePowerOfTwo = {cache_pow2}
useGraphSearch = true
{extra}
"""


def load_positions(n, seed=0):
    """从自有 SGF 取 n 个中盘局面（100~200 手），化成 GTP 着法序列。"""
    from src.data.sgf_parser import SGFParser
    base = os.path.join(REPO, "data", "games", "games")
    pools = []
    for d in sorted(os.listdir(base)):
        fs = sorted(glob.glob(os.path.join(base, d, "*.sgf")))
        if fs:
            pools.append(fs)
    rng = random.Random(seed)
    out = []
    for fs in pools:
        for f in (rng.sample(fs, min(400, len(fs))) if len(fs) > 400 else fs):
            if len(out) >= n:
                break
            try:
                g = SGFParser().parse_file(f)
            except Exception:
                continue
            if g is None or g.board_size != BOARD or len(g.moves) < 140:
                continue
            seq = []
            for mv in g.moves[:130]:
                r, c = mv.position
                seq.append([mv.color, "pass" if (r, c) == (-1, -1)
                            else f"{_COLS[c]}{BOARD - r}"])
            out.append(seq)
        if len(out) >= n:
            break
    return out


def run_one(num_analysis, num_search, batch, positions, visits, conc, cache_pow2=23):
    """跑一个配置，返回 (墙钟秒, 实际处理数, avg_batch, engine_log)。"""
    cfg_path = os.path.join(TMPDIR, f"kb_{num_analysis}x{num_search}_{batch}.cfg")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(make_cfg(num_analysis, num_search, batch, cache_pow2))
    logdir = BENCH_LOGS
    for p in glob.glob(os.path.join(logdir, "*.log")):
        try:
            os.remove(p)                          # 清掉上一轮，便于定位本次日志
        except Exception:
            pass

    p = subprocess.Popen([EXE, "analysis", "-model", MODEL, "-config", cfg_path],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=REPO, text=True,
                         encoding="utf-8", errors="ignore", bufsize=1)
    t_ready = time.perf_counter()
    probe = dict(id="p", moves=[], maxVisits=1, boardXSize=BOARD,
                 boardYSize=BOARD, rules="tromp-taylor", komi=7.5)
    p.stdin.write(json.dumps(probe) + "\n"); p.stdin.flush()
    while True:
        line = p.stdout.readline()
        if line.startswith("{"):
            break
    ready = time.perf_counter() - t_ready

    t0 = time.perf_counter()
    done = 0
    log_copy = ""
    try:
        for i in range(0, len(positions), conc):
            batch_reqs = [dict(id=f"q{i+k}", moves=positions[i+k], maxVisits=visits,
                               boardXSize=BOARD, boardYSize=BOARD,
                               rules="tromp-taylor", komi=7.5)
                          for k in range(min(conc, len(positions) - i))]
            for r in batch_reqs:
                p.stdin.write(json.dumps(r) + "\n")
            p.stdin.flush()
            want = {r["id"] for r in batch_reqs}
            got = set()
            while len(got) < len(want):
                ln = p.stdout.readline()
                if not ln:
                    raise RuntimeError("engine died")
                if ln.startswith("{"):
                    o = json.loads(ln)
                    if o.get("id") in want:
                        got.add(o["id"])
            done += len(want)
            # 引擎的 NN 统计是**每 200 次前向**才打一行（见其日志节奏），
            # 所以中途抓一次当前值，比等退出更稳。
            for lg in glob.glob(os.path.join(logdir, "*.log")):
                try:
                    cp = os.path.join(TMPDIR, "log_snap.copy")
                    shutil.copyfile(lg, cp)
                    log_copy += open(cp, encoding="utf-8",
                                     errors="ignore").read()
                except Exception:
                    pass
        elapsed = time.perf_counter() - t0
    finally:
        try:
            p.stdin.write(json.dumps({"id": "z", "action": "terminate"}) + "\n")
            p.stdin.flush()
        except Exception:
            pass
        try:
            p.wait(timeout=20)
        except Exception:
            p.kill()
        try:
            stderr_txt = p.stderr.read()
        except Exception:
            stderr_txt = ""

    avg_batch = 0.0
    # 中途抓的日志副本优先（引擎退出后原文件可能被锁/删除）
    for src_txt in (log_copy, stderr_txt):
        m = re.findall(r"NN avg batch size:\s*([\d.]+|nan)", src_txt or "")
        if m and m[-1] not in ("nan", "-nan(ind)"):
            avg_batch = float(m[-1])
            break
    return elapsed, done, avg_batch, ready, log_copy


def main():
    ap = argparse.ArgumentParser(description="扫 KataGo analysis 并发参数")
    ap.add_argument("--configs", default="4x2,8x1,16x1,8x2,16x2",
                    help="每项为 A x S = numAnalysisThreads x numSearchThreads")
    ap.add_argument("--visits", type=int, default=80)
    ap.add_argument("--positions", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cache-pow2", type=int, default=23)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pos = load_positions(args.positions, args.seed)
    print(f"取到 {len(pos)} 个中盘局面，maxVisits={args.visits}")
    print("（吞吐 = 只统计搜索阶段，不含引擎启动）\n")
    hdr = f"{'A x S':>8} {'总线程':>7} {'墙钟':>8} {'局面':>5} {'局面/s':>8} {'avgB':>7} {'启动':>7}"
    print(hdr); print("-" * len(hdr))
    results = []
    for spec in args.configs.split(","):
        a, s = (int(x) for x in spec.lower().split("x"))
        conc = a                                   # 并发请求数 = 引擎并发槽
        try:
            el, done, ab, ready, _ = run_one(a, s, args.batch, pos, args.visits,
                                              conc, args.cache_pow2)
        except Exception as e:
            print(f"{spec:>8}  FAILED: {e}")
            continue
        rate = done / max(el, 1e-9)
        print(f"{spec:>8} {a*s:>7} {el:>7.1f}s {done:>5} {rate:>8.2f} {ab:>7.2f} {ready:>6.1f}s")
        results.append((spec, a * s, rate, ab))
    if results:
        best = max(results, key=lambda x: x[2])
        print(f"\n最优: {best[0]}  ->  {best[2]:.2f} 局面/s"
              f"  (avg batch {best[3]:.2f})")


if __name__ == "__main__":
    main()
