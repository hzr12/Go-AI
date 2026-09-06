#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载 yenw/computer-go-dataset 的 AlphaGo Zero 全部对局（去重 + 过滤）。

来源：AI/AlphaGo Zero —— DeepMind 公开的 AlphaGo Zero 对局，共 5 组：
  - Extended Data Figure 1 : 20-block vs AlphaGo Lee
  - Extended Data Figure 4 : 20-block self-play games
  - Extended Data Figure 5 : 40-block self-play games
  - Extended Data Figure 6 : 40-block vs AlphaGo Master
  - Figure 5               : Timeline
全部保留（不做配额截断）。

处理流程：
  1. GitHub contents API 递归枚举 AI/AlphaGo Zero 下所有文件；
  2. 逐个下载（zip/7z 解压，sgf 直收）；
  3. 兼容「一行一盘 SGF」的 txt 存放形式，自动拆成独立 .sgf；
  4. 按「规范化 B/W 着法序列」sha1 去重，过滤非 19 路 / 过短残局。

输出：data/games/games/agz/<序号>.sgf（build_dataset.py 会递归扫描）

用法：
  python scripts/fetch_games.py                 # 全量下载
  python scripts/fetch_games.py --cleanup       # 完成后删除下载与解压的暂存目录
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

GH_REPO = "yenw/computer-go-dataset"
GH_API = f"https://api.github.com/repos/{GH_REPO}/contents"
GH_RAW = f"https://raw.githubusercontent.com/{GH_REPO}/master"

AGZ_DIR = "AI/AlphaGo Zero"

# SGF 着法：;B[pd] / ;W[dd] ；pass 为空坐标 ;B[]
MOVE_RE = re.compile(r";([BW])\[([a-z]{0,2})\]")
SZ_RE = re.compile(r"SZ\[(\d+)\]")


def run(cmd):
    print("    $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def download(url, dest):
    """下载到 dest。用 urllib 而非 curl：本机 curl 走 schannel 会报
    CRYPT_E_NO_REVOCATION_CHECK（吊销检查失败）。带重试与备用域名。"""
    import time

    if dest.exists() and dest.stat().st_size > 0:
        print(f"      已存在，跳过：{dest.name} "
              f"({dest.stat().st_size / 1048576:.1f} MB)", flush=True)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    # raw.githubusercontent.com 偶发 502，准备备用域名重试
    urls = [url]
    if "raw.githubusercontent.com" in url:
        alt = url.replace("raw.githubusercontent.com", "github.com")
        for br in ("/master/", "/main/"):
            if br in alt:
                alt = alt.replace(br, f"/raw{br}")
        urls.append(alt)

    last_err = None
    for attempt in range(4):
        for u in urls:
            try:
                print(f"      GET {Path(u).name}  (第 {attempt + 1} 次)",
                      flush=True)
                with urllib.request.urlopen(u, timeout=300) as r, \
                        open(part, "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    done = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r      {done / 1048576:7.1f} / "
                                  f"{total / 1048576:.1f} MB",
                                  end="", flush=True)
                print(flush=True)
                part.replace(dest)
                print(f"      下载完成：{dest.name} "
                      f"({dest.stat().st_size / 1048576:.1f} MB)", flush=True)
                return dest
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"        失败：{e}", flush=True)
                time.sleep(2 * (attempt + 1))
    if part.exists():
        part.unlink()
    raise RuntimeError(f"下载失败 {url}: {last_err}")


def extract(archive, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(outdir)
    else:
        # 7z / NanaZip：-o 后无空格，-y 全部确认
        run(["7z", "x", str(archive), f"-o{outdir}", "-y"])
    return outdir


def github_list(subdir):
    """列出 GitHub 目录内容（GitHub API 强制要求 User-Agent 头）。"""
    url = f"{GH_API}/{urllib.parse.quote(subdir)}?ref=master"
    req = urllib.request.Request(url, headers={"User-Agent": "goai-fetch-games"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def github_walk(subdir, depth=0):
    """递归枚举目录下所有文件条目。"""
    for e in github_list(subdir):
        if e.get("type") == "file":
            yield e
        elif e.get("type") == "dir" and depth < 3:
            yield from github_walk(f"{subdir}/{e['name']}", depth + 1)


def split_txt_sgf(exdir):
    """「一行一盘 SGF」的 txt 拆成独立 .sgf（部分数据集的存放方式）。"""
    out = []
    for txt in list(exdir.rglob("*.txt")):
        base = txt.parent / (txt.stem + "_split")
        if not any(base.glob("*.sgf")):
            base.mkdir(parents=True, exist_ok=True)
            n = 0
            with open(txt, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("("):
                        continue
                    (base / f"{txt.stem}_{n:05d}.sgf").write_text(
                        line, encoding="utf-8")
                    n += 1
            print(f"      {txt.name} 拆分出 {n} 局", flush=True)
        out.extend(base.glob("*.sgf"))
    return out


def collect_sgf(exdir):
    """收集目录下所有 .sgf；若无则尝试拆分 txt。"""
    got = [p for p in exdir.rglob("*.sgf")]
    if not got:
        got = split_txt_sgf(exdir)
    return got


def canonical_key(text, board_size=19, min_moves=20):
    """规范化着法序列指纹；非目标尺寸或过短残局返回 None。"""
    sz = SZ_RE.search(text)
    if sz and int(sz.group(1)) != board_size:
        return None
    moves = MOVE_RE.findall(text)
    if len(moves) < min_moves:
        return None
    return hashlib.sha1(repr(moves).encode("utf-8")).hexdigest()


def fetch_github_dir(subdir, name, tmp):
    """枚举并下载 GitHub 目录下所有文件，返回收集到的 .sgf 路径列表。"""
    entries = list(github_walk(subdir))
    print(f"  递归枚举到 {len(entries)} 个文件", flush=True)
    files = []
    for i, e in enumerate(entries):
        url = e.get("download_url")
        if not url:
            continue
        print(f"    [{i + 1}/{len(entries)}] {e['name']}", flush=True)
        suffix = Path(e["name"]).suffix.lower().lstrip(".")
        dest = tmp / f"{name}_{i:03d}.{suffix or 'bin'}"
        download(url, dest)
        exdir = tmp / f"{name}_{i:03d}_x"
        if suffix in ("zip", "7z"):
            if not any(exdir.rglob("*.sgf")):
                extract(dest, exdir)
        else:
            # 裸 sgf/txt：放进独立目录统一收集
            exdir.mkdir(parents=True, exist_ok=True)
            target = exdir / e["name"]
            if not target.exists():
                shutil.copy2(dest, target)
        got = collect_sgf(exdir)
        print(f"      -> {len(got)} 局", flush=True)
        files.extend(got)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/games/games",
                    help="输出根目录（build_dataset 直接扫这里）")
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--min-moves", type=int, default=20,
                    help="着法数少于该值视为残局/空局，丢弃")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp", default=".cache_fetch_games",
                    help="下载与解压的暂存目录")
    ap.add_argument("--cleanup", action="store_true",
                    help="完成后删除暂存目录")
    args = ap.parse_args()

    out = Path(args.out)
    tmp = Path(args.tmp)
    rng = __import__("random").Random(args.seed)
    seen = set()

    print(f"\n=== AlphaGo Zero ({GH_REPO}/{AGZ_DIR}) ===", flush=True)
    files = fetch_github_dir(AGZ_DIR, "agz", tmp)
    print(f"  合计收集到 {len(files)} 个 .sgf", flush=True)

    rng.shuffle(files)
    dst = out / "agz"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    kept = dup = bad = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        key = canonical_key(text, args.board_size, args.min_moves)
        if key is None:
            bad += 1
            continue
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        shutil.copy2(f, dst / f"{kept:06d}.sgf")
        kept += 1

    print(f"\n写入 {kept} 局 -> {dst}  "
          f"（重复 {dup}，尺寸/残局过滤 {bad}）", flush=True)
    print("下一步：", flush=True)
    print(f"  python scripts/build_dataset.py --src {args.out} "
          f"--out data/sgf_19x19_agz.npz --board-size {args.board_size}",
          flush=True)

    if args.cleanup:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"已清理暂存目录 {tmp}", flush=True)
    else:
        print(f"暂存目录保留在 {tmp}（可手动删除）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
