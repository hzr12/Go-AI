#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载并去重构建 19 路 SGF 棋谱集（供 build_dataset.py / train_sft.py 使用）。

来源：featurecat/go-dataset Pro —— Fox 职业对局 10,349 局，默认取 10,000 局。

已放弃的备选来源（保留说明以免重复踩坑）：
  - Waltheri（ps.waltheri.net，85,518 局职业棋谱）：只有在线检索与逐局回放，
    没有任何批量下载入口，无法合规地批量抓取。
  - featurecat 9d（Fox 9 段 166,184 局，9d.7z 43.7MB）与 Karesis/GoDatas
    （games.zip 59MB，源自 CWI 棋谱库）：按需求不再下载。
    单源时仍需去重——同一批 Fox 棋谱里也存在完全相同的对局。

去重策略：不比对文件字节（同一局在不同来源里格式/元数据可能不同），而是提取
「规范化 B/W 着法序列」取 sha1，跨源剔除重复。同时过滤非目标尺寸与过短残局。

输出：data/games/games/<source>/<序号>.sgf（build_dataset.py 会递归扫描）

用法：
  python scripts/fetch_games.py                     # 每个来源 10000 局
  python scripts/fetch_games.py --per-source 5000
  python scripts/fetch_games.py --cleanup           # 完成后删除下载的压缩包与解压目录
"""
import argparse
import hashlib
import random
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RAW = "https://raw.githubusercontent.com"

# Pro 拆成两个压缩包：Pro.7z（大头）+ Pro2.7z，合计 10,349 局。
# 只下 Pro.7z 拿不满 1 万局，必须两个都下再合并。
SOURCES = {
    "foxpro": [
        (f"{RAW}/featurecat/go-dataset/master/Pro/Pro.7z", "7z"),
        (f"{RAW}/featurecat/go-dataset/master/Pro/Pro2.7z", "7z"),
    ],
}

# SGF 着法：;B[pd] / ;W[dd] ；pass 为空坐标 ;B[]
MOVE_RE = re.compile(r";([BW])\[([a-z]{0,2})\]")
SZ_RE = re.compile(r"SZ\[(\d+)\]")


def run(cmd):
    print("    $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def download(url, dest):
    """下载到 dest。用 urllib 而非 curl：本机 curl 走 schannel 会报
    CRYPT_E_NO_REVOCATION_CHECK（吊销检查失败），urllib 可正常握手。"""
    import urllib.request

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  已存在，跳过下载：{dest.name} "
              f"({dest.stat().st_size / 1048576:.1f} MB)", flush=True)
        return dest
    import time
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
                print(f"  GET {u}  (第 {attempt + 1} 次)", flush=True)
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
                            print(f"\r  {done / 1048576:7.1f} / "
                                  f"{total / 1048576:.1f} MB",
                                  end="", flush=True)
                print(flush=True)
                if not dest.exists() or part.stat().st_size > 0:
                    part.replace(dest)
                print(f"  下载完成：{dest.name} "
                      f"({dest.stat().st_size / 1048576:.1f} MB)", flush=True)
                return dest
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"    失败：{e}", flush=True)
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


def canonical_key(text, board_size=19, min_moves=20):
    """规范化着法序列指纹；非目标尺寸或过短残局返回 None。"""
    sz = SZ_RE.search(text)
    if sz and int(sz.group(1)) != board_size:
        return None
    moves = MOVE_RE.findall(text)
    if len(moves) < min_moves:
        return None
    return hashlib.sha1(repr(moves).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=10000,
                    help="每个来源保留的局数")
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
    rng = random.Random(args.seed)
    seen = set()
    grand_total = 0

    for name, archives in SOURCES.items():
        print(f"\n=== {name} ===", flush=True)
        files = []
        for idx, (url, kind) in enumerate(archives):
            archive = tmp / f"{name}{idx}.{kind}"
            download(url, archive)
            exdir = tmp / f"{name}{idx}_x"
            if not any(exdir.rglob("*.sgf")):
                extract(archive, exdir)
            got = [p for p in exdir.rglob("*.sgf")]
            print(f"  {archive.name} 解压得到 {len(got)} 个 .sgf", flush=True)
            files.extend(got)
        print(f"  合计 {len(files)} 个 .sgf", flush=True)

        rng.shuffle(files)
        dst = out / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        kept = dup = bad = 0
        for f in files:
            if kept >= args.per_source:
                break
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

        print(f"  写入 {kept} 局 -> {dst}  "
              f"（跨源重复 {dup}，尺寸/残局过滤 {bad}）", flush=True)
        grand_total += kept

    print(f"\n完成：共 {grand_total} 局，输出目录 {out.resolve()}", flush=True)
    print("下一步：", flush=True)
    print(f"  python scripts/build_dataset.py --src {args.out} "
          f"--out data/sgf_19x19.npz --board-size {args.board_size}", flush=True)

    if args.cleanup:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"已清理暂存目录 {tmp}", flush=True)
    else:
        print(f"暂存目录保留在 {tmp}（可手动删除）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
