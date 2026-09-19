#!/usr/bin/env python3
"""处理 Leela Zero SGF 数据，提取最近 5 万局 19x19 棋谱。

用法:
    python scripts/process_leela.py --input data/all.sgf.xz --output data/games/games/leela_zero/
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def parse_sgf_date(text):
    """从 SGF 文本中提取日期。"""
    # 查找 DT[] 字段
    dt_match = re.search(r'DT\[([^\]]+)\]', text)
    if dt_match:
        date_str = dt_match.group(1)
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            pass
    # 如果没有日期，返回最早日期
    return datetime(1900, 1, 1)


def is_19x19(text):
    """检查是否为 19x19 棋盘。"""
    return bool(re.search(r'SZ\[19\]', text))


def has_result(text):
    """检查是否有明确结果。"""
    return bool(re.search(r'RE\[(B|W)[+\-X\s]', text))


def extract_game(text):
    """提取单局棋谱。"""
    # 去除开头括号
    text = text.strip()
    if not text.startswith('('):
        text = '(' + text
    # 确保有结束括号
    if not text.endswith(')'):
        text += ')'
    return text


def process_xz_to_sgf(xz_path, sgf_path):
    """解压 xz 文件到 sgf。"""
    print(f"解压 {xz_path.name} -> {sgf_path.name}")
    # 尝试使用 Python 内置 lzma 模块解压
    import lzma
    with lzma.open(xz_path, 'rb') as f_in:
        with open(sgf_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"  解压完成: {sgf_path} ({sgf_path.stat().st_size / 1e9:.1f} GB)")
    return sgf_path


def split_sgf_to_games(sgf_path, out_dir):
    """将大 SGF 文件拆分成单局 SGF 文件。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"拆分 SGF: {sgf_path.name}")
    print(f"文件大小: {sgf_path.stat().st_size / 1e9:.1f} GB")

    # 分块读取，避免内存溢出
    games = []
    current_game = []
    bracket_count = 0

    with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            chunk = f.read(1024 * 1024 * 10)  # 10MB chunks
            if not chunk:
                break

            for char in chunk:
                if char == '(':
                    bracket_count += 1
                    current_game.append(char)
                elif char == ')':
                    bracket_count -= 1
                    current_game.append(char)
                    if bracket_count == 0:
                        game_text = ''.join(current_game)
                        if len(game_text) > 1000:  # 最小长度过滤
                            games.append(game_text)
                        current_game = []
                else:
                    if bracket_count > 0:
                        current_game.append(char)

    print(f"  提取到 {len(games)} 局棋谱")
    return games


def filter_and_save(games, out_dir, max_games=50000):
    """筛选并保存最近的 max_games 局。"""
    out_dir = Path(out_dir)

    # 解析每局棋谱
    parsed_games = []
    for i, game in enumerate(games):
        if i % 10000 == 0:
            print(f"  解析进度: {i}/{len(games)}")

        # 检查是否为 19x19
        if not is_19x19(game):
            continue

        # 检查是否有结果
        if not has_result(game):
            continue

        # 提取日期
        date = parse_sgf_date(game)

        # 生成唯一 key (基于着法序列)
        moves = re.findall(r';[BW]\[[a-z]*\]', game)
        key = hashlib.sha1(','.join(moves).encode()).hexdigest()[:16]

        parsed_games.append({
            'text': extract_game(game),
            'date': date,
            'key': key,
        })

    print(f"  19x19 且有结果的棋谱: {len(parsed_games)} 局")

    # 去重
    seen_keys = set()
    unique_games = []
    for g in parsed_games:
        if g['key'] not in seen_keys:
            seen_keys.add(g['key'])
            unique_games.append(g)

    print(f"  去重后: {len(unique_games)} 局")

    # 按日期排序，取最近的 max_games 局
    unique_games.sort(key=lambda x: x['date'], reverse=True)
    selected = unique_games[:max_games]

    print(f"  选择最近 {len(selected)} 局")

    # 保存
    saved = 0
    for i, g in enumerate(selected):
        if i % 10000 == 0:
            print(f"  保存进度: {i}/{len(selected)}")
        out_path = out_dir / f"{saved:07d}.sgf"
        out_path.write_text(g['text'], encoding='utf-8')
        saved += 1

    print(f"  保存完成: {saved} 局 -> {out_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser(description='处理 Leela Zero SGF 数据')
    ap.add_argument('--input', required=True, help='输入的 xz 文件路径')
    ap.add_argument('--output', default='data/games/games/leela_zero',
                    help='输出目录')
    ap.add_argument('--max-games', type=int, default=50000,
                    help='最大棋局数 (默认 50000)')
    ap.add_argument('--cleanup', action='store_true',
                    help='处理后删除临时文件')
    args = ap.parse_args()

    xz_path = Path(args.input)
    out_dir = Path(args.output)

    if not xz_path.exists():
        print(f"错误: 文件不存在 {xz_path}")
        return 1

    # 创建临时目录
    tmp_dir = xz_path.parent / '.leela_tmp'
    tmp_dir.mkdir(exist_ok=True)

    try:
        # Step 1: 解压 xz
        sgf_path = tmp_dir / 'all.sgf'
        if not sgf_path.exists():
            process_xz_to_sgf(xz_path, sgf_path)
        else:
            print(f"SGF 文件已存在，跳过解压")

        # Step 2: 拆分棋谱
        games = split_sgf_to_games(sgf_path, tmp_dir)

        # Step 3: 筛选并保存
        saved = filter_and_save(games, out_dir, args.max_games)

        print(f"\n处理完成！")
        print(f"  输出目录: {out_dir.resolve()}")
        print(f"  棋局数: {saved}")

        return 0

    finally:
        # Cleanup
        if args.cleanup:
            print(f"清理临时文件...")
            if sgf_path.exists():
                sgf_path.unlink()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"  已删除临时目录")


if __name__ == '__main__':
    sys.exit(main())
