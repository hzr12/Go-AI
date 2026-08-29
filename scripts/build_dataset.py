"""
云端数据构建：从 SGF（tgz 或目录）流式解析 -> GoBoard 重放 ->
生成紧凑内存数据集并保存为 npz。

紧凑布局见 src/data/dataset.py 顶部注释。全程不落临时文件，常驻内存。

用法:
    python scripts/build_dataset.py --src data/games.tgz --out data/sft_dataset.npz --board-size 19 --max-games 30000
    python scripts/build_dataset.py --src data/games/      --out data/sft_dataset.npz --board-size 19
"""

import argparse
import os
import sys
import tarfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.game.go_rules import GoBoard
from src.data.sgf_parser import SGFParser


def iter_sgf_bytes(src):
    """从 tgz 或目录产出 (name, bytes)。"""
    if os.path.isdir(src):
        import glob
        for fn in glob.glob(os.path.join(src, '**', '*.sgf'), recursive=True):
            with open(fn, 'rb') as f:
                yield fn, f.read()
    elif src.endswith('.tgz') or src.endswith('.tar.gz'):
        with tarfile.open(src, 'r:gz') as t:
            for m in t.getmembers():
                if m.name.lower().endswith('.sgf'):
                    yield m.name, t.extractfile(m).read()
    else:
        with open(src, 'rb') as f:
            yield os.path.basename(src), f.read()


def parse_result_to_value(result_str):
    """解析 RE 字段 -> 黑方视角胜负标签 (+1/-1)。未知返回 None。"""
    if not result_str:
        return None
    r = result_str.strip().upper()
    if r.startswith('B'):
        return 1
    if r.startswith('W'):
        return -1
    try:
        if abs(float(r)) < 1e-6:
            return 0
    except ValueError:
        pass
    return None


def pad3(seq):
    seq = list(seq[-3:])
    while len(seq) < 3:
        seq.insert(0, -1)
    return seq


def split_hist(recent, to_play):
    """
    把 recent（扁平坐标序列，最近一手在末尾）拆成 my/op 各 3 手。
    recent 是按落子顺序排列的；最后一步的执子方是 to_play，往前交替。
    """
    my, op = [], []
    cur = to_play
    for mv in reversed(recent):
        if cur == to_play:
            my.append(mv)
        else:
            op.append(mv)
        cur = -cur
    return my[:3], op[:3]


def build(src, board_size, max_games):
    parser = SGFParser()
    boards, my_hists, op_hists, kos, moves, values, to_plays = [], [], [], [], [], [], []

    # 数据源解析：目录模式下递归收集所有 .tgz/.tar.gz 与子目录（各子源再各自
    # 递归找 .sgf / 解 tgz），合并成一个数据集。单文件/tgz 则只有自身一个源。
    if os.path.isdir(src):
        import glob
        tgzs = (sorted(glob.glob(os.path.join(src, '**', '*.tgz'), recursive=True))
                + sorted(glob.glob(os.path.join(src, '**', '*.tar.gz'), recursive=True)))
        subdirs = sorted(d for d in glob.glob(os.path.join(src, '*')) if os.path.isdir(d))
        sources = tgzs + subdirs
        if not sources:
            sources = [src]  # 目录本身直接含 .sgf
    else:
        sources = [src]

    n_games = 0
    skip = 0
    for s in sources:
        try:
            stream = iter_sgf_bytes(s)
        except Exception as e:  # noqa: BLE001
            print(f"[build] 跳过分片 {s}：{e}")
            continue
        for name, raw in stream:
            if max_games and n_games >= max_games:
                break
            text = raw.decode('utf-8', 'ignore')
            game = parser.parse_string(text)
            # 只跳过「大于目标尺寸」的棋谱（无法放进小棋盘）。
            # 小于目标的（如 9x9 棋谱喂到 19x19）做居中 pad，见下方 off。
            if game is None or game.board_size > board_size:
                skip += 1
                continue
            # 小棋盘居中到大棋盘的偏移量（9x9->19x19 时 off=5，棋形居中不偏）
            off = (board_size - game.board_size) // 2 if game.board_size != board_size else 0
            # 校验所有坐标在原始棋谱尺寸范围内（pass 为 -1 合法），超出则整局丢弃。
            ok = True
            for mv in game.moves:
                r, c = mv.position
                if (r, c) != (-1, -1) and not (0 <= r < game.board_size and 0 <= c < game.board_size):
                    ok = False
                    break
            if not ok:
                skip += 1
                continue
            if len(game.moves) < 2:
                skip += 1
                continue
            value = parse_result_to_value(game.result)
            if value is None:
                skip += 1
                continue

            board = GoBoard(board_size, komi=game.komi)
            history = []  # 扁平坐标序列（pass 记为 -1）
            for mv in game.moves:
                to_play = board.current_player
                recent = history[-3:] if len(history) >= 3 else history
                my_h, op_h = split_hist(recent, to_play)
                ko = board.ko_point
                r, c = mv.position
                # 落子坐标：pass 记 -1；否则把小棋盘坐标居中映射到大棋盘
                target = -1 if (r, c) == (-1, -1) else (r + off) * board_size + (c + off)

                boards.append(board.board.copy())
                my_hists.append(pad3(my_h))
                op_hists.append(pad3(op_h))
                kos.append(ko)
                moves.append(target)
                values.append(value)
                to_plays.append(to_play)

                play_move = -1 if (r, c) == (-1, -1) else (r + off) * board_size + (c + off)
                board.play(play_move)
                history.append(play_move)

            n_games += 1

    if n_games == 0:
        raise RuntimeError("未解析到任何有效棋谱，请检查 --src 与 --board-size")

    data = {
        'boards': np.stack(boards).astype(np.int8),
        'my_hist': np.stack(my_hists).astype(np.int16),
        'op_hist': np.stack(op_hists).astype(np.int16),
        'ko': np.array(kos, dtype=np.int16),
        'moves': np.array(moves, dtype=np.int16),
        'values': np.array(values, dtype=np.int8),
        'to_play': np.array(to_plays, dtype=np.int8),
    }
    return data, n_games, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default='data/sft_dataset.npz')
    ap.add_argument('--board-size', type=int, default=19)
    ap.add_argument('--max-games', type=int, default=0)
    args = ap.parse_args()

    data, n, skip = build(args.src, args.board_size, args.max_games or 0)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"构建完成: 有效局 {n}, 跳过 {skip}, 样本 {data['boards'].shape[0]}, 保存至 {args.out}")


if __name__ == '__main__':
    main()
