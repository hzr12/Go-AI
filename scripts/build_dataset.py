"""
云端数据构建：从 SGF（tgz 或目录）流式解析 -> GoBoard 重放 ->
生成紧凑内存数据集并保存为 npz。

紧凑布局见 src/data/dataset.py 顶部注释。全程不落临时文件，常驻内存。

用法:
    python scripts/build_dataset.py --src data/games.tgz --out data/sft_dataset.npz --board-size 19 --max-games 30000
    python scripts/build_dataset.py --src data/games/      --out data/sft_dataset.npz --board-size 19
    # 流式构建被中断后，把残留的 chunk_*.npz 分片断点合并成最终 npz：
    python scripts/build_dataset.py --merge --src data/sft_build_xxx/ --out data/sft_dataset.npz
"""

import argparse
import os
import sys
import tarfile
import tempfile
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
    """补齐到 3 个元素，最近一手在索引 0（channel 最小），不足的在尾部填 -1。"""
    seq = list(seq[-3:])
    while len(seq) < 3:
        seq.append(-1)
    return seq


def split_hist(recent, to_play):
    """
    把 recent（扁平坐标序列，最近一手在末尾）拆成 my/op 各 3 手。
    recent 是按落子顺序排列的；最后一步的执子方是 to_play，往前交替。
    注意：recent[-1] 是上一手（由 -to_play 下的），所以从 -to_play 开始。
    """
    my, op = [], []
    cur = -to_play  # 最近一手由对手（-to_play）下的
    for mv in reversed(recent):
        if cur == to_play:
            my.append(mv)
        else:
            op.append(mv)
        cur = -cur
    return my[:3], op[:3]


def _flush_chunk(chunk, tmp_dir, idx):
    """把一个 chunk（dict of list）stack 后存成临时 npz 分片，释放内存。"""
    path = os.path.join(tmp_dir, f"chunk_{idx:05d}.npz")
    np.savez(path,
             boards=np.stack(chunk['boards']).astype(np.int8),
             my_hist=np.stack(chunk['my_hists']).astype(np.int16),
             op_hist=np.stack(chunk['op_hists']).astype(np.int16),
             ko=np.array(chunk['kos'], dtype=np.int16),
             moves=np.array(chunk['moves'], dtype=np.int16),
             values=np.array(chunk['values'], dtype=np.int8),
             to_play=np.array(chunk['to_plays'], dtype=np.int8))
    return path


def _merge_chunks(tmp_files, out):
    """把若干 npz 分片按字段 concatenate，压缩合并成单个 npz。"""
    keys = ['boards', 'my_hist', 'op_hist', 'ko', 'moves', 'values', 'to_play']
    merged = {k: [] for k in keys}
    for f in tmp_files:
        d = np.load(f, allow_pickle=False)
        for k in keys:
            merged[k].append(d[k])
        d.close()
    merged = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    np.savez_compressed(out, **merged)
    for f in tmp_files:
        try:
            os.remove(f)
        except OSError:
            pass
    return merged['boards'].shape[0]


def merge_shards(src_dir, out, group=8, clean=False):
    """断点合并：把 src_dir 下已落盘的 chunk 分片（*.npz）合并成单个 out。

    适用场景：流式构建被中断、临时目录（如 sft_build_xxx/）里残留一堆
    chunk_XXXXX.npz，用本函数把它们合成最终 npz，且中途可断点续做。

    特性（断点可恢复 + 内存有界）：
      * 两阶段。阶段1 把每 `group` 个 chunk 合并成一个 part_*.npz（中间分片），
        峰值内存仅约 `group` 个 chunk；阶段2 把全部 part 合并成最终 out。
      * 以“文件是否已处理”作为进度状态：阶段1 中已生成 part 的组会被跳过，
        中断后重跑自动续做剩余组；阶段2 先写 out.tmp 再原子 rename，并落
        <out>.done 标记，避免半截/重复输出。
      * 每个 chunk 一旦并入 part，即从 src_dir 移入 _consumed/（--merge-clean
        则直接删除），保证反复重跑不会重复计数。
    """
    import glob as _glob
    import shutil as _shutil
    keys = ['boards', 'my_hist', 'op_hist', 'ko', 'moves', 'values', 'to_play']
    out_base = os.path.basename(out)
    done_marker = out + '.done'
    if os.path.isfile(done_marker) and os.path.isfile(out):
        print(f"[merge] 已完成（存在 {done_marker}），跳过: {out}")
        return out

    chunks = sorted(
        f for f in _glob.glob(os.path.join(src_dir, '*.npz'))
        if os.path.basename(f) != out_base
        and not os.path.basename(f).startswith('part_')
        and not f.endswith('.done')
    )
    parts_dir = os.path.join(src_dir, 'parts')
    consumed_dir = os.path.join(src_dir, '_consumed')
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(consumed_dir, exist_ok=True)

    # 阶段1：chunk -> part（每组 group 个，内存仅约 group 个 chunk）
    n_groups = (len(chunks) + group - 1) // group if chunks else 0
    for gi in range(n_groups):
        part_path = os.path.join(parts_dir, f"part_{gi:05d}.npz")
        if os.path.isfile(part_path):
            continue  # 断点续做：该组已合并，跳过
        grp = chunks[gi * group:(gi + 1) * group]
        merged = {k: [] for k in keys}
        for cf in grp:
            d = np.load(cf, allow_pickle=False)
            for k in keys:
                merged[k].append(d[k])
            d.close()
        merged = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
        np.savez_compressed(part_path, **merged)
        for cf in grp:
            if clean:
                try:
                    os.remove(cf)
                except OSError:
                    pass
            else:
                _shutil.move(cf, consumed_dir)  # 移走，避免重跑重复计数
        print(f"[merge] 阶段1: 组 {gi + 1}/{n_groups} -> {os.path.basename(part_path)} "
              f"（{merged['boards'].shape[0]} 样本）", flush=True)

    # 阶段2：part -> out（原子写入）
    parts = sorted(_glob.glob(os.path.join(parts_dir, 'part_*.npz')))
    if not parts:
        raise RuntimeError(f"未在 {src_dir} 找到任何可合并的 chunk/part 分片")
    merged = {k: [] for k in keys}
    for pf in parts:
        d = np.load(pf, allow_pickle=False)
        for k in keys:
            merged[k].append(d[k])
        d.close()
    merged = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    tmp_out = out + '.tmp'
    np.savez_compressed(tmp_out, **merged)
    os.replace(tmp_out, out)  # 原子替换，避免半截文件
    with open(done_marker, 'w') as fh:
        fh.write(str(int(merged['boards'].shape[0])))
    for pf in parts:
        if clean:
            try:
                os.remove(pf)
            except OSError:
                pass
        else:
            _shutil.move(pf, consumed_dir)
    try:
        os.rmdir(parts_dir)
    except OSError:
        pass
    print(f"[merge] 合并完成: {len(parts)} 个分片 -> {out}（{merged['boards'].shape[0]} 样本）",
          flush=True)
    return out


def build(src, board_size, max_games, chunk_size=0, out=None, tmp_root=None):
    """构建数据集。

    - chunk_size=0（默认）：全量常驻内存，返回 (data_dict, n_games, skip)，
      兼容 train_sft.load_from_path 的调用。
    - chunk_size>0 且 out 给定：每攒够 chunk_size 个样本就 flush 成临时分片，
      最后合并成单个 npz（峰值内存仅约一个 chunk），返回 (out_path, n_games, skip)。

    tmp_root: 流式分片暂存目录的**父目录**。默认 None 时由调用方（main）传入
      输出文件所在目录——不要落到系统临时目录（Windows 上是 C 盘，大数据集
      会把 C 盘撑爆）。
    """
    parser = SGFParser()
    streaming = chunk_size and out is not None

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

    # 流式模式：用临时目录攒分片
    tmp_dir = None
    tmp_files = []
    chunk_idx = 0
    total_flushed = 0   # 已落盘样本累计（避免每落一片就重读所有分片计数，O(n²)）
    cur = {'boards': [], 'my_hists': [], 'op_hists': [], 'kos': [], 'moves': [], 'values': [], 'to_plays': []}

    def _emit(name, raw):
        """解析单局并产出样本；返回 (n_samples, skip_increment)。"""
        text = raw.decode('utf-8', 'ignore')
        game = parser.parse_string(text)
        # 只跳过「大于目标尺寸」的棋谱（无法放进小棋盘）。
        # 小于目标的（如 9x9 棋谱喂到 19x19）做居中 pad，见下方 off。
        if game is None or game.board_size > board_size:
            return 0, 1
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
            return 0, 1
        if len(game.moves) < 2:
            return 0, 1
        value = parse_result_to_value(game.result)
        if value is None:
            return 0, 1

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

            cur['boards'].append(board.board.copy())
            cur['my_hists'].append(pad3(my_h))
            cur['op_hists'].append(pad3(op_h))
            cur['kos'].append(ko)
            cur['moves'].append(target)
            cur['values'].append(value)
            cur['to_plays'].append(to_play)

            play_move = -1 if (r, c) == (-1, -1) else (r + off) * board_size + (c + off)
            board.play(play_move)
            history.append(play_move)

        return len(game.moves), 0

    n_games = 0
    skip = 0
    if streaming:
        # 分片落在 tmp_root（默认=输出目录），而非系统 temp（Windows 即 C 盘）
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="sft_build_", dir=tmp_root)
    for s in sources:
        try:
            stream = iter_sgf_bytes(s)
        except Exception as e:  # noqa: BLE001
            print(f"[build] 跳过分片 {s}：{e}")
            continue
        for name, raw in stream:
            if max_games and n_games >= max_games:
                break
            n_samp, sk = _emit(name, raw)
            if sk:
                skip += sk
            else:
                n_games += 1
            # 流式模式：累计够一个 chunk 就 flush，清空当前 chunk 释放内存
            if streaming and len(cur['boards']) >= chunk_size:
                total_flushed += len(cur['boards'])
                tmp_files.append(_flush_chunk(cur, tmp_dir, chunk_idx))
                chunk_idx += 1
                print(f"[build] 已落盘分片 #{chunk_idx}（累计样本 {total_flushed}）", flush=True)
                cur = {'boards': [], 'my_hists': [], 'op_hists': [], 'kos': [], 'moves': [], 'values': [], 'to_plays': []}

    if streaming:
        # flush 残余样本并合并
        if cur['boards']:
            tmp_files.append(_flush_chunk(cur, tmp_dir, chunk_idx))
            chunk_idx += 1
        n_samples = _merge_chunks(tmp_files, out)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
        print(f"[build] 流式合并完成：{chunk_idx} 个分片 -> {out}（{n_samples} 样本）", flush=True)
        return out, n_games, skip

    if n_games == 0:
        raise RuntimeError("未解析到任何有效棋谱，请检查 --src 与 --board-size")

    data = {
        'boards': np.stack(cur['boards']).astype(np.int8),
        'my_hist': np.stack(cur['my_hists']).astype(np.int16),
        'op_hist': np.stack(cur['op_hists']).astype(np.int16),
        'ko': np.array(cur['kos'], dtype=np.int16),
        'moves': np.array(cur['moves'], dtype=np.int16),
        'values': np.array(cur['values'], dtype=np.int8),
        'to_play': np.array(cur['to_plays'], dtype=np.int8),
    }
    return data, n_games, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default='data/sft_dataset.npz')
    ap.add_argument('--board-size', type=int, default=19)
    ap.add_argument('--max-games', type=int, default=0)
    ap.add_argument('--chunk-size', type=int, default=50000,
                    help='每攒够这么多样本就落盘一个临时分片，最后合并成单个 npz；'
                         '设为 0 则退回全量常驻内存模式（峰值内存更高）。')
    ap.add_argument('--tmp-dir', default='',
                    help='流式分片暂存目录（默认：输出文件所在目录）。'
                         '大数据集务必确认该目录所在盘有足够空间——不要落到'
                         '系统 temp（Windows 即 C 盘）。')
    ap.add_argument('--merge', action='store_true',
                    help='合并模式：把 --src 目录下已落盘的 chunk_*.npz 分片合并成单个 '
                         '--out（支持断点续做）。用于流式构建中断后，把残留分片合成最终 npz。')
    ap.add_argument('--merge-group', type=int, default=8,
                    help='合并阶段1 每组合并的 chunk 数（控制峰值内存，默认 8）')
    ap.add_argument('--merge-clean', action='store_true',
                    help='合并成功后删除已消费的 chunk/part（默认保留到 _consumed/）')
    args = ap.parse_args()

    if args.merge:
        # 合并模式：把已落盘的分片合成为单个 npz（支持断点续做）
        merge_shards(args.src, args.out, group=args.merge_group, clean=args.merge_clean)
        return

    chunk = args.chunk_size or 0
    out_dir = os.path.dirname(os.path.abspath(args.out)) or '.'
    os.makedirs(out_dir, exist_ok=True)
    # 暂存目录默认跟输出同盘，避免占用系统盘
    tmp_root = args.tmp_dir or out_dir

    if chunk and out_dir:
        # 流式分片落盘：峰值内存仅约一个 chunk
        out_path, n, skip = build(args.src, args.board_size, args.max_games or 0,
                                  chunk_size=chunk, out=args.out,
                                  tmp_root=tmp_root)
        print(f"构建完成(流式): 有效局 {n}, 跳过 {skip}, 保存至 {out_path}")
    else:
        # 全量模式：兼容旧行为
        data, n, skip = build(args.src, args.board_size, args.max_games or 0)
        np.savez_compressed(args.out, **data)
        print(f"构建完成: 有效局 {n}, 跳过 {skip}, 样本 {data['boards'].shape[0]}, 保存至 {args.out}")


if __name__ == '__main__':
    main()
