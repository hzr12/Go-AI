import argparse
import os
import sys
import tarfile
import tempfile
import re
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

def parse_player_rating(player_str):
    """从 SGF 的 PB/PW 字段提取等级评分。"""
    if not player_str:
        return 10
    s = player_str.strip()
    # KGS 评级: "(KGS:9)" 或 "(KGS:15d)"
    m = re.search(r'[Kk][Gg][Ss]:\s*(\d+)([dkDK]?)', s)
    if m:
        rating = int(m.group(1))
        typ = m.group(2).lower()
        return rating * 2 if not typ else (20 + rating * 10)
    # 数字 + d/k 后缀: "9d", "3k", "15k"
    m = re.search(r'(\d+)([dkDK])', s)
    if m:
        rating = int(m.group(1))
        typ = m.group(2).lower()
        if typ == 'k':
            return max(1, 20 - rating * 2) # 业余级数越大越弱
        return 20 + rating * 10 # 职业级数越大越强
    return 10 # 默认中等权重

def compute_game_weight(black_rating, white_rating):
    """根据双方等级计算对局权重（几何平均 + log 压缩）。"""
    avg = (black_rating + white_rating) / 2.0
    return float(np.exp(avg / 20.0)) # exp(10)≈2.2, exp(20)≈4.9, exp(30)≈10.1

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
    修复：把 recent（扁平坐标序列，最近一手在末尾）拆成 my/op 各 3 手。
    确保返回的列表中，索引0是最近的一手。
    """
    my, op = [], []
    cur = -to_play # 最近一手由对手（-to_play）下的
    # 从最近的一手开始遍历
    for mv in reversed(recent):
        if cur == to_play:
            my.append(mv)
        else:
            op.append(mv)
        cur = -cur
    # 此时 my 和 op 是 [旧 -> 新] 的顺序，需要反转为 [新 -> 旧]
    return my[::-1], op[::-1]

def _flush_chunk(chunk, tmp_dir, idx):
    """把一个 chunk（dict of list）stack 后存成临时 npz 分片，释放内存。"""
    path = os.path.join(tmp_dir, f"chunk_{idx:05d}.npz")
    save_dict = {}
    for k, v in chunk.items():
        if k == 'boards':
            save_dict[k] = np.stack(v).astype(np.int8)
        elif k == 'my_hist':
            save_dict[k] = np.stack(v).astype(np.int16)
        elif k == 'op_hist':
            save_dict[k] = np.stack(v).astype(np.int16)
        elif k == 'ko':
            save_dict[k] = np.array(v, dtype=np.int16)
        elif k == 'moves':
            save_dict[k] = np.array(v, dtype=np.int16)
        elif k == 'values':
            save_dict[k] = np.array(v, dtype=np.int8)
        elif k == 'winrates':
            save_dict[k] = np.array(v, dtype=np.float32)
        elif k == 'to_play':
            save_dict[k] = np.array(v, dtype=np.int8)
        elif k == 'game_ids':
            save_dict[k] = np.array(v, dtype=np.int32)
        elif k == 'game_weights':
            save_dict[k] = np.array(v, dtype=np.float32)
    np.savez(path, **save_dict)
    return path

def _merge_chunks(tmp_dir, out):
    """
    优化：直接扫描 tmp_dir 目录下的所有 chunk_*.npz 文件进行合并。
    这避免了在 build 函数中维护一个可能无限增长的 tmp_files 列表。
    """
    import glob as _glob
    keys = ['boards', 'my_hist', 'op_hist', 'ko', 'moves', 'values', 'winrates', 'to_play', 'game_ids', 'game_weights']
    merged = {k: [] for k in keys}
    
    # 直接扫描目录
    chunk_files = sorted(_glob.glob(os.path.join(tmp_dir, 'chunk_*.npz')))
    if not chunk_files:
        raise RuntimeError(f"未在 {tmp_dir} 找到任何可合并的 chunk_*.npz 分片")

    for f in chunk_files:
        d = np.load(f, allow_pickle=False)
        for k in keys:
            if k in d:
                merged[k].append(d[k])
        d.close()
    
    merged = {k: np.concatenate(v, axis=0) for k, v in merged.items() if v}
    np.savez_compressed(out, **merged)
    
    # 合并完成后清理临时分片
    for f in chunk_files:
        try:
            os.remove(f)
        except OSError:
            pass
    return merged['boards'].shape[0]

def merge_shards(src_dir, out, group=8, clean=False):
    """断点合并：把 src_dir 下已落盘的 chunk 分片（*.npz）合并成单个 out。"""
    import glob as _glob
    import shutil as _shutil
    keys = ['boards', 'my_hist', 'op_hist', 'ko', 'moves', 'values', 'winrates', 'to_play', 'game_ids', 'game_weights']
    out_base = os.path.basename(out)
    done_marker = out + '.done'
    if os.path.isfile(done_marker) and os.path.isfile(out):
        print(f"[merge] 已完成（存在 {done_marker}），跳过: {out}")
        return out
    chunks = sorted(
        f for f in _glob.glob(os.path.join(src_dir, '*.npz'))
        if os.path.basename(f) != out_base and not os.path.basename(f).startswith('part_') and not f.endswith('.done')
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
            continue # 断点续做：该组已合并，跳过
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
                _shutil.move(cf, consumed_dir) # 移走，避免重跑重复计数
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
    os.replace(tmp_out, out) # 原子替换，避免半截文件
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
    print(f"[merge] 合并完成: {len(parts)} 个分片 -> {out}（{merged['boards'].shape[0]} 样本）", flush=True)
    return out

def build(src, board_size, max_games, chunk_size=0, out=None, tmp_root=None):
    """构建数据集。"""
    parser = SGFParser()
    streaming = chunk_size and out is not None
    # 数据源解析
    if os.path.isdir(src):
        import glob
        tgzs = (sorted(glob.glob(os.path.join(src, '**', '*.tgz'), recursive=True)) +
                sorted(glob.glob(os.path.join(src, '**', '*.tar.gz'), recursive=True)))
        subdirs = sorted(d for d in glob.glob(os.path.join(src, '*')) if os.path.isdir(d))
        sources = tgzs + subdirs
        if not sources:
            sources = [src]
    else:
        sources = [src]

    # 流式模式：用临时目录攒分片
    tmp_dir = None
    chunk_idx = 0
    total_flushed = 0
    game_id_counter = 0

    def _emit(name, raw):
        """解析单局并产出样本；返回 (n_samples, skip_increment)。"""
        nonlocal game_id_counter
        try:
            text = raw.decode('utf-8', 'ignore')
            game = parser.parse_string(text)
        except Exception as e:
            print(f"解析失败跳过 {name}: {e}")
            return 0, 1

        # 只跳过「大于目标尺寸」的棋谱
        if game is None or game.board_size > board_size:
            return 0, 1
        # 过滤让子棋
        if len(game.moves) >= 2 and game.moves[0].color == game.moves[1].color:
            return 0, 1
        # 小棋盘居中到大棋盘的偏移量
        off = (board_size - game.board_size) // 2 if game.board_size != board_size else 0
        # 校验所有坐标
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
        # SGF 元数据加权
        black_rating = parse_player_rating(game.black_player)
        white_rating = parse_player_rating(game.white_player)
        game_weight = compute_game_weight(black_rating, white_rating)
        board = GoBoard(board_size, komi=game.komi)
        history = []
        n_moves = len(game.moves)
        for i, mv in enumerate(game.moves):
            to_play = board.current_player
            recent = history[-3:] if len(history) >= 3 else history
            my_h, op_h = split_hist(recent, to_play)
            ko = board.ko_point
            r, c = mv.position
            # 落子坐标：pass 记 -1；否则把小棋盘坐标居中映射到大棋盘
            target = -1 if (r, c) == (-1, -1) else (r + off) * board_size + (c + off)
            # position-specific value
            fv = value * to_play
            frac = (i + 1) / n_moves
            alpha = 0.3 + 0.7 * frac
            soft_value = np.tanh(fv * alpha)
            cur['boards'].append(board.board.copy())
            cur['my_hist'].append(pad3(my_h))
            cur['op_hist'].append(pad3(op_h))
            cur['ko'].append(ko)
            cur['moves'].append(target)
            cur['values'].append(round(soft_value))
            cur['winrates'].append(float(soft_value))
            cur['to_play'].append(to_play)
            cur['game_ids'].append(game_id_counter)
            cur['game_weights'].append(game_weight)
            play_move = target
            if not board.play(play_move):
                # 非法走法，跳过整局
                return 0, 1
            history.append(play_move)
        return len(game.moves), 0

    n_games = 0
    skip = 0
    # 初始化 cur
    cur = {
        'boards': [], 'my_hist': [], 'op_hist': [], 'ko': [],
        'moves': [], 'values': [], 'winrates': [], 'to_play': [],
        'game_ids': [], 'game_weights': []
    }
    if streaming:
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="sft_build_", dir=tmp_root)
        for s in sources:
            try:
                stream = iter_sgf_bytes(s)
            except Exception as e:
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
                    game_id_counter += 1
                # 流式模式：累计够一个 chunk 就 flush
                if streaming and len(cur['boards']) >= chunk_size:
                    total_flushed += len(cur['boards'])
                    _flush_chunk(cur, tmp_dir, chunk_idx)
                    chunk_idx += 1
                    cur = {k: [] for k in cur}
                    print(f"[build] 已落盘分片 #{chunk_idx}（累计样本 {total_flushed}）", flush=True)
        if streaming:
            # flush 残余样本并合并
            if cur['boards']:
                _flush_chunk(cur, tmp_dir, chunk_idx)
                chunk_idx += 1
            n_samples = _merge_chunks(tmp_dir, out)
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
        'my_hist': np.stack(cur['my_hist']).astype(np.int16),
        'op_hist': np.stack(cur['op_hist']).astype(np.int16),
        'ko': np.array(cur['ko'], dtype=np.int16),
        'moves': np.array(cur['moves'], dtype=np.int16),
        'values': np.array(cur['values'], dtype=np.int8),
        'winrates': np.array(cur['winrates'], dtype=np.float32),
        'to_play': np.array(cur['to_play'], dtype=np.int8),
        'game_ids': np.array(cur['game_ids'], dtype=np.int32),
        'game_weights': np.array(cur['game_weights'], dtype=np.float32),
    }
    return data, n_games, skip

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default='data/sft_dataset.npz')
    ap.add_argument('--board-size', type=int, default=19)
    ap.add_argument('--max-games', type=int, default=0)
    ap.add_argument('--chunk-size', type=int, default=50000, help='每攒够这么多样本就落盘一个临时分片，最后合并成单个 npz；'
                    '设为 0 则退回全量常驻内存模式（峰值内存更高）。')
    ap.add_argument('--tmp-dir', default='', help='流式分片暂存目录（默认：输出文件所在目录）。'
                    '大数据集务必确认该目录所在盘有足够空间——不要落到'
                    '系统 temp（Windows 即 C 盘）。')
    ap.add_argument('--merge', action='store_true', help='合并模式：把 --src 目录下已落盘的 chunk_*.npz 分片合并成单个 '
                    '--out（支持断点续做）。用于流式构建中断后，把残留分片合成最终 npz。')
    ap.add_argument('--merge-group', type=int, default=8, help='合并阶段1 每组合并的 chunk 数（控制峰值内存，默认 8）')
    ap.add_argument('--merge-clean', action='store_true', help='合并成功后删除已消费的 chunk/part（默认保留到 _consumed/）')
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
        out_path, n, skip = build(args.src, args.board_size, args.max_games or 0, chunk_size=chunk, out=args.out, tmp_root=tmp_root)
        print(f"构建完成(流式): 有效局 {n}, 跳过 {skip}, 保存至 {out_path}")
    else:
        # 全量模式：兼容旧行为
        data, n, skip = build(args.src, args.board_size, args.max_games or 0)
        np.savez_compressed(args.out, **data)
        print(f"构建完成: 有效局 {n}, 跳过 {skip}, 样本 {data['boards'].shape[0]}, 保存至 {args.out}")

if __name__ == '__main__':
    main()