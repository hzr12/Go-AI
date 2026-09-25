"""build_dataset.py 全量模式（chunk_size=0）的回归测试。

原缺陷：build() 里遍历 sources 的循环整个包在 `if streaming:` 内部，而
streaming = chunk_size and out is not None。main() 的全量分支不传
chunk_size（默认 0），因此一个 source 都不会被遍历，随后抛出**误导性**的

    RuntimeError: 未解析到任何有效棋谱，请检查 --src 与 --board-size

——实际与 --src / --board-size 无关，纯粹是这个分支根本没执行。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_dataset import build  # noqa: E402

# 9 路盘的合法坐标只有 a..i。SGF 字母按 a=0 编码，与盘面大小无关，
# 所以在 SZ[9] 里写 W[pp] 会被解析成 (15,15) 并因越界被 build_dataset
# 整局丢弃（build_dataset.py:273）——测试数据必须用盘内坐标。
SGF_9x9 = (
    b"(;FF[4]GM[1]SZ[9]PB[me]PW[you]RE[B+]"
    b";B[dd];W[ee];B[dc];W[ec];B[cc];W[ce];B[ed];W[de];B[cd])"
)
SGF_19x19 = (
    b"(;FF[4]GM[1]SZ[19]PB[me]PW[you]RE[B+]"
    b";B[pp];W[dd];B[dp];W[pq];B[qq];W[pd];W[fq];B[nc];W[nq];B[qq])"
)


def _write_sgf(tmp_path, raw, name='g.sgf'):
    p = os.path.join(str(tmp_path), name)
    with open(p, 'wb') as f:
        f.write(raw)
    return p


def test_full_mode_processes_sources(tmp_path):
    """全量模式（chunk_size=0）必须真的遍历 sources 并返回数据 dict。"""
    _write_sgf(tmp_path, SGF_9x9, 'a.sgf')
    _write_sgf(tmp_path, SGF_9x9, 'b.sgf')
    data, n_games, skip = build(str(tmp_path), 9, 0, chunk_size=0)
    assert n_games > 0, \
        '全量模式未解析出任何棋谱——这正是原缺陷（遍历循环错放在 if streaming 内）'
    assert data['boards'].shape[0] > 0
    assert isinstance(data, dict), '全量模式应返回 dict（供 np.savez_compressed）'


def test_full_mode_error_message_is_not_misleading(tmp_path):
    """空目录下仍应报「未解析到棋谱」，但那必须是真的空，而非分支没跑。"""
    empty = os.path.join(str(tmp_path), 'empty')
    os.makedirs(empty, exist_ok=True)
    try:
        build(empty, 19, 0, chunk_size=0)
        raise AssertionError('空目录应当抛 RuntimeError')
    except RuntimeError as e:
        assert '未解析到任何有效棋谱' in str(e)


def test_full_and_streaming_agree(tmp_path, ):
    """全量与流式模式对同一输入应产出同样数量的样本。"""
    for i in range(3):
        _write_sgf(tmp_path, SGF_9x9, f'g{i}.sgf')
    data_full, n_full, _ = build(str(tmp_path), 9, 0, chunk_size=0)
    out = os.path.join(str(tmp_path), 'streamed.npz')
    _, n_stream, _ = build(str(tmp_path), 9, 0, chunk_size=50000, out=out)
    assert n_full == n_stream, \
        f'两种模式有效局数不一致：全量 {n_full} vs 流式 {n_stream}'
    assert data_full['boards'].shape[0] > 0
    d = np.load(out, allow_pickle=False)
    assert d['boards'].shape[0] == data_full['boards'].shape[0], \
        '两种模式产出的样本数应一致'


def test_board_size_mismatch_still_skipped(tmp_path):
    """棋盘尺寸不匹配仍应被跳过（这是预期行为，不是缺陷）。"""
    _write_sgf(tmp_path, SGF_9x9, 'ok.sgf')
    _write_sgf(tmp_path, SGF_19x19, 'big.sgf')
    _, n_games, skip = build(str(tmp_path), 9, 0, chunk_size=0)
    assert n_games >= 1, '匹配的棋谱应被接受'
    assert skip >= 1, '9 路请求下 19 路棋谱应被跳过'
