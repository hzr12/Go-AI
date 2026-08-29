"""SGF 解析器测试（覆盖 pass / 让子 AB/AW / 贴目 KM）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sgf_parser import SGFParser, GameRecord


def test_pass_parsed():
    """pass 必须被解析为 (-1,-1)，且不失序。"""
    p = SGFParser()
    sgf = "(;SZ[9];B[cc];W[];B[ee])"  # 黑下 cc，白 pass，黑下 ee
    g = p.parse_string(sgf)
    mv = [(m.color, m.position) for m in g.moves]
    assert mv == [('B', (2, 2)), ('W', (-1, -1)), ('B', (4, 4))], mv
    print("PASS test_pass_parsed")


def test_ab_aw_handicap():
    """让子 AB/AW 必须按序进入序列。"""
    p = SGFParser()
    sgf = "(;SZ[9];AB[cc][ee];W[gg];B[hh])"
    g = p.parse_string(sgf)
    mv = [(m.color, m.position) for m in g.moves]
    assert mv == [('B', (2, 2)), ('B', (4, 4)), ('W', (6, 6)), ('B', (7, 7))], mv
    print("PASS test_ab_aw_handicap")


def test_komi():
    """KM 贴目应被解析。"""
    p = SGFParser()
    g = p.parse_string("(;SZ[19];KM[7.5];B[cc];W[dd])")
    assert g.komi == 7.5, g.komi
    print("PASS test_komi")


def test_real_game_roundtrip():
    """用 GoBoard 重放一个完整小型对局，不报错即可。"""
    from src.game.go_rules import GoBoard
    p = SGFParser()
    # 9x9 简单对局
    sgf = "(;SZ[9];B[cc];W[gg];B[cd];W[gh];B[dc];W[gd])"
    g = p.parse_string(sgf)
    board = GoBoard(9, komi=g.komi)
    for m in g.moves:
        r, c = m.position
        mv = -1 if (r, c) == (-1, -1) else r * 9 + c
        board.play(mv)
    assert board.board[2, 2] == 1
    assert board.board[6, 6] == -1
    print("PASS test_real_game_roundtrip")


if __name__ == "__main__":
    test_pass_parsed()
    test_ab_aw_handicap()
    test_komi()
    test_real_game_roundtrip()
    print("\nALL SGF TESTS PASSED")
