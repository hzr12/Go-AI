r"""`src/engine/gtp_client.py` 的契约（用假引擎，不依赖真 katago 二进制）。

为什么用假引擎
--------------
真引擎加载 22 通道模型要几秒、还要 OpenCL 设备 —— 不能进单元测试。真实引擎那条
链路由 `tmp/coding/smoke_katago_gtp.py` 单独验（2026-10-05：19/19、genmove 3/3）。
这里钉的是**协议实现**，尤其是那条最容易写错、且错了还会给出全绿假象的规则。
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.engine.gtp_client import (  # noqa: E402
    GTPError, KataGoGTP, str_to_vertex, vertex_to_str)

#: 所有假引擎都给短超时：stdout 耗尽而进程没退出时，实现会转到「进程已退出」
#: 分支；但万一那条分支有 bug，180 秒的默认超时会把这一个测试挂死。
_FAST = {'timeout': 2.0}


class _ListReader:
    """按预置行返回；行用尽后返回 `''`（与真实 pipe 耗尽的表现一致）。"""

    def __init__(self, lines=()):
        self._q = list(lines)

    def feed(self, lines):
        self._q.extend(lines)

    def readline(self):
        return self._q.pop(0) if self._q else ''

    def close(self):
        pass


class _FakeProc:
    """最小子进程替身：按命令给出预置响应，记录收到的命令。"""

    @property
    def stdin(self):
        return self

    def __init__(self, script=None, lines=(), silent=False):
        self.silent = silent
        self.script = dict(script or {})
        self.received = []
        self.returncode = None
        self.stdout = _ListReader(lines)
        self.stderr = _ListReader()

    # webui 只把它当 stdin 用
    def write(self, s):
        cmd = s.strip()
        self.received.append(cmd)
        if self.silent:
            return len(s)          # 不应答：用于测「进程退出 / 超时」两条路径
        resp = self.script.get(cmd, '=')
        body = resp if isinstance(resp, str) else '\n'.join(resp)
        self.stdout.feed([body + '\n', '\n'])
        return len(s)

    def flush(self):
        pass

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass

    def close(self):
        pass


def _gtp(script=None, lines=(), **kw):
    silent = kw.pop('silent', False)
    gtp = KataGoGTP(exe=__file__, model=__file__, config=__file__, **_FAST, **kw)
    gtp._proc = _FakeProc(script, lines, silent)
    return gtp


# --------------------------------------------------------------------------- #
# 顶点记号
# --------------------------------------------------------------------------- #
def test_vertex_roundtrip_covers_all_cells():
    for flat in range(19 * 19):
        s = vertex_to_str(flat, 19)
        assert str_to_vertex(s, 19) == flat, '%d -> %s 往返失败' % (flat, s)


def test_vertex_skips_the_letter_i():
    """GTP 列字母**跳过 I**（围棋惯例）—— 用了 I 就与所有引擎错位一列。"""
    seen = {vertex_to_str(c, 19)[0] for c in range(19)}
    assert 'I' not in seen
    assert vertex_to_str(0, 19).startswith('A')


def test_vertex_row_order_is_flipped():
    """第 0 行是**上边**（`A19`），不是下边 —— 写反会满盘下在对角另一侧。"""
    assert vertex_to_str(0, 19) == 'A19'
    assert vertex_to_str(18 * 19 + 0, 19) == 'A1'
    assert vertex_to_str(3, 19) == 'D19'   # r=0 -> 第 19 行


def test_pass_and_resign_map_to_distinct_slots():
    n = 19 * 19          # pass 槽 = 扁平下标 361，不是边长 19
    assert vertex_to_str(n, 19) == 'pass'
    assert str_to_vertex('pass', 19) == 361
    assert str_to_vertex('resign', 19) == -1
    assert vertex_to_str(-1, 19) == 'pass'


# --------------------------------------------------------------------------- #
# 协议：多行 + 空行终止
# --------------------------------------------------------------------------- #
def test_multiline_response_is_read_until_the_blank_line():
    """多行响应必须读到**空行**才停 —— 否则后续每条命令都错位。

    错位是**静默**的：汇总仍会打「全部成功」，因为错位拿到的也全是 `=` 开头。
    实测踩过：`version` 读到 `list_commands` 的内容，而 `genmove` 一次都没真跑。
    """
    gtp = _gtp(lines=['= protocol_version', 'name', 'version', '\n',
                      '= quit', '\n'])
    assert gtp._read_response() == ('=', 'protocol_version\nname\nversion')
    assert gtp._read_response() == ('=', 'quit')


def test_single_line_response_loses_its_status_char():
    """单行响应 `= 2` 的**内容是 `2`**，状态位必须剥掉。

    没剥的话调用方拿到的 token 是 `=`，顶点合法性判据会把 `=` 当着点。
    """
    gtp = _gtp(lines=['= 2', '\n', '\n'])
    status, body = gtp._read_response()
    assert status == '='
    assert body == '2'


def test_blank_line_before_response_is_skipped():
    """引擎启动 banner 之后常有多余空行，不该被当成一条空响应。"""
    gtp = _gtp(lines=['\n', '\n', '= 2', '\n', '\n'])
    assert gtp._read_response() == ('=', '2')


def test_error_response_raises_with_the_engine_wording():
    gtp = _gtp(lines=['? unknown command', '\n', '\n'])
    with pytest.raises(GTPError, match='unknown command'):
        gtp.send('place_free b K10')


def test_process_exit_is_reported_with_stderr_tail():
    """进程退出要报出来并**带上 stderr 尾部** —— 否则只剩一句 BrokenPipe。"""
    gtp = _gtp(silent=True)
    gtp._stderr_tail = ['Error creating directory', 'boom']
    gtp._proc.returncode = 1
    with pytest.raises(GTPError) as ei:
        gtp.send('genmove b')
    assert 'Error creating directory' in str(ei.value)
    assert 'code=1' in str(ei.value)


def test_timeout_does_not_hang():
    """响应永不到来时必须按超时退出，而不是死等。"""
    gtp = _gtp(silent=True)          # 不应答 => 只能靠超时退出
    with pytest.raises(GTPError):
        gtp.send('genmove b')


# --------------------------------------------------------------------------- #
# 对局操作
# --------------------------------------------------------------------------- #
def test_genmove_returns_a_flat_coordinate():
    gtp = _gtp({'genmove w': '= D4'})
    assert gtp.genmove('w') == str_to_vertex('D4', 19)


def test_genmove_pass_maps_to_the_pass_slot():
    gtp = _gtp({'genmove b': '= pass'})
    assert gtp.genmove('b') == 19 * 19


def test_genmove_empty_response_raises():
    """引擎搜索中崩溃时可能回空响应 —— 必须报错，不能当成 pass 静默走下去。"""
    gtp = _gtp({'genmove b': '='})
    with pytest.raises(GTPError, match='空响应'):
        gtp.genmove('b')


def test_set_position_replays_from_a_clean_board():
    """`set_position`（同步对局）必须先 `clear_board` 再逐手 `play`。

    不用引擎的 `set_position` 命令：它在不同版本上语义不一致（有的要坐标串、
    有的要 JSON），而 `play` 是 GTP 的最小公倍数。
    """
    gtp = _gtp()
    gtp.set_position([('b', str_to_vertex('D4', 19)),
                      ('w', str_to_vertex('Q16', 19))], to_move='b')
    assert gtp._proc.received == ['clear_board', 'play b D4', 'play w Q16',
                                 'komi 7.5']


def test_start_refuses_a_missing_model():
    gtp = KataGoGTP(exe=__file__,
                    model=os.path.join(ROOT, '不存在的模型.bin.gz'),
                    config=__file__, **_FAST)
    with pytest.raises(GTPError, match='模型不存在'):
        gtp.start()


def test_close_is_safe_to_call_twice():
    gtp = _gtp()
    gtp.close()
    gtp.close()          # 不得抛


# --------------------------------------------------------------------------- #
# 明确不做的事
# --------------------------------------------------------------------------- #
def test_client_does_not_fabricate_visits_or_winrate():
    """**不许**编造 visits / 胜率 / 候选着法。

    普通 GTP 没有这些（`kata-analyze` 才有）。所以本模块刻意不提供 —— 一旦提供了
    并带默认值，调用方就会在 UI 上展示看似正常的假数字，而那正是这个仓一直在
    消灭的那类静默错误。
    """
    for attr in ('visits', 'winrate', 'win_rate', 'analyze', 'kata_analyze',
                 'candidate_moves', 'root_value', 'search_info'):
        assert not hasattr(KataGoGTP, attr), \
            '客户端不应提供 %s（普通 GTP 拿不到，编出来就是假数据）' % attr