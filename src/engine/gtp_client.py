r"""KataGo GTP 客户端：给 webui 当 AI 后端。

为什么需要它
------------
V7（22 通道 `NbtTfNet`）**跑不了原生 MCTS 路径**：`go_rules.py::_check_n_channels`
硬性 `12 <= n <= 17`，而 22 通道里 ch14–17 是梯子、ch18 要贴目，`feature_planes`
给不出来。硬放白名单的后果不是报错，而是模型拿到一份它没见过的输入（梯子全空、
贴目恒 0）却**照样出着法、照样不报错**。

而引擎这条路**天然没有这个缺口**：`.bin.gz` 里带结构描述，引擎自己造全部 22 通道，
它有整局历史、有 komi、有 sv3 六路。2026-10-05 已实测打通（analysis
`Model version 17` / `5545737 params`；GTP 19/19、`genmove` 3/3 合法）。

协议要点（都踩过）
----------------
* 响应是**若干行 + 空行终止**，首字符 `=` 成功 / `?` 失败。**只读一行会让后续每条
  命令全部错位** —— 而汇总仍会打「全部成功」，因为错位拿到的也全是 `=` 开头。
* 就绪探测不能靠日志：引擎把启动日志写进 `logDir`，**stdout 上只有响应**。所以
  「发一发最小查询，看有没有响应」才是就绪信号（沿用 `label_sgf.py::KataLabeler`）。
* 引擎起不来的报错要**带 stderr 尾部**，否则只剩一句 `BrokenPipeError`。

这个客户端**故意不做**的事
------------------------
不解析 visits / 胜率 / 候选着法 —— 普通 GTP 没有这些（`kata-analyze` 才有）。
调用方必须能把它们当「未知」处理，**不许拿默认值冒充**。
"""
import os
import subprocess
import threading
import time

#: GTP 列字母跳过 `I`（围棋惯例）。
_COLS = 'ABCDEFGHJKLMNOPQRST'


def vertex_to_str(flat, board_size=19):
    """扁平坐标 → GTP 顶点记号（`305` → `F4`）。pass / 认输都返回 `'pass'`。

    **`flat < 0` 必须一起挡掉**：负坐标经 `divmod` 会得到负的行，于是产出
    `S20` 这种**看着合法、实际越界**的记号（认输的 -1 就会变成它）。而 GTP 的
    `play` 只认真实顶点与 `pass`，发出去的结果不可预期。
    """
    n = board_size * board_size
    if flat is None or int(flat) < 0 or int(flat) >= n:
        return 'pass'
    r, c = divmod(int(flat), board_size)
    return '%s%d' % (_COLS[c], board_size - r)


def str_to_vertex(s, board_size=19):
    """GTP 顶点记号 → 扁平坐标。`pass` / `resign` 返回 pass 槽（`n`）或 -1。"""
    s = (s or '').strip().upper()
    if s in ('PASS', ''):
        return board_size * board_size
    if s == 'RESIGN':
        return -1
    col = _COLS.index(s[0])
    row = board_size - int(s[1:])
    return row * board_size + col


class GTPError(RuntimeError):
    """引擎不可用或命令被拒绝。**必须带引擎自己的话**，否则无法处置。"""


class KataGoGTP:
    """常驻 KataGo GTP 子进程。

    **常驻**而不是每次落子再启一个：引擎加载 22 通道模型要几秒，逐次启动会让每手
    都卡在那里。
    """

    def __init__(self, exe, model, config, *, cwd=None, board_size=19,
                 komi=7.5, timeout=180.0, log=None):
        self.exe = exe
        self.model = model
        self.config = config
        self.cwd = cwd or os.path.dirname(os.path.abspath(exe))
        self.board_size = int(board_size)
        self.komi = float(komi)
        self.timeout = float(timeout)
        self.log = log
        self._lock = threading.Lock()
        self._proc = None
        self._stderr_tail = []

    # ---- 生命周期 ----
    def start(self):
        """启动进程并**探测就绪**；失败抛 `GTPError`（带 stderr 尾部）。"""
        for path, what in ((self.exe, '引擎'), (self.model, '模型'),
                           (self.config, '配置')):
            if not os.path.isfile(path):
                raise GTPError('%s不存在：%s' % (what, path))
        self._proc = subprocess.Popen(
            [self.exe, 'gtp', '-model', self.model, '-config', self.config],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8',
            errors='replace', bufsize=1, cwd=self.cwd)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # 就绪探测 = 发一发最小查询。引擎把启动日志写进 logDir，stdout 只有响应，
        # 所以「有没有响应」才是可靠的就绪信号。
        try:
            self.command('protocol_version')
        except GTPError as e:
            raise GTPError('引擎启动后没有响应：%s\n--- stderr 尾部 ---\n%s'
                           % (e, self.stderr_tail()))
        self.send('clear_board')
        self.send('boardsize %d' % self.board_size)
        self.send('komi %s' % self.komi)
        return self

    def _drain_stderr(self):
        try:
            for ln in self._proc.stderr:
                self._stderr_tail.append(ln.rstrip('\n'))
                del self._stderr_tail[:-40]
        except (OSError, ValueError):
            pass

    def stderr_tail(self, n=20):
        return '\n'.join(self._stderr_tail[-n:]) or '(空)'

    def close(self):
        with self._lock:
            p = self._proc
            self._proc = None
        if p is None:
            return
        try:
            p.stdin.write('quit\n')
            p.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
        for s in (p.stdin, p.stdout, p.stderr):
            try:
                s.close()
            except (OSError, ValueError):
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False

    # ---- 协议 ----
    def _read_response(self):
        """读一条完整响应（读到**空行**为止）。返回 `(status, body)`。"""
        lines = []
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            ln = self._proc.stdout.readline()
            if ln == '':
                if self._proc.poll() is not None:
                    raise GTPError('引擎进程已退出（code=%s）\n--- stderr 尾部 ---\n%s'
                                   % (self._proc.returncode, self.stderr_tail()))
                continue
            ln = ln.rstrip('\r\n')
            if ln == '':
                if not lines:            # 响应前的空行，跳过
                    continue
                first_rest = lines[0][1:].strip()
                body = ([first_rest] if first_rest else []) + lines[1:]
                return lines[0][0], '\n'.join(body)
            lines.append(ln)
        raise GTPError('等引擎响应超时（%ds），最后读到：%r' % (self.timeout, lines[-3:]))

    def send(self, cmd):
        """发一条命令，返回响应体（成功时）。失败抛 `GTPError`。"""
        with self._lock:
            if self._proc is None:
                raise GTPError('引擎未启动（或已关闭）')
            try:
                self._proc.stdin.write(cmd + '\n')
                self._proc.stdin.flush()
            except (OSError, ValueError) as e:
                raise GTPError('写引擎失败：%s\n--- stderr 尾部 ---\n%s'
                               % (e, self.stderr_tail())) from e
            status, body = self._read_response()
            if status == '?':
                raise GTPError('引擎拒绝 `%s`：%s' % (cmd, body))
            return body

    def command(self, cmd):
        return self.send(cmd)

    # ---- 对局操作 ----
    def set_position(self, moves, to_move='b'):
        """把引擎的棋盘同步成给定着法序列（`[(color, flat), ...]`）。

        用 `clear_board` + 逐个 `play`，**不用** `set_position`：后者在不同引擎
        版本上语义不一致（有的要坐标串、有的要 JSON），而 `play` 是 GTP 最小公倍数。
        """
        self.send('clear_board')
        for color, flat in moves:
            self.send('play %s %s' % (color, vertex_to_str(flat, self.board_size)))
        self.send('komi %s' % self.komi)

    def genmove(self, color):
        """引擎走子。返回扁平坐标（pass 为 `n`，认输为 -1）。"""
        body = self.send('genmove %s' % color)
        tok = body.split()[0] if body.split() else ''
        if not tok:
            raise GTPError('genmove 返回空响应（引擎可能在搜索中崩溃）')
        return str_to_vertex(tok, self.board_size)

    def undo(self):
        return self.send('undo')

    def final_score(self):
        return self.send('final_score')

    def komi_score(self):
        """`final_score` 的数值形式（用于显示目数）。引擎给 `B+15.5` 这种记号。"""
        txt = self.final_score().strip()
        return float(txt[2:]) if txt[:1] in ('B', 'W') else float('nan')