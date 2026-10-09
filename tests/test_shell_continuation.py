"""shell 续行（行尾反斜杠）的**行为**测试。

为什么静态检查抓不到这个缺陷：
`bash -n` 只做语法检查。行尾写成 `\\`（不在引号内）在语法上完全合法，但它的含义
是「反斜杠转义反斜杠」= 一个**字面反斜杠**，换行**不再续行**。于是命令在该行被
截断，字面 `\\` 成为训练脚本的最后一个 argv，其后
--prefetch-workers / --log-every / --early-stop / --out / --export-onnx / --c2net /
"$@" 等全部丢失，脚本再把余下各行当成独立命令 → `command not found`（exit 127）。

实测这正是 shell/ 下多卡脚本的 `--scaler-growth-interval` 那一行。详见
.superpowers/sdd/2026-09-25-v21-roadmap/task-p1-5-report.md。

为什么 tests/test_run_py_sh.py 的 test_shell_script_args_are_parseable 也没抓到：
它确实把 .sh 的参数抽出来真送进了 argparse，但总是追加 `--help`。argparse 在
parse_known_args 阶段遇到 `--help` 就打印帮助并 exit(0)，因此那个多余的 `\\`
位置参数**永远**不会被报成 "unrecognized arguments"（已实测：加上 `\\` 仍然
exit 0、stderr 无 unrecognized）。

所以这里改成真跑 bash：用 stub 顶替 torchrun，把脚本里那段**真实字节**原样接在
stub 之后执行，然后检查 stub 实际收到的 argv。

探针不执行真实训练：stub 必须在被 source 的真实字节**之前**定义，torchrun 与训练脚本
一个都不会被启动。抽取的字节里那些 $DATA/$BATCH 等变量未定义（探针刻意
不带上半部分），bash 默认展开为空串——本测试只关心 argv 的**结构与完整性**，
不校验取值。
"""
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHELL_DIR = os.path.join(ROOT, 'shell')

SENTINEL = '--sentinel-d7-probe'

# 现存的训练脚本（2026-10-09 NPU 移除后只剩 A100 单卡这一支）。
TARGETS = ('train_sft_a100_1card.sh',)

# 续行一旦断掉就会丢掉的参数：全在出问题那一行之后。
REQUIRED_FLAGS = (
    '--prefetch-workers', '--prefetch-depth', '--log-every', '--swanlab-every',
    '--eval-every', '--save-every', '--early-stop', '--early-stop-patience',
    '--out', '--export-onnx', '--ver', '--c2net',
)

# 顶替 torchrun，只把收到的 argv 原样吐出来。用 NUL 分隔而非换行/尖括号，
# 这样任何 argv 元素（含空串、含空格）都不可能与分隔符混淆。
STUB = 'torchrun() {\n  printf \'%s\\0\' "$@"\n}\n'


def _existing_sh():
    if not os.path.isdir(SHELL_DIR):
        return []
    return sorted(f for f in os.listdir(SHELL_DIR) if f.endswith('.sh'))


def _command_source(script_name):
    """抽出该脚本里训练命令的真实字节：从 torchrun 行到含 "$@" 的行。

    刻意不用写死的行号区间（brief 里的 65,88）：行号会随注释增删漂移。
    """
    path = os.path.join(SHELL_DIR, script_name)
    lines = open(path, encoding='utf-8').read().splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.lstrip().startswith('torchrun ')), None)
    assert start is not None, f'{script_name} 未找到 torchrun 命令'
    # "$@" 在文件开头的用法注释里也出现过，故从 start 往后找
    end = next(i for i, l in enumerate(lines[start:], start)
               if '"$@"' in l)
    return '\n'.join(lines[start:end + 1]) + '\n'


def _run_probe(script_name, tmp_path):
    """stub + 真实字节 → bash 执行，返回 (argv, stderr, returncode)。"""
    probe = tmp_path / (script_name + '.probe.sh')
    # 必须写原始字节：Path.write_text 在 Windows 上会把 \n 翻译成 \r\n，
    # 而 `\` 与 `\r` 之间不再是续行。本机（Git for Windows 的 MSYS2 bash）读脚本
    # 时会吞掉 CR，所以 CRLF 在这里侥幸不出错；换成不吞 CR 的 bash 就会误报。
    # 显式写 LF 才能让本测试只对「反斜杠个数」敏感，而不对换行符敏感。
    probe.write_bytes((STUB + _command_source(script_name)).encode('utf-8'))
    r = subprocess.run(['bash', str(probe), SENTINEL],
                       capture_output=True, cwd=ROOT)
    parts = r.stdout.split(b'\0')
    if parts and parts[-1] == b'':      # 去掉末尾分隔符，保留中间的空 argv
        parts.pop()
    return [p.decode('utf-8', 'replace') for p in parts], r.stderr, r.returncode


@pytest.mark.skipif(shutil.which('bash') is None, reason='无 bash')
@pytest.mark.parametrize('script', TARGETS)
def test_sft_scripts_deliver_complete_argv(script, tmp_path):
    """训练命令必须把完整 argv 交给 torchrun，且 "$@" 透传到末尾。"""
    argv, stderr, rc = _run_probe(script, tmp_path)

    assert '\\' not in argv, (
        f'{script} 的 argv 末位是字面反斜杠 —— 行尾写成了 \\\\ 而不是 \\，'
        f'续行在该行断开。argv 尾部: {argv[-6:]}；stderr: '
        f'{stderr.decode("utf-8", "replace").strip()}')

    missing = [f for f in REQUIRED_FLAGS if f not in argv]
    assert not missing, (
        f'{script} 的 argv 缺少 {missing}，说明续行在中途断开。'
        f'实收 {len(argv)} 个参数，尾部: {argv[-6:]}；stderr: '
        f'{stderr.decode("utf-8", "replace").strip()}')

    assert SENTINEL in argv, \
        f'{script} 未透传 "$@"（哨兵 {SENTINEL} 没进 argv）'
    assert argv[-1] == SENTINEL, \
        f'{script} 的 "$@" 不在末尾，实收尾部: {argv[-4:]}'

    assert stderr.strip() == b'', (
        f'{script} 执行有报错输出（应为 0 退出、无 command not found），'
        f'rc={rc}，stderr: {stderr.decode("utf-8", "replace").strip()}')


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_no_double_backslash_line_continuations():
    """shell/*.sh 里不得有行尾两个及以上反斜杠的续行写法。

    行尾 `\\` 不在引号内时表示一个字面反斜杠，续行失效；`bash -n` 查不出来。
    """
    hits = []
    for f in _existing_sh():
        raw = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        for n, line in enumerate(raw.splitlines(), 1):
            if re.search(r'\\{2,}\s*$', line):
                hits.append(f'{f}:{n}: {line.rstrip()}')
    assert not hits, (
        '以下行以两个及以上反斜杠结尾，续行在该行断开（应改为单个 \\）：\n  '
        + '\n  '.join(hits))


@pytest.mark.skipif(shutil.which('bash') is None, reason='无 bash')
@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_bash_syntax_still_valid():
    """每个 .sh 必须仍通过 bash -n（守住「改续行字节没改坏语法」）。"""
    for f in _existing_sh():
        p = os.path.join(SHELL_DIR, f)
        r = subprocess.run(['bash', '-n', p], capture_output=True, text=True)
        assert r.returncode == 0, f'{f} 语法错误: {r.stderr}'
