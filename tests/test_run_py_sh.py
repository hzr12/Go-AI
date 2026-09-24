"""run.py 的 --sh 派发器与 shell/*.sh 脚本测试。

云端平台的三条硬约束：
  1. 只能传 `--参数名 参数值`（不认裸布尔开关、不认位置参数）
  2. 不能设置环境变量
  3. 启动文件必须是 .py

因此多卡训练不能直接用 `torchrun run.py ...`，但可以：
    python run.py --sh shell/train_npu_2card.sh --epochs 3
入口是 .py（run.py），参数是 --名 值（--sh <路径>），bash 由 run.py 内部拉起。

本测试覆盖派发语义、退出码透传、错误提示、参数保序，以及 shell 脚本的
工程不变量（LF 换行、set -euo pipefail、以 "$@" 结尾以支持覆盖）。
"""
import inspect
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run as runpy_mod

SHELL_DIR = os.path.join(ROOT, 'shell')


def _existing_sh():
    if not os.path.isdir(SHELL_DIR):
        return []
    return sorted(f for f in os.listdir(SHELL_DIR) if f.endswith('.sh'))


# --------------------------------------------------------------------------- #
# 派发语义
# --------------------------------------------------------------------------- #
def test_run_sh_dispatches_to_bash_with_passthrough(monkeypatch, tmp_path):
    """派发 bash <脚本> <透传参数>，且 cwd 为仓库根。"""
    script = tmp_path / 'fake.sh'
    script.write_text('#!/usr/bin/env bash\ntrue\n', encoding='utf-8')

    seen = {}

    class _R:
        returncode = 0

    def fake_run(cmd, **kw):
        seen['cmd'] = cmd
        seen['kw'] = kw
        return _R()

    monkeypatch.setattr(runpy_mod.subprocess, 'run', fake_run, raising=False)
    with pytest.raises(SystemExit) as e:
        runpy_mod.run_sh([str(script), '--epochs', '3', '--lr', '0.1'])
    assert e.value.code == 0
    exe = os.path.basename(seen['cmd'][0]).lower()
    assert exe.removesuffix('.exe') in ('bash', 'sh'), \
        f"应派发给 bash/sh，实际 {seen['cmd'][0]}"
    assert os.path.normpath(seen['cmd'][1]) == os.path.normpath(str(script))
    assert seen['cmd'][2:] == ['--epochs', '3', '--lr', '0.1'], \
        f"透传参数未保序: {seen['cmd'][2:]}"
    assert os.path.samefile(seen['kw']['cwd'], ROOT), 'cwd 应为仓库根'


def test_run_sh_resolves_relative_path_against_root(monkeypatch):
    """相对路径按仓库根解析（而非当前工作目录），保证任意 cwd 都能跑。"""
    seen = {}

    class _R:
        returncode = 0

    monkeypatch.setattr(runpy_mod.subprocess, 'run',
                        lambda cmd, **kw: (seen.update(cmd=cmd, kw=kw), _R())[1],
                        raising=False)
    monkeypatch.chdir(os.path.dirname(ROOT))   # 故意切到仓库外
    with pytest.raises(SystemExit):
        runpy_mod.run_sh(['shell/train_sft_npu_2card.sh'])
    assert os.path.isfile(seen['cmd'][1]), \
        f"相对路径应解析到仓库内: {seen['cmd'][1]}"


def test_run_sh_does_not_crash_on_cross_drive_absolute_path(monkeypatch, tmp_path):
    """传入另一个盘符的绝对路径时不得因 relpath 抛 ValueError（Windows 回归）。"""
    script = tmp_path / 'other_drive.sh'
    script.write_text('true\n', encoding='utf-8')

    class _R:
        returncode = 0

    monkeypatch.setattr(runpy_mod.subprocess, 'run', lambda cmd, **kw: _R(),
                        raising=False)
    with pytest.raises(SystemExit) as e:
        runpy_mod.run_sh([str(script)])   # 不得抛 ValueError
    assert e.value.code == 0


def test_run_sh_propagates_exit_code(monkeypatch, tmp_path):
    """非零退出码原样转成 SystemExit（供 torchrun/CI 感知失败）。"""
    script = tmp_path / 'x.sh'
    script.write_text('true\n', encoding='utf-8')

    class _R:
        returncode = 3

    monkeypatch.setattr(runpy_mod.subprocess, 'run',
                        lambda cmd, **kw: _R(), raising=False)
    with pytest.raises(SystemExit) as e:
        runpy_mod.run_sh([str(script)])
    assert e.value.code == 3


def test_run_sh_missing_script_lists_available(monkeypatch, capsys):
    """脚本不存在时给出可读错误，并列出可用脚本。"""
    monkeypatch.setattr(runpy_mod.shutil, 'which', lambda n: '/bin/bash',
                        raising=False)
    with pytest.raises(SystemExit) as e:
        runpy_mod.run_sh(['shell/definitely_not_here.sh'])
    assert e.value.code != 0
    out = capsys.readouterr().out
    assert 'definitely_not_here' in out
    if _existing_sh():
        for f in _existing_sh():
            assert f in out, f"未列出可用脚本 {f}"


def test_run_sh_no_args_shows_usage(capsys):
    """不带脚本路径时打印用法而不是抛栈。"""
    with pytest.raises(SystemExit):
        runpy_mod.run_sh([])
    out = capsys.readouterr().out
    assert '--sh' in out


def test_run_sh_requires_bash(monkeypatch, tmp_path, capsys):
    """找不到 bash 时给出清晰提示（本地 Windows 开发可能没有）。"""
    script = tmp_path / 'x.sh'
    script.write_text('true\n', encoding='utf-8')
    monkeypatch.setattr(runpy_mod.shutil, 'which', lambda n: None,
                        raising=False)
    with pytest.raises(SystemExit) as e:
        runpy_mod.run_sh([str(script)])
    assert e.value.code != 0
    assert 'bash' in capsys.readouterr().out


def test_main_accepts_sh_flag_and_equals_form():
    """main() 需在子命令检查前拦截 --sh，且支持 --sh=<路径> 等号写法。"""
    src = inspect.getsource(runpy_mod.main)
    i_sh = src.find("'--sh'")
    i_cmd = src.find('if cmd not in commands')
    assert i_sh != -1, 'main() 未拦截 --sh'
    assert i_cmd != -1, 'main() 缺少子命令校验'
    assert i_sh < i_cmd, '--sh 必须在子命令校验之前拦截（否则会落进"未知命令"）'
    assert 'startswith' in src and '--sh=' in src, \
        'main() 未支持 --sh=<路径> 等号写法'


def test_existing_subcommands_preserved():
    """sft / selfplay / sp 三个既有子命令不受影响。"""
    assert set(runpy_mod.COMMANDS) >= {'sft', 'selfplay', 'sp'}


# --------------------------------------------------------------------------- #
# 工程不变量
# --------------------------------------------------------------------------- #
def test_gitattributes_forces_lf_for_sh():
    """必须有 .gitattributes 强制 *.sh 为 LF。

    当前 core.autocrlf=true 且无此文件时，Windows checkout 会把 .sh 转成 CRLF，
    Linux 上 bash 报 $'\\r': command not found。
    """
    p = os.path.join(ROOT, '.gitattributes')
    assert os.path.isfile(p), '缺少 .gitattributes（*.sh 会被 checkout 成 CRLF）'
    content = open(p, encoding='utf-8').read()
    assert re.search(r'\*\.sh\s+text\s+eol=lf', content), \
        '.gitattributes 未声明 *.sh text eol=lf'


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_shell_scripts_exist_expected_five():
    """应有 5 个训练脚本：A100 1卡 / 910A 1·2·4卡 / RL 910A 1卡。"""
    got = _existing_sh()
    for expect in ('train_sft_a100_1card.sh', 'train_sft_npu_1card.sh',
                   'train_sft_npu_2card.sh', 'train_sft_npu_4card.sh',
                   'train_rl_npu_1card.sh'):
        assert expect in got, f"缺少脚本 {expect}（现有: {got}）"


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_shell_scripts_are_lf_and_end_with_newline():
    """所有 .sh 必须是 LF 且以换行结尾（防 CRLF 回潮）。"""
    for f in _existing_sh():
        raw = open(os.path.join(SHELL_DIR, f), 'rb').read()
        assert b'\r' not in raw, f"{f} 含 CR（CRLF），Linux bash 会报错"
        assert raw.endswith(b'\n'), f"{f} 未以换行结尾"


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_shell_scripts_have_shebang_and_strict_mode():
    """每个 .sh 需有 shebang 与 set -euo pipefail（fail fast）。"""
    for f in _existing_sh():
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        assert txt.startswith('#!/usr/bin/env bash'), f"{f} 缺少 shebang"
        assert 'set -euo pipefail' in txt, f"{f} 缺少 set -euo pipefail"


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_shell_scripts_end_command_with_dollar_at():
    """训练命令必须以 "$@" 结尾，这样外部传入的参数能覆盖默认值。"""
    for f in _existing_sh():
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        # 取最后一条非注释、非空行，应是带 "$@" 的续行
        lines = [l for l in txt.splitlines() if l.strip() and not l.strip().startswith('#')]
        assert lines, f"{f} 内容为空"
        assert '"$@"' in lines[-1], \
            f"{f} 最后一行应为含 \"$@\" 的续行，实际: {lines[-1]!r}"


@pytest.mark.skipif(shutil.which('bash') is None, reason='无 bash')
@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_shell_scripts_pass_bash_syntax_check():
    """每个 .sh 必须通过 bash -n 语法检查。"""
    for f in _existing_sh():
        p = os.path.join(SHELL_DIR, f)
        r = subprocess.run(['bash', '-n', p], capture_output=True, text=True)
        assert r.returncode == 0, f"{f} 语法错误: {r.stderr}"


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_sft_scripts_use_swanlab_every_10():
    """4 个 SFT 脚本都应带 --swanlab-every 10（曲线加密，stdout 不刷屏）。"""
    for f in _existing_sh():
        if not f.startswith('train_sft'):
            continue
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        assert '--swanlab-every 10' in txt, f"{f} 缺少 --swanlab-every 10"


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_npu_scripts_omit_compile_and_flash_attn():
    """NPU 脚本不应传 --compile / --flash-attn（NPU 上自动禁用，只出警告）。"""
    for f in _existing_sh():
        if 'a100' in f or f.startswith('train_rl'):
            continue
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        cmd = '\n'.join(l for l in txt.splitlines()
                        if not l.strip().startswith('#'))
        assert '--compile' not in cmd, f"{f} 不应传 --compile"
        assert '--flash-attn' not in cmd, f"{f} 不应传 --flash-attn"


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_rl_script_omits_onnx_model():
    """RL 脚本不得带 --onnx-model（当前实现会 torch.load protobuf 而崩溃）。"""
    f = 'train_rl_npu_1card.sh'
    if f not in _existing_sh():
        pytest.skip('RL 脚本尚未创建')
    txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
    cmd = '\n'.join(l for l in txt.splitlines()
                    if not l.strip().startswith('#'))
    assert '--onnx-model' not in cmd, 'RL 脚本带 --onnx-model 会启动即崩'
    assert '--td 1' in cmd, 'RL 脚本应显式带 --td 1'


# --------------------------------------------------------------------------- #
# SwanLab 依赖自动安装
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_shell_scripts_install_swanlab():
    """每个 .sh 在训练前安装 swanlab（云端环境不保证已装）。"""
    for f in _existing_sh():
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        assert re.search(r'-m\s+pip\s+install\s+swanlab', txt), \
            f"{f} 缺少 `python -m pip install swanlab`"


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_swanlab_install_failure_does_not_abort_training():
    """安装失败不得中断训练（脚本有 set -euo pipefail，需显式兜底）。

    云端无外网时 pip 会失败；swanlab 只是可选的指标上报，训练必须照常进行。
    """
    for f in _existing_sh():
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        for line in txt.splitlines():
            if re.search(r'-m\s+pip\s+install\s+swanlab', line) \
                    and not line.strip().startswith('#'):
                assert '||' in line, \
                    f"{f} 的 swanlab 安装未兜底：网络失败会因 set -e 中断整个训练"
                assert 'python' in line and '-m' in line, \
                    f"{f} 应用 python -m pip（保证与训练同一解释器）"
                break
        else:
            pytest.fail(f"{f} 未找到 swanlab 安装命令")


# --------------------------------------------------------------------------- #
# 学习率标定：统一按平方根缩放律，从「有效 batch」推导
# --------------------------------------------------------------------------- #
def _sft_env(txt):
    """从 SFT 脚本抽出 (WORLD_SIZE, BATCH, LR) 三个变量。"""
    env = {}
    for m in re.finditer(r'^(WORLD_SIZE|BATCH|LR)=([0-9.]+)', txt, re.M):
        env[m.group(1)] = m.group(2)
    return env


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_sft_scripts_lr_follows_sqrt_scaling_of_effective_batch():
    """每个 SFT 脚本的 LR 必须等于 0.00356×√(有效batch/2500)。

    有效 batch = BATCH × WORLD_SIZE。改变卡数或每卡 batch 时 LR 必须同步调整，
    否则等效学习率漂移（多卡尤其明显：2 卡有效 6400、4 卡有效 12800）。
    """
    import math
    sft = [f for f in _existing_sh() if f.startswith('train_sft')]
    assert sft, '未找到 SFT 脚本'
    for f in sft:
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        env = _sft_env(txt)
        assert set(env) == {'WORLD_SIZE', 'BATCH', 'LR'}, \
            f"{f} 缺少 WORLD_SIZE/BATCH/LR 变量定义（实得 {sorted(env)}）"
        ws, batch, lr = int(env['WORLD_SIZE']), int(env['BATCH']), float(env['LR'])
        eff = batch * ws
        expect = 0.00356 * math.sqrt(eff / 2500)
        assert abs(lr - expect) < 5e-5, \
            f"{f} LR={lr} 与有效 batch={eff} 的平方根缩放预期 {expect:.5f} 不符"


# --------------------------------------------------------------------------- #
# C2NET（OpenI 启智平台）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_sft_scripts_enable_c2net():
    """SFT 脚本需带 --c2net（走 OpenI 启智平台的数据/输出对接）。"""
    sft = [f for f in _existing_sh() if f.startswith('train_sft')]
    assert sft, '未找到 SFT 脚本'
    for f in sft:
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        m = re.search(r'--c2net\s+"\$C2NET"', txt)
        assert m, f"{f} 缺少 --c2net \"$C2NET\""


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_sft_scripts_c2net_flag_is_zero_or_one():
    """C2NET 必须是 0/1 开关（train_sft.py 的 --c2net 是 type=int choices=[0,1]）。

    平台只接受 --参数名 参数值，故用变量传递；取值越界会在 argparse 报错。
    """
    sft = [f for f in _existing_sh() if f.startswith('train_sft')]
    for f in sft:
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        m = re.search(r'(?m)^C2NET=(\d+)\s*(?:#.*)?$', txt)
        assert m, f"{f} 缺少 C2NET 变量"
        assert m.group(1) in ('0', '1'), \
            f"{f} C2NET={m.group(1)} 非法（--c2net 只接受 0/1）"


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_multicard_sft_scripts_warn_c2net_prepare_all_ranks():
    """多卡 SFT 脚本需提示 c2net 的 prepare() 会被所有 rank 调用。

    train_sft.py 的 c2net 初始化没有 is_main 守卫，DDP 下每个 rank 都会
    prepare()；若该函数有写盘/建连副作用，多卡并发调用可能互相干扰。
    """
    for f in _existing_sh():
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        if not f.startswith('train_sft'):
            continue
        m = re.search(r'(?m)^WORLD_SIZE=(\d+)', txt)
        if not m or int(m.group(1)) < 2:
            continue
        comments = '\n'.join(l for l in txt.splitlines()
                             if l.strip().startswith('#'))
        assert 'prepare' in comments or 'rank' in comments.lower(), \
            f"{f} 是多卡脚本，需注释说明 c2net prepare() 会被所有 rank 调用"


# --------------------------------------------------------------------------- #
# .sh 里的参数必须真能被对应训练脚本解析
# --------------------------------------------------------------------------- #
def _script_args(txt):
    """抽出 .sh 里训练命令的参数（续行以 \\ 结尾，最后一行含 "$@"）。"""
    body = '\n'.join(l for l in txt.splitlines()
                     if not l.strip().startswith('#'))
    body = body.replace('\\\n', ' ')
    for m in re.finditer(r'(?:torchrun[^\n]*?|python)\s+(?:scripts/)?'
                         r'(train_sft|train_sft_ms|selfplay_train)\.py'
                         r'([^\n]*)', body):
        toks = [t for t in m.group(2).split() if t and t != '"$@"']
        return m.group(1), toks
    return None, []


@pytest.mark.skipif(not _existing_sh(), reason='shell/ 下暂无 .sh')
def test_shell_script_args_are_parseable():
    """每个 .sh 的参数必须被对应训练脚本接受（防手写长命令出现拼写错误）。

    run.txt 的命令由 tests/test_run_txt_sync.py 校验，但 .sh 是独立副本，
    同样需要真实解析一遍。
    """
    target = {
        'train_sft': os.path.join(ROOT, 'scripts', 'train_sft.py'),
        'train_sft_ms': os.path.join(ROOT, 'scripts', 'train_sft_ms.py'),
        'selfplay_train': os.path.join(ROOT, 'scripts', 'selfplay_train.py'),
    }
    env = dict(os.environ, PYTHONUTF8='1')
    checked = 0
    for f in _existing_sh():
        txt = open(os.path.join(SHELL_DIR, f), encoding='utf-8').read()
        name, args = _script_args(txt)
        assert name, f"{f} 未找到训练命令"
        if not args:
            continue
        r = subprocess.run([sys.executable, target[name]] + args + ['--help'],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        assert 'unrecognized arguments' not in r.stderr, \
            f"{f} 含无效参数: {r.stderr.strip().splitlines()[-1]}"
        checked += 1
    assert checked == len(_existing_sh()), \
        f"有脚本未完成参数解析校验：checked={checked}, 脚本数={len(_existing_sh())}"
