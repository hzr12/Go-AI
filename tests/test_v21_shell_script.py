"""`shell/train_sft_npu_4card_v21.sh` 的两条禁令守护。

本文件只管**这一个新脚本**，不碰既有 shell 脚本的守护测试
（`tests/test_run_py_sh.py` / `tests/test_npu_graph_compile.py` 那些照旧跑）。

为什么单独开一个文件，而不是往 `test_run_py_sh.py` 里塞
--------------------------------------------------------
那个文件是 v18/v19/A100 时代的通用 shell 守护，它的显存预算用例
（`test_npu_scripts_fit_memory_budget`）会拿 shell 里的结构 flag 去投影
`search_arch.ANCHOR['cfg']`。D1 之后 v21 的形状写死在 `V21_CFG`、结构 flag
不参与构造，那条路径对 v21 是**假绿**——扩展它属于路线图 C11，不在本任务范围。
这里只钉两件与 C11 无关、且一旦破掉就很难发现的事：

1. **不许出现 DDP 包装假设**。`scripts/train_sft.py` 已经从 DDP 换成 FSDP1
   （`FullyShardedDataParallel` + `sharding=SHARD_GRAD_OP` +
   `use_orig_params=True` + 按块类 auto_wrap + rank0 全量 state_dict 汇聚）。
   `torchrun --nproc_per_node=N` 仍注入同一套 `RANK`/`WORLD_SIZE`/`LOCAL_RANK`，
   所以脚本的 torchrun 行与 v19 逐字相同——**这正是要断言的**：一旦有人以为
   「换 FSDP 就要改启动方式」而去动它，或者反过来照着 v19 的注释继续按 DDP
   的显存直觉调 batch，脚本就开始说谎。
2. **不许出现 argparse 不接受的 flag**。D1 明令 v21 不加新 CLI 参数，
   而手写的长命令最容易被后来人「顺手加一个」；`test_run_py_sh.py` 的
   参数解析用例只验语法存在与否，这里额外做一次**静态**核对（对
   `train_sft.py` 源码里 `add_argument('...')` 的全集），顺带把
   「`--arch v21` 之类的分支选择器」也一并禁掉。
"""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHELL_DIR = os.path.join(ROOT, 'shell')
SELF = 'train_sft_npu_4card_v21.sh'
V19 = 'train_sft_npu_4card_v19.sh'

TRAIN_SFT = os.path.join(ROOT, 'scripts', 'train_sft.py')


def _read(name):
    with open(os.path.join(SHELL_DIR, name), encoding='utf-8') as f:
        return f.read()


def _code_lines(txt):
    """剥掉整行注释。守护断言只看真正会被 bash 执行的行。"""
    return [l for l in txt.splitlines() if not l.strip().startswith('#')]


def _which_bash():
    import shutil
    return shutil.which('bash')


def _torchrun_block(txt):
    """取出 torchrun 命令块（去续行反斜杠），归一化 --out / --ver 的取值。"""
    body = '\n'.join(_code_lines(txt)).replace('\\\n', ' ')
    m = re.search(r'torchrun[^\n]*', body)
    assert m, '未找到 torchrun 命令'
    cmd = m.group(0)
    cmd = re.sub(r'--out \S+', '--out <OUT>', cmd)
    cmd = re.sub(r'--ver \S+', '--ver <VER>', cmd)
    return cmd.strip()


def _train_sft_argv(txt):
    """取出传给 train_sft.py 的那段 argv（**不含** torchrun 自己的 flag）。

    torchrun 自身的 `--nproc_per_node` 不是 train_sft.py 的参数，扫 flag 时
    必须从脚本路径之后开始，否则会误报。写法沿用 test_run_py_sh 的 `_script_args`。
    """
    body = '\n'.join(_code_lines(txt)).replace('\\\n', ' ')
    m = re.search(r'(?:torchrun[^\n]*?|python)\s+(?:scripts/)?'
                  r'(train_sft|train_sft_ms|selfplay_train)\.py([^\n]*)', body)
    assert m, '未找到训练命令'
    return [t for t in m.group(2).split() if t and t != '"$@"']


@pytest.fixture(scope='module')
def txt():
    if not os.path.isfile(os.path.join(SHELL_DIR, SELF)):
        pytest.skip('{} 尚未创建'.format(SELF))
    return _read(SELF)


# --------------------------------------------------------------------------- #
# 禁令一：不许有 DDP 包装假设
# --------------------------------------------------------------------------- #
def test_script_does_not_claim_ddp_wrapping(txt):
    """脚本里不许出现 DDP 的包装类名——它已被 FSDP1 取代。"""
    assert 'DistributedDataParallel' not in txt, \
        '脚本仍在提 DDP 的 DistributedDataParallel：train_sft.py 已换 FSDP1'


def test_script_documents_the_fsdp1_facts(txt):
    """FSDP1 的四个关键事实必须写在脚本里（下一个读的人靠这段判断显存）。

    少任何一条都会误导：只说「用了 FSDP」而不说「激活不分片」，读的人会以为
    每卡显存随卡数下降而线性下降，从而去调大 batch。
    """
    for token, why in (
            ('FSDP1', '用的是 FSDP1 而不是 DDP'),
            ('FullyShardedDataParallel', '包装类名'),
            ('SHARD_GRAD_OP', '分片策略'),
            ('use_orig_params', '原参数引用是硬需求'),
            ('FullStateDictConfig', '存档必须在 rank0 汇聚成完整权重'),
            ('激活不分片', '每卡显存的大头仍是激活，故不受分片保护'),
    ):
        assert token in txt, '脚本未说明「{}」（{}）'.format(why, token)


def test_torchrun_invocation_is_unchanged_from_v19(txt):
    """torchrun 行与 v19 逐字相同（只允许 --out / --ver / 累积档位不同）。

    换 FSDP1 不改启动方式：`--nproc_per_node=N` 照旧注入
    RANK/WORLD_SIZE/LOCAL_RANK，`train_sft.py` 读的是同一套环境变量契约。
    任何对这一行的改动都必须先有代码侧的契约变更。

    ⚠ 2026-10-01 起的唯一例外：`--gradient-accumulation-steps`。v21 每卡 2000
    实测 OOM（详见 run.txt【v21 显存上限】），降档的唯一手段是「每卡 batch 减半
    + 累积 2」，而有效 batch 与 LR 都保持不变 —— 所以这一格**必须**与 v19 不同。
    比对时把两边的累积值都归一化成 <ACCUM>，其余仍要求逐字相同。
    """
    if not os.path.isfile(os.path.join(SHELL_DIR, V19)):
        pytest.skip('{} 不在，无法比对'.format(V19))

    def _norm(block):
        return re.sub(r'--gradient-accumulation-steps \S+',
                      '--gradient-accumulation-steps <ACCUM>', block)

    ours, v19 = _torchrun_block(txt), _torchrun_block(_read(V19))
    assert _norm(ours) == _norm(v19), \
        'torchrun 调用与 v19 不一致（只允许 --out/--ver/累积档位不同）：\n{}\n vs \n{}'.format(
            ours, v19)
    # 反过来钉住「累积确实被调高」：降档的意义就在这里，悄悄回到 1 等于没降。
    assert '--gradient-accumulation-steps "$GRAD_ACCUM"' in ours, \
        'v21 必须用 $GRAD_ACCUM 变量传累积步数（硬编码 1 就退回 OOM 档了）'


# --------------------------------------------------------------------------- #
# 禁令二：不许出现 argparse 不接受的 flag
# --------------------------------------------------------------------------- #
def _declared_flags():
    """train_sft.py argparse 声明的全集（静态读源码，不起子进程）。"""
    with open(TRAIN_SFT, encoding='utf-8') as f:
        src = f.read()
    return set(re.findall(r"add_argument\(\s*'(--[A-Za-z0-9-]+)'", src))


def test_every_flag_exists_in_train_sft_argparse(txt):
    """脚本传给 train_sft.py 的每个 flag 都必须在它的 argparse 里有定义。

    只扫脚本路径**之后**的 argv：torchrun 自己的 `--nproc_per_node` 不归
    train_sft.py 管，扫进去会误报。
    """
    declared = _declared_flags()
    argv = ' '.join(_train_sft_argv(txt))
    used = set(re.findall(r'(?<![\w-])(--[A-Za-z0-9-]+)', argv))
    unknown = {f for f in used if f not in declared}
    assert not unknown, \
        '脚本传了 argparse 不接受的 flag：{}。D1 规定 v21 不新增 CLI 参数，' \
        '真要加必须先改 train_sft.py 的 argparse。'.format(sorted(unknown))


def test_no_v21_branch_selector_flag(txt):
    """不许出现 `--arch v21` 之类的架构分支选择器（D1：配置写死在 V21_CFG）。"""
    argv = ' '.join(_train_sft_argv(txt))
    assert not re.search(r'--arch', argv), \
        'D1 之下没有 arch 分支：v21 形状由 alphanet.V21_CFG 写死，' \
        '脚本不应再传 --arch'


def test_flags_survive_a_real_parse(txt):
    """端到端再过一遍 argparse（与 test_run_py_sh 的做法一致，双保险）。

    变量不会被展开（`"$BATCH"` 是字面量），所以只断言「没有 unrecognized
    arguments」——这正是拼错 flag 时 argparse 唯一的报错形态。
    """
    toks = _train_sft_argv(txt)
    r = subprocess.run([sys.executable, TRAIN_SFT] + toks + ['--help'],
                       capture_output=True, text=True,
                       env=dict(os.environ, PYTHONUTF8='1'), cwd=ROOT)
    assert 'unrecognized arguments' not in r.stderr, \
        '参数解析失败: {}'.format(r.stderr.strip().splitlines()[-1:])


# --------------------------------------------------------------------------- #
# 脚本自身的工程不变量（防止它被改坏；与 v19 脚本共享同一套约定）
# --------------------------------------------------------------------------- #
def test_script_is_lf_and_ends_with_newline():
    if not os.path.isfile(os.path.join(SHELL_DIR, SELF)):
        pytest.skip('{} 尚未创建'.format(SELF))
    raw = open(os.path.join(SHELL_DIR, SELF), 'rb').read()
    assert b'\r' not in raw, '{} 含 CR（CRLF）'.format(SELF)
    assert raw.endswith(b'\n'), '{} 未以换行结尾'.format(SELF)


def test_script_has_shebang_strict_mode_and_passthrough(txt):
    assert txt.startswith('#!/usr/bin/env bash'), '缺少 shebang'
    assert 'set -euo pipefail' in txt, '缺少 set -euo pipefail'
    tail = [l for l in txt.splitlines() if l.strip() and not l.strip().startswith('#')][-1]
    assert '"$@"' in tail, '最后一行应是含 "$@" 的续行，实际: {!r}'.format(tail)


@pytest.mark.skipif(_which_bash() is None, reason='无 bash')
def test_script_passes_bash_syntax_check():
    if not os.path.isfile(os.path.join(SHELL_DIR, SELF)):
        pytest.skip('{} 尚未创建'.format(SELF))
    r = subprocess.run(['bash', '-n', os.path.join(SHELL_DIR, SELF)],
                       capture_output=True, text=True)
    assert r.returncode == 0, '语法错误: {}'.format(r.stderr)


def test_exposes_npu_graph_compile_switch(txt):
    """路线图 P4.10 对本脚本的唯一硬要求。"""
    assert re.search(r'(?m)^NPU_GRAPH_COMPILE=(\d)$', txt), \
        'NPU_GRAPH_COMPILE 必须显式赋 0/1（test_npu_graph_compile 的要求）'
    assert '--npu-graph-compile "$NPU_GRAPH_COMPILE"' in txt, \
        '开关必须传给 train_sft.py'


def test_lr_follows_sqrt_scaling_of_effective_batch(txt):
    """有效 batch = BATCH × WORLD_SIZE × GRAD_ACCUM；LR = 0.00356×√(有效batch/2500)±5e-5。

    ⚠ 2026-10-01：v21 每卡 2000 实测 OOM（活数据 20.09 GiB + 碎片 6.93 +
    CANN/HCCL 4.36 ≈ 31.4 GiB，容器可用只有 ~27.6 GiB），改为每卡 1000 +
    累积 2 ⇒ **有效 batch 与 LR 都不变**。所以等式必须含 GRAD_ACCUM，否则这次
    降档会被误判成 LR 漂移（v19 无累积变量，按 1 兜底）。
    """
    import math
    env = {m.group(1): m.group(2)
           for m in re.finditer(
               r'^(WORLD_SIZE|BATCH|LR|GRAD_ACCUM)=([0-9.]+)', txt, re.M)}
    env.setdefault('GRAD_ACCUM', '1')
    assert set(env) == {'WORLD_SIZE', 'BATCH', 'LR', 'GRAD_ACCUM'}, \
        '实得 {}'.format(sorted(env))
    accum = int(env['GRAD_ACCUM'])
    eff = int(env['BATCH']) * int(env['WORLD_SIZE']) * accum
    expect = 0.00356 * math.sqrt(eff / 2500)
    assert abs(float(env['LR']) - expect) < 5e-5, \
        'LR={} 与有效 batch={}（BATCH×WS×ACCUM={}）的平方根缩放预期 {:.5f} 不符'.format(
            env['LR'], eff, accum, expect)
