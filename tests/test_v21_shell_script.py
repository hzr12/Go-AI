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

1. **分布式口径必须与 `scripts/train_sft.py` 对得上，且必须能区分「现行」与
   「历史」**。包裹层换过两次轨（DDP → FSDP1 → DDP，见 spec
   `2026-10-01-ddp-instead-of-fsdp-design.md`），而脚本里**两段论证都保留**
   （spec §6.5：历史不删、只改理由）⇒「全文有没有某个 token」已经判不出谁在生效：
   退役的 `SHARD_GRAD_OP` / `use_orig_params` / `FullStateDictConfig` 全都还在文件里，
   只是被标成了历史。所以必须能分别取到**现行段**与**退役段**（脚本用
   `# ---- 标题 ----` 分节，见 `_comment_sections`）才判得清。
   `torchrun --nproc_per_node=N` 在三种轨道下都注入同一套
   `RANK`/`WORLD_SIZE`/`LOCAL_RANK`，所以脚本的 torchrun 行与 v19 逐字相同
   ——**这正是要断言的**：一旦有人以为「换包裹层就要改启动方式」而去动它，或者
   照着退役段的分片直觉去调 batch，脚本就开始说谎。
2. **不许出现 argparse 不接受的 flag**。D1 明令 v21 不加新 CLI 参数，
   而手写的长命令最容易被后来人「顺手加一个」；`test_run_py_sh.py` 的
   参数解析用例只验语法存在与否，这里额外做一次**静态**核对（对
   `train_sft.py` 源码里 `add_argument('...')` 的全集），顺带把
   「`--arch v21` 之类的分支选择器」也一并禁掉。
"""
import ast
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

#: 上一代（FSDP1，2026-09-30 ~ 2026-10-01 已退役）包裹层的机制词。
#: 它们**只能**出现在被标成历史的退役段里 —— 现行段出现即意味着脚本在按
#: 「参数/梯度/优化器状态都分片」的直觉误导读者（而那套直觉从来没有成立过：
#: 本模型 145 MB 的模型状态，分片只省 0.33%，见脚本「2026-10-01 的换轨」一节）。
_SHARDING_MECHANICS = (
    'FullyShardedDataParallel', 'SHARD_GRAD_OP', 'use_orig_params',
    'FullStateDictConfig', 'FullOptimStateDictConfig',
    'auto_wrap', '_fsdp_wrap_policy', '_fsdp_full_state_dict',
)


def _read(name):
    with open(os.path.join(SHELL_DIR, name), encoding='utf-8') as f:
        return f.read()


def _code_lines(txt):
    """剥掉整行注释。守护断言只看真正会被 bash 执行的行。"""
    return [l for l in txt.splitlines() if not l.strip().startswith('#')]


def _which_bash():
    import shutil
    return shutil.which('bash')


_SECTION_RE = re.compile(r'^#\s*-{2,}\s*(.*?)\s*-{2,}\s*$', re.M)


def _comment_sections(txt):
    """按 `# ---- 标题 ----` 把脚本注释切成若干段，返回 `[(标题, 正文), ...]`。

    为什么需要切段而不直接搜全文：本脚本同时保存着**现行**（DDP）与**退役**
    （FSDP1）两套口径（spec §6.5 要求保留历史）。退役的机制词全都还在文件里 ——
    「全文包含 X」已经恒真，判不出谁在生效。只有分别取到现行段与退役段，才判得清
    「这段话是在描述今天的做法，还是在记昨天的事」。
    """
    marks = list(_SECTION_RE.finditer(txt))
    out = []
    for i, m in enumerate(marks):
        lo = m.end()
        hi = marks[i + 1].start() if i + 1 < len(marks) else len(txt)
        out.append((m.group(1), txt[lo:hi]))
    return out


def _section(txt, *must_contain):
    """取标题含 `must_contain` 全部子串的那一节的 `(标题, 正文)`；找不到直接断言失败。"""
    for title, body in _comment_sections(txt):
        if all(tok in title for tok in must_contain):
            return title, body
    raise AssertionError(
        '脚本里没有标题含 %s 的注释段（现有段：%s）'
        % (' + '.join(repr(t) for t in must_contain),
           [t for t, _ in _comment_sections(txt)]))


def _current_ddp_construction():
    """从 `train_sft.py` 的 **AST** 里取出包裹点那一行构造（`ast.unparse` 后的形态）。

    刻意从代码取，而不是把字符串硬写死在本测试里：脚本声称的构造必须等于代码里
    真正的构造，否则那段注释就是在骗下一个人 —— 而这正是本文件存在的意义。
    走 AST 的 `Call` 节点（而不是文本搜 `DistributedDataParallel`）是因为那个名字在
    `train_sft.py` 里有四处，其中 import 行与解释性注释都不是调用点。
    """
    with open(TRAIN_SFT, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == 'DistributedDataParallel')
                 or (isinstance(n.func, ast.Attribute)
                     and n.func.attr == 'DistributedDataParallel'))]
    assert len(hits) == 1, (
        'train_sft.py 里 DistributedDataParallel 的调用点应恰好 1 个，实得 %d 个'
        % len(hits))
    return ast.unparse(hits[0])


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
# 禁令一：分布式口径必须与 train_sft.py 对得上，且分得清现行 / 退役
# --------------------------------------------------------------------------- #
def test_script_documents_ddp_as_the_current_wrapper(txt):
    """现行段必须写明**代码里真正在用的**构造，并给出读的人判断显存要用的三条事实。

    断言分两层：

    · **构造逐字对得上代码** —— 从 `train_sft.py` 的 AST 现取那一行
      （`_current_ddp_construction()`），要求脚本现行段原样包含它。写死字符串会
      在代码侧改了、脚本没改时静默通过；现取则会在那一刻红，正是这里要的效果。
    · **三条事实都在** —— 「参数不分片」「每 rank 持有完整模型」「存档无需汇聚」。
      少任何一条都会误导：只说「用了 DDP」而不说「每卡持有完整模型」，读的人会
      按「卡数越多每卡模型越省」去调大 batch（2026-09-30 正是这个直觉把脚本带到了
      FSDP1，见退役段）。
    """
    title, current = _section(txt, '现行')
    ctor = _current_ddp_construction()
    assert ctor in current, (
        '现行段（%r）没有写出代码里真正的构造 %r —— 注释与代码脱节，下一个读的人'
        '按注释理解包裹层' % (title, ctor))
    for token, why in (
            ('不分片', '每卡显存的大头仍是激活，模型状态不受卡数摊薄'),
            ('每 rank 持有完整模型', '「卡越多每卡模型越省」这个直觉从来没有成立过'),
            ('存档无需汇聚', 'save_model 剥 module. 之后与包裹前同形，旧 ckpt 继续可读'),
    ):
        assert token in current, \
            '现行段未说明「{}」（{}）'.format(why, token)


def test_current_section_makes_no_sharding_claims(txt):
    """现行段里**不许**出现分片机制词 —— 出现即表示脚本在按已退役的口径说话。

    这是上一代那条禁令（「不许出现 DDP 包装假设」）在换轨后的正确形态：禁令的方向
    翻了，但「注释里不许混进另一套机制」这件事没变。反向也钉一下 —— 这些词确实
    还在文件里（历史不删），所以它们必须待在退役段，见下一条。
    """
    title, current = _section(txt, '现行')
    bad = [tok for tok in _SHARDING_MECHANICS if tok in current]
    assert not bad, (
        '现行段（%r）出现了已退役包裹层的机制词 %s：读的人会以为今天还在分片，'
        '而 2026-10-01 起每 rank 持有完整模型' % (title, bad))


def test_retired_fsdp_history_is_preserved_and_labelled(txt):
    """退役段与换轨段必须**都留着**（历史不删），且各自自带标记与理由。

    spec §6.5 与本仓库一贯做法：「保留降级但改理由」，shell 里的历史论证同理 ——
    抹掉它，半年后会有人「再优化回」分片。所以这里断言的是**存在 + 有标记**，
    而不是「不存在」。脚本把它们记成**两节**（这是它自己的分法，本测试跟着走）：

      · **退役段**（标题含「历史」+「退役」）：2026-09-30 那个决定本身 —— 为什么
        当时选 FSDP1、当时那套配置是什么、以及**事后复盘**三条理由逐条检查后发现
        它们「全是 FSDP 自己制造的问题」。这一节保存的是「当时为什么站得住」。
      · **换轨段**（标题含 `FSDP1 → DDP`）：2026-10-01 的触发事故与规模测算 ——
        没有理由的历史等于没保留，所以事故与日期必须落在这里。

    最后反向钉一次：全部机制词必须**只**出现在退役段（现行段的反向检查见上一条）。
    """
    title, retired = _section(txt, '历史', '退役')
    switch_title, switch = _section(txt, 'FSDP1 → DDP')
    assert '2026-09-30' in title, \
        '退役段标题没有当时的决策日期（%r）' % title
    assert '2026-10-01' in switch_title or '2026-10-01' in switch, \
        '换轨段（%r）没有换轨日期：没有日期的换轨记录读者会当成现状' % switch_title
    for token, why in (
            ('KeyError', '换轨的触发事故（ema.update() 的 KeyError）'),
            ('_fsdp_wrapped_module', '根因在 torch 源码里的具体那一行'),
    ):
        assert token in switch, \
            '换轨段未记下「{}」（{}）'.format(why, token)

    # 机制词只允许待在退役段：逐节扫一遍，出现在退役段以外的节就红。
    offenders = []
    for sec_title, body in _comment_sections(txt):
        if sec_title == title:
            continue
        hit = [tok for tok in _SHARDING_MECHANICS if tok in body]
        if hit:
            offenders.append((sec_title, hit))
    assert not offenders, (
        '分片机制词出现在退役段（%r）之外：%s —— 它们要么该删，要么该被标成历史'
        % (title, offenders))


def test_script_does_not_set_the_wrapper_itself(txt):
    """脚本的**可执行行**里不许出现包裹层类名 —— 包裹是 `train_sft.py` 的事。

    上一代版本这条断言是 `assert 'DistributedDataParallel' not in txt`（对**全文**），
    2026-10-01 换轨后方向翻转：现行包装就是 DDP，全文里必须有它。真正的不变量是不
    变的那个更窄的版本 —— bash 不得去设置包裹层（那会让「代码侧的包裹顺序契约」在
    脚本里被旁路），所以只查**会被 bash 执行**的行（`_code_lines` 剥掉整行注释）。
    """
    offenders = [l for l in _code_lines(txt)
                 if 'DistributedDataParallel' in l or 'FullyShardedDataParallel' in l]
    assert not offenders, \
        '脚本的可执行行里不该出现包裹层类名（包裹是 train_sft.py 的职责）：%s' % offenders


def test_torchrun_invocation_is_unchanged_from_v19(txt):
    """torchrun 行与 v19 逐字相同（只允许 --out / --ver / 累积档位不同）。

    换包裹层不改启动方式（两次换轨都没改）：`--nproc_per_node=N` 照旧注入
    RANK/WORLD_SIZE/LOCAL_RANK，`train_sft.py` 读的是同一套环境变量契约。
    任何对这一行的改动都必须先有代码侧的契约变更。

    ⚠ 正因为逐字相同，**不能用 torchrun 行去判断当前用的是哪种包裹层**（脚本的
    「现行」段把这句话也写进去了）。

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
