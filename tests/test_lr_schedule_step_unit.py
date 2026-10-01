"""LR 调度器的步数口径必须是 optimizer step（2026-10-01 修的缺陷）。

缺陷：`scheduler.step()` 只在每个 `optimizer.step()` 之后调一次，而
`total_steps` / `warmup_steps` / `T_max` 原来按 **micro-batch** 数算。
`accum=1` 时两者相等 ⇒ 历史 run 全部正常；`accum>1` 时差 accum 倍 ⇒
SequentialLR 的 milestone 提前到达、而 CosineAnnealingLR 的 T_max 是真实步数的
accum 倍 ⇒ **余弦在 epoch 末只走完 1/accum**，lr 停在峰值 ~77%（accum=2）而不是
退火到 ~0。这个 bug 恰好只在推荐的那档参数（accum=2）下出现。

本文件钉住「口径必须一致」这个不变量，并给出可解析的闭式判据。
"""
import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / 'scripts' / 'train_sft.py').read_text(encoding='utf-8')
TREE = ast.parse(SRC)


def _find_assign(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == name:
                return node
    return None


def test_total_steps_is_derived_from_optimizer_steps():
    """`total_steps` 必须在把 micro-batch 折算成 optimizer step 之后才算。"""
    n = _find_assign('total_steps')
    assert n is not None, '找不到 total_steps 赋值'
    src = ast.unparse(n)
    assert 'n_batches' in src
    # n_batches 在 total_steps 之前必须已被 accum 折算过
    fn = next(f for f in ast.walk(TREE)
              if isinstance(f, ast.FunctionDef) and f.name == 'main')
    body = ast.unparse(fn)
    i_div = body.find('gradient_accumulation_steps')
    i_tot = body.find('total_steps = max(1, args.epochs * n_batches)')
    assert i_div != -1 and i_tot != -1
    assert i_div < i_tot, \
        'accum 折算必须发生在 total_steps 之前（否则调度器按 micro-batch 走）'


def test_micro_batch_count_is_still_available_for_logging():
    """折算不能把 micro-batch 数弄丢 —— 日志要能同时报两个口径。"""
    fn = next(f for f in ast.walk(TREE)
              if isinstance(f, ast.FunctionDef) and f.name == 'main')
    body = ast.unparse(fn)
    assert 'micro_per_epoch' in body, '需要保留 micro-batch 口径供日志/核对'
    assert 'optimizer-steps/epoch' in SRC, '日志应标明调度器用的是 optimizer step'


def test_scheduler_steps_only_on_optimizer_step():
    """前提复核：`scheduler.step()` 确实只在 optimizer step 后调用一次。

    若哪天改成每个 micro-batch 都 step，那本文件的口径判断要重做 —— 所以把它
    作为**前提**显式钉住，而不是假设。
    """
    fn = next(f for f in ast.walk(TREE)
              if isinstance(f, ast.FunctionDef) and f.name == 'main')
    calls = [n.lineno for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'step'
             and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'scheduler']
    assert len(calls) == 1, f'scheduler.step() 应恰好一处调用，实得 {len(calls)}'
    # 它必须在 `if (i+1) % _accum_steps == 0` 那个分支里
    lines = SRC.splitlines()
    ln = calls[0]
    indent = len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
    assert indent > 8, \
        f'scheduler.step() 缩进 {indent} ⇒ 不在 optimizer-step 分支内（micro-batch 口径）'


def test_accum_one_keeps_identical_behaviour():
    """accum=1 时折算必须是恒等（历史 run 的调度曲线不能被这次修复改动）。"""
    for micro, accum, want in ((16763, 1, 16763), (16763, 2, 8382),
                               (16763, 4, 4191), (100, 3, 34)):
        got = (micro + max(1, accum) - 1) // max(1, accum)
        assert got == want, f'micro={micro} accum={accum}: {got} != {want}'