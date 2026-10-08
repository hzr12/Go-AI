"""train_sft 早停的 DDP 同步（stop_flag + 双层 break）+ selfplay 最佳权重保存兜底。

缺陷 A（train_sft，早停根本没停训、且各 rank 不同步）
------------------------------------------------------
`--early-stop` 的判定块挂在 **step 循环**内的周期 eval 里（eval 是 step 级任务），
触发时只 `break` 跳出 step 循环；外层 `for epoch in range(start_epoch, args.epochs)`
照开下一个 epoch → 早停形同虚设，训练照跑完。
更糟的是判定被 `and is_main` 门住：`early_stop_counter` 只在 rank0 推进、
`break` 也只有 rank0 执行，于是 rank0 进入下一 epoch 的 `set_epoch`/allreduce
而其余 rank 还在 step 循环里做 eval/反向 —— collectives 次序错配 → 挂死或 DDP 崩。

修复形态：判定仍然只在 rank0 做（计数器语义不变），但结论写进 `stop_flag` 张量，
**所有 rank 无条件** broadcast（未置位也必须广播，否则没停的 rank 会跳过 collective
—— 只是把「次序错配」换成「broadcast 挂死」），然后 step 循环内外各 break 一次。

缺陷 B（selfplay，「最佳权重」可能压根不存在）
---------------------------------------------
`best_path` 初始化为 `args.out` 后**从未更新**，而实际落盘的是
`args.out + '.pth'`（`args.out` 不带后缀时两者不是同一个文件）；
buffer 从未攒到训练阈值时 `train_epochs` 一次都没跑，一个权重文件都不存在，
收尾却无条件打印「最佳权重: {best_path}」，下游 eval_elo.py 收到不存在的路径。

覆盖:
  - `_early_stop_decision`：指标方向（loss 越小越好 / top1 越大越好）、
    改善清零、patience 边界（patience-1 不停、恰好 patience 停）
  - `_sync_stop_flag`：DDP + 早停启用时**必定**广播一次（已置位也不短路）；
    单卡或未启用早停时零 collective
  - 结构锁：广播点不在任何 `is_main` 分支内、stop_flag 在 epoch for 之前创建、
    epoch/step 两层各一处 `stop_flag` break
  - 早停计数器只有一个权威：跨指标（top1）复位已删除，判定函数独占归零规则
  - 结构锁：selfplay 无恒真嵌套的 `if is_main:`
  - `_save_checkpoint` / `_finalize_checkpoint`：返回真实落盘路径；
    从未存过时收尾补存、存过则原样返回且不写盘

本文件同时承载 selfplay 侧两个用例（brief 允许二选一并说明）：两者同属
「训练收尾不许宣称不存在的产物」这一条止血线，拆成两个文件只会多一次 pytest 冷启动。

全部 CPU、秒级：不加载真模型（假 ai / 1 元素 CPU 张量），不起分布式进程组，
不落盘到仓库（用 tmp_path）。
"""
import argparse
import ast
import inspect
import os
import sys
import textwrap

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.selfplay_train as st          # noqa: E402
import scripts.train_sft as t                # noqa: E402


def _read_source(module):
    """读被导入模块的源文件全文（结构锁要覆盖 main() 之外的函数体，如判定函数）。"""
    with open(module.__file__, encoding='utf-8') as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# 判定函数：指标方向 / 计数清零 / patience 边界
# --------------------------------------------------------------------------- #
def test_counter_resets_on_improvement_and_stops_at_patience():
    """loss 变小 = 改善（清零），连续不变到第 patience 次 = 停。

    修复前的判定块无法被单测覆盖（内嵌在 main() 的三层循环里），这里把
    「改善 → 计数归零 → 恰好 patience 次才停」三条语义抽成纯函数锁住。
    """
    # loss：越小越好
    assert t._early_stop_decision('loss', 0.5, float('inf'), 0, 3) == (True, 0, False), \
        '首次 eval 的 loss 必然刷新最佳（best 初值 +inf）'

    best, counter = 0.5, 0
    for k in (1, 2):
        new_best, new_counter, stop = t._early_stop_decision('loss', 0.5, best, counter, 3)
        assert (new_best, new_counter, stop) == (False, k, False), \
            f'第 {k} 次无改善：计数应为 {k} 且不停（patience=3）'
        best = 0.5
        counter = new_counter

    new_best, new_counter, stop = t._early_stop_decision('loss', 0.5, best, counter, 3)
    assert (new_best, new_counter, stop) == (False, 3, True), \
        '第 3 次（== patience）必须触发早停'

    # 改善后连续计数必须清零，否则早过一次就直接早停
    assert t._early_stop_decision('loss', 0.4, 0.5, 3, 3) == (True, 0, False)

    # top1：越大越好；变小 = 无改善 = 累加
    assert t._early_stop_decision('top1', 0.6, 0.5, 0, 2) == (True, 0, False)
    assert t._early_stop_decision('top1', 0.4, 0.5, 0, 2) == (False, 1, False), \
        'top1 变小不是改善（方向反了就永远早停不了 / 或一改善就早停）'
    assert t._early_stop_decision('top1', 0.4, 0.5, 1, 2) == (False, 2, True)


def test_patience_boundary_is_exactly_counter_ge_patience():
    """边界：累计到 patience 次停，patience-1 次不停。

    入参 counter 是**此前**已累计的无改善次数，返回的 new_counter 是加上本次后的
    总数，故「恰好第 patience 次停」对应传入 patience-1。
    """
    assert t._early_stop_decision('top1', 0.5, 0.5, 0, 2) == (False, 1, False), \
        'patience=2 时第 1 次无改善不该停'
    assert t._early_stop_decision('top1', 0.5, 0.5, 1, 2) == (False, 2, True), \
        'patience=2 时第 2 次（== patience，含等号）必须停'
    assert t._early_stop_decision('top1', 0.5, 0.5, 0, 1) == (False, 1, True), \
        'patience=1 首次无改善即停'


# --------------------------------------------------------------------------- #
# 停止标志同步：假 dist 记录 broadcast 调用
# --------------------------------------------------------------------------- #
class _FakeDist:
    """只记录 broadcast 的假进程组替身（不建真 PG，不起进程）。"""

    def __init__(self):
        self.calls = []

    def broadcast(self, tensor, src=0):
        self.calls.append((tensor, src))


def test_stop_flag_sync_broadcasts_on_all_ranks(monkeypatch):
    """DDP + 早停启用 → 恰好一次 broadcast(stop_flag, src=0)，且**已置位也不短路**。

    短路是这里最容易踩的坑：未置位的 rank 跳过 broadcast、置位的 rank 阻塞在
    broadcast 上 → 死锁。正确形态是「所有 rank 同点到达、无论结论如何都参加」。
    """
    fake = _FakeDist()
    monkeypatch.setattr(t, 'dist', fake)

    flag = torch.zeros(1, dtype=torch.long)
    t._sync_stop_flag(flag, is_dist=True, early_stop_enabled=True)
    assert len(fake.calls) == 1, f'应恰好广播一次，实际 {len(fake.calls)} 次'
    assert fake.calls[0][0] is flag, '广播的必须是 stop_flag 张量本身（原地被 rank0 改写）'
    assert fake.calls[0][1] == 0, '源必须是 rank0（只有它做早停判定）'

    flag.fill_(1)
    t._sync_stop_flag(flag, is_dist=True, early_stop_enabled=True)
    assert len(fake.calls) == 2, \
        'stop_flag 已置位时仍必须 broadcast —— 加 `if stop_flag.item()` 短路会让未置位的 rank 跳过 collective'


@pytest.mark.parametrize('is_dist,early_stop_enabled', [
    (False, True),    # 单卡：走 collective 只会拖慢，且无同点到达的必要性
    (True, False),    # 没开早停：stop_flag 永远是 0，没必要占一次 collective
    (False, False),
])
def test_single_rank_and_no_early_stop_never_collective(monkeypatch, is_dist, early_stop_enabled):
    """单卡或未启用早停 → 零 broadcast。"""
    fake = _FakeDist()
    monkeypatch.setattr(t, 'dist', fake)

    flag = torch.zeros(1, dtype=torch.long)
    t._sync_stop_flag(flag, is_dist=is_dist, early_stop_enabled=early_stop_enabled)
    assert fake.calls == [], f'不该有 collective，却调了 {len(fake.calls)} 次'


# --------------------------------------------------------------------------- #
# 结构锁（**结构断言**，不验证运行期数值）
# --------------------------------------------------------------------------- #
_MAIN_TREE = ast.parse(textwrap.dedent(inspect.getsource(t.main)))
_PARENTS = {id(c): p for p in ast.walk(_MAIN_TREE) for c in ast.iter_child_nodes(p)}


def _ancestors(node):
    cur = _PARENTS.get(id(node))
    while cur is not None:
        yield cur
        cur = _PARENTS.get(id(cur))


def _epoch_for():
    found = [n for n in ast.walk(_MAIN_TREE)
             if isinstance(n, ast.For) and ast.unparse(n.target) == 'epoch']
    assert len(found) == 1, f'should be exactly 1 epoch loop, got {len(found)}'
    return found[0]


def _stop_flag_breaks(scope):
    """`if <含 stop_flag 读取>: break ...` 的 If 节点。

    判据是「test 里**出现** `stop_flag.item()`」而不是「test 恰好等于
    `stop_flag.item()`」：step 循环内那一处后来加了 `args.early_stop == 1 and`
    的前缀 —— 早停没开时 `stop_flag` 恒为 0（唯一写入点被 `args.early_stop == 1`
    门控），那次 `.item()` 是纯浪费的同步点。锁的仍然是「两处 break 都在」这个
    wiring 不变量，多余的与门只是让它在早停关闭时不白付同步。
    """
    return [n for n in ast.walk(scope)
            if isinstance(n, ast.If) and 'stop_flag.item()' in ast.unparse(n.test)
            and any(isinstance(s, ast.Break) for s in n.body)]


def test_eval_block_sync_is_outside_main_guard():
    """同步点必须在所有 rank 都到达的位置：不在任何 is_main 分支内。

    **结构断言**（读 main() 的 AST，不跑训练）：锁的是「广播点的位置」这一
    wiring 不变量。它无法用行为测试表达 —— 真要复现必须起多进程 DDP。
    """
    calls = [n for n in ast.walk(_MAIN_TREE)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == '_sync_stop_flag']
    assert len(calls) == 1, f'should be exactly 1 sync call site, got {len(calls)}'

    for anc in _ancestors(calls[0]):
        if isinstance(anc, ast.If):
            test_src = ast.unparse(anc.test)
            assert 'is_main' not in test_src, (
                f'停止标志同步被关进 `{test_src}` 分支 —— 单 rank 参与的 broadcast '
                f'会把「epoch/step 次序错配」换成「broadcast 挂死」')

    creates = [n for n in ast.walk(_MAIN_TREE)
               if isinstance(n, ast.Assign)
               and any(isinstance(tg, ast.Name) and tg.id == 'stop_flag' for tg in n.targets)]
    assert creates, 'main() 里找不到 stop_flag 的创建语句'
    assert min(c.lineno for c in creates) < _epoch_for().lineno, \
        'stop_flag 必须在 epoch 循环之前创建（循环内每 epoch 重置成 0 就失去跨轮意义了）'


def test_two_breaks_one_per_loop():
    """epoch 循环内、step 循环内外各一处 stop_flag 触发的 break。

    **结构断言**：只跳出 step 循环的那个 break 就是「早停无效」的原缺陷 ——
    外层 epoch 循环会照开下一轮。这里锁住两层 break 都在。
    """
    epoch_for = _epoch_for()
    step_fors = [n for n in ast.walk(epoch_for)
                 if isinstance(n, ast.For) and ast.unparse(n.target) == 'i']
    assert len(step_fors) == 1, f'epoch 循环里应有且仅有一个 step 循环，实际 {len(step_fors)}'
    step_ids = {id(n) for n in ast.walk(step_fors[0])}

    guarded = _stop_flag_breaks(epoch_for)
    inside = [n for n in guarded if id(n) in step_ids]
    outside = [n for n in guarded if id(n) not in step_ids]
    assert len(guarded) == 2, f'stop_flag 触发的 break 应恰好 2 处，实际 {len(guarded)}'
    assert len(inside) == 1, 'step 循环内必须 break（否则继续训完本 epoch 剩下的 step）'
    assert len(outside) == 1, \
        'epoch 循环内、step 循环外必须再 break 一次（缺它 = 早停只跳 batch，训练照跑完）'

    # step 循环内那处 `.item()` 必须被 `args.early_stop == 1` 门控。
    # `.item()` 是**同步点**（D2H + 阻塞等队列排空），而它每个 step 都执行一次；
    # `stop_flag` 的唯一写入点是 `if args.early_stop == 1 and is_main:` 里的
    # `fill_(1)` ⇒ 早停没开时它恒为 0，那次同步每次都白付：host 被拉回等 GPU，
    # 下一个 step 的 kernel 发射排不上去。
    # 锁在这里是因为 `_stop_flag_breaks` 认的是「test 里出现 `stop_flag.item()`」，
    # `and` 前缀在不在都算命中 —— 只锁「两处 break 都在」的话，删掉前缀仍然全绿。
    assert 'args.early_stop' in ast.unparse(inside[0].test), (
        'step 循环内的 stop_flag 读取没有早停门控（test=%r）—— 早停关闭时它'
        '每个 step 白付一次同步' % ast.unparse(inside[0].test))


# --------------------------------------------------------------------------- #
# 早停计数器：**唯一权威**是判定函数（跨指标复位已删除）
#
# 修复前 main() 在「top1 刷新最佳 → 保存模型」分支里顺手写了第二处
# `early_stop_counter = 0`。top1 与 `--early-stop-metric` 是两个独立指标：
# top1 改善不代表早停指标改善。于是每次 top1 涨就把计数抹平，loss 迟迟不动
# 也攒不满 patience，早停形同虚设。默认 `--early-stop-metric top1` 路径自洽，
# 所以一直没被发现。
# --------------------------------------------------------------------------- #
def test_counter_not_reset_by_other_metric():
    """`--early-stop-metric loss` 时，别的指标（top1）改善不得清零计数器。

    **本用例只锁判定函数的契约**：早停指标未改善时返回 `counter + 1`。
    跨指标交互本身（main() 里那段判据会不会顺手把计数器抹平）**不由本用例覆盖** ——
    修复前它就是绿的（那处复位在 main() 内，这个纯函数调用根本看不到它）。
    跨指标由结构锁 `test_counter_reset_only_inside_decision_function` 覆盖：
    任何在 `_early_stop_decision` 之外改写计数器的写法都会红。
    """
    # 连续 3 次 eval：top1 一路涨（0.50 → 0.60），早停指标 loss 死死不动（0.30）
    best_metric, counter = 0.30, 0
    for k, top1 in enumerate((0.50, 0.55, 0.60), start=1):
        improved, new_counter, should_stop = t._early_stop_decision(
            'loss', 0.30, best_metric, counter, 3)
        assert not improved, \
            f'第 {k} 次：loss 没变就不是改善（同期 top1={top1} 涨了也不算数）'
        assert new_counter == k, (
            f'第 {k} 次：top1={top1} 改善但早停指标未改善，计数器应累加到 {k}，'
            f'实际 {new_counter} —— 跨指标复位会让它永远是 0，patience 攒不满')
        assert should_stop == (k == 3), \
            f'第 {k} 次的 should_stop 应为 {k == 3}，实际 {should_stop}'
        counter = new_counter

    # 早停指标**自身**改善 → 归零（唯一有权归零的地方）
    assert t._early_stop_decision('loss', 0.29, best_metric, counter, 3) == (True, 0, False), \
        '早停指标自身改善时必须清零，否则早过一次就直接早停'


# 用例「top1 路径改善仍归零 / 非改善仍累加 / patience 边界含等号」由 P1.4 已有的
# `test_counter_resets_on_improvement_and_stops_at_patience`（top1 段：0.6>0.5 →
# 0、0.4<0.5 → +1、再 +1 恰停）与 `test_patience_boundary_is_exactly_counter_ge_patience`
# 覆盖，按 brief 要求复用而非重复造。删除 :1743 的跨指标复位不影响它们 ——
# 那条复位在 main() 里，不在判定函数内。

_TRAIN_SRC = _read_source(t)
_SELFPLAY_SRC = _read_source(st)
_TRAIN_TREE = ast.parse(_TRAIN_SRC)
_SELFPLAY_TREE = ast.parse(_SELFPLAY_SRC)


def _parent_map(tree):
    return {id(c): p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}


def _chain(node, parents):
    cur = parents.get(id(node))
    while cur is not None:
        yield cur
        cur = parents.get(id(cur))


def _one_func(tree, name):
    found = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert len(found) == 1, f'should be exactly 1 def {name}, got {len(found)}'
    return found[0]


def _assigns_to(node, name):
    """`node` 是否把 `name` 写成了值 —— 覆盖全部**赋值语句**形态。

    形态：`x = v`（ast.Assign）、`x += v`（ast.AugAssign）、`x: T = v`（ast.AnnAssign）、
    `(x := v)`（ast.NamedExpr）；target 内的解包由 `ast.walk` 展开，故
    `a, x, b = ...` 也收得到。注意后三者的 target 是**单个** `node.target` 而非
    `node.targets` —— 只认 ast.Assign 时它们对整个锁隐形（实测：`x += 1` 与
    `x: int = 0` 都收不到，字面子串计数 `'early_stop_counter = 0'` 也数不出后者）。

    不覆盖（对本变量无现实意义，故有意不锁）：for/with/except-as 的绑定目标、
    `del x`、以及对 `obj.x` 的属性赋值（那不重绑定这个名字本身）。
    """
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
        targets = [node.target]
    else:
        return False
    for tg in targets:
        for sub in ast.walk(tg):
            if isinstance(sub, ast.Name) and sub.id == name:
                return True
    return False


def _calls(node, func_name):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
        and node.func.id == func_name


def _src(texts):
    """把 AST 节点还原成源码片段（给断言消息用）。

    注意：不能对节点调 `.strip()` —— ast 节点没有这个属性，会 AttributeError
    把断言消息本身炸掉（那样锁虽然还是红的，但读不到任何原因）。
    """
    return '; '.join(ast.unparse(n).strip() for n in texts)


def test_counter_reset_only_inside_decision_function():
    """`early_stop_counter` 的写入点只有两种合法形态：init + from_decision。

    写入点收集（`_assigns_to`）覆盖全部**赋值语句**形态 —— 普通赋值 `x = v`
    （含 `a, x, b = ...` 元组解包）、增强赋值 `x += v`（AugAssign）、
    带注解赋值 `x: int = v`（AnnAssign）、海象赋值 `(x := v)`（NamedExpr）；
    其中只有**普通赋值**可能合法：

    - `init`：直接挂在 `main()` 体上的 `early_stop_counter = 0`（一次性初始化）
    - `from_decision`：值来自 `_early_stop_decision(...)`（元组解包写入）

    为什么这两种之外一律 rogue：`x += 1` 是第二套计数规则（绕开判定函数的
    改善/累加语义），`x: int = 0` 改的是声明而非计数状态 —— 两者都会让
    「计数器只有一个权威」这个不变量在结构上无法判定。上面第一行的字面计数
    `early_stop_counter = 0` 只是 brief 的原始判据（粗粒度、只认普通赋值的
    字面形状），真正的锁是后面的 AST 角色判定：它连子串数不到的形态也能收。

    **结构锁，修复前必红**：修复前 main() 的「top1 刷新最佳」分支里还有第三处
    `early_stop_counter = 0` —— 一个计数器两个权威，`--early-stop-metric loss` 时
    patience 永远攒不满。
    """
    assert _TRAIN_SRC.count('early_stop_counter = 0') == 1, (
        f"train_sft.py 里有 {_TRAIN_SRC.count('early_stop_counter = 0')} 处 "
        f"`early_stop_counter = 0`，应恰好 1 处（main() 的初始化）—— 多出来的那处"
        f"是「最佳模型保存」判据(top1)在跨指标复位早停计数器")

    decision = _one_func(_TRAIN_TREE, '_early_stop_decision')
    main_fn = _one_func(_TRAIN_TREE, 'main')
    writes = [n for n in ast.walk(_TRAIN_TREE) if _assigns_to(n, 'early_stop_counter')]

    def _role(w):
        # 唯一合法的两种形态，都必须是**普通赋值**：
        # init = 直接挂在 main() 体上的 `early_stop_counter = 0`（一次性初始化）
        if isinstance(w, ast.Assign) and w in main_fn.body:
            return 'init'
        # from_decision = 值来自 `_early_stop_decision(...)`（含元组解包）
        if isinstance(w, ast.Assign) and _calls(w.value, '_early_stop_decision'):
            return 'from_decision'
        # 增强赋值 / 带注解赋值即便挂在 main() 体上、即便值恰好来自判定函数，
        # 也都不是合法形态（`x += 1` 是第二套计数规则；`x: int = 0` 改的是声明）
        return 'rogue'

    roles = sorted(_role(w) for w in writes)
    rogues = [w for w in writes if _role(w) == 'rogue']
    assert not rogues, (
        f'train_sft.py:{[w.lineno for w in rogues]} 在 `_early_stop_decision` 之外'
        f'以非判定结果的形式改写 `early_stop_counter`（{_src(rogues)}）—— 计数器只能由'
        f'判定函数独占维护，否则「最佳模型保存」判据与「早停判据」两个独立指标会互相污染')
    assert roles == ['from_decision', 'init'], \
        f'`early_stop_counter` 的写入点应恰好是 [初始化, 接收判定结果] 两个，实际 {roles}'

    # 归零规则本身只在判定函数里（new_counter = 0 if improved else counter + 1）
    zero_rules = [n for n in ast.walk(decision)
                  if isinstance(n, ast.Assign) and isinstance(n.value, ast.IfExp)
                  and any(isinstance(tg, ast.Name) and tg.id == 'new_counter'
                          for tg in n.targets)]
    assert len(zero_rules) == 1, \
        f'_early_stop_decision 里应有且仅有一条「改善则归零」规则，实际 {len(zero_rules)}'
    assert isinstance(zero_rules[0].value.body, ast.Constant) \
        and zero_rules[0].value.body.value == 0, '改善分支必须把计数器置 0'
    assert ast.unparse(zero_rules[0].value.test) == 'improved', \
        '归零条件必须是早停指标自身的改善'


def test_no_dead_nested_is_main_in_selfplay():
    """selfplay_train.py 不得有 `if is_main:` 直接嵌在另一个 `if is_main:` 内。

    **结构锁，修复前必红**：内层判据恒真（外层已判过），是纯噪音；P1.4 删掉了第三处，
    剩 async 分支的 NEW BEST 打印与同步分支的 loss 打印两处。
    """
    parents = _parent_map(_SELFPLAY_TREE)
    nested = []
    for n in ast.walk(_SELFPLAY_TREE):
        if isinstance(n, ast.If) and ast.unparse(n.test) == 'is_main':
            if any(isinstance(a, ast.If) and ast.unparse(a.test) == 'is_main'
                   for a in _chain(n, parents)):
                nested.append(n.lineno)

    assert not nested, (
        f'selfplay_train.py 第 {nested} 行是恒真嵌套的 `if is_main:` —— 外层已经判过，'
        f'内层永远为真；应删掉内层判断、缩进外移（打印内容一个字都不改）')


# --------------------------------------------------------------------------- #
# selfplay 侧：最佳权重保存兜底
# --------------------------------------------------------------------------- #
class _FakeModel:
    def __init__(self):
        self.state_dict_calls = 0

    def state_dict(self):
        self.state_dict_calls += 1
        return {'w': torch.zeros(2)}


class _FakeAI:
    """只提供 `_save_checkpoint` 用到的 `.model.state_dict()`。"""

    def __init__(self):
        self.model = _FakeModel()


def _args(out):
    return argparse.Namespace(out=str(out), board_size=9)


def test_save_checkpoint_returns_real_written_path(tmp_path):
    """返回值必须是**实际落盘**的路径：`--out` 不带 .pth 时是 `<out>.pth`。"""
    ai = _FakeAI()
    args = _args(tmp_path / 'az')          # 不带 .pth 后缀

    got = st._save_checkpoint(ai, args, 9, 3)
    assert got == str(tmp_path / 'az.pth'), \
        f'返回的路径与实际写入的路径不是同一个文件: {got}'
    assert os.path.isfile(got), f'返回的路径上没有文件: {got}'
    assert ai.model.state_dict_calls == 1

    payload = torch.load(got, weights_only=False)
    assert set(payload) == {'model', 'iter', 'board_size', 'args'}, \
        f'checkpoint 字段集被改动了: {sorted(payload)}'
    assert payload['iter'] == 3 and payload['board_size'] == 9

    # 已带 .pth 的 --out 不得被补成 "az.pth.pth"
    ai2 = _FakeAI()
    args2 = _args(tmp_path / 'az2.pth')
    got2 = st._save_checkpoint(ai2, args2, 9, 1)
    assert got2 == str(tmp_path / 'az2.pth') and os.path.isfile(got2)


def test_finalize_saves_when_never_saved(tmp_path):
    """buffer 从未达训练阈值（一个权重都没存）→ 收尾必须补存并返回真实路径。"""
    ai = _FakeAI()
    args = _args(tmp_path / 'az')

    got = st._finalize_checkpoint(ai, args, 9, 5, best_path=args.out, saved_any=False)
    assert got == str(tmp_path / 'az.pth'), f'兜底保存应返回真实路径，实际 {got}'
    assert os.path.isfile(got), '收尾宣称了「最佳权重」却没有任何文件落盘'
    assert ai.model.state_dict_calls == 1

    # 已存过 → 原样返回，且**不**再写盘
    ai2 = _FakeAI()
    args2 = _args(tmp_path / 'az2')
    existing = st._save_checkpoint(ai2, args2, 9, 4)
    assert ai2.model.state_dict_calls == 1

    got2 = st._finalize_checkpoint(ai2, args2, 9, 5, best_path=existing, saved_any=True)
    assert got2 == existing, f'saved_any=True 时必须原样返回 best_path，实际 {got2}'
    assert ai2.model.state_dict_calls == 1, 'saved_any=True 时不得再写一次盘'
    assert os.path.isfile(got2)
