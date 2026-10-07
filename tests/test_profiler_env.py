"""`GOAI_PROFILE_STEPS`：剖析窗口长度可配，且剖析器**只收一次尾**。

缺陷
----
2026-10-07 真机（`a_v7.3_npu2`）跑 `GOAI_PROFILE=5` 时暴露了两件事：

1. **窗口长度硬编码 50**（`train_sft.py` 里 `step >= _prof_at + 50`）。只想看
   第 5 步的内核表，也必须等满 50 步 —— 按当时 13 s/step 是 ~11 分钟，而且
   剖析器开着会把 eval 拖到 108 s（53.9 s 的两倍），观测窗口越长污染越重。
2. **收尾放在 `try` 之外、且 `_prof_ctx` 永不置空**：窗口一过，之后**每个**
   `_do_stdout` 步都会再 `__exit__` 一次、再失败一次。真机实际表现是每 2 步
   刷一行 ``'profile' object has no attribute 'key_averages'``；而 `__exit__`
   一旦真的抛异常，因为它在 `try` 外面，会**直接打穿训练循环**。

覆盖
----
  - 契约锁（1）：`_prof_span` 必须从 `GOAI_PROFILE_STEPS` 读，且被 `max(0, ...)`
    钳住（负数会让 `step >= _prof_at + _prof_span` 恒真 ⇒ 起点之前就停表）。
  - 契约锁（1）：退出条件的右操作数必须是 `_prof_span`，不许再出现 `_prof_at + 50`
    这种字面窗口。
  - 行为锁（2）：打印剖析表的那个 `if _do_stdout:` 分支里必须有 `_prof_ctx = None`，
    且 `__exit__` 必须位于 `try` 的 body 内 —— 两条一起才等于「只收一次尾、
    收尾失败不打穿训练」。
  - 行为锁（3，见 `test_key_averages_has_chrome_trace_fallback`）：`key_averages()`
    在 torch_npu 2.1.0.post10 上**不存在**（`profile` 是独立类，只有 8 个实例
    方法），取不到时必须落回 `export_chrome_trace` 解析，而不是被外层
    `except Exception` 吞成一行 warning —— 那正是真机「窗口跑完了、表没有」的原因。

前 3 条只解析 AST、不 import `scripts/train_sft.py`（import 会把整个训练入口
模块跑一遍，也就不需要 NPU / torch_npu）。最后一条**要 import**：它直接喂一份
合成 chrome trace 给 `_prof_parse_trace`，验证聚合口径 —— 这是纯函数，秒级 CPU。
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()
TREE = ast.parse(SRC)


def _prof_stdout_if():
    """`if _do_stdout:` 里打印剖析表的那一个（体内含 `_prof_ctx`）。

    与 `tests/test_swanlab_logging.py::_prof_print_if` 同一套认法：main() 里有
    多个 `if _do_stdout:`，靠「第几个」定位会随排版漂移，按「体内含 _prof_ctx」
    才是它。
    """
    hits = [n for n in ast.walk(TREE)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == '_do_stdout'
            and any('_prof_ctx' in ast.unparse(s) for s in n.body)]
    assert len(hits) == 1, \
        f'体内含 _prof_ctx 的 `if _do_stdout:` 应恰好 1 个，实得 {len(hits)}'
    return hits[0]


def test_prof_span_read_from_goai_profile_steps():
    """窗口长度必须来自 `GOAI_PROFILE_STEPS`，且钳到 >= 0。"""
    assigns = [n for n in ast.walk(TREE)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == '_prof_span'
                       for t in n.targets)]
    assert len(assigns) == 1, \
        f'`_prof_span` 赋值应恰好 1 处，实得 {len(assigns)}'
    src = ast.unparse(assigns[0].value)
    assert 'GOAI_PROFILE_STEPS' in src, \
        f'_prof_span 未从 GOAI_PROFILE_STEPS 读，实得: {src}'
    assert src.startswith('max(0,'), \
        f'_prof_span 应被 max(0, ...) 钳住（负数会让停表条件恒真），实得: {src}'


def test_exit_condition_uses_prof_span_not_a_literal_50():
    """退出条件右操作数是 `_prof_span`，不许再有 `_prof_at + 50` 的硬编码窗口。"""
    adds = [n for n in ast.walk(TREE)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)
            and isinstance(n.left, ast.Name) and n.left.id == '_prof_at']
    assert adds, '找不到 `_prof_at + …` 的窗口表达式'
    rights = sorted({ast.unparse(n.right) for n in adds})
    assert rights == ['_prof_span'], \
        f'`_prof_at + ?` 的右操作数应只有 _prof_span，实得 {rights}'
    assert '_prof_at + 50' not in SRC, '源码里仍有硬编码窗口 `_prof_at + 50`'


def _exit_calls(node):
    """`node` 子树里所有 `….__exit__(…)` 调用。"""
    return [c for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == '__exit__']


def _ids_in_try_bodies(tried):
    """所有 `try` 的 **body** 语句子树的 id 集（不含 else/finalbody）。"""
    ids = set()
    for t in tried:
        for stmt in t.body:
            for n in ast.walk(stmt):
                ids.add(id(n))
    return ids


def test_profiler_teardown_runs_once_and_inside_try():
    """剖析器只收一次尾：`_prof_ctx = None` 在前，`__exit__` 在 `try` 内。

    两条缺一都复现真机故障：缺前者 ⇒ 每个打点步重复 `__exit__` 反复刷 warning；
    缺后者 ⇒ `__exit__` 抛异常时直接打穿训练循环。
    """
    prof_if = _prof_stdout_if()

    nulled = [n for n in ast.walk(prof_if)
              if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == '_prof_ctx'
                      for t in n.targets)
              and isinstance(n.value, ast.Constant) and n.value.value is None]
    assert nulled, '剖析表分支里没有 `_prof_ctx = None`（会反复重入 __exit__）'

    exits = _exit_calls(prof_if)
    assert exits, '剖析表分支里没有 __exit__ 调用（剖析器收不了尾）'
    tried = [t for t in ast.walk(prof_if) if isinstance(t, ast.Try)]
    assert tried, '剖析表分支里没有 try（__exit__ 抛异常会打穿训练循环）'

    inside = _ids_in_try_bodies(tried)
    orphan = [c for c in exits if id(c) not in inside]
    assert not orphan, \
        f'try 的 body 之外还有 __exit__（不受保护/会反复执行）: 行 {[c.lineno for c in orphan]}'


def _handler_catches(handler, exc_name):
    """这个 `except` 是否捕 `exc_name`（裸 `except:` 视为全捕）。"""
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return exc_name in {ast.unparse(t) for t in types}


def test_key_averages_has_chrome_trace_fallback():
    """取 `key_averages()` 必须带 AttributeError 守卫，且落到 chrome trace 退路。

    torch_npu 2.1.0.post10 的 `torch_npu.profiler.profile` 是**独立类**，实例
    方法只有 ``add_metadata / add_metadata_json / export_chrome_trace /
    export_memory_timeline / export_stacks / start / step / stop`` 共 8 个，
    **没有** `key_averages`（`scripts/probe_npu_profiler.py` 实测）。直接取就
    AttributeError，而它落在外层 `except Exception` 里 ⇒ 被吞成一行 warning ⇒
    「窗口跑完了、表一张没有」，这正是 2026-10-07 真机的表现。

    退路必须是 `export_chrome_trace` —— 那 8 个方法里**确实提供**的一个。
    """
    prof_if = _prof_stdout_if()
    guarded = [t for t in ast.walk(prof_if)
               if isinstance(t, ast.Try)
               and any('key_averages' in ast.unparse(s) for s in t.body)]
    assert guarded, '剖析表分支里找不到对 key_averages 的 try'

    fallback_stmts = None
    for t in guarded:
        for h in t.handlers:
            if _handler_catches(h, 'AttributeError'):
                fallback_stmts = h.body
                break
        if fallback_stmts:
            break
    assert fallback_stmts is not None, \
        'key_averages 的 try 没有捕 AttributeError ⇒ 真机上直接落进外层 except'

    fb = ' '.join(ast.unparse(s) for s in fallback_stmts)
    assert '_prof_trace_table' in fb, \
        f'AttributeError 分支没落到 _prof_trace_table: {fb}'
    assert 'export_chrome_trace' in SRC, \
        'train_sft.py 里没有 export_chrome_trace —— 退路没有可走的出口'


def test_prof_parse_trace_aggregates_by_cat_and_name():
    """`_prof_parse_trace` 口径：只收带 `dur` 的事件，按 (cat,name) 求和、降序。"""
    from scripts.train_sft import _prof_parse_trace

    # 单位是**微秒**（chrome trace 规范），所以下面取 1e5~1e6 量级 —— 真机上
    # 单卡一步的 kernel 总时长是 10^7 us ≈ 10 s，正是我们要盯的那个数。
    trace = {'traceEvents': [
        {'name': 'MatMul', 'cat': 'kernel', 'dur': 300_000, 'ph': 'X'},
        {'name': 'MatMul', 'cat': 'kernel', 'dur': 100_000, 'ph': 'X'},
        {'name': 'Conv2D', 'cat': 'kernel', 'dur': 500_000, 'ph': 'X'},
        {'name': 'npu_format_cast', 'cat': 'TransData', 'dur': 900_000, 'ph': 'X'},
        # 无 dur 的三类（元数据 / 瞬时 / 计数）必须被排除 —— 否则「按耗时排序」
        # 会退化成「按事件条数排序」，条数最多的 metadata 会霸榜。
        {'name': 'process_name', 'cat': '__metadata__', 'ph': 'M'},
        {'name': 'step', 'cat': 'i', 'ph': 'i', 'ts': 1},
    ]}
    table, cats = _prof_parse_trace(trace, row_limit=3)
    rows = table.splitlines()[1:]
    assert len(rows) == 3, rows
    assert 'TransData / npu_format_cast' in rows[0], rows   # 900_000
    assert 'kernel / Conv2D' in rows[1], rows               # 500_000
    assert 'kernel / MatMul' in rows[2], rows               # 300_000+100_000
    assert rows[2].split()[1] == '400000', rows[2]          # 求和，不是取最大
    assert 'kernel=0.90s' in cats and 'TransData=0.90s' in cats, cats

    # schema 猜错时（一个带 dur 的都没有）必须显式说空表，不能返回一个看起来
    # 正常的空框 —— 运维会把它读成「采到了，但没 kernel」。
    empty, empty_cats = _prof_parse_trace({'traceEvents': [{'name': 'x', 'ph': 'i'}]})
    assert '没有带时长的事件' in empty, empty
    assert empty_cats == '(空)', empty_cats
