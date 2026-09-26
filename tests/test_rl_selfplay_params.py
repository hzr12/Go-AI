"""self_play_game 两调用点的参数映射回归测试（并行 worker 漏传 rollout-steps）。

背景：`_selfplay_worker`（多进程并行）与 main() 串行分支曾各手写一份
`self_play_game` 参数映射，两者不一致且是**静默**分叉：
  - 并行 worker **没传** `rollout_steps`，落回函数签名默认 None（语义 = 2*N*N 步），
    而 `--rollout-steps` 的 argparse 默认是 60 —— 同一份 args，并行与串行跑出
    **不同的对局**（9 盘 162 步 vs 60 步，19 盘 722 步 vs 60 步）；
  - `spec_prefetch` / `use_diverse_rollout` 在 worker 侧是 int 直传、串行侧是 bool；
  - getattr 兜底值散落两处且彼此不同，改默认值要改两个地方。

覆盖:
  - worker 路径转发完整搜索参数（**rollout_steps == 60 是本任务主证据**：
    修复前该键根本不存在 → 行为性红，不是 ImportError）
  - kwargs 键集覆盖 self_play_game 全部关键字参数（锁死「再加参数又漏一处」）
  - helper 读取的 16 个 argparse dest **确实存在于 main() 的 argparse 块**
    （锁键，不只锁值 —— 防 flag 改名被 getattr 兜底静默掩盖，见 _MAPPED_DESTS）
  - 两处调用点形态一致（结构/wiring 断言）
  - 四个 0/1 标志统一归一化成 bool（默认组 + 显式 CLI 值组）
  - max_moves 兜底 3*bs*bs（bs 取自入参，不是 args.board_size）

注意：**P3 RL 重构（去 MCTS，3.0-g）会改写本文件** —— 届时 RL 不再传 MCTS
参数，下面的参数表断言应整体作废重写，而不是逐条修补。

不在测试范围内（只报告、不改，留给 P3 定夺）：
  - `scripts/async_pipeline.py` 的 `_play_one_game` 自己构造 MCTS（不走
    self_play_game），是**第三份**映射，且其 `--mcts-threads` 默认 2（selfplay 侧 3）；
  - `priors_leaf` / `dir_alpha` / `dir_eps` 由 D10 钉死为函数默认，刻意不接线。

全部 CPU、秒级：假 GoAI / 假 self_play_game 不加载真模型，不起 mp 进程。
"""
import argparse
import ast
import difflib
import inspect
import os
import queue
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.selfplay_train as st


# argparse 默认盘（selfplay_train.py 里 main() 的 add_argument 块）。bs=9 取
# --board-size 默认。行号故意不写进注释：文件一改就漂。
_BS = 9

# brief 的取值表 —— **独立真相源**（不是从实现反推的）。
# 真实 argparse 默认值下，两条路径都应产出这一份 kwargs。
_SPEC_TABLE = {
    'board_size': 9,
    'sims': 400,
    'max_moves': 3 * 9 * 9,          # --max-moves 默认 None → 3*bs*bs
    'temperature': 1.0,
    'expand_topk': 64,
    'expand_chunk': 0,
    'use_rollout': False,             # 0/1 → bool
    'rollout_lambda': 0.25,
    'rollout_steps': 60,              # ← 主证据：修复前 worker 根本没这个键
    'leaf_ab_depth': 2,
    'c_puct': 2.0,
    'virtual_loss': 8.0,
    'num_threads': 3,                 # 注意取 --mcts-threads，不是死的 --num-threads
    'spec_prefetch': True,            # 0/1 → bool
    'use_diverse_rollout': False,     # 0/1 → bool
    'vector_backup': True,            # 0/1 → bool
}

# `_selfplay_kwargs` 读取、且**必须**在 main() 的 argparse 块里真实存在的 dest 键。
# 刻意写成字面清单（与 _SPEC_TABLE 无关、也不从 helper 源码反推）：本清单就是
# 「helper 读了哪些 dest」这条契约本身，从实现反推会让「改成 args.x 直读」之类的
# 改动把断言变空转。注意 `num_threads` 映射自 dest `mcts_threads`（--num-threads
# 是死参数），`vector_backup` 映射自 dest `mcts_vector_backup`。
# `board_size` 由入参 bs 供给、不走 getattr，但同样是 dest（worker 侧就取
# `args.board_size`），一并锁住。
_MAPPED_DESTS = (
    'board_size', 'sims', 'max_moves', 'temperature', 'expand_topk', 'expand_chunk',
    'use_rollout', 'rollout_lambda', 'rollout_steps', 'leaf_ab_depth', 'c_puct',
    'virtual_loss', 'mcts_threads', 'spec_prefetch', 'use_diverse_rollout',
    'mcts_vector_backup',
)


def _real_argparse_defaults():
    """从 main() 的源码 AST 抽出 ap.add_argument 的真实 default。

    测试必须用**真实 parser 默认值**构造 args：手抄一份默认值会与 argparse 块
    悄悄漂移，测出来的就不是用户命令行上真正会得到的参数。这里直接读源码。
    抽出的**键**还要过 _require_real_argparse_dests 一关，所以「flag 改名」
    与「argparse 块被搬走」都会立刻暴露，而不是靠 getattr 兜底静默通过。
    """
    src = textwrap.dedent(inspect.getsource(st.main))
    defaults = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            continue
        flag = node.args[0].value            # add_argument 的位置参数即 flag 名
        for kw in node.keywords:
            if kw.arg == 'default':
                try:
                    defaults[flag.lstrip('-').replace('-', '_')] = ast.literal_eval(kw.value)
                except ValueError:
                    raise AssertionError(
                        f'{flag} 的 default 不是字面量，本 helper 需跟进：'
                        f'{ast.dump(kw.value)}')
    return defaults


def _require_real_argparse_dests(defaults):
    """断言 helper 读取的 dest 键 ⊆ 从 main() argparse 块抽出的默认值的键。

    为什么「只锁值」不够：`_selfplay_kwargs` 里每个 `getattr` 都带一个字面量兜底，
    而测试比对的 _SPEC_TABLE 正是这些键的取值 —— flag 改名 + 改默认值
    （如 `--mcts-threads` → `--search-threads`，默认 4）后，兜底给出旧字面量、
    _SPEC_TABLE 照样满足，**全套测试照绿而生产已静默跑在旧默认值上**。
    整块 `add_argument` 被搬出 main()（例如抽成 `build_parser()`）更极端：
    defaults 变空，测试 2/4/5 全靠兜底空转通过。

    故此处按 **dest 名**（`--rollout-steps` → `rollout_steps`）与抽出的默认值比对：
    与 _SPEC_TABLE 无关，不自我一致；缺键立即失败并报出疑似改成的名字。
    """
    if not defaults:
        raise AssertionError(
            '从 main() 源码里一个 add_argument 默认值都没抽到 —— argparse 块大概'
            '已移出 main()（例如抽成 build_parser()），_real_argparse_defaults '
            '的提取源需跟进')
    missing = [d for d in _MAPPED_DESTS if d not in defaults]
    if missing:
        raise AssertionError(
            f'这些 argparse dest 在 main() 的 add_argument 块里不存在: {missing}；'
            f'若 flag 确实改名了，_selfplay_kwargs 的 getattr 兜底与 _MAPPED_DESTS '
            f'必须一起改，否则生产会静默用兜底里的旧字面量。'
            f'疑似改成了: '
            f'{ {d: difflib.get_close_matches(d, defaults, n=2, cutoff=0.5) for d in missing} }'
        )


def _real_args(**overrides):
    """真实 argparse 默认值的 Namespace + 显式覆盖（等价于命令行给了这些 flag）。"""
    defaults = _real_argparse_defaults()
    _require_real_argparse_dests(defaults)
    ns = argparse.Namespace(**defaults)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_worker_path_forwards_full_search_params(monkeypatch):
    """并行 worker 必须转发**全部**搜索参数，rollout_steps 尤其不能漏。

    修复前：worker 逐个手写 kwargs（漏 rollout_steps，位置传 6 个），本断言
    在 dict 比对上报出缺失键 —— 行为性红。
    """
    captured = {}

    def _fake_self_play_game(*a, **kw):
        captured['args'] = a
        captured['kwargs'] = kw
        return [], 0.0

    # device 默认 'auto' → 打掉设备选择，避免测试依赖 CUDA/NPU 可用性
    monkeypatch.setattr(st, '_auto_select_device', lambda: 'cpu')
    monkeypatch.setattr(st, 'GoAI', lambda **kw: object())
    monkeypatch.setattr(st, 'self_play_game', _fake_self_play_game)

    q = queue.Queue()
    st._selfplay_worker(0, 'x.pth', _real_args(), q)

    item = q.get_nowait()
    assert 'error' not in item, f"worker 抛异常: {item.get('error')}"
    # 先比 kwargs 表：修复前这里报「缺键」，就是本任务要修的那个洞
    assert captured['kwargs'] == _SPEC_TABLE, (
        f"缺键: {sorted(set(_SPEC_TABLE) - set(captured['kwargs']))}；"
        f"多键: {sorted(set(captured['kwargs']) - set(_SPEC_TABLE))}；"
        f"取值不符: { {k: (captured['kwargs'].get(k), v) for k, v in _SPEC_TABLE.items() if captured['kwargs'].get(k, object()) != v} }"
    )
    assert captured['kwargs']['rollout_steps'] == 60, \
        '并行 worker 曾漏传 rollout_steps → 落回 None(=2*N*N 步)，与串行跑出不同的对局'
    assert len(captured['args']) == 1, \
        (f"除 ai 外全部参数必须走 kwargs；留位置参数等于给映射再开一个分叉口子："
         f"{captured['args'][1:]}")


def test_kwargs_cover_self_play_game_signature():
    """kwargs 键集必须覆盖 self_play_game 的全部关键字参数。

    锁死「self_play_game 再加一个参数，这里又漏一处」——漏传是静默的，只会让
    并行与串行悄悄跑出不同的对局。
    """
    # ai 由两处调用点按位置传；priors_leaf/dir_alpha/dir_eps 由 D10 钉死为
    # 函数默认（True/0.3/0.25），P3-B 会连同其他 MCTS 参数一起删除，刻意不接线。
    pinned = {'ai', 'priors_leaf', 'dir_alpha', 'dir_eps'}
    expected = set(inspect.signature(st.self_play_game).parameters) - pinned
    keys = set(st._selfplay_kwargs(_real_args(), _BS))
    assert expected <= keys, f"漏传: {sorted(expected - keys)}"


def test_both_call_sites_share_the_mapping():
    """结构/wiring 断言：恰好 2 处调用，两处调用都展开同一个映射。

    这是**结构**断言而非行为断言：它不验证数值，只验证「两个调用点形状一致、
    且没有第三处调用点」（第三处会再引入一份手写映射）。计数只走 AST（Call 节点）——
    源码文本计数会被 docstring/注释里出现的 `self_play_game(` 误报，而 AST 不会。
    """
    src = inspect.getsource(st)
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'self_play_game']
    assert len(calls) == 2, f'应恰好 2 处调用点，实际 {len(calls)}'
    for call in calls:
        seg = ast.get_source_segment(src, call)
        assert '**_selfplay_kwargs(' in seg, f'调用点未走统一映射: {seg}'


def test_flag_params_normalized_to_bool():
    """四个 0/1 标志两条路径类型一致：都是 bool，不是一个 int 一个 bool。"""
    cases = (
        ({}, {'use_rollout': False, 'spec_prefetch': True,
              'use_diverse_rollout': False, 'vector_backup': True}),
        # 等价于命令行 --use-rollout 1 --spec-prefetch 0
        #              --use-diverse-rollout 1 --mcts-vector-backup 0
        ({'use_rollout': 1, 'spec_prefetch': 0, 'use_diverse_rollout': 1,
          'mcts_vector_backup': 0},
         {'use_rollout': True, 'spec_prefetch': False,
          'use_diverse_rollout': True, 'vector_backup': False}),
    )
    for overrides, expect in cases:
        kw = st._selfplay_kwargs(_real_args(**overrides), _BS)
        for key in expect:
            assert type(kw[key]) is bool, \
                f'{key} 未归一化成 bool（实为 {type(kw[key]).__name__}）'
        assert {k: kw[k] for k in expect} == expect, f'0/1 标志取值错误: {overrides}'


def test_max_moves_fallback():
    """--max-moves 默认 None → 3*bs*bs；显式 N → N。bs 取自入参而非 args。"""
    assert st._selfplay_kwargs(_real_args(), _BS)['max_moves'] == 3 * _BS * _BS
    assert st._selfplay_kwargs(_real_args(max_moves=None), 19)['max_moves'] == 3 * 19 * 19, \
        'bs 必须取入参（串行传 ai.board_size），不是 args.board_size'
    assert st._selfplay_kwargs(_real_args(max_moves=123), _BS)['max_moves'] == 123
