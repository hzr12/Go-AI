"""selfplay_train 参数面 + self_play_game 两调用点的参数映射回归测试。

⚠️ 本文件有**两段互不相干**的断言，请连同下面的分节注释一起读：
  · 第 1 段（_SPEC_TABLE / _MAPPED_DESTS ...）：self_play_game 两调用点的参数映射
    回归，原用于钉「并行 worker 漏传 rollout-steps」。
  · 第 2 段（_EXPECTED_PARAM_SURFACE ...）：P3.0 的 CLI 参数面清单（PPO/KL 三参数
    + --epochs 语义改写）。
两段都会在 P3 RL 重构（去 MCTS）时被 P3-B 整体作废重写，见文末「不在测试范围内」。

第 1 段的背景：`_selfplay_worker`（多进程并行）与 main() 串行分支曾各手写一份
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
  - **P3.0 参数面**：main() 的 argparse 块逐项清单（56 项）、PPO/KL 三参数的
    路线图默认值、`--epochs` 的 PPO epoch 语义

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
    'max_moves': 3 * 9 * 9,          # --max-moves 默认 None → 3*bs*bs
    'temperature': 1.0,
    # ---- N 步 minimax 推演（去 MCTS 后新增的行为参数）----
    'lookahead_depth': 2,
    'lookahead_topk': 12,
    'lookahead_width': 4,
    'lookahead_temp': 0.2,
    'mix': 0.1,                     # --lookahead-mix 默认值（DEFAULT_MIX）
}

# ⚠ 2026-09-30（RL 去 MCTS）：`_SPEC_TABLE` 里**不再有任何 MCTS 参数**。
# 原来 17 个（sims / expand_topk / expand_chunk / c_puct / virtual_loss /
# mcts_threads / num_threads / spec_prefetch / use_rollout / rollout_lambda /
# rollout_steps / leaf_ab_depth / use_diverse_rollout / mcts_vector_backup /
# priors_leaf / dir_alpha / dir_eps …）**必须逐个消失**，而不是变成「默认值也
# 照样转发」——RL 里已没有接收方，转发等于谎报生效。
# 「忽略但留痕」由 `st.ARCHIVED_SEARCH_ARGS` + `_log_archived_search_args` 负责，
# 由 `tests/test_rl_search_archived.py` 单独钉住。

# `_selfplay_kwargs` 读取、且**必须**在 main() 的 argparse 块里真实存在的 dest 键。
# 刻意写成字面清单（与 _SPEC_TABLE 无关、也不从 helper 源码反推）：本清单就是
# 「helper 读了哪些 dest」这条契约本身，从实现反推会让「改成 args.x 直读」之类的
# 改动把断言变空转。注意 `mix` 映射自 dest `lookahead_mix`（CLI 上带前缀，
# kwargs 里是短名）。`board_size` 由入参 bs 供给、不走 getattr，但同样是 dest
# （worker 侧就取 `args.board_size`），一并锁住。
_MAPPED_DESTS = (
    'board_size', 'max_moves', 'temperature',
    'lookahead_depth', 'lookahead_topk', 'lookahead_width',
    'lookahead_temp', 'lookahead_mix',
)


#: 去 MCTS（2026-09-30）新增的 5 个推演参数：从 `_EXPECTED_PARAM_SURFACE` 里
#: 扣掉它们才能还原 P3.0 之前的 53 项基线。
_LOOKAHEAD_NEW_PARAMS = (
    ('--lookahead-depth', 'int', 2),
    ('--lookahead-topk', 'int', 12),
    ('--lookahead-width', 'int', 4),
    ('--lookahead-temp', 'float', 0.2),
    ('--lookahead-mix', 'float', st.DEFAULT_MIX),
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
                    # 允许 default 引用模块常量（如 --lookahead-mix 的
                    # DEFAULT_MIX）：从 st 的命名空间取值，而不是逼实现方把
                    # 常量抄成字面量（那正是「两份真相源」的开始）。
                    if isinstance(kw.value, ast.Name) and hasattr(st, kw.value.id):
                        defaults[flag.lstrip('-').replace('-', '_')] = getattr(st, kw.value.id)
                    else:
                        raise AssertionError(
                            f'{flag} 的 default 既不是字面量也不是 st 的模块常量，'
                            f'本 helper 需跟进：{ast.dump(kw.value)}')
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
    """并行 worker 必须转发**全部**落子参数，且不得再出现任何 MCTS 参数。

    原断言守的是「worker 漏传 rollout_steps ⇒ 串行/并行跑出不同对局」。去 MCTS
    后同一类洞换了形状：新增的 4 个 lookahead 参数（+ mix）任一漏传，都会让并行
    与串行用不同的推演深度/宽度 ⇒ 又是两批不同的对局，而且是**静默**的。
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
    assert len(captured['args']) == 1, \
        (f"除 ai 外全部参数必须走 kwargs；留位置参数等于给映射再开一个分叉口子："
         f"{captured['args'][1:]}")


def test_no_mcts_param_survives_in_kwargs():
    """`_selfplay_kwargs` 的键里不得有任何 MCTS 参数（RL 里已无接收方）。

    这是去 MCTS 的**核心**断言：留着它们会让「我 --sims 48 起得快」这种错觉
    成立（现在每手只做 depth 次批量前向，与 sims 无关）。
    """
    keys = set(st._selfplay_kwargs(_real_args(), _BS))
    forbidden = set(st.ARCHIVED_SEARCH_ARGS) | {
        'expand_chunk_alpha', 'expand_chunk_beta', 'mcts_threads',
        'num_threads', 'rollout_threads', 'dir_alpha', 'dir_eps', 'sims',
    }
    leaked = keys & forbidden
    assert not leaked, (
        '去 MCTS 后 kwargs 里仍有 MCTS 参数：%s（RL 不再构造 MCTS，转发即谎报）'
        % sorted(leaked))
    # 反向：新参数必须在
    for need in ('lookahead_depth', 'lookahead_topk', 'lookahead_width',
                 'lookahead_temp', 'mix'):
        assert need in keys, '落子参数漏了 %s' % need


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
    """落子参数的类型必须**稳定**：残缺 Namespace（int）与完整 Namespace 一致。

    原断言守的是四个 0/1 搜索 flag 的 bool 归一化。去 MCTS 后那些 flag 已归档，
    这里换成同一类风险的残留入口：`--temperature` 等**连续参数**若被写成 int
    而下游按 float 用，症状是采样分布悄悄变差（不报错）。所以断言映射出来的
    值与 argparse 默认**同类型**。
    """
    import argparse as _ap
    ns = _real_args()
    kw = st._selfplay_kwargs(ns, _BS)
    for key in ('temperature', 'lookahead_temp', 'mix'):
        val = kw[key]
        dflt = getattr(ns, 'lookahead_mix' if key == 'mix' else key)
        assert type(val) is type(dflt), \
            (f'{key} 的类型随 Namespace 变化：{type(val).__name__} vs '
             f'{type(dflt).__name__}（残缺 args 会走 getattr 兜底）')
    # 四个 0/1 搜索 flag 已归档：必须**不在** kwargs 里，也不再需要 bool 归一化
    for gone in ('use_rollout', 'spec_prefetch', 'use_diverse_rollout',
                 'vector_backup'):
        assert gone not in kw, '归档的 0/1 flag 又回来了：%s' % gone


def test_archived_search_args_are_kept_but_inert():
    """17 个搜索参数：argparse 里**仍可解析**（旧命令行不改），但不再进 kwargs。

    「忽略但留痕」（D1 口径）：不能报错、也不能生效，只在启动时打一行汇总。
    """
    kw = st._selfplay_kwargs(_real_args(**{k: 7 for k in
                                           st.ARCHIVED_SEARCH_ARGS}), _BS)
    for name in st.ARCHIVED_SEARCH_ARGS:
        assert name not in kw, '%s 已归档却仍在生效路径上' % name
    # 归档参数显式传值也不改变落子参数（真·无效）
    kw2 = st._selfplay_kwargs(_real_args(), _BS)
    assert kw == kw2, '传了归档参数后落子参数变了 ⇒ 它们其实还在生效'


def test_max_moves_fallback():
    """--max-moves 默认 None → 3*bs*bs；显式 N → N。bs 取自入参而非 args。"""
    assert st._selfplay_kwargs(_real_args(), _BS)['max_moves'] == 3 * _BS * _BS
    assert st._selfplay_kwargs(_real_args(max_moves=None), 19)['max_moves'] == 3 * 19 * 19, \
        'bs 必须取入参（串行传 ai.board_size），不是 args.board_size'
    assert st._selfplay_kwargs(_real_args(max_moves=123), _BS)['max_moves'] == 123


# --------------------------------------------------------------------------- #
# 第 2 段：P3.0 的 CLI 参数面
#
# P3.0 **只加参数、不实现损失**。所以本段钉的是「参数面」这一层契约：三个 PPO/KL
# 参数存在且默认值等于路线图 D13 ⑧、--epochs 的语义已改写成 PPO epoch 数。
# 损失侧（P3-C 策略侧裁剪+KL / P3-D value MSE+value clipping）的断言由那两个任务
# 各自新增（路线图指定的 tests/test_rl_ppo.py / test_rl_kl_penalty.py /
# test_rl_value_loss.py），**不在本文件**。
#
# 为什么逐项列清单而不是只数个数：只锁 `len()==56` 时，「删一个旧的 + 加两个新的」
# 恰好抵消，测试照绿而参数面已经变了。逐项 (flag, type, default) 相等才能同时锁住
# 「没删」「没改名」「没改默认值」「没多加」四件事。
# --------------------------------------------------------------------------- #

# 「没写 default=」的哨兵。与 None 区分开：--max-moves 写了 default=None，
# 而 --device/--ver 是**根本没有** default= 关键字。两者语义不同（前者显式 None，
# 后者 argparse 自己给 None），混为一谈会让「删掉 default=」这类改动静默通过。
_NO_DEFAULT = object()

# main() 的 argparse 块**完整**清单，逐项 (flag, type, default)。
# 分组只为可读性；分组边界对应源码里的注释块，P3-B 删 MCTS 参数时按组整段删。
_EXPECTED_PARAM_SURFACE = (
    # ---- 训练 / 生成主参数 ----
    ('--model', 'str', None),
    ('--board-size', 'int', 9),
    ('--iters', 'int', 5),
    ('--games', 'int', 4),
    ('--sims', 'int', 400),
    ('--max-moves', 'int', None),
    ('--temperature', 'float', 1.0),
    # ---- 2026-09-30 去 MCTS：新增 5 个 N 步 minimax 推演参数 ----
    ('--lookahead-depth', 'int', 2),
    ('--lookahead-topk', 'int', 12),
    ('--lookahead-width', 'int', 4),
    ('--lookahead-temp', 'float', 0.2),
    ('--lookahead-mix', 'float', st.DEFAULT_MIX),
    ('--buffer-size', 'int', 500),
    ('--batch-size', 'int', 256),
    # --epochs：P3.0 起语义 = PPO 更新轮数（路线图 D13 ⑦，不新增 --ppo-epochs）
    ('--epochs', 'int', 2),
    ('--lr', 'float', 0.001),
    ('--weight-decay', 'float', 0.0001),
    ('--value-lr-mult', 'float', 0.5),
    ('--result-queue-max', 'int', 100),
    ('--streaming', 'int', 0),
    ('--no-persist', 'int', 0),
    ('--ddp', 'int', 0),
    ('--device', None, 'auto'),
    ('--no-augment', 'int', 0),
    ('--use-ema', 'int', 0),
    ('--clip-grad', 'float', 1.0),
    ('--grad-accum-steps', 'int', 1),
    ('--out', 'str', 'models/az'),
    ('--save-every', 'int', 1),
    ('--parallel-games', 'int', 8),
    ('--onnx-model', 'str', None),
    ('--async-pipeline', 'int', 0),
    ('--games-per-iter', 'int', 10),
    ('--swanlab', 'int', 0),
    ('--swanlab-api-key', 'str', ''),
    ('--ver', None, 'rl'),
    ('--c2net', 'int', 0),
    # ---- MCTS（路线图 D11：P3-B 删 19 个参数，含本组 17 个）----
    ('--expand-topk', 'int', 64),
    ('--expand-chunk', 'int', 0),
    ('--c-puct', 'float', 2.0),
    ('--virtual-loss', 'float', 8.0),
    ('--num-threads', 'int', 8),
    ('--spec-prefetch', 'int', 1),
    ('--leaf-ab-depth', 'int', 2),
    ('--dynamic-topk', 'int', 0),
    ('--dynamic-virtual-loss', 'int', 0),
    ('--policy-pruning-thresh', 'float', 0.01),
    ('--use-diverse-rollout', 'int', 0),
    ('--use-rollout', 'int', 0),
    ('--rollout-lambda', 'float', 0.25),
    ('--rollout-steps', 'int', 60),
    ('--mcts-threads', 'int', 3),
    ('--batch-cap', 'int', 64),
    ('--mcts-vector-backup', 'int', 1),
    # ---- TD 价值标签 ----
    ('--td', 'int', 1),
    ('--td-steps', 'int', 3),
    ('--td-alpha-init', 'float', 0.2),
    ('--td-alpha-end', 'float', 0.9),
    # ---- P3.0 新增：PPO / KL（策略侧由 P3-C 消费，value 侧由 P3-D 消费）----
    ('--ppo-clip', 'float', 0.2),
    ('--kl-coef', 'float', 0.01),
    ('--kl-target', 'float', 0.01),
)

# P3.0 的**增量**：恰好 3 个。路线图 D11 的算式是 53 →(P3-B 删 19)→ 34 →(P3.0 +3)→ 37；
# P3.0 落地时 P3-B 还没跑，故当下是 53 → 56，P3-B 之后才是 34+3=37。
# 下方 test_delta_is_exactly_three_ppo_kl_params 把这条算式的每一段都断言住。
_P3_0_NEW_PARAMS = (
    ('--ppo-clip', 'float', 0.2),
    ('--kl-coef', 'float', 0.01),
    ('--kl-target', 'float', 0.01),
)

# 路线图 D11 钉死的 MCTS 删除条数。P3.0 **不**执行删除（P3-B 的活），这里只把
# 37 = 34 + 3 这个终点记成常量，使 P3-B 落地后本文件能自证总数对得上。
_ROADMAP_POST_P3_B_TOTAL = 37
_MCTS_PARAMS_TO_DELETE = 19


def _real_param_surface():
    """从 main() 源码 AST 抽出 [(flag, type, default, help)]，按源码行号排序。

    与 _real_argparse_defaults 的分工：那个只抽 default（给 _real_args 造 Namespace），
    且**不区分**「没写 default=」与「default=None」；本函数要锁参数面（flag 名 +
    type + default 三者齐全），故必须把 type/无 default 哨兵一起抽出来。
    ast.walk 不保证源码顺序，按 lineno 排一次，失败信息才可读。
    """
    src = textwrap.dedent(inspect.getsource(st.main))
    found = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            continue
        flag = node.args[0].value
        type_name = None
        default = _NO_DEFAULT
        help_text = None
        for kw in node.keywords:
            if kw.arg == 'type':
                type_name = getattr(kw.value, 'id', None)
            elif kw.arg == 'default':
                try:
                    default = ast.literal_eval(kw.value)
                except ValueError:
                    # 允许 default 引用模块常量（--lookahead-mix 的 DEFAULT_MIX）
                    default = (getattr(st, kw.value.id, _NO_DEFAULT)
                               if isinstance(kw.value, ast.Name) else _NO_DEFAULT)
            elif kw.arg == 'help':
                help_text = ast.literal_eval(kw.value)
        found.append((node.lineno, (flag, type_name, default, help_text)))
    return [row for _, row in sorted(found, key=lambda r: r[0])]


def _surface_by_flag():
    return {flag: (type_name, default, help_text)
            for flag, type_name, default, help_text in _real_param_surface()}


def test_param_surface_matches_itemised_list_exactly():
    """参数面逐项 == 清单，且总数 == 清单长度（防「删一个 + 加一个」互相抵消）。"""
    surface = _surface_by_flag()
    expected = {flag: (type_name, default)
                for flag, type_name, default in _EXPECTED_PARAM_SURFACE}
    assert len(_EXPECTED_PARAM_SURFACE) == len(expected), \
        '清单自身有重复 flag，逐项表失去意义'
    missing = sorted(set(expected) - set(surface))
    extra = sorted(set(surface) - set(expected))
    wrong = {f: (surface[f][0], surface[f][1], expected[f]) for f in sorted(set(expected) & set(surface))
             if (surface[f][0], surface[f][1]) != expected[f]}
    assert not missing, f'参数面缺这些 flag: {missing}'
    assert not extra, (f'参数面多出这些 flag: {extra}；'
                       f'路线图 D11 只允许 P3.0 +3 个参数，不许顺手多加')
    assert not wrong, (f'这些参数的 type/default 与清单不符（实为 type, default, 期望）: '
                       f'{wrong}')
    assert len(surface) == len(_EXPECTED_PARAM_SURFACE) == 61, \
        f'参数面应恰好 61 项（P3.0 前 53 + 3，去 MCTS 后 +5 个推演参数），实为 {len(surface)}'


def test_delta_is_exactly_three_ppo_kl_params():
    """P3.0 的增量恰好是那 3 个 PPO/KL 参数；去 MCTS 的增量恰好是 5 个推演参数。

    把路线图 D11 的算式整条钉住：
      53（P3.0 前） → 56（P3.0 后） →(去 MCTS +5 推演参数)→ 61（现在）
    ⚠ 2026-09-30：MCTS 参数**没有**被删除 —— 它们按 D1「忽略但留痕」保留在
    argparse 里（旧命令行一个字不用改），只是不再转发给任何接收方。所以这里
    不再断言「56 − 19 = 37」，而是断言「旧参数一个没少 + 新增恰好 5 个」。
    """
    surface = _surface_by_flag()
    before = (set(_EXPECTED_PARAM_SURFACE) - set(_P3_0_NEW_PARAMS)
              - set(_LOOKAHEAD_NEW_PARAMS))
    assert len(before) == 53, f'P3.0 前的基线应是 53 项，实为 {len(before)}'
    for flag, type_name, default in _P3_0_NEW_PARAMS:
        got = surface.get(flag, (None, _NO_DEFAULT))
        assert (got[0], got[1]) == (type_name, default), \
            f'{flag} 实为 type={got[0]} default={got[1]}，' \
            f'期望 type={type_name} default={default}'
    for flag, type_name, default in _LOOKAHEAD_NEW_PARAMS:
        got = surface.get(flag, (None, _NO_DEFAULT))
        assert (got[0], got[1]) == (type_name, default), \
            f'{flag} 实为 type={got[0]} default={got[1]}，' \
            f'期望 type={type_name} default={default}'
    assert len(surface) - len(before) == 8, \
        f'增量应是 3 个 PPO/KL + 5 个推演参数（{len(surface)} - {len(before)}）'
    assert not {f for f, _, _ in before} - set(surface), \
        '不允许删任何既有参数（MCTS 参数按 D1 保留定义、只归档）'


def test_ppo_clip_default_is_roadmap_epsilon():
    """--ppo-clip 0.2：PPO 裁剪范围 ε（路线图 D13 ①⑧）。P3-D 的 value clipping 复用它。

    ε 的断言只看 help 的**首句**（第一个「：」之前）：完整 help 里 `clip(r,1-ε,1+ε)`
    与末句「复用同一个 ε」都含 ε，查整段的话把首句的 ε 说明删掉也照样绿。
    """
    type_name, default, help_text = _surface_by_flag()['--ppo-clip']
    assert type_name == 'float', f'--ppo-clip 应为 float，实为 {type_name}'
    assert default == 0.2, f'--ppo-clip 默认应为 0.2，实为 {default}'
    assert help_text, '--ppo-clip 必须有 help（否则 --help 里读不到它是干什么的）'
    head = help_text.split('：')[0]
    assert 'ε' in head or 'epsilon' in head.lower(), \
        f'--ppo-clip 的 help 首句必须点明它是 PPO 裁剪范围 ε，实为: {head!r}'


def test_kl_coef_default_is_initial_beta():
    """--kl-coef 0.01 是 β 的**初值**：help 必须写明「初值」与自适应，否则 P3-C 会误当常数。"""
    type_name, default, help_text = _surface_by_flag()['--kl-coef']
    assert type_name == 'float', f'--kl-coef 应为 float，实为 {type_name}'
    assert default == 0.01, f'--kl-coef 默认应为 0.01，实为 {default}'
    assert '初值' in help_text, \
        '--kl-coef 的 help 必须写明它是 β 的初值（P3-C 会做 β 自适应）'
    assert 'β' in help_text, '--kl-coef 的 help 应写明它就是 KL 惩罚系数 β'


def test_kl_target_default_and_two_x_abort_rule():
    """--kl-target 0.01，且 help 必须写明 running-KL > 2×本值 时提前中止（路线图 D13 ⑥）。"""
    type_name, default, help_text = _surface_by_flag()['--kl-target']
    assert type_name == 'float', f'--kl-target 应为 float，实为 {type_name}'
    assert default == 0.01, f'--kl-target 默认应为 0.01，实为 {default}'
    assert '2' in help_text and '×' in help_text, \
        '--kl-target 的 help 必须写明「2×」提前中止因子（P3-C 的硬信任域约束靠它）'
    assert '中止' in help_text or 'abort' in help_text.lower(), \
        '--kl-target 的 help 必须写明超限会提前中止本轮剩余 minibatch'


def test_epochs_semantics_is_ppo_epoch_count():
    """--epochs 保留（不新增 --ppo-epochs），语义改为 PPO 更新轮数。

    路线图 D13 ⑦：PPO 更新轮数 = 现有 --epochs。type/默认值不变（int / 2），
    变的是 help 文本 —— 这是 P3-C 唯一能看出「外层 for _ in range(args.epochs)
    就是 PPO epoch 循环」的锚点。

    断言只查**首个分句**（第一个「（」之前）而不是整段 help：help 末尾那句
    「（原「每轮迭代训练遍数」；…）」是刻意保留的沿革说明，整段查 'PPO' 会被它
    连带满足 —— 那样把主句改回旧语义（全段再无 PPO，但沿革说明里还有）也会照绿。
    """
    type_name, default, help_text = _surface_by_flag()['--epochs']
    assert type_name == 'int', f'--epochs 应为 int，实为 {type_name}'
    assert default == 2, f'--epochs 默认仍应是 2（P3.0 只改语义不改默认值），实为 {default}'
    head = help_text.split('（')[0]
    assert 'PPO' in head, (
        f'--epochs help 的**主句**必须声明它是 PPO 更新轮数（路线图 D13 ⑦），实为: {head!r}')
    assert '轮' in head or 'epoch' in head.lower(), \
        f'--epochs help 的主句必须说明它是个「轮数」，实为: {head!r}'
    assert '--ppo-epochs' not in ' '.join(
        h for _, _, _, h in _real_param_surface() if h), \
        '路线图 D13 ⑦ 明令不新增 --ppo-epochs'


def test_new_params_are_not_wired_into_mcts_or_kwargs_mapping():
    """三个新参数只属损失侧：不得进 _selfplay_kwargs（那是 self_play_game 的搜索参数）。

    误接的后果是静默的：_selfplay_kwargs 的返回值被当作 self_play_game 的关键字
    参数传下去，多一个键就是 TypeError（吵闹但安全）；而若有人图省事给
    self_play_game 加形参接收它们，就会把训练超参漏进搜索路径。故显式钉住。
    """
    keys = set(st._selfplay_kwargs(_real_args(), _BS))
    for flag in ('--ppo-clip', '--kl-coef', '--kl-target'):
        dest = flag.lstrip('-').replace('-', '_')
        assert dest not in keys, f'{dest} 不该出现在 _selfplay_kwargs 里（损失侧参数）'
