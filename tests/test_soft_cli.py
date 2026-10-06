"""软标签的 CLI 四参数与接线（软标签接入 A4）。

四个旗
------
=========================  =========  ==========================================
旗                         默认        语义
=========================  =========  ==========================================
``--soft-index``           ``None``   软标签索引 npz；给了就启用软 CE
``--soft-weight``          ``1.0``    软项的全局缩放
``--soft-only-sampling``   ``0``(关)  行索引空间收窄到 ``soft_row >= 0``
``--soft-every``           ``1``      每 N 步用一个软批
=========================  =========  ==========================================

本文件钉四件事
--------------
1. 四个旗**存在**且默认值逐字正确（从真 argparse 重放，不是抠源码字面量）。
2. ``--soft-index`` 给了 ⇒ 生效 kind 是 ``soft_ce``；没给 ⇒ 原样是 ``ce``，
   且**旧路径数值逐位不变**（这是段 1 通路基线的保证）。
3. ``--soft-every`` 的节奏（哪些步走软 CE）与 ``--soft-only-sampling`` 的收窄
   语义，包括两个必须**报错**（而不是静默退回）的坏组合。
4. 挂了软标签但索引内容不对（缺键 / 越界 / 错盘口）必须报错，不能静默 no-op。

跑：pytest tests/test_soft_cli.py -v
"""
import argparse
import ast
import contextlib
import inspect
import io
import os
import sys
import textwrap

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.train_sft as t  # noqa: E402
from src.data.dataset import SupervisedDataset  # noqa: E402

BS = 9
A = BS * BS + 1

SOFT_FLAGS = ('--soft-index', '--soft-weight', '--soft-only-sampling',
              '--soft-every')
EXPECTED_DEFAULTS = {'--soft-index': None, '--soft-weight': 1.0,
                     '--soft-only-sampling': 0, '--soft-every': 1}


# --------------------------------------------------------------------------- #
# argparse 工具（与 tests/test_huber_loss.py 同一套口径：重放，不执行 main()）
# --------------------------------------------------------------------------- #
def _main_ast():
    return ast.parse(textwrap.dedent(inspect.getsource(t.main)))


def _add_argument_kwargs():
    """{旗名: {kwarg: AST node}} —— main() 里真实的 add_argument 调用。"""
    out = {}
    for call in ast.walk(_main_ast()):
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'add_argument' and call.args):
            continue
        a0 = call.args[0]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
            out[a0.value] = {kw.arg: kw.value for kw in call.keywords}
    return out


def _replay_parser():
    """重放 main() 里全部 add_argument 得到真 parser（不执行 main）。"""
    ap = argparse.ArgumentParser()
    scope = dict(vars(t))
    scope['ap'] = ap
    for node in ast.walk(_main_ast()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            expr = ast.Expression(body=node)
            ast.copy_location(expr, node)
            exec(compile(expr, '<train_sft argparse>', 'eval'), scope)
    return ap


def _help_entry(ap, metavar):
    """从 `--help` 的 options 段里取出某一个选项的整条 help（压平换行）。"""
    txt = ap.format_help()
    lines = txt[txt.rindex('\noptions:') + 1:].splitlines()
    out, on = [], False
    for ln in lines[1:]:
        if ln.startswith('  -') and not ln.startswith('   '):
            if on:
                break
            on = metavar in ln
            if on:
                out.append(ln)
            continue
        if on:
            out.append(ln)
    assert out, f'--help 的 options 段里找不到 {metavar}'
    return ' '.join(' '.join(out).split())


# --------------------------------------------------------------------------- #
# 假数据：造一个带软标签的小数据集 + 写一份索引 npz
# --------------------------------------------------------------------------- #
def _toy_dataset(n=12, soft_rows=(1, 4, 7), seed=0, board_size=BS):
    rng = np.random.default_rng(seed)
    n_sq = board_size * board_size
    data = {
        'boards': rng.integers(-1, 2, (n, board_size, board_size)).astype(np.int8),
        'my_hist': np.full((n, 3), -1, dtype=np.int16),
        'op_hist': np.full((n, 3), -1, dtype=np.int16),
        'ko': np.full(n, -1, dtype=np.int16),
        'moves': rng.integers(0, n_sq, size=n).astype(np.int16),
        'values': rng.choice([-1, 1], size=n).astype(np.int8),
        'to_play': rng.choice([-1, 1], size=n).astype(np.int8),
        'game_ids': np.repeat(np.arange(n // 4), 4).astype(np.int32),
    }
    sp = rng.random((len(soft_rows), n_sq + 1)).astype(np.float32)
    sp /= sp.sum(axis=1, keepdims=True)
    ds = SupervisedDataset(data, n_channels=12,
                           soft_idx=np.asarray(soft_rows, np.int64),
                           soft_policy=sp)
    return ds, data, sp


def _write_index(path, idx, policy):
    np.savez(path, idx=np.asarray(idx, np.int64),
             policy=np.asarray(policy, np.float16))
    return str(path)


# --------------------------------------------------------------------------- #
# 1. 四个旗存在 + 默认值
# --------------------------------------------------------------------------- #
def test_four_soft_flags_exist_with_expected_defaults():
    kw = _add_argument_kwargs()
    missing = [f for f in SOFT_FLAGS if f not in kw]
    assert not missing, f'A4 的四个软标签旗没注册全：缺 {missing}'
    for flag, want in EXPECTED_DEFAULTS.items():
        got = ast.literal_eval(kw[flag]['default'])
        assert got == want and type(got) is type(want), \
            f'{flag} 默认应是 {want!r}（{type(want).__name__}），实得 {got!r}'


def test_soft_flag_types_and_choices():
    kw = _add_argument_kwargs()
    assert isinstance(kw['--soft-weight']['type'], ast.Name) \
        and kw['--soft-weight']['type'].id == 'float', '--soft-weight 应是 float'
    assert ast.literal_eval(kw['--soft-only-sampling']['choices']) == [0, 1], \
        '--soft-only-sampling 沿用本仓的 0/1 布尔约定（与 --use-amp 等一致）'
    assert isinstance(kw['--soft-every']['type'], ast.Name) \
        and kw['--soft-every']['type'].id == '_at_least_one', \
        '--soft-every 的 type 应是 _at_least_one（>=1 校验）'
    assert callable(t._at_least_one)


def test_soft_flag_defaults_from_real_parser():
    ap = _replay_parser()
    ns = ap.parse_args(['--data', 'dummy.npz'])
    assert ns.soft_index is None
    assert ns.soft_weight == 1.0 and isinstance(ns.soft_weight, float)
    assert ns.soft_only_sampling == 0
    assert ns.soft_every == 1 and isinstance(ns.soft_every, int)


def test_soft_flags_accept_explicit_values():
    ap = _replay_parser()
    ns = ap.parse_args(['--data', 'd', '--soft-index', 'idx.npz',
                        '--soft-weight', '0.25', '--soft-only-sampling', '1',
                        '--soft-every', '4'])
    assert (ns.soft_index, ns.soft_weight, ns.soft_only_sampling,
            ns.soft_every) == ('idx.npz', 0.25, 1, 4)


def test_soft_every_rejects_below_one_at_parse_time():
    """`--soft-every 0` 必须在**解析期**被拒（否则 `step % 0` 在训练中炸）。"""
    ap = _replay_parser()
    for bad in ('0', '-1', 'abc', ''):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
            ap.parse_args(['--data', 'd', '--soft-every', bad])
        assert 'soft-every' in err.getvalue(), \
            f'--soft-every {bad!r} 被拒但没指出是哪个旗: {err.getvalue()!r}'
    assert t._at_least_one('1') == 1
    with pytest.raises(argparse.ArgumentTypeError):
        t._at_least_one('0')


def test_soft_only_sampling_rejects_other_values():
    ap = _replay_parser()
    err = io.StringIO()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
        ap.parse_args(['--data', 'd', '--soft-only-sampling', '2'])
    assert 'invalid choice' in err.getvalue()


def test_soft_help_documents_the_danger():
    """help 必须写明「混训会让 index 0 折中」与「索引陈旧 ≠ 没标签」。

    理由同 `--policy-loss`：只钉默认值的话，下一个人照样能把 help 写空，
    而这两条正是**从代码上看不出来**的语义（一个是退路表，一个是缓存失效）。
    """
    ap = _replay_parser()
    only = _help_entry(ap, '--soft-only-sampling {0,1}')
    assert '折中' in only, f'--soft-only-sampling 的 help 没写「混训 ⇒ 折中」: {only}'
    assert '--soft-index' in only, \
        f'--soft-only-sampling 的 help 没提它依赖 --soft-index: {only}'
    idx = _help_entry(ap, '--soft-index SOFT_INDEX')
    assert '陈旧' in idx, f'--soft-index 的 help 没写「索引陈旧」: {idx}'
    every = _help_entry(ap, '--soft-every SOFT_EVERY')
    assert 'one-hot' in every, f'--soft-every 的 help 没写非软步按 one-hot 训: {every}'


# --------------------------------------------------------------------------- #
# 2. 给了 --soft-index 就启用软 CE；没给就逐位走旧路径
# --------------------------------------------------------------------------- #
def test_kind_is_soft_ce_when_index_given_and_ce_otherwise():
    assert t.resolve_policy_loss_kind('ce', 'idx.npz') == 'soft_ce'
    assert t.resolve_policy_loss_kind('ce', None) == 'ce'
    # 段 1 常用的硬目标原样透传
    assert t.resolve_policy_loss_kind('ce', '') == 'ce'
    with pytest.raises(SystemExit, match='soft-index'):
        t.resolve_policy_loss_kind('soft_ce', None)
    with pytest.raises(SystemExit, match='互斥'):
        t.resolve_policy_loss_kind('huber', 'idx.npz')


def test_soft_ce_reaches_the_loss_when_index_given(tmp_path):
    """端到端：索引 → 挂载 → `labels=True` → 软 CE 的数值闭环。"""
    ds, data, sp = _toy_dataset()
    path = _write_index(tmp_path / 'idx.npz', [1, 4, 7], sp)
    ds2, _, _ = _toy_dataset()
    diag = t.attach_soft_index(ds2, path)
    assert diag['n_soft'] == 3 and diag['n_rows'] == 12
    kind = t.resolve_policy_loss_kind('ce', path)
    assert kind == 'soft_ce'

    idxs = np.arange(12)
    _, moves, _, lbl = ds2.sample_batch_numpy(idxs, augment=False, labels=True)
    logits = torch.from_numpy(
        np.random.default_rng(3).standard_normal((12, A)).astype(np.float32))
    loss = t.compute_policy_loss(logits, torch.from_numpy(moves), kind,
                                 soft=torch.from_numpy(lbl['soft']),
                                 soft_mask=torch.from_numpy(lbl['soft_mask']),
                                 soft_weight=1.0)
    want = float(t.soft_cross_entropy(
        logits, torch.from_numpy(lbl['soft']),
        torch.from_numpy(lbl['soft_mask'])))
    assert float(loss) == pytest.approx(want, rel=1e-6)
    # 软 CE 走的确实是软标签而不是 one-hot：把 soft 换成随机噪声，loss 必变
    noisy = t.compute_policy_loss(
        logits, torch.from_numpy(moves), kind,
        soft=torch.rand(12, A), soft_mask=torch.from_numpy(lbl['soft_mask']))
    assert float(noisy) != pytest.approx(want, rel=1e-3)


def test_old_path_is_bitwise_unchanged_without_soft_index():
    """ **没给 `--soft-index` ⇒ 逐位不变**（段 1 通路基线的保证）。

    三层都查：数据（三元组、不带 dict）、损失（与裸 `F.cross_entropy` 逐位）、
    以及「装配软项 kwarg」这一步本身在软标签缺席时必须是**空 dict**
    （传 `soft=None/soft_mask=None/soft_weight=1.0` 虽不改数值，但会让调用点
    多三个实参 —— 段 1 是 34.2M 行的热路径，不该为软标签多搬任何东西）。
    """
    ds, _, _ = _toy_dataset()
    idxs = np.arange(8)
    out = ds.sample_batch_numpy(idxs, augment=False)
    assert len(out) == 3, 'labels=False 必须仍是三元组'
    ref = ds.sample_batch_numpy(idxs, augment=False, labels=True)
    for a, b, name in zip(out, ref[:3], ('states', 'moves', 'values')):
        assert a.tobytes() == b.tobytes(), f'{name} 在 labels 分支里变了'

    # 装配逻辑：软项缺席 ⇒ 空 dict（与 A4 之前 main() 里的实参集合逐字相同）
    soft_kwargs = {} if ref[3] is None else {'soft': ref[3]['soft']}
    logits = torch.from_numpy(
        np.random.default_rng(9).standard_normal((8, A)).astype(np.float32))
    moves = torch.from_numpy(out[1])
    for eps in (0.0, 0.1):
        got = t.compute_policy_loss(logits, moves, t.resolve_policy_loss_kind(
            'ce', None), label_smoothing=eps, huber_beta=0.5, **soft_kwargs)
        want = F.cross_entropy(logits, moves, label_smoothing=eps)
        assert np.asarray(got.detach()).tobytes() == \
            np.asarray(want.detach()).tobytes(), \
            f'eps={eps} 时旧路径不再逐位等于 F.cross_entropy'


def test_npz_without_soft_index_never_asks_for_labels():
    """`--soft-index` 缺席 ⇒ 数据侧**根本不请求** labels（预取器 labels=False）。

    这条是 A3+A4 的交界：请求 labels 会让每个 batch 多搬一份 dict（含
    `soft` (B,362) f32），在 34.2M 行 × 每步都跑的段 1 上是白付的。
    """
    src = inspect.getsource(t.main)
    assert 'labels=_soft_on' in src, '预取器必须按 --soft-index 决定是否取 labels'
    assert 'pf.next()' in src, 'labels=False 时 next() 必须是老的无参调用'


# --------------------------------------------------------------------------- #
# 3. --soft-every 的节奏
# --------------------------------------------------------------------------- #
def test_soft_every_one_means_every_step():
    for s in range(10):
        assert t._soft_kind_for_step('ce', s, 1) == 'soft_ce', \
            f'soft-every=1 时第 {s} 步必须是软步'


def test_soft_every_n_picks_every_nth_step():
    assert [t._soft_kind_for_step('ce', s, 3) for s in range(7)] == \
        ['soft_ce', 'ce', 'ce', 'soft_ce', 'ce', 'ce', 'soft_ce']
    assert t._soft_kind_for_step('ce', 0, 5) == 'soft_ce', '第 0 步必须是软步'


def test_soft_every_zero_is_treated_as_one():
    """防御：`_at_least_one` 之外的直调（0/负数）不许抛 ZeroDivisionError。"""
    for every in (0, -3):
        assert all(t._soft_kind_for_step('ce', s, every) == 'soft_ce'
                   for s in range(4))


# --------------------------------------------------------------------------- #
# 4. --soft-only-sampling 的收窄
# --------------------------------------------------------------------------- #
def test_narrow_to_soft_rows_keeps_only_soft_rows():
    ds, _, _ = _toy_dataset(n=12, soft_rows=(1, 4, 7))
    train_idx = np.arange(12)
    out = t.narrow_to_soft_rows(train_idx, ds)
    assert out.tolist() == [1, 4, 7]


def test_narrow_to_soft_rows_preserves_input_order():
    ds, _, _ = _toy_dataset(n=12, soft_rows=(1, 4, 7))
    shuffled = np.array([7, 0, 4, 11, 1])
    assert t.narrow_to_soft_rows(shuffled, ds).tolist() == [7, 4, 1]


def test_narrow_to_soft_rows_errors_when_empty():
    """收窄到 0 行必须报错（静默退回全量 = 「以为在跑段 2、其实在跑段 1」）。"""
    ds, _, _ = _toy_dataset(n=12, soft_rows=(1, 4, 7))
    with pytest.raises(SystemExit, match='0'):
        t.narrow_to_soft_rows(np.array([0, 2, 3]), ds)


def test_narrow_without_soft_attached_errors():
    """没挂软标签时 `soft_row_mask()` 全 False ⇒ 收窄必然报错（不会退回全量）。"""
    _, data, _ = _toy_dataset(n=12)
    plain = SupervisedDataset(data, n_channels=12)        # 未挂 soft
    assert plain.soft_row is None
    with pytest.raises(SystemExit):
        t.narrow_to_soft_rows(np.arange(12), plain)


def test_soft_row_mask_is_all_false_without_soft():
    """`soft_row_mask()` 未挂标签时全 False（不是全 True）。"""
    ds, data, _ = _toy_dataset(n=12)
    plain = SupervisedDataset(data, n_channels=12)
    m = plain.soft_row_mask()
    assert m.shape == (12,) and m.dtype == bool and not m.any()


def test_main_rejects_soft_only_sampling_without_index():
    """main() 里那条守卫必须在（静态）：软行 0 个时不得静默走全量。"""
    src = inspect.getsource(t.main)
    assert 'soft-only-sampling 需要 --soft-index' in src
    assert 'narrow_to_soft_rows(' in src, 'main 应调用被单测覆盖的收窄函数'


def test_main_updates_n_train_after_narrowing():
    """收窄后 `n_train` 必须跟着改 —— 否则每卡步数/调度步数/log 行数全错。"""
    src = inspect.getsource(t.main)
    i_narrow = src.index('narrow_to_soft_rows(')
    tail = src[i_narrow:]
    assert 'n_train = len(train_idx)' in tail, \
        '收窄后没有重算 n_train（调度步数会按旧行数算）'
    # 重算必须发生在「步数/调度」推导之前
    assert tail.index('n_train = len(train_idx)') < tail.index('total_steps'), \
        'n_train 的重算必须早于 total_steps 的推导'


# --------------------------------------------------------------------------- #
# 5. 索引本身坏了 ⇒ 报错，不静默 no-op
# --------------------------------------------------------------------------- #
def test_attach_soft_index_rejects_missing_keys(tmp_path):
    ds, _, _ = _toy_dataset()
    p = tmp_path / 'bad.npz'
    np.savez(str(p), idx=np.array([1, 2], np.int64))          # 缺 policy
    with pytest.raises(KeyError, match='缺字段'):
        t.attach_soft_index(ds, str(p))


def test_attach_soft_index_rejects_wrong_action_width(tmp_path):
    """盘口不符（policy 宽度 ≠ bs²+1）必须报错，不是静默截断。"""
    ds, _, _ = _toy_dataset()
    p = _write_index(tmp_path / 'narrow.npz', [1, 4], np.zeros((2, BS * BS)))
    with pytest.raises(ValueError, match='soft_policy'):
        t.attach_soft_index(ds, p)


def test_attach_soft_index_rejects_out_of_range_rows(tmp_path):
    ds, _, sp = _toy_dataset(n=12)
    bad = np.repeat(sp[:1], 2, axis=0)          # 2 行 policy，配 2 个行号
    p = _write_index(tmp_path / 'oob.npz', [1, 999], bad)
    with pytest.raises(ValueError, match='越界'):
        t.attach_soft_index(ds, p)


def test_attach_soft_index_is_idempotent_with_constructor():
    """CLI 挂载 ≡ 构造器传参（A1/A4 两条入口不许漂）。"""
    _, data, sp = _toy_dataset()
    idxs = np.array([1, 4, 7])
    a = SupervisedDataset(data, n_channels=12, soft_idx=idxs, soft_policy=sp)
    b = SupervisedDataset(data, n_channels=12)
    b.attach_soft(idxs, sp)
    assert a.soft_row.tolist() == b.soft_row.tolist()
    keys = np.arange(12)
    da = a.sample_batch_numpy(keys, augment=False, labels=True)[3]
    db = b.sample_batch_numpy(keys, augment=False, labels=True)[3]
    assert da['soft'].tobytes() == db['soft'].tobytes()
    assert da['soft_mask'].tolist() == db['soft_mask'].tolist() == \
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]


# --------------------------------------------------------------------------- #
# 6. 防陈旧索引（2026-10-06 真机事故：数据集重建后索引没跟着重建，
#    行号全体平移 ⇒ policy 与局面逐行无关，train_top1 崩到 0.2%）
# --------------------------------------------------------------------------- #
def _write_index_with_meta(path, idx, policy, rows):
    """带 cache_meta（dataset.rows=rows）的索引产物，模拟 build_soft_index 写出。"""
    import json
    meta = {'schema': 1, 'max_repeats': None,
            'dataset': {'rows': rows, 'n_distinct': 3, 'mtime_ns': 1, 'size': 1,
                        'id_key': 'game_ids', 'path': '/x'},
            'labels': {'rows': rows, 'n_distinct': 0, 'mtime_ns': 1,
                       'size': 1, 'id_key': 'pos_hash', 'path': '/y'}}
    np.savez(str(path), idx=np.asarray(idx, np.int64),
             policy=np.asarray(policy, np.float16),
             cache_key=np.str_('k' * 32),
             cache_meta=np.str_(json.dumps(meta)))
    return str(path)


def _write_data_npz(tmp_path, data):
    p = tmp_path / 'toy_data.npz'
    np.savez(str(p), **data)
    return str(p)


def test_attach_soft_index_rejects_index_built_on_other_dataset_rowcount(tmp_path):
    """索引建在行数不同的数据集上 ⇒ 启动即 SystemExit（行号全体平移）。"""
    ds, data, sp = _toy_dataset(n=12, soft_rows=(1, 4))
    data_npz = _write_data_npz(tmp_path, data)
    p = _write_index_with_meta(tmp_path / 'stale.npz', [1, 4], sp, rows=999999)
    with pytest.raises(SystemExit, match='行号全体错位'):
        t.attach_soft_index(ds, p, data_npz=data_npz)


def test_attach_soft_index_accepts_matching_rowcount(tmp_path, capsys):
    """行数一致 ⇒ 挂载成功（mtime/size 漂移只告警，不拦跨机拷贝）。"""
    ds, data, sp = _toy_dataset(n=12, soft_rows=(1, 4))
    data_npz = _write_data_npz(tmp_path, data)
    p = _write_index_with_meta(tmp_path / 'ok.npz', [1, 4], sp, rows=12)
    diag = t.attach_soft_index(ds, p, data_npz=data_npz)
    assert diag['index_rows'] == 2
    assert '字段漂移' in capsys.readouterr().out      # mtime_ns=1 必然漂移 ⇒ 告警


def test_attach_soft_index_without_meta_warns_but_attaches(tmp_path, capsys):
    """旧版产物（无 cache_meta）放行但必须大声告警，不能静默。"""
    ds, _, sp = _toy_dataset(n=12, soft_rows=(1, 4))
    data_npz = _write_data_npz(tmp_path, {'boards': np.zeros((12, BS, BS), np.int8)})
    p = _write_index(tmp_path / 'nometa.npz', [1, 4], sp)
    diag = t.attach_soft_index(ds, p, data_npz=data_npz)
    assert diag['index_rows'] == 2
    assert '没有 cache_meta' in capsys.readouterr().out


def test_swanlab_config_reports_the_derived_kind():
    """SwanLab 的 config 面板必须显示**派生后**的 kind。

    否则段 2 的 run 在面板上写着 `policy_loss: ce` —— 与真实训练口径不一致，
    而这种不一致正是事后复盘时最难发现的一类。
    """
    src = inspect.getsource(t._init_swanlab)
    for key in ('soft_index', 'soft_weight', 'soft_only_sampling', 'soft_every'):
        assert f'"{key}"' in src, f'SwanLab config 缺 {key}'
    assert 'args.policy_loss' in src


def test_main_resolves_kind_before_loading_dataset():
    """kind 解析必须早于 `load_from_path`（坏组合不该先灌 12.3 GB）。"""
    src = inspect.getsource(t.main)
    i_resolve = src.index('resolve_policy_loss_kind(')
    i_load = src.index('load_from_path(')
    assert i_resolve < i_load, \
        '软标签 kind 解析被放在 load_from_path 之后：坏组合要先吃满内存才报错'


def test_attach_soft_index_happens_before_prefetcher_construction():
    """挂载必须早于预取器构造（fork 之后挂只有父进程看得见）。"""
    src = inspect.getsource(t.main)
    assert src.index('attach_soft_index(dataset') < \
        src.index('_BatchPrefetcher(dataset'), \
        '软标签挂载晚于 fork ⇒ worker 会继续造 soft_mask 全 0 的批'