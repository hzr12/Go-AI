"""预取器透传 `labels_dict`（软标签接入 A3）。

A3 的全部内容就是一句「预取器别再只搬三元组」，但它有三条**不能破**的性质：

1. **payload 形状只有一种**（spec §5.5）。`labels=True` 时 worker 回
   `(step, pos, states, moves, values, labels_dict, err)`，主进程沿 batch 轴把
   dict 拼回整批 —— `w` 是**嵌套 dict**，拼接必须递归。
2. **worker 的内存边界**：worker 是 fork 出来的，父进程已经把列读进内存了。
   worker 里任何「重新打开主 npz」的动作 = 每个 worker 再解一次 12.3 GB 的
   `boards`（本机 13.9 GB）。所以这一段源码只许调 `sample_batch_numpy`。
3. **旧路径逐位不变 + 背压不变**：`labels=False`（默认）时 `next()` 仍返回
   numpy 三元组；两个队列仍然**有界**（`maxsize = k·depth`）。

本文件怎么测不靠真进程
--------------------
`_BatchPrefetcher` 用 `mp.Process` 起 worker；本仓库的测试在 Windows 上跑
（默认 spawn ⇒ 得 pickle 假 dataset），真机是 Linux fork。这里把 `mod.mp`
换成 `Process` 不启动、`Queue` 用 `queue.Queue` 的桩，然后**在主进程里直接调**
`_prefetch_worker` 把任务消费掉 —— 于是 payload 的形状、拼接、顺序、递归
全是真的，唯一被替掉的只有「跨进程传递」这一步（那不是本文件要测的东西）。

跑：pytest tests/test_prefetch_labels.py -v
"""
import importlib.util
import os
import queue
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import SupervisedDataset  # noqa: E402

SRC_PATH = os.path.join(ROOT, 'scripts', 'train_sft.py')
SRC = open(SRC_PATH, encoding='utf-8').read()

#: `labels_dict` 必须恒含的键（A1 的契约，见 tests/test_dataset_soft_labels.py）。
REQUIRED_KEYS = ('next_move', 'outcome', 'outcome_black', 'game_weight',
                 'soft', 'soft_mask', 'w', 'score', 'sb_center', 'sb_upper',
                 'global', 'ownership', 'scoring', 'seki', 'future')

#: `w` 里的键（A1 建出来的权重 dict）。
W_KEYS = ('policy', 'policy_opp', 'ownership', 'score', 'scoring', 'seki',
          'futurepos')

BS = 9
A = BS * BS + 1
SEED = 1234            # `_BatchPrefetcher` 的默认 seed


def _load_mod():
    spec = importlib.util.spec_from_file_location('_tsft_a3', SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def mod():
    return _load_mod()


def _toy_dataset(n=12, seed=0, soft_rows=(1, 4, 7)):
    rng = np.random.default_rng(seed)
    data = {
        'boards': rng.integers(-1, 2, (n, BS, BS)).astype(np.int8),
        'my_hist': np.full((n, 3), -1, dtype=np.int16),
        'op_hist': np.full((n, 3), -1, dtype=np.int16),
        'ko': np.full(n, -1, dtype=np.int16),
        'moves': rng.integers(0, BS * BS, size=n).astype(np.int16),
        'values': rng.choice([-1, 1], size=n).astype(np.int8),
        'to_play': rng.choice([-1, 1], size=n).astype(np.int8),
        'game_ids': np.repeat(np.arange(n // 4), 4).astype(np.int32),
        'winrates': rng.uniform(-0.9, 0.9, size=n).astype(np.float32),
    }
    sp = rng.random((len(soft_rows), A)).astype(np.float32)
    sp /= sp.sum(axis=1, keepdims=True)
    ds = SupervisedDataset(data, n_channels=12,
                           soft_idx=np.asarray(soft_rows, np.int64), soft_policy=sp)
    return ds, data


class _StubMP:
    """`Process` 不启动、`Queue` 用真的 `queue.Queue`。"""

    class Process:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def terminate(self):
            pass

    @staticmethod
    def Queue(maxsize=0):                # noqa: N802 - 模拟 mp.Queue 的签名
        return queue.Queue(maxsize=maxsize)


def _make_pf(mod, ds, k=3, depth=2, labels=True):
    real_mp = mod.mp
    mod.mp = _StubMP
    try:
        pf = mod._BatchPrefetcher(ds, num_workers=k, prefetch=depth,
                                  seed=SEED, labels=labels)
    finally:
        mod.mp = real_mp
    return pf


def _run_one_task(mod, pf, task, ds, seed=SEED):
    """让 `_prefetch_worker` 只处理 `task` 这一个，然后收工。

    ⚠ 两个坑（都踩过）：
    1. worker 是 `while True`，处理完一个任务会回去 `get()` 下一个 ⇒ 必须喂
       一个 `None` 哨兵，否则它一直阻塞。
    2. 必须用**临时队列**喂，不能 `put` 回 `pf._task_q` —— 那是 FIFO，放回去
       的任务会被下一次调用先拿走（worker 会一口气吃掉队列里所有任务，
       于是「每个 wi 各自的 rng」这个确定性前提就没了）。
    """
    tmp = queue.Queue()
    tmp.put(task)
    tmp.put(None)
    step, wi, _sub = task
    mod._prefetch_worker(wi, tmp, pf._res_q, seed, ds, pf.labels)


def _drain(mod, pf, ds, seed=SEED):
    """把 `_task_q` 的任务逐个喂给 `_prefetch_worker`（按 `wi` 顺序）。

    等价于 k 个 worker 并行做完，只是换成了串行、且能按 `wi` 拿到**确定性**的
    rng（`seed + wi`）—— 这正是我们要的：拼回来的批必须与「每块自己算」逐位相等。
    """
    n = 0
    while True:
        try:
            task = pf._task_q.get_nowait()
        except queue.Empty:
            return n
        _run_one_task(mod, pf, task, ds, seed)
        n += 1


def _drain_reversed(mod, pf, ds, seed=SEED):
    """同 `_drain`，但**先收齐全部任务再逆序**执行。

    真并发下 worker 的返回顺序是不确定的（谁先算完谁先 put），所以主进程必须
    自己按 `pos` 排序。这里用「逆序回 payload」把那条排序逻辑逼出来。
    """
    tasks = []
    while True:
        try:
            tasks.append(pf._task_q.get_nowait())
        except queue.Empty:
            break
    for task in reversed(tasks):
        _run_one_task(mod, pf, task, ds, seed)
    return len(tasks)


def _baseline(mod, ds, idxs, k, seed=SEED):
    """「每块自己算 + 按 wi 拼」的真值基准（与 `submit` 的切分口径一致）。"""
    n = len(idxs)
    pieces = []
    for wi in range(k):
        sub = idxs[wi * n // k:(wi + 1) * n // k]
        pieces.append(ds.sample_batch_numpy(
            sub, rng=np.random.default_rng(seed + wi), augment=True, labels=True))
    return (np.concatenate([p[0] for p in pieces], axis=0),
            np.concatenate([p[1] for p in pieces], axis=0),
            np.concatenate([p[2] for p in pieces], axis=0),
            mod._concat_label_dicts([p[3] for p in pieces]))


# --------------------------------------------------------------------------- #
# 1. worker 的 payload：4 元组 + dict，且 w 是嵌套 dict
# --------------------------------------------------------------------------- #
def test_worker_payload_is_five_plus_dict_when_labels(mod):
    """`labels=True` 时 worker 回 `(step,pos,s,m,v,labels_dict,err)`。

    `err` 位必须存在（异常走同一个 payload，否则主进程分不清「子块失败」与
    「子块为空」）。
"""
    ds, _ = _toy_dataset()
    got = queue.Queue()
    q = queue.Queue()

    class _DS:
        def sample_batch_numpy(self, idxs, rng=None, labels=False):
            return ds.sample_batch_numpy(idxs, rng=rng, labels=labels)

    q.put((7, 2, np.array([1, 2, 3])))
    q.put(None)   # 哨兵：worker 是 while True，不喂会一直阻塞
    mod._prefetch_worker(2, q, got, SEED, _DS(), True)
    item = got.get_nowait()
    assert len(item) == 7, f'payload 形状变了：{len(item)} 个元素'
    step, pos, s, m, v, lbl, err = item
    assert (step, pos, err) == (7, 2, None)
    assert s.shape[0] == m.shape[0] == v.shape[0] == 3
    assert isinstance(lbl, dict) and isinstance(lbl['w'], dict)
    for k in REQUIRED_KEYS:
        assert k in lbl, f'labels_dict 缺键 {k}'
    for k in W_KEYS:
        assert k in lbl['w'], f"labels_dict['w'] 缺键 {k}"
    assert lbl['soft'].shape == (3, A)


def test_worker_payload_without_labels_carries_none_dict(mod):
    """`labels=False` 时第 6 位是 `None`（形状恒定，不退回短 payload）。"""
    ds, _ = _toy_dataset()
    out = queue.Queue()

    class _DS:
        def sample_batch_numpy(self, idxs, rng=None, labels=False):
            return ds.sample_batch_numpy(idxs, rng=rng, labels=labels)

    q = queue.Queue()
    q.put((0, 0, np.array([0, 1])))
    q.put(None)
    mod._prefetch_worker(0, q, out, SEED, _DS(), False)
    step, pos, s, m, v, lbl, err = out.get_nowait()
    assert lbl is None, 'labels=False 时不该带 dict'
    assert s.shape[0] == 2 and err is None


def test_worker_propagates_exception_inside_payload(mod):
    """子块抛异常 ⇒ 错误在 payload 里回主进程（`next()` 会 raise 它）。"""
    class _Boom:
        def sample_batch_numpy(self, idxs, rng=None, labels=False):
            raise ValueError('boom')

    q = queue.Queue()
    q.put((3, 1, np.array([0])))
    q.put(None)
    out = queue.Queue()
    mod._prefetch_worker(1, q, out, SEED, _Boom(), True)
    step, pos, s, m, v, lbl, err = out.get_nowait()
    assert isinstance(err, ValueError) and 'boom' in str(err)
    assert (s, m, v, lbl) == (None, None, None, None)


# --------------------------------------------------------------------------- #
# 2. next()：拼接 + 递归转张量 + 顺序
# --------------------------------------------------------------------------- #
def test_next_returns_four_tuple_with_tensor_dict(mod):
    ds, _ = _toy_dataset()
    pf = _make_pf(mod, ds, k=3, depth=2, labels=True)
    idxs = np.arange(8)
    pf.submit(idxs)
    assert _drain(mod, pf, ds) == 3, '3 个 worker 应各领到一个非空子块'
    out = pf.next()
    assert len(out) == 4, f'labels=True 时 next() 必须返回 4 项，实得 {len(out)}'
    states, moves, values, lbl = out
    assert states.shape[0] == moves.shape[0] == values.shape[0] == 8
    assert isinstance(lbl, dict)
    for k in REQUIRED_KEYS:
        v = lbl[k]
        if k == 'w':
            assert isinstance(v, dict), 'w 必须是 dict（嵌套）'
            for wk in W_KEYS:
                assert isinstance(v[wk], torch.Tensor), f"w['{wk}'] 不是张量"
            assert v['policy'].shape == (8,)
        else:
            assert isinstance(v, torch.Tensor), f'{k} 没有转成张量（{type(v)}）'
    assert lbl['soft'].shape == (8, A)
    assert lbl['soft_mask'].shape == (8,)


def test_next_dict_is_bitwise_equal_to_direct_sample(mod):
    """拼回来的整批必须与「每块自己算」**逐位相等**。

    这是拼接正确的唯一判据：拼接轴错了（沿 C 轴而不是 batch 轴）时，
    形状多半仍对，但逐位比较立刻红。`w` 的每个叶子都比。
    """
    ds, _ = _toy_dataset(n=12)
    idxs = np.arange(12)
    k = 3
    pf = _make_pf(mod, ds, k=k, depth=2, labels=True)
    pf.submit(idxs)
    assert _drain(mod, pf, ds) == k
    out = pf.next()
    want = _baseline(mod, ds, idxs, k)
    for got, exp, name in zip(out[:3], want[:3], ('states', 'moves', 'values')):
        assert got.dtype == exp.dtype and got.tobytes() == exp.tobytes(), \
            f'{name} 拼接后不是逐位相等'

    def flat(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update({f'{k}.{kk}': vv for kk, vv in flat(v).items()})
            else:
                out[k] = np.asarray(v)
        return out

    got, ref = flat(out[3]), flat(want[3])
    assert set(got) == set(ref)
    for key in sorted(ref):
        assert got[key].dtype == ref[key].dtype
        assert got[key].tobytes() == ref[key].tobytes(), \
            f'拼接后 labels_dict[{key!r}] 不是逐位相等'


def test_next_sorts_subblocks_by_pos(mod):
    """worker **逆序**回 payload 时，`next()` 仍按 `pos` 拼（顺序语义）。

    真并发下谁先算完谁先 put，所以主进程必须自己排。这一条用「同一批、正序
    vs 逆序执行」两个预取器的输出**逐位**对照来钉：若 `next()` 少了排序，
    两者的 moves / soft_mask 会整体错位（而软标签错位不报错）。
    """
    ds, _ = _toy_dataset(n=12)
    idxs = np.arange(12)
    fwd = _make_pf(mod, ds, k=3, depth=2, labels=True)
    fwd.submit(idxs)
    assert _drain(mod, fwd, ds) == 3
    rev = _make_pf(mod, ds, k=3, depth=2, labels=True)
    rev.submit(idxs)
    assert _drain_reversed(mod, rev, ds) == 3
    fa, rb = fwd.next(), rev.next()
    for a, b in zip(fa[:3], rb[:3]):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), \
            '子块顺序没有被 next() 纠正'

    def flat(d):
        out = {}
        for k, v in d.items():
            out.update({f'{k}.{kk}': vv for kk, vv in flat(v).items()}
                       if isinstance(v, dict) else {k: v})
        return out

    ga, gb = flat(fa[3]), flat(rb[3])
    assert set(ga) == set(gb)
    for key in sorted(gb):
        assert np.asarray(ga[key]).tobytes() == np.asarray(gb[key]).tobytes(), \
            f'labels_dict[{key!r}] 未随子块顺序一起纠正'


def test_next_device_none_keeps_labels_on_cpu(mod):
    """`device=None` ⇒ labels_dict 里的张量留在 CPU（不打设备 API）。"""
    ds, _ = _toy_dataset()
    pf = _make_pf(mod, ds, k=2, depth=2, labels=True)   # depth>=2：_drain 需要队列余量放哨兵
    pf.submit(np.arange(6))
    _drain(mod, pf, ds)
    _, _, _, lbl = pf.next()
    assert lbl['soft'].device.type == 'cpu'
    assert lbl['w']['policy'].device.type == 'cpu'


def test_empty_subblocks_are_dropped_not_concatenated(mod):
    """batch 小于 worker 数 ⇒ 有空子块；`next()` 必须滤掉它们。"""
    ds, _ = _toy_dataset()
    pf = _make_pf(mod, ds, k=4, depth=2, labels=True)   # depth>=2：_drain 需要队列余量放哨兵
    idxs = np.array([1, 3])
    pf.submit(idxs)
    assert _drain(mod, pf, ds) == 2, 'k=4 / B=2 ⇒ 只有 2 个非空子块'
    assert pf._res_q.qsize() == 4, '2 个空子块的占位 + 2 个真结果'
    states, moves, values, lbl = pf.next()
    assert states.shape[0] == 2 and moves.shape[0] == 2 and values.shape[0] == 2
    assert lbl['soft'].shape[0] == 2
    assert lbl['soft_mask'].tolist() == [1.0, 0.0], \
        f'idxs=[1,3] 只有行 1 有软标签，实得 {lbl["soft_mask"].tolist()}'


def test_concat_label_dicts_rejects_contract_fork(mod):
    """某个子块缺键 ⇒ **报错**（不许取交集静默少一项）。"""
    a = {'soft': np.zeros((2, A), np.float32),
         'w': {'policy': np.ones(2, np.float32)}}
    b = {'soft': np.zeros((3, A), np.float32)}        # 缺 'w'（老契约）
    with pytest.raises(KeyError, match='w'):
        mod._concat_label_dicts([a, b])
    with pytest.raises(KeyError, match='soft'):
        mod._concat_label_dicts([{'w': {'policy': np.ones(1, np.float32)}}, b])
    with pytest.raises(ValueError):
        mod._concat_label_dicts([])


def test_concat_label_dicts_reports_missing_nested_key(mod):
    """嵌套层缺键也要炸（不能只查顶层）。"""
    a = {'w': {'policy': np.ones(1, np.float32)}}
    b = {'w': {'policy_opp': np.ones(1, np.float32)}}
    with pytest.raises(KeyError, match='policy'):
        mod._concat_label_dicts([a, b])


# --------------------------------------------------------------------------- #
# 3. 零回归 + 背压 + 内存边界
# --------------------------------------------------------------------------- #
def test_labels_false_next_is_the_old_numpy_triple(mod):
    """`labels=False`（默认）⇒ 仍是 numpy 三元组，**没有**第四项。"""
    ds, _ = _toy_dataset()
    pf = _make_pf(mod, ds, k=3, depth=2, labels=False)
    assert pf.labels is False
    idxs = np.arange(8)
    pf.submit(idxs)
    assert _drain(mod, pf, ds) == 3
    out = pf.next()
    assert len(out) == 3, f'旧路径必须仍是三元组，实得 {len(out)} 项'
    for a in out:
        assert isinstance(a, np.ndarray), f'{type(a)} 不是 numpy（旧路径不转张量）'
    # 逐位对照：同种子下与 labels=True 的前三项一致
    pf2 = _make_pf(mod, ds, k=3, depth=2, labels=True)
    pf2.submit(idxs)
    _drain(mod, pf2, ds)
    out2 = pf2.next()
    for a, b, name in zip(out, out2[:3], ('states', 'moves', 'values')):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), \
            f'labels 开关改变了热路径的 {name}（旧路径必须逐位不变）'


def test_queues_stay_bounded(mod):
    """背压语义：两个队列都必须 `maxsize = k·depth`（不许改成无界）。

    无界队列 = 无限预取 = 预取深度失控时把内存吃光，而且症状是「跑着跑着
    被 OOM 杀了」，看不出是队列。
    """
    for k, depth in ((1, 1), (4, 8), (8, 8)):
        pf = _make_pf(mod, _toy_dataset()[0], k=k, depth=depth, labels=True)
        assert pf._task_q.maxsize == k * depth, \
            f'k={k} depth={depth}: 任务队列无界（背压没了）'
        assert pf._res_q.maxsize == k * depth, \
            f'k={k} depth={depth}: 结果队列无界（背压没了）'


def test_worker_source_never_reopens_the_dataset(mod):
    """worker 段里不许出现打开主 npz 的调用（每个 worker 再解一次 12.3 GB）。

    这条是对**源码切片**的静态检查（与
    `tests/test_prefetch_fork_order.py::test_prefetch_workers_do_not_touch_cuda_or_npu`
    同一个切片），因为这件事在真机上只能靠 OOM 事后发现。
    """
    seg = SRC[SRC.find('def _prefetch_worker('):SRC.find('class _BatchPrefetcher')]
    assert 'sample_batch_numpy' in seg, 'worker 应只调用 dataset.sample_batch_numpy'
    for bad in ('np.load(', 'materialize', 'mmap_mode', '.astype(np.int64).reshape'):
        assert bad not in seg, f'worker 里出现了 {bad!r}：会整列解压 boards'
    # 也不许自己造 SupervisedDataset（那是把数据又读一遍的另一种写法）
    assert 'SupervisedDataset' not in seg, 'worker 里不许重建数据集'


def test_future_placeholder_stays_zeros_through_the_prefetcher(mod):
    """🔴 **`future` 占位仍是全 0** —— A6（futurepos 邻行 gather）才填。

    预取器只是搬运：它不许「顺手」把占位换成真值，也**不许**把占位丢掉。
    丢掉的症状是下游按 `future` 建张量的代码在真机上形状错，而预取路径下
    永远测不出来。
    """
    ds, _ = _toy_dataset()
    pf = _make_pf(mod, ds, k=2, depth=2, labels=True)   # depth>=2：_drain 需要队列余量放哨兵
    idxs = np.arange(8)
    pf.submit(idxs)
    _drain(mod, pf, ds)
    _, _, _, lbl = pf.next()
    assert lbl['future'].shape == (8, 2, BS * BS)
    assert torch.count_nonzero(lbl['future']) == 0, \
        'future 占位被填了真值 —— A6 之前它必须恒 0（w["futurepos"] 也恒 0）'
    assert torch.count_nonzero(lbl['w']['futurepos']) == 0
    # 其余占位同样恒 0（消费方必须先看权重）
    for k in ('score', 'sb_center', 'sb_upper', 'global',
              'ownership', 'scoring', 'seki'):
        assert torch.count_nonzero(lbl[k]) == 0, f'占位 {k} 非 0'
    for k in ('ownership', 'score', 'scoring', 'seki', 'futurepos'):
        assert torch.count_nonzero(lbl['w'][k]) == 0, f"w['{k}'] 应恒 0"


def test_soft_mask_and_future_survive_concatenation_of_uneven_blocks(mod):
    """子块大小不等（k 整除 B 不成立）时，`soft_mask` 仍与整批对齐。

    `submit` 的切分是 `wi*n//k`，B < k 或 B % k != 0 时块大小会差 1；
    拼接错一行的话 soft_mask 会整体错位 —— 而 mask 错位不报错，
    只是「某些该训软的行在训 one-hot」。
"""
    ds, _ = _toy_dataset(n=12)
    for B in (5, 7):
        idxs = np.arange(B)
        pf = _make_pf(mod, ds, k=4, depth=2, labels=True)   # depth>=2：_drain 需要队列余量放哨兵
        pf.submit(idxs)
        _drain(mod, pf, ds)
        out = pf.next()
        want = _baseline(mod, ds, idxs, 4)
        assert out[1].tolist() == want[1].tolist(), f'B={B}: moves 错位'
        assert out[3]['soft_mask'].tolist() == \
            want[3]['soft_mask'].tolist(), f'B={B}: soft_mask 错位'


def test_labels_dict_tensor_helper_is_recursive(mod):
    """`_labels_dict_to_tensors` 递归（含嵌套 `w`），且不改数值。"""
    d = {'soft': np.ones((2, A), np.float32), 'w': {'policy': np.zeros(2, np.float32)}}
    t = mod._labels_dict_to_tensors(d)
    assert isinstance(t['w'], dict) and isinstance(t['w']['policy'], torch.Tensor)
    assert t['soft'].dtype == torch.float32
    assert t['soft'].numpy().tobytes() == d['soft'].tobytes()


def test_module_level_helpers_live_outside_the_worker_slice():
    """主进程侧 helper 必须在 worker 切片**之外**。

    `test_prefetch_workers_do_not_touch_cuda_or_npu` 扫的是「worker 定义 →
    `_BatchPrefetcher` 类定义」这一段；`_labels_dict_to_tensors` 里必须有跨设备
    搬运，把它放进去会让那条测试误红（它守的判据是 worker 纯 numpy，
    而搬张量的是主进程）。这条是**布局**守卫，防止以后有人「顺手」把 helper
    挪回 worker 后面。
    """
    i_worker = SRC.find('def _prefetch_worker(')
    i_cls = SRC.find('class _BatchPrefetcher')
    assert i_worker != -1 and i_cls != -1 and i_worker < i_cls
    seg = SRC[i_worker:i_cls]
    for fn in ('_concat_label_dicts', '_concat_tree', '_labels_dict_to_tensors'):
        assert f'def {fn}(' not in seg, \
            f'{fn} 落在 worker 切片里：它是主进程侧逻辑（含跨设备搬运）'
    assert 'torch.' not in seg, 'worker 段里不该出现 torch 调用'

