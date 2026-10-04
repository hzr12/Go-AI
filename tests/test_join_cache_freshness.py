"""join 缓存的防陈旧（软标签接入 A5）。

为什么这是本项目**最危险的失败模式**
------------------------------------
join 缓存陈旧时不会报错，只会**静默挂错标签**：换一批 SGF（主 npz 变）、换标签
文件、或改了散列种子/权重（口径变）—— 三者都让旧缓存指向错误的行。而症状是
「命中率 0」，与「这批局面真的没标签」**长得一模一样**，看不出是缓存陈旧
（spec §5.7 的原话）。

缓存 key 必须覆盖的四类输入
--------------------------
===========================  =========================================
输入                          指纹
===========================  =========================================
主 npz                        (行数, distinct game_ids, mtime, size)
软标签 npz                    (行数, distinct pos_hash, mtime, size)
``max_repeats``               原值进 key
散列口径                      人工版本号 + **从 pos_hash 活常量现算**的指纹
===========================  =========================================

跑：pytest tests/test_join_cache_freshness.py -v
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data import pos_hash as ph  # noqa: E402
from src.data.kata_label_join import (  # noqa: E402
    HASH_SPEC_COLUMNS, HASH_SPEC_VERSION, build_soft_index, diff_cache_meta,
    hash_spec_fingerprint, join_cache_key, join_by_hash, npz_fingerprint,
    read_join_cache,
)
from src.data.pos_hash import BOARD, pos_hash_block  # noqa: E402


def _positions(n, seed):
    rng = np.random.default_rng(seed)
    b = rng.integers(-1, 2, (n, BOARD, BOARD)).astype(np.int8)
    tp = rng.integers(-1, 2, n).astype(np.int8)
    ko = rng.integers(-1, 20, n).astype(np.int16)
    return b, tp, ko


def _dataset_hashes(n, seed, n_games=3):
    b, tp, ko = _positions(n, seed)
    return pos_hash_block(b, tp, ko)


def _write_dataset(path, n=12, seed=0, n_games=3):
    """造一个带 `game_ids` 的主 npz（指纹需要它算 distinct）。返回**路径**。"""
    b, tp, ko = _positions(n, seed)
    np.savez(path, boards=b, to_play=tp, ko=ko,
             game_ids=np.repeat(np.arange(n_games), n // n_games).astype(np.int32))
    return str(path)


def _write_labels(path, dataset_hashes, which=(0, 1, 2)):
    n = len(which)
    pol = np.zeros((n, 362), dtype=np.float16)
    for j, i in enumerate(which):
        pol[j, i * 7] = 1.0
    np.savez(path, policy=pol,
             pos_hash=np.asarray(dataset_hashes)[list(which)],
             root_win=np.full(n, 0.5, np.float32),
             score_mean=np.full(n, 1.0, np.float32),
             score_stdev=np.full(n, 6.0, np.float32),
             visits=np.full(n, 32, np.int16),
             game_idx=np.arange(n, dtype=np.int32),
             move_idx=np.arange(n, dtype=np.int32))
    return str(path)


class _Fixture:
    """一次性的 (主 npz, 标签 npz, 索引产物) 三元组。

     只在 `tmp_path` 上落一次盘：`_build` 每调用一次都重写输入文件的话，
      mtime 每次都变 ⇒ 缓存**永远**命不中（而那正是「缓存失效」测试要模拟的
      东西，会让「命中」用例假红）。
    """

    def __init__(self, tmp_path, which=(0, 1, 2), out='idx.npz'):
        self.ds = _write_dataset(str(tmp_path / 'ds.npz'), n=12, seed=0, n_games=3)
        h = _dataset_hashes(12, 0, 3)
        self.lb = _write_labels(str(tmp_path / 'lb.npz'), h, which)
        self.out = str(tmp_path / out)

    def build(self, log=None, **kw):
        logs = []
        diag = build_soft_index(self.ds, self.lb, self.out,
                                log=logs.append if log is None else log, **kw)
        return diag, logs


def _build(tmp_path, out='idx.npz', which=(0, 1, 2), **kw):
    """建一次索引，返回 (diag, log_lines, fixture)。"""
    fx = _Fixture(tmp_path, which=which, out=out)
    diag, logs = fx.build(**kw)
    return diag, logs, fx


# --------------------------------------------------------------------------- #
# 1. 指纹本身
# --------------------------------------------------------------------------- #
def test_npz_fingerprint_reports_four_facts(tmp_path):
    p = _write_dataset(str(tmp_path / 'ds.npz'), n=12, n_games=3)
    fp = npz_fingerprint(p, id_key='game_ids')
    assert fp['rows'] == 12
    assert fp['n_distinct'] == 3, 'distinct game_ids 必须真的算出来'
    assert fp['size'] > 0 and fp['mtime_ns'] > 0


def test_npz_fingerprint_changes_with_mtime(tmp_path):
    """原地重写同字节内容也要被发现（只靠 size 会漏）。"""
    p = _write_dataset(str(tmp_path / 'ds.npz'), n=12, n_games=3)
    m0 = join_cache_key(p, p)[1]['dataset']
    before = npz_fingerprint(p)['mtime_ns']
    os.utime(p, ns=(before + 10 ** 9, before + 10 ** 9))
    after = npz_fingerprint(p)['mtime_ns']
    assert after != before, 'mtime 没被采到（指纹会漏掉「重写同内容」）'
    m1 = join_cache_key(p, p)[1]['dataset']
    assert m0['mtime_ns'] != m1['mtime_ns']


def test_npz_fingerprint_reads_rows_without_decompressing_boards(tmp_path):
    """ 指纹**不许**整列解压 `boards`（12.3 GB，本机 13.9 GB）。

    这条是行为钉：`np.load(path)['boards']` 会先把整个成员解进内存。
    实现在无 `game_ids` 时只读 `.npy` 头 —— 这里用一个「boards 巨大但读不到」
    的替身来保证：把 `boards` 换成一个一旦被访问就炸的对象。
    """
    p = str(tmp_path / 'ds2.npz')
    np.savez(p, boards=np.zeros((4, BOARD, BOARD), np.int8),
             to_play=np.ones(4, np.int8), ko=np.full(4, -1, np.int16))
    fp = npz_fingerprint(p, id_key='game_ids')     # 无 game_ids ⇒ 走读头分支
    assert fp['rows'] == 4 and fp['n_distinct'] == 0
    # 直接证明：`np.load` 访问 boards 的成员会被记下来（对照组），而指纹函数没有
    assert 'boards' in HASH_SPEC_COLUMNS


def test_labels_fingerprint_uses_pos_hash(tmp_path):
    h = _dataset_hashes(12, 0, 3)
    lb = _write_labels(str(tmp_path / 'lb.npz'), h)
    fp = npz_fingerprint(lb, id_key='pos_hash')
    assert fp['rows'] == 3 and fp['n_distinct'] == 3
    assert fp['id_key'] == 'pos_hash'


# --------------------------------------------------------------------------- #
# 2. 散列口径指纹：改种子/权重必须让 key 变
# --------------------------------------------------------------------------- #
def test_hash_spec_fingerprint_is_derived_from_live_constants():
    """ 口径指纹必须**从 `pos_hash` 的活常量现算**，不能写死。

    写死的版本号靠人记得改 —— 而漏改的后果就是旧缓存静默挂错标签。
    """
    fp = hash_spec_fingerprint()
    assert isinstance(fp, str) and len(fp) == 32
    # 手工重算一遍：口径 = BOARD + 种子 + 加性种子 + 两组权重
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((ph.BOARD, ph.BOARD_CELLS, int(ph._HASH_SEED),
                   int(ph._SEED))).encode('utf-8'))
    for w in (ph._W1, ph._W2):
        h.update(np.ascontiguousarray(w, dtype=np.uint64).tobytes())
    assert fp == h.hexdigest(), \
        '口径指纹不再等于「pos_hash 活常量」的散列：改常量它不会跟着变'


def test_hash_spec_fingerprint_moves_when_a_constant_moves(monkeypatch):
    """改 `_W1[0]`（等价于「改了权重」）⇒ 指纹必须变 ⇒ 旧缓存作废。"""
    before = hash_spec_fingerprint()
    w1 = ph._W1.copy()
    w1[0] = np.uint64(3)          # 换整个数组：numpy 数组没有 dict 接口，
    monkeypatch.setattr(ph, '_W1', w1)   # monkeypatch.setitem 用不了
    assert hash_spec_fingerprint() != before, '改权重常量后口径指纹没变（旧缓存会被静默复用）'


def test_hash_spec_fingerprint_moves_when_seed_moves(monkeypatch):
    before = hash_spec_fingerprint()
    monkeypatch.setattr(ph, '_HASH_SEED', 0x1234)
    assert hash_spec_fingerprint() != before, '改种子后口径指纹没变'


def test_cache_key_contains_spec_version_and_fingerprint(tmp_path):
    p = _write_dataset(str(tmp_path / 'ds.npz'), n=12, n_games=3)
    key, meta = join_cache_key(p, p)
    assert meta['hash_spec_version'] == HASH_SPEC_VERSION
    assert meta['hash_spec_fingerprint'] == hash_spec_fingerprint()
    assert meta['hash_spec_columns'] == list(HASH_SPEC_COLUMNS)
    assert 'dataset' in meta and 'labels' in meta
    key2, _ = join_cache_key(p, p)
    assert key == key2, '同一输入必须给出同一 key（否则缓存永远命不中）'


def test_cache_key_changes_with_hash_spec_version(tmp_path):
    p = _write_dataset(str(tmp_path / 'ds.npz'), n=12, n_games=3)
    k0, _ = join_cache_key(p, p, hash_spec_version=HASH_SPEC_VERSION)
    k1, _ = join_cache_key(p, p, hash_spec_version=HASH_SPEC_VERSION + 1)
    assert k0 != k1, '口径版本号没进 key'


def test_cache_key_changes_with_max_repeats(tmp_path):
    p = _write_dataset(str(tmp_path / 'ds.npz'), n=12, n_games=3)
    assert join_cache_key(p, p, max_repeats=1)[0] != \
        join_cache_key(p, p, max_repeats=2)[0]
    assert join_cache_key(p, p, max_repeats=None)[0] != \
        join_cache_key(p, p, max_repeats=1)[0]


def test_diff_cache_meta_names_the_changed_field(tmp_path):
    p = _write_dataset(str(tmp_path / 'ds.npz'), n=12, n_games=3)
    k0, m0 = join_cache_key(p, p)
    os.utime(p, ns=(0, 10 ** 9))
    k1, m1 = join_cache_key(p, p)
    diff = diff_cache_meta(m0, m1)
    assert diff, '改了文件后 diff 不该是空'
    assert any('mtime_ns' in line for line in diff), f'diff 没指出 mtime：{diff}'


# --------------------------------------------------------------------------- #
# 3. 命中 / 陈旧 / 重建
# --------------------------------------------------------------------------- #
def test_second_call_hits_cache(tmp_path):
    fx = _Fixture(tmp_path)
    diag, logs = fx.build()
    assert diag['cache_hit'] is False and diag['matched_rows'] == 3
    assert os.path.isfile(fx.out)
    again, logs2 = fx.build()
    assert again['cache_hit'] is True, '同一输入第二次必须命中缓存'
    assert again['matched_rows'] == 3
    assert any('命中缓存' in m for m in logs2), f'命中没有可观测日志：{logs2}'
    assert not any('陈旧' in m for m in logs2), '不该报陈旧'


def test_cache_file_carries_key_and_meta(tmp_path):
    fx = _Fixture(tmp_path)
    fx.build()
    payload, stored = read_join_cache(fx.out)
    assert payload is not None and 'idx' in payload and 'policy' in payload
    assert stored['cache_key'] and len(stored['cache_key']) == 32
    meta = stored['cache_meta']
    assert meta['hash_spec_version'] == HASH_SPEC_VERSION
    assert meta['hash_spec_fingerprint'] == hash_spec_fingerprint()
    assert meta['dataset']['rows'] == 12


def test_legacy_index_without_key_is_rebuilt_with_a_warning(tmp_path):
    """**没有指纹的旧产物一律重建 + 告警**（无法判断是否陈旧）。"""
    fx = _Fixture(tmp_path)
    fx.build()
    legacy = str(tmp_path / 'legacy.npz')
    np.savez(legacy, idx=np.array([0, 1], np.int64),
             policy=np.zeros((2, 362), np.float16))      # 无 cache_key
    logs = []
    diag = build_soft_index(fx.ds, fx.lb, legacy, log=logs.append)
    assert diag['cache_hit'] is False
    assert any('没有缓存指纹' in m for m in logs), \
        f'无指纹产物被静默忽略了：{logs}'
    assert diag['matched_rows'] == 3


def test_stale_dataset_fingerprint_triggers_warning_and_rebuild(tmp_path):
    """ 换主 npz ⇒ 缓存失效、**明确告警并列出差异**、重建。"""
    fx = _Fixture(tmp_path)
    diag, _ = fx.build()
    assert diag['matched_rows'] == 3
    # 换一批数据（行数/局数都一样 ⇒ 只靠行数/局数发现不了，靠 mtime/size + 重建）
    _write_dataset(fx.ds, n=12, seed=5, n_games=3)
    _write_labels(fx.lb, _dataset_hashes(12, 5, 3), which=(3, 4, 5))
    again, logs = fx.build()
    assert again['cache_hit'] is False, '主 npz 变了却命中了缓存 ⇒ 会静默挂错标签'
    assert again['stale_reason'] is not None
    assert any('缓存陈旧' in m for m in logs), f'没有明确告警：{logs}'
    assert any('dataset.' in m for m in logs), f'告警没指出是主 npz 变了：{logs}'
    # 重建后的产物确实换成了新数据上的索引（不是旧缓存里的 [0,1,2]）
    assert np.load(fx.out)['idx'].tolist() == [3, 4, 5]


def test_stale_labels_fingerprint_triggers_warning_and_rebuild(tmp_path):
    """换标签文件（同一路径重写）⇒ 失效 + 告警。"""
    fx = _Fixture(tmp_path)
    fx.build()
    _write_labels(fx.lb, _dataset_hashes(12, 0, 3), which=(3, 4, 5))
    again, logs = fx.build()
    assert again['cache_hit'] is False
    assert any('labels.' in m for m in logs), \
        f'告警没指出是标签变了：{logs}'
    assert np.load(fx.out)['idx'].size == again['matched_rows']


def test_stale_hash_spec_version_triggers_warning_and_rebuild(tmp_path):
    """ 改散列口径版本号 ⇒ 失效 + 告警（这是最隐蔽的一种）。"""
    fx = _Fixture(tmp_path)
    fx.build()
    again, logs = fx.build(hash_spec_version=HASH_SPEC_VERSION + 1)
    assert again['cache_hit'] is False, '口径变了却命中缓存 ⇒ 旧散列的口径被沿用'
    assert any('hash_spec_version' in m for m in logs), \
        f'告警没指出是散列口径变了：{logs}'


def test_stale_hash_spec_fingerprint_triggers_rebuild(tmp_path, monkeypatch):
    """改 `pos_hash` 的**权重常量** ⇒ 指纹变 ⇒ 失效（不需要记得改版本号）。"""
    fx = _Fixture(tmp_path)
    fx.build()
    # numpy 数组不能用 monkeypatch.setitem（它要 dict 接口）⇒ 换整个数组
    w1 = ph._W1.copy()
    w1[0] = np.uint64(3)
    monkeypatch.setattr(ph, '_W1', w1)
    again, logs = fx.build()
    assert again['cache_hit'] is False, '改了权重常量却命中缓存'
    assert any('hash_spec_fingerprint' in m for m in logs), \
        f'告警没指出散列指纹变了：{logs}'


def test_max_repeats_change_triggers_rebuild(tmp_path):
    fx = _Fixture(tmp_path)
    fx.build(max_repeats=None)
    again, logs = fx.build(max_repeats=1)
    assert again['cache_hit'] is False
    assert any('max_repeats' in m for m in logs), f'告警没指出 max_repeats：{logs}'


def test_cache_disabled_always_rebuilds(tmp_path):
    """`cache=False` ⇒ 每次重算（诊断用）。"""
    fx = _Fixture(tmp_path)
    fx.build()
    again, _ = fx.build(cache=False)
    assert again['cache_hit'] is False


def test_corrupt_cache_is_treated_as_absent(tmp_path):
    """坏/半截的产物文件 ⇒ 当没有缓存（重算），不抛也不复用。"""
    fx = _Fixture(tmp_path)
    fx.build()
    with open(fx.out, 'wb') as fh:
        fh.write(b'not an npz at all')
    diag, _ = fx.build()
    assert diag['cache_hit'] is False
    assert diag['matched_rows'] == 3, '坏缓存之后必须真的重建出结果'


def test_read_join_cache_returns_none_for_missing_file(tmp_path):
    payload, meta = read_join_cache(str(tmp_path / 'nope.npz'))
    assert payload is None and meta is None


def test_zero_overlap_is_reported_not_cached(tmp_path):
    """命中 0 行 ⇒ 不写文件、不写缓存、**明确告警**。

    「命中率 0」是本项目最难查的症状，所以它必须与「缓存陈旧」在日志上
    可区分：这里的消息必须点明「先确认散列口径」。
    """
    ds = str(tmp_path / 'ds.npz')
    _write_dataset(ds, n=12, seed=0, n_games=3)
    other = pos_hash_block(*_positions(12, seed=99))
    lb = _write_labels(str(tmp_path / 'lb.npz'), other)
    out = str(tmp_path / 'idx.npz')
    logs = []
    diag = build_soft_index(ds, lb, out, log=logs.append)
    assert diag['matched_rows'] == 0 and diag['out'] is None
    assert not os.path.exists(out)
    assert any('命中 0 行' in m for m in logs), f'命中 0 没有告警：{logs}'
    assert any('散列口径' in m for m in logs), \
        f'告警没有把「无交集」与「缓存陈旧」区分开：{logs}'


# --------------------------------------------------------------------------- #
# 4. 索引产物本身（--soft-index 吃的那份）
# --------------------------------------------------------------------------- #
def test_output_keys_match_what_soft_index_expects(tmp_path):
    """产物必须含 `idx` + `policy`（`scripts/train_sft.py::attach_soft_index`
    的硬要求），且 idx 是 int64 / policy 是 (M,362)。"""
    fx = _Fixture(tmp_path)
    fx.build()
    z = np.load(fx.out)
    for k in ('idx', 'policy', 'n_rows', 'n_hit', 'hit_rate', 'n_label',
              'cache_key', 'cache_meta'):
        assert k in z.files, f'产物缺 {k}'
    assert z['idx'].dtype == np.int64
    assert z['policy'].shape[1] == 362
    assert z['idx'].size == z['policy'].shape[0]
    json.loads(str(z['cache_meta']))                  # 必须是合法 JSON


def test_transpositions_attach_to_every_row_by_default(tmp_path):
    """默认口径（``max_repeats=None``）= 同一局面出现几次就挂几次。

    那是 `build_soft_index.py` 的历史口径：那几个行的 `states` 本来一样，
    只挂第一个会白白丢掉 (dup−1)/dup 的监督量。
    """
    b, tp, ko = _positions(6, seed=7)
    b[3] = b[0]
    tp[3] = tp[0]
    ko[3] = ko[0]
    h = pos_hash_block(b, tp, ko)
    rows, labs = join_by_hash(h, h[[0]], max_repeats=None)
    assert sorted(rows.tolist()) == [0, 3]
    assert labs.tolist() == [0, 0]
    rows1, _ = join_by_hash(h, h[[0]], max_repeats=1)
    assert rows1.tolist() == [0]