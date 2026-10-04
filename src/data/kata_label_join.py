"""把 `kata_labels.npz`（KataGo 自打标签）挂到本仓数据集的行上。

输入
----
``kata_labels.npz``（`scripts/label_sgf.py` 产出）::

    policy      float16 (N, 362)  归一化后的访问分布（361 落点 + pass）
    pos_hash    uint64   (N,)     **join 键**，口径见 `src/data/pos_hash.py`
    root_win    float32  (N,)
    score_mean  float32  (N,)
    score_stdev float32  (N,)
    visits      int16    (N,)
    game_idx    int32    (N,)
    move_idx    int32    (N,)

产出
----
**sidecar**（不重写 34.2M 主文件）：`row_index int64 (M,)` + `policy float16 (M,362)`
+ 诊断计数。用 `row_index` 回指主文件即可。

为什么是 sidecar 而不是并进主 npz
-------------------------------
主文件 `data/sgf_19x19_full.npz` 解压后约 **13.4 GB**，本机 13.9 GB ⇒ 整列
`np.load` 会 OOM。重写它还要重跑一遍 34.2M 的写入。sidecar 只增 M 行（M 是被
标注到的行数，1% 约 34 万），几 MB。

 **同一局面会在多局里出现**（transposition）。`pos_hash` 因此在两侧都可能有
重复，join 是**多对多**的。默认 `max_repeats=1`：每个被标注的局面只挂到
**第一个**匹配的数据行。放宽它能把同一个搜索标签喂给多个出现处，但会按
重复次数给高频局面加权（一个布局变例出现 50 次就压过别处 50 倍）。
`dup_factor` 会把实际的重复倍数报出来，>1 的比例高就说明该局面被重复加权。

缓存防陈旧（A5）
--------------
join 的结果**完全**由四个输入决定：主 npz、软标签 npz、`max_repeats`、
散列口径。前三个用文件指纹（行数 / distinct id / mtime / size）锁，第四个用
`HASH_SPEC_VERSION` + `hash_spec_fingerprint()`（从 `pos_hash` 的**活常量**
导出）锁。任何一处变了都必须重建，否则旧缓存会**静默挂错标签** ——
症状是「命中率 0」，与「这批局面真的没标签」长得一模一样，是本项目最危险的
失败模式。命中陈旧缓存时**明确告警并逐项列出差异**，绝不静默复用。
"""

import hashlib
import json
import os
import zipfile

import numpy as np

from . import pos_hash as _ph
from .pos_hash import pos_hash_block

#: `label_sgf.py` 写出的键。缺一个就该报错而不是当默认值 —— 少一个键意味着
#: 那一路监督信号悄悄退化成"全 0 标签"，比崩掉难查得多。
REQUIRED_KEYS = ('policy', 'pos_hash', 'root_win', 'score_mean',
                 'score_stdev', 'visits', 'game_idx', 'move_idx')

#: 主数据集里 join 需要的列。
DATASET_KEYS = ('boards', 'to_play', 'ko')

#: 散列扫描的块大小。34.2M × 361 B = 12.3 GB，一次性读不进 13.9 GB 的机器；
#: 分块后只驻留当前块的 uint64 散列（34.2M × 8 B = 274 MB，可接受）。
_HASH_CHUNK = 200_000

# --------------------------------------------------------------------------- #
# A5 · join 缓存的 key（防陈旧）
# --------------------------------------------------------------------------- #
#: 散列口径的**人工**版本号。口径 = 「哪些列参与散列」+「怎么混」（见
#: `HASH_SPEC_COLUMNS` 与 `src/data/pos_hash.py` 的 docstring）。
#: 只要「参与散列的输入集合」变了（例如将来把历史手数也纳入），这里必须 +1；
#: 「权重常量/种子变了」不需要动它 —— 那种变更由 `hash_spec_fingerprint()`
#: 从**活常量**自动捕获（漏改人工版本号就不会有任何缓存被作废）。
HASH_SPEC_VERSION = 1

#: 参与散列的列（与 `DATASET_KEYS` 同源，单独列一份是因为它要进缓存 key）。
HASH_SPEC_COLUMNS = ('boards', 'to_play', 'ko')

#: 缓存 meta 的 schema 版本。布局本身变了（不是口径变了）才需要 +1。
_CACHE_SCHEMA = 1


def hash_spec_fingerprint():
    """散列口径的指纹 —— 从 `src.data.pos_hash` 的**活常量**现算。

    为什么必须现算而不是写死一个版本号
    ----------------------------------
    「改了种子/权重常量但忘了作废旧缓存」的后果是**静默挂错标签**：join 变空，
    而症状看起来像「这批局面真的没标签」（spec §5.7 的原话）。手工版本号
    靠人记得改 —— 这正是它不够的地方。

    所以这里读的是 `pos_hash` 模块里**实际在用**的 `_HASH_SEED` / `_W1` /
    `_W2` / `_SEED` / `BOARD`：`hash_spec_fingerprint()` 只能通过读模块属性拿到，
    所以改了常量就必须同步这里的清单，否则这里会静默返回旧指纹（见
    `hash_spec_fingerprint_is_derived_from_live_constants` 的守卫）。
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((_ph.BOARD, _ph.BOARD_CELLS, int(_ph._HASH_SEED),
                   int(_ph._SEED))).encode('utf-8'))
    for w in (_ph._W1, _ph._W2):
        h.update(np.ascontiguousarray(w, dtype=np.uint64).tobytes())
    return h.hexdigest()


def _npz_member_rows(path, member):
    """只读 npz 成员 `.npy` 头里的行数，**不解压数据**。

     为什么不用 `np.load(path)[member].shape[0]`：`np.load` 访问成员会先把
    **整个成员解压进内存**再切片 —— 对 `boards` 就是 12.3 GB，本机 13.9 GB。
    npz 是 zip，`.npy` 的 shape 在文件头里，读头就够。
    """
    with zipfile.ZipFile(path) as zf:
        with zf.open(member + '.npy') as fh:
            version = np.lib.format.read_magic(fh)
            reader = getattr(np.lib.format, '_read_array_header', None)
            if reader is None:                      # numpy 未来版本改名了
                reader = (np.lib.format.read_array_header_2_0
                          if version[0] == 2 else np.lib.format.read_array_header_1_0)
            shape = reader(fh, version)[0]
    return int(shape[0]) if len(shape) else 1


def npz_fingerprint(path, id_key='game_ids'):
    """一个 npz 的内容指纹：``(rows, distinct id, mtime, size)``。

    为什么是这四项的组合
    --------------------
    * ``rows``        —— 行数变了，join 的行号含义全变。
    * ``distinct id`` —— 只看行数会漏掉「重写了同样行数但内容不同」（用
      `game_ids` / `pos_hash` 的**去重后**个数当廉价内容探针：真换了数据集
      时局数/局面数几乎不可能相同，而重复的局 id 数量差异会被 `distinct` 吸收）。
    * ``mtime``       —— 原地重写同一份数据（同名字节数）也能被发现。
    * ``size``        —— mtime 被 `touch -r` 之类抹平时还有一道兜底。

     读 ``id_key`` 会解压**那一个成员**（`game_ids` 是 137 MB，可接受）；
    缺该列时退回「读 `boards` 的头拿行数」，且 ``n_distinct`` 记 0 ——
    此时 mtime+size 仍是有效兜底。
    """
    st = os.stat(path)
    n_rows = n_distinct = None
    with np.load(path, allow_pickle=False) as z:
        files = list(z.files)
        if id_key in files:
            ids = np.asarray(z[id_key]).ravel()
            n_rows = int(ids.size)
            n_distinct = int(np.unique(ids).size)
        elif 'boards' in files:
            n_rows = _npz_member_rows(path, 'boards')
            n_distinct = 0
        else:
            n_rows = _npz_member_rows(path, files[0])
            n_distinct = 0
    return {
        'path': os.path.abspath(path),
        'rows': n_rows,
        'n_distinct': n_distinct,
        'mtime_ns': int(st.st_mtime_ns),
        'size': int(st.st_size),
        'id_key': id_key,
    }


def join_cache_key(dataset_npz, labels_npz, max_repeats=1,
                   hash_spec_version=None, extra=None):
    """join 结果的缓存 key —— 含**全部**影响结果的输入。

    Returns:
        ``(key_hex, meta_dict)``。``meta`` 存进产物文件，供失效时**逐项**报差异。

     少任何一项都会退化成「命中率高但结果错」：换一批 SGF（主 npz 变）、
    换标签文件、或改了散列种子 —— 三者都会让旧缓存指向错误的行。
    """
    meta = {
        'schema': _CACHE_SCHEMA,
        'hash_spec_version': (HASH_SPEC_VERSION if hash_spec_version is None
                              else int(hash_spec_version)),
        'hash_spec_fingerprint': hash_spec_fingerprint(),
        'hash_spec_columns': list(HASH_SPEC_COLUMNS),
        'max_repeats': max_repeats,
        'dataset': npz_fingerprint(dataset_npz, id_key='game_ids'),
        'labels': npz_fingerprint(labels_npz, id_key='pos_hash'),
    }
    if extra:
        meta['extra'] = dict(extra)
    blob = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.blake2b(blob, digest_size=16).hexdigest(), meta


def _flatten_meta(meta, prefix=''):
    out = {}
    for k, v in meta.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten_meta(v, prefix=key + '.'))
        else:
            out[key] = v
    return out


def diff_cache_meta(old_meta, new_meta):
    """两份 meta 的**逐项**差异（给告警文案用，别让「哪一项变了」靠猜）。"""
    a, b = _flatten_meta(old_meta or {}), _flatten_meta(new_meta or {})
    keys = sorted(set(a) | set(b))
    return [f'{k}: {a.get(k, "<缺失>")} → {b.get(k, "<缺失>")}'
            for k in keys if a.get(k) != b.get(k)]


def read_join_cache(path, keys=('idx', 'policy', 'n_hit', 'n_rows', 'hit_rate',
                                'n_label')):
    """读产物文件里带的缓存 key 与 payload。**没有 key 的旧文件当没有缓存**。

    Returns:
        ``(payload_dict | None, meta | None)``。文件缺失 / 不可读 / 没有
        `cache_key` 时返回 ``(None, None)`` —— 静默失败只发生在「读缓存」这一侧，
        调用方随后会走重建并告警，不会把坏数据当好数据用。
    """
    if not os.path.isfile(path):
        return None, None
    try:
        with np.load(path, allow_pickle=False) as z:
            if 'cache_key' not in z.files:
                return None, None
            key = str(np.asarray(z['cache_key']).reshape(-1)[0])
            meta = None
            if 'cache_meta' in z.files:
                meta = json.loads(str(np.asarray(z['cache_meta']).reshape(-1)[0]))
            payload = {k: z[k] for k in keys if k in z.files}
    except Exception:  # noqa: BLE001 — 坏缓存一律当没有（随后重建 + 告警）
        return None, None
    if not payload or 'idx' not in payload:
        return None, None
    return payload, {'cache_key': key, 'cache_meta': meta}


def load_kata_labels(path, required=True):
    """读并校验 `kata_labels.npz`。"""
    d = np.load(path, allow_pickle=False)
    if required:
        missing = [k for k in REQUIRED_KEYS if k not in d.files]
        if missing:
            raise KeyError(
                f'{path} 缺字段 {missing}；现有 {sorted(d.files)}。'
                f'少字段会让那一路监督退化成全 0 标签，必须报错。')
    n = len(d['pos_hash'])
    for k in REQUIRED_KEYS:
        if k in d.files and len(d[k]) != n:
            raise ValueError(f'{path}: {k} 长度 {len(d[k])} != pos_hash 长度 {n}')
    if d['policy'].ndim != 2 or d['policy'].shape[1] != 362:
        raise ValueError(f'{path}: policy 应为 (N, 362)，实测 {d["policy"].shape}')
    return d


def materialize_dataset(dataset_npz, out_dir, keys=DATASET_KEYS, progress=None):
    """把 npz 里的 join 用列**一次性**落成 `.npy`，供 `mmap_mode='r'` 分块读。

    为什么必须做这一步
    ------------------
    `np.load('x.npz')['boards']` 会把**整个成员**解压进内存再切片 ——
    不是按需解压。`boards` 是 ``34.2M × 361 B = 12.3 GB``，而本机 13.9 GB，
    任何一次 `z['boards'][s:e]` 都会先吃掉 12.3 GB，余量只剩 1.6 GB，
    任何瞬时开销（numpy 临时量、另一条数据流）就会把它推过 OOM。

    落成 `.npy` 后 `np.load(mmap_mode='r')` 是真正的按需分页：分块读 200K 行
    只驻留 200K × 361 B = 72 MB。实测吞吐 478K 行/s，全量 34.2M 约 72s。

    磁盘代价：``boards.npy`` 12.3 GB + 两个小列（F 盘余 695 GB，可接受）。
    一次落盘、之后反复扫描都省内存。

     用 `open_memmap` 建**完整形状**的可写文件再分块填，而不是先写头再 append
    —— 后者会在 `write_array` 时重复写文件头，得到一个前段全 0 的坏 `.npy`
    （实测踩过：形状对、值全 0、散列全错，且不报错）。

    Returns:
        ``{key: path}``。
    """
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(dataset_npz, allow_pickle=False)
    out = {}
    for k in keys:
        if k not in z.files:
            raise KeyError(f'{dataset_npz} 缺列 {k}；现有 {sorted(z.files)}')
        arr = z[k]
        p = os.path.join(out_dir, f'{k}.npy')
        mm = np.lib.format.open_memmap(p, mode='w+', dtype=arr.dtype,
                                       shape=arr.shape)
        step = 1_000_000
        for s in range(0, arr.shape[0], step):
            e = min(s + step, arr.shape[0])
            mm[s:e] = arr[s:e]
            if progress:
                progress(k, e, arr.shape[0])
        mm.flush()
        del mm
        out[k] = p
    z.close()
    return out


def _open_columns(dataset_npz, out_dir=None, keys=DATASET_KEYS):
    """优先用 `.npy` memmap；没有就退回 npz（会整份解压，调用方要知情）。"""
    if out_dir:
        paths = {k: os.path.join(out_dir, f'{k}.npy') for k in keys}
        if all(os.path.isfile(p) for p in paths.values()):
            return {k: np.load(p, mmap_mode='r') for k, p in paths.items()}, True
    z = np.load(dataset_npz, allow_pickle=False)
    return {k: z[k] for k in keys}, False


def scan_dataset_hashes(npz_path, keys=DATASET_KEYS, chunk=_HASH_CHUNK,
                        limit=None, progress=None, materialized_dir=None):
    """算主数据集每行的 `pos_hash`。

    Args:
        materialized_dir: `materialize_dataset` 的输出目录。**给了它才不会
            把 12.3 GB 的 `boards` 整个解压进内存**（见该函数 docstring）。
        limit: 只算前 N 行（调试用）。

    Returns:
        ``(hashes uint64 (N,), n_rows)``。
    """
    cols, is_mmap = _open_columns(npz_path, materialized_dir, keys)
    if not is_mmap:
        # 明确告知：这一条路径会把整个成员读进内存
        print('[join] 未提供 materialized_dir：npz 成员会被整份解压，'
              'boards 约 12.3 GB。本机 13.9 GB，余量极小。'
              '先跑 materialize_dataset()。', flush=True)
    boards = cols['boards']
    total = boards.shape[0]
    hi = total if limit is None else min(total, int(limit))
    out = np.empty(hi, dtype=np.uint64)
    for s in range(0, hi, chunk):
        e = min(s + chunk, hi)
        out[s:e] = pos_hash_block(boards[s:e], cols['to_play'][s:e],
                                  cols['ko'][s:e])
        if progress is not None:
            progress(e, hi)
    return out, hi


def join_by_hash(dataset_hashes, label_hashes, max_repeats=1):
    """按散列 join，返回 ``(row_index, label_index)``。

    Args:
        dataset_hashes: ``(N,)`` uint64，主数据集每行一个。
        label_hashes: ``(M,)`` uint64，标签每行一个。
        max_repeats: 同一局面最多挂到几个数据行。1 = 只取第一个匹配；
            ``None`` = **全部**匹配（`build_soft_index` 的历史口径：一个局面
            在语料里出现几次就挂几次，因为那几个行的 `states` 本来就一样）。

    Returns:
        两个等长数组：``row_index``（主数据集行号）、``label_index``（标签行号），
        按 ``row_index`` 升序。

     **两侧都可能有重复**。实现用「对标签散列去重 → 排序 → searchsorted」，
    而不是 `np.intersect1d`（后者在有重复时只给唯一值，标签与行的对应关系
    会丢）。代价是 O(N log N + M log M)，34.2M 行约几十秒。
    """
    ds = np.asarray(dataset_hashes)
    lb = np.asarray(label_hashes)
    if ds.size == 0 or lb.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    key = np.argsort(ds, kind='stable')
    ds_sorted = ds[key]
    # 标签侧去重：同一局面只保留第一条（labels 数组里首次出现的那行）
    order_lb = np.argsort(lb, kind='stable')
    lb_sorted = lb[order_lb]
    uniq_mask = np.empty(lb_sorted.size, dtype=bool)
    uniq_mask[0] = True
    np.not_equal(lb_sorted[1:], lb_sorted[:-1], out=uniq_mask[1:])
    uniq_pos = order_lb[uniq_mask]              # 去重后保留的标签行号（已排序）
    uniq_hash = lb_sorted[uniq_mask]

    lo = np.searchsorted(ds_sorted, uniq_hash, side='left')
    if max_repeats is None:
        # 全部重复：用 cumsum 技巧把「每个散列的等长区间」摊平成一个索引数组
        # （一次向量化，没有 Python 循环）。
        hi = np.searchsorted(ds_sorted, uniq_hash, side='right')
        cnt = (hi - lo).astype(np.int64)
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        starts = np.cumsum(cnt) - cnt
        offs = np.repeat(lo.astype(np.int64), cnt) + (
            np.arange(total, dtype=np.int64) - np.repeat(starts, cnt))
        lab_ids = np.repeat(np.arange(uniq_hash.size, dtype=np.int64), cnt)
        row_index = key[offs].astype(np.int64)
        label_index = uniq_pos[lab_ids].astype(np.int64)
        order = np.argsort(row_index, kind='stable')
        return row_index[order], label_index[order]

    rows, labs = [], []
    for take in range(1, int(max_repeats) + 1):
        p = lo + (take - 1)
        ok = p < ds_sorted.size
        cand = np.where(ok, ds_sorted[np.clip(p, 0, ds_sorted.size - 1)], 0)
        hit = ok & (cand == uniq_hash)
        if not hit.any():
            break
        rows.append(key[p[hit]].astype(np.int64))
        labs.append(uniq_pos[hit].astype(np.int64))
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    row_index = np.concatenate(rows)
    label_index = np.concatenate(labs)
    order = np.argsort(row_index, kind='stable')
    return row_index[order], label_index[order]


def duplicate_factor(dataset_hashes, label_hashes):
    """每个被标注局面在**主数据集里**出现的次数（去重前）。

    用来判断 `max_repeats>1` 会不会把权重压到少数高频布局变例上。

    Returns:
        ``(counts int64, uniq_hashes uint64)`` —— 只含**被标注过**的那些局面。
    """
    u, c = np.unique(np.asarray(dataset_hashes), return_counts=True)
    lut = dict(zip(u.tolist(), c.tolist()))
    uniq = np.unique(np.asarray(label_hashes))
    return (np.array([lut.get(int(h), 0) for h in uniq], dtype=np.int64), uniq)


def build_sidecar(dataset_npz, labels_npz, out_path, max_repeats=1,
                  limit=None, progress_every=0, materialized_dir=None):
    """算 join 并写 sidecar npz。

    Args:
        materialized_dir: `materialize_dataset` 的输出目录。**强烈建议给**：
            不给的话 `boards` 会整份解压（12.3 GB），本机只剩 1.6 GB 余量。

    Returns:
        诊断 dict（行数 / 命中率 / 重复倍数 / 实际写出的行数）。

     **不命中时也返回诊断但不写文件** —— 「标签与数据集无交集」是一个需要
    立刻看见的结论，静默不写文件会让调用方以为跑成功了。
    """
    lab = load_kata_labels(labels_npz)

    def _prog(e, hi):
        if progress_every and (e // _HASH_CHUNK) % progress_every == 0:
            print(f'  [join] 散列 {e}/{hi}', flush=True)

    ds_hash, n_rows = scan_dataset_hashes(dataset_npz, limit=limit,
                                          progress=_prog if progress_every
                                          else None,
                                          materialized_dir=materialized_dir)
    row_index, label_index = join_by_hash(ds_hash, lab['pos_hash'],
                                          max_repeats=max_repeats)
    hit = int(row_index.size)
    dup, _ = duplicate_factor(ds_hash, lab['pos_hash']) if hit \
        else (np.empty(0, np.int64), None)
    diag = {
        'dataset_rows': int(n_rows),
        'label_rows': int(len(lab['pos_hash'])),
        'matched_rows': hit,
        'hit_rate_vs_dataset': hit / max(n_rows, 1),
        'hit_rate_vs_labels': hit / max(len(lab['pos_hash']), 1),
        'dup_factor_mean': float(dup.mean()) if dup.size else 0.0,
        'dup_factor_max': int(dup.max()) if dup.size else 0,
    }
    if hit:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez_compressed(
            out_path,
            row_index=row_index.astype(np.int64),
            policy=lab['policy'][label_index].astype(np.float16),
            root_win=lab['root_win'][label_index].astype(np.float32),
            score_mean=lab['score_mean'][label_index].astype(np.float32),
            score_stdev=lab['score_stdev'][label_index].astype(np.float32),
            visits=lab['visits'][label_index].astype(np.int16),
            pos_hash=lab['pos_hash'][label_index].astype(np.uint64),
        )
        diag['out'] = out_path
    return diag


# --------------------------------------------------------------------------- #
# A5 · 训练侧软标签索引（`--soft-index` 吃的就是它）+ 防陈旧缓存
# --------------------------------------------------------------------------- #

def build_soft_index(dataset_npz, labels_npz, out_path, max_repeats=None,
                     chunk=_HASH_CHUNK, cache=True, materialized_dir=None,
                     progress=None, hash_spec_version=None, log=print):
    """算「软标签 → 数据集行号」的 join 索引，带**防陈旧缓存**。

    产出（`--soft-index` 直接读这份）
    --------------------------------
    ``idx`` int64 (M,) · ``policy`` float16 (M,362) · 诊断标量 ·
    ``cache_key`` / ``cache_meta``（缓存指纹，缺这两项的旧文件当没有缓存）

    Args:
        max_repeats: 同一局面挂到几个数据行；``None`` = **全部**（默认，历史口径：
            同一局面在语料里出现几次就挂几次，那几行的 `states` 本来一样）。
        cache: 关掉就每次重算（诊断用；生产必须开着）。
        log: 告警/进度回调（默认 `print`）。**陈旧缓存的告警必须走这里**，
            不能静默 —— 静默复用会挂错标签，而症状只是「命中率 0」。

    Returns:
        诊断 dict，附 ``cache_hit`` / ``cache_key`` / ``stale_reason``。

     **不命中时也返回诊断但不写文件**（同 `build_sidecar`）：「标签与数据集
    无交集」必须立刻看见。
    """
    def _log(msg):
        if log is not None:
            log(msg)

    key, meta = join_cache_key(dataset_npz, labels_npz, max_repeats=max_repeats,
                               hash_spec_version=hash_spec_version)
    stale = None
    if cache:
        cached, stored = read_join_cache(out_path)
        if cached is not None:
            stored_key = (stored or {}).get('cache_key')
            stored_meta = (stored or {}).get('cache_meta')
            if stored_key == key:
                idx = cached['idx']
                _log(f'[soft-index] 命中缓存 {out_path} | 行 {int(idx.size)}'
                     f' | key={key[:12]}（主 npz / 标签 / max_repeats / 散列口径 全部一致）')
                return {
                    'out': out_path, 'cache_hit': True, 'cache_key': key,
                    'stale_reason': None,
                    'dataset_rows': meta['dataset']['rows'],
                    'label_rows': meta['labels']['rows'],
                    'matched_rows': int(idx.size),
                    'n_rows': int(cached['n_rows']) if 'n_rows' in cached else int(idx.size),
                    'n_hit': int(cached['n_hit']) if 'n_hit' in cached else -1,
                    'n_label': int(cached['n_label']) if 'n_label' in cached else -1,
                    'hit_rate': float(cached['hit_rate']) if 'hit_rate' in cached else 0.0,
                    'hit_rate_vs_dataset': 0.0,
                    'hit_rate_vs_labels': (float(cached['hit_rate'])
                                           if 'hit_rate' in cached else 0.0),
                }
            # **陈旧缓存：不静默复用。** 逐项列出差异再重建。
            stale = (stored_meta or {})
            for line in diff_cache_meta(stale, meta):
                _log(f'[soft-index] 缓存已失效 · {line}')
            _log(f'[soft-index] 缓存陈旧（key {str(stored_key)[:12]} ≠ '
                 f'{key[:12]}），正在重建 —— 复用它会**静默挂错标签**')
        elif os.path.isfile(out_path):
            _log(f'[soft-index] {out_path} 已存在但**没有缓存指纹**（旧版产物），'
                 f'重建 —— 无指纹的产物无法判断是否陈旧')

    lab = load_kata_labels(labels_npz)
    ds_hash, n_rows = scan_dataset_hashes(dataset_npz, chunk=chunk,
                                          progress=progress,
                                          materialized_dir=materialized_dir)
    lab_h = np.asarray(lab['pos_hash'])
    row_index, label_index = join_by_hash(ds_hash, lab_h, max_repeats=max_repeats)
    hit = int(row_index.size)
    want = np.unique(lab_h)
    dup, _ = duplicate_factor(ds_hash, lab_h) if hit else (np.empty(0, np.int64), None)
    pol = (lab['policy'][label_index].astype(np.float16) if hit
           else np.zeros((0, lab['policy'].shape[1]), np.float16))
    diag = {
        'out': None, 'cache_hit': False, 'cache_key': key, 'stale_reason': stale,
        'dataset_rows': int(n_rows),
        'label_rows': int(len(lab_h)),
        'n_want': int(want.size),
        'matched_rows': hit,
        'hit_rate_vs_dataset': hit / max(n_rows, 1),
        'hit_rate_vs_labels': hit / max(want.size, 1),
        'dup_factor_mean': float(dup.mean()) if dup.size else 0.0,
        'dup_factor_max': int(dup.max()) if dup.size else 0,
    }
    if not hit:
        _log('[soft-index] join 命中 0 行：标签与数据集无交集。'
             '先确认散列口径一致（见 pos_hash.py），不要把「缓存陈旧」'
             '与「这批局面真的没标签」混为一谈 —— 后者根本不会命中缓存。')
        diag['n_hit'] = 0
        diag['n_rows'] = 0
        return diag
    meta['extra'] = {
        # n_hit = 命中**局面**数（去重后），n_rows = 覆盖的**行**数
        'n_hit': int(np.unique(lab_h[label_index]).size),
        'n_rows': hit,
        'hit_rate': diag['hit_rate_vs_labels'],
        'n_label': int(want.size),
    }
    diag['n_hit'] = meta['extra']['n_hit']
    diag['n_rows'] = hit
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(
        out_path,
        idx=row_index.astype(np.int64),
        policy=pol,
        n_hit=np.int64(meta['extra']['n_hit']),
        n_rows=np.int64(hit),
        hit_rate=np.float64(diag['hit_rate_vs_labels']),
        n_label=np.int64(int(want.size)),
        cache_key=np.str_(key),
        cache_meta=np.str_(json.dumps(meta, sort_keys=True, ensure_ascii=False)),
    )
    diag['out'] = out_path
    return diag
