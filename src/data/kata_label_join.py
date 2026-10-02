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

⚠ **同一局面会在多局里出现**（transposition）。`pos_hash` 因此在两侧都可能有
重复，join 是**多对多**的。默认 `max_repeats=1`：每个被标注的局面只挂到
**第一个**匹配的数据行。放宽它能把同一个搜索标签喂给多个出现处，但会按
重复次数给高频局面加权（一个布局变例出现 50 次就压过别处 50 倍）。
`dup_factor` 会把实际的重复倍数报出来，>1 的比例高就说明该局面被重复加权。
"""

import os

import numpy as np

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


def scan_dataset_hashes(npz_path, keys=DATASET_KEYS, chunk=_HASH_CHUNK,
                        limit=None, progress=None):
    """流式算主数据集每行的 `pos_hash`。

    Returns:
        ``(hashes uint64 (N,), n_rows)``。``limit`` 给定时只算前 N 行（调试用）。
    """
    z = np.load(npz_path, allow_pickle=False)
    for k in keys:
        if k not in z.files:
            raise KeyError(f'{npz_path} 缺列 {k}；现有 {sorted(z.files)}')
    total = z['boards'].shape[0]
    hi = total if limit is None else min(total, int(limit))
    out = np.empty(hi, dtype=np.uint64)
    for s in range(0, hi, chunk):
        e = min(s + chunk, hi)
        out[s:e] = pos_hash_block(z['boards'][s:e], z['to_play'][s:e],
                                  z['ko'][s:e])
        if progress is not None:
            progress(e, hi)
    return out, hi


def join_by_hash(dataset_hashes, label_hashes, max_repeats=1):
    """按散列 join，返回 ``(row_index, label_index)``。

    Args:
        dataset_hashes: ``(N,)`` uint64，主数据集每行一个。
        label_hashes: ``(M,)`` uint64，标签每行一个。
        max_repeats: 同一局面最多挂到几个数据行。1 = 只取第一个匹配。

    Returns:
        两个等长数组：``row_index``（主数据集行号）、``label_index``（标签行号），
        按 ``row_index`` 升序。

    ⚠ **两侧都可能有重复**。实现用「对标签散列去重 → 排序 → searchsorted」，
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
    rows, labs = [], []
    cursor = 0
    for take in range(1, int(max_repeats) + 1):
        p = lo + (take - 1)
        ok = p < ds_sorted.size
        cand = np.where(ok, ds_sorted[np.clip(p, 0, ds_sorted.size - 1)], 0)
        hit = ok & (cand == uniq_hash)
        if not hit.any():
            break
        rows.append(key[p[hit]].astype(np.int64))
        labs.append(uniq_pos[hit].astype(np.int64))
        cursor += int(hit.sum())
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
                  limit=None, progress_every=0):
    """算 join 并写 sidecar npz。

    Returns:
        诊断 dict（行数 / 命中率 / 重复倍数 / 实际写出的行数）。

    ⚠ **不命中时也返回诊断但不写文件** —— 「标签与数据集无交集」是一个需要
    立刻看见的结论，静默不写文件会让调用方以为跑成功了。
    """
    lab = load_kata_labels(labels_npz)

    def _prog(e, hi):
        if progress_every and (e // _HASH_CHUNK) % progress_every == 0:
            print(f'  [join] 散列 {e}/{hi}', flush=True)

    ds_hash, n_rows = scan_dataset_hashes(dataset_npz, limit=limit,
                                          progress=_prog if progress_every
                                          else None)
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
