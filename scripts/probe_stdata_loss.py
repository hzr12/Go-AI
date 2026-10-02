"""用 stdata 一行批次跑通 V7 的 12 项 loss（诊断脚本，不是单测）。

为什么需要它
------------
`tests/test_katago_v7_loss.py` 全部用**合成**标签，形状对、梯度通，但那只能
证明公式自洽。真正的风险在**标签契约**上：stdata 的 7 个目标数组与 spec §5.4
的标签 dict 之间是否有字段缺失、量纲不一致、掩码语义错位 —— 这些只有在
**真实官方数据**上跑一遍才会暴露。

单测里跑这个脚本是可行的（见 `test_katago_npz.py::test_loss_runs_on_real_official_targets`），
但它要解 1.5 GB 的 tgz，太慢，所以这里只做脚本、单测里用更小的批次。

用法::

    python scripts/probe_stdata_loss.py [批次路径] [网络名]
"""

import io
import os
import sys
import tarfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.katago_npz import read_katago_npz, to_v7_labels   # noqa: E402
from src.networks.katago_v7_loss import KataGoV7Loss            # noqa: E402

DEFAULT_ARCHIVE = r'F:\AI\Go-AI\katago\stdata\2026-08-25npzs.tgz'
DEFAULT_NETWORK = 'kata1-tf3-b11c768'

#: 不进 loss 的键（模型输入或纯诊断），逐个列出而不是前缀过滤 ——
#: 前缀过滤会在新增诊断键时静默把它当标签喂进去。
_SKIP = {'spatial', 'board_mask', 'global', 'komi', 'q_value',
         'score_mean_hint', 'lead_hint'}


def load_first_npz(archive, limit=1):
    """流式取前 ``limit`` 个 npz（``getmembers()`` 要扫完整个 1.5 GB 归档）。"""
    out = []
    t = tarfile.open(archive, 'r|gz')
    try:
        for m in t:
            if m.name.endswith('.npz'):
                out.append(np.load(io.BytesIO(t.extractfile(m).read())))
                if len(out) >= limit:
                    break
    finally:
        t.close()
    return out


def slice_labels(lb, b):
    out = {}
    for k, v in lb.items():
        if k in _SKIP or k.startswith('_'):
            continue
        if isinstance(v, tuple):
            out[k] = tuple(x[:b] for x in v)
        elif isinstance(v, dict):
            out[k] = {kk: vv[:b] for kk, vv in v.items()}
        elif hasattr(v, 'shape') and getattr(v, 'shape', ())[:1] == (lb['_kept'],):
            out[k] = v[:b]
        else:
            out[k] = v
    return out


def concat_labels(parts):
    """把多个文件的标签拼起来（tuple / dict 字段要分别处理）。"""
    parts = [p for p in parts if p['_kept'] > 0]
    if not parts:
        return None
    out = {'_kept': sum(p['_kept'] for p in parts),
           '_dropped': sum(p['_dropped'] for p in parts),
           '_network': parts[0]['_network']}
    for k in parts[0]:
        if k.startswith('_'):
            continue
        vs = [p[k] for p in parts]
        if isinstance(vs[0], tuple):
            out[k] = tuple(np.concatenate([v[j] for v in vs])
                           for j in range(len(vs[0])))
        elif isinstance(vs[0], dict):
            out[k] = {kk: np.concatenate([v[kk] for v in vs]) for kk in vs[0]}
        else:
            out[k] = np.concatenate(vs)
    return out


def main(archive=DEFAULT_ARCHIVE, network=DEFAULT_NETWORK, max_rows=8,
         max_files=40):
    raws = load_first_npz(archive, limit=max_files)
    print(f'取到 {len(raws)} 个 npz，原始行数合计 '
          f'{sum(r["binaryInputNCHWPapped" if False else "binaryInputNCHWPacked"].shape[0] for r in raws)}')
    parts = []
    for raw in raws:
        d = read_katago_npz(raw, network=network)
        cand = to_v7_labels(d, policy_topk=16)
        if cand['_kept']:
            parts.append(cand)
        if sum(p['_kept'] for p in parts) >= max_rows:
            break
    lb = concat_labels(parts)
    if lb is None:
        raise SystemExit(f'前 {len(raws)} 个文件里没有 19×19 的行；'
                         f'调大 max_files 或换一个批次')
    print(f'  网络={lb["_network"]}  kept={lb["_kept"]}  dropped={lb["_dropped"]}  '
          f'spatial={lb["spatial"].shape}')

    print('\n---- 标签体检 ----')
    print('  outcome 分布       :', np.bincount(lb['outcome'], minlength=3))
    print('  ownership 取值     :', np.unique(lb['ownership']))
    print('  seki 取值          :', np.unique(lb['seki']))
    print('  scoring 值域       :', float(lb['scoring'].min()),
          float(lb['scoring'].max()))
    print('  futurepos 取值     :', np.unique(lb['futurepos']))
    print('  score 取值(前 8)   :', np.unique(lb['score'])[:8])
    print('  score_distr 行和   :', np.round(lb['score_distr'].sum(1)[:4], 6))
    print('  score_distr 非零/行:', int((lb['score_distr'] != 0).sum(1).max()))
    print('  policy top16 计数和:', np.round(
        lb['policy_player_sparse'][1].sum(1)[:4], 1))
    for k in sorted(lb['w']):
        print(f'  w[{k}] 取值        :', np.unique(lb['w'][k]))
    print('  komi 取值          :', np.unique(lb['komi']))

    b = int(min(max_rows, lb['_kept']))
    g = torch.Generator().manual_seed(0)
    out = {
        'policy_logits': torch.randn(b, 2, 362, generator=g),
        'outcome_logits': torch.randn(b, 3, generator=g),
        'score_mean': torch.randn(b, generator=g),
        'score_stdev': torch.rand(b, generator=g) * 10 + 1,
        'lead': torch.randn(b, generator=g),
        'ownership_pretanh': torch.randn(b, 1, 19, 19, generator=g),
        'scoring': torch.randn(b, 1, 19, 19, generator=g),
        'futurepos': torch.randn(b, 2, 19, 19, generator=g),
        'seki_logits': torch.randn(b, 4, 19, 19, generator=g),
        'scorebelief_logits': torch.randn(b, 842, generator=g),
    }
    sub = slice_labels(lb, b)
    print(f'\n---- 用 b={b} 行真实标签跑 12 项 ----')
    res = KataGoV7Loss()(out, sub)
    print('  LOSS =', float(res['loss']))
    for k, v in res['weighted'].items():
        print('  %-18s weighted=%-14.6f raw=%-14.6f'
              % (k, float(v), float(res['terms'][k])))
    print('  seki 自适应系数 =', float(res['seki_adaptive_scale']))
    return res


if __name__ == '__main__':
    main(*(sys.argv[1:3] if len(sys.argv) > 2 else ()))
