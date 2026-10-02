"""V7 在 stdata 上的端到端冒烟训练（tracer bullet）。

它回答一个问题：**整条链真的能学吗** —— stdata 读取器 → V7 标签 → NbtTfNet
→ 12 项 loss → 反传 → 优化器，一步都不缺地跑几十步，看逐项 loss 是否下降。

为什么先做这个，而不是直接写 dataset 类 + 预取 + DDP + shell
-----------------------------------------------------------
上面那些是**规模**问题；这个是**正确性**问题。规模问题可以慢慢加，正确性
问题一旦埋进去，每个单测都绿、每个形状都对，然后训练几百步才发现 value 头
的尺度反了 —— 那时候要回头的东西比现在多得多。

所以这里的代码刻意**不追求快**：单进程、batch 8、几百行数据、几十步。
它要的是「每一项 loss 都从随机噪声级别掉下来」。

⚠ **只读真实数据，不写任何产物**（不落盘 checkpoint、不改仓库文件）。

用法::

    python scripts/smoke_train_v7.py --files 40 --rows 256 --steps 40
"""

import argparse
import io
import os
import sys
import tarfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.katago_npz import read_katago_npz, to_v7_labels   # noqa: E402
from src.networks.katago_v7_loss import KataGoV7Loss            # noqa: E402

ARCHIVE = os.path.join('katago', 'stdata', '2026-08-25npzs.tgz')
NETWORK = 'kata1-tf3-b11c768'

#: `to_v7_labels` 附带的**诊断**字段，不是 loss 的标签，也不进模型。
_DIAGNOSTIC = ('komi', 'q_value', 'score_mean_hint', 'lead_hint')


def load_rows(archive, network, max_files, max_rows):
    """从 tgz 流式读前若干个 npz，合并标签（只留 19×19 的行）。"""
    t = tarfile.open(archive, 'r|gz')
    parts, raw_rows, nfiles = [], 0, 0
    try:
        for m in t:
            if not m.name.endswith('.npz'):
                continue
            d = np.load(io.BytesIO(t.extractfile(m).read()))
            raw_rows += d['binaryInputNCHWPacked'].shape[0]
            lb = to_v7_labels(read_katago_npz(d, network=network), policy_topk=16)
            if lb['_kept']:
                parts.append(lb)
            nfiles += 1
            if nfiles >= max_files or sum(p['_kept'] for p in parts) >= max_rows:
                break
    finally:
        t.close()
    print(f'读了 {nfiles} 个 npz（原始 {raw_rows} 行），'
          f'19×19 保留 {sum(p["_kept"] for p in parts)} 行', flush=True)
    out = {'_kept': sum(p['_kept'] for p in parts)}
    for k in parts[0]:
        if k.startswith('_') or k in _DIAGNOSTIC:
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


def split_inputs(lb, idxs, device):
    """把 `spatial` / `global` 取成模型输入，其余留给 loss。"""
    sp = torch.as_tensor(lb['spatial'][idxs], dtype=torch.float32, device=device)
    gl = torch.as_tensor(lb['global'][idxs], dtype=torch.float32, device=device)
    sub = {}
    for k, v in lb.items():
        if k.startswith('_') or k in _DIAGNOSTIC or k in ('spatial', 'global'):
            continue
        if isinstance(v, tuple):
            sub[k] = tuple(torch.as_tensor(x[idxs], device=device) for x in v)
        elif isinstance(v, dict):
            sub[k] = {kk: torch.as_tensor(vv[idxs], device=device)
                      for kk, vv in v.items()}
        else:
            sub[k] = torch.as_tensor(v[idxs], device=device)
    return sp, gl, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', default=ARCHIVE)
    ap.add_argument('--network', default=NETWORK)
    ap.add_argument('--files', type=int, default=40)
    ap.add_argument('--rows', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    from src.inference import _build_for_in_channels
    device = torch.device(args.device)
    torch.manual_seed(0)

    lb = load_rows(args.archive, args.network, args.files, args.rows)
    n = min(lb['_kept'], args.rows)
    print(f'用 {n} 行训练，batch={args.batch}，steps={args.steps}', flush=True)

    model = _build_for_in_channels(22, use_checkpoint=False).to(device)
    model.train()
    print(f'模型参数量 {sum(p.numel() for p in model.parameters()):,}', flush=True)
    lossf = KataGoV7Loss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    g = torch.Generator().manual_seed(1)
    first = last = None
    t0 = time.perf_counter()
    for step in range(args.steps):
        idx = np.sort(np.random.default_rng(step).choice(
            n, size=min(args.batch, n), replace=False))
        sp, gl, sub = split_inputs(lb, idx, device)
        out = model(sp, gl)
        res = lossf(out, sub)
        opt.zero_grad(set_to_none=True)
        res['loss'].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        snap = {k: float(v) for k, v in res['weighted'].items()}
        snap['_loss'] = float(res['loss'])
        snap['_gnorm'] = float(gn)
        if step == 0:
            first = snap
        last = snap
        if step % 10 == 0 or step == args.steps - 1:
            print('step %3d  loss=%10.4f  |g|=%8.3f  %s'
                  % (step, snap['_loss'], snap['_gnorm'],
                     ' '.join('%s=%.3f' % (k, v) for k, v in snap.items()
                              if not k.startswith('_'))), flush=True)
    print('\n用时 %.1fs' % (time.perf_counter() - t0), flush=True)
    print('%-18s %12s %12s %9s' % ('term', 'step0', 'last', 'change'))
    worse = []
    for k in first:
        if k.startswith('_'):
            continue
        a, b = first[k], last[k]
        chg = (b - a) / max(abs(a), 1e-9) * 100
        print('%-18s %12.5f %12.5f %+8.1f%%' % (k, a, b, chg))
        if chg > 5 and abs(a) > 1e-6:
            worse.append((k, a, b, chg))
    if worse:
        print('\n⚠ 变差的项：')
        for k, a, b, c in worse:
            print('   %-16s %.5f -> %.5f (%+.1f%%)' % (k, a, b, c))
    else:
        print('\n✓ 全部 12 项都下降')


if __name__ == '__main__':
    main()
