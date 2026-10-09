# -*- coding: utf-8 -*-
"""在真机上把 V7 loss 的**每一个操作数**逐个量一遍（一次性诊断脚本）。

为什么需要它
------------
2026-10-04 云端 910A 的实测症状：

* 每步都被 GradScaler 跳过，缩放值 10240 → 5120 → … → 160 仍 100% 跳过；
* 每步的 NaN 参数计数**完全一样**（211 / 98 / 8）；
* 逐项点名指向 **`score_stdev`**。

而**本机怎么都复现不出来**：fp32 / fp16 / bf16 三种精度下，`score_stdev` 的
前向有限、梯度 NaN 张量 0/322。CPU 与 NPU 的 kernel 实现不同（Welford vs
两遍、softplus 的线性兜底分支、规约的累加顺序），**只有真机能回答**。

而每次真机跑一轮训练要几分钟，且被这个 NaN 挡着根本跑不起来 ⇒ 需要一个
**几十秒、不训练、只量数值**的探针。

它做什么
--------
按训练循环**完全相同**的方式建数据、建模型、开 autocast，跑**一个** batch，然后：

1. 逐项打印 13 项 loss 的值 / 是否有限 / dtype；
2. 对 `score_stdev` 单独打印两个操作数（预测、目标）与中间量
   （softmax 后的 min/max、方差、开方前后）；
3. 反向一次，逐参数统计 inf / nan 数量，并点名是哪些模块；
4. 对任何非有限的操作数，打印它的 dtype 与极值。

用法
----
    python scripts/probe_v7_numerics.py \
        --data data/sgf_19x19_full.npz --v7 1 --device npu --board-size 19 \
        --games-npz data/labels/games.npz

 只读不写：不建 checkpoint、不碰 optimizer、不改任何状态。
 不依赖 torch_npu 也能跑（`--device cpu`）—— 那样只会得到「全都正常」，
  但可以确认脚本本身没写错。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.v7_dataset import V7Dataset  # noqa: E402
from src.networks.katago_v7 import build_katago_v7_net  # noqa: E402
from src.networks.katago_v7_loss import (  # noqa: E402
    SCORE_STDEV_TARGET_FLOOR,
    KataGoV7Loss,
)
from scripts.train_sft import (  # noqa: E402
    build_v7_stage1_loss,
    v7_batch_features,
    v7_loss_labels,
)


def _fin(v):
    """(值, 是否有限, dtype, |极值|) —— 打印用的统一格式。"""
    t = v if torch.is_tensor(v) else torch.as_tensor(v)
    f = t.detach().float()
    finite = bool(torch.isfinite(f).all())
    amax = float(f[torch.isfinite(f)].abs().max()) if finite else float('nan')
    return float(f.mean()), finite, str(t.dtype).replace('torch.', ''), amax


def _row(name, v, note=''):
    mean, finite, dt, amax = _fin(v)
    print('  %-18s %-16s %-9s %-12s %s'
          % (name, 'finite' if finite else ' NON-FINITE', dt,
             '%.6g' % amax, note))
    return finite


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--v7', type=int, default=1)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--board-size', type=int, default=19)
    ap.add_argument('--games-npz', default=None)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--amp', type=int, default=1,
                    help='1 = 开 autocast（910A 上必须；fp32 对照用 0）')
    args = ap.parse_args()

    dev = torch.device(args.device)

    # ---- 数据：与训练同一条路（board 级 V7Dataset + 可选 sidecar）----
    kw = {}
    if args.games_npz:
        z = np.load(args.games_npz, allow_pickle=False)
        gk = z['g_komi'].astype(np.float32)
        gr = z['g_rules'].astype(np.int64)
        z.close()
        kw = {'game_komi': gk, 'game_rules_flags': gr}
    from scripts.train_sft import load_dataset
    base = load_dataset(args.data)
    keys = ('boards', 'my_hist', 'op_hist', 'ko', 'moves', 'values',
            'to_play', 'game_ids', 'game_weights', 'winrates')
    data = {k: getattr(base, k) for k in keys if hasattr(base, k)}
    ds = V7Dataset(data, n_channels=12, verify_history=None, **kw)
    print('[probe] 数据 %d 行 | sidecar=%s'
          % (len(ds.moves), args.games_npz or '(无 ⇒ 贴目按 0)'))

    idx = np.arange(min(args.batch_size, len(ds.moves)))
    sp, gl = v7_batch_features(ds, idx)
    sp_t = torch.from_numpy(sp.astype(np.float32)).to(dev)
    gl_t = torch.from_numpy(gl.astype(np.float32)).to(dev)
    lbl = ds._build_labels(idx)
    moves = np.asarray(ds.moves)[idx]

    # ---- 模型 + loss ----
    net = build_katago_v7_net(board_size=args.board_size).to(dev)
    lossf = build_v7_stage1_loss(action_size=args.board_size ** 2 + 1).to(dev)

    amp_dtype = torch.float16 if args.amp else torch.float32
    print('[probe] device=%s amp=%s dtype(spatial)=%s'
          % (dev.type, args.amp, sp_t.dtype))

    ctx = (torch.autocast(device_type=dev.type, dtype=amp_dtype)
           if args.amp else torch.no_grad())
    # **不能**用 `no_grad` 跑前向：第 [4] 段要反向一次，而反向需要计算图。
    #   打印用的副本另做（`out_det`），loss 用带图的那份（`out`）。
    with ctx:
        out = net(sp_t, gl_t)
    out_det = {k: (v.detach() if torch.is_tensor(v) else v)
               for k, v in out.items()}

    # ---- ① 模型输出逐个 ----
    print('\n[1] 模型输出（forward 之后）')
    print('  %-18s %-16s %-9s %s' % ('键', '状态', 'dtype', '|max|'))
    for k in sorted(out_det):
        if torch.is_tensor(out_det[k]) and out_det[k].is_floating_point():
            _row(k, out_det[k])

    # ---- ② score_stdev 的两个操作数（点名的那一项）----
    print('\n[2] score_stdev 的操作数（逐项点名指向的就是这一项）')
    print('  %-18s %-16s %-9s %s' % ('量', '状态', 'dtype', '|max|'))
    sb = out_det['scorebelief_logits']
    if sb is not None and torch.is_tensor(sb):
        p = torch.softmax(sb.float(), dim=-1)
        mu = p.mean(dim=-1, keepdim=True)
        dev2 = (p - mu).pow(2).mean(dim=-1)
        std_raw = dev2.sqrt()
        std_clamped = std_raw.clamp_min(SCORE_STDEV_TARGET_FLOOR)
        _row('softmax min', p.min(dim=-1).values)
        _row('softmax max', p.max(dim=-1).values)
        _row('方差(两遍)', dev2, 'sqrt 前')
        _row('std 原始', std_raw, 'fp16 下溢成 0 的风险点')
        _row('std 下界后', std_clamped,
             '下界 %g ⇒ v→0 时梯度恒 0（不是 fp16 溢出保护，见 '
             'test_measured_stdev_target_gradient_stays_finite_and_small）'
             % (SCORE_STDEV_TARGET_FLOOR,))
    _row('预测 score_stdev', out_det.get('score_stdev'),
         '= 20·softplus(s[:,1], beta=1)')
    # `s[:,1]`（scores 头的原始第 1 路）取不到：它没有单独暴露成输出键。
    #   若上面的「预测」或「std 原始」有一项 NON-FINITE，那一项就是根因；
    #   需要更细的归因时在这里按需 hook `net.value_head.scores`。

    # ---- ③ 逐项 loss ----
    print('\n[3] 13 项 loss')
    print('  %-18s %-16s %-9s %s' % ('项', '状态', 'dtype', '|max|'))
    res = lossf(out, v7_loss_labels(lbl, torch.from_numpy(moves).to(dev)))
    for k in sorted(res['terms']):
        _row(k, res['terms'][k],
             '系数=%g' % lossf.coeff.get(k, float('nan')))
    print()
    print('  加权总 loss   : %s | total_finite=%s'
          % ('finite' if bool(torch.isfinite(res['loss'])) else ' NON-FINITE',
             res['total_finite']))
    print('  逐项点名      : %s' % (res['nonfinite_terms'] or '（无）'))
    print('  坏在操作数    : %s' % (res['nonfinite_operands'] or '（无）'))
    print('  净化行          : %s'
          '   （`项名`=w==0 上 p 非有限；`项名:w_bad`=权重坏'
          '（inf 或 >exp(10)））'
          % (res['sanitized_rows'] or '（无）'))

    # ---- ④ 反向：谁收到了 inf/nan ----
    # **必须在 clip 之前看**（与 train_sft 同一个坑）：`clip_grad_norm_` 在
    #   `total_norm = inf` 时算 `clip_coef = 0` 并原地 `mul_(0)` ⇒ `inf × 0 = NaN`
    #   ⇒ clip 之后「inf 个数」结构上恒为 0，看到的 nan 全是 clipper 造的。
    print('\n[4] 反向一次（统计 inf / nan 的参数）—— clip **之前**')
    net.zero_grad(set_to_none=True)
    res['loss'].backward()
    _report_grads(net)

    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    print('\n[4b] clip_grad_norm_ 返回的总范数 = %s'
          % ('finite' if bool(torch.isfinite(torch.as_tensor(float(gn))))
             else ' NON-FINITE（clip 前确有 inf）'))
    print('     clip 后 inf 元素数必然是 0（clip_coef=0 ⇒ inf×0=NaN），'
          '所以「NaN 模块名单」才是真正出过 inf 的位置。')
    _report_grads(net, after_clip=True)


def _report_grads(net, after_clip=False):
    tag = 'clip 后' if after_clip else 'clip 前'
    tot_nan = tot_inf = 0
    bad_mod = {}
    for name, p in net.named_parameters():
        if p.grad is None:
            continue
        n = int(torch.isnan(p.grad).sum())
        i = int(torch.isinf(p.grad).sum())
        tot_nan += n
        tot_inf += i
        if n or i:
            top = '.'.join(name.split('.')[:2])
            cur = bad_mod.get(top, (0, 0))
            bad_mod[top] = (cur[0] + i, cur[1] + n)
    n_grad = sum(1 for p in net.parameters() if p.grad is not None)
    print('  [%s] 参数张量 %d，收到梯度的 %d' % (tag, len(list(net.parameters())),
                                       n_grad))
    print('  [%s] NaN 元素 %d | Inf 元素 %d' % (tag, tot_nan, tot_inf))
    if bad_mod:
        print(' [%s] 按模块点名：' % tag)
        for m, (i, n) in sorted(bad_mod.items(), key=lambda x: -(x[1][0] + x[1][1])):
            print('     %-30s inf=%-6d nan=%d' % (m, i, n))
    else:
        print(' [%s] 梯度全部有限' % tag)

    print('\n[probe] 若 [3] 报 NON-FINITE 而 [4] 也报 NaN：把上面两段的点名'
          '合起来就是根因（哪一项 / 哪个操作数 / 哪个模块）。')


if __name__ == '__main__':
    main()