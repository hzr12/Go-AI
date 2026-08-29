"""
19x19 监督训练脚本（AlphaGoZero 风格策略+价值网络）。

依赖 scripts/build_dataset.py 产出的紧凑 npz 数据集。监督学习不涉及
自我对弈，因此绕开了 H1/H2/H4/H5/L1 等自我对弈性能问题。

用法:
    python scripts/train_sft.py --data data/sft_dataset.npz --device cuda --use-amp \
        --batch-size 512 --epochs 5 --save-every 2000 --out models/sft.pt
"""

import argparse
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.networks.alphanet import AlphaGoNet
from src.data.dataset import SupervisedDataset


def maybe_autocast(device):
    """仅在 CUDA 上开启 fp16 autocast，CPU 返回 nullcontext。"""
    if device == 'cuda':
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            try:
                return torch.amp.autocast('cuda')
            except TypeError:
                return torch.cuda.amp.autocast()
    return nullcontext()


def load_dataset(path):
    d = np.load(path, allow_pickle=False)
    return SupervisedDataset({k: d[k] for k in d.files})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--use-amp', action='store_true')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--board-size', type=int, default=19)
    ap.add_argument('--save-every', type=int, default=2000)
    ap.add_argument('--out', default='models/sft.pt')
    # 注意力相关
    ap.add_argument('--attention-mode', default='mix',
                    choices=['none', 'mix', 'all'],
                    help='主干注意力模式: none=纯卷积, mix=卷积+注意力混合, all=全注意力')
    ap.add_argument('--num-attention-layers', type=int, default=4,
                    help='mix 模式下注意力块数量')
    ap.add_argument('--num-heads', type=int, default=4, help='多头注意力头数')
    ap.add_argument('--attention-dropout', type=float, default=0.0)
    ap.add_argument('--attn-mode', default='global',
                    choices=['global', 'window', 'axial'],
                    help='注意力计算模式: global=全配对, window=滑动窗口, axial=轴向')
    ap.add_argument('--attn-window', type=int, default=7, help='window 模式窗口边长')
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--compile', action='store_true',
                    help='用 torch.compile 融合算子（GPU 上约 20-40%% 提速，首次迭代较慢）')
    args = ap.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    use_amp = args.use_amp or (device == 'cuda')
    # V100 是 Volta：无 bf16，使用 fp16
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(min(8, os.cpu_count() or 8))

    dataset = load_dataset(args.data)
    n = len(dataset)
    # 留出 held-out 集用于 top-1 准确率监控
    n_train = int(n * 0.98)
    idx_all = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx_all)
    train_idx = idx_all[:n_train]
    eval_idx = idx_all[n_train:]

    model = AlphaGoNet(
        in_channels=12,
        backbone_channels=128,
        backbone_res_blocks=12,
        attention_mode=args.attention_mode,
        num_attention_layers=args.num_attention_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        attn_mode=args.attn_mode,
        attn_window=args.attn_window,
        action_size=args.board_size * args.board_size + 1,  # +1 为 pass 类别
    ).to(device)

    # torch.compile 融合算子（GPU 上约 20-40% 提速）。首迭代有编译开销，
    # 不支持或失败时自动回退到 eager 模式。
    if args.compile:
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, dynamic=False)
                # torch.compile 惰性，错误在首次前向才暴露；用 dummy 输入
                # 触发真实编译并捕获异常（如 CPU 缺 MSVC cl 编译器）。
                with torch.no_grad():
                    dummy = torch.zeros(1, 12, args.board_size, args.board_size,
                                        device=device)
                    model(dummy)
                print("[train] 已启用 torch.compile 算子融合")
            except Exception as e:  # noqa: BLE001
                print(f"[train] torch.compile 不可用，回退 eager: {e}")
                model = AlphaGoNet(
                    in_channels=12, backbone_channels=128,
                    backbone_res_blocks=12, attention_mode=args.attention_mode,
                    num_attention_layers=args.num_attention_layers,
                    num_heads=args.num_heads, attention_dropout=args.attention_dropout,
                    attn_mode=args.attn_mode, attn_window=args.attn_window,
                    action_size=args.board_size * args.board_size + 1,
                ).to(device)
        else:
            print("[train] 当前 torch 版本不支持 torch.compile，跳过")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    bs = args.batch_size
    step = 0
    best_eval_acc = -1.0
    t0 = time.time()

    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        model.train()
        epoch_loss = 0.0
        n_batches = (len(train_idx) + bs - 1) // bs
        for i in range(n_batches):
            sel = train_idx[i * bs:(i + 1) * bs]
            state, move_t, value_t = dataset.sample_batch(sel, device)
            with maybe_autocast(device):
                policy_logits, value_pred = model(state)
                policy_loss = F.cross_entropy(policy_logits.float(), move_t)
                value_loss = F.mse_loss(value_pred.float().squeeze(), value_t.squeeze())
                loss = policy_loss + value_loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            epoch_loss += loss.item()

            if step % args.eval_every == 0 and len(eval_idx) > 0:
                acc = evaluate(model, dataset, eval_idx, bs, device)
                print(f"[step {step}] train_loss={epoch_loss / max(1, (i + 1)):.4f} "
                      f"eval_top1={acc:.4f} scale={scaler.get_scale():.0f} "
                      f"elapsed={time.time() - t0:.0f}s")
                if acc > best_eval_acc:
                    best_eval_acc = acc
                    torch.save(model.state_dict(), args.out)
                    print(f"  -> 保存最佳模型至 {args.out}")
            if step % args.save_every == 0:
                torch.save(model.state_dict(), args.out + '.latest')
        scheduler.step()
        print(f"epoch {epoch + 1}/{args.epochs} done, loss={epoch_loss / max(1, n_batches):.4f}")

    # 最终保存
    torch.save(model.state_dict(), args.out)
    print(f"训练完成。最佳 eval_top1={best_eval_acc:.4f}，模型已保存至 {args.out}")
    print(f"总耗时 {time.time() - t0:.0f}s")


@torch.no_grad()
def evaluate(model, dataset, eval_idx, bs, device):
    model.eval()
    correct = 0
    total = 0
    for i in range(0, len(eval_idx), bs):
        sel = eval_idx[i:i + bs]
        state, move_t, _ = dataset.sample_batch(sel, device)
        with maybe_autocast(device):
            policy_logits, _ = model(state)
        pred = policy_logits.argmax(dim=-1)
        correct += int((pred.cpu() == move_t.cpu()).sum())
        total += len(sel)
    return correct / max(1, total)


if __name__ == '__main__':
    main()
