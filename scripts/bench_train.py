"""Minimal training benchmark: measures samples/sec through the training loop.

Usage:
    python scripts/bench_train.py --device cuda --batch-size 512 --board-size 19
    python scripts/bench_train.py --device cpu --batch-size 128 --board-size 19
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from src.networks.alphanet import AlphaGoNet


def make_fake_dataset(n, board_size):
    """Create a minimal fake dataset with the right shapes."""
    n_sq = board_size * board_size
    return {
        'boards': np.random.randint(-1, 2, (n, board_size, board_size)).astype(np.int8),
        'my_hist': np.full((n, 3), -1, dtype=np.int16),
        'op_hist': np.full((n, 3), -1, dtype=np.int16),
        'ko': np.full(n, -1, dtype=np.int16),
        'moves': np.random.randint(0, n_sq, n).astype(np.int16),
        'values': np.random.choice([-1, 1], n).astype(np.int8),
        'to_play': np.random.choice([-1, 1], n).astype(np.int8),
    }


def bench(device, board_size, batch_size, warmup_steps, bench_steps,
          channels, res_blocks, attn_layers, heads, attn_mode, attn_window):
    from src.data.dataset import SupervisedDataset

    n_samples = max(1024, (warmup_steps + bench_steps) * batch_size)
    data = make_fake_dataset(n_samples, board_size)
    dataset = SupervisedDataset(data)

    model = AlphaGoNet(
        in_channels=12,
        backbone_channels=channels,
        backbone_res_blocks=res_blocks,
        attention_mode='mix',
        num_attention_layers=attn_layers,
        num_heads=heads,
        attn_mode=attn_mode,
        attn_window=attn_window,
        action_size=board_size * board_size + 1,
    ).to(device)

    # Auto-detect precision
    amp_dtype = torch.float32
    use_scaler = False
    if 'cuda' in str(device) and torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 8:
            amp_dtype = torch.bfloat16
        elif cc[0] >= 7:
            amp_dtype = torch.float16
            use_scaler = True
        elif cc[0] >= 5:
            amp_dtype = torch.float16
            use_scaler = True
    elif 'npu' in str(device):
        amp_dtype = torch.bfloat16

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    idx_all = np.arange(len(dataset))
    rng = np.random.default_rng(42)

    # Warmup
    for step in range(warmup_steps):
        rng.shuffle(idx_all)
        sel = idx_all[:batch_size]
        states_np, moves_np, values_np = dataset.sample_batch_numpy(sel, rng=rng)
        state = torch.from_numpy(states_np).to(device)
        move_t = torch.from_numpy(moves_np).to(device)
        value_t = torch.from_numpy(values_np).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type if 'cuda' in str(device) else 'cpu',
                                dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            policy_logits, value_pred = model(state)
            policy_loss = F.cross_entropy(policy_logits.float(), move_t)
            value_loss = F.mse_loss(value_pred.float().squeeze(), value_t.squeeze())
            loss = policy_loss + value_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if 'cuda' in str(device):
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for step in range(bench_steps):
        rng.shuffle(idx_all)
        sel = idx_all[:batch_size]
        states_np, moves_np, values_np = dataset.sample_batch_numpy(sel, rng=rng)
        state = torch.from_numpy(states_np).to(device)
        move_t = torch.from_numpy(moves_np).to(device)
        value_t = torch.from_numpy(values_np).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type if 'cuda' in str(device) else 'cpu',
                                dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            policy_logits, value_pred = model(state)
            policy_loss = F.cross_entropy(policy_logits.float(), move_t)
            value_loss = F.mse_loss(value_pred.float().squeeze(), value_t.squeeze())
            loss = policy_loss + value_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if 'cuda' in str(device):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_samples = bench_steps * batch_size
    samples_per_sec = total_samples / elapsed
    ms_per_step = elapsed / bench_steps * 1000

    # Memory
    if 'cuda' in str(device):
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_mem = 0.0

    return {
        'samples_per_sec': samples_per_sec,
        'ms_per_step': ms_per_step,
        'peak_mem_gb': peak_mem,
        'elapsed': elapsed,
        'total_samples': total_samples,
        'bench_steps': bench_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--board-size', type=int, default=19)
    ap.add_argument('--warmup-steps', type=int, default=5)
    ap.add_argument('--bench-steps', type=int, default=20)
    ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--res-blocks', type=int, default=12)
    ap.add_argument('--attn-layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--attn-mode', default='sparse', choices=['sparse', 'window', 'global', 'none'])
    ap.add_argument('--attn-window', type=int, default=7)
    args = ap.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Config: channels={args.channels} res_blocks={args.res_blocks} "
          f"attn_layers={args.attn_layers} heads={args.heads} mode={args.attn_mode} "
          f"window={args.attn_window} batch={args.batch_size} board={args.board_size}")
    print(f"Warmup: {args.warmup_steps} steps, Benchmark: {args.bench_steps} steps")
    print("-" * 60)

    result = bench(
        device, args.board_size, args.batch_size,
        args.warmup_steps, args.bench_steps,
        args.channels, args.res_blocks, args.attn_layers, args.heads,
        args.attn_mode, args.attn_window,
    )

    print(f"Results:")
    print(f"  Throughput:     {result['samples_per_sec']:.0f} samples/sec")
    print(f"  Latency:        {result['ms_per_step']:.1f} ms/step")
    print(f"  Peak memory:    {result['peak_mem_gb']:.2f} GB")
    print(f"  Total time:     {result['elapsed']:.2f}s ({result['total_samples']} samples)")
    print(f"  Accuracy:       {result['samples_per_sec'] / args.batch_size:.1f} batches/sec")


if __name__ == '__main__':
    main()
