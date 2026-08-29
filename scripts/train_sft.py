"""
19x19 监督训练脚本（AlphaGoZero 风格策略+价值网络）。

依赖 scripts/build_dataset.py 产出的紧凑 npz 数据集。监督学习不涉及
自我对弈，因此绕开了 H1/H2/H4/H5/L1 等自我对弈性能问题。

用法:
    python scripts/train_sft.py --data data/sft_dataset.npz --device cuda --use-amp \
        --batch-size 512 --epochs 5 --save-every 2000 --out models/sft.pt
"""

import argparse
import logging
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
from scripts.build_dataset import build


def setup_logging(log_file, level=logging.INFO):
    """配置 logging：同时写文件与输出到控制台（无缓冲，实时可见）。"""
    logger = logging.getLogger('train')
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


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
    """加载单个 .npz 训练集。"""
    d = np.load(path, allow_pickle=False)
    return SupervisedDataset({k: d[k] for k in d.files})


def _concat_dicts(dicts):
    """按相同 key 沿第 0 轴拼接多个数据 dict（字段形状一致）。"""
    out = {}
    for k in dicts[0].keys():
        out[k] = np.concatenate([dd[k] for dd in dicts], axis=0)
    return out


def load_from_path(path, board_size, max_games_per_tgz=0):
    """加载训练数据。

    - 若 path 是文件：按 .npz 加载（兼容原行为）。
    - 若 path 是目录：递归扫描其下所有 .tgz/.tar.gz（用 build_dataset.build 解析）
      与 .npz（直接加载），合并成一个 SupervisedDataset。这样可直接喂一个装着
      多个分片 tgz 的文件夹，无需先手动 build_dataset 成单个 npz。
    """
    if os.path.isdir(path):
        import glob
        tgzs = (sorted(glob.glob(os.path.join(path, '**', '*.tgz'), recursive=True))
                + sorted(glob.glob(os.path.join(path, '**', '*.tar.gz'), recursive=True)))
        npzs = sorted(glob.glob(os.path.join(path, '**', '*.npz'), recursive=True))
        # 直接子目录也作为数据源（build 支持递归解析目录内的 .sgf，
        # 例如 data/games/ 这种已解压的棋谱目录）
        subdirs = sorted(d for d in glob.glob(os.path.join(path, '*'))
                         if os.path.isdir(d) and d not in npzs)
        dicts = []
        n_games_total = 0
        n_skip_total = 0

        def _try_build(src):
            """build 可能对单个分片返回 0 有效局并抛 RuntimeError，这里吞掉并跳过。"""
            try:
                d, n_games, skip = build(src, board_size, max_games_per_tgz)
            except RuntimeError as e:
                logger.warning("[data] 跳过分片 %s：%s", src, e)
                return None, 0, 0
            if n_games == 0:
                logger.warning("[data] 跳过分片 %s：0 有效局（与 --board-size %d 不匹配或空）",
                               src, board_size)
                return None, 0, 0
            return d, n_games, skip

        for tg in tgzs + subdirs:
            d, n_games, skip = _try_build(tg)
            if d is not None:
                dicts.append(d)
                n_games_total += n_games
                n_skip_total += skip
        print(f"[data] 已解析 {len(dicts)} 个有效分片，有效局 {n_games_total}，跳过 {n_skip_total}")
        for npz in npzs:
            dd = np.load(npz, allow_pickle=False)
            dicts.append({k: dd[k] for k in dd.files})
        if not dicts:
            raise RuntimeError(
                f"目录 {path} 下未解析到任何有效棋谱，请检查 --board-size 是否与棋谱尺寸匹配")
        merged = _concat_dicts(dicts)
        print(f"[data] 合并后样本数 {merged['boards'].shape[0]}")
        return SupervisedDataset(merged)
    return load_dataset(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help="单个 .npz 训练集，或包含多个 .tgz/.tar.gz/.npz 的目录（自动合并所有分片）")
    ap.add_argument('--max-games-per-tgz', type=int, default=0,
                    help="目录模式下每个 tgz 最多解析的棋局数（0=全部），用于子采样控制内存")
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
    ap.add_argument('--log-every', type=int, default=50,
                    help='每隔多少 step 打印一次训练日志（loss/lr/吞吐/显存）')
    ap.add_argument('--log-file', default='training.log',
                    help='训练日志文件路径（同时输出到控制台），设为空字符串可关闭文件日志')
    ap.add_argument('--compile', action='store_true',
                    help='用 torch.compile 融合算子（GPU 上约 20-40%% 提速，首次迭代较慢）')
    args = ap.parse_args()

    # 配置日志（控制台 + 文件），统一用 logger 输出便于事后排查
    log_file = args.log_file if args.log_file else None
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("启动训练 | torch=%s | device=%s | amp=%s",
                torch.__version__, args.device, args.use_amp)
    logger.info("配置: data=%s board=%d batch=%d epochs=%d lr=%s wd=%s",
                args.data, args.board_size, args.batch_size, args.epochs,
                args.lr, args.weight_decay)
    logger.info("注意力: mode=%s attn_mode=%s window=%d heads=%d layers=%d dropout=%s compile=%s",
                args.attention_mode, args.attn_mode, args.attn_window,
                args.num_heads, args.num_attention_layers, args.attention_dropout, args.compile)
    logger.info("日志: log_every=%d eval_every=%d save_every=%d out=%s",
                args.log_every, args.eval_every, args.save_every, args.out)
    logger.info("=" * 60)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    use_amp = args.use_amp or (device == 'cuda')
    # V100 是 Volta：无 bf16，使用 fp16
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(min(8, os.cpu_count() or 8))

    dataset = load_from_path(args.data, args.board_size, args.max_games_per_tgz)
    n = len(dataset)
    logger.info("[data] 总样本数=%d | 训练=%d | 验证=%d", n, int(n * 0.98), n - int(n * 0.98))
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
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[model] 参数量=%.2fM | 设备=%s", n_params / 1e6, device)

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
                logger.info("[train] 已启用 torch.compile 算子融合")
            except Exception as e:  # noqa: BLE001
                logger.warning("[train] torch.compile 不可用，回退 eager: %s", e)
                model = AlphaGoNet(
                    in_channels=12, backbone_channels=128,
                    backbone_res_blocks=12, attention_mode=args.attention_mode,
                    num_attention_layers=args.num_attention_layers,
                    num_heads=args.num_heads, attention_dropout=args.attention_dropout,
                    attn_mode=args.attn_mode, attn_window=args.attn_window,
                    action_size=args.board_size * args.board_size + 1,
                ).to(device)
        else:
            logger.info("[train] 当前 torch 版本不支持 torch.compile，跳过")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    bs = args.batch_size
    step = 0
    best_eval_acc = -1.0
    t0 = time.time()
    logger.info("[train] 开始训练 | steps/epoch=%d | 总 steps≈%d",
                (n_train + bs - 1) // bs, args.epochs * (n_train + bs - 1) // bs)

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

            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                mem = torch.cuda.memory_reserved(device) / 1e9 if device == 'cuda' else 0.0
                speed = step * bs / max(1e-6, time.time() - t0)
                logger.info("[step %d/%d] loss=%.4f (p=%.4f v=%.4f) lr=%.2e "
                            "scale=%.0f mem=%.2fGB spd=%.0f s/s elapsed=%.0fs",
                            step, args.epochs * n_batches,
                            loss.item(), policy_loss.item(), value_loss.item(),
                            lr, scaler.get_scale(), mem, speed, time.time() - t0)

            if step % args.eval_every == 0 and len(eval_idx) > 0:
                acc = evaluate(model, dataset, eval_idx, bs, device)
                logger.info("[eval] step=%d train_loss=%.4f eval_top1=%.4f scale=%.0f elapsed=%.0fs",
                            step, epoch_loss / max(1, (i + 1)), acc,
                            scaler.get_scale(), time.time() - t0)
                if acc > best_eval_acc:
                    best_eval_acc = acc
                    torch.save(model.state_dict(), args.out)
                    logger.info("  -> 保存最佳模型至 %s", args.out)
            if step % args.save_every == 0:
                torch.save(model.state_dict(), args.out + '.latest')
        scheduler.step()
        logger.info("epoch %d/%d done, loss=%.4f", epoch + 1, args.epochs,
                    epoch_loss / max(1, n_batches))

    # 最终保存
    torch.save(model.state_dict(), args.out)
    logger.info("训练完成。最佳 eval_top1=%.4f，模型已保存至 %s", best_eval_acc, args.out)
    logger.info("总耗时 %.0fs", time.time() - t0)


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
