"""MCTS 搜索速度基准：用于优化前后对比。

为什么需要独立工具
------------------
仓库里没有任何 MCTS 阶段 profiler 或搜索速度基准，于是每个"提速 X 倍"
的说法都只能靠估算。本仓库已经吃过一次亏——scripts/search_arch.py 开头的
记录显示，FLOPs 更高的结构实测反而更快（NPU 上瓶颈是内存带宽而非 FLOPs）。
所以这里只认 wall-clock。

用法
----
    python scripts/bench_mcts.py --model models/az_test/az_9_iter1.pth
    python scripts/bench_mcts.py --model <ckpt> --sims 32 --reps 3 --profile
    python scripts/bench_mcts.py --model <ckpt> --no-rollout
    python scripts/bench_mcts.py --model <ckpt> --threads 1 --threads 3

⚠ expand-topk / rollout-steps 的默认值必须与 selfplay_train.py 的生产默认保持
一致，否则基准衡量的就不是实际要跑的工作点（tests 里有一条守卫测试盯着这点）。

输出
----
    每组配置打印 sims/s 的 mean/min/max，以及（--profile 时）按累计耗时
    排序的阶段热点。同一台机器、同一 workload 下前后对比即可判断优化效果。
"""

import argparse
import cProfile
import io
import os
import pstats
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.game.go_rules import GoBoard
from src.inference import GoAI
from src.search.mcts import MCTS


def build_parser():
    p = argparse.ArgumentParser(description='MCTS 搜索速度基准')
    p.add_argument('--model', default=os.path.join(
        'models', 'az_test', 'az_9_iter1.pth'), help='模型权重路径')
    p.add_argument('--board-size', type=int, default=9)
    p.add_argument('--sims', type=int, default=32, help='每次 search 的模拟数')
    p.add_argument('--reps', type=int, default=3, help='重复次数')
    p.add_argument('--warmup', type=int, default=1, help='预热次数（不计入统计）')
    p.add_argument('--threads', type=int, nargs='+', default=[1, 3],
                   help='要测的 MCTS 线程数组合')
    p.add_argument('--expand-topk', type=int, default=8)
    p.add_argument('--rollout-steps', type=int, default=30)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--no-rollout', action='store_true', help='关闭 rollout')
    p.add_argument('--profile', action='store_true',
                   help='对第一组配置做 cProfile 阶段分解')
    p.add_argument('--profile-top', type=int, default=18)
    return p


def _new_ai(args):
    return GoAI(model_path=args.model, board_size=args.board_size,
                device='cpu', use_amp=True)


def _new_mcts(ai, args, threads):
    return MCTS(
        ai,
        board_size=args.board_size,
        num_threads=threads,
        expand_topk=args.expand_topk,
        temperature=args.temperature,
        spec_prefetch=False,
        leaf_ab_depth=0,
        use_rollout=not args.no_rollout,
        rollout_lambda=0.25,
        rollout_steps=args.rollout_steps,
    )


def _run_once(ai, args, threads):
    """一次固定 workload 的搜索：空盘、指定手数历史、固定模拟数。"""
    mcts = _new_mcts(ai, args, threads)
    board = GoBoard(args.board_size)
    hist = [-1, -1, -3]
    t0 = time.perf_counter()
    mcts.search(board, list(hist), list(hist), 1, simulations=args.sims)
    return time.perf_counter() - t0


def _profile(ai, args, threads):
    pr = cProfile.Profile()
    pr.enable()
    _run_once(ai, args, threads)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats('tottime').print_stats(
        args.profile_top)
    print(buf.getvalue())


def main():
    args = build_parser().parse_args()
    if not os.path.exists(args.model):
        raise SystemExit('模型不存在: {}'.format(args.model))

    ai = _new_ai(args)
    print('model={} board={} sims={} expand_topk={} rollout={}(steps={}) '
          'reps={}'.format(
              args.model, args.board_size, args.sims, args.expand_topk,
              'off' if args.no_rollout else 'on', args.rollout_steps,
              args.reps))

    results = {}
    for threads in args.threads:
        for _ in range(args.warmup):
            _run_once(ai, args, threads)
        times = [_run_once(ai, args, threads) for _ in range(args.reps)]
        sps = [args.sims / t for t in times]
        results[threads] = sps
        print('threads={:<3} wall/s: {}  sims/s mean={:.2f} min={:.2f} '
              'max={:.2f}'.format(
                  threads, ' '.join('{:.2f}'.format(t) for t in times),
                  statistics.mean(sps), min(sps), max(sps)))

    if args.profile:
        print('\n--- cProfile 阶段分解 (threads={}) ---'.format(args.threads[0]))
        _profile(ai, args, args.threads[0])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
