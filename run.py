"""
纯 Python 调用训练脚本（无需命令行环境）。

用法:
    python run.py                  # 运行默认 SFT 训练
    python run.py sft              # 运行 SFT 训练
    python run.py selfplay         # 运行自对弈训练
    python run.py sft --epochs 2   # 覆盖参数
    python run.py sft --help       # 查看 SFT 帮助
"""
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_sft(argv=None):
    """运行 SFT 训练。"""
    from scripts.train_sft import main
    default_argv = [
        'train_sft.py',
        '--data', 'data/sgf_19x19_full.npz',
        '--device', 'cuda',
        '--board-size', '19',
        '--backbone-channels', '192',
        '--backbone-res-blocks', '17',
        '--res-blocks', '8',
        '--convnext-blocks', '4',
        '--attn-blocks', '5',
        '--value-channels', '96',
        '--value-res-blocks', '11',
        '--policy-channels', '128',
        '--policy-layers', '3',
        '--batch-size', '3200',
        '--epochs', '2',
        '--lr', '0.002',
        '--weight-decay', '0.0001',
        '--attention-mode', 'mix',
        '--num-attention-layers', '4',
        '--num-heads', '4',
        '--attn-mode', 'window_global',
        '--attn-window', '7',
        '--attention-dropout', '0.1',
        '--label-smoothing', '0.1',
        '--gradient-accumulation-steps', '1',
        '--use-amp', '1',
        '--use-ema', '1',
        '--compile', '1',
        '--prefetch-workers', '32',
        '--prefetch-depth', '32',
        '--log-every', '100',
        '--eval-every', '1000',
        '--save-every', '500',
        '--early-stop', '1',
        '--early-stop-patience', '3',
        '--out', 'models/sft_19x19_v17.pth',
        '--export-onnx', '0',
        '--swanlab', '1',
        '--c2net', '1',
    ]
    if argv:
        # 合并：默认参数在前，覆盖参数在后（argparse 以后者为准）
        sys.argv = default_argv + argv
    else:
        sys.argv = default_argv
    main()


def run_selfplay(argv=None):
    """运行自对弈训练。"""
    from scripts.selfplay_train import main
    default_argv = [
        'selfplay_train.py',
        '--board-size', '19',
        '--iters', '20',
        '--sims', '48',
        '--parallel-games', '8',
        '--mcts-threads', '3',
        '--onnx-model', 'models/sft_19x19_v17.onnx',
        '--batch-cap', '64',
        '--expand-topk', '16',
        '--buffer-size', '500',
        '--use-rollout', '1',
        '--leaf-ab-depth', '2',
        '--c-puct', '2.0',
        '--virtual-loss', '8.0',
        '--model', 'models/sft_19x19_v17.pth',
        '--out', 'models/az_best.pth',
        '--swanlab', '0',
        '--c2net', '1',
    ]
    if argv:
        sys.argv = default_argv + argv
    else:
        sys.argv = default_argv
    main()


def main():
    """入口：根据子命令路由到对应训练脚本。"""
    commands = {
        'sft': run_sft,
        'selfplay': run_selfplay,
        'sp': run_selfplay,
    }

    if len(sys.argv) < 2 or sys.argv[1] in ('--help', '-h', 'help'):
        print(__doc__.strip())
        print("\n可用命令: sft, selfplay (或 sp)")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in commands:
        print(f"未知命令: {cmd}")
        print("可用命令: " + ", ".join(commands.keys()))
        sys.exit(1)

    # 分离子命令参数和传递给子脚本的参数
    # 支持两种格式:
    #   python run.py sft --epochs 2        (直接传参)
    #   python run.py sft -- --epochs 2     (用 -- 分隔)
    child_argv = []
    if len(sys.argv) > 2:
        if sys.argv[2] == '--':
            child_argv = sys.argv[3:]
        else:
            child_argv = sys.argv[2:]

    commands[cmd](child_argv if child_argv else None)


if __name__ == "__main__":
    main()
