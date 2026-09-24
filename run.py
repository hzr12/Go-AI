"""
纯 Python 调用训练脚本（无需命令行环境）。

用法:
    python run.py                  # 运行默认 SFT 训练
    python run.py sft              # 运行 SFT 训练
    python run.py selfplay         # 运行自对弈训练
    python run.py sft --epochs 2   # 覆盖参数
    python run.py sft --help       # 查看 SFT 帮助
    python run.py --sh shell/train_npu_2card.sh --epochs 3   # 运行 shell 脚本

--sh 说明（云端平台适配）:
    平台只接受 `--参数名 参数值`、不能设环境变量、启动文件必须是 .py。
    因此多卡不用 `torchrun run.py ...`，而是把整条训练命令（含 torchrun）
    放进 shell/*.sh，再由本入口以 `--sh <路径>` 拉起——入口仍是 .py、参数仍是
    `--名 值`，bash 由 run.py 内部调用。脚本末尾的 "$@" 会让追加参数覆盖默认值。
"""
import subprocess
import shutil
import sys
import os

# 确保项目根目录在 path 中
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _list_shell_scripts():
    """列出仓库内可用的 shell 脚本（相对仓库根的路径）。"""
    d = os.path.join(ROOT, 'shell')
    if not os.path.isdir(d):
        return []
    return sorted('shell/' + f for f in os.listdir(d) if f.endswith('.sh'))


def run_sh(argv=None):
    """运行 shell 脚本，并把后续参数原样透传给脚本。

    - 不捕获 stdout/stderr：训练日志实时可见（依赖这个行为）
    - 退出码原样传出，供 CI/平台感知失败
    - 脚本路径相对仓库根解析，cwd 固定为仓库根（相对数据路径处处一致）
    """
    argv = list(argv or [])
    if not argv:
        print(__doc__.strip())
        print("\n用法: python run.py --sh shell/<脚本>.sh [传给脚本的参数...]")
        sys.exit(1)

    script = argv[0]
    passthrough = argv[1:]
    spath = script if os.path.isabs(script) else os.path.join(ROOT, script)
    spath = os.path.normpath(spath)

    if not os.path.isfile(spath):
        print(f"[run] 找不到脚本: {script}")
        avail = _list_shell_scripts()
        if avail:
            print("[run] 可用脚本（用 python run.py --sh <路径> 调用）:")
            for a in avail:
                print("   ", a)
        else:
            print("[run] shell/ 目录下暂无 .sh 脚本")
        sys.exit(1)

    bash = shutil.which('bash') or shutil.which('sh')
    if not bash:
        print("[run] 未找到 bash/sh，无法执行 .sh 脚本。")
        print("[run] Windows 可安装 Git Bash；或直接用 python run.py sft <参数>。")
        sys.exit(1)

    shown = ' '.join(passthrough) if passthrough else '(无额外参数)'
    # 注意：这里刻意显示用户传入的原路径，而不是 relpath(spath, ROOT)——
    # 传入绝对路径且位于另一个盘符时，Windows 的 relpath 会抛 ValueError。
    print(f"[run] 执行 {script}  透传: {shown}", flush=True)
    proc = subprocess.run([bash, spath] + passthrough, cwd=ROOT)
    sys.exit(proc.returncode)


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
        '--batch-size', '4096',
        '--epochs', '2',
        '--lr', '0.00256',
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
        '--compile-mode', 'reduce-overhead',
        '--prefetch-workers', '32',
        '--prefetch-depth', '32',
        '--log-every', '100',
        '--eval-every', '2000',
        '--save-every', '1000',
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
        '--batch-cap', '64',
        '--expand-topk', '16',
        '--buffer-size', '500',
        '--use-rollout', '1',
        '--leaf-ab-depth', '2',
        '--c-puct', '2.0',
        '--virtual-loss', '8.0',
        '--td', '1',
        '--grad-accum-steps', '1',
        '--mcts-vector-backup', '1',
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


COMMANDS = {
    'sft': run_sft,
    'selfplay': run_selfplay,
    'sp': run_selfplay,
    'sh': run_sh,
    'shell': run_sh,
}


def main():
    """入口：根据子命令路由到对应训练脚本。"""
    commands = COMMANDS

    if len(sys.argv) < 2 or sys.argv[1] in ('--help', '-h', 'help'):
        print(__doc__.strip())
        print("\n可用命令: sft, selfplay (或 sp), sh")
        sys.exit(0)

    # --sh <路径> [透传参数...] / --sh=<路径> [透传参数...]
    # 必须放在子命令校验之前：否则 "--sh" 会落进"未知命令"分支。
    a1 = sys.argv[1]
    if a1 == '--sh' or a1.startswith('--sh='):
        rest = sys.argv[2:]
        if a1.startswith('--sh='):
            # 等号写法：脚本路径就是 a1 去掉 "--sh=" 后的部分
            if not a1[len('--sh='):]:
                print("[run] --sh= 需要给出脚本路径")
                sys.exit(1)
            run_sh([a1[len('--sh='):]] + rest)
        else:
            run_sh(rest)
        return

    cmd = a1
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
