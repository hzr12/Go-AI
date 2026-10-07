"""探测真机上 torch_npu 有哪些「张量布局」相关入口（只读，不训练）。

为什么需要它
------------
昇腾的 NHWC 布局 API 在版本间变动很大，而本仓库训练环境是
**torch 2.1.0 + torch_npu 2.1.0**（2023-10，对应 CANN 7.0.RC1）。
截至目前已经在真机上踩了三次不同的错：

    1. model.to(memory_format=channels_last)
       → Only contiguous_format or preserve_format is supported.
    2. param.data.contiguous(memory_format=channels_last)
       → NPU contiguous operator only supportted contiguous memory format.
         [ERROR] ERR01007 OPS feature not supported
    3. torch_npu.npu_format_cast(t, torch_npu.Format.NHWC)
       → AttributeError: module 'torch_npu' has no attribute 'Format'

与其继续猜第四个 API，不如把这个环境**问清楚**：把 torch_npu 上与
format / layout / transpose 相关的公开名字全部枚举出来，外加逐个实测
`npu_format_cast` 在**本版本**是否存在、接受什么签名。

用法（在 NPU 机器上）：
    python scripts/probe_npu_layout.py
    python scripts/probe_npu_layout.py --full     # 连 torch_npu 全部公开名一起列
"""

import argparse
import inspect
import sys


def _fmt_line(k, v):
    t = type(v).__name__
    try:
        sig = str(inspect.signature(v))
    except (TypeError, ValueError):
        sig = '(?)'
    return '  %-34s %-16s %s' % (k, t, sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true',
                    help='连 torch_npu 全部公开名一起列出来')
    args = ap.parse_args()

    try:
        import torch
    except Exception as e:
        print('import torch 失败:', repr(e))
        return 1
    print('torch        = %s' % torch.__version__)

    try:
        import torch_npu
    except Exception as e:
        print('import torch_npu 失败:', repr(e))
        print('\n这不是 NPU 环境，脚本无法探测。')
        return 2
    print('torch_npu    = %s' % getattr(torch_npu, '__version__', '?'))

    # ---- 1. 布局相关候选名 ----
    keys = [k for k in dir(torch_npu)
            if any(w in k.lower() for w in
                   ('format', 'layout', 'transpose', 'memory', 'contiguous'))]
    print('\n[1] torch_npu 上与 布局/format/transpose 相关的公开名：')
    if not keys:
        print('  （一个都没有 ⇒ 该版本完全不给用户侧布局控制）')
    for k in sorted(keys):
        print(_fmt_line(k, getattr(torch_npu, k)))

    # ---- 2. Format 枚举到底有没有 ----
    print('\n[2] Format 枚举：')
    fmt = getattr(torch_npu, 'Format', None)
    print('  torch_npu.Format            = %r' % (fmt,))
    if fmt is not None:
        members = [m for m in dir(fmt) if not m.startswith('_')]
        print('  成员: %s' % ', '.join(members))
        for m in members:
            try:
                print('    %-14s = %r' % (m, getattr(fmt, m)))
            except Exception:
                pass
    else:
        print('  ⇒ 不存在。官方文档那套 `Format.NHWC` 在本版本上用不了。')

    # ---- 3. npu_format_cast 实测 ----
    print('\n[3] npu_format_cast 实测（要真 NPU 张量，故先建一个）：')
    fn = getattr(torch_npu, 'npu_format_cast', None)
    print('  torch_npu.npu_format_cast   = %r' % (fn,))
    if fn is None:
        print('  ⇒ 不存在。')
    else:
        try:
            print('  签名: %s' % inspect.signature(fn))
        except (TypeError, ValueError) as e:
            print('  签名: 取不到 (%s)' % e)
        try:
            x = torch.randn(1, 4, 8, 8)
            if hasattr(x, 'npu'):
                x = x.npu()
        except Exception as e:
            print('  建 NPU 张量失败，跳过实测:', repr(e))
            x = None
        if x is not None:
            getter = getattr(torch_npu, 'get_npu_format', None)
            print('  get_npu_format 存在 = %r' % (getter,))
            if getter is not None:
                try:
                    print('  转换前 format = %r' % getter(x))
                except Exception as e:
                    print('  get_npu_format 读原始张量失败:', repr(e))
            for label, arg in (('int 1 (NHWC)', 1),
                               ('int 2 (ND)', 2)):
                if fmt is not None:
                    break
                try:
                    y = fn(x, arg)
                    print('  npu_format_cast(x, %-14s) -> OK' % label)
                    if getter is not None:
                        try:
                            print('      转换后 format = %r' % getter(y))
                        except Exception:
                            pass
                except Exception as e:
                    print('  npu_format_cast(x, %-14s) -> %r' % (label, e))

    # ---- 4. 全量公开名（排障用）----
    if args.full:
        pub = [k for k in dir(torch_npu) if not k.startswith('_')]
        print('\n[4] torch_npu 全部公开名（%d 个）：' % len(pub))
        for i in range(0, len(pub), 4):
            print('  ' + '  '.join('%-22s' % k for k in pub[i:i + 4]))

    print('\n结论怎么用：')
    print('  · 若 [1]/[2]/[3] 全空 ⇒ 该 torch_npu 版本不支持用户侧改布局，')
    print('    `--npu-channels-last` 在这台机器上只能是 no-op，别再指望它提速；')
    print('    训练照常（代码会自动降级），加速要换别的方向。')
    print('  · 若 [3] 有可用入口 ⇒ 把签名贴回来，我把对应分支固化进 _npu_nhwc_formats()。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
