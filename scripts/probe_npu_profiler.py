"""探测真机上 torch_npu.profiler 的实际可用读法（只读，不训练）。

为什么需要它
------------
`scripts/train_sft.py` 的 ``GOAI_PROFILE`` 走的是这条路径：

    import torch_npu.profiler as _tnpu_prof
    _prof_ctx = _tnpu_prof.profile(activities=[CPU, NPU])
    _prof_ctx.__enter__()                      # 起
    ...
    _prof_ctx.__exit__(None, None, None)       # 止
    _ka = _prof_ctx.key_averages()             # ← 2026-10-07 真机在这里炸

真机（torch 2.1.0 + torch_npu 2.1.0.post10）实际输出：

    profiler.py: Incorrect schedule: Stop profiler while current state is
    RECORD which may result in incomplete parsed data.
    [profile] 打印内核耗时表失败: 'profile' object has no attribute 'key_averages'

两条信息合起来很怪：**对象的类名确实叫 ``profile``，却没有 ``key_averages``**
—— 而 ``torch.profiler.profile`` 是有的。说明 ``_tnpu_prof.profile`` 在本版本
是它自己的类，读结果的入口与 torch 不同；或者它需要额外一步（``stop()`` /
``step()`` / 拿 ``profiler_result``）才把数据挂到对象上。

与其再猜第五个 API（这个环境已经因为猜 NHWC 猜错三次，见
``probe_npu_layout.py``），照同样的办法把 profiler 的公开面**问清楚**：
枚举类型与方法、列出 ``__exit__`` 之后对象身上真实存在的状态，再**真跑一次
极小的 profile**（三次 64×64 matmul）逐个试读 —— 谁能读出数据，谁就是要的入口。

用法（在 NPU 机器上）—— **`--trace` 现在默认开**，一次跑完就能拿到能否导出
trace、cat 直方图与内核样本字段，不必来回两趟：
    python scripts/probe_npu_profiler.py
    python scripts/probe_npu_profiler.py --full        # 连全部公开名一起列
    python scripts/probe_npu_profiler.py --no-trace    # 只看类型/方法，不落盘
"""

import argparse
import json
import os
import sys


#: 只试**无副作用**的读法。`start` / `stop` / `step` 是状态机操作 —— 逐个乱调
#: 会把刚结束的会话重新拉起来或打乱状态，反而看不出原本能读出什么。
READERS = ('key_averages', 'events', 'profiler_result', 'results', 'result')


def _fmt(k, v):
    try:
        sig = str(__import__('inspect').signature(v))
    except (TypeError, ValueError):
        sig = '(?)'
    return '    %-26s %-18s %s' % (k, type(v).__name__, sig)


def _section(title):
    print('\n' + title)


def _probe_type(tp):
    prof = getattr(tp, 'profile', None)
    print('  torch_npu.profiler.profile   = %r' % (prof,))
    if prof is None:
        return None
    print('  type(prof)                   = %r' % (type(prof),))
    try:
        print('  MRO                          = %s'
              % ' -> '.join(c.__name__ for c in type(prof).__mro__))
    except Exception as e:
        print('  MRO 取不到 (%s)' % e)

    names = [n for n in dir(prof) if not n.startswith('_')]
    print('  公开方法（%d 个）: %s' % (len(names), ', '.join(sorted(names))))
    have = [n for n in READERS if hasattr(prof, n)]
    miss = [n for n in READERS if not hasattr(prof, n)]
    print('  关键读法 有: %s' % (', '.join(have) or '（无）'))
    print('  关键读法 缺: %s' % (', '.join(miss) or '（无）'))
    if 'key_averages' in miss:
        print('  ⇒ 复现了真机那条 AttributeError：本版本的 profile 没有 key_averages。')
    for m in ('start', 'stop', 'step', 'export_chrome_trace'):
        print('  状态/导出 %-16s = %s' % (m, '有' if hasattr(prof, m) else '缺'))
    return prof


def _probe_activities(tp):
    pa = getattr(tp, 'ProfilerActivity', None)
    print('  ProfilerActivity            = %r' % (pa,))
    if pa is None:
        return []
    mem = [m for m in dir(pa) if not m.startswith('_') and m.isupper()]
    print('  成员: %s' % ', '.join(mem))
    return [getattr(pa, m) for m in mem]


def _probe_module_names(tp, full):
    pub = [k for k in dir(tp) if not k.startswith('_')]
    print('  torch_npu.profiler 公开名（%d 个）: %s'
          % (len(pub), ', '.join(sorted(pub))))
    if full:
        print('\n  [full] 逐个看签名：')
        for k in sorted(pub):
            print(_fmt(k, getattr(tp, k)))


def _read_after_exit(p, trace):
    """逐个试读：把每种读法的结果/异常原样打出来，谁成功谁就是入口。"""
    _section('[4] __exit__ 之后逐个试读：')
    try:
        keys = ', '.join(sorted(vars(p))) or '（空）'
        print('  p.__dict__ 键: %s' % keys)
    except Exception as e:
        print('  p.__dict__ 取不到 (%s)' % e)

    ok = []
    for name in READERS:
        if not hasattr(p, name):
            print('  %-22s : 对象上没有这个属性' % name)
            continue
        attr = getattr(p, name)
        if not callable(attr):
            print('  %-22s : 非可调用 = %r' % (name, attr))
            ok.append(name)
            continue
        try:
            out = attr()
        except Exception as e:
            print('  %-22s : 调用失败 %r' % (name, e))
            continue
        ok.append(name)
        print('  %-22s : OK -> %s' % (name, type(out).__name__))

        ka = getattr(out, 'key_averages', None)
        if callable(ka):
            try:
                ev = ka()
                print('      └ key_averages() OK，事件数=%d'
                      % len(list(ev)))
                try:
                    txt = ev.table(row_limit=10)
                    print('      └ table(row_limit=10) OK，前 10 行:')
                    for line in txt.splitlines()[:12]:
                        print('         %s' % line)
                except Exception as e:
                    print('      └ table() 失败 %r' % e)
            except Exception as e:
                print('      └ key_averages() 失败 %r' % e)
        else:
            print('      └ 这个返回值上没有 key_averages()')

    if trace:
        _section('[4b] export_chrome_trace + 结构速览：')
        exp = getattr(p, 'export_chrome_trace', None)
        if exp is None:
            print('  对象上没有 export_chrome_trace')
            return ok
        path = os.environ.get('GOAI_TRACE_OUT', '/tmp/goai_prof_probe.json')
        try:
            exp(path)
            size = os.path.getsize(path)
            print('  export_chrome_trace -> %s (%d bytes)' % (path, size))
        except Exception as e:
            print('  export_chrome_trace 失败 %r' % e)
            return ok
        ok.append('export_chrome_trace')
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print('  读回 JSON 失败 %r' % e)
            return ok
        evs = data.get('traceEvents') if isinstance(data, dict) else data
        if isinstance(data, dict):
            print('  顶层键: %s' % ', '.join(sorted(data)))
        if not isinstance(evs, list):
            print('  traceEvents 不是 list: %r' % type(evs))
            return ok
        print('  事件数 = %d' % len(evs))
        cats = {}
        for e in evs:
            if isinstance(e, dict):
                c = e.get('cat') or '?'
                cats[c] = cats.get(c, 0) + 1
        print('  cat 直方图（前 20）:')
        for c, n in sorted(cats.items(), key=lambda kv: -kv[1])[:20]:
            print('    %-28s %d' % (c, n))
        # 每个 cat 的总耗时 —— 决定去哪个 cat 取内核表
        tot = {}
        for e in evs:
            if isinstance(e, dict) and e.get('dur'):
                c = e.get('cat') or '?'
                tot[c] = tot.get(c, 0) + float(e['dur'])
        print('  cat 总耗时（us，前 10）:')
        for c, v in sorted(tot.items(), key=lambda kv: -kv[1])[:10]:
            print('    %-28s %.0f' % (c, v))
        # 挑最像「设备内核」的 cat，抽 3 条原样看字段
        pick = next((c for c, _ in sorted(tot.items(), key=lambda kv: -kv[1])
                     if any(h in c.lower() for h in ('kernel', 'npu', 'device', 'op'))),
                    None)
        print('  候选内核 cat = %r' % pick)
        if pick:
            shown = 0
            for e in evs:
                if isinstance(e, dict) and (e.get('cat') or '?') == pick:
                    print('    样本: %s' % json.dumps(e, ensure_ascii=False)[:400])
                    shown += 1
                    if shown >= 3:
                        break
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true',
                    help='连 torch_npu.profiler 全部公开名一起列')
    ap.add_argument('--no-trace', dest='trace', action='store_false',
                    default=True,
                    help='跳过 export_chrome_trace（默认开 —— 2.1.0.post10 上它'
                         '是唯一的数据出口，关掉就只剩前面几节）')
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

    try:
        import torch_npu.profiler as tp
    except Exception as e:
        print('import torch_npu.profiler 失败:', repr(e))
        return 3
    print('profiler 模块 = %r' % tp)

    _section('[1] torch_npu.profiler 公开名：')
    _probe_module_names(tp, args.full)

    _section('[2] profile 类型：')
    _probe_type(tp)

    _section('[3] ProfilerActivity 成员：')
    acts = _probe_activities(tp)

    _section('[4] 真跑一次极小 profile（3 次 64×64 matmul）：')
    prof_cls = getattr(tp, 'profile', None)
    if prof_cls is None:
        print('  没有 profile 类，跳过实测。')
        return 4
    use = [a for a in acts if not hasattr(a, 'name')
           or 'CPU' in str(a) or 'NPU' in str(a)]
    if not use:
        use = [getattr(tp.ProfilerActivity, 'CPU', None)]
        use = [a for a in use if a is not None]
    print('  activities = %s' % [str(a) for a in use])
    p = None
    try:
        p = prof_cls(activities=use)
    except Exception as e:
        print('  构造失败 %r' % e)
        print('  ⇒ 退一步只测构造签名，不跑真实 profile。')
        import inspect
        try:
            print('  签名: %s' % inspect.signature(prof_cls))
        except Exception as e2:
            print('  签名: 取不到 (%s)' % e2)
        return 5
    import inspect
    try:
        print('  构造签名: %s' % inspect.signature(prof_cls))
    except Exception:
        pass

    try:
        p.__enter__()
        print('  __enter__() OK')
    except Exception as e:
        print('  __enter__() 失败 %r' % e)
        return 6

    try:
        a = torch.randn(64, 64)
        if hasattr(a, 'npu'):
            a = a.npu()
        for _ in range(5):
            c = a @ a
        if hasattr(torch, 'npu') and hasattr(torch.npu, 'synchronize'):
            torch.npu.synchronize()
        print('  采样计算 OK (device=%s)' % a.device)
    except Exception as e:
        print('  采样计算失败 %r' % e)

    try:
        p.__exit__(None, None, None)
        print('  __exit__() OK')
    except Exception as e:
        print('  __exit__() 失败 %r' % e)
        for m in ('stop', 'step'):
            fn = getattr(p, m, None)
            if fn is None:
                continue
            try:
                fn()
                print('  改用 %s() OK' % m)
                break
            except Exception as e2:
                print('  %s() 也失败 %r' % (m, e2))

    ok = _read_after_exit(p, args.trace)

    _section('[5] 结论：')
    if 'key_averages' in ok:
        print('  ⇒ 直接 p.key_averages() 就能读，train_sft.py 不用改。')
    elif 'export_chrome_trace' in ok:
        print('  ⇒ 本版本唯一的数据出口是 chrome trace。把 [4b] 的 cat 直方图')
        print('    与样本贴回去，我把 train_sft.py 的打印段改成「导出 + 解析 JSON」，')
        print('    直接在日志里出 top-18 表。')
    elif ok:
        print('  ⇒ 本版本读结果要走: %s' % ', '.join(ok))
        print('    把 [4] 的输出贴回去，我把 train_sft.py 的打印段改成这条读法。')
    else:
        print('  ⇒ 一条都读不出来。把 [1]~[4] 全部输出贴回去再判。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
