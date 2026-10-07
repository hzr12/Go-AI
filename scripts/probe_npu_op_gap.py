"""在 910C 上一次性列出「本模型用到、但 TorchAir 没有 ge_converter」的 aten 算子。

⚠ 本文件是 **spike（一次性探针）**：产出是**结论**（缺口算子清单 + 改法），
不是要长期维护的代码。结论拿回来后，真实改动写进 `src/networks/katago_v7.py`
并由 `tests/` 里的数值等价测试钉住，届时本文件可删除。

为什么要有这个：`torch.ops.aten.amax.default ge_converter is not implemented!`
是**撞到第一个**才报的 —— 逐个试错每轮十几分钟，`whole` 档可能有好几个缺口，
撞一个补一个要跑好几轮。这个脚本把三件事一次问完：

  ① 本模型前向 + 反向真实用到的 aten 算子集合（aot 给的那两张图，与
     `_npu_backend` 内部交给 compiler 的是同一份）
  ② `ge_converter` 里**明确 `raise NotImplementedError`** 的算子集合（读源码，不用试）
  ③ ①∩② = 必须改的算子；外加 ① 里**连 converter 文件都没有**的算子
  ④ 顺带试编 `amax` 的几种替代写法，一次定下改法

用法（NPU 机器上）::

    python -u scripts/probe_npu_op_gap.py 2>&1 | tee tmp/coding/probe_npu_op_gap.log

只读，不改模型；①②③ 完全不编译真图，只有 ④ 的微测是真编译（每个几行）。
**必须在 NPU 上跑**：②③ 只读 torch_npu 源码，① 在 CPU 上取到的注意力算子
是 `_scaled_dot_product_flash_attention_for_cpu`，与 NPU 上那张图不是一回事。
"""
import importlib.util
import os
import pathlib
import re
import sys

os.environ.setdefault('ASCEND_GLOBAL_LOG_LEVEL', '3')

import torch  # noqa: E402

# 把仓库根放进 sys.path：以 `python scripts/probe_npu_op_gap.py` 运行时
# sys.path[0] 只有 `scripts/`，`from src.networks...` 会 ImportError。
# 与 `probe_npu_graph_memory.py:120-125` 同一套写法。
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# ① 模型前向用到的 aten 算子
# --------------------------------------------------------------------------- #
def collect_model_ops(device):
    """跑一遍前向 + 反向，取到 TorchAir 真正要转 GE 的 **aten** 算子集合。

    关键：不能用普通 dynamo backend —— 它给的是 `torch.cat` / `conv2d` 这类
    builtin/function，名字对不上 ge_converter 的 `aten.*`。要拿 aten 图必须
    把 `aot_autograd(fw_compiler=…, bw_compiler=…)` 当后端：aot 分别把**前向图**
    和**反向图**交给 compiler，正是 `_npu_backend` 内部做的事。

    返回 `(ops, errors, phases)`：`ops` 是 `str(OpOverload)` 集合（形如
    ``aten.amax.default``）；`phases` 是 `{图序号: 'fw'|'bw'}`，方便看出缺口
    在前向还是反向；`errors` 是取图失败的报错（漏掉的算子会造成假阴性）。
    """
    from torch._dynamo.backends.common import aot_autograd

    from src.networks.katago_v7 import build_katago_v7_net

    graphs = []
    errors = []

    def recorder(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch.manual_seed(0)
    model = build_katago_v7_net(use_checkpoint=False)
    model.to(device).train()
    spatial = torch.zeros(1, 22, 19, 19, device=device, requires_grad=False)
    gf = torch.zeros(1, 19, device=device, requires_grad=False)
    # 反向图是**懒编译**的：只跑前向拿不到 bw_compiler 那张图。
    compiled = torch.compile(
        model, backend=aot_autograd(fw_compiler=recorder, bw_compiler=recorder),
        fullgraph=False)
    try:
        out = compiled(spatial, gf)
        flat = _float_tensors(out)
        if not flat:
            raise RuntimeError('前向没有任何浮点输出')
        # 把**每个**输出都加进来，才敢说反向图覆盖全：只 backward 第一个输出，
        # 后面几个头的反向算子就一张都取不到，审计会假阴性。
        loss = torch.zeros((), device=device, dtype=torch.float32)
        for t in flat:
            loss = loss + t.float().sum()
        loss.backward()
    except Exception as e:                # noqa: BLE001
        errors.append(repr(e)[:500])

    ops, phases = set(), {}
    for i, gm in enumerate(graphs):
        phases[i] = 'fw' if i == 0 else 'bw'
        for n in gm.graph.nodes:
            if n.op == 'call_function' and isinstance(
                    n.target, torch._ops.OpOverload):
                ops.add(str(n.target))
    return ops, errors, phases


def _float_tensors(obj):
    if torch.is_tensor(obj):
        return [obj] if obj.is_floating_point() else []
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_float_tensors(v))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out.extend(_float_tensors(v))
        return out
    return []


# --------------------------------------------------------------------------- #
# ② ge_converter 里没实现的算子
# --------------------------------------------------------------------------- #
def find_converter_root():
    """定位 `.../ge_converter` 目录（torchair 与 torch_npu 两种布局都试）。"""
    import glob
    for mod in ('torchair', 'torch_npu'):
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.submodule_search_locations:
            continue
        for base in spec.submodule_search_locations:
            for hit in glob.glob(os.path.join(base, '**', 'ge_converter'),
                                 recursive=True):
                if os.path.isdir(hit) and os.path.isdir(
                        os.path.join(hit, 'aten')):
                    return hit
    return None


# `conveter_aten_amax_default` -> `aten.amax.default`；同时抓 `aten.amax` 写法。
_RE_DEF = re.compile(r'def\s+conveter_(aten_[\w.]+?)\s*\(')
_RE_USE = re.compile(r'(aten\.[\w]+(?:\.[\w]+)?)')


def scan_converters(root):
    """返回 `(unsupported, all_ops)`。

    `unsupported`：源码里出现 `raise NotImplementedError` 的文件所对应的算子
    —— 正是 `amax.py:30` 那种「文件在、函数在、一调就抛」的形态。
    `all_ops`：所有有 converter 文件的算子（用来识别「压根没这个文件」的缺口）。
    """
    unsupported, all_ops = set(), set()
    raise_files = []
    for dirpath, _, files in os.walk(os.path.join(root, 'aten')):
        for fn in files:
            if not fn.endswith('.py') or fn == '__init__.py':
                continue
            path = os.path.join(dirpath, fn)
            try:
                text = open(path, encoding='utf-8', errors='replace').read()
            except Exception:
                continue
            names = set()
            for m in _RE_DEF.finditer(text):
                names.add('aten.' + m.group(1).replace('_', '.'))
            for m in _RE_USE.finditer(text):
                names.add(m.group(1))
            # 文件名本身就是一个算子名（amax.py -> aten.amax）
            names.add('aten.' + fn[:-3])
            all_ops |= names
            if re.search(r'raise\s+NotImplementedError', text):
                unsupported |= names
                raise_files.append(os.path.relpath(path, root))
    return unsupported, all_ops, raise_files


# --------------------------------------------------------------------------- #
# ④ amax 替代写法试编
# --------------------------------------------------------------------------- #
def probe_amax_replacements(device, dtype):
    """试编几种 `x.amax(dim=(2,3))` 的等价写法，返回 `{写法: 结果}`。

    每种都跑「带梯度」和「不带梯度」两遍：前向和反向是**两张不同的图**、
    用的是**两套不同的 converter**，只测前向会漏掉反向缺口。
    """
    import torchair
    torchair_cfg = torchair.CompilerConfig()
    be = torchair.get_npu_backend(compiler_config=torchair_cfg)

    x = torch.randn(2, 8, 4, 4, device=device, dtype=dtype,
                    requires_grad=True)

    def case_flatten_max(x):
        return x.flatten(2).max(dim=2).values

    def case_max_pool(x):
        h, w = x.shape[-2], x.shape[-1]
        return torch.nn.functional.max_pool2d(x, (h, w)).flatten(1)

    def case_amax(x):
        return x.amax(dim=(2, 3))

    def case_sum_max(x):
        # 元素级：max 等于 (x == x.amax).sum … 不行，仍要 amax。占位说明用。
        return x.max(dim=2).values.max(dim=2).values

    out = {}
    for name, fn in (('flatten.max', case_flatten_max),
                     ('max_pool2d', case_max_pool),
                     ('amax(原写法)', case_amax),
                     ('链式 max(dim)', case_sum_max)):
        res = []
        for grad in (False, True):
            try:
                c = torch.compile(fn, backend=be, fullgraph=True)
                y = c(x)
                if grad:
                    y.sum().backward()
                res.append('OK')
            except Exception as e:        # noqa: BLE001
                msg = str(e)
                m = re.search(r'([\w.]+)\s+ge_converter is not implemented',
                              msg)
                res.append('FAIL:%s' % (m.group(1)
                                        if m else msg.splitlines()[0][:90]))
        out[name] = res
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description='列出「本模型用到、TorchAir 没有 ge_converter」的 aten 算子')
    ap.add_argument('--attn', default='prod', choices=('prod', 'math'),
                    help='注意力路径，含义与 probe_npu_graph_memory 一致'
                         '（默认 prod = 复刻 train_sft 在 NPU 上的真值）。'
                         '取 math 则算子集合是手写 _sdpa_math 那张图。')
    args = ap.parse_args(argv)

    print('=' * 74)
    print('① 本模型（%s 注意力路径）前向 + 反向用到的 aten 算子' % args.attn)
    print('=' * 74)
    has_npu = hasattr(torch, 'npu') and torch.npu.is_available()
    device = torch.device('npu' if has_npu else 'cpu')

    # 与 probe_npu_graph_memory 同一套接线：不复刻 train_sft，取到的算子集合
    # 就不是生产口径（见那边 `--attn` 的 help）。
    from src.networks import backbone as _bb
    if args.attn == 'math':
        _bb.set_sdpa_force_math(True)
        _bb.set_npu_fusion_attention(False)
    else:
        _bb.set_sdpa_force_math(False)
        _bb.set_npu_fusion_attention(True)
    print('force_math=%s sfa_env=%s'
          % (bool(_bb._sdpa_force_math), bool(_bb._SFA_ENV_ON)))

    ops, errs, phases = collect_model_ops(device)
    print('取到 %d 个 aten 算子（%d 张图: %s）'
          % (len(ops), len(phases),
             ', '.join('%d=%s' % kv for kv in sorted(phases.items()))))
    for o in sorted(ops):
        print('   ', o)
    if errs:
        print('[warn] 取图时有子图没跑成（可能漏算子）:')
        for e in errs:
            print('   ', e)

    print()
    print('=' * 74)
    print('② ge_converter 里明确 raise NotImplementedError 的文件')
    print('=' * 74)
    root = find_converter_root()
    if root is None:
        print('[fatal] 没找到 ge_converter 目录 —— 手动确认路径后改本脚本')
        return 2
    print('converter 目录:', root)
    unsupported, all_ops, raise_files = scan_converters(root)
    for f in sorted(raise_files):
        print('   ', f)
    print('   （未实现算子数: %d）' % len(unsupported))

    print()
    print('=' * 74)
    print('③ 缺口：本模型用到 ∩ 没实现')
    print('=' * 74)
    gap = sorted(ops & unsupported)
    for o in gap:
        print('   !!', o)
    if not gap:
        print('    （空 —— 说明不是「明确未实现」，而是「压根没这个 converter」'
              '或 CANN 后端报错，见 ④ / 下一轮）')

    print()
    print('    模型用到、且连 converter 文件都没有的（同样会炸）:')
    ghost = sorted(o for o in ops
                   if o not in all_ops
                   and not any(o.startswith(a + '.') or a.startswith(o + '.')
                               for a in all_ops))
    for o in ghost:
        print('   ??', o)
    if not ghost:
        print('     （空）')

    print()
    print('=' * 74)
    print('④ amax 替代写法试编（前向 / 带梯度）')
    print('=' * 74)
    try:
        res = probe_amax_replacements(device, torch.bfloat16)
    except Exception as e:            # noqa: BLE001
        print('[fatal] 取不到 torchair backend:', repr(e)[:300])
        return 2
    for k, v in res.items():
        print('   %-14s fwd=%s  fwd+bwd=%s' % (k, v[0], v[1]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
