"""探测「块级 / 整模型 TorchAir 图编译」在这台 64 GB 卡上装不装得下（只读，不训练）。

⚠ 本文件是 **spike（一次性探针）**：它的产出是**结论**，不是要长期维护的代码。
结论拿回来之后，真实改动（若要做块级编译）应该写进 `scripts/train_sft.py`
并由 `tests/test_npu_graph_compile.py` 重新钉住。届时本文件可直接删除。

为什么需要它
------------
`train_sft.py:3653` 与 `:4059` 记着「整模型 torch.compile 把 26.7~31.1GB 顶到
OOM 边缘」。**这条记录是无效证据**：

  · `--compile` / `--npu-graph-compile` 强制 `_gc = 0`（`train_sft.py:4726`），
    且 `backbone.assert_grad_checkpoint_compile_compatible` 直接禁止 GC 与
    compile 共存 —— 所以那次实验里 GC 一定是关的；
  · 那次实验的 batch 是 A-2 的 6000，而 GC 关掉后每样本按**任一**口径都超预算：
    按 `:727` 的公式 12.67 MB ⇒ 6000×12.67 MB ≈ **76 GB**（> 64 GB）；
    按 `:737` 的 277 MB ⇒ 1662 GB。**无论图缓冲多大都必 OOM**。

  换句话说它测的是「GC 关了放不下」，不是「图缓冲放不下」。
  **图 workspace 的真实大小，全仓库零测量。**

而且「每样本多少 MB」这件事，仓库里同时存在**三个互不相容**的数：

  位置                          每样本      它到底是什么
  ---------------------------  ---------   ------------------------------------------
  `:721` GC 开  `[mem]` 行        4.65     6.9÷2 + 1.2（驻留÷2 + 瞬时）
  `:727` GC 关  `[mem]` 行       12.67     **只算注意力**：11×4×361²×2/1e6 + 1.2
  `:737` GC 关  warning         277.00     实测的 fp32 **驻留**字典值（含一切）
  `:704` docstring              139.7      277÷2 + 1.2 —— 想调和上面两个，但
                                            **与 `:727` 实际打出来的 12.67 不符**
  `run.txt:133`                "4.65 → ~278"  把两个不同单位混写    ✗
  `run.txt:134`                "batch 上限 200 出头"  按 277 算的   ✗

12.67 与 277 差 **21.9×**，而这个差距直接决定成败：

    batch=200，GC 关：按 12.67 ⇒ 占 2.5 GB（图 workspace 随便放）
                     按 277  ⇒ 占 55.4 GB（64 GB 只剩 8.6 GB）

两个口径给出的答案**完全相反**。所以「装不装得下」必须实测，不能推算。

本探针一次性回答三件事
----------------------
1. **真实每样本显存**是多少？对照 12.67（`:727` 公式）与 277（`:737` 驻留值）。
2. 各编译粒度的 **workspace 要几 GB**？
   = 同一 batch、**同一 GC 状态**下，编译档减基线档。
3. 各粒度的 **batch 上限**在哪（扫到第一个 OOM 为止）。

档位（阶梯，逐级更激进；`TIER_SPEC` 是唯一权威表）：

  tier          GC    编译范围                   backend     对应开关 / 意图
  ------------  ----  -------------------------  ----------  -------------------------
  gc            开    无（eager）                          现网生产配置（锚点）
  eager         关    无                                   隔离「关 GC」的代价
  linear        关    每个 nn.Linear 一张图       torchair   `--npu-graph-compile 1`
  ind-linear    关    每个 nn.Linear 一张图       inductor   与 linear 同粒度，只换后端
  ind-all       关    整个 model 一张图           inductor   融合上限（GC 关）
  block         关    每个 `model.blocks[i]`      torchair   （提案，未实现）
  whole         关    整个 model 一张图           torchair   `:3653` 那条记录的形态
  gc-ind        开    每个 `model.blocks[i]`      inductor   ★ GC + inductor，块粒度
  gc-ind-all    开    整个 model 一张图           inductor   ★ GC + inductor，最大融合

`gc` 档与 GC 关的档位**不可比**（编译与 GC 互斥是当前策略），它只用来锚定现网基线。

★ 为什么要试 GC + inductor
--------------------------
这两件事**各解决一半问题，且互不冲突**：

  · **GC** → 每样本 4.65 MB 而非 12.67/277 MB ⇒ batch 6000 照跑，显存不破 64 GB；
  · **inductor** → 融掉 96% 的 elementwise 碎核（`Mul` 1131 / `Add` 987 /
    `Select` 986 / `Mul_StridedSlice` 1232 / `InplaceCopy_ViewCopy` 247），
    而 TorchAir Linear-only 只碰 `aclnnMatmul` 343 核 = **3.5%**。

而仓库**主动禁止**这个组合（`train_sft.py:4726` 强制 `_gc = 0` +
`backbone.assert_grad_checkpoint_compile_compatible` 抛 RuntimeError），理由写在
守卫 docstring 里：`torch.utils.checkpoint` 靠 `saved_tensors_hooks`、compile 靠
Dynamo 图捕获，两者边界会让检查点段退化成 graph break 或 eager。

**那个理由是推理，不是实测。** 本探针绕过守卫直接试，并单独报告「守卫会不会拦」——
把「策略上禁止」和「技术上不行」拆成两个问题分别回答。

为什么 inductor 本身也值得一试
-----------------------------
`train_sft.py:4555` 那条「NPU 上 torch.compile(inductor) 不可用」来自提交
`24a3ab1`，代码是**无条件 warn + 直接 `args.compile = False`**，从没真正试过 ——
它是 2023 年的假设，不是实测（`run.txt:34`、`:4514` 只是转述同一句，不算独立证据）。

外部证据指向「**当前版本没有，新版有**」：

  · 昇腾上的 inductor = `triton-ascend` + `torch_npu._inductor`；
  · **triton-ascend 捆绑的 TorchNPU 是 2.7.1.post8**，本机是 **2.1.0.post10**；
  · 官方「Inductor 编译后端」章节出现在 **TorchNPU 26.1.0** 文档，要求
    **Triton Ascend v3.2.2**，用法 `torch.compile(backend="inductor")`。

所以别猜，跑一次拿一手结论 —— 不可用也会把**确切报错**打出来。

用法（在 NPU 机器上）
--------------------
    # 最快：只回答「GC + inductor 到底能不能跑、对不对、多快」
    python scripts/probe_npu_graph_memory.py --try-inductor

    python scripts/probe_npu_graph_memory.py            # 默认含 gc-ind / gc-ind-all
    python scripts/probe_npu_graph_memory.py --tiers gc,eager,gc-ind,gc-ind-all
    python scripts/probe_npu_graph_memory.py --batch-sizes 200,400,600 \
            --stop-at-first-oom


**不读数据集、不建 DataLoader、不落 checkpoint** —— 只建网 + 随机张量 +
几个 fwd/bwd/step，通常几十秒到几分钟（编译档要等编译）。
"""

import argparse
import ast
import math
import pathlib
import re
import sys
import time
import traceback

# 把仓库根放进 sys.path：以 `python scripts/probe_npu_graph_memory.py` 运行时
# sys.path[0] 只有 `scripts/`，`from src.networks...` 会 ImportError。
# 与 train_sft.py 的做法一致。
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --------------------------------------------------------------------------- #
# 内存 API 探测（torch_npu 各版本名字不统一，宁可逐个试也不要抛）
# --------------------------------------------------------------------------- #
def _mem_api():
    """返回一组已解析好的显存函数；取不到的项为 None。"""
    api = {k: None for k in (
        'total', 'alloc', 'peak_alloc', 'reset_peak', 'empty_cache')}
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception:
        return api

    def pick(*cands):
        for obj, name in cands:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn
        return None

    dev = torch.npu if hasattr(torch, 'npu') else None
    if dev is None:
        return api
    cuda = torch.cuda
    api['alloc'] = pick((dev, 'memory_allocated'), (cuda, 'memory_allocated'))
    api['peak_alloc'] = pick((dev, 'max_memory_allocated'),
                             (cuda, 'max_memory_allocated'))
    api['reset_peak'] = pick((dev, 'reset_peak_memory_stats'),
                             (dev, 'reset_max_memory_allocated'),
                             (cuda, 'reset_peak_memory_stats'))
    api['empty_cache'] = pick((dev, 'empty_cache'), (cuda, 'empty_cache'))
    try:
        api['total'] = int(torch.npu.get_device_properties(
            torch.npu.current_device()).total_memory)
    except Exception:
        api['total'] = None
    return api


def _gb(n):
    return 0.0 if not n else n / 1e9


def _mb(n):
    return 0.0 if not n else n / 1e6


def _fmt_losses(losses, n=3):
    """把逐 step loss 压成一行短文本；NaN/Inf 直接写出来，不参与四舍五入。"""
    if not losses:
        return '-'
    parts = []
    for x in losses[:n]:
        if not math.isfinite(x):
            parts.append('nan' if math.isnan(x) else 'inf')
        else:
            parts.append('%.4f' % x)
    if len(losses) > n:
        parts.append('…')
    return ' '.join(parts)


def _max_rel(a, b):
    """逐位相对差的最大值；任一为空返回 None，遇到非有限值直接返回 inf。"""
    if not a or not b:
        return None
    n = min(len(a), len(b))
    diffs = []
    for x, y in zip(a[:n], b[:n]):
        if not (math.isfinite(x) and math.isfinite(y)):
            return float('inf')
        diffs.append(abs(x - y) / max(abs(y), 1e-12))
    return max(diffs) if diffs else None


def _fmt_rel(v):
    if v is None:
        return 'n/a'
    return '%.3e' % v if math.isfinite(v) else 'inf'


# --------------------------------------------------------------------------- #
# 从 train_sft.py 抽出「现网那份」代码，保证口径一致
# --------------------------------------------------------------------------- #
def _train_sft_path():
    here = pathlib.Path(__file__).resolve()
    return here.parent / 'train_sft.py'


def _extract_funcs(names):
    """用 AST 从 train_sft.py 顶层抽出指定函数定义，`exec` 成真实函数对象。

    **不 import train_sft**：那个模块在模块级就跑 `args = ap.parse_args()`
    （`:4119`），import 会直接吃掉本脚本的命令行并 SystemExit。
    """
    src = _train_sft_path().read_text(encoding='utf-8')
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(src, node)
    missing = [n for n in names if n not in found]
    if missing:
        raise RuntimeError(
            'train_sft.py 里找不到这些函数（改名了？）: %s' % ', '.join(missing))
    ns = {'__name__': 'probe_extracted'}
    import torch
    ns['torch'] = torch
    for name in names:
        exec(compile(found[name], str(_train_sft_path()), 'exec'), ns)
    return ns


# --------------------------------------------------------------------------- #
# 档位定义
# --------------------------------------------------------------------------- #
# tier -> (grad_ckpt, compile_scope, backend_kind)
#   compile_scope : 'none' | 'linear' | 'block' | 'whole'
#   backend_kind  : None | 'torchair' | 'inductor'
TIER_SPEC = {
    'gc':         (True,  'none',   None),
    'eager':      (False, 'none',   None),
    'linear':     (False, 'linear', 'torchair'),
    'ind-linear': (False, 'linear', 'inductor'),
    'ind-all':    (False, 'whole',  'inductor'),
    'block':      (False, 'block',  'torchair'),
    'whole':      (False, 'whole',  'torchair'),
    'gc-ind':     (True,  'block',  'inductor'),
    'gc-ind-all': (True,  'whole',  'inductor'),
    # GC + **TorchAir** —— 910C 上唯一真正可用的组合：inductor 缺 triton-ascend
    # （实测 `ModuleNotFoundError: No module named 'triton'`），而
    # `torchair.get_npu_backend()` 实测能编译能反向。GC 又是必须的：实测每样本
    # GC 开 8.3 MB / 关 63.7 MB，关掉 GC 就装不下 A-2 的 batch=6000。
    # 之前没有这两档 ⇒ 只测了不可用的那条路。
    'gc-torchair':     (True,  'block', 'torchair'),
    'gc-torchair-all': (True,  'whole', 'torchair'),
}
TIERS = tuple(TIER_SPEC)

#: 本次要回答的问题（GC + inductor）；`--try-inductor` 的被测档位。
INDUCTOR_TIERS = ('gc-ind', 'gc-ind-all')
#: `--try-inductor` 实际跑的档位：**必须带上 `gc` 基线**，否则第 ② 项的
#: 梯度/loss 逐位对比没有对照组，跑通了也判不了对错。
INDUCTOR_SMOKE_TIERS = ('gc',) + INDUCTOR_TIERS

#: 同上，但走 TorchAir 后端 —— 910C 上该跑的就是这组。
TORCHAIR_TIERS = ('gc-torchair', 'gc-torchair-all')
TORCHAIR_SMOKE_TIERS = ('gc',) + TORCHAIR_TIERS


def _apply_tier(tier, model, backends, ns):
    """按档位给 model 上编译；返回 ``(编译过的子模块个数, 实际要用的 model)``。

    **必须接住返回的 model**：`torch.compile(model)` 不改原对象，而是返回一个新的
    `OptimizedModule` 包装。原实现丢掉了这个返回值 —— 于是「整模型」档实际上
    编译了个没人用的包装，跑的还是 eager（本地用 stub backend 抓到的）。
    `linear` / `block` 两档是就地改子模块，model 本体不变，返回原引用即可。
    """
    import torch
    _, scope, kind = TIER_SPEC[tier]
    if scope == 'none':
        return 0, model
    backend = backends.get(kind) if kind else None
    if backend is None:
        raise RuntimeError(
            '档位 %s 需要 %s 后端，但没取到' % (tier, kind))
    if scope == 'linear':
        # 现网那份（train_sft._compile_linear_submodules），原样抽取复用。
        ns['_compile_linear_submodules'](model, backend)
        return sum(1 for _, m in model.named_modules()
                   if hasattr(m, '_orig_mod')), model
    if scope == 'block':
        # 块粒度：每个 nbt2 外块一张图（含其 `.inner` 的 GAU/nbt 子块）。
        n = 0
        for i, blk in enumerate(model.blocks):
            model.blocks[i] = torch.compile(blk, backend=backend,
                                            dynamic=False)
            n += 1
        return n, model
    if scope == 'whole':
        # `:3653` 说的形态。整模型被包成 OptimizedModule ⇒ 顶层出现 `_orig_mod`。
        return 1, torch.compile(model, backend=backend, dynamic=False)
    raise ValueError('未知 compile_scope: %r' % scope)


def _guard_verdict(model):
    """绕过守卫做实验，但**如实报告**落地时守卫会不会拦。

    返回 `(would_block, message)`。直接复用守卫**自己的判据**
    `backbone.compiled_module_paths`（守卫的实现就是 `bad = compiled_module_paths(m);
    if bad: raise`），这样即使运行时已经把守卫换成了 no-op，判定仍然成立。
    守卫是策略（「检测到 `_orig_mod` 就抛」），它不区分 backend —— 所以
    GC + inductor 一定也会被拦；把这件事显式打出来，是为了避免「probe 跑通了
    ⇒ 以为能直接上线」的误判。
    """
    try:
        from src.networks.backbone import compiled_module_paths
    except Exception as e:
        return None, '守卫导入失败: %r' % (e,)
    bad = compiled_module_paths(model)
    if not bad:
        return False, '守卫放行'
    return True, ('梯度检查点与 torch.compile 互斥：以下子模块已被 torch.compile '
                  '包成 OptimizedModule（带 _orig_mod）：%s'
                  % ', '.join(bad[:8]))


def _bypass_inforward_guard(tiers):
    """GC+编译同开时，把 `backbone` 里那道**前向中途**的守卫换成 no-op。

    关键事实：`assert_grad_checkpoint_compile_compatible` 不只在启动期检查 ——
    `backbone.run_grad_segment` 在 `active and guard_root` 时**每次前向都调它**
    （`src/networks/backbone.py:421-423`）。所以不做这个实验是跑不起来的：
    第一次 fwd 就抛，连数据都吃不到。

    只在确实要跑 GC+编译档位时才动手；动了手要**显式打出来**。
    返回 True 表示已改动模块级行为。
    """
    if not any(TIER_SPEC[t][0] and TIER_SPEC[t][1] != 'none' for t in tiers):
        return False
    try:
        import src.networks.backbone as bb
    except Exception as e:
        print('[guard] 无法 import backbone 放行守卫: %r' % (e,))
        return False
    bb.assert_grad_checkpoint_compile_compatible = _noop_guard
    return True


def _noop_guard(module, where=''):
    return None


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def _build(device, dtype, use_checkpoint):
    from src.networks.katago_v7 import build_katago_v7_net
    model = build_katago_v7_net(use_checkpoint=bool(use_checkpoint))
    model.set_grad_checkpointing(bool(use_checkpoint))
    return model.to(device=device, dtype=dtype)


def _loss_of(out):
    import torch
    terms = [v.float().pow(2).mean()
             for v in out.values()
             if torch.is_tensor(v) and v.is_floating_point()]
    return sum(terms) if terms else None


def _grad_norm(model):
    """全参数梯度的 L2 范数（fp64 在主机累加，避开 NPU 归约的不确定顺序）。

    与 `train_sft._assert_init_weights_identical` 同样的理由：NPU 上的分块归约
    不保证跨运行顺序一致，会把「梯度真的不一样」和「归约顺序抖动」混在一起。
    这里只要一个可跨档比较的标量，所以宁可慢一点也要稳定。
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.float().pow(2).sum().to('cpu').double())
    return total ** 0.5


# --------------------------------------------------------------------------- #
# 单次试验
# --------------------------------------------------------------------------- #
def _sps(r):
    """samples/s —— **本项目对外的吞吐口径**。

    不能用「step 耗时」或「多少步跑完」来比档位：关掉 GC 会让 batch 上限从
    ~929 掉不下去、但也可能迫使你换 batch，而 step 数本身随 batch 变。
    单位时间处理的样本数与 batch 无关，才是 spd 的定义。
    """
    try:
        return r['batch'] * r['steps'] / r['elapsed']
    except Exception:  # noqa: BLE001
        return None


def print_throughput(rows):
    """③ 吞吐段 —— **不能**依赖「编译档是否跑通」。

    曾经整段挂在 `if ok_rows:` 里：编译档全挂的那一轮（目前的常态）就**整个
    打不出 samples/s**，而「GC 关 vs 开 谁快」恰恰是编译挂了也必须回答的问题
    （spd 的口径就是 samples/s）。所以这里只吃 `rows`，谁跑出了 elapsed 谁上。
    """
    sp_rows = [x for x in rows
               if x.get('elapsed') and not x.get('oom')
               and not x.get('err') and x.get('losses')]
    if not sp_rows:
        return
    print('③ **快吗 / 省吗** —— 吞吐按 samples/s（与 batch 无关的口径）：')
    # 基线档（gc / eager）也要列出来，否则「× vs gc」永远算不出来 ——
    # 它们不在 `ok_rows`（那只是被测的编译档），但在 `rows` 里。
    # 同 batch 才能算比值：samples/s 自身也随 batch 变，跨 batch 直接比
    # 会把「batch 效应」算进「档位效应」。
    sps_by = {}
    for x in sp_rows:
        sps_by.setdefault(x['batch'], {})[x['tier']] = _sps(x)
    for r in sorted(sp_rows, key=lambda x: (x['tier'], x['batch'])):
        sps = _sps(r)
        base = sps_by.get(r['batch'], {}).get('gc')
        rel = ('  ×%s vs gc' % ('%.2f' % (sps / base))
               if (base and sps and r['tier'] != 'gc') else '')
        print('     %-10s bs=%-5d  %7.2fs  %8.0f samples/s%s  '
              'loss=%s   |grad|=%s'
              % (r['tier'], r['batch'], r['elapsed'], sps or 0.0, rel,
                 _fmt_losses(r['losses']),
                 _fmt_losses(r.get('grad_norms') or [], n=1)))
    best = {}
    for r in sp_rows:
        sps = _sps(r)
        if sps and (r['tier'] not in best or sps > best[r['tier']][0]):
            best[r['tier']] = (sps, r['batch'])
    if len(best) > 1:
        print('     每档最优吞吐：' + '   '.join(
            '%s=%.0f (bs=%d)' % (t, v[0], v[1])
            for t, v in sorted(best.items())))
        print('     ⇒ 各档在**各自能装下的最大 batch**下比 —— samples/s '
              '本来就不该用同 batch 绑死；跨 batch 直接比这一列即可。')


def workspace_pairs(rows, tiers, batches):
    """第 2 项的数据：每个编译档相对**同 GC 状态**基线的峰值差。

    返回 ``[(tier, batch, delta_bytes, base_tier, n_compiled), ...]``。

    **每个 batch 都要报**：原实现在内层循环里打完一个就 `break`，而
    `sorted(batches)` 是升序 ⇒ 永远只报最小那个 batch，bs=800 被整条吞掉
    （2026-10-07 实测：只打出 `linear bs=400 +0.02 GB`，bs=800 的 +0.00 不见了）。
    """
    out = []
    for tier in tiers:
        if TIER_SPEC[tier][2] is None:
            continue
        base_tier = 'gc' if TIER_SPEC[tier][0] else 'eager'
        for bs in sorted(batches):
            a = next((x for x in rows if x['tier'] == tier
                      and x['batch'] == bs and not x.get('oom')
                      and not x.get('err') and x.get('peak') is not None), None)
            e = next((x for x in rows if x['tier'] == base_tier
                      and x['batch'] == bs and not x.get('oom')
                      and not x.get('err') and x.get('peak') is not None), None)
            if a and e:
                out.append((tier, bs, a['peak'] - e['peak'], base_tier,
                            a.get('n_compiled')))
    return out


def batch_ceiling(tier, rows):
    """第 3 项的数据：`(实测成功过的最大 batch, OOM 的 batch, 失败的 batch)`。

    返回 **max 不是 min** —— 这是「batch 上限」。原实现取 `min(ok)`：测了
    800 和 400 都过，却打印「≥ 400」，把实测到的上限**低估一半**
    （2026-10-07 实测 gc/eager/linear 三档明明 800 都过）。
    """
    ok = [r['batch'] for r in rows if r['tier'] == tier
          and not r.get('oom') and not r.get('err')]
    oom = [r['batch'] for r in rows if r['tier'] == tier and r.get('oom')]
    fail = [r['batch'] for r in rows if r['tier'] == tier
            and r.get('err') and not r.get('oom')]
    return (max(ok) if ok else None), oom, fail


def _is_oom(exc):
    """是不是 OOM。

    ⚠ 原实现是 `'%s: %s' % (type(exc).__name__, exc).lower()` —— 属性引用的优先级
    高于 `%`，于是 `.lower()` 打在了**那个元组**上，任何异常都会先抛
    `AttributeError: 'tuple' object has no attribute 'lower'`。而它又是在
    `except` 块里被调的 ⇒ 异常处理自己炸了 ⇒ 整个探针崩掉、一个 FAIL 都没记下。
    括号必须把格式化结果整个包住。
    """
    try:
        s = ('%s: %s' % (type(exc).__name__, exc)).lower()
    except Exception:
        s = type(exc).__name__.lower()
    return ('out of memory' in s or 'oom' in s
            or 'alloc failed' in s or 'memory not enough' in s)


def _cause_chain(e, limit=6):
    """把异常的「外层 → 内层」链拉平：`inner_exception` → `__cause__` → `__context__`。

    为什么必须自己走：`torch._dynamo.exc.BackendCompilerFailed` **不设 `__cause__`**
    （实测 `__cause__ is None`），它把内层异常存在 `self.inner_exception` 上。
    只走 `__cause__` 拿不到；只走 `__context__` 也拿不到（Dynamo 自己构造，没有
    「处理异常时抛出」的语义）。`inner_exception` 优先。
    """
    seen, out, cur = set(), [], e
    while cur is not None and len(out) < limit and id(cur) not in seen:
        seen.add(id(cur))
        out.append(cur)
        nxt = getattr(cur, 'inner_exception', None)
        if not isinstance(nxt, BaseException):
            nxt = getattr(cur, '__cause__', None)
        if not isinstance(nxt, BaseException):
            nxt = getattr(cur, '__context__', None)
        cur = nxt if isinstance(nxt, BaseException) else None
    return out


def _backend_tag(e):
    """从 `BackendCompilerFailed` 的消息里读出**后端名**，归一成 `inductor` / `torchair`。

    消息形如 ``backend=<backend_name!r> raised:``。inductor 的名字就是
    ``'inductor'``；TorchAir 的名字是整段 ``functools.partial(<function _npu_backend
    …, compiler_config=…, decompositions={…})`` 的 repr —— 几百字节，不能原样进表格，
    所以按 `_npu_backend` / `compiler_config` 归类成 `torchair`。
    """
    try:
        raw = str(e)
    except Exception:
        return None
    m = re.search(r"backend=(['\"])(.*?)\1\s*raised", raw, re.S)
    if not m:
        return None
    name = m.group(2)
    if ('_npu_backend' in name or 'torchair' in name
            or 'compiler_config' in name):
        return 'torchair'
    return name if len(name) <= 32 else None


def _project_frames(e, limit=40):
    """异常链里**落在本仓库**的帧（`文件:行 in 函数`），走栈 + 走消息两遍。

    为什么要单独捞：FakeTensor / CANN 这类报错的栈极长，`--full-trace` 也难免
    截断，而**最先被删掉的中间段恰恰是模型自己的帧**。模型帧丢了就只能猜
    「哪一行造出了那个真张量」，而 910C 上一轮往返十几分钟。
    消息里也扫一遍：Dynamo 常把用户源码位置写进 message 而不放进 `__traceback__`。
    """
    root = str(_REPO_ROOT)
    seen, hits = set(), []

    def add(rel, line, fn):
        key = (rel, line)
        if key not in seen:
            seen.add(key)
            hits.append('%s:%s in %s' % (rel, line, fn))

    for cur in _cause_chain(e):
        tb = cur.__traceback__
        while tb is not None:
            path = tb.tb_frame.f_code.co_filename
            if path.startswith(root):
                add(path[len(root):].lstrip('/\\'), tb.tb_lineno,
                    tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
    try:
        text = ''.join(traceback.format_exception(
            type(e), e, e.__traceback__))
    except Exception:  # noqa: BLE001
        text = str(e)
    for m in re.finditer(r'([\w./\\-]+\.py):(\d+)', text):
        rel, line = m.group(1), m.group(2)
        if re.search(r'(^|/|\\)(src|scripts)(/|\\)', rel) and (
                rel, line) not in seen:
            add(rel, line, '?')
    return hits[:limit]


def _full_trace(e, limit_lines=0):
    """整条异常链的完整栈 —— 失败诊断专用，不进表格。

    表格那行只能放一句话；`Please convert all Tensors to FakeTensors…` 这种报错
    的**真正线索在栈里**（哪个算子、哪一行代码造出了那个真张量）。没有栈就只能靠
    猜，而 910C 上一轮往返要十几分钟。

    `limit_lines=0`（默认）= **不截断**。原先默认 90 行（head 30 + tail 60），
    而被删掉的正是**中间** —— innermost 那条栈里 `aten.clone` 从哪来就在中间。
    日志走 tee 落文件，几百行不构成问题。
    """
    out = []
    for cur in _cause_chain(e):
        try:
            out.append(''.join(traceback.format_exception(
                type(cur), cur, cur.__traceback__)).rstrip())
        except Exception:  # noqa: BLE001
            out.append('%s: %r' % (type(cur).__name__, cur))
    lines = '\n'.join(out).splitlines()
    if limit_lines and len(lines) > limit_lines:
        head = lines[:limit_lines // 3]
        tail = lines[-(limit_lines - len(head)):]
        lines = head + ['...（中间省略 %d 行）...'
                        % (len(lines) - len(head) - len(tail))] + tail
    return '\n'.join(lines)


def _err_line(e, limit=1200):
    """**最深一层**的原因打头，外层只留类型（可带后端标签）；多行消息**首尾都留**。

    两条历史教训，都来自 910C 实测：

    1. 不能按顺序拼 —— `BackendCompilerFailed` 的首行是
       ``backend=<后端 repr> raised:``，TorchAir 后端那截 repr 就 300+ 字节，
       合理 limit 下真原因恒被截掉。真原因在链尾。
    2. 也不能只留首行 —— CANN 的图编译失败是 ``E19999: Inner Error!`` +
       一串 ``[FUNC:…][FILE:…]`` + ``TraceBack`` 帧，**根因在最后一行**，
       首行 `E19999: Inner Error!` 本身零信息量。所以**先整条拼起来**，
       只有超过 limit 才退回「首行 + 末行」。limit=1200：2026-10-07 实测
       FakeTensor 那条报错本身就有 400+ 字节，320 会把后半截砍掉。
       更长的线索（栈）走 `--full-trace`，不进这一行。
    """
    chain = _cause_chain(e)
    inner = chain[-1]
    try:
        lines = [l.strip() for l in str(inner).splitlines() if l.strip()]
    except Exception:
        lines = []
    if not lines:
        try:
            body = repr(inner)
        except Exception:
            body = '<unprintable exception>'
        lines = [body]
    body = ' / '.join(lines)
    if len(body) > limit:
        # 超长才退回「首 + 末」：CANN 的 `E19999` 根因在最后一行，
        # 首行 `E19999: Inner Error!` 本身零信息量，只留首行等于什么都没说。
        first, last = lines[0], lines[-1]
        room = limit - len(first) - 5
        body = ('%s ... %s' % (first, last[:room])
                if room > 40 else last[:limit])
    if inner is e:
        return body[:limit]
    head = type(e).__name__
    tag = _backend_tag(e)
    if tag:
        head = '%s[%s]' % (head, tag)
    return ('%s: %s' % (head, body))[:limit]


def _trace_tail(e, limit=240):
    """`↳` 那行：**类型链**；单层异常才带上消息。

    链长 >1 时外层消息几乎总是一大段 backend repr（`BackendCompilerFailed`），
    与 `_err_line` 里的原因重复且更长，只会把输出挤爆 —— 这种情况只留类型名。
    """
    chain = _cause_chain(e)
    noise = ('TORCHDYNAMO_VERBOSE', 'TORCH_LOGS', 'developer context',
             'During handling', 'The above exception')

    def _clean(txt):
        return [l.strip() for l in txt.splitlines()
                if l.strip() and not any(n in l for n in noise)]

    if len(chain) == 1:
        try:
            txt = ''.join(traceback.format_exception_only(type(e), e))
        except Exception:
            return ''
        ls = _clean(txt)
        if len(ls) > 2:
            # 同 `_err_line`：CANN 的根因在末行，首行往往只是 `E19999: Inner Error!`。
            room = limit - len(ls[0]) - 5
            return ('%s ... %s' % (ls[0], ls[-1][:room]) if room > 40
                    else ' '.join(ls))[:limit]
        return ' '.join(ls)[:limit]

    names = list(dict.fromkeys(type(c).__name__ for c in chain))
    tail = ': '.join(names) if names else type(e).__name__
    inner = chain[-1]
    try:
        body = ' '.join(_clean(str(inner)))
    except Exception:
        body = ''
    if body and inner is not e:
        tail = '%s -> %s' % (tail, body)
    return tail[:limit]


def _fail_stage(err, tail, e=None):
    """失败发生在哪一层 —— 这决定了还要不要拿到目标设备上再试一次。

    'dynamo'   : 追踪阶段就挂了。**与设备无关** ⇒ 换设备也一样。
    'inductor' : **已经过了追踪、是 inductor 后端抛的** ⇒ 与后端/设备有关，
                 本机结论不能外推，必须在目标设备上重测。
    'torchair' : 同上，但后端是 TorchAir。单独一档是因为两者的补救办法不同：
                 inductor 缺件要装 triton-ascend，TorchAir 要查 CompilerConfig。
                 合并成一个 'inductor' 会让 TorchAir 的失败被误判成「装 triton」。
    'other'    : 其它（OOM、模型构造、数据形状等）。

    `BackendCompilerFailed` 必须**优先于** `torch._dynamo` 关键字判定：它的类型名
    带 `torch._dynamo` 前缀，但按定义意味着 Dynamo 已经追踪成功、是 **backend**
    抛的（torch 源码 `BackendCompilerFailed(backend_fn, inner_exception, …)`）。
    旧实现先匹配 `torch._dynamo` ⇒ 把 backend 阶段的失败误判成 dynamo 阶段 ⇒
    推出「与设备无关、NPU 也会一样挂」，而当时本机就在 NPU 上，自相矛盾。
    """
    blob = ('%s %s' % (err or '', tail or '')).lower()
    if e is not None:
        # 后端名要从**未截断**的 `str(e)` 里读 —— tail 被 limit 截过，
        # TorchAir 的 functools.partial repr 截断后可能已经看不到 `_npu_backend`。
        try:
            blob = '%s %s' % (blob, str(e).lower())
        except Exception:
            pass
        tag = _backend_tag(e)
        if tag in ('inductor', 'torchair'):
            return tag
    if ('backendcompilerfailed' in blob
            or ("backend='" in blob and ' raised' in blob)
            or ('backend="' in blob and ' raised' in blob)):
        if ('npu_backend' in blob or 'torchair' in blob
                or 'compiler_config' in blob):
            return 'torchair'
        return 'inductor'
    # CANN 图编译失败：TorchAir 把图交给 CANN 的 graph manager 编译，报
    # `E19999: Inner Error!` + `[Call][PreRun] Failed ... [FUNC: CompileGraph]
    # [FILE: graph_manager.cc]`。这同样是**后端阶段**（追踪已过），归 'other'
    # 会让结论段以为「报错不在这两层」，从而漏掉「必须在目标设备上重测」这句。
    if any(k in blob for k in ('e19999', 'e19998', 'graph_manager.cc',
                               'compilegraph', 'call][prerun')):
        return 'torchair'
    if any(k in blob for k in ('torch._inductor', 'inductorerror',
                               'triton', 'codegen', 'compiler:')):
        return 'inductor'
    if any(k in blob for k in ('torch._dynamo', 'dynamo', 'notimplementederror',
                               'checkpoint not implemented', 'unimplemented')):
        return 'dynamo'
    return 'other'


#: 已在本机用 3×4 矩阵实测过的 Dynamo 限制（`tmp/coding/repro_context_fn.py`）。
#: torch 源码：`torch/_dynamo/variables/higher_order_ops.py:3849-3861`
#:
#:     if "context_fn" in kwargs and kwargs["context_fn"] is not noop_context_fn:
#:         ctx = kwargs.pop("context_fn")
#:         if isinstance(ctx, UserFunctionVariable): ...
#:         elif isinstance(ctx, FunctoolsPartialVariable): ...
#:         else: raise NotImplementedError(
#:             f"checkpoint not implemented for {type(ctx)} context_fn")
#:
#: **`type(ctx)` 是 `context_fn` 这个实参的类型，行尾 `" context_fn"` 是消息里的
#: 固定字面量** —— 所以报错指的是 `context_fn`，**不是**被 checkpoint 的函数。
#: `NestedUserFunctionVariable` 的 MRO 是 `BaseUserFunctionVariable → VariableTracker`，
#: 不是 `UserFunctionVariable` 的子类 ⇒ isinstance 必然落空。
_CTXFN_KEYWORD = 'checkpoint not implemented'


def ctxfn_hint(err, tail):
    """命中已知的 `context_fn` 限制时，返回诊断；否则 None。"""
    blob = ('%s %s' % (err or '', tail or '')).lower()
    if _CTXFN_KEYWORD not in blob:
        return None
    return [
        '  ⇒ **这条报错指的是 `context_fn`，不是被 checkpoint 的函数。**',
        '    报错源：torch/_dynamo/variables/higher_order_ops.py:3849-3861，',
        '    `type(ctx)` 是 **context_fn 实参**的 Dynamo 变量类型；行尾的',
        '    `" context_fn"` 是消息里的固定字面量，不是变量名。',
        '    （NestedUserFunctionVariable 不是 UserFunctionVariable 的子类 ⇒ isinstance 落空）',
        '',
        '  已在本机用 3×4 矩阵实测（tmp/coding/repro_context_fn.py，backend=eager，',
        '  只测 Dynamo 追踪 ⇒ 与设备无关，NPU 上会原样复现）：',
        '      被 ckpt 的函数      context_fn      结果',
        '      嵌套闭包            嵌套            FAIL   ← 现状',
        '      纯模块级函数        嵌套            FAIL   ← 改写 checkpoint 治不了',
        '      自 bound method     嵌套            FAIL',
        '      嵌套闭包            模块级 def      PASS',
        '      嵌套闭包            functools.partial PASS',
        '      嵌套闭包            不传            PASS',
        '',
        '  ⇒ **修法是把 `context_fn` 提到模块级（或改用 functools.partial 传闭包里的',
        '    ac/guard），把 checkpoint 目标改写成纯函数是无效的** —— 上表第 2 行已反证。',
        '    落地点：src/networks/backbone.py:505 的 `def context_fn()`（嵌套）。',
        '    注意 `tests/test_katago_v7_grad_checkpointing.py:279` 断言 `_checkpointed`',
        '    源码里含 `context_fn` —— 修法必须保留该 kwarg，否则那条测试会红。',
    ]


def trial(tier, batch, *, device, dtype, use_checkpoint, amp_dtype, steps,
          backends, ns, api, board=19, seed=1234):
    """建一个全新模型 + 优化器，跑 `steps` 轮 fwd/bwd/step，返回峰值统计。

    **每次都重建**：跨试验复用模型会让分配器保留池串味，峰值不再可比。

    **固定随机种子**（两次：建网前 + 生成数据前）：这样不同档位拿到**完全相同的
    初始权重和完全相同的输入**，跨档的 loss 才能逐位对比。只 seed 一次是不够的
    —— 中间的 `torch.compile` 可能消耗 RNG，会让数据错位，对比就失效了。

    额外回答两件显存之外的事：
      · `guard`      —— 落地时 `assert_grad_checkpoint_compile_compatible` 会不会拦；
      · `losses` / `grad_norms` —— 逐 step 的 loss 与全参数梯度 L2 范数。

    **梯度范数是这里最重要的一个数**：检查点重算 + Dynamo 图断开的经典症状是
    **静默算错梯度** —— loss 看着完全正常，梯度却已经不对，模型慢慢学偏。
    只看 loss 会漏掉。种子固定（见上）⇒ 不同档位拿到相同权重和相同输入，
    梯度范数才可跨档逐位对比。
    """
    import torch
    import torch.optim as optim

    if api['empty_cache']:
        api['empty_cache']()
    if api['reset_peak']:
        try:
            api['reset_peak']()
        except TypeError:
            api['reset_peak']()

    torch.manual_seed(seed)
    model = _build(device, dtype, use_checkpoint)
    n_compiled, model = _apply_tier(tier, model, backends, ns)
    guard_block, guard_msg = _guard_verdict(model)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    base = api['alloc']() if api['alloc'] else None

    torch.manual_seed(seed)          # 重新播种，抵消编译阶段可能消耗的 RNG
    spatial = torch.randn(batch, 22, board, board, device=device, dtype=dtype)
    gf = torch.randn(batch, 19, device=device, dtype=dtype)

    losses = []
    grad_norms = []
    t0 = time.perf_counter()
    for _ in range(max(1, int(steps))):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            out = model(spatial, gf)
        loss = _loss_of(out)
        if loss is None:
            raise RuntimeError('前向没有任何浮点输出，无法反传')
        # `float()` 会同步一次，把 D2H 塞进计时里 —— 故意的：每个档位都付同样的
        # 代价，横向比较才成立（本模型太小，这点开销可忽略）。
        losses.append(float(loss.detach()))
        loss.backward()
        grad_norms.append(_grad_norm(model))
        opt.step()
    if device.type == 'npu':
        torch.npu.synchronize()
    elapsed = time.perf_counter() - t0

    peak = api['peak_alloc']() if api['peak_alloc'] else None
    now = api['alloc']() if api['alloc'] else None
    total = api['total']

    n_params = sum(p.numel() for p in model.parameters())
    return {
        'tier': tier, 'batch': batch, 'oom': False,
        'n_compiled': n_compiled, 'n_params': n_params,
        'base': base, 'peak': peak, 'now': now, 'total': total,
        'elapsed': elapsed, 'steps': max(1, int(steps)),
        'guard': guard_block, 'guard_msg': guard_msg,
        'losses': losses, 'grad_norms': grad_norms,
        # NaN 和 Inf 都算坏：inf 说明溢出，nan 说明除零/inf-inf，
        # 两者都是「能跑但学错」—— 融合路径真机出过前向 NaN，必须单独判。
        'bad_loss': any(not math.isfinite(x) for x in losses + grad_norms),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiers', default='gc,eager,gc-torchair,gc-torchair-all',
                    help='逗号分隔的档位（可选值: %s）。默认走 TorchAir —— '
                         '910C 实测 inductor 缺 triton-ascend 跑不了，'
                         'TorchAir 能编译能反向。' % ','.join(TIERS))
    ap.add_argument('--batch-sizes', default='100,200,300,400,600,800',
                    help='逗号分隔的 batch 扫描点；每档从大到小扫，OOM 即止')
    ap.add_argument('--steps', type=int, default=2,
                    help='每个 (档位,batch) 跑几轮 fwd/bwd/step（默认 2，'
                         '第 1 轮吃编译/预热，第 2 轮才是稳态峰值）')
    ap.add_argument('--try-torchair', action='store_true',
                    help='只回答「GC + TorchAir 能不能跑、对不对、多快」：'
                         '档位固定为 %s（含 gc 对照组），batch=4，steps=1。'
                         'batch 刻意取小 —— 这一档只测**能否追踪/编译 + 数值是否'
                         '正确**，与显存无关（显存走 --batch-sizes）。'
                         % ','.join(TORCHAIR_SMOKE_TIERS))
    ap.add_argument('--try-inductor', action='store_true',
                    help='只回答「GC + inductor 能不能跑、对不对、多快」：'
                         '档位固定为 %s（含 gc 对照组），batch=4，steps=1。'
                         'batch 刻意取小 —— 这一档只测**能否追踪/编译 + 数值是否'
                         '正确**，与显存无关（显存走 --batch-sizes）；CPU 上 bs=64 '
                         '要 45s 且占内存，拖慢定位循环。'
                          % ','.join(INDUCTOR_SMOKE_TIERS))
    ap.add_argument('--full-trace', action='store_true',
                    help='FAIL 时把**完整异常链的完整栈**打出来（默认关）。'
                         '报错行只能放一句话，`Please convert all Tensors to '
                         'FakeTensors…` 这种的线索在栈里 —— 没有栈就只能猜，'
                         '而 910C 上一轮往返十几分钟。')

    ap.add_argument('--no-checkpoint', dest='ckpt', action='store_false',
                    default=True,
                    help='连 gc / gc-ind 档也关掉梯度检查点（默认开）')
    ap.add_argument('--stop-at-first-oom', action='store_true',
                    help='某档一 OOM 就跳到下一档（默认继续往小扫）')
    ap.add_argument('--device', default='npu')
    args = ap.parse_args(argv)

    if args.try_torchair:
        args.tiers = ','.join(TORCHAIR_SMOKE_TIERS)
        args.batch_sizes = '4'
        args.steps = 1
    if args.try_inductor:
        args.tiers = ','.join(INDUCTOR_SMOKE_TIERS)
        args.batch_sizes = '4'
        args.steps = 1

    tiers = [t.strip() for t in args.tiers.split(',') if t.strip()]
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        print('未知档位 %s（可选: %s）' % (bad, ','.join(TIERS)), file=sys.stderr)
        return 2
    batches = sorted({int(b) for b in args.batch_sizes.split(',') if b.strip()},
                     reverse=True)

    # ---- 放行前向守卫（GC+编译档位才需要）----
    if _bypass_inforward_guard(tiers):
        print('\n[guard] ⚠ 已把 src.networks.backbone 的'
              ' assert_grad_checkpoint_compile_compatible 换成 no-op。')
        print('[guard]   那道守卫在 run_grad_segment 里**每次前向都调'
              '（backbone.py:421-423）**，不放行则 GC+编译第一次 fwd 就抛。')
        print('[guard]   本探针**只报告**它会不会拦，不改仓库代码 —— '
              '结论落地时必须自己决定是否动这两处：')
        print('[guard]     scripts/train_sft.py 编译时强制 _gc=0 的分支')
        print('[guard]     src/networks/backbone.py:303 的守卫')

    # ---- 环境 ----
    print('=' * 78)
    print('[env] probe_npu_graph_memory —— 图编译显存可行性探针（spike，只读）')
    print('=' * 78)
    try:
        import torch
    except Exception as e:
        print('import torch 失败:', repr(e))
        return 1
    print('[env] torch        = %s' % torch.__version__)

    try:
        import torch_npu
        print('[env] torch_npu    = %s' % getattr(torch_npu, '__version__', '?'))
    except Exception as e:
        print('[env] import torch_npu 失败:', repr(e))
        if str(args.device).startswith('npu'):
            print('[env] 这不是 NPU 环境，探针无法工作。')
            return 2
        print('[env] ⚠ 用 --device %s 继续 —— **只能验证探针本身**，'
              '显存/耗时结论一概不适用于 NPU。' % args.device)

    have_torchair = True
    try:
        import torchair
        print('[env] torchair     = %s' % getattr(torchair, '__version__', '?'))
    except Exception as e:
        have_torchair = False
        print('[env] import torchair 失败:', repr(e))
        print('[env] ⇒ torchair 档位（%s）全部跳过；'
              'inductor 档不受影响。'
              % ','.join(t for t in TIERS if TIER_SPEC[t][2] == 'torchair'))

    api = _mem_api()
    total = api['total']
    print('[env] 显存总量     = %s' % (
        '%.1f GB' % _gb(total) if total else '查不到'))
    if args.device == 'npu':
        try:
            print('[env] 设备型号     = %s' % torch.npu.get_device_name())
        except Exception:
            pass

    # ---- 口径对照（静态预测，不占显存）----
    ts = _extract_funcs(['_compile_linear_submodules',
                         '_rollback_linear_submodules'])
    import src.networks.katago_v7 as v7
    cfg = dict(v7.NBT_TF_CFG)
    n_layers = int(cfg.get('num_blocks', 11))
    heads = int(cfg['num_heads'])
    tokens = 19 * 19
    resident = {True: 6.9, False: 277.0}   # 同 train_sft.V7_RESIDENT_MB_PER_SAMPLE
    transient = 1.2                         # 同 V7_TRANSIENT_MB_PER_SAMPLE_FP16
    pred_fp16_ckpt = resident[True] / 2.0 + transient
    attn_mb_fp16 = (n_layers * heads * tokens * tokens * 2) / 1e6
    pred_fp16_nockpt = attn_mb_fp16 + transient

    print('\n[口径] 静态预测每样本显存（不实测，只列出两个互相矛盾的数）：')
    print('       GC 开  fp16  : %6.2f MB/样本   ← train_sft.py:721 用这个'
          % pred_fp16_ckpt)
    print('       GC 关  fp16  : %6.2f MB/样本   ← train_sft.py:727 用这个'
          % pred_fp16_nockpt)
    print('       GC 关  fp32  : %6.2f MB/样本   ← train_sft.py:737 warning 用这个'
          % resident[False])
    print('       （模型 %d 层 × %d 头 × %d² token；两种口径差 %.1f×，'
          % (n_layers, heads, tokens, resident[False] / pred_fp16_nockpt))
    print('        batch=200 时分别占 %.1f GB / %.1f GB ⇒ 结论完全相反）'
          % (_gb(pred_fp16_nockpt * 1e6 * 200),
             _gb(resident[False] * 1e6 * 200)))

    # ---- 后端 ----
    need = {TIER_SPEC[t][2] for t in tiers if TIER_SPEC[t][2]}
    backends = {}

    # 后端不可用时被丢掉的档位 —— 结论段要能区分「没请求」和「请求了但被跳过」，
    # 否则报告会显示「没有跑到任何 … 档位」，读的人分不清是自己没选还是环境问题。
    skipped = []

    def _drop(kind):
        gone = [t for t in tiers if TIER_SPEC[t][2] == kind]
        skipped.extend(gone)
        return [t for t in tiers if TIER_SPEC[t][2] != kind]

    if 'torchair' in need:
        if not have_torchair:
            tiers = _drop('torchair')
        else:
            try:
                import torchair
                _cfg = torchair.CompilerConfig()
                backends['torchair'] = torchair.get_npu_backend(
                    compiler_config=_cfg)
                print('[env] torchair backend = 已取得')
            except Exception as e:
                print('[env] 取 torchair backend 失败:', repr(e))
                tiers = _drop('torchair')

    if 'inductor' in need:
        # 先独立问一次「inductor 在这个栈上存不存在」—— 这是与本探针无关的
        # 环境事实，即使后面试验失败也已经拿到一手证据，不必再靠
        # `train_sft.py:4555` 那条 2023 年的、从未真正试过的假设。
        try:
            from torch._inductor import config as _ind_cfg  # noqa: F401
            print('[env] inductor      = torch._inductor 可导入')
        except Exception as e:
            print('[env] inductor      = torch._inductor 导入失败: %r' % (e,))
        try:
            import triton
            print('[env] triton        = %s'
                  % getattr(triton, '__version__', '?'))
        except Exception as e:
            print('[env] triton        = 未安装（昇腾 inductor 需要 triton-ascend）: %r'
                  % (e,))
        # backend 名就是字符串 'inductor'；真正的失败发生在首次 torch.compile，
        # 由 trial() 捕获并原样打印报错 —— 那正是我们要的证据。
        backends['inductor'] = 'inductor'

    device = torch.device(args.device)
    dtype = torch.float32
    # NPU 走 BF16（run.txt:28，910C 全程 BF16）。CPU 上**刻意不**开 autocast：
    # 本机实测 CPU 的 bf16 autocast 让单步从 5.0s 变 76.8s（15×，bf16 走的是
    # 慢速模拟路径），会把本地验证拖到没法用。CPU 跑只用于验证探针本身。
    amp_dtype = torch.bfloat16 if device.type == 'npu' else None

    # ---- 扫描 ----
    rows = []
    if not tiers:
        print('\n[scan] 没有可用档位（后端全部不可用）', file=sys.stderr)
        return 2
    print('\n[scan] 档位 × batch：每档从大到小扫，OOM 即标记并（可选）止损')
    print('-' * 78)
    for tier in tiers:
        # GC 只按 TIER_SPEC 开 —— 与「编译」不再由代码硬性互斥，
        # 互斥是**策略**（backbone 的守卫），probe 要测的正是这个策略值不值得。
        ckpt = bool(args.ckpt) and TIER_SPEC[tier][0]
        oomed = False
        for bs in batches:
            if oomed and args.stop_at_first_oom:
                rows.append({'tier': tier, 'batch': bs, 'oom': 'skipped'})
                continue
            label = '%-10s bs=%-5d' % (tier, bs)
            try:
                r = trial(tier, bs, device=device, dtype=dtype,
                          use_checkpoint=ckpt, amp_dtype=amp_dtype,
                          steps=args.steps, backends=backends, ns=ts, api=api)
                rows.append(r)
                print('  %s OK     peak=%6.2f GB  base=%5.2f GB  '
                      'compiled=%-3d  %.2fs  %8.0f samples/s  loss=%s%s%s'
                      % (label, _gb(r['peak']), _gb(r['base']),
                         r['n_compiled'], r['elapsed'], _sps(r) or 0.0,
                         _fmt_losses(r['losses']),
                         '  ⚠NaN' if r['bad_loss'] else '',
                         '  ⚠守卫会拦' if r.get('guard') else ''))
            except Exception as e:
                # 首行 = 根因（`_err_line` 已经保证多行取首尾、真因在链尾），
                # `tail` = 类型链，`stage` 决定这条结论能不能外推。
                first = _err_line(e)
                tail = _trace_tail(e)
                stage = _fail_stage(first, tail, e)
                if _is_oom(e):
                    oomed = True
                    rows.append({'tier': tier, 'batch': bs, 'oom': True,
                                 'err': first, 'trace_tail': tail,
                                 'stage': stage})
                    print('  %s OOM    %s' % (label, first))
                else:
                    rows.append({'tier': tier, 'batch': bs, 'oom': False,
                                 'err': first, 'trace_tail': tail,
                                 'stage': stage,
                                 'trace': traceback.format_exc()})
                    print('  %s FAIL   %s  [%s]' % (label, first, stage))
                    if tail and tail != first:
                        print('  %s        ↳ %s' % (' ' * len(label), tail))
                    if args.full_trace:
                        pf = _project_frames(e)
                        if pf:
                            print('  %s        ---- 本仓库的帧（截断也丢不了）----'
                                  % (' ' * len(label)))
                            for ln in pf:
                                print('  %s        %s' % (' ' * len(label), ln))
                        print('  %s        ---- 完整异常 ----'
                              % (' ' * len(label)))
                        for ln in _full_trace(e).splitlines():
                            print('  %s        %s' % (' ' * len(label), ln))
            finally:
                if api['empty_cache']:
                    try:
                        api['empty_cache']()
                    except Exception:
                        pass

    # ---- 汇总表 ----
    print('\n' + '=' * 78)
    print('[result] 峰值显存汇总（GB）')
    print('=' * 78)
    hdr = '%-11s' % 'tier' + ''.join('%9d' % b for b in sorted(batches))
    print(hdr)
    print('-' * len(hdr))
    for tier in tiers:
        line = '%-11s' % tier
        for bs in sorted(batches):
            r = next((x for x in rows
                      if x['tier'] == tier and x['batch'] == bs), None)
            if r is None:
                line += '%9s' % '-'
            elif r.get('oom') == 'skipped':
                line += '%9s' % 'skip'
            elif r.get('oom'):
                line += '%9s' % 'OOM'
            elif r.get('peak') is None:
                line += '%9s' % 'n/a'
            else:
                line += '%9.2f' % _gb(r['peak'])
        print(line)
    if total:
        print('%-11s %s' % ('total', '%.1f GB（横线以上任何数接近它就危险）'
                            % _gb(total)))

    # ---- 结论 ----
    print('\n' + '=' * 78)
    print('[verdict] 直接回答三个问题')
    print('=' * 78)

    # 1) 真实每样本
    ref = next((r for r in rows if r['tier'] == 'eager' and not r.get('oom')
                and r.get('peak') is not None), None)
    if ref and ref['base'] is not None:
        act = (ref['peak'] or 0) - (ref['base'] or 0)
        real = act / float(ref['batch'])
        print('1) GC 关掉后，实测每样本 = %.1f MB'
              % (_mb(real) if act > 0 else 0.0))
        print('     对照 fp16 预测 %6.1f MB  → 实测/预测 = %.2f'
              % (pred_fp16_nockpt, (real / 1e6) / pred_fp16_nockpt))
        print('     对照 fp32 字典  %6.1f MB  → 实测/预测 = %.2f'
              % (resident[False], (real / 1e6) / resident[False]))
        print('   ⇒ run.txt:134「batch 上限 200 出头」若按 277 算，是错的：')
        if total:
            print('     按实测推 batch 上限 ≈ %.0f（预算 %.1f GB ÷ 实测每样本）'
                  % ((total * 0.9) / (real if real > 0 else 1),
                     _gb(total * 0.9)))
    else:
        print('1) 拿不到 eager 档的干净峰值 ⇒ 无法判定每样本，见上面的 FAIL。')

    # 2) workspace —— **按 GC 开/关配对**：GC-off 的档减 eager，
    #    GC-on 的档减 gc。拿 GC-on 减 GC-off 会把「关 GC 省下的 12.67 MB×batch」
    #    全算到 workspace 头上，结论直接错一个数量级。
    print('\n2) 各档相对**同 GC 状态**基线的额外占用（= 图 workspace + 图缓冲）：')
    pairs = workspace_pairs(rows, tiers, batches)
    for tier, bs, d, base_tier, n_compiled in pairs:
        print('     %-10s bs=%-5d  workspace = %+6.2f GB'
              '   （vs %-5s，编译了 %s 个子模块）'
              % (tier, bs, _gb(d), base_tier, n_compiled))
    if not pairs:
        print('     没有可相减的 (tier, 基线) 配对 ⇒ 见上面的 FAIL/OOM。')

    # 3) batch 上限 —— 判据是**有没有 OOM**，与拿不拿得到峰值无关；
    #    用 peak 非空当判据会让「跑成功但内存 API 查不到」被误报成失败。
    print('\n3) 各档在 %.1f GB 卡上的 batch 上限（最后一个没 OOM 的点）：'
          % _gb(total) if total else '\n3) 各档 batch 上限：')
    for tier in tiers:
        best, bad_, fail_ = batch_ceiling(tier, rows)
        if best is not None:
            extra = ''
            if bad_:
                extra = '（更小的点 OOM: %s）' % sorted(bad_)
            if fail_:
                extra += '（失败: %s）' % sorted(fail_)
            print('     %-10s ≥ %d%s' % (tier, best, extra))
        else:
            why = 'OOM: %s' % sorted(bad_) if bad_ else ''
            if fail_:
                why += ('；' if why else '') + '失败: %s' % sorted(fail_)
            print('     %-10s 全部失败：%s' % (tier, why or '无数据'))

    # 4) 本次的正题：编译档（inductor 与 TorchAir 都算）
    print('\n' + '=' * 78)
    print('[verdict] 编译档到底行不行（inductor / TorchAir）')
    print('=' * 78)
    # 「编译档」= 任何后端非空的档位。原来写死成 INDUCTOR_TIERS + TORCHAIR_TIERS，
    # 于是 `linear / block / whole`（TorchAir 但 **GC 关**）**永远进不了这一段**：
    # 2026-10-07 实测 block/whole 双 batch 全挂 E19999，而 ①②④ 一个字不提，
    # 只在第 3 项留下一句「全部失败」—— 真因（反向图 CANN PreRun）反而没进结论。
    gc_compile_tiers = tuple(t for t in tiers if TIER_SPEC[t][2])
    ind_rows = [r for r in rows if r['tier'] in gc_compile_tiers]
    if not ind_rows:
        req = [t for t in gc_compile_tiers if t in skipped]
        if req:
            print('请求了 %s，但**一档都没跑** —— 后端不可用被跳过。' % ','.join(req))
            print('  见上面 [env] 对后端可用性的判断；这不是「跑挂了」，是没跑。')
        else:
            print('本次没有请求任何「编译」档位（可选: %s）。'
                  % ','.join(t for t in TIERS if TIER_SPEC[t][2]))
            print('  910C 上建议：`--try-torchair`（inductor 缺 triton-ascend 跑不了）。')
        print_throughput(rows)
    else:
        # 「跑通」= 没抛异常。**不能**要求 peak 非空 —— 内存 API 取不到时
        # （CPU、或 torch_npu 版本差异）peak 是 None，但试验本身是成功的。
        ok_rows = [r for r in ind_rows
                   if not r.get('oom') and not r.get('err')
                   and r.get('losses')]
        fail_rows = [r for r in ind_rows if r.get('err') and not r.get('oom')]

        if fail_rows:
            print('① **能用吗**：不能 —— 首次 torch.compile 就抛：')
            seen = set()
            for r in fail_rows:
                key = (r['tier'], r['err'])
                if key in seen:
                    continue
                seen.add(key)
                print('     [%s] %s' % (r['tier'], r['err']))
                if r.get('trace_tail') and r['trace_tail'] != r['err']:
                    print('     %s↳ %s' % (' ' * (len(r['tier']) + 2),
                                           r['trace_tail']))

            # 关键分层：**哪一层**挂的，决定本机结论能不能外推到 NPU。
            stages = {r.get('stage', 'other') for r in fail_rows}
            if 'dynamo' in stages:
                print('   ⇒ 其中 **追踪阶段（Dynamo）就挂了** —— 这一层**与设备无关**，')
                print('     不随设备变化（本机是 %s，结论不因换卡而改变）。' % args.device)
                for r in fail_rows:
                    hint = ctxfn_hint(r.get('err'), r.get('trace_tail'))
                    if hint:
                        print('')
                        for ln in hint:
                            print(ln)
                        break
                else:
                    print('     （不是已知的 context_fn 限制，见首行原文。）')
            backend_stages = stages & {'inductor', 'torchair'}
            if backend_stages:
                print('   ⇒ 其中后端阶段失败（%s）—— **追踪已过**，卡在代码生成/'
                      '图编译；这一层与后端和设备都有关，' % '+'.join(sorted(backend_stages)))
                print('     本机结论不能外推到别的环境，**必须在目标环境重测**才算数。')
            if 'dynamo' not in stages and not backend_stages and stages:
                print('   ⇒ 报错既不在追踪层也不在后端层，见上面的首行。')

            # inductor 档 != TorchAir 档，别把一侧的缺件当成另一侧的结论。
            if 'inductor' in stages:
                print()
                print('   ⚠ 注意档位口径：`gc-ind` / `gc-ind-all` 走的是 **torch.compile 的')
                print('     inductor 后端**，不是 TorchAir。TorchAir 档是 linear/block/whole')
                print('     / gc-torchair / gc-torchair-all（TIER_SPEC 里 kind==`torchair`）。')
                print('     `Compiler: cl is not found` 是**本机 Windows 缺 C++ 编译器**，')
                print('     与 TorchAir 能否跑通**无关**，不要据此给 TorchAir 下结论。')
            if 'torchair' in stages:
                print()
                print('   ⚠ 这是 TorchAir / CANN 侧的失败，**与 inductor 是否安装无关**。')
                print('     不要据此得出「要升级 torch_npu / 装 triton-ascend」的结论。')

            blob = ' '.join((r.get('err') or '') + ' ' + (r.get('trace_tail') or '')
                            for r in fail_rows).lower()
            version_issue = any(k in blob for k in (
                'triton', 'not registered', 'unknown backend',
                'npu backend', 'torch_npu', 'ascend'))
            if version_issue:
                print('   ⇒ 这是**一手证据**，不是 2023 年那条没试过的假设。')
                print('     报错指向后端本身不存在 ⇒ 要走这条路必须先升级 torch_npu')
                print('     （triton-ascend 捆绑 2.7.1.post8，本机 2.1.0.post10；')
                print('     官方 inductor 章节在 26.1.0）。')
            else:
                print('   ⇒ 这是**一手证据**，不是 2023 年那条没试过的假设。')
                print('     报错**不是**「后端不存在」，不要直接归因于版本。')
        elif ok_rows:
            print('① **能用吗**：能 —— %d 个 GC+inductor 试验跑通。' % len(ok_rows))

        nan_rows = [r for r in ok_rows if r.get('bad_loss')]
        if ok_rows:
            if nan_rows:
                print('② **对吗**：✗ 出现 NaN/Inf（loss 或梯度范数）—— %s'
                      % ', '.join('%s@bs=%d' % (r['tier'], r['batch'])
                                  for r in nan_rows))
                print('   ⇒ 组合可用但**数值不正确**，不能上。')

            # 与同 batch 的**同 GC 状态**基线逐位对比（种子固定 ⇒ 权重与输入相同）。
            # 梯度范数是**决定性**的那一项：checkpoint 重算 + Dynamo 图断开的经典
            # 症状是 loss 完全正常、梯度却已经错了，只看 loss 会漏掉。
            printed_hdr = False
            for r in sorted(ok_rows, key=lambda x: (x['tier'], x['batch'])):
                # GC-off 的档必须拿 `eager` 当基线：拿 `gc` 比会把「重算省下的
                # 显存/时间」与「图编译的数值差异」混成一个数，方向都可能反。
                base = 'gc' if TIER_SPEC[r['tier']][0] else 'eager'
                ref = next((x for x in rows if x['tier'] == base
                            and x['batch'] == r['batch']
                            and not x.get('oom') and not x.get('err')
                            and x.get('losses')), None)
                if not ref or not ref.get('losses'):
                    continue
                if not printed_hdr:
                    print('     与同 GC 状态基线（%s）逐位对比（同权重同输入）：'
                          % '/'.join(sorted({('gc' if TIER_SPEC[x['tier']][0]
                                              else 'eager')
                                             for x in ok_rows})))
                    printed_hdr = True
                ldiff = _max_rel(r['losses'], ref['losses'])
                gdiff = _max_rel(r.get('grad_norms') or [],
                                 ref.get('grad_norms') or [])
                lflag = '✓' if (ldiff is not None and ldiff < 1e-3) else '✗'
                gflag = '✓' if (gdiff is not None and gdiff < 1e-3) else '✗'
                print('       %-10s bs=%-5d  loss rel err = %-10s %s | '
                      'grad rel err = %-10s %s'
                      % (r['tier'], r['batch'],
                         _fmt_rel(ldiff), lflag, _fmt_rel(gdiff), gflag))
            if printed_hdr:
                print('     （阈值 1e-3；超出即判定该档位算错了）')
            elif ok_rows and not nan_rows:
                print('② **对吗**：跑通且 loss/梯度均有限，但**没有对照组**，'
                      '判不了对错 —— 加上 `--tiers gc,...` 再跑一次。')

        # 3) 吞吐 —— 定义在外面，两个分支共用一份实现。
        print_throughput(rows)

        # 4) 落地障碍：守卫
        blocked = [r for r in ok_rows if r.get('guard')]
        if blocked:
            print('④ **能直接上吗**：不能 —— 仓库守卫会拦：')
            print('     %s' % blocked[0].get('guard_msg', ''))
            print('   ⇒ 即使技术上跑通，落地还要改两处策略：')
            print('     `train_sft.py` 编译时强制 `_gc = 0` 的分支')
            print('     `src/networks/backbone.py:303` 的守卫')
            print('   ⇒ 改之前必须先看 ② 的数值结论。')

    print('\n下一步怎么用这些数：')
    print('  · 若 ① 失败在 import/注册阶段 ⇒ 当前栈没有 inductor，结论是「要升级」，')
    print('    并把确切报错贴进 run.txt 替换 `:4555` 那条 2023 年的假设。')
    print('  · 若 ① 失败但报错是环境缺件（编译器/triton 没装）⇒ 还判不了 inductor')
    print('    本身，先补环境再跑一次，不要直接下结论。')
    print('  · 若 ① 成功、② 有 NaN ⇒ GC+inductor 数值不可用，回到 TorchAir 路线，')
    print('    改去调 CompilerConfig 的融合参数。')
    print('  · 若 ①② 都过、④ 拦住 ⇒ 技术可行、策略要改；此时先比 ③ 的耗时，')
    print('    确认真的比现网快，再去动 train_sft.py 的 _gc=0 分支和 backbone 守卫。')
    print('  · 若实测每样本接近 %.1f（fp16 预测）⇒ 两个口径里 fp16 那个对，'
          % pred_fp16_nockpt)
    print('    run.txt:134 按 277 算的「200 出头」是错的，可回填 train_sft.py:727/:737。')
    print('  · 若实测接近 %.1f（fp32 字典值）⇒ GC 关掉后 batch 要按 277 算，'
          % resident[False])
    print('    此时 GC-off 的块级编译没余量。')
    print('  · 若 workspace（第 2 项）只有几百 MB ⇒ `:3653` 的 OOM 记录确系')
    print('    GC 互斥所致，可重新评估整模型/块级编译。')
    print('\n⚠ 本文件是 spike：结论拿到后请删除，真实改动写进 train_sft.py 并')
    print('   让 tests/test_npu_graph_compile.py 重新钉住。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
