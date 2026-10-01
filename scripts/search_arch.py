"""架构参数搜索：实测参数量 / FLOPs / 激活显存，并按显存预算排序。

为什么需要这个工具
------------------
调参时最容易出错的是**用 FLOPs 代替实测显存**。本仓库自己的实测就证明了
二者会给出相反的结论：

    v18 (res8 + convnext4 + attn5)   FLOPs 8.49G  激活 31.0G
    同结构但 convnext4 -> res4       FLOPs 9.54G  激活 25.2G   <- FLOPs 更高却更快

原因是 ConvNeXtBlock 的 pwconv1 把通道从 C 扩到 4C，要物化 (B, 4C, H, W)
的巨大张量；逐点卷积 FLOPs 极低，但**内存带宽消耗极高**。在 NPU 上带宽才是
瓶颈，于是「纸面省算力」的 ConvNeXt 实际更慢。

本工具的三项测量都是实测，不含手算：
  * 参数量    —— 直接 sum(p.numel())
  * FLOPs     —— forward hook 统计卷积/线性，按 batch=1
  * 激活显存  —— torch.autograd.graph.saved_tensors_hooks 精确计量
                 **真正被反向保留**的张量总量，因此天然正确反映
                 gradient checkpointing（backbone 开了 checkpoint 后
                 只保留块输入，value head 未开则全量保留）。
                 ⚠ 检查点的**语义由本探针钉死**（见 measure 的
                 use_checkpoint 参数）：主干的 checkpoint 实现怎么改，
                 计量口径都不动——否则 k 标定会随实现漂移。

用法
----
    python scripts/search_arch.py --list
    python scripts/search_arch.py --preset            # 整表（= --preset all）
    python scripts/search_arch.py --preset v12        # 既有预设单项
    python scripts/search_arch.py --preset se         # KataGo SE 候选（KATAGO_SE_CFG 结构）
    python scripts/search_arch.py --grid
    python scripts/search_arch.py --sweep attn_window --sweep value_res_blocks

局限
----
**准确率无法在此测量**（需真实数据 + NPU 数小时）。本工具只覆盖
速度与显存两个维度；准确率需实跑 A/B。若某配置的架构在历史上有
实测记录（如 V12 的 50% top1），会在结果里标注作为参考。

**显存/耗时的绝对值只在 v18 锚点上实测标定**（单点，k=0.8235）；对其他
结构（如 arch='se_bottleneck' 的 SE 候选）属**外推，误差未知** —— 每项输出
的实测/外推口径见 project() 的 docstring，引用时不要混。
"""

import argparse
import itertools
import os
import sys

import torch
import torch.nn as nn
import torch.utils.checkpoint  # noqa: F401  （探针自己包检查点，显式 import）

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.networks.alphanet import AlphaGoNet  # noqa: E402

ACTION_SIZE = 362          # 19 路
PROBE_BS = 8               # 显存探针 batch（小 batch 省内存，结果线性外推）
NPU_GB = 32.0              # 910A 单卡 HBM
SAFE_FRAC = 0.90           # 留 10% 余量给碎片/workspace
SAFE_GB = NPU_GB * SAFE_FRAC

# 已实测的锚点（4 卡 910A，batch 2800/卡，attn-window 5，use_checkpoint 1）
ANCHOR = {
    'name': 'v18',
    'mem_gb': 31.12,
    'step_s': 4.25,
    'samples_per_s': 2635,
    'flops': 8_489_705_920,
    'cfg': dict(backbone_channels=192, backbone_res_blocks=17,
                attention_mode='none', num_attention_layers=0, num_heads=4,
                attn_mode='window_global', attn_window=5,
                res_blocks=8, convnext_blocks=4, attn_blocks=5,
                value_channels=96, value_res_blocks=8,
                policy_channels=128, policy_layers=3),
}

# 历史上有实测准确率记录的配置（仅作参考，不代表本工具能预测准确率）
V12_CFG = dict(backbone_channels=192, backbone_res_blocks=17,
               attention_mode='mix', num_attention_layers=4, num_heads=4,
               attn_mode='window_global', attn_window=5,
               res_blocks=0, convnext_blocks=0, attn_blocks=0,
               value_channels=64, value_res_blocks=2,
               policy_channels=32, policy_layers=2)

# KataGo SE-bottleneck 候选（**现行**训练结构，katago-se-v1）
# ---------------------------------------------------------------------------
# 本表 = scripts/train_sft.py 的 KATAGO_SE_CFG **结构键**镜像到
# AlphaGoNet.__init__ 的参数名（channels->backbone_channels、
# blocks->backbone_res_blocks）。两处必须同步：
# tests/test_search_arch.py::test_se_preset_mirrors_katago_se_cfg 用**活的**
# KATAGO_SE_CFG 逐键核对本表，漂移当场变红。
#
# 刻意不进本表的 KATAGO_SE_CFG 键：
#   * in_channels=12 —— 归 preset_in_channels('se')（通道数是预设属性，
#     不是可拨的旋钮，理由见该函数 docstring）；
#   * grad_checkpoint —— 行为键；计量口径由探针钉死（见 measure）；
#   * params_backbone / params_total —— train_sft 侧对账硬数；本工具不抄，
#     由 test_se_measure_matches_katago_budget 现场实测后与之对账。
# attn_mode/attn_window 显式写出：KATAGO_SE_CFG 不传它们，AlphaGoNet 默认
# global/7 —— 写出来让本表自洽（emit/扫描读键时不会退回 window_global/5）。
SE_CFG = dict(backbone_channels=240, backbone_res_blocks=17,
              attention_mode='mix', num_attention_layers=4, num_heads=4,
              attn_mode='global', attn_window=7,
              res_blocks=0, convnext_blocks=0, attn_blocks=0,
              value_channels=96, value_res_blocks=2,
              policy_channels=128, policy_layers=3,
              arch='se_bottleneck')

# （退役记录，勿复活）v21 权威参数锚点 ANCHOR_V21 随 v21 架构（Mamba /
# Transformer / CrossAttn，曾是此处的 200+ 行字面量）于 katago-se-v1 一并删除：
# 锚点记录的结构已从 src/networks/ 拔除，静态表失去被测对象，`--preset v21`
# 与 run_anchor_v21 随之移除（tests/test_search_arch.py::
# test_v21_machinery_is_retired 钉住退场）。它当年的用途 —— P4.2 预算仲裁 ——
# 由 KATAGO_SE_CFG 的实测对账（本文件 SE_CFG + measure）接替。


class _ProbeCheckpointBlocks(nn.Module):
    """探针专属的检查点包裹：`use_checkpoint=True` 时替换 `backbone.blocks`。

    为什么计量要自己包，而不是把 `use_checkpoint` 交给主干
    -----------------------------------------------------
    锚点 31.12GB（v18 / 4 卡 910A / use_checkpoint 1）是用
    `torch.utils.checkpoint.checkpoint_sequential(blocks, len(blocks), x)`
    实测出来的，该函数的**文档语义**就是「除最后一段外都不保留中间激活」
    （源码注释 "the last chunk has to be non-volatile"）——最后一块的内部
    激活在那次实测里是**保留**的。标定因子 k≈0.83（= 文档里的「高估 ~21%」）
    与 `test_attention_window_affects_memory`（attn_window 影响显存）都建立
    在这套口径上：锚点配置的块序是 res→convnext→attn，注意力块恰好是最后一块。

    P4.6b 把主干的 `use_checkpoint=True` 换成了「每一块都检查点」（新
    `run_grad_segment` 的逐块循环没有沿用「最后一段不检查点」），同一探针下
    锚点保留量 115MB→70MB、k 0.83→1.35、attn_window 敏感性被抹平。教训：
    **计量口径不能由被测实现决定**——主干的检查点策略一变，k 和所有绝对
    预测就静默漂移（且方向是低估，会把装不下的配置判成 fits）。

    所以：`measure()` 构造模型时恒传 `use_checkpoint=False` 并显式关掉主干
    自己的开关，再由本类按锚点口径包裹。`use_checkpoint=False` 则完全不包。
    """
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        # 与锚点实测同一函数、同一粒度：segments=len(blocks) ⇒ 逐块检查点，
        # 最后一块保留（checkpoint_sequential 的文档语义，torch 2.12 实测
        # 锚点保留量 115,197,536 B / k=0.8235 与 HEAD 逐字节一致）。
        return torch.utils.checkpoint.checkpoint_sequential(
            self.blocks, len(self.blocks), x, use_reentrant=False)


def measure(cfg, probe_bs=PROBE_BS, use_checkpoint=True, in_channels=12):
    """返回 (参数量, FLOPs@batch1, 保留激活字节@probe_bs)。

    in_channels 默认 12 —— 既有调用方零回归；其他通道数显式传入即可测。

    架构由 cfg 自带（`arch` 键，缺省 'resnet'）
    -------------------------------------------
    cfg 是「喂给 AlphaGoNet.__init__ 的结构键」，所以 arch 跟着 cfg 走：
    SE 候选（SE_CFG）自带 `arch='se_bottleneck'`，既有调用方不带 arch ⇒ 仍按
    'resnet' 建网，零回归。先 pop 再展开，避免与显式 arch= 撞成重复关键字。

    use_checkpoint 的语义由**本探针**钉死（确定性契约）
    -----------------------------------------------------
    * True（默认）：把 `model.backbone.blocks` 换成 `_ProbeCheckpointBlocks`
      （锚点口径的 `checkpoint_sequential` 逐块检查点）；
    * False：完全不检查点。
    两种情况下主干自己的 `use_checkpoint` / `set_grad_checkpointing` 都被
    显式关掉——探针**不继承**主干策略。理由见 `_ProbeCheckpointBlocks` 的
    docstring：k 标定要求「计量口径 ≡ 锚点实测口径」，而主干的检查点实现
    是会变的（P4.6b 就变了）；语义归探针，主干怎么改都不影响计量。
    """
    kwargs = dict(cfg)
    arch = kwargs.pop('arch', 'resnet')
    model = AlphaGoNet(in_channels=in_channels, action_size=ACTION_SIZE,
                       arch=arch, attention_dropout=0.0,
                       use_checkpoint=False, **kwargs)
    backbone = getattr(model, 'backbone', None)
    if backbone is None or not hasattr(backbone, 'blocks'):
        raise RuntimeError(
            'AlphaGoNet 没有 backbone.blocks —— 探针的检查点语义要挂在这上面，'
            '请同步修改 scripts/search_arch.py 的 measure()')
    # 主干自己的检查点开关：恒关（哪怕构造函数/子类把它打开）。语义归探针。
    backbone.use_checkpoint = False
    if use_checkpoint:
        backbone.blocks = _ProbeCheckpointBlocks(backbone.blocks)
    n_params = sum(p.numel() for p in model.parameters())

    # ---- FLOPs：必须 batch=1，否则被 batch 放大 ----
    flops = [0]

    def cf(mod, inp, out):
        oc, oh, ow = out.shape[1], out.shape[2], out.shape[3]
        k = mod.kernel_size[0] * mod.kernel_size[1]
        flops[0] += int(mod.in_channels // mod.groups * oc * k * oh * ow * 2)

    def lf(mod, inp, out):
        flops[0] += int(mod.in_features * mod.out_features
                        * (out.numel() / out.shape[-1]) * 2)

    handles = [m.register_forward_hook(cf)
               for m in model.modules() if isinstance(m, nn.Conv2d)]
    handles += [m.register_forward_hook(lf)
                for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        model(torch.zeros(1, in_channels, 19, 19))
    for h in handles:
        h.remove()

    # ---- 激活：精确统计反向真正保留的**底层存储**字节 ----
    # 注意不能按 t.numel() 累加：反向会保存很多视图（如切片、转置），它们共享
    # 同一块 storage，逐个累加会严重重复计数（实测高估 48%）。必须按
    # storage 的 data_ptr 去重。
    retained = [0]
    seen = set()

    def pack(t):
        try:
            st = t.untyped_storage()
            key = st.data_ptr()
            if key not in seen:
                seen.add(key)
                retained[0] += st.nbytes()
        except Exception:      # 少数非张量/元张量退回数值估计
            retained[0] += t.numel() * t.element_size()
        return t

    def unpack(t):
        return t

    model.train()          # checkpointing 只在 training 时生效
    x = torch.randn(probe_bs, in_channels, 19, 19, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        policy, value = model(x)
        loss = policy.square().mean() + value.square().mean()
        loss.backward()
    model.zero_grad(set_to_none=True)
    del model
    return n_params, flops[0], retained[0]


def act_gb(retained_bytes, batch):
    """把探针结果外推到目标 batch 的激活显存（GB）。

    注意这里统计的是「反向保存的全部张量之和」，而 gradient checkpointing 下
    同一时刻只有**一个块**的重算临时量存活，故本值**系统性高估**实测值
    （在 v18 锚点上高估约 21%）。高估方向对所有配置一致，因此**相对排序
    可信**；绝对值由 CALIB 因子标定到实测锚点。
    """
    return retained_bytes * batch / PROBE_BS / 1024 ** 3


def overhead_gb(n_params):
    """与配置无关的常驻显存：权重 + 梯度 + AdamW 双矩 + EMA shadow。

    参数是 FP32（train_sft.py 只设 memory_format，从未 .half()），
    故 4B/元素；AdamW 两份状态 8B；梯度 4B；EMA shadow 4B。
    """
    return n_params * (4 + 4 + 8 + 4) / 1024 ** 3


_CALIB = None


def calib_factor():
    """标定因子：把「保存张量之和」换算到实测占用。

    延迟计算并缓存。用 v18 的唯一实测点（4 卡 910A / B=2800 / 31.12GB）
    做单点标定——只有相对排序可靠时，绝对值才有意义。
    """
    global _CALIB
    if _CALIB is None:
        n, _, ret = measure(ANCHOR['cfg'])
        pred = act_gb(ret, 2800) + overhead_gb(n)
        _CALIB = ANCHOR['mem_gb'] / pred if pred > 0 else 1.0
    return _CALIB


def project(cfg, batch, anchor=ANCHOR, in_channels=12):
    """给出某配置在目标 batch 下的预测指标（in_channels 默认 12，零回归）。

    ⚠ 返回值的**口径分两类**，引用时不能混为一谈：
    * **本机实测**：params（sum(p.numel())）、flops（batch=1 forward hook）；
    * **自 v18 锚点外推（误差未知）**：act_gb（探针实测保留字节 × batch 线性
      外推 × calib_factor() 单点标定）、total_gb（act + 常驻开销）、step_s
      （v18 的 4.25s × FLOPs 比）、samples_per_s、fits。
      锚点是 v18 / 4 卡 910A / B=2800 / 31.12GB —— 对**非 v18 结构**（如
      arch='se_bottleneck' 的 SE 候选），后四项属跨结构外推，工具无法给出
      误差界；`--calibrate` 只能证明 v18 自身往返自洽。
    """
    n, f, ret = measure(cfg, in_channels=in_channels)
    act = act_gb(ret, batch) * calib_factor()
    total = act + overhead_gb(n)
    ratio = f / anchor['flops']
    step = anchor['step_s'] * ratio
    sps = batch * 4 / step
    return dict(params=n, flops=f, act_gb=act, total_gb=total,
                step_s=step, samples_per_s=sps, fits=total <= SAFE_GB)


def calibrate(verbose=True):
    """用实测锚点验证外推模型是否可信。"""
    n, _, ret = measure(ANCHOR['cfg'])
    act = act_gb(ret, 2800)
    raw = act + overhead_gb(n)
    k = calib_factor()
    scaled = raw * k
    err = (scaled - ANCHOR['mem_gb']) / ANCHOR['mem_gb']
    if verbose:
        print('=== 校准核对（锚点 {} 4卡 910A B=2800 实测 {:.2f}GB）==='.format(
            ANCHOR['name'], ANCHOR['mem_gb']))
        print('  原始估算 {:.2f}GB（高估 {:.0%}，因 checkpoint 重算临时量'
              '被按同时存活计）'.format(raw, raw / ANCHOR['mem_gb'] - 1))
        print('  标定因子 {:.4f} -> {:.2f}GB，相对误差 {:.1%}'.format(
            k, scaled, err))
        print('  相对排序可信（高估方向对所有配置一致）；绝对值仅靠单点锚点，'
              '换硬件/配置需重标')
    return err


def fmt_row(name, r, note=''):
    return '{:<30}{:>9}{:>9}{:>9}{:>9}{:>8}{:>10}  {}'.format(
        name, '{:.2f}M'.format(r['params'] / 1e6),
        '{:.2f}G'.format(r['flops'] / 1e9),
        '{:.1f}G'.format(r['act_gb']), '{:.1f}G'.format(r['total_gb']),
        '{:.2f}x'.format(r['step_s'] / ANCHOR['step_s']),
        '{:.0f}'.format(r['samples_per_s']),
        ('OK  ' if r['fits'] else '超显存 ') + note)


HEADER = '{:<30}{:>9}{:>9}{:>9}{:>9}{:>8}{:>10}  {}'.format(
    '配置', '参数', 'FLOPs', '激活', '合计显存', 's/step', 'samples/s', '判定')
SEP = '-' * 108


def preset_cfgs():
    A = ANCHOR['cfg']
    out = [
        ('v18 原样（锚点）', A, 2800, '当前基线，显存超限'),
        ('V12 原样', V12_CFG, 2800, '历史 top1≈50% 的结构'),
        ('B: ConvNeXt4→Res4', {**A, 'res_blocks': 12, 'convnext_blocks': 0}, 2800,
         'FLOPs 更高但显存更低'),
        ('F: V12块序+v18头', {**V12_CFG, 'value_channels': 96,
                             'value_res_blocks': 8,
                             'policy_channels': 128, 'policy_layers': 3},
         2800, ''),
        # SE 候选：目标单卡 batch 3200（train_sft 实际要跑的结构）。
        # note 必须带「外推」字样 —— 它的显存/s/step 不是在它自己身上实测的，
        # 而是从 v18 锚点标定外推的（见 project 的 docstring）。
        ('SE: KataGo 240ch（候选）', SE_CFG, 3200,
         'KATAGO_SE_CFG 结构；显存/s/step 自 v18 锚点外推（误差未知）'),
    ]
    return out


# --preset 的取值形态：all=整表（裸 --preset / 无参数时的默认），
# v18/v12/b/f=既有预设单项，se=KataGo SE 候选单项。
PRESET_CHOICES = ('all', 'v18', 'v12', 'b', 'f', 'se')
_PRESET_INDEX = {'v18': 0, 'v12': 1, 'b': 2, 'f': 3, 'se': 4}   # 对应 preset_cfgs() 下标


def preset_entries(key='all'):
    """把 --preset 的取值映射到 (name, cfg, bs, note) 列表（纯选择，不测量）。"""
    if key == 'all':
        return preset_cfgs()
    return [preset_cfgs()[_PRESET_INDEX[key]]]


def preset_in_channels(key):
    """预设的输入通道数：**来源 = 预设推导**（非 CLI 自由参数）。

    通道数是被测结构的固有属性，不是可随意拨动的旋钮：若开 `--in-channels`
    这类显式旗标，12ch 训练的 cfg 配 17ch 输入能照跑不误，产出一张看似正常、
    实际错位的对比表——而对比表正是本工具的全部价值。故每个预设自带通道数。
    当前全部预设（含 SE 候选 = KATAGO_SE_CFG['in_channels']）都是 12ch；
    17ch 的 v21 锚点已随 v21 架构退役（见 SE_CFG 上方的退役记录）。
    key 保留形参位：日后若有非 12ch 的预设，按 key 分派即可。
    """
    return 12


def run_preset(batch=None, key='all'):
    print(HEADER)
    print(SEP)
    rows = []
    in_ch = preset_in_channels(key)
    for name, cfg, bs, note in preset_entries(key):
        if batch:
            bs = batch
        r = project(cfg, bs, in_channels=in_ch)
        print(fmt_row(name, r, note))
        rows.append((name, r))
    return rows


def run_sweep(keys, base=None, batch=2800, extra=None):
    """对若干维度做单变量扫描。"""
    base = dict(base or ANCHOR['cfg'])
    if extra:
        base.update(extra)
    DOMAIN = {
        'attn_window': [3, 5, 7, 11, 19],
        'num_attention_layers': [2, 4, 6, 8],
        'num_heads': [2, 4, 8],
        'backbone_channels': [128, 144, 160, 176, 192, 224],
        'backbone_res_blocks': [12, 17, 20, 24, 30],
        'res_blocks': [4, 8, 12, 16, 20],
        'convnext_blocks': [0, 2, 4, 8],
        'attn_blocks': [0, 2, 4, 5, 8],
        'value_res_blocks': [1, 2, 3, 5, 8, 11],
        'value_channels': [32, 48, 64, 96, 128],
        'policy_channels': [32, 64, 128, 192],
        'policy_layers': [2, 3],
    }
    base_r = project(base, batch)
    print(HEADER)
    print(SEP)
    print(fmt_row('基准', base_r, 'batch={}'.format(batch)))
    for k in keys:
        if k not in DOMAIN:
            print('未知维度: {}（可选: {}）'.format(k, ', '.join(DOMAIN)))
            continue
        print()
        for v in DOMAIN[k]:
            cfg = dict(base)
            cfg[k] = v
            r = project(cfg, batch)
            # mix 模式下 res/convnext/attn_blocks 被忽略，标注出来
            note = ''
            if k in ('res_blocks', 'convnext_blocks', 'attn_blocks') and \
                    base.get('attention_mode') == 'mix' and base.get(k, 0) == 0:
                note = '(mix 模式下被忽略)'
            print(fmt_row('{}={}'.format(k, v), r, note))
    return base_r


def run_grid(batch=2800):
    """在显存预算内做粗网格，按预测吞吐排序。"""
    axes = dict(
        backbone_channels=[160, 192],
        backbone_res_blocks=[17, 20, 24],
        num_attention_layers=[4, 6],
        value_res_blocks=[2, 3, 5],
    )
    keys = list(axes)
    base = {**ANCHOR['cfg'], 'attention_mode': 'mix',
            'res_blocks': 0, 'convnext_blocks': 0, 'attn_blocks': 0}
    results = []
    print(HEADER)
    print(SEP)
    for combo in itertools.product(*(axes[k] for k in keys)):
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        r = project(cfg, batch)
        name = 'ch{}/blk{}/attn{}/v{}'.format(*combo)
        print(fmt_row(name, r))
        if r['fits']:
            results.append((name, r, dict(cfg)))
    print()
    print(SEP)
    print('显存预算内，按预测吞吐排序:')
    for name, r, _ in sorted(results, key=lambda t: -t[1]['samples_per_s']):
        print('  {:<24}{:>9.2f}M  {:>6.1f}G  {:>8.0f} samples/s'.format(
            name, r['params'] / 1e6, r['total_gb'], r['samples_per_s']))
    return results


def max_batch(cfg, limit=4200, lo=800):
    """求该配置在显存预算内能装下的最大 batch（二分）。

    吞吐 ≈ batch / s_per_step，而 s_per_step 与 batch 近似无关（compute-bound），
    所以「显存允许的最大 batch」就是该配置的最优吞吐点。
    """
    while lo < limit:
        mid = (lo + limit + 1) // 2
        if project(cfg, mid)['fits']:
            lo = mid
        else:
            limit = mid - 1
    return lo


def run_speed_curve(batch_list=None):
    """给出一条「通道数 vs 可用 batch / 吞吐」曲线，供取舍。

    准确率无法在此测量，所以只呈现速度-显存侧的可行域；通道数是准确率风险
    最主要的来源（V12 用 192ch 拿到 top1≈50%），故把通道数单列为横轴。
    """
    print('通道数敏感度：每个通道数在显存预算内能装下的最大 batch 与对应吞吐')
    print(SEP)
    print('{:<12}{:>10}{:>10}{:>11}{:>10}{:>10}{:>9}'.format(
        '通道数', '块构成', '参数', 'FLOPs', '最大B', '显存', 'samples/s'))
    SEG = dict(ANCHOR['cfg'])
    MIX = {**ANCHOR['cfg'], 'attention_mode': 'mix', 'num_attention_layers': 4,
           'res_blocks': 0, 'convnext_blocks': 0, 'attn_blocks': 0}
    for ch in (128, 144, 160, 176, 192):
        for label, base in (('segmented', SEG), ('mix', MIX)):
            cfg = dict(base)
            cfg['backbone_channels'] = ch
            # 先把 value 降到 3 块腾出预算（value head 未 checkpoint，占用与
            # 通道数无关的固定份额）
            cfg['value_res_blocks'] = 3
            b = max_batch(cfg)
            r = project(cfg, b)
            blocks = '{}R+{}C+{}A'.format(
                cfg.get('res_blocks', 0), cfg.get('convnext_blocks', 0),
                cfg.get('attn_blocks', 0)) if label == 'segmented' \
                else '17blk mix(4A)'
            print('{:<12}{:>10}{:>9.2f}M{:>10.2f}G{:>10}{:>9.1f}G{:>9.0f}'.format(
                ch, blocks, r['params'] / 1e6, r['flops'] / 1e9, b,
                r['total_gb'], r['samples_per_s']))
    print()
    print('注：全部已按 value_res_blocks=3 设定（value head 不受 checkpoint '
          '保护，8->3 约省 2.3GB，近乎不损准确率——二分类输出）')


def build_parser():
    ap = argparse.ArgumentParser(description='架构参数搜索（实测，非手算）')
    ap.add_argument('--calibrate', action='store_true', help='只做校准核对')
    ap.add_argument('--list', action='store_true', help='列出预设配置')
    ap.add_argument('--preset', nargs='?', const='all', default=None,
                    choices=PRESET_CHOICES,
                    help='跑预设配置：all=整表（裸 --preset 或无参数的默认），'
                         'v18/v12/b/f=既有预设单项，se=KataGo SE 候选')
    ap.add_argument('--grid', action='store_true', help='跑粗网格搜索')
    ap.add_argument('--curve', action='store_true',
                    help='通道数 vs 可用最大 batch / 吞吐 的可行域曲线')
    ap.add_argument('--sweep', action='append', default=[],
                    help='单变量扫描的维度名，可重复')
    ap.add_argument('--batch', type=int, default=2800, help='每卡 batch')
    ap.add_argument('--base', default='v18', choices=['v18', 'v12'],
                    help='扫描的基准配置')
    ap.add_argument('--mix', action='store_true', help='扫描时用 mix 构建模式')
    ap.add_argument('--emit-flags', action='store_true',
                    help='为通过的候选输出 train_sft.py 参数')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not (args.calibrate or args.list or args.preset is not None or args.grid
            or args.sweep or args.curve):
        args.preset = 'all'

    if args.calibrate:
        err = calibrate()
        sys.exit(0 if abs(err) < 0.12 else 1)

    if args.list:
        print('预设配置:')
        for name, cfg, bs, note in preset_cfgs():
            print('  {:<30} batch={:<5} {}'.format(name, bs, note))
        print('\n可用扫描维度: attn_window num_attention_layers num_heads '
              'backbone_channels\n'
              '              backbone_res_blocks res_blocks convnext_blocks '
              'attn_blocks\n'
              '              value_res_blocks value_channels policy_channels '
              'policy_layers')
        return

    if args.curve:
        calibrate(verbose=True)
        print()
        run_speed_curve()
        return

    if args.preset:
        calibrate(verbose=True)
        print()
        rows = run_preset(args.batch, key=args.preset)
        if args.emit_flags:
            print()
            for name, cfg, _, _ in preset_cfgs():
                r = project(cfg, args.batch)
                if cfg.get('arch', 'resnet') != 'resnet':
                    # 结构不由 CLI flag 决定：train_sft 只认 KATAGO_SE_CFG，
                    # 旧 --backbone-* flag 已归档（接受但不参与建网）。发旧
                    # flag 等于发一套会被静默忽略的假旋钮，故明说「无 flag」。
                    print('--- {} ({:.0f} samples/s) ---'.format(
                        name, r['samples_per_s']))
                    print('# 无 flag 可发：结构由 scripts/train_sft.py 的 '
                          'KATAGO_SE_CFG 决定（旧 --backbone-* 等结构 flag '
                          '已归档，不参与建网）')
                    continue
                if not r['fits']:
                    continue
                print('--- {} ({} samples/s) ---'.format(name, r['samples_per_s']))
                print(emit_flags(cfg))
        return

    if args.grid:
        calibrate(verbose=True)
        print()
        run_grid(args.batch)
        return

    if args.sweep:
        calibrate(verbose=True)
        print()
        base = ANCHOR['cfg'] if args.base == 'v18' else V12_CFG
        extra = {}
        if args.mix:
            extra = dict(attention_mode='mix', num_attention_layers=base.get(
                'num_attention_layers', 4), res_blocks=0, convnext_blocks=0,
                attn_blocks=0)
        run_sweep(args.sweep, base=base, batch=args.batch, extra=extra)


def emit_flags(cfg):
    lines = ['--board-size 19 --backbone-channels {} '
             '--backbone-res-blocks {}'.format(
                 cfg['backbone_channels'], cfg['backbone_res_blocks'])]
    if cfg.get('res_blocks', 0) or cfg.get('convnext_blocks', 0) or \
            cfg.get('attn_blocks', 0):
        lines.append('--res-blocks {} --convnext-blocks {} --attn-blocks {}'.format(
            cfg.get('res_blocks', 0), cfg.get('convnext_blocks', 0),
            cfg.get('attn_blocks', 0)))
    else:
        lines.append('--attention-mode mix --num-attention-layers {}'.format(
            cfg.get('num_attention_layers', 4)))
    lines.append('--attn-mode {} --attn-window {} --num-heads {}'.format(
        cfg.get('attn_mode', 'window_global'), cfg.get('attn_window', 5),
        cfg.get('num_heads', 4)))
    lines.append('--value-channels {} --value-res-blocks {}'.format(
        cfg['value_channels'], cfg['value_res_blocks']))
    lines.append('--policy-channels {} --policy-layers {}'.format(
        cfg['policy_channels'], cfg['policy_layers']))
    return ' \\\n  '.join(lines)


if __name__ == '__main__':
    main()
