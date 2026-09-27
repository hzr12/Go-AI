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
                 只保留块输入，value head 未开则全量保留）

用法
----
    python scripts/search_arch.py --list
    python scripts/search_arch.py --preset            # 既有整表（= --preset all）
    python scripts/search_arch.py --preset v12        # 既有预设单项
    python scripts/search_arch.py --preset v21        # ANCHOR_V21 静态参数锚点
    python scripts/search_arch.py --grid
    python scripts/search_arch.py --sweep attn_window --sweep value_res_blocks

局限
----
**准确率无法在此测量**（需真实数据 + NPU 数小时）。本工具只覆盖
速度与显存两个维度；准确率需实跑 A/B。若某配置的架构在历史上有
实测记录（如 V12 的 50% top1），会在结果里标注作为参考。
"""

import argparse
import itertools
import os
import sys

import torch
import torch.nn as nn

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

# v21 权威参数锚点（用户给定数值，逐字面写死）。
#
# 为什么必须是独立字面量常量：本锚点供 test_v21_budget（P4.2）与 C11 的
# v21 显存投影做**仲裁**。若它从 src/networks/** 实测值推导，两个测试就只是
# 互相印证，锚点失去意义。tests/test_search_arch.py::test_anchor_v21_is_
# independent_of_network_code 用 AST 断言本赋值只含字面量（无 Name/Call/
# Attribute/Subscript），钉死这条独立性。
#
# 版本：P4.9b 跟随 P4.1 详细结构表更新数值与键名。块13-14 408,664→448,960
# （每块 224,480）、块15-16 613,272→652,832（每块 326,416）、
# backbone_total 6,419,944→6,499,800、total_params 8,928,691→9,008,419，
# FFN 中间维 276→240（ratio 1.5→1.304）。**旧值与 light_attn_* 键名作废。**
#
# 版本：P4.9c 更正 policy 头：2,366,602→2,366,730、total_params
# 9,008,419→9,008,547。P4.9b 的裁定是「以分项实算为准」，但那组分项里**有
# 一行本身就是笔误**（见下 (b)）——用一个含笔误的分项和去否定表头，等于让
# 规则自己失效。故本轮不再看「哪边是表头」，只看「哪边能被形状的精确算术
# 唯一确定」（即下方 arbitration_rule）。
#
# 键名对齐 P4.1 的类名：MambaLTI / TransformerBlock / CrossAttnRes
# （块13-14 旧称 LightAttn，用户本轮改称「Transformer 块」，旧称作废）。
#
# 仲裁规则（唯一一条；权威表内的每处冲突都按它判定，见 arbitration_rule）：
#   **形状被唯一钉死时，取该形状的精确算术值；与它冲突的表记数字即笔误。
#     形状本身自相矛盾时，精确算术无法裁决，两口径都留着待用户裁决，
#     实施暂取与已发布实现一致的那个。**
# 关键在于「可唯一确定」这个前提——它把两类表面同型的证据（一个数字 vs 另一个
# 数字）分开：(b) 里表把 fc2 的形状与 bias 都写死了，精确算术**有**唯一解，
# 规则给出结论；(a) 里表头文字与它自己给的参数量指向**不同**形状，精确算术
# 压根选不出谁对，规则只能升级为「待裁决」。P4.9b 对这两处用了两套相反的标
# 准（(a) 弃表取实算、(b) 取实算弃表头），本轮统一为这一条。
#
# 权威表内有两处笔误/冲突（见 P4.1 brief §3）。已按上述规则处理，**并**把
# 两个口径都显式记在此处与 known_discrepancies——不可静默抹掉：
#   (a) stem：表头文字写「7×7 Conv 17→184」，同给的参数量 28,152 却是 3×3
#       的值（3×3: 9×17×184 = 28,152；7×7: 49×17×184 = 153,272）。
#       **冲突在形状本身**，28,152 与 153,272 在各自形状下都精确 ⇒ 规则不能
#       裁决。28,152+368(BN) = 28,520 正好是现有 3×3 stem+BN，故实施暂取
#       3×3。⚠ 硬依赖：若最终确认改 7×7，stem 合计变 153,640
#       （+125,120），本表**整表作废**，backbone_total/total_params 必须重算。
#       （表/裁定里把 7×7 写成 153,296 / stem 153,664，比精确值多 24，是同
#       一处表内笔误；两个口径都记在 known_discrepancies 里待裁决。）
#   (b) policy 头：笔误在**分项**那一行，不在合计。分项写「FC 128→256
#       带bias：32,896」，而 nn.Linear(128, 256) 的 bias 恒为 out_features =
#       256 个 ⇒ 精确值 32,768+256 = **33,024**；bias=False 是 32,768。
#       **没有任何 bias 设置给出 32,896**（它 = 32,768+128，即把 bias 误记成
#       in_features；32,896 唯一能落地的 Linear 读法是转置的 Linear(256,128)，
#       与层链矛盾——fc1 出 128，fc2 的入维是 128 而非 256）。形状被钉死 ⇒
#       规则裁决 33,024。修正后分项之和 17,856+4,704+2,218,112+33,024+
#       93,034 = **2,366,730**，与表头自己写的合计**一致**——P4.9b 声明
#       「表头错了」是误判，真相是分项错、表头对。本锚点取 2,366,730，
#       total_params = 6,499,800+2,366,730+142,017 = 9,008,547，与已发布的
#       FCPolicyHead 实测（tests/test_arch_v21_blocks.py:64）重新一致。
# 两条同时以结构化字面量记在 known_discrepancies，测试逐字段断言其存在。
ANCHOR_V21 = {
    'name': 'v21',
    'in_channels': 17,               # 特征平面通道（P4.3 补齐后）
    'backbone_channels': 184,
    # 冲突取值标准（上方注释的长版规则）。P4.2 仲裁预算时**只按这一条**走，
    # 不得对同型证据换个标准——测试逐字断言其存在。
    'arbitration_rule': '形状被唯一钉死时取该形状的精确算术值（冲突的表记数字'
                        '=笔误）；形状本身矛盾时精确算术不裁决，两口径都留着，'
                        '实施暂取与已发布实现一致者',
    # 块布局：哪些块、几个。键名 = P4.1 的类名，勿再改回旧称
    'layout': {
        'stem': 'Conv3x3(17->184)+BN',   # 按 3×3 记 28,152，见上 (a)
        'res_blocks': 8,             # ResBlock ×8 @184
        'mamba_lti_blocks': 4,       # MambaLTI ×4 @184
        'transformer_blocks': 2,     # TransformerBlock ×2 @184
        'transformer_heads': 4,
        'ffn_hidden': 240,           # FFN 184→240→184，块13-14/15-16 共用
        'ffn_ratio': 1.304,          # 240/184 ≈ 1.304（旧表 1.5 作废）
        'cross_attn_res_blocks': 2,  # CrossAttnRes ×2 @184
        'cross_attn_concat_blocks': (1, 5, 9),
        'out': '1x1 Conv+BN',
    },
    # 各部分参数量（权威给定，勿由网络代码反算）
    'params': {
        'stem': 28_520,
        'res_blocks': 4_881_152,     # 8 × 610,144
        'mamba_lti': 454_112,        # 4 × 113,528
        'transformer': 448_960,      # 2 × 224,480
        'cross_attn_res': 652_832,   # 2 × 326,416
        'out': 34_224,
    },
    'backbone_total': 6_499_800,     # 主干合计（= params 六项之和，测试断言自洽）
    # 头：权威表只给了参数量；维度钉 in/out 两端（隐藏维未给定，不臆造）
    'heads': {
        'in': 184,                   # 头喂入 = 主干输出通道
        # policy 头 = 表头自己写的合计 2,366,730。分项「FC 128→256 带bias：
        # 32,896」是笔误（应为 128*256+256=33,024），见上 (b)。
        'policy_params': 2_366_730,
        'policy_out': 362,           # 19×19 + pass
        'value_params': 142_017,     # value 头（新增部分）
        'value_out': 1,
    },
    'total_params': 9_008_547,       # 主干 + 两头（测试断言自洽）
    # 上文 (a)(b) 两条矛盾的机器可读记录。id/about/stated/recorded/delta/
    # resolution/status 缺一不可：任一字段被静默改动/删除都会被
    # test_anchor_v21_documents_known_discrepancies 抓住。resolution 取
    # arbitration_rule 的两个分支之一，测试按同一把尺子逐条核。
    'known_discrepancies': (
        {
            'id': 'stem_kernel_3x3_vs_7x7',
            'about': 'stem 卷积核：表头文字 7×7，给的参数量却是 3×3 的值',
            'stated': 153_296,        # 7×7：表/裁定里写的口径
            'stated_exact': 153_272,  # 7×7：49×17×184 精确值（表记多 24，同属表内笔误）
            'recorded': 28_152,       # 3×3：9×17×184（本锚点采用）
            'delta': 125_144,        # 153,296 - 28,152（按表记口径；精确口径 125,120）
            # 冲突在**形状**上：28,152 与 153,272 在各自形状下都精确 ⇒ 规则不裁决
            'resolution': 'unresolved_shape_conflict',
            'status': '未裁决：按 3×3 实施；若改 7×7 则 stem 合计 153,640'
                      '（精确口径）/153,664（表记口径），整表作废需重算',
        },
        {
            'id': 'policy_fc2_128_off',
            'about': 'policy 头分项「FC 128→256 带bias：32,896」是笔误：'
                     'bias 被记成了 in_features，表头合计 2,366,730 反而是对的',
            'stated': 32_896,         # 表的分项行原样写的值（32,768+128）
            'recorded': 33_024,       # 128×256+256：同一形状的唯一精确值（本锚点采用）
            'delta': 128,             # 33,024 - 32,896（recorded - stated）
            'total_stated': 2_366_730,  # 表头自己写的合计：与修正后的分项和一致
            # 形状（含 bias 标志）被表写死 ⇒ 精确算术有唯一解 ⇒ 规则裁决 recorded
            'resolution': 'resolved_by_exact_arithmetic',
            'status': '已裁定（P4.9c）：分项行改按 33,024，合计 2,366,730 与表头'
                      '一致，total_params 9,008,547；与已发布 FCPolicyHead 实测一致',
        },
    ),
}


def measure(cfg, probe_bs=PROBE_BS, use_checkpoint=True, in_channels=12):
    """返回 (参数量, FLOPs@batch1, 保留激活字节@probe_bs)。

    in_channels 默认 12 —— 既有调用方零回归；17ch（v21）显式传入即可测。
    """
    model = AlphaGoNet(in_channels=in_channels, action_size=ACTION_SIZE,
                       arch='resnet', attention_dropout=0.0,
                       use_checkpoint=use_checkpoint, **cfg)
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
    """给出某配置在目标 batch 下的预测指标（in_channels 默认 12，零回归）。"""
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
    ]
    return out


# --preset 的取值形态：all=既有整表（裸 --preset / 无参数时的默认），
# v18/v12/b/f=既有预设单项，v21=ANCHOR_V21 静态锚点。
PRESET_CHOICES = ('all', 'v18', 'v12', 'b', 'f', 'v21')
_PRESET_INDEX = {'v18': 0, 'v12': 1, 'b': 2, 'f': 3}   # 对应 preset_cfgs() 下标


def preset_entries(key='all'):
    """把 --preset 的取值映射到 (name, cfg, bs, note) 列表（纯选择，不测量）。

    'v21' 返回空列表：v21 结构 AlphaGoNet 造不出来，走 run_anchor_v21()
    打印静态锚点表，不做实测投影。
    """
    if key == 'all':
        return preset_cfgs()
    if key == 'v21':
        return []
    return [preset_cfgs()[_PRESET_INDEX[key]]]


def preset_in_channels(key):
    """预设的输入通道数：**来源 = 预设推导**（非 CLI 自由参数）。

    通道数是被测结构的固有属性，不是可随意拨动的旋钮：若开 `--in-channels`
    这类显式旗标，12ch 训练的 cfg 配 17ch 输入能照跑不误，产出一张看似正常、
    实际错位的对比表——而对比表正是本工具的全部价值。故每个预设自带通道数
    （既有预设恒 12 = 改前行为），v21 的值直接取 ANCHOR_V21['in_channels']，
    单一事实源，不在别处再写一遍 17。
    """
    return ANCHOR_V21['in_channels'] if key == 'v21' else 12


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


def run_anchor_v21():
    """陈列 v21 权威参数锚点（ANCHOR_V21，静态字面量，非本工具实测）。

    P4.2 接线落地前 AlphaGoNet 还造不出 MambaLTI/TransformerBlock/
    CrossAttnRes 结构，故 `--preset v21` 不做 forward 投影，只打印锚点表，
    并当场核算主干合计与总合计——表内数字自相矛盾会直接暴露在输出里。
    表内那两处**已知**笔误（stem 3×3 vs 7×7 的形状冲突、policy 分项 fc2 的
    32,896）不在此判对错，只按 known_discrepancies 逐条陈列，避免看起来像
    被静默抹平。
    """
    A = ANCHOR_V21
    lay, vh = A['layout'], A['heads']
    ffn = 'ffn={}(ratio {})'.format(lay['ffn_hidden'], lay['ffn_ratio'])
    # label[k] 会让 params 每加一个键就 KeyError。这张表要长，故缺键退回键名
    label = {
        'stem': 'stem {}'.format(lay['stem']),
        'res_blocks': 'ResBlock ×{} @{}'.format(
            lay['res_blocks'], A['backbone_channels']),
        'mamba_lti': 'MambaLTI ×{} @{}'.format(
            lay['mamba_lti_blocks'], A['backbone_channels']),
        'transformer': 'TransformerBlock ×{} @{} heads={} {}'.format(
            lay['transformer_blocks'], A['backbone_channels'],
            lay['transformer_heads'], ffn),
        'cross_attn_res': 'CrossAttnRes ×{} @{} concat={} {}'.format(
            lay['cross_attn_res_blocks'], A['backbone_channels'],
            '/'.join(str(i) for i in lay['cross_attn_concat_blocks']), ffn),
        'out': 'out {}'.format(lay['out']),
    }
    print('v21 参数锚点（ANCHOR_V21，用户权威给定；**非**本工具实测）')
    print('仲裁规则（冲突取值标准）：{}'.format(A['arbitration_rule']))
    print(SEP)
    total = 0
    for k, v in A['params'].items():
        print('{:<52}{:>12,}'.format(label.get(k, k), v))
        total += v
    print('{:<52}{:>12,}  {}'.format(
        '主干合计', total,
        'OK' if total == A['backbone_total'] else 'MISMATCH!'))
    print('{:<52}{:>12,}'.format('policy 头（新增部分）', vh['policy_params']))
    print('{:<52}{:>12,}'.format('value 头（新增部分）', vh['value_params']))
    grand = total + vh['policy_params'] + vh['value_params']
    print('{:<52}{:>12,}  {}'.format(
        '总参数', grand, 'OK' if grand == A['total_params'] else 'MISMATCH!'))
    print('in={} → backbone {}ch → 头 in={}，policy out={}，value out={}'.format(
        A['in_channels'], A['backbone_channels'], vh['in'],
        vh['policy_out'], vh['value_out']))
    for d in A['known_discrepancies']:
        print('⚠ 已知矛盾[{}]：{}（表记 {:,} / 本锚点采 {:,}，差 {:,}）{}'.format(
            d['id'], d['about'], d['stated'], d['recorded'], d['delta'],
            d['status']))
    print('注：v21 结构在 P4.2 接线落地前不可实测；17ch measure 已支持'
          '（measure/in_channels=17），--emit-flags 需 V21_CFG，暂不支持')
    return dict(parts=dict(A['params']), backbone_total=total,
                grand_total=grand)


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
                    help='跑预设配置：all=既有整表（裸 --preset 或无参数的默认），'
                         'v18/v12/b/f=既有预设单项，v21=ANCHOR_V21 静态锚点')
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
        print('  {:<30} batch={:<5} {}'.format(
            'v21（ANCHOR_V21 锚点 17ch）', '-',
            '静态参数锚点，P4.2 前不可实测'))
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

    if args.preset == 'v21':
        # 静态锚点：不 calibrate、不 forward（v21 结构 P4.2 前造不出来）
        run_anchor_v21()
        return

    if args.preset:
        calibrate(verbose=True)
        print()
        rows = run_preset(args.batch, key=args.preset)
        if args.emit_flags:
            print()
            for name, cfg, _, _ in preset_cfgs():
                r = project(cfg, args.batch)
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
