"""search_arch.py 的测量正确性测试。

三项测量都必须可信，否则整个搜索无意义：
  * 参数量  —— 直接求和，最容易错的是漏算/重算
  * FLOPs   —— 必须按 batch=1 测，否则被 batch 放大
  * 激活显存 —— 必须按 **storage 去重**，且要正确反映 gradient checkpointing

尤其第三项：曾用 `t.numel() * element_size()` 累加，实测高估 48%
（反向保存大量视图，共享同一块 storage，被重复计数）；改按
`untyped_storage().nbytes()` + `data_ptr()` 去重后误差降到 21%，剩余
部分来自 checkpoint 重算临时量被按「同时存活」计入。

P4.9 另覆盖三件事：`--preset` 取值修型（store_true -> choices，含 v21）、
measure 的 17 通道支持（默认 12 零回归）、ANCHOR_V21 锚点的数值与独立性。

P4.9c 更正 ANCHOR_V21 的 policy 头：2,366,602 → **2,366,730**
（total_params 9,008,419 → **9,008,547**）。原值来自权威表 policy 分项行，
而该行的「FC 128→256 带bias：32,896」是笔误（bias 被记成 in_features）。
本文件随之把该条矛盾的 recorded 换成 32,896→33,024、补上一条**单一**仲裁规则
（按其两个分支逐条核），并给 print 格式补上此前完全缺失的测试。
"""

import ast
import inspect
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from src.networks.alphanet import AlphaGoNet  # noqa: E402
import search_arch as S  # noqa: E402

SMALL = dict(backbone_channels=32, backbone_res_blocks=2,
             attention_mode='none', num_attention_layers=0, num_heads=4,
             attn_mode='window_global', attn_window=5,
             res_blocks=1, convnext_blocks=1, attn_blocks=1,
             value_channels=16, value_res_blocks=2,
             policy_channels=16, policy_layers=2)


def _model(**over):
    cfg = dict(SMALL)
    cfg.update(over)
    return AlphaGoNet(in_channels=12, action_size=362, arch='resnet',
                      attention_dropout=0.0, **cfg)


def test_param_count_matches_model():
    n, _, _ = S.measure(SMALL, probe_bs=2)
    m = _model()
    assert n == sum(p.numel() for p in m.parameters())
    del m


def test_flops_are_batch_independent():
    """FLOPs 必须按 batch=1 测——否则乘上 batch，候选比较全错。"""
    n0, f1, _ = S.measure(SMALL, probe_bs=2)
    n1, f2, _ = S.measure(SMALL, probe_bs=8)
    assert f1 == f2, 'FLOPs 随 probe_bs 变化，说明被 batch 放大了'
    assert n0 == n1


def test_activation_is_deduplicated_by_storage():
    """反向会保存共享同一 storage 的视图，必须去重（否则高估 48%）。

    构造两个共享底层存储的视图，各自存进反向图，计量只应算一份。
    """
    counted = [0]
    seen = set()

    def pack(t):
        st = t.untyped_storage()
        k = st.data_ptr()
        if k not in seen:
            seen.add(k)
            counted[0] += st.nbytes()
        return t
    base = torch.zeros(64, 64, requires_grad=True)
    v1 = base[:32]
    v2 = base[32:]
    # 前向必须在 hook 上下文**内**：mul 是在前向时保存操作数的。
    # 用 sum 也不行——sum 不保存任何输入。
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = (v1 * v2).sum() * 2
        y.backward()
    # v1 与 v2 是 base 的两个视图，共享同一 storage -> 只应计一份 64*64*4
    assert counted[0] == 64 * 64 * 4, \
        '共享 storage 的视图被重复计数: {} != {}'.format(counted[0], 64 * 64 * 4)


def test_checkpointing_reduces_retained_memory():
    """开 gradient checkpointing 后保留的激活应显著变少。

    这正是 v18 实测能跑在 97% 显存的原因；若此关系不成立，说明计量有 bug。
    """
    _, _, ret_plain = S.measure(SMALL, probe_bs=2, use_checkpoint=False)
    _, _, ret_ckpt = S.measure(SMALL, probe_bs=2, use_checkpoint=True)
    assert ret_ckpt < ret_plain, \
        '开启 checkpoint 后保留显存未减少（{} vs {}），计量或实现有问题'.format(
            ret_ckpt, ret_plain)


def test_attention_window_affects_memory():
    """attn_window 会改变窗口划分数，显存必须随之变化。

    注意 FLOPs 对它不敏感——注意力是 matmul 不是 conv/linear，hook 数不到。
    这是本工具的已知盲区，测试只锁住显存这一侧确实敏感。
    """
    mems = {}
    for ws in (3, 5, 19):
        _, _, ret = S.measure(dict(SMALL, attn_window=ws), probe_bs=2)
        mems[ws] = ret
    assert mems[3] != mems[19], 'attn_window 对显存毫无影响，疑似计量失灵'
    assert mems[3] > mems[5] > mems[19] or mems[3] > mems[5], \
        '小窗口（nW 大）应占用更多显存，实测 {}'.format(mems)


def test_backbone_res_blocks_is_a_dead_parameter():
    """backbone_res_blocks 在分段模式下完全无效（已知死参数）。

    backbone.py:613-617 把它算成 total_blocks 传给 _build_segmented_blocks，
    但该函数体只用 res_count/convnext_count/attn_count，total_blocks 从不使用。
    本测试把这个事实钉住，避免哪天有人以为调它有用。
    """
    base = dict(SMALL, attention_mode='none', res_blocks=2,
                convnext_blocks=1, attn_blocks=1)
    outs = []
    for nb in (3, 8, 20):
        n, f, ret = S.measure(dict(base, backbone_res_blocks=nb), probe_bs=2)
        outs.append((n, f, ret))
    assert outs[0] == outs[1] == outs[2], \
        'backbone_res_blocks 竟然影响了输出，说明死参数已被修复或配置不对: {}'.format(outs)


def test_num_attention_layers_ignored_in_segmented_mode():
    """分段模式下 num_attention_layers 同样被忽略（由 attn_blocks 取代）。"""
    base = dict(SMALL, attention_mode='none', res_blocks=2,
                convnext_blocks=1, attn_blocks=1)
    a = S.measure(dict(base, num_attention_layers=2), probe_bs=2)
    b = S.measure(dict(base, num_attention_layers=8), probe_bs=2)
    assert a == b, '分段模式下 num_attention_layers 不应影响结果'


def test_overhead_accounting():
    """常驻开销 = 权重+梯度+AdamW双矩+EMA，参数为 FP32 故 20 B/元素。"""
    assert abs(S.overhead_gb(1_000_000) - 20e6 / 1024 ** 3) < 1e-9


def test_calibration_is_self_consistent():
    """标定后，锚点配置的预测显存应等于实测值。"""
    err = S.calibrate(verbose=False)
    assert abs(err) < 1e-6, '标定不自洽，误差 {}'.format(err)
    k = S.calib_factor()
    assert 0.7 < k < 0.95, \
        '标定因子应在 0.7~0.95 之间（checkpoint 高估 ~21%），实为 {}'.format(k)


def test_max_batch_respects_budget():
    """max_batch 返回的 batch 必须真的装得下，且再大一点就装不下。"""
    cfg = dict(S.ANCHOR['cfg'], backbone_channels=128, value_res_blocks=3)
    b = S.max_batch(cfg)
    assert S.project(cfg, b)['fits']
    if b < 4200:
        assert not S.project(cfg, b + 40)['fits'], \
            'batch={} 之后仍有大把余量，二分搜索没找准'.format(b)


# ---------------------------------------------------------------------------
# P4.9：--preset 修型（store_true -> 取值）/ measure 17ch / ANCHOR_V21
# ---------------------------------------------------------------------------

# 既有 4 个预设在**改前**的实测金标（params/flops 均为整数，机器无关；
# 改前用 scripts/search_arch.py 逐一 project 采集）。任何使既有预设行为
# 漂移的改动都会撞上它。
GOLDEN_PRESETS = [
    ('v18 原样（锚点）', 12_858_978, 8_489_705_920),
    ('V12 原样', 12_652_834, 8_987_335_872),
    ('B: ConvNeXt4→Res4', 14_315_874, 9_540_475_840),
    ('F: V12块序+v18头', 14_056_866, 9_354_165_184),
]


def test_preset_accepts_v21(capsys):
    """`--preset v21` 必须能被解析并路由到 v21 分支。

    改前红：`--preset` 是 store_true，`--preset v21` 会把 `v21` 当成
    unrecognized argument 直接 SystemExit(2)。
    """
    args = S.build_parser().parse_args(['--preset', 'v21'])
    assert args.preset == 'v21'
    S.main(['--preset', 'v21'])
    out = capsys.readouterr().out
    assert 'ANCHOR_V21' in out, 'v21 分支没有陈列 ANCHOR_V21'
    assert '6,499,800' in out and '9,008,547' in out, \
        'v21 分支没有输出合计，自洽核算缺失'
    # v21 走静态锚点，不得触发 v18 校准/实测路径（P4.2 前造不出该结构）
    assert '校准核对' not in out


def test_preset_choices_contain_legacy_values(capsys):
    """既有预设仍在 choices 里、旧命令行仍可用，且行为与改前逐项一致。"""
    p = S.build_parser()
    act = next(a for a in p._actions if a.dest == 'preset')
    assert set(S.PRESET_CHOICES) == {'all', 'v18', 'v12', 'b', 'f', 'v21'}
    for k in S.PRESET_CHOICES:
        assert k in act.choices, '{} 不在 --preset choices 里'.format(k)
        assert p.parse_args(['--preset', k]).preset == k
    # 旧命令行 1：裸 --preset（store_true 时代的唯一形态）仍是整表
    assert p.parse_args(['--preset']).preset == 'all'
    # 旧命令行 2：完全无参数，由 main 兜底成整表
    assert p.parse_args([]).preset is None

    # 既有预设定义与改前逐项一致（golden 字面量，独立于 S.preset_cfgs 实现）
    cfgA = dict(backbone_channels=192, backbone_res_blocks=17,
                attention_mode='none', num_attention_layers=0, num_heads=4,
                attn_mode='window_global', attn_window=5,
                res_blocks=8, convnext_blocks=4, attn_blocks=5,
                value_channels=96, value_res_blocks=8,
                policy_channels=128, policy_layers=3)
    cfgV = dict(backbone_channels=192, backbone_res_blocks=17,
                attention_mode='mix', num_attention_layers=4, num_heads=4,
                attn_mode='window_global', attn_window=5,
                res_blocks=0, convnext_blocks=0, attn_blocks=0,
                value_channels=64, value_res_blocks=2,
                policy_channels=32, policy_layers=2)
    golden_cfgs = [
        ('v18 原样（锚点）', cfgA, 2800, '当前基线，显存超限'),
        ('V12 原样', cfgV, 2800, '历史 top1≈50% 的结构'),
        ('B: ConvNeXt4→Res4', {**cfgA, 'res_blocks': 12, 'convnext_blocks': 0},
         2800, 'FLOPs 更高但显存更低'),
        ('F: V12块序+v18头', {**cfgV, 'value_channels': 96,
                             'value_res_blocks': 8, 'policy_channels': 128,
                             'policy_layers': 3}, 2800, ''),
    ]
    assert S.preset_cfgs() == golden_cfgs, '既有预设定义被改动（与改前不一致）'

    # 选择映射：整表 == 既有清单；单项 == 对应下标；v21 不进实测表
    assert S.preset_entries('all') == S.preset_cfgs()
    for i, k in enumerate(('v18', 'v12', 'b', 'f')):
        assert S.preset_entries(k) == [golden_cfgs[i]], \
            '{} 选错了预设'.format(k)
    assert S.preset_entries('v21') == []

    # 各跑一次，结果与改前实测金标一致（params/flops 为整数）
    for (gname, gparams, gflops), k in zip(
            GOLDEN_PRESETS, ('v18', 'v12', 'b', 'f')):
        (name, r), = S.run_preset(key=k)
        assert name == gname
        assert r['params'] == gparams, \
            '{} params {} != 改前 {}'.format(k, r['params'], gparams)
        assert r['flops'] == gflops, \
            '{} flops {} != 改前 {}'.format(k, r['flops'], gflops)

    # 通道数来源 = 预设推导：既有预设恒 12（= 改前），v21 取自 ANCHOR_V21
    for k in ('all', 'v18', 'v12', 'b', 'f'):
        assert S.preset_in_channels(k) == 12
    assert S.preset_in_channels('v21') == S.ANCHOR_V21['in_channels'] == 17

    # --list：既有 4 行逐字节不变，v21 仅为新增行
    S.main(['--list'])
    out = capsys.readouterr().out
    for gname, _, gbs, gnote in golden_cfgs:
        line = '  {:<30} batch={:<5} {}'.format(gname, gbs, gnote)
        assert line in out, '--list 既有行丢失/改动: {!r}'.format(line)
    assert 'ANCHOR_V21' in out


def test_measure_supports_17_channels():
    """measure 必须能测 17 通道（改前红：in_channels 硬编码 12，kwarg 报错）。"""
    n17, f17, r17 = S.measure(SMALL, probe_bs=2, in_channels=17)
    n12, f12, r12 = S.measure(SMALL, probe_bs=2)
    # 与输入通道相关的只有 stem 卷积：Conv3×3(12→32) vs (17→32)，BN 不变
    assert n17 - n12 == (17 - 12) * SMALL['backbone_channels'] * 3 * 3, \
        '17ch 参数增量不等于 stem 卷积增量'
    # FLOPs 增量 = 多出的输入通道在 stem 上的 2·ic·oc·k·H·W
    assert f17 - f12 == (17 - 12) * SMALL['backbone_channels'] * 9 * 19 * 19 * 2
    assert r17 > 0 and r12 > 0
    # 17ch 实测与直构模型一致（钉死 measure 内部没有第二处写死 12）
    m17 = AlphaGoNet(in_channels=17, action_size=362, arch='resnet',
                     attention_dropout=0.0, **SMALL)
    assert n17 == sum(p.numel() for p in m17.parameters())
    del m17


def test_measure_default_unchanged():
    """零回归：默认 in_channels=12，与显式 12 逐项相等；老调用形态可用。"""
    assert inspect.signature(S.measure).parameters['in_channels'].default == 12
    assert inspect.signature(S.project).parameters['in_channels'].default == 12
    a = S.measure(SMALL, probe_bs=2)
    b = S.measure(SMALL, probe_bs=2, in_channels=12)
    assert a == b, '默认通道数与显式 12 结果不一致（零回归被破坏）'
    # 既有调用方（test_run_py_sh / test_v19_budget）的形态：不传 in_channels
    r1 = S.project(dict(SMALL), 2800)
    r2 = S.project(dict(SMALL), 2800, in_channels=12)
    assert r1 == r2


def _anchor_source_with_comments():
    """返回 ANCHOR_V21 赋值 + 其上方紧邻注释块的源码文本。

    数值能被结构化字段钉住，「为什么这么取」的依赖说明只能活在注释里，
    故单独取出来做断言——否则日后有人把注记删了，数字仍绿，理由却没了。
    """
    path = os.path.join(ROOT, 'scripts', 'search_arch.py')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)
    node = next(stmt.value for stmt in tree.body
                if isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'ANCHOR_V21'
                        for t in stmt.targets))
    lines = src.splitlines()
    i = node.lineno - 2                      # 赋值行的 0-based 前一行
    while i >= 0 and lines[i].lstrip().startswith('#'):
        i -= 1
    return '\n'.join(lines[i + 1:node.end_lineno])


def test_anchor_v21_constants():
    """ANCHOR_V21 逐项等于 P4.1 详细结构表（权威），且合计自洽。

    ⚠ P4.9c：policy 头由 2,366,602 更正为 **2,366,730**（表头自己的数字），
    total_params 相应 9,008,419→**9,008,547**。原值 2,366,602 来自权威表
    的 policy 分项行，而那一行里「FC 128→256 带bias：32,896」是笔误——bias
    被记成 in_features。下面的 policy_lines 断言是本测试的关键：把表自己的
    5 行分项（fc2 换成精确值 33,024）加总，必须正好等于 heads.policy_params。
    分项与合计对不上时，错的几乎总是分项。
    """
    A = S.ANCHOR_V21
    assert A['name'] == 'v21'
    assert A['in_channels'] == 17
    assert A['backbone_channels'] == 184
    lay = A['layout']
    assert lay['stem'] == 'Conv3x3(17->184)+BN'
    assert lay['res_blocks'] == 8
    assert lay['mamba_lti_blocks'] == 4
    assert lay['transformer_blocks'] == 2
    assert lay['transformer_heads'] == 4
    assert lay['ffn_hidden'] == 240, 'FFN 中间维必须是 240（旧表 276 作废）'
    assert abs(lay['ffn_ratio'] - 240 / 184) < 1e-3, \
        'ffn_ratio 与 240/184 不符: {}'.format(lay['ffn_ratio'])
    assert lay['cross_attn_res_blocks'] == 2
    assert lay['cross_attn_concat_blocks'] == (1, 5, 9)
    assert lay['out'] == '1x1 Conv+BN'
    # 权威表逐项（P4.1 brief §2）
    parts = dict(stem=28_520, res_blocks=4_881_152, mamba_lti=454_112,
                 transformer=448_960, cross_attn_res=652_832, out=34_224)
    assert A['params'] == parts, '各部分参数量与权威表不符'
    # 每块单价也对得上（448,960 = 2×224,480；652,832 = 2×326,416）
    assert parts['transformer'] == 2 * 224_480
    assert parts['cross_attn_res'] == 2 * 326_416
    assert parts['res_blocks'] == 8 * 610_144
    assert parts['mamba_lti'] == 4 * 113_528
    # 合计自洽：六项之和 == backbone_total == 6,499,800（权威表）
    assert sum(parts.values()) == A['backbone_total'] == 6_499_800, \
        '主干合计与各部分之和不自洽'
    h = A['heads']
    assert h['in'] == 184
    assert h['policy_params'] == 2_366_730, \
        'policy 头 = 表头合计 2,366,730（分项 fc2 的 32,896 是笔误，见 §3(b)）'
    assert h['policy_out'] == 362
    assert h['value_params'] == 142_017
    assert h['value_out'] == 1
    assert A['total_params'] == 9_008_547
    assert A['backbone_total'] + h['policy_params'] + h['value_params'] \
        == A['total_params'], '总合计与 主干+两头 不自洽'
    # policy 头的分项复核（权威表 §2 policy 行的 5 个数，fc2 取精确值）：
    # 17,856(1×1Conv+BN) / 4,704 / 2,218,112 / 33,024 / 93,034
    assert 17_856 == 184 * 96 + 2 * 96, '1×1 Conv 184→96 无 bias + BN(96)'
    assert 4_704 == 96 * 48 + 2 * 48, '1×1 Conv 96→48 无 bias + BN(48)'
    assert 2_218_112 == 48 * 19 * 19 * 128 + 128, 'FC 17328→128 带 bias'
    assert 33_024 == 128 * 256 + 256, 'FC 128→256 带 bias：bias 恒为 out_features'
    assert 93_034 == 256 * 362 + 362, 'FC 256→362 带 bias'
    policy_lines = [17_856, 4_704, 2_218_112, 33_024, 93_034]
    assert sum(policy_lines) == h['policy_params'] == 2_366_730, \
        'policy 头与权威表分项之和不自洽: {} vs {}'.format(
            sum(policy_lines), h['policy_params'])
    # 被否掉的那行确实是「bias 记成 in_features」，且任何 bias 设置都给不出它
    assert 32_896 == 128 * 256 + 128
    assert 128 * 256 + 256 == 33_024 and 128 * 256 == 32_768
    assert 6_499_800 + 2_366_730 + 142_017 == 9_008_547


def test_anchor_v21_layout_names_match_p41_class_names(capsys):
    """布局/参数键名与 P4.1 的类名逐项对齐：旧称 LightAttn 必须绝迹。

    P4.1 brief §1 定的类名是 MambaLTI / TransformerBlock / CrossAttnRes
    （块13-14 旧称 LightAttn 作废）。键名一旦与类名脱节，锚点表就会与被实现
    的结构各说各话——而锚点正是 P4.2 预算仲裁的依据。
    """
    A = S.ANCHOR_V21
    lay = A['layout']

    def strings(obj):
        """递归取出对象里所有字符串（键与值都算）。"""
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from strings(k)
                yield from strings(v)
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                yield from strings(v)

    # 布局键 = P4.1 类名的 snake_case，三个注意力/序列块都在
    assert {'mamba_lti_blocks', 'transformer_blocks',
            'cross_attn_res_blocks'} <= set(lay), \
        '布局缺 P4.1 的三个块类键: {}'.format(sorted(lay))
    # 旧称 LightAttn / light_attn_* 全部消失（大小写不敏感）
    for s in list(strings(lay)) + list(lay) + list(A['params']):
        low = s.lower()
        assert 'lightattn' not in low and 'light_attn' not in low, \
            '布局/参数里还残留旧称 LightAttn: {!r}'.format(s)
    assert 'transformer' in A['params'] and 'light_attn' not in A['params']

    # 打印表用 P4.1 的类名（run_anchor_v21 的 label 跟着键名走）
    S.main(['--preset', 'v21'])
    out = capsys.readouterr().out
    assert 'TransformerBlock ×2' in out, '打印表没有用 TransformerBlock'
    assert 'LightAttn' not in out, '打印表仍出现旧称 LightAttn'
    assert 'MambaLTI ×4' in out and 'CrossAttnRes ×2' in out
    assert 'ffn=240' in out, '打印表没有陈列 FFN 中间维 240'


def test_anchor_v21_arbitration_rule_is_single_and_applied_to_both_entries():
    """**一条**仲裁规则，两处冲突都按它判——不许对同型证据换标准。

    规则（ANCHOR_V21['arbitration_rule']，注释里有长版）：形状被唯一钉死时取
    该形状的精确算术值（冲突的表记数字 = 笔误）；形状本身矛盾时精确算术不裁
    决，两口径都留着，实施暂取与已发布实现一致者。

    规则落到两条 entry 的 `resolution` 上，必须**互不相同**且各归其位：
      (a) stem：表头文字 7×7 与它自己给的参数量 3×3 指向不同形状 ⇒ 形状未钉死
          ⇒ unresolved_shape_conflict（不能裁决，只能记着等用户拍板）。
      (b) policy：fc2 的形状与 bias 都被表写死 ⇒ 精确算术有唯一解 ⇒
          resolved_by_exact_arithmetic（规则必须给出结论）。
    「两条 resolution 必须同时存在且分属两支」正是 P4.9b 缺的东西：它对 (a)
    弃表取实算、对 (b) 取实算弃表头，同型证据两套相反标准。
    """
    A = S.ANCHOR_V21
    rule = A['arbitration_rule']
    assert rule and isinstance(rule, str), '锚点必须带一条可读的仲裁规则'
    for branch in ('精确算术', '形状本身矛盾'):
        assert branch in rule, '仲裁规则缺分支：{}'.format(branch)
    d = {x['id']: x for x in A['known_discrepancies']}
    assert d['stem_kernel_3x3_vs_7x7']['resolution'] == 'unresolved_shape_conflict', \
        '(a) 的冲突在形状上，规则不该裁决它'
    assert d['policy_fc2_128_off']['resolution'] == 'resolved_by_exact_arithmetic', \
        '(b) 的形状被钉死，规则必须裁决它（而不是含糊其辞）'
    # 规则与结论方向一致：被裁决的 (b) 必须站「精确算术」这一侧
    pol = d['policy_fc2_128_off']
    assert pol['recorded'] == 128 * 256 + 256 > pol['stated'] == 128 * 256 + 128, \
        '被裁决的条目必须采纳精确算术值、否掉表记值'
    # 未裁决的 (a) 两侧都精确 ⇒ 谁也不比谁权威，status 必须仍是「未裁决」
    stem = d['stem_kernel_3x3_vs_7x7']
    assert stem['recorded'] == 3 * 3 * 17 * 184, '3×3 精确'
    assert stem['stated_exact'] == 7 * 7 * 17 * 184, '7×7 精确'
    assert stem['status'].startswith('未裁决'), \
        '形状未定的那条不得被写成已裁定（否则 P4.2 会当成定论读）'


def test_anchor_v21_documents_known_discrepancies():
    """权威表的两处笔误必须被记录在锚点里，不许静默抹掉。

    (a) stem 表头写 7×7，参数却是 3×3 的值（28,152 vs 153,296）。锚点按 3×3
        取值 —— 若日后确认改 7×7，stem 变 153,640，**整张预算表作废**。
    (b) policy 分项「FC 128→256 带bias」写 32,896，精确值 33,024（bias 恒为
        out_features=256，不是 in_features=128）。⚠ P4.9c 已把方向翻过来：
        笔误在**分项**，表头合计 2,366,730 才是对的；P4.9b 写的「以分项实算
        2,366,602 为权威」是误判，已作废。

    这两条一旦被「顺手改整齐」，P4.2 的预算窗口就会在无人察觉下偏移
    125,144 / 128 个参数。故既断言结构化字段，也断言锚点源码的注释里写了
    依赖关系（防止只留数字、不留理由）。
    """
    A = S.ANCHOR_V21
    d = {x['id']: x for x in A['known_discrepancies']}
    assert set(d) == {'stem_kernel_3x3_vs_7x7', 'policy_fc2_128_off'}, \
        'known_discrepancies 缺条目或多出条目: {}'.format(sorted(d))

    # (a) stem：记录值必须就是 params 里采用的那个，且两个口径的算式对得上
    stem = d['stem_kernel_3x3_vs_7x7']
    assert stem['recorded'] == 3 * 3 * 17 * 184 == 28_152
    # 7×7：表记 153,296，但 49×17×184 精确值是 153,272（表内笔误，多 24）。
    # 两个口径都必须留着——日后真改 7×7 的人只有靠这条才知道该用哪个。
    assert stem['stated'] == 153_296, '表记的 7×7 口径被改动'
    assert stem['stated_exact'] == 7 * 7 * 17 * 184 == 153_272
    assert stem['stated'] - stem['stated_exact'] == 24, '表内 24 的笔误未记录'
    assert stem['delta'] == stem['stated'] - stem['recorded'] == 125_144
    # delta 记的是**表记口径**（含那 24 的笔误），而上方注释里的 +125,120 是
    # 精确口径的增量。两者必须差恰好 24——否则注释与字段各说各话。
    assert stem['delta'] - 24 == 153_272 - 28_152 == 125_120, \
        'stem 的 delta 与「精确口径增量 125,120」对不上（表记多 24）'
    assert 28_152 + 368 == 28_520, '注释里的 3×3 stem+BN 合计'
    assert 153_272 + 368 == 153_640, '注释里的 7×7 stem+BN 合计（精确口径）'
    assert A['params']['stem'] == stem['recorded'] + 368, \
        'params.stem 应 = 3×3 卷积 + BN，锚点却没采用 recorded 口径'
    assert '3x3' in A['layout']['stem'].lower(), \
        '布局未标明 stem 按 3×3 记'
    assert '作废' in stem['status'] and '153,640' in stem['status'], \
        'stem 记录的 status 必须写明「改 7×7 则整表作废」这个依赖'

    # (b) policy：错的是分项行。stated/recorded 记录的是**同一形状**下的两个值
    pol = d['policy_fc2_128_off']
    assert pol['stated'] == 32_896 == 128 * 256 + 128, \
        '表记的分项行（bias 被误记成 in_features）'
    assert pol['recorded'] == 33_024 == 128 * 256 + 256, \
        '同一形状 Linear(128,256) 的唯一精确值'
    assert pol['delta'] == pol['recorded'] - pol['stated'] == 128
    # 表头合计不再是被否的一方，而是与修正后的分项和**互相印证**的那个
    assert pol['total_stated'] == 2_366_730 == A['heads']['policy_params']
    assert sum([17_856, 4_704, 2_218_112, pol['recorded'], 93_034]) \
        == pol['total_stated'], '修正后的分项和必须回到表头合计'
    assert A['total_params'] == A['backbone_total'] + pol['total_stated'] \
        + A['heads']['value_params'], 'total_params 必须按表头合计而非笔误分项'
    # 旧裁定（以 2,366,602 为权威）必须已被彻底换掉：那两个数一个都不许出现
    assert 2_366_602 not in (A['heads']['policy_params'], A['total_params'],
                             pol['stated'], pol['recorded'], pol['delta'])
    assert '已裁定' in pol['status'], \
        '(b) 已被形状的精确算术裁决，status 不该再写「未裁决」'

    # 理由也必须留在锚点的注释里（防止只剩数字、丢掉依赖说明）
    src = _anchor_source_with_comments()
    for marker in ('3×3', '7×7', '28,152', '153,296', '153,640', '153,272',
                   '2,366,730', '33,024', '32,896', 'in_features',
                   'arbitration_rule', 'known_discrepancies'):
        assert marker in src, '锚点注释里没有记下 {}'.format(marker)
    # 作废的旧裁定也必须留痕（否则日后有人从 git 里翻出来会以为它有效）
    assert '2,366,602' in src, '锚点注释未记录被作废的 2,366,602 旧裁定'


def test_anchor_v21_is_independent_of_network_code():
    """ANCHOR_V21 必须是独立字面量常量，不得由 src/networks/** 推导。

    若锚点由被测代码算出，test_v21_budget（P4.2）与本锚点就只是互相
    印证——两边同时错也照样绿。故对 ANCHOR_V21 的赋值表达式做 AST 断言：
    只允许字面量节点（Dict/Tuple/List/Constant），连 Name/Call/Attribute/
    Subscript 都不许有。这比 brief 里「模块没有 src.networks import」的
    字面读法**更强**且可行——本模块必须合法地 import AlphaGoNet 供
    measure 用，但锚点一个名字都不许引用，更不许调用模型构造。
    """
    path = os.path.join(ROOT, 'scripts', 'search_arch.py')
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    # ① 全模块**恰好一条** Assign 指向该名字（AnnAssign / 第二次赋值都不行）
    assigns = [st for st in tree.body
               if isinstance(st, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'ANCHOR_V21'
                       for t in st.targets)]
    assert len(assigns) == 1, \
        'ANCHOR_V21 必须只有一条模块级赋值，实得 {} 条'.format(len(assigns))
    node = assigns[0].value
    assert isinstance(node, ast.Dict), 'ANCHOR_V21 应是字面量 dict'
    allowed = (ast.Dict, ast.Tuple, ast.List, ast.Constant, ast.UnaryOp,
               ast.Load)
    offenders = sorted({type(n).__name__ for n in ast.walk(node)
                        if not isinstance(n, allowed)})
    assert not offenders, \
        'ANCHOR_V21 不是纯字面量，疑似由被测代码推导: {}'.format(offenders)
    # 兜底：显式点名三类最危险的节点（也含在上面的 offenders 检查里）
    used = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)})
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    assert not used and not calls, \
        'ANCHOR_V21 引用了名字 {} 或调用了 {} 个函数'.format(used, len(calls))
    # ② 没有任何 Store/Del 的下标或属性写入落在 ANCHOR_V21 上——上面的字面量
    #    检查只管那一条语句，管不到 `ANCHOR_V21['heads']['policy_params'] = …`
    #    或 `ANCHOR_V21.update(…)`：这类改写会静默地把锚点调偏（改前无人抓）。
    #    代码今天是干净的（一条赋值、零改写），本断言是**钉住这条性质**。
    writes = [type(n).__name__ for n in ast.walk(tree)
              if isinstance(n, (ast.Subscript, ast.Attribute))
              and isinstance(n.ctx, (ast.Store, ast.Del))
              and any(isinstance(v, ast.Name) and v.id == 'ANCHOR_V21'
                      for v in ast.walk(n.value))]
    assert not writes, \
        'ANCHOR_V21 在模块级被改写过（字面量检查管不到这类写入）: {}'.format(writes)


def test_anchor_v21_prints_known_discrepancies_in_a_pinned_format(capsys):
    """`⚠ 已知矛盾` 这条 print 改前**零测试覆盖**，是三处记录点里唯一裸奔的。

    它是把「已知矛盾原样可见」交给 P4.2 审计的唯一凭据：格式一变（字段
    改名、少打一个数字、status 被截断），矛盾就等于被静默抹掉了。
    故按行断言：每条 entry 各占一行，行内必须同时出现 id、带千分位的
    stated/recorded/delta、以及 status 全文。
    """
    S.main(['--preset', 'v21'])
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith('⚠ 已知矛盾[')]
    entries = S.ANCHOR_V21['known_discrepancies']
    assert len(lines) == len(entries) == 2, \
        '已知矛盾行数不符: {} vs {}'.format(len(lines), len(entries))
    for x in entries:
        hit = [ln for ln in lines if ln.startswith('⚠ 已知矛盾[{}]'.format(x['id']))]
        assert len(hit) == 1, '条目 {} 的行没打出来或重复: {}'.format(x['id'], lines)
        ln = hit[0]
        for field in ('stated', 'recorded', 'delta'):
            assert '{:,}'.format(x[field]) in ln, \
                '{} 的 {}={:,} 没打出来: {!r}'.format(x['id'], field, x[field], ln)
        assert x['about'] in ln and x['status'] in ln, \
            '条目 {} 丢了 about/status: {!r}'.format(x['id'], ln)
    # policy 那条的 128 必须双向可见：表记 32,896 / 采 33,024
    pol = next(x for x in entries if x['id'] == 'policy_fc2_128_off')
    pol_line = next(ln for ln in lines if 'policy_fc2_128_off' in ln)
    assert '32,896' in pol_line and '33,024' in pol_line, \
        'policy 的笔误值与修正值必须都在输出里: {!r}'.format(pol_line)
    assert pol['delta'] == 128
    # 仲裁规则也被打出来（v21 分支是给人看的表，规则不该只活在源码里）
    assert '仲裁规则' in out and S.ANCHOR_V21['arbitration_rule'] in out, \
        '打印表没有陈列仲裁规则'


def test_anchor_v21_print_tolerates_a_new_params_key(capsys, monkeypatch):
    """params 每加一个键都必须能打出来——这张表就是要长的。

    改前是 `label[k]` 硬索引：下一个结构块加进来（params 多一项）就直接
    KeyError 崩掉整张表。现在缺键退回键名本身。合计仍当场自洽核算，故把
    backbone_total 一起加平，MISMATCH 不得出现。
    """
    grown = dict(S.ANCHOR_V21,
                 params={**S.ANCHOR_V21['params'], 'future_block': 1_234})
    grown['backbone_total'] = S.ANCHOR_V21['backbone_total'] + 1_234
    grown['total_params'] = grown['backbone_total'] + \
        S.ANCHOR_V21['heads']['policy_params'] + S.ANCHOR_V21['heads']['value_params']
    monkeypatch.setattr(S, 'ANCHOR_V21', grown)
    S.main(['--preset', 'v21'])
    out = capsys.readouterr().out
    assert 'future_block' in out and '1,234' in out, \
        'params 的新键没有被打出来（label 缺键时应退回键名）'
    assert 'MISMATCH' not in out, '加键后合计核算没跟上'
    # 既有六行仍逐行在场（没有因新键被挤掉）
    for k in S.ANCHOR_V21['params']:
        assert '{:,}'.format(S.ANCHOR_V21['params'][k]) in out, \
            '加键后原有分项 {} 丢失'.format(k)

