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

P4.9d 跟随用户「Mamba-1 换 Mamba-2」（块9-12）：`MambaLTI` 改名 `Mamba2`，
参数 454,112 → **516,960**（每块 113,528 → **129,240**），backbone_total
6,499,800 → **6,562,648**，total_params 9,008,547 → **9,071,395**。形状字段
（N 64 / r 4 / d_conv 4 / dt_rank 4 / expand 2 / A 结构化低秩 992）入锚点，
好让 P4.2 只读锚点就能拿到块布局。

⚠ **total_params 是 9,071,395 而不是任务书写的 9,071,389**：后者与本表其余
字面量无法自洽（6,562,648+2,366,730+142,017 = 9,071,395，差 6），从旧值顺推
（9,008,547+62,848）也等于 9,071,395。按 arbitration_rule 前半句（形状被钉死
→ 取精确算术值），9,071,389 属表内笔误。

P4.9c review 的 Important 5 是「主干做了分项级核对、头没做」。本轮把同样的
分项核对**常驻**到 Mamba-2 块上（8 个分项之和 == 129,240，逐项对上精确算术），
并新增 policy 首层仍是 128 的守卫——防有人把已撤回的 120 变更「修」回来。

P4.9e 跟随用户「Mamba-2 的 A 用 **per-head scalar decay**」：head 数 P=4 把
C=184 切成 4×46，A 只剩每 head 一个标量 ⇒ A 的记账 992 → **4**（A_log 形状
(4,)），`mamba2_rank` 键与「结构化低秩 C*r + r*N」口径**整体作废**。参数
516,960 → **513,008**（每块 129,240 → **128,252**），backbone_total
6,562,648 → **6,558,696**，total_params 9,071,395 → **9,067,443**。分项核对
随之变成 368+67,712+920+24,288+920+**4**+184+33,856 = 128,252。

P4.9e 另一支：stem 的 3×3 vs 7×7 **形状冲突已由用户 2026-09-27 显式裁决**
（取 3×3）。`arbitration_rule` 因此从两支变三支：②「形状矛盾、算术不裁决」
是个**有终态**的中间态，③「经用户显式裁决」才是它的归宿（resolution =
`resolved_by_user_ruling`，被否的 7×7 口径以 `voided_by_user_ruling` 整份
留痕，含它会把合计顶到多少）。stem 参数仍是 28,520，所有合计不变。
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
    assert '6,558,696' in out and '9,067,443' in out, \
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

    ⚠ P4.9e：块9-12 的 A 改 **per-head scalar decay**（P=4 heads × 46 通道），
    mamba 项 516,960 → **513,008**（每块 129,240 → **128,252**），
    backbone_total 6,562,648 → **6,558,696**，total_params 9,071,395 →
    **9,067,443**。三个数只由 A 那 4×(992-4) = 3,952 驱动，其余分项一字未动。
    ⚠ P4.9d：块9-12 换成 Mamba-2（454,112 → 516,960，旧值 113,528/块）。
    键名同时由 `mamba_lti*` 改为 `mamba2*`（用户「从 1 换 2」）。

    ⚠ P4.9c：policy 头由 2,366,602 更正为 **2,366,730**（表头自己的数字）。
    原值 2,366,602 来自权威表的 policy 分项行，而那一行里「FC 128→256
    带bias：32,896」是笔误——bias 被记成 in_features。下面的 policy_lines
    断言是本测试的关键：把表自己的 5 行分项（fc2 换成精确值 33,024）加总，
    必须正好等于 heads.policy_params。分项与合计对不上时，错的几乎总是分项。
    """
    A = S.ANCHOR_V21
    assert A['name'] == 'v21'
    assert A['in_channels'] == 17
    assert A['backbone_channels'] == 184
    lay = A['layout']
    assert lay['stem'] == 'Conv3x3(17->184)+BN'
    assert lay['res_blocks'] == 8
    assert lay['mamba2_blocks'] == 4
    assert lay['transformer_blocks'] == 2
    assert lay['transformer_heads'] == 4
    assert lay['ffn_hidden'] == 240, 'FFN 中间维必须是 240（旧表 276 作废）'
    assert abs(lay['ffn_ratio'] - 240 / 184) < 1e-3, \
        'ffn_ratio 与 240/184 不符: {}'.format(lay['ffn_ratio'])
    assert lay['cross_attn_res_blocks'] == 2
    assert lay['cross_attn_concat_blocks'] == (1, 5, 9)
    assert lay['out'] == '1x1 Conv+BN'
    # 权威表逐项（P4.1 brief §2；块9-12 已是 Mamba-2 + per-head scalar A）
    parts = dict(stem=28_520, res_blocks=4_881_152, mamba2=513_008,
                 transformer=448_960, cross_attn_res=652_832, out=34_224)
    assert A['params'] == parts, '各部分参数量与权威表不符'
    # 每块单价也对得上（448,960 = 2×224,480；652,832 = 2×326,416）
    assert parts['transformer'] == 2 * 224_480
    assert parts['cross_attn_res'] == 2 * 326_416
    assert parts['res_blocks'] == 8 * 610_144
    assert parts['mamba2'] == 4 * 128_252, 'Mamba-2 每块必须是 128,252'
    # 合计自洽：六项之和 == backbone_total == 6,558,696（权威表）
    assert sum(parts.values()) == A['backbone_total'] == 6_558_696, \
        '主干合计与各部分之和不自洽'
    h = A['heads']
    assert h['in'] == 184
    assert h['policy_params'] == 2_366_730, \
        'policy 头 = 表头合计 2,366,730（分项 fc2 的 32,896 是笔误，见 §3(b)）'
    assert h['policy_out'] == 362
    assert h['value_params'] == 142_017
    assert h['value_out'] == 1
    assert A['total_params'] == 9_067_443
    assert A['backbone_total'] + h['policy_params'] + h['value_params'] \
        == A['total_params'], '总合计与 主干+两头 不自洽'
    # 增量的另一种算法：从 P4.9c 的旧值顺推，必须得到同一个数
    assert 9_008_547 + (513_008 - 454_112) == 9_067_443, \
        'total_params 与「旧值 + Mamba 增量」对不上'
    # 增量还必须是 A 的记账差 × 4（防止有人顺手改了别的分项却仍凑对合计）
    assert 4 * (992 - 4) == 3_952 == 516_960 - 513_008
    assert 6_562_648 - 3_952 == 6_558_696 and 9_071_395 - 3_952 == 9_067_443
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
    assert 6_558_696 + 2_366_730 + 142_017 == 9_067_443


# 块9-12 Mamba-2 的 8 个分项（权威表逐项，顺序 = 子模块定义序）。
# 每项的算式钉在 test_anchor_v21_mamba2_blockwise_sums 里：任一项被「顺手改
# 整齐」都会当场变红，而不是等到与 128,252 对不上才被人发现。
MAMBA2_BLOCK_PARTS = [
    ('norm', 368),            # LayerNorm2d(184)：weight+bias = 2×184
    ('in_proj', 67_712),      # Linear(184→368) 无 bias：d_inner=C，扇出 expand×C
    ('dw_conv', 920),         # Conv1d(184,184,k=4,groups=184,bias)：184×4+184
    ('x_proj', 24_288),       # Linear(184→132) 无 bias：132 = dt_rank + 2N
    ('dt_proj', 920),         # Linear(4→184) 带 bias：4×184+184
    ('A', 4),                 # per-head 标量：P 个（旧结构化低秩 992 作废）
    ('D', 184),               # 逐通道直通参数
    ('out_proj', 33_856),     # Linear(184→184) 无 bias
]


def test_anchor_v21_mamba2_blockwise_sums():
    """块9-12 的 8 个分项之和必须精确等于 128,252，且**逐项**对上精确算术。

    P4.9c 的 review Important 5 指出：分项级核对当时只做在主干、没做在头。
    本轮把同样的核对常驻到 Mamba-2 块上——不是只对总和：只对总和的话，把 A
    写成某个凑数用的数、同时把某一项写成相反差，也能凑出同一个和；逐项钉死
    后任何一项漂移都无处藏身。

    ⚠ `expand=2` 是 **in_proj 的扇出**（184→368），**不是**分支内宽度扩张：
    d_inner 仍是 C=184。按 Mamba 惯例误读成 d_inner=2C 会让 in_proj 变成
    184→736（135,424），整块凭空多出 67,712——权威表明确只认扇出这一种。

    ⚠ P4.9e：A 由「结构化低秩 C*r + r*N」改为 **per-head 标量**（P=4），
    6 个分项一字未动，只有 A 那项 992 → 4。两种旧记账各自会让块变成
    129,240（低秩 992）/ 140,024（稠密 C×N），都不是 128,252。
    """
    A, lay = S.ANCHOR_V21, S.ANCHOR_V21['layout']
    C, N = 184, 64
    P = lay['mamba2_n_heads']
    d_conv, dt_rank, expand = lay['mamba2_d_conv'], lay['mamba2_dt_rank'], 2
    d_inner = C                                  # 见 docstring：扇出而非内宽
    x_proj_out = dt_rank + 2 * N                 # 4 + 2×64 = 132

    exact = [
        2 * C,                                   # norm: weight+bias
        d_inner * (expand * C),                  # in_proj 无 bias
        d_inner * d_conv + d_inner,              # dw_conv 深度卷积带 bias
        d_inner * x_proj_out,                    # x_proj 无 bias
        dt_rank * d_inner + d_inner,             # dt_proj 带 bias
        P,                                       # A：每 head 一个标量（**不是** C×N）
        C,                                       # D
        d_inner * C,                             # out_proj 无 bias
    ]
    names = [n for n, _ in MAMBA2_BLOCK_PARTS]
    stated = [v for _, v in MAMBA2_BLOCK_PARTS]
    assert names == ['norm', 'in_proj', 'dw_conv', 'x_proj', 'dt_proj',
                     'A', 'D', 'out_proj'], '分项清单本身被改动: {}'.format(names)
    for (name, got), want in zip(MAMBA2_BLOCK_PARTS, exact):
        assert got == want, \
            'Mamba-2 分项 {} 被改动：{:,} != 精确算术 {:,}'.format(name, got, want)
    assert len(stated) == 8, '必须是 8 个分项'
    assert sum(stated) == 128_252, \
        'Mamba-2 单块 8 分项之和 {:,} != 128,252'.format(sum(stated))
    # 逐块单价 × 4 必须回到锚点里那一项（不是各自独立写死的两个数）
    assert A['params']['mamba2'] == 4 * sum(stated) == 513_008
    # dw_conv 与 dt_proj 同为 920 是巧合（184×4+184 与 4×184+184），非笔误
    assert MAMBA2_BLOCK_PARTS[2][1] == MAMBA2_BLOCK_PARTS[4][1] == 920
    assert lay['mamba2_A_params'] == P == 4, '锚点里的 A 记账必须与分项一致'
    # A 若仍按旧的结构化低秩 / 稠密 C×N 记账，块会变成 129,240 / 140,024
    head = sum(stated) - 4
    assert head + 992 == 129_240, '结构化低秩口径作废，单块不该再是 129,240'
    assert 184 * 64 == 11_776 > 4, '稠密 C×N 远大于 per-head 标量'
    assert head + 11_776 == 140_024
    # per-head 记账与 head 切分必须自洽：P 个标量 × 4 = 4 个 head × 46 通道 = 184
    assert P * lay['mamba2_head_dim'] == C == 184


def test_anchor_v21_records_mamba2_shape():
    """Mamba-2 的形状字段必须入锚点，且旧称/旧参数ization 绝迹。

    P4.2 只读 ANCHOR_V21 常量就要能搭出块9-12，形状（N / P / head_dim /
    d_conv / dt_rank / expand / A 的记账）不能只活在实现代码或本文件里。

    P4.9e：A 由结构化低秩改为 **per-head scalar decay**——A 的标量个数 = head
    数 P=4，A_log 形状 (4,)。旧的 `mamba2_rank` 键与 C*r + r*N 口径必须
    **彻底绝迹**（键名、描述文字、数字三样都查）：留着它们等于给后续 agent
    留一张按旧参数ization 改回来的票据。
    """
    A, lay = S.ANCHOR_V21, S.ANCHOR_V21['layout']
    assert lay['mamba2_blocks'] == 4
    assert lay['mamba2_d_state'] == 64, 'N：Mamba-1 的 16 作废（Mamba-2 为 64）'
    assert lay['mamba2_d_conv'] == 4
    assert lay['mamba2_dt_rank'] == 4
    assert lay['mamba2_expand'] == 2
    # per-head 标量：A 的参数 = head 数 P，A_log 形状 (4,)，与 head 切分自洽
    C, P, P2 = A['backbone_channels'], lay['mamba2_n_heads'], lay['mamba2_head_dim']
    assert (P, P2) == (4, 46), 'head 数/head 维必须是 4×46（用户裁决 P=4）'
    assert P * P2 == C == 184, 'head 切分必须把 C=184 铺满，不许有缝'
    assert lay['mamba2_A_params'] == P == 4, 'A 只剩 P=4 个标量'
    assert 'per-head' in lay['mamba2_A_form'] and '(4,)' in lay['mamba2_A_form'], \
        'A 的记账口径必须写在锚点里: {!r}'.format(lay['mamba2_A_form'])
    # x_proj 出向 = dt_rank + 2N，形状字段自洽（否则 24,288 无从复核）
    assert lay['mamba2_dt_rank'] + 2 * lay['mamba2_d_state'] == 132
    # 旧称 MambaLTI 只允许出现在源码注释的历史记录里，锚点数据与打印表都不许有
    blob = repr(lay) + repr(A['params']) + repr(A['heads'])
    assert 'mambalti' not in blob.lower() and 'mamba_lti' not in blob, \
        '锚点数据里还残留旧键名 mamba_lti'

    # ---- 低秩参数ization 的残留：一个都不许有（键名 / 口径 / 数字） ----
    def all_keys(obj):
        """递归取出对象里所有键名。"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from all_keys(v)
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                yield from all_keys(v)

    assert 'mamba2_rank' not in set(all_keys(lay)), \
        'mamba2_rank 键还在：per-head 标量没有 r 这个量'
    for k in all_keys(A):
        assert 'rank' not in k or k == 'mamba2_dt_rank', \
            'A 的低秩键不该存在（dt_rank 是另一回事）: {!r}'.format(k)
    data = repr(A)
    for bad, why in (('C*r', '低秩口径'), ('r*N', '低秩口径'),
                     ('low-rank', '低秩口径'), ('992', '旧 A 记账'),
                     ('1920', '凑数用的假 A 记账'), ('11776', '稠密 C×N 记账')):
        assert bad not in data, '锚点数据里还残留{}（{}）'.format(bad, why)
    # 但**理由**必须留在源码注释里：992 是怎么被否掉的，日后要能查
    src = _anchor_source_with_comments()
    assert 'per-head' in src and '992' in src and '作废' in src, \
        '锚点注释必须记下 A 的低秩口径为何作废（否则只剩数字、丢掉理由）'


def test_anchor_v21_policy_head_still_128(capsys):
    """policy 头首层仍是 128、合计仍是 2,366,730——防有人"修"回已撤回的 120。

    曾有一版把 Flatten 之后的首个 FC 从 128 改成 120 的草稿，被用户**撤回**：
    该变更并不存在。因此锚点里既没有它、也不该出现任何指向它的条目（登记
    一条 user_directed_change 等于给后续 agent 留张"待办"票据）。本测试把
    "没变"这件事本身钉住：合计、首层宽度、以及分项里 128×256+256 的形状。
    """
    A = S.ANCHOR_V21
    h = A['heads']
    assert h['policy_params'] == 2_366_730, \
        'policy 头必须保持 2,366,730（用户明确「policy 不变」）'
    # 分项里带出首层 128 的那两行：17328→128 与 128→256
    assert 2_218_112 == 48 * 19 * 19 * 128 + 128, '首层 FC 17328→128'
    assert 33_024 == 128 * 256 + 256, '次层 FC 128→256：bias 恒为 out_features'
    assert sum([17_856, 4_704, 2_218_112, 33_024, 93_034]) == 2_366_730
    # 首层宽度同时出现在乘式里：48*19*19*128 换成 120 就凑不出 2,218,112
    assert 48 * 19 * 19 * 120 + 120 == 2_079_480 != 2_218_112, \
        '首层若被改成 120，分项就与 2,366,730 对不上了（差 {:,}）'.format(
            2_218_112 - 2_079_480)
    # 锚点里不得存在任何把 120 当作权威值的条目
    for d in A['known_discrepancies']:
        assert d['recorded'] not in (2_079_480, 120), \
            '锚点里混进了已撤回的 120 变更条目: {}'.format(d)
    S.main(['--preset', 'v21'])
    out = capsys.readouterr().out
    assert '2,366,730' in out, '打印表没有陈列 policy 合计'
    assert '2,366,602' not in out, '被作废的旧裁定仍在打印表里'


def test_anchor_v21_layout_names_match_p41_class_names(capsys):
    """布局/参数键名与 P4.1 的类名逐项对齐：旧称必须绝迹。

    P4.1 brief §1 定的类名是 Mamba2 / TransformerBlock / CrossAttnRes
    （块13-14 旧称 LightAttn 作废；块9-12 旧称 MambaLTI，用户本轮明确
    「从 1 换成 2」，一并作废）。键名一旦与类名脱节，锚点表就会与被实现的
    结构各说各话——而锚点正是 P4.2 预算仲裁的依据。
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
    assert {'mamba2_blocks', 'transformer_blocks',
            'cross_attn_res_blocks'} <= set(lay), \
        '布局缺 P4.1 的三个块类键: {}'.format(sorted(lay))
    # 旧称 LightAttn / light_attn_* / MambaLTI / mamba_lti_* 全部消失
    # （大小写不敏感；旧键名会让 P4.2 读不到新布局，看起来像「没改」）
    for s in list(strings(lay)) + list(lay) + list(A['params']):
        low = s.lower()
        assert 'lightattn' not in low and 'light_attn' not in low, \
            '布局/参数里还残留旧称 LightAttn: {!r}'.format(s)
        assert 'mambalti' not in low and 'mamba_lti' not in low, \
            '布局/参数里还残留旧称 MambaLTI（Mamba-1）: {!r}'.format(s)
    assert 'mamba2' in A['params'] and 'mamba_lti' not in A['params']
    # 打印表用 P4.1 的类名（run_anchor_v21 的 label 跟着键名走）
    S.main(['--preset', 'v21'])
    out = capsys.readouterr().out
    assert 'TransformerBlock ×2' in out, '打印表没有用 TransformerBlock'
    assert 'LightAttn' not in out, '打印表仍出现旧称 LightAttn'
    assert 'MambaLTI' not in out, '打印表仍出现旧称 MambaLTI'
    assert 'Mamba2 ×4' in out and 'CrossAttnRes ×2' in out
    assert 'ffn=240' in out, '打印表没有陈列 FFN 中间维 240'


def test_anchor_v21_arbitration_rule_is_single_and_applied_to_both_entries():
    """**一条**仲裁规则，两处冲突都按它判——不许对同型证据换标准。

    规则（ANCHOR_V21['arbitration_rule']，注释里有长版）三支：
      ①形状被唯一钉死 → 取精确算术值（冲突的表记数字 = 笔误）
        ⇒ resolved_by_exact_arithmetic
      ②形状本身矛盾 → 精确算术**不裁决**，两口径都留着待用户拍板
        ⇒ unresolved_shape_conflict（一个**有终态**的中间态）
      ③经用户显式裁决 → 按裁决取值，被否口径 voided_by_user_ruling 留痕
        ⇒ resolved_by_user_ruling

    规则落到两条 entry 的 `resolution` 上，必须**互不相同**且各归其位：
      (a) stem：表头文字 7×7 与它自己给的参数量 3×3 指向不同形状 ⇒ 算术裁决
          不了，先走②；用户 2026-09-27 显式裁决 3×3 ⇒ 转入③
          （resolved_by_user_ruling）。**不允许**退回②/悬着。
      (b) policy：fc2 的形状与 bias 都被表写死 ⇒ 精确算术有唯一解 ⇒ 走①
          （resolved_by_exact_arithmetic，规则必须给出结论）。
    「两条 resolution 必须同时存在且分属不同支」正是 P4.9b 缺的东西：它对 (a)
    弃表取实算、对 (b) 取实算弃表头，同型证据两套相反标准。P4.9e 只是把②补
    上了归宿（③），没有放松任何一支。
    """
    A = S.ANCHOR_V21
    rule = A['arbitration_rule']
    assert isinstance(rule, str) and rule, '锚点必须带一条可读的仲裁规则'
    for branch in ('精确算术', '形状本身矛盾', '显式裁决'):
        assert branch in rule, '仲裁规则缺分支：{}'.format(branch)
    # 三个 resolution 标记必须都写在规则里（②不能是「无归宿」的分支）
    for res in ('resolved_by_exact_arithmetic', 'unresolved_shape_conflict',
                'resolved_by_user_ruling', 'voided_by_user_ruling'):
        assert res in rule, '仲裁规则缺 resolution 标记：{}'.format(res)
    d = {x['id']: x for x in A['known_discrepancies']}
    stem = d['stem_kernel_3x3_vs_7x7']
    assert stem['resolution'] == 'resolved_by_user_ruling', \
        '(a) 的形状冲突已由用户显式裁决（3×3），不许退回「悬着」'
    assert d['policy_fc2_128_off']['resolution'] == 'resolved_by_exact_arithmetic', \
        '(b) 的形状被钉死，规则必须裁决它（而不是含糊其辞）'
    # 两条 entry 仍必须分属不同支（不是「反正都裁决了就都写同一个词」）
    assert stem['resolution'] != d['policy_fc2_128_off']['resolution']
    # 规则与结论方向一致：被算术裁决的 (b) 必须站「精确算术」这一侧
    pol = d['policy_fc2_128_off']
    assert pol['recorded'] == 128 * 256 + 256 > pol['stated'] == 128 * 256 + 128, \
        '被裁决的条目必须采纳精确算术值、否掉表记值'
    # 被用户裁决的 (a)：两侧都精确（谁也不比谁权威）⇒ 结论只能来自裁决本身，
    # 所以条目里必须写明裁决者，且 status 不许再写「未裁决」
    assert stem['recorded'] == 3 * 3 * 17 * 184, '3×3 精确'
    assert stem['stated_exact'] == 7 * 7 * 17 * 184, '7×7 精确'
    assert stem['ruled_by'].startswith('user_ruling'), \
        '走③的条目必须写明裁决来自用户（不是算术）'
    assert '未裁决' not in stem['status'] and stem['status'].startswith('已裁决')
    # 悬而未决的那一支现在是空的：两条都已定案，不许有第三条偷偷挂着
    assert not [x['id'] for x in A['known_discrepancies']
                if x['resolution'] == 'unresolved_shape_conflict'], \
        '已无待裁决条目，不该再挂在 unresolved_shape_conflict 上'


def test_anchor_v21_stem_is_3x3_and_7x7_is_voided():
    """stem = 3×3 已由用户显式裁决；7×7 口径**作废但整份留痕**。

    2026-09-27 之前，这条冲突在锚点里挂了四轮 `unresolved_shape_conflict`
    （3×3 的 28,152 vs 7×7 的 153,272，各自形状下都精确 ⇒ 算术裁决不了）。
    现在它已定案：stem 取 3×3，合计 28,520 不变，**所有合计一字未动**。

    为什么作废的那一支还要留着：7×7 的提案只存在于 git 历史里，日后有人
    `git log` 翻出来会以为它「还没被评估过」。故 voided_alternative 必须
    连同它**会把合计顶到多少**一起记下（153,272 / stem 153,640 / backbone
    6,683,816 / total 9,192,563），让复活它的成本一眼可见。
    """
    A = S.ANCHOR_V21
    stem = next(x for x in A['known_discrepancies']
                if x['id'] == 'stem_kernel_3x3_vs_7x7')
    # 采用的一侧：3×3，且与 params.stem 逐项对上（9×17×184 + BN 368）
    assert A['params']['stem'] == stem['recorded'] + 368 == 28_520
    assert stem['recorded'] == 3 * 3 * 17 * 184 == 28_152
    assert '3x3' in A['layout']['stem'].lower(), '布局必须标明 stem 按 3×3 记'
    # 作废的一侧：形状与两个口径的数字都在，且明确标成 voided 而非「待定」
    void = dict(stem['voided_alternative'])
    assert stem['voided'] == 'voided_by_user_ruling', \
        '7×7 口径必须标成被用户裁决作废，不是「还没定」'
    assert void['kernel'] == '7x7'
    assert void['conv_params'] == 7 * 7 * 17 * 184 == 153_272
    assert void['stem_with_bn'] == void['conv_params'] + 368 == 153_640
    # 作废口径的代价：把它换回去，两个合计会变成多少（= 本表合计 + stem 增量）
    assert void['backbone_total'] == A['backbone_total'] \
        - A['params']['stem'] + void['stem_with_bn'] == 6_683_816
    assert void['total_params'] == void['backbone_total'] \
        + A['heads']['policy_params'] + A['heads']['value_params'] == 9_192_563
    # 裁决书里同时流传的 6,687,768 / 9,196,515 是按 P4.9d 基数算的（多 3,952），
    # 已被本锚点按精确算术取代——但**不许从记录里抹掉**，否则日后无从解释
    assert '6,687,768' in void['note'] and '9,196,515' in void['note']
    assert 6_687_768 - void['backbone_total'] == 3_952 == 4 * (992 - 4)
    # 表记口径的 24 个笔误仍在（153,296 vs 精确 153,272）
    assert stem['stated'] == 153_296 and stem['stated_exact'] == 153_272
    # status 必须写明作废那一支的代价（否则复活它的人看不见代价）
    assert '作废' in stem['status'] and '6,683,816' in stem['status']
    # 打印表里也要能看出这条已定案（resolution 逐条打出行内，见打印格式测试）
    assert stem['resolution'] in ('resolved_by_user_ruling',
                                  'resolved_by_exact_arithmetic'), \
        'stem 已定案（用户裁决 3×3），不许退回 unresolved_shape_conflict'
    assert stem['resolution'] in S.ANCHOR_V21['arbitration_rule'], \
        'resolution 必须是仲裁规则里写明的那一支（否则是凭空发明的状态）'


def test_anchor_v21_documents_known_discrepancies():
    """权威表的两处笔误必须被记录在锚点里，不许静默抹掉。

    (a) stem 表头写 7×7，参数却是 3×3 的值（28,152 vs 153,296）。锚点按 3×3
        取值（**用户 2026-09-27 显式裁决**，规则第③支）—— 7×7 那支已作废，
        但仍连同它会把合计顶到多少一起留痕，见 test_anchor_v21_stem_is_3x3_
        and_7x7_is_voided。若日后确认改 7×7，stem 变 153,640，**整表作废**。
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
    # (a) 已定案：走规则第③支，结论来自用户裁决而不是算术
    assert stem['resolution'] == 'resolved_by_user_ruling' \
        and stem['voided'] == 'voided_by_user_ruling' \
        and stem['ruled_by'].startswith('user_ruling')

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
                   'arbitration_rule', 'known_discrepancies',
                   '显式裁决', 'per-head'):
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
    故按行断言：每条 entry 各占一行，行内必须同时出现 id、resolution、带
    千分位的 stated/recorded/delta、以及 status 全文。⚠ P4.9e 起 resolution
    也进打印：一条「已定案」和一条「还悬着」在输出里必须长得不一样。
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
        assert 'resolution={}'.format(x['resolution']) in ln, \
            '条目 {} 的 resolution 没打出来（已定案/悬着必须一眼可分）: {!r}'.format(
                x['id'], ln)
    # 两行落在**不同**分支上，且 stem 那行带的是用户裁决
    stem_line = next(ln for ln in lines if 'stem_kernel_3x3_vs_7x7' in ln)
    pol_line = next(ln for ln in lines if 'policy_fc2_128_off' in ln)
    assert 'resolution=resolved_by_user_ruling' in stem_line
    assert 'resolution=resolved_by_exact_arithmetic' in pol_line
    # 已无待裁决条目：**矛盾行**里不该还出现 unresolved（仲裁规则本身要提到
    # ②那一支的名字，那是规则文本，不是「还有条目悬着」）
    assert not [ln for ln in lines if 'unresolved' in ln], \
        '已无待裁决条目，矛盾行里不该再出现 unresolved'
    # policy 那条的 128 必须双向可见：表记 32,896 / 采 33,024
    pol = next(x for x in entries if x['id'] == 'policy_fc2_128_off')
    assert '32,896' in pol_line and '33,024' in pol_line, \
        'policy 的笔误值与修正值必须都在输出里: {!r}'.format(pol_line)
    assert pol['delta'] == 128
    # 作废的 7×7 那一支的代价也得打在行内（复活它的人第一眼就要看见）
    assert '作废' in stem_line and '6,683,816' in stem_line
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

