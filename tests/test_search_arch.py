"""search_arch.py 的测量正确性测试。

三项测量都必须可信，否则整个搜索无意义：
  * 参数量  —— 直接求和，最容易错的是漏算/重算
  * FLOPs   —— 必须按 batch=1 测，否则被 batch 放大
  * 激活显存 —— 必须按 **storage 去重**，且要正确反映 gradient checkpointing

尤其第三项：曾用 `t.numel() * element_size()` 累加，实测高估 48%
（反向保存大量视图，共享同一块 storage，被重复计数）；改按
`untyped_storage().nbytes()` + `data_ptr()` 去重后误差降到 21%，剩余
部分来自 checkpoint 重算临时量被按「同时存活」计入。

P4.9 另覆盖两件事：`--preset` 取值修型（store_true -> choices）、measure 的
多通道支持（默认 12 零回归）。

katago-se-v1：v21 架构（Mamba/Transformer/CrossAttn）从 src/networks/ 删除时，
其静态锚点 ANCHOR_V21、`--preset v21`、run_anchor_v21 一并从
scripts/search_arch.py 退役 —— 锚点记录的结构已不存在，静态表失去被测对象。
现行候选 = KataGo SE-bottleneck：search_arch.SE_CFG 镜像
scripts/train_sft.py 的 KATAGO_SE_CFG **结构键**，measure 从 cfg 自带的
`arch` 建网，`--preset se` 走与 v18 完全相同的实测投影路径 —— 参数量/FLOPs
是本机实测，显存与 s/step 仍**自 v18 锚点外推（误差未知）**，两者的口径差
写在 project() 的 docstring 里，测试不把外推值钉成金标（只有 params/flops
这类实测整数进 GOLDEN）。
"""

import inspect
import os
import sys
import pytest

import torch

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


def test_probe_owns_checkpoint_semantics(monkeypatch):
    """计量口径必须由探针钉死，不随主干的检查点策略漂移（P4.6b 回归钉）。

    背景：锚点 31.12GB（use_checkpoint 1）是 `checkpoint_sequential` 口径
    实测的（逐块检查点、最后一块保留），k≈0.83 与 attn_window 显存敏感性
    都建立在它上面。P4.6b 把主干 `use_checkpoint=True` 换成「每块都检查点」
    后，同一探针 k 漂到 1.35、attn_window 敏感性消失——上两个测试因此变红。

    修复契约：`measure()` 构造后**显式关掉主干自己的检查点开关**、自己包裹
    `_ProbeCheckpointBlocks`。故无论子类在构造时把主干策略设成 True 还是
    False，`measure(use_checkpoint=True)` 的结果都必须逐项相等。开/关两种
    策略至少一种会在「探针继承主干策略」的旧实现上撞出差异。
    """
    baseline = S.measure(SMALL, probe_bs=2)

    def force_policy(policy):
        class ForceNet(S.AlphaGoNet):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.backbone.use_checkpoint = policy
                if hasattr(self.backbone, 'set_grad_checkpointing'):
                    self.backbone.set_grad_checkpointing(policy)
        return ForceNet

    for policy in (False, True):
        monkeypatch.setattr(S, 'AlphaGoNet', force_policy(policy))
        got = S.measure(SMALL, probe_bs=2)
        assert got == baseline, \
            '主干检查点策略={} 影响了计量（探针没有钉死语义）: {} != {}'.format(
                policy, got, baseline)


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
# P4.9：--preset 修型（store_true -> 取值）/ measure 多通道 / 预设金标
# ---------------------------------------------------------------------------

# 既有 4 个预设在**改前**的实测金标（params/flops 均为整数，机器无关；
# 改前用 scripts/search_arch.py 逐一 project 采集）。任何使既有预设行为
# 漂移的改动都会撞上它。
# 第 5 行 SE 是 katago-se-v1 新增：params 与 train_sft 的 KATAGO_SE_CFG
# ['params_total']=9,112,005 对账（test_se_measure_matches_katago_budget），
# flops 是本工具 batch=1 forward hook 实测整数。
GOLDEN_PRESETS = [
    ('v18 原样（锚点）', 12_858_978, 8_489_705_920),
    ('V12 原样', 12_652_834, 8_987_335_872),
    ('B: ConvNeXt4→Res4', 14_315_874, 9_540_475_840),
    ('F: V12块序+v18头', 14_056_866, 9_354_165_184),
    ('SE: KataGo 240ch（候选）', 9_112_005, 6_212_830_720),
]


def test_preset_accepts_se(capsys):
    """`--preset se` 必须能被解析并走**实测**投影路径（SE 是现行训练结构）。

    与退役的 v21 分支正相反：SE 结构 AlphaGoNet 造得出来 ⇒ 有校准、有
    forward、有显存投影行。note 里必须带「外推」标注 —— 显存/s/step 不是在
    SE 自己身上实测的，是从 v18 锚点外推的（口径见 project 的 docstring）。
    """
    args = S.build_parser().parse_args(['--preset', 'se'])
    assert args.preset == 'se'
    S.main(['--preset', 'se', '--batch', '1000'])
    out = capsys.readouterr().out
    assert '校准核对' in out, 'SE 预设应走 v18 校准（实测投影路径）'
    name, _, _, note = S.preset_entries('se')[0]
    assert name in out, 'SE 预设行没有出现在输出里'
    assert note in out, 'SE 行没有带「自 v18 锚点外推（误差未知）」标注'
    assert '9.11M' in out, 'SE 参数量应显示 9.11M（实测 9,112,005）'


def test_preset_choices_contain_legacy_values(capsys):
    """既有预设仍在 choices 里、旧命令行仍可用，且行为与改前逐项一致。"""
    p = S.build_parser()
    act = next(a for a in p._actions if a.dest == 'preset')
    assert set(S.PRESET_CHOICES) == {'all', 'v18', 'v12', 'b', 'f', 'se'}
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
    # 既有 4 条逐字节不变（前缀钉死）；SE 是新增的第 5 条
    assert S.preset_cfgs()[:4] == golden_cfgs, '既有预设定义被改动（与改前不一致）'
    se_cfg = dict(backbone_channels=240, backbone_res_blocks=17,
                  attention_mode='mix', num_attention_layers=4, num_heads=4,
                  attn_mode='global', attn_window=7,
                  res_blocks=0, convnext_blocks=0, attn_blocks=0,
                  value_channels=96, value_res_blocks=2,
                  policy_channels=128, policy_layers=3,
                  arch='se_bottleneck')
    se_name = 'SE: KataGo 240ch（候选）'
    se_note = 'KATAGO_SE_CFG 结构；显存/s/step 自 v18 锚点外推（误差未知）'
    assert len(S.preset_cfgs()) == 5
    assert S.preset_cfgs()[4] == (se_name, se_cfg, 3200, se_note), \
        'SE 预设定义被改动（结构键 / 默认 batch / 外推标注）'

    # 选择映射：整表 == 全清单；单项 == 对应下标；既有 4 项与金标逐项一致
    assert S.preset_entries('all') == S.preset_cfgs()
    for i, k in enumerate(('v18', 'v12', 'b', 'f', 'se')):
        assert S.preset_entries(k) == [S.preset_cfgs()[i]], \
            '{} 选错了预设'.format(k)
    for i, k in enumerate(('v18', 'v12', 'b', 'f')):
        assert S.preset_entries(k) == [golden_cfgs[i]], \
            '{} 选错了既有预设'.format(k)

    # 各跑一次，结果与实测金标一致（params/flops 为整数）
    for (gname, gparams, gflops), k in zip(
            GOLDEN_PRESETS, ('v18', 'v12', 'b', 'f', 'se')):
        (name, r), = S.run_preset(key=k)
        assert name == gname
        assert r['params'] == gparams, \
            '{} params {} != 金标 {}'.format(k, r['params'], gparams)
        assert r['flops'] == gflops, \
            '{} flops {} != 金标 {}'.format(k, r['flops'], gflops)

    # 通道数来源 = 预设推导：全部预设（含 SE = KATAGO_SE_CFG['in_channels']）恒 12
    for k in ('all', 'v18', 'v12', 'b', 'f', 'se'):
        assert S.preset_in_channels(k) == 12

    # --list：既有 4 行逐字节不变，SE 为新增行，v21 行绝迹
    S.main(['--list'])
    out = capsys.readouterr().out
    for gname, _, gbs, gnote in golden_cfgs:
        line = '  {:<30} batch={:<5} {}'.format(gname, gbs, gnote)
        assert line in out, '--list 既有行丢失/改动: {!r}'.format(line)
    se_line = '  {:<30} batch={:<5} {}'.format(se_name, 3200, se_note)
    assert se_line in out, '--list 缺 SE 行: {!r}'.format(se_line)
    assert 'v21' not in out, '--list 仍有 v21 残留'


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


# ---------------------------------------------------------------------------
# katago-se-v1：KataGo SE-bottleneck 候选（KATAGO_SE_CFG）的测量与投影
# ---------------------------------------------------------------------------


def test_se_preset_mirrors_katago_se_cfg():
    """SE_CFG 必须逐键等于**活的** KATAGO_SE_CFG 结构键（单一事实源对账）。

    train_sft.py 改 KATAGO_SE_CFG（宽度/块数/头）而忘了同步 search_arch 时，
    本测试当场变红 —— 否则 search_arch 投影的是一张不存在的结构表，预算结论
    会静默错位。
    """
    from scripts.train_sft import KATAGO_SE_CFG as K
    cfg = S.SE_CFG
    # 结构键逐一对应（键名映射：channels->backbone_channels、blocks->...）
    assert cfg['backbone_channels'] == K['channels'] == 240
    assert cfg['backbone_res_blocks'] == K['blocks'] == 17
    assert cfg['attention_mode'] == K['attention_mode'] == 'mix'
    assert cfg['num_attention_layers'] == K['num_attention_layers'] == 4
    assert cfg['num_heads'] == K['num_heads'] == 4
    assert cfg['value_channels'] == K['value_channels'] == 96
    assert cfg['value_res_blocks'] == K['value_res_blocks'] == 2
    assert cfg['policy_channels'] == K['policy_channels'] == 128
    assert cfg['policy_layers'] == K['policy_layers'] == 3
    assert cfg['arch'] == K['arch'] == 'se_bottleneck'
    assert S.preset_in_channels('se') == K['in_channels'] == 12
    # attn_mode/attn_window：KATAGO_SE_CFG 不传 ⇒ 必须等于 AlphaGoNet 默认
    # （build_katago_se_net 正是这么拿到 global/7 的）
    dflt = inspect.signature(S.AlphaGoNet).parameters
    assert cfg['attn_mode'] == dflt['attn_mode'].default == 'global'
    assert cfg['attn_window'] == dflt['attn_window'].default == 7
    # 行为键与对账硬数不进结构表（理由见 SE_CFG 上方注释）
    for banned in ('in_channels', 'grad_checkpoint', 'params_backbone',
                   'params_total'):
        assert banned not in cfg, \
            '{} 不该进 search_arch 的结构表'.format(banned)


def test_se_measure_matches_katago_budget():
    """SE 候选的实测参数量必须 == KATAGO_SE_CFG['params_total']（对账硬数）。

    这是「工具测出来的」与「train_sft 表里写的」唯一一次碰面：两个数都可能
    过期，只有互相钉住才有人负责。FLOPs 是 forward hook 实测整数，进金标。
    """
    from scripts.train_sft import KATAGO_SE_CFG as K
    n, f, ret = S.measure(dict(S.SE_CFG), probe_bs=2)
    assert n == K['params_total'] == 9_112_005, \
        '实测参数 {} != KATAGO_SE_CFG[params_total] {}'.format(
            n, K['params_total'])
    assert n == GOLDEN_PRESETS[4][1], 'SE 金标参数与实测不符'
    assert f == GOLDEN_PRESETS[4][2] > 0, 'SE 金标 FLOPs 与实测不符'
    assert ret > 0, '保留激活字节应为正'


def test_se_cfg_actually_builds_se_and_attn_blocks():
    """SE 预设必须真的建成 13×SEBottleneck + 4×AttentionResBlock 的主干。

    防「cfg 写着 se_bottleneck、实际建出 resnet」的静默回退：参数量对账抓得住
    宽度错，却抓不住「参数量恰好相同的另一种块」—— 直接数块类型才钉得住。
    块配比 = blocks - num_attention_layers（KATAGO_SE_CFG 的 17/4）。
    """
    from src.networks.se_bottleneck import SEBottleneck
    from src.networks.backbone import AttentionResBlock
    from scripts.train_sft import KATAGO_SE_CFG as K
    m = AlphaGoNet(in_channels=12, action_size=362,
                   attention_dropout=0.0, use_checkpoint=False, **dict(S.SE_CFG))
    blocks = list(m.backbone.blocks)
    n_se = sum(isinstance(b, SEBottleneck) for b in blocks)
    n_attn = sum(isinstance(b, AttentionResBlock) for b in blocks)
    assert len(blocks) == K['blocks'] == 17
    assert n_se == K['blocks'] - K['num_attention_layers'] == 13
    assert n_attn == K['num_attention_layers'] == 4
    assert sum(p.numel() for p in m.parameters()) == K['params_total']
    del m


def test_se_project_reports_budget_fields():
    """project(SE, batch) 的字段与口径：params/flops 实测，其余自 v18 锚点外推。

    **不**钉显存绝对值当金标（单点标定的外推值不配当金标）；钉住的是口径
    关系与两个训练目标的判定：
      * fits 与 SAFE_GB 一致、total = act + 常驻开销；
      * act ∝ batch（显存模型的线性外推性质）；
      * batch=1000 装得下（shell 的 BATCH=1000 有依据）、batch=3200 装不下
        —— 若哪天翻转，是结构或预算变了，要人来复核，不是测试静默通过。
    """
    cfg = dict(S.SE_CFG)
    r1 = S.project(cfg, 1000)
    r32 = S.project(cfg, 3200)
    for r in (r1, r32):
        assert set(r) == {'params', 'flops', 'act_gb', 'total_gb',
                          'step_s', 'samples_per_s', 'fits'}
        assert r['params'] == 9_112_005
        assert r['flops'] == 6_212_830_720
        assert r['fits'] == (r['total_gb'] <= S.SAFE_GB)
    assert r1['fits'], \
        'batch 1000 预测应装得下（否则 shell 的 BATCH=1000 无依据）'
    assert not r32['fits'], \
        'batch 3200 预测 {:.1f}GB 超预算 —— 结构/预算变了，需人工复核'.format(
            r32['total_gb'])
    assert abs(r32['act_gb'] / r1['act_gb'] - 3.2) < 0.01, \
        'act 必须随 batch 线性外推（3200/1000 = 3.2）'
    assert abs(r32['total_gb']
               - (r32['act_gb'] + S.overhead_gb(r32['params']))) < 1e-9


def test_emit_flags_se_branch_says_struct_flags_are_archived(capsys, monkeypatch):
    """--emit-flags 对 SE 预设不发旧结构 flag，明说结构归 KATAGO_SE_CFG。

    旧 flag（--backbone-channels 等）在 train_sft 已归档：接受但不参与建网。
    发它们等于发一套会被静默忽略的假旋钮。既有预设的发旗行为保持不变。
    """
    fake = dict(params=9_112_005, flops=6_212_830_720, act_gb=1.0,
                total_gb=1.0, step_s=1.0, samples_per_s=100.0, fits=True)
    monkeypatch.setattr(S, 'project', lambda *a, **k: dict(fake))
    monkeypatch.setattr(S, 'calibrate', lambda verbose=True: 0.0)
    S.main(['--preset', 'v18', '--emit-flags'])
    out = capsys.readouterr().out
    se_name, _, _, _ = S.preset_entries('se')[0]
    assert se_name in out, 'emit 输出里没有 SE 预设'
    se_block = out.split(se_name)[1]
    assert '无 flag 可发' in se_block and 'KATAGO_SE_CFG' in se_block, \
        'SE 分支没有说明结构 flag 已归档: {!r}'.format(se_block[:200])
    assert '--backbone-channels 240' not in out, \
        'SE 不该收到会被 train_sft 静默忽略的旧结构 flag'
    assert '--backbone-channels 192' in out, \
        '既有预设的 --emit-flags 行为被改动（仍应发旧 flag）'


def test_v21_machinery_is_retired():
    """v21 架构删除后，ANCHOR_V21 / run_anchor_v21 / --preset v21 必须退场。

    锚点曾是 P4.2 预算仲裁的静态表；结构从 src/networks/ 拔掉后，表留着只会
    让人以为还能量、还能投影。谁想复活 v21，先过本测试（改测试要写理由）。
    """
    assert not hasattr(S, 'ANCHOR_V21'), 'ANCHOR_V21 不该复活（v21 结构已删除）'
    assert not hasattr(S, 'run_anchor_v21'), 'run_anchor_v21 不该复活'
    assert 'v21' not in S.PRESET_CHOICES, 'v21 不该回到 --preset choices'
    with pytest.raises(SystemExit):
        # argparse 对非法 choice 走 parser.error -> SystemExit(2)
        S.build_parser().parse_args(['--preset', 'v21'])
    # SE 是它的接任者：现行训练结构在 choices 里
    assert hasattr(S, 'SE_CFG')
    assert 'se' in S.PRESET_CHOICES

