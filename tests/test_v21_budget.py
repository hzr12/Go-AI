"""v21 全网预算与接线锁（P4.2：`V21_CFG` / `V21Backbone` / `V21Net` / `build_v21_net`）。

分工
----
* `test_arch_v21_blocks.py` 锁**块类**（MambaLTI / TransformerBlock /
  CrossAttnRes / 两头）的逐项预算与既有类不动（D5）—— 本文件不重复它。
* `test_grad_checkpointing.py` 锁**检查点机制**（段划分、tap、开关持有者契约）——
  本文件只钉 v21 的默认开关取值，不重测机制。
* 本文件锁**组装出来的整网**与它的接线：
    1. 参数量硬数：主干 6,558,696 / 全网 9,067,443（不是窗口 —— 窗口会放过
       「某层 184→192 而总数不变」这类错误结构）；
    2. `V21_CFG` 与 `scripts/search_arch.ANCHOR_V21` **两份独立记录逐键相等**。
       按 P4.9 的规矩 search_arch 不得 import 网络代码（否则标定会被实现牵着
       走），所以对账只能放在这里：改任何一边都会红；
    3. 17ch 构建器已注册进 `src.inference`（GoAI 双代接缝）且过 stem 校验；
    4. 旧结构参数**保留定义但不参与构造**（D1 / 硬约束 C11）；
    5. 没有 `arch=='v21'` 分支、没有新增 CLI（D1）；
    6. grad checkpointing 默认开（用户 2026-09-27 裁决）、compile 组合下显式关。

全部 CPU、秒级：9×9 与 19×19 各一次前向，不加载真实权重、不读真实数据。
"""
import ast
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from src.networks.alphanet import (  # noqa: E402
    V21Backbone,
    V21_CFG,
    V21Net,
    build_v21_net,
)
from src.networks.backbone import (  # noqa: E402
    GC_CROSS_ATTN_RES,
    GC_LEGACY,
    GC_MAMBA,
    GC_RES,
    GC_TRANSFORMER,
)
import search_arch as S  # noqa: E402

# 锚（双份记录的另一侧就在这里；数字本身来自权威表，勿在此处改）
BACKBONE_PARAMS = 6_558_696
TOTAL_PARAMS = 9_067_443


def _n(mod):
    return sum(p.numel() for p in mod.parameters())


def _shapes(model):
    return [(k, tuple(v.shape)) for k, v in model.state_dict().items()]


def _build(action_size=362, **kw):
    return build_v21_net(action_size=action_size, **kw)


# --------------------------------------------------------------------------- #
# 1. 参数量硬数：整网 + 逐段（段名与 ANCHOR_V21['params'] 的键一一对应）
# --------------------------------------------------------------------------- #
def test_backbone_and_total_params_match_anchor():
    """主干 6,558,696、全网 9,067,443 —— 权威表给的硬数，精确相等。"""
    m = _build()
    assert _n(m.backbone) == BACKBONE_PARAMS == S.ANCHOR_V21['backbone_total']
    assert _n(m) == TOTAL_PARAMS == S.ANCHOR_V21['total_params']


def test_section_params_match_anchor_parts():
    """逐段对账：六段 + 两头，错一段就错一个数（比总数更早报错）。"""
    m = _build()
    b = m.backbone
    p = S.ANCHOR_V21['params']
    n_res, n_mamba, n_trans = b.n_res, b.n_mamba, b.n_trans
    assert _n(b.stem) + _n(b.stem_bn) == p['stem'] == 28_520
    assert _n(b.blocks[:n_res]) == p['res_blocks'] == 4_881_152
    assert _n(b.blocks[n_res:n_res + n_mamba]) == p['mamba2'] == 513_008
    assert _n(b.blocks[n_res + n_mamba:n_res + n_mamba + n_trans]) == \
        p['transformer'] == 448_960
    assert _n(b.blocks[n_res + n_mamba + n_trans:]) == p['cross_attn_res'] == 652_832
    assert _n(b.out) == p['out'] == 34_224
    assert _n(m.policy) == S.ANCHOR_V21['heads']['policy_params'] == 2_366_730
    assert _n(m.value) == S.ANCHOR_V21['heads']['value_params'] == 142_017
    # 锚点自身自洽（两份记录之一坏了，先在这一条上暴露）
    assert sum(p.values()) == S.ANCHOR_V21['backbone_total']
    h = S.ANCHOR_V21['heads']
    assert S.ANCHOR_V21['backbone_total'] + h['policy_params'] + h['value_params'] \
        == S.ANCHOR_V21['total_params']


# --------------------------------------------------------------------------- #
# 2. V21_CFG ↔ ANCHOR_V21 逐键对账（两份独立记录，改一边必红）
# --------------------------------------------------------------------------- #
def test_v21_cfg_matches_search_arch_anchor():
    a, lay = S.ANCHOR_V21, S.ANCHOR_V21['layout']
    assert V21_CFG['in_channels'] == a['in_channels'] == 17
    assert V21_CFG['channels'] == a['backbone_channels'] == 184
    assert (V21_CFG['n_res'], V21_CFG['n_mamba'],
            V21_CFG['n_trans'], V21_CFG['n_cross']) == (
        lay['res_blocks'], lay['mamba2_blocks'],
        lay['transformer_blocks'], lay['cross_attn_res_blocks']) == (8, 4, 2, 2)
    assert V21_CFG['ffn_hidden'] == lay['ffn_hidden'] == 240
    assert V21_CFG['num_heads'] == lay['transformer_heads'] == 4
    assert V21_CFG['params_backbone'] == a['backbone_total']
    assert V21_CFG['params_total'] == a['total_params']
    # stem 是 3×3（用户 2026-09-27 裁决）：从 CFG 建出来的卷积核就得是 3×3
    m = _build()
    assert tuple(m.backbone.stem.kernel_size) == (3, 3)
    assert m.backbone.stem.in_channels == V21_CFG['in_channels'] == 17


def test_v21_cfg_values_are_not_what_the_archived_shell_flags_say():
    """旧 shell 结构 flag（192 通道 / 17 res 等）**不是** v21 的结构来源。

    这条防的是「把 CFG 改成和某个旧 shell 一致」：D1 之下 shell 的结构 flag
    已归档，CFG 只认权威表锚点。
    """
    assert V21_CFG['channels'] != 192
    assert V21_CFG['n_res'] != 17


# --------------------------------------------------------------------------- #
# 3. 接线：17ch 构建器注册进 src.inference（GoAI 双代接缝），stem 过校验
# --------------------------------------------------------------------------- #
def test_17ch_builder_registered_and_passes_stem_check():
    from src import inference as I
    assert I._IN_CHANNEL_BUILDERS.get(17) is build_v21_net
    # 走 GoAI 建网的真实路径：含 stem 形状回读校验（构建器无视 in_channels 会在这里炸）
    m = I._build_for_in_channels(17, action_size=362)
    assert isinstance(m, V21Net)
    assert m.backbone.stem.in_channels == 17


def test_12ch_builder_still_the_legacy_net():
    """双代接缝的另一半：12ch 仍归旧 `AlphaGoNet`（P4.8 零回归）。"""
    from src import inference as I
    from src.networks.alphanet import AlphaGoNet
    assert I._IN_CHANNEL_BUILDERS.get(12) is I._legacy_alpha_go_net
    m = I._build_for_in_channels(12, action_size=82)
    assert type(m) is AlphaGoNet
    assert m.backbone.conv1.in_channels == 12


def test_wrong_in_channels_is_loud():
    """v21 没有 12ch/14ch 变体：传错通道数必须报错，不许静默换 stem。"""
    try:
        build_v21_net(in_channels=12, action_size=82)
    except ValueError as e:
        assert '17' in str(e)
    else:
        raise AssertionError('in_channels=12 应当被 V21 构建器拒绝')


# --------------------------------------------------------------------------- #
# 4. 旧结构参数：保留定义但不参与构造（D1 / C11）
# --------------------------------------------------------------------------- #
def test_archived_structure_flags_are_accepted_but_ignored():
    """`--arch convnext --backbone-channels 192 …` 全传进去，建出的还是同一张网。

    D1 原文：现有结构参数**保留定义但不参与构造**。这里用构建器等价于
    argparse 之后那条路径（train_sft 直接调 build_v21_net 并只透传行为参数；
    旧 shell 的 flag 落进 kwargs 后被忽略）。
    """
    base = _build()
    legacy = build_v21_net(
        action_size=362,
        arch='convnext', backbone_channels=192, backbone_res_blocks=17,
        res_blocks=17, convnext_blocks=5, attn_blocks=9,
        attention_mode='light', num_attention_layers=6, num_heads=8,
        attn_mode='linear', attn_window=64,
        policy_channels=96, value_channels=96, policy_layers=4,
        use_checkpoint=False, attention_dropout=0.3,
    )
    # 结构（state_dict 键与形状）与预算完全一致 = 结构 flag 一个都没生效
    assert _shapes(legacy) == _shapes(base)
    assert _n(legacy) == _n(base) == TOTAL_PARAMS
    # 行为参数仍生效：attention_dropout=0.3 被采纳（CFG 默认 0.1）
    assert legacy.backbone.cfg['attn_dropout'] == 0.3
    assert V21_CFG['attn_dropout'] == 0.1


def test_attention_dropout_alias_from_goai_is_honoured():
    """GoAI 侧键名是 `attention_dropout`：构建器必须对齐到 `attn_dropout`。"""
    m = build_v21_net(action_size=362, attention_dropout=0.42)
    assert m.backbone.cfg['attn_dropout'] == 0.42


# --------------------------------------------------------------------------- #
# 5. D1：没有 arch=='v21' 分支、没有新增 CLI（AST 级，防以后加回去）
# --------------------------------------------------------------------------- #
def _train_sft_tree():
    with open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
              encoding='utf-8') as f:
        return ast.parse(f.read())


def _add_argument_opts(tree):
    opts = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            continue
        kws = {k.arg: k.value for k in node.keywords}
        opts[node.args[0].value] = kws
    return opts


def test_no_v21_cli_and_old_flags_still_defined():
    """D1：不加任何新 CLI；旧结构参数必须**保留定义**（C11：旧 shell 可解析）。"""
    opts = _add_argument_opts(_train_sft_tree())
    assert not any('v21' in name for name in opts), \
        f'D1 禁止 v21 专属 CLI，发现 {[n for n in opts if "v21" in n]}'
    for keep in ('--arch', '--backbone-channels', '--res-blocks',
                 '--policy-layers', '--use-checkpoint'):
        assert keep in opts, f'{keep} 必须保留定义（否则旧 shell 解析即失败）'
    # --arch 的取值表里没有 'v21'（分支选择器禁止）
    choices = opts['--arch'].get('choices')
    assert choices is not None and 'v21' not in [
        c.value for c in choices.elts], 'arch 取值表混进了 v21'


def test_train_sft_builds_v21_from_cfg_not_legacy_net():
    """train_sft 的建网那一段：只认 V21_CFG + build_v21_net。"""
    with open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
              encoding='utf-8') as f:
        src = f.read()
    assert 'build_v21_net(' in src, 'train_sft 未改走 v21 构建器'
    assert 'AlphaGoNet(' not in src, 'train_sft 仍在建旧代网络'
    assert "V21_CFG['in_channels']" in src, '通道数没走 V21_CFG 常量'
    assert 'in_channels=12' not in src, 'train_sft 还有 12 通道硬编码'


def test_train_sft_warmup_dummies_use_v21_cfg_not_12():
    """C9 / P4.4：编译预热的 dummy 输入通道必须是 `V21_CFG['in_channels']`。

    硬编码 12 的后果不是「跑不起来」而是**静默回退**：v21 的 stem 吃 17 路，
    12 路输入在预热前向就形状错，异常被 except 吃掉后整条图编译路径降级 eager，
    日志上只剩一句「回退 eager」。
    """
    with open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
              encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)

    def _zeros_second_arg_is(node, pred):
        if not (isinstance(node, ast.Call) and len(node.args) >= 2):
            return False
        f = node.func
        if isinstance(f, ast.Name) and f.id == 'zeros':
            pass
        elif isinstance(f, ast.Attribute) and f.attr == 'zeros':
            pass
        else:
            return False
        return pred(node.args[1])

    dummy12 = [n for n in ast.walk(tree)
               if _zeros_second_arg_is(
                   n, lambda a: isinstance(a, ast.Constant) and a.value == 12)]
    assert not dummy12, \
        f'train_sft 还有硬编码 12 通道的 zeros 输入: 行 {[d.lineno for d in dummy12]}'
    dummy_cfg = [n for n in ast.walk(tree)
                 if _zeros_second_arg_is(n, lambda a: isinstance(a, ast.Subscript))]
    assert len(dummy_cfg) >= 2, \
        f'预热 dummy 应有两处（TorchAir + torch.compile），实得 {len(dummy_cfg)}'
    for d in dummy_cfg:
        assert 'V21_CFG' in ast.unparse(d.args[1]), ast.unparse(d)


def _planes_calls(path):
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'feature_planes_batched'):
            for kw in node.keywords:
                if kw.arg == 'n_channels':
                    yield node, kw.value


def test_selfplay_and_async_planes_follow_the_model_not_a_literal():
    """P4.2 释放的两处 12 钉子：采集通道数必须随 `ai.in_channels` 走。

    写死 12 的后果是**形状错**（buffer 17 路 vs 模型 12 路），不是静默错值；
    写死 17 又会打死旧权重的自对弈。两边都兼容的唯一写法就是随模型驱动。
    """
    for name in ('selfplay_train.py', 'async_pipeline.py'):
        calls = list(_planes_calls(os.path.join(ROOT, 'scripts', name)))
        assert calls, f'{name} 找不到 feature_planes_batched 调用'
        for node, val in calls:
            assert not (isinstance(val, ast.Constant)
                        and isinstance(val.value, int)), \
                f'{name}:{node.lineno} 的 n_channels 还是字面量 {val.value}'
            assert isinstance(val, ast.Attribute) and val.attr == 'in_channels', \
                f'{name}:{node.lineno} 的 n_channels 应取 ai.in_channels'


def _dataset_calls(tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'SupervisedDataset'):
            continue
        kws = {k.arg: k.value for k in node.keywords if k.arg}
        yield node, kws


def test_train_sft_dataset_channels_follow_v21_cfg():
    """数据侧与模型侧必须**同时**改（dataset.py 的 C7 注释点名的那条耦合）。

    模型恒 v21 = 17 路 stem，而 `SupervisedDataset` 的默认 `n_channels=12`
    （给旧权重回归留的）。train_sft 里两处构造点（单文件 / 目录合并）只要有一处
    忘传 `n_channels=V21_CFG['in_channels']`，第一个 batch 就是 12 路平面喂
    17 路 stem —— 训练启动即形状错。
    """
    with open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
              encoding='utf-8') as f:
        tree = ast.parse(f.read())
    calls = list(_dataset_calls(tree))
    assert len(calls) >= 2, f'应有两处 SupervisedDataset 构造，实得 {len(calls)}'
    for node, kws in calls:
        nc = kws.get('n_channels')
        assert nc is not None, \
            f'train_sft.py:{node.lineno} 的 SupervisedDataset 没传 n_channels'
        assert ast.unparse(nc) == "V21_CFG['in_channels']", \
            f'train_sft.py:{node.lineno} 的 n_channels 应为 V21_CFG，实为 {ast.unparse(nc)}'


def test_one_sft_step_on_17ch_planes_trains_v21():
    """端到端冒烟：合成样本 → 17 路平面 → v21 前向 + 反向（训练真能跑起来）。

    本文件其余用例都是静态/形状断言；这一条真的走一遍 train_sft 的数据流，
    抓「数据侧 12 / 模型侧 17」这类只有第一个 batch 才会暴露的接线错。
    """
    import numpy as np
    from src.data.dataset import SupervisedDataset

    n = 4
    data = {
        'boards': np.zeros((n, 9, 9), dtype=np.int8),
        'my_hist': np.full((n, 3), -1, dtype=np.int16),
        'op_hist': np.full((n, 3), -1, dtype=np.int16),
        'ko': np.full((n,), -1, dtype=np.int16),
        'moves': np.arange(n, dtype=np.int16),
        'values': np.ones((n,), dtype=np.int8),
        'to_play': np.ones((n,), dtype=np.int8),
    }
    ds = SupervisedDataset(data, n_channels=V21_CFG['in_channels'])
    states, moves, vals = ds.sample_batch_numpy(np.arange(n), augment=False)
    assert states.shape == (n, 17, 9, 9), states.shape
    assert moves.shape == (n,) and vals.shape == (n, 1)

    m = build_v21_net(action_size=9 * 9 + 1)
    m.train()
    logits, value = m(torch.from_numpy(states))
    loss = (torch.nn.functional.cross_entropy(
                logits, torch.from_numpy(moves.astype(np.int64)))
            + (value.squeeze(-1) - torch.from_numpy(vals[:, 0])).pow(2).mean())
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, '反向后没有任何梯度 —— 图没接上'
    assert all(torch.isfinite(g).all() for g in grads), '梯度里有 NaN/Inf'


# --------------------------------------------------------------------------- #
# 6. grad checkpointing：默认开（用户裁决）、compile 组合下关、eval 惯性
# --------------------------------------------------------------------------- #
def test_grad_checkpointing_defaults_match_user_ruling():
    m = _build()
    b = m.backbone
    assert b.grad_checkpointing_for(GC_RES) is True
    assert b.grad_checkpointing_for(GC_MAMBA) is True
    assert b.grad_checkpointing_for(GC_TRANSFORMER) is True
    # 性价比不合算的两格（P4.6b 裁决）：默认不开
    assert b.grad_checkpointing_for(GC_CROSS_ATTN_RES) is False
    assert b.grad_checkpointing_for(GC_LEGACY) is False


def test_grad_checkpointing_off_when_compile_wins():
    """compile=1 的组合下 train_sft 传 grad_checkpoint=0（P4.6b §8.2④）。"""
    m = _build(grad_checkpoint=0)
    b = m.backbone
    assert b.grad_checkpointing_for(GC_RES) is False
    assert b.grad_checkpointing_for(GC_MAMBA) is False


def test_eval_mode_is_where_inference_never_checkpoints():
    m = _build(action_size=82).eval()
    assert m.training is False and m.backbone.training is False
    with torch.no_grad():
        p, v = m(torch.randn(1, 17, 9, 9))
    assert p.shape == (1, 82) and v.shape == (1, 1)


# --------------------------------------------------------------------------- #
# 7. 前向形状：两代棋盘尺寸（action_size = n²+1）
# --------------------------------------------------------------------------- #
def test_forward_shapes_board9_and_board19():
    for n in (82, 362):
        board = int(round((n - 1) ** 0.5))
        m = _build(action_size=n).eval()
        with torch.no_grad():
            p, v = m(torch.randn(2, 17, board, board))
        assert p.shape == (2, n), f'board={board}: policy {p.shape}'
        assert v.shape == (2, 1), f'board={board}: value {v.shape}'


def test_action_size_must_be_board_squared_plus_one():
    try:
        _build(action_size=100)
    except ValueError as e:
        assert 'action_size' in str(e)
    else:
        raise AssertionError('action_size=100 不是 n²+1，应当被拒绝')


def test_backbone_taps_satisfy_cross_attn_contract():
    """抽头 s1/s5/s9 = Res#1/Res#5/Mamba#1，喂给 CrossAttnRes 的 tap_channels。"""
    m = _build()
    b = m.backbone
    assert (b.n_res, b.n_mamba, b.n_trans) == (8, 4, 2)
    x = torch.randn(1, 17, 9, 9)
    b.eval()
    with torch.no_grad():
        taps = []
        hooks = []
        for i, blk in enumerate(b.blocks):
            if i in (0, 4, 8):
                hooks.append(blk.register_forward_hook(
                    lambda mod, inp, out, _l=taps: _l.append(out.shape)))
        b(x)
        for h in hooks:
            h.remove()
    assert [t[1] for t in taps] == [184, 184, 184], taps
