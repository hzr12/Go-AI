"""KataGo 风格 SE-bottleneck 残差块的契约测试。

这个块要守住的四件事（都是 KataGo v2+ 的刻意选择，不是风格偏好）
----------------------------------------------------------------
1. **pre-activation**：BN 在 conv **之前**。`ResBlock` 是 post-activation，两者在
   同一份权重下**数值不同** —— 所以「顺序」必须被真的观察到，不能只靠 AST 读源码。
2. **残差支路没有 BN**：`conv3` 之后直接进 SE、再加 x。KataGo v2 起刻意去掉残差
   支路的 BN —— BN 的 running 统计在按 group 采样的训练里不稳，残差支路是它的
   放大器（每多一个块就多两次统计更新）。
3. **SE 门控真的在乘**：门控 sigmoid ∈ (0,1) 逐通道缩放 `conv3` 的输出。用常数场
   把它钉成精确值（`y == 0.5*r + x`），而不是「输出变了」这种弱判据。
4. **参数量**：C=160 上 **87,050**，约为 `ResBlock(160)`（461,440）的 1/5.3。
   这是 9.25M 总预算能装下 17 个块的前提。

接线面（`SharedBackbone(arch="se_bottleneck")`）也在这里 —— 但**默认路径逐位不变**
的守护交给 `tests/test_katago_se.py::test_legacy_class_signatures_untouched`，
本文件只加一条「另外两条 arch 没被顺手改」的对照。

梯度检查点的 BN 机制（`_BatchNormStatGuard`）在 `tests/test_grad_checkpointing.py`
里有机制级覆盖；本文件只保证**新块的 BN 落进那条机制的范围**（能被
`_collect_batchnorms` 收进来、重算后统计逐位还原），不重复它的机制断言。

全部 CPU、秒级：不加载真实权重、不读真实数据。
"""
import ast
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.networks.alphanet import AlphaGoNet  # noqa: E402
from src.networks.backbone import (  # noqa: E402
    ConvNeXtBlock,
    ResBlock,
    SharedBackbone,
)
from src.networks.se_bottleneck import (  # noqa: E402
    SE_REDUCTION_DEFAULT,
    SEBottleneck,
    SEGate,
)

SE_ARCH = 'se_bottleneck'
#: C=160 的实测值（逐项：conv1 12,800 + bn1 320 + conv2 57,600 + bn2 160
#: + conv3 12,800 + se.fc1 1,610 + se.fc2 1,760）。
SE_PARAMS_AT_160 = 87050
#: 同宽度的 `ResBlock` 对照值（2×(160×160×9) + 2×320）。
RES_PARAMS_AT_160 = 461440


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _bn_snapshot(mod):
    """所有 BN 的 (running_mean, running_var, num_batches_tracked) 快照。"""
    out = {}
    for name, m in mod.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            out[name] = (m.running_mean.clone(), m.running_var.clone(),
                         int(m.num_batches_tracked))
    return out


def _bn_layers(mod):
    return [m for m in mod.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def _residual_branch(blk, x):
    """**独立复算** conv3 分支（不含 SE、不含残差加法）。

    测试自己重写一遍前向（而不是调块内部的任何辅助方法），这样
    「门控乘在 conv3 的输出上」才是被验证的事实，而不是实现自说自话。
    """
    h = F.relu(blk.bn1(x))
    h = blk.conv1(h)
    h = F.relu(blk.bn2(h))
    h = blk.conv2(h)
    return blk.conv3(h)


def _block_forward_ast():
    """`SEBottleneck.forward` 的 AST（源码而非运行时的旁证）。"""
    path = os.path.join(ROOT, 'src', 'networks', 'se_bottleneck.py')
    tree = ast.parse(open(path, encoding='utf-8').read())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'SEBottleneck')
    fn = next(f for f in cls.body
              if isinstance(f, ast.FunctionDef) and f.name == 'forward')
    return path, cls, fn


def _self_attr_lines(fn):
    """`forward` 里每个 `self.<name>` 首次出现的行号。"""
    lines = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == 'self'):
            lines[node.attr] = min(lines.get(node.attr, 1 << 30), node.lineno)
    return lines


# =========================================================================== #
# 一、形状 / dtype
# =========================================================================== #
def test_forward_shape_dtype_and_finiteness_at_target_width():
    """目标配置 (B=2, C=160, 19×19) 下的前向契约。"""
    torch.manual_seed(3)
    blk = SEBottleneck(160)
    x = torch.randn(2, 160, 19, 19)
    y = blk(x)
    assert y.shape == x.shape, y.shape
    assert y.dtype == torch.float32, y.dtype
    assert torch.isfinite(y).all(), '输出出现 inf/nan'


def test_forward_preserves_module_dtype():
    """输出 dtype 跟**模块**走，块内不许有手写的 fp32 兜底。

    反面判据是那种「为稳定起见把输出 `.float()` 一下」的写法：AMP fp16 下它会把
    激活在输出端翻回 fp32，体积翻倍 —— 910A 的 4 卡 OOM 就是这么来的。这里把
    整个块 `.double()`，输出必须仍是 float64。
    """
    torch.manual_seed(4)
    blk = SEBottleneck(16).eval().double()
    y = blk(torch.randn(2, 16, 5, 5, dtype=torch.float64))
    assert y.dtype == torch.float64, y.dtype
    assert y.shape == (2, 16, 5, 5)


def test_forward_under_autocast_fp16_is_finite():
    """AMP fp16 下前向不炸、不出 nan（910A 无 bf16，这条是那条路的门槛）。"""
    torch.manual_seed(5)
    blk = SEBottleneck(32).eval()
    x = torch.randn(2, 32, 7, 7)
    with torch.autocast(device_type='cpu', dtype=torch.float16):
        y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y.float()).all()


# =========================================================================== #
# 二、SE 门控确实在起作用
# =========================================================================== #
def test_se_gate_is_bounded_and_below_one_at_init():
    """门控形状是 (B, C, 1, 1)，取值严格落在 (0,1) 且随机初始化下**压得住**通道。"""
    torch.manual_seed(11)
    blk = SEBottleneck(32).eval()
    x = torch.randn(4, 32, 7, 7)
    with torch.no_grad():
        g = blk.se(x)
        lo, hi = float(g.min()), float(g.max())
    assert g.shape == (4, 32, 1, 1), g.shape
    assert bool((g > 0).all()) and bool((g < 1).all())
    # 若门控恒为 1（等价于没接 SE），这条会红 —— 判据是「有通道被明显压制」
    assert lo < 0.9, '门控在随机初始化下没有压制任何通道：min=%.4f' % lo
    assert hi > 0.1


def test_se_gate_scales_the_conv3_output_by_an_exact_factor():
    """常数场上把门控钉成精确常数，验证「输出 = conv3 分支 × 门控 + x」。

    做法：`se.fc2` 归零 ⇒ 第二层输出恒 0 ⇒ `sigmoid(0) = 0.5` **精确**。于是
    `y == 0.5 * r + x` 是逐位断言，不需要 `assert_allclose` 的容差。
    换成 `se.fc2.bias = 30`（sigmoid 饱和到 1.0）则必须得到 `y == r + x` ——
    两个精确值之间的差就是门控的作用量。
    """
    torch.manual_seed(12)
    blk = SEBottleneck(16).eval()
    x = torch.full((2, 16, 5, 5), 0.25)   # 常数场：空间维处处相等 ⇒ 解析可算

    r = _residual_branch(blk, x)

    with torch.no_grad():
        blk.se.fc2.weight.zero_()
        blk.se.fc2.bias.zero_()
    torch.testing.assert_close(blk(x), r * 0.5 + x, rtol=0, atol=0)

    with torch.no_grad():
        blk.se.fc2.bias.fill_(30.0)
    torch.testing.assert_close(blk(x), r + x, rtol=0, atol=0)


def test_se_gate_is_computed_from_the_block_input_not_conv3_output():
    """门控从**块入口 x** 算（KataGo 的 `applySE(input, afterConv3, ...)`）。

    变异守卫：`out = out * self.se(out)`。这里**不**把门控钉成常数 —— 那样两种
    接法都给 0.5，判别不出来。改为随机初始化下的精确复算：
    `y == r * se(x) + x` 逐位成立；再单独断言 `se(x) != se(r)`，坐实两个候选
    输入在这个输入上确实不同（否则第一条断言就退化成恒等式，守不住任何东西）。

     通道数取 160 而不是小值：SE 的隐藏层宽度是 `C // r`，`C=16, r=16` 时隐藏
    层只有 1 维、`fc1` 的输出被 ReLU 整片夹到 0 ⇒ 门控退化成与输入无关的常数
    `sigmoid(b2)`，任何判别式都会空转绿。160 也正是生产形状（隐藏层 10 维）。
    """
    torch.manual_seed(13)
    blk = SEBottleneck(160).eval()
    with torch.no_grad():
        # 把 ReLU 拉进激活区：随机初始化的 fc1 偏置在 160 通道下可能整片为负，
        # 那会让门控退化成常数（同上），判别式再次空转。
        blk.se.fc1.bias.fill_(1.0)
    x = torch.full((2, 160, 5, 5), 0.25)   # 常数场：空间维处处相等 ⇒ 解析可算

    with torch.no_grad():
        r = _residual_branch(blk, x)
        gate_in = blk.se(x)
        gate_out = blk.se(r)
        y = blk(x)

    assert blk.se.hidden == 10, blk.se.hidden
    assert not torch.allclose(gate_in, gate_out, rtol=1e-4, atol=1e-5), \
        '两个候选输入给出同一个门控 —— 本测试退化成恒等式，守不住任何东西'
    torch.testing.assert_close(y, r * gate_in + x, rtol=0, atol=0)


def test_se_gate_reduction_defaults_to_16():
    """默认 reduction=16，可配。"""
    assert SE_REDUCTION_DEFAULT == 16
    blk = SEBottleneck(160)
    assert blk.se.fc1.out_features == 160 // 16 == 10
    assert blk.se.fc1.in_features == 160
    assert blk.se.fc2.in_features == 10 and blk.se.fc2.out_features == 160
    # 可配：reduction=8 → 隐藏层 20；reduction 超过通道数时兜底到 1 维
    assert SEBottleneck(160, se_reduction=8).se.fc1.out_features == 20
    assert SEBottleneck(160, se_reduction=160).se.fc1.out_features == 1
    # 用 320 而不是 160：`160 // 160 == 1`，删掉下限那行这条仍会绿
    assert SEBottleneck(160, se_reduction=320).se.fc1.out_features == 1


# =========================================================================== #
# 三、结构：pre-activation 与「残差支路无 BN」
# =========================================================================== #
def test_pre_activation_bn_precedes_its_conv():
    """AST：`bn1` 在 `conv1` 之前、`bn2` 在 `conv2` 之前（pre-activation）。"""
    _, _, fn = _block_forward_ast()
    at = _self_attr_lines(fn)
    assert {'bn1', 'conv1', 'bn2', 'conv2', 'conv3'} <= set(at), sorted(at)
    for bn, conv in (('bn1', 'conv1'), ('bn2', 'conv2')):
        assert at[bn] < at[conv], (
            '%s:%d 出现在 %s:%d 之后 —— BN 在 conv 之后就是 post-activation 了'
            % ('bn', at[bn], conv, at[conv]))


def test_bn1_cannot_be_post_activation_shape_wise():
    """`bn1` 的 pre-activation 顺序**由形状钉死**：它吃 `channels`，`conv1` 出 `mid`。

    所以 `bn1` 的顺序不是自由变量 —— post-activation 的接法在形状上根本建不起来。
    这条把「`bn1` 可以在后面」这种变异当场挡住（而不只是靠源码行号）。
    """
    blk = SEBottleneck(16)
    assert blk.bn1.num_features == blk.conv1.in_channels == 16
    assert blk.conv1.out_channels == 8, '瓶颈压缩没了，bn1 的形状钉就不成立'
    with pytest.raises(RuntimeError):
        blk.bn1(blk.conv1(torch.randn(2, 16, 5, 5)))   # 8 通道喂 16 通道的 BN


def test_bn2_conv2_order_is_observable_not_merely_declarative():
    """`bn2` / `conv2` 在通道数上**完全可交换**（都吃 mid）⇒ 顺序是自由变量。

    只读源码的顺序断言在这里有个盲区：万一 conv/BN 数值上恰好可交换，那条断言就
    守不住任何东西。所以用**同一份权重**按 `ResBlock` 那种 post-activation 顺序
    复算一遍，输出必须与前向不等。
    """
    torch.manual_seed(14)
    blk = SEBottleneck(16).eval()
    x = torch.randn(2, 16, 5, 5)
    got = blk(x)

    h = F.relu(blk.bn1(x))
    h = blk.conv1(h)
    h = F.relu(blk.bn2(blk.conv2(h)))    # post-activation：conv → BN → ReLU
    h = blk.conv3(h)
    alt = h * blk.se(x) + x
    assert not torch.allclose(got, alt, rtol=1e-4, atol=1e-5), \
        'pre-activation 与 post-activation 输出相同 —— 顺序测试守不住任何东西'


def test_residual_branch_has_no_batchnorm():
    """残差支路（`conv3` 之后）不得有 BN —— 且全块只有 2 个 BN。"""
    blk = SEBottleneck(24)
    assert len(_bn_layers(blk)) == 2, \
        '期望 bn1/bn2 两个 BN，实际 %d 个：%s' % (
            len(_bn_layers(blk)), [type(m).__name__ for m in _bn_layers(blk)])
    assert not hasattr(blk, 'bn3')

    _, _, fn = _block_forward_ast()
    at = _self_attr_lines(fn)
    after = [a for a, ln in at.items() if ln > at['conv3']]
    assert not [a for a in after if a.startswith('bn')], \
        'conv3 之后仍出现了 BN：%s' % after


def test_block_is_pre_activation_and_has_no_zero_init_shortcut():
    """KataGo 语义 vs ResBlock 的 zero-init 恒等捷径：SE 块不做那个初始化。"""
    torch.manual_seed(15)
    res = ResBlock(32).eval()
    assert torch.allclose(res(torch.ones(2, 32, 5, 5) * 0.5),
                          torch.ones(2, 32, 5, 5) * 0.5, atol=1e-5), \
        'ResBlock 的 zero-init 前提失效（对照组本身坏了）'
    se = SEBottleneck(32).eval()
    x = torch.ones(2, 32, 5, 5) * 0.5
    assert not torch.allclose(se(x), x, atol=1e-3), \
        'SE 块在初始化时就是恒等映射 —— 说明有人给它加了 zero-init'


# =========================================================================== #
# 四、卷积规格与参数量
# =========================================================================== #
def test_conv_specs_match_the_katago_bottleneck():
    """1×1(C→mid) → 3×3(mid→mid) → 1×1(mid→C)，全 bias=False，mid 默认 C//2。"""
    blk = SEBottleneck(160)
    assert blk.mid_channels == 80, blk.mid_channels
    assert blk.conv1.kernel_size == (1, 1) and blk.conv1.in_channels == 160 \
        and blk.conv1.out_channels == 80 and blk.conv1.bias is None
    assert blk.conv2.kernel_size == (3, 3) and blk.conv2.padding == (1, 1) \
        and blk.conv2.in_channels == 80 and blk.conv2.out_channels == 80 \
        and blk.conv2.bias is None
    assert blk.conv3.kernel_size == (1, 1) and blk.conv3.in_channels == 80 \
        and blk.conv3.out_channels == 160 and blk.conv3.bias is None
    # mid 可配
    assert SEBottleneck(160, mid_channels=40).conv1.out_channels == 40


def test_parameter_count_at_c160_is_87050():
    """C=160 实测 87,050 参数；同时钉住「比 ResBlock 小一个量级」这个前提。"""
    torch.manual_seed(16)
    blk = SEBottleneck(160)
    n = sum(p.numel() for p in blk.parameters())
    print('\n[se_bottleneck] C=160 参数量 = %d' % n)
    assert n == SE_PARAMS_AT_160, '参数量实测 %d != %d' % (n, SE_PARAMS_AT_160)
    # 范围断言（预算口径留的余量）
    assert 86_000 <= n <= 88_000, n
    # SE 是可省的：否则 9.25M 装不下 17 个块
    r = sum(p.numel() for p in ResBlock(160).parameters())
    assert r == RES_PARAMS_AT_160, r
    assert n < r / 5, 'SE 块 %d 参数，ResBlock %d —— 比例 %0.2f，不是省下来的' % (n, r, r / n)


def test_parameter_breakdown_matches_the_budget_table():
    """逐子模块对账：哪一项吃掉了参数，一处都不许悄悄多长。"""
    blk = SEBottleneck(160)

    def np_(m):
        return sum(p.numel() for p in m.parameters())

    assert np_(blk.conv1) == 12_800
    assert np_(blk.bn1) == 320
    assert np_(blk.conv2) == 57_600
    assert np_(blk.bn2) == 160
    assert np_(blk.conv3) == 12_800
    assert np_(blk.se) == 3_370
    assert isinstance(blk.se, SEGate)


# =========================================================================== #
# 五、与 ResBlock 的行为差异（不许只是重命名）
# =========================================================================== #
def test_behaviour_differs_from_resblock():
    """同一输入下两条路径形状相同、数值/符号/规模都不同。"""
    torch.manual_seed(17)
    se = SEBottleneck(32).eval()
    res = ResBlock(32).eval()
    x = torch.randn(2, 32, 5, 5)

    ys, yr = se(x), res(x)
    assert ys.shape == yr.shape
    assert not torch.allclose(ys, yr, rtol=1e-3, atol=1e-4)

    with torch.no_grad():
        lo_s, lo_r = float(ys.min()), float(yr.min())
    # `ResBlock` 末尾有 ReLU ⇒ 输出非负；pre-activation + 无残差 BN 的 SE 块不是
    assert lo_r >= 0.0
    assert lo_s < 0.0, 'SE 块输出全非负 —— 它被 ReLU 了，不合 pre-activation 语义'

    # 结构面：卷积核尺寸与通道走向
    assert res.conv1.kernel_size == (3, 3) and res.conv1.in_channels == 32
    assert se.conv1.kernel_size == (1, 1) and se.conv1.out_channels == 16
    assert not hasattr(res, 'se') and hasattr(se, 'se')


def test_se_block_is_scale_equivariant_under_bn_in_eval():
    """eval 下 BN 退化成固定仿射 ⇒ SE 块对正数逐通道缩放是**可解析**的。

    用常数场把「门控是唯一的非线性来源之一」钉住：把门控钉成 0.5 之后，
    输出必须精确等于 `0.5*r + x`，与 x 的绝对大小无关（线性段）。
    """
    torch.manual_seed(18)
    blk = SEBottleneck(16).eval()
    with torch.no_grad():
        blk.se.fc2.weight.zero_()
        blk.se.fc2.bias.zero_()
    for v in (0.1, 0.25, 2.0):
        x = torch.full((2, 16, 5, 5), v)
        torch.testing.assert_close(blk(x), _residual_branch(blk, x) * 0.5 + x,
                                   rtol=1e-5, atol=1e-6, msg='v=%g' % v)


# =========================================================================== #
# 六、eval / train 的 BN 统计
# =========================================================================== #
def test_train_mode_updates_running_stats_but_eval_does_not():
    """train 下每块 BN 的统计各前进一次；eval 下**一次都不许**动。"""
    torch.manual_seed(19)
    blk = SEBottleneck(8)
    assert len(_bn_layers(blk)) == 2

    blk.train()
    blk(torch.randn(4, 8, 5, 5))
    trained = _bn_snapshot(blk)
    for name, (_, _, n) in trained.items():
        assert n == 1, '%s: num_batches_tracked=%d（train 下应各更新一次）' % (name, n)

    blk.eval()
    before = _bn_snapshot(blk)
    for seed in range(3):
        torch.manual_seed(100 + seed)
        blk(torch.randn(4, 8, 5, 5))
    after = _bn_snapshot(blk)
    assert set(after) == set(before)
    for name in before:
        assert torch.equal(after[name][0], before[name][0]), f'{name}: eval 下 running_mean 被更新'
        assert torch.equal(after[name][1], before[name][1]), f'{name}: eval 下 running_var 被更新'
        assert after[name][2] == before[name][2], f'{name}: eval 下 num_batches_tracked 前进'
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
               for a, b in ((trained[k], after[k]) for k in trained)), \
        'eval 前后的快照与 train 之后那次不一致'


def test_bn_running_var_is_fp32_under_autocast():
    """AMP fp16 下 BN 的 buffer 仍是 fp32（autocast 自己处理，别显式 cast）。"""
    torch.manual_seed(20)
    blk = SEBottleneck(16)
    blk(torch.randn(4, 16, 5, 5))   # 先在 fp32 下让统计非平凡
    with torch.autocast(device_type='cpu', dtype=torch.float16):
        blk(torch.randn(4, 16, 5, 5))
    for m in _bn_layers(blk):
        assert m.running_var.dtype == torch.float32, m.running_var.dtype
        assert m.running_mean.dtype == torch.float32, m.running_mean.dtype
        assert m.num_batches_tracked.dtype == torch.int64, m.num_batches_tracked.dtype


# =========================================================================== #
# 七、梯度检查点的 BN 覆盖（`_BatchNormStatGuard` 机制对新块生效）
# =========================================================================== #
def test_block_batchnorms_are_collected_by_the_checkpoint_helper():
    """新块的 BN 必须落进 `_collect_batchnorms` 的口径（否则 guard 形同虚设）。"""
    from src.networks.backbone import _collect_batchnorms

    blk = SEBottleneck(16)
    assert len(_collect_batchnorms([blk])) == 2


def test_grad_checkpointing_leaves_bn_stats_bit_identical():
    """开检查点：重算不得让 BN 统计前进两次（guard 的既有语义对新块成立）。"""
    from src.networks.backbone import GC_LEGACY, run_grad_segment

    torch.manual_seed(21)
    blocks = [SEBottleneck(16), SEBottleneck(16), SEBottleneck(16)]
    x = torch.randn(4, 16, 5, 5)

    def run(use_checkpoint):
        for b in blocks:                     # 同一份权重，只让 BN 统计走一遍
            b.train()
            b.bn1.reset_running_stats()
            b.bn2.reset_running_stats()
        return run_grad_segment(blocks, (x,), use_checkpoint=use_checkpoint,
                                per_block=True, uncapped_last=True, kind=GC_LEGACY)

    off, _ = run(False)
    snap_off = [(b.bn1.num_batches_tracked.clone(), b.bn2.num_batches_tracked.clone(),
                 b.bn1.running_var.clone(), b.bn2.running_var.clone()) for b in blocks]
    on, _ = run(True)
    snap_on = [(b.bn1.num_batches_tracked.clone(), b.bn2.num_batches_tracked.clone(),
                b.bn1.running_var.clone(), b.bn2.running_var.clone()) for b in blocks]
    assert torch.equal(off, on), '开检查点改了前向输出（max|Δ|=%.3e）' % (off - on).abs().max()
    for i, (a, b) in enumerate(zip(snap_off, snap_on)):
        for j, (u, v) in enumerate(zip(a, b)):
            assert torch.equal(u, v), '块%d 第%d 项 BN 统计被重算污染' % (i, j)


# =========================================================================== #
# 八、接进 SharedBackbone（默认路径必须逐位不变）
# =========================================================================== #
def test_shared_backbone_accepts_se_bottleneck_in_none_mode():
    from src.networks.backbone import SharedBackbone

    bb = SharedBackbone(in_channels=12, channels=32, num_res_blocks=4,
                        attention_mode='none', arch=SE_ARCH)
    assert len(bb.blocks) == 4
    assert all(isinstance(b, SEBottleneck) for b in bb.blocks)
    # stem / 输出层走的是 resnet 那条（3×3 conv + BN + ReLU）—— KataGo 的形态，
    # 所以 `forward` 里不需要任何 arch 分支改动。
    assert hasattr(bb, 'bn1') and hasattr(bb, 'bn_out')
    y = bb(torch.randn(2, 12, 19, 19))
    assert y.shape == (2, 32, 19, 19)


def test_shared_backbone_mix_mode_keeps_attention_positions():
    """mix 模式：注意力位置仍是 `AttentionResBlock`，其余换成 `SEBottleneck`。"""
    bb = SharedBackbone(in_channels=12, channels=32, num_res_blocks=6,
                        attention_mode='mix', num_attention_layers=2,
                        num_heads=4, arch=SE_ARCH)
    kinds = [type(b).__name__ for b in bb.blocks]
    assert kinds.count('AttentionResBlock') == 2, kinds
    assert kinds.count('SEBottleneck') == 4, kinds


def test_shared_backbone_other_arch_paths_are_untouched():
    """对照组：resnet / convnext 两条老路仍装各自的块。"""
    for arch, cls in (('resnet', ResBlock), ('convnext', ConvNeXtBlock)):
        bb = SharedBackbone(in_channels=12, channels=16, num_res_blocks=3,
                            attention_mode='none', arch=arch)
        assert len(bb.blocks) == 3
        assert all(isinstance(b, cls) for b in bb.blocks), (arch, bb.blocks)


def test_alphanet_end_to_end_with_se_bottleneck():
    """端到端：`AlphaGoNet(arch="se_bottleneck")` 出 policy + value，形状正确。"""
    torch.manual_seed(22)
    net = AlphaGoNet(in_channels=12, backbone_channels=32, backbone_res_blocks=4,
                     attention_mode='none', action_size=362, arch=SE_ARCH)
    pol, val = net(torch.randn(2, 12, 19, 19))
    assert pol.shape == (2, 362) and val.shape == (2, 1)
    assert torch.isfinite(pol).all() and torch.isfinite(val).all()
    assert sum(1 for p in net.parameters()) > 0


def test_alphanet_default_arch_is_untouched():
    """默认 arch 仍走 `ResBlock`（`AlphaGoNet` 的签名一个字节都没动）。"""
    net = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=3,
                     attention_mode='none', action_size=82)
    assert net.backbone.arch == 'resnet'
    assert all(isinstance(b, ResBlock) for b in net.backbone.blocks)


def test_se_bottleneck_backbone_param_total_at_v18_target():
    """v18 目标形状 (C=160, 17 块, mix/4 注意力, value_res_blocks=2) 的实测装配量。

     **这条记录的数与任务书里的 9.25M 不是一回事**：同一形状下 `arch="resnet"`
    实测 8.85M（全网），`arch="convnext"` 5.57M，`arch="se_bottleneck"` 只有
    **3.99M**。也就是 SE 块在这个形状上把参数量砍到 resnet 的 0.45 倍 —— 9.25M
    那个标定值几乎肯定是 resnet 路径测出来的。想把 SE 路径的预算填到 9.25M，
    得加块数或加宽通道，不是改 SE 块本身。给 search_arch 的人一个对照数。
    """
    net = AlphaGoNet(in_channels=12, backbone_channels=160, backbone_res_blocks=17,
                     attention_mode='mix', num_attention_layers=4, num_heads=4,
                     action_size=362, arch=SE_ARCH, value_res_blocks=2)
    n_se = sum(1 for b in net.backbone.blocks if isinstance(b, SEBottleneck))
    assert n_se == 17 - 4, n_se
    total = sum(p.numel() for p in net.parameters())
    backbone = sum(p.numel() for p in net.backbone.parameters())
    print('\n[v18 目标形状] 主干 %.3fM / 全网 %.3fM（SE 块 %d 个）'
          % (backbone / 1e6, total / 1e6, n_se))
    # 逐段对账（主干 = stem + 13 个 SE 块 + 4 个 AttentionResBlock + 输出层）
    assert backbone == 3_740_930, backbone
    assert total == 3_986_468, total
    assert n_se * SE_PARAMS_AT_160 < backbone      # SE 块占主干一半以下
    # 同一形状下 resnet 路径的对照（证明上面 3.99M 不是「装漏了块」）
    ref = AlphaGoNet(in_channels=12, backbone_channels=160, backbone_res_blocks=17,
                     attention_mode='mix', num_attention_layers=4, num_heads=4,
                     action_size=362, arch='resnet', value_res_blocks=2)
    n_ref = sum(p.numel() for p in ref.parameters())
    assert 8.0e6 <= n_ref <= 9.5e6, n_ref
    assert total < n_ref / 2, 'SE 路径 %d vs resnet %d —— 省得不够多' % (total, n_ref)
