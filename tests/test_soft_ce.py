"""`compute_policy_loss(kind='soft_ce')` 的软 CE 分支测试（软标签接入 A2）。

三种 kind 的分工
---------------
    'ce'       人类 / 自对弈的 **one-hot** 目标（默认，数值逐位不变）
    'huber'    同上的概率域回归（保留仅供复现实验）
    'soft_ce'  **KataGo 访问分布**软标签（蒸馏，spec §5.7）

本文件锁四件事
--------------
1. **掩码是逐行二选一**：`soft_mask=0` 的行贡献**恰好 0**，不退化成 one-hot CE、
   也不做插值；把那一行的软目标换成任何东西，loss 都必须逐位不变。
2. 软 target 是合法分布时，loss 精确等于该行的软交叉熵（公式钉死）。
3. `soft_weight=0` ⇒ 软项恒 0（旋钮有效）。
4. `huber` / `ce` 两条既有路径的数值与加入本分支**之前逐位相等**（裸
   `F.cross_entropy` / `F.smooth_l1_loss` 作 oracle）。

跑：pytest tests/test_soft_ce.py -v
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_sft import compute_policy_loss, soft_cross_entropy  # noqa: E402

A_19 = 362          # 生产动作数：19² 落点 + pass
A_SMALL = 10        # 小动作数（测试里让手算/公式校验可读）


def _logits(B=4, A=A_SMALL, seed=0, requires_grad=False):
    """固定种子的 logits（与 torch 全局 RNG 无关 ⇒ 每次跑逐位可复现）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, A, generator=g, dtype=torch.float32)
    x.requires_grad_(requires_grad)
    return x


def _soft(B=4, A=A_SMALL, seed=1):
    """固定种子的软标签：每行随机后归一化成**合法分布**（和为 1）。"""
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(B, A, generator=g, dtype=torch.float32)
    return s / s.sum(dim=-1, keepdim=True)


def _bits(x):
    """标量张量的**逐位**表示（torch.Tensor 没有 `.tobytes()`）。"""
    return np.asarray(x.detach().cpu()).tobytes()


def _ref_soft_ce(logits, soft):
    """oracle：逐行 −Σ soft·log_softmax(logits)，然后 mean。"""
    return -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


# --------------------------------------------------------------------------- #
# 1. 公式：软 target 是合法分布时，loss == 该行的软交叉熵
# --------------------------------------------------------------------------- #
def test_soft_ce_equals_the_row_soft_cross_entropy():
    """全 mask=1、B=1 时，loss 精确等于那一行的软 CE。"""
    logits = _logits(B=1, seed=2)
    soft = _soft(B=1, seed=3)
    assert abs(float(soft.sum()) - 1.0) < 1e-6, "前提：soft 是合法分布"
    got = compute_policy_loss(logits, torch.tensor([0]), 'soft_ce',
                              soft=soft, soft_mask=torch.ones(1))
    want = float(_ref_soft_ce(logits, soft))
    assert float(got) == pytest.approx(want, rel=1e-6), f"{float(got)} != {want}"


def test_soft_ce_batch_mean_over_masked_rows_only():
    """全 mask=1 时，loss = 逐行软 CE 的 batch 平均。"""
    logits = _logits(B=5, seed=4)
    soft = _soft(B=5, seed=5)
    got = compute_policy_loss(logits, torch.zeros(5, dtype=torch.long), 'soft_ce',
                              soft=soft, soft_mask=torch.ones(5))
    per_row = -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    assert float(got) == pytest.approx(float(per_row.mean()), rel=1e-6)
    assert float(got) != pytest.approx(float(per_row[0]), rel=1e-3), \
        "B=5 却等于单行 —— 归约口径错了"


def test_soft_ce_ignores_move_t():
    """`soft_ce` 的目标**只**来自软标签：`move_t` 改成什么都不影响结果。"""
    logits = _logits(B=3, seed=6)
    soft = _soft(B=3, seed=7)
    a = compute_policy_loss(logits, torch.tensor([0, 1, 2]), 'soft_ce',
                            soft=soft, soft_mask=torch.ones(3))
    b = compute_policy_loss(logits, torch.tensor([9, 8, 7]), 'soft_ce',
                            soft=soft, soft_mask=torch.ones(3))
    assert float(a) == float(b), "soft_ce 不该看 one-hot 目标"


def test_soft_ce_on_one_hot_target_matches_plain_ce():
    """软 target 恰为 one-hot 时，软 CE == `F.cross_entropy`（无 smoothing）。

    这是「软 CE 是 CE 的推广、不是另一个损失」这条语义的判据。
    """
    logits = _logits(B=4, seed=8)
    moves = torch.tensor([0, 3, 5, 9])
    soft = torch.zeros_like(logits)
    soft.scatter_(1, moves.view(-1, 1), 1.0)
    got = compute_policy_loss(logits, moves, 'soft_ce', soft=soft,
                              soft_mask=torch.ones(4))
    want = F.cross_entropy(logits, moves, label_smoothing=0.0)
    assert float(got) == pytest.approx(float(want), rel=1e-6)


def test_soft_ce_matches_label_smoothed_ce_when_target_is_smoothed():
    """软 target = label-smoothed one-hot 时 == `F.cross_entropy(label_smoothing=eps)`。"""
    logits = _logits(B=3, seed=9)
    moves = torch.tensor([1, 4, 7])
    eps = 0.1
    A = logits.shape[-1]
    soft = torch.full_like(logits, eps / A)
    soft.scatter_(1, moves.view(-1, 1), 1.0 - eps + eps / A)
    got = compute_policy_loss(logits, moves, 'soft_ce', soft=soft,
                              soft_mask=torch.ones(3), label_smoothing=eps)
    want = F.cross_entropy(logits, moves, label_smoothing=eps)
    assert float(got) == pytest.approx(float(want), rel=1e-6), \
        "软 CE 与 label-smoothed CE 不是同一个东西（软 CE 不做 smoothing）"


# --------------------------------------------------------------------------- #
# 2. 掩码逐行二选一：mask=0 的行贡献恰好 0
# --------------------------------------------------------------------------- #
def test_masked_row_contributes_exactly_zero():
    """同 batch 内 mask=[1,0]：第 2 行的软目标随便换，loss 必须**逐位不变**。

    这比「loss 变小」强得多：只要第 2 行有任何贡献（哪怕 1e-9），逐位比较就会红。
    """
    logits = _logits(B=2, seed=10)
    soft_a = _soft(B=2, seed=11)
    soft_b = soft_a.clone()
    soft_b[1] = soft_b[1].flip(0)
    soft_b[1] /= soft_b[1].sum()               # 仍是一个合法分布，但完全不同

    a = compute_policy_loss(logits, torch.zeros(2, dtype=torch.long), 'soft_ce',
                            soft=soft_a, soft_mask=torch.tensor([1.0, 0.0]))
    b = compute_policy_loss(logits, torch.zeros(2, dtype=torch.long), 'soft_ce',
                            soft=soft_b, soft_mask=torch.tensor([1.0, 0.0]))
    assert float(a) == float(b), "mask=0 的行参与了损失（不是二选一）"
    # 且等于「只有第 0 行有值、分母仍是 B」的 1/B 倍
    want = float(_ref_soft_ce(logits[:1], soft_a[:1])) / 2.0
    assert float(a) == pytest.approx(want, rel=1e-6), f"{float(a)} != {want}"


def test_mask_is_not_a_mix_with_one_hot_ce():
    """掩码**不是**与 one-hot CE 插值：mask=0 的行不会被 CE 兜底。

    若实现写成 `(1-m)·soft_ce + m·one_hot_ce`，`mask=[0,0]` 时 loss 会等于
    普通 CE（≈ ln A）；这里它必须是 0。
    """
    logits = _logits(B=3, seed=12)
    soft = _soft(B=3, seed=13)
    zero = compute_policy_loss(logits, torch.tensor([0, 1, 2]), 'soft_ce',
                               soft=soft, soft_mask=torch.zeros(3))
    assert float(zero) == 0.0, f"全掩码时 loss 应恰为 0，实得 {float(zero)}"
    assert float(zero) != pytest.approx(float(np.log(A_SMALL)), rel=1e-3), \
        "看起来退化成了 one-hot CE"


def test_mixed_mask_selects_rowwise():
    """mask=[1,0,1,0]：每个 1 行的软 CE 各进一次，0 行完全不进。"""
    logits = _logits(B=4, seed=14)
    soft = _soft(B=4, seed=15)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    got = compute_policy_loss(logits, torch.zeros(4, dtype=torch.long), 'soft_ce',
                              soft=soft, soft_mask=mask)
    per_row = -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    want = (per_row * mask).mean()            # 分母恒为 B（见 docstring）
    assert float(got) == pytest.approx(float(want), rel=1e-6)


def test_all_masked_rows_give_zero_gradient():
    """全掩码 ⇒ loss 与梯度都恰好 0（不是 0.0 但有 NaN 尾巴）。"""
    logits = _logits(B=3, seed=16, requires_grad=True)
    soft = _soft(B=3, seed=17)
    loss = compute_policy_loss(logits, torch.zeros(3, dtype=torch.long), 'soft_ce',
                               soft=soft, soft_mask=torch.zeros(3))
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
    assert torch.isfinite(logits.grad).all()


def test_denominator_is_batch_size_not_mask_sum():
    """分母恒为 B：软项量级随软行占比线性变化（「二选一」的直接推论）。

    这是刻意钉住的口径 —— 若哪天改成 sum/Σmask，本用例就会红，
    提醒改动者同时改 `soft_weight` 的语义说明。
    """
    logits = _logits(B=4, seed=18)
    soft = _soft(B=4, seed=19)
    per_row = -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    half = compute_policy_loss(logits, torch.zeros(4, dtype=torch.long), 'soft_ce',
                               soft=soft, soft_mask=torch.tensor([1., 0., 1., 0.]))
    full = compute_policy_loss(logits, torch.zeros(4, dtype=torch.long), 'soft_ce',
                               soft=soft, soft_mask=torch.ones(4))
    assert float(half) == pytest.approx(float((per_row[[0, 2]].sum()) / 4), rel=1e-6)
    assert float(half) < float(full), "少一半软行 ⇒ 量级应减半（分母恒为 B）"


# --------------------------------------------------------------------------- #
# 3. soft_weight：软项的全局缩放
# --------------------------------------------------------------------------- #
def test_soft_weight_scales_linearly():
    logits = _logits(B=3, seed=20)
    soft = _soft(B=3, seed=21)
    kw = dict(soft=soft, soft_mask=torch.ones(3))
    base = float(compute_policy_loss(logits, torch.zeros(3, dtype=torch.long),
                                     'soft_ce', **kw))
    for w in (0.0, 0.15, 1.0, 2.5):
        got = float(compute_policy_loss(logits, torch.zeros(3, dtype=torch.long),
                                        'soft_ce', soft_weight=w, **kw))
        assert got == pytest.approx(w * base, rel=1e-6, abs=1e-12), \
            f"soft_weight={w}: {got} != {w}*{base}"


def test_soft_weight_zero_gives_exactly_zero():
    """`soft_weight=0` ⇒ 软项恒 0（旋钮必须真的能关掉，不是乘了个极小数）。"""
    logits = _logits(B=4, seed=22, requires_grad=True)
    soft = _soft(B=4, seed=23)
    loss = compute_policy_loss(logits, torch.zeros(4, dtype=torch.long), 'soft_ce',
                               soft=soft, soft_mask=torch.ones(4), soft_weight=0.0)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_soft_weight_defaults_to_one():
    """默认 `soft_weight=1.0`（与 CLI `--soft-weight` 的默认值一致）。"""
    import inspect
    p = inspect.signature(compute_policy_loss).parameters
    assert p['soft_weight'].default == 1.0
    assert p['soft'].default is None and p['soft_mask'].default is None


# --------------------------------------------------------------------------- #
# 4. 零回归：huber / ce 两条旧路径逐位不变
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('eps', [0.0, 0.1, 0.3])
def test_ce_path_bitwise_unchanged(eps):
    """`kind='ce'` 与裸 `F.cross_entropy` 逐位相等（含 label_smoothing）。"""
    logits = _logits(B=6, seed=24)
    moves = torch.randint(0, A_SMALL, (6,), generator=torch.Generator().manual_seed(25))
    got = compute_policy_loss(logits, moves, 'ce', label_smoothing=eps)
    want = F.cross_entropy(logits, moves, label_smoothing=eps)
    assert got.dtype == want.dtype
    assert _bits(got) == _bits(want), "ce 分支数值被本任务改了"


@pytest.mark.parametrize('beta,eps', [(0.5, 0.1), (0.5, 0.0), (1.0, 0.25)])
def test_huber_path_bitwise_unchanged(beta, eps):
    """`kind='huber'` 与裸 `F.smooth_l1_loss` 逐位相等。

    oracle = label-smoothed one-hot 目标上的逐元素 smooth L1，类内 sum over A
    再 batch mean（P4.5-fix 的归约口径）。
    """
    logits = _logits(B=6, seed=26)
    moves = torch.randint(0, A_SMALL, (6,), generator=torch.Generator().manual_seed(27))
    A = logits.shape[-1]
    target = torch.full_like(logits, eps / A)
    target.scatter_(1, moves.view(-1, 1), 1.0 - eps + eps / A)
    want = F.smooth_l1_loss(F.softmax(logits, dim=-1), target, beta=beta,
                            reduction='none').sum(dim=-1).mean()
    got = compute_policy_loss(logits, moves, 'huber', label_smoothing=eps,
                              huber_beta=beta)
    assert got.dtype == want.dtype
    assert _bits(got) == _bits(want), "huber 分支数值被本任务改了"


@pytest.mark.parametrize('kind,eps', [('ce', 0.1), ('huber', 0.1)])
def test_old_paths_ignore_the_new_kwargs(kind, eps):
    """**新形参绝不能漏进旧路径**：`soft`/`soft_mask`/`soft_weight` 传了也不变。"""
    logits = _logits(B=5, seed=28)
    moves = torch.randint(0, A_SMALL, (5,), generator=torch.Generator().manual_seed(29))
    base = compute_policy_loss(logits, moves, kind, label_smoothing=eps)
    noisy = compute_policy_loss(
        logits, moves, kind, label_smoothing=eps,
        soft=_soft(B=5, seed=30) * 7.0, soft_mask=torch.zeros(5),
        soft_weight=99.0)
    assert _bits(base) == _bits(noisy), f"{kind} 路径被新形参污染了"


def test_default_kind_is_still_ce_flag_unchanged():
    """CLI 的 `--policy-loss` choices 含 `soft_ce`，**但默认仍是 `ce`**。

    ⚠ **A4（2026-10-02）改写了这条断言，理由必须留着**：

    A2 交付时 `soft_ce` 只是**函数级** kind，CLI 侧 `--policy-loss` 的 choices
    冻结在 `['huber','ce']`，本文件与 `tests/test_huber_loss.py::
    test_no_new_cli_params`（D1 的 61 flag 冻结）一起把它钉住。A4 要加
    `--soft-index` / `--soft-weight` / `--soft-only-sampling` / `--soft-every`
    四个 CLI 旗，于是 `--policy-loss` 必须多接受一个 `soft_ce`
    ——`--soft-index` 一给就把生效口径派生为 `soft_ce`（见
    `scripts/train_sft.py::resolve_policy_loss_kind`）。

    本文件原来的断言是**逐字** `choices=['huber', 'ce']`，它守的判据其实有两条，
    现在分开守且**一条都没削弱**：

      1. 「默认不是 huber、也不是 soft_ce」→ `default == 'ce'` 逐字不变；
      2. 「三种 kind 都在 choices 里」→ 从**真 argparse**（`_replay_parser`）
         读，而不是从源码里抠字符串字面量。后者更抗排版改写，且顺带钉住
         「choices 不是空/非空」那种不可证伪的弱化。

    `tests/test_huber_loss.py::test_no_new_cli_params` 的 61→65 冻结集与
    `test_policy_loss_default_is_ce` 的 choices 断言是同一次改动，注释互指。
    """
    import argparse
    import ast
    import inspect
    import textwrap
    import scripts.train_sft as t

    kw = {}
    for call in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(t.main)))):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'add_argument' and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
                and call.args[0].value == '--policy-loss'):
            kw = {k.arg: k.value for k in call.keywords}
    assert kw, 'main() 里找不到 --policy-loss 的 add_argument'
    # ① 默认值逐字不变：P4.5b 用户裁决的「默认不是 huber」在 A4 之后**更重要**
    #    （多了 soft_ce 这个新选项，「默认会不会漂到它」成了新的风险面）。
    assert ast.literal_eval(kw['default']) == 'ce', \
        '--policy-loss 的默认必须是 ce（P4.5b 用户裁决，A4 加了 soft_ce 后不变）'
    # ② 三种 kind 齐备，从**真 parser** 读（不是源码字面量）。
    ap = argparse.ArgumentParser()
    scope = dict(vars(t))
    scope['ap'] = ap
    node = next(c for c in ast.walk(ast.parse(textwrap.dedent(
        inspect.getsource(t.main))))
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
        and c.func.attr == 'add_argument' and c.args
        and getattr(c.args[0], 'value', None) == '--policy-loss')
    expr = ast.Expression(body=node)
    ast.copy_location(expr, node)
    exec(compile(expr, '<policy-loss argparse>', 'eval'), scope)
    got = list(ap._actions[-1].choices)
    assert got == ['huber', 'ce', 'soft_ce'], \
        f'--policy-loss 的 choices 变成了 {got}（A4 之后应为 huber/ce/soft_ce）'
    assert ap.parse_args([]).policy_loss == 'ce', '默认解析结果不是 ce'
    for k in ('huber', 'ce', 'soft_ce'):
        assert ap.parse_args(['--policy-loss', k]).policy_loss == k

    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'scripts', 'train_sft.py'),
        encoding='utf-8').read()
    assert "huber|ce|soft_ce" in src, "分派器的报错信息应列出 soft_ce"


# --------------------------------------------------------------------------- #
# 5. 梯度 / 数值健壮性
# --------------------------------------------------------------------------- #
def test_gradient_flows_to_logits():
    """梯度必须能流到 logits（软 CE 打在 log-prob 上，不是常量）。"""
    logits = _logits(B=4, seed=31, requires_grad=True)
    soft = _soft(B=4, seed=32)
    loss = compute_policy_loss(logits, torch.zeros(4, dtype=torch.long), 'soft_ce',
                               soft=soft, soft_mask=torch.tensor([1., 0., 1., 1.]))
    loss.backward()
    g = logits.grad
    assert g is not None and g.shape == logits.shape
    assert torch.isfinite(g).all(), "梯度出现 NaN/Inf"
    assert torch.count_nonzero(g) > 0, "梯度恒 0 —— 软项没接上"
    # 被掩掉的行不该有梯度（mask=0 ⇒ 逐行贡献 0 ⇒ 该行梯度也是 0）
    assert torch.count_nonzero(g[1]) == 0, "mask=0 的行拿到了梯度"
    assert torch.count_nonzero(g[[0, 2, 3]]) > 0


def test_gradient_matches_closed_form():
    """d(soft CE)/d(logit_j) = p_j − y_j（与 CE 同一条链式法则）。

    这一条同时钉住「软 CE 定义在 log-prob 上 ⇒ 梯度与 A 无关」这个结构性质
    （spec §5.7 选 CE 系而不是 Huber 系的理由）。
    """
    logits = _logits(B=2, seed=33, requires_grad=True)
    soft = _soft(B=2, seed=34)
    mask = torch.tensor([1.0, 0.0])
    loss = compute_policy_loss(logits, torch.zeros(2, dtype=torch.long), 'soft_ce',
                               soft=soft, soft_mask=mask)
    loss.backward()
    p = F.softmax(logits, dim=-1).detach()
    want = (p - soft) * mask.view(-1, 1) / logits.shape[0]
    assert torch.allclose(logits.grad, want, atol=1e-6)


def test_low_precision_logits_are_computed_in_float32():
    """bf16/fp16 的 logits 也按 fp32 算（软 target 在低精度下会被舍没）。"""
    for dt in (torch.bfloat16, torch.float16):
        logits = _logits(B=2, seed=35).to(dt).requires_grad_(True)
        soft = _soft(B=2, seed=36)
        loss = compute_policy_loss(logits, torch.zeros(2, dtype=torch.long),
                                   'soft_ce', soft=soft, soft_mask=torch.ones(2))
        assert loss.dtype == torch.float32, f"{dt} 下返回了 {loss.dtype}"
        ref = _ref_soft_ce(logits.detach().float(), soft)
        assert float(loss.detach()) == pytest.approx(float(ref), rel=1e-2)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_pass_slot_receives_soft_mass():
    """最后一项是 pass：软标签在 pass 上的质量必须真的进 loss（不能只算 361 落点）。"""
    bs = 9
    A = bs * bs + 1
    logits = _logits(B=1, A=A, seed=37)
    soft = torch.zeros(1, A)
    soft[0, A - 1] = 1.0                    # 全压在 pass 上
    got = compute_policy_loss(logits, torch.zeros(1, dtype=torch.long), 'soft_ce',
                              soft=soft, soft_mask=torch.ones(1))
    lp = F.log_softmax(logits, dim=-1)[0, A - 1]
    assert float(got) == pytest.approx(float(-lp), rel=1e-6)


# --------------------------------------------------------------------------- #
# 6. 契约守卫
# --------------------------------------------------------------------------- #
def test_soft_ce_requires_soft_and_mask():
    """没给 soft/soft_mask 时必须**报错**，不能静默退化成 one-hot CE。"""
    logits = _logits(B=2, seed=38)
    with pytest.raises(ValueError):
        compute_policy_loss(logits, torch.zeros(2, dtype=torch.long), 'soft_ce')
    with pytest.raises(ValueError):
        compute_policy_loss(logits, torch.zeros(2, dtype=torch.long), 'soft_ce',
                            soft=_soft(B=2, seed=39))


def test_shape_mismatches_are_rejected():
    logits = _logits(B=3, A=A_SMALL, seed=40)
    with pytest.raises(ValueError):
        compute_policy_loss(logits, torch.zeros(3, dtype=torch.long), 'soft_ce',
                            soft=_soft(B=3, A=A_19, seed=41),
                            soft_mask=torch.ones(3))
    with pytest.raises(ValueError):
        compute_policy_loss(logits, torch.zeros(3, dtype=torch.long), 'soft_ce',
                            soft=_soft(B=3, seed=42), soft_mask=torch.ones(2))


def test_unknown_kind_still_rejected():
    logits = _logits(B=2, seed=43)
    with pytest.raises(ValueError):
        compute_policy_loss(logits, torch.zeros(2, dtype=torch.long), 'bce')


def test_soft_cross_entropy_is_callable_directly():
    """`soft_cross_entropy` 是可单测的独立入口（`compute_policy_loss` 内部转发）。"""
    logits = _logits(B=3, seed=44)
    soft = _soft(B=3, seed=45)
    mask = torch.tensor([1.0, 0.0, 1.0])
    a = soft_cross_entropy(logits, soft, mask)
    b = compute_policy_loss(logits, torch.zeros(3, dtype=torch.long), 'soft_ce',
                            soft=soft, soft_mask=mask)
    assert float(a) == float(b)


def test_end_to_end_from_dataset_dict_to_loss():
    """A1 的 dict 直接喂 A2 的 loss：形状/语义对得上（软 CE 那条线打通）。

    19 路 + 真挂软标签，掩码二选一：一个批里混合软行与 one-hot-only 行。
    """
    import scripts.train_sft as t
    from src.data.dataset import SupervisedDataset

    bs, B = 19, 6
    A = bs * bs + 1
    rng = np.random.default_rng(0)
    n = 8
    data = {
        'boards': np.zeros((n, bs, bs), dtype=np.int8),
        'my_hist': np.full((n, 3), -1, dtype=np.int16),
        'op_hist': np.full((n, 3), -1, dtype=np.int16),
        'ko': np.full(n, -1, dtype=np.int16),
        'moves': rng.integers(0, bs * bs, size=n).astype(np.int16),
        'values': rng.choice([-1, 1], size=n).astype(np.int8),
        'to_play': rng.choice([-1, 1], size=n).astype(np.int8),
        'game_ids': np.zeros(n, dtype=np.int32),
    }
    sp = rng.random((2, A)).astype(np.float32)
    sp /= sp.sum(1, keepdims=True)
    ds = SupervisedDataset(data, n_channels=12,
                           soft_idx=np.array([1, 5]), soft_policy=sp)
    _, moves_out, _, d = ds.sample_batch_numpy(np.arange(B), augment=False,
                                               labels=True)
    assert d['soft'].shape == (B, A)
    assert d['soft_mask'].tolist() == [0, 1, 0, 0, 0, 1, 0, 0][:B]

    logits = torch.from_numpy(rng.standard_normal((B, A)).astype(np.float32))
    logits.requires_grad_(True)
    loss = t.compute_policy_loss(
        logits, torch.from_numpy(moves_out), 'soft_ce',
        soft=torch.from_numpy(d['soft']),
        soft_mask=torch.from_numpy(d['soft_mask']), soft_weight=0.7)
    loss.backward()
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    grad = logits.grad
    # 软行有梯度、one-hot-only 行的梯度恒 0（未被这个 loss 触及）
    hit = d['soft_mask'] > 0
    assert torch.count_nonzero(grad[hit]) > 0
    assert torch.count_nonzero(grad[~hit]) == 0
