"""P3-D value 侧单元测试：MSE（tanh∈[-1,1] 直接回归 z∈[-1,1]）+ PPO value clipping。

被测实现：`scripts/selfplay_train.py::_ppo_value_loss`
规格出处：路线图 `docs/superpowers/plans/2026-09-25-v21-roadmap.md:294-302`（P3-D 段）：

    L_v = E[ max( (v_new - z)² , (clip(v_new, v_old ± ε) - z)² ) ]，ε = --ppo-clip

覆盖（路线图 P3-D 测试 ①②③）＋封闭形式验证：
  1. ① 手算 MSE 对照 ＋ ② clip 三段（未截 / 截断上界 / 截断下界，参数化 12 例）
  2. **封闭形式最优点**：目标在带内 → L_v ≡ 纯 MSE、最优 v*=z、损失 0、梯度不变；
     目标在带外 → 损失在「band 近侧 ↔ 对侧」之间是一段**平台**，平台值
     =(z-近侧)²、梯度**恰为 0**（对照无 clip 的 MSE 梯度 2(v-z) ≠ 0 → clip 真的
     改了梯度）；边界点 v=v_old±ε 梯度仍是 2(v_boundary-z)（非零，向内拉）
  3. 归约口径：逐样本 max 后 mean（不是 sum、不是「两项各自 mean 再取 max」）
  4. clip 锚点是 v_old 且**对称**（上/下界两例手算 + 镜像不变性 + 单侧裁剪反例）
  5. 变异防线：锚点误用 v_new（clip 退化成恒等 → 任何统计量都不变，P3-C 栽过的坑）、
     v_old 与 z 两列接反
  6. 接线：train_epochs 总损失 ≡ 手算 L_v（A≡0、kl_coef=0 → 策略项恒 0）；
     `_ppo_value_loss` 实收的 (value, z, v_old, clip_eps) 逐列对齐 buffer；
     `--ppo-clip` 是 value 侧唯一的 ε（调大 ε → L_v 单调不增）；
     优势只用 v_old、不被 v_new 污染、每 minibatch 只算一次
  7. ③ 源断言：value 侧无 BCE / 无稳健损失 / 无 sigmoid；两个损失都在 helper 里；
     策略侧零改动；value 侧不新增 CLI 参数（D1）

⚠ 与 `tests/test_rl_ppo_policy.py` 的分工：那个文件钉**策略侧**（`test_rl_ppo_policy.py:352`
  断言 `train_epochs` 源码里没有 `F.mse_loss`），本文件钉**value 侧**。两处断言互补、
  不重叠 —— 路线图 P3-D ③ 明确要求这一点。
"""
import argparse
import ast
import inspect
import math
import os
import sys
import textwrap
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.selfplay_train as st  # noqa: E402
from scripts.selfplay_train import _ppo_value_loss, train_epochs  # noqa: E402

_EPS = 0.2          # --ppo-clip 的路线图默认值


def _t(*vals):
    return torch.tensor(vals, dtype=torch.float32)


def _loss(v, z, v_old, eps=_EPS):
    """单样本闭包：返回 L_v 的 python float（默认 ε=0.2）。"""
    return _ppo_value_loss(_t(*v), _t(*z), _t(*v_old), eps)[0].item()


def _grad(v, z, v_old, eps=_EPS):
    """单样本闭包：返回 (L_v, dL_v/dv_new) —— 梯度是「clip 是否真起作用」的主证据。"""
    x = _t(*v).requires_grad_()
    loss = _ppo_value_loss(x, _t(*z), _t(*v_old), eps)[0]
    loss.backward()
    return loss.item(), x.grad.item()


def _code_only(func):
    """函数的**代码**（去 docstring、去注释，ast.unparse 归一化）。

    源断言只对代码下网：把「路线图规定用 MSE 而不是 SFT 的稳健损失」这类**说明**
    写进 docstring 不该让 `not in` 断言变红；反过来注释里提一句旧实现名也不该
    让 `in` 断言误判成「接线还在」。
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        fn.body = fn.body[1:]
    return ast.unparse(fn)


# --------------------------------------------------------------------------- #
# 1. ① 手算 MSE + ② clip 三段
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('v,z,v_old,expect,expect_mse,expect_clip_frac', [
    # --- 带内（|v-v_old| <= ε）：clip 恒等，L_v ≡ 纯 MSE ---
    (0.10, 0.50, 0.00, 0.1600, 0.1600, 0.0),    # 未截：v 在带内
    (-0.15, -0.60, 0.00, 0.2025, 0.2025, 0.0),  # 未截：带内、目标更远
    (0.20, 0.00, 0.00, 0.0400, 0.0400, 0.0),    # 未截：恰在上边界（闭区间）
    (-0.50, -0.30, -0.60, 0.0400, 0.0400, 0.0),  # 未截：锚点偏移、v 仍带内
    # --- 出上界（v > v_old+ε）：目标在带外时 clip 项才是悲观项 ---
    (0.50, 0.90, 0.00, 0.4900, 0.1600, 1.0),    # 截断上界：clip(0.5)=0.2 → (0.2-0.9)²
    (0.90, 0.50, 0.00, 0.1600, 0.1600, 1.0),    # 出上界但目标在带内 → 纯 MSE 项更大
    (0.50, -0.50, 0.00, 1.0000, 1.0000, 1.0),   # 出上界、目标在下带外 → 仍取纯 MSE
    (0.90, 0.95, 0.30, 0.2025, 0.0025, 1.0),    # 截断上界（锚点 0.3 → 带 [0.1,0.5]）
    # --- 出下界（v < v_old-ε）---
    (-0.50, -0.90, 0.00, 0.4900, 0.1600, 1.0),  # 截断下界：clip(-0.5)=-0.2
    (-0.90, -0.50, 0.00, 0.1600, 0.1600, 1.0),  # 出下界但目标在带内 → 纯 MSE
    (-0.90, -0.95, -0.30, 0.2025, 0.0025, 1.0),  # 截断下界（锚点 -0.3 → 带 [-0.5,-0.1]）
    (0.10, 0.40, -0.60, 0.6400, 0.0900, 1.0),   # 出上界、锚点 -0.6 → clip 到 -0.4
])
def test_value_loss_hand_computed(v, z, v_old, expect, expect_mse,
                                  expect_clip_frac):
    """① 手算 MSE 对照 ＋ ② clip 三段（未截 / 截断上界 / 截断下界）。"""
    loss, stats = _ppo_value_loss(_t(v), _t(z), _t(v_old), _EPS)
    assert loss.item() == pytest.approx(expect, abs=1e-6), (
        f'v={v} z={z} v_old={v_old}：得 {loss.item():.6f}，手算 {expect}')
    assert stats['v_mse'] == pytest.approx(expect_mse, abs=1e-6), \
        'v_mse 应是未裁剪的纯 MSE'
    assert stats['v_clip_frac'] == pytest.approx(expect_clip_frac)


def test_clip_is_never_optimistic_and_monotone_in_eps():
    """悲观性 L_v >= 纯 MSE 恒成立；ε 越大 → 裁剪越松 → L_v 单调不增。

    悲观性是 PPO value clipping 的**定义性质**，也是「删掉 clip 不可能更优」的依据；
    单调性把 ε 的语义（信任域宽度）钉死 —— 参数传错（1-ε / 2ε / 硬编码）会露馅。
    """
    rng = np.random.default_rng(0)
    v = torch.tensor(rng.uniform(-1, 1, 200), dtype=torch.float32)
    z = torch.tensor(rng.uniform(-1, 1, 200), dtype=torch.float32)
    v_old = torch.tensor(rng.uniform(-1, 1, 200), dtype=torch.float32)
    mse = float(F.mse_loss(v, z).item())
    loss, stats = _ppo_value_loss(v, z, v_old, _EPS)
    assert loss.item() >= stats['v_mse'] - 1e-6
    assert stats['v_mse'] == pytest.approx(mse, abs=1e-6)
    looser = [_ppo_value_loss(v, z, v_old, e)[0].item()
              for e in (0.05, 0.2, 0.5, 2.0)]
    assert looser == sorted(looser, reverse=True), looser
    # ε 覆盖整个值域（2.0 > 1.8 的最大可能偏差）时 clip 退化为恒等 → 精确等于纯 MSE
    assert looser[-1] == pytest.approx(mse, abs=1e-6)
    assert looser[0] > mse, '本构造里 ε=0.05 必须真的比纯 MSE 悲观'


# --------------------------------------------------------------------------- #
# 2. 封闭形式最优点（brief 硬要求）
# --------------------------------------------------------------------------- #
def test_closed_form_optimum_target_inside_band():
    """目标在信任域内：L_v(v) ≡ (v-z)²，最优 v*=z、损失 0，梯度恒为 2(v-z)。

    这一段里 clip **完全无副作用**（带内的 v 恒等；带外的 v 因目标更近，纯 MSE 项
    恒大于 clip 项）——「目标在信任域内」时 P3-D 的 value clipping 与纯 MSE 逐点
    相同，这是公式的自洽性检查（也是「clip 到底管什么」的最短答案：它只管
    「目标比旧价值更远」的那部分样本）。
    """
    v_old, z, eps = 0.0, 0.10, _EPS          # |z - v_old| = 0.1 <= 0.2
    for v in (-0.9, -0.2, 0.0, 0.1, 0.2, 0.5, 0.9):
        got, grad = _grad([v], [z], [v_old], eps)
        assert got == pytest.approx((v - z) ** 2, abs=1e-6), f'v={v}'
        assert grad == pytest.approx(2.0 * (v - z), abs=1e-5), f'v={v} grad={grad}'
    assert _grad([z], [z], [v_old], eps)[0] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize('z,v_old,v_inside', [
    (0.90, 0.0, (0.30, 0.45, 0.70)),         # 目标在上带外 → 平台 [0.2, 1.6]
    (-0.90, 0.0, (-0.70, -0.45, -0.30)),     # 目标在下带外 → 平台 [-1.6, -0.2]
    (0.80, -0.30, (0.0, 0.10, 0.30)),        # 锚点偏移 → 平台 [-0.1, 1.7]
    (0.55, 0.30, (0.52,)),                   # 平台极窄（0.05）：仍必须有 0 梯度
])
def test_closed_form_optimum_target_outside_band(z, v_old, v_inside):
    """目标在信任域外：L_v 有一段可闭式求出的**平台**，平台上梯度 ≡ 0。

    记 ε=--ppo-clip、d = |z - clip(z, v_old±ε)|（= z 到 band 近侧的距离）：

        clip 后的 v_clipped ≡ band 近侧（在平台区间内是常数）
        平台值 = d²
        平台区间 = [z - d, z + d]        （d 之外恢复纯 MSE 的二次增长）
        平台上梯度 ≡ 0

    梯度恰 0 是这条的核心：越过信任域后**不再有学习信号**（纯 MSE 梯度恒为
    2(v-z) ≠ 0）。把 clip 删掉的实现会在这里取到 0 损失并给出非零梯度。
    """
    eps = _EPS
    near = v_old - eps if z < v_old else v_old + eps
    d = abs(z - near)
    expect = d * d
    assert expect > 1e-6, '本例要求目标严格在带外（否则走「带内」那条测试）'
    for v in v_inside:
        got, grad = _grad([v], [z], [v_old], eps)
        assert abs(v - z) < d, f'构造失效：v={v} 不在平台区间内'
        assert got == pytest.approx(expect, abs=1e-6), \
            f'v={v} 应落在平台上（值恒为 {expect}）'
        assert grad == pytest.approx(0.0, abs=1e-7), (
            f'v={v} 在平台上梯度应恰为 0，实为 {grad}'
            f'（无 clip 的 MSE 梯度是 {2 * (v - z):.4f} ≠ 0）')
    # 平台的两端之外 → 恢复纯 MSE 的二次增长
    for v in (z - d - 0.5, z + d + 0.5):
        got, grad = _grad([v], [z], [v_old], eps)
        assert got == pytest.approx((v - z) ** 2, abs=1e-6), f'v={v}'
        assert grad == pytest.approx(2.0 * (v - z), abs=1e-5), f'v={v}'
    # 解析最小值：平台值 > 0，而纯 MSE 可以到 0 → 信任域确实**限制**了可达损失
    assert _grad([z], [z], [v_old], eps)[0] == pytest.approx(expect, abs=1e-6), \
        'v=z 也在平台上（两项都等于 d²）'


def test_closed_form_gradient_at_clip_boundary():
    """边界点 v = v_old±ε：损失被压在该处，梯度仍是 2(v_boundary - z)（非零）。

    把 clip 的两个语义分开钉死：
      · 目标函数在边界处**压住**（值 = (z - v_boundary)²，不再随 v 增长）；
      · 梯度在边界处**不为零**（仍是 d(v-z)²/dv）→「被 clip 掉」≠「学不动」；
        真要学不动得再越过 ε 一步（见上一条的平台梯度 0）。
    """
    eps, v_old, z = _EPS, 0.0, 0.9
    v_edge = v_old + eps
    got, grad = _grad([v_edge], [z], [v_old], eps)
    assert got == pytest.approx((z - v_edge) ** 2, abs=1e-6)
    assert grad == pytest.approx(2.0 * (v_edge - z), abs=1e-5)
    assert abs(grad) > 1e-3, \
        '边界梯度为 0 → 实现把 clip 后的项当成了唯一项（丢梯度）'
    # 刚越过边界一步 → 立刻进平台（梯度 0）：信任域是硬边界
    assert _grad([v_edge + 1e-3], [z], [v_old], eps)[1] == pytest.approx(0.0, abs=1e-7)
    # 下边界同理（z 在下带外）
    v_lo = v_old - eps
    got_lo, grad_lo = _grad([v_lo], [z], [v_old], eps)
    assert got_lo == pytest.approx((z - v_lo) ** 2, abs=1e-6)
    assert grad_lo == pytest.approx(2.0 * (v_lo - z), abs=1e-5)


# --------------------------------------------------------------------------- #
# 3. 归约口径：逐样本 max 后 mean
# --------------------------------------------------------------------------- #
def test_reduction_is_per_sample_max_then_mean():
    """L_v = mean_i max(·,·)（逐样本悲观），不是 sum、也不是 max(mean(·), mean(·))。

    构造：两个样本各自被选中的项**相反**（A 选 clip 项、B 选纯 MSE 项），此时
    「逐样本 max 后 mean」(0.745) 严格小于「两项各自 mean 再取 max」(0.58)、
    也远小于 sum(1.49) —— 三种归约口径给出三个互不相同的数。
    """
    # A: v=0.5, z=0.9, v_old=0 → 纯 MSE 0.16 / clip 项 (0.2-0.9)²=0.49 → 取 0.49
    # B: v=0.5, z=-0.5, v_old=0 → 纯 MSE 1.00 / clip 项 (0.2+0.5)²=0.49 → 取 1.00
    v, z, v_old = _t(0.5, 0.5), _t(0.9, -0.5), _t(0.0, 0.0)
    unclipped = [(0.5 - 0.9) ** 2, (0.5 + 0.5) ** 2]      # [0.16, 1.00]
    clipped = [(0.2 - 0.9) ** 2, (0.2 + 0.5) ** 2]        # [0.49, 0.49]
    per_sample = [max(a, b) for a, b in zip(unclipped, clipped)]   # [0.49, 1.00]
    expect = sum(per_sample) / 2.0                          # 0.745
    max_of_means = max(sum(unclipped) / 2, sum(clipped) / 2)      # 0.58
    assert expect != pytest.approx(max_of_means, abs=1e-3), '构造失效：与 max(mean,mean) 相等'
    assert expect != pytest.approx(sum(per_sample), abs=1e-3), '构造失效：与 sum 相等'

    loss, stats = _ppo_value_loss(v, z, v_old, _EPS)
    assert loss.item() == pytest.approx(expect, abs=1e-6)
    assert stats['v_mse'] == pytest.approx(sum(unclipped) / 2, abs=1e-6)
    # 除错分母的三种变体都变红
    assert loss.item() != pytest.approx(sum(per_sample), abs=1e-4)   # sum
    assert loss.item() != pytest.approx(expect * 2, abs=1e-4)        # 重复除 batch
    assert loss.item() != pytest.approx(max_of_means, abs=1e-4)      # 批级 max


# --------------------------------------------------------------------------- #
# 4. clip 锚点是 v_old 且对称
# --------------------------------------------------------------------------- #
def test_clip_is_symmetric_around_v_old():
    """对称性：band = [v_old-ε, v_old+ε]。上下界手算对齐 ＋ 镜像不变 ＋ 单侧反例。"""
    eps = _EPS
    # 上界截断：v_old=-0.5 → band=[-0.7,-0.3]；v=0.9、z=0.5
    #   纯 MSE (0.9-0.5)²=0.16 ；clip(0.9)=-0.3 → (-0.3-0.5)²=0.64 → 取 0.64
    assert _loss([0.9], [0.5], [-0.5], eps) == pytest.approx(0.64, abs=1e-6)
    # 下界截断（镜像）：v_old=+0.5 → band=[0.3,0.7]；v=-0.5、z=-0.9
    #   纯 MSE (-0.5+0.9)²=0.16 ；clip(-0.5)=0.3 → (0.3+0.9)²=1.44 → 取 1.44
    assert _loss([-0.5], [-0.9], [0.5], eps) == pytest.approx(1.44, abs=1e-6)
    # 镜像不变：整组 (v, z, v_old) 取负 → 损失不变（非对称实现必红）
    rng = np.random.default_rng(7)
    for _ in range(50):
        v, z, vo = rng.uniform(-1, 1, 3)
        assert _loss([v], [z], [vo], eps) == pytest.approx(
            _loss([-v], [-z], [-vo], eps), abs=1e-6)
    # 「只裁一侧」的变体必须能被区分出来（否则这条断言是空转）
    cases = ((0.9, 0.5, -0.5), (-0.5, -0.9, 0.5), (0.5, 0.9, 0.0),
             (-0.5, -0.9, 0.0), (0.7, 0.9, 0.0), (-0.7, -0.9, 0.0))
    up_diff, dn_diff = [], []
    for v, z, vo in cases:
        got = _loss([v], [z], [vo], eps)
        only_up = max((v - z) ** 2, (min(v, vo + eps) - z) ** 2)
        only_dn = max((v - z) ** 2, (max(v, vo - eps) - z) ** 2)
        up_diff.append(abs(got - only_up) > 1e-6)
        dn_diff.append(abs(got - only_dn) > 1e-6)
    assert any(up_diff), '只裁上界的实现与本实现在所有样本上都无法区分 → 防线空转'
    assert any(dn_diff), '只裁下界的实现与本实现在所有样本上都无法区分 → 防线空转'


# --------------------------------------------------------------------------- #
# 5. 变异防线：锚点 / 列接反
# --------------------------------------------------------------------------- #
def test_reference_is_v_old_not_v_new_nor_z():
    """锚点必须是 v_old。用 v_new 或 z 当锚点都会变红。

    最阴的一格是「用 v_new 当锚点」：clip(v_new, v_new±ε) ≡ 恒等 → L_v 退化成纯
    MSE，而纯 MSE 恰好是任何 value 回归都在算的东西 → 线上跑得动、日志也正常，
    只是信任域彻底没生效（P3-C 报告 §7 的 M11/M12 同型覆盖洞）。故显式断言
    「有样本越出信任域时 L_v 必须严格大于纯 MSE」。
    """
    v, z, v_old = _t(0.5, -0.4, 0.45), _t(0.9, -0.9, 0.85), _t(0.0, 0.0, 0.0)
    loss, stats = _ppo_value_loss(v, z, v_old, _EPS)
    mse = float(F.mse_loss(v, z).item())
    assert stats['v_clip_frac'] == 1.0, '三个样本都应越出信任域'
    assert loss.item() > mse + 1e-3, 'L_v 竟等于纯 MSE → clip 没生效（锚点用错？）'
    # 逐样本手算（锚点 = v_old = 0 → band = [-0.2, 0.2]）
    expect = []
    for vi, zi in zip(v.tolist(), z.tolist()):
        c = min(max(vi, -_EPS), _EPS)
        expect.append(max((vi - zi) ** 2, (c - zi) ** 2))
    assert loss.item() == pytest.approx(sum(expect) / 3, abs=1e-6)

    # 锚点换成 v_new → 恒等 → 精确等于纯 MSE（这就是「丢掉 clip」的观测后果）
    degenerate = _ppo_value_loss(v, z, v, _EPS)[0].item()
    assert degenerate == pytest.approx(mse, abs=1e-6)
    assert degenerate != pytest.approx(loss.item(), abs=1e-4)
    # 锚点换成 z（v_old / z 两列接反）→ 另一个数
    swapped = _ppo_value_loss(v, z, z, _EPS)[0].item()
    assert swapped != pytest.approx(loss.item(), abs=1e-4)


# --------------------------------------------------------------------------- #
# 6. 接线：train_epochs 真把 buffer 的列与 --ppo-clip 传进了 value 损失
# --------------------------------------------------------------------------- #
class _TinyNet(nn.Module):
    """最小可训练网络：(B,12,n,n) -> (policy (B,A), value (B,1))。"""

    def __init__(self, board=3, actions=5, value_const=None):
        super().__init__()
        self.c = nn.Conv2d(12, 8, 3, padding=1)
        self.p = nn.Linear(8 * board * board, actions)
        self.v = nn.Linear(8 * board * board, 1)
        if value_const is not None:          # 把 value 头钉成常数 c
            self.v.weight.data.zero_()
            self.v.bias.data.fill_(float(value_const))

    def forward(self, x):
        h = F.relu(self.c(x)).flatten(1)
        return self.p(h), self.v(h)


def _args(**over):
    base = dict(
        batch_size=8, epochs=1, lr=1e-2, weight_decay=1e-4, value_lr_mult=0.5,
        clip_grad=1.0, use_ema=0, grad_accum_steps=1,
        ppo_clip=_EPS, kl_coef=0.0, kl_target=1e9,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _make_ai(seed=0, board=3, actions=5, value_const=None):
    torch.manual_seed(seed)
    return types.SimpleNamespace(
        model=_TinyNet(board=board, actions=actions, value_const=value_const))


def _flat_buffer(n=8, board=3, actions=5, seed=0, z=0.75, v_old=0.0):
    """buffer 行 = P3-C 6 元组 (planes, action, logp_old, z, v_old, mask)。

    **每行的 z 与 v_old 相同**、且 z - v_old 是常数 → A = z - v_old 经 minibatch
    标准化后**恒为 0**（不是「减掉 v_old 之前为 0」，而是标准化把它压平）。再配
    kl_coef=0 → 策略项精确为 0，于是 `train_epochs` 的返回值**精确等于** L_v。

    两条性质缺一不可：
      · z ≡ v_old（差值恒定）→ 优势恒 0，且对 minibatch 抽到哪些行**不敏感**；
      · z ≠ v_old → 目标落在信任域外，clip 真的会生效（z ≡ v_old 时目标就是锚点、
        恒在带内，clip 永远退化成恒等）。

    **z 必须取 2 的可精确表示值（0.75/0.5/0.25…），这是硬约束不是口味**：
    `torch.mean` 在 float32 里对 8 个相同的 0.4 求平均会偏 1 ULP
    （0.4000000059604645 → 0.4000000357627869），于是
    `adv - mean ≈ ±2.98e-08`，再被 `_standardize_advantage` 的 `max(std, 1e-6)`
    放大成 A ≈ ±0.03 —— 策略项不再为 0，「返回值 ≡ L_v」的精确断言全崩。
    0.75 这类值的部分和（0.75/1.5/2.25/3.0…）逐位精确 → mean 无偏 → A 精确 0。
    下面的自检把这条约束钉死：换常数而没换对，构造时就直接红。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append((rng.random((12, board, board), dtype=np.float32),
                     int(rng.integers(0, actions)),
                     float(math.log(rng.uniform(0.1, 0.9))),
                     np.float32(z), np.float32(v_old),
                     np.ones(actions, dtype=bool)))
    adv = st._compute_advantage(
        torch.tensor([r[3] for r in rows]).unsqueeze(1),
        torch.tensor([r[4] for r in rows]).unsqueeze(1))
    assert torch.all(adv == 0), (
        'A 没有精确归零（adv={}）——常数 z 取了 float32 不可精确表示的值，'
        'mean 的 ULP 噪声会被标准放大成伪优势'.format(adv.flatten()[:3].tolist()))
    return rows


def _draw_idx(n, batch_size, seed):
    """复现 train_epochs 的那一次有放回抽样（它用全局 np.random，无放回抽 minibatch）。

    调用前必须**先** np.random.seed(seed)（seed 在 train_epochs 之前调用才复现得动，
    之后再 seed 已经晚了 —— 这正是第一次写这个测试时的失败原因）。
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return np.random.randint(0, n, size=min(batch_size, n))
    finally:
        np.random.set_state(state)


def _seeded_train_epochs(ai, buf, args, seed=2):
    """在固定 numpy 全局种子的前提下跑一次 train_epochs（并把种子状态还原）。"""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return train_epochs(ai, buf, args, 'cpu')
    finally:
        np.random.set_state(state)


def test_train_epochs_total_loss_equals_closed_form_value_loss():
    """A≡0、kl_coef=0 → 策略项精确为 0 → train_epochs 的返回值 ≡ 手算 L_v。

    一次钉死：value 侧公式、逐样本归约、ε 来源，以及「BCE 已删」——旧的
    BCE(WithLogits) 口径在这里给出的数（0.537）与 0.3025 对不上。
    """
    c = 0.5                                   # value 头常数 → v_new ≡ 0.5
    buf = _flat_buffer(z=0.75, v_old=0.0)    # z - v_old ≡ 0.75（可精确表示）→ A ≡ 0
    ai = _make_ai(seed=1, value_const=c)
    loss = _seeded_train_epochs(ai, buf, _args(), seed=1)
    # 手算：v=0.5、z=0.75、v_old=0、ε=0.2 → band=[-0.2, 0.2]，clip(0.5)=0.2
    #   max((0.5-0.75)², (0.2-0.75)²) = max(0.0625, 0.3025) = 0.3025
    #   （clip 项胜出 → 这个数同时证明 max 取的是悲观侧，min 会得 0.0625）
    assert loss == pytest.approx(0.3025, abs=1e-6)
    # 被删掉的旧口径：BCE(logit=0.5, target=(0.75+1)/2=0.875) = 0.537
    bce = float(F.binary_cross_entropy_with_logits(_t(c), _t(0.875)).item())
    assert abs(bce - 0.5366) < 1e-3, '对照数本身算错了，变异防线失效'
    assert loss != pytest.approx(bce, abs=1e-3), '还在算 BCE'


def test_value_loss_wires_buffer_columns_and_eps(monkeypatch):
    """`_ppo_value_loss` 实收的 (value, z, v_old, clip_eps) 必须逐列对齐 buffer。

    变异防线：z / v_old 两列接反、eps 传成别的数（1-ε / 2ε / 硬编码 0.2）都会红。
    """
    n, board, actions = 8, 3, 5
    rng = np.random.default_rng(4)
    z_col = rng.uniform(-1, 1, n).astype(np.float32)
    vo_col = rng.uniform(-1, 1, n).astype(np.float32)
    assert not np.allclose(z_col, vo_col), '两列必须不同，否则接反了也测不出来'
    buf = [(rng.random((12, board, board), dtype=np.float32),
            int(rng.integers(0, actions)), float(math.log(0.5)),
            np.float32(z_col[i]), np.float32(vo_col[i]),
            np.ones(actions, dtype=bool)) for i in range(n)]
    seen = []
    real = st._ppo_value_loss

    def spy(value, z, v_old, clip_eps):
        out = real(value, z, v_old, clip_eps)
        seen.append((value.detach().clone(), z.detach().clone(),
                     v_old.detach().clone(), float(clip_eps)))
        return out

    monkeypatch.setattr(st, '_ppo_value_loss', spy)
    ai = _make_ai(seed=2, board=board, actions=actions)
    # 期望值必须在 train_epochs **之前**算：它内部会 opt.step()，权重一动，
    # 事后再前向就对不上 spy 抓到的那次（那是 step 前的值）。列对齐与优化无关。
    idx = _draw_idx(n, n, seed=2)
    planes = torch.from_numpy(np.stack([buf[i][0] for i in idx])).float()
    with torch.no_grad():
        _, v_expect = ai.model(planes)
    _seeded_train_epochs(ai, buf, _args(batch_size=n, ppo_clip=0.13), seed=2)
    assert seen, 'train_epochs 没调 _ppo_value_loss（value 侧未接线？）'
    assert len(seen) == 1, f'1 epoch × 1 minibatch 应只调一次，实为 {len(seen)}'
    value, z, v_old, eps = seen[0]
    assert eps == pytest.approx(0.13), f'value 侧的 ε 应直接取 --ppo-clip，实为 {eps}'
    assert torch.allclose(value.reshape(-1), v_expect.reshape(-1), atol=1e-6)
    assert np.allclose(z.numpy().ravel(), z_col[idx], atol=1e-6), 'z 不是 buffer 第 3 列'
    assert np.allclose(v_old.numpy().ravel(), vo_col[idx], atol=1e-6), 'v_old 不是 buffer 第 4 列'
    assert not np.allclose(z.numpy().ravel(), vo_col[idx], atol=1e-3), '两列接反了却测不出来'


def test_ppo_clip_is_the_only_value_side_epsilon():
    """value 侧只有一个 ε、且就是 --ppo-clip（不新增参数）：调大 ε → L_v 单调不增。"""
    c, z, v_old = 0.5, 0.75, 0.0
    buf = _flat_buffer(z=z, v_old=v_old)      # z - v_old ≡ 0.75 → A ≡ 0
    losses, expect = [], []
    for eps in (0.05, 0.2, 0.5):
        ai = _make_ai(seed=1, value_const=c)
        losses.append(_seeded_train_epochs(ai, buf, _args(ppo_clip=eps), seed=1))
        cc = min(max(c, v_old - eps), v_old + eps)
        expect.append(max((c - z) ** 2, (cc - z) ** 2))
    for got, exp in zip(losses, expect):
        assert got == pytest.approx(exp, abs=1e-6), f'ε 手算 {exp}，实得 {got}'
    # 单调**不增**（ε 越大 → 信任域越宽 → 悲观目标越松）。原来写的是升序
    # sorted()，与本函数 docstring 相反 —— 它当时能绿纯粹靠 A 浮点噪声把
    # 策略项抬成随 ε 上升；A 精确归零后真值暴露，方向必须改成 reverse。
    assert losses == sorted(losses, reverse=True), f'L_v 必须对 ε 单调不增：{losses}'
    # 策略项恒 0（kl_coef=0 且 A≡0）→ 差异只可能来自 value 侧读了 ε
    assert abs(losses[0] - losses[2]) > 1e-3


def test_advantage_is_not_recomputed_from_new_value(monkeypatch):
    """裁剪后**不**重算优势：A 只由 (z, v_old) 决定，v_new 永远不进 A（规格歧义 C）。

    行为证据：把 `_compute_advantage` 的实参截下来 —— 第二个实参必须逐样本等于
    buffer 的 v_old 列、且**不等于**网络当前 value 输出；每个 minibatch 只调一次
    （不存在「先算一次、value 损失后再补一次」的重算）。
    """
    n, board, actions = 8, 3, 5
    rng = np.random.default_rng(11)
    z_col = rng.uniform(-1, 1, n).astype(np.float32)
    vo_col = rng.uniform(-1, 1, n).astype(np.float32)
    buf = [(rng.random((12, board, board), dtype=np.float32),
            int(rng.integers(0, actions)), float(math.log(0.5)),
            np.float32(z_col[i]), np.float32(vo_col[i]),
            np.ones(actions, dtype=bool)) for i in range(n)]
    calls = []
    real = st._compute_advantage

    def spy(z, v_old):
        calls.append((z.detach().clone(), v_old.detach().clone()))
        return real(z, v_old)

    monkeypatch.setattr(st, '_compute_advantage', spy)
    ai = _make_ai(seed=2, board=board, actions=actions, value_const=0.9)
    _seeded_train_epochs(ai, buf, _args(batch_size=n), seed=2)
    assert len(calls) == 1, f'每个 minibatch 只该算一次优势，实为 {len(calls)} 次'
    z, v_old = calls[0]
    idx = _draw_idx(n, n, seed=2)
    assert np.allclose(z.numpy().ravel(), z_col[idx], atol=1e-6)
    assert np.allclose(v_old.numpy().ravel(), vo_col[idx], atol=1e-6)
    # v_old 绝不能是 v_new（value 头被钉成常数 0.9，与 v_old 列可区分）
    assert not np.allclose(v_old.numpy(), np.full(n, 0.9, dtype=np.float32), atol=1e-3)


# --------------------------------------------------------------------------- #
# 7. ③ 源断言
# --------------------------------------------------------------------------- #
def test_source_assertions_value_side():
    """③ value 侧无 BCE / 无稳健损失 / 无 sigmoid；两个损失都在 helper 里；策略侧零改动。

    与 `test_rl_ppo_policy.py::test_train_epochs_source_wires_ppo_only` 互补：
    那个钉「策略侧无 MSE 回归」，本文件钉「value 侧无 BCE、公式收在 helper 里」。
    """
    src = _code_only(st.train_epochs)
    assert '_ppo_value_loss' in src, 'value 侧未走 _ppo_value_loss'
    assert '_ppo_policy_loss' in src, '策略侧被 P3-D 动过（路线图：策略侧零改动）'
    assert '_compute_advantage' in src, '优势计算被 P3-D 动过'
    # 两个损失都搬进了 helper → train_epochs 里两个损失函数名都不该出现。
    # 这同时让 P3-C 的 `'F.mse_loss' not in src`（策略侧无 MSE 回归）无需削弱。
    assert 'binary_cross_entropy' not in src, 'BCE 分支未删（C8 的 RL 侧语义错）'
    assert 'mse_loss' not in src, 'value 损失被内联进 train_epochs（绕开 helper 口径）'

    vsrc = _code_only(st._ppo_value_loss)
    for bad in ('binary_cross_entropy', 'huber', 'smooth_l1', 'softmax',
                'sigmoid', 'logit'):
        assert bad not in vsrc.lower(), f'value 损失里出现了 {bad}'
    # 只有一处 clamp（信任域）与一处 maximum（逐样本悲观）：多处 clamp = 另加了约束
    assert vsrc.count('clamp(') == 1, f'clamp 出现 {vsrc.count("clamp(")} 次'
    assert vsrc.count('maximum(') == 1, f'maximum 出现 {vsrc.count("maximum(")} 次'

    # 参数面没动：value 侧复用 --ppo-clip，D1 不新增参数
    msrc = inspect.getsource(st.main)
    assert '--ppo-clip' in msrc
    for flag in ('--value-clip', '--v-clip', '--value-clip-eps', '--v-eps'):
        assert flag not in msrc, f'value 侧不得新增参数（{flag}）'
    # 两个统计键必须真的进 swanlab —— helper 的 docstring 声称「由 main() 写」，
    # 少了这两行就是空话（变异：删掉 update 分支 → 红）
    assert '"v_mse": _ppo_stats["v_mse"]' in msrc, 'v_mse 没写进 swanlab'
    assert '"v_clip_frac": _ppo_stats["v_clip_frac"]' in msrc, 'v_clip_frac 没写进 swanlab'
