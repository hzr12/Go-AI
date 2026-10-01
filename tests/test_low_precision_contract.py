"""低精度化契约（2026-10-01）：哪些 fp32 可以退、哪些绝对不能退。

背景：4 卡 910A 的 OOM 排查最后落到两处「本该是 fp16 却是 fp32」：

  1. **checkpoint 重算不在 autocast 里** —— `backward()` 时前向的
     `with autocast(...)` 早已退出，于是检查点段的重算全部退回 fp32。
     证据是那个 `Tried to allocate 1.40 GiB`：它恰好等于
     `(B,Hh,P,l,N) = (1000,4,46,32,64)` 的 **fp32** 体积。
  2. **输入特征平面全程 fp32** —— 主机内存（预取队列）+ H2D 字节 + 设备侧
     输入张量都白带一倍。

（历史第 3 条「Mamba 的 `A_log` 是 fp32 参数」随 v21 硬删除一并退役：
`MambaLTI` 不复存在，扫描 dtype 用例已删除。）

本文件把「退了也不改数值」的那几条钉住，并明确列出**不能退**的那些 —— 否则下一
个人会为了省显存把优化器状态或 loss 累加也改成 fp16。
"""
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.networks.backbone as bb  # noqa: E402
from scripts.train_sft import KATAGO_SE_CFG, build_katago_se_net  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. 特征平面：fp16 且**逐位精确**（值域全是计数/布尔）
# --------------------------------------------------------------------------- #
def test_feature_planes_are_fp16_and_exact():
    """planes 用 fp16，且每个取值都是 fp16 能逐位表示的整数。

    为什么必须 fp16 而不是 bf16：bf16 只有 8 位尾数，整数只精确到 **256**；
    捕获计数在极端局面能超过它。fp16 精确到 **2048**，且这里还有 fp32 的
    2 倍指数位余量。
    """
    from src.game.go_rules import GoBoard
    n = 9
    my, op = [0, -1, -3], [1, -1, -3]
    b = GoBoard(n)
    b.play(0)
    b.play(n * n - 1)
    p1 = b.feature_planes(my, op, b.current_player, n_channels=17)
    p2 = GoBoard.feature_planes_batched(
        b.board[None], [my], [op], [b.current_player], [b.ko_point],
        n_channels=17)[0]
    for name, arr in (('feature_planes', p1), ('feature_planes_batched', p2)):
        assert arr.dtype == np.float16, f'{name} 应为 fp16，实为 {arr.dtype}'
        vals = np.unique(arr.astype(np.float32))
        assert np.all(vals == np.round(vals)), \
            f'{name} 出现了非整数值 {vals[:8]}（fp16 就不再是精确表示）'
        assert np.all(np.abs(vals) <= 2048), \
            f'{name} 出现了超出 fp16 精确整数范围的值 {vals}'


def test_single_and_batched_planes_agree_bitwise():
    """单图与批量两条路必须同 dtype 且逐位一致（训练走批量、RL 走单图）。"""
    from src.game.go_rules import GoBoard
    n = 9
    my, op = [4, -1, -3], [5, -1, -3]
    b = GoBoard(n)
    b.play(4)
    p1 = b.feature_planes(my, op, b.current_player, n_channels=17)
    p2 = GoBoard.feature_planes_batched(
        b.board[None], [my], [op], [b.current_player], [b.ko_point],
        n_channels=17)[0]
    assert p1.dtype == p2.dtype
    assert np.array_equal(p1, p2), '两路 planes 不一致（精度或语义都变了）'


# --------------------------------------------------------------------------- #
# 2. 检查点重算：必须恢复前向的 autocast 低精度（OOM 直接机制）
# --------------------------------------------------------------------------- #
# 已删除：test_scan_dtype_follows_dt_not_the_fp32_parameter —— v21 被硬删除
# （MambaLTI/_chunked_scan 不复存在），扫描 dtype 契约随 Mamba 一起无对象。

def test_checkpoint_recompute_restores_low_precision():
    """块级检查点的**重算**必须在低精度下跑（旧路径 16 段全覆盖）。

    这是 OOM 的直接机制：`backward()` 时前向的 autocast 已退出，不显式恢复就
    整段 fp32，体积翻倍。判据用「上下文里做一次 matmul 看输出 dtype」，而不是
    `torch.is_autocast_enabled()` —— 后者只报 CUDA 的状态，对 cpu/npu 读不到。

    段数口径（v21 硬删除后按存活架构重定）：`KATAGO_SE_CFG` 的 17 块走
    `SharedBackbone` 的 GC_LEGACY 逐块粒度 + `uncapped_last=True`
    ⇒ 恰好 16 个检查点段（第 17 块按 checkpoint_sequential 语义豁免），
    与 `SharedBackbone.forward` 注释里的「16 段入口」一致。
    """
    seen = []
    orig = bb._autocast_like

    def spy(t):
        ctx = orig(t)
        import contextlib

        @contextlib.contextmanager
        def wrapped():
            with ctx:
                a = torch.ones(4, 4)
                seen.append(str((a @ a).dtype).replace('torch.', ''))
                yield
        return wrapped()

    bb._autocast_like = spy
    try:
        net, _cfg = build_katago_se_net(
            action_size=362,
            grad_checkpoint=KATAGO_SE_CFG['grad_checkpoint'])
        net.train()
        x = torch.randn(2, KATAGO_SE_CFG['in_channels'], 19, 19)
        with torch.autocast(device_type='cpu', dtype=torch.float16):
            out = net(x)
            loss = (out[0].float().pow(2).mean() if isinstance(out, (tuple, list))
                    else out.float().pow(2).mean())
        loss.backward()
    finally:
        bb._autocast_like = orig
    assert seen, '重算上下文一次都没进入 —— 检查点没生效，测试无意义'
    assert all(d == 'float16' for d in seen), \
        f'有段在 fp32 下重算：{sorted(set(seen))}'
    assert len(seen) == 16, f'应覆盖 16 个段，实得 {len(seen)}'


# --------------------------------------------------------------------------- #
# 3. 明确**不能**退的那几处（防下一个人为省显存去动它们）
# --------------------------------------------------------------------------- #
def test_parameters_stay_fp32():
    """AMP 下**参数主权重必须 fp32** —— 优化器状态按它建立，GradScaler 也靠它。

    本轮把激活/重算的计算降到 autocast 精度，但参数本身仍是 fp32；
    这两件事不能一起改，否则 Adam 的更新量在 fp16 下会下溢。
    """
    net, _cfg = build_katago_se_net(action_size=362)
    for name, p in net.named_parameters():
        assert p.dtype == torch.float32, f'{name} 应为 fp32 参数，实为 {p.dtype}'


def test_value_and_loss_stay_fp32():
    """TD 目标 / value / loss 累加保持 fp32（数值敏感，退了会静默劣化）。"""
    from scripts.selfplay_train import _compute_advantage, _ppo_value_loss
    z = torch.randn(16)
    v = torch.randn(16)
    assert _compute_advantage(z, v).dtype == torch.float32
    loss, _stats = _ppo_value_loss(torch.randn(16), z, v, 0.2)
    assert loss.dtype == torch.float32


def test_optimizer_states_are_not_created_in_low_precision():
    """AdamW 的 exp_avg/exp_avg_sq 必须 fp32（Adam 的分母对精度极敏感）。"""
    p = torch.nn.Parameter(torch.randn(8))          # fp32 参数
    opt = torch.optim.AdamW([p], lr=1e-3)
    p.grad = torch.randn(8)
    opt.step()
    st = opt.state[p]
    assert st['exp_avg'].dtype == torch.float32
    assert st['exp_avg_sq'].dtype == torch.float32


def test_grad_scaler_is_enabled_on_npu_path():
    """910A 无 BF16，FP16 必须配 GradScaler —— 它依赖 fp32 的状态量。"""
    src = (ROOT / 'scripts' / 'train_sft.py').read_text(encoding='utf-8')
    assert 'torch.npu.amp.GradScaler' in src, 'NPU 路径必须用 GradScaler'
    assert 'amp_dtype = torch.float32' in src, \
        'CPU 路径才用 fp32；NPU 走 fp16（910A 无 BF16）'