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
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
# 已删除：test_scan_dtype_follows_dt_not_the_fp32_parameter —— v21 被硬删除
# （MambaLTI/_chunked_scan 不复存在），扫描 dtype 契约随 Mamba 一起无对象。
# 已删除：test_checkpoint_recompute_restores_low_precision —— 梯度检查点子系统
# 已整体移除，没有「重算恢复 autocast」这回事了（`_autocast_like` 随之退场）。
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 2. 明确**不能**退的那几处（防下一个人为省显存去动它们）
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



def test_grad_scaler_is_enabled_on_v100_fp16_path():
    """**V100（sm_70）没有 bf16** ⇒ 那一档是 FP16，必须配 GradScaler。

    2026-10-08 硬件换成 A100（SFT）+ V100（RL）之后，这条不变量换了个载体：
    原来盯的是 NPU/910A 的 fp16 路径，现在盯 V100。断言形态不变 ——
    缩放/反缩放依赖 fp32 的状态量，而 V100 上 AMP 就是 fp16。
    """
    src = (ROOT / 'scripts' / 'train_sft.py').read_text(encoding='utf-8')
    assert 'GradScaler' in src, 'FP16 路径必须有 GradScaler'
    # V100（compute_cap < 8.0）必须真的建 scaler
    assert 'use_scaler = use_amp' in src, \
        'V100 落在保守分支（fp16 + GradScaler），那段 use_scaler 赋值必须还在'
