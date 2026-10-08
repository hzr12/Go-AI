import contextlib
import functools
import logging as _logging_mod
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  （`torch.utils.checkpoint` 的显式 import）

# 绝对导入而非 `from .se_bottleneck import …`：
# `tests/test_backbone_block_classes.py` 用 `spec_from_file_location` 把本文件
# **当独立脚本**加载（没有包上下文），相对导入会 `ImportError: attempted relative
# import with no known parent package`。绝对导入与本仓其余跨包引用的写法一致
# （见 `src/inference.py` / `src/search/*.py`），两条路径都能过。
from src.networks.se_bottleneck import SEBottleneck


# ---- NPU 融合 RMSNorm（可选加速，带运行时回退）----------------------------
# torch_npu 缺失或 npu_rms_norm 不可用时，RMSNorm 退化为下方的标准实现
# （含 fp16→fp32 硬化），数值行为完全不变。NPU 机上若 npu_rms_norm 因签名/
# 设备异常而调用失败，forward 内的 try/except 也会退回标准路径，故融合不会
# 破坏训练（最坏只是退回原速度）。
try:
    import torch_npu  # 仅在 NPU 环境可导入
    _HAS_NPU_RMS_NORM = hasattr(torch_npu, 'npu_rms_norm')
except Exception:
    torch_npu = None
    _HAS_NPU_RMS_NORM = False


def _npu_rms_norm(x, weight, eps):
    """torch_npu.npu_rms_norm 的薄封装，兼容返回 (out, invvar) 或 out 的版本差异。"""
    out = torch_npu.npu_rms_norm(x, weight, eps)
    return out[0] if isinstance(out, (tuple, list)) else out


class RMSNorm(nn.Module):
    """RMSNorm（兼容 PyTorch 2.1，不依赖 nn.RMSNorm）。

    沿**最后一维**（token/通道维）求 RMS —— 与 `katago_v7.RMSNormMask`
    （在空间维 ``(H,W)`` 上求）分工不同，两者不可互换。
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x):
        # NPU 融合路径：npu_rms_norm 把「平方+mean+rsqrt+乘权重」合成一个 kernel，
        # 减少 kernel launch / 显存往返。fp16 仍先 cast 到 fp32 再算（保留下方标准
        # 的硬化，避免 fp16 平方溢出），融合只改「执行方式」、不改数值。
        if x.device.type == 'npu' and _HAS_NPU_RMS_NORM:
            try:
                if x.dtype == torch.float16:
                    return _npu_rms_norm(x.float(), self.weight.float(), self.eps).to(x.dtype)
                return _npu_rms_norm(x, self.weight, self.eps)
            except Exception as _e:
                # 融合失败（签名/设备异常等）不要静默吞掉：至少告警一次，否则会
                # 永远退回慢速但正确的标准路径，没人知道融合路径其实是坏的。
                import logging
                logging.getLogger(__name__).warning(
                    "[RMSNorm] NPU 融合失败，退回标准路径: %s", _e)
        # 标准路径（CPU / GPU / 非 NPU）：
        #   - fp16：平方易溢出（910A 走 fp16 AMP，x 超 ~256 撞 65504 上限 ⇒
        #     `mean=inf` ⇒ `rsqrt=0` ⇒ 前向全 0、存下的却是 inf、梯度永远回不来，
        #     GradScaler 也救不回）。故**仅 fp16** 先升 fp32 中间量再算、末尾 cast 回。
        #   - bf16 / fp32 / fp64：**保留输入 dtype、与朴素写法 `x*rms*w` 逐位一致**
        #     （test_non_fp16_dtypes_are_bitwise_unchanged 的零回归契约）。bf16 的指数
        #     位与 fp32 同宽、无 fp16 那种平方溢出，刻意不碰它 —— 否则「只修 fp16」
        #     会悄悄改成 bf16/fp64 数值，违反契约。bf16 输出经 `*weight(fp32)` 自然
        #     提升为 fp32，与朴素写法一致（autocast 行为不变）。
        if x.dtype == torch.float16:
            xf = x.float()
            rms = (xf.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
            return (xf * rms * self.weight.float()).to(x.dtype)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return x * rms * self.weight


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization (ConvNeXt style)。"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 风格残差块：深度卷积 (5x5) + LayerNorm + PWConv (1x1) + GELU。

    相比原有 ResBlock：
      - 5x5 深度卷积代替 3x3 逐点卷积，扩大感受野
      - LayerNorm 代替 BatchNorm，训练更稳定
      - GELU 代替 ReLU，非线性更平滑
      - 参数量相近（~4× 因 expand factor=4）
    """

    def __init__(self, channels):
        super().__init__()
        # Depthwise Conv (5x5 大核，扩大感受野)
        self.dwconv = nn.Conv2d(channels, channels, 5, padding=2,
                                groups=channels, bias=False)
        self.norm = LayerNorm2d(channels)
        # Pointwise Conv 1 (expand)
        self.pwconv1 = nn.Conv2d(channels, channels * 4, 1, bias=False)
        # Pointwise Conv 2 (contract)
        self.pwconv2 = nn.Conv2d(channels * 4, channels, 1, bias=False)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        return x + residual


class ResBlock(nn.Module):
    """纯卷积残差块（保持原有结构，用于浅层局部特征提取）。"""

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Zero-init bn2 gamma 使残差块初始为恒等映射（He et al. 2016）
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


# ============================================================================
# 梯度检查点（P4.6b）
#
# 本节是**通用机制**，不含任何具体主干容器 —— 容器归 `SharedBackbone` /
# `AlphaGoNet`（src/networks/alphanet.py）。两者通过 `GradCheckpointMixin` +
# `run_grad_segment()` 这一对接口对接。
# ============================================================================

#: 块类型标识。现役只有 `legacy`：`SharedBackbone.blocks` 那个混合列表（resnet /
#: convnext / se_bottleneck / attention 混排）走这一段。
#:
#: `res` / `transformer` 两个 kind **注册但当前无人调用** —— `ResBlock` 与
#: `TransformerBlock` 都还在（前者是 v12/v18 的基本块，后者是可复用的 pre-norm
#: 块），但没有任何现役主干按 kind 分段调用它们。保留这两个常量是为了
#: `run_segment(..., kind=...)` 的调用面与既有测试不必因为「删了一个类」而重写；
#: 随 v21 一起退役的 `mamba` / `cross_attn_res` 两个 kind 已删除（对应块类
#: `MambaLTI` / `CrossAttnRes` 不复存在，留着就是两个永远点不到的死常量）。
GC_RES = 'res'
GC_TRANSFORMER = 'transformer'
GC_LEGACY = 'legacy'

#: 逐 kind 的默认开关。语义不变：`legacy` 默认**关**（不改变既有路径的行为）。
GRAD_CHECKPOINT_DEFAULTS = {
    GC_RES: True,
    GC_TRANSFORMER: True,
    GC_LEGACY: False,
}

#: 旧路径的**段粒度**：`checkpoint_sequential(blocks, len(blocks), x)` 的粒度
#: 就是「每块一段」，所以旧路径必须逐块，不能并段。
#:
#: 这**不是**一个可以随手「顺手统一」的默认值 —— 它是旧路径显存口径的一部分。
#: v18 形状（17 块、192 通道）实测留给反向的字节：
#:     逐块（本值）  16 个段入口   ≈ 与 HEAD 的 `checkpoint_sequential` 一致
#:     并段                1 个段入口   ≈ 省 16×
#: 换句话说：把本表清空（`{}`）会让旧路径的真实训练显存**静默大降**，而
#: `search_arch` 的标定因子 k 是照着「逐块」那一档锚在 v18 的 31.12GB 实测上的。
#: 并段本身是更好的显存策略，但它是一次**需要重标 k** 的行为变化，不能混在
#: 「机制重构」里做。见 `tests/test_grad_checkpointing.py::
#: test_legacy_granularity_is_pinned_to_checkpoint_sequential`。
#: 见上面那段裁决：逐块。
GC_PER_BLOCK_DEFAULT = {
    GC_RES: True, GC_TRANSFORMER: True,
    GC_LEGACY: True,
}


# 粒度裁决（2026-09-30，用户 4×910A OOM 之后实测；历史，裁决对象已退役）
# --------------------------------------------------------
# 原裁决：某些 kind 用**段级**（同类型连续块合并成一个 checkpoint 段，brief §2 的
# (b)），理由是「(a) 会在每块边界留激活、(b) 只留 1 个；重算的调度开销两者相同，(b)
# 的 autograd 钩子还少 4/5」。
#
# 那个比较**只看了留给反向的存储，没看重算期的峰值** —— 而 OOM 是被后者打爆的：
# 段级把整段包进**一个** `checkpoint`（`run_grad_segment` 的 `if not per_block:`
# 分支），于是反向重算时**整段内部所有块的中间激活同时活着**，峰值按「段」计。
#
# 实测（tests 同款 harness，B=32 / 19 路 / 184ch / fp32 CPU，区间峰值增量）：
#     段级   fwd 292.9 MB   fwd+bwd 1306.6 MB   fwd+bwd 26.36 s
#     逐块   fwd 189.9 MB   fwd+bwd  149.0 MB   fwd+bwd 26.08 s
# 逐块在**两个峰值上都更低**（B=8 时前向驻留反而高 91.7 vs 72.5，那是 16 个段的开销
# 还压得住的小 batch 情形），而**耗时不增**（重算总量本来就一样，只是峰值从「一段」
# 降到「一块」）。
#
# 与分片并行的关系：分片包装本来就是按块类切（每个块是一个 wrap unit），逐块检查点
# 与它 1:1 对齐，重算时只重新 all-gather 一个 unit —— 段级则会让重算横跨多个 unit。
# 留下这段是为了**别再犯同一个错**：本表里现役只有 `legacy`，而 `legacy` 是
# **逐块**；任何新增的分段主干请直接沿用逐块，除非愿意像上面那样实测重算峰值。


class _BatchNormStatGuard:
    """**只包住重算**的 BatchNorm running stats 快照/还原（P4.6b §4.2）。

    为什么必须有它
    --------------
    `nn.BatchNorm2d` 在 `self.training` 下每次前向都会原地更新
    `running_mean` / `running_var` / `num_batches_tracked`。检查点段在反向时会把
    段内前向**再跑一遍**，于是统计被更新两次：`num_batches_tracked` 变成 2、
    `running_mean` 被 `(1-m)²·orig + …` 污染。`ResBlock` 有两个 BN，8 个块就是
    16 次额外更新。

    为什么是「快照 + 原地还原」而不是 brief 建议的 `track_running_stats=False`
    -------------------------------------------------------------------
    `track_running_stats=False` 会**改变反向要存的张量集合**：
    `native_batch_norm` 只有在拿到 `running_mean/running_var` 时才把 `save_mean`
    与 `save_invstd` 留给反向更新统计，实测 saved tensor 数从 9 掉到 7，
    `torch.utils.checkpoint` 的 `determinism_check='default'` 直接抛
    `CheckpointError: A different number of tensors was saved during the original
    forward and recomputation`。要么关掉 determinism 检查（那会连带关掉对真正
    非确定性的防护），要么改用本类。
     归一化数值本身两者**相同**（`training=True` 时 BN 一律按 batch 统计归一化，
    `track_running_stats` 只决定要不要写 buffer）—— 所以本方案不碰计算图，
    只在重算前后把三个 buffer 还原，输出/梯度/`determinism_check` 全部一字不变，
    而 buffer 与单次前向**逐位相等**（含 `num_batches_tracked` 回到 1）。

    用 `context_fn` 的第二个 context（recompute）而不是「数调用次数」
    ------------------------------------------------------------
    `checkpoint(..., use_reentrant=False, context_fn=f)` 的 `f()` 返回
    `(forward_ctx, recompute_ctx)`；`recompute_ctx` **只在重算期间**被进入
    （实测进入 1 次），`forward_ctx` 包原前向。语义正好对上「重算时把统计还回去」。
    """

    def __init__(self, bns):
        self._bns = list(bns)
        self._snap = []
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        self._snap = [
            (bn,
             None if bn.running_mean is None else bn.running_mean.clone(),
             None if bn.running_var is None else bn.running_var.clone(),
             None if bn.num_batches_tracked is None else bn.num_batches_tracked.clone())
            for bn in self._bns
        ]
        return self

    def __exit__(self, exc_type, exc, tb):
        for bn, rm0, rv0, nb0 in self._snap:
            if bn.running_mean is not None and rm0 is not None:
                bn.running_mean.copy_(rm0)
            if bn.running_var is not None and rv0 is not None:
                bn.running_var.copy_(rv0)
            if bn.num_batches_tracked is not None and nb0 is not None:
                bn.num_batches_tracked.copy_(nb0)
        return False


def _collect_batchnorms(blocks):
    """收集段内所有 `_BatchNorm`（含段元素本身就是 BN 的情况）。"""
    out = []
    for b in blocks:
        for m in b.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                out.append(m)
    return out


def compiled_module_paths(module):
    """返回 `module` 子树里被 `torch.compile` 包过的模块名（P4.7 的 D2 形态）。

    P4.7 不再 `torch.compile(整模型)`，而是**就地**把每个 `nn.Linear` 换成
    `torch.compile(...)` 的 `OptimizedModule`；`OptimizedModule` 恒有 `_orig_mod`
    属性（`inference.py` 的 `getattr(model, '_orig_mod', model)`、`train_sft.py` 的
    `_ema_key` 归一化都靠它），所以 `_orig_mod` 就是「这段子图被编译了」的结构判据。
    顺带把 `module` 自身也算进去（整模型被 `torch.compile` 包住的情形）。
    """
    bad = []
    if hasattr(module, '_orig_mod'):
        bad.append('<self>')
    for name, m in module.named_modules():
        if name and hasattr(m, '_orig_mod'):
            bad.append(name)
    return bad


def assert_grad_checkpoint_compile_compatible(module, where=''):
    """梯度检查点与 `torch.compile` 互斥守卫（P4.6b；P4.7b 必须保留）。

    冲突面：`torch.utils.checkpoint` 靠 `saved_tensors_hooks` 工作，而
    `torch.compile` 靠 Dynamo/inductor 的图捕获工作；P4.7 把 `nn.Linear` 逐个编译
    之后，一个检查点段内部会同时出现「编译过的 Linear」与「重算钩子」，收益
    不可预期（段的重算要么被 Dynamo 拆成 graph break、要么整段退化成 eager），
    在 NPU/TorchAir 上还会让图捕获跨过 `saved_tensors_hooks` 这个动态边界。

     覆盖面：这里只能看见**本子树内**的 `_orig_mod`。若整个模型被
    `torch.compile(model)` 包住（`inference.py` 的 CUDA `--compile` 路径），
    `module` 是被包在里面的那个原始模块、自树扫不到 —— 那一条要在**编译的调用点**
    上再查一次，用同一个函数（`assert_grad_checkpoint_compile_compatible(model)`
    在 `torch.compile` 之后调）。本函数在「检查点开着的前向」里也会被调一次，
    所以 P4.7 的逐 Linear 形态在训练第一步之前就会被抓住。
    """
    bad = compiled_module_paths(module)
    if bad:
        raise RuntimeError(
            '梯度检查点与 torch.compile 互斥%s：以下子模块已被 torch.compile 包成 '
            'OptimizedModule（带 _orig_mod）：%s。请二选一 —— 关掉梯度检查点'
            '（model.set_grad_checkpointing(False)）或关掉编译。'
            % ('（%s）' % where if where else '', ', '.join(bad[:8])))


def _segment_runner(blocks, taps_want):
    """造出「按顺序跑 `blocks`、顺带记下若干抽头」的函数。"""
    tapset = set(taps_want)

    def run(*args):
        cur = tuple(args)
        captured = {}
        for i, blk in enumerate(blocks):
            out = blk(*cur)
            # 段内的块不必是同一个签名：单输入块（如 `ResBlock` /
            # `TransformerBlock` / `SEBottleneck`）收 `(x,)`，多输入块收
            # `(x, taps)` 并只返回新的 x。规则：
            # 返回张量 -> 只替换**第一个**实参、保留其余（`taps` 沿途不变）；
            # 返回 tuple -> 实参个数跟着返回值走。不假设整段同签名。
            # 抽头取该块的**输出**（不是下一个块的输入）—— 两者在 i>0 时是同一个
            # 张量对象，只有 i=0 才差一个块，而那正是 P4.1 §7.3 编号歧义最容易
            # 踩错的位置（s1 = ResBlock #1 的输出 ⇒ `tap_positions=(0,)`，不是 stem
            # 的输出）。判据见 test_grad_checkpointing 里那条逐块 hook 对照。
            if i in tapset:
                captured[i] = out[0] if isinstance(out, tuple) else out
            cur = tuple(out) if isinstance(out, tuple) else (out,) + cur[1:]
        if tapset:
            return cur[0], tuple(captured[i] for i in taps_want)
        return cur[0]

    return run


def run_grad_segment(blocks, args, use_checkpoint=False, tap_positions=(),
                     per_block=None, uncapped_last=False, kind=None,
                     guard_root=None):
    """把 `blocks` 当作**一个段**跑；`use_checkpoint` 时整段走梯度检查点。

    Args:
        blocks: 段的模块序列（同一块类型的一段，见 §2 的粒度裁决）。
        args: 段入口的位置参数元组。`ResBlock`/`TransformerBlock`/`SEBottleneck`
            是 `(x,)`；多输入块是 `(x, taps)`。**逐位置传入**，所以不假设签名。
        use_checkpoint: 段是否走检查点。调用方（`GradCheckpointMixin.run_segment`）
            已把 `self.training` 与总开关合并进来。
        tap_positions: 需要额外返回的**块下标**（0-based，升序）—— 取的是这些块的
            **输出**（`tap_positions=(0,)` = 段内第 1 块的输出，不是段入口）。
            这些张量作为检查点段的**额外出参**返回，因此仍参与反向，但段内的其它
            中间激活不留。
        per_block: 逐块检查点（`True`）还是整段合并（`False`）。默认按 kind 取
            `GC_PER_BLOCK_DEFAULT`（只有旧路径为 True，与 `checkpoint_sequential`
            的既有粒度一致）。
        uncapped_last: **段的最后一块不检查点**。这是
            `torch.utils.checkpoint.checkpoint_sequential` 的文档语义
            （"All segments except the last will not store the intermediate
            activations" / 源码注释 "the last chunk has to be non-volatile"），
            旧路径必须照搬，否则 `use_checkpoint=True` 就是一次**未声明的既有
            行为变化**（见 `SharedBackbone.forward` 的注释与
            `task-p4-6b-fix-report.md` B1）。逐块粒度下**不要**开这个选项 ——
            豁免末块等于白放弃最后一块的收益。
        kind: 仅用于取默认粒度与报错信息。
        guard_root: 传模块则每次前向做一次 compile 互斥检查（便宜：一个模型 ~100
            个子模块，微秒级；换来「编译发生在构造之后」也能被抓住）。

    Returns:
        `(out, taps)`；`taps` 是长度为 `len(tap_positions)` 的元组，顺序与
        `tap_positions` 的升序一致。

    `use_reentrant=False` 的理由（不是默认值偷懒，是有依据地选）
    -----------------------------------------------------------
    1. `context_fn` **只在非重入路径可用**（重入路径直接 `ValueError`）——
       `_BatchNormStatGuard` 就挂在这里，这是硬需求。
    2. 重入路径要求「至少一个输入张量 `requires_grad`」，否则整段**静默不产生
       梯度**。非重入路径靠段内的 `saved_tensors_hooks` 建图，段入口不需要
       `requires_grad`（上游被 `no_grad` / 冻结时也不会悄悄丢梯度）。
    3. 非重入支持 `autograd.grad`、非张量出参、以及嵌套检查点；重入路径不支持
       非张量出参（本函数在有抽头时返回 tuple）。
    4. 重入路径不支持关键字参数与 `torch.autograd.function` 的一些边角。
    代价：重入路径对「有副作用的段」更宽容，而本文件的段**没有**副作用
    （BN 的 buffer 写已由 `_BatchNormStatGuard` 兜住）—— 所以这条代价在这里是 0。
    """
    blocks = list(blocks)
    taps_want = sorted(int(p) for p in tap_positions)
    for p in taps_want:
        if not 0 <= p < len(blocks):
            raise ValueError('抽头下标 %d 越界：段里只有 %d 个块' % (p, len(blocks)))
    if per_block is None:
        per_block = GC_PER_BLOCK_DEFAULT.get(kind, False)

    # `active` 里的 `torch.is_grad_enabled()` **不是**死代码，但**在经
    # `GradCheckpointMixin.run_segment` 的路径上确实观察不到**（mixin 的
    # `grad_checkpointing_for` 已经先算过一遍同样的条件，见下）。它守的是
    # `run_grad_segment` 这个**公开函数被直接调用**的路径 —— 那条路上没有 mixin
    # 也没有 `self.training`，`use_checkpoint` 是调用者自己传的。这不是理论上的
    # 假设：`tests/test_grad_checkpointing.py::
    # test_run_grad_segment_itself_never_checkpoints_under_no_grad` 就走这条路，
    # 而把本行删掉会让那条测试变红。两条门必须**都**在：mixin 的门负责
    # eval/推理的零行为变化，本行的门负责直接调用者的 no_grad 契约。
    active = bool(use_checkpoint) and torch.is_grad_enabled()
    if active and guard_root is not None:
        assert_grad_checkpoint_compile_compatible(
            guard_root, 'kind=%s' % kind if kind else '')

    # `uncapped_last`：最后一块**不进**任何检查点段（`checkpoint_sequential`
    # 的文档语义）。段长 <= 1 时于是「无可检查点」，与 torch 在单段时的
    # 落空行为（`range(0, 0, 1)` 空循环 → 直接跑 `functions[-1]`）一致。
    capped = blocks[:-1] if uncapped_last else blocks

    if not active or not capped:
        ret = _segment_runner(blocks, taps_want)(*tuple(args))
        if taps_want:
            return ret[0], tuple(ret[1])
        return ret, ()

    if not per_block:
        out = _checkpointed(_segment_runner(capped, taps_want), args,
                            _collect_batchnorms(capped))
        if taps_want:
            return out[0], tuple(out[1])
        return out, ()

    want = set(taps_want)
    cur = tuple(args)
    captured = {}
    for i, blk in enumerate(blocks):
        if i < len(capped):
            o = _checkpointed(_segment_runner([blk], ()), cur,
                              _collect_batchnorms([blk]))
        else:
            o = blk(*cur)
        new_x = o[0] if isinstance(o, tuple) else o
        if i in want:
            captured[i] = new_x
        cur = tuple(o) if isinstance(o, tuple) else (new_x,) + cur[1:]
    if not want:
        return cur[0], ()
    return cur[0], tuple(captured[i] for i in taps_want)


def _autocast_like(t):
    """按**区域输入张量的 dtype** 造一个等价的 autocast 上下文（重算期间用）。

    为什么不用「查 autocast 状态」：2026-10-01 的 OOM 根因就是重算不在 autocast
    里（`backward()` 时前向的 `with autocast(...)` 早已退出），而查状态这条路
    不可靠 —— `torch.is_autocast_enabled()` 只反映 **CUDA** 的 autocast，对
    `device_type='npu'` 读不到（torch_npu 自己实现 autocast），实测在本地就抓到
    None ⇒ 修复静默失效。

    输入张量的 dtype 反而是**最准**的信号：它就是前向在该边界实际产出的精度。
      输入 fp16/bf16 ⇒ 前向在低精度下 ⇒ 重算也该在低精度下；
      输入 fp32      ⇒ 前向本来就是全精度 ⇒ 不需要任何 autocast。
    dtype 沿参数推进时可能被提升（如 Mamba 的 `A_log` 是 fp32 参数，
    `dt(fp16) * A(fp32)` → fp32），这由 ops 自己的类型提升处理，与 autocast 无关。

    抓不到 / 构造失败一律退回 `nullcontext` ⇒ 最坏是「没改善」，不会更差。
    """
    if not isinstance(t, torch.Tensor):
        return contextlib.nullcontext()
    if t.dtype not in (torch.float16, torch.bfloat16):
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type=t.device.type, dtype=t.dtype)
    except Exception:  # noqa: BLE001
        return contextlib.nullcontext()


def _recompute_context_fn(ac, guard):
    """`torch.utils.checkpoint(context_fn=…)` 的工厂 —— **必须定义在模块级**。

    为什么不能在 `_checkpointed` 里就地 `def context_fn()`
    ----------------------------------------------------
    Dynamo 对 `context_fn` 只认两种形态（torch 源码
    `torch/_dynamo/variables/higher_order_ops.py:3849-3861`）::

        ctx = kwargs.pop("context_fn")
        if isinstance(ctx, UserFunctionVariable):        # 模块级 def
            ...
        elif isinstance(ctx, FunctoolsPartialVariable):  # functools.partial
            ...
        else:
            raise NotImplementedError(
                f"checkpoint not implemented for {type(ctx)} context_fn")

    函数体内就地定义的 `context_fn` 得到的是 `NestedUserFunctionVariable`，其 MRO 是
    `BaseUserFunctionVariable → VariableTracker`，**不是** `UserFunctionVariable`
    的子类 ⇒ isinstance 必然落空。于是 GC 一旦与 `torch.compile` 同用（块级/整模型
    编译），前向第一帧就抛上面那条 `NotImplementedError`。注意行尾的
    ``" context_fn"`` 是**消息里的字面量**，``type(ctx)`` 指的才是实参类型 ——
    这条报错说的不是被 checkpoint 的那个函数。

    闭包里的 `ac` / `guard` 改由参数传入，调用方用 `functools.partial` 绑定
    （对应上面第二个分支）。

    实测：`tmp/coding/repro_context_fn.py` 的 3×4 矩阵证明**把被 checkpoint 的
    函数改写成纯模块级函数治不了这个错**（那样仍然 FAIL），改 `context_fn` 才行；
    `tmp/coding/repro_ctxfn_real_guard.py` 用**真的** `_BatchNormStatGuard` 复验 ——
    过 Dynamo、输出与梯度与原写法逐位相等、`num_batches_tracked` 仍为 1，
    而「不传 context_fn」则会挂在 `modified by an inplace operation`（即守卫必需）。
    """
    @contextlib.contextmanager
    def recompute_ctx():
        # 重算期间依次做两件事，**两者都不能省**：
        #   1. 恢复前向的精度（fp16/bf16）—— 否则整段退回 fp32，体积翻倍，
        #      这是 4 卡 910A OOM 的直接原因（`Tried to allocate 1.40 GiB`
        #      恰是 (1000,4,46,32,64) 的 fp32 体积）；
        #   2. BN guard 把统计还回去 —— P4.5b 为 `determinism_check` 装的
        #      （前向与重算保存的张量数必须一致，否则 CheckpointError）。
        with ac:
            with guard:
                yield

    # 第二个 context 只在**重算期间**进入（实测进入 1 次），第一个包原前向。
    # 传的是 `recompute_ctx()` —— **实例**不是工厂：checkpoint 拿到的
    #   是「已构造好的上下文管理器」，直接 `with` 它。
    return contextlib.nullcontext(), recompute_ctx()


def _first_tensor(args):
    """`args` 里第一个 `torch.Tensor`，没有则 None。

    为什么不用 `next((a for a in args if isinstance(a, torch.Tensor)), None)`：
    **那个写法会让每个检查点段各断一次图。** `next(genexpr, default)` 是**两个**
    位置参数，Dynamo 的 builtin 处理器只认有限几种 arity，落到
    `torch/_dynamo/variables/builtin.py` 的 `call_next` 分支上就报
    `incorrect arg count ... too many positional arguments and no constant
    handler` 并**放弃整段**。而本函数在 `_checkpointed` 里、被 Dynamo 追踪，
    于是 A100 上开 `--compile` + `--gc-with-compile 1` 时，每个 block 边界
    都断一次 —— 融合收益被吃光，而日志里只是一行 WARNING。

    改写成显式循环：对**静态 tuple** 的 `for` + `isinstance` 是 Dynamo 的
    基本功（直接展开），不会断图。语义与原来逐位相同。
    """
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
    return None


def _checkpointed(runner, args, bns):
    guard = _BatchNormStatGuard(bns)
    _ac = _autocast_like(_first_tensor(args))
    # `context_fn` 走 functools.partial（Dynamo 认 FunctoolsPartialVariable）。
    # **不要**改回就地 `def context_fn()` —— 那会变成 NestedUserFunctionVariable，
    # 与 torch.compile 同用时前向直接抛 NotImplementedError。缘由见
    # `_recompute_context_fn` 的文档；`tests/test_katago_v7_grad_checkpointing.py`
    # 的 `test_heads_recompute_under_the_same_autocast_as_the_forward` 盯重算精度，
    # `test_heads_recompute_is_safe_when_a_probe_needs_no_sync` 盯本函数源码里
    # 必须保留 `context_fn` 与 `preserve_rng_state=True` 两处字面量。
    context_fn = functools.partial(_recompute_context_fn, _ac, guard)

    return torch.utils.checkpoint.checkpoint(
        runner, *tuple(args),
        use_reentrant=False,
        preserve_rng_state=True,
        context_fn=context_fn,
    )


class GradCheckpointMixin:
    """按块类型分组的梯度检查点开关（P4.6b）。

    为什么是 mixin 而不是某个主干容器的一个方法
    ----------------------------------------
    现役挂载点是 `SharedBackbone`；新架构（KataGo SE-bottleneck）走的也是它。
    而 `SharedBackbone.__init__` 的签名被
    `tests/test_katago_se.py::test_legacy_class_signatures_untouched`
    逐字锁死（不能加 kwarg）—— 所以开关只能是**构造后**可调的方法。

    开关形态
    --------
    * 不带任何 CLI 参数，由模型属性控制（`SharedBackbone.use_checkpoint` 的
      setter 就是它的入口）。
    * 默认值见 `GRAD_CHECKPOINT_DEFAULTS`：`res`/`transformer` = True，
      `legacy` = False（不改变既有路径的行为）。
    * `set_grad_checkpointing()` 可以在**不重建模型**的情况下切换 —— 属性不进
      `state_dict`（既不是 parameter 也不是 buffer），所以切换前后存档逐位相同。

     **开关属于「调用 `run_segment` 的那个模块」**（接线必读）
    ----------------------------------------------------------
    开关是**普通实例属性**，不自动向子模块传播。若某个全网类持有 mixin 而
    `forward` 只调 `self.backbone(x)`，那么 `net.set_grad_checkpointing(True)`
    只会改到 net 自己，主干容器里的 `run_segment` 看不到 —— **静默不生效**。
    正确形态二选一：
      (a) mixin 只挂在**主干容器**上，调用方走
          `net.backbone.set_grad_checkpointing(...)`（`SharedBackbone` 就是这一条）；
      (b) mixin 挂在 net 上，但 net 必须**显式转发**给 `self.backbone`。
    两条路都要求「开关的持有者 == `run_segment` 的调用者」。测试
    `test_switch_owner_must_be_the_module_that_calls_run_segment` 把这个坑钉住。
    """

    GRAD_CHECKPOINT_KINDS = (GC_RES, GC_TRANSFORMER, GC_LEGACY)

    #: 逐 kind 的默认开关。**做成类属性**是为了让子类能扩自己的 kind
    #: （V7 加了 `stem` / `heads`，见 `src/networks/katago_v7.py`）而不用去改
    #: 这张模块级表——那张表被 `tests/test_grad_checkpointing.py` 逐字钉着。
    #: 默认仍指向模块级常量，所以 `SharedBackbone` 的行为一个字节都不变。
    GRAD_CHECKPOINT_DEFAULTS = GRAD_CHECKPOINT_DEFAULTS

    def _init_grad_checkpointing(self, enabled=True, **kinds):
        unknown = sorted(set(kinds) - set(self.GRAD_CHECKPOINT_KINDS))
        if unknown:
            raise ValueError(
                '未知的块类型 %s；可用的是 %s'
                % (unknown, list(self.GRAD_CHECKPOINT_KINDS)))
        self._gc_enabled = bool(enabled)
        self._gc_kinds = dict(self.GRAD_CHECKPOINT_DEFAULTS)
        for k, v in kinds.items():
            if v is not None:
                self._gc_kinds[k] = bool(v)

    @property
    def grad_checkpointing(self):
        """总开关。`False` 时所有段都不走检查点（`eval` / 推理恒为「不生效」）。"""
        return getattr(self, '_gc_enabled', False)

    def grad_checkpointing_kinds(self):
        """返回逐类型的开关副本（`dict`，改它不影响模型）。"""
        return dict(getattr(self, '_gc_kinds', GRAD_CHECKPOINT_DEFAULTS))

    def set_grad_checkpointing(self, enabled=None, **kinds):
        """在**不重建模型**的前提下改开关。返回 `self`（便于链式）。

        Args:
            enabled: `True`/`False` 改总开关；`None`（默认）表示不动。
            **kinds: 逐类型覆盖，如 `res=False`。值 `None` 表示不动。

        打开时会立刻做一次 compile 互斥检查（`torch.compile` 若已在模型里生效
        就地报错），避免「训练跑了几百步才发现两条优化互相抵消」。
        """
        self._init_grad_checkpointing(
            enabled=self.grad_checkpointing if enabled is None else enabled,
            **kinds)
        if self.grad_checkpointing:
            assert_grad_checkpoint_compile_compatible(
                self, 'set_grad_checkpointing')
        return self

    def grad_checkpointing_for(self, kind):
        """某一段在**当前**是否真的走检查点。

        `eval` / `no_grad` / `inference_mode` 下一律 False —— 「检查点只在
        `self.training` 且启用时生效」是 eval/推理零行为变化的**唯一**保证。
        """
        if kind not in self.GRAD_CHECKPOINT_KINDS:
            raise ValueError('未知的块类型 %r' % (kind,))
        if not self.grad_checkpointing:
            return False
        if not bool(getattr(self, '_gc_kinds', {}).get(kind, False)):
            return False
        return bool(self.training) and torch.is_grad_enabled()

    def run_segment(self, blocks, args, kind, tap_positions=(), per_block=None,
                    uncapped_last=False):
        """跑一段。`kind` 决定是否检查点；返回 `(out, taps)`，语义同
        `run_grad_segment`。主干 forward 唯一要调的入口。

        `uncapped_last=True` 只给**旧路径**用（照搬 `checkpoint_sequential`
        的「最后一块不检查点」）；逐块粒度的段不要传。"""
        on = self.grad_checkpointing_for(kind)
        return run_grad_segment(
            blocks, args,
            use_checkpoint=on,
            tap_positions=tap_positions,
            per_block=per_block,
            uncapped_last=uncapped_last,
            kind=kind,
            guard_root=self if on else None,
        )


# flash-attn 内核的 batch 维参与 CUDA grid 坐标，受 grid y/z 维上限 65535 约束。
# window/sparse 注意力把 batch 展开为 B*G（如 512*128=65536，恰好超限 1），会报
# "CUDA error: invalid configuration argument"。阈值需低于 flash 内核 grid 上限 65535：
# 取 60000，使 ws=7、B=4800 时 B*nW=43200 可走 flash；超过则强制回退
# 手写 math——这些分块调用的 seq 仅 ~50（49 窗口 + 全局 token），math 成本可忽略。
_FLASH_BATCH_LIMIT = 60000  # ws=7, B=4800 -> B*nW=43200 < 60000

# --------------------------------------------------------------------------- #
# NPU 融合注意力（SFA / PFA）：可选加速，**优先于内置 SDPA**，带三重回退
# --------------------------------------------------------------------------- #
# 动机：CANN 的 `npu_fusion_attention`（SFA/PFA）是与 `npu_swiglu` 同源的融合
# kernel，理论收益在 fp32 累加（数值更稳）与更省 workspace。
#
# 为什么必须有探针：**`npu_fusion_attention` 的签名随 torch_npu 版本变**（2.1 的
# `dim_head` 参数、`return_lse`、还有 v2 版本），写死任何一种都会在另一种上直接
# TypeError ⇒ 整训崩。所以：能力探针**逐个候选实跑**，选中第一个不抛的；再用
# 第一次真实调用的 (q,k,v) 与内置 SDPA 做**数值自检**（fp16/bf16 融合 kernel 与
# SDPA 只允许容差内一致，不要求逐位），自检不过就永久回退内置 SDPA。
#
# 三个开关/状态：
#   `set_npu_fusion_attention(False)` ⇒ 完全不尝试（紧急回退；对应 CLI
#     `--npu-sfa 0`，2026-10-06 从环境变量 GOAI_NPU_SFA 搬来）
#   默认 ⇒ 尝试
#   `_sfa_state` ⇒ 进程内一次性探测/自检的结果（variant/checked/ok/why）
_SFA_ENV_ON = True
_sfa_state = {'variant': None, 'checked': False, 'ok': False, 'why': '未探测'}

#: 候选调用形式（按 torch_npu 版本从新到旧）。`scale=None` 时不传该 kw（SDPA 的
#: 原生默认就是 1/sqrt(d)，与本仓调用方在 scale=None 时的约定一致）。
_SFA_VARIANTS = ('scale_causal', 'scale', 'basic')


def _sfa_call(fn, name, q, k, v, scale):
    """按候选名调用融合注意力；返回 out 或 None（该形式不可用）。"""
    if name == 'scale_causal':
        r = fn(q, k, v, scale=scale, causal=False)
    elif name == 'scale':
        r = fn(q, k, v, scale=scale)
    else:
        r = fn(q, k, v)
    if isinstance(r, (tuple, list)):      # (out, lse) / (out, lse, ...)
        r = r[0]
    if isinstance(r, dict):               # 少数版本返回 dict
        r = r.get('out', r.get('output'))
    return r


def _sfa_probe_and_check(q, k, v, scale):
    """首次调用：能力探针 + 数值自检。返回 (ok, variant, why)。"""
    if not _HAS_NPU_RMS_NORM or torch_npu is None:      # torch_npu 缺失的等价判据
        return False, None, 'torch_npu 不可导入'
    fn = getattr(torch_npu, 'npu_fusion_attention', None)
    if fn is None:
        return False, None, 'torch_npu 没有 npu_fusion_attention'
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False, None, f'dtype {q.dtype} 不在 SFA 支持集'
    picked = None
    errs = []
    for name in _SFA_VARIANTS:
        try:
            with torch.no_grad():
                out = _sfa_call(fn, name, q, k, v, scale)
            if out is not None and out.shape == q.shape:
                picked = name
                break
        except Exception as e:  # noqa: BLE001 — 探针要吃下所有版本差异
            errs.append(f'{name}: {type(e).__name__}')
    if picked is None:
        return False, None, '所有候选形式都失败：' + '; '.join(errs)
    # 数值自检：与内置 SDPA 容差内一致（fp16 融合 kernel 不保证逐位）。
    try:
        with torch.no_grad():
            ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
            got = _sfa_call(fn, picked, q, k, v, scale)
        ok = bool(got is not None and torch.allclose(
            got.float(), ref.float(), atol=2e-2, rtol=2e-2))
        if not ok:
            d = (got.float() - ref.float()).abs().max().item() if got is not None else float('inf')
            return False, None, f'自检不一致（max|Δ|={d:.3e}）'
    except Exception as e:  # noqa: BLE001
        return False, None, f'自检抛错：{type(e).__name__}: {e}'
    return True, picked, 'probe+自检通过'


def _sfa_try(q, k, v, scale):
    """在 NPU 上尝试融合注意力；不可用/自检不过/出错 ⇒ None（回退内置 SDPA）。

    **只在 dropout_p == 0 时启用**：SFA 的调用形式里我们不传 dropout（各版本签名
    不统一），若调用方的 dropout_p > 0 却被静默忽略，12 通道那 0.1 的注意力
    dropout 就会「开着但不生效」—— 语义静默改变，不接受。
    """
    if not _SFA_ENV_ON or q.device.type != 'npu':
        return None
    if not _sfa_state['checked']:
        _sfa_state['checked'] = True
        ok, variant, why = _sfa_probe_and_check(q, k, v, scale)
        _sfa_state['ok'], _sfa_state['variant'], _sfa_state['why'] = ok, variant, why
        _logging_mod.getLogger(__name__).info(
            "[_sdpa] NPU 融合注意力(SFA/PFA) 探测：%s | variant=%s",
            why, variant)
        if not ok:
            return None
    elif not _sfa_state['ok']:
        return None
    try:
        with torch.no_grad():
            out = _sfa_call(torch_npu.npu_fusion_attention, _sfa_state['variant'],
                            q, k, v, scale)
        return out
    except Exception as e:  # noqa: BLE001 — 运行期失败永久回退，不刷屏
        _sfa_state['ok'] = False
        _sfa_state['why'] = f'运行期失败：{type(e).__name__}: {e}'
        _logging_mod.getLogger(__name__).warning(
            "[_sdpa] SFA/PFA 运行期失败，本进程内永久回退内置 SDPA: %s", e)
        return None


def set_npu_fusion_attention(enabled: bool) -> None:
    """训练脚本启动时的显式开关（对应 CLI ``--npu-sfa``）。

    传 False 立刻回到「只用内置 SDPA」的状态（等价于把 SFA 永久判定为不可用）。
    """
    global _SFA_ENV_ON
    _SFA_ENV_ON = bool(enabled)


def _sdpa(q, k, v, dropout_p=0.0, use_math=False, scale=None):
    """注意力计算。

    q,k,v: (B, Hh, N, head_dim)。q 已在调用处预乘 scale。
    use_math: 跳过 F.scaled_dot_product_attention 的所有后端，直接走手写 softmax 注意力。
        用于 window 注意力：其 batch 维被展开为 B*N（可能极大，如 512*361≈18万），
        且序列长度仅 ws*ws（很小）。在 Volta(V100, sm_70) 等老架构上 FlashAttention 内核
        不可用（会报 "invalid configuration argument"），故走手写 math 路径最稳；
        window 的 seq=49，49×49 注意力成本可忽略。
        在 Ampere+(A100/H100, sm_80+) 上则由调用方传 use_math=False，自动走
        FlashAttention / Memory-Efficient 后端，速度更快、显存更省，且能被 torch.compile 融合。

    模块级开关 _sdpa_force_math（在 train_sft.py 里按 GPU 能力设置）会覆盖 use_math：
    V100 强制 math，A100 强制走 SDPA 后端。

    注意力后端优先级（非 math 时）：
      1. flash-attn 独立库（set_flash_attn(True) 成功加载时）——最快、显存最低，
         仅 Ampere+ CUDA 可用；
      2. torch 内置 F.scaled_dot_product_attention（自动选 flash/mem-efficient 后端）；
      3. 手写 math（use_math=True 时直接走这条）。
    """
    # 模块级覆盖：训练脚本按 GPU 能力设置（A100 走 Flash，V100 走 math）
    if _sdpa_force_math:
        use_math = True
    # batch 超过 flash 内核 grid 上限时强制 math（内置 SDPA 的 flash/mem-efficient
    # 后端对同配置有相同限制，一并排除）。典型触发：window/sparse 的 B*G=65536。
    if q.shape[0] > _FLASH_BATCH_LIMIT:
        use_math = True
    if not use_math and _flash_attn_func is not None:
        # flash-attn 只接受 fp16/bf16。正常由 autocast 保证 bf16；若上游发生 dtype
        # 泄漏（如 graph-break resume 段的 eager 重算），这里兜底转 bf16，避免
        # "FlashAttention only support fp16 and bf16 data type" 直接崩溃。
        _flash_orig_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        # flash-attn 的 `scale` 在 kernel 里写死为 1/sqrt(head_dim)，Python 侧没有
        # 参数可传。调用方若传了别的 `scale`，必须在这里手动预乘 q 把它抵消掉，
        # 否则 SDPA 路径（已透传 scale）和 flash 路径会给出不同结果。
        if scale is not None:
            q = q * scale * math.sqrt(q.shape[-1])
        # flash-attn 独立库：要求 (B, S, Hh, d) 布局（头维 -2、head_dim -1）。
        # 我们的 (B, Hh, N, d) 中 head_dim 为最内层（stride=1），transpose 后满足
        # flash-attn 的 last-dim contiguous 要求，无需显式 .contiguous() 拷贝。
        out = _flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=dropout_p, causal=False)
        out = out.transpose(1, 2)
        # 仅在原输入非 fp16/bf16（被兜底转 bf16）时转回，保持与 math/SDPA 路径一致的
        # dtype；fp16/bf16 输入则维持 flash 原生输出（默认不开 flash，V7 走 force_math）。
        return out.to(_flash_orig_dtype) if _flash_orig_dtype not in (
            torch.float16, torch.bfloat16) else out
    if use_math or not hasattr(F, "scaled_dot_product_attention"):
        return _sdpa_math(q, k, v, dropout_p=dropout_p, scale=scale)
    # ---- NPU 融合注意力（SFA/PFA）：优先于内置 SDPA，失败永久回退 ----
    # dropout_p > 0 时**不启用**（SFA 调用形式里不传 dropout，静默忽略会改变
    # 12 通道那条 0.1 注意力 dropout 的语义）。
    if dropout_p == 0.0:
        _sfa_out = _sfa_try(q, k, v, scale)
        if _sfa_out is not None:
            return _sfa_out
    # SDPA 路径：SDPA 自带默认缩放 1/sqrt(d)，但 PyTorch>=2.1 支持显式 `scale=`。
    # 必须把调用方传入的 `scale` 透传进去，否则本函数会**无条件**套用 1/sqrt(d)：
    #   - 调用方传 `scale=None`（V7 的 MHSA，旧写法预乘过 q）⇒ 恰好 1/sqrt(d)，巧合正确
    #   - 调用方传 `scale=self.scale`（本文件里 815/882/1040 三处）⇒ 被**忽略**，
    #     只剩 1/sqrt(d)。当 self.scale != 1/sqrt(head_dim) 时结果就是错的。
    # scale=None 时保持 SDPA 的原生默认（等价 1/sqrt(d)），向后兼容。
    # 运行时兜底：部分怪 CANN / 老 torch_npu「有 API 但调用即抛 RuntimeError」
    # （如 invalid configuration argument）。放开 NPU SDPA 后若撞上这类版本会直接
    # 整训崩溃，故捕获一次并回退手写 math（与 set_sdpa_force_math 的静态判定互补）。
    try:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                              scale=scale)
    except RuntimeError as _sdpa_err:
        if _sdpa_force_math:
            # 本就强制 math，不该走到这；如实抛出便于定位。
            raise
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[_sdpa] F.scaled_dot_product_attention 运行时失败，回退 math: %s",
            _sdpa_err)
        return _sdpa_math(q, k, v, dropout_p=dropout_p, scale=scale)


# flash-attn 独立库的内核句柄（None=未启用）。由 train_sft.py 启动时按
# 「库已安装 + Ampere+ CUDA」条件调用 set_flash_attn(True) 加载。
_flash_attn_func = None


def set_flash_attn(enabled: bool):
    """尝试加载 flash-attn 独立库内核。返回 (是否启用, 状态描述)。

    enabled=False 直接卸载回退 SDPA；enabled=True 时 import flash_attn，
    成功则 _sdpa 优先走 flash-attn 内核，失败（未安装/导入错误）返回原因并回退。
    注意：window/sparse 注意力不走 flash（形状为每窗口 1×S 微序列），
    flash 仅用于 global/axial 等标准 MHA 形状。
    """
    global _flash_attn_func
    if not enabled:
        _flash_attn_func = None
        return False, "已禁用"
    try:
        import flash_attn  # type: ignore[import-not-found]
        from flash_attn import flash_attn_func  # type: ignore[import-not-found]  # noqa: F401
        _flash_attn_func = flash_attn_func
        return True, "已启用 (v%s)" % getattr(flash_attn, "__version__", "?")
    except Exception as e:
        _flash_attn_func = None
        return False, "不可用: %s" % e


# 模块级开关：是否强制走手写 math 注意力。
# 默认 True（最保守，兼容 V100 等老卡）；train_sft.py 在检测到 Ampere+ 后会设为 False
# 以启用 FlashAttention 后端。窗口/稀疏注意力在 A100 上 batch 维被展开为 B*N，
# 但若序列长度极小（ws²+ng≈数十），较新 torch 的 SDPA 后端能正常处理，无需 math。
_sdpa_force_math = True


def set_sdpa_force_math(flag: bool) -> None:
    """由训练脚本在启动时按 GPU 能力设置。flag=True 强制手写 math（V100）。"""
    global _sdpa_force_math
    _sdpa_force_math = bool(flag)


# 模块级开关：手写 math 注意力的 **query 分块长度**（0 = 关闭，走整条 N×N）。
#
# 为什么粒度要在**算子内部**：Mamba 那边外层循环本来就按 chunk 走，但 `drive`
# 是在进循环之前整条物化的 ⇒ 外层分块对峰值毫无帮助，赢在把乘法搬进循环内部。
# 注意力没有外层循环，整个 N×N 是一次算子，所以必须新增一层循环。
#
# 默认 64：19 路棋盘 N=361 ⇒ 6 块，每份 (B,4,64,361) fp16 在 B=1000 下
# 0.17 GiB（原 0.97）。设 0 或 ≥N 即退回原路径（逐位不变）。
_attn_query_chunk = 64

#: 逐 chunk 梯度检查点（2026-10-04 新增）。见 `_sdpa` math 分块循环里的说明。
#: 默认开：它只影响**保留**的显存（反向时逐块重算），数学上逐位等价，唯一
#: 例外是 `dropout_p > 0` 时 mask 取样位置会变（分布等价）。设 0 关闭。
_attn_chunk_checkpoint = True


def set_attn_chunk_checkpoint(flag: int) -> None:
    """开关逐 chunk 检查点（0 = 关闭）。由训练脚本启动时调用。"""
    global _attn_chunk_checkpoint
    _attn_chunk_checkpoint = bool(flag)


def _attn_chunk_fn(qc, kt, v, dropout_p):
    """一个 query 块的注意力（``(B,Hh,chunk,N)``）。

    必须是**模块级函数**：`torch.utils.checkpoint` 只接受可 pickle / 可重入的
    callable，闭包与 lambda 在 `use_reentrant=False` 下也可能被
    `torch.compile` 判为 graph break。
    """
    a = (qc @ kt).softmax(dim=-1)
    if dropout_p > 0.0:
        a = torch.nn.functional.dropout(a, p=dropout_p)
    return a @ v


# ---- online-softmax 注意力（flash 风格，不物化 (Nq,Nk) 注意力矩阵）----
#: 是否启用 online-softmax 注意力（False ⇒ 走 `_sdpa_math` 的 materialize 实现）。
#: 由 `set_attn_online` / 训练脚本（`GOAI_ATTN_ONLINE=1`）设置。
#:
#: **默认 False**，虽然它已可用且有测试覆盖。三个理由：
#:
#: 1. **它不是逐位等价的** —— online 走 rescale 累加、materialize 走一次
#:    softmax，算术次序不同，实测 fp32 下 `max|Δ| ≈ 7e-07`。而
#:    `tests/test_mcts_in_channels.py::test_twelve_channel_path_bit_identical`
#:    正是靠 12 通道路径**逐位一致**来发现意外的数值变化；默认打开会让那条
#:    基线失效（实测 `search50_value` 漂到第 8 位有效数字）。
#: 2. **真机未验证** —— bf16/fp16 内核在 910A 上的实际加速比与数值稳定性都还没有
#:    数据，而 12 通道的 window/sparse 路径也走这里。
#: 3. `dropout_p > 0` 时它本来就会回退 materialize（V7 的 `attn_dropout` 默认
#:    0.0，所以这条不构成日常约束）—— 也就是说它**总是**在关键路径上生效，
#:    默认开关必须保守。
_attn_online = False


def set_attn_online(flag):
    """开关 online-softmax 注意力（False = 走 materialize 实现）。

    打开前请先跑 `tests/test_online_softmax_attn.py`（前向等价 + 梯度 gradcheck），
    并知道它会让 12 通道的 bit-identical 基线失效。
    """
    global _attn_online
    _attn_online = bool(flag)


class _OnlineSoftmaxAttn(torch.autograd.Function):
    """flash 风格 online-softmax 注意力（纯 math，不物化 (Nq,Nk) 注意力矩阵）。

    forward 在 ``no_grad`` 下沿 **K 维**流式分 tile 累积输出与统计量 ``(m, l)``，
    注意力权重 ``P`` 不进入 autograd 图；backward 沿 K 重算 ``P`` 并用标准
    softmax 反向公式累积梯度。``dropout_p`` 必须为 0 —— 训练态带 dropout 时
    调用方应回退 `_sdpa_math` 的 materialize 路径（保证 dropout mask 行为）。

    数值上 online 的 ``P = exp(s-m)/l`` 与标准 ``softmax(s)`` 等价（fp 下仅细微
    差异），故 ``dropout=0`` 时与原 materialize 路径**逐位等价**。相比 materialize
    路径它还带来两个好处：不把 ``(Nq,Nk)`` 矩阵留在 autograd 图里（省反向保留
    显存），且 online rescaling 在 fp16 下比标准 softmax 更不易溢出。
    """

    @staticmethod
    def forward(ctx, q, k, v, scale):
        # q:(B,Hh,Nq,d) k:(B,Hh,Nk,d) v:(B,Hh,Nk,dv)
        if scale is None:
            scale = 1.0
        scale = float(scale)
        B, Hh, Nq, d = q.shape
        Nk = k.shape[-2]
        dv = v.shape[-1]
        tile = Nk if Nk <= 512 else 512
        # ---- 统计量与累加的 dtype：**至少 fp32**，但不降精度 ----
        # `promote_types(q.dtype, float32)` 而不是写死 `float32`：
        #  - fp16 / bf16 → fp32：autocast 下 `q @ kj.T` 出 bf16，若 m/l/acc 按
        #    `q.dtype` 建就是 fp32 减 bf16，直接
        #    `RuntimeError: expected m1 and m2 to have the same dtype`
        #    （V7 head 重算那条用例就是这么炸的）；
        #  - fp32 → fp32；
        #  - **fp64 → fp64**：写死 fp32 会把 `gradcheck`（用 double）打挂 ——
        #    `dS @ kj` 变成 fp32 混 fp64。降精度同样是一种静默的错。
        # 而 **matmul 始终留在计算 dtype**（autocast 的低精度内核照用），不为
        # 数值稳就把 GEMM 拉回 fp32 那种慢路径。
        acc_dtype = torch.promote_types(q.dtype, torch.float32)
        m = torch.full((B, Hh, Nq, 1), float('-inf'), dtype=acc_dtype,
                       device=q.device)
        l = torch.zeros((B, Hh, Nq, 1), dtype=acc_dtype, device=q.device)
        acc = torch.zeros((B, Hh, Nq, dv), dtype=acc_dtype, device=q.device)
        with torch.no_grad():
            for j in range(0, Nk, tile):
                kj = k[..., j:j + tile, :]            # (B,Hh,tile,d)
                vj = v[..., j:j + tile, :]            # (B,Hh,tile,dv)
                # GEMM 走计算 dtype（autocast 内核），随后升到 acc_dtype 做统计
                s = ((q @ kj.transpose(-2, -1)) * scale).to(acc_dtype)
                m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
                p = torch.exp(s - m_new)              # ∈(0,1]
                corr = torch.exp(m - m_new)           # 首轮为 exp(-inf)=0
                l = l * corr + p.sum(dim=-1, keepdim=True)
                # p 回到 vj 的 dtype 再做第二个 GEMM（acc 仍按 acc_dtype 累加）
                acc = acc * corr + (p.to(vj.dtype) @ vj).to(acc_dtype)
                m = m_new
            out = acc / l
        ctx.save_for_backward(q, k, v, acc, l, m)
        ctx.scale = scale
        ctx.acc_dtype = acc_dtype
        # 输出回落到 q.dtype：调用方（MHSA）拿到的必须是它传进来的那个 dtype，
        # 否则下一层 Linear 在 autocast 下会遇到意外 promotion。
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, d_out):
        q, k, v, acc, l, m = ctx.saved_tensors
        scale = ctx.scale
        acc_dtype = ctx.acc_dtype
        Nk = k.shape[-2]
        tile = Nk if Nk <= 512 else 512
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        O = acc / l
        # d_out 是 q.dtype（可能低精度），D 必须按 acc_dtype 算 —— 与 forward 同口径
        D = (d_out.to(acc_dtype) * O).sum(dim=-1, keepdim=True)   # (B,Hh,Nq,1)
        with torch.no_grad():
            for j in range(0, Nk, tile):
                kj = k[..., j:j + tile, :]
                vj = v[..., j:j + tile, :]
                s = ((q @ kj.transpose(-2, -1)) * scale).to(acc_dtype)
                p = torch.exp(s - m) / l               # = softmax(s)
                dP = (d_out @ vj.transpose(-2, -1)).to(acc_dtype)
                dS = p * (dP - D)                      # softmax 反向
                # dS 回到 kj 的 dtype 再做 GEMM：梯度按入参 dtype 累加，
                # 不把 fp32 的中间结果直接加进 fp16 的 dq（那是隐式降精度）。
                dq = dq + (dS.to(kj.dtype) @ kj) * scale
                dk = dk + (dS.to(q.dtype).transpose(-2, -1)
                           @ (q * scale))
                dv = dv + (p.to(d_out.dtype).transpose(-2, -1) @ d_out)
        return dq, dk, dv, None


def set_attn_query_chunk(n: int) -> None:
    """设置 query 分块长度（0/负数 = 关闭）。由训练脚本启动时调用。"""
    global _attn_query_chunk
    n = int(n)
    _attn_query_chunk = n if n > 0 else 0


# 模块级开关：是否将 window/sparse 注意力排除出 torch.compile 图。
# V100(sm_70)/torch2.1 上 F.unfold + 动态 view/permute 链会让 inductor 触发
# PolynomialError，必须 disable；Ampere+(A100) 上 inductor 成熟，能正常编译 unfold，
# 故不 disable，使整个注意力被编译融合，提速更明显。
_compile_disable_sparse = True


def set_compile_disable_sparse(flag: bool) -> None:
    """由训练脚本按 GPU 能力设置。flag=True 时 window/sparse 注意力排除编译（V100）。"""
    global _compile_disable_sparse
    _compile_disable_sparse = bool(flag)


def _run_with_optional_disable(fn, *args):
    """运行时按开关决定是否将 fn 排除出 torch.compile 图。

    注意：不能用装饰器在类定义时静态包装——装饰器在 import 求值时模块开关还是
    默认值 True，运行时 set_compile_disable_sparse(False)（A100）无法撤销已应用的
    torch._dynamo.disable。disable 会造成 graph break，break 后的 eager resume 段
    中 autocast 已退出，qkv Linear 以 FP32 执行，进而让 flash-attn 收到 fp32 报错。
    这里改为调用时动态包装。注意不能每次调用都 torch.compiler.disable(fn)——
    那会为每次调用生成新包装器对象，dynamo 视其为新函数反复 trace，64 次后触发
    cache_size_limit 告警并整体放弃。此处按 fn（bound method 按 __func__+__self__
    相等）缓存包装器，实例数有限（每注意力块一个），trace 一次后稳定复用。
    """
    if _compile_disable_sparse:
        wrapped = _disabled_wrapper_cache.get(fn)
        if wrapped is None:
            wrapped = torch.compiler.disable(fn)
            _disabled_wrapper_cache[fn] = wrapped
        return wrapped(*args)
    return fn(*args)


# disable 包装器缓存：key 为 bound method（__eq__ 按 __func__+__self__，可命中）
_disabled_wrapper_cache = {}


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力，支持三种模式以平衡速度与长程建模能力：

    - mode="global" : 标准全局全配对注意力（最贵，长程最强）
    - mode="window" : 滑动窗口局部注意力（最快，复杂度 O(N·w²)）
    - mode="axial"  : 轴向注意力，先按行、再按列两次 1D 注意力
                      （保长程、复杂度约 O(2N·√N)，围棋网格友好）

    内部统一使用 torch.nn.functional.scaled_dot_product_attention，
    在支持的 GPU 上自动走 FlashAttention / Memory-Efficient 路径，
    不物化 N×N 注意力矩阵，显著降低显存与耗时；CPU 自动回退。
    """

    def __init__(self, channels, num_heads=4, dropout=0.0,
                 mode="global", window_size=7):
        super(MultiHeadSelfAttention, self).__init__()
        assert channels % num_heads == 0, "channels 必须能被 num_heads 整除"
        if mode in ('window', 'sparse', 'window_global'):
            assert window_size % 2 == 1, f"window_size 必须为奇数，收到 {window_size}"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.mode = mode
        self.window_size = window_size

        self.ln1 = RMSNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.attn_drop = dropout

        self.ln2 = RMSNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.ffn_drop = nn.Dropout(dropout)

    @property
    def attn_drop_p(self):
        """当前阶段**实际生效**的注意力 dropout 概率：eval/inference 阶段返回 0.0。

        为什么需要这个派生属性：注意力 dropout 走的是**函数式** API
        （`F.scaled_dot_product_attention(..., dropout_p=p)` /
        `F.dropout(x, p)` / `_flash_attn_func(..., dropout_p=p)`），
        它们的 `training` 形参**默认 True**，而 `_sdpa` 是模块级函数、内联的
        `F.dropout` 在方法体里，两者都拿不到 `self.training` → `model.eval()`
        关不掉注意力 dropout，SFT 评估的 logits 会逐次抖动、最佳模型选择是在噪声上做的。
        同 block 的 `ffn_drop = nn.Dropout(...)`（模块式）本来就自动遵守 `self.training`，
        注意力这一路是唯一的例外；MindSpore 孪生实现同样用 `nn.Dropout` 模块。
        故所有 5 个注意力 dropout 站点的取值一律经由本属性。

        训练期（`self.training is True`）返回 `self.attn_drop`，与修复前逐位相同。
        """
        return self.attn_drop if self.training else 0.0

    def _to_heads(self, t, B, N):
        # t: (B, N, C) -> (B, Hh, N, head_dim)
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _global_attn(self, q, k, v):
        return _sdpa(q, k, v, dropout_p=self.attn_drop_p, scale=self.scale)

    def _local_windows(self, t, H, W):
        """真 2D 局部窗口提取（2026-09 语义修正版）。

        返回 (B, N, Hh, ws², d)：第 n=(i,j) 个位置对应以其为中心的 ws×ws 窗口，
        kernel 顺序 kh*ws+kw，越界补零。与 F.unfold 的 im2col 真窗口逐位一致。

        实现：F.pad + Tensor.unfold（纯 strided view，零拷贝取窗）+ 一次满带宽
        contiguous 拷贝，替代「F.unfold im2col + view/permute 二次重排」——
        比旧路径少一次整块拷贝，且消除旧 view(B,Hh,d,N,ws²) 的 kernel/position
        divmod 交换 bug（旧「窗口」实为 raster 展平序列上起点 (n·ws²) mod N 的
        1D 循环滑窗，并非 2D 局部窗口）。
         语义与旧 checkpoint 不兼容（旧权重在 scramble 语义下训练，需重训/重评估）。
        """
        ws = self.window_size
        B, Hh, N, d = t.shape
        pad = ws // 2
        tp = F.pad(t.reshape(B, Hh * d, H, W), (pad, pad, pad, pad))
        tv = tp.unfold(2, ws, 1).unfold(3, ws, 1)           # (B,C,H,W,kh,kw) 纯 view
        del tp  # unfold view 已建立，padded 输入不再需要，提前释放 ~460MB
        tv = tv.reshape(B, Hh, d, H, W, ws, ws) \
               .permute(0, 3, 4, 1, 5, 6, 2)                # (B,H,W,Hh,kh,kw,d)
        return tv.reshape(B, N, Hh, ws * ws, d)             # 满带宽拷贝

    def _window_attn(self, q, k, v, H, W):
        """块状窗口注意力（2026-09 定稿，Swin 风格，替代滑动窗口）。

        将 H×W pad 到 ws 整除后切成不重叠 ws×ws 窗口，窗口内 token 全配对
        做标准多头注意力（window partition + SDPA），每 token 与同块内
        ws²-1 个邻居交互（块边界两侧不互看，棋盘外 pad 补零）。

        为什么替代滑动窗口（A100 profiler 驱动）：
          - 滑动窗口需为每个位置展开 ws² 个 key（窗口张量 (B*N,Hh,ws²,d)，
            ws=7/batch640 时 2.9GB/个 ×2(k,v)×4 层），copy_/clone/reshape
            搬运占 CUDA ~45%；且 (1×d)@(d×ws²) 的 bmm 退化为 batched GEMV，
            cuBLAS 走 gemvx/sm_75 低效内核（bmm 家族占 CUDA ~67%）。
          - 块状窗口的 partition 重排仅 O(N·C)（滑动窗口的 1/ws² 搬运量），
            注意力恢复为标准 MHA 形状 (B*nW, Hh, ws², ws²)——直接走
            FlashAttention / SDPA 高效内核，注意力矩阵不物化。
          - 语义变更：滑动 → 块状。 与滑动窗口 checkpoint 不兼容，需重训。

        q,k,v: (B, Hh, N, head_dim)（q 已预乘 scale），N=H*W。返回 (B, N, C)。
        """
        ws = self.window_size
        B, Hh, N, d = q.shape
        H2 = ((H + ws - 1) // ws) * ws
        W2 = ((W + ws - 1) // ws) * ws
        ph, pw = H2 - H, W2 - W
        nwH, nwW = H2 // ws, W2 // ws
        nW = nwH * nwW

        def part(t):   # (B,Hh,N,d) -> (B*nW, Hh, ws², d)
            x = t.view(B, Hh, H, W, d)
            if ph or pw:
                x = F.pad(x, (0, 0, 0, pw, 0, ph))   # (d 无, W 右, H 下)
            x = x.view(B, Hh, nwH, ws, nwW, ws, d)
            return x.permute(0, 2, 4, 1, 3, 5, 6) \
                    .reshape(B * nW, Hh, ws * ws, d)

        def unpart(o):  # (B*nW, Hh, ws², d) -> (B, N, Hh*d)
            x = o.view(B, nwH, nwW, Hh, ws, ws, d)
            x = x.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, Hh, H2, W2, d)
            if ph or pw:
                x = x[:, :, :H, :W, :]               # 裁掉 pad（零化仅 ~40MB）
            return x.permute(0, 2, 3, 1, 4).reshape(B, N, Hh * d)

        return unpart(_sdpa(part(q), part(k), part(v), dropout_p=self.attn_drop_p, scale=self.scale))

    def _sparse_attn(self, q, k, v, H, W):
        """稀疏注意力（固定稀疏模式）：局部滑动窗口 + 跨步长全局 token。

        每个 query 位置只与两类 key 交互：
          1) 自身 (ws×ws) 局部窗口内的 token（局部性，同 window）；
          2) 每隔 stride=ws 下采样的「全局代表 token」（长程信息通路）。

        性能设计（2026-09 重写，含语义修正）：
          - 语义修正：旧实现的 view(B,Hh,d,N,ws²) 把 F.unfold 的 kernel 槽与
            position 按 divmod(n·ws²+w, N) 交换了——「窗口」实为 raster 展平
            序列上的 1D 循环滑窗，并非 docstring 宣称的 2D 局部窗口。本版修正
            为以 (i,j) 为中心的真 2D 局部窗口（见 _local_windows）。
             与旧 checkpoint 不兼容（旧权重在 scramble 语义下训练，需重训/重评估）。
          - 性能：k/v 用 pad + Tensor.unfold strided view + 一次满带宽拷贝，
            消除 F.unfold im2col（profiler 占 27.6%）与二次重排；全链零 slice
            分块节点；q 只取窗口中心（= 自身位置，O(N·d) 小拷贝，不再为它做
            O(N·ws²·d) 的全量 unfold）。
          - 显存优化：全局 key/value 不做 expand().reshape()（省 ~426MB/层），
            改用 einsum 广播计算注意力 logits，值聚合也用 einsum 避免物化 expanded tensor。

        q,k,v: (B, Hh, N, head_dim)，N = H*W。返回 (B, N, Hh*d)。
        """
        ws = self.window_size
        stride = ws  # 全局 token 每隔 stride 取一个（块中心）
        B, Hh, N, d = q.shape

        # ---- 1) k/v：真 2D 局部窗口，单次满带宽拷贝（无 im2col、无 slice 节点）----
        kw = self._local_windows(k, H, W).reshape(B * N, Hh, ws * ws, d)
        vw = self._local_windows(v, H, W).reshape(B * N, Hh, ws * ws, d)

        # ---- 2) q 即窗口中心（= 自身位置）：纯 head 主序重排，一次小拷贝 ----
        qc = q.permute(0, 2, 1, 3).reshape(B * N, Hh, 1, d)   # (B*N, Hh, 1, d)

        # ---- 3) 全局代表 token：每 stride×stride 块取中心，覆盖完整棋盘 ----
        # 余数块（底部/右侧不足 stride 的行/列）也取中心，确保无盲区
        gh = (H + stride - 1) // stride  # 向上取整，覆盖所有行
        gw = (W + stride - 1) // stride
        ng = gh * gw
        kg = k.reshape(B, Hh, H, W, d)                       # (B,Hh,H,W,d)
        vg = v.reshape(B, Hh, H, W, d)
        # 为每个块计算中心坐标（clamp 到有效范围）
        row_centers = torch.arange(gh, device=k.device) * stride + stride // 2
        row_centers = row_centers.clamp(max=H - 1)
        col_centers = torch.arange(gw, device=k.device) * stride + stride // 2
        col_centers = col_centers.clamp(max=W - 1)
        # 用 meshgrid 构建 (gh, gw) 索引网格，advanced indexing 取出所有全局 token
        r_idx, c_idx = torch.meshgrid(row_centers, col_centers, indexing='ij')
        kg = kg[:, :, r_idx, c_idx].reshape(B, Hh, ng, d)    # (B,Hh,ng,d)
        vg = vg[:, :, r_idx, c_idx].reshape(B, Hh, ng, d)

        # ---- 4) 注意力 logits：全局 key 用 einsum 广播，避免 expand().reshape() 拷贝 ----
        local_logits = (qc * self.scale) @ kw.transpose(-2, -1)             # (B*N, Hh, 1, ws²)
        qc_5d = qc.view(B, N, Hh, 1, d)
        global_logits = torch.einsum('bnhid,bnhgd->bnhig',
                                     qc_5d * self.scale, kg.unsqueeze(1))  # (B, N, Hh, 1, ng)
        global_logits = global_logits.reshape(B * N, Hh, 1, ng)
        all_logits = torch.cat([local_logits, global_logits], dim=-1)  # (B*N, Hh, 1, ws²+ng)
        attn = all_logits.softmax(dim=-1)
        if self.training and self.attn_drop > 0.0:
            attn = torch.nn.functional.dropout(attn, p=self.attn_drop)

        # ---- 5) 值聚合：local 用 bmm，global 用 einsum 广播（避免 expand vg）----
        local_attn = attn[:, :, :, :ws * ws]                 # (B*N, Hh, 1, ws²)
        global_attn = attn[:, :, :, ws * ws:]                # (B*N, Hh, 1, ng)
        local_out = local_attn @ vw                           # (B*N, Hh, 1, d)
        global_out = torch.einsum('bnhig,bhgd->bnhid',
                                 global_attn.reshape(B, N, Hh, 1, ng),
                                 vg)                          # (B, N, Hh, 1, d)
        global_out = global_out.reshape(B * N, Hh, 1, d)
        oc = local_out + global_out

        # ---- 6) 回到 (B, N, Hh*d)：head 主序展平 ----
        return oc.view(B, N, Hh, d).reshape(B, N, Hh * d)

    def _window_global_attn(self, q, k, v, H, W):
        """窗口注意力 + 全局 token，SDPA 联合注意力加速版。

        融合 window（高效块状分区）+ sparse（全局 token 长程通路）。

        每个 query 只与两类 key 交互：
          1) 同块内 ws² 个局部 key；
          2) 棋盘均匀采样的 ng 个全局 key。

        两类 key 沿序列维拼接后做一次注意力——数学上等价于
        「两路 logits 拼接 → 联合 softmax → 分路聚合再相加」，但
        softmax + 聚合融合进单个内核（CUDA 上自动走 flash / mem-efficient
        后端），不再物化 (BnW, Hh, ws², ws²+ng) 注意力矩阵。

        q,k,v: (B, Hh, N, head_dim)，N = H*W。返回 (B, N, Hh*d)。
        """
        ws = self.window_size
        B, Hh, N, d = q.shape

        # ---- 1) grid 参数 ----
        H2 = ((H + ws - 1) // ws) * ws
        W2 = ((W + ws - 1) // ws) * ws
        ph, pw = H2 - H, W2 - W
        nwH, nwW = H2 // ws, W2 // ws
        nW = nwH * nwW

        # ---- 2) window partition ----
        def part(t):  # (B,Hh,N,d) -> (B*nW, Hh, ws², d)
            x = t.view(B, Hh, H, W, d)
            if ph or pw:
                x = F.pad(x, (0, 0, 0, pw, 0, ph))
            x = x.view(B, Hh, nwH, ws, nwW, ws, d)
            return x.permute(0, 2, 4, 1, 3, 5, 6).reshape(B * nW, Hh, ws * ws, d)

        q_p = part(q)  # (BnW, Hh, ws², d)
        k_p = part(k)
        v_p = part(v)

        # ---- 3) 全局 token 采样（均匀网格，ng ≈ (H/ws)×(W/ws)）----
        stride = ws
        gh = (H + stride - 1) // stride
        gw = (W + stride - 1) // stride
        ng = gh * gw
        row_c = (torch.arange(gh, device=q.device) * stride + stride // 2).clamp(max=H - 1)
        col_c = (torch.arange(gw, device=q.device) * stride + stride // 2).clamp(max=W - 1)
        ri, ci = torch.meshgrid(row_c, col_c, indexing='ij')
        k4 = k.view(B, Hh, H, W, d)
        v4 = v.view(B, Hh, H, W, d)
        kg = k4[:, :, ri, ci].reshape(B, Hh, ng, d)  # (B, Hh, ng, d)
        vg = v4[:, :, ri, ci].reshape(B, Hh, ng, d)

        # ---- 4) 联合注意力：key 序列 = [局部 ws² 个；全局 ng 个]，一次 SDPA ----
        BnW = B * nW
        ws2 = ws * ws
        # 全局 token 广播到每个窗口（与原「两路 logits 联合 softmax」语义一致）。
        # 注意先 permute 把 nW 挪到第 1 维再 reshape——(B,Hh,nW,...) 直接
        # reshape 成 (BnW,...) 会因 Hh 夹在中间而错位。
        kg_w = kg.unsqueeze(2).permute(0, 2, 1, 3, 4).expand(B, nW, Hh, ng, d).reshape(BnW, Hh, ng, d)
        vg_w = vg.unsqueeze(2).permute(0, 2, 1, 3, 4).expand(B, nW, Hh, ng, d).reshape(BnW, Hh, ng, d)
        k_full = torch.cat([k_p, kg_w], dim=2)  # (BnW, Hh, ws²+ng, d)
        v_full = torch.cat([v_p, vg_w], dim=2)
        out = _sdpa(q_p, k_full, v_full, dropout_p=self.attn_drop_p,
                    scale=self.scale)  # (BnW, Hh, ws², d)

        # ---- 5) unpartition ----
        x = out.view(B, nwH, nwW, Hh, ws, ws, d)
        x = x.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, Hh, H2, W2, d)
        if ph or pw:
            x = x[:, :, :H, :W, :]
        return x.permute(0, 2, 3, 1, 4).reshape(B, N, Hh * d)

    def _axial_attn(self, q, k, v, H, W):
        """轴向注意力：先按行、再按列做 1D 自注意力。

        q,k,v: (B, Hh, N, d)，N=H*W。轴向注意力把二维 token 在单轴上交互，
        复杂度约 O(2·N·max(H,W))，远低于 O(N²)，同时保留长程（整行/整列）依赖。
        """
        B, Hh, N, d = q.shape

        def attn_1d(tokens):
            # tokens: (B*Hh*L, S, d) -> 把 Hh 融进 batch 做标准 MHA
            t = tokens.view(-1, Hh, tokens.shape[1], d)
            return _sdpa(t, t, t, dropout_p=self.attn_drop_p, scale=self.scale).view(-1, tokens.shape[1], d)

        # 行注意力：每行 H 个 token 互相看，把 (B,Hh,H,W,d) 重排为 (B*Hh*H, W, d)
        qr = q.view(B, Hh, H, W, d).reshape(B * Hh * H, W, d)
        kr = k.view(B, Hh, H, W, d).reshape(B * Hh * H, W, d)
        vr = v.view(B, Hh, H, W, d).reshape(B * Hh * H, W, d)
        out_r = attn_1d(qr)  # (B*Hh*H, W, d)
        out_r = out_r.view(B, Hh, H, W, d)

        # 列注意力：转置后同理，把 (B,Hh,W,H,d) 重排为 (B*Hh*W, H, d)
        qc = out_r.transpose(2, 3).reshape(B * Hh * W, H, d)
        kc = k.view(B, Hh, H, W, d).transpose(2, 3).reshape(B * Hh * W, H, d)
        vc = v.view(B, Hh, H, W, d).transpose(2, 3).reshape(B * Hh * W, H, d)
        out_c = attn_1d(qc)  # (B*Hh*W, H, d)
        out_c = out_c.view(B, Hh, W, H, d).transpose(2, 3)  # (B,Hh,H,W,d)
        return out_c.reshape(B, N, self.num_heads * d)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        seq = x.flatten(2).transpose(1, 2)  # (B, N, C)

        residual = seq
        h = self.ln1(seq)
        qkv = self.qkv(h)  # (B, N, 3C)
        q, k, v = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        # 注意：不对 q 预乘 scale。_sdpa 的 math 路径内部处理缩放，
        # SDPA/flash 路径自带 1/sqrt(d)，预乘会导致双重缩放。

        if self.mode == "window":
            out = _run_with_optional_disable(self._window_attn, q, k, v, H, W)
        elif self.mode == "window_global":
            out = _run_with_optional_disable(self._window_global_attn, q, k, v, H, W)
        elif self.mode == "axial":
            out = self._axial_attn(q, k, v, H, W)
        elif self.mode == "sparse":
            out = _run_with_optional_disable(self._sparse_attn, q, k, v, H, W)  # (B, N, C)
        else:  # global
            out = self._global_attn(q, k, v)  # (B, Hh, N, d)
            out = out.transpose(1, 2).contiguous().view(B, N, C)

        seq = residual + out

        # 前馈
        residual = seq
        seq = residual + self.ffn_drop(self.ffn(self.ln2(seq)))

        return seq.transpose(1, 2).view(B, C, H, W)


class AttentionResBlock(nn.Module):
    """卷积残差 + 多头自注意力 混合块。

    顺序：卷积残差 -> 自注意力（均带残差）。注意力负责捕捉长程依赖
    （大龙死活、全局厚薄），卷积负责局部形状。
    """

    def __init__(self, channels, num_heads=4, dropout=0.0,
                 attention_mode="global", window_size=7):
        super(AttentionResBlock, self).__init__()
        self.conv = ResBlock(channels)
        self.attn = MultiHeadSelfAttention(
            channels, num_heads=num_heads, dropout=dropout,
            mode=attention_mode, window_size=window_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return x


class SharedBackbone(GradCheckpointMixin, nn.Module):
    """共享表示网络：将棋盘状态编码为隐藏状态。

    注意力模式（attention_mode 控制主干如何堆叠注意力块）：
        - "none" : 全部用纯卷积 ResBlock（最快，局部性最好）
        - "mix"  : 在 num_res_blocks 个块中穿插 num_attention_layers 个
                   AttentionResBlock（推荐：卷积打底 + 注意力提质）
        - "all"  : 全部使用 AttentionResBlock

    注意力块内部的计算模式由 attn_mode 控制（全局/窗口/轴向），
    通过 --attn-mode 配置；窗口大小由 --attn-window 控制。

     mixin 写在 `nn.Module` **前面**（P4.6b fix B5）
    ------------------------------------------------
    mixin 优先于 `nn.Module`。本类的方法名（`_init_grad_checkpointing` /
    `grad_checkpointing` / `set_grad_checkpointing` / `run_segment` /
    `use_checkpoint`）**全部**可能被 `nn.Module` 将来新增的同名成员静默劫持 ——
    `nn.Module` 优先时，mixin 的实现会被 `nn.Module.__getattr__` 之前的正常属性
    查找挡住，症状是「方法不见了」或「调到了 nn.Module 的实现」，而不是报错。
    `run_segment` 这种通用名尤其危险。反过来写（mixin 在后）没有任何好处。
    """

    def __init__(self, in_channels=12, channels=128, num_res_blocks=12,
                 attention_mode="mix", num_attention_layers=4,
                 num_heads=4, attention_dropout=0.0,
                 attn_mode="global", attn_window=7,
                 use_checkpoint=False, arch="convnext",
                 res_blocks=0, convnext_blocks=0, attn_blocks=0):
        """
        Args:
            attention_mode:   主干堆叠模式 "none"|"mix"|"all"
            num_attention_layers: mix 模式下注意力块数量
            num_heads:         多头注意力头数
            attention_dropout: 注意力 dropout
            attn_mode:         注意力计算模式 "global"|"window"|"axial"
            attn_window:       window 模式的窗口边长
            use_checkpoint:    是否启用梯度检查点（显存优化）
            arch:              网络架构风格 "resnet" (默认，向后兼容) | "convnext"
                                 | "se_bottleneck"（KataGo 风格 bottleneck+SE）
            res_blocks:        ResBlock 数量（浅层，局部细节），0 表示使用默认模式
            convnext_blocks:   ConvNeXtBlock 数量（中层，大感受野），0 表示使用默认模式
            attn_blocks:       AttentionResBlock 数量（深层，全局关系），0 表示使用默认模式
        """
        super(SharedBackbone, self).__init__()
        self.channels = channels
        self.attention_mode = attention_mode
        # P4.6b：走与全仓库同一套检查点机制，但默认关（不改变既有路径的行为）。
        # `use_checkpoint=True` 与 `set_grad_checkpointing(True)` 等价 ——
        # 赋这个属性就是走 `use_checkpoint` 的 property setter（见类末尾），
        # 它把总开关与 `GC_LEGACY` 段开关**一起**设成同一个值，不会分叉。
        self.use_checkpoint = use_checkpoint
        self.arch = arch

        # 输入卷积
        if arch == "convnext":
            self.conv1 = nn.Conv2d(in_channels, channels, 7, stride=1, padding=3, bias=False)
            self.norm = LayerNorm2d(channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)

        # 三段式架构优先级：分段配置 > 默认 mix 模式
        if res_blocks > 0 or convnext_blocks > 0 or attn_blocks > 0:
            total_blocks = num_res_blocks
            blocks = self._build_segmented_blocks(
                total_blocks, res_blocks, convnext_blocks, attn_blocks,
                channels, num_heads, attention_dropout, attn_mode, attn_window,
                arch)
        else:
            blocks = self._build_blocks(
                num_res_blocks, attention_mode, num_attention_layers,
                channels, num_heads, attention_dropout, attn_mode, attn_window, arch)
        self.blocks = nn.Sequential(*blocks)

        # 输出层
        if arch == "convnext":
            self.norm_out = LayerNorm2d(channels)
            self.conv_out = nn.Conv2d(channels, channels, 1, bias=False)
        else:
            self.conv_out = nn.Conv2d(channels, channels, 1, bias=False)
            self.bn_out = nn.BatchNorm2d(channels)

    @staticmethod
    def _build_blocks(num_res_blocks, mode, num_attn, channels, num_heads,
                      dropout, attn_mode, attn_window, arch="resnet"):
        if mode == "none" or num_attn <= 0:
            if arch == "convnext":
                return [ConvNeXtBlock(channels) for _ in range(num_res_blocks)]
            if arch == "se_bottleneck":
                return [SEBottleneck(channels) for _ in range(num_res_blocks)]
            return [ResBlock(channels) for _ in range(num_res_blocks)]
        if mode == "all":
            return [AttentionResBlock(channels, num_heads, dropout, attn_mode, attn_window)
                    for _ in range(num_res_blocks)]
        # mix：均匀地把 num_attn 个注意力块插入到卷积块之间
        num_attn = min(num_attn, num_res_blocks)
        attn_idx = set(
            int(round(i * (num_res_blocks - 1) / max(num_attn - 1, 1)))
            for i in range(num_attn)
        )
        blocks = []
        for i in range(num_res_blocks):
            if i in attn_idx:
                blocks.append(AttentionResBlock(
                    channels, num_heads, dropout, attn_mode, attn_window))
            else:
                if arch == "convnext":
                    blocks.append(ConvNeXtBlock(channels))
                elif arch == "se_bottleneck":
                    blocks.append(SEBottleneck(channels))
                else:
                    blocks.append(ResBlock(channels))
        return blocks

    @staticmethod
    def _build_segmented_blocks(total_blocks, res_count, convnext_count, attn_count,
                                 channels, num_heads, dropout, attn_mode, attn_window,
                                 arch="resnet"):
        """三段式架构：浅层 ResNet + 中层 ConvNeXt + 深层 Attention"""
        blocks = []

        # 浅层：ResBlock (局部细节)。`arch="se_bottleneck"` 也落在这个槽位：
        # 段内中/深层分别由 convnext_blocks / attn_blocks 显式指定，不看 arch。
        shallow_cls = SEBottleneck if arch == "se_bottleneck" else ResBlock
        for _ in range(res_count):
            blocks.append(shallow_cls(channels))

        # 中层：ConvNeXtBlock (大感受野)
        for _ in range(convnext_count):
            blocks.append(ConvNeXtBlock(channels))

        # 深层：AttentionResBlock (全局关系)
        for _ in range(attn_count):
            blocks.append(AttentionResBlock(
                channels, num_heads, dropout, attn_mode, attn_window))

        return blocks

    def forward(self, x):
        if self.arch == "convnext":
            out = self.norm(self.conv1(x))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        if self.training and self.use_checkpoint:
            # P4.6b：旧路径保持 `checkpoint_sequential` 的**逐块**粒度
            # （`per_block=True`），但换成本节的新实现 —— 于是多了 BN
            # running stats 的重算保护（旧的 `checkpoint_sequential` 会把
            # `num_batches_tracked` 翻倍、污染 `running_mean/var`）。
            # `uncapped_last=True` 是 B1：`checkpoint_sequential` 的文档语义
            # 是「除最后一段外都检查点」（源码注释 "the last chunk has to be
            # non-volatile"）。不传它，`use_checkpoint=True` 的显存口径就从
            # 「16 段入口」变成「16 个段入口且末块也重算」，v18/v19 的真实
            # 训练显存会低于 31.12GB 锚点（`search_arch.py:63` 的 k 标定基准），
            # 而锚点重测已由业主裁决取消（D15 同期裁决：不重测）⇒ 这里必须
            # 逐字还原旧行为，让锚点与 k=0.8235 继续有效。
            out, _ = self.run_segment(self.blocks, (out,), GC_LEGACY,
                                      uncapped_last=True)
        else:
            out = self.blocks(out)
        if self.arch == "convnext":
            out = self.norm_out(self.conv_out(out))
        else:
            out = F.relu(self.bn_out(self.conv_out(out)))
        return out

    def set_grad_checkpointing(self, enabled=None, **kinds):
        """旧路径的开关入口。`use_checkpoint` 是本类的 property，它读的
        **不是**总开关，而是「总开关 ∧ `legacy` 段开关」—— 见下面 getter。

        `kinds.setdefault(GC_LEGACY, True)` 只在没显式传 `legacy=` 时补
        `legacy=True`：`SharedBackbone` 只用得到 legacy 段，`set_grad_checkpointing(True)`
        若不补，就会变成「总开关 True 而什么都不检查点」（P4.6b fix B4 的
        反面），而 `GRAD_CHECKPOINT_DEFAULTS[GC_LEGACY] is False` 是
        **默认**（不传就关）不是本方法的语义。
        """
        kinds.setdefault(GC_LEGACY, True)
        super(SharedBackbone, self).set_grad_checkpointing(enabled, **kinds)
        return self

    @property
    def use_checkpoint(self):
        """历史上的开关属性（P4.6b 之前 `SharedBackbone.forward` 直接读它）。

        做成 property 而不是普通属性，是为了**堵住「直接赋值静默失效」这个坑**：
        若它只是 `self.__dict__` 里的一个 bool，而 `forward` 改读 mixin 的
        `_gc_enabled`，那么 `model.backbone.use_checkpoint = True`（既有代码里
        常见的写法）就会变成**看起来开了、其实没开**的静默回退。setter 把赋值
        路由进 mixin，读写两侧因此永远一致。

        读的是「总开关 ∧ legacy 段开关」（P4.6b fix B4）：只读总开关会在
        `set_grad_checkpointing(True, legacy=False)` 时读出 True 而实际什么都不
        检查点 —— property 必须等于**本类 forward 真正生效的状态**，否则
        `if self.use_checkpoint` 与属性读数就会分叉。要看全部五类的逐项状态
        用 `grad_checkpointing_kinds()`；eval 下 `grad_checkpointing_for` 的
        training/no_grad 门不影响这里（读的是**开关状态**，不是"此刻是否生效"）。
        """
        return (bool(getattr(self, '_gc_enabled', False))
                and bool(self.grad_checkpointing_kinds().get(GC_LEGACY, False)))

    @use_checkpoint.setter
    def use_checkpoint(self, value):
        self._init_grad_checkpointing(enabled=bool(value), **{GC_LEGACY: True})


# ============================================================================
# 可复用的 pre-norm Transformer 块（P4.1 引入）
#
# 本节的 `MHSA` / `TransformerBlock` **当前没有现役调用方** —— 随 v21 一起退役的
# `MambaLTI` / `CrossAttnRes` 是同节里另两个块类，而 `AttentionResBlock` 走的是
# 父类 `MultiHeadSelfAttention`、不是 `MHSA`。这两个类与具体主干无关，是通用件，
# 保留给后续架构复用。
# 它们**不进** `SharedBackbone` 的块栈 ⇒ 不参与任何现役建网，也不影响任何
# 现存 checkpoint。
# ============================================================================


class MHSA(MultiHeadSelfAttention):
    """拆成四个独立无 bias Linear（Wq/Wk/Wv/Wo）的多头自注意力核心。

    `TransformerBlock` 用它做注意力的那半条恒等捷径。四路参数与父类的融合
    `qkv`（`nn.Linear(C, 3C, bias=False)`）**数值上相等**，但 state_dict 布局
    不同（无法逐权从融合 qkv 的旧模型迁移）。

    为什么**继承** MultiHeadSelfAttention 而不是新写一个注意力类
    ------------------------------------------------------------
    1) 参数预算：父类除融合 qkv 外还自带 `ln1/ln2/ffn/ffn_drop`，直接复用会
       多出一整套本块用不到的 FFN。故不能直接复用父类的 forward。
    2) **注意力 dropout 的 eval 闸门（P2.2b）**：`_sdpa(..., dropout_p=...)` 是
       文件级静态锁 `tests/test_attn_dropout_eval.py::test_all_five_attn_dropout_sites_are_gated`
       的对象，它对本文件做**文件级** AST 扫描并断言取 `self.attn_drop_p` 的
       `_sdpa` 站点**恰好 4 处**。在 backbone.py 里新增第 5 处 `_sdpa` 站点会
       让那条既有测试变红。本类因此**不自己调 `_sdpa`**，而是复用父类
       `_global_attn()`（闸门站点之一，`backbone.py` 内唯一的 global 注意力入口）。
       这样既拿到 4 个独立无 bias Linear，又不多造一处未经闸门覆盖的站点。

    因此 `__init__` 刻意**不**调用 `MultiHeadSelfAttention.__init__`（那会建出融合
    qkv 与 FFN），只手工填 `_global_attn` / `_to_heads` / `attn_drop_p` 真正读到的
    最小属性集。本类**只支持 global 模式**（window/sparse 等模式的参数与形状
    不在本类的承担范围内）。
    """

    def __init__(self, channels, num_heads=4, dropout=0.0):
        nn.Module.__init__(self)
        assert channels % num_heads == 0, "channels 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = dropout
        # 父类 forward 的 window/sparse 分支会读 mode/window_size；本类只用 global。
        # 这两个属性是**惰性占位**，不是安全网：父类 forward 还会读
        # `self.ln1/ln2/ffn/ffn_drop`，本类**刻意不建**这些，
        # 所以误用父类 forward 照样 AttributeError。填它们只是为了让
        # `getattr(m, 'mode', None)` 这类查询拿到合法值，不至于在别处炸出
        # 「缺属性」而不是「用错类」这种更难排查的错。
        self.mode = "global"
        self.window_size = 7

        self.wq = nn.Linear(channels, channels, bias=False)
        self.wk = nn.Linear(channels, channels, bias=False)
        self.wv = nn.Linear(channels, channels, bias=False)
        self.wo = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        # x: (B, C, H, W) -> 内部 (B, N=H*W, C)，**不含**残差（残差归块所有）
        B, C, H, W = x.shape
        N = H * W
        seq = x.flatten(2).transpose(1, 2)                    # (B, N, C)
        q = self._to_heads(self.wq(seq), B, N)                # (B, Hh, N, d)
        k = self._to_heads(self.wk(seq), B, N)
        v = self._to_heads(self.wv(seq), B, N)
        # 复用既有闸门站点：_global_attn 内部是 _sdpa(..., dropout_p=self.attn_drop_p,
        # scale=self.scale)。不要对 q 预乘 scale（math 路径会内部再乘一次）。
        out = self._global_attn(q, k, v)                       # (B, Hh, N, d)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.wo(out).transpose(1, 2).reshape(B, C, H, W)
        return out


class TransformerBlock(nn.Module):
    """pre-norm Transformer 块：MHSA + FFN，两条恒等捷径（残差）：
        x = x + MHSA(LN1(x))
        x = x + FFN(LN2(x))

    默认形状沿用引入时的取值（184 通道 / FFN 中间维 240，即 ratio ≈ 1.304）；
    宽度与中间维都是构造参数，换主干时显式传。
    注意力用**四个独立无 bias Linear**（Wq/Wk/Wv/Wo），不是融合 qkv。
    注意力 dropout 走 MHSA 内部的既有闸门 `attn_drop_p`（P2.2b）。
    """

    def __init__(self, channels=184, num_heads=4, ffn_hidden=240,
                 attn_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.ffn_hidden = ffn_hidden
        self.norm1 = LayerNorm2d(channels)                   # 368
        self.attn = MHSA(channels, num_heads=num_heads, dropout=attn_dropout)  # 135,424
        self.norm2 = LayerNorm2d(channels)                   # 368
        self.fc1 = nn.Linear(channels, ffn_hidden, bias=False)   # 44,160
        self.fc2 = nn.Linear(ffn_hidden, channels, bias=False)   # 44,160

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.attn(self.norm1(x))                      # 恒等捷径 1
        h = self.norm2(x).flatten(2).transpose(1, 2)          # (B, N, C)
        h = self.fc2(F.gelu(self.fc1(h)))                    # (B, N, C)
        x = x + h.transpose(1, 2).reshape(B, C, H, W)        # 恒等捷径 2
        return x


def _sdpa_math(q, k, v, dropout_p=0.0, scale=None):
    """手写注意力（`_sdpa` 的 math 分支）：分块 + 逐 chunk 梯度检查点。

    ## `dropout_p` 的来源（别把它当闸门看）

    本函数**不**做任何 training 判断，只把形参原样用掉。它之所以受管，是因为
    它的**所有**调用方传进来的都是 `self.attn_drop_p`
    （= ``self.attn_drop if self.training else 0.0``）：

    - `_sdpa` 的 math 路径（`use_math=True` / V100 / batch 超限 / 无 SDPA API）；
    - `_sdpa` 的 SDPA 运行时回退（部分 CANN / 老 torch_npu 有 API 但调用即抛
      ``RuntimeError``）。

    闸门在**调用方**那一层，不在这里。所以 eval 下这里的 `dropout_p` 恒为 0.0，
    行为证据见 `tests/test_attn_dropout_eval.py::
    test_sdpa_runtime_fallback_respects_eval`（把 SDPA 打桩成必抛，验证 eval
    下输出逐位可复现、train 下确实在丢）。
    """

    step = _attn_query_chunk
    nq = q.shape[-2]

    # online-softmax 分支（flash 风格）：仅 `dropout_p == 0` 时启用；训练态带
    # dropout 回退下方 materialize 实现以保证 dropout mask 行为。不物化 (Nq,Nk)
    # 注意力矩阵 ⇒ 反向不保留该矩阵（省显存），且 online rescaling 在 fp16 下更
    # 不易溢出。**数值等价但非逐位相同**（实测 fp32 `max|Δ| ≈ 7e-07`），
    # 所以默认关闭、由 `GOAI_ATTN_ONLINE=1` 显式打开；见 `_attn_online` 的理由
    # 与 `tests/test_online_softmax_attn.py`（前向等价 + 梯度 gradcheck + bf16）。
    use_online = _attn_online and dropout_p == 0.0
    if use_online:
        if step and nq > step:
            outs = [_OnlineSoftmaxAttn.apply(
                q[..., i:i + step, :], k, v, scale) for i in range(0, nq, step)]
            return torch.cat(outs, dim=-2)
        return _OnlineSoftmaxAttn.apply(q, k, v, scale)

    # 手写注意力：math 路径需手动缩放 q
    if scale is not None:
        q = q * scale
    # query 分块（2026-10-01）：**峰值**显存由「同时活着的最大张量」决定，
    # 不是总量。整条 (B,Hh,N,N) 分数矩阵在 N=361、4 head、fp16 下是
    #   1000×4×361×361×2B = 0.97 GiB/份，softmax+dropout 再各留一份 ⇒
    # 一次调用峰值约 2.9 GiB、反向要重取约 2 份。
    # 而 softmax 沿 **key 轴**（dim=-1），每行 query 只跟自己那 N 个 key
    # 有关 ⇒ 按 query 切块在数学上**精确**，峰值变成 ∝ chunk 而不是 ∝ N。
    # chunk=64 时每份 0.97 → 0.17 GiB（5.6×）。
    #
    # 唯一的**行为**变化：`dropout_p > 0`（训练态）时 mask 的随机取样位置
    # 会变（分布等价、**不逐位相同**）。eval 态 dropout 恒为 0（见
    # `attn_drop_p`），故评估指标不受影响。
    if step and nq > step:
        kt = k.transpose(-2, -1)
        outs = []
        # **逐 chunk 梯度检查点**（2026-10-04 新增，为了真正省显存）。
        #   分块只降**瞬时**峰值，**不降保留量** —— 每块的 softmax 输出
        #   `(B,Hh,chunk,N)` 都被 autograd 存下来等反向，6 块加起来与
        #   整条 `(B,Hh,N,N)` **一样多**。实测 V7（22 层注意力、N=361、4 头）
        #   每样本每层 1.043 MB ⇒ B=3000 时约 68.8 GB。
        #
        #   为什么不能只靠 block 级 `torch.utils.checkpoint`：那依赖后端
        #   autograd 的支持程度，而实测云端 910A 上 64 GB ≈ **无** checkpoint
        #   的估算值（本地 CPU 上 checkpoint 是有效的：278.5 → 7.0 MB/样本，
        #   40×）。逐 chunk 检查点把占大头的注意力矩阵从「保留」变成
        #   「反向时一块一块重算」，**不依赖 block 级那层是否生效**。
        #
        # 行为变化只有一处：`dropout_p > 0` 时 mask 的取样位置会变
        #     （分布等价、不逐位相同）。eval 态 dropout 恒 0，指标不受影响。
        #     V7 的 `attn_dropout` 默认 0.0 ⇒ 这条对 V7 不适用。
        use_ckpt = _attn_chunk_checkpoint and torch.is_grad_enabled() \
            and q.requires_grad
        for i in range(0, nq, step):
            if use_ckpt:
                outs.append(torch.utils.checkpoint.checkpoint(
                    _attn_chunk_fn, q[..., i:i + step, :], kt, v,
                    dropout_p, use_reentrant=False))
            else:
                a = (q[..., i:i + step, :] @ kt).softmax(dim=-1)
                if dropout_p > 0.0:
                    a = torch.nn.functional.dropout(a, p=dropout_p)
                outs.append(a @ v)
        return torch.cat(outs, dim=-2)
    attn = (q @ k.transpose(-2, -1))
    attn = attn.softmax(dim=-1)
    if dropout_p > 0.0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p)
    return attn @ v
