import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm（兼容 PyTorch 2.1，不依赖 nn.RMSNorm）。"""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x):
        # 保持原有精度，不强制转 FP32（BF16 下减少转换开销）
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


# flash-attn 内核的 batch 维参与 CUDA grid 坐标，受 grid y/z 维上限 65535 约束。
# window/sparse 注意力把 batch 展开为 B*G（如 512*128=65536，恰好超限 1），会报
# "CUDA error: invalid configuration argument"。阈值需低于 flash 内核 grid 上限 65535：
# 取 60000，使 ws=7、B=4800 时 B*nW=43200 可走 flash；超过则强制回退
# 手写 math——这些分块调用的 seq 仅 ~50（49 窗口 + 全局 token），math 成本可忽略。
_FLASH_BATCH_LIMIT = 60000  # ws=7, B=4800 -> B*nW=43200 < 60000


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
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        # flash-attn 独立库：要求 (B, S, Hh, d) 布局（头维 -2、head_dim -1）。
        # 我们的 (B, Hh, N, d) 中 head_dim 为最内层（stride=1），transpose 后满足
        # flash-attn 的 last-dim contiguous 要求，无需显式 .contiguous() 拷贝。
        out = _flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=dropout_p, causal=False)
        return out.transpose(1, 2)
    if use_math or not hasattr(F, "scaled_dot_product_attention"):
        # 手写注意力：math 路径需手动缩放 q
        if scale is not None:
            q = q * scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        if dropout_p > 0.0:
            attn = torch.nn.functional.dropout(attn, p=dropout_p)
        return attn @ v
    # SDPA 路径：内部自带 1/sqrt(d) 缩放，q 不能预乘 scale，否则双重缩放
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


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
        ⚠ 语义与旧 checkpoint 不兼容（旧权重在 scramble 语义下训练，需重训/重评估）。
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
          - 语义变更：滑动 → 块状。⚠ 与滑动窗口 checkpoint 不兼容，需重训。

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
            ⚠ 与旧 checkpoint 不兼容（旧权重在 scramble 语义下训练，需重训/重评估）。
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


class SharedBackbone(nn.Module):
    """共享表示网络：将棋盘状态编码为隐藏状态。

    注意力模式（attention_mode 控制主干如何堆叠注意力块）：
        - "none" : 全部用纯卷积 ResBlock（最快，局部性最好）
        - "mix"  : 在 num_res_blocks 个块中穿插 num_attention_layers 个
                   AttentionResBlock（推荐：卷积打底 + 注意力提质）
        - "all"  : 全部使用 AttentionResBlock

    注意力块内部的计算模式由 attn_mode 控制（全局/窗口/轴向），
    通过 --attn-mode 配置；窗口大小由 --attn-window 控制。
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
            res_blocks:        ResBlock 数量（浅层，局部细节），0 表示使用默认模式
            convnext_blocks:   ConvNeXtBlock 数量（中层，大感受野），0 表示使用默认模式
            attn_blocks:       AttentionResBlock 数量（深层，全局关系），0 表示使用默认模式
        """
        super(SharedBackbone, self).__init__()
        self.channels = channels
        self.attention_mode = attention_mode
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
                channels, num_heads, attention_dropout, attn_mode, attn_window)
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
                else:
                    blocks.append(ResBlock(channels))
        return blocks

    @staticmethod
    def _build_segmented_blocks(total_blocks, res_count, convnext_count, attn_count,
                                 channels, num_heads, dropout, attn_mode, attn_window):
        """三段式架构：浅层 ResNet + 中层 ConvNeXt + 深层 Attention"""
        blocks = []

        # 浅层：ResBlock (局部细节)
        for _ in range(res_count):
            blocks.append(ResBlock(channels))

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
            out = torch.utils.checkpoint.checkpoint_sequential(
                self.blocks, len(self.blocks), out, use_reentrant=False)
        else:
            out = self.blocks(out)
        if self.arch == "convnext":
            out = self.norm_out(self.conv_out(out))
        else:
            out = F.relu(self.bn_out(self.conv_out(out)))
        return out


# ============================================================================
# v21 新块类（P4.1）
#
# 这三个类 + 两个头（P4.1 落 policy_network.py / value_network.py）只**新增**，
# 既有 resnet / convnext 路径与既有类签名一字未改（D5）：v21 走自己的构建类
# （P4.2 的 V21_CFG），不经过 SharedBackbone / AttentionResBlock。
#
# 权威结构 = 用户给定的逐层参数表（P4.1 brief §2）。本节所有类的参数量都被
# tests/test_arch_v21_blocks.py 逐类**精确相等**锁住（113,528 / 224,480 /
# 326,416），不是窗口 —— 全网预算窗口留给 P4.2 的 test_v21_budget。
# ============================================================================


class MHSA(MultiHeadSelfAttention):
    """v21 的多头自注意力核心（Wq/Wk/Wv/Wo **四个独立无 bias** Linear）。

    `TransformerBlock` 与 `CrossAttnRes` 共用本类，两处预算都是
    3×184×184（Wq/Wk/Wv）+ 184×184（Wo）= 135,424。

    为什么**继承** MultiHeadSelfAttention 而不是新写一个注意力类
    ------------------------------------------------------------
    1) 预算：父类是**融合** qkv（`nn.Linear(C, 3C, bias=False)`），参数量数值上
       恰好等于 3 个独立 Linear，但 state_dict 布局不同（无法逐权从旧模型迁移），
       而且父类还自带 `ln1/ln2/ffn/ffn_drop`（368+368+44,160×2+128 = 89,184 个
       多余参数），整体会超出 v21 的分项预算。故不能直接复用父类的 forward。
    2) **注意力 dropout 的 eval 闸门（P2.2b）**：`_sdpa(..., dropout_p=...)` 是
       文件级静态锁 `tests/test_attn_dropout_eval.py::test_all_five_attn_dropout_sites_are_gated`
       的对象，它对本文件做**文件级** AST 扫描并断言取 `self.attn_drop_p` 的
       `_sdpa` 站点**恰好 4 处**。在 backbone.py 里新增第 5 处 `_sdpa` 站点会
       让那条既有测试变红。本类因此**不自己调 `_sdpa`**，而是复用父类
       `_global_attn()`（闸门站点之一，`backbone.py` 内唯一的 global 注意力入口）。
       这样既拿到 4 个独立无 bias Linear，又不多造一处未经闸门覆盖的站点。

    因此 `__init__` 刻意**不**调用 `MultiHeadSelfAttention.__init__`（那会建出融合
    qkv 与 FFN），只手工填 `_global_attn` / `_to_heads` / `attn_drop_p` 真正读到的
    最小属性集。本类**只支持 global 模式**（v21 的块布局里 MHSA 就是全局注意力；
    window/sparse 等模式的参数与形状不属本预算）。
    """

    def __init__(self, channels, num_heads=4, dropout=0.0):
        nn.Module.__init__(self)
        assert channels % num_heads == 0, "channels 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = dropout
        # 父类 forward 的 window/sparse 分支会读 mode/window_size；v21 只用 global，
        # 这里填一个合法值只为避免误用父类 forward 时抛 AttributeError。
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


class MambaLTI(nn.Module):
    """Mamba 风格的**线性时序（LTI）**块：dt 步长的因果累积递推（C=184, expand=2,
    d_conv=4, d_state N=16, ddt rank=4），逐块 **113,528** 参数。

    语义（x: (B, C, H, W) -> 同形；内部按 (B, T=H*W, ·) 的**行主序**序列看）：

        h  = LayerNorm(x)                                    # pre-LN（唯一一层）
        [xs, z] = split(in_proj(h), 2)                       # in_proj 184→368，各 184
        xc     = SiLU(causal_dwconv1d_k4(xs))                # 仅 x 路，深度卷积
        [dt_raw, bc] = split(x_proj(xc), rank, 2N)           # 4 + 32 = 36
        dt     = softplus(dt_proj(dt_raw))                   # (B,T,184)，逐通道 > 0
        [Bm, Cm] = split(bc, N, N)                           # 各 16
        y_t     = ( Σ_{s≤t} exp( A ⊙ (Σ_{r=s+1..t} dt_r) ) · dt_s·xc_s ⊙ Bm_s ) · Cm_t
        y       = y + D * xc                                 # D：逐通道直通
        out     = out_proj(y) * SiLU(z)                      # z 走 SiLU 门
        返回    x + out                                      # pre-norm 残差

    ⚠ `expand=2` 是 **in_proj 的扇出**（C→2C 拆 x/z），不是分支内宽度扩张 ——
    见 `__init__` 里的注释（关系到 113,528 这个数）。

    LTI 递推（状态更新 + 输出投影）的闭式，见
    `.superpowers/sdd/2026-09-25-v21-roadmap/task-p4-1-report.md` §3；
    `tests/test_arch_v21_blocks.py` 用测试内独立写的双重 for 循环朴素参考逐步对齐。

    因果性：递推只用 s ≤ t 的量，深度卷积**左**填充 k-1=3，三处都没有未来信息。
    `test_ssm_causality` 直接钉这条（改 t 之后的输入不得影响 t 的输出）。

    实现选择（无参数开销、但影响梯度流，已在 report 里列为待用户确认项）：
      * 递推按 T 步**顺序**扫描（内存 O(B·C·S)，19×19 → 361 步）。闭式的
        (T,T) 衰减矩阵需要 B·T²·C·S 个数（184 通道 × 16 状态时 T=361 要 30e9），
        不可物化；若将来要提速，方向是分块扫描 / 关联扫描（不是换语义）。
      * `A` 以 log 形式 `A_log` 存储（`A = -exp(A_log)`，初值 log(1..N) ⇒
        A ∈ {-1..-16}），与 Mamba 参考实现同；参数量与 `A` 直接存储完全相同
        （184×16 = 2,944），但初值落在稳定区间而不是围绕 0 抖动。
      * `z` 的 SiLU 门作用在 **z** 上（Mamba 参考实现：`out_proj(y) * silu(z)`），
        而不是作用在 y 上；见 report §3 的说明。
    """

    def __init__(self, channels=184, expand=2, d_conv=4, d_state=16,
                 ddt_rank=4):
        super().__init__()
        # ⚠ `expand` 的含义以**权威表**为准，不是 Mamba 惯例。表里给的是
        #   `in_proj 184→368`（=67,712）、`dw conv1d Conv1d(184,184,...)`、
        #   `out_proj 184→184` ⇒ 每个分支的宽度就是 C=184，in_proj 的**扇出**
        #   才是 2×C（拆两半各 184：x 路 / z 路），即 d_inner = C。
        #   若按惯例理解成「分支内扩张 d_inner = 2C = 368」，in_proj 会变成
        #   184→736（135,424），本块变 223,560 —— 比权威的 113,528 多 110,032。
        assert expand == 2, (
            '权威表只定义 expand=2（in_proj 扇出 2C，拆 x/z 各 C）；'
            '其它 expand 值未在参数表里定义，不臆造。收到 {}'.format(expand))
        self.channels = channels
        self.expand = expand
        self.d_inner = channels                  # 184（= d_model，见上）
        self.d_state = d_state                  # 16
        self.d_conv = d_conv                    # 4
        self.ddt_rank = ddt_rank                # 4
        self.dt_project_rank = ddt_rank + 2 * d_state   # 36 = x_proj 输出维

        self.norm = LayerNorm2d(channels)                                   # 368
        self.in_proj = nn.Linear(channels, expand * channels, bias=False)  # 67,712
        # 深度因果卷积，**仅**作用于 x 路
        self.dw_conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                 groups=self.d_inner, bias=True)            # 920
        self.x_proj = nn.Linear(self.d_inner, self.dt_project_rank, bias=False)  # 6,624
        self.dt_proj = nn.Linear(ddt_rank, self.d_inner, bias=True)         # 920
        # A：逐通道 184×16 的状态转移率矩阵（log 形式存储，见类 docstring）
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).log()
        self.A_log = nn.Parameter(a_init.repeat(channels, 1))               # 2,944
        self.D = nn.Parameter(torch.ones(channels))                         # 184
        self.out_proj = nn.Linear(self.d_inner, channels, bias=False)       # 33,856

    def _selective_scan(self, decay, drive, cvec):
        """顺序扫描的 dt 步长因果累积递推。

        decay: (B, T, d_inner, N) = exp(dt_t ⊙ A)，元素 ∈ (0, 1]
        drive: (B, T, d_inner, N) = dt_t · xc_t ⊙ B_t
        cvec:  (B, T, N)         = C_t

        返回 (B, T, d_inner)。T 步的 Python 循环换来 O(B·d_inner·N) 的显存；
        换成闭式 (T,T) 衰减矩阵会物化 T²·C·N 个元素，19×19 下不可接受。
        """
        B, T, d_inner, n = decay.shape
        state = decay.new_zeros(B, d_inner, n)
        out = []
        for t in range(T):
            state = state * decay[:, t] + drive[:, t]
            out.append((state * cvec[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(out, dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        T = H * W

        h = self.norm(x).flatten(2).transpose(1, 2)            # (B, T, C)
        hs, z = self.in_proj(h).split(self.d_inner, dim=-1)  # 各 (B, T, 184)

        # 深度因果卷积：左填充 k-1，保证 out[t] 只看 xs[t-3..t]
        xs = F.pad(hs.transpose(1, 2), (self.d_conv - 1, 0))
        xc = F.silu(self.dw_conv(xs).transpose(1, 2))        # (B, T, d_inner)

        # x_proj 36 维拆成 ddt_rank(=4) + 2N(=32)：注意必须按「前 4 / 后 32」
        # 切，不能用 split(4)（那会切成 9 块）。
        proj_db = self.x_proj(xc)
        dt_raw, bc = proj_db[..., :self.ddt_rank], proj_db[..., self.ddt_rank:]
        dt = F.softplus(self.dt_proj(dt_raw))                # (B, T, d_inner) > 0
        b_vec, c_vec = bc.split(self.d_state, dim=-1)        # 各 (B, T, 16)

        A = -torch.exp(self.A_log)                           # (d_inner, 16) < 0
        decay = torch.exp(dt.unsqueeze(-1) * A)              # (B,T,d_inner,16)
        drive = dt.unsqueeze(-1) * xc.unsqueeze(-1) * b_vec.unsqueeze(2)
        y = self._selective_scan(decay, drive, c_vec)        # (B, T, d_inner)
        y = y + self.D * xc                                 # D 逐通道直通

        out = self.out_proj(y) * F.silu(z)                   # z 走 SiLU 门
        return (x + out.transpose(1, 2).reshape(B, C, H, W))  # pre-norm 残差


class TransformerBlock(nn.Module):
    """pre-norm Transformer 块：MHSA + FFN(184→240→184)，逐块 **224,480** 参数。

    两条恒等捷径（残差）：
        x = x + MHSA(LN1(x))
        x = x + FFN(LN2(x))

    FFN 中间维 **240**（ratio 240/184 ≈ 1.304；旧表的 276 / 1.5 已作废）。
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


class CrossAttnRes(nn.Module):
    """跨层注意力残差块：拼接主干第 1/5/9 号块的输出后投影回来，逐块
    **326,416** 参数。

        taps = (s1, s5, s9)   # 主干第 1、5、9 号块（1-based）的输出
        x = x + proj(concat(LN1(s1), LN1(s5), LN1(s9)))      # 跨层投影支路
        x = x + MHSA(LN2(x))                                  # 恒等捷径
        x = x + FFN(LN3(x))                                   # 恒等捷径

    三路的分工（哪一路是 identity 捷径）
    ------------------------------------
    **第一路 `s1`（主干第 1 号块，紧邻 stem 的浅层抽头）是 identity 捷径**：
    它表征最浅、离输入最近，投影支路主要靠它把局部/原始特征直通过来；第二路 `s5`
    与第三路 `s9` 提供中/深层多尺度语义。若用户本意是 `s9` 才算 identity 捷径，
    只需改本类 docstring 与 P4.2 的抽头顺序 —— 参数量与计算图完全不受影响。
    （顶层接线归 P4.2：本类只消费抽头，不自己去找主干。）

    为什么 LN1 是 **184 维**、且三路共用一个
    --------------------------------------
    权威表给「pre-LN ×3：1,104」= 3 × 368，即**三个** 184 通道的 LayerNorm。
    若 LN1 直接作用在 concat 后的 552 通道上，它自己就是 1,104（552×2），
    加上 LN2/LN3 的 736 → 1,840，本块会变成 327,152（比权威的 326,416 多 736）。
    既要满足 326,416、又要保住 `proj(LN(concat(...)))` 的「先 norm 再 proj」次序，
    唯一解是：**一个 184 维 LN 逐路作用后 concat**（三路共用同一套仿射参数）。
    这是本任务里唯一需要我自己钉死的实现细节（brief §4「需要你自己钉死」），
    已在 report §4 单列，供用户确认是否改用 `LN1(proj(concat))`（同参数、
    换归一化位置与统计口径）。

    明确**没有** BatchNorm 也没有 ReLU：权威表的 326,416 不含 BN 的 368。
    """

    def __init__(self, channels=184, tap_channels=(184, 184, 184),
                 num_heads=4, ffn_hidden=240, attn_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.tap_channels = tuple(int(c) for c in tap_channels)
        if len(self.tap_channels) != 3:
            raise ValueError(
                'CrossAttnRes 需要恰好 3 路抽头（主干第 1/5/9 号块），'
                '收到 {} 路'.format(len(self.tap_channels)))
        self.ffn_hidden = ffn_hidden
        self.norm_tap = LayerNorm2d(channels)                 # 368（逐路复用）
        self.proj = nn.Conv2d(sum(self.tap_channels), channels, 1, bias=False)  # 101,568
        self.norm_attn = LayerNorm2d(channels)               # 368
        self.attn = MHSA(channels, num_heads=num_heads, dropout=attn_dropout)  # 135,424
        self.norm_ffn = LayerNorm2d(channels)                # 368
        self.fc1 = nn.Linear(channels, ffn_hidden, bias=False)   # 44,160
        self.fc2 = nn.Linear(ffn_hidden, channels, bias=False)   # 44,160

    def _check_taps(self, taps):
        if len(taps) != 3:
            raise ValueError(
                'CrossAttnRes.forward 需要 3 路抽头（主干第 1/5/9 号块的输出），'
                '收到 {} 路'.format(len(taps)))
        for i, (t, want) in enumerate(zip(taps, self.tap_channels)):
            if t.dim() != 4:
                raise ValueError(
                    '第 {} 路抽头应为 (B, C, H, W)，收到形状 {}'.format(i + 1, tuple(t.shape)))
            if t.shape[1] != want:
                raise ValueError(
                    '第 {} 路抽头通道数应为 {}（构造时给的 tap_channels[{}]），'
                    '收到 {} —— 抽头宽度必须与构造参数一致，'
                    '否则 proj 的 in_channels={} 与实际 concat 宽度对不上'
                    .format(i + 1, want, i, t.shape[1], sum(self.tap_channels)))

    def forward(self, x, taps):
        B, C, H, W = x.shape
        self._check_taps(taps)
        s1, s5, s9 = (self.norm_tap(t) for t in taps)
        merged = self.proj(torch.cat([s1, s5, s9], dim=1))  # 无 BN、无 ReLU

        x = x + merged                                       # 跨层投影支路
        x = x + self.attn(self.norm_attn(x))                 # 恒等捷径
        h = self.norm_ffn(x).flatten(2).transpose(1, 2)
        h = self.fc2(F.gelu(self.fc1(h)))
        x = x + h.transpose(1, 2).reshape(B, C, H, W)       # 恒等捷径
        return x
