import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """纯卷积残差块（保持原有结构，用于浅层局部特征提取）。"""

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


# flash-attn 内核的 batch 维参与 CUDA grid 坐标，受 grid y/z 维上限 65535 约束。
# window/sparse 注意力把 batch 展开为 B*G（如 512*128=65536，恰好超限 1），会报
# "CUDA error: invalid configuration argument"。取保守阈值 32768：超过则强制回退
# 手写 math——这些分块调用的 seq 仅 ~50（49 窗口 + 全局 token），math 成本可忽略。
_FLASH_BATCH_LIMIT = 32768


def _sdpa(q, k, v, dropout_p=0.0, use_math=False):
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
        # 手写注意力（q 已预乘 scale）
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        if dropout_p > 0.0:
            attn = torch.nn.functional.dropout(attn, p=dropout_p)
        return attn @ v
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


# flash-attn 独立库的内核句柄（None=未启用）。由 train_sft.py 启动时按
# 「库已安装 + Ampere+ CUDA」条件调用 set_flash_attn(True) 加载。
_flash_attn_func = None


def set_flash_attn(enabled: bool):
    """尝试加载 flash-attn 独立库内核。返回 (是否启用, 状态描述)。

    enabled=False 直接卸载回退 SDPA；enabled=True 时 import flash_attn，
    成功则 _sdpa 优先走 flash-attn 内核，失败（未安装/导入错误）返回原因并回退。
    注意：window/sparse 注意力不走 flash（形状为每窗口 1×S 微序列，见
    _window_sdpa 的 docstring），flash 仅用于 global/axial 等标准 MHA 形状。
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


def _window_sdpa(qw, kw, vw, dropout_p=0.0):
    """window/sparse 注意力的 batch 展开调用（math 一次成型）。

    qw: (M, Hh, 1, d)；kw/vw: (M, Hh, S, d)，M = B*N 可达数十万。

    为什么不用 FlashAttention（2026-09 A100 profiler 实测定案）：
      本注意力的形状是每窗口 1 query × S(=25) keys 的微序列——flash 内核按
      128-token tiling 设计，25-token 序列浪费 ~80% 算力，且 varlen backward
      实测为 forward 的 3.6 倍（11s vs 3s/50steps，占训练 CUDA 时间 39%）。
      正确工具是 bmm+softmax：attn 矩阵仅 (M, Hh, 1, S)≈55MB/层，完全可物化，
      3 个大 kernel 一次成型——零 slice/cat/clone（旧分块的 slice_backward +
      copy_ + fill_ 零化家族占 CUDA 时间 ~36%，一并消失）。
      公式与旧 _sdpa math 路径逐位一致（q@kᵀ 无 scale），训练动力学零变化。

    返回 (M, Hh, 1, d)。
    """
    attn = (qw @ kw.transpose(-2, -1)).softmax(dim=-1)   # (M, Hh, 1, S)
    if dropout_p > 0.0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p)
    return attn @ vw                                     # (M, Hh, 1, d)


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
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.mode = mode
        self.window_size = window_size

        self.ln1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.attn_drop = dropout

        self.ln2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def _to_heads(self, t, B, N):
        # t: (B, N, C) -> (B, Hh, N, head_dim)
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _global_attn(self, q, k, v):
        return _sdpa(q, k, v, dropout_p=self.attn_drop)

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
        tv = tv.reshape(B, Hh, d, H, W, ws, ws) \
               .permute(0, 3, 4, 1, 5, 6, 2)                # (B,H,W,Hh,kh,kw,d)
        return tv.reshape(B, N, Hh, ws * ws, d)             # 满带宽拷贝

    def _window_attn(self, q, k, v, H, W):
        """滑动窗口注意力：对每个位置仅与其 (ws×ws) 2D 局部窗口内 token 交互。

        取每个窗口的 ws*ws 个 token，对窗口中心槽做注意力后输出。复杂度 O(N * ws²)。

        性能设计（2026-09 重写定稿）：
          - 单一路径：bmm+softmax 一次成型（_window_sdpa），零分块/零 slice/零
            cat。attn 矩阵仅 (B*N, Hh, 1, ws²)（中心 1 query），~55MB/层可物化，
            不需要 FlashAttention（1×25 微序列形状对 flash tiling 极不友好，
            A100 profiler 实测 varlen backward 为 forward 的 3.6 倍）。
          - 窗口内容为真 2D 局部窗口（语义修正见 _local_windows）；中心槽按纠缠
            语义取自窗口张量（与旧实现同位）。
        q,k,v: (B, Hh, N, head_dim)，N = H*W。
        """
        ws = self.window_size
        B, Hh, N, d = q.shape
        center = (ws * ws) // 2

        # 取窗口邻居: (B, N, Hh, ws², d)——真 2D 局部窗口
        qu = self._local_windows(q, H, W)
        ku = self._local_windows(k, H, W)
        vu = self._local_windows(v, H, W)

        # 窗口中心槽（纠缠语义，与旧实现同位）作为 query；k/v 零拷贝 view
        qc = qu[:, :, :, center, :].reshape(B * N, Hh, 1, d)
        kw = ku.reshape(B * N, Hh, ws * ws, d)
        vw = vu.reshape(B * N, Hh, ws * ws, d)
        oc = _window_sdpa(qc, kw, vw, dropout_p=self.attn_drop)  # (B*N, Hh, 1, d)
        return oc.view(B, N, Hh, d).reshape(B, N, Hh * d)

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
          - 不再分块：query 只有中心 1 个 token，注意力矩阵
            (B*N, Hh, 1, ws²+ng) 极小，单次成型即可。

        q,k,v: (B, Hh, N, head_dim)，N = H*W。返回 (B, N, Hh*d)。
        """
        ws = self.window_size
        stride = ws  # 全局 token 每隔 stride 取一个（块中心）
        B, Hh, N, d = q.shape

        # ---- 1) k/v：真 2D 局部窗口，单次满带宽拷贝（无 im2col、无 slice 节点）----
        kw = self._local_windows(k, H, W).reshape(B * N, Hh, ws * ws, d)
        vw = self._local_windows(v, H, W).reshape(B * N, Hh, ws * ws, d)

        # ---- 2) q 即窗口中心（= 自身位置）：纯 head 主序重排，一次小拷贝 ----
        # 注意不可经纠缠 reshape(B,Hh*d,H,W) 取中心——纠缠空间的中心槽并非自身 token。
        qc = q.permute(0, 2, 1, 3).reshape(B * N, Hh, 1, d)   # (B*N, Hh, 1, d) 小拷贝

        # ---- 4) 全局代表 token：块中心（pattern 与旧实现一致）----
        gh, gw = H // stride, W // stride
        ng = gh * gw
        kg = k.reshape(B, Hh, H, W, d)[:, :, :gh * stride, :gw * stride, :]
        kg = kg.view(B, Hh, gh, stride, gw, stride, d)[:, :, :, stride // 2, :, stride // 2, :]
        kg = kg.reshape(B, Hh, ng, d)                       # (B,Hh,ng,d)
        vg = v.reshape(B, Hh, H, W, d)[:, :, :gh * stride, :gw * stride, :]
        vg = vg.view(B, Hh, gh, stride, gw, stride, d)[:, :, :, stride // 2, :, stride // 2, :]
        vg = vg.reshape(B, Hh, ng, d)

        # ---- 5) 广播到每个 token：expand 纯 view + 一次 reshape 拷贝 ----
        k_glob = kg.unsqueeze(1).expand(B, N, Hh, ng, d).reshape(B * N, Hh, ng, d)
        v_glob = vg.unsqueeze(1).expand(B, N, Hh, ng, d).reshape(B * N, Hh, ng, d)

        # ---- 6) 拼接 + 单次成型注意力（A100 走 varlen flash，V100 math 一次成型）----
        k_all = torch.cat([kw, k_glob], dim=2)              # (B*N, Hh, ws²+ng, d)
        v_all = torch.cat([vw, v_glob], dim=2)
        oc = _window_sdpa(qc, k_all, v_all, dropout_p=self.attn_drop)  # (B*N, Hh, 1, d)

        # ---- 7) 回到 (B, N, Hh*d)：head 主序展平，纯 view 无拷贝 ----
        return oc.view(B, N, Hh, d).reshape(B, N, Hh * d)

    def _axial_attn(self, q, k, v, H, W):
        """轴向注意力：先按行、再按列做 1D 自注意力。

        q,k,v: (B, Hh, N, d)，N=H*W。轴向注意力把二维 token 在单轴上交互，
        复杂度约 O(2·N·max(H,W))，远低于 O(N²)，同时保留长程（整行/整列）依赖。
        """
        B, Hh, N, d = q.shape

        def attn_1d(tokens):
            # tokens: (B*Hh*L, S, d) -> 把 Hh 融进 batch 做标准 MHA
            t = tokens.view(-1, Hh, tokens.shape[1], d)
            return _sdpa(t, t, t, dropout_p=self.attn_drop).view(-1, tokens.shape[1], d)

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
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._to_heads(q, B, N)
        k = self._to_heads(k, B, N)
        v = self._to_heads(v, B, N)
        # 预乘 scale 到 q，便于 _sdpa / 手写统一
        q = q * self.scale

        if self.mode == "window":
            out = _run_with_optional_disable(self._window_attn, q, k, v, H, W)
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
                 attn_mode="global", attn_window=7):
        """
        Args:
            attention_mode:   主干堆叠模式 "none"|"mix"|"all"
            num_attention_layers: mix 模式下注意力块数量
            num_heads:         多头注意力头数
            attention_dropout: 注意力 dropout
            attn_mode:         注意力计算模式 "global"|"window"|"axial"
            attn_window:       window 模式的窗口边长
        """
        super(SharedBackbone, self).__init__()
        self.channels = channels
        self.attention_mode = attention_mode

        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        blocks = self._build_blocks(
            num_res_blocks, attention_mode, num_attention_layers,
            channels, num_heads, attention_dropout, attn_mode, attn_window)
        self.blocks = nn.Sequential(*blocks)

        self.conv_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)

    @staticmethod
    def _build_blocks(num_res_blocks, mode, num_attn, channels, num_heads,
                      dropout, attn_mode, attn_window):
        if mode == "none" or num_attn <= 0:
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
                blocks.append(ResBlock(channels))
        return blocks

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = F.relu(self.bn_out(self.conv_out(out)))
        return out
