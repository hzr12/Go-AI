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
    """
    # 模块级覆盖：训练脚本按 GPU 能力设置（A100 走 Flash，V100 走 math）
    if _sdpa_force_math:
        use_math = True
    if use_math or not hasattr(F, "scaled_dot_product_attention"):
        # 手写注意力（q 已预乘 scale）
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        if dropout_p > 0.0:
            attn = torch.nn.functional.dropout(attn, p=dropout_p)
        return attn @ v
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


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


def _maybe_compiler_disable(fn):
    """条件性 torch.compiler.disable 装饰器。

    仅当 _compile_disable_sparse=True（V100 等老卡）时排除编译；
    A100 上直接返回原函数，纳入编译图。
    """
    if _compile_disable_sparse:
        return torch.compiler.disable(fn)
    return fn


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

    @_maybe_compiler_disable
    def _window_attn(self, q, k, v, H, W):
        """滑动窗口注意力：对每个位置仅与窗口内 token 交互。

        用 F.unfold 取每个窗口的 ws*ws 个 token，把窗口维并入 batch 后用
        手写 softmax 注意力（use_math，绕开 V100 上不可用的 FlashAttention）计算
        局部注意力，再取中心位置输出。复杂度 O(N * ws²)。

        显存优化：unfold 后沿窗口维 N 分块（chunk），每块只对 G 个窗口做 SDPA，
        避免一次性构造 (B*N, Hh, ws², d) 大矩阵（B*N 在 19x19 上可达数万，直接 OOM）。
        分块后峰值激活只与 B*G 相关，与总 batch 解耦，可支持大 batch。

        @_maybe_compiler_disable: 仅 V100(sm_70)/torch2.1 上 F.unfold 接动态 view/permute
        链会让 inductor 触发 PolynomialError，此时排除出编译图；A100(sm_80+) 上 inductor
        成熟可正常编译 unfold，不禁用，使整个注意力被融合。use_math 由 _SDPA_FORCE_MATH
        模块开关决定（V100 走手写 softmax，A100 走 FlashAttention 后端）。
        q,k,v: (B, Hh, N, head_dim)，N = H*W。
        """
        ws = self.window_size
        B, Hh, N, d = q.shape

        # 取窗口邻居: (B, Hh*d, N, ws*ws)
        qu = F.unfold(q.reshape(B, Hh * d, H, W), kernel_size=ws, padding=ws // 2)
        ku = F.unfold(k.reshape(B, Hh * d, H, W), kernel_size=ws, padding=ws // 2)
        vu = F.unfold(v.reshape(B, Hh * d, H, W), kernel_size=ws, padding=ws // 2)

        # 重排为 (B, N, Hh, ws*ws, d) 便于沿 N 分块
        qu = qu.view(B, Hh, d, N, ws * ws).permute(0, 3, 1, 4, 2)  # (B, N, Hh, ws², d)
        ku = ku.view(B, Hh, d, N, ws * ws).permute(0, 3, 1, 4, 2)
        vu = vu.view(B, Hh, d, N, ws * ws).permute(0, 3, 1, 4, 2)

        center = (ws * ws) // 2
        out_chunks = []
        # 每块 G 个窗口，峰值激活正比于 B*G。G 取小值（默认 256）即可让大 batch 不 OOM：
        # 典型 (B=200, G=256, Hh=4, ws²=49, d=32, 前后向×2) ≈ 200*256*4*49*32*2*4B ≈ 1.3GB。
        G = max(1, getattr(self, 'window_chunk', 128))
        for s in range(0, N, G):
            e = min(s + G, N)
            qc = qu[:, s:e].reshape(B * (e - s), Hh, ws * ws, d)
            kc = ku[:, s:e].reshape(B * (e - s), Hh, ws * ws, d)
            vc = vu[:, s:e].reshape(B * (e - s), Hh, ws * ws, d)
            oc = _sdpa(qc, kc, vc, dropout_p=self.attn_drop,
                       use_math=_sdpa_force_math)  # V100 走 math；A100 走 Flash
            oc = oc[:, :, center:center + 1, :]  # (B*G, Hh, 1, d)
            out_chunks.append(oc.reshape(B, e - s, Hh, d))
        out = torch.cat(out_chunks, dim=1)  # (B, N, Hh, d)
        return out.reshape(B, N, Hh * d)

    @_maybe_compiler_disable
    def _sparse_attn(self, q, k, v, H, W):
        """稀疏注意力（固定稀疏模式）：局部滑动窗口 + 跨步长全局 token。

        每个 query 位置只与两类 key 交互：
          1) 自身 (ws×ws) 局部窗口内的 token（局部性，同 window）；
          2) 每隔 stride=ws 下采样的「全局代表 token」（长程信息通路）。
        总序列长度 = ws² + ng（ng≈(H/ws)×(W/ws)），与 window 同量级显存，
        但全局 token 之间又各自在窗口内连通，使整盘信息可经 2~3 跳传播，
        显著优于纯 window 的长程建模，且远省于全局 O(N²) 注意力。

        显存：沿窗口维 N 分块（同 _window_attn 的 window_chunk），避免 B*N 大矩阵。

        @_maybe_compiler_disable: 同 _window_attn，仅 V100 上排除编译，A100 纳入编译图。
        use_math 由 _SDPA_FORCE_MATH 模块开关决定（V100 走手写 softmax，A100 走 FlashAttention）。
        q,k,v: (B, Hh, N, head_dim)，N = H*W。
        """
        ws = self.window_size
        stride = ws  # 全局 token 每隔 stride 取一个（块中心）
        B, Hh, N, d = q.shape

        # ---- 局部窗口邻居 (B, N, Hh, ws², d) ----
        qu = F.unfold(q.reshape(B, Hh * d, H, W), kernel_size=ws, padding=ws // 2)
        ku = F.unfold(k.reshape(B, Hh * d, H, W), kernel_size=ws, padding=ws // 2)
        vu = F.unfold(v.reshape(B, Hh * d, H, W), kernel_size=ws, padding=ws // 2)
        qu = qu.view(B, Hh, d, N, ws * ws).permute(0, 3, 1, 4, 2)  # (B, N, Hh, ws², d)
        ku = ku.view(B, Hh, d, N, ws * ws).permute(0, 3, 1, 4, 2)
        vu = vu.view(B, Hh, d, N, ws * ws).permute(0, 3, 1, 4, 2)

        # ---- 全局代表 token：从 k/v 按 stride 下采样块中心 ----
        # k: (B, Hh, N, d) -> (B, Hh, H, W, d)
        kg = k.reshape(B, Hh, H, W, d)
        vg = v.reshape(B, Hh, H, W, d)
        gh, gw = H // stride, W // stride
        # 截断到 stride 的整数倍（丢弃边界余数行/列，仅影响全局 token 覆盖，
        # 不影响局部窗口对全图的覆盖）
        kg = kg[:, :, :gh * stride, :gw * stride, :]
        vg = vg[:, :, :gh * stride, :gw * stride, :]
        # 取每 stride×stride 块的中心坐标
        kg = kg.view(B, Hh, gh, stride, gw, stride, d)[:, :, :, stride // 2, :, stride // 2, :]
        vg = vg.view(B, Hh, gh, stride, gw, stride, d)[:, :, :, stride // 2, :, stride // 2, :]
        ng = gh * gw
        kg = kg.reshape(B, Hh, ng, d)  # (B, Hh, ng, d)
        vg = vg.reshape(B, Hh, ng, d)

        center = (ws * ws) // 2
        out_chunks = []
        G = max(1, getattr(self, 'window_chunk', 128))
        for s in range(0, N, G):
            e = min(s + G, N)
            qc = qu[:, s:e].reshape(B * (e - s), Hh, ws * ws, d)
            kc = ku[:, s:e].reshape(B * (e - s), Hh, ws * ws, d)
            vc = vu[:, s:e].reshape(B * (e - s), Hh, ws * ws, d)
            # query 用局部中心 token 的 q；key/value = 局部窗口 + 全局 token
            q_center = qc[:, :, center:center + 1, :]  # (B*G, Hh, 1, d)
            # 全局 token 广播到当前窗口块 (B*G, Hh, ng, d)
            k_glob = kg.unsqueeze(1).expand(B, e - s, Hh, ng, d).reshape(B * (e - s), Hh, ng, d)
            v_glob = vg.unsqueeze(1).expand(B, e - s, Hh, ng, d).reshape(B * (e - s), Hh, ng, d)
            k_all = torch.cat([kc, k_glob], dim=2)  # (B*G, Hh, ws²+ng, d)
            v_all = torch.cat([vc, v_glob], dim=2)
            oc = _sdpa(q_center, k_all, v_all, dropout_p=self.attn_drop,
                       use_math=_sdpa_force_math)  # V100 走 math；A100 走 Flash
            out_chunks.append(oc.reshape(B, e - s, Hh, d))
        out = torch.cat(out_chunks, dim=1)  # (B, N, Hh, d)
        return out.reshape(B, N, Hh * d)

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
            out = self._window_attn(q, k, v, H, W)
        elif self.mode == "axial":
            out = self._axial_attn(q, k, v, H, W)
        elif self.mode == "sparse":
            out = self._sparse_attn(q, k, v, H, W)  # (B, N, C)
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
