"""MindSpore 版 AlphaGoNet（与 src/networks/alphanet.py 结构/参数名一一对应）。

设计要点
--------
1. **参数名与 torch 版完全一致**，`scripts/convert_ckpt.py` 才能双向转换权重。
   命名映射（MS -> torch）：
     Conv2d.weight   -> Conv2d.weight        （形状相同，无需转置）
     Dense.weight    -> Linear.weight        （形状相同 (out,in)，无需转置）
     BatchNorm.gamma/beta/moving_mean/moving_variance
                     -> weight/bias/running_mean/running_var
     LayerNorm.gamma/beta -> weight/bias
2. **注意力一律走手写 math 路径**（q 预乘 scale，无额外缩放），与 torch 版
   `set_sdpa_force_math(True)` 的语义逐位一致——NPU/CANN 上 torch 版也正是走
   这条路径，因此两边数值可比。MS 无 SDPA/flash-attn，也不做 torch.compile。
3. GELU 用 **exact(erf)** 而非 tanh 近似：torch `nn.GELU()` 默认是 exact，
   MindSpore `nn.GELU()` 默认是 approximate=True，必须显式关掉，否则数值不等价。
4. 支持 4 种 attn_mode：global / window / axial / sparse。

仅在 Ascend 910B + MindSpore 环境下可用；本地无 MindSpore 时 import 会失败，
但不影响任何 torch 代码路径。
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


# --------------------------------------------------------------------------- #
# 基础块
# --------------------------------------------------------------------------- #
class ResBlock(nn.Cell):
    """纯卷积残差块（与 torch 版 ResBlock 同名同参）。"""

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, pad_mode="pad",
                               padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, pad_mode="pad",
                               padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


# --------------------------------------------------------------------------- #
# 多头自注意力
# --------------------------------------------------------------------------- #
class MultiHeadSelfAttention(nn.Cell):
    """多头自注意力，4 种模式：global / window / axial / sparse。

    与 torch 版的关键差异：
      - 无 SDPA / flash-attn / torch.compile，统一手写 math 注意力；
      - q 在进入各模式前已预乘 scale（与 torch 版 forward 一致），
        因此内部**不再**除以 sqrt(d)。
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

        self.ln1 = nn.LayerNorm([channels])
        self.qkv = nn.Dense(channels, channels * 3, has_bias=False)
        self.attn_drop = dropout

        self.ln2 = nn.LayerNorm([channels])
        self.ffn = nn.SequentialCell(
            nn.Dense(channels, channels * 2),
            nn.GELU(approximate=False),      # 对齐 torch 默认 exact GELU
            nn.Dense(channels * 2, channels),
        )
        self.ffn_drop = nn.Dropout(keep_prob=1.0 - dropout)
        # attention dropout 用模块而非 functional：GRAPH_MODE 下不能在 construct
        # 里读 self.training 做 Python 分支，交给 nn.Dropout 内部处理
        self.attn_dropout = nn.Dropout(keep_prob=1.0 - dropout)

        self.softmax = nn.Softmax(axis=-1)

    # -- 通用：math 注意力 ------------------------------------------------ #
    def _math_attn(self, q, k, v):
        """q 已预乘 scale。q/k/v: (B, Hh, N, d) -> (B, Hh, N, d)。"""
        attn = self.softmax(ops.matmul(q, k.transpose(0, 1, 3, 2)))
        if self.attn_drop > 0.0:
            attn = self.attn_dropout(attn)
        return ops.matmul(attn, v)

    def _to_heads(self, t, B, N):
        # (B, N, C) -> (B, Hh, N, head_dim)
        return t.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    # -- global ----------------------------------------------------------- #
    def _global_attn(self, q, k, v):
        return self._math_attn(q, k, v)

    # -- window（Swin 风格块状窗口）---------------------------------------- #
    def _window_attn(self, q, k, v, H, W):
        ws = self.window_size
        B, Hh, N, d = q.shape
        H2 = ((H + ws - 1) // ws) * ws
        W2 = ((W + ws - 1) // ws) * ws
        ph, pw = H2 - H, W2 - W
        nwH, nwW = H2 // ws, W2 // ws
        nW = nwH * nwW

        def part(t):
            # (B,Hh,N,d) -> (B*nW, Hh, ws², d)
            x = t.reshape(B, Hh, H, W, d)
            if ph or pw:
                # 在 H、W 两维右侧补零（d 维不动）
                x = ops.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw), (0, 0)))
            x = x.reshape(B, Hh, nwH, ws, nwW, ws, d)
            return x.transpose(0, 2, 4, 1, 3, 5, 6).reshape(B * nW, Hh, ws * ws, d)

        def unpart(o):
            # (B*nW, Hh, ws², d) -> (B, N, Hh*d)
            x = o.reshape(B, nwH, nwW, Hh, ws, ws, d)
            x = x.transpose(0, 3, 1, 4, 2, 5, 6).reshape(B, Hh, H2, W2, d)
            if ph or pw:
                x = x[:, :, :H, :W, :]
            return x.transpose(0, 2, 3, 1, 4).reshape(B, N, Hh * d)

        return unpart(self._math_attn(part(q), part(k), part(v)))

    # -- 稀疏：局部窗口 + 跨步长全局 token --------------------------------- #
    def _local_windows(self, t, H, W):
        """以每个 (i,j) 为中心的 ws×ws 真 2D 局部窗口，越界补零。

        t: (B, Hh, N, d) -> (B, N, Hh, ws*ws, d)
        实现：pad 后对 ws² 个偏移分别取切片再 concat。MS 无 Tensor.unfold，
        故用显式切片；GRAPH_MODE 下 ws 为常量，循环在编译期展开。
        ⚠ 内存：会物化 (B,N,Hh,ws²,d)，与 torch 版同量级（sparse 模式本身就贵）。
        """
        ws = self.window_size
        B, Hh, N, d = t.shape
        pad = ws // 2
        # (B, Hh*d, H, W)
        x = t.reshape(B, Hh * d, H, W)
        x = ops.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        slices = []
        for kh in range(ws):
            for kw in range(ws):
                # 取 [kh : kh+H, kw : kw+W]
                s = x[:, :, kh:kh + H, kw:kw + W]        # (B, Hh*d, H, W)
                s = s.reshape(B, Hh, d, H, W)
                slices.append(s.transpose(0, 3, 4, 1, 2))  # (B,H,W,Hh,d)
        # 沿 axis=4 堆叠得 (B,H,W,Hh,ws²,d)，与 torch 版 (B,H,W,Hh,kh,kw,d) 同序
        out = ops.stack(slices, axis=4)
        return out.reshape(B, N, Hh, ws * ws, d)

    def _sparse_attn(self, q, k, v, H, W):
        ws = self.window_size
        stride = ws
        B, Hh, N, d = q.shape

        kw = self._local_windows(k, H, W).reshape(B * N, Hh, ws * ws, d)
        vw = self._local_windows(v, H, W).reshape(B * N, Hh, ws * ws, d)
        qc = q.transpose(0, 2, 1, 3).reshape(B * N, Hh, 1, d)

        gh, gw = H // stride, W // stride
        ng = gh * gw
        kg = k.reshape(B, Hh, H, W, d)[:, :, :gh * stride, :gw * stride, :]
        kg = kg.reshape(B, Hh, gh, stride, gw, stride, d)[:, :, :, stride // 2, :, stride // 2, :]
        kg = kg.reshape(B, Hh, ng, d)
        vg = v.reshape(B, Hh, H, W, d)[:, :, :gh * stride, :gw * stride, :]
        vg = vg.reshape(B, Hh, gh, stride, gw, stride, d)[:, :, :, stride // 2, :, stride // 2, :]
        vg = vg.reshape(B, Hh, ng, d)

        # 广播到每个 token：(B,Hh,ng,d) -> (B*N,Hh,ng,d)
        k_glob = ops.broadcast_to(kg.reshape(B, 1, Hh, ng, d),
                                  (B, N, Hh, ng, d)).reshape(B * N, Hh, ng, d)
        v_glob = ops.broadcast_to(vg.reshape(B, 1, Hh, ng, d),
                                  (B, N, Hh, ng, d)).reshape(B * N, Hh, ng, d)

        k_all = ops.concat([kw, k_glob], axis=2)     # (B*N, Hh, ws²+ng, d)
        v_all = ops.concat([vw, v_glob], axis=2)
        oc = self._math_attn(qc, k_all, v_all)       # (B*N, Hh, 1, d)
        return oc.reshape(B, N, Hh, d).reshape(B, N, Hh * d)

    # -- axial ------------------------------------------------------------ #
    def _axial_attn(self, q, k, v, H, W):
        B, Hh, N, d = q.shape

        # 严格镜像 torch 版的 reshape 序列（含其 head/position 映射语义），
        # 保证与 torch 版数值逐位一致。注意：torch 版把 Hh 融进 batch 后 reshape
        # 回 (B, Hh, H, W, d)，其中 head 与位置的对应并非「头对齐」，而是行主序
        # 重排的结果——要等价就必须照搬同样的 reshape，不能按语义"修正"。
        def attn_1d(tokens):
            # tokens: (X*Hh*L, S, d) -> (X', Hh, S, d) 做 MHA
            t = tokens.reshape(-1, Hh, tokens.shape[1], d)
            return self._math_attn(t, t, t).reshape(-1, tokens.shape[1], d)

        # 行注意力：每行 W 个 token 互看
        qr = q.reshape(B, Hh, H, W, d).reshape(B * Hh * H, W, d)
        kr = k.reshape(B, Hh, H, W, d).reshape(B * Hh * H, W, d)
        vr = v.reshape(B, Hh, H, W, d).reshape(B * Hh * H, W, d)
        out_r = attn_1d(qr).reshape(B, Hh, H, W, d)

        # 列注意力：H/W 转置后同理
        qc = out_r.transpose(0, 1, 3, 2, 4).reshape(B * Hh * W, H, d)
        kc = k.reshape(B, Hh, H, W, d).transpose(0, 1, 3, 2, 4).reshape(B * Hh * W, H, d)
        vc = v.reshape(B, Hh, H, W, d).transpose(0, 1, 3, 2, 4).reshape(B * Hh * W, H, d)
        out_c = attn_1d(qc).reshape(B, Hh, W, H, d).transpose(0, 1, 3, 2, 4)
        return out_c.reshape(B, N, self.num_heads * d)

    # -- forward ---------------------------------------------------------- #
    def construct(self, x):
        B, C, H, W = x.shape
        N = H * W
        seq = x.reshape(B, C, N).transpose(0, 2, 1)   # (B, N, C)

        residual = seq
        h = self.ln1(seq)
        qkv = self.qkv(h)                              # (B, N, 3C)
        # 注意：MindSpore ops.split 签名是 (x, axis=0, output_num=1)，
        # 第二位置参数是 axis 而非份数——与 torch 的 chunk(3, dim) 顺序不同。
        q, k, v = ops.split(qkv, axis=-1, output_num=3)
        q = self._to_heads(q, B, N)
        k = self._to_heads(k, B, N)
        v = self._to_heads(v, B, N)
        q = q * self.scale                              # 预乘 scale（与 torch 一致）

        if self.mode == "window":
            out = self._window_attn(q, k, v, H, W)
        elif self.mode == "axial":
            out = self._axial_attn(q, k, v, H, W)
        elif self.mode == "sparse":
            out = self._sparse_attn(q, k, v, H, W)
        else:  # global
            out = self._global_attn(q, k, v)              # (B, Hh, N, d)
            out = out.transpose(0, 2, 1, 3).reshape(B, N, C)

        seq = residual + out
        seq = seq + self.ffn_drop(self.ffn(self.ln2(seq)))
        return seq.transpose(0, 2, 1).reshape(B, C, H, W)


class AttentionResBlock(nn.Cell):
    """卷积残差 + 多头自注意力混合块（与 torch 版同名同参）。"""

    def __init__(self, channels, num_heads=4, dropout=0.0,
                 attention_mode="global", window_size=7):
        super(AttentionResBlock, self).__init__()
        self.conv = ResBlock(channels)
        self.attn = MultiHeadSelfAttention(
            channels, num_heads=num_heads, dropout=dropout,
            mode=attention_mode, window_size=window_size)

    def construct(self, x):
        x = self.conv(x)
        return self.attn(x)


# --------------------------------------------------------------------------- #
# Backbone / 头 / 整体
# --------------------------------------------------------------------------- #
class SharedBackbone(nn.Cell):
    """共享表示网络（与 torch 版 SharedBackbone 同名同参）。"""

    def __init__(self, in_channels=12, channels=128, num_res_blocks=12,
                 attention_mode="mix", num_attention_layers=4,
                 num_heads=4, attention_dropout=0.0,
                 attn_mode="global", attn_window=7):
        super(SharedBackbone, self).__init__()
        self.channels = channels
        self.attention_mode = attention_mode

        self.conv1 = nn.Conv2d(in_channels, channels, 3, pad_mode="pad",
                               padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        blocks = self._build_blocks(
            num_res_blocks, attention_mode, num_attention_layers,
            channels, num_heads, attention_dropout, attn_mode, attn_window)
        self.blocks = nn.CellList(list(blocks))   # 索引命名，与 torch Sequential 一致

        self.conv_out = nn.Conv2d(channels, channels, 1, has_bias=False)
        self.bn_out = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    @staticmethod
    def _build_blocks(num_res_blocks, mode, num_attn, channels, num_heads,
                      dropout, attn_mode, attn_window):
        if mode == "none" or num_attn <= 0:
            return [ResBlock(channels) for _ in range(num_res_blocks)]
        if mode == "all":
            return [AttentionResBlock(channels, num_heads, dropout, attn_mode, attn_window)
                    for _ in range(num_res_blocks)]
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

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        for blk in self.blocks:
            out = blk(out)
        return self.relu(self.bn_out(self.conv_out(out)))


class PolicyNetwork(nn.Cell):
    """策略头（SequentialCell 索引与 torch nn.Sequential 一致：0=Conv,1=BN,5=Linear）。"""

    def __init__(self, in_channels=64, hidden_channels=32, action_size=81):
        super(PolicyNetwork, self).__init__()
        self.action_size = action_size
        self.policy_head = nn.SequentialCell(
            nn.Conv2d(in_channels, hidden_channels, 1, has_bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dense(hidden_channels, action_size),
        )

    def construct(self, x):
        return self.policy_head(x)


class ValueNetwork(nn.Cell):
    """价值头（Conv→ReLU→Pool→Dense，输出 raw logit）。"""

    def __init__(self, in_channels=64, hidden_channels=64):
        super(ValueNetwork, self).__init__()
        self.value_head = nn.SequentialCell(
            nn.Conv2d(in_channels, hidden_channels, 1, has_bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dense(hidden_channels, 1),
        )

    def construct(self, x):
        return self.value_head(x)


class AlphaGoNet(nn.Cell):
    """MindSpore 版策略-价值网络（与 torch 版 AlphaGoNet 参数名一致）。"""

    def __init__(self,
                 in_channels: int = 12,
                 backbone_channels: int = 128,
                 backbone_res_blocks: int = 12,
                 attention_mode: str = "mix",
                 num_attention_layers: int = 4,
                 num_heads: int = 4,
                 attention_dropout: float = 0.0,
                 attn_mode: str = "global",
                 attn_window: int = 7,
                 policy_channels: int = 32,
                 value_channels: int = 64,
                 action_size: int = 361):
        super(AlphaGoNet, self).__init__()
        self.action_size = action_size

        self.backbone = SharedBackbone(
            in_channels=in_channels,
            channels=backbone_channels,
            num_res_blocks=backbone_res_blocks,
            attention_mode=attention_mode,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            attn_mode=attn_mode,
            attn_window=attn_window,
        )
        self.policy = PolicyNetwork(
            in_channels=backbone_channels,
            hidden_channels=policy_channels,
            action_size=action_size,
        )
        self.value = ValueNetwork(
            in_channels=backbone_channels,
            hidden_channels=value_channels,
        )

    def construct(self, observation):
        """observation: (B, 12, H, W) -> policy logits (B, A), value (B, 1)。"""
        shared = self.backbone(observation)
        return self.policy(shared), self.value(shared)
