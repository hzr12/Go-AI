import torch
import torch.nn as nn
import torch.nn.functional as F


#: squeeze-excitation 的 reduction（通道数除以它得到 SE 的隐藏层宽度）。
#: 16 是 SE-Net 原论文与 KataGo v4+ 的共同取值。
SE_REDUCTION_DEFAULT = 16


class SEGate(nn.Module):
    """逐通道 squeeze-excitation 门控。

    路径：全局平均池化 → Linear(C→C/r) → ReLU → Linear(C/r→C) → sigmoid，
    输出形状 `(B, C, 1, 1)`，与输入**逐通道**相乘即完成重标定。

    为什么不写成 1×1 卷积
    --------------------
    池化之后空间维已经是 1×1，`nn.Conv2d(C, C//r, 1)` 与 `nn.Linear` 在此完全
    等价，但只有 Linear 能把权重表达成 `(out_features, in_features)` 的二维矩阵
    —— 参数对账、逐通道权重的可读性、以及 `test_param_groups` 那一类按
    「卷积 vs 全连接」分组的逻辑都更省事。

     数值口径：910A 无 bf16、AMP 走 fp16，而 `sigmoid` 的输出恒在 `(0,1)`，
    逐通道相乘**不会**放大激活值，所以这里不需要额外的 fp32 兜底。两个
    `nn.Linear` 的累加在 fp16 下仍然可能溢出，但那是它前面 BN 已经归一化过的
    张量，量级与 `ResBlock` 里同样的两级结构一致。
    """

    def __init__(self, channels, reduction=SE_REDUCTION_DEFAULT):
        super(SEGate, self).__init__()
        # reduction 比通道数还大时隐藏层会变成 0 维，那会让 Linear 直接报错；
        # 夹一个下限 1，让「把 SE 压到最狠」仍是一个可表达的配置。
        hidden = max(1, channels // reduction)
        self.channels = channels
        self.reduction = reduction
        self.hidden = hidden
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (B, C, H, W) -> squeeze -> (B, C) -> 激励 -> (B, C) -> 广播回 (B,C,1,1)
        w = self.fc1(self.pool(x).flatten(1))
        w = self.fc2(F.relu(w))
        return torch.sigmoid(w).reshape(-1, self.channels, 1, 1)


class SEBottleneck(nn.Module):
    """KataGo v4+ 的 bottleneck 残差块 + squeeze-excitation。

    结构
    ----
        x → BN1 → ReLU → Conv 1×1 (C → mid) → BN2 → ReLU → Conv 3×3 (mid → mid)
          → Conv 1×1 (mid → C)
          → 逐通道乘 SE(x)
          → + x

    三条与 `ResBlock` 不同的**语义**选择（不是风格偏好，别顺手改回去）
    ------------------------------------------------------------------
    1. **pre-activation**：BN 在 conv **之前**。`ResBlock` 是 conv → BN → ReLU
       （post-activation），两者的数值不等 —— 顺序是被测试钉住的，见
       `tests/test_se_bottleneck.py::test_post_activation_wiring_would_give_a_
       different_answer`。
    2. **残差支路没有 BN**：`conv3` 之后直接进 SE、再加 x。KataGo v2 起刻意去掉了
       残差支路的 BN —— BN 的 `running_var` 在按 group 采样的训练下不稳定，而
       残差支路把这种不稳定沿 17 个块逐层放大。代价是本块初始化时不是恒等映射
       （`ResBlock` 靠 `bn2.weight` 归零换取恒等捷径，这里**不**做那个初始化）。
    3. **瓶颈 + SE**：中段压到 `mid = C // 2`，SE 的 reduction 默认 16。C=160 上
       本块 **87,050** 参数，同宽度的 `ResBlock` 是 461,440 —— 省下的 5.3 倍
       正是 9.25M 预算能装下 17 个块的前提。

     SE 的输入是**块入口的 x**（KataGo 的 `applySE(input, afterConv3, ...)`），
    不是 `conv3` 的输出：门控描述的是「这一层该关注哪些通道」，用块入口才与
    该层的输入分布对齐。

     关于梯度检查点：本块的 BN 与 `ResBlock` 的 BN 在 `GC_LEGACY` 段里是同类
    东西，重算时的 running stats 还原由 `backbone.py` 的 `_BatchNormStatGuard`
    统一兜住（`_collect_batchnorms` 遍历 `modules()`，本块的两个 BN 自动被收进去），
    这里**不需要**、也刻意不做任何 BN 统计的额外处理。
    """

    def __init__(self, channels, mid_channels=None, se_reduction=SE_REDUCTION_DEFAULT):
        super(SEBottleneck, self).__init__()
        self.channels = channels
        self.mid_channels = (channels // 2) if mid_channels is None else int(mid_channels)
        self.se_reduction = se_reduction

        # pre-activation：BN1 挂在 conv1 之前，BN2 挂在 conv2 之前；
        # conv3 之后**没有** BN3（见类文档第 2 条）。
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, self.mid_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, 3,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(self.mid_channels, channels, 1, bias=False)
        self.se = SEGate(channels, reduction=se_reduction)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.conv3(out)
        out = out * self.se(x)   # 逐通道重标定，门控由块入口 x 算出
        return out + x           # 残差支路无 BN
