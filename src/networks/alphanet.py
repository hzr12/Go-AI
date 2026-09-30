import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .backbone import (
    CrossAttnRes,
    GC_CROSS_ATTN_RES,
    GC_MAMBA,
    GC_RES,
    GC_TRANSFORMER,
    GradCheckpointMixin,
    MambaLTI,
    ResBlock,
    SharedBackbone,
    TransformerBlock,
)
from .policy_network import FCPolicyHead, PolicyNetwork
from .value_network import FCValueHead, ValueNetwork


class AlphaGoNet(nn.Module):
    """
    监督学习用的策略-价值网络（AlphaGoZero 风格）。

    相比原版删除了 fast_policy 头：在 9x9 自我对弈里 fast_policy 占约 48% 前向算力
    却从未被 MCTS/rollout 使用，纯监督训练更不需要它。

    输入通道从 19 降到 12（原 19 通道里有 13 个恒为 0）。12 通道布局见
    src/data/dataset.py 的 build_state_tensor / GoBoardFeature 文档。
    """

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
                 action_size: int = 362,
                 use_checkpoint: bool = False,
                 arch: str = "resnet",
                 res_blocks: int = 0,
                 convnext_blocks: int = 0,
                 attn_blocks: int = 0,
                 value_res_blocks: int = 3,
                 policy_layers: int = 2):
        """
        Args:
            attention_mode:      主干中注意力的使用方式
                                 "none"(纯卷积) | "mix"(卷积+注意力混合) | "all"(全注意力)
            num_attention_layers: mix 模式下注意力块数量
            num_heads:           多头注意力头数
            attention_dropout:   注意力 dropout
            attn_mode:           注意力计算模式 "global"|"window"|"axial"
            attn_window:         window 模式的窗口边长
            arch:                网络架构风格 "resnet" (默认) | "convnext"
        """
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
            use_checkpoint=use_checkpoint,
            arch=arch,
            res_blocks=res_blocks,
            convnext_blocks=convnext_blocks,
            attn_blocks=attn_blocks,
        )

        self.policy = PolicyNetwork(
            in_channels=backbone_channels,
            hidden_channels=policy_channels,
            action_size=action_size,
            num_layers=policy_layers,
        )

        self.value = ValueNetwork(
            in_channels=backbone_channels,
            hidden_channels=value_channels,
            num_res_blocks=value_res_blocks,
            arch=arch,
        )

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observation: (batch, in_channels, H, W)
        Returns:
            policy: (batch, action_size) logits
            value:  (batch, 1) 黑方视角价值，值域 [-1,1]（-1=白胜 1=黑胜）。
                     ⚠ 不是「sigmoid 映射到 [0,1] 的胜率」：v19 的 ValueNetwork 是
                     裸线性输出（无 Tanh），v21 的 FCValueHead 末尾有 nn.Tanh()，
                     口径按 `inference.py:6` 的「tanh 后落在 [-1,1]」解释。
                     要胜率请自行做 (v+1)/2，不要套 sigmoid。
        """
        shared_state = self.backbone(observation)
        policy = self.policy(shared_state)
        value = self.value(shared_state)
        return policy, value

    def get_policy(self, observation: torch.Tensor) -> torch.Tensor:
        """⚠ 若同时需要 policy 和 value，请用 forward() 避免 backbone 重复计算。"""
        return self.policy(self.backbone(observation))

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """⚠ 若同时需要 policy 和 value，请用 forward() 避免 backbone 重复计算。"""
        return self.value(self.backbone(observation))


# =========================================================================== #
# v21：D1 的唯一训练结构（P4.2 落地）
# =========================================================================== #
#: v21 结构的**唯一来源**（D1 v2：训练全部走 v21，CLI 结构 flag已归档）。
#:
#: 与 `scripts/search_arch.py::ANCHOR_V21` 是**两份独立记录**，必须相等 ——
#: search_arch 那份按 P4.9 的规矩**不得 import 网络代码**（否则标定会被网络
#: 实现牵着走），所以对账只能放在测试里：`tests/test_v21_budget.py` 逐键比对。
#: 改任何一边都会让对账测试转红。
#:
#: 参数量锚（硬数，非窗口）：主干 6,558,696 / 全网 9,067,443
#:   （FCPolicyHead 2,366,730 + FCValueHead 142,017 + 主干）。
#: `grad_checkpoint` 默认 1 是**用户 2026-09-27 裁决**（ResBlocks 必开、
#: Mamba/Transformer 建议开），不是随手默认；它只在 `self.training` 且未开
#: compile 时生效（P4.6b 的互斥守卫，见 `build_v21_net`）。
V21_CFG = {
    # ---- 形状 ----
    'in_channels': 17,          # P4.3 的 17 路特征平面
    'channels': 184,            # 主干宽度
    'n_res': 8,                 # ResBlock ×8（占 74% 参数的大户）
    'n_mamba': 4,               # MambaLTI ×4（Mamba-2，per-head scalar A）
    'n_trans': 2,               # TransformerBlock ×2
    'n_cross': 2,               # CrossAttnRes ×2（消费第 1/5/9 号块的输出）
    'ffn_hidden': 240,
    'num_heads': 4,
    'attn_dropout': 0.1,        # 与 train_sft `--attention-dropout` 默认一致
    # ---- 行为（非结构，可被调用方覆盖）----
    'grad_checkpoint': 1,       # 用户裁决：必开；compile=1 时由构建器关掉并记日志
    # ---- 预算锚（tests/test_v21_budget.py 逐项断言）----
    'params_backbone': 6_558_696,
    'params_total': 9_067_443,
}


class V21Backbone(GradCheckpointMixin, nn.Module):
    """v21 主干（P4.2）：stem + 8/4/2/2 十六块 + out。

    段划分 = `task-p4-6b-report.md` §2 的 (b) 粒度：**同类型连续块合并成一段**，
    三路抽头用 `run_segment(..., tap_positions=...)` 从段内取，不切段边界：
        stem → [ResBlock ×8](tap 0,4 = s1/s5) → [MambaLTI ×4](tap 0 = s9)
             → [TransformerBlock ×2] → [CrossAttnRes ×2] → out
    抽头下标按 P4.1 §7.3 读法 A/B（**stem 不计数**）：s1=ResBlock#1、
    s5=ResBlock#5、s9=MambaLTI#1。

    ⚠ 与 `tests/test_grad_checkpointing.py::V21BackboneHarness` 是**同一契约的
    两份实现**（harness 先行、本类照抄，P4.6b 报告 §8.2 指定）—— 改一处必须
    同步另一处，`test_v21_budget` 与 `test_grad_checkpointing` 会分别抓住。
    两处**有意不同**的只有 stem 的写法：harness 用 `Sequential(Conv, BN)`，
    本类把 Conv 单独命名为 `stem`（BN 另放 `stem_bn`），为的是让 state_dict
    键落在 `backbone.stem.weight` 上 —— GoAI 的 `_STEM_WEIGHT_KEYS` 靠这个键
    做通道数推断与回读校验（见 `__init__` 内注释）。数学完全等价。
    mixin 在 `nn.Module` **之前**（B5：防 `run_segment` 这类通用名被将来
    `nn.Module` 的同名成员静默劫持）。
    """

    def __init__(self, cfg=None, *, in_channels=None, attn_dropout=None,
                 grad_checkpoint=None):
        super().__init__()
        c = dict(V21_CFG if cfg is None else cfg)
        ic = c['in_channels'] if in_channels is None else int(in_channels)
        if ic != c['in_channels']:
            raise ValueError(
                f'v21 的 in_channels 固定为 {c["in_channels"]}（D1：结构由 V21_CFG '
                f'唯一决定，没有 12ch/14ch 之类的变体），收到 {ic}。'
                f'旧 12 通道结构走 AlphaGoNet（arch=resnet/convnext）。')
        drop = c['attn_dropout'] if attn_dropout is None else attn_dropout
        gc = c['grad_checkpoint'] if grad_checkpoint is None else int(grad_checkpoint)
        # 记下**实际生效**的取值（覆盖发生时 ≠ CFG 默认），测试据此判断
        # 行为参数（dropout / 检查点开关）有没有被采纳 —— 结构参数不在此列，
        # 它们压根不进这张表。
        c['attn_dropout'] = drop
        c['grad_checkpoint'] = gc
        self.channels = c['channels']
        self.cfg = c

        # stem 必须让 `backbone.stem.weight` 这个键存在（GoAI 的
        # `_STEM_WEIGHT_KEYS` 只认 backbone.conv1 / backbone.stem / stem / conv1
        # 四个键，推断与回读校验都靠它）—— 所以不用 `nn.Sequential`：
        # Sequential 会把卷积埋进 `.0`，键变成 `backbone.stem.0.weight`，
        # 双代接缝会以「找不到 stem」拒绝这个构建器。
        self.stem = nn.Conv2d(ic, c['channels'], 3, padding=1, bias=False)  # 3×3（用户裁决）
        self.stem_bn = nn.BatchNorm2d(c['channels'])
        blocks = ([ResBlock(c['channels']) for _ in range(c['n_res'])]
                  + [MambaLTI(c['channels']) for _ in range(c['n_mamba'])]
                  + [TransformerBlock(c['channels'], ffn_hidden=c['ffn_hidden'],
                                      attn_dropout=drop)
                     for _ in range(c['n_trans'])]
                  + [CrossAttnRes(c['channels'], tap_channels=(c['channels'],) * 3,
                                  ffn_hidden=c['ffn_hidden'], attn_dropout=drop)
                     for _ in range(c['n_cross'])])
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Sequential(
            nn.Conv2d(c['channels'], c['channels'], 1, bias=False),
            nn.BatchNorm2d(c['channels']))
        # 段切片（与 forward 的四段一一对应；测试用同一套常数复核）
        self.n_res = c['n_res']
        self.n_mamba = c['n_mamba']
        self.n_trans = c['n_trans']
        # 开关必须在 super().__init__() 之后、第一次 forward 之前开（§8.2-2）
        self._init_grad_checkpointing(
            res=bool(gc), mamba=bool(gc), transformer=bool(gc),
            cross_attn_res=False,   # §2 裁决：性价比不合算，且默认不开（P4.6b）
            legacy=False)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))   # = harness 的 F.relu(stem(x))
        r_end = self.n_res
        m_end = r_end + self.n_mamba
        t_end = m_end + self.n_trans
        x, taps = self.run_segment(self.blocks[:r_end], (x,), GC_RES,
                                   tap_positions=(0, 4))
        s1, s5 = taps
        x, taps = self.run_segment(self.blocks[r_end:m_end], (x,), GC_MAMBA,
                                   tap_positions=(0,))
        s9 = taps[0]
        x, _ = self.run_segment(self.blocks[m_end:t_end], (x,), GC_TRANSFORMER)
        x, _ = self.run_segment(self.blocks[t_end:], (x, (s1, s5, s9)),
                                GC_CROSS_ATTN_RES)
        return F.relu(self.out(x))


class V21Net(AlphaGoNet):
    """D1 的 v21 全网 = `V21Backbone` + `FCPolicyHead` + `FCValueHead`。

    继承 `AlphaGoNet` 是为了 `isinstance(model, AlphaGoNet)` 的既有判断继续
    成立（`forward`/`get_policy`/`get_value` 全部复用，它只碰
    `self.backbone/self.policy/self.value` 三个属性）。

    ⚠ **不调用 `AlphaGoNet.__init__`**：D5 把它的签名逐字锁死
    （`test_legacy_class_signatures_untouched` 按 AST 比对），v21 的结构又与
    它完全不同，硬塞分支进去既违反 D1（CLI 不可见的 arch 分支）也碰不动锁。
    于是这里直接走 `nn.Module.__init__` 后自建三件套 —— 这是有意的偏离，
    不是漏写 `super().__init__()`。
    """

    def __init__(self, cfg=None, *, in_channels=None, action_size=362,
                 attn_dropout=None, grad_checkpoint=None, **arch_kwargs):
        nn.Module.__init__(self)          # ← 见类 docstring（绕开被锁死的 __init__）
        c = dict(V21_CFG if cfg is None else cfg)
        self.action_size = int(action_size)
        # FCPolicyHead 的 flatten 维度随棋盘边长走：48·board²→128
        board = int(round((self.action_size - 1) ** 0.5))
        if board * board + 1 != self.action_size:
            raise ValueError(
                f'action_size={self.action_size} 不是 board²+1 的形式；'
                f'v21 的 policy 头按棋盘边长铺平，无法构造。')
        self.backbone = V21Backbone(c, in_channels=in_channels,
                                    attn_dropout=attn_dropout,
                                    grad_checkpoint=grad_checkpoint)
        self.policy = FCPolicyHead(c['channels'], board_size=board,
                                   action_size=self.action_size)
        self.value = FCValueHead(c['channels'])

    def set_grad_checkpointing(self, enabled=None, **kinds):
        """开关转发给主干（§8.2 形态 (b)：开关的持有者 == `run_segment` 的调用者）。"""
        self.backbone.set_grad_checkpointing(enabled, **kinds)
        return self


def build_v21_net(*, in_channels=None, action_size=362, attn_dropout=None,
                  grad_checkpoint=None, **arch_kwargs):
    """17ch v21 构建器（D1；`src.inference` 已注册为 17 通道的构建器）。

    契约（`inference._build_for_in_channels` 的调用形态）：
        builder(*, in_channels: int, **arch_kwargs) -> nn.Module

    `arch_kwargs` 里传进来的旧结构参数（`backbone_channels` / `attention_mode`
    / `arch` / `res_blocks` / `policy_layers` …）按 **D1 归档：一律忽略** ——
    旧 shell 的结构 flag 仍会被 argparse 接受，但不再改变建出的网络（这正是
    「结构参数已归档」那句 run.txt 注释的实现）。真正被采纳的只有行为参数：
    `action_size`（棋盘尺寸）、`attention_dropout`→`attn_dropout`（GoAI 侧的
    键名不同，这里做一次对齐）、`grad_checkpoint`（检查点开关）。
    """
    if attn_dropout is None:
        attn_dropout = arch_kwargs.get('attention_dropout')
    return V21Net(in_channels=in_channels, action_size=action_size,
                  attn_dropout=attn_dropout, grad_checkpoint=grad_checkpoint,
                  **arch_kwargs)
