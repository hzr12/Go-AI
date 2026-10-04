"""
围棋 AI 推理引擎（监督学习 / SFT 模型）

兼容 src/networks.alphanet.AlphaGoNet 输出的 (policy, value)：
    - policy: 棋盘上每点 + 虚着(pass) 的概率分布
    - value : 当前执子方视角的局面胜率，tanh 后落在 [-1, 1]

特征由 src.game.go_rules.GoBoard.feature_planes 统一生成，通道数与模型 stem 的
in_channels 一致（现役 12，由 `_infer_in_channels` 从权重形状推断），
与 src.data.dataset 使用同一套特征工程，避免训练/推理不一致。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import re
import time
import numpy as np
import torch

from src.game.go_rules import GoBoard
from src.networks.alphanet import AlphaGoNet


def _ensure_torch_npu():
    """导入 torch_npu（必须先于 .to('npu') 调用，注册 Ascend 后端）。返回是否可用。"""
    try:
        import torch_npu  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 通道数 → 网络构建器注册表（P4.8/P4.13）
#
# 接线方式：
#     from src.inference import register_in_channels_builder
#     register_in_channels_builder(N, builder)
# builder 契约：builder(*, in_channels: int, **arch_kwargs) -> nn.Module
#     - in_channels 由 GoAI 从权重 stem 形状推断后传入，必须用于 stem；
#     - arch_kwargs 是 GoAI.__init__ 的架构参数字典，不认识的键直接忽略；
#     - 返回裸模块，.to(device) / eval() 由 GoAI 负责；
#     - stem 必须命名在 `_STEM_WEIGHT_KEYS` 之内：推断（读 checkpoint）与校验
#       （回读建好的模型）两端都靠这几个键，命名落在外面 = 通道数无从校验，
#       GoAI 会直接报错而不是放行一个随机 stem。
# --------------------------------------------------------------------------- #
_STEM_WEIGHT_KEYS = (
    # resnet / convnext 现行架构的 stem；后面几项是给未来架构变体留的回退
    "backbone.conv1.weight",
    "backbone.stem.weight",
    "stem.weight",
    "conv1.weight",
)

_IN_CHANNEL_BUILDERS = {}


def _stem_in_channels(tensors):
    """从 state_dict 的张量**形状**读 stem 的 in 维；找不到可识别键返回 None。

    读形状而不是读配置字段：checkpoint 里可能没有配置，形状是不会说谎的那一侧。
    """
    for k in _STEM_WEIGHT_KEYS:
        v = tensors.get(k)
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            return int(v.shape[1])
    return None


def register_in_channels_builder(in_channels, builder):
    """注册 in_channels → 网络构建器（同通道数后注册者生效，测试可临时覆盖）。"""
    _IN_CHANNEL_BUILDERS[int(in_channels)] = builder


def _state_key_sample(state, limit=6):
    """checkpoint 顶层键样例（只用于报错文案；键类型可能混着非字符串，故 str 化）。"""
    try:
        return [str(k) for k in list(state)[:limit]]
    except TypeError:  # pragma: no cover - state 不是可迭代映射
        return [type(state).__name__]


#: 旧代 value 残差块的键名前缀（`value.resN.*`，N 从 1 起）。
_LEGACY_VALUE_BLOCK_RE = re.compile(r'^value\.res(\d+)\.(.+)$')
#: 当前 value 残差块的键名前缀（`value.res_blocks.N.*`，N 从 0 起）。
_VALUE_BLOCK_RE = re.compile(r'^value\.res_blocks\.(\d+)\.')


def remap_legacy_value_keys(sd):
    """把旧版 `ValueNetwork` 的键名归一到当前代码；返回 (state_dict, 移动数)。

    当前 `value_network.py` 用 `self.res_blocks = nn.ModuleList([...])`
    → `value.res_blocks.0.*`；`e2d0b57` 之前直接在 value 下挂 `res1 / res2 / ...`
    → `value.res1.*`。现役 `models/sft_19x19_v12.pth` 就是那一代的权重（2 块）。

    只改键名、**不动任何数值**：块类（`_ValueResBlock`）、顺序、通道数都没变，
    所以重映射后与源模型逐位等价。块数由 `_infer_architecture` 另外从权重
    推断（`value_res_blocks`）—— 只重映射不改块数的话，建出来的 3 块 head 里
    第 3 块仍然没有权重可载，那块还是随机的（这正是本函数存在的另一半理由）。

    `scripts/inspect_ckpt.py` 复用本函数（同一份正则与口径），避免「加载能过
    但 inspect 说缺键」这类两份实现互相打架的毛病。
    """
    if any(k.startswith('value.res_blocks.') for k in sd):
        return sd, 0            # 已是当前命名：幂等
    if not any(_LEGACY_VALUE_BLOCK_RE.match(k) for k in sd):
        return sd, 0            # 没有旧版键也没得可映射
    out = {}
    moved = 0
    for k, v in sd.items():
        m = _LEGACY_VALUE_BLOCK_RE.match(k)
        if m:
            out['value.res_blocks.%d.%s' % (int(m.group(1)) - 1, m.group(2))] = v
            moved += 1
        else:
            out[k] = v
    return out, moved


def infer_value_head(state):
    """从 state_dict 读 value 头的两个形状参数：**两种命名都认**。

    返回 `{'value_channels': int|None, 'value_res_blocks': int|None}`（读不到的
    键给 None，调用方据此跳过重建）。

    - `value_channels`：`value.downsample.0.weight` 的 out 维。两代都在，形状
      不会说谎。 旧实现查的是 `value.value_head.0.weight` —— 那个键**两代都
      不存在**（value 头一直是 `downsample` Sequential + 残差块 + `fc`），
      所以过去这条推断等于没跑，value_channels 恒为默认 64。
    - `value_res_blocks`：`res_blocks.N` 的最大 N+1（旧命名 `resN` 折成 N-1）。
      不推断的后果就是「权重 2 块、模型 3 块 ⇒ 第 3 块随机初始化」。
    """
    out = {'value_channels': None, 'value_res_blocks': None}
    for k, v in state.items():
        if k == 'value.downsample.0.weight' and isinstance(v, torch.Tensor):
            out['value_channels'] = int(v.shape[0])
            break
    idx = set()
    for k in state:
        m = _VALUE_BLOCK_RE.match(k)
        if m:
            idx.add(int(m.group(1)))
            continue
        m = _LEGACY_VALUE_BLOCK_RE.match(k)
        if m:
            idx.add(int(m.group(1)) - 1)
    if idx:
        out['value_res_blocks'] = max(idx) + 1
    return out


def _katago_v7_net(*, in_channels: int, **arch_kwargs):
    """22 通道 = KataGo NBT+Transformer（spec §3/§4，配置 `b11c256h4nbttflrs-...`）。

    22 是官方 `fillRowV7` 的空间通道数（19 全局特征走独立的 `global_fc`，
    不经 stem）。`in_channels` 必须等于 22 —— `NbtTfNet` 的 stem 是
    `Conv2d(22→256, 3×3)`，给它别的数就是建出了与 spec 不符的结构。

    `arch_kwargs` 里本模型**只认** `use_checkpoint` 与 `attn_dropout` 两个行为
    参数；结构键**不接受覆盖**（与 `KATAGO_SE_CFG` 同一立场：结构只由
    `NBT_TF_CFG` 决定）。传别的键会被静默忽略 —— 刻意不报错，因为
    `GoAI.__init__` 会把一整套 12 通道时代的结构参数一起透传下来。
    """
    from src.networks.katago_v7 import NBT_TF_CFG, build_katago_v7_net
    if int(in_channels) != NBT_TF_CFG['in_channels']:
        raise ValueError(
            f'V7 构建器只接 {NBT_TF_CFG["in_channels"]} 通道，收到 {in_channels}。'
            f'22 是官方 fillRowV7 的空间通道数，改它等于改 spec。')
    return build_katago_v7_net(
        use_checkpoint=arch_kwargs.get('use_checkpoint'),
        attn_dropout=arch_kwargs.get('attn_dropout'))


register_in_channels_builder(22, _katago_v7_net)


def _build_for_in_channels(in_channels, **arch_kwargs):
    """按通道数建网；建完立刻校验 stem 通道数与请求一致后返回。

    校验抓的是「构建器无视 in_channels、静默建了别的通道数」——这种错
    在 load_state_dict 之前必须拦下，报错同时给出期望与实际两个数字。

    `built is None`（构建器返回的模型里四个候选 stem 键一个都没有）同样是
    **硬错误**而不是「跳过校验」：那条路上通道数对不对根本无从判断，等于
    放行一个 stem 可能是随机权重的模型。注册进来的构建器必须把 stem 命名在
    `_STEM_WEIGHT_KEYS` 之内，否则 GoAI 侧所有通道数保证失效。
    """
    in_channels = int(in_channels)
    builder = _IN_CHANNEL_BUILDERS.get(in_channels)
    if builder is None:
        registered = ", ".join(str(k) for k in sorted(_IN_CHANNEL_BUILDERS))
        raise RuntimeError(
            f"in_channels={in_channels} 没有可用的网络构建器（已注册: {registered}）。"
            f"{in_channels} 通道结构未接线：调用 "
            f"src.inference.register_in_channels_builder({in_channels}, builder)，"
            f"builder 签名 builder(*, in_channels: int, **arch_kwargs) -> nn.Module"
            f"（{in_channels}ch 的结构类由该 builder 提供，且必须把 in_channels "
            f"传给 stem）。")
    model = builder(in_channels=in_channels, **arch_kwargs)
    built = _stem_in_channels(model.state_dict())
    if built is None:
        raise RuntimeError(
            f"构建器返回的模型里找不到 stem 卷积（GoAI 只认这几个键: "
            f"{', '.join(_STEM_WEIGHT_KEYS)}），无法校验 in_channels={in_channels}。"
            f"构建器必须返回一个 stem 键落在上述候选内的 nn.Module —— "
            f"否则通道数校验形同虚设，等于放行随机 stem。")
    if built != in_channels:
        raise RuntimeError(
            f"通道数不匹配：期望 in_channels={in_channels}（由权重 stem 形状推断），"
            f"但构建器返回的模型 stem 实际为 in_channels={built}。"
            f"检查为 {in_channels} 注册的构建器是否把 in_channels 传给了 stem。")
    return model


def _legacy_alpha_go_net(in_channels, **arch_kwargs):
    """12 通道现行结构（resnet / convnext / se_bottleneck）的构建器。

    现在**只有这一个通道数接线**。17 路那一代（v21）随它的构建器一起退役，
    未注册的通道数会走 `_build_for_in_channels` 的报错分支，那条分支已经把
    「谁来接、怎么接」的契约写在消息里了。
    """
    return AlphaGoNet(in_channels=in_channels, **arch_kwargs)


register_in_channels_builder(12, _legacy_alpha_go_net)


class GoAI:
    """基于 SFT 模型的围棋对弈 / 分析引擎。

    用法示例::

        ai = GoAI(model_path="models/sft_19x19.pth", board_size=19, device="cuda")
        # 自对弈一局
        ai.self_play(verbose=True, temperature=0.8)
        # 人机对弈（人类执黑先手）
        ai.play_against_human(human_color=1)
        # 分析某个局面的 top-k 候选着法
        ai.analyze()
    """

    # NPU batch 归桶：CANN 按输入形状编译算子，MCTS 的零散 batch（尾批 6、7 等）
    # 每种形状都要单独编译一次；归桶补零后形状固定，编译缓存才能跨调用命中。
    _NPU_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256)

    def __init__(self, model_path=None, board_size=19, device="auto", use_amp=False,
                 backbone_channels=128, backbone_res_blocks=12, policy_channels=32, value_channels=64,
                 attention_mode="mix", num_attention_layers=4, num_heads=4, attention_dropout=0.0,
                 attn_mode="global", attn_window=7, compile=False, tf32=False,
                 channels_last=True, policy_layers=2, value_res_blocks=3):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "npu" if _ensure_torch_npu() and torch.npu.is_available() else "cpu")
        self.device = device
        self.is_npu = device.startswith("npu")
        if self.is_npu and not _ensure_torch_npu():
            raise RuntimeError("--device npu 需要 torch_npu（须与 CANN 版本匹配，"
                               "910A 用 torch_npu 1.11~2.1 均可）")
        # 910A 不支持 bf16，NPU 上一律 fp16 autocast；CPU 不开 amp
        self.use_amp = use_amp and (self.device.startswith("cuda") or self.is_npu)
        if self.is_npu:
            print("[GoAI] Ascend NPU 推理：fp16 autocast（910A 无 bf16），"
                  "math 注意力，勿开 --compile。若每次启动 warmup 都超过 1 分钟，"
                  "先执行 export ASCEND_CACHE_PATH=~/ascend_cache 持久化算子编译缓存")
        self.board_size = board_size
        self._ort = None
        self._ort_dynamic_batch = False
        # 网络侧压：tf32 让 V100/Amp 上的 fp32 matmul 走 TensorFloat-32（约 2-4x 提速，
        # 精度损失对推理可忽略）；channels_last 让 conv 走 NHWC 内存布局（conv 友好）。
        self.tf32 = tf32 and self.device.startswith("cuda")
        if self.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.channels_last = channels_last and self.device.startswith("cuda")

        # ---- 双代加载（P4.8/P4.13）：先读权重、由 stem 形状定通道数，再建网 ----
        # torch.load 不消耗 RNG，把它提到建网之前不改变随机初始化的 RNG 流；
        # 建网次数与改前一致（无 checkpoint/架构匹配 = 1 次，失配重建 = 2 次）。
        state = None
        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            # 兼容直接保存的 state_dict 或 {"model": state_dict}
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
        self._in_channels = 12  # 无 checkpoint → 旧默认（零回归）
        if state is not None:
            # 读了 checkpoint 就必须能从中读出通道数。回退 12 是本设计里唯一
            # 「不响就走」的路：回退后建出的是 12ch 模型，而未知架构的权重会被
            # load_state_dict 整份报成 missing，strict=False 照单全收 —— 玩家
            # 拿到的是一个随机初始化、形状却完全正常的模型（P4.8 fix 轮堵死）。
            inferred_ic = self._infer_in_channels(state)
            if inferred_ic is None:
                raise RuntimeError(
                    f"权重里找不到可识别的 stem 卷积（GoAI 只认: "
                    f"{', '.join(_STEM_WEIGHT_KEYS)}），无法确定 in_channels；"
                    f"拒绝回退默认 12 —— 那样会用随机初始化的 12ch 模型"
                    f"「装」一份陌生架构的权重并静默开跑。checkpoint 顶层键样例: "
                    f"{_state_key_sample(state)}（GoAI 只解包 'model' 键下的 "
                    f"state_dict；若是 optimizer 打包形态请改存 "
                    f"{{'model': state_dict}}）。")
            self._in_channels = inferred_ic
            if inferred_ic != 12:
                print(f"[GoAI] 权重 stem 推断 in_channels={inferred_ic}，"
                      f"特征/前向通道数随模型驱动")
            # 旧代 value 头命名（value.res1/res2/...）归一到当前命名（e2d0b57
            # 之前 vs 之后）。纯键名重映射，数值不动 —— 不做这一步的话旧权重
            # 的 value 块会被 strict=False 报成 unexpected 丢掉、模型里的
            # value 块则停在随机初始化，而这一切**不报错**。
            state, _moved_value_keys = remap_legacy_value_keys(state)
            if _moved_value_keys:
                print(f"[GoAI] 旧版 value 头键名（value.res1/res2/...）已归一为 "
                      f"value.res_blocks.*：{_moved_value_keys} 个张量（数值未改）")

        net_kwargs = dict(
            backbone_channels=backbone_channels,
            backbone_res_blocks=backbone_res_blocks,
            attention_mode=attention_mode,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            attn_mode=attn_mode,
            attn_window=attn_window,
            policy_channels=policy_channels,
            value_channels=value_channels,
            value_res_blocks=value_res_blocks,
            action_size=board_size * board_size + 1,  # +1 = 虚着
            policy_layers=policy_layers,
        )
        self.model = _build_for_in_channels(self.in_channels, **net_kwargs).to(
            self.device)
        # 这里就进 eval，别等下面那行 —— torch.compile 的 warmup 前向在
        # `self.model.eval()`（下面）**之前**就跑，而训练出来的模型可能带 grad
        # checkpointing（train_sft 默认开）：GC 只在 training 态生效，warmup 时
        # 模型默认 training=True 会让 compile 撞上 GC 互斥守卫并静默回退 eager
        # （P4.6b §8.2④ 那组组合）。eval 下检查点恒关闭，compile 正常走。
        self.model.eval()
        # torch.compile 融合算子（GPU 上约 20-40% 提速），不支持时回退 eager。
        # 注意：torch.compile 是惰性的，错误在首次前向才抛出，因此编译后用
        # dummy 输入做一次 warmup 以触发真实编译并捕获异常。
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if compile and self.is_npu:
            print("[GoAI] NPU 不支持 torch.compile，已忽略")
        elif compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, dynamic=False)
                with torch.inference_mode():
                    dummy = torch.zeros(1, self.in_channels, self.board_size,
                                        self.board_size, device=self.device)
                    if self.channels_last:
                        dummy = dummy.to(memory_format=torch.channels_last)
                    self.model(dummy)
                print("[GoAI] 已启用 torch.compile 算子融合")
            except Exception as e:  # noqa: BLE001
                print(f"[GoAI] torch.compile 不可用，回退 eager: {e}")
                # 重建未编译模型（前述 compile 包装可能已部分生效）
                self.model = _build_for_in_channels(self.in_channels,
                                                    **net_kwargs).to(self.device)
                if self.channels_last:
                    self.model = self.model.to(memory_format=torch.channels_last)
        if self.tf32:
            print("[GoAI] 已启用 TF32 矩阵乘法加速 (CUDA)")
        self.model.eval()

        if state is not None:
            # 从权重形状自动推断架构参数，防止 mismatch
            inferred_bs = self._infer_board_size(state)
            inferred_arch = self._infer_architecture(state)
            needs_rebuild = False
            if inferred_bs is not None and inferred_bs != self.board_size:
                print(f"[GoAI] 权重按 {inferred_bs} 路训练（当前 board_size={self.board_size}），"
                      f"已按权重自动调整棋盘大小")
                self.board_size = inferred_bs
                needs_rebuild = True
            # 检查 backbone_channels 等是否匹配
            if inferred_arch.get("backbone_channels") and \
                    inferred_arch["backbone_channels"] != backbone_channels:
                print(f"[GoAI] 权重 backbone_channels={inferred_arch['backbone_channels']}"
                      f"（默认 {backbone_channels}），已自动调整")
                net_kwargs["backbone_channels"] = inferred_arch["backbone_channels"]
                needs_rebuild = True
            if inferred_arch.get("backbone_res_blocks") and \
                    inferred_arch["backbone_res_blocks"] != backbone_res_blocks:
                print(f"[GoAI] 权重 res_blocks={inferred_arch['backbone_res_blocks']}"
                      f"（默认 {backbone_res_blocks}），已自动调整")
                net_kwargs["backbone_res_blocks"] = inferred_arch["backbone_res_blocks"]
                needs_rebuild = True
            if inferred_arch.get("num_attention_layers") and \
                    inferred_arch["num_attention_layers"] != num_attention_layers:
                print(f"[GoAI] 权重 attention_layers={inferred_arch['num_attention_layers']}"
                      f"（默认 {num_attention_layers}），已自动调整")
                net_kwargs["num_attention_layers"] = inferred_arch["num_attention_layers"]
                needs_rebuild = True
            if inferred_arch.get("policy_channels") and \
                    inferred_arch["policy_channels"] != policy_channels:
                net_kwargs["policy_channels"] = inferred_arch["policy_channels"]
                needs_rebuild = True
            if inferred_arch.get("value_channels") and \
                    inferred_arch["value_channels"] != value_channels:
                print(f"[GoAI] 权重 value_channels={inferred_arch['value_channels']}"
                      f"（默认 {value_channels}），已自动调整")
                net_kwargs["value_channels"] = inferred_arch["value_channels"]
                needs_rebuild = True
            if inferred_arch.get("value_res_blocks") and \
                    inferred_arch["value_res_blocks"] != value_res_blocks:
                # 少一块就是「value 头有一块停在随机初始化」：形状全对、不报错，
                # 但 value 估值是噪声。这条以前根本没推断过（块数恒默认 3）。
                print(f"[GoAI] 权重 value 残差块={inferred_arch['value_res_blocks']}"
                      f"（默认 {value_res_blocks}），已自动调整")
                net_kwargs["value_res_blocks"] = inferred_arch["value_res_blocks"]
                needs_rebuild = True
            if inferred_arch.get("policy_layers") and \
                    inferred_arch["policy_layers"] != policy_layers:
                print(f"[GoAI] 权重 policy_layers={inferred_arch['policy_layers']}"
                      f"（默认 {policy_layers}），已自动调整")
                net_kwargs["policy_layers"] = inferred_arch["policy_layers"]
                needs_rebuild = True
            if needs_rebuild:
                # 改写的是**初始建网那份 net_kwargs**，不是重抄一遍 12 个架构参数：
                # 构建器签名是 (**arch_kwargs)，手抄版漏一个键不会报错，只会静默
                # 落回构建器自己的默认值（compile 回退漏 policy_layers 踩过一次）。
                # action_size 跟 board_size 联动，这里也一并按调整后的值重算。
                net_kwargs["action_size"] = self.board_size * self.board_size + 1
                self.model = _build_for_in_channels(
                    self.in_channels, **net_kwargs).to(self.device)
                if self.channels_last:
                    self.model = self.model.to(memory_format=torch.channels_last)
            incompatible = self.model.load_state_dict(state, strict=False)
            self._verify_loaded_state(incompatible, state)
            # needs_rebuild 的重建发生在 init 早期那次 eval() **之后**：新模型默认
            # training=True（BN 走 batch 统计、dropout 开），不补 eval 会让重建路径
            # 的输出依赖 batch 组成与 RNG（改前的真实 bug）。这里统一收口。
            self.model.eval()
            print(f"[GoAI] 已加载模型: {model_path}  ({device})")

        # NPU 首次前向会触发 CANN 算子初始化（可能 1-3 分钟且无输出），
        # 这里主动 warmup 并打印进度，避免被误认为卡死。
        if self.is_npu and model_path:
            print("[GoAI] NPU 首次前向 warmup 中（CANN 算子初始化，可能需要 1-3 分钟）…",
                  flush=True)
            t0 = time.time()
            # 预热 predict_batch 真实路径（含 autocast + batch 归桶补零）：
            # 让每个桶形状的 CANN 图在启动时一次性编译并落盘到 ASCEND_CACHE_PATH，
            # 避免自对弈每个新进程 / 新 batch 形状都冷编译 ~100s 而误判卡死。
            board = GoBoard(self.board_size)
            my_hist = [[-1, -1, -3], [-1, -1, -3]]
            # 只预热自对弈实际会命中的桶形状（expand-chunk=16 → B≤16）。
            # 32..256 自对弈用不到，留到运行时按需冷编译一次（已落盘缓存，安全）。
            max_warmup_batch = 16
            for nb in [b for b in self._NPU_BATCH_BUCKETS if b <= max_warmup_batch]:
                states = [(board, list(my_hist[0]), list(my_hist[1]), 1)] * nb
                with torch.inference_mode():
                    self.predict_batch(states)
                print(f"  [warmup] batch={nb} 编译完成 ({time.time() - t0:.1f}s)",
                      flush=True)
            print(f"[GoAI] NPU warmup 完成: {time.time() - t0:.1f}s（仅首次，后续为毫秒级）",
                  flush=True)

    def _infer_board_size(self, state):
        """从 policy 头输出层权重形状推断训练棋盘大小（输出维 = n²+1）。"""
        import math
        best = None
        for k, v in state.items():
            if "policy" in k and isinstance(v, torch.Tensor) and v.dim() == 2:
                n1 = v.shape[0]
                n = int(round(math.sqrt(max(n1 - 1, 0))))
                if n >= 5 and n * n + 1 == n1:
                    best = n
        return best

    def _infer_architecture(self, state):
        """从 state_dict 推断 backbone_channels / res_blocks / attention_layers / heads。"""
        info = {}
        # backbone_channels: backbone.conv1.weight shape = [C, in_ch, 3, 3]
        for k, v in state.items():
            if k == "backbone.conv1.weight" and isinstance(v, torch.Tensor):
                info["backbone_channels"] = v.shape[0]
                break
        # res_blocks: backbone.blocks.{i} 存在的最大 i + 1
        block_ids = set()
        for k in state:
            if k.startswith("backbone.blocks.") and ".conv" in k:
                try:
                    block_ids.add(int(k.split(".")[2]))
                except (ValueError, IndexError):
                    pass
        if block_ids:
            info["backbone_res_blocks"] = max(block_ids) + 1
        # attention_layers: backbone.blocks.{i}.attn.qkv.weight 存在的数量
        attn_count = sum(1 for k in state if k.endswith("attn.qkv.weight"))
        if attn_count:
            info["num_attention_layers"] = attn_count
        # num_heads: backbone.blocks.{i}.attn.qkv.weight shape = [3*C, C] → heads 由 C 推断
        # policy_channels: policy.conv1.weight shape = [P, C, 1, 1]
        for k, v in state.items():
            if k == "policy.conv1.weight" and isinstance(v, torch.Tensor):
                info["policy_channels"] = v.shape[0]
                break
        # value_channels / value_res_blocks: value 头形状（两代命名都认，见
        # infer_value_head）。旧实现查的 `value.value_head.0.weight` 两代都不存在，
        # 等于没查 ⇒ value_channels 恒默认、块数恒默认 3。
        info.update(infer_value_head(state))
        # policy_layers: 检查是否有 conv3 和 bn2
        has_conv3 = any(k == "policy.conv3.weight" for k in state)
        has_bn2 = any(k == "policy.bn2.weight" for k in state)
        if has_conv3 and has_bn2:
            info["policy_layers"] = 3
        elif not has_conv3 and not has_bn2:
            info["policy_layers"] = 2
        return info

    @staticmethod
    def _infer_in_channels(state):
        """从权重 stem 卷积的形状推断输入通道数（现役 12）。

        判定依据：stem 是唯一消费输入特征平面的层，读它的 in 维（张量 shape[1]）；
        policy/value 头的第一层吃的是 backbone 输出通道，与输入通道数无关，
        不能用来判通道数。键按 `_STEM_WEIGHT_KEYS` 优先级回退。
        失败模式：
        - 找不到任何可识别的 stem 键 → 返回 None，**调用方必须报错**（P4.8 fix
          轮改的：原为「回退旧默认 12 + 告警」，那条路会拿随机 12ch 模型去装
          一份陌生架构的权重并静默开跑）；
        - stem 通道数越出 12..17（feature_planes 的白名单）→ ValueError：
          特征端给不出对应通道，早失败优于前向时的形状错或静默错答案。
        """
        ic = _stem_in_channels(state)
        if ic is None:
            return None
        if not (12 <= ic <= 17):
            raise ValueError(
                f"权重 stem 的 in_channels={ic} 超出支持范围 12..17"
                f"（feature_planes 通道白名单），无法加载")
        return ic

    def _verify_loaded_state(self, incompatible, state):
        """`strict=False` 只保证「不抛异常」，不保证「权重都进去了」。

        拦一条：**模型自己的 stem 键不在权重里**（incompatible.missing_keys 命中
        `_STEM_WEIGHT_KEYS`）。那意味着输入投影层停在随机初始化 —— 正是本任务
        要消灭的「随机初始化、形状却完全正常」那条静默路：checkpoint 的 stem 若
        叫别的名字（例如 `backbone.patch_embed.weight`），推断层仍可能读到
        另一个候选键而给出正确 in_channels，缺口正好落在 stem 上。

        不拦的部分（只告警）：非 stem 的缺键，例如 checkpoint 里缺了某个
        value/policy 层的张量。**这类缺口一律打印出来**（哪些键、缺几个），
        不能靠 strict=False 静默咽下。

        注：旧版 value 头命名（`value.res1/res2/...`，`e2d0b57` 之前）过去
        长期落在这一类里 —— 本仓库现役 `models/sft_19x19_v12.pth` 实测带 36 个
        `value.res_blocks.*` missing + 24 个 `value.res1.*` unexpected，也就是
        「模型跑得动、value 头却有一整块是随机初始化」。现已由
        `remap_legacy_value_keys` + `infer_value_head` 修掉（加载前归一键名、
        按权重推断块数与通道），不再依赖这条告警兜底。
        """
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        model_state = self.model.state_dict()
        stem_missing = [k for k in _STEM_WEIGHT_KEYS
                        if k in model_state and k in missing]
        if stem_missing:
            shape = tuple(model_state[stem_missing[0]].shape)
            raise RuntimeError(
                f"权重里没有模型 stem 的参数 {stem_missing[0]}"
                f"（形状 {shape}，按 in_channels={self.in_channels} 建出来的输入"
                f"投影层）—— 继续加载的话这一层是随机初始化的：形状对、脑子乱，"
                f"且不报错。checkpoint 顶层键样例 {_state_key_sample(state)}，"
                f"多余的键 {unexpected[:6]}；GoAI 只认 stem 键 "
                f"{', '.join(_STEM_WEIGHT_KEYS)}，请确认 checkpoint 的 stem 命名"
                f"与模型一致（或用 register_in_channels_builder 换成对应结构）。")
        if missing or unexpected:
            print(f"[GoAI] 警告：权重与模型未完全对齐（缺 {len(missing)} 个 / 多 "
                  f"{len(unexpected)} 个键），这些层保持随机初始化："
                  f"missing[:3]={missing[:3]} unexpected[:3]={unexpected[:3]}")

    @property
    def in_channels(self):
        """特征平面通道数（**只读**）：由 checkpoint 的 stem 形状推断，无 checkpoint 时 12。

        做成 property 而不是普通属性，是为了掐死 `ai.in_channels = 17` 这个
        最顺手的「绕过 mcts.py 里还钉着 12 的预取路径」的写法：它会造出
        12ch 模型吃 17ch 特征（或反之）的错配，而这种错配在本设计里**不报错**
        —— 恰恰是本设计要消灭的那一类。要改通道数只能换 checkpoint。
        """
        return self._in_channels

    # ------------------------------------------------------------------ #
    # 特征构造
    # ------------------------------------------------------------------ #
    def _build_state(self, board, my_hist, op_hist, to_play, planes=None):
        """用统一的 feature_planes 构造与模型通道数一致的状态张量。

        planes: 可选，预先算好的 (in_channels,H,W) np.ndarray。传入可避免重复
        feature_planes 计算（MCTS 增量特征场景）。
        通道数由 `_infer_in_channels` 从权重形状推断的 `self.in_channels` 驱动
        （P4.3 临时钉的 `n_channels=12` 已改推断驱动，P4.8/P4.13）。
        """
        if planes is None:
            planes = board.feature_planes_batched(
                board.board[None], [list(my_hist)], [list(op_hist)],
                [to_play], [board.ko_point], n_channels=self.in_channels)[0]
        elif planes.shape[0] != self.in_channels:
            raise RuntimeError(
                f"预计算特征通道数不匹配：期望 {self.in_channels}"
                f"（模型 in_channels），实际 {planes.shape[0]}")
        x = torch.from_numpy(np.ascontiguousarray(planes)).unsqueeze(0).to(self.device).float()
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        return x

    # ------------------------------------------------------------------ #
    # 核心：模型前向 + 采样
    # ------------------------------------------------------------------ #
    def predict(self, board, my_hist, op_hist, to_play):
        """单局面前向。返回 (policy_np, value)。

        policy_np: shape=(bs*bs+1,) 概率（已 softmax）
        value    : float, 当前 to_play 视角 [-1,1]
        """
        x = self._build_state(board, my_hist, op_hist, to_play)
        policy, value = self._forward_batch(x)
        return policy[0], float(value[0].item())

    def predict_batch(self, states):
        """批量前向，MCTS 叶子评估的核心加速点。

        Args:
            states: list[(board, my_hist, op_hist, to_play)] 或
                    list[(None, my_hist, op_hist, to_play, planes)]（带预计算特征）
                    长度 B
        Returns:
            policies: np.ndarray (B, bs*bs+1) 已 softmax
            values : np.ndarray (B,) 当前 to_play 视角 [-1,1]
        """
        if not states:
            return np.zeros((0, self.board_size * self.board_size + 1)), np.zeros(0)
        planes_list = []
        for st in states:
            if len(st) == 5:
                b, mh, oh, tp, planes = st
            else:
                b, mh, oh, tp = st
                planes = None
            if planes is None:
                planes = b.feature_planes_batched(
                    b.board[None], [list(mh)], [list(oh)], [tp], [b.ko_point],
                    n_channels=self.in_channels)[0]
            elif planes.shape[0] != self.in_channels:
                # 失败模式 F5：MCTS 5 元组路径喂进来的预计算特征通道数与模型不符。
                # 17ch 模型配现行 MCTS（那边还钉着 n_channels=12）就会落在这里。
                #
                # `src/search/mcts.py:905` 那个预取 worker 用
                # `except Exception: prefetch_leaf.prefetch = None` 吞掉本异常
                # —— **不要**把它「修」成只捕获特定异常或去掉：那层 except 存在的
                # 理由是预取失败必须退回同步评估（叶子随后会在 evaluate 路径上重新
                # 算一遍，异常在那里照常逃逸，正确性不漏）。它只是让用户看不到
                # 这条诊断。真正的修法是把 mcts.py 那几处 `n_channels=12` 换成
                # `self.ai.in_channels`（GoAI 已把它做成只读属性），而不是放宽这里。
                raise RuntimeError(
                    f"预计算特征通道数不匹配：期望 {self.in_channels}"
                    f"（模型 in_channels），实际 {planes.shape[0]}")
            planes_list.append(np.ascontiguousarray(planes, dtype=np.float32))
        # ONNX 快速路径：全程 numpy，避免 numpy→torch→numpy 往返
        if self._ort is not None:
            xnp = np.stack(planes_list, axis=0)  # (B,in_channels,H,W)
            policies, values = self._forward_batch_onnx_numpy(xnp)
            return policies, values
        x = torch.from_numpy(np.stack(planes_list, axis=0))
        x = x.to(self.device, non_blocking=True)
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        policies, values = self._forward_batch(x)
        return policies, values.squeeze(-1).cpu().numpy().astype(np.float32)

    def _forward_batch_onnx_numpy(self, xnp):
        """ONNX 快速路径：直接接受 numpy 数组，跳过 torch 转换。"""
        if self._ort_dynamic_batch:
            pol, val = self._ort.run(None, {"x": xnp})
        else:
            batch_size = min(8, xnp.shape[0])
            all_pol, all_val = [], []
            for i in range(0, xnp.shape[0], batch_size):
                chunk = xnp[i:i+batch_size]
                p, v = self._ort.run(None, {"x": chunk})
                all_pol.append(np.asarray(p, dtype=np.float32))
                all_val.append(np.asarray(v, dtype=np.float32))
            pol = np.concatenate(all_pol, axis=0)
            val = np.concatenate(all_val, axis=0)
        pol = np.asarray(pol, dtype=np.float32)
        val = np.asarray(val, dtype=np.float32)
        # softmax
        pol -= pol.max(axis=-1, keepdims=True)
        np.exp(pol, out=pol)
        pol /= pol.sum(axis=-1, keepdims=True)
        return pol, val.reshape(-1)

    def _forward_batch(self, x):
        """输入 (B,in_channels,H,W)，输出 (policies_np(B,A), values(B,1))。"""
        B = x.shape[0]
        if self.is_npu and B > 1:
            # batch 归桶补零：稳定算子形状，命中 CANN 编译缓存
            bucket = next((b for b in self._NPU_BATCH_BUCKETS if b >= B), None)
            if bucket is not None and bucket > B:
                x = torch.cat([x, x.new_zeros(bucket - B, *x.shape[1:])], 0)
        with torch.inference_mode():
            if self.use_amp:
                if self.is_npu:
                    # 兼容老 torch_npu：新 torch.autocast("npu") API 不可用时
                    # 回退 torch.npu.amp.autocast()
                    try:
                        with torch.autocast(device_type="npu", dtype=torch.float16):
                            policy_logits, value = self.model(x)
                    except (RuntimeError, AttributeError, TypeError):
                        with torch.npu.amp.autocast():
                            policy_logits, value = self.model(x)
                else:
                    with torch.cuda.amp.autocast():
                        policy_logits, value = self.model(x)
            else:
                policy_logits, value = self.model(x)
        policies = torch.softmax(policy_logits[:B], dim=-1).cpu().numpy()
        return policies, value[:B]

    # ------------------------------------------------------------------ #
    # CPU 推理加速：int8 动态量化 / ONNX Runtime 后端
    # ------------------------------------------------------------------ #
    def quantize_dynamic(self):
        """CPU int8 动态量化（Linear + Conv2d 层，x86 VNNI 收益明显）。

        须在 torch.compile 之前调用（量化编译后模型无意义）。失败时保持 fp32。
        """
        try:
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
            self.model.eval()
            print("[GoAI] 已启用 CPU int8 动态量化 (Linear + Conv2d)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[GoAI] 动态量化失败，保持 fp32: {e}")
            return False

    def quantize_int4_torchao(self):
        """GPU weight-only INT4 量化（torchao，需 pip install torchao）。

        对 CUDA 推理有效，模型体积 ~1/8，显存带宽受限场景提速 1.5-3x。
        须在 torch.compile 之前调用。失败时保持原始精度。
        """
        try:
            from torchao.quantization import quantize_, int4_weight_only
            quantize_(self.model, int4_weight_only())
            self.model.eval()
            print("[GoAI] 已启用 GPU weight-only INT4 量化 (torchao)")
            return True
        except ImportError:
            print("[GoAI] torchao 未安装（pip install torchao），跳过 INT4 量化")
            return False
        except Exception as e:  # noqa: BLE001
            print(f"[GoAI] INT4 量化失败，保持原始精度: {e}")
            return False

    def export_onnx(self, onnx_path, ort_intra_threads=None, quantize_int8=False):
        """导出 ONNX 并把推理后端切换到 onnxruntime（CPU 提速约 1.5-3x）。

        需 `pip install onnx onnxruntime`。失败时保持 torch 后端并返回 False。
        仅支持 CPU 推理（use_amp 自动失效）。

        ort_intra_threads: 每个 ort.run 的内部线程数。None 时取满物理核；
            MCTS 多线程并发调用时建议传入 max(1, ncpu // num_threads) 避免超线程争抢。
        quantize_int8: True 时导出后应用 int8 动态量化（模型体积 ~1/4，CPU 推理 ~2x）。
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("[GoAI] 未安装 onnx/onnxruntime（pip install onnx onnxruntime），保持 torch 后端")
            return False
        try:
            self.model = self.model.cpu().eval()
            dummy = torch.zeros(1, self.in_channels, self.board_size,
                                self.board_size)
            if quantize_int8:
                # int8 量化必须用 legacy exporter（dynamo=False）：
                # 新 torch.export-based 导出器产生的图结构导致 onnxruntime
                # quantizer 的 ShapeInference 失败（32 vs 361）。
                # 固定 batch=1 + opset=17，量化后加载到 ORT。
                quant_path = onnx_path.replace(".onnx", "_static.onnx")
                torch.onnx.export(
                    self.model, dummy, quant_path,
                    input_names=["x"], output_names=["policy", "value"],
                    opset_version=17, dynamo=False)
                quantized = False
                try:
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    quantize_dynamic(
                        model_input=quant_path,
                        model_output=onnx_path,
                        weight_type=QuantType.QInt8)
                    quantized = True
                    print(f"[GoAI] 已应用 ONNX int8 动态量化: {onnx_path}")
                except Exception as e:  # noqa: BLE001
                    print(f"[GoAI] int8 量化失败，使用 FP32 模型: {e}")
                    # 量化失败：用静态 FP32 模型代替
                    os.replace(quant_path, onnx_path)
                # 清理残留静态模型文件
                if os.path.exists(quant_path):
                    try:
                        os.remove(quant_path)
                    except OSError:
                        pass
            else:
                torch.onnx.export(
                    self.model, dummy, onnx_path,
                    input_names=["x"], output_names=["policy", "value"],
                    dynamic_axes={"x": {0: "batch"},
                                  "policy": {0: "batch"},
                                  "value": {0: "batch"}},
                    opset_version=18)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            _ncpu = max(1, (os.cpu_count() or 1))
            so.intra_op_num_threads = ort_intra_threads or _ncpu
            so.inter_op_num_threads = 1
            so.enable_mem_pattern = True
            # 启用 CPU 内存优化：减少峰值内存占用
            so.enable_cpu_mem_arena = True
            self._ort = ort.InferenceSession(
                onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
            # 检测 ONNX 模型是否支持动态 batch（一次性检测，缓存结果）
            self._ort_dynamic_batch = False
            try:
                inp = self._ort.get_inputs()[0]
                dims = inp.shape
                if len(dims) > 0 and isinstance(dims[0], str):
                    self._ort_dynamic_batch = True
            except Exception:  # noqa: BLE001
                pass
            print(f"[GoAI] 已切换 ONNX Runtime 推理后端: {onnx_path} "
                  f"(intra_threads={so.intra_op_num_threads}, "
                  f"dynamic_batch={self._ort_dynamic_batch})")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[GoAI] ONNX 导出失败，保持 torch 后端: {e}")
            return False

    def choose_move(self, board, my_hist, op_hist, to_play, legal_mask, temperature=1.0, topk=10):
        """根据策略分布与合法着法掩码，采样一个着法。

        返回 (move_int, is_pass, value)，move_int 为 board_size*board_size 表示虚着。
        """
        policy, value = self.predict(board, my_hist, op_hist, to_play)
        bs = self.board_size
        n_actions = bs * bs + 1

        illegal = np.ones(n_actions, dtype=bool)
        illegal[:len(legal_mask)] = ~legal_mask
        # 始终允许虚着
        illegal[n_actions - 1] = False
        masked = policy.copy()
        masked[illegal] = 0.0
        s = masked.sum()
        if s <= 0:
            return n_actions - 1, True, value

        if temperature <= 0:
            # 贪心
            move_int = int(np.argmax(masked))
        else:
            probs = masked / s
            # 可选 top-k 截断，降低随机性
            if topk and topk < len(probs):
                idx = np.argsort(probs)[::-1][:topk]
                p2 = np.zeros_like(probs)
                p2[idx] = probs[idx]
                probs = p2 / p2.sum()
            # 温度缩放：对 log 缩放后重新 softmax
            if temperature != 1.0:
                logp = np.log(probs + 1e-12)
                probs = np.exp(logp / max(temperature, 1e-3))
                probs = probs / probs.sum()
            move_int = int(np.random.choice(len(probs), p=probs))

        is_pass = (move_int == n_actions - 1)
        return move_int, is_pass, value

    def choose_move_mcts(self, board, my_hist, op_hist, to_play, legal_mask,
                         simulations=400, temperature=1.0, mcts=None, num_threads=4,
                         use_rollout=False, rollout_lambda=0.25, path_moves=None):
        """用 MCTS 搜索选着法（推理提速核心）。返回 (move_int, is_pass, value)。

        use_rollout / rollout_lambda 启用 LightPLS：叶子价值融合轻量 rollout。
        path_moves: 从上次 MCTS 搜索根局面到当前的着法序列（GoBoard 编码，
            pass=-1），提供时复用上次搜索树的子树（访问/价值统计继承）。
        """
        from src.search.mcts import MCTS
        if mcts is None:
            mcts = MCTS(self, board_size=self.board_size, num_threads=num_threads,
                        temperature=temperature, use_rollout=use_rollout,
                        rollout_lambda=rollout_lambda)
        move_int, is_pass, value = mcts.best_move(
            board, my_hist, op_hist, to_play, simulations=simulations,
            temperature=temperature, return_value=True, path_moves=path_moves)
        # 若 MCTS 选了非法着法（极端情况下），回退到纯策略
        if not legal_mask[move_int] and move_int != self.board_size * self.board_size:
            return self.choose_move(board, my_hist, op_hist, to_play, legal_mask,
                                    temperature=temperature)
        return move_int, is_pass, value

    # ------------------------------------------------------------------ #
    # 对局循环
    # ------------------------------------------------------------------ #
    def _move_int_to_coord(self, move_int):
        bs = self.board_size
        if move_int == bs * bs:
            return None  # pass
        r, c = divmod(move_int, bs)
        return (r, c)

    def self_play(self, num_games=1, max_moves=400, temperature=1.0, topk=10,
                   verbose=False, use_mcts=False, simulations=400, num_threads=4,
                   use_rollout=False, rollout_lambda=0.25):
        """模型自我对弈 num_games 局，返回每局结果（黑方视角 +1/-1）。

        use_mcts=True 时每步用 MCTS 搜索选点（棋力显著强于纯策略 argmax）。
        use_rollout=True 时叶子价值融合轻量 rollout（LightPLS）。
        """
        from src.search.mcts import MCTS
        mcts = MCTS(self, board_size=self.board_size, num_threads=num_threads,
                    temperature=temperature, use_rollout=use_rollout,
                    rollout_lambda=rollout_lambda) if use_mcts else None
        results = []
        for g in range(num_games):
            board = GoBoard(self.board_size)
            my_hist = [[-1, -1, -3], [-1, -1, -3]]  # 对当前执子方：最近3手
            passes = 0
            move_count = 0
            path_moves = []  # 自上次 MCTS 搜索根以来的着法序列（树复用），新局重置
            while passes < 2 and move_count < max_moves:
                to_play = board.current_player
                legal = board.get_legal_moves()
                if not legal.any():
                    passes += 1
                    board.play(-1)
                    path_moves.append(-1)
                    move_count += 1
                    continue
                if use_mcts:
                    if to_play == 1:
                        move_int, is_pass, value = self.choose_move_mcts(
                            board, my_hist[0], my_hist[1], to_play, legal,
                            simulations=simulations, temperature=temperature, mcts=mcts,
                            num_threads=num_threads, use_rollout=use_rollout,
                            rollout_lambda=rollout_lambda, path_moves=path_moves)
                    else:
                        move_int, is_pass, value = self.choose_move_mcts(
                            board, my_hist[1], my_hist[0], to_play, legal,
                            simulations=simulations, temperature=temperature, mcts=mcts,
                            num_threads=num_threads, use_rollout=use_rollout,
                            rollout_lambda=rollout_lambda, path_moves=path_moves)
                else:
                    if to_play == 1:
                        move_int, is_pass, value = self.choose_move(
                            board, my_hist[0], my_hist[1], to_play, legal,
                            temperature=temperature, topk=topk)
                    else:
                        move_int, is_pass, value = self.choose_move(
                            board, my_hist[1], my_hist[0], to_play, legal,
                            temperature=temperature, topk=topk)
                if is_pass:
                    board.play(-1)
                    passes += 1
                    path_moves.append(-1)
                else:
                    r, c = self._move_int_to_coord(move_int)
                    mv_flat = r * self.board_size + c
                    if not board.play(mv_flat):
                        board.play(-1); passes += 1
                        path_moves.append(-1)
                    else:
                        passes = 0
                        path_moves.append(mv_flat)
                        # 更新历史（最近3手，最新在末尾）
                        hist = my_hist[0] if to_play == 1 else my_hist[1]
                        hist.pop(0)
                        hist.append(mv_flat)
                move_count += 1
                if verbose and (move_count % 10 == 0):
                    print(f"  game {g} move {move_count} to_play={to_play} "
                          f"value={value:+.3f} pass={is_pass}")
            score = board.score()
            # 黑方(1)视角
            result = 1.0 if score > 0 else -1.0
            results.append(result)
            if verbose:
                print(f"game {g} finished: score(黑-白)={score:+.1f} -> "
                      f"{'黑胜' if result > 0 else '白胜'}")
        return results

    def play_against_human(self, human_color=1, max_moves=400, temperature=0.6,
                           use_mcts=False, simulations=400, num_threads=4,
                           use_rollout=False, rollout_lambda=0.25):
        """人机对弈，人类通过终端输入坐标（如 'ce' 或 'pass'）。"""
        from src.search.mcts import MCTS
        mcts = MCTS(self, board_size=self.board_size, num_threads=num_threads,
                    temperature=temperature, use_rollout=use_rollout,
                    rollout_lambda=rollout_lambda) if use_mcts else None
        board = GoBoard(self.board_size)
        my_hist = [[-1, -1, -3], [-1, -1, -3]]
        passes = 0
        move_count = 0
        path_moves = []  # 自上次 MCTS 搜索根以来的着法序列（树复用）
        while passes < 2 and move_count < max_moves:
            to_play = board.current_player
            legal = board.get_legal_moves()
            if to_play == human_color:
                print(board.to_string())
                # 掩码是长度恒为 n*n 的 ndarray，所以「合法着法数」必须数真值，不能取
                # len()（那是 n*n）。P2.6a 起掩码里多了禁自杀与 PSK 两类 False，真值数
                # 与空点数不再相等，这个数字现在真的会骗人。
                print(f"合法着法数: {int(legal.sum())}  | 输入坐标(如 ce)，或 pass，或 resign")
                inp = input("你的着法: ").strip().lower()
                if inp in ("pass", ""):
                    board.play(-1)
                    passes += 1
                    path_moves.append(-1)
                elif inp in ("resign", "quit"):
                    print("你认输。")
                    break
                else:
                    ok, mv = board.parse_move_str(inp, human_color)
                    if not ok or not legal[mv]:
                        print("非法着法，请重试。")
                        continue
                    board.play(mv)
                    passes = 0
                    path_moves.append(mv)
                    hist = my_hist[0] if human_color == 1 else my_hist[1]
                    hist.pop(0); hist.append(mv)
            else:
                if not legal.any():
                    board.play(-1); passes += 1
                    path_moves.append(-1)
                else:
                    if use_mcts:
                        if to_play == 1:
                            move_int, is_pass, value = self.choose_move_mcts(
                                board, my_hist[0], my_hist[1], to_play, legal,
                                simulations=simulations, temperature=temperature, mcts=mcts,
                                num_threads=num_threads, use_rollout=use_rollout,
                                rollout_lambda=rollout_lambda, path_moves=path_moves)
                        else:
                            move_int, is_pass, value = self.choose_move_mcts(
                                board, my_hist[1], my_hist[0], to_play, legal,
                                simulations=simulations, temperature=temperature, mcts=mcts,
                                num_threads=num_threads, use_rollout=use_rollout,
                                rollout_lambda=rollout_lambda, path_moves=path_moves)
                    else:
                        if to_play == 1:
                            move_int, is_pass, value = self.choose_move(
                                board, my_hist[0], my_hist[1], to_play, legal,
                                temperature=temperature)
                        else:
                            move_int, is_pass, value = self.choose_move(
                                board, my_hist[1], my_hist[0], to_play, legal,
                                temperature=temperature)
                    if is_pass:
                        board.play(-1); passes += 1
                        path_moves.append(-1)
                        print(f"AI 虚着 (value={value:+.3f})")
                    else:
                        r, c = self._move_int_to_coord(move_int)
                        mv = r * self.board_size + c
                        if not board.play(mv):
                            board.play(-1); passes += 1
                            path_moves.append(-1)
                            print(f"AI 虚着 (value={value:+.3f})")
                        else:
                            passes = 0
                            path_moves.append(mv)
                            hist = my_hist[0] if to_play == 1 else my_hist[1]
                            hist.pop(0); hist.append(mv)
                            print(f"AI 落子 {chr(ord('a')+c)}{chr(ord('a')+r)} (value={value:+.3f})")
            move_count += 1
        print(board.to_string())
        score = board.score()
        print(f"终局 score(黑-白)={score:+.1f} -> {'黑胜' if score > 0 else '白胜'}")

    def analyze(self, max_moves=400, temperature=0.0):
        """交互式分析：从当前局面出发，展示 AI 的 top-k 候选着法。"""
        board = GoBoard(self.board_size)
        my_hist = [[-1, -1, -1], [-1, -1, -1]]
        print("逐步分析（输入坐标落子，pass 虚着，auto 让 AI 走，q 退出）")
        while True:
            to_play = board.current_player
            print(board.to_string())
            legal = board.get_legal_moves()
            policy, value = self.predict(board, my_hist[0], my_hist[1], to_play)
            bs = self.board_size
            ranked = []
            for m in np.where(legal)[0]:
                ranked.append((int(m), float(policy[m])))
            ranked.append((bs * bs, float(policy[bs * bs])))  # pass
            ranked.sort(key=lambda t: t[1], reverse=True)
            print(f"当前 to_play={to_play}  value={value:+.3f}")
            for m, p in ranked[:8]:
                if m == bs * bs:
                    print(f"  pass        p={p:.3f}")
                else:
                    r, c = divmod(m, bs)
                    print(f"  {chr(ord('a')+c)}{chr(ord('a')+r)} (idx {m:3d})  p={p:.3f}")
            inp = input("> ").strip().lower()
            if inp in ("q", "quit"):
                break
            if inp == "auto":
                mv = ranked[0][0]
                if mv == bs * bs:
                    board.play(-1)
                else:
                    board.play(mv)
                    hist = my_hist[0] if to_play == 1 else my_hist[1]
                    hist.pop(0); hist.append(mv)
                continue
            if inp in ("pass", ""):
                board.play(-1); continue
            ok, mv = board.parse_move_str(inp, to_play)
            if not ok or not legal[int(mv)]:
                print("非法，重试。"); continue
            board.play(mv)
            hist = my_hist[0] if to_play == 1 else my_hist[1]
            hist.pop(0); hist.append(mv)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="围棋 SFT 模型推理 / 对弈")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--mode", type=str, default="selfplay",
                        choices=["selfplay", "human", "analyze"])
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--human-color", type=int, default=1)
    parser.add_argument("--attention-mode", default="mix", choices=["none", "mix", "all"])
    parser.add_argument("--num-attention-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--policy-channels", type=int, default=32,
                        help="policy 头隐层通道（须与训练时一致，训练默认 32）")
    parser.add_argument("--value-channels", type=int, default=64,
                        help="value 头隐层通道（须与训练时一致，训练默认 64）")
    parser.add_argument("--attn-mode", default="global",
                        choices=["global", "window", "axial", "sparse", "window_global"],
                        help="注意力计算模式: global=全配对, window=滑动窗口, axial=轴向, "
                             "sparse=窗口+全局token, window_global=块状窗口+全局token")
    parser.add_argument("--attn-window", type=int, default=7, help="window 模式窗口边长")
    parser.add_argument("--compile", action="store_true", help="用 torch.compile 融合算子（GPU 提速）")
    parser.add_argument("--tf32", action="store_true",
                        help="CUDA tf32 matmul（V100/Amp 上约 2-4x fp32 提速，精度损失可忽略）")
    parser.add_argument("--use-mcts", action="store_true",
                        help="用 MCTS 搜索选点（棋力显著强于纯策略 argmax）")
    parser.add_argument("--simulations", type=int, default=400,
                        help="MCTS 每步模拟次数（V100S 上 400~800 仅 1-2s/步）")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="MCTS 并行模拟线程数（虚拟损失）")
    parser.add_argument("--use-rollout", action="store_true",
                        help="LightPLS：叶子价值融合轻量 rollout（Tromp-Taylor 快数子）")
    parser.add_argument("--rollout-lambda", type=float, default=0.25,
                        help="LightPLS rollout 价值权重 (0=只用网络, 1=只用 rollout)")
    args = parser.parse_args()

    ai = GoAI(model_path=args.model, board_size=args.board_size,
              device=args.device, use_amp=args.use_amp,
              attention_mode=args.attention_mode,
              num_attention_layers=args.num_attention_layers,
              num_heads=args.num_heads,
              attention_dropout=args.attention_dropout,
              policy_channels=args.policy_channels,
              value_channels=args.value_channels,
              attn_mode=args.attn_mode,
              attn_window=args.attn_window,
              compile=args.compile, tf32=args.tf32)
    if args.mode == "selfplay":
        res = ai.self_play(num_games=args.games, temperature=args.temperature, topk=args.topk,
                           use_mcts=args.use_mcts, simulations=args.simulations,
                           num_threads=args.num_threads,
                           use_rollout=args.use_rollout, rollout_lambda=args.rollout_lambda)
        wr = sum(1 for r in res if r > 0) / max(len(res), 1)
        tag = " (MCTS"
        if args.use_rollout:
            tag += "+LightPLS"
        tag += ")"
        print(f"自对弈 {len(res)} 局，黑方胜率 {wr:.2%}"
              f"{tag if args.use_mcts else ''}")
    elif args.mode == "human":
        ai.play_against_human(human_color=args.human_color, temperature=args.temperature,
                              use_mcts=args.use_mcts, simulations=args.simulations,
                              num_threads=args.num_threads,
                              use_rollout=args.use_rollout, rollout_lambda=args.rollout_lambda)
    elif args.mode == "analyze":
        ai.analyze(temperature=args.temperature)


if __name__ == "__main__":
    main()
