"""KataGo 化重构后**仍然通用**的断言（与 v21 / Mamba / 旧 v21 shell 无关的部分）。

为什么有这个文件
----------------
v21（`V21Backbone` / `V21Net` / `build_v21_net` / `MambaLTI` / `CrossAttnRes`）
整体退役时，有三批断言随之失去被测对象，但它们**本身**与 v21 无关、且守护的是
仍然保留的两代网络（`AlphaGoNet` + `SharedBackbone`）和 GoAI 的双代加载接缝。
那三批断言原籍：

  * `test_goai_dual_generation.py` —— 15 条，只测 `src/inference.py` 的
    `in_channels` 推断 / 构建器注册 / 加载诊断（17 路那一代用**占位构建器**
    演示接缝，并不 import v21）；
  * `test_v21_budget.py` 的两条 —— 12ch 仍归旧 `AlphaGoNet`、采集侧的特征通道数
    必须跟模型走；
  * `test_arch_v21_blocks.py` 的两条 —— D5「既有类一字未改」（`AlphaGoNet` /
    `SharedBackbone` / `ResBlock` / `ConvNeXtBlock` / `AttentionResBlock` /
    `ValueNetwork` / `PolicyNetwork` 的签名与 forward 片段）。

范围与取舍
----------
* **不搬**：任何 import `MambaLTI` / `CrossAttnRes` / `build_v21_net` /
  `_scan_chunk` 的断言 —— 它们钉的是随 v21 一起消失的实现细节。
* **不搬**：`test_v21_shell_script.py`。它整份只管
  `shell/train_sft_npu_4card_v21.sh` 这一个脚本；它那几条「工程不变量」
  （LF 结尾、shebang + `set -euo pipefail`、`bash -n`、flag 必须在
  `train_sft.py` 的 argparse 里有定义）在 `tests/test_run_py_sh.py` 里对**其余
  脚本**仍有一份等价的守护，不因为这个文件消失而丢覆盖面。
* 梯度检查点的**机制**（段划分 / tap / 开关持有者契约）归
  `tests/test_grad_checkpointing.py`，本文件不重复。

全部 CPU、秒级：不加载真实权重、不读真实数据。
"""
import ast
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.game.go_rules import GoBoard  # noqa: E402
from src.inference import GoAI, _build_for_in_channels  # noqa: E402
from src.networks.alphanet import AlphaGoNet  # noqa: E402

N = 9


# =========================================================================== #
# 一、双代加载接缝：in_channels 由权重形状驱动
# =========================================================================== #
def _make_ckpt(tmp_path, in_channels, seed=1234):
    """造一个真的小 checkpoint（state_dict 存在 "model" 键下，与生产同形）。"""
    torch.manual_seed(seed)
    m = AlphaGoNet(in_channels=in_channels, backbone_channels=16,
                   backbone_res_blocks=2, action_size=N * N + 1, policy_layers=2)
    p = str(tmp_path / f"tiny_{in_channels}ch.pth")
    torch.save({"model": m.state_dict()}, p)
    return p


def _load_state(ck):
    state = torch.load(ck, map_location="cpu")
    return state["model"] if isinstance(state, dict) and "model" in state else state


def _register_17_placeholder():
    """注册一个 17ch 占位构建器：**只**用来证明接缝与通道驱动是通的。

    刻意不注册真结构 —— 本文件要判的是「通道数驱动的接线」，与 17 路背后
    装的是哪种网络无关。哪一代网络真的接进这个槽位，由接缝的守护用例负责。
    """
    from src import inference as inf

    def builder(*, in_channels, **arch_kwargs):
        return AlphaGoNet(in_channels=in_channels, **arch_kwargs)

    inf.register_in_channels_builder(17, builder)


@pytest.fixture(autouse=True)
def _restore_registry():
    """注册表是进程级全局：任何用例注册的构建器用完即还原。"""
    from src import inference as inf
    saved = dict(inf._IN_CHANNEL_BUILDERS)
    yield
    inf._IN_CHANNEL_BUILDERS.clear()
    inf._IN_CHANNEL_BUILDERS.update(saved)


def _fixed_states():
    """一组固定局面（4 个），与数值取证用的是同一组。"""
    b1 = GoBoard(N)
    for mv in (20, 30, 40, 50):
        b1.play(mv)
    b2 = GoBoard(N)
    for mv in (0, 10, 20, 1, 11):
        b2.play(mv)
    b3 = GoBoard(N)
    return [
        (b1, [20, 40, -1], [30, 50, -1], 1),
        (b2, [0, 20, 1], [10, 40, -1], -1),
        (b3, [-1, -1, -3], [-1, -1, -3], 1),
        (b1, [50, -1, -1], [40, -1, -1], -1),
    ]


def _reference_forward(ckpt, states, batched=False):
    """把接缝改造前的关键路径写出来当参照。

    旧行为 = 显式 in_channels=12 建网 + feature_planes n_channels=12 +
    手动前向 + softmax —— 每一步都钉死旧常量，任何「推断错通道 / pin 没改」
    都会让它与 GoAI 当前输出分叉。权重全部来自同一 checkpoint（strict 加载），
    与网络结构的随机初始化无关。
    batched=False → 逐局面 (1,C,H,W) 前向（对 predict）；
    batched=True  → 整批 (B,C,H,W) 一次前向（对 predict_batch）。
    两种粒度分别对齐：batch 与单样本的 fp32 舍入不同（~1e-9），不能交叉比较。
    """
    ref = AlphaGoNet(in_channels=12, backbone_channels=16,
                     backbone_res_blocks=2, num_attention_layers=2,
                     action_size=N * N + 1, policy_layers=2)
    ref.load_state_dict(_load_state(ckpt), strict=True)
    ref.eval()
    xs = []
    for (b, mh, oh, tp) in states:
        pl = b.feature_planes_batched(
            b.board[None], [list(mh)], [list(oh)], [tp], [b.ko_point],
            n_channels=12)[0]
        xs.append(np.ascontiguousarray(pl, dtype=np.float32))
    policies, values = [], []
    groups = [np.stack(xs, axis=0)] if batched else [x[None] for x in xs]
    with torch.inference_mode():
        for x in groups:
            logits, value = ref(torch.from_numpy(x))
            policies.append(torch.softmax(logits, dim=-1).numpy())
            values.append(value.squeeze(-1).numpy().astype(np.float32))
    if batched:
        return policies[0], values[0]
    return np.concatenate(policies, axis=0), np.concatenate(values, axis=0)


# ---- 推断 ----------------------------------------------------------------- #
def test_infer_in_channels_from_weights(tmp_path):
    """真 checkpoint 的 stem 形状 → 推断值（12 与 17 各一，不是纯函数单测）。"""
    for ic in (12, 17):
        state = _load_state(_make_ckpt(tmp_path, ic))
        assert GoAI._infer_in_channels(state) == ic, f"{ic}ch 推断错"

    # 端到端（12ch 走内置构建器）
    ai12 = GoAI(model_path=_make_ckpt(tmp_path, 12), board_size=N, device="cpu")
    assert ai12.in_channels == 12

    # 端到端（17ch 走注册接缝；推断值必须驱动 self.in_channels）
    _register_17_placeholder()
    ai17 = GoAI(model_path=_make_ckpt(tmp_path, 17), board_size=N, device="cpu")
    assert ai17.in_channels == 17


def test_12ch_builder_still_the_legacy_net():
    """双代接缝的另一半：12ch 仍归旧 `AlphaGoNet`（旧权重零回归）。"""
    from src import inference as I
    assert I._IN_CHANNEL_BUILDERS.get(12) is I._legacy_alpha_go_net
    m = I._build_for_in_channels(12, action_size=82)
    assert type(m) is AlphaGoNet
    assert m.backbone.conv1.in_channels == 12


def test_mismatched_in_channels_raises_with_both_numbers(tmp_path):
    """构建器无视 in_channels → load 前就拦下，报错同时含期望值与实际值。"""
    from src import inference as inf

    def poisoned(*, in_channels, **arch_kwargs):
        # 坏构建器：无视 in_channels，永远建 12ch
        return AlphaGoNet(in_channels=12, **arch_kwargs)

    inf.register_in_channels_builder(14, poisoned)
    ck = _make_ckpt(tmp_path, 14)
    with pytest.raises(RuntimeError) as ei:
        GoAI(model_path=ck, board_size=N, device="cpu")
    msg = str(ei.value)
    assert "14" in msg, f"报错缺少期望值 14: {msg}"
    assert "12" in msg, f"报错缺少实际值 12: {msg}"
    assert "期望" in msg and "实际" in msg, f"报错未按契约给期望/实际: {msg}"


# ---- 12ch 零回归（逐位） --------------------------------------------------- #
def test_12ch_path_is_bit_identical(tmp_path):
    """GoAI 当前输出与改造前关键路径参照逐位一致（同权重、同输入）。"""
    ck = _make_ckpt(tmp_path, 12)
    ai = GoAI(model_path=ck, board_size=N, device="cpu")
    states = _fixed_states()
    ref_p, ref_v = _reference_forward(ck, states)              # 逐局面 B=1
    ref_bp, ref_bv = _reference_forward(ck, states, batched=True)  # 整批 B=4

    # 单局面路径（predict）
    for i, (b, mh, oh, tp) in enumerate(states):
        pol, val = ai.predict(b, mh, oh, tp)
        np.testing.assert_array_equal(pol, ref_p[i])
        assert val == ref_v[i], f"state{i} value 不逐位一致"

    # 批量路径（predict_batch，B=4 一次前向）
    pol_b, val_b = ai.predict_batch(states)
    np.testing.assert_array_equal(pol_b, ref_bp)
    np.testing.assert_array_equal(val_b, ref_bv)

    # 5 元组预计算特征路径（MCTS 形状）
    pre = []
    for (b, mh, oh, tp) in states:
        pl = b.feature_planes_batched(
            b.board[None], [list(mh)], [list(oh)], [tp], [b.ko_point],
            n_channels=12)[0]
        pre.append((None, list(mh), list(oh), tp, pl))
    pol_pre, val_pre = ai.predict_batch(pre)
    np.testing.assert_array_equal(pol_pre, ref_bp)
    np.testing.assert_array_equal(val_pre, ref_bv)


def test_feature_plane_channel_count_follows_model(tmp_path, monkeypatch):
    """端到端：12ch 权重 → 特征 12 通道；17ch 权重 → 特征 17 通道（不许静默按 12 算）。"""
    seen = []
    orig = GoBoard.feature_planes_batched

    def spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        seen.append((kwargs.get("n_channels"), out.shape[1]))
        return out

    monkeypatch.setattr(GoBoard, "feature_planes_batched", staticmethod(spy))
    board = GoBoard(N)
    board.play(20)

    ai12 = GoAI(model_path=_make_ckpt(tmp_path, 12), board_size=N, device="cpu")
    ai12.predict(board, [-1, -1, -1], [-1, -1, -1], 1)
    assert ai12.in_channels == 12
    assert seen[-1] == (12, 12), f"12ch 路径特征通道数不对: {seen[-1]}"
    ai12.predict_batch([(board, [-1, -1, -1], [-1, -1, -1], 1)])
    assert seen[-1] == (12, 12), f"12ch 批量路径特征通道数不对: {seen[-1]}"

    _register_17_placeholder()
    ai17 = GoAI(model_path=_make_ckpt(tmp_path, 17), board_size=N, device="cpu")
    ai17.predict(board, [-1, -1, -1], [-1, -1, -1], 1)
    assert ai17.in_channels == 17
    assert seen[-1] == (17, 17), f"17ch 路径特征通道数不对: {seen[-1]}"
    ai17.predict_batch([(board, [-1, -1, -1], [-1, -1, -1], 1)])
    assert seen[-1] == (17, 17), f"17ch 批量路径特征通道数不对: {seen[-1]}"


# ---- 没接线的通道：报错必须可操作 ----------------------------------------- #
def test_unregistered_channel_reports_missing_wiring(tmp_path):
    """**没**接线的通道数（16）报错仍须可操作：谁来接、怎么接。

    换一个没人注册的通道数，文案一个字都不能少（否则下一次接线的人又要逆向
    猜契约）。
    """
    ck = _make_ckpt(tmp_path, 16)
    with pytest.raises(RuntimeError) as ei:
        GoAI(model_path=ck, board_size=N, device="cpu")
    msg = str(ei.value)
    assert "16" in msg
    assert "register_in_channels_builder(16" in msg, f"报错没给接线调用方式: {msg}"
    assert "in_channels" in msg and "arch_kwargs" in msg, \
        f"报错没给 builder 契约: {msg}"


def test_missing_wiring_hint_names_the_actual_channel_count(tmp_path):
    """接线指引必须指名**本次请求的**通道数：14ch 权重不许被告知去注册 17。"""
    ck = _make_ckpt(tmp_path, 14)
    with pytest.raises(RuntimeError) as ei:
        GoAI(model_path=ck, board_size=N, device="cpu")
    msg = str(ei.value)
    assert "register_in_channels_builder(14" in msg, \
        f"指引里的通道数不是实际请求的 14: {msg}"
    assert "register_in_channels_builder(17" not in msg, \
        f"指引里仍写死 17，会把接线的人带偏: {msg}"


# ---- 默认值与越界 --------------------------------------------------------- #
def test_no_checkpoint_defaults_to_12(tmp_path):
    """无 checkpoint → 旧默认 12（旧入口零回归的第一道门）。"""
    ai = GoAI(board_size=N, device="cpu")
    assert ai.in_channels == 12
    board = GoBoard(N)
    pol, val = ai.predict(board, [-1, -1, -1], [-1, -1, -1], 1)
    assert pol.shape == (N * N + 1,)


def test_out_of_range_in_channels_raises(tmp_path):
    """stem 通道数越出 feature_planes 白名单 12..17 → 早失败且报错带该数值。"""
    torch.manual_seed(7)
    m = AlphaGoNet(in_channels=11, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, policy_layers=2)
    ck = str(tmp_path / "tiny_11ch.pth")
    torch.save({"model": m.state_dict()}, ck)
    with pytest.raises(ValueError) as ei:
        GoAI(model_path=ck, board_size=N, device="cpu")
    msg = str(ei.value)
    assert "11" in msg and "12..17" in msg, f"越界报错不合格: {msg}"


# ---- 预计算特征通道数不符（此前零覆盖的失败模式） ------------------------- #
def test_precomputed_planes_channel_mismatch_raises(tmp_path):
    """17ch 模型收到 12ch 预计算特征 → 报错，消息里**期望 17 与实际 12 都在**。

    这是「非默认通道的模型 + 现行 MCTS（那边还钉着 n_channels=12）」唯一的
    响亮出口。删掉 `predict_batch` 里那个 `planes.shape[0] != self.in_channels`
    检查，本用例立刻转红。
    """
    _register_17_placeholder()
    ai = GoAI(model_path=_make_ckpt(tmp_path, 17), board_size=N, device="cpu")
    assert ai.in_channels == 17
    board = GoBoard(N)
    board.play(20)
    mh, oh, tp = [20, -1, -1], [-1, -1, -1], 1

    planes12 = board.feature_planes_batched(
        board.board[None], [list(mh)], [list(oh)], [tp], [board.ko_point],
        n_channels=12)[0]
    assert planes12.shape[0] == 12
    with pytest.raises(RuntimeError) as ei:
        ai.predict_batch([(None, mh, oh, tp, planes12)])
    msg = str(ei.value)
    assert "17" in msg, f"报错缺少期望通道数 17: {msg}"
    assert "12" in msg, f"报错缺少实际通道数 12: {msg}"
    assert "期望" in msg and "实际" in msg, f"报错未按契约给期望/实际: {msg}"

    # 对照组：17ch 特征走同一条 5 元组路径必须通（证明门是「通道数」不是「5 元组」）
    planes17 = board.feature_planes_batched(
        board.board[None], [list(mh)], [list(oh)], [tp], [board.ko_point],
        n_channels=17)[0]
    pol, val = ai.predict_batch([(None, mh, oh, tp, planes17)])
    assert pol.shape == (1, N * N + 1) and val.shape == (1,)

    # 同一道门的另一扇（_build_state 的 planes 分支）：今天只有 predict_batch
    # 会传 planes 进来，但门得两边都在，否则先改到的那一边是裸的。
    with pytest.raises(RuntimeError) as ei2:
        ai._build_state(board, mh, oh, tp, planes=planes12)
    assert "17" in str(ei2.value) and "12" in str(ei2.value)
    assert ai._build_state(board, mh, oh, tp, planes=planes17).shape == (
        1, 17, N, N)


# ---- 「唯一剩下的静默路」三层封堵 ----------------------------------------- #
def test_stemless_checkpoint_raises_instead_of_defaulting_to_12(tmp_path):
    """权重里读不出 stem → 报错；**不许**回退 12 装出一整个随机模型。

    曾经的路径：打印一句告警 → 建 12ch 模型 → strict=False 把整份 state 报成
    missing 后照单全收 → 玩家对着一个随机初始化的、形状却完全正常的模型下棋。
    这里用 optimizer 打包形态（`{'model_state_dict': ...}`，GoAI 只解包 'model'）
    作样本：它读不出 stem，落到全部键 missing，正是那条静默路。
    """
    torch.manual_seed(21)
    m = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, policy_layers=2)
    ck = str(tmp_path / "nested.pth")
    torch.save({"model_state_dict": m.state_dict(),
                "optimizer_state_dict": {"x": torch.zeros(1)}}, ck)
    with pytest.raises(RuntimeError) as ei:
        GoAI(model_path=ck, board_size=N, device="cpu", backbone_res_blocks=2)
    msg = str(ei.value)
    # 钉这一层自己的诊断（不回退默认），而不是任何一条「stem 缺键」都能糊弄过去：
    # 即使 `_verify_loaded_state` 那层兜底也拦得住，消息也得说清是「读不出通道数」。
    assert "无法确定 in_channels" in msg, f"报错没说清是通道数读不出来: {msg}"
    assert "拒绝回退" in msg, f"报错没点明拒绝回退默认 12: {msg}"
    assert "12" in msg, f"报错没点出被拒绝的回退值 12: {msg}"
    assert "backbone.conv1.weight" in msg, f"报错没列出 GoAI 认的 stem 键: {msg}"


def test_renamed_stem_raises_instead_of_random_stem(tmp_path):
    """checkpoint 的 stem 挪到回退候选键 → 推断成功但模型 stem 缺键 → 报错。

    构造要点：`_infer_architecture` 读 backbone_channels 用的也是
    `backbone.conv1.weight`，所以要让「推断不出架构」成为**唯一**的救命稻草，
    backbone_channels 必须恰好等于 GoAI 的默认 128（否则先在 size mismatch 上崩，
    那是另一条更早的响亮出口，不测这条）。
    """
    torch.manual_seed(22)
    m = AlphaGoNet(in_channels=12, backbone_channels=128, backbone_res_blocks=1,
                   attention_mode="none", action_size=N * N + 1, policy_layers=2)
    state = {("backbone.stem.weight" if k == "backbone.conv1.weight" else k): v
             for k, v in m.state_dict().items()}
    assert "backbone.stem.weight" in state
    ck = str(tmp_path / "renamed_stem.pth")
    torch.save({"model": state}, ck)
    with pytest.raises(RuntimeError) as ei:
        GoAI(model_path=ck, board_size=N, device="cpu", attention_mode="none")
    msg = str(ei.value)
    assert "backbone.conv1.weight" in msg, f"报错没点名缺失的模型 stem 键: {msg}"
    assert "随机初始化" in msg, f"报错没说清继续加载的后果是随机 stem: {msg}"
    assert "12" in msg, f"报错没给 in_channels 期望值: {msg}"
    assert "backbone.stem.weight" in msg, \
        f"报错没把 checkpoint 实际用的 stem 键摆出来: {msg}"


def test_builder_with_unrecognizable_stem_raises(tmp_path):
    """构建器返回的模型没有可识别 stem → `_build_for_in_channels` 硬报错。

    早先是 `if built is not None and built != in_channels`：键认不出来就**跳过
    校验**放行，等于把「构建器无视 in_channels」这条防线在陌生命名下悄悄关掉。
    """
    from src import inference as inf

    class _NoRecognizableStem(torch.nn.Module):
        def __init__(self, in_channels, **kw):
            super().__init__()
            self.patch_embed = torch.nn.Conv2d(in_channels, 16, 3, padding=1)

    def builder(*, in_channels, **arch_kwargs):
        return _NoRecognizableStem(in_channels, **arch_kwargs)

    inf.register_in_channels_builder(17, builder)
    with pytest.raises(RuntimeError) as ei:
        _build_for_in_channels(17, backbone_channels=16)
    msg = str(ei.value)
    assert "17" in msg, f"报错没给请求的通道数: {msg}"
    assert "backbone.conv1.weight" in msg, f"报错没列出 GoAI 认的 stem 键: {msg}"


def test_partial_load_still_loads_with_warning(tmp_path, capsys):
    """非 stem 的缺键**只告警不阻断**：存量模型不能因为这道门开不起来。

    实测存量权重带 30 个 `value.res_blocks.*` missing + 24 个 `value.res1.*`
    unexpected（旧 value 头命名，之前就这么加载的）。把这条钉住，免得后来者把
    「缺键即错」一刀切下去、把 webui 的存量模型打死。
    """
    ck = _make_ckpt(tmp_path, 12)
    state = _load_state(ck)
    dropped = "value.fc.weight"
    assert dropped in state
    del state[dropped]
    ck2 = str(tmp_path / "tiny_12ch_partial.pth")
    torch.save({"model": state}, ck2)

    ai = GoAI(model_path=ck2, board_size=N, device="cpu", backbone_res_blocks=2)
    assert ai.in_channels == 12
    out = capsys.readouterr().out
    assert dropped in out, f"缺键没有告警，等于继续静默咽下 strict=False: {out}"


def test_rebuild_path_propagates_every_inferred_arch_param(tmp_path):
    """needs_rebuild 重建出来的模型必须带着**全部**推断出来的架构参数。

    观测口径用「权重真的进了模型」而不是「没报错」：policy_layers=3 的权重
    配 policy_layers=2 的模型，conv3 落进 unexpected_keys、bn2 落进 missing_keys，
    strict=False 两边照单全收，模型跑得动但 policy 头是随机初始化的。
    这条同时钉住重建路径与初始建网共用同一份 net_kwargs（少一个键 = 落回
    构建器自己的默认值，且不报错）。
    """
    torch.manual_seed(23)
    m = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, policy_layers=3)
    state = m.state_dict()
    assert "policy.conv3.weight" in state and "policy.bn2.weight" in state
    ck = str(tmp_path / "tiny_12ch_p3.pth")
    torch.save({"model": state}, ck)

    ai = GoAI(model_path=ck, board_size=N, device="cpu")
    loaded = ai.model.state_dict()
    assert "policy.conv3.weight" in loaded, \
        "重建时丢了 policy_layers=3（模型还是 2 层头，policy 是随机的）"
    for k in ("policy.conv3.weight", "policy.bn2.weight", "policy.bn2.bias"):
        assert torch.equal(loaded[k], state[k]), f"{k} 没从 checkpoint 加载进来"
    pol, val = ai.predict(GoBoard(N), [-1, -1, -1], [-1, -1, -1], 1)
    assert pol.shape == (N * N + 1,)


def test_in_channels_is_read_only(tmp_path):
    """`ai.in_channels = 17` 必须失败：它是本设计最顺手的绕路写法。

    写成可写属性的后果正是这一类错配 —— 12ch 模型吃 17ch 特征，而且不报错
    （只有特征侧那道检查会响，而那响得毫无上下文）。
    """
    ai = GoAI(model_path=_make_ckpt(tmp_path, 12), board_size=N, device="cpu")
    assert ai.in_channels == 12
    with pytest.raises(AttributeError):
        ai.in_channels = 17
    assert ai.in_channels == 12
    # 建网之后 / 前向之后都还是只读（不是只在 __init__ 早期被保护）
    ai.predict(GoBoard(N), [-1, -1, -1], [-1, -1, -1], 1)
    with pytest.raises(AttributeError):
        ai.in_channels = 17


# =========================================================================== #
# 二、采集侧的特征通道数必须跟模型走（不许写死字面量）
# =========================================================================== #
def _planes_calls(path):
    """产出每个 `feature_planes_batched(..., n_channels=X)` 的调用点与那个实参。"""
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'feature_planes_batched'):
            for kw in node.keywords:
                if kw.arg == 'n_channels':
                    yield node, kw.value


def test_selfplay_and_async_planes_follow_the_model_not_a_literal():
    """采集通道数必须随 `ai.in_channels` 走，**字面量一律禁止**。

    写死 12 的后果不是「静默错值」而是**形状错**（buffer 17 路 vs 模型 12 路）；
    写死 17 又会打死旧权重的自对弈。两边都兼容的唯一写法就是随模型驱动。

    RL 去 MCTS 之后 selfplay 的 planes 构造搬进了
    `src/search/policy_sampler.py`（与 minimax 推演**共用同一次特征计算**，不再
    算两遍），所以扫描范围是「两个采集脚本 + 走子器」。判据相应放宽一档：
    允许 `n_channels=n_channels` 这种局部变量，但那个变量必须**派生自
    `ai.in_channels`**，字面量仍然一律禁止。
    """
    import re as _re
    total = 0
    for rel in ('scripts/selfplay_train.py', 'scripts/async_pipeline.py',
                'src/search/policy_sampler.py'):
        calls = list(_planes_calls(os.path.join(ROOT, rel)))
        total += len(calls)
        for node, val in calls:
            assert not (isinstance(val, ast.Constant)
                        and isinstance(val.value, int)), \
                f'{rel}:{node.lineno} 的 n_channels 还是字面量 {val.value}'
            if isinstance(val, ast.Attribute):
                assert val.attr == 'in_channels', \
                    f'{rel}:{node.lineno} 的 n_channels 应取 ai.in_channels'
            else:
                assert isinstance(val, ast.Name), \
                    f'{rel}:{node.lineno} 的 n_channels 表达式形态异常'
    assert total > 0, '三个采集落点里一个 feature_planes_batched 调用都没找到'

    # 走子器里的局部变量必须派生自 ai.in_channels（不能用 12/17 兜底成常量）
    src = open(os.path.join(ROOT, 'src', 'search', 'policy_sampler.py'),
               encoding='utf-8').read()
    assert _re.search(r'n_channels\s*=\s*getattr\(\s*ai\s*,\s*[\'"]in_channels[\'"]',
                      src), 'policy_sampler 的 n_channels 必须取自 ai.in_channels'


# =========================================================================== #
# 三、既有网络契约：一个字都不许改
# =========================================================================== #
def _init_sig(path, cls):
    """`class X.__init__` 的参数列表（去掉 self），按源码顺序拼成可读字符串。"""
    tree = ast.parse(open(path, encoding='utf-8').read())
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == cls), None)
    assert node is not None, f'{os.path.basename(path)} 里找不到类 {cls}'
    fn = next((f for f in node.body if isinstance(f, ast.FunctionDef)
               and f.name == '__init__'), None)
    assert fn is not None, f'{cls} 没有 __init__'
    a = fn.args
    assert a.posonlyargs == [] and not a.kwonlyargs and not a.vararg and not a.kwarg, \
        f'{cls} 的 __init__ 出现了新的参数种类（positional-only / kwonly / *args / **kwargs）'
    names = [ast.unparse(x) for x in a.args[1:]]          # 去掉 self
    n_def = len(a.defaults)
    for i, d in enumerate(a.defaults):
        names[len(names) - n_def + i] += '=' + ast.unparse(d)
    return ', '.join(names)


def test_legacy_class_signatures_untouched():
    """既有类的 `__init__` 签名逐字未变（含 `AlphaGoNet` / `SharedBackbone`）。

    这是 KataGo 化重构的**地基**：新结构是替换实现，不是替换接口。旧两代
    （12ch 的 resnet/convnext 路径与共享主干 `SharedBackbone`）的构造签名一旦
    被顺手改过，存量 checkpoint 与存量脚本都会在**形状对不上**的地方崩，
    而崩点离改动点很远。
    """
    expect_backbone = {
        # norm 层也被锁：ConvNeXt 路径依赖 LayerNorm2d、MultiHeadSelfAttention 依赖
        # RMSNorm，改它们的签名同样是违规（mutation 实测：只锁块类时漏掉了）
        'RMSNorm': 'channels, eps=1e-06',
        'LayerNorm2d': 'channels',
        'ResBlock': 'channels',
        'ConvNeXtBlock': 'channels',
        'MultiHeadSelfAttention': (
            "channels, num_heads=4, dropout=0.0, mode='global', window_size=7"),
        'AttentionResBlock': (
            "channels, num_heads=4, dropout=0.0, attention_mode='global', window_size=7"),
        'SharedBackbone': (
            "in_channels=12, channels=128, num_res_blocks=12, attention_mode='mix', "
            "num_attention_layers=4, num_heads=4, attention_dropout=0.0, "
            "attn_mode='global', attn_window=7, use_checkpoint=False, arch='convnext', "
            "res_blocks=0, convnext_blocks=0, attn_blocks=0"),
    }
    for cls, want in expect_backbone.items():
        got = _init_sig(os.path.join(ROOT, 'src', 'networks', 'backbone.py'), cls)
        assert got == want, f'backbone.{cls}.__init__ 签名被改了：\n  实得 {got}\n  期望 {want}'

    assert _init_sig(os.path.join(ROOT, 'src', 'networks', 'policy_network.py'),
                     'PolicyNetwork') == (
        'in_channels=64, hidden_channels=32, action_size=81, num_layers=2')
    assert _init_sig(os.path.join(ROOT, 'src', 'networks', 'value_network.py'),
                     'ValueNetwork') == (
        "in_channels=64, hidden_channels=32, num_res_blocks=3, arch='resnet'")

    # AlphaGoNet.__init__ 零改动
    got = _init_sig(os.path.join(ROOT, 'src', 'networks', 'alphanet.py'),
                    'AlphaGoNet')
    assert got == (
        "in_channels: int=12, backbone_channels: int=128, backbone_res_blocks: int=12, "
        "attention_mode: str='mix', num_attention_layers: int=4, num_heads: int=4, "
        "attention_dropout: float=0.0, attn_mode: str='global', attn_window: int=7, "
        "policy_channels: int=32, value_channels: int=64, action_size: int=362, "
        "use_checkpoint: bool=False, arch: str='resnet', res_blocks: int=0, "
        "convnext_blocks: int=0, attn_blocks: int=0, value_res_blocks: int=3, "
        "policy_layers: int=2"), f'AlphaGoNet.__init__ 签名被改了：{got}'


def test_legacy_forward_untouched():
    """既有类的 forward 源码片段未变（防「顺手重构」掉旧代行为）。"""
    src = open(os.path.join(ROOT, 'src', 'networks', 'backbone.py'),
               encoding='utf-8').read()
    for frag in (
        'out = F.relu(self.bn_out(self.conv_out(out)))',
        'self.qkv = nn.Linear(channels, channels * 3, bias=False)',
        'return self.attn_drop if self.training else 0.0',
    ):
        assert frag in src, f'backbone.py 缺片段 {frag!r} —— 既有行为被改动了'
    vsrc = open(os.path.join(ROOT, 'src', 'networks', 'value_network.py'),
                encoding='utf-8').read()
    assert 'x = self.gap(x).flatten(1)\n        x = self.fc(x)\n        return x' in vsrc, \
        'ValueNetwork.forward 被改了（旧代 value 头必须零回归）'
