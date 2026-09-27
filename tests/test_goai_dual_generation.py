"""GoAI 双代加载测试（P4.8 + P4.13）。

六个用例对应 brief §2 的 1..6，外加失败模式守护（7..13）：
  1. test_infer_in_channels_from_weights        — 真 checkpoint（12ch/17ch）形状推断
  2. test_mismatched_in_channels_raises_with_both_numbers — 构建器无视 in_channels → 期望/实际都进报错
  3. test_12ch_path_is_bit_identical            — 12ch 输出与改前关键路径逐位一致（自写参照）
  4. test_feature_plane_channel_count_follows_model — 特征通道数跟模型走（12/17 端到端）
  5. test_17ch_branch_reports_missing_wiring    — 17 分支报错可操作（P4.2 接线指引）
  6. test_no_checkpoint_defaults_to_12          — 无 checkpoint → 旧默认 12（零回归入口）
  7. test_out_of_range_in_channels_raises       — stem 越出 12..17 → 早失败
  8. test_precomputed_planes_channel_mismatch_raises — F5：预计算特征通道数不符 → 期望/实际进报错
  9. test_stemless_checkpoint_raises            — 权重读不出 stem → 报错，不回退 12
 10. test_renamed_stem_raises_instead_of_random_stem — 模型 stem 键缺失 → 报错，不静默随机
 11. test_builder_with_unrecognizable_stem_raises   — 构建器模型没有可识别 stem → 报错
 12. test_partial_load_still_loads_with_warning — 非 stem 的缺键只告警（存量模型不能因此开不起来）
 13. test_in_channels_is_read_only              — in_channels 不可写（掐死「手改」绕路）

webui 构建路径的覆盖在 tests/test_webui_model_loading.py（不再在本文件重复）。
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.go_rules import GoBoard
from src.inference import GoAI, _build_for_in_channels
from src.networks.alphanet import AlphaGoNet

N = 9


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
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
    """P4.2 接线的占位演示：真 v21 结构归 P4.2，这里只证明接缝与通道驱动。"""
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
    """与改前/改后数值取证同一组固定局面（4 个）。"""
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
    """把改前（P4.8 之前）GoAI 的关键路径写出来当参照。

    改前行为 = 显式 in_channels=12 建网 + feature_planes n_channels=12 +
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


# --------------------------------------------------------------------------- #
# 1. 推断
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 2. 不匹配报错带期望/实际
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 3. 12ch 零回归（逐位）
# --------------------------------------------------------------------------- #
def test_12ch_path_is_bit_identical(tmp_path):
    """GoAI 当前输出与改前关键路径参照逐位一致（同权重、同输入）。"""
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


# --------------------------------------------------------------------------- #
# 4. 特征通道数跟模型走
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 5. 17 分支报错可操作
# --------------------------------------------------------------------------- #
def test_17ch_branch_reports_missing_wiring(tmp_path):
    """17ch 未注册时的报错必须说清：谁来接（P4.2）、怎么接（注册函数+签名）。"""
    ck = _make_ckpt(tmp_path, 17)
    with pytest.raises(RuntimeError) as ei:
        GoAI(model_path=ck, board_size=N, device="cpu")
    msg = str(ei.value)
    assert "17" in msg
    assert "P4.2" in msg, f"报错没点名接线任务: {msg}"
    assert "register_in_channels_builder(17" in msg, f"报错没给接线调用方式: {msg}"
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


# --------------------------------------------------------------------------- #
# 6/7. 默认值与越界
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 8. F5：预计算特征通道数不符（此前零覆盖的失败模式）
# --------------------------------------------------------------------------- #
def test_precomputed_planes_channel_mismatch_raises(tmp_path):
    """17ch 模型收到 12ch 预计算特征 → 报错，消息里**期望 17 与实际 12 都在**。

    这是「17ch 模型 + 现行 MCTS（那边还钉着 n_channels=12）」唯一的响亮出口。
    删掉 `predict_batch` 里那个 `planes.shape[0] != self.in_channels` 检查，本用例
    立刻转红（变异实测记录在报告 §Fix 增补）。
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


# --------------------------------------------------------------------------- #
# 9/10/11. 「唯一剩下的静默路」三层封堵
# --------------------------------------------------------------------------- #
def test_stemless_checkpoint_raises_instead_of_defaulting_to_12(tmp_path):
    """权重里读不出 stem → 报错；**不许**回退 12 装出一整个随机模型。

    曾经的路径：F1 打印一句告警 → 建 12ch 模型 → strict=False 把整份 state 报成
    missing 后照单全收 → 玩家对着一个随机初始化的、形状却完全正常的模型下棋。
    这里用 optimizer 打包形态（`{'model_state_dict': ...}`，GoAI 只解包 'model'）
    作样本：它读不出 stem，落到 88 个键全 missing，正是那条静默路。
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

    改前是 `if built is not None and built != in_channels`：键认不出来就**跳过
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
    """非 stem 的缺键**只告警不阻断**：存量模型不能因为这道新门开不起来。

    `models/sft_19x19_v12.pth` 实测带 30 个 `value.res_blocks.*` missing +
    24 个 `value.res1.*` unexpected（旧 value 头命名，改前就这么加载的）。
    把这条钉住，免得后来者把「缺键即错」一刀切下去、把 webui 的存量模型打死。
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

    写成可写属性的后果正是本任务要消灭的那一类错配 —— 12ch 模型吃 17ch 特征，
    而且不报错（只有特征侧的 F5 会响，而那响得毫无上下文）。
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

