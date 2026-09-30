"""旧版 value 头命名（`value.res1/res2/...`）必须能**完整**加载。

背景（根因）：commit `e2d0b57` 把 `ValueNetwork` 的残差块从
`self.res1 / self.res2 / ...` 改成 `self.res_blocks = nn.ModuleList([...])`，
state_dict 键随之从 `value.res1.*` 变成 `value.res_blocks.0.*`。但加载侧的
架构推断（`GoAI._infer_architecture`）只覆盖 backbone / policy，**从不推断
value 头**，于是：

* 建出来的 value 头恒为默认 3 块（`value_res_blocks=3`）—— 而 `e2d0b57` 之前
  存的权重（现役 `models/sft_19x19_v12.pth` 就是那一代）只有 `res1`/`res2`
  **2 块**，`value.res_blocks.2.*` 无权重可载 → 随机初始化；
* 旧的 24 个 `value.res1/res2.*` 张量被 `strict=False` 报成 unexpected 丢弃。

结果就是「模型跑得动、形状全对，但 value 头有一块是随机的」——正是本仓库
反复在消灭的那一类静默错。`scripts/inspect_ckpt.py` 早有
`remap_legacy_value_keys` + value 头推断（纯键名重映射，数学完全等价），
本文件要求加载侧复用同一套口径，避免两处各写一份。

观测口径一律用「权重真的进了模型」（`torch.equal`），不是「没报错」：
`strict=False` 对缺键从来都不报错。
"""
import io
import os
import re
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.inference import GoAI, remap_legacy_value_keys  # noqa: E402
from src.networks.alphanet import AlphaGoNet, build_v21_net  # noqa: E402

N = 19


def _legacy_rename(state):
    """把当前命名的 value 块键改回旧代命名（模拟 e2d0b57 之前存的权重）。"""
    out = {}
    for k, v in state.items():
        m = re.match(r'^value\.res_blocks\.(\d+)\.(.+)$', k)
        out['value.res%d.%s' % (int(m.group(1)) + 1, m.group(2)) if m else k] = v
    return out


def _save(state, path):
    torch.save({'model': state}, str(path))
    return str(path)


@pytest.fixture(autouse=True)
def _restore_registry():
    from src import inference as inf
    saved = dict(inf._IN_CHANNEL_BUILDERS)
    yield
    inf._IN_CHANNEL_BUILDERS.clear()
    inf._IN_CHANNEL_BUILDERS.update(saved)


# --------------------------------------------------------------------------- #
# 1. 纯键名重映射本身
# --------------------------------------------------------------------------- #
def test_remap_is_pure_key_rename():
    """重映射只改键名、张量一个不动，且当前命名原样返回。"""
    torch.manual_seed(7)
    m = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, value_res_blocks=2)
    src_state = m.state_dict()
    legacy = _legacy_rename(src_state)

    out, moved = remap_legacy_value_keys(legacy)
    assert moved == 24, f'2 块 x 12 键 = 24，实际重映射 {moved} 个'
    assert set(out) == set(src_state), '重映射后的键集与当前命名不一致'
    for k, v in src_state.items():
        assert torch.equal(out[k], v), f'{k} 的数值被重映射动了'
    # 当前命名必须是幂等的（0 个被移动，键集不变）
    again, moved2 = remap_legacy_value_keys(src_state)
    assert moved2 == 0 and set(again) == set(src_state)


# --------------------------------------------------------------------------- #
# 2. 端到端：旧命名权重必须零缺键、且 value 权重逐张量相等
# --------------------------------------------------------------------------- #
def test_legacy_named_value_head_loads_completely(tmp_path, capsys):
    torch.manual_seed(11)
    m = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, value_res_blocks=2)
    src_state = m.state_dict()
    ck = _save(_legacy_rename(src_state), tmp_path / 'legacy_value_2blk.pth')

    ai = GoAI(model_path=ck, board_size=N, device='cpu')
    out = capsys.readouterr().out
    loaded = ai.model.state_dict()

    assert len(ai.model.value.res_blocks) == 2, \
        f'value 块数没跟着权重走：{len(ai.model.value.res_blocks)}（源是 2）'
    for k, v in src_state.items():
        assert torch.equal(loaded[k], v), f'{k} 没从 checkpoint 加载进来'
    assert '未完全对齐' not in out, f'旧命名权重仍报未对齐：{out}'
    assert 'res_blocks.2' not in ' '.join(loaded), '多建了一块随机 value 块'


def test_legacy_ckpt_value_output_matches_source_model(tmp_path):
    """数值等价而不只是「键都在」：旧权重加载后的 value 输出必须与源模型逐位相同。"""
    torch.manual_seed(12)
    m = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, value_res_blocks=2)
    m.eval()
    ck = _save(_legacy_rename(m.state_dict()), tmp_path / 'legacy_value_num.pth')

    ai = GoAI(model_path=ck, board_size=N, device='cpu')
    ai.model.eval()
    x = torch.randn(3, 12, N, N)
    with torch.inference_mode():
        _, v_ref = m(x)
        _, v_got = ai.model(x)
    assert torch.allclose(v_ref, v_got, atol=0, rtol=0), \
        f'value 输出不一致（差 {(v_ref - v_got).abs().max().item():.3e}）'


# --------------------------------------------------------------------------- #
# 3. value 头形状推断（当前命名）
# --------------------------------------------------------------------------- #
def test_value_res_blocks_inferred_from_current_naming(tmp_path):
    """当前命名的 2 块权重也必须把块数推断出来（不能恒用默认 3）。"""
    torch.manual_seed(13)
    m = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, value_res_blocks=2)
    ck = _save(m.state_dict(), tmp_path / 'cur_value_2blk.pth')

    ai = GoAI(model_path=ck, board_size=N, device='cpu')
    assert len(ai.model.value.res_blocks) == 2
    for k, v in m.state_dict().items():
        assert torch.equal(ai.model.state_dict()[k], v), f'{k} 没加载进来'


def test_value_channels_inferred(tmp_path):
    """value_channels 也要跟着权重走（`_infer_architecture` 过去查的是一个
    两代都不存在的键 `value.value_head.0.weight`，等于没查）。"""
    torch.manual_seed(14)
    m = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, value_channels=24, value_res_blocks=2)
    ck = _save(m.state_dict(), tmp_path / 'value_ch24.pth')

    ai = GoAI(model_path=ck, board_size=N, device='cpu')
    assert ai.model.value.downsample[0].weight.shape[0] == 24, \
        'value_channels 没从权重推断出来'
    for k, v in m.state_dict().items():
        assert torch.equal(ai.model.state_dict()[k], v), f'{k} 没加载进来'


# --------------------------------------------------------------------------- #
# 4. 构造参数本身必须被采纳（不依赖「有没有触发重建」）
# --------------------------------------------------------------------------- #
def test_value_res_blocks_ctor_arg_is_honored_without_checkpoint():
    """显式传 `value_res_blocks` 必须真的建出那么多块。

    这条不是凑数：`net_kwargs` 里若漏了这个键，只在**没有触发重建**时才暴露
    （重建路径会用推断值把它补回去），而那正是调用方显式指定形状的场景 ——
    漏了会静默落回 `AlphaGoNet` 的默认 3 块。
    """
    ai = GoAI(model_path=None, board_size=N, device='cpu', value_res_blocks=5)
    assert len(ai.model.value.res_blocks) == 5, \
        f'显式 value_res_blocks=5 没生效：{len(ai.model.value.res_blocks)}'


# --------------------------------------------------------------------------- #
# 5. 不得牵连 v21 / 不得回归别的告警
# --------------------------------------------------------------------------- #
def test_v21_path_unaffected_by_legacy_remap(tmp_path, capsys):
    """v21（17ch）不受影响：FCValueHead 无 res 块，remap 必须是 0 命中。"""
    torch.manual_seed(15)
    m = build_v21_net(in_channels=17, action_size=N * N + 1)
    state = m.state_dict()
    assert not any(k.startswith('value.res') for k in state)
    _, moved = remap_legacy_value_keys(state)
    assert moved == 0

    ck = _save(state, tmp_path / 'v21.pth')
    ai = GoAI(model_path=ck, board_size=N, device='cpu')
    out = capsys.readouterr().out
    assert ai.in_channels == 17
    assert '未完全对齐' not in out, out
    for k, v in state.items():
        assert torch.equal(ai.model.state_dict()[k], v), f'{k} 没加载进来'


def test_inspect_ckpt_shares_the_same_remap():
    """`scripts/inspect_ckpt.py` 必须**转发**到同一个函数（单一真相源）。

    两处各写一份的后果：一边改了正则/命名，另一边还在按老口径报告，
    「加载能过但 inspect 说缺键」这种矛盾最难查。所以这里不只比行为，
    还要确认脚本里没有**第二份**重映射实现（源码级）。
    """
    from scripts import inspect_ckpt

    src = io.open(os.path.join(ROOT, 'scripts', 'inspect_ckpt.py'),
                  encoding='utf-8').read()
    body = src.split('def remap_legacy_value_keys', 1)[1].split('\ndef ', 1)[0]
    assert 're.compile' not in body and 're.match' not in body, \
        'inspect_ckpt 里又自己写了一份重映射（应转发到 src.inference）'
    torch.manual_seed(21)
    m = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, value_res_blocks=2)
    legacy = _legacy_rename(m.state_dict())
    out_a, moved_a = remap_legacy_value_keys(legacy)
    out_b, moved_b = inspect_ckpt.remap_legacy_value_keys(legacy)
    assert moved_a == moved_b == 24
    assert set(out_a) == set(out_b)


def test_non_value_missing_keys_still_only_warn(tmp_path, capsys):
    """非 stem、非 value 的缺键仍然只告警（这道门是 P4.8 立的，不能顺手拆）。"""
    torch.manual_seed(16)
    m = AlphaGoNet(in_channels=12, backbone_channels=16, backbone_res_blocks=2,
                   action_size=N * N + 1, value_res_blocks=2)
    state = m.state_dict()
    del state['value.fc.weight']
    ck = _save(state, tmp_path / 'dropped_fc.pth')

    ai = GoAI(model_path=ck, board_size=N, device='cpu')
    out = capsys.readouterr().out
    assert 'value.fc.weight' in out, f'缺键没有告警：{out}'
    assert ai.in_channels == 12