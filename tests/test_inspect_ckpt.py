"""inspect_ckpt.py 的测试：从 state_dict 反推架构。

背景：`save_model()` 只保存 `model.state_dict()`，**不保存任何配置**。
于是决定结构的参数（backbone_channels / res_blocks / value_res_blocks /
policy_layers …）在 checkpoint 里彻底消失，日志丢失后只能靠猜——run.txt 里
「V17 Value: 1.01M (96ch, 11 blocks)」就是这么错的（1.01M 对应 5 blocks）。

本测试用合成 state_dict 验证反推逻辑，不依赖 models/ 下真实权重。
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from src.networks.alphanet import AlphaGoNet  # noqa: E402
from inspect_ckpt import infer_config, remap_legacy_value_keys, verify  # noqa: E402


def _sd(**over):
    cfg = dict(in_channels=12, action_size=362, backbone_channels=192,
               backbone_res_blocks=17, attention_mode='mix',
               num_attention_layers=4, num_heads=4, attention_dropout=0.0,
               attn_mode='window_global', attn_window=5,
               res_blocks=8, convnext_blocks=4, attn_blocks=5,
               value_channels=96, value_res_blocks=8,
               policy_channels=128, policy_layers=3, arch='resnet')
    cfg.update(over)
    return AlphaGoNet(**cfg).state_dict()


def test_infers_v18_current_architecture():
    """当前 v18 配置（分段模式）必须被正确反推。"""
    sd = _sd()
    cfg, kinds = infer_config(sd, board_size=19)
    assert cfg['_mode'] == 'segmented'
    assert cfg['backbone_channels'] == 192
    assert cfg['in_channels'] == 12
    assert cfg['res_blocks'] == 8
    assert cfg['convnext_blocks'] == 4
    assert cfg['attn_blocks'] == 5
    assert cfg['value_channels'] == 96
    assert cfg['value_res_blocks'] == 8
    assert cfg['policy_channels'] == 128
    assert cfg['policy_layers'] == 3
    assert len(kinds) == 17
    assert kinds[:8] == ['res'] * 8
    assert kinds[8:12] == ['convnext'] * 4
    assert kinds[12:] == ['attn'] * 5


def test_infers_mix_mode_architecture():
    """mix 模式（V12 那一类）必须与分段模式区分开。"""
    cfg, kinds = infer_config(_sd(res_blocks=0, convnext_blocks=0,
                                  attn_blocks=0), board_size=19)
    assert cfg['_mode'] == 'mix'
    assert cfg['backbone_res_blocks'] == 17
    assert cfg['num_attention_layers'] == 4
    assert kinds.count('attn') == 4
    assert kinds.count('res') == 13
    # attention 应当穿插在 res 之间，而不是全堆在末尾
    assert kinds[0] == 'attn' and kinds[-1] == 'attn'


def test_policy_layers_discriminator():
    """2 层与 3 层 policy 头必须区分开。

    判别依据是 conv2 的 **第 0 维**：2 层为 Conv2d(P,1,1,1) -> (1,P,1,1)；
    3 层为 Conv2d(P,P,3,3) -> (P,P,3,3)。曾误用第 1 维导致 2 层被判成 3 层。
    """
    cfg2, _ = infer_config(_sd(policy_layers=2), 19)
    assert cfg2['policy_layers'] == 2
    cfg3, _ = infer_config(_sd(policy_layers=3), 19)
    assert cfg3['policy_layers'] == 3


def test_legacy_value_key_remap():
    """旧版 value.res1/res2 键名必须被重映射为 res_blocks.0/1。"""
    sd = _sd(value_res_blocks=2)
    legacy = {}
    for k, v in sd.items():
        legacy[k.replace('value.res_blocks.0.', 'value.res1.')
                .replace('value.res_blocks.1.', 'value.res2.')] = v
    assert any(k.startswith('value.res1.') for k in legacy)
    assert not any(k.startswith('value.res_blocks.') for k in legacy)

    fixed, moved = remap_legacy_value_keys(legacy)
    assert moved > 0
    assert any(k.startswith('value.res_blocks.0.') for k in fixed)
    assert not any(k.startswith('value.res1.') for k in fixed)

    cfg, _ = infer_config(fixed, 19)
    assert cfg['value_res_blocks'] == 2, \
        '重映射后仍须能数出 value 块数（曾因只认旧命名而算成 0）'


def test_remap_is_idempotent_and_noop_on_current_keys():
    """已是当前键名时不应改动；重复调用也不应重复映射。"""
    sd = _sd(value_res_blocks=3)
    again, moved = remap_legacy_value_keys(sd)
    assert moved == 0
    assert set(again) == set(sd)
    assert all(torch.equal(again[k], sd[k]) for k in sd)


def test_verify_detects_exact_match():
    """配置正确时 verify 必须报完全一致。"""
    sd = _sd()
    cfg, _ = infer_config(sd, 19)
    cfg.update(attention_mode='mix', num_attention_layers=4, num_heads=4,
               attn_mode='window_global', attn_window=5)
    ok, n = verify(cfg, sd, verbose=False)
    assert ok, '推断配置应能严格加载'
    # verify 返回 model.parameters() 的总数，**不含** buffers（BN running_mean/var
    # 等），所以既不能与 state_dict 全部张量之和比，也无法从中反推。
    ref = AlphaGoNet(in_channels=12, action_size=362,
                     backbone_channels=192, backbone_res_blocks=17,
                     attention_mode='mix', num_attention_layers=4, num_heads=4,
                     attention_dropout=0.0, attn_mode='window_global',
                     attn_window=5, res_blocks=8, convnext_blocks=4,
                     attn_blocks=5, value_channels=96, value_res_blocks=8,
                     policy_channels=128, policy_layers=3, arch='resnet')
    assert n == sum(p.numel() for p in ref.parameters())


def test_verify_reports_mismatch_without_raising():
    """配置错误时 verify 必须给出可读诊断，而不是抛异常。

    注意 load_state_dict(strict=False) 对**尺寸不匹配**仍会抛 RuntimeError，
    故必须先自行比对形状。
    """
    sd = _sd()
    cfg, _ = infer_config(sd, 19)
    cfg.update(attention_mode='mix', num_attention_layers=4, num_heads=4,
               attn_mode='window_global', attn_window=5)
    cfg['value_res_blocks'] = 3          # 故意改错
    ok, _ = verify(cfg, sd, verbose=False)
    assert not ok, '错误的 value 深度应被检出'


def test_board_size_cannot_be_inferred_is_documented():
    """board_size 不在 state_dict 里——这是必须写明的已知限制。"""
    sd = _sd()
    cfg, _ = infer_config(sd, board_size=19)
    assert cfg['board_size'] == 19
    assert cfg['action_size'] == 362
    # 换 board_size 不会引起任何形状变化，故只能由调用方提供
    cfg9, _ = infer_config(sd, board_size=9)
    assert cfg9['action_size'] == 82
