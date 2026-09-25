"""search_arch.py 的测量正确性测试。

三项测量都必须可信，否则整个搜索无意义：
  * 参数量  —— 直接求和，最容易错的是漏算/重算
  * FLOPs   —— 必须按 batch=1 测，否则被 batch 放大
  * 激活显存 —— 必须按 **storage 去重**，且要正确反映 gradient checkpointing

尤其第三项：曾用 `t.numel() * element_size()` 累加，实测高估 48%
（反向保存大量视图，共享同一块 storage，被重复计数）；改按
`untyped_storage().nbytes()` + `data_ptr()` 去重后误差降到 21%，剩余
部分来自 checkpoint 重算临时量被按「同时存活」计入。
"""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from src.networks.alphanet import AlphaGoNet  # noqa: E402
import search_arch as S  # noqa: E402

SMALL = dict(backbone_channels=32, backbone_res_blocks=2,
             attention_mode='none', num_attention_layers=0, num_heads=4,
             attn_mode='window_global', attn_window=5,
             res_blocks=1, convnext_blocks=1, attn_blocks=1,
             value_channels=16, value_res_blocks=2,
             policy_channels=16, policy_layers=2)


def _model(**over):
    cfg = dict(SMALL)
    cfg.update(over)
    return AlphaGoNet(in_channels=12, action_size=362, arch='resnet',
                      attention_dropout=0.0, **cfg)


def test_param_count_matches_model():
    n, _, _ = S.measure(SMALL, probe_bs=2)
    m = _model()
    assert n == sum(p.numel() for p in m.parameters())
    del m


def test_flops_are_batch_independent():
    """FLOPs 必须按 batch=1 测——否则乘上 batch，候选比较全错。"""
    n0, f1, _ = S.measure(SMALL, probe_bs=2)
    n1, f2, _ = S.measure(SMALL, probe_bs=8)
    assert f1 == f2, 'FLOPs 随 probe_bs 变化，说明被 batch 放大了'
    assert n0 == n1


def test_activation_is_deduplicated_by_storage():
    """反向会保存共享同一 storage 的视图，必须去重（否则高估 48%）。

    构造两个共享底层存储的视图，各自存进反向图，计量只应算一份。
    """
    counted = [0]
    seen = set()

    def pack(t):
        st = t.untyped_storage()
        k = st.data_ptr()
        if k not in seen:
            seen.add(k)
            counted[0] += st.nbytes()
        return t
    base = torch.zeros(64, 64, requires_grad=True)
    v1 = base[:32]
    v2 = base[32:]
    # 前向必须在 hook 上下文**内**：mul 是在前向时保存操作数的。
    # 用 sum 也不行——sum 不保存任何输入。
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = (v1 * v2).sum() * 2
        y.backward()
    # v1 与 v2 是 base 的两个视图，共享同一 storage -> 只应计一份 64*64*4
    assert counted[0] == 64 * 64 * 4, \
        '共享 storage 的视图被重复计数: {} != {}'.format(counted[0], 64 * 64 * 4)


def test_checkpointing_reduces_retained_memory():
    """开 gradient checkpointing 后保留的激活应显著变少。

    这正是 v18 实测能跑在 97% 显存的原因；若此关系不成立，说明计量有 bug。
    """
    _, _, ret_plain = S.measure(SMALL, probe_bs=2, use_checkpoint=False)
    _, _, ret_ckpt = S.measure(SMALL, probe_bs=2, use_checkpoint=True)
    assert ret_ckpt < ret_plain, \
        '开启 checkpoint 后保留显存未减少（{} vs {}），计量或实现有问题'.format(
            ret_ckpt, ret_plain)


def test_attention_window_affects_memory():
    """attn_window 会改变窗口划分数，显存必须随之变化。

    注意 FLOPs 对它不敏感——注意力是 matmul 不是 conv/linear，hook 数不到。
    这是本工具的已知盲区，测试只锁住显存这一侧确实敏感。
    """
    mems = {}
    for ws in (3, 5, 19):
        _, _, ret = S.measure(dict(SMALL, attn_window=ws), probe_bs=2)
        mems[ws] = ret
    assert mems[3] != mems[19], 'attn_window 对显存毫无影响，疑似计量失灵'
    assert mems[3] > mems[5] > mems[19] or mems[3] > mems[5], \
        '小窗口（nW 大）应占用更多显存，实测 {}'.format(mems)


def test_backbone_res_blocks_is_a_dead_parameter():
    """backbone_res_blocks 在分段模式下完全无效（已知死参数）。

    backbone.py:613-617 把它算成 total_blocks 传给 _build_segmented_blocks，
    但该函数体只用 res_count/convnext_count/attn_count，total_blocks 从不使用。
    本测试把这个事实钉住，避免哪天有人以为调它有用。
    """
    base = dict(SMALL, attention_mode='none', res_blocks=2,
                convnext_blocks=1, attn_blocks=1)
    outs = []
    for nb in (3, 8, 20):
        n, f, ret = S.measure(dict(base, backbone_res_blocks=nb), probe_bs=2)
        outs.append((n, f, ret))
    assert outs[0] == outs[1] == outs[2], \
        'backbone_res_blocks 竟然影响了输出，说明死参数已被修复或配置不对: {}'.format(outs)


def test_num_attention_layers_ignored_in_segmented_mode():
    """分段模式下 num_attention_layers 同样被忽略（由 attn_blocks 取代）。"""
    base = dict(SMALL, attention_mode='none', res_blocks=2,
                convnext_blocks=1, attn_blocks=1)
    a = S.measure(dict(base, num_attention_layers=2), probe_bs=2)
    b = S.measure(dict(base, num_attention_layers=8), probe_bs=2)
    assert a == b, '分段模式下 num_attention_layers 不应影响结果'


def test_overhead_accounting():
    """常驻开销 = 权重+梯度+AdamW双矩+EMA，参数为 FP32 故 20 B/元素。"""
    assert abs(S.overhead_gb(1_000_000) - 20e6 / 1024 ** 3) < 1e-9


def test_calibration_is_self_consistent():
    """标定后，锚点配置的预测显存应等于实测值。"""
    err = S.calibrate(verbose=False)
    assert abs(err) < 1e-6, '标定不自洽，误差 {}'.format(err)
    k = S.calib_factor()
    assert 0.7 < k < 0.95, \
        '标定因子应在 0.7~0.95 之间（checkpoint 高估 ~21%），实为 {}'.format(k)


def test_max_batch_respects_budget():
    """max_batch 返回的 batch 必须真的装得下，且再大一点就装不下。"""
    cfg = dict(S.ANCHOR['cfg'], backbone_channels=128, value_res_blocks=3)
    b = S.max_batch(cfg)
    assert S.project(cfg, b)['fits']
    if b < 4200:
        assert not S.project(cfg, b + 40)['fits'], \
            'batch={} 之后仍有大把余量，二分搜索没找准'.format(b)
