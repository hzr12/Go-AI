"""v19 精简档（约 8.77M 参数）的参数量与显存预算守卫。

v19 相对 v18 砍掉 32% 容量以换速度与显存余量。**这个取舍必须有据可查**，
故把参数量落在 8.5M~9.2M 区间、显存余量、以及 LR 标定都钉成断言，
避免以后有人改了某个 blocks 数字却没察觉架构已经漂移。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import search_arch as S  # noqa: E402
from src.networks.alphanet import AlphaGoNet  # noqa: E402

SH = os.path.join(ROOT, 'shell', 'train_sft_npu_4card_v19.sh')

# 脚本里真实生效的结构参数（backbone-res-blocks 是死参数，故不计入）
EXPECT = dict(backbone_channels=192, res_blocks=4, convnext_blocks=2,
              attn_blocks=5, value_channels=96, value_res_blocks=3,
              policy_channels=128, policy_layers=3, attn_window=5,
              num_heads=4)


def _flags():
    txt = open(SH, encoding='utf-8').read()
    code = '\n'.join(l for l in txt.splitlines()
                     if not l.lstrip().startswith('#'))
    out = {}
    for k in ('backbone-channels', 'res-blocks', 'convnext-blocks',
              'attn-blocks', 'value-channels', 'value-res-blocks',
              'policy-channels', 'policy-layers', 'attn-window', 'num-heads',
              'batch-size'):
        m = re.search(r'--' + k + r'\s+"?\$?([A-Za-z_]+|\d+)', code)
        assert m, '脚本缺少 --{}'.format(k)
        out[k] = m.group(1)
    return out, code


def test_v19_script_exists():
    assert os.path.isfile(SH), '缺少 v19 脚本'


def test_v19_matches_expected_structure():
    f, _ = _flags()
    for k, v in EXPECT.items():
        key = k.replace('_', '-')
        if f[key] != str(v):
            # batch 之类走变量，单独处理
            assert str(v) in f[key], \
                '--{} 期望 {}，脚本给的是 {}'.format(key, v, f[key])


def test_v19_param_count_in_budget():
    """参数量必须落在 8.5M~9.2M —— 偏离说明架构被改动而取舍记录已失效。"""
    f, _ = _flags()
    cfg = dict(S.ANCHOR['cfg'],
               backbone_channels=int(f['backbone-channels']),
               res_blocks=int(f['res-blocks']),
               convnext_blocks=int(f['convnext-blocks']),
               attn_blocks=int(f['attn-blocks']),
               value_channels=int(f['value-channels']),
               value_res_blocks=int(f['value-res-blocks']),
               policy_channels=int(f['policy-channels']),
               policy_layers=int(f['policy-layers']),
               attn_window=int(f['attn-window']),
               num_heads=int(f['num-heads']))
    n = sum(p.numel() for p in AlphaGoNet(
        in_channels=12, action_size=362, arch='resnet',
        attention_dropout=0.0, use_checkpoint=True, **cfg).parameters())
    assert 8.5e6 <= n <= 9.2e6, \
        'v19 参数量 {:.2f}M 已漂出 8.5M~9.2M 预算（取舍记录失效）'.format(n / 1e6)


def test_v19_fits_memory_with_margin():
    """B=3000 时显存余量必须 >= 4GB，不能贴着 32GB 上限跑。"""
    f, _ = _flags()
    b = int(re.search(r'(?m)^BATCH=(\d+)', open(SH, encoding='utf-8').read())
            .group(1))
    cfg = dict(S.ANCHOR['cfg'],
               backbone_channels=int(f['backbone-channels']),
               res_blocks=int(f['res-blocks']),
               convnext_blocks=int(f['convnext-blocks']),
               attn_blocks=int(f['attn-blocks']),
               value_channels=int(f['value-channels']),
               value_res_blocks=int(f['value-res-blocks']))
    r = S.project(cfg, b)
    assert r['fits'], 'B={} 显存 {:.1f}G 超出预算'.format(b, r['total_gb'])
    margin = S.NPU_GB - r['total_gb']
    assert margin >= 4.0, \
        '显存余量仅 {:.1f}GB，太薄（标定只经过单一实测点）'.format(margin)


def test_v19_lr_follows_sqrt_law():
    """LR 必须等于 0.00356 × √(有效 batch / 2500)。"""
    txt = open(SH, encoding='utf-8').read()
    b = int(re.search(r'(?m)^BATCH=(\d+)', txt).group(1))
    ws = int(re.search(r'(?m)^WORLD_SIZE=(\d+)', txt).group(1))
    lr = float(re.search(r'(?m)^LR=([\d.]+)', txt).group(1))
    assert abs(lr - 0.00356 * ((b * ws / 2500) ** 0.5)) < 2e-5, \
        'LR {} 与平方根律标定值不符（batch {} x {} 卡）'.format(lr, b, ws)


def test_v19_has_scaler_guardrails():
    """必须带上修好的 GradScaler 参数——旧 run 正是被缩放值崩塌毁掉的。"""
    txt = open(SH, encoding='utf-8').read()
    assert '--scaler-init-scale' in txt
    assert '--scaler-growth-interval' in txt
    assert re.search(r'(?m)^SCALER_INIT=(\d+)', txt)
    init = int(re.search(r'(?m)^SCALER_INIT=(\d+)', txt).group(1))
    assert 512 <= init <= 4096, \
        'SCALER_INIT={} 超出实测平衡区间 512~2048 的邻近档'.format(init)


def test_v19_attn_window_is_5():
    """attn-window 实测 ws=5/7 最优（ws=3 因 nW=49 暴涨到 38.5G）。"""
    f, _ = _flags()
    assert f['attn-window'] == '5'


def test_v19_does_not_raise_value_depth():
    """value head 未开 checkpoint，是唯一按块线性吃显存的部件。

    96ch x 11 blocks 实测 32.7G，直接越过 32GB 卡的上限。
    """
    f, _ = _flags()
    assert int(f['value-res-blocks']) <= 3, \
        'value-res-blocks 过大：每块约 0.46GB 线性吃显存'
