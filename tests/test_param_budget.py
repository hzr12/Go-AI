"""参数量基准测试：用实际实例化测量，锁住 run.txt 里的数字。

起因：run.txt 曾写「V17 Value: 1.01M (96ch, 11 blocks)」，而实测
96ch+11blocks = 1,995,169（2.00M），1.01M 其实对应 96ch + 5 blocks。
文字与数字自相矛盾，导致「v18 是否比 v12 差」这类对照建立在错误基准上。

本测试把各子模块的实测值钉死，并校验 run.txt 与代码实际一致。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.networks.alphanet import AlphaGoNet  # noqa: E402
from src.networks.value_network import ValueNetwork  # noqa: E402

# v18 实际配置：对应 shell/train_sft_npu_4card.sh 的全部结构参数
V18 = dict(
    in_channels=12, action_size=362, use_checkpoint=True, arch='resnet',
    backbone_channels=192, backbone_res_blocks=17,
    attention_mode='mix', num_attention_layers=4, num_heads=4,
    attention_dropout=0.1, attn_mode='window_global', attn_window=5,
    res_blocks=8, convnext_blocks=4, attn_blocks=5,
    value_channels=96, value_res_blocks=8,
    policy_channels=128, policy_layers=3,
)

EXPECTED = {
    'backbone': 11189952,
    'value': 1496353,
    'policy': 172673,
    'total': 12858978,
}


def _n(mod):
    return sum(p.numel() for p in mod.parameters())


def test_v18_submodule_param_counts():
    """v18 各子模块参数量必须与 run.txt 记录一致。"""
    m = AlphaGoNet(**V18)
    assert _n(m.backbone) == EXPECTED['backbone'], \
        'backbone 参数量变了，run.txt 需同步更新'
    assert _n(m.value) == EXPECTED['value'], 'value 参数量变了'
    assert _n(m.policy) == EXPECTED['policy'], 'policy 参数量变了'
    assert sum(p.numel() for p in m.parameters()) == EXPECTED['total']


def test_submodules_sum_to_total():
    """三个子模块之和必须等于总数（防止漏算或重复计算）。"""
    m = AlphaGoNet(**V18)
    s = _n(m.backbone) + _n(m.value) + _n(m.policy)
    assert s == sum(p.numel() for p in m.parameters()), \
        'backbone+value+policy != total'


def test_value_head_table_in_run_txt_is_accurate():
    """run.txt 里的 value head 参数量表必须与实测一致。

    旧记录把 1.01M 标成「96ch, 11 blocks」，实际 11 blocks 是 2.00M，
    1.01M 对应 5 blocks——正是这类错误让架构对照失去意义。
    """
    known = {
        (96, 3): 664993, (96, 5): 997537, (96, 6): 1163809,
        (96, 7): 1330081, (96, 8): 1496353, (96, 11): 1995169,
        (64, 8): 702657, (64, 11): 924609,
    }
    for (vc, rb), exp in known.items():
        got = _n(ValueNetwork(in_channels=192, hidden_channels=vc,
                              num_res_blocks=rb, arch='resnet'))
        assert got == exp, f'value {vc}ch/{rb}blk 实测 {got} != 记录 {exp}'


# 已删：test_run_txt_records_are_consistent
# 它要求 run.txt 保留 `# v18 架构参数` 小节并逐项写实测值。但 v18 已退役
# （现役是 V7 / KATAGO_SE_CFG 12ch），而 run.txt 已改为「简洁」的一屏版，
# 不再承载退役架构的参数表。与 test_run_txt_sync.py 属同一类保证，
# 按同一决定移除。
# 随之失去的保证：run.txt 里的数字与实测值的自动比对。补参数时
#   需人工核对（README §「静默出错的坑」第 8 条记录了同类的 flag 冻结问题）。
#   若日后 run.txt 再次变长、值得机器校验，应改为校验「run.txt 实际写了哪些
#   数字」而不是「必须写 v18 的数字」。


def test_shell_script_config_matches_v18_definition():
    """4 卡 NPU 脚本的结构参数必须与本测试的 V18 定义一致。"""
    txt = open(os.path.join(ROOT, 'shell', 'train_sft_npu_4card.sh'),
               encoding='utf-8').read()
    for flag, expect in (
        ('--backbone-channels', 192), ('--backbone-res-blocks', 17),
        ('--res-blocks', 8), ('--convnext-blocks', 4), ('--attn-blocks', 5),
        ('--value-channels', 96), ('--value-res-blocks', 8),
        ('--policy-channels', 128), ('--policy-layers', 3),
    ):
        mm = re.search(re.escape(flag) + r'\s+(\d+)', txt)
        assert mm, f'4 卡脚本缺少 {flag}'
        assert int(mm.group(1)) == expect, \
            f'4 卡脚本 {flag}={mm.group(1)}，与 V18 定义 {expect} 不符'
