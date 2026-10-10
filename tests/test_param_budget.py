"""参数量基准测试：用实际实例化测量，锁住各子模块的实测值。

起因：run.txt 曾写「V17 Value: 1.01M (96ch, 11 blocks)」，而实测
96ch+11blocks = 1,995,169（2.00M），1.01M 其实对应 96ch + 5 blocks。
文字与数字自相矛盾，导致「v18 是否比 v12 差」这类对照建立在错误基准上。

⚠ 本文件**不再**校验 run.txt（`test_run_txt_records_are_consistent` 与
`test_run_txt_sync.py` 都随 v18 退役删除，见下方说明）⇒ run.txt 里的数字与
flag 现在**必须人工核对**。
"""
import os
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

# 已删：test_shell_script_config_matches_v18_definition（2026-10-09）
# 它 open `shell/train_sft_npu_4card.sh` 并逐项断言 9 个结构 flag 等于本测试的
# **V18** 定义。两头都已退役：
#   · 脚本随 NPU 后端一起删除，`shell/` 现在只剩 `train_sft_a100_1card.sh`；
#   · V18 架构已退役（现役是 V7 / `KATAGO_SE_CFG` 12ch）。
# **改指 A100 脚本也不成立**：那一支的 `--value-res-blocks` 是 **11** 而 V18 是 8
# （这处差异 `tests/test_run_py_sh.py` 另有记录），说明现役脚本本来就不是 V18 配置。
#
# 它提供的保证现在由谁接手：结构参数的**唯一真相源**是 `KATAGO_SE_CFG`，由
# `tests/test_no_undefined_names.py` 与 train_sft 自己的建网日志兜底；脚本里的
# flag **能不能被 argparse 接受**由 `tests/test_run_py_sh.py` 真送 argparse 把关。
# 换句话说「flag 存在但已不参与建网」这件事，本来就没有可断言的对象了 ——
# 结构参数已归档在 CLI 之外（见 `train_sft.py` 里 `KATAGO_SE_CFG` 上方的说明）。
