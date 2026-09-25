"""NPU TorchAir 图编译：受控实验，默认关闭，失败必须回退 eager。

背景
----
`train_sft.py` 此前对 NPU 硬禁用 `torch.compile`（inductor 在 NPU 不可用），
全程 eager。实测 4 卡 910A 上 NPU 报 `Aicore Usage Rate` 85~100% 而实际只有
659 samples/s/卡——AICore 一直"忙"但产出极低，典型的 eager 小算子形态。
TorchAir 是昇腾官方的图编译后端，是唯一可能带来量级提升的路径。

环境（本项目实测）：torch 2.1.0 / torch_npu 2.1.0.post3 / CANN 8.0.RC1 /
aarch64。属 2023 年代组合，功能成熟度存疑，故定位为**受控实验**。

必须守住的安全性质
----------------
1. **失败必须回退 eager**，绝不能让整个训练崩掉。
2. **导入顺序强制** `torch_npu` 先于 `torchair`，否则图模式**静默降级为
   eager** 而不报错——那样就白开了，还白付了预热代价。
3. **不得传 `mode` / `options`**：昇腾 NPU 不支持；且 `reduce-overhead` 正是
   A100 上 OOM 的元凶（CUDA Graphs 私有内存池不归还），NPU 上同样要避开。
4. **显存是真实风险**：本项目 4 卡 910A 常驻已到 26.7~31.1GB / 32GB，
   图模式额外的 workspace/图缓冲极易再炸。默认关闭即为此。
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()
CODE = '\n'.join(l for l in SRC.splitlines() if not l.lstrip().startswith('#'))


def _graph_block():
    m = re.search(r'if _npu_graph:(.*?)\n    elif args\.compile', CODE, re.S)
    assert m, '未找到 _npu_graph 分支'
    return m.group(1)


def test_flag_exists_and_defaults_off():
    assert '--npu-graph-compile' in SRC
    m = re.search(r"'--npu-graph-compile',\s*type=int,\s*default=(\d)", CODE)
    assert m, '未找到 --npu-graph-compile 定义'
    assert int(m.group(1)) == 0, \
        '图编译必须默认关闭：显存已到 26.7~31.1GB/32GB，且版本组合成熟度存疑'


def test_restricted_to_npu_backend():
    assert re.search(
        r"_npu_graph = \(args\.npu_graph_compile == 1 and _backend == 'npu'\)", CODE
    ), '图编译必须限定 NPU 后端（inductor 路径与 torchair 路径互斥）'


def test_import_order_npu_before_torchair():
    """torch_npu 必须先于 torchair，否则图模式静默降级为 eager 而不报错。"""
    body = _graph_block()
    i_npu = body.find('import torch_npu')
    i_air = body.find('import torchair')
    assert i_npu != -1, '缺少 import torch_npu'
    assert i_air != -1, '缺少 import torchair'
    assert i_npu < i_air, \
        'torch_npu 必须在 torchair 之前导入，否则图模式静默失效'


def test_uses_torchair_backend():
    body = _graph_block()
    assert 'torchair.CompilerConfig()' in body
    assert 'torchair.get_npu_backend(' in body
    assert re.search(r'torch\.compile\(\s*model,\s*backend=', body), \
        '必须把 torchair 的 backend 显式传给 torch.compile'
    assert 'dynamic=False' in body, '固定形状应关掉 dynamic（避免多次重编译）'


def test_never_sets_mode_or_options():
    """昇腾 NPU 不支持 mode/options；reduce-overhead 更是 A100 OOM 的元凶。"""
    body = _graph_block()
    assert 'mode=' not in body, 'NPU 图编译不应传 mode（昇腾不支持且有 OOM 前科）'
    assert 'options=' not in body, 'NPU 图编译不应传 options（昇腾不支持）'
    assert 'reduce-overhead' not in body


def test_falls_back_to_eager_on_failure():
    """编译失败必须回退 eager 并告警——不能让整个训练崩掉。"""
    body = _graph_block()
    assert 'except Exception' in body, '图编译失败必须被捕获'
    assert "getattr(model, '_orig_mod', model)" in body, \
        '回退时必须剥离 OptimizedModule 包装'
    assert '回退 eager' in body, '回退必须有明确告警'


def test_warms_up_under_autocast():
    """预热前向必须与真实训练一致地包 autocast。

    否则 FP32 输入干灌会报 flash-attn 只接受 fp16/bf16，导致图编译被误判为
    不可用而回退 eager（且这个误判很难排查）。
    """
    body = _graph_block()
    assert 'maybe_autocast' in body, '预热前向应包 autocast'
    assert 'model(' in body or 'model(' in body.replace(' ', ''), '缺少预热前向'


def test_warns_about_memory_risk():
    """开启时必须就显存风险给出提示（常驻已 26.7~31.1GB / 32GB）。"""
    assert '显存' in SRC, '应就显存风险给出提示'
    m = re.search(r'--npu-graph-compile', SRC)
    assert m, '缺少参数说明'
    # 参数 help 里应提到显存/风险
    help_txt = SRC[m.start():m.start() + 600]
    assert ('显存' in help_txt or '风险' in help_txt), \
        '--npu-graph-compile 的 help 未说明显存风险'


def test_npu_still_disables_inductor_path():
    """NPU 上 inductor 路径仍应被禁用（本项是另一条互斥路径）。"""
    m = re.search(
        r'if args\.compile == 1:(.*?)args\.compile = False', CODE, re.S)
    assert m, 'NPU 上 inductor 路径的禁用逻辑不见了'
    body = m.group(1)
    assert 'inductor' in body and '不可用' in body, \
        'NPU 上应继续禁用 inductor 并说明原因'
    assert '--npu-graph-compile' in body, \
        '禁用告警应把用户引导到 --npu-graph-compile 而不是已废弃的旧说法'


def test_shell_scripts_expose_the_switch():
    """NPU 训练脚本应能方便地开关它，且默认关闭。"""
    import glob
    npu = [f for f in glob.glob(os.path.join(ROOT, 'shell', 'train_sft_npu*.sh'))]
    assert npu, '未找到 NPU SFT 脚本'
    for f in npu:
        txt = open(f, encoding='utf-8').read()
        assert 'NPU_GRAPH_COMPILE' in txt, \
            '{} 未暴露 NPU_GRAPH_COMPILE 开关'.format(os.path.basename(f))
        m = re.search(r'(?m)^NPU_GRAPH_COMPILE=(\d)', txt)
        assert m, '{} 的 NPU_GRAPH_COMPILE 未显式赋 0/1'.format(os.path.basename(f))
        assert int(m.group(1)) in (0, 1)
        assert '--npu-graph-compile "$NPU_GRAPH_COMPILE"' in txt, \
            '{} 未把开关传给 train_sft.py'.format(os.path.basename(f))
