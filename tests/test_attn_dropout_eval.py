"""注意力 dropout 的 eval 泄漏：functional dropout 不认 `self.training`，`model.eval()` 关不掉。

缺陷
----
`src/networks/backbone.py` 里注意力 dropout 全部走**函数式** API：
`F.scaled_dot_product_attention(..., dropout_p=p)` / `torch.nn.functional.dropout(x, p)`
/ `_flash_attn_func(..., dropout_p=p)`。这三者的 `training` 语义都是「**默认 True**」，
而它们所在的 `_sdpa` 是**模块级函数**、内联的 `F.dropout` 在方法体里也读不到
`self.training` 以外的开关 —— 于是 `model.eval()` 之后注意力 dropout **照旧生效**。

对照证据：同一个 block 里的 `ffn_drop = nn.Dropout(dropout)`（`:261`）本来就自动遵守
`self.training`；MindSpore 孪生实现 `src/networks_ms/alphanet_ms.py:87` 用的是真正的
`nn.Dropout` 模块。所以注意力这一路是**唯一的例外**，是 bug 不是建模选择。

后果（`--attention-dropout` 默认 0.1、run.txt 三条 SFT 命令都显式传 0.1，即生产配置）：
  ① 同一份权重、同一个 eval 集，指标不可复现（logits 抖 ~1e-2 量级）；
  ② **最佳模型选择被污染** —— `train_sft.py` 用 `metrics['top1'] > best_eval_acc`
     决定是否存盘，而 top1 抖 ±1e-2 时 top1 着法会直接翻转（与设备无关，恢复 RNG 救不了）；
  ③ GPU/NPU 上 eval 频率仍改写训练轨迹（dropout 掩码抽自 device 生成器，
     既没被 P2.2 的 `_isolated_global_rng` 快照也没被还原）。

覆盖
----
  - 行为锁（用例 1，**本任务主证据**）：`attn_mode` 的**全部 5 个取值**参数化，
    `attention_dropout=0.1` + `.eval()` + 同输入连跑两次 → policy/value logits **逐位相等**，
    且 `torch.get_rng_state()` 前后不变（eval 期间不再消耗任何 torch 随机）。
    修复前必红：5 个模式全红（实测 max|Δlogits| 1.5e-2~5.8e-2，top1 着法翻转）。
  - 回归护栏（用例 2，修复前后都应绿）：`.train()` 下同输入两次 → 输出**不同**且
    `torch.get_rng_state()` **推进**。证明「只关 eval、训练照旧」。
  - 等价性（用例 3）：`attention_dropout=0.1` 的 A 与 `0.0` 的 B 灌同一份 `state_dict`，
    都 `.eval()`、同输入 → 输出**逐位相等**（= eval 阶段的输出与 dropout 无关）。
    任何一处站点漏改都会红 —— 这是最强的完备性检查。
  - 单元锁（用例 4）：`attn_drop_p` 派生属性 `.eval()` → 0.0、`.train()` → `self.attn_drop`。
  - 覆盖锁（用例 5，AST 扫描）：文件里不得再出现 `dropout_p=self.attn_drop,` 这类
    无闸门的直接透传，也不得出现不带 `self.training` 的 `if self.attn_drop > 0.0:` 内联门。
    修复前必红（当前 4 处直接透传 + 1 处内联无 training 门）。

空转守卫（防假绿）
----------------
用例 1/2/3 都各带一条「这套断言真的对 dropout 敏感」的对照：把 `attention_dropout`
换成 0.0 之后，同一模型在 `eval()` 下必须**逐位相同、在 `train()` 下必须不消耗随机**；
若不成立，说明被测断言对 dropout 根本不敏感（哪怕全绿也说明不了问题）。
用例 3 还额外断言「A 在 `train()` 下确实与 B 不同」—— 证明 0.1 的 dropout 真在计算路径上，
不是恰好乘 0 蒙对。

全部 CPU、秒级：真 `AlphaGoNet` 但通道/层数取小值（64 通道 / 4 个 block / 9x9 棋盘 /
batch 4），不加载任何真实权重，不读真实数据。
"""
import ast
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.networks.backbone import MultiHeadSelfAttention  # noqa: E402
from src.networks.alphanet import AlphaGoNet  # noqa: E402

BACKBONE_PY = os.path.join(ROOT, 'src', 'networks', 'backbone.py')

ATTN_MODES = ['global', 'window', 'window_global', 'axial', 'sparse']
DROPOUT = 0.1          # train_sft.py:868 的默认值，run.txt 的 SFT 命令也显式传 0.1
BOARD = 9
BATCH = 4


def _make_model(attn_mode, dropout, seed=4242):
    """真 `AlphaGoNet`，但通道/层数取小值以保证 CPU 秒级（不加载任何真实权重）。"""
    torch.manual_seed(seed)
    return AlphaGoNet(
        in_channels=12,
        backbone_channels=64,
        backbone_res_blocks=4,
        attention_mode="all",
        num_attention_layers=4,
        num_heads=4,
        attention_dropout=dropout,
        attn_mode=attn_mode,
        attn_window=3,
        policy_channels=32,
        value_channels=32,
        action_size=BOARD * BOARD + 1,
        res_blocks=2,
        attn_blocks=2,
    )


def _obs(seed=99):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, 12, BOARD, BOARD, generator=g)


def _forward(model, obs):
    with torch.no_grad():
        policy, value = model(obs)
    return policy.clone(), value.clone()


def _logits_equal(a, b):
    """逐位相等（不能用 `==`：它会得到逐元素布尔 Tensor，`assert` 语义完全不同）。"""
    return torch.equal(a, b)


# --------------------------------------------------------------------------- #
# 1. 行为锁：eval 下 5 个注意力模式全部逐位可复现（本任务主证据）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attn_mode", ATTN_MODES)
def test_eval_mode_is_bit_identical_across_attn_modes(attn_mode):
    """`attention_dropout=0.1` + `.eval()` + 同输入连跑两次 → policy/value 逐位相等，RNG 不动。

    修复前必红（本缺陷本身）：注意力 dropout 走 functional API，`training` 默认 True，
    `model.eval()` 关不掉 → 每次前向抽的掩码不同 → logits 抖动、top1 着法翻转、
    `torch.get_rng_state()` 被推进。5 个模式对应 5 个不同站点，必须**逐个**参数化：
    漏改任何一个站点，那一个模式就红。

    空转守卫：同构的 `dropout=0.0` 模型在 eval 下必须**逐位相同且不消耗 RNG** ——
    否则「相等」可能只是因为这个模式下 dropout 压根没进计算路径。
    """
    model = _make_model(attn_mode, DROPOUT).eval()
    obs = _obs()

    torch.manual_seed(31337)
    rng_before = torch.get_rng_state().clone()
    policy1, value1 = _forward(model, obs)
    policy2, value2 = _forward(model, obs)
    rng_after = torch.get_rng_state()

    assert _logits_equal(policy1, policy2), (
        f'[{attn_mode}] eval 下同输入两次前向的 policy logits 不逐位相同：'
        f'max|Δ|={(policy1 - policy2).abs().max().item():.3e} —— '
        f'model.eval() 关不掉注意力 dropout，SFT 评估不可复现、'
        f'最佳模型选择是在噪声上做的')
    assert _logits_equal(value1, value2), (
        f'[{attn_mode}] eval 下同输入两次前向的 value 不逐位相同：'
        f'max|Δ|={(value1 - value2).abs().max().item():.3e}')
    assert torch.equal(rng_before, rng_after), (
        f'[{attn_mode}] eval 期间消耗了 torch 随机 —— dropout 掩码抽自全局流，'
        f'eval 频率仍会改写训练轨迹（device 侧更甚）')

    # --- 空转守卫：dropout=0.0 的同构模型在 eval 下必须逐位相同且不消耗 RNG ---
    zero = _make_model(attn_mode, 0.0).eval()
    torch.manual_seed(31337)
    z_rng_before = torch.get_rng_state().clone()
    z1, _ = _forward(zero, obs)
    z2, _ = _forward(zero, obs)
    assert _logits_equal(z1, z2), \
        f'[{attn_mode}] 连 dropout=0.0 的模型在 eval 下都不逐位相同 —— 说明这条断言的' \
        f'「空转守卫」失效（问题在别处，不在注意力 dropout）'
    assert torch.equal(z_rng_before, torch.get_rng_state()), \
        f'[{attn_mode}] dropout=0.0 的模型在 eval 下竟消耗了 torch RNG —— ' \
        f'说明这个模式的 eval 前向里还有别的随机源，「RNG 不动」这条断言证明不了' \
        f'注意力 dropout 已被关掉'


# --------------------------------------------------------------------------- #
# 2. 回归护栏：train 下 dropout 照旧（只关 eval，不动训练期）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attn_mode", ATTN_MODES)
def test_train_mode_still_drops_out(attn_mode):
    """`.train()` 下同输入两次 → 输出**不同**且 RNG **推进**；`dropout=0.0` 时则两者都不发生。

    这是本修复的**风险边界**：修复只允许改 eval 语义，训练期有效值必须与今天逐位相同。
    修复前后都应绿。

    `dropout=0.0` 的对照（输出逐位相同 + RNG 不动）在本例里**同时是风险边界的守卫**：
    如果哪天「修 eval」顺手把 train 也关了，这条会红。
    """
    model = _make_model(attn_mode, DROPOUT).train()
    obs = _obs()

    torch.manual_seed(31337)
    rng_before = torch.get_rng_state().clone()
    policy1, _ = _forward(model, obs)
    policy2, _ = _forward(model, obs)
    rng_after = torch.get_rng_state()

    assert not _logits_equal(policy1, policy2), (
        f'[{attn_mode}] train 下同输入两次前向逐位相同 —— 训练期的注意力 dropout 被误关了，'
        f'这会改变训练轨迹（正则项消失），不属于本任务的修复范围')
    assert not torch.equal(rng_before, rng_after), (
        f'[{attn_mode}] train 下前向没有推进 torch RNG —— 训练期 dropout 没生效')

    zero = _make_model(attn_mode, 0.0).train()
    torch.manual_seed(31337)
    rng0_before = torch.get_rng_state().clone()
    q1, _ = _forward(zero, obs)
    q2, _ = _forward(zero, obs)
    assert _logits_equal(q1, q2) and torch.equal(rng0_before, torch.get_rng_state()), (
        f'[{attn_mode}] dropout=0.0 的模型在 train 下既不逐位相同又消耗了 RNG —— '
        f'这条对照本身失效，用例 3 的「eval 等价于 0.0」也就失去了参照系')


# --------------------------------------------------------------------------- #
# 3. 等价性：eval 阶段的输出与 dropout 无关（最强的完备性检查）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attn_mode", ATTN_MODES)
def test_eval_output_equals_zero_dropout_model(attn_mode):
    """A(dropout=0.1) 与 B(dropout=0.0) 灌同一份 `state_dict`，都 eval → 输出逐位相等。

    修复前必红，且**任何一处站点漏改都会红**（5 个站点各自被 5 个模式的参数化覆盖到）——
    这比逐站点检查更硬：它不关心实现长什么样，只问「eval 的输出还依不依赖 dropout」。

    末尾的 train 侧对照证明 0.1 的 dropout 真的在计算路径上（不是恰好乘 0 蒙对）。
    """
    obs = _obs()
    with_drop = _make_model(attn_mode, DROPOUT).eval()
    no_drop = _make_model(attn_mode, 0.0).eval()
    no_drop.load_state_dict(with_drop.state_dict(), strict=True)

    a_policy, a_value = _forward(with_drop, obs)
    b_policy, b_value = _forward(no_drop, obs)
    assert _logits_equal(a_policy, b_policy), (
        f'[{attn_mode}] eval 下「有 dropout」与「无 dropout」的同权重模型输出不同：'
        f'max|Δ|={(a_policy - b_policy).abs().max().item():.3e} —— '
        f'该模式的注意力路径上还有一处 functional dropout 没被 self.training 闸门管住')
    assert _logits_equal(a_value, b_value), (
        f'[{attn_mode}] eval 下 value 输出依赖了 dropout：'
        f'max|Δ|={(a_value - b_value).abs().max().item():.3e}')

    # --- 对照：train 侧必须**不**相等（0.1 的 dropout 真在路径上） ---
    with_drop.train()
    no_drop.train()
    t_a, _ = _forward(with_drop, obs)
    t_b, _ = _forward(no_drop, obs)
    assert not _logits_equal(t_a, t_b), (
        f'[{attn_mode}] train 下有/无 dropout 的输出逐位相同 —— 该模式的注意力 dropout '
        f'可能根本没接进计算路径，上面那条「相等」就没有意义')


# --------------------------------------------------------------------------- #
# 4. 单元锁：attn_drop_p 派生属性遵守 self.training
# --------------------------------------------------------------------------- #
def test_attn_drop_p_property_respects_training():
    """`.eval()` → `attn_drop_p == 0.0`；`.train()` → `attn_drop_p == self.attn_drop`。

    修复前必红：`MultiHeadSelfAttention` 上根本没有 `attn_drop_p`（AttributeError）。
    属性是 5 个站点的**唯一取值来源**，所以这条同时锁住了「闸门存在」与「闸门认 training」。
    """
    attn = MultiHeadSelfAttention(32, num_heads=4, dropout=DROPOUT, mode="global")
    assert attn.attn_drop == DROPOUT

    attn.train()
    assert attn.attn_drop_p == attn.attn_drop == DROPOUT, (
        f'train 下 attn_drop_p 应等于构造时的 dropout，实得 {attn.attn_drop_p!r}')

    attn.eval()
    assert attn.attn_drop_p == 0.0, (
        f'eval 下 attn_drop_p 应为 0.0（函数式 dropout 不看 self.training，必须显式关），'
        f'实得 {attn.attn_drop_p!r}')

    # 构造参数本身不被改写（推理入口靠 attention_dropout=0.0 的默认值拿确定性）
    assert attn.attn_drop == DROPOUT, 'attn_drop 不该被 attn_drop_p 之类的读取改写'
    assert isinstance(type(attn).attn_drop_p, property), \
        'attn_drop_p 必须是派生 property（而不是构造期算好的普通属性，否则不跟 training 走）'


# --------------------------------------------------------------------------- #
# 5. 覆盖锁：5 个站点一个不漏（AST 扫描，修复前必红）
# --------------------------------------------------------------------------- #
def _sdpa_dropout_arg_values(tree):
    """抽出 `_sdpa(...)` 调用里 `dropout_p=` 关键字的实参（源码形式）。

    按「调用点」而不是「出现次数」统计：函数定义里的形参默认值不算站点。
    """
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, 'id', None) == '_sdpa'):
            for kw in node.keywords:
                if kw.arg == 'dropout_p':
                    out.append(ast.unparse(kw.value))
    return out


def test_all_five_attn_dropout_sites_are_gated():
    """AST 扫描：`_sdpa` 的 4 个 `dropout_p` 全部走 `attn_drop_p`，内联门带 `self.training`。

    4 处 `_sdpa` 调用分别服务 `_global_attn`(:268) / `_window_attn`(:335) /
    `_window_global_attn`(:472) / `_axial_attn` 的 `attn_1d`(:493)，第 5 处是
    `_sparse_attn` 里的内联 `F.dropout`（`:395-396`）。缺任何一个，行为用例 1/3 都会红；
    这条再加一层静态锁，让「漏改」在 review 与 CI 里都直接可见。
    """
    src = open(BACKBONE_PY, encoding='utf-8').read()
    tree = ast.parse(src)

    passthrough = [v for v in _sdpa_dropout_arg_values(tree) if v == 'self.attn_drop']
    assert not passthrough, (
        f'_sdpa 的 dropout_p 仍在直接透传 self.attn_drop（{len(passthrough)} 处）—— '
        f'functional API 的 training 默认 True，eval 下关不掉')

    gated = [v for v in _sdpa_dropout_arg_values(tree) if v == 'self.attn_drop_p']
    assert len(gated) == 4, (
        f'应恰好有 4 处 _sdpa 调用取 self.attn_drop_p（global/window/window_global/axial），'
        f'实得 {len(gated)} 处：{_sdpa_dropout_arg_values(tree)}')

    # 第 5 处：_sparse_attn 里的内联门必须带 self.training
    bare_gates = [ast.unparse(n.test) for n in ast.walk(tree)
                  if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.attn_drop > 0.0']
    assert not bare_gates, (
        f'仍有不带 self.training 的内联 dropout 门（{bare_gates}）—— sparse 模式的'
        f'注意力 dropout 在 eval 下照旧生效')
    assert any(ast.unparse(n.test) == 'self.training and self.attn_drop > 0.0'
               for n in ast.walk(tree) if isinstance(n, ast.If)), \
        '找不到形如 `if self.training and self.attn_drop > 0.0:` 的内联门 —— ' \
        'sparse 模式（_sparse_attn 的内联 F.dropout）的 dropout 闸门缺失'

    # attn_drop_p 必须是 property，且带说明「为什么函数式 API 要显式关」
    prop = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'MultiHeadSelfAttention':
            prop = next((f for f in node.body
                         if isinstance(f, ast.FunctionDef) and f.name == 'attn_drop_p'), None)
    assert prop is not None, 'MultiHeadSelfAttention 上没有 attn_drop_p 属性'
    assert any(ast.unparse(d) == 'property' for d in prop.decorator_list), \
        'attn_drop_p 必须是 @property（取值要跟 self.training 走，不能构造期算死）'
    assert 'self.training' in ast.unparse(prop), \
        'attn_drop_p 的实现里没有读 self.training —— 它对 eval/训练一视同仁'
