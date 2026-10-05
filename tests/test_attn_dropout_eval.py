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
  - 覆盖锁（用例 5，AST **逐站点**扫描）：枚举 backbone.py 里每一个 `_sdpa(...)` **调用点**
    （定义里的默认值不算站点），要求每处的 dropout 实参都取 `self.attn_drop_p`
    —— 既挡无闸门直传（`dropout_p=self.attn_drop`），也挡位置参数绕过（`_sdpa(q, k, v, 0.1)`）；
    并要求文件里没有读不到 `self.training` 的内联 `if self.attn_drop ...` 门。
    断言语义里**不含任何具体计数**，站点身份用「宿主类.函数[.嵌套函数]」而不是行号，
    所以红的时候能直接指名**哪条路径**漏了闸门；新增合法站点不会让本文件变红。

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


@pytest.mark.parametrize("attn_mode", ATTN_MODES)
def test_sdpa_runtime_fallback_respects_eval(attn_mode, monkeypatch):
    """`_sdpa` 的 SDPA→math 运行时回退上也**不许**出现 dropout（登记表的凭据）。

    ## 为什么要有这条

    `_sdpa` 里有一段运行时回退：SDPA 在部分 CANN / 老 torch_npu 上有 API 但
    运行时抛 `RuntimeError`，此时它递归回自己的手写 math 分支，并转发
    `dropout_p=dropout_p`。因为那是 `_sdpa` **自己**的调用点，AST 逐站点扫描
    会把它算成一处「没走 `self.attn_drop_p` 闸门」的站点 —— 它已被登记进
    `SDPA_SITES_GATED_BY`，但**登记不能只是一句注释**：必须证明它真的受管。

    ## 怎么证明

    把 `F.scaled_dot_product_attention` 打桩成**必抛**，强制每次前向都走那条
    回退分支，然后：
      - eval 下两次前向必须逐位相同（没丢 dropout）；
      - train 下必须**不**相同（否则上面那条只是因为回退分支压根没跑而恒成立）。

    train 侧那条对照是关键：没有它，一个「回退分支永远走不到」的桩也会让
    eval 那条通过。
    """
    def _boom(*_a, **_kw):
        raise RuntimeError("invalid configuration argument (桩：强制回退)")

    # backbone 里是 `F.scaled_dot_product_attention(...)` 这样调的，而 `F` 就是
    # `torch.nn.functional` 本身，所以 patch 这个模块的属性能真的把调用打掉。
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _boom)

    obs = _obs()
    with_drop = _make_model(attn_mode, DROPOUT).eval()
    a1, a2 = _forward(with_drop, obs), _forward(with_drop, obs)
    assert _logits_equal(a1[0], a2[0]) and _logits_equal(a1[1], a2[1]), (
        f'[{attn_mode}] 强制走 SDPA 回退分支时，eval 下两次前向输出不同 ⇒ '
        f'回退分支上有 dropout 没被 self.training 闸门管住')

    with_drop.train()
    t1, t2 = _forward(with_drop, obs), _forward(with_drop, obs)
    assert not (_logits_equal(t1[0], t2[0]) and _logits_equal(t1[1], t2[1])), (
        f'[{attn_mode}] train 下走回退分支时两次前向逐位相同 ⇒ 桩没生效或回退'
        f'分支没跑到，上面 eval 那条就失去意义')


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
# 5. 覆盖锁：每个注意力站点都受 self.training 闸门管住（AST 逐站点扫描）
# --------------------------------------------------------------------------- #

# 站点登记表：今天 backbone.py 里的注意力路径，**按「类.函数[.嵌套函数]」逐个列出**，
# 而不是写死一个计数。
#
# 这条锁要回答的是「**哪一条**注意力路径漏了闸门」，不是「有几条路径」。计数回答的是后者，
# 代价很具体：将来多一条合法站点（例如某个 v21 块决定自己独立注册一条注意力路径）会让
# 本文件误报一次，而对误报最省事的错误反应就是「把 4 改成 5」——锁就这么被掏空了。
#
# 新增注意力路径时**在这里登记一行，并写一句它为什么同样受 self.training 管住**：
#   · 下面只做**子集**校验（登记 ⊆ 实际站点），所以**新增站点不会让本文件变红** ——
#     反过来，某条登记路径被删掉或改名会让登记条目变陈旧并变红，逼一次有意的动作。
#   · 「为什么同样受管」由真正的锁回答：这里登记的每个 scope 都必须出现在
#     `_sdpa_call_sites()` 的结果里，而那里扫到的**每一个**站点都必须取 `self.attn_drop_p`。
SDPA_SITES_GATED_BY = (
    # global 路径：_sdpa(q, k, v, dropout_p=self.attn_drop_p, scale=self.scale)
    'MultiHeadSelfAttention._global_attn',
    # window 路径：窗口切分后逐窗 _sdpa(... dropout_p=self.attn_drop_p ...)
    'MultiHeadSelfAttention._window_attn',
    # window+global 路径：全局支路 _sdpa(q_p, k_full, v_full, dropout_p=self.attn_drop_p ...)
    'MultiHeadSelfAttention._window_global_attn',
    # axial 路径：嵌套的 attn_1d 里 _sdpa(t, t, t, dropout_p=self.attn_drop_p, ...)
    'MultiHeadSelfAttention._axial_attn.attn_1d',
    # 注：`_sdpa` 曾多出一条「模块级函数体内递归自调用」的站点（SDPA 运行时抛
    #   RuntimeError ⇒ 递归回 math 分支，转发形参 `dropout_p`）。它曾登记在这里，
    #   现已**随结构改造消失**：math 分支抽成具名的 `_sdpa_math`，那条调用不再以
    #   `_sdpa(...)` 的形式出现，所以登记表里也不再有它（上面的 stale 检查会抓）。
    #   那条路径**依然存在、依然受管**（`self.attn_drop_p` 由调用方给，转发不加工），
    #   由行为测试 `test_sdpa_runtime_fallback_respects_eval` 兜住 ——
    #   判据本身没有为它放宽过。
)


def _parent_map(tree):
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _scope_of(node, parent):
    """所在命名空间 `类.函数[.嵌套函数]`；不在类里则是 `函数[.嵌套函数]`。

    站点名取**定义 + 调用链**而非写死计数：本文件里同一段函数体内的直接调用
    归属该函数，跨类、跨函数时也各有归属。带点的嵌套函数（嵌套函数定义在
    方法体里，归到 ``'..._axial_attn.attn_1d'``）表示同一段里的不同嵌套位置
    —— 它们调用同一份逻辑，但**登记表要能按位置分别登记**。

    模块级**函数**体内的调用报该函数名（不是 ``'<module>'``）：``'<module>'``
    只能表示真正的模块级语句，而把 `_sdpa` 这类函数体内的递归自调用算成
    ``'<module>'`` 会让「登记一条合法路径」变成「放行全部模块级站点」——
    登记表立刻失去分辨力，那才是真正削弱这条测试。
    """
    fns = []
    cur = parent.get(node)
    while cur is not None:
        if isinstance(cur, ast.ClassDef):
            return '.'.join([cur.name] + list(reversed(fns)))
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fns.append(cur.name)
        cur = parent.get(cur)
    return '.'.join(reversed(fns)) if fns else '<module>'


def _sdpa_dropout_param_index(tree):
    """`_sdpa` 签名里 `dropout_p` 形参的位置序号（从函数定义里读，不写死序号）。

    必须从定义读，因为 `def _sdpa(q, k, v, dropout_p=0.0, ...)` 里那个 `0.0`
    **不是站点**（是默认值，不是调用）。读定义既拿到位置序号，又把「默认值不算站点」
    这件事显式化。定义被改名/挪走，或签名里没有 `dropout_p` 形参时返回 None，由调用方报错。
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_sdpa':
            for i, arg in enumerate(node.args.args):
                if arg.arg == 'dropout_p':
                    return i
    return None


def _sdpa_call_sites(tree, parent=None):
    """backbone.py 里**每一个** `_sdpa(...)` **调用点**，连同它的宿主 scope 与 AST 节点。

    只认 `ast.Call`，所以函数定义里的形参默认值天然不算站点（见上一条 docstring）。
    裸名 `_sdpa(...)` 和属性访问 `self._sdpa(...)` 都认 —— 后者是位置参数绕过闸门的
    自然藏身处，只扫裸名会漏掉它。
    """
    parent = _parent_map(tree) if parent is None else parent
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        named = getattr(fn, 'id', None) == '_sdpa' or (
            isinstance(fn, ast.Attribute) and fn.attr == '_sdpa')
        if named:
            sites.append((_scope_of(node, parent), node))
    return sites


def _sdpa_dropout_arg(site, index):
    """取一个 `_sdpa` 站点上**真正生效**的 dropout 实参（源码形式），没传就返回 `(None, None)`。

    位置参数优先：Python 语法不允许同一个形参既按位置又按关键字传，两者互斥。
    关键字扫描会漏掉位置实参，所以这里两条路都得走一遍。
    """
    _, node = site
    if index is not None and len(node.args) > index:
        return ast.unparse(node.args[index]), f'第 {index + 1} 个位置参数 dropout_p'
    kw = next((k for k in node.keywords if k.arg == 'dropout_p'), None)
    if kw is None:
        return None, None
    return ast.unparse(kw.value), 'dropout_p='


def _sdpa_gate_faults(tree):
    """返回 `[(scope, 原因, 行号)]`：每一条是一处**没被 self.training 闸门管住**的注意力路径。

    合规判据只有一个：dropout 实参里出现 `self.attn_drop_p`（那个 `.eval()` → 0.0 的派生物）。
    这一条判据同时挡得住三种失败形态，缺一不可：
      ① 无闸门直传 `dropout_p=self.attn_drop`（值里有 `attn_drop` 但不是 `attn_drop_p`）；
      ② 位置参数绕过关键字扫描：`_sdpa(q, k, v, 0.1)`；
      ③ 任何与派生物无关的实参（字面量、别的变量）—— 值里看不到闸门就等于没闸门。
    判据里**不含「有几条站点」**，所以新增一条合法站点不会让本文件变红。
    """
    index = _sdpa_dropout_param_index(tree)
    faults = []
    for site in _sdpa_call_sites(tree):
        scope, node = site
        given, where = _sdpa_dropout_arg(site, index)
        if given is None:
            # 这个站点压根没传 dropout（吃默认 0.0）→ 天然不漏
            continue
        if 'self.attn_drop_p' in given:
            continue
        if 'self.attn_drop' in given:
            reason = f'{where} 无闸门直传 {given}'
        else:
            reason = f'{where} 传的是 {given}，不是 self.attn_drop_p（值里看不到闸门）'
        faults.append((scope, reason, node.lineno))
    return faults


def _ungated_inline_gates(tree, parent=None):
    """文件里所有「提到 `self.attn_drop` 却不提 `self.training`」的 `if` 判据。

    按**属性名**判而不是按子串判：`self.attn_drop_p` 派生物本身已经读过 `self.training`
    （见 `attn_drop_p` 的 property），子串匹配会把 `if self.attn_drop_p > 0.0:` 这种
    合法的、已经受管的内联门误报成无闸门。
    """
    parent = _parent_map(tree) if parent is None else parent
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        attrs = {a.attr for a in ast.walk(node.test) if isinstance(a, ast.Attribute)}
        if 'attn_drop' in attrs and 'training' not in attrs:
            out.append((_scope_of(node, parent), ast.unparse(node.test), node.lineno))
    return out


def _mhsa_class(tree):
    """`MultiHeadSelfAttention` 的 ClassDef 节点（按名字定位，不写死行号）。"""
    return next((n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == 'MultiHeadSelfAttention'),
                None)


def _sparse_attn_inline_gates(tree):
    """`MultiHeadSelfAttention._sparse_attn` **函数体内**所有 `if` 的判据（源码形式）。

    返回 `None` 表示这个方法找不到了（被挪走或改名）。

    定位必须限定在函数体里，不能用文件级的 `any(...)`：文件级的检查分不清
    「sparse 这一处内联 `F.dropout` 的闸门在 `_sparse_attn` 里」与「文件里**别的**地方
    有一处同形的门」。P4.1 要往本文件加新块类，届时同形的门会不止一处 —— 那时文件级
    `any(...)` 依然为真，却完全证明不了 sparse 那一处被管住了。这条锁要能挡住以后新增的
    **同类站点**，前提是它锁的是站点而不是文本。
    """
    cls = _mhsa_class(tree)
    if cls is None:
        return None
    fn = next((f for f in cls.body
               if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
               and f.name == '_sparse_attn'), None)
    if fn is None:
        return None
    return [ast.unparse(n.test) for n in ast.walk(fn) if isinstance(n, ast.If)]


def test_all_five_attn_dropout_sites_are_gated():
    """AST **逐站点**扫描：每个 `_sdpa` 站点的 dropout 都取 `self.attn_drop_p`，
    任何内联 `if self.attn_drop ...` 门都带 `self.training`。

    （测试名保留 `five` 是因为 `src/networks/backbone.py:791` 按节点 id 引用了它；
     断言本身与站点数量无关。）

    这条锁**按站点**回答「哪一条注意力路径漏了闸门」，而不是回答「有几条路径」：
      · 站点身份是「宿主类.函数[.嵌套函数]」，不是行号 —— 行号会随无关改动整体漂移；
        红的时候能直接指名是哪条路径，这才是这条锁存在的理由。
      · 断言语义里**不含任何具体站点计数**：多一条合法站点不会让本文件变红
        （新增路径请在 `SDPA_SITES_GATED_BY` 里登记一行并说明它为什么同样受管），
        少一条（某条路径被删/改名）会让登记条目变陈旧并变红。

    下面查的是**同一处闸门的多种失败形态**，缺一不可：
      ① `_sdpa` 的 dropout 实参没走 `self.attn_drop_p`：既包括无闸门直传
        （`dropout_p=self.attn_drop`），也包括位置参数从关键字扫描下滑过去
        （`_sdpa(q, k, v, 0.1)`）；
      ② 文件里出现读不到 `self.training` 的内联门 `if self.attn_drop > 0.0:`
        （`self.attn_drop` 永远是构造期的常量，读不到 eval 语义）；
      ③ 该内联门存在，但**不在** `_sparse_attn` 里（被挪到别的函数/别的类）。
    ②③ 都需要站点级的定位，见 `_sparse_attn_inline_gates` 的 docstring。
    """
    src = open(BACKBONE_PY, encoding='utf-8').read()
    tree = ast.parse(src)

    assert _sdpa_dropout_param_index(tree) is not None, (
        'backbone.py 里找不到 `def _sdpa(...)`，或它的签名里没有 dropout_p 形参 —— '
        '本条锁靠这个签名定位 dropout 形参的位置序号（好让位置参数也逃不掉），'
        '无法继续逐站点核对；站点清单需重新核对')

    # ① 每个站点的 dropout 实参都必须是受 training 闸门管住的派生物
    faults = _sdpa_gate_faults(tree)
    assert not faults, (
        '以下注意力路径的 dropout 没走 self.attn_drop_p 闸门（逐站点指名）：\n'
        + '\n'.join(f'  · {scope}（backbone.py:{lineno}）：{reason}'
                    for scope, reason, lineno in faults)
        + '\n  —— 函数式 dropout API 的 training 参数默认 True，模块的 self.training '
          '传不进去，model.eval() 关不掉，SFT 评估不可复现。'
          '修法：实参改取 self.attn_drop_p；若这是新增的合法路径，'
          '请在 SDPA_SITES_GATED_BY 里登记一行并说明它为什么同样受管。')

    # 空转守卫：扫到一个站点都扫不到的话，上面那条断言就恒真了（= 假绿）
    sites = [scope for scope, _ in _sdpa_call_sites(tree)]
    assert sites, (
        'backbone.py 里一处 `_sdpa(...)` 调用点都没扫到 —— 本条锁已经空转，'
        '「逐站点核对」没有核对任何东西（是不是 _sdpa 被改名或挪走了？）')

    # 子集校验：登记过的路径必须还在（**只挡删改/改名，不挡新增** —— 新增请先登记）
    stale = [s for s in SDPA_SITES_GATED_BY if s not in sites]
    assert not stale, (
        f'SDPA_SITES_GATED_BY 里登记的这些注意力路径在 backbone.py 里已经找不到：{stale}'
        f'；实际扫到的站点是 {sites}。要么是被删了/改了名，要么是挪进了别的类或函数 —— '
        f'前者要先确认这条路径的消失是故意的，后者请更新登记表')

    # ② 文件里任何提到 self.attn_drop 却不提 self.training 的内联门
    ungated = _ungated_inline_gates(tree)
    assert not ungated, (
        '以下内联 dropout 门读不到 self.training：\n'
        + '\n'.join(f'  · {scope}（backbone.py:{lineno}）：if {test} →'
                    for scope, test, lineno in ungated)
        + '\n  —— self.attn_drop 是构造期算死的常量，功能式 dropout 也不看模块的 '
          'self.training，这类门在 eval 下照旧放行')

    # sparse 这一处：_sparse_attn 里的内联门必须带 self.training，且必须就在 _sparse_attn 里
    sparse_gates = _sparse_attn_inline_gates(tree)
    assert sparse_gates is not None, \
        ('MultiHeadSelfAttention 上找不到 _sparse_attn 方法 —— 站点被挪走或改了名，'
         '本文件的站点清单已失效，需重新核对')
    assert 'self.training and self.attn_drop > 0.0' in sparse_gates, (
        f'_sparse_attn 的函数体里找不到 `if self.training and self.attn_drop > 0.0:` '
        f'内联门，实得 {sparse_gates} —— sparse 模式（内联 F.dropout）的 dropout 闸门'
        f'缺失，或闸门被挪到了别的函数里（文件级同形的门不算数）')
    assert 'self.attn_drop > 0.0' not in sparse_gates, (
        f'_sparse_attn 的函数体里还有不带 self.training 的内联门（{sparse_gates}）—— '
        f'sparse 模式的注意力 dropout 在 eval 下照旧生效')

    # attn_drop_p 必须是 property，且带说明「为什么函数式 API 要显式关」
    cls = _mhsa_class(tree)
    prop = next((f for f in cls.body
                 if isinstance(f, ast.FunctionDef) and f.name == 'attn_drop_p'), None) \
        if cls is not None else None
    assert prop is not None, 'MultiHeadSelfAttention 上没有 attn_drop_p 属性'
    assert any(ast.unparse(d) == 'property' for d in prop.decorator_list), \
        'attn_drop_p 必须是 @property（取值要跟 self.training 走，不能构造期算死）'
    assert 'self.training' in ast.unparse(prop), \
        'attn_drop_p 的实现里没有读 self.training —— 它对 eval/训练一视同仁'
