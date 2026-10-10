"""MHSA 的 q/k/v 合成：前向等价 + 对外契约不变

背景
----
V7 的 dim=128，MHSA 里 q/k/v 各是一个 ``Linear(128,128)``。M 极小，
每个 GEMM 几乎全部时间花在 ACL 启动开销上。11 block × 2 inner = 22 处，
每处三次启动。合成成一个 ``Linear(128,384)`` 后每处只剩一次启动。

合成**只发生在训练时的前向组织**。对外契约一律不变：

  1. ``state_dict()`` 仍给 ``attn.q/k/v.weight`` 三个键（所以导出器
     ``katago_export.py:268-270`` 读的键没变、``.bin.gz`` 逐字节相同）；
  2. 旧 checkpoint 的三键能直接 ``load_state_dict``（``strict=True``）；
  3. ``named_parameters()`` 是 ``qkv``（EMA 用），旧 ckpt 的 EMA shadow
     由 ``_migrate_ema_shadow_qkv`` 迁移。

这些是**契约**，不是实现细节：一旦破坏，症状是「引擎加载失败」或
「resume 崩」，而且往往离现场很远。所以逐条钉住。
"""

import math

import torch
import torch.nn as nn

from src.networks.katago_v7 import (
    MHSA,
    NBT_TF_CFG,
    RoPE2D,
    _migrate_ema_shadow_qkv,
    _ScaledLinear,
    _sdpa,
    build_katago_v7_net,
)


DIM, HEADS, N_BLOCKS, INNER = 128, 4, 11, 2


class _OldMHSA(nn.Module):
    """合成前的原始实现：三个独立 ``Linear``，作为前向等价的参照物。"""

    def __init__(self, dim=DIM, num_heads=HEADS, attn_dropout=0.0):
        super().__init__()
        self.dim, self.num_heads = int(dim), int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q = _ScaledLinear(self.dim, self.dim, bias=False)
        self.k = _ScaledLinear(self.dim, self.dim, bias=False)
        self.v = _ScaledLinear(self.dim, self.dim, bias=False)
        self.out = _ScaledLinear(self.dim, self.dim, bias=False)
        self.rope = RoPE2D(self.num_heads, self.head_dim)
        self.attn_dropout = float(attn_dropout)

    def _heads(self, t):
        b, n, _ = t.shape
        return t.reshape(b, n, self.num_heads, self.head_dim) \
                .permute(0, 2, 1, 3).contiguous()

    def forward(self, t, pos):
        b, n, c = t.shape
        q = self._heads(self.q(t))
        k = self._heads(self.k(t))
        v = self._heads(self.v(t))
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        ctx = _sdpa(q, k, v, dropout_p=self.attn_dropout, scale=self.scale)
        return self.out(ctx.permute(0, 2, 1, 3).reshape(b, n, c))


def _inputs(b=3, n=11, dim=DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(b, n, dim, generator=g)
    pos = torch.stack([
        torch.randint(0, 19, (b, n), generator=g).float(),
        torch.randint(0, 19, (b, n), generator=g).float(),
    ], -1)
    return t, pos


# --------------------------------------------------------------------------- #
# 1. 前向等价 —— 合成是纯优化，数值必须逐位相同
# --------------------------------------------------------------------------- #
def test_fused_forward_is_bitwise_identical():
    """fused MHSA 与三个独立 Linear 的输出必须**逐位**相同（不是 allclose）。"""
    torch.manual_seed(0)
    fused = MHSA(DIM, HEADS).eval()
    old = _OldMHSA(DIM, HEADS).eval()

    # 顺便验证「对外是 q/k/v 三键」：旧结构能无缺口地吃下新 state_dict
    missing, unexpected = old.load_state_dict(fused.state_dict(), strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    t, pos = _inputs()
    with torch.no_grad():
        y_fused = fused(t, pos)
        y_old = old(t, pos)
    assert y_fused.shape == y_old.shape == (3, 11, DIM)
    assert torch.equal(y_fused, y_old), (
        '前向不再逐位相同：最大差 %.3e'
        % (y_fused - y_old).abs().max().item())


def test_fused_chunk_is_view_not_copy():
    """切 q/k/v 只能切出 view；否则每次前向多一次 384×N 的拷贝，收益全吐回去。"""
    net = MHSA(DIM, HEADS)
    fused = net.qkv.weight
    assert fused.shape == (3 * DIM, DIM)
    base = fused.untyped_storage().data_ptr()
    q, k, v = net._split_qkv(fused)
    for part in (q, k, v):
        assert part.shape == (DIM, DIM)
        # 注意比较 **storage** 而不是 data_ptr：k/v 段有偏移，data_ptr 必然不同，
        # 但它们必须仍与 qkv 共用同一块 storage（即是 view 而非拷贝）。
        assert part.untyped_storage().data_ptr() == base, '切出了拷贝'


# --------------------------------------------------------------------------- #
# 2. 对外契约：state_dict 键名不变（导出与旧 ckpt 都靠它）
# --------------------------------------------------------------------------- #
def test_state_dict_still_exposes_qkv_split_keys():
    """``state_dict()`` 必须给 q/k/v 三键，且**不含** qkv。"""
    sd = MHSA(DIM, HEADS).state_dict()
    for nm in ('q', 'k', 'v', 'out'):
        assert nm + '.weight' in sd, (nm, sorted(sd))
    assert not [k for k in sd if 'qkv' in k], sorted(sd)
    # q/k/v 三段合起来才是那个 3*dim 的 qkv
    assert sum(sd[nm + '.weight'].shape[0] for nm in ('q', 'k', 'v')) == 3 * DIM


def test_state_dict_split_keys_are_not_aliased():
    """三个键不能别名同一块 storage，否则 load 进去会互相污染。"""
    sd = MHSA(DIM, HEADS).state_dict()
    ptrs = {sd[n + '.weight'].untyped_storage().data_ptr() for n in ('q', 'k', 'v')}
    assert len(ptrs) == 3, 'q/k/v 三个切片共享了 storage'


def test_whole_net_param_count_and_key_count_unchanged():
    """合成只是重排：参数量与 state_dict 键数都必须与合成前一致。"""
    net = build_katago_v7_net()
    n = sum(p.numel() for p in net.parameters())
    assert n == NBT_TF_CFG['params_total'], '%d != %d' % (n, NBT_TF_CFG['params_total'])

    sd = net.state_dict()
    # q/k/v 各 22 处 ⇒ 66 个键。合成前就是这个数，合成后不能变。
    split = [k for k in sd
             if any(k.endswith('.%s.weight' % nm) for nm in ('q', 'k', 'v'))]
    assert len(split) == N_BLOCKS * INNER * 3, len(split)
    assert not [k for k in sd if 'qkv' in k], '对外不该出现 qkv 键'


def test_named_parameters_exposes_qkv_for_ema():
    """EMA 以 ``named_parameters()`` 为键 ⇒ 新 shadow 必须是 qkv。"""
    net = build_katago_v7_net()
    keys = [n for n, _ in net.named_parameters() if 'qkv' in n]
    assert len(keys) == N_BLOCKS * INNER, len(keys)


# --------------------------------------------------------------------------- #
# 3. 旧 checkpoint 兼容（--resume / --model）
# --------------------------------------------------------------------------- #
def test_legacy_three_key_checkpoint_loads_strict():
    """旧 ckpt（只有 q/k/v）灌进新结构必须 strict=True 通过且逐位一致。"""
    src = build_katago_v7_net()
    legacy = {k: (v.clone() if torch.is_tensor(v) else v)
              for k, v in src.state_dict().items()}
    assert not any('qkv' in k for k in legacy), '前提：state_dict 不该有 qkv'

    dst = build_katago_v7_net()
    dst.load_state_dict(legacy, strict=True)   # 缺 qkv 就得在这里炸

    out = dst.state_dict()
    for k, v in legacy.items():
        assert torch.equal(out[k], v), k


def test_legacy_ema_shadow_is_migrated():
    """旧 ckpt 的 EMA shadow（q/k/v）迁移成 qkv，非 MHSA 键原样透传。"""
    legacy = {}
    for bi in (0, 5):
        for ui in range(INNER):
            for nm in ('q', 'k', 'v'):
                legacy['blocks.%d.inner.%d.attn.%s.weight' % (bi, ui, nm)] = \
                    torch.randn(DIM, DIM)
    legacy['stem.conv.weight'] = torch.randn(8, 32, 3, 3)

    migrated, moved = _migrate_ema_shadow_qkv(legacy)

    assert moved == 4, moved
    assert 'stem.conv.weight' in migrated
    assert torch.equal(migrated['stem.conv.weight'], legacy['stem.conv.weight'])
    assert not [k for k in migrated if k.endswith('.q.weight')], '旧三键应被消费'
    for bi in (0, 5):
        for ui in range(INNER):
            base = 'blocks.%d.inner.%d.attn' % (bi, ui)
            assert base + '.qkv.weight' in migrated
            for i, nm in enumerate(('q', 'k', 'v')):
                seg = migrated[base + '.qkv.weight'][i * DIM:(i + 1) * DIM]
                assert torch.equal(seg, legacy['%s.%s.weight' % (base, nm)])


def test_migrate_is_identity_when_already_fused():
    """已是 qkv 的 shadow 不该被改动（幂等）。"""
    fused = {'blocks.0.inner.0.attn.qkv.weight': torch.randn(3 * DIM, DIM),
             'stem.conv.weight': torch.randn(8, 32, 3, 3)}
    migrated, moved = _migrate_ema_shadow_qkv(fused)
    assert moved == 0
    assert set(migrated) == set(fused)
    assert all(torch.equal(migrated[k], fused[k]) for k in fused)


def test_works_without_torch22_state_dict_post_hook():
    """⚠ 必须能在**没有** `register_state_dict_post_hook` 的 torch 上建网。

    事故：实现最初用 `register_state_dict_post_hook` 展开 q/k/v，本地 torch
    2.12 有这个 API ⇒ 12 条测试全绿；但训练环境是 **torch 2.1.0**（Linux），
    那个 API 不存在 ⇒ `MHSA.__init__` 直接 AttributeError，**整个模型建不起来**，
    一行训练代码都跑不到。

    这条把 2.2+ 的 API 从 `nn.Module` 上摘掉再重建网，把「只能在 2.2+ 上跑」
    这类依赖钉死在测试里，而不是留给真机去发现。
    """
    import torch.nn as _nn
    saved = getattr(_nn.Module, 'register_state_dict_post_hook', None)
    if saved is not None:
        delattr(_nn.Module, 'register_state_dict_post_hook')
    try:
        assert not hasattr(_nn.Module, 'register_state_dict_post_hook')
        m = MHSA(DIM, HEADS)                      # 曾经的崩溃点
        assert 'qkv.weight' not in m.state_dict()
        net = build_katago_v7_net(board_size=19)  # 全网也要能建
        sd = net.state_dict()
        assert not [k for k in sd if 'qkv' in k]
        assert sum(p.numel() for p in net.parameters()) == NBT_TF_CFG['params_total']
    finally:
        if saved is not None:
            _nn.Module.register_state_dict_post_hook = saved


def test_state_dict_expansion_works_when_nested_in_parent():
    """父模块递归时带着 `destination=`/`prefix=` 下来，展开仍必须生效。

    只测 `MHSA(...).state_dict()` 会漏掉这个：独立调用时 prefix 为空，
    而嵌在 `NbtTfNet` 里前缀是 `blocks.N.inner.M.attn.`。之前正是这里静默
    失效（全网还残留 22 个 qkv 键）。
    """
    net = build_katago_v7_net(board_size=19)
    sd = net.state_dict()
    assert not [k for k in sd if 'qkv' in k], '嵌套下 qkv 键没被展开'
    q = [k for k in sd if k.endswith('.attn.q.weight')]
    assert len(q) == N_BLOCKS * INNER, len(q)
    # ⚠ 不能拿全网的 q 键拼 —— 12 通路的 stem 之类也有 `.q.weight` 同名键
    # （`cat` 会得到 2816 行）。按 MHSA 的真实前缀精确定位一个 block。
    # 用 MHSA 自己的 dim（不要写死 DIM：真实配置由 NBT_TF_CFG 决定）
    dim = net.blocks[0].inner[0].attn.dim
    fused = torch.cat([sd['blocks.0.inner.0.attn.%s.weight' % n] for n in ('q', 'k', 'v')],
                      dim=0)
    assert fused.shape == (3 * dim, dim), fused.shape
    assert torch.equal(fused, net.blocks[0].inner[0].attn.qkv.weight.detach())


def test_migrate_passes_stray_q_weight_through_untouched():
    """只有 q 而没有 k/v 时**不拼**，原样透传。

    ⚠ 别把这条读成「多余项要保留到 EMA 报错为止」—— 那是错的：
    `EMA.update()`（`train_sft.py` 的 `EMA` 类）是遍历
    `model.named_parameters()` 再去 shadow 取键，**从不遍历 shadow 自己的键**
    ⇒ 多余项永远不会被读到，既不会算错值也**不会抛 KeyError**。
    真正会 KeyError 的是反方向：「模型有、shadow 没有」。
    真正丢弃多余项的动作在 resume 分支做，且判据是对 `named_parameters`
    求差（那里拿得到 `model`）—— 见 `train_sft.py` 同名注释。
    """
    odd = {'blocks.0.inner.0.attn.q.weight': torch.randn(DIM, DIM)}
    migrated, moved = _migrate_ema_shadow_qkv(odd)
    assert moved == 0
    # 透传 = 本函数不越权判断「模型里没有它」。丢弃是调用点的职责。
    assert set(migrated) == set(odd)


def test_stale_shadow_pruning_never_drops_a_real_q_parameter():
    """**模型里真的有 `.q.weight` 参数时，清理逻辑不许把它当垃圾丢掉。**

    这是把判据从「后缀长得像」改成「对 `named_parameters()` 求差」的原因。
    今天的两个现役模型（V7 278 个 / 12 通道 226 个参数）里 `.q/.k/.v.weight`
    都是 0 个，所以按后缀删**今天也安全** —— 正因为安全，这个 bug 不会有任何
    现存测试能抓到，直到有人给某个模块加一个真的 `.q.weight` 参数那天。
    那时按后缀删掉的就是**活参数**的 shadow 项，`EMA.update()` 第一步直接
    KeyError —— 恰好是它本想防止的故障。
    """
    from scripts.train_sft import EMA, _ema_key

    class _Attn(torch.nn.Module):
        """真的带 `.q/.k/.v.weight` 三个参数（必须**嵌套**，顶层 `self.q`
        的参数名是 `q.weight`，没有前导的点，测不到后缀陷阱）。"""

        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(2, 2)
            self.k = torch.nn.Linear(2, 2)
            self.v = torch.nn.Linear(2, 2)

    class _Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attn()

    net = _Fake()
    ema = EMA(net)
    live = {_ema_key(n) for n, _ in net.named_parameters()}
    assert {'attn.q.weight', 'attn.k.weight', 'attn.v.weight'} <= live, \
        sorted(live)

    # shadow 里既有活键、也有一个模型里不存在的垃圾键
    shadow = dict(ema.shadow)
    shadow['blocks.0.inner.0.attn.q.weight'] = torch.randn(2, 2)
    kept = {k: v for k, v in shadow.items() if k in live}

    assert any(k.endswith('.q.weight') for k in kept), \
        '活着的 .q.weight 被误当成残留删掉了 —— 这就是按后缀匹配的陷阱'
    assert 'blocks.0.inner.0.attn.q.weight' not in kept, \
        '模型里不存在的残留项应当被丢掉'
    # 关键：剩下的是一套完整、可直接挂回去的 shadow
    ema.shadow = kept
    ema.update()          # 缺键 / 多键都不会炸，缺键才会
    ema.apply_shadow()


# --------------------------------------------------------------------------- #
# 4. 训练侧：梯度真的流到 qkv
# --------------------------------------------------------------------------- #
def test_gradients_reach_fused_qkv():
    torch.manual_seed(0)
    net = build_katago_v7_net()
    net.train()
    out = net(torch.randn(2, 22, 19, 19), torch.randn(2, 19))
    out['policy_logits'].float().pow(2).mean().backward()

    qkv = [(n, p) for n, p in net.named_parameters() if 'qkv' in n]
    assert len(qkv) == N_BLOCKS * INNER
    assert all(p.grad is not None for _, p in qkv), '有 qkv 没拿到梯度'
    assert all(p.grad.abs().sum().item() > 0 for _, p in qkv), 'qkv 梯度全零'


def test_init_is_seed_stable_across_split_and_fused():
    """分段 initialize 必须与「整块初始化」同分布，且三段 std 只取决于 in_features。

    这里断言的是**可复现性**：同 seed 下两次构造得到同样的初值。
    """
    torch.manual_seed(1234)
    a = MHSA(DIM, HEADS).initialize()
    torch.manual_seed(1234)
    b = MHSA(DIM, HEADS).initialize()
    assert torch.equal(a.qkv.weight, b.qkv.weight)
    assert a.qkv.weight.std().item() > 0
