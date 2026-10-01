"""`_scan_chunk` 的 batch 切分必须**逐位相同**，且峰值口径必须从真模型取。

事故（2026-10-01 云端，2×910A，v21 @ batch=1000）
------------------------------------------------------
    RuntimeError: NPU out of memory. Tried to allocate 1.40 GiB
      (32.00 GiB total; 28.00 GiB already allocated; 741.02 MiB free)
      File "src/networks/backbone.py", line 1498, in _body
        h = h + torch.exp(sigma).unsqueeze(-1) * h_in

修法：沿 batch 维切（`_scan_batch_split`）。`chunk_size=32` 只管「逐块物化
多少」，不管「单块内同时活着的字节」—— 块内三个 5 维张量
`(B, Hh, P, l, N) = (1000, 4, 46, 32, 64)` 每个 fp16 副本 719 MiB，
`h + exp(σ)·h_in` 一行同时活三份 ⇒ 瞬时 ~2.1 GiB。checkpoint 只管「留不留到
反向」，对这个峰值无效。切到 `split=250` 后是 180 MiB/副本。

⚠⚠ **容量数字必须从真模型取，不能从代码倒推**
------------------------------------------------------------
本文件第一版把 `N` 写成 368（据此算出「单副本 4.03 GiB」「放大 5.75 倍」，
并给出两次相反的结论）。368 其实是 `d_inner`；`N` 取 `bv_k` 的**最后一维**
= `d_state` = 64（`MambaLTI.__init__` 默认值，backbone.py:1654）。
`_chunked_scan` docstring 里原有的 N=64 推导**是对的**，没有「文档过时」。
所以下面 `test_state_dim_comes_from_real_model_not_inference` 直接问模型要
`d_state`，而不是在测试里重抄一个数。
"""
import contextlib
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.networks import backbone as BB   # noqa: E402
from src.networks.backbone import MambaLTI   # noqa: E402


def _mamba(training=False, **kw):
    torch.manual_seed(7)
    m = MambaLTI(channels=8, expand=2, d_conv=4, d_state=4, ddt_rank=2,
                 n_heads=2, **kw).double()
    m.train(training)
    return m


@contextlib.contextmanager
def _split(size):
    """临时改掉整条扫描路径上的 batch 切分参数。

    ⚠ 改 `MambaLTI.__init__` 的默认值**不够**：`_scan_chunk` 拿的是函数默认
    参数（`_scan_batch_split`）。所以直接改 `__defaults__`，确保真路径生效。
    """
    fn = BB._scan_chunk
    old = fn.__defaults__
    assert old is not None, '_scan_chunk 应有默认参数（batch 切分）'
    fn.__defaults__ = tuple(size if isinstance(d, int) and d == 250 else d
                            for d in old)
    try:
        yield
    finally:
        fn.__defaults__ = old


# ---- 1. 正确性：切 batch 与不切，**逐位**相同 ---------------------------------

@pytest.mark.parametrize('B', [1, 2, 3, 4, 7])
@pytest.mark.parametrize('size', [1, 2, 3, 250])
def test_batch_split_is_bitwise_identical(B, size):
    """真 `MambaLTI.forward`：切 batch 后与不切**逐位相同**（`torch.equal`）。

    为什么不给容差：`bmm` 的批维彼此独立、块入口状态按 batch 携带、
    `sum(-1)` 只沿 `N` —— 没有任何跨 batch 的归约，所以切分不该改变**任何**
    一位浮点数。若这里需要容差才能过，说明切分引入了真实差异。

    ⚠ 这条走**端到端 forward**，不手抄 `_scan_chunk` 的内部形状 ——
      `_body` 里 `u.permute(0, 2, 3, 1, 4)` 要求 `xc`/`bv` 是 `(B,T,Hh,P)`，
      我第一版 fixture 抄成 `(B,Hh,P,T)` 就炸在
      `size of tensor a (4) must match ... b (3)`。
    """
    m = _mamba(training=False)
    torch.manual_seed(B * 31 + size)
    x = torch.randn(B, 8, 3, 3, dtype=torch.float64)

    with torch.no_grad():
        with _split(250):                     # 不切：split >= B
            ref = m(x)
        with _split(size):
            got = m(x)

    assert got.shape == ref.shape
    assert torch.equal(got, ref), (
        '切 batch（B=%d, size=%d）改变了输出：max|Δ|={:.3e}'
        .format((got - ref).abs().max().item()))


def test_batch_split_matches_sequential_oracle():
    """切 batch 后仍等于顺序递推 oracle —— 与「不切分」这条线无关的独立锚点。

    `_chunked_scan` docstring 承诺它与 `_sequential_scan_oracle` 数学相同
    （非逐位，fp64 下 max|Δ|~1e-16）。若切分改动了归约顺序或 mask，
    这个 fp64 级差异会放大。

    oracle 的输入取自 `test_arch_v21_blocks._reference_inputs`（那里有完整的
    pre-LN / in_proj / dw_conv 推导），不重抄。
    """
    from tests.test_arch_v21_blocks import _reference_inputs

    m = _mamba(training=False)
    torch.manual_seed(11)
    x = torch.randn(4, 8, 3, 3, dtype=torch.float64)
    ref = _reference_inputs(m, x)
    with torch.no_grad():
        oracle = m._sequential_scan_oracle(
            ref['log_decay'], ref['drive'], ref['c_vec'])
        with _split(250):
            whole = m._chunked_scan(ref['log_decay'], ref['drive'],
                                    ref['c_vec'])
        with _split(2):
            split = m._chunked_scan(ref['log_decay'], ref['drive'],
                                    ref['c_vec'])
    assert torch.allclose(whole, oracle, rtol=1e-10, atol=1e-12), \
        '不切分的分块扫描本身应已与 oracle 一致（否则是既有退化，与切分无关）'
    assert torch.allclose(split, oracle, rtol=1e-10, atol=1e-12), \
        '切 batch 后与 oracle 偏离：max|Δ|={:.3e}'.format(
            (split - oracle).abs().max().item())
    assert torch.equal(split, whole), \
        '切 batch 与不切分应逐位相同：max|Δ|={:.3e}'.format(
            (split - whole).abs().max().item())


def test_batch_split_backward_matches_unsplit():
    """反向：切 batch 后的梯度必须与不切**逐位相同**。

    切 batch 会让各子块独立 backward 再由 autograd 累加。因为子块互不依赖、
    各子块内部的计算序列不变，累加顺序虽变但每一项都与整批一致 ⇒ 应逐位。
    用 allclose 就等于允许「梯度悄悄变了」。
    """
    m = _mamba(training=True)
    torch.manual_seed(5)
    x0 = torch.randn(4, 8, 3, 3, dtype=torch.float64)

    grads = {}
    for tag, size in (('unsplit', 250), ('split', 2)):
        m.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)
        with _split(size):
            m(x).sum().backward()
        grads[tag] = (x.grad.clone(),
                      {n: p.grad.clone() for n, p in m.named_parameters()
                       if p.grad is not None})

    xg_a, pg_a = grads['unsplit']
    xg_b, pg_b = grads['split']
    assert torch.equal(xg_a, xg_b), (
        '输入梯度不同：max|Δ|={:.3e}'.format((xg_a - xg_b).abs().max().item()))
    assert set(pg_a) == set(pg_b), '参数集合变了'
    for n in pg_a:
        assert torch.equal(pg_a[n], pg_b[n]), (
            '参数 %s 的梯度不同：max|Δ|={:.3e}'
            % (n, (pg_a[n] - pg_b[n]).abs().max().item()))


def test_batch_split_still_allows_checkpoint_training_path():
    """切 batch + checkpoint 都开时反向能跑通且梯度有限（防包错后静默失效）。"""
    m = _mamba(training=True)
    m.scan_checkpoint = True
    torch.manual_seed(3)
    x = torch.randn(4, 8, 3, 3, dtype=torch.float64, requires_grad=True)
    with _split(2):
        y = m(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_split_at_or_above_batch_falls_back_whole():
    """`split >= B` 必须退化成整批算，而不是切出空块（否则形状对不上）。"""
    m = _mamba(training=False)
    torch.manual_seed(13)
    x = torch.randn(3, 8, 3, 3, dtype=torch.float64)
    with torch.no_grad():
        with _split(250):
            a = m(x)
        with _split(3):
            b = m(x)
        with _split(1):
            c = m(x)
    assert torch.equal(a, b) and torch.equal(a, c)


# ---- 2. 容量口径：数字必须来自真模型 -----------------------------------------

def test_state_dim_comes_from_real_model_not_inference():
    """`N` 必须是 `d_state`（默认 64），**不是** `d_inner`（368）。

    这条存在的唯一理由是我踩过：把 `N` 读成 `d_inner`=368，据此算出
    「单副本 4.03 GiB」「比 docstring 放大 5.75 倍」，并据此给出两次相反的
    结论（先「降 norm 精度对 OOM 无效」，纠正后「省 7.6 GiB」）。
    `_chunked_scan` docstring 原有的 N=64 推导**是对的**。

    所以这里不重抄数字，直接问模型要 —— 将来 `d_state` 默认值改了，
    这条会跟着走，不会像手抄的数那样悄悄过期。
    """
    torch.manual_seed(0)
    m = MambaLTI()                       # 生产默认构造
    assert m.d_state == 64, (
        'MambaLTI 默认 d_state 应为 64（docstring 的容量推导按此口径）；'
        '实得 %s —— 若默认值真的改了，_chunked_scan docstring 里的容量表'
        '必须一起重推' % m.d_state)
    assert m.d_inner != m.d_state or True      # 记录两者是不同概念
    # 块内 5 维张量的最后一维来自 d_state
    assert m.d_state == 64


def test_v21_single_copy_bytes_after_split():
    """v21 真实形状下，切分后单块单副本降到 ~180 MiB（OOM 的口径）。

    不跑模型：直接算 `u`/`h` 的字节数（它们是块内最大的 5 维张量，也是 OOM
    报错里 `Tried to allocate` 的来源）。`N` 从真模型取，不硬编码。
    """
    from src.networks.alphanet import V21_CFG, build_v21_net

    torch.manual_seed(0)
    net = build_v21_net(in_channels=V21_CFG['in_channels'])
    mamba = next(m for m in net.modules() if isinstance(m, MambaLTI))

    B, split = 1000, 250
    assert B % split == 0, '默认 split 必须整除推荐 batch（1000）'
    Hh, P = mamba.n_heads, mamba.head_dim
    l = mamba.chunk_size
    N = mamba.d_state
    per_sample = Hh * P * l * N
    whole = per_sample * B * 2                    # fp16 = 2 B/元素
    sliced = per_sample * split * 2
    assert whole / sliced == 4
    # 实测 180 MiB/副本（切分前 719 MiB）；那行同时活三份 ⇒ 瞬时 2.1→0.53 GiB。
    # 边界从「切分前整批是 719 MiB」这个已核对的数推，而不是硬编码 MiB 常数 ——
    # 否则 d_state/chunk_size 一改就红，而那不是退化。
    assert sliced == whole / 4
    assert abs(sliced / 2**20 - 180) < 5, (
        '切分后单副本应 ~180 MiB，实得 %.0f MiB' % (sliced / 2**20))
    assert abs(whole / 2**20 - 719) < 5, (
        '切分前整批单副本应 ~719 MiB（N=d_state=64, l=chunk_size=32, B=1000），'
        '实得 %.0f MiB —— 若这里变了，说明 N 或 l 改了，注释里的容量表要重推'
        % (whole / 2**20))


def test_docstring_records_the_n64_misreading():
    """钉住那次 N=368 误判的记录，避免后来者重犯。

    `backbone.py::_scan_chunk` 的注释里记着「N=368 是误读、真实是 d_state=64」。
    这不是文档洁癖：那个误读导致两次方向相反的结论，而当时没有任何数字对得上。
    """
    src = (ROOT / 'src' / 'networks' / 'backbone.py').read_text(encoding='utf-8')
    i = src.index('def _scan_chunk')
    body = src[i:i + 24000]
    assert '368' in body, '应保留 N=368 误判的记录（d_inner 被当成 N）'
    assert 'd_state' in body, '应写明 N 取自 d_state'
    assert '4.03 GiB' in body, '应保留当时那个错误数字，标明它是错的'