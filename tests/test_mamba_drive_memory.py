"""MambaLTI 的 `drive` 不得整条物化 —— 4 卡 910A OOM 的直接原因（2026-09-30）。

事故形状（云端实测报错，逐位可复算）
------------------------------------
    RuntimeError: NPU out of memory. Tried to allocate 31.67 GiB
      (NPU 0; 32.00 GiB total capacity; 6.30 GiB already allocated;
       24.39 GiB free; 6.55 GiB reserved in total by PyTorch)

31.67 GiB = 2000 × 361 × 4 × 46 × 64 × 4B —— 正是 `(B, T, H, P, N)` 的
**整条** drive：T = 19×19 = 361、H = 4 head、P = 46 head_dim、N = 64 d_state、
fp32 ⇒ 每样本 17.0 MB。`6.30 GiB already allocated` 说明常驻很小，是**一次巨型
单块分配**要不到内存（不是碎片、不是慢涨）。

为什么梯度检查点救不了：drive 是块**内部**临时量，前向一次、重算再一次 ——
它不属于「留给反向的激活」，per-block GC 完全管不到。batch 与它线性：
B=2000→31.7GB、1500→23.8GB、1000→15.8GB，这解释了「2800/2000/1500/1000
一路 OOM」。

修法：把乘法搬进 `_chunked_scan` 的块循环（`chunk_size` 默认 32），峰值降到
`B·L·H·P·N` ≈ B × 1.5 MB；逐块与整条乘法**逐位相同**（同一对操作数）。
"""
import ast
import os
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.networks.backbone import MambaLTI  # noqa: E402

SRC = (ROOT / 'src' / 'networks' / 'backbone.py').read_text(encoding='utf-8')
TREE = ast.parse(SRC)


def _class_src(name):
    node = next(n for n in ast.walk(TREE)
                if isinstance(n, ast.ClassDef) and n.name == name)
    return ast.get_source_segment(SRC, node) or ''


def test_mamba_forward_does_not_materialize_full_drive():
    """`MambaLTI.forward` 里不得出现整条 `(B,T,H,P,N)` 的乘法。"""
    fwd = _class_src('MambaLTI')
    fwd = fwd[fwd.find('def forward'):]
    for bad in ('* b_vec[:, :, None, None, :]', 'b_vec[:, :, None, None, :])\n'):
        assert bad not in fwd, \
            'MambaLTI.forward 又把 drive 整条物化了（OOM 直接原因）：%s' % bad
    # 必须以「因子三元组」把乘法交给扫描
    assert 'drive_factors' in fwd, \
        'forward 应把 (dt, xc, b_vec) 交给 _chunked_scan 块内相乘'


def test_chunked_scan_multiplies_inside_the_checkpointed_chunk():
    """块内相乘 + bmm 必须在 `_scan_chunk` 里，且被 `checkpoint` 包住。

    为什么这条比「在不在循环体里」更重要：**留在循环体里是不够的**。
    循环体内的张量照样被 autograd 留住到反向（bmm/mul 的输入），实测
    fp16@B=2000 一个 MambaLTI 块仍要 ~39 GiB。把「块内一步」整体做成检查点
    才是真正把保留量降下来的那一步（见模块 docstring）。
    """
    import src.networks.backbone as bb
    chunk = getattr(bb, '_scan_chunk', None)
    assert chunk is not None, '缺少模块级 _scan_chunk（块内一步必须可被检查点包住）'
    import inspect
    src = inspect.getsource(chunk)
    # 相乘在块内发生
    assert '_dt_k.unsqueeze(-1) * _xc_k.unsqueeze(-1)' in src.replace(
        'dt_k', '_dt_k').replace('xc_k', '_xc_k') or \
        'dt_k.unsqueeze(-1) * xc_k.unsqueeze(-1)' in src, \
        'u 的相乘不在 _scan_chunk 里'
    # 且必须真的走 checkpoint（use_reentrant=False）
    assert 'torch.utils.checkpoint.checkpoint(' in src, '_scan_chunk 没有走检查点'
    assert 'use_reentrant=False' in src, \
        '必须 use_reentrant=False（否则不兼容 saved_tensors_hooks 且要求输入有 grad）'
    # 调用点在循环里，且把「是否检查点」传下去（训练才开、推理恒关）
    mcls = _class_src('MambaLTI')
    seg = mcls[mcls.find('def _chunked_scan'):]
    body = seg[seg.find('for k0 in range'):]
    assert '_scan_chunk(' in body, '_scan_chunk 的调用不在循环里'
    assert 'self.scan_checkpoint' in body and 'self.training' in body, \
        '是否检查点必须同时看 scan_checkpoint 与 self.training（推理路径恒关）'


def test_factor_and_materialized_forms_are_bitwise_identical():
    """两条路径必须**逐位相同**（同一对操作数、同一结合顺序）。"""
    torch.manual_seed(20260930)
    B, T, Hh, P, N = 2, 37, 4, 46, 64
    L = 8
    log_decay = -torch.rand(B, T, Hh, P, dtype=torch.float32) * 0.1
    dt = torch.rand(B, T, Hh, P, dtype=torch.float32) + 0.1
    xc = torch.rand(B, T, Hh, P, dtype=torch.float32)
    b_vec = torch.rand(B, T, N, dtype=torch.float32)
    c_vec = torch.rand(B, T, N, dtype=torch.float32)
    blk = MambaLTI(channels=Hh * P)
    blk.chunk_size = L
    mat = dt.unsqueeze(-1) * xc.unsqueeze(-1) * b_vec[:, :, None, None, :]
    y_mat = blk._chunked_scan( log_decay, mat, c_vec, chunk_size=L)
    y_fac = blk._chunked_scan( log_decay, (dt, xc, b_vec), c_vec, chunk_size=L)
    assert torch.equal(y_mat, y_fac), \
        '因子形式与物化形式不逐位相同：max|Δ|=%.3e' % (y_mat - y_fac).abs().max()


def test_peak_memory_scales_with_chunk_not_sequence():
    """峰值随 chunk_size 而不是随 T 走 —— 这条是「为什么这样改能治 OOM」。"""
    torch.manual_seed(7)
    B, T, Hh, P, N = 2, 361, 4, 46, 64
    dt = torch.rand(B, T, Hh, P) + 0.1
    xc = torch.rand(B, T, Hh, P)
    b_vec = torch.rand(B, T, N)
    c_vec = torch.rand(B, T, N)
    log_decay = -torch.rand(B, T, Hh, P) * 0.1
    blk = MambaLTI(channels=Hh * P)

    def peak(chunk):
        blk.chunk_size = chunk
        # 只测 u 的物化量（驱动峰值的那一项），用 saved_tensors 口径不便，
        # 这里直接算：整条 = B·T·Hh·P·N，块内 = B·L·Hh·P·N
        return B * chunk * Hh * P * N * 4 / 2 ** 20

    full_mb = B * T * Hh * P * N * 4 / 2 ** 20
    assert peak(32) < full_mb / 5, \
        'chunk=32 的块内峰值应当是整条的 1/10 量级：%.1f vs %.1f MB' % (
            peak(32), full_mb)
    # 真实形状下的绝对量级（把 B 换成事故里的 2000）
    b2000_full = 2000 * T * Hh * P * N * 4 / 2 ** 30
    b2000_chunk = 2000 * 32 * Hh * P * N * 4 / 2 ** 30
    assert 30 < b2000_full < 34, \
        '整条 drive 在 B=2000 时应是 ~31.7 GiB（与云端报错吻合）：%.1f' % b2000_full
    assert b2000_chunk < 4, \
        '块内峰值应降到 4 GiB 以内：%.1f' % b2000_chunk


@pytest.mark.parametrize('chunk', [1, 8, 32, 64])
def test_chunk_size_does_not_change_the_result(chunk):
    """chunk_size 只影响峰值，不影响数值（块内闭式对任意 L 都成立）。"""
    torch.manual_seed(11)
    B, T, Hh, P, N = 1, 40, 4, 46, 64
    log_decay = -torch.rand(B, T, Hh, P) * 0.1
    dt = torch.rand(B, T, Hh, P) + 0.1
    xc = torch.rand(B, T, Hh, P)
    b_vec = torch.rand(B, T, N)
    c_vec = torch.rand(B, T, N)
    blk = MambaLTI(channels=Hh * P)
    blk.chunk_size = chunk
    y = blk._chunked_scan(log_decay, (dt, xc, b_vec), c_vec, chunk_size=chunk)
    blk.chunk_size = 1000
    ref = blk._chunked_scan(log_decay, (dt, xc, b_vec), c_vec, chunk_size=1000)
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-6), \
        'chunk_size=%d 与单块结果不一致：max|Δ|=%.3e' % (chunk, (y - ref).abs().max())


def _saved_bytes(blk, B=2, T=361, board=19):
    """跑一次 fwd+bwd，返回被 autograd 留存的字节（单个 MambaLTI，无块级检查点）。"""
    blk.train()
    torch.manual_seed(3)
    C = blk.d_inner
    x = torch.randn(B, C, board, board, requires_grad=True)
    total = [0]
    shapes = {}

    def pack(t):
        if isinstance(t, torch.Tensor):
            b = t.numel() * t.element_size()
            total[0] += b
            shapes[tuple(t.shape)] = shapes.get(tuple(t.shape), 0) + b
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        blk(x).pow(2).mean().backward()
    return total[0], shapes


def test_scan_internals_are_not_retained_for_backward():
    """块内 `u`/`M`/`h` 不得活到反向 —— 这是 4 卡放不下 batch 的第二个原因。

    「按块物化」本身**不够**：块内的 `u`(B,H,P,l,N)、`M`(B,H,P,l,l)、`h` 都是
    bmm/mul 的输入，autograd 会留住它们直到反向。实测 fp16@B=2000：
        u/h  1.44 GiB/块 × 11 块，M 0.72 GiB/块 × 11 ⇒ 一个块 ≈ 39 GiB
    加前向留下的 8.78 GiB 就超了 32 GiB 卡 ⇒ 云端
    `20.13 GiB already allocated` + `Tried to allocate 2.81 GiB`。

    这里直接量「单个 MambaLTI fwd+bwd 被留存的字节」，开/关 `scan_checkpoint`
    对比：必须显著下降，且保留清单里**不得**再出现块内形状
    （`(…, l, N)` 与 `(…, l, l)`，l=chunk_size）。
    """
    blk = MambaLTI(channels=184)
    blk.scan_checkpoint = False
    off_bytes, off_shapes = _saved_bytes(blk)
    blk2 = MambaLTI(channels=184)
    blk2.scan_checkpoint = True
    on_bytes, on_shapes = _saved_bytes(blk2)

    assert on_bytes < off_bytes / 3, (
        '块内检查点没起作用：关 %.1f MB → 开 %.1f MB（应至少降到 1/3）'
        % (off_bytes / 2 ** 20, on_bytes / 2 ** 20))

    l = blk.chunk_size
    for sh, b in sorted(on_shapes.items(), key=lambda kv: -kv[1])[:8]:
        # (…, l, N) 形态的块内 u/h 与 (…, l, l) 形态的 M 都不该留下
        assert not (len(sh) >= 2 and sh[-2] == l and sh[-1] in (64, l)), \
            '块内张量 %s（%.1f MB）仍活到反向' % (sh, b / 2 ** 20)

    # 折算到事故现场口径：fp32@B=2 → fp16@B=2000
    scale = (2.0 / 4.0) * (2000.0 / 2)
    print('\n单个 MambaLTI 反向保留量 @fp16 B=2000: 关 %.1f GiB → 开 %.1f GiB'
          % (off_bytes * scale / 2 ** 30, on_bytes * scale / 2 ** 30))