"""from-scratch 起手时各 rank 的初始权重必须逐位相同（4 卡 910A 的真 bug）。

事故形状
--------
`scripts/train_sft.py:335-339` 的 `_wrap_fsdp1` docstring 要求 from-scratch 也要
同步初始权重：「各 rank 独立、各自不同，这里必须同步，否则第一步 all-gather
出来的就是拼错的权重」。但 `sync_module_states` 从未被传（`_fsdp_ctor_kwargs`
在 :399-423，白名单在 :393-396），全文件唯一的 `dist.broadcast` 是
`_sync_stop_flag` 的 stop_flag（:1495）——**没有任何参数广播**。

`shell/train_sft_npu_4card_v21.sh` 不传 `--resume`/`--model` ⇒ from-scratch
⇒ `SHARD_GRAD_OP` 的第一步 all-gather 把 4 份不同的随机权重拼成一个逻辑权重。

测试分两组：**顺序**靠源码位置（唯一能证明时机的手段，真机上观察不到
「谁先谁后」），**护栏**用 monkeypatch 的假通信域验证「权重不同 ⇒ 真的抛」。

⚠ 相对实现计划的两处修正（两处都是**原测试自己不可满足**，不是削弱断言）
----------------------------------------------------------------------
1. `test_sync_call_precedes_fsdp_wrap` 原本取 `_call_lines('FullyShardedData
   Parallel')`。本文件里那个符号只有一个裸名字调用点 —— L383，在
   `_wrap_fsdp1` **函数体内**（`handle = FullyShardedDataParallel(model,
   **_ctor_kwargs)`），它在文本上早于 `main()`（L2300+）里任何东西。而广播点
   按契约必须落在 `main()` 的 `.to(device)` 之后 ⇒ 断言 `sync < 383` 恒假，
   与「同步点位置」无关。真正代表「FSDP 包裹发生在这里」的是**运行期**的调用
   点 `_wrap_fsdp1(...)`（L2615，在 `main()` 里、在广播点之后）。改用它之后
   断言钉的仍是原意（同步点早于 FSDP 包裹），且不再是恒假。
2. `test_broadcast_covers_every_parameter_once` 原本用
   `torch.nn.Linear(4, 3, bias=True)` 并注释「3 个参数张量」、断言
   `len(seen) == 3`。**`Linear` 只有 2 个**（weight / bias）—— 断言恒假。
   换成显式的 3 参数小模块，让「3 个参数张量」这句话成真，并顺带把每次广播的
   **形状**一起钉住（原来记录的 `tuple(t.shape)` 从没被断言过，等于白记）。
"""
import ast
import importlib.util
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

SRC_PATH = ROOT / 'scripts' / 'train_sft.py'
SRC = SRC_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SRC)


def _load_module():
    spec = importlib.util.spec_from_file_location('_train_sft_sync_probe', SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_lines(fname):
    """返回 `fname(` 作为**裸名字调用**的行号列表。"""
    return [n.lineno for n in ast.walk(TREE)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == fname]


class _RecLogger:
    """记录 info/warning 调用的假 logger（避免依赖真实 logger 配置）。"""

    def info(self, msg, *a):
        pass

    def warning(self, msg, *a):
        pass


# ---- 顺序（源码位置）----------------------------------------------------- #

def test_sync_call_precedes_ema_construction():
    """同步点必须早于 `EMA(`：EMA 构造时就把参数 clone 进 shadow。"""
    sync = _call_lines('_sync_init_weights_from_rank0')
    assert len(sync) == 1, '同步点应唯一（唯一才能证明时机）：%s' % sync
    ema = _call_lines('EMA')
    assert ema, '没找到 EMA( 调用点'
    assert sync[0] < min(ema), (
        '_sync_init_weights_from_rank0 在 L%d，但它在 L%d 的 EMA() 之后 —— '
        'EMA 构造时已把**广播前**的随机权重 clone 进 shadow，之后每个 step 的 '
        'ema.update() 都往这个陈旧 shadow 上混 ⇒ EMA 轨迹全程错'
        % (sync[0], min(ema)))


def test_sync_call_precedes_fsdp_wrap():
    """同步点必须早于 FSDP 包裹：包裹后参数存储被替换成 flat shard。

    取 `_wrap_fsdp1(` 而不是 `FullyShardedDataParallel(`：后者在本文件里只出现
    在 `_wrap_fsdp1` **函数体内**（构造那一行，文本上早于 `main()`），拿它当
    「包裹发生在这里」的代理是恒假的判据；`_wrap_fsdp1(` 才是 `main()` 里真正
    发生包裹的位置。见本文件 docstring 的 ⚠ 第 1 条。
    """
    sync = _call_lines('_sync_init_weights_from_rank0')
    assert len(sync) == 1, '同步点应唯一：%s' % sync
    wrap = _call_lines('_wrap_fsdp1')
    assert wrap, '没找到 _wrap_fsdp1( 调用点'
    assert sync[0] < min(wrap), (
        '同步点（L%d）必须早于 FSDP 包裹（L%d）：包裹后参数存储被替换成 flat '
        'shard，broadcast 无意义' % (sync[0], min(wrap)))


def test_sync_and_assert_are_both_called():
    """广播与自检成对出现——只有广播没有自检等于把注释里的承诺又还回去。"""
    assert _call_lines('_sync_init_weights_from_rank0'), \
        'main() 里没有调用 _sync_init_weights_from_rank0'
    assert _call_lines('_assert_init_weights_identical'), \
        'main() 里没有调用 _assert_init_weights_identical：广播完不验证等于没做'


# ---- 护栏（真行为）------------------------------------------------------- #

def test_noop_when_dist_not_active():
    """world_size==1 / 通信域未建时是 no-op，单卡路径逐字不变。"""
    mod = _load_module()
    m = torch.nn.Linear(3, 3)
    sent = []

    class _Boom:
        @staticmethod
        def broadcast(*a, **kw):
            sent.append('broadcast')
            raise AssertionError('通信域没建却发了 collective')

        @staticmethod
        def all_gather(*a, **kw):
            sent.append('all_gather')
            raise AssertionError('通信域没建却发了 collective')

    real_dist, real_active = mod.dist, mod._dist_active
    mod.dist, mod._dist_active = _Boom, (lambda: False)
    try:
        mod._sync_init_weights_from_rank0(m, _RecLogger())
        mod._assert_init_weights_identical(m, _RecLogger())
    finally:
        mod.dist, mod._dist_active = real_dist, real_active
    assert sent == [], 'no-op 路径发了 collective：%s' % sent


class _ThreeParam(torch.nn.Module):
    """恰好 3 个参数张量（`Linear` 只有 weight/bias 两个，凑不出 3）。

    形状故意互不相同（标量 / 一维 / 二维），这样「每个参数恰好广播一次」可以
    顺带按**形状**核对，而不只是数个数。
    """

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(4))
        self.b = torch.nn.Parameter(torch.zeros(3, 5))
        self.c = torch.nn.Parameter(torch.zeros(()))


def test_broadcast_covers_every_parameter_once():
    """每个参数恰好广播一次，且 src 恒为 0。"""
    mod = _load_module()
    m = _ThreeParam()                                # 3 个参数张量
    seen = []

    class _FakeDist:
        @staticmethod
        def broadcast(t, src=0):
            seen.append((tuple(t.shape), src))

        @staticmethod
        def get_world_size():
            return 4

    real_dist, real_active = mod.dist, mod._dist_active
    mod.dist, mod._dist_active = _FakeDist, (lambda: True)
    try:
        mod._sync_init_weights_from_rank0(m, _RecLogger())
    finally:
        mod.dist, mod._dist_active = real_dist, real_active
    assert len(seen) == 3, '每个参数应恰好广播一次，实得 %d 次：%s' % (len(seen), seen)
    assert all(src == 0 for _, src in seen), 'src 必须恒为 0：%s' % seen
    assert sorted(shapes for shapes, _ in seen) == sorted([(4,), (3, 5), ()]), (
        '广播的必须正是本模块的三个参数（按形状核对，漏掉/多掉一个都会露出来）：%s'
        % (seen,))


def test_checksum_detects_divergence():
    """各 rank checksum 不一致 ⇒ 抛 RuntimeError（消息含两种可能的原因）。"""
    mod = _load_module()
    m = torch.nn.Linear(4, 4)

    class _Divergent:
        @staticmethod
        def get_world_size():
            return 3

        @staticmethod
        def all_gather(out, buf):
            for i, t in enumerate(out):
                t.fill_(1.0 + float(i))

    real_dist, real_active = mod.dist, mod._dist_active
    real_snap = mod._dist_env_snapshot
    mod.dist, mod._dist_active = _Divergent, (lambda: True)
    # `_dist_env_snapshot` 内部**自己** `import torch.distributed as dist`（本文件
    # L164-165），所以 monkeypatch `mod.dist` 触不到它；无真实 PG 时它会在
    # `get_rank()` 上抛 ValueError，把「自检失败」这个真信号盖掉。这里换掉它 ——
    # 断言只关心消息里的「初始权重」/「NaN」两处诊断，环境快照是上下文。
    mod._dist_env_snapshot = (lambda: 'SNAPSHOT=<stub>')
    try:
        with pytest.raises(RuntimeError) as ei:
            mod._assert_init_weights_identical(m, _RecLogger())
    finally:
        mod.dist, mod._dist_active = real_dist, real_active
        mod._dist_env_snapshot = real_snap
    msg = str(ei.value)
    assert '初始权重' in msg, msg
    assert 'NaN' in msg, (
        '消息必须提示第二种可能：`torch.equal` 对 NaN 返回 False，会把'
        '「初始权重里有 NaN/Inf」误报成「权重不同」：%s' % msg)


def test_checksum_passes_when_all_ranks_agree():
    """checksum 一致时不抛——护栏不误伤。"""
    mod = _load_module()
    m = torch.nn.Linear(4, 4)

    class _Agreeing:
        @staticmethod
        def get_world_size():
            return 4

        @staticmethod
        def all_gather(out, buf):
            for t in out:
                t.copy_(buf)

    real_dist, real_active = mod.dist, mod._dist_active
    mod.dist, mod._dist_active = _Agreeing, (lambda: True)
    try:
        mod._assert_init_weights_identical(m, _RecLogger())
    finally:
        mod.dist, mod._dist_active = real_dist, real_active
