"""from-scratch 起手时各 rank 的初始权重必须逐位相同（4 卡 910A 的真 bug）。

事故形状
--------
**已退役的包裹层时代（FSDP1，2026-10-01 换轨前的现场）**：`scripts/train_sft.py`
当时的 `_wrap_fsdp1` docstring 要求 from-scratch 也要同步初始权重：「各 rank 独立、
各自不同，这里必须同步，否则第一步 all-gather 出来的就是拼错的权重」。但
`sync_module_states` 从未被传（`_fsdp_ctor_kwargs` 与 `FSDP_CTOR_KWARGS_WHITELIST`
都没有），**修复前**全文件唯一的 `dist.broadcast` 是 `_sync_stop_flag` 的 stop_flag
—— **没有任何参数广播**。

`shell/train_sft_npu_4card_v21.sh` 不传 `--resume`/`--model` ⇒ from-scratch
⇒ 当时 `SHARD_GRAD_OP` 的第一步 all-gather 把 4 份不同的随机权重拼成一个逻辑权重。

 换轨（2026-10-01，FSDP1 → DDP）**没有作废这个 bug，也没有作废 Task 1 的修复**：
DDP 的 `DistributedDataParallel.__init__` 在**它自己构造时**也会把 rank0 的
params/buffers 广播出去（`_ddp_init_helper` → `_sync_module_states`），而那个构造点
在 EMA 构造**之后** —— 依赖它会让 rank1~3 的 EMA shadow 抓住各自被丢弃的随机权重，
`ema.update()` 每步把正确权重混进陈旧 shadow，**EMA 跨 rank 发散且不报错**。
所以显式广播仍必须留在 EMA 之前，本文件钉的顺序不变量一字未改。

测试分两组：**顺序**靠源码位置（唯一能证明时机的手段，真机上观察不到
「谁先谁后」），**护栏**用 monkeypatch 的假通信域验证真行为（含「checksum 的
**计算**本身」—— 它是启动期硬失败闸门的判据，见
`test_checksum_depends_on_every_parameter`）。

坐标约定
--------
本文件**刻意不写绝对行号**：`scripts/train_sft.py` 每轮都在长，写死的行号下一次
改动就全错（上一版就栽在这上面）。指路一律用「函数名 + 内容锚点」；测试真需要
行号时走 AST 现算的 `_call_lines` / `_main_body_calls`，那两个是**自维护**的。

只有下面这张表必须锚在具体 commit 上 —— 因为「事故现场」指的是一个**历史
commit**，不是当前文件。下次改 `train_sft.py` 后右列会变，**左列不会**。表里
`_wrap_fsdp1` / `FSDP_CTOR_KWARGS_WHITELIST` / `_fsdp_ctor_kwargs` /
`FullyShardedDataParallel` 四个符号已于 2026-10-01 随换轨删除，**保留它们是为了
让下一次事故能查到当时的现场**，不是暗示它们还在：

| 现场                                        | 修复前 3dc9672 | 590f267 |
|---------------------------------------------|---------------:|--------:|
| `_wrap_fsdp1` docstring（「必须同步」那段）   |          :335-339 | :409-425 |
| `FSDP_CTOR_KWARGS_WHITELIST`                 |         :393-396 | :478-481 |
| `_fsdp_ctor_kwargs`                          |         :399-423 |      :484 |
| `_sync_stop_flag` 的 stop_flag 广播            |           :1495 |     :1580 |
| `FullyShardedDataParallel` 唯一裸名字调用      |           :383 |      :468 |

关于「分布式包裹点」为什么不能用**文本搜索**（这一节在换轨前后各踩过一次）
--------------------------------------------------------------------
包裹点是本文件唯一必须锚定的「下游事件」：广播必须早于它。但「它叫什么」换过两次：

* **换轨前（FSDP1 时代）**：`train_sft.py` 里 `FullyShardedDataParallel` 只有一个裸
  名字调用点，且它在 `_wrap_fsdp1` **函数体内**（`handle =
  FullyShardedDataParallel(model, **_ctor_kwargs)`），文本上早于 `def main()`
  （590f267 里 :1934，3dc9672 里 :1849 —— 两者都不是原先 docstring 写的 `L2300+`）。
  而广播点按契约必须落在 `main()` 里 ⇒ 断言 `sync < 那个行号` **恒假**，与同步点放哪
  无关。当时真正代表「包裹发生在这里」的是运行期调用点 `_wrap_fsdp1(...)`。
* **换轨后（DDP，2026-10-01）**：`_wrap_fsdp1` 已删除，包裹点是 `main()` 里的
  `DistributedDataParallel(model, device_ids=[local_rank])`。此时**换成文本搜索同样
  恒假**，而且更隐蔽 —— `DistributedDataParallel` 这个名字在文件里还有两处非调用点：
  顶部那行 `from torch.nn.parallel import DistributedDataParallel`（在 `main()` 之前）
  以及包裹点上方那段解释换轨理由的注释/docstring。用 `SRC.find('DistributedDataParallel')`
  会命中 import（早于 `main()`）⇒ `sync < import 行号` 恒假；用全文计数 / 出现即真
  之类的判据则恒真。**两种写法都不会因为同步点放错而红。**

⇒ 唯一可靠的锚点是 **AST 的 `Call` 节点 + 限定在 `main()` 函数体内**，也就是本文件
既有的 `_main_body_calls`（见下方 §`坐标约定`）。这与 `test_dist_wrap.py` 里
`_ddp_construct_calls()` 的做法一致。

 相对实现计划的一处修正（是**原测试自己不可满足**，不是削弱断言）
------------------------------------------------------------------
`test_broadcast_covers_every_parameter_once` 原本用
`torch.nn.Linear(4, 3, bias=True)` 并注释「3 个参数张量」、断言
`len(seen) == 3`。**`Linear` 只有 2 个**（weight / bias）—— 断言恒假。
换成显式的 3 参数小模块，让「3 个参数张量」这句话成真，并顺带把每次广播的
**形状**一起钉住（原来记录的 `tuple(t.shape)` 从没被断言过，等于白记）。
"""
import ast
import importlib.util
import pathlib
import re
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


def _calls_in(node, fname):
    """`fname(` 作为**裸名字调用**的行号列表（限 `node` 子树内）。"""
    return [n.lineno for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == fname]


def _call_lines(fname):
    """全文件级的裸名字调用行号。"""
    return _calls_in(TREE, fname)


def _main_fn():
    return next(n for n in TREE.body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')


def _main_body_calls(fname):
    """`fname(` 在 **`main()` 函数体内**的裸名字调用行号。

    比 `_call_lines` 窄一档，是刻意的：全文件级的检查会把「同步点被挪进某个辅助
    函数」这种退化放过（helper 里发广播，训练路径根本走不到）。凡是消息里写着
    「main() 里没有调用」的检查，就必须真的是 main() 级。
    """
    return _calls_in(_main_fn(), fname)


class _RecLogger:
    """记录 info/warning 调用的假 logger（避免依赖真实 logger 配置）。"""

    def info(self, msg, *a):
        pass

    def warning(self, msg, *a):
        pass


# ---- 顺序（源码位置）----------------------------------------------------- #

def test_sync_call_precedes_ema_construction():
    """同步点必须早于 `EMA(`：EMA 构造时就把参数 clone 进 shadow。"""
    sync = _main_body_calls('_sync_init_weights_from_rank0')
    assert len(sync) == 1, '同步点应唯一（唯一才能证明时机）：%s' % sync
    ema = _main_body_calls('EMA')
    assert ema, 'main() 里没找到 EMA( 调用点'
    assert sync[0] < min(ema), (
        '_sync_init_weights_from_rank0 在 L%d，但它在 L%d 的 EMA() 之后 —— '
        'EMA 构造时已把**广播前**的随机权重 clone 进 shadow，之后每个 step 的 '
        'ema.update() 都往这个陈旧 shadow 上混 ⇒ EMA 轨迹全程错'
        % (sync[0], min(ema)))


def test_sync_call_precedes_dist_wrap():
    """同步点必须早于分布式包裹（DDP 构造点）。

    锚点取 **`main()` 体内的 `DistributedDataParallel(` AST `Call` 节点**，
    而不是源码文本里的字符串 —— 见本文件 docstring 的「关于『分布式包裹点』」
    一节：文本搜索会命中顶部那行 `from torch.nn.parallel import
    DistributedDataParallel`（在 `main()` 之前 ⇒ 判据恒假）以及包裹点上方的注释
    （⇒ 「出现过就算」类判据恒真）。`_main_body_calls` 走的正是
    `isinstance(n.func, ast.Name) and n.func.id == fname`，且**限定在 `main()`
    子树**，所以 helper 里的引用不会被算进来。

     判据本身（`sync < wrap`）**没有**因为换轨而放宽：2026-10-01 之前这里锚的是
    已删除的 `_wrap_fsdp1(`。之所以还要单独钉这一条（而不只靠
    `sync < EMA < wrap` 传递推出），是因为它是**不依赖其它测试文件**的直接断言 ——
    `EMA < wrap` 那半条在 `tests/test_dist_wrap.py` 里，两边任一被改坏都能定位到。

    换轨后这条为什么仍然有意义（不是因为「参数被展平成 flat shard」了 —— DDP 不
    展平参数）：DDP 在**构造那一刻**做一次性的 params/buffers 广播，且它发生在 EMA
    构造之后。若把本函数挪到包裹之后，就等于把「同步」交给 DDP 自己那次广播 ——
    那时 EMA 的 shadow 已经抓住了各 rank 被丢弃的随机权重 ⇒ EMA 跨 rank 发散、
    不报错（spec §5.3 / §5.4）。
    """
    sync = _main_body_calls('_sync_init_weights_from_rank0')
    assert len(sync) == 1, '同步点应唯一：%s' % sync
    wrap = _main_body_calls('DistributedDataParallel')
    assert wrap, (
        'main() 里没找到 DistributedDataParallel( 的**调用**节点（包裹点不见了？）'
        '—— 注意别改成在源码文本里搜这个名字：会命中 import 行与注释，判据恒假')
    assert sync[0] < min(wrap), (
        '同步点（L%d）必须早于 DDP 包裹（L%d）：DDP 在构造那一刻就把 rank0 的 '
        'params/buffers 广播出去，而那一刻已在 EMA 构造之后 ⇒ 把同步挪到包裹之后就'
        '等于交给 DDP 自带的那次广播，EMA 的 shadow 会抓住各 rank 被丢弃的随机权重'
        % (sync[0], min(wrap)))


def test_sync_and_assert_are_both_called():
    """广播与自检成对出现——只有广播没有自检等于把注释里的承诺又还回去。"""
    assert _main_body_calls('_sync_init_weights_from_rank0'), \
        'main() 里没有调用 _sync_init_weights_from_rank0'
    assert _main_body_calls('_assert_init_weights_identical'), \
        'main() 里没有调用 _assert_init_weights_identical：广播完不验证等于没做'


def test_sync_call_precedes_assert_call():
    """自检必须在广播**之后** —— 拿广播前的 checksum 比对，必然抛。"""
    sync = _main_body_calls('_sync_init_weights_from_rank0')
    chk = _main_body_calls('_assert_init_weights_identical')
    assert sync, 'main() 里没有调用 _sync_init_weights_from_rank0'
    assert chk, 'main() 里没有调用 _assert_init_weights_identical'
    assert min(sync) < min(chk), (
        '自检（L%d）必须在广播（L%d）之后：自检拿的是**本地** checksum 去跨 rank '
        '比对，而广播前各 rank 的随机初始化本来就不同 ⇒ from-scratch 起手必然抛 '
        'RuntimeError，启动即挂。这是个能挡住真实 run 的硬闸门，不是洁癖。'
        % (min(chk), min(sync)))


# ---- 护栏（真行为）------------------------------------------------------- #

def test_dist_active_real_body_decision_table():
    """**真正**的 `_dist_active()`（不 stub 它自己）在各种 PG 状态下的判定。

    其余护栏测试都把 `_dist_active` 整个 stub 成 `lambda: False/True`，于是
    「通信域没建 / 没 torch_npu ⇒ 安全退化」这条全局契约**一次都没被真调过**。
    这里只换掉它依赖的 `mod.dist`，让 `_dist_active` 的函数体真跑一遍判定表。
    """
    mod = _load_module()

    class _PG:
        def __init__(self, available, initialized, world):
            self._a, self._i, self._w = available, initialized, world

        def is_available(self):
            return self._a

        def is_initialized(self):
            return self._i

        def get_world_size(self):
            return self._w

    class _Boom:
        @staticmethod
        def is_available():
            raise RuntimeError('老版本 / 驱动缺失路径')

    # 第 4 行的 `initialized=True` 是**故意的**，别"简化"回 False：`available=False`
    # 与 `initialized=False` 会短路到同一个 False，删掉 `dist.is_available()` 那道门
    # 它照样过 ⇒ 这一行就不判别了。只有把 initialized 置 True、让两道门给出**不同**
    # 的短路结果，删掉 `is_available()` 才会露出来。
    cases = [
        ((True, False, 4), False, 'PG 未建（from-scratch 起手前的那一瞬）'),
        ((True, True, 1), False, 'world_size==1（单卡路径必须 no-op）'),
        ((True, True, 4), True, 'PG 已建且多卡（该广播了）'),
        ((False, True, 4), False, 'torch.distributed 不可用（无 torch_npu 环境）'),
    ]
    real_dist = mod.dist
    try:
        for (a, i, w), want, why in cases:
            mod.dist = _PG(a, i, w)
            got = mod._dist_active()
            assert got is want, (
                '%s：(available=%s, initialized=%s, world=%s) ⇒ %s，期望 %s'
                % (why, a, i, w, got, want))
        mod.dist = _Boom
        assert mod._dist_active() is False, \
            'is_available() 抛异常时必须按「未建域」退化，不能把异常放出去'
    finally:
        mod.dist = real_dist


def test_dist_active_is_false_on_this_test_machine():
    """本机（pytest 单进程、无 4 卡 PG）真调一次必须 False。

    防的是「上面的判定表把某个分支钉死了，但真实 `dist` 上的整体接线是错的」。
    注意本仓库另有测试（`test_dist_preflight.py` / `test_dist_wrap.py`）
    会用 gloo `init_process_group(world_size=1)`，所以这里**只断言结果**、
    不预设「PG 未建」——world_size==1 分支同样必须 False。
    """
    mod = _load_module()
    assert mod._dist_active() is False, (
        '本测试机是单进程、world_size 最多 1，_dist_active() 必须 False；'
        '若为 True，单卡路径会发 collective（= test_noop_when_dist_not_active '
        '要防的事故）')


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
    assert sorted(shape for shape, _ in seen) == sorted([(4,), (3, 5), ()]), (
        '广播的必须正是本模块的三个参数（按形状核对，漏掉/多掉一个都会露出来）：%s'
        % (seen,))


# ---- checksum 的**计算**本身（不是它的比对逻辑）--------------------------- #

class _CapturingDist:
    """记下 `all_gather` 收到的 `buf`（dtype + 值的拷贝），然后让各 rank 一致。

    这一族测试针对的是 checksum 怎么**算出来**的。原先的
    `test_checksum_detects_divergence` / `test_checksum_passes_when_all_ranks_agree`
    两个假 `all_gather` 一个无视 `buf`、一个原样回显，于是 `buf` 的内容从头到尾
    没有任何断言碰过 —— 实测把实现改成「只算最后一个参数」/「`acc + s * 0.0`」
    /「只算第一个参数」/「丢掉 fp64」，7 条测试全绿。自检是启动期**硬失败闸门**，
    判据本身不能没有覆盖。
    """

    def __init__(self):
        self.seen = []

    def as_dist(self):
        return self

    def get_world_size(self):
        return 2

    def all_gather(self, out, buf):
        self.seen.append((buf.dtype, buf.detach().clone()))
        for t in out:
            t.copy_(buf)


def _captured_checksum(mod, model):
    """跑一次 `_assert_init_weights_identical`，返回它送进 all_gather 的 buf。

    顺带 stub 掉 `_dist_env_snapshot`：它在 `train_sft.py` 里**自己**
    `import torch.distributed as dist`，所以 patch `mod.dist` 触不到它，无真实 PG
    时会在 `get_rank()` 上抛 ValueError。成功路径本来走不到它，这里是防御性兜底。
    """
    cap = _CapturingDist()
    real = (mod.dist, mod._dist_active, mod._dist_env_snapshot)
    mod.dist, mod._dist_active = cap.as_dist(), (lambda: True)
    mod._dist_env_snapshot = (lambda: 'SNAPSHOT=<stub>')
    try:
        mod._assert_init_weights_identical(model, _RecLogger())
    finally:
        mod.dist, mod._dist_active, mod._dist_env_snapshot = real
    assert len(cap.seen) == 1, '每次自检只该发一次 all_gather：%s' % len(cap.seen)
    return cap.seen[0]


def test_checksum_depends_on_every_parameter():
    """checksum 必须真由**全部**参数决定，且是**有符号的精确**求和。

    这里断言的是**精确值**而不是「变了」。`_ThreeParam` 基线是全 0 ⇒ checksum 恰好
    `0.0`；只把某一个参数 `add_(-1.0)`，checksum 就必须**恰好**等于该参数的
    `−numel`。逐个参数核对精确值，一次钉住四类退化：

    · 「只算最后一个」/「只算第一个」/「`acc + s * 0.0`」⇒ 实得 0.0，与期望不符；
    · 「`acc + s * 0.5`」⇒ 实得期望值的一半；
    · 「`acc + s.abs()`」⇒ 实得**正**的绝对值。

     偏移量取**负数**是刻意的：`_ThreeParam` 的基线是全 0，若用 `+1.0` 则所有输入
    非负，`abs()` 变成恒等 —— 符号不敏感的 checksum 就混过去了，而它会放过一个
    「权重恰好是 rank0 取负」的 rank。负偏移让 abs()/取负类变异无处躲。

    期望值是 `-numel`（`-4.0 / -15.0 / -1.0`），在 fp32 里同样是**精确整数**，所以用
    `==` 判定、无需容差。
    """
    mod = _load_module()
    dtype, base = _captured_checksum(mod, _ThreeParam())
    # fp32 不是「精度没写对」，是 **910A 没有 fp64**（2026-10-01 云端实测：
    #   `Device do not support double dtype now` + AICPU kernel 挂掉）。见下面
    #   `test_checksum_contract_after_losing_fp64`。期望值 −4 / −15 / −1 在
    #   **fp32 里也是精确整数**，所以下面所有 `==` 判定仍然严格，无需容差。
    assert dtype == torch.float32, (
        'checksum 标量应在 fp32 上累加（fp64 会让 910A 挂 AICPU），实得 %s' % dtype)
    assert base.numel() >= 2, (
        'all_gather 的载荷应至少 2 个 fp32（hi/lo 拆分）才能承载主机侧 fp64 的'
        'checksum，实得 %s' % (base.shape,))
    _sum = lambda t: float(t.double().sum())
    assert _sum(base) == 0.0, (
        '全 0 参数的 checksum 应恰好 0.0（精确，无需容差），实得 %.17g'
        % _sum(base))

    # 参数名 -> 只把它 add_(-1.0) 之后 checksum 应**恰好**变成的值（= −元素数）
    expected = {'a': -4.0, 'b': -15.0, 'c': -1.0}
    for name in sorted(expected):
        want = expected[name]
        m2 = _ThreeParam()
        getattr(m2, name).data.add_(-1.0)          # 只动一个参数，且取**负**偏移
        _, got = _captured_checksum(mod, m2)
        assert _sum(got) == want, (
            '只把 %s（%d 个元素）加 -1.0 时，checksum 应恰好 %.8g，实得 %.17g —— '
            '该参数没进 checksum（漏算 ⇒ 4 卡拼错权重也检测不出来）、权重被折算'
            '（如 *0.5）、或求和不是有符号精确求和（如 .abs()）'
            % (name, -int(want), want, _sum(got)))


def test_checksum_contract_after_losing_fp64():
    """checksum 的真实契约：**逐位相同的权重 ⇒ 逐位相同的 fp32 checksum**。

    这条替换掉原 `test_checksum_accumulates_in_fp64_not_fp32` —— 后者的**全部**
    前提（「必须在 fp64 上累加」）已被硬件证伪。那不是参数没写对，是 910A 压根没有
    fp64（实测报 `Device do not support double dtype now`，并让 AICPU kernel 挂掉）。

    换成 fp32 到底损失了什么，写清楚，别让它藏在换轨里：

    · **不损失**的：本检查要判的是「广播有没有生效」。DDP 构造时 rank0 的参数被
      broadcast 给所有 rank，θ 因此**逐位相同**；相同输入 + 相同形状 + 相同归约
      顺序 ⇒ fp32 的和也逐位相同。检测能力来自「不同 ⇒ 不同」这一侧存在真实差异。
    · **损失**的：**微小**差异的可见性。fp32 只有 24 位尾数，若某 rank 的 θ 与
      rank0 的差值小于「累加到当前量级时的一个 fp32 ulp」，这个检查会漏。原 fp64
      版本在 1e8 量级能分辨 1.0，fp32 在那里 ulp=8、分辨不了。
    · 为什么可以接受：要漏的前提是「广播**部分**生效，或 θ 被后续随机数污染到差
      1 个 ulp 以内」。而广播要么整份生效要么整份没生效；真出现不同，量级也是
      「一次重新随机初始化」那种**整体**差异，不是 1e-8 的微差 —— 那种情况 fp32
      一样抓得到（下面用真实量级验证）。

    `test_checksum_detects_divergence` 钉的是「不同 ⇒ 一定抓到」；这条钉的是
    「相同 ⇒ 一定不误报」，两者缺一不可：前者防漏报，后者防误报。
    """
    mod = _load_module()

    def _checksum_of(t):
        m = torch.nn.Module()
        m.register_parameter('v', torch.nn.Parameter(t.clone()))
        _, got = _captured_checksum(mod, m)
        return float(got.double().sum())

    # v21 真实量级：9.07M 参数、典型初值 ~1e-2 ⇒ checksum 量级 ~1e3~1e8。
    torch.manual_seed(0)
    big = torch.randn(4_000_000) * 1e-2
    a = _checksum_of(big)
    b = _checksum_of(big)                       # 逐位相同的输入
    assert a == b, (
        '逐位相同的权重必须给出逐位相同的 fp32 checksum（这是本检查赖以成立的前提），'
        '实得 %.17g vs %.17g —— 说明归约顺序随调用变化（多线程/原子累加），'
        '跨 rank 比对会变成随机误报' % (a, b))


def test_checksum_is_fp32_because_npu_has_no_fp64():
    """checksum 的累加 dtype 与 all_gather 载荷都必须是 fp32。

    事故记录（2026-10-01 云端，两轮）：
      · 第一轮崩在 `torch.equal(...)` ⇒ 我误判成「比较算子的问题」；
      · 改用 `.item()` 后崩在同一处 ⇒ 说明故障是**异步**的：AICPU 故障由上游的
        `sum(dtype=torch.float64)` 排队，在后面第一次同步时才报出来，栈看起来指向
        无辜的算子。
    真正的线索一直在日志里：`Warning: Device do not support double dtype now`。
    """
    mod = _load_module()
    _, base = _captured_checksum(mod, _ThreeParam())
    assert base.dtype == torch.float32, (
        'all_gather 载荷必须是 fp32（HCCL 原生支持；fp64 会走 AICPU），实得 %s'
        % base.dtype)

    src = (ROOT / 'scripts' / 'train_sft.py').read_text(encoding='utf-8')
    body = src[src.index('def _assert_init_weights_identical'):]
    body = body[:body.index('\ndef ')]
    # 剥掉整行注释：该函数的注释里**故意**提到 `torch.equal`（记录事故与理由），
    # 带着注释判会自己把自己判红。
    body = '\n'.join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith('#'))
    # 约束的准确形状是「**设备侧**不得有 fp64」，不是「代码里不得出现 fp64」。
    # 主机侧 fp64 不受限（`compute_l2_report` 早就在用），而把归约搬到主机恰恰是
    # 2026-10-05 那次 4 卡误报的解药：设备侧 fp32 的噪声底是总量的 1 ulp，
    # 既误报（设备侧归约的分块顺序跨 rank 不一致）又漏报（微小分歧）。
    body = re.sub(r'(?s)("""|\'\'\').*?\1', '', body)
    assert 'dtype=torch.float64, device' not in body, \
        'fp64 不得作为设备张量的 dtype（910A 不支持，会派发 AICPU kernel）'
    assert '.sum(dtype=torch.float64)' in body, (
        'checksum 应在**主机侧**以 fp64 累加（先搬到 cpu 再求和）')
    assert 'sum(dtype=torch.float32)' not in body, \
        '不得在设备张量上直接 sum fp32 —— 分块顺序跨 rank 不一致，会末位误报'
    assert 'torch.equal' not in body, \
        'torch.equal 在 910A 上派发到 AICPU kernel，比对要放到主机侧做'


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
    # `_dist_env_snapshot` 内部**自己** `import torch.distributed as dist`（在
    # `_dist_env_snapshot` 里），所以 monkeypatch `mod.dist` 触不到它；无真实 PG 时
    # 它会在 `get_rank()` 上抛 ValueError，把「自检失败」这个真信号盖掉。这里换掉
    # 它 —— 断言只关心消息里的「初始权重」/「NaN」两处诊断，环境快照是上下文。
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
