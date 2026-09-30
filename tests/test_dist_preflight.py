"""分布式启动自检（`_dist_preflight_check` / `_downgrade_npu_dist_debug`）。

为什么这两件事要有测试：它们守的是「只在多卡启动时发生」的错误，而本仓库的
本地环境 `world_size=1` → `is_dist=False` → **整段代码永不执行**。2026-09-30 的
4 卡事故正是如此：通信域（HCCP）建不起来，报在「第一个 batch 的前向」里，且报成
HCCL 通用错误，真因 `EJ0001 ... Maybe the last training process is running` 藏在
日志更前面 —— 查了 20 分钟。`_dist_preflight_check` 的职责就是把这类问题从
「训练途中」搬到「启动那一刻」，并把处置步骤直接写在异常里。

所以这里的测试策略是：
  · 真跑：起一个 world_size=1 的 **gloo** 进程组（CPU，不需要加速器），
    验证自检真的通过、真的打日志；
  · 假失败：monkeypatch 掉 `dist.all_reduce` 制造异常，断言异常里**带着可执行的
    处置步骤**（残留进程 / 等待 / 逐卡复位）与环境快照 —— 这是这个函数存在的
    全部意义，断言必须落在文本上，否则「报错但没告诉人怎么办」也算通过；
  · 顺序：AST 断言降级发生在 `init_process_group` 之前、自检发生在之后
    （顺序错了等于没做：DETAIL 必须在通信域建立前降，自检必须在建好之后）。
"""
import ast
import os
import pathlib
import socket
import sys

import pytest
import torch
import torch.distributed as dist

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / 'scripts' / 'train_sft.py'
SRC = SRC_PATH.read_text(encoding='utf-8')
TREE = ast.parse(SRC)

sys.path.insert(0, str(ROOT))
from scripts.train_sft import (_dist_debug_level,  # noqa: E402
                               _dist_env_snapshot, _dist_preflight_check,
                               _downgrade_npu_dist_debug)


def _func(name):
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError('train_sft.py 里没有函数 {}'.format(name))


class _Log:
    """够用的 logger 替身：记录 info/warning，便于断言「说了什么」。

    ⚠ 方法名就是 `info` / `warning`（代码按 logging 的形态调），所以记录用的
    列表必须换名字 —— 否则属性会把方法遮蔽，报 `'list' object is not callable`。
    """

    def __init__(self):
        self.lines_info = []
        self.lines_warning = []

    def info(self, msg, *a):
        self.lines_info.append(msg % a if a else msg)

    def warning(self, msg, *a):
        self.lines_warning.append(msg % a if a else msg)

    @property
    def info_lines(self):
        return self.lines_info


@pytest.fixture(scope='module')
def _gloo_pg():
    """world_size=1 的 gloo 进程组；不可用则如实 skip。"""
    if dist.is_initialized():
        yield
        return
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip('本机无 gloo，跳过通信自检测试')
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = str(port)
    try:
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    except Exception as e:  # noqa: BLE001
        pytest.skip('无法初始化 gloo 进程组（跳过）：{}'.format(e))
    yield
    dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# 1. 自检：真跑一遍（world_size=1 gloo / CPU）
# --------------------------------------------------------------------------- #
def test_preflight_passes_and_logs(_gloo_pg):
    log = _Log()
    _dist_preflight_check('gloo', 'cpu', log)
    text = ' '.join(log.lines_info)
    assert '通信域自检通过' in text, '自检通过时必须留一条正向日志：%s' % text
    for token in ('torch=', 'WORLD_SIZE=', 'RANK='):
        assert token in text, '正向日志应带环境快照（缺 %s）：%s' % (token, text)


def test_env_snapshot_is_read_only_and_complete():
    snap = _dist_env_snapshot()
    for token in ('torch=', 'RANK=', 'WORLD_SIZE=', 'MASTER_ADDR='):
        assert token in snap, snap


# --------------------------------------------------------------------------- #
# 2. 自检：失败时必须给出「可执行的处置步骤」
# --------------------------------------------------------------------------- #
def test_preflight_failure_message_carries_remedy(monkeypatch, _gloo_pg):
    """通信建不起来时，异常里必须写清「残留进程 → 等待 → 逐卡复位」三步。"""
    import scripts.train_sft as st

    def _boom(*a, **k):
        raise RuntimeError('[ERROR] HCCL error in: ProcessGroupHCCL.cpp:64')

    monkeypatch.setattr(dist, 'all_reduce', _boom)
    with pytest.raises(RuntimeError) as ei:
        # 设备用 cpu：本机没有 npu，而我们要测的是「all_reduce 抛异常时异常长什么样」
        _dist_preflight_check('hccl', 'cpu', _Log())
    msg = str(ei.value)
    # 真因线索要带（用户看到 HCCL 通用错误时，靠这句知道去翻什么）
    assert 'HCCL error' in msg, msg
    assert 'EJ0001' in msg and 'last training process is running' in msg, \
        '异常里必须复述真正的报错线索（EJ0001 ... last training process）：%s' % msg
    # 处置三步（这是这个函数存在的意义）
    assert 'pkill' in msg, '缺少「杀掉残留进程」的处置：%s' % msg
    assert 'sleep 30' in msg, '缺少「等待 HCCP 清理」的处置：%s' % msg
    assert 'npu-smi info -t reset' in msg, '缺少「逐卡复位」的处置：%s' % msg
    # 环境快照
    assert 'WORLD_SIZE=' in msg and 'LOCAL_RANK=' in msg, msg
    assert st is not None


def test_preflight_detects_wrong_sum(monkeypatch, _gloo_pg):
    """all_reduce「成功」但数值不对（设备被别人占用/结果被污染）也必须报。

    本机是 world_size=1，此时「期望值 = 实得值 = 0」，所以要把 world_size 假成 4
    才能造出偏差 —— 判据本身与卡数无关，它只比「各 rank 求和后的期望值」。
    """
    def _noop(*a, **k):
        return None

    monkeypatch.setattr(dist, 'all_reduce', _noop)
    monkeypatch.setattr(dist, 'get_world_size', lambda *a, **k: 4)
    with pytest.raises(RuntimeError) as ei:
        _dist_preflight_check('hccl', 'cpu', _Log())
    msg = str(ei.value)
    assert '数值不符' in msg, msg


# --------------------------------------------------------------------------- #
# 3. DETAIL 降级：行为 + 位置
# --------------------------------------------------------------------------- #
def test_downgrade_only_touches_detail(monkeypatch):
    log = _Log()
    monkeypatch.setenv('TORCH_DISTRIBUTED_DEBUG', 'DETAIL')
    _downgrade_npu_dist_debug(log)
    assert _dist_debug_level() == 'OFF', 'DETAIL 应被降为 OFF'
    assert any('DETAIL' in m for m in log.lines_info), log.lines_info

    # 非 DETAIL 一律不动（含未设置 / WARN / OFF）
    for value in (None, 'WARN', 'OFF'):
        if value is None:
            monkeypatch.delenv('TORCH_DISTRIBUTED_DEBUG', raising=False)
        else:
            monkeypatch.setenv('TORCH_DISTRIBUTED_DEBUG', value)
        log2 = _Log()
        _downgrade_npu_dist_debug(log2)
        assert _dist_debug_level() == (value or ''), \
            '非 DETAIL 的取值不该被改：%r' % value
        assert not log2.lines_info, \
            '没降级就不该打日志：%s' % log2.lines_info


def test_downgrade_is_before_init_and_preflight_after():
    """顺序即语义：DETAIL 降级在通信域建立**之前**，自检在**之后**。"""
    main = _func('main')
    seg = ast.get_source_segment(SRC, main) or ''
    i_down = seg.find('_downgrade_npu_dist_debug(')
    i_init = seg.find('init_process_group(')
    i_pre = seg.find('_dist_preflight_check(')
    assert -1 not in (i_down, i_init, i_pre), \
        'main 里三步必须都在：downgrade=%d init=%d preflight=%d' % (i_down, i_init, i_pre)
    assert i_down < i_init, 'DETAIL 降级必须早于 init_process_group（否则通信域/FSDP 已按 DETAIL 建好）'
    assert i_pre > i_init, '通信自检必须晚于 init_process_group（要先有通信域才试得起来）'


def test_preflight_is_called_exactly_once():
    """自检只能有一次 —— 每次 forward 试一次会把训练拖死。"""
    main = _func('main')
    calls = [n for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == '_dist_preflight_check']
    assert len(calls) == 1, '通信自检出现 %d 次调用点，应恰好 1 次' % len(calls)


def test_train_sft_still_compiles():
    import subprocess
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(SRC_PATH)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))