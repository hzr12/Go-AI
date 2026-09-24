"""SwanLab 上报频率与日志同步点测试。

两个问题：
1. `--log-every` 同时控制 stdout 和 swanlab（train_sft.py 同一 if 分支），
   曲线只有 log_every 的密度——log-every 50 时一个 epoch 仅约 200 个点，
   看不出细节。新增 `--swanlab-every`（默认 0 = 跟随 log_every）把两者解耦。
2. 同一打点里 loss/policy_loss/value_loss 各被 `.item()` 取了两次
   （一次给 stdout、一次给 swanlab），共 6 次设备同步，其中 3 次完全重复。
   改为单一同步点后每打点只同步 3 次，且两处用的是同一份值。

本测试覆盖频率决策、单同步点、以及若干结构性不变量。
"""
import inspect
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_sft import _read_log_scalars, _should_log


# --------------------------------------------------------------------------- #
# 频率决策：swanlab_every=0 跟随 log_every
# --------------------------------------------------------------------------- #
def test_zero_swanlab_every_follows_log_every():
    """默认 0：stdout 与 swanlab 频率完全一致（与改动前行为相同）。"""
    for step in range(1, 201):
        do_out, do_swan = _should_log(step, log_every=50,
                                      swanlab_every=0, swanlab_on=True)
        assert do_out == (step % 50 == 0)
        assert do_swan == do_out, f"step={step}: swanlab 应跟随 stdout"


def test_swanlab_every_decoupled_from_stdout():
    """swanlab_every=5、log_every=50：swanlab 密 10 倍，stdout 不变。"""
    swan_steps = [s for s in range(1, 501)
                  if _should_log(s, 50, 5, True)[1]]
    out_steps = [s for s in range(1, 501)
                 if _should_log(s, 50, 5, True)[0]]
    assert len(out_steps) == 10, f"stdout 应仍为 10 次，实得 {len(out_steps)}"
    assert len(swan_steps) == 100, f"swanlab 应为 100 次，实得 {len(swan_steps)}"
    assert all(s % 50 == 0 for s in out_steps)


def test_swanlab_off_never_logs():
    """swanlab_logger 为 None 时，上报开关恒为 False（零开销）。"""
    for step in range(1, 101):
        assert _should_log(step, 50, 1, False)[1] is False


def test_swanlab_every_one_logs_every_step():
    """swanlab_every=1 时每步都该上报（stdout 仍按 log_every）。"""
    swan = [s for s in range(1, 51) if _should_log(s, 50, 1, True)[1]]
    out = [s for s in range(1, 51) if _should_log(s, 50, 1, True)[0]]
    assert len(swan) == 50
    assert out == [50]


# --------------------------------------------------------------------------- #
# 单一同步点：三个张量各取一次
# --------------------------------------------------------------------------- #
class _CountingTensor:
    """记录 .item() 调用次数的张量替身。"""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def item(self):
        self.calls += 1
        return self.value


def test_read_log_scalars_syncs_each_tensor_exactly_once():
    """每打点 loss/policy_loss/value_loss 各只 .item() 一次（3 次同步，非 6 次）。"""
    a, b, c = _CountingTensor(1.5), _CountingTensor(0.25), _CountingTensor(0.125)
    got = _read_log_scalars(a, b, c)
    assert a.calls == 1, f"loss 被取 {a.calls} 次"
    assert b.calls == 1, f"policy_loss 被取 {b.calls} 次"
    assert c.calls == 1, f"value_loss 被取 {c.calls} 次"
    assert got == (1.5, 0.25, 0.125)


def test_read_log_scalars_returns_plain_floats():
    """返回值是普通 float（供 %-格式化与 swanlab 字典直接使用）。

    真实的 torch 浮点张量 .item() 就返回 python float，这里用 float 替身对齐该语义。
    """
    a, b, c = _CountingTensor(3.0), _CountingTensor(2.0), _CountingTensor(1.0)
    vals = _read_log_scalars(a, b, c)
    assert all(isinstance(v, float) for v in vals)


# --------------------------------------------------------------------------- #
# 结构性不变量
# --------------------------------------------------------------------------- #
def test_main_uses_helpers_and_no_longer_reads_tensors_twice():
    """main() 的打点应走上述两个 helper，stdout 与 swanlab 共用同一份标量。"""
    import scripts.train_sft as t
    src = inspect.getsource(t.main)
    assert '_should_log(' in src, 'main() 未使用 _should_log 做频率决策'
    assert '_read_log_scalars(' in src, 'main() 未使用 _read_log_scalars 做单同步'

    # 打点区域内 .item() 只应出现在 _read_log_scalars 的调用处
    log_region = src[src.find('_should_log('):]
    log_region = log_region[:log_region.find('if _prof_ctx')]
    assert log_region.count('.item()') == 0, \
        '打点区域不应再有裸 .item()（应全部经 _read_log_scalars）'


def test_swanlab_log_uses_cached_scalars():
    """swanlab 上报必须用缓存的标量变量，而不是重新取张量。"""
    import scripts.train_sft as t
    src = inspect.getsource(t.main)
    log_start = src.find('swanlab_logger.log(')
    assert log_start != -1, 'main() 未调用 swanlab_logger.log'
    region = src[log_start:log_start + 400]
    assert '.item()' not in region, 'swanlab.log 内不得再取 .item()'


def test_epoch_loss_removed():
    """死变量 epoch_loss 已删除（grep 证实原本从未被读取，且注释误导）。"""
    src_path = os.path.join(ROOT, 'scripts', 'train_sft.py')
    src = open(src_path, encoding='utf-8').read()
    assert 'epoch_loss' not in src, 'epoch_loss 死变量仍未删除'


def test_swanlab_every_arg_exists_with_default_zero():
    """--swanlab-every 必须是合法参数且默认 0（保持现状）。"""
    import argparse
    import re
    import scripts.train_sft as t
    src = inspect.getsource(t.main)
    m = re.search(r'add_argument\(\s*[\'"]--swanlab-every[\'"].*?default=([^\s,]+)',
                  src, re.S)
    assert m, 'train_sft.py 缺少 --swanlab-every'
    assert m.group(1) == '0', f"--swanlab-every 默认应为 0，实得 {m.group(1)}"


def test_swanlab_every_logged_in_config():
    """启动日志应打印 swanlab 上报频率，便于确认加密是否生效。"""
    import scripts.train_sft as t
    src = inspect.getsource(t.main)
    i = src.find('SwanLab: 启用=')
    assert i != -1, 'main() 未打印 SwanLab 配置状态'
    stmt = src[i:i + 500]
    assert 'swanlab_every' in stmt, 'SwanLab 状态日志未包含 --swanlab-every'
    assert 'log_every' in stmt, 'SwanLab 状态日志未说明与 --log-every 的关系'


# --------------------------------------------------------------------------- #
# swanlab_logger 落地方式（回归：曾因把 import 抽走而留下未定义名）
# --------------------------------------------------------------------------- #
def test_main_logs_through_swanlab_logger_not_bare_swanlab():
    """main() 里的记录必须走 swanlab_logger，不能直接引用 `swanlab`。

    回归：把 `import swanlab` 抽进 _init_swanlab 后，main() 中残留的
    `swanlab.log(...)` 会变成未定义名——平时被 `if swanlab_logger is not None`
    挡住看不出来，但**一旦 swanlab 真能启用，首个打点就会 NameError 崩溃**。
    """
    import scripts.train_sft as t
    src = inspect.getsource(t.main)
    assert 'swanlab.log(' not in src, \
        "main() 不应直接引用 swanlab.log（swanlab 在此作用域未定义）"
    assert 'swanlab.finish(' not in src, \
        "main() 不应直接引用 swanlab.finish（同上）"
    assert 'swanlab_logger.log(' in src, "main() 应通过 swanlab_logger 记录"
    assert 'swanlab_logger.finish(' in src, "main() 应通过 swanlab_logger 收尾"
    # 确认每个 log 调用都在 None 守卫之内。守卫有两种等价写法：
    #   · if swanlab_logger is not None:   —— 直接判空
    #   · if _do_swanlab:                  —— 经 _should_log 判空（内部含
    #                                         `swanlab_on = swanlab_logger is not None`）
    # 后者见 test_swanlab_off_never_logs 验证其确实含判空语义。
    src_lines = src.splitlines()
    for i, line in enumerate(src_lines):
        if 'swanlab_logger.log(' in line or 'swanlab_logger.finish(' in line:
            window = '\n'.join(src_lines[max(0, i - 14):i + 1])
            assert ('swanlab_logger is not None' in window
                    or 'if _do_swanlab:' in window), \
                f'swanlab_logger 调用缺少 None 守卫: {line.strip()}'


def test_init_returns_none_on_failure():
    """初始化失败必须返回 None（而非抛出），保证训练不被跟踪功能拖垮。"""
    import logging
    import sys as _sys
    import types as _types
    import scripts.train_sft as t

    class _Args:
        swanlab_api_key = ''
        board_size = 19
        ver = 'v1'
        backbone_channels = 1
        backbone_res_blocks = 1
        res_blocks = 1
        convnext_blocks = 1
        attn_blocks = 1
        value_channels = 1
        value_res_blocks = 1
        policy_channels = 1
        policy_layers = 1
        batch_size = 1
        lr = 1.0
        epochs = 1

    fake = _types.ModuleType('swanlab')

    def _boom(*a, **k):
        raise RuntimeError('boom')

    fake.init = _boom
    old = _sys.modules.get('swanlab')
    _sys.modules['swanlab'] = fake
    try:
        assert t._init_swanlab(_Args(), logging.getLogger('test')) is None
    finally:
        if old is None:
            _sys.modules.pop('swanlab', None)
        else:
            _sys.modules['swanlab'] = old


def test_dns_precheck_removed():
    """DNS 预检已按决策移除（不再有 socket 依赖与 getaddrinfo 探测）。"""
    src = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()
    assert '_swanlab_reachable' not in src, 'swanlab DNS 预检仍未移除'
    assert 'getaddrinfo' not in src, '仍残留 DNS 探测调用'
    assert '\nimport socket' not in src, 'socket import 仍未移除'
