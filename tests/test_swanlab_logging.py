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


# --------------------------------------------------------------------------- #
# 禁止在训练进程内 pip install swanlab
# --------------------------------------------------------------------------- #
_TRAIN_SCRIPTS = ('train_sft.py', 'selfplay_train.py')


@pytest.mark.parametrize('fname', _TRAIN_SCRIPTS)
def test_no_inprocess_pip_install(fname):
    """训练脚本不得在进程内 `pip install swanlab`。

    原因：调用点位于 torch / torch_npu **已加载之后**，此时改动 site-packages
    可能破坏后续惰性导入；且 shell/*.sh 在启动 python 前已装过一次，那次失败
    的话这里必然也失败，只是白等一轮。手动装（进程启动前）没问题正是此因。
    """
    src = open(os.path.join(ROOT, 'scripts', fname), encoding='utf-8').read()
    # 只匹配真正调用 pip 的 argv 形式，不误伤「请执行 pip install swanlab」这类提示文本
    assert "'-m', 'pip'" not in src, f'{fname} 仍在训练进程内 pip install swanlab'
    assert '"-m", "pip"' not in src, f'{fname} 仍在训练进程内 pip install swanlab'
    assert 'check_call' not in src, f'{fname} 仍在训练进程内 pip install swanlab'


def _swanlab_present(spec_result):
    """把 importlib.util.find_spec('swanlab') 固定为指定返回值。"""
    import importlib.util
    return lambda name: spec_result


def test_absent_swanlab_degrades_with_actionable_message(caplog):
    """swanlab 真的没装时：返回 None，且提示如何安装（而不是自己装）。"""
    import logging
    import types as _types
    import scripts.train_sft as t

    class _Args:
        swanlab_api_key = ''
        board_size = 19
        ver = 'v1'
        backbone_channels = backbone_res_blocks = res_blocks = 1
        convnext_blocks = attn_blocks = value_channels = 1
        value_res_blocks = policy_channels = policy_layers = 1
        batch_size = 1
        lr = 1.0
        epochs = 1

    saved = {m: sys.modules.pop(m) for m in list(sys.modules) if m == 'swanlab'}
    import importlib.util
    real_find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name: None
    logger = logging.getLogger('test_absent')
    try:
        with caplog.at_level(logging.DEBUG, logger='test_absent'):
            assert t._init_swanlab(_Args(), logger) is None
    finally:
        importlib.util.find_spec = real_find_spec
        for m in saved:
            sys.modules[m] = saved[m]
        assert not [m for m in sys.modules if m == 'swanlab']

    text = caplog.text
    assert 'pip install swanlab' in text, f'未给出安装指引: {text!r}'
    assert '自动安装' not in text, f'不应再声称会自动安装: {text!r}'


def test_broken_swanlab_reports_real_cause_not_missing(tmp_path, caplog):
    """装了但坏（pydantic 1.x 缺 TypeAdapter）时，要报真实原因而非「未安装」。

    这是云端实际踩到的坑：swanlab 依赖 pydantic>=2，而 MindSpore / torch_npu
    常把 pydantic 钉在 1.x，于是 `import swanlab` 抛 ImportError。旧代码把
    任何 ImportError 都当成「没装」而误触发 pip install。
    """
    import logging
    import scripts.train_sft as t

    class _Args:
        swanlab_api_key = ''
        board_size = 19
        ver = 'v1'
        backbone_channels = backbone_res_blocks = res_blocks = 1
        convnext_blocks = attn_blocks = value_channels = 1
        value_res_blocks = policy_channels = policy_layers = 1
        batch_size = 1
        lr = 1.0
        epochs = 1

    # 造一个「存在但 import 就炸」的 swanlab
    pkg = tmp_path / 'swanlab'
    pkg.mkdir()
    (pkg / '__init__.py').write_text(
        "raise ImportError(\"cannot import name 'TypeAdapter' from 'pydantic'\")\n",
        encoding='utf-8')

    saved = sys.modules.pop('swanlab', None)
    sys.path.insert(0, str(tmp_path))
    import subprocess
    pip_calls = []
    real_check_call = subprocess.check_call
    subprocess.check_call = lambda *a, **k: pip_calls.append(a)
    logger = logging.getLogger('test_broken')
    try:
        with caplog.at_level(logging.DEBUG, logger='test_broken'):
            assert t._init_swanlab(_Args(), logger) is None
    finally:
        subprocess.check_call = real_check_call
        sys.path.remove(str(tmp_path))
        sys.modules.pop('swanlab', None)
        if saved is not None:
            sys.modules['swanlab'] = saved

    assert not pip_calls, f'不该在进程内 pip install，实际调用了: {pip_calls}'
    text = caplog.text
    assert 'TypeAdapter' in text, f'未报出真实原因，实际日志: {text!r}'
    assert '未安装' not in text, f'把「装了但坏」误报成「未安装」: {text!r}'


@pytest.mark.parametrize('fname', _TRAIN_SCRIPTS)
def test_import_error_never_triggers_autoinstall(fname):
    """结构性不变量：两个训练脚本都不得在 import 失败分支里调 pip。"""
    src = open(os.path.join(ROOT, 'scripts', fname), encoding='utf-8').read()
    assert 'except ImportError:' in src or 'ImportError' in src
    # find_spec 存在 => 能区分「没装」与「装了但坏」
    assert 'find_spec' in src, \
        f'{fname} 未用 find_spec 区分「未安装」与「安装损坏」'
    assert 'getaddrinfo' not in src, '仍残留 DNS 探测调用'
    assert '\nimport socket' not in src, 'socket import 仍未移除'
