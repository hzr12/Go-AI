r"""fp16 溢出必须是**全局**判定，否则四卡权重会永久分叉。

事故（2026-10-05，4×910A，A 段 batch=1900/卡）
---------------------------------------------
    scale=8   skip=7   （50 步内从 1024 一路减半 7 次）

`GradScaler` 的 `found_inf` 是**本地**的：`unscale_` 只登记本 rank 检出的非有限，
`step` 只看本地标记，**全程没有任何 collective**。于是在多卡下：

    某 rank 有 inf -> 它跳过；其余 rank 照常 `optimizer.step()`

=> **各 rank 的权重从此永久不同**，之后每一次 `all_reduce` 都在混合**四个不同
模型**的梯度。这不是"少训几步"的损失，是训练从此无效。而且各 rank 的 scale 各自
独立减半 => 进一步漂移（`skip=7` 这个计数本身也是 per-rank 的）。

以 14% 的边际溢出率算，"四张卡同一步一起溢出"的概率极低 => **分叉几乎立刻发生**。

所以本文件钉住三件事：

* `C1` **判定要跨 rank**：任一 rank 有非有限 => 全体跳步（本地干净也不例外）；
* `C2` **scale 要跨 rank 一致**：跳步后各 rank 的缩放值必须**逐位相同**，
  否则 `scaler.unscale_` 在不同 rank 上除的是不同的数，梯度尺度就错位了；
* `C3` **定位要赶在 clip 之前**：`clip_grad_norm_(max_norm=1.0)` 在
  `total_norm = inf` 时算出 `clip_coef = 0` 并 `grad.mul_(0)` => `inf * 0 = NaN`，
  而 `inf => clip_coef = 0 < 1` 这个分支**一定会进** => clip 之后「inf 个数」
  **结构上恒为 0**。诊断排在 clip 之后就只能打出「溢出可能发生在已被释放的中间
  张量里」—— 那是工具的盲区，不是关于计算的结论。

用桩替掉 `GradScaler` 而不是真的造一个：本机没有 CUDA，
`torch.amp.GradScaler(enabled=True)` 会自我禁用（实测），那样测的就不是被测物了。
"""
import io
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.train_sft as mod  # noqa: E402


class _FakeScaler:
    """只实现 `_scaler_step_global` 用到的那几个方法，且行为完全可预测。"""

    def __init__(self, scale=1024.0, backoff=0.5, growth=2.0):
        self._scale = float(scale)
        self._backoff = float(backoff)
        self._growth = float(growth)
        self.updates = []          # 记录每次 update 给了什么

    def get_scale(self):
        return self._scale

    def get_backoff_factor(self):
        return self._backoff

    def update(self, new_scale=None):
        self.updates.append(new_scale)
        if new_scale is not None:
            self._scale = float(new_scale)


class _Op:
    MAX = 'MAX'


class _MaxAllReduceDist:
    """假通信域：`all_reduce(..., MAX)` 把张量填成 `_answer`。

    单进程里模拟"另一个 rank 报了溢出"：本地判定干净，但归约结果说有。
    """

    ReduceOp = _Op

    def __init__(self, answer):
        self.answer = int(answer)
        self.calls = 0

    def as_dist(self):
        return self

    def get_world_size(self):
        return 4

    def get_rank(self):
        return 0

    def all_reduce(self, t, op=None):
        self.calls += 1
        t.fill_(self.answer)


class _StepRecorder:
    """记下 `AdamW.step()` 有没有被调用。"""

    def __init__(self):
        self.n = 0
        self._orig = torch.optim.AdamW.step

    def __enter__(self):
        rec = self

        def spy(optimizer, *a, **kw):
            rec.n += 1

        torch.optim.AdamW.step = spy
        return self

    def __exit__(self, *exc):
        torch.optim.AdamW.step = self._orig
        return False


def _opt_with_grads(vals):
    p = torch.nn.Parameter(torch.zeros(len(vals)))
    opt = torch.optim.AdamW([p], lr=1e-4)
    p.grad = torch.tensor(vals, dtype=torch.float32)
    return opt, p


def _with_fake_dist(answer):
    cap = _MaxAllReduceDist(answer)
    real = (mod.dist, mod._dist_active)
    mod.dist, mod._dist_active = cap.as_dist(), (lambda: True)
    return cap, real


def _restore(real):
    mod.dist, mod._dist_active = real


# --------------------------------------------------------------------------- #
# C1 判定跨 rank
# --------------------------------------------------------------------------- #
def test_clean_local_rank_still_skips_when_another_rank_overflowed():
    """本地梯度干净、但别的 rank 有 inf => 本 rank **也必须跳**。

    这是整个修复的核心：原来 `scaler.step()` 只看本地，于是这个 rank 会照常更新，
    四份权重就此分叉。
    """
    scaler = _FakeScaler(1024.0)
    opt, p = _opt_with_grads([1.0, -2.0, 3.0])          # 本地全有限
    cap, real = _with_fake_dist(answer=1)              # 别的 rank 报了溢出
    try:
        with _StepRecorder() as rec:
            stepped = mod._scaler_step_global(
                scaler, opt, scale_before=1024.0, use_scaler=True)
    finally:
        _restore(real)

    assert cap.calls >= 1, '判定必须真的做一次 all_reduce（跨 rank 归约）'
    assert stepped is False, '别的 rank 溢出时本 rank 也必须跳步'
    assert rec.n == 0, '跳步时绝不能调用 optimizer.step()（否则权重分叉）'
    assert scaler.get_scale() == 512.0, (
        '跳步必须按 backoff 降缩放值，实得 %r' % scaler.get_scale())


def test_every_rank_skips_and_scales_stay_identical():
    """两个 rank 都判溢出 => 两边都跳，且跳完后的缩放值**逐位相同**。

    `scaler.unscale_` 是在各 rank 上各自除以自己的 scale 的；scale 一旦漂移，
    梯度尺度就错位，而 all_reduce 会把不同尺度的梯度混在一起。
    """
    scales = []
    for rank in (0, 1):
        torch.manual_seed(rank)
        scaler = _FakeScaler(1024.0)
        opt, p = _opt_with_grads([1.0, float('inf'), 3.0])
        cap, real = _with_fake_dist(answer=1)
        try:
            with _StepRecorder():
                stepped = mod._scaler_step_global(
                    scaler, opt, scale_before=1024.0, use_scaler=True)
        finally:
            _restore(real)
        assert stepped is False, 'rank%d 应当跳步' % rank
        scales.append(scaler.get_scale())
    assert scales[0] == scales[1], (
        '两个 rank 跳步后的缩放值不同：%r —— scale 一旦漂移，unscale_ 除的数就不同'
        % (scales,))


def test_all_clean_ranks_step_normally():
    """全局都干净 => 正常更新（不能为了"保险"把每步都跳掉）。"""
    scaler = _FakeScaler(1024.0)
    opt, p = _opt_with_grads([1.0, -2.0, 3.0])
    cap, real = _with_fake_dist(answer=0)
    try:
        with _StepRecorder() as rec:
            stepped = mod._scaler_step_global(
                scaler, opt, scale_before=1024.0, use_scaler=True)
    finally:
        _restore(real)
    assert stepped is True, '全局干净时必须真的更新'
    assert rec.n == 1, 'optimizer.step 应被调用 1 次，实得 %d' % rec.n
    assert scaler.get_scale() == 1024.0, '全局干净时缩放值不该变'
    assert scaler.updates == [None], (
        '成功路径必须走 update() 免参版（growth_interval 逻辑在里面），实得 %r'
        % (scaler.updates,))


def test_local_overflow_is_detected_without_any_rank_help():
    """本地就有 inf、且归约也说有 => 跳（这条不该依赖"别的 rank"）。"""
    scaler = _FakeScaler(1024.0)
    opt, p = _opt_with_grads([float('nan'), 1.0])
    cap, real = _with_fake_dist(answer=1)
    try:
        with _StepRecorder() as rec:
            stepped = mod._scaler_step_global(
                scaler, opt, scale_before=1024.0, use_scaler=True)
    finally:
        _restore(real)
    assert stepped is False
    assert rec.n == 0


def test_no_dist_falls_back_to_local_only():
    """单卡（通信域未建）=> 退化成"只看本地"，且不要求 collective。"""
    scaler = _FakeScaler(1024.0)
    opt, p = _opt_with_grads([1.0, float('inf')])
    real = (mod.dist, mod._dist_active)
    mod._dist_active = lambda: False
    try:
        with _StepRecorder() as rec:
            stepped = mod._scaler_step_global(
                scaler, opt, scale_before=1024.0, use_scaler=True)
    finally:
        _restore(real)
    assert stepped is False, '本地有 inf 时单卡也必须跳'
    assert rec.n == 0


def test_scaler_disabled_still_steps():
    """`use_scaler=False`（bf16 路径）=> 直接 step，不做任何溢出判定。"""
    opt, p = _opt_with_grads([1.0, float('inf')])
    scaler = _FakeScaler(1.0)
    with _StepRecorder() as rec:
        stepped = mod._scaler_step_global(
            scaler, opt, scale_before=1.0, use_scaler=False)
    assert stepped is True, 'bf16 路径没有 GradScaler，必须照常更新'
    assert rec.n == 1


# --------------------------------------------------------------------------- #
# C3 定位要赶在 clip 之前
# --------------------------------------------------------------------------- #
def _main_src():
    with io.open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
                 encoding='utf-8') as fh:
        return fh.read()


def _optimizer_step_window():
    """从 `scaler.unscale_(optimizer)` 起、往后 4 KB 的**代码**（剥掉注释）。

    必须**先剥注释、再切片**：反过来（先切 4 KB 源码、再剥注释）时，注释体积
    会挤占窗口 —— 2026-10-07 给 `g` 补 g1/g2/g3 判读说明多写了十几行注释，
    就把 `_scaler_step_global(` 挤出了窗口，`test_training_loop_does_not_call_
    scaler_step_directly` 因此误红。那条断言查的是**代码**，不该被「注释写了
    多少」影响；先剥注释，窗口才是稳定的「4 KB 代码」而非「4 KB 源码」。

    限定窗口是为了让断言精确：`clip_grad_norm_` 这个词在文件前部的注释里也大量
    出现，直接全文件 `index()` 会命中注释里的那一处，断言就变成永远绿。
    """
    src = _main_src()
    src = '\n'.join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith('#'))
    i = src.index('scaler.unscale_(optimizer)')
    win = src[i:i + 4000]
    win = re.sub(r'(?s)("""|\'\'\').*?\1', '', win)
    return win


def test_locate_overflow_is_called_before_clip_grad_norm():
    """`_locate_overflow` 必须排在 `clip_grad_norm_` **之前**。

    `clip_grad_norm_(max_norm=1.0)` 在 `total_norm = inf` 时算出 `clip_coef = 0`
    并 `grad.mul_(0)` => `inf * 0 = NaN`，而 `inf => clip_coef = 0 < 1` 一定会进
    => clip 之后「inf 个数」结构上恒为 0，看到的 nan 全是 clipper 造的。排在 clip
    之后就只能打出「溢出可能发生在已被释放的中间张量里」—— 那是工具的盲区，不是
    关于计算的结论（2026-10-05 真机 4 卡日志证实）。
    """
    win = _optimizer_step_window()
    i_locate = win.index('_locate_overflow(')
    i_clip = win.index('clip_grad_norm_(')
    assert i_locate < i_clip, (
        '_locate_overflow 排在 clip_grad_norm_ 之后 => clip_coef=0 把 inf 变成 NaN，'
        '定位必然失明（真机表现为「溢出可能发生在已被释放的中间张量里」那种假结论）')


def test_training_loop_does_not_call_scaler_step_directly():
    """训练循环不许再出现裸的 `scaler.step(optimizer)` —— 那正是"只看本地"的来源。"""
    win = _optimizer_step_window()
    assert 'scaler.step(optimizer)' not in win, \
        '训练路径里还有裸的 scaler.step(optimizer) —— 它只查本地 found_inf'
    assert '_scaler_step_global(' in win, '应改走 _scaler_step_global'


def test_skip_counters_are_driven_by_the_global_verdict():
    """跳步计数必须挂在**全局**判定上，而不是"本地 scale 有没有下降"。

    原来 `_real_step = scaler.get_scale() < _scale_now` 是 per-rank 的判断，
    于是报出来的 `skip` 占比在多卡下没有意义（每个 rank 各数各的）。
    """
    win = _optimizer_step_window()
    assert '_real_step = not (use_scaler\n' not in win, \
        '不要用"本地 scale 是否下降"来判跳步'
    assert 'get_scale() < _scale_now' not in win, \
        '跳步判定必须来自全局归约的结果，而不是本地 scale 的变化'