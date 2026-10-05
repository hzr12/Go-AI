r"""`selfplay_train.py`（RL/PPO）的跳步不得推进 LR 计划与 EMA。

与 SFT 路径的同一类缺陷，2026-10-05 补上
------------------------------------------
`scripts/train_sft.py` 在 `91c7d53`「跳过的步不再消耗 LR 计划与 EMA」修过一次，
但 **RL 路径从来没修**：

    scaler.step(opt)        # 内部跳过 optimizer.step()，但对调用方「成功返回」
    scaler.update()
    opt.zero_grad()
    scheduler.step()        # ← 权重一动没动，warmup/cosine 照样前进
    if ema is not None:
        ema.update()        # ← shadow 被多拉一次且计数 +1

后果与 SFT 那次实测一样：100% 跳步时 558 步的 warmup 被 **0 次学习**消耗掉，
等于训练一开始就拿到一个已经退火的 LR；而 eval 是在 EMA shadow 上评的 ⇒ shadow
会一路收敛到**初始权重**。真机 SFT 那轮就是这样被带偏的。

另外 RL 侧此前**一个跳步计数都没有** —— 于是「这段训练是否有效」完全不可观测，
而 SFT 侧有 `skipped_steps` / `skip_rate_pct`。

为什么不复用 `train_sft._scaler_step_global`
-------------------------------------------
那个函数做的是**跨 rank MAX 归约**，因为多卡 SFT 下各 rank 的 `found_inf` 是本地
的、会做出不同的跳/不跳决定 ⇒ 权重永久分叉。而 RL 按 `run.txt` 只跑单卡
（"硬件只支持单张 910A，RL 那一段也一样"），所以这里用「缩放值有没有下降」这个
O(1) 判据就够，也不必在每步的路径上付一次扫 5.5M 参数的全量 D2H。

真要上多卡 RL，这一条必须换成 `train_sft._scaler_step_global`。
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = io.open(os.path.join(ROOT, 'scripts', 'selfplay_train.py'),
              encoding='utf-8').read()


def _code_only(text):
    """剥掉 docstring 与整行注释，只留可执行代码。

    本文件的断言是「某调用必须被 `_real_step` 门控」这类结构判定，而注释里为了
    讲清事故会**故意引用**那些调用（`scheduler.step()`…）。带着注释判会自己把
    自己判红。
    """
    text = re.sub(r'(?s)("""|\'\'\').*?\1', '', text)
    return '\n'.join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith('#'))


def _ppo_step_block():
    """取 PPO 步进那一段（从清零 `accum_counter` 到 `losses.append`）。"""
    code = _code_only(SRC)
    i = code.index('accum_counter = 0')
    j = code.index('losses.append(', i)
    return code[i:j]


def test_scheduler_step_is_gated_on_real_step():
    blk = _ppo_step_block()
    assert 'if _real_step:' in blk, \
        '跳步的门控缺失：权重没动却照样推进 LR 计划'
    # scheduler.step() 必须落在 if _real_step: 之内（缩进更深）
    lines = blk.splitlines()
    idx_gate = next(k for k, ln in enumerate(lines)
                    if ln.strip().startswith('if _real_step:'))
    indent_gate = len(lines[idx_gate]) - len(lines[idx_gate].lstrip())
    idx_sched = next(k for k, ln in enumerate(lines)
                     if 'scheduler.step()' in ln)
    indent_sched = len(lines[idx_sched]) - len(lines[idx_sched].lstrip())
    assert indent_sched > indent_gate, \
        'scheduler.step() 与 if _real_step: 同级 ⇒ 跳步时仍在推进 LR 计划'


def test_ema_update_is_gated_on_real_step():
    blk = _ppo_step_block()
    lines = blk.splitlines()
    idx_gate = next(k for k, ln in enumerate(lines)
                    if ln.strip().startswith('if _real_step:'))
    indent_gate = len(lines[idx_gate]) - len(lines[idx_gate].lstrip())
    idx_ema = next(k for k, ln in enumerate(lines) if 'ema.update()' in ln)
    indent_ema = len(lines[idx_ema]) - len(lines[idx_ema].lstrip())
    assert indent_ema > indent_gate, \
        'ema.update() 与 if _real_step: 同级 ⇒ 跳步时 shadow 仍被拉向当前权重'


def test_skipped_steps_are_counted_and_reported():
    """跳步必须**可观测** —— RL 侧此前一个数都没有。"""
    blk = _ppo_step_block()
    assert '_n_skipped += 1' in blk, 'RL 侧没有跳步计数'
    assert '跳步' in blk, '跳步应有日志（否则跑完一轮看不出发生过）'
    assert re.search(r'if not _real_step:', blk), \
        '计数必须挂在「这一步没真的更新」上'


def test_real_step_comes_from_the_scale_not_from_a_grad_scan():
    """判据用缩放值下降，而不是扫梯度有限性。

    `GradScaler` 只在跳步时降 scale（成功时只等 `growth_interval` 到才 ×2），
    所以缩放值是 O(1) 的可靠信号；扫 5.5M 个参数判有限性要在每步路径上付一次
    全量 D2H。**若将来上多卡 RL，这条必须换成 `train_sft._scaler_step_global`**
    —— 那时才需要跨 rank 归约。
    """
    blk = _ppo_step_block()
    assert 'scaler.get_scale() < _scale_before' in blk, \
        '跳步判据应为「缩放值有没有下降」'
    assert 'isfinite' not in blk, '不应在每步路径上扫梯度有限性（付全量 D2H）'
    assert '_scale_before = scaler.get_scale() if scaler is not None' in blk, \
        '必须在 step/update **之前**记下缩放值，否则无从比较'