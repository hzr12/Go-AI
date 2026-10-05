r"""V7 的前向 NaN 是**硬失败**，不得继续跑。

事故（2026-10-05，4×910A，`--lr 9.77e-3 --batch-size 4000`）
--------------------------------------------------------
    step 400      lr=7.07e-03  loss=12.3146  scale=81920   skip=0    健康
    step 419      ← warmup 结束（= 10% × 4198），LR 到顶
    step 430~440  loss=nan      scale 峰值 163840
    step 450      lr=7.37e-03  loss=nan      scale=10      skip=13

从健康到爆炸 LR 只差 **4.2%**（7.07e-3 → 7.37e-3），而 scale 在此期间一路**涨**
到 163840 —— 也就是**梯度侧还有巨大余量，瓶颈在前向**。

为什么必须硬失败
----------------
`GradScaler` **只缩放梯度**，它看不见前向。而这里的 NaN 来自前向
（`[v7] 被净化的坏行` 里 `ownership`/`score_mean`/`scorebelief_*`/`scoring`/`seki`
**每一行**都非有限，全部出自 `ValueHead`）。NaN 一旦进到梯度，权重就再也动不了：

* `scaler.step` 一直跳步 ⇒ 每步都是 NaN；
* 从 81920 减到 10 是 **14 次减半**，**一次也没救回来** —— 因为病根不在梯度；
* 而日志上除 `loss=nan` 之外**一切正常**：速度 412 s/step、显存 58.53 GB、耗时照涨。

⇒ 「训练死了但还在跑」，剩下 3748 步 × 412 s ≈ **12 小时**的纯浪费，而且产出的
模型必然是废的。这条闸门就是让下一次爆炸在**几秒内**停住并留下现场。

为什么不看"哪些项坏了"而只看 `total`
-------------------------------------
`total_finite is False` 就等于「**被优化的那个标量**非有限」—— 那正是要反向传播
的东西。只要它非有限，继续跑就没有意义。反过来，`total` 有限而某项坏了的那种情况
（零系数项被净化成 0）**不该**中止，所以判据必须是 `total` 而不是 `_bad_terms`。
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = io.open(os.path.join(ROOT, 'scripts', 'train_sft.py'),
              encoding='utf-8').read()


def _code_only(text):
    text = re.sub(r'(?s)("""|\'\'\').*?\1', '', text)
    return '\n'.join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith('#'))


def _v7_bad_block():
    i = SRC.index("if _tot_ok is False:")
    return _code_only(SRC[max(0, i - 300):i + 2600])


def test_nonfinite_total_loss_aborts_the_run():
    """`total_finite is False` ⇒ 抛异常中止，不得只打日志继续跑。"""
    blk = _v7_bad_block()
    assert 'raise RuntimeError(' in blk, \
        '加权总 loss 非有限时必须中止（否则会「训练死了但还在跑」白烧机时）'
    assert '_tot_ok is False' in blk, '闸门的判据必须是 total 是否有限'


def test_abort_message_says_it_is_not_a_gradient_overflow():
    """中止信息必须点明「这几乎不是梯度溢出」，否则下一次仍会被带偏。

    之前排查被带偏过一次：栈与 `GradScaler` 的行为都指向「反向算子溢出」，
    而真因在前向 —— GradScaler 只缩放梯度，降 scale 对前向 NaN 一点用都没有。
    """
    blk = _v7_bad_block()
    assert '前向' in blk, '中止信息要点明病根在前向'
    assert 'GradScaler' in blk, '要点明 GradScaler 只管梯度、管不到前向'
    assert 'scale' in blk, '要把中止时的缩放值一并打出来（判断梯度侧余量用）'


def test_abort_message_carries_the_evidence():
    """逐项点名 / 操作数 / 被净化的坏行 / scale —— 四样都要在中止信息里。

    这四样是**已经算出来**的（`_bad_terms` / `_bad_ops` / `_san_rows` /
    `scaler.get_scale()`），中止时只是没打出来。不打就等于下一次还得重跑一轮
    真机训练才能拿到同样的事实。
    """
    blk = _v7_bad_block()
    for token, why in (('_bad_terms', '逐项点名'),
                       ('_bad_ops', '坏在操作数'),
                       ('_san_rows', '被净化的坏行'),
                       ('get_scale', '中止时的缩放值')):
        assert token in blk, '中止信息缺 %s（%s）' % (token, why)


def test_abort_does_not_fire_when_total_is_finite():
    """零系数项被净化时 `total` 仍有限 —— 那种情况**不许**中止。

    这是本闸门最容易写坏的地方：若改成「`_bad_terms` 非空就中止」，
    就会在 A 段（九项系数为 0、恒被净化）每步都炸。
    """
    blk = _v7_bad_block()
    assert re.search(r'if _tot_ok is False:\s*\n\s*raise RuntimeError', blk), \
        'raise 必须**只**在 `total` 非有限这一支里'


def test_overflow_advice_does_not_mention_attn_window_for_v7():
    """诊断文案不得建议调 `--attn-window` —— V7 根本没有这个参数。

    本仓自己就写着这件事（`train_sft.py` 里 V7 的 attn-window 提示、
    `tests/test_no_aicpu_ops_in_startup_check.py`），而溢出诊断里却建议调它。
    这条自相矛盾的建议已经把人带偏过一次（真去调了一个不存在的旋钮）。
    """
    src = SRC[SRC.index('def _locate_overflow('):]
    src = src[:src.index('\ndef ')]
    body = _code_only(src)
    for ln in body.splitlines():
        if '可考虑调小 --attn-window' in ln or '降低 logits 幅度' in ln:
            raise AssertionError(
                '溢出诊断仍在建议调 --attn-window，而 V7 没有这个参数：%s' % ln.strip())
    assert '不要' in SRC and 'attn-window' in SRC, \
        '应当明确写出「不要调 --attn-window，V7 没有它」'