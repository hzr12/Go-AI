"""SFT 验证集评估的确定性：固定采样源 + 与训练 RNG 流隔离。

缺陷
----
`src/data/dataset.py:50-75` 的 `sample_batch_numpy(idxs, rng=None)` 给每个样本抽一个
8 路对称变换，`rng is None` 时用**全局** `np.random.randint(0, 8, size=B)`。而
`scripts/train_sft.py` 的两个评估函数都不给 rng（函数其实早就支持 `rng=`）→ 后果有两层：

  ① 同一份权重、同一份 `eval_idx` 连跑两次 eval，指标不同（KL/Brier 尤其抖）——
     「模型变好了」和「这次抽到的变换不一样」根本分不开；
  ② 更严重：每次 eval 都推进全局流 → 训练侧 `rng.shuffle(train_idx)` 与训练 batch 的
     增强抽样结果取决于「eval 跑过几次、什么时候跑」→ **eval 频率会改写训练轨迹**。

覆盖
----
  - 行为锁：同一假模型 + 同一 eval_idx 连调两次 `evaluate_metrics`，全部返回指标逐位
    相同（1）；不插 eval 与插一次 eval 后从全局流抽到的数逐位相同（3）
  - 状态锁：eval **窗口内消耗的**随机不逃逸出去 —— 假模型在 forward 里真的抽 torch
    随机（`consume_torch_rng=True`），返回后 numpy 四段状态与 torch CPU 状态逐位不变
    （2）；`_isolated_global_rng()` 在正常路径与异常路径上都还原两条流（8）
  - 契约锁：传进 `sample_batch_numpy` 的 rng 是由 `EVAL_SAMPLING_SEED` 播种的
    Generator；不传 `rng` 时也必须落到 `_eval_rng()` 而不是把 None 透传（4、5）
  - 既有性质加锁：train/eval 分割用 `default_rng(0)`、按棋局分支的 `eval_idx` 升序（6）
  - 可见性锁：两处 `[eval]` 日志行都打出 eval 采样种子（7）

用例 6 是给**既有**性质加锁（分割本来就是确定性的），不是缺陷用例，修复前后都应绿。
用例 1/3 是行为性红：修复前它们因「指标不等 / 全局流错位」而失败。

已知边界（**读用例 1 前必看**）
------------------------------
本文件锁的是**采样源**（8 路对称增强）与**全局随机流**这两件事。用例 1 的假模型不含
dropout，所以它证明的是「抽样固定后，同权重两次 eval 逐位相同」。真实模型那一路
（注意力 dropout）由 `tests/test_attn_dropout_eval.py` 用**真模型**覆盖，本文件不重复。

真实 `AlphaGoNet` 的注意力 dropout 曾经走函数式 API：`F.dropout(..., p)` /
`F.scaled_dot_product_attention(..., dropout_p=p)`（`src/networks/backbone.py:133,141-145`，
内联那处在 `_sparse_attn` 的 `:413-414`）。这些 API 的 `training` 默认 True，模块级
helper 又拿不到 `self.training`，于是 `model.eval()` **关不掉**它们。**该闸门已在
commit `64e382f` 落地**：5 处站点（4 处 `dropout_p` + 1 处内联 `F.dropout`）全部改走
`self.training` 派生的 `attn_drop_p`。`--attention-dropout` 的默认值 0.1
（`train_sft.py:868`）现在只作用于训练期。

所以「同权重两次 eval 逐位相同」这个性质**现在在 CPU 与 device（NPU/CUDA）两侧都成立**：
eval 前向在任何 device 上都不再抽 dropout 掩码，device 生成器不再被 eval 推进。

仍然成立的那一半 —— **本任务的隔离机制只覆盖 CPU 生成器**：`_isolated_global_rng()`
快照/还原的是 `np.random` 全局流与 **CPU** 的 MT19937（`torch.get_rng_state()`），
**不覆盖 device 生成器**。也就是说 device 侧的「eval 频率改写训练轨迹」现在是被
`64e382f` 的闸门挡住的，**不是**被这里的隔离挡住的：本文件这 8 条只看 CPU 流，
闸门一旦被回退，它们仍然全绿，而 device 侧的缺陷会静默复活。守住那个前提是
`tests/test_attn_dropout_eval.py` 的职责（它的用例 1 断言 eval 期间
`torch.get_rng_state()` 不动）。别把这 8 条读成「device 侧也已由隔离机制兜底」。

全部 CPU、秒级：假 model（定长无并列 logits）+ 假 dataset（逐条照抄真实
`sample_batch_numpy` 的随机源语义），不加载真模型、不读真数据。
"""
import ast
import inspect
import os
import sys
import textwrap

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.train_sft as t                # noqa: E402

SRC = open(os.path.join(ROOT, 'scripts', 'train_sft.py'), encoding='utf-8').read()
BS = 4              # 假 batch size
N_SAMPLES = 40      # 10 批
N_ACTIONS = 32      # 假动作数（须 >= 10，evaluate_metrics 内部 topk(10)）
AMP = torch.float32

# `EVAL_SAMPLING_SEED = 1234` 与训练侧 `_BatchPrefetcher.__init__` 的默认
# `seed=1234`（`train_sft.py:630`）**数值相同但互不相干**，不要以为有耦合：
# 前者喂 `np.random.default_rng` 给验证集的 8 路增强抽样，后者喂预取器各子进程的
# `default_rng(seed + wi)`。改其中一个不会动到另一个。


# --------------------------------------------------------------------------- #
# 假 model / 假 dataset
# --------------------------------------------------------------------------- #
class _FakeDataset:
    """镜像真实 `SupervisedDataset.sample_batch_numpy` 的**随机源语义**。

    真实实现（`src/data/dataset.py:72-75`）：每个样本抽一个 8 路对称变换，
    `rng is None` 时取**全局** `np.random.randint`，否则取传入 rng 的 `integers`
    （Generator 用 `integers` 而非 `randint`）。这里逐条照抄，包括这处
    RandomState/Generator 的 API 差异 —— 抄错就等于没在测真 bug。

    `forced_tform` 是给「指标对变换敏感吗」那条守卫用的旁路：指定它就整批用同一个
    变换编号、完全不抽随机。假 dataset 唯一不忠实于真实实现的地方，仅此一处。
    真值着法 `moves` 只由样本 id 决定（真实数据里专家着法是固定的，被变换的是它的
    坐标），因此变换一变，假模型的预测就跟着变 → 指标跟着变。
    """

    def __init__(self, n_samples=N_SAMPLES, n_actions=N_ACTIONS, forced_tform=None):
        self.n_samples = n_samples
        self.n_actions = n_actions
        self.forced_tform = forced_tform
        self.batch_sizes = []
        self.rngs = []            # 每次调用收到的 rng 对象（None = 走了全局）
        self.first_draw = None    # 首次调用抽到的变换向量

    def sample_batch_numpy(self, idxs, rng=None, augment=True):
        # `augment` 只为签名兼容而存在（真实实现新增了它，评估路径传 False）。
        # 本假 dataset **刻意忽略**它、继续照常抽变换：本文件锁的是「采样源被固定 +
        # 消耗不逃逸」，它的断言（尤其用例 1 的空转守卫 forced_tform 0 → top1=1.0）
        # 就建立在「eval 真的消费了抽到的变换」这个前提上。关增强那一半由
        # tests/test_eval_no_augment.py 单独覆盖。
        sel = np.asarray(list(idxs), dtype=np.int64)
        B = len(sel)
        self.batch_sizes.append(B)
        self.rngs.append(rng)
        if self.forced_tform is not None:
            tforms = np.full(B, self.forced_tform, dtype=np.int64)
        elif rng is None:
            tforms = np.random.randint(0, 8, size=B)
        else:
            tforms = rng.integers(0, 8, size=B)
        if self.first_draw is None:
            self.first_draw = np.array(tforms, copy=True)

        # 把样本 id 与变换编号一起编码进特征，假模型据此产出预测
        states = np.zeros((B, 2, 1, 1), dtype=np.float32)
        states[:, 0, 0, 0] = sel.astype(np.float32)
        states[:, 1, 0, 0] = tforms.astype(np.float32)
        moves = ((sel * 7 + 1) % self.n_actions).astype(np.int64)
        values = np.where(sel % 2 == 0, 1.0, -1.0).reshape(B, 1).astype(np.float32)
        return states, moves, values


class _FakeModel:
    """定长**无并列** logits（rank1=5.0、rank2=4.5、其余 0,-1,-2… 的斜坡）+ 随变换变的 value。

    最优着法 = 真值着法 + 5*t (mod A)：t=0 时 top1 命中，t≠0 时 5*t mod 32 必不等于
    0（32 不是 5 的因子）→ 全部落错。于是 top1 恰好是「抽到 t=0 的样本比例」，对变换
    极其敏感。无并列 → topk 结果唯一确定，指标不依赖 torch 怎么打破并列。

    `consume_torch_rng=True` 时每次前向都真的抽一个 torch 随机数（抽完丢掉，不进
    logits）—— 用来模拟真模型在 eval 期的 dropout 消耗，见用例 2。默认关闭：其余用例
    关心的是抽样源，让前向保持纯确定性，免得混入无关变量。
    """

    def __init__(self, n_actions=N_ACTIONS, consume_torch_rng=False):
        self.n_actions = n_actions
        self.consume_torch_rng = consume_torch_rng

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, state):
        if self.consume_torch_rng:
            torch.rand(1)                 # 抽了就丢：只模拟「消耗了 torch 全局随机」
        B = state.shape[0]
        ids = state[:, 0, 0, 0].long()
        tforms = state[:, 1, 0, 0].long()
        ramp = -torch.arange(self.n_actions, dtype=torch.float32)
        logits = ramp.unsqueeze(0).repeat(B, 1)
        rows = torch.arange(B)
        peak = (ids * 7 + 1 + 5 * tforms) % self.n_actions
        logits[rows, peak] = 5.0
        logits[rows, (peak + 1) % self.n_actions] = 4.5
        value = ((tforms % 5) - 2).to(torch.float32).unsqueeze(-1) * 0.25
        return logits, value


def _run_metrics(ds, model=None, idxs=None, **kwargs):
    """跑一次 `evaluate_metrics`，返回 metrics。"""
    model = model if model is not None else _FakeModel()
    idxs = np.arange(ds.n_samples) if idxs is None else idxs
    kwargs.setdefault('device', 'cpu')
    kwargs.setdefault('amp_dtype', AMP)
    kwargs.setdefault('max_batches', 0)          # 0 = 跑满，别让批数上限掺和进来
    return t.evaluate_metrics(model, ds, idxs, BS, **kwargs)


def _seeded_first_draw(size):
    """`EVAL_SAMPLING_SEED` 播种的流抽出的**第一段**变换向量。"""
    return np.random.default_rng(t.EVAL_SAMPLING_SEED).integers(0, 8, size=size)


def _assert_numpy_state_unchanged(before, after, ctx):
    """断言 `np.random.get_state()` 元组的**四段**全等。

    元组是 `('MT19937', keys(624,), pos, has_gauss, cached_gaussian)`：keys 数组必须用
    `np.array_equal`（`==` 会得到逐元素布尔数组，`assert` 语义完全不同），其余标量直接
    比。**四段都要比**：状态字未回绕时 `keys` 数组原地不动、只有 `pos` 前进，只比 keys
    会漏掉最常见的那种流推进（见报告第四节的实测 `(32,0,0.0) -> (72,0,0.0)`）。
    """
    assert before[0] == after[0], \
        f'{ctx}：全局随机源算法被换了: {before[0]!r} -> {after[0]!r}'
    assert np.array_equal(before[1], after[1]), \
        f'{ctx}：全局 np.random 的 624 个状态字被改动了 —— 训练 RNG 流被偷走'
    assert before[2:] == after[2:], \
        f'{ctx}：全局 np.random 的 pos/高斯缓存被改动: {before[2:]} -> {after[2:]}'


# --------------------------------------------------------------------------- #
# 1. 行为锁：同一份权重连跑两次 eval，指标逐位相同
# --------------------------------------------------------------------------- #
def test_repeated_eval_is_bit_identical():
    """同一假模型 + 同一 eval_idx 连调两次 `evaluate_metrics` → 每个指标精确相等。

    修复前必红（这正是缺陷本身）：两次 eval 各自从全局 np.random 抽一组 8 路对称
    变换，变换不同 → 真值着法被搬到不同坐标 → top1/top5/top10/kl/brier 全变。同一份
    权重的两次评估对不上，「这次指标变好了」就分不清是模型进步还是抽样噪声。

    末尾的空转守卫：先把变换全钉成 0 与全钉成 1 各跑一次，断言 top1 恰好是 1.0 与
    0.0 —— 证明这套假数据对变换**真的敏感**，上面的相等不是因为它怎么算都一样。

    覆盖范围：只管**采样源**。真实模型的注意力 dropout 是另一条独立的随机源，已由
    commit `64e382f` 的 `self.training` 闸门关掉，并由 `tests/test_attn_dropout_eval.py`
    用真模型逐位锁住，本文件不重复覆盖。
    """
    ds = _FakeDataset()
    model = _FakeModel()
    idxs = np.arange(N_SAMPLES)

    first = _run_metrics(ds, model, idxs)
    second = _run_metrics(ds, model, idxs)

    assert ds.batch_sizes == [BS] * 20, \
        f'两次 eval 应各跑满 10 批 × {BS} 样本（同一个 ds 累计 20 次调用），实得 {ds.batch_sizes}'
    assert first['batches'] == second['batches'] == 10, \
        f"两次 eval 各应跑 10 批，实得 {first['batches']} / {second['batches']}"
    for key in ('top1', 'top5', 'top10', 'kl', 'brier', 'n', 'batches', 'truncated'):
        assert first[key] == second[key], (
            f'同权重两次 eval 的 {key} 不一致: {first[key]!r} vs {second[key]!r} —— '
            f'eval 的增强抽样没有固定种子，指标不可复现')

    # --- 空转守卫：指标必须随变换而变 ---
    zeros = _run_metrics(_FakeDataset(forced_tform=0))
    ones = _run_metrics(_FakeDataset(forced_tform=1))
    assert zeros['top1'] == 1.0, \
        f'变换全为 0 时 top1 应恰好 1.0，实得 {zeros["top1"]!r}'
    assert ones['top1'] == 0.0, \
        f'变换全为 1 时 top1 应恰好 0.0，实得 {ones["top1"]!r}'


# --------------------------------------------------------------------------- #
# 2. 状态锁：eval 窗口内消耗的随机不逃逸（numpy + torch CPU 两条流）
# --------------------------------------------------------------------------- #
def test_eval_does_not_disturb_global_numpy_state():
    """eval **窗口内消耗掉的**随机不会逃出去：返回后 numpy 与 torch 全局状态逐位不变。

    断言的契约是「**消耗不逃逸**」，不是「eval 没消耗随机」—— 这一点决定它能不能承重。
    早期版本让假模型不碰 torch 随机、却在 eval **返回之后**才比 `torch.get_rng_state()`：
    上下文管理器退出时已经把状态还原了，于是该断言对「窗口内到底消耗了没有」恒为真 ——
    一个前向里带活 dropout 的真模型照样通过，红不了（结构上不可能失败）。

    现在假模型在 forward 里**真的抽 torch 随机**（`consume_torch_rng=True`），断言真正
    在问的是「eval 抽走的这些随机数会不会漏出去」。把 `_isolated_global_rng()` `finally`
    里的 `torch.set_rng_state(...)` 去掉，这条立刻红（报告里有自检输出）。

    为什么用假模型手工制造消耗，而不是直接用真 `AlphaGoNet`：`64e382f` 给 functional
    dropout 补上 `self.training` 闸门之后，真模型在 eval 期间**已经不消耗任何 torch 随机**
    （CPU 与 device 都不消耗，见模块 docstring「已知边界」）—— 拿真模型来测这段契约，
    窗口内根本没有消耗可还原，断言会退化成恒真。所以这里让假模型在 forward 里**真的抽**
    torch 随机（`consume_torch_rng=True`），人为造出「窗口内有消耗」这个前提，才能证明
    `_isolated_global_rng()` 把它还原了回去。「eval 期的真模型不消耗 torch 随机」这个
    前提本身由 `tests/test_attn_dropout_eval.py` 用真模型钉（它断言 eval 期间
    `torch.get_rng_state()` 前后不变）。

    numpy 侧四段全比（算法名 / 624 个状态字 / pos / has_gauss+cached_gaussian）；torch 侧
    用 `torch.equal`（ByteTensor，同样要逐位比而不是 `==`，后者会得到逐元素布尔 Tensor）。

    修复前必红：eval 抽变换时走了全局 `np.random.randint`，状态被推进（pos 32→72）。
    """
    np.random.seed(20260926)
    torch.manual_seed(20260926)
    np.random.random(16)                     # 先把两条流推到中间位置（pos != 0）
    torch.rand(8)
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    # 关键：前向里真的消耗 torch 全局随机，否则下面那条断言是恒真的
    _run_metrics(_FakeDataset(), model=_FakeModel(consume_torch_rng=True))

    _assert_numpy_state_unchanged(np_before, np.random.get_state(), '跑完一次 eval 后')
    torch_after = torch.get_rng_state()
    assert torch.equal(torch_before, torch_after), (
        'eval 窗口内消耗的 torch 随机逃逸了：前向抽的随机数把训练的 CPU RNG 流推进了 —— '
        '训练轨迹又会依赖 eval 频率。去掉 _isolated_global_rng() 的 torch.set_rng_state '
        '即可复现')


# --------------------------------------------------------------------------- #
# 3. 行为锁：eval 不改变训练侧的随机流（训练轨迹与 eval 频率无关）
# --------------------------------------------------------------------------- #
def test_training_stream_unaffected_by_interleaved_eval():
    """在全局流上抽 5 个 → 跑一次 eval → 再抽 5 个，后 5 个必须与「不插 eval」逐位相同。

    这是本任务**最贵**的一条性质：训练侧 `rng.shuffle(train_idx)` 与训练 batch 的增强
    抽样都吃全局流，eval 推进它就等于让「eval 频率 / eval 步数」改写训练轨迹 —— 同一
    份配置换个 `--eval-every` 就训出另一个模型。

    修复前必红：eval 抽走了一批随机数，后 5 个整体错位。
    """
    seed = 20260926
    np.random.seed(seed)
    np.random.random(5)                      # 复位后重抽前 5 个（占位，对齐两条流）
    _run_metrics(_FakeDataset())             # 插一次 eval
    after = np.random.random(5)

    np.random.seed(seed)
    expect_after = np.random.random(10)[5:]  # 「不插 eval」时本该抽到的后 5 个
    assert np.array_equal(after, expect_after), (
        f'插入一次 eval 后全局流错位了：\n'
        f'  插 eval: {after.tolist()}\n'
        f'  不插 eval: {expect_after.tolist()}\n'
        f'eval 改变了训练的随机流 → 训练轨迹依赖 eval 频率')


# --------------------------------------------------------------------------- #
# 4. 契约锁：传进 sample_batch_numpy 的 rng 由 EVAL_SAMPLING_SEED 播种
# --------------------------------------------------------------------------- #
def test_eval_rng_is_seeded_from_the_constant():
    """两个评估函数都必须把 `_eval_rng()` 的产物传进 `sample_batch_numpy`。

    用捕获桩断言拿到的是 `np.random.Generator`，且它抽到的**第一段**变换向量与
    `np.random.default_rng(EVAL_SAMPLING_SEED)` 的对应结果逐位相同 —— 引用模块常量
    而不是把 1234 抄两份，种子改了这条测试会跟着走（不会变成一个钉死旧魔法的快照）。
    """
    ref = _seeded_first_draw(BS)              # 首批大小 = BS
    for fname in ('evaluate_metrics', 'evaluate_top1'):
        ds = _FakeDataset()
        model = _FakeModel()
        idxs = np.arange(N_SAMPLES)
        if fname == 'evaluate_metrics':
            _run_metrics(ds, model, idxs)
        else:
            t.evaluate_top1(model, ds, idxs, BS, 'cpu', AMP, max_batches=0)

        assert ds.rngs, f'{fname} 根本没调 sample_batch_numpy'
        assert all(r is not None for r in ds.rngs), \
            f'{fname} 有调用没给 rng —— 那些批次的变换抽自全局 np.random'
        assert isinstance(ds.rngs[0], np.random.Generator), \
            f'{fname} 传的 rng 是 {type(ds.rngs[0])!r}，应为 np.random.Generator'
        assert np.array_equal(ds.first_draw, ref), (
            f'{fname} 传给 sample_batch_numpy 的 rng 不是 EVAL_SAMPLING_SEED 播种的：'
            f'首批变换 {ds.first_draw.tolist()} != 种子流 {ref.tolist()}')


# --------------------------------------------------------------------------- #
# 5. 契约锁：默认（不传 rng）必须落到 _eval_rng()，不能把 None 透传
# --------------------------------------------------------------------------- #
def test_default_rng_argument_is_deterministic_not_none():
    """不传 `rng` 时拿到的必须是**新建的确定性 Generator**，不是 None。

    「默认值即保障」：任何调用点（包括将来新写的、忘了传 rng 的）都不可能不小心拿到
    不确定性 —— 所以 `rng` 形参的默认值是 `None`，但 `None` 在函数内部必须先被
    `_eval_rng()` 接住，绝不能原样透传给 `sample_batch_numpy`。
    """
    for fname in ('evaluate_metrics', 'evaluate_top1'):
        fn = getattr(t, fname)
        params = inspect.signature(fn).parameters
        assert 'rng' in params, f'{fname} 没有 rng 形参，无法固定采样源'
        assert params['rng'].default is None, \
            f"{fname} 的 rng 默认值是 {params['rng'].default!r}，应为 None（" \
            f'默认值即保障：不传就走 _eval_rng()）'
        src = textwrap.dedent(inspect.getsource(fn))
        assert '_eval_rng()' in src, \
            f'{fname} 的函数体里没有 _eval_rng() —— rng=None 时会退回全局随机'
        assert 'sample_batch_numpy(sel, rng=' in src, \
            f'{fname} 没把 rng 透传给 sample_batch_numpy(sel, rng=...)'

        # 行为侧：不传 rng，捕获桩拿到的必须是非 None 的 Generator
        ds = _FakeDataset()
        idxs = np.arange(N_SAMPLES)
        if fname == 'evaluate_metrics':
            _run_metrics(ds, idxs=idxs)
        else:
            t.evaluate_top1(_FakeModel(), ds, idxs, BS, 'cpu', AMP, max_batches=0)
        assert ds.rngs and all(isinstance(r, np.random.Generator) for r in ds.rngs), \
            f'{fname} 不传 rng 时传下去的却是 {[type(r).__name__ for r in ds.rngs]}'


# --------------------------------------------------------------------------- #
# 6. 既有性质加锁：train/eval 分割确定、eval_idx 升序
# --------------------------------------------------------------------------- #
def test_eval_split_order_is_deterministic():
    """锁住「分割本来就是确定的」这一既有事实 —— 本任务只加锁，不改分割逻辑。

    两条性质（按内容锚点定位，不写死行号）：

      ① 两条分割分支都用 `np.random.default_rng(0)`（字面量 0，种子不会被变量悄悄改掉）
      ② 按棋局分割分支的 `eval_idx` 由对 `range(...)` 的**顺序扫描**构造 → 天然升序
         （另一条分支是 `idx_all[n_train:]`，顺序由 ① 的定种子洗牌决定 —— 同样固定，
         但不是升序，故不写死成升序）
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(t.main)))

    seeds = [ast.unparse(n.args[0]) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'default_rng' and n.args]
    assert '0' in seeds, f'main() 里分割用的不是 default_rng(0)，实得 {seeds}'
    assert seeds.count('0') >= 2, \
        f'两条分割分支（按棋局 / 按样本）都该用 default_rng(0)，实得 {seeds}'

    # eval_idx 两处赋值都被 `np.array(...)` 包了一层，先剥掉这层壳再看真实形状
    def _unwrap(node):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('array', 'asarray') and len(node.args) == 1):
            return node.args[0]
        return node

    assigns = [_unwrap(node.value) for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(isinstance(tg, ast.Name) and tg.id == 'eval_idx'
                       for tg in node.targets)]
    assert len(assigns) == 2, f'main() 里应有 2 处 eval_idx 赋值，实得 {len(assigns)}'

    comps = [v for v in assigns if isinstance(v, ast.ListComp)]
    assert len(comps) == 1, \
        '按棋局分割分支的 eval_idx 应是列表推导（顺序扫描），否则「升序」无从谈起'
    gens = comps[0].generators
    assert len(gens) == 1, f'eval_idx 的推导应有且只有一个 for，实得 {len(gens)}'
    scan = gens[0].iter                      # 推导的扫描源（不是 ListComp.iter）
    assert (isinstance(scan, ast.Call) and getattr(scan.func, 'id', None) == 'range'
            and len(scan.args) == 1), \
        f'eval_idx 的推导应扫描一个 range(...) 才能保证升序，实得 {ast.unparse(scan)!r}'

    slices = [v for v in assigns if isinstance(v, ast.Subscript)]
    assert len(slices) == 1 and ast.unparse(slices[0]) == 'idx_all[n_train:]', \
        f'按样本分割分支的 eval_idx 写法变了: {[ast.unparse(v) for v in slices]}'


# --------------------------------------------------------------------------- #
# 7. 可见性锁：日志里能看出 eval 抽样是固定种子的
# --------------------------------------------------------------------------- #
def test_eval_log_reports_sampling_seed():
    """两处 `[eval]` 日志都要在**同一个 `logger.info(...)` 调用**里打出 eval 采样种子。

    读日志的人只有从这一行才能知道「这批指标是在固定种子的增强抽样下算出来的」，
    否则一次 KL 抖动到底该信几分无从判断。SwanLab 侧不要求（它收的是指标本身）。

    定位方式是「取到那个 `logger.info` 调用本身的源码」，而不是在锚点后面截一段固定
    长度的窗口：窗口会把紧邻的下一条无关日志（`[eval] 验证集被截断：…`）也圈进来，于是
    「格式串里有 `seed=%d`」与「实参里有 `EVAL_SAMPLING_SEED`」可以由两处互不相干的
    文本分别凑出来 → 假绿。现在两个条件必须在同一个调用内同时成立。
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(t.main)))
    calls = [ast.unparse(n) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'info']
    for anchor in ('[eval] step=%d', '[eval] FINAL'):
        matched = [c for c in calls if anchor in c]
        assert matched, f'main() 里找不到含 {anchor} 的 logger.info 调用'
        for call in matched:
            assert 'seed=%d' in call, \
                f'{anchor} 的 logger.info 调用里没有打印 eval 采样种子'
            assert 'EVAL_SAMPLING_SEED' in call, \
                f'{anchor} 的 logger.info 调用里种子不是取自 EVAL_SAMPLING_SEED 常量'


def test_isolation_helper_restores_both_streams():
    """`_isolated_global_rng()` 必须在退出时还原 numpy 与 torch 两条流（含异常路径）。

    这是 eval 侧的兜底机制：万一将来有代码路径仍去碰全局随机源（老调用点、新增的
    增强、第三方库内部的 `np.random`），也不会把状态推进出去。`finally` 语义一并锁住
    —— eval 中途 OOM 抛错不该顺手改掉训练的随机流。

    numpy 侧四段全比（含 `pos` 与高斯缓存）：只比 624 个状态字会漏掉「状态字未回绕、
    只有 pos 前进」这种最常见的流推进。

    这条与用例 2 互补：这里直接把上下文管理器拎出来测「窗口内的消耗被还原」；用例 2
    走完整的 `evaluate_metrics` 路径。前向里的 torch 消耗由用例 2 的假模型提供。
    """
    np.random.seed(7)
    torch.manual_seed(7)
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    with t._isolated_global_rng():
        np.random.random(32)
        np.random.randint(0, 8, size=64)
        torch.rand(64)

    _assert_numpy_state_unchanged(np_before, np.random.get_state(),
                                  '_isolated_global_rng() 正常退出后')
    assert torch.equal(torch_before, torch.get_rng_state()), \
        '_isolated_global_rng() 退出后没有还原 torch RNG'

    # 异常路径也必须还原
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    try:
        with t._isolated_global_rng():
            np.random.random(32)
            torch.rand(32)
            raise RuntimeError('模拟 eval 中途失败')
    except RuntimeError:
        pass
    else:
        raise AssertionError('模拟的异常没冒出来，测试本身失效')
    _assert_numpy_state_unchanged(np_before, np.random.get_state(),
                                  'eval 抛异常时')
    assert torch.equal(torch_before, torch.get_rng_state()), \
        'eval 抛异常时没有还原 torch RNG'
