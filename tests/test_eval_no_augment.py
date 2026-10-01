"""SFT 验证集评估不该吃 8 路随机对称增强（`augment=False`）。

缺陷
----
`src/data/dataset.py:50-114` 的 `sample_batch_numpy(idxs, rng=None)` 给批内**每个样本**
抽一个 `tforms ∈ [0,8)`（`rng is None` 时抽**全局** `np.random.randint`），再用同一个
`tforms` 同时驱动两件事：

  ① `states`（`(B,12,H,W)`）按 `tforms` 翻转/旋转（`:88-99`）
  ② `moves` 经 `SYMMETRIES[t]` **同步重映射**（`:101-114`）

而 `scripts/train_sft.py` 的两个评估函数（`evaluate_metrics` / `evaluate_top1`）都直接调
`dataset.sample_batch_numpy(sel, rng=eval_rng)` → **每次 eval 的每个样本都被随机翻转/旋转过**。
后果有三层：

  ① 同一份权重、不同对称下的指标不同 —— 推理时不会出现随机翻转的棋盘，这层方差与
     线上真实输入分布不符；
  ② top-1 的分母里混进了「对称等价但标签已同步」的一致性，掩盖真实错误；
  ③ 最佳模型选择（P2.2b 之后 top1 已干净）仍带着这一层不该有的方差。

P2.2 用 `EVAL_SAMPLING_SEED` 把抽样**固定**了，但「固定」不等于「正确」——被评估的仍然是
一个随机挑了对称的验证集。本任务把增强整个关掉：评估**零随机**，于是「eval 与采样种子
无关」这条最终性质才真正成立（用例 5）。

覆盖
----
编号与本文件**章节注释**一致，9 个测试函数一一对应（P2.3b 对齐；此前模块 docstring 写的是
「用例 1–8」，既漏了 5b、又多出一条并不存在的用例 8，报告 §表又用的是另一套编号）：

  - 用例 1（`test_augment_false_returns_raw_features_and_moves`）：`augment=False` 的
    `states` 与直接调 `GoBoard.feature_planes_batched` 的结果**逐位相等**；
    `moves` 等于原始 `moves`（仅做非法/越界 → `bs*bs` 规范化）
  - 用例 2（`test_augment_false_consumes_no_randomness`，**行为性红**）：`augment=False`
    不消耗任何随机 —— 全局 `np.random.get_state()` 四段、**传进来的 rng 自身**都不推进；
    `torch.get_rng_state()` 也不动
  - 用例 3（`test_augment_true_still_transforms_consistently`）：`augment=True` 仍按同一个
    `t` 同步变换 `states` 与 `moves`（`t=4` → W 轴翻转，move `(r,c) → (r, bs-1-c)`），
    且与 `augment=False` 的输出不同（默认路径没被改坏）
  - 用例 4（`test_invalid_move_normalized_in_both_modes`）：`moves == -1`（pass）与越界标签
    在**两种模式**下都归一到 `bs*bs` —— 标签规范化必须与对称变换**解耦**，否则 eval 与
    训练对「pass」的定义就不一样了
  - 用例 5（`test_eval_metrics_independent_of_sampling_seed`，**行为性红，端到端**）：两个
    不同 `EVAL_SAMPLING_SEED` 下 `evaluate_metrics` 的八个指标**逐位相同**（假 dataset）
  - 用例 5b（`test_real_dataset_eval_is_seed_independent`）：同上，但喂**真**
    `SupervisedDataset` —— 假 dataset 只能证明接线，真数据才证明 `augment=False` 在真实
    抽样路径上确实生效
  - 用例 6a（`test_eval_call_sites_pass_augment_false`，**结构锁，修复前必红**）：按函数定位，
    两个 eval 函数内的 `sample_batch_numpy(...)` 都带 `augment=False` 字面量
  - 用例 6b（`test_only_eval_call_sites_disable_augmentation`）：按全文件扫描定边界 ——
    传 `augment=False` 的调用点**全部**落在两个 eval 函数内，预取器 worker 连 `augment`
    kwarg 都不许出现。（P2.3b 删掉了它原先的「全文件恰好 3 处调用」计数锁：那是实现形状锁，
    与真实不变式无关，详见该用例 docstring）
  - 用例 7（`test_training_path_default_is_augment_true`）：训练路径零回归 ——
    `sample_batch(...)`（不传 `augment`）的输出与显式 `augment=True` 逐位相同

空转守卫（防假绿）
----------------
用例 1/2/3/5/7 都各带一条「这套断言真的对增强敏感」的对照：用钉死 tform 的 rng（或
`augment=True`）跑一遍，输出必须与 `augment=False` **不同**；用例 2 先证明默认路径**确实**
推进了全局流；用例 5 先证明同一套假数据在两个不同 rng 下**确实**给出不同指标。若这些
对照不成立，被测断言就是恒真（哪怕全绿也说明不了问题）。

全部 CPU、秒级：真 `SupervisedDataset`（手工构造 6 条 9 路样本）+ 假 model/假 dataset
（只吐确定性 numpy），不读真实数据、不加载真模型。
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

import scripts.train_sft as t                          # noqa: E402
from src.data.dataset import SupervisedDataset          # noqa: E402
from src.game.go_rules import GoBoard                  # noqa: E402

TRAIN_SRC = os.path.join(ROOT, 'scripts', 'train_sft.py')
TRAIN_SRC_TEXT = open(TRAIN_SRC, encoding='utf-8').read()

BOARD = 9                # 9 路小棋盘（与 tests/test_attn_dropout_eval.py 一致，CPU 秒级）
BS2 = BOARD * BOARD      # 81：pass/越界标签的取值
AMP = torch.float32
EVAL_BS = 4              # 假 batch size
N_SAMPLES = 40           # 10 批（EVAL_BS=4）
N_ACTIONS = 32           # 假动作数（须 >= 10，evaluate_metrics 内部 topk(10)）

# 真数据集里 6 条样本的构造参数（手工摆成**非对称**盘面）：
#   样本 0: 黑子 (0,0)，my_hist 首手 (0,0)，着法落在**右上角** (0, 8) = 8
#   样本 1: 白子 (8,0)，op_hist 首手 (8,0)，着法落在**左下角** (8, 0) = 72
#   样本 2: 空盘，着法 -1（pass）
#   样本 3: 空盘，着法 84（**越界**，> 80）
#   样本 4: 黑白各一子（0,0)/(8,8)，着法天元 (4,4) = 40
#   样本 5: 有劫（ko=5），着法 (5,1) = 46
# 0/1/4/5 的着法都不在对称轴上 → 任何非恒等变换都会把它搬到别的点，
# 「若偷偷做了变换就会被抓到」靠的就是这四条。
SAMPLE_MOVES = np.array([0 * 9 + 8, 8 * 9 + 0, -1, BS2 + 3, 4 * 9 + 4, 5 * 9 + 1],
                        dtype=np.int16)
INVALID_MOVES = (2, 3)     # 样本 2（-1 = pass）与样本 3（84 = 越界）
# 样本 4 的着法在天元 (4,4)：9 路棋盘上它是全部 8 个对称的**不动点**，故它本身证明不了
# 「有没有被变换过」。0/1/5 的着法都离开 W 轴对称轴（(0,8)/(8,0)/(5,1)），真正提供区分度。


# --------------------------------------------------------------------------- #
# 真数据集（9 路、6 条手工样本）
# --------------------------------------------------------------------------- #
def _make_dataset(n=len(SAMPLE_MOVES)):
    """构造一个**非对称**的小 `SupervisedDataset`（真实现，不是替身）。

    前 6 条是上面手工摆好的（着法含 pass / 越界 / 角落，天元，劫）；`n > 6` 时用固定
    种子 `default_rng(20260926)` 补一批随机样本出来 —— 种子写死，故这批数据是**常量**，
    用例 5b 的逐位比较不会因数据抖动而变flaky。
    """
    k = len(SAMPLE_MOVES)
    if n < k:
        raise ValueError(f'n 必须 >= {k}（前 {k} 条是手工样本）')
    gen = np.random.default_rng(20260926)
    boards = np.zeros((n, BOARD, BOARD), dtype=np.int8)
    boards[:k, 0, 0] = 1                       # 黑子左上
    boards[1, 8, 0] = -1                   # 白子左下
    boards[4, 0, 0] = 1
    boards[4, 8, 8] = -1
    if n > k:
        boards[k:] = gen.integers(-1, 2, (n - k, BOARD, BOARD)).astype(np.int8)

    my_hist = np.full((n, 3), -1, dtype=np.int16)
    op_hist = np.full((n, 3), -1, dtype=np.int16)
    my_hist[0, 0] = 0 * BOARD + 0             # 己方最近一手在 (0,0)
    op_hist[1, 0] = 8 * BOARD + 0             # 对手最近一手在 (8,0)
    my_hist[4, 0] = 0 * BOARD + 0
    op_hist[4, 0] = 8 * BOARD + 8

    ko = np.full(n, -1, dtype=np.int16)
    ko[5] = 5                                 # 样本 5 有劫 → 通道 8 会挖掉 (0,5)
    to_play = np.ones(n, dtype=np.int8)
    values = np.ones(n, dtype=np.int8)
    to_play[:k] = np.array([1, -1, 1, 1, 1, 1], dtype=np.int8)
    values[:k] = np.array([1, -1, 1, -1, 1, 1], dtype=np.int8)
    moves = SAMPLE_MOVES.copy()
    if n > k:
        tail = gen.integers(0, BS2, n - k).astype(np.int16)
        tail[::4] = -1                                   # 每 4 条插一个 pass
        moves = np.concatenate([moves, tail])
        my_hist[k:] = gen.integers(-1, BS2, (n - k, 3)).astype(np.int16)
        op_hist[k:] = gen.integers(-1, BS2, (n - k, 3)).astype(np.int16)
        ko[k:] = gen.integers(-1, BS2, n - k).astype(np.int16)
        to_play[k:] = gen.choice([1, -1], n - k).astype(np.int8)
        values[k:] = gen.choice([1, -1], n - k).astype(np.int8)
    return SupervisedDataset({
        'boards': boards, 'my_hist': my_hist, 'op_hist': op_hist,
        'ko': ko, 'moves': moves, 'values': values, 'to_play': to_play,
    })


def _raw_features(ds, idxs):
    """直接调 `GoBoard.feature_planes_batched` 拿原始 12 通道（不经增强）的输入。

    ⚠ `n_channels=12` 是**显式钉住**的：P4.3 起该函数默认 17 通道（v21 stem 的
      布局），而本文件守的是 P2.3「评估期不施加增强」的语义、输入侧固定是旧
      12 通道布局（`_ASYM_WEIGHTS` 也是 12 维）。不钉住这里会与 dataset 的
      默认 12 通道失配。本文件**不测**通道数本身 —— 那是
      `tests/test_go_feature_planes_v21.py` 的职责。
    """
    idxs = np.asarray(idxs)
    return GoBoard.feature_planes_batched(ds.boards[idxs], ds.my_hist[idxs],
                                          ds.op_hist[idxs], ds.to_play[idxs],
                                          ds.ko[idxs], n_channels=12)


def _expected_moves(ds, idxs):
    """`augment=False` 应得的 moves：原始 moves，仅把非法/越界归一到 `bs*bs`。"""
    mv = np.asarray(ds.moves[np.asarray(idxs)], dtype=np.int64)
    valid = (mv >= 0) & (mv < BS2)
    return np.where(valid, mv, BS2).astype(np.int64)


class _FixedTformRng:
    """把 `tforms` 钉成同一个 `t` 的假 Generator。

    真实调用是 `rng.integers(0, 8, size=B)`（Generator 用 `integers` 而非 `randint`），
    这里只实现那一个方法。`calls` 记录被调用了几次 —— 用例 3 靠它证明「变换确实发生了」
    而不是碰巧与恒等变换同值。
    """

    def __init__(self, tform):
        self.tform = int(tform)
        self.calls = 0

    def integers(self, low, high=None, size=None):
        self.calls += 1
        return np.full(size, self.tform, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 假 model / 假 dataset（只服务用例 5：端到端的「与采样种子无关」）
# --------------------------------------------------------------------------- #
class _AugDataset:
    """镜像真实 `sample_batch_numpy` 的 **augment 语义**：`augment=False` 时不抽随机。

    特征里同时编码样本 id（`states[:,0,0,0]`）与变换编号（`states[:,1,0,0]`），假模型据此
    产出预测 → 变换一变指标就变。于是「关掉增强 ⇒ 指标不再依赖种子」这条性质可以被
    端到端地测出来，而不是靠「两次调用返回同一个 dict」的空转。

    `honor_augment=False` 时**无视**传入的 `augment` 一律抽变换，供空转守卫造对照。
    """

    def __init__(self, n_samples=N_SAMPLES, n_actions=N_ACTIONS,
                 honor_augment=True):
        self.n_samples = n_samples
        self.n_actions = n_actions
        self.honor_augment = honor_augment
        self.batch_sizes = []
        self.augment_flags = []

    def sample_batch_numpy(self, idxs, rng=None, augment=True):
        sel = np.asarray(list(idxs), dtype=np.int64)
        B = len(sel)
        self.batch_sizes.append(B)
        self.augment_flags.append(augment)
        if augment or not self.honor_augment:
            if rng is None:
                tforms = np.random.randint(0, 8, size=B)
            else:
                tforms = rng.integers(0, 8, size=B)
        else:
            tforms = np.zeros(B, dtype=np.int64)      # 不碰任何随机源
        states = np.zeros((B, 2, 1, 1), dtype=np.float32)
        states[:, 0, 0, 0] = sel.astype(np.float32)
        states[:, 1, 0, 0] = tforms.astype(np.float32)
        moves = ((sel * 7 + 1) % self.n_actions).astype(np.int64)
        values = np.where(sel % 2 == 0, 1.0, -1.0).reshape(B, 1).astype(np.float32)
        return states, moves, values


class _TformModel:
    """定长**无并列** logits + 随变换变的 value：指标对变换编号极其敏感。

    最优着法 = 真值着法 + 5*t (mod 32)：32 不是 5 的因子，故 t≠0 时必不命中 → top1 恰好
    等于「抽到 t=0 的样本比例」。value 取 `((t%5)-2)*0.25` → brier 也随 t 变。kl 随 logits
    变。于是「变换变了指标就变」是这套假数据的结构性事实，不是运气。
    """

    def __init__(self, n_actions=N_ACTIONS):
        self.n_actions = n_actions

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, state):
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


def _run_metrics(ds, **kwargs):
    kwargs.setdefault('device', 'cpu')
    kwargs.setdefault('amp_dtype', AMP)
    kwargs.setdefault('max_batches', 0)          # 0 = 跑满，别让批数上限掺和进来
    return t.evaluate_metrics(_TformModel(N_ACTIONS), ds,
                              np.arange(N_SAMPLES), EVAL_BS, **kwargs)


# --------------------------------------------------------------------------- #
# 1. augment=False：states 逐位等于原始特征、moves 逐位等于原始着法
# --------------------------------------------------------------------------- #
def test_augment_false_returns_raw_features_and_moves():
    """`augment=False` 既不翻转/旋转 `states`，也不重映射 `moves`。

    两条都拿**独立参照系**比，不是「与 augment=True 比」：

      · `states` 与直接调 `GoBoard.feature_planes_batched(...)` 的返回值逐位相等
        （dtype 与形状也一起比）—— 若实现里还留着 `out = np.empty_like` 那一遍变换，
        非对称盘面上必然对不上；
      · `moves` 等于原始 `moves` 经「非法/越界 → `bs*bs`」规范化后的结果。

    空转守卫：盘面与着法都必须在 W 轴翻转下**真的改变**（否则「没变换」与「变换了但
    恰好等价」分不开）。样本 0 的着法在右上角 (0,8)，翻成 (0,0) —— 差 8 格；
    样本 1 在左下角 (8,0) → (8,8)，样本 5 在 (5,1) → (5,7)。
    """
    ds = _make_dataset()
    idxs = np.arange(len(ds))
    raw = _raw_features(ds, idxs)

    # --- 空转守卫：这份数据对 W 轴翻转敏感（否则下面两条相等是恒真） ---
    assert not np.array_equal(raw, raw[:, :, :, ::-1]), \
        '构造的盘面恰好左右对称 → 用例 1 的 states 断言没有牙齿'
    # 着法按**整批**判：样本 4 的着法在天元 (4,4)，在 9 路棋盘上是全部 8 个对称的
    # 不动点，拿单条着法判会误报；这里只要求「至少有一条合法着法离开 W 轴」，
    # 足以保证「偷偷做一次翻转」会被 moves 断言抓到。
    mv_all = np.asarray(SAMPLE_MOVES, dtype=np.int64)
    cols = np.where((mv_all >= 0) & (mv_all < BS2), mv_all % BOARD, -1)
    off_axis = [i for i in range(len(mv_all))
                if cols[i] >= 0 and cols[i] != BOARD - 1 - cols[i]]
    assert len(off_axis) >= 2, \
        f'只有 {len(off_axis)} 条合法着法离开 W 轴对称轴（{off_axis}）—— ' \
        f'用例 1 的 moves 断言可能没有牙齿'

    states, moves_out, values = ds.sample_batch_numpy(idxs, augment=False)

    assert states.dtype == np.float16 and states.shape == (len(idxs), 12, BOARD, BOARD), \
        f'shapes/dtype 不符: {states.dtype} {states.shape}'
    assert np.array_equal(states, raw), (
        f'augment=False 的 states 与原始特征平面不逐位相等 —— 特征侧仍被变换过；'
        f'最大偏差 {np.abs(states - raw).max() if states.shape == raw.shape else "形状不同"}')
    want_moves = _expected_moves(ds, idxs)
    assert np.array_equal(moves_out, want_moves), (
        f'augment=False 的 moves 被重映射了: {moves_out.tolist()} != {want_moves.tolist()}')
    assert values.shape == (len(idxs), 1)


# --------------------------------------------------------------------------- #
# 2. augment=False 不消耗任何随机（行为性红）
# --------------------------------------------------------------------------- #
def test_augment_false_consumes_no_randomness():
    """`augment=False` 前后，全局 numpy / torch、以及**传进来的 rng 自身**都不推进。

    修复前必红：`:72-73` 无条件抽 `np.random.randint(0, 8, size=B)`（`rng is None` 时走
    **全局**流），全局状态被推进；传了 rng 时那个 Generator 也会前进 —— 后者正是
    「eval 指标依赖采样种子」的根源。

    断言分三段：全局 numpy 四段（算法名 / 624 个状态字 / pos / 高斯缓存）、torch CPU
    状态（`torch.equal`）、以及传入 Generator 的 `integers` 计数。只比 624 个状态字会
    漏掉最常见的「状态字未回绕、只有 pos 前进」。

    空转守卫（第一段）：默认路径（`augment=True` + `rng=None`）**必须**推进全局流 ——
    证明下面那些「不变」断言不是恒真。
    """
    ds = _make_dataset()
    idxs = np.arange(len(ds))

    # --- 空转守卫：默认路径确实在抽全局随机（这条修复前后都应绿） ---
    np.random.seed(20260926)
    np.random.random(16)                          # 把流推到中间（pos != 0）
    before_guard = np.random.get_state()
    ds.sample_batch_numpy(idxs)
    guard_after = np.random.get_state()
    assert before_guard[2] != guard_after[2], \
        '默认路径竟然没推进全局 np.random 的 pos —— 用例 2 的「不变」断言已失去牙齿'

    # --- 主断言：augment=False 不碰任何随机源 ---
    np.random.seed(20260926)
    np.random.random(16)
    torch.manual_seed(20260926)
    torch.rand(8)
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    passed_rng = np.random.default_rng(4242)

    ds.sample_batch_numpy(idxs, rng=passed_rng, augment=False)

    np_after = np.random.get_state()
    assert np_before[0] == np_after[0], \
        f'全局随机源算法被换了: {np_before[0]!r} -> {np_after[0]!r}'
    assert np.array_equal(np_before[1], np_after[1]), \
        '全局 np.random 的 624 个状态字被改动了 —— 训练 RNG 流被偷走'
    assert np_before[2:] == np_after[2:], \
        f'全局 np.random 的 pos/高斯缓存被改动: {np_before[2:]} -> {np_after[2:]}'
    assert torch.equal(torch_before, torch.get_rng_state()), \
        'augment=False 竟然消耗了 torch 随机'
    # 传进来的 Generator 自身也不能被推进：它就是 eval 的采样源，推进它 = 指标依赖种子
    after_draw = passed_rng.integers(0, 8, size=4)
    ref_rng = np.random.default_rng(4242)
    assert np.array_equal(after_draw, ref_rng.integers(0, 8, size=4)), (
        f'augment=False 仍然从传入的 rng 抽了随机：抽到 {after_draw.tolist()}，'
        f'未被消耗时应为 {np.random.default_rng(4242).integers(0, 8, size=4).tolist()}')


# --------------------------------------------------------------------------- #
# 3. augment=True 仍按同一个 t 同步变换 states 与 moves（默认路径没被改坏）
# --------------------------------------------------------------------------- #
def test_augment_true_still_transforms_consistently():
    """钉死 `tforms ≡ 4`（W 轴翻转）：`states` 与 `moves` 必须按**同一个** t 变换。

    这是「两侧共享同一个 `tforms`」这条耦合的锁 —— 若哪一侧忘了用 tforms，moves 就会
    与棋盘对不上（监督目标指向被翻转前的点），而这种错误在训练里表现为悄悄学坏。

      · `states` 应逐位等于 `raw[:, :, :, ::-1]`（t=4 → flip W，k=4%4=0 → 不旋转）
      · `moves` 应逐位等于 `r*bs + (bs-1-c)`（`SYMMETRIES[4]` = W 轴翻转）
      · 两者都与 `augment=False` 的输出**不同**（证明增强确实在生效，且默认路径没被改坏）
    """
    ds = _make_dataset()
    idxs = np.arange(len(ds))
    raw = _raw_features(ds, idxs)

    rng = _FixedTformRng(4)
    states, moves_out, _ = ds.sample_batch_numpy(idxs, rng=rng, augment=True)
    assert rng.calls == 1, \
        f'augment=True 应从传入的 rng 抽一次 tforms，实得 {rng.calls} 次'

    assert np.array_equal(states, raw[:, :, :, ::-1]), (
        'tforms=4 时 states 应恰为原始特征沿 W 轴翻转的结果 —— '
        '默认路径的对称变换被改坏了')
    mv = np.asarray(SAMPLE_MOVES, dtype=np.int64)
    valid = (mv >= 0) & (mv < BS2)
    r, c = np.divmod(np.where(valid, mv, 0), BOARD)
    want = np.where(valid, r * BOARD + (BOARD - 1 - c), BS2)
    assert np.array_equal(moves_out, want), (
        f'tforms=4 时 moves 应按 (r,c) -> (r, bs-1-c) 重映射: '
        f'{moves_out.tolist()} != {want.tolist()}')

    # --- 与 augment=False 的输出必须不同（否则上面两条可能只是恒等变换） ---
    no_states, no_moves, _ = ds.sample_batch_numpy(idxs, augment=False)
    assert not np.array_equal(states, no_states), \
        'augment=True 与 augment=False 的 states 相同 → 钉 tforms 没起作用'
    assert not np.array_equal(moves_out, no_moves), \
        'augment=True 与 augment=False 的 moves 相同 → 钉 tforms 没起作用'
    assert np.array_equal(moves_out[list(INVALID_MOVES)],
                          np.full(len(INVALID_MOVES), BS2, dtype=np.int64)), \
        'tforms=4 时非法/越界标签应仍归一到 bs*bs'


# --------------------------------------------------------------------------- #
# 4. 非法/越界标签的归一化与对称变换解耦
# --------------------------------------------------------------------------- #
def test_invalid_move_normalized_in_both_modes():
    """`moves == -1`（pass）与越界标签在**两种 augment 取值**下都归一到 `bs*bs`。

    归一化是动作空间的一部分（pass = 第 `bs*bs` 类），与「要不要翻转棋盘」无关。
    漏掉它 → eval 拿一个训练里从不出现的标签去算 top1，准确率凭空少一截。
    """
    ds = _make_dataset()
    idxs = np.arange(len(ds))

    for augment in (False, True):
        _, moves_out, _ = ds.sample_batch_numpy(idxs, augment=augment)
        got = moves_out[list(INVALID_MOVES)]
        assert np.array_equal(got, np.array([BS2, BS2], dtype=np.int64)), (
            f'augment={augment} 时非法/越界标签应都归一到 {BS2}，实得 {got.tolist()}')
        # 合法标签不受影响（否则这条会把「全归一」也算通过）
        valid_idx = [i for i in idxs if i not in INVALID_MOVES]
        assert not np.all(moves_out[valid_idx] == BS2), \
            f'augment={augment} 时合法着法也被抹成 pass 了'


# --------------------------------------------------------------------------- #
# 5. 端到端：评估指标与采样种子无关（行为性红，本任务主证据）
# --------------------------------------------------------------------------- #
def test_eval_metrics_independent_of_sampling_seed(monkeypatch):
    """两个不同 `EVAL_SAMPLING_SEED` 下 `evaluate_metrics` 的八个指标**逐位相同**。

    修复前必红（这正是本任务要消灭的性质）：P2.2 把抽样固定住了，但固定的是**哪一组**
    对称 —— 换个种子就是另一组随机翻转/旋转过的验证集，top1/top5/top10/kl/brier 全变。
    真实数据集上，eval 指标本不该与「用哪个种子」有关；它应该与「验证集原始长什么样」有关。

    P2.2 之后剩下的最后一层方差就是这一层。关掉增强后 eval **零随机**，两个种子下的
    指标必然逐位相同。

    空转守卫（第一段）：同一套假数据 + 两个不同的**显式** rng 必须给出不同指标 ——
    否则「两个种子相同」只是因为这套假数据对变换根本不敏感。
    """
    # --- 空转守卫：这套假数据对 tforms 真的敏感 ---
    guarded = []
    for seed in (11, 222):
        ds = _AugDataset(honor_augment=False)      # 无视 augment，一律抽变换
        guarded.append(_run_metrics(ds, rng=np.random.default_rng(seed)))
    assert guarded[0]['top1'] != guarded[1]['top1'], (
        f'两个不同 rng 下 top1 相同（{guarded[0]["top1"]}）—— '
        f'这套假数据对变换不敏感，用例 5 的相等断言已失去牙齿')

    # --- 主断言：换 EVAL_SAMPLING_SEED，八个指标逐位相同 ---
    runs = []
    for seed in (1234, 987654321):
        monkeypatch.setattr(t, 'EVAL_SAMPLING_SEED', seed)
        runs.append(_run_metrics(_AugDataset()))

    keys = ('top1', 'top5', 'top10', 'kl', 'brier', 'n', 'batches', 'truncated')
    for key in keys:
        assert runs[0][key] == runs[1][key], (
            f'EVAL_SAMPLING_SEED 改了但 {key} 变了: {runs[0][key]!r} vs {runs[1][key]!r} —— '
            f'eval 仍在吃 8 路随机对称增强，指标依赖采样种子')
    # 变化范围也确认一下：跑满 40 样本 / bs=4 → 10 批，truncated 必为 False
    assert runs[0]['batches'] == runs[1]['batches'] == 10
    assert runs[0]['truncated'] is False and runs[1]['truncated'] is False


# 5b. 同一条性质的**真数据集**版本：防止用例 5 的假 dataset 与真实实现语义漂移
# --------------------------------------------------------------------------- #
# 权重刻意**非对称**（arange 递增）：特征与它做加权和，翻转/旋转后和值必变 → 假模型的
# 最优着法随之变 → 指标对对称变换敏感。若换成常数权重，8 个对称下和值全一样，这条就废了。
_ASYM_WEIGHTS = torch.arange(12 * BOARD * BOARD,
                             dtype=torch.float32).reshape(1, 12, BOARD, BOARD)


class _RealStateModel:
    """吃**真** `(B,12,H,W)` 特征的假模型：最优着法 = 特征与非对称权重的加权和。

    权重是 `arange`（对 W 翻转不对称）→ 同一批样本被翻转后 score 变 → peak 变 →
    指标变。于是「关掉增强 ⇒ 指标不再依赖种子」这条性质可以在**真实数据集**上端到端
    验证，而不是只验证用例 5 那只假 dataset 的 augment 语义。
    """

    def __init__(self, n_actions=N_ACTIONS):
        self.n_actions = n_actions

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, state):
        B = state.shape[0]
        score = (state * _ASYM_WEIGHTS).reshape(B, -1).sum(dim=1)   # (B,)
        peak = score.abs().long() % self.n_actions
        ramp = -torch.arange(self.n_actions, dtype=torch.float32)
        logits = ramp.unsqueeze(0).repeat(B, 1)
        rows = torch.arange(B)
        logits[rows, peak] = 5.0
        logits[rows, (peak + 1) % self.n_actions] = 4.5
        return logits, score.unsqueeze(-1) * 1e-4


def test_real_dataset_eval_is_seed_independent(monkeypatch):
    """**真** `SupervisedDataset` 喂 `evaluate_metrics`：换 `EVAL_SAMPLING_SEED` 指标不变。

    用例 5 用的是假 dataset，它自己实现了一份 augment 语义；万一真实实现的
    `augment=False` 与那份语义不一致（例如真实实现偷偷抽了 tforms、或多做了重映射），
    用例 5 是察觉不到的。本条把真实 `SupervisedDataset` 接进真实的 `evaluate_metrics`，
    补上这一段 —— 它与用例 1/2/3/4（真实数据集的单元级语义）合起来才是端到端的保证。

    修复前必红：两个种子抽到不同的 tforms → 特征被翻转 → peak 变 → 五个指标全变。

    空转守卫：先确认这份真数据在 W 轴翻转下加权 score 真的会变（否则本条可能恒真）。
    """
    n_samples = 10 * EVAL_BS
    ds = _make_dataset(n_samples)
    idxs = np.arange(n_samples)
    raw = torch.from_numpy(_raw_features(ds, idxs))

    # --- 空转守卫：真数据 + 非对称权重 → 翻转后 score 必变 ---
    assert not torch.equal((raw * _ASYM_WEIGHTS).sum(),
                           (raw.flip(-1) * _ASYM_WEIGHTS).sum()), \
        '这份真数据在 W 轴翻转下加权 score 相同 → 本条的「种子无关」断言可能恒真'

    def _run():
        return t.evaluate_metrics(_RealStateModel(), ds, idxs, EVAL_BS, 'cpu', AMP,
                                  max_batches=0)

    runs = []
    for seed in (1234, 987654321):
        monkeypatch.setattr(t, 'EVAL_SAMPLING_SEED', seed)
        runs.append(_run())

    keys = ('top1', 'top5', 'top10', 'kl', 'brier', 'n', 'batches', 'truncated')
    for key in keys:
        assert runs[0][key] == runs[1][key], (
            f'真数据集上 EVAL_SAMPLING_SEED 改了但 {key} 变了: '
            f'{runs[0][key]!r} vs {runs[1][key]!r}')
    assert runs[0]['batches'] == runs[1]['batches'] == 10
    assert runs[0]['n'] == runs[1]['n'] == n_samples

    # --- 对照：把真实实现强行拨回 augment=True，指标必须变（证明「相等」不是因为
    #     这套数据怎么算都一样） ---
    import src.data.dataset as dsmod
    original = dsmod.SupervisedDataset.sample_batch_numpy

    def _forced_augment(self, sel, rng=None, augment=True):
        return original(self, sel, rng=rng, augment=True)

    monkeypatch.setattr(dsmod.SupervisedDataset, 'sample_batch_numpy', _forced_augment)
    try:
        forced = t.evaluate_metrics(_RealStateModel(), ds, idxs, EVAL_BS, 'cpu', AMP,
                                    max_batches=0)
    finally:
        monkeypatch.setattr(dsmod.SupervisedDataset, 'sample_batch_numpy', original)

    assert (runs[0]['kl'], runs[0]['brier']) != (forced['kl'], forced['brier']), (
        f'强制走 augment=True 后指标与关增强时相同（{forced["kl"]}, {forced["brier"]}）—— '
        f'本条的相等断言已失去牙齿')


# --------------------------------------------------------------------------- #
# 6. 结构锁：两个 eval 调用点都传 augment=False，且只有这两处（修复前必红）
# --------------------------------------------------------------------------- #
def _sample_batch_calls(fn):
    """`fn` 源码里所有 `....sample_batch_numpy(...)` 调用（AST，不写死行号）。"""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'sample_batch_numpy']


def test_eval_call_sites_pass_augment_false():
    """`evaluate_metrics` 与 `evaluate_top1` 内的 `sample_batch_numpy` 必须带 `augment=False`。

    **结构锁，按内容锚点定位**（不写死行号 —— 行号会被无关改动推着走）。漏掉任何一处
    = 只修了一半的评估路径，那种「有时生效」比不生效更难排查。断言同时钉住实参是
    `False` 字面量（写成变量或 `not augment` 都不算数）。

    本文件其余用例全靠**行为**证明增强已关；这条兜住「行为对但接线漏了」的情况 ——
    例如某处误写成 `augment=augment`，行为层的用例仍可能因为别的原因而绿。
    """
    for fname in ('evaluate_metrics', 'evaluate_top1'):
        calls = _sample_batch_calls(getattr(t, fname))
        assert len(calls) == 1, \
            f'{fname} 里应有且仅有 1 处 sample_batch_numpy 调用，实得 {len(calls)} 处'
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
        assert 'augment' in kwargs, (
            f'{fname} 的 sample_batch_numpy 调用没传 augment —— '
            f'实参为 {kwargs}（修复前就是缺这个）')
        assert kwargs['augment'] == 'False', (
            f"{fname} 传的 augment={kwargs['augment']!r}，应为 False")


def test_only_eval_call_sites_disable_augmentation():
    """全文件扫描：传 `augment=False` 的调用点**全部**落在两个 eval 函数内，其余不传。

    这是风险边界锁。`sample_batch_numpy` 的 `augment` 默认 True，训练路径靠默认值吃增强；
    若有人顺手给 `_prefetch_worker` 或别处加上 `augment=False`，训练就静默失去了 8 倍增强
    —— 训练指标照样好看，只是模型变弱，事后极难归因。

    **P2.3b 修正：这里曾断言「全文件恰好 3 处 `sample_batch_numpy` 调用」（top1 / metrics /
    预取器），已删除。** 那是一条把当前**实现形状**抄进测试的计数锁：P2.3 写进
    `_eval_rng` docstring 的合法演进路径是「将来评估若要重新开启某种抽样，显式 `rng=` 就是
    那条入口」—— 将来合法地多一个调用点，它就会红，挡住的不是回归，只是「文件长变了」。

    真实不变式与调用点**数量**无关，故按「归属 + 实参」判定（行号区间找最内层函数，不靠
    出现顺序，行号会被无关改动推着走）：

      ① 两个 eval 函数内的**每**一处调用都带 `augment=False`（字面量 False，写成变量不算）；
      ② eval 之外的调用点**不得**带 `augment=False`；
      ③ `_prefetch_worker` 内的调用连 `augment` 这个 kwarg 都不许出现（比 ② 更严：训练侧
         必须走默认值，理由同 docstring 第一段）；
      ④ eval 之外的调用点至少存在一个 —— 否则 ② 恒真，本用例空转。

    ② 只禁 `False`、不禁「显式 `True`」是有意的：训练路径将来写一句 `augment=True` 只是把
    既有默认写明白，行为不变，不该被这条边界锁拦下（那又变成另一种形状锁）。
    """
    tree = ast.parse(TRAIN_SRC_TEXT)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'sample_batch_numpy']
    assert calls, 'train_sft.py 里一处 sample_batch_numpy 调用都没有 —— 扫描本身已失效'

    def _fn_of(node):
        """`node` 所在的最内层函数名（模块级则 None）。"""
        owners = [f for f in ast.walk(tree)
                  if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and f.lineno <= node.lineno <= (f.end_lineno or f.lineno)]
        return min(owners, key=lambda f: f.lineno).name if owners else None

    def _augment_kwarg(node):
        """`augment` 实参的源码文本（没传这个 kwarg 则 None）。"""
        for kw in node.keywords:
            if kw.arg == 'augment':
                return ast.unparse(kw.value)
        return None

    EVAL_FNS = ('evaluate_metrics', 'evaluate_top1')
    inside = [n for n in calls if _fn_of(n) in EVAL_FNS]
    outside = [n for n in calls if _fn_of(n) not in EVAL_FNS]

    assert inside, \
        f'两个 eval 函数（{"/".join(EVAL_FNS)}）里一处 sample_batch_numpy 调用都没找到 —— ' \
        f'归属判定失效或函数被改名'

    # ① eval 内的每一处都必须关增强
    for n in inside:
        assert _augment_kwarg(n) == 'False', (
            f'{_fn_of(n)} 里的 sample_batch_numpy 没传 augment=False'
            f'（实参 {_augment_kwarg(n)!r}）：{ast.unparse(n)}')

    # ② eval 之外一律不许关增强
    for n in outside:
        assert _augment_kwarg(n) != 'False', (
            f'{_fn_of(n)} 里的 sample_batch_numpy 不该传 augment=False：{ast.unparse(n)} —— '
            f'训练路径靠默认 True 吃 8 倍增强，关掉它训练指标照样好看、只是模型变弱，'
            f'事后极难归因')

    # ④ 空转守卫：非 eval 调用点确实存在（预取器 worker），否则 ② 是恒真
    assert outside, \
        '一个 eval 之外的 sample_batch_numpy 调用点都没找到 —— ② 已恒真，本用例空转'

    # ③ 预取器 worker：连 augment 这个 kwarg 都不许出现
    worker_calls = _sample_batch_calls(t._prefetch_worker)
    assert len(worker_calls) == 1, \
        f'_prefetch_worker 里应有 1 处 sample_batch_numpy 调用，实得 {len(worker_calls)} 处'
    worker_kwargs = {kw.arg for kw in worker_calls[0].keywords}
    assert 'augment' not in worker_kwargs, \
        f'预取器 worker 不该传 augment（训练路径靠默认 True 吃增强），实得 {worker_kwargs}'


# --------------------------------------------------------------------------- #
# 7. 训练路径零回归：不传 augment == 显式 augment=True（含 sample_batch）
# --------------------------------------------------------------------------- #
def test_training_path_default_is_augment_true():
    """`sample_batch_numpy` 与 `sample_batch` 不传 `augment` 时，输出与显式 `True` 逐位相同。

    `augment` 的默认值就是「训练路径不变」这条承诺的全部依托。做法是把全局随机源
    复位到同一状态后分别调用两次 —— 两次抽到的 `tforms` 完全相同，于是输出逐位相同
    就等价于「默认走的是增强分支」（若默认值被改成 False，`tforms` 根本不会被抽，
    输出会与 `augment=True` 那一路差开，这条立刻红）。

    `sample_batch` 是训练真正用的入口（`train_sft.py` 预取器关闭时走它），也一并锁住。
    """
    ds = _make_dataset()
    idxs = np.arange(len(ds))
    seed = 13579

    np.random.seed(seed)
    def_states, def_moves, def_values = ds.sample_batch_numpy(idxs)
    np.random.seed(seed)
    aug_states, aug_moves, aug_values = ds.sample_batch_numpy(idxs, augment=True)

    assert np.array_equal(def_states, aug_states), \
        '不传 augment 与显式 augment=True 的 states 不一致 → 默认值不是 True'
    assert np.array_equal(def_moves, aug_moves), \
        '不传 augment 与显式 augment=True 的 moves 不一致 → 默认值不是 True'
    assert np.array_equal(def_values, aug_values)

    np.random.seed(seed)
    t_states, t_moves, t_values = ds.sample_batch(idxs)          # 训练入口
    assert np.array_equal(t_states.numpy(), aug_states), \
        'sample_batch 的 states 与增强分支不一致'
    assert np.array_equal(t_moves.numpy(), aug_moves)
    assert np.array_equal(t_values.numpy(), aug_values)

    # --- 空转守卫：默认路径确实在变换（与 augment=False 不同） ---
    no_states, no_moves, _ = ds.sample_batch_numpy(idxs, augment=False)
    assert not np.array_equal(def_states, no_states), \
        '默认路径的 states 与 augment=False 相同 → 这批样本的 tforms 全是 0，' \
        '本用例的相等断言可能是空转'
    assert not np.array_equal(def_moves, no_moves), \
        '默认路径的 moves 与 augment=False 相同 → 本用例的相等断言可能是空转'
