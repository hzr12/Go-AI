"""SFT 验证集评估的覆盖度可控（`--eval-max-batches`）与截断可见化。

缺陷
----
`scripts/train_sft.py` 的两个评估函数都把批数上限**硬编码成 50**
（`n_batches = min(ceil(len(idxs)/bs), max_batches)`），两个调用点
（训练中周期 eval、训练结束最终 eval）又都不传 `max_batches`。后果：

  ① 命令行无法调整覆盖度 —— 验证集再大也只跑前 50 批（bs=256 时上限
     12800 样本），而 `--eval-max-batches` 这个 argparse 选项根本不存在；
  ② 日志与返回值都看不出「被截断了」—— 读日志的人无从判断指标可信度；
  ③ 同一份日志里，改验证集大小不会让指标口径变化，横向比较是假的。

覆盖
----
  - 参数存在性：默认 50、可显式传值、`<=0` 能传进、紧跟 `--eval-every`
    声明、且**不得**出现在 run.txt 的 RL 参数表里（那是 selfplay_train.py
    的表，误加会同时污染 `tests/test_run_txt_sync.py` 的双向齐全断言）
  - 截断语义：上限生效时只跑上限那么多批、`<=0`/`None` 与不传都跑满验证集；
    `evaluate_top1`（本模块无调用点）也有直接调用的执行覆盖，同一套预算语义
  - 零回归：不传参数与显式 `max_batches=50` 的既有指标逐位相同
  - 截断可见化：返回值含 `batches` / `truncated`，日志行与 SwanLab 上报
    也带这两个信息（旧键一律不动）
  - 结构锁：两个 `evaluate_metrics(...)` 调用点都传了
    `max_batches=args.eval_max_batches`（漏一处 = 只修了一半）

全部 CPU、秒级：假 model（定长 logits/value）+ 假 dataset（`sample_batch_numpy`
只吐确定性 numpy 并统计被请求的批数），不加载真模型、不读真数据。
"""
import argparse
import ast
import inspect
import os
import pathlib
import re
import sys
import textwrap

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.train_sft as t                # noqa: E402

RUN_TXT = os.path.join(ROOT, 'run.txt')
SELFPLAY_SRC = os.path.join(ROOT, 'scripts', 'selfplay_train.py')
TRAIN_SRC = os.path.join(ROOT, 'scripts', 'train_sft.py')
SRC = open(TRAIN_SRC, encoding='utf-8').read()
BS = 4              # 假 batch size：小到「批数」与「样本数」不会混淆
N_ACTIONS = 32      # 假动作数（须 >= 10，evaluate_metrics 内部 topk(10)）
AMP = torch.float32


# --------------------------------------------------------------------------- #
# 假 model / 假 dataset
# --------------------------------------------------------------------------- #
class _FakeDataset:
    """按 `sel` 吐确定性样本，并统计被请求的批数与每次的批大小。

    特征里编码样本 id（`states[i,0,0,0] = i`），假模型据此产出确定性 logits，
    于是同一输入的指标完全可复现，跨调用逐位可比。
    """

    def __init__(self, n_samples, n_actions=N_ACTIONS):
        self.n_samples = n_samples
        self.n_actions = n_actions
        self.batch_sizes = []

    @property
    def calls(self):
        """被请求的批数（= 评估函数实际跑的批数）。"""
        return len(self.batch_sizes)

    def sample_batch_numpy(self, sel, rng=None, augment=True, labels=False):
        """`rng` 是评估侧固定采样源传进来的随机源（P2.2 新增），本假 dataset **刻意不用它**。

        本模块锁的是「批数预算 / 截断可见化」，数值必须由 `sel` 单独决定，才谈得上
        「跑满 vs 截断」「默认 vs 显式 50」的逐位比较 —— 掺进抽样噪声后那些相等断言就
        只是在比噪声。抽样确定性由 `tests/test_eval_determinism.py` 负责。

        `augment` 同理只为签名兼容而存在（真实实现新增了它，评估路径传 False），本假
        dataset 刻意不用 —— 它要能对同一份 `sel` 重复给出同一批数值。

        `labels` 同理只为签名兼容而存在（真实实现的 `labels=True` 会多返一个
        labels_dict，见 dataset.py 的 4 元组契约）。评估路径永远传 False，
        形参存在只是为了让 `train_sft.py` 任何一处传 `labels=True` 都不 TypeError。
        """
        sel = np.asarray(list(sel), dtype=np.int64)
        B = len(sel)
        self.batch_sizes.append(B)
        states = np.zeros((B, 1, 1, 1), dtype=np.float32)
        states[:, 0, 0, 0] = sel.astype(np.float32)
        moves = (sel * 7) % self.n_actions
        values = np.where(sel % 2 == 0, 1.0, -1.0).reshape(B, 1).astype(np.float32)
        return states, moves.astype(np.int64), values


class _FakeModel:
    """定长 logits/value 的假模型：最优着法由 state 里的样本 id 决定。

    logits 取**互不相同**的值（rank1=5.0、rank2=4.5、其余是 -0,-1,... 的斜坡），
    没有并列 → topk 结果唯一确定，指标不依赖 torch 版本怎么打破并列。真值着法
    随样本 id 落在不同 rank 上，于是「评估了前多少个样本」会真实改变指标：
    截断与跑满的数值因此必然不同（否则相等只是因为都没截断）。
    """

    def __init__(self, n_actions=N_ACTIONS):
        self.n_actions = n_actions
        self.batches = 0

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, state):
        self.batches += 1
        B = state.shape[0]
        ids = state[:, 0, 0, 0].long()
        ramp = -torch.arange(self.n_actions, dtype=torch.float32)
        logits = ramp.unsqueeze(0).repeat(B, 1)
        rows = torch.arange(B)
        peak = ids % self.n_actions
        logits[rows, peak] = 5.0
        logits[rows, (peak + 1) % self.n_actions] = 4.5
        return logits, torch.full((B, 1), 0.25)


def _run_eval(n_samples, **kwargs):
    """跑一次 evaluate_metrics，返回 (metrics, dataset, model)。"""
    ds = _FakeDataset(n_samples)
    model = _FakeModel()
    idxs = np.arange(n_samples)
    kwargs.setdefault('device', 'cpu')
    kwargs.setdefault('amp_dtype', AMP)
    metrics = t.evaluate_metrics(model, ds, idxs, BS, **kwargs)
    return metrics, ds, model


# --------------------------------------------------------------------------- #
# 1. 参数：存在、默认 50、可传值、位置正确、不得进 run.txt 的 RL 参数表
# --------------------------------------------------------------------------- #
def _build_parser():
    """重放 main() 里的 `ap.add_argument(...)` 语句，拿到语义等价的 parser。

    main() 内联建 parser 且紧接着就加载数据集起训练，直接调用代价太大；
    只把 add_argument 语句重放一遍，`parse_args` / `--help` 都可用，且完全
    离线秒级（不 import 真训练流程）。作用域带模块 globals，以防某个默认值
    引用了模块级常量。
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(t.main)))
    ap = argparse.ArgumentParser()
    scope = dict(vars(t))
    scope['ap'] = ap
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            expr = ast.Expression(body=node)
            ast.copy_location(expr, node)
            exec(compile(expr, '<train_sft argparse>', 'eval'), scope)
    return ap


def test_v7_shard_dir_in_run_txt_matches_the_converter_output():
    """🔴 run.txt 里的 C 段 `--data` 必须**就是**转换器的输出目录。

    2026-10-04 踩过：转换器把 8 个分片写在 `data/` 根下，而 C 段命令写的是
    `--data data/stdata_v7`（那个目录**从来不存在**）⇒ 命令一跑就报
    「既不是文件也不是目录」。

    而它不能简单改成 `--data data`：`resolve_shards` 对目录是
    「收集其中所有 `*.npz`」，而 `data/` 根下还有 `sgf_19x19_full.npz`
    和 `labels/`，会把它们也当分片 ⇒ `V7PackedDataset._check_layout` 当场报错。

    ⇒ 约定：**分片落在 `data/stdata/` 这个专用目录里**，转换器 `--out` 给
    `data/stdata/stdata_v7.npz`，C 段 `--data data/stdata`。
    """
    txt = (pathlib.Path(ROOT) / 'run.txt').read_text(encoding='utf-8')
    assert '--out data/stdata/stdata_v7.npz' in txt, \
        '转换器 --out 应指向 data/stdata/ 目录内'
    for ln in txt.splitlines():
        if '--data data/stdata' in ln:
            assert '--data data/stdata_v7' not in ln, \
                f'C 段仍指向不存在的目录：{ln.strip()}'
    # 真实存在的那个目录必须只含分片（否则 resolve_shards 会把别的 npz 也收进来）
    d = pathlib.Path(ROOT) / 'data' / 'stdata'
    if d.is_dir():
        others = [p.name for p in d.iterdir()
                  if p.is_file() and p.suffix == '.npz'
                  and not p.name.startswith('stdata_v7_s')]
        assert not others, f'data/stdata 下混进了非分片 npz：{others}'


def test_eval_max_batches_flag_parses():
    """`--eval-max-batches` 必须是真参数：默认 50、int、能传 0/负数/help 正常。

    声明位置也锁：紧跟 `--eval-every` 之后、且在 `ap.parse_args()` 之前 ——
    加到别的脚本或段落里，replay 出来的 parser 不会有它。
    """
    ap = _build_parser()
    args = ap.parse_args(['--data', 'dummy.npz'])          # --data 是必填项
    assert args.eval_max_batches == 50, \
        f"--eval-max-batches 默认应为 50（= 今天的硬编码值，零回归），实得 {args.eval_max_batches}"
    assert isinstance(args.eval_max_batches, int)

    for raw, want in (('8', 8), ('0', 0), ('-1', -1), ('100000', 100000)):
        got = ap.parse_args(['--data', 'dummy.npz',
                             '--eval-max-batches', raw]).eval_max_batches
        assert got == want, f'--eval-max-batches {raw} 解析成 {got}，应为 {want}'

    # --help 必须正常打印，且把「<=0 = 不截断」的语义写给用户看。
    # rfind：usage 摘要里也出现一次该选项名，要看的是后面「选项说明」段那一条。
    flat = ' '.join(ap.format_help().split())
    j = flat.rfind('--eval-max-batches')
    assert j != -1, '该参数没出现在 --help 里'
    win = flat[j:j + 200]
    assert '<=0' in win, f'--help 未写明 <=0 的语义: {win}'
    assert '截断' in win and '验证集' in win, \
        f'--help 未写明「验证集 / 截断」: {win}'

    src = textwrap.dedent(inspect.getsource(t.main))
    i_every = src.find("'--eval-every'")
    i_new = src.find("'--eval-max-batches'")
    i_parse = src.find('ap.parse_args()')
    assert i_every != -1 and i_new != -1 and i_parse != -1, (
        '源码里找不到 --eval-every / --eval-max-batches / ap.parse_args —— '
        '若日后把 argparse 抽成 build_parser()，本测试与 _build_parser 需同步改')
    assert i_every < i_new < i_parse, (
        '--eval-max-batches 应声明在 --eval-every 之后、ap.parse_args() 之前，'
        f'实际位置 {i_every} / {i_new} / {i_parse}')


def test_eval_max_batches_absent_from_run_txt_rl_table():
    """新旗标不得进 run.txt 的 ③' RL 参数表（那是 selfplay_train.py 的表）。

    顺带把「新增 train_sft 选项不影响 test_run_txt_sync」这条前提本身锁住：
    该测试的参数集是从 selfplay_train.py 抽的，所以只要 `--eval-max-batches`
    不是 selfplay_train 的选项，它就既不需要文档行、也不会引起双向齐全失败。
    """
    sp = open(SELFPLAY_SRC, encoding='utf-8').read()
    opts = set(re.findall(r'ap\.add_argument\(\s*[\'"](--[a-z0-9-]+)[\'"]', sp))
    assert '--eval-max-batches' not in opts, \
        'selfplay_train.py 竟然也有 --eval-max-batches —— 那 run.txt 若照抄它就要求它有文档行'

    # 原实现是「在 run.txt 的 ③' RL 完整参数表（照抄 selfplay_train.py 的那段）
    # 里不许出现」，把上面这个源码级检查做了双保险。但 run.txt 已改为一屏简洁版
    # （7d1d5fd），不再承载逐项参数表，那一节不存在了 ⇒ 断言静默退化成对一个
    # 不存在的节做 find，检查力归零却不报红 —— 这类「锚点消失」比断言红更危险。
    # 改为**整篇**不许出现：既不再依赖已退役的小节结构，守卫也比原来更强 ——
    # train_sft 没有这个选项，那 run.txt 里任何位置都不该教人用它。
    txt = open(RUN_TXT, encoding='utf-8').read()
    assert '--eval-max-batches' not in txt, \
        'run.txt 不得出现 --eval-max-batches（train_sft.py 没有这个 argparse 选项）'


# --------------------------------------------------------------------------- #
# 2. 截断语义：上限生效 / <=0 与 None 不截断 / 不传即默认（两个评估函数）
# --------------------------------------------------------------------------- #
def test_evaluate_metrics_respects_cap():
    """40 样本 / bs=4 → 验证集共 10 批。

      · max_batches=3    → 只跑 3 批、n=12、truncated=True
      · max_batches=0    → 跑满 10 批、n=40、truncated=False
      · max_batches=-1   → 同 0
      · max_batches=None → 同 0（本仓既有约定：`train_sft_ms.py` 的
        `evaluate_top1(..., max_batches=None)` 就是「跑满」的传法）
      · 不传             → 默认 50 > 10 → 跑满 10 批、truncated=False
    """
    n_samples = 10 * BS

    m3, ds3, _ = _run_eval(n_samples, max_batches=3)
    assert ds3.calls == 3, f'上限 3 却跑了 {ds3.calls} 批'
    assert m3['n'] == 3 * BS, f"n={m3['n']}，应为 {3 * BS}"
    assert m3['truncated'] is True, '因上限少跑了批，truncated 必须为 True'

    # None 与 <=0 同义：跑满整个验证集（`max_batches is None` 是独立分支，
    # 漏测它等于让「显式传 None」这条调用方式零覆盖）。
    for cap in (0, -1, None):
        m, ds, _ = _run_eval(n_samples, max_batches=cap)
        assert ds.calls == 10, f'max_batches={cap} 应跑满 10 批，实跑 {ds.calls} 批'
        assert m['n'] == n_samples, f"max_batches={cap}: n={m['n']}，应为 {n_samples}"
        assert m['truncated'] is False, f'max_batches={cap} 未截断，truncated 不该为 True'

    m_def, ds_def, _ = _run_eval(n_samples)
    assert ds_def.calls == 10, f'不传参数（默认 50 > 10）应跑满，实跑 {ds_def.calls} 批'
    assert m_def['n'] == n_samples
    assert m_def['truncated'] is False


def _run_top1(n_samples, **kwargs):
    """跑一次 evaluate_top1，返回 (accuracy, num_samples, dataset)。"""
    ds = _FakeDataset(n_samples)
    model = _FakeModel()
    idxs = np.arange(n_samples)
    kwargs.setdefault('device', 'cpu')
    kwargs.setdefault('amp_dtype', AMP)
    acc, n = t.evaluate_top1(model, ds, idxs, BS, **kwargs)
    return acc, n, ds


def test_evaluate_top1_shares_budget_semantics():
    """直接调用 `evaluate_top1`：它与 `evaluate_metrics` 共用同一批数预算。

    train_sft.py 内它没有调用点（见 `test_evaluate_top1_has_no_call_site_in_module`），
    所以这是它唯一的执行覆盖：上限生效、`<=0` 与 `None` 跑满、不传 ≡ 显式 0。
    返回形状按历史契约仍是 `(accuracy, num_samples)` 二元组（截断信息只走日志）。
    """
    n_samples = 10 * BS

    acc3, n3, ds3 = _run_top1(n_samples, max_batches=3)
    assert ds3.calls == 3, f'上限 3 却跑了 {ds3.calls} 批'
    assert n3 == 3 * BS, f'n={n3}，应为 {3 * BS}'

    for cap in (0, -1, None):
        _, n, ds = _run_top1(n_samples, max_batches=cap)
        assert ds.calls == 10, f'max_batches={cap} 应跑满 10 批，实跑 {ds.calls} 批'
        assert n == n_samples, f'max_batches={cap}: n={n}，应为 {n_samples}'

    # 不传（默认 50 > 10）≡ 显式 0：加 max_batches 参数没改它的数值
    acc_def, n_def, ds_def = _run_top1(n_samples)
    acc_0, n_0, ds_0 = _run_top1(n_samples, max_batches=0)
    assert (ds_def.calls, n_def) == (ds_0.calls, n_0) == (10, n_samples)
    assert acc_def == acc_0, \
        f'默认与显式 0 的准确率不一致: {acc_def!r} vs {acc_0!r}（数值必须逐位不变）'
    assert acc_def != acc3, \
        '截断与跑满的准确率竟然相同 —— 假数据失去区分度，这条测试已失效'


# --------------------------------------------------------------------------- #
# 3. 零回归：默认 ≡ 显式 50（覆盖度真的被截断的场景下也比）
# --------------------------------------------------------------------------- #
def test_default_matches_explicit_50():
    """120 批验证集（>50）下，不传参数与显式 `max_batches=50` 逐位相同。

    这条是「新增只加信息、不改数值」的锁：既有 6 个指标（top1/top5/top10/
    kl/brier/n）一个都不能动。同时对比「跑满」确认这批样本上截断**确实**会
    改变指标 —— 否则相等只是因为两者都没截断，这条测试就是空转。
    """
    n_samples = 120 * BS
    old_keys = ('top1', 'top5', 'top10', 'kl', 'brier', 'n')

    m_def, ds_def, _ = _run_eval(n_samples)
    m_50, ds_50, _ = _run_eval(n_samples, max_batches=50)

    assert ds_def.calls == ds_50.calls == 50, \
        f'两个调用应各跑 50 批，实得 {ds_def.calls} / {ds_50.calls}'
    for k in old_keys:
        assert m_def[k] == m_50[k], \
            f'默认与显式 50 的 {k} 不一致: {m_def[k]!r} vs {m_50[k]!r}（既有指标必须逐位不变）'

    m_full, ds_full, _ = _run_eval(n_samples, max_batches=0)
    assert ds_full.calls == 120, '跑满时应为 120 批'
    assert m_full['n'] == n_samples, f'跑满的样本数应为 {n_samples}，实得 {m_full["n"]}'
    assert m_full['n'] != m_def['n'], \
        '跑满的样本数必须与截断到 50 批不同，否则上面的相等是空转'
    assert m_full['top1'] != m_def['top1'] or m_full['brier'] != m_def['brier'], \
        '跑满与截断的指标竟然完全一样 —— 假数据失去区分度，这条测试已失效'


# --------------------------------------------------------------------------- #
# 4. 截断可见化：返回值 / 日志 / SwanLab
# --------------------------------------------------------------------------- #
def test_metrics_report_truncation():
    """返回值必须含 `batches`（实际跑的批数）与 `truncated`（是否因上限少跑）。

    修复前必红：两个键都不存在。`batches` 在循环内按**实际**跑掉的批数计数，属于
    防御性设计：批数预算已用 `ceil` 把尾部残批算作整一批，而循环里的 `break` 只在
    `sel` 为空时触发（非空 idxs 下不可达），所以它当前恒等于计划批数；按实际计数
    是为了将来循环若可能提前退出，报出来的仍是真值。
    """
    n_samples = 10 * BS

    m, ds, _ = _run_eval(n_samples, max_batches=3)
    assert 'batches' in m, f"返回值缺少 'batches'，实测键集: {sorted(m)}"
    assert 'truncated' in m, f"返回值缺少 'truncated'，实测键集: {sorted(m)}"
    assert m['batches'] == ds.calls == 3, \
        f"batches={m['batches']}，实际跑了 {ds.calls} 批"
    assert m['truncated'] is True

    m_full, ds_full, _ = _run_eval(n_samples, max_batches=0)
    assert m_full['batches'] == ds_full.calls == 10
    assert m_full['truncated'] is False

    # 既有 6 个键必须原样保留（旧键一律不动）
    assert set(('top1', 'top5', 'top10', 'kl', 'brier', 'n')) <= set(m)
    # 尾部残批：残批必须真的被跑到并计入 batches / n（否则 n 会少 1）
    n_tail = 3 * BS + 1
    want_tail_batches = -(-n_tail // BS)                     # 向上取整
    tail_size = n_tail - (want_tail_batches - 1) * BS        # 末批的样本数
    m_tail, ds_tail, _ = _run_eval(n_tail, max_batches=0)
    assert m_tail['batches'] == ds_tail.calls == want_tail_batches, \
        (f'{n_tail} 样本 / bs={BS} 应跑满 {want_tail_batches} 批'
         f'（末批 {tail_size} 个样本），实得 {m_tail["batches"]} 批')
    assert m_tail['n'] == n_tail, \
        f'n 应为 {n_tail}（含末批那 {tail_size} 个样本），实得 {m_tail["n"]}'


def test_eval_log_and_swanlab_report_truncation():
    """`[eval]` 日志行（周期 + 最终）与 SwanLab 上报都要带批数/截断信息。

    **结构锁**：这两处是读者唯一能看到的「这次指标只覆盖了多少验证集」的地方，
    缺了它截断就是静默的 —— 而这正是本任务要消灭的失败模式。按内容锚点
    (`"[eval] step=%d` / `"[eval] FINAL`) 定位，不用行号。
    """
    main_src = textwrap.dedent(inspect.getsource(t.main))
    for anchor in ('"[eval] step=%d', '"[eval] FINAL'):
        i = main_src.find(anchor)
        assert i != -1, f'main() 里找不到 {anchor} 日志行'
        region = main_src[i:i + 700]
        assert 'batches=%d' in region, f'{anchor} 日志行未打印实际批数字段'
        assert 'truncated=%s' in region, f'{anchor} 日志行未打印是否截断字段'
        assert "['batches']" in region and "['truncated']" in region, \
            f'{anchor} 日志行的批次/截断字段没有取自返回值'

    assert '"eval_batches"' in SRC, 'SwanLab 未上报 eval_batches'
    assert '"eval_truncated"' in SRC, 'SwanLab 未上报 eval_truncated'
    # 旧键一律不动
    for key in ('"eval_top1"', '"eval_top5"', '"eval_top10"', '"eval_kl"',
                '"eval_brier"', '"eval_n"', '"best_eval_acc"',
                '"final_top1"', '"final_kl"', '"final_brier"'):
        assert key in SRC, f'SwanLab 旧键 {key} 被动过'


# --------------------------------------------------------------------------- #
# 5. 结构锁：两个 evaluate_metrics 调用点都接线（修复前必红）
# --------------------------------------------------------------------------- #
_MAIN_TREE = ast.parse(textwrap.dedent(inspect.getsource(t.main)))
# `inspect.getsource(main)` 的行号从 1 起算、且首行被 dedent 掉，与真实文件差一个
# 固定偏移；断言消息要报真实行号，否则会指向一个根本不存在的行。
_MAIN_OFFSET = next(i for i, line in enumerate(SRC.splitlines(), 1)
                    if line.startswith('def main(')) - 1


def _file_line(node):
    """把 main() 源码内的相对行号换算成 train_sft.py 的真实行号。"""
    return node.lineno + _MAIN_OFFSET


def _call_sites(func_name):
    return [n for n in ast.walk(_MAIN_TREE)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == func_name]


def test_both_call_sites_pass_the_flag():
    """`evaluate_metrics` 的每个调用点都必须传 `max_batches=args.eval_max_batches`。

    **结构锁，按内容锚点定位（不写死行号）**：行号会被无关改动推着走。漏掉
    任何一个调用点 = 用户改了参数却只有一半的评估听他的，那种「有时生效」的
    行为比不生效更难排查。
    """
    calls = _call_sites('evaluate_metrics')
    assert len(calls) == 2, \
        f'main() 里应有 2 个 evaluate_metrics 调用点（周期 eval + 最终 eval），实际 {len(calls)}'

    for c in calls:
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in c.keywords}
        assert 'max_batches' in kwargs, (
            f'train_sft.py:{_file_line(c)} 的 evaluate_metrics 没传 max_batches —— '
            f'该调用点会退回硬编码的 50，--eval-max-batches 对它无效: '
            f'evaluate_metrics({ast.unparse(c)})')
        assert kwargs['max_batches'] == 'args.eval_max_batches', (
            f'train_sft.py:{_file_line(c)} 的 max_batches 传的是 '
            f'{kwargs["max_batches"]!r}，应为 args.eval_max_batches')


def test_evaluate_top1_has_no_call_site_in_module():
    """复核（不是本任务的锁）：train_sft.py 里 `evaluate_top1` 无任何调用点。

    函数体已与 `evaluate_metrics` 共用同一个批数预算函数（`<=0` 语义一致），
    但返回值形状保持 `(accuracy, num_samples)` 不变 —— 截断信息只走日志。
    若将来要接它，这两件事都已就位，本测试的复核前提随之失效、应改写。
    """
    tree = ast.parse(SRC)
    calls = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'evaluate_top1']
    defs = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_top1']
    assert len(defs) == 1, f'应恰好有 1 个 evaluate_top1 定义，实际 {len(defs)}'
    assert not calls, \
        f'train_sft.py:{calls} 现在有 evaluate_top1 调用点了 —— 本测试的复核前提变了'
