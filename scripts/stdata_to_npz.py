# -*- coding: utf-8 -*-
"""stdata 归档 → **单个** `.npz`（22 空间通道 + 19 全局通道，直接喂 V7 训练）。

它替代的是什么
--------------
官方 distributed-training 数据是一堆**归档里的 npz**（`katago/stdata/*.tgz` /
`*.tar`），十万量级的成员、每成员几十行。要拿它训 V7，得先把 19×19 的行挑出来、
把标签整理成 `KataGoV7Loss` 认的形状。这一步以前只存在于
`smoke_train_v7.py::load_rows` 那种「读进内存立刻用掉」的临时函数里。

 **为什么必须是单个 `.npz`，而不是 memmap `.npy`**
----------------------------------------------------
云端训练机 256 GB RAM 的用法是「**父进程整份载入 → fork → COW 共享给
worker**」。所以：
  · 要的是一个**能整份读进内存**的文件（`.npz` 满足）；
  · **不能**产出需要 `mmap_mode='r'` 分页的中间物（`feature_v7_gather._reject_npz`
    明确拦着拿 npz 当 memmap 用，而 `.npy` 会诱使下游走那条路）；
  · **磁盘紧张 ⇒ 不许留中间物**：一次写成，缓冲只驻留内存，崩了删掉重来，
    不生成第二份全量副本。

 **`globalTargetsNC` 的列数随网络版本变 —— 而且是「逐成员」变的**
--------------------------------------------------------------
实测三个归档里**混着多种布局**（`2026-08-25npzs.tgz` 全量 57,386 个成员）：

    ===================================  =======  ========  ==========
    批次（= tar 里的目录名）                80 列    64 列      行数
    ===================================  =======  ========  ==========
    ``kata1-tf3-b11c768-s11001M``          27,393        0  1,608,110
    ``kata1-zhizi-b40c768nbt-s11472M``     21,315    8,678  1,756,156
    ===================================  =======  ========  ==========

`2026-07-30npzs.tgz` 同样（15,825 / 672 个成员），`zzb28c512` 全是 64 列。
⇒ **同一个网络名在同一个归档里同时产出 64 列与 80 列的数据**，而
`GLOBAL_TARGET_LAYOUT['kata1-zhizi-b40c768nbt']['cols'] = 64` 与数据脱节
⇒ 按「归档 → 网络」分派会在换成员时抛「登记为 64 列，实测 80 列」，
**整批转换直接失败**（本脚本开发时真的撞到了）。
⇒ 所以 :func:`resolve_member_network` **逐成员**分派：列数唯一就直接定；
列数有歧义（64 有两个网络）就用**归档路径里的批次名**消歧；都定不下来就报错。

 **本文件里一个列号字面量都没有**（:data:`GLOBAL_TARGET_COLUMNS` 是唯一的
列号出处，每个都引用 `katago_npz` 的 ``COL_*`` 常量）—— 硬编码会在换批次时
**静默错位**：不报错，只是权重列读成了别的语义。

 **输入保持 bit-packed `(N,22,46) uint8`**
-----------------------------------------
`to_v7_labels` 会把空间通道解包成 `(N,22,19,19)` float32（方便对拍与喂 loss），
但那**不是存储格式**：解包后每行 31.8 KB，3.2M 行 = 101 GB；packed 只要 3.24 GB。
转换期解包一次是对的，存回去必须还是 packed。

 **policy 目标存 top-16 稀疏**（rank + 权重两列）
-------------------------------------------------
稠密 `(N,362)` 在 3.2M 行下是 2.3 GB，top-16 只要 410 MB。
计算仍在 362 维稠密分布上做（`katago_v7_loss.policy_dense_from_sparse`），
存储才截断。

 **ch18 / ch19 按原样写入，不在转换期「修」；`komi` 列已实测不是「本局贴目」**
------------------------------------------------------------------------
ch18/ch19：官方算 area 前先提死子，本仓 `GoBoard.score()` 不做（spec §5.1 刻意
保留）⇒ 与本仓口径**已知不一致**，且**绝对量是批次特定的**（换归档不复现）。
转换器若在这里"修正"，等于凭空发明第三种偏差。本脚本按原样写入，并把该批次
标进 `meta_json`（逐通道资格表直接从 `scripts/crosscheck_stdata.py` 的
`SPATIAL_SPEC` / `GLOBAL_SPEC` 引用，**不重抄一份** —— 两份资格表迟早分叉）。

`komi`：`katago_npz.COL_KOMI = 47` 读到的其实是 **selfKomi**（相对当前行棋方、
带官方 draw-jitter）—— 实测三个归档上 `globalTargetsNC[:,47]` 与
`globalInputNC[:,5] × 20` **逐行完全相等**（匹配率 1.0000），而后者正是
`crosscheck_stdata._global_ch18` 认定的 selfKomi。`katago_npz` 的 docstring 写
「47 komi（±7.5）」—— 在「全是 komi 7.5」的批次上碰巧成立。
本脚本**不修**（不猜贴目在哪一列），只把它圈进 :func:`check_plausibility`
的物理可能域并**逐批打标记**。

此外 `globalTargetsNC[:, 0:3]` 实测是**软的三分类分布**（行和恒为 1，大量行
`max < 0.999`），不是 `to_v7_labels` 注释里说的「硬 one-hot + 全零标记和棋」；
`to_v7_labels` 的 `argmax` 把置信度丢了，「全 0 ⇒ 无结果」分支从不触发。
这两条都属 `src` 的问题，**只报告、不动手**。

`game_ids` 为什么是**每行一个不同的值**
--------------------------------------
stdata 是**逐行独立样本**：没有着法序列，行与行之间没有任何已知的先后关系；
19×19 过滤还会把归档**打碎**（丢掉 25~37% 的行），输出数组里相邻两行更是毫无
关系。而邻行 gather（`feature_v7_gather.gather_neighbors`）的跨局守卫
`game_ids[j] == game_ids[i]` 一旦放行，就会拿到**看起来完全合法的别人的盘面**
（它就是某个真实盘面），只是不属于这一手 ⇒ 静默错标签。
⇒ 所以 `game_ids` **逐行唯一**：这不是冗余列，它是跨局守卫的**载体**，让下游
任何 `gather_neighbors(..., game_ids=z['game_ids'])` 都必然判不可用。
真要重建 ch15/ch16/futurepos 的邻行关系，只能回到有 SGF 的那 34.2M 行语料。

用法
----
::

    # 第 0 遍：只数行数（便宜；不解其余 6 个成员）
    python scripts/stdata_to_npz.py --archive katago/stdata/2026-08-25npzs.tgz \\
        --network kata1-tf3-b11c768 --count-only

    # 小样本先验结构
    python scripts/stdata_to_npz.py --archive katago/stdata/2026-08-25npzs.tgz \\
        --network kata1-tf3-b11c768 --limit 2000 --out tmp/stdata_2000.npz

    # 三个归档合一个文件（每个归档各自的 --network —— 列数不同的归档就是这么对上的）
    python scripts/stdata_to_npz.py --out tmp/stdata_v7.npz \\
        --source katago/stdata/2026-08-25npzs.tgz:kata1-tf3-b11c768 \\
        --source katago/stdata/2026-07-30npzs.tgz:kata1-zhizi-b40c768nbt \\
        --source katago/stdata/zzb28c512nfd4-s8264801024-d4596884264.tar:zzb28c512nfd4
"""

import argparse
import io
import json
import os
import sys
import tarfile
import time
import zipfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.crosscheck_stdata import (  # noqa: E402
    ALIGNED,
    GLOBAL_SPEC,
    KNOWN_DIVERGENT,
    NOT_COMPARABLE,
    SPATIAL_SPEC,
)
from src.data.katago_npz import (  # noqa: E402
    BOARD_STRIDE,
    COL_FINAL_SCORE,
    COL_GLOBAL_WEIGHT,
    COL_KOMI,
    COL_LEAD,
    COL_LOSS,
    COL_NORESULT,
    COL_SCORE_MEAN,
    COL_VAR_TIME_LEFT,
    COL_WIN,
    COL_W_FUTUREPOS,
    COL_W_LEAD,
    COL_W_OWNERSHIP,
    COL_W_POLICY_OPP,
    COL_W_POLICY_PLAYER,
    COL_W_SCORING,
    COL_W_VALUE,
    GLOBAL_CHANNELS,
    GLOBAL_TARGET_LAYOUT,
    PACKED_BYTES,
    SCORE_DISTR_BINS,
    SPATIAL_CHANNELS,
    board_size_from_packed,
    read_katago_npz,
    to_v7_labels,
    unpack_binary_input,
)

#: 目标模型（**记录用**：npz 本身与模型无关，22 空间 + 19 全局是 stdata 与
#: b11c256h4nbttflrs 的共同输入约定；写在这里是为了让下游不必猜这份数据喂谁）。
TARGET_MODEL = 'b11c256h4nbttflrs'

#: schema 版本。**改动任何键名 / 形状 / 语义都要 +1** —— 下游据此拒绝读旧文件，
#: 而不是靠"字段差不多就凑合读"。
SCHEMA_VERSION = 1

#: policy 稀疏目标的 K（rank 列与权重列各 K 个）。
POLICY_TOPK = 16

#: 转换期峰值内存 ÷ 未压缩总量的**实测**余量。
#:
#: 实测（Windows / 16 逻辑核 / 13.9 GB，``2026-08-25npzs.tgz``）：
#: ``--limit 100000`` 峰值 1.50 GB、``--limit 400000`` 峰值 5.72 GB ⇒ 每行
#: 14.0–14.6 KB，而 spec 算出的未压缩是 11.9 KB/行 ⇒ **余量约 19%**。
#: 那一部分来自 zip/deflate 的工作缓冲、`np.empty` 的对齐与页碎片。
#:
#: **别把它删掉当"保守估计"**：``--ram-budget-gb`` 是唯一能在开跑前拦住
#: "装不下"的闸门，按未压缩量估会在临界配置上**放行一个必然 OOM 的任务** ——
#: 而 OOM 发生在写了几个 GB 之后，现场既没有堆栈也没有块边界可查。
RAM_OVERHEAD_FACTOR = 1.19

BOARD = BOARD_STRIDE
CELLS = BOARD * BOARD

DEFAULT_ARCHIVE = os.path.join('katago', 'stdata', '2026-08-25npzs.tgz')
DEFAULT_NETWORK = 'kata1-tf3-b11c768'

#: 每行的未压缩字节数（`float32` 平面 + packed 输入 + 稀疏 policy）的量级，
#: 只用于报错文案里的换算；**真实数字用 `uncompressed_bytes_estimate(spec, N)`**。
BYTES_PER_ROW = 11_500


class ConversionError(RuntimeError):
    """转换层面的失败（归档不在 / 列布局对不上 / 写失败）。⇒ **报错退出**。"""


# --------------------------------------------------------------------------- #
# 全局目标列：语义 → 列号。**本文件里唯一的列号出处。**
# --------------------------------------------------------------------------- #
#: ``globalTargetsNC`` 里「语义名 → 列号」的唯一映射。
#:
#: **这里没有任何字面量**：每个列号都引用 `src/data/katago_npz.py` 的
#: ``COL_*`` 常量。那张常量表是在两个网络版本（64 列 / 80 列）上分别核对过的。
#: 若哪天实测发现某版本列号不同构，**改 `katago_npz.py` 的常量表**，不要在这里
#: 打补丁 —— 两份列号表必然分叉，而分叉的后果是**静默错标签**。
GLOBAL_TARGET_COLUMNS = {
    'outcome_hard': (COL_WIN, COL_LOSS, COL_NORESULT),
    'score_mean': (COL_SCORE_MEAN,),
    'final_score': (COL_FINAL_SCORE,),
    'lead': (COL_LEAD,),
    # varTimeLeft = 官方 `sv3Mul` 六通道的**第 3 路**（下标 3）。
    # 语义已由 `trainingwrite.cpp:603-616` 定案（winloss 期望到达时间），
    # 实测交叉验证通过（全非负 / 与分差零相关 / 均值 10.95）。
    # 训练数据里存的**已经是最终物理量**，直接回归，不要再乘 40。
    'var_time_left': (COL_VAR_TIME_LEFT,),
    'global_weight': (COL_GLOBAL_WEIGHT,),
    'komi': (COL_KOMI,),
    'w_policy_player': (COL_W_POLICY_PLAYER,),
    'w_ownership': (COL_W_OWNERSHIP,),
    'w_policy_opp': (COL_W_POLICY_OPP,),
    'w_lead': (COL_W_LEAD,),
    'w_futurepos': (COL_W_FUTUREPOS,),
    'w_scoring': (COL_W_SCORING,),
    'w_value': (COL_W_VALUE,),
}


def resolve_network_key(network, cols=None):
    """网络名 → `GLOBAL_TARGET_LAYOUT` 的键；给了 ``cols`` 就**核对列数**。

     ``network`` **不给默认值**是刻意的：64 列这一族有**两个**网络
    （``b40c768nbt`` 有 Q 值 / ``b28c512`` 没有），光看列数分不出来。
    猜错会让权重列静默错位，所以这条在 CLI 边界就报错。
    """
    if not network:
        raise ConversionError(
            '--network 是**必填**项：64 列这一族有两个网络（b40c768nbt / b28c512），'
            '光看 globalTargetsNC 的列数分不出来；猜错会让权重列静默错位。')
    key = next((n for n in GLOBAL_TARGET_LAYOUT if n in network), None)
    if key is None:
        raise ConversionError(
            f'未登记的网络 {network!r}；已知 {sorted(GLOBAL_TARGET_LAYOUT)}。'
            f'新增网络前请先用 metrics_pytorch.py / dataio.cpp 确认其列布局。')
    layout = GLOBAL_TARGET_LAYOUT[key]
    if cols is not None and int(cols) != int(layout['cols']):
        raise ConversionError(
            f'{key} 登记为 {layout["cols"]} 列，实测 {int(cols)} 列。'
            f' 这说明 `GLOBAL_TARGET_LAYOUT` 与数据脱节了 —— '
            f'**别改本文件的列号去迁就**，去核对布局表。')
    return key


def extract_global_targets(global_targets, network):
    """``globalTargetsNC`` → **按语义命名**的全局目标列。

    这是**唯一**读列号的地方，也是「列数不同的归档得到同一组全局目标」这条
    契约的实现处：两个归档各自按自己的网络查 `GLOBAL_TARGET_LAYOUT`，
    输出的键名与取值口径完全一致。

    Args:
        global_targets: ``(N, cols)``；``cols`` 必须与该网络的登记列数相符。
        network: 网络名（必填，见 :func:`resolve_network_key`）。

    Returns:
        ``{语义名: 数组}``：``'outcome_hard'`` 是 ``(N,3)``
        （win / loss / noResult，**三列都给**，不替调用方做 argmax ——
        「三列全 0 = 和棋」那条判据在 `to_v7_labels` 里，不该在这里重写一遍），
        其余是 ``(N,)``。

    Raises:
        ConversionError: 未登记的网络 / 列数与登记不符 / 某语义列越界。
    """
    g = np.asarray(global_targets)
    if g.ndim != 2:
        raise ConversionError(f'globalTargetsNC 形状不符：{g.shape}，期望 (N, cols)')
    key = resolve_network_key(network, g.shape[1])
    need = max(max(v) for v in GLOBAL_TARGET_COLUMNS.values())
    if need >= g.shape[1]:
        raise ConversionError(
            f'{key} 只有 {g.shape[1]} 列，但语义 {sorted(GLOBAL_TARGET_COLUMNS)} '
            f'要用到第 {need} 列（col{KOMI_HINT}）⇒ 该网络的列布局与 '
            f'`GLOBAL_TARGET_LAYOUT` 登记表对不上。**报错，不猜** —— '
            f'静默填 0 会让 komi / 权重变成「看起来合理」的错标签。')
    out = {'_network': key, '_cols': int(g.shape[1])}
    for name, cols in GLOBAL_TARGET_COLUMNS.items():
        arr = np.ascontiguousarray(g[:, list(cols)].astype(np.float32, copy=False))
        out[name] = arr if len(cols) > 1 else arr[:, 0]
    return out


#: 上面那条报错里点名的列（`komi` 落在第 47 列，是所有语义里最靠后的）。
KOMI_HINT = COL_KOMI


# --------------------------------------------------------------------------- #
# 逐**成员**的布局分派
# --------------------------------------------------------------------------- #
def batch_token(member_name):
    """从 tar 成员路径里取出**批次名**（批次名就是网络名）。

    ``2026-08-25npzs/kata1-zhizi-b40c768nbt-s11472M-d5982M/3AF1….npz``
    → ``'kata1-zhizi-b40c768nbt'``。

     **为什么需要它**：``64`` 列这一族有**两个**网络，光看列数分不出来；
      而归档路径里的批次名是数据自带的、可靠的消歧信号。
    """
    for part in member_name.split('/'):
        for net in GLOBAL_TARGET_LAYOUT:
            if part.startswith(net):
                return net
    return None


def resolve_member_network(member_name, cols, default=None):
    """**逐成员**解析网络 → `GLOBAL_TARGET_LAYOUT` 的键。

     **为什么必须逐成员而不是逐归档**
    ----------------------------------
    实测两个归档里都**混着多种布局**（2026-08-25 的 57,386 个成员：

    ===================================  =======  ========  ==========
    批次（= 目录名）                        80 列    64 列      行数
    ===================================  =======  ========  ==========
    ``kata1-tf3-b11c768-s11001M``          27,393        0  1,608,110
    ``kata1-zhizi-b40c768nbt-s11472M``     21,315    8,678  1,756,156
    ===================================  =======  ========  ==========

    ⇒ **同一个网络名（b40c768nbt）在同一个归档里同时产出 64 列与 80 列的数据**，
    2026-07-30 那个归档也一样（15,825 / 672 个成员）。
    `GLOBAL_TARGET_LAYOUT['kata1-zhizi-b40c768nbt']['cols'] = 64` 因此**与数据
    脱节**（本仓登记表的缺口，见报告）⇒ 按归档名分派会在换成员时抛
    「登记为 64 列，实测 80 列」，**整批转换直接失败**。

    ⇒ 分派顺序：**列数唯一 ⇒ 直接定；列数有歧义（64）⇒ 用批次名消歧；
      批次名也定不下来 ⇒ 报错**（不猜）。

    Args:
        member_name: tar 成员路径。
        cols: 该成员 ``globalTargetsNC`` 的实测列数。
        default: ``--network`` 给的网络名（歧义时的第二优先依据）。

    Returns:
        ``(key, how)``，``how ∈ {'cols', 'cols+batch', 'cols+default'}``。

    Raises:
        ConversionError: 列数未登记 / 歧义且无法消歧。
    """
    by_cols = sorted(n for n, v in GLOBAL_TARGET_LAYOUT.items()
                     if v['cols'] == int(cols))
    if not by_cols:
        raise ConversionError(
            f'{member_name}：globalTargetsNC 是 {int(cols)} 列，'
            f'已知布局只有 {sorted({v["cols"] for v in GLOBAL_TARGET_LAYOUT.values()})}'
            f' 列 ⇒ 列布局未登记。**报错，不猜**：未登记的布局会让权重列'
            f'静默错位，比转换失败糟得多。')
    token = batch_token(member_name)
    if len(by_cols) == 1:
        return by_cols[0], 'cols'
    hit = [n for n in by_cols if token and n in token]
    if len(hit) == 1:
        return hit[0], 'cols+batch'
    if default:
        hit = [n for n in by_cols if n in default]
        if len(hit) == 1:
            return hit[0], 'cols+default'
    raise ConversionError(
        f'{member_name}：{int(cols)} 列有多个候选网络 {by_cols}，'
        f'而批次名 {token!r} 与 --network {default!r} 都定不下来。\n'
        f'  **报错，不猜** —— 猜错会让权重列静默错位。\n'
        f'  解法：把该成员所属批次按 --network 显式指定，或先核对 '
        f'metrics_pytorch.py / dataio.cpp 的列布局再登记。')

# --------------------------------------------------------------------------- #
# 列值的**物理可能域**检查
# --------------------------------------------------------------------------- #
#: **这不是「我猜的语义」，是「围棋里不可能出现的数」**
#: ------------------------------------------------------
#: 目的只有一个：**在列映射对不上的批次上当场抓出来，而不是让它混进训练**。
#: 越界只说明「这一批的这几列读出来不是那个量」，**不说明该改成什么** ——
#: 所以本脚本**不猜、不改、不填 0**，只把该批次标进 `meta_json` 并在 stderr 上
#: 大声报出来（这正是 ch18/ch19 那一类问题的处理方式：按原样写入 + 打标记）。
#:
#: 域的取法是「物理上不可能」的边界，不是「实测到的范围」—— 实测范围会随批次
#: 变，拿它当门禁就变成拿已知错位迁就新错位。


DOMAIN_HARD_BOUNDS = {
    # 围棋贴目不会超过 ±30（官方自对弈是 7.5）。 实测 `kata1-tf3-b11c768` 上
    # col47 ∈ {±7.5} 看着完美，而 zzb28c512 批上同一列到 **±51** —— 差 7 倍。
    'komi': (-30.0, 30.0),
    # 19×19 满盘 361 点 + 贴目，几百是硬上限。
    'final_score': (-400.0, 400.0),
    'score_mean': (-400.0, 400.0),
    'lead': (-400.0, 400.0),
}

#: 权重列：取值只能是 0 或 1（官方权重是开关，不是连续量）。
WEIGHT_COLUMNS = ('global_weight', 'w_policy_player', 'w_ownership',
                  'w_policy_opp', 'w_lead', 'w_futurepos', 'w_scoring',
                  'w_value')


def check_plausibility(targets, sample_cap=200_000):
    """逐语义列查「物理可能域」，返回 ``{列: {…, verdict}}``。

    三条断言，全部是**结构性质**而不是语义猜测：

    1. **硬边界**：分差 / 贴目 / lead 不可能超出 `DOMAIN_HARD_BOUNDS`；
    2. **权重是开关**：``WEIGHT_COLUMNS`` 必须落在 ``[0, 1]``；
    3. **outcome 三元组是概率行**：每行**和为 1**，或是全零（`to_v7_labels`
       假定的和棋标记）。

    ⇒ 任一条不过就标 ``SUSPECT``。样本大时按行抽样（``sample_cap``），
      免得为了算 min/max 把整列扫两遍 —— 转换期内存已经吃满了。

     第 3 条的实测结论与 `to_v7_labels` 的注释**不一致**：三个归档上
      ``globalTargetsNC[:, 0:3]`` 都**恰好和为 1**，而且**大量行是软的**
      （实测 zzb28c512 上 ``max < 0.999`` 的行占多数，例如
      ``[0.8236, 0.1764, 0]``）⇒ 它是**软的三分类分布**，不是「硬 one-hot +
      全零标记和棋」。`to_v7_labels` 用 ``argmax`` 把它压成硬标签，并把
      「全 0 ⇒ 无结果」当兜底 —— 那个兜底分支在这三个归档上**从不触发**。
      这里只**记录** ``frac_soft``（不改 `to_v7_labels`，见报告）。
    """
    out = {}

    def take(arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.shape[0] > sample_cap:
            step = int(np.ceil(a.shape[0] / sample_cap))
            a = a[::step]
        return a

    for name, (lo, hi) in DOMAIN_HARD_BOUNDS.items():
        if name not in targets:
            continue
        a = take(targets[name])
        bad = float(((a < lo) | (a > hi)).mean())
        out[name] = {'min': float(a.min()), 'max': float(a.max()),
                     'domain': [lo, hi], 'frac_violating': bad,
                     'verdict': 'SUSPECT' if bad > 0 else 'ok'}
    for name in WEIGHT_COLUMNS:
        if name not in targets:
            continue
        a = take(targets[name])
        bad = float(((a < 0.0) | (a > 1.0)).mean())
        out[name] = {'min': float(a.min()), 'max': float(a.max()),
                     'domain': [0.0, 1.0], 'frac_violating': bad,
                     'verdict': 'SUSPECT' if bad > 0 else 'ok'}
    hard = np.asarray(targets.get('outcome_hard', np.zeros((0, 3))),
                      dtype=np.float64)
    if hard.ndim == 2 and hard.size:
        h = take(hard)
        s = h.sum(1)
        draws = np.abs(s) < 1e-6                     # `to_v7_labels` 的和棋标记
        probs = np.abs(s - 1.0) < 1e-3               # 概率三元组（含 one-hot）
        out['outcome_hard'] = {
            'min': float(h.min()), 'max': float(h.max()),
            'domain': '行和为 1 的三元组，或全零（和棋标记）',
            'frac_violating': float((~(probs | draws)).mean()),
            # 软标签的占比：`to_v7_labels` 的 argmax 把这部分信息丢掉了。
            'frac_soft': float((h.max(1) < 0.999).mean()),
            'verdict': 'SUSPECT' if float((~(probs | draws)).mean()) > 0 else 'ok'}
    return out


def suspect_columns(plausibility):
    """``{列: 越界行数}`` —— 只要 ``verdict == 'SUSPECT'`` 的列。"""
    return {k: v for k, v in plausibility.items()
            if v.get('verdict') == 'SUSPECT'}


# --------------------------------------------------------------------------- #
# 输出 schema
# --------------------------------------------------------------------------- #
def output_spec(policy_topk=POLICY_TOPK, keep_qvalue=False):
    """``[(键, dtype, 每行尾形状), ...]`` —— npz 成员的**顺序**与 dtype 的唯一定义。

    键名沿用 `katago_npz.to_v7_labels` 的标签 dict（下游 loader 几乎可以照抄
    `smoke_train_v7.py::split_inputs`），只有两处**刻意不同**：

    * ``spatial_packed`` 而不是 ``spatial`` —— bit-packed，不解包（约束 3）；
    * policy 四列是 ``(N,K) rank + 权重`` 而不是稠密 ``(N,362)``（约束 4）。

     除这两处外**所有键的 dtype 与语义都与 `to_v7_labels` 逐字一致** ——
      少一处转换，就少一处「存的时候和读的时候口径不同」的可能。
    """
    cell = (BOARD, BOARD)
    k = int(policy_topk)
    spec = [
        # ---- 输入：bit-packed，不解包（约束 3）----
        ('spatial_packed', np.uint8, (SPATIAL_CHANNELS, PACKED_BYTES)),
        ('global', np.float32, (GLOBAL_CHANNELS,)),
        # ---- policy：top-K 稀疏（约束 4）+ 截断掉的尾部质量 ----
        ('policy_player_rank', np.int32, (k,)),
        ('policy_player_prob', np.float32, (k,)),
        ('policy_opp_rank', np.int32, (k,)),
        ('policy_opp_prob', np.float32, (k,)),
        ('policy_player_resid', np.float32, ()),
        ('policy_opp_resid', np.float32, ()),
        # ---- 标量目标 ----
        ('outcome', np.int64, ()),
        ('score', np.float32, ()),
        ('score_mean_hint', np.float32, ()),
        ('lead_hint', np.float32, ()),
        # varTimeLeft（官方 sv3Mul 六通道的下标 3）。语义见
        #   `katago_npz.COL_VAR_TIME_LEFT`（trainingwrite.cpp:603-616 定案）。
        # 这三处**硬编码列表**（SPEC / chunk / 前向补齐）必须同步改 ——
        #   落盘键不是从 labels dict 推导的，漏改任何一处都静默丢标签。
        ('var_time_left', np.float32, ()),
        ('komi', np.float32, ()),
        ('game_weight', np.float32, ()),
        # ---- 平面目标 ----
        ('score_distr', np.float32, (SCORE_DISTR_BINS,)),
        ('ownership', np.float32, (1,) + cell),
        ('seki', np.float32, (1,) + cell),
        ('futurepos', np.float32, (2,) + cell),
        ('scoring', np.float32, (1,) + cell),
        # ---- 逐行权重：`to_v7_labels` 的 `w` 子 dict 摊平成 `w_*` ----
        ('w_policy_opp', np.float32, ()),
        ('w_ownership', np.float32, ()),
        ('w_score', np.float32, ()),
        ('w_lead', np.float32, ()),
        ('w_futurepos', np.float32, ()),
        ('w_scoring', np.float32, ()),
        # ---- 溯源 + 跨局守卫 ----
        ('game_ids', np.int32, ()),
        ('source_id', np.int32, ()),
    ]
    if keep_qvalue:
        spec.append(('q_value', np.float32, (3, CELLS + 1)))
    return spec


def policy_resid(policy_all, top_vals):
    """逐行 ``1 − Σtopk / Σall``：**截断丢掉的尾部权重占比**。

    `topk_policy` 的 docstring 说「丢掉的尾部质量由**调用方**用
    `1 − Σtopk/Σall` 记账」，`policy_dense_from_sparse` 又说「由标签侧单独记账
    （`to_v7_labels` 的 `policy_resid`）」—— 而 `to_v7_labels` 其实**没有**产出
    这个字段。本函数把它补齐：它是「K=16 够不够大」的哨兵，不补就永远没人知道。

     ``Σall == 0`` 的行记 **0** 而不是 nan —— 全零行会让哨兵自己变成 nan，
      而 nan 会在报告里伪装成「K 太小」。
    """
    all_sum = np.asarray(policy_all, dtype=np.float64).sum(1)
    top_sum = np.asarray(top_vals, dtype=np.float64).sum(1)
    out = np.zeros(all_sum.shape, dtype=np.float64)
    ok = all_sum > 0
    out[ok] = 1.0 - top_sum[ok] / all_sum[ok]
    return out


def labels_to_chunk(labels, spatial_packed, policy_all, idx, source_id,
                    first_row, policy_topk=POLICY_TOPK, keep_qvalue=False):
    """一个归档成员 → 一块输出行（键与 :func:`output_spec` 一一对应）。

    ``spatial_packed`` / ``policy_all`` 单独传进来而不是从 ``labels`` 里取：
    ``labels['spatial']`` 是**解包后 float32**（每行 31.8 KB），落盘要用 packed，
    转换期也不该留着它。
    """
    n = int(idx.size)
    pi_idx, pi_val = labels['policy_player_sparse']
    po_idx, po_val = labels['policy_opp_sparse']
    pol = np.asarray(policy_all)[idx]
    out = {
        'spatial_packed': spatial_packed[idx],
        'global': labels['global'],
        'policy_player_rank': pi_idx.astype(np.int32, copy=False),
        'policy_player_prob': pi_val,
        'policy_opp_rank': po_idx.astype(np.int32, copy=False),
        'policy_opp_prob': po_val,
        'policy_player_resid': policy_resid(
            pol[:, 0, :], pi_val).astype(np.float32),
        'policy_opp_resid': policy_resid(
            pol[:, 1, :], po_val).astype(np.float32),
        'outcome': labels['outcome'],
        'score': labels['score'],
        'score_mean_hint': labels['score_mean_hint'],
        'lead_hint': labels['lead_hint'],
        'var_time_left': labels['var_time_left'],
        'komi': labels['komi'],
        'game_weight': labels['game_weight'],
        'score_distr': labels['score_distr'],
        'ownership': labels['ownership'],
        'seki': labels['seki'],
        'futurepos': labels['futurepos'],
        'scoring': labels['scoring'],
        'game_ids': np.arange(first_row, first_row + n, dtype=np.int32),
        'source_id': np.full(n, int(source_id), dtype=np.int32),
    }
    for name in ('policy_opp', 'ownership', 'score', 'lead', 'futurepos',
                 'scoring'):
        out['w_' + name] = labels['w'][name]
    if keep_qvalue:
        out['q_value'] = labels['q_value']
    want = {k for k, _, _ in output_spec(policy_topk, keep_qvalue)}
    if set(out) != want:
        raise ConversionError('chunk 与 output_spec 的键不一致：多 %s / 少 %s'
                              % (sorted(set(out) - want), sorted(want - set(out))))
    return out


def trim_chunk(chunk, take):
    """把一块行裁到前 ``take`` 行（``--limit`` 不整除成员大小时用）。"""
    first = next(iter(chunk))
    if take >= int(np.asarray(chunk[first]).shape[0]):
        return chunk
    return {k: v[:take] for k, v in chunk.items()}


# --------------------------------------------------------------------------- #
# 分片（把全量拆成 N 块分别转换，峰值内存 ÷ N）
# --------------------------------------------------------------------------- #
def shard_mask(n_rows, global_offset, num_shards, shard_id):
    """本成员内属于本 shard 的行的**局部**下标（升序 int64）。

    分片规则是**全局行号取模**：``(global_offset + j) % num_shards == shard_id``。

    为什么取模而不是切连续区间
    ------------------------
    ① **可证明**：N 块的并集恒等于全集、两两不相交，且每块大小至多差 1。
       切连续区间则要事先知道每个归档各有多少行（得多跑一遍计数）。
    ② 与成员边界无关 —— 成员大小不均（实测每个 npz 约 63 行）也不会让某块偏大。

     **``global_offset`` 必须按「分片过滤**之前**」的行数推进。**
    若误用过滤后的行数当偏移，每块算出的全局行号会随自己的过滤结果漂移
    ⇒ 从第 2 块起，同一行会同时落进两块（重复），另一些行谁都不落（丢失）。
    这是取模分片最经典的一个错，而且**全程不报任何错** ——
    实测症状是各块 `game_ids` 区间重叠，而 `game_ids` 正是这里唯一的溯源列。
    """
    n_rows = int(n_rows)
    if num_shards is None or int(num_shards) <= 1:
        return np.arange(n_rows, dtype=np.int64)
    num_shards = int(num_shards)
    j = np.arange(n_rows, dtype=np.int64)
    return j[((int(global_offset) + j) % num_shards) == int(shard_id)]


def subset_labels(labels, j):
    """按**局部行下标** ``j`` 取 :func:`katago_npz.to_v7_labels` 结果的子集。

     **不能写成 ``{k: v[j] for k, v in labels.items()}``** ——
    `to_v7_labels` 的返回值混着四类东西，只有两类带行轴：

    ==========  ==========================================  ==============
    类别        例子                                        处理
    ==========  ==========================================  ==============
    行轴 ndarray ``outcome`` / ``score_distr`` / ``spatial``   ``v[j]``
    嵌套 dict   ``w``（6 个权重列）                            逐键 ``v[j]``
    tuple       ``policy_player_sparse`` = ``(idx, val)``    每半都 ``x[j]``
    标量/字符串  ``_kept`` / ``_network`` / ``_dropped``       原样带走
    ==========  ==========================================  ==============

    统一判据用「第 0 轴长度 == ``_kept``」，所以以后 `to_v7_labels`
    新增带行轴的键时**不会漏切**（漏切会让块内行数与 ``game_ids`` 不符，
    而那正是写手 ``close()`` 里会当场报错的地方）。

     ``spatial`` 是**解包后 float32、每行 31.8 KB** —— 分片时切它纯属白切
    （``labels_to_chunk`` 用的是 packed），但不能因为「用不上」就跳过：
    跳过会让这个键的行数与 ``_kept`` 不一致，将来谁改成用解包版就会静默错位。
    """
    n = int(labels['_kept'])
    j = np.asarray(j, dtype=np.int64)
    out = {}
    for key, val in labels.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n:
            out[key] = val[j]
        elif isinstance(val, dict):
            out[key] = {
                k2: (v2[j] if isinstance(v2, np.ndarray) and v2.ndim >= 1
                     and v2.shape[0] == n else v2)
                for k2, v2 in val.items()}
        elif isinstance(val, tuple):
            out[key] = tuple(
                x[j] if isinstance(x, np.ndarray) and x.ndim >= 1
                and x.shape[0] == n else x for x in val)
        else:
            out[key] = val
    out['_kept'] = int(j.size)
    # `_dropped` 记的是「本批丢了多少非 19×19 行」，与分片无关 ⇒ 刻意不动。
    return out


# --------------------------------------------------------------------------- #
# 流式 npz 写手
# --------------------------------------------------------------------------- #
class NpzChunkedWriter:
    """攒块 → **顺序**写一个 `.npz`（每个键一个 ``<键>.npy`` 成员）。

     **为什么不能"边收边追加"**
    --------------------------
    zip 格式的每个成员都是一段**连续**的字节流，而 ``ZipFile`` 同一时刻只允许
    一个写句柄（多开直接 ``ValueError: another write handle open``）。
    ⇒ 30 个键的成员**无法交错**写。要交错就只能每个键自己走一遍归档（= 30 遍
    gzip 解压），那比重跑转换还贵。

    ⇒ 所以峰值内存 = **未压缩总量**（3.2M 行 ≈ 36.7 GB），这是 zip 容器的固有
    下限，不是本实现的偷懒。`uncompressed_bytes_estimate` 把它算出来，
    ``--ram-budget-gb`` 可以在跑之前拦住"装不下"。

    为什么不用 ``np.savez``
    -----------------------
    功能上等价，但 ``np.savez`` 有三件这里更好的事：
    ① **压缩**（``--compress-level``，实测 14×，2.6 GB vs 36.7 GB）；
    ② 逐键**校验** dtype / 形状 / 行数（本实现每个键都查，查错在写之前）；
    ③ 写完删掉该键的块缓冲（``self._chunks[key] = None``）—— 但这只在
       ``close()`` **逐键落盘**的那一刻才发生，此前**所有键的累积列都同时
       在内存里**。 所以别把这句话读成"峰值 = 最大单键"：实测峰值是
       **未压缩总量**（3.13M 行 ≈ 37.4 GB）再加 :data:`RAM_OVERHEAD_FACTOR`
       的余量 ≈ 44.8 GB，**不是**最大单键 `score_distr` 的 10.55 GB。
       要降峰值只能分片（``--num-shards``），见模块说明。

     ``force_zip64=True`` 是必需的：``futurepos`` 单成员就 9.2 GB，
      超过 ZIP 的 4 GB 单成员上限（未压缩）/ 2 GB（压缩）。
    """

    def __init__(self, path, spec, n_rows, compress=True, level=1):
        self.path = path
        self.spec = [(k, np.dtype(t), tuple(tail)) for k, t, tail in spec]
        self.n_rows = int(n_rows)
        self.queued = 0
        self.written = 0
        self._chunks = {k: [] for k, _, _ in self.spec}
        self._scalars = {}
        self._zf = None

    # ---- 缓冲 ----
    def write(self, chunk):
        """追加一块行。``chunk`` 的键必须与 ``spec`` 完全一致（逐键校验）。"""
        missing = [k for k, _, _ in self.spec if k not in chunk]
        if missing:
            raise ConversionError(f'chunk 缺键 {missing}')
        extra = [k for k in chunk if k not in self._chunks]
        if extra:
            raise ConversionError(f'chunk 有 spec 之外的键 {extra}')
        n = int(np.asarray(chunk[self.spec[0][0]]).shape[0])
        if n == 0:
            return
        if self.queued + n > self.n_rows:
            raise ConversionError(
                f'已缓冲 {self.queued} 行 + 这一块 {n} 行 > 头里声明的 '
                f'{self.n_rows} 行。 第 0 遍的行数与第 1 遍对不上 —— '
                f'归档在两遍之间变了？')
        for key, dt, tail in self.spec:
            a = np.asarray(chunk[key])
            if a.dtype != dt:
                raise ConversionError(f'{key} dtype {a.dtype} ≠ spec 的 {dt}')
            if a.shape[0] != n:
                raise ConversionError(f'{key} 行数 {a.shape[0]} ≠ {n}')
            if tail and a.shape[1:] != tail:
                raise ConversionError(f'{key} 形状 {a.shape} ≠ {(n,) + tail}')
            self._chunks[key].append(a if a.flags.c_contiguous
                                     else np.ascontiguousarray(a))
        self.queued += n

    def add_scalars(self, scalars):
        """登记**零行**成员（``meta_json`` 等）：形状由值自己定，与行数无关。"""
        self._scalars.update(scalars)

    # ---- 落盘 ----
    def close(self, compress=True, level=1):
        """顺序写完所有成员并关掉 zip。可重复调用（幂等）。"""
        if self._zf is not None:
            return
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        zf = zipfile.ZipFile(self.path, 'w', mode, allowZip64=True,
                             compresslevel=level if compress else None)
        self._zf = zf                      # 先置位：写失败时 close() 仍能关掉它
        try:
            for key, dt, tail in self.spec:
                arr = np.empty((self.n_rows,) + tail, dtype=dt)
                s = 0
                for piece in self._chunks[key]:
                    arr[s:s + piece.shape[0]] = piece
                    s += piece.shape[0]
                self._chunks[key] = None            # 写完立刻放掉这块
                if s != self.n_rows:
                    raise ConversionError(
                        f'{key} 缓冲了 {s} 行，头里声明 {self.n_rows} 行')
                self._write_member(key, arr)
                del arr
            for key, value in self._scalars.items():
                self._write_member(key, np.asarray(value))
            self.written = self.n_rows
        finally:
            zf.close()
            self._zf = None

    def _write_member(self, key, arr):
        with self._zf.open(key + '.npy', 'w', force_zip64=True) as fp:
            np.lib.format.write_array(fp, arr, allow_pickle=False)

    def discard(self):
        """扔掉缓冲并删掉半截文件（约束 5：不留行数对不上的 npz）。"""
        self._chunks = {}
        self._scalars = {}
        if self._zf is not None:
            self._zf.close()
            self._zf = None
        if os.path.exists(self.path):
            os.remove(self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        return False


def uncompressed_bytes_estimate(spec, n_rows):
    """``spec × 行数`` → 未压缩总字节（= 峰值内存的主项）。"""
    return int(sum(np.dtype(dt).itemsize * n_rows
                   * (int(np.prod(tail)) if tail else 1)
                   for _, dt, tail in spec))


def human_bytes(n):
    """字节数 → 人读得懂的字符串（GB / MB / B 三档，别印 "0.0 GB"）。"""
    n = float(n)
    for unit, div in (('TB', 1e12), ('GB', 1e9), ('MB', 1e6)):
        if n >= div:
            return '%.2f %s' % (n / div, unit)
    return '%d B' % int(n)


# --------------------------------------------------------------------------- #
# 读回来：`to_v7_labels` 的形状
# --------------------------------------------------------------------------- #
def npz_to_v7_labels(z, unpack=True):
    """把本脚本产出的 npz → **`katago_npz.to_v7_labels` 那个形状**的 dict。

    为什么在转换器里放读取端
    ----------------------
    「产出结构正确」这句话得有个可执行的定义，否则只能靠读键名猜。这里给出
    **唯一**一份映射，于是下游（`smoke_train_v7.py::split_inputs` 那类 loader）
    可以直接 `npz_to_v7_labels(np.load(path))` 拿到标签 dict，而不必各自
    重新拼一遍键名 —— 拼错键名不报错，只会静默少一项 loss。

    Args:
        z: 已 ``np.load`` 的 npz（或本函数自己 load 后关掉）。
        unpack: ``True`` ⇒ ``'spatial'`` 解包成 ``(N,22,19,19) float32``
            （模型直接吃）；``False`` ⇒ 保留 packed（省内存，喂 loss 时不需要）。

    Returns:
        标签 dict，键与 `to_v7_labels` 一致，另有 ``'global'`` / ``'game_ids'`` /
        ``'source_id'`` / ``'policy_*_resid'``。 **不含** ``'spatial'`` 之外的
        输入侧诊断列（``board_mask`` 在 19×19 过滤后恒为 True，不占空间）。
    """
    if int(z['schema_version']) != SCHEMA_VERSION:
        raise ConversionError(
            f'这个 npz 的 schema_version 是 {int(z["schema_version"])}，'
            f'本脚本写的是 {SCHEMA_VERSION} ⇒ 键名/形状可能对不上，'
            f'**拒绝读**（别靠"字段差不多就凑合读"）。')
    lb = {
        'policy_player_sparse': (z['policy_player_rank'].astype(np.int64),
                                 z['policy_player_prob']),
        'policy_opp_sparse': (z['policy_opp_rank'].astype(np.int64),
                              z['policy_opp_prob']),
        'policy_player_resid': z['policy_player_resid'],
        'policy_opp_resid': z['policy_opp_resid'],
        'outcome': z['outcome'],
        'score': z['score'],
        'score_mean_hint': z['score_mean_hint'],
        'lead_hint': z['lead_hint'],
        'var_time_left': z['var_time_left'],
        'komi': z['komi'],
        'score_distr': z['score_distr'],
        'ownership': z['ownership'],
        'seki': z['seki'],
        'futurepos': z['futurepos'],
        'scoring': z['scoring'],
        'game_weight': z['game_weight'],
        'global': z['global'],
        'game_ids': z['game_ids'],
        'source_id': z['source_id'],
        'w': {
            'policy_opp': z['w_policy_opp'],
            'ownership': z['w_ownership'],
            'score': z['w_score'],
            'lead': z['w_lead'],
            'futurepos': z['w_futurepos'],
            'scoring': z['w_scoring'],
        },
    }
    lb['spatial'] = (unpack_binary_input(z['spatial_packed']).astype(np.float32)
                     if unpack else z['spatial_packed'])
    if 'q_value' in z.files:
        lb['q_value'] = z['q_value']
    return lb


def load_v7_npz(path, unpack=True):
    """``np.load`` + :func:`npz_to_v7_labels`（父进程整份载入后 fork COW 的入口）。"""
    z = np.load(path, allow_pickle=False)
    try:
        return npz_to_v7_labels(z, unpack=unpack)
    finally:
        z.close()


def _render_suspect(archive, network, suspect):
    """列值越出物理可能域时的**大声**报告（写进 meta，也打到 stdout）。"""
    L = ['', ' %s（%s）有 %d 个全局目标列读出来**物理上不可能**：'
         % (archive, network, len(suspect)),
         ' 列映射与该批数据对不上 ⇒ 这些列的标签是错的。',
         ' 本脚本**按原样写入并打标记**（见 meta_json.archives[*]'
         '.suspect_columns）—— **不猜、不改、不填 0**：',
         '      凭空发明一个值比留一个已知错位的值更难查。',
         '    列                    实测范围            应在        越界行占比']
    for col, v in sorted(suspect.items()):
        L.append('    %-20s [%9.4f, %9.4f]  %-12s %6.2f%%'
                 % (col, v['min'], v['max'], v['domain'],
                    100.0 * v['frac_violating']))
    L.append('    ⇒ 这一批的这些列**不要用于训练**；其余列不受影响。')
    return '\n'.join(L)


# --------------------------------------------------------------------------- #
# 归档遍历与计数（第 0 遍）
# --------------------------------------------------------------------------- #
def iter_npz_members(archive):
    """流式产出 ``(成员名, NpzFile)``。

     **流式读，禁止 ``getmembers()``**：那要把整个 1.5 GB 归档走一遍才能开始
      干活（同一纪律见 `crosscheck_stdata.load_sample` 与
      `scripts/build_dataset.py`）。
     模式用 ``r|*`` 而不是 ``r|gz`` —— 三个归档里有一个是**不压缩**的
      ``.tar``，``r|gz`` 会在它上面直接抛异常。
    """
    if not os.path.isfile(archive):
        raise ConversionError(f'stdata 归档不存在：{archive}\n'
                              f'  用 --archive / --source 指定，或先下载该批次。')
    tf = tarfile.open(archive, 'r|*')
    try:
        for m in tf:
            if not m.isfile() or not m.name.endswith('.npz'):
                continue
            fp = tf.extractfile(m)
            if fp is None:
                continue
            yield m.name, np.load(io.BytesIO(fp.read()))
    finally:
        tf.close()


def count_kept_rows(archive, network=None, limit=None, log=None,
                    num_shards=1, shard_id=0, global_offset=0):
    """数出这个归档过滤后有多少行 19×19（= 第 1 遍要写的行数）。

     **只解 ``binaryInputNCHWPacked`` 与 ``globalTargetsNC`` 两个成员**
      （``np.load`` 对 zip 成员是惰性的，其余 5 个成员不解）⇒ 这一遍比全量
      转换便宜一个数量级。第二个成员只为读**列数**（逐成员分派布局，见
      :func:`resolve_member_network`），它很小（N × 64 × 4 B）。

     **布局解不出来的成员不计数** —— 两遍必须对同一批成员达成一致，
      否则第 1 遍写的行数会比预算少，`convert` 会当场报错。

    分片
    ----
    ``num_shards``/``shard_id``/``global_offset`` 让这一遍只数**属于本块**的行。
     **两遍必须用同一套全局偏移算术** —— `global_offset` 是「本归档之前
    全部归档累计的 19×19 行数」（**过滤前**），`member_base` 每个成员按它
    自己保留的行数推进。任何一处改成「过滤后」，第 0 遍与第 1 遍就会对同一批
    行给出不同的归属，而 `convert` 的 ``a_rows != budget`` 检查**抓不到**
    （两边会一起错）。

     ``--limit`` 的语义随之变成「**本块**最多留多少行」，不再是整个归档的。
      所以 ``--limit`` + 分块只是抽样工具，**不能**用来重建某个已知名单 ——
      全量构建请不要带 ``--limit``（不带时 N 块严格构成全集的划分）。
    """
    if network:
        resolve_network_key(network)             # 早失败：网络名错就别读 1.5 GB
    raw = kept = kept_shard = members = no_input = skipped_layout = 0
    member_base = int(global_offset)
    skipped_examples = []
    t0 = time.perf_counter()
    for mname, npz in iter_npz_members(archive):
        members += 1
        if 'binaryInputNCHWPacked' not in npz.files:
            no_input += 1
            npz.close()
            continue
        packed = npz['binaryInputNCHWPacked']
        raw += int(packed.shape[0])
        cols = int(np.asarray(npz['globalTargetsNC']).shape[1])
        try:
            resolve_member_network(mname, cols, network)
        except ConversionError as e:
            skipped_layout += 1
            if len(skipped_examples) < 3:
                skipped_examples.append(str(e).splitlines()[0])
            npz.close()
            continue
        n_keep = int((board_size_from_packed(packed) == BOARD).sum())
        kept += n_keep
        kept_shard += int(shard_mask(n_keep, member_base, num_shards,
                                     shard_id).size)
        member_base += n_keep # 按过滤前的行数推进
        npz.close()
        if limit is not None and kept_shard >= limit:
            break
    capped = kept_shard if limit is None else min(kept_shard, limit)
    info = {
        'archive': os.path.abspath(archive),
        'archive_bytes': int(os.path.getsize(archive)),
        'network': resolve_network_key(network) if network else None,
        'rows_raw': raw,
        'rows_kept': kept if limit is None else min(kept, limit),
        'rows_kept_seen': kept,
        'rows_kept_shard': capped,
        'members_scanned': members,
        'members_without_binary_input': no_input,
        'members_layout_unresolved': skipped_layout,
        'layout_unresolved_examples': skipped_examples,
        'count_seconds': round(time.perf_counter() - t0, 2),
    }
    if log:
        log('[count] %s → 原始 %d 行 / 保留 %d 行%s'
            '（扫 %d 个成员，布局解不出 %d 个，%.1fs）'
            % (os.path.basename(archive), raw, info['rows_kept'],
               (' / 本块 %d 行' % capped) if num_shards > 1 else '',
               members, skipped_layout, info['count_seconds']))
    return info


# --------------------------------------------------------------------------- #
# 主转换
# --------------------------------------------------------------------------- #
def convert(archives, out_path, *, policy_topk=POLICY_TOPK, limit=None,
            keep_qvalue=False, compress=True, level=1, ram_budget_gb=0.0,
            log_every=2_000, log=None, num_shards=1, shard_id=0):
    """把若干 stdata 归档写进**一个** `.npz`。

    Args:
        archives: ``[(路径, 网络名), ...]``。每个归档**各自的**网络名 ——
            列数不同的归档（80 / 64）就是这么在同一个文件里对上的。
        limit: 每个归档**本块**最多保留多少行 19×19（`None` = 全量）。
        num_shards / shard_id: 分片（见 :func:`shard_mask`）。``num_shards == 1``
            时行为与不分片**逐位相同**；``num_shards <= 0`` 是**非法值** ⇒ 报错，
            不再被静默夹成 1（见函数体里的 ）。峰值内存 ≈ 未压缩总量 /
            ``num_shards``。
        ram_budget_gb: 峰值内存上限（GB）。``0`` = 不检查。 估的是**未压缩
            总量 × :data:`RAM_OVERHEAD_FACTOR`**（见 :class:`NpzChunkedWriter`：
            zip 容器无法交错写成员 ⇒ 转换期必须把未压缩数据整个放内存）。
        log: ``str -> None`` 的回调（进度）。

    Returns:
        统计 dict（行数、逐归档明细、落盘体积、逐成员未压缩字节）。

    Raises:
        ConversionError: 任何一步对不上（列布局 / 两遍行数 / 写出行数）。
             **失败时把半截文件删掉** —— 磁盘紧张，一个行数对不上的 npz
            比没有更糟（下游不会报错，只会在训练中途崩）。
    """
    log = log or (lambda *a, **k: None)
    # 这里曾写 `num_shards = max(1, int(num_shards))` —— 那是**静默降级**，
    # 而 CLI 校验修好之后它对 CLI 已经不可达了；留着它等于给直接调用
    # `convert()` 的代码留一个「传 0/负数 ⇒ 悄悄拿到全集」的洞（`meta['shard']`
    # 还会把 num_shards 记成 1，看起来完全正常）。降级本身就该报错。
    num_shards = int(num_shards)
    if num_shards < 1:
        raise ConversionError(
            f'num_shards={num_shards} 非法：必须 >= 1'
            f'（1 = 不分片，与 shard_mask 的 num_shards<=1 语义一致）。'
            f'\n 别指望它被夹成 1：那会让分片调用**静默**退回全量转换，'
            f'而输出里的行数是个完全正常的数字。')
    shard_id = int(shard_id)
    if not 0 <= shard_id < num_shards:
        raise ConversionError(
            f'--shard-id {shard_id} 越界：应有 0 <= shard-id < '
            f'--num-shards({num_shards})')
    t_start = time.perf_counter()

    # ---- 第 0 遍：数行数（`.npy` 头里就是形状，没有增量布局）-----------------
    counts = []
    total_rows = 0
    archive_base = 0
    for archive, network in archives:
        info = count_kept_rows(archive, network, limit=limit, log=log,
                               num_shards=num_shards, shard_id=shard_id,
                               global_offset=archive_base)
        counts.append(info)
        total_rows += info['rows_kept_shard']
        # 下一归档的偏移按「过滤前」的行数推进（见 shard_mask 的警告）
        archive_base += info['rows_kept']
        if info['rows_kept'] == 0:
            raise ConversionError(
                f'{archive} 过滤后**一行 19×19 都没有**（原始 {info["rows_raw"]} 行、'
                f'扫了 {info["members_scanned"]} 个成员）⇒ 网络名或归档搞错了？')
    if total_rows == 0:
        raise ConversionError(
            f'第 {shard_id}/{num_shards} 块一行都没分到 —— '
            f'本块行数取模后的归属依赖于第 0/1 遍完全一致的偏移算术，'
            f'出现 0 行说明两者已经分叉（别猜是哪一遍错，先查 global_offset）。')
    spec = output_spec(policy_topk, keep_qvalue)
    need = int(uncompressed_bytes_estimate(spec, total_rows) * RAM_OVERHEAD_FACTOR)
    log('[plan] %d 个归档 → %s：第 %d/%d 块，%d 行，未压缩共 %s'
        '（= 转换期峰值内存的主项）'
        % (len(archives), out_path, shard_id, num_shards, total_rows,
           human_bytes(need)))
    if ram_budget_gb and need > ram_budget_gb * 1e9:
        raise ConversionError(
            f'未压缩 {need / 1e9:.1f} GB > --ram-budget-gb 给的 '
            f'{ram_budget_gb:.1f} GB。\n'
            f' 这不是"压缩后放不放得下"的问题：zip 容器无法交错写成员 ⇒ '
            f'转换期必须把未压缩数据整个放内存（见 `NpzChunkedWriter`）。\n'
            f'  · **加 --num-shards N 把峰值除以 N**（推荐，见 --num-shards）\n'
            f'  · 降 `--limit` 先出小样本。\n'
            f'  · `--keep-qvalue` 是最大的一个可省项（+4.3 KB/行）。')

    writer = NpzChunkedWriter(out_path, spec, total_rows)
    written = members = source_counter = 0
    global_seen = 0 # 按过滤前的行数推进，见 shard_mask
    per_archive = []
    try:
        for (archive, network), cinfo in zip(archives, counts):
            budget = int(cinfo['rows_kept_shard'])
            first_source_id = source_counter
            a_rows = a_members = 0
            plaus_all = {}
            per_network = {}
            layout_mismatch = skipped_layout = 0
            default_key = resolve_network_key(network) if network else None
            t0 = time.perf_counter()
            if budget == 0:
                continue
            for mname, npz in iter_npz_members(archive):
                if a_rows >= budget:
                    break
                if 'binaryInputNCHWPacked' not in npz.files:
                    npz.close()
                    continue
                cols = int(np.asarray(npz['globalTargetsNC']).shape[1])
                try:
                    key, how = resolve_member_network(mname, cols, network)
                except ConversionError as e:
                    skipped_layout += 1
                    log(' 跳过 %s：%s' % (mname, str(e).splitlines()[0]))
                    npz.close()
                    continue
                slot = per_network.setdefault(
                    key, {'rows': 0, 'members': 0, 'cols': int(cols),
                          'how': how, 'expected_cols':
                              GLOBAL_TARGET_LAYOUT[key]['cols']})
                policy_all = np.asarray(npz['policyTargetsNCMove'])
                d = read_katago_npz(npz, network=key)   # 校验布局 + 反推盘面尺寸
                idx = np.flatnonzero(d['_board_size'] == BOARD)
                if idx.size == 0:
                    npz.close()
                    continue
                # 列映射的**物理可能域**检查（见 check_plausibility）：
                # 这一批的 globalTargetsNC 若与登记表对不上，当场抓住并打标记，
                # 而不是让它混进训练。**不猜、不改、不填 0。**
                plaus = check_plausibility(extract_global_targets(
                    d['globalTargetsNC'][idx], key))
                for col, v in plaus.items():
                    agg = plaus_all.setdefault(
                        col, {'min': v['min'], 'max': v['max'],
                              'domain': v['domain'], 'frac_violating': 0.0,
                              'verdict': 'ok'})
                    agg['min'] = min(agg['min'], v['min'])
                    agg['max'] = max(agg['max'], v['max'])
                    agg['frac_violating'] = max(agg['frac_violating'],
                                                v['frac_violating'])
                    if v['verdict'] == 'SUSPECT':
                        agg['verdict'] = 'SUSPECT'
                labels = to_v7_labels(d, policy_topk=policy_topk)
                # 与 to_v7_labels 自己的过滤结果对齐（它只给 `_kept`，不给 idx）
                if int(idx.size) != labels['_kept']:
                    raise ConversionError(
                        f'{mname}：本脚本按 ch0 反推取到 {idx.size} 行 19×19，'
                        f'`to_v7_labels` 说 {labels["_kept"]} 行 ⇒ 两处过滤已分叉，'
                        f'**停下来**，别猜哪个对。')
                # ---- 分片：按**全局行号取模**挑出属于本块的行 ------------------
                # 刻意放在 `to_v7_labels` **之后**：早过滤（先切 d 再算标签）能省
                # 7/8 的标签计算，但要在 `d` 上做一次全键切片，而 `d` 里混着
                # 行轴数组与标量诊断，切错一个键就是静默错位。实测全量标签计算
                # 约 6.4k 行/s，8 块各跑一遍共约 80 分钟 —— 一次性构建可接受，
                # 换来的好处是**分片逻辑完全不影响标签计算路径**。
                n_local = int(idx.size)
                j = shard_mask(n_local, global_seen, num_shards, shard_id)
                global_seen += n_local # 先按过滤前的行数推进
                if j.size == 0:
                    labels = d = None
                    npz.close()
                    continue
                if j.size != n_local:
                    idx = idx[j]                 # 仍是**原始成员行号**
                    labels = subset_labels(labels, j)
                take = min(int(idx.size), budget - a_rows)
                chunk = labels_to_chunk(
                    labels, d['binaryInputNCHWPacked'], policy_all, idx,
                    source_counter, written, policy_topk=policy_topk,
                    keep_qvalue=keep_qvalue)
                if take < idx.size:
                    chunk = trim_chunk(chunk, take)
                writer.write(chunk)
                a_rows += take
                a_members += 1
                slot['members'] += 1
                if slot['cols'] != slot['expected_cols']:
                    layout_mismatch += 1
                slot['rows'] += take
                written += take
                labels = d = chunk = None
                npz.close()
                if log_every and a_members % log_every == 0:
                    el = time.perf_counter() - t0
                    log('  %s: %d/%d 行（%d 个成员，%.0f 行/s）'
                        % (os.path.basename(archive), a_rows, budget,
                           a_members, a_rows / max(el, 1e-9)))
            if a_rows != budget:
                raise ConversionError(
                    f'{archive}：第 0 遍数到 {budget} 行 19×19，第 1 遍只写出 '
                    f'{a_rows} 行 ⇒ 归档在两遍之间变了，或第 0 遍算错。'
                    f'**不写一个行数对不上的 npz**。')
            suspect = suspect_columns(plaus_all)
            per_archive.append({
                'archive': os.path.abspath(archive),
                'network': default_key or '(按成员分派)',
                'global_target_cols': sorted({v['cols']
                                              for v in per_network.values()}),
                'rows': int(a_rows),
                'rows_raw': int(cinfo['rows_raw']),
                'members_used': int(a_members),
                'first_source_id': int(first_source_id),
                'per_network': per_network,
                'members_layout_unresolved': int(skipped_layout),
                'members_with_unregistered_cols': int(layout_mismatch),
                'plausibility': plaus_all,
                'suspect_columns': sorted(suspect),
                'seconds': round(time.perf_counter() - t0, 2),
            })
            source_counter += a_members
            members += a_members
            log('[done] %s: %d 行 / %d 个成员（%.1fs）'
                % (os.path.basename(archive), a_rows, a_members,
                   per_archive[-1]['seconds']))
            for nk, v in sorted(per_network.items()):
                log('       布局 %-22s %d 列 %8d 行 %6d 成员（分派依据 %s%s）'
                    % (nk, v['cols'], v['rows'], v['members'], v['how'],
                       '， 与登记表 %d 列不一致' % v['expected_cols']
                       if v['cols'] != v['expected_cols'] else ''))
            if suspect:
                log(_render_suspect(os.path.basename(archive),
                                    default_key or '按成员分派', suspect))

        meta = build_meta(
            out_path=out_path, per_archive=per_archive, counts=counts,
            n_rows=written, members=members, policy_topk=policy_topk,
            keep_qvalue=keep_qvalue, compress=compress, level=level, spec=spec,
            elapsed=time.perf_counter() - t_start,
            num_shards=num_shards, shard_id=shard_id,
            total_rows_all_shards=archive_base)
        writer.add_scalars({
            'meta_json': np.array(json.dumps(meta, ensure_ascii=False)),
            'schema_version': np.array(SCHEMA_VERSION, dtype=np.int32),
            'target_model': np.array(TARGET_MODEL),
        })
        writer.close(compress=compress, level=level)
    except BaseException:
        writer.discard()                # 约束 5：不留半截的全量副本
        raise

    meta['file_bytes'] = int(os.path.getsize(out_path))
    meta['bytes_per_row'] = round(meta['file_bytes'] / max(written, 1), 2)
    meta['elapsed_seconds'] = round(time.perf_counter() - t_start, 2)
    log('[out] %s: %d 行，%s（%.1f B/行），%.1fs'
        % (out_path, written, human_bytes(meta['file_bytes']),
           meta['bytes_per_row'], meta['elapsed_seconds']))
    return meta


def build_meta(*, out_path, per_archive, counts, n_rows, members,
               policy_topk, keep_qvalue, compress, level, spec, elapsed,
               num_shards=1, shard_id=0, total_rows_all_shards=None):
    """`meta_json` 的内容。**通道资格表直接引用 `crosscheck_stdata`。**"""
    return {
        'tool': 'scripts/stdata_to_npz.py',
        'schema_version': SCHEMA_VERSION,
        'target_model': TARGET_MODEL,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        # ---- 分片：下游据此判断「我拿到的是全集还是一块」----
        # `rows_all_shards` 是**未分片时的总行数**（按过滤前的 19×19 行数累计）。
        # 下游可以用 n_rows / num_shards ≈ rows_all_shards / num_shards 做自检；
        # 若有人只跑了 3 块里的 1 块就开训，这一列会让缺口显形而不是静默少数据。
        'shard': {
            'num_shards': int(num_shards),
            'shard_id': int(shard_id),
            'rows_in_shard': int(n_rows),
            'rows_all_shards': (None if total_rows_all_shards is None
                                else int(total_rows_all_shards)),
            'partition_rule': 'global_row_index % num_shards == shard_id',
            'game_ids_note': ('game_ids 是**本块内**的局部行号（0 起）；'
                              '分块后各块的游戏编号不再全局唯一，'
                              '但 game_ids 逐行唯一这个跨局守卫语义不变'),
        },
        'out_path': os.path.abspath(out_path),
        'rows': int(n_rows),
        'rows_raw': int(sum(c['rows_raw'] for c in counts)),
        'members': int(members),
        'kept_board_size': BOARD,
        'policy_topk': int(policy_topk),
        'spatial_storage': 'bit-packed (N,22,46) uint8；'
                           '解包用 `katago_npz.unpack_binary_input`',
        'policy_storage': 'top-%d 稀疏（rank + 权重）；稠密化用 '
                          '`katago_v7_loss.policy_dense_from_sparse`' % policy_topk,
        'compression': ('zipfile ZIP_DEFLATED level %d' % level) if compress
        else 'ZIP_STORED（不压缩）',
        'keep_qvalue': bool(keep_qvalue),
        'archives': per_archive,
        'count_pass': counts,
        'columns': {k: [int(n_rows)] + list(tail) for k, _, tail in spec},
        'uncompressed_bytes': {
            k: int(np.dtype(t).itemsize * n_rows
                   * (int(np.prod(tail)) if tail else 1))
            for k, t, tail in spec},
        'spatial_channel_eligibility': {
            str(ch): {'eligibility': e, 'reason': r}
            for ch, (e, r) in SPATIAL_SPEC.items()},
        'global_channel_eligibility': {
            str(ch): {'eligibility': e, 'reason': r}
            for ch, (e, r) in GLOBAL_SPEC.items()},
        'aligned_channels': [ch for ch, v in SPATIAL_SPEC.items()
                             if v[0] == ALIGNED],
        'known_divergent': {
            'spatial': [ch for ch, v in SPATIAL_SPEC.items()
                        if v[0] == KNOWN_DIVERGENT],
            'why': '官方算 area 前先提死子，本仓 GoBoard.score() 不做'
                   '（spec §5.1 刻意保留）⇒ 对不齐是已知口径差，**不是 bug**。'
                   ' 绝对量是**批次特定**的，换归档不复现 ⇒ 本转换器按原样写入，'
                   '**不在转换期"修"**：凭空发明第三种偏差比留着已知偏差更糟。'
                   '要按通道查某一批的 provenance 用 `source_id`。',
        },
        'not_comparable': {
            'spatial': [ch for ch, v in SPATIAL_SPEC.items()
                        if v[0] == NOT_COMPARABLE],
            'global': [ch for ch, v in GLOBAL_SPEC.items()
                       if v[0] == NOT_COMPARABLE],
            'why': 'stdata 缺重建这些通道所需的输入（着法序列 / encore_phase），'
                   '比出来的任何对齐率都是巧合。 它们同样**按原样写入** —— '
                   '私自改成 0 等于把「不可比」变成「看起来像已对齐」。',
        },
        'game_ids_semantics':
            '**每行一个不同的值**（= 0..N−1）。stdata 是逐行独立样本，没有着法'
            '序列，行与行之间没有任何已知的先后关系；19×19 过滤还把归档打碎了。'
            '⇒ `gather_neighbors(..., game_ids=z["game_ids"])` 对**任何** ±k '
            '偏移都判不可用。这是**故意的**：跨局取到的盘面看起来完全合法'
            '（它就是某个真实盘面），只是不属于这一手 ⇒ 静默错标签。',
        'source_id_semantics':
            '行来自哪个归档成员（全局递增、跨归档唯一）。'
            '`archives[i].first_source_id` 是每个归档的起点，'
            '可反查 (归档, 成员序号) —— 用于定位 ch18/19 这类'
            '**绝对量批次特定**的通道。',
        'known_gaps': [
            'ch9–13 / ch15 / ch16：stdata 无着法序列与前一手/前二手盘，'
            '按原样带入；本仓从 SGF 重建时走的是「官方回退复制」分支，'
            '两边口径不同（crosscheck 的 NOT_COMPARABLE）。',
            'ch7 / ch20 / ch21：本仓源数据有 encore_phase 而 stdata 没有，'
            '不可比；按原样带入。',
            ' **`globalTargetsNC[:, 0:3]` 是软的三分类分布，不是硬 one-hot**：'
            '三个归档上都**恰好行和 = 1**，且大量行 `max < 0.999`'
            '（如 `[0.8236, 0.1764, 0]`）。`to_v7_labels` 用 `argmax` 压成硬 '
            '`outcome`，其「三列全 0 ⇒ 无结果」的兜底分支在这三个归档上'
            '**从不触发** ⇒ 置信度信息被丢掉。逐批的软标签占比见 '
            '`archives[*].plausibility.outcome_hard.frac_soft`。'
            '（按纪律**只报告不改 `src`**。）',
            ' **`COL_KOMI`(47) 读到的不是「本局贴目」，是 selfKomi'
            '（相对当前行棋方、带官方 draw-jitter）** —— 实测 `zzb28c512` 批上'
            'col47 ∈ [−41.5, 41.5]，而 `globalInputNC[:,5] × 20`（'
            '`feature_v7.KOMI_SCALE` 口径的 selfKomi）**逐行等于它**；'
            '`kata1-tf3-b11c768` 批因为全是 komi 7.5 才看着像贴目。'
            '⇒ npz 里的 `komi` 列对非 7.5 贴目的批次**不是贴目**，'
            '本脚本会把它标进 `suspect_columns`。'
            '（`katago_npz.py` 在 `COL_KOMI` 上注「两个网络版本上位置一致」—— '
            '列位置确实一致，**语义标注不成立**。下游要 selfKomi 请从 '
            '`global[:,5] × 20` 取。）',
        ],
        'suspect_columns': {a['archive']: a['suspect_columns']
                            for a in per_archive if a['suspect_columns']},
        'elapsed_seconds': round(elapsed, 2),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_source(text):
    """``'路径:网络名'`` 或光 ``'路径'`` → ``(路径, 网络名或 None)``。

     用 ``rpartition(':', 1)`` 而不是 ``split(':')`` —— Windows 的盘符
    （``F:\\...``）自带冒号，按第一个冒号切会把盘符切掉。
    """
    path, sep, network = text.rpartition(':')
    if not sep:
        return text, None
    if not path or not network:
        raise argparse.ArgumentTypeError(
            "--source 要写成 '归档路径' 或 '归档路径:网络名'，例如 "
            "'katago/stdata/2026-08-25npzs.tgz:kata1-tf3-b11c768'")
    return path, network


def build_argparser():
    ap = argparse.ArgumentParser(
        description='stdata 归档 → 单个 .npz（V7 22 通道训练用）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--source', action='append', type=parse_source, default=None,
                    metavar='ARCHIVE:NETWORK',
                    help='一个归档及其网络名，可重复；全部写进同一个 .npz。'
                         ' 网络名必填：64 列这一族有两个网络，列数分不出来')
    ap.add_argument('--archive', default=None,
                    help='单归档快捷方式（等价于一个 --source）')
    ap.add_argument('--network', default=None,
                    help='该归档的**主**网络名。 可不给：实测归档里混着多种布局，'
                         '默认**逐成员按列数分派**（列数有歧义时用归档路径里的'
                         '批次名消歧）。给了它只在消歧时当第二依据，'
                         '且它的列数与成员实测不符时会记进 '
                         'meta.archives[*].members_with_unregistered_cols')
    ap.add_argument('--out', default=None, help='输出 .npz 路径')
    ap.add_argument('--limit', type=int, default=None,
                    help='每个归档**本块**最多保留多少行 19×19（默认全量）。'
                         ' 是**过滤后**的行数')
    ap.add_argument('--num-shards', type=int, default=1, metavar='N',
                    help='把全量拆成 N 块分别转换，峰值内存 ≈ ÷N。'
                         ' 实测峰值是**未压缩总量 ×1.19**（不是最大单键）：'
                         '全量 3.13M 行 ≈ 44.8 GB，13.9 GB 的机器必须分块。'
                         'N=8 时 ≈5.6 GB。分块后**各块串行跑**（并行的峰值是 '
                         'N 倍之和）。'
                         ' 必须 N >= 1（1 = 不分片）；N <= 0 直接报错，'
                         '**不**静默当成不分片。N >= 2 时**必须**同时给 '
                         '--shard-id，除非用 --convert-all（它自己跑完全部 N 块）')
    ap.add_argument('--convert-all', action='store_true',
                    help='**一条命令跑完全部 N 块**（推荐；对齐 build_dataset 的 '
                         '--merge 断点续做范式）：逐块**串行**转换，已完成的块'
                         '自动跳过，中断后重跑同一条命令即可续做。产出 <out> 的 '
                         'N 个兄弟文件（foo_s0.npz … foo_s{N-1}.npz）加一份 '
                         '<out>.shards.json 清单（记录各块行数/文件/指纹）。'
                         ' 与 --shard-id 互斥（那是手动单块模式）；'
                         ' --count-only 时本参数让计数也逐块跑并打进清单')
    ap.add_argument('--shard-id', type=int, default=None, metavar='I',
                    help='本块编号，2 <= N 且 0 <= I < N。**手动单块模式**，'
                         '逐块跑：--num-shards 8 --shard-id 0/1/…/7'
                         '（规则是全局行号取模，N 块严格构成全集的划分）。'
                         ' **不给 = 不分片**（default 是 None 而不是 0：'
                         '「不分片」与「第 0 块」在 0 这个值上无法区分，'
                         '真值判断会让最常用的第 0 块跳过校验）。'
                         '给了就必须在分片（N >= 2），否则报错')
    ap.add_argument('--policy-topk', type=int, default=POLICY_TOPK,
                    help='policy 稀疏目标的 K')
    ap.add_argument('--ram-budget-gb', type=float, default=0.0,
                    help='峰值内存上限（GB），超了就在开跑前报错（0 = 不检查）。'
                         ' 估的是**未压缩总量 ×1.19**（RAM_OVERHEAD_FACTOR，'
                         '实测）：zip 容器无法交错写成员，转换期必须把未压缩'
                         '数据整个放内存（见 `NpzChunkedWriter` 的 docstring）')
    ap.add_argument('--keep-qvalue', action='store_true',
                    help='保留 qValueTargetsNCMove（float32 (3,362) ≈ 4.3 KB/行）。'
                         ' V7 的 12 项 loss **不消费**它，默认丢弃只为省内存/磁盘')
    ap.add_argument('--no-compress', dest='compress', action='store_false',
                    help='ZIP_STORED 不压缩（读得更快，体积约 14×）')
    ap.add_argument('--compress-level', type=int, default=1,
                    help='ZIP_DEFLATED 级别（实测 1 已达 ~14× 且 200+ MB/s）')
    ap.add_argument('--log-every', type=int, default=2000,
                    help='每扫多少个成员打一次进度（0 = 不打）')
    ap.add_argument('--count-only', action='store_true',
                    help='只跑第 0 遍（数行数），不写文件')
    ap.add_argument('--json', dest='json_path', default=None,
                    help='把统计写成 JSON')
    return ap


def resolve_archives(args):
    """``args`` → ``[(路径, 网络名或 None), ...]``；名字填错就在这里报错。"""
    srcs = list(args.source or [])
    if args.archive:
        srcs.append((args.archive, args.network))
    if not srcs:
        srcs = [(DEFAULT_ARCHIVE, DEFAULT_NETWORK)]
    for _, network in srcs:
        if network:
            resolve_network_key(network)      # CLI 边界就报未登记的网络名
    return srcs


def shard_paths(out_path, num_shards, shard_id):
    """``out`` + ``(N, I)`` → 这一块的路径：``foo_s{I}.npz``（``num_shards<=1`` 时就是 ``out``）。

     命名**不**加 ``_s`` 前缀的另一种方案是按块建子目录（``foo/s0.npz``）。
      选兄弟文件是因为 ``build_dataset.merge_shards`` 用的是 ``glob('*.npz')``
      扫同目录 —— 子目录会让它扫不到，兄弟文件能与既有习惯对齐。
    """
    if int(num_shards) <= 1:
        return out_path
    base, ext = os.path.splitext(out_path)
    return '%s_s%d%s' % (base, int(shard_id), ext or '.npz')


def convert_all_shards(archives, out_path, *, num_shards, log=print, **kw):
    """**逐块串行**把全量转成 ``num_shards`` 个 npz，支持断点续做。

    为什么需要它（而不是让用户自己写 for 循环）
    ------------------------------------------
    ① **断点续做**：8 块全量约 80 分钟，中途 Ctrl-C 很常见。已完成的块靠
       ``<块>.done`` 标记跳过，重跑同一条命令即可 —— 这正是
       ``build_dataset.py --merge`` 的范式，本仓已有这个习惯。
    ② **串行**：并行的峰值是 N 倍之和（本机 8 块并行 = 44.8 GB，直接 OOM）。
    ③ **清单**：写一份 ``<out>.shards.json``，记录每块的行数 / 文件 / 状态。
       只建了 1 块就开训是最容易发生的事故，清单让它显形。

     **跳过靠 ``.done`` 标记而不是「文件存在」**：npz 即使中途被杀也会留下
      一个**行数对不上**的合法 zip，而训练端不会报错、只会在中途崩
      （见 :class:`NpzChunkedWriter` 的约定 5）。标记由 ``convert`` 成功后写出。

    Args:
        num_shards: 块数。``<= 1`` 直接退化成不分片（只跑一次）。
        **kw: 透传给 :func:`convert`。

    Returns:
        清单 dict（也写到 ``<out>.shards.json``）。

    Raises:
        ConversionError: 任何一块失败。 已完成的块**保留**（含 ``.done``），
            所以修掉问题后重跑同一条命令会从失败处继续。
    """
    out_path = os.path.abspath(out_path)
    num_shards = int(num_shards)
    if num_shards < 1:
        raise ConversionError('--convert-all 需要 --num-shards >= 1')
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    manifest_path = out_path + '.shards.json'
    manifest = {
        'tool': 'scripts/stdata_to_npz.py',
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'out_base': out_path,
        'num_shards': num_shards,
        'partition_rule': 'global_row_index % num_shards == shard_id',
        'archives': [{'path': a, 'network': n} for a, n in archives],
        'shards': [],
    }
    if num_shards <= 1:
        log('[all] --num-shards 1 ⇒ 不分片，只跑一次')
    t0 = time.perf_counter()
    for sid in range(num_shards):
        p = shard_paths(out_path, num_shards, sid)
        done = p + '.done'
        entry = {'shard_id': sid, 'path': p, 'done_marker': done}
        if os.path.isfile(done) and os.path.isfile(p):
            rows = int(open(done).read().strip())
            entry.update(status='done', rows=rows)
            manifest['shards'].append(entry)
            log('[all] 块 %d/%d 已完成（%d 行），跳过 %s'
                % (sid + 1, num_shards, rows, os.path.basename(p)))
            continue
        log('[all] ── 块 %d/%d → %s'
            % (sid + 1, num_shards, os.path.basename(p)))
        t1 = time.perf_counter()
        meta = convert(archives, p, num_shards=num_shards, shard_id=sid, **kw)
        with open(done, 'w', encoding='utf-8') as fh:
            fh.write(str(int(meta['rows'])))
        entry.update(status='done', rows=int(meta['rows']),
                     file_bytes=int(meta.get('file_bytes', 0)),
                     seconds=round(time.perf_counter() - t1, 1))
        manifest['shards'].append(entry)
        _dump(manifest_path, manifest)      # 每块后落盘 ⇒ 中断也留得下进度
    done_n = [s for s in manifest['shards'] if s['status'] == 'done']
    total_rows = sum(s.get('rows', 0) for s in done_n)
    manifest['shards_done'] = len(done_n)
    manifest['rows_total'] = total_rows
    manifest['elapsed_seconds'] = round(time.perf_counter() - t0, 1)
    _dump(manifest_path, manifest)
    log('[all] 完成 %d/%d 块，共 %d 行，清单：%s（%.1fs）'
        % (len(done_n), num_shards, total_rows,
           os.path.basename(manifest_path), manifest['elapsed_seconds']))
    if len(done_n) < num_shards:
        log('[all] 还有 %d 块没跑完 —— **不要**用现有这几块开训，'
            '重跑同一条命令即可续做。' % (num_shards - len(done_n)))
    return manifest


def resolve_shard_args(num_shards, shard_id, convert_all=False):
    """CLI 的 ``--num-shards/--shard-id`` → 归一化后的 ``(num_shards, shard_id)``。

     **整个文件里唯一一处分片参数校验** —— `--count-only` 与 `convert` 两条
    路径都调它。两条路径各写一份校验就是「一个校验一个不校验」的温床：那正是
    本函数修掉的那个洞（`--count-only` 曾经只查 ``shard_id`` 越界、完全不看
    ``num_shards``，于是 ``--num-shards 0`` 静默变成不分片并把**全量行数**
    当成本块行数报出来，rc=0、数字看着正常，没有任何异常信号）。

    三态，不接受第四种
    ------------------
    ==========================  ==========================================
    ``--num-shards 1`` 单独给   不分片 ⇒ 归一化成 ``(1, 0)``（合法）
    ``--num-shards N`` 无 id 报错：N 块里到底是哪一块？
    ``--num-shards N --shard-id I``  第 I/N 块（``N >= 2``、``0 <= I < N``）
    ==========================  ==========================================

    为什么不把 ``N >= 2 --shard-id`` 也当成合法
    -------------------------------------------
    ``shard_mask`` 对 ``num_shards <= 1`` 直接返回全部行（那是**函数层的合法
    语义**，不归这里改），所以 ``--num-shards 1 --shard-id 0`` 里那个
    ``--shard-id`` **根本不生效**：它是个哑参数。收下它 = 再收一个静默无操作
    参数，而这次要消灭的正是「参数被静默降级」。N=1 的正确写法是一个都不传。

    为什么 ``--num-shards N`` 无 ``--shard-id`` 要报错
    --------------------------------------------------
    若把它默认为「第 0 块」，用户会拿到第 0 块却以为参数没生效；若按「不分片」
    处理，就会在**声称分 N 块的同时**写出全集 —— 分片流程的前提是「各块行数
    之和 == 全集行数」，而这个输出**看起来是正常的数字**。两种默认值都只在
    数字对不上时才显形，那时已经写完几 GB 了。

    Raises:
        ConversionError: 任何一种非法组合（⇒ CLI ``rc=2``，非 0）。
    """
    num_shards = int(num_shards)
    if num_shards < 1:
        # 不再 `max(1, ...)` 静默降级成不分片：分片的前提是各块行数之和
        # == 全集行数，而降级后报出来的是**全量行数**，数字完全正常。
        raise ConversionError(
            f'--num-shards {num_shards} 非法：必须 >= 1（1 = 不分片）。'
            f'\n 早先的实现把它静默夹成 1 ⇒ 变成分片流程里最坏的一种失败：'
            f'仍然 rc=0、仍然报一个行数，只是那个行数是**全量**。'
            f'分片流程的前提是「各块行数之和 == 全集行数」，'
            f'而这个输出**看起来是正常的数字**。')
    if shard_id is None:
        if num_shards != 1 and not convert_all:
            raise ConversionError(
                f'--num-shards {num_shards} 少了 --shard-id：分 {num_shards} 块'
                f'时必须指明本块是第几号（0 <= I < {num_shards}）。'
                f'\n  · 想要第 0 块 ⇒ --shard-id 0'
                f'\n  · 想要全量（不分片）⇒ 别给 --num-shards'
                f'\n  · 想一条命令跑完 N 块 ⇒ 加 --convert-all')
        # `convert_all=True` ⇒ 返回 (N, None)：调用方自己遍历 0..N-1，
        # 这里给 0 会被 `shard_paths` 当「第 0 块」而只建一块。
        return (num_shards, None) if convert_all else (1, 0)
    shard_id = int(shard_id)
    if num_shards < 2:
        raise ConversionError(
            f'--shard-id {shard_id} 只能在**分片**时给，而 --num-shards '
            f'{num_shards} 不是分片（N 必须 >= 2）。'
            f'\n N=1 时 `shard_mask` 原样返回全部行 ⇒ 这个 --shard-id '
            f'是个**不生效的哑参数**。不分片就别传它。')
    if not 0 <= shard_id < num_shards:
        raise ConversionError(
            f'--shard-id {shard_id} 越界：应有 '
            f'0 <= --shard-id < --num-shards({num_shards})')
    return num_shards, shard_id


def main(argv=None):
    args = build_argparser().parse_args(argv)

    def log(msg):
        print(msg, flush=True)

    try:
        archives = resolve_archives(args)
        # 分片校验在分叉之前、只做一次 ⇒ `--count-only` 与 `convert` 对同一
        #   组参数的合法性判定必然一致（归一化后的值直接喂给两条路径）。
        if args.convert_all and args.shard_id is not None:
            raise ConversionError(
                '--convert-all 与 --shard-id 互斥：前者自己跑完 N 块，'
                '后者是手动单块模式。\n'
                '  · 想一条命令跑完 ⇒ 只给 --convert-all --num-shards 8\n'
                '  · 想只跑第 3 块 ⇒ 只给 --shard-id 3')
        if args.convert_all:
            # `--convert-all` 自己会遍历 0..N-1 ⇒ **不能**先过那条
            #   「N>=2 必须配 --shard-id」的三态表。`convert_all=True` 让它
            #   只校验 `num_shards` 本身（`--num-shards 0` 照样报错）。
            num_shards, shard_id = resolve_shard_args(
                args.num_shards, None, convert_all=True)
        else:
            num_shards, shard_id = resolve_shard_args(
                args.num_shards, args.shard_id)
        if args.count_only:
            # `--convert-all --count-only` 必须**逐块**数，不能只数第 0 块 ——
            #   那会报出「第 0/8 块 = 12.5%」而用户以为看到了全量的分配。
            #   逐块也顺带验证了「各块行数之和 == 全集」这个分片前提。
            if args.convert_all:
                sharded_counts = []
                for sid in range(num_shards):
                    per, b = [], 0
                    for p, n in archives:
                        info = count_kept_rows(p, n, limit=args.limit, log=None,
                                               num_shards=num_shards,
                                               shard_id=sid, global_offset=b)
                        per.append(info)
                        b += info['rows_kept']
                    mine = sum(i['rows_kept_shard'] for i in per)
                    allrows = sum(i['rows_kept'] for i in per)
                    sharded_counts.append({'shard_id': sid,
                                           'rows': mine, 'rows_all': allrows})
                    log('  第 %d/%d 块 → %s 行（%.2f%%，理论 %.2f%%）'
                        % (sid, num_shards, mine,
                           100.0 * mine / max(allrows, 1),
                           100.0 / num_shards))
                tot = sum(s['rows'] for s in sharded_counts)
                want = sharded_counts[0]['rows_all'] if sharded_counts else 0
                log('[count] 各块合计 %d 行 / 全集 %d 行 %s'
                    % (tot, want, ' 一致' if tot == want else ' 不一致！'))
                if tot != want:
                    raise ConversionError(
                        f'各块行数之和 {tot} != 全集 {want} ⇒ 分片的偏移算术'
                        f'在第 0 遍与第 1 遍之间分叉了。**不要写文件**，'
                        f'先查 global_offset 是否两遍一致。')
                if args.json_path:
                    _dump(args.json_path, {'count_only': True,
                                           'num_shards': num_shards,
                                           'rows_total': tot,
                                           'rows_all': want,
                                           'shards': sharded_counts})
                return 0
            infos = []
            base = 0 # 按过滤前的行数推进，与 convert 一致
            for p, n in archives:
                info = count_kept_rows(p, n, limit=args.limit, log=log,
                                       num_shards=num_shards,
                                       shard_id=shard_id,
                                       global_offset=base)
                infos.append(info)
                base += info['rows_kept']
            raw = sum(i['rows_raw'] for i in infos)
            kept = sum(i['rows_kept'] for i in infos)
            log('\n合计：原始 %d 行 → 19×19 保留 %d 行（%.1f%%）'
                % (raw, kept, 100.0 * kept / max(raw, 1)))
            if num_shards > 1:
                mine = sum(i['rows_kept_shard'] for i in infos)
                log('  第 %d/%d 块 → %d 行（%.2f%% of 保留，理论值 %.2f%%）'
                    % (shard_id, num_shards, mine,
                       100.0 * mine / max(kept, 1),
                       100.0 / num_shards))
            if args.json_path:
                _dump(args.json_path, {'count_only': True, 'archives': infos,
                                       'rows_raw': raw, 'rows_kept': kept,
                                       'shard_id': shard_id,
                                       'num_shards': num_shards})
            return 0
        out = args.out or os.path.join('tmp', 'stdata_v7.npz')
        kwargs = dict(policy_topk=args.policy_topk, limit=args.limit,
                      ram_budget_gb=args.ram_budget_gb,
                      keep_qvalue=args.keep_qvalue, compress=args.compress,
                      level=args.compress_level, log_every=args.log_every,
                      log=log)
        if args.convert_all:
            convert_all_shards(archives, out, num_shards=num_shards,
                               **kwargs)
            return 0
        meta = convert(archives, out, num_shards=num_shards,
                       shard_id=shard_id, **kwargs)
    except ConversionError as e:
        sys.stderr.write('stdata_to_npz: 失败\n%s\n' % e)
        return 2
    except KeyboardInterrupt:
        sys.stderr.write('stdata_to_npz: 中断（已完成的块保留，重跑同一条命令续做）\n')
        return 130

    print(render_report(meta), flush=True)
    if args.json_path:
        _dump(args.json_path, meta)
        print('JSON 已写入 %s' % args.json_path)
    return 0


def _dump(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def render_report(meta):
    rows = max(int(meta['rows']), 1)
    L = ['=' * 78,
         '[stdata → npz] %s' % meta['out_path'],
         '  目标模型 %s · schema v%d · %s'
         % (meta['target_model'], meta['schema_version'], meta['compression']),
         '  行数 %d（原始 %d，保留 %.1f%%）/ %d 个归档成员'
         % (meta['rows'], meta['rows_raw'],
            100.0 * meta['rows'] / max(meta['rows_raw'], 1), meta['members'])]
    sh = meta.get('shard') or {}
    if int(sh.get('num_shards', 1)) > 1:
        L.append(' 分片：第 %d/%d 块（本块 %d 行，全量 %s 行）'
                 '⇒ 单个 npz **不是**全量，训练要同时读入全部 N 块'
                 % (sh['shard_id'], sh['num_shards'], sh['rows_in_shard'],
                    sh['rows_all_shards']))
    L += ['-' * 78, '  归档明细:']
    for a in meta['archives']:
        L.append('    %-46s %8d 行 %6d 成员'
                 % (os.path.basename(a['archive'])[:46], a['rows'],
                    a['members_used']))
        for nk, v in sorted(a.get('per_network', {}).items()):
            L.append('        %-24s %2d 列 %8d 行 %6d 成员  分派 %s%s'
                     % (nk, v['cols'], v['rows'], v['members'], v['how'],
                        ' 登记表是 %d 列' % v['expected_cols']
                        if v['cols'] != v['expected_cols'] else ''))
        if a.get('members_layout_unresolved'):
            L.append(' %d 个成员的布局解不出来，已跳过'
                     % a['members_layout_unresolved'])
    L += ['-' * 78,
          '  体积 %s（%d B/行，%.1fs）'
          % (human_bytes(meta['file_bytes']), meta['bytes_per_row'],
             meta['elapsed_seconds']),
          '  逐成员未压缩字节:']
    for k, b in sorted(meta['uncompressed_bytes'].items(), key=lambda kv: -kv[1]):
        if b < meta['rows']:
            continue                      # 4 B/行的小列不值得占屏
        L.append('    %-22s %7d B/行 %10s'
                 % (k, b // rows, human_bytes(b)))
    nc_s = meta['not_comparable']['spatial']
    nc_g = meta['not_comparable']['global']
    L += ['-' * 78,
          '  按原样带入（不修、不填 0、不猜）:',
          '    KNOWN_DIVERGENT 空间 ch%s —— 官方先提死子、本仓 score() 不做；'
          '绝对量批次特定 ⇒ 用 source_id 定位批次'
          % meta['known_divergent']['spatial'],
          '    NOT_COMPARABLE  空间 ch%s / 全局 ch%s —— 源数据缺输入'
          % (nc_s, nc_g),
          '    ALIGNED         空间 ch%s —— 逐位相同，直接用'
          % meta['aligned_channels'],
          '  game_ids 逐行唯一 ⇒ 邻行 gather 恒判不可用（故意的，'
          '见 meta_json.game_ids_semantics）']
    if meta.get('suspect_columns'):
        L.append('-' * 78)
        L.append(' 列值越出物理可能域（**已按原样写入 + 打标记，不要用于训练**）:')
        for archive, cols in meta['suspect_columns'].items():
            L.append('    %-46s %s'
                     % (os.path.basename(archive)[:46], sorted(cols)))
    L.append('=' * 78)
    return '\n'.join(L)


if __name__ == '__main__':
    sys.exit(main())
