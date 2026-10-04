# -*- coding: utf-8 -*-
"""B7 · 与官方 stdata 的**交叉校验器**（常驻可复跑工具，不是单测）。

为什么要有它
------------
「我们与官方 stdata 对齐到哪一步」这件事，本仓的证据曾经散在
``tests/test_v7_assemble.py`` / ``test_v7_ladders.py`` / ``test_v7_planes.py``
三处，其中一部分还只是 docstring 里的一段文字。这些数字：

  · **口径不统一** —— 有的报「逐行精确相等率」，有的报「逐格一致率」，
    两者混在一张表里；
  · **不可比性与真实偏差不分开** —— ch15/ch16 的 0.7276 是**巧合**
    （stdata 无「前一手盘」，本仓走官方回退复制分支），ch9..13 同理
    （stdata 无着法序列）。把它们当成「对齐率」汇报出去就是在撒谎；
  · **没法一条命令复跑** —— 想复核一次 200 npz 的数字只能去改测试文件。

这个脚本把上述三件事变成一次命令的输出：逐通道**两个口径** + **比对资格**
标注 + 「多出/少掉的点」的分布（按「子 vs 空点」拆开）。

用法::

    python scripts/crosscheck_stdata.py --npz-count 200 --rows-per-npz 40
    python scripts/crosscheck_stdata.py --only ch14,ch17 --verbose
    python scripts/crosscheck_stdata.py --json out.json
    python scripts/crosscheck_stdata.py --min-rate 1.0        # 门禁（只查 ALIGNED）

 **两个口径必须一起看，缺一不可**
--------------------------------
``row_exact``（整张 19×19 完全相同才算这一行相等）是**唯一可以拿去做门禁**的
口径；``cell_agree``（逐格）**会骗人**，而且骗得很厉害：

    ==============  ==============  =======================================
    通道            row_exact       cell_agree
    ==============  ==============  =======================================
    ch9             **0.000000**    **0.997200**   ← 逐格看着几乎完美
    ==============  ==============  =======================================

ch9 每一行只差 1 个点（361 个点里），逐格一致率因此高达 0.9972，而**没有一行**
是整张对的。汇报 ch9 = 99.7% 对齐是把一个彻底错的通道说成几乎完美。
⇒ 脚本两个都印，但门禁只看 ``row_exact``，且**只对 ``ALIGNED`` 通道**断言。

 **比对资格（``eligibility``）—— 不许跳过这一列**
-------------------------------------------------
``NOT_COMPARABLE`` 的通道**没有对齐率这回事**，报出来就是伪造证据：

    ==============  ==================  ==========================================
    通道            eligibility         为什么
    ==============  ==================  ==========================================
    ch9..ch13       NOT_COMPARABLE      stdata 是逐行独立样本、无着法序列
    ch15 / ch16     NOT_COMPARABLE      stdata 无「前一手/前二手盘」⇒ 本仓走
                                       官方回退复制分支；301 行上的 0.7276
                                       是**巧合**（对齐的其实是 ch14）
    ch7 / ch20/21   NOT_COMPARABLE      官方只在 encore 行置位，而
                                       ``spatial_channels_v7`` 没有
                                       encore_phase 入参 ⇒ 结构上到不了
ch18 / ch19     ALIGNED             Benson + 围空 + 双活过滤；本工具按官方
                                        自己的全局列**逐行**喂规则，只比
                                        TERRITORY 之外的行（TERRITORY 要
                                        ``secondEncoreStartColors``，
                                        stdata 给不出 ⇒ 不可比）
    其余            ALIGNED             可当 oracle
    ==============  ==================  ==========================================

 ``ALIGNED`` 的含义是**「本仓与官方逐位一致」**，实测就是 1.000000。门槛
  ``--min-rate 1.0`` 之所以能当门禁，靠的就是这个集合里没有一个通道到不了 1.0。
  ch18/19 要做到这一点，**不能**用一套默认 ``rules_flags`` 跑全样本 ——
  官方 ch18/19 是逐行按规则分岔的（``nninputs.cpp:2391-2439``）。所以本工具对
  这两个通道按官方 ``globalInputNC`` 逐行分派，并剔掉 TERRITORY 行。

 **归档缺失时本脚本「报错退出」，而 pytest 里的 ``skipif`` 是「跳过」。**
  这是刻意的差异：这个工具的存在意义就是「回答一个问题」，归档不在就回答不了，
  静默退出 0 等于假装成功。测试里用 ``skipif`` 是因为测试的职责是验代码、
  不是保证归档存在；两者的失败语义不能混。
"""

import argparse
import io
import json
import os
import sys
import tarfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.feature_v7 import (  # noqa: E402
    GLOBAL_CHANNELS,
    KOMI_SCALE,
    SCORING_AREA,
    SCORING_TERRITORY,
    SPATIAL_CHANNELS,
    TAX_ALL,
    TAX_NONE,
    TAX_SEKI,
    KO_POSITIONAL,
    KO_SIMPLE,
    KO_SITUATIONAL,
    GameRow,
    global_features_v7,
    komi_parity_wave,
    rules_flags_from,
    spatial_channels_v7,
)
from src.data.katago_npz import (  # noqa: E402
    BOARD_STRIDE,
    board_size_from_packed,
    unpack_binary_input,
)

# --------------------------------------------------------------------------- #
# 比对资格
# --------------------------------------------------------------------------- #
#: 官方通道与我们**逐位对齐**，可以当 oracle。
ALIGNED = 'ALIGNED'
#: 官方通道与我们**已知不一致**，且不一致的原因已定位（不是「还没查出来」）。
KNOWN_DIVERGENT = 'KNOWN_DIVERGENT'
#: **根本不可比**：官方那份数据里缺少重建这个通道所需的输入 ⇒ 比出来的任何
#: 数字都是无意义的（ch15/ch16 的 0.73 就是这么来的）。
NOT_COMPARABLE = 'NOT_COMPARABLE'

#: 官方 ch1 = pla / ch2 = opp，本仓 ch1 = 黑 / ch2 = 白（**绝对色**）。
#: ``to_play = +1`` 时 pla == 黑 ⇒ 同解。这是 stdata 实测的 ``to_play`` 取值
#: （`tests/test_v7_ladders.py`：+1 ⇒ 1.000000，−1 ⇒ 0.186813）。
OFFICIAL_TO_PLAY = 1

#: 全局 ch18 是浮点，用绝对容差比。
GLOBAL_ATOL = 2e-6

_HISTORY_NC = ('stdata 是逐行独立样本、**无着法序列** ⇒ ch9..13 靠历史列重建，'
               '无从重建（spec §9.4）')

#: ch7 / ch20 / ch21 是**结构上到不了**的那三个通道，理由与 ch9..13 同类：
#: 官方**只在 encore 行**置位它们，而 ``spatial_channels_v7`` **没有 encore_phase
#: 入参**（本仓压根不接这个维度）⇒ 那些行我们无论怎么填都是 0。
#: 实测（301 行、恰好 1 行 encorePhase=2）三个通道都只在那 1 行上不一致，
#: 其余 300 行逐位相同 —— 这个「300/301」**不是**对齐率，
#: 所以归 ``NOT_COMPARABLE`` 而不是 ``ALIGNED``：归 ALIGNED 会让
#: ``--min-rate 1.0`` 永远过不了（它到不了 1.0），而那等于把一个永不满足的
#: 门槛写进门禁；归 ALIGNED 还会让读表的人在「ALIGNED」那一列停下目光。
#: 与 `tests/test_v7_assemble.py` 的 ``_PINNED_EXACT`` 一致：它也**没有**钉
#: 这三个通道。
_ENCORE_NC = ('官方仅在 encorePhase≥1（ch20/21 为 ≥2）的行置位，而 '
              '`spatial_channels_v7` 没有 encore_phase 入参 ⇒ 这些行'
              '**结构上到不了**；非 encore 行上逐位相同，但这不是对齐率')

#: 空间 22 通道的资格表。**逐条列出，不靠区间推断** —— 写一遍
#: ``9 <= ch <= 13`` 的话，加通道时就会静默继承错误的资格。
SPATIAL_SPEC = {
    0: (ALIGNED, 'on-board 掩码（19×19 无填充 ⇒ 恒 1）'),
    1: (ALIGNED, '黑子（**绝对色**；官方 ch1=pla，to_play=+1 时同解）'),
    2: (ALIGNED, '白子（**绝对色**；官方 ch2=opp）'),
    3: (ALIGNED, '1 气桶（不分色；官方不做额外处理 ⇒ 可当 oracle）'),
    4: (ALIGNED, '2 气桶（不分色）'),
    5: (ALIGNED, '3 气桶（不分色）'),
    6: (ALIGNED, 'ko 禁点（encore=0 ⇒ 只有 ko_loc，官方同口径）'),
    7: (NOT_COMPARABLE, 'koRecapBlocked · ' + _ENCORE_NC),
    8: (ALIGNED, '官方注释写「6,7,8」但**从未写入** ⇒ 恒 0'),
    9: (NOT_COMPARABLE, _HISTORY_NC),
    10: (NOT_COMPARABLE, _HISTORY_NC),
    11: (NOT_COMPARABLE, _HISTORY_NC),
    12: (NOT_COMPARABLE, _HISTORY_NC),
    13: (NOT_COMPARABLE, _HISTORY_NC),
    14: (ALIGNED, '当前盘梯子（实测 1.000000 / 4368 行）'),
    15: (NOT_COMPARABLE,
         'stdata 无「前一手盘」⇒ 本仓走官方**回退复制**分支 ⇒ 比出来的 0.7276 '
         '是巧合（我们对齐的其实是 ch14）'),
    16: (NOT_COMPARABLE,
         'stdata 无「前二手盘」⇒ 同上，比出来的 0.7076 是巧合'),
    17: (ALIGNED, '梯子 working-move（依赖 to_play，实测官方恒 +1）'),
    18: (ALIGNED,
         'Benson 无条件存活（`board.cpp:2159-2195`）+ 围空（`:2214-2243`）+ '
         'tax≠NONE 时的双活整块过滤（`:2264-2296`），逐字照抄 '
         '`board.cpp:1853-2327`。官方按规则分岔（`nninputs.cpp:2391-2439`），'
         '本工具据官方自己的 `globalInputNC` **逐行**喂 tax/suicide，'
         '且**只比 TERRITORY 之外的行**（TERRITORY 要 '
         '`secondEncoreStartColors`，stdata 给不出 ⇒ 不可比，'
         '行数记在 `n_rows_not_comparable`）'),
    19: (ALIGNED, '同 ch18'),
    20: (NOT_COMPARABLE, 'second-encore 起始子 · ' + _ENCORE_NC),
    21: (NOT_COMPARABLE, '同上 · ' + _ENCORE_NC),
}

_PASS_NC = ('依赖着法序列（pass 标志）⇒ stdata 无着法序列，'
            '**不可比**')
_RULE_MAP = '规则位 → 通道的映射（喂进去的 rules_flags 由官方自己的列反解）'

#: 全局 19 通道的资格表。
GLOBAL_SPEC = {
    0: (NOT_COMPARABLE, _PASS_NC),
    1: (NOT_COMPARABLE, _PASS_NC),
    2: (NOT_COMPARABLE, _PASS_NC),
    3: (NOT_COMPARABLE, _PASS_NC),
    4: (NOT_COMPARABLE, _PASS_NC),
    5: (NOT_COMPARABLE,
        'currentSelfKomi 含官方 draw-jitter；本仓喂的是 `game_row.komi` '
        '⇒ 不是同一个量'),
    6: (ALIGNED, 'ko 规则三态 · ' + _RULE_MAP),
    7: (ALIGNED, 'ko 规则三态 · ' + _RULE_MAP),
    8: (ALIGNED, 'multiStoneSuicideLegal · ' + _RULE_MAP),
    9: (ALIGNED, 'territory 计分标志 · ' + _RULE_MAP),
    10: (ALIGNED, 'tax 三态 · ' + _RULE_MAP),
    11: (ALIGNED, 'tax 三态 · ' + _RULE_MAP),
    12: (ALIGNED, 'encorePhase>0（按官方自己的 ch12/ch13 分组喂 encore_phase）'),
    13: (ALIGNED, 'encorePhase>1（同上）'),
    14: (NOT_COMPARABLE, 'passWouldEndPhase ' + _PASS_NC),
    15: (NOT_COMPARABLE,
         'PDA · stdata 不给重建它的依据（本仓无 PDA ⇒ 恒 0）。 **小样本上官方'
         '这两格也恰好恒 0**（301 行），于是 row_exact 会显示 **1.000000** —— '
         '那是巧合不是对齐；样本一大就露馅（200 npz / 4368 行上官方有 133 个'
         '非零点 ⇒ 0.9696）'),
    16: (NOT_COMPARABLE, 'PDA 第二段 —— 同 ch15'),
    17: (ALIGNED, 'hasButton · ' + _RULE_MAP),
    18: (ALIGNED, '贴目 × 棋盘奇偶三角波（selfKomi 由官方 ch5×20 反推，'
                  '是精确代入）'),
}


class CrosscheckError(RuntimeError):
    """工具层面的失败（归档不在 / 一个 19×19 行都没取到）。⇒ **报错退出**。"""


# --------------------------------------------------------------------------- #
# 取样：流式，绝不 getmembers()
# --------------------------------------------------------------------------- #
def load_sample(archive, npz_count=12, rows_per_npz=40, board_size=19):
    """从 stdata 归档里取**前** ``npz_count`` 个 npz 的**前** ``rows_per_npz``
    个 ``board_size``×``board_size`` 行。

     **流式读**（``r|gz`` + ``next()``），**禁止 ``getmembers()``**：那要把
    1.5 GB 的归档整个走一遍才能开始干活（同一纪律见
    ``tests/test_katago_npz.py`` 与 ``scripts/probe_stdata_loss.py``）。

     **按行取，不要按成员取**：19×19 由 ch0（on-board 掩码）的 1 的个数反推。
      同一个 npz 里混着 9/11/13/18 路与非方阵，直接当 19×19 喂会**静默错位**
      （``board_size_from_packed`` 对非方阵记 0，这里靠它丢掉）。

     **前缀不是随机采样**：归档是按 npz 顺序排的，前几个文件与整批不同分布
      ⇒ 任何命中率都必须**连样本量一起报**（`report['sample']` 会带上取法）。

    Returns:
        ``(spatial, glob, ko, info)`` —— ``spatial`` ``(B,22,19,19) bool``、
        ``glob`` ``(B,19)`` 或 ``None``、``ko`` ``(B,)`` int64、
        ``info`` 取样过程的账（扫了几个成员、丢了多少行）。
    """
    if not os.path.isfile(archive):
        raise CrosscheckError(
            f'官方 stdata 归档不存在：{archive}\n'
            f' 本工具**不静默跳过**（与 pytest 的 skipif 刻意不同）：'
            f'归档不在就回答不了「我们对齐到哪一步」。\n'
            f'  用 --archive 指定，或先下载 stdata 批次。')
    if npz_count < 1 or rows_per_npz < 1:
        raise CrosscheckError(
            f'--npz-count / --rows-per-npz 必须 ≥ 1（收到 {npz_count} / '
            f'{rows_per_npz}）')

    spatial, glob, ko = [], [], []
    skipped_no_key = skipped_wrong_size = 0
    files = 0
    tf = tarfile.open(archive, 'r|gz')          # 流式，**不要 r: + getmembers**
    try:
        while files < npz_count:
            member = tf.next()
            if member is None:
                break
            if not member.name.endswith('.npz'):
                continue
            files += 1
            data = np.load(io.BytesIO(tf.extractfile(member).read()))
            if 'binaryInputNCHWPacked' not in data.files:
                skipped_no_key += 1
                data.close()
                continue
            packed = data['binaryInputNCHWPacked']
            keep = np.flatnonzero(
                board_size_from_packed(packed) == board_size)[:rows_per_npz]
            if keep.size == 0:
                skipped_wrong_size += 1
                data.close()
                continue
            sp = unpack_binary_input(packed)[keep]
            spatial.append(sp)
            if 'globalInputNC' in data.files:
                glob.append(np.asarray(data['globalInputNC'])[keep])
            # 官方 iterLadders 用盘面自带的 ko_loc（不给它会让「2 气的块的气恰好
            # 是劫点」的行分叉）—— 与 tests/test_v7_ladders.py 同一取法。
            k = np.full(keep.size, -1, np.int64)
            for i, mask in enumerate(sp[:, 6]):
                pts = np.argwhere(mask)
                if pts.shape[0]:
                    k[i] = pts[0][0] * board_size + pts[0][1]
            ko.append(k)
            data.close()
    finally:
        tf.close()

    if not spatial:
        raise CrosscheckError(
            f'{archive} 的前 {npz_count} 个 npz 里没有一行 '
            f'{board_size}×{board_size} 的数据（无 binaryInputNCHWPacked 的 '
            f'{skipped_no_key} 个、非 {board_size} 路的 {skipped_wrong_size} '
            f'个）⇒ 换个批次或调大 --npz-count')
    glob_arr = np.concatenate(glob) if glob else None
    info = {
        'archive': os.path.abspath(archive),
        'archive_bytes': os.path.getsize(archive),
        'stream_mode': 'r|gz (tarfile 流式；**未**使用 getmembers())',
        'npz_count_requested': npz_count,
        'npz_count_scanned': files,
        'rows_per_npz': rows_per_npz,
        'board_size': board_size,
        'rows_kept': int(sum(s.shape[0] for s in spatial)),
        'npz_without_binary_input': skipped_no_key,
        'npz_without_matching_board_size': skipped_wrong_size,
        'has_global_input': glob_arr is not None,
        'selection': (
            '**前缀**，不是随机采样：按归档里的 npz 顺序取前 %d 个 npz，'
            '每个取**前** %d 个 %d×%d 的行。⇒ 命中率**必须连样本量一起引用**，'
            '前几个文件与整批不同分布。'
            % (files, rows_per_npz, board_size, board_size)),
        'to_play_assumption': (
            'to_play = %+d（官方 stdata 实测恒 +1：ch17 在 +1 下 1.000000、'
            '在 -1 下 0.186813；本仓 ch1/ch2 是**绝对色**，官方是 pla/opp，'
            'to_play=+1 时同解）' % OFFICIAL_TO_PLAY),
        'history_assumption': (
            'my_hist / op_hist 全 -1 ⇒ 无着法、无前盘 ⇒ ch9..13 与 ch15/ch16 '
            '**不可比**，这两组不会也不该有对齐率'),
    }
    return (np.concatenate(spatial), glob_arr, np.concatenate(ko), info)


def boards_from_spatial(spatial):
    """官方 ch1/ch2 → 本仓约定的 ``(B,19,19)`` int8 盘面（+1 黑 / −1 白）。"""
    return (spatial[:, 1].astype(np.int8) - spatial[:, 2].astype(np.int8))


# --------------------------------------------------------------------------- #
# 空间通道：两个口径 + 差异分布
# --------------------------------------------------------------------------- #
def _diff_stats(mine, official, stone):
    """差异分布，**按「子 vs 空点」拆开**。

    为什么要拆：修复前 ch18/19 的 21,896 个差异点里 **19,913 个是子、1,983 个
    是空点**，分布本身就能否掉「官方先提死子」—— 提子只会动**子**，
    而真正的成因（Benson 判死 + 双活整块过滤）**两边都动**。
     这段 docstring 曾经把该拆分写成「『官方先提死子』这个猜测最硬的线索」。
      那是**错的**，已作废：官方 `calculateArea*` 既不提子也不做死子判定
      （`board.cpp:1949-2327` 的输入只有 `colors`）。留着会让下一个人重复这个
      已被推翻的诊断。
    """
    extra = mine & ~official
    missing = ~mine & official
    return {
        'extra_ours': int(extra.sum()),
        'extra_ours_stone': int((extra & stone).sum()),
        'extra_ours_empty': int((extra & ~stone).sum()),
        'missing_ours': int(missing.sum()),
        'missing_ours_stone': int((missing & stone).sum()),
        'missing_ours_empty': int((missing & ~stone).sum()),
        'rows_touched': int((extra | missing).any(axis=(1, 2)).sum()),
    }


#: ch18 / ch19：官方是**逐行按规则分岔**的（`nninputs.cpp:2391-2439`），而本工具
#: 默认用**一套** `rules_flags` 跑全样本 ⇒ 对这两个通道，全样本 `row_exact` 不是
#: 对齐率。官方分支与可重建性：
#:   AREA   + TAX_NONE            → `calculateArea`（无双活过滤）        可重建
#:   AREA   + TAX_SEKI/TAX_ALL    → `calculateIndependentLifeArea`        可重建
#:   TERRITORY（任意 tax）          → 仅 `encorePhase >= 2` 才发 area，且兜底要
#:                                   `secondEncoreStartColors`            **不可重建**
#: 所以本工具对 ch18/19 只在**可比子集**（`globalInputNC[:,9] == 0`）上按官方自己
#: 的 tax 列逐行分派后算 `row_exact`；不可比的行数记在 `n_rows_not_comparable`。
_AREA_CHANNELS = (18, 19)


def _area_rows_by_rule(glob):
    """官方 `globalInputNC` → ch18/19 的**逐行规则**与可比行掩码。

     `globalInputNC[:,8]` 是 `hist.rules.multiStoneSuicideLegal`，而官方传给
      `calculateArea*` 的是 `getSuicideLegalForPassAlive(hist)`
      （`nninputs.cpp:964`）= 那个标志 `|| alwaysComputePassAliveUnderSuicideRules`，
      后半个来自 `hist.modes`、**19 个全局通道都没编码**。实测把 `ch8 == 0` 的行
      按 `False` 喂仍逐位相等（301 行 / 4368 行上都是 1.000000），所以这里取
      `ch8` 本身；这个假设一旦被推翻，表现是 ch18/19 的 `row_exact` 掉下来，
      而不是静默变好。
    """
    if glob is None:
        return None, None
    tax = np.where(glob[:, 10] < 0.5, TAX_NONE,
                   np.where(glob[:, 11] > 0.5, TAX_ALL, TAX_SEKI))
    suicide = glob[:, 8] > 0.5
    comparable = glob[:, 9] < 0.5          # 非 TERRITORY
    return (tax, suicide), comparable


def _area_ownership_by_rule(boards, glob):
    """按官方分支逐行算 ch18/19（不可重建的 TERRITORY 行留 0）。"""
    from src.data.feature_v7 import area_ownership_map
    rule, comparable = _area_rows_by_rule(glob)
    if rule is None or not comparable.any():
        return (np.zeros(boards.shape, bool), np.zeros(boards.shape, bool),
                comparable)
    tax, suicide = rule
    m18 = np.zeros(boards.shape, bool)
    m19 = np.zeros(boards.shape, bool)
    for t in (TAX_NONE, TAX_SEKI, TAX_ALL):
        for s in (False, True):
            sel = comparable & (tax == t) & (suicide == s)
            if not sel.any():
                continue
            own = area_ownership_map(boards[sel],
                                     is_multi_stone_suicide_legal=s,
                                     tax_rule=t)
            m18[sel] = own > 0
            m19[sel] = own < 0
    return m18, m19, comparable


def compare_spatial(spatial, ko, glob=None):
    """逐通道算 ``row_exact`` / ``cell_agree`` / 差异分布 / 资格。"""
    boards = boards_from_spatial(spatial)
    n = boards.shape[-1]
    B = boards.shape[0]
    mine = spatial_channels_v7(
        boards, np.full(B, OFFICIAL_TO_PLAY, np.int8), ko,
        np.full((B, 3), -1, np.int16), np.full((B, 3), -1, np.int16),
    ).astype(bool)
    assert mine.shape == (B, SPATIAL_CHANNELS, n, n)
    stone = boards != 0

    # ch18/19 换成「按官方规则逐行分派 + 只看可比行」的口径（见 _AREA_CHANNELS）
    area18 = area19 = None
    area_cmp = None
    if glob is not None:
        area18, area19, area_cmp = _area_ownership_by_rule(boards, glob)
        mine[:, 18] = area18
        mine[:, 19] = area19

    out = {}
    for ch in range(SPATIAL_CHANNELS):
        same = mine[:, ch] == spatial[:, ch]
        elig, reason = SPATIAL_SPEC.get(ch, (NOT_COMPARABLE, ' 未登记的通道！'))
        if area_cmp is not None and area_cmp.any():
            same = same[area_cmp]
            ch_cmp = mine[area_cmp, ch]
            off_cmp = spatial[area_cmp, ch]
            nz_ours, nz_off = int(ch_cmp.sum()), int(off_cmp.sum())
            diff = _diff_stats(ch_cmp, off_cmp, stone[area_cmp])
        else:
            nz_ours, nz_off = int(mine[:, ch].sum()), int(spatial[:, ch].sum())
            diff = _diff_stats(mine[:, ch], spatial[:, ch], stone)
        row = {
            'channel': ch,
            'scope': 'spatial',
            'eligibility': elig,
            'reason': reason,
            'row_exact': float(same.all(axis=(1, 2)).mean()),
            'cell_agree': float(same.mean()),
            'ours_nz': nz_ours,
            'official_nz': nz_off,
            'diff': diff,
            'n_rows_compared': int(same.shape[0]),
            'n_rows_not_comparable': int(B - same.shape[0]),
        }
        # 逐格会骗人：整张对不上、但逐格还很高 ⇒ 必须显式点出来。
        row['cell_agree_misleads'] = bool(
            row['cell_agree'] >= 0.99 and row['row_exact'] <= 0.5)
        out[('spatial', ch)] = row
    return out


# --------------------------------------------------------------------------- #
# 全局通道：只有逐行口径（不是 19×19 平面，没有「逐格」这回事）
# --------------------------------------------------------------------------- #
def _group_rows(g):
    """把官方行按 ``(规则组合, encorePhase)`` 分组。

    ⇒ 喂进去的 ``rules_flags`` / ``encore_phase`` 都是**从官方自己的列反解**的，
    所以测的是「规则位 → 通道」这段映射，不是「规则位怎么来的」（后者是 SGF
    `RU` 的事）。这样每个规则的**每个取值**都被官方数据验证过，而不只是
    「默认值恰好对」。

     必须连 ``encorePhase`` 一起分组：全局 ch12/ch13 由它门控，只按规则组合
      分组的话这两格会对着官方 encore 行报 0。
    """
    ko_rule = np.where(g[:, 6] > 0.5,
                       np.where(g[:, 7] > 0, KO_POSITIONAL, KO_SITUATIONAL),
                       KO_SIMPLE)
    tax = np.where(g[:, 10] < 0.5, TAX_NONE,
                   np.where(g[:, 11] > 0.5, TAX_ALL, TAX_SEKI))
    suicide = g[:, 8] > 0.5
    territory = g[:, 9] > 0.5
    button = g[:, 17] > 0.5
    enc = np.where(g[:, 13] > 0.5, 2, np.where(g[:, 12] > 0.5, 1, 0))
    rule_key = (((ko_rule.astype(np.int64) * 100 + tax) * 4 + suicide) * 2
                + territory) * 2 + button
    key = rule_key * 3 + enc
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(int(k), []).append(i)
    return (ko_rule, tax, suicide, territory, button, enc, groups)


def _ours_global(g, groups):
    """按 ``(规则组合, encorePhase)`` 分组喂 ``global_features_v7``。"""
    B = g.shape[0]
    ko_rule, tax, suicide, territory, button, enc, _ = _group_rows(g)
    my = np.full((B, 3), -1, np.int16)
    op = np.full((B, 3), -1, np.int16)
    out = np.full((B, GLOBAL_CHANNELS), np.nan, np.float32)
    seen = np.zeros(B, bool)
    for k, idxs in sorted(groups.items()):
        idxs = np.asarray(idxs)
        fl = rules_flags_from(
            scoring=SCORING_TERRITORY if territory[idxs[0]] else SCORING_AREA,
            tax=tax[idxs[0]], ko=ko_rule[idxs[0]],
            multi_stone_suicide=suicide[idxs[0]], has_button=button[idxs[0]])
        out[idxs] = global_features_v7(
            GameRow(komi=0.0), my[idxs], op[idxs], history_length=0,
            rules_flags=fl, to_play=np.ones(len(idxs), np.int8),
            encore_phase=int(enc[idxs[0]])).astype(np.float32)
        seen[idxs] = True
    assert seen.all(), f'全局只覆盖了 {int(seen.sum())}/{B} 行'
    return out


def _global_ch18(g):
    """全局 ch18 三角波：用官方 ch5 反推 ``selfKomi``（**精确代入**，非近似）。

     不能走 ``global_features_v7``：那个 API 的 ``GameRow(komi=...)`` 是**整批
      一个标量**，而官方 ch5 带 draw-jitter、逐行不同 ⇒ 只有逐行调
      ``komi_parity_wave`` 才拿得到精确代入。官方 ch5 上 komi=7.5 的行恰为
      0.375 = 7.5/20，而 ch18 用的正是同一个 selfKomi。
    """
    sk = (g[:, 5].astype(np.float64) * KOMI_SCALE).astype(np.float32)
    pred = np.empty(g.shape[0], dtype=np.float32)
    for i in range(g.shape[0]):
        terr = g[i, 9] > 0.5
        e = 2 if g[i, 13] > 0.5 else (1 if g[i, 12] > 0.5 else 0)
        pred[i] = komi_parity_wave(
            sk[i:i + 1], BOARD_STRIDE * BOARD_STRIDE,
            SCORING_TERRITORY if terr else SCORING_AREA, e)[0]
    hit = np.isclose(pred.astype(np.float64), g[:, 18].astype(np.float64),
                     atol=GLOBAL_ATOL)
    return pred, hit


def compare_global(g):
    """全局 ``(B,19)`` 逐通道对拍。

     全局通道**不是 19×19 平面** ⇒ 没有「逐格」口径，``cell_agree`` 记
      ``null`` 并注明原因。硬凑一个「逐元素一致率」只会让两个量长得像一回事。
    """
    B = g.shape[0]
    _, _, _, _, _, _, groups = _group_rows(g)
    ours = _ours_global(g, groups)
    pred18, hit18 = _global_ch18(g)

    out = {}
    for ch in range(GLOBAL_CHANNELS):
        elig, reason = GLOBAL_SPEC.get(ch, (NOT_COMPARABLE, ' 未登记的通道！'))
        if ch == 18:
            # ch18 由 _global_ch18 逐行精确代入算出，见那里的说明。
            exact = hit18
            mine = pred18
        else:
            mine = ours[:, ch]
            exact = np.isclose(mine.astype(np.float64),
                               g[:, ch].astype(np.float64), atol=GLOBAL_ATOL)
        row = {
            'channel': ch,
            'scope': 'global',
            'eligibility': elig,
            'reason': reason,
            'row_exact': float(exact.mean()),
            'cell_agree': None,          # 全局不是平面 ⇒ 这个口径不存在
            'cell_agree_note': (
                '全局通道是 (B,19) 标量序列、不是 19×19 平面 ⇒ 没有「逐格」'
                '口径，硬凑会与空间口径混淆'),
            'ours_nz': int((mine != 0).sum()),
            'official_nz': int((g[:, ch] != 0).sum()),
            'diff': {'rows_mismatched': int((~exact).sum())},
        }
        if ch == 18:
            row['detail'] = {
                'rows_matched': int(hit18.sum()),
                'rows_total': int(hit18.size),
                'self_komi_source':
                    '官方 globalInputNC[:,5] × 20（= 本仓 KOMI_SCALE，'
                    '含 draw-jitter）⇒ 精确代入',
                'atol': GLOBAL_ATOL,
            }
        elif ch in (6, 7, 8, 9, 10, 11, 12, 13, 17):
            row['detail'] = {
                'groups': len(groups),
                'note': ('喂进去的 rules_flags / encore_phase 是从官方自己的 '
                         '列反解的 ⇒ 测「规则位 → 通道」映射'),
            }
        out[('global', ch)] = row
    assert B == g.shape[0]
    return out


# --------------------------------------------------------------------------- #
# 通道选择
# --------------------------------------------------------------------------- #
def parse_only(spec):
    """``'ch14,ch17'`` / ``'gch18'`` / ``'14,17'``（裸数字 = 空间）→
    ``{('spatial',14), ...}``。也接受 ``ch6-11`` 这样的闭区间。"""
    if not spec:
        return None
    out = set()
    for tok in spec.split(','):
        tok = tok.strip().lower()
        if not tok:
            continue
        scope = 'global' if tok.startswith('gch') else 'spatial'
        if scope == 'global':
            tok = tok[3:]
        elif tok.startswith('ch'):
            tok = tok[2:]
        if '-' in tok:
            lo, _, hi = tok.partition('-')
            rng = range(int(lo), int(hi) + 1)
        else:
            rng = [int(tok)]
        for c in rng:
            out.add((scope, c))
    return out or None


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
#: 逐格口径骗人的实例，随报告一起印出来（brief 点名的那个例子）。
MISLEAD_CASE = ('ch9', 0.000000, 0.997200,
                '逐格看着几乎完美，但没有一行是整张对的')


def build_report(spatial, glob, ko, sample_info):
    rows = compare_spatial(spatial, ko, glob)
    if glob is not None:
        rows.update(compare_global(glob))
    tally = {ALIGNED: 0, KNOWN_DIVERGENT: 0, NOT_COMPARABLE: 0}
    for r in rows.values():
        tally[r['eligibility']] += 1
    return {
        'tool': 'scripts/crosscheck_stdata.py',
        'sample': sample_info,
        'spatial': [rows[('spatial', c)] for c in range(SPATIAL_CHANNELS)],
        'global': ([rows[('global', c)] for c in range(GLOBAL_CHANNELS)]
                   if glob is not None else None),
        'tally': tally,
        'warnings': [
            ' **逐格一致率（cell_agree）会骗人**：整张 19×19 全对才算一行相等'
            '（row_exact）才是能用来做门禁的口径。实例：%s row_exact=%.6f / '
            'cell_agree=%.6f —— %s。' % MISLEAD_CASE,
            ' **eligibility 不是装饰**：`NOT_COMPARABLE` 的通道没有对齐率这'
            '回事 —— stdata 缺重建它们所需的输入，比出来的数字是巧合。'
            'ch15/ch16 的 ~0.73 就是这么来的（我们走官方的回退复制分支，'
            '对齐的其实是 ch14）。',
            ' 任何命中率都要**连样本量一起引用**：取样是归档的**前缀**，'
            '不是随机采样（见 sample.selection）。',
            ' 归档缺失时本工具**报错退出**（与 pytest 的 skipif 刻意不同：'
            '工具回答不了问题就不该假装成功）。',
        ],
    }


def check_gate(report, min_rate):
    """``--min-rate`` 门禁：**只**对 ``ALIGNED`` 通道断言。

     ``KNOWN_DIVERGENT`` 绝不能被误杀 —— ch18/19 对不齐是**已知口径差**
      （spec §5.1 刻意不补死子判定），拿它当回归会逼着人去「修」一个不是 bug
      的东西，或者反过来把阈值调到 0.44 来「迁就」它。
     ``NOT_COMPARABLE`` 绝不能被当对齐率 —— 见上。
    """
    if min_rate is None:
        return []
    bad = []
    for row in report['spatial'] + (report['global'] or []):
        if row['eligibility'] != ALIGNED:
            continue
        if row['row_exact'] < min_rate:
            bad.append(
                'ch%s(%s) row_exact=%.6f < 门禁 %.6f —— '
                ' **查 `spatial_channels_v7` / `global_features_v7` 的组装顺序，'
                '不要改底层实现**'
                % (row['channel'], row['scope'], row['row_exact'], min_rate))
    return bad


def _fmt(v, w=9):
    return ('%*s' % (w, 'n/a' if v is None else '%.6f' % v))


def render(report, verbose=False):
    s = report['sample']
    L = []
    L.append('=' * 100)
    L.append('[stdata 交叉校验] %s' % s['archive'])
    L.append('  取样: %s' % s['selection'])
    L.append('  扫描: %d/%d 个 npz → %d 行 %d×%d; 无 binaryInput 的 %d 个; '
             '非 %d 路的 %d 个'
             % (s['npz_count_scanned'], s['npz_count_requested'],
                s['rows_kept'], s['board_size'], s['board_size'],
                s['npz_without_binary_input'], s['board_size'],
                s['npz_without_matching_board_size']))
    L.append('  读取: %s; %.1f MB'
             % (s['stream_mode'], s['archive_bytes'] / 1048576))
    L.append('  假设: %s' % s['to_play_assumption'])
    L.append('  假设: %s' % s['history_assumption'])
    L.append('=' * 100)
    for w in report['warnings']:
        L.append(w)
    L.append('')

    head = ('  ch  %-15s %s  %s  %8s %8s  %s'
            % ('', 'row_exact', 'cell_agree', 'ours_nz', 'off_nz', '备注'))
    L.append('---- 空间 22 通道（%d 行）----' % s['rows_kept'])
    L.append(head)
    for row in report['spatial']:
        note = '%s — %s' % (row['eligibility'], row['reason'])
        if row['cell_agree_misleads']:
            note += (' 逐格 %.4f 看着完美而逐行只有 %.4f ⇒ 逐格在骗人'
                     % (row['cell_agree'], row['row_exact']))
        L.append('  %2d  %s  %s  %8d %8d  %s'
                 % (row['channel'], _fmt(row['row_exact']),
                    _fmt(row['cell_agree']), row['ours_nz'],
                    row['official_nz'], note))
    if report['global'] is not None and report['global']:
        L.append('')
        L.append('---- 全局 19 通道（标量序列，无「逐格」口径）----')
        L.append('  ch  %-15s %s  %8s %8s  %s'
                 % ('', 'row_exact', 'ours_nz', 'off_nz', '备注'))
        for row in report['global']:
            L.append('  %2d  %s  %s  %8d %8d  %s — %s'
                     % (row['channel'], _fmt(row['row_exact']),
                        '     n/a ', row['ours_nz'], row['official_nz'],
                        row['eligibility'], row['reason']))

    L.append('')
    shown = report['spatial'] + (report['global'] or [])
    L.append('  资格汇总（本表 %d 个通道）: %s' % (
        len(shown), ', '.join(
            '%s=%d' % (k, sum(1 for r in shown if r['eligibility'] == k))
            for k in (ALIGNED, KNOWN_DIVERGENT, NOT_COMPARABLE))))

    L.append('')
    if any(r['scope'] == 'spatial' for r in report['spatial']):
        L.append('---- 「我们多出/少掉的点」按「子 vs 空点」的分布 ----')
        L.append(' 为什么要拆：只有分开才知道偏差落在「死子**本身**」还是'
                 '「死子周围那片**空**」')
        L.append('     ⇒ 绝大部分落在子上 ⇒ 官方算 area 前先提死子')
        L.append('     （参考量，非本次运行：zzb28c512 批 1254 行上 ch18 的 '
                 '21,896 个')
        L.append('      差异点里 19,913 是子 / 1,983 是空点，见 '
                 '`area_ownership_map`）')
        L.append('  ch  资格            多出(子/空)     少掉(子/空)   受影响行')
        for row in report['spatial']:
            d = row['diff']
            if not (d['extra_ours'] or d['missing_ours']):
                continue
            L.append('  %2d  %-15s %7d/%-7d %7d/%-7d %6d/%d'
                     % (row['channel'], row['eligibility'],
                        d['extra_ours_stone'], d['extra_ours_empty'],
                        d['missing_ours_stone'], d['missing_ours_empty'],
                        d['rows_touched'], s['rows_kept']))

    if verbose:
        L.append('')
        L.append('---- 逐通道明细（--verbose）----')
        for row in report['spatial'] + (report['global'] or []):
            L.append('  ch%d(%s) %s' % (row['channel'], row['scope'],
                                        row['eligibility']))
            L.append('     row_exact=%.6f  cell_agree=%s'
                     % (row['row_exact'],
                        'n/a' if row['cell_agree'] is None
                        else '%.6f' % row['cell_agree']))
            L.append('     %s' % row['reason'])
            if 'detail' in row:
                L.append('     detail=%s' % json.dumps(row['detail'],
                                                       ensure_ascii=False))
    L.append('')
    return '\n'.join(L)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser():
    ap = argparse.ArgumentParser(
        description='与官方 KataGo stdata 的交叉校验（两个口径 + 比对资格）')
    ap.add_argument('--archive', default=os.path.join(
        'katago', 'stdata', '2026-08-25npzs.tgz'), help='stdata 归档路径')
    ap.add_argument('--npz-count', type=int, default=12,
                    help='取归档里前 N 个 npz（默认 12，实测 301 行）')
    ap.add_argument('--rows-per-npz', type=int, default=40,
                    help='每个 npz 取前 M 行 19×19（默认 40）')
    ap.add_argument('--only', default=None,
                    help='只看这些通道，如 ch14,ch17 / gch18 / ch6-11'
                         '（裸数字 = 空间通道）')
    ap.add_argument('--min-rate', type=float, default=None,
                    help='门禁：ALIGNED 通道的 row_exact 下限（默认关）。'
                         ' 只对 ALIGNED 生效，KNOWN_DIVERGENT / '
                         'NOT_COMPARABLE 不会被误杀')
    ap.add_argument('--json', dest='json_path', default=None,
                    help='把完整报告写到该 JSON 文件')
    ap.add_argument('--verbose', action='store_true', help='打印逐通道明细')
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    try:
        spatial, glob, ko, info = load_sample(
            args.archive, args.npz_count, args.rows_per_npz)
    except CrosscheckError as e:
        sys.stderr.write('crosscheck: 失败\n%s\n' % e)
        return 2

    report = build_report(spatial, glob, ko, info)
    only = parse_only(args.only)
    if only is not None:
        report['spatial'] = [r for r in report['spatial']
                             if ('spatial', r['channel']) in only]
        if report['global'] is not None:
            report['global'] = [r for r in report['global']
                                if ('global', r['channel']) in only]
        report['warnings'] = [w for w in report['warnings']
                              if not w.startswith(' **逐格一致率')]

    bad = check_gate(report, args.min_rate)
    report['gate'] = {'min_rate': args.min_rate, 'violations': bad,
                      'scope': '只对 ALIGNED 通道断言'}
    print(render(report, verbose=args.verbose))
    if args.json_path:
        with open(args.json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('JSON 已写入 %s' % args.json_path)
    if bad:
        sys.stderr.write('crosscheck: 门禁未通过（min_rate=%.6f）\n'
                         % args.min_rate)
        for b in bad:
            sys.stderr.write(' %s\n' % b)
        return 1
    if args.min_rate is not None:
        print('门禁通过：所有 ALIGNED 通道的 row_exact ≥ %.6f'
              % args.min_rate)
    return 0


if __name__ == '__main__':
    sys.exit(main())
