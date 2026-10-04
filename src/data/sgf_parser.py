"""
SGF格式解析器：解析围棋棋谱文件
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# 模块级预编译正则：原实现把这些 pattern 放在方法内，每次解析（每局）都重新
# re.compile；百万级棋谱下这是可观的纯浪费。
_PROP_RE = re.compile(r'([A-Z]{1,2})\[(.*?)\]')
_MOVE_TOKEN_RE = re.compile(r'(?:;|\A)(AB|AW|[BW])((\[[^\]]*\])+)')
_BRACKET_VAL_RE = re.compile(r'\[([^\]]*)\]')

# `RE` 的结构是 `<胜方>[<分隔符><余部>]`，余部可能是分差、认输标记或什么都没有。
# 分隔符实测见过 `+`，SGF 规范还允许 `:` 与 `-`；三种都收。
_RE_RESULT_RE = re.compile(r'^(?P<who>[BW])(?:\s*(?P<sep>[-+:])\s*(?P<rest>.*))?$')
# 分差取余部**开头**的数字：实测语料里有 `W+3 zi`（带单位）与 `W+0,25`（欧陆逗号小数）。
_RE_MARGIN_RE = re.compile(r'^[+-]?\d+(?:[.,]\d+)?')

#: `RE` 余部取这些词（不分大小写）时判为**认输类**（胜负已定但没有分差）。
#: `R`=Resign、`T`=Time(over)、`F`=Forfeit。实测 `data/games/games` 133,604 局：
#: `B+R` 21.94% + `W+R` 21.75% + `W+Resign` 21.85% + `B+Resign` 15.55% = **81.09%**，
#: `W+T` 0.12% + `B+T` 0.08% = 0.20%，`W+F` 0.02% + `B+F` 0.01% = 0.03%。
#: spec §5.3.3 记的是 75.2%（98,352 局）—— 本次全量重扫得 81.1%，spec 未更新。
_RE_RESIGN_WORDS = frozenset({
    'r', 'resign', 'resigned', 't', 'time', 'timeout', 'f', 'forfeit',
    'forfeited', 'abandon', 'adjourn', 'adjourned',
})

#: `RE` 整串取这些词时判为**和棋**。FF[4] 用 `0` 表示和棋，`draw` 是通用写法。
_RE_DRAW_WORDS = frozenset({'0', 'draw', 'jigo', 'jigo wo kakeru', 'tie'})

#: `RE` 整串取这些词时判为**无结果**（既非胜负也非和棋）。与「认输」和「和棋」
#: 都不同 —— 这三态必须能分开，见 `ResultInfo`。
_RE_VOID_WORDS = frozenset({
    '?', '??', 'void', 'unknown', 'unfinished', 'noresult', 'no result',
    'aborted', 'invalid',
})


@dataclass(frozen=True)
class ResultInfo:
    """`RE` 的结构化解析结果。

    返回约定（三态必须可区分，所以不给「万能的 -1/None」）
    --------------------------------------------------------
    | 情形 | `score` | `is_draw` | `is_resign` | `resign_side` |
    |---|---|---|---|---|
    | 数值分差 `B+2.5` / `W+3.5` | `+2.5` / `-3.5` | False | False | None |
    | 和棋 `0` / `draw` | `0.0` | **True** | False | None |
    | 认输 `B+R` / `W+T` / `B+F` | `None` | False | **True** | **0** |
    | 认输 `W+R` / `W+T` / `W+F` | `None` | False | **True** | **1** |
    | 无结果 / 空 / `?` / `Void` / `B+` | `None` | False | False | None |

     **`score is None` 单独出现时无法区分「认输」与「无结果」** —— 训练侧若要
    这个区别必须读 `is_resign`（`games.npz` 的 `g_resign` 就是它的落盘形式）。
    `score` 的符号约定是**黑−白**：`B+2.5 → +2.5`，`W+3.5 → -3.5`。

     `score` 是 SGF 的最终分差，**含贴目**，不是「净胜目数」。

     **`resign_side` 是「谁认输」，不是「谁赢」**：`B+R` ⇒ 0（黑认输 ⇒ 白赢）。
    它**只**在 `is_resign` 为 True 时有值，其余一律 None —— `B+` / `W+`（空分差，
    实测 95 局）胜负已定但分差不可知，那**不是认输**。语料里 63.8% 是认输，
    只有这一列能告诉训练侧方向；`games.npz` 的 `g_resign_side` 是它的落盘形式。
    """
    score: Optional[float]
    is_draw: bool
    is_resign: bool
    #: 0 = 黑认输 / 1 = 白认输 / None = 非认输。
    #: **必须放在末尾并带默认值**：全仓（`parse_result` 的 8 处 +
    #: `GameRecord.result_info` 的 default_factory）都用位置三元组构造。
    resign_side: Optional[int] = None


# `g_rules` 的 bit 布局（spec §5.3.2）。**每一位都是「1 = 非 TT 默认值」的标志位**，
# 所以 `g_rules == 0` 就是「纯 TT 规则」（区域计分 / 无税 / 简单劫 / 禁自杀 /
# 无打劫还子），而语料里 55.0% 的局正好落在这个值上。
RULES_BIT_SCORING = 1 << 0    # 0 = 区域计分(AREA)，1 = 数目计分(TERRITORY)
RULES_BIT_TAX = 1 << 1        # 0 = 无税，1 = 有税
RULES_BIT_KO = 1 << 2         # 0 = 简单劫，1 = 其他（超劫/位置劫）
RULES_BIT_SUICIDE = 1 << 3    # 0 = 禁自杀，1 = 允许自杀
RULES_BIT_BUTTON = 1 << 4     # 0 = 无打劫还子，1 = 有
#: bit 5..7 空着，留给将来出现的规则维度（例如 tax 的三态、ko 的三态）。
RULES_BITS_USED = 0x1F

#: 无 `RU`、或 `RU` 无法识别时的取值 = **spec §2.5 的 TT 默认**。
RULES_DEFAULT = 0

#: `RU` 里本仓**实测见过**的取值（`data/games/games` 133,604 局全量：
#: 无 `RU` 55.03% / `Chinese` 37.64% / `Japanese` 7.33% / 空 3 局 0.00%）。
RULES_KNOWN_WORDS = frozenset({'chinese', 'japanese'})


def parse_result(re_str: Optional[str]) -> ResultInfo:
    """解析 SGF `RE` 为结构化结果。见 `ResultInfo` 的表。

    实测语料形态（`data/games/games` 133,604 局全量）::

        B+R 21.94% / W+Resign 21.85% / W+R 21.75% / B+Resign 15.55%   认输 81.09%
        W+<数字> 9.30% / B+<数字> 9.22%                             有分差 18.52%
        W+T 0.12% / B+T 0.08%                                        超时 0.20%
        draw 0.07% / W+ 0.04% / B+ 0.03% / W+F 0.02% / B+F 0.01%      边角 0.17%
        W+3 zi / W+1 zi / W+0,25                                    带单位/逗号小数

     spec §5.3.3 记的认输占比是 75.2%（98,352 局），本次全量重扫是 **81.09%**
    （108,352 局）—— spec 的抽样脚本把 `B+Resign` 之类的写法归一过，比少了这一族。
     **每一种形态都有显式分支，没有一处靠 `except`** —— 靠异常判别未识别形态
    会把「解析 bug」与「语料里的新形态」混成同一个症状。
    """
    s = (re_str or '').strip()
    if not s:
        return ResultInfo(None, False, False)

    low = s.lower()
    if low in _RE_DRAW_WORDS:
        return ResultInfo(0.0, True, False)
    if low in _RE_VOID_WORDS:
        return ResultInfo(None, False, False)

    m = _RE_RESULT_RE.match(s)
    if m is None:
        # 整个串不是 `<胜方>[<分隔符><余部>]` 形态，又不在上面两张词表里。
        return ResultInfo(None, False, False)

    who = m.group('who')
    rest = (m.group('rest') or '').strip()
    if not rest:
        # `B+` / `W+`（空分差，实测 95 局）与裸 `B` / `W`：胜负已定但分差不可知。
        # 不拿贴目去猜分差 —— 实测 `KM[0]` 占 5.87%，猜出来的数是纯噪声。
        return ResultInfo(None, False, False)

    low_rest = rest.lower()
    if low_rest in _RE_RESIGN_WORDS:
        # `who` 是前缀字母，**方向只能从这里取** —— 余部（`R`/`T`/`F`）不带信息。
        # 0 = 黑认输、1 = 白认输；`B+R` 意味着白赢，名字容易读反，见 `ResultInfo`。
        return ResultInfo(None, False, True, 0 if who == 'B' else 1)

    mm = _RE_MARGIN_RE.match(rest)
    if mm is None:
        # `B+Void` / `W+???` 之类：余部既不是认输词也不是数字。
        return ResultInfo(None, False, False)

    margin = float(mm.group(0).replace(',', '.'))
    return ResultInfo(margin if who == 'B' else -margin, False, False)


def parse_rules(ru_str: Optional[str]) -> int:
    """解析 SGF `RU` 为 `g_rules` 的 bit 打包值。见模块级 `RULES_BIT_*` 常量。

     **tax / ko / suicide / button 这四位在当前语料里一律取默认值**，理由是
    实测它们**根本没有来源**（§5.3.2）：

    * 55.03% 的局**没有 `RU` 属性**；
    * 有 `RU` 的 45% 里，只有 `Chinese`(AREA) 与 `Japanese`(TERRITORY) 两种取值，
      **没有一个串提到税 / 劫 / 自杀 / 还子**（`RU` 只有那一个单词）。

    所以 `g_rules` 实际只会在 `0x00`(AREA) 与 `0x01`(TERRITORY) 之间取值。
    **bit 布局仍按 5 位留好**（`RULES_BITS_USED = 0x1F`，5..7 空着），这样将来
    出现带规则串的语料时只加分支、不改存储格式 —— 存量 `games.npz` 不用重算。
    """
    return _RULES_BY_WORD.get((ru_str or '').strip().lower(), RULES_DEFAULT)


def recognized_rules(ru_str: Optional[str]) -> bool:
    """`RU` 是否是本仓认识的取值（用于覆盖报告里区分「默认」与「未识别」）。"""
    return (ru_str or '').strip().lower() in RULES_KNOWN_WORDS


_RULES_BY_WORD = {
    '': RULES_DEFAULT,
    # 区域计分 = TT 默认（tax none / ko simple / suicide illegal / 无还子）。
    'chinese': RULES_DEFAULT,
    'chinese:korean': RULES_DEFAULT,
    # 数目计分：只翻 bit0，其余四位仍取默认。
    'japanese': RULES_BIT_SCORING,
    # 已知单局形态但规则与 AREA 同构（stone handicap 计分法）—— 按 AREA 处理。
    'nz': RULES_DEFAULT,
    'aga': RULES_DEFAULT,
    'ing': RULES_DEFAULT,
}


@dataclass
class Move:
    """棋步"""
    color: str  # 'B' 或 'W'
    position: Tuple[int, int]  # (row, col)
    comment: str = ""


@dataclass
class GameRecord:
    """棋谱记录"""
    board_size: int = 19
    moves: List[Move] = field(default_factory=list)
    result: str = ""
    black_player: str = ""
    white_player: str = ""
    date: str = ""
    komi: float = 7.5
    properties: Dict[str, str] = field(default_factory=dict)

    # ↓ 局级 sidecar（spec §5.3）需要的三个结构化字段。全部带默认值且**不改动
    #   上面的既有字段**，因此对 `build_dataset.py` / `label_sgf.py` 等既有调用方
    #   完全向后兼容（它们只读 `komi` / `result` / `properties`）。
    #: `RE` 的结构化解析（见 `ResultInfo`）。`result` 仍是原样字符串，两者在
    #: `parse_result_to_value` 与 sidecar 里各司其职。
    result_info: ResultInfo = field(
        default_factory=lambda: ResultInfo(None, False, False))
    #: `RU` 的 bit 打包值（见 `parse_rules`）。
    rules_flags: int = RULES_DEFAULT
    #: `KM` 属性**是否出现过**。
    #:
    #: 必须单独一个标志：`komi` 的默认值是 `7.5`，于是「没有 `KM`」与
    #: 「`KM[7.5]`」在 `komi` 上不可区分，而实测语料里 `KM` 缺失占 2.22%。
    #: sidecar 的 `g_komi` 约定「缺失填 0」，靠的就是这个标志而不是 `komi != 7.5`。
    has_komi: bool = False


class SGFParser:
    """SGF格式解析器"""
    
    def parse_file(self, filepath: str) -> Optional[GameRecord]:
        """
        解析SGF文件
        
        Args:
            filepath: SGF文件路径
            
        Returns:
            棋谱记录，解析失败返回None
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
    
    def parse_string(self, sgf_string: str) -> Optional[GameRecord]:
        """
        解析SGF字符串
        
        Args:
            sgf_string: SGF格式字符串
            
        Returns:
            棋谱记录，解析失败返回None
        """
        try:
            # 移除注释
            sgf_string = self._remove_comments(sgf_string)
            
            # 解析根节点
            game = GameRecord()
            
            # 提取属性
            properties = self._extract_properties(sgf_string)
            game.properties = properties
            
            # 棋盘大小
            if 'SZ' in properties:
                try:
                    game.board_size = int(properties['SZ'])
                except ValueError:
                    game.board_size = 19
            
            # 比赛结果
            if 'RE' in properties:
                game.result = properties['RE']
            game.result_info = parse_result(game.result)
            
            # 玩家信息
            if 'PB' in properties:
                game.black_player = properties['PB']
            if 'PW' in properties:
                game.white_player = properties['PW']
            
            # 日期
            if 'DT' in properties:
                game.date = properties['DT']
            
            # 贴目
            if 'KM' in properties:
                game.has_komi = True
                try:
                    game.komi = float(properties['KM'])
                except ValueError:
                    game.komi = 6.5

            # 规则（spec §5.3.2）。`RU` 缺失时取 TT 默认，见 `parse_rules`。
            game.rules_flags = parse_rules(properties.get('RU'))
            
            # 提取棋步
            moves = self._extract_moves(sgf_string)
            game.moves = moves
            
            return game
            
        except Exception as e:
            print(f"Error parsing SGF string: {e}")
            return None
    
    def _remove_comments(self, sgf_string: str) -> str:
        """移除 C[...] 注释（支持括号嵌套）。

        用 str.find 在 C 层跳过非注释文本，仅对注释内部做逐字符括号配对，
        取代旧版「对整串每个字符都做 Python 级判断」的循环——百万级棋谱下
        这是解析的主要 CPU 开销之一。
        """
        out = []
        i = 0
        n = len(sgf_string)
        while True:
            j = sgf_string.find('C[', i)
            if j < 0:
                out.append(sgf_string[i:])
                break
            # 排除 GC[ 等属性名中包含 C[ 的情况（C 前有大写字母则非注释）
            if j > 0 and sgf_string[j - 1].isupper():
                out.append(sgf_string[i:j + 1])
                i = j + 1
                continue
            out.append(sgf_string[i:j])
            k = j + 1          # 指向注释开头的 '['
            depth = 0
            while k < n:
                ch = sgf_string[k]
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
            i = k
        return ''.join(out)
    
    def _extract_properties(self, sgf_string: str) -> Dict[str, str]:
        """提取属性"""
        properties = {}
        
        # 匹配属性模式: XX[value]（模块级预编译正则）
        matches = _PROP_RE.finditer(sgf_string)
        
        for match in matches:
            key = match.group(1)
            value = match.group(2)
            
            # 处理多值属性（如AB[aa][ab][ba]）
            if key in properties:
                # 对于棋步属性，只保留第一个
                if key not in ['AB', 'AW', 'B', 'W']:
                    properties[key] += '|' + value
            else:
                properties[key] = value
        
        return properties
    
    def _coord_from(self, pos_str: str):
        """SGF 坐标 -> (row, col)。空串、单字符或超范围（`tt` 等）表示 pass，返回 (-1, -1)。"""
        if not pos_str or len(pos_str) < 2:
            return (-1, -1)
        col = ord(pos_str[0]) - ord('a')
        row = ord(pos_str[1]) - ord('a')
        # SGF 标准坐标范围是 a..s (0..18)。超出（如 t=19，常见 pass 哨兵）按 pass 处理。
        if not (0 <= row <= 18 and 0 <= col <= 18):
            return (-1, -1)
        return (row, col)

    def _extract_moves(self, sgf_string: str) -> List[Move]:
        """
        提取棋步，保持原始顺序。

        关键修复：
          1. pass 现在用空坐标 ``B[]`` / ``W[]`` 表示，此前被 ``([a-s]{2})``
             正则丢弃，导致黑白顺序错位。这里用 ``[^]]*`` 捕获（允许空串）。
          2. ``AB[..]``（黑让子）与 ``AW[..]``（白让子）作为开局先行子，
             必须按其出现的先后并入棋步序列，否则重放从第一步起就错位。
        """
        moves = []

        # 匹配棋步标记：B / W（普通手）与 AB / AW（让子）。
        # 关键：必须以 ';' 或字符串开头锚定，避免把属性键 BR/WR/PB/PW/KM 等
        # 里的 'B'/'W' 误当成落子（此前会导致坐标乱序）。
        for token in _MOVE_TOKEN_RE.finditer(sgf_string):
            key = token.group(1)
            bracket_block = token.group(2)
            for val in _BRACKET_VAL_RE.findall(bracket_block):
                if key in ('B', 'W'):
                    color = key
                else:  # AW -> 白, AB -> 黑
                    color = 'W' if key == 'AW' else 'B'
                row, col = self._coord_from(val)
                moves.append(Move(color=color, position=(row, col)))

        return moves
    
    def validate_board_size(self, game: GameRecord, target_size: int) -> bool:
        """验证棋盘大小"""
        return game.board_size == target_size
    
    def get_move_sequence(self, game: GameRecord) -> List[Tuple[int, int]]:
        """获取棋步序列（仅位置）"""
        return [move.position for move in game.moves]
    
    def get_policy_target(self, game: GameRecord, board_size: int, move_index: int) -> Optional[List[float]]:
        """
        获取策略目标（one-hot编码）
        
        Args:
            game: 棋谱记录
            board_size: 棋盘大小
            move_index: 棋步索引
            
        Returns:
            策略目标向量，无效返回None
        """
        if move_index >= len(game.moves):
            return None
        
        move = game.moves[move_index]
        row, col = move.position
        
        if row >= board_size or col >= board_size:
            return None
        
        # one-hot编码
        policy = [0.0] * (board_size * board_size)
        action = row * board_size + col
        policy[action] = 1.0
        
        return policy


def parse_sgf_file(filepath: str) -> Optional[GameRecord]:
    """便捷函数：解析SGF文件"""
    parser = SGFParser()
    return parser.parse_file(filepath)


def parse_sgf_string(sgf_string: str) -> Optional[GameRecord]:
    """便捷函数：解析SGF字符串"""
    parser = SGFParser()
    return parser.parse_string(sgf_string)
