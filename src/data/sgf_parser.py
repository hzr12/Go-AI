"""
SGF格式解析器：解析围棋棋谱文件
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


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
    komi: float = 6.5
    properties: Dict[str, str] = field(default_factory=dict)


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
                try:
                    game.komi = float(properties['KM'])
                except ValueError:
                    game.komi = 6.5
            
            # 提取棋步
            moves = self._extract_moves(sgf_string)
            game.moves = moves
            
            return game
            
        except Exception as e:
            print(f"Error parsing SGF string: {e}")
            return None
    
    def _remove_comments(self, sgf_string: str) -> str:
        """移除注释"""
        # 移除C[...]注释
        result = []
        i = 0
        in_comment = False
        bracket_count = 0
        
        while i < len(sgf_string):
            if sgf_string[i:i+2] == 'C[' and not in_comment:
                in_comment = True
                bracket_count = 1
                i += 2
                continue
            
            if in_comment:
                if sgf_string[i] == '[':
                    bracket_count += 1
                elif sgf_string[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        in_comment = False
                i += 1
                continue
            
            result.append(sgf_string[i])
            i += 1
        
        return ''.join(result)
    
    def _extract_properties(self, sgf_string: str) -> Dict[str, str]:
        """提取属性"""
        properties = {}
        
        # 匹配属性模式: XX[value]
        pattern = r'([A-Z]{1,2})\[(.*?)\]'
        matches = re.finditer(pattern, sgf_string)
        
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
    
    def _extract_moves(self, sgf_string: str) -> List[Move]:
        """提取棋步"""
        moves = []
        
        # 匹配棋步模式: B[aa] 或 W[ab]
        pattern = r'([BW])\[([a-s]{2})\]'
        matches = re.finditer(pattern, sgf_string)
        
        for match in matches:
            color = match.group(1)
            pos_str = match.group(2)
            
            # 转换坐标
            col = ord(pos_str[0]) - ord('a')
            row = ord(pos_str[1]) - ord('a')
            
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
