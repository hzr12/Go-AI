"""
棋谱数据加载器：加载和处理SGF棋谱文件
"""

import os
import tarfile
import io
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .sgf_parser import SGFParser, GameRecord, Move


@dataclass
class TrainingExample:
    """训练数据样本"""
    state: np.ndarray      # 棋盘状态 (19, board_size, board_size)
    action: int            # 棋步位置 (0 to board_size*board_size-1)
    policy: np.ndarray     # 策略目标 (board_size*board_size,)
    value: float           # 价值目标 (1.0, -1.0, 0.0)
    current_player: int    # 当前玩家 (1=黑, -1=白)


class GoGameSimulator:
    """围棋游戏模拟器：用于生成训练数据"""
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1=黑, -1=白
        self.move_count = 0
    
    def reset(self):
        """重置棋盘"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
    
    def make_move(self, move: Tuple[int, int]) -> bool:
        """执行着法"""
        row, col = move
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        if self.board[row, col] != 0:
            return False
        
        self.board[row, col] = self.current_player
        self.current_player = -self.current_player
        self.move_count += 1
        return True
    
    def get_state_tensor(self) -> np.ndarray:
        """获取状态张量"""
        tensor = np.zeros((19, self.board_size, self.board_size), dtype=np.float32)
        
        # 当前玩家棋子
        tensor[0][self.board == self.current_player] = 1
        # 对手棋子
        tensor[8][self.board == -self.current_player] = 1
        # 当前玩家标记
        tensor[16] = 1 if self.current_player == 1 else -1
        
        return tensor
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """获取合法着法"""
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    legal_moves.append((i, j))
        return legal_moves
    
    def get_result(self) -> float:
        """获取游戏结果（简化版：数子法）"""
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        
        if black_count > white_count:
            return 1.0  # 黑胜
        elif white_count > black_count:
            return -1.0  # 白胜
        else:
            return 0.0  # 平局


class GameLoader:
    """棋谱数据加载器"""
    
    def __init__(self, board_size: int = 19):
        """
        初始化加载器
        
        Args:
            board_size: 棋盘大小
        """
        self.board_size = board_size
        self.parser = SGFParser()
    
    def load_directory(self, dirpath: str, max_games: int = 0) -> List[GameRecord]:
        """
        加载目录下所有SGF文件
        
        Args:
            dirpath: 目录路径
            max_games: 最大加载数量（0表示全部）
            
        Returns:
            棋谱记录列表
        """
        game_records = []
        
        if not os.path.exists(dirpath):
            print(f"Directory not found: {dirpath}")
            return game_records
        
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.lower().endswith('.sgf'):
                    filepath = os.path.join(root, file)
                    game = self.parser.parse_file(filepath)
                    
                    if game is not None:
                        # 验证棋盘大小
                        if game.board_size == self.board_size:
                            game_records.append(game)
                        
                        if max_games > 0 and len(game_records) >= max_games:
                            return game_records
        
        return game_records
    
    def load_file(self, filepath: str) -> Optional[GameRecord]:
        """
        加载单个SGF文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            棋谱记录，加载失败返回None
        """
        game = self.parser.parse_file(filepath)
        
        if game is not None and game.board_size != self.board_size:
            print(f"Board size mismatch: {game.board_size} vs {self.board_size}")
            return None
        
        return game
    
    def load_tgz(self, tgz_path: str, max_games: int = 0) -> List[GameRecord]:
        """
        从tgz/tar.gz压缩包直接加载SGF棋谱（无需解压）
        
        Args:
            tgz_path: tgz文件路径
            max_games: 最大加载数量（0表示全部）
            
        Returns:
            棋谱记录列表
        """
        game_records = []
        
        if not os.path.exists(tgz_path):
            print(f"File not found: {tgz_path}")
            return game_records
        
        with tarfile.open(tgz_path, 'r:gz') as tar:
            sgf_members = [m for m in tar.getmembers() 
                          if m.isfile() and m.name.lower().endswith('.sgf')]
            
            for member in sgf_members:
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    content = f.read().decode('utf-8', errors='ignore')
                    game = self.parser.parse_string(content)
                    
                    if game is not None and game.board_size == self.board_size:
                        game_records.append(game)
                    
                    if max_games > 0 and len(game_records) >= max_games:
                        return game_records
                except Exception:
                    continue
        
        return game_records
    
    def load_tgz_auto(self, data_dir: str = 'data', max_games: int = 0) -> List[GameRecord]:
        """
        自动扫描目录下所有tgz文件并加载
        
        Args:
            data_dir: 数据目录路径
            max_games: 最大加载数量（0表示全部）
            
        Returns:
            棋谱记录列表
        """
        game_records = []
        
        if not os.path.exists(data_dir):
            print(f"Directory not found: {data_dir}")
            return game_records
        
        tgz_files = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(('.tgz', '.tar.gz')):
                    tgz_files.append(os.path.join(root, f))
        
        if not tgz_files:
            print(f"No .tgz/.tar.gz files found in {data_dir}")
            return game_records
        
        for tgz_path in tgz_files:
            print(f"Loading {os.path.basename(tgz_path)}...")
            remaining = max_games - len(game_records) if max_games > 0 else 0
            games = self.load_tgz(tgz_path, max_games=remaining)
            game_records.extend(games)
            print(f"  Loaded {len(games)} games (total: {len(game_records)})")
            
            if max_games > 0 and len(game_records) >= max_games:
                break
        
        return game_records
    
    def game_to_training_data(self, game: GameRecord) -> List[TrainingExample]:
        """
        将棋谱转换为训练数据
        
        Args:
            game: 棋谱记录
            
        Returns:
            训练数据列表
        """
        training_data = []
        simulator = GoGameSimulator(self.board_size)
        simulator.reset()
        
        result = game.result
        game_value = self._parse_result(result)
        
        for i, move in enumerate(game.moves):
            # 获取当前状态
            state = simulator.get_state_tensor()
            current_player = simulator.current_player
            
            # 获取策略目标
            row, col = move.position
            action = row * self.board_size + col
            policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
            policy[action] = 1.0
            
            # 计算价值（从当前玩家视角）
            if game_value == 0.0:
                value = 0.0
            elif (game_value > 0 and current_player == 1) or (game_value < 0 and current_player == -1):
                value = 1.0  # 当前玩家获胜
            else:
                value = -1.0  # 当前玩家失败
            
            # 创建训练样本
            example = TrainingExample(
                state=state,
                action=action,
                policy=policy,
                value=value,
                current_player=current_player
            )
            training_data.append(example)
            
            # 执行着法
            simulator.make_move((row, col))
        
        return training_data
    
    def _parse_result(self, result: str) -> float:
        """解析比赛结果"""
        if not result:
            return 0.0
        
        result = result.strip().upper()
        
        if 'B+' in result or 'BLACK' in result:
            return 1.0
        elif 'W+' in result or 'WHITE' in result:
            return -1.0
        elif 'DRAW' in result or 'JIGO' in result:
            return 0.0
        else:
            # 尝试解析分数
            try:
                # 移除非数字字符
                score_str = ''.join(c for c in result if c.isdigit() or c == '.')
                if score_str:
                    score = float(score_str)
                    if 'B' in result:
                        return 1.0 if score > 0 else -1.0
                    elif 'W' in result:
                        return -1.0 if score > 0 else 1.0
            except ValueError:
                pass
        
        return 0.0
    
    def augment_data(self, training_data: List[TrainingExample]) -> List[TrainingExample]:
        """
        数据增强：8种对称变换
        
        Args:
            training_data: 原始训练数据
            
        Returns:
            增强后的训练数据
        """
        augmented_data = []
        
        for example in training_data:
            # 原始数据
            augmented_data.append(example)
            
            # 8种对称变换
            for transform_id in range(1, 8):
                aug_state, aug_policy = self._apply_symmetry(
                    example.state, example.policy, transform_id
                )
                
                aug_example = TrainingExample(
                    state=aug_state,
                    action=example.action,  # 动作需要根据变换调整
                    policy=aug_policy,
                    value=example.value,
                    current_player=example.current_player
                )
                augmented_data.append(aug_example)
        
        return augmented_data
    
    def _apply_symmetry(self, state: np.ndarray, policy: np.ndarray, 
                       transform_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """应用对称变换"""
        board_size = self.board_size
        
        if transform_id == 0:
            return state, policy
        
        if transform_id == 1:  # 旋转90度
            new_state = np.rot90(state, axes=(1, 2))
            new_policy = policy.reshape(board_size, board_size)
            new_policy = np.rot90(new_policy)
            return new_state, new_policy.flatten()
        
        if transform_id == 2:  # 旋转180度
            new_state = np.rot90(state, 2, axes=(1, 2))
            new_policy = policy.reshape(board_size, board_size)
            new_policy = np.rot90(new_policy, 2)
            return new_state, new_policy.flatten()
        
        if transform_id == 3:  # 旋转270度
            new_state = np.rot90(state, 3, axes=(1, 2))
            new_policy = policy.reshape(board_size, board_size)
            new_policy = np.rot90(new_policy, 3)
            return new_state, new_policy.flatten()
        
        if transform_id == 4:  # 水平翻转
            new_state = np.flipud(state)
            new_policy = policy.reshape(board_size, board_size)
            new_policy = np.flipud(new_policy)
            return new_state, new_policy.flatten()
        
        if transform_id == 5:  # 垂直翻转
            new_state = np.fliplr(state)
            new_policy = policy.reshape(board_size, board_size)
            new_policy = np.fliplr(new_policy)
            return new_state, new_policy.flatten()
        
        if transform_id == 6:  # 转置
            new_state = state.transpose(0, 2, 1)
            new_policy = policy.reshape(board_size, board_size)
            new_policy = new_policy.T
            return new_state, new_policy.flatten()
        
        if transform_id == 7:  # 反转置
            new_state = state.transpose(0, 2, 1)
            new_state = np.fliplr(new_state)
            new_state = np.flipud(new_state)
            new_policy = policy.reshape(board_size, board_size)
            new_policy = new_policy.T
            new_policy = np.fliplr(new_policy)
            new_policy = np.flipud(new_policy)
            return new_state, new_policy.flatten()
        
        return state, policy
