import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from ..networks.resnet import MuZeroNet
from ..search.minimax import MinimaxSearch
from ..training.trainer import GoGame


class Evaluator:
    """评估器"""
    
    def __init__(self, 
                 model: MuZeroNet,
                 board_size: int = 9,
                 search_depth: int = 10,
                 top_k: int = 5,
                 device: str = 'cpu'):
        """
        初始化评估器
        
        Args:
            model: MuZero网络
            board_size: 棋盘大小
            search_depth: 搜索深度
            top_k: 候选着法数量
            device: 设备
        """
        self.model = model.to(device)
        self.device = device
        self.board_size = board_size
        
        # 搜索算法
        self.search = MinimaxSearch(
            model=model,
            board_size=board_size,
            search_depth=search_depth,
            top_k=top_k,
            device=device
        )
    
    def play_game(self, player1_search: MinimaxSearch, player2_search: MinimaxSearch) -> Dict:
        """
        让两个搜索算法对弈
        
        Args:
            player1_search: 玩家1的搜索算法（黑棋）
            player2_search: 玩家2的搜索算法（白棋）
            
        Returns:
            游戏记录
        """
        game = GoGame(self.board_size)
        game.reset()
        
        moves = []
        states = []
        
        while not game.is_game_over():
            current_search = player1_search if game.current_player == 1 else player2_search
            
            # 获取着法
            move, score = current_search.search(game.board, game.current_player)
            
            moves.append(move)
            states.append(game.get_state_tensor().copy())
            
            # 执行着法
            game.make_move(move)
        
        result = game.get_result()
        
        return {
            'moves': moves,
            'states': states,
            'result': result,
            'num_moves': len(moves)
        }
    
    def evaluate_against_random(self, num_games: int = 100) -> Dict:
        """
        与随机玩家对弈评估
        
        Args:
            num_games: 对弈局数
            
        Returns:
            评估结果
        """
        wins = 0
        losses = 0
        draws = 0
        
        for game_idx in range(num_games):
            # 随机玩家搜索（简单实现）
            class RandomSearch:
                def __init__(self, board_size):
                    self.board_size = board_size
                
                def search(self, board, current_player):
                    legal_moves = []
                    for i in range(self.board_size):
                        for j in range(self.board_size):
                            if board[i, j] == 0:
                                legal_moves.append(i * self.board_size + j)
                    
                    if not legal_moves:
                        return 0, 0.0
                    
                    move = np.random.choice(legal_moves)
                    return move, 0.0
            
            random_search = RandomSearch(self.board_size)
            
            # 让模型先手
            if game_idx % 2 == 0:
                result = self.play_game(self.search, random_search)
                if result['result'] == 1:
                    wins += 1
                elif result['result'] == -1:
                    losses += 1
                else:
                    draws += 1
            else:
                # 让模型后手
                result = self.play_game(random_search, self.search)
                if result['result'] == -1:
                    wins += 1
                elif result['result'] == 1:
                    losses += 1
                else:
                    draws += 1
            
            if (game_idx + 1) % 10 == 0:
                print(f"Evaluation: {game_idx + 1}/{num_games} games completed")
        
        win_rate = wins / num_games
        
        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'num_games': num_games
        }
    
    def calculate_elo(self, results: List[Dict], initial_elo: float = 1500) -> float:
        """
        计算ELO评分
        
        Args:
            results: 对弈结果列表
            initial_elo: 初始ELO评分
            
        Returns:
            ELO评分
        """
        elo = initial_elo
        k_factor = 32  # K因子
        
        for result in results:
            if result['result'] == 1:
                score = 1.0
            elif result['result'] == -1:
                score = 0.0
            else:
                score = 0.5
            
            # 简化的ELO计算（假设对手ELO为1500）
            expected_score = 1 / (1 + 10 ** ((1500 - elo) / 400))
            elo += k_factor * (score - expected_score)
        
        return elo
    
    def evaluate_elo(self, num_games: int = 100) -> Dict:
        """
        使用ELO评分评估
        
        Args:
            num_games: 对弈局数
            
        Returns:
            ELO评估结果
        """
        results = []
        
        for game_idx in range(num_games):
            # 与固定ELO的对手对弈（简化）
            result = self.play_game(self.search, self.search)
            results.append(result)
            
            if (game_idx + 1) % 10 == 0:
                current_elo = self.calculate_elo(results)
                print(f"ELO Evaluation: {game_idx + 1}/{num_games} games, Current ELO: {current_elo:.0f}")
        
        final_elo = self.calculate_elo(results)
        
        return {
            'elo': final_elo,
            'num_games': num_games,
            'results': results
        }
    
    def evaluate_position(self, board: np.ndarray, current_player: int) -> Dict:
        """
        评估单个局面
        
        Args:
            board: 棋盘状态
            current_player: 当前玩家
            
        Returns:
            评估结果
        """
        move, score = self.search.search(board, current_player)
        
        # 获取所有着法的概率
        policy = self.search.get_move_probabilities(board, current_player)
        
        return {
            'best_move': move,
            'score': score,
            'policy': policy,
            'board': board.copy()
        }
    
    def benchmark(self, num_positions: int = 100) -> Dict:
        """
        性能基准测试
        
        Args:
            num_positions: 测试局面数量
            
        Returns:
            性能结果
        """
        import time
        
        times = []
        moves = []
        
        for _ in range(num_positions):
            # 随机生成棋盘
            board = np.random.choice([-1, 0, 1], size=(self.board_size, self.board_size))
            current_player = np.random.choice([-1, 1])
            
            # 计时
            start_time = time.time()
            move, score = self.search.search(board, current_player)
            end_time = time.time()
            
            times.append(end_time - start_time)
            moves.append(move)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        return {
            'avg_time': avg_time,
            'max_time': max_time,
            'min_time': min_time,
            'num_positions': num_positions,
            'positions_per_second': 1.0 / avg_time if avg_time > 0 else 0
        }
