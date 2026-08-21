import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from ..networks.alphanet import AlphaGoNet
from ..training.trainer import GoGame


class AlphaEvaluator:
    """AlphaGoNet 专用评估器（无搜索，纯网络推理）"""
    
    def __init__(self, model: AlphaGoNet, board_size: int = 9,
                 temperature: float = 1.0, device: str = 'cpu'):
        self.model = model.to(device).eval()
        self.device = device
        self.board_size = board_size
        self.temperature = temperature
    
    def board_to_tensor(self, board: np.ndarray, current_player: int) -> torch.Tensor:
        tensor = np.zeros((1, 19, self.board_size, self.board_size), dtype=np.float32)
        tensor[0, 0][board == current_player] = 1
        tensor[0, 8][board == -current_player] = 1
        tensor[0, 16] = 1 if current_player == 1 else -1
        for move in self._get_legal_moves(board):
            i, j = move // self.board_size, move % self.board_size
            tensor[0, 17, i, j] = 1
        return torch.FloatTensor(tensor).to(self.device)
    
    def _get_legal_moves(self, board: np.ndarray) -> List[int]:
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    moves.append(i * self.board_size + j)
        return moves
    
    def select_move(self, board: np.ndarray, current_player: int,
                    temperature: float = 0) -> int:
        state = self.board_to_tensor(board, current_player)
        
        with torch.no_grad():
            policy_logits, value, _ = self.model(state)
        
        probs = F.softmax(policy_logits / max(temperature, 1e-8), dim=-1)
        legal = self._get_legal_moves(board)
        
        mask = torch.zeros_like(probs)
        mask[0, legal] = 1
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        if temperature > 0:
            move = torch.multinomial(probs, 1).item()
        else:
            move = torch.argmax(probs).item()
        
        return move
    
    def play_game(self, player_temp: float = 0, opponent_temp: float = 0,
                  player_is_black: bool = True) -> Dict:
        game = GoGame(self.board_size)
        game.reset()
        
        step = 0
        max_steps = self.board_size * self.board_size
        
        while not game.is_game_over() and step < max_steps:
            is_player_turn = (game.current_player == 1) == player_is_black
            temp = player_temp if is_player_turn else opponent_temp
            
            move = self.select_move(game.board, game.current_player, temperature=temp)
            game.make_move(move)
            step += 1
        
        result = game.get_result()
        return {'result': result, 'num_moves': step}
    
    def play_two_models(self, model_a: AlphaGoNet, model_b: AlphaGoNet,
                        temp_a: float = 0, temp_b: float = 0) -> Dict:
        game = GoGame(self.board_size)
        game.reset()
        
        step = 0
        max_steps = self.board_size * self.board_size
        
        while not game.is_game_over() and step < max_steps:
            current_model = model_a if game.current_player == 1 else model_b
            temp = temp_a if game.current_player == 1 else temp_b
            
            state = self.board_to_tensor(game.board, game.current_player)
            with torch.no_grad():
                policy_logits, _, _ = current_model(state)
            
            probs = F.softmax(policy_logits / max(temp, 1e-8), dim=-1)
            legal = self._get_legal_moves(game.board)
            
            mask = torch.zeros_like(probs)
            mask[0, legal] = 1
            probs = probs * mask
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            if temp > 0:
                move = torch.multinomial(probs, 1).item()
            else:
                move = torch.argmax(probs).item()
            
            game.make_move(move)
            step += 1
        
        result = game.get_result()
        return {'result': result, 'num_moves': step}
    
    def evaluate_against_random(self, num_games: int = 100,
                                player_first: bool = True) -> Dict:
        wins = 0
        losses = 0
        draws = 0
        
        for game_idx in range(num_games):
            is_first = (game_idx % 2 == 0) == player_first
            
            game = GoGame(self.board_size)
            game.reset()
            step = 0
            max_steps = self.board_size * self.board_size
            
            while not game.is_game_over() and step < max_steps:
                legal = game.get_legal_moves()
                
                if (game.current_player == 1) == is_first:
                    move = self.select_move(game.board, game.current_player, temperature=0)
                else:
                    move = np.random.choice(legal)
                
                game.make_move(move)
                step += 1
            
            result = game.get_result()
            if is_first:
                if result == 1: wins += 1
                elif result == -1: losses += 1
                else: draws += 1
            else:
                if result == -1: wins += 1
                elif result == 1: losses += 1
                else: draws += 1
            
            if (game_idx + 1) % 10 == 0:
                print(f"  {game_idx + 1}/{num_games}: W={wins} L={losses} D={draws}")
        
        return {
            'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins / num_games, 'num_games': num_games
        }
    
    def evaluate_elo(self, num_games: int = 100,
                     opponent_elo: float = 1000) -> Dict:
        elo = opponent_elo
        k_factor = 32
        results = []
        
        for game_idx in range(num_games):
            game = GoGame(self.board_size)
            game.reset()
            step = 0
            max_steps = self.board_size * self.board_size
            
            while not game.is_game_over() and step < max_steps:
                legal = game.get_legal_moves()
                
                if game.current_player == 1:
                    move = self.select_move(game.board, game.current_player, temperature=0)
                else:
                    move = np.random.choice(legal)
                
                game.make_move(move)
                step += 1
            
            result = game.get_result()
            results.append(result)
            
            score = 1.0 if result == 1 else (0.0 if result == -1 else 0.5)
            expected = 1 / (1 + 10 ** ((opponent_elo - elo) / 400))
            elo += k_factor * (score - expected)
            
            if (game_idx + 1) % 10 == 0:
                print(f"  ELO: {game_idx + 1}/{num_games}, Current={elo:.0f}")
        
        return {
            'elo': elo,
            'wins': results.count(1),
            'losses': results.count(-1),
            'draws': results.count(0),
            'num_games': num_games
        }
    
    def benchmark(self, num_positions: int = 100) -> Dict:
        times = []
        
        for _ in range(num_positions):
            board = np.random.choice([-1, 0, 1], size=(self.board_size, self.board_size))
            current_player = np.random.choice([-1, 1])
            
            start = time.time()
            self.select_move(board, current_player, temperature=0)
            times.append(time.time() - start)
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'positions_per_second': 1.0 / np.mean(times),
            'num_positions': num_positions
        }
