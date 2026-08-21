import torch
import numpy as np
from typing import List, Tuple, Optional
from ..networks.resnet import MuZeroNet


class MinimaxSearch:
    """Minimax搜索算法，带Alpha-Beta剪枝"""
    
    def __init__(self, 
                 model: MuZeroNet,
                 board_size: int = 9,
                 search_depth: int = 10,
                 top_k: int = 5,
                 use_alpha_beta: bool = True,
                 device: str = 'cpu'):
        """
        初始化Minimax搜索
        
        Args:
            model: MuZero网络模型
            board_size: 棋盘大小
            search_depth: 搜索深度
            top_k: 策略网络选择的候选着法数量
            use_alpha_beta: 是否使用Alpha-Beta剪枝
            device: 设备 (cpu/cuda)
        """
        self.model = model
        self.board_size = board_size
        self.search_depth = search_depth
        self.top_k = top_k
        self.use_alpha_beta = use_alpha_beta
        self.device = device
        self.action_size = board_size * board_size
    
    def get_legal_moves(self, board: np.ndarray) -> List[int]:
        """获取合法着法（简单实现：空位）"""
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    legal_moves.append(i * self.board_size + j)
        return legal_moves
    
    def board_to_tensor(self, board: np.ndarray, current_player: int) -> torch.Tensor:
        """将棋盘转换为网络输入张量"""
        # 简单实现：19通道
        # 通道0-7: 当前玩家棋子位置（8步历史）
        # 通道8-15: 对手棋子位置（8步历史）
        # 通道16: 当前玩家标记
        # 通道17: 合法着法标记
        # 通道18: 棋盘大小标记
        
        tensor = torch.zeros(1, 19, self.board_size, self.board_size, device=self.device)
        
        # 当前玩家棋子
        mask_current = torch.tensor(board == current_player, dtype=torch.bool, device=self.device)
        tensor[0, 0][mask_current] = 1
        # 对手棋子
        mask_opponent = torch.tensor(board == -current_player, dtype=torch.bool, device=self.device)
        tensor[0, 8][mask_opponent] = 1
        # 当前玩家标记
        tensor[0, 16] = 1 if current_player == 1 else -1
        # 合法着法标记
        legal_moves = self.get_legal_moves(board)
        for move in legal_moves:
            i, j = move // self.board_size, move % self.board_size
            tensor[0, 17, i, j] = 1
        
        return tensor
    
    def get_candidate_moves(self, board: np.ndarray, current_player: int) -> List[Tuple[int, float]]:
        """使用策略网络获取候选着法"""
        with torch.no_grad():
            tensor = self.board_to_tensor(board, current_player)
            state, policy, value = self.model.initial_inference(tensor)
            
            # 获取策略概率
            policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
            
            # 获取合法着法
            legal_moves = self.get_legal_moves(board)
            
            # 按概率排序
            move_probs = [(move, policy_probs[move]) for move in legal_moves]
            move_probs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回Top-K候选
            return move_probs[:self.top_k]
    
    def minimax(self, 
                state: torch.Tensor, 
                depth: int, 
                is_maximizing: bool, 
                alpha: float = float('-inf'), 
                beta: float = float('inf')) -> float:
        """
        Minimax搜索
        
        Args:
            state: 当前状态
            depth: 剩余搜索深度
            is_maximizing: 是否为最大化玩家
            alpha: Alpha值（用于Alpha-Beta剪枝）
            beta: Beta值（用于Alpha-Beta剪枝）
            
        Returns:
            评估值
        """
        if depth == 0:
            # 叶节点：使用价值网络评估
            with torch.no_grad():
                _, value = self.model.prediction(state)
                return value.item()
        
        # 获取候选着法
        policy, _ = self.model.prediction(state)
        policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
        
        # 获取合法着法（简化：假设所有位置都合法）
        legal_moves = list(range(self.action_size))
        
        # 按概率排序并取Top-K
        move_probs = [(move, policy_probs[move]) for move in legal_moves]
        move_probs.sort(key=lambda x: x[1], reverse=True)
        candidate_moves = [move for move, _ in move_probs[:self.top_k]]
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in candidate_moves:
                # 创建动作one-hot
                action_one_hot = torch.zeros(1, 1, self.board_size, self.board_size, device=self.device)
                action_one_hot.view(1, -1).scatter_(1, torch.tensor([[move]], device=self.device), 1)
                
                # 递归推理
                next_state, reward = self.model.dynamics(state, action_one_hot)
                next_state = next_state + 0.0  # 残差连接
                
                eval_score = self.minimax(next_state, depth - 1, False, alpha, beta)
                eval_score += reward.item()  # 加上即时奖励
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if self.use_alpha_beta and beta <= alpha:
                    break  # Beta剪枝
            
            return max_eval
        else:
            min_eval = float('inf')
            for move in candidate_moves:
                # 创建动作one-hot
                action_one_hot = torch.zeros(1, 1, self.board_size, self.board_size, device=self.device)
                action_one_hot.view(1, -1).scatter_(1, torch.tensor([[move]], device=self.device), 1)
                
                # 递归推理
                next_state, reward = self.model.dynamics(state, action_one_hot)
                next_state = next_state + 0.0  # 残差连接
                
                eval_score = self.minimax(next_state, depth - 1, True, alpha, beta)
                eval_score -= reward.item()  # 对手的奖励是我们的损失
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if self.use_alpha_beta and beta <= alpha:
                    break  # Alpha剪枝
            
            return min_eval
    
    def search(self, board: np.ndarray, current_player: int) -> Tuple[int, float]:
        """
        执行搜索，返回最佳着法
        
        Args:
            board: 当前棋盘状态
            current_player: 当前玩家 (1 或 -1)
            
        Returns:
            (最佳着法, 评估值)
        """
        with torch.no_grad():
            # 获取初始状态
            tensor = self.board_to_tensor(board, current_player)
            state, initial_policy, initial_value = self.model.initial_inference(tensor)
            
            # 获取候选着法
            candidates = self.get_candidate_moves(board, current_player)
            
            best_move = None
            best_score = float('-inf')
            
            # 对每个候选着法进行评估
            for move, prob in candidates:
                # 创建动作one-hot
                action_one_hot = torch.zeros(1, 1, self.board_size, self.board_size, device=self.device)
                action_one_hot.view(1, -1).scatter_(1, torch.tensor([[move]], device=self.device), 1)
                
                # 递归推理
                next_state, reward = self.model.dynamics(state, action_one_hot)
                next_state = next_state + 0.0  # 残差连接
                
                # Minimax搜索
                score = self.minimax(next_state, self.search_depth - 1, False)
                score += reward.item()  # 加上即时奖励
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            return best_move, best_score
    
    def get_move_probabilities(self, board: np.ndarray, current_player: int) -> np.ndarray:
        """获取所有着法的概率分布（用于训练）"""
        with torch.no_grad():
            tensor = self.board_to_tensor(board, current_player)
            state, policy, value = self.model.initial_inference(tensor)
            
            # 获取策略概率
            policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
            
            # 归一化
            total = policy_probs.sum()
            if total > 0:
                policy_probs /= total
            else:
                # 如果没有概率，使用均匀分布
                policy_probs = np.ones(self.action_size) / self.action_size
            
            return policy_probs
