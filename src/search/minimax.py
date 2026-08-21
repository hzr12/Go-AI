import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from ..networks.resnet import MuZeroNet


class MinimaxSearch:
    """Minimax搜索算法，GPU批量推理优化"""
    
    def __init__(self, 
                 model: MuZeroNet,
                 board_size: int = 9,
                 search_depth: int = 10,
                 top_k: int = 5,
                 use_alpha_beta: bool = True,
                 device: str = 'cpu',
                 use_amp: bool = False):
        """
        初始化Minimax搜索
        
        Args:
            model: MuZero网络模型
            board_size: 棋盘大小
            search_depth: 搜索深度
            top_k: 策略网络选择的候选着法数量
            use_alpha_beta: 是否使用Alpha-Beta剪枝
            device: 设备 (cpu/cuda)
            use_amp: 是否使用自动混合精度
        """
        self.model = model
        self.board_size = board_size
        self.search_depth = search_depth
        self.top_k = top_k
        self.use_alpha_beta = use_alpha_beta
        self.device = device
        self.action_size = board_size * board_size
        self.use_amp = use_amp
        
        # 预创建所有action one-hot模板
        self._create_action_templates()
    
    def _create_action_templates(self):
        """预创建所有可能的action one-hot张量"""
        self.action_templates = torch.zeros(
            self.action_size, 1, self.board_size, self.board_size, 
            device=self.device
        )
        for i in range(self.action_size):
            self.action_templates[i].view(1, -1).scatter_(
                1, torch.tensor([[i]], device=self.device), 1
            )
    
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
        if legal_moves:
            legal_tensor = torch.tensor(legal_moves, device=self.device)
            rows = legal_tensor // self.board_size
            cols = legal_tensor % self.board_size
            tensor[0, 17, rows, cols] = 1
        
        return tensor
    
    def _forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """统一的网络前向传播，自动处理AMP"""
        if self.use_amp and self.device == 'cuda':
            with torch.cuda.amp.autocast():
                return self.model.initial_inference(tensor)
        else:
            return self.model.initial_inference(tensor)
    
    def _prediction(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """统一的prediction网络前向传播"""
        if self.use_amp and self.device == 'cuda':
            with torch.cuda.amp.autocast():
                return self.model.prediction(state)
        else:
            return self.model.prediction(state)
    
    def _dynamics(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """统一的dynamics网络前向传播"""
        if self.use_amp and self.device == 'cuda':
            with torch.cuda.amp.autocast():
                return self.model.dynamics(state, action)
        else:
            return self.model.dynamics(state, action)
    
    def _batch_dynamics(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量dynamics推理：一次评估多个候选着法
        state: (1, C, H, W)
        actions: (N, 1, H, W) - N个候选着法的one-hot
        返回: next_states (N, C, H, W), rewards (N, 1)
        """
        # 扩展state以匹配batch_size
        batch_size = actions.shape[0]
        state_expanded = state.expand(batch_size, -1, -1, -1)
        
        if self.use_amp and self.device == 'cuda':
            with torch.cuda.amp.autocast():
                next_states, rewards = self.model.dynamics(state_expanded, actions)
        else:
            next_states, rewards = self.model.dynamics(state_expanded, actions)
        
        return next_states, rewards
    
    def _batch_prediction(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量prediction推理：一次评估多个状态
        states: (N, C, H, W)
        返回: policies (N, action_size), values (N, 1)
        """
        if self.use_amp and self.device == 'cuda':
            with torch.cuda.amp.autocast():
                policies, values = self.model.prediction(states)
        else:
            policies, values = self.model.prediction(states)
        
        return policies, values
    
    def get_candidate_moves_gpu(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU上获取候选着法（不传回CPU）
        返回: candidate_indices (top_k), policy_probs (action_size)
        """
        policy, _ = self._prediction(state)
        policy_probs = F.softmax(policy, dim=1).squeeze(0)  # (action_size,)
        
        # 使用torch.topk在GPU上选择Top-K
        top_k_values, top_k_indices = torch.topk(policy_probs, self.top_k)
        
        return top_k_indices, policy_probs
    
    def minimax_batch(self, 
                      state: torch.Tensor, 
                      depth: int, 
                      is_maximizing: bool, 
                      alpha: float = float('-inf'), 
                      beta: float = float('inf')) -> float:
        """
        GPU批量Minimax搜索
        """
        if depth == 0:
            # 叶节点：批量评估
            with torch.no_grad():
                _, value = self._prediction(state)
                return value.item()
        
        # 获取Top-K候选（GPU上）
        candidate_indices, policy_probs = self.get_candidate_moves_gpu(state)
        
        # 批量构建候选着法的one-hot
        candidate_actions = self.action_templates[candidate_indices]  # (top_k, 1, H, W)
        
        # 批量dynamics推理（一次性评估所有候选）
        with torch.no_grad():
            next_states, rewards = self._batch_dynamics(state, candidate_actions)
        
        # 逐个评估（因为alpha-beta剪枝需要顺序执行）
        if is_maximizing:
            max_eval = float('-inf')
            for i in range(min(self.top_k, candidate_indices.shape[0])):
                eval_score = self.minimax_batch(next_states[i].unsqueeze(0), depth - 1, False, alpha, beta)
                eval_score += rewards[i].item()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if self.use_alpha_beta and beta <= alpha:
                    break
            
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(min(self.top_k, candidate_indices.shape[0])):
                eval_score = self.minimax_batch(next_states[i].unsqueeze(0), depth - 1, True, alpha, beta)
                eval_score -= rewards[i].item()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if self.use_alpha_beta and beta <= alpha:
                    break
            
            return min_eval
    
    def search(self, board: np.ndarray, current_player: int) -> Tuple[int, float]:
        """
        执行搜索，返回最佳着法
        """
        with torch.no_grad():
            # 获取初始状态
            tensor = self.board_to_tensor(board, current_player)
            state, initial_policy, initial_value = self._forward(tensor)
            
            # 获取Top-K候选（GPU上）
            candidate_indices, policy_probs = self.get_candidate_moves_gpu(state)
            
            # 批量构建候选着法的one-hot
            candidate_actions = self.action_templates[candidate_indices]
            
            # 批量dynamics推理（一次性评估所有候选）
            next_states, rewards = self._batch_dynamics(state, candidate_actions)
            
            # 逐个搜索（alpha-beta剪枝需要顺序执行）
            best_move = None
            best_score = float('-inf')
            
            for i in range(min(self.top_k, candidate_indices.shape[0])):
                move = candidate_indices[i].item()
                
                # Minimax搜索（确保4D输入）
                score = self.minimax_batch(next_states[i].unsqueeze(0), self.search_depth - 1, False)
                score += rewards[i].item()
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            return best_move, best_score
    
    def get_move_probabilities(self, board: np.ndarray, current_player: int) -> np.ndarray:
        """获取所有着法的概率分布（用于训练）"""
        with torch.no_grad():
            tensor = self.board_to_tensor(board, current_player)
            state, policy, value = self._forward(tensor)
            
            # 获取策略概率
            policy_probs = F.softmax(policy, dim=1).cpu().numpy()[0]
            
            # 归一化
            total = policy_probs.sum()
            if total > 0:
                policy_probs /= total
            else:
                policy_probs = np.ones(self.action_size) / self.action_size
            
            return policy_probs
    
    def get_candidate_moves(self, board: np.ndarray, current_player: int) -> List[Tuple[int, float]]:
        """使用策略网络获取候选着法（兼容旧接口）"""
        with torch.no_grad():
            tensor = self.board_to_tensor(board, current_player)
            state, policy, value = self._forward(tensor)
            
            # 获取策略概率
            policy_probs = F.softmax(policy, dim=1).cpu().numpy()[0]
            
            # 获取合法着法
            legal_moves = self.get_legal_moves(board)
            
            # 按概率排序
            move_probs = [(move, policy_probs[move]) for move in legal_moves]
            move_probs.sort(key=lambda x: x[1], reverse=True)
            
            return move_probs[:self.top_k]
