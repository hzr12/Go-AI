import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from src.networks.alphanet import AlphaGoNet


class GoAI:
    """AlphaGo风格的围棋AI推理引擎"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 board_size: int = 9,
                 device: str = 'cpu',
                 use_amp: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 5,
                 use_value: bool = True):
        """
        初始化围棋AI
        
        Args:
            model_path: 模型路径（可选）
            board_size: 棋盘大小
            device: 设备 (cpu/cuda)
            use_amp: 是否使用自动混合精度
            temperature: 温度参数（控制探索）
            top_k: 候选着法数量
            use_value: 是否使用价值网络评估
        """
        self.board_size = board_size
        self.device = device
        self.use_amp = use_amp
        self.temperature = temperature
        self.top_k = top_k
        self.use_value = use_value
        
        # 创建网络
        self.model = AlphaGoNet(
            in_channels=19,
            backbone_channels=64,
            backbone_res_blocks=4,
            policy_channels=32,
            value_channels=16,
            fast_channels=72,
            fast_res_blocks=3,
            action_size=board_size * board_size
        ).to(device)
        
        # 加载模型（如果提供）
        if model_path:
            self.load_model(model_path)
        
        # 棋盘状态
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1=黑, -1=白
        self.move_history = []
    
    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def reset(self):
        """重置棋盘"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.move_history = []
    
    def get_legal_moves(self) -> list:
        """获取合法着法"""
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    legal_moves.append(i * self.board_size + j)
        return legal_moves
    
    def make_move(self, move: int) -> bool:
        """
        执行着法
        
        Args:
            move: 着法位置 (0-80)
            
        Returns:
            是否成功
        """
        if move not in self.get_legal_moves():
            return False
        
        i, j = move // self.board_size, move % self.board_size
        
        # 放置棋子
        self.board[i, j] = self.current_player
        self.move_history.append(move)
        
        # 切换玩家
        self.current_player = -self.current_player
        
        return True
    
    def board_to_tensor(self) -> torch.Tensor:
        """将棋盘转换为网络输入张量"""
        tensor = torch.zeros(1, 19, self.board_size, self.board_size, device=self.device)
        
        # 当前玩家棋子
        mask_current = torch.tensor(self.board == self.current_player, dtype=torch.bool, device=self.device)
        tensor[0, 0][mask_current] = 1
        # 对手棋子
        mask_opponent = torch.tensor(self.board == -self.current_player, dtype=torch.bool, device=self.device)
        tensor[0, 8][mask_opponent] = 1
        # 当前玩家标记
        tensor[0, 16] = 1 if self.current_player == 1 else -1
        # 合法着法标记
        legal_moves = self.get_legal_moves()
        if legal_moves:
            legal_tensor = torch.tensor(legal_moves, device=self.device)
            rows = legal_tensor // self.board_size
            cols = legal_tensor % self.board_size
            tensor[0, 17, rows, cols] = 1
        
        return tensor
    
    def get_move(self) -> Tuple[int, Dict]:
        """
        获取AI着法
        
        Returns:
            (着法, 详细信息)
        """
        self.model.eval()
        with torch.no_grad():
            # 获取当前状态
            tensor = self.board_to_tensor()
            
            # 获取三个网络的输出
            policy_logits, value, fast_policy = self.model(tensor)
            
            # 获取合法着法
            legal_moves = self.get_legal_moves()
            
            # 策略网络输出
            policy_probs = F.softmax(policy_logits / self.temperature, dim=-1).cpu().numpy()[0]
            
            # 快速策略网络输出
            fast_probs = F.softmax(fast_policy / self.temperature, dim=-1).cpu().numpy()[0]
            
            # 选择着法
            if self.use_value:
                # 使用策略+价值网络结合
                top_k_indices = np.argsort(policy_probs)[-self.top_k:][::-1]
                
                best_score = -float('inf')
                best_move = legal_moves[0] if legal_moves else 0
                
                for move_idx in top_k_indices:
                    if move_idx in legal_moves:
                        # 模拟下一步
                        temp_board = self.board.copy()
                        i, j = move_idx // self.board_size, move_idx % self.board_size
                        temp_board[i, j] = self.current_player
                        
                        # 获取下一步的价值评估
                        next_tensor = self._board_to_tensor(temp_board, -self.current_player)
                        _, next_value, _ = self.model(next_tensor)
                        
                        # 计算分数
                        score = 0.7 * policy_probs[move_idx] + 0.3 * next_value.item()
                        
                        if score > best_score:
                            best_score = score
                            best_move = move_idx
                
                move = best_move
            else:
                # 仅使用策略网络
                # 屏蔽非法着法
                mask = np.zeros_like(policy_probs)
                mask[legal_moves] = 1
                masked_probs = policy_probs * mask
                masked_probs = masked_probs / masked_probs.sum()
                
                move = np.argmax(masked_probs)
            
            # 详细信息
            info = {
                'policy': policy_probs,
                'value': value.item(),
                'fast_policy': fast_probs,
                'board': self.board.copy(),
                'current_player': self.current_player
            }
            
            return move, info
    
    def _board_to_tensor(self, board: np.ndarray, current_player: int) -> torch.Tensor:
        """将指定棋盘状态转换为张量"""
        tensor = torch.zeros(1, 19, self.board_size, self.board_size, device=self.device)
        
        mask_current = torch.tensor(board == current_player, dtype=torch.bool, device=self.device)
        tensor[0, 0][mask_current] = 1
        mask_opponent = torch.tensor(board == -current_player, dtype=torch.bool, device=self.device)
        tensor[0, 8][mask_opponent] = 1
        tensor[0, 16] = 1 if current_player == 1 else -1
        
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    legal_moves.append(i * self.board_size + j)
        if legal_moves:
            legal_tensor = torch.tensor(legal_moves, device=self.device)
            rows = legal_tensor // self.board_size
            cols = legal_tensor % self.board_size
            tensor[0, 17, rows, cols] = 1
        
        return tensor
    
    def get_move_probabilities(self) -> np.ndarray:
        """获取所有着法的概率分布"""
        self.model.eval()
        with torch.no_grad():
            tensor = self.board_to_tensor()
            policy_logits, _, _ = self.model(tensor)
            probs = F.softmax(policy_logits / self.temperature, dim=-1).cpu().numpy()[0]
            return probs
    
    def evaluate_position(self) -> Dict:
        """评估当前局面"""
        self.model.eval()
        with torch.no_grad():
            tensor = self.board_to_tensor()
            policy_logits, value, fast_policy = self.model(tensor)
            
            policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            fast_probs = F.softmax(fast_policy, dim=-1).cpu().numpy()[0]
            
            best_move = np.argmax(policy_probs)
            best_prob = policy_probs[best_move]
            
            return {
                'best_move': best_move,
                'best_prob': best_prob,
                'value': value.item(),
                'policy': policy_probs,
                'fast_policy': fast_probs
            }
    
    def suggest_moves(self, num_moves: int = 5) -> list:
        """推荐多个着法"""
        probs = self.get_move_probabilities()
        sorted_indices = np.argsort(probs)[::-1]
        
        suggestions = []
        for i in range(min(num_moves, len(sorted_indices))):
            move = sorted_indices[i]
            prob = probs[move]
            row, col = move // self.board_size, move % self.board_size
            
            suggestions.append({
                'move': move,
                'position': (row, col),
                'probability': prob
            })
        
        return suggestions
    
    def play_against_human(self):
        """人机对弈"""
        print("AlphaGo围棋AI - 人机对弈")
        print("输入格式: 行号 列号 (0-8)")
        print("输入 'quit' 退出")
        print("输入 'board' 显示棋盘")
        print("输入 'eval' 评估当前局面")
        print()
        
        while True:
            # 显示棋盘
            self.print_board()
            
            if self.current_player == 1:
                # 人类玩家（黑棋）
                user_input = input("\n黑棋着法: ").strip()
                
                if user_input.lower() == 'quit':
                    print("游戏结束")
                    break
                elif user_input.lower() == 'board':
                    continue
                elif user_input.lower() == 'eval':
                    eval_result = self.evaluate_position()
                    print(f"最佳着法: {eval_result['best_move']}")
                    print(f"置信度: {eval_result['best_prob']:.4f}")
                    print(f"价值: {eval_result['value']:.4f}")
                    continue
                
                try:
                    row, col = map(int, user_input.split())
                    move = row * self.board_size + col
                except:
                    print("输入格式错误，请重新输入")
                    continue
            else:
                # AI玩家（白棋）
                print("\nAI思考中...")
                move, info = self.get_move()
                row, col = move // self.board_size, move % self.board_size
                print(f"AI着法: {row} {col}")
                print(f"策略网络置信度: {info['policy'][move]:.4f}")
                print(f"价值评估: {info['value']:.4f}")
            
            # 执行着法
            if not self.make_move(move):
                print("非法着法，请重新输入")
                continue
            
            # 检查游戏是否结束
            if len(self.get_legal_moves()) == 0:
                self.print_board()
                print("\n游戏结束！")
                black_count = np.sum(self.board == 1)
                white_count = np.sum(self.board == -1)
                if black_count > white_count:
                    print("黑棋获胜！")
                elif white_count > black_count:
                    print("白棋获胜！")
                else:
                    print("平局！")
                break
    
    def print_board(self):
        """打印棋盘"""
        print("\n  ", end="")
        for j in range(self.board_size):
            print(f"{j} ", end="")
        print()
        
        for i in range(self.board_size):
            print(f"{i} ", end="")
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    print("● ", end="")
                elif self.board[i, j] == -1:
                    print("○ ", end="")
                else:
                    print(". ", end="")
            print()
        
        print(f"\n当前玩家: {'黑棋' if self.current_player == 1 else '白棋'}")
        print(f"已下{len(self.move_history)}步")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaGo围棋AI推理')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--board-size', type=int, default=9, help='棋盘大小')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--use-amp', action='store_true', help='使用自动混合精度')
    parser.add_argument('--use-value', action='store_true', default=True, help='使用价值网络')
    parser.add_argument('--mode', type=str, default='play', 
                       choices=['play', 'eval', 'analyze'],
                       help='运行模式')
    
    args = parser.parse_args()
    
    # 创建AI
    ai = GoAI(
        model_path=args.model,
        board_size=args.board_size,
        device=args.device,
        use_amp=args.use_amp,
        use_value=args.use_value
    )
    
    if args.mode == 'play':
        # 人机对弈
        ai.play_against_human()
    elif args.mode == 'eval':
        # 评估当前局面
        eval_result = ai.evaluate_position()
        print(f"最佳着法: {eval_result['best_move']}")
        print(f"置信度: {eval_result['best_prob']:.4f}")
        print(f"价值: {eval_result['value']:.4f}")
    elif args.mode == 'analyze':
        # 分析着法
        eval_result = ai.evaluate_position()
        print(f"策略网络Top-5:")
        top5 = np.argsort(eval_result['policy'])[-5:][::-1]
        for i, move in enumerate(top5):
            row, col = move // ai.board_size, move % ai.board_size
            print(f"  {i+1}. {row} {col}: {eval_result['policy'][move]:.4f}")
        
        print(f"\n快速策略Top-5:")
        top5_fast = np.argsort(eval_result['fast_policy'])[-5:][::-1]
        for i, move in enumerate(top5_fast):
            row, col = move // ai.board_size, move % ai.board_size
            print(f"  {i+1}. {row} {col}: {eval_result['fast_policy'][move]:.4f}")


if __name__ == '__main__':
    main()
