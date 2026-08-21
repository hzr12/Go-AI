import torch
import numpy as np
from typing import Tuple, Optional, Dict
from src.networks.resnet import MuZeroNet
from src.search.minimax import MinimaxSearch


class GoAI:
    """围棋AI推理引擎"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 board_size: int = 9,
                 search_depth: int = 10,
                 top_k: int = 5,
                 use_alpha_beta: bool = True,
                 device: str = 'cpu',
                 use_amp: bool = False):
        """
        初始化围棋AI
        
        Args:
            model_path: 模型路径（可选）
            board_size: 棋盘大小
            search_depth: 搜索深度
            top_k: 候选着法数量
            use_alpha_beta: 是否使用Alpha-Beta剪枝
            device: 设备 (cpu/cuda)
            use_amp: 是否使用自动混合精度
        """
        self.board_size = board_size
        self.device = device
        self.use_amp = use_amp
        
        # 创建网络
        self.model = MuZeroNet(
            in_channels=19,
            channels=64,
            num_res_blocks=4,
            action_size=board_size * board_size
        ).to(device)
        
        # 加载模型（如果提供）
        if model_path:
            self.load_model(model_path)
        
        # 创建搜索算法
        self.search = MinimaxSearch(
            model=self.model,
            board_size=board_size,
            search_depth=search_depth,
            top_k=top_k,
            use_alpha_beta=use_alpha_beta,
            device=device,
            use_amp=use_amp
        )
        
        # 棋盘状态
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1=黑, -1=白
        self.move_history = []
    
    def load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path: str):
        """
        保存模型
        
        Args:
            model_path: 模型路径
        """
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
    
    def get_move(self) -> Tuple[int, float, Dict]:
        """
        获取AI着法
        
        Returns:
            (着法, 评估值, 详细信息)
        """
        # 使用搜索获取最佳着法
        move, score = self.search.search(self.board, self.current_player)
        
        # 获取详细信息
        policy = self.search.get_move_probabilities(self.board, self.current_player)
        
        # 获取候选着法
        candidates = self.search.get_candidate_moves(self.board, self.current_player)
        
        info = {
            'score': score,
            'policy': policy,
            'candidates': candidates,
            'board': self.board.copy(),
            'current_player': self.current_player
        }
        
        return move, score, info
    
    def get_move_probabilities(self) -> np.ndarray:
        """获取所有着法的概率分布"""
        return self.search.get_move_probabilities(self.board, self.current_player)
    
    def evaluate_position(self) -> Dict:
        """评估当前局面"""
        policy = self.get_move_probabilities()
        
        # 找出最佳着法
        best_move = np.argmax(policy)
        best_prob = policy[best_move]
        
        # 获取Top-5候选
        top_moves = np.argsort(policy)[-5:][::-1]
        top_probs = policy[top_moves]
        
        return {
            'best_move': best_move,
            'best_prob': best_prob,
            'top_moves': top_moves,
            'top_probs': top_probs,
            'all_probs': policy
        }
    
    def play_against_human(self):
        """人机对弈"""
        print("围棋AI - 人机对弈")
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
                move, score, info = self.get_move()
                row, col = move // self.board_size, move % self.board_size
                print(f"AI着法: {row} {col} (评估值: {score:.4f})")
            
            # 执行着法
            if not self.make_move(move):
                print("非法着法，请重新输入")
                continue
            
            # 检查游戏是否结束
            if len(self.get_legal_moves()) == 0:
                self.print_board()
                print("\n游戏结束！")
                # 计算结果
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
    
    def analyze_move(self, move: int) -> Dict:
        """
        分析某个着法
        
        Args:
            move: 着法位置
            
        Returns:
            分析结果
        """
        # 临时执行着法
        original_board = self.board.copy()
        original_player = self.current_player
        
        self.make_move(move)
        
        # 评估新局面
        eval_result = self.evaluate_position()
        
        # 恢复原状
        self.board = original_board
        self.current_player = original_player
        
        return {
            'move': move,
            'eval_after': eval_result
        }
    
    def suggest_moves(self, num_moves: int = 5) -> list:
        """
        推荐多个着法
        
        Args:
            num_moves: 推荐着法数量
            
        Returns:
            推荐着法列表
        """
        policy = self.get_move_probabilities()
        
        # 按概率排序
        sorted_indices = np.argsort(policy)[::-1]
        
        suggestions = []
        for i in range(min(num_moves, len(sorted_indices))):
            move = sorted_indices[i]
            prob = policy[move]
            row, col = move // self.board_size, move % self.board_size
            
            suggestions.append({
                'move': move,
                'position': (row, col),
                'probability': prob
            })
        
        return suggestions


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='围棋AI推理')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--board-size', type=int, default=9, help='棋盘大小')
    parser.add_argument('--search-depth', type=int, default=10, help='搜索深度')
    parser.add_argument('--top-k', type=int, default=5, help='候选着法数量')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--mode', type=str, default='play', 
                       choices=['play', 'eval', 'analyze'],
                       help='运行模式')
    
    args = parser.parse_args()
    
    # 创建AI
    ai = GoAI(
        model_path=args.model,
        board_size=args.board_size,
        search_depth=args.search_depth,
        top_k=args.top_k,
        device=args.device
    )
    
    if args.mode == 'play':
        # 人机对弈
        ai.play_against_human()
    elif args.mode == 'eval':
        # 评估当前局面
        eval_result = ai.evaluate_position()
        print(f"最佳着法: {eval_result['best_move']}")
        print(f"置信度: {eval_result['best_prob']:.4f}")
        print(f"Top-5候选: {eval_result['top_moves']}")
    elif args.mode == 'analyze':
        # 分析着法（示例：分析位置0）
        analysis = ai.analyze_move(0)
        print(f"着法分析: {analysis}")


if __name__ == '__main__':
    main()
