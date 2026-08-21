import numpy as np
import torch
from typing import List, Tuple
import os


def board_to_string(board: np.ndarray) -> str:
    """将棋盘转换为字符串"""
    board_size = board.shape[0]
    lines = []
    
    # 添加列号
    lines.append("  " + " ".join(str(i) for i in range(board_size)))
    
    # 添加棋盘内容
    for i in range(board_size):
        row = f"{i} "
        for j in range(board_size):
            if board[i, j] == 1:
                row += "● "
            elif board[i, j] == -1:
                row += "○ "
            else:
                row += ". "
        lines.append(row)
    
    return "\n".join(lines)


def print_board(board: np.ndarray, current_player: int = 1):
    """打印棋盘"""
    print(board_to_string(board))
    print(f"当前玩家: {'黑棋' if current_player == 1 else '白棋'}")


def save_game_record(filepath: str, moves: List[int], result: int):
    """
    保存棋谱
    
    Args:
        filepath: 文件路径
        moves: 着法列表
        result: 游戏结果 (1: 黑胜, -1: 白胜, 0: 平局)
    """
    np.savez(filepath, moves=moves, result=result)
    print(f"Game record saved to {filepath}")


def load_game_record(filepath: str) -> Tuple[List[int], int]:
    """
    加载棋谱
    
    Args:
        filepath: 文件路径
        
    Returns:
        (着法列表, 游戏结果)
    """
    data = np.load(filepath)
    moves = data['moves'].tolist()
    result = int(data['result'])
    return moves, result


def move_to_position(move: int, board_size: int = 9) -> Tuple[int, int]:
    """将着法转换为坐标"""
    row = move // board_size
    col = move % board_size
    return row, col


def position_to_move(row: int, col: int, board_size: int = 9) -> int:
    """将坐标转换为着法"""
    return row * board_size + col


def count_liberties(board: np.ndarray, position: Tuple[int, int]) -> int:
    """
    计算棋子的气（简化版本）
    
    Args:
        board: 棋盘状态
        position: 棋子位置 (行, 列)
        
    Returns:
        气数
    """
    row, col = position
    board_size = board.shape[0]
    color = board[row, col]
    
    if color == 0:
        return 0
    
    liberties = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < board_size and 0 <= new_col < board_size:
            if board[new_row, new_col] == 0:
                liberties += 1
    
    return liberties


def calculate_score(board: np.ndarray) -> Tuple[int, int]:
    """
    计算得分（简化版本）
    
    Args:
        board: 棋盘状态
        
    Returns:
        (黑棋得分, 白棋得分)
    """
    black_score = np.sum(board == 1)
    white_score = np.sum(board == -1)
    
    return black_score, white_score


def is_valid_move(board: np.ndarray, move: int, current_player: int) -> bool:
    """
    检查着法是否有效（简化版本）
    
    Args:
        board: 棋盘状态
        move: 着法位置
        current_player: 当前玩家
        
    Returns:
        是否有效
    """
    board_size = board.shape[0]
    row, col = move_to_position(move, board_size)
    
    # 检查位置是否为空
    if board[row, col] != 0:
        return False
    
    # 检查是否在棋盘内
    if row < 0 or row >= board_size or col < 0 or col >= board_size:
        return False
    
    return True


def augment_data(state: np.ndarray, policy: np.ndarray, board_size: int = 9):
    """
    数据增强（旋转和翻转）
    
    Args:
        state: 状态
        policy: 策略
        board_size: 棋盘大小
        
    Returns:
        增强后的数据列表
    """
    augmented_states = []
    augmented_policies = []
    
    # 原始数据
    augmented_states.append(state)
    augmented_policies.append(policy)
    
    # 旋转90度
    state_90 = np.rot90(state, axes=(1, 2))
    policy_90 = np.rot90(policy.reshape(board_size, board_size)).flatten()
    augmented_states.append(state_90)
    augmented_policies.append(policy_90)
    
    # 旋转180度
    state_180 = np.rot90(state, k=2, axes=(1, 2))
    policy_180 = np.rot90(policy.reshape(board_size, board_size), k=2).flatten()
    augmented_states.append(state_180)
    augmented_policies.append(policy_180)
    
    # 旋转270度
    state_270 = np.rot90(state, k=3, axes=(1, 2))
    policy_270 = np.rot90(policy.reshape(board_size, board_size), k=3).flatten()
    augmented_states.append(state_270)
    augmented_policies.append(policy_270)
    
    # 水平翻转
    state_hflip = np.flip(state, axis=2)
    policy_hflip = np.flip(policy.reshape(board_size, board_size), axis=1).flatten()
    augmented_states.append(state_hflip)
    augmented_policies.append(policy_hflip)
    
    # 垂直翻转
    state_vflip = np.flip(state, axis=1)
    policy_vflip = np.flip(policy.reshape(board_size, board_size), axis=0).flatten()
    augmented_states.append(state_vflip)
    augmented_policies.append(policy_vflip)
    
    # 对角线翻转
    state_diag = np.transpose(state, (0, 2, 1))
    policy_diag = np.transpose(policy.reshape(board_size, board_size)).flatten()
    augmented_states.append(state_diag)
    augmented_policies.append(policy_diag)
    
    # 反对角线翻转
    state_antidiag = np.flip(np.transpose(state, (0, 2, 1)), axis=2)
    policy_antidiag = np.flip(np.transpose(policy.reshape(board_size, board_size)), axis=1).flatten()
    augmented_states.append(state_antidiag)
    augmented_policies.append(policy_antidiag)
    
    return augmented_states, augmented_policies


def set_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = 'auto') -> torch.device:
    """获取设备"""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, filepath, **kwargs):
    """保存检查点"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return {k: v for k, v in checkpoint.items() 
            if k not in ['model_state_dict', 'optimizer_state_dict']}
