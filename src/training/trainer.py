import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from typing import List, Tuple, Dict, Optional
from collections import deque
import random
from ..networks.alphanet import AlphaGoNet


class GoGame:
    """围棋游戏环境"""
    
    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1=黑, -1=白
        self.move_count = 0
        self.ko_position = None  # 劫争位置
    
    def reset(self):
        """重置棋盘"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
        self.ko_position = None
    
    def get_legal_moves(self) -> List[int]:
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
        self.move_count += 1
        
        # 切换玩家
        self.current_player = -self.current_player
        
        return True
    
    def is_game_over(self) -> bool:
        """检查游戏是否结束"""
        if self.move_count >= self.board_size * self.board_size:
            return True
        return False
    
    def get_result(self) -> Optional[int]:
        """
        获取游戏结果
        
        Returns:
            1: 黑胜, -1: 白胜, 0: 平局, None: 未结束
        """
        if not self.is_game_over():
            return None
        
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        
        if black_count > white_count:
            return 1
        elif white_count > black_count:
            return -1
        else:
            return 0
    
    def get_state_tensor(self) -> np.ndarray:
        """获取状态张量（用于训练）"""
        tensor = np.zeros((19, self.board_size, self.board_size), dtype=np.float32)
        
        # 当前玩家棋子
        tensor[0][self.board == self.current_player] = 1
        tensor[8][self.board == -self.current_player] = 1
        
        # 当前玩家标记
        tensor[16] = 1 if self.current_player == 1 else -1
        
        # 合法着法标记
        for move in self.get_legal_moves():
            i, j = move // self.board_size, move % self.board_size
            tensor[17, i, j] = 1
        
        return tensor


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, policy: np.ndarray):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done, policy))
    
    def sample(self, batch_size: int) -> Tuple:
        """采样批次"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, actions, rewards, next_states, dones, policies = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(policies)
        )
    
    def __len__(self):
        return len(self.buffer)


class AlphaGoTrainer:
    """AlphaGo风格的多网络训练器"""
    
    def __init__(self, 
                 model: AlphaGoNet,
                 board_size: int = 9,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 64,
                 buffer_size: int = 1000,
                 device: str = 'cpu',
                 use_amp: bool = False,
                 policy_weight: float = 1.0,
                 value_weight: float = 1.0,
                 fast_weight: float = 0.5,
                 temperature: float = 1.0,
                 top_k: int = 5):
        """
        初始化训练器
        
        Args:
            model: AlphaGoNet网络
            board_size: 棋盘大小
            lr: 学习率
            weight_decay: 权重衰减
            batch_size: 批量大小
            buffer_size: 缓冲区大小
            device: 设备
            use_amp: 是否使用自动混合精度
            policy_weight: 策略损失权重
            value_weight: 价值损失权重
            fast_weight: 快速策略损失权重
            temperature: 温度参数（控制探索）
            top_k: 候选着法数量
        """
        self.model = model.to(device)
        self.device = device
        self.board_size = board_size
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.temperature = temperature
        self.top_k = top_k
        
        # 损失权重
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.fast_weight = fast_weight
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=5000, 
            eta_min=1e-5
        )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 损失函数
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # 梯度裁剪
        self.max_grad_norm = 1.0
        
        # 训练/推理模式
        self.training = True
        
        # 混合精度训练（GPU）
        self.scaler = torch.cuda.amp.GradScaler() if (device == 'cuda' and use_amp) else None
    
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
    
    def select_move(self, policy_logits: torch.Tensor, legal_moves: List[int]) -> Tuple[int, torch.Tensor]:
        """
        根据策略网络输出选择着法
        
        Args:
            policy_logits: 策略网络输出 (batch, action_size)
            legal_moves: 合法着法列表
            
        Returns:
            (选择的着法, 策略概率分布)
        """
        # 应用温度
        probs = F.softmax(policy_logits / self.temperature, dim=-1)
        
        # 屏蔽非法着法
        mask = torch.zeros_like(probs)
        mask[0, legal_moves] = 1
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # 采样或选择概率最高的
        if self.training:
            move = torch.multinomial(probs, 1).item()
        else:
            move = torch.argmax(probs).item()
        
        return move, probs
    
    def self_play(self, num_games: int = 100) -> List[Dict]:
        """
        自我对弈生成数据
        
        Args:
            num_games: 对弈局数
            
        Returns:
            游戏记录列表
        """
        games_data = []
        total_start = time.time()
        
        self.model.eval()
        with torch.no_grad():
            for game_idx in range(num_games):
                game_start = time.time()
                game = GoGame(self.board_size)
                game.reset()
                
                game_states = []
                game_actions = []
                game_policies = []
                
                step = 0
                max_steps = self.board_size * self.board_size
                
                while not game.is_game_over() and step < max_steps:
                    # 获取当前状态
                    state_tensor = self.board_to_tensor(game.board, game.current_player)
                    
                    # 获取策略网络输出
                    policy_logits, value, _ = self.model(state_tensor)
                    
                    # 选择着法
                    legal_moves = game.get_legal_moves()
                    move, policy_probs = self.select_move(policy_logits, legal_moves)
                    
                    # 存储数据
                    game_states.append(state_tensor.cpu().numpy()[0])
                    game_actions.append(move)
                    game_policies.append(policy_probs.cpu().numpy()[0])
                    
                    # 执行着法
                    game.make_move(move)
                    step += 1
                
                # 计算游戏结果
                result = game.get_result()
                
                # 为每个步骤分配奖励
                for i in range(len(game_states)):
                    # 简化：最后一步获得完整奖励，其他步骤为0
                    if i == len(game_states) - 1:
                        reward = float(result) if result is not None else 0.0
                    else:
                        reward = 0.0
                    
                    # 获取下一个状态
                    if i < len(game_states) - 1:
                        next_state = game_states[i + 1]
                    else:
                        next_state = game_states[i]
                    
                    done = (i == len(game_states) - 1)
                    
                    # 存储到经验回放
                    self.replay_buffer.push(
                        game_states[i], 
                        game_actions[i], 
                        reward, 
                        next_state, 
                        done,
                        game_policies[i]
                    )
                
                game_time = time.time() - game_start
                games_data.append({
                    'states': game_states,
                    'actions': game_actions,
                    'policies': game_policies,
                    'result': result
                })
                
                # 每局输出进度
                result_str = {1: '黑胜', -1: '白胜', 0: '平局'}.get(result, '未知')
                print(f"  Game {game_idx + 1}/{num_games}: {step}步, 结果={result_str}, 耗时={game_time:.1f}s, 缓冲区={len(self.replay_buffer)}")
        
        total_time = time.time() - total_start
        print(f"Self-play完成: {num_games}局, 总耗时={total_time:.1f}s, 平均={total_time/num_games:.1f}s/局")
        
        return games_data
    
    def train_step(self) -> Dict[str, float]:
        """
        执行一步训练
        
        Returns:
            损失字典
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        self.model.train()
        
        # 采样批次
        states, actions, rewards, next_states, dones, target_policies = \
            self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        target_policies_tensor = torch.FloatTensor(target_policies).to(self.device)
        
        # 混合精度训练
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # 前向传播
                policy_logits, value, fast_policy = self.model(states_tensor)
                
                # 计算损失
                # 策略损失：从自我对弈策略学习
                policy_loss = self.policy_loss_fn(policy_logits, target_policies_tensor)
                
                # 价值损失：从游戏结果学习
                value_loss = self.value_loss_fn(value.squeeze(), rewards_tensor)
                
                # 快速策略损失：从策略网络蒸馏
                fast_policy_loss = self.policy_loss_fn(fast_policy, F.softmax(policy_logits.detach(), dim=-1))
                
                # 总损失
                total_loss = (self.policy_weight * policy_loss + 
                             self.value_weight * value_loss + 
                             self.fast_weight * fast_policy_loss)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准训练
            policy_logits, value, fast_policy = self.model(states_tensor)
            
            # 计算损失
            policy_loss = self.policy_loss_fn(policy_logits, target_policies_tensor)
            value_loss = self.value_loss_fn(value.squeeze(), rewards_tensor)
            fast_policy_loss = self.policy_loss_fn(fast_policy, F.softmax(policy_logits.detach(), dim=-1))
            
            total_loss = (self.policy_weight * policy_loss + 
                         self.value_weight * value_loss + 
                         self.fast_weight * fast_policy_loss)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # 更新学习率
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'fast_policy_loss': fast_policy_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def train(self, num_games: int = 5000, games_per_batch: int = 256, 
              save_interval: int = 100, save_path: str = 'model.pth'):
        """
        完整训练流程
        
        Args:
            num_games: 总对弈局数
            games_per_batch: 每批对弈局数
            save_interval: 保存间隔
            save_path: 保存路径
        """
        total_games = 0
        total_steps = 0
        total_start = time.time()
        
        while total_games < num_games:
            # 自我对弈
            remaining_games = min(games_per_batch, num_games - total_games)
            print(f"\n{'='*50}")
            print(f"Self-play phase: {remaining_games} games (无搜索，纯网络推理)")
            print(f"{'='*50}")
            self.self_play(remaining_games)
            total_games += remaining_games
            
            # 训练
            print(f"\n{'='*50}")
            print(f"Training phase")
            print(f"{'='*50}")
            num_train_steps = min(100, len(self.replay_buffer) // self.batch_size)
            
            for step in range(num_train_steps):
                losses = self.train_step()
                total_steps += 1
                
                if step % 10 == 0 and losses:
                    print(f"  Step {total_steps}: "
                          f"Loss={losses['total_loss']:.4f}, "
                          f"Policy={losses['policy_loss']:.4f}, "
                          f"Value={losses['value_loss']:.4f}, "
                          f"Fast={losses['fast_policy_loss']:.4f}, "
                          f"LR={losses['learning_rate']:.6f}")
            
            # 保存模型
            if total_games % save_interval == 0:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'total_games': total_games,
                    'total_steps': total_steps
                }, save_path)
                elapsed = time.time() - total_start
                print(f"\nModel saved: {save_path} (Games: {total_games}, Elapsed: {elapsed:.0f}s)")
        
        total_time = time.time() - total_start
        print(f"\nTraining completed! Games: {total_games}, Steps: {total_steps}, Time: {total_time:.0f}s")


# 兼容旧接口的Trainer类
class Trainer:
    """兼容旧接口的训练器"""
    
    def __init__(self, 
                 model: AlphaGoNet,
                 board_size: int = 9,
                 search_depth: int = 10,
                 search_depth_self_play: int = 3,
                 top_k: int = 5,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 64,
                 buffer_size: int = 1000,
                 device: str = 'cpu',
                 use_amp: bool = False):
        """
        初始化训练器（兼容旧接口）
        """
        self.alpha_trainer = AlphaGoTrainer(
            model=model,
            board_size=board_size,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            buffer_size=buffer_size,
            device=device,
            use_amp=use_amp,
            top_k=top_k
        )
        self.board_size = board_size
        self.model = model
    
    def self_play(self, num_games: int = 100) -> List[Dict]:
        """自我对弈"""
        return self.alpha_trainer.self_play(num_games)
    
    def train_step(self) -> Dict[str, float]:
        """执行一步训练"""
        return self.alpha_trainer.train_step()
    
    def train(self, num_games: int = 5000, games_per_batch: int = 256, 
              save_interval: int = 100, save_path: str = 'model.pth'):
        """完整训练流程"""
        self.alpha_trainer.train(num_games, games_per_batch, save_interval, save_path)
