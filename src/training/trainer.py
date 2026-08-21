import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from collections import deque
import random
from ..networks.resnet import MuZeroNet
from ..search.minimax import MinimaxSearch


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
                    # 简化：不考虑禁着点和自杀
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
        
        # 简化：不实现提子和劫争逻辑
        # 实际实现需要检查气、提子、劫争等
        
        # 切换玩家
        self.current_player = -self.current_player
        
        return True
    
    def is_game_over(self) -> bool:
        """检查游戏是否结束"""
        # 简化：当棋盘满或达到最大步数时结束
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
        
        # 简化：计算领地（实际需要实现领地计算）
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
        # 19通道表示
        tensor = np.zeros((19, self.board_size, self.board_size), dtype=np.float32)
        
        # 当前玩家棋子（简化：只用当前状态）
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


class Trainer:
    """训练器"""
    
    def __init__(self, 
                 model: MuZeroNet,
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
        初始化训练器
        
        Args:
            model: MuZero网络
            board_size: 棋盘大小
            search_depth: 推理时搜索深度
            search_depth_self_play: 自我对弈时搜索深度（应较小以加快速度）
            top_k: 候选着法数量
            lr: 学习率
            weight_decay: 权重衰减
            batch_size: 批量大小
            buffer_size: 缓冲区大小
            device: 设备
            use_amp: 是否使用自动混合精度
        """
        self.model = model.to(device)
        self.device = device
        self.board_size = board_size
        self.batch_size = batch_size
        self.use_amp = use_amp
        
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
        
        # 推理用搜索算法（深搜索）
        self.search = MinimaxSearch(
            model=model,
            board_size=board_size,
            search_depth=search_depth,
            top_k=top_k,
            device=device,
            use_amp=use_amp
        )
        
        # 自我对弈用搜索算法（浅搜索，加速训练）
        self.search_self_play = MinimaxSearch(
            model=model,
            board_size=board_size,
            search_depth=search_depth_self_play,
            top_k=top_k,
            device=device,
            use_amp=use_amp
        )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 损失函数
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.reward_loss_fn = nn.MSELoss()
        
        # 梯度裁剪
        self.max_grad_norm = 1.0
        
        # 混合精度训练（GPU）
        self.scaler = torch.cuda.amp.GradScaler() if (device == 'cuda' and use_amp) else None
    
    def self_play(self, num_games: int = 100) -> List[Dict]:
        """
        自我对弈生成数据
        
        Args:
            num_games: 对弈局数
            
        Returns:
            游戏记录列表
        """
        import time
        games_data = []
        total_start = time.time()
        
        for game_idx in range(num_games):
            game_start = time.time()
            game = GoGame(self.board_size)
            game.reset()
            
            game_states = []
            game_actions = []
            game_policies = []
            game_rewards = []
            
            step = 0
            max_steps = self.board_size * self.board_size
            
            while not game.is_game_over() and step < max_steps:
                # 获取当前状态
                state_tensor = game.get_state_tensor()
                
                # 使用浅搜索获取着法（只调用一次）
                move, score = self.search_self_play.search(game.board, game.current_player)
                policy = self.search_self_play.get_move_probabilities(game.board, game.current_player)
                
                # 存储数据
                game_states.append(state_tensor)
                game_actions.append(move)
                game_policies.append(policy)
                
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
                    next_state = game_states[i]  # 游戏结束时
                
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
        
        # 采样批次
        states, actions, rewards, next_states, dones, target_policies = \
            self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        target_policies_tensor = torch.FloatTensor(target_policies).to(self.device)
        
        # 混合精度训练
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # 前向传播
                state, initial_policy, initial_value = self.model.initial_inference(states_tensor)
                
                # 动态模型预测（使用隐藏状态，而非原始输入）
                action_one_hot = torch.zeros(states_tensor.shape[0], 1, self.board_size, self.board_size, device=self.device)
                action_one_hot.view(-1, 81).scatter_(1, actions_tensor.unsqueeze(1), 1)
                
                next_state_pred, reward_pred = self.model.dynamics(state, action_one_hot)
                next_policy_pred, next_value_pred = self.model.prediction(next_state_pred)
                
                # 目标隐藏状态（将next_states_tensor通过表示网络）
                target_state, _, _ = self.model.initial_inference(next_states_tensor)
                
                # 计算损失
                policy_loss = self.policy_loss_fn(initial_policy, target_policies_tensor)
                value_loss = self.value_loss_fn(initial_value.squeeze(), rewards_tensor)
                reward_loss = self.reward_loss_fn(reward_pred.squeeze(), rewards_tensor)
                
                # 动态模型损失（状态预测）
                state_loss = F.mse_loss(next_state_pred, target_state)
                
                # 总损失
                total_loss = policy_loss + value_loss + 0.5 * reward_loss + 0.5 * state_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准训练
            state, initial_policy, initial_value = self.model.initial_inference(states_tensor)
            
            action_one_hot = torch.zeros(states_tensor.shape[0], 1, self.board_size, self.board_size, device=self.device)
            action_one_hot.view(-1, 81).scatter_(1, actions_tensor.unsqueeze(1), 1)
            
            next_state_pred, reward_pred = self.model.dynamics(state, action_one_hot)
            next_policy_pred, next_value_pred = self.model.prediction(next_state_pred)
            
            # 目标隐藏状态（将next_states_tensor通过表示网络）
            target_state, _, _ = self.model.initial_inference(next_states_tensor)
            
            # 计算损失
            policy_loss = self.policy_loss_fn(initial_policy, target_policies_tensor)
            value_loss = self.value_loss_fn(initial_value.squeeze(), rewards_tensor)
            reward_loss = self.reward_loss_fn(reward_pred.squeeze(), rewards_tensor)
            state_loss = F.mse_loss(next_state_pred, target_state)
            
            total_loss = policy_loss + value_loss + 0.5 * reward_loss + 0.5 * state_loss
            
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
            'reward_loss': reward_loss.item(),
            'state_loss': state_loss.item(),
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
        import time
        total_games = 0
        total_steps = 0
        total_start = time.time()
        
        while total_games < num_games:
            # 自我对弈
            remaining_games = min(games_per_batch, num_games - total_games)
            print(f"\n{'='*50}")
            print(f"Self-play phase: {remaining_games} games (搜索深度={self.search_self_play.search_depth})")
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
