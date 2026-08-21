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
from ..data.game_loader import GameLoader, TrainingExample


class GoGame:
    """围棋游戏环境"""
    
    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
        self.ko_position = None
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
        self.ko_position = None
    
    def get_legal_moves(self) -> List[int]:
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    legal_moves.append(i * self.board_size + j)
        return legal_moves
    
    def make_move(self, move: int) -> bool:
        if move not in self.get_legal_moves():
            return False
        i, j = move // self.board_size, move % self.board_size
        self.board[i, j] = self.current_player
        self.move_count += 1
        self.current_player = -self.current_player
        return True
    
    def is_game_over(self) -> bool:
        return self.move_count >= self.board_size * self.board_size
    
    def get_result(self) -> Optional[int]:
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
    
    def get_territory_diff(self, player: int) -> float:
        """计算领地差（当前玩家视角）"""
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        total = self.board_size * self.board_size
        diff = (black_count - white_count) / total
        return diff * player
    
    def get_state_tensor(self) -> np.ndarray:
        tensor = np.zeros((19, self.board_size, self.board_size), dtype=np.float32)
        tensor[0][self.board == self.current_player] = 1
        tensor[8][self.board == -self.current_player] = 1
        tensor[16] = 1 if self.current_player == 1 else -1
        for move in self.get_legal_moves():
            i, j = move // self.board_size, move % self.board_size
            tensor[17, i, j] = 1
        return tensor


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, policy):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, policy))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, policy)
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), 
                                   p=probs, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, policies = zip(*batch)
        
        # 重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(policies),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)


def apply_symmetry(board: np.ndarray, policy: np.ndarray, transform_id: int, board_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """应用对称变换"""
    if transform_id == 0:
        return board, policy
    
    if transform_id == 1:  # 旋转90度
        new_board = np.rot90(board)
        new_policy = policy.reshape(board_size, board_size)
        new_policy = np.rot90(new_policy)
        return new_board, new_policy.flatten()
    
    if transform_id == 2:  # 旋转180度
        new_board = np.rot90(board, 2)
        new_policy = policy.reshape(board_size, board_size)
        new_policy = np.rot90(new_policy, 2)
        return new_board, new_policy.flatten()
    
    if transform_id == 3:  # 旋转270度
        new_board = np.rot90(board, 3)
        new_policy = policy.reshape(board_size, board_size)
        new_policy = np.rot90(new_policy, 3)
        return new_board, new_policy.flatten()
    
    if transform_id == 4:  # 水平翻转
        new_board = np.fliplr(board)
        new_policy = policy.reshape(board_size, board_size)
        new_policy = np.fliplr(new_policy)
        return new_board, new_policy.flatten()
    
    if transform_id == 5:  # 垂直翻转
        new_board = np.flipud(board)
        new_policy = policy.reshape(board_size, board_size)
        new_policy = np.flipud(new_policy)
        return new_board, new_policy.flatten()
    
    if transform_id == 6:  # 转置
        new_board = board.T
        new_policy = policy.reshape(board_size, board_size)
        new_policy = new_policy.T
        return new_board, new_policy.flatten()
    
    if transform_id == 7:  # 反转置
        new_board = board.T
        new_board = np.fliplr(new_board)
        new_board = np.flipud(new_board)
        new_policy = policy.reshape(board_size, board_size)
        new_policy = new_policy.T
        new_policy = np.fliplr(new_policy)
        new_policy = np.flipud(new_policy)
        return new_board, new_policy.flatten()
    
    return board, policy


class AlphaGoTrainer:
    """AlphaGo风格的多网络训练器"""
    
    def __init__(self, 
                 model: AlphaGoNet,
                 board_size: int = 9,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 device: str = 'cpu',
                 use_amp: bool = False,
                 policy_weight: float = 1.0,
                 value_weight: float = 1.0,
                 fast_weight: float = 0.5,
                 temperature: float = 1.0,
                 temperature_min: float = 0.1,
                 temperature_decay: float = 0.9999,
                 top_k: int = 5,
                 n_step: int = 5,
                 gamma: float = 0.99,
                 use_augmentation: bool = True,
                 use_prioritized_replay: bool = True):
        
        self.model = model.to(device)
        self.device = device
        self.board_size = board_size
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.current_temperature = temperature
        self.top_k = top_k
        
        # N-step回报参数
        self.n_step = n_step
        self.gamma = gamma
        
        # 数据增强
        self.use_augmentation = use_augmentation
        
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
        
        # 优先经验回放
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.buffer_size = buffer_size
        self.use_prioritized_replay = use_prioritized_replay
        
        # 损失函数
        self.policy_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.value_loss_fn = nn.MSELoss(reduction='none')
        
        # 梯度裁剪
        self.max_grad_norm = 1.0
        
        # 训练模式
        self.training = True
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if (device == 'cuda' and use_amp) else None
    
    def pretrain_on_games(self, game_records: List, epochs: int = 10, 
                         batch_size: int = 256, augment: bool = True,
                         save_path: str = '') -> Dict[str, float]:
        """
        在棋谱数据上预训练
        
        Args:
            game_records: 棋谱记录列表
            epochs: 训练轮数
            batch_size: 批量大小
            augment: 是否使用数据增强
            
        Returns:
            训练损失字典
        """
        if not game_records:
            print("No game records provided for pretraining")
            return {}
        
        print(f"\n{'='*50}")
        print(f"Pretraining on {len(game_records)} games")
        print(f"{'='*50}")
        
        # 加载棋谱数据
        loader = GameLoader(self.board_size)
        all_training_data = []
        
        for i, game in enumerate(game_records):
            training_data = loader.game_to_training_data(game)
            all_training_data.extend(training_data)
            
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(game_records)} games")
        
        print(f"Total training examples: {len(all_training_data)}")
        
        # 数据增强
        if augment:
            all_training_data = loader.augment_data(all_training_data)
            print(f"After augmentation: {len(all_training_data)} examples")
        
        # 训练
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            random.shuffle(all_training_data)
            
            epoch_loss = 0.0
            epoch_batches = 0
            
            for i in range(0, len(all_training_data), batch_size):
                batch = all_training_data[i:i+batch_size]
                
                # 准备批次数据
                states = np.array([ex.state for ex in batch])
                actions = np.array([ex.action for ex in batch])
                policies = np.array([ex.policy for ex in batch])
                values = np.array([ex.value for ex in batch], dtype=np.float32)
                
                # 转换为张量
                states_tensor = torch.FloatTensor(states).to(self.device)
                actions_tensor = torch.LongTensor(actions).to(self.device)
                policies_tensor = torch.FloatTensor(policies).to(self.device)
                values_tensor = torch.FloatTensor(values).to(self.device)
                
                # 前向传播
                policy_logits, value, fast_policy = self.model(states_tensor)
                
                # 计算损失
                policy_loss = F.cross_entropy(policy_logits, policies_tensor)
                value_loss = F.mse_loss(value.squeeze(), values_tensor)
                fast_policy_loss = F.kl_div(
                    F.log_softmax(fast_policy, dim=-1),
                    F.softmax(policy_logits.detach(), dim=-1),
                    reduction='batchmean'
                )
                
                total_loss_batch = (self.policy_weight * policy_loss + 
                                   self.value_weight * value_loss + 
                                   self.fast_weight * fast_policy_loss)
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                epoch_loss += total_loss_batch.item()
                epoch_batches += 1
            
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            total_loss += epoch_loss
            num_batches += epoch_batches
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")
        
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Pretraining completed: Avg Loss={avg_total_loss:.4f}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'phase': 'pretrain',
                'epochs': epochs,
                'pretrain_loss': avg_total_loss
            }, save_path)
            print(f"Pretrained model saved: {save_path}")
        
        return {
            'pretrain_loss': avg_total_loss,
            'epochs': epochs,
            'total_examples': len(all_training_data)
        }
    
    def board_to_tensor(self, board: np.ndarray, current_player: int) -> torch.Tensor:
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
    
    def select_move(self, policy_logits: torch.Tensor, legal_moves: List[int]) -> Tuple[int, torch.Tensor]:
        probs = F.softmax(policy_logits / self.current_temperature, dim=-1)
        mask = torch.zeros_like(probs)
        mask[0, legal_moves] = 1
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        if self.training:
            move = torch.multinomial(probs, 1).item()
        else:
            move = torch.argmax(probs).item()
        
        return move, probs
    
    def self_play(self, num_games: int = 100) -> List[Dict]:
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
                game_players = []
                game_territory_diffs = []
                
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
                    
                    # 记录当前玩家和领地差
                    current_player = game.current_player
                    territory_diff = game.get_territory_diff(current_player)
                    
                    # 存储数据
                    game_states.append(state_tensor.cpu().numpy()[0])
                    game_actions.append(move)
                    game_policies.append(policy_probs.cpu().numpy()[0])
                    game_players.append(current_player)
                    game_territory_diffs.append(territory_diff)
                    
                    # 执行着法
                    game.make_move(move)
                    step += 1
                
                # 计算游戏结果
                result = game.get_result()
                
                # 计算n-step回报
                rewards = []
                for i in range(len(game_states)):
                    # 密集奖励：每步都给领地差作为即时奖励
                    immediate_reward = game_territory_diffs[i]
                    
                    # 最后一步加上最终结果
                    if i == len(game_states) - 1:
                        final_reward = float(result) if result is not None else 0.0
                        reward = immediate_reward + final_reward
                    else:
                        # N-step回报：累加n步的奖励
                        n_step_reward = 0
                        for j in range(min(self.n_step, len(game_states) - i)):
                            n_step_reward += (self.gamma ** j) * game_territory_diffs[i + j]
                        reward = n_step_reward
                    
                    rewards.append(reward)
                
                # 存储到经验回放（带数据增强）
                for i in range(len(game_states)):
                    # 获取下一个状态
                    if i < len(game_states) - 1:
                        next_state = game_states[i + 1]
                    else:
                        next_state = game_states[i]
                    
                    done = (i == len(game_states) - 1)
                    
                    # 原始数据
                    self.replay_buffer.push(
                        game_states[i], 
                        game_actions[i], 
                        rewards[i], 
                        next_state, 
                        done,
                        game_policies[i]
                    )
                    
                    # 数据增强：8种对称变换
                    if self.use_augmentation:
                        for transform_id in range(1, 8):
                            aug_state, aug_policy = apply_symmetry(
                                game_states[i][:2].transpose(1, 2, 0).reshape(self.board_size, self.board_size, 2),
                                game_policies[i],
                                transform_id,
                                self.board_size
                            )
                            # 重新构建状态张量
                            aug_state_tensor = game_states[i].copy()
                            aug_state, aug_policy = apply_symmetry(
                                game_states[i][0],  # 只变换棋盘
                                game_policies[i],
                                transform_id,
                                self.board_size
                            )
                            self.replay_buffer.push(
                                aug_state_tensor,
                                game_actions[i],
                                rewards[i],
                                next_state,
                                done,
                                aug_policy
                            )
                
                game_time = time.time() - game_start
                games_data.append({
                    'states': game_states,
                    'actions': game_actions,
                    'policies': game_policies,
                    'result': result
                })
                
                if game_idx == num_games - 1:
                    result_str = {1: '黑胜', -1: '白胜', 0: '平局'}.get(result, '未知')
                    print(f"  Last game: {step}步, 结果={result_str}")
        
        total_time = time.time() - total_start
        print(f"Self-play完成: {num_games}局, 总耗时={total_time:.1f}s, 平均={total_time/num_games:.1f}s/局")
        
        return games_data
    
    def train_step(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        self.model.train()
        
        # 采样批次
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, target_policies, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones, target_policies, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights_tensor = torch.ones(len(states), device=self.device)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        target_policies_tensor = torch.FloatTensor(target_policies).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # 混合精度训练
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                policy_logits, value, fast_policy = self.model(states_tensor)
                _, next_value, _ = self.model(next_states_tensor)
                
                # 策略损失（带权重）
                policy_loss_per_sample = self.policy_loss_fn(policy_logits, target_policies_tensor)
                policy_loss = (policy_loss_per_sample * weights_tensor).mean()
                
                # 价值损失（带n-step回报和TD-error）
                target_value = rewards_tensor + (1 - dones_tensor) * self.gamma ** self.n_step * next_value.squeeze()
                value_loss_per_sample = self.value_loss_fn(value.squeeze(), target_value.detach())
                value_loss = (value_loss_per_sample * weights_tensor).mean()
                
                # 快速策略损失
                fast_policy_loss_per_sample = self.policy_loss_fn(fast_policy, F.softmax(policy_logits.detach(), dim=-1))
                fast_policy_loss = (fast_policy_loss_per_sample * weights_tensor).mean()
                
                total_loss = (self.policy_weight * policy_loss + 
                             self.value_weight * value_loss + 
                             self.fast_weight * fast_policy_loss)
            
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, value, fast_policy = self.model(states_tensor)
            _, next_value, _ = self.model(next_states_tensor)
            
            target_value = rewards_tensor + (1 - dones_tensor) * self.gamma ** self.n_step * next_value.squeeze()
            
            policy_loss_per_sample = self.policy_loss_fn(policy_logits, target_policies_tensor)
            policy_loss = (policy_loss_per_sample * weights_tensor).mean()
            
            value_loss_per_sample = self.value_loss_fn(value.squeeze(), target_value.detach())
            value_loss = (value_loss_per_sample * weights_tensor).mean()
            
            fast_policy_loss_per_sample = self.policy_loss_fn(fast_policy, F.softmax(policy_logits.detach(), dim=-1))
            fast_policy_loss = (fast_policy_loss_per_sample * weights_tensor).mean()
            
            total_loss = (self.policy_weight * policy_loss + 
                         self.value_weight * value_loss + 
                         self.fast_weight * fast_policy_loss)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # 更新优先级
        if self.use_prioritized_replay:
            with torch.no_grad():
                td_errors = (value.squeeze() - target_value).abs().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # 衰减温度
        self.current_temperature = max(self.temperature_min, 
                                       self.current_temperature * self.temperature_decay)
        
        # 更新学习率
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'fast_policy_loss': fast_policy_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'temperature': self.current_temperature
        }
    
    def train(self, num_games: int = 5000, games_per_batch: int = 256, 
              save_interval: int = 100, save_path: str = 'model.pth'):
        total_games = 0
        total_steps = 0
        total_start = time.time()
        
        # Phase 1: 填充缓冲区
        buffer_fill_games = min(num_games, self.buffer_size // self.board_size // self.board_size + 1)
        buffer_fill_games = min(buffer_fill_games, num_games)
        print(f"\n{'='*50}")
        print(f"Phase 1: Filling buffer ({buffer_fill_games} games)")
        print(f"{'='*50}")
        self.self_play(buffer_fill_games)
        total_games += buffer_fill_games
        
        # Phase 2: 交替自我对弈和训练
        while total_games < num_games:
            remaining_games = min(games_per_batch, num_games - total_games)
            print(f"\n{'='*50}")
            print(f"Self-play: {remaining_games} games")
            print(f"{'='*50}")
            self.self_play(remaining_games)
            total_games += remaining_games
            
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
                          f"P={losses['policy_loss']:.4f}, "
                          f"V={losses['value_loss']:.4f}, "
                          f"F={losses['fast_policy_loss']:.4f}, "
                          f"T={losses['temperature']:.3f}")
            
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
        return self.alpha_trainer.self_play(num_games)
    
    def train_step(self) -> Dict[str, float]:
        return self.alpha_trainer.train_step()
    
    def train(self, num_games: int = 5000, games_per_batch: int = 256, 
              save_interval: int = 100, save_path: str = 'model.pth'):
        self.alpha_trainer.train(num_games, games_per_batch, save_interval, save_path)
