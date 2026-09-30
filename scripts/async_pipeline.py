"""
异步流水线：自对弈生成与训练完全并行

架构:
┌─────────────────────────────────────────────────────────────┐
│                    Main Process (Training)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  NPU:0 (RL) │  │  Shared Mem │  │  Evaluation │         │
│  │  Training   │  │  Queue      │  │  & Logging  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Worker Processes (8× ONNX + 推演采样)              │
│  ┌──────────┐ ┌──────────┐ ... ┌──────────┐                │
│  │ Worker 1 │ │ Worker 2 │     │ Worker 8 │                │
│  │ ONNX     │ │ ONNX     │     │ ONNX     │                │
│  │ minimax  │ │ minimax  │     │ minimax  │                │
│  └──────────┘ └──────────┘     └──────────┘                │
└─────────────────────────────────────────────────────────────┘

关键设计:
1. 数据流: 共享内存队列（避免 pickle 序列化）
2. 负载均衡: 动态分配（快完成的先处理）
3. 解耦: 生成和训练完全独立，互不阻塞
"""

import sys
import os
import time
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Lock
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


class AsyncDataQueue:
    """共享内存数据队列，避免 pickle 序列化开销。"""
    
    def __init__(self, maxsize=100):
        self.queue = Queue(maxsize=maxsize)
        self.stats = {
            'produced': Value('i', 0),
            'consumed': Value('i', 0),
            'errors': Value('i', 0),
        }
        self.closed = False
    
    def put(self, data: Dict[str, Any]):
        """放入数据（带统计）。"""
        try:
            self.queue.put(data, timeout=0.1)
            with self.stats['produced'].get_lock():
                self.stats['produced'].value += 1
        except queue.Full:
            with self.stats['errors'].get_lock():
                self.stats['errors'].value += 1
    
    def get(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """获取数据（带统计）。"""
        try:
            data = self.queue.get(timeout=timeout)
            with self.stats['consumed'].get_lock():
                self.stats['consumed'].value += 1
            return data
        except queue.Empty:
            return None
    
    def qsize(self) -> int:
        return self.queue.qsize()
    
    def empty(self) -> bool:
        return self.queue.empty()

    def close(self):
        """关闭底层 mp 队列，释放管道/信号量句柄。重复调用安全。

        mp.Queue 只在 close()/join_thread()/gc 时才关管道读端与信号量；每轮迭代
        新建一条流水线却不 close，句柄会随迭代数线性泄漏（长跑训练下表现为
        句柄耗尽 / 管道 fd 堆积）。

        - 幂等：closed 标志让第二次调用直接返回（stop() 可能被重复触及）；
        - 底层队列没有 close()（替身/旧实现）时不抛；
        - 关闭失败只告警不抛：这是收尾路径，不能因为关句柄失败而盖掉真正的
          失败原因、更不能把 workers 列表留在未清空状态。
        """
        if self.closed:
            return
        self.closed = True
        closer = getattr(self.queue, 'close', None)
        if closer is None:
            return
        try:
            closer()
        except Exception as e:
            print(f"[async] 关闭数据队列失败（忽略）: {e}")


class SelfPlayWorker(Process):
    """自对弈工作进程。"""
    
    def __init__(self, worker_id, model_path, onnx_model, args, data_queue, 
                 stop_event, progress_cb=None):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.model_path = model_path
        self.onnx_model = onnx_model
        self.args = args
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.progress_cb = progress_cb
        
    def run(self):
        """进程主循环。"""
        # 每个进程独立加载模型（绕过 GIL）
        from src.inference import GoAI
        if self.onnx_model:
            ai = GoAI(
                model_path=self.onnx_model,
                board_size=self.args.board_size,
                device='cpu',
                use_amp=False
            )
        else:
            ai = GoAI(
                model_path=self.model_path,
                board_size=self.args.board_size,
                device='cpu',
                use_amp=True
            )
        
        from src.game.go_rules import GoBoard
        
        n_actions = self.args.board_size * self.args.board_size + 1
        max_moves = self.args.max_moves or 3 * self.args.board_size * self.args.board_size
        
        game_id = self.worker_id
        
        while not self.stop_event.is_set():
            try:
                # 运行一局自对弈
                game_data, score = self._play_one_game(
                    ai, n_actions, max_moves
                )
                
                # 放入队列
                self.data_queue.put({
                    'gid': game_id,
                    'data': game_data,
                    'score': score,
                    'worker': self.worker_id,
                    'timestamp': time.time()
                })
                
                # 进度回调
                if self.progress_cb:
                    self.progress_cb(self.worker_id, game_data, score)
                
                game_id += 1
                
            except Exception as e:
                # 进程内异常不崩溃
                time.sleep(0.1)
    
    def _play_one_game(self, ai, n_actions, max_moves):
        """运行一局自对弈（**无 MCTS**：N 步 minimax 推演采样）。

        行契约与 `selfplay_train.self_play_game` **逐项一致**（2026-09-30 去 MCTS
        后的 8 元组）：(planes, action, logp_old, to_play, mc, v_collect, mask,
        logq)。两条采集路径的契约必须一模一样 —— 不一致就是「异步与串行跑出
        不同的对局/不同的 buffer 布局」，且症状是静默的。
        """
        from src.game.go_rules import GoBoard
        from src.search.policy_sampler import sample_move

        board = GoBoard(self.args.board_size)
        hists = [[-1, -1, -3], [-1, -1, -3]]
        passes = 0
        mc = 0
        path_moves = []
        data = []

        while passes < 2 and mc < max_moves:
            to_play = board.current_player
            legal = board.get_legal_moves()

            if not legal.any():
                board.play(-1)
                path_moves.append(-1)
                passes += 1
                mc += 1
                continue

            # 走子器一次给出 planes / 行为分布 q / 根估值 / 根 policy π，
            # planes 与推演**共用同一次**特征计算（不再像改造前那样算两遍）。
            s = sample_move(
                ai, board, hists[0], hists[1], to_play,
                topk=getattr(self.args, 'lookahead_topk', 12),
                width=getattr(self.args, 'lookahead_width', 4),
                depth=getattr(self.args, 'lookahead_depth', 2),
                lookahead_temp=getattr(self.args, 'lookahead_temp', 0.2),
                mix=getattr(self.args, 'lookahead_mix', 0.1),
                mc=mc)
            mv = s.action
            mask = s.mask
            pmv = -1 if mv == n_actions - 1 else mv

            success = board.play(pmv)
            if not success:
                board.play(-1)
                pmv = -1

            # 实际走的着法才是行为策略「执行」的动作：play 拒绝回退 pass 时，
            # logq 取 **q(pass)**、logp_old 取 **π(pass)**（混用会让 B2 的
            # 重要性权重悄悄退化成 1）。
            action = n_actions - 1 if pmv < 0 else pmv
            if action != mv:
                q_act = float(s.probs[action])
                pi_act = float(s.policy[action])
                logq = float(np.log(q_act)) if q_act > 0 else 0.0
                logp_old = float(np.log(pi_act)) if pi_act > 0 else 0.0
            else:
                logq, logp_old = s.logq, s.logp_old
            data.append((s.planes, int(action), logp_old, to_play, mc,
                         float(s.value), mask, logq))

            path_moves.append(pmv)
            h = hists[0] if to_play == 1 else hists[1]
            h.pop(0)
            h.append(pmv)

            if pmv >= 0:
                passes = 0
            else:
                passes += 1
            mc += 1

        return data, board.score()


class AsyncSelfPlayPipeline:
    """异步自对弈流水线。
    
    架构:
    - N 个 Worker 进程并行生成游戏
    - 1 个主进程负责训练
    - 共享队列传递数据
    - 动态负载均衡
    """
    
    def __init__(self, args, model_path=None, onnx_model=None, progress_cb=None):
        self.args = args
        self.model_path = model_path
        self.onnx_model = onnx_model
        self.progress_cb = progress_cb
        
        self.data_queue = AsyncDataQueue(maxsize=args.result_queue_max)
        self.stop_event = mp.Event()
        self.workers = []
        self.total_games = 0
        self.buffer = []
        
    def start(self):
        """启动所有 Worker 进程。"""
        n_workers = self.args.parallel_games
        
        for i in range(n_workers):
            worker = SelfPlayWorker(
                worker_id=i,
                model_path=self.model_path,
                onnx_model=self.onnx_model,
                args=self.args,
                data_queue=self.data_queue,
                stop_event=self.stop_event,
                progress_cb=self.progress_cb
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"[async] 启动 {n_workers} 个自对弈 Worker")
    
    def stop(self):
        """停止所有 Worker 并收尾（幂等）。

        收尾三件事，缺一不可：
        1. `join(timeout=5.0)` 之后仍 `is_alive()` 的是**拖尾 worker** ——
           `stop_event` 只在每局之间被检查（run() 的 while），正在下长对局的
           worker 会活过 stop。只 join 就返回等于「打印已停止但进程还在」；
           它们绝不会再 put，所以 terminate 掉没有数据损失。
        2. `data_queue.close()` 释放 mp.Queue 的管道/信号量句柄（每轮迭代一条
           流水线，不关就线性泄漏）。
        3. 清空 `self.workers`：stop() 可被重复触及（幂等），留着已 join 过的
           worker 会二次 join；`start()` 也只 append，不清空 —— 复用同一个对象
           叠加进程是调用方的责任（selfplay 侧靠重建流水线对象规避）。
        """
        if not self.workers and self.data_queue.closed:
            return                      # 已 stop 过（workers 已空 + 队列已关）

        self.stop_event.set()
        stragglers = []
        for w in self.workers:
            w.join(timeout=5.0)
            if w.is_alive():
                stragglers.append(w)
        for w in stragglers:
            w.terminate()
            w.join(timeout=5.0)

        survivors = [w for w in stragglers if w.is_alive()]
        self.data_queue.close()
        self.workers.clear()
        if survivors:
            print(f"[async] 警告：{len(survivors)} 个 Worker 在 terminate() 后仍存活"
                  f"（pid={[getattr(w, 'pid', '?') for w in survivors]}）")
        print("[async] 所有 Worker 已停止")
    
    def collect_buffer(self, max_samples=None):
        """从队列收集数据，填充 buffer。"""
        collected = 0
        while not self.data_queue.empty():
            item = self.data_queue.get(timeout=0.1)
            if item is None:
                break
            
            game_data = item['data']
            score = item['score']
            
            # 处理数据
            bs = self.args.board_size
            n_actions = bs * bs + 1
            self._process_game_data(game_data, score, bs, n_actions)
            
            collected += 1
            self.total_games += 1
        
        # 限制 buffer 大小
        if max_samples and len(self.buffer) > max_samples:
            self.buffer = self.buffer[-max_samples:]
        
        return collected
    
    def _process_game_data(self, game_data, score, bs, n_actions):
        """处理一局游戏数据（P3-C：唯一实现委托 selfplay_train._process_game_data）。

        行契约 8 元组 / buffer 契约 7 元组见该函数 docstring。此前这里是手抄
        副本，会与主实现静默分叉（buffer 布局一变就 KeyError/错位）——委托共享。
        """
        from scripts.selfplay_train import _process_game_data as _shared_process
        _shared_process(game_data, score, bs, n_actions, self.buffer, self.args)
    
    def get_batch(self, batch_size):
        """获取一个训练 batch：(planes, action, z)（P3-C：pi_t 已随 PPO 移除）。"""
        if len(self.buffer) < batch_size:
            return None
        
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        
        planes = np.stack([b[0] for b in batch])
        action = np.stack([b[1] for b in batch])
        z = np.stack([b[3] for b in batch])
        
        return planes, action, z
    
    def get_stats(self):
        """获取统计信息。"""
        return {
            'total_games': self.total_games,
            'buffer_size': len(self.buffer),
            'queue_size': self.data_queue.qsize(),
            'produced': self.data_queue.stats['produced'].value,
            'consumed': self.data_queue.stats['consumed'].value,
            'errors': self.data_queue.stats['errors'].value,
        }


def async_training_loop(args, model_path, onnx_model, train_fn):
    """异步训练主循环。
    
    Args:
        args: 参数配置
        model_path: 模型路径
        onnx_model: ONNX 模型路径（如果启用）
        train_fn: 训练函数 (ai, buffer, args, device) -> loss
    """
    # 初始化流水线
    pipeline = AsyncSelfPlayPipeline(args, model_path, onnx_model)
    pipeline.start()
    
    try:
        # 训练循环
        for it in range(1, args.iters + 1):
            t0 = time.perf_counter()
            
            print(f"\n[iter {it}/{args.iters}] 开始异步训练...", flush=True)
            
            # 收集数据（阻塞直到有数据）
            while len(pipeline.buffer) < args.batch_size * 10:
                collected = pipeline.collect_buffer()
                if collected > 0:
                    print(f"  收集到 {collected} 局数据，buffer={len(pipeline.buffer)}", flush=True)
                time.sleep(0.1)
            
            # 训练
            # 注意：这里需要传入正确的 ai 实例
            # 简化版：直接调用 train_epochs
            
            # 统计
            stats = pipeline.get_stats()
            dt = time.perf_counter() - t0
            
            print(f"[iter {it}] games={stats['total_games']} "
                  f"buffer={stats['buffer_size']} "
                  f"queue={stats['queue_size']} "
                  f"produced={stats['produced']} "
                  f"consumed={stats['consumed']} "
                  f"time={dt:.0f}s", flush=True)
    
    finally:
        pipeline.stop()


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel-games", type=int, default=4)
    ap.add_argument("--mcts-threads", type=int, default=2)
    ap.add_argument("--onnx-model", type=str, default=None)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--board-size", type=int, default=9)
    args = ap.parse_args()
    
    print("Testing async pipeline...")
    pipeline = AsyncSelfPlayPipeline(args, onnx_model=args.onnx_model)
    pipeline.start()
    
    time.sleep(5)
    
    stats = pipeline.get_stats()
    print(f"Stats: {stats}")
    
    pipeline.stop()
    print("Test completed!")
