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
│              Worker Processes (8× ONNX + MCTS)               │
│  ┌──────────┐ ┌──────────┐ ... ┌──────────┐                │
│  │ Worker 1 │ │ Worker 2 │     │ Worker 8 │                │
│  │ ONNX     │ │ ONNX     │     │ ONNX     │                │
│  │ MCTS(3t) │ │ MCTS(3t) │     │ MCTS(3t) │                │
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
import faulthandler
import traceback
import multiprocessing as mp
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


class AsyncDataQueue:
    """共享内存数据队列，避免 pickle 序列化开销。"""
    
    def __init__(self, maxsize=100, ctx=None):
        self.ctx = ctx if ctx is not None else mp.get_context()
        self.queue = self.ctx.Queue(maxsize=maxsize)
        self.stats = {
            'produced': self.ctx.Value('i', 0),
            'consumed': self.ctx.Value('i', 0),
            'errors': self.ctx.Value('i', 0),
        }
    
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


class SelfPlayWorker:
    """自对弈工作进程主体。

    刻意**不**继承 Process：multiprocessing 公开 API 无法向 Process 子类
    注入 start method，而这里必须用 spawn（见 _make_worker）。改为普通类
    + 模块级函数 target，由 ctx.Process(target=...) 启动。
    """
    
    def __init__(self, worker_id, model_path, onnx_model, args, data_queue, 
                 stop_event, progress_cb=None):
        self.worker_id = worker_id
        self.model_path = model_path
        self.onnx_model = onnx_model
        self.args = args
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.progress_cb = progress_cb
        
    def _build_ai(self):
        """构建本 worker 的推理模型（独立加载，绕过 GIL）。"""
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
        self._report_device(ai)
        return ai

    def _report_device(self, ai):
        """一次性打印模型与参数的真实设备。

        云端出现过"CPU 张量却报 CANN 的 matmul primitive"（F.linear 处）。既然
        `import torch` 后 torch_npu 并未自动加载、且同环境下纯 CPU matmul 正常，
        那唯一还没被证实的前提就是"模型其实不在 CPU 上"。这个打印把该前提变成
        可核对的事实，而不是推测。只在必要时打印，不进入热路径。
        """
        try:
            param = next(ai.model.parameters())
            dev = str(param.device)
        except Exception as exc:  # noqa: BLE001
            dev = '取不到({!r})'.format(exc)
        print("[worker {}] 推理设备: 请求=cpu 参数={} torch_npu已加载={} "
              "cann可见设备={!r}".format(
                  self.worker_id, dev,
                  'torch_npu' in sys.modules,
                  os.environ.get('ASCEND_RT_VISIBLE_DEVICES')),
              file=sys.stderr, flush=True)

    def _report_error(self, where, exc):
        """上报失败：写 stderr（flush）并累加共享错误计数。

        共享计数让父进程在 worker 全死时能报出"发生过 N 次错误"，而不是
        只看到一句无从查证的 0/8 存活。
        """
        try:
            with self.data_queue.stats['errors'].get_lock():
                self.data_queue.stats['errors'].value += 1
        except Exception:  # noqa: BLE001
            pass
        print("[worker {}] {} 失败: {!r}".format(self.worker_id, where, exc),
              file=sys.stderr, flush=True)
        traceback.print_exc()

    def run(self):
        """进程主循环。"""
        # 模型构建也必须在保护之内：原先它在 try 之外，抛异常会直接杀掉
        # 子进程，父进程只看到 0/8 存活却完全拿不到原因。
        try:
            ai = self._build_ai()
        except Exception as exc:  # noqa: BLE001
            self._report_error('模型加载', exc)
            return

        from src.search.mcts import MCTS
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
                
            except Exception as exc:  # noqa: BLE001
                # 不崩溃，但**必须上报**：原先 `except Exception: sleep(0.1)`
                # 把异常整个吞掉，worker 永远活着却零产出，表现为"卡住"。
                self._report_error('对局', exc)
                time.sleep(0.1)
    
    def _play_one_game(self, ai, n_actions, max_moves):
        """运行一局自对弈。"""
        from src.search.mcts import MCTS
        from src.game.go_rules import GoBoard
        
        mcts = MCTS(
            ai, 
            board_size=self.args.board_size,
            num_threads=self.args.mcts_threads,
            expand_topk=self.args.expand_topk,
            expand_chunk=self.args.expand_chunk,
            priors_leaf=True,
            temperature=self.args.temperature,
            dirichlet_alpha=self.args.dir_alpha if hasattr(self.args, 'dir_alpha') else 0.3,
            dirichlet_eps=self.args.dir_eps if hasattr(self.args, 'dir_eps') else 0.25,
            spec_prefetch=bool(getattr(self.args, 'spec_prefetch', False)),
            use_rollout=getattr(self.args, 'use_rollout', False),
            rollout_lambda=getattr(self.args, 'rollout_lambda', 0.25),
            rollout_steps=getattr(self.args, 'rollout_steps', None),
            batch_cap=getattr(self.args, 'batch_cap', None),
            leaf_ab_depth=getattr(self.args, 'leaf_ab_depth', 2),
            c_puct=getattr(self.args, 'c_puct', 2.0),
            virtual_loss=getattr(self.args, 'virtual_loss', 8.0),
            dynamic_topk=getattr(self.args, 'dynamic_topk', True),
            dynamic_virtual_loss=getattr(self.args, 'dynamic_virtual_loss', True),
            vector_backup=getattr(self.args, 'mcts_vector_backup', 1) == 1
        )
        
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
            
            visits, probs, root_value = mcts.search(
                board, hists[0], hists[1], to_play,
                simulations=self.args.sims,
                path_moves=path_moves
            )
            
            # 记录样本
            planes = np.ascontiguousarray(board.feature_planes_batched(
                board.board[None], [list(hists[0])], [list(hists[1])],
                [to_play], [board.ko_point])[0])
            
            vt = np.zeros(n_actions)
            vs = visits.sum()
            if vs > 0:
                vt[:n_actions - 1] = visits[:n_actions - 1] / vs
                vt[n_actions - 1] = visits[n_actions - 1] / vs
            
            data.append((planes, vt, to_play, mc, float(root_value)))
            
            # 温度衰减
            progress = min(1.0, mc / max(30, 1))
            temp = 1.0 - progress * (1.0 - 0.1)
            p = np.asarray(probs).reshape(-1).astype(np.float64)
            p[-1] = max(p[-1], 0.0)
            
            if temp > 0 and temp != 1.0:
                p = p ** (1.0 / temp)
            
            s = p.sum()
            mv = n_actions - 1 if s <= 0 else int(np.random.choice(n_actions, p=p / s))
            pmv = -1 if mv == n_actions - 1 else mv
            
            success = board.play(pmv)
            if not success:
                board.play(-1)
                pmv = -1
            
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


def _async_selfplay_worker(worker_id, model_path, onnx_model, args,
                           data_queue, stop_event, progress_cb=None):
    """spawn 可 pickle 的 worker 入口。

    必须是模块级函数：spawn 会 pickle 整个 worker 实例，局部闭包无法序列化。
    """
    # 原生崩溃（SIGSEGV/SIGABRT）不会产生 Python traceback，faulthandler 是
    # 子进程里唯一能拿到栈的途径。
    try:
        faulthandler.enable()
    except Exception:  # noqa: BLE001
        pass
    # torch 默认用 os.cpu_count() 个 intra-op 线程。不钉住的话，N 个 worker
    # 就是 N×核数 个线程抢同样多的核（8 worker × 24 核 = 192 线程抢 24 核），
    # 过度订阅会同时压低吞吐与尾延迟，也让 mcts-threads 的配比失去意义。
    # MCTS 自身已有线程结构（选路径线程 + 消费线程），这里不再叠加。
    try:
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    # 某些 CANN 镜像会用 CANN 算子抢占 aten::linear，导致 CPU 上 nn.Linear
    # 直接抛 "could not create a primitive descriptor for a matmul primitive"
    #（同进程 mm/addmm/matmul/conv2d 均正常）。worker 恒为 CPU，故装等价垫片。
    try:
        from src.inference import install_cpu_linear_workaround
        install_cpu_linear_workaround(device='cpu')
    except Exception:  # noqa: BLE001
        pass
    SelfPlayWorker(worker_id, model_path, onnx_model, args, data_queue,
                   stop_event, progress_cb).run()


_SIGNAL_NAMES = {
    4: 'SIGILL', 6: 'SIGABRT', 8: 'SIGFPE', 9: 'SIGKILL',
    11: 'SIGSEGV', 13: 'SIGPIPE', 15: 'SIGTERM', 24: 'SIGXCPU', 25: 'SIGXFSZ',
}


def _describe_exitcode(code):
    """把 multiprocessing 的 exitcode 翻成人话。

    约定：负值 = 被该信号杀死（-11 → SIGSEGV）。这是"worker 全部退出、
    既无 traceback 也无错误计数"这类现象唯一能定位死因的线索。
    """
    if code is None:
        return '仍在运行'
    if code < 0:
        sig = -code
        return '被信号 {}（{}）杀死'.format(sig, _SIGNAL_NAMES.get(sig, '未知信号'))
    if code == 0:
        return '正常退出(0)'
    return '以退出码 {} 结束'.format(code)


class AsyncSelfPlayPipeline:
    """异步自对弈流水线。
    
    架构:
    - N 个 Worker 进程并行生成游戏
    - 1 个主进程负责训练
    - 共享队列传递数据
    - 动态负载均衡
    """
    
    def __init__(self, args, model_path=None, onnx_model=None, progress_cb=None,
                 start_method='spawn'):
        self.args = args
        self.model_path = model_path
        self.onnx_model = onnx_model
        self.progress_cb = progress_cb

        # 所有跨进程原语与 worker 必须来自**同一个** context。原先这里用
        # 模块级 Queue 与 mp.Event()（绑定默认 context，Linux 上即 fork），
        # 却把对象交给 spawn context 的进程：跨 context 传递同步原语不受支持，
        # 子进程会无声死掉——没有 traceback，连错误计数都还是 0。
        self.ctx = mp.get_context(start_method)
        self.data_queue = AsyncDataQueue(maxsize=args.result_queue_max,
                                         ctx=self.ctx)
        self.stop_event = self.ctx.Event()
        self.workers = []
        self.total_games = 0
        self.buffer = []
        
    def _make_worker(self, worker_id):
        """用 spawn context 创建单个 worker 进程（不启动）。

        不可用裸 Process.start()：Linux 默认 fork，而父进程此前已在 NPU 上
        建模型并 warmup（起了 CANN 线程），fork 出的子进程继承被锁死的
        CANN 上下文 → 表现为 worker 全部存活、CPU 0%、永远 0 产出。
        同步路径早已用 spawn 修过同一问题（selfplay_train.py 内标记 "N2"），
        异步路径此前漏修。
        """
        return self.ctx.Process(
            target=_async_selfplay_worker,
            args=(worker_id, self.model_path, self.onnx_model, self.args,
                  self.data_queue, self.stop_event, self.progress_cb),
            daemon=True,
        )

    def start(self):
        """启动所有 Worker 进程。"""
        n_workers = self.args.parallel_games
        
        for i in range(n_workers):
            worker = self._make_worker(i)
            worker.start()
            self.workers.append(worker)
        
        print(f"[async] 启动 {n_workers} 个自对弈 Worker")
    
    def stop(self):
        """停止所有 Worker。"""
        self.stop_event.set()
        for w in self.workers:
            w.join(timeout=5.0)
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
        """处理一局游戏数据（TD 价值标签 + 8 对称增强，与 selfplay_train 共享逻辑）。"""
        from scripts.selfplay_train import compute_td_target, augment8
        td = getattr(self.args, 'td', 0) == 1
        td_steps = getattr(self.args, 'td_steps', 3)
        td_ai = getattr(self.args, 'td_alpha_init', 0.2)
        td_ae = getattr(self.args, 'td_alpha_end', 0.9)
        players = np.asarray([row[2] for row in game_data])
        root_values = np.asarray([row[4] if len(row) > 4 else 0.0
                                  for row in game_data])
        for mc_idx, row in enumerate(game_data):
            planes, vt = row[0], row[1]
            z, _z_raw, _alpha = compute_td_target(
                players, root_values, score, mc_idx,
                td, td_steps, td_ai, td_ae)
            if getattr(self.args, 'no_augment', False):
                self.buffer.append((planes, vt, z))
            else:
                for pl, tv in augment8(planes, vt, bs):
                    self.buffer.append((pl, tv, z))
    
    def get_batch(self, batch_size):
        """获取一个训练 batch。"""
        if len(self.buffer) < batch_size:
            return None
        
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        
        planes = np.stack([b[0] for b in batch])
        pi_t = np.stack([b[1] for b in batch])
        z = np.stack([b[2] for b in batch])
        
        return planes, pi_t, z
    
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
