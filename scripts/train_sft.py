"""
19x19 监督训练脚本（AlphaGoZero 风格策略+价值网络）。

依赖 scripts/build_dataset.py 产出的紧凑 npz 数据集。监督学习不涉及
自我对弈，因此绕开了 H1/H2/H4/H5/L1 等自我对弈性能问题。

用法:
    python scripts/train_sft.py --data data/sft_dataset.npz --device cuda --use-amp \
        --batch-size 512 --epochs 5 --save-every 2000 --out models/sft.pt
"""

import argparse
import logging
import multiprocessing as mp
import os
import queue
import random
import sys
import threading
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---- NPU(Ascend/CANN) 后端兼容辅助 ----
# torch_npu 是可选依赖，未安装时不应让静态分析/运行时报错。所有 NPU 访问都经
# 过这里统一守卫；未安装 torch_npu 的环境（纯 CUDA 开发机）这些函数返回安全默认值。
def npu_is_available() -> bool:
    if not hasattr(torch, 'npu'):
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def npu_get_device_name(idx: int = 0) -> str:
    try:
        return str(torch.npu.get_device_name(idx))
    except Exception:
        return 'Ascend-NPU'


def npu_memory_reserved(device) -> float:
    try:
        return float(torch.npu.memory_reserved(device))
    except Exception:
        return 0.0


def npu_empty_cache() -> None:
    try:
        torch.npu.empty_cache()
    except Exception:
        pass


def npu_grad_scaler(enabled: bool):
    return torch.npu.amp.GradScaler(enabled=enabled)


def npu_out_of_memory_error_type():
    # 未装 torch_npu 的环境（纯 CUDA 开发机）torch.npu 属性不存在，
    # 必须先 hasattr 守卫，否则 OOM 异常处理路径本身会抛 AttributeError
    if not hasattr(torch, 'npu'):
        return None
    return getattr(torch.npu, 'OutOfMemoryError', None)


def _auto_select_device():
    """自动选择最优训练设备。

    策略（按优先级）：
      1. 探测 CUDA 与 NPU 各卡的空闲显存，挑空闲显存最大的那张卡。
      2. 若两后端都可用，选「空闲显存更大」的后端（A100 通常 > 910B，但按实测）。
      3. 都不可用则回退 CPU。
    返回形如 'cuda:0' / 'npu:1' / 'cpu' 的具体设备串。
    """
    def _cuda_free(idx):
        try:
            torch.cuda.synchronize(idx)
            total = torch.cuda.get_device_properties(idx).total_memory
            alloc = torch.cuda.memory_allocated(idx)
            return max(0, total - alloc)
        except Exception:
            return 0

    def _npu_free(idx):
        try:
            total = torch.npu.get_device_properties(idx).total_memory
            alloc = torch.npu.memory_allocated(idx)
            return max(0, total - alloc)
        except Exception:
            return 0

    best = None  # (free_bytes, backend, idx)
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            free = _cuda_free(i)
            if best is None or free > best[0]:
                best = (free, 'cuda', i)
    if npu_is_available():
        try:
            n = torch.npu.device_count()
        except Exception:
            n = 0
        for i in range(n):
            free = _npu_free(i)
            if best is None or free > best[0]:
                best = (free, 'npu', i)
    if best is None:
        return 'cpu'
    _, backend, idx = best
    return f'{backend}:{idx}'


def _check_training_env(logger):
    """启动时检查三大加速能力并打印诊断：flash-attn 库 / torch.compile / 混合精度。

    纯检测不改变行为；实际的启用决策在设备路径确定后进行（见 main 中
    set_flash_attn / args.compile / amp_dtype 分支）。检测结果会写入日志，
    便于比对云端/本地环境差异。
    """
    # 1) flash-attn 独立库（可选依赖，仅 Ampere+ CUDA 有收益）
    try:
        import flash_attn  # type: ignore[import-not-found]
        fa_status = "已安装 v%s" % getattr(flash_attn, "__version__", "?")
    except Exception:
        fa_status = ("未安装（可选；Ampere+ CUDA 上比内置 SDPA 再快 20-30%%，"
                     "安装: pip install flash-attn --no-build-isolation）")
    # 2) torch.compile（inductor 后端）
    if hasattr(torch, 'compile'):
        try:
            from torch._inductor import config as _ind_cfg  # noqa: F401
            comp_status = "可用 (inductor)"
        except Exception:
            comp_status = "可用"
    else:
        comp_status = "不可用（torch<2.0）"
    # 3) 混合精度（按后端能力）
    try:
        import torch_npu  # noqa: F401 — 确保 torch.npu 可用
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        cap = (p.major, p.minor)
        if cap >= (8, 0):
            amp_status = "bf16+fp16 可用（%s, sm_%d%d）" % (p.name, cap[0], cap[1])
        else:
            amp_status = "仅 fp16（%s, sm_%d%d，Volta/Turing 无 bf16）" % (p.name, cap[0], cap[1])
    elif npu_is_available():
        amp_status = "bf16(910B)/fp16(910A) 可用（Ascend NPU，运行时按型号选择）"
    else:
        amp_status = "不支持（CPU 走 FP32）"
    logger.info("[env] flash-attn: %s", fa_status)
    logger.info("[env] torch.compile: %s | 混合精度: %s", comp_status, amp_status)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.networks.alphanet import AlphaGoNet
from src.data.dataset import SupervisedDataset
from scripts.build_dataset import build


def save_model(model, path):
    """保存模型权重，并剥离 DDP 包裹产生的 'module.' 前缀与 torch.compile 产生的
    '_orig_mod.' 前缀，保证存档无论是否经 DDP/compile 都能被后续普通加载/resume 使用。"""
    sd = model.state_dict()
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    if any(k.startswith('_orig_mod.') for k in sd.keys()):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(sd, path)


def setup_logging(log_file, level: int = logging.INFO, rank: int = 0) -> logging.Logger:
    """配置 logging：同时写文件与输出到控制台（无缓冲，实时可见）。

    rank>0（DDP 非主进程）时仅保留 ERROR 以上到 stderr，避免多卡日志刷屏——
    各进程是独立进程，各自的 logger 互不干扰，这里只控制本进程输出量。
    """
    logger = logging.getLogger('train')
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    else:
        eh = logging.StreamHandler(sys.stderr)
        eh.setLevel(logging.ERROR)
        eh.setFormatter(fmt)
        logger.addHandler(eh)
    return logger


class EMA:
    """指数移动平均（Exponential Moving Average）权重。

    维护模型参数的 shadow copy，eval/save 时用 EMA 权重可提升 1-3% accuracy。
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach()
                       for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name].data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        self.backup = {name: param.clone()
                       for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data = self.shadow[name].data

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name].data
        del self.backup


def maybe_autocast(device, dtype=torch.float16):
    """在 CUDA/NPU 上开启 autocast，dtype 由设备能力决定（A100/BF16、V100/FP16、NPU/BF16）。
    CPU 或 amp 关闭时返回 nullcontext。device 字符串支持 'cuda'/'cuda:0'/'npu'/'npu:0' 等。"""
    dev = device.split(':')[0] if isinstance(device, str) else str(device)
    if dev in ('cuda', 'npu'):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            try:
                return torch.amp.autocast(dev, dtype=dtype)
            except TypeError:
                # 老接口回退
                if dev == 'cuda':
                    return torch.cuda.amp.autocast(enabled=True, dtype=dtype)
                return torch.npu.amp.autocast(enabled=True, dtype=dtype)
    return nullcontext()


def load_dataset(path):
    """加载单个 .npz 训练集。"""
    d = np.load(path, allow_pickle=False)
    return SupervisedDataset({k: d[k] for k in d.files})


def evaluate_top1(model, dataset, idxs, bs, device, amp_dtype, max_batches=50,
                  use_channels_last=False):
    """验证集 top-1 着法准确率。返回 (accuracy, num_samples)。"""
    model.eval()
    correct = 0
    total = 0
    n_batches = min((len(idxs) + bs - 1) // bs, max_batches)
    with torch.no_grad():
        for b in range(n_batches):
            sel = idxs[b * bs:(b + 1) * bs]
            if len(sel) == 0:
                break
            states_np, moves_np, _ = dataset.sample_batch_numpy(sel)
            state = torch.from_numpy(states_np).to(device)
            if use_channels_last:
                state = state.to(memory_format=torch.channels_last)
            move_t = torch.from_numpy(moves_np).to(device)
            with maybe_autocast(device, amp_dtype):
                policy_logits, _ = model(state)
            pred = policy_logits.argmax(dim=-1)
            correct += int((pred == move_t).sum())
            total += len(sel)
    model.train()
    return correct / max(total, 1), total


def evaluate_metrics(model, dataset, idxs, bs, device, amp_dtype, max_batches=50,
                     use_channels_last=False):
    """验证集综合指标：top-1/5/10 准确率 + policy KL + value Brier score。

    返回 dict:
        top1, top5, top10 : 着法准确率 (0~1)
        kl                : model softmax vs expert one-hot 的 KL散量
        brier             : value 预测 vs 实际胜负的 Brier score（越小越好）
        n                 : 样本数
    """
    import torch.nn.functional as F
    model.eval()
    correct1 = correct5 = correct10 = 0
    total = 0
    kl_sum = 0.0
    brier_sum = 0.0
    n_batches = min((len(idxs) + bs - 1) // bs, max_batches)
    with torch.inference_mode():
        for b in range(n_batches):
            sel = idxs[b * bs:(b + 1) * bs]
            if len(sel) == 0:
                break
            states_np, moves_np, values_np = dataset.sample_batch_numpy(sel)
            state = torch.from_numpy(states_np).to(device)
            if use_channels_last:
                state = state.to(memory_format=torch.channels_last)
            move_t = torch.from_numpy(moves_np).to(device)
            value_t = torch.from_numpy(values_np).to(device)  # (B,1) in {-1,+1}
            with maybe_autocast(device, amp_dtype):
                policy_logits, value_pred = model(state)
            B = len(sel)
            total += B

            # --- top-k 准确率 ---
            topk = policy_logits.topk(10, dim=-1).indices  # (B,10)
            correct1 += int((topk[:, 0] == move_t).sum())
            correct5 += int((topk[:, :5] == move_t.unsqueeze(1)).any(dim=1).sum())
            correct10 += int((topk == move_t.unsqueeze(1)).any(dim=1).sum())

            # --- policy KL 散量 ---
            # expert: one-hot at move_t -> log prob; model: log_softmax
            log_p = F.log_softmax(policy_logits, dim=-1)  # (B, A)
            with torch.inference_mode():
                target = torch.zeros_like(log_p).scatter_(
                    1, move_t.unsqueeze(1).clamp(
                        max=log_p.shape[1] - 1), 1.0)
            # KL(expert || model) = sum(expert * (log_expert - log_model))
            # expert 为 one-hot，简化为 -log_p[expert_move]（即 cross-entropy）
            # 但更标准的 KL = sum(expert * log(expert / model))
            # = sum(expert * (0 - log_model)) for one-hot = -log_p[expert_move]
            # 即 NLL，与 CE 等价。若要真正 KL(expert||model) 需要 expert 有分布
            # 此处用 model 分布 vs uniform 的 KL 作为策略集中度指标：
            # KL(model || uniform) = log(A) + sum(p * log(p))
            A = log_p.shape[1]
            p = F.softmax(policy_logits, dim=-1)
            kl_per_sample = (torch.log(torch.tensor(A, dtype=p.dtype))
                             + (p * log_p).sum(dim=-1))  # (B,)
            kl_sum += float(kl_per_sample.sum())

            # --- value Brier score ---
            # Brier = mean((pred - actual)^2), actual in {-1,+1}, pred in tanh output
            # 映射到 [0,1]: actual_01 = (actual+1)/2, pred_01 = (pred+1)/2
            pred_01 = (value_pred.squeeze(-1) + 1) / 2
            act_01 = (value_t.squeeze(-1) + 1) / 2
            brier_sum += float(((pred_01 - act_01) ** 2).sum())

    model.train()
    n = max(total, 1)
    return {
        'top1': correct1 / n,
        'top5': correct5 / n,
        'top10': correct10 / n,
        'kl': kl_sum / n,
        'brier': brier_sum / n,
        'n': total,
    }


def _concat_dicts(dicts):
    """按相同 key 沿第 0 轴拼接多个数据 dict（字段形状一致）。"""
    out = {}
    for k in dicts[0].keys():
        out[k] = np.concatenate([dd[k] for dd in dicts], axis=0)
    return out


def load_from_path(path, board_size, max_games_per_tgz=0):
    """加载训练数据。

    - 若 path 是文件：按 .npz 加载（兼容原行为）。
    - 若 path 是目录：递归扫描其下所有 .tgz/.tar.gz（用 build_dataset.build 解析）
      与 .npz（直接加载），合并成一个 SupervisedDataset。这样可直接喂一个装着
      多个分片 tgz 的文件夹，无需先手动 build_dataset 成单个 npz。
    """
    if os.path.isdir(path):
        import glob
        tgzs = (sorted(glob.glob(os.path.join(path, '**', '*.tgz'), recursive=True))
                + sorted(glob.glob(os.path.join(path, '**', '*.tar.gz'), recursive=True)))
        npzs = sorted(glob.glob(os.path.join(path, '**', '*.npz'), recursive=True))
        # 直接子目录也作为数据源（build 支持递归解析目录内的 .sgf，
        # 例如 data/games/ 这种已解压的棋谱目录）
        subdirs = sorted(d for d in glob.glob(os.path.join(path, '*'))
                         if os.path.isdir(d) and d not in npzs)
        dicts = []
        n_games_total = 0
        n_skip_total = 0

        def _try_build(src):
            """build 可能对单个分片返回 0 有效局并抛 RuntimeError，这里吞掉并跳过。"""
            _log = logging.getLogger('train')
            try:
                d, n_games, skip = build(src, board_size, max_games_per_tgz)
            except RuntimeError as e:
                _log.warning("[data] 跳过分片 %s：%s", src, e)
                return None, 0, 0
            if n_games == 0:
                _log.warning("[data] 跳过分片 %s：0 有效局（与 --board-size %d 不匹配或空）",
                               src, board_size)
                return None, 0, 0
            return d, n_games, skip

        for tg in tgzs + subdirs:
            d, n_games, skip = _try_build(tg)
            if d is not None:
                dicts.append(d)
                n_games_total += n_games
                n_skip_total += skip
        print(f"[data] 已解析 {len(dicts)} 个有效分片，有效局 {n_games_total}，跳过 {n_skip_total}")
        for npz in npzs:
            dd = np.load(npz, allow_pickle=False)
            dicts.append({k: dd[k] for k in dd.files})
        if not dicts:
            raise RuntimeError(
                f"目录 {path} 下未解析到任何有效棋谱，请检查 --board-size 是否与棋谱尺寸匹配")
        merged = _concat_dicts(dicts)
        print(f"[data] 合并后样本数 {merged['boards'].shape[0]}")
        return SupervisedDataset(merged)
    return load_dataset(path)


_worker_dataset = None  # multiprocessing worker 进程中由 initializer 设置


def _prefetch_worker_init(dataset):
    """multiprocessing worker 初始化：在子进程中保存 dataset 引用。"""
    global _worker_dataset
    _worker_dataset = dataset


def _prefetch_worker(wi, task_q, res_q, seed, dataset):
    """multiprocessing worker：从 task_q 取任务，计算后放 res_q。"""
    rng = np.random.default_rng(seed + wi)
    while True:
        item = task_q.get()
        if item is None:
            return
        step, pos, sub_idx = item
        try:
            s, m, v = dataset.sample_batch_numpy(sub_idx, rng=rng)
            res_q.put((step, pos, s, m, v, None))
        except Exception as e:  # noqa: BLE001
            res_q.put((step, pos, None, None, None, e))


class _BatchPrefetcher:
    """后台多进程并行构造训练 batch，与 NPU 前向/反向重叠。

    把每个 batch 的样本下标切成 num_workers 个子块，由 num_workers 个后台进程
    并行调用 dataset.sample_batch_numpy()（绕过 GIL，numpy 操作真正并行），
    主进程按序拼回整批。

    两步流水：submit() 投递一个 batch 的下标，next() 取回构造好的 numpy 数组。
    两个有界队列提供背压，避免无限预取吃内存。各进程用独立 np.random.Generator。
    """

    def __init__(self, dataset, num_workers=4, prefetch=2, seed=1234):
        self.dataset = dataset
        self.k = max(1, int(num_workers))
        self.prefetch = max(1, int(prefetch))
        cap = self.k * self.prefetch
        self._task_q: mp.Queue = mp.Queue(maxsize=cap)
        self._res_q: mp.Queue = mp.Queue(maxsize=cap)
        self._step = 0     # 下一个待投递 batch 的编号
        self._expect = 0   # 下一个待取回 batch 的编号
        self._pending: dict = {}  # step -> [(pos, s, m, v, err)] 已收到但还没收集完的
        self._processes = []
        for wi in range(self.k):
            p = mp.Process(
                target=_prefetch_worker,
                args=(wi, self._task_q, self._res_q, seed, dataset),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

    def submit(self, idxs):
        """投递一个 batch 的下标（队列满时阻塞，提供背压）。"""
        step = self._step
        self._step += 1
        n = len(idxs)
        for wi in range(self.k):
            start = wi * n // self.k
            end = (wi + 1) * n // self.k
            sub = idxs[start:end]
            if len(sub) == 0:
                self._res_q.put((step, wi, None, None, None, None))
            else:
                self._task_q.put((step, wi, sub))

    def next(self):
        """取回下一个 batch，返回 (states_np, moves_np, values_np)。"""
        step = self._expect
        self._expect += 1
        # 从 _pending 中取出之前缓存的该 step 结果
        parts = self._pending.pop(step, [])
        # 如果不够 k 个，从队列中继续收
        while len(parts) < self.k:
            r_step, pos, s, m, v, err = self._res_q.get()
            if r_step == step:
                parts.append((pos, s, m, v, err))
            else:
                # 缓存未来 step 的结果
                self._pending.setdefault(r_step, []).append((pos, s, m, v, err))
        # 检查错误
        for pos, s, m, v, err in parts:
            if err is not None:
                raise err
        # 按 pos 排序并拼接
        parts = [(pos, s, m, v) for pos, s, m, v, _ in parts if s is not None]
        parts.sort(key=lambda x: x[0])
        if not parts:
            raise RuntimeError(f"step {step}: 所有子块为空")
        states = np.concatenate([p[1] for p in parts], axis=0)
        moves = np.concatenate([p[2] for p in parts], axis=0)
        values = np.concatenate([p[3] for p in parts], axis=0)
        return states, moves, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help="单个 .npz 训练集，或包含多个 .tgz/.tar.gz/.npz 的目录（自动合并所有分片）")
    ap.add_argument('--max-games-per-tgz', type=int, default=0,
                    help="目录模式下每个 tgz 最多解析的棋局数（0=全部），用于子采样控制内存")
    ap.add_argument('--device', default='auto')
    ap.add_argument('--use-amp', action='store_true')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--board-size', type=int, default=19)
    ap.add_argument('--save-every', type=int, default=2000)
    ap.add_argument('--out', default='models/sft.pt')
    # 注意力相关
    ap.add_argument('--backbone-channels', type=int, default=128,
                    help='主干卷积通道数。容量主开关，实测（19路, mix/window'
                         ' 注意力, 4 层注意力头）：128×12≈4.06M，192×17≈12.41M，'
                         '224×12≈12.36M，256×12≈16.13M')
    ap.add_argument('--backbone-res-blocks', type=int, default=12,
                    help='主干残差块数量（与 --backbone-channels 共同决定容量）')
    ap.add_argument('--attention-mode', default='mix',
                    choices=['none', 'mix', 'all'],
                    help='主干注意力模式: none=纯卷积, mix=卷积+注意力混合, all=全注意力')
    ap.add_argument('--num-attention-layers', type=int, default=4,
                    help='mix 模式下注意力块数量')
    ap.add_argument('--num-heads', type=int, default=4, help='多头注意力头数')
    ap.add_argument('--attention-dropout', type=float, default=0.1)
    ap.add_argument('--attn-mode', default='global',
                    choices=['global', 'window', 'axial', 'sparse', 'window_global'],
                    help='注意力计算模式: global=全配对, window=块状窗口, '
                         'sparse=窗口+全局token, window_global=块状窗口+全局token(手写math)')
    ap.add_argument('--attn-window', type=int, default=7, help='window 模式窗口边长')
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--log-every', type=int, default=50,
                    help='每隔多少 step 打印一次训练日志（loss/lr/吞吐/显存）')
    ap.add_argument('--log-file', default='training.log',
                    help='训练日志文件路径（同时输出到控制台），设为空字符串可关闭文件日志')
    ap.add_argument('--prefetch-workers', type=int, default=4,
                    help='数据预取线程数：每个 batch 切块并行造特征并与 GPU 计算重叠；'
                         '<=1 关闭预取（回退同步取样）。')
    ap.add_argument('--prefetch-depth', type=int, default=2,
                    help='预取流水深度（提前多少个 batch 造好数据，控制内存/吞吐平衡）')
    ap.add_argument('--resume', default='',
                    help='断点续训：指定已保存的 .pth 模型路径，会从该权重 + 同目录 '
                         '.train_state.pt 恢复 optimizer/scheduler/step 计数继续训练')
    ap.add_argument('--value-loss-weight', type=float, default=5.0,
                    help='value loss 权重（BCE loss 下需更大权重平衡 policy/value 梯度）')
    ap.add_argument('--value-lr-mult', type=float, default=5.0,
                    help='value head 学习率倍数（相对主干 LR，补偿参数量小的梯度不足）')
    ap.add_argument('--label-smoothing', type=float, default=0.1,
                    help='policy loss label smoothing（0=不平滑，0.1=标准值）')
    ap.add_argument('--use-checkpoint', action='store_true',
                    help='用 gradient checkpointing 减少显存占用（约省 50%%，训练慢 ~30%%）')
    ap.add_argument('--gradient-accumulation-steps', type=int, default=1,
                    help='梯度累积步数（模拟更大 batch size，效果等同于 batch_size * N）')
    ap.add_argument('--compile', action='store_true',
                    help='用 torch.compile 融合算子（GPU 上约 20-40%% 提速，首次迭代较慢）')
    ap.add_argument('--compile-mode', default='default',
                    choices=['default', 'max-autotune', 'reduce-overhead'],
                    help='torch.compile 模式: default=常规融合, max-autotune=A100 上进一步 '
                         '自动调优提速（编译更久）, reduce-overhead=小 batch 低开销')
    args = ap.parse_args()

    # ---- 分布式训练环境变量（由 torchrun / mp.spawn 注入）----
    # RANK/WORLD_SIZE/LOCAL_RANK 同时存在且 WORLD_SIZE>1 时进入 DDP 模式。
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_dist = world_size > 1
    is_main = (rank == 0)

    # 配置日志（控制台 + 文件），统一用 logger 输出便于事后排查
    log_file = args.log_file if args.log_file else None
    logger = setup_logging(log_file, rank=rank)
    logger.info("=" * 60)
    _check_training_env(logger)
    logger.info("=" * 60)
    logger.info("配置: data=%s board=%d batch=%d epochs=%d lr=%s wd=%s",
                args.data, args.board_size, args.batch_size, args.epochs,
                args.lr, args.weight_decay)
    logger.info("注意力: mode=%s attn_mode=%s window=%d heads=%d layers=%d dropout=%s compile=%s",
                args.attention_mode, args.attn_mode, args.attn_window,
                args.num_heads, args.num_attention_layers, args.attention_dropout, args.compile)
    logger.info("日志: log_every=%d eval_every=%d save_every=%d out=%s",
                args.log_every, args.eval_every, args.save_every, args.out)
    logger.info("=" * 60)

    # ---- 分布式训练：设备由 LOCAL_RANK 决定，忽略 --device 卡号 ----
    # 后端选择：NPU 走 hccl，CUDA 走 nccl。多卡前必须 init_process_group，
    # 否则后续 .to(device) / DDP 包裹会失败或各卡不互通。
    if is_dist:
        _dist_backend = (args.device.split(':')[0]
                         if args.device not in ('auto', '') else
                         ('npu' if npu_is_available() else 'cuda'))
        if _dist_backend == 'npu':
            import torch_npu  # noqa: F401 — 注册 HCCL 后端
            dist.init_process_group('hccl')
            torch.npu.set_device(local_rank)
        else:
            dist.init_process_group('nccl')
            torch.cuda.set_device(local_rank)
        device = f'{_dist_backend}:{local_rank}'
        if is_main:
            logger.info("[ddp] 初始化分布式训练 | backend=%s world_size=%d",
                        _dist_backend, world_size)
    else:
        if args.device == 'auto':
            device = _auto_select_device()
        else:
            device = args.device

    use_amp = args.use_amp or (device.split(':')[0] in ('cuda', 'npu'))

    # ---- 多后端自适应路径（CUDA / NPU / CPU）----
    # 各后端能力差异很大，逐后端决定：
    #   - amp_dtype:       A100/A800/H100(sm_80+) -> bfloat16（原生支持）
    #                     Ascend 910B/910A -> float16（NPU autocast 仅支持 FP16）
    #                     V100(sm_70, Volta) -> float16（无 bf16）
    #   - use_scaler:      BF16 下关闭 GradScaler（不下溢）；FP16 下开启
    #   - use_channels_last: A100 卷积走 NHWC 更快；NPU/CPU 收益有限默认关
    #   - sdpa_force_math: NPU 上 FlashAttention 后端不稳（CANN SDPA 与 CUDA 不同），
    #                     强制走手写 math 注意力最稳；V100 同样强制 math；A100 走 Flash
    #   - compile_disable_sparse: 所有后端统一禁用——unfold 产生 (B, Hh*d, N, ws²) 巨型
    #                      中间张量，inductor freezing 常量折叠会以 fp32 物化
    #                      (B,N,Hh,ws²,d)（batch512 下单个 4.3GB）直接编译期 OOM；
    #                      稀疏/窗口注意力走 eager+autocast（V100 验证过的稳定路径），
    #                      编译图仅覆盖卷积/线性/FFN。NPU 上 inductor 本身不可用
    amp_dtype = torch.float16
    use_scaler = use_amp
    use_channels_last = False
    sdpa_force_math = True
    compile_disable_sparse = True
    gpu_name = 'N/A'
    compute_cap = (0, 0)
    _backend = device.split(':')[0]
    # 具体卡号（device 形如 'cuda:1' / 'npu:0' / 'cpu'），无索引时默认 0
    try:
        _dev_idx = int(device.split(':')[1]) if ':' in device else 0
    except ValueError:
        _dev_idx = 0
    if _backend == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # A100+ 上开启 TF32：未被 autocast 覆盖的 fp32 matmul/conv 走 TF32 tensor core
        torch.set_float32_matmul_precision('high')
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        props = torch.cuda.get_device_properties(_dev_idx)
        gpu_name = props.name
        compute_cap = (props.major, props.minor)
        is_ampere_plus = compute_cap >= (8, 0)
        if is_ampere_plus:
            amp_dtype = torch.bfloat16
            use_scaler = False  # BF16 几乎不下溢，去掉 GradScaler 省一次 CUDA 同步
            use_channels_last = True
            sdpa_force_math = False  # A100 走 FlashAttention 后端
            # unfold 巨型中间张量会触发 inductor freezing 以 fp32 物化 (B,N,Hh,ws²,d)
            # 导致编译期 OOM（batch512 下单个 4.3GB），必须排除出编译图
            compile_disable_sparse = True
            logger.info("[device] %s (sm_%d%d) | 启用 A100 路径: BF16 + FlashAttn + "
                        "channels_last + compile(卷积/线性/FFN)", gpu_name, *compute_cap)
        else:
            # V100 等老卡：保守路径（与原行为一致）
            amp_dtype = torch.float16
            use_scaler = use_amp
            use_channels_last = False
            sdpa_force_math = True
            compile_disable_sparse = True
            logger.info("[device] %s (sm_%d%d) | 走保守路径: FP16 + 手写 math 注意力 + "
                        "稀疏注意力禁用编译", gpu_name, *compute_cap)
    elif _backend == 'npu' and npu_is_available():
        # Ascend 910B / 910A：CANN + torch_npu 后端
        gpu_name = npu_get_device_name(_dev_idx)
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        # 关键：按芯片型号选精度。910B/910Pro 原生 BF16；910A 无 BF16，必须走
        # FP16 + GradScaler（与 V100 路径一致）。CANN 上 FlashAttention 后端不稳，
        # 强制手写 math 注意力；channels_last 对 NPU 卷积无明确收益，关闭；
        # torch.compile(inductor) 在 NPU 不可用，禁用。
        if '910B' in gpu_name or '910Pro' in gpu_name or '910-2' in gpu_name:
            amp_dtype = torch.float16  # NPU autocast 仅支持 FP16
            use_scaler = True  # FP16 需要 GradScaler 防下溢
            logger.info("[device] %s (NPU/CANN) | 910B 路径: FP16 + GradScaler + 手写 math "
                        "注意力 + 禁用 torch.compile(inductor)", gpu_name)
        else:
            amp_dtype = torch.float16
            use_scaler = use_amp  # 910A 无 BF16，FP16 必须开 GradScaler
            logger.info("[device] %s (NPU/CANN) | 910A 路径: FP16 + GradScaler + 手写 math "
                        "注意力 + 禁用 torch.compile(inductor)", gpu_name)
        use_channels_last = False
        sdpa_force_math = True
        compile_disable_sparse = True
        if args.compile:
            logger.warning("[device] NPU 上 torch.compile(inductor) 不可用，已忽略 --compile；"
                           "如需图编译请用 torchair (torch_npu.experimental_config)。")
            args.compile = False
    else:
        # CPU 或其他：纯 FP32，无 AMP、无 channels_last
        amp_dtype = torch.float32
        use_scaler = False
        use_channels_last = False
        sdpa_force_math = True
        compile_disable_sparse = True
        logger.info("[device] CPU | 走 FP32 路径（无 AMP/编译）")

    # 把注意力后端/编译开关透传给 backbone 模块（所有分支统一设置）
    from src.networks import backbone as _backbone
    _backbone.set_sdpa_force_math(sdpa_force_math)
    _backbone.set_compile_disable_sparse(compile_disable_sparse)

    # flash-attn 独立库启用决策：仅「Ampere+ CUDA 且走非 math 路径」时尝试加载。
    # 加载失败自动回退内置 SDPA，不影响训练启动。
    # 环境变量 GOAI_FLASH=0 可强制禁用（A/B 实测用：稀疏注意力分块 seq 仅 ~30，
    # flash 小 kernel 密集发射在部分配置下比手写 math 更慢，需实测决定）。
    _flash_wanted = os.environ.get('GOAI_FLASH', '1') != '0'
    if _flash_wanted and _backend == 'cuda' and compute_cap >= (8, 0) and not sdpa_force_math:
        fa_ok, fa_msg = _backbone.set_flash_attn(True)
        if fa_ok:
            logger.info("[env] 注意力内核: flash-attn %s（优先于内置 SDPA）", fa_msg)
        else:
            logger.info("[env] 注意力内核: 内置 SDPA（flash-attn %s）", fa_msg)
    else:
        _backbone.set_flash_attn(False)
        if not _flash_wanted:
            logger.info("[env] 注意力内核: 手写 math（GOAI_FLASH=0 已禁用 flash）")
        else:
            logger.info("[env] 注意力内核: %s",
                        "手写 math（%s 不支持 Flash）" % (_backend.upper(),)
                        if sdpa_force_math else "内置 SDPA")

    logger.info("启动训练 | torch=%s | device=%s | amp_dtype=%s scaler=%s channels_last=%s",
                torch.__version__, device, amp_dtype, use_scaler, use_channels_last)

    dataset = load_from_path(args.data, args.board_size, args.max_games_per_tgz)
    n = len(dataset)
    # 按棋局分割 train/eval（避免同一棋局的相邻位置同时出现在 train 和 eval）
    if dataset.game_ids is not None:
        unique_games = np.unique(dataset.game_ids)
        rng = np.random.default_rng(0)
        rng.shuffle(unique_games)
        n_train_games = int(len(unique_games) * 0.98)
        train_game_set = set(unique_games[:n_train_games])
        train_idx = np.array([i for i in range(n) if dataset.game_ids[i] in train_game_set])
        eval_idx = np.array([i for i in range(n) if dataset.game_ids[i] not in train_game_set])
        logger.info("[data] 按棋局分割：总棋局=%d 训练棋局=%d 验证棋局=%d",
                    len(unique_games), n_train_games, len(unique_games) - n_train_games)
    else:
        n_train = int(n * 0.98)
        idx_all = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(idx_all)
        train_idx = idx_all[:n_train]
        eval_idx = idx_all[n_train:]
    logger.info("[data] 总样本数=%d | 训练=%d | 验证=%d", n, len(train_idx), len(eval_idx))

    # 分布式：每张卡用 DistributedSampler 取到不相交的训练分片（会自动 pad 到
    # 能被 world_size 整除），各卡步数因此一致，避免 DDP 在 barrier 处互相等待。
    if is_dist:
        train_sampler = torch.utils.data.DistributedSampler(
            torch.arange(len(train_idx)), num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    model = AlphaGoNet(
        in_channels=12,
        backbone_channels=args.backbone_channels,
        backbone_res_blocks=args.backbone_res_blocks,
        attention_mode=args.attention_mode,
        num_attention_layers=args.num_attention_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        attn_mode=args.attn_mode,
        attn_window=args.attn_window,
        action_size=args.board_size * args.board_size + 1,  # +1 为 pass 类别
        use_checkpoint=args.use_checkpoint,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[model] 参数量=%.2fM | 设备=%s", n_params / 1e6, device)

    # A100 上把卷积型特征（N,C,H,W）转 channels_last(NHWC)，卷积算子走更快内存布局。
    # 输入 state 也需同步转格式（见训练/评估循环），故这里仅转换模型权重布局。
    if use_channels_last and _backend == 'cuda':
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        logger.info("[model] 已启用 channels_last (NHWC) 内存格式（A100 卷积加速）")

    # 参数组：value head 独立 LR（参数量小，需要更高学习率补偿梯度不足）
    # 排除 bias / BatchNorm / LayerNorm 参数的 weight decay（标准做法）
    no_decay_params = set()
    for name, param in model.named_parameters():
        if param.ndim == 1:  # bias, BN/LN weight, BN/LN bias
            no_decay_params.add(name)

    value_decay = [p for n, p in model.value.named_parameters()
                   if n not in no_decay_params]
    value_no_decay = [p for n, p in model.value.named_parameters()
                      if n in no_decay_params]
    other_decay = [p for n, p in model.named_parameters()
                   if 'value' not in n and n not in no_decay_params]
    other_no_decay = [p for n, p in model.named_parameters()
                      if 'value' not in n and n in no_decay_params]

    optimizer = torch.optim.AdamW([
        {'params': other_decay, 'lr': args.lr,
         'weight_decay': args.weight_decay},
        {'params': other_no_decay, 'lr': args.lr, 'weight_decay': 0.0},
        {'params': value_decay, 'lr': args.lr * args.value_lr_mult,
         'weight_decay': args.weight_decay},
        {'params': value_no_decay, 'lr': args.lr * args.value_lr_mult,
         'weight_decay': 0.0},
    ])
    # BF16 后端（A100/NPU）下 use_scaler=False（BF16 不下溢，省去 loss scaling 的额外同步）；
    # V100/FP16 下开启 GradScaler。按设备选择 GradScaler 实现。
    if _backend == 'npu':
        scaler = npu_grad_scaler(enabled=use_scaler)
    else:
        scaler = torch.amp.GradScaler(_backend, enabled=use_scaler)

    # EMA（指数移动平均）：eval/save 时用 shadow 权重，提升 1-3% accuracy
    ema = EMA(model, decay=0.999)

    # ---- 学习率调度：基于“总 step 数”而非 epoch 数 ----
    # 旧版用 T_max=args.epochs 导致余弦在第 1 个 epoch 结束就被砍到 ~0，
    # 后续 epoch 在 lr≈0 附近横盘。这里用真实总 step 数，并加前 5% step 线性 warmup。
    # DDP 下“每卡”样本数约为 n_train/world_size，调度按每卡步数推进，
    # 这样各卡 LR 曲线一致；有效全局 batch = batch_size * world_size。
    if is_dist:
        per_rank = (n_train + world_size - 1) // world_size
        n_batches = (per_rank + args.batch_size - 1) // args.batch_size
    else:
        n_batches = (n_train + args.batch_size - 1) // args.batch_size
    total_steps = max(1, args.epochs * n_batches)
    warmup_steps = max(1, int(total_steps * 0.10))
    after_warmup = max(1, total_steps - warmup_steps)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=after_warmup)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    scheduler.step()  # 初始化 LR 为 warmup 起始值（0.1 * base_lr），避免第一步用满 lr

    bs = args.batch_size
    step = 0
    best_eval_acc = -1.0
    start_epoch = 0
    t0 = time.time()

    # ---- 断点续训：从 --resume 指定的模型权重 + 同目录 .train_state.pt 恢复 ----
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume 指定的模型不存在: {args.resume}")
        state_path = args.resume + '.train_state'
        if not os.path.isfile(state_path):
            # 兼容训练中途 save_every 的快照命名：<out>.latest.train_state
            _latest = args.resume + '.latest.train_state'
            if os.path.isfile(_latest):
                state_path = _latest
                logger.info("[resume] 使用中途快照训练状态: %s", _latest)
        logger.info("[resume] 加载模型权重: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        # 统一前缀：checkpoint 可能带（来自 compile 存档）或不带 "_orig_mod." 前缀，
        # 目标模型也可能被 compile 包成 OptimizedModule（内部为 _orig_mod）。
        # 先全部规范化成不带前缀，再按需补回，保证任意组合都能匹配。
        ckpt = {k.replace('_orig_mod.', '', 1): v for k, v in ckpt.items()}
        target = getattr(model, '_orig_mod', model)  # compile 后为真实模块
        if hasattr(model, '_orig_mod'):
            ckpt = {'_orig_mod.' + k: v for k, v in ckpt.items()}
            logger.info("[resume] 模型已 torch.compile 包装，权重按 _orig_mod. 前缀对齐")
        target.load_state_dict(ckpt)
        if os.path.isfile(state_path):
            tstate = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(tstate['optimizer'])
            scheduler.load_state_dict(tstate['scheduler'])
            try:
                scaler.load_state_dict(tstate['scaler'])
            except (RuntimeError, KeyError):
                logger.warning("[resume] scaler 状态不兼容（可能是 BF16→FP16 切换），从头开始")
            step = tstate.get('step', 0)
            best_eval_acc = tstate.get('best_eval_acc', -1.0)
            start_epoch = tstate.get('epoch', 0)
            if 'rng' in tstate:
                torch.set_rng_state(tstate['rng'].cpu())
            logger.info("[resume] 恢复训练状态 | step=%d best_eval_acc=%.4f epoch=%d",
                        step, best_eval_acc, start_epoch)
        else:
            logger.warning("[resume] 未找到 %s（仅恢复模型权重，optimizer/scheduler 从头开始）",
                           state_path)

    # torch.compile 融合算子（GPU 上约 20-40%% 提速）。必须在 resume 加载之后再做，
    # 否则模型会被包成 OptimizedModule，其 state_dict 带 "_orig_mod." 前缀，与
    # checkpoint 的 "backbone.xxx" 不匹配导致 load 失败。
    if args.compile:
        if hasattr(torch, 'compile'):
            try:
                # 4 个注意力块实例 × window/sparse 两个禁用点 × train/eval 两态，
                # dynamo 按 fn 身份分缓存，默认上限 64 会被打满触发反复重编译。
                # 注意：函数内不能写 `import torch._dynamo`（会把 torch 绑定为
                # 局部变量，导致函数开头 UnboundLocalError），用 importlib 规避。
                import importlib
                _dynamo_mod = importlib.import_module('torch._dynamo')
                _dynamo_mod.config.cache_size_limit = max(
                    256, _dynamo_mod.config.cache_size_limit)
            except Exception:  # noqa: BLE001
                pass
            try:
                model = torch.compile(model, dynamic=False, mode=args.compile_mode)
                # 预热前向必须与真实训练一致地包 autocast：flash-attn 只接受 fp16/bf16，
                # FP32 输入直灌会报 "FlashAttention only support fp16 and bf16 data
                # type"，导致 compile 被误判为不可用而回退 eager。
                with torch.no_grad(), maybe_autocast(device, amp_dtype):
                    dummy = torch.zeros(1, 12, args.board_size, args.board_size,
                                        device=device)
                    model(dummy)
                logger.info("[train] 已启用 torch.compile 算子融合")
            except Exception as e:  # noqa: BLE001
                # 真正回退 eager：剥离 OptimizedModule 包装，恢复原始模块引用
                model = getattr(model, '_orig_mod', model)
                logger.warning("[train] torch.compile 不可用，回退 eager: %s", e)
        else:
            logger.info("[train] 当前 torch 版本不支持 torch.compile，跳过")

    # 分布式：DDP 包裹需在 torch.compile 之后（算子融合与梯度同步可共存）。
    # DDP 会为 state_dict 加 "module." 前缀，save_model 已做剥离处理。
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)
        if is_main:
            logger.info("[ddp] 已包裹 DistributedDataParallel | world_size=%d", world_size)

    if is_main:
        logger.info("[train] 开始训练 | steps/epoch=%d | 总 steps≈%d | warmup=%d",
                    n_batches, total_steps, warmup_steps)

    # 数据预取器：后台多线程并行造特征，与 GPU 前向/反向重叠（workers<=1 时关闭）
    pf = None
    if args.prefetch_workers > 1:
        pf = _BatchPrefetcher(dataset, num_workers=args.prefetch_workers,
                              prefetch=args.prefetch_depth)
        if is_main:
            logger.info("[data] 预取器已启用 | workers=%d depth=%d",
                        args.prefetch_workers, args.prefetch_depth)

    for epoch in range(start_epoch, args.epochs):
        # DDP：每卡取本 rank 的不相交分片；set_epoch 让每 epoch 重新洗牌
        if is_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)
            perm = [int(train_idx[j]) for j in train_sampler]
        else:
            rng.shuffle(train_idx)
            perm = train_idx
        model.train()
        epoch_loss = 0.0
        n_batches = (len(perm) + bs - 1) // bs
        # 预取器：每 epoch 先灌满流水线（提前 depth 个 batch 造好数据）
        if pf is not None:
            for _j in range(min(args.prefetch_depth, n_batches)):
                pf.submit(perm[_j * bs:(_j + 1) * bs])
        # 内核级剖析（诊断用）：GOAI_PROFILE=<step> 从该 step 起 profiling 50 个
        # step，结束打印 top CUDA kernel 耗时表，用于定位 740ms/step 的去向。
        _prof_at = int(os.environ.get('GOAI_PROFILE', '0') or 0)
        _prof_ctx = None
        _accum_steps = args.gradient_accumulation_steps
        for i in range(n_batches):
            try:
                if i % _accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                if _prof_at > 0 and step == _prof_at and _prof_ctx is None:
                    try:
                        from torch.profiler import (profile, ProfilerActivity)
                        _prof_ctx = profile(
                            activities=[ProfilerActivity.CPU,
                                        ProfilerActivity.CUDA])
                        _prof_ctx.__enter__()
                        logger.info("[profile] 已开始内核剖析（50 steps）...")
                    except Exception as pe:  # noqa: BLE001
                        logger.warning("[profile] 不可用: %s", pe)
                        _prof_at = 0
                if pf is not None:
                    # P0: 先提交下一个 batch，再取当前 batch（给 worker 更多预计算时间）
                    nxt = i + args.prefetch_depth
                    if nxt < n_batches:
                        pf.submit(perm[nxt * bs:(nxt + 1) * bs])
                    states_np, moves_np, values_np = pf.next()
                    if _backend == 'cuda':
                        state = torch.tensor(states_np, dtype=torch.float32, pin_memory=True)
                        move_t = torch.tensor(moves_np, dtype=torch.int64, pin_memory=True)
                        value_t = torch.tensor(values_np, dtype=torch.float32, pin_memory=True)
                    else:
                        state = torch.from_numpy(states_np.copy())
                        move_t = torch.from_numpy(moves_np.copy())
                        value_t = torch.from_numpy(values_np.copy())
                    if use_channels_last:
                        state = state.to(memory_format=torch.channels_last)
                    state = state.to(device, non_blocking=True)
                    move_t = move_t.to(device, non_blocking=True)
                    value_t = value_t.to(device, non_blocking=True)
                else:
                    sel = perm[i * bs:(i + 1) * bs]
                    state, move_t, value_t = dataset.sample_batch(sel, device)
                    # A100 上转 NHWC 以匹配模型 channels_last 布局，卷积更快
                    if use_channels_last:
                        state = state.to(memory_format=torch.channels_last)
                with maybe_autocast(device, amp_dtype):
                    policy_logits, value_logit = model(state)
                    policy_loss = F.cross_entropy(policy_logits.float(), move_t,
                                                  label_smoothing=args.label_smoothing)
                    # BCEWithLogitsLoss: target ±1 → 0/1，logit 直接输入无 Tanh
                    value_target = (value_t.squeeze().float() + 1) / 2  # ±1 → 0/1
                    value_loss = F.binary_cross_entropy_with_logits(
                        value_logit.float().squeeze(), value_target)
                    loss = policy_loss + args.value_loss_weight * value_loss
                (loss / _accum_steps).backward()
                if (i + 1) % _accum_steps == 0 or (i + 1) == n_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update()
            except Exception as oom_exc:
                # 同时捕获 CUDA 与 NPU 的 OOM（两后端异常类型不同）
                _oom_types = [torch.cuda.OutOfMemoryError]
                _npu_oom = npu_out_of_memory_error_type()
                if _npu_oom is not None:
                    _oom_types.append(_npu_oom)
                if not isinstance(oom_exc, tuple(_oom_types)):
                    raise
                if _backend == 'npu':
                    npu_empty_cache()
                else:
                    torch.cuda.empty_cache()
                # 保存当前进度，便于减小 batch 后用 --resume 续训（仅主进程写盘，避免多卡并发写同一文件）
                if is_main:
                    save_model(model, args.out + '.latest')
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'step': step,
                        'epoch': epoch,
                        'best_eval_acc': best_eval_acc,
                        'rng': torch.get_rng_state(),
                    }, args.out + '.latest.train_state')
                logger.error("=" * 60)
                logger.error("%s 显存不足 (OOM)！当前 --batch-size=%d 过大。",
                             device.upper(), bs)
                logger.error("window 注意力在 19x19 上把 batch 展开为 B*361，显存增长很快。")
                logger.error("建议减小 --batch-size（如 128/96/64），或调小 --attn-window。")
                logger.error("已保存进度至 %s.latest(.train_state)，可用 --resume 续训。",
                             args.out)
                logger.error("已清理显存并退出，请调整参数后重跑。")
                logger.error("=" * 60)
                sys.exit(1)
            if (i + 1) % _accum_steps == 0 or (i + 1) == n_batches:
                scheduler.step()   # 每个 optimizer step 推进一步
            step += 1
            # P1: 每 log_every 步才 sync NPU 取 loss 值，其余步用 None 占位
            if step % args.log_every == 0:
                epoch_loss += loss.item()
            else:
                epoch_loss += 0.0  # 占位，避免 NPU 同步

            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                if _backend == 'cuda':
                    mem = torch.cuda.memory_reserved(device) / 1e9
                elif _backend == 'npu':
                    mem = npu_memory_reserved(device) / 1e9
                else:
                    mem = 0.0
                speed = step * bs / max(1e-6, time.time() - t0)
                logger.info("[step %d/%d] loss=%.4f (p=%.4f v=%.4f) lr=%.2e "
                            "scale=%.0f mem=%.2fGB spd=%.0f s/s elapsed=%.0fs",
                            step, total_steps,
                            loss.item(), policy_loss.item(), value_loss.item(),
                            lr, scaler.get_scale(), mem, speed, time.time() - t0)

                # 内核剖析结束：打印 top CUDA kernel 耗时表
                if _prof_ctx is not None and step >= _prof_at + 50:
                    _prof_ctx.__exit__(None, None, None)
                    try:
                        table = _prof_ctx.key_averages().table(
                            sort_by='cuda_time_total', row_limit=18)
                        logger.info("[profile] 内核耗时 top-18（CUDA 时间排序）:\n%s",
                                    table)
                    except Exception as e:
                        logger.warning("[profile] 打印内核耗时表失败: %s", e)

            # 定期保存快照（仅主进程写盘）
            if is_main and args.save_every > 0 and step % args.save_every == 0:
                save_model(model, args.out + '.latest')
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'best_eval_acc': best_eval_acc,
                    'rng': torch.get_rng_state(),
                }, args.out + '.latest.train_state')

            # 定期评估：综合指标（所有 rank 都做 eval，避免 barrier 死锁）
            if args.eval_every > 0 and step % args.eval_every == 0 and len(eval_idx) > 0:
                ema.apply_shadow()
                metrics = evaluate_metrics(
                    model, dataset, eval_idx, bs, device, amp_dtype,
                    use_channels_last=use_channels_last)
                ema.restore()
                if is_main:
                    logger.info(
                        "[eval] step=%d top1=%.4f top5=%.4f top10=%.4f "
                        "kl=%.4f brier=%.4f (n=%d)%s",
                        step, metrics['top1'], metrics['top5'], metrics['top10'],
                        metrics['kl'], metrics['brier'], metrics['n'],
                        " ★ new best" if metrics['top1'] > best_eval_acc else "")
                if metrics['top1'] > best_eval_acc:
                    best_eval_acc = metrics['top1']
                    if is_main:
                        ema.apply_shadow()
                        save_model(model, args.out)
                        ema.restore()

if __name__ == "__main__":
    main()