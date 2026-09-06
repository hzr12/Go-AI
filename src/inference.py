"""
围棋 AI 推理引擎（监督学习 / SFT 模型）

兼容 src/networks.alphanet.AlphaGoNet 输出的 (policy, value)：
    - policy: 棋盘上每点 + 虚着(pass) 的概率分布
    - value : 当前执子方视角的局面胜率，tanh 后落在 [-1, 1]

特征由 src.game.go_rules.GoBoard.feature_planes 统一生成（12 通道），
与 src.data.dataset 使用同一套特征工程，避免训练/推理不一致。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import numpy as np
import torch

from src.game.go_rules import GoBoard
from src.networks.alphanet import AlphaGoNet


def _ensure_torch_npu():
    """导入 torch_npu（必须先于 .to('npu') 调用，注册 Ascend 后端）。返回是否可用。"""
    try:
        import torch_npu  # noqa: F401
        return True
    except Exception:
        return False


class GoAI:
    """基于 SFT 模型的围棋对弈 / 分析引擎。

    用法示例::

        ai = GoAI(model_path="models/sft_19x19.pth", board_size=19, device="cuda")
        # 自对弈一局
        ai.self_play(verbose=True, temperature=0.8)
        # 人机对弈（人类执黑先手）
        ai.play_against_human(human_color=1)
        # 分析某个局面的 top-k 候选着法
        ai.analyze()
    """

    # NPU batch 归桶：CANN 按输入形状编译算子，MCTS 的零散 batch（尾批 6、7 等）
    # 每种形状都要单独编译一次；归桶补零后形状固定，编译缓存才能跨调用命中。
    _NPU_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256)

    def __init__(self, model_path=None, board_size=19, device="auto", use_amp=False,
                 backbone_channels=128, backbone_res_blocks=12, policy_channels=32, value_channels=16,
                 attention_mode="mix", num_attention_layers=4, num_heads=4, attention_dropout=0.0,
                 attn_mode="global", attn_window=7, compile=False, tf32=False,
                 channels_last=True):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "npu" if _ensure_torch_npu() and torch.npu.is_available() else "cpu")
        self.device = device
        self.is_npu = device.startswith("npu")
        if self.is_npu and not _ensure_torch_npu():
            raise RuntimeError("--device npu 需要 torch_npu（须与 CANN 版本匹配，"
                               "910A 用 torch_npu 1.11~2.1 均可）")
        # 910A 不支持 bf16，NPU 上一律 fp16 autocast；CPU 不开 amp
        self.use_amp = use_amp and (self.device.startswith("cuda") or self.is_npu)
        if self.is_npu:
            print("[GoAI] Ascend NPU 推理：fp16 autocast（910A 无 bf16），"
                  "math 注意力，勿开 --compile。若每次启动 warmup 都超过 1 分钟，"
                  "先执行 export ASCEND_CACHE_PATH=~/ascend_cache 持久化算子编译缓存")
        self.board_size = board_size
        # 网络侧压：tf32 让 V100/Amp 上的 fp32 matmul 走 TensorFloat-32（约 2-4x 提速，
        # 精度损失对推理可忽略）；channels_last 让 conv 走 NHWC 内存布局（conv 友好）。
        self.tf32 = tf32 and self.device.startswith("cuda")
        if self.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.channels_last = channels_last and self.device.startswith("cuda")

        self.model = AlphaGoNet(
            in_channels=12,
            backbone_channels=backbone_channels,
            backbone_res_blocks=backbone_res_blocks,
            attention_mode=attention_mode,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            attn_mode=attn_mode,
            attn_window=attn_window,
            policy_channels=policy_channels,
            value_channels=value_channels,
            action_size=board_size * board_size + 1,  # +1 = 虚着
        ).to(self.device)
        # torch.compile 融合算子（GPU 上约 20-40% 提速），不支持时回退 eager。
        # 注意：torch.compile 是惰性的，错误在首次前向才抛出，因此编译后用
        # dummy 输入做一次 warmup 以触发真实编译并捕获异常。
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if compile and self.is_npu:
            print("[GoAI] NPU 不支持 torch.compile，已忽略")
        elif compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, dynamic=False)
                with torch.no_grad():
                    dummy = torch.zeros(1, 12, self.board_size, self.board_size,
                                        device=self.device)
                    if self.channels_last:
                        dummy = dummy.to(memory_format=torch.channels_last)
                    self.model(dummy)
                print("[GoAI] 已启用 torch.compile 算子融合")
            except Exception as e:  # noqa: BLE001
                print(f"[GoAI] torch.compile 不可用，回退 eager: {e}")
                # 重建未编译模型（前述 compile 包装可能已部分生效）
                self.model = AlphaGoNet(
                    in_channels=12, backbone_channels=backbone_channels,
                    backbone_res_blocks=backbone_res_blocks,
                    attention_mode=attention_mode,
                    num_attention_layers=num_attention_layers, num_heads=num_heads,
                    attention_dropout=attention_dropout, attn_mode=attn_mode,
                    attn_window=attn_window, policy_channels=policy_channels,
                    value_channels=value_channels,
                    action_size=board_size * board_size + 1,
                ).to(self.device)
                if self.channels_last:
                    self.model = self.model.to(memory_format=torch.channels_last)
        if self.tf32:
            print("[GoAI] 已启用 TF32 矩阵乘法加速 (CUDA)")
        self.model.eval()

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            # 兼容直接保存的 state_dict 或 {"model": state_dict}
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            # 从 policy 头输出层形状推断训练棋盘大小，防止 19 路权重载入
            # 9 路网络时 strict=False 静默留下随机 policy 头
            inferred = self._infer_board_size(state)
            if inferred is not None and inferred != self.board_size:
                print(f"[GoAI] 权重按 {inferred} 路训练（当前 board_size={self.board_size}），"
                      f"已按权重自动调整棋盘大小")
                self.board_size = inferred
                self.model = AlphaGoNet(
                    in_channels=12,
                    backbone_channels=backbone_channels,
                    backbone_res_blocks=backbone_res_blocks,
                    attention_mode=attention_mode,
                    num_attention_layers=num_attention_layers,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    attn_mode=attn_mode,
                    attn_window=attn_window,
                    policy_channels=policy_channels,
                    value_channels=value_channels,
                    action_size=inferred * inferred + 1,
                ).to(self.device)
                if self.channels_last:
                    self.model = self.model.to(memory_format=torch.channels_last)
            self.model.load_state_dict(state, strict=False)
            print(f"[GoAI] 已加载模型: {model_path}  ({device})")

        # NPU 首次前向会触发 CANN 算子初始化（可能 1-3 分钟且无输出），
        # 这里主动 warmup 并打印进度，避免被误认为卡死。
        if self.is_npu and model_path:
            print("[GoAI] NPU 首次前向 warmup 中（CANN 算子初始化，可能需要 1-3 分钟）…",
                  flush=True)
            t0 = time.time()
            # 预热 predict_batch 真实路径（含 autocast + batch 归桶补零）：
            # 让每个桶形状的 CANN 图在启动时一次性编译并落盘到 ASCEND_CACHE_PATH，
            # 避免自对弈每个新进程 / 新 batch 形状都冷编译 ~100s 而误判卡死。
            board = GoBoard(self.board_size)
            my_hist = [[-1, -1, -3], [-1, -1, -3]]
            # 只预热自对弈实际会命中的桶形状（expand-chunk=16 → B≤16）。
            # 32..256 自对弈用不到，留到运行时按需冷编译一次（已落盘缓存，安全）。
            max_warmup_batch = 16
            for nb in [b for b in self._NPU_BATCH_BUCKETS if b <= max_warmup_batch]:
                states = [(board, list(my_hist[0]), list(my_hist[1]), 1)] * nb
                with torch.no_grad():
                    self.predict_batch(states)
                print(f"  [warmup] batch={nb} 编译完成 ({time.time() - t0:.1f}s)",
                      flush=True)
            print(f"[GoAI] NPU warmup 完成: {time.time() - t0:.1f}s（仅首次，后续为毫秒级）",
                  flush=True)
        else:
            print(f"[GoAI] 未加载权重（随机初始化），仅用于流程验证。device={device}")

    def _infer_board_size(self, state):
        """从 policy 头输出层权重形状推断训练棋盘大小（输出维 = n²+1）。"""
        import math
        best = None
        for k, v in state.items():
            if "policy" in k and isinstance(v, torch.Tensor) and v.dim() == 2:
                n1 = v.shape[0]
                n = int(round(math.sqrt(max(n1 - 1, 0))))
                if n >= 5 and n * n + 1 == n1:
                    best = n
        return best

    # ------------------------------------------------------------------ #
    # 特征构造
    # ------------------------------------------------------------------ #
    def _build_state(self, board, my_hist, op_hist, to_play, planes=None):
        """用统一的 feature_planes 构造 12 通道状态张量。

        planes: 可选，预先算好的 12 通道 np.ndarray (12,H,W)。传入可避免重复
        feature_planes 计算（MCTS 增量特征场景）。
        """
        if planes is None:
            planes = board.feature_planes(my_hist, op_hist, to_play=to_play)
        x = torch.from_numpy(np.ascontiguousarray(planes)).unsqueeze(0).to(self.device).float()
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        return x

    # ------------------------------------------------------------------ #
    # 核心：模型前向 + 采样
    # ------------------------------------------------------------------ #
    def predict(self, board, my_hist, op_hist, to_play):
        """单局面前向。返回 (policy_np, value)。

        policy_np: shape=(bs*bs+1,) 概率（已 softmax）
        value    : float, 当前 to_play 视角 [-1,1]
        """
        x = self._build_state(board, my_hist, op_hist, to_play)
        policy, value = self._forward_batch(x)
        return policy[0], float(value[0].item())

    def predict_batch(self, states):
        """批量前向，MCTS 叶子评估的核心加速点。

        Args:
            states: list[(board, my_hist, op_hist, to_play)] 或
                    list[(None, my_hist, op_hist, to_play, planes)]（带预计算特征）
                    长度 B
        Returns:
            policies: np.ndarray (B, bs*bs+1) 已 softmax
            values : np.ndarray (B,) 当前 to_play 视角 [-1,1]
        """
        if not states:
            return np.zeros((0, self.board_size * self.board_size + 1)), np.zeros(0)
        xs = []
        for st in states:
            if len(st) == 5:
                # 增量特征：第 5 项为预计算 12 通道 planes
                b, mh, oh, tp, planes = st
                xs.append(self._build_state(b, mh, oh, tp, planes=planes))
            else:
                b, mh, oh, tp = st
                xs.append(self._build_state(b, mh, oh, tp))
        x = torch.cat(xs, dim=0)  # (B,12,H,W)
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        policies, values = self._forward_batch(x)
        return policies, values.squeeze(-1).cpu().numpy().astype(np.float32)

    def _forward_batch(self, x):
        """输入 (B,12,H,W)，输出 (policies_np(B,A), values(B,1))。"""
        B = x.shape[0]
        if self.is_npu and B > 1:
            # batch 归桶补零：稳定算子形状，命中 CANN 编译缓存
            bucket = next((b for b in self._NPU_BATCH_BUCKETS if b >= B), None)
            if bucket is not None and bucket > B:
                x = torch.cat([x, x.new_zeros(bucket - B, *x.shape[1:])], 0)
        with torch.no_grad():
            if self.use_amp:
                if self.is_npu:
                    # 兼容老 torch_npu：新 torch.autocast("npu") API 不可用时
                    # 回退 torch.npu.amp.autocast()
                    try:
                        with torch.autocast(device_type="npu", dtype=torch.float16):
                            policy_logits, value = self.model(x)
                    except (RuntimeError, AttributeError, TypeError):
                        with torch.npu.amp.autocast():
                            policy_logits, value = self.model(x)
                else:
                    with torch.cuda.amp.autocast():
                        policy_logits, value = self.model(x)
            else:
                policy_logits, value = self.model(x)
        policies = torch.softmax(policy_logits[:B], dim=-1).cpu().numpy()
        return policies, value[:B]

    # ------------------------------------------------------------------ #
    # CPU 推理加速：int8 动态量化 / ONNX Runtime 后端
    # ------------------------------------------------------------------ #
    def quantize_dynamic(self):
        """CPU int8 动态量化（Linear 层，x86 VNNI 收益明显）。

        须在 torch.compile 之前调用（量化编译后模型无意义）。失败时保持 fp32。
        """
        try:
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.model.eval()
            print("[GoAI] 已启用 CPU int8 动态量化 (Linear)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[GoAI] 动态量化失败，保持 fp32: {e}")
            return False

    def export_onnx(self, onnx_path):
        """导出 ONNX 并把推理后端切换到 onnxruntime（CPU 提速约 1.5-3x）。

        需 `pip install onnx onnxruntime`。失败时保持 torch 后端并返回 False。
        仅支持 CPU 推理（use_amp 自动失效）。
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("[GoAI] 未安装 onnx/onnxruntime（pip install onnx onnxruntime），保持 torch 后端")
            return False
        try:
            self.model = self.model.cpu().eval()
            dummy = torch.zeros(1, 12, self.board_size, self.board_size)
            torch.onnx.export(
                self.model, dummy, onnx_path,
                input_names=["x"], output_names=["policy", "value"],
                dynamic_axes={"x": {0: "batch"},
                              "policy": {0: "batch"},
                              "value": {0: "batch"}},
                opset_version=18)
            self._ort = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"])
            self._forward_batch = self._forward_batch_onnx  # 实例属性遮蔽方法
            print(f"[GoAI] 已切换 ONNX Runtime 推理后端: {onnx_path}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[GoAI] ONNX 导出失败，保持 torch 后端: {e}")
            return False

    def _forward_batch_onnx(self, x):
        """onnxruntime 后端。输入 (B,12,H,W)，返回值形状与 _forward_batch 对齐。

        导出捕获的是模型原始输出（policy 为 logits），此处补 softmax 与
        torch 后端（_forward_batch 内 softmax）对齐。
        """
        xnp = x.detach().cpu().numpy().astype(np.float32)
        pol, val = self._ort.run(None, {"x": xnp})
        pol = torch.from_numpy(np.asarray(pol))
        pol = torch.softmax(pol, dim=-1)
        return pol, torch.from_numpy(np.asarray(val)).reshape(-1, 1)

    def choose_move(self, board, my_hist, op_hist, to_play, legal_mask, temperature=1.0, topk=10):
        """根据策略分布与合法着法掩码，采样一个着法。

        返回 (move_int, is_pass, value)，move_int 为 board_size*board_size 表示虚着。
        """
        policy, value = self.predict(board, my_hist, op_hist, to_play)
        bs = self.board_size
        n_actions = bs * bs + 1

        illegal = np.ones(n_actions, dtype=bool)
        for m in legal_mask:
            illegal[m] = False
        # 始终允许虚着
        illegal[n_actions - 1] = False
        masked = policy.copy()
        masked[illegal] = 0.0
        s = masked.sum()
        if s <= 0:
            return n_actions - 1, True, value

        if temperature <= 0:
            # 贪心
            move_int = int(np.argmax(masked))
        else:
            probs = masked / s
            # 可选 top-k 截断，降低随机性
            if topk and topk < len(probs):
                idx = np.argsort(probs)[::-1][:topk]
                p2 = np.zeros_like(probs)
                p2[idx] = probs[idx]
                probs = p2 / p2.sum()
            # 温度缩放：对 log 缩放后重新 softmax
            if temperature != 1.0:
                logp = np.log(probs + 1e-12)
                probs = np.exp(logp / max(temperature, 1e-3))
                probs = probs / probs.sum()
            move_int = int(np.random.choice(len(probs), p=probs))

        is_pass = (move_int == n_actions - 1)
        return move_int, is_pass, value

    def choose_move_mcts(self, board, my_hist, op_hist, to_play, legal_mask,
                         simulations=400, temperature=1.0, mcts=None, num_threads=4,
                         use_rollout=False, rollout_lambda=0.25, path_moves=None):
        """用 MCTS 搜索选着法（推理提速核心）。返回 (move_int, is_pass, value)。

        use_rollout / rollout_lambda 启用 LightPLS：叶子价值融合轻量 rollout。
        path_moves: 从上次 MCTS 搜索根局面到当前的着法序列（GoBoard 编码，
            pass=-1），提供时复用上次搜索树的子树（访问/价值统计继承）。
        """
        from src.search.mcts import MCTS
        if mcts is None:
            mcts = MCTS(self, board_size=self.board_size, num_threads=num_threads,
                        temperature=temperature, use_rollout=use_rollout,
                        rollout_lambda=rollout_lambda)
        move_int, is_pass, value = mcts.best_move(
            board, my_hist, op_hist, to_play, simulations=simulations,
            temperature=temperature, return_value=True, path_moves=path_moves)
        # 若 MCTS 选了非法着法（极端情况下），回退到纯策略
        if move_int not in legal_mask and move_int != self.board_size * self.board_size:
            return self.choose_move(board, my_hist, op_hist, to_play, legal_mask,
                                    temperature=temperature)
        return move_int, is_pass, value

    # ------------------------------------------------------------------ #
    # 对局循环
    # ------------------------------------------------------------------ #
    def _move_int_to_coord(self, move_int):
        bs = self.board_size
        if move_int == bs * bs:
            return None  # pass
        r, c = divmod(move_int, bs)
        return (r, c)

    def self_play(self, num_games=1, max_moves=400, temperature=1.0, topk=10,
                   verbose=False, use_mcts=False, simulations=400, num_threads=4,
                   use_rollout=False, rollout_lambda=0.25):
        """模型自我对弈 num_games 局，返回每局结果（黑方视角 +1/-1）。

        use_mcts=True 时每步用 MCTS 搜索选点（棋力显著强于纯策略 argmax）。
        use_rollout=True 时叶子价值融合轻量 rollout（LightPLS）。
        """
        from src.search.mcts import MCTS
        mcts = MCTS(self, board_size=self.board_size, num_threads=num_threads,
                    temperature=temperature, use_rollout=use_rollout,
                    rollout_lambda=rollout_lambda) if use_mcts else None
        results = []
        for g in range(num_games):
            board = GoBoard(self.board_size)
            my_hist = [[-1, -1, -3], [-1, -1, -3]]  # 对当前执子方：最近3手
            passes = 0
            move_count = 0
            path_moves = []  # 自上次 MCTS 搜索根以来的着法序列（树复用），新局重置
            while passes < 2 and move_count < max_moves:
                to_play = board.current_player
                legal = board.get_legal_moves()
                if len(legal) == 0:
                    passes += 1
                    board.play(-1)
                    path_moves.append(-1)
                    move_count += 1
                    continue
                if use_mcts:
                    move_int, is_pass, value = self.choose_move_mcts(
                        board, my_hist[0], my_hist[1], to_play, legal,
                        simulations=simulations, temperature=temperature, mcts=mcts,
                        num_threads=num_threads, use_rollout=use_rollout,
                        rollout_lambda=rollout_lambda, path_moves=path_moves)
                else:
                    move_int, is_pass, value = self.choose_move(
                        board, my_hist[0], my_hist[1], to_play, legal,
                        temperature=temperature, topk=topk)
                if is_pass:
                    board.play(-1)
                    passes += 1
                    path_moves.append(-1)
                else:
                    r, c = self._move_int_to_coord(move_int)
                    board.play(r * self.board_size + c)
                    passes = 0
                    path_moves.append(r * self.board_size + c)
                    # 更新历史（最近3手，最新在末尾）
                    hist = my_hist[0] if to_play == 1 else my_hist[1]
                    hist.pop(0)
                    hist.append(r * self.board_size + c)
                move_count += 1
                if verbose and (move_count % 10 == 0):
                    print(f"  game {g} move {move_count} to_play={to_play} "
                          f"value={value:+.3f} pass={is_pass}")
            score = board.score()
            # 黑方(1)视角
            result = 1.0 if score > 0 else -1.0
            results.append(result)
            if verbose:
                print(f"game {g} finished: score(黑-白)={score:+.1f} -> "
                      f"{'黑胜' if result > 0 else '白胜'}")
        return results

    def play_against_human(self, human_color=1, max_moves=400, temperature=0.6,
                           use_mcts=False, simulations=400, num_threads=4,
                           use_rollout=False, rollout_lambda=0.25):
        """人机对弈，人类通过终端输入坐标（如 'ce' 或 'pass'）。"""
        from src.search.mcts import MCTS
        mcts = MCTS(self, board_size=self.board_size, num_threads=num_threads,
                    temperature=temperature, use_rollout=use_rollout,
                    rollout_lambda=rollout_lambda) if use_mcts else None
        board = GoBoard(self.board_size)
        my_hist = [[-1, -1, -3], [-1, -1, -3]]
        passes = 0
        move_count = 0
        path_moves = []  # 自上次 MCTS 搜索根以来的着法序列（树复用）
        while passes < 2 and move_count < max_moves:
            to_play = board.current_player
            legal = board.get_legal_moves()
            if to_play == human_color:
                print(board.to_string())
                print(f"合法着法数: {len(legal)}  | 输入坐标(如 ce)，或 pass，或 resign")
                inp = input("你的着法: ").strip().lower()
                if inp in ("pass", ""):
                    board.play(-1)
                    passes += 1
                    path_moves.append(-1)
                elif inp in ("resign", "quit"):
                    print("你认输。")
                    break
                else:
                    ok, mv = board.parse_move_str(inp, human_color)
                    if not ok or mv not in legal:
                        print("非法着法，请重试。")
                        continue
                    board.play(mv)
                    passes = 0
                    path_moves.append(mv)
                    hist = my_hist[0] if human_color == 1 else my_hist[1]
                    hist.pop(0); hist.append(mv)
            else:
                if len(legal) == 0:
                    board.play(-1); passes += 1
                    path_moves.append(-1)
                else:
                    if use_mcts:
                        move_int, is_pass, value = self.choose_move_mcts(
                            board, my_hist[0], my_hist[1], to_play, legal,
                            simulations=simulations, temperature=temperature, mcts=mcts,
                            num_threads=num_threads, use_rollout=use_rollout,
                            rollout_lambda=rollout_lambda, path_moves=path_moves)
                    else:
                        move_int, is_pass, value = self.choose_move(
                            board, my_hist[0], my_hist[1], to_play, legal,
                            temperature=temperature)
                    if is_pass:
                        board.play(-1); passes += 1
                        path_moves.append(-1)
                        print(f"AI 虚着 (value={value:+.3f})")
                    else:
                        r, c = self._move_int_to_coord(move_int)
                        mv = r * self.board_size + c
                        board.play(mv); passes = 0
                        path_moves.append(mv)
                        hist = my_hist[0] if to_play == 1 else my_hist[1]
                        hist.pop(0); hist.append(mv)
                        print(f"AI 落子 {chr(ord('a')+c)}{chr(ord('a')+r)} (value={value:+.3f})")
            move_count += 1
        print(board.to_string())
        score = board.score()
        print(f"终局 score(黑-白)={score:+.1f} -> {'黑胜' if score > 0 else '白胜'}")

    def analyze(self, max_moves=400, temperature=0.0):
        """交互式分析：从当前局面出发，展示 AI 的 top-k 候选着法。"""
        board = GoBoard(self.board_size)
        my_hist = [[-1, -1, -1], [-1, -1, -1]]
        print("逐步分析（输入坐标落子，pass 虚着，auto 让 AI 走，q 退出）")
        while True:
            to_play = board.current_player
            print(board.to_string())
            legal = board.get_legal_moves()
            policy, value = self.predict(board, my_hist[0], my_hist[1], to_play)
            bs = self.board_size
            ranked = []
            for m in legal:
                ranked.append((m, float(policy[m])))
            ranked.append((bs * bs, float(policy[bs * bs])))  # pass
            ranked.sort(key=lambda t: t[1], reverse=True)
            print(f"当前 to_play={to_play}  value={value:+.3f}")
            for m, p in ranked[:8]:
                if m == bs * bs:
                    print(f"  pass        p={p:.3f}")
                else:
                    r, c = divmod(m, bs)
                    print(f"  {chr(ord('a')+c)}{chr(ord('a')+r)} (idx {m:3d})  p={p:.3f}")
            inp = input("> ").strip().lower()
            if inp in ("q", "quit"):
                break
            if inp == "auto":
                mv = ranked[0][0]
                if mv == bs * bs:
                    board.play(-1)
                else:
                    board.play(mv)
                    hist = my_hist[0] if to_play == 1 else my_hist[1]
                    hist.pop(0); hist.append(mv)
                continue
            if inp in ("pass", ""):
                board.play(-1); continue
            ok, mv = board.parse_move_str(inp, to_play)
            if not ok or int(mv) not in legal.tolist():
                print("非法，重试。"); continue
            board.play(mv)
            hist = my_hist[0] if to_play == 1 else my_hist[1]
            hist.pop(0); hist.append(mv)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="围棋 SFT 模型推理 / 对弈")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--mode", type=str, default="selfplay",
                        choices=["selfplay", "human", "analyze"])
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--human-color", type=int, default=1)
    parser.add_argument("--attention-mode", default="mix", choices=["none", "mix", "all"])
    parser.add_argument("--num-attention-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--policy-channels", type=int, default=32,
                        help="policy 头隐层通道（须与训练时一致，训练默认 32）")
    parser.add_argument("--value-channels", type=int, default=16,
                        help="value 头隐层通道（须与训练时一致，训练默认 16）")
    parser.add_argument("--attn-mode", default="global",
                        choices=["global", "window", "axial"],
                        help="注意力计算模式: global=全配对, window=滑动窗口, axial=轴向")
    parser.add_argument("--attn-window", type=int, default=7, help="window 模式窗口边长")
    parser.add_argument("--compile", action="store_true", help="用 torch.compile 融合算子（GPU 提速）")
    parser.add_argument("--tf32", action="store_true",
                        help="CUDA tf32 matmul（V100/Amp 上约 2-4x fp32 提速，精度损失可忽略）")
    parser.add_argument("--use-mcts", action="store_true",
                        help="用 MCTS 搜索选点（棋力显著强于纯策略 argmax）")
    parser.add_argument("--simulations", type=int, default=400,
                        help="MCTS 每步模拟次数（V100S 上 400~800 仅 1-2s/步）")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="MCTS 并行模拟线程数（虚拟损失）")
    parser.add_argument("--use-rollout", action="store_true",
                        help="LightPLS：叶子价值融合轻量 rollout（Tromp-Taylor 快数子）")
    parser.add_argument("--rollout-lambda", type=float, default=0.25,
                        help="LightPLS rollout 价值权重 (0=只用网络, 1=只用 rollout)")
    args = parser.parse_args()

    ai = GoAI(model_path=args.model, board_size=args.board_size,
              device=args.device, use_amp=args.use_amp,
              attention_mode=args.attention_mode,
              num_attention_layers=args.num_attention_layers,
              num_heads=args.num_heads,
              attention_dropout=args.attention_dropout,
              policy_channels=args.policy_channels,
              value_channels=args.value_channels,
              attn_mode=args.attn_mode,
              attn_window=args.attn_window,
              compile=args.compile, tf32=args.tf32)
    if args.mode == "selfplay":
        res = ai.self_play(num_games=args.games, temperature=args.temperature, topk=args.topk,
                           use_mcts=args.use_mcts, simulations=args.simulations,
                           num_threads=args.num_threads,
                           use_rollout=args.use_rollout, rollout_lambda=args.rollout_lambda)
        wr = sum(1 for r in res if r > 0) / max(len(res), 1)
        tag = " (MCTS"
        if args.use_rollout:
            tag += "+LightPLS"
        tag += ")"
        print(f"自对弈 {len(res)} 局，黑方胜率 {wr:.2%}"
              f"{tag if args.use_mcts else ''}")
    elif args.mode == "human":
        ai.play_against_human(human_color=args.human_color, temperature=args.temperature,
                              use_mcts=args.use_mcts, simulations=args.simulations,
                              num_threads=args.num_threads,
                              use_rollout=args.use_rollout, rollout_lambda=args.rollout_lambda)
    elif args.mode == "analyze":
        ai.analyze(temperature=args.temperature)


if __name__ == "__main__":
    main()
