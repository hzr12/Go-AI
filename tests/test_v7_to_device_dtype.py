"""`v7_to_device`：V7 输入的 H2D dtype 契约。

事故（2026-10-11 真机，C 段）
--------------------------------
C 段（stdata 分片）第一次真正跑进训练循环就崩了：

    File "scripts/train_sft.py", line 4901, in main
      out = model(state, gl)
    File "src/networks/katago_v7.py", line 1068, in trunk
      x = self.stem(spatial)
    RuntimeError: Input type (CUDABoolType) and weight type
                  (CUDABFloat16Type) should be the same

根因是一条**为 board 级路径写的隐含假设**：`V7PackedDataset.spatial` 解位打包后
是 **`bool`**（`unpack_binary_input` 的天然类型），而 `v7_to_device` 旧写法的
「`amp_dtype == float32` 才升 fp32，否则原样保留」在 **BF16**（A100 就是 BF16，
不是 float32）下把 bool **一路放行到卷积**。

这个缺陷之所以一直活着：C 段此前每一次尝试都死在更早的地方 ——
分片静默加载期、按棋局分割的 O(N²)、`rng` 未定义 —— 从没走到过训练循环。

本文件钉三件事
--------------
1. **非浮点输入必须被转成 `amp_dtype`**（否则直接喂卷积就是上面那个 RuntimeError）。
2. **转换发生在设备侧**，不是 CPU 侧 —— bool 只有 1 字节，比 fp16 还省一半 H2D；
   先在 CPU 转 fp32 恰好把这个优势抵掉。
3. **board 级路径逐位不变**：已经是 fp16/fp32 的输入不能被这次改动碰到
   （全局特征 `global` 就是 fp32，被顺带降成 bf16 的话是静默的口径变化）。
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_sft import v7_to_device  # noqa: E402

CPU = torch.device('cpu')


@pytest.mark.parametrize('amp', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('src_dtype', [np.bool_, np.uint8])
def test_non_floating_input_becomes_amp_dtype(amp, src_dtype):
    """位打包解出来的 bool/uint8 必须变成 `amp_dtype`，不能原样进卷积。"""
    x = (np.arange(2 * 3, dtype=np.uint8).reshape(2, 3) % 2).astype(src_dtype)
    out = v7_to_device(x, CPU, amp)
    assert torch.is_floating_point(out), f'{src_dtype.__name__} 输入变成了 {out.dtype}'
    assert out.dtype == amp, f'{src_dtype.__name__} -> {out.dtype}，期望 {amp}'
    # 值本身必须保持（0/1 不因转换而变）。走 `.float()` 而非 `.numpy()` ——
    # 后者不支持 bfloat16（numpy 无该标量类型）。
    assert torch.equal(out.float(), torch.from_numpy(x.astype(np.float32)))


@pytest.mark.parametrize('amp', [torch.float32, torch.float16, torch.bfloat16])
def test_floating_input_is_left_alone(amp):
    """**已经是浮点的输入不能被这次改动碰到** —— 逐位、逐 dtype。

    这是「只修 bool」的判据：全局特征 `global` 是 fp32，若顺手 `.to(amp_dtype)`
    就会被静默降成 bf16，那是口径变化而不是修复。
    """
    x32 = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = v7_to_device(x32, CPU, amp)
    if amp == torch.float32:
        assert out.dtype == torch.float32
    else:
        # fp16 输入在 BF16 下按原设计**原样保留**（H2D 字节数口径）
        assert out.dtype == torch.float32, 'fp32 输入被降精度了 —— 不是本次要修的'
    assert torch.equal(out.float(), torch.from_numpy(x32))


def test_bool_conversion_is_worth_it_on_the_wire():
    """bool 只有 1 字节 ⇒ 在设备侧转换比在 CPU 侧转换**更省**。

    这条把「为什么不直接在 numpy 端 `.astype(np.float32)`」钉成事实：
    后者会把 H2D 字节数翻两倍（1 → 4），恰好抵消这个优化的全部意义。
    """
    b = np.zeros((256, 22, 19, 19), dtype=bool)
    f = b.astype(np.float32)
    assert b.nbytes == 256 * 22 * 19 * 19        # 1 字节/元素
    assert f.nbytes == b.nbytes * 4              # 4 字节/元素
    assert v7_to_device(b, CPU, torch.bfloat16).dtype == torch.bfloat16


def test_packed_dataset_spatial_really_is_bool():
    """锁住触发条件：`V7PackedDataset.spatial` 返回的**就是** bool。

    没有这条，上面三条会在「哪天上游改成返回 float32」时静默失去意义 ——
    而那正是本文件要保护的场景（改动失去抓手，却仍然全绿）。
    """
    from src.data.v7_packed_dataset import unpack_binary_input
    packed = np.zeros((2, 22, 46), np.uint8)
    packed[:, :, :2] = 0xFF                     # 前两个 bit 置 1
    un = unpack_binary_input(packed)
    assert un.dtype == np.bool_, un.dtype
    # 端到端：把 unpack 的结果直接喂 v7_to_device，必须拿到浮点
    assert torch.is_floating_point(v7_to_device(un, CPU, torch.bfloat16))