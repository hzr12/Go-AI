#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""torch .pth  <->  MindSpore .ckpt 权重双向转换（以 npz 为中间格式）。

为什么要绕一道 npz
------------------
你的两台机器环境不对等：
  - A100 机：有 torch，**没有 MindSpore**
  - 910B 机：有 MindSpore，**没有 torch_npu（也没有 torch）**
所以不能在单机上同时 import 两个框架。本脚本拆成 4 个单向子命令，
每次只依赖其中一个框架，中间产物是纯 numpy 的 .npz（可随意跨机拷贝）。

npz 内部统一使用 **torch 风格命名**（weight / bias / running_mean / running_var），
写入 MindSpore 时再按规则转成 gamma / beta / moving_mean / moving_variance。

参数命名映射（MindSpore -> torch）
--------------------------------
    Conv2d.weight        -> Conv2d.weight        （形状相同，不转置）
    Dense.weight         -> Linear.weight        （形状相同 (out,in)，不转置）
    BatchNorm.gamma      -> BatchNorm.weight
    BatchNorm.beta       -> BatchNorm.bias
    BatchNorm.moving_mean      -> running_mean
    BatchNorm.moving_variance  -> running_var
    LayerNorm.gamma/beta -> weight/bias

用法
----
# ① A100 机（torch）：pth -> npz
python scripts/convert_ckpt.py to-npz  --in models/sft_19x19_v4.pth --out v4.npz --framework torch

# ② 910B 机（MindSpore）：npz -> ckpt
python scripts/convert_ckpt.py from-npz --in v4.npz --out models/sft_19x19_v4.ckpt --framework ms

# ③ 910B 机（MindSpore）：ckpt -> npz
python scripts/convert_ckpt.py to-npz  --in models/az_iter3.ckpt --out az.npz --framework ms

# ④ A100 机（torch）：npz -> pth
python scripts/convert_ckpt.py from-npz --in az.npz --out models/az_iter3.pth --framework torch
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 归一化层模块名（torch 侧）：这些模块的 weight/bias 对应 MS 的 gamma/beta
_NORM_MODS = ("bn1", "bn2", "bn_out", "ln1", "ln2")


def _is_norm_param(torch_name: str) -> bool:
    """判断 torch 名是否为 BN/LayerNorm 的 weight/bias。

    依据模块名：...bn1.weight / ...ln2.bias / policy_head.1.weight（Sequential 里
    index 1 是 BatchNorm2d）/ value_head.1.weight。
    """
    parts = torch_name.split(".")
    if len(parts) < 2:
        return False
    mod = parts[-2]
    if mod in _NORM_MODS:
        return True
    if mod == "1" and ("policy_head" in torch_name or "value_head" in torch_name):
        return True
    return False


def torch_name_to_ms(name: str) -> str:
    """torch 风格名 -> MindSpore 风格名。"""
    if name.endswith(".running_mean"):
        return name[: -len(".running_mean")] + ".moving_mean"
    if name.endswith(".running_var"):
        return name[: -len(".running_var")] + ".moving_variance"
    if name.endswith(".weight") and _is_norm_param(name):
        return name[: -len(".weight")] + ".gamma"
    if name.endswith(".bias") and _is_norm_param(name):
        return name[: -len(".bias")] + ".beta"
    return name


def ms_name_to_torch(name: str) -> str:
    """MindSpore 风格名 -> torch 风格名（后缀自解释，无歧义）。"""
    if name.endswith(".moving_mean"):
        return name[: -len(".moving_mean")] + ".running_mean"
    if name.endswith(".moving_variance"):
        return name[: -len(".moving_variance")] + ".running_var"
    if name.endswith(".gamma"):
        return name[: -len(".gamma")] + ".weight"
    if name.endswith(".beta"):
        return name[: -len(".beta")] + ".bias"
    return name


# --------------------------------------------------------------------------- #
# torch 侧
# --------------------------------------------------------------------------- #
def torch_load(pth_path):
    import torch
    sd = torch.load(pth_path, map_location="cpu")
    if hasattr(sd, "state_dict"):          # 完整模型对象
        sd = sd.state_dict()
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def torch_save(arrays, out_path):
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
    torch.save(sd, out_path)


# --------------------------------------------------------------------------- #
# MindSpore 侧
# --------------------------------------------------------------------------- #
def ms_load(ckpt_path):
    """读 .ckpt，返回 torch 风格命名的 {name: ndarray}。"""
    import mindspore as ms
    raw = ms.load_checkpoint(ckpt_path)
    out = {}
    for k, v in raw.items():
        arr = v.asnumpy() if hasattr(v, "asnumpy") else np.asarray(v)
        out[ms_name_to_torch(k)] = arr
    return out


def ms_save(arrays, out_path):
    """以 torch 风格命名的 {name: ndarray} 写 .ckpt。

    直接构造 save_checkpoint 需要的 param_list，无需实例化网络
    （因此不需要知道 backbone 配置）。
    """
    import mindspore as ms
    from mindspore import Tensor
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    param_list = []
    for k, v in arrays.items():
        param_list.append({"name": torch_name_to_ms(k),
                           "data": Tensor(np.ascontiguousarray(v))})
    ms.save_checkpoint(param_list, out_path)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="torch <-> MindSpore 权重转换")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("to-npz", help="框架格式 -> 中间 npz")
    p1.add_argument("--in", dest="src", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--framework", choices=["torch", "ms"], required=True)

    p2 = sub.add_parser("from-npz", help="中间 npz -> 框架格式")
    p2.add_argument("--in", dest="src", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--framework", choices=["torch", "ms"], required=True)

    args = ap.parse_args()

    if args.cmd == "to-npz":
        arrays = torch_load(args.src) if args.framework == "torch" else ms_load(args.src)
        np.savez(args.out, **arrays)
        n = sum(a.size for a in arrays.values())
        print(f"[to-npz] {len(arrays)} 个张量 / {n / 1e6:.2f}M 参数 -> {args.out}")

    else:  # from-npz
        d = np.load(args.src)
        arrays = {k: d[k] for k in d.files}
        if args.framework == "torch":
            torch_save(arrays, args.out)
        else:
            ms_save(arrays, args.out)
        n = sum(a.size for a in arrays.values())
        print(f"[from-npz] {len(arrays)} 个张量 / {n / 1e6:.2f}M 参数 -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
