"""从 .pth 反推 AlphaGoNet 的完整架构配置，并给出可直接用的 train_sft.py 参数。

为什么需要这个
--------------
`save_model()` 只保存 `model.state_dict()`，**不保存任何配置**。于是
`--backbone-channels` / `--res-blocks` / `--value-res-blocks` 这些决定结构的
参数在 checkpoint 里彻底消失，日志一旦丢失就只能靠猜——run.txt 里的
V17 记录「Value: 1.01M (96ch, 11 blocks)」就是这么来的：1.01M 其实对应
96ch + 5 blocks，文字与数字自相矛盾。

实测 V12：反推出的 64ch x 2 blocks / 32ch x 2 层 / 192ch x 17 块 mix，
参数量 12.67M，与 run.txt 记录吻合。

已知限制
--------
board_size **无法**从 state_dict 读出：policy 头是 conv 到 1 通道后 flatten，
输出维度不出现在权重形状里。需通过 --board-size 指定（默认 19）。
"""

import argparse
import math
import re
import sys

import torch

sys.path.insert(0, __file__.rsplit('scripts', 1)[0])
from src.networks.alphanet import AlphaGoNet  # noqa: E402


def remap_legacy_value_keys(sd):
    """把旧版 ValueNetwork 的键名归一到当前代码。

    当前 src/networks/value_network.py:79-85 用
        self.res_blocks = nn.ModuleList([...])      -> value.res_blocks.0.*
    早期版本直接在 value 下挂 res1 / res2 / ...      -> value.res1.*

    V12 checkpoint 就是旧命名，直接 load_state_dict(strict=True) 会因
    「ckpt 缺失 value.res_blocks.0.* / ckpt 多余 value.res1.*」而失败。
    这里做纯键名重映射，不动任何数值。
    """
    # 已是当前命名则无需处理
    if any(k.startswith('value.res_blocks.') for k in sd):
        return sd, 0
    # 没有旧版键也没得可映射
    if not any(re.match(r'^value\.res\d+\.', k) for k in sd):
        return sd, 0
    out = {}
    moved = 0
    pat = re.compile(r'^value\.res(\d+)\.(.+)$')
    for k, v in sd.items():
        m = pat.match(k)
        if m:
            out['value.res_blocks.%d.%s' % (int(m.group(1)) - 1, m.group(2))] = v
            moved += 1
        else:
            out[k] = v
    return out, moved


def infer_config(sd, board_size=19):
    """从 state_dict 推断架构配置字典。"""
    cfg = {}

    # ---- backbone 通道与输入通道 ----
    w = sd['backbone.conv1.weight']
    cfg['backbone_channels'] = int(w.shape[0])
    cfg['in_channels'] = int(w.shape[1])

    # ---- 逐块指纹，判定块类型 ----
    idxs = sorted({int(m.group(1)) for k in sd
                   if (m := re.match(r'backbone\.blocks\.(\d+)\.', k))})
    n_blocks = (max(idxs) + 1) if idxs else 0
    kinds = []
    for i in range(n_blocks):
        pref = 'backbone.blocks.%d.' % i
        sub = [k[len(pref):] for k in sd if k.startswith(pref)]
        # 三种块的键名互不重叠，可直接用特征键判定：
        #   AttentionResBlock -> attn.qkv / attn.ffn / conv.conv1（conv 是嵌套子模块）
        #   ConvNeXtBlock     -> dwconv / norm.norm / pwconv1 / pwconv2
        #   ResBlock          -> conv1 / conv2 / bn1 / bn2（扁平）
        if any(s.startswith('attn.') for s in sub):
            kinds.append('attn')
        elif any(s.startswith('dwconv.') for s in sub):
            kinds.append('convnext')
        else:
            kinds.append('res')

    n_attn = kinds.count('attn')
    n_cnx = kinds.count('convnext')
    n_res = kinds.count('res')

    # 分段模式：res 全在前面、然后 convnext、再 attn，且 res_blocks>0
    # mix 模式：attention 穿插在 res 之间
    if n_res and n_cnx:
        seg = (kinds == ['res'] * n_res + ['convnext'] * n_cnx + ['attn'] * n_attn)
        if seg:
            cfg.update(res_blocks=n_res, convnext_blocks=n_cnx,
                       attn_blocks=n_attn, backbone_res_blocks=n_blocks)
            cfg['_mode'] = 'segmented'
        else:
            cfg.update(backbone_res_blocks=n_blocks, res_blocks=0,
                       convnext_blocks=0, attn_blocks=0,
                       num_attention_layers=n_attn)
            cfg['_mode'] = 'mix'
    elif n_cnx and not n_res:
        cfg.update(backbone_res_blocks=n_blocks, res_blocks=0,
                   convnext_blocks=0, attn_blocks=0,
                   num_attention_layers=0)
        cfg['_mode'] = 'convnext-all'
    else:
        cfg.update(backbone_res_blocks=n_blocks, res_blocks=0,
                   convnext_blocks=0, attn_blocks=0,
                   num_attention_layers=n_attn)
        cfg['_mode'] = 'mix'

    # ---- value head ----
    cfg['value_channels'] = int(sd['value.downsample.0.weight'].shape[0])
    # 兼容两种命名：当前的 res_blocks.N 与旧版的 resN（1 起）
    vblocks = {int(m.group(1)) for k in sd
               if (m := re.match(r'^value\.res_blocks\.(\d+)\.', k))}
    vblocks |= {int(m.group(1)) - 1 for k in sd
                if (m := re.match(r'^value\.res(\d+)\.', k))}
    cfg['value_res_blocks'] = len(vblocks)

    # ---- policy head ----
    cfg['policy_channels'] = int(sd['policy.conv1.weight'].shape[0])
    c2 = sd['policy.conv2.weight']
    # 2 层结构 conv1 1x1(P) -> conv2 1x1(P->1)，故 conv2 形状是 (1, P, 1, 1)
    # 3 层结构 conv1 1x1(P) -> conv2 3x3(P->P) -> conv3 1x1(P->1)，
    # 此时 conv2 形状是 (P, P, 3, 3)
    cfg['policy_layers'] = 2 if int(c2.shape[0]) == 1 else 3

    cfg['board_size'] = board_size
    cfg['action_size'] = board_size * board_size + 1
    return cfg, kinds


def verify(cfg, sd, verbose=True):
    """用推断出的配置重建模型并严格加载，是反推正确性的硬证据。"""
    model = AlphaGoNet(
        in_channels=cfg['in_channels'],
        action_size=cfg['action_size'],
        backbone_channels=cfg['backbone_channels'],
        backbone_res_blocks=cfg['backbone_res_blocks'],
        attention_mode=cfg.get('attention_mode', 'mix'),
        num_attention_layers=cfg.get('num_attention_layers', 4),
        num_heads=4, attention_dropout=0.0,
        attn_mode=cfg.get('attn_mode', 'window_global'),
        attn_window=cfg.get('attn_window', 5),
        res_blocks=cfg.get('res_blocks', 0),
        convnext_blocks=cfg.get('convnext_blocks', 0),
        attn_blocks=cfg.get('attn_blocks', 0),
        value_channels=cfg['value_channels'],
        value_res_blocks=cfg['value_res_blocks'],
        policy_channels=cfg['policy_channels'],
        policy_layers=cfg['policy_layers'],
        arch='resnet',
    )
    # strict=False 只能容忍「缺键/多键」，**尺寸不匹配仍会抛 RuntimeError**，
    # 所以先自行比对形状，给出可读诊断，再决定是否真的加载。
    model_sd = model.state_dict()
    missing = [k for k in model_sd if k not in sd]
    unexpected = [k for k in sd if k not in model_sd]
    mismatched = [(k, tuple(sd[k].shape), tuple(model_sd[k].shape))
                  for k in model_sd
                  if k in sd and tuple(sd[k].shape) != tuple(model_sd[k].shape)]
    ok = not missing and not unexpected and not mismatched
    n = sum(p.numel() for p in model.parameters())
    if verbose:
        print()
        print('=== 重建校验 ===')
        print('  参数量      : {:,}'.format(n))
        print('  形状比对    : {}'.format('完全一致' if ok else '存在差异'))
        for k, got, exp in mismatched[:8]:
            print('    尺寸不符  {}: ckpt {} vs 重建 {}'.format(k, got, exp))
        if missing:
            print('    ckpt 缺失  {}'.format(missing[:5]))
        if unexpected:
            print('    ckpt 多余  {}'.format(unexpected[:5]))
        if ok:
            # 形状全对再真加载一次，确认键集合也严格一致
            model.load_state_dict(sd, strict=True)
    return ok, n


def main():
    ap = argparse.ArgumentParser(description='从 .pth 反推架构配置')
    ap.add_argument('ckpt')
    ap.add_argument('--board-size', type=int, default=19,
                    help='棋盘大小。state_dict 里读不出（policy 头 flatten '
                         '后输出维度不进权重形状），需显式指定。默认 19。')
    ap.add_argument('--emit-flags', action='store_true',
                    help='打印可直接粘贴的 train_sft.py 参数')
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location='cpu')
    if not isinstance(sd, dict) or not any(k.startswith('backbone.') for k in sd):
        print('这不是 AlphaGoNet 的 state_dict（顶层键: {}）'.format(list(sd)[:5]))
        sys.exit(1)

    sd, moved = remap_legacy_value_keys(sd)
    if moved:
        print('注意：检测到旧版 ValueNetwork 键名（value.res1/res2/...），'
              '已重映射为当前的 value.res_blocks.0/1/...（共 %d 个张量）。'
              '原始 checkpoint 未被修改。\n' % moved)

    cfg, kinds = infer_config(sd, args.board_size)

    print('=== 从 {} 推断出的架构 ==='.format(args.ckpt))
    print('  构建模式      : {}'.format(cfg['_mode']))
    print('  in_channels   : {}'.format(cfg['in_channels']))
    print('  backbone_channels : {}'.format(cfg['backbone_channels']))
    print('  块数          : {}'.format(len(kinds)))
    print('  块序列        : {}'.format(
        ''.join({'res': 'R', 'convnext': 'C', 'attn': 'A'}[k] for k in kinds)))
    if cfg['_mode'] == 'segmented':
        print('  res_blocks    : {}  convnext_blocks: {}  attn_blocks: {}'.format(
            cfg['res_blocks'], cfg['convnext_blocks'], cfg['attn_blocks']))
    else:
        print('  num_attention_layers : {}'.format(cfg['num_attention_layers']))
    print('  value_channels: {}   value_res_blocks: {}'.format(
        cfg['value_channels'], cfg['value_res_blocks']))
    print('  policy_channels: {}  policy_layers: {}'.format(
        cfg['policy_channels'], cfg['policy_layers']))
    print('  board_size    : {} (命令行指定，state_dict 内无此信息)'.format(
        cfg['board_size']))

    ok, n = verify(cfg, sd)

    if args.emit_flags:
        print()
        print('=== 可直接用于 train_sft.py 的结构参数 ===')
        print('  --board-size {} \\'.format(cfg['board_size']))
        print('  --backbone-channels {} --backbone-res-blocks {} \\'.format(
            cfg['backbone_channels'], cfg['backbone_res_blocks']))
        if cfg['_mode'] == 'segmented':
            print('  --res-blocks {} --convnext-blocks {} --attn-blocks {} \\'.format(
                cfg['res_blocks'], cfg['convnext_blocks'], cfg['attn_blocks']))
        else:
            print('  --num-attention-layers {} \\'.format(cfg['num_attention_layers']))
        print('  --value-channels {} --value-res-blocks {} \\'.format(
            cfg['value_channels'], cfg['value_res_blocks']))
        print('  --policy-channels {} --policy-layers {}'.format(
            cfg['policy_channels'], cfg['policy_layers']))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
