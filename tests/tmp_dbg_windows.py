"""定位 new vs F.unfold 第一个不一致元素。验证后删除。"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, Hh, ws, H, W, d = 1, 1, 3, 3, 3, 1
pad = ws // 2
C = Hh * d
t = torch.arange(C * H * W, dtype=torch.float).reshape(B, C, H, W)
print('input t[0,0]:\n', t[0, 0])

qu = F.unfold(t, kernel_size=ws, padding=pad)            # (B, C*ws^2, N)
print('F.unfold shape:', tuple(qu.shape))
ref = qu.view(B, Hh, d, H * W, ws * ws).permute(0, 3, 1, 4, 2) \
        .reshape(B * H * W, Hh, ws * ws, d)

tp = F.pad(t, (pad, pad, pad, pad))
tv = tp.unfold(2, ws, 1).unfold(3, ws, 1)
print('tv shape:', tuple(tv.shape))
nv = tv.reshape(B, Hh, d, H, W, ws, ws) \
       .permute(0, 3, 4, 1, 5, 6, 2) \
       .reshape(B * H * W, Hh, ws * ws, d)

neq = (nv != ref)
idx = neq.nonzero()
print('num mismatched elements:', idx.shape[0], '/', nv.numel())
if idx.shape[0]:
    n0, h0, w0, dd0 = [int(x) for x in idx[0]]
    i0, j0 = n0 // W, n0 % W
    kh0, kw0 = w0 // ws, w0 % ws
    print(f'first mismatch: n={n0}(i={i0},j={j0}) h={h0} w={w0}(kh={kh0},kw={kw0}) d={dd0}')
    print(f'  new={nv[n0,h0,w0,dd0].item()}  ref={ref[n0,h0,w0,dd0].item()}')
    print(f'  new 窗口 n={n0}:\n', nv[n0, 0, :, :, 0].reshape(ws, ws))
    print(f'  ref 窗口 n={n0}:\n', ref[n0, 0, :, :, 0].reshape(ws, ws))
    print(f'  F.unfold 原始 col n={n0}:\n', qu[0, :, n0].reshape(ws, ws))
