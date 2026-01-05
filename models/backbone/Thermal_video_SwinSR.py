import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange

# =========================================================
# MLP
# =========================================================
class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

# =========================================================
# Window utils
# =========================================================
def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0], window_size[0],
        H // window_size[1], window_size[1],
        W // window_size[2], window_size[2],
        C
    )
    return x.permute(0,1,3,5,2,4,6,7).contiguous() \
            .view(-1, reduce(mul, window_size), C)

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        *window_size, -1
    )
    return x.permute(0,1,4,2,5,3,6,7).contiguous() \
            .view(B, D, H, W, -1)

def get_window_size(x_size, window_size, shift_size):
    ws, ss = list(window_size), list(shift_size)
    for i in range(3):
        if x_size[i] <= window_size[i]:
            ws[i] = x_size[i]
            ss[i] = 0
    return tuple(ws), tuple(ss)

# =========================================================
# Attention
# =========================================================
class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2*window_size[0]-1)*(2*window_size[1]-1)*(2*window_size[2]-1),
                num_heads
            )
        )

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size[0]),
            torch.arange(window_size[1]),
            torch.arange(window_size[2]),
            indexing="ij"
        ))
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1,2,0)
        relative_coords[...,0] += window_size[0]-1
        relative_coords[...,1] += window_size[1]-1
        relative_coords[...,2] += window_size[2]-1
        relative_coords[...,0] *= (2*window_size[1]-1)*(2*window_size[2]-1)
        relative_coords[...,1] *= (2*window_size[2]-1)
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads)
        q, k, v = qkv.permute(2,0,3,1,4)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index[:N,:N].reshape(-1)
        ].reshape(N, N, -1).permute(2,0,1)

        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1,2).reshape(B_, N, C))

# =========================================================
# Swin Block
# =========================================================
class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x, mask):
        shortcut = x
        x = self.norm1(x)

        B, D, H, W, C = x.shape
        ws, ss = get_window_size((D,H,W), self.window_size, self.shift_size)

        pad = [(ws[i]-x.shape[i+1]%ws[i])%ws[i] for i in range(3)]
        x = F.pad(x, (0,0,0,pad[2],0,pad[1],0,pad[0]))

        if any(ss):
            x = torch.roll(x, shifts=(-ss[0],-ss[1],-ss[2]), dims=(1,2,3))

        x_windows = window_partition(x, ws)
        x_windows = self.attn(x_windows, mask)
        x_windows = x_windows.view(-1,*ws,C)

        x = window_reverse(x_windows, ws, B, *x.shape[1:4])

        if any(ss):
            x = torch.roll(x, shifts=ss, dims=(1,2,3))

        x = x[:, :D, :H, :W, :]
        x = shortcut + x
        return x + self.mlp(self.norm2(x))

# =========================================================
# Mask
# =========================================================
@lru_cache()
def compute_mask(D,H,W,window_size,shift_size,device):
    img_mask = torch.zeros((1,D,H,W,1), device=device)
    cnt = 0
    for d in (slice(-window_size[0]), slice(-window_size[0],-shift_size[0]), slice(-shift_size[0],None)):
        for h in (slice(-window_size[1]), slice(-window_size[1],-shift_size[1]), slice(-shift_size[1],None)):
            for w in (slice(-window_size[2]), slice(-window_size[2],-shift_size[2]), slice(-shift_size[2],None)):
                img_mask[:,d,h,w,:] = cnt
                cnt += 1
    mask = window_partition(img_mask, window_size).squeeze(-1)
    mask = mask.unsqueeze(1) - mask.unsqueeze(2)
    return mask.masked_fill(mask!=0, float(-100.0))

# =========================================================
# PatchEmbed (NO DOWNSAMPLING)
# =========================================================
class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, 3, 1, 1)

    def forward(self, x):
        return self.proj(x)

# =========================================================
# FINAL BACKBONE (SR SAFE)
# =========================================================
class SwinTransformer3D(nn.Module):
    def __init__(
        self,
        in_chans=1,
        embed_dim=96,
        depths=(2,2,6,2),
        num_heads=(3,6,12,24),
        window_size=(2,4,4)
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim)
        self.stages = nn.ModuleList()

        for i in range(len(depths)):
            self.stages.append(nn.ModuleList([
                SwinTransformerBlock3D(
                    embed_dim, num_heads[i],
                    window_size,
                    (0,0,0) if j%2==0 else tuple(ws//2 for ws in window_size)
                ) for j in range(depths[i])
            ]))

        self.num_features = embed_dim

    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            for blk in stage:
                B,C,D,H,W = x.shape
                ws, ss = get_window_size((D,H,W), blk.window_size, blk.shift_size)
                mask = compute_mask(D,H,W,ws,ss,x.device)
                x = rearrange(x,'b c d h w->b d h w c')
                x = blk(x, mask)
                x = rearrange(x,'b d h w c->b c d h w')
        return x


