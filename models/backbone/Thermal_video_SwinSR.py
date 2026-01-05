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
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

# =========================================================
# Window utilities
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
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, reduce(mul, window_size), C)
    )
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0], window_size[1], window_size[2],
        -1
    )
    x = (
        x.permute(0, 1, 4, 2, 5, 3, 6, 7)
        .contiguous()
        .view(B, D, H, W, -1)
    )
    return x

def get_window_size(x_size, window_size, shift_size):
    use_window = list(window_size)
    use_shift = list(shift_size)
    for i in range(3):
        if x_size[i] <= window_size[i]:
            use_window[i] = x_size[i]
            use_shift[i] = 0
    return tuple(use_window), tuple(use_shift)

# =========================================================
# Attention
# =========================================================
class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads
            )
        )

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size[0]),
                torch.arange(window_size[1]),
                torch.arange(window_size[2]),
                indexing="ij"
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)

        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ].reshape(N, N, -1).permute(2, 0, 1)

        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)

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
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )

        pad = [(window_size[i] - x.shape[i+1] % window_size[i]) % window_size[i] for i in range(3)]
        x = F.pad(x, (0, 0, 0, pad[2], 0, pad[1], 0, pad[0]))

        if any(shift_size):
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

        x_windows = window_partition(x, window_size)
        x_windows = self.attn(x_windows, mask)
        x_windows = x_windows.view(-1, *window_size, C)

        x = window_reverse(x_windows, window_size, B, x.shape[1], x.shape[2], x.shape[3])

        if any(shift_size):
            x = torch.roll(x, shifts=shift_size, dims=(1, 2, 3))

        x = x[:, :D, :H, :W, :]
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

# =========================================================
# Mask (FIXED)
# =========================================================
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for h in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for w in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0))

# =========================================================
# Patch layers
# =========================================================
class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1,4,4), stride=(1,4,4))

    def forward(self, x):
        return self.proj(x)

class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = torch.cat([
            x[:, :, 0::2, 0::2],
            x[:, :, 1::2, 0::2],
            x[:, :, 0::2, 1::2],
            x[:, :, 1::2, 1::2]
        ], dim=-1)
        x = self.reduction(self.norm(x))
        return rearrange(x, 'b t h w c -> b c t h w')

# =========================================================
# FINAL BACKBONE
# =========================================================
class SwinTransformer3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24)):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(depths)):
            dim = embed_dim * (2 ** i)
            blocks = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim, num_heads[i],
                    window_size=(2,7,7),
                    shift_size=(0,0,0) if j % 2 == 0 else (1,3,3)
                )
                for j in range(depths[i])
            ])
            self.stages.append(blocks)
            if i < len(depths) - 1:
                self.downsamples.append(PatchMerging3D(dim))

        self.num_features = dim * 2

    def forward(self, x):
        x = self.patch_embed(x)

        for i, stage in enumerate(self.stages):
            for blk in stage:
                B, C, D, H, W = x.shape
                window_size, shift_size = get_window_size(
                    (D, H, W), blk.window_size, blk.shift_size
                )
                attn_mask = compute_mask(D, H, W, window_size, shift_size, x.device)
                x = rearrange(x, 'b c d h w -> b d h w c')
                x = blk(x, attn_mask)
                x = rearrange(x, 'b d h w c -> b c d h w')

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        return x


