import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionHead(nn.Module):
    def __init__(self, in_channels, scale=4, out_channels=1):
        super().__init__()
        self.scale = scale

        # IMPORTANT: produce channels = out_channels * scale^2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * scale * scale,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # 1️⃣ Merge time into batch
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)                # (B*T, C, H, W)

        # 2️⃣ Upsample spatially
        x = self.conv(x)                          # (B*T, C*s^2, H, W)
        x = F.pixel_shuffle(x, self.scale)        # (B*T, out_ch, H*s, W*s)

        # 3️⃣ Restore time dimension
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, T, C_out, H_out, W_out)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # (B, C, T, H, W)

        return x
