import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionHead(nn.Module):
    def __init__(self, in_channels, scale=4, out_channels=1):
        super().__init__()

        assert scale in [2, 4, 8], "Scale must be power of 2"

        self.scale = scale
        num_upsamples = int(math.log2(scale))

        layers = []
        channels = in_channels

        for _ in range(num_upsamples):
            layers += [
                nn.Conv3d(channels, channels // 2, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            channels = channels // 2

        self.upsample_layers = nn.Sequential(*layers)

        self.final_conv = nn.Conv3d(
            channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        for layer in self.upsample_layers:
            x = layer(x)

            # Upsample ONLY spatial dimensions
            if isinstance(layer, nn.GELU):
                x = F.interpolate(
                    x,
                    scale_factor=(1, 2, 2),
                    mode="trilinear",
                    align_corners=False
                )

        x = self.final_conv(x)
        return x
