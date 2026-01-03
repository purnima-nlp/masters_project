import torch.nn as nn
import torch.nn.functional as F


class ReconstructionHead(nn.Module):
    """
    Upsample feature maps to HR frames.
    """

    def __init__(self, in_channels, out_channels=1, scale=4):
        super().__init__()
        self.scale = scale

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels * (scale ** 2), kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        return: (B, 1, T, H*scale, W*scale)
        """
        x = self.conv(x)
        # Upsample only spatially
        x = F.pixel_shuffle(x, self.scale)
        return x

