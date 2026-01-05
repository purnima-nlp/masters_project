import torch
import torch.nn as nn

from models.backbone.Thermal_video_SwinSR import SwinTransformer3D
from models.heads.reconstruction_head import ReconstructionHead


class VideoSRModel(nn.Module):
    """
    Full Video Super-Resolution Model

    LR Video --> SwinTransformer3D --> ReconstructionHead --> HR Video
    """

    def __init__(
        self,
        scale=4,
        in_chans=1,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
    ):
        super().__init__()

        # -------- Backbone --------
        self.backbone = SwinTransformer3D(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
        )

        # -------- Reconstruction Head --------
        self.head = ReconstructionHead(
            in_channels=self.backbone.num_features,
            scale=scale,
            out_channels=in_chans,
        )

        self.scale = scale

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        feats = self.backbone(x)
        out = self.head(feats)
        return out


if __name__ == "__main__":
    model = VideoSRModel(scale=4)
    dummy = torch.randn(1, 1, 8, 64, 64)
    out = model(dummy)
    print("Output shape:", out.shape)


