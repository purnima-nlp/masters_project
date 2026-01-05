import torch
import torch.nn as nn

from models.backbone.Thermal_video_SwinSR import SwinTransformer3D
from models.heads.reconstruction_head import ReconstructionHead


class VideoSRModel(nn.Module):
    """
    Full Video Super-Resolution Model

    Pipeline:
        LR Video  --> SwinTransformer3D (backbone)
                 --> ReconstructionHead (upsampling)
                 --> HR Video
    """

    def __init__(
        self,
        scale=4,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(2, 7, 7),
        
    ):
        super().__init__()

        # -------- Backbone --------
        self.backbone = SwinTransformer3D(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            
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
        Args:
            x: LR video tensor of shape (B, C, T, H, W)

        Returns:
            HR video tensor of shape (B, C, T, H*scale, W*scale)
        """
        feats = self.backbone(x)     # (B, C_feat, T, H', W')
        out = self.head(feats)       # (B, C, T, H*scale, W*scale)
        return out


if __name__ == "__main__":
    # Quick sanity check
    model = VideoSRModel(scale=4)
    dummy = torch.randn(1, 3, 8, 64, 64)
    out = model(dummy)
    print("Output shape:", out.shape)

