import torch
import torch.nn as nn


class L1Loss(nn.Module):
    """
    Standard L1 Loss for Video Super-Resolution

    Computes mean absolute error between
    predicted HR video and ground-truth HR video.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: predicted HR video (B, C, T, H, W)
            target: ground-truth HR video (B, C, T, H, W)
        """
        return torch.mean(torch.abs(pred - target))

