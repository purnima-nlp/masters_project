import cv2
import random
import torch
import numpy as np


class RGB2Thermal:
    """
    Convert RGB frames to pseudo-thermal frames.
    Currently uses grayscale conversion as a proxy.
    """

    def __call__(self, results):
        rgb_frames = results['frames']
        thermal_hr = []

        for frame in rgb_frames:
            # Convert BGR (OpenCV) to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thermal_hr.append(gray)

        results['thermal_hr'] = thermal_hr
        return results


class GenerateLR:
    """
    Generate low-resolution frames from high-resolution thermal frames.

    Supports multi-scale SR by sampling from a list of scales.
    """

    def __init__(self, scales, mode='random'):
        """
        Args:
            scales (list[int]): e.g. [2, 3, 4]
            mode (str): 'random' or 'cycle'
        """
        self.scales = scales
        self.mode = mode
        self._idx = 0  # used for cycling

    def _get_scale(self):
        if self.mode == 'random':
            return random.choice(self.scales)
        else:  # cycle
            scale = self.scales[self._idx]
            self._idx = (self._idx + 1) % len(self.scales)
            return scale

    def __call__(self, results):
        thermal_hr = results['thermal_hr']
        scale = self._get_scale()
        thermal_lr = []

        for frame in thermal_hr:
            h, w = frame.shape
            lr = cv2.resize(
                frame,
                (w // scale, h // scale),
                interpolation=cv2.INTER_CUBIC
            )
            thermal_lr.append(lr)

        results['thermal_lr'] = thermal_lr
        results['scale'] = scale
        return results


class ToTensor:
    """
    Convert numpy arrays to PyTorch tensors.
    Output shape:
        LR: (T, 1, H, W)
        HR: (T, 1, H, W)
    """

    def __call__(self, results):
        hr_frames = results['thermal_hr']
        lr_frames = results['thermal_lr']

        hr_tensor = torch.stack([
            torch.from_numpy(f).float().unsqueeze(0) / 255.0
            for f in hr_frames
        ])

        lr_tensor = torch.stack([
            torch.from_numpy(f).float().unsqueeze(0) / 255.0
            for f in lr_frames
        ])

        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
            'scale': results['scale']
        }

