import os
import torch
from torch.utils.data import Dataset


class VideoSRDataset(Dataset):
    """
    Simple Video Super-Resolution Dataset (Demo + Training friendly)

    Expected directory structure (Vimeo-90K style or mini version):

    root_dir/
      └── test/
          └── 00001/
              └── 0001/
                  ├── im1.png
                  ├── im2.png
                  └── ...
    """

    def __init__(self, root_dir, scale=4, split="test", pipeline=None):
        """
        Args:
            root_dir (str): dataset root directory
            scale (int): super-resolution scale
            split (str): train / test
            pipeline (callable, optional): preprocessing pipeline
        """
        self.root_dir = root_dir
        self.scale = scale
        self.split = split
        self.pipeline = pipeline

        self.seq_paths = self._scan_sequences()

        if len(self.seq_paths) == 0:
            raise RuntimeError(f"No sequences found in {self.root_dir}/{self.split}")

    def _scan_sequences(self):
        """
        Scan dataset folders and collect sequence paths.
        """
        seq_root = os.path.join(self.root_dir, self.split)
        seq_paths = []

        if not os.path.isdir(seq_root):
            raise FileNotFoundError(f"Split folder not found: {seq_root}")

        for folder1 in sorted(os.listdir(seq_root)):
            path1 = os.path.join(seq_root, folder1)
            if not os.path.isdir(path1):
                continue

            for folder2 in sorted(os.listdir(path1)):
                seq_path = os.path.join(path1, folder2)
                if os.path.isdir(seq_path):
                    seq_paths.append(seq_path)

        return seq_paths

    def __len__(self):
        return len(self.seq_paths)

    def __getitem__(self, idx):
        seq_path = self.seq_paths[idx]

        results = {
            "seq_path": seq_path,
            "scale": self.scale
        }

        # Run pipeline (e.g. LoadVimeoFrames → LR/HR generation)
        if self.pipeline is not None:
            results = self.pipeline(results)
        else:
            raise RuntimeError("Pipeline is required to load data")

        # Expected pipeline outputs:
        # results['lr'] : Tensor (C, T, H, W)
        # results['hr'] : Tensor (C, T, sH, sW)

        lr = results["lr"]
        hr = results["hr"]

        return lr, hr, self.scale

