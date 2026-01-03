from torch.utils.data import Dataset


class VideoSRDataset(Dataset):
    """
    PyTorch Dataset for Thermal Video Super-Resolution.
    """

    def __init__(self, video_list, pipeline, sampler=None):
        """
        Args:
            video_list (list[str]): list of video file paths
            pipeline (callable): composed pipeline
            sampler (object, optional): clip sampler
        """
        self.video_list = video_list
        self.pipeline = pipeline
        self.sampler = sampler

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        results = {
            'video_path': self.video_list[idx]
        }

        # Run preprocessing pipeline
        results = self.pipeline(results)

        # Apply temporal sampling if sampler is provided
        if self.sampler is not None:
            indices = self.sampler.sample(len(results['lr']))
            results['lr'] = results['lr'][indices]
            results['hr'] = results['hr'][indices]

        return results['lr'], results['hr'], results['scale']

