import random


class ClipSampler:
    """
    Sample a fixed-length clip from a long video.

    This sampler selects frame indices only.
    """

    def __init__(self, clip_len):
        """
        Args:
            clip_len (int): number of frames in one clip
        """
        self.clip_len = clip_len

    def sample(self, total_frames):
        """
        Args:
            total_frames (int): total number of frames in the video

        Returns:
            List[int]: indices of sampled frames
        """
        if total_frames <= self.clip_len:
            return list(range(total_frames))

        start = random.randint(0, total_frames - self.clip_len)
        return list(range(start, start + self.clip_len))

