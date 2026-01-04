import os
import cv2


class LoadVimeoFrames:
    """
    Load all PNG frames from a Vimeo-90K sequence folder.

    Expects:
        results['seq_path'] : str
            Path like:
            vimeo_septuplet/sequences/00001/0001

    Adds:
        results['frames'] : list[np.ndarray] (H, W, 3) in BGR format
    """

    def __call__(self, results):
        seq_path = results['seq_path']

        if not os.path.isdir(seq_path):
            raise FileNotFoundError(f"Sequence folder not found: {seq_path}")

        frame_files = sorted(
            f for f in os.listdir(seq_path) if f.endswith(".png")
        )

        if len(frame_files) == 0:
            raise ValueError(f"No PNG frames found in {seq_path}")

        frames = []
        for fname in frame_files:
            img_path = os.path.join(seq_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img is None:
                raise IOError(f"Failed to read image: {img_path}")

            frames.append(img)

        results['frames'] = frames
        return results

