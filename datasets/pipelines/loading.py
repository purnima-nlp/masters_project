import cv2


class LoadFrames:
    """
    Load all frames from a video file.

    Expects:
        results['video_path'] : str

    Adds:
        results['frames'] : list of numpy arrays (H, W, 3)
    """

    def __call__(self, results):
        video_path = results['video_path']
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames read from video: {video_path}")

        results['frames'] = frames
        return results

