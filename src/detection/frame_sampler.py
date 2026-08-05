import numpy as np


class FrameSampler:
    """Selects a fixed number of evenly-spaced frames from a buffer.

    Keeps downstream stages (OCR) from being flooded with redundant frames
    while preserving enough coverage for statistical filtering.
    """

    def __init__(self, n_frames: int = 20):
        self.n_frames = n_frames

    def sample(self, frames: list) -> list:
        """Return up to n_frames evenly-spaced frames from the input list.

        Returns all frames if len(frames) <= n_frames.
        Returns empty list if frames is empty.
        """
        if not frames:
            return []
        if len(frames) <= self.n_frames:
            return list(frames)
        idx = np.linspace(0, len(frames) - 1, self.n_frames, dtype=int)
        return [frames[i] for i in idx]
