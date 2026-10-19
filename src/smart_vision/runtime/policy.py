"""Adaptive inference policy: skip detection when the scene is static.

The gate compares a small grayscale thumbnail of the current frame with the
thumbnail of the last frame that was actually processed. Detection runs when
the mean absolute difference exceeds `threshold`, or when `max_skip` frames in
a row were skipped (a safety net so slow changes are never missed forever).
"""
from __future__ import annotations

import numpy as np


class MotionGate:
    def __init__(self, threshold: float = 2.0, max_skip: int = 10, thumb: int = 64):
        if max_skip < 0:
            raise ValueError("max_skip must be >= 0")
        self.threshold, self.max_skip, self.thumb = threshold, max_skip, thumb
        self._ref: np.ndarray | None = None
        self._skipped = 0

    def _thumbnail(self, image: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.resize(gray, (self.thumb, self.thumb), interpolation=cv2.INTER_AREA).astype(np.float32)

    def should_run(self, image: np.ndarray) -> bool:
        thumb = self._thumbnail(image)
        if self._ref is None:
            run = True
        else:
            diff = float(np.mean(np.abs(thumb - self._ref)))
            run = diff > self.threshold or self._skipped >= self.max_skip
        if run:
            self._ref = thumb
            self._skipped = 0
        else:
            self._skipped += 1
        return run

    def reset(self) -> None:
        self._ref, self._skipped = None, 0
