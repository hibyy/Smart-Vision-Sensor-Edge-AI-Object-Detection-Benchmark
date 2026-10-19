"""Detector interface. Concrete backends (PyTorch, ONNX FP32/FP16/INT8) arrive in Phases 4-5."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from smart_vision.types import Detection


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run inference on one BGR image and return detections in pixel coordinates."""
        ...
