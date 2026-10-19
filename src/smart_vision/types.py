"""Shared data types: what flows between sensor, perception and event layers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_s: float
    image: np.ndarray  # BGR, HxWx3 (OpenCV convention)
    source: str = ""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    track_id: int | None = None


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float | None = None
