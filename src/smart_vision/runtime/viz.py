"""Drawing helpers for annotated frames."""
from __future__ import annotations

import numpy as np

from smart_vision.types import Detection


def _color(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id * 7919 + 13)
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    import cv2

    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = _color(d.class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.0%}" + (f" #{d.track_id}" if d.track_id is not None else "")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - th - base - 2, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + th + base + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, top + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
