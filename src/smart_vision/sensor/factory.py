"""Build a frame source from a user-supplied string."""
from __future__ import annotations

from pathlib import Path

from smart_vision.sensor.sources import ImageFolderSource, VideoSource, WebcamSource


def open_source(spec: str, stride: int = 1, fps: float = 1.0):
    """'webcam' / 'webcam:1' -> webcam, folder -> image folder, otherwise a video file."""
    if spec.startswith("webcam"):
        device = int(spec.split(":", 1)[1]) if ":" in spec else 0
        return WebcamSource(device, stride=stride)
    path = Path(spec)
    if path.is_dir():
        return ImageFolderSource(path, fps=fps)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {spec}")
    return VideoSource(path, stride=stride)
