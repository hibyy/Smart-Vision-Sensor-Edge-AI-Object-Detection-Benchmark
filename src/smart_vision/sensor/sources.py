"""Visual input: image folders, video files and webcams, all yielding Frame objects."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from smart_vision.types import Frame

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderSource:
    """Iterate over images in a folder (sorted by name) as if they were video frames."""

    def __init__(self, folder: str | Path, fps: float = 1.0):
        self.folder = Path(folder)
        self.fps = fps
        self.paths = sorted(p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.folder}")

    def __iter__(self) -> Iterator[Frame]:
        import cv2

        for i, path in enumerate(self.paths):
            image = cv2.imread(str(path))
            if image is None:
                raise IOError(f"Could not read image: {path}")
            yield Frame(index=i, timestamp_s=i / self.fps, image=image, source=path.name)


class _CaptureSource:
    def __init__(self, target, name: str, stride: int = 1):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.target, self.name, self.stride = target, name, stride

    def __iter__(self) -> Iterator[Frame]:
        import cv2

        cap = cv2.VideoCapture(self.target)
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {self.name}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        index = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if index % self.stride == 0:
                    yield Frame(index=index, timestamp_s=index / fps, image=image, source=self.name)
                index += 1
        finally:
            cap.release()


class VideoSource(_CaptureSource):
    def __init__(self, path: str | Path, stride: int = 1):
        super().__init__(str(path), Path(path).name, stride)


class WebcamSource(_CaptureSource):
    def __init__(self, device: int = 0, stride: int = 1):
        super().__init__(device, f"webcam{device}", stride)
