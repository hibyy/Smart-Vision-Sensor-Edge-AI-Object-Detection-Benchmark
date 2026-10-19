"""Ultralytics-based detector that loads PyTorch (.pt) or ONNX (.onnx) weights.

Using one backend for every format keeps pre-processing and post-processing
identical across formats, so differences in the results come from the model
format, not from different glue code.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np

from smart_vision.types import Detection

_ORT = {"threads": None, "patched": False}


def set_ort_threads(n: int | None) -> None:
    """Force ONNX Runtime intra-op threads for every session created afterwards.

    Ultralytics creates the ONNX session itself, so we wrap InferenceSession once
    and read the desired thread count from module state at session creation time.
    """
    _ORT["threads"] = n
    if n is None or _ORT["patched"]:
        return
    import onnxruntime as ort

    original = ort.InferenceSession

    class _ThreadedSession(original):  # type: ignore[misc, valid-type]
        # Ultralytics passes sess_options positionally, so keep the same signature as ORT.
        def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
            opts = sess_options or ort.SessionOptions()
            if _ORT["threads"] is not None:
                opts.intra_op_num_threads = int(_ORT["threads"])
            super().__init__(path_or_bytes, opts, *args, **kwargs)

    ort.InferenceSession = _ThreadedSession
    _ORT["patched"] = True


@contextlib.contextmanager
def _chdir(path: Path):
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def ensure_weights(name: str, models_dir: str | Path) -> Path:
    """Return models_dir/<name>.pt, downloading it with Ultralytics if missing."""
    models_dir = Path(models_dir).resolve()
    target = models_dir / f"{name}.pt"
    if not target.exists():
        from ultralytics import YOLO

        with _chdir(models_dir):
            YOLO(f"{name}.pt")  # triggers the download into the current directory
    if not target.exists():
        raise FileNotFoundError(f"Could not obtain weights for {name} in {models_dir}")
    return target


class YoloDetector:
    def __init__(self, weights: str | Path, imgsz: int = 640, conf: float = 0.25,
                 iou: float = 0.7, device: str | int = "cpu", threads: int | None = None,
                 track: bool = False, tracker: str = "bytetrack.yaml", max_det: int = 300):
        from ultralytics import YOLO

        self.weights = Path(weights)
        self.name = self.weights.stem
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device, self.track, self.tracker, self.max_det = device, track, tracker, max_det
        self.last_speed: dict[str, float] = {}
        self.threads = threads
        if threads is not None:
            set_ort_threads(threads)
            try:
                import torch

                torch.set_num_threads(threads)
            except ImportError:
                pass
        self.model = YOLO(str(self.weights), task="detect")

    def detect(self, image: np.ndarray) -> list[Detection]:
        kwargs = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device,
                      max_det=self.max_det, verbose=False)
        if self.track:
            results = self.model.track(image, persist=True, tracker=self.tracker, **kwargs)
        else:
            results = self.model.predict(image, **kwargs)
        result = results[0]
        self.last_speed = dict(result.speed)  # preprocess / inference / postprocess (ms)
        return self._to_detections(result)

    def reset_tracker(self) -> None:
        """Forget tracker state (call between videos)."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for t in trackers:
                t.reset()

    @staticmethod
    def _to_detections(result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        names = result.names
        return [
            Detection(int(c), str(names[int(c)]), float(p), tuple(float(v) for v in b),
                      None if i is None else int(i))
            for b, p, c, i in zip(xyxy, conf, cls, ids)
        ]
