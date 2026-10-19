"""Illumination-aware preprocessing: brighten dark frames only, leave normal frames untouched.

A smart sensor should adapt to its conditions. COCO-trained detectors lose accuracy in the
dark, so the sensor measures the mean luminance of each frame (cheaply, on a sub-sampled grid)
and, only when the frame is darker than `luma_threshold`, applies an adaptive correction:

* gamma:       one lookup table, gamma chosen so the mean luminance moves to `target_luma`
* clahe:       contrast-limited adaptive histogram equalization on the L channel (LAB)
* gamma_clahe: both

Geometry is never changed, so bounding boxes stay valid.
"""
from __future__ import annotations

import math
import time

import numpy as np

METHODS = ("gamma", "clahe", "gamma_clahe")


class LowLightEnhancer:
    def __init__(self, luma_threshold: float = 70.0, target_luma: float = 110.0, method: str = "gamma",
                 min_gamma: float = 0.25, clip_limit: float = 2.0):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.luma_threshold, self.target_luma, self.method = luma_threshold, target_luma, method
        self.min_gamma, self.clip_limit = min_gamma, clip_limit
        self.last_luma: float | None = None
        self.last_applied = False

    @staticmethod
    def mean_luma(image: np.ndarray) -> float:
        sub = image[::8, ::8].astype(np.float32)  # BGR
        return float(0.114 * sub[..., 0].mean() + 0.587 * sub[..., 1].mean() + 0.299 * sub[..., 2].mean())

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2

        luma = self.mean_luma(image)
        self.last_luma = luma
        if luma >= self.luma_threshold:
            self.last_applied = False
            return image
        self.last_applied = True
        out = image
        if "gamma" in self.method:
            gamma = math.log(self.target_luma / 255.0) / math.log(max(luma, 1.0) / 255.0)
            gamma = min(max(gamma, self.min_gamma), 1.0)  # never darken
            lut = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
            out = cv2.LUT(out, lut)
        if "clahe" in self.method:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            lab[..., 0] = clahe.apply(lab[..., 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return out


class EnhancedDetector:
    """Wraps any detector so every frame goes through the enhancer first."""

    def __init__(self, detector, enhancer: LowLightEnhancer):
        self.inner = detector
        self.enhancer = enhancer
        self.name = f"{detector.name}+{enhancer.method}"
        self.last_enhance_ms = 0.0

    def detect(self, image: np.ndarray):
        t0 = time.perf_counter()
        enhanced = self.enhancer(image)
        self.last_enhance_ms = (time.perf_counter() - t0) * 1000
        return self.inner.detect(enhanced)

    def __getattr__(self, item):  # last_speed, reset_tracker, class_names, runtime_info ...
        if item in ("inner", "enhancer"):
            raise AttributeError(item)
        return getattr(self.inner, item)
