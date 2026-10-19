"""Seeded image degradations for robustness tests (BGR uint8 in, BGR uint8 out)."""
from __future__ import annotations

from functools import partial

import numpy as np


def blur(img: np.ndarray, rng: np.random.Generator, sigma: float = 3.0) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(img, (0, 0), sigma)


def dark(img: np.ndarray, rng: np.random.Generator, factor: float = 0.35) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def noise(img: np.ndarray, rng: np.random.Generator, std: float = 25.0) -> np.ndarray:
    noisy = img.astype(np.float32) + rng.normal(0, std, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def occlusion(img: np.ndarray, rng: np.random.Generator, area_frac: float = 0.2) -> np.ndarray:
    """Cover one random rectangle (about area_frac of the image) with flat gray."""
    h, w = img.shape[:2]
    rh, rw = max(1, int(h * np.sqrt(area_frac))), max(1, int(w * np.sqrt(area_frac)))
    y = int(rng.integers(0, max(1, h - rh + 1)))
    x = int(rng.integers(0, max(1, w - rw + 1)))
    out = img.copy()
    out[y:y + rh, x:x + rw] = 127
    return out


def jpeg(img: np.ndarray, rng: np.random.Generator, quality: int = 15) -> np.ndarray:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


CORRUPTIONS = {
    "blur": partial(blur, sigma=3.0),
    "dark": partial(dark, factor=0.35),
    "noise": partial(noise, std=25.0),
    "occlusion": partial(occlusion, area_frac=0.2),
    "jpeg": partial(jpeg, quality=15),
}
