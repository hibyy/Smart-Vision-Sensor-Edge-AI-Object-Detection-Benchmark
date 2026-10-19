"""Latency measurement for one detector configuration (run inside a fresh process)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def peak_memory_mb() -> float:
    """Peak resident memory of this process in MB (best effort across platforms)."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak:
            return round(peak / 1024**2, 1)
    except ImportError:
        pass
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # Linux: KB
    except ImportError:
        return 0.0


def current_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        return 0.0


def load_images(images_dir: str | Path, n: int) -> list[np.ndarray]:
    import cv2

    paths = sorted(Path(images_dir).glob("*.jpg"))[:n]
    images = [cv2.imread(str(p)) for p in paths]
    images = [im for im in images if im is not None]
    if not images:
        raise FileNotFoundError(f"No readable images in {images_dir}")
    return images


def measure_latency(detector, images: list[np.ndarray], warmup: int = 20, runs: int = 200) -> dict:
    """Batch-1 latency. Images are pre-loaded, so disk I/O is excluded.

    e2e = wall time of detect() (pre-processing + inference + post-processing).
    model = Ultralytics-reported inference time only.
    """
    for i in range(warmup):
        detector.detect(images[i % len(images)])
    e2e, model, pre, post = [], [], [], []
    for i in range(runs):
        img = images[i % len(images)]
        t0 = time.perf_counter()
        detector.detect(img)
        e2e.append((time.perf_counter() - t0) * 1000)
        sp = detector.last_speed
        model.append(sp.get("inference", 0.0))
        pre.append(sp.get("preprocess", 0.0))
        post.append(sp.get("postprocess", 0.0))

    def pct(v, q):
        return round(float(np.percentile(v, q)), 3)

    median = pct(e2e, 50)
    return {
        "e2e_median_ms": median, "e2e_p95_ms": pct(e2e, 95), "e2e_mean_ms": round(float(np.mean(e2e)), 3),
        "model_median_ms": pct(model, 50), "pre_median_ms": pct(pre, 50), "post_median_ms": pct(post, 50),
        "fps": round(1000.0 / median, 2) if median > 0 else 0.0,
        "warmup_runs": warmup, "timed_runs": runs, "peak_memory_mb": peak_memory_mb(),
    }


def main(argv: list[str] | None = None) -> None:
    """Worker entry point: prints one JSON line. Run in a fresh process per configuration."""
    import argparse

    from smart_vision.perception.yolo_backend import YoloDetector

    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=200)
    args = p.parse_args(argv)

    import ultralytics  # noqa: F401  (import the framework first so it is part of the baseline)

    images = load_images(args.images_dir, args.n_images)
    device = int(args.device) if str(args.device).isdigit() else args.device
    baseline_mb = current_rss_mb()
    detector = YoloDetector(args.weights, imgsz=args.imgsz, conf=0.25, device=device, threads=args.threads)
    stats = measure_latency(detector, images, args.warmup, args.runs)
    # peak_memory_mb includes the PyTorch runtime that Ultralytics always imports, so compare
    # formats with memory_delta_mb (growth caused by loading and running this model).
    stats["memory_delta_mb"] = round(max(0.0, current_rss_mb() - baseline_mb), 1)
    sys.stdout.write("RESULT_JSON " + json.dumps(stats) + "\n")


if __name__ == "__main__":
    main()
