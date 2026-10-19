"""COCO bbox evaluation (pycocotools) for any detector, with optional corruptions."""
from __future__ import annotations

import contextlib
import io
import random
import time
from pathlib import Path

import numpy as np

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]


def select_image_ids(all_ids: list[int], max_images: int | None, seed: int) -> list[int]:
    ids = sorted(all_ids)
    if max_images is None or max_images >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, max_images))


def evaluate_coco(detector, coco_dir: str | Path, max_images: int | None = None, seed: int = 2026,
                  corruption=None, progress: bool = False) -> dict:
    """Run `detector` on COCO val2017 images and return COCO bbox metrics.

    The detector must be built with a low confidence threshold (0.001) so the
    precision-recall curve is complete. `corruption` is a function (img, rng) -> img.
    """
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_dir = Path(coco_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(coco_dir / "annotations" / "instances_val2017.json"))
    cat_ids = sorted(gt.getCatIds())
    if len(cat_ids) != 80:
        raise ValueError(f"Expected 80 COCO categories, found {len(cat_ids)}")

    image_ids = select_image_ids(gt.getImgIds(), max_images, seed)
    results: list[dict] = []
    start = time.perf_counter()
    iterator = image_ids
    if progress:
        from tqdm import tqdm

        iterator = tqdm(image_ids, desc="evaluating")
    for image_id in iterator:
        info = gt.imgs[image_id]
        image = cv2.imread(str(coco_dir / "val2017" / info["file_name"]))
        if image is None:
            raise IOError(f"Could not read {info['file_name']}")
        if corruption is not None:
            image = corruption(image, np.random.default_rng(seed + image_id))
        for d in detector.detect(image):
            x1, y1, x2, y2 = d.bbox
            results.append({"image_id": image_id, "category_id": cat_ids[d.class_id],
                            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                            "score": round(d.confidence, 5)})
    elapsed = time.perf_counter() - start

    if not results:
        stats = [0.0] * 6
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(results)
            ev = COCOeval(gt, dt, "bbox")
            ev.params.imgIds = image_ids
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        stats = [float(max(s, 0.0)) for s in ev.stats[:6]]
    out = {name: round(val, 4) for name, val in zip(STAT_NAMES, stats)}
    out.update({"images": len(image_ids), "detections": len(results), "eval_seconds": round(elapsed, 1)})
    return out
