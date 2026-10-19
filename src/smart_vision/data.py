"""COCO data preparation: download, verify, and select calibration images.

Design rules
- Accuracy is always measured on the full COCO val2017 set.
- INT8 calibration images come from train2017 only (never val2017).
- Selection is seeded and the chosen list is saved as a manifest, so the
  experiment is reproducible.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 1 << 20


class DataError(RuntimeError):
    """Raised when the dataset on disk does not match expectations."""


# ----------------------------------------------------------------- download
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: str | Path, retries: int = 3, timeout: int = 60) -> Path:
    """Download url to dest (atomic: writes .part then renames). Skips existing files."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, part.open("wb") as f:
                shutil.copyfileobj(resp, f, CHUNK)
            part.replace(dest)
            return dest
        except Exception as exc:  # network errors vary widely
            last_error = exc
            part.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    raise DataError(f"Could not download {url}: {last_error}")


def _extract(zip_path: Path, out_dir: Path, members: list[str] | None = None) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for name in members if members is not None else z.namelist():
            z.extract(name, out_dir)


def ensure_val_images(coco_dir: Path, url: str, expected: int) -> Path:
    val_dir = coco_dir / "val2017"
    if val_dir.is_dir() and len(list(val_dir.glob("*.jpg"))) >= expected:
        return val_dir
    zip_path = download_file(url, coco_dir / "_downloads" / "val2017.zip")
    _extract(zip_path, coco_dir)
    zip_path.unlink(missing_ok=True)
    return val_dir


def ensure_annotations(coco_dir: Path, url: str, splits: tuple[str, ...] = ("val",)) -> dict[str, Path]:
    """Return {split: path to instances_<split>2017.json}, downloading if needed."""
    paths = {s: coco_dir / "annotations" / f"instances_{s}2017.json" for s in splits}
    missing = [s for s, p in paths.items() if not p.exists()]
    if missing:
        zip_path = download_file(url, coco_dir / "_downloads" / "annotations_trainval2017.zip")
        _extract(zip_path, coco_dir, [f"annotations/instances_{s}2017.json" for s in missing])
        zip_path.unlink(missing_ok=True)
    return paths


# ------------------------------------------------------------------- verify
def verify_val(coco_dir: Path, expected_images: int = 5000, expected_categories: int = 80,
               decode_sample: int = 200, seed: int = 2026) -> dict:
    """Check that val2017 images and annotations are complete and readable."""
    ann_path = coco_dir / "annotations" / "instances_val2017.json"
    img_dir = coco_dir / "val2017"
    if not ann_path.exists() or not img_dir.is_dir():
        raise DataError(f"Missing val2017 images or annotations under {coco_dir}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images, annotations, categories = data["images"], data["annotations"], data["categories"]

    if len(images) != expected_images:
        raise DataError(f"Expected {expected_images} images in annotations, found {len(images)}")
    if len(categories) != expected_categories:
        raise DataError(f"Expected {expected_categories} categories, found {len(categories)}")

    missing, empty = [], []
    for img in images:
        p = img_dir / img["file_name"]
        if not p.exists():
            missing.append(img["file_name"])
        elif p.stat().st_size == 0:
            empty.append(img["file_name"])
    if missing or empty:
        raise DataError(f"{len(missing)} missing and {len(empty)} empty image files "
                        f"(e.g. {(missing + empty)[:3]})")

    # Decode a seeded sample to catch corrupt files.
    from PIL import Image

    rng = random.Random(seed)
    sample = rng.sample(images, min(decode_sample, len(images)))
    for img in sample:
        with Image.open(img_dir / img["file_name"]) as im:
            im.verify()

    ids = {img["id"] for img in images}
    if any(a["image_id"] not in ids for a in annotations):
        raise DataError("Found annotations that reference unknown image ids")

    return {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "images_without_annotations": len(ids - {a["image_id"] for a in annotations}),
        "decoded_sample": len(sample),
        "annotation_sha256": sha256_file(ann_path),
    }


# -------------------------------------------------------------- calibration
def select_calibration_images(images: list[dict], annotations: list[dict], n: int,
                              seed: int, min_annotations: int = 1) -> list[dict]:
    """Pick n images reproducibly. Eligibility: at least min_annotations annotations."""
    counts: dict[int, int] = {}
    for a in annotations:
        counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1
    eligible = sorted((img for img in images if counts.get(img["id"], 0) >= min_annotations),
                      key=lambda img: img["id"])
    if len(eligible) < n:
        raise DataError(f"Only {len(eligible)} eligible images, need {n}")
    chosen = random.Random(seed).sample(eligible, n)
    return sorted(({"id": c["id"], "file_name": c["file_name"]} for c in chosen),
                  key=lambda c: c["id"])


def build_calibration_manifest(train_ann_path: Path, n: int, seed: int, min_annotations: int) -> dict:
    # NOTE: instances_train2017.json is ~450 MB and needs a few GB of RAM to parse.
    data = json.loads(train_ann_path.read_text(encoding="utf-8"))
    items = select_calibration_images(data["images"], data["annotations"], n, seed, min_annotations)
    return {
        "source_split": "train2017",
        "seed": seed,
        "num_images": len(items),
        "min_annotations_per_image": min_annotations,
        "images": items,
    }


def download_calibration_images(items: list[dict], out_dir: Path, base_url: str, workers: int = 12) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    def fetch(item: dict) -> None:
        download_file(base_url + item["file_name"], out_dir / item["file_name"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, it) for it in items]
        for fut in as_completed(futures):
            fut.result()  # re-raise any download error
    return len(list(out_dir.glob("*.jpg")))


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
