import json
from pathlib import Path

import numpy as np
import pytest

from smart_vision.bench import corruptions
from smart_vision.bench.export import letterbox, to_model_input
from smart_vision.bench.report import build_summary, pareto_front, plot_pareto, robustness_table, to_markdown
from smart_vision.bench.results import append_row, csv_list, done_keys, grid, model_path, read_rows
from smart_vision.types import Detection

cv2 = pytest.importorskip("cv2")


def test_letterbox_and_input_shape():
    img = np.zeros((480, 640, 3), np.uint8)
    out = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    x = to_model_input(img, 320)
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32 and x.max() <= 1.0


def test_corruptions_are_deterministic_and_change_image():
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    for name, fn in corruptions.CORRUPTIONS.items():
        a = fn(img, np.random.default_rng(1))
        b = fn(img, np.random.default_rng(1))
        assert a.shape == img.shape and a.dtype == np.uint8, name
        assert np.array_equal(a, b), name
        assert not np.array_equal(a, img), name


def test_results_csv_resume(tmp_path):
    p = tmp_path / "r.csv"
    append_row(p, {"model": "a", "imgsz": 640, "v": 1})
    append_row(p, {"model": "b", "imgsz": 320, "v": 2})
    assert done_keys(p, ["model", "imgsz"]) == {("a", "640"), ("b", "320")}
    assert len(read_rows(p)) == 2


def test_grid_and_paths():
    exp = {"models": ["m1", "m2"], "imgsz": [320, 640], "formats": ["pytorch_fp32", "onnx_int8"]}
    assert len(grid(exp)) == 8
    assert grid(exp, models=["m1"], imgsz=[640], formats=["onnx_int8"]) == [("m1", 640, "onnx_int8")]
    assert model_path("models", "m1", 640, "onnx_int8") == Path("models/onnx/m1_640_int8.onnx")
    assert model_path("models", "m1", 640, "onnx_int8_mixed") == Path("models/onnx/m1_640_int8_mixed.onnx")
    assert csv_list("1, 2,3", int) == [1, 2, 3]
    with pytest.raises(ValueError):
        model_path("models", "m1", 640, "bogus")


def test_summary_pareto_and_plots(tmp_path):
    acc = [{"model": m, "imgsz": s, "format": f, "AP": ap, "AP50": 0.5, "AP_small": 0.1, "images": 5000}
           for m, s, f, ap in [("a", 640, "onnx_fp32", 0.40), ("a", 320, "onnx_fp32", 0.30),
                               ("b", 640, "onnx_int8", 0.35), ("b", 320, "onnx_int8", 0.20)]]
    lat = [{"model": m, "imgsz": s, "format": f, "threads": 4, "size_mb": 10, "e2e_median_ms": ms,
            "e2e_p95_ms": ms * 1.2, "model_median_ms": ms * 0.8, "fps": 1000 / ms, "peak_memory_mb": 800, "memory_delta_mb": 50}
           for m, s, f, ms in [("a", 640, "onnx_fp32", 100), ("a", 320, "onnx_fp32", 30),
                               ("b", 640, "onnx_int8", 60), ("b", 320, "onnx_int8", 35)]]
    summary = build_summary(acc, lat, threads=4)
    assert summary[0]["memory_delta_mb"] == 50.0 and len(summary) == 4 and summary == sorted(summary, key=lambda r: r["e2e_median_ms"])
    front = pareto_front(summary)
    assert {(r["model"], r["imgsz"]) for r in front} == {("a", 640), ("a", 320), ("b", 640)}  # b@320 is dominated
    assert plot_pareto(summary, tmp_path / "p.png").exists()
    assert "| model |" in to_markdown(summary, ["model", "AP"])


def test_robustness_table():
    rows = [{"model": "a", "imgsz": 640, "format": "onnx_fp32", "corruption": c, "AP": ap}
            for c, ap in [("clean", 0.4), ("blur", 0.3)]]
    t = robustness_table(rows)
    assert t[0]["blur_drop_%"] == 25.0


class OracleDetector:
    """Returns the ground-truth boxes, so AP must be ~1.0 (validates the evaluation plumbing)."""

    def __init__(self, gt_by_file):
        self.gt = gt_by_file
        self.current = None

    def detect(self, image):
        return self.gt[int(image[0, 0, 0])]  # image[0,0,0] encodes the image index


def test_evaluate_coco_with_oracle(tmp_path):
    pytest.importorskip("pycocotools")
    from smart_vision.bench.evaluate import evaluate_coco

    coco = tmp_path / "coco"
    (coco / "val2017").mkdir(parents=True)
    (coco / "annotations").mkdir()
    cats = [{"id": 1 + i * 2, "name": f"c{i}"} for i in range(80)]  # non-contiguous ids on purpose
    images, anns, gt_by_idx = [], [], {}
    for idx in range(1, 4):
        img = np.full((100, 100, 3), idx, np.uint8)
        cv2.imwrite(str(coco / "val2017" / f"{idx:012d}.png"), img)  # png keeps pixel values exact
        images.append({"id": idx, "file_name": f"{idx:012d}.png", "width": 100, "height": 100})
        cls = idx  # class index 1..3
        box = (10, 10, 50, 60)
        anns.append({"id": idx, "image_id": idx, "category_id": cats[cls]["id"], "bbox": [10, 10, 40, 50],
                     "area": 2000, "iscrowd": 0})
        gt_by_idx[idx] = [Detection(cls, f"c{cls}", 0.99, box)]
    (coco / "annotations" / "instances_val2017.json").write_text(
        json.dumps({"images": images, "annotations": anns, "categories": cats}))
    stats = evaluate_coco(OracleDetector(gt_by_idx), coco)
    assert stats["images"] == 3 and stats["AP"] > 0.99 and stats["AP50"] > 0.99
