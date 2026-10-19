import json

import pytest
from PIL import Image

from smart_vision import data


def _fake_coco(n_images=10):
    images = [{"id": i, "file_name": f"{i:012d}.jpg"} for i in range(1, n_images + 1)]
    # odd ids have 2 annotations, even ids have none
    annotations = [{"id": 100 * i + k, "image_id": i} for i in range(1, n_images + 1) if i % 2 for k in (1, 2)]
    return images, annotations


def test_selection_is_deterministic_and_eligible():
    images, annotations = _fake_coco(20)
    a = data.select_calibration_images(images, annotations, n=5, seed=1)
    b = data.select_calibration_images(images, annotations, n=5, seed=1)
    assert a == b
    assert all(item["id"] % 2 == 1 for item in a)  # only images with annotations
    assert a == sorted(a, key=lambda x: x["id"])


def test_selection_changes_with_seed():
    images, annotations = _fake_coco(40)
    a = data.select_calibration_images(images, annotations, n=5, seed=1)
    b = data.select_calibration_images(images, annotations, n=5, seed=2)
    assert a != b


def test_selection_raises_when_not_enough_images():
    images, annotations = _fake_coco(4)
    with pytest.raises(data.DataError):
        data.select_calibration_images(images, annotations, n=10, seed=1)


def _write_fake_val(tmp_path, n=3, categories=2, drop=None):
    coco = tmp_path / "coco"
    (coco / "val2017").mkdir(parents=True)
    (coco / "annotations").mkdir()
    images = [{"id": i, "file_name": f"{i:012d}.jpg"} for i in range(1, n + 1)]
    for img in images:
        if img["id"] != drop:
            Image.new("RGB", (8, 8)).save(coco / "val2017" / img["file_name"])
    payload = {
        "images": images,
        "annotations": [{"id": 1, "image_id": 1}],
        "categories": [{"id": c, "name": str(c)} for c in range(categories)],
    }
    (coco / "annotations" / "instances_val2017.json").write_text(json.dumps(payload))
    return coco


def test_verify_val_ok(tmp_path):
    coco = _write_fake_val(tmp_path)
    report = data.verify_val(coco, expected_images=3, expected_categories=2, decode_sample=3)
    assert report["images"] == 3 and report["images_without_annotations"] == 2


def test_verify_val_detects_missing_image(tmp_path):
    coco = _write_fake_val(tmp_path, drop=2)
    with pytest.raises(data.DataError):
        data.verify_val(coco, expected_images=3, expected_categories=2)


def test_verify_val_detects_wrong_counts(tmp_path):
    coco = _write_fake_val(tmp_path)
    with pytest.raises(data.DataError):
        data.verify_val(coco, expected_images=5000, expected_categories=2)
