"""Download and verify COCO val2017; optionally build the INT8 calibration set.

Examples
  python scripts/01_prepare_data.py --data-root data
  python scripts/01_prepare_data.py --data-root /kaggle/working/data --calibration
"""
import argparse
from pathlib import Path

from smart_vision import data
from smart_vision.config import load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--calibration", action="store_true",
                        help="also select and download calibration images from train2017")
    parser.add_argument("--skip-val", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    root = project_root()
    data_root = Path(args.data_root or cfg["paths"]["data_root"])
    if not data_root.is_absolute() and args.data_root is None:
        data_root = root / data_root
    coco_dir = data_root / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_val:
        print("[1/3] val2017 images ...")
        data.ensure_val_images(coco_dir, ds["val_images_url"], ds["expected_val_images"])
        print("[2/3] annotations ...")
        data.ensure_annotations(coco_dir, ds["annotations_url"], ("val",))
        print("[3/3] verifying ...")
        report = data.verify_val(coco_dir, ds["expected_val_images"], ds["expected_categories"],
                                 seed=cfg["seed"])
        out = root / cfg["paths"]["results_root"] / "manifests" / "val2017_verification.json"
        data.write_json(out, report)
        print(f"val2017 OK: {report}")
        print(f"Saved: {out}")

    if args.calibration:
        cal = cfg["calibration"]
        manifest_path = root / cal["manifest"]
        if manifest_path.exists():
            import json
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"Using existing calibration manifest: {manifest_path}")
        else:
            print("Selecting calibration images from train2017 (needs ~3 GB RAM) ...")
            train_ann = data.ensure_annotations(coco_dir, ds["annotations_url"], ("train",))["train"]
            manifest = data.build_calibration_manifest(
                train_ann, cal["num_images"], cfg["seed"], cal["min_annotations_per_image"])
            data.write_json(manifest_path, manifest)
            print(f"Saved manifest (commit this file): {manifest_path}")
        n = data.download_calibration_images(
            manifest["images"], coco_dir / "calibration", ds["train_image_base_url"])
        print(f"Calibration images on disk: {n}")


if __name__ == "__main__":
    main()
