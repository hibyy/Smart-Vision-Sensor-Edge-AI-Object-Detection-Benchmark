"""AP under image degradations (blur, dark, noise, occlusion, jpeg) plus a clean baseline. Resumable.

Usage: python scripts/06_robustness.py --tag kaggle_cpu --data-root data --models yolo11n --imgsz 640 --formats onnx_fp32,onnx_int8
"""
import argparse
from pathlib import Path

from smart_vision.bench.corruptions import CORRUPTIONS
from smart_vision.bench.evaluate import evaluate_coco
from smart_vision.bench.results import append_row, csv_list, done_keys, grid, model_path
from smart_vision.config import load_config, project_root
from smart_vision.perception.yolo_backend import YoloDetector


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--imgsz", default=None)
    p.add_argument("--formats", default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    root = project_root()
    models_dir = root / cfg["paths"]["models_dir"]
    coco_dir = Path(args.data_root or root / cfg["paths"]["data_root"]) / "coco"
    max_images = args.max_images or cfg["robustness"]["max_images"]
    out_csv = root / cfg["paths"]["results_root"] / "tables" / f"robustness_{args.tag}.csv"
    done = done_keys(out_csv, ["model", "imgsz", "format", "corruption", "max_images"])
    device = int(args.device) if str(args.device).isdigit() else args.device
    conditions = ["clean"] + list(cfg["robustness"]["corruptions"])

    for model, imgsz, fmt in grid(cfg["experiment"], csv_list(args.models), csv_list(args.imgsz, int),
                                  csv_list(args.formats)):
        path = model_path(models_dir, model, imgsz, fmt)
        if not path.exists():
            print(f"skip (missing file): {path}")
            continue
        detector = None
        for cond in conditions:
            if (model, str(imgsz), fmt, cond, str(max_images)) in done:
                continue
            detector = detector or YoloDetector(path, imgsz=imgsz, conf=cfg["evaluation"]["conf"],
                                                iou=cfg["evaluation"]["iou"], device=device)
            print(f"{model} {imgsz} {fmt} [{cond}] ...")
            stats = evaluate_coco(detector, coco_dir, max_images, cfg["seed"],
                                  corruption=None if cond == "clean" else CORRUPTIONS[cond], progress=True)
            append_row(out_csv, {"tag": args.tag, "model": model, "imgsz": imgsz, "format": fmt,
                                 "corruption": cond, "max_images": max_images, **stats})
            print(f"  AP={stats['AP']}")


if __name__ == "__main__":
    main()
