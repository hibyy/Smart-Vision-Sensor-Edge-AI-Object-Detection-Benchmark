"""COCO accuracy (AP) for each (model, imgsz, format). Resumable: finished rows are skipped.

Usage: python scripts/04_eval_accuracy.py --tag kaggle_cpu --data-root /kaggle/working/data \
           --models yolo11n --imgsz 640 --formats pytorch_fp32,onnx_fp32,onnx_int8 [--max-images 500]
"""
import argparse
from pathlib import Path

from smart_vision.bench.evaluate import evaluate_coco
from smart_vision.bench.export import file_size_mb
from smart_vision.bench.results import append_row, csv_list, done_keys, grid, model_path
from smart_vision.config import load_config, project_root
from smart_vision.perception.yolo_backend import YoloDetector


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="hardware tag, e.g. kaggle_cpu or kaggle_t4")
    p.add_argument("--data-root", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--imgsz", default=None)
    p.add_argument("--formats", default=None)
    p.add_argument("--max-images", type=int, default=None, help="default: config value (null = full val2017)")
    p.add_argument("--device", default="cpu", help="cpu or a CUDA index such as 0")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    root = project_root()
    models_dir = root / cfg["paths"]["models_dir"]
    coco_dir = Path(args.data_root or root / cfg["paths"]["data_root"]) / "coco"
    max_images = args.max_images if args.max_images is not None else cfg["evaluation"]["max_images"]
    out_csv = root / cfg["paths"]["results_root"] / "tables" / f"accuracy_{args.tag}.csv"
    key_cols = ["model", "imgsz", "format", "device", "max_images"]
    done = done_keys(out_csv, key_cols)
    device = int(args.device) if str(args.device).isdigit() else args.device

    configs = grid(cfg["experiment"], csv_list(args.models), csv_list(args.imgsz, int), csv_list(args.formats))
    for model, imgsz, fmt in configs:
        key = (model, str(imgsz), fmt, str(args.device), str(max_images or "full"))
        if key in done:
            print(f"skip (done): {key}")
            continue
        path = model_path(models_dir, model, imgsz, fmt)
        if not path.exists():
            print(f"skip (missing file): {path}")
            continue
        print(f"evaluating {model} {imgsz} {fmt} on {args.device} ...")
        detector = YoloDetector(path, imgsz=imgsz, conf=cfg["evaluation"]["conf"],
                                iou=cfg["evaluation"]["iou"], device=device, threads=args.threads)
        stats = evaluate_coco(detector, coco_dir, max_images, cfg["seed"], progress=True)
        append_row(out_csv, {"tag": args.tag, "model": model, "imgsz": imgsz, "format": fmt,
                             "device": args.device, "max_images": max_images or "full",
                             "size_mb": file_size_mb(path), **stats})
        print(f"  AP={stats['AP']}  AP50={stats['AP50']}  images={stats['images']}")


if __name__ == "__main__":
    main()
