"""Export ONNX FP32, then derive FP16 and INT8 variants (resumable).

INT8 needs the calibration images from step 01 (--calibration).
Usage: python scripts/03_export_models.py --data-root data --models yolo11n --imgsz 640
"""
import argparse
from pathlib import Path

from smart_vision.bench.export import convert_fp16, export_onnx_fp32, file_size_mb, quantize_int8
from smart_vision.bench.results import csv_list, model_path
from smart_vision.config import load_config, project_root
from smart_vision.perception.yolo_backend import ensure_weights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--imgsz", default=None)
    p.add_argument("--formats", default="onnx_fp32,onnx_fp16,onnx_int8,onnx_int8_mixed")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    root = project_root()
    models_dir = root / cfg["paths"]["models_dir"]
    data_root = Path(args.data_root or root / cfg["paths"]["data_root"])
    calib_dir = data_root / "coco" / "calibration"
    formats = csv_list(args.formats)

    for model in csv_list(args.models) or cfg["experiment"]["models"]:
        weights = ensure_weights(model, models_dir)
        for imgsz in csv_list(args.imgsz, int) or cfg["experiment"]["imgsz"]:
            fp32 = export_onnx_fp32(weights, imgsz, model_path(models_dir, model, imgsz, "onnx_fp32"))
            print(f"{model} {imgsz} onnx_fp32: {file_size_mb(fp32)} MB")
            if "onnx_fp16" in formats:
                out = convert_fp16(fp32, model_path(models_dir, model, imgsz, "onnx_fp16"))
                print(f"{model} {imgsz} onnx_fp16: {file_size_mb(out)} MB")
            for fmt, mixed in (("onnx_int8", False), ("onnx_int8_mixed", True)):
                if fmt in formats:
                    out = quantize_int8(fp32, model_path(models_dir, model, imgsz, fmt), calib_dir,
                                        imgsz, exclude_head=mixed,
                                        max_images=cfg["calibration"]["num_images"])
                    print(f"{model} {imgsz} {fmt}: {file_size_mb(out)} MB")


if __name__ == "__main__":
    main()
