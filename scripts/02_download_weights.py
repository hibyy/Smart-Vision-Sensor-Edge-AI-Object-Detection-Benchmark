"""Download pretrained weights for the configured models into models/.

Usage: python scripts/02_download_weights.py [--models yolo11n,yolo26n]
"""
import argparse

from smart_vision.bench.results import csv_list
from smart_vision.config import load_config, project_root
from smart_vision.perception.yolo_backend import ensure_weights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=None)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    models_dir = args.models_dir or project_root() / cfg["paths"]["models_dir"]
    for name in csv_list(args.models) or cfg["experiment"]["models"]:
        path = ensure_weights(name, models_dir)
        print(f"OK {name}: {path}")


if __name__ == "__main__":
    main()
