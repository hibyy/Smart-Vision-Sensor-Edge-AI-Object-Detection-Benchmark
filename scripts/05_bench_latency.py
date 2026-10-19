"""CPU latency for each (model, imgsz, format, threads). One fresh process per configuration
so thread settings and peak memory are not polluted by earlier runs. Resumable.

Usage: python scripts/05_bench_latency.py --tag laptop_cpu --data-root data --models yolo11n --imgsz 640 --threads 1,4
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from smart_vision.bench.export import file_size_mb
from smart_vision.bench.results import append_row, csv_list, done_keys, grid, model_path
from smart_vision.config import load_config, project_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="e.g. laptop_cpu")
    p.add_argument("--data-root", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--imgsz", default=None)
    p.add_argument("--formats", default=None)
    p.add_argument("--threads", default=None, help="comma list, default from config")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    root = project_root()
    models_dir = root / cfg["paths"]["models_dir"]
    images_dir = Path(args.data_root or root / cfg["paths"]["data_root"]) / "coco" / "val2017"
    lat = cfg["latency"]
    out_csv = root / cfg["paths"]["results_root"] / "tables" / f"latency_{args.tag}.csv"
    done = done_keys(out_csv, ["model", "imgsz", "format", "threads"])

    for model, imgsz, fmt in grid(cfg["experiment"], csv_list(args.models), csv_list(args.imgsz, int),
                                  csv_list(args.formats)):
        path = model_path(models_dir, model, imgsz, fmt)
        if not path.exists():
            print(f"skip (missing file): {path}")
            continue
        for threads in csv_list(args.threads, int) or cfg["experiment"]["cpu_threads"]:
            if (model, str(imgsz), fmt, str(threads)) in done:
                print(f"skip (done): {model} {imgsz} {fmt} threads={threads}")
                continue
            cmd = [sys.executable, "-m", "smart_vision.bench.latency", "--weights", str(path),
                   "--imgsz", str(imgsz), "--threads", str(threads), "--images-dir", str(images_dir),
                   "--n-images", str(lat["n_images"]), "--warmup", str(lat["warmup_runs"]),
                   "--runs", str(args.runs or lat["timed_runs"])]
            print(f"benchmarking {model} {imgsz} {fmt} threads={threads} ...")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
            if proc.returncode != 0 or line is None:
                print(f"  FAILED: {proc.stderr.strip()[-400:]}")
                continue
            stats = json.loads(line[len("RESULT_JSON "):])
            append_row(out_csv, {"tag": args.tag, "model": model, "imgsz": imgsz, "format": fmt,
                                 "threads": threads, "size_mb": file_size_mb(path), **stats})
            print(f"  median e2e {stats['e2e_median_ms']} ms | model {stats['model_median_ms']} ms | {stats['fps']} FPS")


if __name__ == "__main__":
    main()
