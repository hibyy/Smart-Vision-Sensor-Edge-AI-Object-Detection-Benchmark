"""Deployment advisor: pick the best configuration for a latency / size / accuracy budget.

Reads the accuracy and latency CSVs produced by scripts 04 and 05.
Example: python scripts/10_recommend_config.py --acc-tag kaggle_cpu --lat-tag laptop_cpu --threads 2 \
             --max-latency-ms 60 --max-size-mb 15 --min-ap 0.25
"""
import argparse

from smart_vision.bench.report import build_summary, recommend, to_markdown
from smart_vision.bench.results import read_rows
from smart_vision.config import load_config, project_root

COLS = ["model", "imgsz", "format", "AP", "e2e_median_ms", "e2e_p95_ms", "size_mb", "images"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--acc-tag", required=True)
    p.add_argument("--lat-tag", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-latency-ms", type=float, default=None)
    p.add_argument("--max-size-mb", type=float, default=None)
    p.add_argument("--min-ap", type=float, default=None, help="COCO AP as a fraction, e.g. 0.25")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    tables = project_root() / cfg["paths"]["results_root"] / "tables"
    rows = build_summary(read_rows(tables / f"accuracy_{args.acc_tag}.csv"),
                         read_rows(tables / f"latency_{args.lat_tag}.csv"), args.threads)
    if not rows:
        raise SystemExit("No matching accuracy/latency rows. Check the tags and --threads.")
    best = recommend(rows, args.max_latency_ms, args.max_size_mb, args.min_ap)
    if not best:
        fastest = sorted(rows, key=lambda r: r["e2e_median_ms"])[:3]
        print("No configuration satisfies the budget. Fastest available:\n")
        print(to_markdown(fastest, COLS))
        return
    print(f"Budget: latency <= {args.max_latency_ms} ms, size <= {args.max_size_mb} MB, AP >= {args.min_ap} "
          f"({args.threads} threads)\n")
    print(to_markdown(best, COLS))
    top = best[0]
    ext = "pt" if top["format"] == "pytorch_fp32" else "onnx"
    path = (f"models/{top['model']}.pt" if ext == "pt" else
            f"models/onnx/{top['model']}_{top['imgsz']}_{top['format'].replace('onnx_', '')}.onnx")
    print("Run it:\n  python scripts/07_run_sensor.py --source <video> "
          f"--weights {path} --imgsz {top['imgsz']} --threads {args.threads}")


if __name__ == "__main__":
    main()
