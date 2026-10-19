"""Generate result tables and figures from the CSV files (no hand-typed numbers).

Usage: python scripts/08_make_report.py --acc-tag kaggle_cpu --lat-tag laptop_cpu --threads 4 [--rob-tag kaggle_cpu]
"""
import argparse

from smart_vision.bench.report import (build_summary, pareto_front, plot_pareto, plot_robustness,
                                       robustness_table, to_markdown)
from smart_vision.bench.results import read_rows
from smart_vision.config import load_config, project_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--acc-tag", required=True)
    p.add_argument("--lat-tag", required=True)
    p.add_argument("--rob-tag", default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    res = project_root() / cfg["paths"]["results_root"]
    tables, figures = res / "tables", res / "figures"

    summary = build_summary(read_rows(tables / f"accuracy_{args.acc_tag}.csv"),
                            read_rows(tables / f"latency_{args.lat_tag}.csv"), args.threads)
    if not summary:
        raise SystemExit("No matching accuracy/latency rows. Check the tags and the --threads value.")
    cols = ["model", "imgsz", "format", "AP", "AP50", "AP_small", "e2e_median_ms", "e2e_p95_ms",
            "size_mb", "memory_delta_mb", "images"]
    front = pareto_front(summary)
    md = (f"# Results\n\nAccuracy: `{args.acc_tag}` | Latency: `{args.lat_tag}` ({args.threads} threads, batch 1, CPU)\n\n"
          f"## All configurations\n\n{to_markdown(summary, cols)}\n"
          f"## Pareto-optimal configurations\n\n{to_markdown(front, cols)}\n")
    plot_pareto(summary, figures / "pareto_ap_vs_latency.png")

    if args.rob_tag:
        rob = robustness_table(read_rows(tables / f"robustness_{args.rob_tag}.csv"))
        if rob:
            keys = sorted({k for r in rob for k in r})
            md += f"\n## Robustness (AP per condition)\n\n{to_markdown(rob, ['model', 'imgsz', 'format'] + [k for k in keys if k not in ('model', 'imgsz', 'format')])}\n"
            plot_robustness(read_rows(tables / f"robustness_{args.rob_tag}.csv"), figures / "robustness.png")

    (tables / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved: {tables / 'summary.md'} and figures in {figures}")


if __name__ == "__main__":
    main()
