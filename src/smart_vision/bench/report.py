"""Merge result CSVs into the final tables and figures. Everything is generated from files."""
from __future__ import annotations

from pathlib import Path


def build_summary(acc_rows: list[dict], lat_rows: list[dict], threads: int) -> list[dict]:
    """Join accuracy and latency rows on (model, imgsz, format); latency taken at `threads`."""
    lat = {(r["model"], str(r["imgsz"]), r["format"]): r for r in lat_rows if str(r["threads"]) == str(threads)}
    out = []
    for a in acc_rows:
        key = (a["model"], str(a["imgsz"]), a["format"])
        l = lat.get(key)
        if l is None:
            continue
        out.append({
            "model": a["model"], "imgsz": int(a["imgsz"]), "format": a["format"],
            "AP": float(a["AP"]), "AP50": float(a["AP50"]), "AP_small": float(a["AP_small"]),
            "images": int(float(a["images"])), "size_mb": float(l["size_mb"]),
            "e2e_median_ms": float(l["e2e_median_ms"]), "e2e_p95_ms": float(l["e2e_p95_ms"]),
            "model_median_ms": float(l["model_median_ms"]), "fps": float(l["fps"]),
            "memory_delta_mb": float(l.get("memory_delta_mb") or 0.0),
        })
    return sorted(out, key=lambda r: (r["e2e_median_ms"], -r["AP"]))


def pareto_front(rows: list[dict], x: str = "e2e_median_ms", y: str = "AP") -> list[dict]:
    """Non-dominated rows: no other row is at least as fast AND at least as accurate (one strictly better)."""
    front = []
    for r in rows:
        dominated = any(o[x] <= r[x] and o[y] >= r[y] and (o[x] < r[x] or o[y] > r[y]) for o in rows)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r[x])


def to_markdown(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_no rows_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)
    return head + body


def plot_pareto(rows: list[dict], path: str | Path, title: str = "Accuracy vs CPU latency") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"pytorch_fp32": "o", "onnx_fp32": "s", "onnx_fp16": "^", "onnx_int8": "D", "onnx_int8_mixed": "P"}
    for fmt in sorted({r["format"] for r in rows}):
        pts = [r for r in rows if r["format"] == fmt]
        ax.scatter([r["e2e_median_ms"] for r in pts], [r["AP"] for r in pts],
                   s=[40 + 6 * r["size_mb"] ** 0.5 * 4 for r in pts], marker=markers.get(fmt, "o"),
                   alpha=0.75, label=fmt)
    front = pareto_front(rows)
    ax.plot([r["e2e_median_ms"] for r in front], [r["AP"] for r in front], "k--", lw=1, label="Pareto front")
    for r in front:
        ax.annotate(f"{r['model']}@{r['imgsz']}", (r["e2e_median_ms"], r["AP"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("End-to-end median latency (ms, CPU)")
    ax.set_ylabel("COCO AP (0.5:0.95)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def robustness_table(rows: list[dict]) -> list[dict]:
    """One row per (model, imgsz, format) with AP per condition and relative drop vs clean."""
    grouped: dict[tuple, dict] = {}
    for r in rows:
        key = (r["model"], int(r["imgsz"]), r["format"])
        grouped.setdefault(key, {})[r["corruption"]] = float(r["AP"])
    out = []
    for (model, imgsz, fmt), aps in sorted(grouped.items()):
        clean = aps.get("clean")
        row = {"model": model, "imgsz": imgsz, "format": fmt, "clean": clean}
        for cond, ap in aps.items():
            if cond == "clean":
                continue
            row[cond] = ap
            row[f"{cond}_drop_%"] = round(100 * (clean - ap) / clean, 1) if clean else ""
        out.append(row)
    return out


def plot_robustness(rows: list[dict], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    table = robustness_table(rows)
    conds = sorted({c[:-7] for r in table for c in r if c.endswith("_drop_%")})
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(1, len(table))
    for i, r in enumerate(table):
        drops = [r.get(f"{c}_drop_%") or 0 for c in conds]
        ax.bar(np.arange(len(conds)) + i * width, drops, width, label=f"{r['model']}@{r['imgsz']} {r['format']}")
    ax.set_xticks(np.arange(len(conds)) + width * (len(table) - 1) / 2)
    ax.set_xticklabels(conds)
    ax.set_ylabel("AP drop vs clean (%)")
    ax.set_title("Robustness to image degradation")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
