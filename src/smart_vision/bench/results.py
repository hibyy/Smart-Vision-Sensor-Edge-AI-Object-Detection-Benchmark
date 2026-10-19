"""Resumable CSV result files: one row per configuration, never overwritten."""
from __future__ import annotations

import csv
from pathlib import Path


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def done_keys(path: str | Path, key_cols: list[str]) -> set[tuple]:
    return {tuple(str(r[c]) for c in key_cols) for r in read_rows(path)}


def append_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


FORMATS = ["pytorch_fp32", "onnx_fp32", "onnx_fp16", "onnx_int8", "onnx_int8_mixed"]


def model_path(models_dir: str | Path, model: str, imgsz: int, fmt: str) -> Path:
    """Where the weights for (model, imgsz, format) live."""
    models_dir = Path(models_dir)
    if fmt == "pytorch_fp32":
        return models_dir / f"{model}.pt"
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format: {fmt}")
    return models_dir / "onnx" / f"{model}_{imgsz}_{fmt.replace('onnx_', '')}.onnx"


def csv_list(value: str | None, cast=str) -> list | None:
    """'a,b,c' -> ['a','b','c'] (None stays None)."""
    if value is None:
        return None
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def grid(cfg_experiment: dict, models=None, imgsz=None, formats=None) -> list[tuple[str, int, str]]:
    """All (model, imgsz, format) combinations, optionally restricted."""
    from itertools import product

    models = models or cfg_experiment["models"]
    imgsz = imgsz or cfg_experiment["imgsz"]
    formats = formats or cfg_experiment["formats"]
    return list(product(models, imgsz, formats))
