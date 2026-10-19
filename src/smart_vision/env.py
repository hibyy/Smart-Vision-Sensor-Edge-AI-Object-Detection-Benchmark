"""Collect hardware/software information so every result can be tied to its environment."""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = [
    "ultralytics", "torch", "torchvision", "onnx", "onnxruntime",
    "onnxruntime-gpu", "opencv-python-headless", "numpy", "pycocotools", "pandas", "pillow",
]


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _cpu_name() -> str:
    try:
        import cpuinfo  # py-cpuinfo

        brand = cpuinfo.get_cpu_info().get("brand_raw")
        if brand:
            return brand
    except Exception:
        pass
    return platform.processor() or "unknown"


def collect_env() -> dict:
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": _cpu_name(),
        "machine": platform.machine(),
        "packages": {p: _version(p) for p in PACKAGES if _version(p)},
    }
    try:
        import psutil

        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def save_env(info: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path
