"""Configuration loading."""
from __future__ import annotations

from pathlib import Path

import yaml


def project_root() -> Path:
    """Repository root (src/smart_vision/config.py -> repo)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else project_root() / "configs" / "benchmark.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
