"""How bright are typical images? Use this to choose the low-light threshold with data, not a guess.

Usage: python scripts/11_luma_stats.py --data-root data [--n 1000]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from smart_vision.config import load_config, project_root
from smart_vision.runtime.enhance import LowLightEnhancer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    folder = Path(args.data_root or project_root() / cfg["paths"]["data_root"]) / "coco" / "val2017"
    paths = sorted(folder.glob("*.jpg"))[: args.n]
    if not paths:
        raise SystemExit(f"No images in {folder}")
    lumas = np.array([LowLightEnhancer.mean_luma(cv2.imread(str(p))) for p in paths])
    print(f"{len(lumas)} images from {folder}")
    print("mean luminance percentiles (0-255):",
          {q: round(float(np.percentile(lumas, q)), 1) for q in (1, 5, 10, 25, 50, 75, 95)})
    for thr in (30, 40, 50, 60, 70, 90):
        print(f"  fraction of normal images darker than {thr}: {np.mean(lumas < thr):.1%}")
    print("A good threshold flags very few normal images, so the enhancer only fires in real darkness.")


if __name__ == "__main__":
    main()
