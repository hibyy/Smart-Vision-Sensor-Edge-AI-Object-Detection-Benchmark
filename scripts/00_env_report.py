"""Record hardware/software details to results/env/<tag>.json.

Usage:  python scripts/00_env_report.py --tag laptop_cpu
"""
import argparse
import json

from smart_vision.config import load_config, project_root
from smart_vision.env import collect_env, save_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="e.g. laptop_cpu, kaggle_t4")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    info = collect_env()
    out = project_root() / cfg["paths"]["results_root"] / "env" / f"{args.tag}.json"
    save_env(info, out)
    print(json.dumps(info, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
