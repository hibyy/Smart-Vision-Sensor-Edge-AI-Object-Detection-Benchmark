"""Hypothesis H5: compare the adaptive (motion-gated) pipeline with the full-rate reference.

For each video: run the full pipeline once, then each (threshold, max_skip) setting.
Reports compute fraction, speed, and event agreement against the full-rate run.
Usage: python scripts/09_eval_adaptive.py --tag laptop_cpu --videos data/videos/a.mp4,data/videos/b.mp4 \
           --weights models/yolo11n.pt --imgsz 640 --threads 4
"""
import argparse
from itertools import product
from pathlib import Path

from smart_vision.bench.results import append_row, csv_list
from smart_vision.config import load_config, project_root
from smart_vision.events.agreement import event_agreement
from smart_vision.events.engine import EventEngine
from smart_vision.perception.yolo_backend import YoloDetector
from smart_vision.runtime import MotionGate, SensorPipeline
from smart_vision.sensor.factory import open_source


def run_once(weights, imgsz, threads, video, gate, cfg, max_frames):
    s = cfg["sensor"]
    detector = YoloDetector(weights, imgsz=imgsz, conf=s["min_confidence"], threads=threads, track=True)
    engine = EventEngine(None, s["min_confidence"], s["confirm_frames"], s["lost_frames"])
    pipeline = SensorPipeline(detector, engine, gate)
    for _ in pipeline.run(open_source(video), max_frames):
        pass
    return pipeline.events, pipeline.telemetry.summary()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--videos", required=True, help="comma-separated video paths")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    ad = cfg["adaptive"]
    out_csv = project_root() / cfg["paths"]["results_root"] / "tables" / f"adaptive_{args.tag}.csv"

    for video in csv_list(args.videos):
        ref_events, ref_tel = run_once(args.weights, args.imgsz, args.threads, video, None, cfg, args.max_frames)
        print(f"{Path(video).name}: reference {ref_tel['fps']} FPS, {len(ref_events)} events")
        append_row(out_csv, {"tag": args.tag, "video": Path(video).name, "mode": "full", "threshold": "",
                             "max_skip": "", "compute_fraction": 1.0, "fps": ref_tel["fps"],
                             "mean_ms_per_frame": ref_tel["mean_ms_per_frame"], "event_precision": 1.0,
                             "event_recall": 1.0, "event_f1": 1.0, "reference_events": len(ref_events),
                             "bandwidth_reduction": ref_tel["bandwidth_reduction"]})
        for thr, skip in product(ad["thresholds"], ad["max_skip"]):
            events, tel = run_once(args.weights, args.imgsz, args.threads, video,
                                   MotionGate(thr, skip), cfg, args.max_frames)
            agree = event_agreement(ref_events, events, ad["tolerance_frames"])
            append_row(out_csv, {"tag": args.tag, "video": Path(video).name, "mode": "adaptive",
                                 "threshold": thr, "max_skip": skip,
                                 "compute_fraction": tel["compute_fraction"], "fps": tel["fps"],
                                 "mean_ms_per_frame": tel["mean_ms_per_frame"],
                                 "event_precision": agree["precision"], "event_recall": agree["recall"],
                                 "event_f1": agree["f1"], "reference_events": agree["reference_events"],
                                 "bandwidth_reduction": tel["bandwidth_reduction"]})
            print(f"  thr={thr} skip={skip}: compute={tel['compute_fraction']:.0%} "
                  f"fps={tel['fps']} F1={agree['f1']}")


if __name__ == "__main__":
    main()
