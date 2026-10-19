"""Run the smart vision sensor on a video, an image folder or a webcam.

Outputs compact JSONL events and prints telemetry.
Examples:
  python scripts/07_run_sensor.py --source clip.mp4 --weights models/yolo11n.pt
  python scripts/07_run_sensor.py --source webcam --weights models/onnx/yolo11n_640_int8.onnx --imgsz 640 --adaptive
"""
import argparse
import json
from pathlib import Path

from smart_vision.config import load_config, project_root
from smart_vision.events.engine import EventEngine
from smart_vision.perception.yolo_backend import YoloDetector
from smart_vision.runtime import MotionGate, SensorPipeline
from smart_vision.runtime.viz import draw_detections
from smart_vision.sensor.factory import open_source


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--classes", default=None, help="comma list of class names; default all")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--motion-threshold", type=float, default=2.0)
    p.add_argument("--max-skip", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out", default=None, help="JSONL output path")
    p.add_argument("--save-video", default=None, help="optional annotated mp4 path")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    s = cfg["sensor"]
    conf = args.conf if args.conf is not None else s["min_confidence"]
    classes = set(args.classes.split(",")) if args.classes else (set(s["classes"]) if s["classes"] else None)
    out = Path(args.out) if args.out else project_root() / cfg["paths"]["results_root"] / "sensor_events.jsonl"

    detector = YoloDetector(args.weights, imgsz=args.imgsz, conf=conf, threads=args.threads, track=True)
    engine = EventEngine(classes, conf, s["confirm_frames"], s["lost_frames"])
    gate = MotionGate(args.motion_threshold, args.max_skip) if args.adaptive else None
    pipeline = SensorPipeline(detector, engine, gate, jsonl_path=out)

    writer = None
    for frame, dets, record in pipeline.run(open_source(args.source), args.max_frames):
        if args.save_video:
            import cv2

            annotated = draw_detections(frame.image, dets)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
            writer.write(annotated)
        for ev in record["events"]:
            print(f"[frame {ev['frame']}] {ev['type']}: {ev['class']} (track {ev['track_id']})")
    if writer:
        writer.release()
    print(json.dumps(pipeline.telemetry.summary(), indent=2))
    print(f"Events written to {out}")


if __name__ == "__main__":
    main()
