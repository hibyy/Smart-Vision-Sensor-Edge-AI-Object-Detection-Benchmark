"""End-to-end smoke test (about 1-3 minutes on CPU, needs internet once for the weights).

It checks the whole chain on tiny inputs: weights -> ONNX FP32/FP16/INT8 export ->
detection in every format -> latency worker -> sensor pipeline (full vs adaptive) -> event agreement.
It does NOT produce research results; use scripts 03-09 for that.

Usage: python scripts/smoke_test.py [--model yolo11n] [--imgsz 320]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from smart_vision.bench.export import convert_fp16, export_onnx_fp32, file_size_mb, quantize_int8
from smart_vision.config import project_root
from smart_vision.events.agreement import event_agreement
from smart_vision.events.engine import EventEngine
from smart_vision.perception.yolo_backend import YoloDetector, ensure_weights
from smart_vision.runtime import MotionGate, SensorPipeline
from smart_vision.sensor.sources import VideoSource


def sample_images():
    from ultralytics.utils import ASSETS

    return [cv2.imread(str(ASSETS / n)) for n in ("bus.jpg", "zidane.jpg")]


def make_calibration(folder: Path, n: int = 24) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    srcs = sample_images()
    for i in range(n):
        im = srcs[i % 2]
        h, w = im.shape[:2]
        y, x = int(rng.integers(0, h // 5)), int(rng.integers(0, w // 5))
        crop = im[y:h - int(rng.integers(0, h // 5)), x:w - int(rng.integers(0, w // 5))]
        crop = np.clip(crop.astype(np.float32) * rng.uniform(0.7, 1.2), 0, 255).astype(np.uint8)
        cv2.imwrite(str(folder / f"{i:03d}.jpg"), crop)


def make_video(path: Path) -> None:
    """bus (static) -> zidane (static) -> empty gray -> bus again. Known, easy-to-reason-about changes."""
    bus, zidane = sample_images()
    size = (640, 480)
    clips = [cv2.resize(bus, size), cv2.resize(zidane, size), np.full((size[1], size[0], 3), 120, np.uint8),
             cv2.resize(bus, size)]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    for clip in clips:
        for _ in range(25):
            writer.write(clip)
    writer.release()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n")
    p.add_argument("--imgsz", type=int, default=320)
    args = p.parse_args()
    root = project_root()
    work = Path(tempfile.mkdtemp(prefix="svs_smoke_"))
    print(f"workspace: {work}")

    print("\n[1/5] weights")
    weights = ensure_weights(args.model, root / "models")
    print(f"  {weights} ({file_size_mb(weights)} MB)")

    print("\n[2/5] export and quantization")
    make_calibration(work / "calib")
    fp32 = export_onnx_fp32(weights, args.imgsz, work / "onnx" / "fp32.onnx")
    paths = {
        "pytorch_fp32": weights,
        "onnx_fp32": fp32,
        "onnx_fp16": convert_fp16(fp32, work / "onnx" / "fp16.onnx"),
        "onnx_int8": quantize_int8(fp32, work / "onnx" / "int8.onnx", work / "calib", args.imgsz),
        "onnx_int8_mixed": quantize_int8(fp32, work / "onnx" / "int8m.onnx", work / "calib", args.imgsz,
                                         exclude_head=True),
    }

    print("\n[3/5] detection in every format (bus.jpg) + latency worker")
    bus = sample_images()[0]
    print(f"  {'format':16s} {'MB':>7s} {'objects':>8s} {'bus found':>10s} {'median ms (10 runs)':>20s}")
    for fmt, path in paths.items():
        det = YoloDetector(path, imgsz=args.imgsz, conf=0.25, threads=2)
        dets = det.detect(bus)
        found = any(d.class_name == "bus" for d in dets)
        cmd = [sys.executable, "-m", "smart_vision.bench.latency", "--weights", str(path), "--imgsz",
               str(args.imgsz), "--threads", "2", "--images-dir", str(work / "calib"), "--n-images", "8",
               "--warmup", "3", "--runs", "10"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        if line is None:
            raise SystemExit(f"latency worker failed for {fmt}: {proc.stderr[-500:]}")
        lat = json.loads(line[len("RESULT_JSON "):])
        print(f"  {fmt:16s} {file_size_mb(path):7.2f} {len(dets):8d} {str(found):>10s} {lat['e2e_median_ms']:20.1f}")
        if fmt in ("pytorch_fp32", "onnx_fp32", "onnx_fp16"):
            assert found, f"{fmt}: expected to detect the bus"

    print("\n[4/5] sensor pipeline on a synthetic video (full vs adaptive)")
    video = work / "demo.avi"
    make_video(video)
    results = {}
    for mode, gate in (("full", None), ("adaptive", MotionGate(2.0, 10))):
        det = YoloDetector(paths["onnx_fp32"], imgsz=args.imgsz, conf=0.25, threads=2, track=True)
        engine = EventEngine(None, 0.25, 3, 5)
        pipe = SensorPipeline(det, engine, gate, jsonl_path=work / f"events_{mode}.jsonl")
        for _ in pipe.run(VideoSource(video)):
            pass
        tel = pipe.telemetry.summary()
        results[mode] = (pipe.events, tel)
        kinds = sorted({(e["type"], e["class"]) for e in pipe.events})
        print(f"  {mode:8s} frames={tel['frames']} inferred={tel['frames_inferred']} "
              f"compute={tel['compute_fraction']:.0%} fps={tel['fps']} records={tel['records_emitted']} "
              f"bandwidth_reduction={tel['bandwidth_reduction']:.4%}")
        print(f"           events: {kinds[:8]}")
    agree = event_agreement(results["full"][0], results["adaptive"][0], tolerance_frames=10)
    print(f"  agreement adaptive vs full: {agree}")
    assert results["adaptive"][1]["compute_fraction"] < 1.0, "adaptive policy should skip some frames"
    assert results["full"][1]["frames_inferred"] == results["full"][1]["frames"]

    print("\n[5/5] done")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
