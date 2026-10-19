"""Smart Vision Sensor: live dashboard.

Run:  streamlit run app/dashboard.py
Shows detections, class statistics, FPS / latency, adaptive-inference savings and an event log.
"""
from __future__ import annotations

import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from smart_vision.config import load_config, project_root
from smart_vision.events.engine import EventEngine
from smart_vision.perception.yolo_backend import YoloDetector, ensure_weights
from smart_vision.runtime import MotionGate, SensorPipeline
from smart_vision.runtime.viz import draw_detections
from smart_vision.sensor.sources import VideoSource, WebcamSource
from smart_vision.types import Frame

ROOT = project_root()
CFG = load_config()
MODELS_DIR = ROOT / CFG["paths"]["models_dir"]

st.set_page_config(page_title="Smart Vision Sensor", page_icon="👁", layout="wide")


def list_models() -> list[Path]:
    found = sorted(MODELS_DIR.glob("*.pt")) + sorted((MODELS_DIR / "onnx").glob("*.onnx"))
    return [p for p in found if not p.name.endswith("_pre.onnx")]


def infer_imgsz(path: Path) -> int | None:
    """Exported ONNX files have a fixed input size encoded in the name: <model>_<size>_<fmt>.onnx."""
    m = re.search(r"_(\d+)_(?:fp32|fp16|int8_mixed|int8)\.onnx$", path.name)
    return int(m.group(1)) if m else None


@st.cache_resource(show_spinner="Loading model...")
def load_detector(weights: str, imgsz: int, conf: float) -> YoloDetector:
    return YoloDetector(weights, imgsz=imgsz, conf=conf, track=True)


def image_frames(files) -> list[Frame]:
    import cv2

    frames = []
    for i, f in enumerate(files):
        arr = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            frames.append(Frame(i, float(i), arr, f.name))
    return frames


def sample_frames() -> list[Frame]:
    """Two demo images that ship with Ultralytics, repeated so the tracker has something to follow."""
    import cv2
    from ultralytics.utils import ASSETS

    imgs = [cv2.imread(str(ASSETS / n)) for n in ("bus.jpg", "zidane.jpg")]
    order = [0] * 5 + [1] * 5
    return [Frame(i, i * 0.1, imgs[k], f"sample{k}") for i, k in enumerate(order)]


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Sensor settings")
models = list_models()
if not models:
    st.sidebar.warning("No weights found in models/.")
    if st.sidebar.button("Download yolo11n.pt"):
        with st.spinner("Downloading..."):
            ensure_weights("yolo11n", MODELS_DIR)
        st.rerun()
    st.info("Add weights to `models/` (run `python scripts/02_download_weights.py`) or use the download button.")
    st.stop()

weights = st.sidebar.selectbox("Model", models, format_func=lambda p: p.name)
fixed = infer_imgsz(weights)
if fixed:
    imgsz = fixed
    st.sidebar.caption(f"Input size fixed by the exported model: {fixed}px")
else:
    imgsz = st.sidebar.select_slider("Input size", options=CFG["experiment"]["imgsz"], value=640)
conf = st.sidebar.slider("Min confidence", 0.05, 0.95, float(CFG["sensor"]["min_confidence"]), 0.05)
adaptive = st.sidebar.checkbox("Adaptive inference (skip static frames)", value=False)
motion_thr = st.sidebar.slider("Motion threshold", 0.5, 10.0, 2.0, 0.5, disabled=not adaptive)
max_skip = st.sidebar.slider("Max skipped frames", 1, 30, 10, disabled=not adaptive)
max_frames = st.sidebar.number_input("Max frames", 10, 5000, 300, step=10)

source_kind = st.sidebar.radio("Source", ["Sample images", "Upload video", "Upload images", "Webcam"])
upload = None
if source_kind == "Upload video":
    upload = st.sidebar.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
elif source_kind == "Upload images":
    upload = st.sidebar.file_uploader("Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
run = st.sidebar.button("Start", type="primary")

# --------------------------------------------------------------------- main
st.title("Smart Vision Sensor")
st.caption("General-purpose object detection sensor: outputs compact events instead of video.")

feed_col, stats_col = st.columns([3, 2])
feed_col.subheader("Live feed")
feed_slot = feed_col.empty()
stats_col.subheader("Detection statistics")
chart_slot = stats_col.empty()
m1, m2, m3, m4, m5, m6 = st.columns(6)
tiles = [m.empty() for m in (m1, m2, m3, m4, m5, m6)]
st.subheader("Event log")
log_slot = st.empty()


def refresh(tel: dict, current: int, unique: int, avg_conf: float, cumulative: Counter, log: list[dict]) -> None:
    tiles[0].metric("Objects in frame", current)
    tiles[1].metric("Unique tracked", unique)
    tiles[2].metric("Avg confidence", f"{avg_conf:.0%}")
    tiles[3].metric("FPS", tel["fps"])
    tiles[4].metric("Inference (ms)", tel["median_inference_ms"])
    tiles[5].metric("Compute saved", f"{(1 - tel['compute_fraction']):.0%}")
    if cumulative:
        chart_slot.bar_chart(pd.Series(cumulative, name="detections"))
    if log:
        log_slot.dataframe(pd.DataFrame(log[-20:][::-1]), use_container_width=True, hide_index=True)


if run:
    if source_kind in ("Upload video",) and upload is None:
        st.warning("Upload a video first.")
        st.stop()
    if source_kind == "Upload images" and not upload:
        st.warning("Upload at least one image.")
        st.stop()

    if source_kind == "Sample images":
        source = sample_frames()
    elif source_kind == "Upload video":
        tmp = Path(tempfile.mkdtemp()) / upload.name
        tmp.write_bytes(upload.getvalue())
        source = VideoSource(tmp)
    elif source_kind == "Upload images":
        source = image_frames(upload)
    else:
        source = WebcamSource(0)

    detector = load_detector(str(weights), imgsz, conf)
    engine = EventEngine(None, conf, CFG["sensor"]["confirm_frames"], CFG["sensor"]["lost_frames"])
    gate = MotionGate(motion_thr, max_skip) if adaptive else None
    jsonl = Path(tempfile.mkdtemp()) / "events.jsonl"
    pipeline = SensorPipeline(detector, engine, gate, jsonl_path=jsonl)

    cumulative: Counter = Counter()
    seen_tracks: set[int] = set()
    log: list[dict] = []
    for frame, dets, record in pipeline.run(source, int(max_frames)):
        feed_slot.image(draw_detections(frame.image, dets)[:, :, ::-1], channels="RGB", use_container_width=True)
        if record["inference"]["ran"]:
            cumulative.update(d.class_name for d in dets)
        seen_tracks.update(d.track_id for d in dets if d.track_id is not None)
        log.extend(record["events"])
        if frame.index % 3 == 0:
            avg = float(np.mean([d.confidence for d in dets])) if dets else 0.0
            refresh(pipeline.telemetry.summary(), len(dets), len(seen_tracks), avg, cumulative, log)

    tel = pipeline.telemetry.summary()
    refresh(tel, 0, len(seen_tracks), 0.0, cumulative, log)
    st.success(f"Done: {tel['frames']} frames, {tel['frames_inferred']} inferred, "
               f"{tel['records_emitted']} records emitted "
               f"({tel['bytes_emitted']} bytes vs {tel['raw_frame_bytes']:,} raw frame bytes).")
    with st.expander("Telemetry"):
        st.json(tel)
    st.download_button("Download events (JSONL)", jsonl.read_bytes(), "events.jsonl")
else:
    st.info("Choose a model and a source in the sidebar, then press Start.")
