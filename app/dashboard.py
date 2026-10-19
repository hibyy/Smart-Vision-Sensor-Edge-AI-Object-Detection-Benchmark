"""Smart Vision Sensor: live dashboard.

Run:  streamlit run app/dashboard.py
Shows detections, class statistics, FPS / latency, adaptive-inference savings, an event log
and, importantly, exactly which hardware and runtime produced the numbers on screen.
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
from smart_vision.env import describe_hardware
from smart_vision.events.engine import EventEngine
from smart_vision.perception.yolo_backend import YoloDetector, ensure_weights
from smart_vision.runtime import MotionGate, SensorPipeline
from smart_vision.runtime.enhance import METHODS, EnhancedDetector, LowLightEnhancer
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


def gpu_usable(weights: Path) -> tuple[bool, str]:
    """Is a GPU option meaningful for this model file?"""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "No CUDA GPU detected."
    except ImportError:
        return False, "PyTorch not installed."
    if weights.suffix.lower() == ".onnx":
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False, "ONNX models need onnxruntime-gpu for GPU; running on CPU."
    return True, ""


@st.cache_resource(show_spinner="Loading model...")
def load_detector(weights: str, imgsz: int, conf: float, device, threads) -> YoloDetector:
    det = YoloDetector(weights, imgsz=imgsz, conf=conf, device=device, threads=threads, track=True)
    det.detect(np.zeros((imgsz, imgsz, 3), np.uint8))  # warm-up: creates the runtime session
    det.reset_tracker()
    return det


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

with st.sidebar.expander("Get more models"):
    wanted = [m for m in CFG["experiment"]["models"] if not (MODELS_DIR / f"{m}.pt").exists()]
    if wanted:
        pick = st.multiselect("Download pretrained weights", wanted)
        if st.button("Download") and pick:
            for name in pick:
                with st.spinner(f"Downloading {name}..."):
                    ensure_weights(name, MODELS_DIR)
            st.rerun()
    else:
        st.caption("All configured PyTorch weights are present.")
    st.caption("ONNX / INT8 variants appear here after `python scripts/03_export_models.py`.")

if not models:
    st.info("No weights in `models/`. Use **Get more models** in the sidebar (try yolo11n) "
            "or run `python scripts/02_download_weights.py`.")
    st.stop()

weights = st.sidebar.selectbox("Model", models, format_func=lambda p: p.name)
fixed = infer_imgsz(weights)
if fixed:
    imgsz = fixed
    st.sidebar.caption(f"Input size fixed by the exported model: {fixed}px")
else:
    imgsz = st.sidebar.select_slider("Input size", options=CFG["experiment"]["imgsz"], value=640)

st.sidebar.subheader("Hardware")
gpu_ok, gpu_msg = gpu_usable(weights)
device_choice = st.sidebar.radio("Run on", ["CPU", "GPU"] if gpu_ok else ["CPU"], horizontal=True)
if not gpu_ok and gpu_msg:
    st.sidebar.caption(gpu_msg)
threads_choice = st.sidebar.selectbox("CPU threads (emulate a weaker device)", ["auto", 1, 2, 4, 8],
                                      disabled=device_choice == "GPU")
device = 0 if device_choice == "GPU" else "cpu"
threads = None if (threads_choice == "auto" or device_choice == "GPU") else int(threads_choice)

st.sidebar.subheader("Sensor")
conf = st.sidebar.slider("Min confidence", 0.05, 0.95, float(CFG["sensor"]["min_confidence"]), 0.05)
adaptive = st.sidebar.checkbox("Adaptive inference (skip static frames)", value=False)
motion_thr = st.sidebar.slider("Motion threshold", 0.5, 10.0, 2.0, 0.5, disabled=not adaptive)
max_skip = st.sidebar.slider("Max skipped frames", 1, 30, 10, disabled=not adaptive)
enhance_on = st.sidebar.checkbox("Low-light enhancement (experimental)", value=False,
                                 help="Brightens frames darker than the threshold before detection.")
enh_method = st.sidebar.selectbox("Method", METHODS, disabled=not enhance_on)
luma_thr = st.sidebar.slider("Dark if mean luminance below", 20, 120, 70, disabled=not enhance_on)
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

base_detector = load_detector(str(weights), imgsz, conf, device, threads)
runtime = {**describe_hardware(), **base_detector.runtime_info()}
st.info(f"**Running on:** {runtime['cpu'] if device == 'cpu' else runtime['gpu']}  |  "
        f"{runtime['backend']} ({runtime.get('execution provider', '?')})  |  "
        f"threads: {runtime.get('threads effective', runtime['threads requested'])}  |  "
        f"{runtime['model file']} at {runtime['input size']}")
with st.expander("Hardware & runtime details", expanded=False):
    st.table(pd.Series({k: str(v) for k, v in runtime.items()}, name="value"))

feed_col, stats_col = st.columns([3, 2])
feed_col.subheader("Live feed")
feed_slot = feed_col.empty()
note_slot = feed_col.empty()
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
    if source_kind == "Upload video" and upload is None:
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

    base_detector.reset_tracker()  # the cached detector must not carry tracks over from a previous run
    enhancer = LowLightEnhancer(luma_threshold=float(luma_thr), method=enh_method) if enhance_on else None
    detector = EnhancedDetector(base_detector, enhancer) if enhancer else base_detector
    engine = EventEngine(None, conf, CFG["sensor"]["confirm_frames"], CFG["sensor"]["lost_frames"])
    gate = MotionGate(motion_thr, max_skip) if adaptive else None
    jsonl = Path(tempfile.mkdtemp()) / "events.jsonl"
    pipeline = SensorPipeline(detector, engine, gate, jsonl_path=jsonl)

    cumulative: Counter = Counter()
    seen_tracks: set[int] = set()
    log: list[dict] = []
    for frame, dets, record in pipeline.run(source, int(max_frames)):
        feed_slot.image(draw_detections(frame.image, dets)[:, :, ::-1], channels="RGB", use_container_width=True)
        if enhancer is not None and enhancer.last_luma is not None:
            note_slot.caption(f"Frame brightness {enhancer.last_luma:.0f} "
                              f"-> {'enhanced (' + enh_method + ')' if enhancer.last_applied else 'left unchanged'}")
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
    with st.expander("Telemetry (with the hardware that produced it)"):
        st.json({"telemetry": tel, "runtime": {k: str(v) for k, v in runtime.items()}})
    st.download_button("Download events (JSONL)", jsonl.read_bytes(), "events.jsonl")
else:
    st.info("Choose a model and a source in the sidebar, then press Start.")
