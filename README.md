# Smart Vision Sensor with Edge AI

A general-purpose smart vision sensor: it detects any of the 80 COCO object classes on
CPU-class edge hardware and outputs compact events (per-class counts, *appeared* /
*disappeared*) instead of raw video. The project also measures how model size, input
resolution, numeric precision (FP32, FP16, INT8) and an adaptive inference policy affect
accuracy, latency and size, using one reproducible protocol.

> **Status:** code complete and smoke-tested on a small scale. Research results are **not**
> in this repository until you run the full experiments (see `docs/RUNBOOK.md`). Every
> number in `results/` is produced by a script from saved files; none is typed by hand.

## What is in the box

```
Sensor input (image folder / video / webcam)
   -> Perception: YOLO11 / YOLO26 (PyTorch or ONNX FP32 / FP16 / INT8) + ByteTrack
   -> Event engine: per-class counts, debounced appeared / disappeared events
   -> Output: compact JSONL (report by exception) + live dashboard
   -> Telemetry: FPS, latency, compute fraction, bandwidth saved
```

* **Adaptive inference:** a motion gate skips detection on near-static frames.
* **Benchmark suite:** COCO AP, CPU latency (median / p95, per thread count), size, memory,
  robustness (blur, low light, noise, occlusion, JPEG), Pareto plot.
* **Dashboard:** `streamlit run app/dashboard.py`.

## Quick start (laptop, CPU)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest -q                 # unit tests
python scripts/smoke_test.py        # end-to-end check with a real model (needs internet once)
streamlit run app/dashboard.py      # choose "Sample images" and press Start
```

Full experiments: follow **`docs/RUNBOOK.md`** (laptop and Kaggle).

## Layout

| Path | Purpose |
|------|---------|
| `configs/benchmark.yaml` | All experiment settings |
| `src/smart_vision/sensor/` | Frame sources: image folder, video, webcam |
| `src/smart_vision/perception/` | Detector interface and the Ultralytics backend (.pt / .onnx) |
| `src/smart_vision/events/` | Event engine and event-agreement metric |
| `src/smart_vision/runtime/` | Sensor pipeline, motion gate, telemetry, drawing |
| `src/smart_vision/bench/` | Export / quantization, COCO evaluation, latency, corruptions, report |
| `scripts/` | Numbered entry points: `00` env, `01` data, `02` weights, `03` export, `04` accuracy, `05` latency, `06` robustness, `07` run sensor, `08` report, `09` adaptive eval, `smoke_test.py` |
| `app/dashboard.py` | Streamlit dashboard |
| `docs/` | Proposal, references, runbook, hardware description |
| `tests/` | Unit and dashboard tests |

## Honest limitations

* Laptop and Kaggle CPUs are not embedded devices; no energy measurement.
* Post-training INT8 quantization only; the INT8 variants may lose accuracy on the detection head,
  which is why an "int8_mixed" variant is included.
* Event agreement (adaptive vs full-rate) compares against our own pipeline, not ground truth.
* Pretrained COCO models may not transfer to other domains.
