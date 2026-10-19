# Smart Vision Sensor with Edge AI: Project Proposal (v2)

**Phase 1 deliverable** | Status: draft v2 | Author: Hiba Zoubir

## 1. Problem statement

Cameras produce far more data than most applications need. A *smart vision
sensor* processes images locally and outputs only meaning (what objects are
present, when they appear or disappear) instead of raw video. On edge devices
this must run on CPU-class hardware with tight latency, memory and storage
limits. This project builds a general-purpose smart vision sensor that detects
any object class a COCO-trained detector supports (80 classes), and measures
how model choice, input resolution, numeric precision and an adaptive
inference policy affect accuracy, latency and size.

## 2. System architecture

```
Sensor input (images / video / webcam)
        -> Perception (optimized detector + tracker)
        -> Event engine (per-class counts, appeared / disappeared events)
        -> Output (compact JSON events, live dashboard)
        -> Telemetry (latency, FPS, memory, model size)
```

The sensor is class-agnostic: the set of monitored classes is a
configuration option (default: all classes), not a hard-coded application.

Example output for one frame:

```json
{"frame": 120, "t": 4.0,
 "counts": {"person": 2, "backpack": 1},
 "events": [{"type": "appeared", "class": "dog", "track_id": 7, "frame": 120}]}
```

## 3. Research question

> How can a smart vision sensor running on CPU-class edge hardware minimize
> computation and latency while preserving detection accuracy and the
> reliability of its event output?

## 4. Hypotheses (to be tested, not assumed)

| ID | Hypothesis | How it is falsified |
|----|------------|---------------------|
| H1 | ONNX INT8 post-training quantization cuts model size by about 3-4x and CPU latency by at least 1.5x, with at most 2 AP points lost. | Smaller speed-up, slower than FP32, or larger AP loss. |
| H2 | Lowering input resolution (640 to 416) reduces latency roughly in proportion to pixel count and hurts small-object AP the most. | Latency does not scale, or AP-small is not the most affected. |
| H3 | YOLO26n has lower CPU ONNX latency than YOLO11n at similar AP (vendor-claimed, verified independently). | No measurable latency gain on our hardware. |
| H4 | At equal latency, a larger model at lower resolution can beat a smaller model at higher resolution (or the reverse). | Pareto analysis shows which family dominates. |
| H5 | An adaptive policy (skip detection on near-static frames, track in between) cuts average per-frame compute by at least 30% on static-scene videos while keeping event agreement with the full-rate pipeline at 95% or more. | Savings below 30% or agreement below 95%. |

Negative results are valid results. Our pilot study showed a two-pass
adaptive-resolution strategy was slower than fixed 640, because it only breaks
even when fewer than about 12% of images need a second pass. H5 is designed
with that lesson in mind.

## 5. Scope

**In scope**
- Models: YOLO11n/s and YOLO26n/s (pretrained Ultralytics weights).
- Input sizes 320, 416, 512, 640; formats PyTorch FP32, ONNX FP32, ONNX FP16, ONNX INT8 and
  ONNX INT8-mixed (INT8 with the detection head kept in float).
- CPU thread counts 1, 2, 4.
- Accuracy on full COCO val2017 (5000 images, 80 classes).
- Sensor modules: input sources, detector backends, tracker, event engine,
  dashboard demo.
- Event-level evaluation on a small set of public videos (selected and
  license-checked in Phase 4).

**Out of scope**
- Custom hardware, energy measurement, training from scratch.
- Claims about embedded boards that were not measured.
- Domain-specific applications (counting, safety zones). The architecture
  supports them through configuration, but they are not evaluated.

## 6. Metrics

| Metric | Definition |
|--------|------------|
| AP, AP50, AP-small/medium/large | COCO bbox metrics (pycocotools), full val2017 |
| Latency (ms/frame) | Median and p95, batch 1, 20 warm-up + 200 timed runs; model-only and end-to-end |
| Throughput (FPS) | 1000 / median latency |
| Model size (MB), memory growth (MB) | File size; growth of process memory caused by loading and running the model (the process peak is also stored but includes the PyTorch runtime) |
| Event agreement | For H5: fraction of events of the full-rate pipeline that the adaptive pipeline reproduces (same class, within a time tolerance), plus false events. This measures agreement with the reference pipeline, not ground-truth accuracy. |
| Robustness | AP change under blur, low brightness, occlusion (Phase 6) |

## 7. Experimental protocol

1. Accuracy is measured with the runtime that would be deployed: ONNX formats
   run on CPU through ONNX Runtime, so INT8 kernels are really exercised. The
   core grid uses the full COCO val2017 set (5000 images); a quick mode with a
   seeded subset exists, and the number of evaluated images is stored in every
   result row. Runs are resumable across Kaggle sessions.
2. Latency is measured on CPU with ONNX Runtime on the author's laptop and on
   the Kaggle CPU. Every table states the hardware and software versions
   (`results/env/*.json`, `docs/hardware.md`).
3. Laptop conditions: plugged in, high-performance power plan, heavy apps
   closed, fixed thread count. Report model-only and end-to-end latency.
4. INT8 calibration uses train2017 images, never val2017.
5. Every number in the README or report is generated by a script from saved
   result files. No hand-typed metrics.
6. Seeds, manifests and configs are committed.

## 8. Success criteria

- AP on full val2017 for all 4 models x 4 resolutions x 4 formats, or a
  documented reason for any skipped configuration.
- CPU latency (median, p95) for every configuration.
- An accuracy-latency-size Pareto plot and a results table.
- H1-H5 each answered as supported, rejected or inconclusive, with evidence.
- A working sensor demo that outputs events and a live dashboard.
- A fresh clone reproduces the main results table following the README.

## 9. Known limitations (declared up front)

- Laptop and Kaggle CPUs are not embedded devices.
- No energy measurement.
- Pretrained COCO models may not transfer to other domains.
- Post-training quantization only; quantization-aware training is future work.
- All formats run through the Ultralytics runtime for pre- and post-processing, so
  end-to-end latency includes that overhead. FP16 is expected to reduce size but not
  CPU latency; we measure it instead of assuming.
- Event agreement (H5) is measured against our own full-rate pipeline.

## 10. Deliverables by phase

| Phase | Deliverable |
|-------|-------------|
| 1 | This proposal, `docs/papers.md`, repository skeleton |
| 2 | Environment, verified COCO data, calibration manifest, tests |
| 3 | Baseline AP for all models on full val2017 |
| 4 | Sensor pipeline (sources, detector, tracker, events) and dashboard demo |
| 5 | ONNX export, FP16/INT8, CPU benchmark harness, adaptive policy (H5) |
| 6 | Full results, Pareto plot, robustness tests, failure analysis |
| 7 | README, technical report, CV entry, release |

## 11. Relation to the pilot study

The earlier Kaggle notebook (500-image COCO subset, T4 GPU) is kept as a
pilot. Its findings motivate the questions above but are not reused as final
results.
