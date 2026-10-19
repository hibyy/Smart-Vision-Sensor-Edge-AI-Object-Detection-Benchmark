# Project guide (plain language)

Read this first if you want to understand, explain or defend the project.

## 1. The idea in three sentences
A normal camera sends video. A **smart vision sensor** looks at the video itself and only sends
*meaning*: "2 persons and 1 backpack are visible; a dog appeared". This project builds that sensor in
software, runs it on ordinary CPUs (the kind of hardware an edge device has, not a big GPU), and
**measures** how model size, image size and number precision change accuracy, speed and size.

## 2. What was done, phase by phase
| Phase | What it produced | Where |
|---|---|---|
| 0. Review | Your roadmap, your Kaggle notebook and your local setup were read; weaknesses listed (500 images, GPU-only, no export, hand-typed numbers) | `docs/pilot_findings.md` |
| 1. Framing | Research question, hypotheses H1 to H6, scope, metrics, protocol; 7 verified references | `docs/01_proposal.md`, `docs/papers.md` |
| 2. Foundations | Config file, environment report, COCO download and verification, seeded calibration set, tests | `configs/`, `src/smart_vision/data.py`, `env.py`, scripts `00`, `01` |
| 3. Baseline tools | One detector class for .pt and .onnx, a COCO evaluator (checks class names, supports seeded subsets), resumable result files | `perception/yolo_backend.py`, `bench/evaluate.py`, `bench/results.py`, script `04` |
| 4. The sensor | Frame sources, tracker, event engine (counts, appeared, disappeared), pipeline that writes only changes, telemetry, dashboard | `sensor/`, `events/`, `runtime/`, `app/dashboard.py`, script `07` |
| 5. Optimization | ONNX export, FP16, INT8, INT8-mixed; latency harness (median, p95, threads, memory growth); motion-gated adaptive inference | `bench/export.py`, `bench/latency.py`, `runtime/policy.py`, scripts `03`, `05`, `09` |
| 6. Robustness and decisions | Image degradations (blur, dark, low-light, noise, occlusion, JPEG), low-light enhancer (experimental), tables, Pareto plot, deployment advisor | `bench/corruptions.py`, `runtime/enhance.py`, `bench/report.py`, scripts `06`, `08`, `10`, `11` |
| 7. Packaging | README, runbook, smoke test, 40+ unit tests | `README.md`, `docs/RUNBOOK.md`, `scripts/smoke_test.py`, `tests/` |

## 3. How data flows
```
video / images / webcam  ->  [low-light enhancer, optional]  ->  detector (.pt or .onnx)
   -> tracker (gives each object an id)  ->  event engine  ->  compact JSON records + dashboard
                                   ^
   motion gate: skips the detector when the scene did not change
```

## 4. Words you will need
* **AP** (average precision): COCO's accuracy score for detection, 0 to 1 (often shown as %).
* **Latency**: time to process one image. *Median* is typical, *p95* is "almost worst case".
* **ONNX**: a portable model file format; ONNX Runtime runs it fast on CPUs.
* **Quantization (INT8)**: store numbers as 8-bit integers instead of 32-bit floats: smaller and often faster.
  *Post-training* means no re-training, only a small calibration set (500 train images here).
* **FP16**: 16-bit floats. Smaller file; on CPUs it usually does *not* get faster (we measure, not assume).
* **Tracker**: gives the same object the same id across frames, which is what makes "appeared" events possible.
* **Motion gate**: skips detection when the image barely changed.
* **Pareto front**: configurations where no other one is both faster and more accurate.

## 5. Where the strengths are (and are not)
Real strengths, all visible in the code:
1. **Systems thinking**: a working sensor that outputs events and measures its own bandwidth saving,
   not just a model in a notebook.
2. **Measurement discipline**: fresh process per benchmark, warm-up, median and p95, thread control,
   memory growth, hardware recorded next to every result, no hand-typed numbers.
3. **You report negative results.** Your pilot found that the adaptive strategy was *slower* and
   `pilot_findings.md` explains why with arithmetic (71.5% rerun vs a 12% break-even). Reviewers value this.
4. **Edge-relevant engineering**: INT8 quantization pipeline, a "head kept in float" ablation, CPU thread
   limits to emulate weaker devices, a deployment advisor that picks a configuration for a latency budget.
5. **Reproducibility**: resumable runs, seeded subsets, tests, a smoke test, one runbook.

Honest weak spots (say them before a reviewer does):
* No real embedded device and no energy measurement. A borrowed Raspberry Pi, Jetson or phone run would
  be the single biggest credibility upgrade (the benchmark scripts should run on ARM Linux, untested).
* Pretrained models, no new model and no new algorithm: this is a rigorous engineering study.
* Final numbers do not exist until you run the experiments in `docs/RUNBOOK.md`.
* Low light is a real weakness: our simple brightening did **not** help in two quick checks and often hurt
  (it amplifies noise). The ExDark paper (`docs/papers.md`) explains why simple fixes fall short. Report this
  honestly, and treat real-night evaluation or low-light-aware training as future work.

## 6. What is tested
Unit tests cover data checks, event logic, the motion gate, the pipeline, the COCO evaluator (with an
oracle detector that must score about 1.0), corruptions, reports, the advisor and the dashboard.
`scripts/smoke_test.py` runs the real chain with a real model on tiny inputs. Not tested by anyone yet:
the full COCO download, Windows paths, Kaggle, the webcam, long runs.
