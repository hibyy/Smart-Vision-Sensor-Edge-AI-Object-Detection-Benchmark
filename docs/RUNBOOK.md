# Runbook: from a fresh clone to the final results

Run steps in order. Every script is resumable: finished rows are skipped, so you can stop and
restart (important on Kaggle sessions). All outputs go to `results/`.

## 0. Check everything works (laptop, ~3 minutes)

```powershell
python -m pytest -q
python scripts/smoke_test.py
python scripts/00_env_report.py --tag laptop_cpu
```
Fill `docs/hardware.md` from `results/env/laptop_cpu.json`.

## 1. Kaggle: data, calibration, exports, accuracy

Settings: Internet On. CPU is enough (GPU is optional). See `notebooks/kaggle_runner.md`.

```bash
python scripts/00_env_report.py --tag kaggle_cpu
python scripts/01_prepare_data.py --data-root /kaggle/working/data --calibration
python scripts/02_download_weights.py
# quick pass first (500 images, one model) to catch problems early
python scripts/03_export_models.py --data-root /kaggle/working/data --models yolo11n --imgsz 640
python scripts/04_eval_accuracy.py --tag kaggle_cpu --data-root /kaggle/working/data --models yolo11n --imgsz 640 --max-images 500
# full runs (5000 images): core grid, then extend if time allows
python scripts/03_export_models.py --data-root /kaggle/working/data
python scripts/04_eval_accuracy.py --tag kaggle_cpu --data-root /kaggle/working/data --max-images 5000
python scripts/06_robustness.py --tag kaggle_cpu --data-root /kaggle/working/data --models yolo11n,yolo26n --imgsz 640 --formats onnx_fp32,onnx_int8
```
Download from Kaggle output: `results/tables/*.csv`, `results/env/*.json`,
`configs/manifests/calibration_train2017.json` (commit it). Do not commit weights or data.

Time budget: the full 4 x 4 x 5 grid on full val2017 is many CPU-hours. Start with
`--imgsz 640,416` and the formats you care about, run again later for the rest.

## 2. Laptop: latency (the edge-proxy numbers)

```powershell
python scripts/01_prepare_data.py --data-root data            # val2017 images (about 1 GB)
python scripts/02_download_weights.py
python scripts/03_export_models.py --data-root data --models yolo11n,yolo26n --imgsz 640,416
python scripts/05_bench_latency.py --tag laptop_cpu --data-root data --threads 1,2,4
```
Conditions: plugged in, high-performance power plan, other heavy apps closed.

## 3. Adaptive policy (H5)

Put 3 to 5 short videos in `data/videos/` (check each license; keep a list in `docs/videos.md`).
Use clips with both static and busy periods.
```powershell
python scripts/09_eval_adaptive.py --tag laptop_cpu --videos data/videos/a.mp4,data/videos/b.mp4 --weights models/yolo11n.pt --imgsz 640 --threads 4
```

## 4. Report

```powershell
python scripts/08_make_report.py --acc-tag kaggle_cpu --lat-tag laptop_cpu --rob-tag kaggle_cpu --threads 4
```
Outputs: `results/tables/summary.md`, `results/figures/pareto_ap_vs_latency.png`,
`results/figures/robustness.png`. Accuracy and latency are joined on (model, imgsz, format).

## 5. Demo

```powershell
streamlit run app/dashboard.py
python scripts/07_run_sensor.py --source data/videos/a.mp4 --weights models/onnx/yolo11n_640_int8.onnx --imgsz 640 --adaptive --save-video results/demo.mp4
```
Record a short screen capture of the dashboard for the README.

## 6. Write-up checklist

* Answer H1 to H5 as supported / rejected / inconclusive, citing table rows.
* State the hardware rows from `docs/hardware.md` next to every latency table.
* Note which results used a subset of images (the `images` column).
* Include failure examples (low light, occlusion) from the robustness runs.
