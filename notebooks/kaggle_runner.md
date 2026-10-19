# Running on Kaggle

The repo is the source of truth. The notebook only clones it and runs scripts.
Settings: Internet = On. CPU is enough; a GPU is optional.

```python
# Cell 1: code
!git clone https://github.com/hibyy/smart-vision-sensor.git
%cd smart-vision-sensor
!pip install -q -r requirements.txt
!pip install -q -e .
!python -m pytest -q

# Cell 2: environment record
!python scripts/00_env_report.py --tag kaggle_cpu

# Cell 3: data (about 1 GB) and calibration set (needs ~3 GB RAM the first time)
!python scripts/01_prepare_data.py --data-root /kaggle/working/data --calibration

# Cell 4: weights and a first quick experiment
!python scripts/02_download_weights.py --models yolo11n
!python scripts/03_export_models.py --data-root /kaggle/working/data --models yolo11n --imgsz 640
!python scripts/04_eval_accuracy.py --tag kaggle_cpu --data-root /kaggle/working/data --models yolo11n --imgsz 640 --max-images 500
```

Then continue with `docs/RUNBOOK.md`. Files under `/kaggle/working` are lost when the session ends:
download `results/tables/*.csv`, `results/env/*.json` and `configs/manifests/calibration_train2017.json`
(commit the manifest). Adjust the clone URL if your repository name differs.
