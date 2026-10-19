# Pilot study (Kaggle notebook, 500-image COCO subset)

Source: `notebooks/00_pilot_study.ipynb`. All numbers below are copied from that notebook's saved
outputs and report cell. They are **subset results on a Tesla T4 GPU**, not official COCO scores, and
the notebook's timing protocols differ between models (see its own caveat), so treat the speeds as
indicative only.

## Setup
* Images: 500 from COCO val2017 (494 contain objects, 3,842 ground-truth objects, 79 of 80 classes).
* Models: pretrained YOLO11n/s and YOLO26n/s, confidence 0.001, COCO bbox AP via pycocotools.
* Class mapping between the model and COCO categories was verified (0 mismatches).

## Experiment A: model comparison at 640 px
| Model | AP | AP50 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 41.70% | 57.50% | 23.10% | 42.10% | 61.20% |
| YOLO26n | 43.55% | 59.93% | 24.05% | 44.66% | 63.62% |
| YOLO11s | 49.00% | 66.00% | 32.40% | 53.00% | 68.80% |
| YOLO26s | 50.59% | 67.09% | 30.71% | 54.65% | 69.65% |

YOLO26 was ahead overall in both sizes (+1.85 AP nano, +1.59 AP small), but YOLO26s had *lower*
small-object AP than YOLO11s (30.71% vs 32.40%): aggregate AP hides size-specific behaviour.

## Experiment B: input resolution (YOLO11s)
640 px: AP 49.00%, small-object AP 32.40%. 512 px: AP 46.21%, small-object AP 28.17%
(-2.79 AP overall, -4.23 small). Elapsed time for 500 images fell from about 6.57 s to 5.15 s (-21.6%).

## Experiment C: confidence-triggered adaptive resolution (YOLO26n)
Run at 512 px; rerun at 640 px if `1 - max confidence >= 0.1`. Threshold chosen on 300 calibration
images, evaluated on 200 held-out images (seed 2026).

| Strategy | AP | Mean runtime (5 runs, batch 8) |
|---|---:|---:|
| Fixed 512 | 44.65% | 1.799 s |
| Fixed 640 | 47.54% | 2.044 s |
| Adaptive 512 to 640 | 46.44% | 3.433 s |

The adaptive policy recovered part of the accuracy lost at 512 but was about 68% *slower* than fixed 640,
because 143 of 200 images (71.5%) were rerun.

**Why it failed, as arithmetic.** With a first pass of cost t512 on every image and a second pass of
cost t640 on a fraction f of them, adaptive beats fixed 640 only if t512 + f * t640 < t640, i.e.
f < 1 - t512/t640 = 1 - 1.799/2.044 = **12.0%**. The measured rerun rate was 71.5%, six times too high.
(This is a derived calculation from the notebook's numbers, assuming the second pass costs the same as a
fixed-640 pass.)

## How this pilot shaped the project
* The evaluation method (predict, convert to COCO JSON, COCOeval) and the class-mapping check were kept.
* The 12% break-even rule is why hypothesis H5 uses a **motion gate** (skip frames when nothing changes)
  instead of a second pass on most frames.
* Known weaknesses that the main project fixes: 500-image subset only, GPU-only timing, no export or
  quantization, no memory or size measurements, results typed into cells by hand, one big notebook.
