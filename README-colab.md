# Running Road Damage Detection on Google Colab

## 1. Enable GPU Runtime

**Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**

Verify before doing anything else:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## 2. Mount Google Drive

Checkpoints are written directly to Drive during training, so they survive session
disconnects. Mount it first — every session, before running any training command.

```python
from google.colab import drive
drive.mount('/content/drive')

CKPT_DIR = "/content/drive/MyDrive/road_damage_ckpts"
```

---

## 3. Upload Project Files

Upload or clone your project files. Files required:

```
train.py
yolov11_model.py
rtdetr_model.py
losses.py
rdd2022_dataset.py
ablation.py
yolov11_baseline.yaml
yolov11_improved.yaml
rtdetr_baseline.yaml
rtdetr_improved.yaml
ablation.yaml
```

---

## 4. Install Dependencies

```python
!pip install ultralytics albumentations rich PyYAML opencv-python-headless kagglehub
```

---

## 5. Get the Dataset

Downloading directly inside Colab (~3 GB) is much faster than uploading manually.

### Step 1 — Kaggle API key

Go to [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**.
Upload the downloaded `kaggle.json`:

```python
from google.colab import files
files.upload()   # select kaggle.json
```

```python
import os, shutil
os.makedirs('/root/.kaggle', exist_ok=True)
shutil.copy('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)
```

### Step 2 — Download

```python
import kagglehub
path = kagglehub.dataset_download("aliabdelmenam/rdd-2022")
print("Downloaded to:", path)
```

### Step 3 — Find the RDD_SPLIT directory

```python
import os
for root, dirs, _ in os.walk(path):
    if 'train' in dirs and 'val' in dirs:
        DATA_DIR = root
        break
print("DATA_DIR =", DATA_DIR)
os.listdir(DATA_DIR)   # should show: ['train', 'val', 'test']
```

---

## 6. Run Training

Pass `--data-dir` and `--checkpoint-dir` on every run.
The script auto-detects `last.pt` in `CKPT_DIR` and resumes automatically after a disconnect.
A `training_history.json` with per-epoch loss and mAP is saved alongside each checkpoint.

### Model overview

| Model | Config | imgsz | Batch | Epochs | Notes |
|---|---|---|---|---|---|
| YOLOv11 Baseline | `yolov11_baseline.yaml` | 512 | 16 | 50 | Reference |
| YOLOv11 Improved | `yolov11_improved.yaml` | 512 | 16 | 50 | +FPN4, +WIoU, +domain aug |
| RT-DETR Baseline | `rtdetr_baseline.yaml` | 640 | 8 | 100 | ~2.5× better mAP than YOLO |
| RT-DETR Improved | `rtdetr_improved.yaml` | 640 | 8 | 100 | +domain aug, +mosaic, +copy-paste |

> **Recommended order:** run RT-DETR models first — they outperform YOLO significantly
> and RT-DETR Improved is the strongest model in this project.

### Commands

```bash
# YOLOv11 Baseline
!python train.py --config yolov11_baseline.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR --imgsz 512 --batch 16

# YOLOv11 Improved
!python train.py --config yolov11_improved.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR --imgsz 512 --batch 16

# RT-DETR Baseline
!python train.py --config rtdetr_baseline.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR

# RT-DETR Improved  ← strongest model
!python train.py --config rtdetr_improved.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR
```

**After a disconnect:** re-run the exact same command. The script prints
`✓ Checkpoint found — auto-resuming` and continues from the last saved epoch.

To force-resume from a specific checkpoint:
```bash
!python train.py --config rtdetr_improved.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR \
    --resume "$CKPT_DIR/rtdetrv2_improved/weights/last.pt"
```

---

## 7. Run Ablation Study

Run this **after all four models are trained**. It loads each model from its saved
checkpoint, evaluates on the val set, reads the training history, and writes five
plots + a CSV to `runs/ablation/`.

### Full run (eval + all plots + detection grid)

```bash
!python ablation.py --config ablation.yaml \
    --checkpoint-dir $CKPT_DIR \
    --data-dir $DATA_DIR \
    --device cuda
```

### Outputs

| File | What it shows |
|---|---|
| `metrics_comparison.png` | Bar chart: mAP@50, mAP@50-95, Precision, Recall per model |
| `loss_curves.png` | Train + val box / cls / DFL loss over epochs for each model |
| `metric_curves.png` | mAP@50, Precision, Recall curves over epochs |
| `per_class_mAP.png` | Heatmap of per-class mAP@50 (D00–D44) across all models |
| `detection_grid.png` | Sample val images with each model's detections side-by-side |
| `ablation_results.csv` | Final metrics table (mAP, Precision, Recall, F1, Δ vs baseline) |

### Display plots inline

```python
from IPython.display import Image, display
import os

OUTPUT = "/content/road_damage_detection/runs/ablation"

for fname in ["metrics_comparison.png", "loss_curves.png",
              "metric_curves.png", "per_class_mAP.png", "detection_grid.png"]:
    path = os.path.join(OUTPUT, fname)
    if os.path.exists(path):
        print(f"\n── {fname} ──")
        display(Image(path))
```

### Useful flags

```bash
# Plot training curves only — no GPU, no evaluation needed
!python ablation.py --config ablation.yaml --checkpoint-dir $CKPT_DIR --curves-only

# Skip detection grid (faster when GPU memory is tight)
!python ablation.py --config ablation.yaml \
    --checkpoint-dir $CKPT_DIR --data-dir $DATA_DIR --no-detection

# Evaluate a single model only
!python ablation.py --config ablation.yaml \
    --checkpoint-dir $CKPT_DIR --data-dir $DATA_DIR \
    --model yolov11_baseline

# Control how many sample images appear in the detection grid (default 6)
!python ablation.py --config ablation.yaml \
    --checkpoint-dir $CKPT_DIR --data-dir $DATA_DIR \
    --num-samples 8
```

---

## 8. Monitor Training with TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir $CKPT_DIR
```

---

## Troubleshooting

| Situation | Fix |
|---|---|
| CUDA out of memory (YOLO) | Add `--batch 8` |
| CUDA out of memory (RT-DETR) | Add `--batch 4` |
| Session disconnects | Re-run the same command — auto-resumes from Drive checkpoint |
| First epoch very slow | Normal — Ultralytics builds the disk cache; all later epochs are faster |
| Corrupt label warnings (`class 4`) | Delete `$DATA_DIR/train/labels.cache` and `$DATA_DIR/val/labels.cache` then re-run |
| Training stops before `epochs` | Expected — early stopping (YOLO: patience 30, RT-DETR: patience 40) |
| Quick smoke test | Add `--epochs 5` to verify the full pipeline runs |
| Check what epoch you're on | Open `$CKPT_DIR/<model_name>/training_history.json` |
