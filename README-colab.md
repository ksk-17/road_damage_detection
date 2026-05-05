# Running Road Damage Detection on Google Colab

## 1. Enable GPU Runtime

Before running anything:
**Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**

Verify GPU is active:
```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # Tesla T4
```

---

## 2. Upload Project Files

Upload the project files to your Colab session or mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then either clone from GitHub or copy files from Drive. The files you need:
```
train.py
yolov11_model.py
rtdetr_model.py
losses.py
yolov11_base-line.yaml
yolov11_improved.yaml
rtdetr_baseline.yaml
ablation.yaml
ablation.py
```

---

## 3. Get the Dataset — Download from Kaggle (Recommended)

Downloading directly inside Colab is much faster than uploading (~3GB). No manual upload needed.

### Step 1 — Add your Kaggle API key

Go to [kaggle.com](https://www.kaggle.com) → Account → API → Create New Token.
This downloads `kaggle.json`. Upload it to Colab:

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

### Step 2 — Download the dataset

```python
import kagglehub
path = kagglehub.dataset_download("aliabdelmenam/rdd-2022")
print("Downloaded to:", path)
```

### Step 3 — Find the RDD_SPLIT directory

```python
import os
for root, dirs, files in os.walk(path):
    if 'train' in dirs and 'val' in dirs:
        data_dir = root
        print("Data dir:", data_dir)
        break
```

Set it as a variable (example — your path will differ):
```python
DATA_DIR = data_dir   # e.g. /root/.cache/kagglehub/datasets/aliabdelmenam/rdd-2022/versions/1/RDD_SPLIT
```

Verify the structure:
```python
os.listdir(DATA_DIR)   # should show: ['train', 'val', 'test']
```

---

## 4. Install Dependencies

```python
!pip install ultralytics albumentations rich PyYAML opencv-python-headless
```

---

## 5. Mount Google Drive for Checkpoints

Checkpoints survive session disconnects only if saved to Drive. Mount it first:

```python
from google.colab import drive
drive.mount('/content/drive')

CKPT_DIR = "/content/drive/MyDrive/road_damage_ckpts"
```

---

## 6. Run Training

Pass `--data-dir` and `--checkpoint-dir` on every run. The script auto-detects an existing
checkpoint and resumes — no manual intervention needed after a disconnect.

```bash
# YOLOv11 baseline
!python train.py --config yolov11_base-line.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR --imgsz 512

# YOLOv11 with all improvements
!python train.py --config yolov11_improved.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR --imgsz 512

# RT-DETRv2 baseline
!python train.py --config rtdetr_baseline.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR --imgsz 512
```

**After a disconnect:** just re-run the exact same command. The script will find
`$CKPT_DIR/<model_name>/weights/last.pt` and continue from that epoch automatically.

If you hit OOM on T4:
```bash
!python train.py --config yolov11_base-line.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR --imgsz 512 --batch 8
```

To force-resume from a specific checkpoint manually:
```bash
!python train.py --config yolov11_base-line.yaml \
    --data-dir $DATA_DIR --checkpoint-dir $CKPT_DIR \
    --resume "$CKPT_DIR/yolov11_baseline/weights/last.pt"
```

After each run a `training_history.json` is saved inside `$CKPT_DIR/<model_name>/`
with per-epoch loss and mAP values, accumulated across all sessions.

---

## 7. Run Ablation Study

```bash
!python ablation.py --config ablation.yaml --data-dir $DATA_DIR --device cuda
```

Run a single experiment:
```bash
!python ablation.py --config ablation.yaml --data-dir $DATA_DIR --experiment fpn4_only
```

---

## 8. Monitor Training with TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir runs/
```

---

## 9. Save Results to Google Drive

Colab sessions reset after ~12 hours. Save checkpoints to Drive so you don't lose them:

```python
import shutil
shutil.copytree('runs/', '/content/drive/MyDrive/road_damage_runs/', dirs_exist_ok=True)
```

Or stream directly during training by setting the output dir in the YAML:
```bash
!python train.py --config yolov11_base-line.yaml --data-dir $DATA_DIR
# Then manually copy runs/ to Drive when done
```

---

## Tips

| Situation | Fix |
|---|---|
| CUDA out of memory | Add `--batch 16` (YOLO) or `--batch 4` (RT-DETR) |
| Session disconnects | Use `--resume` with the last checkpoint path |
| First epoch is slow | Normal — Ultralytics is building the disk cache; all later epochs are much faster |
| Corrupt label warnings | Delete `$DATA_DIR/train/labels.cache` and `$DATA_DIR/val/labels.cache` |
| Training stops early | Expected — early stopping (patience=30) kicks in when mAP plateaus |
| Want faster results | Add `--epochs 30` for a quick benchmark run |
