# 🛣️ Road Damage Detection from Aerial Imagery Using Deep Learning

> **CMP 295 — Modern Deep Learning Pipeline | San José State University**

---

## 👥 Team Members

| Name | SJSU ID | Email |
|------|---------|-------|
| Sumanth Kumar Kotagudem | 018284214 | sumanthkumar.kotagudem@sjsu.edu |
| Akanksha Thalla | 018298189 | akanksha.thalla@sjsu.edu |
| Arathi Boddam | 018297903 | arathi.boddam@sjsu.edu |

---

## 📌 Project Overview

This project builds an **end-to-end object detection pipeline** for automatic road damage classification from drone and aerial imagery. We fine-tune and rigorously compare two SOTA architectures — **YOLOv11** (CNN-based) and **RT-DETRv2** (transformer-based) — on the **RDD2022** benchmark dataset, and introduce three targeted improvements to push beyond both baselines.

**Problem:** Road infrastructure deteriorates continuously, but manual inspection is slow, expensive, and inconsistent. Automated detection from drone imagery enables scalable, low-cost monitoring.

**Goal:** Train models that accurately classify 4 damage types (longitudinal cracks, transverse cracks, alligator cracks, potholes) across diverse countries and road conditions.

**Github Link**: https://github.com/ksk-17/road_damage_detection

---

## 📦 Dataset

**Primary: RDD2022 (Road Damage Dataset 2022)**
- **47,000** road images across **6 countries**: Japan, India, USA, Czech Republic, Norway, China
- **4 damage categories:**

| Class | Type | Description |
|-------|------|-------------|
| D00 | Longitudinal Crack | Cracks running parallel to road direction |
| D10 | Transverse Crack | Cracks running perpendicular to road direction |
| D20 | Alligator Crack | Interconnected crack networks (fatigue damage) |
| D40 | Pothole | Bowl-shaped surface depressions |

- Annotations in **PASCAL VOC XML** format (bounding boxes)
- **Class imbalance:** Normal road >> crack pixels — addressed via Focal Loss
- **Cross-country variation** in road texture, lighting, and camera angle enables domain adaptation experiments
- **Supplementary:** RDD2020 (additional volume), DeepCrack (segmentation experiments)

**Citation:** Arya, D. et al. (2022). RDD2022: A multi-national image dataset for automatic Road Damage Detection. IEEE DataPort. https://doi.org/10.21227/ke5f-n977

---

## 🧠 Approach

### Models

**YOLOv11** (Ultralytics, 2024) — Primary baseline
- Latest YOLO release with improved backbone and enhanced feature extraction
- Strong mAP/speed tradeoff; ideal for real-time drone deployment
- Variant: `yolo11m` — 20M parameters, 67.7 GFLOPs

**RT-DETRv2** (Baidu, 2024) — Architectural comparison
- Real-Time Detection Transformer with deformable multi-scale attention
- Hungarian bipartite matching (no NMS required)
- Genuinely distinct paradigm from CNN-based YOLO, enabling a fair architectural comparison
- Variant: `rtdetr-l` — 32M parameters, 103.5 GFLOPs

### Three Improvements

**Improvement 1 — Extra 4th FPN Detection Scale**
Standard YOLO uses 3 FPN scales (P3=80×80, P4=40×40, P5=20×20). We add a **P2 head (160×160)** to capture hairline longitudinal (D00) and thin transverse (D10) cracks that are missed at coarser resolutions.
- *Ablation:* mAP with 3 scales vs. 4 scales

**Improvement 2 — Focal Loss + WIoU Box Regression**
Replace CIoU with **Focal Loss** (classification) + **Wise-IoU** (box regression). Focal Loss down-weights easy negatives to address heavy class imbalance. WIoU applies a dynamic momentum-based focusing mechanism that amplifies gradients on hard, poorly-regressed boxes — especially beneficial for thin, elongated crack shapes.
- *Ablation:* CIoU baseline → Focal only → Focal+WIoU
- *Reference:* Tong et al. arXiv:2301.10051

**Improvement 3 — Augmentation-Based Domain Adaptation**
Train on Japan (largest/cleanest split), evaluate zero-shot on India and USA, then apply **domain randomization** (brightness, contrast, saturation, hue, blur, noise, JPEG quality) to simulate cross-country variation and improve out-of-distribution generalization.
- *Ablation:* Japan-trained model on India/USA, with vs. without domain randomization

---

## 📁 Repository Structure

```
road_damage_detection/
├── README.md
├── requirements.txt
│
├── yolov11_baseline.yaml           # YOLOv11 baseline (CIoU, 3-scale FPN)
├── yolov11_improved.yaml           # YOLOv11 + FPN4 + Focal+WIoU + domain adapt
├── rtdetr_baseline.yaml            # RT-DETRv2 baseline (Hungarian matching)
├── rtdetr_improved.yaml            # RT-DETRv2 + domain adapt + mosaic + copy-paste
├── ablation.yaml                   # Model registry for ablation study (no training)
│
├── yolov11_model.py                # YOLOv11 wrapper + FourScaleFPNHead implementation
├── rtdetr_model.py                 # RT-DETRv2 wrapper + HungarianMatcher
├── losses.py                       # FocalLoss, WIoULoss, CIoULoss, RoadDamageDetectionLoss
│
├── train.py                        # Main training entry point (supports all models/configs)
├── ablation.py                     # Checkpoint-based evaluator + comparison charts/tables
│
└── road_detection.ipynb            # End-to-end Colab notebook with all results and plots
```

---

## ⚙️ Setup & Usage

### 1. Clone & Install

```bash
git clone https://github.com/<your-org>/road-damage-detection.git
cd road-damage-detection
pip install -r requirements.txt
```

### 2. Download & Prepare RDD2022

```bash
# Show download instructions
python scripts/prepare_dataset.py --download-info

# After downloading and extracting to ./data/rdd2022:
python scripts/prepare_dataset.py --root ./data/rdd2022 \
    --countries Japan India USA \
    --convert-yolo \
    --gen-yaml
```

Dataset: https://doi.org/10.21227/ke5f-n977

### 3. Train Models

```bash
# YOLOv11 baseline
python train.py --config yolov11_baseline.yaml --data-dir ./data/RDD_SPLIT --imgsz 512

# YOLOv11 with all 3 improvements
python train.py --config yolov11_improved.yaml --data-dir ./data/RDD_SPLIT --imgsz 512

# RT-DETRv2 baseline
python train.py --config rtdetr_baseline.yaml --data-dir ./data/RDD_SPLIT

# RT-DETRv2 with improvements  ← best model
python train.py --config rtdetr_improved.yaml --data-dir ./data/RDD_SPLIT
```

### 4. Run Ablation Study

Requires all four models to be trained first. Loads from checkpoints — no retraining.

```bash
python ablation.py --config ablation.yaml \
    --checkpoint-dir ./runs \
    --data-dir ./data/RDD_SPLIT
```

Outputs written to `runs/ablation/`: bar charts, loss curves, metric curves,
per-class mAP heatmap, detection grid, and `ablation_results.csv`.

### 5. Launch Web Demo

```bash
streamlit run app/demo.py
```

---

## ⚙️ Training Parameters

| | YOLOv11 Baseline | YOLOv11 Improved | RT-DETRv2 Baseline | RT-DETRv2 Improved |
|---|---|---|---|---|
| Variant | yolo11m | yolo11m | rtdetr-l | rtdetr-l |
| Parameters | 20M | 20M | 32M | 32M |
| GFLOPs | 67.7 | 67.7 | 103.5 | 103.5 |
| Image size | 512 | 512 | 640 | 640 |
| Epochs | 50 | 50 | 50 | 50 (stopped at 56) |
| Batch size | 32 | 32 | 32 | 32 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| LR | 1e-3 | 1e-3 | 1e-4 | 1e-4 |
| Box loss | CIoU | Focal+WIoU | L1+GIoU | L1+GIoU |
| Cls loss | BCE | Focal | Focal | Focal |
| FPN scales | 3 | **4** | — | — |
| Mosaic | 1.0 | 1.0 | 0.0 | **0.5** |
| Copy-paste | 0.1 | 0.15 | 0.0 | **0.3** |
| Domain randomization | ❌ | **✅** | ❌ | **✅** |
| Patience | 30 | 30 | 20 | **40** |

---

## 📊 Results

All models trained on **Japan** split (RDD2022), evaluated on the full val set.

### Overall Metrics

| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 | ΔmAP@50 |
|---|---|---|---|---|---|---|
| YOLOv11 Baseline | 0.354 | 0.161 | 0.490 | 0.356 | 0.412 | — |
| YOLOv11 Improved | 0.366 | 0.175 | 0.490 | 0.373 | 0.423 | +0.012 |
| RT-DETRv2 Baseline | 0.523 | 0.238 | 0.598 | 0.507 | 0.549 | +0.169 |
| **RT-DETRv2 Improved** | **0.552** | **0.242** | **0.612** | **0.531** | **0.569** | **+0.198** |

RT-DETRv2 Improved achieves **+19.8 mAP@50 points** over the YOLOv11 baseline — a 56% relative improvement.

### Per-Class mAP@50

| Class | YOLOv11 Base | YOLOv11 Impr | RT-DETRv2 Base | RT-DETRv2 Impr |
|---|---|---|---|---|
| D00 — Longitudinal Crack | 0.364 | 0.375 | 0.485 | 0.499 |
| D10 — Transverse Crack | 0.350 | 0.367 | 0.435 | 0.457 |
| D20 — Alligator Crack | 0.448 | 0.447 | 0.587 | **0.609** |
| D40 — Pothole | 0.417 | 0.441 | 0.693 | **0.720** |
| D44 — Other Damage | 0.190 | 0.199 | 0.416 | **0.473** |

RT-DETRv2's Hungarian matching loss is particularly effective for pothole (D40) detection, achieving **0.720 mAP@50** vs. 0.417 for the YOLO baseline — a near 2× gain on the most safety-critical damage class.

---

## ✅ Completed

- [x] `yolov11_model.py` — YOLOv11 fine-tuning wrapper with `FourScaleFPNHead` (Improvement 1)
- [x] `rtdetr_model.py` — RT-DETRv2 wrapper with `HungarianMatcher` implementation
- [x] `losses.py` — `WIoULoss`, `FocalLoss`, `CIoULoss`, and `RoadDamageDetectionLoss` (Improvement 2)
- [x] `train.py` — Unified training entry point for both architectures with auto-resume
- [x] `ablation.py` — Checkpoint-based evaluator with 5 comparison plots and CSV output
- [x] All 4 YAML configs: `yolov11_baseline`, `yolov11_improved`, `rtdetr_baseline`, `rtdetr_improved`
- [x] YOLOv11 baseline trained — mAP@50: **0.354**
- [x] YOLOv11 improved trained — mAP@50: **0.366** (+3.4%)
- [x] RT-DETRv2 baseline trained — mAP@50: **0.523** (+47.8% vs YOLO baseline)
- [x] RT-DETRv2 improved trained — mAP@50: **0.552** (+55.9% vs YOLO baseline)
- [x] Ablation study complete with full comparison plots and per-class breakdown
- [x] End-to-end Colab notebook (`road_detection.ipynb`) with all results

---

## 📚 References

- Arya, D. et al. (2022). RDD2022: A multi-national image dataset for automatic Road Damage Detection. IEEE DataPort. https://doi.org/10.21227/ke5f-n977
- Jocher, G. et al. (2024). Ultralytics YOLOv11. https://github.com/ultralytics/ultralytics
- Lv, W. et al. (2024). RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer. arXiv:2407.17140. https://github.com/lyuwenyu/RT-DETR
- Tong, Z. et al. (2023). Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism. arXiv:2301.10051
- CRDDC2022 Challenge Baselines: https://github.com/sekilab/RoadDamageDetector

---

## 📄 License

Developed for academic purposes as part of CMP 295 at San José State University.
