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

**RT-DETRv2** (Baidu, 2024) — Architectural comparison
- Real-Time Detection Transformer with deformable multi-scale attention
- Hungarian bipartite matching (no NMS required)
- Genuinely distinct paradigm from CNN-based YOLO, enabling a fair architectural comparison

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
├── configs/
│   ├── yolov11_baseline.yaml       # YOLOv11 baseline (CIoU, 3-scale FPN)
│   ├── yolov11_improved.yaml       # YOLOv11 + all 3 improvements
│   ├── rtdetr_baseline.yaml        # RT-DETRv2 baseline
│   └── ablation.yaml               # 6-experiment ablation study config
│
├── data/
│   └── rdd2022_dataset.py          # Dataset loader, VOC XML parser, augmentation pipelines
│
├── models/
│   ├── yolov11_model.py            # YOLOv11 wrapper + FourScaleFPNHead implementation
│   └── rtdetr_model.py             # RT-DETRv2 wrapper + HungarianMatcher
│
├── training/
│   └── losses.py                   # FocalLoss, WIoULoss, CIoULoss, RoadDamageDetectionLoss
│
├── evaluation/
│   └── metrics.py                  # mAP@50, mAP@50-95, per-class AP, visualization
│
├── utils/
│   └── visualization.py            # Bounding box drawing, training curve plots
│
├── scripts/
│   ├── prepare_dataset.py          # Download guide, VOC→YOLO conversion, dataset.yaml gen
│   ├── train.py                    # Main training entry point (supports all models/configs)
│   └── ablation.py                 # 6-experiment ablation runner + comparison charts/tables
│
└── app/
    └── demo.py                     # Streamlit web demo: upload → detect → visualize
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
python scripts/train.py --config configs/yolov11_baseline.yaml

# RT-DETRv2 baseline
python scripts/train.py --config configs/rtdetr_baseline.yaml

# YOLOv11 with all 3 improvements
python scripts/train.py --config configs/yolov11_improved.yaml
```

### 4. Run Ablation Study

```bash
python scripts/ablation.py --config configs/ablation.yaml --plot
```

### 5. Launch Web Demo

```bash
streamlit run app/demo.py
```

---

## 🔬 Ablation Design

| Experiment | FPN Scales | Box Loss | Domain Adapt |
|------------|-----------|----------|--------------|
| Baseline | 3 | CIoU | ❌ |
| + 4th FPN Scale | **4** | CIoU | ❌ |
| + Focal+WIoU | 3 | **Focal+WIoU** | ❌ |
| + Domain Adapt | 3 | CIoU | **✅** |
| + FPN4 + Focal+WIoU | **4** | **Focal+WIoU** | ❌ |
| **All Improvements** | **4** | **Focal+WIoU** | **✅** |

---

## ✅ Progress & Next Steps

### Completed ✔️
- [x] Project proposal finalized
- [x] Repository initialized with full modular project structure
- [x] `data/rdd2022_dataset.py` — RDD2022 loader, PASCAL VOC XML parser, domain randomization augmentation
- [x] `models/yolov11_model.py` — YOLOv11 fine-tuning wrapper with `FourScaleFPNHead` (Improvement 1)
- [x] `models/rtdetr_model.py` — RT-DETRv2 wrapper with `HungarianMatcher` implementation
- [x] `training/losses.py` — `WIoULoss`, `FocalLoss`, `CIoULoss`, and `RoadDamageDetectionLoss` from scratch (Improvement 2)
- [x] `scripts/train.py` — Unified training entry point for both architectures
- [x] `scripts/ablation.py` — Full 6-experiment ablation runner with charts and CSV output
- [x] `scripts/prepare_dataset.py` — Dataset verification, format conversion, YAML generation
- [x] `evaluation/metrics.py` — mAP@50/50-95, per-class AP, precision-recall, comparison plots
- [x] `app/demo.py` — Streamlit demo with live detection, model comparison, and ablation tabs
- [x] All 4 YAML configs (YOLOv11 baseline/improved, RT-DETRv2, ablation)
- [x] `requirements.txt` with all dependencies

### In Progress 🔄
- [ ] Downloading and verifying RDD2022 dataset locally
- [ ] Running initial YOLOv11 baseline training on Japan subset
- [ ] End-to-end pipeline validation (data → train → eval)

### Next Steps 📋
- [ ] Complete YOLOv11 baseline training — log mAP@50 result
- [ ] Complete RT-DETRv2 baseline training
- [ ] Run all 6 ablation experiments and generate comparison table
- [ ] Evaluate domain generalization on India and USA splits
- [ ] Integrate real model weights into Streamlit demo
- [ ] Record final demo walkthrough video
- [ ] Write final project report with full results

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
