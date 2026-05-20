# Running the Demo Server Locally

This guide covers running the Road Damage Detection web demo on your own machine
(CPU or GPU) without Google Colab.

---

## Prerequisites

| Tool   | Version     | Notes                                                    |
| ------ | ----------- | -------------------------------------------------------- |
| Python | 3.10 – 3.12 | [python.org](https://www.python.org/downloads/)          |
| pip    | any recent  | comes with Python                                        |
| FFmpeg | any         | required for video tab; skip if only using image/compare |

**Install FFmpeg (Windows):**

```powershell
winget install ffmpeg
# or download from https://ffmpeg.org/download.html and add to PATH
```

**Install FFmpeg (macOS):**

```bash
brew install ffmpeg
```

**Install FFmpeg (Linux/Ubuntu):**

```bash
sudo apt install ffmpeg
```

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/ksk-17/road_damage_detection.git
cd road_damage_detection
```

---

## Step 2 — Install Python Dependencies

```bash
pip install fastapi "uvicorn[standard]" python-multipart \
            ultralytics opencv-python pillow numpy rich
```

> **GPU users:** install the CUDA build of PyTorch first so inference runs on
> your GPU:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
>
> Then install the rest of the dependencies above.

---

## Step 3 — Download Checkpoints from Google Drive

The trained model weights are stored in the shared Google Drive folder:
`CMPE_258/training_checkpoints/`

Download all 4 model folders. You only need the `weights/best.pt` file inside
each:

```
road_checkpoints/
├── yolov11_baseline/
│   └── weights/
│       └── best.pt
├── yolov11_improved/
│   └── weights/
│       └── best.pt
├── rtdetrv2_baseline/
│   └── weights/
│       └── best.pt
└── rtdetrv2_improved/
    └── weights/
        └── best.pt
```

Put them anywhere convenient, e.g. `D:\road_checkpoints\` on Windows or
`~/road_checkpoints/` on macOS/Linux.

> **Optional — ONNX (faster CPU inference):** If you have ONNX files
> (`best.onnx`) alongside the `.pt` files, the server will auto-detect and
> prefer them. Export with:
>
> ```bash
> pip install onnxruntime   # CPU  (or onnxruntime-gpu for CUDA)
> python export_onnx.py --checkpoint-dir D:\road_checkpoints
> ```

---

## Step 4 — Start the Server

```bash
# Windows
python server.py --checkpoint-dir ./road_checkpoints --port 8000

# macOS / Linux
python server.py --checkpoint-dir ~/road_checkpoints --port 8000
```

You should see output like:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Model                  ┃ Format   ┃ Size   ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ yolov11_baseline       │ PYTORCH  │ 39.7MB │ ✓      │
│ yolov11_improved       │ PYTORCH  │ 39.7MB │ ✓      │
│ rtdetrv2_baseline      │ PYTORCH  │ 121MB  │ ✓      │
│ rtdetrv2_improved      │ PYTORCH  │ 121MB  │ ✓      │
└────────────────────────┴──────────┴────────┴────────┘

Server ready → http://localhost:8000  (4 model(s) loaded)
```

Open **http://localhost:8000** in your browser.

---

## Step 5 — Use the Web UI

The interface has four tabs:

| Tab                 | What it does                                                          |
| ------------------- | --------------------------------------------------------------------- |
| **📸 Image**        | Upload a road photo → get annotated detections + latency breakdown    |
| **🎬 Video**        | Upload an MP4/AVI → process frame-by-frame, download annotated output |
| **⚖️ Compare**      | Run the same image through all 4 models side-by-side                  |
| **📊 Eval Metrics** | Pre-computed val-set mAP, precision, recall, F1, per-class breakdown  |

Use the **sidebar** to switch models, adjust confidence threshold, and IoU
threshold.

---

## Performance on CPU

| Model              | CPU inference     | Notes                    |
| ------------------ | ----------------- | ------------------------ |
| YOLOv11 Baseline   | ~100–250 ms/frame | Fast for demo            |
| YOLOv11 Improved   | ~120–280 ms/frame | Similar speed            |
| RT-DETRv2 Baseline | ~400–700 ms/frame | Slower but more accurate |
| RT-DETRv2 Improved | ~400–700 ms/frame | Best accuracy            |

> With ONNX + `onnxruntime`, CPU inference is roughly **2× faster** than
> PyTorch.

---

## Troubleshooting

**`Model not loaded` error** The server couldn't find `best.pt` or `best.onnx`
at the expected path. Double-check your folder structure matches exactly:

```
<checkpoint-dir>/<model-name>/weights/best.pt
```

Model names must be: `yolov11_baseline`, `yolov11_improved`,
`rtdetrv2_baseline`, `rtdetrv2_improved`.

---

**Video plays blank / no video after processing** FFmpeg is not installed or not
on PATH. Install it (see Prerequisites) and restart the server. Without FFmpeg,
the video is encoded with mp4v which most browsers cannot play inline.

Verify FFmpeg is reachable:

```bash
ffmpeg -version
```

---

**`ModuleNotFoundError: No module named 'ultralytics'`** Run
`pip install ultralytics` and restart the server.

---

**Port 8000 already in use** Use a different port:

```bash
python server.py --checkpoint-dir D:\road_checkpoints --port 8080
```

Then open http://localhost:8080.

---

**GPU not being used** Ultralytics auto-selects CUDA if available. Verify with:

```python
import torch; print(torch.cuda.is_available())
```

If `False`, reinstall PyTorch with the correct CUDA version from
[pytorch.org](https://pytorch.org).
