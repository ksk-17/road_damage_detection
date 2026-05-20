"""
server.py
FastAPI Inference Server — Road Damage Detection

Serves all 4 trained models with real latency metrics.
Prefers ONNX (faster) over .pt automatically.

Endpoints:
  GET  /api/models                  — list loaded models + metadata
  POST /api/predict/image           — single image inference
  POST /api/predict/video           — video inference (frame-by-frame)
  POST /api/predict/compare         — same image through multiple models
  GET  /api/health
  GET  /                            — frontend (static/index.html)

Usage:
    python server.py --checkpoint-dir /content/drive/MyDrive/road_ckpts
    python server.py --checkpoint-dir ./runs --port 8000

On Colab (public URL):
    !pip install pyngrok
    from pyngrok import ngrok
    !python server.py --checkpoint-dir $CKPT_DIR &
    print(ngrok.connect(8000))

CMP 295 SJSU | Road Damage Detection
"""

import argparse
import base64
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["D00", "D10", "D20", "D40", "D44"]

CLASS_COLORS_BGR = {
    "D00": (60,  80,  239),   # red
    "D10": (22,  147, 249),   # orange
    "D20": (22,  196, 232),   # yellow
    "D40": (34,  197, 94),    # green
    "D44": (168, 85,  247),   # purple
}

MODEL_META: Dict[str, dict] = {
    "yolov11_baseline":  {
        "label":        "YOLOv11 Baseline",
        "description":  "YOLOv11m · CIoU · 3-scale FPN · no domain adapt",
        "type":         "yolo",
        "default_conf": 0.25,
        "default_iou":  0.45,
    },
    "yolov11_improved":  {
        "label":        "YOLOv11 Improved",
        "description":  "YOLOv11m · Focal+WIoU · 4-scale FPN · domain adapt",
        "type":         "yolo",
        "default_conf": 0.25,
        "default_iou":  0.45,
    },
    "rtdetrv2_baseline": {
        "label":        "RT-DETRv2 Baseline",
        "description":  "rtdetr-l · Hungarian matching · no NMS",
        "type":         "rtdetr",
        "default_conf": 0.30,
        "default_iou":  0.50,
    },
    "rtdetrv2_improved": {
        "label":        "RT-DETRv2 Improved",
        "description":  "rtdetr-l · domain adapt · mosaic · copy-paste",
        "type":         "rtdetr",
        "default_conf": 0.25,
        "default_iou":  0.50,
    },
}

# Known eval results — shown in the Compare panel
EVAL_RESULTS = {
    "yolov11_baseline":  {"mAP50": 0.354, "mAP50_95": 0.161, "precision": 0.490, "recall": 0.356, "f1": 0.412},
    "yolov11_improved":  {"mAP50": 0.366, "mAP50_95": 0.175, "precision": 0.490, "recall": 0.373, "f1": 0.423},
    "rtdetrv2_baseline": {"mAP50": 0.523, "mAP50_95": 0.238, "precision": 0.598, "recall": 0.507, "f1": 0.549},
    "rtdetrv2_improved": {"mAP50": 0.552, "mAP50_95": 0.242, "precision": 0.612, "recall": 0.531, "f1": 0.569},
}

PER_CLASS_RESULTS = {
    "yolov11_baseline":  {"D00": 0.364, "D10": 0.350, "D20": 0.448, "D40": 0.417, "D44": 0.190},
    "yolov11_improved":  {"D00": 0.375, "D10": 0.367, "D20": 0.447, "D40": 0.441, "D44": 0.199},
    "rtdetrv2_baseline": {"D00": 0.485, "D10": 0.435, "D20": 0.587, "D40": 0.693, "D44": 0.416},
    "rtdetrv2_improved": {"D00": 0.499, "D10": 0.457, "D20": 0.609, "D40": 0.720, "D44": 0.473},
}

# ─── Global model registry ────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, dict] = {}
CHECKPOINT_DIR = "./runs"
OUTPUT_DIR = Path("static/outputs")

# Resolve ffmpeg binary: prefer imageio-ffmpeg (bundled, no system install needed),
# then fall back to system ffmpeg, then None.
def _resolve_ffmpeg() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return "ffmpeg"
    except Exception:
        pass
    return None

FFMPEG_BIN: Optional[str] = _resolve_ffmpeg()


# ─── Model loading ────────────────────────────────────────────────────────────

def _resolve_weights(checkpoint_dir: str, model_name: str) -> tuple[Optional[str], str]:
    """Return (path, format) — prefers ONNX over .pt."""
    base = Path(checkpoint_dir) / model_name / "weights"
    for fname, fmt in [("best.onnx", "onnx"), ("best.pt", "pytorch"), ("last.pt", "pytorch")]:
        p = base / fname
        if p.exists():
            return str(p), fmt
    return None, "missing"


def load_all_models(checkpoint_dir: str):
    from rich.console import Console
    from rich.table import Table
    console = Console()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table = Table(title="Loading Models", header_style="bold cyan")
    table.add_column("Model",   min_width=22)
    table.add_column("Format",  justify="center")
    table.add_column("Size",    justify="right")
    table.add_column("Status",  justify="center")

    for name, meta in MODEL_META.items():
        path, fmt = _resolve_weights(checkpoint_dir, name)
        if path is None:
            table.add_row(name, "—", "—", "[yellow]not found[/yellow]")
            continue

        try:
            if meta["type"] == "rtdetr":
                from ultralytics import RTDETR
                model = RTDETR(path)
            else:
                from ultralytics import YOLO
                model = YOLO(path)

            size_mb = round(Path(path).stat().st_size / 1024 / 1024, 1)
            MODEL_REGISTRY[name] = {
                **meta,
                "model":    model,
                "format":   fmt,
                "size_mb":  size_mb,
                "eval":     EVAL_RESULTS.get(name, {}),
                "per_class": PER_CLASS_RESULTS.get(name, {}),
            }
            table.add_row(name, fmt.upper(), f"{size_mb} MB", "[green]✓[/green]")

        except Exception as exc:
            table.add_row(name, fmt, "—", f"[red]ERR: {exc}[/red]")

    console.print(table)


# ─── Inference helpers ────────────────────────────────────────────────────────

def _infer(entry: dict, bgr: np.ndarray, conf: float, iou: float) -> tuple[list, dict, np.ndarray]:
    """
    Run inference on a single BGR frame.
    Returns (detections, timing_dict, annotated_bgr).
    """
    model = entry["model"]

    t0 = time.perf_counter()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t1 = time.perf_counter()

    results = model.predict(source=pil, conf=conf, iou=iou, verbose=False)
    t2 = time.perf_counter()

    detections = []
    annotated  = bgr.copy()

    for res in results:
        if res.boxes is None:
            continue
        boxes   = res.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs   = res.boxes.conf.cpu().numpy()

        for box, cls_id, score in zip(boxes, cls_ids, confs):
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            color = CLASS_COLORS_BGR.get(name, (200, 200, 200))
            x1, y1, x2, y2 = box

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 7), (x1 + tw + 5, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            detections.append({
                "class":      name,
                "confidence": round(float(score), 3),
                "box":        [int(x1), int(y1), int(x2), int(y2)],
            })

    t3 = time.perf_counter()

    pre_ms  = round((t1 - t0) * 1000, 2)
    inf_ms  = round((t2 - t1) * 1000, 2)
    post_ms = round((t3 - t2) * 1000, 2)
    total   = pre_ms + inf_ms + post_ms

    timing = {
        "preprocess_ms":  pre_ms,
        "inference_ms":   inf_ms,
        "postprocess_ms": post_ms,
        "total_ms":       round(total, 2),
        "fps":            round(1000 / max(total, 0.1), 1),
    }

    return detections, timing, annotated


def _class_counts(detections: list) -> dict:
    counts: dict = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    return counts


def _to_b64(bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Road Damage Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API routes ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": list(MODEL_REGISTRY.keys())}


@app.get("/api/models")
def list_models():
    return {
        name: {
            "label":        e["label"],
            "description":  e["description"],
            "format":       e["format"],
            "size_mb":      e["size_mb"],
            "type":         e["type"],
            "default_conf": e["default_conf"],
            "default_iou":  e["default_iou"],
            "eval":         e["eval"],
            "per_class":    e["per_class"],
        }
        for name, e in MODEL_REGISTRY.items()
    }


@app.post("/api/predict/image")
async def predict_image(
    file:       UploadFile = File(...),
    model_name: str   = Form("rtdetrv2_improved"),
    conf:       float = Form(0.25),
    iou:        float = Form(0.45),
):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model '{model_name}' not loaded. Available: {list(MODEL_REGISTRY)}")

    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Cannot decode image — ensure it is a valid JPG/PNG")

    entry = MODEL_REGISTRY[model_name]
    dets, timing, annotated = _infer(entry, bgr, conf, iou)

    return JSONResponse({
        "model":            entry["label"],
        "format":           entry["format"],
        "size_mb":          entry["size_mb"],
        "image_b64":        _to_b64(annotated),
        "detections":       dets,
        "class_counts":     _class_counts(dets),
        "total_detections": len(dets),
        "timing":           timing,
        "input_shape":      list(bgr.shape[:2]),
        "eval_metrics":     entry["eval"],
        "per_class_map":    entry["per_class"],
    })


@app.post("/api/predict/video")
async def predict_video(
    file:       UploadFile = File(...),
    model_name: str   = Form("rtdetrv2_improved"),
    conf:       float = Form(0.25),
    iou:        float = Form(0.45),
    max_frames: int   = Form(300),
):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model '{model_name}' not loaded.")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_in = tmp.name

    uid      = uuid.uuid4().hex
    raw_name = f"{uid}_raw.mp4"
    out_name = f"{uid}.mp4"
    raw_path = str(OUTPUT_DIR / raw_name)
    out_path = str(OUTPUT_DIR / out_name)

    try:
        cap    = cv2.VideoCapture(tmp_in)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Always write raw frames with mp4v first (OpenCV can always write this)
        writer = cv2.VideoWriter(
            raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
        )

        entry        = MODEL_REGISTRY[model_name]
        latencies    = []
        all_dets     = []
        frame_data   = []
        processed    = 0

        while processed < min(max_frames, total if total > 0 else max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            dets, timing, annotated = _infer(entry, frame, conf, iou)
            writer.write(annotated)
            latencies.append(timing["total_ms"])
            all_dets.extend(dets)
            frame_data.append({
                "frame":      processed,
                "latency_ms": timing["total_ms"],
                "detections": len(dets),
            })
            processed += 1

        cap.release()
        writer.release()

        # Re-encode to H.264 so the browser can play it.
        # FFMPEG_BIN is either the imageio-bundled binary or system ffmpeg.
        if FFMPEG_BIN:
            subprocess.run(
                [
                    FFMPEG_BIN, "-y", "-i", raw_path,
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "fast", "-crf", "23",
                    out_path,
                ],
                check=True, capture_output=True, timeout=300,
            )
            os.unlink(raw_path)
        else:
            # No ffmpeg at all — rename raw file and warn (video likely won't play in browser)
            os.rename(raw_path, out_path)

        avg_ms  = round(sum(latencies) / max(len(latencies), 1), 2)
        avg_fps = round(1000 / max(avg_ms, 0.1), 1)

        return JSONResponse({
            "model":             entry["label"],
            "format":            entry["format"],
            "video_url":         f"/outputs/{out_name}",
            "frames_processed":  processed,
            "total_frames":      total,
            "avg_latency_ms":    avg_ms,
            "avg_fps":           avg_fps,
            "total_detections":  len(all_dets),
            "class_counts":      _class_counts(all_dets),
            "per_frame":         frame_data,
            "ffmpeg_used":       FFMPEG_BIN is not None,
        })

    finally:
        os.unlink(tmp_in)


@app.post("/api/predict/compare")
async def compare_models(
    file:        UploadFile = File(...),
    model_names: str   = Form(...),   # comma-separated
    conf:        float = Form(0.25),
    iou:         float = Form(0.45),
):
    names   = [n.strip() for n in model_names.split(",") if n.strip()]
    missing = [n for n in names if n not in MODEL_REGISTRY]
    if missing:
        raise HTTPException(404, f"Models not loaded: {missing}")

    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Cannot decode image")

    out = {}
    for name in names:
        entry = MODEL_REGISTRY[name]
        dets, timing, annotated = _infer(entry, bgr, conf, iou)
        out[name] = {
            "label":            entry["label"],
            "format":           entry["format"],
            "size_mb":          entry["size_mb"],
            "image_b64":        _to_b64(annotated),
            "detections":       dets,
            "class_counts":     _class_counts(dets),
            "total_detections": len(dets),
            "timing":           timing,
            "eval_metrics":     entry["eval"],
        }

    return JSONResponse(out)


# ─── Static file serving ──────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    html = Path("static/index.html")
    if html.exists():
        return FileResponse(str(html))
    return JSONResponse({"message": "Frontend not found — place static/index.html"})


@app.get("/outputs/{filename}")
def serve_output(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Output file not found")
    return FileResponse(str(path), media_type="video/mp4")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Road Damage Detection Inference Server")
    parser.add_argument("--checkpoint-dir", default="./runs",
                        help="Root dir containing per-model checkpoint folders")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global CHECKPOINT_DIR
    CHECKPOINT_DIR = args.checkpoint_dir

    load_all_models(args.checkpoint_dir)

    from rich.console import Console
    con = Console()
    con.print(
        f"\n[bold green]Server ready[/bold green] → "
        f"http://localhost:{args.port}  "
        f"([dim]{len(MODEL_REGISTRY)} model(s) loaded[/dim])"
    )
    if FFMPEG_BIN:
        src = "imageio-ffmpeg (bundled)" if FFMPEG_BIN != "ffmpeg" else "system ffmpeg"
        con.print(f"[dim]FFmpeg ready ({src}) — video encoded as H.264 (browser-compatible)[/dim]\n")
    else:
        con.print(
            "[yellow]FFmpeg not found — run `pip install imageio-ffmpeg` to fix video playback[/yellow]\n"
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
