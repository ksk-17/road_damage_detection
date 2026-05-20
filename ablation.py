"""
ablation.py
Ablation Study — Load trained models from checkpoints, evaluate, and compare.
No training happens here. Run train.py for each model first.

Outputs (saved to evaluation.output_dir):
  metrics_comparison.png    — bar chart: mAP50 / mAP50-95 / precision / recall / F1
  loss_curves.png           — train + val loss per epoch for each model
  metric_curves.png         — mAP50, precision, recall curves per epoch
  per_class_mAP.png         — per-class mAP50 heatmap across models
  pr_curves.png             — precision-recall curves (object-level, mean across classes)
  f1_confidence_curves.png  — F1 vs confidence threshold with optimal operating point
  roc_curves.png            — image-level ROC (positive = image has ≥1 damage annotation)
  timing_comparison.png     — inference latency (ms/image) per model
  detection_grid.png        — N sample images run through all models side-by-side
  ablation_results.csv      — final metrics table

Usage:
    # Full run (evaluate + all plots + detection grid):
    python ablation.py --config ablation.yaml \\
                       --checkpoint-dir /content/drive/MyDrive/road_ckpts \\
                       --data-dir /content/RDD_SPLIT

    # Plot training curves only (no GPU needed):
    python ablation.py --config ablation.yaml \\
                       --checkpoint-dir /content/drive/MyDrive/road_ckpts \\
                       --curves-only

    # Single model:
    python ablation.py --config ablation.yaml \\
                       --checkpoint-dir /content/drive/MyDrive/road_ckpts \\
                       --data-dir /content/RDD_SPLIT \\
                       --model yolov11_baseline

    # Skip ROC (faster, skips per-image predict sweep):
    python ablation.py --config ablation.yaml \\
                       --checkpoint-dir /content/drive/MyDrive/road_ckpts \\
                       --data-dir /content/RDD_SPLIT \\
                       --no-roc

CMP 295 SJSU | Road Damage Detection
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from scripts.train import build_model, load_config
except ModuleNotFoundError:
    from train import build_model, load_config

console = Console()

CLASS_NAMES = ["D00", "D10", "D20", "D40", "D44"]

CLASS_COLORS = {
    "D00": "#ef4444",
    "D10": "#f97316",
    "D20": "#eab308",
    "D40": "#22c55e",
    "D44": "#3b82f6",
}

MODEL_COLORS = {
    "yolov11_baseline":  "#64748b",
    "yolov11_improved":  "#3b82f6",
    "rtdetrv2_baseline": "#f59e0b",
    "rtdetrv2_improved": "#10b981",
}


# ─── Checkpoint Loading ───────────────────────────────────────────────────────

def resolve_checkpoint(checkpoint_dir: str, model_name: str) -> str:
    """Return path to best.pt (or last.pt) for model_name inside checkpoint_dir."""
    base = Path(checkpoint_dir) / model_name / "weights"
    for fname in ("best.pt", "last.pt"):
        p = base / fname
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"No checkpoint found for '{model_name}' in {base}\n"
        "Make sure you trained this model with train.py before running ablation."
    )


def load_model_from_checkpoint(entry: dict, checkpoint_path: str, config_root: Path, device: str):
    """Build model wrapper and load its trained weights from checkpoint."""
    cfg_path = config_root / entry["config"]
    config = load_config(str(cfg_path))
    config["model"]["pretrained_weights"] = checkpoint_path
    return build_model(config, device), config


# ─── Training History ─────────────────────────────────────────────────────────

def load_training_history(checkpoint_dir: str, model_name: str) -> Optional[List[dict]]:
    """
    Load per-epoch metrics from training_history.json (written by train.py).
    Falls back to results.csv (written by Ultralytics) if JSON is missing.
    Returns a flat list of epoch dicts across all resumed runs, or None.
    """
    history_path = Path(checkpoint_dir) / model_name / "training_history.json"
    csv_path = Path(checkpoint_dir) / model_name / "results.csv"

    if history_path.exists():
        with open(history_path) as f:
            data = json.load(f)
        epochs = []
        for run in data.get("runs", []):
            epochs.extend(run.get("per_epoch", []))
        if epochs:
            return epochs

    if csv_path.exists():
        epochs = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    epochs.append({
                        "epoch":        int(float(s.get("epoch", 0))),
                        "box_loss":     float(s.get("train/box_loss", 0)),
                        "cls_loss":     float(s.get("train/cls_loss", 0)),
                        "dfl_loss":     float(s.get("train/dfl_loss", 0)),
                        "val_box_loss": float(s.get("val/box_loss", 0)),
                        "val_cls_loss": float(s.get("val/cls_loss", 0)),
                        "val_dfl_loss": float(s.get("val/dfl_loss", 0)),
                        "precision":    float(s.get("metrics/precision(B)", 0)),
                        "recall":       float(s.get("metrics/recall(B)", 0)),
                        "mAP50":        float(s.get("metrics/mAP50(B)", 0)),
                        "mAP50_95":     float(s.get("metrics/mAP50-95(B)", 0)),
                    })
                except (ValueError, KeyError):
                    continue
        if epochs:
            return epochs

    return None


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _curve_mean(arr: np.ndarray) -> np.ndarray:
    """Return mean over axis-0 if 2-D (per-class curves), else return as-is."""
    arr = np.asarray(arr, dtype=float)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def evaluate_model(model, model_name: str, data_dir: str, model_config: dict) -> dict:
    """
    Run validation on the val split.
    Calls the raw Ultralytics val() via model.model to extract both scalar metrics
    and PR/F1-confidence curve arrays. Falls back to model.evaluate() if that fails.
    """
    import tempfile
    nc = model_config.get("model", {}).get("num_classes", 5)
    ds_yaml = {
        "path": str(Path(data_dir).resolve()),
        "train": "train/images",
        "val":   "val/images",
        "nc":    nc,
        "names": CLASS_NAMES[:nc],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(ds_yaml, f)
        tmp = f.name

    metrics: dict = {}

    # Try calling the underlying Ultralytics model directly to get curve data
    try:
        raw = model.model.val(data=tmp, split="val", verbose=False)
        metrics = {
            "mAP50":           float(raw.box.map50),
            "mAP50_95":        float(raw.box.map),
            "precision":       float(raw.box.mp),
            "recall":          float(raw.box.mr),
            "per_class_mAP50": raw.box.ap50.tolist(),
        }
        # Extract PR and F1-confidence curves
        try:
            cr = raw.curves_results  # list of [x, y, xlabel, ylabel]
            if cr and len(cr) >= 2:
                # cr[0] → PR: x=recall, y=precision[nc,1000] or [1000]
                # cr[1] → F1: x=confidence, y=f1[nc,1000] or [1000]
                metrics["curves"] = {
                    "pr":      (_curve_mean(cr[0][0]), _curve_mean(cr[0][1])),
                    "f1_conf": (_curve_mean(cr[1][0]), _curve_mean(cr[1][1])),
                }
        except Exception as exc:
            console.print(f"[dim]Curve extraction skipped for {model_name}: {exc}[/dim]")
    except Exception:
        # Wrapper fallback (no curve data)
        metrics = model.evaluate(tmp, split="val")

    console.print(
        f"  [green]{model_name}[/green]: "
        f"mAP@50={metrics['mAP50']:.4f}  "
        f"mAP@50-95={metrics['mAP50_95']:.4f}  "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}"
    )
    return metrics


# ─── ROC & Timing Helpers ─────────────────────────────────────────────────────

def compute_roc_data(
    model,
    data_dir: str,
    max_images: int = 500,
    seed: int = 42,
) -> Optional[dict]:
    """
    Image-level binary ROC curve.
    Positive  = image has ≥1 road damage annotation (non-empty label file).
    Negative  = image has no annotation (background / undamaged road).
    For each confidence threshold the model's max detection score on each image
    is compared against the threshold to predict positive/negative.

    Returns dict with: fprs, tprs, auc, n_images — or None if val set is missing.
    """
    val_img_dir = Path(data_dir) / "val" / "images"
    val_lbl_dir = Path(data_dir) / "val" / "labels"

    images = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))
    if not images:
        console.print(f"[yellow]ROC: no val images found in {val_img_dir}[/yellow]")
        return None

    rng = random.Random(seed)
    if len(images) > max_images:
        images = rng.sample(images, max_images)

    # Ground truth: positive if the label file exists and is non-empty
    gt = np.array([
        (val_lbl_dir / (p.stem + ".txt")).exists()
        and (val_lbl_dir / (p.stem + ".txt")).stat().st_size > 0
        for p in images
    ], dtype=bool)

    n_pos = int(gt.sum())
    n_neg = len(gt) - n_pos
    if n_pos == 0 or n_neg == 0:
        console.print("[yellow]ROC: val set has no negative or no positive images — skipping[/yellow]")
        return None

    # Run predict at very low confidence to capture all detections
    console.print(f"  [dim]ROC: running predict on {len(images)} val images (conf=0.001)…[/dim]")
    max_scores = np.zeros(len(images), dtype=float)
    for i, img_path in enumerate(images):
        try:
            preds = model.predict(str(img_path), conf=0.001, verbose=False)
            if preds and preds[0].boxes is not None and len(preds[0].boxes) > 0:
                max_scores[i] = float(preds[0].boxes.conf.max().cpu())
        except Exception:
            pass

    # Vectorised threshold sweep
    thresholds = np.linspace(0.0, 1.0, 101)
    tprs = np.zeros(len(thresholds))
    fprs = np.zeros(len(thresholds))
    for k, t in enumerate(thresholds):
        pred_pos = max_scores >= t
        tprs[k] = (pred_pos & gt).sum() / n_pos
        fprs[k] = (pred_pos & ~gt).sum() / n_neg

    # Sort by FPR ascending for correct AUC via trapezoid rule
    order    = np.argsort(fprs)
    fprs_s   = fprs[order]
    tprs_s   = tprs[order]
    auc      = float(np.trapz(tprs_s, fprs_s))

    return {
        "fprs":     fprs_s.tolist(),
        "tprs":     tprs_s.tolist(),
        "auc":      auc,
        "n_images": len(images),
    }


def measure_inference_time(
    model,
    data_dir: str,
    n: int = 30,
    seed: int = 42,
) -> Optional[float]:
    """
    Measure mean inference latency (ms / image) by timing n val images.
    First 5 images are used as a warmup pass.
    Returns None if val images cannot be found.
    """
    val_img_dir = Path(data_dir) / "val" / "images"
    images = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))
    if not images:
        return None

    rng = random.Random(seed)
    samples = rng.sample(images, min(n + 5, len(images)))

    # Warmup
    try:
        for img in samples[:5]:
            model.predict(str(img), conf=0.25, verbose=False)
    except Exception:
        return None

    times_ms: List[float] = []
    for img in samples[5:]:
        t0 = time.perf_counter()
        model.predict(str(img), conf=0.25, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times_ms)) if times_ms else None


# ─── Plots ────────────────────────────────────────────────────────────────────

def _color(name: str) -> str:
    return MODEL_COLORS.get(name, "#94a3b8")


def plot_metrics_comparison(results: List[dict], output_dir: Path):
    """4-panel bar chart comparing mAP50, mAP50-95, precision, recall."""
    keys = ["mAP50", "mAP50_95", "precision", "recall"]
    titles = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    labels = [r["label"] for r in results]
    colors = [_color(r["name"]) for r in results]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Model Comparison — Road Damage Detection (RDD2022)", fontsize=14, fontweight="bold")

    for ax, key, title in zip(axes, keys, titles):
        vals = [r["metrics"].get(key, 0) for r in results]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", width=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, min(max(vals) * 1.2 + 0.05, 1.05))
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    plt.tight_layout()
    path = output_dir / "metrics_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    console.print(f"[green]Metrics comparison → {path}[/green]")
    plt.close()


def plot_per_class_mAP(results: List[dict], output_dir: Path):
    """Heatmap of per-class mAP50 for each model."""
    valid = [r for r in results if r["metrics"].get("per_class_mAP50")]
    if not valid:
        return

    model_labels = [r["label"] for r in valid]
    nc = len(CLASS_NAMES)
    data = np.array([r["metrics"]["per_class_mAP50"][:nc] for r in valid])

    fig, ax = plt.subplots(figsize=(10, max(3, len(model_labels) * 1.3 + 1)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="mAP@50")

    ax.set_xticks(range(nc))
    ax.set_yticks(range(len(model_labels)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(model_labels, fontsize=11)

    for i in range(len(model_labels)):
        for j in range(nc):
            val = data[i, j]
            text_color = "black" if 0.2 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    ax.set_title("Per-Class mAP@50 Across Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "per_class_mAP.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    console.print(f"[green]Per-class mAP heatmap → {path}[/green]")
    plt.close()


def plot_loss_curves(histories: Dict[str, List[dict]], output_dir: Path):
    """Train and val loss curves (box, cls, dfl) per epoch, one line per model."""
    if not histories:
        return

    loss_keys = [
        ("box_loss", "val_box_loss", "Box Loss"),
        ("cls_loss", "val_cls_loss", "Cls Loss"),
        ("dfl_loss", "val_dfl_loss", "DFL Loss"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight="bold")

    for ax, (train_key, val_key, title) in zip(axes, loss_keys):
        for name, epochs in histories.items():
            color = _color(name)
            label = name.replace("_", " ").title()
            xs = [e["epoch"] for e in epochs]
            ax.plot(xs, [e.get(train_key, 0) for e in epochs],
                    color=color, linewidth=1.5, label=f"{label} train")
            ax.plot(xs, [e.get(val_key, 0) for e in epochs],
                    color=color, linewidth=1.5, linestyle="--", alpha=0.65, label=f"{label} val")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(histories) * 2, 8), bbox_to_anchor=(0.5, -0.06), fontsize=9)
    plt.tight_layout()
    path = output_dir / "loss_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    console.print(f"[green]Loss curves → {path}[/green]")
    plt.close()


def plot_metric_curves(histories: Dict[str, List[dict]], output_dir: Path):
    """mAP50, precision, and recall over epochs for each model."""
    if not histories:
        return

    metric_keys = [("mAP50", "mAP@50"), ("precision", "Precision"), ("recall", "Recall")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Validation Metric Curves", fontsize=14, fontweight="bold")

    for ax, (key, title) in zip(axes, metric_keys):
        for name, epochs in histories.items():
            xs = [e["epoch"] for e in epochs]
            ax.plot(xs, [e.get(key, 0) for e in epochs],
                    color=_color(name), linewidth=2,
                    label=name.replace("_", " ").title())
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = output_dir / "metric_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    console.print(f"[green]Metric curves → {path}[/green]")
    plt.close()


def plot_pr_curves(results: List[dict], output_dir: Path):
    """
    Precision-Recall curves (mean across all classes) for each model.
    Uses the per-threshold curve data extracted from Ultralytics val().
    """
    has_curves = [r for r in results if r["metrics"].get("curves", {}).get("pr")]
    if not has_curves:
        console.print("[yellow]PR curves: no curve data available (curves_results not returned by val)[/yellow]")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Precision-Recall Curves (Mean Across Classes)", fontsize=13, fontweight="bold")

    for r in has_curves:
        recall, precision = r["metrics"]["curves"]["pr"]
        mAP = r["metrics"].get("mAP50", 0.0)
        ax.plot(recall, precision, color=_color(r["name"]), linewidth=2,
                label=f"{r['label']}  (mAP@50={mAP:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = output_dir / "pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    console.print(f"[green]PR curves → {path}[/green]")
    plt.close()


def plot_f1_confidence_curves(results: List[dict], output_dir: Path):
    """
    F1 score vs confidence threshold for each model.
    Vertical dashed lines mark the peak F1 operating point per model.
    """
    has_curves = [r for r in results if r["metrics"].get("curves", {}).get("f1_conf")]
    if not has_curves:
        console.print("[yellow]F1-confidence curves: no curve data available[/yellow]")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("F1-Confidence Curves", fontsize=13, fontweight="bold")

    for r in has_curves:
        conf, f1 = r["metrics"]["curves"]["f1_conf"]
        best_idx  = int(np.argmax(f1))
        best_conf = float(conf[best_idx])
        best_f1   = float(f1[best_idx])
        color     = _color(r["name"])

        ax.plot(conf, f1, color=color, linewidth=2,
                label=f"{r['label']}  (peak F1={best_f1:.3f} @ conf={best_conf:.2f})")
        ax.axvline(best_conf, color=color, linewidth=0.9, linestyle="--", alpha=0.55)

    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = output_dir / "f1_confidence_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    console.print(f"[green]F1-confidence curves → {path}[/green]")
    plt.close()


def plot_roc_curves(roc_data: Dict[str, dict], output_dir: Path):
    """
    Image-level ROC curves.
    Each model is scored by its maximum detection confidence on a val image;
    at each threshold that score classifies the image as damaged or not.
    Ground-truth positive = image has ≥1 bounding-box annotation.
    """
    if not roc_data:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("ROC Curves — Image-Level Damage Detection", fontsize=13, fontweight="bold")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random (AUC=0.500)")

    for name, data in roc_data.items():
        label = name.replace("_", " ").title()
        ax.plot(data["fprs"], data["tprs"], color=_color(name), linewidth=2,
                label=f"{label}  (AUC={data['auc']:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Note on methodology
    n_img = next(iter(roc_data.values()), {}).get("n_images", "?")
    ax.text(0.98, 0.06, f"n={n_img} val images sampled",
            transform=ax.transAxes, ha="right", fontsize=8, color="#64748b")

    plt.tight_layout()
    path = output_dir / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    console.print(f"[green]ROC curves → {path}[/green]")
    plt.close()


def plot_timing_comparison(timing_data: Dict[str, float], output_dir: Path, device: str = "cuda"):
    """
    Grouped bar chart of mean inference latency (ms / image) per model.
    Includes FPS annotation and model parameter count where known.
    """
    if not timing_data:
        return

    names  = list(timing_data.keys())
    labels = [n.replace("_", " ").title() for n in names]
    times  = [timing_data[n] for n in names]
    colors = [_color(n) for n in names]
    fps    = [1000.0 / t if t > 0 else 0 for t in times]

    fig, ax = plt.subplots(figsize=(9, 5))
    device_label = device.upper() if device != "cuda" else "GPU (CUDA)"
    ax.set_title(f"Inference Latency per Image ({device_label})", fontsize=13, fontweight="bold")

    bars = ax.bar(labels, times, color=colors, alpha=0.85, edgecolor="white", width=0.6)
    top  = max(times) if times else 1.0

    for bar, ms, fp in zip(bars, times, fps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.02,
                f"{ms:.0f} ms\n({fp:.1f} FPS)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("ms / image", fontsize=11)
    ax.tick_params(axis="x", rotation=10)
    ax.set_ylim(0, top * 1.35)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = output_dir / "timing_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    console.print(f"[green]Timing comparison → {path}[/green]")
    plt.close()


# ─── Detection Grid ───────────────────────────────────────────────────────────

def collect_sample_images(data_dir: str, n: int, seed: int = 42) -> List[str]:
    """Pick N random images from val/images."""
    val_dir = Path(data_dir) / "val" / "images"
    images = sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.png"))
    if not images:
        console.print(f"[yellow]No val images found in {val_dir}[/yellow]")
        return []
    random.seed(seed)
    return [str(p) for p in random.sample(images, min(n, len(images)))]


def _draw_boxes(ax, img_path: str, predictions, title: str):
    """Render an image with bounding boxes from an Ultralytics result object."""
    from PIL import Image
    img = np.array(Image.open(img_path).convert("RGB"))
    ax.imshow(img)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    ax.axis("off")

    if not predictions:
        return
    result = predictions[0]
    if result.boxes is None or len(result.boxes) == 0:
        return

    boxes   = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs   = result.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        x1, y1, x2, y2 = box
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        color = CLASS_COLORS.get(cls_name, "#ffffff")
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none",
        ))
        ax.text(x1, y1 - 3, f"{cls_name} {conf:.2f}",
                fontsize=6, color=color,
                bbox=dict(facecolor="black", alpha=0.45, pad=1, edgecolor="none"))


def plot_detection_grid(models_loaded: List[dict], sample_images: List[str], output_dir: Path):
    """
    Grid where rows = sample images, cols = models.
    Each cell is the image with that model's detections drawn on it.
    """
    if not sample_images or not models_loaded:
        return

    n_imgs   = len(sample_images)
    n_models = len(models_loaded)

    fig, axes = plt.subplots(n_imgs, n_models, figsize=(n_models * 4, n_imgs * 3.5))
    # Normalise axes to always be 2-D
    if n_imgs == 1:
        axes = axes[np.newaxis, :]
    if n_models == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("Detection Results Across Models", fontsize=13, fontweight="bold")

    for col, entry in enumerate(models_loaded):
        model = entry["model"]
        conf  = entry.get("conf", 0.25)
        for row, img_path in enumerate(sample_images):
            ax = axes[row, col]
            header = entry["label"] if row == 0 else ""
            try:
                preds = model.predict(img_path, conf=conf)
                _draw_boxes(ax, img_path, preds, header)
            except Exception as exc:
                ax.axis("off")
                ax.set_title(f"{header}\nError: {exc}", fontsize=7)

    plt.tight_layout()
    path = output_dir / "detection_grid.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    console.print(f"[green]Detection grid → {path}[/green]")
    plt.close()


# ─── Results Table ────────────────────────────────────────────────────────────

def print_results_table(results: List[dict]) -> pd.DataFrame:
    table = Table(title="Ablation Study — Final Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Model",       style="white", min_width=22)
    table.add_column("mAP@50",      justify="right")
    table.add_column("mAP@50-95",   justify="right")
    table.add_column("Precision",   justify="right")
    table.add_column("Recall",      justify="right")
    table.add_column("F1",          justify="right")
    table.add_column("ΔmAP@50",     justify="right")

    baseline_mAP = next(
        (r["metrics"].get("mAP50", 0) for r in results if "baseline" in r["name"]),
        results[0]["metrics"].get("mAP50", 0),
    )

    rows = []
    for r in results:
        m = r["metrics"]
        p, rec = m.get("precision", 0), m.get("recall", 0)
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        delta = m.get("mAP50", 0) - baseline_mAP
        sign  = "+" if delta > 0 else ""
        color = "green" if delta > 0 else ("red" if delta < 0 else "white")
        table.add_row(
            r["label"],
            f"{m.get('mAP50', 0):.4f}",
            f"{m.get('mAP50_95', 0):.4f}",
            f"{p:.4f}",
            f"{rec:.4f}",
            f"{f1:.4f}",
            f"[{color}]{sign}{delta:.4f}[/{color}]",
        )
        rows.append({
            "Model":     r["label"],
            "mAP50":     m.get("mAP50", 0),
            "mAP50_95":  m.get("mAP50_95", 0),
            "Precision": p,
            "Recall":    rec,
            "F1":        f1,
        })

    console.print(table)
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study — evaluate trained models, compare metrics, generate plots"
    )
    parser.add_argument("--config",         required=True,
                        help="Path to ablation.yaml")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Root dir containing per-model checkpoint folders")
    parser.add_argument("--data-dir",       default=None,
                        help="Override dataset root (e.g. /content/RDD_SPLIT)")
    parser.add_argument("--output-dir",     default=None,
                        help="Override output dir for plots and CSV")
    parser.add_argument("--device",         default="cuda",
                        help="cuda / cpu / mps")
    parser.add_argument("--model",          default=None,
                        help="Evaluate only this model name (e.g. yolov11_baseline)")
    parser.add_argument("--curves-only",    action="store_true",
                        help="Skip evaluation — only plot training curves from saved history")
    parser.add_argument("--no-detection",   action="store_true",
                        help="Skip detection grid (saves time when GPU memory is tight)")
    parser.add_argument("--no-roc",         action="store_true",
                        help="Skip ROC curve computation (skips per-image predict sweep)")
    parser.add_argument("--num-samples",    type=int, default=6,
                        help="Number of val images for detection grid (default: 6)")
    parser.add_argument("--roc-images",     type=int, default=500,
                        help="Max val images used for ROC computation (default: 500)")
    args = parser.parse_args()

    abl_cfg     = load_config(args.config)
    config_root = Path(args.config).parent.resolve()
    output_dir  = Path(args.output_dir or abl_cfg["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or abl_cfg["evaluation"]["data_dir"]

    entries = abl_cfg["models"]
    if args.model:
        entries = [e for e in entries if e["name"] == args.model]
        if not entries:
            console.print(f"[red]Model '{args.model}' not found in config.[/red]")
            sys.exit(1)

    console.print(Panel(
        f"[bold]Ablation Study[/bold] — {len(entries)} model(s)\n"
        f"Checkpoints : {args.checkpoint_dir}\n"
        f"Data        : {data_dir}\n"
        f"Output      : {output_dir}",
        border_style="blue",
    ))

    # ── Load training histories (CPU-only, no model weights needed) ───────────
    histories: Dict[str, List[dict]] = {}
    for e in entries:
        h = load_training_history(args.checkpoint_dir, e["name"])
        if h:
            histories[e["name"]] = h
            console.print(f"[dim]History: {e['name']} — {len(h)} epochs[/dim]")
        else:
            console.print(f"[yellow]No training history for {e['name']}[/yellow]")

    # ── Evaluate models ───────────────────────────────────────────────────────
    results:       List[dict] = []
    models_loaded: List[dict] = []

    if not args.curves_only:
        console.print("\n[bold]Evaluating models on val set...[/bold]")
        for e in entries:
            name  = e["name"]
            label = e.get("label", name.replace("_", " ").title())
            try:
                ckpt = resolve_checkpoint(args.checkpoint_dir, name)
                model, model_config = load_model_from_checkpoint(e, ckpt, config_root, args.device)
                metrics = evaluate_model(model, name, data_dir, model_config)
                results.append({"name": name, "label": label, "metrics": metrics})
                models_loaded.append({
                    "name": name, "label": label, "model": model,
                    "conf": model_config.get("evaluation", {}).get("conf_threshold", 0.25),
                })
            except FileNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")
    else:
        # Build a minimal results list from the last epoch of each history
        for e in entries:
            name  = e["name"]
            label = e.get("label", name.replace("_", " ").title())
            h = histories.get(name)
            if h:
                last = h[-1]
                results.append({"name": name, "label": label, "metrics": {
                    "mAP50":     last.get("mAP50", 0),
                    "mAP50_95":  last.get("mAP50_95", 0),
                    "precision": last.get("precision", 0),
                    "recall":    last.get("recall", 0),
                }})

    # ── Print and save results table ──────────────────────────────────────────
    if results:
        console.print()
        df = print_results_table(results)
        csv_path = output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        console.print(f"[green]Results CSV → {csv_path}[/green]")

    # ── Generate static plots ─────────────────────────────────────────────────
    console.print("\n[bold]Generating plots...[/bold]")

    if results and not args.curves_only:
        plot_metrics_comparison(results, output_dir)
        plot_per_class_mAP(results, output_dir)
        plot_pr_curves(results, output_dir)
        plot_f1_confidence_curves(results, output_dir)

    if histories:
        plot_loss_curves(histories, output_dir)
        plot_metric_curves(histories, output_dir)

    # ── ROC curves and timing (require running predict on val images) ─────────
    if not args.curves_only and not args.no_roc and models_loaded:
        console.print("\n[bold]Computing ROC curves and timing...[/bold]")
        roc_data:    Dict[str, dict]  = {}
        timing_data: Dict[str, float] = {}

        for entry in models_loaded:
            name  = entry["name"]
            model = entry["model"]
            console.print(f"  [cyan]{name}[/cyan]")

            roc = compute_roc_data(model, data_dir, max_images=args.roc_images)
            if roc:
                roc_data[name] = roc
                console.print(f"    ROC AUC = {roc['auc']:.4f}")

            t_ms = measure_inference_time(model, data_dir)
            if t_ms is not None:
                timing_data[name] = t_ms
                console.print(f"    Latency = {t_ms:.1f} ms/img  ({1000/t_ms:.1f} FPS)")

        plot_roc_curves(roc_data, output_dir)
        plot_timing_comparison(timing_data, output_dir, device=args.device)

    # ── Detection grid ────────────────────────────────────────────────────────
    if not args.no_detection and models_loaded:
        console.print(f"\n[bold]Running detection on {args.num_samples} sample images...[/bold]")
        samples = collect_sample_images(data_dir, n=args.num_samples)
        if samples:
            plot_detection_grid(models_loaded, samples, output_dir)

    console.print(Panel(
        f"All outputs in [bold cyan]{output_dir}[/bold cyan]",
        title="Done", border_style="green",
    ))


if __name__ == "__main__":
    main()
