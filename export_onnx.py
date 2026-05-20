"""
export_onnx.py
Export trained .pt checkpoints to ONNX for fast hardware-agnostic inference.

ONNX models are ~2x faster than PyTorch on CPU, and compatible with
TensorRT, CoreML, OpenVINO, and ONNX Runtime.

Usage:
    python export_onnx.py --checkpoint-dir /content/drive/MyDrive/road_ckpts
    python export_onnx.py --checkpoint-dir ./runs --model yolov11_baseline

CMP 295 SJSU | Road Damage Detection
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

MODEL_CONFIGS = {
    "yolov11_baseline":  {"type": "yolo",   "imgsz": 640},
    "yolov11_improved":  {"type": "yolo",   "imgsz": 640},
    "rtdetrv2_baseline": {"type": "rtdetr", "imgsz": 640},
    "rtdetrv2_improved": {"type": "rtdetr", "imgsz": 640},
}


def export_model(checkpoint_dir: str, model_name: str) -> tuple[bool, str]:
    """Export best.pt → best.onnx. Returns (success, onnx_path_or_error)."""
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        return False, f"Unknown model: {model_name}"

    weights_dir = Path(checkpoint_dir) / model_name / "weights"
    pt_path     = weights_dir / "best.pt"
    onnx_path   = weights_dir / "best.onnx"

    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        console.print(f"[yellow]Already exists: {onnx_path} ({size_mb:.1f} MB)[/yellow]")
        return True, str(onnx_path)

    if not pt_path.exists():
        return False, f"No checkpoint: {pt_path}"

    console.print(f"[bold]Exporting {model_name} ({pt_path.name} → best.onnx)...[/bold]")

    try:
        if cfg["type"] == "rtdetr":
            from ultralytics import RTDETR
            model = RTDETR(str(pt_path))
        else:
            from ultralytics import YOLO
            model = YOLO(str(pt_path))

        model.export(
            format="onnx",
            imgsz=cfg["imgsz"],
            simplify=True,
            dynamic=False,
            opset=12,
        )

        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            pt_mb   = pt_path.stat().st_size / 1024 / 1024
            console.print(
                f"[green]✓ {model_name}: {pt_mb:.1f} MB → {size_mb:.1f} MB ONNX[/green]"
            )
            return True, str(onnx_path)

        return False, "ONNX file not found after export"

    except Exception as exc:
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Export road damage models to ONNX")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Root checkpoint directory (e.g. /content/drive/MyDrive/road_ckpts)")
    parser.add_argument("--model", default=None,
                        help="Export only this model name (exports all if omitted)")
    args = parser.parse_args()

    targets = [args.model] if args.model else list(MODEL_CONFIGS.keys())

    table = Table(title="ONNX Export Results", header_style="bold cyan", show_header=True)
    table.add_column("Model",      style="white", min_width=22)
    table.add_column("Status",     justify="center")
    table.add_column("ONNX Path / Error")

    for name in targets:
        ok, detail = export_model(args.checkpoint_dir, name)
        status = "[green]✓ OK[/green]" if ok else "[red]✗ FAILED[/red]"
        table.add_row(name, status, detail)

    console.print("\n")
    console.print(table)
    console.print(
        "\n[dim]Run server.py to serve these models — it auto-detects ONNX over .pt.[/dim]"
    )


if __name__ == "__main__":
    main()
