"""
scripts/train.py
Main Training Entry Point — Road Damage Detection
Supports: YOLOv11 baseline, RT-DETRv2, and improved variants

Usage:
    python scripts/train.py --config configs/yolov11_baseline.yaml
    python scripts/train.py --config configs/yolov11_improved.yaml
    python scripts/train.py --config configs/rtdetr_baseline.yaml

CMP 295 SJSU | Road Damage Detection
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from models.yolov11_model import YOLOv11RoadDamage
    from models.rtdetr_model import RTDETRv2RoadDamage
except ModuleNotFoundError:
    from yolov11_model import YOLOv11RoadDamage
    from rtdetr_model import RTDETRv2RoadDamage

console = Console()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_config_summary(config: dict, config_path: str):
    table = Table(title=f"Training Configuration: {config_path}", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    ds_cfg = config.get("dataset", {})
    loss_cfg = config.get("loss", {})
    aug_cfg = config.get("augmentation", {})

    table.add_row("Model", f"{model_cfg.get('name', 'N/A')} ({model_cfg.get('variant', 'N/A')})")
    table.add_row("FPN Scales", str(model_cfg.get("fpn_scales", 3)))
    table.add_row("Classes", str(model_cfg.get("num_classes", 4)))
    table.add_row("Image Size", str(ds_cfg.get("image_size", 640)))
    table.add_row("Train Countries", str(ds_cfg.get("train_countries", [])))
    table.add_row("Epochs", str(train_cfg.get("epochs", 100)))
    table.add_row("Batch Size", str(train_cfg.get("batch_size", 16)))
    table.add_row("Box Loss", loss_cfg.get("box", "ciou"))
    table.add_row("Cls Loss", loss_cfg.get("cls", "bce"))
    table.add_row("Domain Randomization", str(aug_cfg.get("domain_randomization", False)))
    table.add_row("Device", train_cfg.get("device", "cuda"))

    console.print(table)


def check_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]⚠ CUDA not available, falling back to CPU[/yellow]")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        console.print("[yellow]⚠ MPS not available, falling back to CPU[/yellow]")
        return "cpu"
    return requested


def validate_dataset(config: dict):
    ds_cfg = config.get("dataset", {})
    root = Path(ds_cfg.get("root", "")).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Update dataset.root in the config to point to data/RDD_SPLIT."
        )

    missing_paths = []
    for split in ("train", "val"):
        img_dir = root / split / "images"
        if not img_dir.exists():
            missing_paths.append(str(img_dir))

    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Dataset structure is incomplete. Missing required directories:\n"
            f"{missing}\n"
            "Expected layout: <dataset.root>/train/images  and  <dataset.root>/val/images"
        )


def print_improvements_active(config: dict):
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    aug_cfg = config.get("augmentation", {})

    improvements = []
    if model_cfg.get("fpn_scales", 3) == 4:
        improvements.append("✓ [bold green]Improvement 1[/bold green]: Extra P2 FPN head (4 scales)")
    if loss_cfg.get("box") in ("wiou", "focal_wiou"):
        improvements.append("✓ [bold green]Improvement 2[/bold green]: WIoU / Focal+WIoU loss")
    if aug_cfg.get("domain_randomization", False):
        improvements.append("✓ [bold green]Improvement 3[/bold green]: Domain randomization augmentation")

    if improvements:
        console.print(Panel("\n".join(improvements), title="Active Improvements", border_style="green"))
    else:
        console.print(Panel("Running [bold]baseline[/bold] configuration (no improvements active)", border_style="yellow"))


def build_model(config: dict, device: str):
    model_cfg = config["model"]
    name = model_cfg["name"].lower()

    if name == "yolov11":
        return YOLOv11RoadDamage(
            variant=model_cfg["variant"],
            weights=model_cfg.get("pretrained_weights"),
            fpn_scales=model_cfg.get("fpn_scales", 3),
            num_classes=model_cfg["num_classes"],
            device=device,
        )
    elif name in ("rtdetrv2", "rtdetr"):
        return RTDETRv2RoadDamage(
            variant=model_cfg["variant"],
            weights=model_cfg.get("pretrained_weights"),
            num_classes=model_cfg["num_classes"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: yolov11, rtdetrv2")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Road Damage Detection Model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset root directory (e.g. /content/RDD_SPLIT)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu/mps)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size (e.g. 512 for faster Colab runs)")
    parser.add_argument("--workers", type=int, default=None, help="Override dataloader workers (use 2 on Colab)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    console.print(f"\n[bold blue]Road Damage Detection — Training Pipeline[/bold blue]")
    console.print(f"Config: {args.config}\n")

    config = load_config(args.config)

    # Apply CLI overrides
    if args.data_dir:
        config["dataset"]["root"] = args.data_dir
    if args.device:
        config["training"]["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch:
        config["training"]["batch_size"] = args.batch
    if args.imgsz:
        config["dataset"]["image_size"] = args.imgsz
    if args.workers is not None:
        config["training"]["num_workers"] = args.workers
    if args.resume:
        config["model"]["pretrained_weights"] = args.resume

    # ── Validate device ───────────────────────────────────────────────────────
    device = check_device(config["training"].get("device", "cuda"))
    config["training"]["device"] = device

    # ── Validate dataset ──────────────────────────────────────────────────────
    validate_dataset(config)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_config_summary(config, args.config)
    print_improvements_active(config)

    if args.dry_run:
        console.print("\n[yellow]Dry run complete — config valid, no training performed.[/yellow]")
        return

    # ── Build model ───────────────────────────────────────────────────────────
    console.print("\n[bold]Building model...[/bold]")
    model = build_model(config, device)

    # ── Train ─────────────────────────────────────────────────────────────────
    console.print("\n[bold green]Starting training...[/bold green]")
    start_time = time.time()

    results = model.train(config)

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    # ── Results summary ───────────────────────────────────────────────────────
    console.print(f"\n[bold green]Training complete![/bold green] "
                  f"({int(hours)}h {int(minutes)}m {int(seconds)}s)")

    save_dir = config["output"]["save_dir"]
    console.print(f"Weights saved to: [cyan]{save_dir}[/cyan]")

    # Print final metrics if available
    if results is not None:
        try:
            table = Table(title="Final Validation Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("mAP@50", f"{results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
            table.add_row("mAP@50-95", f"{results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
            table.add_row("Precision", f"{results.results_dict.get('metrics/precision(B)', 0):.4f}")
            table.add_row("Recall", f"{results.results_dict.get('metrics/recall(B)', 0):.4f}")
            console.print(table)
        except Exception:
            pass


if __name__ == "__main__":
    main()
