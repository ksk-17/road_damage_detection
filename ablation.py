"""
scripts/ablation.py
Ablation Study Runner — Road Damage Detection
Tests each improvement independently and combined, generates comparison table.

Experiments:
  1. Baseline (CIoU, 3-scale FPN, no domain adapt)
  2. FPN-4 only
  3. Focal+WIoU only
  4. Domain adaptation only
  5. FPN-4 + Focal+WIoU
  6. All improvements

Usage:
    python scripts/ablation.py --config configs/ablation.yaml
    python scripts/ablation.py --config configs/ablation.yaml --results-only ./runs/ablation
    python scripts/ablation.py --config configs/ablation.yaml --plot

CMP 295 SJSU | Road Damage Detection
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train import build_model, load_config

console = Console()

# ─── Ablation Experiment ──────────────────────────────────────────────────────

EXPERIMENT_LABELS = {
    "baseline":           "Baseline",
    "fpn4_only":          "+ 4th FPN Scale",
    "focal_wiou_only":    "+ Focal+WIoU",
    "domain_adapt_only":  "+ Domain Adapt",
    "fpn4_focal_wiou":    "+ FPN4 + Focal+WIoU",
    "full_improved":      "All Improvements",
}

IMPROVEMENT_COLORS = {
    "baseline":           "#64748b",
    "fpn4_only":          "#3b82f6",
    "focal_wiou_only":    "#f59e0b",
    "domain_adapt_only":  "#10b981",
    "fpn4_focal_wiou":    "#8b5cf6",
    "full_improved":      "#ef4444",
}


def apply_overrides(base_config: dict, overrides: dict) -> dict:
    """Deep-apply override key.subkey: value to a config dict."""
    import copy
    config = copy.deepcopy(base_config)
    for key_path, value in overrides.items():
        keys = key_path.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config


def run_experiment(
    name: str,
    exp_config: dict,
    base_config: dict,
    output_dir: Path,
    countries: List[str],
    device: str,
) -> Dict:
    """Train and evaluate a single ablation experiment."""

    overrides = exp_config.get("overrides", {})
    config = apply_overrides(base_config, overrides)
    config["logging"]["name"] = name
    config["output"]["save_dir"] = str(output_dir / name)

    console.print(f"\n[bold]Running experiment: [cyan]{name}[/cyan][/bold]")
    console.print(f"  {exp_config['description']}")

    model = build_model(config, device)
    train_results = model.train(config)

    # Evaluate on each test country
    country_metrics = {}
    for country in countries:
        import tempfile, yaml as _yaml
        ds_yaml = {
            "path": str(Path(config["dataset"]["root"]).resolve()),
            "train": [f"{config['dataset']['train_countries'][0]}/train/images"],
            "val":   [f"{country}/train/images"],
            "nc": 4,
            "names": ["D00", "D10", "D20", "D40"],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            _yaml.dump(ds_yaml, f)
            tmp_path = f.name

        metrics = model.evaluate(tmp_path, split="val")
        country_metrics[country] = metrics
        console.print(f"  {country}: mAP@50={metrics['mAP50']:.4f} | mAP@50-95={metrics['mAP50_95']:.4f}")

    result = {
        "name": name,
        "label": EXPERIMENT_LABELS.get(name, name),
        "description": exp_config["description"],
        "per_country": country_metrics,
        "overall_mAP50": np.mean([m["mAP50"] for m in country_metrics.values()]),
        "overall_mAP50_95": np.mean([m["mAP50_95"] for m in country_metrics.values()]),
    }

    # Save result
    result_path = output_dir / f"{name}_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"  Saved to {result_path}")

    return result


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_ablation_results(results: List[Dict], output_dir: Path):
    """Generate ablation comparison charts."""

    names = [r["name"] for r in results]
    labels = [r["label"] for r in results]
    mAP50 = [r["overall_mAP50"] for r in results]
    mAP50_95 = [r["overall_mAP50_95"] for r in results]
    colors = [IMPROVEMENT_COLORS.get(n, "#94a3b8") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ablation Study — Road Damage Detection (RDD2022)", fontsize=14, fontweight="bold")

    for ax, values, metric in zip(axes, [mAP50, mAP50_95], ["mAP@50", "mAP@50-95"]):
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.85, edgecolor="white")
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(values) * 1.15)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Value labels
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    chart_path = output_dir / "ablation_bar_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    console.print(f"\n[green]Bar chart saved to {chart_path}[/green]")
    plt.close()

    # Per-country heatmap for full_improved vs baseline
    _plot_domain_comparison(results, output_dir)


def _plot_domain_comparison(results: List[Dict], output_dir: Path):
    """Cross-country mAP comparison heatmap."""
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    full = next((r for r in results if r["name"] == "full_improved"), None)
    if not baseline or not full:
        return

    countries = list(baseline["per_country"].keys())
    metrics = ["mAP50", "mAP50_95"]

    data = np.array([
        [baseline["per_country"][c][m] for c in countries] for m in metrics
    ] + [
        [full["per_country"][c][m] for c in countries] for m in metrics
    ])

    row_labels = ["Baseline mAP@50", "Baseline mAP@50-95", "Improved mAP@50", "Improved mAP@50-95"]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.2, vmax=0.7)
    plt.colorbar(im, ax=ax, label="mAP Score")

    ax.set_xticks(range(len(countries)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(countries)
    ax.set_yticklabels(row_labels)

    for i in range(len(row_labels)):
        for j in range(len(countries)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="black" if 0.3 < data[i,j] < 0.6 else "white")

    ax.set_title("Cross-Country Generalization: Baseline vs. All Improvements", fontweight="bold")
    fig.tight_layout()

    path = output_dir / "domain_adaptation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    console.print(f"[green]Domain heatmap saved to {path}[/green]")
    plt.close()


def print_ablation_table(results: List[Dict]):
    """Rich table summary of all ablation results."""
    table = Table(title="Ablation Study Results", show_header=True, header_style="bold cyan")
    table.add_column("Experiment", style="white")
    table.add_column("mAP@50", justify="right")
    table.add_column("mAP@50-95", justify="right")
    table.add_column("ΔmAP@50 vs. Baseline", justify="right")

    baseline_mAP50 = next(r["overall_mAP50"] for r in results if r["name"] == "baseline")

    for r in results:
        delta = r["overall_mAP50"] - baseline_mAP50
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        color = "green" if delta > 0 else ("red" if delta < 0 else "white")
        table.add_row(
            r["label"],
            f"{r['overall_mAP50']:.4f}",
            f"{r['overall_mAP50_95']:.4f}",
            f"[{color}]{delta_str}[/{color}]",
        )

    console.print(table)

    # Save CSV
    df = pd.DataFrame([{
        "Experiment": r["label"],
        "mAP@50": r["overall_mAP50"],
        "mAP@50-95": r["overall_mAP50_95"],
    } for r in results])
    return df


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Ablation Study for Road Damage Detection")
    parser.add_argument("--config", required=True, help="Path to ablation YAML config")
    parser.add_argument("--results-only", type=str, default=None,
                        help="Skip training, load results from this dir and plot/print")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run only one named experiment (e.g. fpn4_only)")
    args = parser.parse_args()

    abl_config = load_config(args.config)
    base_config = load_config(abl_config["base_config"])
    output_dir = Path(abl_config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    countries = abl_config["evaluation"]["countries"]
    device = args.device

    experiments = abl_config["experiments"]
    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            console.print(f"[red]Experiment '{args.experiment}' not found in config.[/red]")
            sys.exit(1)

    # ── Results-only mode ─────────────────────────────────────────────────────
    if args.results_only:
        results_dir = Path(args.results_only)
        results = []
        for exp in experiments:
            p = results_dir / f"{exp['name']}_results.json"
            if p.exists():
                with open(p) as f:
                    results.append(json.load(f))
            else:
                console.print(f"[yellow]Missing: {p}[/yellow]")
    else:
        # ── Run all experiments ───────────────────────────────────────────────
        console.print(Panel(
            f"Running [bold]{len(experiments)}[/bold] ablation experiments\n"
            f"Output: {output_dir}",
            title="Ablation Study", border_style="blue"
        ))

        results = []
        for exp in experiments:
            result = run_experiment(
                name=exp["name"],
                exp_config=exp,
                base_config=base_config,
                output_dir=output_dir,
                countries=countries,
                device=device,
            )
            results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print("\n" + "=" * 60)
    df = print_ablation_table(results)

    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"\n[green]Results saved to {csv_path}[/green]")

    if args.plot or not args.results_only:
        plot_ablation_results(results, output_dir)


if __name__ == "__main__":
    main()
