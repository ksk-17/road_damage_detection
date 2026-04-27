"""
scripts/prepare_dataset.py
RDD2022 Dataset Preparation Script
  - Download from IEEE DataPort / Kaggle
  - Convert to YOLO format (optional)
  - Verify integrity and print statistics

Usage:
    python scripts/prepare_dataset.py --root ./data/rdd2022 --countries Japan India USA
    python scripts/prepare_dataset.py --root ./data/rdd2022 --stats-only
    python scripts/prepare_dataset.py --root ./data/rdd2022 --convert-yolo

CMP 295 SJSU | Road Damage Detection
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

console = Console()

# ─── Class Definitions ────────────────────────────────────────────────────────
CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

COUNTRIES = ["Japan", "India", "USA", "Czech", "Norway", "China"]


# ─── Verification ─────────────────────────────────────────────────────────────
def verify_dataset(root: Path, countries: List[str]) -> Dict:
    """Verify dataset structure and count images/annotations."""
    stats = {}
    missing = []

    for country in countries:
        for split in ["train", "test"]:
            img_dir = root / country / split / "images"
            ann_dir = root / country / split / "annotations" / "xmls"

            if not img_dir.exists():
                missing.append(str(img_dir))
                continue

            images = list(img_dir.glob("*.jpg"))
            xml_files = list(ann_dir.glob("*.xml")) if ann_dir.exists() else []

            key = f"{country}/{split}"
            stats[key] = {
                "images": len(images),
                "annotations": len(xml_files),
                "has_labels": ann_dir.exists() and len(xml_files) > 0,
            }

    return stats, missing


def print_dataset_stats(root: Path, countries: List[str]):
    """Print detailed dataset statistics."""
    console.print("\n[bold blue]=== RDD2022 Dataset Statistics ===[/bold blue]")

    for country in countries:
        train_dir = root / country / "train"
        if not train_dir.exists():
            console.print(f"[red]{country}: NOT FOUND at {train_dir}[/red]")
            continue

        img_dir = train_dir / "images"
        ann_dir = train_dir / "annotations" / "xmls"
        images = list(img_dir.glob("*.jpg")) if img_dir.exists() else []
        xmls = list(ann_dir.glob("*.xml")) if ann_dir.exists() else []

        # Count per-class annotations
        class_counts = {c: 0 for c in CLASS_NAMES}
        total_boxes = 0
        empty_images = 0

        for xml_path in track(xmls, description=f"Parsing {country}...", transient=True):
            tree = ET.parse(xml_path)
            root_elem = tree.getroot()
            objs = root_elem.findall("object")
            if len(objs) == 0:
                empty_images += 1
            for obj in objs:
                name = obj.find("name").text.strip()
                if name in class_counts:
                    class_counts[name] += 1
                    total_boxes += 1

        table = Table(title=f"[bold]{country}[/bold] ({len(images)} images, {total_boxes} boxes)")
        table.add_column("Class", style="cyan")
        table.add_column("Description")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")

        descriptions = {
            "D00": "Longitudinal Crack",
            "D10": "Transverse Crack",
            "D20": "Alligator Crack",
            "D40": "Pothole",
        }

        for cls, cnt in class_counts.items():
            pct = 100 * cnt / total_boxes if total_boxes > 0 else 0
            table.add_row(cls, descriptions[cls], str(cnt), f"{pct:.1f}%")

        table.add_row("", "[dim]Empty images[/dim]", str(empty_images), "", style="dim")
        console.print(table)


# ─── YOLO Format Conversion ───────────────────────────────────────────────────
def convert_voc_to_yolo(xml_path: Path, img_width: int, img_height: int) -> List[str]:
    """
    Convert PASCAL VOC XML annotation to YOLO format.
    YOLO: <class_id> <cx> <cy> <w> <h>  (all normalized 0-1)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_TO_IDX:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Clamp to image bounds
        xmin = max(0, min(xmin, img_width))
        ymin = max(0, min(ymin, img_height))
        xmax = max(0, min(xmax, img_width))
        ymax = max(0, min(ymax, img_height))

        if xmax <= xmin or ymax <= ymin:
            continue

        cx = (xmin + xmax) / 2 / img_width
        cy = (ymin + ymax) / 2 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        lines.append(f"{CLASS_TO_IDX[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def convert_dataset_to_yolo(root: Path, countries: List[str]):
    """Convert entire dataset from VOC XML to YOLO .txt format."""
    from PIL import Image

    console.print("\n[bold]Converting dataset to YOLO format...[/bold]")

    for country in countries:
        for split in ["train"]:
            img_dir = root / country / split / "images"
            ann_dir = root / country / split / "annotations" / "xmls"
            yolo_dir = root / country / split / "labels"
            yolo_dir.mkdir(exist_ok=True)

            if not img_dir.exists():
                console.print(f"[yellow]Skipping {country}/{split} — not found[/yellow]")
                continue

            converted = 0
            for img_path in track(list(img_dir.glob("*.jpg")), description=f"{country}/{split}"):
                xml_path = ann_dir / (img_path.stem + ".xml")
                yolo_path = yolo_dir / (img_path.stem + ".txt")

                # Get image dimensions
                with Image.open(img_path) as img:
                    w, h = img.size

                if xml_path.exists():
                    lines = convert_voc_to_yolo(xml_path, w, h)
                else:
                    lines = []  # Empty label = no damage (negative sample)

                with open(yolo_path, "w") as f:
                    f.write("\n".join(lines))

                converted += 1

            console.print(f"  {country}/{split}: {converted} label files created → {yolo_dir}")


# ─── Dataset YAML Generator ───────────────────────────────────────────────────
def generate_dataset_yaml(root: Path, train_countries: List[str], val_countries: List[str]):
    """Generate YOLO-format dataset.yaml file."""
    import yaml

    cfg = {
        "path": str(root.resolve()),
        "train": [f"{c}/train/images" for c in train_countries],
        "val":   [f"{c}/train/images" for c in val_countries],
        "nc": 4,
        "names": CLASS_NAMES,
    }

    yaml_path = root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    console.print(f"\n[green]Dataset YAML saved to {yaml_path}[/green]")
    console.print("  Use this path with: python scripts/train.py --config configs/...")
    return yaml_path


# ─── Download Instructions ────────────────────────────────────────────────────
def print_download_instructions(root: Path):
    console.print(Panel(
        """[bold]RDD2022 Dataset Download Instructions[/bold]

1. [cyan]IEEE DataPort (official):[/cyan]
   https://doi.org/10.21227/ke5f-n977
   Requires free IEEE DataPort account.

2. [cyan]Kaggle mirror:[/cyan]
   kaggle datasets download -d hahajjjjj/rdd2022
   
3. [cyan]Direct (CRDDC2022 challenge):[/cyan]
   https://github.com/sekilab/RoadDamageDetector

After downloading, extract to:
    """ + str(root) + """

Expected structure:
    rdd2022/
      Japan/train/images/*.jpg
      Japan/train/annotations/xmls/*.xml
      India/train/images/*.jpg
      ...

Then run:
    python scripts/prepare_dataset.py --root """ + str(root) + """ --stats-only
""",
        title="Download Instructions", border_style="blue"
    ))


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare RDD2022 Dataset")
    parser.add_argument("--root", default="./data/rdd2022", help="Dataset root directory")
    parser.add_argument("--countries", nargs="+", default=COUNTRIES, help="Countries to process")
    parser.add_argument("--stats-only", action="store_true", help="Only print statistics")
    parser.add_argument("--convert-yolo", action="store_true", help="Convert VOC XML → YOLO .txt")
    parser.add_argument("--gen-yaml", action="store_true", help="Generate dataset.yaml")
    parser.add_argument("--train-countries", nargs="+", default=["Japan"])
    parser.add_argument("--val-countries", nargs="+", default=["Japan"])
    parser.add_argument("--download-info", action="store_true", help="Show download instructions")
    args = parser.parse_args()

    root = Path(args.root)

    if args.download_info:
        print_download_instructions(root)
        return

    if not root.exists():
        console.print(f"[red]Dataset root not found: {root}[/red]")
        print_download_instructions(root)
        sys.exit(1)

    # Stats
    print_dataset_stats(root, args.countries)

    if args.convert_yolo:
        convert_dataset_to_yolo(root, args.countries)

    if args.gen_yaml:
        generate_dataset_yaml(root, args.train_countries, args.val_countries)

    if not args.stats_only and not args.convert_yolo and not args.gen_yaml:
        # Run all preparation steps
        convert_dataset_to_yolo(root, args.countries)
        generate_dataset_yaml(root, args.train_countries, args.val_countries)
        console.print("\n[bold green]Dataset preparation complete![/bold green]")


if __name__ == "__main__":
    main()
