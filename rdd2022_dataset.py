"""
RDD2022 Dataset — YOLO-label loader with augmentation support
Directory structure: data/RDD_SPLIT/{train,val,test}/{images,labels}/
CMP 295 SJSU | Road Damage Detection
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Class Definitions ───────────────────────────────────────────────────────
CLASS_NAMES = ["D00", "D10", "D20", "D40", "D44"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "D44": "White Line Blur",
}

# Country prefixes as they appear in filenames (e.g. "China_Drone_000001.jpg")
COUNTRIES = ["Japan", "India", "United", "Czech", "Norway", "China"]


def _country_from_filename(name: str) -> str:
    """Extract country prefix from an image filename."""
    return name.split("_")[0]


# ─── RDD2022 Dataset ─────────────────────────────────────────────────────────
class RDD2022Dataset(Dataset):
    """
    PyTorch Dataset for RDD2022 Road Damage Detection.

    Directory structure expected:
        root/
          train/
            images/   *.jpg
            labels/   *.txt  (YOLO format: class cx cy w h, normalised)
          val/
            images/   *.jpg
            labels/   *.txt
          test/
            images/   *.jpg
            labels/   *.txt

    Args:
        root:         Path to RDD_SPLIT root directory
        split:        'train', 'val', or 'test'
        countries:    Optional list of country prefixes to include (e.g. ['Japan']).
                      None means all countries.
        transform:    Albumentations transform pipeline
        image_size:   Resize target (square)
        filter_empty: Skip images with no annotations
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        countries: Optional[List[str]] = None,
        transform=None,
        image_size: int = 640,
        filter_empty: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.countries = set(countries) if countries else None
        self.transform = transform
        self.image_size = image_size
        self.filter_empty = filter_empty

        self.samples: List[Dict] = []
        self._load_samples()

        country_info = list(self.countries) if self.countries else "all"
        print(
            f"[RDD2022] Loaded {len(self.samples)} images "
            f"from {country_info} ({split} split)"
        )

    def _load_samples(self):
        img_dir = self.root / self.split / "images"
        lbl_dir = self.root / self.split / "labels"

        if not img_dir.exists():
            print(f"  [WARN] {img_dir} not found, skipping.")
            return

        for img_path in sorted(img_dir.glob("*.jpg")):
            country = _country_from_filename(img_path.name)
            if self.countries and country not in self.countries:
                continue

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            boxes_yolo, labels = [], []

            if lbl_path.exists():
                boxes_yolo, labels = self._parse_yolo_label(lbl_path)

            if self.filter_empty and len(boxes_yolo) == 0:
                continue

            self.samples.append(
                {
                    "image_path": str(img_path),
                    "lbl_path": str(lbl_path) if lbl_path.exists() else None,
                    "boxes_yolo": boxes_yolo,   # List of [cx, cy, w, h] normalised
                    "labels": labels,            # List of int class indices
                    "country": country,
                }
            )

    def _parse_yolo_label(self, lbl_path: Path) -> Tuple[List, List]:
        """Parse YOLO-format label file (class cx cy w h, all normalised 0-1)."""
        boxes, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if cls not in IDX_TO_CLASS:
                    continue
                if w <= 0 or h <= 0:
                    continue
                boxes.append([cx, cy, w, h])
                labels.append(cls)
        return boxes, labels

    def _yolo_to_pascal(self, boxes_yolo: List, img_w: int, img_h: int) -> np.ndarray:
        """Convert YOLO normalised [cx, cy, w, h] → absolute [xmin, ymin, xmax, ymax]."""
        if not boxes_yolo:
            return np.zeros((0, 4), dtype=np.float32)
        arr = np.array(boxes_yolo, dtype=np.float32)
        cx, cy, w, h = arr[:, 0] * img_w, arr[:, 1] * img_h, arr[:, 2] * img_w, arr[:, 3] * img_h
        xmin = np.clip(cx - w / 2, 0, img_w)
        ymin = np.clip(cy - h / 2, 0, img_h)
        xmax = np.clip(cx + w / 2, 0, img_w)
        ymax = np.clip(cy + h / 2, 0, img_h)
        return np.stack([xmin, ymin, xmax, ymax], axis=1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        boxes = self._yolo_to_pascal(sample["boxes_yolo"], img_w, img_h)
        labels = np.array(sample["labels"], dtype=np.int64) if sample["labels"] else np.zeros(0, dtype=np.int64)

        if self.transform and len(boxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes.tolist(),
                class_labels=labels.tolist(),
            )
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32) if transformed["bboxes"] else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(transformed["class_labels"], dtype=np.int64) if transformed["class_labels"] else np.zeros(0, dtype=np.int64)
        elif self.transform:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed["image"]

        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_path": sample["image_path"],
            "country": sample["country"],
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Returns count per damage class across dataset."""
        dist = {c: 0 for c in CLASS_NAMES}
        for s in self.samples:
            for lbl in s["labels"]:
                dist[IDX_TO_CLASS[lbl]] += 1
        return dist


# ─── Augmentation Pipelines ──────────────────────────────────────────────────
def get_train_transforms(image_size: int = 640, domain_randomization: bool = False) -> A.Compose:
    """
    Training augmentation pipeline.
    Optionally applies domain randomization for cross-country generalization.
    """
    base_transforms = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=114),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=5, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=50, val_shift_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ]

    domain_transforms = []
    if domain_randomization:
        # Simulates different road conditions across countries
        domain_transforms = [
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=1.0),
                A.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.5, hue=0.05, p=1.0),
            ], p=0.7),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.1),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
        ]

    all_transforms = base_transforms + domain_transforms + [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    return A.Compose(
        all_transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3),
    )


def get_val_transforms(image_size: int = 640) -> A.Compose:
    """Validation/test transforms — resize + normalize only."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=114),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


# ─── DataLoader Factory ───────────────────────────────────────────────────────
def build_dataloaders(
    root: str,
    image_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    domain_randomization: bool = False,
    train_countries: Optional[List[str]] = None,
    val_countries: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""

    train_dataset = RDD2022Dataset(
        root=root,
        split="train",
        countries=train_countries,
        transform=get_train_transforms(image_size, domain_randomization),
        image_size=image_size,
        filter_empty=False,
    )

    val_dataset = RDD2022Dataset(
        root=root,
        split="val",
        countries=val_countries,
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate: handles variable-length boxes/labels per image."""
    images = torch.stack([b["image"] for b in batch])
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    paths = [b["image_path"] for b in batch]
    countries = [b["country"] for b in batch]
    return {"images": images, "boxes": boxes, "labels": labels, "paths": paths, "countries": countries}


# ─── Dataset Stats ────────────────────────────────────────────────────────────
def print_dataset_stats(root: str, countries: Optional[List[str]] = None):
    """Print class distribution per split for analysis."""
    print("\n=== RDD2022 Dataset Statistics ===")
    for split in ("train", "val", "test"):
        ds = RDD2022Dataset(root=root, split=split, countries=countries)
        dist = ds.get_class_distribution()
        total = sum(dist.values())
        print(f"\n{split}: {len(ds)} images, {total} annotations")
        for cls, cnt in dist.items():
            pct = 100 * cnt / total if total > 0 else 0
            print(f"  {cls} ({CLASS_DESCRIPTIONS[cls]}): {cnt:,} ({pct:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/RDD_SPLIT")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_dataset_stats(args.root)
    else:
        # Quick smoke test
        ds = RDD2022Dataset(root=args.root, split=args.split,
                            transform=get_train_transforms())
        print(f"Dataset size: {len(ds)}")
        if len(ds) > 0:
            sample = ds[0]
            print(f"Image shape: {sample['image'].shape}")
            print(f"Boxes: {sample['boxes']}")
            print(f"Labels: {sample['labels']}")
