"""
models/rtdetr_model.py
RT-DETRv2 Model Wrapper for Road Damage Detection
  - Transformer-based detection (genuine architectural contrast to YOLOv11)
  - Bipartite Hungarian matching loss
  - Multi-scale deformable attention

Reference: Lv et al. (2024). RT-DETRv2: Improved Baseline with Bag-of-Freebies
for Real-Time Detection Transformer. arXiv:2407.17140

CMP 295 SJSU | Road Damage Detection
"""

import csv
import json
import torch
import torch.nn as nn
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict
from ultralytics import RTDETR


# ─── RT-DETRv2 Wrapper ────────────────────────────────────────────────────────
class RTDETRv2RoadDamage:
    """
    RT-DETRv2 wrapper for RDD2022 road damage detection.

    Key architectural differences vs YOLOv11:
      - Transformer decoder with deformable attention (not CNN-only)
      - Hungarian bipartite matching (no NMS needed at inference)
      - Flexible multi-scale feature fusion (AIFI + CCFM)
      - 300 object queries instead of dense anchors

    This provides a genuine architectural comparison for our ablation study.

    Usage:
        model = RTDETRv2RoadDamage(variant='rtdetr-l')
        model.train(config)
        results = model.predict('image.jpg')
    """

    def __init__(
        self,
        variant: str = "rtdetr-l",
        weights: Optional[str] = None,
        num_classes: int = 4,
        device: str = "cuda",
    ):
        self.variant = variant
        self.num_classes = num_classes
        self.device = device

        weight_path = weights or f"{variant}.pt"
        self.model = RTDETR(weight_path)

        print(f"[RT-DETRv2] Loaded {variant} | Classes: {num_classes}")
        print(f"[RT-DETRv2] Architecture: Transformer decoder | No NMS | Hungarian matching")

    def train(self, config: dict) -> dict:
        """Fine-tune RT-DETRv2 on RDD2022."""
        dataset_yaml = self._build_dataset_yaml(config)

        resuming = config["training"].get("resume", False)
        checkpoint_dir = config.get("output", {}).get("checkpoint_dir")
        project = checkpoint_dir if checkpoint_dir else config["logging"]["save_dir"]
        model_name = config["logging"]["name"]

        train_args = dict(
            data=dataset_yaml,
            epochs=config["training"]["epochs"],
            batch=config["training"]["batch_size"],
            imgsz=config["dataset"]["image_size"],
            device=self.device,
            workers=config["training"]["num_workers"],
            amp=True,
            rect=True,
            cache=config["dataset"].get("cache", False),
            patience=config["training"].get("patience", 30),
            optimizer=config["optimizer"]["name"],
            lr0=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"],
            warmup_epochs=config["scheduler"]["warmup_epochs"],
            lrf=config["optimizer"].get("backbone_lr_multiplier", 0.1),
            # Augmentation (minimal for RT-DETR — no mosaic)
            hsv_h=config["augmentation"]["hsv_h"],
            hsv_s=config["augmentation"]["hsv_s"],
            hsv_v=config["augmentation"]["hsv_v"],
            degrees=config["augmentation"]["degrees"],
            translate=config["augmentation"]["translate"],
            scale=config["augmentation"]["scale"],
            fliplr=config["augmentation"]["fliplr"],
            mosaic=config["augmentation"].get("mosaic", 0.0),
            # Logging
            project=project,
            name=model_name,
            save_period=config["output"]["save_period"],
            exist_ok=True,
            verbose=True,
        )

        if resuming:
            train_args["resume"] = True

        results = self.model.train(**train_args)
        self._save_training_history(config, project, model_name, resuming)
        return results

    def evaluate(self, data_yaml: str, split: str = "val") -> dict:
        """Run evaluation. Returns mAP metrics."""
        results = self.model.val(data=data_yaml, split=split)
        return {
            "mAP50": results.box.map50,
            "mAP50_95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "per_class_mAP50": results.box.ap50.tolist(),
        }

    def predict(self, source, conf: float = 0.3, iou: float = 0.5, max_det: int = 300):
        """Run inference. RT-DETR uses higher default conf (0.3) due to no NMS."""
        return self.model.predict(source=source, conf=conf, iou=iou, max_det=max_det, verbose=False)

    def _build_dataset_yaml(self, config: dict) -> str:
        import yaml
        root = config["dataset"]["root"]
        train_countries = config["dataset"]["train_countries"]
        val_countries = config["dataset"]["val_countries"]

        dataset_cfg = {
            "path": str(Path(root).resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": self.num_classes,
            "names": ["D00", "D10", "D20", "D40", "D44"],
        }

        import tempfile
        yaml_path = Path(tempfile.gettempdir()) / f"rdd2022_{self.variant}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dataset_cfg, f)
        return str(yaml_path)

    def _save_training_history(self, config: dict, project: str, model_name: str, resumed: bool):
        """Reads results.csv and appends this run's per-epoch metrics to training_history.json."""
        run_dir = Path(project) / model_name
        csv_path = run_dir / "results.csv"
        history_path = run_dir / "training_history.json"

        if not csv_path.exists():
            print(f"[RT-DETRv2] results.csv not found at {csv_path}, skipping history save.")
            return

        per_epoch = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stripped = {k.strip(): v.strip() for k, v in row.items()}
                per_epoch.append({
                    "epoch":        int(float(stripped.get("epoch", 0))),
                    "box_loss":     float(stripped.get("train/box_loss", 0)),
                    "cls_loss":     float(stripped.get("train/cls_loss", 0)),
                    "dfl_loss":     float(stripped.get("train/dfl_loss", 0)),
                    "val_box_loss": float(stripped.get("val/box_loss", 0)),
                    "val_cls_loss": float(stripped.get("val/cls_loss", 0)),
                    "val_dfl_loss": float(stripped.get("val/dfl_loss", 0)),
                    "precision":    float(stripped.get("metrics/precision(B)", 0)),
                    "recall":       float(stripped.get("metrics/recall(B)", 0)),
                    "mAP50":        float(stripped.get("metrics/mAP50(B)", 0)),
                    "mAP50_95":     float(stripped.get("metrics/mAP50-95(B)", 0)),
                })

        if not per_epoch:
            return

        best_mAP50 = max(e["mAP50"] for e in per_epoch)
        run_entry = {
            "run_id":         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "resumed":        resumed,
            "model":          self.variant,
            "start_epoch":    per_epoch[0]["epoch"],
            "end_epoch":      per_epoch[-1]["epoch"],
            "epochs_trained": len(per_epoch),
            "best_mAP50":     best_mAP50,
            "final_mAP50":    per_epoch[-1]["mAP50"],
            "config": {
                "image_size": config["dataset"]["image_size"],
                "batch_size": config["training"]["batch_size"],
                "box_loss":   config["loss"]["box"],
            },
            "per_epoch": per_epoch,
        }

        history = {"model_name": model_name, "runs": []}
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

        history["runs"].append(run_entry)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"[RT-DETRv2] Training history saved → {history_path}")
        print(f"[RT-DETRv2] Epochs this run: {run_entry['start_epoch']}–{run_entry['end_epoch']} | Best mAP50: {best_mAP50:.4f}")

    def load_weights(self, path: str):
        self.model = RTDETR(path)
        print(f"[RT-DETRv2] Loaded weights from {path}")


# ─── Hungarian Matching Loss ──────────────────────────────────────────────────
class HungarianMatcher(nn.Module):
    """
    Bipartite matching between predicted and ground truth boxes.
    Used internally by RT-DETR (Ultralytics handles this, provided here
    for reference and custom training loops).

    Cost = lambda_cls * cost_cls + lambda_box * cost_box + lambda_giou * cost_giou
    """

    def __init__(
        self,
        cost_cls: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,   # [B, num_queries, num_classes]
        pred_boxes: torch.Tensor,    # [B, num_queries, 4] cxcywh normalized
        target_labels: List[torch.Tensor],  # list of [N_i] per image
        target_boxes: List[torch.Tensor],   # list of [N_i, 4] per image
    ):
        """
        Returns: List of (row_indices, col_indices) tuples per image.
        Row = prediction index, Col = GT index.
        """
        from scipy.optimize import linear_sum_assignment
        import torch.nn.functional as F

        B, Q, _ = pred_logits.shape
        indices = []

        for b in range(B):
            if len(target_labels[b]) == 0:
                indices.append((torch.tensor([]), torch.tensor([])))
                continue

            out_prob = pred_logits[b].sigmoid()   # [Q, C]
            out_box = pred_boxes[b]               # [Q, 4]
            tgt_lbl = target_labels[b]            # [N]
            tgt_box = target_boxes[b]             # [N, 4]

            # Focal classification cost
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            neg_cost_cls = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_cls = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_cls = pos_cost_cls[:, tgt_lbl] - neg_cost_cls[:, tgt_lbl]

            # L1 box cost
            cost_bbox = torch.cdist(out_box, tgt_box, p=1)

            # GIoU cost
            cost_giou = -self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(out_box),
                self._box_cxcywh_to_xyxy(tgt_box),
            )

            # Combined cost matrix [Q, N]
            C = (
                self.cost_cls * cost_cls
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            row_idx, col_idx = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.tensor(row_idx, dtype=torch.long),
                torch.tensor(col_idx, dtype=torch.long),
            ))

        return indices

    def _box_cxcywh_to_xyxy(self, boxes):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    def _generalized_box_iou(self, boxes1, boxes2):
        """Compute GIoU for all pairs. Returns [N, M] matrix."""
        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter

        enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
        enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
        enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
        enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
        enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

        iou = inter / union.clamp(min=1e-6)
        giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
        return giou


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Hungarian Matcher Test ===\n")
    matcher = HungarianMatcher()

    pred_logits = torch.randn(2, 300, 4)   # B=2, 300 queries, 4 classes
    pred_boxes  = torch.rand(2, 300, 4)    # normalized cxcywh

    target_labels = [torch.tensor([0, 2, 3]), torch.tensor([1])]
    target_boxes  = [torch.rand(3, 4), torch.rand(1, 4)]

    indices = matcher(pred_logits, pred_boxes, target_labels, target_boxes)
    for i, (r, c) in enumerate(indices):
        print(f"Image {i}: matched {len(r)} pairs | row={r.tolist()} col={c.tolist()}")

    print("\nHungarian matcher OK ✓")
