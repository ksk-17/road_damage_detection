"""
training/losses.py
Custom Loss Functions for Road Damage Detection
  - Focal Loss: handles severe class imbalance (normal road >> cracks)
  - WIoU (Wise-IoU): dynamic focusing for thin, elongated crack bounding boxes
  - Combined Focal + WIoU for box regression + classification

Reference:
  Tong et al. (2023). Wise-IoU: Bounding Box Regression Loss with Dynamic
  Focusing Mechanism. arXiv:2301.10051
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for classification — addresses class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    High gamma → more focus on hard/misclassified examples.
    Alpha balances positive vs. negative class.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Raw logits, shape [N, C] or [N]
            targets: Ground truth class indices, shape [N]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─── IoU Utilities ────────────────────────────────────────────────────────────
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes (xyxy format).
    Returns IoU matrix of shape [N, M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def bbox_iou_paired(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    """
    Compute IoU and related metrics for paired (pred, target) boxes.
    Both tensors: [N, 4] in xyxy format.

    Returns: (iou, union, enclosing_diag_sq, center_dist_sq, wh_ratio_factor)
    """
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    pred_area = (px2 - px1) * (py2 - py1)
    tgt_area = (tx2 - tx1) * (ty2 - ty1)
    union = pred_area + tgt_area - inter + eps
    iou = inter / union

    # Enclosing box (for CIoU / WIoU)
    enc_x1 = torch.min(px1, tx1)
    enc_y1 = torch.min(py1, ty1)
    enc_x2 = torch.max(px2, tx2)
    enc_y2 = torch.max(py2, ty2)
    enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Center distance
    pc_x, pc_y = (px1 + px2) / 2, (py1 + py2) / 2
    tc_x, tc_y = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    center_dist_sq = (pc_x - tc_x) ** 2 + (pc_y - tc_y) ** 2

    return iou, union, enc_diag_sq, center_dist_sq


# ─── CIoU Loss (Baseline) ─────────────────────────────────────────────────────
class CIoULoss(nn.Module):
    """
    Complete IoU Loss — baseline box regression loss.
    CIoU = IoU - (center_dist / enc_diag) - aspect_ratio_penalty
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.sum() * 0

        iou, _, enc_diag_sq, center_dist_sq = bbox_iou_paired(pred, target)

        pw = pred[:, 2] - pred[:, 0]
        ph = pred[:, 3] - pred[:, 1]
        tw = target[:, 2] - target[:, 0]
        th = target[:, 3] - target[:, 1]

        v = (4 / (torch.pi ** 2)) * (torch.atan(tw / (th + 1e-6)) - torch.atan(pw / (ph + 1e-6))) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-6)

        ciou = iou - center_dist_sq / enc_diag_sq - alpha * v
        loss = 1 - ciou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─── WIoU Loss (Improvement 2) ───────────────────────────────────────────────
class WIoULoss(nn.Module):
    """
    Wise-IoU Loss — dynamic focusing mechanism.

    Unlike CIoU which uses a fixed aspect ratio penalty, WIoU uses a
    momentum-based 'wise factor' (beta) that dynamically focuses the loss
    on outlier/anchor boxes, beneficial for thin elongated crack shapes.

    WIoU_v3 = r * IoU_loss
    where r = exp( (x1-x2)^2 / (X^2) + (y1-y2)^2 / (Y^2) ) / momentum

    Reference: Tong et al. arXiv:2301.10051
    """

    def __init__(self, momentum: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.momentum = momentum
        self.reduction = reduction
        # Running mean IoU for dynamic focusing
        self.register_buffer("running_mean_iou", torch.tensor(0.5))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.sum() * 0

        iou, _, enc_diag_sq, center_dist_sq = bbox_iou_paired(pred, target)

        # Update running mean IoU (EMA)
        with torch.no_grad():
            mean_iou = iou.detach().mean()
            self.running_mean_iou = (
                self.momentum * self.running_mean_iou + (1 - self.momentum) * mean_iou
            )

        # Wise focusing factor: boxes with IoU << running mean get higher weight
        # This focuses training on difficult/outlier boxes (hairline cracks)
        beta = iou.detach() / self.running_mean_iou.detach()

        # Base IoU loss with distance penalty
        iou_loss = 1 - iou + center_dist_sq / enc_diag_sq

        # Apply wise focusing weight
        r = torch.exp(beta - 1)    # r > 1 for easy boxes, < 1 for hard boxes
        loss = iou_loss / r        # Amplify hard box loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─── Combined Detection Loss ──────────────────────────────────────────────────
class RoadDamageDetectionLoss(nn.Module):
    """
    Combined loss for road damage detection:
      - Box regression: WIoU (or CIoU for baseline)
      - Classification: Focal Loss
      - Objectness: BCE

    Handles severe class imbalance (normal >> crack pixels) and
    thin elongated bounding boxes (D00, D10 crack types).
    """

    def __init__(
        self,
        num_classes: int = 4,
        box_loss_type: str = "wiou",      # "ciou" | "wiou" | "focal_wiou"
        cls_loss_type: str = "focal",     # "bce" | "focal"
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        obj_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        wiou_momentum: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight

        # Box regression loss
        if box_loss_type in ("wiou", "focal_wiou"):
            self.box_loss = WIoULoss(momentum=wiou_momentum)
        else:
            self.box_loss = CIoULoss()

        # Classification loss
        if cls_loss_type == "focal":
            self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.cls_loss = nn.BCEWithLogitsLoss()

        # Objectness loss (always BCE)
        self.obj_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_cls: torch.Tensor,
        pred_obj: torch.Tensor,
        target_boxes: torch.Tensor,
        target_cls: torch.Tensor,
        target_obj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_boxes:   [N, 4] predicted boxes (xyxy)
            pred_cls:     [N, num_classes] class logits
            pred_obj:     [N, 1] objectness logits
            target_boxes: [N, 4] ground truth boxes
            target_cls:   [N, num_classes] one-hot class targets
            target_obj:   [N, 1] objectness targets
        """
        box_l = self.box_loss(pred_boxes, target_boxes)
        cls_l = self.cls_loss(pred_cls, target_cls.float())
        obj_l = self.obj_loss(pred_obj, target_obj.float())

        total = self.box_weight * box_l + self.cls_weight * cls_l + self.obj_weight * obj_l

        return {
            "total": total,
            "box": box_l,
            "cls": cls_l,
            "obj": obj_l,
        }


# ─── Loss Factory ─────────────────────────────────────────────────────────────
def build_loss(config: dict) -> RoadDamageDetectionLoss:
    """Build loss function from config dict."""
    return RoadDamageDetectionLoss(
        num_classes=config.get("num_classes", 4),
        box_loss_type=config.get("box", "wiou"),
        cls_loss_type=config.get("cls", "focal"),
        box_weight=config.get("box_weight", 7.5),
        cls_weight=config.get("cls_weight", 0.5),
        obj_weight=config.get("obj_weight", 1.0),
        focal_alpha=config.get("focal_alpha", 0.25),
        focal_gamma=config.get("focal_gamma", 2.0),
        wiou_momentum=config.get("wiou_momentum", 0.5),
    )


# ─── Quick Tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Loss Function Smoke Tests ===\n")
    N = 8
    pred_boxes = torch.rand(N, 4).abs()
    pred_boxes[:, 2:] += pred_boxes[:, :2]  # xmax > xmin
    tgt_boxes = torch.rand(N, 4).abs()
    tgt_boxes[:, 2:] += tgt_boxes[:, :2]

    # CIoU
    ciou = CIoULoss()
    print(f"CIoU loss:  {ciou(pred_boxes, tgt_boxes):.4f}")

    # WIoU
    wiou = WIoULoss()
    print(f"WIoU loss:  {wiou(pred_boxes, tgt_boxes):.4f}")

    # Focal
    logits = torch.randn(N, 4)
    targets = torch.zeros(N, 4)
    targets[torch.arange(N), torch.randint(0, 4, (N,))] = 1
    fl = FocalLoss()
    print(f"Focal loss: {fl(logits, targets):.4f}")

    print("\nAll loss functions OK ✓")
