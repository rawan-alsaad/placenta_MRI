"""Loss and segmentation metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


# ---------------------------------------------------------------------------
# Training loss: binary cross-entropy + soft Dice
# ---------------------------------------------------------------------------
def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    soft_dice = (2 * intersection + eps) / (denominator + eps)
    return bce + (1 - soft_dice.mean())


# ---------------------------------------------------------------------------
# Evaluation metrics (binary numpy arrays; work for 2D slices or 3D stacks)
# ---------------------------------------------------------------------------
def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def relative_volume_difference(prediction: np.ndarray, target: np.ndarray) -> float:
    """Signed RVD in percent (positive = over-segmentation), from pixel counts."""
    predicted, true = float(prediction.sum()), float(target.sum())
    if true == 0:
        return 0.0 if predicted == 0 else float("inf")
    return 100.0 * (predicted - true) / true


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance in pixels (inf if one mask is empty)."""
    prediction, target = prediction.astype(bool), target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("inf")
    pred_surface = prediction ^ ndimage.binary_erosion(prediction)
    true_surface = target ^ ndimage.binary_erosion(target)
    distances = np.concatenate([
        ndimage.distance_transform_edt(~true_surface)[pred_surface],
        ndimage.distance_transform_edt(~pred_surface)[true_surface],
    ])
    return float(np.percentile(distances, 95))
