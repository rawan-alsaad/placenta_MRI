"""Inference and patient-level aggregation on the held-out test cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import make_loader
from .metrics import dice, hd95, relative_volume_difference


@torch.no_grad()
def predict(model, df: pd.DataFrame, cfg: Config, device: torch.device):
    """Binary predictions (probability >= 0.5) and reference masks on the 256x256 grid."""
    model.eval()
    predictions = np.zeros((len(df), cfg.image_size, cfg.image_size), dtype=bool)
    targets = np.zeros_like(predictions)
    for batch in make_loader(df, cfg, train=False):
        probabilities = torch.sigmoid(model(batch["image"].to(device))).float().cpu().numpy()
        index = batch["index"].numpy()
        predictions[index] = probabilities[:, 0] >= 0.5
        targets[index] = batch["mask"].numpy()[:, 0] >= 0.5
    return predictions, targets


def patient_metrics(df: pd.DataFrame, predictions: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """Aggregate slices per patient.

    * ``dice``: primary outcome; Dice over all pixels of all slices of the patient.
    * ``rvd_percent``: relative volume (pixel-count) difference over all slices.
    * ``mean_hd95_px``: mean of finite slice HD95 values (pixels);
      ``empty_predictions`` counts slices with no predicted foreground.
    """
    rows = []
    for subject, group in df.groupby("subject_id"):
        idx = group.index.to_numpy()
        pred, true = predictions[idx], targets[idx]
        slice_hd95 = np.array([hd95(p, t) for p, t in zip(pred, true)])
        finite = slice_hd95[np.isfinite(slice_hd95)]
        rows.append({
            "subject_id": subject,
            "n_slices": len(idx),
            "dice": dice(pred, true),
            "rvd_percent": relative_volume_difference(pred, true),
            "mean_hd95_px": float(finite.mean()) if len(finite) else np.nan,
            "empty_predictions": int((~pred.any(axis=(1, 2))).sum()),
        })
    return pd.DataFrame(rows)
