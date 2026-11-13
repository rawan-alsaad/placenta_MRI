"""Patient-level summary statistics (Tables 1 and 2 of the paper)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, wilcoxon

from .config import BTFE, CONDITIONS, SSH_TSE


def bca_mean_ci(values, n_resamples: int = 10_000, seed: int = 2026) -> tuple[float, float, float]:
    """Mean with a 95% bias-corrected and accelerated (BCa) bootstrap CI."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    result = bootstrap((x,), np.mean, n_resamples=n_resamples, confidence_level=0.95,
                       method="BCa", rng=np.random.default_rng(seed))
    ci = result.confidence_interval
    return float(x.mean()), float(ci.low), float(ci.high)


def _dice(results: pd.DataFrame, condition: str) -> pd.Series:
    """Patient-level Dice of one condition, indexed by subject."""
    return results.loc[results["condition"] == condition].set_index("subject_id")["dice"]


def table1_dice(results: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    """Mean patient-level Dice with BCa 95% CI for each training -> test condition."""
    rows = []
    for label, _, _ in CONDITIONS:
        values = _dice(results, label)
        mean, low, high = bca_mean_ci(values, n_resamples)
        rows.append({"condition": label, "n_patients": len(values),
                     "mean_dice": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def _paired(a: pd.Series, b: pd.Series, n_resamples: int) -> dict:
    """Mean paired difference a - b with BCa CI, n(a > b), and two-sided Wilcoxon p."""
    a, b = a.align(b, join="inner")
    difference = a - b
    mean, low, high = bca_mean_ci(difference, n_resamples)
    return {"mean": mean, "ci_low": low, "ci_high": high,
            "n_positive": f"{int((difference > 0).sum())}/{len(difference)}",
            "p": float(wilcoxon(a, b, alternative="two-sided").pvalue)}


def table2_cross_sequence(results: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    """Cross-sequence Dice loss, recovery with pooled training, and pooled vs specific.

    * loss     = sequence-specific (in-domain) Dice - cross-sequence Dice
    * recovery = pooled-training Dice - cross-sequence Dice
    * pooled_minus_specific = pooled-training Dice - sequence-specific Dice
    """
    targets = {  # target: (in-domain, cross-sequence, pooled) condition labels
        BTFE: ("BTFE -> BTFE", "ssh-TSE -> BTFE", "Pooled -> BTFE"),
        SSH_TSE: ("ssh-TSE -> ssh-TSE", "BTFE -> ssh-TSE", "Pooled -> ssh-TSE"),
    }
    rows = []
    for target, (in_domain, cross, pooled) in targets.items():
        specific, transfer, joint = (_dice(results, c) for c in (in_domain, cross, pooled))
        row = {"target_sequence": target}
        for name, (a, b) in {"loss": (specific, transfer),
                             "recovery": (joint, transfer),
                             "pooled_minus_specific": (joint, specific)}.items():
            row.update({f"{name}_{key}": value
                        for key, value in _paired(a, b, n_resamples).items()})
        rows.append(row)
    return pd.DataFrame(rows)
