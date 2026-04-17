"""Binary classification metrics for the tabular pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return a flat dict with ROC-AUC / PR-AUC / F1 / Brier for reporting."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    if y_true.shape != probs.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {probs.shape}")
    if y_true.size == 0:
        raise ValueError("empty y_true")

    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "f1": float(f1_score(y_true, preds)),
        "brier": float(brier_score_loss(y_true, probs)),
        "threshold": float(threshold),
        "n": int(y_true.size),
        "positive_rate": float(y_true.mean()),
    }
