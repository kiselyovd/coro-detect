"""Reliability diagram for probabilistic binary predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def save_calibration_plot(
    y_true: np.ndarray,
    probs: np.ndarray,
    out_path: str | Path,
    *,
    bins: int = 10,
    title: str = "Calibration curve",
) -> None:
    """Write a reliability-diagram PNG."""
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
