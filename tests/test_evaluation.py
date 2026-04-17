"""Metrics shape + calibration plot generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cardio_risk_rf.evaluation.calibration import save_calibration_plot
from cardio_risk_rf.evaluation.metrics import compute_metrics


def test_compute_metrics_keys() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.2, 0.6, 0.05, 0.55])
    m = compute_metrics(y, probs, threshold=0.5)
    for key in ("roc_auc", "pr_auc", "f1", "brier", "threshold", "n", "positive_rate"):
        assert key in m
    assert 0.0 < m["roc_auc"] <= 1.0
    assert m["n"] == len(y)


def test_save_calibration_plot(tmp_path: Path) -> None:
    y = np.random.default_rng(0).integers(0, 2, 200)
    probs = np.random.default_rng(0).random(200)
    out = tmp_path / "calib.png"
    save_calibration_plot(y, probs, out, bins=10)
    assert out.exists()
    assert out.stat().st_size > 500
