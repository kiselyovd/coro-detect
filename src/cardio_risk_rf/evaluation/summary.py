"""Merge main + baseline metrics into a single summary JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_summary(
    main_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    out_path: str | Path,
) -> Path:
    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    summary: dict[str, Any] = {
        "main_model": "LightGBM",
        "main_roc_auc": pct(main_metrics["roc_auc"]),
        "main_pr_auc": pct(main_metrics["pr_auc"]),
        "main_f1": pct(main_metrics["f1"]),
        "main_brier": round(main_metrics["brier"], 4),
        "baseline_model": "RandomForest",
        "baseline_roc_auc": pct(baseline_metrics["roc_auc"]),
        "baseline_pr_auc": pct(baseline_metrics["pr_auc"]),
        "baseline_f1": pct(baseline_metrics["f1"]),
        "baseline_brier": round(baseline_metrics["brier"], 4),
        "test_size": main_metrics["n"],
        "positive_rate": f"{main_metrics['positive_rate'] * 100:.1f}%",
        "threshold": main_metrics["threshold"],
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return p
