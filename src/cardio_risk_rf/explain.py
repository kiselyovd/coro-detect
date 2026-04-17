"""SHAP wrappers for the LightGBM main model and RF baseline.

TreeExplainer supports both natively. For global reports we pass a sample of
val or test; for per-instance serving we pass a single row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _unwrap(model: Any) -> Any:
    """Accept either a bundle dict or a sklearn Pipeline and return the Pipeline."""
    if isinstance(model, dict) and "model" in model:
        return model["model"]
    return model


def _extract_tree_and_frame(model: Any, x: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """Return the fitted estimator + the preprocessor-transformed frame SHAP sees."""
    pipe = _unwrap(model)
    # Feed preprocessor-transformed data to SHAP so feature names align.
    frame = x.copy()
    steps = list(pipe.named_steps.items())
    for _name, step in steps[:-1]:
        transformed = step.transform(frame)
        frame = pd.DataFrame(transformed, columns=list(frame.columns), index=frame.index)
    estimator = steps[-1][1]
    return estimator, frame


def explain_global(
    model: Any,
    x: pd.DataFrame,
    *,
    out_png: str | Path,
    out_csv: str | Path,
) -> dict[str, Any]:
    estimator, frame = _extract_tree_and_frame(model, x)
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(frame)
    if isinstance(values, list):
        # Older SHAP returns a list [neg, pos] for binary classifiers
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        # Newer SHAP on RF binary: shape (n_samples, n_features, n_classes)
        values = values[..., 1]
    imp = np.abs(values).mean(axis=0)
    order = np.argsort(imp)[::-1]

    importance_df = pd.DataFrame(
        {"feature": np.asarray(list(frame.columns))[order], "mean_abs_shap": imp[order]}
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_csv, index=False)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    shap.summary_plot(values, frame, show=False, plot_type="dot", max_display=15)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    return {
        "top_features": importance_df["feature"].head(10).tolist(),
        "importance_csv": str(out_csv),
        "summary_png": str(out_png),
    }


def explain_instance(model: Any, row: pd.DataFrame) -> list[dict[str, Any]]:
    if len(row) != 1:
        raise ValueError("explain_instance expects a single-row DataFrame")
    estimator, frame = _extract_tree_and_frame(model, row)
    explainer = shap.TreeExplainer(estimator)
    vals = explainer.shap_values(frame)
    if isinstance(vals, list):
        # Older SHAP: list [neg, pos] for binary classifiers
        vals = vals[1]
    arr = np.asarray(vals)
    if arr.ndim == 3:
        # Newer SHAP on RF binary: shape (n_samples, n_features, n_classes) → take positive class
        arr = arr[..., 1]
    arr = arr.reshape(-1)
    order = np.argsort(np.abs(arr))[::-1][:5]
    results: list[dict[str, Any]] = []
    raw = row.iloc[0]
    for i in order:
        feat = str(frame.columns[i])
        v_raw = raw[feat]
        results.append(
            {
                "feature": feat,
                "value": None
                if pd.isna(v_raw)
                else (float(v_raw) if isinstance(v_raw, (int, float, np.floating)) else v_raw),
                "shap": float(arr[i]),
            }
        )
    return results
