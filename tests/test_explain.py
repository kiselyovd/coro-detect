"""SHAP wrapper returns right shapes + top-5 ordering consistent with magnitude."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cardio_risk_rf.data.framingham import FEATURES
from cardio_risk_rf.explain import explain_global, explain_instance
from cardio_risk_rf.models.factory import build_main


def _fit_toy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f: rng.normal(size=200) for f in FEATURES})
    y = (X["age"] + X["sysBP"] + rng.normal(size=200) > 0).astype(int)
    model = build_main(scale_pos_weight=1.0, random_state=42)
    model.fit(X, y)
    return model, X


def test_explain_global_writes_png_and_csv(tmp_path: Path) -> None:
    model, X = _fit_toy()
    out_png = tmp_path / "shap_summary.png"
    out_csv = tmp_path / "shap_importance.csv"
    info = explain_global(model, X.iloc[:100], out_png=out_png, out_csv=out_csv)
    assert out_png.exists() and out_png.stat().st_size > 500
    assert out_csv.exists()
    assert set(info["top_features"]) <= set(FEATURES)


def test_explain_instance_returns_top_5() -> None:
    model, X = _fit_toy()
    row = X.iloc[0:1]
    out = explain_instance(model, row)
    assert len(out) == 5
    for item in out:
        assert set(item.keys()) == {"feature", "value", "shap"}
    shap_abs = [abs(x["shap"]) for x in out]
    assert shap_abs == sorted(shap_abs, reverse=True), "top-5 must be sorted by |SHAP| desc"
