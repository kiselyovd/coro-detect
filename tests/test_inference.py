"""joblib round-trip + null-field handling for both models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cardio_risk_rf.data.cardio import FEATURES
from cardio_risk_rf.models.factory import build_baseline, build_main


def _fit_and_save(builder, path: Path) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f: rng.normal(size=200) for f in FEATURES})
    y = pd.Series((rng.random(200) < 0.2).astype(int))
    mask = rng.random(X.shape) < 0.05
    X = X.mask(mask)
    model = builder()
    model.fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def test_main_joblib_roundtrip_with_nulls(tmp_path: Path) -> None:
    path = tmp_path / "main.joblib"
    _fit_and_save(lambda: build_main(scale_pos_weight=1.0, random_state=42), path)
    loaded = joblib.load(path)
    row = pd.DataFrame([{f: (None if i % 3 == 0 else 0.1) for i, f in enumerate(FEATURES)}]).astype(
        float
    )
    probs = loaded.predict_proba(row)
    assert probs.shape == (1, 2)
    assert np.isfinite(probs).all()


def test_baseline_joblib_roundtrip_with_nulls(tmp_path: Path) -> None:
    path = tmp_path / "baseline.joblib"
    _fit_and_save(lambda: build_baseline(random_state=42), path)
    loaded = joblib.load(path)
    row = pd.DataFrame([{f: (None if i % 3 == 0 else 0.1) for i, f in enumerate(FEATURES)}]).astype(
        float
    )
    probs = loaded.predict_proba(row)
    assert probs.shape == (1, 2)
    assert np.isfinite(probs).all()
