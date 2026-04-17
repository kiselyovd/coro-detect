"""Inference smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cardio_risk_rf.inference.predict import predict
from cardio_risk_rf.models import build_pipeline


def test_predict_returns_shape():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] > 0).astype(int)
    pipe = build_pipeline("lgbm", n_estimators=10).fit(X, y)
    result = predict(pipe, dict(X.iloc[0]))
    assert "pred" in result
    assert "proba" in result
