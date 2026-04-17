"""Training smoke — one-batch overfit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cardio_risk_rf.models import build_pipeline


def test_fit_predict_tiny():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] > 0).astype(int)
    pipe = build_pipeline("lgbm", n_estimators=10)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (20,)
