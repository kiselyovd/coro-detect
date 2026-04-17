"""Model smoke tests (forward pass, output shape)."""

from __future__ import annotations

from cardio_risk_rf.models import build_pipeline


def test_lgbm_pipeline_builds():
    pipe = build_pipeline("lgbm", n_estimators=10)
    assert pipe.steps[-1][0] == "clf"
