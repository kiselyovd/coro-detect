"""FastAPI routes: /health and /predict (main + baseline by query param)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..data.cardio import FEATURES
from ..explain import explain_instance
from .schemas import PatientFeatures, PredictionResponse

router = APIRouter()

_MODELS: dict[str, Any] = {}
_MODEL_FILES: dict[str, str] = {
    "main": os.environ.get("CARDIO_MAIN_CKPT", "artifacts/main/cardio_risk_lgbm.joblib"),
    "baseline": os.environ.get("CARDIO_BASELINE_CKPT", "artifacts/baseline/cardio_risk_rf.joblib"),
}
_MODEL_NAMES: dict[str, str] = {
    "main": "cardio_risk_lgbm",
    "baseline": "cardio_risk_rf",
}
_MODEL_VERSION: str = os.environ.get("CARDIO_MODEL_VERSION", "v0.1.0")


def _load(tag: str) -> Any:
    if tag not in _MODELS:
        path = Path(_MODEL_FILES[tag])
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Checkpoint missing: {path}")
        bundle = joblib.load(path)
        _MODELS[tag] = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    return _MODELS[tag]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": _MODEL_VERSION}


@router.post("/predict", response_model=PredictionResponse, response_model_by_alias=True)
def predict(
    features: PatientFeatures,
    model: str = Query("main", pattern="^(main|baseline)$"),
) -> PredictionResponse:
    raw = features.model_dump()
    if all(v is None for v in raw.values()):
        raise HTTPException(status_code=422, detail="All features null; cannot score.")

    row = pd.DataFrame([{f: raw.get(f) for f in FEATURES}]).astype(float)
    pipe = _load(model)
    prob = float(pipe.predict_proba(row)[0, 1])
    threshold = float(os.environ.get("CARDIO_THRESHOLD", "0.5"))
    cls = int(prob >= threshold)
    top5 = explain_instance(pipe, row)
    return PredictionResponse(
        probability=prob,
        cls=cls,  # type: ignore[call-arg]  # populated via alias="class"; revisit in backport
        threshold=threshold,
        shap_top5=top5,  # type: ignore[arg-type]
        model_version=_MODEL_VERSION,
        model_name=_MODEL_NAMES[model],
        request_id=uuid.uuid4().hex[:12],
    )
