"""FastAPI routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile

from .. import __version__
from ..inference.predict import predict
from .dependencies import get_model
from .errors import InferenceError
from .schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_model()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(
        status="ok" if loaded else "degraded", model_loaded=loaded, version=__version__,
    )


from .schemas import FeaturesRequest, TabularResponse


@router.post("/predict", response_model=TabularResponse)
def predict_endpoint(req: FeaturesRequest, model=Depends(get_model)) -> TabularResponse:
    result = predict(model, req.features)
    return TabularResponse(**result)
