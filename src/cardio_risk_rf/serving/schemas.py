"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str


class FeaturesRequest(BaseModel):
    features: dict


class TabularResponse(BaseModel):
    pred: int
    proba: list[float]
