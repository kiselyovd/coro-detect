"""Pydantic request/response schemas for /predict."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Field names intentionally preserve Framingham Heart Study column casing
# (currentSmoker, cigsPerDay, BPMeds, ...) so that requests can be serialized
# directly into a pandas DataFrame aligned with the training schema.


class PatientFeatures(BaseModel):
    """Single-patient feature payload; fields mirror Framingham column names."""

    male: int | None = Field(None, ge=0, le=1)
    age: int | None = Field(None, ge=0, le=120)
    education: float | None = Field(None, ge=0)
    currentSmoker: int | None = Field(None, ge=0, le=1)  # noqa: N815 — Framingham column name
    cigsPerDay: float | None = Field(None, ge=0)  # noqa: N815 — Framingham column name
    BPMeds: float | None = Field(None, ge=0, le=1)
    prevalentStroke: int | None = Field(None, ge=0, le=1)  # noqa: N815 — Framingham column name
    prevalentHyp: int | None = Field(None, ge=0, le=1)  # noqa: N815 — Framingham column name
    diabetes: int | None = Field(None, ge=0, le=1)
    totChol: float | None = Field(None, ge=0)  # noqa: N815 — Framingham column name
    sysBP: float | None = Field(None, ge=0)  # noqa: N815 — Framingham column name
    diaBP: float | None = Field(None, ge=0)  # noqa: N815 — Framingham column name
    BMI: float | None = Field(None, ge=0)
    heartRate: float | None = Field(None, ge=0)  # noqa: N815 — Framingham column name
    glucose: float | None = Field(None, ge=0)


class ShapEntry(BaseModel):
    """Single SHAP contribution for one feature."""

    feature: str
    value: float | int | None
    shap: float


class PredictionResponse(BaseModel):
    """Response schema for /predict."""

    probability: float
    cls: int = Field(..., alias="class")
    threshold: float
    shap_top5: list[ShapEntry]
    model_version: str
    model_name: str
    request_id: str

    class Config:
        """Pydantic configuration — allow populating `cls` by field or alias."""

        populate_by_name = True
