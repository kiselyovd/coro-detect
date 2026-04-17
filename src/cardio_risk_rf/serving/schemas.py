"""Pydantic request/response schemas for /predict."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    male: int | None = Field(None, ge=0, le=1)
    age: int | None = Field(None, ge=0, le=120)
    education: float | None = Field(None, ge=0)
    currentSmoker: int | None = Field(None, ge=0, le=1)
    cigsPerDay: float | None = Field(None, ge=0)
    BPMeds: float | None = Field(None, ge=0, le=1)
    prevalentStroke: int | None = Field(None, ge=0, le=1)
    prevalentHyp: int | None = Field(None, ge=0, le=1)
    diabetes: int | None = Field(None, ge=0, le=1)
    totChol: float | None = Field(None, ge=0)
    sysBP: float | None = Field(None, ge=0)
    diaBP: float | None = Field(None, ge=0)
    BMI: float | None = Field(None, ge=0)
    heartRate: float | None = Field(None, ge=0)
    glucose: float | None = Field(None, ge=0)


class ShapEntry(BaseModel):
    feature: str
    value: float | int | None
    shap: float


class PredictionResponse(BaseModel):
    probability: float
    cls: int = Field(..., alias="class")
    threshold: float
    shap_top5: list[ShapEntry]
    model_version: str
    model_name: str
    request_id: str

    class Config:
        populate_by_name = True
