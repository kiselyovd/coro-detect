"""FastAPI smoke: /health, /predict, /predict?model=baseline, bad payload."""

from __future__ import annotations

from fastapi.testclient import TestClient

from cardio_risk_rf.serving.main import app


def _payload() -> dict:
    return {
        "male": 1,
        "age": 58,
        "education": 3.0,
        "currentSmoker": 1,
        "cigsPerDay": 20.0,
        "BPMeds": 0.0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "diabetes": 0,
        "totChol": 275.0,
        "sysBP": 145.0,
        "diaBP": 95.0,
        "BMI": 29.4,
        "heartRate": 78.0,
        "glucose": 82.0,
    }


def test_health_returns_200() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_main_returns_contract() -> None:
    client = TestClient(app)
    r = client.post("/predict", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["class"] in (0, 1)
    assert body["threshold"] == 0.5
    assert len(body["shap_top5"]) == 5
    assert body["model_name"] == "cardio_risk_lgbm"


def test_predict_baseline_query_param() -> None:
    client = TestClient(app)
    r = client.post("/predict?model=baseline", json=_payload())
    assert r.status_code == 200
    assert r.json()["model_name"] == "cardio_risk_rf"


def test_predict_all_null_returns_422() -> None:
    client = TestClient(app)
    empty = {k: None for k in _payload().keys()}
    r = client.post("/predict", json=empty)
    assert r.status_code == 422
