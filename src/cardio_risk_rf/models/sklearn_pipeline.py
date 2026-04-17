"""scikit-learn pipeline builder."""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(model_name: str = "lgbm", **model_params: Any) -> Pipeline:
    """Build an sklearn Pipeline by model name."""
    if model_name == "lgbm":
        clf = lgb.LGBMClassifier(random_state=42, **model_params)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(random_state=42, **model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
