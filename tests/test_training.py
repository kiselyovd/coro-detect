"""Training smoke: main + baseline converge on toy data and produce joblib artefacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cardio_risk_rf.data.framingham import FEATURES, TARGET
from cardio_risk_rf.training.train import train_baseline, train_main


def _toy_split(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    mask = rng.random(X.shape) < 0.08
    X = X.mask(mask)
    y = pd.Series((rng.random(n) < 0.2).astype(int), name=TARGET)
    full = pd.concat([X, y], axis=1)
    train, rest = train_test_split(full, test_size=0.3, stratify=full[TARGET], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=seed)
    return train, val, test


def test_train_main_writes_joblib(tmp_path: Path) -> None:
    train, val, _ = _toy_split()
    out = tmp_path / "main.joblib"
    train_main(
        train_df=train,
        val_df=val,
        out_path=out,
        optuna_trials=3,
        seed=42,
    )
    assert out.exists()


def test_train_baseline_writes_joblib(tmp_path: Path) -> None:
    train, val, _ = _toy_split()
    out = tmp_path / "baseline.joblib"
    train_baseline(
        train_df=train,
        val_df=val,
        out_path=out,
        cv_folds=3,
        seed=42,
    )
    assert out.exists()
