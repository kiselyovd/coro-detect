"""Tests for Framingham loader + stratified split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cardio_risk_rf.data.framingham import FEATURES, TARGET, load_framingham, split_stratified


def _toy_df(n: int = 200, pos_rate: float = 0.15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "male": rng.integers(0, 2, n),
        "age": rng.integers(32, 80, n),
        "education": rng.integers(1, 5, n),
        "currentSmoker": rng.integers(0, 2, n),
        "cigsPerDay": rng.integers(0, 60, n).astype(float),
        "BPMeds": rng.integers(0, 2, n).astype(float),
        "prevalentStroke": rng.integers(0, 2, n),
        "prevalentHyp": rng.integers(0, 2, n),
        "diabetes": rng.integers(0, 2, n),
        "totChol": rng.uniform(100, 400, n),
        "sysBP": rng.uniform(90, 200, n),
        "diaBP": rng.uniform(50, 130, n),
        "BMI": rng.uniform(15, 45, n),
        "heartRate": rng.uniform(40, 120, n),
        "glucose": rng.uniform(50, 250, n),
        "TenYearCHD": (rng.random(n) < pos_rate).astype(int),
    }
    return pd.DataFrame(data)


def test_load_framingham_returns_expected_columns(tmp_path) -> None:
    df = _toy_df(100)
    path = tmp_path / "framingham.csv"
    df.to_csv(path, index=False)
    loaded = load_framingham(path)
    assert list(loaded.columns) == [*FEATURES, TARGET]
    assert len(loaded) == 100
    assert loaded[TARGET].dtype == np.int64 or loaded[TARGET].dtype == np.int32


def test_split_stratified_respects_ratios_and_positives() -> None:
    df = _toy_df(n=1000, pos_rate=0.15)
    train, val, test = split_stratified(df, seed=42)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert abs(len(train) / total - 0.70) < 0.01
    assert abs(len(val) / total - 0.15) < 0.01
    assert abs(len(test) / total - 0.15) < 0.01
    for part in (train, val, test):
        share = part[TARGET].mean()
        assert abs(share - 0.15) < 0.04, f"stratification broken: {share}"


def test_split_no_row_leak() -> None:
    df = _toy_df(n=500)
    df["row_id"] = range(len(df))
    train, val, test = split_stratified(df, seed=42)
    seen = set()
    for part in (train, val, test):
        for rid in part["row_id"].tolist():
            assert rid not in seen
            seen.add(rid)
    assert len(seen) == len(df)
