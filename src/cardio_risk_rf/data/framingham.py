"""Framingham Heart Study loader + stratified split.

Dataset columns (as published on Kaggle):
    male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke,
    prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose,
    TenYearCHD (target).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES: list[str] = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]
TARGET: str = "TenYearCHD"


def load_framingham(csv_path: str | Path) -> pd.DataFrame:
    """Read the Framingham CSV and return it with a stable column order."""
    df = pd.read_csv(csv_path)
    missing = {TARGET, *FEATURES} - set(df.columns)
    if missing:
        raise ValueError(f"Framingham CSV missing columns: {sorted(missing)}")
    df = df[[*FEATURES, TARGET]].copy()
    df[TARGET] = df[TARGET].astype("int64")
    return df


def split_stratified(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) with stratification on the target column."""
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio out of range: {train_ratio}")
    if not 0 < val_ratio < 1 - train_ratio:
        raise ValueError(f"val_ratio out of range: {val_ratio}")
    test_ratio = 1.0 - train_ratio - val_ratio

    train_df, rest = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df[TARGET],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        rest,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=rest[TARGET],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
