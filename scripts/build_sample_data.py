"""Write 5 representative patients to data/sample/{sample.csv,sample.json}
and a single widget example to data/widget/sample_patient.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cardio_risk_rf.data.cardio import FEATURES, TARGET


def main() -> None:
    test = pd.read_parquet("data/processed/test.parquet")
    pos = test[test[TARGET] == 1].head(3)
    neg = test[test[TARGET] == 0].head(2)
    sample = pd.concat([pos, neg], ignore_index=True)[[*FEATURES, TARGET]]

    Path("data/sample").mkdir(parents=True, exist_ok=True)
    sample.to_csv("data/sample/sample.csv", index=False)
    sample.to_json("data/sample/sample.json", orient="records", indent=2)

    Path("data/widget").mkdir(parents=True, exist_ok=True)
    widget = {
        f: (None if pd.isna(sample.iloc[0][f]) else _clean(sample.iloc[0][f])) for f in FEATURES
    }
    Path("data/widget/sample_patient.json").write_text(
        json.dumps(widget, indent=2), encoding="utf-8"
    )
    print("wrote data/sample and data/widget")


def _clean(v: Any) -> Any:
    """Coerce numpy scalars to plain Python types for JSON serialization."""
    import numpy as np

    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


if __name__ == "__main__":
    main()
