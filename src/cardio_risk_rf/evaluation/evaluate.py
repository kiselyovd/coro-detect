"""CLI: score a trained model on a Parquet split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from ..data.cardio import FEATURES, TARGET
from .metrics import compute_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    df = pd.read_parquet(args.split)
    probs = model.predict_proba(df[FEATURES])[:, 1]
    m = compute_metrics(df[TARGET].to_numpy(), probs, threshold=args.threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
