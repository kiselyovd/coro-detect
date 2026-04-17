"""Inference CLI — load a checkpoint and predict on input(s)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


import joblib


def load_model(path: str | Path):
    return joblib.load(path)


def predict(model, features: dict) -> dict:
    import pandas as pd

    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0].tolist()
    pred = int(model.predict(X)[0])
    return {"pred": pred, "proba": proba}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    configure_logging()
    model = load_model(args.checkpoint)
    result = predict(model, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
