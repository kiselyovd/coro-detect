"""Dataset implementations."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(csv_path: Path | str) -> pd.DataFrame:
    """Load a CSV into a dataframe."""
    return pd.read_csv(csv_path)
