"""CLI to produce data/processed/{train,val,test}.parquet from raw CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cardio import load_cardio, split_stratified


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/cardio.csv")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = load_cardio(args.raw)
    train_df, val_df, test_df = split_stratified(df, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out / "train.parquet", index=False)
    val_df.to_parquet(out / "val.parquet", index=False)
    test_df.to_parquet(out / "test.parquet", index=False)
    print(f"wrote train={len(train_df)} val={len(val_df)} test={len(test_df)} to {out}")


if __name__ == "__main__":
    main()
