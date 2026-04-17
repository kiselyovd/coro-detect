"""Orchestrate: train main (LGBM+Optuna) → train baseline (RF+Grid) → score on test
→ calibration plot on val → global SHAP → summary JSON."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from cardio_risk_rf.data.cardio import FEATURES, TARGET
from cardio_risk_rf.evaluation.calibration import save_calibration_plot
from cardio_risk_rf.evaluation.metrics import compute_metrics
from cardio_risk_rf.evaluation.summary import build_summary
from cardio_risk_rf.explain import explain_global
from cardio_risk_rf.training.train import train_baseline, train_main


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    processed = Path("data/processed")
    artifacts = Path("artifacts")
    reports = Path("reports")

    train_df = pd.read_parquet(processed / "train.parquet")
    val_df = pd.read_parquet(processed / "val.parquet")
    test_df = pd.read_parquet(processed / "test.parquet")
    _log(f"splits train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    _log("=== LightGBM main (Optuna 50) ===")
    main_ckpt = train_main(
        train_df=train_df,
        val_df=val_df,
        out_path=artifacts / "main" / "cardio_risk_lgbm.joblib",
        optuna_trials=50,
        seed=42,
    )

    _log("=== RandomForest baseline (Grid 5-fold) ===")
    base_ckpt = train_baseline(
        train_df=train_df,
        val_df=val_df,
        out_path=artifacts / "baseline" / "cardio_risk_rf.joblib",
        cv_folds=5,
        seed=42,
    )

    _log("=== Scoring main on test ===")
    import joblib as _jl

    main_bundle = _jl.load(main_ckpt)
    main_model = main_bundle["model"] if isinstance(main_bundle, dict) else main_bundle
    probs_main = main_model.predict_proba(test_df[FEATURES])[:, 1]
    m_main = compute_metrics(test_df[TARGET].to_numpy(), probs_main, threshold=0.5)
    (reports / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    Path(reports / "metrics.json").write_text(json.dumps(m_main, indent=2), encoding="utf-8")
    _log(f"main metrics: {m_main}")

    _log("=== Scoring baseline on test ===")
    base_bundle = _jl.load(base_ckpt)
    base_model = base_bundle["model"] if isinstance(base_bundle, dict) else base_bundle
    probs_base = base_model.predict_proba(test_df[FEATURES])[:, 1]
    m_base = compute_metrics(test_df[TARGET].to_numpy(), probs_base, threshold=0.5)
    Path(reports / "metrics_baseline.json").write_text(
        json.dumps(m_base, indent=2), encoding="utf-8"
    )
    _log(f"baseline metrics: {m_base}")

    _log("=== Calibration plot on val ===")
    probs_val = main_model.predict_proba(val_df[FEATURES])[:, 1]
    save_calibration_plot(
        val_df[TARGET].to_numpy(),
        probs_val,
        reports / "calibration.png",
        bins=10,
        title="LightGBM calibration (val)",
    )

    _log("=== Global SHAP ===")
    explain_global(
        main_model,
        val_df[FEATURES],
        out_png=reports / "shap_summary.png",
        out_csv=reports / "shap_importance.csv",
    )

    _log("=== Summary ===")
    build_summary(m_main, m_base, out_path=reports / "metrics_summary.json")

    _log(f"TOTAL TIME: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
