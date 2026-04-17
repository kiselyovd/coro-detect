# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] — 2026-04-17

Initial release — Milestone 4 of the `kiselyovd` ML portfolio refactor.

### Added

- **Main model:** LightGBM classifier tuned with Optuna (50 trials, TPE sampler, early stopping on val). Native NaN handling — no imputer in the pipeline. `scale_pos_weight = N_neg / N_pos` for class imbalance.
- **Baseline model:** RandomForest classifier with `SimpleImputer(strategy="median")` + `GridSearchCV` (5-fold stratified), `class_weight="balanced"`.
- **Dataset:** Framingham Heart Study 10-Year CHD Risk (4240 patients × 16 features), fetched via `scripts/sync_data.sh` from Kaggle (`aasheesh200/framingham-heart-study-dataset`) with an HF Datasets fallback mirror; 70/15/15 stratified split.
- **Evaluation:** ROC-AUC / PR-AUC / F1 / Brier + calibration plot on val, F1-optimal threshold search (`reports/metrics_thresholded.json`).
- **Explainability:** SHAP global summary (beeswarm + importance CSV) on val; local `shap_top5` on each `/predict` call.
- **Serving:** FastAPI `/predict?model={main|baseline}` returning `{probability, class, threshold, shap_top5, model_version, model_name, request_id}`; `GET /health`; `GET /metrics` (Prometheus).
- **Docker:** multi-stage `python:3.13-slim` runtime, no torch / CUDA.
- **CI/CD:** matrix pytest (3.12 + 3.13), ruff + mypy + deptry + bandit + interrogate + actionlint + codespell, MkDocs Material auto-deploy, GHCR docker push on tag, GitHub Release with auto-generated notes.
- **HF Hub:** `kiselyovd/cardio-risk-rf` publishing via `scripts/publish_to_hf.py` with Jinja-rendered model card (pills, widget, `model-index`).
- **Quality gates:** full backport from M2 (brain-mri-segmentation) — pinned tool versions, pre-commit hooks, interrogate ≥30%, bandit skips for ML conventions.
- **Notebooks:** `01_eda.ipynb` (distribution, correlations, missing-value comparison), `02_demo.ipynb` (`/predict` + SHAP waterfall example — scaffolded).
- **Legacy archival:** original 49-row biomarker dataset preserved at `docs/legacy/original_dataset.csv` with provenance note.

### Metrics (held-out test, n=637, positive rate 15.2%)

| Model | ROC-AUC | PR-AUC | F1 @ t\* | Brier | t\* |
|---|---|---|---|---|---|
| **LightGBM** (main) | 66.2% | 26.6% | 30.6% | 0.128 | 0.27 |
| RandomForest (baseline) | 66.0% | 27.5% | 32.2% | 0.135 | 0.25 |

F1 reported at val-set optimal threshold `t*` (default 0.5 produces ≈0 F1 for the main model on imbalanced Framingham — both models output well-calibrated probabilities that rarely exceed 0.5).

### Fixed

- `scripts/sync_data.sh` now supports `KAGGLE_USERNAME`/`KAGGLE_KEY` and `KAGGLE_API_TOKEN` env-var auth in addition to `~/.kaggle/kaggle.json`.
- `/predict` and CLI `predict()` now coerce the input DataFrame to `float` dtype — prevents LightGBM rejecting `None`-valued fields as `object` dtype.
- 3-D SHAP output (modern `shap` + scikit-learn RF) handled by `explain_global` / `explain_instance`.

### Known limitations

- Main model F1 at the default 0.5 threshold is ≈0 — this is expected on Framingham-level imbalance; use `t*` from `reports/metrics_thresholded.json` or the `operating points` section in training docs.
- Docker image is ~900 MB (tabular deps including matplotlib, optuna, shap, pandas, pyarrow). A follow-up will split base vs. training deps for a slim <250 MB serving image.
