# cardio-risk-rf

Production-grade tabular cardiovascular-risk classifier on the Framingham Heart Study — 10-year coronary heart disease risk prediction with LightGBM (main) and RandomForest (baseline), served through FastAPI with per-patient SHAP explanations.

## At a glance

- **Dataset:** [Framingham Heart Study 10-Year CHD](https://www.kaggle.com/datasets/neisha/heart-disease-prediction-using-logistic-regression) — 4240 patients × 16 features, ~15% positive rate on `TenYearCHD`, stratified 70/15/15 split.
- **Main model:** LightGBM with native NaN handling, tuned by Optuna (50 trials, TPE sampler, early stopping on val).
- **Baseline:** RandomForest with `SimpleImputer(median)` + `GridSearchCV` — gives a calibration reference for the main model.
- **Stack:** Python 3.12 / 3.13 · scikit-learn · LightGBM · Optuna · SHAP · FastAPI · Hydra · DVC · MkDocs Material · uv.
- **Serving:** FastAPI `/predict` returns `{probability, class, threshold, shap_top5, model_version, request_id}`. CPU-only — no GPU needed for training or inference.

## Navigation

- [Architecture](architecture.md) — data flow, pipeline layout, mermaid diagram, and the main-vs-baseline design decisions.
- [Training](training.md) — CLI commands, Optuna/Grid hyperparameter notes, and the one-time Framingham mirror runbook.
- [Serving](serving.md) — `/predict` endpoint contract, Pydantic schemas, curl example.
- [API reference](api.md) — mkdocstrings-generated reference for the `cardio_risk_rf` package.

## Links

- GitHub: [kiselyovd/cardio-risk-rf](https://github.com/kiselyovd/cardio-risk-rf)
- Hugging Face model: [kiselyovd/cardio-risk-rf](https://huggingface.co/kiselyovd/cardio-risk-rf)
- Russian README: [README.ru.md](https://github.com/kiselyovd/cardio-risk-rf/blob/main/README.ru.md)
- Template: [kiselyovd/ml-project-template](https://github.com/kiselyovd/ml-project-template)

## Intended use and disclaimer

This model is a **portfolio/research demo** trained on the public Framingham Heart Study subset. It is **not a medical device** and must not be used for clinical decision-making, diagnosis, or patient-facing risk communication. Calibration, fairness, and distribution shift have not been validated outside the original cohort. Use only for educational purposes, ML-engineering review, and to compare against other baselines on the same dataset.
