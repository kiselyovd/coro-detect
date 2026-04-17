# cardio-risk-rf

[![CI](https://img.shields.io/github/actions/workflow/status/kiselyovd/cardio-risk-rf/test.yml?branch=main&style=for-the-badge&label=ci)](https://github.com/kiselyovd/cardio-risk-rf/actions)
[![codecov](https://img.shields.io/codecov/c/github/kiselyovd/cardio-risk-rf?style=for-the-badge&logo=codecov&logoColor=white)](https://codecov.io/gh/kiselyovd/cardio-risk-rf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HF Model](https://img.shields.io/badge/🤗%20HF%20Hub-model-FFD21E?style=for-the-badge)](https://huggingface.co/kiselyovd/cardio-risk-rf)

Production-grade tabular cardiovascular-risk classifier on the Framingham Heart Study (4240 patients, 10-year CHD). Main model **LightGBM** with native NaN handling and SHAP explainability; baseline **RandomForest** with median imputation. Hydra-configured, Optuna-tuned, evaluated with ROC-AUC / PR-AUC / F1 / Brier + calibration plot, served by FastAPI as `/predict` with local SHAP top-5, distributed through Hugging Face Hub and MkDocs Material.

> **Part of the [kiselyovd ML portfolio](https://github.com/kiselyovd#ml-portfolio)** — production-grade ML projects sharing one [cookiecutter template](https://github.com/kiselyovd/ml-project-template).

📖 [English docs](https://kiselyovd.github.io/cardio-risk-rf/) • 🇷🇺 [Русский README](README.ru.md) • 🤗 [HF Hub model](https://huggingface.co/kiselyovd/cardio-risk-rf)

## Dataset

[Framingham Heart Study 10-Year CHD Risk](https://www.kaggle.com/datasets/neisha/heart-disease-prediction-using-logistic-regression) (CC-BY-4.0). 4240 rows × 16 features, ~15% positive rate on `TenYearCHD`. Fetched by `scripts/sync_data.sh` from Kaggle with an HF Datasets fallback mirror. Split 70/15/15 stratified by target (`train=2968, val=636, test=636`).

The original 49-row dataset from the `coro-detect` author is archived at `docs/legacy/original_dataset.csv` — see `docs/legacy/README.md` for rationale.

## Results

Filled in from `reports/metrics_summary.json` once the v0.1.0 run completes.

| Model | ROC-AUC | PR-AUC | F1 | Brier |
|---|---|---|---|---|
| **LightGBM** (main) | — | — | — | — |
| RandomForest (baseline) | — | — | — | — |

Metrics computed on held-out test split (n≈636). Calibration plot on val → `reports/calibration.png`.

### Global SHAP (main model)

![SHAP summary](reports/shap_summary.png)

## Quick Start

```bash
uv sync --all-groups
bash scripts/sync_data.sh
uv run python -m cardio_risk_rf.data.prepare --raw data/raw/framingham.csv --out data/processed
uv run python scripts/train_all.py
```

## Serving

```bash
docker run --rm -p 8000:8000 ghcr.io/kiselyovd/cardio-risk-rf:v0.1.0
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @data/widget/sample_patient.json
```

## License

MIT — see [LICENSE](LICENSE). Dataset under CC-BY-4.0 (Framingham Heart Study).
