# cardio-risk-rf

[![CI](https://img.shields.io/github/actions/workflow/status/kiselyovd/cardio-risk-rf/test.yml?branch=main&style=for-the-badge&label=ci)](https://github.com/kiselyovd/cardio-risk-rf/actions)
[![codecov](https://img.shields.io/codecov/c/github/kiselyovd/cardio-risk-rf?style=for-the-badge&logo=codecov&logoColor=white)](https://codecov.io/gh/kiselyovd/cardio-risk-rf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HF Model](https://img.shields.io/badge/🤗%20HF%20Hub-model-FFD21E?style=for-the-badge)](https://huggingface.co/kiselyovd/cardio-risk-rf)

Промышленный табличный классификатор сердечно-сосудистого риска на Framingham Heart Study (4240 пациентов, 10-летний риск CHD). Основная модель — **LightGBM** с нативной обработкой NaN и объяснимостью через SHAP; baseline — **RandomForest** с медианной импутацией. Конфигурация через Hydra, подбор гиперпараметров через Optuna, метрики ROC-AUC / PR-AUC / F1 / Brier + calibration plot, сервинг через FastAPI (`/predict` с локальным SHAP top-5), дистрибуция через Hugging Face Hub и MkDocs Material.

> **Часть [ML-портфолио kiselyovd](https://github.com/kiselyovd#ml-portfolio)** — промышленные ML-проекты, основанные на одном [cookiecutter-шаблоне](https://github.com/kiselyovd/ml-project-template).

📖 [Документация (EN)](https://kiselyovd.github.io/cardio-risk-rf/) • 🇬🇧 [English README](README.md) • 🤗 [Модель на HF Hub](https://huggingface.co/kiselyovd/cardio-risk-rf)

## Датасет

[Framingham Heart Study 10-Year CHD Risk](https://www.kaggle.com/datasets/neisha/heart-disease-prediction-using-logistic-regression) (CC-BY-4.0). 4240 строк × 16 признаков, доля положительного класса `TenYearCHD` ~15%. Загружается через `scripts/sync_data.sh` с Kaggle, fallback — зеркало на HF Datasets. Разбиение 70/15/15 со стратификацией по таргету (`train=2968, val=636, test=636`).

Исходный 49-строчный датасет от автора `coro-detect` заархивирован в `docs/legacy/original_dataset.csv` — обоснование см. в `docs/legacy/README.md`.

## Результаты

Заполняются из `reports/metrics_summary.json` после прогона v0.1.0.

| Модель | ROC-AUC | PR-AUC | F1 | Brier |
|---|---|---|---|---|
| **LightGBM** (основная) | — | — | — | — |
| RandomForest (baseline) | — | — | — | — |

Метрики считаются на отложенном test-сплите (n≈636). Calibration plot — на val → `reports/calibration.png`.

### Глобальный SHAP (основная модель)

![SHAP summary](reports/shap_summary.png)

## Быстрый старт

```bash
uv sync --all-groups
bash scripts/sync_data.sh
uv run python -m cardio_risk_rf.data.prepare --raw data/raw/framingham.csv --out data/processed
uv run python scripts/train_all.py
```

## Деплой

```bash
docker run --rm -p 8000:8000 ghcr.io/kiselyovd/cardio-risk-rf:v0.1.0
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @data/widget/sample_patient.json
```

## Лицензия

MIT — см. [LICENSE](LICENSE). Датасет под CC-BY-4.0 (Framingham Heart Study).
