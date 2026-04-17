#!/usr/bin/env bash
# Fetch sulianova Cardiovascular Disease Dataset into data/raw/cardio.csv.
# Primary: Kaggle CLI (expects ~/.kaggle/kaggle.json OR KAGGLE_USERNAME+KAGGLE_KEY
# OR KAGGLE_API_TOKEN env vars).
# Fallback: huggingface_hub download from kiselyovd/cardio-disease-sulianova mirror.
set -euo pipefail

ROOT="${CARDIO_REPO_ROOT:-$(pwd)}"
RAW_DIR="${ROOT}/data/raw"
TARGET="${RAW_DIR}/cardio.csv"

mkdir -p "${RAW_DIR}"

if [[ -s "${TARGET}" ]]; then
  echo "[sync_data] ${TARGET} already present, skipping."
  exit 0
fi

if command -v kaggle >/dev/null 2>&1 && { [[ -f "${HOME}/.kaggle/kaggle.json" ]] || [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]] || [[ -n "${KAGGLE_API_TOKEN:-}" ]]; }; then
  echo "[sync_data] Trying Kaggle..."
  tmp="$(mktemp -d)"
  kaggle datasets download -d sulianova/cardiovascular-disease-dataset -p "${tmp}" --unzip
  # sulianova zip contains cardio_train.csv; rename to cardio.csv for the pipeline.
  if [[ -f "${tmp}/cardio_train.csv" ]]; then
    mv "${tmp}/cardio_train.csv" "${TARGET}"
  else
    mv "${tmp}/cardio.csv" "${TARGET}"
  fi
  rm -rf "${tmp}"
  echo "[sync_data] Downloaded via Kaggle."
  exit 0
fi

echo "[sync_data] Kaggle path unavailable, trying HF Datasets mirror..."
uv run python - <<'PY'
import os, pathlib
from huggingface_hub import hf_hub_download
root = pathlib.Path(os.environ.get("CARDIO_REPO_ROOT") or ".").resolve()
target = root / "data" / "raw" / "cardio.csv"
path = hf_hub_download(
    repo_id="kiselyovd/cardio-disease-sulianova",
    filename="cardio.csv",
    repo_type="dataset",
)
target.parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(path).replace(target)
print(f"[sync_data] Downloaded via HF to {target}")
PY
