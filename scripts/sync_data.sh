#!/usr/bin/env bash
# Fetch Framingham Heart Study CSV into data/raw/framingham.csv.
# Primary: Kaggle CLI (expects ~/.kaggle/kaggle.json).
# Fallback: huggingface_hub download from kiselyovd/framingham mirror.
set -euo pipefail

ROOT="${CARDIO_REPO_ROOT:-$(pwd)}"
RAW_DIR="${ROOT}/data/raw"
TARGET="${RAW_DIR}/framingham.csv"

mkdir -p "${RAW_DIR}"

if [[ -s "${TARGET}" ]]; then
  echo "[sync_data] ${TARGET} already present, skipping."
  exit 0
fi

if command -v kaggle >/dev/null 2>&1 && { [[ -f "${HOME}/.kaggle/kaggle.json" ]] || [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]] || [[ -n "${KAGGLE_API_TOKEN:-}" ]]; }; then
  echo "[sync_data] Trying Kaggle..."
  tmp="$(mktemp -d)"
  kaggle datasets download -d aasheesh200/framingham-heart-study-dataset -p "${tmp}" --unzip
  mv "${tmp}/framingham.csv" "${TARGET}"
  rm -rf "${tmp}"
  echo "[sync_data] Downloaded via Kaggle."
  exit 0
fi

echo "[sync_data] Kaggle path unavailable, trying HF Datasets mirror..."
uv run python - <<'PY'
import os, pathlib
from huggingface_hub import hf_hub_download
root = pathlib.Path(os.environ.get("CARDIO_REPO_ROOT") or ".").resolve()
target = root / "data" / "raw" / "framingham.csv"
path = hf_hub_download(
    repo_id="kiselyovd/framingham",
    filename="framingham.csv",
    repo_type="dataset",
)
target.parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(path).replace(target)
print(f"[sync_data] Downloaded via HF to {target}")
PY
