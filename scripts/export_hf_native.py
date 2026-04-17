"""Export trained model to HuggingFace-native format (safetensors + config.json).

Run BEFORE publish_to_hf.py so the HF repo gets proper pipeline pills / Inference
Providers instead of just a raw Lightning .ckpt.

Usage:
    python scripts/export_hf_native.py \\
        --checkpoint artifacts/checkpoints/best.ckpt \\
        --out artifacts/hf_export \\
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Tabular models have no HF native export; upload joblib directly via publish_to_hf. "
        "Place your serialized model at artifacts/model.joblib and run publish_to_hf.py normally."
    )


if __name__ == "__main__":
    main()
