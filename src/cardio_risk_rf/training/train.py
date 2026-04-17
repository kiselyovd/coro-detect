"""Training entrypoint (Hydra-powered)."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..utils import configure_logging, get_logger, seed_everything

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))
    log.info("train.start", config=OmegaConf.to_container(cfg, resolve=True))

    import json

    import joblib
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    from ..data import load_dataset
    from ..models import build_pipeline

    df = load_dataset(cfg.data.csv_path)
    target = cfg.data.target_col
    X = df.drop(columns=[target])
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.seed, stratify=y,
    )
    pipe = build_pipeline(cfg.model.name, **OmegaConf.to_container(cfg.model.params, resolve=True))
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    metrics = {"report": classification_report(y_te, y_pred, output_dict=True)}
    if hasattr(pipe, "predict_proba") and len(pipe.classes_) == 2:
        y_proba = pipe.predict_proba(X_te)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_te, y_proba))
    out_dir = Path(cfg.trainer.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("train.done", out=str(out_dir))


if __name__ == "__main__":
    main()
