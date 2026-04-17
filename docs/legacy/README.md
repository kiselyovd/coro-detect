# Legacy dataset

`original_dataset.csv` is the 49-row cardiovascular-biomarker table shipped with the original `coro-detect` repository by the same author, with Cyrillic column names (`болезнь, Возраст, Лейкоциты, КДО, склеростин, WIF-1, GSK-3α`).

It is preserved here for provenance. The production pipeline was **switched** to the Framingham Heart Study dataset because 49 rows do not support a legitimate held-out test split or reliable calibration reporting. See `docs/superpowers/specs/2026-04-17-cardio-risk-rf-design.md` §1 for the full rationale. This spec is maintained in the portfolio's top-level `docs/superpowers/specs/` directory, not in this repo.

To replay the original on the new pipeline for a demo, see `notebooks/02_demo.ipynb`.
