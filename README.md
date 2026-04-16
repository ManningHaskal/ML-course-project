# Wildlife strike damage prediction

End-to-end pipeline for the course **FAA wildlife strike** tabular competition: clean raw CSVs, train a **stratified out-of-fold (OOF) gradient-boosting ensemble**, tune a **convex blend** on OOF predictions to maximize **ROC-AUC**, then refit on the full training set and export **class-1 probabilities** for submission.

Confirm the official evaluation metric and submission format on your course page or Kaggle; this project optimizes **ROC-AUC** on held-out folds, which matches typical binary damage-class leaderboards.

## Method

1. **Cleaning** (`wildlife-clean`): normalize sentinels, derive numeric time features, reconcile date fields, encode high-cardinality text as lengths where appropriate, and align train/test schemas. See `wildlife_strikes/cleaning.py` for the exact rules.

2. **Ensemble** (`wildlife_strikes/ensemble.py`):
   - **Primary model:** `CatBoostClassifier` with native categorical features (no one-hot explosion).
   - **Secondary model:** `LGBMClassifier` with pandas `category` dtypes for the same columns—**or**, if LightGBM cannot load (common on macOS without OpenMP), a **second CatBoost** with different depth/regularization for diversity.
   - **Stratified K-fold OOF** produces out-of-fold positive-class probabilities from both models.
   - **Blend weight** \(w \in [0,1]\) is chosen on a grid so that \(w \cdot p_{\text{CB}} + (1-w) \cdot p_{\text{sec}}\) maximizes ROC-AUC on OOF targets.
   - **Final fit:** both models are retrained on **all** training rows; test predictions use the same \(w\).

3. **Output:** CSV with `INDEX_NR` and `INDICATED_DAMAGE` (probability of damage).

On **very small** training sets (or OOF folds), row subsampling is automatically disabled and LightGBM’s `min_child_samples` is reduced so CatBoost/LightGBM do not fail during local tests or debugging with `--max-train-rows`.

## Setup

```bash
cd ML-course-project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

### LightGBM on macOS

If `import lightgbm` fails with a **dynamic library** / `libomp` error, install OpenMP and retry:

```bash
brew install libomp
```

Without LightGBM, the pipeline still runs using the **dual-CatBoost** fallback (slightly less diversity than CatBoost + LightGBM).

## Data layout

Place Kaggle (or course) files in the project root (or pass paths explicitly):

| File        | Role                          |
|------------|--------------------------------|
| `train.csv` | Raw training (includes `INDICATED_DAMAGE`) |
| `test.csv`  | Raw test (no target)         |

Generated:

| File            | Role                    |
|----------------|-------------------------|
| `train_clean.csv` | Cleaned train        |
| `test_clean.csv`    | Cleaned test         |
| `submission.csv`    | Default submission output |

## Commands

**Clean raw data:**

```bash
wildlife-clean
# or: python -m wildlife_strikes.cleaning
```

**Train and write submission:**

```bash
wildlife-train --output submission.csv
```

`wildlife-train` now also writes a binary-label companion file by default:
`submission_binary.csv`. When OOF is enabled, the binary cutoff is auto-tuned on
OOF predictions for the selected metric (default: **balanced accuracy**).

**Convert probability submission to binary labels (if required):**

```bash
wildlife-binarize --input submission.csv --output submission_binary.csv --threshold 0.5
```

Useful flags:

- `--skip-oof` — skip OOF and use default blend `0.5` (faster, worse calibration of \(w\)).
- `--n-splits 5` — number of stratified folds for OOF.
- `--max-train-rows N` — cap rows for debugging.
- `--binary-output path.csv` — where to write binary predictions.
- `--binary-threshold 0.5` — override threshold (otherwise OOF-optimized when available).
- `--binary-metric balanced_accuracy` — threshold tuning objective (`balanced_accuracy` or `accuracy`).

If you already have a fitted bundle and probability submission, you can retune
the cutoff without retraining:

```bash
wildlife-retune-threshold \
  --train train_clean.csv \
  --submission submission.csv \
  --bundle artifacts/ensemble_bundle.joblib \
  --output submission_binary_balanced.csv \
  --metric balanced_accuracy
```

Artifacts:

- `artifacts/ensemble_bundle.joblib` — fitted models and metadata.
- `artifacts/ensemble_bundle.json` — OOF metrics and blend weight.

## Tests

```bash
pytest
# include slow integration test (boosting):
pytest -m slow
# coverage:
pytest --cov=wildlife_strikes --cov-report=term-missing
```

## Legacy scripts

Older one-off scripts (`train_pipeline.py`, `submission_cv_run.py`, `preprocess_export.py`, etc.) predate the `wildlife_strikes` package. Prefer **`wildlife-clean`** and **`wildlife-train`** for reproducible competition runs.

## Project layout

```
src/wildlife_strikes/
  config.py      # column names and default paths
  cleaning.py    # raw → clean CSV
  ensemble.py    # OOF, blend, full fit, predict
  train_cli.py   # CLI entry for training + submission
tests/
  test_cleaning.py
  test_ensemble_unit.py
  test_ensemble_smoke.py
```

## License

MIT (see `pyproject.toml`).
