"""
Stratified k-fold ROC-AUC on train, then refit on full train and write test submission.

Defaults: HistGradientBoosting (strong/fast on tabular) + probability output for AUC-style
leaderboards. Same preprocessing as train_pipeline (frequency + one-hot categoricals).

Usage:
  python3 submission_cv_run.py --out submission_4_11_2.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import joblib

from train_pipeline import (
    ID_COL,
    TARGET_COL,
    _make_preprocessor,
    infer_column_types,
    load_xy,
    split_categorical_features,
)


def build_hgb_pipeline(
    numeric_features: list[str],
    cat_low: list[str],
    cat_high: list[str],
    *,
    max_categories: int,
    random_state: int,
) -> Pipeline:
    """HGB + same ColumnTransformer as train_pipeline --model hgb (no numeric scaling)."""
    pre = _make_preprocessor(
        numeric_features,
        cat_low,
        cat_high,
        max_categories=max_categories,
        scale_numeric=False,
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=12,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=0.5,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )
    return Pipeline([("preprocess", pre), ("model", clf)])


def main() -> None:
    parser = argparse.ArgumentParser(description="CV then full-fit submission (probabilities).")
    parser.add_argument("--train", type=Path, default=Path("train_clean.csv"))
    parser.add_argument("--test", type=Path, default=Path("test_clean.csv"))
    parser.add_argument("--out", type=Path, default=Path("submission_4_11_2.csv"))
    parser.add_argument("--out-model", type=Path, default=Path("artifacts/model_cv_hgb.joblib"))
    parser.add_argument("--cv", type=int, default=5, help="Stratified folds for ROC-AUC.")
    parser.add_argument("--max-categories", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()

    X, y, _ = load_xy(args.train)
    if y is None:
        raise SystemExit(f"{args.train} must include {TARGET_COL}")

    numeric_features, categorical_features = infer_column_types(X)
    cat_low, cat_high = split_categorical_features(categorical_features)

    pipe = build_hgb_pipeline(
        numeric_features,
        cat_low,
        cat_high,
        max_categories=args.max_categories,
        random_state=args.random_state,
    )

    skf = StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=args.random_state,
    )

    print(f"Train rows: {len(X):,} | {args.cv}-fold stratified CV (ROC-AUC)…")
    cv_scores = cross_val_score(
        pipe,
        X,
        y,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
    )
    print(f"Fold ROC-AUC: {cv_scores}")
    print(f"Mean ± std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("Refitting on full training set…")
    pipe.fit(X, y)

    y_hat = pipe.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, y_hat)
    print(f"In-sample train ROC-AUC (reference only): {train_auc:.4f}")

    X_test, _, test_ids = load_xy(args.test)
    proba = pipe.predict_proba(X_test)[:, 1]

    out = pd.DataFrame({ID_COL: test_ids, TARGET_COL: proba})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote probabilities to {args.out} ({len(out):,} rows)")

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "numeric_features": numeric_features,
            "categorical_onehot": cat_low,
            "categorical_frequency": cat_high,
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "target_col": TARGET_COL,
        },
        args.out_model,
    )
    print(f"Saved model bundle to {args.out_model}")

    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
