"""Retune binary cutoff for an existing submission using a fitted model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wildlife_strikes.config import ID_COL, TARGET_COL
from wildlife_strikes.ensemble import optimize_binary_threshold, predict_test_blend


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Retune binary threshold from trained bundle + train labels.")
    p.add_argument("--train", type=Path, default=Path("train_clean.csv"))
    p.add_argument("--submission", type=Path, default=Path("submission.csv"))
    p.add_argument("--bundle", type=Path, default=Path("artifacts/ensemble_bundle.joblib"))
    p.add_argument("--output", type=Path, default=Path("submission_binary_balanced.csv"))
    p.add_argument(
        "--metric",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "accuracy"],
        help="Metric to maximize when selecting threshold.",
    )
    p.add_argument("--grid-size", type=int, default=1001, help="Threshold grid size")
    args = p.parse_args(argv)

    train = pd.read_csv(args.train, low_memory=False)
    if TARGET_COL not in train.columns:
        raise SystemExit(f"{args.train} must contain {TARGET_COL}")
    y = train[TARGET_COL].astype(np.int32).to_numpy()

    bundle = joblib.load(args.bundle)
    cb_model = bundle["catboost"]
    secondary_model = bundle["secondary_model"]
    feature_cols = bundle["feature_cols"]
    cat_cols = bundle["cat_columns"]
    blend_weight = float(bundle["blend_weight_catboost"])

    train_proba = predict_test_blend(
        cb_model,
        secondary_model,
        train,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        blend_weight=blend_weight,
    )
    threshold, score = optimize_binary_threshold(y, train_proba, grid_size=args.grid_size, metric=args.metric)

    sub = pd.read_csv(args.submission)
    required = {ID_COL, TARGET_COL}
    missing = required - set(sub.columns)
    if missing:
        raise SystemExit(f"{args.submission} missing columns: {sorted(missing)}")
    probs = pd.to_numeric(sub[TARGET_COL], errors="coerce")
    if probs.isna().any():
        raise SystemExit(f"{args.submission}:{TARGET_COL} must be numeric probabilities")

    out = pd.DataFrame({ID_COL: sub[ID_COL], TARGET_COL: (probs >= threshold).astype(np.int8)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(
        f"Wrote {args.output} ({len(out):,} rows) | threshold={threshold:.4f} "
        f"| {args.metric} (train-fit estimate)={score:.5f}"
    )


if __name__ == "__main__":
    main()
