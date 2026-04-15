"""CLI: OOF ensemble (CatBoost + LightGBM), blend, full fit, Kaggle submission CSV."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wildlife_strikes.config import DEFAULT_SUBMISSION, DEFAULT_TEST_CLEAN, DEFAULT_TRAIN_CLEAN, ID_COL, TARGET_COL
from wildlife_strikes.ensemble import fit_full_models, predict_test_blend, run_oof_ensemble

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_train_test(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    if TARGET_COL not in train.columns:
        raise SystemExit(f"{train_path} must contain {TARGET_COL}")
    y = train[TARGET_COL].astype(np.int32).to_numpy()
    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    if set(feature_cols) != set(c for c in test.columns if c != ID_COL):
        raise SystemExit("Train and test must share the same feature columns (aside from target).")
    return train, test, y, feature_cols


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Stratified OOF ensemble (CatBoost + LightGBM or dual CatBoost) → submission CSV"
    )
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN_CLEAN, help="Cleaned training CSV")
    p.add_argument("--test", type=Path, default=DEFAULT_TEST_CLEAN, help="Cleaned test CSV")
    p.add_argument("--output", type=Path, default=DEFAULT_SUBMISSION, help="Submission path")
    p.add_argument("--bundle", type=Path, default=Path("artifacts/ensemble_bundle.joblib"), help="Save fitted models + meta")
    p.add_argument("--n-splits", type=int, default=5, help="Stratified CV folds for OOF")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-oof", action="store_true", help="Skip OOF (faster; uses default blend 0.5)")
    p.add_argument("--max-train-rows", type=int, default=0, help="Debug: cap training rows (0=all)")
    args = p.parse_args(argv)

    train, test, y, feature_cols = load_train_test(args.train, args.test)

    if args.max_train_rows > 0:
        train = train.iloc[: args.max_train_rows].reset_index(drop=True)
        y = y[: args.max_train_rows]
        logger.warning("Using first %d rows only (--max-train-rows)", args.max_train_rows)

    blend_w = 0.5
    meta = None

    if not args.skip_oof:
        _, _, meta = run_oof_ensemble(
            train,
            y,
            feature_cols=feature_cols,
            n_splits=args.n_splits,
            random_state=args.seed,
        )
        blend_w = meta.weight_catboost
        logger.info("OOF blend weight (CatBoost): %.5f", blend_w)

    logger.info("Training full models on all rows (primary CatBoost + secondary)…")
    cb, lgbm, cat_cols = fit_full_models(train, y, feature_cols=feature_cols, random_state=args.seed)

    proba = predict_test_blend(cb, lgbm, test, feature_cols=feature_cols, cat_cols=cat_cols, blend_weight=blend_w)

    out = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: proba})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info("Wrote %s (%d rows)", args.output, len(out))

    bundle = {
        "catboost": cb,
        "secondary_model": lgbm,
        "cat_columns": cat_cols,
        "blend_weight_catboost": blend_w,
        "feature_cols": feature_cols,
        "oof_meta": meta.__dict__ if meta else None,
    }
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.bundle)
    logger.info("Saved bundle %s", args.bundle)

    meta_path = args.bundle.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "oof_meta": meta.__dict__ if meta else None,
                "blend_weight_catboost": blend_w,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_features": len(feature_cols),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
