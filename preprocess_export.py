"""
Fit the same ColumnTransformer as train_pipeline (MLP-style: scaled numerics,
one-hot + frequency categoricals) — no model training.

Writes dense numeric matrices as CSV for spreadsheets / notebooks, plus small
preview files you can open quickly.

Usage:
  python3 preprocess_export.py
  python3 preprocess_export.py --out-dir preprocessed --sample-rows 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_pipeline import (
    ID_COL,
    TARGET_COL,
    _make_preprocessor,
    infer_column_types,
    load_xy,
    split_categorical_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export preprocessed train/test matrices (no model).")
    parser.add_argument("--train", type=Path, default=Path("train_clean.csv"))
    parser.add_argument("--test", type=Path, default=Path("test_clean.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--max-categories", type=int, default=25, help="OneHotEncoder max_categories (same as train_pipeline).")
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Median-impute numerics only (no StandardScaler); matches train_pipeline --model hgb style.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=500,
        help="Also write train_preview.csv / test_preview.csv with this many rows (0 = skip previews).",
    )
    parser.add_argument(
        "--save-preprocessor",
        action="store_true",
        help="Save fitted ColumnTransformer to out-dir/preprocessor.joblib for later predict/transform.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="If >0, only use this many train rows to fit/transform (0 = full train).",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="If >0, only export this many test rows (0 = full test).",
    )
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    X_train, y_train, id_train = load_xy(args.train)
    if y_train is None:
        raise ValueError(f"{args.train} must include {TARGET_COL}")

    if args.max_train_rows > 0:
        X_train = X_train.iloc[: args.max_train_rows].reset_index(drop=True)
        y_train = y_train[: args.max_train_rows]
        id_train = id_train.iloc[: args.max_train_rows].reset_index(drop=True)
        print(f"Using first {args.max_train_rows:,} train rows (--max-train-rows).")

    X_test, _, id_test = load_xy(args.test)
    if args.max_test_rows > 0:
        X_test = X_test.iloc[: args.max_test_rows].reset_index(drop=True)
        id_test = id_test.iloc[: args.max_test_rows].reset_index(drop=True)
        print(f"Using first {args.max_test_rows:,} test rows (--max-test-rows).")

    numeric_features, categorical_features = infer_column_types(X_train)
    cat_low, cat_high = split_categorical_features(categorical_features)

    pre = _make_preprocessor(
        numeric_features,
        cat_low,
        cat_high,
        max_categories=args.max_categories,
        scale_numeric=not args.no_scale,
    )

    print(f"Fitting preprocessor on {len(X_train):,} training rows…")
    pre.fit(X_train)

    print("Transforming train and test…")
    Xt = pre.transform(X_train)
    Xv = pre.transform(X_test)

    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:
        feat_names = [f"f{i}" for i in range(Xt.shape[1])]

    train_df = pd.DataFrame(Xt, columns=feat_names, dtype=np.float64)
    train_df.insert(0, ID_COL, id_train.values)
    train_df[TARGET_COL] = y_train

    test_df = pd.DataFrame(Xv, columns=feat_names, dtype=np.float64)
    test_df.insert(0, ID_COL, id_test.values)

    train_path = out / "train_preprocessed.csv"
    test_path = out / "test_preprocessed.csv"

    float_fmt = "%.8g"
    print(f"Writing {train_path} ({train_df.shape[0]:,} × {train_df.shape[1]} cols)…")
    train_df.to_csv(train_path, index=False, float_format=float_fmt)
    print(f"Writing {test_path} ({test_df.shape[0]:,} × {test_df.shape[1]} cols)…")
    test_df.to_csv(test_path, index=False, float_format=float_fmt)

    names_path = out / "feature_names.txt"
    names_path.write_text("\n".join(feat_names), encoding="utf-8")
    print(f"Wrote {names_path} ({len(feat_names)} feature names)")

    if args.sample_rows > 0:
        n = min(args.sample_rows, len(train_df))
        prev_train = out / "train_preview.csv"
        prev_test = out / "test_preview.csv"
        train_df.head(n).to_csv(prev_train, index=False, float_format=float_fmt)
        test_df.head(min(args.sample_rows, len(test_df))).to_csv(prev_test, index=False, float_format=float_fmt)
        print(f"Wrote {prev_train} and {prev_test} (first {n} / {min(args.sample_rows, len(test_df))} rows)")

    if args.save_preprocessor:
        ppath = out / "preprocessor.joblib"
        joblib.dump(
            {
                "preprocessor": pre,
                "numeric_features": numeric_features,
                "categorical_onehot": cat_low,
                "categorical_frequency": cat_high,
                "max_categories": args.max_categories,
                "scale_numeric": not args.no_scale,
            },
            ppath,
        )
        print(f"Wrote {ppath}")

    print("Done.")


if __name__ == "__main__":
    main()
