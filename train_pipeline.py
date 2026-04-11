"""
Train a sklearn pipeline on train_clean.csv and predict test_clean.csv.

- Drops INDEX_NR from features (keeps it for submission).
- MLP (default): median imputation + StandardScaler on numeric columns;
  low-cardinality categoricals -> one-hot (max_categories cap);
  high-cardinality nominals (e.g. AIRPORT_ID) -> frequency encoding (single
  float in [0,1] per column — no false ordering like label encoding 0..K-1).
- Optional --model hgb: same preprocessing, HistGradientBoostingClassifier.

Usage:
  python3 train_pipeline.py
  python3 train_pipeline.py --subsample 50000 --cv 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ID_COL = "INDEX_NR"
TARGET_COL = "INDICATED_DAMAGE"

# Nominal IDs / text with many levels: map category -> training frequency (not integer codes).
HIGH_CARDINALITY_CAT = frozenset(
    {
        "AIRPORT_ID",
        "AIRPORT",
        "LOCATION",
        "OPID",
        "OPERATOR",
        "SPECIES_ID",
        "SPECIES",
        "REG",
        "FLT",
        "LUPDATE",
    }
)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Replace each category with its relative frequency in the training set.
    Unknown / unseen categories get the mean training frequency for that column.
    Output is dense float (one column per input column) — MLP-friendly.
    """

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        self.maps_: list[dict[str, float]] = []
        self.fallback_: list[float] = []
        for i in range(self.n_features_in_):
            col = pd.Series(X[:, i]).apply(_cell_to_key)
            vc = col.value_counts(normalize=True)
            freq_map = {k: float(v) for k, v in vc.items()}
            self.maps_.append(freq_map)
            self.fallback_.append(float(vc.mean()) if len(vc) else 0.0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        cols = []
        for i in range(self.n_features_in_):
            keys = pd.Series(X[:, i]).apply(_cell_to_key)
            m = self.maps_[i]
            fb = self.fallback_[i]
            vals = keys.map(m).astype("float64")
            cols.append(vals.fillna(fb).to_numpy())
        return np.column_stack(cols) if cols else np.empty((len(X), 0))


def _cell_to_key(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Unknown"
    s = str(v).strip()
    return s if s else "Unknown"


def split_categorical_features(categorical: list[str]) -> tuple[list[str], list[str]]:
    """Low-card (one-hot) vs high-card (frequency) columns present in data."""
    high = [c for c in categorical if c in HIGH_CARDINALITY_CAT]
    low = [c for c in categorical if c not in HIGH_CARDINALITY_CAT]
    return low, high


def load_xy(path: Path) -> tuple[pd.DataFrame, np.ndarray | None, pd.Series | None]:
    df = pd.read_csv(path, low_memory=False)
    ids = df[ID_COL]
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(np.int32).to_numpy()
        X = df.drop(columns=[ID_COL, TARGET_COL])
        return X, y, ids
    X = df.drop(columns=[ID_COL])
    return X, None, ids


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def _make_preprocessor(
    numeric_features: list[str],
    cat_low: list[str],
    cat_high: list[str],
    *,
    max_categories: int,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Numeric branch + one-hot (low card) + frequency (high card nominals)."""
    if scale_numeric:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        num_pipe = SimpleImputer(strategy="median")

    transformers: list = [("num", num_pipe, numeric_features)]

    if cat_low:
        cat_oh = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        max_categories=max_categories,
                        sparse_output=False,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        transformers.append(("cat_oh", cat_oh, cat_low))

    if cat_high:
        cat_freq = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("freq", FrequencyEncoder()),
            ]
        )
        transformers.append(("cat_freq", cat_freq, cat_high))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    max_categories: int,
    random_state: int,
    model: str,
) -> Pipeline:
    model = model.lower().strip()
    cat_low, cat_high = split_categorical_features(categorical_features)
    scale_numeric = model == "mlp"
    preprocessor = _make_preprocessor(
        numeric_features,
        cat_low,
        cat_high,
        max_categories=max_categories,
        scale_numeric=scale_numeric,
    )
    if model == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=1024,
            learning_rate_init=0.001,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        )
    elif model == "hgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=10,
            max_iter=300,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        )
    else:
        raise ValueError("model must be 'mlp' or 'hgb'")

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train damage classifier pipeline.")
    parser.add_argument("--train", type=Path, default=Path("train_clean.csv"))
    parser.add_argument("--test", type=Path, default=Path("test_clean.csv"))
    parser.add_argument("--out-model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--out-pred", type=Path, default=Path("artifacts/submission.csv"))
    parser.add_argument("--cv", type=int, default=3, help="Stratified CV folds for OOF AUC (0 = skip CV)")
    parser.add_argument("--subsample", type=int, default=0, help="If >0, train on this many stratified rows.")
    parser.add_argument("--max-categories", type=int, default=25, help="OneHotEncoder max_categories per feature.")
    parser.add_argument("--model", choices=("mlp", "hgb"), default="mlp", help="Backbone classifier.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--submission-proba",
        action="store_true",
        help="Write INDICATED_DAMAGE as probability in [0,1]; default is binary 0/1 via predict().",
    )
    args = parser.parse_args()

    X, y, train_ids = load_xy(args.train)
    if y is None:
        raise ValueError(f"{args.train} must include {TARGET_COL}")

    numeric_features, categorical_features = infer_column_types(X)
    cat_low, cat_high = split_categorical_features(categorical_features)
    print(f"Numeric features ({len(numeric_features)}): {numeric_features[:8]}{'...' if len(numeric_features) > 8 else ''}")
    print(f"Categorical ({len(categorical_features)}): {len(cat_low)} one-hot, {len(cat_high)} frequency — {cat_high[:5]}{'...' if len(cat_high) > 5 else ''}")

    if args.subsample > 0 and args.subsample < len(X):
        idx = np.arange(len(X))
        idx_train, _ = train_test_split(
            idx,
            train_size=args.subsample,
            stratify=y,
            random_state=args.random_state,
        )
        X = X.iloc[idx_train].reset_index(drop=True)
        y = y[idx_train]
        train_ids = train_ids.iloc[idx_train].reset_index(drop=True)
        print(f"Subsampled to {len(X):,} rows for training.")

    pipe = build_pipeline(
        numeric_features,
        categorical_features,
        max_categories=args.max_categories,
        random_state=args.random_state,
        model=args.model,
    )
    print(f"Model: {args.model}")

    if args.cv > 1:
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
        cv_params = None
        if args.model == "mlp":
            sample_weight = compute_sample_weight("balanced", y)
            cv_params = {"model__sample_weight": sample_weight}
        y_proba = cross_val_predict(
            pipe,
            X,
            y,
            cv=skf,
            method="predict_proba",
            n_jobs=-1,
            params=cv_params,
        )[:, 1]
        auc = roc_auc_score(y, y_proba)
        print(f"Stratified {args.cv}-fold CV ROC-AUC (out-of-fold): {auc:.4f}")
        if args.model == "mlp":
            sw = compute_sample_weight("balanced", y)
            pipe.fit(X, y, model__sample_weight=sw)
        else:
            pipe.fit(X, y)
    else:
        if args.model == "mlp":
            sw = compute_sample_weight("balanced", y)
            pipe.fit(X, y, model__sample_weight=sw)
        else:
            pipe.fit(X, y)
        print("Fitted on full training set (CV skipped).")

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "categorical_onehot": cat_low,
            "categorical_frequency": cat_high,
            "high_cardinality_set": sorted(HIGH_CARDINALITY_CAT),
            "target_col": TARGET_COL,
            "model_type": args.model,
        },
        args.out_model,
    )
    print(f"Saved pipeline to {args.out_model}")

    X_test, _, test_ids = load_xy(args.test)
    if args.submission_proba:
        out_pred = pipe.predict_proba(X_test)[:, 1]
    else:
        out_pred = pipe.predict(X_test).astype(np.int32)
    args.out_pred.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: out_pred})
    sub.to_csv(args.out_pred, index=False)
    kind = "probabilities" if args.submission_proba else "binary labels"
    print(f"Wrote {kind} for {len(sub):,} rows to {args.out_pred}")


if __name__ == "__main__":
    main()
