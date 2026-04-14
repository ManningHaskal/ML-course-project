"""
Score a saved train_pipeline model on the training CSV (in-sample metrics).

Usage:
  python3 score_training.py
  python3 score_training.py --model artifacts/model_4_11.joblib --train train_clean.csv
"""

from __future__ import annotations

import argparse
import __main__
from pathlib import Path

import joblib
import train_pipeline as tp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model on training data.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model_4_11.joblib"))
    parser.add_argument("--train", type=Path, default=Path("train_clean.csv"))
    args = parser.parse_args()

    setattr(__main__, "FrequencyEncoder", tp.FrequencyEncoder)

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]

    X, y, _ = tp.load_xy(args.train)
    if y is None:
        raise SystemExit(f"{args.train} must include {tp.TARGET_COL}")

    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]

    print("=== Training-set evaluation (in-sample; optimistic vs. held-out test) ===")
    print(f"Model: {args.model}")
    print(f"Train: {args.train} ({len(y):,} rows)\n")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, y_proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y, y_proba):.4f}")
    print(f"F1 (pos): {f1_score(y, y_pred, pos_label=1):.4f}")
    print()
    print("Confusion matrix [rows=true 0,1 | cols=pred 0,1]:")
    print(confusion_matrix(y, y_pred))
    print()
    print(classification_report(y, y_pred, digits=4))


if __name__ == "__main__":
    main()
