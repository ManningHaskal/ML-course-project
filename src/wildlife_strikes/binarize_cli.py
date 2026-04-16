"""Convert probabilistic submission CSV into binary labels via thresholding."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wildlife_strikes.config import ID_COL, TARGET_COL


def binarize_submission(
    input_path: Path,
    output_path: Path,
    *,
    threshold: float = 0.5,
) -> tuple[int, int]:
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")

    df = pd.read_csv(input_path)
    required = {ID_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    probs = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if probs.isna().any():
        raise ValueError(f"{TARGET_COL} contains non-numeric values; cannot binarize")

    out = pd.DataFrame(
        {
            ID_COL: df[ID_COL],
            TARGET_COL: (probs >= threshold).astype("int8"),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    positives = int(out[TARGET_COL].sum())
    return len(out), positives


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert submission probabilities to binary labels.")
    parser.add_argument("--input", type=Path, default=Path("submission.csv"), help="Input submission CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_binary.csv"),
        help="Output binary submission CSV",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive class threshold")
    args = parser.parse_args(argv)

    rows, positives = binarize_submission(args.input, args.output, threshold=args.threshold)
    print(
        f"Wrote {args.output} ({rows:,} rows; positives={positives:,}; threshold={args.threshold:.4f})"
    )


if __name__ == "__main__":
    main()
