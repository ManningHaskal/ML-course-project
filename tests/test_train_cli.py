"""Integration test for training CLI (small data, skip OOF for speed)."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.slow
def test_train_cli_writes_submission(tmp_path: Path, tiny_raw_frames):
    from wildlife_strikes.cleaning import clean_dataframe
    from wildlife_strikes.train_cli import main

    train_raw, test_raw = tiny_raw_frames
    train_clean = clean_dataframe(train_raw, has_target=True)
    test_clean = clean_dataframe(test_raw, has_target=False)

    train_p = tmp_path / "train_clean.csv"
    test_p = tmp_path / "test_clean.csv"
    out_p = tmp_path / "sub.csv"
    bundle_p = tmp_path / "bundle.joblib"

    train_clean.to_csv(train_p, index=False)
    test_clean.to_csv(test_p, index=False)

    main(
        [
            "--train",
            str(train_p),
            "--test",
            str(test_p),
            "--output",
            str(out_p),
            "--bundle",
            str(bundle_p),
            "--skip-oof",
            "--max-train-rows",
            "4",
        ]
    )

    sub = pd.read_csv(out_p)
    assert "INDEX_NR" in sub.columns and "INDICATED_DAMAGE" in sub.columns
    assert len(sub) == len(test_clean)
    assert sub["INDICATED_DAMAGE"].between(0, 1).all()
    assert bundle_p.is_file()
    assert bundle_p.with_suffix(".json").is_file()
