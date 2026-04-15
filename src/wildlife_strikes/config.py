"""Paths and column names shared across the package."""

from pathlib import Path

ID_COL = "INDEX_NR"
TARGET_COL = "INDICATED_DAMAGE"

DEFAULT_TRAIN_RAW = Path("train.csv")
DEFAULT_TEST_RAW = Path("test.csv")
DEFAULT_TRAIN_CLEAN = Path("train_clean.csv")
DEFAULT_TEST_CLEAN = Path("test_clean.csv")
DEFAULT_SUBMISSION = Path("submission.csv")
