"""
Clean train.csv / test.csv for the wildlife-strike damage project.

Writes train_clean.csv and test_clean.csv. Original files are unchanged.

Design:
- Strip whitespace; treat empty / common sentinel tokens as missing (before fills).
- Preserve INDEX_NR for joining predictions.
- Drop constant columns (e.g. TRANSFER); add HAS_BIRD_BAND; REMARKS/COMMENTS -> lengths.
- Fix mis-parsed strike counts (Excel date '10-Feb' -> '1-2').
- TIME (H:MM) -> TIME_DECIMAL (decimal hours); whitespace-only times -> missing.
- INCIDENT_DATE -> INCIDENT_DAY; when the date parses, reconcile MONTH/YEAR with it.
- AMA is FAA alphanumeric aircraft codes — kept as string (do not coerce to float).
- Coerce true numeric columns; categoricals -> explicit 'Unknown' for missing.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_COL = "INDEX_NR"

# Columns to coerce to float (AMA is excluded — mixed numeric + letter codes.)
NUMERIC_COLS = [
    "TIME_DECIMAL",
    "INCIDENT_DAY",
    "LATITUDE",
    "LONGITUDE",
    "AMO",
    "EMA",
    "EMO",
    "AC_MASS",
    "NUM_ENGS",
    "ENG_1_POS",
    "ENG_2_POS",
    "ENG_3_POS",
    "ENG_4_POS",
    "HEIGHT",
    "SPEED",
    "DISTANCE",
    "OUT_OF_RANGE_SPECIES",
    "REMAINS_COLLECTED",
    "REMAINS_SENT",
    "HAS_BIRD_BAND",
]

STRIKE_COUNT_COLS = ("NUM_SEEN", "NUM_STRUCK")

# Whole-cell sentinels (after strip) treated as missing for strings.
_SENTINEL_UPPER = frozenset(
    {
        "",
        ".",
        "-",
        "?",
        "NA",
        "N/A",
        "NULL",
        "NONE",
        "UNK",
        "UNKNOWN",
    }
)


def _normalize_string_series(s: pd.Series) -> pd.Series:
    """Strip; empty/whitespace-only -> NA; common placeholder tokens -> NA."""
    s = s.astype("string")
    s = s.str.strip()
    s = s.replace({"": pd.NA})
    upper = s.str.upper()
    s = s.where(~upper.isin(_SENTINEL_UPPER), pd.NA)
    return s


def _normalize_object_columns(df: pd.DataFrame, *, skip: frozenset[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in skip:
            continue
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = _normalize_string_series(out[c])
    return out


def _time_to_decimal_hours(s: pd.Series) -> pd.Series:
    """Clock time H:MM -> decimal hours, e.g. 15:30 -> 15.50. Rounded to nearest 0.01."""
    s = s.astype("string").str.strip()
    s = s.replace({"", "<NA>"}, pd.NA)
    extracted = s.str.extract(r"^(\d{1,2}):(\d{1,2})$", expand=True)
    h = pd.to_numeric(extracted[0], errors="coerce")
    m = pd.to_numeric(extracted[1], errors="coerce")
    out = h + m / 60.0
    return out.round(2)


def _parse_incident_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(s, errors="coerce")


def _fix_strike_count(s: pd.Series) -> pd.Series:
    """Map Excel-mangled '10-Feb' (likely '1-2') to a stable label."""
    s = s.astype("string")
    s = s.str.strip()
    s = s.replace({"10-Feb": "1-2"})
    s = s.replace({"<NA>", pd.NA}, np.nan)
    return s


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop = [c for c in out.columns if out[c].nunique(dropna=False) <= 1]
    return out.drop(columns=drop, errors="ignore")


def _add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, new in (("REMARKS", "REMARKS_LEN"), ("COMMENTS", "COMMENTS_LEN")):
        if col in out.columns:
            out[new] = out[col].fillna("").astype(str).str.len().astype(np.int32)
            out = out.drop(columns=[col])
    return out


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in NUMERIC_COLS:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _fill_categoricals(df: pd.DataFrame, target_col: str | None) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == target_col or c == INDEX_COL:
            continue
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].astype("string")
            out[c] = out[c].replace({"<NA>", ""}, pd.NA).fillna("Unknown")
    return out


def clean_dataframe(df: pd.DataFrame, *, has_target: bool) -> pd.DataFrame:
    out = df.copy()
    target_col = "INDICATED_DAMAGE" if has_target and "INDICATED_DAMAGE" in out.columns else None
    skip_norm: frozenset[str] = frozenset({INDEX_COL})
    if target_col:
        skip_norm = frozenset({INDEX_COL, target_col})

    out = _normalize_object_columns(out, skip=skip_norm)

    if "BIRD_BAND_NUMBER" in out.columns:
        out["HAS_BIRD_BAND"] = out["BIRD_BAND_NUMBER"].notna().astype(np.int8)
        out = out.drop(columns=["BIRD_BAND_NUMBER"])

    out = _add_text_length_features(out)
    out = _drop_constant_columns(out)

    for c in STRIKE_COUNT_COLS:
        if c in out.columns:
            out[c] = _fix_strike_count(out[c])

    if "TIME" in out.columns:
        out["TIME_DECIMAL"] = _time_to_decimal_hours(out["TIME"])
        out = out.drop(columns=["TIME"])

    if "INCIDENT_DATE" in out.columns:
        dt = _parse_incident_datetime(out["INCIDENT_DATE"])
        out["INCIDENT_DAY"] = pd.to_numeric(dt.dt.day, errors="coerce")
        ok = dt.notna()
        if ok.any():
            out.loc[ok, "INCIDENT_MONTH"] = dt.loc[ok].dt.month.to_numpy()
            out.loc[ok, "INCIDENT_YEAR"] = dt.loc[ok].dt.year.to_numpy()
        out = out.drop(columns=["INCIDENT_DATE"])

    # Integer calendar fields when present
    for c in ("INCIDENT_MONTH", "INCIDENT_YEAR"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].round().astype("Int64")

    out = _coerce_numerics(out)
    out = _fill_categoricals(out, target_col=target_col)

    if target_col and target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce").astype("Int64")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean train/test CSVs for wildlife strike project.")
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out-train", type=Path, default=Path("train_clean.csv"))
    parser.add_argument("--out-test", type=Path, default=Path("test_clean.csv"))
    args = parser.parse_args()

    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)

    train_clean = clean_dataframe(train, has_target=True)
    test_clean = clean_dataframe(test, has_target=False)

    feature_cols = [c for c in train_clean.columns if c != "INDICATED_DAMAGE"]
    missing_in_test = set(feature_cols) - set(test_clean.columns)
    extra_in_test = set(test_clean.columns) - set(feature_cols)
    if missing_in_test or extra_in_test:
        raise ValueError(f"Train/test feature mismatch: missing_in_test={missing_in_test}, extra_in_test={extra_in_test}")

    test_clean = test_clean[feature_cols]
    train_clean = train_clean[feature_cols + ["INDICATED_DAMAGE"]]

    train_clean.to_csv(args.out_train, index=False)
    test_clean.to_csv(args.out_test, index=False)

    print(f"Wrote {args.out_train} ({len(train_clean):,} rows, {train_clean.shape[1]} cols)")
    print(f"Wrote {args.out_test} ({len(test_clean):,} rows, {test_clean.shape[1]} cols)")


if __name__ == "__main__":
    main()
