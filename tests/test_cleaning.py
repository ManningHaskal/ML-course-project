import pandas as pd

from wildlife_strikes.cleaning import clean_dataframe


def test_clean_preserves_rows_and_target(tiny_raw_frames):
    train, _ = tiny_raw_frames
    out = clean_dataframe(train.copy(), has_target=True)
    assert len(out) == len(train)
    assert out["INDICATED_DAMAGE"].notna().all()
    assert "TIME" not in out.columns
    assert "TIME_DECIMAL" in out.columns
    assert "REMARKS_LEN" in out.columns and "REMARKS" not in out.columns


def test_strike_count_excel_mangle_fixed(tiny_raw_frames):
    train, _ = tiny_raw_frames
    out = clean_dataframe(train.copy(), has_target=True)
    assert (out["NUM_STRUCK"] == "1-2").any()


def test_train_test_column_alignment(tiny_raw_frames):
    from wildlife_strikes.cleaning import run_clean_files
    from pathlib import Path
    import tempfile

    train, test = tiny_raw_frames
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        tp = base / "tr.csv"
        vp = base / "te.csv"
        train.to_csv(tp, index=False)
        test.to_csv(vp, index=False)
        run_clean_files(tp, vp, base / "tr_clean.csv", base / "te_clean.csv")
        tr = pd.read_csv(base / "tr_clean.csv")
        te = pd.read_csv(base / "te_clean.csv")
        fc = [c for c in tr.columns if c != "INDICATED_DAMAGE"]
        assert list(te.columns) == list(fc)
