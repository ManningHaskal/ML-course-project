"""Expensive imports only inside tests."""

import numpy as np
import pytest


@pytest.mark.slow
def test_oof_runs_on_tiny_data(tiny_raw_frames):
    from wildlife_strikes.cleaning import clean_dataframe
    from wildlife_strikes.ensemble import run_oof_ensemble

    train, _ = tiny_raw_frames
    clean = clean_dataframe(train, has_target=True)
    y = clean["INDICATED_DAMAGE"].astype(np.int32).to_numpy()
    feature_cols = [c for c in clean.columns if c not in ("INDEX_NR", "INDICATED_DAMAGE")]
    X = clean

    oof_cb, oof_lgb, meta = run_oof_ensemble(
        X,
        y,
        feature_cols=feature_cols,
        n_splits=2,
        random_state=0,
        catboost_params={"iterations": 30, "early_stopping_rounds": 5},
        lgb_params={"n_estimators": 50},
    )
    assert oof_lgb.shape == (len(y),)
    assert np.isfinite(oof_cb).all() and np.isfinite(oof_lgb).all()
    assert 0.0 <= meta.weight_catboost <= 1.0
    assert 0.0 <= meta.oof_auc_secondary <= 1.0
