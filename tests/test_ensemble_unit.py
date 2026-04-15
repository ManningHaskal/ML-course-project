"""Fast unit tests for ensemble helpers (no heavy model training)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from wildlife_strikes.ensemble import categorical_column_names, optimize_blend


def test_optimize_blend_finds_best_weight():
    y = np.array([0, 0, 1, 1, 0, 1])
    oof_a = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    oof_b = np.array([0.4, 0.5, 0.6, 0.55, 0.45, 0.52])
    w, auc_blend = optimize_blend(y, oof_a, oof_b)
    manual = max(
        roc_auc_score(y, w_ * oof_a + (1.0 - w_) * oof_b) for w_ in np.linspace(0.0, 1.0, 201)
    )
    assert 0.0 <= w <= 1.0
    assert abs(auc_blend - manual) < 1e-9


def test_categorical_column_names_includes_object_and_bool():
    df = pd.DataFrame(
        {
            "n": [1.0, 2.0],
            "o": ["a", "b"],
            "b": [True, False],
        }
    )
    cats = categorical_column_names(df, ["n", "o", "b"])
    assert set(cats) == {"o", "b"}


@pytest.mark.parametrize("dtype", ["boolean", "bool"])
def test_categorical_column_names_bool_dtype_string(dtype):
    df = pd.DataFrame({"x": pd.array([True, False], dtype=dtype)})
    assert categorical_column_names(df, ["x"]) == ["x"]
