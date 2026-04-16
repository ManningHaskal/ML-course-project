"""Fast unit tests for ensemble helpers (no heavy model training)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from wildlife_strikes.ensemble import categorical_column_names, optimize_binary_threshold, optimize_blend


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


def test_optimize_binary_threshold_returns_valid_threshold():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    p = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95], dtype=float)
    thr, score = optimize_binary_threshold(y, p, grid_size=201, metric="balanced_accuracy")
    assert 0.0 <= thr <= 1.0
    assert score == pytest.approx(1.0)


def test_optimize_binary_threshold_accuracy_metric():
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    p = np.array([0.1, 0.3, 0.7, 0.9], dtype=float)
    _, score = optimize_binary_threshold(y, p, metric="accuracy")
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", ["boolean", "bool"])
def test_categorical_column_names_bool_dtype_string(dtype):
    df = pd.DataFrame({"x": pd.array([True, False], dtype=dtype)})
    assert categorical_column_names(df, ["x"]) == ["x"]
