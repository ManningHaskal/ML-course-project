"""
Out-of-fold CatBoost + LightGBM, convex blend on OOF predictions, full retrain for submission.

Tabular competition SOTA: gradient boosting with native categoricals (CatBoost + LightGBM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def categorical_column_names(X: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Non-numeric feature columns (CatBoost cat_features)."""
    cats: list[str] = []
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            cats.append(c)
        elif str(X[c].dtype) in ("boolean", "bool"):
            cats.append(c)
    return cats


def prepare_lgbm_categories(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = X.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


@dataclass
class BlendResult:
    weight_catboost: float
    oof_auc_blend: float
    oof_auc_catboost: float
    oof_auc_secondary: float


def _try_import_lightgbm():
    """LightGBM requires OpenMP on macOS (`brew install libomp`)."""
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        return LGBMClassifier, early_stopping, log_evaluation
    except OSError as e:
        logger.warning(
            "LightGBM not loadable (%s). Using a second CatBoost variant for diversity. "
            "For full ensemble: brew install libomp",
            e,
        )
        return None


def _params_for_sample_size(
    n_rows: int,
    *,
    cb_base: dict,
    lgb_base: dict,
    cb_secondary: dict,
) -> tuple[dict, dict, dict]:
    """Relax row subsampling / min_child_samples so boosting runs on small train sets (tests, tiny folds)."""
    cb = dict(cb_base)
    lgb = dict(lgb_base)
    sec = dict(cb_secondary)
    if n_rows < 64:
        cb["subsample"] = 1.0
        sec["subsample"] = 1.0
        lgb["subsample"] = 1.0
        lgb["colsample_bytree"] = 1.0
        lgb["min_child_samples"] = max(1, min(lgb.get("min_child_samples", 35), n_rows // 2 or 1))
    return cb, lgb, sec


def optimize_blend(y_true: np.ndarray, oof_cb: np.ndarray, oof_lgb: np.ndarray) -> tuple[float, float]:
    best_w, best_auc = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 201):
        p = w * oof_cb + (1.0 - w) * oof_lgb
        auc = roc_auc_score(y_true, p)
        if auc > best_auc:
            best_auc, best_w = auc, w
    return best_w, best_auc


def run_oof_ensemble(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    feature_cols: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    catboost_params: dict | None = None,
    lgb_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, BlendResult]:
    from catboost import CatBoostClassifier

    lgb_imports = _try_import_lightgbm()

    cat_cols = categorical_column_names(X, feature_cols)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_cb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))

    cb_base = dict(
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=150,
        random_seed=random_state,
        verbose=False,
        auto_class_weights="Balanced",
        task_type="CPU",
        allow_writing_files=False,
        border_count=254,
        subsample=0.85,
    )
    if catboost_params:
        cb_base.update(catboost_params)

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    lgb_base = dict(
        objective="binary",
        learning_rate=0.03,
        n_estimators=5000,
        num_leaves=96,
        max_depth=12,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    if lgb_params:
        lgb_base.update(lgb_params)

    cb_secondary = dict(
        iterations=2000,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=150,
        random_seed=random_state + 999,
        verbose=False,
        auto_class_weights="Balanced",
        task_type="CPU",
        allow_writing_files=False,
        border_count=128,
        subsample=0.9,
    )
    if catboost_params:
        cb_secondary.update({k: v for k, v in catboost_params.items() if k in ("iterations", "early_stopping_rounds")})

    X_feat = X[feature_cols]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_feat, y)):
        X_tr, X_va = X_feat.iloc[tr_idx], X_feat.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        n_tr = len(X_tr)
        cb_fold, lgb_fold, sec_fold = _params_for_sample_size(
            n_tr, cb_base=cb_base, lgb_base=lgb_base, cb_secondary=cb_secondary
        )

        cb = CatBoostClassifier(**cb_fold)
        cb.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va), use_best_model=True)
        oof_cb[va_idx] = cb.predict_proba(X_va)[:, 1]

        if lgb_imports:
            LGBMClassifier, early_stopping, log_evaluation = lgb_imports
            X_tr_l = prepare_lgbm_categories(X_tr, cat_cols)
            X_va_l = prepare_lgbm_categories(X_va, cat_cols)

            lgbm = LGBMClassifier(**lgb_fold)
            lgbm.fit(
                X_tr_l,
                y_tr,
                eval_set=[(X_va_l, y_va)],
                eval_metric="auc",
                categorical_feature="auto",
                callbacks=[
                    early_stopping(stopping_rounds=150, verbose=False),
                    log_evaluation(period=0),
                ],
            )
            oof_lgb[va_idx] = lgbm.predict_proba(X_va_l)[:, 1]
        else:
            cb_b = CatBoostClassifier(**sec_fold)
            cb_b.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va), use_best_model=True)
            oof_lgb[va_idx] = cb_b.predict_proba(X_va)[:, 1]

        logger.info("Fold %d/%d complete", fold + 1, n_splits)

    auc_cb = roc_auc_score(y, oof_cb)
    auc_lgb = roc_auc_score(y, oof_lgb)
    w, auc_blend = optimize_blend(y, oof_cb, oof_lgb)

    meta = BlendResult(
        weight_catboost=w,
        oof_auc_blend=auc_blend,
        oof_auc_catboost=auc_cb,
        oof_auc_secondary=auc_lgb,
    )
    logger.info(
        "OOF ROC-AUC: primary=%.5f secondary=%.5f | blend(w=%.3f)=%.5f",
        auc_cb,
        auc_lgb,
        w,
        auc_blend,
    )
    return oof_cb, oof_lgb, meta


def fit_full_models(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    feature_cols: list[str],
    random_state: int = 42,
    catboost_params: dict | None = None,
    lgb_params: dict | None = None,
):
    from catboost import CatBoostClassifier

    lgb_imports = _try_import_lightgbm()

    cat_cols = categorical_column_names(X, feature_cols)
    X_feat = X[feature_cols]

    cb_base = dict(
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
        auto_class_weights="Balanced",
        task_type="CPU",
        allow_writing_files=False,
        border_count=254,
        subsample=0.85,
    )
    if catboost_params:
        cb_base.update(catboost_params)

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    lgb_base = dict(
        objective="binary",
        learning_rate=0.03,
        n_estimators=5000,
        num_leaves=96,
        max_depth=12,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    if lgb_params:
        lgb_base.update(lgb_params)

    cb_secondary_full = dict(
        iterations=2000,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state + 999,
        verbose=False,
        auto_class_weights="Balanced",
        task_type="CPU",
        allow_writing_files=False,
        border_count=128,
        subsample=0.9,
    )

    n_full = len(X_feat)
    cb_fit, lgb_fit, sec_fit = _params_for_sample_size(
        n_full, cb_base=cb_base, lgb_base=lgb_base, cb_secondary=cb_secondary_full
    )

    cb = CatBoostClassifier(**cb_fit)
    cb.fit(X_feat, y, cat_features=cat_cols)

    if lgb_imports:
        LGBMClassifier, _, _ = lgb_imports
        X_l = prepare_lgbm_categories(X_feat, cat_cols)
        lgbm = LGBMClassifier(**lgb_fit)
        lgbm.fit(X_l, y, categorical_feature="auto")
        secondary = lgbm
    else:
        secondary = CatBoostClassifier(**sec_fit)
        secondary.fit(X_feat, y, cat_features=cat_cols)

    return cb, secondary, cat_cols


def predict_test_blend(
    cb_model,
    secondary_model,
    X_test: pd.DataFrame,
    *,
    feature_cols: list[str],
    cat_cols: list[str],
    blend_weight: float,
) -> np.ndarray:
    from catboost import CatBoostClassifier

    Xt = X_test[feature_cols]
    p_cb = cb_model.predict_proba(Xt)[:, 1]
    if isinstance(secondary_model, CatBoostClassifier):
        p_sec = secondary_model.predict_proba(Xt)[:, 1]
    else:
        Xt_l = prepare_lgbm_categories(Xt, cat_cols)
        p_sec = secondary_model.predict_proba(Xt_l)[:, 1]
    return blend_weight * p_cb + (1.0 - blend_weight) * p_sec
