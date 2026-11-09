# omiclass/train/stacking_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss, roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import get_scorer

from skopt import BayesSearchCV
from omiclass.train.cv_engine import nested_oof_cv, CVResult, _safe_stratified_kfold
from omiclass.train.selection import fold_selection_score

def train_oof_stacker(
    *,
    base_specs: List[Tuple[str, Callable[[], Any], Dict[str, Any]]],
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int,
    seed: int,
    tune_iter: int,
    tune_scoring: str,
    tune_inner_cv: int,
    n_jobs: int,
    feature_cfg: Any,
    meta_builder: Callable[[], Any],
) -> Tuple[CVResult, np.ndarray]:
    """
    Returns:
      - CVResult for the meta-learner evaluated via OOF stacking
      - Z_oof: the meta features (concatenated base OOF probas) aligned to TRAIN rows
    """
    base_oof_probas: List[np.ndarray] = []
    base_fold_logs: Dict[str, CVResult] = {}

    for name, builder, space in base_specs:
        cvres = nested_oof_cv(
            model_builder=builder,
            X=X,
            y=y,
            outer_splits=outer_splits,
            seed=seed,
            tune_space=space,
            tune_iter=tune_iter,
            tune_scoring=tune_scoring,
            tune_inner_cv=tune_inner_cv,
            n_jobs=n_jobs,
            feature_cfg=feature_cfg,
            eval_calibrated_in_cv=False,
            calib_method="",
            calib_cv=0,
            calibration_ensemble=False,
        )
        base_fold_logs[name] = cvres
        if cvres.oof_proba is None:
            raise ValueError(f"Base model '{name}' has no predict_proba; cannot stack safely.")
        base_oof_probas.append(cvres.oof_proba)

    Z_oof = np.concatenate(base_oof_probas, axis=1)

    # Now evaluate meta-learner with outer CV using Z_oof as fixed features 
    outer = _safe_stratified_kfold(y, desired_splits=int(outer_splits), seed=int(seed))
    if outer == 0:
        meta = meta_builder()
        meta.fit(Z_oof, y)
        pred = meta.predict(Z_oof)
        proba = meta.predict_proba(Z_oof) if hasattr(meta, "predict_proba") else None
        return (
            CVResult(
                oof_true=y.to_numpy(),
                oof_pred=np.asarray(pred),
                oof_proba=np.asarray(proba) if proba is not None else None,
                per_fold_metrics=[],
                fold_recipes=[],
            ),
            Z_oof,
        )

    n = len(y)
    oof_pred = np.full(n, -1, dtype=int)
    oof_proba = None
    per_fold = []

    for fold_idx, (tr_idx, va_idx) in enumerate(outer.split(Z_oof, y), start=1):
        Ztr, Zva = Z_oof[tr_idx], Z_oof[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        meta = meta_builder()
        meta.fit(Ztr, ytr)

        pred = meta.predict(Zva)
        oof_pred[va_idx] = pred

        if hasattr(meta, "predict_proba"):
            proba = meta.predict_proba(Zva)
            if oof_proba is None:
                oof_proba = np.zeros((n, proba.shape[1]), dtype=float)
            oof_proba[va_idx] = proba

        per_fold.append({
            "fold": fold_idx,
            "f1_macro": float(f1_score(yva, pred, average="macro")),
            "accuracy": float(accuracy_score(yva, pred)),
        })
        per_fold[-1]["selection_score"] = fold_selection_score(
            tune_scoring, meta, Zva, yva, per_fold[-1]
        )

    return (
        CVResult(
            oof_true=y.to_numpy(),
            oof_pred=oof_pred,
            oof_proba=oof_proba,
            per_fold_metrics=per_fold,
            fold_recipes=[],
        ),
        Z_oof,
    )
