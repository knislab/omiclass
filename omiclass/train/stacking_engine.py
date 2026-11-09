"""Nested evaluation of the existing probability + logistic-regression stack."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from omiclass.errors.errors import CVError
from omiclass.models.stacked import HonestStackingClassifier, aligned_base_probabilities
from omiclass.train.cv_engine import (
    CVResult, require_stratified_cv, inner_select_recipe, fit_fold_model,
    calibrate_within_training, _align_proba_to_canonical, _safe_prob_metrics,
)
from omiclass.train.selection import fold_selection_score


def stacking_oof(*, X, y, fit_bases, splits, seed):
    """Fit the entire base-selection procedure without each predicted row."""
    cv = require_stratified_cv(y, splits, seed, context="Stacking base OOF")
    classes = np.unique(y)
    Z = None
    scopes = []
    names = None
    for tr, va in cv.split(X, y):
        estimators, details = fit_bases(X.iloc[tr], y.iloc[tr])
        current_names = [name for name, _ in estimators]
        if names is not None and names != current_names:
            raise CVError("Stack base family/order changed between OOF folds")
        names = current_names
        block = aligned_base_probabilities(estimators, X.iloc[va], classes)
        if Z is None:
            Z = pd.DataFrame(np.nan, index=X.index, columns=block.columns)
        Z.iloc[va] = block.to_numpy()
        scopes.append(dict(train_ids=X.index[tr].tolist(), validation_ids=X.index[va].tolist(),
                           base_selection=details))
    if Z is None or not np.isfinite(Z.to_numpy()).all():
        raise CVError("Stacking OOF predictions are incomplete or nonfinite")
    return Z, scopes


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
    base_calibration: Optional[Dict[str, Dict[str, Any]]] = None,
    fit_bases: Optional[Callable] = None,
    passthrough: bool = False,
    passthrough_preprocessor: Any = None,
) -> Tuple[CVResult, np.ndarray]:
    """Evaluate complete stacks on held-out outer folds, then make final-fit OOF.

    The optional internal callback fits/selects the existing base families using
    only its arguments. Fixed base_specs retain the helper's original interface.
    Returned full-training OOF is used only for the final meta fit, never sliced
    into the outer evaluation.
    """
    if (not isinstance(X, pd.DataFrame) or not X.index.is_unique or not X.columns.is_unique
            or not isinstance(y, pd.Series) or not y.index.equals(X.index)):
        raise ValueError("Stack CV requires unique sample IDs with aligned label indices")
    outer = require_stratified_cv(y, int(outer_splits), int(seed), context="Stacking OOF evaluation")

    if fit_bases is None:
        from omiclass.train.scoped import PartitionClassifier
        scoped_specs = []
        for name, builder, space in base_specs:
            if getattr(feature_cfg, 'enabled', False) and not isinstance(builder(), PartitionClassifier):
                def scoped_builder(builder=builder):
                    return PartitionClassifier(builder(), feature_cfg=feature_cfg,
                                               seed=seed, inner_cv=tune_inner_cv)
                builder, space = scoped_builder, {'estimator__' + key: value for key, value in space.items()}
            scoped_specs.append((name, builder, space))

        def fit_bases(values, labels):
            estimators = []
            for name, builder, space in scoped_specs:
                columns, params, _, _ = inner_select_recipe(
                    model_builder=builder, Xtr=values, ytr=labels, tune_space=space,
                    tune_iter=tune_iter, tune_scoring=tune_scoring, tune_inner_cv=tune_inner_cv,
                    seed=seed, n_jobs=n_jobs, feature_cfg=feature_cfg)
                used = values.loc[:, columns] if columns is not None else values
                calibration = (base_calibration or {}).get(name, {})
                if calibration.get('calibrate', False):
                    estimator = builder().set_params(**params)
                    estimator = calibrate_within_training(
                        fitted_or_builder=estimator, Xtr_used=used, ytr=labels,
                        method=calibration.get('method', 'sigmoid'), cv=outer_splits,
                        seed=seed, ensemble=calibration.get('ensemble', False))
                else:
                    estimator = fit_fold_model(model_builder=builder, Xtr_used=used,
                                               ytr=labels, best_params=params)
                estimators.append((name, estimator))
            return estimators, {}

    canonical = np.unique(y)
    predictions = np.empty(len(y), dtype=np.asarray(y).dtype)
    probabilities = np.full((len(y), len(canonical)), np.nan)
    metrics = []
    for fold, (tr, va) in enumerate(outer.split(X, y), 1):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Ztr, scopes = stacking_oof(X=Xtr, y=ytr, fit_bases=fit_bases,
                                   splits=outer_splits, seed=seed)
        estimators, details = fit_bases(Xtr, ytr)
        stack = HonestStackingClassifier(estimators=estimators, final_estimator=meta_builder(),
            cv='prefit', stack_method='predict_proba', passthrough=passthrough, n_jobs=n_jobs)
        if passthrough and passthrough_preprocessor is not None:
            stack.passthrough_preprocessor_ = clone(passthrough_preprocessor).fit(Xtr, ytr)
        stack.fit_honest(Xtr, ytr, Ztr)
        pred = stack.predict(X.iloc[va])
        proba = _align_proba_to_canonical(stack.predict_proba(X.iloc[va]), stack.classes_, canonical)
        predictions[va], probabilities[va] = pred, proba
        row = dict(fold=fold, f1_macro=float(f1_score(y.iloc[va], pred, average='macro')),
            accuracy=float(accuracy_score(y.iloc[va], pred)),
            train_ids=Xtr.index.tolist(), validation_ids=X.index[va].tolist(),
            stacking_oof_scopes=scopes, base_selection=details,
            probability_columns=list(stack.probability_columns_),
            meta_feature_columns=list(stack.meta_feature_columns_),
            passthrough=bool(passthrough))
        row.update(_safe_prob_metrics(y.iloc[va].to_numpy(), proba, canonical))
        row['selection_score'] = fold_selection_score(tune_scoring, stack, X.iloc[va], y.iloc[va], row)
        metrics.append(row)

    Z_final, final_scopes = stacking_oof(X=X, y=y, fit_bases=fit_bases, splits=outer_splits, seed=seed)
    result = CVResult(oof_true=y.to_numpy(), oof_pred=predictions, oof_proba=probabilities,
                      per_fold_metrics=metrics, fold_recipes=[], classes=canonical)
    result.stacking_final_oof_scopes = final_scopes
    result.stacking_probability_columns = list(Z_final.columns)
    return result, Z_final.to_numpy()
