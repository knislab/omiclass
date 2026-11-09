# omiclass/train/cv_engine.py
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
from omiclass.train.selection import fold_selection_score


def _bulkformer_representation_from_model(model):
    candidates = [model]
    candidates.extend(
        getattr(item, "estimator", None)
        for item in (getattr(model, "calibrated_classifiers_", None) or [])
    )
    for candidate in candidates:
        if candidate is None:
            continue
        named = getattr(candidate, "named_steps", {})
        representation = named.get("bulkformer_representation")
        if representation is not None:
            return representation
    return None


# ---------- data structures ----------

@dataclass
class FoldRecipe:
    outer_fold: int
    selected_cols: Optional[List[str]]
    best_params: Dict[str, Any]
    fs_best_score: Optional[float] = None
    fs_scoring: Optional[str] = None


@dataclass
class CVResult:
    oof_true: np.ndarray
    oof_pred: np.ndarray
    oof_proba: Optional[np.ndarray]
    per_fold_metrics: List[Dict[str, Any]]
    fold_recipes: List[FoldRecipe]


# ---------- safe CV helpers ----------

def _safe_stratified_kfold(y: Union[pd.Series, np.ndarray], desired_splits: int, seed: int) -> Union[StratifiedKFold, int]:
    y = pd.Series(y)
    min_count = int(y.value_counts().min())
    if min_count < 2:
        return 0
    n_splits = max(2, min(int(desired_splits), min_count))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))


def _normalize_proba_rows(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s


def _align_proba_to_canonical(
    proba: np.ndarray,
    fitted_classes: Sequence[Any],
    canonical_classes: Sequence[Any],
) -> np.ndarray:
    fitted_classes = list(fitted_classes)
    canonical_classes = list(canonical_classes)
    out = np.zeros((proba.shape[0], len(canonical_classes)), dtype=float)
    for j, cls in enumerate(canonical_classes):
        if cls in fitted_classes:
            src = fitted_classes.index(cls)
            out[:, j] = proba[:, src]
    return out


def _safe_prob_metrics(y_true: np.ndarray, proba_aligned: Optional[np.ndarray], classes: np.ndarray) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"logloss": None, "brier": None, "auroc_macro": None, "auprc_macro": None}
    if proba_aligned is None:
        return out

    y_true = np.asarray(y_true)
    p = np.asarray(proba_aligned, dtype=float)

    if p.ndim != 2 or p.shape[0] != y_true.shape[0] or p.shape[1] != len(classes):
        return out

    p = _normalize_proba_rows(p)

    try:
        out["logloss"] = float(log_loss(y_true, p, labels=list(classes)))
    except Exception:
        out["logloss"] = None

    try:
        if p.shape[1] == 2:
            pos_prob = p[:, 1]
            y01 = (y_true == classes[1]).astype(float)
            out["brier"] = float(np.mean((pos_prob - y01) ** 2))
    except Exception:
        out["brier"] = None

    try:
        if len(np.unique(y_true)) < 2:
            return out

        if p.shape[1] == 2:
            pos_prob = p[:, 1]
            out["auroc_macro"] = float(roc_auc_score(y_true, pos_prob))
            out["auprc_macro"] = float(average_precision_score((y_true == classes[1]).astype(int), pos_prob))
        else:
            out["auroc_macro"] = float(roc_auc_score(y_true, p, multi_class="ovr", average="macro"))
            Y = (y_true[:, None] == classes[None, :]).astype(int)
            out["auprc_macro"] = float(average_precision_score(Y, p, average="macro"))
    except Exception:
        out["auroc_macro"] = None
        out["auprc_macro"] = None

    return out

# ---------- feature selection ----------

def _feature_rank_by_importance(
    model_builder: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    tuned_params: Dict[str, Any],
    inner_splits: List[Tuple[np.ndarray, np.ndarray]],
    reduce_sampler_pipelines: bool = True,
) -> Optional[np.ndarray]:
    # Rank features using mean feature_importances_ across inner folds (fold-train only).
    imps: List[np.ndarray] = []
    for tr_idx, _va_idx in inner_splits:
        Xtr = X.iloc[tr_idx]
        ytr = y.iloc[tr_idx]

        est = model_builder()
        named_steps = getattr(est, "named_steps", {})
        sampler = named_steps.get("sampler")
        if (
            not reduce_sampler_pipelines
            and sampler is not None
            and hasattr(sampler, "fit_resample")
        ):
            # Historical compatibility: the old implementation did not look
            # through imbalanced-learn pipelines for feature_importances_.
            return None
        if tuned_params:
            try:
                est.set_params(**tuned_params)
            except Exception:
                return None

        est.fit(Xtr, ytr)

        importance_estimator = est
        # Walk through sklearn/imblearn pipelines to the fitted classifier.
        while not hasattr(importance_estimator, "feature_importances_"):
            named_steps = getattr(importance_estimator, "named_steps", None)
            if not named_steps:
                break
            importance_estimator = list(named_steps.values())[-1]
        if not hasattr(importance_estimator, "feature_importances_"):
            return None
        imp = np.asarray(importance_estimator.feature_importances_, dtype=float)
        if imp.shape[0] != X.shape[1]:
            return None
        imps.append(imp)

    if not imps:
        return None

    mean_imp = np.mean(imps, axis=0)
    ranked_idx = np.argsort(-mean_imp, kind="mergesort")
    return ranked_idx


def _candidate_feature_sizes(feature_cfg: Any, n_features_total: int) -> List[int]:
    subset_fracs = list(getattr(feature_cfg, "subset_fracs", [0.1, 0.2, 0.3]))
    min_features = int(getattr(feature_cfg, "min_features", 10))
    max_features_cfg = getattr(feature_cfg, "max_features", None)
    max_features = int(max_features_cfg) if max_features_cfg is not None else n_features_total

    sizes = set()
    for frac in subset_fracs:
        try:
            f = float(frac)
        except Exception:
            continue
        if f <= 0:
            continue
        k = int(round(f * n_features_total))
        k = max(min_features, min(k, max_features))
        if 1 <= k <= n_features_total:
            sizes.add(k)

    out = sorted(sizes)
    if not out:
        out = [min(n_features_total, max(min_features, 1))]
    return out

# ---------- inner selection ----------

def inner_select_recipe(
    *,
    model_builder: Callable[[], Any],
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    tune_space: Dict[str, Any],
    tune_iter: int,
    tune_scoring: str,
    tune_inner_cv: int,
    seed: int,
    n_jobs: int,
    feature_cfg: Any,
) -> Tuple[Optional[List[str]], Dict[str, Any], Optional[float], Optional[str]]:
    inner_cv = _safe_stratified_kfold(ytr, desired_splits=int(tune_inner_cv), seed=int(seed))
    if inner_cv == 0 or int(tune_iter) <= 0:
        return None, {}, None, None

    splits = list(inner_cv.split(Xtr, ytr))

    fs_enabled = bool(getattr(feature_cfg, "enabled", False)) if feature_cfg is not None else False
    if not fs_enabled:
        # Tune directly on all features
        base = model_builder()
        search = BayesSearchCV(
            estimator=base,
            search_spaces=tune_space,
            n_iter=int(tune_iter),
            scoring=tune_scoring,
            cv=splits,
            n_jobs=1,
            random_state=int(seed),
            refit=False,
            verbose=0,
            error_score="raise",
        )
        search.fit(Xtr, ytr)
        return None, dict(search.best_params_ or {}), float(search.best_score_), tune_scoring

    n_total = Xtr.shape[1]
    candidate_sizes = _candidate_feature_sizes(feature_cfg, n_total)

    base0 = model_builder()
    search0 = BayesSearchCV(
        estimator=base0,
        search_spaces=tune_space,
        n_iter=max(5, min(10, int(tune_iter))), 
        scoring=tune_scoring,
        cv=splits,
        n_jobs=1,
        random_state=int(seed),
        refit=False,
        verbose=0,
        error_score="raise",
    )
    search0.fit(Xtr, ytr)
    tuned_for_rank = dict(search0.best_params_ or {})

    ranked_idx = _feature_rank_by_importance(
        model_builder,
        Xtr,
        ytr,
        tuned_for_rank,
        splits,
        reduce_sampler_pipelines=bool(
            getattr(feature_cfg, "reduce_sampler_pipelines", True)
        ),
    )
    if ranked_idx is None:
        # If no importances exist, fallback: tune on all features
        base = model_builder()
        search = BayesSearchCV(
            estimator=base,
            search_spaces=tune_space,
            n_iter=int(tune_iter),
            scoring=tune_scoring,
            cv=splits,
            n_jobs=1,
            random_state=int(seed),
            refit=False,
            verbose=0,
            error_score="raise",
        )
        search.fit(Xtr, ytr)
        return None, dict(search.best_params_ or {}), float(search.best_score_), tune_scoring

    ranked_cols = Xtr.columns.to_numpy()[ranked_idx]

    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    best_cols: Optional[List[str]] = None

    for k in candidate_sizes:
        cols_k = list(ranked_cols[:k])
        Xk = Xtr[cols_k]

        base = model_builder()
        search = BayesSearchCV(
            estimator=base,
            search_spaces=tune_space,
            n_iter=int(tune_iter),
            scoring=tune_scoring,
            cv=splits,
            n_jobs=1,
            random_state=int(seed),
            refit=False,
            verbose=0,
            error_score="raise",
        )
        search.fit(Xk, ytr)

        score_k = float(search.best_score_)
        if score_k > best_score:
            best_score = score_k
            best_params = dict(search.best_params_ or {})
            best_cols = cols_k

    return best_cols, best_params, float(best_score), tune_scoring

# ---------- fold fit + optional calibration ----------

def fit_fold_model(
    *,
    model_builder: Callable[[], Any],
    Xtr_used: pd.DataFrame,
    ytr: pd.Series,
    best_params: Dict[str, Any],
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    est = model_builder()
    if best_params:
        est.set_params(**best_params)
    est.fit(Xtr_used, ytr, **(fit_kwargs or {}))
    return est


def calibrate_within_training(
    *,
    fitted_or_builder: Any,
    Xtr_used: pd.DataFrame,
    ytr: pd.Series,
    method: str,
    cv: int,
    seed: int,
    use_prefit: bool = False,
    ensemble: bool = False,
) -> Any:
    inner = _safe_stratified_kfold(ytr, desired_splits=int(cv), seed=int(seed))
    if inner == 0:
        if use_prefit:
            return fitted_or_builder
        fitted_or_builder.fit(Xtr_used, ytr)
        return fitted_or_builder

    if use_prefit:
        cal = CalibratedClassifierCV(fitted_or_builder, method=str(method), cv="prefit")
        cal.fit(Xtr_used, ytr)
        return cal

    cal = CalibratedClassifierCV(
        fitted_or_builder, method=str(method), cv=inner, ensemble=bool(ensemble)
    )
    cal.fit(Xtr_used, ytr)
    return cal

# ---------- overall CV engine ----------

def nested_oof_cv(
    *,
    model_builder: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int,
    seed: int,
    tune_space: Dict[str, Any],
    tune_iter: int,
    tune_scoring: str,
    tune_inner_cv: int,
    n_jobs: int,
    feature_cfg: Any,
    eval_calibrated_in_cv: bool,
    calib_method: str,
    calib_cv: int,
    calibration_ensemble: bool = False,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> CVResult:
    outer = _safe_stratified_kfold(y, desired_splits=int(outer_splits), seed=int(seed))
    if outer == 0:
        est = model_builder()
        est.fit(X, y, **(fit_kwargs or {}))
        pred = est.predict(X)
        proba = est.predict_proba(X) if hasattr(est, "predict_proba") else None
        return CVResult(
            oof_true=y.to_numpy(),
            oof_pred=np.asarray(pred),
            oof_proba=np.asarray(proba) if proba is not None else None,
            per_fold_metrics=[],
            fold_recipes=[],
        )

    canonical = np.array(sorted(pd.unique(y)))
    n = len(y)
    oof_pred = np.full(n, -1, dtype=int)
    oof_proba = np.full((n, len(canonical)), np.nan, dtype=float)

    per_fold_metrics: List[Dict[str, Any]] = []
    fold_recipes: List[FoldRecipe] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(outer.split(X, y), start=1):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        selected_cols, best_params, fs_best_score, fs_scoring = inner_select_recipe(
            model_builder=model_builder,
            Xtr=Xtr,
            ytr=ytr,
            tune_space=tune_space,
            tune_iter=int(tune_iter),
            tune_scoring=str(tune_scoring),
            tune_inner_cv=int(tune_inner_cv),
            seed=int(seed),
            n_jobs=int(n_jobs),
            feature_cfg=feature_cfg,
        )

        if selected_cols is not None:
            Xtr_used = Xtr[selected_cols]
            Xva_used = Xva[selected_cols]
        else:
            Xtr_used = Xtr
            Xva_used = Xva

        if eval_calibrated_in_cv:
            base = model_builder()
            if best_params:
                base.set_params(**best_params)
            model = calibrate_within_training(
                fitted_or_builder=base,
                Xtr_used=Xtr_used,
                ytr=ytr,
                method=calib_method,
                cv=int(calib_cv),
                seed=int(seed),
                use_prefit=False,
                ensemble=bool(calibration_ensemble),
            )
        else:
            fold_fit_kwargs = dict(fit_kwargs or {})
            for weight_key in [key for key in fold_fit_kwargs if key.endswith("sample_weight")]:
                fold_fit_kwargs[weight_key] = np.asarray(fold_fit_kwargs[weight_key])[tr_idx]

            model = fit_fold_model(
                model_builder=model_builder,
                Xtr_used=Xtr_used,
                ytr=ytr,
                best_params=best_params,
                fit_kwargs=fold_fit_kwargs,
            )
        pred = model.predict(Xva_used)
        oof_pred[va_idx] = pred

        prob_metrics = {"logloss": None, "brier": None, "auroc_macro": None, "auprc_macro": None}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xva_used)
            fitted_classes = getattr(model, "classes_", canonical)
            aligned = _align_proba_to_canonical(np.asarray(proba), fitted_classes, canonical)
            oof_proba[va_idx, :] = aligned
            prob_metrics = _safe_prob_metrics(yva.to_numpy(), aligned, canonical)

        m = {
            "fold": fold_idx,
            "f1_macro": float(f1_score(yva, pred, average="macro")),
            "accuracy": float(accuracy_score(yva, pred)),
            "n_features": int(Xtr_used.shape[1]),
            "selected_features": list(selected_cols) if selected_cols is not None else list(X.columns),
            "fs_best_score": fs_best_score,
            "fs_scoring": fs_scoring,
        }
        representation = _bulkformer_representation_from_model(model)
        if representation is not None:
            m["bulkformer_reduction_strategy"] = representation.strategy
            m["bulkformer_input_dimensions"] = int(representation.n_features_in_)
            m["bulkformer_output_dimensions"] = int(representation.effective_n_features_)
            selector = getattr(representation, "selector_", None)
            if selector is not None:
                m["bulkformer_selected_dimensions"] = list(selector.selected_features_)
                m["bulkformer_close_pairs"] = list(selector.selected_close_pairs_)
        m.update(prob_metrics)
        m["selection_score"] = fold_selection_score(tune_scoring, model, Xva_used, yva, m)
        per_fold_metrics.append(m)

        fold_recipes.append(
            FoldRecipe(
                outer_fold=fold_idx,
                selected_cols=list(selected_cols) if selected_cols is not None else None,
                best_params=dict(best_params or {}),
                fs_best_score=fs_best_score,
                fs_scoring=fs_scoring,
            )
        )

    if not np.isfinite(oof_proba).any():
        oof_proba_out = None
    else:
        oof_proba_out = oof_proba

    return CVResult(
        oof_true=y.to_numpy(),
        oof_pred=oof_pred,
        oof_proba=oof_proba_out,
        per_fold_metrics=per_fold_metrics,
        fold_recipes=fold_recipes,
    )
