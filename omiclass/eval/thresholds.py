from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score
from .prediction import validate_probabilities


def ordered_thresholds(info, classes):
    values = info.get('thresholds', info)
    saved_classes = list(info.get('classes', classes))
    if len(saved_classes) != len(classes) or set(saved_classes) != set(classes):
        raise ValueError('Threshold class schema differs from fitted model classes')
    try:
        mapping = {label: float(values[str(label)] if str(label) in values else values[label])
                   for label in saved_classes}
        result = np.asarray([mapping[label] for label in classes])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Threshold values must cover every fitted class') from error
    if not np.isfinite(result).all() or (result <= 0).any() or (result > 1).any():
        raise ValueError('Threshold values must be finite and within (0, 1]')
    return result


def _normalize_proba_rows(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s


def apply_multiclass_thresholds(
    proba: np.ndarray,
    thresholds: Sequence[float],
    classes: Sequence[Any],
) -> np.ndarray:
    """
    Convert class probabilities to predictions using per-class thresholds.

    A lower threshold makes a class easier to select; prediction is the class
    with the largest probability/threshold ratio.
    """
    p = validate_probabilities(proba, classes, len(proba))
    thr = np.asarray(thresholds, dtype=float)
    if p.ndim != 2:
        raise ValueError("proba must be a 2D array")
    if p.shape[1] != len(thr):
        raise ValueError("threshold count must match probability columns")
    if p.shape[1] != len(classes):
        raise ValueError("class count must match probability columns")

    if not np.isfinite(thr).all() or (thr <= 0).any() or (thr > 1).any():
        raise ValueError('Threshold values must be finite and within (0, 1]')
    idx = np.argmax(p / thr[None, :], axis=1)
    return np.asarray(classes)[idx]


def learn_multiclass_thresholds(
    cv_result: Any,
    *,
    average: str = "macro",
    grid: Optional[Sequence[float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Learn per-class probability thresholds from OOF predictions.

    The thresholds are optimized only on CV out-of-fold probabilities and can
    then be applied to an untouched holdout set.
    """
    raw_truth = getattr(cv_result, "oof_true", None)
    if raw_truth is None:
        return None
    y_true = np.asarray(raw_truth)
    proba = getattr(cv_result, "oof_proba", None)
    if y_true.size == 0 or proba is None:
        return None

    stored_classes = getattr(cv_result, 'classes', None)
    # Historical CVResult columns were generated in sorted training-label order.
    classes = np.asarray(stored_classes if stored_classes is not None else np.unique(y_true))
    if len(classes) < 2 or set(classes) != set(y_true) or len(set(classes)) != len(classes):
        raise ValueError('OOF threshold optimization requires the complete training class schema')
    p = _normalize_proba_rows(validate_probabilities(proba, classes, len(y_true)))
    base_pred = getattr(cv_result, 'oof_pred', None)
    if base_pred is None:
        base_pred = classes[np.argmax(p, axis=1)]
    if np.asarray(base_pred).shape != y_true.shape or not set(base_pred).issubset(set(classes)):
        raise ValueError('Invalid OOF estimator predictions')
    base_score = float(f1_score(y_true, base_pred, labels=classes, average=average, zero_division=0))

    candidates = (
        np.asarray(list(grid), dtype=float)
        if grid is not None
        else np.linspace(0.05, 0.95, 19, dtype=float)
    )
    candidates = np.unique(np.clip(candidates, 1e-6, 1.0))

    thresholds = np.full(len(classes), 0.5, dtype=float)
    best_score = float(f1_score(
        y_true,
        apply_multiclass_thresholds(p, thresholds, classes),
        average=average,
        zero_division=0,
    ))

    # Small coordinate search: stable, deterministic, and cheap for typical K.
    improved = True
    passes = 0
    while improved and passes < 3:
        improved = False
        passes += 1
        for j in range(len(classes)):
            local_best = thresholds[j]
            local_score = best_score
            for candidate in candidates:
                trial = thresholds.copy()
                trial[j] = candidate
                pred = apply_multiclass_thresholds(p, trial, classes)
                score = float(f1_score(y_true, pred, average=average, zero_division=0))
                if score > local_score:
                    local_score = score
                    local_best = candidate
            if local_score > best_score:
                thresholds[j] = local_best
                best_score = local_score
                improved = True

    info = {
        "method": "per_class_probability_ratio",
        "source": "OOF training predictions",
        "optimized_metric": f"f1_{average}",
        "classes": classes.tolist(),
        "thresholds": {
            str(cls): float(thr)
            for cls, thr in zip(classes.tolist(), thresholds.tolist())
        },
        "baseline_oof_f1": base_score,
        "optimized_oof_f1": best_score,
        "grid": [float(x) for x in candidates.tolist()],
    }
    try:
        cv_result.thresholds = info
        cv_result.threshold_optimization = info
    except Exception:
        pass
    return info
