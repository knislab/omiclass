from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score


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
    p = _normalize_proba_rows(np.asarray(proba, dtype=float))
    thr = np.asarray(thresholds, dtype=float)
    if p.ndim != 2:
        raise ValueError("proba must be a 2D array")
    if p.shape[1] != len(thr):
        raise ValueError("threshold count must match probability columns")
    if p.shape[1] != len(classes):
        raise ValueError("class count must match probability columns")

    thr = np.clip(thr, 1e-6, 1.0)
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
    y_true = np.asarray(getattr(cv_result, "oof_true", None))
    proba = getattr(cv_result, "oof_proba", None)
    if y_true.size == 0 or proba is None:
        return None

    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[0] != y_true.shape[0]:
        return None

    finite_rows = np.isfinite(p).all(axis=1)
    if not finite_rows.any():
        return None

    y_true = y_true[finite_rows]
    p = _normalize_proba_rows(p[finite_rows])
    classes = np.array(sorted(np.unique(y_true)))
    if p.shape[1] != len(classes):
        return None

    base_pred = classes[np.argmax(p, axis=1)]
    base_score = float(f1_score(y_true, base_pred, average=average, zero_division=0))

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
