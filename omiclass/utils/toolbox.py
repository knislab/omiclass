from __future__ import annotations
from dataclasses import asdict
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Any
import inspect
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

def _build_class_name_map(
    y: pd.Series, 
    config: Any 
) -> Dict[int, str]:
    # 1. Determine unique classes in data (always needed for the keys)
    classes = pd.Series(y).unique()
    classes = sorted(classes)
    
    # Initialize the default map based on the classes found in y
    default_map = {int(cls): f"class {cls}" for cls in classes}
    
    # --- Attempt to use configured class names ---
    
    project_cfg = getattr(config, "project", None)
    if project_cfg is None:
        return default_map  # Config missing 'project' -> use default

    classes_names = getattr(project_cfg, "classes_names", None)
    if not classes_names:
        return default_map 

    n_classes = len(classes)

    # Length mismatch? -> ignore map and use default
    if len(classes_names) != n_classes:
        return default_map

    # --- If all checks pass, build and return the configured map ---
    
    # Build dict mapping class index -> provided name
    class_name_map = {}
    for i, cls in enumerate(classes):
        # Use the configured name for the corresponding class index
        class_name_map[int(cls)] = str(classes_names[i])

    return class_name_map

def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # dataclass
    try:
        return asdict(obj)  # type: ignore
    except Exception:
        pass
    # pydantic v1/v2
    for attr in ("model_dump", "dict"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    return {}

def as_numpy_y(y_like):
    if hasattr(y_like, "values"):
        y = y_like.values
    else:
        y = np.asarray(y_like)
    # cast "0"/"1" to ints if possible
    if y.dtype.kind in {"U", "S", "O"}:
        try:
            return y.astype(int)
        except Exception:
            return y
    return y

def _aggregate_best_rounds(per_fold_rounds, *, method="median", pad=0.05, min_rounds=50, max_rounds: int | None = None) -> int | None:
    arr = np.array([int(r) for r in per_fold_rounds if r is not None and r > 0], dtype=int)
    if arr.size == 0:
        return None
    base = int(np.median(arr)) if method == "median" else int(np.mean(arr))
    padded = max(min_rounds, int(round(base * (1.0 + pad))))
    if max_rounds is not None:
        padded = min(padded, int(max_rounds))
    return padded

def xgb_fit_accepts_eval_metric() -> bool:
    """True if XGBClassifier.fit(...) supports eval_metric kwarg (pre-2.0)."""
    try:
        import xgboost as xgb
        sig = inspect.signature(xgb.XGBClassifier.fit)
        return "eval_metric" in sig.parameters
    except Exception:
        # Fallback to version parse
        try:
            import xgboost as xgb
            major, minor, *_ = map(int, xgb.__version__.split("."))
            return (major, minor) < (2, 0)
        except Exception:
            # Be conservative: assume not supported
            return False

def sanitize_fit_kwargs_for_xgb(model_key: str, fit_kwargs: dict) -> dict:
    if model_key == "xgboost" and not xgb_fit_accepts_eval_metric():
        fit_kwargs = dict(fit_kwargs or {})
        fit_kwargs.pop("eval_metric", None)
    return fit_kwargs

def _safe_row_normalized_cm(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.divide(cm, row_sums, where=(row_sums != 0))
    cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return cm, cm_norm

def unwrap_for_export(model):
    # 1) Calibrated -> keep as-is
    if isinstance(model, CalibratedClassifierCV):
        return model

    # 2) XGBoost custom wrapper → try to pull inner XGBClassifier
    mod = model.__class__.__module__
    if mod.startswith("omiclass.models.xgboost"):
        if isinstance(model, xgb.XGBClassifier):
            return model
        for attr in ("inner", "model", "estimator_", "clf"):
            inner = getattr(model, attr, None)
            if isinstance(inner, xgb.XGBClassifier):
                return inner
        return model  # fallback

    # 3) Everything else unchanged
    return model

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
        p = np.asarray(pvals, dtype=float)
        q = np.full_like(p, np.nan, dtype=float)

        ok = np.isfinite(p)
        if not ok.any():
            return q

        p_ok = p[ok]
        m = p_ok.size
        order = np.argsort(p_ok)
        ranked = p_ok[order]

        # BH: q_i = p_i * m / i, then monotone from end
        q_raw = ranked * m / (np.arange(1, m + 1))
        q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
        q_mono = np.clip(q_mono, 0.0, 1.0)

        q_ok = np.empty_like(p_ok)
        q_ok[order] = q_mono
        q[ok] = q_ok
        return q

def get_class_display_name(c: Any, cfg: Any = None, class_values: Optional[Sequence[Any]] = None,) -> str:
    """
    Convert class index/key to display name using project.classes_names.
    Falls back to the original value if unavailable.
    """
    classes_names = []

    if cfg:
        project_cfg = cfg.get("project", {}) if isinstance(cfg, dict) else getattr(cfg, "project", {})
        if isinstance(project_cfg, dict):
            classes_names = project_cfg.get("classes_names", []) or []
        else:
            classes_names = getattr(project_cfg, "classes_names", []) or []

    try:
        idx = int(c)

        # Detect true 1-based labels only when the full class label set is available
        is_one_based = False
        if class_values is not None and len(class_values) > 0:
            class_idxs = sorted(int(x) for x in class_values)
            is_one_based = class_idxs == list(range(1, len(class_idxs) + 1))

        if is_one_based:
            idx = idx - 1

        if 0 <= idx < len(classes_names):
            return str(classes_names[idx])

    except (TypeError, ValueError):
        pass

    c = str(c).replace("_", " ")
    return str(c)