# omiclass/train/auto_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional
import pandas as pd


@dataclass
class TaskInfo:
    n_classes: int
    is_binary: bool
    is_multiclass: bool


def infer_task_info(y: pd.Series) -> TaskInfo:
    """Infer basic task properties from the labels."""
    n_classes = int(pd.Series(y).nunique())
    is_binary = (n_classes == 2)
    is_multiclass = (n_classes > 2)
    return TaskInfo(
        n_classes=n_classes,
        is_binary=is_binary,
        is_multiclass=is_multiclass,
    )


def _missing_or_auto(params: Dict[str, Any], name: str) -> bool:
    v = params.get(name, None)
    if v is None:
        return True
    if isinstance(v, str) and v.lower() == "auto":
        return True
    return False


def _apply_lightgbm_defaults(params: Dict[str, Any], info: TaskInfo) -> Dict[str, Any]:
    """Fill LightGBM objective / metric / eval_metric / num_class iff missing/auto."""
    out = dict(params)

    if _missing_or_auto(out, "objective"):
        out["objective"] = "binary" if info.is_binary else "multiclass"

    if info.is_multiclass and _missing_or_auto(out, "num_class"):
        out["num_class"] = info.n_classes

    metric = "binary_logloss" if info.is_binary else "multi_logloss"

    if _missing_or_auto(out, "metric"):
        out["metric"] = metric

    if _missing_or_auto(out, "eval_metric"):
        out["eval_metric"] = metric

    return out


def _apply_xgboost_defaults(params: Dict[str, Any], info: TaskInfo) -> Dict[str, Any]:
    """Fill XGBoost objective / eval_metric / num_class iff missing/auto."""
    out = dict(params)

    # objective
    if _missing_or_auto(out, "objective"):
        out["objective"] = "binary:logistic" if info.is_binary else "multi:softprob"
        if info.is_multiclass and _missing_or_auto(out, "num_class"):
            out["num_class"] = info.n_classes

    # eval_metric
    if _missing_or_auto(out, "eval_metric"):
        out["eval_metric"] = "logloss" if info.is_binary else "mlogloss"

    return out


# registry of model-specific auto-config rules
_HANDLER_REGISTRY: Dict[str, Callable[[Dict[str, Any], TaskInfo], Dict[str, Any]]] = {
    "light_gbm": _apply_lightgbm_defaults,
    "weighted_light_gbm": _apply_lightgbm_defaults,
    "xgboost": _apply_xgboost_defaults,
    "weighted_xgboost": _apply_xgboost_defaults,
}


def apply_auto_config(
    model_key: str,
    init_params: Dict[str, Any],
    y: pd.Series,
) -> Dict[str, Any]:
    """
    Entry point: given a model key, raw init_params, and labels y,
    return init_params with task-based defaults applied (if any).

    - Does NOT override explicit user choices.
    - Only fills fields that are missing or set to 'auto'.
    """
    info = infer_task_info(y)
    handler: Optional[Callable[[Dict[str, Any], TaskInfo], Dict[str, Any]]] = _HANDLER_REGISTRY.get(model_key)
    if handler is None:
        return init_params

    return handler(init_params, info)
