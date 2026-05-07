# generic-classifier/models
from __future__ import annotations
from typing import Callable, Dict, Union
from .rf import build_rf_plain, build_rf_class_weighted
from .rf_smote import build_rf_smote, build_rf_adasyn, build_rf_svm_smote, build_rf_bl_smote
from .lgbm import build_lgbm_plain
from .xgboost import build_xgboost_plain
from .stacked import build_stacked_ensemble

REGISTRY: Dict[str, Callable[..., object]] = {
    "plain_random_forest": build_rf_plain,
    "weighted_random_forest": build_rf_class_weighted,
    "smote_random_forest": build_rf_smote,
    "adasyn_random_forest": build_rf_adasyn,
    "svm_smote_random_forest": build_rf_svm_smote,
    "bl_smote_random_forest": build_rf_bl_smote,

    "light_gbm": build_lgbm_plain, 
    "weighted_light_gbm": build_lgbm_plain,

    "xgboost": build_xgboost_plain, 
    "weighted_xgboost": build_xgboost_plain,
    
    "stacked_ensemble": build_stacked_ensemble,
}

def create(model_key: str, *, random_state: int, n_jobs: int):
    if model_key not in REGISTRY:
        raise KeyError(f"Unknown model key '{model_key}'. Known: {list(REGISTRY)}")
    return REGISTRY[model_key](random_state=random_state, n_jobs=n_jobs)

def get_builder(model_key: str, *, random_state: int, n_jobs: int, override_params: Union[dict, None] = None, **special,):
    if model_key not in REGISTRY:
        raise KeyError(f"Unknown model key '{model_key}'")

    def builder():
        if model_key == "stacked_ensemble":
            est = REGISTRY[model_key](random_state=random_state, n_jobs=n_jobs, **special)
        else:
            est = REGISTRY[model_key](random_state=random_state, n_jobs=n_jobs)
        if override_params:
            safe_params = _filter_override_params(est, override_params)
            if safe_params:
                est.set_params(**safe_params)
        return est

    return builder

def _filter_override_params(est, override_params: dict) -> dict:
    if not override_params:
        return {}
    valid = set(est.get_params(deep=True).keys())

    filtered = {}
    has_rf_prefix = any(k.startswith("rf__") for k in valid)

    for k, v in override_params.items():
        if k in valid:
            filtered[k] = v
            continue
        if has_rf_prefix:
            rk = f"rf__{k}"
            if rk in valid:
                filtered[rk] = v
                continue
    return filtered