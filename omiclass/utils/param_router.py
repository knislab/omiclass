from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from ..utils.toolbox import xgb_fit_accepts_eval_metric

@dataclass(frozen=True)
class ModelParamSpec:
    allowed_init: tuple[str, ...]
    allowed_fit: tuple[str, ...]
    aliases: Dict[str, str] = None 

    def alias(self, d: Dict[str, Any]) -> Dict[str, Any]:
        if not self.aliases:
            return dict(d)
        return {self.aliases.get(k, k): v for k, v in d.items()}

# LightGBM (sklearn API)
LGBM_SPEC = ModelParamSpec(
    allowed_init=(
        "num_leaves","max_depth","learning_rate","n_estimators",
        "min_child_samples","feature_fraction","bagging_fraction","bagging_freq",
        "lambda_l1","lambda_l2","verbosity","random_state","n_jobs",
        "seed","feature_fraction_seed","bagging_seed","data_random_seed","drop_seed",
        "deterministic","force_col_wise",
        "objective","metric","num_class","class_weight",
    ),
    # we pass these to fit()
    allowed_fit=("eval_metric","early_stopping_rounds"),
)

# XGBoost (sklearn API)
XGB_SPEC = ModelParamSpec(
    allowed_init=(
        "learning_rate","n_estimators","max_depth","min_child_weight","gamma",
        "reg_alpha","reg_lambda","subsample","colsample_bytree","colsample_bylevel",
        "colsample_bynode","verbosity","random_state","n_jobs","nthread",
        "seed","objective","num_class","eval_metric","scale_pos_weight",
    ),
    allowed_fit=("eval_metric","early_stopping_rounds"),
)

# RandomForest (and RF+samplers wrappers use RF core params)
RF_SPEC = ModelParamSpec(
    allowed_init=(
        "n_estimators","max_depth","min_samples_split","min_samples_leaf",
        "max_features","oob_score","class_weight","random_state","n_jobs","verbose"
    ),
    allowed_fit=(),
)

SPECS: Dict[str, ModelParamSpec] = {
    "light_gbm": LGBM_SPEC,
    "weighted_light_gbm": LGBM_SPEC,

    "xgboost":   XGB_SPEC,
    "weighted_xgboost": XGB_SPEC,

    "plain_random_forest":   RF_SPEC,
    "weighted_random_forest":RF_SPEC,
    "smote_random_forest":   RF_SPEC,
    "adasyn_random_forest":  RF_SPEC,
    "svm_smote_random_forest": RF_SPEC,
    "bl_smote_random_forest":  RF_SPEC,
}

def normalize_eval_metric(model_key: str, fit_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # Keep a guard in case someone slips mlogloss under LGBM
    if "light_gbm" in model_key:
        if fit_kwargs.get("eval_metric") == "mlogloss":
            fit_kwargs["eval_metric"] = "multi_logloss"
    return fit_kwargs

def split_init_and_fit_kwargs(model_key: str, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    spec = SPECS[model_key]
    aliased = spec.alias(raw)
    init_params = {k: v for k, v in aliased.items() if k in spec.allowed_init}
    fit_kwargs  = {k: v for k, v in aliased.items() if k in spec.allowed_fit}
    fit_kwargs  = normalize_eval_metric(model_key, fit_kwargs)
    if "xgboost" in model_key and "eval_metric" in fit_kwargs and not xgb_fit_accepts_eval_metric():
        # On XGB >= 2.0, move eval_metric from fit() to estimator params
        init_params["eval_metric"] = fit_kwargs.pop("eval_metric")
    return init_params, fit_kwargs
