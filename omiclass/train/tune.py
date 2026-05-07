# generic-classifier/train/tune.py
from __future__ import annotations
from skopt.space import Integer, Real, Categorical
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def resolve_cv_folds(cfg, y: pd.Series) -> None:
    """
    Mutates cfg in-place:
      - cfg.runtime.cv_folds
      - cfg.runtime.tuning.inner_cv
      - cfg.runtime.feature_selection.inner_cv
      - (optional) calibration cv
    """
    # counts
    vc = y.value_counts(dropna=False)
    min_class = int(vc.min()) if len(vc) else 0
    n = int(len(y))

    def pick_k(requested, default_cap):
        if requested is not None:
            return int(requested)
        # auto:
        k = min(default_cap, min_class)
        k = max(2, k)
        return k

    if min_class < 2:
        raise ValueError(
            f"Not enough samples per class for CV (min_class={min_class}, n={n}). "
            "Provide holdout split or collect more data."
        )
    if cfg.runtime.cv_folds is None:
        cfg.runtime.cv_folds = pick_k(cfg.runtime.cv_folds, default_cap=5)

    if cfg.runtime.tuning.enabled and cfg.runtime.tuning.inner_cv is None:
        cfg.runtime.tuning.inner_cv = pick_k(cfg.runtime.tuning.inner_cv, default_cap=3)


def _norm_int_range(x, *, name: str) -> tuple[int, int]:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(f"{name} must be a 2-item list/tuple [lo, hi], got: {x!r}")
    lo, hi = int(x[0]), int(x[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi

def _norm_float_range(x, *, name: str) -> tuple[float, float]:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(f"{name} must be a 2-item list/tuple [lo, hi], got: {x!r}")
    lo, hi = float(x[0]), float(x[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _to_plain_dict(cfg_space: Optional[Any]) -> Dict[str, Any]:
    if cfg_space is None:
        return {}
    # Pydantic v2
    if hasattr(cfg_space, "model_dump"):
        return cfg_space.model_dump()
    # Pydantic v1
    if hasattr(cfg_space, "dict"):
        return cfg_space.dict()
    # already a dict-like
    if isinstance(cfg_space, dict):
        return cfg_space
    # last resort
    return dict(cfg_space)

def make_default_rf_search_space(X: pd.DataFrame) -> Dict[str, Any]:
    n_samples, n_features = X.shape

    # Trees: more samples → more trees
    if n_samples < 200:
        n_estimators = [100, 300]
    elif n_samples < 1000:
        n_estimators = [200, 600]
    else:
        n_estimators = [500, 1000]

    # Depth: more features → allow deeper trees
    if n_features <= 20:
        max_depth = [3, 15]
    elif n_features <= 100:
        max_depth = [5, 25]
    else:
        max_depth = [8, 35]

    # Splits: keep sensible ranges relative to sample size
    if n_samples < 500:
        min_samples_split = [2, 10]
    else:
        min_samples_split = [2, 50]

    # Leaf size: avoid leaves with 1 sample on huge datasets
    max_leaf = max(1, n_samples // 100)  # ~1% of samples
    min_samples_leaf = [1, max_leaf]

    # max_features: change behaviour for low/high dimension
    if n_features <= 30:
        max_features = ["sqrt", "log2", 0.5, 1.0]
    else:
        max_features = ["sqrt", "log2", 0.2, 0.5]

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }

def _int_or_fixed(lo: int, hi: int, *, name: str):
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        return Categorical([lo]) 
    return Integer(lo, hi)



def make_default_lgbm_param_grid(X: pd.DataFrame) -> Dict[str, Any]:
    n_samples, n_features = X.shape

    # Learning rate + estimators: standard band
    learning_rate = [0.01, 0.2]

    if n_samples < 500:
        n_estimators = [100, 600]
    elif n_samples < 5000:
        n_estimators = [200, 1500]
    else:
        n_estimators = [500, 3000]

    # num_leaves: should be << n_samples
    # cap upper bound both by n_samples and a global max
    max_leaves = min(255, max(31, n_samples // 10))
    num_leaves = [15, max_leaves]

    # Depth: -1 means "no limit" in LightGBM, we allow up to ~12 for control
    max_depth = [-1, 12]

    feature_fraction = [0.7, 1.0]
    bagging_fraction = [0.7, 1.0]

    # Child samples: control overfitting
    upper_child = min(200, max(20, n_samples // 20))
    min_child_samples = [5, upper_child]

    # Bagging frequency: 0 – no bagging; 10 – aggressive bagging
    bagging_freq = [0, 10]

    # Regularization
    lambda_l1 = [1e-4, 10.0]
    lambda_l2 = [1e-4, 10.0]

    return {
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "n_estimators": n_estimators,
        "min_child_samples": min_child_samples,
        "bagging_freq": bagging_freq,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
    }

def make_default_xgb_param_grid(X: pd.DataFrame) -> Dict[str, Any]:
    n_samples, n_features = X.shape

    # Learning rate + estimators
    learning_rate = [0.01, 0.3]

    if n_samples < 500:
        n_estimators = [100, 600]
    elif n_samples < 5000:
        n_estimators = [200, 1500]
    else:
        n_estimators = [500, 3000]

    # Depth: shallow to moderately deep trees
    max_depth = [3, 10]

    # min_child_weight: effective minimum Hessian / sample count in leaves
    # (just a broad range)
    min_child_weight = [0.1, 10.0]

    # Gamma: minimum loss reduction
    gamma = [0.0, 5.0]

    # Regularization
    reg_alpha = [0.0, 10.0]
    reg_lambda = [0.0, 10.0]

    # Row / column subsampling
    subsample = [0.5, 1.0]
    colsample_range = [0.5, 1.0]

    return {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "subsample": subsample,
        "colsample_bytree": colsample_range,
        "colsample_bylevel": colsample_range,
        "colsample_bynode": colsample_range,
    }

# 2) merge helper
def merge_user_with_defaults(user: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """User overrides defaults, but missing keys fall back to defaults."""
    merged = defaults.copy()
    for k, v in user.items():
        if v is not None:
            merged[k] = v
    return merged

# 3) public helpers that trainer calls
def build_rf_search_space(X, yaml_space, *, prefix: str = ""):
    defaults = make_default_rf_search_space(X)
    yaml_dict = _to_plain_dict(yaml_space)
    merged = merge_user_with_defaults(yaml_dict or {}, defaults)
    return rf_space_from_yaml(merged, prefix=prefix)


def build_lgbm_param_grid(X, yaml_space):
    defaults = make_default_lgbm_param_grid(X)
    yaml_dict = _to_plain_dict(yaml_space)
    merged = merge_user_with_defaults(yaml_dict or {}, defaults)
    return lighGBM_space_from_yaml(merged)

def build_xgb_param_grid(X, yaml_space):
    defaults = make_default_xgb_param_grid(X)
    yaml_dict = _to_plain_dict(yaml_space)
    merged = merge_user_with_defaults(yaml_dict or {}, defaults)
    return xgb_space_from_yaml(merged)


def rf_space_from_yaml(search_space: dict, prefix: str = "") -> dict:
    """
    Build a skopt search space from your YAML RF search_space.
    prefix="" for plain RF; prefix="rf__" for pipeline variants.
    """
    ne_lo, ne_hi = _norm_int_range(search_space["n_estimators"], name="n_estimators")
    md_lo, md_hi = _norm_int_range(search_space["max_depth"], name="max_depth")
    mss_lo, mss_hi = _norm_int_range(search_space["min_samples_split"], name="min_samples_split")
    msl_lo, msl_hi = _norm_int_range(search_space["min_samples_leaf"], name="min_samples_leaf")

    space = {
        prefix + "n_estimators": _int_or_fixed(ne_lo, ne_hi, name="n_estimators"),
        prefix + "max_depth": _int_or_fixed(md_lo, md_hi, name="max_depth"),
        prefix + "min_samples_split": _int_or_fixed(mss_lo, mss_hi, name="min_samples_split"),
        prefix + "min_samples_leaf": _int_or_fixed(msl_lo, msl_hi, name="min_samples_leaf"),
        prefix + "max_features": Categorical(search_space["max_features"]),
    }
    return space

def lighGBM_space_from_yaml(search_space: dict, prefix: str = "") -> dict:
    nl_lo, nl_hi = _norm_int_range(search_space["num_leaves"], name="num_leaves")
    md_lo, md_hi = _norm_int_range(search_space["max_depth"], name="max_depth")
    ne_lo, ne_hi = _norm_int_range(search_space["n_estimators"], name="n_estimators")
    mcs_lo, mcs_hi = _norm_int_range(search_space["min_child_samples"], name="min_child_samples")
    bf_lo, bf_hi = _norm_int_range(search_space["bagging_freq"], name="bagging_freq")
    lr_lo, lr_hi = _norm_float_range(search_space["learning_rate"], name="learning_rate")
    ff_lo, ff_hi = _norm_float_range(search_space["feature_fraction"], name="feature_fraction")
    bf_lo2, bf_hi2 = _norm_float_range(search_space["bagging_fraction"], name="bagging_fraction")
    l1_lo, l1_hi = _norm_float_range(search_space["lambda_l1"], name="lambda_l1")
    l2_lo, l2_hi = _norm_float_range(search_space["lambda_l2"], name="lambda_l2")

    space = {
        prefix + "num_leaves": _int_or_fixed(nl_lo, nl_hi, name="num_leaves"),
        prefix + "max_depth": _int_or_fixed(md_lo, md_hi, name="max_depth"),
        prefix + "n_estimators": _int_or_fixed(ne_lo, ne_hi, name="n_estimators"),
        prefix + "min_child_samples": _int_or_fixed(mcs_lo, mcs_hi, name="min_child_samples"),
        prefix + "bagging_freq": _int_or_fixed(bf_lo, bf_hi, name="bagging_freq"),
        prefix + "learning_rate": Real(lr_lo, lr_hi, prior="log-uniform"),
        prefix + "feature_fraction": Real(ff_lo, ff_hi),
        prefix + "bagging_fraction": Real(bf_lo2, bf_hi2),
        prefix + "lambda_l1": Real(l1_lo, l1_hi, prior="log-uniform"),
        prefix + "lambda_l2": Real(l2_lo, l2_hi, prior="log-uniform"),
    }
    return space

def xgb_space_from_yaml(search_space: dict, prefix: str = "") -> dict:
    """
    Build a skopt search space from your YAML XGBoost `param_grid`.
    prefix="" for plain XGB; prefix="xgb__" for pipeline variants.
    """
    ne_lo, ne_hi = _norm_int_range(search_space["n_estimators"], name="n_estimators")
    md_lo, md_hi = _norm_int_range(search_space["max_depth"], name="max_depth")
    lr_lo, lr_hi = _norm_float_range(search_space["learning_rate"], name="learning_rate")
    mcw_lo, mcw_hi = _norm_float_range(search_space["min_child_weight"], name="min_child_weight")
    g_lo, g_hi = _norm_float_range(search_space["gamma"], name="gamma")
    ra_lo, ra_hi = _norm_float_range(search_space["reg_alpha"], name="reg_alpha")
    rl_lo, rl_hi = _norm_float_range(search_space["reg_lambda"], name="reg_lambda")
    ss_lo, ss_hi = _norm_float_range(search_space["subsample"], name="subsample")
    cs_lo, cs_hi = _norm_float_range(search_space["colsample_bytree"], name="colsample_bytree")


    space = {
        prefix + "n_estimators": _int_or_fixed(ne_lo, ne_hi, name="n_estimators"),
        prefix + "max_depth": _int_or_fixed(md_lo, md_hi, name="max_depth"),
        prefix + "learning_rate": Real(lr_lo, lr_hi, prior="log-uniform"),
        prefix + "min_child_weight": Real(mcw_lo, mcw_hi, prior="log-uniform"),
        prefix + "gamma": Real(g_lo, g_hi),
        prefix + "reg_alpha": Real(ra_lo, ra_hi),
        prefix + "reg_lambda": Real(rl_lo, rl_hi),

        prefix + "subsample": Real(ss_lo, ss_hi),
        prefix + "colsample_bytree": Real(cs_lo, cs_hi),
        prefix + "colsample_bylevel": Real(cs_lo, cs_hi),
        prefix + "colsample_bynode": Real(cs_lo, cs_hi),

    }
    return space