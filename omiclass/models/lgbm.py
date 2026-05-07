# generic-classifier/models/lgbm.py
from __future__ import annotations
from .base import safe_kwargs
import lightgbm as lgb
import contextlib
import os
from typing import Any, Dict 
import sys

class _SilentLightGBMLogger:
    """Swallow all LightGBM log messages (Info + Warning)."""

    def info(self, msg: str) -> None:
        # Ignore all info-level messages
        return

    def warning(self, msg: str) -> None:
        # Ignore all warning-level messages
        return

lgb.register_logger(_SilentLightGBMLogger())

def build_lgbm_plain(*, random_state: int, n_jobs: int, **cfg):
    cfg = _canonicalize_lgbm_cfg(cfg)
    base = dict(
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,   
        verbose=-1,
    )
    merged = {**base, **cfg}
    return lgb.LGBMClassifier(**safe_kwargs(lgb.LGBMClassifier, **merged))


def _canonicalize_lgbm_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)  # work on a copy

    # alias -> canonical
    alias_map = {
        "subsample": "bagging_fraction",
        "subsample_freq": "bagging_freq",
        "colsample_bytree": "feature_fraction",
        "reg_alpha": "lambda_l1",
        "reg_lambda": "lambda_l2",
        "verbose": "verbosity",
    }

    for alias, canon in alias_map.items():
        if alias in cfg:
            # Only map if canonical not already explicitly set
            if canon not in cfg:
                cfg[canon] = cfg[alias]
            # Remove the alias key to avoid duplicates
            del cfg[alias]

    return cfg