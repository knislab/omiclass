# generic-classifier/models/xgboost.py
from __future__ import annotations
import xgboost as xgb
import inspect
from .base import safe_kwargs

_ESTOP_KEYS = ("early_stopping_rounds", "early_stopping")

class XGBNoEarlyStop(xgb.XGBClassifier):
    def set_params(self, **params):
        # kill anytime CV/search tries to set it
        for k in _ESTOP_KEYS:
            params.pop(k, None)
        return super().set_params(**params)

def build_xgboost_plain(*, random_state: int, n_jobs: int, **cfg):
    merged = dict(random_state=random_state, n_jobs=n_jobs, nthread=n_jobs)
    merged.update(cfg)
    # Remove early-stopping keys BEFORE construction
    for k in _ESTOP_KEYS:
        merged.pop(k, None)
    clean_kwargs = safe_kwargs(xgb.XGBClassifier, **merged)
    return XGBNoEarlyStop(**clean_kwargs)