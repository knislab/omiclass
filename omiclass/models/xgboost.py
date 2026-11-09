# generic-classifier/models/xgboost.py
from __future__ import annotations
import xgboost as xgb
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from .base import safe_kwargs

_ESTOP_KEYS = ("early_stopping_rounds", "early_stopping")

class XGBNoEarlyStop(xgb.XGBClassifier):
    def set_params(self, **params):
        # kill anytime CV/search tries to set it
        for k in _ESTOP_KEYS:
            params.pop(k, None)
        return super().set_params(**params)


class FitLocalWeightedXGB(XGBNoEarlyStop):
    """The existing weighted XGBoost policy, calculated for each fit's labels.

    Keeping this at the classifier boundary also covers sklearn search clones,
    importance ranking, calibration folds, and stacking base fits. The legacy
    XGBNoEarlyStop class remains unchanged for historical artifacts.
    """

    def fit(self, X, y, *, sample_weight=None, **kwargs):
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        if len(classes) < 2:
            raise ValueError("Weighted XGBoost requires at least two training classes")
        if len(classes) == 2:
            # XGBClassifier encodes binary labels as 0 and 1; class 1 is positive.
            if not np.array_equal(classes, [0, 1]):
                raise ValueError("Binary XGBoost training labels must be encoded as 0 and 1")
            self.set_params(scale_pos_weight=float(counts[0] / counts[1]))
        else:
            self.set_params(scale_pos_weight=None)
            balanced = compute_sample_weight("balanced", y)
            sample_weight = balanced if sample_weight is None else balanced * np.asarray(sample_weight)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)

def build_xgboost_plain(*, random_state: int, n_jobs: int, **cfg):
    merged = dict(cfg)
    merged.update(random_state=random_state, seed=random_state, n_jobs=1, nthread=1)
    # Remove early-stopping keys BEFORE construction
    for k in _ESTOP_KEYS:
        merged.pop(k, None)
    clean_kwargs = safe_kwargs(xgb.XGBClassifier, **merged)
    return XGBNoEarlyStop(**clean_kwargs)


def build_xgboost_weighted(*, random_state: int, n_jobs: int, **cfg):
    plain = build_xgboost_plain(random_state=random_state, n_jobs=n_jobs, **cfg)
    return FitLocalWeightedXGB(**plain.get_params(deep=False))
