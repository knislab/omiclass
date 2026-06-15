# generic-classifier/models/rf.py
from __future__ import annotations
import logging
from sklearn.ensemble import RandomForestClassifier
from .base import safe_kwargs

_LOG = logging.getLogger(__name__)
_RF_N_JOBS_LOGGED = False


def _rf_base_params(random_state: int, **cfg):
    global _RF_N_JOBS_LOGGED
    if not _RF_N_JOBS_LOGGED:
        _LOG.info(
            "RandomForest estimator-level n_jobs is forced to 1 for reproducibility; "
            "runtime n_jobs may still be used for safe outer-level orchestration."
        )
        _RF_N_JOBS_LOGGED = True
    merged = dict(cfg)
    merged.update(random_state=random_state, n_jobs=1)
    return merged


def build_rf_plain(*, random_state: int, n_jobs: int, **cfg):
    merged = _rf_base_params(random_state, **cfg)
    return RandomForestClassifier(**safe_kwargs(RandomForestClassifier, **merged))

def build_rf_class_weighted(*, random_state: int, n_jobs: int, **cfg):
    merged = _rf_base_params(random_state, **cfg)
    if merged.get("class_weight") is None:
        merged["class_weight"] = "balanced"
    return RandomForestClassifier(**safe_kwargs(RandomForestClassifier, **merged))
