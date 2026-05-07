# generic-classifier/models/rf.py
from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from .base import safe_kwargs

def build_rf_plain(*, random_state: int, n_jobs: int, **cfg):
    merged = dict(random_state=random_state, n_jobs=n_jobs)
    merged.update(cfg)
    return RandomForestClassifier(**safe_kwargs(RandomForestClassifier, **merged))

def build_rf_class_weighted(*, random_state: int, n_jobs: int, **cfg):
    merged = dict(random_state=random_state, n_jobs=n_jobs, class_weight="balanced")
    merged.update(cfg)
    return RandomForestClassifier(**safe_kwargs(RandomForestClassifier, **merged))