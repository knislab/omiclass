# generic-classifier/models/base.py
from __future__ import annotations
import inspect
from typing import Protocol, Optional
import numpy as np

class IModel(Protocol):
    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None, eval_set=None):
        ...
    def predict(self, X):
        ...
    def predict_proba(self, X):
        ...

def safe_kwargs(EstimatorCtor, **kwargs):
    """
    Keep only kwargs that the estimator __init__ accepts.
    Prevents crashes when config contains keys irrelevant
    to a specific estimator.
    """
    params = inspect.signature(EstimatorCtor.__init__).parameters
    allowed = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}