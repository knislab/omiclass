# generic-classifier/models/rf_smote.py
from __future__ import annotations
from imblearn.pipeline import Pipeline
from ..models.rf import build_rf_plain
from ..sampling.smote import SafeSMOTE
from ..sampling.adasyn import SafeADASYN
from ..sampling.svmsmote import SafeSVMSMOTE
from ..sampling.blsmote import SafeBorderlineSMOTE

def _pipe(sampler, *, random_state: int, n_jobs: int, **rf_cfg):
    return Pipeline(steps=[
        ("sampler", sampler),
        ("rf", build_rf_plain(random_state=random_state, n_jobs=n_jobs, **rf_cfg)),
    ])

def build_rf_smote(*, random_state: int, n_jobs: int, **rf_cfg):
    return _pipe(SafeSMOTE(random_state=random_state), random_state=random_state, n_jobs=n_jobs, **rf_cfg)

def build_rf_adasyn(*, random_state: int, n_jobs: int, **rf_cfg):
    return _pipe(SafeADASYN(random_state=random_state), random_state=random_state, n_jobs=n_jobs, **rf_cfg)

def build_rf_svm_smote(*, random_state: int, n_jobs: int, **rf_cfg):
    return _pipe(SafeSVMSMOTE(random_state=random_state), random_state=random_state, n_jobs=n_jobs, **rf_cfg)

def build_rf_bl_smote(*, random_state: int, n_jobs: int, **rf_cfg):
    return _pipe(SafeBorderlineSMOTE(kind="borderline-1", random_state=random_state), random_state=random_state, n_jobs=n_jobs, **rf_cfg)
