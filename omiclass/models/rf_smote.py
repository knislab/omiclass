# generic-classifier/models/rf_smote.py
from __future__ import annotations
import logging
from imblearn.pipeline import Pipeline
from ..models.rf import build_rf_plain
from ..sampling.smote import SafeSMOTE
from ..sampling.adasyn import SafeADASYN
from ..sampling.svmsmote import SafeSVMSMOTE
from ..sampling.blsmote import SafeBorderlineSMOTE

_LOG = logging.getLogger(__name__)
_RF_SMOTE_N_JOBS_LOGGED = False

def _pipe(sampler, *, random_state: int, n_jobs: int, **rf_cfg):
    global _RF_SMOTE_N_JOBS_LOGGED
    if not _RF_SMOTE_N_JOBS_LOGGED:
        _LOG.info(
            "SMOTE-family RandomForest estimator-level n_jobs is forced to 1 for "
            "reproducibility; runtime n_jobs may still be used for safe outer-level orchestration."
        )
        _RF_SMOTE_N_JOBS_LOGGED = True
    return Pipeline(steps=[
        ("sampler", sampler),
        ("rf", build_rf_plain(random_state=random_state, n_jobs=1, **rf_cfg)),
    ])

def build_rf_smote(*, random_state: int, n_jobs: int, **rf_cfg):
    return _pipe(SafeSMOTE(random_state=random_state), random_state=random_state, n_jobs=n_jobs, **rf_cfg)

def build_rf_adasyn(*, random_state: int, n_jobs: int, **rf_cfg):
    return _pipe(SafeADASYN(random_state=random_state), random_state=random_state, n_jobs=n_jobs, **rf_cfg)

def build_rf_svm_smote(*, random_state: int, n_jobs: int, **rf_cfg):
    return _pipe(SafeSVMSMOTE(random_state=random_state), random_state=random_state, n_jobs=n_jobs, **rf_cfg)

def build_rf_bl_smote(*, random_state: int, n_jobs: int, **rf_cfg):
    return _pipe(SafeBorderlineSMOTE(kind="borderline-1", random_state=random_state), random_state=random_state, n_jobs=n_jobs, **rf_cfg)
