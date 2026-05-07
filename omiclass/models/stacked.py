from __future__ import annotations
from typing import Optional, Any
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Helper stacked wrapper 
class HonestStackingClassifier(StackingClassifier):
    def fit_honest(self, X, y, Z_oof):
        # 1. Setup fitted state
        self.estimators_ = [est for _, est in self.estimators]
        first_est = self.estimators_[0]
        if hasattr(first_est, "classes_"):
            actual_labels = first_est.classes_
        elif hasattr(first_est, "estimator") and hasattr(first_est.estimator, "classes_"):
            actual_labels = first_est.estimator.classes_
        else:
            actual_labels = np.unique(y)

        n_classes = len(actual_labels)
        
        extended_names = []
        for name, _ in self.estimators:
            for lbl in actual_labels:
                extended_names.append(f"{name}_prob_{lbl}")

        if Z_oof.shape[1] != len(extended_names):
            raise ValueError(
                f"Dimension Mismatch: Z_oof has {Z_oof.shape[1]} columns, "
                f"but we expected {len(extended_names)} based on "
                f"{len(self.estimators)} models and {n_classes} classes."
            )

        Z_df = pd.DataFrame(Z_oof, index=X.index, columns=extended_names)
        
        Z_final = pd.concat([Z_df, X], axis=1) if self.passthrough else Z_df

        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(Z_final, y)
        
        self.classes_ = actual_labels
        self.stack_method_ = [self.stack_method] * len(self.estimators_)

        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(y)
        
        self._n_features_in = X.shape[1]
        if hasattr(X, "columns"):
            self._feature_names_in = np.array(X.columns, dtype=object)
            
        # Also try to set the public ones with a try-except just in case
        try:
            self.n_features_in_ = X.shape[1]
            if hasattr(X, "columns"):
                self.feature_names_in_ = np.array(X.columns, dtype=object)
        except AttributeError:
            pass

        self._base_features_in_ = {}
        for (name, est) in self.estimators:
            feats = self._get_est_feature_names(est)
            if feats is not None:
                self._base_features_in_[name] = feats

        if self.passthrough and hasattr(X, "columns"):
            self._passthrough_features_in_ = list(X.columns)


        return self
    
    def fit(self, X, y, sample_weight=None):
        if not hasattr(self, "estimators_"):
            self.estimators_ = [est for _, est in self.estimators]

        Z = self._transform(X)
        
        from sklearn.base import clone
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(Z, y)
        
        self.classes_ = np.unique(y)
        if not hasattr(self, "_label_encoder"):
            self._label_encoder = LabelEncoder().fit(y)
            
        self.stack_method_ = [self.stack_method] * len(self.estimators_)
        self._n_features_in = X.shape[1]
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
            self._feature_names_in = np.array(X.columns, dtype=object)
        
        return self
    
    def _get_est_feature_names(self, est):
        # 1) sklearn-style
        fn = getattr(est, "feature_names_in_", None)
        if fn is not None:
            return list(fn)

        # 2) xgboost sklearn wrapper
        try:
            booster = est.get_booster()
            fn = booster.feature_names 
            if fn is not None:
                return list(fn)
        except Exception:
            pass

        return None
    
    def _transform(self, X):
        if not hasattr(X, "reindex"):
            raise ValueError("HonestStackingClassifier requires pandas DataFrame for feature alignment.")

        methods = getattr(self, "stack_method_", None)
        if methods is None:
            methods = [self.stack_method] * len(self.estimators)

        preds = []
        for (name, est), meth in zip(self.estimators, methods):
            Xi = X
            feats = getattr(self, "_base_features_in_", {}).get(name)

            if feats is not None:
                Xi = X.reindex(columns=feats, fill_value=0.0)

            preds.append(getattr(est, meth)(Xi))

        Z = np.hstack(preds)

        if self.passthrough:
            Xp = X
            feats_p = getattr(self, "_passthrough_features_in_", None)
            if feats_p is not None:
                Xp = X.reindex(columns=feats_p, fill_value=0.0)

            Z = np.hstack([Z, Xp.to_numpy()])

        return Z
    
def unwrap_calibrated(est: Any):
    if not isinstance(est, CalibratedClassifierCV):
        return est

    base = getattr(est, "estimator", None)
    if base is None:
        base = getattr(est, "base_estimator", None)

    if base is None:
        ccs = getattr(est, "calibrated_classifiers_", None)
        if ccs:
            cand = getattr(ccs[0], "estimator", None)
            if cand is None:
                cand = getattr(ccs[0], "base_estimator", None)
            if cand is not None:
                base = cand

    return base if base is not None else est



def _clone_or_build(
    trained_estimator,
    *,
    registry_key: str,
    random_state: int,
    n_jobs: int,
    override_params: dict | None = None,
    prefit: bool = False,
):
    if trained_estimator is not None:
        if prefit:
            return trained_estimator 
        else:
            return clone(trained_estimator)
    else:
        from .registry import get_builder
        builder = get_builder(
            registry_key,
            random_state=random_state,
            n_jobs=n_jobs,
            override_params=override_params,
        )
        est = builder()

    if hasattr(est, "get_params") and "n_jobs" in est.get_params(deep=False):
        est.set_params(n_jobs=n_jobs)
    return est

def build_stacked_ensemble(*, random_state: int, n_jobs: int, max_iter, solver, rf_trained: Optional[object] = None, xgb_trained: Optional[object] = None, lgbm_trained: Optional[object] = None,
            rf_key: str = "plain_random_forest", xgb_key: str = "xgboost", lgbm_key: str = "light_gbm",
            cv: int = 5, passthrough: bool = True, meta: str = "logreg", override_params: dict | None = None):
    is_prefit = (cv == "prefit")
    rf = _clone_or_build(rf_trained, registry_key=rf_key, random_state=random_state, n_jobs=n_jobs, override_params=override_params, prefit=is_prefit)
    if not is_prefit:
        rf = unwrap_calibrated(rf)
    xgb = _clone_or_build(xgb_trained, registry_key=xgb_key, random_state=random_state, n_jobs=n_jobs, override_params=override_params, prefit=is_prefit)
    lgbm = _clone_or_build(lgbm_trained, registry_key=lgbm_key, random_state=random_state, n_jobs=n_jobs, override_params=override_params, prefit=is_prefit)
    
    if meta == "logreg":
        meta_est = LogisticRegression(max_iter=max_iter, solver=solver, random_state=random_state)
    else:
        raise ValueError(f"Unsupported meta='{meta}'")

    return HonestStackingClassifier(
            estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
            final_estimator=meta_est,
            cv=cv,
            n_jobs=n_jobs,
            passthrough=passthrough,
            stack_method="predict_proba",
        )