from __future__ import annotations
from typing import Optional, Any
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
import pandas as pd
import numpy as np


def probability_columns(names, classes):
    # Retain both columns for binary classification, as in the existing stack.
    return [f"{name}_prob_{label}" for name in names for label in classes]


def aligned_base_probabilities(estimators, X, classes):
    """One probability layout for OOF generation and deployed stack inference."""
    from omiclass.train.scoped import fitted_partitions
    if not estimators:
        raise ValueError("Stack probability layout requires fitted base estimators")
    blocks = []
    for name, estimator in estimators:
        features = HonestStackingClassifier._get_est_feature_names(estimator)
        Xi = X
        if features is not None and not fitted_partitions(estimator):
            missing = [feature for feature in features if feature not in X.columns]
            if missing:
                raise ValueError(f"Stack base '{name}' lacks fitted input features: {missing[:5]}")
            Xi = X.loc[:, features]
        fitted_classes = list(estimator.classes_)
        if len(fitted_classes) != len(classes) or set(fitted_classes) != set(classes):
            raise ValueError(f"Stack base '{name}' has incompatible fitted classes")
        proba = np.asarray(estimator.predict_proba(Xi), dtype=float)
        if proba.shape != (len(X), len(fitted_classes)) or not np.isfinite(proba).all():
            raise ValueError(f"Stack base '{name}' returned invalid probabilities")
        blocks.append(proba[:, [fitted_classes.index(label) for label in classes]])
    return pd.DataFrame(np.concatenate(blocks, axis=1), index=X.index,
                        columns=probability_columns([name for name, _ in estimators], classes))


# Helper stacked wrapper 
class HonestStackingClassifier(StackingClassifier):
    def fit_honest(self, X, y, Z_oof):
        if not isinstance(X, pd.DataFrame) or not X.index.is_unique or not X.columns.is_unique:
            raise ValueError("Stack fitting requires unique DataFrame sample IDs and feature names")
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError("Stack labels must follow the input sample index")
        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_
        if len(self.classes_) < 2:
            raise ValueError("Stack fitting requires at least two classes")
        names = [name for name, _ in self.estimators]
        if not names or len(names) != len(set(names)):
            raise ValueError("Stack base names must be nonempty and unique")
        self.estimators_ = [est for _, est in self.estimators]
        for estimator in self.estimators_:
            check_is_fitted(estimator)
            if set(estimator.classes_) != set(self.classes_):
                raise ValueError("Stack base classes must match the training classes")
        self.named_estimators_ = Bunch(**dict(self.estimators))
        self.probability_columns_ = probability_columns(names, self.classes_)
        if isinstance(Z_oof, pd.DataFrame):
            if not Z_oof.index.equals(X.index) or list(Z_oof.columns) != self.probability_columns_:
                raise ValueError("Stack OOF sample index or probability-column layout mismatch")
        values = np.asarray(Z_oof, dtype=float)
        if values.shape != (len(X), len(self.probability_columns_)) or not np.isfinite(values).all():
            raise ValueError("Stack OOF probabilities have invalid shape or nonfinite values")
        Z_df = pd.DataFrame(values, index=X.index, columns=self.probability_columns_)
        prep = getattr(self, 'passthrough_preprocessor_', None)
        if self.passthrough:
            X_pass = prep.transform(X) if prep is not None else X
            self._passthrough_features_in_ = list(X_pass.columns)
            self.passthrough_columns_ = [f"passthrough_{i}_{name}" for i, name in enumerate(X_pass.columns)]
            Z_df = pd.concat([Z_df, X_pass.set_axis(self.passthrough_columns_, axis=1)], axis=1)
        self.meta_feature_columns_ = list(Z_df.columns)
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(Z_df, self._label_encoder.transform(y))
        self.stack_method_ = ['predict_proba'] * len(self.estimators_)
        self._n_features_in = X.shape[1]
        self.feature_names_in_ = self._feature_names_in = np.asarray(X.columns, dtype=object)
        self._base_features_in_ = {name: self._get_est_feature_names(est) for name, est in self.estimators}
        self._stack_schema_version_ = 1
        return self
    
    def fit(self, X, y, sample_weight=None):
        raise ValueError("Stack fitting requires training-only OOF probabilities; use Trainer or fit_honest")

    @property
    def n_features_in_(self):
        if getattr(self, '_stack_schema_version_', None) == 1:
            return self._n_features_in
        return super().n_features_in_
    
    @staticmethod
    def _get_est_feature_names(est):
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
        if getattr(self, '_stack_schema_version_', None) != 1:
            return self._legacy_transform(X)
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("Stack prediction requires a DataFrame with unique feature names")
        Z = aligned_base_probabilities(self.estimators, X, self.classes_)
        if self.passthrough:
            prep = getattr(self, 'passthrough_preprocessor_', None)
            current = prep.transform(X) if prep is not None else X
            missing = set(self._passthrough_features_in_) - set(current.columns)
            if missing:
                raise ValueError(f"Stack passthrough lacks fitted features: {sorted(missing, key=str)[:5]}")
            current = current.loc[:, self._passthrough_features_in_]
            Z = pd.concat([Z, current.set_axis(self.passthrough_columns_, axis=1)], axis=1)
        return Z.loc[:, self.meta_feature_columns_]

    def _legacy_transform(self, X):
        # Historical saved models retain their original ordering/zero-fill path.
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
            prep = getattr(self, 'passthrough_preprocessor_', None)
            Xp = prep.transform(X) if prep is not None else X
            feats_p = getattr(self, "_passthrough_features_in_", None)
            if feats_p is not None:
                Xp = Xp.reindex(columns=feats_p, fill_value=0.0)

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
