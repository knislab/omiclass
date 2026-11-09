"""Cloneable fitting boundary around the existing classifier and selection rules."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class PartitionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, preprocessor=None, feature_cfg=None,
                 subset_fraction=None, seed=80, inner_cv=3, rank_space=None,
                 rank_iterations=0, scoring='f1_macro'):
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.feature_cfg = feature_cfg
        self.subset_fraction = subset_fraction
        self.seed = seed
        self.inner_cv = inner_cv
        self.rank_space = rank_space
        self.rank_iterations = rank_iterations
        self.scoring = scoring

    def fit(self, X, y, **fit_kwargs):
        from omiclass.train.cv_engine import _safe_stratified_kfold, _feature_rank_by_importance, _candidate_feature_sizes
        self.fit_ids_ = X.index.tolist()
        y = pd.Series(np.asarray(y), index=X.index)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.preprocessor_ = clone(self.preprocessor) if self.preprocessor is not None else None
        current = self.preprocessor_.fit_transform(X, y) if self.preprocessor_ is not None else X
        self.selected_features_ = list(current.columns)
        if self.subset_fraction is not None and self.feature_cfg is not None:
            cv = _safe_stratified_kfold(y, self.inner_cv, self.seed)
            if cv != 0:
                rank_estimator = clone(self.estimator)
                if self.rank_space is not None and self.rank_iterations > 0:
                    # Preserve preliminary Bayesian tuning for importance ranking,
                    # now nested inside this fit's permitted training population.
                    from skopt import BayesSearchCV
                    preliminary = PartitionClassifier(clone(self.estimator), self.preprocessor,
                        seed=self.seed, inner_cv=self.inner_cv)
                    search = BayesSearchCV(preliminary,
                        {'estimator__' + key: value for key, value in self.rank_space.items()},
                        n_iter=self.rank_iterations, scoring=self.scoring, cv=cv, n_jobs=1,
                        random_state=self.seed, refit=False, error_score='raise')
                    search.fit(X, y, **fit_kwargs)
                    rank_estimator.set_params(**{key.removeprefix('estimator__'): value
                                                 for key, value in search.best_params_.items()})
                ranking = _feature_rank_by_importance(lambda: clone(rank_estimator), current, y, {},
                    list(cv.split(current, y)), reduce_sampler_pipelines=self.feature_cfg.reduce_sampler_pipelines)
                if ranking is not None:
                    cfg = self.feature_cfg.model_copy(update={'subset_fracs': [self.subset_fraction]})
                    size = _candidate_feature_sizes(cfg, current.shape[1])[0]
                    self.selected_features_ = current.columns.to_numpy()[ranking[:size]].tolist()
        self.estimator_ = clone(self.estimator)
        kwargs = {k.removeprefix('estimator__'): v for k, v in fit_kwargs.items()}
        self.estimator_.fit(current.loc[:, self.selected_features_], y, **kwargs)
        self.classes_ = self.estimator_.classes_
        return self

    def prepare(self, X):
        current = self.preprocessor_.transform(X) if self.preprocessor_ is not None else X
        return current.loc[:, self.selected_features_]

    def predict(self, X):
        return self.estimator_.predict(self.prepare(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.prepare(X))


def fitted_partitions(model):
    if isinstance(model, PartitionClassifier):
        return [model]
    candidates = [getattr(item, 'estimator', getattr(item, 'base_estimator', None))
                  for item in getattr(model, 'calibrated_classifiers_', [])]
    return [item for item in candidates if isinstance(item, PartitionClassifier)]
