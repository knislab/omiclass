"""Shared fitted-class and feature contracts for evaluation and inference."""
import numpy as np
import pandas as pd


def canonical_classes(model):
    classes = np.asarray(getattr(model, 'classes_', []))
    if classes.ndim != 1 or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('Prediction requires a fitted, unique classifier class schema')
    return classes


def validate_probabilities(proba, classes, n_samples):
    p = np.asarray(proba, dtype=float)
    if len(classes) == 2:
        if p.ndim == 1:
            p = np.column_stack([1 - p, p])
        elif p.ndim == 2 and p.shape[1] == 1:
            p = np.column_stack([1 - p[:, 0], p[:, 0]])
    if (p.shape != (n_samples, len(classes)) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('Invalid probabilities for the fitted class schema')
    return p


def align_probabilities(proba, source_classes, target_classes):
    source, target = list(source_classes), list(target_classes)
    if len(source) != len(target) or set(source) != set(target):
        raise ValueError('Probability class schemas differ')
    return np.asarray(proba)[:, [source.index(label) for label in target]]


def match_features(X, model):
    """Order known bare-estimator features; fitted pipelines own their transforms."""
    from omiclass.train.scoped import fitted_partitions
    from omiclass.train.cv_engine import _bulkformer_representation_from_model
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError('Prediction requires a DataFrame with unique feature names')
    if fitted_partitions(model) or getattr(model, '_stack_schema_version_', None) == 1:
        return X
    # Historical stacks retain their inference implementation, but the public
    # prediction/evaluation boundary must not silently create required inputs.
    if hasattr(model, 'fit_honest'):
        for name, estimator in model.estimators:
            match_features(X, estimator)
            expected = getattr(model, '_base_features_in_', {}).get(name)
            if expected is not None and set(expected) - set(X.columns):
                raise ValueError(f'Prediction lacks required features for stack base {name}')
        prep = getattr(model, 'passthrough_preprocessor_', None)
        if model.passthrough and prep is not None:
            prep.transform(X)
        elif model.passthrough:
            missing = set(getattr(model, '_passthrough_features_in_', [])) - set(X.columns)
            if missing:
                raise ValueError(f'Prediction lacks required passthrough features: {sorted(missing, key=str)[:5]}')
        return X
    representation = _bulkformer_representation_from_model(model)
    if representation is not None:
        expected = list(representation.feature_names_in_)
        missing = set(expected) - set(X.columns)
        extra = set(X.columns) - set(expected)
        if missing or (extra and getattr(representation, 'strategy', 'full') != 'full'):
            raise ValueError(f'Embedding schema mismatch; missing={sorted(missing, key=str)[:5]}, extra={sorted(extra, key=str)[:5]}')
        return X.loc[:, expected]
    expected = getattr(model, 'feature_names_in_', None)
    if expected is None:
        expected = getattr(model, '_feature_names_in', None)
    if expected is None:
        expected = getattr(model, 'feature_name_', None)
    if expected is None and hasattr(model, 'get_booster'):
        expected = model.get_booster().feature_names
    if expected is not None:
        missing = set(expected) - set(X.columns)
        if missing:
            raise ValueError(f'Prediction lacks required features: {sorted(missing, key=str)[:5]}')
        return X.loc[:, list(expected)]
    width = getattr(model, 'n_features_in_', None)
    if width is not None and X.shape[1] != width:
        raise ValueError(f'Prediction feature count differs from fitted width {width}')
    return X


def predict_estimator(model, X):
    """Keep estimator decisions, including SVC decisions that differ from argmax."""
    classes = canonical_classes(model)
    pred = np.asarray(model.predict(X))
    if pred.shape != (len(X),) or not set(pred).issubset(set(classes)):
        raise ValueError('Predictions differ from the fitted class schema')
    p = (validate_probabilities(model.predict_proba(X), classes, len(X))
         if hasattr(model, 'predict_proba') else None)
    return pred, p
