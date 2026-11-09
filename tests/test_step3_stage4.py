"""Synthetic leakage, schema and compatibility checks for the existing stack."""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from skopt.space import Integer

from omiclass.config.loaders import load_config
from omiclass.config.schema import FeatureSelectionCfg
from omiclass.errors.errors import CVError
from omiclass.models.stacked import (
    HonestStackingClassifier, aligned_base_probabilities, probability_columns,
)
from omiclass.train.cv_engine import CVResult, inner_select_recipe, calibrate_within_training
from omiclass.train.preprocessing import TrainingPreprocessor
from omiclass.train.scoped import PartitionClassifier
from omiclass.train.stacking_engine import stacking_oof, train_oof_stacker
from omiclass.train.trainer import Trainer


def data(counts=(24, 24)):
    X = pd.DataFrame(np.random.RandomState(18).normal(size=(sum(counts), 6)),
        index=[f's{i}' for i in range(sum(counts))], columns=[f'g{i}' for i in range(6)])
    y = pd.Series(np.repeat(np.arange(len(counts)), counts), index=X.index)
    return X, y


def config(tmp_path, **values):
    return load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label',
        cv_folds=2, bayesian_iterations=0, feature_selection_enabled=False,
        log_transform=False, **values))


def rf():
    return RandomForestClassifier(n_estimators=3, max_depth=2, random_state=80)


def engine_args(X, y, **values):
    result = dict(base_specs=[('rf', rf, {})], X=X, y=y, outer_splits=2, seed=80,
        tune_iter=0, tune_scoring='neg_log_loss', tune_inner_cv=2, n_jobs=1,
        feature_cfg=None, meta_builder=lambda: LogisticRegression(max_iter=1000, random_state=80))
    result.update(values)
    return result


class ReverseClassesRF(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.model_ = rf().fit(X, y)
        self.classes_ = self.model_.classes_[::-1]
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, ::-1]

    def predict(self, X):
        return self.model_.predict(X)


@pytest.mark.parametrize('passthrough', [False, True])
def test_nested_stack_matches_manual_replay_and_uses_fresh_training_oof(passthrough):
    X, y = data()
    kwargs = engine_args(X, y, passthrough=passthrough)
    result, final_oof = train_oof_stacker(**kwargs)
    final_expected = np.full((len(y), 2), np.nan)
    for fold, (tr, va) in zip(result.per_fold_metrics,
            StratifiedKFold(2, shuffle=True, random_state=80).split(X, y)):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        local = np.full((len(tr), 2), np.nan)
        for inner_tr, inner_va in StratifiedKFold(2, shuffle=True, random_state=80).split(Xtr, ytr):
            model = rf().fit(Xtr.iloc[inner_tr], ytr.iloc[inner_tr])
            local[inner_va] = model.predict_proba(Xtr.iloc[inner_va])
        outer_model = rf().fit(Xtr, ytr)
        Zva = outer_model.predict_proba(X.iloc[va])
        if passthrough:
            local = np.column_stack([local, Xtr])
            Zva = np.column_stack([Zva, X.iloc[va]])
        meta = kwargs['meta_builder']().fit(local, ytr)
        np.testing.assert_allclose(result.oof_proba[va], meta.predict_proba(Zva))
        np.testing.assert_array_equal(result.oof_pred[va], meta.predict(Zva))
        assert fold['selection_score'] == pytest.approx(get_scorer('neg_log_loss')(meta, Zva, y.iloc[va]))
        assert len(fold['meta_feature_columns']) == 2 + (6 if passthrough else 0)
        for scope in fold['stacking_oof_scopes']:
            assert set(scope['train_ids']).isdisjoint(scope['validation_ids'])
            assert set(scope['train_ids'] + scope['validation_ids']) == set(Xtr.index)
            assert set(scope['train_ids'] + scope['validation_ids']).isdisjoint(X.index[va])
        final_expected[va] = outer_model.predict_proba(X.iloc[va])
    np.testing.assert_allclose(final_oof, final_expected)


def test_outer_validation_label_changes_cannot_change_that_folds_predictions(monkeypatch):
    import omiclass.train.stacking_engine as engine
    X, y = data()
    fixed = list(StratifiedKFold(2, shuffle=True, random_state=80).split(X, y))
    original = engine.require_stratified_cv
    def split(labels, desired_splits, seed, *, context):
        if context == 'Stacking OOF evaluation':
            return SimpleNamespace(split=lambda *args: iter(fixed))
        return original(labels, desired_splits, seed, context=context)
    monkeypatch.setattr(engine, 'require_stratified_cv', split)
    seen = []
    def fit_bases(values, labels):
        seen.append((values.index.tolist(), labels.to_numpy().copy()))
        return [('rf', rf().fit(values, labels))], {}
    original_result, _ = train_oof_stacker(**engine_args(X, y, fit_bases=fit_bases))
    changed = y.copy()
    changed.iloc[fixed[0][1]] = 1 - changed.iloc[fixed[0][1]]
    changed_result, _ = train_oof_stacker(**engine_args(X, changed, fit_bases=fit_bases))
    np.testing.assert_array_equal(original_result.oof_proba[fixed[0][1]],
                                  changed_result.oof_proba[fixed[0][1]])
    assert all(len(ids) < len(X) for ids, _ in seen)


@pytest.mark.parametrize('counts', [(18, 18), (18, 18, 18)])
@pytest.mark.parametrize('passthrough', [False, True])
def test_probability_order_per_base_schemas_and_serialized_prediction(tmp_path, counts, passthrough):
    X, y = data(counts)
    y = y * 2 + 2  # Also exercise the meta learner's label encoding.
    bases = [('rf', rf().fit(X[['g0', 'g1']], y)),
             ('xgb', ReverseClassesRF().fit(X[['g2', 'g3']], y)),
             ('lgbm', rf().fit(X[['g4', 'g5']], y))]
    classes = np.unique(y)
    Z = aligned_base_probabilities(bases, X, classes)
    np.testing.assert_allclose(Z.iloc[:, len(counts):2 * len(counts)],
                              bases[1][1].model_.predict_proba(X[['g2', 'g3']]))
    stack = HonestStackingClassifier(estimators=bases, final_estimator=LogisticRegression(max_iter=1000),
                                     cv='prefit', stack_method='predict_proba', passthrough=passthrough)
    stack.fit_honest(X, y, Z)
    assert stack.probability_columns_ == probability_columns(['rf', 'xgb', 'lgbm'], classes)
    assert stack.n_features_in_ == X.shape[1]
    assert list(stack.transform(X).columns) == stack.meta_feature_columns_
    np.testing.assert_array_equal(stack.predict(X),
        stack._label_encoder.inverse_transform(stack.final_estimator_.predict(stack.transform(X))))
    np.testing.assert_array_equal(stack.predict_proba(X), stack.final_estimator_.predict_proba(stack.transform(X)))
    reordered = X.loc[:, list(reversed(X.columns))].assign(irrelevant=123)
    np.testing.assert_array_equal(stack.predict_proba(reordered), stack.predict_proba(X))
    with pytest.raises(ValueError, match='lacks fitted input'):
        stack.predict_proba(X.drop(columns='g3'))
    path = tmp_path / 'stack.joblib'
    joblib.dump(stack, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(reordered), stack.predict_proba(X))
    np.testing.assert_array_equal(loaded.predict(X), stack.predict(X))


@pytest.mark.parametrize('invalid', ['row_order', 'column_order', 'nonfinite', 'width'])
def test_oof_schema_errors_are_explicit(invalid):
    X, y = data()
    bases = [('rf', rf().fit(X, y))]
    Z = aligned_base_probabilities(bases, X, np.unique(y))
    if invalid == 'row_order':
        Z = Z.iloc[::-1]
    elif invalid == 'column_order':
        Z = Z.iloc[:, ::-1]
    elif invalid == 'nonfinite':
        Z.iloc[0, 0] = np.nan
    else:
        Z = Z.to_numpy()[:, :1]
    stack = HonestStackingClassifier(estimators=bases, final_estimator=LogisticRegression())
    with pytest.raises(ValueError, match='OOF'):
        stack.fit_honest(X, y, Z)


def test_in_sample_fit_and_infeasible_nested_oof_fail_instead_of_falling_back():
    X, y = data((2, 2))
    with pytest.raises(ValueError, match='training-only OOF'):
        HonestStackingClassifier(estimators=[('rf', rf())]).fit(X, y)
    def forbidden(values, labels):
        raise AssertionError('Infeasible nested OOF must fail before base fitting')
    with pytest.raises(CVError, match='Stacking base OOF'):
        train_oof_stacker(**engine_args(X, y, fit_bases=forbidden))


def test_legacy_saved_stack_prediction_layout_and_zero_filling_are_preserved(tmp_path):
    X, y = data()
    bases = [('rf', rf().fit(X[['g0', 'g1']], y)), ('xgb', rf().fit(X[['g2', 'g3']], y))]
    # Build the historical persisted state, including its unprefixed passthrough.
    legacy = HonestStackingClassifier(estimators=bases, final_estimator=LogisticRegression(max_iter=1000),
                                      cv='prefit', stack_method='predict_proba', passthrough=True)
    legacy.estimators_ = [est for _, est in bases]
    legacy.stack_method_ = ['predict_proba'] * 2
    legacy.classes_ = np.unique(y)
    legacy._label_encoder = LabelEncoder().fit(y)
    legacy._base_features_in_ = {name: list(est.feature_names_in_) for name, est in bases}
    legacy._passthrough_features_in_ = list(X.columns)
    Z = np.column_stack([est.predict_proba(X.loc[:, est.feature_names_in_]) for _, est in bases] + [X])
    legacy.final_estimator_ = clone(legacy.final_estimator).fit(Z, y)
    expected = legacy.final_estimator_.predict_proba(Z)
    path = tmp_path / 'legacy.joblib'
    joblib.dump(legacy, path)
    before = path.read_bytes()
    saved = joblib.load(path)
    assert not hasattr(saved, '_stack_schema_version_')
    np.testing.assert_array_equal(saved.predict_proba(X), expected)
    np.testing.assert_array_equal(saved.predict(X), legacy.final_estimator_.predict(Z))
    missing = X.drop(columns=['g1', 'g5'])
    manual = X.copy()
    manual[['g1', 'g5']] = 0.0
    np.testing.assert_array_equal(saved.predict_proba(missing), saved.predict_proba(manual))
    assert path.read_bytes() == before


def test_fitted_preprocessors_own_their_schema_in_direct_and_holdout_prediction(tmp_path):
    from omiclass.eval.holdout import _match_features_to_model, evaluate_holdout
    X, y = data()
    X['unused_constant'] = 1.0
    prep = TrainingPreprocessor(config(tmp_path))
    base = PartitionClassifier(rf(), prep).fit(X, y)
    stack = HonestStackingClassifier(estimators=[('rf', base)],
        final_estimator=LogisticRegression(max_iter=1000), passthrough=True, stack_method='predict_proba')
    stack.passthrough_preprocessor_ = clone(prep).fit(X, y)
    stack.fit_honest(X, y, aligned_base_probabilities([('rf', base)], X, np.unique(y)))
    external = X.drop(columns='unused_constant')
    assert _match_features_to_model(external, stack) is external
    np.testing.assert_array_equal(stack.predict_proba(external), stack.predict_proba(X))
    with pytest.raises(ValueError, match='lacks fitted preprocessing'):
        evaluate_holdout(stack, external.drop(columns='g0'), y)


def test_stack_de_tuning_importance_and_calibration_are_within_current_base_fit(tmp_path, monkeypatch):
    from omiclass.de.limma_voom_runner import LimmaVoomRunner
    X, y = data((32, 32))
    cfg = config(tmp_path)
    audit, active = [], []
    def de(self, raw_counts_df, meta_df, **kwargs):
        ids = set(meta_df.index)
        assert active and ids <= active[-1]
        assert list(raw_counts_df.columns) == list(meta_df.index)
        np.testing.assert_array_equal(meta_df['cluster'], y.loc[meta_df.index])
        audit.append(ids)
        return dict(selected_genes=list(X.columns[:4]))
    monkeypatch.setattr(LimmaVoomRunner, 'run_limma_voom', de)
    prep = TrainingPreprocessor(cfg, de=True, raw_expression=X.T,
        metadata=pd.DataFrame({'unused_outcome': 999}, index=X.index), work_dir=tmp_path / 'de')
    feature_cfg = FeatureSelectionCfg(enabled=True, subset_fracs=[.5], min_features=1)
    def builder():
        return PartitionClassifier(rf(), prep, feature_cfg, subset_fraction=.5, inner_cv=2)
    def fit_bases(values, labels):
        active.append(set(values.index))
        try:
            _, params, _, _ = inner_select_recipe(model_builder=builder, Xtr=values, ytr=labels,
                tune_space={'estimator__max_depth': Integer(1, 2)}, tune_iter=1,
                tune_scoring='f1_macro', tune_inner_cv=2, seed=80, n_jobs=1, feature_cfg=None)
            model = calibrate_within_training(fitted_or_builder=builder().set_params(**params),
                Xtr_used=values, ytr=labels, method='isotonic', cv=2, seed=80, ensemble=True)
        finally:
            active.pop()
        return [('rf', model)], {}
    result, _ = train_oof_stacker(**engine_args(X, y, fit_bases=fit_bases))
    assert np.isfinite(result.oof_proba).all() and audit
    assert min(map(len, audit)) < len(X) / 4  # Inner tuning and calibration, not just outer fits.


def test_fixed_base_helper_keeps_feature_selection_inside_each_fit(monkeypatch):
    X, y = data()
    selected = []
    original = PartitionClassifier.fit
    def fit(self, values, labels, **kwargs):
        result = original(self, values, labels, **kwargs)
        if self.feature_cfg is not None:
            selected.append((set(values.index), list(self.selected_features_)))
        return result
    monkeypatch.setattr(PartitionClassifier, 'fit', fit)
    cfg = FeatureSelectionCfg(enabled=True, subset_fracs=[.5], min_features=1)
    result, _ = train_oof_stacker(**engine_args(X, y, tune_iter=1,
        feature_cfg=cfg, base_specs=[('rf', rf, {'max_depth': Integer(1, 2)})]))
    assert np.isfinite(result.oof_proba).all() and selected
    assert all(len(features) == 3 for _, features in selected)
    assert max(len(ids) for ids, _ in selected) <= len(X) / 2


def small_native_builders(monkeypatch):
    import omiclass.train.trainer as module
    original = module.get_builder
    def small(key, **kwargs):
        if key == 'stacked_ensemble':
            return original(key, **kwargs)
        params = dict(kwargs.pop('override_params', {}) or {})
        params.update(n_estimators=3, max_depth=2)
        if 'light_gbm' in key:
            params.update(verbosity=-1, min_child_samples=2, num_leaves=4)
        return original(key, override_params=params, **kwargs)
    monkeypatch.setattr(module, 'get_builder', small)
    monkeypatch.setattr(module, 'report_cv', lambda **kwargs: None)


@pytest.mark.parametrize('counts,passthrough,method,ensemble', [
    ((30, 24, 18), False, 'sigmoid', False),
    ((30, 24, 18), True, 'isotonic', True),
    ((24, 40), True, 'sigmoid', False),
])
def test_real_native_suite_nested_weighting_calibration_and_external_prediction(
        tmp_path, monkeypatch, counts, passthrough, method, ensemble):
    X, y = data(counts)
    cfg = config(tmp_path, rf_calibration_method=method, rf_calibration_ensemble=ensemble,
                 **{'models.stacked_ensemble.passthrough': passthrough})
    small_native_builders(monkeypatch)
    calls = []
    original = xgb.XGBClassifier.fit
    def fit(self, values, labels, **kwargs):
        if type(self).__name__ == 'FitLocalWeightedXGB':
            local_counts = np.bincount(labels)
            if len(local_counts) == 2:
                assert self.scale_pos_weight == pytest.approx(local_counts[0] / local_counts[1])
            else:
                np.testing.assert_allclose(kwargs['sample_weight'], compute_sample_weight('balanced', labels))
            calls.append(len(labels))
        return original(self, values, labels, **kwargs)
    monkeypatch.setattr(xgb.XGBClassifier, 'fit', fit)
    trainer = Trainer(cfg, tmp_path / 'models', TrainingPreprocessor(cfg))
    summary = trainer.run(X, y)
    assert summary.metrics.status.eq('ok').all(), summary.metrics[['model', 'status', 'error']].to_string()
    stack = trainer.final_models['stacked_ensemble']
    assert isinstance(stack.estimators[0][1], CalibratedClassifierCV)
    assert stack.estimators[0][1].method == method
    assert stack.estimators[0][1].ensemble is ensemble
    assert len(stack.meta_feature_columns_) == 3 * len(counts) + (6 if passthrough else 0)
    assert calls and min(calls) < len(X) / 2
    scope = json.loads((trainer.artifacts_dir / 'stacking_fit_scope.json').read_text())
    assert scope['final_base_models']['xgb'] in ['xgboost', 'weighted_xgboost']
    for fold in scope['outer_folds']:
        assert fold['passthrough'] is passthrough
        assert len(fold['probability_columns']) == 3 * len(counts)
        for item in fold['base_selection'].values():
            assert item['fit_ids'] == fold['train_ids']
    external = X.iloc[:4].copy()
    external.index = [f'external_{i}' for i in range(4)]
    saved = joblib.load(summary.artifact_paths['stacked_ensemble'])
    np.testing.assert_array_equal(saved.predict_proba(external.iloc[:, ::-1]), stack.predict_proba(external))
    assert np.isfinite(saved.predict_proba(external)).all()


def test_rf_consensus_does_not_truncate_other_bases_or_passthrough(tmp_path, monkeypatch):
    import omiclass.train.trainer as module
    X, y = data()
    cfg = config(tmp_path)
    cfg.runtime.feature_selection.enabled = True
    trainer = Trainer(cfg, tmp_path / 'models')
    keys = ['plain_random_forest', 'xgboost', 'light_gbm']
    column_sets = [['g0', 'g1'], ['g2', 'g3'], ['g4', 'g5']]
    trainer.final_models = {key: rf().fit(X[columns], y) for key, columns in zip(keys, column_sets)}
    trainer.cv_results = {key: CVResult(y.to_numpy(), y.to_numpy(), None,
        [dict(f1_macro=.5)], []) for key in keys}
    trainer.cv_results[keys[0]].geneset_consensus_genes = ['g0']
    monkeypatch.setattr(trainer, '_partition_active', lambda: False)  # Historical truncation path.
    def fit_bases(values, labels):
        return [(name, rf().fit(values[columns], labels))
                for name, columns in zip(['rf', 'xgb', 'lgbm'], column_sets)], {}
    monkeypatch.setattr(trainer, '_fit_stacking_bases', fit_bases)
    monkeypatch.setattr(module, 'report_cv', lambda **kwargs: None)
    _, path = trainer.run_stacked('stacked_ensemble', X, y, 'stack.joblib', cfg.models.stacked_ensemble)
    stack = joblib.load(path)
    assert list(stack.feature_names_in_) == list(X.columns)
    assert stack._passthrough_features_in_ == list(X.columns)
    np.testing.assert_array_equal(stack.transform(X).iloc[:, 2:4],
                                  trainer.final_models['xgboost'].predict_proba(X[['g2', 'g3']]))


def test_local_base_variant_selection_recomputes_dispatch_and_excludes_final_fit_failures(tmp_path, monkeypatch):
    import omiclass.train.trainer as module
    X, y = data((24, 12))  # Existing mild candidate set.
    trainer = Trainer(config(tmp_path), tmp_path / 'models')
    seen = []
    original_keys = trainer._model_keys_for_labels
    def keys(labels):
        seen.append(labels.copy())
        return original_keys(labels)
    monkeypatch.setattr(trainer, '_model_keys_for_labels', keys)
    def setup(key, values, labels, model_conf=None):
        def builder():
            estimator = rf()
            estimator.stack_key = key
            return estimator
        return builder, {}, {}, None
    monkeypatch.setattr(trainer, '_candidate_setup', setup)
    def cv(**kwargs):
        estimator = kwargs['model_builder']()
        score = .9 if estimator.stack_key.startswith('weighted') else .5
        return SimpleNamespace(per_fold_metrics=[dict(selection_score=score)])
    monkeypatch.setattr(module, 'nested_oof_cv', cv)
    original_fit = trainer._fit_final
    def fit(builder, *args, **kwargs):
        if builder().stack_key == 'weighted_xgboost':
            raise RuntimeError('Synthetic weighted final-fit failure')
        return original_fit(builder, *args, **kwargs)
    monkeypatch.setattr(trainer, '_fit_final', fit)
    estimators, details = trainer._fit_stacking_bases(X, y)
    assert details['rf']['selected_key'] == 'weighted_random_forest'
    assert details['xgb']['selected_key'] == 'xgboost'
    assert 'weighted_xgboost' in details['xgb']['failures']
    assert details['lgbm']['selected_key'] == 'weighted_light_gbm'
    pd.testing.assert_series_equal(seen[0], y)
    assert [name for name, _ in estimators] == ['rf', 'xgb', 'lgbm']
