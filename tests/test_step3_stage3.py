"""Small synthetic checks of the existing native candidates' fitting policies."""
import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from skopt import BayesSearchCV
from skopt.space import Integer

from omiclass.config.loaders import load_config
from omiclass.config.schema import FeatureSelectionCfg
from omiclass.errors.errors import CVError
from omiclass.models.registry import get_builder
from omiclass.models.xgboost import XGBNoEarlyStop, FitLocalWeightedXGB
from omiclass.train.cv_engine import (
    nested_oof_cv, calibrate_within_training, _feature_rank_by_importance,
)
from omiclass.train.scoped import PartitionClassifier
from omiclass.train.stacking_engine import train_oof_stacker
from omiclass.train.trainer import Trainer


def data(counts=(23, 16, 11)):
    y = pd.Series(np.repeat(np.arange(len(counts)), counts))
    X = pd.DataFrame(np.random.RandomState(31).normal(size=(len(y), 6)),
                     columns=[f'g{i}' for i in range(6)])
    return X, y


def weighted_builder():
    return get_builder('weighted_xgboost', random_state=80, n_jobs=1,
                       override_params=dict(n_estimators=3, max_depth=2))()


def oof_kwargs(X, y, **values):
    return dict(model_builder=weighted_builder, X=X, y=y, outer_splits=3,
                seed=80, tune_space={}, tune_iter=0, tune_scoring='neg_log_loss',
                tune_inner_cv=2, n_jobs=1, feature_cfg=None,
                eval_calibrated_in_cv=False, calib_method='sigmoid', calib_cv=2,
                **values)


@pytest.fixture
def weight_spy(monkeypatch):
    """Observe the actual XGB fit arguments after the fit-local policy runs."""
    calls = []
    original = xgb.XGBClassifier.fit

    def fit(self, X, y, **kwargs):
        calls.append(dict(ids=np.asarray(X.index) if hasattr(X, 'index') else np.arange(len(y)),
                          y=np.asarray(y).copy(), scale=self.get_params()['scale_pos_weight'],
                          weights=None if kwargs.get('sample_weight') is None else
                              np.asarray(kwargs['sample_weight']).copy()))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, 'fit', fit)
    return calls


def assert_local_weights(calls):
    assert calls
    for call in calls:
        counts = np.bincount(call['y'])
        if len(counts) == 2:
            assert call['scale'] == pytest.approx(counts[0] / counts[1])
            assert call['weights'] is None
        else:
            assert call['scale'] is None
            np.testing.assert_allclose(call['weights'], compute_sample_weight('balanced', call['y']))


@pytest.mark.parametrize('counts', [(4, 11), (11, 4), (6, 6), (13, 7, 5)])
def test_xgb_weights_come_from_fit_labels_and_preserve_clone_params(counts, weight_spy):
    X, y = data(counts)
    model = clone(weighted_builder()).set_params(max_depth=3).fit(X, y)
    assert isinstance(model, FitLocalWeightedXGB)
    assert model.get_params()['max_depth'] == 3
    assert_local_weights(weight_spy)
    assert model.predict_proba(X).shape == (len(y), len(counts))


def test_multiclass_weights_combine_with_explicit_observation_weights(weight_spy):
    X, y = data()
    observation = np.linspace(.5, 1.5, len(y))
    weighted_builder().fit(X, y, sample_weight=observation)
    np.testing.assert_allclose(weight_spy[0]['weights'], observation * compute_sample_weight('balanced', y))


@pytest.mark.parametrize('counts', [(17, 29), (23, 16, 11)])
def test_inner_outer_ranking_and_final_fits_have_partition_local_weights(counts, weight_spy):
    X, y = data(counts)
    # Withhold some rows entirely; no fit in this test is allowed to see them.
    train, holdout = next(StratifiedKFold(5, shuffle=True, random_state=9).split(X, y))
    Xtr, ytr = X.iloc[train], y.iloc[train]
    kwargs = oof_kwargs(Xtr, ytr)
    kwargs.update(tune_iter=1, tune_space={'max_depth': Integer(1, 2)})
    result = nested_oof_cv(**kwargs)
    outer_fit_sizes = {len(tr) for tr, _ in StratifiedKFold(3, shuffle=True, random_state=80).split(Xtr, ytr)}
    assert any(len(call['y']) < min(outer_fit_sizes) for call in weight_spy)  # inner search
    assert any(len(call['y']) in outer_fit_sizes for call in weight_spy)
    assert result.oof_proba.shape == (len(ytr), len(counts))
    assert np.isfinite(result.oof_proba).all()
    splits = list(StratifiedKFold(2, shuffle=True, random_state=80).split(Xtr, ytr))
    offset = len(weight_spy)
    ranking = _feature_rank_by_importance(weighted_builder, Xtr, ytr, {}, splits)
    assert sorted(ranking) == list(range(X.shape[1]))
    for call, (tr, _) in zip(weight_spy[offset:], splits):
        np.testing.assert_array_equal(call['ids'], Xtr.iloc[tr].index)
    Trainer.__new__(Trainer)._fit_final(weighted_builder, Xtr, ytr, calibrate=False,
        method='sigmoid', cv=2, seed=80)
    assert len(weight_spy[-1]['y']) == len(train)
    assert all(set(call['ids']).isdisjoint(holdout) for call in weight_spy)
    assert_local_weights(weight_spy)


def test_partition_and_representation_clones_keep_weighted_policy(weight_spy):
    X, y = data()
    feature_cfg = FeatureSelectionCfg(enabled=True, subset_fracs=[.5], min_features=2)
    builder = lambda: PartitionClassifier(
        Pipeline([('scale', StandardScaler()), ('classifier', weighted_builder())]),
        feature_cfg=feature_cfg, subset_fraction=.5, inner_cv=2,
    )
    search = BayesSearchCV(builder(), {'estimator__classifier__max_depth': Integer(1, 2)},
                          n_iter=1, cv=2, refit=True, random_state=80, error_score='raise')
    search.fit(X, y)
    assert len(search.best_estimator_.selected_features_) == 3
    assert len({len(call['y']) for call in weight_spy}) >= 3
    assert_local_weights(weight_spy)


def test_explicit_fit_weights_follow_rows_through_outer_and_inner_search(weight_spy):
    X, y = data()
    observation = np.linspace(.5, 1.5, len(y))
    kwargs = oof_kwargs(X, y, fit_kwargs={'sample_weight': observation})
    kwargs.update(tune_iter=1, tune_space={'max_depth': Integer(1, 2)})
    nested_oof_cv(**kwargs)
    for call in weight_spy:
        np.testing.assert_allclose(call['weights'], observation[call['ids']] *
                                   compute_sample_weight('balanced', call['y']))


def test_explicit_fit_weights_follow_rows_through_scoped_preliminary_tuning_and_ranking(weight_spy):
    X, y = data()
    observation = np.linspace(.5, 1.5, len(y))
    model = PartitionClassifier(weighted_builder(),
        feature_cfg=FeatureSelectionCfg(enabled=True, min_features=2),
        subset_fraction=.5, inner_cv=2, rank_space={'max_depth': Integer(1, 2)}, rank_iterations=1)
    model.fit(X, y, estimator__sample_weight=observation)
    assert len({len(call['y']) for call in weight_spy}) >= 2
    for call in weight_spy:
        np.testing.assert_allclose(call['weights'], observation[call['ids']] *
                                   compute_sample_weight('balanced', call['y']))


@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [False, True])
def test_rf_cv_matches_manual_calibrated_final_procedure(method, ensemble):
    X, y = data((20, 20))
    builder = lambda: RandomForestClassifier(n_estimators=4, random_state=80)
    kwargs = oof_kwargs(X, y)
    kwargs.update(model_builder=builder, eval_calibrated_in_cv=True,
                  calib_method=method, calibration_ensemble=ensemble)
    result = nested_oof_cv(**kwargs)
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=80).split(X, y):
        model = Trainer.__new__(Trainer)._fit_final(builder, X.iloc[tr], y.iloc[tr],
            calibrate=True, method=method, cv=2, seed=80, ensemble=ensemble)
        assert isinstance(model, CalibratedClassifierCV)
        assert model.method == method and model.ensemble is ensemble
        np.testing.assert_allclose(result.oof_proba[va], model.predict_proba(X.iloc[va]))
        np.testing.assert_array_equal(result.oof_pred[va], model.predict(X.iloc[va]))


@pytest.mark.parametrize('calibrate', [False, True])
def test_rf_trainer_passes_the_same_configured_policy_to_cv_and_final(tmp_path, monkeypatch, calibrate):
    import omiclass.train.trainer as module
    X, y = data((20, 20))
    cfg = load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label',
        cv_folds=3, bayesian_iterations=0, feature_selection_enabled=False,
        rf_calibrate=calibrate, rf_calibration_method='isotonic', rf_calibration_ensemble=True))
    trainer = Trainer(cfg, tmp_path / 'training')
    monkeypatch.setattr(module, 'report_cv', lambda **kwargs: None)
    monkeypatch.setattr(module, 'get_builder', lambda *a, **kw:
                        lambda: RandomForestClassifier(n_estimators=3, random_state=80))
    seen = []
    original = module.nested_oof_cv
    def record(**kwargs):
        seen.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(module, 'nested_oof_cv', record)
    _, path = trainer.run_rf_variant('plain_random_forest', X, y, 'rf.joblib')
    assert seen[0]['eval_calibrated_in_cv'] is calibrate
    assert seen[0]['calib_method'] == 'isotonic'
    assert seen[0]['calibration_ensemble'] is True
    model = joblib.load(path)
    assert isinstance(model, CalibratedClassifierCV) is calibrate
    np.testing.assert_array_equal(model.predict_proba(X), trainer.final_models['plain_random_forest'].predict_proba(X))


def test_calibration_slices_fit_arguments_and_keeps_local_xgb_weights(weight_spy):
    X, y = data()
    model = calibrate_within_training(fitted_or_builder=PartitionClassifier(weighted_builder()),
        Xtr_used=X, ytr=y, method='sigmoid', cv=3, seed=80,
        fit_kwargs={'estimator__sample_weight': np.ones(len(y))})
    assert np.isfinite(model.predict_proba(X)).all()
    assert any(len(call['y']) < len(y) for call in weight_spy)
    assert len(weight_spy[-1]['y']) == len(y)
    assert_local_weights(weight_spy)


@pytest.mark.parametrize('counts', [(1, 5), (6,), ()])
def test_impossible_oof_never_fits_or_returns_resubstitution(counts):
    X, y = data(counts)
    def forbidden():
        raise AssertionError('Infeasible OOF must fail before constructing a candidate')
    kwargs = oof_kwargs(X, y)
    kwargs['model_builder'] = forbidden
    with pytest.raises(CVError, match='training_class_counts'):
        nested_oof_cv(**kwargs)
    with pytest.raises(CVError, match='Stacking OOF'):
        train_oof_stacker(base_specs=[('xgb', forbidden, {})], X=X, y=y,
            outer_splits=3, seed=80, tune_iter=0, tune_scoring='f1_macro',
            tune_inner_cv=2, n_jobs=1, feature_cfg=None, meta_builder=forbidden)


def test_requested_single_fold_cannot_be_silently_replaced():
    X, y = data((6, 6))
    kwargs = oof_kwargs(X, y)
    kwargs['outer_splits'] = 1
    with pytest.raises(CVError, match='requested_splits=1'):
        nested_oof_cv(**kwargs)


def test_calibration_infeasible_inside_outer_training_does_not_fall_back():
    X, y = data((2, 2))
    kwargs = oof_kwargs(X, y)
    kwargs.update(eval_calibrated_in_cv=True, outer_splits=2)
    with pytest.raises(CVError, match='Calibration'):
        nested_oof_cv(**kwargs)
    # Ordinary, uncalibrated two-fold OOF remains valid on these labels.
    kwargs['eval_calibrated_in_cv'] = False
    result = nested_oof_cv(**kwargs)
    assert len(result.per_fold_metrics) == 2
    with pytest.raises(CVError, match='Calibration'):
        Trainer.__new__(Trainer)._fit_final(weighted_builder, X.iloc[:3], y.iloc[:3],
            calibrate=True, method='sigmoid', cv=2, seed=80)


def test_stacking_base_fits_are_weighted_and_rf_calibration_is_reused(weight_spy):
    X, y = data()
    rf = lambda: RandomForestClassifier(n_estimators=3, random_state=80)
    kwargs = dict(base_specs=[('rf', rf, {}), ('xgb', weighted_builder, {})],
        X=X, y=y, outer_splits=3, seed=80, tune_iter=0, tune_scoring='f1_macro',
        tune_inner_cv=2, n_jobs=1, feature_cfg=None,
        meta_builder=lambda: LogisticRegression(max_iter=1000))
    result, Z = train_oof_stacker(**kwargs, base_calibration={
        'rf': dict(calibrate=True, method='isotonic', ensemble=True)})
    rf_kwargs = oof_kwargs(X, y)
    rf_kwargs.update(model_builder=rf, eval_calibrated_in_cv=True, calib_method='isotonic',
                     calib_cv=3, calibration_ensemble=True, tune_scoring='f1_macro')
    expected = nested_oof_cv(**rf_kwargs)
    np.testing.assert_allclose(Z[:, :3], expected.oof_proba)
    assert np.isfinite(result.oof_proba).all()
    assert len(weight_spy) == 3
    assert_local_weights(weight_spy)


@pytest.mark.parametrize('counts', [(13, 23), (23, 16, 11)])
def test_trainer_weighted_candidate_and_saved_predictions(tmp_path, monkeypatch, weight_spy, counts):
    import omiclass.train.trainer as module
    X, y = data(counts)
    cfg = load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label',
        cv_folds=3, bayesian_iterations=0, feature_selection_enabled=False))
    trainer = Trainer(cfg, tmp_path / 'training')
    monkeypatch.setattr(module, 'report_cv', lambda **kw: None)
    original_builder = module.get_builder
    def small_builder(key, **kwargs):
        params = dict(kwargs.pop('override_params', {}) or {})
        params.update(n_estimators=3, max_depth=2)
        return original_builder(key, override_params=params, **kwargs)
    monkeypatch.setattr(module, 'get_builder', small_builder)
    row, path = trainer.run_boosting('weighted_xgboost', X, y, 'weighted.joblib', cfg.models.xgboost)
    assert np.isfinite(row['selection_score_mean'])
    assert len(weight_spy) == 4
    assert_local_weights(weight_spy)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(X), trainer.final_models['weighted_xgboost'].predict_proba(X))


def test_legacy_xgb_class_keeps_prediction_and_explicit_weight_behavior(tmp_path):
    X, y = data((8, 16))
    historical = XGBNoEarlyStop(n_estimators=3, max_depth=2, n_jobs=1,
                               random_state=80, scale_pos_weight=2).fit(X, y)
    path = tmp_path / 'historical.joblib'
    joblib.dump(historical, path)
    before = path.read_bytes()
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(X), historical.predict_proba(X))
    assert type(loaded) is XGBNoEarlyStop and loaded.scale_pos_weight == 2
    assert path.read_bytes() == before


def test_corrected_binary_weights_match_reference_and_change_old_probabilities():
    X, y = data((8, 24))
    corrected = weighted_builder().fit(X, y)
    params = dict(n_estimators=3, max_depth=2, n_jobs=1, random_state=80, seed=80, nthread=1)
    reference = XGBNoEarlyStop(**params, scale_pos_weight=8 / 24).fit(X, y)
    historical = XGBNoEarlyStop(**params, scale_pos_weight=24 / 8).fit(X, y)
    np.testing.assert_allclose(corrected.predict_proba(X), reference.predict_proba(X))
    assert np.max(np.abs(corrected.predict_proba(X) - historical.predict_proba(X))) > .01


@pytest.mark.parametrize('key', [
    'plain_random_forest', 'weighted_random_forest', 'smote_random_forest',
    'adasyn_random_forest', 'svm_smote_random_forest', 'bl_smote_random_forest',
])
def test_existing_rf_variants_support_fold_local_calibrated_oof(key):
    X, y = data((40, 24))
    kwargs = oof_kwargs(X, y)
    builder = get_builder(key, random_state=80, n_jobs=1,
                          override_params={'n_estimators': 3, 'max_depth': 2})
    kwargs.update(model_builder=lambda: PartitionClassifier(builder()),
                  eval_calibrated_in_cv=True)
    result = nested_oof_cv(**kwargs)
    assert result.oof_proba.shape == (len(y), 2)
    assert np.isfinite(result.oof_proba).all()
    for metrics, (tr, va) in zip(result.per_fold_metrics,
            StratifiedKFold(3, shuffle=True, random_state=80).split(X, y)):
        assert metrics['preprocessing_fit_ids'] == [X.iloc[tr].index.tolist()]
        assert set(metrics['preprocessing_fit_ids'][0]).isdisjoint(X.iloc[va].index)


def test_candidate_failures_remain_failed_and_cannot_supply_stacker(tmp_path, monkeypatch):
    import omiclass.train.trainer as module
    from omiclass.utils.imbalance import Severity
    X, y = data((1, 5))
    cfg = load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label',
        cv_folds=3, bayesian_iterations=0, feature_selection_enabled=False))
    monkeypatch.setattr(module, 'assess_imbalance', lambda *a, **kw: {'severity': Severity.BALANCED})
    trainer = Trainer(cfg, tmp_path / 'training')
    summary = trainer.run(X, y)
    assert not summary.artifact_paths and not trainer.final_models
    board = summary.metrics.set_index('model')
    assert board.loc['stacked_ensemble', 'status'] == 'skipped'
    for key in ['plain_random_forest', 'xgboost', 'light_gbm']:
        assert board.loc[key, 'status'] == 'failed'
        assert 'training_class_counts' in board.loc[key, 'error']
    with pytest.raises(ValueError, match='No completed'):
        trainer.best_rf_key_from_cv()
