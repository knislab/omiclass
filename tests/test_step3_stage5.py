"""Targeted evaluation and inference checks without encoders or manuscript data."""
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from omiclass.config.loaders import load_config
from omiclass.errors.errors import DataError, SurvivalPredictionError
from omiclass.eval.holdout import (
    evaluate_holdout, evaluate_holdout_with_thresholds, write_holdout_csvs,
)
from omiclass.eval.prediction import match_features, predict_estimator
from omiclass.eval.thresholds import learn_multiclass_thresholds, ordered_thresholds
from omiclass.train.cv_engine import CVResult
from omiclass.train.preprocessing import TrainingPreprocessor
from omiclass.train.scoped import PartitionClassifier


class TableModel:
    def __init__(self, classes=(2, 0, 1), probabilities=(.4, .35, .25), predicted=0):
        self.classes_ = np.asarray(classes)
        self.feature_names_in_ = np.asarray(['g0', 'g1'])
        self.probabilities = np.asarray(probabilities)
        self.predicted = predicted

    def fit(self, *args, **kwargs):
        raise AssertionError('Inference must not refit')

    def predict(self, X):
        assert list(X.columns) == list(self.feature_names_in_)
        return np.full(len(X), self.predicted)

    def predict_proba(self, X):
        assert list(X.columns) == list(self.feature_names_in_)
        return np.tile(self.probabilities, (len(X), 1))


def data(n=12):
    X = pd.DataFrame(np.random.RandomState(31).uniform(1, 4, size=(n, 2)),
                     columns=['g0', 'g1'], index=[f's{i}' for i in range(n)])
    return X, pd.Series([0, 1] * (n // 2), index=X.index)


def config(tmp_path, **values):
    return load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label',
        output_dir=tmp_path / 'run', feature_selection_enabled=False, **values))


@pytest.mark.parametrize('truth', [[0], [0, 1], [0, 1, 2]])
def test_metrics_keep_all_fitted_classes_and_probability_mass(tmp_path, truth):
    X, _ = data(6)
    X = X.iloc[:len(truth)]
    model = TableModel()
    metrics = evaluate_holdout(model, X, pd.Series(truth, index=X.index))
    assert metrics['classes'] == [2, 0, 1]
    assert np.asarray(metrics['confusion_matrix']).shape == (3, 3)
    np.testing.assert_array_equal(metrics['plot_payload']['y_proba'], model.predict_proba(X))
    expected = log_loss(truth, np.tile([.35, .25, .4], (len(truth), 1)), labels=[0, 1, 2])
    assert metrics['logloss'] == pytest.approx(expected)
    assert metrics['f1_macro'] == pytest.approx(f1_score(truth, [0] * len(truth),
        labels=[2, 0, 1], average='macro', zero_division=0))
    if len(truth) < 3:
        assert np.isnan(metrics['auroc_ovr_macro'])
    paths = write_holdout_csvs(tmp_path, 'model', metrics=metrics)
    rows = pd.read_csv(tmp_path / 'tables/model_per_class.csv')
    assert rows['class'].tolist() == [2, 0, 1]  # No micro-average row masquerading as a class.
    assert 'test_metrics_csv' in paths


def test_reversed_binary_order_uses_the_fitted_positive_column():
    class Reversed(TableModel):
        def predict_proba(self, X):
            return np.array([[.9, .1], [.2, .8]])
    X, _ = data(2)
    model = Reversed(classes=(1, 0), predicted=1)
    result = evaluate_holdout(model, X, pd.Series([1, 0], index=X.index))
    assert result['auroc_ovr_macro'] == 1.0
    assert result['logloss'] == pytest.approx(-np.log([.9, .8]).mean())
    assert result['plot_payload']['labels'] == ['1', '0']


@pytest.mark.parametrize('probabilities', [(1.,), (.5, .3, .4), (np.nan, .5, .5), (-.1, .5, .6)])
def test_invalid_probabilities_are_not_padded_truncated_or_normalized_away(probabilities):
    X, y = data()
    with pytest.raises(ValueError, match='Invalid probabilities'):
        evaluate_holdout(TableModel(probabilities=probabilities), X, y)


def test_unknown_holdout_labels_are_rejected():
    X, _ = data(2)
    with pytest.raises(ValueError, match='Evaluation labels'):
        evaluate_holdout(TableModel(), X, pd.Series([0, 9], index=X.index))


def test_estimator_decisions_remain_primary_in_core_and_foundation_evaluation():
    from omiclass.foundation.runner import _probabilities, _evaluation
    X, y = data()
    model = TableModel(classes=(0, 1), probabilities=(.1, .9), predicted=0)
    metrics = evaluate_holdout(model, X, y)
    pred, p = _probabilities(model, X, None, X.index, 2)
    np.testing.assert_array_equal(metrics['plot_payload']['y_pred'], np.zeros(len(X)))
    np.testing.assert_array_equal(pred, metrics['plot_payload']['y_pred'])
    np.testing.assert_array_equal(p.argmax(1), np.ones(len(X)))
    assert _evaluation(y, pred, p, ['a', 'b'])['accuracy'] == metrics['accuracy']


def test_no_probability_estimator_does_not_get_fabricated_probability_metrics():
    class HardOnly:
        classes_ = np.asarray([0, 1])
        feature_names_in_ = np.asarray(['g0', 'g1'])
        def predict(self, X):
            return np.zeros(len(X), dtype=int)
    X, y = data()
    result = evaluate_holdout(HardOnly(), X, y)
    assert result['plot_payload']['y_proba'] is None
    assert np.isnan(result['logloss']) and np.isnan(result['auroc_ovr_macro'])


def test_threshold_classes_are_explicit_and_independent_of_holdout_labels():
    X, y = data()
    source = np.tile([.2, .5, .3], (len(X), 1))
    truth = np.tile([2, 0, 1], 4)
    cv = CVResult(truth, np.zeros(len(X), dtype=int), source, [], [], classes=np.array([2, 0, 1]))
    info = learn_multiclass_thresholds(cv, grid=[.2, .5, .8])
    before = json.dumps(info, sort_keys=True)
    model = TableModel(classes=(1, 2, 0), probabilities=(.3, .2, .5))
    first = evaluate_holdout_with_thresholds(model, X, y, info)
    changed = evaluate_holdout_with_thresholds(model, X, 1 - y, info)
    np.testing.assert_array_equal(first['plot_payload']['y_pred'], changed['plot_payload']['y_pred'])
    np.testing.assert_array_equal(first['plot_payload']['y_proba'], model.predict_proba(X))
    assert first['logloss'] == evaluate_holdout(model, X, y)['logloss']
    assert json.dumps(info, sort_keys=True) == before
    assert info['baseline_oof_f1'] == pytest.approx(f1_score(truth, cv.oof_pred, average='macro'))
    assert info['classes'] == [2, 0, 1]


def test_historical_cv_results_keep_sorted_class_assumption():
    truth = np.tile([0, 1], 6)
    cv = CVResult(truth, truth, np.eye(2)[truth], [], [])
    cv.__dict__.pop('classes', None)
    assert learn_multiclass_thresholds(cv)['classes'] == [0, 1]
    cv.oof_proba[0, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid probabilities'):
        learn_multiclass_thresholds(cv)


@pytest.mark.parametrize('info', [
    {'classes': [0], 'thresholds': {'0': .5}},
    {'classes': [0, 1], 'thresholds': {'0': .5}},
    {'classes': [0, 1], 'thresholds': {'0': np.nan, '1': .5}},
])
def test_invalid_threshold_artifacts_cannot_silently_default(info):
    with pytest.raises(ValueError, match='Threshold'):
        ordered_thresholds(info, [1, 0])


def test_legacy_flat_threshold_mapping_with_numeric_keys_remains_readable():
    np.testing.assert_array_equal(ordered_thresholds({0: .4, 1: .6}, [1, 0]), [.6, .4])


@pytest.mark.parametrize('wrapped', [False, True])
def test_saved_prediction_replays_preprocessing_and_feature_order_without_fit(tmp_path, monkeypatch, wrapped):
    import omiclass.survival.predict as module
    X, y = data(24)
    cfg = config(tmp_path)
    base = RandomForestClassifier(n_estimators=3, random_state=80)
    if wrapped:
        model = PartitionClassifier(base, TrainingPreprocessor(cfg)).fit(X, y)
    else:
        model = base.fit(X, y)
    path = tmp_path / 'saved.joblib'
    joblib.dump(model, path)
    before = path.read_bytes()
    expected_pred, expected_p = model.predict(X), model.predict_proba(X)
    def forbidden(*args, **kwargs):
        raise AssertionError('Prediction must only transform, never fit')
    monkeypatch.setattr(TrainingPreprocessor, 'fit', forbidden)
    monkeypatch.setattr(RandomForestClassifier, 'fit', forbidden)
    monkeypatch.setattr(module, 'plot_confidence_histogram', lambda **kw: '')
    monkeypatch.setattr(module, 'plot_confidence_violin', lambda **kw: '')
    monkeypatch.setattr(module, 'render_survival_prediction_html', lambda **kw: tmp_path / 'report.html')
    external = X.iloc[:, ::-1].assign(unused=123)
    result = module.run_survival_prediction(external.T, model_path=path,
        survival_path=tmp_path / 'absent.csv', time_col='time', event_col='event',
        time_scale=1, artifacts_dir=tmp_path / 'predictions')
    np.testing.assert_array_equal(result.prediction, expected_pred)
    np.testing.assert_array_equal(result.probabilities, expected_p)
    assert path.read_bytes() == before
    with pytest.raises(SurvivalPredictionError, match='features'):
        module.run_survival_prediction(external.drop(columns='g0').T, model_path=path,
            survival_path=tmp_path / 'absent.csv', time_col='time', event_col='event',
            time_scale=1, artifacts_dir=tmp_path / 'bad')


def test_bare_feature_matching_does_not_mutate_or_zero_fill():
    X, _ = data()
    external = X[['g1']].copy()
    original = external.copy()
    with pytest.raises(ValueError, match='required features'):
        match_features(external, TableModel())
    pd.testing.assert_frame_equal(external, original)


def test_legacy_cptac_pipeline_cannot_refit_missing_preprocessing_state(tmp_path, monkeypatch):
    from omiclass.pipeline.steps.cptac_preprocess_step import CPTACPreprocessingStep
    from omiclass.data.processor import DataProcessor
    monkeypatch.setattr(DataProcessor, 'preprocess_cptac_ms', lambda *a, **kw: pytest.fail('External refit'))
    ctx = SimpleNamespace(cfg=config(tmp_path, modality='cptac_ms'), out_dir=tmp_path,
                          cache={}, opts=SimpleNamespace(mode='clean_and_predict'))
    with pytest.raises(ValueError, match='lacks fitted preprocessing state'):
        CPTACPreprocessingStep().run(ctx)


def test_raw_count_normalization_replays_frozen_lengths_and_gene_universe(tmp_path, monkeypatch):
    from omiclass.pipeline.steps.raw_count_norm_step import RawCountNormalizationStep
    from omiclass.data.processor import DataProcessor
    X, _ = data()
    counts = (X * 100).round().T
    source = tmp_path / 'lengths.csv'
    pd.Series([1000, 2000], index=counts.index, name='length').to_csv(source)
    cfg = config(tmp_path, gene_lengths_path=source)
    root = tmp_path / 'normalization'
    train = SimpleNamespace(cfg=cfg, out_dir=root, cache={'full_expr': counts},
        opts=SimpleNamespace(mode='train-only'))
    RawCountNormalizationStep().run(train)
    saved = root / 'preprocessing/raw_count_gene_lengths.joblib'
    original_bytes = saved.read_bytes()
    source.write_text('gene,length\ng0,999999\ng1,1\n')  # Only a worktree test fixture.
    monkeypatch.setattr(DataProcessor, 'load_precomputed_gene_lengths', lambda *a, **kw:
                        pytest.fail('Saved normalization must not reread or re-estimate lengths'))
    external = counts.iloc[::-1].copy()
    external.loc['extra'] = 123
    pred = SimpleNamespace(cfg=cfg, out_dir=root, cache={'full_expr': external},
        opts=SimpleNamespace(mode='clean_and_predict'))
    RawCountNormalizationStep().run(pred)
    pd.testing.assert_frame_equal(pred.cache['full_expr'], train.cache['full_expr'])
    assert saved.read_bytes() == original_bytes
    pred.cache['full_expr'] = external.drop(index='g0')
    with pytest.raises(DataError, match='required normalization genes'):
        RawCountNormalizationStep().run(pred)
    (root / 'preprocessing/scoped.json').write_text(json.dumps({'frozen_raw_count_normalization': True}))
    saved.unlink()
    with pytest.raises(DataError, match='normalization state is missing'):
        RawCountNormalizationStep().run(pred)


def test_feature_reduction_is_explicitly_disabled_without_fitting(tmp_path):
    from omiclass.eval.feature_reduction import evaluate_feature_reduction
    from omiclass.eval.feature_reduction_runner import FeatureReductionStep
    X, y = data()
    with pytest.raises(ValueError, match='validation labels'):
        evaluate_feature_reduction(estimator=TableModel(), X=X, y=y,
            k_ratios=[.5, 1], min_features=1, n_splits=2, random_state=80)
    ctx = SimpleNamespace(out_dir=tmp_path, cache={})
    FeatureReductionStep().run(ctx)
    assert json.loads((tmp_path / 'feature_reduction/status.json').read_text())['status'] == 'disabled'
    assert not (tmp_path / 'feature_reduction/feature_reduction.csv').exists()


@pytest.mark.parametrize('holdout_label', [0, 1])
def test_core_pipeline_keeps_threshold_appendix_and_training_artifact_when_holdout_worsens(
        tmp_path, monkeypatch, holdout_label):
    import api
    import omiclass.train.trainer as training
    import omiclass.pipeline.steps.train_eval_step as evaluation
    X, y = data(24)
    train, test = X.index[:16].tolist(), X.index[16:].tolist()
    y.loc[test] = holdout_label
    expression, meta, markers = [tmp_path / name for name in ['expression.csv', 'meta.csv', 'genes.txt']]
    X.T.to_csv(expression)
    pd.DataFrame({'label': y}).to_csv(meta)
    markers.write_text('g0\ng1\n')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(dict(train_index=train, test_index=test)))
    def fit(self, values, labels):
        model = TableModel(classes=(0, 1), probabilities=(.4, .6), predicted=0)
        path = self.artifacts_dir / 'custom-name.joblib'
        joblib.dump(model, path)
        self.cv_results['plain_random_forest'] = CVResult(labels.to_numpy(), model.predict(values),
            model.predict_proba(values), [], [], classes=model.classes_)
        return training.TrainSummary(pd.DataFrame([dict(model='plain_random_forest', status='ok',
            f1_macro_mean=.5, selection_score_mean=.5, selection_scoring='f1_macro')]),
            {'plain_random_forest': path})
    monkeypatch.setattr(training.Trainer, 'run', fit)
    captured = []
    monkeypatch.setattr(evaluation, 'finalize_report_for_model', lambda **kwargs: None)
    monkeypatch.setattr(evaluation, 'FinalSummaryPlotter', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(evaluation, 'render_final_summary_html', lambda **kwargs: captured.append(kwargs))
    monkeypatch.chdir(tmp_path)
    result = api.run(expression, meta_path=meta, classification_column='label', markers_path=markers,
        output_dir=Path('runs') if holdout_label else tmp_path / 'runs', run_de=False, split_json=split, log_transform=False,
        feature_selection_enabled=False, bayesian_iterations=0, feature_reduction_enabled=True,
        **{'batch_correction.algorithm': 'none'})
    assert captured[-1]['include_threshold_appendix'] is True
    assert captured[-1]['plotter'] is not captured[-1]['threshold_plotter']
    stored = json.loads((result.out_dir / 'train_artifacts/thresholds.json').read_text())
    assert stored['plain_random_forest']['thresholds'] == {'0': .5, '1': .5}
    metrics = pd.read_csv(result.combined_test_csv).iloc[0]
    assert metrics['threshold_f1_macro'] == (0.0 if holdout_label == 0 else .5)
    assert metrics['f1_macro'] == (.5 if holdout_label == 0 else 0.0)
    scope = json.loads((result.out_dir / 'preprocessing/scoped.json').read_text())
    assert scope['model_path'] == 'train_artifacts/custom-name.joblib'
    assert 'disabled' in captured[-1]['meta']['Feature reduction diagnostic']
    assert captured[-1]['plotter'].labels == ['0', '1']


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('holdout_improved', [False, True])
def test_external_survival_uses_training_thresholds_regardless_of_holdout_score(
        tmp_path, monkeypatch, legacy, holdout_improved):
    import omiclass.pipeline.steps.survival_step as module
    root = tmp_path / 'run'
    (root / 'train_artifacts').mkdir(parents=True)
    path = root / 'train_artifacts/custom.joblib'
    path.touch()
    cfg = config(tmp_path)
    cfg.models.random_forest.variants.plain_random_forest.filename = 'custom.joblib'
    info = dict(classes=[0, 1], thresholds={'0': .5, '1': .5}, source='OOF training predictions')
    if legacy:
        pd.DataFrame([dict(model='plain_random_forest', status='ok', f1_macro_mean=.6)]).to_csv(
            root / 'leaderboard.csv', index=False)
        (root / 'reports').mkdir()
        (root / 'reports/plain_random_forest.report.json').write_text(json.dumps(dict(
            test=dict(metrics=dict(f1_macro=.9)), threshold_optimized_holdout=dict(
                thresholds=info, metrics=dict(f1_macro=1.0 if holdout_improved else .1)))))
    else:
        (root / 'preprocessing').mkdir()
        (root / 'preprocessing/scoped.json').write_text(json.dumps(dict(selected_model='plain_random_forest',
            model_path='train_artifacts/custom.joblib')))
        (root / 'train_artifacts/thresholds.json').write_text(json.dumps({'plain_random_forest': info}))
    seen = []
    monkeypatch.setattr(module, 'run_survival_prediction', lambda **kwargs: seen.append(kwargs))
    X, _ = data()
    ctx = SimpleNamespace(cfg=cfg, out_dir=root, cache={'X_recorrected': X.T},
                          opts=SimpleNamespace(mode='clean_and_predict'))
    module.SurvivalStep().run(ctx)
    assert seen[0]['threshold_info'] == info
    assert seen[0]['model_path'] == path


@pytest.mark.parametrize('logged', [False, True])
def test_transductive_limma_keeps_its_workflow_and_replays_saved_input_transform(tmp_path, monkeypatch, logged):
    import sys
    import omiclass.pipeline.steps.batch_step as batch
    import omiclass.pipeline.steps.rebatch_step as rebatch
    from omiclass.data.processor import DataProcessor
    X, _ = data()
    train_input = X.T * (1000 if logged else 1)
    external = X.T * (1 if logged else 1000)  # The cohort scale heuristic would now reverse its decision.
    external.columns = ['new_' + sample for sample in external.columns]
    calls = []
    class Limma:
        def run_limma(self, expr_df, output_dir, out_csv, **kwargs):
            calls.append(expr_df.copy())
            expr_df.to_csv(output_dir / out_csv)
            return {}
    monkeypatch.setitem(sys.modules, 'omiclass.batch.limma_runner', SimpleNamespace(LimmaRunner=Limma))
    monkeypatch.setattr(DataProcessor, 'load_batch_metadata', lambda *a, **kw: (None, None))
    monkeypatch.setattr(DataProcessor, 'load_batch_metadata_prediction', lambda *a, **kw: (None, None, None))
    monkeypatch.setattr(batch, 'write_limma_report', lambda **kw: None)
    monkeypatch.setattr(rebatch, 'write_limma_report', lambda **kw: None)
    cfg = config(tmp_path, log_transform=True)
    root = tmp_path / 'correction'
    root.mkdir()
    train = SimpleNamespace(cfg=cfg, out_dir=root, cache={'full_expr': train_input})
    batch.BatchCorrectionStep().run(train)
    assert train.cache['batch_input_log1p'] is logged
    (root / 'preprocessing').mkdir()
    (root / 'preprocessing/scoped.json').write_text(json.dumps(dict(
        correction_transductive=True, batch_input_log1p=logged,
        train_ids=X.index[:6].tolist(), test_ids=X.index[6:].tolist())))
    pred = SimpleNamespace(cfg=cfg, out_dir=root, cache={'full_expr': external})
    rebatch.RebatchCorrectionStep().run(pred)
    expected = np.log1p(external) if logged else external
    pd.testing.assert_frame_equal(calls[1], expected)
    pd.testing.assert_frame_equal(pred.cache['X_recorrected'], expected)
    assert len(calls) == 2  # Intended transductive correction still runs on both supplied cohorts.


@pytest.mark.parametrize('mode', ['log', 'cptac', 'reference'])
def test_fitted_log_imputation_and_reference_state_replay_without_refitting(tmp_path, monkeypatch, mode):
    X, y = data(24)
    options = {'log_transform': mode == 'log'}
    if mode == 'log':
        X = (X * 1000).round()
    elif mode == 'cptac':
        X.iloc[0, 0] = np.nan
        options.update(modality='cptac_ms', cptac_log2_transform=True, cptac_imputation='median_by_feature')
    else:
        options['batch_correction.algorithm'] = 'frozen_quantile'
    cfg = config(tmp_path, **options)
    prep = TrainingPreprocessor(cfg, cptac=mode == 'cptac', reference=mode == 'reference')
    model = PartitionClassifier(RandomForestClassifier(n_estimators=3, random_state=80), prep).fit(X, y)
    if mode == 'log':
        assert model.preprocessor_.did_log_ is True
    expected = model.predict_proba(X)
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    monkeypatch.setattr(TrainingPreprocessor, 'fit', lambda *a, **kw: pytest.fail('Refitting during prediction'))
    saved = joblib.load(path)
    np.testing.assert_array_equal(predict_estimator(saved, match_features(X.iloc[:, ::-1], saved))[1], expected)


def test_foundation_external_prediction_restores_manifest_feature_order_and_does_not_fit(tmp_path, monkeypatch):
    import api
    from omiclass.foundation import runner, extraction
    from omiclass.features.bulkformer_reduction import EmbeddingRepresentation
    from sklearn.svm import SVC
    X, y = data(24)
    expr, metadata = tmp_path / 'expression.csv', tmp_path / 'meta.csv'
    X.T.to_csv(expr)
    pd.DataFrame({'label': y}).to_csv(metadata)
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(dict(train_index=X.index[:16].tolist(), test_index=X.index[16:].tolist())))
    source = tmp_path / 'encoder'
    source.mkdir()
    (source / 'encoder.py').write_text('VERSION = 1\n')
    reverse = [False]
    def extract(counts, cfg, out, metadata=None):
        out.mkdir(parents=True, exist_ok=True)
        values = counts.T.astype(float)
        return extraction.Extraction(values.iloc[:, ::-1] if reverse[0] else values, {'mocked_encoder': True})
    monkeypatch.setattr(extraction, 'extract', extract)
    run = api.run(expr, meta_path=metadata, classification_column='label', output_dir=tmp_path / 'runs',
        bulkformer_root_dir=source, foundation=dict(enabled=True, classifier='svm', split_path=split))
    before = {str(path): path.read_bytes() for path in run.out_dir.rglob('*') if path.is_file()}
    monkeypatch.setattr(SVC, 'fit', lambda *a, **kw: pytest.fail('External SVM fit'))
    monkeypatch.setattr(EmbeddingRepresentation, 'fit', lambda *a, **kw: pytest.fail('External representation fit'))
    original = runner._probabilities
    def check(model, values, *args, **kwargs):
        assert values.columns.tolist() == ['g0', 'g1']
        return original(model, values, *args, **kwargs)
    monkeypatch.setattr(runner, '_probabilities', check)
    reverse[0] = True
    external = X.iloc[:3].copy()
    external.index = ['external_a', 'external_b', 'external_c']
    path = tmp_path / 'external.csv'
    external.T.to_csv(path)
    prediction = api.predict(path, run.out_dir, tmp_path / 'predictions')
    assert pd.read_csv(prediction.combined_test_csv).sample_id.tolist() == external.index.tolist()
    assert before == {str(path): path.read_bytes() for path in run.out_dir.rglob('*') if path.is_file()}


def test_external_confidence_corresponds_to_ordinary_prediction_not_probability_argmax(tmp_path, monkeypatch):
    import omiclass.survival.predict as module
    X, _ = data()
    model = TableModel(classes=(0, 1), probabilities=(.1, .9), predicted=0)
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    monkeypatch.setattr(module, 'plot_confidence_histogram', lambda **kw: '')
    monkeypatch.setattr(module, 'plot_confidence_violin', lambda **kw: '')
    monkeypatch.setattr(module, 'render_survival_prediction_html', lambda **kw: tmp_path / 'report.html')
    result = module.run_survival_prediction(X.T, model_path=path, survival_path=tmp_path / 'missing.csv',
        time_col='time', event_col='event', time_scale=1, artifacts_dir=tmp_path / 'predictions')
    np.testing.assert_array_equal(result.prediction, np.zeros(len(X)))
    saved = pd.read_csv(tmp_path / 'predictions/model_predictions.csv')
    np.testing.assert_allclose(saved['proba'], .1)


def test_summary_plot_schema_preserves_class_order_and_rejects_probability_padding(tmp_path):
    from omiclass.plotting.plotter import FinalSummaryPlotter
    plotter = FinalSummaryPlotter(labels=['2', '0', '1'], y_true=[0],
        per_model={}, per_model_train={}, out_dir=tmp_path)
    np.testing.assert_array_equal(plotter._get_label_index(), [2, 0, 1])
    with pytest.raises(ValueError, match='Invalid probabilities'):
        plotter._ensure_proba_shape(np.array([[.5, .5]]), 3)
