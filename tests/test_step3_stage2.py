"""Synthetic fitting-scope checks; no encoder/checkpoint or manuscript runs."""
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from omiclass.config.loaders import load_config
from omiclass.cli.runtime import RuntimeOptions, create_runtime_options_from_path_conf_kwargs
from omiclass.train.preprocessing import TrainingPreprocessor
from omiclass.train.scoped import PartitionClassifier
from omiclass.train.cv_engine import nested_oof_cv, inner_select_recipe
from omiclass.utils.splits import apply_split, holdout_split, load_split
from omiclass.pipeline.steps.split_step import SplitStep


def cfg(tmp_path, **values):
    return load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label',
                            output_dir=tmp_path / 'run', **values))


def data():
    X = pd.DataFrame(np.random.RandomState(19).normal(size=(40, 8)),
                     index=[f's{i}' for i in range(40)], columns=[f'g{i}' for i in range(8)])
    return X, pd.Series([0, 1] * 20, index=X.index)


@pytest.mark.parametrize('split', [
    dict(train_index=['s0', 's1'], test_index=['s1']),
    dict(train_index=[], test_index=['s0']),
    dict(train_index=['s0', 's0'], test_index=['s1']),
    dict(train_index=['absent'], test_index=['s0']),
    dict(train_index=['s0', 's2'], test_index=['s1']),
])
def test_invalid_splits_cannot_change_population(split):
    X, y = data()
    with pytest.raises(ValueError):
        apply_split(X, y, split)


def test_explicit_holdout_does_not_silently_fall_back(tmp_path):
    X, y = data()
    path = tmp_path / 'holdout.txt'
    with pytest.raises(FileNotFoundError):
        holdout_split(X, y, hold_out_path=path)
    path.write_text('absent\n')
    with pytest.raises(ValueError):
        holdout_split(X, y, hold_out_path=path)


def test_supplied_split_is_honored_once_and_validated_after_filtering(tmp_path):
    X, y = data()
    pd.DataFrame({'label': y}).to_csv(tmp_path / 'meta.csv')
    split = dict(train_index=X.index[:30].tolist(), test_index=X.index[30:].tolist())
    supplied = tmp_path / 'provided.json'
    supplied.write_text(json.dumps(split))
    config = cfg(tmp_path, split_json=supplied, reuse_split=False)
    opts = create_runtime_options_from_path_conf_kwargs(config, tmp_path / 'counts.csv', {'split_json': supplied})
    ctx = SimpleNamespace(cfg=config, opts=opts, cache={'full_expr': X.T}, split_out=tmp_path / 'run')
    SplitStep().run(ctx)
    assert ctx.cache['split'] == split == load_split(ctx.cache['split_json'])
    with pytest.raises(ValueError, match='absent'):
        apply_split(X.drop(index='s0'), y, ctx.cache['split'])
    assert opts.reuse_split is False  # An explicit path is sufficient on its own.


def test_native_log_and_constant_decisions_ignore_holdout_values(tmp_path):
    from omiclass.foundation.heads import fit_native
    from omiclass.train.trainer import Trainer, TrainSummary
    from unittest.mock import patch
    X, y = data()
    X = (X.abs() * 100).round()
    X['train_constant'] = [1] * 30 + list(range(10))
    seen = []
    def train(self, values, labels):
        model = self._with_bulkformer_reduction(lambda: RandomForestClassifier(n_estimators=2, random_state=0))().fit(values, labels)
        seen.append(model.preprocessor_)
        path = self.artifacts_dir / 'rf.joblib'
        joblib.dump(model, path)
        return TrainSummary(pd.DataFrame([dict(model='rf', status='ok', f1_macro_mean=.5)]), {'rf': path})
    config = cfg(tmp_path, foundation=dict(enabled=True, classifier='omiclass'))
    with patch.object(Trainer, 'run', train):
        for i in range(2):
            out = tmp_path / f'native{i}'
            out.mkdir()
            fit_native(config, X, y, X.index[:30].tolist(), out)
            X.loc[X.index[30:], 'g0'] = -100
    assert all(p.did_log_ for p in seen)
    assert seen[0].selected_features_ == seen[1].selected_features_
    assert 'train_constant' not in seen[0].selected_features_
    assert seen[0].fit_ids_ == X.index[:30].tolist()


@pytest.fixture
def audited_de(monkeypatch):
    import omiclass.de.limma_voom_runner as module
    calls = []
    active = []
    original_fit = PartitionClassifier.fit
    def partition_fit(self, values, labels, **kwargs):
        active.append((values.index.tolist(), np.asarray(labels)))
        try:
            return original_fit(self, values, labels, **kwargs)
        finally:
            active.pop()
    monkeypatch.setattr(PartitionClassifier, 'fit', partition_fit)
    def fit(self, raw_counts_df, meta_df, **kwargs):
        assert list(raw_counts_df.columns) == list(meta_df.index)
        assert kwargs['report'] is False
        assert active and meta_df.index.tolist() == active[-1][0]
        np.testing.assert_array_equal(meta_df['cluster'], active[-1][1])
        calls.append((raw_counts_df.copy(), meta_df.copy()))
        # Select a different dimension when supplied labels change; exercises
        # actual fitted-state dependence without an R/limma installation.
        choice = int(np.asarray(meta_df['cluster']).sum()) % 3
        return dict(selected_features=[raw_counts_df.index[choice], raw_counts_df.index[choice + 1]])
    monkeypatch.setattr(module.LimmaVoomRunner, 'run_limma_voom', fit)
    return calls


def test_de_outer_and_inner_cv_receive_only_corresponding_training_labels(tmp_path, audited_de):
    X, y = data()
    X = X.iloc[:32].copy()
    y = y.loc[X.index]
    raw = (X.abs() * 100).round().T
    metadata = pd.DataFrame({'covariate': np.arange(len(X))}, index=X.index)
    config = cfg(tmp_path, log_transform=False, feature_selection_enabled=False)
    def builder():
        return PartitionClassifier(RandomForestClassifier(n_estimators=2, random_state=0),
            TrainingPreprocessor(config, raw_expression=raw, metadata=metadata,
                                 de=True, work_dir=tmp_path / 'fits'))
    result = nested_oof_cv(model_builder=builder, X=X, y=y, outer_splits=2, seed=80,
        tune_space={'estimator__max_depth': [2, 3]}, tune_iter=1, tune_scoring='f1_macro',
        tune_inner_cv=2, n_jobs=1, feature_cfg=None, eval_calibrated_in_cv=False,
        calib_method='', calib_cv=2)
    scopes = []
    for train, validation in StratifiedKFold(2, shuffle=True, random_state=80).split(X, y):
        outer = set(X.index[train])
        scopes.append(outer)
        for local_train, _ in StratifiedKFold(2, shuffle=True, random_state=80).split(X.iloc[train], y.iloc[train]):
            scopes.append(set(X.index[train][local_train]))
    assert audited_de and len(audited_de) >= 6
    for values, meta in audited_de:
        assert set(values.columns) in scopes
        np.testing.assert_array_equal(meta['cluster'], y.loc[meta.index])
        pd.testing.assert_frame_equal(values, raw.loc[:, meta.index])
        assert len(meta) < len(X)
    assert len(result.oof_pred) == len(X)


def test_heldout_labels_and_prediction_never_enter_de_fit(tmp_path, audited_de):
    X, y = data()
    config = cfg(tmp_path, log_transform=False, feature_selection_enabled=False)
    train, holdout = X.index[:30], X.index[30:]
    prep = TrainingPreprocessor(config, raw_expression=X.loc[train].T,
        metadata=pd.DataFrame(index=train), de=True, work_dir=tmp_path / 'fits')
    model = PartitionClassifier(RandomForestClassifier(n_estimators=3, random_state=80), prep).fit(X.loc[train], y.loc[train])
    baseline = model.predict_proba(X.loc[holdout])
    y.loc[holdout] = 42
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    saved = joblib.load(path)
    np.testing.assert_array_equal(saved.predict_proba(X.loc[holdout]), baseline)
    assert len(audited_de) == 1
    assert set(audited_de[0][1].index) == set(train)
    assert saved.preprocessor.raw_expression is None and saved.preprocessor_.metadata is None


def test_importance_rankings_are_refitted_inside_each_scoring_partition(tmp_path, monkeypatch):
    import omiclass.train.cv_engine as cv
    X, y = data()
    config = cfg(tmp_path, log_transform=False, feature_selection_min_features=1,
        feature_selection_subset_fracs=[.5, 1.0])
    seen = []
    original = cv._feature_rank_by_importance
    def audit(builder, values, labels, *args, **kwargs):
        seen.append(set(values.index))
        return original(builder, values, labels, *args, **kwargs)
    monkeypatch.setattr(cv, '_feature_rank_by_importance', audit)
    def builder():
        return PartitionClassifier(RandomForestClassifier(n_estimators=2, random_state=80),
            TrainingPreprocessor(config), config.runtime.feature_selection, seed=80, inner_cv=2)
    _, params, _, _ = inner_select_recipe(model_builder=builder, Xtr=X, ytr=y,
        tune_space={'estimator__max_depth': [2, 3]}, tune_iter=1, tune_scoring='f1_macro',
        tune_inner_cv=2, seed=80, n_jobs=1, feature_cfg=config.runtime.feature_selection)
    allowed = [set(X.index[tr]) for tr, _ in StratifiedKFold(2, shuffle=True, random_state=80).split(X, y)]
    assert seen and all(scope in allowed for scope in seen)
    assert params['subset_fraction'] in {.5, 1.0}


def test_cptac_transform_matches_existing_training_math_and_freezes_medians(tmp_path):
    from omiclass.data.processor import DataProcessor
    X = pd.DataFrame([[1., 2., 10.], [3., np.nan, 20.], [5., 6., 30.]], columns=['a', 'b', 'c'])
    config = cfg(tmp_path, modality='cptac_ms', cptac_log2_transform=True,
        cptac_imputation='median_by_feature', log_transform=False)
    prep = TrainingPreprocessor(config, cptac=True).fit(X)
    expected, _ = DataProcessor(config).preprocess_cptac_ms(X.T)
    pd.testing.assert_frame_equal(prep.transform(X), expected.T)
    external = pd.DataFrame([[9., np.nan, 1000.]], columns=X.columns)
    output = prep.transform(external)
    assert output['b'].iloc[0] == pytest.approx(prep.medians_['b'])
    np.testing.assert_array_equal(prep.medians_, np.log2(X).median())


def test_frozen_reference_and_holdout_share_one_explicit_partition(tmp_path):
    X, y = data()
    X = X.abs() + 1
    pd.DataFrame({'label': y}).to_csv(tmp_path / 'meta.csv')
    ids = tmp_path / 'reference-ids.txt'
    ids.write_text('\n'.join(X.index[:30]))
    config = cfg(tmp_path, **{'batch_correction.algorithm': 'frozen_quantile',
                              'batch_correction.training_samples_path': ids})
    ctx = SimpleNamespace(cfg=config, opts=RuntimeOptions(mode='batch-train'),
        cache={'full_expr': X.T}, split_out=tmp_path / 'run')
    SplitStep().run(ctx)
    prep = TrainingPreprocessor(config, reference=True).fit(X.iloc[:30], y.iloc[:30])
    assert list(prep.reference_.training_samples_) == ctx.cache['split']['train_index']
    assert ctx.cache['split']['test_index'] == X.index[30:].tolist()
    assert prep.transform(X.iloc[30:]).shape == (10, 8)


def test_de_reporting_does_not_supply_the_classifier_gene_list(tmp_path, monkeypatch):
    import omiclass.pipeline.steps.de_step as module
    X, y = data()
    pd.DataFrame({'label': y}).to_csv(tmp_path / 'meta.csv')
    config = cfg(tmp_path)
    calls = []
    def run(self, raw_counts_df, meta_df, **kwargs):
        calls.append((raw_counts_df, meta_df))
        return dict(selected_features=['g0'])
    monkeypatch.setattr(module.LimmaVoomRunner, 'run_limma_voom', run)
    split = dict(train_index=X.index[:30].tolist(), test_index=X.index[30:].tolist())
    ctx = SimpleNamespace(cfg=config, out_dir=tmp_path / 'run',
        cache={'full_expr': X.T, 'gene_list': X.columns.tolist(), 'split': split})
    module.DEStep().run(ctx)
    assert ctx.cache['gene_list'] == X.columns.tolist()
    assert ctx.cache['de_report_features'] == ['g0']
    assert list(calls[0][0].columns) == list(calls[0][1].index) == split['train_index']


def test_de_fit_report_separation_preserves_markers_and_min_count(tmp_path, monkeypatch):
    import omiclass.de.limma_voom_runner as module
    X, y = data()
    X = (X.abs() * 100).round()
    config = cfg(tmp_path, min_features=99,
        **{'de_gene_selection.min_genes': 2, 'de_gene_selection.max_genes': 4,
           'de_gene_selection.min_genes_per_class': 1, 'de_gene_selection.max_genes_per_class': 2,
           'de_gene_selection.close_pair_fraction': 0.0})
    calls, reports = [], []
    def r_fit(path):
        settings = json.loads(Path(path).read_text())
        calls.append(settings)
        frame = pd.DataFrame({'gene_id': X.columns, 'Contrast': '0 vs all',
            'logFC': np.linspace(-3, 3, X.shape[1]), 'P.Value': np.linspace(.001, .01, X.shape[1]),
            'adj.P.Val': .01})
        frame.to_csv(settings['output_filepath'], index=False)
        return [json.dumps({'status': 'success', 'filepath': settings['output_filepath']})]
    ro = SimpleNamespace(r=SimpleNamespace(source=lambda path: None),
                         globalenv={'execute_limma_voom_de': r_fit})
    monkeypatch.setattr(module, '_r_runtime', lambda: (ro, RuntimeError))
    monkeypatch.setattr(module.LimmaVoomRunner, '_render_report', lambda *a: reports.append(a))
    meta = pd.DataFrame({'cluster': y}, index=X.index)
    runner = module.LimmaVoomRunner(config.de, config)
    results = []
    for report in [False, True]:
        results.append(runner.run_limma_voom(X.T, meta, tmp_path / str(report), 'cluster',
                                            min_count=7, report=report))
    assert results[0]['selected_features'] == results[1]['selected_features']
    assert results[0]['selected_features'] and len(reports) == 1
    assert all(call['min_count'] == 7 for call in calls)
    assert len(calls) == 2  # No extra fit for rendering.


def test_real_trainer_de_recipe_and_final_fit_exclude_holdout(tmp_path, monkeypatch, audited_de):
    import omiclass.train.trainer as module
    X, y = data()
    train, holdout = X.index[:32], X.index[32:]
    config = cfg(tmp_path, cv_folds=2, tuning_inner_cv=2, bayesian_iterations=1,
        rf_calibrate=False, log_transform=False, feature_selection_min_features=1,
        feature_selection_subset_fracs=[.5, 1.0], rf_search_n_estimators=[2, 2],
        rf_search_max_depth=[2, 2], rf_search_min_samples_split=[2, 2],
        rf_search_min_samples_leaf=[1, 1], rf_search_max_features=['sqrt'])
    monkeypatch.setattr(module, 'report_cv', lambda **kw: None)
    prep = TrainingPreprocessor(config, raw_expression=X.loc[train].T,
        metadata=pd.DataFrame(index=train), de=True, work_dir=tmp_path / 'fits')
    trainer = module.Trainer(config, tmp_path / 'training', preprocessor=prep)
    row, path = trainer.run_rf_variant('plain_random_forest', X.loc[train], y.loc[train], 'rf.joblib')
    model = joblib.load(path)
    assert np.isfinite(row['selection_score_mean'])
    assert set(model.preprocessor_.de_fit_ids_) == set(train)
    assert model.subset_fraction in {.5, 1.0}
    assert model.predict_proba(X.loc[holdout]).shape == (8, 2)
    assert audited_de and all(set(meta.index) <= set(train) for _, meta in audited_de)
    assert any(len(meta) == 8 for _, meta in audited_de)  # Inner fits within outer training.


def test_calibration_clones_preprocessing_and_de_inside_training(tmp_path, audited_de):
    from sklearn.calibration import CalibratedClassifierCV
    X, y = data()
    config = cfg(tmp_path, log_transform=False, feature_selection_enabled=False)
    train = X.index[:32]
    prep = TrainingPreprocessor(config, raw_expression=X.loc[train].T,
        metadata=pd.DataFrame(index=train), de=True, work_dir=tmp_path / 'fits')
    estimator = PartitionClassifier(RandomForestClassifier(n_estimators=2, random_state=80), prep)
    model = CalibratedClassifierCV(estimator, cv=StratifiedKFold(2, shuffle=True, random_state=80), ensemble=False)
    model.fit(X.loc[train], y.loc[train])
    assert sorted(len(meta) for _, meta in audited_de) == [16, 16, 32]
    assert all(set(meta.index) <= set(train) for _, meta in audited_de)
    assert model.predict_proba(X.iloc[32:]).shape == (8, 2)


def test_metadata_classes_outside_input_do_not_change_dense_labels(tmp_path):
    X, y = data()
    meta = pd.DataFrame({'label': ['b', 'c'] * 20}, index=X.index)
    meta.loc['not_in_expression'] = ['a']
    meta.to_csv(tmp_path / 'meta.csv')
    config = cfg(tmp_path)
    ctx = SimpleNamespace(cfg=config, opts=RuntimeOptions(), cache={'full_expr': X.T}, split_out=tmp_path / 'run')
    SplitStep().run(ctx)
    assert set(ctx.cache['labels']) == {0, 1}
    assert config.project.classes_names == ['b', 'c']


def test_new_prediction_route_does_not_reapply_saved_reference(tmp_path):
    from omiclass.pipeline.steps.rebatch_step import RebatchCorrectionStep
    X, y = data()
    config = cfg(tmp_path, **{'batch_correction.algorithm': 'frozen_quantile'})
    root = config.project.out_dir
    (root / 'preprocessing').mkdir(parents=True)
    (root / 'preprocessing/scoped.json').write_text('{}')
    ctx = SimpleNamespace(cfg=config, out_dir=root, cache={'full_expr': X.T},
                          opts=RuntimeOptions(mode='clean_and_predict', input_mode='batch_correction'))
    RebatchCorrectionStep().run(ctx)
    pd.testing.assert_frame_equal(ctx.cache['X_recorrected'], X.T)


def test_public_core_pipeline_shares_split_and_ignores_holdout_label_changes(tmp_path, monkeypatch):
    import api
    import omiclass.de.limma_voom_runner as de_module
    import omiclass.train.trainer as trainer_module
    import omiclass.pipeline.steps.train_eval_step as eval_module
    X, y = data()
    counts = (X.abs() * 100 + 1).round().T
    expression, metadata = tmp_path / 'counts.csv', tmp_path / 'meta.csv'
    counts.to_csv(expression)
    lengths = tmp_path / 'lengths.csv'
    pd.Series(1000, index=counts.index, name='length').to_csv(lengths)
    split = dict(train_index=X.index[:32].tolist(), test_index=X.index[32:].tolist())
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps(split))
    fits = []
    def de_fit(self, raw_counts_df, meta_df, **kwargs):
        fits.append((raw_counts_df.copy(), meta_df.copy()))
        return dict(selected_features=['g0', 'g1'])
    monkeypatch.setattr(de_module.LimmaVoomRunner, 'run_limma_voom', de_fit)
    monkeypatch.setattr(trainer_module, 'get_builder', lambda *a, **kw:
        lambda: RandomForestClassifier(n_estimators=3, random_state=80))
    monkeypatch.setattr(trainer_module, 'report_cv', lambda **kw: None)
    def train_one(self, values, labels):
        row, path = self.run_rf_variant('plain_random_forest', values, labels, 'rf.joblib')
        row.update(status='ok')
        return trainer_module.TrainSummary(pd.DataFrame([row]), {'plain_random_forest': path})
    monkeypatch.setattr(trainer_module.Trainer, 'run', train_one)
    monkeypatch.setattr(eval_module, 'finalize_report_for_model', lambda **kw: None)
    monkeypatch.setattr(eval_module, 'FinalSummaryPlotter', lambda **kw: None)
    monkeypatch.setattr(eval_module, 'render_final_summary_html', lambda **kw: None)
    models = []
    for number in range(2):
        meta = pd.DataFrame({'label': y, 'sex': ['m', 'z'] * 20})
        meta.loc[split['test_index'], 'sex'] = 'a'
        meta.to_csv(metadata)
        result = api.run(expression, meta_path=metadata, classification_column='label',
            output_dir=tmp_path / f'run{number}', raw_count=True, gene_lengths_path=lengths,
            split_json=split_path, feature_selection_enabled=False, cv_folds=2,
            de_covariates=['sex'],
            bayesian_iterations=0, rf_calibrate=False,
            **{'batch_correction.algorithm': 'none'})
        assert load_split(result.split_json) == split
        model = joblib.load(result.out_dir / 'train_artifacts/rf.joblib')
        assert model.preprocessor_.de_fit_ids_ == split['train_index']
        models.append(model)
        y.loc[split['test_index']] = 1 - y.loc[split['test_index']]
    from omiclass.data.processor import DataProcessor
    transformed = DataProcessor(cfg(tmp_path, gene_lengths_path=lengths)).normalize_raw_counts(counts).T
    np.testing.assert_array_equal(models[0].predict_proba(transformed), models[1].predict_proba(transformed))
    assert len(fits) == 8  # Per run: report, two outer fits, final fit.
    for raw, meta in fits:
        assert set(meta.index) <= set(split['train_index'])
        assert set(meta['sex']) == {0, 1}
        np.testing.assert_array_equal(raw, counts.loc[:, meta.index])


def test_fractional_numeric_labels_are_not_truncated(tmp_path):
    X, y = data()
    pd.DataFrame({'label': [.1, .2] * 20}, index=X.index).to_csv(tmp_path / 'meta.csv')
    config = cfg(tmp_path)
    ctx = SimpleNamespace(cfg=config, opts=RuntimeOptions(), cache={'full_expr': X.T}, split_out=tmp_path / 'run')
    SplitStep().run(ctx)
    assert set(ctx.cache['labels']) == {0, 1}


def test_new_prediction_recovers_saved_raw_input_mode(tmp_path):
    root = tmp_path / 'run'
    (root / 'preprocessing').mkdir(parents=True)
    (root / 'leaderboard.csv').write_text('model\nrf\n')
    (root / 'preprocessing/scoped.json').write_text(json.dumps({'input_mode': 'raw_counts'}))
    config = cfg(tmp_path)
    opts = create_runtime_options_from_path_conf_kwargs(config, tmp_path / 'external.csv', {})
    assert opts.mode == 'clean_and_predict' and opts.input_mode == 'raw_counts'
    with pytest.raises(ValueError, match='input scale'):
        create_runtime_options_from_path_conf_kwargs(config, tmp_path / 'external.csv', {'raw_count': False})


def test_stack_passthrough_reuses_fitted_preprocessing(tmp_path):
    from omiclass.models.stacked import build_stacked_ensemble
    X, y = data()
    X = X.abs() + 1
    X.loc[X.index[::4], 'g0'] = np.nan
    config = cfg(tmp_path, modality='cptac_ms', log_transform=False,
                 cptac_imputation='median_by_feature', feature_selection_enabled=False)
    prep = TrainingPreprocessor(config, cptac=True)
    base = PartitionClassifier(RandomForestClassifier(n_estimators=3, random_state=80), prep).fit(X, y)
    model = build_stacked_ensemble(random_state=80, n_jobs=1, max_iter=100, solver='lbfgs',
        cv='prefit', passthrough=True, rf_trained=base, xgb_trained=base, lgbm_trained=base)
    model.passthrough_preprocessor_ = clone(prep).fit(X, y)
    Z = np.concatenate([base.predict_proba(X)] * 3, axis=1)
    model.fit_honest(X, y, Z)
    p = model.predict_proba(X)
    assert np.isfinite(p).all()
    path = tmp_path / 'stack.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(X), p)


def test_saved_log_transform_rejects_incompatible_external_values(tmp_path):
    X, _ = data()
    X = (X.abs() * 100).round()
    prep = TrainingPreprocessor(cfg(tmp_path)).fit(X)
    external = X.iloc[:1].copy()
    external.iloc[0, 0] = -2
    with pytest.raises(ValueError, match='fitted log1p'):
        prep.transform(external)


def test_saved_cptac_offset_is_not_refitted_on_external_values(tmp_path):
    X = pd.DataFrame({'a': [-2., 1., 3.], 'b': [1., 2., 3.]})
    config = cfg(tmp_path, modality='cptac_ms', cptac_log2_transform=True)
    prep = TrainingPreprocessor(config, cptac=True).fit(X)
    with pytest.raises(ValueError, match='fitted CPTAC'):
        prep.transform(pd.DataFrame({'a': [-3.], 'b': [1.]}))


def test_transductive_limma_still_receives_the_full_cohort(tmp_path, monkeypatch):
    import sys
    import omiclass.pipeline.steps.batch_step as module
    X, y = data()
    X = X.abs() + 1
    config = cfg(tmp_path, batch_column='batch')
    metadata = pd.DataFrame({'batch': ['a', 'b'] * 20}, index=X.index)
    calls = []
    class Runner:
        def run_limma(self, **kwargs):
            calls.append(kwargs)
            kwargs['expr_df'].to_csv(kwargs['output_dir'] / kwargs['out_csv'])
            return {}
    monkeypatch.setitem(sys.modules, 'omiclass.batch.limma_runner', SimpleNamespace(LimmaRunner=Runner))
    monkeypatch.setattr(module, 'DataProcessor', lambda cfg: SimpleNamespace(load_batch_metadata=lambda **kw: (metadata, None)))
    monkeypatch.setattr(module, 'write_limma_report', lambda **kw: None)
    ctx = SimpleNamespace(cfg=config, opts=RuntimeOptions(mode='batch-train'), out_dir=tmp_path / 'run',
        cache={'full_expr': X.T, '_did_raw_count_norm': True,
               'split': dict(train_index=X.index[:32].tolist(), test_index=X.index[32:].tolist())})
    module.BatchCorrectionStep().run(ctx)
    pd.testing.assert_frame_equal(calls[0]['expr_df'], X.T)
    assert list(ctx.cache['X_corrected'].columns) == X.index.tolist()


def test_frozen_reference_is_refitted_within_each_cv_training_scope(tmp_path, monkeypatch):
    from omiclass.batch.reference import FrozenQuantile
    X, y = data()
    X, y = X.iloc[:32].abs() + 1, y.iloc[:32]
    config = cfg(tmp_path, **{'batch_correction.algorithm': 'frozen_quantile'})
    original, seen = FrozenQuantile.fit, []
    def fit(self, frame, metadata=None):
        seen.append(set(frame.columns))
        return original(self, frame, metadata)
    monkeypatch.setattr(FrozenQuantile, 'fit', fit)
    def builder():
        return PartitionClassifier(RandomForestClassifier(n_estimators=2, random_state=80),
            TrainingPreprocessor(config, reference=True))
    nested_oof_cv(model_builder=builder, X=X, y=y, outer_splits=2, seed=80,
        tune_space={'estimator__max_depth': [2, 3]}, tune_iter=1, tune_scoring='f1_macro',
        tune_inner_cv=2, n_jobs=1, feature_cfg=None, eval_calibrated_in_cv=False,
        calib_method='', calib_cv=2)
    allowed = []
    for train, _ in StratifiedKFold(2, shuffle=True, random_state=80).split(X, y):
        allowed.append(set(X.index[train]))
        for inner, _ in StratifiedKFold(2, shuffle=True, random_state=80).split(X.iloc[train], y.iloc[train]):
            allowed.append(set(X.index[train][inner]))
    assert len(seen) == 6 and all(scope in allowed for scope in seen)


def test_reference_preserves_numeric_outcome_design(tmp_path):
    X, y = data()
    X = X.abs() + 1
    controls = tmp_path / 'controls.txt'
    controls.write_text('g0\ng1\ng2\n')
    config = cfg(tmp_path, classes_names=['low', 'high'],
        **{'batch_correction.algorithm': 'reference_ruv',
           'batch_correction.control_genes_path': controls,
           'batch_correction.control_justification': 'Synthetic negative controls for scope testing'})
    meta = pd.DataFrame(index=X.index)
    meta.attrs.update(outcome_numeric=True, outcome_values={0: 10, 1: 20})
    prep = TrainingPreprocessor(config, metadata=meta, reference=True).fit(X, y)
    assert 'mean' in prep.reference_.design_encoding_['label']
    assert prep.reference_.design_encoding_['label']['mean'] == 15


def test_historical_native_wrapper_keeps_saved_log_semantics(tmp_path):
    from omiclass.foundation.heads import NativeModel
    X, y = data()
    X = (X.abs() * 100).round()
    estimator = RandomForestClassifier(n_estimators=3, random_state=80).fit(np.log1p(X), y)
    historical = NativeModel(estimator, X.columns, True)
    path = tmp_path / 'historical-native.joblib'
    joblib.dump(historical, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(X), estimator.predict_proba(np.log1p(X)))
    from omiclass.train.scoped import fitted_partitions
    assert fitted_partitions(SimpleNamespace(calibrated_classifiers_=[SimpleNamespace(base_estimator=estimator)])) == []
