"""Stage 1 reliability checks with synthetic data and no encoder inference."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold

import api
from omiclass.config.loaders import freeze_config, load_config
from omiclass.config.overrides import normalize_user_overrides
from omiclass.config.schema import Config
from omiclass.errors.errors import ConfigError
from omiclass.foundation import extraction, runner
from omiclass.train.cv_engine import nested_oof_cv
from omiclass.train.selection import rank_candidates, selection_summary


def config(tmp_path, **overrides):
    return load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label', **overrides))


@pytest.mark.parametrize('overrides', [
    {'runtime.typo': 1}, {'runtime': {'typo': 1}},
    {'models': {'random_forest': {'defaults': {'search_space': {'typo': [1, 2]}}}}},
    {'models.random_forest.defaults.search_space.typo': [1, 2]},
    {'foundation': {'article_head': {'typo': 1}}},
    {'runtime': {'tuning': {'n_iter': 2}}},
    {'runtime': {'imbalance': {'mild_threshold': 8}}},
])
def test_unknown_and_frozen_fields_are_not_silently_ignored(overrides):
    with pytest.raises(ConfigError):
        normalize_user_overrides(overrides)


def test_nested_and_dotted_controls_merge_without_losing_values(tmp_path):
    cfg = config(tmp_path, runtime={'tuning': {'enabled': False}, 'cv_folds': 2},
                 **{'runtime.tuning.inner_cv': 3, 'models.stacked_ensemble.passthrough': False})
    assert cfg.runtime.tuning.enabled is False
    assert cfg.runtime.tuning.inner_cv == 3
    assert cfg.runtime.cv_folds == 2
    assert cfg.models.stacked_ensemble.passthrough is False
    for values in [dict(runtime={'cv_folds': 2}, **{'runtime.cv_folds': 3}),
                   {'runtime.cv_folds': 3, 'runtime': {'cv_folds': 2}}]:
        with pytest.raises(ConfigError, match='Conflicting'):
            normalize_user_overrides(values)


def test_explicit_empty_optional_search_block_preserves_schema_defaults(tmp_path):
    assert config(tmp_path).models.random_forest.defaults.search_space is None
    cfg = config(tmp_path, **{'models.random_forest.defaults.search_space': {}})
    assert cfg.models.random_forest.defaults.search_space.n_estimators == [200, 600]
    assert normalize_user_overrides({'runtime': {'tuning': {}}}) == {'runtime': {'tuning': {}}}


@pytest.mark.parametrize('values', [
    {'runtime': {'tuning': {'method': 'grid'}}},
    {'runtime.tuning.method': 'random'}, {'lgbm_early_stopping': 20},
    {'runtime.scoring': 'not_a_scorer'}, {'bayesian_iterations': -1},
])
def test_unimplemented_or_invalid_settings_fail_early(tmp_path, values):
    with pytest.raises((ConfigError, ValueError)):
        config(tmp_path, **values)


@pytest.mark.parametrize('values', [
    {'runtime.tuning.enabled': False}, {'runtime': {'tuning': {'enabled': False}}},
    {'bayesian_iterations': 0},
])
def test_inner_and_final_native_tuning_can_be_disabled(tmp_path, monkeypatch, values):
    from omiclass.train.trainer import Trainer
    import omiclass.train.trainer as module
    X = pd.DataFrame(np.random.RandomState(2).normal(size=(24, 4)))
    y = pd.Series([0, 1] * 12)
    cfg = config(tmp_path, cv_folds=2, feature_selection_enabled=False, rf_calibrate=False, **values)
    trainer = Trainer(cfg, tmp_path / 'training')
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled tuning must not construct a search')
    monkeypatch.setattr(module, 'BayesSearchCV', forbidden)
    import omiclass.train.cv_engine as cv_module
    monkeypatch.setattr(cv_module, 'BayesSearchCV', forbidden)
    monkeypatch.setattr(module, 'get_builder', lambda *a, **k: lambda: RandomForestClassifier(n_estimators=3, random_state=80))
    reports = []
    monkeypatch.setattr(module, 'report_cv', lambda **kw: reports.append(kw))
    row, path = trainer.run_rf_variant('plain_random_forest', X, y, 'rf.joblib')
    assert path.is_file() and np.isfinite(row['selection_score_mean'])
    assert row['selection_scoring'] == 'f1_macro'
    assert reports[0]['cfg_short']['bayes_iter'] == 0
    assert all(not fold.best_params for fold in trainer.cv_results['plain_random_forest'].fold_recipes)


def test_requested_probability_scorer_drives_cv_and_ranking():
    X = pd.DataFrame(np.random.RandomState(2).normal(size=(24, 4)))
    y = pd.Series([0, 1] * 12)
    result = nested_oof_cv(model_builder=lambda: RandomForestClassifier(n_estimators=3, random_state=80),
        X=X, y=y, outer_splits=2, seed=80, tune_space={}, tune_iter=0,
        tune_scoring='neg_log_loss', tune_inner_cv=2, n_jobs=1,
        feature_cfg=None, eval_calibrated_in_cv=False, calib_method='', calib_cv=2)
    for fold, (train, validation) in zip(result.per_fold_metrics, StratifiedKFold(2, shuffle=True, random_state=80).split(X, y)):
        model = RandomForestClassifier(n_estimators=3, random_state=80).fit(X.iloc[train], y.iloc[train])
        expected = get_scorer('neg_log_loss')(model, X.iloc[validation], y.iloc[validation])
        assert fold['selection_score'] == pytest.approx(expected)
    summary = selection_summary(result.per_fold_metrics, 'neg_log_loss')
    board = pd.DataFrame([
        dict(model='a', status='ok', f1_macro_mean=.9, **summary),
        dict(model='b', status='ok', f1_macro_mean=.1, **{**summary, 'selection_score_mean': -.01}),
        dict(model='failed', status='failed', **{**summary, 'selection_score_mean': 1}),
        dict(model='no_artifact', status='ok', **{**summary, 'selection_score_mean': 2}),
    ])
    ranked, _ = rank_candidates(board, ['a', 'b', 'failed'], 'neg_log_loss')
    assert ranked.model.tolist() == ['b', 'a']
    with pytest.raises(ValueError, match='No completed'):
        rank_candidates(board, ['failed'], 'neg_log_loss')


def test_default_ties_and_missing_requested_metric():
    board = pd.DataFrame([dict(model='z', f1_macro_mean=.5), dict(model='a', f1_macro_mean=.5)])
    ranked, column = rank_candidates(board, ['a', 'z'], 'f1_macro')
    assert ranked.model.tolist() == ['a', 'z'] and column == 'f1_macro_mean'
    with pytest.raises(ValueError, match='requested scorer'):
        rank_candidates(board, ['a', 'z'], 'accuracy')


@pytest.mark.parametrize('key', ['xgboost', 'light_gbm'])
def test_real_boosting_with_disabled_search_uses_requested_metric(tmp_path, monkeypatch, key):
    import joblib
    import omiclass.train.trainer as module
    from omiclass.models.registry import get_builder
    X = pd.DataFrame(np.random.RandomState(4).normal(size=(30, 4)), columns=list('abcd'))
    y = pd.Series([0, 1] * 15)
    cfg = config(tmp_path, cv_folds=2, bayesian_iterations=0, feature_selection_enabled=False,
                 **{'runtime.scoring': 'neg_log_loss'})
    def small_builder(model_key, **kwargs):
        parameters = dict(kwargs.pop('override_params', {}) or {})
        parameters['n_estimators'] = 3
        return get_builder(model_key, override_params=parameters, **kwargs)
    monkeypatch.setattr(module, 'get_builder', small_builder)
    monkeypatch.setattr(module, 'report_cv', lambda **kwargs: None)
    trainer = module.Trainer(cfg, tmp_path / 'training')
    model_cfg = cfg.models.xgboost if key == 'xgboost' else cfg.models.light_gbm
    row, path = trainer.run_boosting(key, X, y, f'{key}.joblib', model_cfg)
    assert row['selection_scoring'] == 'neg_log_loss'
    assert np.isfinite(row['selection_score_mean'])
    np.testing.assert_array_equal(joblib.load(path).predict_proba(X), trainer.final_models[key].predict_proba(X))


def test_stacker_base_selection_excludes_leftover_failed_cv(tmp_path):
    from omiclass.train.trainer import Trainer
    trainer = Trainer(config(tmp_path), tmp_path / 'training')
    trainer.cv_results = {key: SimpleNamespace(per_fold_metrics=[dict(f1_macro=score)])
                          for key, score in [('xgboost', .4), ('weighted_xgboost', .99)]}
    trainer.final_models = {'xgboost': object()}
    assert trainer.best_boosting_key_from_cv(['xgboost', 'weighted_xgboost']) == 'xgboost'
    trainer.final_models['weighted_xgboost'] = object()
    trainer._failed_keys = {'weighted_xgboost'}
    assert trainer.best_boosting_key_from_cv(['xgboost', 'weighted_xgboost']) == 'xgboost'
    trainer.final_models.clear()
    with pytest.raises(ValueError, match='No completed'):
        trainer.best_boosting_key_from_cv(['xgboost', 'weighted_xgboost'])


def test_native_suite_preserves_strict_completion_and_diagnostics(tmp_path, monkeypatch):
    from omiclass.foundation.heads import fit_native
    from omiclass.train.trainer import Trainer, TrainSummary
    X = pd.DataFrame(np.random.RandomState(6).normal(size=(12, 3)))
    y = pd.Series([0, 1] * 6)
    def partial(self, values, labels):
        return TrainSummary(pd.DataFrame([
            dict(model='rf', status='ok', f1_macro_mean=.5),
            dict(model='xgboost', status='failed', error='Synthetic final fit failure')]),
            {'rf': self.artifacts_dir / 'rf.joblib'})
    monkeypatch.setattr(Trainer, 'run', partial)
    out = tmp_path / 'native'
    out.mkdir()
    cfg = config(tmp_path, foundation=dict(enabled=True, classifier='omiclass'))
    with pytest.raises(RuntimeError, match='completed suite'):
        fit_native(cfg, X, y, X.index.tolist(), out)
    board = pd.read_csv(out / 'native_leaderboard.csv')
    assert 'Synthetic final fit failure' in board.error.tolist()
    assert not (out / 'foundation.json').exists()


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    counts = pd.DataFrame(np.random.RandomState(1).randint(1, 20, size=(4, 24)),
                          index=['g1', 'g2', 'g3', 'g4'], columns=[f's{i}' for i in range(24)])
    expression, meta, split = [tmp_path / name for name in ['counts.csv', 'meta.csv', 'split.json']]
    counts.to_csv(expression)
    pd.DataFrame({'label': ['a', 'b'] * 12}, index=counts.columns).to_csv(meta)
    split.write_text(json.dumps(dict(train_index=counts.columns[:20].tolist(), test_index=counts.columns[20:].tolist())))
    calls = []
    def mock_extract(values, cfg, out, metadata=None):
        calls.append(cfg)
        out.mkdir(parents=True)
        return extraction.Extraction(values.T.astype(float), {'mocked_encoder': True})
    monkeypatch.setattr(extraction, 'extract', mock_extract)
    source = tmp_path / 'encoder'
    source.mkdir()
    (source / 'encoder.py').write_text('VERSION = 1\n')
    def train(model='bulkformer', **kwargs):
        backend = 'cellfm' if model.startswith('cellfm') else model
        return api.run(expression, meta_path=meta, classification_column='label', output_dir=tmp_path / 'runs',
            foundation=dict(enabled=True, model=model, classifier='svm', split_path=split,
                input_scale='tpm' if model == 'bulkrnabert' else 'counts'),
            **{f'{backend}_root_dir': source, **kwargs})
    external = counts.iloc[:, :2].copy()
    external.columns = ['external_a', 'external_b']
    external_path = tmp_path / 'external.csv'
    external.to_csv(external_path)
    return SimpleNamespace(train=train, source=source, external=external_path, calls=calls, tmp=tmp_path)


@pytest.mark.parametrize('model,asset', [
    ('bulkformer', 'encoder.py'), ('cellfm_1view', 'encoder.py'),
    ('bulkrnabert', 'encoder.py'),
    ('bulkrnabert', 'checkpoints/bulk_rna_bert_gtex_encode/config.json'),
    ('bulkrnabert', 'checkpoints/bulk_rna_bert_gtex_encode/params.joblib'),
])
def test_changed_encoder_sources_and_directory_checkpoints_are_rejected(synthetic_run, model, asset):
    fixture = synthetic_run
    path = fixture.source / asset
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('original')
    run = fixture.train(model)
    manifest = json.loads((run.out_dir / 'foundation.json').read_text())
    assert manifest['format_version'] == 2
    path.write_text('changed')
    with pytest.raises(ValueError, match='Encoder assets changed'):
        api.predict(fixture.external, run.out_dir, fixture.tmp / 'prediction')
    assert len(fixture.calls) == 1


def test_vendor_binary_change_is_part_of_identity(tmp_path):
    cfg = config(tmp_path, foundation=dict(enabled=True, model='cellfm_20view'), cellfm_vendor_dir=tmp_path / 'vendor')
    cfg.cellfm.vendor_dir.mkdir()
    library = cfg.cellfm.vendor_dir / 'kernel.so'
    library.write_bytes(b'before')
    before = runner._resource_digests(cfg)
    library.write_bytes(b'after')
    assert before != runner._resource_digests(cfg)


def test_version_one_prediction_remains_readable_and_read_only(synthetic_run):
    fixture = synthetic_run
    run = fixture.train()
    config_path = run.out_dir / 'config.freeze.yaml'
    frozen = yaml.safe_load(config_path.read_text())
    frozen['models']['stacked_ensemble'].pop('passthrough')  # Original schema.
    config_path.write_text(yaml.safe_dump(frozen))
    cfg = Config(**frozen)
    manifest_path = run.out_dir / 'foundation.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(format_version=1, encoder_assets=runner._resource_digests(cfg, version=1))
    manifest['artifact_sha256']['config.freeze.yaml'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.out_dir.rglob('*') if p.is_file()}
    prediction = api.predict(fixture.external, run.out_dir, fixture.tmp / 'predictions')
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.out_dir.rglob('*') if p.is_file()}
    assert before == after
    record = json.loads((prediction.out_dir / 'prediction.json').read_text())
    assert record['encoder_verification'] == 'legacy_file_checks_only'
    assert len(pd.read_csv(prediction.artifacts['predictions'])) == 2


def test_saved_tpm_path_is_not_inherited_but_explicit_external_path_is(tmp_path):
    cfg = config(tmp_path, foundation=dict(enabled=True, model='bulkrnabert', classifier='svm'),
                 bulkrnabert_tpm_input_path=tmp_path / 'training-tpm.csv')
    saved = tmp_path / 'saved'
    freeze_config(cfg, saved)
    prediction = dict(foundation=dict(enabled=True, mode='predict', run_dir=saved))
    loaded = load_config(prediction)
    assert loaded.bulkrnabert.tpm_input_path is None
    external = tmp_path / 'external-tpm.csv'
    loaded = load_config(dict(**prediction, bulkrnabert_tpm_input_path=external))
    assert loaded.bulkrnabert.tpm_input_path == external
    assert yaml.safe_load((saved / 'config.freeze.yaml').read_text())['bulkrnabert']['tpm_input_path'].endswith('training-tpm.csv')


def test_legacy_prediction_can_load_historical_ignored_search_method(tmp_path):
    cfg = config(tmp_path, output_dir=tmp_path / 'saved')
    cfg.runtime.tuning.method = 'grid'  # Historically accepted, actually fitted with Bayes.
    cfg.runtime.tuning.inner_cv = 1  # Historical helper capped this to two.
    cfg.runtime.scoring = 'historical_unused_scorer'
    freeze_config(cfg, cfg.project.out_dir)
    (cfg.project.out_dir / 'leaderboard.csv').write_text('model,f1_macro_mean\nrf,0.5\n')
    loaded = load_config(dict(output_dir=cfg.project.out_dir))
    assert loaded.runtime.tuning.method == 'grid'
    assert loaded.runtime.tuning.inner_cv == 1
    assert loaded.runtime.scoring == 'historical_unused_scorer'


def test_saved_models_must_have_checksums_before_loading(synthetic_run):
    fixture = synthetic_run
    run = fixture.train()
    path = run.out_dir / 'foundation.json'
    manifest = json.loads(path.read_text())
    manifest['models']['unverified'] = 'models/other.joblib'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='lack checksums'):
        api.predict(fixture.external, run.out_dir, fixture.tmp / 'prediction')
    assert len(fixture.calls) == 1


@pytest.mark.parametrize('script', ['batch/limma_correction.R', 'de/limma_voom_de.R'])
def test_r_dependency_failure_does_not_install_packages(script, tmp_path):
    executable = shutil.which('Rscript')
    if executable is None:
        pytest.skip('Rscript not available; dependency guard requires R only, no R packages')
    source = Path(__file__).resolve().parents[1] / 'omiclass' / script
    check = tmp_path / 'check.R'
    check.write_text('scope <- new.env()\n'
        'scope$requireNamespace <- function(...) FALSE\n'
        'scope$install.packages <- function(...) stop("INSTALL_ATTEMPT")\n'
        'scope$library <- function(...) stop("LIBRARY_ATTEMPT")\n'
        f'source({json.dumps(str(source))}, local=scope)\n')
    result = subprocess.run([executable, '--vanilla', str(check)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Missing R dependency: limma' in result.stderr
    assert 'INSTALL_ATTEMPT' not in result.stderr and 'LIBRARY_ATTEMPT' not in result.stderr
