"""Synthetic-only Subtask 3 checks. Never run an article experiment."""
import copy
import hashlib
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from test_bulkrnabert_logistic import make_fixture, load
from final_results_runs.test_bulkrnabert.logistic import inputs
from final_results_runs.test_bulkrnabert.logistic.run import evaluation_metrics
from final_results_runs.test_bulkrnabert.standalone import models, run


def synthetic(tmp_path, binary=False, apollo=False):
    fixture = make_fixture(tmp_path, binary=binary)
    data = load(fixture)
    if apollo:
        data['cohort'].loc[data['train_ids']] = 'TCGA'
        data['cohort'].loc[data['holdout_ids']] = 'APOLLO'
    return fixture, data


@pytest.mark.parametrize('binary', [False, True])
def test_svm_shapes_scaler_and_determinism(tmp_path, binary):
    _, data = synthetic(tmp_path, binary=binary)
    train, test = data['train_ids'], data['holdout_ids']
    X, y = data['X'].loc[train], data['y'].loc[train].to_numpy(dtype=int)
    first = models.fit_svm(X, y, 80)
    second = models.fit_svm(X, y, 80)
    np.testing.assert_allclose(first['scaler'].mean_, X.mean().to_numpy(), atol=1e-7)
    assert first['scaler'].n_samples_seen_ == len(train)
    values = models.embeddings(data['X'].loc[test])
    np.testing.assert_array_equal(first.predict(values), second.predict(values))
    np.testing.assert_array_equal(first.predict_proba(values), second.predict_proba(values))
    assert first.predict_proba(values).shape == (6, len(data['classes']))
    assert first.decision_function(values).shape == ((6,) if binary else (6, 3))
    np.testing.assert_allclose(first.predict_proba(values).sum(axis=1), 1)


@pytest.mark.parametrize('bad', [np.zeros((3, 255)), np.full((3, 256), np.nan), np.full((3, 256), np.inf)])
def test_invalid_embeddings_rejected(bad):
    with pytest.raises(ValueError, match='256'):
        models.fit_svm(bad, [0, 1, 0], 80)
    with pytest.raises(ValueError, match='256'):
        models.fit_mlp(bad, [0, 1, 0], ['a', 'b', 'c'], 2, models.MLPConfig(), 80)


def test_exact_mlp_architecture_and_forward():
    model = models.make_mlp(3)
    assert [type(layer) for layer in model] == [torch.nn.Linear, torch.nn.SELU,
        torch.nn.Linear, torch.nn.SELU, torch.nn.Linear]
    assert [(layer.in_features, layer.out_features) for layer in model if isinstance(layer, torch.nn.Linear)] == [(256, 256), (256, 128), (128, 3)]
    output = model(torch.zeros(4, 256))
    assert output.shape == (4, 3)
    assert torch.isfinite(torch.nn.CrossEntropyLoss()(output, torch.tensor([0, 1, 2, 0])))


def test_mlp_updates_reproducible_frozen_inputs_and_best_epoch(tmp_path):
    _, data = synthetic(tmp_path)
    X = data['X'].loc[data['train_ids']].to_numpy()
    original = X.copy()
    y = data['y'].loc[data['train_ids']].to_numpy(dtype=int)
    config = models.MLPConfig(max_epochs=6, batch_size=7, patience=2, min_delta=1e6)
    with models.torch_reproducibility(80, 1):
        initial = copy.deepcopy(models.make_mlp(3).state_dict())
    first, info = models.fit_mlp(X, y, data['train_ids'], 3, config, 80)
    second, repeated = models.fit_mlp(X, y, data['train_ids'], 3, config, 80)
    assert info == repeated
    assert info['best_epoch'] == 1 and info['epochs_trained'] == 3
    one_epoch, _ = models.fit_mlp(X, y, data['train_ids'], 3,
        models.MLPConfig(max_epochs=1, batch_size=7), 80)
    assert any(not torch.equal(initial[k], first.state_dict()[k]) for k in initial)
    for key in initial:
        assert torch.equal(first.state_dict()[key], second.state_dict()[key])
        assert torch.equal(first.state_dict()[key], one_epoch.state_dict()[key])
    np.testing.assert_array_equal(X, original)
    assert set(info['inner_train_ids']).isdisjoint(info['inner_validation_ids'])
    assert set(info['inner_train_ids'] + info['inner_validation_ids']) == set(data['train_ids'])
    assert set(data['holdout_ids']).isdisjoint(info['inner_train_ids'] + info['inner_validation_ids'])
    pred, probabilities, logits = models.predict_mlp(first, X)
    assert pred.shape == (len(X),) and logits.shape == probabilities.shape == (len(X), 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)


@pytest.mark.parametrize('classifier', ['svm', 'mlp'])
def test_apollo_absent_from_all_training_and_validation(tmp_path, monkeypatch, classifier):
    _, data = synthetic(tmp_path / 'input', apollo=True)
    # A large external shift makes inadvertent scaler fitting observable.
    data['X'].loc[data['holdout_ids']] += 10000
    before = data['X'].copy(deep=True)
    captured = []
    original = getattr(models, f'fit_{classifier}')
    def spy(X, y, *args):
        captured.extend(X.index)
        assert set(X.index) == set(data['train_ids'])
        assert set(X.index).isdisjoint(data['holdout_ids'])
        return original(X, y, *args)
    monkeypatch.setattr(models, f'fit_{classifier}', spy)
    output = tmp_path / 'result'
    run.evaluate(data, 'apolo', classifier, output, mlp_config=models.MLPConfig(max_epochs=2))
    manifest = inputs.json_file(output / 'manifest.json')
    assert captured == data['train_ids']
    trained = manifest['training']['inner_train_ids'] + manifest['training']['inner_validation_ids']
    assert set(trained) == set(data['train_ids'])
    assert set(trained).isdisjoint(data['holdout_ids'])
    if classifier == 'svm':
        pipeline = joblib.load(output / 'pipeline.joblib')
        np.testing.assert_allclose(pipeline['scaler'].mean_, data['X'].loc[data['train_ids']].mean(), atol=1e-7)
    pd.testing.assert_frame_equal(before, data['X'])
    predictions = pd.read_csv(output / 'predictions.csv')
    assert predictions.loc[predictions.split.eq('holdout'), 'sample_id'].tolist() == data['holdout_ids']
    assert predictions.loc[predictions.split.eq('holdout'), 'evaluation_role'].eq('external_apollo').all()
    # Changing evaluation values/labels cannot affect trained parameters or epoch selection.
    changed = copy.deepcopy(data)
    changed['X'].loc[data['holdout_ids']] *= -3
    changed['y'].loc[data['holdout_ids']] = (changed['y'].loc[data['holdout_ids']] + 1) % 3
    other = tmp_path / 'other'
    run.evaluate(changed, 'apolo', classifier, other, mlp_config=models.MLPConfig(max_epochs=2))
    assert inputs.json_file(other / 'manifest.json')['training'] == manifest['training']
    if classifier == 'mlp':
        first = torch.load(output / 'model.pt', weights_only=True)['state_dict']
        second = torch.load(other / 'model.pt', weights_only=True)['state_dict']
        assert all(torch.equal(first[k], second[k]) for k in first)
    else:
        first = joblib.load(output / 'pipeline.joblib')['svm']
        second = joblib.load(other / 'pipeline.joblib')['svm']
        np.testing.assert_array_equal(first.support_vectors_, second.support_vectors_)
        np.testing.assert_array_equal(first.probA_, second.probA_)


@pytest.mark.parametrize('classifier', ['svm', 'mlp'])
@pytest.mark.parametrize('binary', [False, True])
def test_outputs_metrics_resume_and_identity(tmp_path, classifier, binary):
    _, data = synthetic(tmp_path / 'input', binary=binary)
    output = tmp_path / 'output'
    cfg = models.MLPConfig(max_epochs=2)
    task = 'cll' if binary else 'ball'
    run.evaluate(data, task, classifier, output, mlp_config=cfg)
    manifest = inputs.json_file(output / 'manifest.json')
    identity = hashlib.sha256(json.dumps(manifest['identity'], sort_keys=True).encode()).hexdigest()
    assert identity == manifest['experiment_fingerprint']
    predictions = pd.read_csv(output / 'predictions.csv')
    holdout = predictions.loc[predictions.split.eq('holdout')]
    expected = evaluation_metrics(holdout.y_true.astype(int), holdout.y_pred,
        holdout[[f'probability_class_{i}' for i in range(len(data['classes']))]].to_numpy(),
        list(range(len(data['classes']))))
    for key, value in expected.items():
        if isinstance(value, float):
            assert manifest['metrics'][key] == pytest.approx(value)
        else:
            assert manifest['metrics'][key] == value
    if binary:
        assert manifest['positive_class'] == 'unmutated'
        assert 'auroc_binary' in manifest['metrics'] and 'auroc_macro' not in manifest['metrics']
    run.evaluate(data, task, classifier, output, mlp_config=cfg, resume=True)
    with pytest.raises(FileExistsError):
        run.evaluate(data, task, classifier, output, mlp_config=cfg)
    changed = copy.deepcopy(data)
    changed['provenance']['embedding_cache_sha256'] = 'different'
    with pytest.raises(ValueError, match='differs'):
        run.evaluate(changed, task, classifier, output, mlp_config=cfg, resume=True)
    with (output / 'predictions.csv').open('a') as file:
        file.write('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        run.evaluate(data, task, classifier, output, mlp_config=cfg, resume=True)


def test_cli_dry_run_no_training_or_writes(tmp_path, monkeypatch):
    fixture, _ = synthetic(tmp_path / 'input')
    monkeypatch.setattr(inputs, 'load_protocol', lambda task: fixture.protocol)
    monkeypatch.setattr(run, 'evaluate', lambda *a, **kw: pytest.fail('Dry-run fitted a model'))
    destination = tmp_path / 'outputs'
    assert run.main(['--task', 'ball', '--classifier', 'all', '--source-root', str(fixture.root),
        '--output-root', str(destination), '--dry-run']) == 0
    assert not destination.exists()


def test_cli_all_and_preflight_failure(tmp_path, monkeypatch):
    fixture, _ = synthetic(tmp_path / 'input')
    fixture.protocol['metadata'].loc[fixture.protocol['train_ids'], 'cohort'] = 'TCGA'
    fixture.protocol['metadata'].loc[fixture.protocol['holdout_ids'], 'cohort'] = 'APOLLO'
    monkeypatch.setattr(inputs, 'load_protocol', lambda task: fixture.protocol)
    planned = []
    def evaluate(data, task, classifier, output, **kwargs):
        planned.append((task, classifier))
        return pd.DataFrame([dict(task=task, classifier=classifier, seed=80,
            checkpoint=fixture.checkpoint, representation='full', split='holdout')])
    monkeypatch.setattr(run, 'evaluate', evaluate)
    assert run.main(['--task', 'all', '--classifier', 'all', '--source-root', str(fixture.root),
        '--output-root', str(tmp_path / 'outputs')]) == 0
    assert planned == [(task, classifier) for task in run.TASKS for classifier in run.CLASSIFIERS]
    planned.clear()
    def protocol(task):
        if task == 'brca':
            raise FileNotFoundError('Missing paired input')
        return fixture.protocol
    monkeypatch.setattr(inputs, 'load_protocol', protocol)
    with pytest.raises(FileNotFoundError):
        run.main(['--task', 'all', '--classifier', 'all', '--source-root', str(fixture.root),
            '--output-root', str(tmp_path / 'untouched')])
    assert not planned and not (tmp_path / 'untouched').exists()


def test_boundary_rejects_overlap_external_cohort_and_rare_class(tmp_path):
    _, data = synthetic(tmp_path, apollo=True)
    changed = copy.deepcopy(data)
    changed['train_ids'][0] = changed['holdout_ids'][0]
    with pytest.raises(ValueError, match='overlap'):
        run.validate_data(changed, 'apolo')
    changed = copy.deepcopy(data)
    changed['cohort'].loc[changed['train_ids'][0]] = 'APOLLO'
    with pytest.raises(ValueError, match='TCGA'):
        run.validate_data(changed, 'apolo')
    with pytest.raises(ValueError, match='two training samples'):
        models.inner_partition(['a', 'b', 'c'], [0, 0, 1], models.MLPConfig(), 80)


@pytest.mark.parametrize('fault', ['dimension', 'duplicate', 'nan', 'checkpoint', 'membership'])
def test_cli_reuses_cache_validation(tmp_path, monkeypatch, fault):
    fixture, _ = synthetic(tmp_path / 'input')
    with np.load(fixture.cache, allow_pickle=False) as cache:
        values, ids = cache['embeddings'].copy(), cache['sample_ids'].copy()
        metadata = json.loads(str(cache['metadata']))
    if fault == 'dimension':
        values = values[:, :255]
    elif fault == 'duplicate':
        ids[0] = ids[1]
    elif fault == 'nan':
        values[0, 0] = np.nan
    elif fault == 'checkpoint':
        metadata['signature']['model_name'] = 'wrong'
    else:
        ids[0] = 'wrong_sample'
    np.savez_compressed(fixture.cache, embeddings=values, sample_ids=ids, metadata=json.dumps(metadata))
    monkeypatch.setattr(inputs, 'load_protocol', lambda task: fixture.protocol)
    with pytest.raises(ValueError):
        run.main(['--task', 'ball', '--classifier', 'svm', '--source-root', str(fixture.root), '--dry-run'])


def test_entry_point_help_and_no_encoder_or_omiclass_imports():
    result = subprocess.run([sys.executable, str(run.ROOT / 'run_bulkrnabert_standalone.py'), '--help'],
                            capture_output=True, text=True, check=True)
    assert '--classifier {svm,mlp,all}' in result.stdout
    script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'jax', 'haiku', 'omiclass', 'multiomics_open_research'}:
            raise AssertionError('Forbidden SVM dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from final_results_runs.test_bulkrnabert.standalone.models import fit_svm
import numpy as np
model = fit_svm(np.random.default_rng(0).normal(size=(20, 256)), np.arange(20) % 2, 80)
from final_results_runs.test_bulkrnabert.standalone.run import parser
assert parser().parse_args(['--task', 'ball', '--classifier', 'svm']).classifier == 'svm'
'''
    subprocess.run([sys.executable, '-c', script], cwd=run.ROOT, check=True, capture_output=True, text=True)


def test_smote_inner_training_only_and_reproducible(monkeypatch):
    from imblearn.over_sampling import SMOTE
    X = np.random.default_rng(3).normal(size=(30, 256)).astype('float32')
    y = np.array([0] * 20 + [1] * 10)
    ids = [f's{i}' for i in range(30)]
    cfg = models.MLPConfig(max_epochs=2, resampling='smote')
    train, validation = models.inner_partition(ids, y, cfg, 80)
    captured = []
    original = SMOTE.fit_resample
    def spy(self, features, labels):
        np.testing.assert_array_equal(features, X[train])
        np.testing.assert_array_equal(labels, y[train])
        result = original(self, features, labels)
        captured.append(result[0].copy())
        return result
    monkeypatch.setattr(SMOTE, 'fit_resample', spy)
    model, info = models.fit_mlp(X, y, ids, 2, cfg, 80)
    repeated, other = models.fit_mlp(X, y, ids, 2, cfg, 80)
    assert info == other
    assert all(torch.equal(value, repeated.state_dict()[key]) for key, value in model.state_dict().items())
    X[validation] += 1000
    models.fit_mlp(X, y, ids, 2, cfg, 80)
    np.testing.assert_array_equal(captured[0], captured[2])
    assert info['resampling']['class_counts_after'] == [16, 16]
    assert info['resampling']['synthetic_sample_count'] == 8
    assert info['inner_train_ids'] == [ids[i] for i in train]
    assert models.smote_settings(np.array([0, 0, 1, 1]), 5)['effective_k_neighbors'] is None
    assert models.smote_settings(np.array([0, 0, 0, 1, 1]), 5)['effective_k_neighbors'] == 1
    with pytest.raises(ValueError, match='inner-training'):
        models.smote_settings(np.array([0, 0, 1]), 5)


def test_smote_outputs_and_summary_coexist(tmp_path):
    _, data = synthetic(tmp_path / 'input')
    frames = []
    for method in ['none', 'smote']:
        cfg = models.MLPConfig(max_epochs=2, resampling=method)
        output = tmp_path / method
        frame = run.evaluate(data, 'ball', 'mlp', output, mlp_config=cfg)
        assert set(frame.resampling) == {method}
        manifest = inputs.json_file(output / 'manifest.json')
        assert manifest['training']['resampling']['method'] == method
        run.evaluate(data, 'ball', 'mlp', output, mlp_config=cfg, resume=True)
        frames.append(frame)
    # Simulate an existing pre-SMOTE summary without the new column.
    frames[0].drop(columns='resampling').to_csv(tmp_path / 'summary.csv', index=False)
    run.write_summary(tmp_path, [frames[1]])
    assert set(pd.read_csv(tmp_path / 'summary.csv').resampling) == {'none', 'smote'}


def test_smote_cli_preflight(tmp_path, monkeypatch, capsys):
    fixture, _ = synthetic(tmp_path / 'input')
    monkeypatch.setattr(inputs, 'load_protocol', lambda task: fixture.protocol)
    monkeypatch.setattr(run, 'evaluate', lambda *a, **kw: pytest.fail('Dry-run fitted a model'))
    destination = tmp_path / 'outputs'
    run.main(['--task', 'ball', '--classifier', 'mlp', '--resampling', 'smote',
              '--source-root', str(fixture.root), '--output-root', str(destination), '--dry-run'])
    assert '/mlp_smote/ball/output/' in capsys.readouterr().out
    assert not destination.exists()
    with pytest.raises(ValueError, match='only with'):
        run.main(['--task', 'all', '--classifier', 'all', '--resampling', 'smote'])
