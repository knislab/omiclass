"""Synthetic tests of the standalone article LR adapter; never extract or train article data."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from final_results_runs import evaluate_foundation_logistic as article
from final_results_runs.test_bulkrnabert.logistic import inputs
from final_results_runs.test_bulkrnabert.logistic import run as runner


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))


def make_fixture(tmp_path, binary=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    task = 'cll' if binary else 'ball'
    classes = ['mutated', 'unmutated'] if binary else ['A', 'B', 'C']
    sample_ids = [f'train_{i}' for i in range(24)] + [f'test_{i}' for i in range(6)] + ['unlabeled']
    train, test = sample_ids[:24], sample_ids[24:30]
    labels = [classes[i % len(classes)] for i in range(30)] + [None]
    metadata = pd.DataFrame({'label': labels, 'cohort': ['TRAIN'] * 24 + ['EXTERNAL'] * 6 + [None]}, index=sample_ids)
    metadata_path = tmp_path / 'labels.csv'
    metadata.to_csv(metadata_path)
    cfgdir = tmp_path / 'weights'
    checkpoint = inputs.DEFAULT_CHECKPOINT
    save_json(cfgdir / checkpoint / 'config.json', dict(n_genes=19062, embed_dim=256, num_layers=4, use_log_normalization=True, use_max_normalization=True))
    (cfgdir / checkpoint / 'params.joblib').write_bytes(b'identity only, never unpickled')
    vocabulary = tmp_path / 'vocabulary.txt'
    vocabulary.write_text('\n'.join(f'ENSG{i:011d}' for i in range(19062)))
    reduction = inputs.effective_reduction({}, 'full')
    # Small fold sizes for synthetic tests; the real protocol is tested separately.
    protocol = dict(task=task, metadata_path=metadata_path.resolve(), metadata=metadata,
                    label_column='label', sample_ids=sample_ids, train_ids=train, holdout_ids=test,
                    reference_seed=80, hashes={'labels': inputs.digest(metadata_path)}, references={})
    for rep in runner.REPRESENTATIONS:
        protocol['references'][rep] = {'manifest': 'synthetic_reference', 'outer_folds': 2,
                'inner_folds': 2, 'scoring': 'f1_macro', 'reduction': inputs.effective_reduction({}, rep)}
    rng = np.random.RandomState(4)
    # Deliberately shuffle cache IDs: IDs, never positions, select paired patients.
    order = np.arange(len(sample_ids))[::-1]
    values = rng.normal(size=(len(sample_ids), 256)).astype(np.float32)[order]
    shuffled = [sample_ids[i] for i in order]
    signature = dict(cache_version=1, model_name=checkpoint, sample_ids=shuffled,
            pooling='embeddings_4_gene_mean', preprocessing='official_log10_TPM_then_tokenizer',
            config_sha256=inputs.digest(cfgdir / checkpoint / 'config.json'),
            weights_sha256=inputs.digest(cfgdir / checkpoint / 'params.joblib'),
            vocabulary_sha256=inputs.digest(vocabulary))
    cache_metadata = {'signature': signature, 'embeddings_sha256': hashlib.sha256(values.tobytes()).hexdigest()}
    cache_path = tmp_path / (hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest() + '.npz')
    np.savez_compressed(cache_path, embeddings=values, sample_ids=np.array(shuffled), metadata=json.dumps(cache_metadata))
    config = {'project': {'meta_path': str(metadata_path), 'cluster_column': 'label'},
        'runtime': {'random_state': 80, 'cv_folds': 2, 'scoring': 'f1_macro', 'tuning': {'inner_cv': 2}},
        'bulkrnabert': {'enabled': True, 'model_name': checkpoint, 'checkpoint_directory': str(cfgdir),
                        'vocabulary_path': str(vocabulary), 'reduction': reduction}}
    source = tmp_path / 'sources' / task / 'output/full/Omiclass_BulkRNABert-synthetic'
    source.mkdir(parents=True)
    (source / 'config.freeze.yaml').write_text(yaml.safe_dump(config))
    save_json(source / 'split.json', {'train_index': train, 'test_index': test})
    save_json(source / 'bulkrnabert/bulkrnabert_metadata.json', {
        'backend': 'bulkrnabert', 'feature_shape': [256, len(sample_ids)], 'sample_ids': shuffled,
        'signature': signature, 'cache_path': str(cache_path)})
    return SimpleNamespace(protocol=protocol, source=source, cache=cache_path, checkpoint=checkpoint,
                           cfgdir=cfgdir, vocabulary=vocabulary, values=values, shuffled=shuffled,
                           root=tmp_path / 'sources', config=config)


@pytest.fixture
def fixture(tmp_path):
    return make_fixture(tmp_path)


def load(f):
    return inputs.load_source(f.source, f.protocol, 80, f.checkpoint)


def cache_load(f):
    return inputs.read_cache(f.cache, checkpoint=f.checkpoint, checkpoint_directory=f.cfgdir,
                            vocabulary_path=f.vocabulary, expected_ids=f.protocol['sample_ids'])


def mutate_cache(f, mutator):
    with np.load(f.cache, allow_pickle=False) as z:
        value = {k: z[k].copy() for k in z.files}
    mutator(value)
    np.savez(f.cache, **value)


def test_cache_alignment_and_prespecified_partition(fixture):
    data = load(fixture)
    assert data['train_ids'] == fixture.protocol['train_ids']
    assert data['holdout_ids'] == fixture.protocol['holdout_ids']
    np.testing.assert_array_equal(data['X'].loc['train_0'], fixture.values[-1])
    assert data['X'].shape == (31, 256)
    assert data['unlabeled_ids'] == ['unlabeled']
    assert data['external_ids'] == []
    assert data['classes'] == ['A', 'B', 'C']


@pytest.mark.parametrize('problem', ['dimension', 'nonfinite', 'duplicate', 'checksum', 'dataset', 'checkpoint', 'vocabulary'])
def test_invalid_cache_is_rejected(fixture, problem):
    if problem == 'dimension':
        mutate_cache(fixture, lambda z: z.update(embeddings=z['embeddings'][:, :255]))
    elif problem == 'nonfinite':
        def corrupt(z): z['embeddings'][0, 0] = np.nan
        mutate_cache(fixture, corrupt)
    elif problem == 'duplicate':
        def corrupt(z): z['sample_ids'][0] = z['sample_ids'][1]
        mutate_cache(fixture, corrupt)
    elif problem == 'checksum':
        def corrupt(z): z['embeddings'][0, 0] += 1
        mutate_cache(fixture, corrupt)
    elif problem == 'dataset':
        fixture.protocol['sample_ids'] = fixture.protocol['sample_ids'][:-1] + ['other_dataset']
    elif problem == 'checkpoint':
        fixture.checkpoint = 'other_checkpoint'
    else:
        fixture.vocabulary.write_text('changed')
    with pytest.raises(ValueError):
        cache_load(fixture)


def test_alternative_checkpoint_requires_explicit_identity(fixture):
    fixture.checkpoint = 'supported_alternative'
    old = fixture.cfgdir / inputs.DEFAULT_CHECKPOINT
    old.rename(fixture.cfgdir / fixture.checkpoint)
    with np.load(fixture.cache, allow_pickle=False) as z:
        saved = json.loads(str(z['metadata']))
        signature = saved['signature']
        signature['model_name'] = fixture.checkpoint
        new_path = fixture.cache.with_name(hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest() + '.npz')
        np.savez(new_path, embeddings=z['embeddings'], sample_ids=z['sample_ids'], metadata=json.dumps(saved))
    fixture.cache = new_path
    frame, _, _ = cache_load(fixture)
    assert frame.shape[1] == 256


def test_select_sources_never_generates_missing_seed(fixture):
    selected = inputs.select_sources(fixture.root, fixture.protocol, [80], fixture.checkpoint)
    assert selected == {80: fixture.source}
    with pytest.raises(FileNotFoundError, match='never extracts or generates splits'):
        inputs.select_sources(fixture.root, fixture.protocol, [80, 11], fixture.checkpoint)


def test_wrong_split_and_labels_rejected(fixture):
    original = inputs.json_file(fixture.source / 'split.json')
    changed = copy.deepcopy(original)
    changed['train_index'][0], changed['test_index'][0] = changed['test_index'][0], changed['train_index'][0]
    save_json(fixture.source / 'split.json', changed)
    with pytest.raises(ValueError, match='prespecified article partition'):
        load(fixture)
    save_json(fixture.source / 'split.json', original)
    fixture.protocol['metadata'].loc['test_0', 'label'] = 'unseen_external_class'
    with pytest.raises(ValueError, match='absent from training'):
        load(fixture)


def test_source_cache_provenance_mismatch_rejected(fixture):
    p = fixture.source / 'bulkrnabert/bulkrnabert_metadata.json'
    data = inputs.json_file(p)
    data['signature']['weights_sha256'] = 'other checkpoint'
    save_json(p, data)
    with pytest.raises(ValueError, match='does not describe'):
        load(fixture)


def test_real_article_protocols_are_consistent_without_fitting():
    # Read-only reference checks: no feature extraction or training on article data.
    for task, sizes in [('ball', (1256, 477)), ('cll', (538, 87)), ('brca', (635, 159)), ('apolo', (803, 68))]:
        protocol = inputs.load_protocol(task)
        assert (len(protocol['train_ids']), len(protocol['holdout_ids'])) == sizes
        assert protocol['reference_seed'] == 80
        for reference in protocol['references'].values():
            assert (reference['outer_folds'], reference['inner_folds'], reference['scoring']) == (5, 3, 'f1_macro')
            assert reference['reduction']['supervised_dimensions'] == 200
            assert reference['reduction']['pca_components'] == 50


def test_exact_existing_lr_protocol_and_pure_imports(fixture):
    data = load(fixture)
    run = inputs.loaded_run(fixture.source, data, fixture.protocol, 'full')
    pipe = article.build_pipeline(run)
    lr = pipe.named_steps['logistic_regression']
    assert isinstance(lr, LogisticRegression)
    assert lr.get_params()['solver'] == 'lbfgs' and lr.penalty == 'l2'
    assert lr.max_iter == 5000 and lr.random_state == 80 and lr.tol == 1e-4
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)
    assert article._parameter_grid() == {'logistic_regression__C': [0.01, 0.1, 1., 10., 100.],
                                        'logistic_regression__class_weight': [None, 'balanced']}
    subprocess.run([sys.executable, '-c',
        'import sys; from final_results_runs.test_bulkrnabert.logistic import run; '
        'assert not any(k.startswith(("omiclass.train", "omiclass.models", "jax", "haiku")) for k in sys.modules)'],
        cwd=inputs.ROOT, check=True)


@pytest.mark.parametrize('representation', runner.REPRESENTATIONS)
def test_all_modes_fit_on_nested_training_only(fixture, monkeypatch, representation, tmp_path):
    data = load(fixture)
    run = inputs.loaded_run(fixture.source, data, fixture.protocol, representation)
    fits = []
    original_fit = article.EmbeddingRepresentation.fit
    def fit(self, X, y=None):
        fits.append(set(X.index))
        assert set(X.index) <= set(run.train_ids)
        return original_fit(self, X, y)
    monkeypatch.setattr(article.EmbeddingRepresentation, 'fit', fit)
    # Same pipeline/search machinery, one grid cell keeps this leakage test small.
    monkeypatch.setattr(article, '_parameter_grid', lambda: {'logistic_regression__C': [1.], 'logistic_regression__class_weight': [None]})
    original_scaler_fit = StandardScaler.fit
    scaler_fits = []
    def scale(self, X, y=None, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            assert set(X.index) <= set(run.train_ids)
            scaler_fits.append(set(X.index))
        return original_scaler_fit(self, X, y, sample_weight=sample_weight)
    monkeypatch.setattr(StandardScaler, 'fit', scale)
    original_selector = article.SupervisedEmbeddingSelector.fit
    def select(self, X, y):
        assert set(X.index) <= set(run.train_ids)
        return original_selector(self, X, y)
    monkeypatch.setattr(article.SupervisedEmbeddingSelector, 'fit', select)
    output = tmp_path / f'result-{representation}'
    summary = runner.evaluate(run, output, data['provenance'], fixture.protocol)
    assert fits and scaler_fits
    outer = article.safe_stratified_cv(run.y.loc[run.train_ids], 2, 80)
    allowed = [set(run.train_ids)]
    Xtrain, ytrain = run.X.loc[run.train_ids], run.y.loc[run.train_ids].astype(int)
    for train_idx, _ in outer.split(Xtrain, ytrain):
        fold_X, fold_y = Xtrain.iloc[train_idx], ytrain.iloc[train_idx]
        allowed.append(set(fold_X.index))
        inner = article.safe_stratified_cv(fold_y, 2, 80)
        allowed.extend(set(fold_X.iloc[ix].index) for ix, _ in inner.split(fold_X, fold_y))
    inner = article.safe_stratified_cv(ytrain, 2, 80)
    allowed.extend(set(Xtrain.iloc[ix].index) for ix, _ in inner.split(Xtrain, ytrain))
    assert all(group in allowed for group in fits)
    assert all(group in allowed for group in scaler_fits)
    model = joblib.load(output / 'pipeline.joblib')
    before = pickle.dumps(model.named_steps['representation'])
    model.predict(run.X.loc[run.holdout_ids])
    assert before == pickle.dumps(model.named_steps['representation'])
    if representation in ['pca', 'supervised_pca']:
        fitted_rep = model.named_steps['representation']
        selected = fitted_rep.selector_.transform(Xtrain) if fitted_rep.selector_ is not None else Xtrain
        np.testing.assert_allclose(fitted_rep.scaler_.mean_, selected.to_numpy(dtype=np.float64).mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(fitted_rep.pca_.mean_, fitted_rep.scaler_.transform(selected).mean(axis=0), atol=1e-7)
    prediction = pd.read_csv(output / 'predictions.csv')
    probabilities = prediction.filter(regex='^probability_class_').to_numpy()
    assert probabilities.shape == (31, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert {'precision_macro', 'recall_macro', 'auroc_macro', 'average_precision_macro'} <= set(summary)
    assert set(summary.seed) == {80}
    manifest = inputs.json_file(output / 'manifest.json')
    assert manifest['partitions']['train_index'] == run.train_ids
    assert manifest['partitions']['test_index'] == run.holdout_ids
    assert manifest['classifier'] == 'logistic_regression'
    assert manifest['input_dimensions'] == 256
    expected_dimension = {'full': 256, 'supervised': 200, 'pca': 23, 'supervised_pca': 23}[representation]
    assert manifest['output_dimensions'] == expected_dimension


def test_binary_protocol_reproducibility_and_resume(tmp_path):
    f = make_fixture(tmp_path, binary=True)
    data = load(f)
    run = inputs.loaded_run(f.source, data, f.protocol, 'full')
    summaries = []
    for name in ['first', 'second']:
        summaries.append(runner.evaluate(run, tmp_path / name, data['provenance'], f.protocol))
    first = pd.read_csv(tmp_path / 'first/predictions.csv')
    second = pd.read_csv(tmp_path / 'second/predictions.csv')
    pd.testing.assert_frame_equal(first, second)
    probabilities = first.filter(regex='^probability_class_')
    assert probabilities.shape == (31, 2)
    scores = pd.read_csv(tmp_path / 'first/metrics.csv').set_index('split')
    assert 'average_precision_macro' not in scores and 'auroc_macro' not in scores
    heldout = first.loc[first['split'] == 'holdout']
    assert scores.loc['holdout', 'f1_binary'] == pytest.approx(f1_score(heldout.y_true, heldout.y_pred))
    assert scores.loc['holdout', 'f1_macro'] == pytest.approx(f1_score(heldout.y_true, heldout.y_pred, average='macro'))
    assert scores.loc['holdout', 'auroc_binary'] == pytest.approx(roc_auc_score(heldout.y_true, heldout.probability_class_1))
    assert scores.loc['holdout', 'average_precision_binary'] == pytest.approx(average_precision_score(heldout.y_true, heldout.probability_class_1))
    before = (tmp_path / 'first/pipeline.joblib').stat().st_mtime_ns
    runner.evaluate(run, tmp_path / 'first', data['provenance'], f.protocol, resume=True)
    assert (tmp_path / 'first/pipeline.joblib').stat().st_mtime_ns == before
    runner.write_summary(tmp_path, [summaries[0], summaries[1].assign(seed=11)])
    runner.write_summary(tmp_path, [summaries[0]])
    combined = pd.read_csv(tmp_path / 'summary.csv')
    assert len(combined) == 4 and set(combined.seed) == {11, 80}
    with (tmp_path / 'first/predictions.csv').open('a') as handle: handle.write('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        runner.evaluate(run, tmp_path / 'first', data['provenance'], f.protocol, resume=True)


def test_apollo_membership_and_no_leakage(tmp_path, monkeypatch):
    train = [f'TCGA_{i}' for i in range(803)]
    test = [f'APOLLO_{i}' for i in range(68)]
    protocol = {'task': 'apolo', 'reference_seed': 80, 'train_ids': train, 'holdout_ids': test,
                'metadata': pd.DataFrame({'cohort': ['TCGA'] * 803 + ['APOLLO'] * 68}, index=train + test)}
    inputs.validate_partition(protocol, train, test, 80)
    with pytest.raises(ValueError):
        inputs.validate_partition(protocol, train + test[:1], test[1:], 80)
    # Tiny synthetic APOLLO evaluation exercises actual nested fitting/output logic.
    f = make_fixture(tmp_path / 'small')
    data = load(f)
    run = inputs.loaded_run(f.source, data, f.protocol, 'pca')
    run.candidate = article.RunCandidate('bulkrnabert', 'apolo', 'pca', f.source, '', True, ())
    monkeypatch.setattr(article, '_parameter_grid', lambda: {'logistic_regression__C': [1.], 'logistic_regression__class_weight': [None]})
    original = article._search
    seen = []
    def search(run, X, y, **kwargs):
        seen.append(set(X.index))
        assert not set(run.holdout_ids) & set(X.index)
        return original(run, X, y, **kwargs)
    monkeypatch.setattr(article, '_search', search)
    summary = runner.evaluate(run, tmp_path / 'apollo-result', data['provenance'], f.protocol)
    assert seen and summary.loc[summary['split'].eq('holdout'), 'evaluation_role'].item() == 'external_apollo'


def test_cli_all_modes_and_multiple_seeds_validate_without_fitting(fixture, monkeypatch, tmp_path):
    other = fixture.source.with_name('Omiclass_BulkRNABert-seed11')
    other.mkdir()
    config = copy.deepcopy(fixture.config)
    config['runtime']['random_state'] = 11
    (other / 'config.freeze.yaml').write_text(yaml.safe_dump(config))
    save_json(other / 'split.json', inputs.json_file(fixture.source / 'split.json'))
    save_json(other / 'bulkrnabert/bulkrnabert_metadata.json', inputs.json_file(fixture.source / 'bulkrnabert/bulkrnabert_metadata.json'))
    monkeypatch.setattr(inputs, 'load_protocol', lambda task: fixture.protocol)
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run must never fit')
    monkeypatch.setattr(article, 'evaluate_run', forbidden)
    output = tmp_path / 'must-not-exist'
    assert runner.main(['--task', 'ball', '--seeds', '80', '11', '--representation', 'all',
        '--source-run', f'80={fixture.source}', '--source-run', f'11={other}',
        '--output-root', str(output), '--dry-run']) == 0
    assert not output.exists()
