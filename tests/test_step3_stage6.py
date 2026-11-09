"""Synthetic Stage 6 checks; no real encoder, checkpoint or JAX execution."""
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from omiclass.config.loaders import load_config
from omiclass.foundation import extraction, article_head, heads
from omiclass.foundation.io import load_token_arrays, load_token_manifest, token_schema, write_json
from omiclass.utils.reproducibility import random_state_scope


def snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.random.get_rng_state().clone(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        warn=torch.is_deterministic_algorithms_warn_only_enabled(), threads=torch.get_num_threads(),
        benchmark=torch.backends.cudnn.benchmark, cudnn=torch.backends.cudnn.deterministic,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def assert_snapshot(expected):
    current = snapshot()
    for key in set(expected) - {'numpy', 'cpu'}:
        assert current[key] == expected[key], key
    np.testing.assert_array_equal(current['numpy'][1], expected['numpy'][1])
    assert current['numpy'][0] == expected['numpy'][0]
    assert current['numpy'][2:] == expected['numpy'][2:]
    assert torch.equal(current['cpu'], expected['cpu'])


@pytest.mark.parametrize('fail', [False, True])
def test_shared_scope_restores_random_states_and_settings_even_on_failure(fail):
    before = snapshot()
    try:
        with random_state_scope(17, threads=1, deterministic=True, cudnn_benchmark=False, cublas=True):
            random.random(); np.random.random(); torch.rand(3)
            torch.use_deterministic_algorithms(False, warn_only=True)
            torch.backends.cudnn.deterministic = not before['cudnn']
            torch.backends.cudnn.allow_tf32 = not before['cudnn_tf32']
            torch.backends.cuda.matmul.allow_tf32 = not before['matmul_tf32']
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            if fail:
                raise RuntimeError('synthetic scope failure')
    except RuntimeError:
        assert fail
    assert_snapshot(before)


def config(tmp_path, model='bulkformer', **foundation):
    return load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label',
        foundation=dict(enabled=True, model=model, classifier='svm', **foundation),
        bulkformer_device='cpu', cellfm_device='cpu'))


@pytest.mark.parametrize('model', ['bulkformer', 'cellfm_1view', 'bulkrnabert'])
@pytest.mark.parametrize('fail', [False, True])
def test_extraction_restores_state_and_reports_the_effective_seed(tmp_path, monkeypatch, model, fail):
    cfg = config(tmp_path, model, extraction_seed=51)
    cfg.cellfm.random_state = 13
    counts = pd.DataFrame([[2., 3.], [4., 5.]], index=['g1', 'g2'], columns=['a', 'b'])
    monkeypatch.setattr(extraction, 'continuous_input', lambda values, cfg: values)
    def run(values):
        random.random(); np.random.random()
        if model != 'bulkrnabert':
            assert torch.are_deterministic_algorithms_enabled() == before['deterministic']
            assert torch.backends.cudnn.benchmark == before['benchmark']
            torch.rand(3)
            torch.backends.cudnn.benchmark = True
        if fail:
            raise RuntimeError('synthetic encoder failure')
        return extraction.Extraction(values.T, {})
    if model == 'bulkrnabert':
        monkeypatch.setattr(extraction, 'extract_bert', lambda values, *args: run(values))
    else:
        from omiclass.features import bulkformer, cellfm
        class Encoder:
            def __init__(self, *args, **kwargs): pass
            def extract(self, values):
                result = run(values)
                return SimpleNamespace(features=result.features.T, metadata={})
        monkeypatch.setattr(bulkformer if model == 'bulkformer' else cellfm,
            'BulkFormerFeatureExtractor' if model == 'bulkformer' else 'CellFMFeatureExtractor', Encoder)
    before = snapshot()
    if fail:
        with pytest.raises(RuntimeError, match='synthetic encoder'):
            extraction.extract(counts, cfg, tmp_path / 'out')
    else:
        extraction.extract(counts, cfg, tmp_path / 'out')
        info = json.loads((tmp_path / 'out/extraction.json').read_text())
        assert info['encoder_seed'] == {'bulkformer': 51, 'cellfm_1view': 13, 'bulkrnabert': 0}[model]
        assert info['configured_extraction_seed'] == 51
    assert_snapshot(before)


def article_args(tmp_path):
    ids = [f's{i}' for i in range(8)]
    features = tmp_path / 'features.csv'
    pd.DataFrame(np.arange(24, dtype=np.float32).reshape(3, 8), columns=ids).to_csv(features)
    labels, split = tmp_path / 'labels.csv', tmp_path / 'split.json'
    pd.DataFrame({'sample_id': ids, 'label': ['a', 'b'] * 4}).to_csv(labels, index=False)
    write_json(split, dict(train_index=ids[:6], test_index=ids[6:]))
    return SimpleNamespace(variant='mean', smote_k_neighbors=5, resampling='none', out=tmp_path / 'training',
        hidden_dims=[64], token_dim=8, epochs=1, batch_size=2, dropout=.2,
        learning_rate=.001, weight_decay=.0001, features=features, cache=None,
        seed=80, device='cpu', labels=labels, split=split, id_column='sample_id', label_column='label')


@pytest.mark.parametrize('fail', [False, True])
def test_article_training_restores_state_on_success_and_mid_training_failure(tmp_path, monkeypatch, fail):
    args = article_args(tmp_path)
    before = snapshot()
    if fail:
        def broken(*a, **kw):
            random.random(); np.random.random(); torch.rand(1)
            raise RuntimeError('synthetic optimizer failure')
        monkeypatch.setattr(torch.optim.AdamW, 'step', broken)
        with pytest.raises(RuntimeError, match='optimizer failure'):
            article_head.train(args)
    else:
        article_head.train(args)
        assert (args.out / 'head.pt').is_file()
    assert_snapshot(before)


def test_head_dispatch_restores_state_after_training_failure(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.foundation.classifier = 'lr'
    def broken(*a, **kw):
        random.random(); np.random.random(); torch.rand(2)
        torch.use_deterministic_algorithms(True)
        raise RuntimeError('synthetic fit failure')
    monkeypatch.setattr(heads, 'fit_lr', broken)
    before = snapshot()
    with pytest.raises(RuntimeError, match='fit failure'):
        heads.fit(cfg, pd.DataFrame(), pd.Series(dtype=int), [], [], [], None, tmp_path / 'head')
    assert_snapshot(before)


@pytest.mark.parametrize('fail', [False, True])
def test_cuda_rng_restoration_with_mocked_devices(monkeypatch, fail):
    state = {'value': torch.tensor([1, 2, 3], dtype=torch.uint8)}
    before = state['value'].clone()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [state['value'].clone()])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda values: state.update(value=values[0].clone()))
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: state.update(value=torch.tensor([seed])))
    try:
        with random_state_scope(17):
            assert state['value'].item() == 17
            state['value'] = torch.tensor([99])
            if fail: raise RuntimeError('mock CUDA failure')
    except RuntimeError:
        assert fail
    assert torch.equal(state['value'], before)


def test_cpu_scope_does_not_initialize_unused_cuda_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: pytest.fail('Unused CUDA initialization'))
    with random_state_scope(17):
        torch.rand(1)


class Prepare:
    """Small collator operations matching the existing selection contract."""
    def seperate(self, data):
        return data, np.flatnonzero(data), np.flatnonzero(data == 0)
    def sample(self, data, nonzero, zero):
        probabilities = np.log1p(data[nonzero]); probabilities /= probabilities.sum()
        selected = np.sort(np.random.choice(nonzero, 3, replace=False, p=probabilities))
        return data, selected, np.array([]), np.array([]), len(selected)
    def compress(self, data, indices):
        return data, data[indices], indices
    def bayes(self, data, total):
        return data, data, total, total
    def normalize(self, data, total):
        return np.log1p(data / total * 1e4).astype(np.float32), total
    def cat_st(self, selected, total):
        return np.array([selected, total], dtype=np.float32)
    def attn_mask(self, length):
        return np.arange(4) < length
    def mask(self, values):
        return np.column_stack([np.ones(len(values)), values]), None
    def pad_zero(self, values):
        return np.concatenate([values, np.zeros((4-len(values), *values.shape[1:]), dtype=values.dtype)])


def prepared(ids=('a', 'b'), reverse_genes=False):
    from omiclass.features.cellfm import CellFMPreparedInput
    values = dict(a=[1, 4, 8, 16, 2, 0], b=[2, 7, 3, 1, 9, 0])
    genes = np.arange(1, 7)
    order = np.arange(6)[::-1] if reverse_genes else np.arange(6)
    return CellFMPreparedInput(np.array([values[s] for s in ids])[:, order], genes[order], list(ids),
        [str(g) for g in genes[order]], [], np.array([31] * len(ids)))


def test_single_view_is_sample_and_batch_stable_and_uses_view_zero_seed(monkeypatch):
    from omiclass.features import cellfm
    monkeypatch.setattr(cellfm, 'CELLFM_EMBEDDING_DIMENSION', 3)
    def pool(model, expression, genes, st, valid):
        return torch.stack([expression[:, :, 1].sum(1), genes[:, 0].float(), genes[:, 1].float()], dim=1)
    monkeypatch.setattr(cellfm, '_pool_cellfm_batch', pool)
    def run(ids, batch, reverse=False):
        return cellfm._extract_configured_views(cfg=SimpleNamespace(num_views=1, batch_size=batch, random_state=17),
            utils=None, torch=torch, model=None, prep=Prepare(), prepared=prepared(ids, reverse), device='cpu')
    before = snapshot()
    a = run(('a', 'b'), 1)
    b = run(('b', 'a'), 2, True)
    external = run(('a',), 1)
    np.testing.assert_array_equal(a[0], b[0][::-1])
    np.testing.assert_array_equal(a[0][0], external[0][0])
    expr, genes, _, mask = cellfm._tokenize_sample_view(Prepare(), prepared().counts[0], prepared().gene_ids,
        seed=cellfm._stable_view_seed('a', 17, 0))
    assert a[0][0, 1] == genes[0] and a[0][0, 2] == genes[1]
    assert a[1] == [3, 3] and a[3] == 'single_view'
    assert a[2] == 'sha256_sample_id_global_seed_zero_based_view_index'
    assert_snapshot(before)


def test_sampling_failure_restores_numpy_rng():
    from omiclass.features.cellfm import _tokenize_sample_view
    class Broken(Prepare):
        def sample(self, *args):
            np.random.rand(10)
            raise RuntimeError('sampling failure')
    before = snapshot()
    with pytest.raises(RuntimeError, match='sampling failure'):
        _tokenize_sample_view(Broken(), np.ones(6), np.arange(1, 7), seed=10)
    assert_snapshot(before)


@pytest.mark.parametrize('views', [1, 20])
@pytest.mark.parametrize('fail', [False, True])
def test_corrected_cellfm_preserves_sampled_ids_masks_auxiliaries_and_restores_state(tmp_path, monkeypatch, views, fail):
    from omiclass.features import cellfm
    from omiclass.foundation import cellfm_corrected
    cfg = config(tmp_path, f'cellfm_{views}view').cellfm
    cfg.batch_size = 2
    vocab = [str(i) for i in range(1, 7)]
    source = prepared(reverse_genes=True)
    counts = pd.DataFrame(source.counts.T, index=source.mapped_genes, columns=source.sample_ids)
    models = []
    class Model:
        def __init__(self, **kwargs):
            random.random(); np.random.random(); torch.rand(1)
            self.net = SimpleNamespace(encode=self.encode, value_enc=torch.nn.Identity(), encoder=[torch.nn.Identity()])
            models.append(self)
        def to(self, device): return self
        def load_model(self, **kwargs): pass
        def eval(self): return self
        def requires_grad_(self, value): return self
        def encode(self, expression, genes, st, valid):
            values = self.net.encoder[0](self.net.value_enc(expression[:, :, 1]))
            if fail:
                random.random(); np.random.random(); torch.rand(1)
                raise RuntimeError('corrected encoder failure')
            special = torch.zeros((len(values), 3))
            return torch.cat([special, values], dim=1)[:, :, None].expand(-1, -1, 1536), None
    utils = SimpleNamespace(Config_80M=lambda: SimpleNamespace(nonz_len=2048, enc_dims=1536),
                            Prepare=lambda **kwargs: Prepare())
    monkeypatch.setattr(cellfm, '_validate_resources', lambda cfg: None)
    monkeypatch.setattr(cellfm, 'load_cellfm_vocabulary', lambda path: vocab)
    monkeypatch.setattr(cellfm, '_import_cellfm', lambda cfg: (utils, SimpleNamespace(Cell_FM=Model)))
    original = cellfm_corrected.continuous_expression(counts, cfg)
    corrected = original[['a']] + .25
    observer = extraction.TokenObserver(tmp_path / 'tokens', 'cellfm', views)
    before = snapshot()
    if fail:
        with pytest.raises(RuntimeError, match='corrected encoder failure'):
            cellfm_corrected.extract_corrected(counts, corrected, cfg, tmp_path, observer)
    else:
        result, metadata = cellfm_corrected.extract_corrected(counts, corrected, cfg, tmp_path, observer)
        observer.finish(counts.columns)
        assert result.shape == (1, 1536) and metadata['num_views'] == views
        assert metadata['view_seed_strategy'] == 'sha256_sample_id_global_seed_zero_based_view_index'
        assert load_token_manifest(tmp_path / 'tokens')['metadata']['token_schema']['identity_mode'] == 'sampled_positions'
    for path in (tmp_path / 'cellfm_views').glob('*.npz'):
        with np.load(path) as audit:
            ids, valid = audit['gene_ids'], audit['valid'].astype(bool)
            prior, after = audit['original_values'], audit['corrected_values']
            # Unmask channels, padding and the non-target sample remain unchanged.
            np.testing.assert_array_equal(prior[:, :, 0], after[:, :, 0])
            np.testing.assert_array_equal(prior[~valid], after[~valid])
            np.testing.assert_array_equal(prior[1], after[1])
            selected = [vocab[int(i)-1] for i in ids[0, valid[0]]]
            np.testing.assert_array_equal(after[0, valid[0], 1], corrected.loc[selected, 'a'])
            assert np.all(np.diff(ids[0, valid[0]]) > 0)
    assert models[0].net.encode == models[0].encode
    assert not models[0].net.value_enc._forward_hooks
    assert not models[0].net.encoder[0]._forward_hooks
    assert_snapshot(before)


class Tokenizer:
    def __init__(self, bins=64, factor=6):
        self.bins, self.factor, self.calls = bins, factor, 0
    def batch_tokenize(self, values):
        self.calls += 1
        values /= self.factor
        tokens = np.digitize(values, np.linspace(0, 1, self.bins))
        tokens[values == 0] = 0
        return tokens


def test_bert_parent_and_worker_paths_apply_log_and_tokenizer_normalization_once(tmp_path):
    from omiclass.features.bulkrnabert import N_GENES, prepare_tpm
    from omiclass.features.bulkrnabert_worker import tokenize_values
    genes = [f'ENSG{i:011d}' for i in range(N_GENES)]
    path = tmp_path / 'genes.txt'; path.write_text('\n'.join(genes))
    cfg = config(tmp_path, 'bulkrnabert', input_scale='tpm')
    cfg.bulkrnabert.vocabulary_path = path
    tpm = pd.DataFrame({'b': np.linspace(0, 2000, N_GENES), 'a': np.linspace(1, 3000, N_GENES)}, index=genes)
    frame = extraction.continuous_input(tpm.iloc[::-1], cfg)
    values, _ = prepare_tpm(tpm, genes, ['b', 'a'])
    np.testing.assert_array_equal(frame.to_numpy().T, np.log10(values + 1))
    checkpoint = SimpleNamespace(n_genes=N_GENES, n_expressions_bins=64)
    calls = []
    def official(values, cfg):
        calls.append(True)
        return np.log10(values.to_numpy() + 1)
    parent = Tokenizer(); worker = Tokenizer()
    original_logged, original_tpm = frame.to_numpy().copy(), values.copy()
    tokens_a = tokenize_values(frame.to_numpy().T, input_scale='log10_tpm', config=checkpoint,
                              tokenizer=parent, preprocess=official)
    assert not calls
    tokens_b = tokenize_values(values, input_scale='tpm', config=checkpoint, tokenizer=worker, preprocess=official)
    np.testing.assert_array_equal(tokens_a, tokens_b)
    np.testing.assert_array_equal(frame.to_numpy(), original_logged)
    np.testing.assert_array_equal(values, original_tpm)
    assert len(calls) == parent.calls == worker.calls == 1


@pytest.mark.parametrize('values', [[[0, 6]], [[0, 7]], [[0, -1]], [[0, np.nan]]])
def test_bert_tokenizer_bounds_reject_overflow_and_negative_or_nonfinite_values(values):
    from omiclass.features.bulkrnabert_worker import tokenize_values
    cfg = SimpleNamespace(n_genes=2, n_expressions_bins=64)
    with pytest.raises(ValueError):
        tokenize_values(np.array(values, dtype=np.float32), input_scale='log10_tpm', config=cfg,
                        tokenizer=Tokenizer(), preprocess=lambda *a: pytest.fail('Double log'))


@pytest.mark.parametrize('bad', ['genes_shape', 'mask_values', 'mask_order', 'duplicate_gene', 'padding_gene', 'outside_vocab'])
def test_cellfm_token_identity_mask_and_padding_contract(tmp_path, bad):
    observer = extraction.TokenObserver(tmp_path / 'tokens', 'cellfm', 1)
    observer.set_cellfm_vocabulary(['a', 'b'], random_state=17)
    x = np.zeros((1, 3, 2), dtype=np.float32)
    mask, genes = np.array([[1, 1, 0]]), np.array([[1, 2, 0]])
    if bad == 'genes_shape': genes = genes[:, :2]
    if bad == 'mask_values': mask[0, 0] = 2
    if bad == 'mask_order': mask, genes = np.array([[1, 0, 1]]), np.array([[1, 0, 2]])
    if bad == 'duplicate_gene': genes[0, 1] = 1
    if bad == 'padding_gene': genes[0, 2] = 1
    if bad == 'outside_vocab': genes[0, 1] = 3
    with pytest.raises(ValueError): observer(x, mask, genes, ['sample'], 0)


def test_fixed_identity_order_cannot_change_and_cellfm_membership_can(tmp_path):
    x = np.array([[[1., 2.], [3., 4.]]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    fixed = extraction.TokenObserver(tmp_path / 'fixed', 'bulkformer', 1)
    fixed(x, mask, np.array([['g1', 'g2']]), ['a'], 0)
    with pytest.raises(ValueError, match='gene order'):
        fixed(x, mask, np.array([['g2', 'g1']]), ['b'], 0)
    variable = extraction.TokenObserver(tmp_path / 'variable', 'cellfm', 1)
    variable(x, mask, np.array([[1, 2]]), ['a'], 0)
    variable(x, mask, np.array([[2, 3]]), ['b'], 0)
    variable.finish(['a', 'b'])
    schema = load_token_manifest(tmp_path / 'variable')['metadata']['token_schema']
    assert schema['identity_mode'] == 'sampled_positions'
    assert 'gene_order_sha256' not in schema


def test_bert_shared_gene_identity_file_and_legacy_npy_cache(tmp_path):
    values = np.ones((3, 2), np.float32)
    np.save(tmp_path / 'tokens.npy', values)
    np.save(tmp_path / 'gene_ids.npy', np.array(['a', 'b', 'c']))
    schema = token_schema('bulkrnabert', values.shape, ['a', 'b', 'c'])
    record = dict(file='tokens.npy', gene_ids_file='gene_ids.npy', sample_id='a', view=0)
    x, mask = load_token_arrays(tmp_path, record, schema=schema)
    np.testing.assert_array_equal(x, values); assert mask.all()
    np.save(tmp_path / 'gene_ids.npy', np.array(['b', 'a', 'c']))
    with pytest.raises(ValueError, match='gene order'):
        load_token_arrays(tmp_path, record, schema=schema)
    # An actual historical record has no explicit gene file or expected schema.
    load_token_arrays(tmp_path, dict(file='tokens.npy'))


@pytest.mark.parametrize('backend', ['bulkformer', 'cellfm'])
def test_saved_token_schema_replay_and_view_or_identity_mismatch(tmp_path, backend):
    def cache(name, genes, views=1):
        directory = tmp_path / name
        observer = extraction.TokenObserver(directory, backend, views)
        if backend == 'cellfm':
            observer.set_cellfm_vocabulary(['a', 'b', 'c'], random_state=17)
        for view in range(views):
            observer(np.ones((1, 2, 2), dtype=np.float32), np.ones((1, 2), bool),
                     np.asarray([genes]), ['sample'], view)
        observer.finish(['sample'])
        return directory
    genes = ['g1', 'g2'] if backend == 'bulkformer' else [1, 2]
    original = cache('original', genes)
    schema = load_token_manifest(original)['metadata']['token_schema']
    with random_state_scope(80):
        network = article_head.make_head(2, 2, [64], 8, .2, 2)
        state = {key: value.detach().numpy().copy() for key, value in network.state_dict().items()}
    model = heads.TokenModel(state, dict(family='article', width=2, token_count=2,
        hidden_dims=[64], token_dim=8, dropout=.2), ['a', 'b'])
    model.token_schema_, model.token_num_views_ = schema, 1
    path = tmp_path / 'model.joblib'; joblib.dump(model, path); before = path.read_bytes()
    saved = joblib.load(path)
    np.testing.assert_array_equal(saved.predict_proba(original, ['sample']), model.predict_proba(original, ['sample']))
    different_views = cache('two_views', genes, 2)
    with pytest.raises(ValueError, match='view count'):
        saved.predict_proba(different_views, ['sample'])
    changed = cache('changed', ['g2', 'g1'] if backend == 'bulkformer' else [2, 3])
    if backend == 'bulkformer':
        with pytest.raises(ValueError, match='protocol differs'):
            saved.predict_proba(changed, ['sample'])
    else:
        # Varying membership is allowed by the approved sampled-position protocol.
        np.testing.assert_array_equal(saved.predict_proba(changed, ['sample']), saved.predict_proba(original, ['sample']))
    legacy_cache = tmp_path / 'legacy_cache'; legacy_cache.mkdir()
    np.save(legacy_cache / 'sample.npy', np.ones((2, 2), np.float32))
    write_json(legacy_cache / 'manifest.json', dict(format_version=1, backend=backend,
        metadata=dict(num_views=1), records=[dict(sample_id='sample', view=0, file='sample.npy')]))
    with pytest.raises(ValueError, match='protocol differs'):
        saved.predict_proba(legacy_cache, ['sample'])
    historical = joblib.load(path)
    del historical.token_schema_, historical.token_num_views_
    np.testing.assert_array_equal(historical.predict_proba(legacy_cache, ['sample']),
                                  saved.predict_proba(original, ['sample']))
    assert path.read_bytes() == before


@pytest.mark.parametrize('legacy', [False, True])
def test_saved_cellfm_bundle_replays_its_sampling_protocol(tmp_path, monkeypatch, legacy):
    import api
    from omiclass.foundation import runner
    counts = pd.DataFrame(np.ones((2, 24)), index=['g1', 'g2'], columns=[f's{i}' for i in range(24)])
    expr, meta = tmp_path / 'counts.csv', tmp_path / 'meta.csv'
    counts.to_csv(expr)
    pd.DataFrame({'label': ['a', 'b'] * 12}, index=counts.columns).to_csv(meta)
    split = tmp_path / 'split.json'
    write_json(split, dict(train_index=counts.columns[:16].tolist(), test_index=counts.columns[16:].tolist()))
    source = tmp_path / 'encoder'; source.mkdir(); (source / 'model.py').write_text('# fixture')
    seen = []
    def extract(values, cfg, out, metadata=None, **kwargs):
        out.mkdir(parents=True)
        seen.append(kwargs.get('legacy_cellfm_single_view', False))
        return extraction.Extraction(values.T.astype(float), {})
    monkeypatch.setattr(extraction, 'extract', extract)
    result = api.run(expr, meta_path=meta, classification_column='label', cellfm_root_dir=source,
        output_dir=tmp_path / 'runs', foundation=dict(enabled=True, model='cellfm_1view', classifier='svm', split_path=split))
    manifest_path = result.out_dir / 'foundation.json'
    manifest = json.loads(manifest_path.read_text())
    if legacy:
        manifest.pop('extraction_protocol')
        write_json(manifest_path, manifest)
    before = {str(path): path.read_bytes() for path in result.out_dir.rglob('*') if path.is_file()}
    external = counts.iloc[:, :2].copy(); external.columns = ['external1', 'external2']
    path = tmp_path / 'external.csv'; external.to_csv(path)
    api.predict(path, result.out_dir, tmp_path / 'prediction')
    assert seen[-1] is legacy
    assert before == {str(path): path.read_bytes() for path in result.out_dir.rglob('*') if path.is_file()}


@pytest.mark.parametrize('strategy', [None, 'legacy_global_numpy_random_state',
    'sha256_sample_id_global_seed_zero_based_view_index', 'unknown'])
def test_core_cellfm_single_view_replay_requires_saved_sampling_metadata(tmp_path, monkeypatch, strategy):
    from omiclass.pipeline.steps import bulkformer_step
    saved = tmp_path / 'cellfm/cellfm_metadata.json'
    if strategy is not None:
        write_json(saved, dict(view_seed_strategy=strategy))
    cfg = config(tmp_path, 'cellfm_1view')
    cfg.foundation.enabled, cfg.cellfm.enabled = False, True
    counts = pd.DataFrame([[1., 2.]], index=['g1'], columns=['a', 'b'])
    seen = []
    class Extractor:
        def __init__(self, cfg, **kwargs): seen.append(kwargs.get('legacy_single_view', False))
        def extract(self, values): return SimpleNamespace(features=values, metadata={})
    monkeypatch.setattr(bulkformer_step, 'CellFMFeatureExtractor', Extractor)
    ctx = SimpleNamespace(cfg=cfg, out_dir=tmp_path, opts=SimpleNamespace(mode='clean_and_predict'),
                          cache={'full_expr': counts}, artifacts={})
    step = bulkformer_step.EmbeddingFeatureExtractionStep()
    if strategy is None or strategy == 'unknown':
        with pytest.raises(ValueError, match='sampling'):
            step.run(ctx)
        assert not seen
    else:
        before = saved.read_bytes()
        step.run(ctx)
        assert seen == [strategy == 'legacy_global_numpy_random_state']
        assert saved.read_bytes() == before


def test_isolated_bert_worker_writes_fixed_identity_contract_without_real_jax(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from omiclass.features import bulkrnabert_worker as worker
    genes = np.array([f'g{i}' for i in range(19062)])
    packet, output, cache = tmp_path / 'input.npz', tmp_path / 'output.npz', tmp_path / 'tokens'
    np.savez(packet, tpm=np.ones((1, len(genes)), dtype=np.float32), sample_ids=['sample'], gene_ids=genes)
    cfg = SimpleNamespace(n_genes=19062, embed_dim=256, num_layers=4, use_log_normalization=True,
                          use_max_normalization=True, normalization_factor=6, n_expressions_bins=64)
    keys = []
    def apply(parameters, key, tokens):
        keys.append(key)
        assert tokens.dtype == np.int32 and tokens.shape == (1, 19062)
        return {'embeddings_4': np.broadcast_to(np.arange(256, dtype=np.float32), (1, 19062, 256))}
    for name in ['multiomics_open_research', 'multiomics_open_research.common', 'multiomics_open_research.bulk_rna_bert']:
        package = ModuleType(name); package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, 'multiomics_open_research.bulk_rna_bert.pretrained', SimpleNamespace(
        get_bulkrnabert_pretrained_model=lambda *a, **kw: ({}, object(), Tokenizer(), cfg)))
    monkeypatch.setitem(sys.modules, 'multiomics_open_research.common.preprocess', SimpleNamespace(
        preprocess_omic=lambda values, cfg: np.log10(values.to_numpy()+1)))
    monkeypatch.setitem(sys.modules, 'haiku', SimpleNamespace(__version__='stub',
        transform=lambda fn: SimpleNamespace(apply=apply)))
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace(__version__='stub', numpy=np, jit=lambda fn: fn,
        random=SimpleNamespace(PRNGKey=lambda seed: seed), devices=lambda: ['stub']))
    monkeypatch.setitem(sys.modules, 'jax.numpy', np)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    monkeypatch.setenv('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    monkeypatch.setattr(sys, 'argv', ['worker', '--root-dir', str(tmp_path), '--checkpoint-directory', str(tmp_path),
        '--input', str(packet), '--output', str(output), '--token-dir', str(cache), '--device', 'cpu'])
    worker.main()
    manifest = load_token_manifest(cache)
    schema = manifest['metadata']['token_schema']
    assert schema['identity_mode'] == 'fixed_gene_order' and keys == [0]
    x, mask = load_token_arrays(cache, manifest['records'][0], schema=schema)
    assert x.shape == (19062, 256) and mask.all()
    with np.load(output) as result:
        np.testing.assert_allclose(result['embeddings'][0], np.arange(256))
        assert json.loads(str(result['metadata']))['encoder_seed'] == 0
