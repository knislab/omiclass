"""Synthetic-only tests. No article extraction/training or JAX dependency required."""
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from omiclass.config.loaders import load_config
from omiclass.config.schema import BulkRNABertCfg
from omiclass.cli.runtime import create_runtime_options_from_path_conf_kwargs
from omiclass.errors.errors import DataError, SchemaValidationError
from omiclass.features import bulkrnabert as bert
from omiclass.features.embedding_backend import embedding_backend, embedding_config
from omiclass.features.bulkformer_reduction import EmbeddingRepresentation, build_embedding_representation
from omiclass.pipeline.steps.bulkformer_step import EmbeddingFeatureExtractionStep
from omiclass.train.trainer import Trainer
from final_results_runs.test_bulkrnabert import prepare as prep


def base(tmp_path):
    return {'project_name': 'synthetic', 'output_dir': str(tmp_path / 'out'),
            'meta_path': str(tmp_path / 'meta.csv'), 'class_col': 'class'}


def config(tmp_path, **kwargs):
    return load_config({**base(tmp_path), 'bulkrnabert_enabled': True,
                        'bulkrnabert_tpm_input_path': str(tmp_path / 'tpm.csv'),
                        'feature_selection_enabled': False, **kwargs})


def vocab():
    return [f'ENSG{i:011d}' for i in range(bert.N_GENES)]


@pytest.fixture
def resources(tmp_path):
    root = tmp_path / 'official'
    checkpoint = root / 'checkpoints/bulk_rna_bert_gtex_encode'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text('{}')
    (checkpoint / 'params.joblib').write_bytes(b'synthetic weights')
    impl = root / 'multiomics_open_research'
    impl.mkdir()
    (impl / '__init__.py').write_text('# fixture')
    vocabulary = tmp_path / 'vocabulary.txt'
    vocabulary.write_text('\n'.join(vocab()) + '\n')
    tpm = pd.DataFrame({'b': np.full(bert.N_GENES, 2.0), 'a': np.ones(bert.N_GENES)}, index=vocab())
    path = tmp_path / 'tpm.csv'
    tpm.to_csv(path)
    cfg = BulkRNABertCfg(root_dir=root, vocabulary_path=vocabulary,
                         tpm_input_path=path, cache_dir=tmp_path / 'cache',
                         python_executable=tmp_path / 'isolated/python')
    return cfg, tpm


@pytest.mark.parametrize('backend', [None, 'bulkformer', 'cellfm'])
def test_existing_config_and_dispatch_remain_unchanged(tmp_path, backend):
    args = base(tmp_path)
    if backend:
        args[backend + '_enabled'] = True
    cfg = load_config(args)
    assert embedding_backend(cfg) == backend
    assert not cfg.bulkrnabert.enabled
    assert 'bulkrnabert' not in cfg.model_dump(mode='json')
    assert 'bulkrnabert' not in json.loads(cfg.model_dump_json())
    opts = create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / 'counts.csv', args)
    if backend:
        assert opts.mode == 'train-only' and not opts.run_de and not opts.run_batch


def test_config_aliases_and_shared_route(tmp_path):
    cfg = config(tmp_path, bulkrnabert_reduction_strategy='supervised_pca',
                 bulkrnabert_pca_components=3, bulkrnabert_model_name='other_checkpoint',
                 bulkrnabert_checkpoint_directory=str(tmp_path / 'weights'))
    assert embedding_backend(cfg) == 'bulkrnabert'
    assert embedding_config(cfg) is cfg.bulkrnabert
    assert cfg.bulkrnabert.model_name == 'other_checkpoint'
    assert cfg.bulkrnabert.checkpoint_directory == tmp_path / 'weights'
    assert cfg.bulkrnabert.reduction.pca_components == 3
    assert 'bulkrnabert' in cfg.model_dump(mode='json')
    opts = create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / 'counts.csv', {})
    assert opts.mode == 'train-only' and not opts.run_de and not opts.run_batch
    assert not opts.surface_markers
    assert isinstance(build_embedding_representation(cfg.bulkrnabert, 80, 'bulkrnabert'), EmbeddingRepresentation)


@pytest.mark.parametrize('overrides', [
    {'bulkformer_enabled': True}, {'cellfm_enabled': True},
    {'modality': 'cptac_ms'}, {'bulkrnabert_batch_size': 0},
    {'bulkrnabert_tpm_input_path': None},
    {'bulkrnabert_reduction_strategy': 'pca', 'feature_selection_enabled': True},
])
def test_invalid_backend_configuration(tmp_path, overrides):
    with pytest.raises(SchemaValidationError):
        config(tmp_path, **overrides)


def test_gene_order_zero_fill_and_sample_alignment():
    genes = vocab()
    frame = pd.DataFrame({'b': np.arange(18000) + 1., 'a': np.arange(18000) + 2.},
                         index=[g + '.7' for g in genes[:18000]])
    values, metadata = bert.prepare_tpm(frame.iloc[::-1], genes, ['a', 'b'])
    assert values.shape == (2, 19062)
    np.testing.assert_array_equal(values[0, :18000], frame.a)
    assert (values[:, 18000:] == 0).all()
    assert metadata['sample_ids'] == ['a', 'b'] and metadata['matched_genes'] == 18000
    with pytest.raises(DataError, match='coverage'):
        bert.prepare_tpm(frame.iloc[:17999], genes, ['a'])
    with pytest.raises(DataError, match='missing samples'):
        bert.prepare_tpm(frame, genes, ['missing'])
    duplicate = pd.concat([frame, frame.iloc[:1].rename(index={frame.index[0]: genes[0] + '.8'})])
    with pytest.raises(DataError, match='duplicate'):
        bert.prepare_tpm(duplicate, genes, ['a'])


@pytest.mark.parametrize('value', [-1, np.nan, np.inf])
def test_reject_invalid_expression_before_sanitization(tmp_path, value):
    path = tmp_path / 'invalid.csv'
    pd.DataFrame({'sample': [value, 1]}, index=['g1', 'g2']).to_csv(path)
    with pytest.raises(DataError, match='finite'):
        bert.read_expression_csv(path)


def test_duplicate_csv_headers_and_empty_sample_rejected(tmp_path):
    path = tmp_path / 'duplicate.csv'
    path.write_text(',a,a\ng,1,2\n')
    with pytest.raises(DataError, match='duplicate'):
        bert.read_expression_csv(path)
    with pytest.raises(DataError, match='empty-expression'):
        bert.validate_expression(pd.DataFrame({'a': [0.]}, index=['g']))


def fake_subprocess(calls):
    def run(command, env, check):
        calls.append((command, env))
        with np.load(command[command.index('--input') + 1], allow_pickle=False) as src:
            ids = src['sample_ids'].copy()
            values = np.repeat(src['tpm'][:, :1], 256, axis=1)
        np.savez(command[command.index('--output') + 1], embeddings=values,
                 sample_ids=ids, metadata=json.dumps({'embeddings_4_shape': [len(ids), 19062, 256]}))
    return run


def test_cache_worker_isolation_and_ids(resources, monkeypatch):
    cfg, tpm = resources
    calls = []
    monkeypatch.setattr(bert.subprocess, 'run', fake_subprocess(calls))
    monkeypatch.setenv('PYTHONPATH', '/must/not/leak')
    extractor = bert.BulkRNABertFeatureExtractor(cfg)
    # Deliberately poisonous processed expression: adapter uses only these IDs.
    counts = pd.DataFrame([[np.nan, -500]], columns=['a', 'b'])
    first = extractor.extract(counts)
    assert first.features.shape == (256, 2)
    assert first.features.columns.tolist() == ['a', 'b']
    np.testing.assert_array_equal(first.features.iloc[0], [1, 2])
    assert not first.metadata['cache_hit']
    cfg.reduction.strategy = 'pca'
    cfg.reduction.random_state = 999
    again = extractor.extract(counts)
    assert again.metadata['cache_hit'] and len(calls) == 1
    pd.testing.assert_frame_equal(first.features, again.features)
    assert 'PYTHONPATH' not in calls[0][1]
    assert calls[0][1]['PYTHONNOUSERSITE'] == '1'
    assert calls[0][0][0] == str(cfg.python_executable.resolve())
    reverse = extractor.extract(counts[['b', 'a']])
    np.testing.assert_array_equal(reverse.features.iloc[0], [2, 1])
    assert len(calls) == 2
    tpm.loc[tpm.index[0], 'a'] = 3
    tpm.to_csv(cfg.tpm_input_path)
    changed = extractor.extract(counts)
    assert len(calls) == 3 and changed.features.iloc[0, 0] == 3
    weights = cfg.checkpoint_directory / cfg.model_name / 'params.joblib'
    weights.write_bytes(b'changed checkpoint')
    extractor.extract(counts)
    assert len(calls) == 4


def test_corrupt_cache_and_worker_failure(resources, monkeypatch):
    cfg, tpm = resources
    monkeypatch.setattr(bert.subprocess, 'run', fake_subprocess([]))
    extractor = bert.BulkRNABertFeatureExtractor(cfg)
    result = extractor.extract(tpm)
    Path(result.metadata['cache_path']).write_bytes(b'corrupt')
    with pytest.raises(DataError, match='Invalid BulkRNABert cache'):
        extractor.extract(tpm)
    Path(result.metadata['cache_path']).unlink()
    def fail(*args, **kwargs):
        raise OSError('missing Python')
    monkeypatch.setattr(bert.subprocess, 'run', fail)
    with pytest.raises(DataError, match='worker failed'):
        extractor.extract(tpm)
    assert not list(cfg.cache_dir.glob('*.npz'))


@pytest.mark.parametrize('values', [np.zeros((2, 255)), np.full((2, 256), np.inf)])
def test_invalid_embedding_shape_and_values(values):
    with pytest.raises(DataError, match='Invalid BulkRNABert embeddings'):
        bert.feature_matrix(values, ['a', 'b'])


def test_step_delivers_shared_feature_contract(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    features = bert.feature_matrix(np.ones((2, 256)), ['a', 'b'])
    monkeypatch.setattr(bert.BulkRNABertFeatureExtractor, 'extract',
                        lambda self, counts: bert.BulkRNABertExtractionResult(features, {'backend': 'bulkrnabert'}))
    ctx = SimpleNamespace(cfg=cfg, cache={'full_expr': pd.DataFrame([[3, 4]], columns=['a', 'b'])},
                          opts=SimpleNamespace(mode='train-only'), out_dir=tmp_path, artifacts={})
    step = EmbeddingFeatureExtractionStep()
    assert step.should_run(ctx)
    step.run(ctx)
    pd.testing.assert_frame_equal(ctx.cache['full_expr_t'], features.T)
    assert ctx.cache['feature_source'] == 'bulkrnabert'
    assert ctx.cache['gene_list'] == list(features.index)
    assert ctx.artifacts['bulkrnabert_features'].exists()


@pytest.mark.parametrize('strategy', ['full', 'pca', 'supervised', 'supervised_pca'])
def test_shared_training_reductions_never_fit_holdout_or_external(tmp_path, monkeypatch, strategy):
    cfg = config(tmp_path, bulkrnabert_reduction_strategy=strategy,
                 bulkrnabert_supervised_dimensions=8, bulkrnabert_pca_components=3,
                 bulkrnabert_close_pair_pca_components=2)
    trainer = Trainer(cfg, tmp_path / 'artifacts')
    builder = trainer._with_bulkformer_reduction(lambda: RandomForestClassifier(n_estimators=2, random_state=80))
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.normal(size=(24, 256)), index=[f'TCGA_train_{i}' for i in range(24)],
                     columns=[f'bulkrnabert_feature_{i:04d}' for i in range(256)])
    y = pd.Series([0, 1] * 12, index=X.index)
    external = pd.DataFrame(rng.normal(size=(4, 256)) + 10000,
                            index=['APOLLO_1', 'APOLLO_2', 'holdout_1', 'holdout_2'], columns=X.columns)
    seen = []
    original = EmbeddingRepresentation.fit
    def recording_fit(self, X, y=None):
        seen.append(set(X.index))
        return original(self, X, y)
    monkeypatch.setattr(EmbeddingRepresentation, 'fit', recording_fit)
    folds = list(StratifiedKFold(2).split(X, y))
    search = GridSearchCV(builder(), {'classifier__max_depth': [2]}, cv=folds, n_jobs=1)
    search.fit(X, y)
    assert seen == [set(X.index[train]) for train, _ in folds] + [set(X.index)]
    representation = search.best_estimator_.named_steps['bulkformer_representation']
    before = pickle.dumps(representation)
    search.predict(external)
    assert pickle.dumps(representation) == before
    assert all(not set(external.index) & ids for ids in seen)


def test_counts_to_tpm_denominator_and_missing_lengths():
    counts = pd.DataFrame({'s': [10, 20, 30]}, index=['g1', 'g2', 'outside_vocabulary'])
    lengths = pd.Series([1000, 2000, 1000], index=counts.index)
    tpm = prep.counts_to_tpm(counts, lengths)
    np.testing.assert_allclose(tpm.s, [200000, 200000, 600000])
    with pytest.raises(ValueError, match='every source gene'):
        prep.counts_to_tpm(counts, lengths.iloc[:2])


def test_gdc_aliquot_selected_by_original_counts(tmp_path):
    paths = []
    for i in [1, 2]:
        p = tmp_path / f'{i}.tsv'
        p.write_text('gene_id\tunstranded\ttpm_unstranded\nN_unmapped\t999\t\n'
                     f'ENSG00000000001.2\t{i}\t1000000\n')
        paths.append(p)
    reference = pd.Series([2.], index=['ENSG00000000001'])
    path, column = prep.select_matching_gdc_sample('TCGA_s', reference, paths)
    assert path == paths[1] and column.name == 'TCGA_s'
    assert column.iloc[0] == 1000000
    with pytest.raises(ValueError, match='found 2'):
        prep.select_matching_gdc_sample('TCGA_s', reference, [paths[1], paths[1]])


def test_apollo_partition_has_only_external_ids(tmp_path):
    meta_path = tmp_path / 'data/apolo_brca1/processed/combined_metadata.csv'
    holdout = tmp_path / 'final_results_runs/apolo_brca1/apollo_test_samples.txt'
    meta_path.parent.mkdir(parents=True)
    holdout.parent.mkdir(parents=True)
    ids = ['TCGA_train'] + [f'APOLLO_{i}' for i in range(68)]
    pd.DataFrame({'sample': ids, 'cohort': ['TCGA'] + ['APOLLO'] * 68}).to_csv(meta_path, index=False)
    holdout.write_text('sample\n' + '\n'.join(ids[1:]) + '\n')
    prep.check_external_partition(tmp_path, ids)
    holdout.write_text('sample\n' + '\n'.join(ids[:-1]) + '\n')
    with pytest.raises(ValueError, match='68 APOLLO'):
        prep.check_external_partition(tmp_path, ids)


def test_cll_legacy_aliases_resolve_distinct_native_samples(tmp_path):
    import gzip
    reference_path = tmp_path / 'reference.csv'
    reference_path.write_text(',patient,patient\nENSG00000000001,10,20\nENSG00000000002,20,10\n')
    reference, aliases = prep.read_cll_reference(reference_path)
    assert reference.columns.tolist() == ['patient', 'patient.1']
    assert aliases == {'patient': 'patient', 'patient.1': 'patient'}
    mapping = tmp_path / 'metadata/all_deidentifier_patAndSamp_20220308.tsv'
    mapping.parent.mkdir()
    mapping.write_text('original_id\tdeidentified_id\nnative_a\tpatient\nnative_b\tpatient\n')
    folder = tmp_path / 'rnaseq/all_rnaseqc_20200506'
    folder.mkdir(parents=True)
    for suffix, values in [('counts', ['10\t20', '20\t10']), ('tpm', ['100\t200', '300\t400'])]:
        path = folder / f'all_rnaseqc_20200506.rnaseqc_{suffix}.gct.gz'
        with gzip.open(path, 'wt') as handle:
            handle.write('#1.2\n2\t2\nName\tDescription\tnative_a\tnative_b\n'
                         f'ENSG00000000001.3\tgene1\t{values[0]}\n'
                         f'ENSG00000000002.1\tgene2\t{values[1]}\n')
    tpm, paths, selected = prep.load_cll_tpm(tmp_path, reference, aliases)
    assert selected == {'patient': 'native_a', 'patient.1': 'native_b'}
    assert tpm.columns.tolist() == reference.columns.tolist()
    np.testing.assert_array_equal(tpm.to_numpy(), [[100, 200], [300, 400]])
    reference.loc['ENSG00000000001', 'patient.1'] = 999
    with pytest.raises(ValueError, match='count-vector match'):
        prep.load_cll_tpm(tmp_path, reference, aliases)


def test_preparation_replacement_provenance_cache_and_staleness(tmp_path):
    counts = tmp_path / prep.TASK_COUNTS['brca']
    counts.parent.mkdir(parents=True)
    tpm = pd.DataFrame({'s1': np.ones(19062), 's2': np.full(19062, 2.)}, index=vocab())
    tpm.to_csv(counts)
    replacement = tmp_path / 'replacement.csv'
    tpm[['s2', 's1']].to_csv(replacement)
    vocabulary = tmp_path / 'BulkRNABert/data/bulkrnabert/common_gene_id.txt'
    vocabulary.parent.mkdir(parents=True)
    vocabulary.write_text('\n'.join(vocab()) + '\n')
    adapter = tmp_path / 'omiclass/features/bulkrnabert.py'
    adapter.parent.mkdir(parents=True)
    adapter.write_text('# synthetic adapter fingerprint')
    output = tmp_path / 'prepared/tpm.csv'
    prep.prepare('brca', output, root=tmp_path, replacement=replacement)
    loaded = bert.read_expression_csv(output)
    pd.testing.assert_frame_equal(loaded, tpm)
    metadata = json.loads(output.with_suffix('.csv.json').read_text())
    assert metadata['method'] == 'user_supplied_unlogged_TPM'
    assert metadata['sample_ids'] == ['s1', 's2']
    before = output.stat().st_mtime_ns
    prep.prepare('brca', output, root=tmp_path, replacement=replacement)
    assert output.stat().st_mtime_ns == before
    tpm.loc[tpm.index[0], 's1'] = 99
    tpm.to_csv(replacement)
    with pytest.raises(ValueError, match='Stale/incomplete'):
        prep.prepare('brca', output, root=tmp_path, replacement=replacement)
