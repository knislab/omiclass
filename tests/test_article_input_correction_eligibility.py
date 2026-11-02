import pytest
from experiments.article_input_correction.eligibility import restrict_split


def test_shared_cohort_preserves_partitions_and_order():
    original = dict(train_index=['b', 'a', 'missing_metadata'], test_index=['d', 'missing_tpm'], seed=80)
    for model in ['bulkformer', 'cellfm', 'bulkrnabert']:
        for condition in ['corrected', 'uncorrected']:
            split, report = restrict_split(original, ['a', 'b', 'd', 'missing_metadata', 'missing_tpm', 'unused'],
                                            ['a', 'b', 'd', 'missing_tpm', 'unused'], {'a', 'b', 'd', 'unused'})
            assert split == dict(train_index=['b', 'a'], test_index=['d'], seed=80)
            assert report['sample_ids'] == ['a', 'b', 'd']
            assert report['removed_by_partition'] == {'train_index': ['missing_metadata'], 'test_index': ['missing_tpm']}
    assert original['train_index'] == ['b', 'a', 'missing_metadata']


def test_empty_partition_rejected():
    with pytest.raises(ValueError, match='emptied'):
        restrict_split({'train_index': ['a'], 'test_index': ['b']}, ['a', 'b'], ['a'], ['a', 'b'])


def test_duplicate_partition_ids_rejected():
    with pytest.raises(ValueError, match='disjoint'):
        restrict_split({'train_index': ['a'], 'test_index': ['a']}, ['a'], ['a'], ['a'])


def test_loader_uses_core_metadata_alignment(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import pandas as pd
    from experiments.article_input_correction.eligibility import eligible_brca_samples
    counts = pd.DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd'])
    metadata = pd.DataFrame({'cohort': ['x'] * 4, 'age': [40, None, 50, 60]}, index=counts.columns)
    monkeypatch.setitem(sys.modules, 'omiclass.config.schema', SimpleNamespace(Config=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'omiclass.data.processor', SimpleNamespace(DataProcessor=lambda cfg: SimpleNamespace(
        load_expression_and_genes=lambda p: SimpleNamespace(X=counts),
        load_batch_metadata=lambda **kw: (metadata, None))))
    tpm = tmp_path / 'tpm.csv'
    pd.DataFrame([[0, 0, 0]], columns=['a', 'b', 'c']).to_csv(tpm)
    split, report = eligible_brca_samples('unused', {'batch_correction': {'batch_column': 'cohort', 'design_columns': ['age']}},
        tpm, {'train_index': ['a', 'b'], 'test_index': ['c', 'd']})
    assert split == {'train_index': ['a'], 'test_index': ['c']}
    assert report['alignment_removed_ids'] == ['b']
    assert report['missing_tpm_ids'] == ['d']


@pytest.mark.parametrize('model', ['bulkformer', 'cellfm', 'bulkrnabert'])
@pytest.mark.parametrize('corrected', [False, True])
def test_all_encoder_inputs_use_shared_cohort(tmp_path, monkeypatch, model, corrected):
    import sys
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from experiments.article_input_correction import features
    counts = pd.DataFrame([[1., 2., 3.], [4., 5., 6.]], index=['g1', 'g2'], columns=['keep_test', 'excluded', 'keep_train'])
    vocabulary = tmp_path / 'genes.txt'
    vocabulary.write_text('g1\ng2\n')
    cfg = SimpleNamespace(bulkformer=SimpleNamespace(gene_lengths_path='unused'), cellfm=object(),
                          bulkrnabert=SimpleNamespace(vocabulary_path=vocabulary, tpm_input_path='unused'))
    monkeypatch.setitem(sys.modules, 'omiclass.config.schema', SimpleNamespace(Config=lambda **kw: cfg))
    monkeypatch.setitem(sys.modules, 'omiclass.data.processor', SimpleNamespace(DataProcessor=lambda cfg: SimpleNamespace(
        load_expression_and_genes=lambda p: SimpleNamespace(X=counts.copy()))))
    monkeypatch.setitem(sys.modules, 'omiclass.features.bulkformer', SimpleNamespace(
        load_bulkformer_gene_lengths=lambda p: None, normalize_counts_to_log_tpm=lambda frame, lengths: frame))
    monkeypatch.setitem(sys.modules, 'omiclass.features.bulkrnabert', SimpleNamespace(
        read_expression_csv=lambda p: counts,
        prepare_tpm=lambda frame, genes, ids: (frame.loc[genes, ids].T.to_numpy(dtype=np.float32), {})))
    monkeypatch.setitem(sys.modules, 'experiments.article_input_correction.cellfm_fixed_views', SimpleNamespace(
        continuous_expression=lambda frame, cfg: frame))
    spec = dict(config={}, counts='unused', model=model,
                sample_eligibility={'sample_ids': ['keep_test', 'keep_train']})
    monkeypatch.setattr(features, 'read_json', lambda p: spec)
    features.inputs.cache_clear()
    try:
        _, selected, continuous, _ = features.inputs('synthetic', corrected)
        assert selected.columns.tolist() == continuous.columns.tolist() == ['keep_test', 'keep_train']
    finally:
        features.inputs.cache_clear()
