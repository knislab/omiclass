"""Input-boundary regression tests; no real encoder/checkpoint or native R."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import api
from omiclass.config.loaders import load_config
from omiclass.errors.errors import DataError
from omiclass.features.cellfm import prepare_cellfm_input
from omiclass.foundation import extraction
from omiclass.foundation.io import write_json
from omiclass.utils.io import load_matrix, load_meta_matrix


def test_cellfm_symbols_and_intentional_duplicate_count_aggregation(tmp_path, monkeypatch):
    from omiclass.features import cellfm
    monkeypatch.setattr(cellfm, '_load_ensembl_symbol_map', lambda: {'ENSG1': 'AC118549.1'})
    path = tmp_path / 'counts.csv'
    path.write_text('gene,a,b\nAC118549.1,2,3\nAC118549.1,4,5\nENSG1.7,1,2\n')
    frame = extraction.read_expression(path, protocol=extraction.CELLFM_INPUT_IDENTIFIERS)
    assert frame.index.tolist() == ['AC118549.1', 'AC118549.1', 'ENSG1.7']
    prepared = prepare_cellfm_input(frame, ['AC118549.1'])
    assert prepared.mapped_genes == ['AC118549.1']
    np.testing.assert_array_equal(prepared.counts, [[7], [10]])


@pytest.mark.parametrize('backend', ['bulkformer', 'bulkrnabert'])
def test_other_encoders_keep_their_historical_identifier_processing(tmp_path, backend):
    cfg = load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label',
                           foundation=dict(enabled=True, model=backend, classifier='svm')))
    path = tmp_path / 'counts.csv'; path.write_text('gene,a\nENSG1.7,5\n')
    protocol = extraction.input_identifier_protocol(cfg)
    assert protocol == extraction.LEGACY_INPUT_IDENTIFIERS
    assert extraction.read_expression(path, protocol=protocol).index.tolist() == ['ENSG1']


@pytest.mark.parametrize('loader', [load_matrix, load_meta_matrix])
@pytest.mark.parametrize('text', ['id,a,a\ng1,1,2\n', 'id,a,b\ng1,1,2\ng1,3,4\n',
    'id,,b\ng1,1,2\n', 'id,a,b\n,1,2\n', 'id,a ,b\ng1,1,2\n'])
@pytest.mark.parametrize('suffix', ['csv', 'tsv'])
def test_core_tables_reject_ambiguous_or_empty_identifiers(tmp_path, loader, text, suffix):
    path = tmp_path / f'input.{suffix}'
    path.write_text(text if suffix == 'csv' else text.replace(',', '\t'))
    with pytest.raises(DataError, match='[Ii]dentifiers'):
        loader(path)


@pytest.mark.parametrize('protocol', [extraction.CELLFM_INPUT_IDENTIFIERS, extraction.LEGACY_INPUT_IDENTIFIERS])
def test_foundation_duplicate_samples_fail_before_pandas_renames_them(tmp_path, protocol):
    path = tmp_path / 'counts.csv'; path.write_text('gene,s,s\ng1,2,3\n')
    with pytest.raises(DataError, match='[Dd]uplicate'):
        extraction.read_expression(path, protocol=protocol)


def test_bulk_identifier_collisions_still_fail(tmp_path):
    path = tmp_path / 'counts.csv'; path.write_text('gene,s\nENSG1.1,2\nENSG1.2,3\n')
    with pytest.raises(DataError, match='duplicate'):
        extraction.read_expression(path)


@pytest.mark.parametrize('legacy', [False, True])
def test_saved_cellfm_input_protocol_is_replayed_without_modifying_bundle(tmp_path, monkeypatch, legacy):
    ids = [f's{i}' for i in range(24)]
    expr, meta, split = [tmp_path / name for name in ['counts.csv', 'meta.csv', 'split.json']]
    pd.DataFrame(np.arange(48).reshape(2, 24)+1, index=['AC118549.1', 'g2'], columns=ids).to_csv(expr)
    pd.DataFrame({'label': ['a', 'b']*12}, index=ids).to_csv(meta)
    write_json(split, dict(train_index=ids[:16], test_index=ids[16:]))
    seen = []
    def extract(frame, cfg, out, metadata=None, **kwargs):
        out.mkdir(parents=True)
        seen.append(frame.index.tolist())
        values = frame.T.copy(); values.columns = ['embedding_0', 'embedding_1']
        return extraction.Extraction(values, {})
    monkeypatch.setattr(extraction, 'extract', extract)
    run = api.run(expr, meta_path=meta, classification_column='label', output_dir=tmp_path / 'runs',
        foundation=dict(enabled=True, model='cellfm_1view', classifier='svm', split_path=split))
    assert seen[-1] == ['AC118549.1', 'g2']
    manifest_path = run.out_dir / 'foundation.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['extraction_protocol']['input_identifiers'] == extraction.CELLFM_INPUT_IDENTIFIERS
    if legacy:
        manifest['extraction_protocol'].pop('input_identifiers')
        write_json(manifest_path, manifest)
    before = {p.relative_to(run.out_dir): p.read_bytes() for p in run.out_dir.rglob('*') if p.is_file()}
    external = pd.DataFrame([[2, 3], [5, 6]], index=['AC118549.1', 'g2'], columns=['external1', 'external2'])
    path = tmp_path / 'external.csv'; external.to_csv(path)
    prediction = api.predict(path, run.out_dir, tmp_path / 'prediction')
    assert seen[-1] == ['AC118549' if legacy else 'AC118549.1', 'g2']
    assert json.loads((prediction.out_dir / 'prediction.json').read_text())['input_identifier_protocol'] == (
        extraction.LEGACY_INPUT_IDENTIFIERS if legacy else extraction.CELLFM_INPUT_IDENTIFIERS)
    assert before == {p.relative_to(run.out_dir): p.read_bytes() for p in run.out_dir.rglob('*') if p.is_file()}


def test_unknown_or_incompatible_input_protocol_is_rejected(tmp_path):
    cfg = load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='label',
                           foundation=dict(enabled=True, model='bulkformer')))
    for protocol in ['unknown', extraction.CELLFM_INPUT_IDENTIFIERS]:
        with pytest.raises(ValueError, match='protocol'):
            extraction.input_identifier_protocol(cfg, {'extraction_protocol': {'input_identifiers': protocol}})


@pytest.mark.parametrize('loader', [load_matrix, load_meta_matrix])
def test_parquet_identifier_validation_cannot_silently_drop_rows(monkeypatch, loader):
    frame = pd.DataFrame({'sample': [1, 2]}, index=['same', 'same'])
    monkeypatch.setattr(pd, 'read_parquet', lambda path: frame)
    with pytest.raises(DataError, match='Duplicate'):
        loader('input.parquet')


def test_new_cellfm_reader_preserves_literal_numeric_identifiers(tmp_path):
    path = tmp_path / 'counts.csv'; path.write_text('gene,s\n001,2\n')
    assert extraction.read_expression(path, protocol=extraction.CELLFM_INPUT_IDENTIFIERS).index.tolist() == ['001']


def test_foundation_metadata_rejects_duplicate_label_columns(tmp_path):
    from omiclass.foundation.runner import _metadata
    path = tmp_path / 'meta.csv'; path.write_text('id,label,label\ns,a,b\n')
    cfg = SimpleNamespace(project=SimpleNamespace(meta_path=path))
    with pytest.raises(ValueError, match='Duplicate'):
        _metadata(cfg)
