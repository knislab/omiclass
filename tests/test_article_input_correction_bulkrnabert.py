"""Synthetic checks only: no Limma execution, encoder inference, or experiments."""
import sys
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from experiments.article_input_correction import features


def test_clip_corrected_preserves_positive_values_and_restored_zeros(tmp_path):
    frame = pd.DataFrame([[-1.2, 0.4], [0.0, 0.0], [-1e-12, 5.4]],
                         index=['a', 'restored', 'b'], columns=['train', 'test'])
    original = frame.copy()
    result = features.clip_bulkrnabert_corrected(frame, tmp_path)
    expected = pd.DataFrame([[0.0, 0.4], [0.0, 0.0], [0.0, 5.4]],
                            index=frame.index, columns=frame.columns)
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(frame, original)
    report = json.loads((tmp_path / 'clipping.json').read_text())
    assert report['clipped_entries'] == 2
    assert report['clipped_fraction'] == pytest.approx(2 / 6)
    assert report['minimum_before'] == -1.2
    assert report['minimum_after'] == 0.0


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_clip_refuses_nonfinite_values(tmp_path, value):
    with pytest.raises(features.IncompatibleInput, match='nonfinite'):
        features.clip_bulkrnabert_corrected(pd.DataFrame([[value]]), tmp_path)


def test_restore_exact_order_and_only_missing_rows():
    before = pd.DataFrame([[0, 0], [2, 3], [0, 0]], index=['z', 'a', 'b'], columns=['s1', 's2'])
    corrected = pd.DataFrame([[9.0, 8.0]], index=['a'], columns=['s2', 's1'])
    restored, genes = features.restore_bulkrnabert_zero_genes(before, corrected)
    assert genes == ['z', 'b']
    assert restored.index.tolist() == before.index.tolist()
    assert restored.columns.tolist() == ['s2', 's1']
    pd.testing.assert_frame_equal(restored.loc[['a']], corrected)
    assert (restored.loc[genes].to_numpy() == 0).all()
    pd.testing.assert_frame_equal(features.restore_bulkrnabert_zero_genes(before, restored)[0], restored)


@pytest.mark.parametrize('value', [1.0, 1e-30, np.nan, np.inf])
def test_refuse_arbitrary_missing_values(value):
    before = pd.DataFrame([[value, 0], [1, 2]], index=['missing', 'kept'], columns=['a', 'b'])
    with pytest.raises(features.IncompatibleInput, match='nonzero or nonfinite'):
        features.restore_bulkrnabert_zero_genes(before, before.loc[['kept']])


def test_check_only_retained_samples():
    before = pd.DataFrame([[1, 0], [1, 2]], index=['missing', 'kept'], columns=['dropped', 'retained'])
    restored, genes = features.restore_bulkrnabert_zero_genes(before, before.loc[['kept'], ['retained']])
    assert genes == ['missing'] and restored.loc['missing', 'retained'] == 0


@pytest.mark.parametrize('external', [False, True])
def test_correction_restores_per_call_and_preserves_split(tmp_path, monkeypatch, external):
    # Only stub the interfaces imported by corrected_expression; never load R.
    monkeypatch.setitem(sys.modules, 'omiclass.config.schema', SimpleNamespace(
        Config=lambda **kw: SimpleNamespace(project=SimpleNamespace(meta_path=None))))
    meta = pd.DataFrame({'batch': ['x', 'y', 'z']}, index=['train', 'test', 'external'])
    monkeypatch.setitem(sys.modules, 'omiclass.data.processor', SimpleNamespace(
        DataProcessor=lambda cfg: SimpleNamespace(load_batch_metadata_prediction=lambda **kw: (meta, None, None))))
    expression = pd.DataFrame([[0, 0, 4], [1, 2, 3]], index=['zero', 'kept'], columns=meta.index)
    counts = expression.loc[:, ['train', 'test']]
    spec = dict(model='bulkrnabert', split={'train_index': ['train'], 'test_index': ['test']},
                correction_config={'batch_correction': {'batch_column': 'batch'}})
    if external:
        spec['external'] = {'metadata': 'unused'}
    monkeypatch.setattr(features, 'inputs', lambda *args: (spec, counts, expression, meta))
    monkeypatch.setattr(features, 'diagnostics', lambda *args: None)
    calls = []
    def limma(frame, *args):
        calls.append(frame.copy())
        return frame.loc[['kept']].copy() if len(frame.columns) == 2 else frame.copy()
    monkeypatch.setattr(features, 'limma', limma)
    targets = ['train', 'test'] + (['external'] if external else [])
    result = features.corrected_expression('unused', ['train'], targets, tmp_path)
    assert result.columns.tolist() == targets
    assert result.index.tolist() == ['zero', 'kept']
    assert result.loc['zero', 'train'] == 0
    if external:
        assert result.loc['zero', 'external'] == 4
    provenance = json.loads((tmp_path / 'correction.json').read_text())
    assert provenance['correction_calls'][0]['restored_zero_vocabulary_genes'] == ['zero']
    assert len(calls) == (2 if external else 1)


def test_external_restoration_checks_reference_samples_too():
    # A gene nonzero in a reference sample is unsafe even if zero in external IDs.
    before = pd.DataFrame([[2, 0], [3, 4]], index=['missing', 'kept'], columns=['reference', 'external'])
    with pytest.raises(features.IncompatibleInput):
        features.restore_bulkrnabert_zero_genes(before, before.loc[['kept']])
