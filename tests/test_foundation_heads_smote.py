"""Training-only SMOTE and standalone artifact regression checks."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from omiclass.foundation import article_head as trainer

SCRIPT = Path(trainer.__file__)


def resample(counts, k=5):
    y = np.repeat(np.arange(len(counts)), counts)
    x = np.random.default_rng(2).normal(size=(len(y), 4)).astype('float32')
    return trainer.resample_training(dict(enumerate(x)), dict(enumerate(y)), list(range(len(y))),
                                     seed=80, k_neighbors=k, classes=list(map(str, range(len(counts)))))


def test_multiclass_deterministic_and_adaptive_neighbors():
    x, y, info = resample([6, 3, 2])
    other, labels, repeated = resample([6, 3, 2])
    np.testing.assert_array_equal(x, other)
    np.testing.assert_array_equal(y, labels)
    assert info == repeated
    assert info['effective_k_neighbors'] == 1
    assert info['synthetic_sample_count'] == 7
    assert info['class_counts_after'] == {'0': 6, '1': 6, '2': 6}
    assert resample([8, 7], k=3)[2]['effective_k_neighbors'] == 3


def test_balanced_and_singleton():
    assert resample([2, 2])[2]['synthetic_sample_count'] == 0
    assert resample([1, 1])[2]['effective_k_neighbors'] is None
    with pytest.raises(ValueError, match='at least two'):
        resample([3, 1])
    with pytest.raises(ValueError, match='positive'):
        resample([3, 2], k=0)


@pytest.mark.parametrize('method', ['none', 'smote'])
def test_training_and_heldout_isolation(tmp_path, monkeypatch, method):
    torch.set_num_threads(1)
    ids = [f's{i}' for i in range(10)]
    split = dict(train_index=ids[:6], test_index=ids[6:8], validation_index=ids[8:])
    (tmp_path / 'split.json').write_text(json.dumps(split))
    pd.DataFrame({'sample_id': ids, 'label': ['a']*4 + ['b']*2 + ['a', 'b']*2}).to_csv(tmp_path / 'labels.csv', index=False)
    features = tmp_path / 'mean_features.csv'
    values = np.random.default_rng(80).normal(size=(10, 4)).astype('float32')
    pd.DataFrame(values.T, columns=ids).to_csv(features)
    calls = []
    original = trainer.resample_training
    def spy(vectors, targets, train_ids, **kwargs):
        assert train_ids == ids[:6]
        result = original(vectors, targets, train_ids, **kwargs)
        calls.append(result[0].copy())
        return result
    monkeypatch.setattr(trainer, 'resample_training', spy)
    for attempt in range(2):
        out = tmp_path / f'run{attempt}'
        argv = [str(SCRIPT), '--variant', 'mean', '--features', str(features), '--labels', str(tmp_path / 'labels.csv'),
                '--label-column', 'label', '--split', str(tmp_path / 'split.json'), '--out', str(out),
                '--epochs', '2', '--seed', '80', '--device', 'cpu']
        if method == 'smote':
            argv += ['--resampling', method]
        monkeypatch.setattr(sys, 'argv', argv)
        trainer.train(trainer.parse_args())
        run = json.loads((out / 'run.json').read_text())
        assert run['resampling']['method'] == method
        assert set(pd.read_csv(out / 'metrics.csv').resampling) == {method}
        for partition in ['test', 'validation']:
            assert pd.read_csv(out / f'{partition}_predictions.csv').sample_id.tolist() == split[f'{partition}_index']
            assert (out / f'{partition}_classification_report.json').is_file()
            assert (out / f'{partition}_confusion_matrix.csv').is_file()
        assert (out / 'head.pt').is_file()
        if method == 'smote':
            assert run['resampling']['synthetic_sample_count'] == 2
        # Alter evaluation vectors only, keeping all training vectors identical.
        values[6:] += 100
        pd.DataFrame(values.T, columns=ids).to_csv(features)
    if method == 'smote':
        np.testing.assert_array_equal(calls[0], calls[1])
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'run0/history.csv'), pd.read_csv(tmp_path / 'run1/history.csv'))
    first = torch.load(tmp_path / 'run0/head.pt', weights_only=False)['state_dict']
    second = torch.load(tmp_path / 'run1/head.pt', weights_only=False)['state_dict']
    assert all(torch.equal(first[key], second[key]) for key in first)
    with pytest.raises(FileExistsError):
        trainer.train(trainer.parse_args())


def test_full_rejected_before_inputs_read(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--variant', 'full', '--resampling', 'smote',
        '--labels', 'missing', '--label-column', 'label', '--split', 'missing', '--out', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='not full tokens'):
        trainer.train(trainer.parse_args())
    assert not (tmp_path / 'out').exists()


def test_missing_dependency_is_clear(monkeypatch):
    monkeypatch.setitem(sys.modules, 'imblearn', None)
    with pytest.raises(ImportError, match='training Python environment'):
        resample([3, 2])


@pytest.mark.parametrize('entry', ['parser', 'train'])
def test_removed_variant_rejected_before_inputs_read(tmp_path, monkeypatch, entry):
    from types import SimpleNamespace
    if entry == 'parser':
        monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--variant', 'cls',
            '--labels', 'missing', '--label-column', 'label', '--split', 'missing',
            '--out', str(tmp_path / 'out')])
        with pytest.raises(SystemExit) as exc:
            trainer.parse_args()
        assert exc.value.code == 2
    else:
        with pytest.raises(ValueError, match='Variant must be full or mean'):
            trainer.train(SimpleNamespace(variant='cls'))
    assert not (tmp_path / 'out').exists()
