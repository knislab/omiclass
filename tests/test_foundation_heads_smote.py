"""Training-only SMOTE and standalone artifact regression checks."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/article_foundation_heads/train.py'
spec = importlib.util.spec_from_file_location('foundation_heads_smote', SCRIPT)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


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


@pytest.mark.parametrize('variant', ['mean', 'cls'])
@pytest.mark.parametrize('method', ['none', 'smote'])
def test_training_and_heldout_isolation(tmp_path, monkeypatch, variant, method):
    torch.set_num_threads(1)
    ids = [f's{i}' for i in range(10)]
    split = dict(train_index=ids[:6], test_index=ids[6:8], validation_index=ids[8:])
    (tmp_path / 'split.json').write_text(json.dumps(split))
    pd.DataFrame({'sample_id': ids, 'label': ['a']*4 + ['b']*2 + ['a', 'b']*2}).to_csv(tmp_path / 'labels.csv', index=False)
    cache = tmp_path / 'cache'
    cache.mkdir()
    values = np.random.default_rng(80).normal(size=(10, 4)).astype('float32')
    records = []
    for i, sid in enumerate(ids):
        for view in range(2):
            file = f'{sid}_{view}.npz'
            np.savez(cache / file, cls=values[i] + view)
            records.append(dict(sample_id=sid, view=view, file=file))
    (cache / 'manifest.json').write_text(json.dumps(dict(format_version=1, backend='cellfm', metadata={'num_views': 2}, records=records)))
    pd.DataFrame(values.T, columns=ids).to_csv(cache / 'mean_features.csv')
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
        argv = [str(SCRIPT), '--variant', variant, '--cache', str(cache), '--labels', str(tmp_path / 'labels.csv'),
                '--label-column', 'label', '--split', str(tmp_path / 'split.json'), '--out', str(out),
                '--epochs', '2', '--seed', '80', '--device', 'cpu']
        if method == 'smote':
            argv += ['--resampling', method]
        monkeypatch.setattr(sys, 'argv', argv)
        trainer.main()
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
        pd.DataFrame(values.T, columns=ids).to_csv(cache / 'mean_features.csv')
        for record in records:
            i = ids.index(record['sample_id'])
            if i >= 6:
                np.savez(cache / record['file'], cls=values[i] + record['view'])
    if method == 'smote':
        np.testing.assert_array_equal(calls[0], calls[1])
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'run0/history.csv'), pd.read_csv(tmp_path / 'run1/history.csv'))
    first = torch.load(tmp_path / 'run0/head.pt', weights_only=False)['state_dict']
    second = torch.load(tmp_path / 'run1/head.pt', weights_only=False)['state_dict']
    assert all(torch.equal(first[key], second[key]) for key in first)
    with pytest.raises(FileExistsError):
        trainer.main()


def test_full_rejected_before_inputs_read(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--variant', 'full', '--resampling', 'smote',
        '--labels', 'missing', '--label-column', 'label', '--split', 'missing', '--out', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='not full tokens'):
        trainer.main()
    assert not (tmp_path / 'out').exists()


def test_missing_dependency_is_clear(monkeypatch):
    monkeypatch.setitem(sys.modules, 'imblearn', None)
    with pytest.raises(ImportError, match='training Python environment'):
        resample([3, 2])


def test_launcher_smote_destinations(tmp_path, monkeypatch, capsys):
    launcher_path = SCRIPT.parents[1] / 'article_four_tasks/run.py'
    spec = importlib.util.spec_from_file_location('smote_launcher', launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    tasks = {}
    out = tmp_path / 'outputs'
    for task in ['BALL', 'CLLmap', 'BRCA-TCGA', 'BRCA-TCGA-APOLLO']:
        tasks[task] = dict(counts=str(tmp_path / 'counts.csv'), labels=str(tmp_path / 'labels.csv'), sources={})
        for backend in ['bulkformer', 'cellfm']:
            base = out / task / backend
            source = tmp_path / 'sources' / task / backend
            (source / backend).mkdir(parents=True)
            (source / 'split.json').write_text(json.dumps(dict(train_index=['a', 'b'], test_index=['c'])))
            for name in ['config.freeze.yaml', f'{backend}/{backend}_features.csv', f'{backend}/{backend}_metadata.json']:
                (source / name).touch()
            tasks[task]['sources'][backend] = str(source)
            (base / 'prepared').mkdir(parents=True)
            for name in ['backend.json', 'split.json', 'labels.csv', 'provenance.json']:
                (base / 'prepared' / name).touch()
            (base / 'cache').mkdir()
            (base / 'cache/mean_features.csv').touch()
            (base / 'cache/manifest.json').write_text(json.dumps({'records': []}))
            (base / 'mean_mlp_seed80').mkdir()  # Baseline collisions must be harmless.
    for name in ['counts.csv', 'labels.csv']:
        (tmp_path / name).touch()
    task_file = tmp_path / 'tasks.json'
    task_file.write_text(json.dumps(tasks))
    argv = [str(launcher_path), '--tasks-file', str(task_file), '--out', str(out),
            '--stage', 'train', '--resampling', 'smote', '--dry-run']
    monkeypatch.setattr(sys, 'argv', argv)
    launcher.main()
    output = capsys.readouterr().out
    assert '12 sequential commands' in output
    assert output.count('mean_smote_mlp_seed80') == 8
    assert output.count('cls_smote_mlp_seed80') == 4
    assert '--variant full' not in output
    monkeypatch.setattr(sys, 'argv', argv + ['--backends', 'cellfm'])
    launcher.main()
    assert '8 sequential commands' in capsys.readouterr().out
    assert not list(out.glob('*/*/*smote*'))
    (out / 'BALL/cellfm/mean_smote_mlp_seed80').mkdir()
    with pytest.raises(FileExistsError):
        launcher.main()
