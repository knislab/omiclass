import importlib.util
import sys
from pathlib import Path
import pandas as pd

script = Path(__file__).resolve().parents[1] / 'final_results_runs/evaluate_foundation_logistic.py'
spec = importlib.util.spec_from_file_location('logistic_incremental_test', script)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

def test_single_view_discovery(tmp_path):
    import json
    import yaml

    for output, views, timestamp in [('output', 20, '20260909-131102'),
                                     ('output_one_view', 1, '20260907-131102')]:
        for rep in m.REPRESENTATIONS:
            run = tmp_path / 'apolo' / output / rep / ('CellFM-' + timestamp)
            (run / 'cellfm').mkdir(parents=True)
            for name in m.REQUIRED_TOP_LEVEL:
                (run / name).write_text('')
            config = {
                'project': {'meta_path': str(m.APOLLO_BRCA_COMBINED_METADATA),
                            'hold_out_path': str(m.APOLLO_BRCA_HOLDOUT)},
                'cellfm': {'enabled': True, 'num_views': views,
                           'reduction': {'strategy': rep}},
            }
            (run / 'config.freeze.yaml').write_text(yaml.safe_dump(config))
            (run / 'cellfm/cellfm_features.csv').write_text('sample,x\na,1\n')
            (run / 'cellfm/cellfm_metadata.json').write_text(json.dumps({'num_views': views}))
    candidates = m.discover_runs({'cellfm': tmp_path}, tasks={'apolo'},
                                backends={'cellfm'}, cellfm_num_views=1)
    selected = [c for c in candidates if c.selected]
    assert {c.representation for c in selected} == set(m.REPRESENTATIONS)
    assert len(selected) == 4
    assert all('output_one_view' in c.run_dir.parts for c in selected)

def test_filtered_resume_preserves_other_runs(tmp_path, monkeypatch):
    candidate = m.RunCandidate(backend='cellfm', task='apolo', representation='full', run_dir=tmp_path/'new', timestamp='20260907-131102', completed=True, missing=(), selected=True)
    columns = ['backend','task','representation','source_run','result_dir','split','n_samples','accuracy','f1_macro','average_precision_macro','auroc_macro','probability_metric_reason','best_C','best_class_weight']
    old = dict.fromkeys(columns, '')
    old.update(source_run='/previous/run', split='holdout', accuracy=.75)
    pd.DataFrame([old]).to_csv(tmp_path/'summary.csv', index=False)
    new = dict(old, source_run=str(candidate.run_dir), accuracy=.8)
    result_dir = tmp_path/'existing'
    result_dir.mkdir()
    monkeypatch.setattr(m, 'discover_runs', lambda *a, **kw: [candidate])
    monkeypatch.setattr(m, '_result_dir', lambda *a: result_dir)
    monkeypatch.setattr(m, '_existing_summary', lambda *a: pd.DataFrame([new]))
    assert m.main(['--output-root', str(tmp_path), '--resume']) == 0
    result = pd.read_csv(tmp_path/'summary.csv')
    assert set(result.source_run) == {'/previous/run', str(candidate.run_dir)}
    assert len(result) == 2
