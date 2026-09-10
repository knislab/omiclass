"""Supplement collector tests use tiny saved artifacts, never ML training."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from final_results_runs.create_sup import (
    Collector, OWNER, SEEDS, complete_group, holm_adjust, notebook_assignment,
    pair_values, validate_destination, foundation_single_run, COMPARISONS, METRICS,
)
from final_results_runs.plots.plot_repeated_seeds import discover_runs


def notebook(root):
    path = root / 'final_results_runs/plots/plot_repeated_seeds.ipynb'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': [
        'TASKS = {}\nraise RuntimeError("must not execute")']}]}))
    return path


def collector(tmp_path):
    notebook(tmp_path)
    return Collector(tmp_path, tmp_path / 'out')


def test_notebook_reads_only_task_assignment(tmp_path):
    path = notebook(tmp_path)
    assert notebook_assignment(path, 'TASKS', tmp_path) == {}
    data = json.loads(path.read_text())
    data['cells'][0]['source'] = ['TASKS = __import__("os").getcwd()']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        notebook_assignment(path, 'TASKS', tmp_path)


def test_holm_undefined_keeps_family_slot():
    np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])
    result = holm_adjust([.01, np.nan])
    assert result[0] == .02 and np.isnan(result[1])


def test_pairing_uses_seed_not_row_order():
    a = pd.DataFrame({'seed': SEEDS, 'accuracy': [.7, .8, .9, .6, .5], 'source_file': '[]'})
    b = a.assign(accuracy=a.accuracy - .05).sample(frac=1, random_state=3)
    pairs = pair_values(a, b, 'accuracy')
    np.testing.assert_allclose(pairs.accuracy_corrected - pairs.accuracy_uncorrected, .05)
    for bad in [b.iloc[:-1], pd.concat([b, b.iloc[:1]]), b.assign(accuracy=np.nan)]:
        with pytest.raises(ValueError):
            pair_values(a, bad, 'accuracy')


def test_complete_classification_groups_require_all_five_unique_seeds():
    assert complete_group(pd.DataFrame({'seed': SEEDS}))
    assert not complete_group(pd.DataFrame({'seed': [80]}))
    assert not complete_group(pd.DataFrame({'seed': [80] * 5}))


def test_overwrite_requires_ownership_and_flag(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    with pytest.raises(ValueError):
        validate_destination(output, False)
    with pytest.raises(ValueError):
        validate_destination(output, True)
    (output / '.collector.json').write_text(json.dumps({'owner': OWNER}))
    validate_destination(output, True)
    link = tmp_path / 'link'
    link.symlink_to(output)
    with pytest.raises(ValueError):
        validate_destination(link, True)


def test_discovery_uses_cv_manifest_not_best_holdout(tmp_path):
    run = tmp_path / 'seed_11' / 'Omiclass-20260901-120000'
    base = run / 'pdf_plots/base_files'
    base.mkdir(parents=True)
    (base / 'manifest.json').write_text(json.dumps({'classification': {'selected_model': 'cv_winner'}}))
    (run / 'config.freeze.yaml').write_text('runtime:\n  random_state: 11\n')
    pd.DataFrame({'model': ['holdout_winner', 'cv_winner'], 'accuracy': [1, .7]}).to_csv(run / 'test_metrics.csv', index=False)
    runs, missing = discover_runs(tmp_path)
    assert not missing and runs[0].selected_model == 'cv_winner'


def test_binary_curves_and_macro_classification_have_distinct_definitions(tmp_path):
    c = collector(tmp_path)
    frame = pd.DataFrame({'sample_id': ['a', 'b', 'c', 'd'], 'y_true': [0, 0, 0, 1],
                          'y_pred': [0, 0, 1, 1], 'proba__0': [.9, .8, .3, .1], 'proba__1': [.1, .2, .7, .9]})
    context = {'dataset': 'CLL', 'analysis': 'internal', 'seed': 11}
    result = c.prediction_outputs(frame, ['0', '1'], {'0': 'mutated', '1': 'unmutated'}, context, [])
    assert result['f1_averaging'] == 'macro'
    assert result['curve_averaging'] == 'binary'
    assert result['positive_class'] == 'unmutated'
    assert 'ovr_accuracy' in {r['metric'] for r in c.classes}
    assert 'accuracy' not in {r['metric'] for r in c.classes}
    assert result['prediction_macro_f1'] != pytest.approx(2 / 3)


def test_csv_provenance_records_source_hash(tmp_path):
    c = collector(tmp_path)
    source = tmp_path / 'source.csv'
    source.write_text('value\n0.123456789012345\n')
    c.copy(source, 'data/classification/copy.csv', {'dataset': 'test', 'analysis': 'test', 'seed': 11})
    entry = c.manifest[0]
    assert json.loads(entry['source_file']) == [str(source)]
    assert len(json.loads(entry['source_sha256'])[str(source)]) == 64
    assert (c.output / entry['supplementary_file']).read_bytes() == source.read_bytes()


def test_discovery_filters_other_seeds_and_reports_missing(tmp_path):
    c = collector(tmp_path)
    source = tmp_path / 'final_results_runs/current'
    for seed in [11, 17]:
        run = source / f'Omiclass_seed{seed}'
        base = run / 'pdf_plots/base_files'
        base.mkdir(parents=True)
        (base / 'manifest.json').write_text(json.dumps({'classification': {'selected_model': 'cv_winner'}}))
        (run / 'config.freeze.yaml').write_text(f'runtime:\n  random_state: {seed}\n')
    spec = {'path': source, 'dataset': 'test', 'output_task': 'test', 'output_condition': None}
    c.tasks['test'] = spec
    selected = c.selected_runs('test', spec)
    assert [r.seed for r in selected] == ['11']
    assert {i['seed'] for i in c.issues if i['severity'] == 'error'} == {61, 73, 80, 89}
    assert any(r['status'] == 'excluded_non_manuscript_seed' for r in c.run_inventory)


def test_batch_json_excludes_figures_and_retains_numerical_precision(tmp_path):
    c = collector(tmp_path)
    run = tmp_path / 'run'
    directory = run / 'limma_correction'
    directory.mkdir(parents=True)
    value = .123456789012345
    source = directory / 'limma.report.json'
    source.write_text(json.dumps({'config': {'design_cols': ['age']},
                                  'eval': {'silhouette_scores': {'cohort': value}},
                                  'figures_corrected': {'heatmap': '<svg>not exported</svg>'}}))
    c.raw_run(run, {'dataset': 'test', 'analysis': 'test', 'seed': 11})
    result = json.loads((c.output / 'data/batch_correction/test/seed_11/limma_correction/limma.report.json').read_text())
    assert 'figures_corrected' not in result
    assert result['eval']['silhouette_scores']['cohort'] == value
    assert c.batch_diagnostics[0]['parameter'] == 'silhouette_scores.cohort'


def test_foundation_benchmark_uses_only_seed_80_without_aggregation():
    rows = pd.DataFrame([
        dict(dataset='CLL', foundation_model=backend, num_views=1, representation='full',
             classifier='Omiclass', seed=seed, f1=.8, ap=np.nan)
        for backend, seed in [('cellfm', 80), ('cellfm', 17), ('none', 80)]
    ])
    result = foundation_single_run(rows)
    assert len(result) == 1 and result.iloc[0].seed == 80
    assert result.iloc[0].foundation_model == 'cellfm'
    assert pd.isna(result.iloc[0].ap)
    assert 'mean' not in result and 'sd' not in result
    assert len(rows) == 3  # Original available results are preserved.
    with pytest.raises(ValueError, match='duplicate'):
        foundation_single_run(pd.concat([result, result]))


@pytest.mark.parametrize('old_reference_exists', [True, False])
def test_old_statistics_are_provenance_not_validation_authority(tmp_path, monkeypatch, old_reference_exists):
    c = collector(tmp_path)
    rows = []
    for a, b in COMPARISONS.values():
        for i, seed in enumerate(SEEDS):
            for task, value in [(a, .8 + .01 * i), (b, .7 + .005 * i)]:
                rows.append(dict(analysis=task, seed=seed, source_file='[]', **{m: value for m in METRICS}))
    if old_reference_exists:
        path = c.base / 'plots/correction_pvalues/csv/correction_pvalues.csv'
        path.parent.mkdir(parents=True)
        # Deliberately stale/empty reference: no matching comparison rows.
        pd.DataFrame(columns=['comparison', 'metric']).to_csv(path, index=False)
    checked = []
    # Synthetic numbers are intentionally unrelated to the user-provided APOLLO
    # regression values. Capture those checks separately from provenance issues.
    monkeypatch.setattr(c, 'check', lambda name, passed, detail='': checked.append((name, passed)))
    c.statistics(pd.DataFrame(rows))
    assert not any(i['severity'] == 'error' for i in c.issues)
    assert all(i['severity'] == 'info' for i in c.issues)
    assert not any('saved statistical reference' in name for name, _ in checked)
    assert ('16 comparisons with five paired seeds', True) in checked
    result = pd.read_csv(c.output / 'tables/Table_S4_corrected_vs_uncorrected_statistics.csv')
    assert len(result) == 16 and result.n_pairs.eq(5).all()
    assert result.p_ttest.notna().all()


def test_collector_reconstructs_s7_and_rejects_stale_sources(tmp_path):
    from final_results_runs.create_sup import digest
    from final_results_runs.plots.figure5_numerics import REPRESENTATIONS
    c = collector(tmp_path)
    directory = c.base / 'plots/figure5_numerical'
    directory.mkdir(parents=True)
    authority = c.base / 'plots/figure5.ipynb'
    authority.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': ['plot_stacked_umaps()']}]}))
    rows, sample_rows = [], []
    for rep in REPRESENTATIONS:
        rows.append(dict(dataset='test', representation=rep, n_samples=4, expected_n_samples=4,
                         k=1, knn_purity=1/3, sample_mean_knn_purity=.5, matches_saved_annotation=True))
        for i, (label, value, weight) in enumerate(zip(['rare', 'common', 'common', 'common'], [0, 0, 1, 1], [.5, 1/6, 1/6, 1/6])):
            sample_rows.append(dict(dataset='test', representation=rep, sample_id=str(i), class_label=label,
                                    sample_knn_purity=value, class_balanced_weight=weight, k=1))
    pd.DataFrame(rows).to_csv(directory / 'knn_purity.csv', index=False)
    pd.DataFrame(sample_rows).to_csv(directory / 'knn_purity_samples.csv', index=False)
    pd.DataFrame({'passed': [True]}).to_csv(directory / 'validation.csv', index=False)
    (directory / 'manifest.json').write_text(json.dumps(dict(
        authority=str(authority), expected_panels=6, source_sha256={str(authority): digest(authority)},
        output_sha256={p.name: digest(p) for p in directory.glob('*.csv')})))
    c.optional_plot_data()
    assert all(check['passed'] for check in c.checks)
    assert not c.issues
    assert len(pd.read_csv(c.output / 'tables/Table_S7_knn_purity.csv')) == 6
    assert len(pd.read_csv(c.output / 'plot_data/Figure5_knn_purity_source.csv')) == 24
    authority.write_text(authority.read_text() + '\n')
    c.optional_plot_data()
    assert any(i['severity'] == 'error' and 'stale' in i['reason'] for i in c.issues)
