"""Regression checks for Figure 5's original PCA-space, class-balanced metric."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from final_results_runs.plots import bulkformer_overview as plots
from final_results_runs.plots import viz_data
from final_results_runs.plots.figure5_numerics import knn_purity_details, REPRESENTATIONS


def legacy_purity(matrix, labels):
    # Frozen pre-refactor definition, for regression only (not an exporter).
    values = labels.to_numpy()
    k = min(15, int(pd.Series(values).value_counts().min()), len(values) - 1)
    indices = NearestNeighbors(n_neighbors=k).fit(matrix).kneighbors(return_distance=False)
    sample_values = np.equal(values[:, None], values[indices]).mean(axis=1)
    return np.mean([sample_values[values == label].mean() for label in np.unique(values)])


def inputs():
    ids = [f's{i}' for i in range(20)]
    x = pd.DataFrame(np.random.RandomState(2).normal(size=(20, 8)) * np.arange(1, 9), index=ids)
    meta = pd.DataFrame({'class': ['rare'] * 3 + ['common'] * 17}, index=ids)
    return x, meta


def test_original_adaptive_k_and_class_balanced_mean():
    matrix = np.array([[0.], [1.], [10.], [11.]])
    labels = pd.Series(['rare', 'common', 'common', 'common'], index=['a', 'b', 'c', 'd'])
    detail = knn_purity_details(matrix, labels)
    assert detail['k'] == 1  # Do not change to fixed k=15.
    assert detail['knn_purity'] == legacy_purity(matrix, labels) == 1 / 3
    assert detail['sample_mean_knn_purity'] == .5
    values = detail['samples']
    np.testing.assert_allclose(values.sample_knn_purity, [0, 0, 1, 1])
    assert np.isclose(np.sum(values.sample_knn_purity * values.class_balanced_weight), detail['knn_purity'])


def test_expression_export_and_plot_use_identical_scaled_pca(monkeypatch):
    x, meta = inputs()
    matrices = []

    class UmapStub:
        def __init__(self, **kwargs):
            pass
        def fit_transform(self, matrix):
            matrices.append(matrix.copy())
            return np.zeros((len(matrix), 2))  # Purity must not use these coordinates.

    monkeypatch.setattr(viz_data.umap, 'UMAP', UmapStub)
    fig, ax = plt.subplots()
    plotted = viz_data.plot_umap_by_meta(x, meta, 'class', ax=ax)[0]
    numerical_matrices = []
    original = viz_data.numerical_panel
    def capture(matrix, *args, **kwargs):
        numerical_matrices.append(matrix.copy())
        return original(matrix, *args, **kwargs)
    monkeypatch.setattr(viz_data, 'numerical_panel', capture)
    numerical = viz_data.plot_umap_by_meta(x, meta, 'class', numerical_only=True)
    assert len(matrices) == 1  # The export never calls UMAP.
    expected = PCA(n_components=8, random_state=42).fit_transform(StandardScaler().fit_transform(x.values))
    np.testing.assert_array_equal(matrices[0], numerical_matrices[0])
    np.testing.assert_allclose(matrices[0], expected, rtol=0, atol=2e-13)
    assert numerical['knn_purity'] == plotted.attrs['separation_scores']['knn_purity'] == legacy_purity(expected, meta['class'])
    assert numerical['standardized'] is True
    plt.close(fig)


def make_embeddings(tmp_path, backend, views, x):
    run = tmp_path / f'{backend}_{views}'
    folder = run / backend
    folder.mkdir(parents=True)
    x.T.to_csv(folder / f'{backend}_features.csv')
    (folder / f'{backend}_metadata.json').write_text(json.dumps({'num_views': views, 'view_aggregation': 'mean'}))
    return run


def test_embedding_export_and_plot_use_identical_unscaled_float32_pca(tmp_path, monkeypatch):
    x, meta = inputs()
    run = make_embeddings(tmp_path, 'cellfm', 20, x)
    matrices = []

    class UmapStub:
        def __init__(self, **kwargs):
            pass
        def fit_transform(self, matrix):
            matrices.append(matrix.copy())
            return np.zeros((len(matrix), 2))

    monkeypatch.setattr(plots.umap, 'UMAP', UmapStub)
    fig, plotted = plots.plot_embedding_umap(run, meta, 'class', return_data=True)
    numerical_matrices = []
    original = plots.numerical_panel
    def capture(matrix, *args, **kwargs):
        numerical_matrices.append(matrix.copy())
        return original(matrix, *args, **kwargs)
    monkeypatch.setattr(plots, 'numerical_panel', capture)
    numerical = plots.plot_embedding_umap(run, meta, 'class', numerical_only=True)
    assert len(matrices) == 1
    loaded = pd.read_csv(run / 'cellfm/cellfm_features.csv', index_col=0).T
    expected = PCA(n_components=8, random_state=42).fit_transform(loaded.to_numpy(dtype=np.float32, copy=True))
    np.testing.assert_array_equal(matrices[0], numerical_matrices[0])
    np.testing.assert_allclose(matrices[0], expected, rtol=0, atol=3e-5)
    assert numerical['knn_purity'] == plotted.attrs['separation_scores']['knn_purity'] == legacy_purity(expected, meta['class'])
    assert numerical['standardized'] is False and numerical['num_views'] == 20
    plt.close(fig)


def test_stacked_numerical_mode_creates_no_umap_or_figure(tmp_path, monkeypatch):
    x, meta = inputs()
    paths = [make_embeddings(tmp_path, backend, views, x) for backend, views in [('cellfm', 1), ('cellfm', 20), ('bulkformer', 0)]]
    def forbidden(*args, **kwargs):
        raise AssertionError('Numerical export attempted UMAP or plotting')
    monkeypatch.setattr(plots.umap, 'UMAP', forbidden)
    monkeypatch.setattr(plt, 'subplots', forbidden)
    panels = plots.plot_stacked_umaps(x, meta, 'class', x.columns[:4].tolist(),
                                     cellfm_single_view_results_dir=paths[0], cellfm_results_dir=paths[1],
                                     bulkformer_results_dir=paths[2], numerical_only=True)
    assert tuple(p['representation'] for p in panels) == REPRESENTATIONS
    assert all(p['n_samples'] == 20 and p['k'] == 3 for p in panels)
    assert [p['standardized'] for p in panels] == [True, True, True, False, False, False]
