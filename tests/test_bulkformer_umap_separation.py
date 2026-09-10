import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors


MODULE_PATH = Path(__file__).parents[1] / "final_results_runs" / "plots" / "bulkformer_overview.py"
SPEC = importlib.util.spec_from_file_location("bulkformer_overview", MODULE_PATH)
bulkformer_overview = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bulkformer_overview)


def test_separation_scores_are_macro_averaged_in_pca_space():
    coordinates = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
    )
    labels = pd.Series(["left", "left", "right", "right"])

    actual = bulkformer_overview._separation_scores(coordinates, labels)

    label_values = labels.to_numpy()
    samples = silhouette_samples(coordinates, label_values)
    expected_silhouette = np.mean([samples[:2].mean(), samples[2:].mean()])
    indices = NearestNeighbors(n_neighbors=2).fit(coordinates).kneighbors(
        return_distance=False
    )
    sample_purity = np.equal(
        label_values[:, None], label_values[indices]
    ).mean(axis=1)
    expected_purity = np.mean([
        sample_purity[:2].mean(),
        sample_purity[2:].mean(),
    ])

    assert np.isclose(actual["silhouette"], expected_silhouette)
    assert np.isclose(actual["knn_purity"], expected_purity)

def test_purity_with_singleton_group_uses_nearest_other_sample():
    coordinates = np.array([[0.0], [1.0], [10.0], [11.0]])
    labels = pd.Series(["rare", "common", "common", "common"])
    scores = bulkformer_overview._separation_scores(coordinates, labels)
    # Rare class: 0; common class: (0 + 1 + 1) / 3.
    assert np.isclose(scores["knn_purity"], 1 / 3)
